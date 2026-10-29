"""Copy WRF v4.6.1 module_ra_rrtmg_sw.F and sw_fixture_driver.F90 with the
spectral (per-band and per-g-point) flux dump added -- the oracle of
rrtmg_spectral_output (tests/test_rrtmg_spectral_oracle.py).

module_ra_rrtmg_sw.F, spcvmc_sw: WRF carries the per-band accumulation as
inactive code ("Accumulate spectral fluxes over bands - inactive").  The
patch makes it live, indexed by band, beside the whole-spectrum sums and
in the same statement form:

    oracle_band(ikl,ibm,q) = oracle_band(ikl,ibm,q) + zincflx(iw)*flux
    oracle_gpt(ikl,iw,q)   = zincflx(iw)*flux

for q = fu, fd, cu, cd, fddir, cddir (the order of
woof.core.rrtmg_sw.SPECTRAL_QUANTITIES).  Nothing else in WRF changes.
The driver dumps both arrays after its one-shot rrtmg_sw call.

module_ra_rrtmg_lw.F, rtrnmc: each band's term of the total flux,
``uflux(lev)*delwave(iband)``, times the fluxfac the total is scaled by:

    oracle_lwband(lev,iband,q) = uflux(lev)*delwave(iband)*fluxfac

for q = u, d, uc, dc (woof.core.rrtmg_lw.LW_SPECTRAL_QUANTITIES).
lw_extract.F90 dumps it as out/lw_band after its rrtmg_lw call.

    python spectral_patch.py WRF_SOURCE_ROOT BUILD_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

QUANTITIES = (("zfu(jk,iw)", 1), ("zfd(jk,iw)", 2), ("zcu(jk,iw)", 3), ("zcd(jk,iw)", 4),
              ("ztdbt_nodel(jk)", 5), ("ztdbtc_nodel(jk)", 6))


def _once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"spectral_patch: expected one occurrence of {old!r}, found {text.count(old)}")
    return text.replace(old, new)


def patch_module(text: str) -> str:
    start = text.index("      module rrtmg_sw_spcvmc")
    contains = text.index("      contains", start)
    text = text[:contains] + ("      real(kind=rb), allocatable, public :: oracle_band(:,:,:), "
                              "oracle_gpt(:,:,:)\n\n") + text[contains:]
    sub = text.index("      subroutine spcvmc_sw")
    zero = text.index("      do jk=1,klev+1\n         pbbcd(jk)=0._rb", sub)
    text = text[:zero] + ("      if (allocated(oracle_band)) deallocate(oracle_band)\n"
                          "      if (allocated(oracle_gpt)) deallocate(oracle_gpt)\n"
                          "      allocate(oracle_band(klev+1,14,6), oracle_gpt(klev+1,112,6))\n"
                          "      oracle_band = 0._rb\n"
                          "      oracle_gpt = 0._rb\n\n") + text[zero:]
    anchor = "               pbbcddir(ikl) = pbbcddir(ikl) + zincflx(iw)*ztdbtc_nodel(jk)\n"
    added = "".join(f"               oracle_band(ikl,ibm,{q}) = oracle_band(ikl,ibm,{q}) + zincflx(iw)*{flux}\n"
                    f"               oracle_gpt(ikl,iw,{q}) = zincflx(iw)*{flux}\n"
                    for flux, q in QUANTITIES)
    return _once(text, anchor, anchor + added)


LW_QUANTITIES = (("uflux(lev)", 1), ("dflux(lev)", 2), ("uclfl(lev)", 3), ("dclfl(lev)", 4))


def patch_lw_module(text: str) -> str:
    start = text.index("      module rrtmg_lw_rtrnmc")
    contains = text.index("      contains", start)
    text = text[:contains] + ("      real(kind=rb), allocatable, public :: oracle_lwband(:,:,:)\n\n"
                              ) + text[contains:]
    sub = text.index("      subroutine rtrnmc", contains)
    zero = text.index("      urad(0) = 0.0_rb\n      drad(0) = 0.0_rb\n      totuflux(0) = 0.0_rb", sub)
    text = text[:zero] + ("      if (allocated(oracle_lwband)) deallocate(oracle_lwband)\n"
                          "      allocate(oracle_lwband(0:nlayers,16,4))\n"
                          "      oracle_lwband = 0.0_rb\n\n") + text[zero:]
    anchor = "            totdclfl(lev) = totdclfl(lev) + dclfl(lev) * delwave(iband)\n"
    added = "".join(f"            oracle_lwband(lev,iband,{q}) = {flux}*delwave(iband)*fluxfac\n"
                    for flux, q in LW_QUANTITIES)
    return _once(text, anchor, anchor + added)


def patch_lw_driver(text: str) -> str:
    text = _once(text, "  use rrtmg_lw_rtrnmc,  only : rtrnmc\n",
                 "  use rrtmg_lw_rtrnmc,  only : rtrnmc, oracle_lwband\n")
    anchor = "     call wr_r2('out/hrc', dhrc)\n"
    return _once(text, anchor, anchor + "     call wr_r3('out/lw_band', oracle_lwband)\n")


def patch_driver(text: str) -> str:
    text = _once(text, "  use rrtmg_sw_spcvmc,  only : spcvmc_sw\n",
                 "  use rrtmg_sw_spcvmc,  only : spcvmc_sw, oracle_band, oracle_gpt\n")
    dump = ("     call verify_wrf_level()\n"
            "     call wr2(trim(cpre)//'spectral/band', "
            "reshape(oracle_band, (/ size(oracle_band,1)*14, 6 /)))\n"
            "     call wr2(trim(cpre)//'spectral/gpt', "
            "reshape(oracle_gpt, (/ size(oracle_gpt,1)*112, 6 /)))\n")
    return _once(text, "     call verify_wrf_level()\n", dump)


def main(argv=None):
    source_root, build_dir = (Path(a) for a in (argv or sys.argv[1:]))
    build_dir.mkdir(parents=True, exist_ok=True)
    module = (source_root / "phys" / "module_ra_rrtmg_sw.F").read_text()
    (build_dir / "module_ra_rrtmg_sw_spectral.F").write_text(patch_module(module))
    driver = (HERE / "sw_fixture_driver.F90").read_text()
    (build_dir / "sw_spectral_driver.F90").write_text(patch_driver(driver))
    lw = (source_root / "phys" / "module_ra_rrtmg_lw.F").read_text()
    (build_dir / "module_ra_rrtmg_lw_spectral.F").write_text(patch_lw_module(lw))
    extract = (HERE / "lw_extract.F90").read_text()
    (build_dir / "lw_spectral_extract.F90").write_text(patch_lw_driver(extract))
    print("spectral_patch: OK")


if __name__ == "__main__":
    main()
