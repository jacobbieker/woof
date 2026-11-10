"""RADM2/SORGAM anthropogenic emissions against WRF v4.7.1, word for word.

tools/chem_wrf471_oracle/radm2emis/run_radm2emis.F90 runs the unmodified
add_anthropogenics (chem/module_emissions_anthropogenics.F, compiled whole)
and the byte-copied sorgam_addemiss (chem/module_aerosols_sorgam.F) for
chem_opt = 2, emiss_opt = 3 on 16 columns of 6 levels, three cases varying
dtstep and kemit (up to the whole column), with densities, layer depths and
rates spanning decades and some rates exactly zero.  Every chem row must
equal the Fortran's float32 word after the call, the rows no emission
touches included.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify.chem_oracle import load

ROOT = Path(__file__).resolve().parents[1] / "tests/data/oracles/chem/radm2emis"
#: Registry package radm2sorg (registry.chem:3930), after the index-1 dummy.
CHEM = ("so2,sulf,no2,no,o3,hno3,h2o2,ald,hcho,op1,op2,paa,ora1,ora2,nh3,n2o5,no3,"
        "pan,hc3,hc5,hc8,eth,co,ol2,olt,oli,tol,xyl,aco3,tpan,hono,hno4,ket,gly,mgly,"
        "dcb,onit,csl,iso,hcl,ho,ho2,so4aj,so4ai,nh4aj,nh4ai,no3aj,no3ai,naaj,naai,"
        "claj,clai,orgaro1j,orgaro1i,orgaro2j,orgaro2i,orgalk1j,orgalk1i,orgole1j,"
        "orgole1i,orgba1j,orgba1i,orgba2j,orgba2i,orgba3j,orgba3i,orgba4j,orgba4i,"
        "orgpaj,orgpai,ecj,eci,p25j,p25i,antha,seas,soila,nu0,ac0,corn").split(",")
#: Registry package eradmsorg (registry.chem:4055), after the index-1 dummy.
EMIS = ("e_iso,e_so2,e_no,e_no2,e_co,e_ch4,e_eth,e_hc3,e_hc5,e_hc8,e_xyl,e_ol2,"
        "e_olt,e_oli,e_tol,e_csl,e_hcho,e_ald,e_ket,e_ora2,e_nh3,e_terp,e_hono,e_co2,"
        "e_pm25i,e_pm25j,e_pm_10,e_eci,e_ecj,e_orgi,e_orgj,e_so4i,e_so4j,e_no3i,"
        "e_no3j,e_naaj,e_naai,e_orgi_a,e_orgj_a,e_orgi_bb,e_orgj_bb,e_hcl,e_cli,"
        "e_clj,e_ch3cl").split(",")


def _engine(a):
    """Fortran (i, k, j) -> WOOF (k, j, i)."""
    return np.ascontiguousarray(np.transpose(a, (1, 2, 0)), dtype=np.float32)


def test_fixture_checksums_and_coverage():
    for line in (ROOT / "oracle-sha256sums.txt").read_text().splitlines():
        digest, rel = line.split()
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    cases = load(ROOT)
    nz = cases["init"]["chem"].shape[1]
    kemits = sorted(int(cases[f"case{c}"]["kemit"]) for c in (1, 2, 3))
    assert kemits[0] == 1 and kemits[-1] == nz
    for c in (1, 2, 3):
        e = cases[f"case{c}"]["emis_ant"]
        assert np.any(e[..., 1:] == 0) and np.all(e[..., 1:] >= 0)


def test_tables_match_the_registry_packages():
    from woof.core.chem_radm2 import SPECIES
    from woof.core.chem_radm2_emiss import AEROSOL_EMISSIONS, AEROSOL_ROWS, GAS_TERMS
    from woof.core.chem_sorgam import AEROSOLS
    for row, e in GAS_TERMS:
        assert (row in SPECIES or row == "hcl") and row in CHEM and e in EMIS, (row, e)
    assert set(AEROSOL_ROWS) <= set(AEROSOLS)
    assert set(AEROSOL_EMISSIONS) <= set(EMIS)


def test_factors_are_the_oracles():
    from woof.core.chem_radm2_emiss import factor_values
    init = load(ROOT)["init"]
    for name, value in factor_values().items():
        assert np.float32(init[name]).view(np.uint32) == value.view(np.uint32), name


@requires_gpu
@pytest.mark.parametrize("case", [1, 2, 3])
@pytest.mark.parametrize("chem_opt", [1, 2])
def test_emissions_match_wrf_word_for_word(case, chem_opt):
    import cupy as cp
    from woof.core.chem_radm2_emiss import (AEROSOL_EMISSIONS, AEROSOL_ROWS, GAS_TERMS,
                                            launch_radm2_emiss)
    cases = load(ROOT)
    init, out = cases["init"], cases[f"case{case}"]
    chem = init["chem"]
    rows = {name: cp.asarray(_engine(chem[:, :, :, i + 1])) for i, name in enumerate(CHEM)}
    emis = out["emis_ant"]
    frame = {name: cp.asarray(_engine(emis[:, :, :, i + 1])) for i, name in enumerate(EMIS)}
    f3 = {n: cp.asarray(_engine(init[n])) for n in ("rho_phy", "alt", "dz8w")}
    kw = {}
    if chem_opt == 2:
        kw = dict(aerosol=[rows[n] for n in AEROSOL_ROWS],
                  aerosol_emis=[frame[n] for n in AEROSOL_EMISSIONS])
    launch_radm2_emiss([rows[r] for r, _ in GAS_TERMS], [frame[e] for _, e in GAS_TERMS],
                       f3["rho_phy"], f3["alt"], f3["dz8w"], float(out["dtstep"]),
                       kemit=int(out["kemit"]), **kw)
    want = out["chem"]
    bad = []
    for i, name in enumerate(CHEM):
        # chem_opt = 1 carries no aerosol rows: they must be untouched.
        exp = _engine((want if chem_opt == 2 or name not in AEROSOL_ROWS else chem)[:, :, :, i + 1])
        got = rows[name].get()
        if not np.array_equal(got.view(np.uint32), exp.view(np.uint32)):
            bad.append((name, int(np.sum(got.view(np.uint32) != exp.view(np.uint32)))))
    assert not bad, f"case {case}, chem_opt = {chem_opt}: words differ in {bad}"


@requires_gpu
def test_refuses_kemit_past_the_column_with_aerosol():
    import cupy as cp
    from woof.core.chem_radm2_emiss import (AEROSOL_EMISSIONS, AEROSOL_ROWS, GAS_TERMS,
                                            launch_radm2_emiss)
    z = lambda: cp.zeros((3, 2, 2), cp.float32)  # noqa: E731
    gas = [z() for _ in GAS_TERMS]
    with pytest.raises(ValueError, match="kemit"):
        launch_radm2_emiss(gas, [z() for _ in GAS_TERMS], z(), z(), z(), 60., kemit=4,
                           aerosol=[z() for _ in AEROSOL_ROWS],
                           aerosol_emis=[z() for _ in AEROSOL_EMISSIONS])


def test_gas_terms_are_add_anthropogenics_radm2_branch():
    """Read off the pinned WRF source when a stock tree is present: every
    ``chem(p_x) += emis_ant(p_e_y)*conv_rho`` line of the RADM2 branch whose
    two rows both exist in radm2sorg / eradmsorg, in WRF's order."""
    import os
    import re
    tree = Path(os.environ.get("WOOF_WRF471_STOCK_TREE", str(Path.home() / "Development/wrf-src/WRF-4.7.1")))
    src = tree / "chem" / "module_emissions_anthropogenics.F"
    if not src.is_file():  # pragma: no cover
        pytest.skip(f"stock WRF v4.7.1 tree not present at {tree}")
    for line in (Path(__file__).resolve().parents[1]
                 / "tools/chem_wrf471_oracle/radm2emis/SOURCES.sha256").read_text().splitlines():
        if line and not line.startswith("#"):
            digest, rel = line.split()
            assert hashlib.sha256((tree / rel).read_bytes()).hexdigest() == digest, rel
    from woof.core.chem_radm2_emiss import GAS_TERMS
    text = src.read_text()
    body = text[text.index("subroutine add_anthropogenics"):text.index("END subroutine add_anthropogenics")]
    head = body[:body.index("is_mozart:if")]
    tail = body[body.index("else is_mozart"):body.index("endif is_mozart")]
    pat = re.compile(r"chem\((?:its:ite|i),k,j,p_(\w+)\)\s*=\s*chem\((?:its:ite|i),k,j,p_\1\)\s*\+\s*"
                     r"(?:0\.5\*)?emis_ant\((?:its:ite|i),k,j,p_(e_\w+)\)\*conv_rho")
    found = [(m.group(1), m.group(2)) for part in (head, tail) for m in pat.finditer(part)]
    have = [(r, e) for r, e in found if r in CHEM and e in EMIS]
    assert have == list(GAS_TERMS)
