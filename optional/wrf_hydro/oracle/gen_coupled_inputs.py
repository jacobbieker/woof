# WOOF-OPTIN: wrf-hydro
"""Write the coupled oracle's inputs: the synthetic domain
(tests/wrf_hydro_synthetic.py) and ``coupled-inputs.txt``, the LSM fields
handed to wrf_cpl_HYDRO on each cycle, as binary32 bit patterns.

    pixi run -e default python optional/wrf_hydro/oracle/gen_coupled_inputs.py OUT_DIR

The parity test (tests/test_wrf_hydro_coupled_parity.py) regenerates the
same inputs with :func:`cycles` and checks their hash against the build's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

NX, NY, NSOIL, NCYCLES, HYDRO_DT = 8, 6, 4, 4, 60.0
DZS = (0.1, 0.3, 0.6, 1.0)
#: the mass grid and AGGFACTRT per variant; GWBASESWCRT = 3 runs on
#: AGGFACTRT = 1 with JXRT >= IXRT (woof_wrf_hydro.routing_domain, gw2d)
GRIDS = {"": (NX, NY, 4), "da": (NX, NY, 4), "gw2d": (8, 10, 1), "reach": (NX, NY, 4),
         "musk": (NX, NY, 4)}


def vegsoil(variant=""):
    NX, NY, _ = GRIDS[variant]
    veg = np.full((NY, NX), 10, np.int32)
    veg[:, -1] = 17
    sol = np.full((NY, NX), 6, np.int32)
    sol[:, -1] = 14
    return veg, sol


def cycles(variant=""):
    """Per cycle: smois, sh2o (nsoil, ny, nx), infxsrt, soldrain (ny, nx),
    binary32.  Saturated deep layers (lateral subsurface flow), frozen water
    in the top layer of one column, ponding that varies cell to cell."""
    NX, NY, _ = GRIDS[variant]
    f = np.float32
    out = []
    jj, ii = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    for c in range(NCYCLES):
        smois = np.empty((NSOIL, NY, NX), f)
        smois[0] = f(0.30) + f(0.01) * ((ii + jj + c) % 5).astype(f)
        smois[1] = f(0.33)
        smois[2] = f(0.41) + f(0.005) * (ii % 3).astype(f)
        smois[3] = f(0.436)
        smois[:, :, -1] = f(1.0)
        sh2o = smois.copy()
        sh2o[0, 2, 3] = f(smois[0, 2, 3] - f(0.05))
        infx = (f(4.0) + f(3.0) * ((ii * 7 + jj * 3 + c) % 4).astype(f)).astype(f)
        infx[:, -1] = f(0.0)
        sold = (f(0.05) + f(0.01) * jj.astype(f)).astype(f)
        sold[:, -1] = f(0.0)
        out.append((smois, sh2o, infx, sold))
    return out


def write(out_dir, variant=""):
    """``variant = "da"``: the lake is lake_option 3's USGS hybrid
    (wrf_hydro_synthetic.write_reservoir_da); ``"gw2d"``: the groundwater is
    GWBASESWCRT = 3's 2-D aquifer (wrf_hydro_synthetic.write_gw2d); ``"reach"``
    and ``"musk"``: Route_Link reaches routed by Muskingum-Cunge
    (channel_option 2) and Muskingum (1) (wrf_hydro_synthetic.write_route_link)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
    import wrf_hydro_synthetic as synthetic

    from woof.wrf_hydro_route import soil_centres

    NX, NY, agg = GRIDS[variant]
    out = Path(out_dir)
    synthetic.write(out, nx=NX, ny=NY, agg=agg, dxrt=250.0)
    if variant == "da":
        from datetime import datetime

        synthetic.write_reservoir_da(out, start=datetime(2020, 1, 1))
    if variant == "gw2d":
        synthetic.write_gw2d(out)
    if variant in ("reach", "musk"):
        synthetic.write_route_link(out, channel_option=2 if variant == "reach" else 1)
    veg, sol = vegsoil(variant)
    bits = lambda a: np.asarray(a, np.float32).view(np.int32)  # noqa: E731
    lines = [f"{NX} {NY} {NSOIL} {NCYCLES} {HYDRO_DT}"]
    lines += [str(int(b)) for b in bits(soil_centres(DZS))]
    lines += [f"{int(veg[j, i])} {int(sol[j, i])}" for j in range(NY) for i in range(NX)]
    for smois, sh2o, infx, sold in cycles(variant):
        for j in range(NY):
            for i in range(NX):
                for k in range(NSOIL):
                    lines += [str(int(bits(smois[k, j, i]))), str(int(bits(sh2o[k, j, i])))]
                lines += [str(int(bits(infx[j, i]))), str(int(bits(sold[j, i])))]
    (out / "coupled-inputs.txt").write_text("\n".join(lines) + "\n")
    # read_rt_nlst reads &CROCUS_nlist from namelist.hrldas unconditionally
    # (config.F90 :724, :1218-1222) and stops when the file is absent, even
    # coupled to WRF; with no such group crocus_opt stays 0
    (out / "namelist.hrldas").write_text("")


if __name__ == "__main__":
    write(sys.argv[1], *sys.argv[2:3])
