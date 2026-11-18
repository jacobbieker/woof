"""Run WOOF's RRTMG-K on the oracle columns.

Two arms per regime:

``engine``  woof.core.rrtmgk.rrtmgk_columns fed the WRF record's own driver
            inputs (t8w, cldfra, o3rad, coszen, solcon), so the wrapper,
            McICA and both K engines are graded alone;
``adapter`` the forecast adapter woof.core.rrtmgk.RRTMGKRadiation called as
            the forecast calls it (``__call__(atmosphere=, fields=, state=,
            cfg=)``) on a (1, ncol) grid: its own t8w, cal_cldfra1, CAM
            ozone and sun.

The process arithmetic is whatever GPUWM_WRF_EXACT says at import; the
output name records which (woof_<arm>_<exact>_<regime>.npz).

usage: python run_woof.py COLUMNS_DIR WRF_DIR OUT_DIR [REGIME ...]
  WRF_DIR/out_<regime>.bin is the oracle record (engine-arm inputs)

tests/test_rrtmg_k_column_oracle.py calls run_engine / run_adapter with the
fixture's columns and its packed records (record(): "<field>" -> columns).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _compare():
    """This directory's compare.py, loaded by path: a ``compare`` already
    imported from another oracle directory must not answer for it."""
    import importlib.util
    name = "rrtmg_k_column_oracle_compare"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parent / "compare.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


read_records, to_cols = _compare().read_records, _compare().to_cols

F = np.float32
SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")
OUTPUTS = ("rthratenlw", "rthratensw", "glw", "olr", "gsw", "lwupt",
           "lwuptc", "lwdnt", "lwdntc", "lwupb", "lwupbc", "lwdnb",
           "lwdnbc", "swupt", "swuptc", "swdnt", "swdntc", "swupb",
           "swupbc", "swdnb", "swdnbc", "re_qc", "re_qi", "re_qs")


def record(wrf_dir, name, ncol):
    """The oracle's out_<regime>.bin as columns: (ncol, nk) or (ncol,)."""
    return {k: (to_cols(v, ncol) if v.size % ncol == 0 else v) for k, v in
            read_records(Path(wrf_dir) / f"out_{name}.bin").items()}


def run_engine(cols_dir, rec, meta, name):
    """woof.core.rrtmgk.rrtmgk_columns on WRF's own driver inputs ``rec``."""
    import cupy as cp
    from woof.core import rrtmgk

    r = meta["regimes"][name]
    g = dict(np.load(Path(cols_dir) / f"columns_{r['grid']}.npz"))
    ncol, nz = meta["ncol"], meta["nz"]
    on = dict(zip(SPECIES, r["species"]))

    def dev(a):
        return cp.asarray(np.ascontiguousarray(np.asarray(a, F)))
    zero = np.zeros((ncol, nz), F)
    q = {s: dev(g[s] if on[s] else zero) for s in SPECIES}
    out = rrtmgk.rrtmgk_columns(
        p3d=dev(g["p3d"]), p8w=dev(g["p8w"]),
        t3d=dev(g["t3d"]), t8w=dev(rec["t8w"]),
        rho=dev(g["rho"]), qv=q["qv"], qc=q["qc"], qi=q["qi"], qs=q["qs"],
        qg=q["qg"], nc=dev(g["qnc"] if r["f_qnc"] else zero),
        cldfra=dev(rec["cldfra"]),
        o33d=dev(rec["o3rad"]), tsk=dev(g["tsk"]),
        emiss=dev(g["emiss"]), albedo=dev(g["albedo"]),
        coszen=dev(rec["coszen"]), solcon=F(np.ravel(rec["solcon"])[0]),
        f_qc=on["qc"], f_qi=on["qi"], f_qs=on["qs"], f_qnc=bool(r["f_qnc"]),
        o3input=r["o3input"], radii=True)
    res = {k: cp.asnumpy(out[k]) for k in OUTPUTS}
    lw, sw = res["rthratenlw"], res["rthratensw"]
    total = (lw + sw).astype(F)
    res["rthratensw_wrapper"] = sw
    res["rthraten"] = total
    res["rthratensw"] = (total - lw).astype(F)
    res["swdown"] = (res["gsw"] / (F(1.0) - np.asarray(g["albedo"], F))).astype(F)
    return res


def run_adapter(cols_dir, rec, meta, name):
    """The forecast adapter on the regime's columns (``rec`` unused)."""
    import cupy as cp
    from woof.core.rrtmgk import RRTMGKRadiation

    r = meta["regimes"][name]
    g = dict(np.load(Path(cols_dir) / f"columns_{r['grid']}.npz"))
    ncol, nz = meta["ncol"], meta["nz"]
    ny, nx = 1, ncol

    def grid3(a):            # (ncol, nk) -> (nk, 1, ncol) device
        return cp.asarray(np.ascontiguousarray(np.asarray(a, F).T.reshape(-1, ny, nx)))

    def grid2(a):
        return cp.asarray(np.ascontiguousarray(np.asarray(a, F).reshape(ny, nx)))

    on = dict(zip(SPECIES, r["species"]))
    state = SimpleNamespace(
        fnm=cp.asarray(g["fnm"]), fnp=cp.asarray(g["fnp"]),
        p_top=F(g["p_top"][0]), elapsed_seconds=float(meta["elapsed_seconds"]),
        physics=None)
    for s in SPECIES:
        setattr(state, s, grid3(g[s]) if on[s] else None)
    state.nc = grid3(g["qnc"]) if r["f_qnc"] else None
    atmosphere = {"pressure": grid3(g["p3d"]), "p_interface": grid3(g["p8w"]),
                  "temperature": grid3(g["t3d"]), "exner": grid3(g["pi3d"]),
                  "dz": grid3(g["dz8w"]), "z_interface": grid3(g["z_at_w"]),
                  "rho": grid3(g["rho"])}
    fields = {k: grid2(g[k]) for k in ("tsk", "emiss", "albedo", "xland",
                                       "xice", "snow")}
    cfg = SimpleNamespace(mp_physics=r["mp"], radt=0.0,
                          radt_minutes=meta["radt_minutes"], icloud=1,
                          cldovrlp=2, idcor=0, ghg_input=0, aer_opt=0,
                          o3input=r["o3input"], bl_pbl_physics=0, icloud_bl=0,
                          cu_physics=0, shcu_physics=0, dt=20.0,
                          rrtmg_spectral_output="none", use_theta_m=0,
                          solar_generation="wrf_461")
    adapter = RRTMGKRadiation(
        datetime.fromisoformat(meta["start"]), g["xlat"].reshape(ny, nx),
        g["xlong"].reshape(ny, nx), o3input=r["o3input"])
    res = adapter(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    out = {}
    for key in ("rthratenlw", "rthratensw", "glw", "olr", "gsw", "swdown",
                "coszen", "cldfra"):
        v = cp.asnumpy(getattr(res, key))
        out[key] = v.reshape(nz, ncol).T.copy() if v.ndim == 3 else v.reshape(ncol)
    if res.rthraten_total is not None:   # use_theta_m = 1 only
        out["rthraten"] = cp.asnumpy(res.rthraten_total).reshape(nz, ncol).T.copy()
    else:                                # WRF: RTHRATEN = LW, then += SW
        out["rthraten"] = (out["rthratenlw"] + out["rthratensw"]).astype(F)
    for key, v in res.boundary_fluxes.items():
        out[key] = cp.asnumpy(v).reshape(ncol)
    o3 = adapter.o33d_device()
    if o3 is not None:
        out["o3rad"] = cp.asnumpy(o3).reshape(nz, ncol).T.copy()
    return {k: np.asarray(v, F) for k, v in out.items()}


def main():
    cols_dir, wrf_dir, out_dir = (Path(a) for a in sys.argv[1:4])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((cols_dir / "regimes.json").read_text())
    names = sys.argv[4:] or list(meta["regimes"])
    from woof import wrf_exact
    exact = "strict" if wrf_exact.ENABLED else "default"
    arms = os.environ.get("RRTMGK_ARMS", "engine,adapter").split(",")
    for name in names:
        rec = record(wrf_dir, name, meta["ncol"])
        for arm in arms:
            fn = run_engine if arm == "engine" else run_adapter
            out = fn(cols_dir, rec, meta, name)
            np.savez_compressed(out_dir / f"woof_{arm}_{exact}_{name}.npz", **out)
            print(f"{arm} {exact} {name}: ok", flush=True)


if __name__ == "__main__":
    main()
