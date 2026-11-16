"""Columns and regimes for the RRTMG-K column oracle.

The 128 columns of the legacy RRTMG column oracle
(tools/rrtmg_legacy_column_oracle/make_columns.py: convective land,
tropical and marine cloud, stable night, snow and sea ice, polar night,
high terrain, grazing sun, 22 edge cases, mixed random columns) plus a
cloud droplet number ``qnc`` (kg-1) for the schemes that carry one
(f_qnc): physical where there is cloud water, and pushed to effectRad's
branch edges elsewhere (no droplets under cloud water, droplets without
cloud water, tiny and huge counts, negative counts).

RRTMG-K's shortwave cloud optics only run with explicit snow
(iceflag 5, f_qs); every regime here carries qs.  Regimes: Thompson
(CAM ozone, wrapper ozone, p_top 2 kPa), aerosol-aware Thompson with
QNC (f_qnc), WSM6, Morrison.

usage: python make_columns.py OUT_DIR
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_rrtmg_legacy_make_columns",
    HERE.parent / "rrtmg_legacy_column_oracle" / "make_columns.py")
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)

F = np.float32
NCOL, NZ = legacy.NCOL, legacy.NZ

#: regime name -> options (aer_opt 0 throughout: RRTMG-K's aerod is aer_opt 1).
REGIMES = {
    "thompson8":        dict(mp=8,  grid="ptop5000", o3input=2, aer_opt=0),
    "thompson8_o3wrap": dict(mp=8,  grid="ptop5000", o3input=0, aer_opt=0),
    "thompson8_ptop2k": dict(mp=8,  grid="ptop2000", o3input=2, aer_opt=0),
    "thompson28":       dict(mp=28, grid="ptop5000", o3input=2, aer_opt=0),
    "wsm6":             dict(mp=6,  grid="ptop5000", o3input=2, aer_opt=0),
    "morrison10":       dict(mp=10, grid="ptop5000", o3input=2, aer_opt=0),
}
#: Registry packages that carry QNC (moist_scalar qnc): Thompson aerosol-aware.
F_QNC = {8: 0, 28: 1, 6: 0, 10: 0}


def droplet_number(grid, seed=20261116):
    """qnc (kg-1) per column and level, float32."""
    rng = np.random.default_rng(seed)
    qc = grid["qc"]
    nc = np.where(qc > 0, rng.uniform(2.0e7, 6.0e8, qc.shape), 0.0)
    # sparse droplets with no cloud water above the cloud
    nc = np.where((qc == 0) & (rng.uniform(size=qc.shape) < 0.05),
                  rng.uniform(1.0, 1.0e6, qc.shape), nc)
    edge = NCOL - 22            # legacy edge columns follow the families
    nc[edge + 0, :] = 0.0                           # cloud water, no droplets
    nc[edge + 1, :] = np.where(qc[edge + 1] > 0, 1.0e-3, 0.0)   # rnc below r2
    nc[edge + 2, :] = np.where(qc[edge + 2] > 0, 5.0e10, 0.0)   # very many, lamc large
    nc[edge + 3, :] = -1.0e7                        # negative count
    nc[edge + 4, ::2] = 1.0e5                       # near r2: rho*nc vs 1e-6
    return nc.astype(F)


def write_input(path, grid, regime):
    legacy.write_fortran_input(path, grid, regime)
    raw = bytearray(path.read_bytes())
    hdr = np.frombuffer(bytes(raw[:96]), np.int32).copy()
    hdr[22] = F_QNC[regime["mp"]]
    raw[:96] = hdr.tobytes()
    qnc = grid["qnc"] if F_QNC[regime["mp"]] else np.zeros_like(grid["qnc"])
    raw += np.asfortranarray(np.asarray(qnc, F)).tobytes(order="F")
    path.write_bytes(bytes(raw))


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    grids = {g: legacy.build_grid(p) for g, p in legacy.P_TOP.items()}
    for g, grid in grids.items():
        grid["qnc"] = droplet_number(grid)
        np.savez_compressed(out / f"columns_{g}.npz", **grid)
    meta = {"start": legacy.START.isoformat(), "elapsed_seconds": legacy.ELAPSED,
            "radt_minutes": legacy.RADT_MINUTES, "ncol": NCOL, "nz": NZ,
            "calendar": {k: float(v) for k, v in legacy.calendar().items()},
            "regimes": {}}
    for name, r in REGIMES.items():
        write_input(out / f"in_{name}.bin", grids[r["grid"]], r)
        meta["regimes"][name] = dict(r, species=legacy.SPECIES[r["mp"]],
                                     f_qnc=F_QNC[r["mp"]])
    (out / "regimes.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(REGIMES)} regimes x {NCOL} columns")


if __name__ == "__main__":
    main()
