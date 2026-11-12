#!/usr/bin/env python3
"""Deterministic synthetic mesh columns for the p_lev / z_lev oracle (no data from any case).

Writes ``columns.bin``: little endian int32 header (nc, nz, extrap, n_p, n_z) then float32 arrays in this
order, each C-contiguous: p_levels[n_p], z_levels[n_z], then (nz, nc) mass fields u, v, theta, qv, pp, pb,
then (nz+1, nc) interface fields zgrid, pw.  ``pw`` is the interface pressure of
woof.hex.mpasdiag.plevel_diag_v842.interface_pressure (the oracle takes it as an input: it is the one input
the mesh route defines instead of WRF's p_hyd_w).

    make_columns.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from woof.hex.mpasdiag import plevel_diag_v842 as pl  # noqa: E402

NC, NZ = 96, 30
P_LEVELS = (100000.0, 92500.0, 85000.0, 70000.0, 50000.0, 30000.0, 5000.0, 85000.0, 100000.0, 20000.0)
Z_LEVELS = (-10.0, -100.0, -500.0, 1000.0, 3000.0, 6000.0, 12000.0, 40000.0, 500.0, -50.0)


def make(extrap: int):
    rng = np.random.default_rng(20261112 + extrap)
    f = np.float32
    terrain = rng.uniform(0.0, 2200.0, NC).astype(f)
    top = 18000.0
    eta = np.linspace(0.0, 1.0, NZ + 1)[:, None] ** 1.3
    zgrid = (terrain[None, :] + (top - terrain[None, :]) * eta).astype(f)
    zmid = 0.5 * (zgrid[:-1] + zgrid[1:])
    p0 = (101325.0 * np.exp(-terrain / 8400.0) * rng.uniform(0.985, 1.015, NC)).astype(f)
    pb = (p0[None, :] * np.exp(-(zmid - terrain[None, :]) / 8000.0)).astype(f)
    pp = (rng.normal(0.0, 120.0, (NZ, NC))).astype(f)
    theta = (288.0 + 5.0e-3 * (zmid - 0) + rng.normal(0, 1.2, (NZ, NC))).astype(f)
    qv = np.maximum(0.012 * np.exp(-(zmid - terrain[None, :]) / 2500.0) * rng.uniform(0.6, 1.2, (NZ, NC)), 0).astype(f)
    qv[rng.random((NZ, NC)) < 0.01] = -1.0e-5  # WRF clips negative vapour at zero
    u = (rng.normal(8, 6, (NZ, NC)) * (1 + zmid / 8000.0)).astype(f)
    v = rng.normal(-2, 5, (NZ, NC)).astype(f)
    p = (pb + pp).astype(f)
    pw = pl.interface_pressure(np, p, zgrid).astype(f)
    return dict(u=u, v=v, theta=theta, qv=qv, pp=pp, pb=pb, zgrid=zgrid, pw=pw)


def write(path: Path, extrap: int, fields) -> None:
    with path.open("wb") as out:
        np.array([NC, NZ, extrap, len(P_LEVELS), len(Z_LEVELS)], dtype="<i4").tofile(out)
        np.array(P_LEVELS, dtype="<f4").tofile(out)
        np.array(Z_LEVELS, dtype="<f4").tofile(out)
        for name in ("u", "v", "theta", "qv", "pp", "pb", "zgrid", "pw"):
            np.ascontiguousarray(fields[name], dtype="<f4").tofile(out)


if __name__ == "__main__":
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    for extrap in (1, 2):
        write(target / f"columns_extrap{extrap}.bin", extrap, make(extrap))
