"""Cases for the module_stoch pattern-arithmetic oracle (run_stoch.F90).

    python make_stoch_inputs.py OUT_DIR [--seed N]

Writes OUT_DIR/stoch_inputs.bin.  Deterministic.  Each case is one
STOCHPERT.TBL field on one domain -- even and odd staggered sizes, square
and not, 1.5 to 27 km grids, WRF-Solar's table rows (2-D and random-phase
3-D) and a few off-table ones -- advanced over three updates with supplied
Gaussian draws, truncated at |z| < 3 as UPDATE_STOCH truncates them.  The
first case is the WRF-Solar default QVAPOR row on a 3 km grid updated
every 60 s step (spdt = -1).
"""
import argparse
import pathlib

import numpy as np

F = np.float32
# (lmax, kmax, nz, dx, dy, time_step, stepsp, std, length, tau, cutoff, vertstruc)
CASES = [
    (41, 33, 4, 3000.0, 3000.0, 18, 1, 0.05, 100000.0, 3600.0, 3.0, 1),
    (41, 33, 3, 3000.0, 3000.0, 18, 1, 0.10, 100000.0, 86400.0, 3.0, 0),
    (60, 48, 2, 9000.0, 9000.0, 54, 5, 0.25, 100000.0, 3600.0, 3.0, 0),
    (37, 52, 4, 1500.0, 1500.0, 9, 10, 0.001, 100000.0, 3600.0, 3.0, 1),
    (64, 64, 3, 27000.0, 27000.0, 150, 2, 0.10, 80000.0, 21600.0, 3.0, 1),
    (45, 31, 3, 4000.0, 3500.0, 20, 3, 0.05, 80000.0, 600.0, 2.0, 1),
    (30, 30, 2, 12000.0, 12000.0, 60, 1, 0.30, 500000.0, 1800.0, 1.5, 0),
]
NSTEP = 3


def truncated(rng, shape):
    z = rng.normal(0.0, 1.0, shape)
    bad = np.abs(z) >= 3.0
    while bad.any():
        z[bad] = rng.normal(0.0, 1.0, bad.sum())
        bad = np.abs(z) >= 3.0
    return z.astype(F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=471)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stoch_inputs.bin", "wb") as f:
        np.array([len(CASES)], "<i4").tofile(f)
        for (lmax, kmax, nz, dx, dy, ts, stepsp, std, length, tau, cutoff,
             vert) in CASES:
            np.array([lmax, kmax, nz, stepsp * ts, vert, NSTEP], "<i4").tofile(f)
            np.array([dx, dy, std, length, tau, cutoff], "<f4").tofile(f)
            for _ in range(NSTEP):
                # (kmax, lmax) row-major == Fortran (lmax, kmax)
                truncated(rng, (kmax, lmax)).astype("<f4").tofile(f)
                truncated(rng, (kmax, lmax)).astype("<f4").tofile(f)
    print("wrote", out / "stoch_inputs.bin", len(CASES), "cases")


if __name__ == "__main__":
    main()
