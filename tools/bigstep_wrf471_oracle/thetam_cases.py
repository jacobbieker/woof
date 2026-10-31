"""Write the theta_m / avgflx oracle's inputs from the real WRF patch.

    thetam_cases.py STATE_NPZ OUT_DIR

The thermodynamic state is tests/data/wrf471_bigstep/state-real.npz (a
12 x 10 x 49 patch of a real WRF v4.7.1 initialization: dry T, QVAPOR,
P/PB, PH/PHB, MU/MUB, ALB and the hybrid coefficients).  The remaining
inputs are seeded draws (numpy PCG64, seed 471) at WRF magnitudes:

* the microphysics' theta increment and vapour/cloud change, with 5% of the
  columns given increments past 0.07 K/s*dt so the clamped call
  (mp_tend_lim = 0.07) clamps while the default call (10 K/s) does not;
* the held coupled physics tendencies (rthblten ~ mu*1e-4 K/s, rqvblten ~
  mu*1e-7 /s) conv_t_tendf_to_moist converts;
* the AVGFLX means and sumflux fluxes (mu-coupled winds ~ 1e5 Pa m/s), with
  some flux words exactly -0.0 and +0.0 so zero_avgflx's reset is graded
  on the sign of zero.

The c3/c4 coefficients the hypsometric_opt = 2 EOS reads (the patch does
not carry them) are a hybrid B(eta) with etac = 0.2 and p_top = 5000 Pa,
evaluated in float64 and rounded once: both sides read the same words.

Writes OUT_DIR/thetam_inputs.bin (the driver's format, thetam_run.F90) and
OUT_DIR/thetam_inputs.npz (the same arrays in woof's (k, j, i) layout).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SEED = 471
P_TOP = 5000.0
DT = 12.0
LIM_A = 10.0       # Registry default mp_tend_lim
LIM_B = 0.07       # HRRR's mp_tend_lim (parm/conus/hrrr_wrfpre.nl)
ETAC = 0.2
P1000MB = 100000.0

FIELDS3 = ("t_dry", "qv", "qc", "ph", "phb", "p", "pb", "alb", "alt", "u",
           "v", "th_post", "qv_post", "qc_post", "t_tendf", "moist_tend",
           "avg_rum", "avg_rvm", "avg_wwm", "ru_m", "rv_m", "ww_m")
FIELDS2 = ("mu", "mub")
FIELDS1 = ("c1h", "c2h", "c3h", "c4h", "c1f", "c2f", "c3f", "c4f", "znu",
           "znw", "dnw", "rdnw", "rdn", "fnm", "fnp")


def _hybrid_c34(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WRF's B(eta) and (eta - B)(p0 - p_top) for hybrid_opt = 2."""
    eta = np.asarray(eta, np.float64)
    one = 1.0
    b1 = 2.0 * ETAC**2 * (one - ETAC)
    b2 = -ETAC * (4.0 - 3.0 * ETAC - ETAC**3)
    b3 = 2.0 * (one - ETAC**3)
    b4 = -(one - ETAC**2)
    b5 = (one - ETAC)**4
    b = (b1 + b2 * eta + b3 * eta**2 + b4 * eta**3) / b5
    b = np.where(eta < ETAC, 0.0, b)
    c4 = (eta - b) * (P1000MB - P_TOP)
    return b.astype(np.float32), c4.astype(np.float32)


def build(state_npz: Path) -> dict[str, np.ndarray]:
    z = np.load(state_npz)
    rng = np.random.Generator(np.random.PCG64(SEED))
    f32 = np.float32
    nz, ny, nx = z["T"].shape
    out: dict[str, np.ndarray] = {}
    out["t_dry"] = z["T"].astype(f32)
    out["qv"] = z["QVAPOR"].astype(f32)
    out["qc"] = z["QCLOUD"].astype(f32)
    out["ph"] = z["PH"].astype(f32)
    out["phb"] = z["PHB"].astype(f32)
    out["p"] = z["P"].astype(f32)
    out["pb"] = z["PB"].astype(f32)
    out["alb"] = z["ALB"].astype(f32)
    out["alt"] = (z["AL"] + z["ALB"]).astype(f32)
    out["u"] = z["U"].astype(f32)
    out["v"] = z["V"].astype(f32)
    mu = z["MU"].astype(f32)
    mub = z["MUB"].astype(f32)
    out["mu"], out["mub"] = mu, mub
    mut = (mub + mu).astype(f32)
    shape = (nz, ny, nx)
    # The scheme's theta increment (K) and vapour/cloud exchange.
    dth = rng.normal(0.0, 0.05, shape).astype(f32)
    hot = rng.random((ny, nx)) < 0.05
    dth[:, hot] += rng.uniform(1.0, 3.0, (nz, int(hot.sum()))).astype(f32)
    out["th_post"] = dth
    dq = (rng.normal(0.0, 2e-5, shape) * np.minimum(1.0, z["QVAPOR"] * 100.0)).astype(f32)
    out["qv_post"] = np.maximum(z["QVAPOR"] + dq, 0.0).astype(f32)
    out["qc_post"] = np.maximum(z["QCLOUD"] - dq, 0.0).astype(f32)
    out["t_tendf"] = (mut[None] * rng.normal(0.0, 1e-4, shape)).astype(f32)
    out["moist_tend"] = (mut[None] * rng.normal(0.0, 1e-7, shape)).astype(f32)
    for name, stag in (("rum", (nz, ny, nx + 1)), ("rvm", (nz, ny + 1, nx)),
                       ("wwm", (nz + 1, ny, nx))):
        out["avg_" + name] = rng.normal(0.0, 1e5, stag).astype(f32)
        flux = rng.normal(0.0, 1e5, stag).astype(f32)
        flat = flux.reshape(-1)
        flat[::97] = f32(-0.0)
        flat[1::97] = f32(0.0)
        out[name[0:2] + "_m"] = flux
    for name in ("C1H", "C2H", "C1F", "C2F", "ZNU", "ZNW", "DNW", "RDNW",
                 "RDN", "FNM", "FNP"):
        out[name.lower()] = z[name].astype(f32)
    out["c3h"], out["c4h"] = _hybrid_c34(z["ZNU"])
    out["c3f"], out["c4f"] = _hybrid_c34(z["ZNW"])
    return out


def _memory(array: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    """woof's (k, j, i) array in the driver's (1:nx+1, 1:nz+1, 1:ny+1)."""
    mem = np.zeros((nx + 1, nz + 1, ny + 1), np.float32)
    k, j, i = array.shape
    mem[:i, :k, :j] = np.transpose(array, (2, 0, 1))
    return mem


def write(cases: dict[str, np.ndarray], out: Path) -> None:
    nz, ny, nx = cases["t_dry"].shape
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "thetam_inputs.bin", "wb") as handle:
        handle.write(np.array([nx, ny, nz, 3], "<i4").tobytes())
        handle.write(np.array([P_TOP, DT, LIM_A, LIM_B], "<f4").tobytes())
        for name in FIELDS3:
            handle.write(_memory(cases[name], nx, ny, nz).astype("<f4")
                         .tobytes(order="F"))
        for name in FIELDS2:
            mem = np.zeros((nx + 1, ny + 1), np.float32)
            mem[:nx, :ny] = cases[name].T
            handle.write(mem.astype("<f4").tobytes(order="F"))
        for name in FIELDS1:
            mem = np.zeros(nz + 1, np.float32)
            mem[:cases[name].size] = cases[name]
            handle.write(mem.astype("<f4").tobytes())
    np.savez(out / "thetam_inputs.npz", **cases)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("state_npz", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    write(build(args.state_npz), args.out)


if __name__ == "__main__":
    main()
