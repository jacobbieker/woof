"""Write the second theta_m oracle's inputs from the real WRF patch.

    thetam_ext_cases.py STATE_NPZ OUT_DIR

The thermodynamic state is the first theta_m oracle's
(tools/bigstep_wrf471_oracle/thetam_cases.py over
tests/data/wrf471_bigstep/state-real.npz, a 12 x 10 x 49 patch of a real
WRF v4.7.1 initialization), with the prognostic t_m = real.exe's theta_m of
its dry T in float32 (any words do: both sides read these).  Seeded draws
(numpy PCG64, seed 4712) at WRF magnitudes add:

* the coupled advective forcing pair phy_prep_part2 decouples and converts
  (RTHFTEN ~ mut*3e-4 K/s, RQVFTEN ~ mut*1e-7 /s, signed);
* the surface fluxes vertical_diffusion_2's hflux reads: HFX (W m-2, -50 to
  400) and QFX (kg m-2 s-1, -2e-5 to 3e-4), with exact zeros in some
  columns; and the namelist tke_heat_flux 0.24 K m s-1 (isfflx = 0, 2).

Writes OUT_DIR/thetam_ext_inputs.bin (thetam_ext_run.F90's format) and
OUT_DIR/thetam_ext_inputs.npz (woof's (k, j, i) layout).
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

SEED = 4712
P_TOP = 5000.0
HEAT_FLUX = 0.24
RVOVRD = np.float32(461.6) / np.float32(287.0)

FIELDS3 = ("t_m", "qv", "qc", "ph", "phb", "p", "pb", "alt", "u", "v",
           "t_tendf", "rthften", "rqvften")
FIELDS2 = ("mu", "mub", "hfx", "qfx")
FIELDS1 = ("c1h", "c2h", "c1f", "c2f", "dnw", "fnm", "fnp", "znw")


def _base_cases():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "thetam_cases", here / "thetam_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(state_npz: Path) -> dict[str, np.ndarray]:
    base = _base_cases().build(state_npz)
    f32 = np.float32
    rng = np.random.Generator(np.random.PCG64(SEED))
    out = {name: base[name] for name in
           ("qv", "qc", "ph", "phb", "p", "pb", "alt", "u", "v", "t_tendf",
            "mu", "mub", "c1h", "c2h", "c1f", "c2f", "dnw", "fnm", "fnp",
            "znw")}
    t = base["t_dry"].astype(f32)
    qv = base["qv"].astype(f32)
    out["t_m"] = ((t + f32(300.0)) * (f32(1.0) + RVOVRD * qv)
                  - f32(300.0)).astype(f32)
    nz, ny, nx = t.shape
    mut = (base["mub"] + base["mu"]).astype(f32)
    out["rthften"] = (mut[None] * rng.normal(0.0, 3e-4, (nz, ny, nx))).astype(f32)
    out["rqvften"] = (mut[None] * rng.normal(0.0, 1e-7, (nz, ny, nx))).astype(f32)
    hfx = rng.uniform(-50.0, 400.0, (ny, nx)).astype(f32)
    qfx = rng.uniform(-2e-5, 3e-4, (ny, nx)).astype(f32)
    hfx[::4, ::3] = f32(0.0)
    qfx[1::5, ::2] = f32(0.0)
    out["hfx"], out["qfx"] = hfx, qfx
    return out


def _memory(array: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    mem = np.zeros((nx + 1, nz + 1, ny + 1), np.float32)
    k, j, i = array.shape
    mem[:i, :k, :j] = np.transpose(array, (2, 0, 1))
    return mem


def write(cases: dict[str, np.ndarray], out: Path) -> None:
    nz, ny, nx = cases["t_m"].shape
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "thetam_ext_inputs.bin", "wb") as handle:
        handle.write(np.array([nx, ny, nz, 3], "<i4").tobytes())
        handle.write(np.array([P_TOP, HEAT_FLUX], "<f4").tobytes())
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
    np.savez(out / "thetam_ext_inputs.npz", **cases)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("state_npz", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    write(build(args.state_npz), args.out)


if __name__ == "__main__":
    main()
