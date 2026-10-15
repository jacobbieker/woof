"""woof side of the ClimaAtmos idealized benchmark.

Builds woof equivalents of ClimaAtmos's idealized configs, times a block
of dynamics steps after a warm-up (CuPy kernel compilation and allocator
warm-up excluded), optionally integrates to the end time, and writes
``result.json`` plus the final fields (``fields.npz``) for
:mod:`tools.compare.climaatmos_idealized.analysis`.

Cases
-----
``schar``
    ClimaAtmos ``plane_schar_mountain_float32_test.yml``: 100 km periodic
    x-z channel, Schaer mountain h = 25 m exp(-(x/5 km)^2) cos^2(pi x / 4 km),
    U = 10 m/s, N = 0.01 1/s, T_sfc = 288 K, p_sfc = 101325 Pa, top 21 km,
    sin^2 Rayleigh damping of w above 13 km with coefficient 0.1 1/s
    (ClimaAtmos ``alpha_rayleigh_w``/``zd_rayleigh``; woof ``damp_opt=3``
    is the same w-only sin^2 profile).  Level heights default to the
    ClimaAtmos face heights (``schar_z_faces.json``) mapped to woof's dry
    hydrostatic eta coordinate.
``dc``
    Density current, 51.2 km x 6.4 km, uniform dz.  ``--ic straka`` is the
    Straka et al. (1993) bubble (T' = -15 K at z = 3 km) with the
    published constant K = 75 m^2/s; ``--ic clima`` is ClimaAtmos's stock
    ``DryDensityCurrentProfile`` bubble (theta' = -15 K at z = 2 km).

Run on the GPU under the shared-card lock (see README.md).  ``--mode
timing`` stops after the timed block; ``--mode accuracy`` continues to
``--run-seconds`` and writes the fields.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from tools.compare.climaatmos_idealized import analysis as an

HERE = Path(__file__).resolve().parent


def _sync() -> None:
    import cupy as cp
    cp.cuda.Device().synchronize()


def build_schar(args):
    """woof RunConfig/base/state for the ClimaAtmos Schaer setup."""
    from woof.config import RunConfig
    from woof.core.dycore import set_w_surface
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest

    nx = args.nx
    dx = args.length / nx
    cfg = RunConfig(nx=nx, ny=1, nz=args.nz, dx=dx, dy=dx, ztop=args.ztop,
                    dt=args.dt, run_seconds=args.run_seconds,
                    p_surf=an.MSLP, time_step_sound=args.sound_steps,
                    damp_opt=3, zdamp=args.ztop - an.SCHAR_ZD_RAYLEIGH,
                    dampcoef=an.SCHAR_ALPHA_RAYLEIGH_W,
                    khdif=0.0, kvdif=0.0, terrain_opt=1, hill_height=25.0,
                    case="schar_clima")
    theta = an.constant_n_theta
    z_faces = _schar_z_faces(args)
    eta = an.eta_levels_for_heights(z_faces, theta, an.MSLP, args.ztop)
    coord = make_vertical_coord(cfg.nz, eta_levels=eta)
    x = an.cell_centers(nx, dx)
    h = an.schar_height(x)[None, :].copy()
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop,
                           terrain_z=h)
    state = init_at_rest(cfg, coord, base, terrain_z=h)
    state.u[...] = an.SCHAR_U
    set_w_surface(state, cfg)
    if args.w_init == "wrf-fill":
        # WRF start_em fill_w_flag decay w(k) = w(0) znw(k)^2 (as hill2d).
        state.w[1:] = state.w[0][None] * (state.znw[1:, None, None] ** 2)
    return cfg, base, state, {"z_faces_target": list(map(float, z_faces))}


def _schar_z_faces(args) -> np.ndarray:
    if args.z_faces == "uniform":
        return np.linspace(0.0, args.ztop, args.nz + 1)
    path = Path(args.z_faces) if args.z_faces else HERE / "schar_z_faces.json"
    z = np.asarray(json.loads(path.read_text())["z_faces"], dtype=np.float64)
    if z.size != args.nz + 1:
        # Resample the stretching profile to the requested level count.
        s_old = np.linspace(0.0, 1.0, z.size)
        z = np.interp(np.linspace(0.0, 1.0, args.nz + 1), s_old, z)
    z = z * (args.ztop / z[-1])
    return z


def build_dc(args):
    """woof RunConfig/base/state for the density current."""
    from woof.config import RunConfig
    from woof.core import constants as c
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_theta_perturbation

    nx = args.nx
    dx = args.length / nx
    k = args.diffusion
    cfg = RunConfig(nx=nx, ny=args.ny, nz=args.nz, dx=dx, dy=dx,
                    ztop=args.ztop, dt=args.dt, run_seconds=args.run_seconds,
                    time_step_sound=args.sound_steps,
                    khdif=k, kvdif=k, case=f"dc_{args.ic}")
    theta = an.isentropic_theta
    z_faces = np.linspace(0.0, args.ztop, args.nz + 1)
    eta = an.eta_levels_for_heights(z_faces, theta, cfg.p_surf, args.ztop)
    coord = make_vertical_coord(cfg.nz, eta_levels=eta)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    exner_b = (np.asarray(base.pb, dtype=np.float64) / c.P0) ** c.RCP

    def thp_func(x, z):
        thp = an.density_current_theta_prime(x[None, :], z[:, None],
                                             exner_b[:, None], ic=args.ic)
        return np.broadcast_to(thp[:, None, :], (cfg.nz, cfg.ny, cfg.nx))

    state = init_theta_perturbation(cfg, coord, base, thp_func)
    return cfg, base, state, {}


def run(args) -> dict:
    import cupy as cp

    from woof.core.dycore import run_steps, stability_report

    t_start = time.perf_counter()
    builder = build_schar if args.case == "schar" else build_dc
    t0 = time.perf_counter()
    cfg, base, state, extra = builder(args)
    _sync()
    setup_s = time.perf_counter() - t0

    n_total = int(round(args.run_seconds / cfg.dt))
    if args.mode == "timing":
        # Timing blocks may run past the case's end time.
        n_total = max(n_total, args.warm_steps + args.timed_steps)
    warm = min(args.warm_steps, n_total)
    timed = min(args.timed_steps, n_total - warm)
    if timed <= 0:
        raise SystemExit("run too short for the warm-up + timed block")
    t0 = time.perf_counter()
    run_steps(state, cfg, n=warm)
    _sync()
    warm_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    run_steps(state, cfg, n=timed)
    _sync()
    timed_s = time.perf_counter() - t0
    sec_per_step = timed_s / timed
    points = cfg.nx * cfg.ny * cfg.nz

    result = {
        "model": "woof",
        "case": args.case,
        "ic": getattr(args, "ic", None),
        "mode": args.mode,
        "float_type": "Float32",
        "device": "CUDA",
        "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "dt": cfg.dt,
        "time_step_sound": cfg.time_step_sound,
        "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz, "dx": cfg.dx, "ztop": cfg.ztop,
        "khdif": cfg.khdif, "kvdif": cfg.kvdif,
        "unique_points": points,
        "stored_points": points,
        "setup_seconds": setup_s,
        "warm_steps": warm,
        "warm_seconds": warm_s,
        "timed_steps": timed,
        "timed_seconds": timed_s,
        "sec_per_step": sec_per_step,
        "point_steps_per_sec": points / sec_per_step,
        "stored_point_steps_per_sec": points / sec_per_step,
        "sim_seconds_per_wall_second": cfg.dt / sec_per_step,
        "python": platform.python_version(),
        "cupy_gpu_memory_limit": os.environ.get("CUPY_GPU_MEMORY_LIMIT", ""),
    }
    result.update(extra)
    if args.case == "schar":
        result["w_init"] = args.w_init

    if args.mode == "accuracy":
        done = warm + timed
        t0 = time.perf_counter()
        run_steps(state, cfg, n=n_total - done)
        _sync()
        result["rest_wall_seconds"] = time.perf_counter() - t0
        result["t_final"] = n_total * cfg.dt
        report = stability_report(state, cfg)
        w = cp.asnumpy(state.w).astype(np.float64)[:, 0, :]
        u = cp.asnumpy(state.u).astype(np.float64)[:, 0, :]
        thp = cp.asnumpy(state.thp).astype(np.float64)[:, 0, :]
        php = cp.asnumpy(state.php).astype(np.float64)[:, 0, :]
        phb = np.asarray(base.phb, dtype=np.float64)
        phb = phb[:, 0, :] if phb.ndim == 3 else np.broadcast_to(
            phb[:, None], php.shape)
        thb = np.asarray(base.thb, dtype=np.float64)
        thb = thb[:, 0, :] if thb.ndim == 3 else np.broadcast_to(
            thb[:, None], thp.shape)
        zw = (phb + php) / 9.81
        x = an.cell_centers(cfg.nx, cfg.dx)
        np.savez_compressed(
            Path(args.out) / "fields.npz", x=x, zw=zw, w=w,
            u=0.5 * (u[:, :-1] + u[:, 1:]), theta=thb + thp, dx=cfg.dx)
        result["nan"] = bool(report["nan"] or not np.isfinite(w).all())
        result["w_max"] = float(report["w_max"])
    else:
        result["nan"] = bool(stability_report(state, cfg)["nan"])
    result["total_wall_seconds"] = time.perf_counter() - t_start
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--case", choices=["schar", "dc"], required=True)
    p.add_argument("--ic", choices=["straka", "clima"], default="straka")
    p.add_argument("--mode", choices=["accuracy", "timing"],
                   default="accuracy")
    p.add_argument("--out", required=True)
    p.add_argument("--nx", type=int)
    p.add_argument("--nz", type=int)
    p.add_argument("--ny", type=int, default=1,
                   help="dc only: y points (the bubble is uniform in y; "
                        "for 3-D throughput, not accuracy)")
    p.add_argument("--length", type=float, help="domain length in x (m)")
    p.add_argument("--ztop", type=float)
    p.add_argument("--dt", type=float)
    p.add_argument("--sound-steps", type=int, default=4)
    p.add_argument("--run-seconds", type=float)
    p.add_argument("--diffusion", type=float, default=75.0,
                   help="dc: constant K (m^2/s) for khdif and kvdif")
    p.add_argument("--z-faces", default="",
                   help="schar: JSON with z_faces, or 'uniform' (default: "
                        "the committed ClimaAtmos face heights)")
    p.add_argument("--w-init", choices=["wrf-fill", "surface"],
                   default="surface",
                   help="schar: initial w above the surface: zero with "
                        "only the kinematic surface BC (default; like "
                        "ClimaAtmos's u = U, w = 0 start), or WRF's "
                        "fill_w decay w(0) znw^2 (as verify/cases/hill2d)")
    p.add_argument("--warm-steps", type=int, default=20)
    p.add_argument("--timed-steps", type=int, default=200)
    args = p.parse_args(argv)
    defaults = {
        # Matched to ClimaAtmos: 100 elements x 3 unique GLL nodes in x,
        # 100 levels; dt is woof's own split-explicit step (see README).
        "schar": dict(nx=300, nz=100, length=100e3, ztop=21e3, dt=2.0,
                      run_seconds=86400.0),
        # 80 elements x 3 nodes, 90 levels: dx = 213.3 m, dz = 71.1 m.
        "dc": dict(nx=240, nz=90, length=51200.0, ztop=6400.0, dt=1.0,
                   run_seconds=900.0),
    }[args.case]
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    result = run(args)
    result["argv"] = sys.argv[1:] if argv is None else list(argv)
    (Path(args.out) / "result.json").write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in (
        "case", "sec_per_step", "point_steps_per_sec",
        "sim_seconds_per_wall_second", "nan")}))
    return 1 if result["nan"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
