"""woof side of the Breeze dry-CBL comparison (the FastEddy SWiFT case).

Two modes, one case (``case.py``):

``timing``    build the case, run ``--warmup`` steps (kernel compilation,
              first-touch allocation: excluded), then ``--reps`` timed blocks
              of ``--steps`` steps, each bracketed by a device synchronize.
              No profile sampling, no output inside the timed blocks.  Writes
              one JSON with per-rep seconds/step and the median.
``simulate``  integrate ``--hours`` of model time, sampling slab-mean
              profiles every ``--sample-every`` seconds, and write the common
              profile npz + receipt JSON that ``analyze.py`` reads (the same
              schema ``run_breeze.jl`` writes).

What is pinned here
-------------------
* Closure: WRF ``km_opt=3`` (3-D Smagorinsky, ``c_s = 0.18``) or
  ``km_opt=2`` (1.5-order TKE, ``c_k = 0.10``), ``mix_isotropic = 1``,
  ``diff_opt = 2``, PBL off -- exactly the em_les configuration
  ``woof.verify.cases.convective_boundary_layer.make_config`` builds; only
  the forcing, the grid and the sound-step count are changed here.
* Advection: woof's (and WRF's) defaults -- 5th-order upwind horizontal,
  3rd-order upwind vertical, for momentum and scalars alike.
* Acoustics: ``time_step_sound = --substeps`` (WRF distribution: 1 substep
  of dt/3 in RK stage 1, ns/2 and ns substeps of dt/ns in stages 2 and 3),
  ``epssm = 0.1``, ``smdiv = 0.1`` (woof defaults).
* Vertical grid: an eta ladder computed from the case sounding so the
  base-state layers are UNIFORM IN HEIGHT (dz = Lz/nz), matching Breeze's
  rectilinear z grid; a plain uniform-eta ladder would thicken layers by
  ~30% over the 3 km column.
* Surface: ``isfflx = 0`` -- constant kinematic heat flux 0.35 K m/s, bulk
  drag ``Cd = case.drag_coefficient(nz)``.
* Rotation: woof's WRF Coriolis kernel on the full wind (f-plane, e = 0)
  plus the geostrophic pressure-gradient force ``(-f Vg, +f Ug)``, which
  woof has no namelist knob for: the harness adds it to the velocity once
  per full step (operator split; exact for a constant forcing).  It is one
  elementwise CuPy add per step and is INSIDE the timed loop.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.compare.breeze_cbl import case  # noqa: E402


def uniform_height_eta(nz: int, ztop: float, p_surf: float) -> np.ndarray:
    """Eta interfaces whose base-state heights are k * ztop / nz.

    Integrates the Exner function on the same 1 m fine grid
    ``woof.core.grid.make_base_state`` uses, so ``p(z)`` is the base
    state's own, then maps each target height to dry-pressure eta.
    """
    from woof.core import constants as c
    zf = np.append(np.arange(0.0, ztop, 1.0), ztop)
    dz = np.diff(zf)
    z_mid = 0.5 * (zf[:-1] + zf[1:])
    pi_surf = (p_surf / c.P0) ** c.RCP
    pi = np.concatenate(([pi_surf], pi_surf - np.cumsum(
        c.G * dz / (c.CP * case.theta_profile(z_mid)))))
    p_of_z = c.P0 * pi ** (1.0 / c.RCP)
    z_target = np.linspace(0.0, ztop, nz + 1)
    p_target = np.interp(z_target, zf, p_of_z)
    eta = (p_target - p_of_z[-1]) / (p_of_z[0] - p_of_z[-1])
    eta[0], eta[-1] = 1.0, 0.0
    return eta


def make_cfg(nx: int, ny: int, nz: int, *, km_opt: int, substeps: int,
             dt: float = case.DT, hours: float = 2.0):
    from woof.verify.cases import convective_boundary_layer as cbl
    dx, dy, _ = case.grid_spacing(nx, ny, nz)
    if abs(dx - dy) > 1e-9:
        raise ValueError("the case is square: nx must equal ny")
    cfg = cbl.make_config(
        nx=nx, ny=ny, nz=nz, dx=dx, ztop=case.LZ, dt=dt,
        minutes=hours * 60.0, heat_flux=case.HEAT_FLUX,
        drag_coefficient=case.drag_coefficient(nz), c_s=case.C_S,
        km_opt=km_opt, c_k=case.C_K, surface="prescribed",
        lateral="periodic")
    return replace(cfg, p_surf=case.P_SURFACE, time_step_sound=substeps)


def build_state(cfg, seed: int):
    """Case state: sounding, uniform-z ladder, perturbations, wind, f."""
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_theta_perturbation

    coord = make_vertical_coord(cfg.nz, eta_levels=uniform_height_eta(
        cfg.nz, cfg.ztop, cfg.p_surf))
    base = make_base_state(coord, case.theta_profile, p_surf=cfg.p_surf,
                           ztop=cfg.ztop)

    def thp_func(x, z):
        rng = np.random.default_rng(seed)
        z = np.asarray(z, float)
        pert = rng.uniform(-case.PERT_AMPLITUDE, case.PERT_AMPLITUDE,
                           (cfg.nz, cfg.ny, cfg.nx))
        pert[z >= case.PERT_DEPTH] = 0.0
        return pert

    state = init_theta_perturbation(cfg, coord, base, thp_func)
    state.u[...] = case.UG
    state.v[...] = case.VG
    state.set_map_coriolis(f=np.full((cfg.ny, cfg.nx), case.coriolis_f()))
    return state


def geostrophic_forcing(cfg):
    """Return ``force(state)``: one step of the geostrophic PGF."""
    f = case.coriolis_f()
    du = np.float32(-f * case.VG * cfg.dt)
    dv = np.float32(f * case.UG * cfg.dt)

    def force(state):
        if du != 0.0:
            state.u += du
        if dv != 0.0:
            state.v += dv
    return force


def advance(state, cfg, n: int, force) -> None:
    from woof.core.dycore import step
    for _ in range(n):
        step(state, cfg)
        force(state)


def _sync():
    import cupy as cp
    cp.cuda.Device().synchronize()


def _env() -> dict:
    import cupy as cp
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props["name"]
    return {
        "gpu": name.decode() if isinstance(name, bytes) else str(name),
        "cupy": cp.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "cupy_gpu_memory_limit": os.environ.get("CUPY_GPU_MEMORY_LIMIT"),
    }


def _meta(cfg, args, label: str) -> dict:
    return {
        "model": "woof",
        "label": label,
        "dynamics": "compressible_splitexplicit",
        "advection": "WRF upwind h5/v3 (momentum and scalars)",
        "closure": {2: "km_opt=2 1.5-order TKE (c_k=0.10, Pr=1/3)",
                    3: "km_opt=3 3-D Smagorinsky (c_s=0.18, Pr=1/3)"}[
                        cfg.km_opt],
        "km_opt": cfg.km_opt,
        "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz,
        "dx": cfg.dx, "dz": case.LZ / cfg.nz, "dt": cfg.dt,
        "substeps": cfg.time_step_sound,
        "substep_distribution": "wrf (1, ns/2, ns)",
        "acoustic_steps_per_dt": 1 + cfg.time_step_sound // 2
        + cfg.time_step_sound,
        "epssm": cfg.epssm, "smdiv": cfg.smdiv,
        "drag": "bulk", "drag_coefficient": cfg.tke_drag_coefficient,
        "heat_flux": cfg.tke_heat_flux, "f": case.coriolis_f(),
        "ug": case.UG, "vg": case.VG, "p_surf": cfg.p_surf,
        "float_type": "Float32", "seed": args.seed,
        **_env(),
    }


def _profiles(state, cfg) -> dict:
    """Slab-mean profiles in the common schema (see analyze.py)."""
    import cupy as cp

    from woof.core import constants as c
    from woof.verify.cases.convective_boundary_layer import _slab_profiles
    p = _slab_profiles(state, cfg)
    phb = state.phb
    ph = ((phb if phb.ndim == 3 else phb[:, None, None]) + state.php)
    z_f = cp.asnumpy(ph.astype(cp.float64).mean(axis=(1, 2))) / c.G
    w = state.w.astype(cp.float64)
    wbar = w.mean(axis=(1, 2))
    ww = cp.asnumpy(((w - wbar[:, None, None]) ** 2).mean(axis=(1, 2)))
    u = cp.asnumpy(state.u[:, :, :cfg.nx].astype(cp.float64).mean(
        axis=(1, 2)))
    v = cp.asnumpy(state.v[:, :cfg.ny, :].astype(cp.float64).mean(
        axis=(1, 2)))
    return {"z_f": z_f, "theta": p["theta"], "u": u, "v": v, "ww": ww,
            "wth_res": p["wth_res"], "wth_sgs": p["wth_sgs"],
            "tke_res": p["tke_res"], "e_sgs": p["e_sgs"]}


def run_timing(args) -> dict:
    cfg = make_cfg(args.nx, args.ny, args.nz, km_opt=args.km_opt,
                   substeps=args.substeps, dt=args.dt)
    force = geostrophic_forcing(cfg)
    t0 = time.perf_counter()
    state = build_state(cfg, args.seed)
    _sync()
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    advance(state, cfg, args.warmup, force)
    _sync()
    warmup_s = time.perf_counter() - t0
    per_step = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        advance(state, cfg, args.steps, force)
        _sync()
        per_step.append((time.perf_counter() - t0) / args.steps)
    from woof.core.dycore import stability_report
    rep = stability_report(state, cfg)
    med = statistics.median(per_step)
    npts = cfg.nx * cfg.ny * cfg.nz
    out = {
        **_meta(cfg, args, args.label or f"woof_km{args.km_opt}"),
        "mode": "timing", "warmup_steps": args.warmup,
        "steps_per_rep": args.steps, "reps": args.reps,
        "build_seconds": build_s, "warmup_seconds": warmup_s,
        "seconds_per_step": per_step,
        "median_seconds_per_step": med,
        "mpts_steps_per_second": npts / med / 1e6,
        "simulated_seconds_per_wall_second": cfg.dt / med,
        "finite": not bool(rep["nan"]), "w_max": float(rep["w_max"]),
        **_memory(),
    }
    return out


def _memory() -> dict:
    """Live bytes (comparable with Breeze's CUDA.used_memory) and the
    pool's reservation, which is what the card actually gives up."""
    import cupy as cp
    pool = cp.get_default_memory_pool()
    return {"memory_gib": pool.used_bytes() / 2.0**30,
            "memory_reserved_gib": pool.total_bytes() / 2.0**30,
            "memory_counter": "CuPy default pool used_bytes (live arrays)"}


def run_simulate(args) -> tuple[dict, dict]:
    from woof.core.dycore import stability_report
    cfg = make_cfg(args.nx, args.ny, args.nz, km_opt=args.km_opt,
                   substeps=args.substeps, dt=args.dt, hours=args.hours)
    force = geostrophic_forcing(cfg)
    state = build_state(cfg, args.seed)
    n_total = int(round(args.hours * 3600.0 / cfg.dt))
    n_chunk = max(int(round(args.sample_every / cfg.dt)), 1)
    samples = [dict(_profiles(state, cfg), t=0.0)]
    done = 0
    step_wall = 0.0
    w_max = 0.0
    finite = True
    while done < n_total:
        n = min(n_chunk, n_total - done)
        _sync()
        t0 = time.perf_counter()
        advance(state, cfg, n, force)
        _sync()
        step_wall += time.perf_counter() - t0
        done += n
        rep = stability_report(state, cfg)
        if rep["nan"]:
            finite = False
            w_max = float("nan")     # as Julia's max: a blow-up reads NaN
            print(f"non-finite state at t={done * cfg.dt:.0f}s", flush=True)
            break
        w_max = max(w_max, float(rep["w_max"]))
        samples.append(dict(_profiles(state, cfg), t=done * cfg.dt))
        print(f"  woof t={done * cfg.dt:7.0f}s w_max={rep['w_max']:6.2f} "
              f"wall={step_wall:7.1f}s", flush=True)
    arrays = {k: np.stack([s[k] for s in samples])
              for k in samples[0] if k != "t"}
    arrays["t_seconds"] = np.array([s["t"] for s in samples])
    meta = {
        **_meta(cfg, args, args.label or f"woof_km{args.km_opt}"),
        "mode": "simulate", "hours": args.hours,
        "sample_every_s": args.sample_every, "steps": done,
        "stepping_wall_seconds": step_wall,
        "note_wall": ("stepping wall time includes the first steps' kernel "
                      "compilation; use timing mode for throughput"),
        "finite": finite, "w_max": w_max, **_memory(),
    }
    return meta, arrays


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("timing", "simulate"),
                    required=True)
    ap.add_argument("--size", default="128x128x64",
                    help="NXxNYxNZ over the fixed 12x12x3 km box")
    ap.add_argument("--km-opt", type=int, choices=(2, 3), default=3)
    ap.add_argument("--substeps", type=int, default=12,
                    help="time_step_sound (acoustic substeps per dt)")
    ap.add_argument("--dt", type=float, default=case.DT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--sample-every", type=float, default=300.0)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", type=Path, required=True,
                    help="output path stem (writes STEM.json [+ STEM.npz])")
    args = ap.parse_args(argv)
    args.nx, args.ny, args.nz = (int(s) for s in args.size.lower().split("x"))
    if args.substeps % 2:
        ap.error("--substeps must be even (WRF's stage-2 count is ns/2)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "timing":
        out = run_timing(args)
        Path(str(args.out) + ".json").write_text(
            json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({k: out[k] for k in (
            "label", "nx", "ny", "nz", "median_seconds_per_step",
            "mpts_steps_per_second", "finite")}))
        return 0 if out["finite"] else 1
    meta, arrays = run_simulate(args)
    np.savez_compressed(str(args.out) + ".npz", **arrays)
    Path(str(args.out) + ".json").write_text(
        json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return 0 if meta["finite"] else 1


if __name__ == "__main__":
    sys.exit(main())
