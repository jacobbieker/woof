"""Idealised grass fire after WRF's test/em_fire (LES, no PBL): the E2E.

A flat, doubly open box under the em_fire hill_simple sounding (288 K at
the ground, +0.7 K per 100 m to 2 km, 9 g/kg vapour falling with height,
a uniform 3 m/s westerly), km_opt = 2 (1.5-order TKE), diff_opt = 2, no
PBL, no surface layer, no microphysics -- em_fire's physics -- with WRF's
coupled fire (ifire = 2) on a 4 x 4 fire mesh in short grass (Anderson 1),
lit along a north-south line, and the smoke tracer carrying the fire's
PM2.5.  Reports, every ``--sample-s``:

* the burned area (fire-mesh cells with lfn <= 0) and its growth;
* the peak fire heat flux on the fire mesh and the atmospheric grid;
* the strongest updraft above the fire (w max over the burning columns);
* the smoke: its column-integrated mass, its peak, and the height of its
  centre of mass over the fire, which rises with the plume;

and the per-step cost with the fire on against the same box with it off.

Run::

    pixi run python tools/fire_grass_e2e.py --minutes 20 --output grass.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: test/em_fire/input_sounding_hill_simple: height (m), theta (K), qv (g/kg)
SOUNDING = np.array([
    (10.0, 288.00, 9.0), (100.0, 288.70, 8.84), (200.0, 289.40, 8.68),
    (300.0, 290.10, 8.52), (400.0, 290.80, 8.36), (500.0, 291.50, 8.20),
    (600.0, 292.20, 8.04), (700.0, 292.90, 7.88), (800.0, 293.60, 7.72),
    (900.0, 294.30, 7.56), (1000.0, 295.00, 7.40), (1100.0, 295.75, 7.21),
    (1200.0, 296.50, 7.02), (1300.0, 297.25, 6.83), (1400.0, 298.00, 6.64),
    (1500.0, 298.75, 6.45), (1600.0, 299.50, 6.26), (1700.0, 300.25, 6.07),
    (1800.0, 301.00, 5.88), (1900.0, 301.75, 5.69), (2000.0, 302.50, 5.5),
    (3000.0, 308.79, 4.00)])
U0 = 3.0


def theta_of(z):
    return np.interp(np.asarray(z, float), SOUNDING[:, 0], SOUNDING[:, 1])


def qv_of(z):
    return 1.0e-3 * np.interp(np.asarray(z, float), SOUNDING[:, 0],
                              SOUNDING[:, 2])


def make_config(*, fire: bool, minutes: float, nx=80, ny=60, nz=40,
                dx=50.0, ztop=2000.0, dt=0.5, smoke=True):
    from woof.config import RunConfig, validate_run_config

    fire_keys = {}
    if fire:
        fire_keys = dict(
            ifire=2, sr_x=4, sr_y=4, fire_fuel_read=0, fire_fuel_cat=1,
            fire_wind_height=6.096,
            fire_ignitions=((1000.0, 1200.0, 1000.0, 1800.0, 0.0, 0.0, 0.0,
                             0.0, 25.0, 0.5, 30.0, 40.0),))
    cfg = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, ztop=ztop, dt=dt,
        run_seconds=minutes * 60.0, km_opt=2, bl_pbl_physics=0,
        sf_sfclay_physics=0, moist=True, mp_physics=0, isfflx=0,
        tke_heat_flux=0.0, tke_drag_coefficient=0.0, time_step_sound=6,
        open_x=True, open_y=True, p_surf=101710.0,
        smoke_tracer=bool(smoke and fire), **fire_keys)
    return validate_run_config(cfg)


def build(cfg):
    import cupy as cp

    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced

    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, theta_of, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, qv_of)
    state.u[...] = cp.float32(U0)
    return state


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def sample(state, cfg, t):
    """One sampling record (device-to-host reads only here)."""
    import cupy as cp

    rec = {"t_s": t}
    if not getattr(cfg, "ifire", 0):
        return rec
    mesh = state._fire.mesh
    lfn = _host(state.fire_lfn)[1:-1, 1:-1]
    rec["burned_area_m2"] = float(np.count_nonzero(lfn <= 0)) * float(
        mesh.dx) * float(mesh.dy)
    rec["fgrnhfx_max_W_m2"] = float(cp.max(state.fire_fgrnhfx))
    rec["grnhfx_max_W_m2"] = float(cp.max(state.fire_grnhfx))
    rec["fire_area_m2"] = float(cp.sum(state.fire_area))
    rec["fire_heat_W"] = float(cp.sum(state.fire_heat_flux)) * cfg.dx * cfg.dy
    burning = _host(state.fire_grnhfx) > 0
    w = _host(state.w)
    rec["w_max_m_s"] = float(w.max())
    if burning.any():
        rec["w_max_over_fire_m_s"] = float(w[:, burning].max())
        # The plume leans downwind: the updraft over the burning columns
        # and the 10 cells (500 m) east of them, two rows either side.
        plume = burning.copy()
        for di in range(1, 11):
            plume[:, di:] |= burning[:, :-di]
        for dj in (1, 2):
            plume[dj:] |= plume[:-dj].copy()
            plume[:-dj] |= plume[dj:].copy()
        rec["w_max_plume_m_s"] = float(w[:, plume].max())
        kmax = int(np.unravel_index(np.argmax(np.where(
            plume[None], w, -np.inf)), w.shape)[0])
        rec["w_max_plume_level"] = kmax
        cols = np.argwhere(burning)
        jc, ic = cols.mean(axis=0).round().astype(int)
        rec["fire_centre_ij"] = [int(jc), int(ic)]
    if getattr(state, "smoke", None) is not None:
        smoke = _host(state.smoke)
        z = _host(state.height_half()) if callable(
            getattr(state, "height_half", None)) else None
        col = _host(state.smoke_col)
        rec["smoke_column_total_kg"] = float(col.sum()) * cfg.dx * cfg.dy \
            * 1e-9
        rec["smoke_max_ug_kg"] = float(smoke.max())
        if z is not None and smoke.sum() > 0:
            zz = z if z.ndim == 3 else z[:, None, None] * np.ones_like(smoke)
            rec["smoke_centroid_height_m"] = float((smoke * zz).sum()
                                                   / smoke.sum())
            top = smoke > 1e-3 * smoke.max()
            rec["smoke_top_height_m"] = float(zz[top].max())
    return rec


def run(cfg, *, sample_s=60.0, progress=True):
    import cupy as cp

    from woof.core.dycore import step

    state = build(cfg)
    n_total = int(round(cfg.run_seconds / cfg.dt))
    every = max(int(round(sample_s / cfg.dt)), 1)
    records = []
    # warm-up step compiles every kernel outside the timed window
    step(state, cfg)
    cp.cuda.get_current_stream().synchronize()
    done = 1
    wall = 0.0
    while done < n_total:
        n = min(every - (done % every) or every, n_total - done)
        cp.cuda.get_current_stream().synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            step(state, cfg)
        cp.cuda.get_current_stream().synchronize()
        wall += time.perf_counter() - t0
        done += n
        rec = sample(state, cfg, done * cfg.dt)
        records.append(rec)
        if progress:
            print(json.dumps(rec), flush=True)
    return state, records, wall / max(n_total - 1, 1)


def acquire_gpu_lock(run_id: str, *, poll_s: float = 20.0):
    """woof's UUID-keyed GPU lock on device 0, waiting while it is held."""
    import subprocess

    from woof.supervisor import GPUAlreadyLockedError, GPUFileLock

    uuid = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i",
         "0"], capture_output=True, text=True, check=True).stdout.strip()
    while True:
        try:
            return GPUFileLock(uuid, run_id=run_id).acquire()
        except GPUAlreadyLockedError as exc:
            print(f"[{run_id}] {exc}; retrying in {poll_s:.0f} s",
                  file=sys.stderr, flush=True)
            time.sleep(poll_s)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--sample-s", type=float, default=60.0)
    parser.add_argument("--off-minutes", type=float, default=2.0,
                        help="length of the fire-off timing run")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    lock = acquire_gpu_lock("fire_grass_e2e")
    try:
        cfg = make_config(fire=True, minutes=args.minutes)
        _, records, per_step_on = run(cfg, sample_s=args.sample_s)
        off = make_config(fire=False, minutes=args.off_minutes)
        _, _, per_step_off = run(off, sample_s=args.sample_s,
                                 progress=False)
    finally:
        lock.release()
    result = {"config": {"nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz,
                         "dx_m": cfg.dx, "dt_s": cfg.dt, "sr": 4,
                         "minutes": args.minutes},
              "samples": records,
              "per_step_s": {"fire_on": per_step_on,
                             "fire_off": per_step_off}}
    print(json.dumps(result["per_step_s"]))
    if args.output:
        args.output.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
