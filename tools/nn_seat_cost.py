"""Per-step cost of an applying ``--nn`` seat on this machine's backend.

    python tools/nn_seat_cost.py MODEL [--nx 100 --ny 80] [--steps 40] [--mode replace]

Builds an idealised moist YSU case whose level count matches the artefact
(``MODEL``, an ``mlp-npz`` directory), steps it ``--steps`` times without a
seat and with one, and prints the wall time per step of each, their
difference (what the seat adds to a step) and the time of the model call
alone on the same ``[features, columns]`` array.  Each timing is closed by
a device-to-host read, so queued work is counted.  The idealised columns
lie outside a real-case artefact's trained range; the safety screen still
runs and the model still runs on every column, so the cost is the cost.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def _case(nx, ny, levels):
    import cupy as cp

    from woof.config import RunConfig, validate_run_config
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import DECLARED_CONSTANT_GLW_WM2, initialize_physics

    for nz in (levels + 1, levels):
        cfg = validate_run_config(RunConfig(
            nx=nx, ny=ny, nz=nz, dx=9000.0, dy=9000.0, ztop=20000.0, dt=45.0, run_seconds=0.0,
            moist=True, mp_physics=6, bldt=0.0, bl_pbl_physics=1, sf_sfclay_physics=91))
        coord = make_vertical_coord(cfg.nz)
        base = make_base_state(coord, lambda z: 296.0 + 0.004 * np.asarray(z), p_surf=cfg.p_surf,
                               ztop=cfg.ztop)
        state = init_moist_balanced(cfg, coord, base, lambda z: 0.018 * np.exp(-np.asarray(z) / 3500.0))
        if state.p.shape[0] >= levels:
            break
    initialize_physics(state, cfg, landmask=1.0, tsk=305.0, glw=DECLARED_CONSTANT_GLW_WM2, swdown=0.0)
    y = np.arange(ny)[None, :, None]
    x = np.arange(nx)[None, None, :]
    wave = np.sin(2 * np.pi * x / nx) * np.cos(2 * np.pi * y / ny)
    nzm = state.p.shape[0]
    state.thp += cp.asarray(np.broadcast_to(0.5 * wave, (nzm, ny, nx)).copy(), dtype=cp.float32)
    state.u[...] = cp.float32(5.0)
    return cp, cfg, state


def _per_step(cp, state, cfg, steps):
    from woof.core import dycore

    for _ in range(3):  # compile and warm
        dycore.step(state, cfg)
    cp.asnumpy(state.thp[:1, :1, :1])
    start = time.perf_counter()
    for _ in range(steps):
        dycore.step(state, cfg)
    cp.asnumpy(state.thp[:1, :1, :1])
    return (time.perf_counter() - start) / steps


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=80)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--mode", default="replace", choices=("replace", "residual"))
    parser.add_argument("--backend", default="auto")
    args = parser.parse_args(argv)

    from woof import backend
    from woof.nn.launch import NNSettings
    from woof.nn.manifest import load_manifest
    from woof.nn.wrf_seat import attach

    manifest = load_manifest(args.model)
    levels = max(f.levels for f in manifest.outputs)
    cp, cfg, plain = _case(args.nx, args.ny, levels)
    without = _per_step(cp, plain, cfg, args.steps)

    cp, cfg, state = _case(args.nx, args.ny, levels)
    settings = NNSettings(source=None, mode=args.mode, scheme="pbl", inputs=None, outputs=None, levels=None,
                          tap=None, tap_every=1, tap_columns=1, tap_seed=0, model=manifest.path.parent,
                          backend=args.backend)
    seat = attach(state, cfg, settings=settings)
    with_seat = _per_step(cp, state, cfg, args.steps)

    x, _ = seat.columns(state, cfg, state.physics.pbl_tendencies, seat.all_columns)
    for _ in range(3):
        seat.model(x)
    cp.asnumpy(seat.model(x).sum())
    start = time.perf_counter()
    total = None
    for _ in range(args.steps):
        y = seat.model(x).sum()  # a reduction over every output: a lazy runtime evaluates the whole call
        total = y if total is None else total + y
    cp.asnumpy(total)
    model_only = (time.perf_counter() - start) / args.steps
    summary = seat.close()
    nz, ny, nx = state.p.shape
    print(json.dumps({
        "backend": backend.requested(), "runner": seat.model.backend, "grid": [nz, ny, nx],
        "columns": ny * nx, "features": [manifest.n_inputs, manifest.n_outputs],
        "architecture": manifest.architecture, "steps": args.steps,
        "step_ms_without": round(without * 1e3, 3), "step_ms_with": round(with_seat * 1e3, 3),
        "seat_ms_per_step": round((with_seat - without) * 1e3, 3),
        "model_call_ms": round(model_only * 1e3, 3),
        "columns_out_of_range": summary["columns_out_of_range"], "columns_seen": summary["columns_seen"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
