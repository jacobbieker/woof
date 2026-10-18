"""Throughput and memory of a 100 m, 256-level LES + RTE-RRTMGP step.

The calibration behind ``woof estimate-tracks`` (woof/track_estimate.py)
and the A6000 row of docs/energy-hindcast/cost.md.  The moving-nest leaf
it prices is 100 m with 256 levels, km_opt=2 TKE closure under diff_opt=2,
the PBL off, Thompson, RTE-RRTMGP every 5 minutes and dt 0.5 s; nothing
timed in this tree before was within a factor of five of that column
count.

Workload: ``tilestream.harness.make_config`` on a periodic, flat domain,
built by ``tilestream.physics_inventory.default_builder`` (the WK82
sounding, seeded qv/qc, a PhysicsDriver attached) -- the same builder
``tools/profile_launch_overhead.py`` times.  Pinned limitations: no
lateral-boundary or nest-interpolation kernels, flat terrain, a synthetic
sounding.  The step is ``dycore.step`` plus the resident loop's per-step
``stability_report`` readback.

``--steps`` leaf steps are timed with radiation on its own cadence (forcing
it off for longer than ``radt`` trips the stale-carrier contract), then
``--radiation-steps`` with radiation forced due (``radiation_due_override``,
the adaptive clock's own hook).  The ordinary step is the cadence run less
its radiation steps, and the mean step is composed from the two at the
requested ``radt``, so a short run is not decided by whether a radiation
step fell inside it.  Setup, kernel compilation and warm-up are excluded.

``--tree`` adds a d01 and a d02 to the same process (independent periodic
domains, no nest coupling) and steps them 1 : 5 : 15 against the leaf, so
the memory a 3-domain tree holds on one card and its per-d01-step wall time
can be read off.  Each case runs in its own child process, so an
out-of-memory case is recorded rather than fatal and every peak starts from
an empty card.

The GPU must be idle: utilisation under ``--max-util`` and no other compute
process above ``--max-other-mib``.  ``--wait`` polls every ``--poll-s``
until it is, up to ``--wait-s``.  The woof UUID-keyed GPU lock
(woof.supervisor.GPUFileLock) is held for the whole measurement, so a woof
run cannot start on the card underneath it.

Run::

    pixi run python tools/bench_les_256.py --sizes 300,325 --steps 600 \\
        --wait --output bench.json
    pixi run python tools/bench_les_256.py --tree --sizes 325,400 \\
        --steps 30 --output tree.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA = "woof-bench-les-256-v1"

_MOIST = dict(moist=True, mp_physics=8, ra_sw_physics=4, ra_lw_physics=4,
              ra_rrtmg_variant="rte-rrtmgp", sf_surface_physics=2,
              cu_physics=0, time_step_sound=4)
#: The leaf (and d02): LES, PBL off, revised MM5 surface layer.
LES = dict(_MOIST, bl_pbl_physics=0, sf_sfclay_physics=1, km_opt=2,
           diff_opt=2)
#: d01 at 1.5 km: MYNN surface layer + PBL, the km_opt 4 horizontal mixing.
MESO = dict(_MOIST, bl_pbl_physics=5, sf_sfclay_physics=5, km_opt=4,
            diff_opt=2)

#: d01 and d02 of the priced tree.  d02 is the 241 x 241 x 256 300 m
#: domain (14.9 M cells); d01 is a typical 1.5 km envelope (~5 M cells).
TREE = {"d01": dict(nx=140, ny=140, dx=1500.0, dt=7.5, suite=MESO),
        "d02": dict(nx=241, ny=241, dx=300.0, dt=1.5, suite=LES)}
#: Leaf steps per d02 step and d02 steps per d01 step (dt 7.5/1.5/0.5).
RATIOS = {"d01": 1, "d02": 5, "d03": 15}


# --------------------------------------------------------------------------
# the card
# --------------------------------------------------------------------------

def _smi(query: str, kind: str = "gpu") -> list[list[str]]:
    flag = "--query-gpu" if kind == "gpu" else "--query-compute-apps"
    out = subprocess.run(
        ["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True).stdout
    return [[c.strip() for c in line.split(",")]
            for line in out.splitlines() if line.strip()]


def card_conditions(index: int = 0) -> dict:
    """What nvidia-smi says about the card and who else is on it."""
    row = _smi("name,uuid,utilization.gpu,memory.used,memory.total,"
               "temperature.gpu,clocks.sm,clocks.mem,power.draw,"
               "driver_version")[index]
    names = ("name", "uuid", "utilization_pct", "memory_used_mib",
             "memory_total_mib", "temperature_c", "clock_sm_mhz",
             "clock_mem_mhz", "power_w", "driver")
    out = dict(zip(names, row))
    for key in names[2:-1]:
        try:
            out[key] = float(out[key])
        except ValueError:
            pass
    apps = []
    for pid, uuid, used in _smi("pid,gpu_uuid,used_memory", "apps"):
        if uuid == out["uuid"]:
            apps.append({"pid": int(pid), "used_mib": float(used or 0)})
    out["compute_apps"] = apps
    out["at_unix"] = time.time()
    return out


def idle(cond: dict, max_util: float, max_other_mib: float,
         own: set[int] = frozenset()) -> tuple[bool, str]:
    other = [a for a in cond["compute_apps"] if a["pid"] not in own]
    other_mib = sum(a["used_mib"] for a in other)
    if cond["utilization_pct"] >= max_util:
        return False, f"utilisation {cond['utilization_pct']:.0f}%"
    if other_mib > max_other_mib:
        return False, (f"{len(other)} other compute process(es) hold "
                       f"{other_mib:.0f} MiB")
    return True, "idle"


def wait_for_idle(args, index: int) -> dict:
    deadline = time.monotonic() + (args.wait_s if args.wait else 0.0)
    while True:
        # Two looks a few seconds apart: a short test between kernels reads
        # 0% on one sample.
        a = card_conditions(index)
        time.sleep(3.0)
        b = card_conditions(index)
        ok_a, why_a = idle(a, args.max_util, args.max_other_mib)
        ok_b, why_b = idle(b, args.max_util, args.max_other_mib)
        if ok_a and ok_b:
            return b
        why = why_a if not ok_a else why_b
        if time.monotonic() >= deadline:
            raise SystemExit(f"GPU not idle ({why}); "
                             + ("gave up waiting" if args.wait
                                else "pass --wait to poll"))
        print(f"[bench] GPU busy ({why}); retrying in {args.poll_s:.0f} s",
              file=sys.stderr, flush=True)
        time.sleep(args.poll_s)


# --------------------------------------------------------------------------
# one case (child process)
# --------------------------------------------------------------------------

def _config(nx, ny, nz, *, dx, dt, suite, ztop, radt_minutes):
    from tilestream import harness

    return harness.make_config(nx, ny, nz, dx=dx, dy=dx, dt=dt, ztop=ztop,
                               radt_minutes=radt_minutes, **suite)


def _own_mib(pid: int, uuid: str) -> float | None:
    try:
        for p, u, used in _smi("pid,gpu_uuid,used_memory", "apps"):
            if int(p) == pid and u == uuid:
                return float(used)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None
    return None


def _quiet(state) -> None:
    """Drop the builder's geopotential, mass and tendency noise.

    Its amplitudes were chosen for ~400 m layers.  On 256 levels the lowest
    layers are a few tens of metres thick, the same 20 m2 s-2 of php noise
    is a 60 K temperature error there, and RRTMGP refuses the column
    (tlay outside [160, 355] K).  The theta, wind and moisture noise stay.
    """
    from tilestream import harness
    from woof.core.diagnostics import update_diagnostics

    for name in ("php", "mup"):
        getattr(state, name)[...] = 0.0
    for name, *_ in harness._SEED_TENDENCIES + harness._SEED_ACOUSTIC:
        getattr(state, name)[...] = 0.0
    update_diagnostics(state)


def run_case(spec: dict) -> dict:
    """Build every domain of ``spec``, warm up, time; return one row."""
    import cupy as cp

    from tilestream import physics_inventory as physinv
    from woof.core import dycore

    pool = cp.get_default_memory_pool()
    uuid = spec["uuid"]
    free0, total = cp.cuda.runtime.memGetInfo()
    domains = {}
    t0 = time.perf_counter()
    for name, d in spec["domains"].items():
        cfg = _config(d["nx"], d["ny"], spec["nz"], dx=d["dx"], dt=d["dt"],
                      suite=d["suite"], ztop=spec["ztop"],
                      radt_minutes=spec["radt_minutes"])
        state, driver = physinv.default_builder(cfg, spec["seed"])
        _quiet(state)
        domains[name] = (cfg, state, driver)
    cp.cuda.runtime.deviceSynchronize()
    setup_s = time.perf_counter() - t0
    built_pool = pool.total_bytes()

    def one(name, radiation=None):
        # None leaves radiation on its own cadence: forcing it off for
        # longer than radt trips the stale-carrier contract
        # (woof.core.radiation_carriers), as a real run never would.
        cfg, state, driver = domains[name]
        driver.radiation_due_override = radiation
        dycore.step(state, cfg)
        rep = dycore.stability_report(state, cfg)
        if rep["nan"]:
            raise FloatingPointError(f"{name}: non-finite state")

    t0 = time.perf_counter()
    for name in domains:          # compile both step kinds on every domain
        one(name, True)
        for _ in range(spec["warmup"]):
            one(name, False)
        one(name)
    cp.cuda.runtime.deviceSynchronize()
    warmup_s = time.perf_counter() - t0

    leaf = "d03"
    ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()

    def timed(fn, n):
        """``(wall, event, per-step wall)``; the per-step health readback
        synchronises every step, so per-step wall is device time."""
        per = []
        cp.cuda.runtime.deviceSynchronize()
        ev0.record()
        w0 = time.perf_counter()
        for _ in range(n):
            s0 = time.perf_counter()
            fn()
            per.append(time.perf_counter() - s0)
        ev1.record()
        ev1.synchronize()
        return (time.perf_counter() - w0,
                cp.cuda.get_elapsed_time(ev0, ev1) / 1e3, per)

    row = {"domains": {}, "setup_s": setup_s, "warmup_s": warmup_s}
    if spec["tree"]:
        # One d01 step and the d02/d03 steps under it, radiation on cadence.
        # A short window right after the forced-radiation warm-up holds no
        # radiation step on any domain, so this times ORDINARY steps; the
        # calls counted below say whether that held.
        def tree_step():
            for name in ("d01", "d02", "d03"):
                for _ in range(RATIOS[name]):
                    one(name)
        before = {n: d[2].call_counts["radiation"]
                  for n, d in domains.items()}
        n = spec["tree_steps"]
        wall, event, _ = timed(tree_step, n)
        row["tree"] = {"d01_steps": n, "wall_s": wall, "event_s": event,
                       "s_per_d01_step": wall / n,
                       "radiation_calls": {
                           k: domains[k][2].call_counts["radiation"] - v
                           for k, v in before.items()}}
    n_cad, n_rad = spec["steps"], spec["radiation_steps"]
    driver = domains[leaf][2]
    calls0 = driver.call_counts["radiation"]
    took_radiation = []

    def cadence_step():
        before = driver.call_counts["radiation"]
        one(leaf)
        took_radiation.append(driver.call_counts["radiation"] != before)

    wall_c, event_c, per = timed(cadence_step, n_cad)
    rad_in_cadence = driver.call_counts["radiation"] - calls0
    wall_r, event_r, _ = (timed(lambda: one(leaf, True), n_rad)
                          if n_rad else (0.0, 0.0, []))
    one(leaf)
    cfg = domains[leaf][0]
    cells = cfg.nx * cfg.ny * cfg.nz
    # Ordinary steps: the cadence run less the steps that called radiation.
    plain = [s for s, rad in zip(per, took_radiation) if not rad] or per
    s_ord = sum(plain) / len(plain)
    s_rad = wall_r / n_rad if n_rad else s_ord
    every = max(1, round(spec["radt_minutes"] * 60.0 / cfg.dt))
    s_mean = s_ord + (s_rad - s_ord) / every
    peak_pool = pool.total_bytes()
    free1, _ = cp.cuda.runtime.memGetInfo()
    own = _own_mib(os.getpid(), uuid)
    all_cells = sum(c.nx * c.ny * c.nz for c, _, _ in domains.values())
    row["domains"] = {
        n: {"nx": c.nx, "ny": c.ny, "nz": c.nz, "dx": c.dx, "dt": c.dt,
            "cells": c.nx * c.ny * c.nz} for n, (c, _, _) in domains.items()}
    row.update({
        "leaf_cells": cells, "total_cells": all_cells,
        "cadence": {"steps": n_cad, "wall_s": wall_c, "event_s": event_c,
                    "radiation_calls": rad_in_cadence,
                    "s_per_step": wall_c / n_cad,
                    "s_per_step_median": sorted(per)[n_cad // 2],
                    "s_per_step_max": max(per)},
        "ordinary": {"steps": len(plain), "s_per_step": s_ord},
        "radiation": {"steps": n_rad, "wall_s": wall_r, "event_s": event_r,
                      "s_per_step": s_rad},
        "radiation_every_steps": every,
        "s_per_step_mean": s_mean,
        "cell_steps_per_s": cells / s_mean,
        "cell_steps_per_s_ordinary": cells / s_ord,
        "pool_built_gib": built_pool / 2**30,
        # The CuPy pool keeps every block it allocated (nothing here frees
        # it), so these end-of-case readings are the high-water marks.
        "pool_peak_gib": peak_pool / 2**30,
        "device_used_delta_gib": (free0 - free1) / 2**30,
        "process_mib_nvidia_smi": own,
        "process_gib_nvidia_smi": None if own is None else own / 1024.0,
        "bytes_per_cell_pool": peak_pool / all_cells,
        "device_total_gib": total / 2**30,
    })
    return row


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _child(spec_json: str) -> int:
    spec = json.loads(spec_json)
    try:
        row = {"ok": True, **run_case(spec)}
    except Exception as exc:  # recorded, not raised: OOM is a result here
        kind = type(exc).__name__
        row = {"ok": False, "error": f"{kind}: {exc}"[:2000],
               "oom": "OutOfMemory" in kind or "out of memory" in str(exc)}
    print("BENCH-ROW " + json.dumps(row), flush=True)
    return 0


def _spec(args, size: int, uuid: str) -> dict:
    domains = {}
    if args.tree:
        for name, d in TREE.items():
            domains[name] = dict(d)
        domains["d01"]["nx"] = domains["d01"]["ny"] = args.d01_n
    domains["d03"] = dict(nx=size, ny=size, dx=args.dx, dt=args.dt,
                          suite=LES)
    return {"domains": domains, "nz": args.nz, "ztop": args.ztop,
            "radt_minutes": args.radt_minutes, "seed": args.seed,
            "warmup": args.warmup, "steps": args.steps,
            "radiation_steps": args.radiation_steps, "tree": args.tree,
            "tree_steps": args.tree_steps, "uuid": uuid}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", default="300,325",
                   help="comma list of leaf sides (columns per side)")
    p.add_argument("--nz", type=int, default=256,
                   help="mass levels (256 = 257 eta levels)")
    p.add_argument("--ztop", type=float, default=25000.0)
    p.add_argument("--dx", type=float, default=100.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--radt-minutes", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=600,
                   help="timed leaf steps, radiation on its own cadence")
    p.add_argument("--radiation-steps", type=int, default=3,
                   help="timed leaf steps with radiation forced due")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=20_260_731)
    p.add_argument("--tree", action="store_true",
                   help="add the d01 and d02 domains to every case")
    p.add_argument("--d01-n", type=int, default=TREE["d01"]["nx"],
                   help="d01 columns per side (140 x 140 x 256 = 5.0 M)")
    p.add_argument("--tree-steps", type=int, default=2,
                   help="timed d01 steps (each 5 d02 + 15 d03 steps)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-util", type=float, default=10.0)
    p.add_argument("--max-other-mib", type=float, default=1024.0)
    p.add_argument("--wait", action="store_true",
                   help="poll until the card is idle instead of refusing")
    p.add_argument("--poll-s", type=float, default=300.0)
    p.add_argument("--wait-s", type=float, default=7200.0)
    p.add_argument("--no-lock", action="store_true",
                   help="do not take the woof GPU lock")
    p.add_argument("--output", type=Path)
    p.add_argument("--child", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.child:
        return _child(args.child)

    from woof.supervisor import GPUAlreadyLockedError, GPUFileLock

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    before = wait_for_idle(args, args.gpu)
    lock = None
    if not args.no_lock:
        while True:
            try:
                lock = GPUFileLock(before["uuid"],
                                   run_id="bench_les_256").acquire()
                break
            except GPUAlreadyLockedError as exc:
                if not args.wait:
                    raise SystemExit(str(exc))
                print(f"[bench] {exc}; retrying in {args.poll_s:.0f} s",
                      file=sys.stderr, flush=True)
                time.sleep(args.poll_s)
                before = wait_for_idle(args, args.gpu)
    rows = []
    try:
        for size in sizes:
            # Re-check before each case: another worker's short test may
            # have started on the card since the last one.
            cond = wait_for_idle(args, args.gpu)
            spec = _spec(args, size, cond["uuid"])
            # By UUID: CUDA's default device order is not nvidia-smi's, so an
            # index could time a different card from the one checked idle.
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=cond["uuid"])
            print(f"[bench] case {size}x{size}x{args.nz}"
                  + (" + d01/d02" if args.tree else ""),
                  file=sys.stderr, flush=True)
            t0 = time.time()
            proc = subprocess.run(
                [sys.executable, __file__, "--child", json.dumps(spec)],
                stdout=subprocess.PIPE, text=True, env=env)
            line = [ln for ln in proc.stdout.splitlines()
                    if ln.startswith("BENCH-ROW ")]
            row = (json.loads(line[-1][len("BENCH-ROW "):]) if line else
                   {"ok": False, "error": "child died without a row; "
                    "its stderr is above", "returncode": proc.returncode})
            row.update(size=size, wall_case_s=time.time() - t0,
                       conditions_before=cond,
                       conditions_after=card_conditions(args.gpu))
            rows.append(row)
            summary = (f"{row['cell_steps_per_s'] / 1e6:.2f} M cell-steps/s, "
                       f"{row['s_per_step_mean']:.3f} s/step, pool "
                       f"{row['pool_peak_gib']:.2f} GiB, process "
                       f"{row['process_gib_nvidia_smi']} GiB"
                       if row.get("ok") else row.get("error", "")[-300:])
            print(f"[bench]   {summary}", file=sys.stderr, flush=True)
    finally:
        if lock is not None:
            lock.release()
    doc = {"schema": SCHEMA, "argv": sys.argv[1:], "nz": args.nz,
           "ztop": args.ztop, "dx": args.dx, "dt": args.dt,
           "radt_minutes": args.radt_minutes, "tree": args.tree,
           "suites": {"les": LES, "meso": MESO}, "rows": rows,
           "limitations": "periodic flat idealized domain (WK82 sounding); "
                          "no lateral-boundary, nest or output kernels"}
    text = json.dumps(doc, indent=2, default=str)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)
    return 0 if all(r.get("ok") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
