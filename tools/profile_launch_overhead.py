"""Per-step launch-overhead profile of the resident step on small domains.

Built for the energy hindcast's question: hundreds of SMALL domains (0.1-2 M
cells, 49 levels, 300 m) packed onto one card, where the cost of issuing a
step may exceed the cost of executing it.  One process runs one LANE on one
suite and size, so a caller can interleave A/B runs and take medians:

``timing``
    wall time per step (``perf_counter`` around ``dycore.step`` plus the
    resident loop's per-step ``stability_report`` readback, closed by a device
    synchronise) and CUDA-event time over the same interval, with host
    seconds spent inside each physics section.  No CUPTI: nothing here is
    perturbed by instrumentation.
``cupti``
    the same steps under ``tools/launch_profile`` (CUPTI): per step, kernel
    launches, memcpys, memsets, host synchronisations, allocations and every
    CUDA API call, plus the device-busy union of executed kernels/copies and
    the per-section split of both (launches are tagged with the section that
    ISSUED them).
``alloc``
    CuPy memory-pool traffic per step (pool mallocs/frees and raw cudaMalloc).
``cprofile``
    a cProfile of the steps, top functions by own time.

Steps are split into ORDINARY steps and RADIATION steps by setting the
driver's ``radiation_due_override`` (the adaptive clock's own hook), so the
cadence does not decide what gets measured.

Suites (``tilestream.harness.make_config`` on a periodic, flat domain, built by
``tilestream.physics_inventory.default_builder``):

``windfarm``  the 300 m leaf: Thompson mp 8, MYNN surface layer + PBL, Noah,
              RTE-RRTMGP 4/4, no cumulus, km_opt 4, dt 1.5 s.
``hrrr``      configs/hrrr_native_3km_demo.toml's selectors: the same with
              legacy RRTMG, dx 3 km, dt 15 s.
``control``   tilestream's ``full(real74) +KF`` rung (YSU, Morrison, KF).
``dry``       the dry dynamical core.

Pinned limitations: periodic lateral boundaries (no specified-boundary
kernels), flat terrain, and a synthetic sounding; launch counts of a real
leaf differ by its boundary kernels.

Run::

    python tools/profile_launch_overhead.py --suite windfarm --nx 100 \\
        --ny 100 --nz 49 --lane cupti --output out.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "launch_profile"))

SCHEMA = "woof-launch-overhead-profile-v1"

_MOIST = dict(moist=True, ztop=20000.0)
SUITES = {
    "dry": dict(),
    "control": dict(_MOIST, mp_physics=10, km_opt=4, sf_sfclay_physics=91,
                    bl_pbl_physics=1, bldt=0.0, sf_surface_physics=2,
                    ra_sw_physics=4, ra_lw_physics=4, radt_minutes=12.0,
                    cu_physics=1, cudt_minutes=5.0),
    "windfarm": dict(_MOIST, mp_physics=8, km_opt=4, sf_sfclay_physics=5,
                     bl_pbl_physics=5, bldt=0.0, sf_surface_physics=2,
                     ra_sw_physics=4, ra_lw_physics=4, radt_minutes=5.0,
                     cu_physics=0, dx=300.0, dy=300.0, dt=1.5,
                     time_step_sound=4),
    "hrrr": dict(_MOIST, mp_physics=8, km_opt=4, sf_sfclay_physics=5,
                 bl_pbl_physics=5, bldt=0.0, sf_surface_physics=2,
                 ra_sw_physics=4, ra_lw_physics=4, radt_minutes=12.0,
                 ra_rrtmg_variant="rrtmg_legacy", cu_physics=0,
                 dx=3000.0, dy=3000.0, dt=15.0, time_step_sound=4),
}

#: Section tags (CUPTI tag index -> name).  0 is everything not wrapped:
#: the dynamical core and the step's own glue.
SECTIONS = ("dynamics", "radiation", "surface_layer", "land", "pbl",
            "microphysics", "physics_other", "loop_health")


class Sections:
    """Wraps the physics entry points so each knows its section.

    Nested sections restore the outer tag on exit, so the driver's own glue
    around the schemes stays attributed to ``physics_other``.
    """

    def __init__(self, prof=None):
        self.prof = prof
        self.stack = [0]
        self.host = {name: 0.0 for name in SECTIONS}
        self._t = [time.perf_counter()]

    def _switch(self, tag):
        """Charge the time since the last switch to the CURRENT top of the
        stack, then make ``tag`` the device tag."""
        now = time.perf_counter()
        self.host[SECTIONS[self.stack[-1]]] += now - self._t[0]
        self._t[0] = now
        if self.prof is not None:
            self.prof.set_tag(tag)

    def enter(self, tag):
        self._switch(tag)
        self.stack.append(tag)

    def leave(self):
        outer = self.stack[-2]
        self._switch(outer)
        self.stack.pop()

    def reset(self):
        self.host = {name: 0.0 for name in SECTIONS}
        self._t[0] = time.perf_counter()

    def wrap(self, owner, name, tag):
        original = getattr(owner, name)
        sections = self

        def wrapped(*args, **kwargs):
            sections.enter(tag)
            try:
                return original(*args, **kwargs)
            finally:
                sections.leave()

        wrapped.__wrapped__ = original
        setattr(owner, name, wrapped)


def install_sections(cfg, prof=None) -> Sections:
    from woof.core import dycore
    from woof.core.physics import PhysicsDriver, resolve_physics_dispatch

    sections = Sections(prof)
    sections.wrap(PhysicsDriver, "compute", SECTIONS.index("physics_other"))
    sections.wrap(PhysicsDriver, "_run_radiation", SECTIONS.index("radiation"))
    dispatch = resolve_physics_dispatch(cfg)
    for key, tag in (("sf_sfclay_physics", "surface_layer"),
                     ("sf_surface_physics", "land"),
                     ("bl_pbl_physics", "pbl")):
        method = dispatch.get(key)
        if method is not None:
            sections.wrap(PhysicsDriver, method, SECTIONS.index(tag))
    sections.wrap(dycore, "apply_microphysics",
                  SECTIONS.index("microphysics"))
    sections.wrap(dycore, "stability_report", SECTIONS.index("loop_health"))
    return sections


def build(suite: str, nx: int, ny: int, nz: int, seed: int):
    from tilestream import harness
    from tilestream import physics_inventory as physinv

    cfg = harness.make_config(nx, ny, nz, **SUITES[suite])
    if suite == "dry":
        state = harness.make_state(cfg, seed)
    else:
        state, _ = physinv.default_builder(cfg, seed)
    return cfg, state


def plan_steps(ordinary: int, radiation: int) -> list[bool]:
    """Radiation steps spread evenly through the ordinary ones."""
    kinds = [False] * ordinary
    if radiation:
        gap = max(1, ordinary // (radiation + 1))
        for r in range(radiation):
            kinds.insert(min(len(kinds), (r + 1) * gap + r), True)
    return kinds


def run(args) -> dict:
    import cupy as cp

    prof = None
    if args.lane == "cupti":
        import cupti_counter

        prof = cupti_counter.load(Path(args.build_dir))
        prof.start(activity=True)

    from woof.core import dycore

    cfg, state = build(args.suite, args.nx, args.ny, args.nz, args.seed)
    driver = getattr(state, "physics", None)
    sections = install_sections(cfg, prof) if driver is not None else Sections(prof)
    stepper = dycore.step

    def one_step(radiation: bool):
        if driver is not None:
            driver.radiation_due_override = bool(radiation)
        stepper(state, cfg)
        dycore.stability_report(state, cfg)

    kinds = plan_steps(args.steps, args.radiation_steps
                       if driver is not None else 0)
    # Warm-up: compile every kernel on both step kinds.
    for kind in ([True] if driver is not None else []) + [False] * args.warmup:
        one_step(kind)
    cp.cuda.runtime.deviceSynchronize()

    rows = []
    hook = None
    if args.lane == "alloc":
        hook = _AllocHook()
        hook.__enter__()
    profiler = None
    if args.lane == "cprofile":
        import cProfile

        profiler = cProfile.Profile()
    start_ev, end_ev = cp.cuda.Event(), cp.cuda.Event()
    for kind in kinds:
        cp.cuda.runtime.deviceSynchronize()
        if prof is not None:
            prof.reset()
        if hook is not None:
            hook.reset()
        sections.reset()
        if profiler is not None:
            profiler.enable()
        start_ev.record()
        t0 = time.perf_counter()
        one_step(kind)
        t_issue = time.perf_counter()
        end_ev.record()
        end_ev.synchronize()
        t1 = time.perf_counter()
        if profiler is not None:
            profiler.disable()
        sections._switch(sections.stack[-1])
        row = {"radiation": kind, "wall_s": t1 - t0,
               "issue_s": t_issue - t0,
               "event_s": cp.cuda.get_elapsed_time(start_ev, end_ev) / 1e3,
               "host_s": dict(sections.host)}
        if prof is not None:
            window = prof.read()
            row["cupti"] = window
            if not rows or kind != rows[-1]["radiation"]:
                row["apis"] = prof.apis()
        if hook is not None:
            row["alloc"] = dict(hook.counts)
        rows.append(row)
    if hook is not None:
        hook.__exit__(None, None, None)

    out = {"schema": SCHEMA, "suite": args.suite, "nx": args.nx,
           "ny": args.ny, "nz": args.nz, "cells": args.nx * args.ny * args.nz,
           "lane": args.lane, "graphs": args.graphs, "pid": os.getpid(),
           "health_readback": os.environ.get("WOOF_HEALTH_READBACK", ""),
           "started_unix": time.time(), "steps": rows,
           "summary": summarise(rows)}
    if profiler is not None:
        import io
        import pstats

        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(30)
        out["cprofile_top"] = stream.getvalue()
        # Who issues the blocking reads: the callers of every device->host
        # read CuPy exposes (each is a host synchronisation).
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).print_callers(
            r"\{method '(get|item|tolist)' of|asnumpy|_all_finite|"
            r"_consumer_cells_exist")
        out["cprofile_sync_callers"] = stream.getvalue()
    if args.digest:
        from tilestream import physics_inventory as physinv
        from tilestream.test_gate import hash_arrays

        out["digest"] = hash_arrays(physinv.carrier_inventory(state))
    return out


class _AllocHook:
    """Counts CuPy pool mallocs/frees and raw device allocations."""

    def __init__(self):
        import cupy as cp

        outer = self

        class Hook(cp.cuda.MemoryHook):
            name = "woof-launch-overhead"

            def malloc_postprocess(self, **kw):
                outer.counts["pool_malloc"] += 1
                outer.counts["pool_bytes"] += int(kw["size"])

            def free_postprocess(self, **kw):
                outer.counts["pool_free"] += 1

            def alloc_postprocess(self, **kw):
                outer.counts["cuda_malloc"] += 1

        self.hook = Hook()
        self.reset()

    def reset(self):
        self.counts = {"pool_malloc": 0, "pool_bytes": 0, "pool_free": 0,
                       "cuda_malloc": 0}

    def __enter__(self):
        self.hook.__enter__()
        return self

    def __exit__(self, *exc):
        return self.hook.__exit__(*exc)


def summarise(rows) -> dict:
    out = {}
    for kind, label in ((False, "ordinary"), (True, "radiation")):
        sel = [r for r in rows if r["radiation"] == kind]
        if not sel:
            continue
        med = lambda xs: statistics.median(xs)  # noqa: E731
        block = {"n": len(sel),
                 "wall_ms": 1e3 * med([r["wall_s"] for r in sel]),
                 "issue_ms": 1e3 * med([r["issue_s"] for r in sel]),
                 "event_ms": 1e3 * med([r["event_s"] for r in sel]),
                 "host_ms": {k: 1e3 * med([r["host_s"][k] for r in sel])
                             for k in SECTIONS}}
        if "cupti" in sel[0]:
            counts = sel[0]["cupti"]["counts"]
            block["counts"] = {k: med([r["cupti"]["counts"][k] for r in sel])
                               for k in counts}
            block["busy_ms"] = 1e3 * med(
                [r["cupti"]["busy_ns"] / 1e9 for r in sel])
            block["kernels_executed"] = med(
                [r["cupti"]["kernels_executed"] for r in sel])
            block["sync_wait_ms"] = 1e3 * med(
                [r["cupti"]["sync_wait_ns"] / 1e9 for r in sel])
            block["host_submit_ms"] = 1e3 * med(
                [r["wall_s"] - r["cupti"]["sync_wait_ns"] / 1e9 for r in sel])
            block["section_launches"] = {
                name: med([r["cupti"]["tag_launch"][i] for r in sel])
                for i, name in enumerate(SECTIONS)}
            block["section_busy_ms"] = {
                name: 1e3 * med([r["cupti"]["tag_busy_ns"][i] / 1e9
                                 for r in sel])
                for i, name in enumerate(SECTIONS)}
            block["dropped"] = max(r["cupti"]["dropped"] for r in sel)
        if "alloc" in sel[0]:
            block["alloc"] = {k: med([r["alloc"][k] for r in sel])
                              for k in sel[0]["alloc"]}
        out[label] = block
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=sorted(SUITES), default="windfarm")
    p.add_argument("--nx", type=int, default=100)
    p.add_argument("--ny", type=int, default=100)
    p.add_argument("--nz", type=int, default=49)
    p.add_argument("--seed", type=int, default=20_260_731)
    p.add_argument("--lane", choices=("timing", "cupti", "alloc", "cprofile"),
                   default="timing")
    p.add_argument("--steps", type=int, default=8,
                   help="ordinary steps measured")
    p.add_argument("--radiation-steps", type=int, default=2)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--graphs", choices=("off",), default="off",
                   help="recorded in the output; resident CUDA-graph replay "
                        "is not implemented (see "
                        "docs/dev/perf-launch-overhead-20261015.md section 5)")
    p.add_argument("--digest", action="store_true",
                   help="record the carrier-inventory digest after the run")
    p.add_argument("--build-dir",
                   default=os.environ.get("WOOF_CUPTI_BUILD",
                                          "/tmp/woof-cupti-build"))
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    out = run(args)
    text = json.dumps(out, indent=1, default=str)
    if args.output:
        args.output.write_text(text)
    s = out["summary"]
    for label, block in s.items():
        line = (f"{args.suite} {args.nx}x{args.ny}x{args.nz} {args.lane} "
                f"graphs={args.graphs} {label}: wall {block['wall_ms']:.2f} ms "
                f"issue {block['issue_ms']:.2f} ms event {block['event_ms']:.2f} ms")
        if "busy_ms" in block:
            c = block["counts"]
            line += (f" busy {block['busy_ms']:.2f} ms submit {block['host_submit_ms']:.2f} ms"
                     f" launches {c['launch']}"
                     f" graph {c['graph_launch']} syncs {c['sync'] + c['sync_copy']}"
                     f" api {c['total']}")
        if "alloc" in block:
            line += f" alloc {block['alloc']}"
        print(line)
    if "digest" in out:
        print("digest", out["digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
