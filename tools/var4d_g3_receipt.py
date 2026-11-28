"""G3 receipt for the 4D-Var window (docs/dev/WRFDA-4DVAR-PLAN.md section 6).

Times P3a's window (:class:`woof.grad.wrfplus.window.Window`) over a whole
4D-Var window: its forward (the base-only build), its TL (slots saved) and
its AD (forcing at every slot) under each checkpoint policy, against the
forward window in the same precision and against woof's own binary32
forecast over the same steps (``dycore.step``, the forward model G3 names).
With ``--l1-lane`` (pb) it also times
:class:`woof.da.var.window4d.WindowModel`'s TL and AD on the WRFDA tutorial
window (lane L1: 90x60x41, specified, moist, 60 steps), host transfers
included: one inner iteration's model cost.

Rows are interleaved per repeat and the median is reported.  Run it with
the card otherwise idle (check nvidia-smi); on Metal one job at a time.

    python -m tools.var4d_g3_receipt [--nx 90 --ny 60 --nz 41 --nsteps 60 --sps 10]
        [--dtypes float64,float32] [--repeats 3] [--l1-lane DIR --work DIR] [--out receipt.json]

Strict numerics are set before woof is imported (the window refuses
anything else).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

STRICT = dict(GPUWM_WRF_EXACT="1", WOOF_WRF_EXACT_DIAGNOSTICS="1", WOOF_WRF_EXACT_ADVECTION="1",
              GPUWM_WRF_EXACT_DIFFUSION="1", GPUWM_WRF_EXACT_BIGSTEP="1")


def _first(o):
    while isinstance(o, (dict, list, tuple)):
        if not o:
            return None
        o = next(iter(o.values())) if isinstance(o, dict) else o[0]
    return o


def _sync(cp, out=None):
    """Wait for the device: a host read of one value forces a lazy backend too."""
    try:
        cp.cuda.get_current_stream().synchronize()
    except Exception:
        pass
    v = _first(out)
    if v is not None:
        float(v.ravel()[0])


def _timed(cp, fn):
    _sync(cp)
    t = time.perf_counter()
    out = fn()
    _sync(cp, out)
    return time.perf_counter() - t, out


def _synthetic(nx, ny, nz, seed=1):
    import dataclasses

    import numpy as np
    from tilestream import harness as H

    from woof.core import dycore as D

    cfg = H.make_config(nx, ny, nz, h_sca_adv_order=5, terrain_opt=1, moist=True, moist_adv_opt=0)
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    ridge = 400.0 * np.exp(-(((ii - nx / 2) / (nx / 5)) ** 2 + ((jj - ny / 2) / (ny / 4)) ** 2))
    geo = dataclasses.replace(H.neutral_geography(cfg), terrain=ridge)

    def make():
        st = H.make_state(cfg, seed, geography=geo)
        D.step(st, cfg)
        return st
    return cfg, make


def _rand(cp, np, ref: dict, names, rng, scale=1e-3):
    return {n: cp.asarray(rng.standard_normal(tuple(ref[n].shape)) * float(cp.abs(ref[n]).max() * scale + 1e-12)
                          ).astype(ref[n].dtype) for n in names}


def window_rows(cfg, make, nsteps, sps, dtypes, repeats, metal):
    """Per dtype: the forward, TL and AD windows (policies all and slot) and woof's binary32 forecast."""
    import numpy as np
    import cupy as cp

    from woof.core import dycore as D
    from woof.grad.wrfplus import window as W

    slots = list(range(0, nsteps + 1, sps))
    out = {}
    for dtn in dtypes:
        dt = np.dtype(dtn)
        w = W.Window(make(), cfg, nsteps, dtype=dt, checkpoint_every=1)
        rng = np.random.default_rng(3)
        dx0 = _rand(cp, np, w._start, w.inputs, rng)
        forcing = {k: _rand(cp, np, w._start, w.inputs, rng) for k in slots}
        state_bytes = sum(int(w._start[n].size) * dt.itemsize for n in w.inputs)
        times = {k: [] for k in ("primal32", "forward_all", "forward_slot", "tl", "ad_all", "ad_slot")}
        for r in range(repeats + 1):   # repeat 0 warms the builds
            st = make()

            def primal():
                for _ in range(nsteps):
                    D.step(st, cfg)
                return {"u": st.u}
            row = {"primal32": _timed(cp, primal)[0]}
            w.every = 1
            row["forward_all"] = _timed(cp, w.forward)[0]
            row["tl"] = _timed(cp, lambda: w.tl(dx0, save_at=slots))[0]
            row["ad_all"] = _timed(cp, lambda: w.ad(None, forcing))[0]
            w.every = sps
            row["forward_slot"] = _timed(cp, w.forward)[0]
            row["ad_slot"] = _timed(cp, lambda: w.ad(None, forcing))[0]
            if r:
                for k, v in row.items():
                    times[k].append(v)
            del st
        med = {k: statistics.median(v) for k, v in times.items()}
        fwd, prim = med["forward_all"], med["primal32"]
        out[dtn] = {
            "seconds": med, "per_step_ms": {k: 1e3 * v / nsteps for k, v in med.items()},
            "vs_forward_window": {k: v / fwd for k, v in med.items()},
            "vs_woof_binary32_forecast": {k: v / prim for k, v in med.items()},
            "memory": {"state_bytes": state_bytes, "all_stored": nsteps * state_bytes,
                       "slot_stored_peak": (len(slots) - 1 + sps - 1) * state_bytes},
        }
        del w
        if not metal:
            cp.get_default_memory_pool().free_all_blocks()
    return out


def l1_rows(lane: Path, work: Path, repeats):
    """WindowModel's TL and AD (host transfers included) on lane L1's window, each policy."""
    import numpy as np
    import cupy as cp

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests._var4d_l1_case import build_window, prepare_case
    from woof.da.var import window4d as w4

    nslots, sps = 7, 10
    case = prepare_case(lane, work / "case")
    c = build_window(case, (nslots - 1) * sps, checkpoint_every=1)
    win = c.window
    model = w4.WindowModel(win, sps, checkpoint="all")
    rng = np.random.default_rng(7)
    names = {"u_2": "u", "v_2": "v", "w_2": "w", "t_2": "thp", "mu_2": "mup", "ph_2": "php", "p": "p_pert",
             "qv": "qv"}

    def rnd():
        return {k: 1e-3 * rng.standard_normal(tuple(win._start[w].shape)[::-1]) for k, w in names.items()}
    g0, fs = rnd(), [rnd() for _ in range(nslots)]
    times = {k: [] for k in ("forward", "model_tl", "model_ad_all", "model_ad_slot")}
    reports = {}
    for r in range(repeats + 1):
        row = {}
        model = w4.WindowModel(win, sps, checkpoint="all")
        row["forward"] = _timed(cp, lambda: win.forward(win._checkpoints[0]))[0]
        row["model_tl"] = _timed(cp, lambda: model.tl(g0, nslots))[0]
        row["model_ad_all"] = _timed(cp, lambda: model.ad(fs))[0]
        reports["all"] = model.checkpoint_report()
        model = w4.WindowModel(win, sps, checkpoint="slot")
        row["model_ad_slot"] = _timed(cp, lambda: model.ad(fs))[0]
        reports["slot"] = model.checkpoint_report()
        if r:
            for k, v in row.items():
                times[k].append(v)
    med = {k: statistics.median(v) for k, v in times.items()}
    return {"seconds": med, "vs_forward_window": {k: v / med["forward"] for k, v in med.items()},
            "checkpoint_report": reports, "nsteps": win.nsteps, "dims": [c.cfg.nx, c.cfg.ny, c.cfg.nz]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--nx", type=int, default=90)
    p.add_argument("--ny", type=int, default=60)
    p.add_argument("--nz", type=int, default=41)
    p.add_argument("--nsteps", type=int, default=60)
    p.add_argument("--sps", type=int, default=10, help="steps per slot (var4d_bin / time_step)")
    p.add_argument("--dtypes", default=None, help="comma list; default float64,float32 (float32 on Metal)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--l1-lane", type=Path, default=None, help="run_4dvar.sh l1's run directory (CUDA)")
    p.add_argument("--work", type=Path, default=Path("/tmp/var4d-g3"))
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    if a.nsteps % a.sps:
        p.error("--sps must divide --nsteps")
    for k, v in STRICT.items():
        os.environ.setdefault(k, v)
    import woof  # noqa: F401  (installs the backend's cupy)
    import cupy as cp
    from woof import backend

    metal = backend.active() == "mlx"
    if cp.__name__ != "cupy" or getattr(cp, "__woof_facade__", False):
        import woof.core.resident_admission as RA
        RA.device_free_bytes = lambda: None
    dtypes = (a.dtypes or ("float32" if metal else "float64,float32")).split(",")
    if metal and "float64" in dtypes:
        p.error("Metal has no binary64")
    cfg, make = _synthetic(a.nx, a.ny, a.nz)
    rec = {"backend": backend.active(), "case": {"synthetic": [a.nx, a.ny, a.nz], "nsteps": a.nsteps,
                                                 "sps": a.sps, "moist": True, "periodic": True},
           "repeats": a.repeats, "window": window_rows(cfg, make, a.nsteps, a.sps, dtypes, a.repeats, metal)}
    if a.out:   # the window rows survive a failing L1 part
        a.out.write_text(json.dumps(rec, indent=1) + "\n")
    if a.l1_lane is not None:
        rec["l1_window_model"] = l1_rows(a.l1_lane, a.work, a.repeats)
    text = json.dumps(rec, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
