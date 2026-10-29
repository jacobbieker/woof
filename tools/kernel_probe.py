"""Run one kernel on saved arguments on this Mac's two builds of the same
CUDA source -- the Metal translation and the host C++ build (CUDA's
arithmetic on the CPU, tools/thompson_real_column_parity/host_backend.py)
-- and diff every output word.  A probe edit can be applied to the source
first, to watch an intermediate value through an output slot.

The saved arguments are an .npz from tools/launch_trace.py
(WOOF_LAUNCH_SAVE=SEQ:path): arrays ``arg<i>``; scalars come from the trace
row, given here as ``--scalars``.

    pixi run -e mlx python tools/kernel_probe.py noahmp_water k_water \\
        /tmp/kw.npz --scalars 5:int32:113 --threads 113 \\
        [--edit 'ANCHOR' 'TEXT TO INSERT BEFORE IT']
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _args(npz, scalars):
    data = np.load(npz)
    arrays = {int(k[3:]): data[k] for k in data.files}
    out = dict(arrays)
    for spec in scalars:
        i, dtype, value = spec.split(":")
        out[int(i)] = np.dtype(dtype).type(value)
    return [out[i] for i in sorted(out)]


def _edit(text, edits):
    for anchor, insert in edits:
        at = text.find(anchor)
        if at < 0:
            raise SystemExit(f"probe anchor not found: {anchor!r}")
        text = text[:at] + insert + text[at:]
    return text


def _shape(threads, tpb, grid, block):
    """The launch: ``--grid``/``--block`` as given, else 1-D over ``threads``."""
    if grid:
        return grid, block or (tpb, 1, 1)
    return ((threads + tpb - 1) // tpb,), (tpb,)


def run_here(unit, kernel, npz, scalars, threads, edits, host, grid=None, block=None):
    args = _args(npz, scalars)
    if host:
        sys.path.insert(0, str(ROOT / "tools" / "thompson_real_column_parity"))
        import host_backend
        host_backend.install()
        if edits:
            host_backend.set_transform(unit, lambda t: _edit(t, edits))
        fn = host_backend.host_get_kernel(unit, kernel)
        arrays = [np.array(a, copy=True) if isinstance(a, np.ndarray) else a for a in args]
        fn(*_shape(threads, 128, grid, block), tuple(arrays))
        return arrays
    import woof  # noqa: F401
    import cupy as cp
    from woof.core import kernels
    if edits:
        original = kernels.module_source
        kernels.module_source = lambda name, *a, **k: (
            _edit(original(name, *a, **k), edits) if name == unit else original(name, *a, **k))
    fn = kernels.get_kernel(unit, kernel)
    dev = [cp.asarray(a) if isinstance(a, np.ndarray) else a for a in args]
    fn(*_shape(threads, int(os.environ.get("PROBE_TPB", "128")), grid, block), tuple(dev))
    return [cp.asnumpy(a) if isinstance(a, cp.ndarray) else a for a in dev]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("unit"); ap.add_argument("kernel"); ap.add_argument("npz")
    ap.add_argument("--scalars", nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--grid", type=lambda t: tuple(int(x) for x in t.split(",")),
                    help="a launch grid X,Y,Z (instead of --threads)")
    ap.add_argument("--block", type=lambda t: tuple(int(x) for x in t.split(",")))
    ap.add_argument("--edit", nargs=2, action="append", default=[], metavar=("ANCHOR", "TEXT"))
    ap.add_argument("--repeat", type=int, default=0,
                    help="Metal only: replay N times in one process and compare the repeats "
                         "(a kernel whose words vary is nondeterministic)")
    ap.add_argument("--side", choices=("metal", "host"), help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.repeat:
        runs = [run_here(a.unit, a.kernel, a.npz, a.scalars, a.threads, a.edit, False,
                         a.grid, a.block)
                for _ in range(a.repeat)]
        for i, r in enumerate(runs[1:], 1):
            bad = [k for k, (x, y) in enumerate(zip(runs[i - 1], r)) if isinstance(x, np.ndarray)
                   and not np.array_equal(x.view(np.uint8), y.view(np.uint8))]
            print(f"repeat {i}: " + (f"identical to repeat {i - 1}" if not bad
                                     else f"arguments {bad} differ from repeat {i - 1}"))
        return 0
    if a.side:
        res = run_here(a.unit, a.kernel, a.npz, a.scalars, a.threads, a.edit, a.side == "host",
                       a.grid, a.block)
        np.savez(a.out, **{f"arg{i}": r for i, r in enumerate(res) if isinstance(r, np.ndarray)})
        return 0
    results = {}
    for side in ("metal", "host"):
        out = f"/tmp/kernel_probe_{side}.npz"
        env = dict(os.environ, GPUWM_NUMERICS="strict")
        if side == "host":
            sys.path.insert(0, str(ROOT / "tools" / "thompson_real_column_parity"))
            from host_backend import host_environment
            env = host_environment(env)
        cmd = [sys.executable, __file__, a.unit, a.kernel, a.npz, "--threads", str(a.threads),
               "--side", side, "--out", out, "--scalars", *a.scalars]
        for flag, value in (("--grid", a.grid), ("--block", a.block)):
            if value:
                cmd += [flag, ",".join(map(str, value))]
        for anchor, text in a.edit:
            cmd += ["--edit", anchor, text]
        subprocess.run(cmd, check=True, env=env, cwd=ROOT)
        results[side] = np.load(out)
    m, h = results["metal"], results["host"]
    for k in m.files:
        x, y = m[k], h[k]
        bits = (x.view(np.uint32) if x.dtype.itemsize == 4 else x.view(np.uint64)) != \
               (y.view(np.uint32) if y.dtype.itemsize == 4 else y.view(np.uint64))
        if bits.any():
            print(f"{k} {x.shape}: {int(bits.sum())} words differ")
            if x.ndim == 2:
                cols = {int(j): int(bits[:, j].sum()) for j in range(x.shape[1]) if bits[:, j].any()}
                print("   by field:", json.dumps(cols)[:300])
                i = int(np.flatnonzero(bits.any(1))[0])
                js = np.flatnonzero(bits[i])[:6]
                print(f"   row {i}:", [(int(j), float(x[i, j]), float(y[i, j])) for j in js])
        else:
            print(f"{k} {x.shape}: identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
