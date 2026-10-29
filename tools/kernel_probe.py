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


def run_here(unit, kernel, npz, scalars, threads, edits, host):
    args = _args(npz, scalars)
    if host:
        sys.path.insert(0, str(ROOT / "tools" / "thompson_real_column_parity"))
        import host_backend
        host_backend.install()
        if edits:
            host_backend.set_transform(unit, lambda t: _edit(t, edits))
        fn = host_backend.host_get_kernel(unit, kernel)
        arrays = [np.array(a, copy=True) if isinstance(a, np.ndarray) else a for a in args]
        fn(((threads + 127) // 128,), (128,), tuple(arrays))
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
    tpb = int(os.environ.get("PROBE_TPB", "128")); fn(((threads + tpb - 1) // tpb,), (tpb,), tuple(dev))
    return [cp.asnumpy(a) if isinstance(a, cp.ndarray) else a for a in dev]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("unit"); ap.add_argument("kernel"); ap.add_argument("npz")
    ap.add_argument("--scalars", nargs="*", default=[])
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--edit", nargs=2, action="append", default=[], metavar=("ANCHOR", "TEXT"))
    ap.add_argument("--side", choices=("metal", "host"), help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.side:
        res = run_here(a.unit, a.kernel, a.npz, a.scalars, a.threads, a.edit, a.side == "host")
        np.savez(a.out, **{f"arg{i}": r for i, r in enumerate(res) if isinstance(r, np.ndarray)})
        return 0
    results = {}
    for side in ("metal", "host"):
        out = f"/tmp/kernel_probe_{side}.npz"
        env = dict(os.environ, GPUWM_NUMERICS="strict")
        if side == "host":
            env.update(WOOF_BACKEND="cuda", PYTHONPATH="")
        cmd = [sys.executable, __file__, a.unit, a.kernel, a.npz, "--threads", str(a.threads),
               "--side", side, "--out", out, "--scalars", *a.scalars]
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
