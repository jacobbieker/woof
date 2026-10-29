"""Record every kernel launch's array words, to find where two backends part.

``install(path)`` (before woof imports cupy) makes each launch append one
JSON line to ``path``: the sequence number, the kernel name, and for every
array argument its dtype, shape and a SHA-256 of its words *after* the
launch, then of its words before it (scalar arguments verbatim).  Two
backends' words are compared only where the launch was given the same
words: an uninitialised ``cupy.empty`` buffer holds old pool data on CUDA
and zeros on MLX, and a kernel that leaves it alone is not a divergence.  Run the same scenario on CUDA and on
MLX, then ``compare(a, b)`` reports the first launch whose arrays differ --
the first kernel to check, given that every launch before it agreed.

On CUDA the hook wraps ``cupy.RawModule``, ``cupy.RawKernel`` and woof's
own direct-NVRTC module wrapper (CuPy's classes are Cython, so their
methods cannot be patched in place); on MLX it wraps
``woof.backend.metal.loader.MetalFunction.__call__``.

Usage::

    WOOF_LAUNCH_TRACE=/tmp/a.jsonl python -c "import launch_trace; launch_trace.install(); ..."
    python tools/launch_trace.py compare /tmp/cuda.jsonl /tmp/metal.jsonl
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

_STATE = {"path": None, "seq": 0}


def _digest(a):
    import numpy as np
    import cupy as cp
    host = cp.asnumpy(a)
    return [str(host.dtype), list(host.shape), hashlib.sha256(np.ascontiguousarray(host).tobytes()).hexdigest()[:16]]


def _before(args):
    """The digests of a launch's arrays as it is given them."""
    if not _STATE.get("path"):
        return None
    import cupy as cp
    return [_digest(a)[2] if isinstance(a, cp.ndarray) else None for a in args]


def _record(name, args, before=None):
    if not _STATE.get("path"):
        return
    import numpy as np
    import cupy as cp
    row = {"seq": _STATE["seq"], "kernel": name, "args": []}
    _STATE["seq"] += 1
    for i, a in enumerate(args):
        if isinstance(a, cp.ndarray):
            row["args"].append(_digest(a) + ([before[i]] if before else []))
        elif isinstance(a, (np.generic, int, float, bool)):
            row["args"].append(repr(a.item() if isinstance(a, np.generic) else a))
        else:
            row["args"].append(type(a).__name__)
    with open(_STATE["path"], "a") as f:
        f.write(json.dumps(row) + "\n")
    save = os.environ.get("WOOF_LAUNCH_SAVE", "")       # "SEQ:path.npz": keep one launch's arrays
    if save and int(save.split(":", 1)[0]) == row["seq"]:
        np.savez(save.split(":", 1)[1], **{f"arg{i}": cp.asnumpy(a) for i, a in enumerate(args)
                                          if isinstance(a, cp.ndarray)})


def _save_before(args):
    """WOOF_LAUNCH_SAVE_BEFORE=SEQ:path.npz: the arrays a launch is given,
    before it runs (tools/kernel_probe.py replays them)."""
    save = os.environ.get("WOOF_LAUNCH_SAVE_BEFORE", "")
    if save and int(save.split(":", 1)[0]) == _STATE["seq"]:
        import numpy as np
        import cupy as cp
        np.savez(save.split(":", 1)[1], **{f"arg{i}": cp.asnumpy(a) for i, a in enumerate(args)
                                          if isinstance(a, cp.ndarray)})


class _Traced:
    def __init__(self, fn, name):
        self._fn, self._name = fn, name

    def __call__(self, grid, block, args, *rest, **kw):
        _save_before(args)
        before = _before(args)
        out = self._fn(grid, block, args, *rest, **kw)
        _record(self._name, args, before)
        return out

    def __getattr__(self, attr):
        return getattr(self._fn, attr)


def set_output(path: str) -> None:
    """Send later launches to ``path`` (truncated), numbering from 0."""
    _STATE["path"], _STATE["seq"] = path, 0
    open(path, "w").close()


def install(path: str | None = None) -> None:
    if _STATE.get("installed"):
        if path:
            set_output(path)
        return
    _STATE["installed"] = True
    set_output(path or os.environ["WOOF_LAUNCH_TRACE"])
    from woof import backend
    if backend.requested() != "cuda":
        from woof.backend.metal import loader
        original = loader.MetalFunction.__call__

        def call(self, grid, block, args, shared_mem=0, stream=None):
            _save_before(args)
            before = _before(args)
            out = original(self, grid, block, args, shared_mem=shared_mem, stream=stream)
            _record(self.name, args, before)
            return out
        loader.MetalFunction.__call__ = call
        return
    import cupy

    class RawModule(cupy.RawModule):
        def get_function(self, name):
            return _Traced(super().get_function(name), name)

    class RawKernel:
        def __init__(self, code, name, *a, **k):
            self._k, self._name = cupy.RawKernel.__wrapped_original__(code, name, *a, **k), name

        def __call__(self, grid, block, args, *rest, **kw):
            _save_before(args)
            before = _before(args)
            out = self._k(grid, block, args, *rest, **kw)
            _record(self._name, args, before)
            return out

        def __getattr__(self, attr):
            return getattr(self._k, attr)

    RawKernel.__wrapped_original__ = cupy.RawKernel
    cupy.RawModule = RawModule
    cupy.RawKernel = RawKernel
    # woof's direct-NVRTC loads (cupy.cuda.function.Module, whose type CuPy
    # checks) reach kernels through woof's own wrapper
    from woof.core import kernels
    get = kernels._DriverModule.get_function
    kernels._DriverModule.get_function = lambda self, name: _Traced(get(self, name), name)
    # ... and through woof.core.ieee_module's IEEE-option loader
    from woof.core import ieee_module
    get_ieee = ieee_module._LoadedModule.get_function
    ieee_module._LoadedModule.get_function = lambda self, name: _Traced(get_ieee(self, name), name)


def differing_args(x, y):
    """The arguments of two records of one launch whose words differ after
    it, among those given the same words before it."""
    bad = []
    for i, (p, q) in enumerate(zip(x["args"], y["args"])):
        if p == q:
            continue
        if isinstance(p, list) and isinstance(q, list) and len(p) == len(q) == 4:
            if p[3] != q[3] or p[:3] == q[:3]:
                continue
        bad.append(i)
    return bad


def compare(a_path: str, b_path: str) -> int:
    a = [json.loads(l) for l in open(a_path)]
    b = [json.loads(l) for l in open(b_path)]
    for x, y in zip(a, b):
        if x["kernel"] != y["kernel"]:
            print(f"launch {x['seq']}: the sequences diverge: {x['kernel']} vs {y['kernel']}")
            return 1
        bad = differing_args(x, y)
        if bad:
            print(f"launch {x['seq']}: {x['kernel']} differs in argument(s) {bad}")
            for i in bad:
                print(f"   arg {i}: {x['args'][i]}  vs  {y['args'][i]}")
            return 1
    print(f"{min(len(a), len(b))} launches agree" + ("" if len(a) == len(b) else f" (lengths {len(a)} vs {len(b)})"))
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
