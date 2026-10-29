"""Record every kernel launch's array words, to find where two backends part.

``install(path)`` (before woof imports cupy) makes each launch append one
JSON line to ``path``: the sequence number, the kernel name, and for every
array argument its dtype, shape and a SHA-256 of its words *after* the
launch (scalar arguments verbatim).  Run the same scenario on CUDA and on
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


def _record(name, args):
    import numpy as np
    import cupy as cp
    row = {"seq": _STATE["seq"], "kernel": name, "args": []}
    _STATE["seq"] += 1
    for a in args:
        if isinstance(a, cp.ndarray):
            row["args"].append(_digest(a))
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


class _Traced:
    def __init__(self, fn, name):
        self._fn, self._name = fn, name

    def __call__(self, grid, block, args, *rest, **kw):
        out = self._fn(grid, block, args, *rest, **kw)
        _record(self._name, args)
        return out

    def __getattr__(self, attr):
        return getattr(self._fn, attr)


def install(path: str | None = None) -> None:
    _STATE["path"] = path or os.environ["WOOF_LAUNCH_TRACE"]
    open(_STATE["path"], "w").close()
    from woof import backend
    if backend.requested() != "cuda":
        from woof.backend.metal import loader
        original = loader.MetalFunction.__call__

        def call(self, grid, block, args, shared_mem=0, stream=None):
            out = original(self, grid, block, args, shared_mem=shared_mem, stream=stream)
            _record(self.name, args)
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
            out = self._k(grid, block, args, *rest, **kw)
            _record(self._name, args)
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


def compare(a_path: str, b_path: str) -> int:
    a = [json.loads(l) for l in open(a_path)]
    b = [json.loads(l) for l in open(b_path)]
    for x, y in zip(a, b):
        if x["kernel"] != y["kernel"]:
            print(f"launch {x['seq']}: the sequences diverge: {x['kernel']} vs {y['kernel']}")
            return 1
        if x["args"] != y["args"]:
            bad = [i for i, (p, q) in enumerate(zip(x["args"], y["args"])) if p != q]
            print(f"launch {x['seq']}: {x['kernel']} differs in argument(s) {bad}")
            for i in bad:
                print(f"   arg {i}: {x['args'][i]}  vs  {y['args'][i]}")
            return 1
    print(f"{min(len(a), len(b))} launches agree" + ("" if len(a) == len(b) else f" (lengths {len(a)} vs {len(b)})"))
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
