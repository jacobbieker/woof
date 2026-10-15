"""ctypes front end for ``woof_cupti.c``: per-window CUDA API census + busy.

Build-on-demand: :func:`load` compiles ``woof_cupti.c`` with the system C
compiler against the CUDA toolkit's CUPTI (``$CUDA_HOME`` or ``/opt/cuda``)
into ``build_dir`` and loads it.  Must be started BEFORE the first CUDA call
whose activity is wanted, and after ``import cupy`` is fine (CuPy creates its
context lazily).

Usage::

    prof = load(build_dir)
    prof.start(activity=True)
    prof.reset(); prof.set_tag(0)
    ... work ...
    cupy.cuda.runtime.deviceSynchronize()
    window = prof.read()      # dict: counts, per-tag launches/busy, busy_ns

Counts are pinned to the categories in :data:`COUNTERS`; ``sync`` counts the
explicit synchronise calls, ``sync_copy`` the blocking memcpys (also counted
under their direction), so host synchronisations = ``sync + sync_copy``.
``sync_wait_ns`` is the host time spent inside the synchronising calls AND
every copy (a copy to pageable host memory blocks until the stream drains),
so a window's wall time minus it is the host's own submission time.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import subprocess

COUNTERS = ("total", "launch", "graph_launch", "h2d", "d2h", "d2d",
            "memcpy_other", "memset", "sync", "sync_copy", "alloc", "free",
            "query", "other")

_SRC = Path(__file__).with_name("woof_cupti.c")


def _cuda_home() -> Path:
    return Path(os.environ.get("CUDA_HOME") or "/opt/cuda")


def build(build_dir: Path) -> Path:
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    # Named by the source digest, so two checkouts sharing a build
    # directory never load each other's library.
    digest = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:12]
    out = build_dir / f"libwoof_cupti-{digest}.so"
    if out.exists():
        return out
    cuda = _cuda_home()
    lib = cuda / "lib64"
    subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", str(out), str(_SRC),
                    f"-I{cuda / 'include'}", f"-L{lib}", f"-Wl,-rpath,{lib}",
                    "-lcupti", "-lpthread"], check=True)
    return out


class Profiler:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        self.ncounters = self.lib.woofprof_ncounters()
        self.ntags = self.lib.woofprof_ntags()
        assert self.ncounters == len(COUNTERS), "C/Python counter table drift"
        ll = ctypes.POINTER(ctypes.c_longlong)
        self.lib.woofprof_read.argtypes = [ll, ll, ll, ll]
        self.lib.woofprof_read.restype = ctypes.c_longlong
        self.lib.woofprof_api.argtypes = [ctypes.c_int, ll]
        self.lib.woofprof_api.restype = ctypes.c_char_p
        self.activity = False

    def start(self, activity: bool = True) -> None:
        rc = self.lib.woofprof_start(int(bool(activity)))
        if rc != 0:
            raise RuntimeError(f"CUPTI start failed ({rc})")
        self.activity = bool(activity)

    def set_tag(self, tag: int) -> None:
        self.lib.woofprof_set_tag(int(tag))

    def reset(self) -> None:
        self.lib.woofprof_reset()

    def read(self) -> dict:
        counts = (ctypes.c_longlong * self.ncounters)()
        tl = (ctypes.c_longlong * self.ntags)()
        tb = (ctypes.c_longlong * self.ntags)()
        extra = (ctypes.c_longlong * 5)()
        dropped = self.lib.woofprof_read(counts, tl, tb, extra)
        return {"counts": dict(zip(COUNTERS, (int(c) for c in counts))),
                "tag_launch": [int(x) for x in tl],
                "tag_busy_ns": [int(x) for x in tb],
                "busy_ns": int(extra[0]), "span_ns": int(extra[2] - extra[1]),
                "kernels_executed": int(extra[3]),
                "sync_wait_ns": int(extra[4]), "dropped": int(dropped)}


    def apis(self) -> dict:
        """``{api name: calls}`` since the last :meth:`reset`."""
        out = {}
        count = ctypes.c_longlong()
        i = 0
        while True:
            name = self.lib.woofprof_api(i, ctypes.byref(count))
            if name is None:
                return out
            out[name.decode()] = out.get(name.decode(), 0) + int(count.value)
            i += 1


def load(build_dir: Path) -> Profiler:
    return Profiler(build(build_dir))
