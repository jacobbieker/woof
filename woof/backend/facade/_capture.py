"""CUDA-graph capture for the facade: record the launches, replay them.

CuPy's ``Stream.begin_capture()`` ... ``end_capture()`` records the work
launched in between WITHOUT running it, and ``Graph.launch()`` runs that
work again, on the same buffers, each time it is called.  woof captures a
step once and launches it every step (the global step, obs nudging, surface
FDDA, t-route, prepared wave runs, ...).

The facade used to run the work at capture and replay nothing, so a captured
step ran once and never again: the forecast stood still with no error.  Now:

* a translated kernel's launch during capture (``MetalFunction.__call__``)
  is recorded -- the function, grid, block, dynamic shared memory, and its
  arguments: arrays by reference (the replay binds the same storages, as a
  CUDA graph binds the same addresses), by-value arguments as their values
  at capture (a CUDA kernel node copies its parameters);
* a host function (``Stream.launch_host_func``) and an event record are
  recorded in their place in the sequence; so are the facade's FFTs (a cuFFT
  plan's ``fft``, ``cupy.fft.*``), which run on the host: a captured FFT's
  result array is allocated once at capture and rewritten by each replay;
* ``Graph.launch()`` runs the recorded sequence in order.

Anything else the facade would do during capture -- a facade array operation
or allocation, an in-place write (``a[...] = v``, ``fill``, a memcpy), a
synchronize or a read back to the host -- has no recorded form: the MLX
value it computes is fixed at capture.  The capture is then refused, loudly
(:class:`GraphCaptureRefused` from ``end_capture``, naming the first such
operation), rather than replayed wrong.  The refusal is also raised at the
operation itself when ``WOOF_GRAPH_CAPTURE_STRICT=1``, for diagnosis.
"""

from __future__ import annotations

import os
import threading

import numpy as np

_STATE = threading.local()


class GraphCaptureRefused(RuntimeError):
    """The captured work holds an operation the facade cannot replay."""


class _Recording:
    __slots__ = ("steps", "refused", "stream")

    def __init__(self, stream):
        self.steps = []
        self.refused = None
        self.stream = stream


def recording():
    """The thread's active recording, or None."""
    return getattr(_STATE, "rec", None)


def begin(stream):
    rec = recording()
    if rec is not None and rec.refused is None:
        raise GraphCaptureRefused("a graph capture is already active on this thread")
    # a refused capture whose end_capture never ran (its body raised) is
    # abandoned: nothing of it can be replayed
    _STATE.rec = _Recording(stream)


def end(stream):
    rec = recording()
    _STATE.rec = None
    if rec is None:
        raise GraphCaptureRefused("end_capture without begin_capture")
    if rec.refused is not None:
        raise GraphCaptureRefused(
            f"this CUDA-graph capture cannot be replayed on the mlx backend: {rec.refused}. "
            "Only translated kernel launches, host functions and event records are "
            "recorded; launch the work directly (woof.hip_backend.supports_graphs() is "
            "the callers' switch)")
    return rec.steps


def refuse(what: str):
    """An operation with no recorded form, during a capture."""
    rec = recording()
    if rec is None:
        return
    if rec.refused is None:
        rec.refused = what
    if os.environ.get("WOOF_GRAPH_CAPTURE_STRICT") == "1":
        _STATE.rec = None
        raise GraphCaptureRefused(f"{what} during a CUDA-graph capture")


class suspended:
    """Facade work that is part of a recorded step's setup, not of the
    replayed sequence: the output buffer a recorded host step writes (a
    captured FFT's result, which CUDA's graph allocates once and keeps)."""

    def __enter__(self):
        self._rec = recording()
        _STATE.rec = None
        return self

    def __exit__(self, *exc):
        _STATE.rec = self._rec
        return False


def record_launch(function, grid, block, args, shared_mem, stream=None) -> bool:
    """Record a kernel launch; True when recorded (the caller does not run it)."""
    rec = recording()
    if rec is None:
        return False
    if not _on_capturing_stream(rec, stream):
        # work outside the capturing stream (a body that raised and left its
        # `with stream:`, or another stream's launch): it runs now, and the
        # capture cannot be a replay of the stream's work
        refuse("a launch on another stream")
        return False
    kept = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in args)
    rec.steps.append(("kernel", function, grid, block, kept, shared_mem))
    return True


def _on_capturing_stream(rec, stream=None) -> bool:
    """The work goes to the capturing stream: the one it names, else the
    thread's current stream."""
    from woof.backend.facade.cuda import get_current_stream
    return (stream if stream is not None else get_current_stream()) is rec.stream


def record_host(callback, arg) -> bool:
    rec = recording()
    if rec is None or not _on_capturing_stream(rec):
        return False
    rec.steps.append(("host", callback, arg))
    return True


def record_event(event) -> bool:
    rec = recording()
    if rec is None or not _on_capturing_stream(rec):
        return False
    rec.steps.append(("event", event))
    return True


def replay(steps):
    """Run a recorded sequence, in order (never inside another capture)."""
    if recording() is not None:
        raise GraphCaptureRefused("launching a graph inside a capture")
    from woof.backend.facade.cuda import _drain
    for step in steps:
        kind = step[0]
        if kind == "kernel":
            _, function, grid, block, args, shared_mem = step
            function._launch(grid, block, args, shared_mem)
        elif kind == "host":
            _drain()
            step[1](step[2])
        else:
            step[1].record()
