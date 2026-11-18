"""Untimed events on the Metal facade record without draining the queue.

``cupy.cuda.Event(disable_timing=True).record()`` marks a point that only its
own waits (``synchronize``, ``done``) read; on the one Metal queue a record that
drained it cost a full ``mx.synchronize`` per deferred device check (ysu_gwdo
and friends).  The record queues a one-byte marker behind the thread's queued
work; ``synchronize`` and ``done`` evaluate the marker, which waits for that
work and no later work, from any thread.  A timed event keeps the blocking
record (``get_elapsed_time`` reads the host clock at the drain).
``WOOF_METAL_LAZY_EVENTS=0`` restores the draining record.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")


@pytest.fixture
def cp():
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import cupy
    return cupy


@pytest.fixture
def drains(monkeypatch):
    from woof.backend.facade import cuda as fcuda
    calls = []
    real = fcuda._drain

    def counting():
        calls.append(1)
        real()
    monkeypatch.setattr(fcuda, "_drain", counting)
    return calls


def _lazy(monkeypatch, on):
    from woof.backend.facade import cuda as fcuda
    fcuda._lazy_events.cache_clear()
    monkeypatch.setenv("WOOF_METAL_LAZY_EVENTS", "1" if on else "0")


def test_untimed_record_waits_at_the_wait(cp, drains, monkeypatch):
    _lazy(monkeypatch, True)
    try:
        event = cp.cuda.Event(disable_timing=True)
        x = cp.arange(1 << 16, dtype=cp.float32) * np.float32(3)
        event.record()
        assert not drains and event._marker is not None
        assert event.done and event._marker is None
        assert event.done                                # already waited
        event.record(cp.cuda.get_current_stream())
        event.synchronize()
        assert event._marker is None and not drains
        assert np.array_equal(x.get(), np.arange(1 << 16, dtype=np.float32) * np.float32(3))
        with pytest.raises(RuntimeError):
            cp.cuda.get_elapsed_time(event, event)     # CUDA refuses untimed events too
    finally:
        from woof.backend.facade import cuda as fcuda
        fcuda._lazy_events.cache_clear()


def test_a_wait_on_another_thread_waits_for_the_recorded_work(cp, monkeypatch):
    """MLX streams are per thread: the wait must reach the recording thread's
    queue, not drain its own."""
    import threading
    import time
    import mlx.core as mx
    _lazy(monkeypatch, True)
    try:
        box = {}

        def produce():
            x = mx.ones((2048, 2048))
            for _ in range(12):
                x = x @ x * 1e-4
            mx.async_eval(x)
            event = cp.cuda.Event(disable_timing=True)
            event.record()
            box["x"], box["event"] = x, event
        worker = threading.Thread(target=produce)
        worker.start()
        worker.join()
        box["event"].synchronize()
        start = time.perf_counter()
        mx.eval(box["x"])                    # already complete: no wait left
        assert time.perf_counter() - start < 0.005
    finally:
        from woof.backend.facade import cuda as fcuda
        fcuda._lazy_events.cache_clear()


def test_timed_and_switched_off_records_drain(cp, drains, monkeypatch):
    _lazy(monkeypatch, True)
    try:
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        end.record()
        assert len(drains) == 2
        assert cp.cuda.get_elapsed_time(start, end) >= 0.0
        cp.cuda.get_current_stream().record()           # CuPy's Stream.record: a timed event
        assert len(drains) == 3
        _lazy(monkeypatch, False)
        cp.cuda.Event(disable_timing=True).record()
        assert len(drains) == 4
    finally:
        from woof.backend.facade import cuda as fcuda
        fcuda._lazy_events.cache_clear()


def test_deferred_checks_refuse_with_untimed_events(cp, drains, monkeypatch):
    """A deferred device check still refuses at its step, and a clean check
    stays quiet, when its events are untimed (no drain per defer)."""
    from woof.hex.mpasphys.deferred_checks_v842 import DeferredChecks
    _lazy(monkeypatch, True)
    try:
        checks = DeferredChecks(cp)
        for _ in range(5):
            checks.defer(cp.zeros(16, dtype=cp.int32), "clean")
        assert not drains
        checks.flush(step=1)
        checks.poll(block=True)
        checks.defer(cp.asarray(np.array([0, 0, 7, 0], dtype=np.int32)), lambda v: f"bad {v}")
        checks.flush(step=2)
        with pytest.raises(RuntimeError, match=r"bad 7 \(deferred device check, step 2\)"):
            checks.poll(block=True)
    finally:
        from woof.backend.facade import cuda as fcuda
        fcuda._lazy_events.cache_clear()
