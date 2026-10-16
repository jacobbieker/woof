"""GPU tier of tests/test_history_output_path.py: the device snapshot.

* a snapshot frame, a direct frame and the synchronous ``state_frame``
  path give byte-identical files;
* in snapshot mode the producer stream runs AHEAD of the host copy (it is
  not fenced behind it), and a mutation it makes right after the submit
  never reaches the file;
* the named fallbacks: unpriced, and a failed device allocation;
* the page-locked pool is released at the drain point.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from woof.io import history_staging

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _clean_plans(monkeypatch):
    monkeypatch.delenv(history_staging.ASYNC_HISTORY_ENV, raising=False)
    history_staging.forget_priced_plans()
    yield
    history_staging.forget_priced_plans()


def _price_room(nbytes=2 ** 30):
    import cupy as cp
    history_staging.record_priced_plan(
        {"free_bytes": int(nbytes), "need_bytes": 0, "what": "test"},
        device=cp.cuda.Device().id)


def _writer(cfg):
    from woof.config import soil_layer_count
    from woof.io.wrfout import AsyncDomainWrfoutWriter

    return AsyncDomainWrfoutWriter(
        nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
        title="snapshot test", global_attrs={},
        soil_layers=soil_layer_count(cfg))


def _spin_kernel():
    import cupy as cp
    spin = cp.RawKernel(r"""
    extern "C" __global__ void spin(long long cycles) {
        long long start = clock64();
        while (clock64() - start < cycles) { }
    }""", "spin")
    warm = cp.zeros(4, cp.float32)
    warm += cp.float32(1.0)
    spin((1,), (1,), (np.int64(1),))
    cp.cuda.Device().synchronize()
    return spin


def test_snapshot_direct_and_synchronous_frames_are_byte_identical(tmp_path):
    from tilestream import harness
    from woof.config import soil_layer_count
    from woof.io.wrfout import WrfoutWriter, state_frame

    cfg = harness.make_config(16, 12, 6)
    state = harness.make_state(cfg)
    when = datetime(2026, 1, 1, 12)
    sync = tmp_path / "sync"
    frame = state_frame(state, include_diagnostic_pressure=True)
    with WrfoutWriter(sync, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx,
                      dy=cfg.dy, title="snapshot test", global_attrs={},
                      soil_layers=soil_layer_count(cfg),
                      field_schema=frame) as tape:
        tape.write_frame(when.strftime("%Y-%m-%d_%H:%M:%S"), frame)

    direct = _writer(cfg)
    direct.submit(tmp_path / "direct", when, state)
    direct.close()
    assert direct.receipt()["d2h"]["mode"] == "direct"
    assert direct.receipt()["d2h"]["reason"] == "unpriced"

    _price_room()
    snap = _writer(cfg)
    snap.submit(tmp_path / "snapshot", when, state)
    snap.submit(tmp_path / "snapshot2", when + timedelta(hours=1), state)
    snap.close()
    receipt = snap.receipt()
    assert receipt["d2h"]["mode"] == "snapshot", receipt
    assert receipt["stalls"]["frames_by_mode"] == {"snapshot": 2}
    assert receipt["identity"].get("streamed") == 2
    assert (tmp_path / "snapshot").read_bytes() == (tmp_path / "direct").read_bytes()
    # The synchronous frame omits nothing the async one writes.
    import netCDF4
    with netCDF4.Dataset(sync) as a, netCDF4.Dataset(tmp_path / "snapshot") as b:
        for name in a.variables:
            assert np.asarray(a.variables[name][:]).tobytes() == \
                np.asarray(b.variables[name][:]).tobytes(), name


def test_snapshot_producer_runs_ahead_of_the_host_copy(tmp_path):
    import cupy as cp
    from tilestream import harness

    cfg = harness.make_config(16, 12, 6)
    state = harness.make_state(cfg)
    when = datetime(2026, 1, 1, 12)
    reference = tmp_path / "reference"
    _price_room()
    first = _writer(cfg)
    first.submit(reference, when, state)
    first.close()

    spin = _spin_kernel()
    # Load the mutation's own kernels NOW (CuPy specializes them by rank):
    # a first-time module load synchronizes the whole device and would
    # look exactly like the fence this test is about.  Both are exact no-ops.
    state.u += cp.float32(0.0)
    state.thp *= cp.float32(1.0)
    cp.cuda.Device().synchronize()
    writer = _writer(cfg)
    writer.submit(tmp_path / "warm", when, state)    # warm the frame's kernels
    writer.drain()
    try:
        # Hold the side stream ~1-2 s behind the producer.
        with writer.stream:
            spin((1,), (1,), (np.int64(3_000_000_000),))
        path = tmp_path / "held"
        writer.submit(path, when, state)
        assert writer.receipt()["d2h"]["mode"] == "snapshot"
        # The model moves on at once, on the producer stream.
        state.u += cp.float32(1.0)
        state.thp *= cp.float32(2.0)
        moved = cp.cuda.Event()
        moved.record(cp.cuda.get_current_stream())
        moved.synchronize()
        assert not writer.stream.done, (
            "the producer's mutation waited for the held host copy")
        writer.drain()
    finally:
        writer.close()
    assert path.read_bytes() == reference.read_bytes()


def test_direct_mode_still_fences_the_producer(tmp_path):
    import cupy as cp
    from tilestream import harness

    cfg = harness.make_config(16, 12, 6)
    state = harness.make_state(cfg)
    spin = _spin_kernel()
    writer = _writer(cfg)
    writer.submit(tmp_path / "warm", datetime(2026, 1, 1), state)
    writer.drain()
    try:
        with writer.stream:
            spin((1,), (1,), (np.int64(1_000_000_000),))
        writer.submit(tmp_path / "direct", datetime(2026, 1, 1), state)
        moved = cp.cuda.Event()
        moved.record(cp.cuda.get_current_stream())
        moved.synchronize()
        assert writer.stream.done, "a direct frame must fence the producer"
    finally:
        writer.close()


def test_failed_snapshot_allocation_falls_back_to_direct(tmp_path, monkeypatch):
    import cupy as cp
    from tilestream import harness

    cfg = harness.make_config(16, 12, 6)
    state = harness.make_state(cfg)
    _price_room()
    writer = _writer(cfg)
    real_empty = cp.empty

    def no_room(*args, **kwargs):
        raise cp.cuda.memory.OutOfMemoryError(1, 0)

    monkeypatch.setattr(cp, "empty", no_room)
    try:
        writer.submit(tmp_path / "a", datetime(2026, 1, 1), state)
    finally:
        monkeypatch.setattr(cp, "empty", real_empty)
    writer.submit(tmp_path / "b", datetime(2026, 1, 1, 1), state)
    writer.close()
    receipt = writer.receipt()
    assert receipt["d2h"]["mode"] == "direct"
    assert receipt["d2h"]["reason"].startswith("device-allocation-failed")
    assert receipt["stalls"]["frames_by_mode"] == {"direct": 2}
    # Nothing stays reserved against the plan.
    assert history_staging.priced_plan(cp.cuda.Device().id)["reserved_bytes"] == 0


def test_closing_a_writer_returns_its_snapshot_and_pinned_blocks(tmp_path):
    import cupy as cp
    from tilestream import harness

    cfg = harness.make_config(16, 12, 6)
    state = harness.make_state(cfg)
    _price_room()
    writer = _writer(cfg)
    writer.submit(tmp_path / "a", datetime(2026, 1, 1), state)
    device = cp.cuda.Device().id
    assert history_staging.priced_plan(device)["reserved_bytes"] > 0
    writer.close()
    assert history_staging.priced_plan(device)["reserved_bytes"] == 0
    assert history_staging.free_pinned_pool() > 0
    assert cp.get_default_pinned_memory_pool().n_free_blocks() == 0
