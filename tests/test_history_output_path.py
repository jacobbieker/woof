"""History output off the model loop (woof.io.history_staging and friends).

CPU tier.  Pinned here:

* the Rust engine hashes each tape WHILE writing it, and that digest is the
  SHA-256 of the published file, so the output identity reads no payload;
  ``WOOF_OUTPUT_IDENTITY_VERIFY`` re-reads and refuses a disagreement; an
  older library falls back to the re-read, by name;
* the final name appears only after the identity is complete;
* the netCDF lock covers the Rust engine's validation only, so two domains'
  Rust writes overlap, while netCDF4-engine sessions stay serial;
* queue depth: config, env, validation and the page-locked budget cap;
* the device snapshot is admitted only inside the priced plan, by name;
* a ``[tiles]`` snapshot frame is not drained after submit, a views frame
  is, and the snapshot's REFL_10CM is a copy;
* the hex asynchronous writer writes write_history's bytes, in order, and
  reports its first failure.

The GPU tier is tests/test_history_output_path_gpu.py.
"""
from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from woof import output_identity
from woof.io import history_staging, nc_writer_bridge
from woof.io.wrfout import WrfoutWriter
from test_wrfout import _manual_async_writer, _queue_cpu_ticket


def _fields():
    return {"T": np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 0.5,
            "MU": np.arange(12, dtype=np.float32).reshape(3, 4)}


def _streamed_digest_supported() -> bool:
    try:
        return nc_writer_bridge._bind_digest(nc_writer_bridge.load())
    except Exception:  # noqa: BLE001 - no library
        return False


needs_digest = pytest.mark.skipif(
    not _streamed_digest_supported(),
    reason="the loaded netcdf-writer predates the streamed digest "
           "(build tools/rustwx and point WOOF_NCWRITE_BRIDGE at it)")


def _async_writer_one_frame(tmp_path, fields=None):
    writer = _manual_async_writer(Event())
    writer.nx, writer.ny, writer.nz = 4, 3, 2
    path = tmp_path / "wrfout_d01_frame"
    _queue_cpu_ticket(writer, path, fields=fields or _fields())
    writer.close()
    return writer, path


# ---------------------------------------------------------------------------
# hash while writing
# ---------------------------------------------------------------------------

@needs_digest
def test_streamed_identity_is_the_file_digest_and_reads_no_payload(
        tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("the identity re-read the payload")

    monkeypatch.setattr(output_identity, "_hash_open_file", forbidden)
    monkeypatch.delenv(output_identity.IDENTITY_VERIFY_ENV, raising=False)
    writer, path = _async_writer_one_frame(tmp_path)
    proof, = writer.completed_records
    data = path.read_bytes()
    assert proof.sha256 == hashlib.sha256(data).hexdigest()
    assert proof.size == len(data)
    assert writer._identity_modes == {"streamed": 1}
    monkeypatch.undo()
    # file_record reuses the writer's proof for the untouched file.
    assert output_identity.file_record(path, completed=proof) == proof.record()


@needs_digest
def test_async_tape_is_byte_identical_to_the_plain_writer(tmp_path):
    reference = tmp_path / "reference"
    with WrfoutWriter(reference, nx=4, ny=3, nz=2, dx=1., dy=1., title="test",
                      soil_layers=4, field_schema=_fields()) as tape:
        tape.write_frame("1974-04-03_12:00:00", _fields())
    writer, path = _async_writer_one_frame(tmp_path)
    assert path.read_bytes() == reference.read_bytes()


@needs_digest
def test_strict_verify_reads_back_and_refuses_a_wrong_digest(
        tmp_path, monkeypatch):
    monkeypatch.setenv(output_identity.IDENTITY_VERIFY_ENV, "1")
    writer, path = _async_writer_one_frame(tmp_path)
    assert writer._identity_modes == {"streamed+verified": 1}
    assert writer.completed_records[0].sha256 == hashlib.sha256(
        path.read_bytes()).hexdigest()

    tape = WrfoutWriter(tmp_path / "wrong", nx=4, ny=3, nz=2, dx=1., dy=1.,
                        title="test", soil_layers=4, field_schema=_fields(),
                        _retain_identity_handle=True, _expect_frames=1)
    tape.write_frame("1974-04-03_12:00:00", _fields())
    tape.close()
    size, _digest = tape.stream_identity
    tape.stream_identity = (size, "0" * 64)
    with pytest.raises(output_identity.OutputChangedError, match="hashed"):
        tape.complete_output_identity()
    assert not (tmp_path / "wrong").exists()


def test_library_without_the_digest_falls_back_to_the_reread(
        tmp_path, monkeypatch):
    monkeypatch.setattr(nc_writer_bridge, "_bind_digest", lambda _lib: False)
    writer, path = _async_writer_one_frame(tmp_path)
    proof, = writer.completed_records
    assert proof.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert writer._identity_modes["reread"] == 1
    assert "predates the streamed digest" in writer._identity_modes["reread_reason"]


@needs_digest
def test_frame_written_out_of_file_order_rereads(tmp_path):
    from woof.io.classic_tape import ClassicTape

    tape = ClassicTape(tmp_path / "tape", expect_records=1)
    tape.createDimension("Time", None)
    tape.createDimension("x", 3)
    a = tape.createVariable("a", "f4", ("Time", "x"))
    b = tape.createVariable("b", "f4", ("Time", "x"))
    b[0] = np.ones(3, np.float32)          # second variable first
    a[0] = np.zeros(3, np.float32)
    tape.close()
    assert tape.stream_identity is None
    assert "did not continue" in tape.stream_identity_refusal


@needs_digest
def test_final_name_appears_only_after_the_identity(tmp_path, monkeypatch):
    capture = output_identity.completed_file_record
    seen = []

    def observe(candidate, **kwargs):
        final = tmp_path / "wrfout_d01_frame"
        seen.append((Path(candidate).exists(), final.exists(),
                     kwargs.get("streamed") is not None))
        return capture(candidate, **kwargs)

    monkeypatch.setattr(output_identity, "completed_file_record", observe)
    writer, path = _async_writer_one_frame(tmp_path)
    assert seen == [(True, False, True)]
    assert path.exists() and writer.paths == [path]


# ---------------------------------------------------------------------------
# the netCDF lock
# ---------------------------------------------------------------------------

@needs_digest
def test_rust_engine_writes_of_two_domains_overlap_and_validate_locked(
        tmp_path, monkeypatch):
    import woof.io.wrfout as wrfout
    from woof.io.netcdf_serialization import NETCDF4_IO_LOCK

    monkeypatch.setenv(wrfout.WRFOUT_WRITER_ENV, "rust")
    intervals = {}
    both = threading.Barrier(2)
    original_write = nc_writer_bridge.ClassicWriter.write_record

    def slow_write(self, recno, varid, values):
        name = Path(self.path).name
        if name not in intervals:
            # The first slab of each tape waits for the other domain's.
            intervals[name] = [time.perf_counter(), None]
            both.wait(timeout=10)
        original_write(self, recno, varid, values)
        intervals[name][1] = time.perf_counter()

    validation_locked = []
    original_validate = wrfout.validate_wrfout_file

    def validate(path, **kwargs):
        validation_locked.append(NETCDF4_IO_LOCK.locked())
        return original_validate(path, **kwargs)

    monkeypatch.setattr(nc_writer_bridge.ClassicWriter, "write_record", slow_write)
    monkeypatch.setattr(wrfout, "validate_wrfout_file", validate)
    abort = Event()
    first = _manual_async_writer(abort)
    second = _manual_async_writer(abort)
    for writer in (first, second):
        writer.nx, writer.ny, writer.nz = 4, 3, 2
    _queue_cpu_ticket(first, tmp_path / "d01", fields=_fields())
    _queue_cpu_ticket(second, tmp_path / "d02", fields=_fields())
    first.close()
    second.close()
    assert len(intervals) == 2
    (a0, a1), (b0, b1) = intervals.values()
    assert a0 < b1 and b0 < a1, "the two Rust sessions did not overlap"
    assert validation_locked == [True, True]
    assert (tmp_path / "d01").read_bytes() == (tmp_path / "d02").read_bytes()


# ---------------------------------------------------------------------------
# queue depth and budgets
# ---------------------------------------------------------------------------

def test_queue_depth_resolution(monkeypatch):
    from woof.io.history_selection import HistorySelection

    monkeypatch.delenv(history_staging.QUEUE_DEPTH_ENV, raising=False)
    assert history_staging.resolve_queue_depth(None) == {
        "requested": 1, "source": "default"}
    selection = HistorySelection.from_mapping({"history_queue_depth": 3})
    assert history_staging.resolve_queue_depth(selection) == {
        "requested": 3, "source": "[output] history_queue_depth"}
    monkeypatch.setenv(history_staging.QUEUE_DEPTH_ENV, "2")
    assert history_staging.resolve_queue_depth(selection)["requested"] == 2
    for bad in (0, 9, "two", True, 1.5):
        with pytest.raises(ValueError, match="history_queue_depth"):
            HistorySelection.from_mapping({"history_queue_depth": bad})
    monkeypatch.setenv(history_staging.QUEUE_DEPTH_ENV, "0")
    with pytest.raises(ValueError, match=history_staging.QUEUE_DEPTH_ENV):
        history_staging.resolve_queue_depth(None)


def test_queue_depth_is_capped_by_the_pinned_budget():
    mib = 2 ** 20
    assert history_staging.effective_queue_depth(1, 10 ** 12) == (1, None)
    assert history_staging.effective_queue_depth(
        4, 100 * mib, budget=(10 * 100 * mib, 0)) == (4, None)
    depth, reason = history_staging.effective_queue_depth(
        4, 100 * mib, budget=(4 * 100 * mib, 0))
    assert depth == 2 and reason.startswith("depth-capped-by-pinned-budget")
    depth, reason = history_staging.effective_queue_depth(
        4, 100 * mib, budget=(100 * mib, 0))
    assert depth == 1 and "using 1" in reason


def test_writer_enforces_the_capped_depth(tmp_path):
    import woof.io.wrfout as wrfout

    writer = _manual_async_writer(Event(),
                                  ticket_queue=wrfout.AsyncDomainWrfoutWriter
                                  ._new_ticket_queue(3))
    assert writer._queue.maxsize == 3
    writer._depth_cap = 1
    gate = Event()
    original = wrfout.WrfoutWriter

    class Held(original):
        def write_frame(self, *args, **kwargs):
            gate.wait(10)
            return super().write_frame(*args, **kwargs)

    wrfout.WrfoutWriter = Held
    try:
        _queue_cpu_ticket(writer, tmp_path / "a")      # taken by the worker
        deadline = time.time() + 5
        while writer._queue.qsize() and time.time() < deadline:
            time.sleep(0.01)
        _queue_cpu_ticket(writer, tmp_path / "b")      # queued: 1 == cap
        admitted = Event()

        def third():
            _queue_cpu_ticket(writer, tmp_path / "c")
            admitted.set()

        producer = threading.Thread(target=third)
        producer.start()
        assert not admitted.wait(0.3), "a capped queue admitted a second frame"
        gate.set()
        assert admitted.wait(10)
        producer.join()
        writer.close()
    finally:
        gate.set()
        wrfout.WrfoutWriter = original
    assert [p.name for p in writer.paths] == ["a", "b", "c"]


def test_device_snapshot_admission_is_named(monkeypatch):
    from woof.core.devices_memory import FRAME_SNAPSHOT_LIMIT_BYTES

    history_staging.forget_priced_plans()
    try:
        monkeypatch.delenv(history_staging.ASYNC_HISTORY_ENV, raising=False)
        assert history_staging.reserve_device_snapshot(0, 10)[:2] == (
            False, "unpriced")
        history_staging.record_priced_plan(
            {"free_bytes": 1000, "need_bytes": 600, "what": "test"}, device=0)
        assert history_staging.reserve_device_snapshot(0, 300) == (
            True, None, "priced-plan")
        ok, reason, _ = history_staging.reserve_device_snapshot(0, 200)
        assert not ok and reason.startswith("outside-priced-plan")
        history_staging.release_device_snapshot(0, 300)
        assert history_staging.reserve_device_snapshot(0, 400)[0]
        history_staging.record_priced_plan(
            {"free_bytes": 2 * FRAME_SNAPSHOT_LIMIT_BYTES, "need_bytes": 0},
            device=1)
        ok, reason, _ = history_staging.reserve_device_snapshot(
            1, FRAME_SNAPSHOT_LIMIT_BYTES + 1)
        assert not ok and reason.startswith("over-cap")
        monkeypatch.setenv(history_staging.ASYNC_HISTORY_ENV, "off")
        assert history_staging.reserve_device_snapshot(1, 1)[:2] == (
            False, "disabled")
        monkeypatch.setenv(history_staging.ASYNC_HISTORY_ENV, "on")
        assert history_staging.reserve_device_snapshot(
            5, 10, live_free=2 ** 31) == (True, None, "live-free-memory")
        assert history_staging.reserve_device_snapshot(
            6, 10, live_free=100)[:2] == (False, "no-live-free-memory")
        monkeypatch.setenv(history_staging.ASYNC_HISTORY_ENV, "maybe")
        with pytest.raises(ValueError, match=history_staging.ASYNC_HISTORY_ENV):
            history_staging.async_history_mode()
    finally:
        history_staging.forget_priced_plans()


def test_host_snapshot_admission_is_named(monkeypatch):
    monkeypatch.delenv(history_staging.ASYNC_HISTORY_ENV, raising=False)
    assert history_staging.admit_host_snapshot(10, budget=(100, 1)) == (True, None)
    ok, reason = history_staging.admit_host_snapshot(1000, budget=(100, 1))
    assert not ok and reason.startswith("insufficient-host-ram")
    monkeypatch.setenv(history_staging.ASYNC_HISTORY_ENV, "off")
    assert history_staging.admit_host_snapshot(10, budget=(100, 1)) == (
        False, "disabled")


# ---------------------------------------------------------------------------
# [tiles] host-store frames
# ---------------------------------------------------------------------------

def _streamed_writers(monkeypatch, tmp_path, mode):
    import woof.io.wrfout as wrfout
    import woof.runtime as runtime

    calls = []

    class CapturingWriter:
        def __init__(self, **_kwargs):
            self.pending = 0
            self.paths = []
            self.global_attrs = {}
            self.submissions = []
            self.pending_work_bytes = 0

        def submit(self, *args, **kwargs):
            calls.append("submit")
            self.submissions.append((args, kwargs))

        def drain_staging(self):
            calls.append("drain_staging")

        def drain(self):
            calls.append("drain")

        def close(self):
            pass

    store = {"U": np.full((1, 2, 2), 3.0, np.float32)}

    class Streamed:
        history_snapshot_mode = None

        def __init__(self):
            self.snapshot_requests = []

        def _ranked_road(self):
            return False

        def history_fields(self, *, snapshot=False):
            calls.append("history_fields")
            self.snapshot_requests.append(snapshot)
            if self.history_snapshot_mode is None:
                self.history_snapshot_mode = {
                    "mode": mode, "reason": None if mode == "snapshot"
                    else "insufficient-host-ram: test", "snapshot_bytes": 16}
            if mode == "snapshot":
                return {name: value.copy() for name, value in store.items()}
            return dict(store)

        def add_store_guard(self, guard):
            raise AssertionError("not the ranked road")

    monkeypatch.setattr(runtime, "_metadata_frame", lambda *a: {})
    monkeypatch.setattr(runtime, "_global_wrf_attrs", lambda *a, **k: {})
    monkeypatch.setattr(wrfout, "AsyncDomainWrfoutWriter", CapturingWriter)
    monkeypatch.setattr(wrfout, "streamed_carrier_provenance_attrs",
                        lambda _streamed: {})
    cfg = SimpleNamespace(
        grid_id=1, start_time=None,
        run=SimpleNamespace(nx=2, ny=2, nz=1, dx=1.0, dy=1.0,
                            sf_surface_physics=0, num_soil_layers=4))
    streamed = Streamed()
    state = SimpleNamespace(_streamed_domain=streamed, physics=None)
    case = SimpleNamespace(static_fields={}, geog_selection=None,
                           initial_result=SimpleNamespace(
                               coord=SimpleNamespace(hybrid_opt=2, etac=0.2)))
    node = SimpleNamespace(cfg=cfg, grid=object(), state=state,
                           clock=SimpleNamespace(
                               tick_den=1, spec=SimpleNamespace(dt_fp32=30.0)))
    model = SimpleNamespace(_prepared_by_grid_id={1: case},
                            walk_parent_first=lambda: (node,))
    writers = wrfout.PerDomainWrfoutWriters(
        model, tmp_path, start_time=datetime(1974, 4, 3, 12), title="test")
    return writers, node, streamed, calls


def test_tiles_snapshot_frame_is_not_drained_after_submit(monkeypatch, tmp_path):
    monkeypatch.delenv(history_staging.ASYNC_HISTORY_ENV, raising=False)
    writers, node, streamed, calls = _streamed_writers(
        monkeypatch, tmp_path, "snapshot")
    refl = np.full((1, 2, 2), 9.0, np.float32)
    writers.submit(node, 0, refl_field=refl)
    writers.submit(node, 900, refl_field=refl)
    # The previous frame's staging is consumed BEFORE the next frame is
    # taken (its buffers are refilled), never right after a submit.
    assert calls == ["drain_staging", "history_fields", "submit",
                     "drain_staging", "history_fields", "submit"]
    assert streamed.snapshot_requests == [True, True]
    submitted = writers._writers[1].submissions[0][1]["refl_field"]
    assert submitted is not refl and np.array_equal(submitted, refl)
    receipt = writers.history_writes_receipt()
    assert receipt["schema"] == history_staging.RECEIPT_SCHEMA


def test_tiles_views_frame_is_drained_after_submit(monkeypatch, tmp_path):
    monkeypatch.delenv(history_staging.ASYNC_HISTORY_ENV, raising=False)
    writers, node, streamed, calls = _streamed_writers(
        monkeypatch, tmp_path, "views")
    refl = np.full((1, 2, 2), 9.0, np.float32)
    writers.submit(node, 0, refl_field=refl)
    writers.submit(node, 900, refl_field=refl)
    assert calls == ["drain_staging", "history_fields", "submit", "drain_staging",
                     "history_fields", "submit", "drain_staging"]
    assert writers._writers[1].submissions[0][1]["refl_field"] is refl


def test_tiles_switch_off_restores_the_synchronous_path(monkeypatch, tmp_path):
    monkeypatch.setenv(history_staging.ASYNC_HISTORY_ENV, "off")
    writers, node, streamed, calls = _streamed_writers(
        monkeypatch, tmp_path, "snapshot")
    writers.submit(node, 0)
    assert calls == ["history_fields", "submit", "drain_staging"]
    assert streamed.snapshot_requests == [False]


# ---------------------------------------------------------------------------
# hex
# ---------------------------------------------------------------------------

class _Mesh:
    def __init__(self, cells=5, edges=7):
        self.arrays = {"latCell": np.linspace(-1, 1, cells),
                       "lonCell": np.linspace(0, 2, cells),
                       "latEdge": np.linspace(-1, 1, edges),
                       "cellsOnEdge": np.zeros((edges, 2), np.int32)}
        self.dimensions = {"nCells": cells, "nEdges": edges}


def _hex_fields(seed):
    rng = np.random.default_rng(seed)
    return {"theta": rng.random((3, 5)).astype(np.float32),
            "u": rng.random((3, 7)).astype(np.float32),
            "surface_pressure": rng.random(5)}


def test_hex_async_writer_writes_write_history_bytes_in_order(tmp_path):
    from woof.hex.output import AsyncHistoryWriter, write_history

    mesh = _Mesh()
    when = [datetime(2026, 1, 1, h) for h in range(3)]
    for index, valid in enumerate(when):
        write_history(tmp_path / f"sync{index}.nc", mesh, _hex_fields(index), valid,
                      initial_time=when[0])
    writer = AsyncHistoryWriter(depth=1)
    order = []
    for index, valid in enumerate(when):
        fields = _hex_fields(index)
        writer.submit_history(tmp_path / f"async{index}.nc", mesh, fields, valid,
                              initial_time=when[0])
        # The caller may reuse its arrays at once: the writer holds a copy.
        for value in fields.values():
            value[...] = -1
        writer.submit(lambda index=index: order.append(index), label=f"mark{index}")
    writer.close()
    assert order == [0, 1, 2]
    for index in range(3):
        assert ((tmp_path / f"async{index}.nc").read_bytes()
                == (tmp_path / f"sync{index}.nc").read_bytes())
    assert writer.receipt["mode"] == "async" and len(writer.receipt["frames"]) == 6


def test_hex_async_writer_reports_its_first_failure_and_skips_later_jobs(tmp_path):
    from woof.hex.output import AsyncHistoryWriteError, AsyncHistoryWriter

    writer = AsyncHistoryWriter(depth=2)
    ran = []

    def fail():
        raise OSError("disk full")

    writer.submit(fail, label="bad")
    writer.submit(lambda: ran.append("later"), label="later")
    with pytest.raises(AsyncHistoryWriteError, match="disk full"):
        writer.drain()
    assert ran == []
    assert not issubclass(AsyncHistoryWriteError, RuntimeError)
    assert isinstance(writer.clear_failure(), OSError)
    writer.submit(lambda: ran.append("after"), label="after")
    writer.close()
    assert ran == ["after"]


def test_hex_async_writer_holds_the_netcdf_lock_while_writing(tmp_path):
    from woof.hex.output import AsyncHistoryWriter
    from woof.io.netcdf_serialization import NETCDF4_IO_LOCK

    writer = AsyncHistoryWriter()
    seen = []
    writer.submit(lambda: seen.append(NETCDF4_IO_LOCK.locked()))
    writer.close()
    assert seen == [True]


def test_snapshot_cap_is_per_writer_and_the_plan_room_per_card(monkeypatch):
    from woof.core.devices_memory import FRAME_SNAPSHOT_LIMIT_BYTES as cap

    monkeypatch.delenv(history_staging.ASYNC_HISTORY_ENV, raising=False)
    history_staging.forget_priced_plans()
    try:
        history_staging.record_priced_plan(
            {"free_bytes": 4 * cap, "need_bytes": 0}, device=0)
        # Two writers each under the cap share the card's room.
        assert history_staging.reserve_device_snapshot(0, cap - 1)[0]
        assert history_staging.reserve_device_snapshot(
            0, cap - 1, writer_bytes=0)[0]
        ok, reason, _ = history_staging.reserve_device_snapshot(
            0, 2, writer_bytes=cap - 1)
        assert not ok and reason.startswith("over-cap")
    finally:
        history_staging.forget_priced_plans()


@needs_digest
def test_switch_off_restores_the_session_lock_and_the_reread(tmp_path, monkeypatch):
    import woof.io.wrfout as wrfout

    monkeypatch.setenv(history_staging.ASYNC_HISTORY_ENV, "off")
    monkeypatch.setenv(wrfout.WRFOUT_WRITER_ENV, "rust")
    writer, path = _async_writer_one_frame(tmp_path)
    assert writer._identity_modes["reread"] == 1
    assert writer.completed_records[0].sha256 == hashlib.sha256(
        path.read_bytes()).hexdigest()


def test_les_stamp_snapshot_is_a_host_copy_taken_at_capture():
    from woof.hex.drivers.run_cuda_v841_forecast import _les_stamp_snapshot

    kdiff = np.ones((3, 4), np.float32)
    live = SimpleNamespace(
        n_cells_real=3,
        closure=SimpleNamespace(
            config=SimpleNamespace(les_surface="x"),
            last=SimpleNamespace(kdiff=kdiff, eddy_visc_vert=kdiff * 2),
            tke=None, cp=SimpleNamespace(asnumpy=lambda a: np.array(a, copy=True))))
    frozen = _les_stamp_snapshot(live)
    kdiff[...] = 7.0                      # the model steps on
    assert np.all(frozen.closure.last.kdiff == 1.0)
    assert frozen.closure.tke is None and frozen.n_cells_real == 3
    assert _les_stamp_snapshot(None) is None
