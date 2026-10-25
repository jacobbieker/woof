"""Asynchronous checkpoints, the streamed member writer, and their caches.

CPU tier: the tree fixture from ``test_restart`` (NumPy-backed states)
drives the real ``write_tree_restart`` through the asynchronous writer, so
the publish ordering, the at-most-one-in-flight rule, the drain, a failure
while writing, history ordering and every named fallback are exercised on
real archives.  A host-side stand-in for a device array exercises the
one-member-at-a-time staging path and pins it byte-identical to the
materialising writer it replaced.

GPU tier: the same on CuPy states, including a state mutated on the
producer stream immediately after the snapshot was queued -- the archive
must hold the values of the checkpoint instant.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
import time
import zipfile
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from test_restart import (
    _LIFECYCLE_FREE_CHILD_DIGEST, _LIFECYCLE_FREE_ROOT_DIGEST,
    _PINNED_UNDER_MEASURED_CLOCK, _canonical_member_digest, _cfg, _extension_state, _fill_serialized,
    _fill_setup, _historical_setup_fingerprint, _sealed_tree_fixture,
    _shim_state)
from woof import state_serialization_contract as contract
from woof.io import async_checkpoint, restart
from woof.io.async_checkpoint import (
    ASYNC_CHECKPOINTS_ENV, CheckpointWriteError, CheckpointWriter,
    resolve_checkpoint_mode)

ASYNC = {"requested": "async", "source": "test"}
SYNC = {"requested": "sync", "source": "test"}


def _writer(model, mode=ASYNC) -> CheckpointWriter:
    writer = CheckpointWriter(model, mode)
    model._checkpoint_writer = writer
    model._checkpoint_receipt = writer.receipt
    return writer


def _tree(monkeypatch):
    # Upstream 1.1.0 pins the lifecycle-free digests under the measured
    # terrain clock (tests/test_background_checkpoint.py does the same).
    return _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31,
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)


def _members(directory, grid_id):
    return sorted(Path(directory).glob(f"gpuwmrst_d{grid_id:02d}_*.npz"))


class _RootGate:
    """Hold the writer thread just before it writes the ROOT member."""

    def __init__(self, monkeypatch, *, fail_root=False):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail_root = fail_root
        real = restart._publish_npz

        def gated(path, payload, **kwargs):
            if Path(path).name.startswith("gpuwmrst_d01_"):
                self.entered.set()
                assert self.release.wait(30), "the test never released"
                if self.fail_root:
                    raise OSError("injected root publication failure")
            return real(path, payload, **kwargs)

        monkeypatch.setattr(restart, "_publish_npz", gated)


# ---------------------------------------------------------------------------
# publish ordering
# ---------------------------------------------------------------------------

def test_an_async_checkpoint_is_a_resume_point_only_once_durable(
        monkeypatch, tmp_path):
    model, start = _tree(monkeypatch)
    writer = _writer(model)
    gate = _RootGate(monkeypatch)
    published = []
    valid = start + timedelta(seconds=3600)

    returned = restart.write_tree_restart(
        tmp_path, model, valid,
        on_published=lambda path, info: published.append((path, info)))
    # It returns the root's name before the root exists.
    assert returned.name.startswith("gpuwmrst_d01_")
    assert gate.entered.wait(30)
    assert not returned.exists()
    # The child is durable; the root -- the set's commit marker -- is not,
    # so nothing anywhere names this set as a resume point yet.
    assert len(_members(tmp_path, 2)) == 1
    assert _members(tmp_path, 1) == []
    assert writer.poll() is None
    assert model._last_checkpoint is None and published == []

    gate.release.set()
    assert writer._job.done.wait(30)
    # Durable, but publication belongs to the model thread.
    assert model._last_checkpoint is None and published == []
    root = writer.poll()
    assert root == returned and root.exists()
    assert model._last_checkpoint == root
    assert [path for path, _info in published] == [root]
    info = published[0][1]
    assert info["mode"] == "async" and info["status"] == "published"
    assert info["background_write_seconds"] >= 0.0
    assert writer.receipt["ran"] == "async"


def test_async_archives_are_the_synchronous_archives(monkeypatch, tmp_path):
    """Same members, same order, same header: the pinned historical digests."""
    model, start = _tree(monkeypatch)
    valid = start + timedelta(seconds=3600)
    sync_root = restart.write_tree_restart(tmp_path / "sync", model, valid)

    writer = _writer(model)
    pending = restart.write_tree_restart(tmp_path / "async", model, valid)
    writer.close()
    assert pending == model._last_checkpoint
    async_root = model._last_checkpoint
    assert async_root.parent == tmp_path / "async"
    for grid_id, pinned in ((1, _LIFECYCLE_FREE_ROOT_DIGEST),
                            (2, _LIFECYCLE_FREE_CHILD_DIGEST)):
        sync_member, = _members(tmp_path / "sync", grid_id)
        async_member, = _members(tmp_path / "async", grid_id)
        assert _canonical_member_digest(sync_member) == pinned
        assert _canonical_member_digest(async_member) == pinned
        with zipfile.ZipFile(sync_member) as a, zipfile.ZipFile(async_member) as b:
            assert a.namelist() == b.namelist()
    assert sync_root != async_root


# ---------------------------------------------------------------------------
# at most one in flight
# ---------------------------------------------------------------------------

def test_at_most_one_checkpoint_is_in_flight_and_the_wait_is_logged(
        monkeypatch, tmp_path, capsys):
    model, start = _tree(monkeypatch)
    writer = _writer(model)
    gate = _RootGate(monkeypatch)
    published = []
    record = lambda path, info: published.append(path)   # noqa: E731
    first = start + timedelta(seconds=3600)
    second = start + timedelta(seconds=7200)

    restart.write_tree_restart(tmp_path, model, first, on_published=record)
    assert gate.entered.wait(30)
    first_job = writer._job
    done = threading.Event()

    def second_checkpoint():
        restart.write_tree_restart(tmp_path, model, second,
                                   on_published=record)
        done.set()

    thread = threading.Thread(target=second_checkpoint, daemon=True)
    thread.start()
    time.sleep(0.4)
    # The second checkpoint is due while the first is still writing: it
    # waits, and has taken no snapshot.
    assert not done.is_set()
    assert writer._job is first_job
    assert published == []

    gate.release.set()
    thread.join(30)
    assert done.is_set()
    # The first was published BEFORE the second's snapshot was taken.
    assert len(published) == 1 and published[0].name.startswith(
        "gpuwmrst_d01_" + first.strftime("%Y-%m-%d_%H_%M_%S"))
    assert writer._job is not first_job
    writer.close()
    assert len(published) == 2 and published[1].name.startswith(
        "gpuwmrst_d01_" + second.strftime("%Y-%m-%d_%H_%M_%S"))
    assert model._last_checkpoint == published[1]
    assert writer.receipt["in_flight_waits"] == 1
    assert writer.receipt["in_flight_wait_seconds"] > 0.0
    rows = writer.receipt["checkpoints"]
    assert rows[1]["waited_for_previous_seconds"] > 0.0
    out = capsys.readouterr().out
    assert "is still being written" in out and "waited" in out


# ---------------------------------------------------------------------------
# drain on stop / failure
# ---------------------------------------------------------------------------

def test_close_drains_and_publishes_even_while_the_run_is_failing(
        monkeypatch, tmp_path):
    model, start = _tree(monkeypatch)
    writer = _writer(model)
    gate = _RootGate(monkeypatch)
    restart.write_tree_restart(tmp_path, model, start + timedelta(seconds=3600))
    assert gate.entered.wait(30)
    releaser = threading.Timer(0.3, gate.release.set)
    releaser.start()
    # The run failed AFTER the snapshot: the snapshot predates the failure,
    # so the drain finishes and publishes it.
    writer.close(RuntimeError("a later step failed"))
    assert model._last_checkpoint is not None and model._last_checkpoint.exists()
    assert len(_members(tmp_path, 1)) == len(_members(tmp_path, 2)) == 1
    assert model._checkpoint_writer is None
    assert writer.closed


def test_a_failed_write_publishes_nothing_and_keeps_the_previous_point(
        monkeypatch, tmp_path):
    model, start = _tree(monkeypatch)
    writer = _writer(model)
    restart.write_tree_restart(tmp_path, model, start + timedelta(seconds=3600))
    writer.wait_published()
    committed = model._last_checkpoint
    committed_files = sorted(tmp_path.iterdir())
    assert committed is not None

    gate = _RootGate(monkeypatch, fail_root=True)
    published = []
    restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=7200),
        on_published=lambda path, info: published.append(path))
    assert gate.entered.wait(30)
    gate.release.set()
    assert writer._job.done.wait(30)
    with pytest.raises(CheckpointWriteError, match="nothing of it was published"):
        writer.poll()
    # The half-written generation is gone -- its child included -- and the
    # earlier set is still the resume point.
    assert sorted(tmp_path.iterdir()) == committed_files
    assert model._last_checkpoint == committed
    assert published == []
    assert writer.receipt["checkpoints"][-1]["status"] == "failed"


def test_a_failed_write_during_a_failing_run_is_a_note_not_a_new_error(
        monkeypatch, tmp_path):
    model, start = _tree(monkeypatch)
    writer = _writer(model)
    gate = _RootGate(monkeypatch, fail_root=True)
    restart.write_tree_restart(tmp_path, model, start + timedelta(seconds=3600))
    gate.release.set()
    original = RuntimeError("the step failed")
    writer.close(original)
    assert any("in-flight checkpoint also failed" in note
               for note in original.__notes__)
    assert list(tmp_path.glob("gpuwmrst_*.npz")) == []
    assert model._last_checkpoint is None

    model2, _ = _tree(monkeypatch)
    writer2 = _writer(model2)
    gate2 = _RootGate(monkeypatch, fail_root=True)
    restart.write_tree_restart(tmp_path / "b", model2,
                               start + timedelta(seconds=3600))
    gate2.release.set()
    with pytest.raises(CheckpointWriteError):
        writer2.close()


def test_an_interrupted_drain_leaves_nothing_publishable(monkeypatch, tmp_path):
    """A second Ctrl-C while draining stops the wait AND the write."""
    model, start = _tree(monkeypatch)
    writer = _writer(model)
    gate = _RootGate(monkeypatch)
    restart.write_tree_restart(tmp_path, model, start + timedelta(seconds=3600))
    assert gate.entered.wait(30)
    job = writer._job

    def interrupted(**_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(writer, "wait_idle", interrupted)
    with pytest.raises(KeyboardInterrupt):
        writer.close(RuntimeError("the step failed"))
    assert job.cancelled.is_set()
    gate.release.set()
    assert job.done.wait(30)
    assert isinstance(job.error, InterruptedError)
    assert list(tmp_path.glob("gpuwmrst_*.npz")) == []
    assert model._last_checkpoint is None


# ---------------------------------------------------------------------------
# history ordering instead of a drain
# ---------------------------------------------------------------------------

class _OrderedHistory:
    """A history writer set with frames still in flight."""

    def __init__(self, *, barrier=True, fail=False):
        self.pending = 3
        self.drains = 0
        self.release = threading.Event()
        self.fail = fail
        if not barrier:
            self.durability_barrier = lambda: None

    def durability_barrier(self):
        def wait():
            assert self.release.wait(30)
            if self.fail:
                raise RuntimeError("a frame the checkpoint follows failed")
        return SimpleNamespace(wait=wait)

    def drain(self):
        self.drains += 1


def test_the_root_waits_for_the_history_it_is_ordered_after(
        monkeypatch, tmp_path):
    model, start = _tree(monkeypatch)
    history = _OrderedHistory()
    model._io_manager = history
    model._runtime_status.pending_d2h = 3
    writer = _writer(model)
    # Frames in flight are ORDERED work, not a refusal.
    pending = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))
    assert writer._job is not None and not pending.exists()
    deadline = time.monotonic() + 30
    while not _members(tmp_path, 2) and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.2)
    assert len(_members(tmp_path, 2)) == 1
    assert _members(tmp_path, 1) == [], "the root beat the frames it follows"
    history.release.set()
    writer.close()
    assert len(_members(tmp_path, 1)) == 1
    assert history.drains == 0, "the model thread drained the history"
    assert "history_order_wait_seconds" in writer.receipt["checkpoints"][-1]


def test_a_failed_frame_blocks_the_checkpoint_ordered_after_it(
        monkeypatch, tmp_path):
    model, start = _tree(monkeypatch)
    history = _OrderedHistory(fail=True)
    model._io_manager = history
    writer = _writer(model)
    restart.write_tree_restart(tmp_path, model, start + timedelta(seconds=3600))
    history.release.set()
    with pytest.raises(CheckpointWriteError, match="frame"):
        writer.close()
    assert list(tmp_path.glob("gpuwmrst_*.npz")) == []
    assert model._last_checkpoint is None


def test_a_writer_set_without_a_barrier_is_drained_as_before(
        monkeypatch, tmp_path):
    model, start = _tree(monkeypatch)
    history = _OrderedHistory(barrier=False)
    history.pending = 0
    model._io_manager = history
    writer = _writer(model)
    restart.write_tree_restart(tmp_path, model, start + timedelta(seconds=3600))
    assert writer._job is not None
    assert history.drains == 1
    writer.close()
    assert len(_members(tmp_path, 1)) == 1


def test_the_wrfout_barrier_waits_for_admitted_frames_only():
    from woof.io.wrfout import AsyncDomainWrfoutWriter, HistoryDurabilityBarrier

    writer = object.__new__(AsyncDomainWrfoutWriter)
    writer._condition = threading.Condition()
    writer._failure = None
    writer._abort_event = threading.Event()
    writer._thread = SimpleNamespace(is_alive=lambda: True)
    writer._admitted_count = 2
    writer._finished_count = 1
    barrier = HistoryDurabilityBarrier(((writer, writer.admitted_count),))
    done = threading.Event()
    thread = threading.Thread(target=lambda: (barrier.wait(), done.set()),
                              daemon=True)
    thread.start()
    time.sleep(0.2)
    assert not done.is_set()
    with writer._condition:
        # A frame admitted AFTER the barrier was taken is not waited for.
        writer._admitted_count = 5
        writer._finished_count = 2
        writer._condition.notify_all()
    thread.join(5)
    assert done.is_set()

    # A failure or abort AFTER the covered frames is not this barrier's:
    # the checkpoint whose history is complete is still publishable.
    with writer._condition:
        writer._finished_count = 3
        writer._first_undurable = 3
    writer._abort_event.set()
    barrier.wait()
    # One that covers the undurable frame refuses.
    covering = HistoryDurabilityBarrier(((writer, 3),))
    with pytest.raises(RuntimeError, match="aborted"):
        covering.wait()


# ---------------------------------------------------------------------------
# named fallbacks and the switch
# ---------------------------------------------------------------------------

def test_insufficient_host_ram_is_a_named_synchronous_fallback(
        monkeypatch, tmp_path, capsys):
    model, start = _tree(monkeypatch)
    writer = _writer(model)
    monkeypatch.setattr(async_checkpoint, "host_ram_available",
                        lambda: (1 << 20, 1 << 40))
    published = []
    root = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600),
        on_published=lambda path, info: published.append(info))
    assert root is not None and root.exists()
    assert model._last_checkpoint == root
    row = writer.receipt["checkpoints"][-1]
    assert (row["mode"], row["reason"]) == ("sync", "insufficient-host-ram")
    assert writer.receipt["fallbacks"] == {"insufficient-host-ram": 1}
    assert published[0]["mode"] == "sync"
    out = capsys.readouterr().out
    assert "insufficient-host-ram" in out and "host RAM" in out
    assert _canonical_member_digest(root) == _LIFECYCLE_FREE_ROOT_DIGEST


def test_the_synchronous_mode_writes_as_it_always_did(monkeypatch, tmp_path):
    model, start = _tree(monkeypatch)
    writer = _writer(model, SYNC)
    root = restart.write_tree_restart(tmp_path, model,
                                      start + timedelta(seconds=3600))
    assert root is not None and model._last_checkpoint == root
    row = writer.receipt["checkpoints"][-1]
    assert (row["mode"], row["reason"]) == ("sync", "disabled")
    assert writer.receipt["ran"] == "sync"
    assert writer.receipt["fallbacks"] == {}


def test_refusals_name_streamed_and_sealed_domains(monkeypatch):
    model, _start = _tree(monkeypatch)
    writer = _writer(model)
    nodes = tuple(model.walk_parent_first())
    assert writer.refusal(nodes, sealed_forcing_extension=False) is None
    assert writer.refusal(nodes, sealed_forcing_extension=True) == \
        "sealed-forcing-extension"
    model.root.state._streamed_domain = object()
    assert writer.refusal(nodes, sealed_forcing_extension=False) == \
        "streamed-domain"
    assert _writer(model, SYNC).refusal(
        nodes, sealed_forcing_extension=False) == "disabled"
    assert set(async_checkpoint.FALLBACK_REASONS) >= {
        "disabled", "streamed-domain", "sealed-forcing-extension",
        "multi-device-state", "insufficient-host-ram",
        "pinned-allocation-failed"}


def test_the_switch_resolves_env_then_config_then_default(monkeypatch):
    from woof.io.history_selection import FULL, HistorySelection

    monkeypatch.delenv(ASYNC_CHECKPOINTS_ENV, raising=False)
    assert resolve_checkpoint_mode(None) == {
        "requested": "async", "source": "default"}
    off = HistorySelection.from_mapping({"async_checkpoints": False})
    experiment = SimpleNamespace(output=off)
    assert resolve_checkpoint_mode(experiment) == {
        "requested": "sync", "source": "[output] async_checkpoints"}
    monkeypatch.setenv(ASYNC_CHECKPOINTS_ENV, "on")
    assert resolve_checkpoint_mode(experiment)["requested"] == "async"
    monkeypatch.setenv(ASYNC_CHECKPOINTS_ENV, "0")
    assert resolve_checkpoint_mode(None) == {
        "requested": "sync", "source": f"env {ASYNC_CHECKPOINTS_ENV}"}
    monkeypatch.setenv(ASYNC_CHECKPOINTS_ENV, "sometimes")
    with pytest.raises(ValueError, match=ASYNC_CHECKPOINTS_ENV):
        resolve_checkpoint_mode(None)

    # The key binds no identity: it is not a field of the selection.
    assert off == FULL and asdict(off) == asdict(FULL)
    assert "async_checkpoints" not in asdict(off)
    with pytest.raises(ValueError, match="true or false"):
        HistorySelection.from_mapping({"async_checkpoints": "no"})


# ---------------------------------------------------------------------------
# stage 1: the streamed member writer and the identity caches
# ---------------------------------------------------------------------------

class _FakeDevice:
    """A host stand-in for a device array: only ``.get`` reaches its bytes."""

    def __init__(self, array):
        self._array = np.ascontiguousarray(array)
        self.shape = self._array.shape
        self.dtype = self._array.dtype
        self.nbytes = self._array.nbytes
        self.ndim = self._array.ndim
        self.flags = self._array.flags
        self.gets = 0

    def get(self, out=None, stream=None, blocking=True):
        self.gets += 1
        if out is None:
            return self._array.copy()
        out[...] = self._array
        return out


class _FrozenClock:
    """Pin both clocks a byte comparison of two archives depends on."""

    @staticmethod
    def install(monkeypatch):
        import time as time_module

        class FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)

        class FixedTime:
            def __getattr__(self, name):
                return getattr(time_module, name)

            @staticmethod
            def time():
                return 1_760_000_000.0

        monkeypatch.setattr(restart, "datetime", FixedDatetime)
        monkeypatch.setattr(zipfile, "time", FixedTime())


def _device_state(monkeypatch, seed=7):
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, seed)
    devices = {}
    for name in restart.STATE_SERIALIZED_ATTRS:
        value = getattr(state, name, None)
        if value is not None:
            devices[name] = _FakeDevice(value)
            setattr(state, name, devices[name])
    return state, cfg, devices


def _materialising_write(path, state, cfg):
    """The writer this release replaced: every member on the host at once."""
    plan = restart._prepare_restart(path, state, cfg)
    payload = {restart._HEADER_KEY: restart._header_member(plan.header)}
    for key, value in plan.members:
        payload[key] = np.asarray(value.get() if hasattr(value, "get")
                                  else value)
    with Path(path).open("wb") as stream:
        np.savez(stream, **payload)
    return Path(path)


def test_the_streamed_writer_is_byte_identical_to_the_materialising_one(
        monkeypatch, tmp_path):
    _FrozenClock.install(monkeypatch)
    monkeypatch.setattr(restart, "_device_array_type", lambda: _FakeDevice)
    sizes = []
    real_buffer = restart.HostBuffer

    class RecordingStaging(real_buffer):
        def __init__(self, nbytes, **kwargs):
            sizes.append(int(nbytes))
            super().__init__(nbytes, **kwargs)

    restart.release_staging_buffer()
    monkeypatch.setattr(restart, "HostBuffer", RecordingStaging)
    state, cfg, devices = _device_state(monkeypatch)
    new = restart.write_restart(tmp_path / "streamed.npz", state, cfg)
    reference = _materialising_write(tmp_path / "materialised.npz", state, cfg)
    assert new.read_bytes() == reference.read_bytes()
    # ONE staging buffer, the size of the largest device member -- not the
    # sum of them -- and every device member went through it once.
    assert sizes == [max(device.nbytes for device in devices.values())]
    assert all(device.gets == 2 for device in devices.values())
    restored = _shim_state(cfg, monkeypatch)
    _fill_setup(restored)
    restart.restore_restart(new, restored, cfg)
    for name, device in devices.items():
        np.testing.assert_array_equal(
            np.asarray(getattr(restored, name)), device._array)


def test_digest_buffers_hash_the_tobytes_bytes():
    rng = np.random.default_rng(0)
    for value in (rng.random((3, 4)), rng.random((3, 4)).T,
                  np.ones((), np.float32), np.arange(10)[::2],
                  np.array([True, False]), rng.random(5) + 1j,
                  np.array(["2020-01-01"], dtype="M8[s]"),
                  rng.random((2, 3)).astype(">f8"), np.zeros((0, 3))):
        expected = hashlib.sha256(value.tobytes()).hexdigest()
        digest = hashlib.sha256()
        digest.update(contract._c_bytes(value))
        assert digest.hexdigest() == expected, value.dtype


def _counting(monkeypatch, name):
    calls = []
    real = getattr(contract, name)

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(contract, name, counted)
    return calls


def test_the_setup_fingerprint_is_cached_per_boundary_inventory(monkeypatch):
    cfg = _cfg(moist=True, mp_physics=1)
    state = _extension_state(cfg, monkeypatch, count=3)
    contract._SETUP_FINGERPRINT_CACHE.clear()
    hashed = _counting(monkeypatch, "_lateral_fingerprint_interval")
    # Three forcing times are two intervals.
    first = restart.setup_fingerprint(state)
    assert first == _historical_setup_fingerprint(state)
    assert len(hashed) == 2
    assert restart.setup_fingerprint(state) == first
    assert len(hashed) == 2, "an unchanged inventory was hashed again"

    # An appended interval is a new inventory.
    longer = _extension_state(cfg, monkeypatch, count=4)
    state.lateral_boundaries = longer.lateral_boundaries
    second = restart.setup_fingerprint(state)
    assert second == _historical_setup_fingerprint(state) != first
    assert len(hashed) == 5
    # Equal bytes in a NEW object are not trusted: identity, not equality.
    state.lateral_boundaries = dataclasses.replace(
        state.lateral_boundaries, intervals=tuple(
            dataclasses.replace(interval)
            for interval in state.lateral_boundaries.intervals))
    assert restart.setup_fingerprint(state) == second
    assert len(hashed) == 8
    # Other controls over the same intervals.
    state.lateral_boundaries = dataclasses.replace(
        state.lateral_boundaries, relax_zone=3)
    assert restart.setup_fingerprint(state) == \
        _historical_setup_fingerprint(state)
    assert len(hashed) == 11
    # A moved setup core misses too.
    state.thb[...] += np.float32(1.0)
    assert restart.setup_fingerprint(state) == \
        _historical_setup_fingerprint(state)
    assert len(hashed) == 14
    # setup_fingerprints shares one core pass and the same cache.
    core, full = contract.setup_fingerprints(state)
    assert (core, full) == (restart.setup_core_fingerprint(state),
                            restart.setup_fingerprint(state))
    assert len(hashed) == 14


def test_a_mutable_interval_is_never_cached(monkeypatch):
    cfg = _cfg(moist=True, mp_physics=1)
    state = _extension_state(cfg, monkeypatch, count=2)
    intervals = []
    for interval in state.lateral_boundaries.intervals:
        fields = {
            name: SimpleNamespace(**{
                side: SimpleNamespace(
                    value=np.array(getattr(boundary, side).value),
                    tendency=np.array(getattr(boundary, side).tendency),
                    time_law=None)
                for side in ("west", "east", "south", "north")})
            for name, boundary in interval.fields.items()}
        intervals.append(SimpleNamespace(
            start_seconds=interval.start_seconds,
            end_seconds=interval.end_seconds, fields=fields))
    state.lateral_boundaries = SimpleNamespace(
        intervals=tuple(intervals), spec_bdy_width=3, spec_zone=1,
        relax_zone=2)
    hashed = _counting(monkeypatch, "_lateral_fingerprint_interval")
    before = restart.setup_fingerprint(state)
    intervals[0].fields["u"].west.value[...] += 1.0
    after = restart.setup_fingerprint(state)
    assert before != after == _historical_setup_fingerprint(state)
    assert len(hashed) == 2


def test_prefix_rows_are_cached_per_interval(monkeypatch):
    cfg = _cfg(moist=True, mp_physics=1, specified=True, spec_bdy_width=3,
               spec_zone=1, relax_zone=2)
    state = _extension_state(cfg, monkeypatch, count=3)
    contract._PREFIX_ROW_CACHE.clear()
    rows = _counting(monkeypatch, "_lateral_boundary_prefix_row")
    first = restart.lateral_boundary_prefix_identity(state)
    assert len(rows) == 2
    again = restart.lateral_boundary_prefix_identity(state)
    assert again == first and len(rows) == 2
    rebuilt = restart.lateral_boundary_prefix_identity(
        state, rebuilt_end_frames=True)
    assert len(rows) == 4
    # Returned rows are copies: a caller that edits one edits no cache.
    again["intervals"][0]["fields"].append("tampered")
    assert restart.lateral_boundary_prefix_identity(state) == first
    contract._PREFIX_ROW_CACHE.clear()
    assert restart.lateral_boundary_prefix_identity(state) == first
    assert restart.lateral_boundary_prefix_identity(
        state, rebuilt_end_frames=True) == rebuilt


def test_a_real_wrfout_worker_counts_what_the_barrier_waits_for(tmp_path):
    """The counters the barrier reads are kept by the real worker loop."""
    from test_wrfout import _manual_async_writer, _queue_cpu_ticket
    from woof.io.wrfout import HistoryDurabilityBarrier

    writer = _manual_async_writer(threading.Event())
    paths = [tmp_path / f"wrfout_d01_1974-04-03_12_00_0{index}"
             for index in range(2)]
    for path in paths:
        _queue_cpu_ticket(writer, path)
    assert writer.admitted_count == 2
    barrier = HistoryDurabilityBarrier(((writer, writer.admitted_count),))
    barrier.wait()
    # Durable means published under its final name, as drain() means it.
    assert all(path.exists() for path in paths)
    assert writer._finished_count == 2
    writer.close()
