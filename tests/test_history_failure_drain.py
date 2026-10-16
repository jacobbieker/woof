"""A failing run publishes every history frame it took, in complete sets.

PerDomainWrfoutWriters.__exit__ with an exception used to abort the queue: a
frame taken a few steps before a NaN gate fired was published or quarantined
depending on how far its write had got.  The policy now is drain before
stopping (woof.io.wrfout.PerDomainWrfoutWriters._finish_after_failure):

* every frame already taken is written and published, and the original
  exception propagates unchanged;
* a frame set (every domain due at one valid time) the failure cut short is
  withdrawn, its written frames quarantined as ``partial-frame-set``;
* the drain is bounded, and a second Ctrl-C or SIGTERM aborts it.

These are CPU-only: the per-domain workers are the real worker loop, the
netCDF writer is a stand-in that writes a temporary file and renames it at
the same point the real one does (``complete_output_identity``).
"""
from __future__ import annotations

import os
import signal
import threading
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import woof.io.wrfout as wrfout
from test_wrfout import _HANDOFF_POLL_SECONDS, _cpu_ticket, _manual_async_writer

HISTORY_TICKS = 30
START = datetime(2026, 9, 26, 17)


class _Gate:
    """Hold the stand-in writer's file writes until released."""

    def __init__(self):
        self.release = threading.Event()
        self.writing = threading.Event()


def _stand_in_writer(hold: _Gate, *, fail_on=(), failure=OSError):
    class StandInWriter:
        """Writes ``.<name>.tmp``; renames it in complete_output_identity."""

        def __init__(self, path, **_kwargs):
            self._final_path = Path(path)
            self._temp_path = self._final_path.with_name(
                f".{self._final_path.name}.tmp")
            self.publication_revision = None

        def __enter__(self):
            return self

        def write_frame(self, _time_str, _fields):
            hold.writing.set()
            hold.release.wait()
            if self._final_path.name in fail_on:
                raise failure(f"injected write failure for {self._final_path.name}")
            self._temp_path.write_bytes(b"frame")

        def __exit__(self, exc_type, *_args):
            if exc_type is None:
                self.publication_revision = 1
            return False

        def complete_output_identity(self, *, cancel_event=None):
            if cancel_event is not None and cancel_event.is_set():
                wrfout.quarantine_file(self._temp_path,
                                       reason="failed-wrfout-identity")
                raise InterruptedError("cancelled before completion")
            os.replace(self._temp_path, self._final_path)
            return SimpleNamespace(address=str(self._final_path.resolve()))

        def release_identity_handle(self, *, preserve=False):
            return None

    return StandInWriter


class _Clock:
    def __init__(self, ticks=0):
        self.ticks = ticks
        self.spec = SimpleNamespace(
            start_ticks=0,
            history_offset_due=lambda elapsed: elapsed % HISTORY_TICKS == 0)


def _writer_set(tmp_path, domains=(1, 2)):
    """A two-domain writer set over real worker threads and a model whose
    clocks the test moves."""
    abort = threading.Event()
    writers = object.__new__(wrfout.PerDomainWrfoutWriters)
    writers._writers = {gid: _manual_async_writer(abort) for gid in domains}
    for gid, writer in writers._writers.items():
        writer.grid_id = gid
    writers._abort_event = abort
    writers.last_durable_wrfout = None
    writers.output_dir = Path(tmp_path)
    writers._episode_by_grid_id = {}
    writers._published_paths = set()
    writers._archived_paths = []
    writers._frame_sets = {}
    writers._frame_set_lock = threading.Lock()
    clocks = {gid: _Clock() for gid in domains}
    writers.model = SimpleNamespace(
        node=lambda gid: SimpleNamespace(clock=clocks[gid]))
    return writers, clocks


def _take(writers, gid, ticks, tmp_path):
    """What PerDomainWrfoutWriters.submit does around the writer's submit."""
    valid_time = START + timedelta(seconds=ticks)
    path = Path(tmp_path) / wrfout.wrfout_filename(valid_time, gid)
    gate = writers._join_frame_set(ticks, valid_time)
    writer = writers._writers[gid]
    writer.before_wait = writers._decide_all_open_sets
    ticket = _cpu_ticket(path)
    ticket.publish_gate = gate
    writer._admit(ticket)
    writers._submitted(gate, gid, path)
    return path


def _boundary(writers, clocks, ticks, tmp_path):
    """Every domain reaches ``ticks`` and takes its frame there."""
    for clock in clocks.values():
        clock.ticks = ticks
    return [_take(writers, gid, ticks, tmp_path) for gid in sorted(clocks)]


def _quarantined(tmp_path):
    folder = Path(tmp_path) / ".quarantine"
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


def _exit_with(writers, error, hold, *, release_after=None):
    """Call __exit__ on THIS thread; release the held writes once the set
    has decided (``release_after`` is an extra action taken first)."""
    def release():
        writers_ready.wait()
        if release_after is not None:
            release_after()
        hold.release.set()

    writers_ready = threading.Event()
    original = writers._withdraw_incomplete_sets

    def withdraw_then_signal():
        withdrawn = original()
        writers_ready.set()
        return withdrawn

    writers._withdraw_incomplete_sets = withdraw_then_signal
    releaser = threading.Thread(target=release, daemon=True)
    releaser.start()
    try:
        assert writers.__exit__(type(error), error, None) is False
    finally:
        hold.release.set()
        releaser.join()


@pytest.fixture
def hold(monkeypatch):
    gate = _Gate()
    monkeypatch.setattr(wrfout, "WrfoutWriter", _stand_in_writer(gate))
    yield gate
    gate.release.set()


@pytest.mark.parametrize("error", [
    RuntimeError("d01 integration produced a non-finite state at model step 39"),
    KeyboardInterrupt(),
], ids=["failure", "first-interrupt"])
def test_every_frame_taken_before_a_failure_is_published(tmp_path, hold, error):
    writers, clocks = _writer_set(tmp_path)
    taken = (_boundary(writers, clocks, 30, tmp_path)
             + _boundary(writers, clocks, 60, tmp_path))
    # The first write of each domain is still running and the second is
    # queued behind it when the failure arrives.
    assert hold.writing.wait(timeout=30)
    _exit_with(writers, error, hold)
    assert all(path.exists() for path in taken)
    assert _quarantined(tmp_path) == []
    assert not writers._abort_event.is_set()
    record = error.woof_history_on_failure
    assert record["outcome"] == "drained"
    assert record["published_after_failure"] == 4
    assert record["withdrawn"] == []
    assert any("4 frame(s) taken before it were published" in note
               for note in error.__notes__)
    assert writers.history_writes_receipt()["on_failure"] == record


def test_a_frame_set_the_failure_cut_short_is_quarantined(tmp_path, hold):
    writers, clocks = _writer_set(tmp_path)
    complete = _boundary(writers, clocks, 30, tmp_path)
    for clock in clocks.values():
        clock.ticks = 60
    partial = _take(writers, 1, 60, tmp_path)
    # d02's frame at 60 was never taken: its submit failed.
    error = RuntimeError("d02 history staging failed")
    _exit_with(writers, error, hold)
    assert all(path.exists() for path in complete)
    assert not partial.exists()
    quarantined = _quarantined(tmp_path)
    assert len(quarantined) == 1
    assert quarantined[0].startswith(f".{partial.name}.tmp.partial-frame-set")
    (row,) = error.woof_history_on_failure["withdrawn"]
    assert row["missing_domains"] == ["d02"]
    assert row["taken_domains"] == ["d01"]
    assert row["reason"].startswith("partial-frame-set: d02")
    assert list(row["quarantined"]) == [str(partial)]
    assert any("withdrew the frame set" in note for note in error.__notes__)


def test_a_set_is_published_once_every_due_domain_has_taken_its_frame(
        tmp_path, hold):
    writers, clocks = _writer_set(tmp_path)
    for clock in clocks.values():
        clock.ticks = 30
    first = _take(writers, 1, 30, tmp_path)
    hold.release.set()
    # Written, but held at the gate until d02 takes its frame at 30.
    writer = writers._writers[1]
    while not writer._identity_pending and writer._pending:
        writer._thread.join(timeout=_HANDOFF_POLL_SECONDS)
    assert not first.exists()
    second = _take(writers, 2, 30, tmp_path)
    writers.drain()
    assert first.exists() and second.exists()
    writers.close()


def test_a_domain_that_moved_past_the_instant_does_not_hold_its_set(
        tmp_path, hold):
    writers, clocks = _writer_set(tmp_path)
    clocks[1].ticks = 30
    clocks[2].ticks = 33          # d02 already past 30 without a frame there
    only = _take(writers, 1, 30, tmp_path)
    hold.release.set()
    writers._writers[1]._drain(include_identity=True)
    assert only.exists()
    writers.close()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM],
                         ids=["SIGINT", "SIGTERM"])
def test_a_second_signal_during_the_drain_aborts(tmp_path, hold, signum):
    writers, clocks = _writer_set(tmp_path, domains=(1,))
    first = _boundary(writers, clocks, 30, tmp_path)[0]
    second = _boundary(writers, clocks, 60, tmp_path)[0]
    assert hold.writing.wait(timeout=30)
    handed_back = []
    previous = signal.getsignal(signal.SIGTERM)
    if signum == signal.SIGTERM:
        signal.signal(signal.SIGTERM,
                      lambda number, _frame: handed_back.append(number))
    error = RuntimeError("non-finite state")
    try:
        def interrupt():
            # The held write is released only once the abort is in place,
            # so the drain cannot finish before the signal is handled.
            os.kill(os.getpid(), signum)
            writers._abort_event.wait(timeout=30)

        _exit_with(writers, error, hold, release_after=interrupt)
    finally:
        signal.signal(signal.SIGTERM, previous)
    record = error.woof_history_on_failure
    assert record["outcome"] == "interrupted"
    assert record["detail"] == ("SIGTERM" if signum == signal.SIGTERM
                                else "Ctrl-C")
    assert writers._abort_event.is_set()
    # The queued frame is abandoned; nothing after the abort is published.
    assert not second.exists()
    assert handed_back == ([signal.SIGTERM] if signum == signal.SIGTERM else [])
    del first


def test_the_drain_bound_falls_back_to_the_abort(tmp_path, hold):
    writers, clocks = _writer_set(tmp_path, domains=(1,))
    _boundary(writers, clocks, 30, tmp_path)
    queued = _boundary(writers, clocks, 60, tmp_path)[0]
    assert hold.writing.wait(timeout=30)
    writers.failure_drain_timeout_s = 0.2
    error = RuntimeError("non-finite state")
    # The held write is released only after the bound has passed.
    timer = threading.Timer(1.0, hold.release.set)
    timer.start()
    try:
        assert writers.__exit__(RuntimeError, error, None) is False
    finally:
        timer.cancel()
        hold.release.set()
    record = error.woof_history_on_failure
    assert record["outcome"] == "timeout"
    assert record["bound_seconds"] == 0.2
    assert writers._abort_event.is_set()
    assert not queued.exists()


@pytest.mark.parametrize("failure", [OSError, TimeoutError],
                         ids=["OSError", "TimeoutError"])
def test_a_drain_error_is_attached_and_never_replaces_the_failure(
        tmp_path, monkeypatch, failure):
    # A write that fails with TimeoutError (ETIMEDOUT on a network share) is
    # a drain error, not the drain's own bound passing.
    hold = _Gate()
    monkeypatch.setattr(wrfout, "WrfoutWriter",
                        _stand_in_writer(hold, failure=failure, fail_on={
                            wrfout.wrfout_filename(
                                START + timedelta(seconds=30), 1)}))
    writers, clocks = _writer_set(tmp_path, domains=(1,))
    _boundary(writers, clocks, 30, tmp_path)
    error = RuntimeError("non-finite state")
    hold.release.set()
    assert writers.__exit__(RuntimeError, error, None) is False
    record = error.woof_history_on_failure
    assert record["outcome"] == "drain-error"
    assert "injected write failure" in record["detail"]
    assert str(error) == "non-finite state"
    assert any("injected write failure" in note for note in error.__notes__)


def test_the_bound_is_the_finalization_bound_at_the_step_floor():
    from woof.supervisor import FINALIZATION_FLOOR_BYTES_PER_SECOND

    assert wrfout.failure_drain_bound_seconds(0) == 120.0
    assert wrfout.failure_drain_bound_seconds(
        10 * FINALIZATION_FLOOR_BYTES_PER_SECOND) == 130.0
