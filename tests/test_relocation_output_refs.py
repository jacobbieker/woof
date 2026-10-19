"""In-flight outputs must not keep a relocated domain's outgoing state alive.

A relocation snapshots the moving domain to host, drops every array its
state holds and rebuilds it at the new placement.  The drop frees device
memory only if nothing else still references those arrays.  Two outputs
did: an asynchronous checkpoint kept its restart plans -- whose members
are the live state arrays, by reference -- until the checkpoint was
published, and a ``direct``-mode history frame keeps the arrays it stages
until the writer thread takes it.  A move that landed while either was in
flight left the outgoing domain on the card beside its rebuild, which on a
long moving-nest run is what exhausted a 48 GB card.

CPU tier: NumPy members and stand-in device arrays, watched through weak
references.  The GPU tier (``test_relocation_output_refs_gpu.py``) measures
the CuPy pool across repeated relocations.
"""

from __future__ import annotations

import gc
import threading
import weakref
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import numpy as np

from test_async_checkpoints import _RootGate, _members, _tree, _writer
from test_wrfout import _await_worker, _manual_async_writer, _queue_cpu_ticket
from woof.io import restart


def _member_refs(model, grid_id):
    """Weak references to every serialized array of one domain's state."""
    node = model.node(grid_id)
    refs = {}
    for name in restart.STATE_SERIALIZED_ATTRS:
        value = getattr(node.state, name, None)
        if isinstance(value, np.ndarray):
            refs[name] = weakref.ref(value)
    assert refs, "the fixture state carries no serialized arrays"
    return refs


def test_an_in_flight_checkpoint_holds_no_state_array(monkeypatch, tmp_path):
    """Dropping a domain's state mid-checkpoint frees every one of its arrays."""
    from woof.core.nest_relocation import release_state_arrays

    model, start = _tree(monkeypatch)
    writer = _writer(model)
    gate = _RootGate(monkeypatch)
    restart.write_tree_restart(tmp_path, model, start + timedelta(seconds=3600))
    # The child is written, the root held: the checkpoint is in flight.
    assert gate.entered.wait(30)
    assert writer._job is not None and not writer._job.done.is_set()
    refs = _member_refs(model, 2)

    # What a relocation does to its outgoing child.
    release_state_arrays(model.node(2).state)
    gc.collect()
    alive = sorted(name for name, ref in refs.items() if ref() is not None)
    assert alive == [], (
        f"the in-flight checkpoint still holds {alive} of the released state")

    gate.release.set()
    assert writer._job.done.wait(30)
    root = writer.poll()
    assert root is not None and root.exists()
    # The archive is the checkpoint instant, released state or not.
    assert len(_members(tmp_path, 2)) == 1
    writer.close()


def test_release_device_refs_waits_for_the_snapshot_then_lets_go():
    from woof.io.async_checkpoint import CheckpointWriter

    calls = []

    class Event:
        def synchronize(self):
            calls.append("synchronize")

    class DeviceArray:
        pass

    writer = object.__new__(CheckpointWriter)
    writer._job = None
    assert writer.release_device_refs() == 0
    source = DeviceArray()
    ref = weakref.ref(source)
    writer._job = SimpleNamespace(event=Event(), device_refs=[source])
    del source
    assert writer.release_device_refs() == 1
    assert calls == ["synchronize"]
    gc.collect()
    assert ref() is None
    # Idempotent: nothing left to wait for.
    assert writer.release_device_refs() == 0
    assert calls == ["synchronize"]


def test_a_queued_history_frame_lets_go_of_its_device_arrays(
        monkeypatch, tmp_path):
    """The frame stays queued and is written; only the device refs go."""
    import woof.io.wrfout as wrfout

    write_started = threading.Event()
    finish_write = threading.Event()
    written = []

    class BlockingWriter:
        def __init__(self, path, **_kwargs):
            self.path = path

        def __enter__(self):
            return self

        def write_frame(self, _time_str, fields):
            written.append(sorted(fields))
            write_started.set()
            finish_write.wait()

        def __exit__(self, *_args):
            return False

    class DeviceArray:
        pass

    monkeypatch.setattr(wrfout, "WrfoutWriter", BlockingWriter)
    writer = _manual_async_writer(threading.Event())
    try:
        # The worker takes the first frame and blocks writing it, so the
        # second sits in the queue, still owning its staged arrays.
        _queue_cpu_ticket(writer, tmp_path / "first")
        _await_worker(write_started, writer, reaching="the first write")
        device = DeviceArray()
        device_ref = weakref.ref(device)
        queued = _queue_cpu_ticket(writer, tmp_path / "second",
                                   device_refs=(device,))
        del device
        gc.collect()
        assert device_ref() is not None
        assert writer.release_device_refs() == 1
        gc.collect()
        assert device_ref() is None
        assert queued.device_refs == ()
        assert writer.release_device_refs() == 0
        finish_write.set()
        writer.drain()
        assert len(written) == 2
        # The worker forgets what it took, so nothing outlives the frame.
        assert not writer._held_device_tickets
    finally:
        finish_write.set()
        writer.close()


def test_a_failing_wait_keeps_the_remaining_tickets_tracked():
    from woof.io.wrfout import AsyncDomainWrfoutWriter

    class Fails:
        def synchronize(self):
            raise RuntimeError("injected device fault")

    class Done:
        def synchronize(self):
            return None

    writer = object.__new__(AsyncDomainWrfoutWriter)
    writer._condition = threading.Condition()
    writer.stream = nullcontext()
    first = SimpleNamespace(event=Fails(), device_refs=(object(),))
    second = SimpleNamespace(event=Done(), device_refs=(object(),))
    writer._held_device_tickets = [first, second]
    try:
        writer.release_device_refs()
    except RuntimeError:
        pass
    else:
        raise AssertionError("the injected fault did not surface")
    assert writer._held_device_tickets == [first, second]
    first.event = Done()
    assert writer.release_device_refs() == 2
    assert writer._held_device_tickets == []


def test_the_writer_set_releases_every_domain():
    from woof.io.wrfout import PerDomainWrfoutWriters

    counts = {1: 2, 2: 0, 3: 5}
    writers = object.__new__(PerDomainWrfoutWriters)
    writers._writers = {
        gid: SimpleNamespace(release_device_refs=lambda n=n: n)
        for gid, n in counts.items()}
    # A domain writer without the method (a stand-in) is skipped.
    writers._writers[4] = SimpleNamespace()
    assert writers.release_device_refs() == 7


def test_the_executor_seam_releases_checkpoint_and_history():
    from woof.core.model import release_output_device_refs

    order = []
    model = SimpleNamespace(
        _checkpoint_writer=SimpleNamespace(
            release_device_refs=lambda: order.append("checkpoint") or 3),
        _io_manager=SimpleNamespace(
            release_device_refs=lambda: order.append("history") or 4))
    assert release_output_device_refs(model) == {"checkpoint": 3, "history": 4}
    assert order == ["checkpoint", "history"]
    # A model with neither output (the idealized cases, a CPU stand-in).
    assert release_output_device_refs(SimpleNamespace()) == {
        "checkpoint": 0, "history": 0}


def test_release_state_arrays_drops_the_scratch_slots():
    from woof.core.nest_relocation import release_state_arrays

    owned = np.zeros(8, np.float32)
    ref = weakref.ref(owned)
    state = SimpleNamespace(u=np.zeros(4, np.float32), physics=None,
                            _scratch={"integration_health_flags": owned})
    del owned
    receipt = release_state_arrays(state)
    gc.collect()
    assert ref() is None
    assert state._scratch == {}
    assert state.u is None
    assert receipt["scratch_device_bytes"] == 0  # host arrays are not device bytes


def test_the_relocation_seam_runs_before_the_outgoing_release():
    """drop_validator (the runner's before_rebuild) releases output refs."""
    import inspect

    from woof.core import model as model_module

    source = inspect.getsource(model_module.execute_experiment)
    seam = source[source.index("def drop_validator"):]
    seam = seam[:seam.index("outcome = relocation_runner.on_period_begin")]
    assert "release_output_device_refs(model)" in seam
