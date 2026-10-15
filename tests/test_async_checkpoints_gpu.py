"""GPU tier of tests/test_async_checkpoints.py (CuPy states, the real
side-stream snapshot into page-locked memory)."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from conftest import requires_gpu
from test_async_checkpoints import (
    _FrozenClock, _RootGate, _materialising_write, _members, _tree, _writer)
from test_restart import (
    _canonical_member_digest, _cfg, _fill_serialized, _fill_setup,
    _shim_state)
from woof.io import restart


# ---------------------------------------------------------------------------
# GPU tier
# ---------------------------------------------------------------------------

def _to_device(model):
    import cupy as cp

    for node in model.walk_parent_first():
        for name in restart.STATE_SERIALIZED_ATTRS:
            value = getattr(node.state, name, None)
            if isinstance(value, np.ndarray):
                setattr(node.state, name, cp.asarray(value))


@requires_gpu
@pytest.mark.gpu
def test_gpu_streamed_writer_is_byte_identical_to_the_materialising_one(
        monkeypatch, tmp_path):
    import cupy as cp

    _FrozenClock.install(monkeypatch)
    cfg = _cfg(moist=True, mp_physics=1)
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, 11)
    for name in restart.STATE_SERIALIZED_ATTRS:
        value = getattr(state, name, None)
        if isinstance(value, np.ndarray):
            setattr(state, name, cp.asarray(value))
    # A non-contiguous device member stages through a contiguous copy.
    state.u = cp.asfortranarray(state.u)
    new = restart.write_restart(tmp_path / "streamed.npz", state, cfg)
    reference = _materialising_write(tmp_path / "materialised.npz", state, cfg)
    assert new.read_bytes() == reference.read_bytes()


@requires_gpu
@pytest.mark.gpu
def test_gpu_snapshot_is_the_checkpoint_instant(monkeypatch, tmp_path):
    """Mutating the state right after the snapshot changes no archive byte."""
    import cupy as cp

    model, start = _tree(monkeypatch)
    _to_device(model)
    valid = start + timedelta(seconds=3600)
    expected = restart.write_tree_restart(tmp_path / "sync", model, valid)
    expected_digests = {
        gid: _canonical_member_digest(_members(tmp_path / "sync", gid)[0])
        for gid in (1, 2)}
    writer = _writer(model)
    gate = _RootGate(monkeypatch)
    # Hold the snapshot stream ~2 s behind the producer, so a mutation the
    # producer did NOT wait for would land before the copies run.
    writer._stream = cp.cuda.Stream(non_blocking=True)
    spin = cp.RawKernel(r"""
    extern "C" __global__ void spin(long long cycles) {
        long long start = clock64();
        while (clock64() - start < cycles) { }
    }""", "spin")
    # Load every kernel the mutation below launches NOW: a first-time
    # module load synchronizes the whole device and would hide the race.
    warm = cp.zeros(4, cp.float32)
    warm += cp.float32(1.0)
    warm *= cp.float32(2.0)
    spin((1,), (1,), (np.int64(1),))
    cp.cuda.Device().synchronize()
    with writer._stream:
        spin((1,), (1,), (np.int64(5_000_000_000),))
    restart.write_tree_restart(tmp_path / "async", model, valid)
    assert writer._job is not None
    # The model moves on immediately, on the producer stream.
    for node in model.walk_parent_first():
        node.state.u += cp.float32(1.0)
        node.state.thp *= cp.float32(2.0)
    gate.release.set()
    writer.close()
    assert writer.receipt["snapshot_arena_pinned"] is True
    assert writer.receipt["snapshot_arena_bytes"] > 0
    for gid in (1, 2):
        member, = _members(tmp_path / "async", gid)
        assert _canonical_member_digest(member) == expected_digests[gid]
    assert expected.exists()
