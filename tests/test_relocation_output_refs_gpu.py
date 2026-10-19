"""GPU tier of tests/test_relocation_output_refs.py.

Repeated relocations of a live child on the card, with an asynchronous
checkpoint of the tree in flight across them: the CuPy pool's live bytes
after every move must stay where the first move left them.  Before the
fix the in-flight checkpoint's restart plans held the outgoing child, so
the first move after each checkpoint added a whole second child to the
pool for as long as the checkpoint took.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from conftest import requires_gpu
from test_async_checkpoints import _RootGate, _writer


def _pool_used() -> int:
    import cupy as cp

    cp.cuda.Device().synchronize()
    return int(cp.get_default_memory_pool().used_bytes())


@requires_gpu
@pytest.mark.gpu
def test_relocations_under_an_in_flight_checkpoint_hold_the_pool_flat(
        monkeypatch, tmp_path):
    pytest.importorskip("cupy")
    from test_nest_relocation_staging import _integrated_tree
    from woof.core.model import release_output_device_refs
    from woof.core.nest_relocation import relocate_child
    from woof.io import restart
    from woof.verify.cases.nest_relocate import _child_preparer

    exp, model = _integrated_tree()
    child = model.node(exp.domains[1].grid_id)
    preparer = _child_preparer(exp.start_time)
    writer = _writer(model)

    def move(step: int) -> dict:
        receipt = relocate_child(
            child,
            i_parent_start=int(child.cfg.i_parent_start) + step,
            j_parent_start=int(child.cfg.j_parent_start),
            on_child_built=preparer, staging="host",
            # The executor's seam (execute_experiment.drop_validator).
            on_before_release=lambda: release_output_device_refs(model))
        return receipt

    # Warm the rebuild path once: first-use kernels and lazy tables are
    # not what this test measures.
    child_bytes = move(+1)["staging"]["released"]["owned_device_bytes"]
    move(-1)
    assert child_bytes > 0
    baseline = _pool_used()

    used = []
    valid = exp.start_time + timedelta(seconds=float(exp.run_seconds))
    for leg in range(2):
        gate = _RootGate(monkeypatch)
        restart.write_tree_restart(tmp_path / f"leg{leg}", model,
                                   valid + timedelta(seconds=leg))
        # The child member is written, the root held: in flight across
        # every move of this leg.
        assert gate.entered.wait(60)
        assert writer._job is not None and not writer._job.done.is_set()
        for index in range(6):
            move(+1 if index % 2 == 0 else -1)
            used.append(_pool_used())
        gate.release.set()
        assert writer._job.done.wait(60)
        assert writer.poll() is not None
    writer.close()

    growth = [value - baseline for value in used]
    # One rebuild's worth of slack: a held outgoing child shows up as a
    # whole child_bytes on the first move of each leg.
    assert max(growth) < 0.25 * child_bytes, (
        f"live pool grew by up to {max(growth)} bytes across "
        f"{len(used)} relocations under an in-flight checkpoint (child "
        f"{child_bytes} bytes): {growth}")
    assert max(used) - min(used) < 0.25 * child_bytes, used
