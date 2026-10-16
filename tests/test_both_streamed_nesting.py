"""Actual shared-executor composition: four storage roads and two-way feedback."""
from __future__ import annotations

import gc

import numpy as np
import pytest
from conftest import requires_gpu


# These tests intentionally use the existing synthetic terrain/Noah/YSU/Morrison
# fixture and the production prepared-domain builder, including rolling child
# tables. There is no source-model-dependent runtime path.
def trajectory(monkeypatch, *, mode, feedback, steps=6,
               checkpoint_dir=None, restart_from=None, lifecycle=False,
               cumulus=True):
    import cupy as cp
    from tilestream import test_moving_nest as moving
    from tilestream import test_nest_executor as executor
    from woof.core import streaming
    from woof.core.model import execute_experiment

    monkeypatch.setattr(executor, "I_PARENT_START", 41)
    monkeypatch.setattr(executor, "J_PARENT_START", 41)
    monkeypatch.setattr(executor, "PARENT_DT", 6.)
    pcfg = moving.parent_cfg(
        nx=96, ny=96, nz=24, dt=6., run_seconds=steps * 6.,
        nwp_diagnostics=int(lifecycle), radt_minutes=.5,
        **(dict(cudt_minutes=.5) if cumulus else
           dict(cu_physics=0, cudt_minutes=0.)))
    ccfg = moving.child_cfg(pcfg, nx=48, ny=48)
    boundaries = moving.domain_boundaries(pcfg, seconds=21600.)
    state, geo = moving.build_parent(pcfg, boundaries=boundaries)
    exp = executor.experiment(pcfg, ccfg, steps=steps)
    if checkpoint_dir is not None or restart_from is not None:
        from dataclasses import replace
        exp = replace(exp, restart_interval_s=18.)
    model = executor.assemble(exp, state, geo.grid, feedback=feedback)
    if feedback:
        executor.initial_feedback(model)
    relocation_runner = None
    if lifecycle:
        from woof.core.relocation_runner import RelocationRunner
        from woof.core.storm_tracking import FollowConfig
        from woof.experiment import RelocationConfig
        from woof.runtime import publish_lifecycle_runners
        follow = FollowConfig(field="uh", threshold=1.e9, fallback_threshold=100.,
                              search_margin_cells=4, min_shift_cells=1,
                              max_shift_cells=4, cooldown_seconds=600.)
        relocation_runner = RelocationRunner(
            config=RelocationConfig(enabled=True, grid_id=2,
                                    max_move_parent_cells=4,
                                    min_overlap_fraction=.25,
                                    cadence_seconds=360., follow=follow),
            schedule=model.schedule, on_child_built=lambda *_: None)
        from dataclasses import replace
        from woof.core.model import publish_declared_experiment
        exp = replace(exp, relocation=relocation_runner.config)
        publish_declared_experiment(model, exp)
        publish_lifecycle_runners(model, relocation_runner=relocation_runner)
        # A checkpoint between consultation boundaries must retain the already
        # accumulated window, even when later individual steps are weaker.
        for node in model.walk_parent_first():
            for slot in ("uh_spawn_window", "uh_follow_window"):
                node.state.existing_scratch(slot).fill(123.25)
    steppers = {}
    for node in model.walk_parent_first():
        gid = int(node.cfg.grid_id)
        if mode == "both" or (mode == "parent" and gid == 1) or (mode == "child" and gid == 2):
            tile = 48 if gid == 1 else 16
            options = streaming.StreamingOptions(
                mode="on", tile_nx=tile, tile_ny=tile, nbuffers=2, store="host")
            decision = streaming.decide(node.cfg.run, options)
            steppers[gid] = streaming.make_stepper(
                node.state, node.cfg.run, options, decision=decision,
                build=streaming.prepared_domain_builder(node, check_geography=False))
    checkpoints = []
    if restart_from is not None:
        from woof.io.restart import restore_tree_restart
        if lifecycle:
            from woof.io.restart import read_tree_lifecycle_header
            from woof.runtime import restore_nest_followers
            peek = read_tree_lifecycle_header(restart_from, model)
            assert restore_nest_followers(model, peek) == [2]
        restored = restore_tree_restart(restart_from, model)
        assert not restored.already_complete
    def checkpoint(tree, ticks):
        from datetime import timedelta
        from woof.io.restart import write_tree_restart
        valid = exp.start_time + timedelta(seconds=ticks / tree.schedule.clock.tick_den)
        checkpoints.append(write_tree_restart(checkpoint_dir, tree, valid))
    result = execute_experiment(
        model, steppers=steppers, validate_state=True, pool_trim_per_period=False,
        restart_handler=checkpoint if checkpoint_dir is not None else None,
        relocation_runner=relocation_runner)
    cp.cuda.runtime.deviceSynchronize()
    domains = {}
    for node in model.walk_parent_first():
        gid = int(node.cfg.grid_id)
        source = steppers[gid].store if gid in steppers else node.state
        digest, fields = moving.carrier_digest(source)
        assert {"state/p", "state/al", "state/alt", "state/mup", "state/qv"} <= fields.keys()
        assert len(fields) >= (150 if cumulus else 140)
        domains[gid] = dict(digest=digest, fields=fields,
                            ticks=node.clock.ticks, step_count=node.clock.step_count)
    output = dict(domains=domains, cadence=moving.cadence_census(model, steppers),
                  forces=result.forces, steps=result.steps,
                  feedback=model.node(2).coupler.feedback_count,
                  force_sync_bytes=model.node(2).coupler.force_sync_bytes,
                  checkpoints=checkpoints)
    del model, steppers, node, source, state, boundaries, geo
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return output


@requires_gpu
@pytest.mark.gpu
def test_both_streamed_one_way_matches_every_storage_road(monkeypatch):
    results = {mode: trajectory(monkeypatch, mode=mode, feedback=0)
               for mode in ("resident", "parent", "child", "both")}
    resident = results["resident"]
    assert resident["forces"] == 6 and resident["steps"] == 24
    for result in results.values():
        assert result["domains"] == resident["domains"]
        assert result["cadence"] == resident["cadence"]
        assert result["feedback"] == 0
    # The streamed-child route packs canonical setup as well as prognostics
    # in bounded donor chunks; the resident-child route still refreshes its
    # pre-existing parent field. Their transfer plans need not be additive.
    assert results["both"]["force_sync_bytes"] > results["child"]["force_sync_bytes"]
    assert results["parent"]["force_sync_bytes"] > 0
    assert results["child"]["force_sync_bytes"] > 0


@requires_gpu
@pytest.mark.gpu
def test_both_streamed_feedback_matches_and_detects_stale_diagnostic_write(monkeypatch):
    from woof.core.nest_operands import NestWindowSource
    resident = trajectory(monkeypatch, mode="resident", feedback=1)
    both = trajectory(monkeypatch, mode="both", feedback=1)
    assert both["domains"] == resident["domains"]
    assert both["cadence"] == resident["cadence"]
    assert both["feedback"] == resident["feedback"] == 7
    one_way = trajectory(monkeypatch, mode="resident", feedback=0)
    assert resident["domains"][1]["digest"] != one_way["domains"][1]["digest"]

    # Independent counterfactual at the actual canonical write seam: leak
    # one attach-time column outside the diagnosed footprint. The old
    # _sync_out hook is no longer on the bounded parent transaction.
    original = NestWindowSource.write
    def stale_diagnostic_write(source, name, window, value):
        import cupy as cp
        original(source, name, window, value)
        if name in ("p", "al", "alt"):
            source.array(name)[..., 0, 0] = cp.asnumpy(
                getattr(source.state, name)[..., 0, 0])
    monkeypatch.setattr(NestWindowSource, "write", stale_diagnostic_write)
    stale = trajectory(monkeypatch, mode="both", feedback=1)
    differing = {name for name, sha in stale["domains"][1]["fields"].items()
                 if sha != resident["domains"][1]["fields"][name]}
    assert differing == {"state/p", "state/al", "state/alt"}
    assert stale["domains"][2] == resident["domains"][2]


def _force_tables_land_late(monkeypatch, milliseconds=200.):
    """Hold FORCE's rolling-table writes back on the stream FORCE runs on.

    After FORCE has queued its tables, they are poisoned with NaN, a spin
    kernel holds the caller's stream for ``milliseconds``, and the true
    tables are written back behind it -- all on the caller's stream.  A
    child sweep ordered after that stream reads the true tables and steps
    exactly as an undelayed run does; one that is not copies the poison (or
    the previous interval) into its tile buffers.  This is the race a
    streamed child had: d02 differed run to run while d01 was identical.
    """
    import cupy as cp
    from woof.core import nest

    original = nest.attach_nest_boundaries
    spin = cp.RawKernel(r'''
extern "C" __global__ void spin(long long cycles) {
    long long start = clock64();
    while (clock64() - start < cycles) {}
}''', "spin")
    # cudaDevAttrClockRate (kHz): still an attribute where CUDA 13 dropped
    # cudaDeviceProp.clockRate.
    rate_khz = cp.cuda.Device().attributes["ClockRate"]
    cycles = np.int64(milliseconds * rate_khz)
    calls = []

    def late(state, fields, **kwargs):
        tables = [array for sides in fields.values()
                  for pair in sides.values() for array in pair]
        held = [array.copy() for array in tables]
        for array in tables:
            array.fill(np.nan)
        spin((1,), (1,), (cycles,))
        for array, keep in zip(tables, held):
            array[...] = keep
        calls.append(len(tables))
        return original(state, fields, **kwargs)

    monkeypatch.setattr(nest, "attach_nest_boundaries", late)
    return calls


@requires_gpu
@pytest.mark.gpu
def test_streamed_child_is_deterministic_when_force_lands_late(monkeypatch):
    # No cumulus, like the 1 km / 333 m Greenland tree that showed it: the
    # Kain-Fritsch expiry probe in the tile inventory happens to block the
    # host on the caller's stream before the first tile steps, which hid
    # the race.
    first = trajectory(monkeypatch, mode="both", feedback=0, cumulus=False)
    second = trajectory(monkeypatch, mode="both", feedback=0, cumulus=False)
    assert second["domains"] == first["domains"]
    late = _force_tables_land_late(monkeypatch)
    delayed = trajectory(monkeypatch, mode="both", feedback=0, cumulus=False)
    assert len(late) == delayed["forces"] == first["forces"] == 6
    assert delayed["domains"] == first["domains"]


@requires_gpu
@pytest.mark.gpu
def test_both_streamed_tree_checkpoint_resumes_on_either_storage_road(monkeypatch, tmp_path):
    uninterrupted = trajectory(monkeypatch, mode="both", feedback=1,
                               checkpoint_dir=tmp_path / "streamed")
    assert len(uninterrupted["checkpoints"]) == 2
    midway = uninterrupted["checkpoints"][0]
    for mode in ("both", "resident"):
        resumed = trajectory(monkeypatch, mode=mode, feedback=1, restart_from=midway)
        assert resumed["steps"] == 12 and resumed["forces"] == 3
        assert resumed["domains"] == uninterrupted["domains"]
        assert resumed["cadence"] == uninterrupted["cadence"]
    resident = trajectory(monkeypatch, mode="resident", feedback=1,
                          checkpoint_dir=tmp_path / "resident")
    assert resident["domains"] == uninterrupted["domains"]
    resumed = trajectory(monkeypatch, mode="both", feedback=1,
                         restart_from=resident["checkpoints"][0])
    assert resumed["domains"] == uninterrupted["domains"]
    assert resumed["cadence"] == uninterrupted["cadence"]
