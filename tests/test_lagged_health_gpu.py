"""The lagged per-step health gate on real device steps (woof.core.lagged_health).

A failure at step k is raised no later than the op after it, names step k,
says it was detected late, and never reaches a history frame or checkpoint;
strict mode keeps stopping on the step itself.
"""
from datetime import datetime

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def _model(*, run_seconds=12., history_s=5., restart_s=10., adaptive=False):
    from woof.core.clock import build_schedule, resolve_clock
    from woof.core.model import DomainNode, ExperimentState, ModelRuntimeStatus
    from woof.experiment import DomainConfig, ExperimentConfig, VerticalConfig
    from tilestream import harness

    cfg = harness.make_config(
        16, 12, 4, dt=1., grid_id=1, run_seconds=run_seconds,
        use_adaptive_time_step=adaptive, starting_time_step=1,
        min_time_step=1, max_time_step=1, step_to_output_time=True)
    state = harness.make_state(cfg)
    domain = DomainConfig(grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1, history_interval_s=history_s,
        run=cfg, time_step=1)
    exp = ExperimentConfig(name="lagged-health", start_time=datetime(2026, 1, 1),
        run_seconds=run_seconds,
        vertical=VerticalConfig(eta_levels=(), p_top=0., hybrid_opt=0, etac=.2),
        projection=None, restart_interval_s=restart_s, domains=(domain,))
    calendar = resolve_clock(exp, lbc_interval_s=1.)
    node = DomainNode(domain, None, state, calendar.clocks()[1], None, [], None)
    model = ExperimentState(node, {1: node}, build_schedule(exp, calendar), None, "a" * 64)
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._scratch_arena = model._dycore_state_workspace = model._io_manager = None
    model._last_checkpoint = None
    return exp, model


def _run(mode, fault_step, fault, **build):
    import cupy as cp
    from woof.core import dycore, health_ledger
    from woof.core.model import execute_experiment

    exp, model = _model(**build)
    history, checkpoints, count = [], [], [0]

    def describe(flags):
        raise FloatingPointError(f"test scheme flagged {flags:#x}")

    def faulted_step(state, cfg, **kwargs):
        dycore.step(state, cfg, **kwargs)
        count[0] += 1
        if fault == "nan-then-error" and count[0] == fault_step + 1:
            # A different failure inside the NEXT step, before its own
            # health record exists: the earlier bad step is the cause.
            raise ValueError("an unrelated failure in the next step")
        if count[0] != fault_step:
            return
        if fault in ("nan", "nan-then-error"):
            state.u[1, 5, 6] = cp.nan
        elif fault == "theta":
            # Finite, so the stability record passes; the full-state gate's
            # theta ceiling does not.
            state.thp[1, 5, 6] = cp.float32(5000.)
        elif fault in ("ledger", "ledger-then-error"):
            # A scheme status word, reported exactly as the real sites do.
            flags = health_ledger.read_status(
                cp.ones(1, dtype=cp.uint32), site="test-scheme",
                describe=describe)
            if flags:
                describe(flags)
            if fault == "ledger-then-error":
                raise ValueError("an unrelated failure later in the step")

    error = None
    try:
        execute_experiment(
            model, steppers={1: faulted_step}, experiment=exp,
            history_handler=lambda tree, node, ticks: history.append(
                node.clock.elapsed_seconds),
            restart_handler=lambda tree, ticks: checkpoints.append(
                tree.root.clock.elapsed_seconds),
            pool_trim_per_period=False, health_check=mode)
    except Exception as exc:  # noqa: BLE001 - asserted by the callers
        error = exc
    return model, error, history, checkpoints, count[0]


@pytest.mark.parametrize("fault", ["nan", "ledger"])
def test_lagged_failure_is_detected_by_the_next_step(fault):
    model, error, history, checkpoints, steps = _run("lagged", 7, fault)
    assert error is not None
    text = str(error) + "".join(getattr(error, "__notes__", ()))
    assert "lagged health check: d01 step 7 failed" in text
    assert "after d01 step 8 was queued" in text
    assert "--health-check strict" in text
    if fault == "nan":
        assert "non-finite state at model step 7" in str(error)
    assert error.lagged_health == {
        "grid_id": 1, "failed_step": 7,
        "detected": "after d01 step 8 was queued"}
    # Step 8 was queued (that is the lag) and its op raised before the
    # clock committed it: the clock stands at the end of the failed step.
    assert steps == 8 and model.root.clock.elapsed_seconds == 7.
    assert history == [0., 5.] and checkpoints == []


@pytest.mark.parametrize("fault", ["nan", "ledger"])
def test_strict_mode_still_stops_on_the_step(fault):
    model, error, history, checkpoints, steps = _run("strict", 7, fault)
    assert error is not None
    assert "lagged health check" not in str(error) + "".join(
        getattr(error, "__notes__", ()))
    assert steps == 7 and model.root.clock.elapsed_seconds == 6.
    assert history == [0., 5.] and checkpoints == []
    assert model._health_check_policy["mode"] == "strict"


@pytest.mark.parametrize("fault_step,stopped,history_seen,checkpoints_seen", [
    (5, 5., [0.], []),          # history instant: settled before the frame
    (10, 10., [0., 5.], []),    # history AND checkpoint instant
    (12, 12., [0., 5., 10.], [10.]),  # the stop instant
])
def test_a_failure_at_an_output_instant_never_reaches_it(
        fault_step, stopped, history_seen, checkpoints_seen):
    model, error, history, checkpoints, steps = _run("lagged", fault_step, "nan")
    assert error is not None
    assert f"non-finite state at model step {fault_step}" in str(error)
    assert "at the synchronous check" in str(error)
    assert steps == fault_step and model.root.clock.elapsed_seconds == stopped
    assert history == history_seen and checkpoints == checkpoints_seen


def test_lagged_full_state_report_names_the_strict_field_and_index():
    from woof.core.health import HealthCheckError

    reports, errors = {}, {}
    for mode in ("strict", "lagged"):
        # History every 6 s: root step 4's periodic full-state gate is NOT a
        # mandatory instant, so the lagged run queues it.
        model, error, history, _checkpoints, steps = _run(
            mode, 4, "theta", history_s=6.)
        assert isinstance(error, HealthCheckError), error
        reports[mode], errors[mode] = error.report, error
        assert history == [0.]
    strict, lagged = reports["strict"], reports["lagged"]
    # The phase is the gate's own in both modes (readers parse the domain
    # out of it); only the value is read one step late, and says so.
    assert (lagged.first_bad_field, lagged.first_bad_index,
            lagged.first_bad_flat_index, lagged.status_bits,
            lagged.phase) == (
        strict.first_bad_field, strict.first_bad_index,
        strict.first_bad_flat_index, strict.status_bits, strict.phase)
    assert strict.phase == "post-d01-sync.d01"
    assert "after the following step" in lagged.reason
    assert "after the following step" not in strict.reason
    assert any("lagged health check: post-d01-sync.d01 after root step 4"
               in note for note in errors["lagged"].__notes__)


def test_healthy_lagged_run_is_bit_identical_and_receipted():
    import cupy as cp
    from tilestream import gather
    from woof.core.model import execute_experiment

    results, policies = {}, {}
    for mode in ("strict", "lagged"):
        exp, model = _model(run_seconds=12., history_s=6., restart_s=12.)
        execute_experiment(model, experiment=exp, pool_trim_per_period=False,
                           history_handler=lambda *a: None,
                           restart_handler=lambda *a: None, health_check=mode)
        results[mode] = {k: cp.asnumpy(v) for k, v in
                         gather.inventory(model.root.state).items()}
        policies[mode] = model._health_check_policy
    for name, value in results["strict"].items():
        np.testing.assert_array_equal(value, results["lagged"][name], err_msg=name)
    lagged = policies["lagged"]
    assert lagged["mode"] == "lagged" and lagged["source"] == "argument"
    # Twelve step records plus the non-mandatory periodic gates (root steps
    # 4 and 8); every one read, most of them one op late.
    assert lagged["submitted"] == 14
    assert lagged["lagged_resolutions"] > 0
    assert lagged["synchronous_flushes"] >= 1
    assert policies["strict"]["mode"] == "strict"


def test_fused_cfl_readback_equals_the_per_domain_reads(monkeypatch):
    import cupy as cp
    from woof.core import dycore

    stat, calls = {}, {}
    monkeypatch.setattr(dycore, "_WRF_CFL_STAT", stat)
    monkeypatch.setattr(dycore, "_WRF_CFL_CALLS", calls)
    rng = np.random.default_rng(7)
    for gid, folds in ((1, 7), (2, 35), (3, 0)):
        stat[gid] = cp.asarray(rng.integers(
            0, 2 ** 31, size=(dycore._WRF_CFL_SLOTS, dycore._WRF_CFL_WORDS),
            dtype=np.uint32))
        calls[gid] = folds
    one_at_a_time = {gid: dycore.take_wrf_cfl(gid) for gid in (1, 2, 3, 4)}
    fused = dycore.take_wrf_cfl_many((1, 2, 3, 4))
    assert fused == one_at_a_time
    assert fused[3] == fused[4] == (0.0, 0.0)


def test_adaptive_lagged_failure_settles_before_the_next_step_is_chosen():
    model, error, history, _checkpoints, steps = _run(
        "lagged", 7, "nan", adaptive=True)
    assert "non-finite state at model step 7" in str(error)
    assert "before the adaptive step choice" in str(error)
    # Raised before step 8 was chosen or queued: the clock still holds the
    # step that failed, which is what a recovery would halve.
    assert steps == 7 and model.root.clock.elapsed_seconds == 7.
    assert model.root.clock.step_ticks == model.root.clock.tick_den
    assert history == [0., 5.]


@pytest.mark.parametrize("fault,expected", [
    ("nan-then-error", RuntimeError), ("ledger-then-error", FloatingPointError)])
def test_an_earlier_bad_step_is_reported_as_the_cause_of_a_later_error(
        fault, expected):
    model, error, _history, _checkpoints, steps = _run("lagged", 7, fault)
    assert type(error) is expected, error
    assert isinstance(error.__cause__, ValueError)
    text = str(error) + "".join(getattr(error, "__notes__", ()))
    assert "d01 step 7 failed" in text
    assert "after a failure" in text
    assert error.lagged_health["failed_step"] == 7
