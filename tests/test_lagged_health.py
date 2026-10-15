"""CPU gates for the health-check mode switch (woof.core.lagged_health)."""
from __future__ import annotations

import pytest

from woof.core.lagged_health import (ENV, HealthCheckModeRefused,
                                     resolve_health_check)


def test_lagged_is_the_default_and_says_where_it_came_from():
    policy = resolve_health_check(environ={})
    assert policy["mode"] == "lagged" and policy["source"] == "default"
    assert "before history" in policy["reason"]
    assert "never past" in policy["max_detection_lag"]


def test_argument_beats_environment_beats_default():
    assert resolve_health_check(environ={ENV: "strict"})["mode"] == "strict"
    assert resolve_health_check(
        environ={ENV: "strict"})["source"] == f"environment {ENV}"
    policy = resolve_health_check("lagged", environ={ENV: "strict"})
    assert (policy["mode"], policy["source"]) == ("lagged", "argument")
    assert resolve_health_check(" STRICT ", environ={})["mode"] == "strict"
    assert resolve_health_check(environ={ENV: "  "})["mode"] == "lagged"


def test_health_debug_forces_strict_and_names_why():
    policy = resolve_health_check("lagged", health_debug=True, environ={})
    assert policy["mode"] == "strict" and policy["requested"] == "lagged"
    assert "--health-debug" in policy["reason"]
    assert policy["max_detection_lag"] == "none"


@pytest.mark.parametrize("value", ["off", "async", "1"])
def test_an_unknown_mode_is_refused_by_name(value):
    with pytest.raises(HealthCheckModeRefused, match=repr(value)):
        resolve_health_check(value, environ={})
    with pytest.raises(HealthCheckModeRefused, match=ENV):
        resolve_health_check(environ={ENV: value})


def _recording_tree(monkeypatch):
    from test_model import _model

    _exp, model = _model()
    phases = []

    class _RecordingValidator:
        """A validator double with no ``submit``: always synchronous."""

        def __init__(self, state):
            self.state = state

        def require_healthy(self, *, phase=None):
            phases.append(phase)

    monkeypatch.setattr(
        "woof.core.health.StateHealthValidator", _RecordingValidator)
    monkeypatch.setattr("woof.core.streaming.step_health",
                        lambda *args, **kwargs: {"nan": False, "cfl": 0.0})
    monkeypatch.setattr("woof.core.dycore.step",
                        lambda *_args, **_kwargs: None)
    return model, phases


def test_executor_records_the_policy_and_keeps_synchronous_doubles(
        monkeypatch):
    from woof.core.model import execute_experiment

    monkeypatch.delenv(ENV, raising=False)
    model, phases = _recording_tree(monkeypatch)
    execute_experiment(model, pool_trim_per_period=False)
    policy = model._health_check_policy
    assert policy["mode"] == "lagged" and policy["source"] == "default"
    # Immediate reports (a stepper that is not resident) are judged at
    # once; a validator that cannot be queued runs exactly as before.
    assert policy["submitted"] == 0
    assert "post-d01-sync.d01" in phases


def test_executor_honours_the_environment_and_refuses_nonsense(monkeypatch):
    from woof.core.model import execute_experiment

    model, _phases = _recording_tree(monkeypatch)
    monkeypatch.setenv(ENV, "strict")
    execute_experiment(model, pool_trim_per_period=False)
    assert model._health_check_policy["mode"] == "strict"
    assert model._health_check_policy["source"] == f"environment {ENV}"

    model, _phases = _recording_tree(monkeypatch)
    with pytest.raises(HealthCheckModeRefused):
        execute_experiment(model, pool_trim_per_period=False,
                           health_check="sometimes")


def test_validate_state_false_records_the_gate_as_off(monkeypatch):
    from woof.core.model import execute_experiment

    model, _phases = _recording_tree(monkeypatch)
    execute_experiment(model, pool_trim_per_period=False,
                       validate_state=False)
    assert model._health_check_policy["mode"] == "off"


@pytest.mark.parametrize("module", [
    "woof.prepared_single_domain_forecast",
    "woof.prepared_domain_tree_forecast",
])
def test_both_prepared_runners_declare_the_switch(module):
    import importlib

    parser = importlib.import_module(module).build_parser()
    (action,) = [action for action in parser._actions
                 if "--health-check" in action.option_strings]
    assert tuple(action.choices) == ("lagged", "strict")
    assert action.default is None and "WOOF_HEALTH_CHECK" in action.help
