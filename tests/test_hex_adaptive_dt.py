"""Host-only tests of the hex adaptive time step (woof.hex.adaptive_dt, HEX-WRF parity item 4.1).

The controller itself is WRF's, graded against the Fortran in tests/test_adaptive_timestep*.py; these tests
pin what the hex clock adds around it: exact landings, the acoustic substep rule, the cadence deadlines, the
restart memory and the refusals by name.
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import pytest

from woof.core.adaptive_timestep import AdaptiveTimestepController
from woof.hex import adaptive_dt
from woof.hex.adaptive_dt import (HexAdaptiveClock, acoustic_ceiling, acoustic_substeps, physics_cadence,
                                  resolve_options)


def _arguments(**values):
    base = dict(adaptive_timestep="on", adaptive_target_cfl=None, adaptive_target_hcfl=None,
                adaptive_max_increase_pct=None, adaptive_starting_dt=None, adaptive_min_dt=None,
                adaptive_max_dt=None)
    base.update(values)
    return SimpleNamespace(**base)


def _options(**values):
    return resolve_options(_arguments(**values), row_dt=30.0, row_substeps=4, order=2, center_latitude=33.8)


def _run(clock, rates):
    """Drive a clock to its end with a CFL-rate function of the step; returns the plans."""

    plans = []
    while not clock.finished:
        plan = clock.propose()
        vert, horiz = rates(plan)
        clock.commit(vert_rate=vert, horiz_rate=horiz)
        plans.append(plan)
    return plans


def test_off_is_none_and_refuses_nothing():
    assert resolve_options(SimpleNamespace(adaptive_timestep="off"), row_dt=30.0, row_substeps=4, order=2,
                           center_latitude=0.0) is None
    assert adaptive_dt.refusal(SimpleNamespace(adaptive_timestep="off", local_timestep=True)) is None
    assert adaptive_dt.forwarded_arguments(SimpleNamespace(adaptive_timestep="off")) == []


def test_defaults_keep_wrf_ratios_to_the_row_step():
    options = _options()
    assert (options.starting_dt, options.min_dt, options.max_dt) == (30, Fraction(45, 2), 60)
    assert (options.target_cfl, options.target_hcfl, options.max_step_increase_pct) == (1.2, 0.84, 5)


@pytest.mark.parametrize("attribute,value,flag", [
    ("local_timestep", True, "--local-timestep"),
    ("fullstep_local_timestep", True, "--fullstep-local-timestep"),
    ("iau_window_seconds", 3600.0, "--iau-window-seconds"),
    ("les_model", "3d_smagorinsky", "--les-model"),
    ("chemistry_config", "chem.json", "--chemistry-config"),
    ("members", "3", "--members"),
])
def test_incompatible_options_are_refused_by_name(attribute, value, flag):
    problem = adaptive_dt.refusal(_arguments(**{attribute: value}))
    assert problem is not None and flag in problem
    with pytest.raises(ValueError, match=flag):
        _options(**{attribute: value})


def test_unset_incompatible_options_pass():
    assert adaptive_dt.refusal(_arguments(local_timestep=False, les_model="off", stochastic=None,
                                          members=None, dt="row")) is None


def test_tropical_preset_inside_and_outside_the_tropics():
    options = resolve_options(_arguments(adaptive_timestep="tropical"), row_dt=30.0, row_substeps=4, order=2,
                              center_latitude=20.9)
    assert options.target_cfl == 1.0 and options.min_dt == 30 and options.max_dt == 60
    with pytest.raises(ValueError, match="tropics"):
        resolve_options(_arguments(adaptive_timestep="tropical"), row_dt=30.0, row_substeps=4, order=2,
                        center_latitude=33.8)


def test_clamps_out_of_order_and_off_lattice_are_refused():
    with pytest.raises(ValueError, match="out of order"):
        _options(adaptive_min_dt="40")
    with pytest.raises(ValueError, match="1/64"):
        _options(adaptive_max_dt="45.01")


def test_acoustic_substeps_hold_the_longest_acoustic_step_the_row_ran():
    row = Fraction(30)
    # MPAS-A v8.4.1's two substeps at the row step: dt_dyn/2 = 5 s at order 2.
    assert adaptive_dt.longest_acoustic_step(row, 4, 2) == 5
    for dt in (Fraction(15), Fraction(30), Fraction(3001, 64), Fraction(60), Fraction(90)):
        n = acoustic_substeps(dt, row, 4, 2)
        assert n >= 4 and n % 2 == 0
        assert max(adaptive_dt.acoustic_steps(dt, n, 2)) <= 5
    assert acoustic_substeps(Fraction(60), row, 4, 2) == 4
    assert acoustic_substeps(Fraction(61), row, 4, 2) == 6
    assert acoustic_ceiling(row, 4, 2) is None
    # Third order: the single first-stage acoustic step dt/9 bounds the step.
    assert acoustic_ceiling(row, 2, 3) == 45
    assert acoustic_ceiling(row, 4, 3) == 45
    options = resolve_options(_arguments(), row_dt=30.0, row_substeps=4, order=3, center_latitude=0.0)
    assert options.max_dt == 45
    with pytest.raises(ValueError, match="acoustic step"):
        resolve_options(_arguments(adaptive_max_dt="50"), row_dt=30.0, row_substeps=4, order=3,
                        center_latitude=0.0)


def test_calm_flow_grows_five_percent_and_lands_on_every_frame_exactly():
    options = _options()
    frames = [Fraction(600 * k) for k in range(7)]
    clock = HexAdaptiveClock(options, end=Fraction(3600), frames=frames, periods=[Fraction(1800)])
    plans = _run(clock, lambda plan: (0.0, 0.0))
    ends = {plan.end for plan in plans}
    assert set(frames[1:]) <= ends
    assert clock.elapsed == 3600
    assert sum(plan.dt for plan in plans) == 3600
    assert all((plan.dt * adaptive_dt.PRECISION).denominator == 1 for plan in plans)
    # Binary64 holds every step and every sum exactly.
    assert float(sum(plan.dt for plan in plans)) == sum(float(plan.dt) for plan in plans)
    assert max(plan.dt for plan in plans) == options.max_dt
    # The first growth is WRF's full increase factor from the starting step (calc_dt's negligible-CFL arm,
    # then the 5 % bound), quantised to 1/64 s: INT(1.05 * 64 + 0.5) = 67.
    assert plans[1].dt == Fraction(30 * 67, 64)


def test_two_step_lookahead_never_takes_a_sliver():
    options = _options(adaptive_min_dt="1", adaptive_max_dt="60")
    clock = HexAdaptiveClock(options, end=Fraction(1000), frames=[Fraction(1000)])
    plans = _run(clock, lambda plan: (0.0, 0.0))
    assert clock.elapsed == 1000
    # The landing pair is split, so neither of the last two steps is a sliver of the one before.
    assert plans[-1].dt >= plans[-3].dt / 2


def test_strong_flow_holds_the_courant_target():
    options = _options(adaptive_min_dt="1")
    clock = HexAdaptiveClock(options, end=Fraction(7200), frames=[Fraction(7200)])
    # |u_n|/dcEdge = 0.84/40 per second: the horizontal target is met at a 40 s step.
    plans = _run(clock, lambda plan: (0.0, 0.84 / 40.0))
    settled = [plan.dt for plan in plans[20:-2]]
    assert settled and all(Fraction(36) <= dt <= Fraction(42) for dt in settled)


def test_vertical_courant_above_target_cuts_the_step():
    options = _options(adaptive_min_dt="1")
    clock = HexAdaptiveClock(options, end=Fraction(600), frames=[Fraction(600)])
    first = clock.propose()
    clock.commit(vert_rate=3.0 / float(first.dt), horiz_rate=0.0)
    second = clock.propose()
    # WRF: target - (cfl - target)/2 over cfl = (1.2 - 0.9)/3 = 0.1, the floor, of 30 s; on the 1/64 lattice
    # INT(0.1 * 64 + 0.5) = 6.
    assert second.dt == Fraction(30 * 6, 64)


def test_min_equals_max_pins_the_step():
    options = _options(adaptive_starting_dt="24", adaptive_min_dt="24", adaptive_max_dt="24")
    clock = HexAdaptiveClock(options, end=Fraction(3600), frames=[Fraction(1800), Fraction(3600)])
    plans = _run(clock, lambda plan: (0.5, 0.01))
    assert {plan.dt for plan in plans} == {24}
    assert len(plans) == 150


def test_restart_state_reproduces_the_continuous_sequence():
    options = _options()
    frames = [Fraction(900 * k) for k in range(5)]

    def rates(plan):
        t = float(plan.start)
        return (0.02 + 1e-5 * t, 0.004 + 2e-6 * t)

    straight = HexAdaptiveClock(options, end=Fraction(3600), frames=frames, periods=[Fraction(1800)])
    whole = _run(straight, rates)
    first = HexAdaptiveClock(options, end=Fraction(3600), frames=frames, periods=[Fraction(1800)])
    head = []
    while first.elapsed < 1800:
        plan = first.propose()
        first.commit(vert_rate=rates(plan)[0], horiz_rate=rates(plan)[1])
        head.append(plan)
    assert first.elapsed == 1800
    payload = first.state()
    resumed = HexAdaptiveClock(options, end=Fraction(3600), frames=frames, periods=[Fraction(1800)],
                               start=Fraction(1800))
    resumed.restore(payload)
    tail = _run(resumed, rates)
    assert [(p.step, p.dt, p.substeps) for p in head + tail] == [(p.step, p.dt, p.substeps) for p in whole]
    with pytest.raises(ValueError, match="adaptive options differ"):
        HexAdaptiveClock(_options(adaptive_max_dt="50"), end=Fraction(3600)).restore(payload)


def test_abandoned_step_moves_nothing():
    clock = HexAdaptiveClock(_options(), end=Fraction(600), frames=[Fraction(600)])
    plan = clock.propose()
    clock.abandon()
    again = clock.propose()
    assert (again.step, again.dt, clock.elapsed) == (plan.step, plan.dt, 0)


def test_non_finite_wind_is_refused():
    clock = HexAdaptiveClock(_options(), end=Fraction(600), frames=[Fraction(600)])
    clock.propose()
    with pytest.raises(FloatingPointError):
        clock.commit(vert_rate=float("nan"), horiz_rate=0.0)


def test_cadence_deadlines_and_welds():
    cadence = physics_cadence(row_dt=Fraction(30), radiation_seconds=1800.0, surface_pbl_seconds=30.0,
                              cumulus_seconds=30.0)
    assert cadence.periods() == (Fraction(1800),)
    assert cadence.due(Fraction(0)) == {"radiation": True, "surface_pbl": None, "cumulus": None}
    assert cadence.due(Fraction(3171, 100))["radiation"] is False
    assert cadence.due(Fraction(1800))["radiation"] is True
    held = physics_cadence(row_dt=Fraction(30), radiation_seconds=900.0, surface_pbl_seconds=180.0,
                           cumulus_seconds=None)
    assert held.periods() == (Fraction(900), Fraction(180))
    assert held.due(Fraction(360)) == {"radiation": False, "surface_pbl": True, "cumulus": None}


def test_the_controller_is_wrfs():
    clock = HexAdaptiveClock(_options(), end=Fraction(600))
    assert isinstance(clock.controller, AdaptiveTimestepController)


def test_forwarded_arguments_round_trip_through_the_driver_parser():
    import argparse

    door = _arguments(adaptive_target_cfl=1.0, adaptive_max_dt="45")
    argv = adaptive_dt.forwarded_arguments(door)
    parser = argparse.ArgumentParser()
    adaptive_dt.add_arguments(parser)
    parsed = parser.parse_args(argv)
    assert (parsed.adaptive_timestep, parsed.adaptive_target_cfl, parsed.adaptive_max_dt) == ("on", 1.0, "45")


def test_adaptive_step_config_validates_as_admitted_and_keeps_the_payload():
    import dataclasses

    from woof.hex.config_v841 import V841MpasColumnPhysicsConfig
    from woof.hex.errors import ConfigurationRefusal

    admitted = V841MpasColumnPhysicsConfig(config_dt=30.0, config_bldt_seconds=30.0, config_cudt_seconds=30.0,
                                           config_radt_seconds=1800.0)
    admitted.validate()
    # 1800 s of radiation is no whole number of 2010/64 s steps: a fixed clock refuses that step ...
    with pytest.raises(ConfigurationRefusal):
        dataclasses.replace(admitted, config_dt=2010 / 64).validate()
    # ... and the adaptive step, which lands on the radiation deadline instead, validates as admitted.
    step = adaptive_dt.adaptive_step_config(admitted, dt=2010 / 64, substeps=6)
    step.validate()
    assert isinstance(step, V841MpasColumnPhysicsConfig)
    assert (step.config_dt, step.config_number_of_sub_steps) == (2010 / 64, 6)
    assert [f.name for f in dataclasses.fields(step)] == [f.name for f in dataclasses.fields(admitted)]
    again = adaptive_dt.adaptive_step_config(step, dt=40.0, substeps=6)
    assert type(again) is type(step) and again.config_dt == 40.0
