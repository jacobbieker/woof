"""The MPAS suite route applies the door's dycore schedule flags (2026-10-27: --acoustic-substeps was dropped)."""
import argparse

import pytest

from woof.hex import forecast_door
from woof.hex.mpasphys.forecast_v842 import namelist_option

TEXT = "&nhyd_model\nconfig_dt=24.0\n/\n&physics\nconfig_physics_suite='convection_permitting'\n/\n"


def args(**kw):
    base = dict(acoustic_substeps=None, time_integration_order=None, monotonic=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_substeps_flag_reaches_the_namelist():
    text = forecast_door.bind_native_dynamics(args(acoustic_substeps=6), TEXT)
    assert int(namelist_option(text, "nhyd_model", "config_number_of_sub_steps")) == 6
    assert float(namelist_option(text, "nhyd_model", "config_dt")) == 24.0
    assert namelist_option(text, "physics", "config_physics_suite") == "convection_permitting"


def test_no_flag_leaves_the_text_alone():
    assert forecast_door.bind_native_dynamics(args(), TEXT) == TEXT


def test_order_and_monotonic_reach_the_namelist():
    text = forecast_door.bind_native_dynamics(args(time_integration_order=3, monotonic=False), TEXT)
    assert int(namelist_option(text, "nhyd_model", "config_time_integration_order")) == 3
    assert forecast_door._namelist_value(namelist_option(text, "nhyd_model", "config_monotonic")) is False


def test_namelist_disagreeing_with_a_flag_is_refused():
    named = TEXT.replace("config_dt=24.0", "config_dt=24.0\nconfig_number_of_sub_steps=4")
    with pytest.raises(ValueError, match="two values"):
        forecast_door.bind_native_dynamics(args(acoustic_substeps=6), named)
    assert forecast_door.bind_native_dynamics(args(acoustic_substeps=4), named) == named


# --dynamics KEY=VALUE on the MPAS suite route (2026-11-04: accepted, recorded, never run).

def test_dynamics_reaches_the_namelist_in_its_group():
    a = args(dynamics=["config_coef_3rd_order=1", "config_epssm_minimum=0.5", "config_epssm_maximum=0.5"])
    text = forecast_door.bind_native_dynamics(a, TEXT)
    assert namelist_option(text, "nhyd_model", "config_coef_3rd_order") == 1.0
    assert namelist_option(text, "damping", "config_epssm_minimum") == 0.5
    assert namelist_option(text, "damping", "config_epssm_maximum") == 0.5
    assert float(namelist_option(text, "nhyd_model", "config_dt")) == 24.0
    assert namelist_option(text, "physics", "config_physics_suite") == "convection_permitting"
    # The real option is held as a real (Registry kind), and run_forecast checks the decoded value against this.
    assert a._native_dynamics_overrides == {"config_coef_3rd_order": 1.0, "config_epssm_minimum": 0.5,
                                            "config_epssm_maximum": 0.5}
    assert isinstance(a._native_dynamics_overrides["config_coef_3rd_order"], float)


def test_no_dynamics_leaves_the_text_and_arguments_alone():
    a = args(dynamics=None)
    assert forecast_door.bind_native_dynamics(a, TEXT) == TEXT
    assert not hasattr(a, "_native_dynamics_overrides")


def test_dynamics_with_a_flag_of_its_own_is_refused():
    with pytest.raises(ValueError, match="flag of its own"):
        forecast_door.bind_native_dynamics(args(dynamics=["config_number_of_sub_steps=6"]), TEXT)


def test_dynamics_the_dycore_does_not_carry_is_refused():
    with pytest.raises(ValueError, match="carries no such option"):
        forecast_door.bind_native_dynamics(args(dynamics=["config_no_such_option=1"]), TEXT)
    with pytest.raises(ValueError, match="carries no such option"):
        forecast_door.bind_native_dynamics(args(dynamics=["config_physics_suite=\"hrrrv5\""]), TEXT)


def test_dynamics_disagreeing_with_the_namelist_is_refused():
    named = TEXT.replace("config_dt=24.0", "config_dt=24.0\nconfig_coef_3rd_order=0.25")
    with pytest.raises(ValueError, match="two values"):
        forecast_door.bind_native_dynamics(args(dynamics=["config_coef_3rd_order=1.0"]), named)
    assert forecast_door.bind_native_dynamics(args(dynamics=["config_coef_3rd_order=0.25"]), named) == named


# Upper damping on the MPAS suite route (2026-11-16: the high-top declaration's runtime block was refused).

def test_the_high_top_runtime_block_reaches_the_namelist():
    from woof.hex.vertical_spec import VerticalSpec

    spec = VerticalSpec.from_file(
        __import__("pathlib").Path(forecast_door.__file__).with_name("data") / "countries" / "s2s80-v1.json")
    parser = argparse.ArgumentParser()
    forecast_door.add_forecast_arguments(parser)
    parsed, _ = parser.parse_known_args(spec.forecast_arguments())
    a = args(**{key: getattr(parsed, key) for key in (
        "cam_coefficient", "cam_damping_levels", "rayleigh_u", "damping_start_m", "radiation_seconds", "dynamics")})
    text = forecast_door.bind_native_dynamics(a, TEXT)
    damping = {key: namelist_option(text, "damping", key) for key in (
        "config_mpas_cam_coef", "config_number_cam_damping_levels", "config_rayleigh_damp_u", "config_zd",
        "config_xnutr", "config_number_rayleigh_damp_u_levels", "config_rayleigh_damp_u_timescale_days")}
    assert forecast_door._namelist_value(damping.pop("config_rayleigh_damp_u")) is True
    assert damping == {"config_mpas_cam_coef": 0.2, "config_number_cam_damping_levels": 8,
                       "config_zd": 40000.0, "config_xnutr": 0.2, "config_number_rayleigh_damp_u_levels": 6,
                       "config_rayleigh_damp_u_timescale_days": 5.0}
    assert float(namelist_option(text, "nhyd_model", "config_dt")) == 24.0


def test_damping_flags_at_their_defaults_leave_the_text_alone():
    a = args(cam_coefficient=0.0, cam_damping_levels=4, rayleigh_u=False, damping_start_m=22000.0)
    assert forecast_door.bind_native_dynamics(a, TEXT) == TEXT


def test_a_damping_flag_disagreeing_with_the_namelist_is_refused():
    named = TEXT + "&damping\nconfig_zd=30000.0\n/\n"
    with pytest.raises(ValueError, match="--damping-start-m.*two values"):
        forecast_door.bind_native_dynamics(args(damping_start_m=40000.0), named)
    assert forecast_door.bind_native_dynamics(args(damping_start_m=30000.0), named) == named


def test_a_radiation_interval_the_suite_does_not_run_is_refused():
    assert forecast_door.bind_native_dynamics(args(radiation_seconds=1800.0), TEXT) == TEXT
    with pytest.raises(ValueError, match="two radiation intervals"):
        forecast_door.bind_native_dynamics(args(radiation_seconds=900.0), TEXT)
    # An explicit 'none' is radiation on every dynamics step (24 s here), not the 30 min Registry default.
    every_step = TEXT.replace("/\n&physics\n", "/\n&physics\nconfig_radtlw_interval='none'\n"
                              "config_radtsw_interval='none'\n")
    with pytest.raises(ValueError, match=r"--radiation-seconds 1800 .*config_radtlw_interval \(24 s\)"):
        forecast_door.bind_native_dynamics(args(radiation_seconds=1800.0), every_step)
    assert forecast_door.bind_native_dynamics(args(radiation_seconds=24.0), every_step) == every_step
    with pytest.raises(ValueError, match="no config_dt"):
        forecast_door.bind_native_dynamics(args(radiation_seconds=1800.0),
                                           every_step.replace("config_dt=24.0\n", ""))


def test_the_route_dycore_carries_the_damping_levels_and_refuses_bad_ones():
    from dataclasses import replace
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex.mpasphys.config_v842 import NativeMpasDycoreConfigV842

    config = NativeMpasDycoreConfigV842()
    # MPAS-A v8.4.2 Registry.xml defaults.
    assert (config.config_number_cam_damping_levels, config.config_number_rayleigh_damp_u_levels,
            config.config_rayleigh_damp_u_timescale_days) == (4, 6, 5.0)
    replace(config, config_rayleigh_damp_u=True, config_number_rayleigh_damp_u_levels=8).validate()
    for knob, value in (("config_number_cam_damping_levels", 0), ("config_number_rayleigh_damp_u_levels", 2.0),
                        ("config_rayleigh_damp_u_timescale_days", 0.0)):
        with pytest.raises(ConfigurationRefusal, match=knob):
            replace(config, **{knob: value}).validate()
