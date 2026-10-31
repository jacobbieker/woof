"""The namelist door honours use_theta_m = 1 where its branches are wired.

WRF's Registry default for an omitted &dynamics/use_theta_m is 1 on the V4
line (Registry.EM_COMMON:2864), and the door already booked omission as 1.
It now integrates theta_m (RunConfig.use_theta_m = 1, written to [shared])
wherever woof.core.theta_m.refusal admits the physics, and keeps booking the
declared dry-theta substitution where it does not (a fused-finish
microphysics scheme, an RTHFTEN cumulus scheme, the WRF-file door).  RunConfig
itself keeps woof's default 0, so no TOML configuration changes.
"""
from __future__ import annotations

import dataclasses

import pytest

from test_namelist_import import INPUT_TEXT, _pair
from woof.config import ConfigurationRefused, RunConfig, validate_run_config
from woof.namelist_import import import_namelists, theta_m_decision


def _kessler(text: str) -> str:
    """The fixture with Kessler and no cumulus on both domains."""
    text = text.replace(" mp_physics = 10, 10,", " mp_physics = 1, 1,")
    return text.replace(" cu_physics = 1, 0,", " cu_physics = 0, 0,")


@pytest.mark.parametrize("spelling", [" use_theta_m = 1,", ""])
def test_wired_physics_imports_the_theta_m_prognostic(tmp_path, spelling):
    text = _kessler(INPUT_TEXT).replace(" use_theta_m = 0,", spelling)
    toml_text, report = import_namelists(*_pair(tmp_path, inp=text))
    assert "use_theta_m = 1" in toml_text
    assert not any(item.key == "use_theta_m" for item in report.substitutions)
    assert not any(item.key == "use_theta_m" for item in report.fixed)


def test_unwired_physics_keeps_the_declared_substitution(tmp_path):
    text = INPUT_TEXT.replace(" use_theta_m = 0,", "")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=text))
    assert "use_theta_m = 1" not in toml_text
    entry = next(item for item in report.substitutions
                 if item.key == "use_theta_m")
    assert (entry.wrf_value, entry.gpuwm_value) == (1, 0)
    assert "mp_physics = 10" in entry.reason


def test_explicit_zero_imports_byte_identically(tmp_path):
    toml_text, report = import_namelists(*_pair(tmp_path, inp=INPUT_TEXT))
    assert "use_theta_m" not in toml_text.split("[experiment]", 1)[1]
    assert any(item.key == "use_theta_m" for item in report.fixed)


def test_the_wrf_file_route_substitutes():
    decision = theta_m_decision(1, wrf_boundary_use_theta_m=1)
    assert decision.substituted and not decision.moist_theta
    assert "WRF-file door" in decision.reason
    assert theta_m_decision(1).moist_theta


def test_runconfig_default_is_dry_and_unwired_physics_is_refused():
    assert RunConfig.__dataclass_fields__["use_theta_m"].default == 0
    base = RunConfig(nx=8, ny=8, nz=10, dx=1000.0, dy=1000.0, ztop=20000.0,
                     dt=6.0, run_seconds=60.0, moist=True, mp_physics=1)
    validate_run_config(dataclasses.replace(base, use_theta_m=1))
    with pytest.raises(ConfigurationRefused, match="mp_physics = 6"):
        validate_run_config(dataclasses.replace(base, use_theta_m=1, mp_physics=6))
    with pytest.raises(ConfigurationRefused, match="Grell-Freitas"):
        validate_run_config(dataclasses.replace(base, use_theta_m=1, cu_physics=3))
    with pytest.raises(ConfigurationRefused, match="use_theta_m must be 0 or 1"):
        validate_run_config(dataclasses.replace(base, use_theta_m=2))
    with pytest.raises(ConfigurationRefused, match="do_avgflx_em must be 0 or 1"):
        validate_run_config(dataclasses.replace(base, do_avgflx_em=3))


def test_do_avgflx_cugd_is_refused_by_name(tmp_path):
    text = INPUT_TEXT.replace("&dynamics\n", "&dynamics\n do_avgflx_cugd = 1,\n")
    with pytest.raises(Exception, match="Grell-Devenyi"):
        import_namelists(*_pair(tmp_path, inp=text))


def test_do_avgflx_em_imports_per_domain(tmp_path):
    text = INPUT_TEXT.replace("&dynamics\n", "&dynamics\n do_avgflx_em = 1, 0,\n")
    toml_text, _report = import_namelists(*_pair(tmp_path, inp=text))
    assert "do_avgflx_em = 1" in toml_text
    assert "do_avgflx_em = 0" in toml_text
