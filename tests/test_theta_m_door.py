"""The namelist door honours use_theta_m = 1 where its branches are wired.

WRF's Registry default for an omitted &dynamics/use_theta_m is 1 on the V4
line (Registry.EM_COMMON:2864), and the door already booked omission as 1.
An explicit use_theta_m = 1 integrates theta_m (RunConfig.use_theta_m = 1,
written to [shared]) wherever woof.core.theta_m.refusal admits the physics --
now every ported microphysics and cumulus scheme and the WRF-file door -- and
keeps booking the declared dry-theta substitution where it does not
(km_opt = 5, mp_tend_radar).  An OMITTED key integrates theta_m only within
the scope it had before (theta_m_omitted_scope); widening that is a
project decision, not a side effect of the wiring.
RunConfig itself keeps woof's default 0, so no TOML configuration changes.
"""
from __future__ import annotations

import dataclasses

import pytest

from test_namelist_import import INPUT_TEXT, _pair
from woof.config import ConfigurationRefused, RunConfig, validate_run_config
from woof.namelist_import import (import_namelists, theta_m_decision,
                                  theta_m_unwired)


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


def test_an_explicit_one_imports_theta_m_for_every_ported_scheme(tmp_path):
    """The fixture's own physics (Morrison, Kain-Fritsch) under an explicit
    use_theta_m = 1: integrated, nothing booked."""
    text = INPUT_TEXT.replace(" use_theta_m = 0,", " use_theta_m = 1,")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=text))
    assert "use_theta_m = 1" in toml_text
    assert not any(item.key == "use_theta_m" for item in report.substitutions)


def test_an_omission_keeps_its_scope(tmp_path):
    """An omitted use_theta_m (WRF's Registry default 1) integrates theta_m
    only within the scope it always has (theta_m_omitted_scope: mp_physics
    0 or 1, no RTHFTEN cumulus, ...); outside it the import keeps booking
    the declared substitution, byte for byte, and its reason says how to
    integrate theta_m."""
    text = INPUT_TEXT.replace(" use_theta_m = 0,", "")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=text))
    assert "use_theta_m = 1" not in toml_text
    entry = next(item for item in report.substitutions
                 if item.key == "use_theta_m")
    assert (entry.wrf_value, entry.gpuwm_value) == (1, 0)
    assert "mp_physics = 10" in entry.reason
    assert "explicit &dynamics use_theta_m = 1" in entry.reason
    from woof.namelist_import import theta_m_omitted_scope
    assert theta_m_omitted_scope({"physics": {"mp_physics": [1]}}) is None
    assert theta_m_omitted_scope({"physics": {"mp_physics": [1]}},
                                 route="wrf_boundary") is not None


def test_radar_tten_keeps_the_declared_substitution():
    """mp_tend_radar = 1 is the one physics selector still unwired; its
    door refuses a namelist without the radar input outright, and the
    decision for one with it is the declared substitution."""
    why = theta_m_unwired({"physics": {"mp_physics": [8], "mp_tend_radar": [1]}})
    assert why is not None and why.startswith("mp_tend_radar = 1")
    decision = theta_m_decision(1, unwired=why)
    assert decision.substituted and not decision.moist_theta
    assert "mp_tend_radar" in decision.reason
    assert theta_m_unwired({"physics": {"mp_physics": [8], "cu_physics": [3],
                                        "bl_pbl_physics": [1]}}) is None


def test_explicit_zero_imports_byte_identically(tmp_path):
    toml_text, report = import_namelists(*_pair(tmp_path, inp=INPUT_TEXT))
    assert "use_theta_m" not in toml_text.split("[experiment]", 1)[1]
    assert any(item.key == "use_theta_m" for item in report.fixed)


def test_the_wrf_file_route_restores_thm():
    decision = theta_m_decision(1, wrf_boundary_use_theta_m=1)
    assert decision.moist_theta and not decision.substituted
    assert decision.route == "wrf_boundary"
    assert "THM" in decision.reason
    assert theta_m_decision(1).moist_theta


def test_runconfig_default_is_dry_and_every_ported_scheme_is_admitted():
    assert RunConfig.__dataclass_fields__["use_theta_m"].default == 0
    base = RunConfig(nx=8, ny=8, nz=10, dx=1000.0, dy=1000.0, ztop=20000.0,
                     dt=6.0, run_seconds=60.0, moist=True, mp_physics=1)
    validate_run_config(dataclasses.replace(base, use_theta_m=1))
    # Every ported microphysics and cumulus scheme is admitted now.
    for mp in (6, 8, 10, 28):
        validate_run_config(dataclasses.replace(base, use_theta_m=1, mp_physics=mp))
    validate_run_config(dataclasses.replace(
        base, use_theta_m=1, cu_physics=3, cudt_minutes=0.0, nz=30,
        bl_pbl_physics=1, sf_sfclay_physics=1))
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


def test_km_opt_5_keeps_the_declared_substitution():
    """SMS-3DTKE's use_theta_m = 1 arm is not ported: the run refuses it by
    name and the door books the substitution rather than writing a TOML the
    run would refuse."""
    from types import SimpleNamespace
    from woof.core.theta_m import refusal
    why = refusal(SimpleNamespace(use_theta_m=1, moist=True, mp_physics=8,
                                  cu_physics=0, km_opt=5, diff_opt=2))
    assert why is not None and "km_opt = 5 (SMS-3DTKE)" in why
    door = theta_m_unwired({"physics": {"mp_physics": [8]},
                            "dynamics": {"diff_opt": [2], "km_opt": [5]}})
    assert door == why
