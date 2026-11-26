"""RunConfig.gf_deep_rain_evaporation_booking: admission, echo and reach (CPU).

The arithmetic is graded on the GF oracle columns in
tests/test_gf_deep_evaporation_booking_gpu.py.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from woof import correct_mode
from woof.config import RunConfig, validate_run_config
from woof.quirk_corrections import run_reads

ROOT = Path(__file__).resolve().parents[1]


def _cfg(**kw):
    base = RunConfig(nx=16, ny=16, nz=40, dx=12000., dy=12000.,
                     ztop=20000., dt=30., run_seconds=600., moist=True)
    return dataclasses.replace(base, **kw)


def _gf(**kw):
    return _cfg(cu_physics=3, bl_pbl_physics=1, sf_sfclay_physics=1, cudt_minutes=0.0, **kw)


def test_default_follows_the_physics_mode():
    # correct-mode finding FQ1: on in a default process, off in exact mode
    assert _cfg().gf_deep_rain_evaporation_booking is correct_mode.CORRECT_KEYS_DEFAULT
    exact = _gf(**correct_mode.resolve("exact").settings)
    assert exact.gf_deep_rain_evaporation_booking is False and not run_reads(exact, "gf_deep_rain_evaporation_booking")


def test_read_with_grell_freitas_only():
    cfg = validate_run_config(_gf(gf_deep_rain_evaporation_booking=True))
    assert run_reads(cfg, "gf_deep_rain_evaporation_booking")
    other = validate_run_config(_cfg(cu_physics=16, bl_pbl_physics=1, sf_sfclay_physics=1, cudt_minutes=0.0,
                                     gf_deep_rain_evaporation_booking=True))
    assert not run_reads(other, "gf_deep_rain_evaporation_booking")


@pytest.mark.parametrize("bad", [1, "true", None])
def test_non_boolean_is_refused(bad):
    with pytest.raises(ValueError, match="must be true or false"):
        validate_run_config(_gf(gf_deep_rain_evaporation_booking=bad))


def test_restart_echo_omits_the_default_only():
    from woof.io import restart
    values = dataclasses.asdict(_gf(gf_deep_rain_evaporation_booking=False))
    restart._drop_default_off_run_keys(values)
    assert "gf_deep_rain_evaporation_booking" not in values
    # on but unread (no Grell-Freitas): absent too
    values = dataclasses.asdict(_cfg(gf_deep_rain_evaporation_booking=True))
    restart._drop_default_off_run_keys(values)
    assert "gf_deep_rain_evaporation_booking" not in values
    assert restart.ECHO_ABSENT_RUN_VALUES["gf_deep_rain_evaporation_booking"] is False
    values = dataclasses.asdict(_gf(gf_deep_rain_evaporation_booking=True))
    restart._drop_default_off_run_keys(values)
    assert values["gf_deep_rain_evaporation_booking"] is True


def test_specialised_kernel_source_carries_the_define():
    from woof.core.spp_kernel_sources import specialized_source
    source = specialized_source("gf", capacity=48, defines=(("GF_DEEP_EVAP_BOOKING", 1),))
    assert "#define GF_DEEP_EVAP_BOOKING 1" in source
    assert "#if GF_DEEP_EVAP_BOOKING" in source and "#define GF_KMAX 48" in source
    with pytest.raises(ValueError):
        specialized_source("gf", defines=(("GF_DEEP_EVAP_BOOKING", 2),))


def _resolve(*physics):
    argv = [sys.executable, "-m", "woof.cli", "hex", "physics", "resolve",
            "--physics-suite", "arwen_mpas_column", "--mesh", "x1.40962"]
    for item in physics:
        argv += ["--physics", item]
    run = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
    return json.loads(run.stdout[run.stdout.index("{"):])


def test_hex_generic_seam_reaches_the_option():
    got = _resolve("cu_physics=3", "gf_deep_rain_evaporation_booking=true")
    assert got["resolved"] is True
    assert got["overrides"]["gf_deep_rain_evaporation_booking"] is True
    # The suite's cumulus is "auto" until the bind (or --convection) decides
    # it, so the setting is judged with GF here ...
    auto = _resolve("gf_deep_rain_evaporation_booking=true")
    assert auto["resolved"] is True
    # ... and with an explicit other cumulus it is not read (on by default
    # in correct mode, a correction is inert where its scheme does not run)
    other = _resolve("config_convection_scheme=cu_ntiedtke",
                     "gf_deep_rain_evaporation_booking=true")
    assert other["resolved"] is True


def test_the_bind_rejudges_a_scoped_setting_with_the_decided_cumulus():
    # ... and the bind hands the engine the cumulus it decided, where a
    # non-GF one does not read it.
    from woof.core.mpas_column_suite import column_physics_plan
    from woof.hex.physics_suites import probe_cumulus
    selection = {"mp_physics": 6, "bl_pbl_physics": 1, "sf_sfclay_physics": 1,
                 "sf_surface_physics": 4, "gf_deep_rain_evaporation_booking": True}
    assert probe_cumulus(selection)["cu_physics"] == 3
    assert probe_cumulus({**selection, "cu_physics": 16})["cu_physics"] == 16
    assert "cu_physics" not in probe_cumulus({**selection, "gf_deep_rain_evaporation_booking": False})
    column_physics_plan({**selection, "cu_physics": 3})
    column_physics_plan({**selection, "cu_physics": 16})


def _tree(tmp_path, **tables):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_gf_closure_choice import _grell_tree_toml
    from woof.experiment import load_experiment
    path = tmp_path / "tree.toml"
    path.write_text(_grell_tree_toml(**tables), encoding="utf-8")
    return [(d.run.cu_physics, run_reads(d.run, "gf_deep_rain_evaporation_booking"))
            for d in load_experiment(path).domains]


def test_experiment_shared_value_reaches_only_grell_freitas_domains(tmp_path):
    on = "gf_deep_rain_evaporation_booking = true\n"
    off = "gf_deep_rain_evaporation_booking = false\n"
    assert _tree(tmp_path, shared=on, root="cu_physics = 3\n") == [(3, True), (0, False)]
    assert _tree(tmp_path, root="cu_physics = 3\n" + on) == [(3, True), (0, False)]
    assert _tree(tmp_path, shared=off, root="cu_physics = 3\n") == [(3, False), (0, False)]
    # without a Grell-Freitas domain nothing reads it
    assert _tree(tmp_path, shared=on) == [(0, False), (0, False)]
    assert _tree(tmp_path, shared=on, root="cu_physics = 5\n") == [(5, False), (0, False)]
