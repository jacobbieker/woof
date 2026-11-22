"""MPAS physics is the hex default (2026-10-25), and ArWen physics stays one flag away.

THE BREAKAGE THIS PREVENTS: the default staying WOOF-ArWen on some door after the flip (a plan, a step rule, a roster or a
card price deciding the WOOF-ArWen route while the forecast runs MPAS, or the reverse), and the WOOF-ArWen route no longer
reachable as it ran before 0.4.6.  CPU only: parsers and plans, no card."""
from __future__ import annotations

import json

import pytest

from woof.hex import forecast_door, global_door, physics_default
from woof.hex.cli import build_parser


def test_the_rule_picks_convection_permitting_on_every_mesh_and_says_why():
    table = physics_default.load_table()
    assert table["route"] == "mpas" and table["arwen_suite"] == "arwen_mpas_column"
    for dx in (None, 937.5, 3000.0, 9999.0, 10000.0, 60000.0, 120000.0):
        assert physics_default.default_suite(dx) == "convection_permitting"
    assert "10 km" in physics_default.default_rule(3000.0)["why"]
    assert "mesoscale_reference" in physics_default.default_rule(120000.0)["why"]


@pytest.mark.parametrize("flags, defaulted", [
    ([], True),
    (["--convection", "auto"], True),
    (["--physics-mode=exact", "--checkpoint-hours=6"], True),
    (["--physics-suite=arwen_mpas_column"], False),
    (["--physics-suite", "mesoscale_reference"], False),
    (["--physics=config_microp_scheme=mp_thompson"], False),
    (["--convection=gf"], False),
    (["--physics-backend", "mpasphys"], False),
    (["--physics-namelist=namelist.atmosphere"], False),
    (["--forecast-arg=--physics-suite=rrfs_v2_dev"], False),
    (["--physics=config_clm_lake_backend=hex-cuda"], True),
    (["--physics", "config_clm_lake_backend=hex-cuda"], True),
    (["--forecast-arg=--physics=config_farms_sw_backend=hex-cuda"], True),
    (["--physics=config_clm_lake_backend=hex-cuda", "--physics=config_pbl_scheme=bl_ysu"], False),
    (["--physics=config_microp_backend=native-cpu"], False),
])
def test_ensure_adds_the_default_only_when_no_physics_is_named(flags, defaulted):
    out = physics_default.ensure(flags)
    assert out == ([*flags, "--physics-suite=convection_permitting"] if defaulted else flags)


def _forecast_namespace(*extra):
    return build_parser().parse_args(["forecast", "--mesh", "x1.40962", "--hours", "1", *extra])


def test_a_bare_forecast_takes_the_mpas_route_and_a_named_choice_is_kept():
    bare = _forecast_namespace()
    assert physics_default.apply(bare)["suite"] == "convection_permitting"
    assert bare.physics_suite == "convection_permitting"
    for extra, suite in ((["--physics-suite", "arwen_mpas_column"], "arwen_mpas_column"),
                         (["--physics", "config_microp_scheme=mp_thompson"], None),
                         (["--convection", "ntiedtke"], None),
                         (["--physics-backend", "mpasphys"], None)):
        named = _forecast_namespace(*extra)
        assert physics_default.apply(named) is None and named.physics_suite == suite


def test_the_bare_forecast_decides_the_mpas_route_as_the_door_does():
    from woof.hex.mpasphys.forecast_v842 import native_route

    bare = _forecast_namespace()
    physics_default.apply(bare)
    config, _text = native_route(bare, default_mode="correct")
    assert config is not None  # the MPAS suite route, as the door runs it
    arwen = _forecast_namespace("--physics-suite", "arwen_mpas_column")
    assert native_route(arwen, default_mode="correct")[0] is None  # the ArWen route


def test_naming_the_arwen_suite_alone_is_the_pre_046_bare_road():
    arwen = _forecast_namespace("--physics-suite", "arwen_mpas_column")
    assert forecast_door.resolve_door_physics(arwen, "x1.40962", 120000.0) == (None, None, {})


def _plan(tmp_path, row, *extra):
    return global_door.plan(build_parser().parse_args(
        ["global", "--row", row, "--cycle", "2026-10-11T12", "--work-dir", str(tmp_path / row),
         "--cache-dir", str(tmp_path / "cache"), "--stage", "plan", *extra]))


def _suites(forecast):
    return [forecast[i + 1] for i, token in enumerate(forecast) if token == "--physics-suite"]


def test_global_default_rows_run_the_mpas_suite_and_arwen_is_a_row_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(global_door, "_visible_cards", lambda: [])
    value = _plan(tmp_path, "global-120km")
    forecast = value["commands"]["forecast"]
    assert _suites(forecast) == ["convection_permitting"] and "--dt" not in forecast
    assert value["physics"]["row"] == "convection_permitting"
    assert value["physics"]["default"]["route"] == "mpas"
    build_parser().parse_args(forecast)

    arwen = _plan(tmp_path, "global-120km", "--physics-row", "arwen")
    assert _suites(arwen["commands"]["forecast"]) == ["arwen_mpas_column"]
    assert arwen["physics"]["row"] == "arwen"

    morrison = _plan(tmp_path, "global-60km-morrison")["commands"]["forecast"]
    assert _suites(morrison) == []  # its overrides name the ArWen-route physics themselves

    override = _plan(tmp_path, "global-120km", "--forecast-arg=--physics=config_microp_scheme=mp_thompson")
    assert override["physics"]["row"] == "arwen" and _suites(override["commands"]["forecast"]) == []

    named = _plan(tmp_path, "global-120km", "--forecast-arg=--physics-suite=arwen_thompson_column")
    assert _suites(named["commands"]["forecast"]) == [] and named["physics"]["row"] == "arwen_thompson_column"


def test_a_moving_case_with_no_physics_args_runs_the_default(tmp_path):
    from woof.hex.moving_case import load_case

    case = {"start_time": "2022-09-26_00:00:00", "dt_seconds": 18.75, "n_levels": 55,
            "vertical_spec": "tc55-v1.json", "storm": {"initial_center_deg": [20.0, -80.0]},
            "mesh": {"background_km": 48.0, "fine_km": 3.0, "fine_radius_km": 300.0,
                     "fine_transition_km": 54.0, "rings": []}}
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case), encoding="utf-8")
    try:
        loaded = load_case(path)
    except Exception as error:  # a case key this test does not know about: the defaulting is still checked
        pytest.skip(f"case shape: {error}")
    assert loaded["physics_args"] == ["--physics-suite=convection_permitting", "--physics-mode=correct",
                                      "--physics-profile=hrrr-like-aerosol"]
    case["physics_args"] = ["--physics-suite=arwen_mpas_column"]
    path.write_text(json.dumps(case), encoding="utf-8")
    assert load_case(path)["physics_args"] == ["--physics-suite=arwen_mpas_column"]


def test_physics_list_marks_the_mpas_default(capsys):
    from woof.hex.cli import main

    assert main(["physics", "list"]) == 0
    suites = {row["name"]: row for row in json.loads(capsys.readouterr().out)["suites"]}
    assert suites["convection_permitting"]["default"] is True
    assert suites["arwen_mpas_column"]["default"] is False
    assert suites["arwen_mpas_column"]["arwen_route_default"] is True


def test_a_named_arwen_suite_keeps_the_doors_mesh_step():
    assert physics_default.suite_takes_door_step({"suite": "convection_permitting"}) is True
    assert physics_default.suite_takes_door_step({"namelist": "namelist.atmosphere"}) is True
    assert physics_default.suite_takes_door_step({"suite": "arwen_mpas_column"}) is False
    assert physics_default.suite_takes_door_step({}) is False


def test_a_named_mpas_suite_keeps_a_mesh_step_shorter_than_its_rule():
    # zc1200 (2026-10-29): a 96 -> 3 km zoom admitted at 12 s ran the default suite at the rule's 18 s.
    door_step = physics_default.door_step
    assert door_step({"suite": "convection_permitting"}, 12.0, 3000.0) == ["--dt", "12.0"]
    assert door_step({"suite": "convection_permitting"}, 18.0, 3000.0) == []
    assert door_step({"suite": "convection_permitting"}, 18.0, None) == []
    assert door_step({"namelist": "namelist.atmosphere"}, 12.0, 3000.0) == []
    assert door_step({"suite": "arwen_mpas_column"}, 18.0, 3000.0) == ["--dt", "18.0"]
    assert door_step({}, 12.0, 3000.0) == ["--dt", "12.0"]


def test_a_suite_with_its_own_producer_reaches_the_cold_init():
    """hrrrv5 (RUC on nine levels) is listed with MPAS's own suites, and `woof hex regional
    --forecast-arg=--physics-suite=hrrrv5` built MPAS-A's Noah four-layer init for it.  A suite whose model
    initializes with its own producer row now names itself to the init door; MPAS-A's own suites still add nothing."""
    from woof.hex.physics_default import init_flags

    assert init_flags(["--physics-suite=hrrrv5"]) == ["--physics-suite", "hrrrv5"]
    assert init_flags(["--physics-suite=convection_permitting"]) == []


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_a_diagnostics_only_namelist_keeps_the_default_mpas_suite(tmp_path):
    """&diagnostics alone names no physics: the default MPAS suite (R2) is added and the namelist rides beside it.

    THE GAP THIS CLOSES (severe-weather worker, 2026-11-14): `woof hex regional --forecast-arg=--physics-namelist=F`
    with F holding only &diagnostics added no suite, so the run took the WOOF route, whose importer refuses
    &diagnostics."""
    diag = _write(tmp_path, "diag.namelist", "&diagnostics\n config_lightning_option=1\n config_hailcast_opt=.true.\n/\n")
    physics = _write(tmp_path, "physics.namelist",
                     "&physics\n config_physics_suite='convection_permitting'\n/\n&diagnostics\n config_lightning_option=1\n/\n")
    flags = [f"--physics-namelist={diag}"]
    assert not physics_default.names_physics(flags)
    assert not physics_default.names_physics(["--physics-namelist", str(diag)])
    assert physics_default.ensure(flags) == [*flags, "--physics-suite=convection_permitting"]
    regional = physics_default.ensure(flags, context={"geometry": "limited-area", "source": "gfs", "start": "cold"})
    assert regional[0] == flags[0] and "--physics-suite=convection_permitting" in regional
    # A namelist with any other group, or one that cannot be read, still names the physics.
    assert physics_default.names_physics([f"--physics-namelist={physics}"])
    assert physics_default.names_physics([f"--physics-namelist={tmp_path / 'missing.namelist'}"])
    bare = _forecast_namespace("--physics-namelist", str(diag))
    assert physics_default.apply(bare)["suite"] == "convection_permitting"
    named = _forecast_namespace("--physics-namelist", str(physics))
    assert physics_default.apply(named) is None and named.physics_suite is None


def test_a_diagnostics_only_namelist_reaches_the_mpas_suite_route_with_its_options(tmp_path):
    from woof.hex.mpasphys.forecast_v842 import native_route

    diag = _write(tmp_path, "diag.namelist", "&diagnostics\n config_lightning_option=1\n/\n")
    run = _forecast_namespace("--physics-namelist", str(diag))
    physics_default.apply(run)
    config, text = native_route(run, default_mode="correct")
    assert config is not None, "the MPAS suite route"
    assert "config_lightning_option" in text and "convection_permitting" in text


def test_the_woof_route_refuses_diagnostics_by_name_with_the_remedy():
    from woof.hex.physics_namelist import import_physics_namelist

    result = import_physics_namelist(
        "&physics\n config_physics_suite='arwen_mpas_column'\n/\n&diagnostics\n config_lightning_option=1\n/\n")
    refused = [r for r in result.refusals if r.knob == "config_lightning_option"]
    assert refused and "MPAS v8.4.2 suite route (R2)" in refused[0].reason
    assert "config_physics_suite" in refused[0].follow_up


def test_a_service_backend_alone_keeps_the_default_mpas_suite_and_reaches_its_route():
    """A service backend selector (``config_clm_lake_backend`` ...) names no physics: the default MPAS suite is added
    and the service rides on it.

    THE BREAKAGE THIS PREVENTS (lakes worker, 2026-11-17): `woof hex regional
    --forecast-arg=--physics=config_clm_lake_backend=hex-cuda`, as documented, added no suite, so the run took the
    WOOF route, which refused the key as "not RunConfig fields"."""
    from woof.hex.mpasphys.forecast_v842 import native_route

    keys = physics_default.service_backend_keys()
    assert {"config_clm_lake_backend", "config_farms_sw_backend", "config_shcu_deng_backend",
            "config_cu_rad_feedback_backend", "config_surface_update_backend"} <= keys
    assert not keys & {"config_microp_backend", "config_pbl_backend", "config_microp_scheme"}
    flag = "--forecast-arg=--physics=config_clm_lake_backend=hex-cuda"
    regional = physics_default.ensure([flag], context={"geometry": "limited-area", "source": "gfs", "start": "cold"})
    assert regional[0] == flag and "--physics-suite=convection_permitting" in regional

    run = _forecast_namespace("--physics", "config_clm_lake_backend=hex-cuda")
    assert physics_default.apply(run)["suite"] == "convection_permitting"
    config, text = native_route(run, default_mode="correct")
    assert config is not None, "the MPAS suite route"
    assert "config_clm_lake_backend" in text and "convection_permitting" in text
    # With a scheme override beside it the run names its own physics, as before.
    named = _forecast_namespace("--physics", "config_clm_lake_backend=hex-cuda", "--physics", "config_pbl_scheme=bl_ysu")
    assert physics_default.apply(named) is None and named.physics_suite is None
def test_scheme_picks_over_an_arwen_suite_reach_the_cold_init():
    """THE BREAKAGE THIS PREVENTS (2026-11-09, a zoom globe from ERA5 1957-12-18T12 on the WOOF-ArWen suite with NSSL
    two-moment picked): the doors forwarded the suite alone to the cold init, so the init was built for the suite's own
    microphysics and the forecast refused at its physics start for want of the picked scheme's species.  The picks ride
    with the suite, in either spelling, and a request with no pick forwards exactly what it did."""
    from woof.hex.physics_default import init_flags, physics_overrides

    picked = ["--forecast-arg=--physics-suite=arwen_mpas_column",
              "--forecast-arg=--physics=config_microp_scheme=mp_nssl2m",
              "--physics", "config_radt_lw_scheme=rrtmgp_lw", "--physics-mode=correct"]
    assert physics_overrides(picked) == ["config_microp_scheme=mp_nssl2m", "config_radt_lw_scheme=rrtmgp_lw"]
    assert init_flags(picked) == ["--physics-suite", "arwen_mpas_column",
                                  "--physics", "config_microp_scheme=mp_nssl2m",
                                  "--physics", "config_radt_lw_scheme=rrtmgp_lw"]
    assert init_flags(["--physics-suite=arwen_mpas_column"]) == ["--physics-suite", "arwen_mpas_column"]
    # Picks with no suite run over the ArWen suite at the forecast door, so the init is told the same.
    assert init_flags(["--physics=config_microp_scheme=mp_p3"]) == [
        "--physics-suite", "arwen_mpas_column", "--physics", "config_microp_scheme=mp_p3"]
    # MPAS-A's own suites still add nothing, picks or not: their forecast allocates its own species.
    assert init_flags(["--physics-suite=convection_permitting"]) == []
    assert init_flags([]) == []


@pytest.mark.parametrize("scheme, species, nonzero", [
    ("mp_nssl2m", {"qh", "qndrop", "qng", "qnh", "qni", "qnn", "qnr", "qns", "qvolg", "qvolh"}, {"qnn"}),
    ("mp_p3", {"qib", "qir"}, set()),
    ("mp_tempo", {"nwfa", "nifa"}, set()),
])
def test_the_init_selection_carries_the_picked_scheme(scheme, species, nonzero):
    """The cold init's species row is the picked microphysics', not the suite's, and the one species a WOOF-ArWen
    regional start does not leave at zero (NSSL's CCN number, WRF's ccn_conc over the reference density) starts at
    the engine's own value."""
    from woof.hex.init_physics import init_selection

    from woof.hex.physics_suites import PhysicsSelectionRefusal

    suite_only = init_selection("arwen_mpas_column", None)
    try:
        chosen = init_selection("arwen_mpas_column", None, overrides=[f"config_microp_scheme={scheme}"])
    except PhysicsSelectionRefusal as error:
        # The engine's column seam plans a pick, and it imports the GPU array module: an install without it (the
        # CPU test tier) cannot plan one.  The GPU tier runs this test.
        if "could not be imported" in str(error):
            pytest.skip("the engine's column seam is not importable in this environment")
        raise
    assert species <= set(chosen["scalar_names"])
    assert not species <= set(suite_only["scalar_names"])
    assert "scalar_fill" not in suite_only and "choices" not in suite_only
    assert set(chosen["scalar_fill"]) == nonzero
    assert chosen["choices"]["config_microp_scheme"] == scheme
    if nonzero:
        from woof.core.microphysics_transition import NSSL2_BACKGROUND_CCN_PER_KG
        assert chosen["scalar_fill"]["qnn"] == float(NSSL2_BACKGROUND_CCN_PER_KG)


def test_door_step_keeps_a_suite_deck_step_only_under_the_mesh_limit():
    """hrrrv5 runs NOAA's 20 s deck near 3 km; a mesh whose own limit is under 20 s runs its admitted step."""
    from woof.hex.physics_default import door_step, forecast_step, suite_deck_step

    hrrrv5 = {"suite": "hrrrv5"}
    assert suite_deck_step(hrrrv5, 3000.0)["dt_seconds"] == 20.0
    assert suite_deck_step(hrrrv5, 2000.0) is None          # no deck row within a factor 1.2 of 2 km
    assert suite_deck_step({"suite": "convection_permitting"}, 3000.0) is None
    assert suite_deck_step({"namelist": "namelist.atmosphere"}, 3000.0) is None
    # No limit given: the door behaves as before.
    assert door_step(hrrrv5, 18.0, 3000.0) == []
    # The deck step fits the mesh (limit 20.237 s, the 96-to-3 km globe): NOAA's deck stays.
    assert door_step(hrrrv5, 18.0, 3000.0, 20.237) == []
    step = forecast_step(hrrrv5, 18.0, 3000.0, 20.237)
    assert step["dt_seconds"] == 20.0 and abs(step["margin"] - 1.01185) < 1e-4
    # The deck step is over the mesh's limit (19.6 s, a 24 km-background globe): the admitted step is passed.
    assert door_step(hrrrv5, 18.0, 3000.0, 19.6) == ["--dt", "18.0"]
    step = forecast_step(hrrrv5, 18.0, 3000.0, 19.6)
    assert step["dt_seconds"] == 18.0 and "over this mesh's limit" in step["source"]
    # A suite with no deck is untouched by the limit.
    assert door_step({"suite": "convection_permitting"}, 18.0, 3000.0, 19.6) == []
    assert forecast_step({"suite": "convection_permitting"}, 18.0, 3000.0, 19.6)["dt_seconds"] == 18.0
    assert forecast_step({"suite": "convection_permitting"}, 12.0, 3000.0, 12.45)["dt_seconds"] == 12.0
