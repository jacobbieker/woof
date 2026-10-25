"""The anchor-mint path for REGIONAL culls at sub-second timesteps.

The mint tooling was written around global meshes at 5 s and above.  A 100 m
corridor cull has an outer-step Courant limit under a second, so its mint runs
at 0.5 s or 0.25 s with convection off, on a limited-area mesh -- and every
assumption about whole seconds, global meshes, anchored controls and 55 levels
shows up there first.

What is held here, all on a CPU-only box and none of it through the gate:

* the schedule receipt at 0.5 s and 0.25 s (pure arithmetic);
* the regional mint PLAN on a synthetic cull -- every part of a mint, the
  exact commands, and each host-checkable rule as a named blocker;
* the refusals WITHOUT the candidate-mint authorization, in the mint harness
  and the campaign, before any card is touched -- and that no plan, refusal
  or marker carries the sentence;
* the sub-second fixes on the way to a card: the door's history stride, the
  regional instrument's frame cadence and source digests, and the order in
  which the binding module applies a candidate override.

No test here supplies the authorization.  Everything past that gate is a
real mint, and a real mint is run only after the user supplies the sentence.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from woof.hex import dt_admission
from woof.hex.dt_admission import DtAdmissionError
from _layout import PACKAGE_DIR

TREE_ROOT = Path(__file__).resolve().parents[1]
TOOLS = TREE_ROOT / "tools"
PLACEHOLDER = "<CANDIDATE_MINT_AUTHORIZATION>"


def _load_tool(filename: str, alias: str):
    sys.modules.pop(alias, None)
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(alias, TOOLS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mint():
    return _load_tool("mint_dt_anchor.py", "_test_regional_mint")


@pytest.fixture(scope="module")
def campaign():
    return _load_tool("run_dt_anchor_campaign.py", "_test_regional_campaign")


def _no_sentence(text: str) -> None:
    """The plan, refusal or marker never carries the authorization sentence."""

    assert dt_admission.CANDIDATE_MINT_AUTHORIZATION not in text


def synthetic_cull(
    directory: Path,
    *,
    finest_edge_m: float = 100.0,
    zone_width: int = 7,
    levels: int = 55,
    n_cells: int = 400,
    unit_sphere: bool = False,
) -> tuple[Path, Path]:
    """A tiny MPAS-shaped regional grid + init: just what the plan reads."""

    from netCDF4 import Dataset

    directory.mkdir(parents=True, exist_ok=True)
    grid = directory / "corridor.grid.nc"
    init = directory / "corridor.init.nc"
    n_edges = 3 * n_cells
    rng = np.random.default_rng(7)
    dc_edge = finest_edge_m * (1.0 + 2.0 * rng.random(n_edges))
    dc_edge[11] = finest_edge_m
    if unit_sphere:
        dc_edge = dc_edge / 6_371_229.0
    with Dataset(str(grid), "w") as dataset:
        dataset.createDimension("nCells", n_cells)
        dataset.createDimension("nEdges", n_edges)
        dataset.sphere_radius = 1.0 if unit_sphere else 6_371_229.0
        dataset.createVariable("dcEdge", "f8", ("nEdges",))[:] = dc_edge
        if zone_width:
            # interior cells 0, then ten cells in each ring 1..zone_width
            mask = np.zeros(n_cells, dtype=np.int32)
            mask[-10 * zone_width:] = np.repeat(np.arange(1, zone_width + 1), 10)
            dataset.createVariable("bdyMaskCell", "i4", ("nCells",))[:] = mask
    with Dataset(str(init), "w") as dataset:
        dataset.createDimension("nCells", n_cells)
        dataset.createDimension("nVertLevels", levels)
    return grid, init


# ---------------------------------------------------------------------------
# the schedule receipt at sub-second dt: pure arithmetic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dt, stepra", [(0.5, 1200), (0.25, 2400)])
def test_the_schedule_receipt_mints_at_sub_second_dt_with_convection_off(
    mint, dt, stepra
):
    steps = int(3600 / dt)
    receipt = mint.mint_schedule_receipt(dt, cumulus_scheme=None, run_steps=steps)
    cadences = receipt["cadences"]
    assert receipt["dt_seconds"] == dt
    assert cadences["stepra"] == stepra
    assert cadences["stepbl"] == 1
    assert cadences["stepcu"] == 0
    assert cadences["cumulus_seconds"] is None
    assert cadences["cumulus_calls_per_hour"] is None
    assert receipt["clock_closure"]["run_steps"] == steps
    assert receipt["clock_closure"]["run_seconds"] == 3600.0
    assert receipt["clock_closure"]["exact_binary64_multiples"] is True
    # the RK schedule is the proven 120 s schedule scaled, not a new shape
    rk = receipt["rk_schedule"]
    assert rk["dynamics_stage_acoustic_steps"] == [1, 3, 6]
    assert rk["scalar_stage_timesteps"] == [dt / 3.0, dt / 2.0, dt]
    assert receipt["wsm6"]["minor_loops"] == 1
    # WOOF 1.1.0: stability is the bind's, on the mesh's own dcEdge
    assert "timestep_admission" in receipt["courant"]["note"]
    assert receipt["known_answer_control"]["reproduces_archived_120s"] is True


def test_a_sub_second_dt_whose_clock_does_not_close_is_refused_by_name(mint):
    with pytest.raises(DtAdmissionError) as caught:
        mint.mint_schedule_receipt(0.3, cumulus_scheme=None, run_steps=4000)
    assert "clock closure fails" in str(caught.value)


def test_the_cli_writes_a_sub_second_receipt(mint, tmp_path, capsys):
    out = tmp_path / "schedule-receipt-dt0p25.json"
    rc = mint.main(
        ["--dt", "0.25", "--cumulus-scheme", "off", "--run-steps", "14400",
         "--out", str(out)]
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["anchored_today"] is False
    stored = json.loads(out.read_text())
    assert stored["dt_seconds"] == 0.25
    assert stored["cadences"]["stepra"] == 2400


# ---------------------------------------------------------------------------
# the regional plan
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dt", [0.5, 0.25])
def test_the_regional_plan_covers_every_part_of_a_mint(mint, tmp_path, dt):
    grid, init = synthetic_cull(tmp_path / "cull")
    facts = mint.measure_regional_mesh(grid, init)
    assert facts["regional"] and facts["boundary_zone_width"] == 7
    assert facts["n_vert_levels"] == 55
    assert facts["finest_edge_m"] == pytest.approx(100.0)

    plan = mint.regional_mint_plan(
        dt, mesh_row="corridor-100m", facts=facts, grid=str(grid), init=str(init),
        work=str(tmp_path / "work"), card="RTX A6000",
    )
    text = json.dumps(plan)
    _no_sentence(text)
    assert plan["runnable"] is True, plan["blockers"]
    assert plan["timestep_evidence"] == "experimental-unanchored"
    checks = plan["checks"]
    assert checks["courant"]["admitted"]
    from woof.hex.dt_admission import stability_limit_seconds

    assert checks["courant"]["limit_seconds"] == pytest.approx(
        stability_limit_seconds(checks["courant"]["finest_edge_m"]))
    assert checks["radiation_steps"] == int(600 / dt)
    assert checks["clock_closes_in_binary64"] and checks["dt_ms_exact"]
    assert checks["regional"] and not checks["already_anchored"]

    parts = plan["parts"]
    assert set(parts) == {
        "schedule_receipt", "integration_anchor", "physics_health",
        "regional_forecast_mint_pair",
    }
    # 1. host only, no authorization
    schedule = parts["schedule_receipt"]
    assert schedule["needs_authorization"] is False
    assert schedule["receipt"]["dt_seconds"] == dt
    assert schedule["command"][schedule["command"].index("--dt") + 1] == f"{dt:g}"
    assert "off" in schedule["command"]

    # 2. the integration pair, authorization by NAME only
    integration = parts["integration_anchor"]
    assert integration["needs_authorization"] is True
    assert integration["steps_per_arm"] == int(3600 / dt)
    command = integration["command"]
    assert command[command.index("--authorization") + 1] == PLACEHOLDER
    assert command[command.index("--convection") + 1] == "off"
    assert "--lbc-dir" in command
    assert "--authorization" not in integration["dry_run_command"]
    assert "--dry-run" in integration["dry_run_command"]

    # 3. no anchored timestep fits a 100 m mesh, so the control halves the step
    control = parts["physics_health"]["control"]
    assert control["kind"] == "step-halving-unanchored"
    assert control["control_dt_seconds"] == dt / 2.0
    assert command[command.index("--control-dt") + 1] == f"{dt / 2.0:g}"
    assert parts["physics_health"]["verdict"].startswith("NOT MEASURED")

    # 4. the dry regional pair: frames land on whole seconds, class key exact
    pair = parts["regional_forecast_mint_pair"]
    assert pair["needs_authorization"] is False
    stride_seconds = pair["history_every_steps"] * dt
    assert stride_seconds == round(stride_seconds)
    assert pair["steps_per_arm"] % pair["history_every_steps"] == 0
    assert pair["class_key"]["dt_ms"] == int(dt * 1000)
    assert pair["class_key"]["n_vert_levels"] == 55
    assert pair["class_key"]["boundary_zone_width"] == 7
    for arm in pair["arm_commands"]:
        assert "--mint-anchor" in arm
        assert arm[arm.index("--dt") + 1] == f"{dt:g}"
        assert float(arm[arm.index("--len-disp") + 1]) == pytest.approx(100.0)
    assert pair["arm_commands"][0][-1] != pair["arm_commands"][1][-1]


def test_a_unit_sphere_grid_is_scaled_to_metres(mint, tmp_path):
    grid, init = synthetic_cull(tmp_path / "cull", unit_sphere=True)
    facts = mint.measure_regional_mesh(grid, init)
    assert facts["dc_edge_scaled_from_unit_sphere"] is True
    assert facts["finest_edge_m"] == pytest.approx(100.0)


def test_the_plan_blocks_a_dt_past_the_meshes_courant_limit(mint, tmp_path):
    grid, init = synthetic_cull(tmp_path / "cull", finest_edge_m=50.0)
    facts = mint.measure_regional_mesh(grid, init)
    coarse = mint.regional_mint_plan(0.5, mesh_row="corridor-50m", facts=facts)
    assert coarse["runnable"] is False
    assert any("Courant limit" in blocker for blocker in coarse["blockers"])
    fine = mint.regional_mint_plan(0.25, mesh_row="corridor-50m", facts=facts)
    assert fine["runnable"] is True, fine["blockers"]


def test_the_plan_refuses_a_global_mesh(mint, tmp_path):
    grid, init = synthetic_cull(tmp_path / "global", zone_width=0)
    facts = mint.measure_regional_mesh(grid, init)
    assert facts["regional"] is False
    plan = mint.regional_mint_plan(0.5, mesh_row="x", facts=facts)
    assert plan["runnable"] is False
    assert any("global mesh" in blocker for blocker in plan["blockers"])


def test_the_plan_refuses_a_dt_the_class_key_cannot_hold(mint, tmp_path):
    grid, init = synthetic_cull(tmp_path / "cull")
    facts = mint.measure_regional_mesh(grid, init)
    # 1/16 s closes its clock and divides 600 s, but is 62.5 ms
    plan = mint.regional_mint_plan(0.0625, mesh_row="x", facts=facts, hours=0.25)
    assert plan["runnable"] is False
    assert any("milliseconds" in blocker for blocker in plan["blockers"])


def test_the_plan_refuses_an_inexact_sub_second_dt(mint, tmp_path):
    grid, init = synthetic_cull(tmp_path / "cull")
    facts = mint.measure_regional_mesh(grid, init)
    plan = mint.regional_mint_plan(0.3, mesh_row="x", facts=facts, hours=0.25)
    assert plan["runnable"] is False
    assert any("schedule receipt does not mint" in b for b in plan["blockers"])


@pytest.mark.parametrize("mint_steps, needle", [(1081, "not a whole second"),
                                                (0, "at least one step")])
def test_the_plan_checks_the_regional_mint_arms_own_length(
    mint, tmp_path, mint_steps, needle
):
    grid, init = synthetic_cull(tmp_path / "cull")
    facts = mint.measure_regional_mesh(grid, init)
    plan = mint.regional_mint_plan(0.25, mesh_row="x", facts=facts, mint_steps=mint_steps)
    assert plan["runnable"] is False
    assert any(needle in blocker for blocker in plan["blockers"])


def test_the_plan_refuses_a_control_dt_that_is_not_finer(mint, tmp_path):
    grid, init = synthetic_cull(tmp_path / "cull")
    facts = mint.measure_regional_mesh(grid, init)
    for control in (0.0, 0.5, 1.0):
        plan = mint.regional_mint_plan(
            0.5, mesh_row="x", facts=facts, control_dt_seconds=control
        )
        assert plan["runnable"] is False
        assert any("step-halving control" in b for b in plan["blockers"])


def test_a_unit_sphere_grid_without_its_radius_attribute_is_still_scaled(mint, tmp_path):
    from netCDF4 import Dataset

    grid, init = synthetic_cull(tmp_path / "cull", unit_sphere=True)
    with Dataset(str(grid), "a") as dataset:
        dataset.delncattr("sphere_radius")
    facts = mint.measure_regional_mesh(grid, init)
    assert facts["dc_edge_scaled_from_unit_sphere"] is True
    assert facts["finest_edge_m"] == pytest.approx(100.0)


def test_the_plan_reads_the_level_ceiling_the_bind_applies(mint, tmp_path):
    # The bind rebinds the driver's 55 to the row's own column up to the
    # frozen lane's physics ceiling, so a cull at that ceiling has its
    # integration arms bind and only a deeper column is blocked.
    from woof.hex.vertical_spec import hex_column_level_ceiling

    ceiling = hex_column_level_ceiling()
    assert mint.FULL_PHYSICS_MAX_LEVELS == ceiling
    grid, init = synthetic_cull(tmp_path / "cull", levels=ceiling)
    facts = mint.measure_regional_mesh(grid, init)
    plan = mint.regional_mint_plan(0.5, mesh_row="x", facts=facts)
    assert plan["blockers"] == [] and plan["integration_blockers"] == []
    assert plan["runnable"] is True
    assert (
        plan["parts"]["regional_forecast_mint_pair"]["class_key"]["n_vert_levels"]
        == ceiling
    )
    assert not plan["parts"]["integration_anchor"]["blocked_by"]
    grid, init = synthetic_cull(tmp_path / "deep", levels=ceiling + 1)
    facts = mint.measure_regional_mesh(grid, init)
    plan = mint.regional_mint_plan(0.5, mesh_row="x", facts=facts)
    assert plan["runnable"] is False
    assert any(f"{ceiling + 1} levels" in b for b in plan["integration_blockers"])
    assert plan["parts"]["integration_anchor"]["blocked_by"]


def test_the_regional_plan_cli(mint, tmp_path, capsys):
    grid, init = synthetic_cull(tmp_path / "cull")
    rc = mint.main(
        ["--regional-plan", "--dt", "0.5", "--mesh", "corridor-100m",
         "--grid", str(grid), "--init", str(init), "--work", str(tmp_path / "w")]
    )
    out = capsys.readouterr().out
    _no_sentence(out)
    plan = json.loads(out)
    assert rc == 0 and plan["runnable"] is True
    assert not (tmp_path / "w").exists(), "a plan writes nothing"

    rc = mint.main(["--regional-plan", "--dt", "0.5", "--mesh", "x",
                    "--grid", str(tmp_path / "missing.nc")])
    assert rc == 1
    assert json.loads(capsys.readouterr().out)["regional_plan"] == "refused"


def test_the_integration_plan_shows_the_authorization_by_name(mint):
    plan = mint.integration_plan(0.5, "corridor-100m", hours=1.0)
    text = json.dumps(plan)
    _no_sentence(text)
    assert PLACEHOLDER in text
    assert plan["steps"] == 7200


# ---------------------------------------------------------------------------
# the authorization sentence no longer gates (WOOF 1.1.0, retired 2026-10-10)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("authorization", ["", "yes", PLACEHOLDER])
def test_a_sub_second_candidate_mint_only_labels_its_authorization(authorization):
    """``candidate_mint`` no longer refuses on the sentence: no timestep is
    refused for lacking an anchor, so the scope only labels the receipt.  It
    still writes no table row."""

    with dt_admission.candidate_mint(
        0.5, authorization=authorization, card="RTX A6000", cumulus_scheme=None,
    ):
        pass
    assert dt_admission.admitted_timestep(0.5, None) is None


def _campaign_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--dt", "0.5", "--card", "RTX A6000", "--mesh", "corridor-100m",
        "--grid", str(tmp_path / "g.nc"), "--static", str(tmp_path / "s.nc"),
        "--init", str(tmp_path / "i.nc"), "--init-source", "synthetic",
        "--lbc-dir", str(tmp_path / "lbc"), "--start-time", "2026-10-10_00:00:00",
        "--hours", "1", "--history-every-minutes", "10", "--convection", "off",
        "--gpuwm-checkout", str(tmp_path / "arwen"),
        "--work", str(tmp_path / "work"), *extra,
    ]


def test_the_campaign_refuses_to_run_without_an_authorization(campaign, tmp_path):
    with pytest.raises(SystemExit) as caught:
        campaign.main(_campaign_argv(tmp_path, "--control-dt", "0.25"))
    assert caught.value.code == 2
    assert not (tmp_path / "work").exists()


def test_the_campaign_dry_run_prints_every_leg_and_writes_nothing(
    campaign, tmp_path, capsys
):
    rc = campaign.main(_campaign_argv(tmp_path, "--control-dt", "0.25", "--dry-run"))
    out = capsys.readouterr().out
    _no_sentence(out)
    plan = json.loads(out)
    assert rc == 0 and plan["runnable"] is True
    assert not (tmp_path / "work").exists()
    assert plan["control_kind"] == "step-halving-unanchored"
    assert plan["expected_steps"] == 7200
    assert plan["cumulus_scheme"] is None
    assert plan["control_schedule_receipt"]["dt_seconds"] == 0.25
    # in the order the real run executes them: control first
    assert [leg["leg"] for leg in plan["legs"]] == ["control", "arm-a", "arm-b"]
    legs = {leg["leg"]: leg["argv"] for leg in plan["legs"]}
    for name, dt in (("control", "0.25"), ("arm-a", "0.5"), ("arm-b", "0.5")):
        leg = legs[name]
        assert "--candidate-forecast" in leg
        assert leg[leg.index("--dt") + 1] == dt
        # the authorization travels in the environment, never in argv
        assert "--authorization" not in leg
        assert leg[leg.index("--authorization-env") + 1] == (
            "WOOF_HEX_CANDIDATE_MINT_AUTHORIZATION"
        )
        assert "--lbc-dir" in leg
    assert legs["arm-a"] != legs["arm-b"]


@pytest.mark.parametrize("control", ["1", "0", "-0.25"])
def test_the_campaign_refuses_a_control_that_is_not_finer(
    campaign, tmp_path, capsys, control
):
    rc = campaign.main(_campaign_argv(tmp_path, "--control-dt", control, "--dry-run"))
    plan = json.loads(capsys.readouterr().out)
    assert rc == 1 and plan["runnable"] is False
    assert "not smaller" in plan["refusal"]


def test_the_campaign_refuses_a_control_dt_beside_a_skipped_control(campaign, tmp_path):
    with pytest.raises(SystemExit) as caught:
        campaign.main(
            _campaign_argv(tmp_path, "--control-dt", "0.25", "--skip-control", "--dry-run")
        )
    assert caught.value.code == 2


def test_the_campaign_decides_auto_convection_from_the_row_not_by_default(
    campaign, tmp_path, capsys
):
    argv = _campaign_argv(tmp_path, "--dry-run")
    argv[argv.index("--convection") + 1] = "auto"
    rc = campaign.main(argv)
    plan = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert "--convection auto needs the registered row" in plan["refusal"]


def test_the_campaign_dry_run_refuses_an_already_anchored_configuration(
    campaign, tmp_path, capsys
):
    argv = _campaign_argv(tmp_path, "--dry-run")
    argv[argv.index("--dt") + 1] = "5"
    argv[argv.index("--hours") + 1] = "1"
    rc = campaign.main(argv)
    plan = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert "already holds a registered anchor" in plan["refusal"]


def test_a_leg_receives_the_authorization_in_its_environment_and_its_log_is_redacted(
    campaign, tmp_path, monkeypatch
):
    """Exercised with a STAND-IN sentence, so the real one is never handled."""

    stand_in = "a stand-in sentence for this test only"
    monkeypatch.setattr(dt_admission, "CANDIDATE_MINT_AUTHORIZATION", stand_in)
    variable = "WOOF_HEX_CANDIDATE_MINT_AUTHORIZATION"
    log = tmp_path / "leg.log"
    rc = campaign._run(
        [sys.executable, "-c",
         f"import os; print('seen' if os.environ['{variable}'] == {stand_in!r} else 'missing')",
         stand_in],
        log,
        tmp_path,
        {variable: stand_in},
    )
    text = log.read_text()
    assert rc == 0
    assert "seen" in text
    argv_line = text.splitlines()[0]
    assert stand_in not in argv_line
    assert PLACEHOLDER in argv_line


# ---------------------------------------------------------------------------
# the sub-second fixes on the way to a card
# ---------------------------------------------------------------------------
def test_the_door_schedules_a_sub_second_row():
    from woof.hex import forecast_door

    assert forecast_door._schedule(1.0, 10, 0.5, "corridor") == (7200, 7)
    assert forecast_door._schedule(0.5, 10, 0.25, "corridor") == (7200, 4)
    with pytest.raises(forecast_door.ForecastDoorRefusal) as caught:
        forecast_door._schedule(1.0, 10, 0.7, "corridor")
    assert "--hours" in str(caught.value) or "history" in str(caught.value)


def test_the_forecast_driver_schedules_a_sub_second_dt(monkeypatch):
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "_test_sub_second_driver", PACKAGE_DIR / "drivers" / "run_cuda_v841_forecast.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "DT_SECONDS", 0.5)
    schedule = module.build_schedule(
        hours=1.0, history_every_minutes=10, start_text="2026-10-10_00:00:00"
    )
    assert schedule["steps"] == 7200
    assert schedule["history_stride_steps"] == 1200
    assert schedule["labels"][7200] == "2026-10-10_01.00.00"
    monkeypatch.setattr(module, "DT_SECONDS", 0.7)
    with pytest.raises(ValueError):
        module.build_schedule(
            hours=0.7, history_every_minutes=10, start_text="2026-10-10_00:00:00"
        )


@pytest.fixture(scope="module")
def regional_instrument():
    return _load_tool("run_cuda_regional_forecast.py", "_test_regional_instrument")


def test_the_regional_instrument_refuses_frames_inside_one_second(regional_instrument):
    with pytest.raises(SystemExit) as caught:
        regional_instrument.history_cadence(0.5, 1080, 1)
    assert "ONE frame file" in str(caught.value.code)
    # the always-written final frame meets the last periodic one
    with pytest.raises(SystemExit):
        regional_instrument.history_cadence(0.5, 1081, 4)
    with pytest.raises(SystemExit):
        regional_instrument.history_cadence(0.25, 1082, 120)
    with pytest.raises(SystemExit):
        regional_instrument.history_cadence(0.1, 10, 1)
    assert regional_instrument.history_cadence(0.5, 1080, 120)[
        "history_stride_seconds"
    ] == 60
    assert regional_instrument.history_cadence(0.25, 1080, 120)[
        "history_stride_seconds"
    ] == 30
    # the reference arms are unchanged: 120 s, every step
    assert regional_instrument.history_cadence(120.0, 15, 1)[
        "history_stride_seconds"
    ] == 120
    # and a fractional cadence above a second, which never collided, still runs
    assert regional_instrument.history_cadence(7.5, 15, 1)["frames"] == 16


def test_the_regional_instrument_digests_the_executing_sources(regional_instrument):
    digests = regional_instrument.source_digests()
    assert len(digests) == 14
    assert "tools/run_cuda_regional_forecast.py" in digests
    assert all(len(value) == 64 for value in digests.values())


def test_a_candidate_override_is_applied_after_every_runtime_row():
    text = (PACKAGE_DIR / "drivers" / "mpas_mesh_binding.py").read_text(encoding="utf-8")
    override = text.index("MESH_BINDINGS = mesh_row_candidate.apply_overrides(MESH_BINDINGS)")
    assert override > text.index("MESH_BINDINGS = mesh_rows.apply_rows(")
    assert override > text.index("MESH_BINDINGS = cascade_row.apply_rows(")
    assert text.count("mesh_row_candidate.apply_overrides(MESH_BINDINGS)") == 1
