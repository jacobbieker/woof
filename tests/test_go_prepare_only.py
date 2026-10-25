"""`woof go --prepare-only` prepares and stops, on a host with no card,
at the seal on every prepared route.

Two breakages, both measured on CPU boxes preparing an HRRR recipe:

* the door probed the card and asked for CuPy before a run that never
  reaches the forecast ("GPU readiness is info: no device", "this command
  needs cupy"), so no CPU host could prepare;
* on the registered (HRRR and staged) routes --prepare-only was not
  handed to the chain at all, so the run went on into the forecast stage
  after its preparation and died there (cudaErrorNoDevice) or held its
  head for later leads the forecast needed.

And the one the second half holds: a staged-route config (icon-global-ml)
launched with ``--prepare-only`` printed ``fetch -> prepare -> forecast``,
sealed its bundle and then integrated the whole forecast.  Those tests
drive the real ``woof go`` door with only the expensive stages faked, and
every forecast entry is replaced by one that fails the test if it is
reached.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import woof.capabilities as capabilities
import woof.go_cli as go_cli
from woof import fetch_routes, runplan
from woof.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _hrrr_config(tmp_path: Path) -> Path:
    text = (ROOT / "configs/recipes/hrrr_v4_gsd41.toml").read_text(
        encoding="utf-8")
    head, _, _ = text.partition("[fetch]")
    path = tmp_path / "hrrr-prepare.toml"
    path.write_text(head + '[fetch]\nsource = "hrrr"\n'
                    'cycle = "2026-10-02T21"\nhours = 2\ncadence = 1\n',
                    encoding="utf-8")
    return path


def test_prepare_only_runs_the_chain_with_prepare_only_and_no_card(
        tmp_path, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("prepare-only asked for the forecast's card")

    monkeypatch.setattr(go_cli, "_require_forecast_device", refuse)
    monkeypatch.setattr(go_cli, "memory_gate", refuse)
    monkeypatch.setattr(capabilities, "require_for_command", refuse)
    monkeypatch.setattr(go_cli, "_registered_launch", refuse)
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    calls = []

    def chain(plan, *, config_path, exp, observer, run_dir, prepare_only):
        calls.append(prepare_only)
        return {"prepared_root": str(run_dir / "prepared"),
                "forecast_started": False}

    monkeypatch.setattr(runplan, "_hrrr_chain", chain)
    config = _hrrr_config(tmp_path)
    rc = main(["go", str(config), "--prepare-only",
               "--outdir", str(tmp_path / "out")])
    assert rc == 0
    assert calls == [True]


def test_a_forecast_run_still_asks_for_the_card(tmp_path, monkeypatch):
    """Only --prepare-only is exempt: a run that forecasts is refused at
    the door on a host whose GPU runtime is missing, as before."""

    class Refused(Exception):
        pass

    def refuse(*_args, **_kwargs):
        raise Refused

    monkeypatch.setattr(capabilities, "require_for_command", refuse)
    config = _hrrr_config(tmp_path)
    with pytest.raises(Refused):
        main(["go", str(config), "--outdir", str(tmp_path / "out")])


def test_prepare_only_prices_hydro_before_the_registered_producer(tmp_path, monkeypatch, capsys):
    from woof import disk_budget
    from woof.hydro import runtime
    config = _hrrr_config(tmp_path)
    calls = []
    monkeypatch.setattr(runtime, "hydro_disk_estimate", lambda *a: (2**40, "hydro fake: 1024 GiB"))
    monkeypatch.setattr(runtime, "_launch_settings", lambda *a, **k: calls.append("hydro"))
    monkeypatch.setattr(runplan, "_hrrr_chain", lambda *a, **k: calls.append("producer"))
    monkeypatch.setattr(disk_budget, "free_bytes", lambda *a: 2**30)
    monkeypatch.setattr(disk_budget, "same_disk", lambda *a: True)
    # Hydro is opt-in (2.9.4.1): the launch asks for it, so the door prices it.
    rc = main(["go", str(config), "--prepare-only", "--explain", "--hydro", "on",
               "--outdir", str(tmp_path / "out")])
    assert rc != 0 and not calls
    assert "Refused before the download and before hydro preparation, so nothing was spent." in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_a_failed_preparation_returns_its_code_without_a_traceback(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(go_cli, "_require_forecast_device", lambda: None)

    def chain(*_args, **_kwargs):
        raise go_cli.GoStageFailed(3)

    monkeypatch.setattr(runplan, "_hrrr_chain", chain)
    config = _hrrr_config(tmp_path)
    rc = main(["go", str(config), "--prepare-only",
               "--outdir", str(tmp_path / "out")])
    assert rc == 3
    assert "--prepare-only stopped: stage exited 3" in capsys.readouterr().err


def test_an_experiment_document_carries_the_card_placement():
    """The HRRR root publishes its authority through this renderer; a
    [devices] table refused it after the fetch on every multi-card run."""

    import tomllib

    from woof.experiment_document import render_experiment_document

    raw = tomllib.loads((ROOT / "configs/recipes/hrrr_v4_gsd41.toml").read_text(
        encoding="utf-8"))
    raw.pop("fetch", None)
    raw["devices"] = {"count": 2, "grid": "1x2"}
    text = render_experiment_document(raw)
    assert tomllib.loads(text)["devices"] == {"count": 2, "grid": "1x2"}


def _emit(tmp_path, source, *, point, cycle, card="16gb"):
    config = tmp_path / f"{source} area.toml"
    assert main(["domain", f"--point={point}", "--card", card,
                 "--ladder", "12", "--source", source, "--cycle", cycle,
                 "--hours", "3", "--out", str(config)]) == 0
    return config


def _forbid_forecast(monkeypatch):
    """Every way a prepared route can start a forecast fails the test."""

    def forecast(*_args, **_kwargs):
        pytest.fail("--prepare-only started a forecast")

    from woof import prepared_domain_tree_forecast, prepared_single_domain_forecast
    monkeypatch.setattr(runplan, "_staged_forecast", forecast)
    monkeypatch.setattr(runplan, "_hrrr_tree_forecast", forecast)
    monkeypatch.setattr(runplan, "_hrrr_single_chain", forecast)
    monkeypatch.setattr(runplan, "_chain_render", forecast)
    monkeypatch.setattr(go_cli, "_run_forecast", forecast)
    monkeypatch.setattr(go_cli, "_render_stage", forecast)
    monkeypatch.setattr(prepared_single_domain_forecast, "main", forecast)
    monkeypatch.setattr(prepared_domain_tree_forecast, "main", forecast)


def _allow_launch(monkeypatch):
    from woof import doctor
    monkeypatch.setattr(doctor, "_cuda_headers_check",
                        lambda: SimpleNamespace(status="verified"))
    monkeypatch.setattr(go_cli, "memory_gate", lambda *a, **k: {
        "verdict": "fits", "refuse": False, "warn": False})
    monkeypatch.setattr(go_cli, "geography_refusal", lambda *a: None)
    monkeypatch.setattr(go_cli, "render_extra_missing",
                        lambda: pytest.fail("--prepare-only renders nothing"))
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a, **k: None)
    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)


def _prepared_line(out: str) -> dict:
    result = json.loads(out.strip().splitlines()[-1])
    assert result["status"] == "PREPARED"
    return result


# The fork's route-level stop tests (staged, HRRR and go routes stopping at
# the seal through the run plan's prepare_only option, and the memory gate
# warning instead of refusing) were retired at the 1.1.0 merge: upstream's
# _prepare_only_launch runs each source's producer directly and is held by
# test_prepare_only_runs_the_chain_with_prepare_only_and_no_card and
# test_a_failed_preparation_returns_its_code_without_a_traceback above.
def _completed(run_dir: Path) -> dict:
    record = runplan.read_events(run_dir / runplan.EVENTS_FILENAME)[-1]
    assert record["event"] == "completed"
    assert record["summary"]["forecast_started"] is False
    return record


def test_staged_dry_run_names_only_fetch_and_prepare(tmp_path, monkeypatch, capsys):
    config = _emit(tmp_path, "icon-eu", point="50.1,8.7", cycle="2026-08-18T06")
    _forbid_forecast(monkeypatch)
    monkeypatch.setattr(runplan, "execute_plan",
                        lambda *a, **k: pytest.fail("a dry run executed"))
    out = tmp_path / "not-created"
    capsys.readouterr()
    assert main(["go", str(config), "--outdir", str(out), "--prepare-only",
                 "--dry-run"]) == 0
    printed = capsys.readouterr().out
    # 1.1.0's _prepare_only_launch owns --prepare-only (both lines had
    # built it); its plan line names the stop instead of a "Run:" echo.
    assert "go: icon-eu, 1 domain(s); fetch -> prepare" in printed
    assert "--prepare-only: stops at the preparation's seal" in printed
    assert "-> forecast" not in printed
    assert not out.exists()


def test_go_route_dry_run_lists_no_forecast(tmp_path, monkeypatch, capsys):
    config = _emit(tmp_path, "gfs", point="35.3,-97.5", cycle="2026-07-29T18",
                   card="24gb")
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: tmp_path / "bridge")
    capsys.readouterr()
    assert main(["go", str(config), "--outdir", str(tmp_path / "out"),
                 "--prepare-only", "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "go: gfs, 1 domain(s); fetch -> prepare\n" in printed
    assert "4. prepare" in printed
    assert "5. forecast" not in printed and "6. render" not in printed


@pytest.mark.parametrize("flag", ["--wps-namelist", "--prepared-root"])
def test_prepare_only_refuses_inputs_that_already_exist(tmp_path, monkeypatch, capsys, flag):
    config = _emit(tmp_path, "icon-eu", point="50.1,8.7", cycle="2026-08-18T06")
    monkeypatch.setattr(runplan, "execute_plan",
                        lambda *a, **k: pytest.fail("must refuse first"))
    _forbid_forecast(monkeypatch)
    existing = config.with_suffix(".namelist.wps") if flag == "--wps-namelist" else tmp_path
    capsys.readouterr()
    assert main(["go", str(config), "--outdir", str(tmp_path / "out"),
                 "--prepare-only", flag, str(existing)]) == 2
    assert "--prepare-only fetches and prepares" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_the_run_plan_option_is_a_prepared_route_stop(tmp_path):
    config = _emit(tmp_path, "icon-eu", point="50.1,8.7", cycle="2026-08-18T06")

    def plan(route="prepared", **options):
        raw = {"schema": runplan.PLAN_SCHEMA, "name": "probe", "route": route,
               "config": {"path": str(config)}, "output_root": str(tmp_path / "run"),
               "run_options": {"prepare_only": True, **options}}
        return runplan.build_plan(raw, source="test", base_dir=tmp_path, sha256="0" * 64)

    assert plan().run_options["prepare_only"] is True
    with pytest.raises(runplan.PlanError, match="prepare_only"):
        plan(prepared_root=str(tmp_path))
    with pytest.raises(runplan.PlanError, match="prepare_only"):
        plan(route="experiment")
    with pytest.raises(runplan.PlanError, match="true or false"):
        plan(prepare_only="yes")


def test_restart_roster_is_refused_with_prepare_only(tmp_path, capsys):
    config = _emit(tmp_path, "icon-eu", point="50.1,8.7", cycle="2026-08-18T06")
    capsys.readouterr()
    assert main(["go", str(config), "--prepare-only",
                 "--restart-roster", str(tmp_path / "roster.json")]) == 2
    assert "--prepare-only starts none" in capsys.readouterr().err
