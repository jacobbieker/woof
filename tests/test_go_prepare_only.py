"""``woof go --prepare-only`` stops at the seal on every prepared route.

The breakage these hold: a staged-route config (icon-global-ml) launched
with ``--prepare-only`` printed ``fetch -> prepare -> forecast``, sealed
its bundle and then integrated the whole forecast, because the run plan
``woof go`` hands a native route to had no stop after preparation.  Each
test drives the real ``woof go`` door with only the expensive stages
faked, and every forecast entry is replaced by one that fails the test
if it is reached.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof import capabilities, fetch_routes, go_cli, runplan
from woof.cli import main


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


def _completed(run_dir: Path) -> dict:
    record = runplan.read_events(run_dir / runplan.EVENTS_FILENAME)[-1]
    assert record["event"] == "completed"
    assert record["summary"]["forecast_started"] is False
    return record


def test_staged_route_stops_after_the_seal(tmp_path, monkeypatch, capsys):
    from woof import stage_cli

    # The reported source is on this route, with icon-eu, which the
    # wizard emits offline.
    assert runplan.prepared_chain_for_source("icon-global-ml") == "prepared:staged"
    assert runplan.prepared_chain_for_source("icon-eu") == "prepared:staged"
    config = _emit(tmp_path, "icon-eu", point="50.1,8.7", cycle="2026-08-18T06")
    _allow_launch(monkeypatch)
    _forbid_forecast(monkeypatch)
    data, geog, out = tmp_path / "data", tmp_path / "GEOG", tmp_path / "out"
    geog.mkdir()
    observed = []

    def fetch(argv, run_dir, **kwargs):
        observed.append("fetch")
        data.mkdir()
        (data / fetch_routes.PREP_ARGUMENTS_NAME).write_text(json.dumps({
            "schema": fetch_routes.PREP_ARGUMENTS_SCHEMA, "source": "icon-eu",
            "prep_source": "icon-eu",
            "argv": ["--source", "icon-eu", "--input-list", str(data / "inputs.txt")],
            "unbound_supplement_roles": [], "member": None, "member_set": None}))
        return {}

    def prep(argv):
        observed.append("prepare")
        root = Path(argv[argv.index("--output-root") + 1])
        root.mkdir(parents=True)
        (root / "proof.json").write_text("{}")

    monkeypatch.setattr(runplan, "_run_fetch", fetch)
    monkeypatch.setattr(runplan, "_run_prep", prep)
    monkeypatch.setattr(stage_cli, "resolve_bundle", lambda root: {
        "document": Path(root) / "proof.json", "schema": "probe",
        "source": "icon-eu", "layout": "single", "domains": 1, "payload": {}})
    monkeypatch.setattr(stage_cli, "sim_command",
                        lambda receipt, **o: [sys.executable, "-m", "runner"])
    capsys.readouterr()
    assert main(["go", str(config), "--outdir", str(out), "--run-stamp", "off",
                 "--data-dir", str(data), "--geog-root", str(geog),
                 "--prepare-only"]) == 0
    printed = capsys.readouterr().out
    assert "go: icon-eu, 1 domain(s); fetch -> prepare\n" in printed
    assert "go: forecast" not in printed and "-> forecast" not in printed
    assert observed == ["fetch", "prepare"]
    prepared = (out / "chain" / "prep").resolve()
    assert Path(_prepared_line(printed)["prepared_root"]) == prepared
    assert _completed(out)["summary"]["prepared_root"] == str(prepared)
    # The run plan carried the stop, so the run's own record says so.
    resolved = next(record for record in runplan.read_events(out / runplan.EVENTS_FILENAME)
                    if record["event"] == "resolved_plan")
    assert resolved["run_options"]["prepare_only"] is True


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
    assert "go: icon-eu, 1 domain(s); fetch -> prepare\n" in printed
    run = next(line for line in printed.splitlines() if line.startswith("Run: "))
    assert run.endswith("--prepare-only")
    assert not out.exists()


def test_hrrr_route_stops_after_the_seal(tmp_path, monkeypatch, capsys):
    from woof import stage_cli

    config = _emit(tmp_path, "hrrr", point="39,-98", cycle="2026-08-18T06")
    assert runplan.prepared_chain_for_source("hrrr") == "prepared:hrrr"
    _allow_launch(monkeypatch)
    _forbid_forecast(monkeypatch)
    data, geog, out = tmp_path / "data", tmp_path / "GEOG", tmp_path / "out"
    geog.mkdir()
    observed = []
    digests = {"proof": "a" * 64, "source_manifest": "b" * 64,
               "prepared_content": "c" * 64}

    def fetch(argv, run_dir, **kwargs):
        observed.append("fetch")
        target = Path(argv[argv.index("--out") + 1])
        target.mkdir(parents=True, exist_ok=True)
        (target / "SHA256SUMS").write_text("x", encoding="utf-8")
        return {}

    def run_stage(label, command, **kwargs):
        observed.append(label)
        root = Path(command[command.index("--output-root") + 1])
        bundle = root / "bundle"
        bundle.mkdir(parents=True)
        (root / "public-wrapper-result.json").write_text(json.dumps({
            "portable_bundle": {
                "prepared_root": str(bundle),
                "proof_sha256": digests["proof"],
                "source_manifest_sha256": digests["source_manifest"],
                "prepared_content_sha256": digests["prepared_content"],
                "experiment_config": str(config),
                "wps_namelist": command[command.index("--wps-namelist") + 1]}}))

    monkeypatch.setattr(runplan, "_run_fetch", fetch)
    monkeypatch.setattr(go_cli, "run_stage", run_stage)
    monkeypatch.setattr(go_cli, "proof_digests", lambda root: dict(digests))
    monkeypatch.setattr(stage_cli, "resolve_bundle", lambda root: {
        "document": Path(root) / "proof.json", "schema": "probe",
        "source": "hrrr", "layout": "single", "domains": 1, "payload": {}})
    capsys.readouterr()
    assert main(["go", str(config), "--outdir", str(out), "--run-stamp", "off",
                 "--data-dir", str(data), "--geog-root", str(geog),
                 "--whole-cycle", "--prepare-only"]) == 0
    printed = capsys.readouterr().out
    assert "go: hrrr, 1 domain(s); fetch -> prepare\n" in printed
    assert observed == ["fetch", "prepare"]
    prepared = (out / "chain" / "hrrr-root-prep" / "bundle").resolve()
    assert Path(_prepared_line(printed)["prepared_root"]) == prepared
    assert _completed(out)["summary"]["prepared_root"] == str(prepared)


def test_go_route_stops_after_the_seal(tmp_path, monkeypatch, capsys):
    config = _emit(tmp_path, "gfs", point="35.3,-97.5", cycle="2026-07-29T18",
                   card="24gb")
    assert runplan.prepared_chain_for_source("gfs") == "prepared:go"
    _allow_launch(monkeypatch)
    _forbid_forecast(monkeypatch)
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: tmp_path / "bridge")
    monkeypatch.setattr(go_cli, "_disk_admission", lambda *a, **k: None)
    monkeypatch.setattr(go_cli, "posts_beside_preparation", lambda plan: False)
    geog, out = tmp_path / "GEOG", tmp_path / "out"
    geog.mkdir()
    observed = []

    def run_stage(label, command, **kwargs):
        observed.append(label)
        if "--manifest-out" in command:
            manifest = Path(command[command.index("--manifest-out") + 1])
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("{}", encoding="utf-8")
        if label == "prepare":
            prepared = Path(command[command.index("--output-root") + 1])
            prepared.mkdir(parents=True, exist_ok=True)
            (prepared / "proof.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(go_cli, "_run_stage", run_stage)
    capsys.readouterr()
    assert main(["go", str(config), "--outdir", str(out), "--run-stamp", "off",
                 "--geog-root", str(geog), "--prepare-only"]) == 0
    printed = capsys.readouterr().out
    assert "go: gfs, 1 domain(s); fetch -> prepare\n" in printed
    assert observed[-1] == "prepare" and "forecast" not in observed
    assert Path(_prepared_line(printed)["prepared_root"]) == out / "prepared"


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


def test_a_forecast_that_does_not_fit_does_not_stop_its_preparation(
        tmp_path, monkeypatch, capsys):
    """The memory gate prices the forecast; --prepare-only starts none."""

    config = _emit(tmp_path, "icon-eu", point="50.1,8.7", cycle="2026-08-18T06")
    _allow_launch(monkeypatch)
    _forbid_forecast(monkeypatch)
    monkeypatch.setattr(go_cli, "memory_gate", lambda *a, **k: {
        "verdict": "forecast needs 40 GiB", "refuse": True, "warn": False})
    monkeypatch.setattr(go_cli, "_require_forecast_device",
                        lambda: pytest.fail("--prepare-only needs no forecast device"))
    executed = []
    monkeypatch.setattr(runplan, "execute_plan",
                        lambda plan, **k: executed.append(plan.run_options["prepare_only"]) or 1)
    capsys.readouterr()
    main(["go", str(config), "--outdir", str(tmp_path / "out"), "--prepare-only"])
    assert executed == [True]
    assert "preparation continues" in capsys.readouterr().err


def test_restart_roster_is_refused_with_prepare_only(tmp_path, capsys):
    config = _emit(tmp_path, "icon-eu", point="50.1,8.7", cycle="2026-08-18T06")
    capsys.readouterr()
    assert main(["go", str(config), "--prepare-only",
                 "--restart-roster", str(tmp_path / "roster.json")]) == 2
    assert "--prepare-only starts none" in capsys.readouterr().err
