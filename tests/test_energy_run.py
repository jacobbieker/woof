"""``woof energy run`` walks a plan parent-first and records a manifest.

Every test replaces the subprocess runner (``woof.energy.run._execute``)
with a fake that writes the files a real run would, so no forecast is
launched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import pytest

from woof.cli import build_parser
from woof.energy import run as energy_run
from woof.energy.contracts import ContractError, Plan, PlanDomain, dump_plan

RING = ((-4.0, 51.6), (-3.9, 51.6), (-3.9, 51.7), (-4.0, 51.7), (-4.0, 51.6))


def _domain(domain_id, topology="wrf-tiles", **overrides):
    record = dict(domain_id=domain_id, topology=topology, role="child",
                  dx_m=100.0, run_dir=f"runs/{domain_id}",
                  output_glob="wrfout_d01_*", footprint=RING,
                  config=f"{domain_id}.toml", grid_id=1)
    record.update(overrides)
    return PlanDomain(**record)


def _write_plan(tmp_path, topology, domains, configs=()):
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    for name in configs:
        (plan_dir / name).write_text("[experiment]\n")
    plan = Plan(topology=topology, dx_m=100.0, start="2026-10-01T00",
                hours=6.0, domains=list(domains))
    return dump_plan(plan, plan_dir / "plan.json")


def _tiles_plan(tmp_path, *, child_args=True):
    args = (dict(extra={"downscale_args": ["--parent-domain", "1",
                                           "--out", "runs/tile1"]})
            if child_args else {})
    args2 = dict(extra={"downscale_args": ["--out", "runs/tile2"]})
    domains = [
        _domain("tile2", parent="tile1", site_ids=("s2",), **args2),
        _domain("tile1", parent="parent", site_ids=("s1",), **args),
        _domain("parent", role="parent", config="parent.toml"),
        _domain("other", role="parent", config="other.toml"),
        _domain("tile3", parent="other", site_ids=("s3",),
                extra={"downscale_args": ["--out", "runs/tile3"]}),
    ]
    return _write_plan(tmp_path, "wrf-tiles", domains,
                       configs=("parent.toml", "other.toml"))


def _nests_plan(tmp_path):
    common = dict(topology="wrf-nests", config="nests.toml",
                  run_dir="runs/nests")
    domains = [
        _domain("d01", role="parent", grid_id=1, **common),
        _domain("d02", grid_id=2, output_glob="wrfout_d02_*",
                site_ids=("s1",), **common),
        _domain("d03", grid_id=3, output_glob="wrfout_d03_*",
                site_ids=("s2",), **common),
    ]
    return _write_plan(tmp_path, "wrf-nests", domains, configs=("nests.toml",))


def _hex_plan(tmp_path, commands=True):
    extra = ({"commands": [["hex", "cull", "--out", "runs/mesh/mesh.nc"],
                           ["woof", "hex", "forecast", "--out", "runs/mesh"]]}
             if commands else {})
    domains = [PlanDomain(domain_id="mesh", topology="hex-swath", role="mesh",
                          dx_m=100.0, run_dir="runs/mesh",
                          output_glob="history.*.nc", footprint=RING,
                          mesh={"mesh_spec": "mesh.json"}, extra=extra)]
    return _write_plan(tmp_path, "hex-swath", domains)


class FakeRunner:
    """Stands in for the subprocess: records argv and writes outputs.

    ``writes`` maps a domain run_dir (relative to the plan dir) to the file
    names the command whose argv mentions it should create; ``codes`` maps
    the same key to an exit status.
    """

    def __init__(self, writes=None, codes=None, interrupt_on=None):
        self.calls: list[list[str]] = []
        self.writes = writes or {}
        self.codes = codes or {}
        self.interrupt_on = interrupt_on

    def __call__(self, argv, cwd, log):
        assert argv[:3] == [sys.executable, "-m", "woof"]
        woof_argv = argv[3:]
        self.calls.append(woof_argv)
        log.write("fake run\n")
        key = self._key(woof_argv)
        if self.interrupt_on is not None and key == self.interrupt_on:
            raise KeyboardInterrupt
        for name in self.writes.get(key, ()):
            path = Path(cwd) / key / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * 16)
        return self.codes.get(key, 0)

    @staticmethod
    def _key(woof_argv):
        if woof_argv[0] == "go":
            return woof_argv[woof_argv.index("--outdir") + 1]
        for flag in ("--out",):
            if flag in woof_argv:
                value = woof_argv[woof_argv.index(flag) + 1]
                return value if not value.endswith(".nc") else str(
                    Path(value).parent)
        raise AssertionError(woof_argv)


def _args(plan, *, dry_run=False, only=None, resume=False):
    return argparse.Namespace(plan=str(plan), dry_run=dry_run, only=only,
                              resume=resume)


def _manifest(plan):
    return json.loads((Path(plan).parent / "runs" / "manifest.json").read_text())


TILE_WRITES = {
    "runs/parent": ["run-x/run/wrfout/wrfout_d01_2026-10-01_00:00:00"],
    "runs/other": ["wrfout_d01_2026-10-01_00:00:00"],
    "runs/tile1": ["wrfout/wrfout_d01_2026-10-01_00:00:00"],
    "runs/tile2": ["wrfout_d01_2026-10-01_00:00:00"],
    "runs/tile3": ["wrfout_d01_2026-10-01_00:00:00"],
}


# --------------------------------------------------------------------------
# order and dry run


def test_tiles_order_and_dry_run(tmp_path, monkeypatch, capsys):
    plan = _tiles_plan(tmp_path)
    monkeypatch.setattr(energy_run, "_execute",
                        lambda *a: pytest.fail("dry run launched a command"))
    assert energy_run.main(_args(plan, dry_run=True)) == 0
    record = json.loads(capsys.readouterr().out)
    order = [s["step"] for s in record["steps"]]
    assert order.index("parent") < order.index("tile1") < order.index("tile2")
    assert order.index("other") < order.index("tile3")
    by_step = {s["step"]: s for s in record["steps"]}
    assert by_step["parent"]["argv"] == [["go", "parent.toml", "--outdir",
                                          "runs/parent"]]
    assert by_step["tile1"]["argv"] == [["downscale", "runs/parent",
                                         "--parent-domain", "1",
                                         "--out", "runs/tile1"]]
    assert any("replaced at run time" in n for n in by_step["tile1"]["notes"])
    assert not (plan.parent / "runs").exists()
    parser = build_parser()
    for step in record["steps"]:
        for argv in step["argv"]:
            parsed = parser.parse_args(argv)
            if argv[0] == "go":
                assert str(parsed.config) == argv[1]
                assert str(parsed.outdir) == step["run_dir"]
            else:
                assert parsed.parent == [argv[1]]
                assert str(parsed.out) == step["run_dir"]


def test_nests_one_go_run(tmp_path, monkeypatch, capsys):
    plan = _nests_plan(tmp_path)
    assert energy_run.main(_args(plan, dry_run=True)) == 0
    record = json.loads(capsys.readouterr().out)
    assert len(record["steps"]) == 1
    step = record["steps"][0]
    assert step["domains"] == ["d01", "d02", "d03"]
    assert step["argv"] == [["go", "nests.toml", "--outdir", "runs/nests"]]

    fake = FakeRunner(writes={"runs/nests": [
        "run-a/run/wrfout/wrfout_d01_x", "run-a/run/wrfout/wrfout_d02_x",
        "run-a/run/wrfout/wrfout_d03_x"]})
    monkeypatch.setattr(energy_run, "_execute", fake)
    assert energy_run.main(_args(plan)) == 0
    assert len(fake.calls) == 1
    manifest = _manifest(plan)
    assert {d: r["status"] for d, r in manifest["domains"].items()} == {
        "d01": "complete", "d02": "complete", "d03": "complete"}
    assert manifest["domains"]["d02"]["outputs"][0]["path"].endswith(
        "wrfout_d02_x")
    assert any("beneath" in n for n in manifest["domains"]["d02"]["notes"])
    assert manifest["domains"]["d01"]["log"] == "runs/nests/energy-run.log"
    assert (plan.parent / "runs/nests/energy-run.log").read_text()


def test_nests_with_different_run_dirs_refused(tmp_path, capsys):
    common = dict(topology="wrf-nests", config="nests.toml")
    plan = _write_plan(tmp_path, "wrf-nests", [
        _domain("d01", role="parent", grid_id=1, run_dir="a", **common),
        _domain("d02", grid_id=2, run_dir="b", site_ids=("s",), **common),
    ], configs=("nests.toml",))
    assert energy_run.main(_args(plan, dry_run=True)) == 2
    assert "different run_dirs" in capsys.readouterr().err


def test_missing_config_refused(tmp_path, capsys):
    plan = _write_plan(tmp_path, "wrf-tiles",
                       [_domain("p", role="parent", config="absent.toml")])
    assert energy_run.main(_args(plan, dry_run=True)) == 2
    assert "does not exist" in capsys.readouterr().err


def test_hex_commands_in_order(tmp_path, monkeypatch, capsys):
    plan = _hex_plan(tmp_path)
    assert energy_run.main(_args(plan, dry_run=True)) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["steps"][0]["argv"] == [
        ["hex", "cull", "--out", "runs/mesh/mesh.nc"],
        ["hex", "forecast", "--out", "runs/mesh"]]
    assert any("began with 'woof'" in n for n in record["steps"][0]["notes"])
    fake = FakeRunner(writes={"runs/mesh": ["history.2026-10-01.nc"]})
    monkeypatch.setattr(energy_run, "_execute", fake)
    assert energy_run.main(_args(plan)) == 0
    assert [c[1] for c in fake.calls] == ["cull", "forecast"]
    assert _manifest(plan)["domains"]["mesh"]["status"] == "complete"


def test_hex_missing_commands_refused(tmp_path, capsys):
    plan = _hex_plan(tmp_path, commands=False)
    assert energy_run.main(_args(plan, dry_run=True)) == 2
    assert "extra['commands']" in capsys.readouterr().err


@pytest.mark.parametrize("commands", [[], [[]], [["hex", 3]], "hex forecast",
                                      [["woof"]]])
def test_hex_malformed_commands_refused(tmp_path, commands):
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    plan = Plan(topology="hex-swath", dx_m=100.0, start="2026-10-01T00",
                hours=6.0, domains=[PlanDomain(
                    domain_id="mesh", topology="hex-swath", role="mesh",
                    dx_m=100.0, run_dir="runs/mesh", output_glob="h*.nc",
                    footprint=RING, mesh={"mesh_spec": "m.json"},
                    extra={"commands": commands})])
    dump_plan(plan, plan_dir / "plan.json")
    with pytest.raises(energy_run.RunRefusal):
        energy_run.run_plan(plan_dir / "plan.json", dry_run=True)


def test_missing_downscale_args_refused(tmp_path, capsys):
    plan = _tiles_plan(tmp_path, child_args=False)
    assert energy_run.main(_args(plan, dry_run=True)) == 2
    assert "downscale_args" in capsys.readouterr().err


def test_downscale_args_without_out_refused(tmp_path):
    plan = _write_plan(tmp_path, "wrf-tiles", [
        _domain("p", role="parent", config="p.toml"),
        _domain("c", parent="p", extra={"downscale_args": ["--hours", "3"]}),
    ], configs=("p.toml",))
    with pytest.raises(energy_run.RunRefusal, match="--out"):
        energy_run.run_plan(plan, dry_run=True)


def test_bad_plan_is_refused(tmp_path, capsys):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema": "woof-energy.plan.v0"}))
    assert energy_run.main(_args(path, dry_run=True)) == 2
    assert "plan.v1" in capsys.readouterr().err


# --------------------------------------------------------------------------
# running


def test_success_marks_complete_and_downscale_reads_frames(tmp_path,
                                                          monkeypatch, capsys):
    plan = _tiles_plan(tmp_path)
    fake = FakeRunner(writes=TILE_WRITES)
    monkeypatch.setattr(energy_run, "_execute", fake)
    assert energy_run.main(_args(plan)) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True
    assert set(summary["domains"].values()) == {"complete"}
    manifest = _manifest(plan)
    assert manifest["schema"] == energy_run.MANIFEST_SCHEMA
    assert len(manifest["plan"]["sha256"]) == 64
    downscale = {c[c.index("--out") + 1]: c for c in fake.calls
                 if c[0] == "downscale"}
    # The parent's frames live under a run-stamped folder; downscale is
    # handed that folder, not the run_dir.
    assert downscale["runs/tile1"][1] == "runs/parent/run-x/run/wrfout"
    assert downscale["runs/tile2"][1] == "runs/tile1/wrfout"
    assert downscale["runs/tile3"][1] == "runs/other"
    record = manifest["domains"]["tile1"]
    assert record["returncode"] == 0
    assert record["start_utc"] and record["end_utc"]
    assert record["outputs"] == [{
        "path": "runs/tile1/wrfout/wrfout_d01_2026-10-01_00:00:00",
        "size": 16}]


def test_exit_zero_without_outputs_fails(tmp_path, monkeypatch, capsys):
    plan = _tiles_plan(tmp_path)
    writes = dict(TILE_WRITES)
    writes["runs/tile1"] = []
    monkeypatch.setattr(energy_run, "_execute", FakeRunner(writes=writes))
    assert energy_run.main(_args(plan)) == 1
    manifest = _manifest(plan)
    assert manifest["domains"]["tile1"]["status"] == "failed"
    assert "matched no new file" in manifest["domains"]["tile1"]["reason"]
    assert manifest["domains"]["tile2"]["status"] == "skipped"


def test_stale_outputs_do_not_count(tmp_path, monkeypatch):
    plan = _write_plan(tmp_path, "wrf-tiles",
                       [_domain("p", role="parent", config="p.toml")],
                       configs=("p.toml",))
    old = plan.parent / "runs/p/wrfout_d01_old"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    past = time.time() - 3600
    os.utime(old, (past, past))
    monkeypatch.setattr(energy_run, "_execute", FakeRunner())
    manifest = energy_run.run_plan(plan)
    record = manifest["domains"]["p"]
    assert record["status"] == "failed"
    assert any("predate" in n for n in record["notes"])


def test_failure_cascade_skips_dependents(tmp_path, monkeypatch, capsys):
    plan = _tiles_plan(tmp_path)
    fake = FakeRunner(writes=TILE_WRITES, codes={"runs/parent": 3})
    monkeypatch.setattr(energy_run, "_execute", fake)
    assert energy_run.main(_args(plan)) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is False
    manifest = _manifest(plan)
    domains = manifest["domains"]
    assert domains["parent"]["status"] == "failed"
    assert domains["parent"]["returncode"] == 3
    assert domains["tile1"]["status"] == "skipped"
    assert domains["tile2"]["status"] == "skipped"
    assert "parent parent failed" in domains["tile2"]["reason"]
    # The independent branch still ran.
    assert domains["other"]["status"] == "complete"
    assert domains["tile3"]["status"] == "complete"
    assert not any(c[0] == "downscale" and "runs/tile1" in c
                   for c in fake.calls)


def test_occupied_downscale_out_is_moved_aside(tmp_path, monkeypatch):
    plan = _tiles_plan(tmp_path)
    leftover = plan.parent / "runs/tile3/child.toml"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("x")
    fake = FakeRunner(writes=TILE_WRITES)
    monkeypatch.setattr(energy_run, "_execute", fake)
    manifest = energy_run.run_plan(plan)
    record = manifest["domains"]["tile3"]
    assert record["status"] == "complete"
    assert any("moved to runs/tile3.previous-" in n for n in record["notes"])
    moved = list((plan.parent / "runs").glob("tile3.previous-*"))
    assert len(moved) == 1 and (moved[0] / "child.toml").read_text() == "x"


def test_downscale_out_outside_run_dir_refused(tmp_path):
    plan = _write_plan(tmp_path, "wrf-tiles", [
        _domain("p", role="parent", config="p.toml"),
        _domain("c", parent="p",
                extra={"downscale_args": ["--out=runs/elsewhere"]}),
    ], configs=("p.toml",))
    with pytest.raises(energy_run.RunRefusal, match="not run_dir"):
        energy_run.run_plan(plan, dry_run=True)


def test_fresh_deep_outputs_found_behind_stale_top_level(tmp_path,
                                                         monkeypatch):
    plan = _write_plan(tmp_path, "wrf-tiles",
                       [_domain("p", role="parent", config="p.toml")],
                       configs=("p.toml",))
    old = plan.parent / "runs/p/wrfout_d01_old"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    past = time.time() - 3600
    os.utime(old, (past, past))
    monkeypatch.setattr(energy_run, "_execute", FakeRunner(
        writes={"runs/p": ["run-y/run/wrfout/wrfout_d01_new"]}))
    record = energy_run.run_plan(plan)["domains"]["p"]
    assert record["status"] == "complete"
    assert [o["path"] for o in record["outputs"]] == [
        "runs/p/run-y/run/wrfout/wrfout_d01_new"]
    assert any("predate" in n for n in record["notes"])


class _SlowProcess:
    """A child that ignores SIGINT; the user presses Ctrl-C again."""

    def __init__(self):
        self.waits = 0
        self.terminated = self.killed = False

    def wait(self, timeout=None):
        self.waits += 1
        if self.killed:
            return -9
        if self.waits == 1:
            raise KeyboardInterrupt
        raise energy_run.subprocess.TimeoutExpired("woof", timeout)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_second_interrupt_still_stops_the_child():
    process = _SlowProcess()
    energy_run._stop(process)
    assert process.terminated and process.killed


def test_interrupt_marks_current_failed(tmp_path, monkeypatch, capsys):
    plan = _tiles_plan(tmp_path)
    fake = FakeRunner(writes=TILE_WRITES, interrupt_on="runs/tile1")
    monkeypatch.setattr(energy_run, "_execute", fake)
    assert energy_run.main(_args(plan)) == 130
    manifest = _manifest(plan)
    assert manifest["interrupted"] is True
    assert manifest["domains"]["parent"]["status"] == "complete"
    assert manifest["domains"]["tile1"]["status"] == "failed"
    assert "interrupted" in manifest["domains"]["tile1"]["reason"]
    assert manifest["domains"]["tile2"]["status"] == "pending"


# --------------------------------------------------------------------------
# resume and --only


def test_resume_skips_complete_domains(tmp_path, monkeypatch, capsys):
    plan = _tiles_plan(tmp_path)
    first = FakeRunner(writes=TILE_WRITES, codes={"runs/tile2": 1})
    monkeypatch.setattr(energy_run, "_execute", first)
    assert energy_run.main(_args(plan)) == 1
    capsys.readouterr()
    # tile2 left a log behind in its --out; clear it as a user would.
    for leftover in (plan.parent / "runs/tile2").rglob("*"):
        leftover.unlink()

    second = FakeRunner(writes=TILE_WRITES)
    monkeypatch.setattr(energy_run, "_execute", second)
    assert energy_run.main(_args(plan, dry_run=True, resume=True)) == 0
    dry = json.loads(capsys.readouterr().out)
    actions = {s["step"]: s["action"] for s in dry["steps"]}
    assert actions["parent"].startswith("skip")
    assert actions["tile2"] == "run"
    # The dry run resolves an already-run parent to its frame directory.
    tile2 = next(s for s in dry["steps"] if s["step"] == "tile2")
    assert tile2["argv"][0][1] == "runs/tile1/wrfout"

    assert energy_run.main(_args(plan, resume=True)) == 0
    assert [c[c.index("--out") + 1] for c in second.calls] == ["runs/tile2"]
    assert _manifest(plan)["domains"]["tile2"]["status"] == "complete"


def test_resume_reruns_when_outputs_changed(tmp_path, monkeypatch):
    plan = _tiles_plan(tmp_path)
    monkeypatch.setattr(energy_run, "_execute", FakeRunner(writes=TILE_WRITES))
    energy_run.run_plan(plan)
    frame = plan.parent / "runs/other" / TILE_WRITES["runs/other"][0]
    frame.write_bytes(b"shorter")
    fake = FakeRunner(writes=TILE_WRITES)
    monkeypatch.setattr(energy_run, "_execute", fake)
    manifest = energy_run.run_plan(plan, resume=True)
    # tile3's own outputs are intact, but its parent reran, so it reruns
    # too (its old --out moved aside).
    ran = [FakeRunner._key(c) for c in fake.calls]
    assert ran == ["runs/other", "runs/tile3"]
    assert manifest["domains"]["other"]["status"] == "complete"
    assert manifest["domains"]["tile3"]["status"] == "complete"
    assert manifest["domains"]["tile1"]["status"] == "complete"


def test_only_parent_rerun_makes_unlisted_children_pending(tmp_path,
                                                           monkeypatch):
    plan = _tiles_plan(tmp_path)
    monkeypatch.setattr(energy_run, "_execute", FakeRunner(writes=TILE_WRITES))
    energy_run.run_plan(plan)
    manifest = energy_run.run_plan(plan, only=("other",))
    assert manifest["domains"]["other"]["status"] == "complete"
    assert manifest["domains"]["tile3"]["status"] == "pending"
    assert any("ran again" in n for n in manifest["domains"]["tile3"]["notes"])
    assert manifest["domains"]["tile1"]["status"] == "complete"

    # A failed rerun of a parent leaves its unlisted child pending, not
    # skipped.
    monkeypatch.setattr(energy_run, "_execute",
                        FakeRunner(writes=TILE_WRITES,
                                   codes={"runs/parent": 1}))
    manifest = energy_run.run_plan(plan, only=("parent",))
    assert manifest["domains"]["parent"]["status"] == "failed"
    assert manifest["domains"]["tile1"]["status"] == "pending"


def test_manifest_without_plan_binding_refused(tmp_path):
    plan = _tiles_plan(tmp_path)
    path = plan.parent / "runs" / "manifest.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"schema": energy_run.MANIFEST_SCHEMA,
                                "plan": "plan.json", "domains": {}}))
    with pytest.raises(energy_run.RunRefusal, match="binds no plan"):
        energy_run.run_plan(plan, resume=True, dry_run=True)


def test_resume_refuses_changed_plan(tmp_path, monkeypatch, capsys):
    plan = _tiles_plan(tmp_path)
    monkeypatch.setattr(energy_run, "_execute", FakeRunner(writes=TILE_WRITES))
    assert energy_run.main(_args(plan)) == 0
    capsys.readouterr()
    document = json.loads(plan.read_text())
    document["hours"] = 12.0
    plan.write_text(json.dumps(document))
    assert energy_run.main(_args(plan, resume=True)) == 2
    assert "plan changed" in capsys.readouterr().err


def test_only_refuses_incomplete_parent(tmp_path, monkeypatch, capsys):
    plan = _tiles_plan(tmp_path)
    monkeypatch.setattr(energy_run, "_execute",
                        lambda *a: pytest.fail("refusal launched a command"))
    assert energy_run.main(_args(plan, only=("tile1",))) == 2
    assert "parent parent is not complete" in capsys.readouterr().err
    assert not (plan.parent / "runs" / "manifest.json").exists()


def test_only_unknown_domain_refused(tmp_path):
    plan = _tiles_plan(tmp_path)
    with pytest.raises(energy_run.RunRefusal, match="not in the plan"):
        energy_run.run_plan(plan, only=("nope",), dry_run=True)


def test_only_with_parent_listed_or_complete(tmp_path, monkeypatch):
    plan = _tiles_plan(tmp_path)
    fake = FakeRunner(writes=TILE_WRITES)
    monkeypatch.setattr(energy_run, "_execute", fake)
    manifest = energy_run.run_plan(plan, only=("parent", "tile1"))
    assert [FakeRunner._key(c) for c in fake.calls] == ["runs/parent",
                                                        "runs/tile1"]
    assert manifest["domains"]["tile2"]["status"] == "pending"
    fake.calls.clear()
    manifest = energy_run.run_plan(plan, only=("tile2",))
    assert [FakeRunner._key(c) for c in fake.calls] == ["runs/tile2"]
    assert manifest["domains"]["parent"]["status"] == "complete"
    assert manifest["domains"]["tile2"]["status"] == "complete"


def test_only_refuses_foreign_manifest(tmp_path, monkeypatch):
    plan = _tiles_plan(tmp_path)
    monkeypatch.setattr(energy_run, "_execute", FakeRunner(writes=TILE_WRITES))
    energy_run.run_plan(plan, only=("parent",))
    document = json.loads(plan.read_text())
    document["hours"] = 12.0
    plan.write_text(json.dumps(document))
    with pytest.raises(energy_run.RunRefusal, match="different plan"):
        energy_run.run_plan(plan, only=("tile1",))


def test_corrupt_manifest_refused(tmp_path):
    plan = _tiles_plan(tmp_path)
    path = plan.parent / "runs" / "manifest.json"
    path.parent.mkdir()
    path.write_text("{not json")
    with pytest.raises(energy_run.RunRefusal, match="not JSON"):
        energy_run.run_plan(plan, dry_run=True)


def test_cli_wires_run(tmp_path):
    plan = _tiles_plan(tmp_path)
    args = build_parser().parse_args(
        ["energy", "run", str(plan), "--dry-run", "--only", "parent"])
    assert args.only == ("parent",) and args.dry_run is True


def test_parent_cycle_is_a_contract_error(tmp_path):
    plan = _write_plan(tmp_path, "wrf-tiles", [
        _domain("a", parent="b", extra={"downscale_args": ["--out", "x"]}),
        _domain("b", parent="a", extra={"downscale_args": ["--out", "y"]}),
    ])
    with pytest.raises(ContractError, match="cycle"):
        energy_run.run_plan(plan, dry_run=True)
