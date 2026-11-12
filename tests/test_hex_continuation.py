"""``woof resume`` / ``woof branch`` on hex runs (woof.hex.continuation).

The first group is card-free: what the verbs DECIDE (the located checkpoint, the rebuilt
``woof hex forecast`` command line, the named refusals) and what they WRITE (receipts, the
checkpoint's hash).  The last test integrates on a card and is gated on a finished hex run
staged by the caller (``WOOF_HEX_CONTINUATION_RUN``): it skips, rather than asserting less,
when none is staged.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

import woof.cli as cli
from woof.hex import continuation as hc

ARGV = ["forecast", "--mesh", "m", "--grid", "g.nc", "--init", "i.nc", "--hours", "1", "--out", "/elsewhere/out",
        "--physics-suite=arwen_thompson_column", "--checkpoint-hours", "0.5"]


def make_run(root: Path, *, argv=ARGV, request=True, labels=("2025-05-31_00.30.00",), frames=()):
    root.mkdir(parents=True, exist_ok=True)
    (root / hc.RUN_RECEIPT_NAME).write_text(json.dumps({"schema": hc.RUN_RECEIPT_SCHEMA}))
    if request:
        (root / hc.REQUEST_NAME).write_text(json.dumps(
            {"schema": hc.REQUEST_SCHEMA, "argv": list(argv), "cwd": str(root), "backend": None}))
    for label in labels:
        (root / f"checkpoint.{label}.pkl").write_bytes(b"checkpoint " + label.encode())
    for frame in frames:
        (root / frame).write_bytes(b"frame")
    return root


def value(argv, flag):
    return hc._flag_value(argv, flag)


def test_detection_is_from_receipts(tmp_path):
    assert not hc.is_hex_run(tmp_path)
    assert hc.is_hex_run(make_run(tmp_path / "run"))
    wrf = tmp_path / "wrf"
    wrf.mkdir()
    (wrf / "gpuwmrst_d01_2025-01-01_00_00_00.npz").write_bytes(b"x")
    assert not hc.is_hex_run(wrf)
    assert hc.hex_target(["resume", "cfg.toml", "--outdir", str(tmp_path / "run")]) == tmp_path / "run"
    assert hc.hex_target(["resume", "cfg.toml", "--outdir", str(wrf)]) is None
    assert hc.hex_target(["branch", "cfg.toml", "--from-run", str(wrf), "--outdir", "n"]) is None
    assert hc.hex_target(["run", str(tmp_path / "run")]) is None


def test_discovery_orders_both_routes_newest_first(tmp_path):
    r1 = make_run(tmp_path / "r1", labels=("2025-05-31_00.30.00", "2025-05-31_01.00.00", "2025-05-30_23.00.00"))
    assert [c.label for c in hc.discover_checkpoints(r1)] == [
        "2025-05-31_01.00.00", "2025-05-31_00.30.00", "2025-05-30_23.00.00"]
    r2 = make_run(tmp_path / "r2", labels=("f003", "t00001800", "f012"))
    assert [c.label for c in hc.discover_checkpoints(r2)] == ["f012", "f003", "t00001800"]
    (r2 / "checkpoint.f024.pkl.part").write_bytes(b"half")
    (r2 / "checkpoint.f024.pkl").write_bytes(b"")
    (r2 / "checkpoint.f001.hex-continuation.pkl").write_bytes(b"x")
    assert hc.discover_checkpoints(r2)[0].label == "f012"


def test_resume_rebuilds_the_forecast_command(tmp_path):
    run = make_run(tmp_path / "run", labels=("2025-05-31_00.30.00", "2025-05-31_00.15.00"))
    stale = ARGV + ["--restart-from", "/old.pkl"]
    (run / hc.REQUEST_NAME).write_text(json.dumps(
        {"schema": hc.REQUEST_SCHEMA, "argv": stale, "cwd": str(run), "backend": "cuda"}))
    plan = hc.plan_resume(run)
    assert plan.argv[0] == "forecast"
    assert value(plan.argv, "--restart-from") == str(run / "checkpoint.2025-05-31_00.30.00.pkl")
    assert plan.argv.count("--restart-from") == 1
    assert value(plan.argv, "--out") == str(run.with_name("run-resume-2025-05-31_00.30.00"))
    assert plan.outdir != run                                # the door never writes into an existing run
    assert value(plan.argv, "--hours") == "1"                # the ORIGINAL run's length
    assert value(plan.argv, "--checkpoint-hours") == "0.5"
    assert plan.backend == "cuda"
    explicit = hc.plan_resume(run, run / "checkpoint.2025-05-31_00.15.00.pkl")
    assert explicit.checkpoint.name == "checkpoint.2025-05-31_00.15.00.pkl"
    assert plan.receipt["checkpoint"]["sha256"] == hashlib.sha256(
        (run / "checkpoint.2025-05-31_00.30.00.pkl").read_bytes()).hexdigest()


def test_resume_refusals_name_the_breakage(tmp_path):
    with pytest.raises(ValueError, match="not a hex forecast directory"):
        hc.plan_resume(tmp_path)
    with pytest.raises(ValueError, match="no checkpoint"):
        hc.plan_resume(make_run(tmp_path / "a", labels=()))
    with pytest.raises(ValueError, match="no forecast-request.json"):
        hc.plan_resume(make_run(tmp_path / "b", request=False))
    with pytest.raises(ValueError, match="DA-cycling"):
        hc.plan_resume(make_run(tmp_path / "c", argv=ARGV + ["--da-analysis", "a.nc"]))
    with pytest.raises(ValueError, match="--members roster"):
        hc.plan_resume(make_run(tmp_path / "d", argv=ARGV + ["--members", "m.json"]))
    taken = make_run(tmp_path / "e")
    (tmp_path / "e-resume-2025-05-31_00.30.00").mkdir()
    (tmp_path / "e-resume-2025-05-31_00.30.00" / "cuda-history.x.nc").write_bytes(b"x")
    with pytest.raises(ValueError, match="exists"):
        hc.plan_resume(taken)
    with pytest.raises(ValueError, match="inside the run"):
        hc.plan_resume(taken, leg_out=taken / "leg")
    with pytest.raises(ValueError, match="not a hex checkpoint"):
        hc.plan_resume(make_run(tmp_path / "f"), tmp_path / "nothing.npz")


def test_branch_plan_changes_only_what_a_checkpoint_does_not_bind(tmp_path):
    run = make_run(tmp_path / "run")
    new = tmp_path / "new"
    plan = hc.plan_branch(run, new, settings=["hours=3", "checkpoint_hours=1,2"])
    assert value(plan.argv, "--hours") == "3" and value(plan.argv, "--checkpoint-hours") == "1,2"
    assert value(plan.argv, "--out") == str(new)
    assert value(plan.argv, "--restart-from") == str(run / "checkpoint.2025-05-31_00.30.00.pkl")
    assert [(r["setting"], r["from"], r["to"]) for r in plan.receipt["overrides"]] == [
        ("hours", "1", "3"), ("checkpoint_hours", "0.5", "1,2")]
    assert not new.exists()                                   # planning writes nothing


@pytest.mark.parametrize("setting, needle", [
    ("physics_suite=other", "backend_state"),
    ("--physics=config_gwdo_scheme=on", "backend_state"),
    ("convection=off", "backend_state"),
    ("horiz-mixing=off", "backend_state"),
    ("dt=10", "what the run IS"),
    ("init=other.nc", "what the run IS"),
    ("history_every_minutes=10", "what the run IS"),
    ("hourz=3", "not a setting a hex branch knows"),
    ("hours", "not a setting"),
])
def test_branch_refuses_by_name_and_leaves_no_directory(tmp_path, setting, needle):
    run = make_run(tmp_path / "run")
    new = tmp_path / "new"
    with pytest.raises(ValueError, match=needle):
        hc.plan_branch(run, new, settings=[setting])
    assert not new.exists()


def test_branch_target_must_be_fresh_and_outside_the_source(tmp_path):
    run = make_run(tmp_path / "run")
    with pytest.raises(ValueError, match="inside the run"):
        hc.plan_branch(run, run / "sub")
    with pytest.raises(ValueError, match="contains the run"):
        hc.plan_branch(run, tmp_path)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x").write_text("x")
    with pytest.raises(ValueError, match="exists"):
        hc.plan_branch(run, busy)


def test_main_hands_the_door_its_argv_and_writes_receipts_after_it(tmp_path):
    run = make_run(tmp_path / "run")
    leg = tmp_path / "run-resume-2025-05-31_00.30.00"
    seen = []

    def door(argv):
        seen.append(list(argv))
        out = Path(value(argv, "--out"))
        assert not out.exists()                               # the door's own rule: a directory that is not there
        out.mkdir()
        return 0
    assert hc.main(["resume", str(run)], runner=door) == 0
    assert seen[0][0] == "forecast" and "--restart-from" in seen[0]
    row = json.loads((leg / hc.RESUME_RECEIPT_NAME).read_text())
    assert row["verb"] == "resume" and row["checkpoint"]["label"] == "2025-05-31_00.30.00"
    assert not (run / hc.RESUME_RECEIPT_NAME).exists()        # the source run is read only

    new = tmp_path / "new"
    seen.clear()
    assert hc.main(["branch", str(run), "--outdir", str(new), "--set", "hours=2"], runner=door) == 0
    receipt = json.loads((new / hc.BRANCH_RECEIPT_NAME).read_text())
    manifest = json.loads((new / hc.BRANCH_MANIFEST_NAME).read_text())
    assert receipt["engine"] == "hex" and receipt["overrides"][0]["to"] == "2"
    assert manifest["members"][0]["sha256"] == receipt["parent"]["checkpoint"]["sha256"]
    assert manifest["access"] == "read-only"
    assert value(seen[0], "--hours") == "2"

    other = tmp_path / "other"
    seen.clear()
    assert hc.main(["branch", str(run), "--outdir", str(other), "--prepare-only"], runner=seen.append) == 0
    assert seen == [] and not other.exists()
    assert hc.main(["branch", str(run), "--outdir", str(tmp_path / "x"), "--set", "physics=1"],
                   runner=seen.append) == 2 and seen == []
    # a door that refuses before making the directory leaves no receipts behind
    refused = tmp_path / "refused"
    assert hc.main(["branch", str(run), "--outdir", str(refused)], runner=lambda argv: 2) == 2
    assert not refused.exists()


def test_front_door_routes_hex_runs_and_leaves_wrf_alone(tmp_path, monkeypatch):
    run = make_run(tmp_path / "run")
    calls = []
    monkeypatch.setattr(hc, "_hex_main", lambda argv: calls.append(list(argv)) or 0)
    monkeypatch.delenv("WOOF_BACKEND", raising=False)
    assert cli.main(["resume", "cfg.toml", "--outdir", str(run)]) == 0
    assert calls and calls[0][0] == "forecast"
    # a directory that is not a hex run is the WRF path's: no hex hook is taken
    assert hc.intercept(["resume", "cfg.toml", "--outdir", str(tmp_path / "wrf")]) is None
    assert hc.intercept(["branch", "cfg.toml", "--from-run", str(tmp_path), "--outdir", "n"]) is None


def test_the_door_records_its_command_line_without_the_restart(tmp_path):
    path = hc.write_request(tmp_path, ARGV + ["--restart-from", "x.pkl"])
    record = json.loads(path.read_text())
    assert record["schema"] == hc.REQUEST_SCHEMA and "--restart-from" not in record["argv"]
    assert record["argv"][:3] == ["forecast", "--mesh", "m"]


def test_the_start_record_writer_leaves_the_request_beside_it(tmp_path, monkeypatch):
    from woof.hex.forecast_start import write_start_record
    monkeypatch.setenv(hc.REQUEST_ENV, hc.request_text(ARGV + ["--restart-from", "x.pkl"]))
    (tmp_path / "out").mkdir()
    grid = tmp_path / "g.nc"
    grid.write_bytes(b"")
    write_start_record(tmp_path / "out", paths={"grid": grid, "init": grid}, schedule={})
    record = hc.read_request(tmp_path / "out")
    assert record["argv"][:3] == ["forecast", "--mesh", "m"] and "--restart-from" not in record["argv"]
    monkeypatch.delenv(hc.REQUEST_ENV)
    (tmp_path / "out2").mkdir()
    write_start_record(tmp_path / "out2", paths={"grid": grid, "init": grid}, schedule={})
    assert not (tmp_path / "out2" / hc.REQUEST_NAME).exists()


def _frames_after(run: Path, checkpoint) -> dict[str, str]:
    """sha256 of every history frame of ``run`` written after the checkpoint instant (R1 labels)."""
    import re
    instant = checkpoint.sort_key[1]
    rows = {}
    for path in sorted(Path(run).glob("*history*.nc")):
        found = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2})", path.name)
        if found and found.group(1) > instant:
            rows[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows


def test_resume_and_branch_reproduce_the_uninterrupted_frames_on_a_card(tmp_path):
    """E4 on a card: resume and an unchanged branch write the frames after the checkpoint byte for byte.

    ``WOOF_HEX_CONTINUATION_RUN`` is a finished hex forecast directory written with the door at this version
    and ``--checkpoint-hours`` (its scratch tree kept).  The verbs integrate the frames after the newest
    checkpoint again, and the hashes must equal the original's.
    """
    source = os.environ.get("WOOF_HEX_CONTINUATION_RUN")
    if not source or not hc.is_hex_run(Path(source)):
        pytest.skip("WOOF_HEX_CONTINUATION_RUN is not a staged finished hex run")
    source = Path(source)
    want = _frames_after(source, hc.resolve_checkpoint(source))
    assert want, "the staged run has no frame after its newest checkpoint"

    import subprocess
    import sys

    def woof(*words):
        # one forecast door per process: the device state a forecast leaves is the process's, not the verb's
        return subprocess.run([sys.executable, "-m", "woof", *words], check=False).returncode

    leg = tmp_path / "leg"
    assert woof("resume", str(source), "--leg-out", str(leg)) == 0
    assert _frames_after(leg, hc.resolve_checkpoint(source)) == want
    assert (leg / hc.RESUME_RECEIPT_NAME).is_file() and (leg / hc.REQUEST_NAME).is_file()

    branched = tmp_path / "branched"
    assert woof("branch", str(source), "--outdir", str(branched)) == 0
    assert _frames_after(branched, hc.resolve_checkpoint(source)) == want
    receipt = json.loads((branched / hc.BRANCH_RECEIPT_NAME).read_text())
    assert receipt["parent"]["checkpoint"]["sha256"] == hc._sha256(hc.resolve_checkpoint(source).path)
    assert json.loads((branched / hc.RUN_RECEIPT_NAME).read_text())["status"] == "passed"
