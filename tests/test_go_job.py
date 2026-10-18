"""``woof go-job``: one ``woof go`` run as a batch job, and the AWS track-run
deployment that drives it.

``woof go`` itself never runs here: the child runner is a fake that writes
the exporter's ``status.json`` the way a real run leaves it, so staging, the
exit-code map, the export resume and the result document are checked on the
CPU.  One test drives a real ``woof energy worker`` over a ``file://`` queue
with the documented ``--runner-cmd``, so the template and go-job's exit code
meet the worker's settle logic for real.  Nothing contacts a cloud.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import types

import pytest

from woof import go_job as gj
from woof.cli import build_parser
from woof.energy import deploy

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "deploy" / "examples" / "aws-track-runs.toml"
RUNNER_CMD = ("{python} -m woof go-job {job} --workdir {workdir} "
              "--result {result}")
TOML = '[experiment]\nname = "tiny"\n'


def _job(job_id: str = "R001", **extra) -> dict:
    document = {"schema": gj.JOB_SCHEMA, "job_id": job_id,
                "config_toml": TOML,
                "icechunk_url": f"file:///stores/{job_id}.icechunk"}
    document.update(extra)
    return document


def _refused(document, match: str, **kwargs):
    with pytest.raises(gj.GoJobRefused, match=match):
        gj.parse_job(document, **kwargs)


# --------------------------------------------------------------------------
# The job document


def test_a_minimal_job_takes_the_documented_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv(gj.OUTPUT_ENV, "s3://bucket/outputs/")
    (tmp_path / "f.toml").write_text(TOML)
    job = gj.parse_job({"schema": gj.JOB_SCHEMA, "job_id": "run-001",
                        "config": "f.toml"}, base=tmp_path)
    assert job.config_ref == str(tmp_path / "f.toml")
    assert job.config_name == "f.toml"
    assert job.icechunk_url == "s3://bucket/outputs/run-001.icechunk"
    assert job.products == "none" and job.go_args == () and job.files == ()
    inline = gj.parse_job(_job())
    assert inline.config_name == "R001.toml" and inline.config_ref is None


def test_relative_references_resolve_against_the_job_location():
    job = gj.parse_job(_job(config_toml=None, config="cfg/a.toml",
                            files=["a.namelist.wps",
                                   {"url": "https://h.example/x/eta.txt",
                                    "path": "levels/eta.txt"}]),
                       base="s3://bkt/jobs")
    assert job.config_ref == "s3://bkt/jobs/cfg/a.toml"
    assert [(f.ref, f.path) for f in job.files] == [
        ("s3://bkt/jobs/a.namelist.wps", "a.namelist.wps"),
        ("https://h.example/x/eta.txt", "levels/eta.txt")]


def test_companions_may_travel_inline_in_the_job(tmp_path):
    job = gj.parse_job(_job(files=[{"path": "paths/track.csv",
                                    "content": "t,lat,lon\n0,1,2\n"}]))
    assert job.files == (gj.Companion(None, "paths/track.csv",
                                      "t,lat,lon\n0,1,2\n"),)
    child = FakeChild([0])
    outcome, fields = gj.run_job(job, tmp_path, child=child,
                                 card_gib=lambda: None)
    assert outcome.exit_code == 0
    assert (tmp_path / "inputs" / "paths" / "track.csv").read_text() == \
        "t,lat,lon\n0,1,2\n"
    assert fields["inputs"][1]["source"] == "inline content"
    for bad in ({"content": "x"}, {"path": "a", "content": 3},
                {"url": "s3://b/a", "content": "x", "path": "a"}):
        _refused(_job(files=[bad]), "files\\[0\\]")


def test_push_uploads_a_folder_and_prints_its_manifest(tmp_path, capsys):
    local = tmp_path / "runs"
    (local / "sub").mkdir(parents=True)
    (local / "run-001.toml").write_text(TOML)
    (local / "sub" / "track.csv").write_text("t\n")
    (local / ".hidden").write_text("no")
    target = tmp_path / "bucket"
    args = build_parser().parse_args(["go-job-push", str(local),
                                      f"{target.as_uri()}/inputs/"])
    assert args.func(args) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["schema"] == gj.PUSH_SCHEMA
    assert [(f["path"], f["url"]) for f in manifest["files"]] == [
        ("run-001.toml", f"{target.as_uri()}/inputs/run-001.toml"),
        ("sub/track.csv", f"{target.as_uri()}/inputs/sub/track.csv")]
    assert (target / "inputs" / "sub" / "track.csv").read_text() == "t\n"
    assert not (target / "inputs" / ".hidden").exists()
    dry = gj.push_inputs(local, "s3://bkt/x", dry_run=True)
    assert [f["url"] for f in dry] == ["s3://bkt/x/run-001.toml",
                                       "s3://bkt/x/sub/track.csv"]
    with pytest.raises(gj.GoJobRefused, match="https://"):
        gj.push_inputs(local, "https://h.example/x")


@pytest.mark.parametrize("document, match", [
    (_job(icechunk="s3://b/x"), "unknown key"),
    (_job(config="s3://b/c.toml"), "exactly one of config"),
    (_job(config_toml="[[broken"), "not TOML"),
    (_job(schema="woof-energy.hindcast-job.v1"), "schema"),
    (_job(job_id="../x"), "job_id"),
    (_job(config_toml=None, config="http://h.example/c.toml"), "http://"),
    (_job(config_toml=None, config="s3://AKIA:pw@bkt/c.toml"), "credentials"),
    (_job(config_toml=None, config="https://h.example/c.toml?X-Amz-"
                                   "Signature=abc"), "query string"),
    (_job(files=[{"url": "s3://b/n.wps", "path": "../n.wps"}]), "relative"),
    (_job(files=[{"url": "s3://b/n.wps", "path": "/etc/n.wps"}]), "relative"),
    (_job(files=["s3://b/R001.toml"]), "staged twice"),
    (_job(go_args=["--icechunk", "s3://b/y"]), "icechunk_url"),
    (_job(go_args=["--outdir=/tmp"]), "--workdir"),
    (_job(go_args=["--restart", "x"]), "not a flag a job may pass"),
    (_job(go_args=["--cycle"]), "needs a value"),
    (_job(go_args=["--whole-cycle=yes"]), "takes no value"),
    (_job(icechunk_url="https://h.example/x"), "https://"),
    (_job(products="-x"), "products"),
    (_job(resources={"gpu_mem_gib": -1}), "gpu_mem_gib"),
])
def test_bad_jobs_are_refused_by_name(document, match):
    _refused(document, match)


def test_a_job_without_a_store_needs_a_prefix(monkeypatch):
    monkeypatch.delenv(gj.OUTPUT_ENV, raising=False)
    document = _job()
    del document["icechunk_url"]
    _refused(document, "no icechunk_url")
    job = gj.parse_job(document, output_prefix="file:///out")
    assert job.icechunk_url == "file:///out/R001.icechunk"


def test_go_args_are_normalised_and_passed_after_the_owned_flags(tmp_path):
    job = gj.parse_job(_job(go_args=["--cycle", "2026-08-20T00",
                                     "--keep-checkpoints=1", "--whole-cycle"],
                            products="t2m"))
    assert job.go_args == ("--cycle=2026-08-20T00", "--keep-checkpoints=1",
                           "--whole-cycle")
    argv = gj.go_command(job, tmp_path / "R001.toml", tmp_path / "go")
    assert argv[:5] == [sys.executable, "-m", "woof", "go",
                        str(tmp_path / "R001.toml")]
    assert argv[5:] == ["--outdir", str(tmp_path / "go"), "--icechunk",
                        "file:///stores/R001.icechunk", "--products", "t2m",
                        *job.go_args]
    # Every flag go-job passes is one woof go's own parser accepts.
    parsed = build_parser().parse_args(argv[3:])
    assert parsed.command == "go" and parsed.icechunk == job.icechunk_url
    assert parsed.render_products == "t2m" and parsed.cycle == "2026-08-20T00"


# --------------------------------------------------------------------------
# Job lists (AWS Batch array jobs)


def test_a_job_list_is_indexed_by_flag_or_array_environment(tmp_path):
    (tmp_path / "b.json").write_text(json.dumps(_job("B")))
    listing = {"schema": gj.LIST_SCHEMA, "jobs": [_job("A"), "b.json"]}
    document, base, source = gj.select_job(listing, tmp_path, 1, tmp_path,
                                           source="list.json")
    assert document["job_id"] == "B" and source == str(tmp_path / "b.json")
    assert gj.array_index(None, {"AWS_BATCH_JOB_ARRAY_INDEX": "0",
                                 "WOOF_WORKER_INDEX": "7"}) == 0
    assert gj.array_index(None, {"WOOF_WORKER_INDEX": "7"}) == 7
    assert gj.array_index(3, {"AWS_BATCH_JOB_ARRAY_INDEX": "0"}) == 3
    with pytest.raises(gj.GoJobRefused, match="pick one with --index"):
        gj.select_job([_job("A"), _job("B")], tmp_path, None, tmp_path,
                      source="l")
    with pytest.raises(gj.GoJobRefused, match="outside the list"):
        gj.select_job([_job("A")], tmp_path, 4, tmp_path, source="l")
    # A worker fleet's own array index is inherited by every runner and
    # says nothing about a single job document; --index 3 on one does.
    single = _job("A")
    assert gj.select_job(single, tmp_path, 5, tmp_path, source="j")[0] is single
    with pytest.raises(gj.GoJobRefused, match="needs a job list"):
        gj.select_job(single, tmp_path, 3, tmp_path, source="j",
                      explicit=True)


# --------------------------------------------------------------------------
# Transfers


def test_inputs_come_from_paths_file_urls_and_s3_with_boto3(tmp_path,
                                                            monkeypatch):
    source = tmp_path / "src.toml"
    source.write_text(TOML)
    assert gj.fetch(str(source), tmp_path / "a" / "x.toml", what="c") == len(TOML)
    assert gj.fetch(source.as_uri(), tmp_path / "b.toml", what="c") == len(TOML)
    calls = []

    class Client:
        def download_file(self, bucket, key, dest):
            calls.append((bucket, key))
            Path(dest).write_text("from s3")

        def upload_file(self, path, bucket, key):
            calls.append(("put", bucket, key, Path(path).read_text()))

    fake = types.SimpleNamespace(client=lambda name: Client())
    monkeypatch.setitem(sys.modules, "boto3", fake)
    dest = tmp_path / "s3.toml"
    gj.fetch("s3://bkt/dir/c.toml", dest, what="c")
    assert dest.read_text() == "from s3" and calls == [("bkt", "dir/c.toml")]
    gj.publish(dest, "s3://bkt/out/r.json", what="r")
    assert calls[-1] == ("put", "bkt", "out/r.json", "from s3")
    with pytest.raises(gj.GoJobRefused, match="not a file"):
        gj.fetch(str(tmp_path / "missing"), tmp_path / "m", what="c")


def test_without_boto3_s3_goes_through_rclone_and_failures_retry(
        tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setattr(gj.shutil, "which",
                        lambda name: "/usr/bin/rclone" if name == "rclone"
                        else None)
    seen = []

    def run(argv, **kwargs):
        seen.append(argv)
        if len(seen) == 1:
            return subprocess.CompletedProcess(argv, 1, "", "throttled")
        Path(argv[3]).write_text("ok")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(gj.subprocess, "run", run)
    waits = []
    gj.fetch("s3://bkt/k.toml", tmp_path / "k.toml", what="c",
             sleep=waits.append)
    assert seen[0][:3] == ["rclone", "copyto",
                           ":s3,provider=AWS,env_auth=true,region=us-west-2:"
                           "bkt/k.toml"]
    assert waits == [2.0] and (tmp_path / "k.toml").read_text() == "ok"
    # A host without any tool is the machine's fault, not the job's:
    # retryable elsewhere (exit 75), and not retried here.
    monkeypatch.setattr(gj.shutil, "which", lambda name: None)
    with pytest.raises(gj.HostUnequipped, match="boto3, rclone or the aws"):
        gj.fetch("s3://bkt/k.toml", tmp_path / "k2.toml", what="c",
                 sleep=waits.append)
    assert waits == [2.0]


def test_a_missing_object_is_permanent_and_other_errors_retry(
        tmp_path, monkeypatch):
    class Missing(Exception):
        response = {"Error": {"Code": "NoSuchKey"},
                    "ResponseMetadata": {"HTTPStatusCode": 404}}

    class Throttled(Exception):
        response = {"Error": {"Code": "SlowDown"},
                    "ResponseMetadata": {"HTTPStatusCode": 503}}

    errors = [Missing()]

    class Client:
        def download_file(self, bucket, key, dest):
            raise errors.pop(0)

    monkeypatch.setitem(sys.modules, "boto3",
                        types.SimpleNamespace(client=lambda name: Client()))
    waits = []
    with pytest.raises(gj.InputMissing, match="does not exist"):
        gj.fetch("s3://bkt/typo.toml", tmp_path / "a", what="c",
                 sleep=waits.append)
    assert waits == []
    errors[:] = [Throttled()] * gj.TRANSFER_ATTEMPTS
    with pytest.raises(gj.TransferFailed, match="SlowDown|Throttled"):
        gj.fetch("s3://bkt/c.toml", tmp_path / "b", what="c",
                 sleep=waits.append)
    assert len(waits) == gj.TRANSFER_ATTEMPTS - 1


def test_a_home_relative_job_resolves_its_references_beside_it(
        tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "jobs").mkdir()
    assert gj._parent_of("~/jobs/R001.json") == tmp_path / "jobs"


# --------------------------------------------------------------------------
# Running one job


class FakeChild:
    """Stands in for ``woof go`` / ``woof icechunk-export``: returns the
    scripted exit codes and leaves the exporter status a real run leaves."""

    def __init__(self, codes, *, export_state="complete", stop_on=None):
        self.codes = list(codes)
        self.calls: list[list[str]] = []
        self.export_state = export_state
        self.stopped_by = None
        self.stop_on = stop_on

    def handle(self, signum, _frame):
        self.stopped_by = signum

    def pause(self, seconds):
        pass

    def run(self, argv, *, cwd):
        self.calls.append(list(argv))
        code = self.codes.pop(0)
        if argv[3] == "go":
            outdir = Path(argv[argv.index("--outdir") + 1])
            export = outdir / "run-20260820-000000Z" / "icechunk-export"
            export.mkdir(parents=True, exist_ok=True)
            self._status(export, "incomplete" if code == 73 else
                         self.export_state)
        elif argv[3] == "icechunk-export" and code == 0:
            self._status(Path(argv[4]), "complete")
        if self.stop_on == len(self.calls):
            self.stopped_by = signal.SIGTERM
        return code

    @staticmethod
    def _status(export: Path, state: str) -> None:
        latest = "2026-08-20T00:58:00Z" if state == "complete" \
            else "2026-08-20T00:40:00Z"
        (export / "status.json").write_text(json.dumps({
            "schema": "woof.icechunk-export.status.v1", "state": state,
            "frames_found": 30, "frames_exported": 30, "frames_refused": 0,
            "last_snapshot": "SNAP1",
            "groups": {"d01": {"latest_exported": "2026-08-20T00:30:00Z"},
                       "d02": {"latest_exported": latest}}}))


def test_a_complete_job_stages_runs_and_reports(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "f.toml").write_text(TOML)
    (source / "f.namelist.wps").write_text("&share /\n")
    job = gj.parse_job(_job(config_toml=None, config="f.toml",
                            files=["f.namelist.wps"],
                            resources={"gpu_mem_gib": 30}), base=source)
    child = FakeChild([0])
    outcome, fields = gj.run_job(job, tmp_path / "w", child=child,
                                 card_gib=lambda: 44.9)
    assert (outcome.exit_code, outcome.status) == (0, "complete")
    assert (tmp_path / "w" / "inputs" / "f.namelist.wps").read_text() == \
        "&share /\n"
    assert child.calls[0][4] == str(tmp_path / "w" / "inputs" / "f.toml")
    assert fields["last_committed_frame"] == "2026-08-20T00:58:00Z"
    assert fields["last_snapshot"] == "SNAP1"
    assert fields["export_state"] == "complete"
    assert fields["go_command"][:2] == ["woof", "go"]
    assert set(fields["timings_s"]) == {"stage", "go"}
    assert [entry["path"] for entry in fields["inputs"]] == [
        "f.toml", "f.namelist.wps"]


@pytest.mark.parametrize("codes, exit_code, status, resumes", [
    ([2], 2, "refused", 0),
    ([1], 1, "failed", 0),
    ([74], 74, "failed", 0),
    ([-9], 137, "failed", 0),
    ([73, 0], 0, "complete", 1),
    ([73, 1, 1, 0], 0, "complete", 3),
    ([73, 1, 1, 1], 73, "export-incomplete", 3),
    ([73, 2], 2, "export-refused", 1),
])
def test_exit_codes_map_to_what_a_scheduler_retries(tmp_path, codes,
                                                    exit_code, status,
                                                    resumes):
    child = FakeChild(codes)
    waits = []
    outcome, fields = gj.run_job(gj.parse_job(_job()), tmp_path, child=child,
                                 sleep=waits.append, card_gib=lambda: None)
    assert (outcome.exit_code, outcome.status) == (exit_code, status)
    assert outcome.go_exit_code == (128 + 9 if codes[0] == -9 else codes[0])
    assert len(child.calls) == 1 + resumes
    assert waits == list(gj.EXPORT_RESUME_WAIT_S[:resumes])
    for call in child.calls[1:]:
        assert call[3:] == ["icechunk-export",
                            str(tmp_path / "go" / "run-20260820-000000Z" /
                                "icechunk-export")]
    if status == "complete" and resumes:
        assert fields["last_committed_frame"] == "2026-08-20T00:58:00Z"
        assert "export_resume" in fields["timings_s"]


def test_a_sigterm_stops_the_job_retryably_without_resuming(tmp_path):
    child = FakeChild([73], stop_on=1)
    outcome, _ = gj.run_job(gj.parse_job(_job()), tmp_path, child=child,
                            card_gib=lambda: None)
    assert (outcome.exit_code, outcome.status) == (75, "stopped")
    assert len(child.calls) == 1
    # A stop that lands as the forecast or the resume finishes cleanly does
    # not turn a complete run into a retry.
    for codes, stop_on in (([0], 1), ([73, 0], 2)):
        child = FakeChild(codes, stop_on=stop_on)
        outcome, _ = gj.run_job(gj.parse_job(_job()), tmp_path / str(stop_on),
                                child=child, sleep=lambda s: None,
                                card_gib=lambda: None)
        assert (outcome.exit_code, outcome.status) == (0, "complete")
    # Stopped before woof go starts: nothing starts.
    real = gj._Child()
    real.stopped_by = signal.SIGTERM
    assert real.run(["false"], cwd=tmp_path) == -signal.SIGTERM
    # ... and a stop cuts a resume wait short instead of outsleeping the
    # scheduler's grace.
    started = __import__("time").monotonic()
    real.pause(300)
    assert __import__("time").monotonic() - started < 1


def test_a_job_larger_than_the_card_is_refused_before_staging(tmp_path):
    job = gj.parse_job(_job(resources={"gpu_mem_gib": 52}))
    with pytest.raises(gj.GoJobRefused, match="H100 tier"):
        gj.run_job(job, tmp_path, child=FakeChild([0]),
                   card_gib=lambda: 44.99)
    assert not (tmp_path / "inputs").exists()


def test_input_that_will_not_come_is_retryable(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise gj.TransferFailed("fetching config failed 4 times")

    monkeypatch.setattr(gj, "fetch", broken)
    job = gj.parse_job(_job(config_toml=None, config="s3://bkt/c.toml"))
    child = FakeChild([0])
    outcome, _ = gj.run_job(job, tmp_path, child=child, card_gib=lambda: None)
    assert (outcome.exit_code, outcome.status) == (75, "input-unavailable")
    assert child.calls == []


# --------------------------------------------------------------------------
# The command


def _main(argv, monkeypatch, child=None):
    if child is not None:
        monkeypatch.setattr(gj, "_Child", lambda: child)
    monkeypatch.setattr(gj, "card_memory_gib", lambda: None)
    args = build_parser().parse_args(["go-job", *argv])
    return args.func(args)


def test_the_command_writes_the_result_and_publishes_a_copy(
        tmp_path, monkeypatch, capsys):
    store = tmp_path / "results" / "R001.result.json"
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(_job(result_url=store.as_uri())))
    monkeypatch.setenv("WOOF_ENERGY_JOB_ATTEMPT", "2")
    code = _main([str(job_path), "--workdir", str(tmp_path / "w")],
                 monkeypatch, FakeChild([0]))
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads((tmp_path / "w" / "result.json").read_text())
    assert printed == written == json.loads(store.read_text())
    assert written["schema"] == gj.RESULT_SCHEMA
    assert (written["status"], written["exit_code"], written["attempt"]) == (
        "complete", 0, 2)
    assert written["icechunk_url"] == "file:///stores/R001.icechunk"
    assert written["last_committed_frame"] == "2026-08-20T00:58:00Z"
    assert written["timings_s"]["total"] >= 0


def test_a_refused_job_exits_2_and_still_writes_its_result(
        tmp_path, monkeypatch, capsys):
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(_job(go_args=["--dry-run"])))
    result = tmp_path / "r" / "result.json"
    code = _main([str(job_path), "--result", str(result)], monkeypatch)
    assert code == 2
    written = json.loads(result.read_text())
    assert written["status"] == "refused" and written["exit_code"] == 2
    assert "--dry-run" in written["error"] and "Why" not in written["error"]
    assert "not a flag a job may pass" in capsys.readouterr().err


def test_an_array_child_runs_its_entry(tmp_path, monkeypatch, capsys):
    listing = tmp_path / "runs.json"
    listing.write_text(json.dumps({"schema": gj.LIST_SCHEMA,
                                   "jobs": [_job("A"), _job("B")]}))
    monkeypatch.setenv("AWS_BATCH_JOB_ARRAY_INDEX", "1")
    monkeypatch.setenv("WOOF_SCRATCH", str(tmp_path / "scratch"))
    child = FakeChild([0])
    assert _main([str(listing)], monkeypatch, child) == 0
    result = json.loads(capsys.readouterr().out)
    assert (result["job_id"], result["index"]) == ("B", 1)
    assert result["workdir"] == str(tmp_path / "scratch" / "go-job" / "B")


def test_the_worker_template_hands_go_job_arguments_it_parses(tmp_path):
    from woof.energy.worker import RunnerRequest, SubprocessRunner

    request = RunnerRequest(
        "R001", tmp_path / "R001.json", 1, tmp_path / "attempt-001",
        tmp_path / "runner.log", tmp_path / "result.json", {}, None,
        threading.Event(), None, "file:///q")
    argv = SubprocessRunner(RUNNER_CMD).argv(request)
    assert argv[:4] == [sys.executable, "-m", "woof", "go-job"]
    args = build_parser().parse_args(argv[3:])
    assert args.func is gj.main
    assert args.job == str(tmp_path / "R001.json")
    assert args.workdir == tmp_path / "attempt-001"
    assert args.result == tmp_path / "result.json"


def _woof(*argv, env=None):
    environment = dict(os.environ, PYTHONPATH=str(REPO),
                       GPUWM_NO_LOCAL_GPU="1", **(env or {}))
    return subprocess.run([sys.executable, "-m", "woof", *argv],
                          capture_output=True, text=True, env=environment,
                          timeout=180)


def test_a_worker_drains_go_jobs_and_a_refusal_is_dead_at_once(tmp_path):
    from woof.energy import workqueue as wq

    url = f"file://{tmp_path / 'q'}"
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    # A config the job cannot have: go-job refuses before woof go starts.
    (jobs / "bad.json").write_text(json.dumps(_job(
        "BAD", config_toml=None, config=f"file://{tmp_path}/absent.toml")))
    assert _woof("energy", "queue", "init", url, "--lease-s", "20"
                 ).returncode == 0
    pushed = _woof("energy", "queue", "push", url, str(jobs),
                   "--schema", gj.JOB_SCHEMA)
    assert pushed.returncode == 0, pushed.stderr
    worker = _woof("energy", "worker", url, "--runner-cmd", RUNNER_CMD,
                   "--no-gpu", "--exit-when-empty", "--poll-s", "0.2",
                   "--workdir", str(tmp_path / "w"),
                   env={"WOOF_GPU_LOCK_ROOT": str(tmp_path / "locks")})
    assert worker.returncode == 0, worker.stderr
    store = wq.open_store(url)
    dead = json.loads(store.get("dead/BAD.json")[0])
    assert dead["failures"] == 1 and dead["reason"].startswith("permanent")
    failed = json.loads(store.get("failed/BAD.json")[0])
    assert [a["exit_code"] for a in failed["attempts"]] == [2]
    log = store.get(failed["attempts"][-1]["log_key"])[0].decode()
    assert "is not a file" in log and '"status": "refused"' in log


# --------------------------------------------------------------------------
# The AWS track-run deployment


def test_the_track_run_example_renders_both_tiers_strictly(tmp_path):
    mapping = deploy.load_config_mapping(EXAMPLE)
    rendered = {}
    for tier in ("l40s", "h100"):
        config = deploy.config_from_mapping(deploy.select_tier(mapping, tier))
        rendered[tier] = deploy.render(config, "aws-batch", strict=True)
    l40s = json.loads(rendered["l40s"].files["compute-environment.json"])
    h100 = json.loads(rendered["h100"].files["compute-environment.json"])
    assert l40s["computeResources"]["instanceTypes"] == ["g6e.xlarge"]
    assert l40s["computeResources"]["type"] == "SPOT"
    assert h100["computeResources"]["instanceTypes"] == ["p5.48xlarge"]
    assert h100["computeResources"]["type"] == "EC2"
    assert h100["computeResources"]["maxvCpus"] == 192
    # The section merge keeps the shared identities.
    assert l40s["computeResources"]["subnets"] == \
        h100["computeResources"]["subnets"]
    for tier, files in rendered.items():
        jobdef = json.loads(files.files["job-definition.json"])
        command = jobdef["containerProperties"]["command"]
        assert command[command.index("--runner-cmd") + 1] == RUNNER_CMD
        assert command[command.index("--queue") + 1] == (
            f"s3://YOUR-BUCKET/woof-queue/track-runs/{tier}/")
        env = {e["name"]: e["value"]
               for e in jobdef["containerProperties"]["environment"]}
        assert env["AWS_REGION"] == "us-west-2"
        assert env["WOOF_ENERGY_OUTPUT_URL"] == \
            "s3://YOUR-BUCKET/woof-outputs/track-runs"
        assert env["WOOF_GEOG_URL"].startswith("s3://YOUR-BUCKET/")
        gpu = [r for r in jobdef["containerProperties"]["resourceRequirements"]
               if r["type"] == "GPU"]
        assert gpu == [{"type": "GPU", "value": "1"}]
        assert {"onExitCode": "2", "action": "EXIT"} in \
            jobdef["retryStrategy"]["evaluateOnExit"]


def test_tiers_merge_sections_and_must_be_chosen(tmp_path, capsys):
    base = {"name": "x", "aws_batch": {"region": "us-west-2",
                                       "instance_types": ["g6e.xlarge"]},
            "tiers": {"big": {"name": "x-big",
                              "aws_batch": {"instance_types": ["p5.48xlarge"]}}}}
    merged = deploy.select_tier(base, "big")
    assert merged["name"] == "x-big" and "tiers" not in merged
    assert merged["aws_batch"] == {"region": "us-west-2",
                                   "instance_types": ["p5.48xlarge"]}
    assert "tiers" in base  # the caller's mapping is not changed
    with pytest.raises(deploy.DeployRefused, match="--tier NAME"):
        deploy.select_tier(base, None)
    with pytest.raises(deploy.DeployRefused, match="not one of big"):
        deploy.select_tier(base, "small")
    with pytest.raises(deploy.DeployRefused, match="no \\[tiers.NAME\\]"):
        deploy.select_tier({"name": "x"}, "big")
    args = build_parser().parse_args([
        "energy", "deploy", "render", "--backend", "aws-batch", "--config",
        str(EXAMPLE), "--tier", "h100", "-o", str(tmp_path / "o")])
    assert args.func(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["workers"] == 2 and summary["placeholders"] == []
