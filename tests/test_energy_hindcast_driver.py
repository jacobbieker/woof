"""``woof energy hindcast``: segment jobs, the manifest, materialize, run-local.

CPU only.  Most tests inject the source selector, the cluster planner and
the runner; two materialize through the real wrf-nests and wrf-tiles
planners (a few seconds each) and dry-run the result with ``woof energy
run``'s own ``run_plan``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import tomllib
import types

import numpy as np
import pytest

from woof.energy import hindcast as hc
from woof.energy.contracts import (Plan, PlanDomain, dump_plan, load_plan,
                                   load_sites, sha256_file)
from woof.energy.run import run_plan

REPO = Path(__file__).resolve().parents[1]
SITES = REPO / "tests" / "fixtures" / "energy" / "sites_wales.json"
NOW = datetime(2026, 10, 14, 12)
RING = ((-4.0, 51.6), (-3.9, 51.6), (-3.9, 51.7), (-4.0, 51.7), (-4.0, 51.6))


def _request(tmp_path, **overrides):
    record = dict(outdir=tmp_path / "hc", start=datetime(2025, 11, 1),
                  end=datetime(2025, 11, 4), sites_path=SITES,
                  source="gfs", vram_gib=80.0)
    record.update(overrides)
    return hc.HindcastRequest(**record)


# --------------------------------------------------------------------------
# segments and time


def test_three_daily_segments_keep_contiguous_windows():
    segments = hc.segment_windows(datetime(2025, 11, 1), datetime(2025, 11, 4),
                                  segment_hours=30, spinup_hours=6,
                                  cycle_hour=0)
    assert [s.run_start for s in segments] == [
        datetime(2025, 11, d) for d in (1, 2, 3)]
    for segment in segments:
        assert segment.run_end - segment.run_start == hc.timedelta(hours=30)
        assert segment.keep_start - segment.run_start == hc.timedelta(hours=6)
        assert segment.keep_end == segment.run_end
    for a, b in zip(segments, segments[1:]):
        assert a.keep_end == b.keep_start


def test_a_leap_year_is_366_segments_and_a_plain_year_365():
    leap = hc.segment_windows(datetime(2024, 1, 1), datetime(2025, 1, 1),
                              segment_hours=30, spinup_hours=6, cycle_hour=0)
    assert len(leap) == 366
    assert datetime(2024, 2, 29) in {s.run_start for s in leap}
    assert leap[-1].keep_end == datetime(2025, 1, 1, 6)
    plain = hc.segment_windows(datetime(2025, 1, 1), datetime(2026, 1, 1),
                               segment_hours=30, spinup_hours=6, cycle_hour=0)
    assert len(plain) == 365
    starts = {s.run_start for s in plain}
    assert datetime(2025, 3, 1) in starts and datetime(2025, 2, 28) in starts
    # Every month boundary is crossed with no gap.
    for a, b in zip(plain, plain[1:]):
        assert a.keep_end == b.keep_start


def test_cycle_hour_18_keeps_whole_utc_days_and_shorter_strides_work():
    segments = hc.segment_windows(datetime(2025, 3, 29), datetime(2025, 3, 31),
                                  segment_hours=30, spinup_hours=6,
                                  cycle_hour=18)
    assert segments[0].keep_start == datetime(2025, 3, 30, 0)
    assert segments[0].keep_end == datetime(2025, 3, 31, 0)
    # 18 h runs keeping 12 h: two per day.
    half = hc.segment_windows(datetime(2025, 1, 1), datetime(2025, 1, 2),
                              segment_hours=18, spinup_hours=6, cycle_hour=0)
    assert [s.run_start.hour for s in half] == [0, 12]


def test_parse_utc_accepts_utc_spellings_and_refuses_offsets():
    assert hc.parse_utc("2025-11-01", "--start") == (datetime(2025, 11, 1),
                                                    False)
    assert hc.parse_utc("2025-11-01T06Z", "--start")[0] == datetime(
        2025, 11, 1, 6)
    assert hc.parse_utc("2025-11-01T06:00:00+00:00", "x")[0] == datetime(
        2025, 11, 1, 6)
    with pytest.raises(hc.HindcastRefused, match="UTC offset"):
        hc.parse_utc("2025-03-30T06:00:00+01:00", "--start")
    with pytest.raises(hc.HindcastRefused, match="whole hour"):
        hc.parse_utc("2025-03-30T06:30", "--start")
    with pytest.raises(hc.HindcastRefused, match="YYYY-MM-DD"):
        hc.parse_utc("30/03/2025", "--start")


@pytest.mark.parametrize("overrides,match", [
    (dict(leaf_history_interval_s=700.0), "does not divide"),
    (dict(spinup_hours=30), "less than --segment-hours"),
    (dict(start=datetime(2025, 11, 1, 6), start_has_hour=True), "cycle-hour"),
    (dict(topology="wrf-tiles", parent_history_interval_s=1800.0),
     "boundary forcing"),
    (dict(gridded_template="{root}/{bucket}/{leaf}"), "bucket"),
    (dict(gridded_template="{root}/{cluster_id}.icechunk"), "{leaf}"),
    (dict(end=datetime(2025, 11, 1)), "after --start"),
    (dict(topology="hex-swath"), "hex-swath"),
    (dict(sites_path=None), "--sites"),
])
def test_requests_that_cannot_be_honoured_are_refused(tmp_path, overrides,
                                                      match):
    with pytest.raises(hc.HindcastRefused, match=match):
        _request(tmp_path, **overrides)


# --------------------------------------------------------------------------
# plan


def test_dry_run_plans_three_jobs_and_writes_nothing(tmp_path):
    result = hc.plan_hindcast(_request(tmp_path), dry_run=True, now=NOW)
    assert result["dry_run"] is True
    jobs = result["jobs"]
    assert len(jobs) == 3
    assert not (tmp_path / "hc").exists()
    job = jobs[0]
    assert job["schema"] == hc.JOB_SCHEMA
    assert job["segment"]["run_hours"] == 30.0
    assert job["segment"]["keep_hours"] == 24.0
    assert job["segment"]["history_begin_s"] == 21600.0
    assert job["history"]["leaf_interval_s"] == 900.0
    assert job["history"]["parent_interval_s"] == 3600.0
    assert job["planner"]["kwargs"]["start"] == "2025-11-01T00"
    assert job["planner"]["kwargs"]["hours"] == 30.0
    assert job["planner"]["kwargs"]["vram_gib"] == 80.0
    assert job["source"]["name"] == "gfs"
    assert job["source"]["selector"] == "fixed"
    assert job["outputs"]["time_window"] == ["2025-11-01T06:00:00Z",
                                             "2025-11-02T06:00:00Z"]
    assert "{leaf}" in job["outputs"]["gridded_template"]
    assert job["job_id"].startswith("sites_wales-20251101T00Z-")
    assert result["manifest"]["counts"] == {"pending": 3}


def test_job_ids_are_deterministic_and_bind_their_inputs(tmp_path):
    first = hc.plan_hindcast(_request(tmp_path), dry_run=True, now=NOW)
    again = hc.plan_hindcast(_request(tmp_path), dry_run=True, now=NOW)
    assert ([j["job_id"] for j in first["jobs"]]
            == [j["job_id"] for j in again["jobs"]])
    assert ([hc._json_bytes(j) for j in first["jobs"]]
            == [hc._json_bytes(j) for j in again["jobs"]])
    other = hc.plan_hindcast(_request(tmp_path, source="ecmwf-open-data"),
                             dry_run=True, now=NOW)
    finer = hc.plan_hindcast(_request(tmp_path, leaf_history_interval_s=300.0),
                             dry_run=True, now=NOW)
    ids = {j["job_id"] for j in first["jobs"]}
    assert ids.isdisjoint(j["job_id"] for j in other["jobs"])
    assert ids.isdisjoint(j["job_id"] for j in finer["jobs"])


def test_plan_writes_jobs_and_replanning_is_idempotent(tmp_path):
    request = _request(tmp_path)
    manifest = hc.plan_hindcast(request, now=NOW)
    outdir = tmp_path / "hc"
    assert (outdir / hc.MANIFEST_NAME).is_file()
    assert not list(outdir.rglob("*.partial"))
    job_ids = manifest["planned_job_ids"]
    for job_id in job_ids:
        job = hc.load_job(outdir / "jobs" / job_id / "job.json")
        sites = outdir / "jobs" / job_id / job["cluster"]["sites"]["path"]
        assert sha256_file(sites) == job["cluster"]["sites"]["sha256"]
    hc.update_job(outdir, job_ids[0], status="complete")
    again = hc.plan_hindcast(request, now=NOW)
    assert again["jobs"][job_ids[0]]["status"] == "complete"
    assert again["counts"] == {"complete": 1, "pending": 2}


def test_a_different_request_needs_resume_and_then_merges(tmp_path):
    outdir = tmp_path / "hc"
    first = hc.plan_hindcast(_request(tmp_path), now=NOW)
    done = first["planned_job_ids"][0]
    hc.update_job(outdir, done, status="complete")
    longer = _request(tmp_path, end=datetime(2025, 11, 6))
    with pytest.raises(hc.HindcastRefused, match="--resume"):
        hc.plan_hindcast(longer, now=NOW)
    merged = hc.plan_hindcast(longer, resume=True, now=NOW)
    assert len(merged["jobs"]) == 5
    assert merged["jobs"][done]["status"] == "complete"
    # A changed source for the same cycles supersedes pending jobs but
    # never a complete one.
    switched = _request(tmp_path, end=datetime(2025, 11, 6),
                        source="ecmwf-open-data")
    merged = hc.plan_hindcast(switched, resume=True, now=NOW)
    assert len(merged["jobs"]) == 5
    assert done in merged["jobs"]
    assert any("superseded" in note for note in merged["notes"])
    assert any("is complete; kept" in note for note in merged["notes"])
    # superseded job directories stay on disk
    assert len(list((outdir / "jobs").iterdir())) == 9


def test_selector_choices_and_changes_are_recorded_per_job(tmp_path):
    calls = []

    def selector(region, cycle, hours, p_top_pa=None, priority=None, *,
                 now=None):
        calls.append((region, cycle, hours, p_top_pa, priority, now))
        name = "met-office-global" if cycle.day % 2 else "icon-global-ml"
        return types.SimpleNamespace(source=name,
                                     reasons=[f"day {cycle.day}"])

    request = _request(tmp_path, source=None,
                       source_priority=("met-office-global", "gfs"))
    result = hc.plan_hindcast(request, source_selector=selector, now=NOW)
    assert len(calls) == 3
    region, cycle, hours, p_top, priority, now = calls[0]
    assert set(region) == {"west", "south", "east", "north"}
    assert cycle == datetime(2025, 11, 1) and hours == 30.0
    assert p_top == 5500.0 and priority == ("met-office-global", "gfs")
    assert now == NOW
    jobs = [hc.load_job(tmp_path / "hc" / "jobs" / j / "job.json")
            for j in result["planned_job_ids"]]
    assert [j["source"]["name"] for j in jobs] == [
        "met-office-global", "icon-global-ml", "met-office-global"]
    assert [j["source"]["changed_from_previous_segment"] for j in jobs] == [
        False, True, True]
    assert jobs[1]["source"]["previous_segment_source"] == "met-office-global"
    assert jobs[1]["source"]["reasons"] == ["day 2"]
    status = hc.hindcast_status(tmp_path / "hc")
    cluster = status["clusters"]["sites_wales"]
    assert cluster["source_changes"] == 2 and cluster["gaps"] == []
    assert sum(1 for n in result["notes"] if "source changes" in n) == 2


def test_a_selector_refusal_names_the_segment(tmp_path):
    def selector(region, cycle, hours, p_top_pa=None, priority=None, *,
                 now=None):
        if cycle.day == 2:
            raise ValueError("nothing serves this cycle")
        return "gfs"

    with pytest.raises(hc.HindcastRefused,
                       match="2025-11-02T00:00:00Z.*nothing serves"):
        hc.plan_hindcast(_request(tmp_path, source=None),
                         source_selector=selector, dry_run=True, now=NOW)


def test_fallback_selector_prefers_met_office_inside_its_window():
    recent = hc.fallback_select(None, datetime(2025, 11, 1), 30.0, now=NOW)
    assert recent.source == "met-office-global"
    old = hc.fallback_select(None, datetime(2023, 1, 1), 30.0, now=NOW)
    assert old.source == "gfs"
    assert any("met-office-global skipped" in r for r in old.reasons)


def test_partition_clusters_become_job_chains(tmp_path):
    sites = load_sites(SITES)
    ids = [s.site_id for s in sites.sites]
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({
        "schema": hc.PARTITION_SCHEMA,
        "sites": {"path": str(SITES), "sha256": sha256_file(SITES)},
        "clusters": [
            {"cluster_id": "west", "site_ids": ids[:10], "leaf_dx_m": 200.0,
             "chain": [5, 3], "region": {"name": "west"}},
            {"cluster_id": "east", "site_ids": ids[10:], "leaf_dx_m": 100.0,
             "topology": "wrf-tiles"},
        ]}))
    request = _request(tmp_path, sites_path=None, partition_path=partition)
    result = hc.plan_hindcast(request, dry_run=True, now=NOW)
    jobs = result["jobs"]
    assert len(jobs) == 6
    west = [j for j in jobs if j["cluster"]["cluster_id"] == "west"]
    assert west[0]["cluster"]["site_count"] == 10
    assert west[0]["planner"]["kwargs"]["dx_m"] == 200.0
    assert west[0]["planner"]["kwargs"]["parent_dx_m"] == 3000.0
    assert west[0]["cluster"]["region"] == {"name": "west"}
    east = [j for j in jobs if j["cluster"]["cluster_id"] == "east"]
    assert east[0]["planner"]["topology"] == "wrf-tiles"
    assert east[0]["history"]["parent_interval_s"] is None
    only = hc.plan_hindcast(_request(tmp_path, sites_path=None,
                                     partition_path=partition,
                                     clusters=("east",)),
                            dry_run=True, now=NOW)
    assert {j["cluster"]["cluster_id"] for j in only["jobs"]} == {"east"}


@pytest.mark.parametrize("document,match", [
    ({"schema": "woof-energy.sites.v1"}, "partition.v1"),
    ({"schema": hc.PARTITION_SCHEMA, "clusters": []}, "no clusters"),
    ({"schema": hc.PARTITION_SCHEMA,
      "clusters": [{"cluster_id": "a", "site_ids": ["nope"]}]},
     "not in the sites document"),
    ({"schema": hc.PARTITION_SCHEMA,
      "clusters": [{"cluster_id": "a", "site_ids": ["x"],
                    "topology": "hex-swath"}]}, "hex-swath"),
])
def test_bad_partitions_are_refused(tmp_path, document, match):
    document = dict(document, sites={"path": str(SITES)})
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps(document))
    with pytest.raises(hc.HindcastRefused, match=match):
        hc.plan_hindcast(_request(tmp_path, sites_path=None,
                                  partition_path=partition),
                         dry_run=True, now=NOW)


# --------------------------------------------------------------------------
# materialize


def test_patch_domain_tables_replaces_and_inserts():
    text = ("[experiment]\nname = 'x'\n[[domain]]\ngrid_id = 1\n"
            "history_interval_s = 3600.0\n[[domain]]\ngrid_id = 2\n"
            "nx = 10\n[shared]\nhistory_interval_s = 5\n")
    out = hc.patch_domain_tables(text, {
        1: {"history_interval_s": 1800.0, "history_begin_s": 21600.0},
        2: {"history_begin_s": 21600.0}})
    raw = tomllib.loads(out)
    assert raw["domain"][0] == {"grid_id": 1, "history_interval_s": 1800.0,
                                "history_begin_s": 21600.0}
    assert raw["domain"][1]["history_begin_s"] == 21600.0
    assert raw["shared"]["history_interval_s"] == 5
    with pytest.raises(hc.HindcastRefused, match=r"\[3\]"):
        hc.patch_domain_tables(text, {3: {"history_begin_s": 1.0}})
    # A commented header and a multi-line array inside the table.
    tricky = ("[[domain]]  # d01\ngrid_id = 1\nladder = [\n    [1.0, 2.0],\n"
              "]\nhistory_interval_s = 3600.0\n")
    raw = tomllib.loads(hc.patch_domain_tables(
        tricky, {1: {"history_begin_s": 600.0}}))
    assert raw["domain"][0]["history_begin_s"] == 600.0
    assert raw["domain"][0]["ladder"] == [[1.0, 2.0]]


def test_job_ids_do_not_depend_on_how_the_sites_path_is_spelled(
        tmp_path, monkeypatch):
    absolute = hc.plan_hindcast(_request(tmp_path), dry_run=True, now=NOW)
    monkeypatch.chdir(SITES.parent)
    relative = hc.plan_hindcast(_request(tmp_path,
                                         sites_path=Path("./" + SITES.name)),
                                dry_run=True, now=NOW)
    assert ([j["job_id"] for j in absolute["jobs"]]
            == [j["job_id"] for j in relative["jobs"]])
    hour = hc.plan_hindcast(_request(tmp_path, start_has_hour=True),
                            dry_run=True, now=NOW)
    assert (hour["manifest"]["request_sha256"]
            == absolute["manifest"]["request_sha256"])


def test_running_jobs_are_never_superseded(tmp_path):
    first = hc.plan_hindcast(_request(tmp_path), now=NOW)
    busy = first["planned_job_ids"][1]
    hc.update_job(tmp_path / "hc", busy, status="running")
    merged = hc.plan_hindcast(_request(tmp_path, source="ecmwf-open-data"),
                              resume=True, now=NOW)
    assert merged["jobs"][busy]["status"] == "running"
    assert any(f"{busy} is running; kept" in n for n in merged["notes"])


def _first_job(tmp_path, **overrides):
    manifest = hc.plan_hindcast(_request(tmp_path, **overrides), now=NOW)
    job_id = manifest["planned_job_ids"][1]
    return tmp_path / "hc" / "jobs" / job_id / "job.json"


def test_materialize_nests_sets_history_begin_on_every_domain(tmp_path):
    job_path = _first_job(tmp_path, source="met-office-global",
                          leaf_history_interval_s=300.0)
    receipt = hc.materialize_job(job_path, now=NOW)
    plan_path = job_path.parent / receipt["plan"]
    plan = load_plan(plan_path)
    assert plan.topology == "wrf-nests"
    assert plan.start.startswith("2025-11-02T00") and plan.hours == 30.0
    config = tomllib.loads(plan.resolve(plan.domains[0].config).read_text())
    assert config["experiment"]["run_seconds"] == 30 * 3600.0
    assert config["shared"]["p_top"] == 5500.0
    leaves = {d.grid_id for d in plan.domains if d.role == "child"}
    for table in config["domain"]:
        assert table["history_begin_s"] == 21600.0
        want = 300.0 if table["grid_id"] in leaves else 3600.0
        assert table["history_interval_s"] == want
    assert not list(plan_path.parent.glob("*hindcast-partial*"))
    for domain in plan.domains:
        assert domain.extra["hindcast"]["keep_window"] == [
            "2025-11-02T06:00:00Z", "2025-11-03T06:00:00Z"]
    assert set(receipt["targets"]["gridded"]) == {
        d.domain_id for d in plan.domains if d.role == "child"}
    assert receipt["cost"]["basis"] == "plan"
    manifest = hc.read_manifest(tmp_path / "hc")
    job_id = receipt["job_id"]
    assert manifest["jobs"][job_id]["status"] == "materialized"
    # woof energy run takes the plan as it stands
    dry = run_plan(plan_path, dry_run=True)
    assert dry["steps"][0]["argv"][0][0] == "go"
    # idempotent, then forced aside
    again = hc.materialize_job(job_path, now=NOW)
    assert again.get("reused") is True
    assert again["plan_sha256"] == receipt["plan_sha256"]
    hc.materialize_job(job_path, now=NOW, force=True)
    assert list(job_path.parent.glob("plan.previous-*"))


def test_materialize_tiles_rebinds_the_leaf_child_config(tmp_path):
    job_path = _first_job(tmp_path, topology="wrf-tiles",
                          source="met-office-global",
                          leaf_history_interval_s=300.0)
    receipt = hc.materialize_job(job_path, now=NOW)
    plan_path = job_path.parent / receipt["plan"]
    plan = load_plan(plan_path)
    parent = plan.domain("parent")
    parent_cfg = tomllib.loads(plan.resolve(parent.config).read_text())
    assert "history_begin_s" not in parent_cfg["domain"][0]
    leaf = next(d for d in plan.domains if d.role == "child")
    toml_path = plan.resolve(leaf.extra["child_config"])
    child = tomllib.loads(toml_path.read_text())
    assert child["run"]["output_interval_s"] == 300.0
    digest = sha256_file(toml_path)
    args = leaf.extra["downscale_args"]
    assert args[args.index("--child-config-sha256") + 1] == digest
    assert leaf.extra["child_config_sha256"] == digest
    assert plan.sites_ref["sha256"] == sha256_file(
        plan.directory / plan.sites_ref["path"])
    dry = run_plan(plan_path, dry_run=True)
    downscale = next(s for s in dry["steps"] if s["kind"] == "downscale")
    assert digest in downscale["argv"][0]


def _fake_planner(job, plan_dir, *, job_dir, now=None):
    """A one-config wrf-nests plan the history patch can read."""

    config = plan_dir / "seg.toml"
    config.write_text(
        "[[domain]]\ngrid_id = 1\nhistory_interval_s = 3600.0\n"
        "[[domain]]\ngrid_id = 2\nhistory_interval_s = 900.0\n")
    domains = [
        PlanDomain(domain_id="d01", topology="wrf-nests", role="parent",
                   dx_m=500.0, run_dir="runs/seg",
                   output_glob="wrfout_d01_*", footprint=RING,
                   config="seg.toml", grid_id=1, extra={"nx": 4, "ny": 4}),
        PlanDomain(domain_id="d02", topology="wrf-nests", role="child",
                   dx_m=100.0, run_dir="runs/seg",
                   output_glob="wrfout_d02_*", footprint=RING,
                   config="seg.toml", grid_id=2, site_ids=("s1",),
                   extra={"nx": 6, "ny": 6})]
    plan = Plan(topology="wrf-nests", dx_m=100.0,
                start=job["planner"]["kwargs"]["start"],
                hours=job["planner"]["kwargs"]["hours"], domains=domains)
    return dump_plan(plan, plan_dir / "plan.json")


@pytest.fixture
def no_loader_check(monkeypatch):
    """The fake config is not a full experiment; accept it as written."""

    class _Domain:
        def __init__(self, table):
            self.grid_id = table["grid_id"]
            self.history_interval_s = table["history_interval_s"]
            self.history_begin_s = table.get("history_begin_s", 0.0)

    def load(path):
        raw = tomllib.loads(Path(path).read_text())
        return types.SimpleNamespace(
            domains=[_Domain(t) for t in raw["domain"]], run_seconds=108000.0)

    import woof.experiment

    monkeypatch.setattr(woof.experiment, "load_experiment", load)


def test_run_local_runs_then_post_steps_and_records_status(tmp_path,
                                                           no_loader_check):
    job_path = _first_job(tmp_path)
    seen = {}

    def runner(plan_path, dry_run=False, resume=False):
        seen["runner"] = (Path(plan_path), dry_run, resume)
        return {"domains": {"d01": {"status": "complete"},
                            "d02": {"status": "complete"}},
                "manifest_path": "runs/manifest.json"}

    def writer(**kwargs):
        seen["writer"] = kwargs
        return {"written": True}

    record = hc.run_local(job_path, runner=runner, writer=writer,
                          extract=False, resume=True,
                          cluster_planner=_fake_planner, now=NOW)
    assert record["ok"] is True
    assert seen["runner"][1:] == (False, True)
    assert seen["writer"]["forecast"] is None
    assert seen["writer"]["targets"]["time_window"] == [
        "2025-11-02T06:00:00Z", "2025-11-03T06:00:00Z"]
    manifest = hc.read_manifest(tmp_path / "hc")
    job = manifest["jobs"][record["job_id"]]
    assert job["status"] == "complete" and job["attempts"] == 1
    assert job["post_step"]["writer"] == {"written": True}
    skipped = hc.run_local(job_path, runner=runner, extract=False,
                           cluster_planner=_fake_planner)
    assert skipped["skipped"] is True
    # A complete job's plan is never silently replanned.
    (job_path.parent / hc.MATERIALIZED_NAME).unlink()
    with pytest.raises(hc.HindcastRefused, match="is complete"):
        hc.materialize_job(job_path, cluster_planner=_fake_planner)
    assert hc.read_manifest(tmp_path / "hc")["jobs"][record["job_id"]][
        "status"] == "complete"


def test_run_local_marks_failed_domains_and_dry_runs_cleanly(
        tmp_path, no_loader_check):
    job_path = _first_job(tmp_path)

    def failing(plan_path, dry_run=False, resume=False):
        if dry_run:
            return {"dry_run": True, "steps": []}
        return {"domains": {"d01": {"status": "failed",
                                    "reason": "exit status 1"},
                            "d02": {"status": "skipped",
                                    "reason": "parent d01 failed"}}}

    dry = hc.run_local(job_path, runner=failing, dry_run=True,
                       cluster_planner=_fake_planner)
    assert dry["dry_run"] is True
    manifest = hc.read_manifest(tmp_path / "hc")
    assert manifest["jobs"][dry["job_id"]]["status"] == "materialized"
    record = hc.run_local(job_path, runner=failing,
                          cluster_planner=_fake_planner)
    assert record["ok"] is False and "exit status 1" in record["reason"]
    manifest = hc.read_manifest(tmp_path / "hc")
    assert manifest["jobs"][record["job_id"]]["status"] == "failed"
    status = hc.hindcast_status(tmp_path / "hc", status="failed")
    assert list(status["selected"]) == [record["job_id"]]


def test_a_running_job_is_not_started_twice(tmp_path, no_loader_check):
    job_path = _first_job(tmp_path)
    job_id = job_path.parent.name
    hc.update_job(tmp_path / "hc", job_id, status="running",
                  start_utc="2026-10-14T00:00:00Z")

    def runner(plan_path, dry_run=False, resume=False):
        return {"domains": {"d01": {"status": "complete"}}}

    with pytest.raises(hc.HindcastRefused, match="is running"):
        hc.run_local(job_path, runner=runner, extract=False,
                     cluster_planner=_fake_planner)
    record = hc.run_local(job_path, runner=runner, extract=False,
                          force=True, cluster_planner=_fake_planner)
    assert record["ok"] is True


def test_partition_plan_cluster_is_used_when_present(tmp_path, monkeypatch,
                                                     no_loader_check):
    sites = load_sites(SITES)
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({
        "schema": hc.PARTITION_SCHEMA, "sites": {"path": str(SITES)},
        "clusters": [{"cluster_id": "all",
                      "site_ids": [s.site_id for s in sites.sites]}]}))
    calls = {}

    def plan_cluster(partition_doc, cluster_id, *, outdir, start, hours,
                     source, vram_gib=None):
        calls.update(partition=partition_doc, cluster_id=cluster_id,
                     start=start, hours=hours, source=source,
                     vram_gib=vram_gib)
        job = {"planner": {"kwargs": {"start": start, "hours": hours}}}
        return _fake_planner(job, Path(outdir), job_dir=None)

    module = types.ModuleType("woof.energy.partition")
    module.plan_cluster = plan_cluster
    monkeypatch.setitem(sys.modules, "woof.energy.partition", module)
    manifest = hc.plan_hindcast(
        _request(tmp_path, sites_path=None, partition_path=partition),
        now=NOW)
    job_path = (tmp_path / "hc" / "jobs" / manifest["planned_job_ids"][0]
                / "job.json")
    receipt = hc.materialize_job(job_path, now=NOW)
    assert calls["cluster_id"] == "all"
    assert calls["partition"]["schema"] == hc.PARTITION_SCHEMA
    assert (calls["start"], calls["hours"], calls["vram_gib"]) == (
        "2025-11-01T00", 30.0, 80.0)
    assert receipt["domains"][0]["history_begin_s"] == 21600.0


def test_the_post_step_keeps_only_the_kept_window():
    from woof.energy.extract import ForecastData

    times = np.arange(np.datetime64("2025-11-01T00:00:00"),
                      np.datetime64("2025-11-02T06:15:00"),
                      np.timedelta64(15, "m")).astype("datetime64[s]")
    data = ForecastData(
        times=times, heights_m=np.array([100.0]),
        coords={"site_id": np.array(["s1"], dtype=object)},
        variables={"wind_speed": np.ones((times.size, 1, 1))},
        variable_notes={}, notes=[], attrs={})
    kept = hc._slice_window(data, datetime(2025, 11, 1, 6),
                            datetime(2025, 11, 2, 6))
    assert kept.times.size == 96
    assert str(kept.times[0]) == "2025-11-01T06:00:00"
    assert str(kept.times[-1]) == "2025-11-02T05:45:00"
    assert kept.variables["wind_speed"].shape == (96, 1, 1)
    with pytest.raises(hc.HindcastRefused, match="no site frame"):
        hc._slice_window(data, datetime(2025, 12, 1), datetime(2025, 12, 2))


# --------------------------------------------------------------------------
# CLI


def _parse(argv):
    from woof.energy.cli import register_cli

    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def test_the_cli_plans_shows_status_and_refuses_cleanly(tmp_path, capsys):
    outdir = tmp_path / "hc"
    args = _parse(["energy", "hindcast", "plan", "--sites", str(SITES),
                   "--start", "2025-11-01", "--end", "2025-11-04",
                   "--source", "gfs", "--vram-gib", "80", "--dry-run",
                   "-o", str(outdir)])
    assert args.func(args) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["jobs"] == 3 and shown["example_job"]["schema"] == \
        hc.JOB_SCHEMA
    args = _parse(["energy", "hindcast", "plan", "--sites", str(SITES),
                   "--start", "2025-11-01", "--end", "2025-11-04",
                   "--source", "gfs", "-o", str(outdir)])
    assert args.func(args) == 0
    capsys.readouterr()
    args = _parse(["energy", "hindcast", "status", str(outdir)])
    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out)["counts"] == {"pending": 3}
    args = _parse(["energy", "hindcast", "plan", "--sites", str(SITES),
                   "--start", "2025-11-01T06", "--end", "2025-11-04",
                   "--source", "gfs", "-o", str(outdir)])
    assert args.func(args) == 2
    assert "cycle-hour" in capsys.readouterr().err
    args = _parse(["energy", "hindcast", "materialize", "no-such-job",
                   "--dir", str(outdir)])
    assert args.func(args) == 2
