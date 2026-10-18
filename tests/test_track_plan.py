"""``woof track plan``: synthetic paths in, loadable track-following trees out.

Every track here is synthetic, at neutral coordinates, and every check is
network-free: the plan reads the source registry and the wizard's own
validators, and the ``woof go --dry-run`` checks pin the decoder bridge.
"""

from __future__ import annotations

import csv
import sys
import json
import math
import tomllib
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from woof.track import plan as tp
from woof.track.model import track_from_fixes


T0 = datetime(2025, 6, 1, 14, 0, 0, tzinfo=timezone.utc)


def _fixes(*, lat0=40.0, lon0=-100.0, speed=10.0, heading=90.0,
           peak=20000.0, climb=5.0, float_s=3600.0, ground=500.0, step=30.0,
           turn_deg_s=0.0):
    """A ground run, a climb at ``climb`` m/s to ``peak``, a float, a
    descent at twice the climb rate and a ground run, drifting at
    ``speed`` m/s on ``heading`` (turning ``turn_deg_s`` per second)."""

    times, lats, lons, heights = [], [], [], []
    t, h, lat, lon = 0.0, ground, lat0, lon0

    def add():
        times.append(T0 + timedelta(seconds=t))
        lats.append(lat)
        lons.append(lon)
        heights.append(h)

    for _ in range(5):
        add()
        t += step
    phase, floated = "up", 0.0
    while phase != "done":
        if phase == "up":
            h = min(peak, h + climb * step)
            phase = "float" if h >= peak else "up"
        elif phase == "float":
            floated += step
            phase = "down" if floated >= float_s else "float"
        else:
            h = max(ground, h - 2 * climb * step)
            phase = "done" if h <= ground else "down"
        distance = speed * step
        lat += distance * math.cos(math.radians(heading)) / 111195.0
        lon += distance * math.sin(math.radians(heading)) / (
            111195.0 * math.cos(math.radians(lat)))
        heading += turn_deg_s * step
        add()
        t += step
    for _ in range(5):
        add()
        t += step
    return times, lats, lons, heights


def _track(name="synthetic", **kwargs):
    times, lats, lons, heights = _fixes(**kwargs)
    return track_from_fixes(name, times, lats, lons, heights)


def _write_csv(path: Path, **kwargs) -> Path:
    times, lats, lons, heights = _fixes(**kwargs)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["when", "y", "x", "alt", "fix"])
        for moment, lat, lon, height in zip(times, lats, lons, heights):
            writer.writerow([moment.strftime("%Y-%m-%d %H:%M:%S"),
                             f"{lat:.6f}", f"{lon:.6f}", f"{height:.1f}", 1])
        # A row a receiver flagged as no-fix, which --drop-when removes.
        writer.writerow([times[-1].strftime("%Y-%m-%d %H:%M:%S"), "0", "0",
                         "0", 0])
    return path


#: This woof's own itinerary lookup, before the fixture below hides it.
_REAL_RELOCATION_MODULE = tp._relocation_module


@pytest.fixture(autouse=True)
def _no_itinerary(monkeypatch):
    """Plan against the woof this test runs in, minus any itinerary module
    a test does not install itself."""

    monkeypatch.setattr(tp, "_relocation_module", lambda: None)


def _config(run) -> dict:
    return tomllib.loads(run.config.read_text(encoding="utf-8"))


def _domains(document) -> dict:
    return {int(table["grid_id"]): table for table in document["domain"]}


# ---------------------------------------------------------------------------
# A straight, fast track: the whole tree


def test_straight_track_plans_a_loadable_three_domain_tree(tmp_path):
    from woof.domain_wizard import experiment_from_text

    track = _track(speed=25.0)
    run = tp.plan_track(track, "straight-run", tmp_path)
    document = _config(run)
    domains = _domains(document)
    assert sorted(domains) == [1, 2, 3]
    # Ladder 1.5 km -> 300 m -> 100 m: d02 72 km, d03 32.4 km, each a
    # whole, even number of its parent's cells.
    assert domains[1]["dx"] == 1500.0
    assert (domains[2]["nx"], domains[3]["nx"]) == (240, 324)
    assert domains[2]["parent_grid_ratio"] == 5
    assert domains[3]["parent_grid_ratio"] == 3
    assert domains[3]["nx"] % 6 == 0 and domains[2]["nx"] % 10 == 0
    exp = experiment_from_text(run.config.read_text(encoding="utf-8"),
                               source=str(run.config))
    assert [float(exp.dt_exact(g)) for g in (1, 2, 3)] == [7.5, 1.5, 0.5]

    # Times: start = launch - 3 h, down to the source's forcing cadence;
    # the end is landing rounded up; d03 is born an hour before launch
    # from its parent, and writes history from launch to landing.
    plan = json.loads((run.directory / "plan.json").read_text())
    cadence = timedelta(seconds=plan["times"]["forcing_cadence_s"])
    start = document["experiment"]["start_time"]
    launch = track.launch.time.replace(tzinfo=None)
    landing = track.landing.time.replace(tzinfo=None)
    assert launch - timedelta(hours=3) - cadence < start
    assert start <= launch - timedelta(hours=3)
    assert (start - datetime(2025, 6, 1)) % cadence == timedelta(0)
    end = start + timedelta(seconds=document["experiment"]["run_seconds"])
    assert landing <= end < landing + cadence
    d03 = domains[3]
    assert d03["start_from"] == "parent"
    born = d03["start_time"]
    assert born == (launch - timedelta(hours=1)).replace(second=0,
                                                         microsecond=0)
    # The window opens and closes on the 120 s history grid from d03's birth.
    assert d03["history_begin_s"] == (
        math.floor((launch - born).total_seconds() / 120) * 120)
    assert d03["history_end_s"] == (
        math.ceil((landing - born).total_seconds() / 120) * 120)
    assert d03["history_interval_s"] == 120.0

    # Vertical: 256 levels, the top 7 km over the peak, clipped.
    shared = document["shared"]
    assert shared["nz"] == 256 and len(shared["eta_levels"]) == 257
    assert tp.P_TOP_RANGE_PA[0] <= shared["p_top"] <= tp.P_TOP_RANGE_PA[1]
    eta = (run.directory / "straight-run.eta.txt").read_text()
    assert eta.count("\n") > 257

    # Nests run large-eddy; the root keeps its PBL scheme.
    for gid in (2, 3):
        assert domains[gid]["bl_pbl_physics"] == 0
        assert domains[gid]["km_opt"] == 2
        assert domains[gid]["diff_opt"] == 2
    assert shared["bl_pbl_physics"] != 0

    # Output: d03 only, to the {name} store, frames removed once committed.
    icechunk = document["output"]["icechunk"]
    assert icechunk == {"url": "{name}.icechunk", "domains": [1, 2, 3],
                        "delete_after_commit": True, "compress": True}

    assert plan["schema"] == tp.PLAN_SCHEMA
    assert plan["source"] in ("hrrr", "gfs")
    estimate = plan["estimate"]
    cells = sum(d["cells"] for d in estimate["domains"])
    assert estimate["total_cells"] == cells
    lo, hi = estimate["gpu_mem_gib_range"]
    assert lo == pytest.approx(3.0 + cells * 0.54 / 1024 ** 2, abs=0.01)
    assert hi == pytest.approx(3.0 + cells * 0.84 / 1024 ** 2, abs=0.01)
    assert estimate["gpu_mem_gib"] >= hi
    fast, slow = estimate["wall_h_range"]
    assert fast == pytest.approx(estimate["cell_steps"] / 20e6 / 3600, abs=0.01)
    assert slow == pytest.approx(estimate["cell_steps"] / 12e6 / 3600, abs=0.01)
    assert estimate["output"]["bytes"] > 0
    # d03 integrates only from its birth.
    d03_est = next(d for d in estimate["domains"] if d["grid_id"] == 3)
    assert d03_est["active_s"] < plan["times"]["run_seconds"]


def test_a_regional_source_short_of_the_top_gives_way_to_gfs(monkeypatch):
    """Auto falls back to GFS where the suggested source's column stops
    below the model top; a named source is refused instead."""

    reach = {"hrrr": 5000.0}
    monkeypatch.setattr(tp, "_source_reach_pa",
                        lambda source: reach.get(source))
    track = _track(lat0=40.0, lon0=-100.0)
    assert track.boundary_source == "hrrr"
    p_top, _ = tp.model_top_pa(track)
    assert p_top < 5000.0
    source, notes = tp.choose_source(track, "auto", p_top)
    assert source == "gfs" and "using gfs" in notes[0]
    with pytest.raises(tp.TrackPlanRefused, match="--source gfs"):
        tp.choose_source(track, "hrrr", p_top)
    # A top the source reaches keeps it.
    assert tp.choose_source(track, "auto", 6000.0) == ("hrrr", [])


def test_a_track_outside_the_regional_grids_takes_gfs():
    track = _track(lat0=-30.0, lon0=20.0)
    source, notes = tp.choose_source(track, "auto", 2000.0)
    assert source == "gfs" and "era5" in notes[0]


def test_hrrr_at_its_own_top_writes_the_route_inputs(tmp_path):
    track = _track(peak=12000.0)
    run = tp.plan_track(track, "regional", tmp_path,
                        tp.PlanOptions(source="hrrr", p_top_pa=5000.0))
    assert run.plan["source"] == "hrrr"
    names = sorted(path.name for path in run.companions)
    assert "regional.namelist.wps" in names
    assert "regional.eta.txt" in names
    document = _config(run)
    # Hourly forcing: the start is launch - 3 h on the hour.
    start = document["experiment"]["start_time"]
    assert start == (track.launch.time.replace(tzinfo=None)
                     - timedelta(hours=3)).replace(minute=0, second=0)
    # namelist.wps carries the same nest placement as the TOML.
    wps = (run.directory / "regional.namelist.wps").read_text()
    domains = _domains(document)
    i_line = next(line for line in wps.splitlines()
                  if line.strip().startswith("i_parent_start"))
    assert i_line.split("=")[1].replace(" ", "").rstrip(",").split(",") == [
        str(domains[g]["i_parent_start"]) for g in (1, 2, 3)]


def test_model_top_is_clipped_to_its_range():
    low = _track(peak=6000.0)
    high = _track(peak=44000.0, climb=10.0)
    assert tp.model_top_pa(low)[0] == tp.P_TOP_RANGE_PA[1]
    assert tp.model_top_pa(high)[0] == tp.P_TOP_RANGE_PA[0]
    assert tp.model_top_pa(low, 2500.0) == (2500.0, "requested")


# ---------------------------------------------------------------------------
# A slow loop: small envelope, static nests hold it


def test_a_slow_loop_is_held_by_the_static_nests(tmp_path):
    track = _track(speed=2.0, turn_deg_s=0.5, float_s=1800.0)
    run = tp.plan_track(track, "loop", tmp_path,
                        tp.PlanOptions(relocation="static"))
    assert run.plan["relocation"]["mode"] == "static"
    assert run.plan["relocation"]["d03_coverage"] == pytest.approx(1.0)
    # d01 is the envelope plus at least the 60 km margin on each side.
    n1x, n1y = run.plan["layout"]["dims"][0]
    assert min(n1x, n1y) * 1.5 >= 2 * tp.DEFAULT_MARGIN_KM
    assert "[relocation]" not in run.config.read_text()


def test_the_itinerary_is_written_when_this_woof_has_one(tmp_path, monkeypatch):
    """With an itinerary module present, its placement and [relocation]
    land in the config, and the config still loads."""

    calls = {}

    class Plan:
        moves = (object(),)
        containment_slides = 0
        worst_margin_cells = 50.0

        def __init__(self, exp, start):
            d03 = exp.domain(3)
            offset = (d03.start_time - exp.start_time).total_seconds()
            self.placements = {2: (int(exp.domain(2).i_parent_start),
                                   int(exp.domain(2).j_parent_start)),
                               3: (int(d03.i_parent_start) + 1,
                                   int(d03.j_parent_start))}
            self.offset = offset

        def relocation_table(self):
            return {"enabled": True, "grid_id": 3,
                    "mode": "discrete-cycle-boundary",
                    "cadence_seconds": 300.0,
                    "move": [{"at_seconds": 300.0 * math.ceil(
                        (self.offset + 600.0) / 300.0),
                              "di_parent_cells": 1, "dj_parent_cells": 0}]}

    def itinerary_from_track(times, lats, lons, exp, start, cadence, margin,
                             **kwargs):
        calls.update(n=len(times), cadence=cadence, margin=margin,
                     heights="heights" in kwargs)
        return Plan(exp, start)

    module = types.SimpleNamespace(itinerary_from_track=itinerary_from_track)
    monkeypatch.setattr(tp, "_relocation_module", lambda: module)
    track = _track(speed=10.0)
    run = tp.plan_track(track, "follow", tmp_path)
    assert calls["cadence"] == 300.0 and calls["margin"] == 40
    assert calls["heights"]
    document = _config(run)
    assert document["relocation"]["grid_id"] == 3
    assert document["relocation"]["move"][0]["di_parent_cells"] == 1
    assert run.plan["relocation"]["mode"] == "track"
    assert run.plan["layout"]["placements"]["d03"][0] == \
        _domains(document)[3]["i_parent_start"]


def test_the_nests_follow_the_track_with_this_woofs_itinerary(
        tmp_path, monkeypatch, capsys):
    """With woof relocation from-track in this woof, d03 follows the path:
    the config carries [[relocation.move]] rows and a d02 containment,
    loads, and passes the woof go dry run."""

    from woof import go_cli
    from woof.cli import main

    monkeypatch.setattr(tp, "_relocation_module", _REAL_RELOCATION_MODULE)
    if tp._relocation_module() is None:
        pytest.skip("this woof has no track itinerary")
    run = tp.plan_track(_track(speed=10.0), "follow-real", tmp_path / "p",
                        tp.PlanOptions(source="gfs"))
    document = _config(run)
    relocation = document["relocation"]
    assert relocation["grid_id"] == 3
    assert relocation["containment"]["grid_id"] == 2
    moves = relocation["move"]
    assert len(moves) >= 5
    # A moving d03 starts with the tree: the forecast refuses a late birth
    # (parent- or analysis-born) for a relocation target under a sliding d02.
    d03 = _domains(document)[3]
    assert "start_time" not in d03 and "start_from" not in d03
    assert all(move["at_seconds"] > 0 for move in moves)
    assert all(move["at_seconds"] % relocation["cadence_seconds"] == 0
               for move in moves)
    assert run.plan["relocation"]["mode"] == "track"
    assert run.plan["relocation"]["moves"] == len(moves)
    bridge = tmp_path / "bridge"
    bridge.write_bytes(b"stub")
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: bridge)
    capsys.readouterr()
    assert main(["go", str(run.config), "--dry-run", "--products", "none",
                 "--outdir", str(tmp_path / "go")]) == 0
    from woof.experiment import load_experiment
    loaded = load_experiment(run.config)
    assert loaded.relocation.enabled
    assert len(loaded.relocation.moves) == len(moves)
    assert loaded.relocation.containment.grid_id == 2


def test_relocation_track_without_the_itinerary_is_refused(tmp_path):
    with pytest.raises(tp.TrackPlanRefused, match="--relocation static"):
        tp.plan_track(_track(), "need", tmp_path,
                      tp.PlanOptions(relocation="track"))


# ---------------------------------------------------------------------------
# Projection edges


def test_a_track_beside_the_antimeridian_plans_across_it(tmp_path):
    """d01 straddles 180 degrees; the plan still loads."""

    track = _track(lat0=-40.0, lon0=179.95, heading=270.0)
    run = tp.plan_track(track, "dateline", tmp_path)
    assert run.plan["source"] == "gfs"
    projection = run.plan["layout"]["projection"]
    n1x, _ = run.plan["layout"]["dims"][0]
    # The root reaches past the antimeridian from a centre west of it.
    assert projection["ref_lon"] + 0.5 * n1x * 1.5 / 85.0 > 180.0


def test_a_track_crossing_the_antimeridian_is_refused_at_ingest():
    times, lats, lons, heights = _fixes(lat0=-40.0, lon0=179.9, heading=90.0)
    lons = [lon - 360.0 if lon > 180.0 else lon for lon in lons]
    with pytest.raises(ValueError, match="antimeridian"):
        track_from_fixes("crossing", times, lats, lons, heights)


def test_a_track_across_the_projection_switch_latitude(tmp_path):
    """The auto projection is chosen at the envelope's centre; a track
    crossing the Mercator/Lambert switch is still wholly inside d01."""

    from woof.domain_wizard import MERCATOR_MAX_LAT
    from woof.source_coverage import _root_grid

    track = _track(lat0=MERCATOR_MAX_LAT - 0.4, lon0=-80.0, heading=0.0,
                   speed=15.0)
    assert min(track.lat) < MERCATOR_MAX_LAT < max(track.lat)
    run = tp.plan_track(track, "switch", tmp_path)
    layout = run.plan["layout"]
    n1x, n1y = layout["dims"][0]
    grid = _root_grid(layout["projection"], n1x, n1y, 1500.0)
    x, y = grid.latlon_to_ij(list(track.lat), list(track.lon))
    margin = tp.DEFAULT_MARGIN_KM / 1.5
    assert min(x) >= margin - 1 and max(x) <= n1x - margin + 1
    assert min(y) >= margin - 1 and max(y) <= n1y - margin + 1


def test_a_polar_track_is_refused(tmp_path):
    track = _track(lat0=89.3, lon0=0.0)
    with pytest.raises(ValueError, match="pole"):
        tp.plan_track(track, "polar", tmp_path)


# ---------------------------------------------------------------------------
# Too fast to follow


class _StayingPlan:
    """An itinerary whose track never leaves d03's margins."""

    moves = ()
    containment_slides = 0
    worst_margin_cells = 40.0

    def __init__(self, exp):
        self.placements = {g: (int(exp.domain(g).i_parent_start),
                               int(exp.domain(g).j_parent_start))
                           for g in (2, 3)}

    def relocation_table(self):
        return {}


def _following(monkeypatch):
    module = types.SimpleNamespace(
        itinerary_from_track=lambda times, lats, lons, exp, *a, **k:
        _StayingPlan(exp))
    monkeypatch.setattr(tp, "_relocation_module", lambda: module)


def test_a_track_too_fast_for_the_cadence_is_refused_with_one_that_works(
        tmp_path, monkeypatch):
    _following(monkeypatch)
    track = _track(speed=150.0)
    with pytest.raises(tp.TrackPlanRefused) as caught:
        tp.plan_track(track, "fast", tmp_path)
    message = str(caught.value)
    assert "--relocation-cadence-s 180" in message
    # The suggested cadence works.
    run = tp.plan_track(track, "fast", tmp_path,
                        tp.PlanOptions(relocation_cadence_s=180.0))
    assert run.plan["follow_check"]["cadence_s"] == 180.0
    # Static nests follow nothing, so no cadence is too slow for them.
    tp.plan_track(track, "fast-static", tmp_path,
                  tp.PlanOptions(relocation="static"))


def test_without_the_itinerary_a_fast_track_plans_static_nests(tmp_path):
    """No cadence matters when d03 cannot move; the coverage says so."""

    run = tp.plan_track(_track(speed=150.0), "fast", tmp_path)
    assert run.plan["relocation"]["mode"] == "static"
    assert run.plan["relocation"]["d03_coverage"] < tp.LOW_COVERAGE
    assert any(note.startswith("WARNING relocation") for note in run.notes)


def test_a_doubling_back_track_spans_its_whole_excursion(tmp_path,
                                                         monkeypatch):
    """A track that goes out and back inside one window is measured by
    everything it spans, not by its distance from the window's start."""

    _following(monkeypatch)
    times, lats, lons, heights = _fixes(speed=10.0)
    # Swing 15 km east and 15 km back past the start inside 300 s.
    base = len(times) // 2
    for k, offset_km in enumerate((0.0, 15.0, 0.0, -15.0, 0.0)):
        lons[base + k] += offset_km / (111.195 * math.cos(math.radians(40.0)))
    track = track_from_fixes("swing", [T0 + timedelta(seconds=60 * k)
                                       for k in range(len(times))],
                             lats, lons, heights)
    with pytest.raises(tp.TrackPlanRefused, match="relocation window"):
        tp.plan_track(track, "swing", tmp_path)


def test_a_track_no_whole_minute_can_follow_names_the_window(tmp_path,
                                                             monkeypatch):
    _following(monkeypatch)
    track = _track(speed=900.0, peak=6000.0, climb=20.0)
    with pytest.raises(tp.TrackPlanRefused, match="--d03-window-km"):
        tp.plan_track(track, "jet", tmp_path,
                      tp.PlanOptions(relocation_cadence_s=600.0,
                                     d03_window_km=12.0))


# ---------------------------------------------------------------------------
# Job documents and the command line


def test_job_documents_validate_inline_and_by_url(tmp_path):
    from woof.go_job import parse_job

    run = tp.plan_track(_track(), "jobbed", tmp_path)
    inline = tp.job_document(run, icechunk_url="s3://bucket/out/{name}.icechunk")
    job = parse_job(inline, source="test")
    assert job.job_id == "jobbed"
    assert job.icechunk_url == "s3://bucket/out/jobbed.icechunk"
    assert job.config_toml == run.config.read_text(encoding="utf-8")
    assert {c.path for c in job.files} == {p.name for p in run.companions}
    assert job.gpu_mem_gib == run.plan["estimate"]["gpu_mem_gib"]
    assert job.products == "none"
    by_url = tp.job_document(run, inputs_prefix="s3://bucket/inputs/")
    job = parse_job(by_url, source="test", output_prefix="s3://bucket/out")
    assert job.config_ref == "s3://bucket/inputs/jobbed/jobbed.toml"
    assert all(c.ref.startswith("s3://bucket/inputs/jobbed/")
               for c in job.files)
    assert "icechunk_url" not in by_url


def test_cli_plans_a_directory_of_paths(tmp_path, capsys):
    from woof.cli import main

    paths = tmp_path / "paths"
    paths.mkdir()
    _write_csv(paths / "alpha.csv", speed=8.0)
    _write_csv(paths / "beta [2].csv", speed=12.0, heading=45.0)
    out = tmp_path / "plans"
    code = main(["track", "plan", str(paths), "--names-from", "stem",
                 "--out", str(out), "--time-col", "when", "--lat-col", "y",
                 "--lon-col", "x", "--height-col", "alt",
                 "--drop-when", "fix=0", "--job-json",
                 "--inputs-prefix", "s3://bucket/inputs",
                 "--icechunk-url", "s3://bucket/out/{name}.icechunk"])
    printed = capsys.readouterr().out
    assert code == 0
    # A stem's characters a job id cannot carry become '-'.
    assert "alpha" in printed and "beta-2" in printed
    assert "woof go-job-push" in printed
    for name in ("alpha", "beta-2"):
        assert (out / name / f"{name}.toml").is_file()
        assert (out / name / "plan.json").is_file()
    jobs = sorted((out / "jobs").rglob("*.json"))
    assert [path.stem for path in jobs] == ["alpha", "beta-2"]
    document = json.loads(jobs[0].read_text())
    assert document["icechunk_url"] == "s3://bucket/out/alpha.icechunk"
    summary = json.loads((out / "plan-summary.json").read_text())
    assert [run["name"] for run in summary["runs"]] == ["alpha", "beta-2"]


def test_cli_refuses_one_store_for_several_runs(tmp_path, capsys):
    from woof.cli import main

    paths = tmp_path / "paths"
    paths.mkdir()
    _write_csv(paths / "a.csv")
    _write_csv(paths / "b.csv")
    code = main(["track", "plan", str(paths), "--names-from", "stem",
                 "--out", str(tmp_path / "o"), "--time-col", "when",
                 "--lat-col", "y", "--lon-col", "x", "--height-col", "alt",
                 "--icechunk-url", "s3://bucket/one.icechunk"])
    assert code != 0
    assert "{name}" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_a_job_larger_than_every_tier_is_not_written(tmp_path, capsys):
    from woof.cli import main

    path = _write_csv(tmp_path / "big.csv")
    code = main(["track", "plan", str(path), "--name", "big",
                 "--out", str(tmp_path / "o"), "--time-col", "when",
                 "--lat-col", "y", "--lon-col", "x", "--height-col", "alt",
                 "--drop-when", "fix=0", "--job-json",
                 "--gpu-mem-gib", "120"])
    assert code == 2
    assert "largest" in capsys.readouterr().err
    assert not (tmp_path / "o" / "jobs").exists()


def test_cli_needs_a_name(tmp_path, capsys):
    from woof.cli import main

    path = _write_csv(tmp_path / "one.csv")
    code = main(["track", "plan", str(path), "--out", str(tmp_path / "o"),
                 "--time-col", "when", "--lat-col", "y", "--lon-col", "x",
                 "--height-col", "alt"])
    assert code != 0
    assert "--name" in capsys.readouterr().err


def test_generated_config_passes_woof_go_dry_run(tmp_path, monkeypatch, capsys):
    from woof import go_cli
    from woof.cli import main

    bridge = tmp_path / "gfs_grib2_bridge"
    bridge.write_bytes(b"stub")
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: bridge)
    for source in ("auto", "gfs"):
        run = tp.plan_track(_track(), f"dry-{source}", tmp_path / "plans",
                            tp.PlanOptions(source=source))
        capsys.readouterr()
        code = main(["go", str(run.config), "--dry-run", "--products",
                     "none", "--outdir", str(tmp_path / "go")])
        printed = capsys.readouterr().out
        assert code == 0, printed
        assert f"dry-{source}.icechunk as it lands" in printed
        if run.plan["source"] == "gfs":
            assert "prepared_domain_tree_forecast" in printed
        assert not (tmp_path / "go").exists()


# ---------------------------------------------------------------------------
# Seams for features that may not be in this woof


def test_physics_takes_a_registered_tree_suite(monkeypatch):
    from woof import domain_wizard as dw
    from woof import physics_compat

    class Tree:
        root = "root-template"

        def nest_overrides(self):
            return {"bl_pbl_physics": 0, "km_opt": 2, "mix_isotropic": 1}

    monkeypatch.setattr(physics_compat, "TREE_PHYSICS_PROFILES",
                        {"les-tree": Tree()}, raising=False)
    original = dw.render_config

    def render_config(*args, nest_physics=None, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(dw, "render_config", render_config)
    physics, notes = tp.resolve_physics("auto", "gfs")
    assert physics["profile"] == "root-template"
    assert physics["nest_physics"]["km_opt"] == 2
    assert notes == []
    with pytest.raises(tp.TrackPlanRefused, match="explicit"):
        tp.resolve_physics("no-such-suite", "gfs")


def test_physics_falls_back_to_explicit_keys(monkeypatch):
    from woof import physics_compat

    monkeypatch.delattr(physics_compat, "TREE_PHYSICS_PROFILES", raising=False)
    physics, notes = tp.resolve_physics("auto", "gfs")
    assert physics["nest"] == tp.EXPLICIT_NEST_PHYSICS
    assert physics["mix"]["choices"] == tp.EXPLICIT_ROOT_PHYSICS
    assert notes and "explicitly" in notes[0]


def test_output_preset_prefers_the_lean_track_preset(monkeypatch):
    from woof.io import history_selection

    presets = dict(history_selection.HISTORY_PRESETS)
    presets.pop("balloon", None)
    monkeypatch.setattr(history_selection, "HISTORY_PRESETS", presets)
    assert tp.resolve_output_preset("auto")[0] == "minimal"
    presets["balloon"] = frozenset()
    assert tp.resolve_output_preset("auto") == ("balloon", [])
    with pytest.raises(tp.TrackPlanRefused):
        tp.resolve_output_preset("no-such-preset")



# ---------------------------------------------------------------------------
# A UK path: the registry's UKV source


def test_a_uk_path_plans_on_ukv(tmp_path, monkeypatch, capsys):
    from woof import go_cli
    from woof.cli import main

    track = _track(lat0=52.0, lon0=-1.0, speed=3.0, heading=0.0)
    assert track.boundary_source == "ukv"
    run = tp.plan_track(track, "uk-short", tmp_path)
    assert run.plan["source"] == "ukv"
    assert run.plan["times"]["forcing_cadence_s"] == 3600
    assert run.plan["fetch"]["cadence"] == 1
    bridge = tmp_path / "bridge"
    bridge.write_bytes(b"stub")
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: bridge)
    capsys.readouterr()
    assert main(["go", str(run.config), "--dry-run", "--products", "none",
                 "--outdir", str(tmp_path / "go")]) == 0


def test_a_long_uk_window_takes_the_three_hourly_runs(tmp_path, monkeypatch):
    """Past the archive, UKV runs step three-hourly beyond T+54: a longer
    window is planned (and fetched) on 3 h boundaries."""

    monkeypatch.setattr(sys.modules[__name__], "T0",
                        datetime(2026, 9, 1, 6, 0, tzinfo=timezone.utc))
    track = _track(lat0=52.0, lon0=-1.0, speed=0.3, heading=0.0,
                   float_s=58 * 3600.0, step=120.0)
    run = tp.plan_track(track, "uk-long", tmp_path,
                        tp.PlanOptions(relocation="static"))
    assert run.plan["source"] == "ukv"
    assert run.plan["times"]["run_seconds"] > 54 * 3600
    assert run.plan["fetch"]["cadence"] == 3
    assert run.plan["times"]["forcing_cadence_s"] == 10800
    assert any("every 3 h" in note for note in run.notes)
    wps = (run.directory / "uk-long.namelist.wps").read_text()
    assert "interval_seconds = 10800" in wps


def test_fit_root_step_makes_steps_divide_history_and_radiation():
    from fractions import Fraction
    from woof.track.plan import _fit_root_step

    def tree(step, num, den):
        return [
            {"grid_id": 1, "time_step": step, "time_step_fract_num": num,
             "time_step_fract_den": den, "radt": 10.0},
            {"grid_id": 2, "parent_id": 1, "parent_time_step_ratio": 3,
             "radt": 5.0},
            {"grid_id": 3, "parent_id": 2, "parent_time_step_ratio": 3,
             "radt": 5.0},
        ]

    # An odd wizard step (d03 at 100/69 s) is shortened, never lengthened.
    domains = tree(13, 3, 69)
    _fit_root_step(domains, 120.0)
    root = domains[0]
    root_dt = root["time_step"] + Fraction(root["time_step_fract_num"],
                                           root["time_step_fract_den"])
    d03_dt = root_dt / 9
    assert root_dt <= Fraction(900, 69)
    assert (Fraction(120) / d03_dt).denominator == 1
    assert (Fraction(600) / root_dt).denominator == 1
    assert (Fraction(300) / (d03_dt * 3)).denominator == 1
    # A step that already fits (the default 7.5/1.5/0.5 s ladder) is kept.
    domains = [
        {"grid_id": 1, "time_step": 7, "time_step_fract_num": 1,
         "time_step_fract_den": 2, "radt": 10.0},
        {"grid_id": 2, "parent_id": 1, "parent_time_step_ratio": 5,
         "radt": 5.0},
        {"grid_id": 3, "parent_id": 2, "parent_time_step_ratio": 3,
         "radt": 5.0},
    ]
    _fit_root_step(domains, 120.0)
    assert (domains[0]["time_step"], domains[0]["time_step_fract_num"],
            domains[0]["time_step_fract_den"]) == (7, 1, 2)


def test_history_window_sits_on_the_history_grid():
    from datetime import datetime
    from woof.track.plan import PlanTimes

    start = datetime(2026, 1, 1, 0, 0, 0)
    times = PlanTimes(
        start=start, end=datetime(2026, 1, 1, 6), launch=datetime(2026, 1, 1, 2, 3, 37),
        landing=datetime(2026, 1, 1, 4, 59, 1), d03_start=start, cadence_s=3600,
        cycle=start, start_hour=0, history_s=120.0)
    begin, end = times.history_window()
    # A launch 7417 s in opens the window at 7320 s and a landing at 17941 s
    # closes it at 18000 s: multiples of the 120 s interval, so every frame
    # falls on a whole d03 step (a 4/3 s step does not divide 7417 s).
    assert (begin, end) == (7320, 18000)
    assert begin % 120 == 0 and end % 120 == 0



def test_every_grid_writes_the_flight_window_on_one_interval(tmp_path):
    """d01 and d02 write history on d03's interval over the same flight
    window (none during the spin-up), and the export takes every grid."""
    track = _track(speed=25.0)
    run = tp.plan_track(track, "all-grids", tmp_path)
    document = _config(run)
    domains = _domains(document)
    d03 = domains[3]
    born = ((d03["start_time"] - document["experiment"]["start_time"])
            .total_seconds() if "start_time" in d03 else 0)
    for gid in (1, 2):
        table = domains[gid]
        assert table["history_interval_s"] == d03["history_interval_s"]
        assert table["history_begin_s"] == d03["history_begin_s"] + born
        assert table["history_end_s"] == d03["history_end_s"] + born
    assert document["output"]["icechunk"]["domains"] == [1, 2, 3]
    out = run.plan["estimate"]["output"]
    assert set(out["bytes_per_frame_by_grid"]) == {"d01", "d02", "d03"}
    assert out["bytes"] == out["bytes_per_frame"] * out["frames"]
