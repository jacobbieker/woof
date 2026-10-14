"""``woof.energy.source_select``: the boundary source for a region and a date.

Every answer is read from tables this build ships (the source registry,
the fetch route table and the calendar's no-network verdict) against an
injected clock, so these tests reach no server.  The two planner hooks
(``--source auto``) are exercised through the real parser and the real
``woof domain`` emission at a coarse spacing.  CPU only.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import shlex

import pytest

from woof.energy import source_select as ss
from woof.energy.contracts import ContractError, Plan, load_plan, load_sites

FIXTURES = Path(__file__).parent / "fixtures" / "energy"
SITES_WALES = FIXTURES / "sites_wales.json"
DOC = Path(__file__).resolve().parents[1] / "docs" / "energy-hindcast" / \
    "source-select.md"

#: The clock every test reads: well after every cycle asked about.
NOW = datetime(2026, 10, 14, 12)
#: Wales (the fixture's sites) and a West Texas wind belt, W,S,E,N.
WALES = (-4.5, 51.4, -3.0, 52.5)
TEXAS = (-101.5, 32.0, -100.5, 33.0)


def _checks(choice, source):
    for candidate in choice.candidates:
        if candidate.source == source:
            return {r.check: r.reason for r in candidate.rejections}
    raise AssertionError(f"{source} is not a candidate")


# --------------------------------------------------------------------------
# the choice by place and date


def test_wales_before_the_met_office_orography_era_takes_ifs():
    choice = ss.select_source(WALES, "2025-11-01T00", 30, p_top_pa=5000.0,
                              now=NOW)
    assert choice.source == "ecmwf-open-data"
    met = _checks(choice, "met-office-global")
    assert set(met) == {"date"}
    assert "orography" in met["date"]
    for regional in ("hrrr-native", "hrrr", "rap"):
        assert "coverage" in _checks(choice, regional)


def test_wales_after_the_orography_era_takes_the_met_office():
    choice = ss.select_source(WALES, "2026-01-15T00", 24, p_top_pa=5000.0,
                              now=NOW)
    assert choice.source == "met-office-global"
    assert choice.rejections.keys() == {"hrrr-native", "hrrr", "rap"}


def test_wales_older_than_every_forecast_archive_takes_era5():
    choice = ss.select_source(WALES, "2019-06-01T12", 24, now=NOW)
    assert choice.source == "era5"
    assert "2021-03-22" in _checks(choice, "gfs")["date"]
    assert "2022-01-25" in _checks(choice, "ecmwf-open-data")["date"]


def test_ifs_before_its_first_preparable_cycle_falls_to_gfs():
    choice = ss.select_source(WALES, "2024-01-15T00", 24, now=NOW)
    assert choice.source == "gfs"
    assert "2024-03-18T12" in _checks(choice, "ecmwf-open-data")["date"]


def test_conus_takes_hrrr_and_names_the_undeclared_native_archive():
    choice = ss.select_source(TEXAS, "2026-01-15T00", 24, p_top_pa=5000.0,
                              now=NOW)
    assert choice.source == "hrrr"
    native = _checks(choice, "hrrr-native")
    assert set(native) == {"date"}
    assert "accept_unknown_archive" in native["date"]


def test_accepting_an_undeclared_archive_takes_hrrr_native():
    choice = ss.select_source(TEXAS, "2026-01-15T00", 24, now=NOW,
                              accept_unknown_archive=True)
    assert choice.source == "hrrr-native"
    assert any("accept_unknown_archive" in note for note in choice.notes)


def test_a_cycle_inside_the_nomads_window_takes_hrrr_native():
    choice = ss.select_source(TEXAS, "2026-10-13T12", 24, now=NOW)
    assert choice.source == "hrrr-native"


def test_spinup_counts_toward_the_horizon():
    # HRRR's 00Z run reaches f048: 40 h fits, 40 + 12 h of spin-up does not
    plain = ss.select_source(TEXAS, "2026-01-15T00", 40, now=NOW,
                             priority=["hrrr", "gfs"])
    assert plain.source == "hrrr"
    spun = ss.select_source(TEXAS, "2026-01-15T00", 40, now=NOW,
                            spinup_hours=12, priority=["hrrr", "gfs"])
    assert spun.source == "gfs"
    assert "f052" in _checks(spun, "hrrr")["horizon"]
    assert spun.horizon_hours == 52.0


def test_rap_extended_runs_only():
    at_03 = ss.select_source(TEXAS, "2026-01-15T03", 30, now=NOW,
                             priority=["rap", "gfs"])
    assert at_03.source == "rap"
    at_00 = ss.select_source(TEXAS, "2026-01-15T00", 30, now=NOW,
                             priority=["rap", "gfs"])
    assert at_00.source == "gfs"
    assert "f021" in _checks(at_00, "rap")["horizon"]


def test_a_run_not_yet_published_is_passed_over():
    choice = ss.select_source(WALES, "2026-10-14T06", 24, now=NOW,
                              priority=["ecmwf-open-data", "gfs"])
    assert choice.source == "gfs"
    assert "not published yet" in _checks(choice, "ecmwf-open-data")["date"]


def test_coverage_margin_reaches_past_the_sites():
    # a box near the HRRR grid's northern edge (about 52.5 N at 100 W) is
    # covered; widened 500 km into Canada it is not
    edge = (-100.0, 50.0, -99.0, 51.0)
    near = ss.select_source(edge, "2026-01-15T00", 24, now=NOW,
                            priority=["hrrr", "gfs"])
    assert near.source == "hrrr"
    wide = ss.select_source(edge, "2026-01-15T00", 24, now=NOW,
                            priority=["hrrr", "gfs"], margin_km=500.0)
    assert wide.source == "gfs"
    assert "coverage" in _checks(wide, "hrrr")
    assert wide.region["margin_km"] == 500.0
    assert wide.region["sampled_bbox"][3] > 51.0 + 4.0


def test_a_run_younger_than_its_usual_publication_delay_is_passed_over():
    # GFS: a whole run is usually published 6 h after its start; the
    # calendar's verdict leaves that to the fetch's probe, the selector
    # (which reaches no server) does not take it
    fresh = datetime(2026, 10, 14, 12, 5)
    with pytest.raises(ss.SourceSelectRefused) as caught:
        ss.select_source(WALES, "2026-10-14T12", 24, now=fresh,
                         priority=["gfs"])
    assert "usually published 6 h" in str(caught.value)
    later = ss.select_source(WALES, "2026-10-14T12", 24,
                             now=datetime(2026, 10, 14, 18, 5),
                             priority=["gfs"])
    assert later.source == "gfs"


def test_a_defaulted_start_keeps_the_preferred_source():
    # minutes after 12Z nothing has posted the 12Z run; the newest cycle
    # the Met Office has published beats GFS's (or anyone's) newer one
    fresh = datetime(2026, 10, 14, 12, 5)
    choice = ss.select_newest(WALES, "2026-10-14T12", 24, p_top_pa=5000.0,
                              now=fresh)
    assert choice.source == "met-office-global"
    assert choice.cycle < datetime(2026, 10, 14, 12)
    assert any(note.startswith("default start: searched 8 cycles")
               for note in choice.notes)
    # a rank-1 source at the newest cycle stops the search there
    texas = ss.select_newest(TEXAS, "2026-10-13T12", 24, now=NOW)
    assert (texas.source, texas.cycle) == ("hrrr-native",
                                            datetime(2026, 10, 13, 12))


def test_a_defaulted_start_with_no_source_anywhere_is_refused():
    with pytest.raises(ss.SourceSelectRefused) as caught:
        ss.select_newest(WALES, "2019-06-01T12", 24, now=NOW,
                         priority=["gfs"])
    assert caught.value.choice is not None


def test_the_margin_is_sampled_across_the_antimeridian():
    lat, lon = ss._samples(ss._widen((-179.5, 51.0, -178.5, 52.0), 200.0))
    assert lon.max() > 170.0 and lon.min() < -170.0
    assert (lon >= -180.0).all() and (lon < 180.0).all()


# --------------------------------------------------------------------------
# column top


def test_a_source_that_cannot_reach_the_model_top_is_passed_over():
    # ICON-D2 certifies a 6000 Pa top and no extension
    choice = ss.select_source(
        (8.0, 50.0, 9.0, 51.0), "2026-10-14T00", 24, p_top_pa=5000.0,
        now=NOW, priority=["icon-d2", "gfs"])
    assert choice.source == "gfs"
    assert "6000" in _checks(choice, "icon-d2")["column_top"]


def test_gfs_extends_its_ladder_to_the_model_top():
    choice = ss.select_source(WALES, "2026-01-15T00", 24, p_top_pa=5000.0,
                              now=NOW, priority=["gfs"])
    assert choice.source == "gfs"
    assert any("extends" in note for note in choice.notes)
    with pytest.raises(ss.SourceSelectRefused) as caught:
        ss.select_source(WALES, "2026-01-15T00", 24, p_top_pa=0.5, now=NOW,
                         priority=["gfs"])
    assert "column_top" in str(caught.value)


def test_no_model_top_skips_the_check():
    choice = ss.select_source(
        (8.0, 50.0, 9.0, 51.0), "2026-10-14T00", 24, now=NOW,
        priority=["icon-d2", "gfs"])
    assert "column_top" not in _checks(choice, "icon-d2")


# --------------------------------------------------------------------------
# priority, refusals and the receipt


def test_priority_takes_aliases_and_a_comma_string():
    choice = ss.select_source(WALES, "2026-01-15T00", 24, now=NOW,
                              priority="ifs,ukmo-global")
    assert choice.source == "ecmwf-open-data"
    assert choice.priority == ("ecmwf-open-data", "met-office-global")
    assert choice.priority_origin == "caller"
    assert choice.candidates[0].requested == "ifs"


@pytest.mark.parametrize("priority, match", [
    (["gfs", "no-such-model"], "unregistered"),
    (["auto", "gfs"], "'auto' cannot appear"),
    (["gfs", "gfs-0p25"], "twice"),
    ([], "empty"),
])
def test_bad_priority_lists_are_refused(priority, match):
    with pytest.raises(ss.SourceSelectRefused, match=match):
        ss.select_source(WALES, "2026-01-15T00", 24, now=NOW,
                         priority=priority)


def test_the_planner_refusal_ranks_a_source_out():
    choice = ss.select_source(TEXAS, "2026-01-15T00", 24, now=NOW,
                              priority=["hrrr", "gfs"],
                              refuse=lambda s: "nope" if s == "hrrr" else None)
    assert choice.source == "gfs"
    assert _checks(choice, "hrrr") == {"planner": "nope"}


def test_wrf_nests_ranks_hrrr_out():
    from woof.energy import plan_wrf_nests as nests

    choice = ss.select_source(TEXAS, "2026-01-15T00", 24, now=NOW,
                              priority=["hrrr", "gfs"],
                              refuse=nests._planner_refusal)
    assert choice.source == "gfs"
    assert "woof domain" in _checks(choice, "hrrr")["planner"]


def test_no_candidate_refuses_with_every_reason():
    with pytest.raises(ss.SourceSelectRefused) as caught:
        ss.select_source(WALES, "2019-06-01T12", 24, now=NOW,
                         priority=["met-office-global", "gfs"])
    choice = caught.value.choice
    assert choice is not None and choice.source is None
    assert set(choice.rejections) == {"met-office-global", "gfs"}
    assert "gfs: date:" in str(caught.value)


def test_unrunnable_and_unfetchable_rows_are_rejected():
    choice = ss.select_source(TEXAS, "2026-01-15T00", 24, now=NOW,
                              priority=["nam", "era5-l137", "gfs"])
    assert choice.source == "gfs"
    assert "registry" in _checks(choice, "nam")
    assert "acquisition" in _checks(choice, "era5-l137")


def test_receipt_is_pinned_and_reproducible(tmp_path):
    first = ss.select_source(WALES, "2025-11-01T00", 30, p_top_pa=5000.0,
                             now=NOW)
    second = ss.select_source(WALES, "2025-11-01T00", 30, p_top_pa=5000.0,
                              now=NOW)
    assert first.to_json() == second.to_json()
    receipt = first.to_json()
    assert receipt["schema"] == "woof-energy.source-choice.v1"
    assert receipt["checks"] == list(ss.CHECKS)
    assert receipt["now"] == "2026-10-14T12:00:00Z"
    assert receipt["priority"] == list(ss.DEFAULT_PRIORITY)
    assert receipt["priority_origin"] == "default"
    assert len(receipt["route_table_sha256"]) == 64
    statuses = {c["source"]: c["status"] for c in receipt["candidates"]}
    assert statuses["ecmwf-open-data"] == "chosen"
    assert statuses["gfs"] == statuses["era5"] == "eligible"
    assert statuses["met-office-global"] == "rejected"
    path = ss.write_receipt(first, tmp_path / "choice.json")
    assert json.loads(path.read_text()) == receipt
    assert not list(tmp_path.glob("*.partial"))
    summary = first.summary()
    assert summary["source"] == "ecmwf-open-data"
    assert set(summary["rejected"]) == {"hrrr-native", "hrrr", "rap",
                                        "met-office-global"}
    note = first.note()
    assert "ecmwf-open-data" in note and "met-office-global (date)" in note


def test_the_wall_clock_is_the_default():
    from datetime import timezone

    before = datetime.now(timezone.utc).replace(tzinfo=None)
    choice = ss.select_source(WALES, "2019-06-01T12", 24,
                              priority=["era5"])
    assert before <= choice.now
    assert (choice.now - before).total_seconds() < 600


# --------------------------------------------------------------------------
# inputs


def test_region_forms_agree():
    ring = [[-4.5, 51.4], [-3.0, 51.4], [-3.0, 52.5], [-4.5, 52.5],
            [-4.5, 51.4]]
    polygon = {"type": "Polygon", "coordinates": [ring]}
    feature = {"type": "Feature", "geometry": polygon, "properties": {}}
    collection = {"type": "FeatureCollection", "features": [feature]}
    for region in (WALES, polygon, feature, collection, {"bbox": WALES},
                   [(-4.5, 51.4), (-3.0, 52.5)]):
        assert ss.region_bbox(region) == WALES


@pytest.mark.parametrize("region, match", [
    ((-170.0, 50.0, 170.0, 51.0), "antimeridian"),
    ((10.0, 50.0, 5.0, 51.0), "west <= east"),
    ({"type": "Nope"}, "not a geometry"),
    ([(1.0, 2.0, 3.0)], "pairs"),
    ([], "no coordinates|pairs"),
    ([(0.0, 95.0)], "degrees"),
])
def test_bad_regions_are_refused(region, match):
    with pytest.raises(ss.SourceSelectRefused, match=match):
        ss.region_bbox(region)


@pytest.mark.parametrize("cycle, hours, kwargs, match", [
    ("2026-01-15T00:30", 24, {}, "whole hour"),
    ("yesterday", 24, {}, "YYYY"),
    ("2026-01-15T00", 0, {}, "positive"),
    ("2026-01-15T00", float("nan"), {}, "finite"),
    ("2026-01-15T00", 24, {"spinup_hours": -1}, "nonnegative"),
    ("2026-01-15T00", 24, {"p_top_pa": -5.0}, "positive pressure"),
    ("2026-01-15T00", 24, {"margin_km": -1.0}, "margin_km"),
])
def test_bad_requests_are_refused(cycle, hours, kwargs, match):
    with pytest.raises(ss.SourceSelectRefused, match=match):
        ss.select_source(WALES, cycle, hours, now=NOW, **kwargs)


def test_aware_datetimes_are_utc():
    from datetime import timedelta, timezone

    aware = datetime(2026, 1, 15, 1, tzinfo=timezone(timedelta(hours=1)))
    choice = ss.select_source(WALES, aware, 24, now=NOW)
    assert choice.cycle == datetime(2026, 1, 15, 0)


# --------------------------------------------------------------------------
# the plan contract


def test_plan_round_trips_the_receipt(tmp_path):
    from woof.energy.contracts import dump_plan

    receipt = ss.select_source(WALES, "2026-01-15T00", 24,
                               now=NOW).to_json()
    plan = Plan(topology="wrf-tiles", dx_m=100.0, start="2026-01-15T00",
                hours=24.0, domains=[], source="met-office-global",
                source_choice=receipt)
    loaded = load_plan(dump_plan(plan, tmp_path / "plan.json"))
    assert loaded.source_choice == receipt
    # the receipt (it carries the clock) is not part of plan equality
    later = dict(receipt, now="2030-01-01T00:00:00Z")
    assert loaded == Plan(topology="wrf-tiles", dx_m=100.0,
                          start="2026-01-15T00", hours=24.0, domains=[],
                          source="met-office-global", source_choice=later)
    bare = Plan(topology="wrf-tiles", dx_m=100.0, start="2026-01-15T00",
                hours=24.0, domains=[])
    assert "source_choice" not in bare.to_json()
    document = bare.to_json()
    document["source_choice"] = {"schema": "something-else"}
    with pytest.raises(ContractError, match="source-choice"):
        Plan.from_json(document)


# --------------------------------------------------------------------------
# the planners' --source auto


def _parse(*argv):
    from woof.cli import build_parser

    return build_parser().parse_args(["energy", "plan", *argv])


def test_cli_help_names_the_default_priority(capsys):
    with pytest.raises(SystemExit):
        _parse("--help")
    text = re.sub(r"\s+", "", capsys.readouterr().out)
    assert ",".join(ss.DEFAULT_PRIORITY) in text


def test_source_priority_without_auto_is_refused(tmp_path):
    from woof.energy import plan_wrf_nests as nests
    from woof.energy import plan_wrf_tiles as tiles

    sites = load_sites(SITES_WALES)
    with pytest.raises(tiles.TilePlanRefusal, match="--source auto"):
        tiles.build_plan(sites, outdir=tmp_path, start="2026-01-15T00",
                         source="gfs", source_priority=("gfs",), now=NOW)
    with pytest.raises(nests.PlanRefused, match="--source auto"):
        nests.build_plan(sites, outdir=tmp_path, start="2026-01-15T00",
                         source="gfs", source_priority=("gfs",),
                         vram_gib=24.0, now=NOW)
    assert not list(tmp_path.iterdir())


def test_hex_swath_refuses_auto(tmp_path, capsys):
    args = _parse(str(SITES_WALES), "--topology", "hex-swath", "--source",
                  "auto", "-o", str(tmp_path))
    assert args.func(args) == 2
    assert "--source auto applies only" in capsys.readouterr().err
    args = _parse(str(SITES_WALES), "--topology", "hex-swath",
                  "--source-priority", "gfs", "-o", str(tmp_path))
    assert args.func(args) == 2


def test_auto_with_no_candidate_refuses_before_writing(tmp_path):
    from woof.energy import plan_wrf_tiles as tiles

    with pytest.raises(tiles.TilePlanRefusal, match="--source auto"):
        tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path,
                         start="2019-06-01T12", source="auto",
                         source_priority=("gfs", "met-office-global"),
                         now=NOW)
    assert not list(tmp_path.iterdir())


def test_tiles_auto_records_the_choice(tmp_path):
    from woof.energy import plan_wrf_tiles as tiles

    plan = tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path,
                            dx_m=1000.0, start="2026-01-15T00", hours=6,
                            source="auto", now=NOW)
    assert plan.source == "met-office-global"
    assert plan.source_choice["source"] == "met-office-global"
    receipt = json.loads((tmp_path / "source-choice.json").read_text())
    assert receipt == load_plan(tmp_path / "plan.json").source_choice
    assert receipt["region"]["margin_km"] > 0.0
    assert any(note.startswith("--source auto chose met-office-global")
               for note in plan.notes)
    import tomllib

    raw = tomllib.loads((tmp_path / "parent" / "parent.toml").read_text())
    assert raw["fetch"]["source"] == "met-office-global"
    # a named source re-plan drops the stale receipt
    tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path, dx_m=1000.0,
                     start="2026-01-15T00", hours=6, source="gfs", now=NOW)
    assert not (tmp_path / "source-choice.json").exists()
    assert load_plan(tmp_path / "plan.json").source_choice is None


def test_tiles_auto_with_a_defaulted_start_plans_the_cycle_it_chose(
        tmp_path):
    from woof.energy import plan_wrf_tiles as tiles

    plan = tiles.build_plan(load_sites(SITES_WALES), outdir=tmp_path,
                            dx_m=1000.0, hours=6, source="auto",
                            now=datetime(2026, 10, 14, 12, 5))
    assert plan.source == "met-office-global"
    assert plan.start == plan.source_choice["cycle"]
    assert plan.start < "2026-10-14T12"
    assert any(note.startswith("default start: searched")
               for note in plan.notes)


def test_nests_auto_records_the_choice(tmp_path):
    from woof.energy import plan_wrf_nests as nests

    plan = nests.build_plan(load_sites(SITES_WALES), outdir=tmp_path,
                            dx_m=1000.0, start="2025-11-01T00", hours=6,
                            source="auto", vram_gib=24.0, now=NOW,
                            source_priority=("met-office-global", "gfs"))
    assert plan.source == "gfs"
    receipt = json.loads((tmp_path / "source-choice.json").read_text())
    assert receipt["priority"] == ["met-office-global", "gfs"]
    assert receipt["candidates"][0]["status"] == "rejected"
    assert load_plan(tmp_path / "plan.json").source_choice == receipt


# --------------------------------------------------------------------------
# the guide


def _doc_commands() -> list[str]:
    found, inside = [], False
    for line in DOC.read_text("utf-8").splitlines():
        fence = line.strip()
        if fence.startswith("```"):
            inside = (not inside) and fence == "```bash"
            continue
        if inside and fence.startswith("woof energy "):
            found.append(fence)
    return found


def test_the_guide_commands_parse():
    commands = _doc_commands()
    assert commands
    for body in commands:
        _parse(*shlex.split(body)[3:])


def test_the_guide_lists_the_default_priority_and_checks():
    text = DOC.read_text("utf-8")
    assert ",".join(ss.DEFAULT_PRIORITY) in text
    for check in ss.CHECKS:
        assert f"`{check}`" in text, check
    assert ss.SCHEMA in text
