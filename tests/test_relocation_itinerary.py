"""CPU contracts for the track-following relocation itinerary.

``woof.relocation_itinerary`` turns a known track into ``[[relocation.move]]``
rows plus a ``[relocation.containment]`` slide.  What is asserted:

- path input: CSV and JSON with caller-named columns (and an optional
  height), the (0, 0) no-fix rows dropped, time order;
- the plan centres the nest on the first fix and, checked through an
  INDEPENDENT geometry path (``ProjectedGrid.nest``, the WPS
  parent_start/ratio arithmetic rather than the planner's own scaling),
  keeps every fix the stated margin inside the mover;
- rows land on the cadence lattice, never before departure, inside the run;
- refusals: a track too fast for the cadence names a cadence that works
  (and it does), a frame too small names the frame, an off-lattice cadence
  names the root step;
- the written config reloads through admission, and the moving-nest
  machinery reads it as a bounded itinerary with a containment slide;
- a path that never needs a move writes no [relocation] at all.

Synthetic paths and neutral coordinates only.
"""

from __future__ import annotations

import json
import math
import tomllib
from datetime import datetime, timedelta

import numpy as np
import pytest

from woof.experiment import load_experiment
from woof.relocation_itinerary import (TreeGeometry, clean_track,
                                       itinerary_from_track, load_track,
                                       write_config_with_plan)
from woof.static.projection import grids_from_projection_config

START = datetime(2025, 6, 1, 12, 0)
DEPART = START + timedelta(hours=3)
ORIGIN = (39.94, -100.25)

_TREE = """\
[experiment]
name = "track_tree"
start_time = 2025-06-01T12:00:00
run_seconds = {run_seconds}
restart_interval_s = 0.0

[projection]
map_proj = "lambert"
ref_lat = 40.3
ref_lon = -98.5
truelat1 = 30.0
truelat2 = 60.0
stand_lon = -98.5

[shared]
nz = 30
ztop = 20000.0
p_top = 5000.0
moist = true
mp_physics = 6
ra_lw_physics = 4
ra_sw_physics = 4
sf_sfclay_physics = 1
sf_surface_physics = 2
bl_pbl_physics = 1
num_soil_layers = 4

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = {nx}
ny = 200
time_step = 7
time_step_fract_num = 1
time_step_fract_den = 2
dx = 1500.0
history_interval_s = 600.0

[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 60    # replaced by the plan
j_parent_start = 50
parent_grid_ratio = 5
parent_time_step_ratio = 5
nx = 240
ny = 240
history_interval_s = 600.0

[[domain]]
grid_id = 3
parent_id = 2
i_parent_start = 60
j_parent_start = 60
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = 324
ny = 324
history_interval_s = 600.0
"""


def _tree(tmp_path, *, nx=400, run_seconds=21600.0):
    path = tmp_path / "tree.toml"
    path.write_text(_TREE.format(nx=nx, run_seconds=run_seconds),
                    encoding="utf-8")
    return path


def _track(speed_m_s, *, hours=2.0, heading_deg=70.0, step_s=10,
           depart=DEPART):
    """A straight path from its start at constant ground speed."""
    times, lats, lons = [], [], []
    east = math.sin(math.radians(heading_deg))
    north = math.cos(math.radians(heading_deg))
    for k in range(0, int(hours * 3600) + 1, step_s):
        d = speed_m_s * k
        times.append(depart + timedelta(seconds=k))
        lats.append(ORIGIN[0] + d * north / 111.2e3)
        lons.append(ORIGIN[1] + d * east
                    / (111.2e3 * math.cos(math.radians(ORIGIN[0]))))
    return times, lats, lons


def _ground_check(config, plan, times, lats, lons, margin):
    """Every fix inside the mover, through ProjectedGrid.nest.

    The mover's GROUND placement is the initial one plus every row before
    the fix (the containment slide compensates the mover, so it never
    moves it over the ground); d02 is held at its initial placement and
    d03 nested at the equivalent start, which the WPS arithmetic places
    identically whatever d02 is doing.
    """
    exp = load_experiment(config)
    frame = grids_from_projection_config(exp)[0]
    (ci, cj), (mi, mj) = plan.placements[2], plan.placements[3]
    container = frame.nest(ci, cj, 5, 241, 241)
    worst = math.inf
    for t, lat, lon in zip(times, lats, lons):
        seconds = (t - START).total_seconds()
        di = sum(m.di_parent_cells for m in plan.moves
                 if m.at_seconds <= seconds)
        dj = sum(m.dj_parent_cells for m in plan.moves
                 if m.at_seconds <= seconds)
        mover = container.nest(mi + di, mj + dj, 3, 325, 325)
        x, y = mover.latlon_to_ij(np.array([lat]), np.array([lon]))
        x, y = float(x[0]) - 0.5, float(y[0]) - 0.5
        kept = min(x, 324 - x, y, 324 - y)
        worst = min(worst, kept)
    assert worst >= margin - 1e-6, worst
    return worst


# ---------------------------------------------------------------------------
# Path input
# ---------------------------------------------------------------------------

def test_csv_columns_come_from_arguments(tmp_path):
    path = tmp_path / "path.csv"
    path.write_text(
        "stamp, y, x, z, note\n"
        "2025-06-01T15:00:10Z,40.01,-100.24,500,a\n"
        "2025-06-01T15:00:00Z,40.00,-100.25,400,b\n"
        "2025-06-01T15:00:20Z,0,0,0,c\n"
        "2025-06-01T15:00:30Z,,,,d\n"
        "2025-06-01T15:00:40Z,40.02,-100.23,600,e\n", encoding="utf-8")
    track = load_track(path, time_col="stamp", lat_col="y", lon_col="x",
                       height_col="z")
    assert track.dropped == 2
    assert [t.second for t in track.times] == [0, 10, 40]
    assert track.times[0].tzinfo is None
    assert track.lats.tolist() == [40.0, 40.01, 40.02]
    assert track.heights.tolist() == [400.0, 500.0, 600.0]
    # No guessing: the default names are not in this file, so it refuses
    # and names the flag that fixes it.
    with pytest.raises(ValueError, match="--time-col"):
        load_track(path)
    assert load_track(path, time_col="stamp", lat_col="y",
                      lon_col="x").heights is None


def test_json_path_with_offsets(tmp_path):
    rows = [{"t": "2025-06-01T15:00:00+00:00", "la": 40.0, "lo": 260.0},
            {"t": "2025-06-01T16:00:00+01:00", "la": 40.1, "lo": -100.0}]
    path = tmp_path / "path.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    track = load_track(path, time_col="t", lat_col="la", lon_col="lo")
    # +01:00 is the same instant as 15:00Z, so it is a duplicate.
    assert len(track) == 1 and track.dropped == 1
    assert track.lons.tolist() == [-100.0]


def test_path_without_any_fix_refuses():
    with pytest.raises(ValueError, match="no usable fix"):
        clean_track(["2025-06-01T15:00:00"], [0.0], [0.0])


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def test_plan_starts_on_the_first_fix_and_keeps_every_fix_inside(tmp_path):
    config = _tree(tmp_path)
    geom = TreeGeometry.from_config(config)
    times, lats, lons = _track(20.0)
    plan = itinerary_from_track(times, lats, lons, geom, START, 300.0, 40)
    # Centred on the first fix to within one d02 cell (300 m).
    exp = load_experiment(config)
    frame = grids_from_projection_config(exp)[0]
    (ci, cj), (mi, mj) = plan.placements[2], plan.placements[3]
    mover = frame.nest(ci, cj, 5, 241, 241).nest(mi, mj, 3, 325, 325)
    x, y = mover.latlon_to_ij(np.array([ORIGIN[0]]), np.array([ORIGIN[1]]))
    assert abs(float(x[0]) - 162.5) <= 3.0
    assert abs(float(y[0]) - 162.5) <= 3.0
    # d02 starts centred on d03 (the containment leg's own centring).
    assert abs(mi - ((240 - 108) // 2 + 1)) <= 2
    assert abs(mj - ((240 - 108) // 2 + 1)) <= 2
    worst = _ground_check(config, plan, times, lats, lons, 40)
    assert plan.worst_margin_cells >= 40
    assert worst >= 40
    # Rows on the cadence lattice, after departure, inside the run.
    depart_s = (DEPART - START).total_seconds()
    assert plan.moves
    for move in plan.moves:
        assert move.at_seconds % 300.0 == 0.0
        assert depart_s <= move.at_seconds < 21600.0
        assert max(abs(move.di_parent_cells),
                   abs(move.dj_parent_cells)) <= plan.max_move_parent_cells
    # A 20 m/s track moves about 20 d02 cells every 300 s window.
    assert 15 <= max(abs(m.di_parent_cells) for m in plan.moves) <= 25
    assert plan.containment_slides >= 1
    assert plan.worst_container_margin_cells >= 10


def test_plan_accepts_an_experiment_and_its_start(tmp_path):
    config = _tree(tmp_path)
    times, lats, lons = _track(5.0, hours=1.0)
    by_geom = itinerary_from_track(times, lats, lons,
                                   TreeGeometry.from_config(config),
                                   START, 600.0, 40)
    by_exp = itinerary_from_track(times, lats, lons,
                                  load_experiment(config), None, 600.0, 40)
    assert by_geom.to_toml() == by_exp.to_toml()


def test_min_move_skips_small_recentring(tmp_path):
    config = _tree(tmp_path)
    times, lats, lons = _track(3.0)
    geom = TreeGeometry.from_config(config)
    every = itinerary_from_track(times, lats, lons, geom, START, 300.0, 40)
    sparse = itinerary_from_track(times, lats, lons, geom, START, 300.0, 40,
                                  min_move_parent_cells=6)
    assert len(sparse.moves) < len(every.moves)
    _ground_check(config, sparse, times, lats, lons, 40)


def test_toml_round_trips_through_admission(tmp_path):
    config = _tree(tmp_path)
    times, lats, lons = _track(20.0)
    plan = itinerary_from_track(times, lats, lons,
                                TreeGeometry.from_config(config), START,
                                300.0, 40)
    table = tomllib.loads(plan.to_toml())["relocation"]
    assert table["grid_id"] == 3 and table["enabled"] is True
    assert table["containment"]["grid_id"] == 2
    assert len(table["move"]) == len(plan.moves)

    out = write_config_with_plan(config, plan, tmp_path / "moving.toml")
    text = out.read_text(encoding="utf-8")
    assert "# replaced by the plan" in text   # comments survive the patch
    exp = load_experiment(out)
    rel = exp.relocation
    assert rel.enabled and rel.grid_id == 3
    assert rel.containment.grid_id == 2
    assert [(m.at_seconds, m.di_parent_cells, m.dj_parent_cells)
            for m in rel.moves] == [
        (m.at_seconds, m.di_parent_cells, m.dj_parent_cells)
        for m in plan.moves]
    for gid in (2, 3):
        dc = exp.domain(gid)
        assert (dc.i_parent_start, dc.j_parent_start) == plan.placements[gid]

    # The moving-nest machinery reads it as a bounded, ground-counted
    # itinerary with a containment ancestor (what the corridor is sized to).
    from woof.core.nest_reach import mover_reach
    from woof.static.corridor import (config_declares_follow_source,
                                      moving_grid_ids)
    assert config_declares_follow_source(exp)
    assert moving_grid_ids(exp) == {2, 3}
    reach = mover_reach(exp, 3)
    assert reach.bounded and reach.ground
    assert reach.hi_i == sum(m.di_parent_cells for m in plan.moves
                             if m.di_parent_cells > 0)

    # The companion namelist.wps `woof go` binds carries the same start.
    assert not (tmp_path / "moving.namelist.wps").exists()

    # Rewriting an already-moving config replaces its block, not appends.
    again = write_config_with_plan(out, plan, tmp_path / "again.toml")
    assert again.read_text(encoding="utf-8").count("[relocation]") == 1


def test_companion_namelist_follows_the_placement(tmp_path):
    config = _tree(tmp_path)
    (tmp_path / "tree.namelist.wps").write_text(
        "&share\n max_dom = 3,\n/\n&geogrid\n"
        " parent_id         = 1, 1, 2,\n"
        " i_parent_start    = 1, 60, 60,\n"
        " j_parent_start    = 1, 50, 60,  ! from woof domain\n"
        " e_we              = 401, 241, 325,\n/\n", encoding="utf-8")
    times, lats, lons = _track(10.0)
    plan = itinerary_from_track(times, lats, lons,
                                TreeGeometry.from_config(config), START,
                                300.0, 40)
    write_config_with_plan(config, plan, tmp_path / "moving.toml")
    text = (tmp_path / "moving.namelist.wps").read_text(encoding="utf-8")
    (ci, cj), (mi, mj) = plan.placements[2], plan.placements[3]
    assert f" i_parent_start    = 1, {ci}, {mi}," in text
    assert f" j_parent_start    = 1, {cj}, {mj},  ! from woof domain" in text
    assert " e_we              = 401, 241, 325," in text


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_too_fast_for_the_cadence_names_one_that_works(tmp_path):
    config = _tree(tmp_path)
    geom = TreeGeometry.from_config(config)
    times, lats, lons = _track(45.0, hours=1.0, heading_deg=90.0)
    with pytest.raises(ValueError, match="cannot be followed at a 900 s") \
            as err:
        itinerary_from_track(times, lats, lons, geom, START, 900.0, 40)
    message = str(err.value)
    assert "--cadence" in message
    better = float(message.split("--cadence ")[1].split(".")[0].split()[0])
    assert better < 900.0
    plan = itinerary_from_track(times, lats, lons, geom, START, better, 40)
    _ground_check(config, plan, times, lats, lons, 40)


def test_frame_too_small_names_the_frame(tmp_path):
    config = _tree(tmp_path, nx=200)
    times, lats, lons = _track(30.0, hours=2.0, heading_deg=90.0)
    with pytest.raises(ValueError, match="Enlarge d01"):
        itinerary_from_track(times, lats, lons,
                             TreeGeometry.from_config(config), START,
                             300.0, 40)


def test_off_lattice_cadence_refuses(tmp_path):
    config = _tree(tmp_path)
    times, lats, lons = _track(5.0, hours=0.5)
    with pytest.raises(ValueError, match="whole number of root steps"):
        itinerary_from_track(times, lats, lons,
                             TreeGeometry.from_config(config), START,
                             301.0, 40)


def test_track_after_the_run_refuses(tmp_path):
    config = _tree(tmp_path, run_seconds=3600.0)
    times, lats, lons = _track(5.0, hours=0.5)   # departs at +3 h
    with pytest.raises(ValueError, match="past the end of the run"):
        itinerary_from_track(times, lats, lons,
                             TreeGeometry.from_config(config), START,
                             300.0, 40)


def test_capped_rows_that_lose_the_track_refuse(tmp_path):
    config = _tree(tmp_path)
    times, lats, lons = _track(20.0)
    with pytest.raises(ValueError, match="cannot be followed"):
        itinerary_from_track(times, lats, lons,
                             TreeGeometry.from_config(config), START,
                             300.0, 40, max_move_parent_cells=5)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_from_track_writes_snippet_and_config(tmp_path, capsys):
    from woof.cli import main

    config = _tree(tmp_path)
    times, lats, lons = _track(10.0)
    track = tmp_path / "path.csv"
    track.write_text("when,phi,lam,h\n" + "".join(
        f"{t.isoformat()}Z,{lat},{lon},{100 + 5 * k}\n"
        for k, (t, lat, lon) in enumerate(zip(times, lats, lons))),
        encoding="utf-8")
    snippet = tmp_path / "relocation.toml"
    moving = tmp_path / "moving.toml"
    code = main(["relocation", "from-track", str(track), str(config),
                 "--time-col", "when", "--lat-col", "phi", "--lon-col", "lam",
                 "--height-col", "h",
                 "--cadence", "300", "--out", str(snippet),
                 "--write-config", str(moving), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["relocation"]["grid_id"] == 3
    assert payload["config"] == str(moving)
    assert payload["height_range_m"][0] == 100.0
    assert tomllib.loads(snippet.read_text(encoding="utf-8"))[
        "relocation"]["containment"]["grid_id"] == 2
    assert load_experiment(moving).relocation.moves


def test_path_that_never_needs_a_move_writes_no_relocation(tmp_path):
    config = _tree(tmp_path)
    times, lats, lons = _track(0.2, hours=0.5)
    plan = itinerary_from_track(times, lats, lons,
                                TreeGeometry.from_config(config), START,
                                300.0, 40, min_move_parent_cells=5)
    assert plan.moves == () and plan.relocation_table() == {}
    assert "[relocation]" not in plan.to_toml().replace(
        "no [relocation] is needed", "")
    out = write_config_with_plan(config, plan, tmp_path / "still.toml")
    exp = load_experiment(out)
    assert not exp.relocation.enabled
    assert (exp.domain(3).i_parent_start,
            exp.domain(3).j_parent_start) == plan.placements[3]


def test_writer_headers_comments_and_directory(tmp_path):
    config = _tree(tmp_path)
    text = config.read_text(encoding="utf-8").replace(
        "[[domain]]\ngrid_id = 3", "[[domain]]  # the mover\ngrid_id = 3")
    text += ("\n[relocation]  # left over\nenabled = true\ngrid_id = 3\n"
             "max_move_parent_cells = 4\n")
    config.write_text(text, encoding="utf-8")
    times, lats, lons = _track(10.0)
    plan = itinerary_from_track(times, lats, lons,
                                TreeGeometry.from_config(config), START,
                                300.0, 40)
    out = write_config_with_plan(config, plan, tmp_path / "moving.toml")
    written = out.read_text(encoding="utf-8")
    assert "left over" not in written
    assert tomllib.loads(written)["relocation"]["containment"]["grid_id"] == 2
    assert load_experiment(out).domain(3).i_parent_start == \
        plan.placements[3][0]
    elsewhere = tmp_path / "sub"
    elsewhere.mkdir()
    with pytest.raises(ValueError, match="relative paths"):
        write_config_with_plan(config, plan, elsewhere / "moving.toml")


# ---------------------------------------------------------------------------
# The HRRR hierarchy sizes its corridor from the config, not the namelists
# ---------------------------------------------------------------------------

def test_hrrr_corridor_is_sized_from_the_config(tmp_path):
    """The namelist-derived tree carries no [relocation]; the corridor
    must read the config's, or the runner refuses the bundle."""
    from woof.hrrr_hierarchy_direct import _corridor_experiment
    from woof.static.corridor import corridor_reach

    config = _tree(tmp_path)
    times, lats, lons = _track(20.0)
    plan = itinerary_from_track(times, lats, lons,
                                TreeGeometry.from_config(config), START,
                                300.0, 40)
    moving = write_config_with_plan(config, plan, tmp_path / "moving.toml")
    declared = load_experiment(moving)
    # The namelist view: same tree, same placement, no relocation.
    plain_path = tmp_path / "plain.toml"
    plain_path.write_text(
        moving.read_text(encoding="utf-8").split("[relocation]")[0]
        .rsplit("# Track itinerary", 1)[0], encoding="utf-8")
    native = load_experiment(plain_path)
    assert not native.relocation.enabled
    assert _corridor_experiment(native, None) is native

    sized = _corridor_experiment(native, moving)
    for gid in (2, 3):
        want = corridor_reach(declared, declared.domain(gid))
        got = corridor_reach(sized, sized.domain(gid))
        assert got == want
        assert got != corridor_reach(native, native.domain(gid))

    # A config that places the tree differently is refused, not sealed.
    shifted = tmp_path / "shifted.toml"
    shifted.write_text(moving.read_text(encoding="utf-8").replace(
        f"i_parent_start = {plan.placements[2][0]}",
        f"i_parent_start = {plan.placements[2][0] + 1}", 1),
        encoding="utf-8")
    with pytest.raises(ValueError, match="does not declare d02"):
        _corridor_experiment(native, shifted)
