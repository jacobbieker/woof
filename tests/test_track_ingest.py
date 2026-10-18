"""3-D path ingestion and the Track3D model (woof.track).

Every track here is synthetic: tests/fixtures/track/track_synthetic.csv
was generated from a made-up balloon profile with one of each glitch the
reader removes, and the rest are written into tmp_path by
:func:`_write_track` from :func:`_profile`.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import pytest

from woof.cli import main as cli_main
from woof.track import ColumnMapping, Track3D, boundary_source, load_track
from woof.track.model import HISTOGRAM_BIN_M, track_from_fixes


FIXTURE = Path(__file__).parent / "fixtures" / "track" / "track_synthetic.csv"
#: The fixture's own column names, deliberately not the defaults.
FIXTURE_MAPPING = ColumnMapping(time="timestamp", lat="latitude", lon="longitude",
                                height="alt_gps_m", height_fallback=("alt_baro_m",),
                                drop_when=(("fix_ok", "0"),))
FIXTURE_ARGS = ["--time-col", "timestamp", "--lat-col", "latitude",
                "--lon-col", "longitude", "--height-col", "alt_gps_m",
                "--height-fallback-col", "alt_baro_m", "--drop-when", "fix_ok=0"]
T0 = datetime(2030, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
COLUMNS = ("time", "lat", "lon", "height", "height2", "valid")


def _profile(*, ground=1000.0, peak=20000.0, land_ground=None, climb=5.0,
             descent=10.0, drift=10.0, lat0=35.0, lon0=-106.0, pre=600.0,
             post=600.0):
    """``(state, end, events)``: ``state(s) -> (lat, lon, height)`` for an
    object launched at ``pre`` seconds, the record's natural last second,
    and the true launch/burst/landing seconds."""

    land_ground = ground if land_ground is None else land_ground
    burst = pre + (peak - ground) / climb
    land = burst + (peak - land_ground) / descent
    metres_per_degree = 111195.0 * math.cos(math.radians(lat0))

    def state(s):
        if s <= pre:
            return lat0, lon0, ground
        moved = drift * (min(s, land) - pre)
        if s <= burst:
            height = ground + climb * (s - pre)
        elif s <= land:
            height = peak - descent * (s - burst)
        else:
            height = land_ground
        return lat0, lon0 + moved / metres_per_degree, height

    return state, land + post, {"launch": pre, "burst": burst, "land": land}


def _row(s, lat, lon, height):
    row = {"time": (T0 + timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "lat": f"{lat:.6f}", "lon": f"{lon:.6f}", "height": f"{height:.1f}",
           "height2": f"{height - 15.0:.2f}", "valid": "1"}
    return row


def _rows(state, end, *, step=60.0, start=0.0):
    rows, s = [], start
    while s <= end + 1e-9:
        rows.append(_row(s, *state(s)))
        s += step
    return rows


def _write_track(path, state, end, *, step=60.0, extra=(), start=0.0):
    rows = _rows(state, end, step=step, start=start) + list(extra)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _seconds(moment):
    return (moment - T0).total_seconds()


# -- the checked-in fixture ----------------------------------------------


def test_the_fixture_is_cleaned_of_every_glitch_it_carries():
    track = load_track(FIXTURE, FIXTURE_MAPPING)

    assert track.name == "track_synthetic"
    for reason in ("no_position", "null_island", "dropped_by_rule", "position_outlier",
                   "height_spike", "duplicate_time", "height_from_fallback"):
        assert track.qc[reason] == 1, reason
    assert track.qc["reordered"] >= 1
    assert track.qc["rows"] == 121
    assert track.qc["fixes"] == len(track.times)
    # The sign-glitched longitude is gone: the track drifts ~55 km east,
    # not across the globe.
    assert 40.0 < track.extent_ew_km < 70.0
    assert all(lon < -105.0 for lon in track.lon)
    assert track.max_height < 21000.0
    assert list(track.times) == sorted(set(track.times))


def test_the_fixture_launch_landing_and_derived_quantities():
    track = load_track(FIXTURE, FIXTURE_MAPPING)
    launch = datetime(2030, 6, 1, 12, 10, 0, tzinfo=timezone.utc)
    landing = datetime(2030, 6, 1, 13, 42, 10, tzinfo=timezone.utc)

    assert abs((track.launch.time - launch).total_seconds()) <= 60.0
    assert abs((track.landing.time - landing).total_seconds()) <= 60.0
    assert track.launch.height == pytest.approx(1600.0)
    assert track.landing.height == pytest.approx(1500.0)
    assert track.launch_observed and track.landing_observed
    assert track.peak_class == "high"
    assert track.height_datum == "msl"
    assert track.duration_s == pytest.approx(
        (track.landing.time - track.launch.time).total_seconds(), abs=1e-3)
    assert sum(track.height_hist_seconds) == pytest.approx(track.duration_s, abs=0.01)
    assert track.height_hist_bin_m == HISTOGRAM_BIN_M
    assert track.boundary_source == "hrrr"
    south, west, north, east = track.envelope
    assert south <= track.launch.lat <= north and west <= track.launch.lon <= east


def test_the_default_mapping_names_the_columns_it_could_not_find():
    with pytest.raises(ValueError, match="--time-col"):
        load_track(FIXTURE)


def test_a_track_survives_a_json_round_trip_exactly(tmp_path):
    # Launch and landing land between whole seconds (5 m/s from a ground
    # fix 30 s before the first airborne one), the case a coarser time
    # format would round.
    state, end, _ = _profile(pre=613.3337)
    track = load_track(_write_track(tmp_path / "t.csv", state, end, step=30.0))
    assert track.launch.time.microsecond != 0
    assert Track3D.from_json(track.to_json()) == track
    document = json.loads(track.to_json())
    assert document["launch"]["time"].endswith("Z")
    assert document["schema"] == "woof-track3d-v1"


def test_a_document_from_another_schema_is_refused():
    document = load_track(FIXTURE, FIXTURE_MAPPING).to_dict()
    document["schema"] = "something-else"
    with pytest.raises(ValueError, match="schema"):
        Track3D.from_dict(document)
    document["schema"] = "woof-track3d-v1"
    document["boundary_source"] = "elsewhere"
    with pytest.raises(ValueError, match="boundary_source"):
        Track3D.from_dict(document)


# -- readers and the column mapping ---------------------------------------


def test_json_and_parquet_read_the_same_track_as_csv(tmp_path):
    state, end, _ = _profile()
    rows = _rows(state, end)
    expected = load_track(_write_track(tmp_path / "t.csv", state, end), name="t")

    (tmp_path / "rows.json").write_text(json.dumps(rows))
    assert load_track(tmp_path / "rows.json", name="t").times == expected.times
    columns = {key: [row[key] for row in rows] for key in COLUMNS}
    (tmp_path / "columns.json").write_text(json.dumps({"records": rows}))
    assert load_track(tmp_path / "columns.json", name="t").height == expected.height
    (tmp_path / "cols.json").write_text(json.dumps(columns))
    assert load_track(tmp_path / "cols.json", name="t").lat == expected.lat

    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet
    table = pyarrow.table({
        "time": [T0 + timedelta(seconds=60.0 * k) for k in range(len(rows))],
        "lat": [float(row["lat"]) for row in rows],
        "lon": [float(row["lon"]) for row in rows],
        "height": [float(row["height"]) for row in rows]})
    pyarrow.parquet.write_table(table, tmp_path / "t.parquet")
    assert load_track(tmp_path / "t.parquet", name="t").height == expected.height


def test_epoch_and_strptime_times(tmp_path):
    state, end, _ = _profile()
    rows = _rows(state, end)
    epoch = [{**row, "time": str((datetime.fromisoformat(row["time"].replace("Z", "+00:00"))
                                  - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds())}
             for row in rows]
    (tmp_path / "epoch.json").write_text(json.dumps(epoch))
    track = load_track(tmp_path / "epoch.json", ColumnMapping(time_format="epoch"))
    assert track.times[0] == T0

    custom = [{**row, "time": datetime.fromisoformat(row["time"].replace("Z", "+00:00"))
               .strftime("%d/%m/%Y %H.%M.%S")} for row in rows]
    (tmp_path / "custom.json").write_text(json.dumps(custom))
    track = load_track(tmp_path / "custom.json",
                       ColumnMapping(time_format="%d/%m/%Y %H.%M.%S"))
    assert track.times[0] == T0


def test_a_mapping_file_and_its_validation(tmp_path):
    (tmp_path / "map.toml").write_text(
        'time = "timestamp"\nlat = "latitude"\nlon = "longitude"\n'
        'height = "alt_gps_m"\nheight_fallback = "alt_baro_m"\n'
        'drop_when = { fix_ok = "0" }\n')
    mapping = ColumnMapping.from_file(tmp_path / "map.toml")
    assert mapping == FIXTURE_MAPPING
    (tmp_path / "bad.json").write_text(json.dumps({"hieght": "h"}))
    with pytest.raises(ValueError, match="hieght"):
        ColumnMapping.from_file(tmp_path / "bad.json")
    with pytest.raises(ValueError, match="datum"):
        ColumnMapping(height_datum="wgs84")


def test_an_agl_track_keeps_its_datum(tmp_path):
    state, end, _ = _profile(ground=0.0)
    track = load_track(_write_track(tmp_path / "t.csv", state, end),
                       ColumnMapping(height_datum="agl"))
    assert track.height_datum == "agl"
    assert track.launch.height == pytest.approx(0.0)


def test_the_height_fallback_stands_in_only_where_the_height_is_blank(tmp_path):
    state, end, _ = _profile()
    lat, lon, height = state(1200.0)
    row = _row(1230.0, lat, lon, height) | {"height": "", "height2": f"{height:.2f}"}
    track = load_track(_write_track(tmp_path / "t.csv", state, end, extra=[row]),
                       ColumnMapping(height_fallback=("height2",)))
    assert track.qc["height_from_fallback"] == 1
    assert 1230.0 in [_seconds(t) for t in track.times]


def test_an_unknown_file_type_is_refused(tmp_path):
    (tmp_path / "t.xlsx").write_bytes(b"")
    with pytest.raises(ValueError, match="not a file type"):
        load_track(tmp_path / "t.xlsx")


def test_a_name_with_a_path_separator_is_refused(tmp_path):
    state, end, _ = _profile()
    path = _write_track(tmp_path / "t.csv", state, end)
    with pytest.raises(ValueError, match="cannot name a file"):
        load_track(path, name="../escape")


# -- launch, landing and the climb rule ------------------------------------


def test_launch_is_placed_inside_its_gap_by_the_measured_climb(tmp_path):
    state, end, events = _profile(pre=700.0)
    track = load_track(_write_track(tmp_path / "t.csv", state, end, step=300.0))
    assert track.launch.window_s == 300.0
    assert _seconds(track.launch.time) == pytest.approx(events["launch"], abs=1.0)


def test_a_record_that_starts_climbing_has_no_observed_launch(tmp_path):
    state, end, events = _profile()
    track = load_track(_write_track(tmp_path / "t.csv", state, end,
                                    start=events["launch"] + 600.0))
    assert not track.launch_observed
    assert track.launch.time == track.times[0]
    assert track.launch.window_s == 0.0


def test_one_low_glitch_at_the_start_does_not_move_the_ground(tmp_path):
    state, end, events = _profile(ground=1600.0)
    glitch = _row(30.0, *state(30.0)[:2], 1520.0)
    track = load_track(_write_track(tmp_path / "t.csv", state, end, extra=[glitch]))
    assert track.launch_observed
    assert track.launch.height == pytest.approx(1600.0)
    assert abs(_seconds(track.launch.time) - events["launch"]) <= 60.0


def test_a_record_that_ends_descending_has_no_observed_landing(tmp_path):
    state, _, events = _profile()
    track = load_track(_write_track(tmp_path / "t.csv", state, events["land"] - 600.0))
    assert not track.landing_observed
    assert track.landing.window_s == 0.0
    assert track.landing.time == track.times[-1]
    assert track.landing.height == track.height[-1]


def test_a_balloon_floating_out_of_contact_is_not_landed(tmp_path):
    state, _, events = _profile(peak=18000.0, descent=1e-9)
    track = load_track(_write_track(tmp_path / "t.csv", state, events["burst"] + 1800.0))
    assert not track.landing_observed


def test_a_landed_payload_dragged_by_its_parachute_lands_at_first_touch(tmp_path):
    state, end, events = _profile()
    land = events["land"]

    def dragged(s):
        lat, lon, height = state(s)
        if s > land:
            lon += min(s - land, 600.0) / 600.0 * 0.0066   # ~600 m of drag
        return lat, lon, height

    track = load_track(_write_track(tmp_path / "t.csv", dragged, end))
    assert track.landing_observed
    assert abs(_seconds(track.landing.time) - land) <= 60.0


def test_ground_jitter_at_one_fix_a_second_still_reads_as_landed():
    state, _, events = _profile()
    seconds = [float(s) for s in range(0, int(events["land"]) + 600)]
    times = [T0 + timedelta(seconds=s) for s in seconds]
    lat, lon, height = zip(*(state(s) for s in seconds))
    # Deterministic +/-3 m scatter on every fix.
    height = [h + 3.0 * math.sin(1.7 * k) for k, h in enumerate(height)]
    track = track_from_fixes("jitter", times, lat, lon, height)
    assert track.landing_observed
    assert abs(_seconds(track.landing.time) - events["land"]) <= 30.0


def test_a_single_fix_after_a_silence_shows_the_descent_stopped(tmp_path):
    state, _, events = _profile()
    rows = _rows(state, events["land"] - 120.0)
    lat, lon, height = state(events["land"] + 900.0)
    rows.append(_row(events["land"] + 900.0, lat, lon, height))
    with open(tmp_path / "t.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    track = load_track(tmp_path / "t.csv")
    assert track.landing_observed
    assert abs(_seconds(track.landing.time) - events["land"]) <= 60.0


def test_low_and_unknown_peaks(tmp_path):
    state, end, _ = _profile(peak=2500.0, climb=3.0, descent=4.0)
    low = load_track(_write_track(tmp_path / "low.csv", state, end))
    assert low.peak_class == "low" and low.landing_observed

    state, _, events = _profile()
    unknown = load_track(_write_track(tmp_path / "unk.csv", state, events["launch"] + 500.0))
    assert unknown.peak_class == "unknown" and not unknown.landing_observed


def test_the_climb_threshold_is_configurable(tmp_path):
    state, end, _ = _profile(peak=1250.0, climb=1.0)
    path = _write_track(tmp_path / "t.csv", state, end)
    with pytest.raises(ValueError, match="300 m"):
        load_track(path)
    assert load_track(path, min_climb_m=200.0).max_height == pytest.approx(1250.0, abs=20.0)


def test_a_climb_without_drift_is_refused(tmp_path):
    state, end, _ = _profile(peak=3000.0, drift=0.0)
    with pytest.raises(ValueError, match="horizontal"):
        load_track(_write_track(tmp_path / "t.csv", state, end))


def test_a_glitch_at_the_end_of_the_record_goes_and_its_neighbour_stays(tmp_path):
    state, end, _ = _profile()
    lat, lon, height = state(end)
    glitch = _row(end + 30.0, lat, lon + 3.0, height)
    track = load_track(_write_track(tmp_path / "t.csv", state, end, extra=[glitch]))
    assert track.qc["position_spike"] == 1
    assert _seconds(track.times[-1]) == pytest.approx(end)


# -- boundary source --------------------------------------------------------


@pytest.mark.parametrize(("lat", "lon", "expected"), [
    ((40.0, 40.5), (-100.0, -98.0), "hrrr"),
    ((45.0, 45.5), (-110.0, -104.0), "hrrr"),
    ((51.0, 51.5), (-1.5, -0.5), "ukv"),
    ((55.0, 55.5), (-4.5, 1.0), "ukv"),
    ((-35.0, -34.5), (150.0, 151.0), "era5"),
    ((61.0, 61.3), (-150.0, -149.0), "era5"),
])
def test_boundary_source_follows_the_track(lat, lon, expected):
    assert boundary_source(lat, lon) == expected


def test_the_boundary_margin_keeps_a_track_off_the_hrrr_edge():
    lat, lon = (21.6, 21.7), (-122.3, -122.2)
    assert boundary_source(lat, lon, margin_km=0.0) == "hrrr"
    assert boundary_source(lat, lon) == "era5"


# -- `woof track inspect` ---------------------------------------------------


def _directory(root):
    root.mkdir()
    for name, lon0 in (("alpha", -106.0), ("bravo", -105.0)):
        state, end, _ = _profile(lon0=lon0)
        _write_track(root / f"{name}.csv", state, end)
    return root


def test_inspect_prints_the_table_and_writes_json(tmp_path, capsys):
    directory = _directory(tmp_path / "tracks")
    out = tmp_path / "out"
    assert cli_main(["track", "inspect", str(directory), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "alpha" in printed and "bravo" in printed
    assert "2 track(s); boundary source: hrrr 2" in printed
    assert sorted(path.name for path in out.iterdir()) == ["alpha.json", "bravo.json"]
    track = Track3D.from_json((out / "alpha.json").read_text())
    assert track == load_track(directory / "alpha.csv")


def test_inspect_reads_the_fixture_through_column_options(tmp_path, capsys):
    out = tmp_path / "out"
    assert cli_main(["track", "inspect", str(FIXTURE), *FIXTURE_ARGS,
                     "--name", "demo", "--json", str(out), "--explain"]) == 0
    printed = capsys.readouterr().out
    assert "demo" in printed and "rows kept" in printed
    assert Track3D.from_json((out / "demo.json").read_text()) == load_track(
        FIXTURE, FIXTURE_MAPPING, name="demo")


def test_inspect_options_override_the_mapping_file(tmp_path, capsys):
    (tmp_path / "map.json").write_text(json.dumps({
        "time": "timestamp", "lat": "latitude", "lon": "longitude",
        "height": "no-such-column", "height_fallback": ["alt_baro_m"],
        "drop_when": ["fix_ok=0"]}))
    assert cli_main(["track", "inspect", str(FIXTURE), "--mapping",
                     str(tmp_path / "map.json"), "--height-col", "alt_gps_m"]) == 0
    assert "track_synthetic" in capsys.readouterr().out


def test_one_bad_file_is_named_and_the_others_still_print(tmp_path, capsys):
    directory = _directory(tmp_path / "tracks")
    (directory / "charlie.csv").write_text("time,lat,lon\n")
    assert cli_main(["track", "inspect", str(directory)]) == 2
    captured = capsys.readouterr()
    assert "charlie.csv" in captured.err and "height" in captured.err
    assert "alpha" in captured.out and "bravo" in captured.out


def test_name_is_refused_for_several_files(tmp_path, capsys):
    directory = _directory(tmp_path / "tracks")
    assert cli_main(["track", "inspect", str(directory), "--name", "x"]) == 2
    assert "--name names one track" in capsys.readouterr().err


# -- review regressions -----------------------------------------------------


def test_an_aircraft_at_cruise_speed_keeps_every_fix(tmp_path):
    state, end, _ = _profile(peak=11000.0, climb=10.0, descent=10.0, drift=250.0)
    track = load_track(_write_track(tmp_path / "t.csv", state, end, step=10.0))
    assert "position_spike" not in track.qc
    assert track.extent_ew_km > 400.0


def test_a_long_dwell_does_not_cut_off_a_long_journey(tmp_path):
    # Two hours on the ground, then a 2,500 km drift: a median taken over
    # the whole track would sit at the launch site.
    state, end, _ = _profile(pre=7200.0, drift=100.0, peak=20000.0, climb=1.5,
                             descent=1.0)
    track = load_track(_write_track(tmp_path / "t.csv", state, end))
    assert "position_outlier" not in track.qc
    assert track.extent_ew_km > 2000.0


def test_a_drop_rule_matches_a_numeric_flag(tmp_path):
    state, end, _ = _profile()
    rows = _rows(state, end)
    rows.append(_row(1230.0, 36.0, -106.0, 9000.0) | {"valid": 0.0})
    (tmp_path / "t.json").write_text(json.dumps(rows))
    track = load_track(tmp_path / "t.json", ColumnMapping(drop_when=(("valid", "0"),)))
    assert track.qc["dropped_by_rule"] == 1


def test_a_low_flight_landing_on_higher_ground_is_observed(tmp_path):
    state, end, _ = _profile(ground=0.0, peak=350.0, land_ground=60.0, climb=2.0,
                             descent=2.0)
    track = load_track(_write_track(tmp_path / "t.csv", state, end, step=20.0))
    assert track.landing_observed
    assert track.landing.height == pytest.approx(60.0)


def test_a_track_across_the_antimeridian_is_refused(tmp_path):
    state, end, _ = _profile(lon0=179.95)
    rows = [row | {"lon": f"{(float(row['lon']) + 180.0) % 360.0 - 180.0:.6f}"}
            for row in _rows(state, end)]
    (tmp_path / "t.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="antimeridian"):
        load_track(tmp_path / "t.json")


def test_a_missing_file_does_not_stop_the_others(tmp_path, capsys):
    directory = _directory(tmp_path / "tracks")
    assert cli_main(["track", "inspect", str(directory / "alpha.csv"),
                     str(directory / "missing.csv"), str(directory / "bravo.csv")]) == 2
    captured = capsys.readouterr()
    assert "missing.csv" in captured.err
    assert "alpha" in captured.out and "bravo" in captured.out
