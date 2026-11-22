"""WIS2 archive reader (woof.obs.wis2) on synthetic, in-memory stores.

Nothing is fetched: each test builds a mapping laid out like
``bkr/obs/wis2.icechunk`` (group path -> arrays with the store's attributes)
with a handful of made-up reports. The values are not observations.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from woof.obs import wis2
from woof.obs.obserr_wrfda import ObsErrTable

NS = {"units": "nanoseconds since 1970-01-01", "_FillValue": np.iinfo(np.int64).min}
SEC = {"units": "seconds since 1970-01-01 00:00:00", "_FillValue": np.iinfo(np.int64).min}
BOX = (30.0, 50.0, -110.0, -80.0)
START, END = "2008-02-05T11:00:00", "2008-02-05T13:00:00"


class Arr:
    """An array with attributes, sliceable like a zarr array."""

    def __init__(self, values, **attrs):
        self.values = np.asarray(values)
        self.attrs = attrs
        self.shape = self.values.shape

    def __getitem__(self, key):
        return self.values[key]


def _t(text, attrs=NS):
    seconds = np.datetime64(text, "s").astype(np.int64)
    return seconds * (1_000_000_000 if attrs is NS else 1)


def _times(texts, attrs=NS):
    return Arr([_t(x, attrs) for x in texts], **attrs)


def _flags(values):
    return Arr(np.asarray(values, np.uint16))


def _dense_levels(bottom, top, ratio=1.1):
    """Pressures from ``bottom`` to ``top`` (Pa), consecutive levels ``ratio`` apart."""
    return bottom / ratio ** np.arange(int(np.log(bottom / top) / np.log(ratio)) + 1)


def _t_of(p):
    """A smooth made-up temperature profile (K): a troposphere and a warming stratosphere."""
    p = np.asarray(p, float)
    return np.maximum(288.0 * (p / 1e5) ** 0.19, 207.0) + np.where(p < 10000.0, (10000.0 - p) / 1000.0, 0.0)


# ----------------------------------------------------------------------------- QC primitives
def test_part_label_from_object_name_or_heading():
    assert wis2.part_label("data/us-noaa-nws/x/temp/TEMP_DD_72340_abc.bufr4") == "DD"
    assert wis2.part_label("", "TTBB 72340") == "BB"
    assert wis2.part_label("", "TTDD72 KWBC 281200") == "DD"  # a WMO heading with its sequence number
    assert wis2.part_label("", "TTDDX KWBC") == ""
    assert wis2.part_label("data/de-dwd-gts-to-wis2/x/sounding.bufr4") == ""


def test_pressure_scale_fault_label_rule_and_signature():
    # A Part D (above 100 hPa) with levels at 984 and 500 "hPa": refused by its own label.
    assert wis2.pressure_scale_fault("DD", [98400.0, 50000.0], [208.0, 209.0]) == "part_above_100hpa"
    assert wis2.pressure_scale_fault("DD", [9840.0, 5000.0], [208.0, 209.0]) == ""
    # An unlabelled part stored at ten times its pressure: detected against the launch's other parts.
    ref_p = _dense_levels(100000.0, 3000.0)
    ref_t = _t_of(ref_p)
    true_p = np.array([9500.0, 8500.0, 7500.0, 6000.0, 4000.0])
    true_t = _t_of(true_p)
    assert wis2.pressure_scale_fault("", true_p * 10.0, true_t, ref_p, ref_t) == "pressure_scale_x10"
    assert wis2.pressure_scale_fault("", true_p, true_t, ref_p, ref_t) == ""
    assert wis2.pressure_scale_fault("", true_p, true_t + 25.0, ref_p, ref_t) == "inconsistent_part"
    # A whole sounding at ten times its pressure has nothing to compare with: refused outright.
    assert wis2.pressure_scale_fault("AA", [9.84e5, 9.25e5], [290.0, 288.0]) == "pressure_out_of_range"
    assert wis2.pressure_scale_fault("", [9.84e5, 9.25e5], [290.0, 288.0]) == "pressure_out_of_range"
    # Too few overlapping levels: no verdict.
    assert wis2.pressure_scale_fault("", true_p[:2] * 10.0, true_t[:2], ref_p, ref_t) == ""


def test_interpolation_refuses_wide_gaps():
    out = wis2._interp_lnp(np.array([10000.0, 1000.0]), np.array([206.0, 225.0]), np.array([5000.0, 10020.0]))
    assert np.isnan(out[0]) and out[1] == 206.0


def test_rh_from_dewpoint_is_obsproc_formula():
    rh = wis2.rh_from_dewpoint(290.0, 280.0)
    assert rh == pytest.approx(100.0 * np.exp(5418.12 * (1.0 / 290.0 - 1.0 / 280.0)), rel=1e-15)
    assert wis2.rh_from_dewpoint(280.0, 281.0) == 100.0


# ----------------------------------------------------------------------------- soundings
def _soundings():
    """One 12Z launch at 72340: AA (good), BB (good) and DD with ten-times pressures; plus an
    unlabelled BUFR report of another station whose upper part is stored at ten times its pressure."""
    full_p, upper_p = _dense_levels(97000.0, 3000.0), np.array([9500.0, 9000.0, 8000.0, 6000.0, 4000.0])
    parts = [
        ("72340", "TEMP_AA_72340_a.bufr4",
         [(98400, 290.0, 285.0, 191.0, 5.0, 180.0, wis2.VSS_SURFACE),
          (92500, 290.35, 280.35, 860.0, 8.0, 190.0, wis2.VSS_STANDARD),
          (85000, 284.35, 278.35, 1574.0, 10.0, 200.0, wis2.VSS_STANDARD),
          (70000, 278.35, 271.35, 3184.0, 12.0, 210.0, wis2.VSS_STANDARD),
          (50000, 260.75, 240.75, 5840.0, 20.0, 220.0, wis2.VSS_STANDARD),
          (30000, 234.55, 218.55, 9530.0, 30.0, 230.0, wis2.VSS_STANDARD),
          (20000, 218.0, 200.0, 12000.0, 35.0, 240.0, wis2.VSS_STANDARD),
          (10000, 208.35, 191.35, 16460.0, 25.0, 250.0, wis2.VSS_STANDARD)]),
        ("72340", "TEMP_BB_72340_b.bufr4",
         [(88100, 286.0, 280.0, np.nan, np.nan, np.nan, 8192),
          (64500, 274.0, 262.0, np.nan, np.nan, np.nan, 8192)]),
        ("72340", "TEMP_DD_72340_c.bufr4",
         [(98400, 208.1, 180.0, np.nan, np.nan, np.nan, 8192),
          (50000, 208.9, 180.0, np.nan, np.nan, np.nan, 8192),
          (6700, 225.6, 180.0, np.nan, np.nan, np.nan, 8192)]),
        ("72649", "data/x/sounding_72649.bufr4",
         [(p, t, t - 8.0, np.nan, np.nan, np.nan, wis2.VSS_SURFACE if k == 0 else 8192)
          for k, (p, t) in enumerate(zip(full_p, _t_of(full_p)))]),
        ("72649", "data/x/sounding_72649_upper.bufr4",  # 95..40 hPa stored as 950..400 hPa
         [(p * 10.0, t, t - 15.0, np.nan, np.nan, np.nan, 8192) for p, t in zip(upper_p, _t_of(upper_p))]),
    ]
    level = {k: [] for k in ("pressure", "temperature", "dewpoint_temperature", "geopotential_height",
                             "wind_speed", "wind_direction", "vss")}
    first, count = [], []
    for _, _, levels in parts:
        first.append(len(level["pressure"]))
        count.append(len(levels))
        for row in levels:
            for k, v in zip(level, row):
                level[k].append(v)
    n, m = len(parts), len(level["pressure"])
    return {"upper_air/soundings": {
        "station_id": Arr([p[0] for p in parts]),
        "source_key": Arr([p[1] for p in parts]),
        "bulletin_heading": Arr([""] * n),
        "launch_time": _times(["2008-02-05T12:00:00"] * 3 + ["2008-02-05T11:00:00"] * 2),
        "first_arrival_time": _times(["2008-02-05T12:40:00"] * n),
        "latitude": Arr([40.0, 40.0, 40.0, 35.0, 35.0]),
        "longitude": Arr([-95.0, -95.0, -95.0, -100.0, -100.0]),
        "station_elevation": Arr([191.0, 191.0, 191.0, 300.0, 300.0]),
        "quality_flag": _flags([0] * n),
        "first_level": Arr(first), "level_count": Arr(count),
        "pressure": Arr(level["pressure"], dtype="f4"),
        "temperature": Arr(level["temperature"]),
        "dewpoint_temperature": Arr(level["dewpoint_temperature"]),
        "geopotential_height": Arr(level["geopotential_height"]),
        "wind_speed": Arr(level["wind_speed"]),
        "wind_direction": Arr(level["wind_direction"]),
        "time_since_launch": Arr(np.full(m, np.nan)),
        "latitude_displacement": Arr(np.zeros(m)),
        "longitude_displacement": Arr(np.zeros(m)),
        "level_quality_flag": _flags([0] * m),
        "vertical_sounding_significance": Arr(np.asarray(level["vss"], np.int32), _FillValue=-1),
    }}


def test_sondes_refuse_ten_times_pressure_parts_and_merge_the_rest():
    counts = Counter()
    reports = wis2.sonde_reports(_soundings(), START, END, BOX, counts)
    assert counts["sonde_part_part_above_100hpa"] == 1
    assert counts["sonde_part_pressure_scale_x10"] == 1
    by = {r.station: r for r in reports}
    assert set(by) == {"72340", "72649"}
    a = by["72340"]
    assert a.extras["parts"] == ["AA", "BB"]
    assert list(a.p) == sorted(a.p, reverse=True)
    for p, t in ((85000, 284.35), (70000, 278.35), (50000, 260.75)):
        assert a.t[np.nonzero(a.p == p)[0][0]] == t
    assert a.z[0] == a.elev == 191.0  # the surface level sits exactly at the station elevation
    assert np.all(a.t[a.p >= 40000] > 250.0)
    b = by["72649"]
    assert b.extras["parts"] == ["bufr"] and b.t.max() > 280.0
    assert b.p.min() == pytest.approx(_dense_levels(97000.0, 3000.0).min())


# ----------------------------------------------------------------------------- land
def _land():
    n = 5
    return {"land/reports": {
        "station_id": Arr(["0-20000-0-72340", "0-20000-0-72340", "0-20000-0-72649", "0-20000-0-72650",
                           "0-20000-0-72651"]),
        "station_name": Arr(["A", "A", "B", "C", "D"]),
        "time": _times(["2008-02-05T12:00:00"] * n),
        "first_arrival_time": _times(["2008-02-05T12:05:00"] * n),
        "latitude": Arr([40.0, 40.0, 41.0, 42.0, 43.0]), "longitude": Arr([-95.0, -95.0, -96.0, -97.0, -98.0]),
        "station_elevation": Arr([191.0, 191.0, 300.0, 250.0, 260.0]),
        "update_sequence_number": Arr([0, 1, 0, 0, 0]),
        "quality_flag": _flags([0, 0, 0, 256, 1]),
        "2m_temperature": Arr([280.0, 281.0, 275.0, 276.0, 277.0]),
        "2m_temperature_quality": Arr(np.zeros(n, np.uint8)),
        "2m_dewpoint_temperature": Arr([275.0, 276.0, 270.0, 280.0, 271.0]),
        "2m_relative_humidity": Arr(np.full(n, np.nan)),
        "10m_wind_speed": Arr([5.0, 5.0, 3.0, 4.0, 0.0]),
        "10m_wind_direction": Arr([270.0, 270.0, 0.0, 90.0, 0.0]),
        "surface_pressure": Arr([98000.0, 98000.0, 97000.0, 97500.0, 97600.0]),
        "mean_sea_level_pressure": Arr([101000.0] * n),
        "temperature_sensor_height": Arr([2.0, 2.0, 2.0, 5.0, np.nan]),
        "wind_sensor_height": Arr([10.0] * n),
    }, "land/synop_tac": {
        "station_id": Arr(["0-20000-0-72649", "0-20000-0-72700"]),
        "time": _times(["2008-02-05T12:00:00"] * 2),
        "arrival_time": _times(["2008-02-05T12:10:00"] * 2),
        "latitude": Arr([41.0, 44.0]), "longitude": Arr([-96.0, -99.0]),
        "station_elevation": Arr([300.0, 200.0]), "quality_flag": _flags([0, 0]),
        "2m_temperature": Arr([275.0, 270.0]), "2m_dewpoint_temperature": Arr([270.0, 265.0]),
        "10m_wind_speed": Arr([3.0, 2.0]), "10m_wind_direction": Arr([180.0, 90.0]),
        "surface_pressure": Arr([97000.0, 98500.0]), "mean_sea_level_pressure": Arr([101000.0, 101100.0]),
    }}


def test_land_flags_corrections_sensor_heights_and_tac_duplicates():
    counts = Counter()
    reports = wis2.land_reports(_land(), START, END, BOX, counts)
    by = {r.station: r for r in reports}
    assert set(by) == {"72340", "72649", "72650", "72700"}  # 72651 has outside_physical_range set
    assert counts["land_outside_physical_range"] == 1
    assert by["72340"].t[0] == 281.0  # the correction (sequence 1) replaces the original
    assert counts["land_superseded_correction"] == 1
    assert np.isnan(by["72650"].t[0]) and counts["land_temperature_sensor_height"] == 1  # 5 m sensor
    assert np.isnan(by["72649"].spd[0]) and counts["land_wind_invalid"] == 1  # 3 m/s from 0 degrees
    assert counts["synop_tac_duplicate_of_bufr"] == 1
    assert by["72340"].rh[0] == pytest.approx(wis2.rh_from_dewpoint(281.0, 276.0)[()])
    assert by["72340"].received == np.datetime64("2008-02-05T12:05:00")


def _ocean():
    kinds = ["ship", "drifting_buoy", "ship", "glider"]
    return {"ocean/surface": {
        "call_sign": Arr(["SHIP1", "", "SHIP1", "G1"]), "wigos_id": Arr(["", "0-22000-0-4401", "", ""]),
        "platform_kind": Arr(kinds), "time": _times(["2008-02-05T12:00:00"] * 4),
        "first_arrival_time": _times(["2008-02-05T12:20:00"] * 4),
        "latitude": Arr([35.0, 36.0, 35.0, 37.0]), "longitude": Arr([-85.0, -86.0, -85.0, -87.0]),
        "quality_flag": _flags([0, 0, 0, 0]),
        "air_temperature": Arr([290.0, 289.0, 290.0, 288.0]),
        "dewpoint_temperature": Arr([285.0, np.nan, 285.0, np.nan]),
        "relative_humidity": Arr([np.nan, 80.0, np.nan, np.nan]),
        "wind_speed": Arr([6.0, 4.0, 6.0, 1.0]), "wind_direction": Arr([200.0, 100.0, 200.0, 10.0]),
        "surface_air_pressure": Arr([101200.0, 101300.0, 101200.0, np.nan]),
        "mean_sea_level_pressure": Arr([101200.0, 101300.0, 101200.0, np.nan]),
    }}


def test_ocean_ships_buoys_and_duplicates():
    counts = Counter()
    reports = wis2.ocean_reports(_ocean(), START, END, BOX, counts)
    assert [(r.station, r.fm) for r in reports] == [("SHIP1", 13), ("0-22000-0-4401", 18)]
    assert counts["ocean_duplicate"] == 1 and counts["ocean_platform_glider"] == 1
    assert reports[1].rh[0] == 80.0


# ----------------------------------------------------------------------------- aircraft, profilers
def _aircraft():
    return {"aircraft": {
        "report_id": Arr(np.array([11, 11, 12], np.int64)),
        "aircraft_identifier": Arr(["EU1", "EU1", "EU2"]),
        "time": _times(["2008-02-05T12:00:00"] * 3, SEC),
        "first_arrival_time": _times(["2008-02-05T12:01:00"] * 3, SEC),
        "latitude": Arr([40.0, 40.0, 41.0]), "longitude": Arr([-95.0, -95.0, -96.0]),
        "quality_flag": _flags([0, 0, 0]),
        "flight_level": Arr(np.array([11000, 11000, 3000], np.int32), _FillValue=-2147483648),
        "temperature": Arr([216.65, 216.65, 268.0]), "dewpoint_temperature": Arr([np.nan] * 3),
        "relative_humidity": Arr([np.nan] * 3),
        "wind_speed": Arr([40.0, 40.0, 10.0]),
        "wind_direction": Arr(np.array([270, 270, -2147483648], np.int32), _FillValue=-2147483648),
    }}


def test_aircraft_pressure_altitude_and_report_id():
    counts = Counter()
    reports = wis2.aircraft_reports(_aircraft(), START, END, BOX, counts)
    assert counts["aircraft_duplicate_report_id"] == 1 and len(reports) == 2
    assert reports[0].p[0] == pytest.approx(22632.06, abs=0.01)  # ISA at 11 km
    assert np.isnan(reports[1].spd[0])  # direction missing (fill value)


def test_profiler_speed_direction_or_components():
    root = {"upper_air/wind_profiler": {
        "station_id": Arr(["0-20000-0-10266"]), "time": _times(["2008-02-05T12:00:00"]),
        "first_arrival_time": _times(["2008-02-05T12:02:00"]),
        "latitude": Arr([40.0]), "longitude": Arr([-95.0]), "station_elevation": Arr([100.0]),
        "quality_flag": _flags([0]), "first_level": Arr([0]), "level_count": Arr([3]),
        "level_quality_flag": _flags([0, 0, 1024]),
        "height": Arr([600.0, 300.0, 900.0]),
        "wind_speed": Arr([10.0, np.nan, 12.0]), "wind_direction": Arr([90.0, np.nan, 100.0]),
        "eastward_wind": Arr([np.nan, 0.0, np.nan]), "northward_wind": Arr([np.nan, -5.0, np.nan]),
    }}
    counts = Counter()
    (rep,) = wis2.profiler_reports(root, START, END, BOX, counts)
    assert list(rep.z) == [300.0, 600.0]  # the provider-flagged gate is refused
    assert rep.spd[0] == 5.0 and rep.wdir[0] == 360.0  # a northerly from its components
    assert rep.spd[1] == 10.0 and rep.wdir[1] == 90.0


# ----------------------------------------------------------------------------- outputs
def _all_reports():
    root = {**_soundings(), **_land(), **_ocean(), **_aircraft()}
    reports, counts = wis2.read_reports(root, START, END, BOX, ("land", "ocean", "sonde", "aircraft"))
    return reports, counts


def test_neutral_table_round_trip(tmp_path):
    from woof.obs.neutral_table import decode_neutral_csv
    reports, _ = _all_reports()
    table = ObsErrTable.load()
    rows, counts = wis2.neutral_rows(reports, table)
    path = tmp_path / "wis2.csv"
    wis2.write_table(rows, path)
    _, decoded, _ = decode_neutral_csv(path.read_text())
    assert len(decoded) == len(rows) > 0
    labels = Counter(r.measurement for r in decoded)
    assert labels["sonde_level"] and labels["aircraft_level"] and labels["screen_temperature_2m"]
    t2 = next(r for r in decoded if r.station_id == "72340" and r.measurement == "screen_temperature_2m")
    assert t2.value == 281.0 and t2.error == table.errors(12, 98000.0, 191.0)["t"]
    assert t2.received_time.isoformat() == "2008-02-05T12:05:00+00:00"
    u = next(r for r in decoded if r.station_id == "EU1" and r.variable == "wind_u_m_s")
    assert u.value == pytest.approx(40.0) and u.level_pa == pytest.approx(22632.06, abs=0.01)


def test_obs_gts_reads_back_through_the_3dvar_reader(tmp_path):
    from woof.da.var import obs_wrfda as ow
    reports, _ = _all_reports()
    table = ObsErrTable.load()
    path = tmp_path / "ob.ascii"
    wis2.write_obs_gts(reports, path, table)
    proj = ow.LambertWRFDA(cen_lat=40.0, cen_lon=-95.0, stand_lon=-95.0, truelat1=30.0, truelat2=60.0,
                           moad_cen_lat=40.0, dx=60000.0, ide=89, jde=59)
    obs = ow.read_obs_ascii(path, proj, analysis_date="2008-02-05_12:00:00",
                            time_window_min="2008-02-05_11:00:00", time_window_max="2008-02-05_13:00:00")
    counts = obs.counts()
    assert (counts["synop"], counts["ships"], counts["buoy"]) == (4, 1, 1)
    assert (counts["sound"], counts["sonde_sfc"], counts["airep"]) == (2, 2, 2)
    sound = obs["sound"]
    t = sound.fields["t"]
    p = sound.coords["p"]
    k = int(np.nonzero(np.asarray(p) == 50000.0)[0][0])
    assert t.inv[k] == pytest.approx(260.75) and t.error[k] == pytest.approx(table.errors(35, 50000.0)["t"], abs=5e-3)
    assert np.all(np.asarray(t.inv)[(np.asarray(p) >= 40000.0) & (np.asarray(t.qc) >= 0)] > 230.0)


def test_sondes_all_parts_flagged():
    root = _soundings()
    root["upper_air/soundings"]["quality_flag"] = _flags([16] * 5)
    counts = Counter()
    assert wis2.sonde_reports(root, START, END, BOX, counts) == []
    assert counts["sonde_position_invalid"] == 5


def test_flagged_correction_does_not_revive_the_original_and_tac_ids_match_wigos():
    root = _land()
    group = root["land/reports"]
    group["quality_flag"] = _flags([0, 2, 0, 256, 1])  # the correction of 72340 is implausible
    root["land/synop_tac"]["station_id"] = Arr(["72649", "72700"])  # TAC with bare WMO ids
    counts = Counter()
    by = {r.station: r for r in wis2.land_reports(root, START, END, BOX, counts)}
    assert "72340" not in by
    assert counts["synop_tac_duplicate_of_bufr"] == 1


def test_gross_bounds_and_unplaced_reports():
    root = _land()
    group = root["land/reports"]
    group["2m_temperature"] = Arr([281.0, 281.0, 9999.0, 276.0, 277.0])
    group["mean_sea_level_pressure"] = Arr([101000.0, 101000.0, 9.99e9, 101000.0, 101000.0])
    group["surface_pressure"] = Arr([98000.0, 98000.0, 97000.0, np.nan, 97600.0])
    group["station_elevation"] = Arr([191.0, 191.0, 300.0, np.nan, 260.0])
    reports, counts = wis2.read_reports(root, START, END, BOX, ("land",))
    by = {r.station: r for r in reports}
    assert np.isnan(by["72649"].t[0]) and np.isnan(by["72649"].rh[0]) and np.isnan(by["72649"].slp)
    assert counts["land_t_out_of_bounds"] == 1 and counts["land_slp_out_of_bounds"] == 1
    assert "72650" not in by and counts["land_no_pressure_or_height"] == 1


def test_obs_gts_names_are_ascii(tmp_path):
    root = _land()
    root["land/reports"]["station_name"] = Arr(["Москва", "Москва", "東京\tX", "C", "D"])
    reports, _ = wis2.read_reports(root, START, END, BOX, ("land",))
    path = tmp_path / "ob.ascii"
    wis2.write_obs_gts(reports, path, ObsErrTable.load())
    text = path.read_text(encoding="ascii")
    assert "??????" in text and "\t" not in text


# ----------------------------------------------------------------------------- satellite point winds
MS = {"units": "milliseconds since 1970-01-01", "_FillValue": np.iinfo(np.int64).min}
I32 = np.iinfo(np.int32).max


def _ms(texts):
    return Arr([np.datetime64(x, "ms").astype(np.int64) for x in texts], **MS)


def _amv():
    """Made-up AMVs: rows 0-1 geostationary (QI 85 and 99), 2 polar AVHRR, 3 polar VIIRS,
    4 a later copy of 0, 5 QI 40, 6 an unknown platform, 7 an unvetted template, 8 above 100 hPa."""
    fill4 = [I32] * 4
    sga = [[6, 5, 4, I32], [4, 5, 6, 7], [6, 5, 4, I32], [6, 5, 4, I32], [6, 5, 4, I32], [6, 5, 4, I32],
           [6, 5, 4, I32], [6, 5, 4, I32], [6, 5, 4, I32]]
    pc = [[35, 85, I32, I32], [100, 99, 97, 54], [90, 88, I32, I32], [90, 88, I32, I32], [35, 85, I32, I32],
          [90, 40, I32, I32], [90, 90, I32, I32], [90, 90, I32, I32], [90, 90, I32, I32]]
    n = len(sga)
    return {"satellite/amv": {
        "time": _ms(["2008-02-05T12:00:00"] * n),
        "first_arrival_time": _times(["2008-02-05T12:10:00", "2008-02-05T12:10:00", "2008-02-05T12:10:00",
                                      "2008-02-05T12:10:00", "2008-02-05T12:30:00"] + ["2008-02-05T12:10:00"] * 4,
                                     SEC),
        "latitude": Arr([[40.0, np.nan, 40.0, np.nan], [41.0] * 4, [42.0] * 4, [43.0] * 4, [40.0, np.nan, 40.0, np.nan],
                         [39.0] * 4, [38.0] * 4, [37.0] * 4, [36.0] * 4]),
        "longitude": Arr([[-95.0] * 4, [-96.0] * 4, [-97.0] * 4, [-98.0] * 4, [-95.0] * 4, [-94.0] * 4,
                          [-93.0] * 4, [-92.0] * 4, [-91.0] * 4]),
        "quality_flag": _flags([0] * n),
        "bufr_template": Arr(["310077"] * 7 + ["310014", "310077"]),
        "satellite_identifier": Arr(np.array([[71] + fill4, [811] + fill4, [852] + fill4, [224] + fill4,
                                              [71] + fill4, [71] + fill4, [999] + fill4, [71] + fill4,
                                              [71] + fill4], np.int32), _FillValue=I32),
        "satellite_instruments": Arr(np.array([fill4, fill4, [591] * 4, [616] * 4, fill4, fill4, fill4, fill4,
                                               fill4], np.int32), _FillValue=I32),
        "pressure": Arr([[50000.0] + [np.nan] * 14] * 8 + [[5000.0] + [np.nan] * 14]),
        "wind_speed": Arr([10.0, 20.0, 15.0, 12.0, 10.0, 8.0, 9.0, 9.0, 30.0]),
        "wind_direction": Arr(np.array([270, 250, 300, 180, 270, 90, 90, 90, 270], np.int32), _FillValue=I32),
        "standard_generating_application": Arr(np.array(sga, np.int32), _FillValue=I32),
        "percent_confidence": Arr(np.array(pc, np.int32), _FillValue=I32),
    }}


def _scat():
    """Made-up ASCAT cells: 0 selected ambiguity 2, 1 no retrieval, 2 land bit (9), 3 low-speed bit
    (13) only, 4 ice, 5 land fraction, 6 a relay copy of 0."""
    n = 7
    land_bit, low_bit = 1 << (24 - 9), 1 << (24 - 13)
    return {"satellite/scatterometer": {
        "time": _ms(["2008-02-05T12:00:00"] * n),
        "first_arrival_time": _times(["2008-02-05T12:40:00"] * n, SEC),
        "latitude": Arr([40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 40.0]),
        "longitude": Arr([-95.0, -95.5, -96.0, -96.5, -97.0, -97.5, -95.0]),
        "quality_flag": _flags([0] * n),
        "bufr_template": Arr(["312061"] * n),
        "satellite_identifier": Arr(np.array([5] * n, np.int32), _FillValue=I32),
        "index_of_selected_wind_vector": Arr(np.array([2, I32, 1, 1, 1, 1, 2], np.int32), _FillValue=I32),
        "number_of_vector_ambiguities": Arr(np.array([2, 0, 2, 2, 2, 2, 2], np.int32), _FillValue=I32),
        "wind_speed_at10_m": Arr([[7.0, 8.0, np.nan, np.nan]] * n),
        "wind_direction_at10_m": Arr([[10.0, 190.0, np.nan, np.nan]] * n),
        "wind_vector_cell_quality": Arr(np.array([0, 0, land_bit, low_bit, 0, 0, 0], np.int32), _FillValue=I32),
        "land_fraction": Arr([[0.0] * 3] * 5 + [[0.0, 0.2, 0.0]] + [[0.0] * 3]),
        "ice_probability": Arr([np.nan, np.nan, 0.0, 0.0, 0.6, 0.0, np.nan]),
    }}


def test_amv_quality_index_platforms_and_copies():
    counts = Counter()
    reports = wis2.amv_reports(_amv(), START, END, BOX, counts)
    by = {r.station: r for r in reports}
    assert set(by) == {"GEO-71", "GEO-811", "AVHRR-852", "VIIRS-224"}
    assert by["GEO-71"].extras["qi"] == 85.0 and by["GEO-811"].extras["qi"] == 99.0
    assert by["GEO-71"].received == np.datetime64("2008-02-05T12:10:00")  # the first copy
    assert counts["amv_duplicate_copy"] == 1 and counts["amv_qi_below_minimum"] == 1
    assert counts["amv_platform_999_unknown"] == 1 and counts["amv_template_310014_not_vetted"] == 1
    assert counts["amv_pressure_out_of_bounds"] == 1 and counts["ahi_amv_group_skipped_no_quality_index"] == 0
    assert by["AVHRR-852"].p[0] == 50000.0 and by["AVHRR-852"].spd[0] == 15.0


def test_scatterometer_selected_ambiguity_and_cell_flags():
    counts = Counter()
    reports = wis2.scat_reports(_scat(), START, END, BOX, counts)
    assert [(r.lat, r.spd[0], r.wdir[0]) for r in reports] == [(40.0, 8.0, 190.0), (41.5, 7.0, 10.0)]
    assert counts["scat_no_selected_wind"] == 1 and counts["scat_wind_vector_cell_quality"] == 1
    assert counts["scat_ice"] == 1 and counts["scat_land"] == 1 and counts["scat_duplicate_copy"] == 1
    assert reports[0].z[0] == 10.0 and reports[0].fm == 281


def test_satellite_rows_found_by_bisection(monkeypatch):
    n = 20000
    times = np.datetime64("2008-02-05T00:00:00", "ms") + np.arange(n) * np.timedelta64(10, "s")
    group = {"time": Arr(times.astype(np.int64), **MS), "latitude": Arr(np.full(n, 40.0)),
             "longitude": Arr(np.full(n, -95.0))}
    monkeypatch.setattr(wis2, "SAT_SCAN_SMALL", 0)
    monkeypatch.setattr(wis2, "SAT_SCAN_SAMPLE", 10)
    monkeypatch.setattr(wis2, "SAT_SCAN_MARGIN", np.timedelta64(1, "h"))
    counts = Counter()
    index, when = wis2._sat_select(group, START, END, BOX, counts, "x")
    brute = np.nonzero((times >= np.datetime64(START)) & (times <= np.datetime64(END)))[0]
    assert np.array_equal(index, brute) and counts["x_rows_scanned"] < n


def test_thinning_keeps_the_best_amv_per_box_and_layer():
    reports, counts = wis2.read_reports(_amv(), START, END, BOX, ("amv",))
    thinned = wis2.thin_satellite_winds(reports, 500.0, counts)
    assert len(thinned) < len(reports) and counts["amv_thinned"] == len(reports) - len(thinned)
    assert max(r.extras["qi"] for r in thinned) == 99.0
    assert wis2.thin_satellite_winds(reports, 0.0, counts) == reports


def test_satellite_winds_in_the_table_and_obs_gts(tmp_path):
    from woof.da.var import obs_wrfda as ow
    from woof.obs.neutral_table import decode_neutral_csv
    reports, _ = wis2.read_reports({**_amv(), **_scat()}, START, END, BOX, ("amv", "scat"))
    table = ObsErrTable.load()
    rows, _ = wis2.neutral_rows(reports, table)
    path = tmp_path / "w.csv"
    wis2.write_table(rows, path)
    _, decoded, counters = decode_neutral_csv(path.read_text())
    assert len(decoded) == len(rows) and counters["rows_unknown_measurement"] == 0
    amv = next(r for r in decoded if r.station_id == "GEO-71" and r.variable == "wind_u_m_s")
    assert amv.measurement == "amv_assigned_pressure" and amv.level_pa == 50000.0
    assert amv.elevation_m == pytest.approx(5574.4, abs=1.0)  # the standard atmosphere's 500 hPa
    assert amv.error == table.errors(88, 50000.0)["u"]
    assert {r.measurement for r in decoded if r.source == "wis2.scat"} == {"scatterometer_wind_10m"}
    out = tmp_path / "ob.ascii"
    fm = wis2.write_obs_gts(reports, out, table)
    assert fm == Counter({88: 3, 281: 2})  # the VIIRS AMV is left out
    proj = ow.LambertWRFDA(cen_lat=40.0, cen_lon=-95.0, stand_lon=-95.0, truelat1=30.0, truelat2=60.0,
                           moad_cen_lat=40.0, dx=60000.0, ide=89, jde=59)
    obs = ow.read_obs_ascii(out, proj, analysis_date="2008-02-05_12:00:00",
                            time_window_min="2008-02-05_11:00:00", time_window_max="2008-02-05_13:00:00")
    counts = obs.counts()
    assert (counts["geoamv"], counts["polaramv"], counts["qscat"]) == (2, 1, 2)


def test_satellite_groups_with_every_row_refused():
    root = {**_amv(), **_scat()}
    root["satellite/amv"]["bufr_template"] = Arr(["310014"] * 9)
    root["satellite/scatterometer"]["quality_flag"] = _flags([16] * 7)
    counts = Counter()
    assert wis2.amv_reports(root, START, END, BOX, counts) == []
    assert wis2.scat_reports(root, START, END, BOX, counts) == []
    assert counts["amv_template_310014_not_vetted"] == 9 and counts["scat_position_invalid"] == 7


def test_thinning_boxes_are_fixed_per_latitude_row():
    def amv(lat, lon, qi):
        return wis2.Report(family="amv", fm=88, station="GEO-71", name="x", lat=lat, lon=lon, elev=np.nan,
                           time=np.datetime64("2008-02-05T12:00:00"), received=np.datetime64("NaT", "s"), p=[50000.0],
                           z=[np.nan], t=[np.nan], td=[np.nan], rh=[np.nan], spd=[5.0], wdir=[90.0],
                           surface=False, extras={"qi": qi})
    # Two AMVs in one 100 km row at 70N, 0.1 degree of longitude apart: one box, whatever their latitudes.
    step = 100.0 / 111.2
    row_lat = np.floor((70.0 + 90.0) / step) * step - 90.0
    kept = wis2.thin_satellite_winds([amv(row_lat + 0.01, 10.0, 90.0), amv(row_lat + step - 0.01, 10.1, 95.0)],
                                     100.0, Counter())
    assert [r.extras["qi"] for r in kept] == [95.0]


def _ro():
    """Made-up occultations (template 310026 layout: bending-angle levels with their tangent points, then
    refractivity levels): 0 four levels (one 0 % confident, one with no height), 1 a later copy of 0,
    2 a profile under 100 % confidence, 3 flagged refractivity non-nominal (0-33-039 bit 6), 4 bending
    angles only, 5 bending-angle and refractivity counts that differ (levels at the occultation point)."""
    n = 6
    heights = [1000, 3000, I32, 15000]
    refr = [[300.0, np.nan, 250.0, np.nan, 200.0, np.nan, 50.0, np.nan]]
    h = [heights] * 4 + [[I32] * 4, [2000, 8000, I32, I32]]
    ref = refr * 4 + [[np.nan] * 8, [280.0, np.nan, 150.0, np.nan] + [np.nan] * 4]
    rep = [[4, 4, 0]] * 4 + [[4, 0, 0], [3, 2, 0]]
    lat = [[40.0, 40.1, 40.2, 40.3, 40.4]] * 4 + [[40.0] * 5, [35.0, 36.0, 36.1, 36.2, np.nan]]
    lon = [[-95.0, -95.1, -95.2, -95.3, -95.4]] * 4 + [[-95.0] * 5, [-100.0, -101.0, -101.1, -101.2, np.nan]]
    conf = [[100] + [90] * 4 + [80, 0, 70, 60] + [I32] * 3]
    conf = conf + conf + [[50] + [90] * 8 + [I32] * 3] + conf + conf + [[100] + [90] * 3 + [80, 80] + [I32] * 6]
    return {"satellite/gnss_ro": {
        "time": _ms(["2008-02-05T12:00:00"] * n),
        "first_arrival_time": _times(["2008-02-05T12:20:00", "2008-02-05T12:50:00"] + ["2008-02-05T12:20:00"] * 4,
                                     SEC),
        "latitude": Arr(lat), "longitude": Arr(lon),
        "quality_flag": _flags([0] * n),
        "bufr_template": Arr(["310026"] * n),
        "radio_occultation_data_quality_flags": Arr(np.array([64, 64, 0, 1 << (16 - 6), 0, 8256], np.int32),
                                                    _FillValue=I32),
        "satellite_identifier": Arr(np.array([750, 750, 751, 523, 3, 44], np.int32), _FillValue=I32),
        "platform_transmitter_id_number": Arr(np.array([21, 21, 5, 7, 9, 11], np.int32), _FillValue=I32),
        "extended_delayed_descriptor_replication_factor": Arr(np.array(rep, np.int32), _FillValue=I32),
        "percent_confidence": Arr(np.array(conf, np.int32), _FillValue=I32),
        "height": Arr(np.array(h, np.int32), _FillValue=I32),
        "atmospheric_refractivity": Arr(np.array(ref)),
    }}


def test_ro_profiles_confidence_flags_copies_and_locations():
    counts = Counter()
    reports = wis2.ro_reports(_ro(), START, END, BOX, counts)
    by = {r.station: r for r in reports}
    assert set(by) == {"75021", "04411"}
    a = by["75021"]
    assert a.fm == 116 and a.received == np.datetime64("2008-02-05T12:20:00")  # the first copy
    assert a.z.tolist() == [1000.0, 15000.0]  # the 0 % level and the level with no height are dropped
    assert a.extras["refractivity"].tolist() == [300.0, 50.0]
    assert a.level_lat.tolist() == [40.1, 40.4] and a.level_lon.tolist() == [-95.1, -95.4]  # tangent points
    assert (a.lat, a.lon) == (40.0, -95.0)
    b = by["04411"]  # counts differ: every level at the occultation point
    assert b.level_lat.tolist() == [35.0, 35.0] and b.extras["refractivity"].tolist() == [280.0, 150.0]
    assert counts["ro_duplicate_copy"] == 1 and counts["ro_profile_confidence_below_100"] == 1
    assert counts["ro_flag_refractivity_non_nominal"] == 1 and counts["ro_bending_angle_only"] == 1
    assert counts["ro_profiles_at_occultation_point"] == 1 and counts["ro_level_confidence_zero"] == 1


def test_gpsref_error_is_wrfdas():
    from woof.obs.obserr_wrfda import gpsref_error
    assert gpsref_error(100.0, 15000.0, 10.0) == pytest.approx(0.3)
    # 3 km at 45 degrees: halfway between the pole's 1.2 % and the equator's 2.3 % of N.
    assert gpsref_error(250.0, 3000.0, 45.0) == pytest.approx(250.0 * 0.0175)
    assert gpsref_error(300.0, 1000.0, 0.0) == pytest.approx(300.0 * 0.025)
    assert gpsref_error(300.0, 1000.0, 90.0) == pytest.approx(300.0 * 0.01 * (1.2 * 11000.0 / 12000.0 + 0.3))


def test_ro_in_the_table_and_obs_gts(tmp_path):
    from woof.da.var import obs_wrfda as ow
    from woof.obs.neutral_table import decode_neutral_csv
    from woof.obs.obserr_wrfda import gpsref_error
    reports, counts = wis2.read_reports(_ro(), START, END, BOX, ("ro",))
    assert counts["ro_reports"] == 2
    rows, row_counts = wis2.neutral_rows(reports, ObsErrTable.load())
    assert row_counts["rows_ro_refractivity_n"] == 4
    path = tmp_path / "ro.csv"
    wis2.write_table(rows, path)
    _, decoded, counters = decode_neutral_csv(path.read_text())
    assert len(decoded) == 4 and counters["rows_unknown_measurement"] == 0
    low = next(r for r in decoded if r.station_id == "75021" and r.elevation_m == 1000.0)
    assert low.variable == "refractivity_n" and low.measurement == "ro_refractivity_tangent_point"
    assert low.value == 300.0 and low.error == pytest.approx(gpsref_error(300.0, 1000.0, 40.1))
    assert (low.latitude_deg, low.longitude_deg) == (40.1, -95.1)
    assert len({r.identity() for r in decoded}) == 4  # each level its own measurement
    out = tmp_path / "ob.ascii"
    assert wis2.write_obs_gts(reports, out, ObsErrTable.load()) == Counter({116: 2})
    assert "GPSRF =      2" in out.read_text()
    proj = ow.LambertWRFDA(cen_lat=40.0, cen_lon=-95.0, stand_lon=-95.0, truelat1=30.0, truelat2=60.0,
                           moad_cen_lat=40.0, dx=60000.0, ide=89, jde=59)
    obs = ow.read_obs_ascii(out, proj, analysis_date="2008-02-05_12:00:00",
                            time_window_min="2008-02-05_11:00:00", time_window_max="2008-02-05_13:00:00")
    assert obs.counts()["gpsref"] == 2
    ref = obs["gpsref"].fields["ref"]
    assert sorted(np.asarray(ref.inv).tolist()) == [50.0, 150.0, 280.0, 300.0]
    k = int(np.nonzero(np.asarray(ref.inv) == 300.0)[0][0])
    assert ref.error[k] == pytest.approx(gpsref_error(300.0, 1000.0, 40.0), abs=5e-3) and ref.qc[k] == 0


def test_ro_gradient_check_refuses_a_duct():
    h = np.arange(0.0, 5000.0, 200.0)
    n = 300.0 * np.exp(-h / 7000.0)
    assert wis2.ro_gradient_ok(h, n).all()  # -43 N/km at the surface: no super-refraction
    ducted = n.copy()
    ducted[:3] += [40.0, 25.0, 0.0]  # -100 N/km near the surface
    ok = wis2.ro_gradient_ok(h, ducted)
    assert not ok[:3].any() and ok[6:].all()


def test_ro_obs_gts_error_never_prints_as_zero(tmp_path):
    from woof.da.var import obs_wrfda as ow
    rep = wis2.Report(family="ro", fm=116, station="75021", name="GNSS-RO 750 G21", lat=40.0, lon=-95.0,
                      elev=np.nan, time=np.datetime64("2008-02-05T12:00:00"), received=np.datetime64("NaT", "s"),
                      p=[np.nan, np.nan], z=[29000.0, 5000.0], t=[np.nan] * 2, td=[np.nan] * 2, rh=[np.nan] * 2,
                      spd=[np.nan] * 2, wdir=[np.nan] * 2, surface=False, extras={"refractivity": np.array([2.0, 150.0])})
    out = tmp_path / "ob.ascii"
    wis2.write_obs_gts([rep], out, ObsErrTable.load())
    proj = ow.LambertWRFDA(cen_lat=40.0, cen_lon=-95.0, stand_lon=-95.0, truelat1=30.0, truelat2=60.0,
                           moad_cen_lat=40.0, dx=60000.0, ide=89, jde=59)
    obs = ow.read_obs_ascii(out, proj, analysis_date="2008-02-05_12:00:00",
                            time_window_min="2008-02-05_11:00:00", time_window_max="2008-02-05_13:00:00")
    ref = obs["gpsref"].fields["ref"]
    by = dict(zip(np.asarray(ref.inv).tolist(), np.asarray(ref.error).tolist()))
    assert by[2.0] == pytest.approx(0.01)  # 2 N * 1 % * 0.3 = 0.006, floored at the format's 0.01
    assert by[150.0] > 0.5
