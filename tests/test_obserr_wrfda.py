"""woof.obs.obserr_wrfda: OBSPROC observation errors, checked against OBSPROC's own output."""

import pytest

from woof.obs.obserr_wrfda import (LEVELS, ObsErrTable, intplin, intplog, locate)

TAB = ObsErrTable.load()

# Expected errors are read off the WRFDA tutorial obs_gts (OBSPROC output, Feb 2008 case,
# ob.ascii).  Each row is (FM, pressure Pa, height m or None) -> the printed error columns
# (p, speed, direction, height, t, td, rh); obs_gts prints two decimals.  Rows with
# pressure None are levels OBSPROC printed without pressure after recovering it from the
# height, so those use the height-based fallback.
OBS_GTS = [
    (13, 102300.0, None, (160.0, 1.1, 1.1, 6.0, 2.0, 2.0, 10.0)),          # FM-13 SHIP
    (12, 101630.0, None, (100.0, 1.1, 1.1, 6.0, 2.0, 2.0, 10.0)),          # FM-12 SYNOP
    (15, 79356.164, None, (100.0, 1.13, 1.13, 6.0, 2.0, 2.0, 10.0)),       # FM-15 METAR
    (15, 67659.25, None, (100.0, 1.49, 1.49, 6.0, 2.0, 2.0, 10.0)),        # FM-15 METAR
    (18, 101940.0, None, (100.0, 1.4, 1.4, 6.0, 2.0, 2.0, 10.0)),          # FM-18 BUOY
    (35, 99100.0, None, (100.0, 1.1, 1.1, 7.06, 1.0, 1.0, 14.72)),         # FM-35 TEMP
    (35, 72600.0, None, (100.0, 1.35, 1.35, 8.49, 1.0, 1.0, 10.0)),        # FM-35 TEMP
    (42, 49564.301, None, (200.0, 3.6, 3.6, 9.49, 1.0, 1.0, 10.0)),        # FM-42 AMDAR
    (97, 22728.5, None, (200.0, 3.6, 3.6, 14.63, 1.0, 1.0, 10.0)),         # FM-97 AIREP
    (97, 69681.016, None, (200.0, 3.6, 3.6, 6.05, 1.0, 1.0, 10.0)),        # FM-97 AIREP
    (88, 63000.0, None, (200.0, 4.5, 4.5, 34.07, 2.67, 2.67, 10.0)),       # FM-88 SATOB
    (88, 46000.0, None, (200.0, 4.5, 4.5, 37.62, 2.49, 2.49, 10.0)),       # FM-88 SATOB
    (111, 88050.0, None, (89.13, 1.8, 35.0, 7.4, 1.62, 1.62, 10.0)),       # FM-111 GPSPW: NCEP default
    (32, None, 274.0, (200.0, 2.2, 2.2, 5.08, 1.0, 1.0, 10.0)),            # FM-32 PILOT, first level
    (32, None, 1200.0, (200.0, 2.2, 2.2, 5.35, 1.0, 1.0, 10.0)),           # FM-32 PILOT
    (132, None, 132.0, (200.0, 2.2, 2.2, 5.04, 1.0, 1.0, 10.0)),           # FM-132 PROFILER
]


@pytest.mark.parametrize("fm,p,h,want", OBS_GTS)
def test_errors_match_obs_gts(fm, p, h, want):
    e = TAB.errors(fm, p, h)
    got = (e["p"], e["speed"], e["direction"], e["height"], e["t"], e["td"], e["rh"])
    assert got == pytest.approx(want, abs=0.0051)
    assert e["u"] == e["v"] == e["speed"] or fm == 111


def test_satem_thickness_and_direction_tables():
    # FM-86 SATEM at 850 hPa: speed 2.5, direction 32.0 (SATEM always reads the SPD/DIR
    # tables), thickness 12.5 (printed in the temperature column).
    e = TAB.errors(86, 85000.0)
    assert (e["speed"], e["direction"], e["thickness"], e["t"]) == pytest.approx((2.5, 32.0, 12.5, 3.5))


def test_synop_height_is_linear_others_log():
    # SYNOP height uses intplin, METAR intplog: they differ between table levels only
    # if the table is not constant, so check them on the RAOBS table (sound) at one level.
    row = TAB.tables["height"]["sound"]
    p = 60000.0
    assert intplog(p, LEVELS, row) != intplin(p, LEVELS, row)
    assert TAB.errors(35, p)["height"] == intplog(p, LEVELS, row)


def test_intplin_edges_and_midpoint():
    xx, yy = [100.0, 50.0, 10.0], [1.0, 2.0, 4.0]  # descending, like the pressure levels
    assert intplin(200.0, xx, yy) == 1.0       # beyond the first level: first value
    assert intplin(5.0, xx, yy) == 4.0         # beyond the last level: last value
    assert intplin(75.0, xx, yy) == 1.5
    assert intplin(100.0, xx, yy) == 1.0 and intplin(10.0, xx, yy) == 4.0  # the end points
    assert intplin(50.0, xx, yy) == 2.0


def test_intplog_edges_and_geometric_midpoint():
    xx, yy = [100.0, 25.0], [1.0, 3.0]
    assert intplog(1000.0, xx, yy) == 1.0 and intplog(1.0, xx, yy) == 3.0
    assert intplog(50.0, xx, yy) == pytest.approx(1.0 + 2.0 * (1.0 / 2.0))  # ln 2 / ln 4 = 1/2


def test_locate_ascending_descending_and_ends():
    assert locate(2.5, [1.0, 2.0, 3.0]) == 2
    assert locate(0.0, [1.0, 2.0, 3.0]) == 0 and locate(9.0, [1.0, 2.0, 3.0]) == 3
    assert locate(2.5, [3.0, 2.0, 1.0]) == 1
    assert locate(1.0, [1.0, 2.0, 3.0]) == 1 and locate(3.0, [1.0, 2.0, 3.0]) == 2


def test_file_values_read_back():
    # RAOBS temperature is 1.0 K and SURFACE LAND 2.0 K at every level; BUOY pressure error is
    # not in the PRESSURE block's cases, so it keeps the NCEP profile (100 Pa at 1000 hPa and below).
    assert TAB.tables["t"]["sound"] == [1.0] * 15 and TAB.tables["t"]["synop"] == [2.0] * 15
    assert TAB.tables["p"]["buoys"][0] == 100.0
    assert TAB.tables["uv"]["sound"][0] == 1.1   # 1100 hPa is the file's last wind value
    assert TAB.tables["uv"]["sound"][-1] == 2.7  # 10 hPa is its first


def test_missing_pressure_needs_height_and_unported_fm_raises():
    with pytest.raises(ValueError):
        TAB.errors(12, None)
    with pytest.raises(NotImplementedError):
        TAB.errors(135, 85000.0)
