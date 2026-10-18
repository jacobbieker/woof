"""The stratospheric top-up: an isobaric upper source joined above a column."""
from __future__ import annotations

import math
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from woof.ingest.horiz import HorizontalSnapshot
from woof.ingest.upper_splice import (
    DEFAULT_SEAM_BOTTOM_PA,
    DEFAULT_SEAM_TOP_PA,
    UpperSpliceRefusal,
    receipt_sentence,
    relative_humidity_from_specific,
    specific_humidity_from_rh,
    splice_upper_snapshot,
)

VALID = datetime(2024, 7, 30, 13)
NY, NX = 3, 4
HRRR_PRS_HPA = np.arange(1000.0, 49.0, -25.0)          # bottom first, 39 levels
GFS_HPA = np.array([10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 70.0, 100.0,
                    150.0, 200.0])                        # top first, as decoded


def _temperature(p_pa):
    # Isothermal lower stratosphere, warming slowly upward.
    return 216.65 + 4.0 * np.log(10000.0 / p_pa)


def _height(p_pa):
    # Hypsometric height through the profile above, from 100 hPa at 16 km.
    return 16000.0 + 29.27 * 220.0 * np.log(10000.0 / p_pa)


def _column(p_pa, function, shape):
    p = np.asarray(p_pa, dtype=np.float64)
    if p.ndim == 1:
        p = np.broadcast_to(p[:, None, None], (p.size, *shape))
    return function(p).astype(np.float32)


def _mass_pressure(levels_hpa):
    return np.broadcast_to(np.asarray(levels_hpa)[:, None, None] * 100.0,
                           (len(levels_hpa), NY, NX)).astype(np.float32)


def _lower(levels_hpa=HRRR_PRS_HPA, *, pressure=None, t_offset=0.0,
           wind_offset=0.0, extra=None, namespace=False, index_levels=False):
    pressure = _mass_pressure(levels_hpa) if pressure is None else pressure
    nlev = pressure.shape[0]
    mass = (nlev, NY, NX)
    fields = {
        "PRES": pressure,
        "TT": (_temperature(pressure.astype(np.float64))
               + t_offset).astype(np.float32),
        "GHT": _height(pressure.astype(np.float64)).astype(np.float32),
        "SPFH": np.full(mass, 4.0e-6, np.float32),
        "UU": np.full((nlev, NY, NX + 1), 10.0 + wind_offset, np.float32),
        "VV": np.full((nlev, NY + 1, NX), -5.0, np.float32),
        "QC": np.full(mass, 1.0e-5, np.float32),
        "QNWFA": np.full(mass, 1.0e8, np.float32),
        "PSFC": np.full((NY, NX), 85000.0, np.float32),
        "Q2": np.full((NY, NX), 0.008, np.float32),
        "SOILT": np.full((9, NY, NX), 290.0, np.float32),
    }
    fields.update(extra or {})
    levels = (np.arange(1.0, nlev + 1.0) if index_levels
              else np.asarray(levels_hpa, dtype=np.float64))
    if namespace:
        return SimpleNamespace(valid_time=VALID, levels_hpa=levels,
                               fields=fields, flag_sh_surface_fallback=False)
    return HorizontalSnapshot(valid_time=VALID, levels_hpa=levels,
                              fields=fields)


def _upper(levels_hpa=GFS_HPA, *, valid=VALID, ght_scale=1.0, rh=5.0):
    p = np.asarray(levels_hpa, dtype=np.float64) * 100.0
    n = p.size
    return HorizontalSnapshot(
        valid_time=valid, levels_hpa=np.asarray(levels_hpa, dtype=np.float64),
        fields={
            "TT": _column(p, _temperature, (NY, NX)),
            "GHT": (_column(p, _height, (NY, NX)) * np.float32(ght_scale)),
            "RH": np.full((n, NY, NX), rh, np.float32),
            "UU": np.full((n, NY, NX + 1), 12.0, np.float32),
            "VV": np.full((n, NY + 1, NX), -4.0, np.float32),
        })


def _profile(field, j=1, i=1):
    return np.asarray(field)[:, j, i].astype(np.float64)


def test_isobaric_lower_gains_the_upper_levels_above_its_top():
    lower = _lower()
    spliced, receipt = splice_upper_snapshot(lower, _upper(), upper_label="gfs")
    assert isinstance(spliced, HorizontalSnapshot)
    assert spliced.upper_source_receipt == receipt
    # 40, 30, 20, 15, 10 hPa sit above the 50 hPa lower top; bottom first.
    np.testing.assert_array_equal(
        spliced.levels_hpa, np.concatenate((HRRR_PRS_HPA,
                                            [40.0, 30.0, 20.0, 15.0, 10.0])))
    assert receipt["appended_levels_hpa"] == [40.0, 30.0, 20.0, 15.0, 10.0]
    assert receipt["composite_top_pa"] == 1000.0
    nlev = spliced.levels_hpa.size
    for name in ("TT", "GHT", "SPFH", "PRES", "QC", "QNWFA"):
        assert spliced.fields[name].shape == (nlev, NY, NX)
        assert spliced.fields[name].dtype == np.float32
    assert spliced.fields["UU"].shape == (nlev, NY, NX + 1)
    assert spliced.fields["VV"].shape == (nlev, NY + 1, NX)
    # 2-D and non-column fields ride along untouched.
    assert spliced.fields["PSFC"] is lower.fields["PSFC"]
    assert spliced.fields["SOILT"] is lower.fields["SOILT"]
    np.testing.assert_allclose(_profile(spliced.fields["PRES"])[-5:],
                               [4000.0, 3000.0, 2000.0, 1500.0, 1000.0])
    # Appended temperature/height are the upper source's own.
    np.testing.assert_allclose(_profile(spliced.fields["TT"])[-1],
                               _temperature(1000.0), rtol=1e-6)
    np.testing.assert_allclose(_profile(spliced.fields["GHT"])[-1],
                               _height(1000.0), rtol=1e-6)
    assert "gfs carries the column above 50 hPa" in receipt_sentence(receipt)


def test_the_seam_blends_linearly_in_log_pressure():
    offset = 2.0
    spliced, receipt = splice_upper_snapshot(_lower(t_offset=offset),
                                             _upper())
    tt = _profile(spliced.fields["TT"])
    levels = HRRR_PRS_HPA
    for k, hpa in enumerate(levels):
        p = hpa * 100.0
        truth = _temperature(p)
        w = min(max(math.log(DEFAULT_SEAM_BOTTOM_PA / p)
                    / math.log(DEFAULT_SEAM_BOTTOM_PA / DEFAULT_SEAM_TOP_PA),
                    0.0), 1.0)
        assert tt[k] == pytest.approx(truth + (1.0 - w) * offset, abs=2e-4)
    # 100 hPa keeps the lower source, 75 hPa is 41.5 % upper, 50 hPa all upper.
    assert receipt["lower_levels_in_seam"] == 3
    assert receipt["seam_statistics"]["max_abs_temperature_K"] == pytest.approx(
        offset, abs=1e-3)
    assert receipt["seam_statistics"]["mean_temperature_K"] == pytest.approx(
        offset, abs=1e-3)
    # Winds blend the same way on their own staggered points.
    uu = _profile(spliced.fields["UU"])
    k75 = int(np.flatnonzero(levels == 75.0)[0])
    w75 = math.log(100.0 / 75.0) / math.log(2.0)
    assert uu[k75] == pytest.approx(10.0 * (1 - w75) + 12.0 * w75, abs=1e-4)
    assert uu[0] == pytest.approx(10.0)
    assert uu[-1] == pytest.approx(12.0)


def test_moisture_and_species_follow_their_rules():
    spliced, receipt = splice_upper_snapshot(_lower(), _upper(rh=5.0))
    nlower = HRRR_PRS_HPA.size
    spfh = _profile(spliced.fields["SPFH"])
    expected = specific_humidity_from_rh(
        5.0, _temperature(1000.0), 1000.0)
    assert spfh[-1] == pytest.approx(expected, rel=1e-5)
    assert spfh[0] == pytest.approx(4.0e-6)
    qc = _profile(spliced.fields["QC"])
    assert qc[0] == pytest.approx(1.0e-5)            # below the seam
    assert qc[nlower - 1] == 0.0                     # seam top: all upper
    assert np.all(qc[nlower:] == 0.0)
    aerosol = _profile(spliced.fields["QNWFA"])
    assert np.all(aerosol[nlower:] == np.float32(1.0e8))
    assert receipt["zeroed_above"] == ["QC"]
    assert receipt["held_above"] == ["QNWFA"]
    assert "rh_to_mxrat1" in receipt["moisture"]


def test_rh_conversion_caps_the_stratosphere_as_wrf_does():
    # Warm, thin and humid: the uncapped vapour would be absurd.
    q = specific_humidity_from_rh(80.0, 270.0, 500.0)
    assert q == pytest.approx(3.0e-6 / (1.0 + 3.0e-6))
    # Below qv_max_p_safe the relation is the plain Bolton one.
    q = specific_humidity_from_rh(50.0, 280.0, 90000.0)
    rh = relative_humidity_from_specific(q, 280.0, 90000.0)
    assert rh == pytest.approx(50.0, rel=1e-9)


def test_hybrid_lower_with_3d_pressure_keeps_an_index_ladder():
    # A native-style column: pressure varies across the grid, top ~1731 Pa.
    base = np.geomspace(85000.0, 1731.475, 50)
    pressure = np.empty((50, NY, NX), np.float32)
    for j in range(NY):
        for i in range(NX):
            pressure[:, j, i] = base * (1.0 + 0.0005 * (j + i) * (base > 20000))
    upper = _upper(np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0,
                             40.0, 50.0, 70.0, 100.0]))
    lower = _lower(pressure=pressure, index_levels=True, namespace=True)
    spliced, receipt = splice_upper_snapshot(lower, upper, upper_label="gfs")
    assert isinstance(spliced, SimpleNamespace)
    assert spliced.flag_sh_surface_fallback is False
    # 15 hPa is above every column's 17.3 hPa top; 10..1 hPa follow.
    assert receipt["appended_levels_hpa"] == [15.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0]
    np.testing.assert_array_equal(spliced.levels_hpa, np.arange(1.0, 58.0))
    pres = _profile(spliced.fields["PRES"])
    assert np.all(np.diff(pres) < 0.0)
    # Lower levels above the seam top already hold the upper column.
    tt = _profile(spliced.fields["TT"])
    above = pres[:50] < DEFAULT_SEAM_TOP_PA
    np.testing.assert_allclose(tt[:50][above], _temperature(pres[:50][above]),
                               rtol=2e-5)
    assert spliced.upper_source_receipt is receipt


def test_top_first_lower_ladder_is_extended_at_its_front():
    levels = HRRR_PRS_HPA[::-1].copy()
    pressure = _mass_pressure(levels)
    spliced, receipt = splice_upper_snapshot(
        _lower(levels, pressure=pressure), _upper())
    np.testing.assert_array_equal(spliced.levels_hpa[:6],
                                  [10.0, 15.0, 20.0, 30.0, 40.0, 50.0])
    assert np.all(np.diff(_profile(spliced.fields["GHT"])) < 0.0)


def test_a_seam_disagreement_is_refused():
    with pytest.raises(UpperSpliceRefusal, match="temperature differs by up to"):
        splice_upper_snapshot(_lower(t_offset=30.0), _upper())
    with pytest.raises(UpperSpliceRefusal, match="u wind differs"):
        splice_upper_snapshot(_lower(wind_offset=60.0), _upper())
    # Geopotential (m2 s-2) mistaken for height: refused at the seam.
    with pytest.raises(UpperSpliceRefusal, match="geopotential height differs"):
        splice_upper_snapshot(_lower(), _upper(ght_scale=9.81))


def test_a_folded_composite_column_is_refused():
    # Within the seam bound but decreasing above the lower top.
    upper = _upper()
    bad = dict(upper.fields)
    ght = np.array(bad["GHT"])
    ght[0] = ght[5] - 50.0          # 10 hPa below 50 hPa in height
    bad["GHT"] = ght
    upper = HorizontalSnapshot(valid_time=VALID, levels_hpa=upper.levels_hpa,
                               fields=bad)
    with pytest.raises(UpperSpliceRefusal, match="does not increase upward"):
        splice_upper_snapshot(_lower(), upper)


def test_mismatched_times_points_and_unknown_columns_are_refused():
    with pytest.raises(UpperSpliceRefusal, match="valid at"):
        splice_upper_snapshot(_lower(), _upper(valid=datetime(2024, 7, 30, 12)))
    small = _upper()
    cropped = HorizontalSnapshot(
        valid_time=VALID, levels_hpa=small.levels_hpa,
        fields={name: value[:, :2] for name, value in small.fields.items()})
    with pytest.raises(UpperSpliceRefusal, match="other points"):
        splice_upper_snapshot(_lower(), cropped)
    unknown = np.zeros((HRRR_PRS_HPA.size, NY, NX), np.float32)
    with pytest.raises(UpperSpliceRefusal, match="'OZONE'"):
        splice_upper_snapshot(_lower(extra={"OZONE": unknown}), _upper())


def test_an_upper_source_short_of_the_lower_top_is_refused():
    lower = _lower(pressure=None)
    # Lower reaches 50 hPa; an upper stopping at 70 hPa cannot replace it.
    with pytest.raises(UpperSpliceRefusal, match="stops at 7000 Pa"):
        splice_upper_snapshot(lower, _upper(np.array([70.0, 100.0, 150.0])))


# -- the source facts and the fetch plan ------------------------------------

def test_hrrr_rows_take_gfs_above_their_column_only_when_needed():
    from woof.source_adapters import (composite_source_top_pa,
                                      fetch_model_top_pa, upper_source_for)

    assert upper_source_for("hrrr", 1000.0) == "gfs"
    assert upper_source_for("hrrr", 2000.0) is None       # inside 1731 Pa top
    assert upper_source_for("hrrr-prs", 4000.0) == "gfs"
    assert upper_source_for("hrrr-prs", 5000.0) is None
    assert upper_source_for("hrrr-prs", 1000.0, "none") is None
    assert upper_source_for("hrrr-prs", 1000.0, "gfs") == "gfs"
    assert upper_source_for("gfs", 1000.0) is None        # extends itself
    with pytest.raises(ValueError, match="takes 'gfs'"):
        upper_source_for("hrrr", 1000.0, "era5")
    assert composite_source_top_pa("hrrr", "gfs") == 1.0
    # The lower fetch itself is never asked for a model top it cannot take.
    assert fetch_model_top_pa("hrrr", 1000.0) is None
    assert fetch_model_top_pa("hrrr-prs", 1000.0) is None


def test_wizard_facts_learn_the_composite_top():
    from woof import domain_wizard

    assert domain_wizard.source_model_top_refusal("hrrr-prs", 1000.0) is None
    assert domain_wizard.source_model_top_refusal("hrrr", 300.0) is None
    assert domain_wizard.source_model_top_refusal("hrrr", 0.5) is not None
    # The default emission is unchanged by the declared lower tops.
    assert domain_wizard.emitted_model_top_pa("hrrr") == 5000.0
    assert domain_wizard.emitted_model_top_pa("hrrr-prs") == 5000.0


def test_the_upper_fetch_covers_the_lower_window_on_hourly_leads():
    from pathlib import Path
    from woof.upper_source import (plan_sentence, upper_fetch_arguments,
                                   upper_fetch_plan)

    table = {"source": "hrrr", "cycle": "2024-07-30T13", "hours": 3,
             "area": "30.61,-112.72,39.91,-100.98"}
    plan = upper_fetch_plan(table, start_time=datetime(2024, 7, 30, 13),
                            p_top_pa=1000.0)
    assert plan == {"source": "gfs", "lower_source": "hrrr",
                    "cycle": "2024-07-30T12", "forecast_start_hour": 1,
                    "hours": 3, "cadence": 1, "p_top_pa": 1000.0,
                    "area": "30.61,-112.72,39.91,-100.98"}
    arguments = upper_fetch_arguments(plan, out=Path("/data/x-upper-source"))
    assert arguments[:10] == ["--source", "gfs", "--cycle", "2024-07-30T12",
                              "--hours", "3", "--cadence", "1",
                              "--p-top-pa", "1000"]
    assert arguments[10:12] == ["--forecast-start-hour", "1"]
    assert arguments[-2:] == ["--out", "/data/x-upper-source"]
    assert "f001-f004" in plan_sentence(plan, lower_top_pa=1731.475)
    # A synoptic start takes the analysis; a top inside the column, nothing.
    plan = upper_fetch_plan(table, start_time=datetime(2024, 7, 30, 18),
                            p_top_pa=1000.0)
    assert plan["cycle"] == "2024-07-30T18" and plan["forecast_start_hour"] == 0
    assert upper_fetch_plan(table, start_time=datetime(2024, 7, 30, 13),
                            p_top_pa=5000.0) is None
    assert upper_fetch_plan({**table, "upper_source": "none"},
                            start_time=datetime(2024, 7, 30, 13),
                            p_top_pa=1000.0) is None
    # Near real time the cycle not yet usually posted is walked back.
    plan = upper_fetch_plan(table, start_time=datetime(2024, 7, 30, 13),
                            p_top_pa=1000.0, now=datetime(2024, 7, 30, 14))
    assert plan["cycle"] == "2024-07-30T06" and plan["forecast_start_hour"] == 7
    with pytest.raises(ValueError, match="whole-hour"):
        upper_fetch_plan(table, start_time=datetime(2024, 7, 30, 13, 30),
                         p_top_pa=1000.0)
    with pytest.raises(ValueError, match="no crop"):
        upper_fetch_plan({"source": "hrrr-prs", "hours": 1},
                         start_time=datetime(2024, 7, 30, 13), p_top_pa=1000.0)


def test_fetch_tables_accept_the_opt_out_and_refuse_another_pairing():
    from pathlib import Path
    from woof.fetch import validate_fetch_hints
    from woof.runplan import _fetch_arguments_from_hints

    table = {"source": "hrrr", "cycle": "2024-07-30T13", "hours": 1,
             "upper_source": "none"}
    validate_fetch_hints(table, source="hrrr")
    validate_fetch_hints({**table, "upper_source": "gfs"}, source="hrrr")
    with pytest.raises(ValueError, match="upper_source"):
        validate_fetch_hints({**table, "upper_source": "era5"}, source="hrrr")
    # The key is the chain's, never a `woof fetch` flag of the lower fetch.
    arguments = _fetch_arguments_from_hints(table, out=Path("/data/x"))
    assert not any("upper" in argument for argument in arguments)
