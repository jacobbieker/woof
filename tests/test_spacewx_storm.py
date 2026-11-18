"""The storm-time correction of the L3a ionosphere (woof.spacewx.storm).

A synthetic index record and a synthetic fit stand in for GFZ's ap and the
shipped coefficients so that the driver, the interpolation, the CLI wiring
and the kernel's foF2 factor are exercised without a download.  The graded
comparison with GNSS TEC maps and ionosondes is in docs/spacewx.md.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from woof.spacewx import indices, storm

from conftest import requires_gpu  # noqa: E402
from test_spacewx_ionosphere import WHEN, _columns, _dipole, _iono


def _record(ap_by_interval, first=datetime(2024, 5, 1)):
    """A GeoIndices whose 3-hourly ap is ``ap_by_interval`` from ``first``."""
    ap = np.asarray(ap_by_interval, dtype=np.float64)
    nday = -(-ap.size // 8)
    grid = np.full(nday * 8, np.nan)
    grid[:ap.size] = ap
    days = np.datetime64(first.date(), "D") + np.arange(nday)
    return indices.GeoIndices(days=days, kp=np.zeros((nday, 8)), ap=grid.reshape(nday, 8),
                              ap_daily=np.zeros(nday), f107=np.full(nday, 150.0),
                              source="synthetic", sha256="0")


def _fit_doc(**over):
    nodes = ([0.0, 20.0, 35.0, 50.0, 62.5, 75.0], [0.0, 4.0, 8.0, 12.0, 16.0, 20.0], [-1.0, 0.0, 1.0])
    coef = np.zeros((6, 6, 3, 2))
    coef[:3, ..., 0] = 0.4        # a positive phase at low latitudes
    coef[3:, ..., 0] = -0.8       # a negative phase at high latitudes
    coef[..., 1] = -0.1
    doc = {"tau": 0.9, "nlag": 24, "mlat_nodes": nodes[0], "lt_nodes": nodes[1], "season_nodes": nodes[2],
           "coef": coef.tolist(), "x_max": 2.5, "ratio_range": [0.2, 3.0], "record": {"note": "synthetic"}}
    doc.update(over)
    return doc


@pytest.fixture
def fit_file(tmp_path):
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(_fit_doc()))
    return path


def test_filtered_ap_reads_only_ended_intervals_and_renormalises():
    ap = np.arange(80, dtype=np.float64)       # interval i holds ap = i
    rec = _record(ap)
    tau, nlag = 0.9, 24
    when = datetime(2024, 5, 8, 13, 30)        # interval 7*8+4 = 60 (12-15 UT) is in progress
    w = (1 - tau) * tau ** np.arange(nlag)
    expect = np.dot(w, 59 - np.arange(nlag)) / w.sum()
    assert storm.filtered_ap(rec, when, tau, nlag) == pytest.approx(expect)
    # The same instant as an aware time in another zone.
    aware = (when + timedelta(hours=2)).replace(tzinfo=timezone(timedelta(hours=2)))
    assert storm.filtered_ap(rec, aware, tau, nlag) == pytest.approx(expect)
    # At the boundary the interval just ended counts, the next does not.
    assert storm.filtered_ap(rec, datetime(2024, 5, 8, 15), tau, nlag) == pytest.approx(
        np.dot(w, 60 - np.arange(nlag)) / w.sum())
    # A missing interval is left out and the weights renormalised.
    gap = ap.copy()
    gap[59] = np.nan
    got = storm.filtered_ap(_record(gap), when, tau, nlag)
    good = np.arange(nlag) != 0
    assert got == pytest.approx(np.dot(w[good], (59 - np.arange(nlag))[good]) / w[good].sum())
    # Too little of the weight present, or before the record: refused.
    with pytest.raises(storm.StormError):
        storm.filtered_ap(_record(np.full(80, np.nan)), when, tau, nlag)
    with pytest.raises(storm.StormError):
        storm.filtered_ap(rec, datetime(2024, 5, 1, 6), tau, nlag)


def test_interpolation_weights_are_a_partition_of_unity():
    lt = storm._hats([0.0, 2.0, 22.0, 23.999, 24.0, -2.0], [0, 4, 8, 12, 16, 20], period=24.0)
    np.testing.assert_allclose(lt.sum(axis=1), 1.0)
    np.testing.assert_allclose(lt[1, :2], [0.5, 0.5])
    np.testing.assert_allclose(lt[2, [5, 0]], [0.5, 0.5])            # 22 h between 20 and 24 = 0
    np.testing.assert_allclose(lt[4], lt[0])
    np.testing.assert_allclose(lt[5], lt[2])
    m = storm._hats([-5.0, 27.5, 90.0], [0.0, 20.0, 35.0, 50.0])
    np.testing.assert_allclose(m, [[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 1]])
    b = storm.basis(([0, 20, 35], [0, 12], [-1, 0, 1]), [10.0, -40.0], [6.0, 18.0], [0.3, -1.0])
    np.testing.assert_allclose(b.sum(axis=1), 1.0)


def test_ratio_follows_the_fit_and_is_clipped(fit_file):
    fit = storm.load_fit(fit_file)
    mlat = np.array([0.0, 10.0, 70.0, -70.0])
    lt = np.zeros(4)
    season = np.zeros(4)
    quiet = storm.nmf2_factor(fit, mlat, lt, season, 0.0)
    np.testing.assert_allclose(quiet, 1.0)                     # no intercept: a quiet day is 1
    x = 1.0
    r = storm.nmf2_factor(fit, mlat, lt, season, 100.0 * x)
    np.testing.assert_allclose(r, np.exp([0.4 * x - 0.1, 0.4 * x - 0.1, -0.8 * x - 0.1, -0.8 * x - 0.1]))
    huge = storm.nmf2_factor(fit, mlat, lt, season, 1e4)       # x held at x_max, then clipped
    x = fit.x_max
    np.testing.assert_allclose(huge, np.clip(np.exp([0.4 * x - 0.1 * x * x] * 2 + [-0.8 * x - 0.1 * x * x] * 2),
                                             *fit.ratio_range))
    assert huge.min() >= fit.ratio_range[0] and huge.max() <= fit.ratio_range[1]
    when = datetime(2024, 3, 20, 12)
    scale = storm.fof2_scale(fit, [0.0, 80.0], [0.0, 0.0], when, 100.0, model=_dipole())
    np.testing.assert_allclose(scale ** 2, storm.nmf2_factor(
        fit, storm.magnetic_latitude([0.0, 80.0], [0.0, 0.0], when, _dipole()), [12.0, 12.0],
        storm.season_coordinate(when, [0.0, 80.0]), 100.0))
    assert storm.season_coordinate(datetime(2024, 6, 20), [10.0])[0] == pytest.approx(1.0, abs=1e-3)
    assert storm.season_coordinate(datetime(2024, 6, 20), [-10.0])[0] == pytest.approx(-1.0, abs=1e-3)
    np.testing.assert_allclose(storm.local_time([90.0, -90.0], when), [18.0, 6.0])


def test_fit_file_is_checked_and_pinned_by_its_hash(tmp_path, fit_file):
    fit = storm.load_fit(fit_file)
    assert len(fit.sha256) == 64 and fit.tau == 0.9 and fit.coef.shape == (6, 6, 3, 2)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_fit_doc(coef=np.zeros((6, 6, 2, 2)).tolist())))
    with pytest.raises(storm.StormError):
        storm.load_fit(bad)
    bad.write_text(json.dumps({"tau": 0.9}))
    with pytest.raises(storm.StormError):
        storm.load_fit(tmp_path / "bad.json")
    doc = storm.receipt(fit, 42.0, "synthetic")
    assert doc["spacewx_storm_fit_sha256"] == fit.sha256 and doc["spacewx_storm_ap_tau"] == 42.0


def test_shipped_fit_loads_and_is_quiet_at_zero_ap():
    fit = storm.load_fit()
    assert 0.0 < fit.tau < 1.0 and fit.nlag > 0
    assert fit.record["source"].startswith("CODE") and fit.record["train_years"] == [2015, 2017, 2023]
    nodes = (fit.mlat_nodes, fit.lt_nodes, fit.season_nodes)
    mlat, lt, season = np.meshgrid(np.linspace(0, 80, 9), np.linspace(0, 23, 6), [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(storm.nmf2_factor(fit, mlat.ravel(), lt.ravel(), season.ravel(), 0.0), 1.0)
    assert storm.basis(nodes, [45.0], [3.0], [0.2]).shape == (1, fit.coef.size // 2)


@pytest.mark.gpu
@requires_gpu
def test_kernel_scales_fof2_and_ones_change_nothing(fit_file):
    import cupy as cp
    from woof.spacewx import ionosphere as ion
    lat, lon = _columns()
    io_ = _iono(lat, lon)
    plain = {k: cp.asnumpy(v) for k, v in io_.evaluate(WHEN, 150.0).items()}
    ones = {k: cp.asnumpy(v) for k, v in io_.evaluate(WHEN, 150.0, fo_scale=np.ones(lat.size)).items()}
    for key in ion.OUTPUTS:
        np.testing.assert_array_equal(plain[key], ones[key])
    fit = storm.load_fit(fit_file)
    scale = storm.fof2_scale(fit, lat, lon, WHEN, 120.0, model=_dipole())
    assert np.ptp(scale) > 0.1
    dev = {k: cp.asnumpy(v).astype(np.float64) for k, v in io_.evaluate(WHEN, 150.0, fo_scale=scale).items()}
    ref = io_.reference(WHEN, 150.0, fo_scale=scale)
    for key in ("fof2", "nmf2", "hmf2", "tec"):
        err = np.abs(dev[key] - ref[key]) / np.maximum(np.abs(ref[key]), 1e-3 * np.abs(ref[key]).max())
        assert err.max() < 1e-3, key
    np.testing.assert_allclose(dev["fof2"], plain["fof2"] * scale, rtol=1e-5)
    with pytest.raises(ion.IonosphereError):
        io_.evaluate(WHEN, 150.0, fo_scale=np.ones(lat.size + 1))


def test_cli_refuses_storm_without_an_index_record(fit_file):
    from woof.spacewx import cli
    with pytest.raises(SystemExit, match="--storm reads ap"):
        cli.main(["spacewx", "ionosphere", "--time", "2024-05-10T18:00", "--f107", "150", "--storm"])
    with pytest.raises(SystemExit, match="add --storm"):
        cli.main(["spacewx", "ionosphere", "--time", "2024-05-10T18:00", "--f107", "150", "--storm-fit", str(fit_file)])
