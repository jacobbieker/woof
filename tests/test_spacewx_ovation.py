"""woof.spacewx.ovation: OVATION Prime 2010 energy flux and Knipp 2004 Joule power.

The coefficient tables here are synthetic.  The port is checked against the
distribution's own test outputs only when the pinned zip is in the
space-weather cache.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import io
from pathlib import Path
import zipfile

import numpy as np
import pytest

from woof.spacewx import assets, ovation as ov, solarwind as sw

from conftest import requires_gpu  # noqa: E402

NORTH = slice(80, 160)


def _premodel(root: Path, *, no_fit_cell=(0, 80)) -> Path:
    """Synthetic premodel files: energy b1 = 0.1 (t + 1), b2 = 1e-4 (s + 1) in the
    northern bins only; probability fit 0.5 except at ``no_fit_cell``."""
    d = root / "premodel"
    d.mkdir(parents=True)
    head = "    1984       1    2005     365       17552       3\n"
    ii, jj = np.meshgrid(np.arange(ov.NMLT), np.arange(ov.NMLAT), indexing="ij")
    north = (jj >= 80).ravel()
    for s, season in enumerate(ov.SEASONS):
        for t, kind in enumerate(ov.TYPES):
            for suffix, (one, two) in (("", (0.1 * (t + 1), 1e-4 * (s + 1))), ("_n", (1e7 * (t + 1), 1e3))):
                b1 = np.where(north, one, 0.0)
                b2 = np.where(north, two, 0.0)
                rows = "".join(f"{i} {j} {a:.9g} {b:.9g} 0.5\n" for i, j, a, b in zip(ii.ravel(), jj.ravel(), b1, b2))
                (d / f"{season}_{kind}{suffix}.txt").write_text(head + rows)
            if t < 3:
                b1p = np.full((ov.NMLT, ov.NMLAT), 0.5)
                b1p[no_fit_cell] = 0.0
                fit = "".join(f"{a:.9g} 0.0\n" for a in b1p.ravel())
                table = "0.25\n" * (ov.NMLT * ov.NMLAT * ov.NDF)   # never used (see the module notes)
                (d / f"{season}_prob_b_{kind}.txt").write_text(head + fit + table)
    return root


@pytest.fixture(scope="module")
def coef(tmp_path_factory):
    return ov.read_coefficients(_premodel(tmp_path_factory.mktemp("ovation")))


# ---------------------------------------------------------------- drivers

def test_season_weights_blend_and_peak_on_the_season_days():
    for doy, peak in ((354, 0), (79, 1), (171, 2), (263, 3)):
        w = ov.season_weights(doy)
        assert w[peak] == pytest.approx(1.0) and sum(w) == pytest.approx(1.0)
    for doy in range(1, 367):
        w = ov.season_weights(doy)
        assert sum(w) == pytest.approx(1.0) and min(w) >= 0.0
    assert ov.season_weights(125) == pytest.approx((0.0, 0.5, 0.5, 0.0))


def test_southern_hemisphere_takes_the_season_half_a_year_on():
    june = datetime(2024, 6, 20)
    assert ov.hemisphere_doy(june, "north") == 172
    assert ov.season_weights(ov.hemisphere_doy(june, "south"))[0] > 0.95       # winter
    assert ov.hemisphere_doy(datetime(2024, 12, 31), "south") == (366 + 182 - 1) % 365 + 1
    with pytest.raises(ov.OvationError):
        ov.hemisphere_doy(june, "east")


def _wind(n=8, start=datetime(2024, 5, 10, 0), **over):
    t = np.array([np.datetime64(start, "h") + np.timedelta64(k, "h") for k in range(n)])
    cols = {"by": np.full(n, 3.0), "bz": np.full(n, -5.0), "density": np.full(n, 5.0),
            "speed": np.full(n, 450.0)}
    for key, (i, v) in over.items():
        cols[key][i] = v
    nan = np.full(n, np.nan)
    return sw.SolarWind(hours=t, kp=nan, al=nan.copy(), source="synthetic", sha256="0" * 64, **cols)


def test_coupling_is_the_weighted_four_hour_newell_mean():
    t0 = datetime(2024, 5, 10, 0)
    wind = _wind(speed=(5, 600.0))
    e450 = float(sw.newell_coupling(450.0, 3.0, -5.0))
    e600 = float(sw.newell_coupling(600.0, 3.0, -5.0))
    # On the hour, the current hour's weight is sqrt(0) = 0: only hours 4, 3 and 2 count.
    assert ov.coupling(wind, t0 + timedelta(hours=5)) == pytest.approx(e450)
    half = t0 + timedelta(hours=5, minutes=30)
    w = np.array([np.sqrt(0.5), 0.65, 0.65 ** 2, 0.65 ** 3])
    want = (w[0] * e600 + w[1:].sum() * e450) / w.sum()
    assert ov.coupling(wind, half) == pytest.approx(want)


def test_coupling_needs_three_good_hours():
    t0 = datetime(2024, 5, 10, 0)
    wind = _wind(bz=(3, np.nan), speed=(4, np.nan))
    with pytest.raises(ov.OvationError, match="fewer than 3"):
        ov.coupling(wind, t0 + timedelta(hours=5, minutes=10))
    # Outside the record counts as missing.
    with pytest.raises(ov.OvationError):
        ov.coupling(_wind(), t0 + timedelta(hours=1, minutes=10))
    assert ov.coupling(_wind(bz=(3, np.nan)), t0 + timedelta(hours=5, minutes=10)) > 0.0


# ---------------------------------------------------------------- the model

def test_flux_is_the_season_regression_times_the_fitted_probability(coef):
    summer = 171
    g = ov.fit_grid(coef, 1000.0, summer, 0)
    assert g[5, 100] == pytest.approx((1000.0 * 3e-4 + 0.1) * 0.5)
    assert g[0, 80] == 0.0                                  # no probability fit: p = 0
    assert np.all(g[:, :80] == 0.0)
    ions = ov.fit_grid(coef, 1000.0, summer, 3)
    assert ions[0, 80] == pytest.approx(1000.0 * 3e-4 + 0.4)      # ions: p = 1
    number = ov.fit_grid(coef, 1000.0, summer, 1, number=True)
    assert number[5, 100] == pytest.approx((1000.0 * 1e3 + 2e7) * 0.5)
    spring_summer = ov.fit_grid(coef, 1000.0, 125, 0)
    assert spring_summer[5, 100] == pytest.approx(0.5 * (1000.0 * 2.5e-4 + 0.1))
    with pytest.raises(ov.OvationError):
        ov.fit_grid(coef, float("nan"), summer, 0)


def test_the_idl_outlier_clamps_are_kept(coef):
    summer = 171
    assert ov.fit_grid(coef, 1.0e5, summer, 0)[5, 100] == 0.5     # 15.05 > 10
    assert ov.fit_grid(coef, 4.0e4, summer, 0)[5, 100] == 5.0     # 6.05 in (5, 10]
    assert ov.fit_grid(coef, 1.0e4, summer, 3)[5, 100] == 2.0     # ions 3.4 in (2, 4]
    assert ov.fit_grid(coef, 2.0e4, summer, 3)[5, 100] == 0.25    # ions 6.4 > 4
    x = np.array([-1.0, 3.0e9, 3.0e10])
    np.testing.assert_array_equal(ov._clamp(x, 0, True), [-1.0, 1.0e9, 0.0])
    np.testing.assert_array_equal(ov._clamp(np.array([2e8, 6e8]), 3, True), [1.0e8, 0.0])


def test_hemispheres_combine_as_draw_je_does():
    g = np.zeros((ov.NMLT, ov.NMLAT))
    g[0, 80], g[0, 0] = 2.0, 4.0           # both: the mean
    g[1, 81] = 3.0                         # one zero: the sum
    g[2, 1] = 5.0
    c = ov.combine_hemispheres(g)
    assert c.shape == (ov.NMLT, 80)
    assert (c[0, 0], c[1, 1], c[2, 1]) == (3.0, 3.0, 5.0)


def test_hemispheric_power_integrates_without_the_dayside_strip():
    ones = np.ones((ov.NMLT, 80))
    lat = 50.001 + 0.5 * np.arange(80)
    band = 2 * np.pi * ov.EARTH_RADIUS_KM ** 2 * np.abs(np.cos(np.deg2rad(90 - lat)) - np.cos(np.deg2rad(90 - lat - 0.5)))
    mlt = 0.001 + 0.25 * np.arange(ov.NMLT)
    day = ((mlt >= 7) & (mlt <= 19)).sum() / ov.NMLT
    want = 1e-6 * (band.sum() - day * band[lat <= 63.0].sum())
    assert ov.hemispheric_power(ones) == pytest.approx(want, rel=1e-12)
    # 1 erg/cm2/s over the whole cap poleward of 50 deg is about 60 GW
    assert 1e-6 * band.sum() == pytest.approx(59.7, rel=0.01)


def test_columns_take_the_bin_of_their_mlat_and_mlt():
    g = {"x": np.arange(ov.NMLT * 80, dtype=float).reshape(ov.NMLT, 80)}
    mlat = np.array([49.99, 50.2, -50.2, 89.99, 90.0, 70.6, 70.6])
    mlt = np.array([0.0, 0.1, 0.1, 23.99, 12.0, 24.0, 6.30])
    got = ov.on_columns(g, mlat, mlt)["x"]
    np.testing.assert_array_equal(got, [0.0, 0.0, 0.0, 95 * 80 + 79, 48 * 80 + 79, 41.0, 25 * 80 + 41])
    np.testing.assert_allclose(ov.mean_energy_kev([1.602176634e-9 * 2e8, 1.0], [2e8, 0.0]), [1.0, 0.0])


def test_reads_the_zip_and_refuses_damaged_tables(tmp_path):
    root = _premodel(tmp_path / "dir")
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as z:
        for f in sorted((root / "premodel").iterdir()):
            z.write(f, f"ovation-prime_2.3.0/premodel/{f.name}")
    path = tmp_path / "op.zip"
    path.write_bytes(blob.getvalue())
    a, b = ov.read_coefficients(path), ov.read_coefficients(root)
    np.testing.assert_array_equal(a.b2e, b.b2e)
    np.testing.assert_array_equal(a.b1p, b.b1p)
    bad = root / "premodel" / "fall_mono.txt"
    bad.write_text("\n".join(bad.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ov.OvationError, match="fall_mono"):
        ov.read_coefficients(root)
    (root / "premodel" / "fall_mono.txt").unlink()
    with pytest.raises(OSError):
        ov.read_coefficients(root)


def test_energy_columns_pick_the_hemisphere_and_derive_the_mean_energy(coef):
    when = datetime(2024, 6, 20, 22, 10)
    grids = ov.hemisphere_grids(coef, 1000.0, when)
    cols = ov.energy_columns(grids, np.array([70.2, -70.2, 40.0]), np.array([3.1, 3.1, 3.1]))
    assert set(cols) == set(ov.COLUMN_FIELDS)
    i, j = 12, 40
    for k, h in enumerate(("north", "south")):
        g = grids[h]
        e = g["eflux_diff"][i, j] + g["eflux_mono"][i, j] + g["eflux_wave"][i, j]
        assert cols["eflux"][k] == pytest.approx(e)
        assert cols["eflux_ion"][k] == g["eflux_ion"][i, j]
    # June: the south's season is winter, so the two hemispheres differ.
    assert cols["eflux"][0] != cols["eflux"][1]
    assert cols["eflux"][2] == 0.0 and cols["mean_energy"][2] == 0.0
    assert ov.electron_power(grids["north"]) > 0.0


def test_only_the_pinned_zip_counts_as_pinned(coef):
    assert not ov.is_pinned(coef)
    assert ov.is_pinned(ov.Coefficients(*([None] * 6), sha256=ov.OVATION_ZIP.sha256, source="x"))


def test_cli_energy_keeps_the_good_times_around_a_gap(coef):
    from woof.spacewx import cli
    t0 = datetime(2024, 5, 10, 0)
    wind = _wind(n=10, bz=(5, np.nan), speed=(6, np.nan))
    good = cli._energy(coef, wind, t0 + timedelta(hours=5, minutes=0))
    assert good.grids is not None and good.scalars["ovation_dphidt"] > 0.0
    assert good.scalars["hp_north_gw"] > 0.0 and good.scalars["hp_south_gw"] > 0.0
    assert np.isnan(good.scalars["joule_north_gw"])          # synthetic wind: no PC(N) or Dst
    gap = cli._energy(coef, wind, t0 + timedelta(hours=7, minutes=30))   # hours 4-7: two of them bad
    assert gap.grids is None
    assert all(np.isnan(v) for v in gap.scalars.values())


def test_cli_refuses_ovation_without_energy():
    from types import SimpleNamespace
    from woof.spacewx import cli
    args = SimpleNamespace(observed_al=False, solar_wind="omni", ovation="op.zip", energy=False, kp=None)
    with pytest.raises(SystemExit, match="--ovation"):
        cli._aurora_activity(args, [datetime(2024, 5, 10)])
    args = SimpleNamespace(observed_al=False, solar_wind=None, ovation=None, energy=True, kp=None)
    with pytest.raises(SystemExit, match="--solar-wind"):
        cli._aurora_activity(args, [datetime(2024, 5, 10)])


# ---------------------------------------------------------------- Joule heating

def test_knipp_seasons_and_fit():
    assert ov.knipp_season(datetime(2024, 10, 21)) == "winter"
    assert ov.knipp_season(datetime(2024, 2, 20)) == "winter"
    assert ov.knipp_season(datetime(2024, 2, 21)) == "equinox"
    assert ov.knipp_season(datetime(2024, 4, 21)) == "summer"
    assert ov.knipp_season(datetime(2024, 8, 21)) == "equinox"
    june = datetime(2024, 6, 1)
    assert ov.joule_power(2.0, -100.0, june) == pytest.approx(29.27 * 2 + 8.18 * 4 - 0.04 * 100 + 0.0126 * 1e4)
    assert ov.joule_power(-2.0, 100.0, june) == ov.joule_power(2.0, -100.0, june)
    assert ov.joule_power(1.0, 0.0, june, season="annual") == pytest.approx(24.89 + 3.41)
    with pytest.raises(ov.OvationError):
        ov.joule_power(float("nan"), 0.0, june)


def test_omni2_carries_dst_and_pcn(tmp_path):
    from test_spacewx_aurora import _omni_row
    t0 = datetime(2024, 5, 10, 0)
    rows = []
    for h in range(3):
        w = _omni_row(t0 + timedelta(hours=h)).split()
        w[40], w[51] = ("-85", "2.4") if h != 1 else ("99999", "999.9")
        rows.append(" ".join(w))
    path = tmp_path / "omni2_2024.dat"
    path.write_text("\n".join(rows) + "\n")
    wind = sw.read_omni2(path)
    assert wind.dst_at(t0) == -85.0 and wind.pcn_at(t0) == pytest.approx(2.4)
    assert np.isnan(wind.dst_at(t0 + timedelta(hours=1))) and np.isnan(wind.pcn_at(t0 + timedelta(hours=1)))
    assert np.all(np.isnan(_wind().dst)) and np.all(np.isnan(_wind().pcn))


# ---------------------------------------------------------------- the real tables

def _cached_zip():
    path = assets.cache_root() / ov.OVATION_ZIP.subdir / ov.OVATION_ZIP.name
    if not path.is_file():
        pytest.skip(f"OVATION Prime 2.3.0 not in the space-weather cache ({assets.cache_root()})")
    return path


def test_port_reproduces_the_distributions_test_outputs():
    """The IDL's text outputs for the storm of 24-25 Oct 2011 (test mode 1, its own
    OMNI2 file), through the validation tool: every energy-flux cell to the
    printed 0.01 and its hemispheric power to 0.1 %, and the number fluxes to
    0.3 % of their peak but for the two outputs the docs explain."""
    from types import SimpleNamespace
    from tools.spacewx_ovation_validate import testdata
    rows = testdata(SimpleNamespace(zip=_cached_zip()))["outputs"]
    energy = [r for r in rows if not r["number"]]
    assert len(energy) == 63 and len(rows) == 123
    for r in energy:
        assert r["max_abs_diff"] < 0.0201, r["file"]
        assert r["hp_gw"] == pytest.approx(r["hp_ref_gw"], rel=1e-3), r["file"]
    known = {"ov_diff_flux_2011_1024_1200.txt", "ov_wave_flux_2011_1025_0300.txt"}
    for r in rows:
        if r["number"] and r["file"] not in known:
            assert r["max_abs_diff"] < 3e-3 * r["max_ref"], r["file"]


# ---------------------------------------------------------------- on the GPU

@pytest.mark.gpu
@requires_gpu
def test_energy_columns_on_the_device_frame_match_the_float64_frame(coef):
    from test_spacewx_ionosphere import _dipole
    import cupy as cp
    from woof.spacewx import aurora as au
    when = datetime(2024, 6, 20, 22, 10)
    lat, lon = np.meshgrid(np.arange(-89.5, 90.0, 1.0), np.arange(-180.0, 180.0, 2.0), indexing="ij")
    oval = au.Aurora(lat, lon, when=when, igrf_model=_dipole())
    dev = oval.evaluate(when, 300.0)
    mlat, mlt = (cp.asnumpy(dev[k]).astype(np.float64) for k in ("mlat", "mlt"))
    ref = oval.reference(when, 300.0)
    grids = ov.hemisphere_grids(coef, 5000.0, when)
    got = ov.energy_columns(grids, mlat.reshape(-1), mlt.reshape(-1))
    want = ov.energy_columns(grids, ref["mlat"], ref["mlt"])
    assert set(got) == set(ov.COLUMN_FIELDS)
    # Columns that sit on a bin edge may fall either side of it in float32.
    edge = (np.abs((np.abs(ref["mlat"]) * 2) - np.round(np.abs(ref["mlat"]) * 2)) < 1e-3) | (
        np.abs(ref["mlt"] * 4 - np.round(ref["mlt"] * 4)) < 1e-3)
    for key in ov.COLUMN_FIELDS:
        assert np.array_equal(got[key][~edge], want[key][~edge]), key
    # Synthetic tables: only the north bins carry flux, and the south reads the
    # same combined grid, so both polar caps light up.
    assert got["eflux"][lat.reshape(-1) > 70].max() > 0 and got["eflux"][lat.reshape(-1) < -70].max() > 0
    assert np.all(got["eflux"][np.abs(lat.reshape(-1)) < 40] == 0.0)


def test_validation_tool_reads_poes_level2_text(tmp_path):
    from tools.spacewx_ovation_validate import read_poes
    head = ("year mo dy hr mi second  dayofyear  sslat  sslon  folat  folon    lval    mlt   pas0  pas90"
            "       ted    echar    pchar    econtr")
    rows = ["2011 10 24 00 00 28.306 297.000328  70.0 10.0  71.0 12.0  6.0 300.0  1.0 2.0  0.5000 5000.0 3000.0 0.8000",
            "2011 10 24 00 00 44.306 297.000513  70.5 10.0  71.5 12.0  6.0 300.0  1.0 2.0 -999.0 -999.0 -999.0 -999.0",
            "2011 10 24 00 01 00.306 297.000698  71.0 10.0  72.0 12.0  6.0 300.0  1.0 2.0  2.0000 5000.0 3000.0 0.2500"]
    path = tmp_path / "poes_n19_20111024.txt"
    path.write_text("\n".join([head] + rows) + "\n")
    rec = read_poes(path)
    np.testing.assert_allclose(rec["eflux"], [0.4, 0.5])
    np.testing.assert_allclose(rec["total"], [0.5, 2.0])
    np.testing.assert_allclose(rec["lat"], [71.0, 72.0])
    assert rec["time"][1] == datetime(2011, 10, 24, 0, 1, 0, 306000)
    bad = tmp_path / "other.txt"
    bad.write_text("a b c\n1 2 3\n")
    with pytest.raises(ValueError, match="POES"):
        read_poes(bad)
    # A header-only day reads as no records.
    empty = tmp_path / "poes_n19_20111025.txt"
    empty.write_text(head + "\n")
    assert read_poes(empty)["time"].size == 0
