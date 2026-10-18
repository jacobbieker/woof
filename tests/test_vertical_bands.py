"""Height-band-biased eta ladders (``woof.vertical_bands``, ``--eta-band``).

The generator must build a stratosphere-biased column -- ~41 levels
in the lowest 2 km from a 10 m first layer, >= 100 levels between 15 and
25 km MSL, exactly 256 mass levels to a 1000 Pa (or 450 Pa) top, every
adjacent layer-depth ratio within 1.08 -- and every number it reports
must be measured off the ladder it returns.  Heights become pressures in
a reference atmosphere and pressures become eta through WRF's hybrid
coordinate; ``wrf-base`` must be the base state the model itself builds.
"""

from __future__ import annotations

import argparse
import tomllib

import numpy as np
import pytest

from woof.cli import main as cli_main
from woof.cli_numbers import eta_levels_list
from woof.domain_wizard import (VerticalColumnRefused, experiment_from_text,
                                resolve_vertical_column, validate_eta_ladder)
from woof.vertical_bands import (BandColumnRefused, HeightBand,
                                 band_biased_column, band_biased_eta,
                                 dz_profile, eta_pressure, levels_in_band,
                                 parse_band, pressure_eta,
                                 reference_height, reference_pressure,
                                 write_eta_file)
from woof.vertical_contract import validate_explicit_eta_grid

SURFACE = HeightBand(0.0, 2000.0, 41, "agl")
TROPOSPHERE = HeightBand(2070.0, 15000.0, 76)
STRATOSPHERE = HeightBand(15000.0, 25000.0, 100)


def _column(p_top_pa=1000.0, **extra):
    return band_biased_column(
        nz=256, dzbot=10.0, max_stretch=1.08, p_top_pa=p_top_pa,
        bands=(SURFACE, TROPOSPHERE, STRATOSPHERE), surface_elevation_m=70.0,
        **extra)


@pytest.fixture(scope="module")
def column():
    return _column()


def test_the_stratosphere_column_has_the_levels_it_was_asked_for(column):
    eta, profile = column.eta, column.profile
    assert eta.shape == (257,) and column.nz == 256
    assert eta[0] == 1.0 and eta[-1] == 0.0
    assert np.all(np.diff(eta) < 0.0)
    dz = profile["dz_m"]
    assert dz[0] == pytest.approx(10.0, abs=0.05)
    assert profile["max_stretch"] <= 1.08
    assert levels_in_band(profile, SURFACE) == 41
    assert levels_in_band(profile, TROPOSPHERE) >= 76
    assert levels_in_band(profile, STRATOSPHERE) >= 100
    inside = ((profile["z_mass_m"] >= 15000.0)
              & (profile["z_mass_m"] <= 25000.0))
    assert dz[inside].max() <= 101.0 and dz[inside].min() >= 85.0
    # Surface layer stretches 10 -> ~80 m (x1.08), as the column intends.
    assert dz[SURFACE.min_levels - 1] == pytest.approx(80.0, rel=0.05)
    # 1000 Pa is ~31 km in the standard atmosphere, and the sponge coarsens.
    assert profile["ztop_m"] == pytest.approx(31055.0, abs=5.0)
    assert dz[-1] > 2.5 * 100.0


def test_the_higher_top_keeps_every_band(column):
    high = _column(p_top_pa=450.0)
    assert high.nz == 256
    assert high.profile["ztop_m"] == pytest.approx(36500.0, abs=100.0)
    assert levels_in_band(high.profile, STRATOSPHERE) >= 100
    assert levels_in_band(high.profile, SURFACE) == 41
    assert high.profile["max_stretch"] <= 1.08


def test_the_ladder_passes_both_eta_validators(column):
    assert validate_eta_ladder(column.eta) == tuple(column.eta)
    # A source reaching the top (GFS extends to 1 Pa, ERA5 to 100 Pa).
    grid = validate_explicit_eta_grid(
        column.eta, nz=256, p_top=column.p_top_pa, context="test",
        source_top_pressure_pa=100.0)
    assert grid.shape == (257,)


def test_reported_profile_is_the_returned_ladder_remeasured(column):
    again = dz_profile(column.eta, p_top_pa=1000.0, surface_elevation_m=70.0)
    np.testing.assert_array_equal(again["dz_m"], column.profile["dz_m"])
    assert band_biased_eta(
        nz=256, dzbot=10.0, p_top_pa=1000.0,
        bands=(SURFACE, TROPOSPHERE, STRATOSPHERE),
        surface_elevation_m=70.0).tolist() == column.eta.tolist()
    lines = column.lines()
    assert "band 15000-25000 m MSL: " in "\n".join(lines)
    assert "max adjacent stretch" in lines[1]


def test_bands_are_minimums_and_spare_levels_go_to_the_density():
    lean = band_biased_column(nz=120, dzbot=20.0, p_top_pa=5000.0,
                              bands=(HeightBand(5000.0, 8000.0, 30),))
    band = lean.bands[0]
    assert levels_in_band(lean.profile, band) == 30
    assert lean.nz == 120


def test_bias_heights_pull_levels_toward_the_targets():
    plain = band_biased_column(nz=80, dzbot=20.0, p_top_pa=5000.0,
                               bias_weight=0.0, bias_heights=())
    biased = band_biased_column(
        nz=80, dzbot=20.0, p_top_pa=5000.0, bias_weight=3.0,
        bias_heights=np.random.default_rng(0).normal(18000.0, 600.0, 500))
    target = HeightBand(17000.0, 19000.0, 1)
    assert (levels_in_band(biased.profile, target)
            > 2 * levels_in_band(plain.profile, target))


def test_standard_atmosphere_matches_its_tables():
    for z, p in ((0.0, 101325.0), (11000.0, 22632.1), (20000.0, 5474.89),
                 (32000.0, 868.019), (47000.0, 110.906)):
        assert float(reference_pressure(z)) == pytest.approx(p, rel=1e-4)
        assert float(reference_height(p)) == pytest.approx(z, abs=1.0)
    z = np.linspace(0.0, 80000.0, 401)
    np.testing.assert_allclose(reference_height(reference_pressure(z)), z,
                               atol=1e-6)


def test_wrf_base_is_the_models_own_base_state():
    from woof.core.grid import analytic_base_pressure, make_vertical_coord
    from woof.ingest.real import _make_real_base_serial

    # Below the isotherm: the grid module's own transcription.
    for z in (0.0, 1500.0, 9000.0):
        assert float(reference_pressure(z, reference="wrf-base")) == \
            pytest.approx(analytic_base_pressure(z), rel=1e-12)
    z = np.linspace(0.0, 40000.0, 81)
    np.testing.assert_allclose(
        reference_height(reference_pressure(z, reference="wrf-base"),
                         reference="wrf-base"), z, atol=1e-6)
    # The ladder's wrf-base heights are the base state's geopotential.
    column = band_biased_column(
        nz=150, dzbot=40.0, p_top_pa=1000.0, reference="wrf-base",
        surface_elevation_m=300.0, bands=(HeightBand(15000.0, 22000.0, 40),))
    coord = make_vertical_coord(150, hybrid_opt=2, etac=0.2,
                                eta_levels=column.eta)
    base = _make_real_base_serial(coord, np.array([[300.0]]), 1000.0, 290.0,
                                  hypsometric_opt=2)
    model_z = base.phb[:, 0, 0] / 9.81
    np.testing.assert_allclose(model_z, column.profile["z_full_m"],
                               rtol=2e-4, atol=0.5)


def test_wrf_base_bands_land_low_in_a_real_stratosphere():
    """The documented caveat: the 200 K base-state isotherm puts a band's
    pressures ~1.2 km low, so a real (standard) stratosphere fills it with
    deeper layers and fewer levels than were asked for."""

    column = band_biased_column(nz=256, dzbot=10.0, p_top_pa=1000.0,
                                reference="wrf-base", bands=(STRATOSPHERE,))
    assert levels_in_band(column.profile, STRATOSPHERE) >= 100
    real = dz_profile(column.eta, p_top_pa=1000.0, reference="standard")
    assert levels_in_band(real, STRATOSPHERE) < 100


@pytest.mark.parametrize("ps", [100000.0, 101325.0, 85000.0])
def test_pressure_eta_inverts_the_hybrid_coordinate(ps):
    eta = np.linspace(1.0, 0.0, 101)
    p = eta_pressure(eta, surface_pressure_pa=ps, p_top_pa=1000.0)
    back = pressure_eta(p, surface_pressure_pa=ps, p_top_pa=1000.0)
    np.testing.assert_allclose(back, eta, atol=1e-12)


def test_written_file_reads_back_through_eta_levels(column, tmp_path):
    path = write_eta_file(tmp_path / "eta.txt", column.eta,
                          profile=column.profile, header=column.lines())
    assert eta_levels_list(str(path)) == tuple(column.eta)
    assert "# k  eta  z_full_m_msl  dz_above_m" in path.read_text()


@pytest.mark.parametrize("text, band", [
    ("15000:25000:100", HeightBand(15000.0, 25000.0, 100, "msl")),
    ("0:2000:41:agl", HeightBand(0.0, 2000.0, 41, "agl")),
    ("1e3:2e3:5:MSL", HeightBand(1000.0, 2000.0, 5, "msl")),
])
def test_band_spelling(text, band):
    assert parse_band(text) == band


@pytest.mark.parametrize("text", ["15000:25000", "2:1:5", "0:1:0",
                                  "0:1:5:asl", "a:b:c", "-10:100:4:agl"])
def test_bad_band_spelling_is_refused(text):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_band(text)


@pytest.mark.parametrize("kwargs, needle", [
    (dict(bands=(HeightBand(25000.0, 40000.0, 10),)), "outside the column"),
    (dict(bands=(HeightBand(1000.0, 2000.0, 300),)),
     "ask for 300 levels"),
    (dict(max_dz=20.0), "cannot reach the model top"),
    (dict(bands=(HeightBand(1000.0, 3000.0, 90),), max_stretch=1.02),
     "cannot reach the model top"),
    (dict(bias_heights=(50000.0,)), "none of the 1 bias heights"),
    (dict(max_stretch=1.0), "above 1"),
    (dict(reference="isa"), "not one of"),
])
def test_impossible_columns_are_refused(kwargs, needle):
    args = dict(nz=kwargs.pop("nz", 100), dzbot=10.0, p_top_pa=5000.0)
    with pytest.raises(BandColumnRefused, match=needle):
        band_biased_column(**args, **kwargs)


def test_exactly_one_top():
    with pytest.raises(BandColumnRefused, match="exactly one model top"):
        band_biased_column(nz=50)
    by_height = band_biased_column(nz=60, dzbot=20.0, ztop_m=20000.0)
    assert by_height.profile["ztop_m"] == pytest.approx(20000.0, abs=0.5)


# -- the wizard door ---------------------------------------------------------


def test_resolve_builds_the_band_column():
    column = resolve_vertical_column(
        source="gfs", p_top_pa=1000.0, dzbot=10.0, eta_band=[STRATOSPHERE])
    assert column["origin"] == "band-biased" and column["nz"] == 256
    assert column["controls"] == {}
    assert column["receipt"]["algorithm"] == "woof-band-biased-eta-v1"
    assert any("band 15000-25000 m MSL" in line for line in column["header"])


@pytest.mark.parametrize("kwargs, needle", [
    (dict(eta_max_stretch=1.1), "only --eta-band or --eta-bias-heights"),
    (dict(eta_band=[STRATOSPHERE], dzstretch_s=1.2), "--eta-max-stretch"),
    (dict(eta_band=[STRATOSPHERE], eta_levels=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0)),
     "one or the other"),
    (dict(eta_band=[HeightBand(40000.0, 45000.0, 10)]), "outside the column"),
])
def test_resolve_refuses_mixed_or_impossible_requests(kwargs, needle):
    with pytest.raises(VerticalColumnRefused, match=needle):
        resolve_vertical_column(source="gfs", p_top_pa=1000.0, **kwargs)


def test_the_wizard_emits_the_band_column(tmp_path, capsys):
    out = tmp_path / "column.toml"
    eta_file = tmp_path / "eta.txt"
    rc = cli_main([
        "domain", "--point=31.2,-94.0", "--root-dx", "12", "--card",
        "24gb", "--source", "gfs", "--cycle", "2026-04-08T12",
        "--p-top-pa", "1000", "--dzbot", "10",
        "--eta-band", "0:2000:41:agl", "--eta-band", "15000:25000:100",
        "--eta-surface-m", "70", "--eta-write", str(eta_file),
        "--out", str(out)])
    stdout = capsys.readouterr()
    assert rc == 0, stdout.err
    text = out.read_text(encoding="utf-8")
    shared = tomllib.loads(text)["shared"]
    assert shared["nz"] == 256 and len(shared["eta_levels"]) == 257
    assert shared["p_top"] == 1000.0
    assert "eta_levels band-biased" in text
    assert "band 15000-25000 m MSL: " in text
    assert "domain: band-biased eta: 256 mass levels" in stdout.out
    assert "domain: band 15000-25000 m MSL: " in stdout.out
    # The 1000 hPa standard-atmosphere note would quote other heights.
    assert "lowest full levels" not in stdout.out
    assert "# heights: standard atmosphere over ground at 70 m MSL" \
        in eta_file.read_text()
    exp = experiment_from_text(text, source=str(out))
    assert tuple(exp.vertical.eta_levels) == tuple(shared["eta_levels"])
    assert eta_levels_list(str(eta_file)) == tuple(shared["eta_levels"])
    profile = dz_profile(shared["eta_levels"], p_top_pa=1000.0,
                         surface_elevation_m=70.0)
    assert levels_in_band(profile, STRATOSPHERE) >= 100


def test_a_shallow_column_takes_bias_heights():
    # The smoothing kernel (4 sigma each side) is longer than this grid.
    column = band_biased_column(nz=30, dzbot=30.0, ztop_m=2500.0,
                                bias_heights=(500.0, 1000.0, 1500.0))
    assert column.nz == 30


def test_a_surface_band_finer_than_dzbot_names_dzbot():
    with pytest.raises(BandColumnRefused, match="at most 6 m .* lower --dzbot"):
        band_biased_column(nz=60, dzbot=10.0, p_top_pa=5000.0,
                           bands=(HeightBand(0.0, 30.0, 5, "agl"),))


def test_bias_heights_read_a_track_file(tmp_path):
    from woof.vertical_bands import parse_bias_heights

    track = tmp_path / "track.csv"
    track.write_text("time,lat,lon,height_m\n"
                     "2025-01-01T00:00:00Z,31.2,-94.0,120.5\n"
                     "2025-01-01T00:02:00Z,31.21,-94.01,  860\n"
                     "# a comment\n"
                     "2025-01-01T00:04:00Z 31.22 -94.02 1604.0\n")
    assert parse_bias_heights(str(track)) == (120.5, 860.0, 1604.0)
    headless = tmp_path / "headless.txt"
    headless.write_text("2025-01-01T00:00:00Z 31.2 -94.0 300\n"
                        "2025-01-01T00:02:00Z 31.2 -94.0 400\n")
    assert parse_bias_heights(str(headless)) == (300.0, 400.0)
    plain = tmp_path / "heights.txt"
    plain.write_text("100\n200, 300\n")
    assert parse_bias_heights(str(plain)) == (100.0, 200.0, 300.0)
    assert parse_bias_heights("5,6") == (5.0, 6.0)
    bad = tmp_path / "bad.csv"
    bad.write_text("t,h\n0,10\n1,high\n")
    with pytest.raises(argparse.ArgumentTypeError, match="row 3"):
        parse_bias_heights(str(bad))
