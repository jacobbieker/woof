"""solar_ssi_opt: the day's solar spectrum on RRTMG SW's 14 bands (woof.spacewx).

The band integration, the shipped reference, the factor, the series and the
admission run on the CPU; on a card (CUDA or Metal) the batched SW engine's
``band_factor`` is checked to be a no-op at None and at ones, to scale the
output linearly, and to move the absorbed ultraviolet where the bands say, and
the driver is checked to set the factor on legacy RRTMG's and RRTMG-K's
shortwave before the call.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime
import inspect

import numpy as np
import pytest

from woof.config import ConfigurationRefused, RunConfig, validate_run_config
from woof.spacewx import solar_spectrum as ss
from woof.spacewx.solar_irradiance import SolarForcingError

from conftest import requires_gpu  # noqa: E402

LEGACY = "rrtmg_legacy"
REF = np.asarray(ss.REF_BAND_FRACTION)


def _cfg(**overrides):
    values = dict(nx=6, ny=4, nz=5, dx=3000.0, dy=3000.0, ztop=20000.0, dt=20.0,
                  run_seconds=3600.0, ra_physics=4, ra_rrtmg_variant=LEGACY)
    values.update(overrides)
    return RunConfig(**values)


# --- the bands and the reference --------------------------------------------------------

def test_the_bands_are_rrtmg_sws():
    from woof.core import rrtmg_sw
    assert len(ss.BANDS_CM1) == rrtmg_sw.NBNDSW == 14
    tables = rrtmg_sw.load_sw_tables()
    np.testing.assert_array_equal(np.asarray(tables.wavenum1, np.float64), [b[0] for b in ss.BANDS_CM1])
    np.testing.assert_array_equal(np.asarray(tables.wavenum2, np.float64), [b[1] for b in ss.BANDS_CM1])
    # Contiguous from 820 to 50000 cm-1 with band 29 (820-2600) last.
    lows = sorted(ss.BANDS_CM1)
    assert lows[0] == (820.0, 2600.0) and lows[-1][1] == 50000.0
    assert all(a[1] == b[0] for a, b in zip(lows, lows[1:]))


def test_band_integration_takes_partial_bins_pro_rata():
    # One 10 nm bin straddling the 441.5 nm edge of bands 26 (345-441.5 nm) and 25.
    edge = ss.BANDS_NM[10][1]
    assert edge == pytest.approx(1.0e7 / 22650.0) == ss.BANDS_NM[9][0]
    out = ss.band_irradiance([edge - 2.5], [10.0], np.array([1.0]))
    assert out[10] == pytest.approx(7.5) and out[9] == pytest.approx(2.5)
    assert out.sum() == pytest.approx(10.0)
    # Below 200 nm and beyond 12.2 um is outside every band.
    assert ss.band_irradiance([150.0, 20000.0], [10.0, 10.0], np.ones(2)).sum() == 0.0


def test_the_shipped_reference_is_the_cycle_24_share():
    assert REF.shape == (14,) and np.all(REF > 0)
    # NRLSSI2 puts all but about 0.04 % of the TSI in 200 nm - 12.2 um.
    assert 0.999 < REF.sum() < 1.0
    # The visible bands carry the most, the 200-263 nm band the least.
    assert REF.argmax() in (7, 9) and REF.argmin() == 12


def test_the_factor_is_one_at_the_reference_and_keeps_the_total():
    np.testing.assert_allclose(ss.band_factor(REF), 1.0, rtol=1e-6)
    uv = REF.copy()
    uv[12] *= 1.05                                  # 5 % more 200-263 nm
    f = ss.band_factor(uv).astype(np.float64)
    assert np.dot(REF, f) == pytest.approx(REF.sum(), rel=1e-6)
    assert f[12] == pytest.approx(1.05 / (1 + 0.05 * REF[12] / REF.sum()), rel=1e-6)
    assert np.all(np.delete(f, 12) < 1.0)
    assert ss.band_factor(uv).dtype == np.float32


# --- the series -------------------------------------------------------------------------

def _ssi_file(path, *, days=3, first=datetime(2024, 5, 10), uv=(1.0, 1.0, 1.0)):
    """A NOAA CDR-layout SSI file with one bin per RRTMG band: the reference shares,
    the 200-263 nm band times ``uv`` per day."""
    netCDF4 = pytest.importorskip("netCDF4")
    lo = np.array([b[0] for b in ss.BANDS_NM])
    hi = np.array([b[1] for b in ss.BANDS_NM])
    order = np.argsort(lo)
    centre, width = (0.5 * (lo + hi))[order], (hi - lo)[order]
    tsi = 1361.0
    start = (first - datetime(1610, 1, 1)).days
    ssi = np.empty((days, 14))
    for d in range(days):
        share = REF.copy()
        share[12] *= uv[d]
        ssi[d] = (tsi * share)[order] / width
    with netCDF4.Dataset(path, "w") as data:
        data.createDimension("time", days)
        data.createDimension("wavelength", 14)
        data.createDimension("bounds", 2)
        t = data.createVariable("time", "f8", ("time",))
        t.units = "days since 1610-01-01 00:00:00"
        t[:] = start + np.arange(days)
        b = data.createVariable("time_bnds", "f8", ("time", "bounds"))
        b[:] = np.stack([start + np.arange(days), start + 1 + np.arange(days)], axis=1)
        data.createVariable("wavelength", "f4", ("wavelength",))[:] = centre
        data.createVariable("Wavelength_Band_Width", "f4", ("wavelength",))[:] = width
        data.createVariable("SSI", "f4", ("time", "wavelength"))[:] = ssi
        data.createVariable("TSI", "f4", ("time",))[:] = np.full(days, tsi)
    return path


def test_noaa_cdr_layout_reads_into_band_shares(tmp_path):
    path = _ssi_file(tmp_path / "ssi.nc", uv=(1.0, 1.1, 1.2))
    series = ss.read_series(path)
    np.testing.assert_allclose(series.fractions_at(datetime(2024, 5, 10, 12)), REF, rtol=1e-5)
    mid = series.fractions_at(datetime(2024, 5, 11, 0))
    assert mid[12] == pytest.approx(REF[12] * 1.05, rel=1e-5)
    assert len(series.sha256) == 64
    with pytest.raises(SolarForcingError, match="outside"):
        series.fractions_at(datetime(2024, 5, 13, 0, 1))


def test_a_gap_and_a_missing_variable_are_refused(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    a = _ssi_file(tmp_path / "a.nc", days=1)
    b = _ssi_file(tmp_path / "b.nc", days=1, first=datetime(2024, 5, 13))
    with pytest.raises(SolarForcingError, match="gap"):
        ss.read_series([a, b]).fractions_at(datetime(2024, 5, 11, 12))
    with pytest.raises(SolarForcingError, match="repeat"):
        ss.read_series([a, a])
    with netCDF4.Dataset(tmp_path / "bad.nc", "w") as data:
        data.createDimension("time", 1)
        data.createVariable("time", "f8", ("time",))[:] = [0.0]
    with pytest.raises(SolarForcingError, match="missing"):
        ss.read_series(tmp_path / "bad.nc")


def test_out_of_band_fill_keeps_the_day_and_a_dead_band_is_refused(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    # The test file with one extra bin below 200 nm, a fill on day 1.
    src = netCDF4.Dataset(_ssi_file(tmp_path / "ssi.nc"))
    n = src.dimensions["wavelength"].size
    out = tmp_path / "wide.nc"
    with netCDF4.Dataset(out, "w") as data:
        data.createDimension("time", 3)
        data.createDimension("bounds", 2)
        data.createDimension("wavelength", n + 1)
        t = data.createVariable("time", "f8", ("time",))
        t.units = src["time"].units
        t[:] = src["time"][:]
        data.createVariable("time_bnds", "f8", ("time", "bounds"))[:] = src["time_bnds"][:]
        data.createVariable("wavelength", "f4", ("wavelength",))[:] = np.r_[150.0, src["wavelength"][:]]
        data.createVariable("Wavelength_Band_Width", "f4", ("wavelength",))[:] = \
            np.r_[10.0, src["Wavelength_Band_Width"][:]]
        ssi = np.c_[np.ones(3), src["SSI"][:]]
        ssi[1, 0] = np.nan
        data.createVariable("SSI", "f4", ("time", "wavelength"))[:] = ssi
        data.createVariable("TSI", "f4", ("time",))[:] = src["TSI"][:]
    src.close()
    assert ss.read_series(out).days.size == 3
    with pytest.raises(SolarForcingError, match="no irradiance"):
        ss.SsiSeries(np.array([1.0]), np.zeros((1, 14)), np.array([1361.0]), "x", "")


def test_nrlssi2_fetch_takes_the_newest_final_file_once(tmp_path):
    import io
    payload = _ssi_file(tmp_path / "served.nc").read_bytes()
    calls = []
    listing = ('<a href="ssi_v03r00_daily_s20240101_e20241231_c20250101.nc">'
               '<a href="ssi_v03r00_daily_s20240101_e20241231_c20250221.nc">'
               '<a href="ssi_v03r00-preliminary_daily_s20250101_e20250331_c20250405.nc">')

    def opener(url):
        calls.append(url)
        return io.BytesIO(listing.encode() if url.endswith("/") else payload)

    cache = tmp_path / "cache"
    paths = ss.fetch_nrlssi2([2024], cache=cache, opener=opener)
    assert [p.name for p in paths] == ["ssi_v03r00_daily_s20240101_e20241231_c20250221.nc"]
    assert ss.fetch_nrlssi2([2024], cache=cache, opener=opener) == paths and len(calls) == 2
    with pytest.raises(SolarForcingError, match="no complete NOAA CDR SSI"):
        ss.fetch_nrlssi2([2025], cache=cache, opener=opener)

    def truncated(url):
        return io.BytesIO(listing.encode() if url.endswith("/") else payload[: len(payload) // 3])

    other = tmp_path / "other"
    with pytest.raises(SolarForcingError, match="does not read"):
        ss.fetch_nrlssi2([2024], cache=other, opener=truncated)
    assert list(other.iterdir()) == []


# --- admission ----------------------------------------------------------------------------

def test_defaults_are_off_and_admitted():
    cfg = _cfg()
    assert (cfg.solar_ssi_opt, cfg.solar_ssi_source) == (0, "")
    validate_run_config(cfg)
    validate_run_config(_cfg(solar_ssi_opt=1, solar_ssi_source="nrlssi2"))
    validate_run_config(_cfg(solar_ssi_opt=1, solar_ssi_source="nrlssi2", ra_physics=0,
                             ra_lw_physics=14, ra_sw_physics=14, ra_rrtmg_variant="rte-rrtmgp",
                             moist=True, mp_physics=10, o3input=2))


@pytest.mark.parametrize("overrides, match", [
    (dict(solar_ssi_opt=2), "solar_ssi_opt"),
    (dict(solar_ssi_opt=True), "solar_ssi_opt"),
    (dict(solar_ssi_opt=1), "needs solar_ssi_source"),
    (dict(solar_ssi_source="nrlssi2"), "would do nothing"),
    (dict(solar_ssi_opt=1, solar_ssi_source="nrlssi2", ra_rrtmg_variant="rte-rrtmgp"), "RRTMGP"),
    (dict(solar_ssi_opt=1, solar_ssi_source="nrlssi2", ra_physics=1), "14 bands"),
    (dict(solar_ssi_opt=1, solar_ssi_source="nrlssi2", ra_physics=0, ra_sw_physics=0,
          ra_lw_physics=0), "14 bands"),
    (dict(solar_ssi_opt=1, solar_ssi_source="nrlssi2", dfi_opt=3), "dfi_opt"),
    (dict(solar_ssi_opt=1, solar_ssi_source="nrlssi2", swint_opt=2), "FARMS"),
])
def test_refusals_name_the_breakage(overrides, match):
    from woof.config import _validate_solar_ssi
    with pytest.raises(ConfigurationRefused, match=match):
        _validate_solar_ssi(_cfg(**overrides))


def test_the_keys_reach_the_echo_only_when_on():
    from woof.io import restart
    values = dataclasses.asdict(_cfg())
    restart._drop_default_off_run_keys(values)
    assert not {"solar_ssi_opt", "solar_ssi_source"} & set(values)
    values = dataclasses.asdict(_cfg(solar_ssi_opt=1, solar_ssi_source="nrlssi2"))
    restart._drop_default_off_run_keys(values)
    assert values["solar_ssi_opt"] == 1 and values["solar_ssi_source"] == "nrlssi2"


def test_hex_refuses_it():
    from woof.hex.physics_suites import PhysicsSelectionRefusal, resolve_physics
    with pytest.raises(PhysicsSelectionRefusal, match="solar_ssi_opt"):
        resolve_physics("arwen_mpas_column", {"solar_ssi_opt": 1}, mesh="x1.40962")


def test_the_factor_reaches_a_wrapped_or_swapped_shortwave(tmp_path):
    """A partial nest's extended-column wrapper forwards attributes to its inner
    adapter; the factor must land on the inner one.  A callable swapped in after
    start-up (a batched ensemble's member radiation) is the one set."""
    from types import SimpleNamespace
    from woof.core.partial_nest_physics import ExtendedColumnRadiation
    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation

    class Leaf(RRTMGLegacyRadiation):
        def __init__(self):
            self.shortwave = True

    series = ss.read_series(_ssi_file(tmp_path / "ssi.nc", uv=(1.2, 1.2, 1.2)))
    sv = ss.SpectralVariability(series, datetime(2024, 5, 10, 12))
    inner = Leaf()
    wrapper = ExtendedColumnRadiation(inner, SimpleNamespace())
    sv.apply(wrapper, 0.0)
    assert "sw_band_factor" not in vars(wrapper) and inner.sw_band_factor[12] > 1.19
    swapped = Leaf()
    sv.apply(swapped, 3600.0)
    assert swapped.sw_band_factor == inner.sw_band_factor


def test_a_shortwave_without_rrtmgs_bands_is_refused():
    from types import SimpleNamespace
    with pytest.raises(SolarForcingError, match="does not take a band factor"):
        ss.SpectralVariability.shortwave_targets(SimpleNamespace(shortwave=True))


def test_the_default_driver_path_runs_nothing_new():
    from woof.core import physics
    src = inspect.getsource(physics.PhysicsDriver)
    assert 'if int(getattr(cfg, "solar_ssi_opt", 0) or 0) and radiation is not None:' in src
    assert 'if getattr(self, "solar_spectrum", None) is not None:' in src
    from woof.core import rrtmg_sw
    sig = inspect.signature(rrtmg_sw.CudaSW.rrtmg_sw_batched_device)
    assert sig.parameters["band_factor"].default is None


# --- on the card ----------------------------------------------------------------------------

def _driver(**overrides):
    from test_physics_driver import _base_config
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics
    # A 25 km top: at _base_config's 8 km top, RRTMG's taumol gives bands 16,
    # 17, 27, 28 and 29 no source (their reference layer is above the column).
    cfg = _base_config(ra_physics=0, ra_lw_physics=4, ra_sw_physics=4, ra_rrtmg_variant=LEGACY,
                       sf_surface_physics=2, sf_sfclay_physics=91, bl_pbl_physics=1,
                       run_seconds=600.0, ztop=25000.0, nz=40, **overrides)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 298.0 + 0.004 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base,
                                lambda z: 0.010 * np.exp(-np.asarray(z, np.float64) / 2400.0))
    driver = initialize_physics(
        state, cfg, radiation_start_time=datetime(2011, 4, 26, 17),
        radiation_latitude=np.full((cfg.ny, cfg.nx), 33.8),
        radiation_longitude=np.full((cfg.ny, cfg.nx), -87.29))
    return driver, state, cfg


def _sw_leaf(driver):
    from woof.core.radiation_composition import radiation_adapters
    return next(a for a in radiation_adapters(driver.radiation_callable)
                if getattr(a, "accepts_sw_band_factor", False))


def _radiate(driver, state, cfg, factor):
    import cupy as cp
    from woof.core.physics import _prepare_atmosphere
    _sw_leaf(driver).sw_band_factor = factor
    driver._run_radiation(_prepare_atmosphere(state), state, cfg)
    return {n: cp.asnumpy(driver.fields[n]).copy() if n in ("swdown", "glw") else
            cp.asnumpy(getattr(driver, n)).copy() for n in ("rthratensw", "rthratenlw", "swdown", "glw")}


@pytest.mark.gpu
@requires_gpu
def test_legacy_rrtmg_band_factor_is_exact_at_one_and_linear():
    driver, state, cfg = _driver()
    assert driver.solar_spectrum is None
    base = _radiate(driver, state, cfg, None)
    assert np.abs(base["rthratensw"]).max() > 0 and base["swdown"].min() > 100.0
    ones = _radiate(driver, state, cfg, (1.0,) * 14)
    for name in base:
        np.testing.assert_array_equal(ones[name], base[name], name)
    double = _radiate(driver, state, cfg, (2.0,) * 14)
    np.testing.assert_allclose(double["rthratensw"], 2 * base["rthratensw"], rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(double["swdown"], 2 * base["swdown"], rtol=1e-5)
    np.testing.assert_array_equal(double["rthratenlw"], base["rthratenlw"])
    # 50 % more 200-263 nm: absorbed by ozone aloft, so heating rises and the
    # surface sees at most the band's own share.
    uv = [1.0] * 14
    uv[12] = 1.5
    more = _radiate(driver, state, cfg, tuple(uv))
    assert np.all(more["rthratensw"] >= base["rthratensw"] - 1e-12)
    rise = more["swdown"] / base["swdown"] - 1.0
    assert np.all(rise >= 0.0) and rise.max() < 0.5 * REF[12]
    with pytest.raises(ValueError, match="band_factor"):
        _radiate(driver, state, cfg, (1.0,) * 13)
    # A renormalised factor (2024's largest UV day) keeps the top-of-atmosphere
    # total: RRTMG's Kurucz band shares are not NRLSSI2's, so it moves a little.
    import cupy as cp
    _radiate(driver, state, cfg, None)
    top = cp.asnumpy(driver.fields["swdnt"]).copy()
    shares = REF.copy()
    shares[12] *= 1.045
    shares[11] *= 1.0046
    _radiate(driver, state, cfg, tuple(float(v) for v in ss.band_factor(shares)))
    moved = np.abs(cp.asnumpy(driver.fields["swdnt"]) / top - 1.0).max()
    print(f"top-of-atmosphere SW down moved by {moved:.2e}")
    assert moved < 3.0e-5


@pytest.mark.gpu
@requires_gpu
def test_the_driver_sets_the_days_factor_before_the_call(tmp_path):
    path = _ssi_file(tmp_path / "ssi.nc", first=datetime(2011, 4, 25), uv=(1.2, 1.2, 1.2))
    driver, state, cfg = _driver(solar_ssi_opt=1, solar_ssi_source=str(path))
    assert driver.solar_spectrum is not None
    assert ss.SpectralVariability.shortwave_targets(driver.radiation_callable) == (_sw_leaf(driver),)
    from woof.core.physics import _prepare_atmosphere
    driver._run_radiation(_prepare_atmosphere(state), state, cfg)
    want = ss.band_factor(np.where(np.arange(14) == 12, REF * 1.2, REF))
    np.testing.assert_allclose(_sw_leaf(driver).sw_band_factor, want, rtol=1e-6)
    assert _sw_leaf(driver).sw_band_factor[12] > 1.19


@pytest.mark.gpu
@requires_gpu
def test_rrtmgk_takes_the_band_factor():
    import cupy as cp
    from tilestream import harness
    from woof.core import rrtmgk
    from woof.core.radiation_composition import retained_ozone_adapter
    cfg = RunConfig(nx=12, ny=12, nz=12, dx=3000., dy=3000., ztop=16000., dt=3.,
                    run_seconds=30., moist=True, mp_physics=10, radt=.2,
                    ra_lw_physics=14, ra_sw_physics=14, o3input=2)
    out = {}
    for key, factor in (("none", None), ("ones", (1.0,) * 14), ("two", (2.0,) * 14)):
        state, driver = harness.make_physics_state(cfg, 104, start_time=datetime(2011, 4, 27, 18))
        adapter = retained_ozone_adapter(driver.radiation_callable)
        assert isinstance(adapter, rrtmgk.RRTMGKRadiation)
        adapter.sw_band_factor = factor
        state.elapsed_seconds = 0.
        driver.compute(state, cfg)
        out[key] = {n: cp.asnumpy(getattr(driver, n)).copy() for n in ("rthratensw", "rthratenlw")}
    assert np.abs(out["none"]["rthratensw"]).max() > 0
    for name in out["none"]:
        np.testing.assert_array_equal(out["ones"][name], out["none"][name], name)
    np.testing.assert_allclose(out["two"]["rthratensw"], 2 * out["none"]["rthratensw"], rtol=1e-5, atol=1e-12)
