"""solar_ssi_opt on photolysis: the day's spectrum on Fast-TUV's extraterrestrial flux.

On the CPU: the shipped reference spectrum reproduces REF_BAND_FRACTION where
both cover the same wavelengths, the bin factor is one at the reference and is
not renormalised, Fast-TUV's bins are wave_data's, and the per-run object reads
a CDR-layout series on those bins.  On a card (CUDA or Metal): the hook is a
no-op off and at ones (every word of WRF's ftuv_driver output kept), scales the
rates exactly with a uniform power-of-two factor, and moves only the rates
whose cross sections see the scaled bin.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.spacewx import solar_spectrum as ss
from woof.spacewx.solar_irradiance import SolarForcingError

from conftest import requires_gpu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FTUV_EDGES = (178.6, 180.0, 183.0, 187.0, 192.0, 198.0, 205.0, 213.0, 222.0, 241.0, 289.9, 305.5,
              313.5, 337.5, 420.0, 475.0, 729.0, 743.6)


def test_the_reference_spectrum_is_ref_band_fractions_product():
    wl, dw, frac = ss.reference_spectrum()
    assert wl.size == dw.size == frac.size == 752 and np.all(frac > 0)
    assert wl[0] - 0.5 * dw[0] == 0.0 and wl[-1] + 0.5 * dw[-1] == 760.0
    # Bands 25-28 (263-625 nm) lie inside the spectrum: the same days, the same numbers.
    # (REF_BAND_FRACTION is printed to 8 decimals.)
    np.testing.assert_allclose(ss.reference_fractions(ss.BANDS_NM[9:13]), ss.REF_BAND_FRACTION[9:13],
                               rtol=0, atol=5.0e-9)
    # Below 760 nm sits a bit over half the TSI.
    assert 0.5 < float(np.dot(frac, dw)) < 0.6


def test_the_bin_factor_is_one_at_the_reference_and_not_renormalised():
    bins = tuple(zip(FTUV_EDGES[:-1], FTUV_EDGES[1:]))
    ref = ss.reference_fractions(bins)
    np.testing.assert_array_equal(ss.bin_factor(ref, bins), np.ones(17))
    uv = ref.copy()
    uv[3] *= 1.05
    f = ss.bin_factor(uv, bins)
    assert f[3] == pytest.approx(1.05) and np.all(np.delete(f, 3) == 1.0)
    for bad in (((100.0, 900.0),), ((300.0, 300.0),)):
        with pytest.raises(SolarForcingError, match="outside the reference"):
            ss.reference_fractions(bad)


def test_ftuvs_bins_are_wave_datas():
    from woof.core.chem_ftuv import etf_bins_nm
    assert etf_bins_nm() == tuple(zip(FTUV_EDGES[:-1], FTUV_EDGES[1:]))


def _spectrum_file(path, *, uv=(1.0, 1.0, 1.0), first=datetime(2024, 5, 10)):
    """A NOAA CDR-layout SSI file on the reference's own bins: the reference
    spectrum times TSI, with 180-241 nm times ``uv`` per day."""
    netCDF4 = pytest.importorskip("netCDF4")
    wl, dw, frac = ss.reference_spectrum()
    days, tsi = len(uv), 1361.0
    start = (first - datetime(1610, 1, 1)).days
    ssi = np.empty((days, wl.size))
    for d in range(days):
        ssi[d] = tsi * frac * np.where((wl > 180.0) & (wl < 241.0), uv[d], 1.0)
    with netCDF4.Dataset(path, "w") as data:
        data.createDimension("time", days)
        data.createDimension("wavelength", wl.size)
        data.createDimension("bounds", 2)
        t = data.createVariable("time", "f8", ("time",))
        t.units = "days since 1610-01-01 00:00:00"
        t[:] = start + np.arange(days)
        b = data.createVariable("time_bnds", "f8", ("time", "bounds"))
        b[:] = np.stack([start + np.arange(days), start + 1 + np.arange(days)], axis=1)
        data.createVariable("wavelength", "f8", ("wavelength",))[:] = wl
        data.createVariable("Wavelength_Band_Width", "f8", ("wavelength",))[:] = dw
        data.createVariable("SSI", "f8", ("time", "wavelength"))[:] = ssi
        data.createVariable("TSI", "f8", ("time",))[:] = np.full(days, tsi)
    return path


def test_the_run_object_reads_the_series_on_photolysis_bins(tmp_path):
    path = _spectrum_file(tmp_path / "ssi.nc", uv=(1.0, 1.1, 1.2))
    bins = tuple(zip(FTUV_EDGES[:-1], FTUV_EDGES[1:]))
    cfg = SimpleNamespace(solar_ssi_opt=1, solar_ssi_source=str(path), run_seconds=86400.0)
    spec = ss.PhotolysisSpectrum.for_run(cfg, datetime(2024, 5, 10, 12), bins, "ftuv")
    np.testing.assert_allclose(spec.start_factor, 1.0, rtol=1e-12)
    f = spec.factor(43200.0)                          # 2024-05-11 00 UTC: halfway to 1.1
    inside = [b for b, (lo, hi) in enumerate(bins) if lo >= 180.0 and hi <= 241.0]
    assert inside == list(range(1, 9))
    np.testing.assert_allclose(f[inside], 1.05, rtol=1e-12)
    np.testing.assert_allclose(np.delete(f, inside), 1.0, rtol=1e-12)
    assert list(spec.receipt()) == ["solar_ssi_ftuv_factor_at_start"]
    assert ss.PhotolysisSpectrum.for_run(SimpleNamespace(solar_ssi_opt=0), None, bins, "ftuv") is None
    with pytest.raises(SolarForcingError, match="start time"):
        ss.PhotolysisSpectrum.for_run(cfg, None, bins, "ftuv")


def test_the_default_step_hands_no_factor():
    from woof.core import chem_ftuv
    ctx = SimpleNamespace(cfg=SimpleNamespace(), state=SimpleNamespace(chem=SimpleNamespace()))
    assert chem_ftuv._solar_spectrum_factor(ctx) is None
    assert not hasattr(ctx.state.chem, "ftuv_solar_spectrum")


# --- on a card ----------------------------------------------------------------------------

def _case1():
    import cupy as cp
    from test_chem_ftuv_wrf471_parity import P_O3, P_QC, P_QV, _cols, _load
    f = _load("case1")
    nz = 30
    g = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
    fields = {
        "t_phy": g(_cols(f["t_phy"])[:nz]), "p_phy": g(_cols(f["p_phy"])[:nz]),
        "rho_phy": g(_cols(f["rho_phy"])[:nz]),
        "p8w": g(_cols(f["p8w"])), "t8w": g(_cols(f["t8w"])), "z_at_w": g(_cols(f["z_at_w"])),
        "qv": g(_cols(f["moist"][P_QV - 1])[:nz]), "qc": g(_cols(f["moist"][P_QC - 1])[:nz]),
        "o3": g(_cols(f["chem"][P_O3 - 1])[:nz]),
        "xlat": g(f["xlat"]), "xlong": g(f["xlong"]),
    }
    optics = None
    if int(f["aer_ra_feedback"]) == 1:
        optics = [g(_cols(f[name][n])[:nz]) for name in ("tauaer", "waer", "gaer") for n in range(4)]
    kw = dict(chem_opt=int(f["chem_opt"]), julday=int(f["julday"]), gmt=float(f["gmt"]),
              curr_secs=float(f["curr_secs"]), aer_ra_feedback=int(f["aer_ra_feedback"]),
              aerosol_optics=optics)
    return fields, kw


def _run(etf_factor=None):
    from woof.core.chem_ftuv import PHOTOLYSIS, launch_ftuv
    fields, kw = _case1()
    out = launch_ftuv(fields, etf_factor=etf_factor, **kw)
    return {name: out[name].get() for name in PHOTOLYSIS}


@pytest.mark.gpu
@requires_gpu
def test_the_flux_factor_is_exact_at_one_and_linear():
    base = _run()
    assert max(float(v.max()) for v in base.values()) > 0.0   # a sunlit column
    for name, v in _run(np.ones(17)).items():
        np.testing.assert_array_equal(v.view(np.uint32), base[name].view(np.uint32), err_msg=name)
    tiny = np.finfo(np.float32).tiny
    for name, v in _run(np.full(17, 2.0)).items():   # a power of two: exact above the subnormals
        normal = np.abs(base[name]) >= tiny
        np.testing.assert_array_equal(v[normal], 2.0 * base[name][normal], err_msg=name)
    with pytest.raises(ValueError, match="17 finite positive"):
        _run(np.ones(16))


@pytest.mark.gpu
@requires_gpu
def test_a_visible_bin_moves_no2_and_not_o1d():
    base = _run()
    f = np.ones(17)
    f[14] = 1.1                                     # 420-475 nm: NO2 absorbs, O3 -> O1D does not
    got = _run(f)
    assert np.array_equal(got["o31d"], base["o31d"])
    lit = base["no2"] > 0
    ratio = got["no2"][lit] / base["no2"][lit]
    assert np.all(ratio > 1.0) and np.all(ratio < 1.1)
