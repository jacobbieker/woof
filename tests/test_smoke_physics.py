"""Smoke source physics (woof.smoke): emission, injection, deposition.

CPU tier: the float64 references against hand values and the literature,
the column normalisations and conservation, and the vectorised forms run on
NumPy against the references.  GPU tier (``@pytest.mark.gpu``): the same
vectorised forms on CuPy arrays, float64 and float32, against the
references.  cupy is imported only inside the GPU test bodies, so the CPU
tests stay on ``-m "not gpu"``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import requires_gpu
from woof import smoke
from woof.smoke import deposition as dep
from woof.smoke import emission as em
from woof.smoke import plume_rise as pr

PBL = 1500.0


def _levels(top: float = 12000.0) -> np.ndarray:
    """Stretched interfaces: 25 m at the ground, 8 % growth, 400 m cap."""
    dz = np.minimum(25.0 * 1.08 ** np.arange(80), 400.0)
    zw = np.concatenate([[0.0], np.cumsum(dz)])
    return zw[zw < top]


def _daytime(zw: np.ndarray, inversion: float = 2.0, wind: float = 5.0):
    """Neutral mixed layer to PBL, a 100 m capping inversion, 4 K/km above."""
    zm = 0.5 * (zw[:-1] + zw[1:])
    theta = np.where(zm < PBL, 300.0,
                     np.where(zm < PBL + 100.0,
                              300.0 + inversion * (zm - PBL) / 100.0,
                              300.0 + inversion + 4.0e-3 * (zm - PBL - 100.0)))
    return theta, np.full_like(zm, wind)


def _columns(ncol: int = 6, seed: int = 3):
    """``ncol`` perturbed daytime columns as (nz[+1], 2, ncol//2) fields."""
    rng = np.random.default_rng(seed)
    zw1 = _levels()
    zw = zw1[:, None] * (1.0 + 0.1 * rng.random(ncol))[None, :]
    theta = np.empty((zw.shape[0] - 1, ncol))
    wind = np.empty_like(theta)
    for i in range(ncol):
        theta[:, i], wind[:, i] = _daytime(zw[:, i], 1.0 + 3.0 * rng.random(),
                                           10.0 * rng.random())
    shape = (2, ncol // 2)
    return (zw.reshape((-1,) + shape), theta.reshape((-1,) + shape),
            wind.reshape((-1,) + shape))


def _integral(source, zw):
    return np.sum(np.asarray(source) * np.diff(np.asarray(zw), axis=0),
                  axis=0)


# --------------------------------------------------------------------------
# emission


def test_anderson_categories_map_to_their_fuel_groups():
    assert [em.ANDERSON_FUEL_CLASS[k] for k in range(1, 15)] == (
        ["grass"] * 3 + ["shrub"] * 4 + ["timber"] * 3 + ["slash"] * 3
        + ["none"])
    table = em.emission_factor_table()
    assert math.isnan(table[0]) and table[em.NO_FUEL] == 0.0
    assert table[1] == pytest.approx(8.0e-3)
    assert table[8] == pytest.approx(15.0e-3)
    assert em.emission_factor("shrub") == em.emission_factor(4)


@pytest.mark.parametrize("bad", [0, 15, -1, 2.5])
def test_unknown_fuel_categories_are_refused(bad):
    with pytest.raises(ValueError):
        em.emission_factor(bad)
    with pytest.raises(ValueError):
        em.pm25_emission_flux_np(np.ones(2), np.array([1, bad]))
    with pytest.raises(ValueError):
        em.pm25_emission_flux(np.ones(2), np.array([1, bad]))
    with pytest.raises(ValueError):
        em.emission_factor("peat")


def test_emission_flux_is_factor_times_burn_rate():
    q = 5.0e4
    rate = em.burn_rate_from_heat_flux(q)
    assert rate == pytest.approx(q / 18.608e6, rel=1e-4)
    flux = em.pm25_emission_flux_np(rate, 10)
    assert flux == pytest.approx(rate * 15.0e-3, rel=1e-15)
    with pytest.raises(ValueError):
        em.pm25_emission_flux_np(-1.0, 1)
    with pytest.raises(ValueError):
        em.pm25_emission_flux_np(np.nan, 1)


def test_emission_vectorised_matches_reference():
    rng = np.random.default_rng(0)
    rate = rng.random((3, 4, 5)) * 1e-2
    fuel = rng.integers(1, 15, size=(3, 4, 5))
    ref = em.pm25_emission_flux_np(rate, fuel)
    np.testing.assert_allclose(em.pm25_emission_flux(rate, fuel), ref,
                               rtol=1e-15, atol=0)
    np.testing.assert_allclose(em.pm25_emission_flux(rate, "grass"),
                               em.pm25_emission_flux_np(rate, "grass"),
                               rtol=1e-15)
    assert np.all(ref[fuel == em.NO_FUEL] == 0.0)


def test_caller_factors_replace_the_defaults():
    factors = dict(em.EMISSION_FACTORS)
    factors["timber"] = em.EmissionFactor("timber", 20.0, "local")
    assert em.pm25_emission_flux_np(1.0, 9, factors) == pytest.approx(0.02)


# --------------------------------------------------------------------------
# injection profiles


def test_lowest_layer_profile_puts_everything_in_layer_zero():
    zw, _, _ = _columns()
    flux = np.full(zw.shape[1:], 3.0e-5)
    ref = pr.lowest_layer_profile_np(zw, flux)
    assert np.all(ref[1:] == 0.0)
    np.testing.assert_allclose(_integral(ref, zw), flux, rtol=1e-14)
    np.testing.assert_allclose(pr.lowest_layer_profile(zw, flux), ref,
                               rtol=1e-15)


def test_uniform_profile_shares_by_overlap_and_integrates_to_the_flux():
    zw = np.array([0.0, 10.0, 30.0, 60.0, 100.0])
    s = pr.uniform_profile_np(zw, 1.0, 5.0, 45.0)
    # 5 m of layer 0, all 20 m of layer 1, 15 m of layer 2, out of 40 m.
    np.testing.assert_allclose(s * np.diff(zw), [5 / 40, 20 / 40, 15 / 40, 0],
                               rtol=1e-15)
    # Clipped to the column; degenerate -> the layer holding z_bot.
    assert _integral(pr.uniform_profile_np(zw, 2.0, 50.0, 500.0), zw) == (
        pytest.approx(2.0, rel=1e-15))
    np.testing.assert_allclose(pr.uniform_profile_np(zw, 1.0, 35.0, 35.0)
                               * np.diff(zw), [0, 0, 1, 0])
    np.testing.assert_allclose(pr.uniform_profile_np(zw, 1.0, 100.0, 90.0)
                               * np.diff(zw), [0, 0, 0, 1])
    for bot, top in [(5.0, 45.0), (50.0, 500.0), (35.0, 35.0),
                     (100.0, 90.0), (-5.0, 0.0), (0.0, 10.0)]:
        np.testing.assert_allclose(pr.uniform_profile(zw, 1.0, bot, top),
                                   pr.uniform_profile_np(zw, 1.0, bot, top),
                                   rtol=1e-15, atol=0)


@pytest.mark.parametrize("fraction", [1.0, 0.5, 0.0])
def test_pbl_profile_is_uniform_below_the_fraction_of_pbl(fraction):
    zw, _, _ = _columns()
    flux = np.linspace(1e-6, 1e-4, zw[0].size).reshape(zw.shape[1:])
    pbl = np.full(zw.shape[1:], PBL)
    ref = pr.pbl_uniform_profile_np(zw, flux, pbl, fraction)
    np.testing.assert_allclose(_integral(ref, zw), flux, rtol=1e-13)
    top = fraction * PBL
    if fraction > 0:
        inside = zw[1:] <= top
        mean = flux / top
        np.testing.assert_allclose(ref[inside],
                                   np.broadcast_to(mean, ref.shape)[inside],
                                   rtol=1e-13)
        assert np.all(ref[zw[:-1] >= top] == 0.0)
    else:
        assert np.all(ref[1:] == 0.0)
    np.testing.assert_allclose(pr.pbl_uniform_profile(zw, flux, pbl,
                                                      fraction), ref,
                               rtol=1e-14, atol=0)


def test_plume_rise_profile_integrates_to_the_flux():
    zw, theta, wind = _columns()
    flux = np.full(zw.shape[1:], 2.0e-5)
    heat = np.array([[5e4, 2e4, 0.0], [8e4, 5e4, 1e4]])
    area = np.array([[1e4, 1e5, 1e4], [1e3, 0.0, 1e6]])
    params = pr.PlumeRiseParameters(surface_fraction=0.2)
    ref, plume = pr.plume_rise_profile_np(zw, flux, theta, wind, heat, area,
                                          params=params)
    np.testing.assert_allclose(_integral(ref, zw), flux, rtol=1e-13)
    # No fire (zero heat or zero area): everything stays in layer 0.
    for col in [(0, 2), (1, 1)]:
        assert plume.top[col] == 0.0
        assert np.all(ref[(slice(1, None),) + col] == 0.0)
    # A fire: 20 % in layer 0, the rest between bottom and top.
    zw0 = zw[(slice(None), 0, 0)]
    above = zw0[:-1] >= plume.top[0, 0]
    assert np.all(ref[(slice(None), 0, 0)][above] == 0.0)
    vec, vplume = pr.plume_rise_profile(zw, flux, theta, wind, heat, area,
                                        params=params)
    np.testing.assert_allclose(vec, ref, rtol=1e-11, atol=1e-25)
    np.testing.assert_allclose(vplume.top, plume.top, rtol=1e-12)


# --------------------------------------------------------------------------
# plume rise physics


def _top(heat=5e4, area=1e4, wind=5.0, inversion=2.0, **kw):
    zw = _levels()
    theta, u = _daytime(zw, inversion, wind)
    return float(pr.plume_rise_np(zw, theta, u, heat, area, **kw).top)


def test_typical_fire_tops_out_just_above_the_boundary_layer():
    top = _top()
    assert PBL < top < 2500.0
    zw = _levels()
    theta, u = _daytime(zw)
    plume = pr.plume_rise_np(zw, theta, u, 5e4, 1e4)
    # Initial state closes the heat flux: rho cp w0 dth0 = Q.
    assert (pr.AIR_DENSITY * 1004.5 * plume.initial_velocity
            * plume.initial_excess) == pytest.approx(5e4, rel=1e-12)
    assert plume.bottom == pytest.approx(0.5 * plume.top)


def test_plume_top_responds_to_fire_size_wind_and_stability():
    assert _top(heat=2e4) < _top(heat=5e4) < _top(heat=1e5)
    assert _top(area=1e3) < _top(area=1e4) < _top(area=1e6)
    assert _top(wind=0.0) > _top(wind=5.0) > _top(wind=15.0)
    assert _top(inversion=6.0) < _top(inversion=2.0)


def test_plume_top_converges_with_substeps():
    coarse = _top(params=pr.PlumeRiseParameters(substeps=8))
    fine = _top(params=pr.PlumeRiseParameters(substeps=64))
    assert coarse == pytest.approx(fine, rel=0.02)


def test_unstratified_column_plume_stops_at_the_column_top():
    zw = _levels(3000.0)
    zm = 0.5 * (zw[:-1] + zw[1:])
    top = pr.plume_rise_np(zw, np.full_like(zm, 300.0), np.zeros_like(zm),
                           5e4, 1e4).top
    assert float(top) == zw[-1]


def test_plume_rise_vectorised_matches_reference():
    zw, theta, wind = _columns()
    heat = np.array([[5e4, 2e4, 0.0], [8e4, 5e4, 1e4]])
    area = np.array([[1e4, 1e5, 1e4], [1e3, 0.0, 1e6]])
    ref = pr.plume_rise_np(zw, theta, wind, heat, area)
    vec = pr.plume_rise(zw, theta, wind, heat, area)
    for a, b in zip(vec, ref):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)


def test_bad_columns_are_refused():
    zw = _levels(3000.0)
    theta, wind = _daytime(zw)
    with pytest.raises(ValueError, match="increase"):
        pr.uniform_profile_np(zw[::-1], 1.0, 0.0, 10.0)
    with pytest.raises(ValueError, match="shape"):
        pr.plume_rise_np(zw, theta[:-1], wind, 5e4, 1e4)
    with pytest.raises(ValueError, match="theta"):
        pr.plume_rise(zw, -theta, wind, 5e4, 1e4)
    with pytest.raises(ValueError):
        pr.PlumeRiseParameters(surface_fraction=1.5)
    with pytest.raises(ValueError):
        pr.pbl_uniform_profile_np(zw, 1.0, PBL, fraction=2.0)


# --------------------------------------------------------------------------
# deposition


def test_settling_matches_stokes_cunningham_textbook_values():
    # Seinfeld and Pandis (2016) Table 9.3, unit-density spheres at 298 K
    # and 1 atm: v_s = 8.62e-5, 3.48e-3 and 0.306 cm/s for 0.1, 1, 10 um.
    for d, expected in [(0.1e-6, 8.62e-7), (1.0e-6, 3.48e-5),
                        (10.0e-6, 3.06e-3)]:
        v = dep.settling_velocity_np(298.0, 101325.0, d, 1000.0)
        assert float(v) == pytest.approx(expected, rel=0.03)


def test_pm25_deposition_velocity_is_in_the_measured_range():
    """0.1-1 cm/s for fine particles over vegetation at daytime u*."""
    for surface in ("grass", "needleleaf_forest", "broadleaf_forest",
                    "shrub", "crops"):
        for ustar in (0.3, 0.5):
            v = dep.dry_deposition_velocity_np(ustar, 293.0, 1.0e5, 12.5,
                                               surface)
            assert 1.0e-3 <= float(v.deposition) <= 1.0e-2, (surface, ustar)
    # The accumulation-mode minimum: smaller and much larger both deposit
    # faster than ~1 um.
    v = [float(dep.dry_deposition_velocity_np(
        0.4, 293.0, 1e5, 12.5, "grass", diameter=d).deposition)
        for d in (0.01e-6, 1.0e-6, 20.0e-6)]
    assert v[1] < v[0] and v[1] < v[2]
    # More turbulence, faster deposition; unstable lowers R_a, stable raises.
    slow = dep.dry_deposition_velocity_np(0.1, 293.0, 1e5, 12.5)
    fast = dep.dry_deposition_velocity_np(0.6, 293.0, 1e5, 12.5)
    assert float(slow.deposition) < float(fast.deposition)
    neutral = dep.dry_deposition_velocity_np(0.3, 293.0, 1e5, 12.5)
    unstable = dep.dry_deposition_velocity_np(0.3, 293.0, 1e5, 12.5,
                                              obukhov_length=-50.0)
    stable = dep.dry_deposition_velocity_np(0.3, 293.0, 1e5, 12.5,
                                            obukhov_length=50.0)
    assert (float(unstable.aerodynamic_resistance)
            < float(neutral.aerodynamic_resistance)
            < float(stable.aerodynamic_resistance))
    assert float(neutral.aerodynamic_resistance) == pytest.approx(
        math.log(12.5 / 0.05) / (0.4 * 0.3), rel=1e-14)


def test_deposition_vectorised_matches_reference():
    rng = np.random.default_rng(5)
    shape = (4, 5)
    ustar = 0.05 + rng.random(shape)
    t = 270.0 + 40.0 * rng.random(shape)
    p = 8.0e4 + 2.0e4 * rng.random(shape)
    zr = 5.0 + 30.0 * rng.random(shape)
    length = np.where(rng.random(shape) < 0.5, -1.0, 1.0) * (
        5.0 + 500.0 * rng.random(shape))
    length[0, 0] = np.inf
    for surface in dep.DEPOSITION_SURFACES:
        for d in (0.1e-6, 0.3e-6, 2.5e-6):
            ref = dep.dry_deposition_velocity_np(
                ustar, t, p, zr, surface, d, obukhov_length=length)
            vec = dep.dry_deposition_velocity(
                ustar, t, p, zr, surface, d, obukhov_length=length)
            for a, b in zip(vec, ref):
                np.testing.assert_allclose(a, b, rtol=1e-13, atol=0)
    np.testing.assert_allclose(dep.settling_velocity(t, p),
                               dep.settling_velocity_np(t, p), rtol=1e-14)


def _tracer_column(seed=11):
    rng = np.random.default_rng(seed)
    zw, _, _ = _columns(seed=seed)
    nz = zw.shape[0] - 1
    q = rng.random((nz,) + zw.shape[1:]) * 1e-8
    rho = 1.2 * np.exp(-0.5 * (zw[:-1] + zw[1:]) / 8000.0)
    t = 300.0 - 6.5e-3 * 0.5 * (zw[:-1] + zw[1:])
    p = 1.0e5 * rho / 1.2
    vs = dep.settling_velocity_np(t, p, 2.5e-6, 1400.0)
    vd = 0.002 + 0.002 * rng.random(zw.shape[1:])
    return q, zw, vs, vd, rho


def test_settling_tendency_conserves_column_mass_except_deposition():
    q, zw, vs, vd, rho = _tracer_column()
    out = dep.settling_tendency_np(q, zw, vs, vd, rho)
    column = np.sum(rho * np.diff(zw, axis=0) * out.tendency, axis=0)
    np.testing.assert_allclose(column, -out.surface_flux, rtol=1e-12)
    np.testing.assert_allclose(out.surface_flux, rho[0] * vd * q[0],
                               rtol=1e-15)
    vec = dep.settling_tendency(q, zw, vs, vd, rho)
    np.testing.assert_allclose(vec.tendency, out.tendency, rtol=1e-12,
                               atol=1e-30)
    np.testing.assert_allclose(vec.surface_flux, out.surface_flux,
                               rtol=1e-15)


def test_uniform_settling_leaves_the_interior_unchanged():
    zw = np.linspace(0.0, 1000.0, 11)
    q = np.full(10, 1e-9)
    out = dep.settling_tendency_np(q, zw, 0.01, 0.01)
    np.testing.assert_allclose(out.tendency[1:-1], 0.0, atol=1e-30)
    assert out.tendency[-1] == pytest.approx(-0.01 * 1e-9 / 100.0)


def test_deposited_fraction_matches_the_lowest_layer_tendency():
    """The closed form against an explicit hour of the column tendency."""
    vd, depth, hour = 0.003, 50.0, 3600.0
    zw = np.array([0.0, depth])
    q = np.array([1.0])
    dt = 1.0
    for _ in range(int(hour / dt)):
        q = q + dt * dep.settling_tendency_np(q, zw, 0.0, vd).tendency
    assert 1.0 - q[0] == pytest.approx(
        float(dep.deposited_fraction(vd, depth, hour)), rel=1e-3)


def test_bad_deposition_inputs_are_refused():
    with pytest.raises(ValueError, match="unknown"):
        dep.dry_deposition_velocity_np(0.3, 293.0, 1e5, 10.0, "tundra")
    with pytest.raises(ValueError, match="ustar"):
        dep.dry_deposition_velocity_np(0.0, 293.0, 1e5, 10.0)
    with pytest.raises(ValueError, match="Obukhov"):
        dep.dry_deposition_velocity(0.3, 293.0, 1e5, 10.0,
                                    obukhov_length=0.0)
    with pytest.raises(ValueError):
        dep.settling_velocity_np(293.0, 1e5, diameter=0.0)
    with pytest.raises(ValueError, match="interfaces"):
        dep.settling_tendency_np(np.ones(3), np.arange(3.0), 0.0, 0.0)


def test_package_exports_resolve():
    for name in smoke.__all__:
        assert getattr(smoke, name) is not None


def test_review_edge_cases_are_refused_or_consistent():
    # An incomplete caller table: both forms agree where it covers the
    # fuel and both refuse, by name, where it does not.
    factors = {k: v for k, v in em.EMISSION_FACTORS.items()
               if k in ("grass", "timber")}
    for fn in (em.pm25_emission_flux_np, em.pm25_emission_flux):
        np.testing.assert_allclose(fn(np.ones(3), np.array([1, 8, 14]),
                                      factors), [8e-3, 15e-3, 0.0])
        with pytest.raises(ValueError, match="shrub"):
            fn(np.ones(2), np.array([1, 4]), factors)
    with pytest.raises(ValueError):
        pr.PlumeRiseParameters(substeps=8.0)
    zw = _levels(3000.0)
    theta, wind = _daytime(zw)
    for fn in (pr.plume_rise_np, pr.plume_rise):
        with pytest.raises(ValueError, match="air_density"):
            fn(zw, theta, wind, 5e4, 1e4, air_density=-1.2)
        with pytest.raises(ValueError, match="heat_flux"):
            fn(zw, theta, wind, np.nan, 1e4)
    with pytest.raises(ValueError, match="roughness"):
        dep.dry_deposition_velocity_np(0.3, 293.0, 1e5, 0.8, "urban")
    for fn in (dep.settling_tendency_np, dep.settling_tendency):
        with pytest.raises(ValueError):
            fn(np.ones(2), np.arange(3.0), 0.0, np.nan)
        with pytest.raises(ValueError):
            fn(np.ones(2), np.arange(3.0), np.nan, 0.0)


# --------------------------------------------------------------------------
# GPU parity: the vectorised forms on CuPy arrays against the references


def _device(cp, dtype, *arrays):
    return tuple(cp.asarray(np.asarray(a), dtype=dtype) for a in arrays)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-12),
                                        (np.float32, 2e-6)])
def test_gpu_emission_matches_reference(dtype, rtol):
    import cupy as cp

    rng = np.random.default_rng(1)
    rate = rng.random((8, 9)) * 1e-2
    fuel = rng.integers(1, 15, size=(8, 9))
    ref = em.pm25_emission_flux_np(rate, fuel)
    (rate_d,) = _device(cp, dtype, rate)
    out = em.pm25_emission_flux(rate_d, cp.asarray(fuel))
    assert isinstance(out, cp.ndarray) and out.dtype == dtype
    np.testing.assert_allclose(out.get(), ref, rtol=rtol, atol=0)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("dtype,rtol,ztol", [(np.float64, 1e-11, 1e-12),
                                             (np.float32, 1e-4, 1e-4)])
def test_gpu_injection_profiles_match_reference(dtype, rtol, ztol):
    import cupy as cp

    zw, theta, wind = _columns()
    flux = np.full(zw.shape[1:], 2.0e-5)
    heat = np.array([[5e4, 2e4, 0.0], [8e4, 5e4, 1e4]])
    area = np.array([[1e4, 1e5, 1e4], [1e3, 0.0, 1e6]])
    params = pr.PlumeRiseParameters(surface_fraction=0.1)
    d = _device(cp, dtype, zw, flux, theta, wind, heat, area)
    zw_d, flux_d, theta_d, wind_d, heat_d, area_d = d

    out = pr.lowest_layer_profile(zw_d, flux_d)
    assert out.dtype == dtype
    np.testing.assert_allclose(out.get(), pr.lowest_layer_profile_np(zw, flux),
                               rtol=rtol)
    pbl = np.full(zw.shape[1:], PBL)
    out = pr.pbl_uniform_profile(zw_d, flux_d, cp.asarray(pbl, dtype), 0.7)
    ref = pr.pbl_uniform_profile_np(zw, flux, pbl, 0.7)
    np.testing.assert_allclose(out.get(), ref, rtol=rtol,
                               atol=rtol * float(ref.max()))

    plume = pr.plume_rise(zw_d, theta_d, wind_d, heat_d, area_d,
                          params=params)
    plume_ref = pr.plume_rise_np(zw, theta, wind, heat, area, params=params)
    for a, b in zip(plume, plume_ref):
        assert a.dtype == dtype
        np.testing.assert_allclose(a.get(), b, rtol=ztol, atol=0)
    if dtype == np.float64:
        src, _ = pr.plume_rise_profile(zw_d, flux_d, theta_d, wind_d,
                                       heat_d, area_d, params=params)
        src_ref, _ = pr.plume_rise_profile_np(zw, flux, theta, wind, heat,
                                              area, params=params)
        np.testing.assert_allclose(src.get(), src_ref, rtol=1e-10,
                                   atol=1e-25)
    else:
        # A float32 plume top moves a layer's share by its own rounding;
        # the normalisation is what must hold.
        src, _ = pr.plume_rise_profile(zw_d, flux_d, theta_d, wind_d,
                                       heat_d, area_d, params=params)
        np.testing.assert_allclose(_integral(src.get(), zw), flux, rtol=1e-5)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-12),
                                        (np.float32, 5e-5)])
def test_gpu_deposition_matches_reference(dtype, rtol):
    import cupy as cp

    rng = np.random.default_rng(9)
    shape = (16, 16)
    ustar = 0.05 + rng.random(shape)
    t = 270.0 + 40.0 * rng.random(shape)
    p = 8.0e4 + 2.0e4 * rng.random(shape)
    zr = 5.0 + 30.0 * rng.random(shape)
    length = np.where(rng.random(shape) < 0.5, -1.0, 1.0) * (
        5.0 + 500.0 * rng.random(shape))
    d = _device(cp, dtype, ustar, t, p, zr, length)
    for surface in ("grass", "needleleaf_forest", "water"):
        ref = dep.dry_deposition_velocity_np(ustar, t, p, zr, surface,
                                             obukhov_length=length)
        out = dep.dry_deposition_velocity(*d[:4], surface,
                                          obukhov_length=d[4])
        for a, b in zip(out, ref):
            assert a.dtype == dtype
            np.testing.assert_allclose(a.get(), b, rtol=rtol, atol=0)

    q, zw, vs, vd, rho = _tracer_column()
    ref = dep.settling_tendency_np(q, zw, vs, vd, rho)
    out = dep.settling_tendency(*_device(cp, dtype, q, zw, vs, vd, rho))
    scale = float(np.abs(ref.tendency).max())
    np.testing.assert_allclose(out.tendency.get(), ref.tendency,
                               rtol=rtol, atol=rtol * scale)
    np.testing.assert_allclose(out.surface_flux.get(), ref.surface_flux,
                               rtol=rtol)
    vs_d = dep.settling_velocity(*_device(cp, dtype, t, p))
    np.testing.assert_allclose(vs_d.get(), dep.settling_velocity_np(t, p),
                               rtol=rtol)

