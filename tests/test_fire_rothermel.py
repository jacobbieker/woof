"""Fuel model tables and Rothermel surface spread against published values.

Reference sources, cited at each test:

* [A18] Andrews, P. L. (2018) *The Rothermel surface fire spread model and
  associated developments: a comprehensive explanation*, USDA FS
  RMRS-GTR-371.  Tables 8, 17, 21, 22, 23, 24 and the worked examples of
  sections 5.1.2 (fig. 10-11), 5.3.4 and 5.4.4 (fig. 46).
* [BP] BehavePlus core regression values, firelab/behave
  ``src/testBehave/testBehave.cpp`` (``testSimpleSurface`` and the
  live moisture of extinction of ``testSurfaceSingleFuelModel``), printed
  there to six decimals.
* [A82] Anderson, H. E. (1982) *Aids to determining fuel models for
  estimating fire behavior*, USDA FS GTR INT-122: representative rate of
  spread and flame length of each of the 13 models at 8 % dead and 100 %
  live moisture and 5 mi/h mid-flame wind, printed to two significant
  figures (pp. 4, 7, 10, 13).
* [WRF] WRF v4.7.1 ``phys/module_fr_fire_phys.F`` (``set_fire_params``,
  ``fire_ros``), whose float32 transcription :mod:`woof.fire.wrf_sfire`
  equals WRF's compiled Fortran bit for bit in
  ``tests/test_fire_ros_wrf471_parity.py``; here, its physical properties.

CPU and GPU are held to raw-word equality (max ULP 0) for every output.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import requires_gpu
from woof.fire import fuels as F
from woof.fire import rothermel as R
from woof.fire import wrf_sfire as W

MPH = 0.44704                 # mi/h -> m/s
FT_MIN = F.FT_PER_MIN         # ft/min -> m/s
CH_H = 20.1168 / 3600.0       # chains/h -> m/s


def _fire(fuel, dead, live_herb, live_woody, wind_mph, slope, **kw):
    """Full-model head fire with one dead moisture for all dead classes."""
    return R.rothermel_surface_fire(fuel, dead, dead, dead, live_herb,
                                    live_woody, wind_mph * MPH, slope, **kw)


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def test_standard_tables_have_53_models_with_scott_burgan_dynamic_set():
    assert len(F.ANDERSON_13) == 13 and len(F.SCOTT_BURGAN_40) == 40
    assert len(F.STANDARD_FUEL_MODELS) == 53
    dynamic = {m.code for m in F.SCOTT_BURGAN_40 if m.dynamic}
    # [A18] sec 5.2.2: "the 17 dynamic fuel models in the set of 40".
    assert dynamic == ({f"GR{i}" for i in range(1, 10)}
                       | {f"GS{i}" for i in range(1, 5)}
                       | {"SH1", "SH9", "TU1", "TU3"})
    assert not any(m.dynamic for m in F.ANDERSON_13)
    assert F.fuel_model("gr2") is F.fuel_model(102) is F.STANDARD_FUEL_MODELS[102]
    with pytest.raises(KeyError):
        F.fuel_model(14)


def test_si_conversions_of_published_rows():
    fm1 = F.fuel_model(1)
    # 0.034 lb/ft2 [Albini 1976] = 0.166 kg/m2, WRF-SFIRE's fgi(1).
    assert fm1.load_1h == pytest.approx(0.034 * 4.882427636, rel=1e-9)
    assert round(fm1.load_1h, 3) == F.WRF_SFIRE_FUEL_CATEGORIES[1].fgi
    assert fm1.sav_1h == pytest.approx(3500 / 0.3048)
    assert fm1.depth == pytest.approx(0.3048)
    assert fm1.heat_content_dead == pytest.approx(8000 * 2.326, rel=1e-9)
    assert fm1.particle_density == pytest.approx(512.5908, rel=1e-6)
    gr2 = F.fuel_model("GR2")
    # 1.0 t/ac = 2000/43560 lb/ft2 = 0.2241702 kg/m2.
    assert gr2.load_live_herb == pytest.approx(0.22417023, rel=1e-7)


def test_wrf_sfire_namelist_defaults():
    cats = F.WRF_SFIRE_FUEL_CATEGORIES
    assert sorted(cats) == list(range(1, 15))
    s = F.WRF_SFIRE_FUEL_SCALARS
    assert (s.cmbcnst, s.hfgl, s.fuelmc_g, s.fuelmc_c) == (17.433e6, 17.0e4,
                                                           0.08, 1.00)
    assert (s.nfuelcats, s.no_fuel_cat) == (13, 14)
    assert s.fuelheat_btu_lb == pytest.approx(7496.19)
    assert [cats[k].weight for k in (1, 4, 5, 8, 13)] == [7., 180., 100.,
                                                         900., 900.]
    assert cats[4].ichap == 1 and sum(c.ichap for c in cats.values()) == 1
    assert cats[8].fuel_time == pytest.approx(900.0 / 0.85)
    assert cats[9].fcbr == pytest.approx(1.121 / 120.0)
    assert cats[1].fcbr == 0.0
    # The lumped WRF savr is the characteristic SAV of the standard model
    # ([A18] Table 8a); categories 1-13 are fuel models 1-13.
    for k in range(1, 14):
        assert cats[k].fuelmce == F.fuel_model(k).moisture_extinction_dead
        assert cats[k].fueldepthm == pytest.approx(F.fuel_model(k).depth,
                                                   abs=1e-3)


def test_windrf_is_the_unsheltered_wind_adjustment_factor():
    """WRF-SFIRE's ``windrf`` is Albini & Baughman's unsheltered WAF.

    Andrews (2012) RMRS-GTR-266 eq. 6 applied to each category's depth
    reproduces ``windrf`` to its two printed decimals, except the two
    litter models 8 and 9, whose 0.36 is not a depth formula.
    """
    for k in (1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13):
        cat = F.WRF_SFIRE_FUEL_CATEGORIES[k]
        waf = float(R.unsheltered_wind_adjustment_factor(cat.fueldepthm))
        assert round(waf, 2) == cat.windrf, k
    u = np.array([2.0, 5.0, 10.0])


def test_log_profile_wind():
    assert float(R.log_profile_wind(7.0, 10.0, 10.0, 0.1)) == pytest.approx(7.0)
    expected = 10.0 * math.log(6.096 / 0.05) / math.log(10.0 / 0.05)
    assert float(R.log_profile_wind(10.0, 10.0, 6.096, 0.05)) == pytest.approx(
        expected)


# --------------------------------------------------------------------------
# full model against published values
# --------------------------------------------------------------------------

# [A18] Table 8a-8e: characteristic SAV (ft-1), bulk density (lb ft-3) and
# relative packing ratio, printed to 0 / 2 / 2 decimals.
TABLE_8 = {
    "FM1": (3500, 0.03, 0.25), "FM2": (2784, 0.18, 1.14),
    "FM3": (1500, 0.06, 0.21), "FM4": (1739, 0.12, 0.52),
    "FM5": (1683, 0.08, 0.33), "FM6": (1564, 0.11, 0.43),
    "FM7": (1552, 0.09, 0.34), "FM8": (1889, 1.15, 5.17),
    "FM9": (2484, 0.80, 4.50), "FM10": (1764, 0.55, 2.35),
    "FM11": (1182, 0.53, 1.62), "FM12": (1145, 0.69, 2.06),
    "FM13": (1159, 0.89, 2.68),
    "GR1": (2054, 0.05, 0.22), "GR2": (1820, 0.05, 0.22),
    "GR3": (1290, 0.05, 0.15), "GR4": (1826, 0.05, 0.22),
    "GR5": (1631, 0.09, 0.35), "GR6": (2006, 0.11, 0.51),
    "GR7": (1834, 0.10, 0.43), "GR8": (1302, 0.10, 0.33),
    "GR9": (1612, 0.10, 0.40),
    "GS1": (1832, 0.07, 0.30), "GS2": (1827, 0.08, 0.35),
    "GS3": (1614, 0.08, 0.33), "GS4": (1631, 0.28, 1.12),
    "SH1": (1674, 0.09, 0.36), "SH2": (1672, 0.38, 1.56),
    "SH3": (1371, 0.18, 0.64), "SH4": (1682, 0.07, 0.30),
    "SH5": (1252, 0.07, 0.21), "SH6": (1144, 0.13, 0.39),
    "SH7": (1233, 0.11, 0.35), "SH8": (1386, 0.16, 0.57),
    "SH9": (1378, 0.16, 0.56),
    "TU1": (1606, 0.28, 1.12), "TU2": (1767, 0.19, 0.82),
    "TU3": (1611, 0.11, 0.45), "TU4": (2216, 0.60, 3.06),
    "TU5": (1224, 0.64, 2.03),
    "TL1": (1716, 1.56, 6.49), "TL2": (1806, 1.35, 5.87),
    "TL3": (1532, 0.84, 3.19), "TL4": (1568, 0.71, 2.75),
    "TL5": (1713, 0.62, 2.56), "TL6": (1936, 0.73, 3.37),
    "TL7": (1229, 1.12, 3.56), "TL8": (1770, 1.27, 5.42),
    "TL9": (1733, 1.08, 4.52),
    "SB1": (1653, 0.71, 2.87), "SB2": (1884, 0.59, 2.63),
    "SB3": (1935, 0.43, 1.97), "SB4": (1907, 0.24, 1.08),
}


@pytest.mark.parametrize("code", sorted(TABLE_8))
def test_characteristic_fuel_bed_values_table_8(code):
    sav, rho_b, rpr = TABLE_8[code]
    # Moisture does not enter these, and neither does curing: the dead
    # herbaceous class carries the live herbaceous SAV.
    s = _fire(code, 0.06, 0.6, 0.9, 0.0, 0.0)
    assert float(s.characteristic_sav) / F.PER_FT == pytest.approx(sav, abs=0.5)
    assert float(s.bulk_density) / F.LB_PER_FT3 == pytest.approx(rho_b,
                                                                 abs=0.0051)
    assert float(s.relative_packing_ratio) == pytest.approx(rpr, abs=0.0051)


# [A18] Tables 22, 23, 24 for five models: wind factor at 6 and 12 mi/h,
# packing ratio and slope factor at 20/50/100 %, and the effective wind
# (mi/h) of that slope with no wind.
TABLE_22_24 = {
    "FM2": ((20.5, 72.8), 0.0057, (1.0, 6.2, 24.8), (1.1, 3.1, 6.7)),
    "FM9": ((12.9, 42.7), 0.0250, (0.6, 4.0, 16.0), (1.0, 3.0, 6.8)),
    "GR2": ((31.1, 85.2), 0.0016, (1.5, 9.1, 36.5), (0.7, 2.6, 6.7)),
    "TU5": ((11.1, 25.1), 0.0201, (0.7, 4.3, 17.0), (0.6, 2.6, 8.6)),
    "TL8": ((9.2, 25.0), 0.0397, (0.6, 3.5, 13.9), (0.8, 3.0, 8.0)),
}


@pytest.mark.parametrize("code", sorted(TABLE_22_24))
def test_wind_slope_factors_and_effective_wind_tables_22_to_24(code):
    phi_w, beta, phi_s, u_eff = TABLE_22_24[code]
    for wind, expected in zip((6.0, 12.0), phi_w):
        assert float(_fire(code, 0.06, 1.0, 1.0, wind, 0.0).phi_wind) == (
            pytest.approx(expected, abs=0.051))
    for slope, ps, ue in zip((0.2, 0.5, 1.0), phi_s, u_eff):
        s = _fire(code, 0.06, 1.0, 1.0, 0.0, slope)
        assert float(s.packing_ratio) == pytest.approx(beta, abs=5.1e-5)
        assert float(s.phi_slope) == pytest.approx(ps, abs=0.051)
        assert float(s.effective_wind) / MPH == pytest.approx(ue, abs=0.051)


def test_live_moisture_of_extinction_table_21():
    # [A18] Table 21, percent, at dead fuel moisture 6 / 10 / 14 %.
    table = {"SH2": (98, 44, 15), "SH6": (616, 510, 403),
             "SH8": (184, 160, 136)}
    for code, row in table.items():
        for dead, expected in zip((0.06, 0.10, 0.14), row):
            mxl = float(_fire(code, dead, 1.0, 1.0, 0.0, 0.0)
                        .moisture_extinction_live)
            assert 100.0 * mxl == pytest.approx(expected, abs=0.51)


def test_load_partition_table_17():
    """[A18] Table 17: 2 t/ac split between 1-h (2500 ft-1) and 10-h.

    Depth 1 ft, dead moisture of extinction 20 %, dead moisture 5 %, mid-
    flame wind 5 mi/h, no slope; rate of spread printed in ft/min.
    """
    t_ac = F.TONS_PER_ACRE_IN_LB_PER_FT2 * F.LB_PER_FT2
    rows = ((2.0, 0.0, 2500, 81.6, 17.8), (1.0, 1.0, 2400, 40.2, 17.9),
            (0.1, 1.9, 1417, 32.0, 20.8), (0.0, 2.0, 109, 0.0, 81.9))
    for w1, w10, sav, ros, phi_w in rows:
        custom = F.FuelModel(
            0, "T17", "Table 17", False, w1 * t_ac, w10 * t_ac, 0.0, 0.0, 0.0,
            2500 * F.PER_FT, 1500 * F.PER_FT, 1500 * F.PER_FT, F.FT, 0.20)
        s = _fire(custom, 0.05, 1.0, 1.0, 5.0, 0.0)
        assert float(s.characteristic_sav) / F.PER_FT == pytest.approx(sav,
                                                                       abs=0.5)
        assert float(s.phi_wind) == pytest.approx(phi_w, abs=0.051)
        assert float(s.ros) / FT_MIN == pytest.approx(ros, abs=0.051)


def test_dynamic_curing_gr2_figures_10_and_11():
    """[A18] fig. 10-11: GR2, dead 5 %, mid-flame 6 mi/h, no slope (ft/min)."""
    for herb, expected in ((0.90, 24.1), (1.00, 6.5)):
        s = _fire("GR2", 0.05, herb, herb, 6.0, 0.0)
        assert float(s.ros) / FT_MIN == pytest.approx(expected, abs=0.051)
    for herb, expected in ((0.90, 31.3), (1.00, 29.1)):
        s = _fire("GR2", 0.05, herb, herb, 6.0, 0.0, curing=0.5)
        assert float(s.ros) / FT_MIN == pytest.approx(expected, abs=0.051)


def test_wind_limit_gr1_figure_46():
    """[A18] sec 5.4.4: GR1, dead 8 %, live 100 %, no slope.

    The Rothermel (1972) limit is 1.6 mi/h and caps the rate at 0.7 ft/min;
    without it, 9 mi/h gives 8.2 ft/min.
    """
    limited = _fire("GR1", 0.08, 1.0, 1.0, 9.0, 0.0, wind_limit="rothermel1972")
    assert bool(limited.wind_limited)
    assert float(limited.effective_wind) / MPH == pytest.approx(1.6, abs=0.051)
    assert float(limited.ros) / FT_MIN == pytest.approx(0.7, abs=0.051)
    free = _fire("GR1", 0.08, 1.0, 1.0, 9.0, 0.0)
    assert not bool(free.wind_limited)
    assert float(free.ros) / FT_MIN == pytest.approx(8.2, abs=0.051)
    # Andrews et al. (2013): U = 96.8 I_R^(1/3), a higher limit here.
    andrews = _fire("GR1", 0.08, 1.0, 1.0, 9.0, 0.0, wind_limit="andrews2013")
    ir = float(andrews.reaction_intensity) / F.BTU_PER_FT2_MIN
    assert float(andrews.effective_wind) / FT_MIN == pytest.approx(
        96.8 * ir ** (1.0 / 3.0))
    assert float(limited.ros) < float(andrews.ros) < float(free.ros)


def test_dead_moisture_sensitivity_sh8():
    """[A18] sec 5.3.4: SH8, live 50 %, 6 mi/h: 14 ft/min at 28 %, 26 at 27 %."""
    for dead, expected in ((0.28, 14.0), (0.27, 26.0)):
        s = _fire("SH8", dead, 0.5, 0.5, 6.0, 0.0)
        assert float(s.ros) / FT_MIN == pytest.approx(expected, abs=0.51)


def test_behaveplus_gs4_regression():
    """[BP] ``testSimpleSurface``: GS4, 1/10/100-h 6/7/8 %, herb 60 %, woody
    90 %, 5 mi/h mid-flame wind (WAF 1) blowing up a 30 % slope."""
    s = R.rothermel_surface_fire(124, 0.06, 0.07, 0.08, 0.60, 0.90,
                                 5.0 * MPH, 0.30)
    assert float(s.ros) / CH_H == pytest.approx(34.011429, abs=2e-6)
    assert float(s.flame_length) / F.FT == pytest.approx(15.811421, abs=2e-6)
    assert float(s.fireline_intensity) / F.BTU_PER_FT_S == pytest.approx(
        2292.684759, abs=2e-6)
    # testSurfaceSingleFuelModel: characteristic SAV and live Mx (percent).
    assert float(s.characteristic_sav) / F.PER_FT == pytest.approx(
        1631.128734, abs=2e-6)
    assert 100.0 * float(s.moisture_extinction_live) == pytest.approx(
        137.968551, abs=2e-6)


# [A82]: (rate of spread ch/h, flame length ft) at 8 % dead, 100 % live,
# 5 mi/h mid-flame wind, no slope.
ANDERSON_1982 = {1: (78, 4.0), 3: (104, 12.0), 4: (75, 19.0), 9: (7.5, 2.6),
                 10: (7.9, 4.8)}


@pytest.mark.parametrize("number", sorted(ANDERSON_1982))
def test_anderson_1982_representative_behaviour(number):
    """Anderson's numbers are two-figure values from the 1982 programs; the
    current model sits within 6 % of each (fuel model 4 is 4.8 % low)."""
    ros, flame = ANDERSON_1982[number]
    s = _fire(number, 0.08, 1.0, 1.0, 5.0, 0.0)
    assert float(s.ros) / CH_H == pytest.approx(ros, rel=0.06)
    assert float(s.flame_length) / F.FT == pytest.approx(flame, rel=0.06)


# --------------------------------------------------------------------------
# physical properties of the full model
# --------------------------------------------------------------------------

def test_zero_spread_at_or_above_dead_moisture_of_extinction():
    for number in (1, 3, 8, 9, 11, 12, 13, 181, 186, 201):
        mx = F.fuel_model(number).moisture_extinction_dead
        for m in (mx, mx + 0.01, 0.6):
            s = _fire(number, m, 1.0, 1.0, 10.0, 0.5)
            assert float(s.ros) == 0.0, (number, m)
            assert float(s.reaction_intensity) == 0.0
        assert float(_fire(number, mx - 0.01, 1.0, 1.0, 5.0, 0.0).ros) > 0.0
    # With live fuel, the live moisture of extinction never falls below the
    # dead one, so wet dead fuel with live fuel above its own extinction
    # does not burn either.
    for code in ("FM4", "SH2", "GS4", "TU5"):
        mx = F.fuel_model(code).moisture_extinction_dead
        assert float(_fire(code, mx, 3.0, 3.0, 10.0, 0.5).ros) == 0.0


def test_non_burnable_and_unknown_numbers():
    s = _fire(np.array([91, 98, 1]), 0.06, 1.0, 1.0, 5.0, 0.2)
    assert s.ros[0] == 0.0 and s.ros[1] == 0.0 and s.ros[2] > 0.0
    assert float(_fire(93, 0.06, 1.0, 1.0, 5.0, 0.2).ros) == 0.0
    assert float(_fire("NB9", 0.06, 1.0, 1.0, 5.0, 0.2).ros) == 0.0
    nb1 = F.fuel_model(91)
    assert nb1.code == "NB1" and not nb1.burnable
    assert F.fuel_model("nb8").number == 98
    # 94-97 are reserved, not defined: bad data must not read as a firebreak.
    for bad in (np.array([1, 50]), np.array([95, 1]), 95):
        with pytest.raises(KeyError):
            _fire(bad, 0.06, 1.0, 1.0, 5.0, 0.2)
    # A fractional or non-finite code is not silently another model.
    for bad in (10.7, np.array([101.5, 1.0]), np.array([np.nan, 1.0])):
        with pytest.raises(ValueError):
            _fire(bad, 0.06, 1.0, 1.0, 5.0, 0.2)
    np.testing.assert_array_equal(
        _fire(np.array([10.0, 124.0]), 0.06, 1.0, 1.0, 5.0, 0.2).ros,
        _fire(np.array([10, 124]), 0.06, 1.0, 1.0, 5.0, 0.2).ros)


def test_malformed_custom_model_is_refused():
    base = F.fuel_model(1)
    for field, value in (("depth", 0.0), ("sav_1h", 0.0),
                         ("moisture_extinction_dead", 0.0),
                         ("load_1h", -0.1), ("particle_density", np.inf)):
        kwargs = dict(base.__dict__, **{field: value})
        with pytest.raises(ValueError):
            F.FuelModel(**kwargs)


def test_wind_limit_flag_only_where_it_binds():
    wet = _fire(1, 0.5, 1.0, 1.0, 5.0, 0.1, wind_limit="rothermel1972")
    assert float(wet.ros) == 0.0 and not bool(wet.wind_limited)
    dry = _fire(1, 0.04, 1.0, 1.0, 1.0, 0.0, wind_limit="rothermel1972")
    assert not bool(dry.wind_limited)


def test_monotonic_in_wind():
    wind = np.linspace(0.0, 20.0, 81)            # m/s
    numbers = sorted(F.STANDARD_FUEL_MODELS)
    grid = np.array(numbers)[:, None]
    s = R.rothermel_surface_fire(grid, 0.06, 0.07, 0.08, 0.6, 0.9,
                                 wind[None, :], 0.1)
    assert np.all(np.diff(s.ros, axis=1) > 0.0)
    for limit in ("rothermel1972", "andrews2013"):
        lim = R.rothermel_surface_fire(grid, 0.06, 0.07, 0.08, 0.6, 0.9,
                                       wind[None, :], 0.1, wind_limit=limit)
        assert np.all(np.diff(lim.ros, axis=1) >= 0.0)
        assert np.all(lim.ros <= s.ros * (1.0 + 1e-14))


def test_slope_symmetry_and_growth():
    half = np.linspace(0.0, 1.0, 21)
    slope = np.concatenate([-half[::-1], half])
    s = R.rothermel_surface_fire(np.array([1, 4, 10, 124, 165])[:, None],
                                 0.06, 0.07, 0.08, 0.6, 0.9, 2.0,
                                 slope[None, :])
    np.testing.assert_array_equal(s.ros, s.ros[:, ::-1])
    upslope = s.ros[:, 21:]
    assert np.all(np.diff(upslope, axis=1) > 0.0)


def test_moisture_monotone_and_broadcasting():
    dead = np.linspace(0.02, 0.30, 15)
    s = R.rothermel_surface_fire(10, dead, dead, dead, 1.0, 1.0, 3.0, 0.0)
    assert s.ros.shape == (15,)
    assert np.all(np.diff(s.ros) <= 0.0)
    assert s.ros[-1] == 0.0                     # above FM10's 25 %


# --------------------------------------------------------------------------
# WRF v4.7.1 fire_ros: physical properties (bit parity is in
# tests/test_fire_ros_wrf471_parity.py)
# --------------------------------------------------------------------------

def _wrf(cat, fmc, table="namelist"):
    return W.set_fire_params(np.asarray(cat, np.float32),
                             np.asarray(fmc, np.float32), table=table)


def test_wrf_backing_and_flanking_spread_is_the_base_rate():
    p = _wrf(np.array([1, 2, 3, 8, 10]), 0.06)
    base = p.r_0 * W.r4(".00508")
    for nx, ny in ((-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        back, wind, slope = W.fire_ros(p, np.float32(5.0), np.float32(0.0),
                                       np.float32(0.4), np.float32(0.0),
                                       np.float32(nx), np.float32(ny))
        np.testing.assert_array_equal(back, base)
        np.testing.assert_array_equal(wind, 0.0)
        np.testing.assert_array_equal(slope, 0.0)
    head = W.fire_ros_total(p, np.float32(5.0), np.float32(0.0),
                            np.float32(0.4), np.float32(0.0),
                            np.float32(1.0), np.float32(0.0))
    assert np.all(head > base)


def test_wrf_total_is_zero_at_and_above_extinction():
    """WRF does not clip the moisture damping: above extinction r_0 < 0 and
    the level set's max(rr, 0) stops the fire.  At extinction the cubic is
    its float32 rounding residue."""
    for k in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13):
        mce = float(W.r4(F.WRF471_NAMELIST["fuelmce"][k - 1]))
        at = _wrf(k, mce)
        assert abs(float(at.r_0)) < 1e-5, k
        for fmc in (mce + 0.01, mce + 0.05, 0.9):
            p = _wrf(k, fmc)
            assert float(p.r_0) < 0.0, (k, fmc)
            rr = W.fire_ros_total(p, np.float32(6.0), np.float32(0.0),
                                  np.float32(0.5), np.float32(0.0),
                                  np.float32(1.0), np.float32(0.0))
            assert float(rr) == 0.0, (k, fmc)
    # Chaparral (category 4) spreads by wind alone (Clark et al. 2004):
    # ros_base is not set there and the port returns 0.
    p = _wrf(4, 0.5)
    rb, rw, rs = W.fire_ros(p, *(np.float32(v) for v in (0, 0, 0, 0, 1, 0)))
    assert float(rb) == 0.0 and float(rs) == 0.0
    assert float(rw) == float(W.r4(".03333"))
    p = _wrf(np.array([14, 15, 91]), 0.06)
    np.testing.assert_array_equal(p.r_0, 0.0)
    np.testing.assert_array_equal(
        W.fire_ros_total(p, *(np.float32(v) for v in (5, 0, 0.4, 0, 1, 0))), 0.0)


def test_wrf_category_validation():
    with pytest.raises(ValueError):
        _wrf(np.array([0, 1]), 0.06)                # outside ksb
    with pytest.raises(ValueError):
        _wrf(np.array([205]), 0.06)
    with pytest.raises(ValueError):
        _wrf(np.array([101]), 0.06)                 # S&B, namelist has 13
    # WRF v4.7.1's compiled defaults declare nfuelcats = 53 but put SB4
    # (204) at category 54: set_fire_params crashes on it, and so does this.
    with pytest.raises(ValueError):
        _wrf(np.array([204]), 0.06, table="defaults")
    p = _wrf(np.array([101, 203, 50]), 0.06, table="defaults")
    assert p.r_0[0] > 0.0 and p.r_0[1] > 0.0 and p.r_0[2] == 0.0
    np.testing.assert_array_equal(_wrf(np.array([10.9]), 0.06).r_0,
                                  _wrf(np.array([10]), 0.06).r_0)


def test_wrf_monotone_in_wind_and_mirror_symmetric():
    p = _wrf(np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13])[:, None], 0.06)
    wind = np.linspace(-10.0, 30.0, 81, dtype=np.float32)[None, :]
    f32 = np.float32
    rr = W.fire_ros_total(p, wind, f32(0), f32(0), f32(0), f32(1), f32(0))
    # Monotone up to the 6 m/s cap, where WRF's sequential trim can land a
    # little under it.
    below = rr < 5.9
    assert np.all(np.diff(rr, axis=1)[below[:, 1:]] >= 0.0)
    # WRF's float32 trim (rw - excess*rw/(rw+rs), then the sum) lands a few
    # ULPs either side of the cap: 6.0000024 is seen.
    assert np.all(rr <= 6.0 * (1.0 + 1e-6))
    half = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    up = W.fire_ros(p, f32(0), f32(0), half[None, :], f32(0), f32(1), f32(0))
    down = W.fire_ros(p, f32(0), f32(0), -half[None, :], f32(0), f32(-1),
                      f32(0))
    for a, b in zip(up, down):
        np.testing.assert_array_equal(a, b)
    assert np.all(np.diff(up[2], axis=1) >= 0.0)


def test_wrf_category_field_and_midflame_wind():
    cat = np.array([[1, 4], [10, 14]])
    np.testing.assert_array_equal(W.category_field(cat, "weight"),
                                  np.float32([[7., 180.], [900., 7.]]))
    with pytest.raises(KeyError):
        W.category_field(cat, "name")
    u = np.array([2.0, 5.0, 10.0])
    np.testing.assert_allclose(W.midflame_wind(u, 3), 0.44 * u)
    np.testing.assert_allclose(W.midflame_wind(u, np.array([1, 4, 14])),
                               u * np.array([0.36, 0.55, 1e-7]))


# --------------------------------------------------------------------------
# the full model through WRF's fire_ros
# --------------------------------------------------------------------------

def test_full_model_fire_ros_reproduces_head_fire_and_backs_at_r0():
    """Wind and slope along the normal give back the head-fire rate, up to
    WRF's literal m/s -> ft/min factors (196.850, .00508: 2e-6 raised to B);
    against the normal the rate is r_0."""
    numbers = np.array([1, 2, 3, 4, 8, 10, 102, 124, 145, 165, 183, 202])
    wind, slope = 2.5, 0.3
    p = R.rothermel_fire_params(numbers, 0.06, 0.07, 0.08, 0.6, 0.9)
    rb, rw, rs = R.rothermel_fire_ros(p, wind, 0.0, slope, 0.0, 1.0, 0.0)
    head = R.rothermel_surface_fire(numbers, 0.06, 0.07, 0.08, 0.6, 0.9,
                                    wind, slope)
    np.testing.assert_allclose(rb + rw + rs, head.ros, rtol=5e-6)
    np.testing.assert_allclose(rb, head.ros_no_wind_no_slope, rtol=1e-6)
    back = R.rothermel_fire_ros(p, wind, 0.0, slope, 0.0, -1.0, 0.0)
    np.testing.assert_array_equal(back[0], rb)
    np.testing.assert_array_equal(back[1], 0.0)
    np.testing.assert_array_equal(back[2], 0.0)


# --------------------------------------------------------------------------
# CPU == GPU, raw word for raw word
# --------------------------------------------------------------------------

def _random_cells(n, seed=0):
    rng = np.random.default_rng(seed)
    numbers = np.concatenate([np.array(sorted(F.STANDARD_FUEL_MODELS)),
                              [91, 98]])
    return dict(
        fuel=rng.choice(numbers, n), m1=rng.uniform(0.02, 0.45, n),
        herb=rng.uniform(0.2, 2.5, n), woody=rng.uniform(0.5, 2.5, n),
        wind=rng.uniform(-1.0, 25.0, n), slope=rng.uniform(-1.5, 1.5, n),
        curing=rng.uniform(-0.2, 1.2, n),
        cat=rng.integers(1, 15, n), fmc=rng.uniform(0.0, 0.6, n),
        vec=rng.normal(size=(6, n)) * np.array([[6], [6], [.4], [.4], [1], [1]]))


def _words(a):
    a = np.ascontiguousarray(a)
    return a.view({1: np.uint8, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize])


def _assert_bits(host, dev, label):
    # No cupy import here: a helper that imports it would mark the whole
    # module gpu (conftest._cupy_scope); the device array copies itself.
    assert hasattr(dev, "__cuda_array_interface__"), label
    got = dev.get()
    assert got.dtype == np.asarray(host).dtype, label
    diff = int(np.count_nonzero(_words(got) != _words(np.asarray(host))))
    assert diff == 0, f"{label}: {diff} of {got.size} words differ"


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("wind_limit", R.WIND_LIMITS)
@pytest.mark.parametrize("use_curing", (False, True))
def test_gpu_full_model_is_bit_equal_to_numpy(wind_limit, use_curing):
    import cupy as cp
    d = _random_cells(20000)
    args = (d["fuel"], d["m1"], d["m1"] + 0.01, d["m1"] + 0.02, d["herb"],
            d["woody"], d["wind"], d["slope"])
    curing = d["curing"] if use_curing else None
    host = R.rothermel_surface_fire(*args, wind_limit=wind_limit,
                                    curing=curing)
    dev = R.rothermel_surface_fire(
        *(cp.asarray(a) for a in args), wind_limit=wind_limit,
        curing=None if curing is None else cp.asarray(curing))
    for name in R.SurfaceFire.__dataclass_fields__:
        _assert_bits(getattr(host, name), getattr(dev, name), name)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("fire_advection", (0, 1))
def test_gpu_full_model_fire_ros_is_bit_equal_to_numpy(fire_advection):
    import cupy as cp
    d = _random_cells(20000, seed=1)
    host_p = R.rothermel_fire_params(d["fuel"], d["m1"], 0.07, 0.08, d["herb"],
                                     d["woody"])
    dev_p = R.rothermel_fire_params(cp.asarray(d["fuel"]), cp.asarray(d["m1"]),
                                    0.07, 0.08, cp.asarray(d["herb"]),
                                    cp.asarray(d["woody"]))
    for name in R.RothermelFireParams.__dataclass_fields__:
        _assert_bits(getattr(host_p, name), getattr(dev_p, name), name)
    for ros_max in (6.0, 0.05):
        host = R.rothermel_fire_ros(host_p, *d["vec"], ros_max=ros_max,
                                    fire_advection=fire_advection)
        dev = R.rothermel_fire_ros(dev_p, *(cp.asarray(v) for v in d["vec"]),
                                   ros_max=ros_max,
                                   fire_advection=fire_advection)
        for name, h, g in zip(("ros_base", "ros_wind", "ros_slope"), host, dev):
            _assert_bits(h, g, f"{name} ros_max={ros_max}")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("table", W.TABLES)
@pytest.mark.parametrize("fire_advection", (0, 1))
def test_gpu_wrf_is_bit_equal_to_numpy_beyond_the_oracle(table, fire_advection):
    """Random cells far from the oracle's grid, both tables, both modes."""
    import cupy as cp
    d = _random_cells(20000, seed=2)
    cat = d["cat"] if table == "namelist" else np.random.default_rng(3).choice(
        np.array(sorted(set(F.WRF471_KSB) - {204}) + [14, 50]), d["cat"].size)
    cat = cat.astype(np.float32)
    fmc = d["fmc"].astype(np.float32)
    host_p = W.set_fire_params(cat, fmc, table=table)
    dev_p = W.set_fire_params(cp.asarray(cat), cp.asarray(fmc), table=table)
    for name in W.FireParams.__dataclass_fields__:
        _assert_bits(getattr(host_p, name), getattr(dev_p, name), name)
    vec = d["vec"].astype(np.float32)
    host = W.fire_ros(host_p, *vec, fire_advection=fire_advection)
    dev = W.fire_ros(dev_p, *(cp.asarray(v) for v in vec),
                     fire_advection=fire_advection)
    for name, h, g in zip(("ros_base", "ros_wind", "ros_slope"), host, dev):
        _assert_bits(h, g, name)
    _assert_bits(W.fire_ros_total(host_p, *vec, fire_advection=fire_advection),
                 W.fire_ros_total(dev_p, *(cp.asarray(v) for v in vec),
                                  fire_advection=fire_advection), "total")


@pytest.mark.gpu
@requires_gpu
def test_gpu_scalar_model_and_resident_tables():
    import cupy as cp
    dead = np.linspace(0.03, 0.2, 64)
    herb = np.linspace(0.3, 1.5, 64)
    for fuel, curing in (("GR2", 0.5), ("GS4", None), (F.fuel_model(10), None)):
        host = R.rothermel_surface_fire(fuel, dead, dead, dead, herb, herb,
                                        3.0, 0.2, curing=curing)
        dev = R.rothermel_surface_fire(fuel, cp.asarray(dead), cp.asarray(dead),
                                       cp.asarray(dead), cp.asarray(herb),
                                       cp.asarray(herb), 3.0, 0.2,
                                       curing=curing)
        _assert_bits(host.ros, dev.ros, str(fuel))
    # The fuel tables are uploaded once per namespace and then reused.
    assert R._standard_table(cp)[0] is R._standard_table(cp)[0]
    assert isinstance(R._standard_table(cp)[0], cp.ndarray)
    assert W._table_device(cp, "namelist") is W._table_device(cp, "namelist")
