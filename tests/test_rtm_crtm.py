"""woof.rtm: CRTM atmospheres from model columns, and the opt-in CRTM operator.

The profile tests always run.  The operator tests need a library built by
``python -m woof.rtm.crtm_build`` and CRTM's coefficients, named by
WOOF_CRTM_LIB and WOOF_CRTM_COEFFS; without them they skip.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pytest

from woof.rtm.profiles import Profiles, profiles_from_pressure_levels, profiles_from_wrf_state, standard_ozone_ppmv


def _wrf_state(n=3, nz=40):
    eta = np.linspace(1.0, 0.0, nz + 1)
    p_top, psfc = 5000.0, np.array([101000.0, 95000.0, 85000.0])[:n]
    full = p_top + eta[None, :] * (psfc[:, None] - p_top)
    p = 0.5 * (full[:, 1:] + full[:, :-1])
    t = np.maximum(288.0 - 6.5e-3 * 7400 * np.log(1.0e5 / p), 215.0)
    theta = t * (1.0e5 / p) ** (287.04 / 1004.5)
    q = 0.012 * (p / 1.0e5) ** 3
    return dict(T=theta - 300.0, P=p - 0.9 * p, PB=0.9 * p, QVAPOR=q, PSFC=psfc, P_TOP=p_top,
                TSK=np.array([290.0, 280.0, 270.0])[:n], U10=np.array([3.0, 0.0, -4.0])[:n],
                V10=np.array([4.0, -2.0, 0.0])[:n], LANDMASK=np.array([0.0, 1.0, 1.0])[:n],
                XICE=np.zeros(n), SNOWC=np.array([0.0, 0.0, 1.0])[:n], HGT=np.array([0.0, 500.0, 1500.0])[:n]), t


def test_wrf_state_columns_run_top_first_with_wrf_temperature_and_units():
    state, t = _wrf_state()
    prof = profiles_from_wrf_state(state)
    assert prof.layers == 40 and prof.count == 3
    np.testing.assert_allclose(prof.temperature_k, t[:, ::-1], rtol=1e-12)
    np.testing.assert_allclose(prof.water_vapour_gkg[:, -1], state["QVAPOR"][:, 0] * 1000.0)
    np.testing.assert_allclose(prof.level_pressure_hpa[:, 0], 50.0)
    np.testing.assert_allclose(prof.level_pressure_hpa[:, -1], state["PSFC"] / 100.0)
    # Ocean, bare land and snow-covered land.
    np.testing.assert_allclose(prof.water_fraction, [1, 0, 0])
    np.testing.assert_allclose(prof.land_fraction, [0, 1, 0])
    np.testing.assert_allclose(prof.snow_fraction, [0, 0, 1])
    np.testing.assert_allclose(prof.wind_speed_ms, [5, 2, 4])
    # From the south-west, from the north, from the east.
    np.testing.assert_allclose(prof.wind_direction_deg, [216.869898, 0.0, 90.0], atol=1e-5)


def test_wrf_state_moist_theta_and_psfc_check():
    state, t = _wrf_state()
    moist = dict(state, T=(state["T"] + 300.0) * (1 + (461.6 / 287.04) * state["QVAPOR"]) - 300.0)
    prof = profiles_from_wrf_state(moist, use_theta_m=True)
    np.testing.assert_allclose(prof.temperature_k, t[:, ::-1], rtol=1e-12)
    low = dict(state, PSFC=state["PSFC"] * 0.5)
    with pytest.raises(ValueError, match="PSFC"):
        profiles_from_wrf_state(low)


def test_wrf_state_refuses_missing_fields_and_bad_columns():
    state, _ = _wrf_state()
    with pytest.raises(KeyError, match="SNOWC"):
        profiles_from_wrf_state({k: v for k, v in state.items() if k != "SNOWC"})
    prof = profiles_from_wrf_state(state)
    level = prof.level_pressure_hpa.copy()
    level[0, 5] = level[0, 4]
    with pytest.raises(ValueError, match="increase strictly"):
        Profiles(**{**prof.__dict__, "level_pressure_hpa": level})
    with pytest.raises(ValueError, match="sum to 1"):
        Profiles(**{**prof.__dict__, "ice_fraction": prof.ice_fraction + 0.5})


def test_pressure_levels_anchor_on_the_surface_and_drop_levels_below_ground():
    levels = np.array([1000.0, 925.0, 850.0, 700.0, 500.0, 300.0, 100.0, 10.0])
    t = np.tile([300.0, 295.0, 290.0, 280.0, 260.0, 230.0, 210.0, 220.0], (2, 1))
    t[1, 0] = 400.0  # below ground at 900 hPa: must not leak in
    q = np.tile(0.01 * (levels / 1000.0) ** 3, (2, 1))
    prof = profiles_from_pressure_levels(
        levels, t, q, surface_pressure_pa=[101000.0, 90000.0], surface_temperature_k=[301.0, 294.0],
        surface_specific_humidity=[0.011, 0.009], skin_temperature_k=[302.0, 296.0], u10=[0, 1], v10=[1, 0],
        land_fraction=[0, 1], ice_fraction=[0, 0], snow_fraction=[0, 0], surface_altitude_m=[0, 1000],
        layers=30)
    assert prof.layers == 30
    np.testing.assert_allclose(prof.level_pressure_hpa[:, 0], 10.0)
    np.testing.assert_allclose(prof.level_pressure_hpa[:, -1], [1010.0, 900.0])
    assert prof.temperature_k[1].max() < 300.0
    # Specific humidity becomes a mixing ratio in g/kg: the bottom layer (near
    # 993 hPa) lies between the 925 and 1000 hPa values (7.9 and 10 g/kg).
    assert 7.9 / 0.992 < prof.water_vapour_gkg[0, -1] < 10.0 / 0.99
    assert standard_ozone_ppmv([10.0])[0] == pytest.approx(7.5)


_LIB = os.environ.get("WOOF_CRTM_LIB")
_COEFFS = os.environ.get("WOOF_CRTM_COEFFS")
needs_crtm = pytest.mark.skipif(not (_LIB and _COEFFS), reason="WOOF_CRTM_LIB and WOOF_CRTM_COEFFS name no CRTM")


def _geometry(n):
    return dict(sensor_zenith=np.full(n, 40.0), sensor_scan=np.full(n, 6.0), sensor_azimuth=np.zeros(n),
                solar_zenith=np.full(n, 120.0), solar_azimuth=np.zeros(n), latitude=np.full(n, 35.0),
                longitude=np.full(n, -90.0))


@needs_crtm
def test_crtm_abi_brightness_temperatures_and_jacobians():
    from woof.rtm.crtm import ClearSkyOperator, CrtmError

    state, _ = _wrf_state()
    prof = profiles_from_wrf_state(state)
    when = datetime(2025, 3, 1, 18, tzinfo=timezone.utc)
    with ClearSkyOperator(["abi_g16"]) as crtm:
        assert {8, 9, 10, 14} <= set(crtm.channels("abi_g16").tolist())
        with pytest.raises(CrtmError, match="already open"):
            ClearSkyOperator(["abi_g16"])
        with pytest.raises(CrtmError, match="not \\[2\\]"):
            crtm.simulate("abi_g16", prof, _geometry(3), when, channels=[2])
        result = crtm.simulate("abi_g16", prof, _geometry(3), when, channels=[8, 9, 10, 14], jacobian=True)
        bt = result.brightness_temperature
        assert bt.shape == (3, 4)
        assert np.all((bt > 200.0) & (bt < 310.0))
        # Water vapour bands are colder than the window; the window sees near the skin.
        assert np.all(bt[:, 0] < bt[:, 3])
        assert abs(bt[0, 3] - 290.0) < 8.0
        # The water vapour Jacobian is negative (more vapour, colder) and peaks aloft for 6.2 um.
        assert np.all(result.k_water_vapour[:, 0].sum(axis=1) < 0)
        peak = prof.pressure_hpa[0, np.argmax(result.k_temperature[0, 0])]
        assert 150.0 < peak < 550.0
        # dBT/dT against a finite difference at the 6.2 um peak layer.
        k = int(np.argmax(result.k_temperature[0, 0]))
        bumped = Profiles(**{**prof.__dict__, "temperature_k": prof.temperature_k.copy()})
        bumped.temperature_k[:, k] += 0.5
        again = crtm.simulate("abi_g16", bumped, _geometry(3), when, channels=[8])
        fd = (again.brightness_temperature[0, 0] - bt[0, 0]) / 0.5
        assert fd == pytest.approx(result.k_temperature[0, 0, k], rel=0.05)
        with pytest.raises(CrtmError, match="zenith"):
            crtm.simulate("abi_g16", prof, {**_geometry(3), "sensor_zenith": np.full(3, 70.0)}, when)
    with pytest.raises(CrtmError, match="closed"):
        crtm.simulate("abi_g16", prof, _geometry(3), when)


def test_crtm_refuses_without_a_library(monkeypatch, tmp_path):
    from woof.rtm.crtm import ClearSkyOperator, CrtmError

    monkeypatch.delenv("WOOF_CRTM_LIB", raising=False)
    monkeypatch.setenv("WOOF_CRTM_COEFFS", str(tmp_path))
    with pytest.raises(CrtmError, match="missing CRTM coefficients"):
        ClearSkyOperator(["abi_g16"])
    for kind in ("SpcCoeff", "TauCoeff"):
        (tmp_path / f"abi_g16.{kind}.bin").write_bytes(b"")
    with pytest.raises(CrtmError, match="crtm_build"):
        ClearSkyOperator(["abi_g16"])
