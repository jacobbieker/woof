"""A layer source's RUC soil moisture stays physical at the 0 m level.

``init_soil_3_real`` anchors a layer source's moisture at 0 m with a linear
extrapolation off the top two layer midpoints
(``share/module_soil_pre.F:1945-1947``), which goes below zero wherever the
top layer is much drier than the one beneath it.  Measured on the
2026-10-05 06Z ECMWF open-data Greenland preparation (3.42 km, RUC): three
land columns on the north-west coast (ISLTYP 6, IVGTYP 15/19) reached the
RUC cold start with SMOIS(1) = SH2O(1) of -0.0011, -0.0026 and -0.0038, and
``ruc_initialize_cold_start`` refused the run with "RUC smois must remain
within 0..1 m3 m-3".  Their source samples, recovered from the prepared
levels, are a 0.05 top layer over a ~0.30 second layer.

real.exe never hands RUC those values: ``account_for_zero_soil_moisture``'s
RUC arm (``dyn_em/module_initialize_real.F`` v4.6.1 :3435-3445) floors
every level at 0.005 for a layer source and does nothing for a node
source.  These tests hold woof to that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from woof.ingest.ruc_soil import (
    RUC_ZERO_SOIL_MOISTURE,
    RucSoilState,
    account_for_zero_ruc_soil_moisture,
    preprocess_land_surface_soil,
    remap_soil_to_ruc_levels,
)
from woof.ingest.soil_contract import (
    MAPPED_SOIL_MOISTURE,
    MAPPED_SOIL_TEMPERATURE,
)

_ROOT = Path(__file__).resolve().parents[1]

#: The three refused Greenland columns' source moisture, top to bottom, on
#: the ECMWF composition's four layers (char2int2 midpoints 3/17/64/194 cm).
_GREENLAND_MOISTURE = np.array([
    [0.05, 0.05, 0.05],
    [0.2883, 0.2956, 0.3009],
    [0.356, 0.3684, 0.3729],
    [0.4007, 0.451, 0.3524],
], dtype=np.float32)[:, None, :]
#: What WRF's anchor makes of them at 0 m, and what the prepared input held.
_GREENLAND_LEVEL_ONE = (-0.00107, -0.00263, -0.00376)


def _ecmwf_soil_contract():
    composition = _ROOT / "woof" / "authorities" / (
        "rw-wps-ecmwf-open-data-oper-grib2.composition.json")
    return json.loads(composition.read_text())["soil_layers"]


def _remap(moisture, *, geometry="layers", levels=(3, 17, 64, 194)):
    shape = moisture.shape[1:]
    return remap_soil_to_ruc_levels(
        source_temperature=np.full(moisture.shape, 265.0, np.float32),
        source_moisture=moisture,
        source_levels_cm=np.array(levels, dtype=np.int64),
        source_geometry=geometry,
        skin_temperature=np.full(shape, 258.0, np.float32),
        deep_temperature=np.full(shape, 253.0, np.float32),
        landmask=np.ones(shape, np.float32),
    )


def test_wrfs_zero_meter_anchor_takes_the_greenland_columns_below_zero():
    """The remap is WRF's and stays WRF's: the negative anchor is real.exe's."""
    columns = _remap(_GREENLAND_MOISTURE)
    np.testing.assert_allclose(
        columns.soil_moisture[0, 0], _GREENLAND_LEVEL_ONE, atol=2e-5)
    assert np.all(columns.soil_moisture[1:] > 0.0)


def test_the_ruc_floor_is_wrfs_max_on_a_layer_source():
    columns = _remap(_GREENLAND_MOISTURE)
    floored, receipt = account_for_zero_ruc_soil_moisture(
        columns.soil_moisture, source_geometry="layers",
        landmask=np.ones((1, 3)))
    # MAX(smois, 0.005), level by level, exactly as real.exe.
    want = np.maximum(columns.soil_moisture, np.float32(0.005))
    np.testing.assert_array_equal(floored, want)
    assert floored.dtype == np.float32
    np.testing.assert_array_equal(floored[0, 0], np.float32(0.005))
    # Every value already at or above the floor is byte-identical.
    keep = columns.soil_moisture >= np.float32(0.005)
    np.testing.assert_array_equal(
        floored[keep].view(np.uint32),
        columns.soil_moisture[keep].view(np.uint32))
    assert np.all((floored >= 0.0) & (floored <= 1.0))
    assert receipt["total_floored_values"] == 3
    assert receipt["floored_land_columns"] == 3
    assert set(receipt["fields"]) == {"SMOIS_L1"}
    assert receipt["min_pre_floor"] == pytest.approx(-0.00376, abs=2e-5)
    assert receipt["floor_m3_m3"] == pytest.approx(0.005)


def test_the_ruc_floor_is_silent_and_identical_on_a_healthy_column():
    moisture = np.array([0.12, 0.18, 0.24, 0.30], np.float32)[:, None, None]
    columns = _remap(moisture)
    floored, receipt = account_for_zero_ruc_soil_moisture(
        columns.soil_moisture, source_geometry="layers")
    assert floored is columns.soil_moisture
    assert receipt == {}


def test_a_node_source_is_wrfs_no_op():
    """flag_soil_levels = 1: real.exe leaves a RUC node source unfloored."""
    tiny = np.full((9, 1, 2), 0.001, np.float32)
    kept, receipt = account_for_zero_ruc_soil_moisture(
        tiny, source_geometry="levels")
    assert kept is tiny
    assert receipt == {}
    with pytest.raises(ValueError, match="source_geometry"):
        account_for_zero_ruc_soil_moisture(tiny, source_geometry="nodes")


def test_a_missing_value_is_left_for_the_cold_start_refusal():
    values = np.array([[np.nan, -0.002, 0.2]], np.float32)[:, None, :]
    floored, receipt = account_for_zero_ruc_soil_moisture(
        values, source_geometry="layers")
    assert np.isnan(floored[0, 0, 0])
    assert floored[0, 0, 1] == np.float32(0.005)
    assert floored[0, 0, 2] == np.float32(0.2)
    assert receipt["total_floored_values"] == 1


def _greenland_mapped_fields():
    shape = _GREENLAND_MOISTURE.shape[1:]
    return shape, {
        "LANDSEA": np.ones(shape),
        "SKINTEMP": np.full(shape, 258.0),
        "SST": np.full(shape, 271.0),
        "SEAICE": np.zeros(shape),
        "SNOW": np.zeros(shape),
        "SNOWH": np.zeros(shape),
        MAPPED_SOIL_TEMPERATURE: np.full((4, *shape), 262.0),
        MAPPED_SOIL_MOISTURE: _GREENLAND_MOISTURE.astype(np.float64),
    }


@pytest.mark.parametrize("fractional_seaice", [False, True])
def test_the_ecmwf_mapped_route_hands_ruc_a_physical_column(fractional_seaice):
    """The refused preparation, end to end through the land-surface seam."""
    shape, fields = _greenland_mapped_fields()
    state = preprocess_land_surface_soil(
        fields, sf_surface_physics=3, num_soil_layers=9,
        soil_type=np.full(shape, 6),
        deep_soil_temperature=np.full(shape, 253.0),
        soil_layer_contract=_ecmwf_soil_contract(),
        landmask=np.ones(shape),
        water_temperature_policy="wrf_compat",
        fractional_seaice=fractional_seaice)
    assert isinstance(state, RucSoilState)
    for name in ("soil_moisture", "liquid_moisture"):
        values = np.asarray(getattr(state, name))
        assert values.shape == (9, *shape)
        assert np.all(np.isfinite(values))
        assert np.all((values >= 0.0) & (values <= 1.0)), name
    np.testing.assert_array_equal(
        np.asarray(state.soil_moisture)[0], RUC_ZERO_SOIL_MOISTURE)
    np.testing.assert_array_equal(
        np.asarray(state.liquid_moisture), np.asarray(state.soil_moisture))
    assert state.moisture_floor["total_floored_values"] == 3
    assert state.moisture_floor["floored_land_columns"] == 3


def test_a_healthy_ruc_preparation_carries_no_floor_receipt():
    shape, fields = _greenland_mapped_fields()
    fields[MAPPED_SOIL_MOISTURE] = np.stack(
        [np.full(shape, value) for value in (0.12, 0.18, 0.24, 0.30)])
    state = preprocess_land_surface_soil(
        fields, sf_surface_physics=3, num_soil_layers=9,
        soil_type=np.full(shape, 6),
        deep_soil_temperature=np.full(shape, 253.0),
        soil_layer_contract=_ecmwf_soil_contract(),
        landmask=np.ones(shape),
        water_temperature_policy="wrf_compat")
    assert state.moisture_floor == {}
    assert np.all(np.asarray(state.soil_moisture) >= 0.005)


def test_the_ruc_cold_start_refusal_is_kept():
    """The fix is in preparation; the RUC 0..1 check is not weakened."""
    source = (_ROOT / "woof" / "core" / "ruc.py").read_text()
    assert "RUC smois must remain within 0..1 m3 m-3" in source
