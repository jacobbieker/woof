"""The two init ceilings: snow water at 5000 kg m-2, soil moisture at SMCMAX.

Both follow WRF's Noah-MP initialisation (``module_sf_noahmpdrv.F``,
``NOAHMP_INIT``).  The hex path applies them in ``rw_mpas_init``
(``tools/rustwx/crates/rw-mpas/src/init/caps.rs``, unit-tested in Rust);
the WRF path in :func:`woof.ingest.soil.preprocess_noah_soil` and the RUC
arm beside it.  Each capped cell is counted in a receipt and said in one
line on stderr; nothing is said, and nothing changes, below the ceilings.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CAPS_RS = ROOT / "tools/rustwx/crates/rw-mpas/src/init/caps.rs"
SOILPARM = ROOT / "woof/data/noah_tables/SOILPARM.TBL"


def _stas_maxsmc() -> list[float]:
    lines = SOILPARM.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "STAS")
    rows = lines[start + 2:start + 2 + 19]
    return [float(row.split(",")[4]) for row in rows]


# ---------------------------------------------------------------------------
# one table, three spellings
# ---------------------------------------------------------------------------
def test_every_spelling_of_the_ceilings_is_the_packaged_table():
    from woof.core.noah import SOIL_COLS, load_tables, pack_params
    from woof.hex import init_caps
    from woof.ingest import soil

    table = _stas_maxsmc()
    assert list(init_caps.STAS_MAXSMC) == table
    params = pack_params(load_tables())
    np.testing.assert_allclose(
        params.soil[:params.slcats, SOIL_COLS.index("smcmax")], table)
    text = CAPS_RS.read_text()
    block = re.search(r"STAS_MAXSMC: \[f32; 19\] = \[(.*?)\];", text, re.S).group(1)
    assert [float(v) for v in block.replace("\n", " ").split(",") if v.strip()] == table
    cap = re.search(r"SNOW_WATER_CAP_KG_M2: f32 = ([0-9.]+);", text).group(1)
    assert float(cap) == soil.SNOW_WATER_CAP_KG_M2 == init_caps.SNOW_WATER_CAP_KG_M2 == 5000.0
    assert init_caps.LARGEST_LAND_SMCMAX == 0.484


# ---------------------------------------------------------------------------
# the snow ceiling
# ---------------------------------------------------------------------------
def test_snow_cap_below_at_and_above(capsys):
    from woof.ingest.soil import cap_snow_water

    snow = np.array([[0.0, 4999.0, 5000.0, 5001.0, 51_800.0]])
    snowh = snow * 5.0 / 1000.0
    capped, depth, receipt = cap_snow_water(snow, snowh)
    np.testing.assert_array_equal(capped, [[0.0, 4999.0, 5000.0, 5000.0, 5000.0]])
    # Below and at the cap, untouched to the bit; above it the density is kept.
    np.testing.assert_array_equal(depth[0, :3], snowh[0, :3])
    np.testing.assert_allclose(depth[0, 3:], 25.0, rtol=0, atol=1e-12)
    assert receipt["cap_kg_m2"] == 5000.0
    assert receipt["cells_capped"] == 2
    assert receipt["source_max_kg_m2"] == 51_800.0
    # The deepest snow over every cell, as rw_mpas_init reports it.
    assert receipt["source_max_snow_depth_m"] == pytest.approx(259.0)
    assert receipt["bounding_box"] == {"rows": [0, 0], "columns": [3, 4]}
    line = capsys.readouterr().err.strip()
    assert line == (
        "snow cap: capped SNOW at 5000 kg m-2 on 2 of 5 cell(s) (source value up "
        "to 51800 kg m-2, snow depth up to 259 m), snow depth scaled to keep the "
        "density; WRF NOAHMP_INIT's ceiling")


def test_snow_cap_says_nothing_and_changes_nothing_below_it(capsys):
    from woof.ingest.soil import cap_snow_water

    snow = np.array([[0.0, 415.0, 5000.0]], dtype=np.float32)
    snowh = np.array([[0.0, 2.0, 25.0]], dtype=np.float32)
    capped, depth, receipt = cap_snow_water(snow, snowh)
    assert capped is snow and depth is snowh and receipt == {}
    assert capsys.readouterr().err == ""

    # A capped field keeps the dtype it came in.
    capped, depth, _ = cap_snow_water(np.array([[9000.0, 1.0]], np.float32),
                                      np.array([[45.0, 0.005]], np.float32))
    assert capped.dtype == np.float32 and depth.dtype == np.float32


def _land_case(snow, sm, soil_type, *, landsea=None):
    shape = np.shape(snow)
    fields = {
        "LANDSEA": np.ones(shape) if landsea is None else np.asarray(landsea, float),
        "SKINTEMP": np.full(shape, 275.0),
        "TMN": np.full(shape, 275.0),
        "SNOW": np.asarray(snow, dtype=np.float64),
    }
    for name in ("ST000007", "ST007028", "ST028100", "ST100289"):
        fields[name] = np.full(shape, 275.0)
    for name in ("SM000007", "SM007028", "SM028100", "SM100289"):
        fields[name] = np.asarray(sm, dtype=np.float64)
    return fields, np.asarray(soil_type, dtype=np.float64)


def test_wrf_init_admits_the_met_office_ice_sheet_store_and_caps_it(capsys):
    """51,800 kg m-2 over Greenland used to be refused at 20,000.  On land
    ice it is an ice sheet's snow store, real source data, so it is admitted
    and starts at 5000 kg m-2."""
    from woof.ingest.soil import preprocess_noah_soil

    fields, soil_type = _land_case([[51_800.0, 3_000.0, 0.0]],
                                   [[0.30, 0.30, 0.30]], [[16, 6, 6]])
    soil = preprocess_noah_soil(fields, soil_type=soil_type)
    np.testing.assert_array_equal(soil.snow_water, [[5000.0, 3000.0, 0.0]])
    np.testing.assert_allclose(soil.snow_depth, [[25.0, 15.0, 0.0]])
    assert set(soil.init_caps) == {"snow"}
    assert soil.init_caps["snow"]["cells_capped"] == 1
    assert "capped SNOW at 5000 kg m-2 on 1 of 3 cell(s)" in capsys.readouterr().err


def test_the_unit_error_guard_still_refuses_off_the_ice_sheet():
    """The 20,000 kg m-2 ceiling stands everywhere but land ice and the
    patches of ice-sheet values that touch it: a store smeared one cell off
    the ice margin is admitted, the same value on its own is refused."""
    from woof.ingest.soil import preprocess_noah_soil

    # Land ice (16) beside loam (6): the 30,000 on the loam cell is part of
    # the patch that touches the ice, the interpolated ice margin.
    fields, soil_type = _land_case([[51_800.0, 30_000.0, 0.0, 0.0]],
                                   [[0.3] * 4], [[16, 6, 6, 6]])
    soil = preprocess_noah_soil(fields, soil_type=soil_type)
    np.testing.assert_array_equal(soil.snow_water, [[5000.0, 5000.0, 0.0, 0.0]])
    assert soil.init_caps["snow"]["cells_capped"] == 2

    # The same 30,000 with no land ice in reach is the factor of 1000.
    fields, soil_type = _land_case([[51_800.0, 0.0, 30_000.0, 0.0]],
                                   [[0.3] * 4], [[16, 6, 6, 6]])
    with pytest.raises(ValueError, match="plausibility ceiling") as refusal:
        preprocess_noah_soil(fields, soil_type=soil_type)
    message = str(refusal.value)
    assert "1 value(s) of 4" in message and "these are not" in message

    # And no land ice at all: refused exactly as before the ice-sheet rule.
    fields, soil_type = _land_case([[60_000.0, 0.0]], [[0.3] * 2], [[6, 6]])
    with pytest.raises(ValueError, match="above the plausibility ceiling of 20000"):
        preprocess_noah_soil(fields, soil_type=soil_type)

    # Above the ice-sheet ceiling, refused even on land ice.
    fields, soil_type = _land_case([[250_000.0, 0.0]], [[0.3] * 2], [[16, 6]])
    with pytest.raises(ValueError, match="plausibility ceiling"):
        preprocess_noah_soil(fields, soil_type=soil_type)


# ---------------------------------------------------------------------------
# the soil moisture ceiling
# ---------------------------------------------------------------------------
def test_soil_moisture_cap_below_at_and_above(capsys):
    from woof.ingest.soil import cap_land_moisture_at_smcmax

    table = _stas_maxsmc()
    # Loam (6, 0.439), sand (1, 0.339), land ice (16), water category (14),
    # a category outside the table (0) and a water cell.
    soil_type = np.array([[6, 6, 6, 1, 16, 14, 0, 6]])
    terrestrial = np.array([[True, True, True, True, True, True, True, False]])
    layer = np.array([[0.43, 0.439, 0.86, 0.40, 1.0, 1.0, 0.9, 1.0]])
    soil_m = np.stack([layer, layer, np.full_like(layer, 0.2), layer])
    capped, receipt = cap_land_moisture_at_smcmax(
        soil_m, terrestrial, soil_type, table, table="SOILPARM.TBL STAS MAXSMC")
    expected = soil_m.copy()
    expected[[0, 1, 3], 0, 2] = 0.439
    expected[[0, 1, 3], 0, 3] = 0.339
    np.testing.assert_array_equal(capped, expected)
    assert receipt["cells_capped"] == 2
    assert receipt["layer_values_capped"] == 6
    assert receipt["source_max"] == 0.86
    assert receipt["by_category"] == {
        "1": {"smcmax": 0.339, "cells": 1, "max_pre_cap": 0.40},
        "6": {"smcmax": 0.439, "cells": 1, "max_pre_cap": 0.86},
    }
    assert receipt["fields"]["SMOIS_L1"] == {"capped_cells": 2, "max_pre_cap": 0.86}
    assert "SMOIS_L3" not in receipt["fields"]
    assert receipt["bounding_box"] == {"rows": [0, 0], "columns": [2, 3]}
    line = capsys.readouterr().err.strip()
    assert line == (
        "soil moisture cap: capped at the cell's SOILPARM.TBL STAS MAXSMC on 2 land "
        "cell(s), 6 layer value(s) (source value up to 0.86 m3 m-3; category 1 at "
        "0.339 on 1, category 6 at 0.439 on 1); WRF NOAHMP_INIT's ceiling")

    # Nothing above: the same object back, no receipt, no line.
    calm = np.full((4, 1, 2), 0.3)
    same, receipt = cap_land_moisture_at_smcmax(
        calm, np.ones((1, 2), bool), np.array([[6, 1]]), table, table="t")
    assert same is calm and receipt == {}
    assert capsys.readouterr().err == ""


def test_wrf_init_caps_icon_peat_and_leaves_mineral_soil_alone(capsys):
    from woof.ingest.soil import preprocess_noah_soil

    fields, soil_type = _land_case([[0.0, 0.0, 0.0]], [[0.86, 0.43, 0.86]],
                                   [[13, 6, 16]])
    soil = preprocess_noah_soil(fields, soil_type=soil_type)
    # Organic material (13) holds 0.439; loam at 0.43 is below its 0.439;
    # land ice (16) is the land model's glacier, not a soil column.
    np.testing.assert_allclose(soil.soil_moisture[:, 0, 0], 0.439)
    np.testing.assert_allclose(soil.soil_moisture[:, 0, 1], 0.43)
    np.testing.assert_allclose(soil.soil_moisture[:, 0, 2], 0.86)
    assert np.all(soil.liquid_moisture[:, 0, 0] <= 0.439)
    assert set(soil.init_caps) == {"soil_moisture"}
    cap = soil.init_caps["soil_moisture"]
    assert cap["cells_capped"] == 1
    assert cap["layer_values_capped"] == 4
    assert cap["source_max"] == pytest.approx(0.86)
    assert "soil moisture cap: capped at the cell's SOILPARM.TBL STAS MAXSMC on 1 land" \
        in capsys.readouterr().err


def test_wrf_init_below_both_ceilings_is_unchanged():
    """A preparation neither ceiling reaches carries no receipt at all, so
    its proof is byte for byte what it was."""
    from woof.ingest.soil import preprocess_noah_soil

    fields, soil_type = _land_case([[10.0, 200.0]], [[0.30, 0.43]], [[6, 6]])
    soil = preprocess_noah_soil(fields, soil_type=soil_type)
    np.testing.assert_array_equal(soil.snow_water, [[10.0, 200.0]])
    np.testing.assert_allclose(soil.soil_moisture[:, 0, 1], 0.43)
    assert dict(soil.init_caps) == {}


def test_noah_cap_can_be_left_to_a_caller_that_caps_its_own_levels():
    from woof.ingest.soil import preprocess_noah_soil

    fields, soil_type = _land_case([[0.0]], [[0.86]], [[13]])
    soil = preprocess_noah_soil(fields, soil_type=soil_type, soil_moisture_cap=False)
    np.testing.assert_allclose(soil.soil_moisture[:, 0, 0], 0.86)
    assert dict(soil.init_caps) == {}


def test_hex_and_wrf_caps_agree_on_the_rust_unit_vectors():
    """Parity: the vectors caps.rs's own unit tests use, through the WRF
    path's functions, give the numbers the Rust asserts."""
    from woof.ingest.soil import cap_land_moisture_at_smcmax, cap_snow_water

    snow = np.array([[0.0, 4999.0, 5000.0, 5001.0, 51_800.0]], dtype=np.float32)
    snowh = snow * np.float32(5.0) / np.float32(1000.0)
    capped, depth, receipt = cap_snow_water(snow, snowh)
    assert capped.tolist() == [[0.0, 4999.0, 5000.0, 5000.0, 5000.0]]
    assert np.all(np.abs(depth[0, 3:] - 25.0) < 1e-4)
    assert receipt["cells_capped"] == 2

    # snow_below_at_and_above / soil_moisture_below_at_and_above in caps.rs,
    # minus its last column: land ice by land use alone, which the WRF path
    # cannot see (its init stage carries the soil category, not land use).
    landmask = np.array([[1, 1, 1, 1, 0, 1, 1]])
    isltyp = np.array([[13, 13, 13, 16, 14, 0, 16]])
    rows = np.array([
        [0.30, 0.43, 0.438, 0.20],
        [0.439, 0.439, 0.439, 0.439],
        [0.86, 0.70, 0.44, 0.30],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.9, 0.9, 0.9, 0.9],
        [1.0, 1.0, 1.0, 1.0],
    ], dtype=np.float32)
    soil_m = rows.T[:, None, :]
    capped, receipt = cap_land_moisture_at_smcmax(
        soil_m, landmask == 1, isltyp, _stas_maxsmc(), table="STAS")
    assert capped[:, 0, 2].tolist() == pytest.approx([0.439, 0.439, 0.439, 0.30])
    for column in (0, 1, 3, 4, 5, 6):
        assert capped[:, 0, column].tolist() == soil_m[:, 0, column].tolist()
    assert (receipt["cells_capped"], receipt["layer_values_capped"]) == (1, 3)
    assert receipt["source_max"] == pytest.approx(0.86)


# ---------------------------------------------------------------------------
# the hex doors
# ---------------------------------------------------------------------------
def test_init_door_lifts_the_engine_caps_into_notes():
    from woof.hex.init_caps import engine_cap_notes

    receipt = {"caps": {
        "snow": {"cap_kg_m2": 5000.0, "cells_capped": 3,
                 "note": "rw_mpas_init: capped SNOW at 5000 kg m-2 on 3 cell(s)"},
        "soil_moisture": {"cells_capped": 0, "note": ""},
    }}
    caps, notes = engine_cap_notes(receipt)
    assert caps == receipt["caps"]
    assert notes == ["rw_mpas_init: capped SNOW at 5000 kg m-2 on 3 cell(s)"]
    assert engine_cap_notes({"n_cells": 4}) == (None, [])


def _write_intermediate(path: Path, records):
    from woof.hex.hrrr_intermediate import LatLonTarget, write_intermediate

    target = LatLonTarget(south=60.0, west=-45.0, dlat=1.0, dlon=1.0, nx=3, ny=2)
    write_intermediate(path, records, valid_time=datetime(2026, 10, 11),
                       forecast_hour=0.0, map_source="test", target=target)


def test_intermediate_preview_counts_what_the_init_will_cap(tmp_path):
    from woof.hex.init_caps import preview_intermediate

    path = tmp_path / "MET:2026-10-11_00"
    landsea = np.array([[1, 1, 1], [1, 0, 1]], dtype=np.float32)
    _write_intermediate(path, [
        {"field": "SNOW", "units": "kg m-2", "description": "", "level": 200100.0,
         "values": np.array([[51_800.0, 5000.0, 0.0], [12.0, 6000.0, 4999.0]], np.float32)},
        {"field": "LANDSEA", "units": "", "description": "", "level": 200100.0,
         "values": landsea},
        {"field": "SM000010", "units": "", "description": "", "level": 200100.0,
         "values": np.array([[0.86, 0.43, 1.0], [0.5, 0.9, 0.2]], np.float32)},
    ])
    preview = preview_intermediate(path, ["SNOW", "LANDSEA", "SM000010"])
    assert preview["snow"]["points_above_cap"] == 2
    assert preview["snow"]["source_max_kg_m2"] == 51_800.0
    soil = preview["soil_moisture"]
    # 0.86 and 0.5 on land; 0.9 sits on a water point; 1.0 is counted apart.
    assert soil["land_points_above_largest_smcmax"] == 2
    assert soil["land_points_at_one"] == 1
    assert soil["source_max_below_one"] == pytest.approx(0.86)
    assert len(preview["notes"]) == 2
    assert preview["notes"][0].startswith("SNOW is above 5000 kg m-2 at 2 source point(s)")

    calm = tmp_path / "MET:2026-10-11_03"
    _write_intermediate(calm, [
        {"field": "SNOW", "units": "kg m-2", "description": "", "level": 200100.0,
         "values": np.full((2, 3), 10.0, np.float32)},
    ])
    preview = preview_intermediate(calm, ["SNOW"])
    assert preview["snow"]["points_above_cap"] == 0 and preview["notes"] == []
