"""ECMWF open data's snow reaches the initial state; its sea ice cannot.

Measured with ``tools/grib1_bridge`` ``grib2_inventory --decode`` on the
2026-10-05 06Z and 2026-10-09 06Z ``oper`` files (f000), every one on the
0.25-degree global grid:

* ``0/1/254`` (ECMWF local table 1, centre 98, master table 32, surface,
  PDT 0, no bitmap): IFS ``sd``, snow depth in metres of water equivalent,
  0 .. 10.0002 -- capped at 10 m over the ice sheets.  Present, and until
  this profile bound it a Greenland preparation started with SNOW = 0 on
  every cell of the ice sheet.
* ``0/1/61`` (WMO): ``rsn``, snow density, 50 .. 500 kg m-3.
* ``10/2/1`` (WMO, ocean bitmap, 684,078 of 1,038,240 points): sea-ice
  THICKNESS, 0 .. 7.1 m.  There is no ``10/2/0`` (sea-ice area fraction,
  IFS ``ci``/``siconc``) in either file, so the profile has no sea-ice
  fraction to bind; it says so instead of claiming one exists.

SNOW is bound at water density (m -> kg m-2, x1000), as the ERA5 donor
binds IFS ``sd``; SNOWH is left to real.exe's own 5:1 reconciliation
(``dyn_em/module_initialize_real.F`` v4.6.1 :538-545), which
``woof.ingest.soil.preprocess_noah_soil`` reproduces.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from woof.mapped_source import (
    _GribRecord,
    _assemble_grib,
    _grib2_wanted_indices,
    load_mapping,
)
from woof.source_authorities import packaged_authorities

IFS_PROFILE = "ecmwf-open-data-oper-grib2-v1"
CYCLE = datetime(2026, 10, 5, 6)
NY, NX = 2, 3

#: (discipline, category, parameter, local_table_version, level_type) of
#: the surface records the 2026-10-05 06Z f000 file carries around snow and
#: ice, as its inventory spells them.
SD = (0, 1, 254, 1, 1)
RSN = (0, 1, 61, 0, 1)
SITHICK = (10, 2, 1, 0, 1)


def _mapping():
    return load_mapping(packaged_authorities(IFS_PROFILE)["mapping"])


def _record(index, identity, values, *, master_table_version=32):
    discipline, category, parameter, local_table_version, level_type = identity
    return _GribRecord(
        source=Path("/staged/20261005060000-0h-oper-fc.grib2"), index=index,
        reference_time=CYCLE, valid_time=CYCLE, member=None,
        parameter=parameter, level_type=level_type, level_value=0.0,
        table_version=None, center=98, subcenter=0,
        master_table_version=master_table_version,
        local_table_version=local_table_version,
        discipline=discipline, category=category,
        second_level_type=255, second_level_value=0.0,
        process_identity=None, time_semantics=(0,),
        values=np.asarray(values, dtype=np.float64),
        latitude=np.linspace(72.0, 71.75, NY),
        longitude=np.linspace(-42.0, -41.5, NX),
        grid_fingerprint="grid",
    )


def _inventory_row(index, identity, *, master_table_version=32):
    discipline, category, parameter, local_table_version, level_type = identity
    return {
        "index": str(index), "member": "-", "parameter": str(parameter),
        "level_type": str(level_type), "level_value": "0", "center": "98",
        "subcenter": "0", "master_table_version": str(master_table_version),
        "local_table_version": str(local_table_version),
        "discipline": str(discipline), "category": str(category),
        "second_level_type": "255", "second_level_value": "0", "pdt": "0",
    }


def _snow_only(mapping):
    reduced = dict(mapping)
    reduced["fields"] = {"snow_water_equivalent":
                         mapping["fields"]["snow_water_equivalent"]}
    return reduced


def test_the_profile_binds_ifs_sd_as_snow_water_at_water_density():
    mapping = _mapping()
    field = mapping["fields"]["snow_water_equivalent"]
    assert field["units"] == {"source": "m", "target": "kg m-2",
                              "scale": 1000.0}
    assert field["location"] == "surface"
    (selector,) = field["selectors"]
    assert (selector["discipline"], selector["category"],
            selector["parameter"]) == (0, 1, 254)
    assert (selector["center"], selector["local_table_version"]) == (98, 1)
    target = mapping["target"]
    # Not required: an earlier publication without sd still prepares, as
    # every frame without a non-required field does.
    assert "snow_water_equivalent" not in {
        row["name"] for row in target["required_fields"]}
    policies = target["initialization_policies"]
    assert policies["snow_water_equivalent"].startswith("bound_as_")
    assert "snow_depth" not in mapping["fields"]
    assert "sea_ice_fraction" not in mapping["fields"]
    assert policies["sea_ice_fraction"] == (
        "source_publishes_sea_ice_thickness_only_no_area_fraction")


def test_the_inventory_reads_sd_and_not_density_or_ice_thickness():
    rows = [_inventory_row(69, RSN), _inventory_row(87, SITHICK),
            _inventory_row(132, SD)]
    wanted = _grib2_wanted_indices(_snow_only(_mapping()), rows)
    assert wanted == {132}


def test_sd_under_another_master_table_is_not_read_as_snow():
    """The local parameter is only ECMWF's sd under the table it was read in."""
    rows = [_inventory_row(132, SD, master_table_version=27)]
    assert _grib2_wanted_indices(_snow_only(_mapping()), rows) == set()


def test_sd_assembles_as_snow_water_equivalent_in_kg_m2():
    mapping = _mapping()
    metres = np.array([[0.0, 0.25, 10.000244], [0.0, 0.0, 1.5]])
    records = [_record(0, SD, metres), _record(1, RSN, np.full((NY, NX), 300.0)),
               _record(2, SITHICK, np.full((NY, NX), 1.2))]
    collection = _assemble_grib(_snow_only(mapping), records)
    (snow,) = [value for value in collection.direct.values()
               if value.name == "snow_water_equivalent"]
    # Metres of water equivalent at 1000 kg m-3: the declared unit scale.
    np.testing.assert_allclose(snow.values, metres * 1000.0)


def test_mapped_snow_reaches_the_soil_seam_as_snow_and_a_5_to_1_depth():
    from woof.ingest.ruc_soil import preprocess_land_surface_soil
    from woof.ingest.soil_contract import (MAPPED_SOIL_MOISTURE,
                                            MAPPED_SOIL_TEMPERATURE)

    composition = json.loads(packaged_authorities(IFS_PROFILE)[
        "composition"].read_text(encoding="utf-8"))
    shape = (1, 3)
    snow = np.array([[0.0, 250.0, 10000.244]])
    fields = {
        "LANDSEA": np.ones(shape), "SKINTEMP": np.full(shape, 258.0),
        "SEAICE": np.zeros(shape), "SNOW": snow,
        MAPPED_SOIL_TEMPERATURE: np.full((4, *shape), 262.0),
        MAPPED_SOIL_MOISTURE: np.stack(
            [np.full(shape, value) for value in (0.2, 0.25, 0.3, 0.3)]),
    }
    state = preprocess_land_surface_soil(
        fields, sf_surface_physics=3, num_soil_layers=9,
        soil_type=np.full(shape, 16), deep_soil_temperature=np.full(shape, 253.0),
        soil_layer_contract=composition["soil_layers"],
        landmask=np.ones(shape), water_temperature_policy="wrf_compat")
    # The bound snow reaches the seam; the init ceiling (woof.ingest.soil.
    # SNOW_WATER_CAP_KG_M2) then caps an ice-sheet store, depth scaled by
    # the same factor so the 5:1 density is kept.
    from woof.ingest.soil import SNOW_WATER_CAP_KG_M2

    capped = np.minimum(snow, SNOW_WATER_CAP_KG_M2)
    np.testing.assert_allclose(np.asarray(state.snow_water), capped)
    np.testing.assert_allclose(np.asarray(state.snow_depth),
                               capped / 1000.0 * 5.0, rtol=1e-6)
    assert state.init_caps, "the ice-sheet value above the ceiling is recorded"


@pytest.mark.parametrize("profile", [IFS_PROFILE])
def test_the_pinned_mapping_digest_is_the_file(profile):
    import hashlib

    from woof import source_authorities

    paths = packaged_authorities(profile)
    digest = hashlib.sha256(paths["mapping"].read_bytes()).hexdigest()
    source = Path(source_authorities.__file__).read_text(encoding="utf-8")
    assert digest in source
