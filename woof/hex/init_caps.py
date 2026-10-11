"""The two ceilings ``rw_mpas_init`` applies at init, as the hex doors report them.

``rw_mpas_init`` (``tools/rustwx/crates/rw-mpas/src/init/caps.rs``) caps two
fields native MPAS ``init_atmosphere`` passes through unchanged:

* **snow water equivalent** at 5000 kg m-2 at every cell, the snow depth
  scaled by the same factor so the density is kept -- WRF's Noah-MP
  initialisation (``module_sf_noahmpdrv.F``, ``NOAHMP_INIT``) applies the
  same ceiling.  Met Office Global publishes about 51,800 kg m-2 over the
  Greenland ice sheet, which case 7's 5:1 depth rule would start under
  259 m of snow;
* **volumetric soil moisture** at the cell's own porosity, ``MAXSMC`` for
  its soil category in ``SOILPARM.TBL``'s ``STAS`` block (the table the
  Noah land model reads), on land cells that are not land ice
  (``ivgtyp == isice`` or soil category 16) --
  ``NOAHMP_INIT``'s ``IF ( SMOIS > SMCMAX ) SMOIS = SMCMAX``.  ICON's peat
  reaches 0.86 m3 m-3; no STAS land category holds more than 0.484.

The engine's receipt carries both (cap value, cells capped, largest value
met) under ``caps``; :func:`engine_cap_notes` lifts them into the init
door's own receipt and log.  The intermediate doors cannot know the target
cell's soil category, so :func:`preview_intermediate` counts the source
points the init will meet above each ceiling, from the written file itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

#: kg m-2.  ``SNOW_WATER_CAP_KG_M2`` in caps.rs.
SNOW_WATER_CAP_KG_M2 = 5000.0

#: ``MAXSMC`` of ``SOILPARM.TBL``'s ``STAS`` categories 1..19, as caps.rs
#: carries it.  Category 14 is water (1.0) and 16 is land ice.
STAS_MAXSMC = (0.339, 0.421, 0.434, 0.476, 0.484, 0.439, 0.404, 0.464,
               0.465, 0.406, 0.468, 0.468, 0.439, 1.0, 0.20, 0.421, 0.468,
               0.200, 0.339)

#: The largest porosity of any STAS land category (water excluded): a
#: source soil moisture above it is capped at init whatever the target
#: cell's soil turns out to be, unless that cell is land ice.
LARGEST_LAND_SMCMAX = max(value for category, value in enumerate(STAS_MAXSMC, 1)
                          if category != 14)


def engine_cap_notes(engine_receipt: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """The engine's ``caps`` block and the one-line notes for what it capped.

    ``(None, [])`` for an engine that reports no ``caps`` (a build from
    before the ceilings existed), so a door never invents a count.
    """

    caps = engine_receipt.get("caps") if isinstance(engine_receipt, Mapping) else None
    if not isinstance(caps, Mapping):
        return None, []
    notes = [str(block.get("note")) for block in (caps.get("snow"), caps.get("soil_moisture"))
             if isinstance(block, Mapping) and block.get("note")]
    return dict(caps), notes


def _soil_moisture_names(names: Iterable[str]) -> list[str]:
    return sorted({name for name in names if name.startswith(("SM", "SOILM"))})


def preview_intermediate(path: str | Path, field_names: Iterable[str], *,
                         label: str | None = None) -> dict[str, Any]:
    """What the init's two ceilings will meet in one written intermediate.

    Counts source points, not mesh cells: SNOW values above
    :data:`SNOW_WATER_CAP_KG_M2`, and land soil moisture values (LANDSEA
    >= 0.5 where the file carries LANDSEA on the soil grid) above
    :data:`LARGEST_LAND_SMCMAX`.  Soil moisture of exactly 1.0 is counted
    apart: it is the land-ice and water convention (GFS writes 1.0 on its
    glaciers) and the init leaves land-ice cells to the land model.
    ``label`` names the file in the notes (a door passes the valid time,
    because the path it validates is still the ``.partial`` one).
    """

    from .wps_intermediate import WpsIntermediateReader

    names = set(field_names)
    where = f"the {label} intermediate" if label else Path(path).name
    soil = _soil_moisture_names(names)
    wanted = ({"SNOW"} & names) | set(soil) | ({"LANDSEA"} & names)
    snow_values: list[np.ndarray] = []
    soil_values: list[np.ndarray] = []
    landsea = None
    if wanted:
        with WpsIntermediateReader(path) as reader:
            for record in reader.iter_fields(include=wanted):
                # The slab as read (float32): counted in place, not copied.
                values = np.asarray(record.values)
                if record.field == "SNOW":
                    snow_values.append(values)
                elif record.field == "LANDSEA":
                    landsea = values
                else:
                    soil_values.append(values)
    preview: dict[str, Any] = {
        "applied_by": "rw_mpas_init (woof hex init); counts here are source points in this "
                      "file, the init receipt counts mesh cells",
        "snow": {"cap_kg_m2": SNOW_WATER_CAP_KG_M2, "carried": bool(snow_values)},
        "soil_moisture": {"largest_land_smcmax": LARGEST_LAND_SMCMAX,
                          "table": "SOILPARM.TBL STAS MAXSMC", "layers": len(soil_values)},
        "notes": [],
    }
    if snow_values:
        above = sum(int(np.count_nonzero(v > SNOW_WATER_CAP_KG_M2)) for v in snow_values)
        tops = [float(np.nanmax(v)) for v in snow_values if np.isfinite(v).any()]
        largest = max(tops) if tops else None
        preview["snow"].update(points_above_cap=above, source_max_kg_m2=largest)
        if above:
            preview["notes"].append(
                f"SNOW is above {SNOW_WATER_CAP_KG_M2:g} kg m-2 at {above} source point(s) of "
                f"{where} (up to {largest:g} kg m-2); woof hex init caps it there "
                f"at {SNOW_WATER_CAP_KG_M2:g} kg m-2 and scales the snow depth to keep the "
                f"density (WRF NOAHMP_INIT's ceiling)")
    if soil_values:
        above = saturated = 0
        largest = None
        for values in soil_values:
            land = (landsea >= 0.5) if (landsea is not None and landsea.shape == values.shape) \
                else np.ones(values.shape, dtype=bool)
            sample = values[land & np.isfinite(values)]
            over = sample > LARGEST_LAND_SMCMAX
            ones = sample == 1.0
            saturated += int(np.count_nonzero(ones))
            above += int(np.count_nonzero(over & ~ones))
            beyond = sample[over & ~ones]
            if beyond.size:
                top = float(beyond.max())
                largest = top if largest is None else max(largest, top)
        preview["soil_moisture"].update(
            land_points_above_largest_smcmax=above, source_max_below_one=largest,
            land_points_at_one=saturated)
        if above:
            preview["notes"].append(
                f"soil moisture is above {LARGEST_LAND_SMCMAX:g} m3 m-3, the largest STAS "
                f"land porosity, at {above} land source value(s) of {where} (up to "
                f"{largest:g}); woof hex init caps each land cell that is not land ice at "
                f"its own soil category's SMCMAX (WRF NOAHMP_INIT's ceiling)")
    return preview


__all__ = [
    "LARGEST_LAND_SMCMAX",
    "SNOW_WATER_CAP_KG_M2",
    "STAS_MAXSMC",
    "engine_cap_notes",
    "preview_intermediate",
]
