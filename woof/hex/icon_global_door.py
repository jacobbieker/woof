"""DWD ICON global onto a GLOBAL MPAS mesh: the full-sphere half of ``woof hex intermediate --source icon-global|icon-global-ml``.

ICON publishes on its native unstructured triangle mesh (WMO grid template
3.101, R03B07, 2,949,120 cells, grid UUID a27b8de6-18c4-11e4-820a-b5b098c6a5c0),
which no WPS-intermediate reader takes.  For a mesh that covers the whole
sphere the door therefore regrids each native field ONCE, onto a full-sphere
regular lat-lon grid, and hands that to the same ``met_intermediate`` writer
the GFS and ECMWF rows use:

1. ``gdt101_remap plan --global --dx-deg 0.125`` reads the cycle's CLAT,
   CLON and FR_LAND records and writes one weights plan (four nearest cell
   centres, inverse squared chord distance; land/water classes from
   FR_LAND).  The plan depends on the mesh and the target only, so it is
   built once per invocation and reused for every field and every lead.
2. ``gdt101_remap apply --global`` remaps every field of a lead with the
   method the regional ICON route uses for it (``idw4`` aloft,
   ``surface-idw4`` at the surface, ``soil-nearest`` below ground,
   ``seaice-nearest`` for sea ice) onto 2880 x 1441 points: 0 E eastward
   with no repeated column, the South Pole to the North Pole, the periodic
   convention ``rw_mpas_init`` wraps itself.  Each output keeps its
   original product section.
3. ``met_intermediate --map-source dwd`` with the packaged
   ``Vtable.ICON-GLOBAL.rw`` (``Vtable.ICON-GLOBAL-ML.rw`` for model levels)
   writes one ``PREFIX:YYYY-MM-DD_HH`` file per valid time, which is read
   back and checked by the global door's own validation before anything is
   published.

Inputs are the ICON v1 open-data tree as DWD publishes it, mirrored under
``--grib-dir``::

    p/<PARAM>[/lvt1/<100|150|106>/lv1/<Pa|ordinal|metres>]/r/<YYYY-MM-DDTHH:MM>/s/PT<HHH>H00M.grib2

(historical runs come from the same ``p/`` tree on the source.coop mirror).
The regional half of the ICON door -- a limited-area mesh, or a
``--from-plan``/``--point`` box -- is not this module's; see
:func:`woof.hex.met_intermediate_door.icon_target_kind`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

from .met_intermediate_door import (
    ENGINE_RECEIPT_SCHEMA,
    GlobalIntermediateRefusal,
    GlobalSource,
    _described,
    _partial,
    init_switches,
    packaged_vtable,
    parse_engine_receipt,
    parse_lead_range,
    refuse_occupied_out_dir,
    resolve_engine,
    validate_intermediate,
)

SCHEMA = "gpuwm-hex.icon-global-intermediate/v1"
ATTRIBUTION = "Source: Deutscher Wetterdienst (DWD), CC BY 4.0"
REMAP_NAME = "gdt101_remap"
#: The remap contract is unchanged by the global mode (the regional route's
#: bytes are not touched); the global mode is a marked capability on top of
#: it, and a build without it is refused statically before any work.
REMAP_CONTRACT = "arwen.gdt101-regional-remap.v1"
REMAP_GLOBAL_MODE = "arwen.gdt101-regional-remap.v1+global-latlon"
GLOBAL_DX_DEG = 0.125
#: The full-sphere grid ``--dx-deg 0.125`` writes, as the intermediate
#: header must carry it.
GLOBAL_GRID = {"nx": 2880, "ny": 1441, "start_lat": -90.0, "start_lon": 0.0,
               "dlat": 0.125, "dlon": 0.125}
ICON_CELLS = 2_949_120
ICON_CENTRE = 78
ICON_GRID_UUID = "a27b8de618c411e4820ab5b098c6a5c0"
#: A plan is ~200 MB and each ``apply`` process holds one plus a field
#: (~350 MB measured); this many run side by side by default.
DEFAULT_REMAP_WORKERS = 4
#: Seconds one ``apply`` batch may take before the door gives up on it.
REMAP_TIMEOUT_SECONDS = 3600


# ---------------------------------------------------------------------------
# the field table: selector octets measured on real v1 bytes (2026-09-27T12)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IconField:
    """One DWD parameter, how its records are addressed and how it is remapped."""

    name: str
    discipline: int
    category: int
    parameter: int
    first_type: int
    method: str
    #: None: one record at ``first_value``; "isobaric": lv1 is Pa;
    #: "model": lv1 is a full-level ordinal, layer bounded by ordinal+1;
    #: "half": lv1 is a half-level ordinal (HHL: second surface 101);
    #: "soil": lv1 is a depth in metres; "soil-layer": lv1 is the top of a
    #: layer whose bottom is :data:`SOIL_LAYER_BOTTOMS`.
    levels: str | None = None
    first_value: float = 0.0
    second_type: int = 255
    second_value: float = 0.0

    def selector(self, level: float | None) -> tuple[Any, ...]:
        first, second = self.first_value, self.second_value
        if self.levels is not None:
            if level is None:
                raise GlobalIntermediateRefusal(f"{self.name} needs a level")
            first = float(level)
        if self.levels == "model":
            second = float(level) + 1.0  # type: ignore[arg-type]
        elif self.levels == "soil-layer":
            if first not in SOIL_LAYER_BOTTOMS:
                raise GlobalIntermediateRefusal(
                    f"{self.name} layer top {first:g} m is not one of ICON's soil "
                    f"layers {sorted(SOIL_LAYER_BOTTOMS)}"
                )
            second = SOIL_LAYER_BOTTOMS[first]
        return (self.discipline, self.category, self.parameter, self.first_type,
                first, self.second_type, second)

    def level_type_path(self) -> str | None:
        return {"isobaric": "100", "model": "150", "half": "150",
                "soil": "106", "soil-layer": "106"}.get(self.levels or "")


#: ICON's multi-layer soil: layer top -> bottom, metres.
SOIL_LAYER_BOTTOMS: Mapping[float, float] = {
    0.0: 0.01, 0.01: 0.03, 0.03: 0.09, 0.09: 0.27,
    0.27: 0.81, 0.81: 2.43, 2.43: 7.29, 7.29: 21.87,
}

GEOMETRY = {
    "CLAT": IconField("CLAT", 0, 191, 1, 1, "nearest"),
    "CLON": IconField("CLON", 0, 191, 2, 1, "nearest"),
}
#: Remapped once from the cycle's PT000 and carried into every lead.
INVARIANTS = (
    IconField("FR_LAND", 2, 0, 0, 1, "nearest"),
    IconField("HSURF", 0, 3, 6, 1, "surface-idw4", second_type=101),
)
SURFACE = (
    IconField("PS", 0, 3, 0, 1, "surface-idw4"),
    IconField("PMSL", 0, 3, 1, 101, "idw4"),
    IconField("T_2M", 0, 0, 0, 103, "surface-idw4", first_value=2.0),
    IconField("RELHUM_2M", 0, 1, 1, 103, "surface-idw4", first_value=2.0),
    IconField("U_10M", 0, 2, 2, 103, "surface-idw4", first_value=10.0),
    IconField("V_10M", 0, 2, 3, 103, "surface-idw4", first_value=10.0),
    IconField("T_G", 0, 0, 0, 1, "surface-idw4"),
    IconField("H_SNOW", 0, 1, 11, 1, "surface-idw4"),
    IconField("W_SNOW", 0, 1, 60, 1, "surface-idw4"),
    IconField("FR_ICE", 10, 2, 0, 1, "seaice-nearest"),
    IconField("T_SO", 2, 3, 18, 106, "soil-nearest", levels="soil"),
    IconField("W_SO", 2, 3, 20, 106, "soil-nearest", levels="soil-layer", second_type=106),
)
PRESSURE_LEVEL = tuple(
    IconField(name, 0, category, parameter, 100, "idw4", levels="isobaric")
    for name, category, parameter in (("T", 0, 0), ("U", 2, 2), ("V", 2, 3),
                                      ("RELHUM", 1, 1), ("FI", 3, 4))
)
MODEL_LEVEL = tuple(
    IconField(name, 0, category, parameter, 150, "idw4", levels="model", second_type=150)
    for name, category, parameter in (("T", 0, 0), ("U", 2, 2), ("V", 2, 3),
                                      ("QV", 1, 0), ("P", 3, 0))
) + (IconField("HHL", 0, 3, 6, 150, "idw4", levels="half", second_type=101),)


@dataclass(frozen=True)
class IconSourceRow:
    """``--source icon-global`` or ``icon-global-ml``: which column it carries."""

    name: str
    description: str
    vtable: str
    profile: tuple[IconField, ...]
    use_spechumd: str
    map_source: str = "dwd"
    default_interval_hours: int = 3
    max_hour: int = 180
    #: The Vtable names of the hour-zero fields a later lead borrows
    #: (FR_LAND and HSURF).  MERGE SEAM: these must be the names
    #: Vtable.ICON-GLOBAL*.rw gives them.
    invariant_fields: tuple[str, ...] = ("LANDSEA", "SOILHGT")

    def as_global_source(self) -> GlobalSource:
        return GlobalSource(
            name=self.name, description=self.description, map_source=self.map_source,
            vtable=self.vtable, file_patterns=(), default_interval_hours=self.default_interval_hours,
            max_hour=self.max_hour, invariant_fields=self.invariant_fields,
            use_spechumd=self.use_spechumd,
        )


ICON_GLOBAL_ROWS: Mapping[str, IconSourceRow] = {
    "icon-global": IconSourceRow(
        name="icon-global",
        description="DWD ICON global R03B07 (~13 km, native GDT 3.101) on its "
                    "pressure levels, regridded to a 0.125-degree full-sphere grid",
        vtable="Vtable.ICON-GLOBAL.rw",
        profile=PRESSURE_LEVEL,
        use_spechumd="no",
    ),
    "icon-global-ml": IconSourceRow(
        name="icon-global-ml",
        description="DWD ICON global R03B07 (~13 km, native GDT 3.101) on its "
                    "120 model levels, regridded to a 0.125-degree full-sphere grid",
        vtable="Vtable.ICON-GLOBAL-ML.rw",
        profile=MODEL_LEVEL,
        use_spechumd="yes",
    ),
}


# ---------------------------------------------------------------------------
# the mesh: is it the whole sphere?
# ---------------------------------------------------------------------------
def classify_mesh(path: Path) -> dict[str, Any]:
    """``kind`` "global" or "regional" for an MPAS grid/static/init file.

    Regional is a nonempty ``bdyMaskCell/Edge/Vertex`` zone (the rule
    :mod:`woof.hex.mesh` measured: a global static carries the triple all
    zero).  Global is no zone AND cells that tile the sphere: ``areaCell``
    sums to 4 pi r^2.  Anything else is refused rather than guessed at.
    """

    from netCDF4 import Dataset
    import numpy as np

    from .mesh import REGIONAL_BOUNDARY_MASK_NAMES, _regional_boundary_zone_present

    mesh = Path(path).expanduser()
    if not mesh.is_file():
        raise GlobalIntermediateRefusal(f"--grid {mesh} is not a file")
    try:
        with Dataset(mesh, mode="r") as dataset:
            dataset.set_auto_mask(False)
            if "nCells" not in dataset.dimensions:
                raise GlobalIntermediateRefusal(f"--grid {mesh} has no nCells dimension; it is not an MPAS mesh")
            n_cells = len(dataset.dimensions["nCells"])
            masks = {name: np.asarray(dataset.variables[name][...])
                     for name in REGIONAL_BOUNDARY_MASK_NAMES if name in dataset.variables}
            on_sphere = str(getattr(dataset, "on_a_sphere", "")).strip().upper() == "YES"
            radius = float(getattr(dataset, "sphere_radius", float("nan")))
            area = (float(np.sum(np.asarray(dataset.variables["areaCell"][...], dtype=np.float64)))
                    if "areaCell" in dataset.variables else float("nan"))
    except OSError as error:
        raise GlobalIntermediateRefusal(f"--grid {mesh} is not readable NetCDF: {error}") from error
    facts: dict[str, Any] = {"path": str(mesh.absolute()), "n_cells": n_cells,
                             "on_a_sphere": on_sphere, "sphere_radius": radius}
    if masks and len(masks) != len(REGIONAL_BOUNDARY_MASK_NAMES):
        raise GlobalIntermediateRefusal(
            f"--grid {mesh} carries {sorted(masks)} but not the whole bdyMask triple; a "
            f"partial triple cannot say whether the mesh is regional"
        )
    if _regional_boundary_zone_present(masks):
        facts["kind"] = "regional"
        facts["reason"] = "the bdyMask triple carries a boundary zone"
        return facts
    fraction = area / (4.0 * math.pi * radius * radius) if radius > 0 else float("nan")
    facts["area_fraction_of_sphere"] = fraction
    # A limited-area cull taken before init_atmosphere carries no bdyMask
    # triple at all; its cells cover part of the sphere, and that is the
    # whole of what makes it regional.
    if on_sphere and not masks and math.isfinite(fraction) and 0.0 < fraction < 1.0 - 1e-3:
        facts["kind"] = "regional"
        facts["reason"] = f"no bdyMask triple and the cells cover {fraction:.1%} of the sphere"
        return facts
    if not on_sphere or not math.isfinite(fraction) or abs(fraction - 1.0) > 1e-3:
        raise GlobalIntermediateRefusal(
            f"--grid {mesh} has no regional boundary zone and does not tile the sphere "
            f"(on_a_sphere={'YES' if on_sphere else 'NO'}, cell area / 4 pi r^2 = "
            f"{fraction:.6g}); it is neither a global mesh nor a regional cull this door "
            f"can classify"
        )
    facts["kind"] = "global"
    facts["reason"] = "no boundary zone and the cells tile the sphere"
    return facts


# ---------------------------------------------------------------------------
# the inputs: the v1 tree under --grib-dir
# ---------------------------------------------------------------------------
def tree_root(grib_dir: Path) -> Path:
    """The ``p/`` directory of the mirrored v1 tree, or a refusal."""

    root = Path(grib_dir).expanduser()
    if not root.is_dir():
        raise GlobalIntermediateRefusal(f"--grib-dir {root} is not a directory")
    if (root / "p").is_dir():
        return root / "p"
    if root.name == "p":
        return root
    raise GlobalIntermediateRefusal(
        f"--grib-dir {root} holds no p/ tree; the ICON door reads the v1 layout "
        f"p/<PARAM>[/lvt1/<type>/lv1/<level>]/r/<YYYY-MM-DDTHH:MM>/s/PT<HHH>H00M.grib2"
    )


def object_path(root: Path, field: IconField, level_text: str | None,
                cycle: datetime, lead: int) -> Path:
    base = root / field.name
    if field.levels is not None:
        base = base / "lvt1" / str(field.level_type_path()) / "lv1" / str(level_text)
    return base / "r" / f"{cycle:%Y-%m-%dT%H:%M}" / "s" / f"PT{int(lead):03d}H00M.grib2"


def levels_present(root: Path, field: IconField, cycle: datetime, lead: int) -> list[str]:
    """The lv1 directories of ``field`` holding this cycle and lead, numerically sorted."""

    parent = root / field.name / "lvt1" / str(field.level_type_path()) / "lv1"
    if not parent.is_dir():
        return []
    found = []
    for child in parent.iterdir():
        try:
            value = float(child.name)
        except ValueError:
            continue
        if object_path(root, field, child.name, cycle, lead).is_file():
            found.append((value, child.name))
    return [text for _, text in sorted(found)]


@dataclass(frozen=True)
class RemapObject:
    field: IconField
    level_text: str | None
    path: Path

    @property
    def label(self) -> str:
        return self.field.name if self.level_text is None else f"{self.field.name}@{self.level_text}"

    def selector(self) -> tuple[Any, ...]:
        return self.field.selector(None if self.level_text is None else float(self.level_text))


def locate_lead(root: Path, row: IconSourceRow, cycle: datetime, lead: int) -> list[RemapObject]:
    """Every object of one lead, or one refusal naming every gap."""

    missing: list[str] = []
    objects: list[RemapObject] = []
    level_sets: dict[str, list[str]] = {}
    for field in (*row.profile, *SURFACE):
        if field.levels is None:
            path = object_path(root, field, None, cycle, lead)
            if path.is_file():
                objects.append(RemapObject(field, None, path))
            else:
                missing.append(str(path.relative_to(root.parent)))
            continue
        levels = levels_present(root, field, cycle, lead)
        if not levels:
            missing.append(f"{field.name} (no lvt1/{field.level_type_path()} level for this run and lead)")
            continue
        level_sets[field.name] = levels
        objects += [RemapObject(field, text, object_path(root, field, text, cycle, lead)) for text in levels]
    # One column, one ladder: every profile field must carry the same levels.
    column = [f.name for f in row.profile if f.levels in ("isobaric", "model") and f.name in level_sets]
    if column:
        union = sorted({lv for name in column for lv in level_sets[name]}, key=float)
        for name in column:
            gap = [lv for lv in union if lv not in level_sets[name]]
            if gap:
                missing.append(f"{name} at {', '.join(gap)} (the column's other fields carry them)")
        half = [f.name for f in row.profile if f.levels == "half"]
        for name in half:
            if name in level_sets and len(level_sets[name]) != len(union) + 1:
                missing.append(
                    f"{name}: {len(level_sets[name])} half levels for {len(union)} full levels"
                )
    if missing:
        raise GlobalIntermediateRefusal(
            f"{root.parent} is missing ICON {cycle:%Y-%m-%dT%H}Z PT{lead:03d} objects: "
            f"{'; '.join(missing)}"
        )
    return objects


def locate_invariants(root: Path, cycle: datetime) -> dict[str, Path]:
    """The cycle's PT000 geometry and invariant records."""

    found: dict[str, Path] = {}
    missing = []
    for field in (*GEOMETRY.values(), *INVARIANTS):
        path = object_path(root, field, None, cycle, 0)
        if path.is_file():
            found[field.name] = path
        else:
            missing.append(str(path.relative_to(root.parent)))
    if missing:
        raise GlobalIntermediateRefusal(
            f"{root.parent} is missing the ICON {cycle:%Y-%m-%dT%H}Z mesh records "
            f"{'; '.join(missing)}; the regrid plan is built from CLAT, CLON and "
            f"FR_LAND of the same run"
        )
    return found


# ---------------------------------------------------------------------------
# the remap engine
# ---------------------------------------------------------------------------
def resolve_remap(explicit: Path | None = None) -> Path:
    """``gdt101_remap`` with the global mode, or a refusal naming the remedy."""

    from woof import bridges

    if explicit is not None:
        found: Path | None = Path(explicit).expanduser()
        if not found.is_file():  # type: ignore[union-attr]
            raise GlobalIntermediateRefusal(f"the gdt101_remap {found} is not a file")
    else:
        try:
            found = bridges.find_bridge(REMAP_NAME)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            raise GlobalIntermediateRefusal(f"gdt101_remap could not be resolved: {error}") from error
        if found is None:
            raise GlobalIntermediateRefusal(
                "gdt101_remap is not built or staged.\n" + bridges.bridge_remedy(REMAP_NAME)
            )
    # One static read for both markers, on every route to the binary (the
    # contract is checked first only for the plainer refusal).
    path = Path(found).absolute()  # type: ignore[arg-type]
    payload = path.read_bytes()
    if REMAP_CONTRACT.encode() not in payload:
        raise GlobalIntermediateRefusal(
            f"{path} does not speak the {REMAP_CONTRACT} contract; rebuild tools/grib1_bridge"
        )
    if REMAP_GLOBAL_MODE.encode() not in payload:
        raise GlobalIntermediateRefusal(
            f"{path} predates gdt101_remap --global (no {REMAP_GLOBAL_MODE!r} marker); "
            f"rebuild tools/grib1_bridge"
        )
    return path


def _selector_token(selector: Sequence[Any]) -> list[str]:
    return [repr(float(x)) if isinstance(x, float) else str(int(x)) for x in selector]


def job_row(source: Path, destination: Path, selector: Sequence[Any], method: str,
            cycle: datetime, lead: int) -> str:
    for text in (str(source), str(destination)):
        if any(c in text for c in "\t\r\n"):
            raise GlobalIntermediateRefusal(f"a path holds a tab or newline: {text!r}")
    return "\t".join([str(source), str(destination), *_selector_token(selector), method,
                      f"{cycle:%Y%m%d%H}", str(int(lead) * 3600)]) + "\n"


def _run_remap(argv: list[str], log: Path) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=REMAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        def text(value: Any) -> str:
            return value.decode("utf-8", "replace") if isinstance(value, bytes) else (value or "")

        log.write_text(f"$ {' '.join(argv)}\n[timed out after {REMAP_TIMEOUT_SECONDS} s]\n"
                       f"--- stdout ---\n{text(error.stdout)}\n--- stderr ---\n{text(error.stderr)}\n",
                       encoding="utf-8")
        raise GlobalIntermediateRefusal(
            f"gdt101_remap ran past {REMAP_TIMEOUT_SECONDS} s ({' '.join(argv[:3])} ...); "
            f"its log is {log}"
        ) from error
    elapsed = time.perf_counter() - started
    log.write_text(f"$ {' '.join(argv)}\n[{elapsed:.1f} s, exit {done.returncode}]\n"
                   f"--- stdout ---\n{done.stdout}\n--- stderr ---\n{done.stderr}\n", encoding="utf-8")
    if done.returncode != 0:
        tail = (done.stderr or done.stdout or "").strip().splitlines()
        raise GlobalIntermediateRefusal(
            f"gdt101_remap exited {done.returncode}; its log is {log}"
            + (f".  It said: {tail[-1]}" if tail else "")
        )
    try:
        receipt = json.loads(done.stdout)
    except json.JSONDecodeError as error:
        raise GlobalIntermediateRefusal(
            f"gdt101_remap exited 0 without its JSON receipt ({error}); log {log}"
        ) from error
    if receipt.get("contract") != REMAP_CONTRACT or receipt.get("global") is not True \
            or receipt.get("mode") != REMAP_GLOBAL_MODE:
        raise GlobalIntermediateRefusal(
            f"gdt101_remap's receipt names contract {receipt.get('contract')!r}, mode "
            f"{receipt.get('mode')!r}, global {receipt.get('global')!r}; expected "
            f"{REMAP_CONTRACT!r} + {REMAP_GLOBAL_MODE!r}, global true"
        )
    target = receipt.get("target") or {}
    if {k: target.get(k) for k in ("nx", "ny", "west", "south", "dx", "dy")} != {
            "nx": GLOBAL_GRID["nx"], "ny": GLOBAL_GRID["ny"], "west": 0, "south": -90,
            "dx": GLOBAL_DX_DEG, "dy": GLOBAL_DX_DEG}:
        raise GlobalIntermediateRefusal(f"gdt101_remap wrote target {target}, not the 0.125-degree sphere")
    if receipt.get("grid_uuid") != ICON_GRID_UUID:
        raise GlobalIntermediateRefusal(
            f"the records are on mesh {receipt.get('grid_uuid')!r}, not ICON R03B07 {ICON_GRID_UUID}"
        )
    receipt["wall_seconds"] = round(elapsed, 3)
    return receipt


def build_plan(remap: Path, invariants: Mapping[str, Path], cycle: datetime,
               work: Path) -> tuple[Path, dict[str, Any]]:
    """One full-sphere weights plan from the cycle's CLAT/CLON/FR_LAND."""

    plan = work / "weights-global-0p125.plan"
    land = INVARIANTS[0]
    argv = [str(remap), "plan", "--global", "--dx-deg", repr(GLOBAL_DX_DEG)]
    for field in (GEOMETRY["CLAT"], GEOMETRY["CLON"], land):
        argv += [str(invariants[field.name]), ",".join(_selector_token(field.selector(None)))]
    argv += [str(plan), f"{cycle:%Y%m%d%H}", str(ICON_CELLS), str(ICON_CENTRE)]
    receipt = _run_remap(argv, work / "gdt101_remap.plan.log")
    return plan, receipt


def apply_jobs(remap: Path, plan: Path, rows: Sequence[str], work: Path, tag: str,
               workers: int) -> list[dict[str, Any]]:
    """Run the job rows in ``workers`` parallel ``apply --global`` batches."""

    workers = max(1, min(int(workers), len(rows)))
    batches = [rows[k::workers] for k in range(workers)]

    def one(index: int) -> dict[str, Any]:
        jobs = work / f"{tag}.batch{index}.jobs"
        jobs.write_text("".join(batches[index]), encoding="utf-8", newline="\n")
        try:
            return _run_remap([str(remap), "apply", "--global", str(plan), str(jobs)],
                              work / f"gdt101_remap.{tag}.batch{index}.log")
        finally:
            jobs.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        receipts = list(pool.map(one, range(workers)))
    fields = [item for receipt in receipts for item in receipt.get("fields", [])]
    if len(fields) != len(rows):
        raise GlobalIntermediateRefusal(
            f"gdt101_remap reported {len(fields)} remapped fields for {len(rows)} jobs"
        )
    return receipts


#: met_intermediate holds every matched field of a valid time as float64
#: before it writes: measured 3.78 GB peak for the 118 records of one
#: pressure-level lead on the 4,150,080-point sphere, 32 MB a record.
MET_BYTES_PER_FIELD = GLOBAL_GRID["nx"] * GLOBAL_GRID["ny"] * 8


def available_memory_bytes() -> int | None:
    """MemAvailable from /proc/meminfo, or None where there is no such file."""

    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


#: One ``apply --global`` process: the ~200 MB plan, a native field and
#: its remapped copy (~350 MB measured), rounded up.
APPLY_BYTES_PER_WORKER = 400 * 1024 * 1024
#: The ``--workers`` ceiling the shared parser documents.
MAX_REMAP_WORKERS = 13


def refuse_unaffordable_lead(fields: int, available: int | None = None, *, workers: int = 1) -> None:
    """Refuse a lead whose met_intermediate pass cannot fit in memory.

    A model-level lead is ~720 fields, ~23 GB of float64 on this grid; a
    machine without that much free memory would be killed half-way rather
    than refused, so the door measures first.
    """

    have = available_memory_bytes() if available is None else available
    remap_need = int(workers) * APPLY_BYTES_PER_WORKER
    if have is not None and remap_need > have:
        raise GlobalIntermediateRefusal(
            f"--workers {workers} runs {workers} gdt101_remap processes of about "
            f"{APPLY_BYTES_PER_WORKER / 1e9:.1f} GB each, and this machine has "
            f"{have / 1e9:.1f} GB available; pass fewer --workers"
        )
    need = int(fields) * MET_BYTES_PER_FIELD
    if have is not None and need > have:
        raise GlobalIntermediateRefusal(
            f"one lead is {fields} fields on the {GLOBAL_GRID['nx']} x {GLOBAL_GRID['ny']} "
            f"sphere; met_intermediate holds them all as float64, about {need / 1e9:.1f} GB, "
            f"and this machine has {have / 1e9:.1f} GB available.  Free memory, or use the "
            f"pressure-level row (icon-global: 119 fields, about 4 GB a lead)"
        )


def _timing_summary(receipts: Sequence[Mapping[str, Any]], wall: float) -> dict[str, Any]:
    totals = [float(f["total_seconds"]) for r in receipts for f in r.get("fields", [])]
    return {
        "fields": len(totals), "batches": len(receipts), "wall_seconds": round(wall, 3),
        "per_field_seconds": {
            "min": round(min(totals), 3), "median": round(statistics.median(totals), 3),
            "max": round(max(totals), 3),
        } if totals else None,
        "plan_read_seconds": [r.get("plan_read_seconds") for r in receipts],
    }


# ---------------------------------------------------------------------------
# the door
# ---------------------------------------------------------------------------
@dataclass
class IconGlobalRequest:
    row: IconSourceRow
    grib_dir: Path
    cycle: datetime
    leads: tuple[int, ...]
    interval_hours: int
    out_dir: Path
    mesh: dict[str, Any]
    prefix: str = "MET"
    engine: Path | None = None
    vtable: Path | None = None
    remap: Path | None = None
    workers: int = DEFAULT_REMAP_WORKERS


def _vtable_for(row: IconSourceRow, explicit: Path | None) -> Path:
    if explicit is not None:
        vtable = Path(explicit).expanduser().absolute()
        if not vtable.is_file():
            raise GlobalIntermediateRefusal(f"--vtable {vtable} is not a file")
        return vtable
    packaged = Path(__file__).resolve().parents[1] / "data" / "vtables" / row.vtable
    if not packaged.is_file():
        # MERGE SEAM: the ICON Vtables and met_intermediate's "dwd" rules
        # are one sibling change; without it there is nothing to read the
        # regridded GRIB with, and reinstalling would not help.
        raise GlobalIntermediateRefusal(
            f"{row.vtable} is not in this install: it arrives with met_intermediate's "
            f"'--map-source {row.map_source}' rules (ICON's soil water and layer "
            f"conventions), which this build does not carry.  Use a build with both, "
            f"or pass --vtable with a met_intermediate that knows '{row.map_source}'"
        )
    return packaged_vtable(row.vtable)


def build_icon_global_intermediates(
    request: IconGlobalRequest, *, log: Callable[[str], None] = print
) -> dict[str, Any]:
    from .met_intermediate_door import LeadPlan, _run_one, engine_argv

    row = request.row
    if request.mesh.get("kind") != "global":
        raise GlobalIntermediateRefusal(
            f"the full-sphere ICON regrid is for a global mesh; {request.mesh.get('path')} "
            f"is {request.mesh.get('kind')!r}"
        )
    if request.leads[-1] > row.max_hour:
        raise GlobalIntermediateRefusal(
            f"{row.name} publishes leads to PT{row.max_hour:03d}; --hours asks for "
            f"PT{request.leads[-1]:03d}"
        )
    root = tree_root(request.grib_dir)
    invariants = locate_invariants(root, request.cycle)
    lead_objects = {lead: locate_lead(root, row, request.cycle, lead) for lead in request.leads}
    refuse_unaffordable_lead(max(len(objects) for objects in lead_objects.values()) + len(INVARIANTS),
                             workers=request.workers)
    engine = resolve_engine(request.engine)
    remap = resolve_remap(request.remap)
    vtable = _vtable_for(row, request.vtable)
    source = row.as_global_source()
    out_dir = Path(request.out_dir).expanduser().absolute()
    refuse_occupied_out_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / ".icon-global-regrid"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    digests: dict[Path, dict[str, Any]] = {}
    partials: list[Path] = []
    try:
        log(f"{row.name}: planning the 0.125-degree full-sphere regrid from "
            f"{request.cycle:%Y-%m-%dT%H}Z CLAT/CLON/FR_LAND")
        plan, plan_receipt = build_plan(remap, invariants, request.cycle, work)
        log(f"  plan: {plan_receipt.get('total_seconds')} s, {plan.stat().st_size / 1e6:.0f} MB, "
            f"grid {plan_receipt['grid_uuid']}")
        # The invariants, remapped once from PT000: inputs of hour zero, the
        # --invariant files of every later lead.
        rows, invariant_parts = [], []
        for field in INVARIANTS:
            destination = work / f"invariant.{field.name}.grib2"
            rows.append(job_row(invariants[field.name], destination, field.selector(None),
                                field.method, request.cycle, 0))
            invariant_parts.append(destination)
        started = time.perf_counter()
        invariant_receipts = apply_jobs(remap, plan, rows, work, "invariant", request.workers)
        invariant_timing = _timing_summary(invariant_receipts, time.perf_counter() - started)
        checked_all = []
        for lead in request.leads:
            valid = request.cycle + timedelta(hours=int(lead))
            objects = lead_objects[lead]
            rows, parts = [], []
            for index, obj in enumerate(objects):
                destination = work / f"f{lead:03d}.{index:04d}.grib2"
                rows.append(job_row(obj.path, destination, obj.selector(), obj.field.method,
                                    request.cycle, lead))
                parts.append(destination)
            started = time.perf_counter()
            receipts = apply_jobs(remap, plan, rows, work, f"f{lead:03d}", request.workers)
            timing = _timing_summary(receipts, time.perf_counter() - started)
            log(f"  PT{lead:03d}: {len(objects)} fields remapped in {timing['wall_seconds']} s "
                f"(per field {timing['per_field_seconds']})")
            out = out_dir / f"{request.prefix}:{valid:%Y-%m-%d_%H}"
            lead_plan = LeadPlan(
                lead=lead, valid_time=valid,
                inputs=[*parts, *invariant_parts] if lead == 0 else parts,
                invariants=[] if lead == 0 else list(invariant_parts), out=out,
            )
            lead_plan.argv = engine_argv(engine, vtable, source, lead_plan, _partial(out))
            # Registered before the engine starts, so an interrupt mid-write
            # still removes the half-written file.
            partials.append(_partial(out))
            _, done, elapsed = _run_one(lead_plan, out_dir)
            regridded_bytes = sum(path.stat().st_size for path in parts)
            for path in parts:
                path.unlink()
            if done.returncode != 0:
                tail = (done.stderr or done.stdout or "").strip().splitlines()
                raise GlobalIntermediateRefusal(
                    f"met_intermediate exited {done.returncode} for PT{lead:03d}; its log is "
                    f"{out_dir / f'met_intermediate.f{lead:03d}.log'}"
                    + (f".  It said: {tail[-1]}" if tail else "")
                )
            receipt = parse_engine_receipt(done.stdout)
            checked = validate_intermediate(_partial(out), hdate=lead_plan.hdate, engine_receipt=receipt)
            grid = checked["grid"]
            if {k: grid[k] for k in GLOBAL_GRID} != GLOBAL_GRID:
                raise GlobalIntermediateRefusal(
                    f"{out.name}: the intermediate grid is {grid}, not the full-sphere "
                    f"{GLOBAL_GRID} the regrid wrote"
                )
            checked_all.append((lead_plan, receipt, elapsed, checked, timing, regridded_bytes, objects))
        init_switches(source, [item[3] for item in checked_all])
    except BaseException:
        # Nothing half-made is left for `woof hex lbc` to find; the remap
        # logs stay in the work directory, the gigabytes of GRIB do not.
        for partial in partials:
            partial.unlink(missing_ok=True)
        for leftover in (*work.glob("*.grib2"), *work.glob("*.plan")):
            leftover.unlink(missing_ok=True)
        raise
    files = []
    for lead_plan, receipt, elapsed, checked, timing, regridded_bytes, objects in checked_all:
        out = lead_plan.out
        os.replace(_partial(out), out)
        checked["path"] = str(out)
        checked["lead"] = lead_plan.lead
        checked["engine_seconds"] = round(elapsed, 2)
        checked["remap"] = timing
        checked["regridded_grib"] = {"bytes": regridded_bytes, "fields": len(objects),
                                     "invariant_fields": len(INVARIANTS),
                                     "invariants_as": "inputs" if lead_plan.lead == 0 else "--invariant"}
        checked["inputs"] = [_described(obj.path, digests) for obj in objects]
        checked["engine_receipt"] = {key: receipt.get(key) for key in (
            "schema", "map_source", "grib_messages_read", "grib_messages_matched",
            "records_written", "valid_time_selection", "messages_other_valid_time",
            "messages_time_processed_skipped", "matched_valid_times",
            "isobaric_levels_dropped_above_pmin", "bitmap_masked_points", "invariants",
            "rules", "fields_held")}
        files.append(checked)
        log(f"WROTE {out.name}: {checked['field_records']} records, "
            f"{checked['bytes'] / 1e6:.0f} MB, remap {timing['wall_seconds']} s, "
            f"met_intermediate {checked['engine_seconds']} s")
    shutil.rmtree(work, ignore_errors=True)
    switches = init_switches(source, files)
    receipt = {
        "schema": SCHEMA,
        "source": row.name,
        "source_description": row.description,
        "attribution": ATTRIBUTION,
        "mesh": request.mesh,
        "regridding": {
            "engine": "gdt101_remap --global",
            "contract": REMAP_CONTRACT,
            "mode": REMAP_GLOBAL_MODE,
            "global": True,
            "dx_deg": GLOBAL_DX_DEG,
            "target": dict(GLOBAL_GRID),
            "source_grid_uuid": plan_receipt.get("grid_uuid"),
            "source_cells": plan_receipt.get("source_cells"),
            "methods": {"aloft": "idw4", "surface": "surface-idw4", "soil": "soil-nearest",
                        "sea_ice": "seaice-nearest", "land_fraction": "nearest"},
            "weights": {"built_once_and_reused_for_every_field_and_lead": True,
                        "plan_receipt": plan_receipt},
            "invariants": {"fields": [f.name for f in INVARIANTS], "remap": invariant_timing,
                           "inputs": [_described(invariants[f.name], digests)
                                      for f in (*GEOMETRY.values(), *INVARIANTS)]},
            "remapper": _described(remap, digests),
        },
        "cycle": f"{request.cycle:%Y-%m-%d_%H:%M:%S}",
        "leads": list(request.leads),
        "interval_hours": int(request.interval_hours),
        "valid_times": [f["valid_time"] for f in files],
        "grib_dir": str(Path(request.grib_dir).expanduser().absolute()),
        "map_source": row.map_source,
        "vtable": _described(vtable, digests),
        "engine": _described(engine, digests),
        "engine_receipt_schema": ENGINE_RECEIPT_SCHEMA,
        "files": files,
        "init_switches": switches,
    }
    receipt_path = out_dir / "intermediate-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
                            encoding="utf-8", newline="\n")
    log(f"RECEIPT {receipt_path}")
    return receipt


def request_from_arguments(arguments: argparse.Namespace, mesh: dict[str, Any]) -> IconGlobalRequest:
    from .hrrr_intermediate import parse_cycle

    row = ICON_GLOBAL_ROWS[str(arguments.source).strip().lower()]
    foreign = [flag for flag, attribute in (
        ("--from-plan", "from_plan"), ("--point", "point"), ("--radius-km", "radius_km"),
        ("--spacing-deg", "spacing_deg"), ("--wrfout-glob", "wrfout_glob"),
        ("--cull-region", "cull_region"), ("--halo-km", "halo_km"),
        ("--wrf-edge-cells", "wrf_edge_cells"),
    ) if getattr(arguments, attribute, None) is not None]
    if foreign:
        raise GlobalIntermediateRefusal(
            f"{', '.join(foreign)} do not apply to a global mesh: --source {row.name} onto "
            f"{mesh.get('path')} is regridded to the full sphere at {GLOBAL_DX_DEG} degrees"
        )
    for flag, value in (("--grib-dir", getattr(arguments, "grib_dir", None)),
                        ("--cycle", getattr(arguments, "cycle", None))):
        if value is None:
            raise GlobalIntermediateRefusal(f"--source {row.name} needs {flag}; it has no default")
    interval = getattr(arguments, "interval_hours", None)
    interval = row.default_interval_hours if interval is None else int(interval)
    leads = parse_lead_range(getattr(arguments, "hours", None), interval)
    workers = getattr(arguments, "workers", None)
    workers = DEFAULT_REMAP_WORKERS if workers is None else int(workers)
    if not 1 <= workers <= MAX_REMAP_WORKERS:
        raise GlobalIntermediateRefusal(
            f"--workers {workers} is not 1..{MAX_REMAP_WORKERS} (parallel gdt101_remap processes)"
        )
    return IconGlobalRequest(
        row=row, grib_dir=Path(arguments.grib_dir), cycle=parse_cycle(arguments.cycle),
        leads=leads, interval_hours=interval, out_dir=Path(arguments.out_dir), mesh=mesh,
        prefix=str(getattr(arguments, "prefix", "MET") or "MET"),
        engine=getattr(arguments, "decoder", None), vtable=getattr(arguments, "vtable", None),
        workers=workers,
    )


def run_icon_global_intermediate(arguments: argparse.Namespace, mesh: dict[str, Any]) -> int:
    request = request_from_arguments(arguments, mesh)
    receipt = build_icon_global_intermediates(request)
    switches = receipt["init_switches"]
    out_dir = Path(request.out_dir).expanduser().absolute()
    first = receipt["files"][0]["path"]
    summary: dict[str, Any] = {
        "files": [item["path"] for item in receipt["files"]],
        "receipt": str(out_dir / "intermediate-receipt.json"),
        "attribution": ATTRIBUTION,
        "init_switches": switches,
        "next": (f"woof hex init --met {first} ... --nfglevels {switches['--nfglevels']} "
                 f"--nfgsoillevels {switches['--nfgsoillevels']} --use-spechumd "
                 f"{switches['--use-spechumd']} --extrap-airtemp {switches['--extrap-airtemp']}"),
    }
    print(json.dumps(summary, indent=2))
    return 0


__all__ = [
    "ATTRIBUTION",
    "GLOBAL_DX_DEG",
    "GLOBAL_GRID",
    "ICON_GLOBAL_ROWS",
    "IconField",
    "IconGlobalRequest",
    "IconSourceRow",
    "REMAP_CONTRACT",
    "REMAP_GLOBAL_MODE",
    "SCHEMA",
    "build_icon_global_intermediates",
    "classify_mesh",
    "job_row",
    "locate_invariants",
    "locate_lead",
    "request_from_arguments",
    "resolve_remap",
    "run_icon_global_intermediate",
    "tree_root",
]
