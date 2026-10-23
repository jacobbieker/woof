"""Turbine layout: a turbine table placed on a domain's mass grid.

The woof equivalent of WRF's ``windturbines.txt`` (one power/thrust table
per turbine type) and ``windfarm_ij`` (each turbine's cell), built here as
one frozen :class:`WindFarmLayout` of per-cell arrays that a wind-farm
parameterization reads.  Nothing here runs in the model: this module reads
the table, resolves the curves and places the turbines, and refuses by
name anything it cannot place or describe.

THE TABLE.  CSV (header row) or a GeoJSON FeatureCollection of Points
(coordinates ``[lon, lat]``, the columns as properties), one row per
turbine:

``lat`` / ``lon``
    Degrees (``latitude``/``longitude`` are accepted).
``hub_height_m``
    Hub height above ground.  Required: none is invented (as
    :mod:`woof.energy.sites` does not invent one).
``rotor_diameter_m``
    Optional; blank takes the curve's.  One that disagrees with the
    curve's by more than ``SIZE_ROTOR_TOLERANCE`` (the library's own
    "same machine" tolerance) is refused: the thrust table belongs to
    the curve's rotor.
``curve``
    A ``turbine_library.json`` id (``V112/3000``) or a model name the
    library matches ``exact``, or ``fuzzy`` only when the name parses to
    both a rotor and a rating and the curve is that machine (rating
    within ``FUZZY_RATING_MATCH``): ``V112 3MW``, not a bare ``V112``
    (three machines), never a name-similarity hit (``curve_id``,
    ``model``, ``turbine_type`` and ``type`` are accepted spellings, in
    that order; an optional ``manufacturer`` column must agree with the
    curve's).
    A blank curve takes the caller's ``default_curve``; with none, the
    turbine is refused.  The library's ``size``/``generic`` rungs are never
    used here: a parametric stand-in is not the machine's curve.
``rated_kw``
    Optional; when given it must be within ``RATING_TOLERANCE`` of the
    curve's rating, or the turbine is refused.
``turbine_id``
    Optional (``id``/``name``); default ``t<row>`` counted from 1.

A ``woof-energy.sites.v1`` document (:mod:`woof.energy.sites`) is
accepted too: its ``turbine`` sites, with their own hub heights and the
``default_curve`` for every one of them (a site carries a rating, not a
model), and each site's ``capacity_mw`` checked against that curve's
rating as ``rated_kw`` is.

PLACEMENT IS WRF'S OWN.  A turbine lands where WRF v4.7.1's Fitch
initialisation puts it (:mod:`woof.windfarm.wrf_llij`): its lat/lon as
float32, WRF's real(4) ``map_set`` and ``latlon_to_ij`` for the domain's
projection (Lambert conformal, Mercator or polar stereographic) anchored at
the float32 ``XLAT``/``XLONG`` of the domain's first mass point (the values
woof writes as its own ``XLAT``/``XLONG``) with the domain's own ``dx``, and
``nint``.  A nest is placed the same way from its own corner and spacing,
which is how its parent start and ratio reach WRF.  ``turbine_x``/
``turbine_y`` are WRF's float32 ``ts_rx``/``ts_ry`` (1-based) and the
0-based cell is ``nint - 1``; both are held at max_ulp 0 against the
unmodified WRF source (``tools/windfarm_layout_wrf471_oracle``).  Any other
projection is refused by name: WRF's wind-farm code has no arm for it.  A
turbine outside the mass grid, or in its outermost row or column (the rule
:func:`woof.core.hub_wind_diag.locate_sites` applies to hub sites: those
cells are boundary-forced, not modelled), is refused by name.  (WRF's own
outside test, ``ival < ids .and. ival > ide``, can never be true; woof
refuses where WRF would index out of its patch.)

GROUPS: A CELL IS A LIST OF MACHINES.  A cell may hold turbines of
several curves, hub heights and rotors, as WRF's Fitch scheme allows
when ``windturbines.txt`` lists several types in one cell: each type
acts over its own rotor layers and the cell sums them, turbine by turbine
in the order it reads ``windturbines.txt``.  So a cell is a list of
groups: its turbines in table order, split into runs of consecutive
turbines with the same curve and bit-identical float32 hub height and
rotor.  A kernel that walks a cell's groups in order and applies each
group's per-turbine contribution ``count`` times, one turbine at a time,
adds what WRF adds in WRF's order (WRF subtracts every turbine's
tendency separately, so ``count * contribution`` in one step rounds
differently and is not WRF-exact).  A cell listing A, B, A is three
groups; a uniform cell is one.  ``build_layout(merge_tolerance_m=...)`` merges same-curve
turbines within a tolerance anywhere in the cell into averaged groups to
save memory on dense tables; that is opt-in and NOT WRF-exact.

The groups are stored CSR-style rather than on a fixed
``max_types_per_cell`` axis: flat per-group arrays sorted by row-major
cell, and a dense ``cell_group_offset`` of ``ny * nx + 1`` int32 into
them.  A per-cell kernel (one thread per column, as the dycore's column
physics runs) reads its two offsets and loops over a contiguous slice; a
per-group kernel launches over the flat arrays and finds its column in
``group_j``/``group_i``.  Neither pays for padding, there is no cap on
types per cell to refuse against, and the groups of neighbouring cells
are adjacent in memory.  The per-cell aggregates (turbine count, rated
capacity, swept area, rotor top and bottom) let a kernel skip empty
columns and bound its level loop before it touches the groups.

A turbine whose rotor bottom (hub height minus half the rotor) is below
0 m AGL is refused by name.

THE TABLES ARE WRF'S.  Each distinct curve becomes the table WRF's
``wind-turbine-N.tbl`` would hold (:mod:`woof.windfarm.fitch_tables`:
``turbws``/``turbtc``/``turbpw`` rows, cut-in, cut-out, standing thrust
coefficients), in ``tables``.  The convenience axis tables
``group_thrust_coefficient``, ``group_power_coefficient`` and
``group_power_w`` on ``speeds_ms`` (``0 .. TABLE_MAX_SPEED_MS`` every
``TABLE_SPEED_STEP_MS``) are WRF's ``dragcof`` and ``power1`` evaluated
there for each group, at its curve and its own rotor diameter (the power
coefficient scales with 1 / area): thrust ``stc`` at or below cut-in and ``stc2`` at or above
cut-out, no power outside, and no hysteresis (WRF has none).
:meth:`WindFarmLayout.write_wrf_inputs` writes ``windturbines.txt`` and
the tables for a WRF run of the same farm.

FLOATS ARE WRF'S REAL(4).  Every value a kernel reads (group hub heights
and rotors, per-cell aggregates, tables, positions) is float32 on the host
already; per-cell sums are taken in binary64 in table order (turbine 1
first) and rounded once.  :meth:`WindFarmLayout.to_device` uploads those
words unchanged (int32 indices, float32 values, contiguous) into a
:class:`DeviceWindFarmLayout` a kernel holds for the run: the upload is a
copy, raw-word equal both ways.  Reading, matching and placement are
one-off CPU setup.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from woof.energy.turbine_library import (
    FUZZY_CAPACITY_TOLERANCE,
    FUZZY_ROTOR_TOLERANCE_M,
    SIZE_ROTOR_TOLERANCE,
    TurbineCurve,
    TurbineLibrary,
    TurbineSpec,
    canonical_manufacturer,
    load_turbine_library,
    parse_model,
)
from woof.explain import layered
from woof.windfarm.fitch_tables import (
    MAXVALS,
    PICONST,
    STANDING_THRUST_COEFFICIENT,
    FitchTableRefused,
    FitchTables,
    dragcof_cpu,
    float32_word,
    write_table,
)
from woof.windfarm.wrf_llij import (WrfProjection, WrfProjectionRefused,
                                    nint)

F = np.float32

LAYOUT_SCHEMA = "woof-windfarm.layout-summary.v1"

#: The shared speed axis of every power/thrust table, m s-1.
TABLE_SPEED_STEP_MS = 0.5
TABLE_MAX_SPEED_MS = 35.0

#: A suggested ``merge_tolerance_m`` for the opt-in, not-WRF-exact
#: tolerance grouping (the default groups bit-identical runs only).
GROUP_TOLERANCE_M = 1.0

#: Library match rungs a table's curve name may resolve through, and how
#: closely a fuzzy match's curve must carry the rating its name parses to.
ACCEPTED_MATCHES = ("exact", "fuzzy")
FUZZY_RATING_MATCH = 0.005

#: A turbine that states its own rating (a sites.v1 ``capacity_mw``, a
#: ``rated_kw`` column) must be within this of its curve's.
RATING_TOLERANCE = FUZZY_CAPACITY_TOLERANCE

#: Column spellings, canonical name first.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "lat": ("lat", "latitude"),
    "lon": ("lon", "longitude"),
    "hub_height_m": ("hub_height_m", "hub_height"),
    "rotor_diameter_m": ("rotor_diameter_m", "rotor_diameter"),
    "curve": ("curve", "curve_id", "model", "turbine_type", "type"),
    "rated_kw": ("rated_kw",),
    "turbine_id": ("turbine_id", "id", "name"),
    "manufacturer": ("manufacturer",),
}


class WindFarmLayoutRefused(ValueError):
    """A turbine table cannot be placed on the domain as given."""


@dataclass(frozen=True)
class Turbine:
    """One row of a turbine table (``rotor_diameter_m`` None: the curve's)."""

    turbine_id: str
    lat: float
    lon: float
    hub_height_m: float | None
    rotor_diameter_m: float | None
    curve: str | None
    manufacturer: str | None = None
    rated_kw: float | None = None


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class WindFarmLayout:
    """Turbines on one domain's mass grid: groups per cell, plus tables.

    GROUPS (CSR).  A group is a run of consecutive turbines, in table
    order within one cell, with the same curve and bit-identical float32
    hub height and rotor (the module docstring says why; the opt-in
    ``merge_tolerance_m`` relaxes this and is not WRF-exact).  Groups are
    stored flat, sorted by row-major cell ``j * nx + i`` and then by
    table order within the cell, so cell ``c``'s groups, in the order WRF
    adds them, are ``g`` in ``cell_group_offset[c] : cell_group_offset[c +
    1]``.  Per group: ``group_cell``/``group_j``/``group_i`` (its cell),
    ``group_turbines``, ``group_curve`` (into ``curve_ids``),
    ``group_hub_height_m`` and ``group_rotor_diameter_m``;
    :meth:`kernel_groups` gives them under the Fitch kernel's names.

    Per cell, ``(ny, nx)`` in geo_em ``[j, i]`` order, aggregates a kernel
    reads before its group loop: ``turbine_count`` and ``group_count``
    (int32, 0 where empty); ``rated_capacity_kw`` (sum of the turbines'
    ratings, in table order) and ``swept_area_m2`` (sum over the groups of
    turbines times WRF's ``piconst/4.*diameter**2.``, in group order), 0
    where empty; ``rotor_top_m`` / ``rotor_bottom_m``, the highest
    ``hub + D/2.`` and lowest ``hub - D/2.`` of the cell's groups (WRF's
    ``blade_u_point``/``blade_l_point``), NaN where empty: the levels the
    cell's rotors span.  Sums run in binary64 and round once to float32.

    Per curve: ``rated_kw`` and ``tables``, the WRF Fitch tables.  Per
    group, on ``speeds_ms``: ``group_thrust_coefficient``,
    ``group_power_coefficient`` and ``group_power_w``, WRF's
    ``dragcof``/``power1`` at the group's curve and rotor.  Per
    turbine, in table order: ids, its float32 ``turbine_lat``/
    ``turbine_lon``, WRF's float32 ``turbine_x``/``turbine_y`` (1-based),
    the 0-based cell ``(turbine_j, turbine_i)``, the curve and group
    index.  ``projection`` is WRF's ``proj_info``.  Every array is
    read-only; every float is float32 and every index int32.
    """

    grid_id: int
    ny: int
    nx: int
    dx_m: float
    projection: WrfProjection
    turbine_count: np.ndarray
    group_count: np.ndarray
    rated_capacity_kw: np.ndarray
    swept_area_m2: np.ndarray
    rotor_top_m: np.ndarray
    rotor_bottom_m: np.ndarray
    cell_group_offset: np.ndarray
    group_cell: np.ndarray
    group_j: np.ndarray
    group_i: np.ndarray
    group_turbines: np.ndarray
    group_curve: np.ndarray
    group_hub_height_m: np.ndarray
    group_rotor_diameter_m: np.ndarray
    curve_ids: tuple[str, ...]
    rated_kw: np.ndarray
    tables: FitchTables
    speeds_ms: np.ndarray
    group_thrust_coefficient: np.ndarray
    group_power_coefficient: np.ndarray
    group_power_w: np.ndarray
    turbine_ids: tuple[str, ...]
    turbine_lat: np.ndarray
    turbine_lon: np.ndarray
    turbine_x: np.ndarray
    turbine_y: np.ndarray
    turbine_j: np.ndarray
    turbine_i: np.ndarray
    turbine_curve: np.ndarray
    turbine_group: np.ndarray
    library_sha256: str

    @property
    def n_turbines(self) -> int:
        return len(self.turbine_ids)

    @property
    def n_groups(self) -> int:
        return int(self.group_turbines.size)

    @property
    def cells_occupied(self) -> int:
        return int(np.count_nonzero(self.turbine_count))

    @property
    def rated_capacity_mw(self) -> float:
        return float(self.rated_kw[self.turbine_curve].astype(np.float64)
                     .sum()) / 1000.0

    def groups_of(self, j: int, i: int) -> range:
        """Indices of cell ``(j, i)``'s groups (empty for an empty cell)."""

        c = int(j) * self.nx + int(i)
        return range(int(self.cell_group_offset[c]),
                     int(self.cell_group_offset[c + 1]))

    def cells(self) -> list[dict[str, Any]]:
        """One record per occupied cell, row-major, with its groups."""

        out = []
        for j, i in zip(*np.nonzero(self.turbine_count)):
            groups = []
            for g in self.groups_of(j, i):
                k = int(self.group_curve[g])
                count = int(self.group_turbines[g])
                groups.append({
                    "curve": self.curve_ids[k], "turbines": count,
                    "hub_height_m": _json_number(self.group_hub_height_m[g]),
                    "rotor_diameter_m":
                        _json_number(self.group_rotor_diameter_m[g]),
                    "capacity_mw": count * _json_number(self.rated_kw[k]) / 1000.0,
                })
            out.append({
                "j": int(j), "i": int(i),
                "turbines": int(self.turbine_count[j, i]),
                "capacity_mw": _json_number(self.rated_capacity_kw[j, i]) / 1000.0,
                "swept_area_m2": _json_number(self.swept_area_m2[j, i]),
                "rotor_top_m": _json_number(self.rotor_top_m[j, i]),
                "rotor_bottom_m": _json_number(self.rotor_bottom_m[j, i]),
                "groups": groups,
            })
        return out

    def summary(self) -> dict[str, Any]:
        """The record ``woof windfarm layout --json`` prints."""

        per_curve = np.bincount(self.turbine_curve,
                                minlength=len(self.curve_ids))
        return {
            "schema": LAYOUT_SCHEMA,
            "grid_id": self.grid_id, "ny": self.ny, "nx": self.nx,
            "dx_m": self.dx_m,
            "turbines": self.n_turbines,
            "groups": self.n_groups,
            "cells_occupied": self.cells_occupied,
            "max_turbines_per_cell": int(self.turbine_count.max()),
            "max_groups_per_cell": int(self.group_count.max()),
            "rated_capacity_mw": round(self.rated_capacity_mw, 6),
            "curves": [{"curve": cid, "rated_kw": _json_number(self.rated_kw[k]),
                        "cut_in_ms": _json_number(self.tables.cut_in_ms[k]),
                        "cut_out_ms": _json_number(self.tables.cut_out_ms[k]),
                        "table_rows": int(self.tables.nval[k]),
                        "turbines": int(per_curve[k])}
                       for k, cid in enumerate(self.curve_ids)],
            "cells": self.cells(),
            "turbine_library_sha256": self.library_sha256,
        }

    def kernel_groups(self) -> dict[str, np.ndarray]:
        """The group arrays under the Fitch kernel's field names."""

        return _kernel_groups(self)

    def kernel_curves(self) -> dict[str, np.ndarray]:
        """The WRF tables under the Fitch kernel's field names."""

        return _kernel_curves(self.tables)

    def wrf_types(self) -> tuple[np.ndarray, list[tuple[int, Any, Any]]]:
        """WRF turbine types: ``(type of each group, [(curve, hub, rotor)])``.

        A type is one machine, a distinct ``(curve, hub height, rotor)``
        word triple over the groups, numbered from 0 in that sorted order;
        WRF's ``nkind`` is the type + 1.
        """

        per_group = [(int(self.group_curve[g]),
                      float(self.group_hub_height_m[g]),
                      float(self.group_rotor_diameter_m[g]))
                     for g in range(self.n_groups)]
        keys = sorted(set(per_group))
        index = {key: t for t, key in enumerate(keys)}
        of_group = np.array([index[key] for key in per_group],
                            dtype=np.int32)
        return of_group, [(k, F(h), F(d)) for k, h, d in keys]

    def write_wrf_inputs(self, directory) -> Path:
        """``windturbines.txt`` and one ``wind-turbine-N.tbl`` per type.

        WRF's Fitch inputs for the same farm (``windfarm_opt = 1``,
        ``windfarm_ij = 0``): turbine ``n`` is ``lat lon type`` with its
        float32 ``turbine_lat``/``turbine_lon`` and type from
        :meth:`wrf_types`, every number the shortest decimal that reads
        back as the layout's float32 word.  WRF holds at most ``MAXVALS``
        = 100 types; a layout of more machines is refused.
        """

        of_group, types = self.wrf_types()
        if len(types) > MAXVALS:
            raise WindFarmLayoutRefused(
                f"{len(types)} distinct machines (curve, hub height, rotor); "
                f"WRF's Fitch scheme reads at most MAXVALS = {MAXVALS} "
                "turbine types")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        kind = of_group[self.turbine_group] + 1
        lines = [f"{float32_word(self.turbine_lat[n])} "
                 f"{float32_word(self.turbine_lon[n])} "
                 f"{int(kind[n])}" for n in range(self.n_turbines)]
        (directory / "windturbines.txt").write_text(
            "\n".join(lines) + "\n", encoding="ascii")
        for t, (k, hub, rotor) in enumerate(types):
            write_table(directory / f"wind-turbine-{t + 1}.tbl", self.tables,
                        k, hub_height_m=hub, diameter_m=rotor,
                        nominal_power_mw=F(self.rated_kw[k] / F(1000.0)))
        return directory

    def to_device(self, *, device_id: int | None = None
                  ) -> "DeviceWindFarmLayout":
        """Upload what a wind-farm kernel reads, once, to CuPy.

        A copy of the host words (int32 indices, float32 values, all
        contiguous): ``cp.asnumpy`` of any device array is raw-word equal
        to the host array.  Build it once per domain and hold it; a kernel
        then reads it every step with no host traffic.
        """

        xp = _cupy()

        def put(a):
            host = np.ascontiguousarray(a)
            if host.dtype not in (np.int32, np.float32):
                raise WindFarmLayoutRefused(
                    f"a {host.dtype} array cannot go to the device as is")
            return xp.ascontiguousarray(xp.asarray(host))

        def upload():
            arrays = {name: put(getattr(self, name))
                      for name in DEVICE_ARRAYS}
            tables = FitchTables(**{name: put(value) for name, value
                                    in self.tables.arrays().items()})
            return DeviceWindFarmLayout(
                ny=self.ny, nx=self.nx, n_groups=self.n_groups,
                n_curves=len(self.curve_ids),
                speed_step_ms=float(TABLE_SPEED_STEP_MS), tables=tables,
                **arrays)

        if device_id is None or xp is np:
            return upload()
        with xp.cuda.Device(int(device_id)):
            return upload()


def _kernel_groups(layout) -> dict[str, Any]:
    return {"cell": layout.group_cell, "count": layout.group_turbines,
            "curve": layout.group_curve,
            "hub_height_m": layout.group_hub_height_m,
            "rotor_diameter_m": layout.group_rotor_diameter_m}


def _kernel_curves(tables) -> dict[str, Any]:
    return {"speeds_ms": tables.speeds_ms,
            "thrust_coefficient": tables.thrust_coefficient,
            "power_kw": tables.power_kw, "nval": tables.nval,
            "standing_thrust": tables.standing_thrust,
            "standing_thrust_cutout": tables.standing_thrust_cutout,
            "cut_in_ms": tables.cut_in_ms, "cut_out_ms": tables.cut_out_ms}


def _json_number(value) -> float:
    """A float32 word as the shortest decimal that reads back as it."""

    return float(float32_word(value))


def _cupy():
    """CuPy, imported only when a layout goes to the device (tests swap it
    for NumPy to check the upload on a CPU host)."""

    import cupy
    return cupy


#: The host arrays :meth:`WindFarmLayout.to_device` uploads, by name.
DEVICE_ARRAYS = (
    "turbine_count", "group_count", "rated_capacity_kw", "swept_area_m2",
    "rotor_top_m", "rotor_bottom_m", "cell_group_offset", "group_cell",
    "group_j", "group_i", "group_turbines", "group_curve",
    "group_hub_height_m", "group_rotor_diameter_m", "rated_kw", "speeds_ms",
    "group_thrust_coefficient", "group_power_coefficient", "group_power_w")


@dataclass(frozen=True)
class DeviceWindFarmLayout:
    """A :class:`WindFarmLayout`'s kernel inputs, resident on one device.

    The arrays named in :data:`DEVICE_ARRAYS`, under the host layout's
    names and with its words, plus ``tables`` (:class:`FitchTables` of
    device arrays: ``turbws``/``turbtc``/``turbpw`` and the per-curve
    header values a Fitch kernel's ``dragcof`` reads; see
    ``woof/core/kernels/windfarm_fitch.cu``).  Plain data: no curve names,
    turbine ids or host bookkeeping.
    """

    ny: int
    nx: int
    n_groups: int
    n_curves: int
    speed_step_ms: float
    tables: FitchTables
    turbine_count: Any
    group_count: Any
    rated_capacity_kw: Any
    swept_area_m2: Any
    rotor_top_m: Any
    rotor_bottom_m: Any
    cell_group_offset: Any
    group_cell: Any
    group_j: Any
    group_i: Any
    group_turbines: Any
    group_curve: Any
    group_hub_height_m: Any
    group_rotor_diameter_m: Any
    rated_kw: Any
    speeds_ms: Any
    group_thrust_coefficient: Any
    group_power_coefficient: Any
    group_power_w: Any

    def kernel_groups(self) -> dict[str, Any]:
        """The group arrays under the Fitch kernel's field names."""

        return _kernel_groups(self)

    def kernel_curves(self) -> dict[str, Any]:
        """The WRF tables under the Fitch kernel's field names."""

        return _kernel_curves(self.tables)

    def arrays(self) -> dict[str, Any]:
        """Every device array by name (tables as ``tables.<name>``)."""

        out = {name: getattr(self, name) for name in DEVICE_ARRAYS}
        out.update({f"tables.{name}": value
                    for name, value in self.tables.arrays().items()})
        return out

    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in self.arrays().values()))


# --------------------------------------------------------------------------
# reading


def _number(raw: Any, what: str) -> float | None:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, bool):
        raise WindFarmLayoutRefused(f"{what} is {raw!r}, not a number")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise WindFarmLayoutRefused(
            f"{what} is {raw!r}, not a number") from error
    if not math.isfinite(value):
        raise WindFarmLayoutRefused(f"{what} is {raw!r}, not finite")
    return value


def _text(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _pick(row: Mapping[str, Any], name: str) -> Any:
    for key in COLUMN_ALIASES[name]:
        if key in row and row[key] is not None and str(row[key]).strip():
            return row[key]
    return None


def turbine_from_row(row: Mapping[str, Any], index: int, *,
                     where: str = "turbine table") -> Turbine:
    """One table row (keys already lower-cased) -> :class:`Turbine`.

    ``index`` counts from 1 and names an id-less row.
    """

    turbine_id = _text(_pick(row, "turbine_id")) or f"t{index}"
    what = f"{where}: turbine {turbine_id!r}"
    lat = _number(_pick(row, "lat"), f"{what} lat")
    lon = _number(_pick(row, "lon"), f"{what} lon")
    if lat is None or lon is None:
        raise WindFarmLayoutRefused(f"{what} has no lat/lon")
    if not -90.0 <= lat <= 90.0:
        raise WindFarmLayoutRefused(f"{what} lat {lat} is outside [-90, 90]")
    if not -180.0 <= lon <= 360.0:
        raise WindFarmLayoutRefused(
            f"{what} lon {lon} is outside [-180, 360]")
    lon = ((lon + 180.0) % 360.0) - 180.0
    return Turbine(
        turbine_id=turbine_id, lat=lat, lon=lon,
        hub_height_m=_number(_pick(row, "hub_height_m"),
                             f"{what} hub_height_m"),
        rotor_diameter_m=_number(_pick(row, "rotor_diameter_m"),
                                 f"{what} rotor_diameter_m"),
        curve=_text(_pick(row, "curve")),
        manufacturer=_text(_pick(row, "manufacturer")),
        rated_kw=_number(_pick(row, "rated_kw"), f"{what} rated_kw"))


def _lowered(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k).strip().lower(): v for k, v in row.items()
            if k is not None}


def read_turbine_csv(path: str | Path) -> tuple[Turbine, ...]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = {str(k).strip().lower() for k in reader.fieldnames or ()}
        for need in ("lat", "lon"):
            if not header & set(COLUMN_ALIASES[need]):
                raise WindFarmLayoutRefused(
                    f"turbine table {path} has no {need} column (header: "
                    f"{', '.join(sorted(header)) or 'none'})")
        # A row of separators only (",,,," from a spreadsheet) is blank.
        return tuple(turbine_from_row(_lowered(row), n, where=str(path))
                     for n, row in enumerate(reader, start=1)
                     if any(str(v).strip() for v in row.values()
                            if v is not None))


def turbines_from_geojson(document: Mapping[str, Any], *,
                          where: str = "GeoJSON") -> tuple[Turbine, ...]:
    if document.get("type") != "FeatureCollection":
        raise WindFarmLayoutRefused(f"{where} is not a FeatureCollection")
    out = []
    for n, feature in enumerate(document.get("features") or [], start=1):
        if not isinstance(feature, Mapping) or not isinstance(
                feature.get("properties") or {}, Mapping):
            raise WindFarmLayoutRefused(
                f"{where}: feature {n} is not a GeoJSON Feature object")
        geometry = feature.get("geometry") or {}
        if not isinstance(geometry, Mapping) or \
                geometry.get("type") != "Point":
            raise WindFarmLayoutRefused(
                f"{where}: feature {n} is a "
                f"{geometry.get('type') if isinstance(geometry, Mapping) else geometry!r}"
                ", not a "
                "Point; one turbine is one point")
        coords = geometry.get("coordinates") or []
        if len(coords) < 2:
            raise WindFarmLayoutRefused(
                f"{where}: feature {n} has no [lon, lat]")
        row = _lowered(feature.get("properties") or {})
        if feature.get("id") is not None and _pick(row, "turbine_id") is None:
            row["turbine_id"] = feature["id"]
        row["lon"], row["lat"] = coords[0], coords[1]
        out.append(turbine_from_row(row, n, where=where))
    return tuple(out)


def turbines_from_sites(site_set) -> tuple[Turbine, ...]:
    """The ``turbine`` sites of a ``woof-energy.sites.v1`` SiteSet.

    A site names no model, so every one takes the caller's
    ``default_curve`` when the layout is built.
    """

    out = tuple(
        turbine_from_row({
            "turbine_id": site.site_id, "lat": site.lat, "lon": site.lon,
            "hub_height_m": site.hub_height_m,
            "rated_kw": (None if site.capacity_mw is None
                         else site.capacity_mw * 1000.0)},
            n, where="sites document")
        for n, site in enumerate(site_set.sites, start=1)
        if site.kind == "turbine")
    if not out:
        kinds = sorted({site.kind for site in site_set.sites})
        raise WindFarmLayoutRefused(
            "the sites document holds no turbine sites (kinds: "
            f"{', '.join(kinds) or 'none'})")
    return out


def read_turbines(path: str | Path) -> tuple[Turbine, ...]:
    """A CSV, a GeoJSON FeatureCollection, or a sites.v1 document."""

    path = Path(path)
    if not path.is_file():
        raise WindFarmLayoutRefused(f"turbine table {path} does not exist")
    if path.suffix.lower() in (".csv", ".txt"):
        turbines = read_turbine_csv(path)
    else:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WindFarmLayoutRefused(
                f"turbine table {path} is neither CSV (by suffix) nor JSON: "
                f"{error}") from error
        if not isinstance(document, dict):
            raise WindFarmLayoutRefused(
                f"turbine table {path} is not a JSON object")
        from woof.energy.contracts import SITES_SCHEMA, SiteSet
        if document.get("schema") == SITES_SCHEMA:
            turbines = turbines_from_sites(SiteSet.from_json(document))
        else:
            turbines = turbines_from_geojson(document, where=str(path))
    if not turbines:
        raise WindFarmLayoutRefused(f"turbine table {path} has no turbines")
    return turbines


# --------------------------------------------------------------------------
# curves


def resolve_curve(name: str, library: TurbineLibrary, *,
                  manufacturer: str | None = None,
                  rotor_diameter_m: float | None = None
                  ) -> TurbineCurve | None:
    """A library id, else an ``exact`` or confident ``fuzzy`` match, else
    None."""

    if name in library.curves:
        curve = library.curves[name]
        if manufacturer is not None:
            # The id path honours the manufacturer as the ladder does.
            ours = canonical_manufacturer(manufacturer,
                                          library.manufacturer_aliases)
            theirs = canonical_manufacturer(curve.manufacturer,
                                            library.manufacturer_aliases)
            if theirs is not None and ours != theirs:
                return None
        return curve
    hit = library.match(TurbineSpec(manufacturer=manufacturer, model=name,
                                    rotor_diameter_m=rotor_diameter_m))
    if hit.curve is None or hit.quality not in ACCEPTED_MATCHES:
        return None
    if hit.quality == "fuzzy":
        # Only a parse that names both rotor and rating, and the curve that
        # is exactly that machine: never a name-similarity hit, never a
        # neighbouring rating inside the ladder's 3 % tolerance.
        parsed = parse_model(name)
        if parsed.rotor_m is None or parsed.rated_kw is None:
            return None
        if abs(hit.curve.rotor_diameter_m - parsed.rotor_m) > \
                FUZZY_ROTOR_TOLERANCE_M or \
                abs(hit.curve.rated_kw / parsed.rated_kw - 1.0) > \
                FUZZY_RATING_MATCH:
            return None
    return hit.curve


def speed_axis() -> np.ndarray:
    """The axis tables' float32 speeds (every one exact in binary32)."""

    count = int(round(TABLE_MAX_SPEED_MS / TABLE_SPEED_STEP_MS)) + 1
    return (np.arange(count) * TABLE_SPEED_STEP_MS).astype(F)


# --------------------------------------------------------------------------
# placement


def domain_grid(config: str | Path, grid_id: int = 1):
    """The registered projected grid of one experiment-TOML domain."""

    from woof.experiment import load_experiment
    from woof.static.projection import grids_from_projection_config

    exp = load_experiment(config)
    ids = [int(domain.grid_id) for domain in exp.domains]
    if int(grid_id) not in ids:
        raise WindFarmLayoutRefused(
            f"{config} has no domain d{int(grid_id):02d} (domains: "
            + ", ".join(f"d{g:02d}" for g in ids) + ")")
    return grids_from_projection_config(exp)[ids.index(int(grid_id))]


def wrf_cells(projection: WrfProjection, lat32, lon32):
    """WRF's placement of float32 lat/lon: ``(x, y, j, i)``.

    ``x``/``y`` are ``latlon_to_ij``'s float32 ``ts_rx``/``ts_ry``
    (1-based); the 0-based cell is ``nint - 1``, and -1 where the
    projection gives no finite point.  The one placement both the woof
    table and WRF's own ``windturbines.txt`` are read through.
    """
    n = len(lat32)
    x = np.empty(n, dtype=F)
    y = np.empty(n, dtype=F)
    ii = np.empty(n, dtype=np.int64)
    jj = np.empty(n, dtype=np.int64)
    for k in range(n):
        x[k], y[k] = projection.latlon_to_ij(F(lat32[k]), F(lon32[k]))
        finite = np.isfinite(x[k]) and np.isfinite(y[k])
        ii[k] = nint(x[k]) - 1 if finite else -1
        jj[k] = nint(y[k]) - 1 if finite else -1
    return x, y, jj, ii


def _refuse_list(head: str, names: Sequence[str], remedy: str, why: str):
    shown = ", ".join(names[:12]) + (f" and {len(names) - 12} more"
                                     if len(names) > 12 else "")
    raise WindFarmLayoutRefused(layered(
        f"{head}: {shown}.\n  remedy: {remedy}", why))


def build_layout(turbines: Iterable[Turbine], grid, *, grid_id: int = 1,
                 default_curve: str | None = None,
                 library: TurbineLibrary | None = None,
                 standing_thrust=STANDING_THRUST_COEFFICIENT,
                 merge_tolerance_m: float | None = None) -> WindFarmLayout:
    """Place ``turbines`` on ``grid`` (a Lambert, Mercator or polar
    stereographic :class:`ProjectedGrid`, root or nest).

    ``standing_thrust`` is every WRF table's ``stc``, the thrust
    coefficient of a rotor below cut-in.  ``merge_tolerance_m`` (off by
    default) merges a cell's same-curve turbines whose hub and rotor lie
    within it into one averaged group, wherever they sit in the table: it
    saves memory on dense tables but is NOT WRF-exact.
    """

    turbines = tuple(turbines)
    if not turbines:
        raise WindFarmLayoutRefused("no turbines to place")
    if merge_tolerance_m is not None and not (
            math.isfinite(float(merge_tolerance_m))
            and float(merge_tolerance_m) >= 0.0):
        raise WindFarmLayoutRefused(
            f"merge_tolerance_m must be a finite number of metres >= 0, got "
            f"{merge_tolerance_m!r}")
    library = library if library is not None else load_turbine_library()
    ids = [t.turbine_id for t in turbines]
    duplicated = sorted(i for i, k in Counter(ids).items() if k > 1)
    if duplicated:
        _refuse_list("turbine ids appear more than once", duplicated,
                     "give every turbine its own turbine_id",
                     "Every refusal and every per-turbine record names a "
                     "turbine by id; two with one id cannot be told apart.")

    # -- curves ------------------------------------------------------------
    curves: list[TurbineCurve] = []
    index_of: dict[str, int] = {}
    turbine_curve = np.empty(len(turbines), dtype=np.int32)
    missing, unknown, rotor_bad, hub_bad, rating_bad = [], [], [], [], []
    rotors = np.empty(len(turbines), dtype=np.float64)
    hubs = np.empty(len(turbines), dtype=np.float64)
    for n, turbine in enumerate(turbines):
        name = turbine.curve or default_curve
        if name is None:
            missing.append(turbine.turbine_id)
            continue
        curve = resolve_curve(name, library,
                              manufacturer=turbine.manufacturer,
                              rotor_diameter_m=turbine.rotor_diameter_m)
        if curve is None:
            unknown.append(f"{turbine.turbine_id} ({name!r})")
            continue
        if curve.curve_id not in index_of:
            index_of[curve.curve_id] = len(curves)
            curves.append(curve)
        turbine_curve[n] = index_of[curve.curve_id]
        if turbine.rated_kw is not None and (
                turbine.rated_kw <= 0.0 or abs(turbine.rated_kw
                                               / curve.rated_kw - 1.0)
                > RATING_TOLERANCE):
            rating_bad.append(f"{turbine.turbine_id} ({turbine.rated_kw:g} "
                              f"kW against {curve.curve_id}'s "
                              f"{curve.rated_kw:g} kW)")
        rotor = turbine.rotor_diameter_m
        if rotor is None:
            rotor = curve.rotor_diameter_m
        elif rotor <= 0.0 or abs(rotor / curve.rotor_diameter_m - 1.0) > \
                SIZE_ROTOR_TOLERANCE:
            rotor_bad.append(f"{turbine.turbine_id} ({rotor:g} m against "
                             f"{curve.curve_id}'s {curve.rotor_diameter_m:g}"
                             " m)")
        rotors[n] = rotor
        hub = turbine.hub_height_m
        if hub is None or hub - 0.5 * rotor < 0.0:
            hub_bad.append(
                f"{turbine.turbine_id} (" + ("none" if hub is None else
                                             f"{hub:g} m, rotor bottom "
                                             f"{hub - 0.5 * rotor:g} m")
                + ")")
        hubs[n] = np.nan if hub is None else hub
    if missing:
        _refuse_list(
            "turbines with no curve", missing,
            "fill the table's curve column with a turbine_library.json id "
            "or model, or pass a default curve (--curve)",
            "A turbine's power and thrust come from its curve; woof does "
            "not stand a generic curve in for a machine it was not told.")
    if unknown:
        _refuse_list(
            "turbine curves not in the turbine library", unknown,
            "use an id from woof/data/energy/turbine_library.json or a "
            "model name it matches exactly (with its rating: V112 3MW, "
            "not V112), and a manufacturer that agrees with the curve's",
            "Only the library's exact and confident fuzzy matches name one "
            "real machine; a bare rotor parse is ambiguous, and the size "
            "and generic rungs are parametric stand-ins.")
    if rating_bad:
        _refuse_list(
            "turbine ratings that disagree with the turbine's curve",
            rating_bad,
            "name the curve of the machine with that rating (--curve for a "
            "sites document), or correct the rating",
            f"Capacity and the power table are the curve's; a rating more "
            f"than {RATING_TOLERANCE:.0%} off it is a different machine.")
    if rotor_bad:
        _refuse_list(
            "rotor diameters that disagree with the turbine's curve",
            rotor_bad,
            "fix the rotor_diameter_m column, leave it blank to take the "
            "curve's, or name the curve of the machine with that rotor",
            f"The thrust table is the curve's; a rotor more than "
            f"{SIZE_ROTOR_TOLERANCE:.0%} off it is a different machine.")
    if hub_bad:
        _refuse_list(
            "turbines with no usable hub height", hub_bad,
            "give each turbine its hub_height_m above ground, at least "
            "half its rotor diameter (rotor bottom = hub - rotor/2 >= 0 m)",
            "The hub height places the rotor in the column; none is "
            "invented, and a rotor bottom below 0 m AGL would put the "
            "blade tip underground.")

    # -- cells: WRF's own placement ---------------------------------------
    nx, ny = int(grid.e_we) - 1, int(grid.e_sn) - 1
    try:
        projection = WrfProjection.for_grid(grid)
    except WrfProjectionRefused as error:
        raise WindFarmLayoutRefused(layered(
            f"d{int(grid_id):02d}: {error}.\n  remedy: lay the farm out on "
            "a Lambert, Mercator or polar stereographic domain",
            "Turbines are placed by WRF's own real(4) map_set and "
            "latlon_to_ij so that a WRF run of the same farm puts every "
            "turbine in the same cell; WRF has no wind-farm placement for "
            "any other projection.")) from error
    n_turbines = len(turbines)
    lat32 = np.array([F(t.lat) for t in turbines], dtype=F)
    lon32 = np.array([F(t.lon) for t in turbines], dtype=F)
    x, y, jj, ii = wrf_cells(projection, lat32, lon32)
    outside = [ids[n] for n in range(n_turbines)
               if not (0 <= ii[n] < nx and 0 <= jj[n] < ny)]
    if outside:
        _refuse_list(
            f"turbines outside d{int(grid_id):02d} ({nx} x {ny} cells of "
            f"{grid.dx:g} m)", outside,
            "drop them from the table or use a domain that contains them",
            "A turbine outside the mass grid has no cell to act on.")
    edge = [ids[n] for n in range(n_turbines)
            if ii[n] in (0, nx - 1) or jj[n] in (0, ny - 1)]
    if edge:
        _refuse_list(
            f"turbines in d{int(grid_id):02d}'s outermost row or column",
            edge, "use a domain that holds them at least one cell inside "
                  "its edge",
            "The outermost cells are boundary-forced, not modelled; the "
            "hub-wind site sampler refuses them on the same rule.")

    # -- groups ------------------------------------------------------------
    # WRF-exact (the default): a cell's turbines in table order -- the order
    # WRF reads windturbines.txt and a Fitch kernel adds them -- split into
    # runs of consecutive turbines with the same curve and bit-identical
    # float32 hub height and rotor.  A cell listing A, B, A is three groups.
    # merge_tolerance_m (opt-in, NOT WRF-exact: it reorders and averages
    # what WRF adds turbine by turbine) instead puts a turbine in the first
    # earlier group of its cell and curve whose first turbine is within the
    # tolerance in hub and rotor.
    flat = jj * nx + ii
    hub32 = hubs.astype(F)
    rotor32 = rotors.astype(F)
    members: list[list[int]] = []
    last_in_cell: dict[int, int] = {}
    open_groups: dict[tuple[int, int], list[int]] = {}
    for n in range(n_turbines):
        cell = int(flat[n])
        if merge_tolerance_m is None:
            g = last_in_cell.get(cell)
            if g is not None:
                first = members[g][0]
                if (turbine_curve[n] == turbine_curve[first]
                        and hub32[n].view(np.uint32)
                        == hub32[first].view(np.uint32)
                        and rotor32[n].view(np.uint32)
                        == rotor32[first].view(np.uint32)):
                    members[g].append(n)
                    continue
            last_in_cell[cell] = len(members)
            members.append([n])
            continue
        key = (cell, int(turbine_curve[n]))
        for g in open_groups.setdefault(key, []):
            first = members[g][0]
            if abs(hubs[n] - hubs[first]) <= merge_tolerance_m and \
                    abs(rotors[n] - rotors[first]) <= merge_tolerance_m:
                members[g].append(n)
                break
        else:
            open_groups[key].append(len(members))
            members.append([n])
    # CSR order: by cell, then each cell's groups in the order they opened
    # (their first turbine's table position)
    members.sort(key=lambda m: (int(flat[m[0]]), m[0]))
    turbine_group = np.empty(n_turbines, dtype=np.int32)
    for g, m in enumerate(members):
        turbine_group[m] = g
    group_cell = np.array([flat[m[0]] for m in members], dtype=np.int32)
    group_curve = np.array([turbine_curve[m[0]] for m in members],
                           dtype=np.int32)
    group_turbines = np.array([len(m) for m in members], dtype=np.int32)
    if merge_tolerance_m is None:
        group_hub = np.array([hub32[m[0]] for m in members], dtype=F)
        group_rotor = np.array([rotor32[m[0]] for m in members], dtype=F)
    else:
        group_hub = np.array([hubs[m].mean() for m in members]).astype(F)
        group_rotor = np.array([rotors[m].mean() for m in members]).astype(F)
    rated = np.array([c.rated_kw for c in curves]).astype(F)

    cells = ny * nx
    per_cell = np.bincount(group_cell, minlength=cells)
    offset = np.zeros(cells + 1, dtype=np.int32)
    np.cumsum(per_cell, out=offset[1:])
    count = np.bincount(flat, minlength=cells).astype(np.int32)
    # binary64 sums in a fixed order, rounded once: ratings in table order,
    # swept areas (WRF's own area word per group) in group order
    capacity = np.zeros(cells)
    for n in range(n_turbines):
        capacity[flat[n]] += float(rated[turbine_curve[n]])
    from woof.core.noahmp_libm import powf_array
    area = F(PICONST / F(4.0)) * powf_array(group_rotor, F(2.0))
    swept = np.zeros(cells)
    top = np.full(cells, np.nan, dtype=F)
    bottom = np.full(cells, np.nan, dtype=F)
    upper = (group_hub + group_rotor / F(2.0)).astype(F)
    lower = (group_hub - group_rotor / F(2.0)).astype(F)
    for g in range(len(members)):
        c = group_cell[g]
        swept[c] += float(group_turbines[g]) * float(area[g])
        top[c] = upper[g] if np.isnan(top[c]) else max(top[c], upper[g])
        bottom[c] = lower[g] if np.isnan(bottom[c]) else min(bottom[c],
                                                            lower[g])

    def plane(values, dtype):
        return _readonly(np.asarray(values).astype(dtype).reshape(ny, nx))

    # -- tables --------------------------------------------------------------
    try:
        tables = FitchTables.from_curves(curves,
                                         standing_thrust=standing_thrust)
    except FitchTableRefused as error:
        raise WindFarmLayoutRefused(str(error)) from error
    # the axis tables, per group: WRF's dragcof and power1 at the group's
    # own curve and rotor (powcof scales with 1 / area)
    speeds = speed_axis()
    n_groups = len(members)
    axis = dragcof_cpu(tables, np.repeat(group_curve, speeds.size),
                       np.tile(speeds, n_groups),
                       np.repeat(group_rotor, speeds.size))
    shape = (n_groups, speeds.size)

    return WindFarmLayout(
        grid_id=int(grid_id), ny=ny, nx=nx, dx_m=float(grid.dx),
        projection=projection,
        turbine_count=plane(count, np.int32),
        group_count=plane(per_cell, np.int32),
        rated_capacity_kw=plane(capacity, F),
        swept_area_m2=plane(swept, F),
        rotor_top_m=plane(top, F), rotor_bottom_m=plane(bottom, F),
        cell_group_offset=_readonly(offset),
        group_cell=_readonly(group_cell),
        group_j=_readonly((group_cell // nx).astype(np.int32)),
        group_i=_readonly((group_cell % nx).astype(np.int32)),
        group_turbines=_readonly(group_turbines),
        group_curve=_readonly(group_curve),
        group_hub_height_m=_readonly(group_hub),
        group_rotor_diameter_m=_readonly(group_rotor),
        curve_ids=tuple(c.curve_id for c in curves),
        rated_kw=_readonly(rated), tables=tables,
        speeds_ms=_readonly(speeds),
        group_thrust_coefficient=_readonly(axis["thrcof"].reshape(shape)),
        group_power_coefficient=_readonly(axis["powcof"].reshape(shape)),
        group_power_w=_readonly(axis["power1"].reshape(shape)),
        turbine_ids=tuple(ids),
        turbine_lat=_readonly(lat32), turbine_lon=_readonly(lon32),
        turbine_x=_readonly(x), turbine_y=_readonly(y),
        turbine_j=_readonly(jj.astype(np.int32)),
        turbine_i=_readonly(ii.astype(np.int32)),
        turbine_curve=_readonly(turbine_curve),
        turbine_group=_readonly(turbine_group),
        library_sha256=library.sha256)


def layout_for_config(config: str | Path, turbines: str | Path, *,
                      grid_id: int = 1, default_curve: str | None = None,
                      library_path: str | Path | None = None
                      ) -> WindFarmLayout:
    """Read a turbine file and place it on one domain of an experiment."""

    return build_layout(
        read_turbines(turbines), domain_grid(config, grid_id),
        grid_id=grid_id, default_curve=default_curve,
        library=load_turbine_library(library_path))


__all__ = [
    "LAYOUT_SCHEMA", "TABLE_SPEED_STEP_MS", "TABLE_MAX_SPEED_MS",
    "GROUP_TOLERANCE_M", "ACCEPTED_MATCHES", "FUZZY_RATING_MATCH",
    "RATING_TOLERANCE",
    "WindFarmLayoutRefused",
    "Turbine", "WindFarmLayout", "DeviceWindFarmLayout", "DEVICE_ARRAYS",
    "turbine_from_row",
    "read_turbine_csv",
    "turbines_from_geojson", "turbines_from_sites", "read_turbines",
    "resolve_curve", "speed_axis", "domain_grid", "build_layout",
    "layout_for_config", "wrf_cells",
]
