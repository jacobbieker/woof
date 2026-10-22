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
``rated_kw``
    Optional; when given it must be within ``RATING_TOLERANCE`` of the
    curve's rating, or the turbine is refused.  A blank curve takes the
    caller's ``default_curve``; with none, the turbine is refused.  The
    library's ``size``/``generic`` rungs are never used here: a parametric
    stand-in is not the machine's curve.
``turbine_id``
    Optional (``id``/``name``); default ``t<row>`` counted from 1.

A ``woof-energy.sites.v1`` document (:mod:`woof.energy.sites`) is
accepted too: its ``turbine`` sites, with their own hub heights and the
``default_curve`` for every one of them (a site carries a rating, not a
model), and each site's ``capacity_mw`` checked against that curve's
rating as ``rated_kw`` is.

PLACEMENT.  Each turbine goes through the domain's own projection
(``latlon_to_ij``: :mod:`woof.static.lambert` for Lambert, the WPS
transcription geogrid places cells with), and lands in the mass cell
whose footprint holds it: 1-based mass point ``(i, j)`` sits at projection
coordinate ``(i, j)``, so the 0-based cell is ``floor(x + 0.5) - 1``.  A
turbine outside the mass grid, or in its outermost row or column (the
rule :func:`woof.core.hub_wind_diag.locate_sites` applies to hub sites:
those cells are boundary-forced, not modelled), is refused by name.

ONE MACHINE PER CELL.  The per-cell arrays hold one hub height, one rotor
and one curve per cell, so a cell's turbines must share a curve and agree
on hub height and rotor within ``CELL_SPREAD_TOLERANCE_M``; the cell
value is their mean.  A mixed cell is refused by name (move to a finer
domain, or describe the cell with one machine); the per-turbine records
keep every turbine's own values either way.

THE TABLES.  One row per distinct curve, on the shared speed axis
``0 .. TABLE_MAX_SPEED_MS`` every ``TABLE_SPEED_STEP_MS``:
``power_kw = rated_kw * TurbineCurve.fraction(v)`` and
``thrust_coefficient = TurbineCurve.thrust_coefficient(v)``, both at the
library's reference density 1.225 kg m-3.  Cut-out is NOT applied (it
carries hysteresis; :mod:`woof.energy.power_farm` owns it): each curve's
``cut_out_ms`` is carried beside its table for the consumer.
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

LAYOUT_SCHEMA = "woof-windfarm.layout-summary.v1"

#: The shared speed axis of every power/thrust table, m s-1.
TABLE_SPEED_STEP_MS = 0.5
TABLE_MAX_SPEED_MS = 35.0

#: Turbines in one cell must agree on hub height and rotor within this.
CELL_SPREAD_TOLERANCE_M = 1.0

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
    """Turbines on one domain's mass grid: per-cell arrays plus tables.

    Per cell, ``(ny, nx)`` in geo_em ``[j, i]`` order: ``turbine_count``
    (int32, 0 where empty), ``hub_height_m`` and ``rotor_diameter_m``
    (float64, NaN where empty) and ``curve_index`` (int32 into
    ``curve_ids``, -1 where empty).  Per curve: ``rated_kw``,
    ``cut_out_ms`` and the ``power_kw``/``thrust_coefficient`` rows on
    ``speeds_ms``.  Per turbine, in table order: ids, the 0-based cell
    ``(turbine_j, turbine_i)``, the fractional 1-based projection
    coordinate ``(turbine_x, turbine_y)``, and the curve index.  Every
    array is read-only.
    """

    grid_id: int
    ny: int
    nx: int
    dx_m: float
    turbine_count: np.ndarray
    hub_height_m: np.ndarray
    rotor_diameter_m: np.ndarray
    curve_index: np.ndarray
    curve_ids: tuple[str, ...]
    rated_kw: np.ndarray
    cut_out_ms: np.ndarray
    speeds_ms: np.ndarray
    power_kw: np.ndarray
    thrust_coefficient: np.ndarray
    turbine_ids: tuple[str, ...]
    turbine_j: np.ndarray
    turbine_i: np.ndarray
    turbine_x: np.ndarray
    turbine_y: np.ndarray
    turbine_curve: np.ndarray
    library_sha256: str

    @property
    def n_turbines(self) -> int:
        return len(self.turbine_ids)

    @property
    def cells_occupied(self) -> int:
        return int(np.count_nonzero(self.turbine_count))

    @property
    def rated_capacity_mw(self) -> float:
        return float(self.rated_kw[self.turbine_curve].sum()) / 1000.0

    def cells(self) -> list[dict[str, Any]]:
        """One record per occupied cell, row-major (``j`` then ``i``)."""

        out = []
        for j, i in zip(*np.nonzero(self.turbine_count)):
            k = int(self.curve_index[j, i])
            count = int(self.turbine_count[j, i])
            out.append({
                "j": int(j), "i": int(i), "turbines": count,
                "curve": self.curve_ids[k],
                "hub_height_m": float(self.hub_height_m[j, i]),
                "rotor_diameter_m": float(self.rotor_diameter_m[j, i]),
                "capacity_mw": count * float(self.rated_kw[k]) / 1000.0,
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
            "cells_occupied": self.cells_occupied,
            "max_turbines_per_cell": int(self.turbine_count.max()),
            "rated_capacity_mw": round(self.rated_capacity_mw, 6),
            "curves": [{"curve": cid, "rated_kw": float(self.rated_kw[k]),
                        "cut_out_ms": float(self.cut_out_ms[k]),
                        "turbines": int(per_curve[k])}
                       for k, cid in enumerate(self.curve_ids)],
            "cells": self.cells(),
            "turbine_library_sha256": self.library_sha256,
        }


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
        geometry = (feature or {}).get("geometry") or {}
        if geometry.get("type") != "Point":
            raise WindFarmLayoutRefused(
                f"{where}: feature {n} is a {geometry.get('type')!r}, not a "
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
    count = int(round(TABLE_MAX_SPEED_MS / TABLE_SPEED_STEP_MS)) + 1
    return np.round(np.linspace(0.0, TABLE_MAX_SPEED_MS, count), 6)


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


def _refuse_list(head: str, names: Sequence[str], remedy: str, why: str):
    shown = ", ".join(names[:12]) + (f" and {len(names) - 12} more"
                                     if len(names) > 12 else "")
    raise WindFarmLayoutRefused(layered(
        f"{head}: {shown}.\n  remedy: {remedy}", why))


def build_layout(turbines: Iterable[Turbine], grid, *, grid_id: int = 1,
                 default_curve: str | None = None,
                 library: TurbineLibrary | None = None) -> WindFarmLayout:
    """Place ``turbines`` on ``grid`` (a :class:`ProjectedGrid`)."""

    turbines = tuple(turbines)
    if not turbines:
        raise WindFarmLayoutRefused("no turbines to place")
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
        if hub is None or hub <= 0.5 * rotor:
            hub_bad.append(f"{turbine.turbine_id} ("
                           + ("none" if hub is None else f"{hub:g} m") + ")")
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
            "give each turbine its hub_height_m above ground, more than "
            "half its rotor diameter",
            "The hub height places the rotor in the column; none is "
            "invented, and a hub below half the rotor would put the blade "
            "tip underground.")

    # -- cells -------------------------------------------------------------
    nx, ny = int(grid.e_we) - 1, int(grid.e_sn) - 1
    lat = np.array([t.lat for t in turbines], dtype=np.float64)
    lon = np.array([t.lon for t in turbines], dtype=np.float64)
    x, y = (np.asarray(a, dtype=np.float64)
            for a in grid.latlon_to_ij(lat, lon))
    finite = np.isfinite(x) & np.isfinite(y)
    ii = np.where(finite, np.floor(np.where(finite, x, 0.0) + 0.5), -1.0
                  ).astype(np.int64) - 1
    jj = np.where(finite, np.floor(np.where(finite, y, 0.0) + 0.5), -1.0
                  ).astype(np.int64) - 1
    outside = [ids[n] for n in range(len(turbines))
               if not (0 <= ii[n] < nx and 0 <= jj[n] < ny)]
    if outside:
        _refuse_list(
            f"turbines outside d{int(grid_id):02d} ({nx} x {ny} cells of "
            f"{grid.dx:g} m)", outside,
            "drop them from the table or use a domain that contains them",
            "A turbine outside the mass grid has no cell to act on.")
    edge = [ids[n] for n in range(len(turbines))
            if ii[n] in (0, nx - 1) or jj[n] in (0, ny - 1)]
    if edge:
        _refuse_list(
            f"turbines in d{int(grid_id):02d}'s outermost row or column",
            edge, "use a domain that holds them at least one cell inside "
                  "its edge",
            "The outermost cells are boundary-forced, not modelled; the "
            "hub-wind site sampler refuses them on the same rule.")

    count = np.zeros((ny, nx), dtype=np.int32)
    hub_cell = np.full((ny, nx), np.nan)
    rotor_cell = np.full((ny, nx), np.nan)
    curve_cell = np.full((ny, nx), -1, dtype=np.int32)
    by_cell: dict[tuple[int, int], list[int]] = {}
    for n in range(len(turbines)):
        by_cell.setdefault((int(jj[n]), int(ii[n])), []).append(n)
    mixed = []
    for (j, i), members in sorted(by_cell.items()):
        kinds = {int(turbine_curve[n]) for n in members}
        hub_spread = float(np.ptp(hubs[members]))
        rotor_spread = float(np.ptp(rotors[members]))
        if len(kinds) > 1 or hub_spread > CELL_SPREAD_TOLERANCE_M or \
                rotor_spread > CELL_SPREAD_TOLERANCE_M:
            names = sorted({curves[k].curve_id for k in kinds})
            mixed.append(f"cell (j={j}, i={i}) [{', '.join(ids[n] for n in members)}"
                         f"; curves {', '.join(names)}; hub spread "
                         f"{hub_spread:g} m; rotor spread {rotor_spread:g} m]")
            continue
        count[j, i] = len(members)
        hub_cell[j, i] = float(hubs[members].mean())
        rotor_cell[j, i] = float(rotors[members].mean())
        curve_cell[j, i] = kinds.pop()
    if mixed:
        _refuse_list(
            "cells holding more than one machine", mixed,
            "use a finer domain that separates them, or describe each "
            "cell with one curve, hub height and rotor",
            "Each cell carries one hub height, rotor and curve; turbines "
            f"must share the curve and agree within "
            f"{CELL_SPREAD_TOLERANCE_M:g} m, or the cell would describe "
            "none of them.")

    speeds = speed_axis()
    return WindFarmLayout(
        grid_id=int(grid_id), ny=ny, nx=nx, dx_m=float(grid.dx),
        turbine_count=_readonly(count), hub_height_m=_readonly(hub_cell),
        rotor_diameter_m=_readonly(rotor_cell),
        curve_index=_readonly(curve_cell),
        curve_ids=tuple(c.curve_id for c in curves),
        rated_kw=_readonly(np.array([c.rated_kw for c in curves])),
        cut_out_ms=_readonly(np.array([c.cut_out_ms for c in curves])),
        speeds_ms=_readonly(speeds),
        power_kw=_readonly(np.array([c.fraction(speeds) * c.rated_kw
                                     for c in curves])),
        thrust_coefficient=_readonly(np.array(
            [c.thrust_coefficient(speeds) for c in curves])),
        turbine_ids=tuple(ids),
        turbine_j=_readonly(jj.astype(np.int32)),
        turbine_i=_readonly(ii.astype(np.int32)),
        turbine_x=_readonly(x.copy()), turbine_y=_readonly(y.copy()),
        turbine_curve=_readonly(turbine_curve),
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
    "CELL_SPREAD_TOLERANCE_M", "ACCEPTED_MATCHES", "FUZZY_RATING_MATCH",
    "RATING_TOLERANCE",
    "WindFarmLayoutRefused",
    "Turbine", "WindFarmLayout", "turbine_from_row", "read_turbine_csv",
    "turbines_from_geojson", "turbines_from_sites", "read_turbines",
    "resolve_curve", "speed_axis", "domain_grid", "build_layout",
    "layout_for_config",
]
