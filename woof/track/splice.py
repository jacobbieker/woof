"""``woof track splice``: one atmosphere from a stack of grids.

A fine grid that covers a small footprint and spends its levels near the
ground (``woof downscale --child-levels``, or a live nest) and the coarse,
full-depth grid it was forced from are two readings of one atmosphere.
This module joins any number of them, coarsest first (a "tier" each):

THE PRECEDENCE RULE.  A point is answered by the finest tier that covers
it -- laterally, beyond that tier's edge zone (its boundary rows and
relaxation zone, never used), and vertically, below that tier's cutoff
height -- and falls back to the next coarser tier outside its footprint
or above its cutoff, down to the coarsest tier, which is full depth.
Each join can be soft: a tier fades in linearly across
``lateral_blend_cells`` cells inside its edge zone and across a vertical
layer ``blend_m`` deep centred on its cutoff; the two shares multiply.

Two products:

* **a path sample** -- one value of each field at every fix of a 3-D path
  (time, lat, lon, height; :mod:`woof.track.ingest` reads it), with every
  tier's own value and share beside the spliced one, and a continuity
  statement for each join: fine minus coarse at the cutoff height (the
  vertical join), fine minus coarse at fixes in the footprint's edge band
  (the lateral join), and the largest step of the spliced value where the
  answering tier changes along the path;
* **a merged field set** -- every finer tier's history frames with ``U``,
  ``V``, ``W``, ``T``, ``P`` and ``QVAPOR`` replaced, outside the tier's
  share, by the coarser tiers interpolated to its own grid points and
  heights, plus ``SPLICE_WEIGHT`` (its share, 0..1, at mass points).  The
  frames keep wrfout's layout, so the existing Icechunk writer
  (:func:`woof.energy.icechunk_store.write_gridded_segment`) stores each
  as one more domain group.

Heights are geopotential metres ((PH + PHB) / g), as everywhere in woof's
history; a path's GPS heights are metres above mean sea level, and the
two differ by about 100 m at 25 km (:mod:`woof.vertical_bands` makes the
same choice and says so).  A path whose heights are above ground stands
on the coarsest tier's terrain (``HGT``).

Interpolation, deliberately plain and stated: horizontally bilinear in
the grid's own mass (or staggered) index space, vertically linear in
height within each of the four neighbouring columns, then linear in time
between the two bracketing history frames.  Below a column's lowest level
the lowest value is held (the sample says so); above its top the sample is
missing.  Winds in path samples are earth-relative (``SINALPHA`` /
``COSALPHA``); the merged field set keeps wrfout's grid-relative winds and
is refused when two grids are not on one projection, where the
grid-relative components would not mean the same thing.

Nothing here runs a model or reads a GPU: it is a host-side reader of
finished history, the cheap one-way baseline for fine grids that do not
need to resolve the stratosphere or the whole domain.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from woof.explain import layered

SPLICE_SCHEMA = "woof.track-splice.v1"
#: Standard gravity WRF divides geopotential by (``module_model_constants``).
GRAVITY = 9.81
#: wrfout's potential-temperature offset (``T`` is theta - T0).
T0 = 300.0
#: The fields a path sample reports, in column order.
SAMPLE_FIELDS = ("u", "v", "w", "theta", "pressure", "qvapor")
#: The wrfout variables the merged field set replaces above the splice.
MERGED_VARIABLES = ("U", "V", "W", "T", "P", "QVAPOR")
#: The variable the merged field set adds: the child's share at mass points.
WEIGHT_VARIABLE = "SPLICE_WEIGHT"
#: Elements one vectorized vertical interpolation pass may hold.
_CHUNK_ELEMENTS = 8_000_000

_FRAME_NAME = re.compile(
    r"^wrfout_d(\d+)_(\d{4})-(\d{2})-(\d{2})_(\d{2})[_:](\d{2})[_:](\d{2})")


class SpliceRefused(ValueError):
    """A splice that cannot be made as asked."""


def _netcdf4():
    from woof.io.netcdf_serialization import netCDF4

    return netCDF4


# --------------------------------------------------------------------------
# the vertical splice


def child_weight(height, splice_height_m: float, blend_m: float = 0.0):
    """The child's share at ``height`` (geopotential m): 1 below the
    splice, 0 above it, linear across ``blend_m`` centred on it.

    With no blend layer the child owns every height strictly below the
    splice and the parent owns the splice height and everything above.
    """

    z = np.asarray(height, dtype=np.float64)
    splice = float(splice_height_m)
    blend = float(blend_m)
    if blend < 0.0 or not math.isfinite(blend):
        raise SpliceRefused(f"blend {blend_m!r} m must be finite and >= 0")
    if blend == 0.0:
        return np.where(z < splice, 1.0, 0.0)
    return np.clip((splice + 0.5 * blend - z) / blend, 0.0, 1.0)


# --------------------------------------------------------------------------
# one domain's history


def _frame_time(nc, path: Path) -> np.datetime64:
    netCDF4 = _netcdf4()
    if "Times" not in nc.variables:
        raise SpliceRefused(f"{path.name} has no Times variable")
    raw = np.asarray(nc.variables["Times"][:])
    if raw.ndim == 1:
        raw = raw[None, :]
    stamps = [str(s).strip() for s in np.atleast_1d(netCDF4.chartostring(raw))]
    if len(stamps) != 1:
        raise SpliceRefused(
            f"{path.name} holds {len(stamps)} times; splice reads one "
            "history frame per file")
    return np.datetime64(stamps[0].replace("_", "T"), "s")


def discover_frames(sources: Sequence[str | Path], domain: int | None,
                    *, what: str) -> list[Path]:
    """The wrfout files ``sources`` names: files as given, directories
    searched for ``wrfout_dNN_*`` of ``domain`` (required when a
    directory holds more than one domain)."""

    files: list[Path] = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            found: dict[int, list[Path]] = {}
            for entry in sorted(path.iterdir()):
                match = _FRAME_NAME.match(entry.name)
                if match and entry.is_file():
                    found.setdefault(int(match.group(1)), []).append(entry)
            if not found:
                raise SpliceRefused(f"{what}: {path} holds no wrfout files")
            if domain is None:
                if len(found) > 1:
                    raise SpliceRefused(layered(
                        f"{what}: {path} holds domains "
                        f"{sorted(found)}; name one with --{what}-domain",
                        "A directory of a nested run holds every domain's "
                        "history; splice reads one domain per side."))
                files.extend(next(iter(found.values())))
            else:
                if domain not in found:
                    raise SpliceRefused(
                        f"{what}: {path} holds no d{domain:02d} history "
                        f"(it holds {sorted(found)})")
                files.extend(found[domain])
        elif path.is_file():
            match = _FRAME_NAME.match(path.name)
            if (domain is not None and match is not None
                    and int(match.group(1)) != domain):
                raise SpliceRefused(
                    f"{what}: {path.name} is d{int(match.group(1)):02d}, "
                    f"not the d{domain:02d} asked for")
            files.append(path)
        else:
            raise SpliceRefused(f"{what}: {path} does not exist")
    if not files:
        raise SpliceRefused(f"{what}: no history frames")
    return files


def _projection_grid(nc):
    """A float64 projection object for the domain whose wrfout ``nc`` is,
    and the 0-based mass-index offset of its corner."""

    code = int(nc.getncattr("MAP_PROJ"))
    truelat1 = float(nc.getncattr("TRUELAT1"))
    truelat2 = float(nc.getncattr("TRUELAT2"))
    stand_lon = float(nc.getncattr("STAND_LON"))
    dx = float(nc.getncattr("DX"))
    dy = float(nc.getncattr("DY"))
    xlat = np.asarray(nc.variables["XLAT"][0], dtype=np.float64)
    xlong = np.asarray(nc.variables["XLONG"][0], dtype=np.float64)
    ny, nx = xlat.shape
    args = (float(xlat[0, 0]), float(xlong[0, 0]), truelat1, truelat2,
            stand_lon, dx, dy, nx + 1, ny + 1)
    if code == 1:
        from woof.static.lambert import LambertGrid as Grid
    elif code == 2:
        from woof.static.projection import PolarStereoGrid as Grid
    elif code == 3:
        from woof.static.projection import MercatorGrid as Grid
    else:
        raise SpliceRefused(
            f"MAP_PROJ {code}: splice reads Lambert conformal, polar "
            "stereographic and Mercator domains only")
    grid = Grid(*args, known_x=1.0, known_y=1.0)
    i0, j0 = grid.latlon_to_ij(np.array([xlat[0, 0]]), np.array([xlong[0, 0]]))
    i1, j1 = grid.latlon_to_ij(np.array([xlat[-1, -1]]),
                               np.array([xlong[-1, -1]]))
    # The far corner must land nx-1, ny-1 cells away, or the attributes
    # do not describe this grid.
    if (abs(float(i1[0] - i0[0]) - (nx - 1)) > 0.05
            or abs(float(j1[0] - j0[0]) - (ny - 1)) > 0.05):
        raise SpliceRefused(
            "the domain's projection attributes do not reproduce its "
            "XLAT/XLONG corners; splice cannot locate points in it")
    return grid, (float(i0[0]), float(j0[0])), {
        "MAP_PROJ": code, "TRUELAT1": truelat1, "TRUELAT2": truelat2,
        "STAND_LON": stand_lon}


def _stagger_mean(field: np.ndarray, axis: int) -> np.ndarray:
    """Mass-point values averaged onto the faces along ``axis`` (the two
    edge faces take their one neighbour)."""

    field = np.asarray(field, dtype=np.float64)
    lo = np.take(field, [0], axis=axis)
    hi = np.take(field, [-1], axis=axis)
    inner = 0.5 * (np.take(field, range(0, field.shape[axis] - 1), axis=axis)
                   + np.take(field, range(1, field.shape[axis]), axis=axis))
    return np.concatenate([lo, inner, hi], axis=axis)


class DomainHistory:
    """One domain's history frames, read lazily, two frames cached."""

    def __init__(self, paths: Sequence[Path], *, label: str):
        netCDF4 = _netcdf4()
        self.label = label
        stamped = []
        for path in paths:
            with netCDF4.Dataset(path) as nc:
                stamped.append((_frame_time(nc, path), Path(path)))
        stamped.sort(key=lambda item: item[0])
        times = [t for t, _ in stamped]
        if len(set(times)) != len(times):
            raise SpliceRefused(f"{label}: two history frames share a time")
        self.times = np.array(times, dtype="datetime64[s]")
        self.paths = [p for _, p in stamped]
        with netCDF4.Dataset(self.paths[0]) as nc:
            self.grid, self.offset, self.projection = _projection_grid(nc)
            self.ny = len(nc.dimensions["south_north"])
            self.nx = len(nc.dimensions["west_east"])
            self.nz = len(nc.dimensions["bottom_top"])
            self.dx = float(nc.getncattr("DX"))
            self.sinalpha = np.asarray(nc.variables["SINALPHA"][0],
                                       dtype=np.float64) \
                if "SINALPHA" in nc.variables else None
            self.cosalpha = np.asarray(nc.variables["COSALPHA"][0],
                                       dtype=np.float64) \
                if "COSALPHA" in nc.variables else None
            self.hgt = np.asarray(nc.variables["HGT"][0], dtype=np.float64)
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._fields: OrderedDict[tuple[int, bool], dict] = OrderedDict()

    # -- geometry ---------------------------------------------------------

    def mass_index(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """0-based fractional mass-point indices ``(x, y)`` of points."""

        i, j = self.grid.latlon_to_ij(np.atleast_1d(np.asarray(lat, float)),
                                      np.atleast_1d(np.asarray(lon, float)))
        return (np.asarray(i, float) - self.offset[0],
                np.asarray(j, float) - self.offset[1])

    def inside(self, x, y, *, edge_cells: int = 0) -> np.ndarray:
        """Whether mass-index points lie at least ``edge_cells`` cells
        inside the domain's outermost mass points."""

        e = float(edge_cells)
        return ((x >= e) & (x <= self.nx - 1 - e)
                & (y >= e) & (y <= self.ny - 1 - e))

    # -- frames -------------------------------------------------------------

    def frame(self, index: int) -> dict:
        """The fields of frame ``index``, float64, at their own staggers."""

        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        netCDF4 = _netcdf4()
        with netCDF4.Dataset(self.paths[index]) as nc:
            def read(name):
                return np.asarray(nc.variables[name][0], dtype=np.float64)

            z_full = (read("PH") + read("PHB")) / GRAVITY
            data = {
                "z_full": z_full,
                "z_mass": 0.5 * (z_full[:-1] + z_full[1:]),
                "U": read("U"), "V": read("V"), "W": read("W"),
                "T": read("T"), "P_total": read("P") + read("PB"),
                "QVAPOR": (read("QVAPOR") if "QVAPOR" in nc.variables
                           else None),
            }
        self._cache[index] = data
        while len(self._cache) > 2:
            self._cache.popitem(last=False)
        return data

    def fields(self, index: int, *, earth: bool) -> dict:
        """:func:`_domain_fields` of frame ``index``, cached beside the
        frame so a path of many fixes builds them once per frame."""

        key = (int(index), bool(earth))
        if key in self._fields:
            self._fields.move_to_end(key)
            return self._fields[key]
        built = _domain_fields(self, self.frame(index), earth=earth)
        self._fields[key] = built
        while len(self._fields) > 4:
            self._fields.popitem(last=False)
        return built

    def bracket(self, when: np.datetime64):
        """``((index, weight), ...)`` of the frames bracketing ``when``,
        or ``None`` outside the history."""

        when = np.datetime64(when, "s")
        if when < self.times[0] or when > self.times[-1]:
            return None
        k = int(np.searchsorted(self.times, when, side="left"))
        if self.times[k] == when:
            return ((k, 1.0),)
        span = float((self.times[k] - self.times[k - 1])
                     / np.timedelta64(1, "s"))
        a = float((when - self.times[k - 1]) / np.timedelta64(1, "s")) / span
        return ((k - 1, 1.0 - a), (k, a))


# --------------------------------------------------------------------------
# interpolation


def interp_columns(z_cols: np.ndarray, v_cols: np.ndarray,
                   z_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation in height, column by column.

    ``z_cols``/``v_cols`` are ``(nz, N)`` with heights increasing up each
    column; ``z_target`` is ``(nt, N)``.  Returns the values ``(nt, N)``
    and a ``below`` mask: a target below a column's lowest height holds
    that lowest value, one above its highest is NaN.
    """

    z_cols = np.asarray(z_cols, dtype=np.float64)
    v_cols = np.asarray(v_cols, dtype=np.float64)
    z_target = np.asarray(z_target, dtype=np.float64)
    nz, n = z_cols.shape
    nt = z_target.shape[0]
    out = np.full((nt, n), np.nan)
    below = np.zeros((nt, n), dtype=bool)
    step = max(1, _CHUNK_ELEMENTS // max(1, nz * nt))
    for start in range(0, n, step):
        sl = slice(start, min(n, start + step))
        zc, vc, zt = z_cols[:, sl], v_cols[:, sl], z_target[:, sl]
        count = (zc[None, :, :] <= zt[:, None, :]).sum(axis=1)
        lo = np.clip(count - 1, 0, nz - 2)
        cols = np.arange(zc.shape[1])[None, :]
        z0, z1 = zc[lo, cols], zc[lo + 1, cols]
        v0, v1 = vc[lo, cols], vc[lo + 1, cols]
        with np.errstate(invalid="ignore", divide="ignore"):
            a = (zt - z0) / (z1 - z0)
        value = v0 + a * (v1 - v0)
        low = zt < zc[0][None, :]
        high = zt > zc[-1][None, :]
        value = np.where(low, zc[0][None, :] * 0.0 + vc[0][None, :], value)
        value = np.where(high, np.nan, value)
        out[:, sl] = value
        below[:, sl] = low
    return out, below


def _bilinear_corners(x, y, nx: int, ny: int):
    """The four corner indices and weights of 0-based fractional points,
    clamped onto the grid."""

    x = np.clip(np.asarray(x, float), 0.0, nx - 1.0)
    y = np.clip(np.asarray(y, float), 0.0, ny - 1.0)
    i0 = np.minimum(np.floor(x).astype(np.int64), max(nx - 2, 0))
    j0 = np.minimum(np.floor(y).astype(np.int64), max(ny - 2, 0))
    fx, fy = x - i0, y - j0
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    return ((j0, i0, (1 - fx) * (1 - fy)), (j0, i1, fx * (1 - fy)),
            (j1, i0, (1 - fx) * fy), (j1, i1, fx * fy))


def sample_field(field3: np.ndarray, z3: np.ndarray, x, y,
                 z_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``field3`` (nz, ny, nx) on heights ``z3`` (same shape) at points
    ``(x, y)`` (fractional indices in that field's own stagger) and
    heights ``z_target`` ``(nt, N)``: vertical in each of the four
    neighbouring columns, then bilinear.  Returns ``(values, below)``."""

    nz, ny, nx = field3.shape
    total = np.zeros(z_target.shape)
    below = np.zeros(z_target.shape, dtype=bool)
    for j, i, w in _bilinear_corners(x, y, nx, ny):
        values, low = interp_columns(z3[:, j, i], field3[:, j, i], z_target)
        # A corner with no weight contributes nothing, not its NaN.
        total = total + np.where(w[None, :] > 0.0, w[None, :] * values, 0.0)
        below |= low & (w[None, :] > 0.0)
    return total, below


def _domain_fields(history: DomainHistory, frame: dict, *,
                   earth: bool) -> dict[str, tuple]:
    """``name -> (field, heights, stagger)`` for the sampled fields."""

    z_mass, z_full = frame["z_mass"], frame["z_full"]
    if earth:
        u_mass = 0.5 * (frame["U"][:, :, :-1] + frame["U"][:, :, 1:])
        v_mass = 0.5 * (frame["V"][:, :-1, :] + frame["V"][:, 1:, :])
        if history.sinalpha is not None:
            sina, cosa = history.sinalpha, history.cosalpha
            u_e = u_mass * cosa - v_mass * sina
            v_e = v_mass * cosa + u_mass * sina
        else:
            u_e, v_e = u_mass, v_mass
        fields = {"u": (u_e, z_mass, "mass"), "v": (v_e, z_mass, "mass")}
    else:
        fields = {"U": (frame["U"], _stagger_mean(z_mass, 2), "x"),
                  "V": (frame["V"], _stagger_mean(z_mass, 1), "y")}
    fields.update({
        ("w" if earth else "W"): (frame["W"], z_full, "z"),
        ("theta" if earth else "T"): (frame["T"] + (T0 if earth else 0.0),
                                      z_mass, "mass"),
        ("pressure" if earth else "P_total"): (frame["P_total"], z_mass,
                                               "mass"),
    })
    if frame["QVAPOR"] is not None:
        fields["qvapor" if earth else "QVAPOR"] = (frame["QVAPOR"], z_mass,
                                                   "mass")
    return fields


def _stagger_index(x, y, stagger: str):
    if stagger == "x":
        return x + 0.5, y
    if stagger == "y":
        return x, y + 0.5
    return x, y


# --------------------------------------------------------------------------
# tiers and the precedence rule


#: WRF's default ``spec_bdy_width``: the cells a nest's boundary rows and
#: relaxation zone take together, used when a tier's own width cannot be
#: read beside its history.
DEFAULT_EDGE_CELLS = 5
#: Child cells, inside the edge zone, across which a tier fades in.
DEFAULT_LATERAL_BLEND_CELLS = 3


@dataclass
class Tier:
    """One grid of the stack.

    ``cutoff_m`` is the geopotential height (m) the tier is used below;
    ``None`` is full depth (the coarsest tier).  ``edge_cells`` cells at
    each lateral edge are the tier's own boundary rows and relaxation zone
    and are never used; the tier then fades in across
    ``lateral_blend_cells`` more.
    """

    history: DomainHistory
    cutoff_m: float | None = None
    edge_cells: int = DEFAULT_EDGE_CELLS
    lateral_blend_cells: int = DEFAULT_LATERAL_BLEND_CELLS

    @property
    def label(self) -> str:
        return self.history.label

    def lateral_weight(self, x, y) -> np.ndarray:
        """The tier's lateral share at mass-index points: 0 outside it and
        across its edge zone, rising linearly to 1 over the blend cells."""

        d = np.minimum(np.minimum(x, self.history.nx - 1 - x),
                       np.minimum(y, self.history.ny - 1 - y))
        e = float(self.edge_cells)
        b = float(self.lateral_blend_cells)
        if b <= 0.0:
            return np.where(d >= e, 1.0, 0.0)
        return np.clip((d - e) / b, 0.0, 1.0)


def tier_edge_cells(paths: Sequence[Path]) -> int | None:
    """The ``spec_bdy_width`` a ``child.toml`` beside the history declares
    (the file ``woof downscale`` writes), or ``None``."""

    import tomllib

    folder = Path(paths[0]).parent
    for candidate in (folder / "child.toml",):
        if candidate.is_file():
            try:
                data = tomllib.loads(candidate.read_text("utf-8"))
            except (OSError, ValueError):
                return None
            for table in (data, data.get("run", {}),
                          data.get("experiment", {})):
                if isinstance(table, dict) and "spec_bdy_width" in table:
                    return int(table["spec_bdy_width"])
    return None


def _ordered(tiers: Sequence[Tier]) -> list[Tier]:
    """The documented order: coarsest first (the only full-depth tier),
    then each finer tier; refused when the order is not by spacing."""

    tiers = list(tiers)
    if not tiers:
        raise SpliceRefused("splice needs at least one tier")
    if tiers[0].cutoff_m is not None:
        raise SpliceRefused(layered(
            "the first (coarsest) tier must be full depth, with no cutoff",
            "Every point the finer tiers do not cover falls back to it."))
    for coarse, fine in zip(tiers, tiers[1:]):
        if fine.cutoff_m is None:
            raise SpliceRefused(
                f"tier {fine.label} needs a cutoff height: only the "
                "coarsest tier is full depth")
        if not fine.history.dx < coarse.history.dx:
            raise SpliceRefused(
                f"tiers must go from coarse to fine: {fine.label} "
                f"({fine.history.dx:g} m) follows {coarse.label} "
                f"({coarse.history.dx:g} m)")
    return tiers


# --------------------------------------------------------------------------
# path sampling


@dataclass
class PathSample:
    """Spliced values at every fix, each tier's own values and share, and
    the continuity statement."""

    times: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    height_m: np.ndarray
    spliced: dict[str, np.ndarray]
    tiers: list[str]
    tier_values: list[dict[str, np.ndarray]]
    tier_weights: list[np.ndarray]
    source: np.ndarray
    held_below: np.ndarray
    continuity: dict[str, Any]
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def child_weight(self) -> np.ndarray:
        """The finest tier's share (two-tier shorthand)."""

        return self.tier_weights[-1]

    @property
    def child(self) -> dict[str, np.ndarray]:
        return self.tier_values[-1]

    @property
    def parent(self) -> dict[str, np.ndarray]:
        return self.tier_values[0]

    @property
    def child_inside(self) -> np.ndarray:
        return self.tier_weights[-1] > 0.0

    def rows(self) -> Iterable[dict[str, Any]]:
        for k in range(self.times.size):
            row = {"time": str(self.times[k]) + "Z",
                   "lat": float(self.lat[k]), "lon": float(self.lon[k]),
                   "height_m": float(self.height_m[k]),
                   "source": str(self.source[k]),
                   "held_below_lowest_level": bool(self.held_below[k])}
            for name in SAMPLE_FIELDS:
                value = self.spliced[name][k]
                row[name] = None if not np.isfinite(value) else float(value)
            for label, values, weight in zip(self.tiers, self.tier_values,
                                             self.tier_weights):
                row[f"{label}_weight"] = float(weight[k])
                for name in SAMPLE_FIELDS:
                    value = values[name][k]
                    row[f"{label}_{name}"] = (None if not np.isfinite(value)
                                              else float(value))
            yield row


def _sample_domain_at(history: DomainHistory, when, x, y, z, *,
                      height_datum: str):
    """One domain's earth-relative values at fixes sharing time ``when``;
    ``(None, None)`` when ``when`` is outside its history."""

    bracket = history.bracket(when)
    if bracket is None:
        return None, None
    result: dict[str, np.ndarray] = {}
    below_any = np.zeros(x.shape, dtype=bool)
    if height_datum == "agl":
        ground = np.zeros(x.shape)
        for j, i, w in _bilinear_corners(x, y, history.nx, history.ny):
            ground += w * history.hgt[j, i]
        z = z + ground
    for index, weight in bracket:
        fields = history.fields(index, earth=True)
        for name, (field3, z3, stagger) in fields.items():
            xs, ys = _stagger_index(x, y, stagger)
            values, below = sample_field(field3, z3, xs, ys, z[None, :])
            result[name] = result.get(name, 0.0) + weight * values[0]
            below_any |= below[0]
    return result, below_any


def _sample_tier(tier: Tier, times, x, y, heights, *, height_datum: str,
                 inside):
    """``tier``'s values (NaN where it does not answer) at every fix."""

    n = times.size
    values = {name: np.full(n, np.nan) for name in SAMPLE_FIELDS}
    held = np.zeros(n, dtype=bool)
    for when in np.unique(times):
        rows = np.flatnonzero((times == when) & inside)
        if not rows.size:
            continue
        got, below = _sample_domain_at(tier.history, when, x[rows], y[rows],
                                       heights[rows],
                                       height_datum=height_datum)
        if got is None:
            continue
        for name, value in got.items():
            if name in values:
                values[name][rows] = value
        held[rows] |= below
    return values, held


def sample_path(tiers: Sequence[Tier], *, times, lat, lon, height,
                blend_m: float = 0.0, height_datum: str = "msl"
                ) -> PathSample:
    """Sample the spliced atmosphere at every fix.

    THE PRECEDENCE RULE.  ``tiers`` run coarsest first.  At each fix the
    finest tier that covers it -- inside its footprint beyond its edge
    zone, below its cutoff height, and inside its history in time --
    answers; across its lateral blend cells and its vertical blend layer
    it is mixed linearly with whatever the coarser tiers make of the fix.
    A tier that does not answer (outside, above, or missing) hands the fix
    to the next coarser one, down to the full-depth coarsest tier, which
    alone may leave a fix missing.
    """

    tiers = _ordered(tiers)
    times = np.asarray([np.datetime64(t, "s") for t in times])
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)
    n = times.size
    # The splice is decided on the fix's own height above sea level; an
    # AGL path stands on the coarsest tier's ground, the one reference
    # every tier shares.
    base = tiers[0].history
    bx, by = base.mass_index(lat, lon)
    z_basis = height.copy()
    if height_datum == "agl":
        ground = np.zeros(n)
        for j, i, w in _bilinear_corners(bx, by, base.nx, base.ny):
            ground += w * base.hgt[j, i]
        z_basis = height + ground
    tier_values, tier_weights, lateral, held = [], [], [], np.zeros(n, bool)
    positions = []
    for index, tier in enumerate(tiers):
        x, y = tier.history.mass_index(lat, lon)
        positions.append((x, y))
        if index == 0:
            w_lat = np.where(tier.history.inside(x, y), 1.0, 0.0)
        else:
            w_lat = tier.lateral_weight(x, y)
        # Every tier is asked for the SAME height above sea level: an AGL
        # path stands on the coarsest tier's ground (z_basis), so a blend
        # or a lateral comparison never mixes two different heights.
        values, below = _sample_tier(tier, times, x, y, z_basis,
                                     height_datum="msl",
                                     inside=w_lat > 0.0)
        w_vert = (np.ones(n) if tier.cutoff_m is None
                  else child_weight(z_basis, tier.cutoff_m, blend_m))
        weight = w_lat * w_vert
        # The tier "answers" a fix when its winds and theta are there; a
        # field it lacks there (a dry run's qvapor) is handed down per
        # field below, never blended in as zero.
        answered = np.all([np.isfinite(values[name])
                           for name in ("u", "v", "theta")], axis=0)
        weight = np.where(answered, weight, 0.0)
        held |= below & (weight > 0.0)
        tier_values.append(values)
        tier_weights.append(weight)
        lateral.append(w_lat)
    spliced = {name: np.full(n, np.nan) for name in SAMPLE_FIELDS}
    source = np.full(n, "none", dtype=object)
    for index, (tier, values, weight) in enumerate(
            zip(tiers, tier_values, tier_weights)):
        for name in SAMPLE_FIELDS:
            v = values[name]
            coarse = spliced[name]
            w = np.where(np.isfinite(v), weight, 0.0)
            # Where nothing coarser answered, the fine value stands alone.
            w = np.where((w > 0.0) & ~np.isfinite(coarse), 1.0, w)
            mixed = w * np.where(np.isfinite(v), v, 0.0) + (1.0 - w) * coarse
            spliced[name] = np.where(w >= 1.0, v,
                                     np.where(w > 0.0, mixed, coarse))
        has_coarse = (np.isfinite(spliced["u"]) if index else
                      np.zeros(n, dtype=bool))
        source = np.where(weight >= 1.0, tier.label,
                          np.where(weight > 0.0,
                                   np.where(has_coarse, f"blend:{tier.label}",
                                            tier.label),
                                   source))
    continuity = continuity_statement(
        tiers, tier_values, lateral, positions, times=times, lat=lat,
        lon=lon, z_basis=z_basis, height=height, height_datum=height_datum,
        spliced=spliced, source=source)
    return PathSample(times=times, lat=lat, lon=lon, height_m=height,
                      spliced=spliced, tiers=[t.label for t in tiers],
                      tier_values=tier_values, tier_weights=tier_weights,
                      source=source, held_below=held, continuity=continuity,
                      settings={"blend_m": float(blend_m),
                                "height_datum": height_datum,
                                "tiers": [_tier_settings(t) for t in tiers]})


def _tier_settings(tier: Tier) -> dict[str, Any]:
    return {"label": tier.label, "dx_m": tier.history.dx,
            "levels": tier.history.nz, "nx": tier.history.nx,
            "ny": tier.history.ny, "cutoff_m": tier.cutoff_m,
            "edge_cells": tier.edge_cells,
            "lateral_blend_cells": tier.lateral_blend_cells}


def _stats(diff) -> dict[str, float]:
    diff = np.asarray(diff, dtype=np.float64)
    good = np.isfinite(diff)
    if not good.any():
        return {"n": 0}
    d = diff[good]
    return {"n": int(d.size), "mean": float(d.mean()),
            "rms": float(np.sqrt(np.mean(d * d))),
            "max_abs": float(np.abs(d).max())}


def continuity_statement(tiers, tier_values, lateral, positions, *, times,
                         lat, lon, z_basis, height, height_datum, spliced,
                         source) -> dict[str, Any]:
    """How well each tier meets the next coarser one, along the path.

    * ``vertical``: fine minus coarse AT the fine tier's cutoff height, at
      every fix inside the fine tier laterally -- the jump a hard switch
      puts into a profile there;
    * ``lateral``: fine minus coarse at the fix's own height, at fixes in
      the fine tier's lateral blend band (or, with no band, within one
      cell of its edge zone) and below its cutoff;
    * ``steps``: the largest change of the spliced value between
      consecutive fixes where the answering source changes, beside the
      median change between consecutive fixes elsewhere.
    """

    statement: dict[str, Any] = {"joins": []}
    for index in range(1, len(tiers)):
        fine, coarse = tiers[index], tiers[index - 1]
        x, y = positions[index]
        cx, cy = positions[index - 1]
        inside = lateral[index] > 0.0
        cutoff = np.full(times.size, float(fine.cutoff_m))
        f_at, _ = _sample_tier(fine, times, x, y, cutoff, height_datum="msl",
                               inside=inside)
        c_at, _ = _sample_tier(coarse, times, cx, cy, cutoff,
                               height_datum="msl", inside=inside)
        d = np.minimum(np.minimum(x, fine.history.nx - 1 - x),
                       np.minimum(y, fine.history.ny - 1 - y))
        band = max(float(fine.lateral_blend_cells), 1.0)
        near_edge = ((d >= fine.edge_cells) & (d <= fine.edge_cells + band)
                     & (z_basis < fine.cutoff_m))
        f_vals, c_vals = tier_values[index], tier_values[index - 1]
        statement["joins"].append({
            "fine": fine.label, "coarse": coarse.label,
            "cutoff_m": float(fine.cutoff_m),
            "vertical": {name: _stats(f_at[name] - c_at[name])
                         for name in SAMPLE_FIELDS},
            "lateral": {name: _stats(np.where(near_edge, f_vals[name]
                                              - c_vals[name], np.nan))
                        for name in SAMPLE_FIELDS},
        })
    changed = np.zeros(times.size, dtype=bool)
    changed[1:] = source[1:] != source[:-1]
    steps = {}
    for name in SAMPLE_FIELDS:
        step = np.full(times.size, np.nan)
        step[1:] = np.abs(np.diff(spliced[name]))
        at = step[changed]
        rest = step[~changed]
        steps[name] = {
            "max_at_source_change": (float(np.nanmax(at))
                                     if np.isfinite(at).any() else None),
            "median_elsewhere": (float(np.nanmedian(rest))
                                 if np.isfinite(rest).any() else None),
            "p99_elsewhere": (float(np.nanpercentile(rest, 99))
                              if np.isfinite(rest).any() else None)}
    statement["steps"] = steps
    statement["source_changes"] = int(changed.sum())
    return statement


# --------------------------------------------------------------------------
# merged field set


def _same_projection(a: DomainHistory, b: DomainHistory) -> bool:
    pa, pb = a.projection, b.projection
    return (pa["MAP_PROJ"] == pb["MAP_PROJ"]
            and all(abs(pa[k] - pb[k]) < 1e-5
                    for k in ("TRUELAT1", "TRUELAT2", "STAND_LON")))


def _parent_at(parent: DomainHistory, when, index_of, heights,
               name: str, stagger: str) -> np.ndarray:
    """The parent's ``name`` (grid-relative, own stagger) at the points
    whose mass indices in it are ``index_of`` and heights ``(nt, N)``,
    time-interpolated; NaN where the parent does not reach (outside it,
    or outside its history)."""

    bracket = parent.bracket(when)
    if bracket is None:
        return np.full(heights.shape, np.nan)
    x, y = index_of
    xs, ys = _stagger_index(x, y, stagger)
    total = np.zeros(heights.shape)
    for index, weight in bracket:
        fields = parent.fields(index, earth=False)
        if name not in fields:
            return np.full(heights.shape, np.nan)
        field3, z3, _ = fields[name]
        values, _ = sample_field(field3, z3, xs, ys, heights)
        total = total + weight * values
    outside = ~parent.inside(x, y)
    return np.where(outside[None, :], np.nan, total)


def _coarser_at(below: Sequence[DomainHistory], when, indices, heights,
                name: str, stagger: str) -> np.ndarray:
    """The precedence rule for a field set: the nearest coarser history
    that answers each point, falling back down the stack."""

    result = np.full(heights.shape, np.nan)
    for history, index_of in zip(below, indices):
        missing = ~np.isfinite(result)
        if not missing.any():
            break
        values = _parent_at(history, when, index_of, heights, name, stagger)
        result = np.where(missing, values, result)
    return result


def _stagger_lateral(tier: Tier, stagger: str, shape) -> np.ndarray:
    """``tier``'s lateral weight on one stagger's horizontal points."""

    ny, nx = shape
    jj, ii = np.mgrid[0:ny, 0:nx].astype(float)
    if stagger == "x":
        ii = ii - 0.5
    elif stagger == "y":
        jj = jj - 0.5
    # A boundary face is a point of the grid, not outside it.
    ii = np.clip(ii, 0.0, tier.history.nx - 1.0)
    jj = np.clip(jj, 0.0, tier.history.ny - 1.0)
    return tier.lateral_weight(ii, jj)


def _stagger_latlon(history: DomainHistory, stagger: str):
    """Latitude/longitude of one stagger's points, from the projection."""

    ny, nx = history.ny, history.nx
    if stagger == "x":
        jj, ii = np.mgrid[0:ny, 0:nx + 1].astype(float)
        ii = ii - 0.5
    elif stagger == "y":
        jj, ii = np.mgrid[0:ny + 1, 0:nx].astype(float)
        jj = jj - 0.5
    else:
        jj, ii = np.mgrid[0:ny, 0:nx].astype(float)
    lat, lon = history.grid.ij_to_latlon(
        (ii + history.offset[0]).reshape(-1),
        (jj + history.offset[1]).reshape(-1))
    return (np.asarray(lat, float).reshape(ii.shape),
            np.asarray(lon, float).reshape(ii.shape))


def merge_tier(child: Tier, parent, out_dir, *,
               blend_m: float = 0.0) -> list[Path]:
    """Write one merged frame per frame of ``child`` into ``out_dir``: the
    child below its cutoff and inside its footprint, the coarser stack
    elsewhere, with the vertical and lateral blends.  ``parent`` is one
    history or a sequence of them, nearest coarser first (the merged tier
    below, then the ones below that): each point takes the first that
    answers it.  Frames keep the child's names and wrfout layout."""

    history = child.history
    below = ([parent] if isinstance(parent, DomainHistory)
             else list(parent))
    if not below:
        raise SpliceRefused("a merge needs at least one coarser history")
    parent = below[0]
    if not all(_same_projection(history, other) for other in below):
        raise SpliceRefused(layered(
            "the merged field set keeps wrfout's grid-relative winds, and "
            "the child and parent are not on one projection",
            "Sample a path instead (its winds are earth-relative), or "
            "splice domains of one nested run."))
    netCDF4 = _netcdf4()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    lat_of, lon_of = {}, {}
    with netCDF4.Dataset(history.paths[0]) as nc:
        for stagger, (lat_name, lon_name) in (
                ("mass", ("XLAT", "XLONG")), ("x", ("XLAT_U", "XLONG_U")),
                ("y", ("XLAT_V", "XLONG_V"))):
            if lat_name in nc.variables and lon_name in nc.variables:
                lat_of[stagger] = np.asarray(nc.variables[lat_name][0], float)
                lon_of[stagger] = np.asarray(nc.variables[lon_name][0], float)
            else:
                # woof's history carries no staggered coordinates: place
                # the faces through the domain's own projection.
                lat_of[stagger], lon_of[stagger] = _stagger_latlon(history,
                                                                   stagger)
    lat_of["z"], lon_of["z"] = lat_of["mass"], lon_of["mass"]
    # The child's points never move: locate them in every coarser history
    # once, per stagger, not once per frame and field.
    located = {stagger: [other.mass_index(lat_of[stagger].reshape(-1),
                                          lon_of[stagger].reshape(-1))
                         for other in below]
               for stagger in ("mass", "x", "y")}
    located["z"] = located["mass"]
    cutoff = child.cutoff_m
    for index, (when, source) in enumerate(zip(history.times, history.paths)):
        frame = history.frame(index)
        fields = history.fields(index, earth=False)
        target = out_dir / source.name
        if target.resolve() == Path(source).resolve():
            raise SpliceRefused("the merged field set would overwrite the "
                                "child's own history; choose another folder")
        shutil.copyfile(source, target)
        with netCDF4.Dataset(target, "r+") as nc:
            for name, (field3, z3, stagger) in fields.items():
                shape = field3.shape
                heights = z3.reshape(shape[0], -1)
                w_lat = _stagger_lateral(child, stagger,
                                         shape[1:]).reshape(1, -1)
                weight = w_lat * (np.ones_like(heights) if cutoff is None
                                  else child_weight(heights, cutoff, blend_m))
                if np.all(weight == 1.0):
                    continue
                parent_vals = _coarser_at(below, when, located[stagger],
                                          heights, name, stagger)
                own = field3.reshape(shape[0], -1)
                merged = weight * own + (1.0 - weight) * parent_vals
                # A point the parent cannot answer keeps its own value
                # rather than becoming missing.
                merged = np.where(np.isfinite(merged), merged, own)
                merged = merged.reshape(shape)
                if name == "P_total":
                    pb = np.asarray(nc.variables["PB"][0], dtype=np.float64)
                    nc.variables["P"][0] = (merged - pb).astype(
                        nc.variables["P"].dtype)
                else:
                    nc.variables[name][0] = merged.astype(
                        nc.variables[name].dtype)
            w_lat = _stagger_lateral(child, "mass", frame["z_mass"].shape[1:])
            weight_mass = w_lat[None] * (
                np.ones_like(frame["z_mass"]) if cutoff is None
                else child_weight(frame["z_mass"], cutoff, blend_m))
            if WEIGHT_VARIABLE not in nc.variables:
                var = nc.createVariable(
                    WEIGHT_VARIABLE, "f4",
                    ("Time", "bottom_top", "south_north", "west_east"))
                var.setncattr("description",
                              "this grid's share of the spliced value "
                              "(1 this grid, 0 the coarser one)")
                var.setncattr("units", "1")
                var.setncattr("stagger", "")
                var.setncattr("coordinates", "XLONG XLAT XTIME")
            nc.variables[WEIGHT_VARIABLE][0] = weight_mass.astype(np.float32)
            nc.setncattr("SPLICE_SCHEMA", SPLICE_SCHEMA)
            nc.setncattr("SPLICE_HEIGHT_M",
                         float("nan") if cutoff is None else float(cutoff))
            nc.setncattr("SPLICE_BLEND_M", float(blend_m))
            nc.setncattr("SPLICE_EDGE_CELLS", int(child.edge_cells))
            nc.setncattr("SPLICE_LATERAL_BLEND_CELLS",
                         int(child.lateral_blend_cells))
            nc.setncattr("SPLICE_PARENT_DX", float(parent.dx))
        written.append(target)
    return written


def merge_fields(tiers: Sequence[Tier], out_dir, *, blend_m: float = 0.0
                 ) -> dict[str, list[Path]]:
    """Merge a stack of tiers, coarsest first, into ``out_dir/<label>/``.

    Each finer tier is merged against the MERGED tier below it, so a
    point of the finest grid that lies outside the middle tier's
    footprint takes the coarsest tier's value there: the same precedence
    rule as a path sample.  The coarsest tier is not rewritten.
    """

    tiers = _ordered(tiers)
    out_dir = Path(out_dir)
    result: dict[str, list[Path]] = {}
    # Nearest coarser first: the merged tier just below, then the ones
    # below it, so a point outside the middle tier's footprint falls back
    # to the coarsest, as in a path sample.
    below = [tiers[0].history]
    for tier in tiers[1:]:
        written = merge_tier(tier, below, out_dir / tier.label,
                             blend_m=blend_m)
        result[tier.label] = written
        below = [DomainHistory(written, label=f"merged-{tier.label}"),
                 *below]
    return result


# --------------------------------------------------------------------------
# output


def write_path_csv(sample: PathSample, path) -> Path:
    import csv

    path = Path(path)
    rows = list(sample.rows())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows
                                else ["time"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v)
                             for k, v in row.items()})
    return path


def path_document(sample: PathSample, *, name: str,
                  sources: Mapping[str, Any]) -> dict[str, Any]:
    counts = {label: int((weight > 0.0).sum())
              for label, weight in zip(sample.tiers, sample.tier_weights)}
    return {"schema": SPLICE_SCHEMA, "name": name,
            "settings": sample.settings, "sources": dict(sources),
            "fixes": int(sample.times.size), "fixes_answered_by": counts,
            "continuity": sample.continuity,
            "samples": list(sample.rows())}


def write_path_icechunk(sample: PathSample, store, *, group: str,
                        name: str, sources: Mapping[str, Any],
                        branch: str = "main") -> dict[str, Any]:
    """Write a path sample into ``group`` of the Icechunk repository
    ``store``: one ``(sample,)`` array per column, attributes carrying the
    settings and the continuity statement; one commit."""

    from woof.energy import icechunk_store as ics

    if not group or group.strip("/") != group:
        raise SpliceRefused(f"group {group!r} must be a non-empty relative "
                            "path")
    zarr = ics._zarr()
    repo = ics.open_repository(store)
    epoch = np.datetime64("1970-01-01T00:00:00", "s")
    seconds = ((sample.times - epoch) / np.timedelta64(1, "s")).astype(
        np.int64)
    columns: dict[str, tuple[np.ndarray, dict]] = {
        "time": (seconds, {"units": "seconds since 1970-01-01 00:00:00",
                           "calendar": "proleptic_gregorian"}),
        "lat": (sample.lat, {"units": "degrees_north"}),
        "lon": (sample.lon, {"units": "degrees_east"}),
        "height": (sample.height_m,
                   {"units": "m", "datum": sample.settings["height_datum"]}),
        "held_below_lowest_level": (sample.held_below.astype(np.int8), {}),
    }
    for field_ in SAMPLE_FIELDS:
        columns[field_] = (np.asarray(sample.spliced[field_]), {})
    for label, values, weight in zip(sample.tiers, sample.tier_values,
                                     sample.tier_weights):
        columns[f"{label}_weight"] = (np.asarray(weight), {"units": "1"})
        for field_ in SAMPLE_FIELDS:
            columns[f"{label}_{field_}"] = (np.asarray(values[field_]), {})

    def apply(session):
        root = zarr.open_group(session.store, mode="a")
        if group in {key for key, _ in root.members(max_depth=None)}:
            del root[group]
        zgroup = root.require_group(group)
        for key, (values, attrs) in columns.items():
            values = np.asarray(values)
            dtype = (values.dtype if values.dtype.kind in "iu"
                     else np.dtype(np.float64))
            array = zgroup.create_array(
                key, shape=values.shape, dtype=dtype,
                fill_value=(0 if dtype.kind in "iu" else np.nan),
                dimension_names=("sample",), attributes=attrs)
            array[...] = values.astype(dtype)
        zgroup.attrs.update(json.loads(json.dumps({
            "schema": SPLICE_SCHEMA, "name": name,
            "settings": sample.settings, "sources": dict(sources),
            "tiers": sample.tiers,
            "source_by_sample": [str(s) for s in sample.source],
            "continuity": sample.continuity}, default=_json_default)))
        return {"group": group, "columns": len(columns)}

    result = ics.commit_with_retry(
        repo, branch=branch, apply=apply,
        message=f"woof track splice: {name}",
        metadata={"schema": SPLICE_SCHEMA, "group": group},
        lock=ics.commit_lock(store))
    return {"snapshot": result.snapshot, "committed": result.committed,
            "group": group}


# --------------------------------------------------------------------------
# CLI


def parse_tier(text: str) -> tuple[Path, float | None, int | None]:
    """``--tier PATH[,CUTOFF_M][,dN]`` -> ``(path, cutoff, domain)``."""

    import argparse

    # Read from the right: trailing fields that are a number or dN are the
    # cutoff and the domain, and everything before them is the path, so a
    # path that itself holds a comma survives.
    parts = str(text).split(",")
    cutoff = domain = None
    while len(parts) > 1:
        tail = parts[-1].strip()
        if (domain is None and tail.lower().startswith("d")
                and tail[1:].isdigit()):
            domain = int(tail[1:])
        elif cutoff is None:
            try:
                cutoff = float(tail)
            except ValueError:
                break
            if not math.isfinite(cutoff):
                raise argparse.ArgumentTypeError(
                    f"--tier {text!r}: the cutoff must be finite")
        else:
            break
        parts.pop()
    path = ",".join(parts).strip()
    if not path:
        raise argparse.ArgumentTypeError(
            f"--tier {text!r}: expected PATH[,CUTOFF_M][,dN]")
    return Path(path), cutoff, domain


def register_splice(commands) -> None:
    parser = commands.add_parser(
        "splice", help="join fine low-level grids and a coarse full-depth "
                       "one: a sample along a 3-D path, or a merged field "
                       "set",
        description="Read the history of a stack of grids -- the coarsest "
                    "full depth, finer ones over smaller footprints and "
                    "below cutoff heights -- and answer each point from "
                    "the finest grid that covers it, laterally beyond its "
                    "edge zone and below its cutoff, falling back to the "
                    "next coarser grid outside it, with linear blends at "
                    "both joins.  --path samples every fix of a "
                    "time/lat/lon/height path and states how each grid "
                    "meets the next at its cutoff and its footprint edge; "
                    "--fields-out writes each finer grid's frames merged "
                    "with the coarser ones, for --icechunk.  Two grids "
                    "can be given as --parent/--child/--splice-height-m, "
                    "or any number as --tier, coarsest first.")
    parser.add_argument("--tier", action="append", default=None,
                        metavar="PATH[,CUTOFF_M][,dN]",
                        help="one grid of the stack: its wrfout files' "
                             "folder (or one file), its cutoff height in "
                             "geopotential m (absent on the first, "
                             "coarsest, full-depth grid) and its domain "
                             "number when the folder holds several; "
                             "repeat coarsest first")
    parser.add_argument("--child", nargs="+", default=None, type=Path,
                        metavar="PATH",
                        help="two-grid form: the fine grid's wrfout files "
                             "or run folder")
    parser.add_argument("--parent", nargs="+", default=None, type=Path,
                        metavar="PATH",
                        help="two-grid form: the coarse full-depth grid's "
                             "wrfout files or run folder")
    parser.add_argument("--child-domain", type=int, default=None, metavar="N",
                        help="the child's domain number when a folder holds "
                             "several")
    parser.add_argument("--parent-domain", type=int, default=None,
                        metavar="N",
                        help="the parent's domain number when a folder "
                             "holds several")
    parser.add_argument("--splice-height-m", type=float, default=None,
                        metavar="M",
                        help="two-grid form: the child's cutoff height "
                             "(geopotential m)")
    parser.add_argument("--blend-m", type=float, default=0.0, metavar="M",
                        help="depth of a linear blend layer centred on each "
                             "cutoff height (default 0: a hard switch)")
    parser.add_argument("--edge-cells", type=int, default=None, metavar="K",
                        help="cells at each lateral edge of every finer "
                             "grid that are never used (its boundary rows "
                             "and relaxation zone; default: the "
                             "spec_bdy_width of a child.toml beside its "
                             f"history, else {DEFAULT_EDGE_CELLS})")
    parser.add_argument("--lateral-blend-cells", type=int,
                        default=DEFAULT_LATERAL_BLEND_CELLS, metavar="K",
                        help="cells, inside the edge zone, across which a "
                             "finer grid fades in (default "
                             f"{DEFAULT_LATERAL_BLEND_CELLS}; 0 switches)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--path", type=Path, default=None, metavar="FILE",
                      help="a 3-D path file (time, lat, lon, height), read "
                           "with the `woof track inspect` column options")
    mode.add_argument("--fields-out", type=Path, default=None, metavar="DIR",
                      help="write each finer grid's merged frames into "
                           "DIR/<grid>/")
    parser.add_argument("--out", type=Path, default=None, metavar="FILE",
                        help="path sample output: .csv (one row per fix) or "
                             ".json (rows plus the continuity statement)")
    parser.add_argument("--icechunk", default=None, metavar="URL",
                        help="also write into this Icechunk repository: a "
                             "path sample as one (sample,) group, merged "
                             "frames through the wrfout group writer, one "
                             "group per grid")
    parser.add_argument("--group", default=None, metavar="NAME",
                        help="the Icechunk group (default: splice/<name> "
                             "for a path; spliced/<grid> for a field set)")
    parser.add_argument("--name", default=None, metavar="NAME",
                        help="the path's name (default: the file stem)")
    from woof.track.cli import add_mapping_arguments

    add_mapping_arguments(parser)
    parser.set_defaults(func=splice_main)
    from woof.explain import add_explain_flag

    add_explain_flag(parser, nested=True)


def tiers_from_args(args) -> list[Tier]:
    """The stack ``--tier`` or the two-grid form names, coarsest first."""

    specs: list[tuple[list[Path], float | None, int | None]] = []
    if args.tier:
        if args.child or args.parent or args.splice_height_m is not None:
            raise SpliceRefused(
                "give the stack as --tier ... or as --parent/--child/"
                "--splice-height-m, not both")
        for text in args.tier:
            path, cutoff, domain = parse_tier(text)
            specs.append(([path], cutoff, domain))
    else:
        if not (args.child and args.parent
                and args.splice_height_m is not None):
            raise SpliceRefused(
                "name the grids: --tier PATH[,CUTOFF_M] (repeat, coarsest "
                "first), or --parent, --child and --splice-height-m")
        specs = [(list(args.parent), None, args.parent_domain),
                 (list(args.child), float(args.splice_height_m),
                  args.child_domain)]
    tiers: list[Tier] = []
    labels: set[str] = set()
    for index, (sources, cutoff, domain) in enumerate(specs):
        paths = discover_frames(sources, domain, what=f"tier {index + 1}")
        match = _FRAME_NAME.match(paths[0].name)
        label = f"t{index + 1}" + (f"_d{int(match.group(1)):02d}"
                                   if match else "")
        while label in labels:
            label += "x"
        labels.add(label)
        edge = args.edge_cells
        if edge is None:
            edge = tier_edge_cells(paths)
        if edge is None:
            edge = DEFAULT_EDGE_CELLS
        tiers.append(Tier(DomainHistory(paths, label=label), cutoff_m=cutoff,
                          edge_cells=int(edge),
                          lateral_blend_cells=int(args.lateral_blend_cells)))
    return tiers


def splice_main(args) -> int:
    tiers = _ordered(tiers_from_args(args))
    sources = {"tiers": [_tier_settings(t) for t in tiers],
               "paths": {t.label: [str(p) for p in t.history.paths]
                         for t in tiers}}
    if args.fields_out is not None:
        written = merge_fields(tiers, args.fields_out, blend_m=args.blend_m)
        for label, paths in written.items():
            print(f"{label}: wrote {len(paths)} merged frame(s) to "
                  f"{Path(args.fields_out) / label}")
            if args.icechunk:
                from woof.energy import icechunk_store as ics

                times = DomainHistory(paths, label=label).times
                interval = (int((times[1] - times[0])
                                / np.timedelta64(1, "s"))
                            if times.size > 1 else 3600)
                axis = ics.TimeAxis(str(times[0]), str(times[-1]), interval)
                group = (f"{args.group}/{label}" if args.group
                         else f"spliced/{label}")
                ics.write_gridded_segment(args.icechunk, paths, group=group,
                                          axis=axis, on_existing="overwrite")
                print(f"  icechunk: {args.icechunk} group {group}")
        return 0
    from woof.track.cli import mapping_from_args
    from woof.track.ingest import load_track

    track = load_track(args.path, mapping_from_args(args), name=args.name,
                       min_climb_m=args.min_climb)
    times = [np.datetime64(t.astimezone(timezone.utc).replace(tzinfo=None),
                           "s") for t in track.times]
    sample = sample_path(tiers, times=times, lat=track.lat, lon=track.lon,
                         height=track.height, blend_m=args.blend_m,
                         height_datum=track.height_datum)
    if args.out is not None:
        if args.out.suffix.lower() == ".csv":
            write_path_csv(sample, args.out)
        else:
            args.out.write_text(json.dumps(
                path_document(sample, name=track.name, sources=sources),
                indent=1, allow_nan=False, default=_json_default) + "\n",
                encoding="utf-8")
    if args.icechunk:
        write_path_icechunk(sample, args.icechunk,
                            group=args.group or f"splice/{track.name}",
                            name=track.name, sources=sources)
    answered = ", ".join(
        f"{label} {int((w > 0).sum())}"
        for label, w in zip(sample.tiers, sample.tier_weights))
    print(f"{track.name}: {sample.times.size} fixes; answered by "
          f"{answered}; {sample.continuity['source_changes']} source "
          "change(s)")
    for join in sample.continuity["joins"]:
        for kind in ("vertical", "lateral"):
            stats = join[kind]["u"]
            if stats.get("n"):
                print(f"  {join['fine']} vs {join['coarse']} {kind}: u "
                      f"mean {stats['mean']:+.3g} rms {stats['rms']:.3g} "
                      f"max {stats['max_abs']:.3g} m/s (n={stats['n']})")
    return 0


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(type(value).__name__)


__all__ = ["DomainHistory", "PathSample", "SpliceRefused", "Tier",
           "child_weight", "continuity_statement", "discover_frames",
           "interp_columns", "merge_fields", "merge_tier", "parse_tier",
           "path_document", "register_splice", "sample_field",
           "sample_path", "tier_edge_cells", "write_path_csv",
           "write_path_icechunk"]
