"""``woof hex adapt`` -- adapt the next cycle's mesh to this cycle's forecast.

WHAT THIS DOES.  A cycle's hex history (``cuda-history.*.nc``) is read on
its own mesh, a set of selectable refinement criteria turns it into a
per-cell refinement demand, the demands are combined with weights into a
target cell spacing on a regular latitude/longitude raster, the raster is
gradient-limited and written in the ``woof-hex.density.v1`` contract, and a
hysteresis rule decides whether the target moved far enough from the mesh
the cycle already has to be worth regenerating.  The output is a folder
(``next-spec/``) holding the raster, the regional window and cull shape, a
state document the next cycle continues from, and ``adapt-plan.json``: the
argv lists of every command the next cycle runs and a receipt of why.

Nothing here generates a mesh or opens a device.  ``woof hex cycle run
--adaptive`` is what executes the plan.

THE CRITERIA (each one a row in :data:`CRITERIA`, selected with
``--criteria``):

* ``wind``           -- 10 m wind speed (``u10``/``v10``), or the wind at
                        ``--wind-level`` when the history has no 10 m wind;
* ``shear``          -- the vector wind difference between model level 0
                        and ``--shear-level`` (a level INDEX: the CUDA
                        history carries no ``zgrid``);
* ``icing``          -- liquid water (``qc + qr``) at temperatures below
                        0 C in the lowest ``--icing-levels`` model levels,
                        the conductor-height band of an overhead line;
* ``precip``         -- precipitation rate from consecutive accumulations
                        (``rainnc + rainc``) and the frames' own valid times;
* ``theta-gradient`` -- ``|grad theta|`` at model level 0;
* ``wind-gradient``  -- ``|grad wind speed|`` at model level 0 (10 m when
                        present);
* ``assets``         -- distance to the energy assets of a ``sites.v1`` or
                        ``assets.v1`` document (:mod:`woof.energy.contracts`).

Every criterion maps its field onto a demand in [0, 1] with a two-number
ramp ``LO:HI`` (``--threshold NAME=LO:HI``; for ``assets`` the ramp is a
distance and runs the other way).  The combined demand is the weighted
maximum (``--weight NAME=W``), and the target spacing is the log-linear
interpolation ``background * (fine / background) ** demand``.

WHAT IS APPROXIMATE, SAID ONCE.  Gradients are a least-squares plane fit
over each cell's seven nearest cell centres on the local tangent plane, not
the model's own C-grid operator.  Rasterisation is nearest-cell, limited to
pixels within one cell spacing of a cell of the mesh that carried the field.
The Courant preflight here is on the REQUESTED minimum spacing; the forecast
door's own Courant gate on the delivered ``dcEdge`` is the authority.  The
cell estimate is an area integral over hexagons of the requested spacing.

PYTHON BOUNDARY (``docs/dev/static-rust-port.md``).  The field-wide work
here -- criteria over every cell, a KD-tree rasterisation and the gradient
limiter -- is numpy/scipy, by the same precedent as the swath detector
(:mod:`woof.hex.swath.detect`), which reads the same history files and runs
its detectors over every cell in numpy.  It is a candidate for the Rust
density crate once that exists.

DENSITY WRITER.  The ``woof-hex.density.v1`` writer and gradient limiter
are owned by ``woof.hex.density`` (``density_from_fields``).  When that
module is importable with the contract's signature it is used; otherwise
the minimal writer and limiter in this module, which implement the same
contract, are used and the plan says which one ran.  Replace the internal
pair with ``woof.hex.density`` once both are on one branch.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import glob as _glob
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

from .errors import MpasPortError

PLAN_SCHEMA = "woof-hex.adapt-plan.v1"
STATE_SCHEMA = "woof-hex.adapt-state.v1"
DENSITY_SCHEMA = "woof-hex.density.v1"
#: The state a next cycle continues from, written by :func:`commit_state`
#: once the plan's mesh stages have succeeded.
STATE_NAME = "adapt-state.json"
#: The state as planned, before any stage ran.  Never read by ``--state``.
PROPOSED_STATE_NAME = "adapt-state.proposed.json"

#: Sphere radius MPAS meshes are generated on, km.
EARTH_RADIUS_KM = 6371.229
#: Area of one hexagonal Voronoi cell of centre-to-centre spacing ``s`` is
#: ``HEX_AREA_FACTOR * s**2``.
HEX_AREA_FACTOR = math.sqrt(3.0) / 2.0
#: Boundary rings a limited-area cull carries (relaxation + specified zone).
BOUNDARY_RINGS = 7
#: The largest raster this door writes.  Past it the limiter and the
#: KD-tree lookups are minutes of numpy; a coarser --raster-km is the fix.
MAX_RASTER_PIXELS = 6_000_000
#: The timestep-evidence label a sub-anchor dt carries (unit 1's contract).
EXPERIMENTAL_DT_EVIDENCE = "experimental-unanchored"
#: The registry the forecast door reads runtime mesh rows from.
MESH_ROWS_ENVIRONMENT = "WOOF_HEX_MESH_ROWS"

_LABEL = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})[.:](\d{2})[.:](\d{2})")
_STAMP_FORMAT = "%Y-%m-%d_%H:%M:%S"


class AdaptRefusal(MpasPortError):
    """``woof hex adapt`` will not plan this, and the message says why."""


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Criterion:
    name: str
    units: str
    lo: float
    hi: float
    meaning: str
    #: True when the demand rises as the field FALLS (a distance).
    inverted: bool = False


CRITERIA: Mapping[str, Criterion] = {
    row.name: row
    for row in (
        Criterion("wind", "m s-1", 15.0, 25.0,
                  "10 m wind speed (or the --wind-level wind), max over frames"),
        Criterion("shear", "m s-1", 10.0, 20.0,
                  "|V(--shear-level) - V(level 0)|, max over frames"),
        Criterion("icing", "kg kg-1", 1.0e-5, 2.0e-4,
                  "qc + qr where T < 273.15 K in the lowest --icing-levels "
                  "levels, max over levels and frames"),
        Criterion("precip", "mm h-1", 1.0, 10.0,
                  "rate of rainnc + rainc between consecutive frames, max"),
        Criterion("theta-gradient", "K km-1", 0.05, 0.2,
                  "|grad theta| at level 0, max over frames"),
        Criterion("wind-gradient", "m s-1 km-1", 0.1, 0.5,
                  "|grad wind speed| at 10 m or level 0, max over frames"),
        Criterion("assets", "km", 3.0, 6.0,
                  "distance to the nearest asset point or line vertex; "
                  "demand 1 inside LO, 0 beyond HI", inverted=True),
    )
}
#: ``--criteria gradients`` names both gradient rows.
CRITERIA_ALIASES: Mapping[str, tuple[str, ...]] = {
    "gradients": ("theta-gradient", "wind-gradient"),
}

#: History variable names, first match wins (both dialects
#: :mod:`woof.energy.sample_hex` reads).
SOURCES: Mapping[str, tuple[str, ...]] = {
    "u": ("u_zonal", "uReconstructZonal"),
    "v": ("v_meridional", "uReconstructMeridional"),
    "theta": ("theta",),
    "pressure": ("pressure",),
    "qc": ("qc",),
    "qr": ("qr",),
    "u10": ("u10",),
    "v10": ("v10",),
    "t2": ("t2", "t2m"),
    "rainnc": ("rainnc",),
    "rainc": ("rainc",),
}

#: theta -> T exponent, R_d / c_p as MPAS carries it.
KAPPA = 287.0 / 1004.5
FREEZING_K = 273.15


def resolve_criteria(values: Sequence[str]) -> tuple[str, ...]:
    """``--criteria`` values (comma lists, repeatable) to unique row names."""

    chosen: list[str] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            names = CRITERIA_ALIASES.get(token, (token,))
            for name in names:
                if name not in CRITERIA:
                    raise AdaptRefusal(
                        f"--criteria names {token!r}, which is not a criterion "
                        f"this build has.  The rows are "
                        f"{sorted(CRITERIA) + sorted(CRITERIA_ALIASES)}"
                    )
                if name not in chosen:
                    chosen.append(name)
    if not chosen:
        raise AdaptRefusal(
            "--criteria named nothing.  A target spacing with no criterion "
            "behind it is the background everywhere, and planning a remesh "
            "from that would regenerate a uniform mesh nobody asked for"
        )
    return tuple(chosen)


def _parse_pairs(values: Sequence[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values or ():
        name, sep, rest = str(value).partition("=")
        name = name.strip()
        if not sep or name not in CRITERIA:
            raise AdaptRefusal(
                f"{flag} {value!r} is not NAME=VALUE with NAME one of "
                f"{sorted(CRITERIA)}"
            )
        out[name] = rest.strip()
    return out


def criterion_ramps(
    criteria: Sequence[str], thresholds: Sequence[str] = (),
    weights: Sequence[str] = (),
) -> dict[str, dict[str, float]]:
    """Each chosen criterion's ``lo``, ``hi`` and ``weight``."""

    given = _parse_pairs(thresholds, "--threshold")
    given_weights = _parse_pairs(weights, "--weight")
    stray = sorted((set(given) | set(given_weights)) - set(criteria))
    if stray:
        raise AdaptRefusal(
            f"--threshold/--weight name {stray}, which --criteria did not "
            f"select.  A tuning for a criterion that does not run would be "
            f"recorded as if it shaped the mesh"
        )
    ramps: dict[str, dict[str, float]] = {}
    for name in criteria:
        row = CRITERIA[name]
        lo, hi = row.lo, row.hi
        if name in given:
            text = given[name]
            parts = text.split(":")
            try:
                lo, hi = float(parts[0]), float(parts[1])
            except (IndexError, ValueError) as error:
                raise AdaptRefusal(
                    f"--threshold {name}={text} is not LO:HI"
                ) from error
            if len(parts) != 2:
                raise AdaptRefusal(f"--threshold {name}={text} is not LO:HI")
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo or lo < 0.0:
            raise AdaptRefusal(
                f"criterion {name}: the ramp must satisfy 0 <= LO < HI, got "
                f"{lo}:{hi}"
            )
        weight = 1.0
        if name in given_weights:
            try:
                weight = float(given_weights[name])
            except ValueError as error:
                raise AdaptRefusal(
                    f"--weight {name}={given_weights[name]} is not a number"
                ) from error
        if not math.isfinite(weight) or not 0.0 < weight <= 1.0:
            raise AdaptRefusal(
                f"--weight {name}={weight}: a weight scales a demand in [0, 1] "
                f"and must lie in (0, 1]"
            )
        ramps[name] = {"lo": lo, "hi": hi, "weight": weight}
    return ramps


def ramp(values, lo: float, hi: float, *, inverted: bool = False):
    """A field mapped onto demand in [0, 1]; NaN is no demand."""

    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if inverted:
        demand = (hi - values) / (hi - lo)
    else:
        demand = (values - lo) / (hi - lo)
    demand = np.clip(demand, 0.0, 1.0)
    return np.where(np.isfinite(demand), demand, 0.0)


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------
@dataclass
class Frame:
    path: Path
    valid: datetime | None


@dataclass
class MeshGroup:
    """Every frame written on one mesh, in valid-time order."""

    key: str
    lat: Any  # radians, float64 (nCells,)
    lon: Any
    frames: list[Frame]
    #: KD-tree products that depend only on the mesh (and raster), built
    #: once per group rather than once per frame and criterion.
    cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_cells(self) -> int:
        return int(self.lat.shape[0])


def _label_time(path: Path) -> datetime | None:
    match = _LABEL.search(path.name)
    if match is None:
        return None
    return datetime.strptime(
        f"{match.group(1)}_{match.group(2)}:{match.group(3)}:{match.group(4)}",
        _STAMP_FORMAT,
    )


def expand_history(patterns: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        matches = sorted(_glob.glob(str(pattern)))
        if not matches and Path(pattern).is_file():
            matches = [str(pattern)]
        for item in matches:
            path = Path(item)
            if path.is_file() and path not in found:
                found.append(path)
    if not found:
        raise AdaptRefusal(
            f"--history {list(patterns)} matched no file.  The criteria read "
            f"a forecast this project wrote; with no frame there is nothing to "
            f"adapt to"
        )
    return found


def _read_coordinates(path: Path):
    import numpy as np
    from netCDF4 import Dataset

    with Dataset(str(path)) as ds:
        ds.set_auto_mask(False)
        for name in ("latCell", "lonCell"):
            if name not in ds.variables:
                raise AdaptRefusal(
                    f"{path} carries no {name}.  Every hex history frame "
                    f"carries the mesh's cell centres; a file without them is "
                    f"not one, and its fields cannot be placed"
                )
        lat = np.asarray(ds.variables["latCell"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lonCell"][:], dtype=np.float64)
    if lat.ndim != 1 or lat.shape != lon.shape or lat.size == 0:
        raise AdaptRefusal(f"{path}: latCell/lonCell are not one nCells vector")
    # MPAS stores radians.  A file in degrees is detectable and converted.
    if float(np.nanmax(np.abs(lat))) > math.pi / 2.0 + 1.0e-6:
        lat, lon = np.radians(lat), np.radians(lon)
    return lat, lon


def group_frames(paths: Sequence[Path]) -> list[MeshGroup]:
    """Frames grouped by the mesh they were written on."""

    groups: dict[str, MeshGroup] = {}
    for path in paths:
        lat, lon = _read_coordinates(path)
        digest = hashlib.sha256()
        digest.update(lat.astype("<f8").tobytes())
        digest.update(lon.astype("<f8").tobytes())
        key = f"{lat.size}:{digest.hexdigest()[:16]}"
        group = groups.get(key)
        if group is None:
            group = groups[key] = MeshGroup(key, lat, lon, [])
        group.frames.append(Frame(path, _label_time(path)))
    for group in groups.values():
        group.frames.sort(key=lambda frame: (frame.valid or datetime.min,
                                             frame.path.name))
    return list(groups.values())


def _resolve_name(names: set[str], key: str) -> str | None:
    for name in SOURCES[key]:
        if name in names:
            return name
    return None


def _read(path: Path, key: str, *, levels: int | None = None, level: int | None = None):
    """One variable for one frame, (nCells,) or (nCells, levels)."""

    import numpy as np
    from netCDF4 import Dataset

    with Dataset(str(path)) as ds:
        ds.set_auto_mask(False)
        name = _resolve_name(set(ds.variables), key)
        if name is None:
            return None
        variable = ds.variables[name]
        dims = variable.dimensions
        index: list[Any] = []
        for dim in dims:
            if dim == "Time":
                index.append(0)
            elif dim == "nCells":
                index.append(slice(None))
            elif dim in ("nVertLevels", "nVertLevelsP1"):
                if level is not None:
                    if level >= ds.dimensions[dim].size:
                        raise AdaptRefusal(
                            f"{path}: level {level} requested of {name}, which "
                            f"has {ds.dimensions[dim].size}"
                        )
                    index.append(level)
                elif levels is not None:
                    index.append(slice(0, min(levels, ds.dimensions[dim].size)))
                else:
                    index.append(slice(None))
            else:
                index.append(0)
        data = np.asarray(variable[tuple(index)], dtype=np.float64)
    return data


def _names(path: Path) -> set[str]:
    from netCDF4 import Dataset

    with Dataset(str(path)) as ds:
        return set(ds.variables)


def tangent_gradient(lat, lon, values, *, neighbours: int = 7,
                     cache: dict | None = None):
    """``|grad f|`` per km by a least-squares plane over nearest cells."""

    import numpy as np
    from scipy.spatial import cKDTree

    xyz = _xyz(lat, lon)
    k = min(int(neighbours), xyz.shape[0])
    if k < 3:
        return np.zeros(xyz.shape[0])
    key = ("neighbours", k)
    index = None if cache is None else cache.get(key)
    if index is None:
        _, index = cKDTree(xyz).query(xyz, k=k)
        index = index[:, 1:]
        if cache is not None:
            cache[key] = index
    east = np.column_stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)])
    north = np.column_stack([-np.sin(lat) * np.cos(lon),
                             -np.sin(lat) * np.sin(lon), np.cos(lat)])
    delta = xyz[index] - xyz[:, None, :]
    dx = np.einsum("nkc,nc->nk", delta, east) * EARTH_RADIUS_KM
    dy = np.einsum("nkc,nc->nk", delta, north) * EARTH_RADIUS_KM
    values = np.asarray(values, dtype=np.float64)
    df = values[index] - values[:, None]
    valid = np.isfinite(df)
    dx = np.where(valid, dx, 0.0)
    dy = np.where(valid, dy, 0.0)
    df = np.where(valid, df, 0.0)
    sxx, syy, sxy = (dx * dx).sum(1), (dy * dy).sum(1), (dx * dy).sum(1)
    sxf, syf = (dx * df).sum(1), (dy * df).sum(1)
    det = sxx * syy - sxy * sxy
    with np.errstate(divide="ignore", invalid="ignore"):
        gx = (syy * sxf - sxy * syf) / det
        gy = (sxx * syf - sxy * sxf) / det
    magnitude = np.hypot(gx, gy)
    return np.where(np.isfinite(magnitude), magnitude, np.nan)


def _xyz(lat, lon):
    import numpy as np

    return np.column_stack([np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon), np.sin(lat)])


@dataclass(frozen=True)
class FieldOptions:
    wind_level: int = 0
    shear_level: int = 10
    icing_levels: int = 4


def criterion_fields(
    group: MeshGroup, criteria: Sequence[str], options: FieldOptions,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Per-cell criterion field (max over frames) and how it was formed.

    Missing inputs refuse by name: a criterion silently reading zero would
    plan a mesh coarse exactly where the user asked for resolution.
    """

    import numpy as np

    n = group.n_cells
    fields: dict[str, Any] = {}
    notes: dict[str, Any] = {}
    first = group.frames[0].path
    names = _names(first)

    def need(key: str, criterion: str) -> None:
        if _resolve_name(names, key) is None:
            raise AdaptRefusal(
                f"criterion {criterion} needs {SOURCES[key]} and {first} "
                f"carries none of them"
            )

    def running_max(name: str, value) -> None:
        current = fields.get(name)
        fields[name] = value if current is None else np.fmax(current, value)

    def wind_pair(path: Path):
        if _resolve_name(names, "u10") and _resolve_name(names, "v10"):
            return _read(path, "u10"), _read(path, "v10"), "u10/v10"
        need("u", "wind")
        need("v", "wind")
        return (_read(path, "u", level=options.wind_level),
                _read(path, "v", level=options.wind_level),
                f"u_zonal/v_meridional level {options.wind_level}")

    for name in criteria:
        if name == "assets":
            continue
        if name == "precip":
            need("rainnc", name)
            timed = [frame for frame in group.frames if frame.valid is not None]
            if len(timed) < 2:
                # This mesh cannot form a rate; another mesh in the same
                # call may.  build_plan refuses when none can.
                fields[name] = np.full(n, np.nan)
                notes[name] = {"source": "rainnc + rainc", "frames": len(timed),
                               "skipped": "fewer than two timed frames"}
                continue
            previous = None
            for frame in timed:
                total = _read(frame.path, "rainnc")
                rainc = _read(frame.path, "rainc")
                if rainc is not None:
                    total = total + rainc
                if previous is not None:
                    hours = (frame.valid - previous[0]).total_seconds() / 3600.0
                    if hours <= 0.0:
                        raise AdaptRefusal(
                            f"two frames on mesh {group.key} share valid time "
                            f"{frame.valid}; a rate over zero hours is not one"
                        )
                    rate = np.maximum(total - previous[1], 0.0) / hours
                    running_max(name, rate)
                previous = (frame.valid, total)
            notes[name] = {"source": "rainnc + rainc", "frames": len(timed)}
            continue
        for frame in group.frames:
            path = frame.path
            if name == "wind":
                u, v, source = wind_pair(path)
                running_max(name, np.hypot(u, v))
                notes[name] = {"source": source}
            elif name == "wind-gradient":
                u, v, source = wind_pair(path)
                running_max(name, tangent_gradient(group.lat, group.lon,
                                                   np.hypot(u, v), cache=group.cache))
                notes[name] = {"source": source, "operator": "tangent-plane LSQ, 7 cells"}
            elif name == "shear":
                need("u", name)
                need("v", name)
                du = (_read(path, "u", level=options.shear_level)
                      - _read(path, "u", level=0))
                dv = (_read(path, "v", level=options.shear_level)
                      - _read(path, "v", level=0))
                running_max(name, np.hypot(du, dv))
                notes[name] = {"levels": [0, options.shear_level],
                               "note": "level indices; history carries no zgrid"}
            elif name == "theta-gradient":
                need("theta", name)
                theta = _read(path, "theta", level=0)
                running_max(name, tangent_gradient(group.lat, group.lon, theta,
                                                   cache=group.cache))
                notes[name] = {"source": "theta level 0",
                               "operator": "tangent-plane LSQ, 7 cells"}
            elif name == "icing":
                need("qc", name)
                levels = int(options.icing_levels)
                liquid = _read(path, "qc", levels=levels)
                qr = _read(path, "qr", levels=levels)
                if qr is not None:
                    liquid = liquid + qr
                if liquid.ndim == 1:
                    liquid = liquid[:, None]
                if (_resolve_name(names, "theta") is not None
                        and _resolve_name(names, "pressure") is not None):
                    theta = _read(path, "theta", levels=levels)
                    pressure = _read(path, "pressure", levels=levels)
                    if theta.ndim == 1:
                        theta, pressure = theta[:, None], pressure[:, None]
                    temperature = theta * (pressure / 1.0e5) ** KAPPA
                    source = f"theta*(p/1e5)^kappa, lowest {liquid.shape[1]} levels"
                elif _resolve_name(names, "t2") is not None:
                    temperature = np.repeat(_read(path, "t2")[:, None],
                                            liquid.shape[1], axis=1)
                    source = "t2 applied to the lowest levels (no theta/pressure)"
                else:
                    raise AdaptRefusal(
                        "criterion icing needs a temperature: theta with "
                        f"pressure, or t2; {first} carries neither"
                    )
                supercooled = np.where(temperature < FREEZING_K, liquid, 0.0)
                running_max(name, np.max(supercooled, axis=1))
                notes[name] = {
                    "liquid": "qc + qr" if qr is not None else "qc",
                    "temperature": source,
                }
        if name not in fields:
            fields[name] = np.full(n, np.nan)
    return fields, notes


# ---------------------------------------------------------------------------
# assets
# ---------------------------------------------------------------------------
def asset_points(
    sites: Path | None, assets: Path | None, *, densify_km: float,
) -> tuple[Any, Any, dict[str, Any]]:
    """Latitude/longitude (degrees) of every asset point, lines densified."""

    import numpy as np

    lats: list[float] = []
    lons: list[float] = []
    receipt: dict[str, Any] = {}
    from woof.energy import contracts

    if sites is not None:
        try:
            site_set = contracts.load_sites(sites)
        except contracts.ContractError as error:
            raise AdaptRefusal(f"--sites {sites}: {error}") from error
        lats += [site.lat for site in site_set.sites]
        lons += [site.lon for site in site_set.sites]
        receipt["sites"] = {"path": str(sites), "count": len(site_set),
                            "sha256": contracts.sha256_file(sites)}
    if assets is not None:
        try:
            collection = contracts.load_assets(assets)
        except contracts.ContractError as error:
            raise AdaptRefusal(f"--assets {assets}: {error}") from error
        count = 0
        for asset in collection.assets:
            for line in _geometry_lines(asset.geometry):
                for (lon0, lat0), (lon1, lat1) in zip(line, line[1:] or line):
                    if (lon0, lat0) == (lon1, lat1):
                        lats.append(lat0)
                        lons.append(lon0)
                        continue
                    length = _great_circle_km(lat0, lon0, lat1, lon1)
                    steps = max(1, int(math.ceil(length / densify_km)))
                    for t in np.linspace(0.0, 1.0, steps + 1):
                        lats.append(lat0 + t * (lat1 - lat0))
                        lons.append(lon0 + t * (lon1 - lon0))
            count += 1
        receipt["assets"] = {"path": str(assets), "count": count,
                             "sha256": contracts.sha256_file(assets)}
    if (sites is not None or assets is not None) and not lats:
        raise AdaptRefusal("the --sites/--assets documents hold no positions")
    return np.asarray(lats, float), np.asarray(lons, float), receipt


def _geometry_lines(geometry: Mapping[str, Any]) -> list[list[tuple[float, float]]]:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if kind == "Point":
        return [[(float(coordinates[0]), float(coordinates[1]))]]
    if kind == "MultiPoint":
        return [[(float(x), float(y))] for x, y, *_ in coordinates]
    if kind == "LineString":
        return [[(float(x), float(y)) for x, y, *_ in coordinates]]
    if kind in ("MultiLineString", "Polygon"):
        return [[(float(x), float(y)) for x, y, *_ in part] for part in coordinates]
    if kind == "MultiPolygon":
        return [[(float(x), float(y)) for x, y, *_ in ring]
                for polygon in coordinates for ring in polygon]
    return []


def _great_circle_km(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    from .swath.geometry import great_circle_km

    return great_circle_km(lat0, lon0, lat1, lon1)


# ---------------------------------------------------------------------------
# the raster
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Box:
    lat0: float
    lat1: float
    lon0: float
    lon1: float

    def expanded(self, km: float) -> "Box":
        dlat = km / (EARTH_RADIUS_KM * math.pi / 180.0)
        mid = math.radians(0.5 * (self.lat0 + self.lat1))
        dlon = dlat / max(math.cos(mid), 0.05)
        return Box(max(-89.0, self.lat0 - dlat), min(89.0, self.lat1 + dlat),
                   self.lon0 - dlon, self.lon1 + dlon)

    def contains(self, other: "Box", *, slack_deg: float = 1.0e-6) -> bool:
        return (self.lat0 - slack_deg <= other.lat0
                and other.lat1 <= self.lat1 + slack_deg
                and self.lon0 - slack_deg <= other.lon0
                and other.lon1 <= self.lon1 + slack_deg)

    def ring(self) -> list[list[float]]:
        """Counter-clockwise [lat, lon] vertices, not closed."""

        return [[round(self.lat0, 6), round(self.lon0, 6)],
                [round(self.lat0, 6), round(self.lon1, 6)],
                [round(self.lat1, 6), round(self.lon1, 6)],
                [round(self.lat1, 6), round(self.lon0, 6)]]

    def as_list(self) -> list[float]:
        return [round(v, 6) for v in (self.lat0, self.lat1, self.lon0, self.lon1)]

    @classmethod
    def around(cls, lats, lons) -> "Box":
        import numpy as np

        lons = (np.asarray(lons, float) + 180.0) % 360.0 - 180.0
        return cls(float(np.min(lats)), float(np.max(lats)),
                   float(np.min(lons)), float(np.max(lons)))


def parse_bbox(text: str) -> Box:
    try:
        lat0, lat1, lon0, lon1 = (float(item) for item in str(text).split(","))
    except ValueError as error:
        raise AdaptRefusal(f"--bbox {text!r} is not LAT0,LAT1,LON0,LON1") from error
    if not (-90.0 <= lat0 < lat1 <= 90.0) or not lon0 < lon1:
        raise AdaptRefusal(f"--bbox {text!r}: need LAT0 < LAT1 and LON0 < LON1")
    return Box(lat0, lat1, lon0, lon1)


def raster_axes(box: Box, raster_km: float):
    import numpy as np

    deg_per_km = 180.0 / (math.pi * EARTH_RADIUS_KM)
    dlat = raster_km * deg_per_km
    mid = math.radians(0.5 * (box.lat0 + box.lat1))
    dlon = dlat / max(math.cos(mid), 0.05)
    ny = max(2, int(math.ceil((box.lat1 - box.lat0) / dlat)) + 1)
    nx = max(2, int(math.ceil((box.lon1 - box.lon0) / dlon)) + 1)
    if ny * nx > MAX_RASTER_PIXELS:
        raise AdaptRefusal(
            f"the raster over {box.as_list()} at {raster_km:g} km is "
            f"{ny} x {nx} = {ny * nx:,} pixels, past the {MAX_RASTER_PIXELS:,} "
            f"this door writes.  Pass a coarser --raster-km (the spacing field "
            f"is gradient-limited, so a pixel coarser than the finest cell "
            f"loses nothing but the corner of a ramp) or a smaller --bbox"
        )
    lat = box.lat0 + dlat * np.arange(ny)
    lon = box.lon0 + dlon * np.arange(nx)
    return lat, lon


def pixel_areas_km2(lat, lon):
    """(ny, nx) cell areas of a regular lat/lon raster centred on the axes."""

    import numpy as np

    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    dlat = float(lat[1] - lat[0]) if lat.size > 1 else 0.0
    dlon = float(lon[1] - lon[0]) if lon.size > 1 else 0.0
    south = np.radians(np.clip(lat - dlat / 2.0, -90.0, 90.0))
    north = np.radians(np.clip(lat + dlat / 2.0, -90.0, 90.0))
    band = EARTH_RADIUS_KM ** 2 * (np.sin(north) - np.sin(south)) * math.radians(dlon)
    return np.repeat(band[:, None], lon.size, axis=1)


def rasterize_cells(group: MeshGroup, values, lat, lon):
    """Nearest-cell values on raster pixels, NaN off the mesh."""

    import numpy as np
    from scipy.spatial import cKDTree

    shape = (len(lat), len(lon))
    key = ("pixels", shape, float(lat[0]), float(lat[-1]), float(lon[0]), float(lon[-1]))
    cached = group.cache.get(key)
    if cached is None:
        cells = _xyz(group.lat, group.lon)
        tree = cKDTree(cells)
        if cells.shape[0] > 1:
            spacing, _ = tree.query(cells, k=2)
            spacing = spacing[:, 1]
        else:
            spacing = np.array([np.inf])
        grid_lat, grid_lon = np.meshgrid(np.radians(lat), np.radians(lon), indexing="ij")
        distance, index = tree.query(_xyz(grid_lat.ravel(), grid_lon.ravel()), k=1)
        cached = group.cache[key] = (index, distance <= spacing[index])
    index, covered = cached
    out = np.where(covered, np.asarray(values, float)[index], np.nan)
    return out.reshape(shape), covered.reshape(shape)


def asset_distance_km(points_lat, points_lon, lat, lon):
    import numpy as np
    from scipy.spatial import cKDTree

    tree = cKDTree(_xyz(np.radians(points_lat), np.radians(points_lon)))
    grid_lat, grid_lon = np.meshgrid(np.radians(lat), np.radians(lon), indexing="ij")
    chord, _ = tree.query(_xyz(grid_lat.ravel(), grid_lon.ravel()), k=1)
    angle = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    return (angle * EARTH_RADIUS_KM).reshape(grid_lat.shape)


def limit_gradient(spacing, lat, lon, max_gradient: float, *, max_iterations: int = 500):
    """Lower ``spacing`` until no neighbour pair differs by more than
    ``max_gradient * distance``, on the eight-neighbour stencil.

    Only ever REFINES (the result is pointwise <= the input), so every
    region a criterion asked for keeps at least the resolution it asked for
    and the ramp out of it is paid for in coarser cells.  ``max_gradient`` is
    dimensionless: km of spacing change per km of distance, which is also the
    fractional change per cell (the ``%/cell`` the mesh gates print).
    """

    import numpy as np

    s = np.array(spacing, dtype=np.float64, copy=True)
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    km_per_deg = math.pi * EARTH_RADIUS_KM / 180.0
    dy = float(lat[1] - lat[0]) * km_per_deg
    dx_row = float(lon[1] - lon[0]) * km_per_deg * np.cos(np.radians(lat))
    g = float(max_gradient)
    ny, nx = s.shape
    ix = np.arange(nx, dtype=np.float64)
    iy = np.arange(ny, dtype=np.float64)
    diag = np.hypot(np.minimum(dx_row[1:], dx_row[:-1]), dy)[:, None]
    for iteration in range(int(max_iterations)):
        before = s.copy()
        step = (g * dx_row)[:, None] * ix[None, :]
        s = np.minimum(s, step + np.minimum.accumulate(s - step, axis=1))
        rev = s[:, ::-1]
        rev = np.minimum(rev, step + np.minimum.accumulate(rev - step, axis=1))
        s = rev[:, ::-1]
        step = (g * dy) * iy[:, None]
        s = np.minimum(s, step + np.minimum.accumulate(s - step, axis=0))
        rev = s[::-1, :]
        rev = np.minimum(rev, step + np.minimum.accumulate(rev - step, axis=0))
        s = rev[::-1, :]
        reach = g * diag
        s[1:, 1:] = np.minimum(s[1:, 1:], s[:-1, :-1] + reach)
        s[1:, :-1] = np.minimum(s[1:, :-1], s[:-1, 1:] + reach)
        s[:-1, 1:] = np.minimum(s[:-1, 1:], s[1:, :-1] + reach)
        s[:-1, :-1] = np.minimum(s[:-1, :-1], s[1:, 1:] + reach)
        if np.array_equal(s, before):
            return s, iteration + 1
    raise AdaptRefusal(
        f"the gradient limiter did not converge in {max_iterations} sweeps; "
        f"the raster is not smooth enough for a mesh generator to follow"
    )


def delivered_gradient(spacing, lat, lon) -> float:
    """The steepest neighbour-pair spacing gradient on the raster."""

    import numpy as np

    km_per_deg = math.pi * EARTH_RADIUS_KM / 180.0
    dy = float(lat[1] - lat[0]) * km_per_deg
    dx_row = float(lon[1] - lon[0]) * km_per_deg * np.cos(np.radians(lat))
    worst = 0.0
    worst = max(worst, float(np.max(np.abs(np.diff(spacing, axis=1)) / dx_row[:, None])))
    worst = max(worst, float(np.max(np.abs(np.diff(spacing, axis=0)) / dy)))
    diag = np.hypot(np.minimum(dx_row[1:], dx_row[:-1]), dy)[:, None]
    worst = max(worst, float(np.max(np.abs(spacing[1:, 1:] - spacing[:-1, :-1]) / diag)))
    worst = max(worst, float(np.max(np.abs(spacing[1:, :-1] - spacing[:-1, 1:]) / diag)))
    return worst


def write_density_raster(path: Path, lat, lon, spacing_km, *,
                         attributes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write ``woof-hex.density.v1``: CF netCDF, 1-D ascending lat/lon in
    degrees, ``spacing_km`` float64 (lat, lon), schema and min_spacing_km."""

    import numpy as np
    from netCDF4 import Dataset

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    spacing = np.asarray(spacing_km, dtype=np.float64)
    _check_raster(lat, lon, spacing, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with Dataset(str(temporary), "w", format="NETCDF4") as ds:
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", lon.size)
        variable = ds.createVariable("lat", "f8", ("lat",))
        variable.units = "degrees_north"
        variable.standard_name = "latitude"
        variable[:] = lat
        variable = ds.createVariable("lon", "f8", ("lon",))
        variable.units = "degrees_east"
        variable.standard_name = "longitude"
        variable[:] = lon
        variable = ds.createVariable("spacing_km", "f8", ("lat", "lon"))
        variable.units = "km"
        variable.long_name = "target MPAS cell centre-to-centre spacing"
        variable[:] = spacing
        ds.Conventions = "CF-1.8"
        ds.schema = DENSITY_SCHEMA
        ds.min_spacing_km = float(np.min(spacing))
        for key, value in (attributes or {}).items():
            ds.setncattr(key, value)
    temporary.replace(path)
    return {"path": str(path), "sha256": _sha256(path),
            "min_spacing_km": float(np.min(spacing)),
            "max_spacing_km": float(np.max(spacing)),
            "shape": [int(lat.size), int(lon.size)]}


def _check_raster(lat, lon, spacing, where: str) -> None:
    import numpy as np

    if lat.ndim != 1 or lon.ndim != 1 or lat.size < 2 or lon.size < 2:
        raise AdaptRefusal(f"{where}: lat/lon must be 1-D with at least two values")
    if not (np.all(np.diff(lat) > 0.0) and np.all(np.diff(lon) > 0.0)):
        raise AdaptRefusal(f"{where}: lat and lon must be strictly ascending")
    if spacing.shape != (lat.size, lon.size):
        raise AdaptRefusal(
            f"{where}: spacing_km is {spacing.shape}, the axes say "
            f"{(lat.size, lon.size)}"
        )
    if not np.all(np.isfinite(spacing)) or float(np.min(spacing)) <= 0.0:
        raise AdaptRefusal(f"{where}: spacing_km must be finite and positive")


def read_density_raster(path: Path):
    """Read and validate a ``woof-hex.density.v1`` raster."""

    import numpy as np
    from netCDF4 import Dataset

    try:
        ds = Dataset(str(path))
    except OSError as error:
        raise AdaptRefusal(f"{path} is not a readable density raster: {error}") from error
    with ds:
        ds.set_auto_mask(False)
        schema = getattr(ds, "schema", None)
        if schema != DENSITY_SCHEMA:
            raise AdaptRefusal(
                f"{path} declares schema {schema!r}, not {DENSITY_SCHEMA!r}"
            )
        for name in ("lat", "lon", "spacing_km"):
            if name not in ds.variables:
                raise AdaptRefusal(f"{path} has no {name} variable")
        if ds.variables["spacing_km"].dimensions != ("lat", "lon"):
            raise AdaptRefusal(f"{path}: spacing_km must be dimensioned (lat, lon)")
        if ds.variables["spacing_km"].dtype != np.float64:
            raise AdaptRefusal(f"{path}: spacing_km must be float64")
        lat = np.asarray(ds.variables["lat"][:], np.float64)
        lon = np.asarray(ds.variables["lon"][:], np.float64)
        spacing = np.asarray(ds.variables["spacing_km"][:], np.float64)
        stated = getattr(ds, "min_spacing_km", None)
    _check_raster(lat, lon, spacing, str(path))
    if stated is None or not math.isclose(float(stated), float(spacing.min()),
                                          rel_tol=1e-12, abs_tol=0.0):
        raise AdaptRefusal(
            f"{path}: min_spacing_km {stated} disagrees with the field's own "
            f"minimum {float(spacing.min())}"
        )
    return lat, lon, spacing


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def external_density_writer() -> Callable[..., Any] | None:
    """``woof.hex.density.density_from_fields`` when it has the contract's
    signature, else None (the internal writer runs)."""

    try:
        from woof.hex import density as module  # type: ignore[attr-defined]
    except ImportError:
        return None
    function = getattr(module, "density_from_fields", None)
    if function is None:
        return None
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return None
    if not {"lat", "lon", "spacing_km", "out"} <= set(parameters):
        return None
    return function


def estimated_cells(spacing, areas) -> float:
    import numpy as np

    return float(np.sum(areas / (HEX_AREA_FACTOR * np.asarray(spacing) ** 2)))


# ---------------------------------------------------------------------------
# hysteresis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RemeshPolicy:
    """When a moved target is worth a new mesh.

    The same three-part construction as :mod:`woof.hex.swath.hysteresis`:

    1. TWO THRESHOLDS, ASYMMETRIC.  A pixel is UNDER-resolved when the new
       target asks for cells finer than the current mesh by more than
       ``tolerance_ratio``, and OVER-resolved when the current mesh is finer
       than the target by more than it.  Both are weighted by CELLS (the
       new target's cells for under, the current mesh's for over), so a
       narrow fine corridor counts for what it costs rather than for its
       area.  Under-resolution is a forecast missing what the criteria asked
       for, so it remeshes at ``remesh_under_fraction``; over-resolution is
       only wasted card time, so it needs the much larger
       ``remesh_over_fraction``.
    2. DWELL.  A mesh generated fewer than ``minimum_dwell_cycles`` ago is
       kept, unless under-resolution passes ``force_under_fraction`` -- the
       dwell buys cycles of use for the cost of a generation, but never
       hides weather the mesh cannot see.
    3. DOMAIN.  A forecast domain that is no longer inside the current
       mesh's domain always regenerates: it is OUTSIDE-PARENT-REFINEMENT by
       another name, and in the adaptive cycle regeneration is the remedy.
    """

    tolerance_ratio: float = 1.25
    remesh_under_fraction: float = 0.05
    remesh_over_fraction: float = 0.30
    force_under_fraction: float = 0.20
    minimum_dwell_cycles: int = 2

    def validate(self) -> None:
        if not self.tolerance_ratio > 1.0:
            raise AdaptRefusal("--tolerance-ratio must be greater than 1")
        for name in ("remesh_under_fraction", "remesh_over_fraction",
                     "force_under_fraction"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise AdaptRefusal(f"{name} must lie in [0, 1], got {value}")
        if self.force_under_fraction < self.remesh_under_fraction:
            raise AdaptRefusal(
                "force_under_fraction below remesh_under_fraction would let a "
                "dwell-protected mesh regenerate on less evidence than an "
                "unprotected one"
            )
        if self.minimum_dwell_cycles < 0:
            raise AdaptRefusal("minimum_dwell_cycles must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "tolerance_ratio": self.tolerance_ratio,
            "remesh_under_fraction": self.remesh_under_fraction,
            "remesh_over_fraction": self.remesh_over_fraction,
            "force_under_fraction": self.force_under_fraction,
            "minimum_dwell_cycles": self.minimum_dwell_cycles,
        }


def compare_targets(
    current: tuple[Any, Any, Any], target: tuple[Any, Any, Any],
    *, tolerance_ratio: float, background_km: float,
) -> dict[str, float]:
    """Cell-weighted under/over-resolved fractions of ``target`` against the
    raster the current mesh was generated from (interpolated onto the target
    pixels; outside it the current mesh is ``background_km``)."""

    import numpy as np
    from scipy.interpolate import RegularGridInterpolator

    c_lat, c_lon, c_spacing = current
    t_lat, t_lon, t_spacing = target

    def onto(lat, lon, spacing, at_lat, at_lon):
        interpolate = RegularGridInterpolator(
            (lat, lon), np.log(spacing), bounds_error=False, fill_value=np.nan,
        )
        grid_lat, grid_lon = np.meshgrid(at_lat, at_lon, indexing="ij")
        values = np.exp(interpolate(
            np.column_stack([grid_lat.ravel(), grid_lon.ravel()])))
        values = values.reshape(grid_lat.shape)
        return np.where(np.isfinite(values), values, float(background_km))

    # UNDER on the target's pixels: what the new target asks for that the
    # current mesh (background outside its raster) does not have.
    now_on_target = onto(c_lat, c_lon, c_spacing, t_lat, t_lon)
    new_cells = pixel_areas_km2(t_lat, t_lon) / (HEX_AREA_FACTOR * t_spacing ** 2)
    under = t_spacing * tolerance_ratio < now_on_target
    # OVER on the current raster's pixels: fine cells the current mesh pays
    # for that the new target (background outside its raster) no longer
    # wants -- including a corridor the new raster does not reach at all.
    target_on_current = onto(t_lat, t_lon, t_spacing, c_lat, c_lon)
    now_cells = pixel_areas_km2(c_lat, c_lon) / (HEX_AREA_FACTOR * c_spacing ** 2)
    over = c_spacing * tolerance_ratio < target_on_current
    total_new = float(new_cells.sum())
    total_now = float(now_cells.sum())
    return {
        "under_resolved_cell_fraction": float(new_cells[under].sum() / total_new),
        "over_resolved_cell_fraction": float(now_cells[over].sum() / total_now),
        "target_cells_in_raster": total_new,
        "current_cells_in_raster": total_now,
        "cell_change_fraction": (total_new - total_now) / total_now,
    }


def decide(
    state: Mapping[str, Any] | None, comparison: Mapping[str, float] | None,
    *, policy: RemeshPolicy, domain_inside: bool | None,
) -> dict[str, Any]:
    """``generate`` | ``remesh`` | ``keep``, with the reason and numbers."""

    mesh = (state or {}).get("mesh")
    if not mesh:
        return {"action": "generate",
                "reason": "no current mesh: the first adaptive cycle generates one"}
    held = int(mesh.get("cycles_held", 0))
    numbers = dict(comparison or {})
    if domain_inside is False:
        return {"action": "remesh", "cycles_held": held, **numbers,
                "reason": "the new forecast domain is not inside the current "
                          "mesh's domain (OUTSIDE-PARENT-REFINEMENT); "
                          "regeneration is the remedy"}
    under = float(numbers.get("under_resolved_cell_fraction", 0.0))
    over = float(numbers.get("over_resolved_cell_fraction", 0.0))
    if held < policy.minimum_dwell_cycles:
        if under > policy.force_under_fraction:
            return {"action": "remesh", "cycles_held": held, "dwell_protected": True,
                    **numbers,
                    "reason": f"under-resolved {under:.3f} of target cells, past "
                              f"force_under_fraction {policy.force_under_fraction}: "
                              f"dwell does not hide weather the mesh cannot see"}
        return {"action": "keep", "cycles_held": held, "dwell_protected": True,
                **numbers,
                "reason": f"dwell: the mesh has been held {held} cycle(s), fewer "
                          f"than minimum_dwell_cycles {policy.minimum_dwell_cycles}"}
    if under > policy.remesh_under_fraction:
        return {"action": "remesh", "cycles_held": held, "dwell_protected": False,
                **numbers,
                "reason": f"under-resolved {under:.3f} of target cells, past "
                          f"remesh_under_fraction {policy.remesh_under_fraction}"}
    if over > policy.remesh_over_fraction:
        return {"action": "remesh", "cycles_held": held, "dwell_protected": False,
                **numbers,
                "reason": f"over-resolved {over:.3f} of current cells, past "
                          f"remesh_over_fraction {policy.remesh_over_fraction}"}
    return {"action": "keep", "cycles_held": held, "dwell_protected": False,
            **numbers,
            "reason": f"under-resolved {under:.3f} <= {policy.remesh_under_fraction} "
                      f"and over-resolved {over:.3f} <= "
                      f"{policy.remesh_over_fraction}: the current mesh still "
                      f"covers the target, so it is reused"}


def load_state(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise AdaptRefusal(
            f"--state {path} does not exist.  A missing state silently treated "
            f"as the first cycle would regenerate a mesh every cycle"
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AdaptRefusal(f"--state {path} is not readable JSON: {error}") from error
    if document.get("schema") != STATE_SCHEMA:
        raise AdaptRefusal(
            f"--state {path} declares {document.get('schema')!r}, not "
            f"{STATE_SCHEMA!r}"
        )
    if document.get("committed") is not True:
        raise AdaptRefusal(
            f"--state {path} is a PROPOSED state: it names a mesh that did not "
            f"exist when it was written.  Continue from the {STATE_NAME} that "
            f"commit_state writes once the plan's mesh stages have succeeded "
            f"(`woof hex cycle run --adaptive` does this), or the dwell rule "
            f"would keep a mesh nobody generated"
        )
    return document


def commit_state(next_spec: Path) -> Path:
    """Promote ``adapt-state.proposed.json`` to ``adapt-state.json``.

    Call it only after every planned stage up to the forecast has
    succeeded: from then on the mesh the state names exists.
    """

    next_spec = Path(next_spec)
    proposed = next_spec / PROPOSED_STATE_NAME
    try:
        document = json.loads(proposed.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AdaptRefusal(f"{proposed} is not a readable proposed state: {error}") from error
    if document.get("schema") != STATE_SCHEMA:
        raise AdaptRefusal(f"{proposed} is not a {STATE_SCHEMA} document")
    plan_path = next_spec / "adapt-plan.json"
    try:
        refusals = json.loads(plan_path.read_text(encoding="utf-8")).get("refusals")
    except (OSError, ValueError) as error:
        raise AdaptRefusal(f"{plan_path} is not a readable plan: {error}") from error
    if refusals:
        raise AdaptRefusal(
            f"{plan_path} was refused ({'; '.join(refusals)}); its stages never "
            f"ran, so the mesh its state names does not exist"
        )
    document["committed"] = True
    target = next_spec / STATE_NAME
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8", newline="\n")
    return target


# ---------------------------------------------------------------------------
# argv contract checks
# ---------------------------------------------------------------------------
def _no_abbreviations(parser: argparse.ArgumentParser) -> None:
    """argparse accepts a unique PREFIX of a flag by default, so a flag this
    build lacks (``--pbl``) would pass as one it has (``--pbl-cadence``).
    The check parses exact spellings only, on this parser and every
    subparser under it."""

    parser.allow_abbrev = False
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                _no_abbreviations(child)


def check_argv(argv: Sequence[str]) -> dict[str, Any]:
    """Does THIS build's parser accept ``woof <argv>``?

    ``parsed`` -- every flag known and the parse succeeded;
    ``command-not-in-this-build`` -- the subcommand does not exist yet;
    ``flags-not-in-this-build`` -- the command exists but some flags do not;
    ``rejected`` -- the parser refused (a missing required flag, a bad
    choice).  The plan records it so nobody runs a stage whose door has not
    merged; ``woof hex cycle run --adaptive`` refuses before spending
    anything when any stage is not ``parsed``.
    """

    argv = [str(item) for item in argv]
    if not argv:
        return {"status": "rejected", "detail": "empty argv"}
    if argv[0] == "hex":
        from .cli import build_parser

        parser = build_parser()
        rest = argv[1:]
        commands = next(
            action for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        if not rest or rest[0] not in commands.choices:
            return {"status": "command-not-in-this-build",
                    "detail": f"woof hex has no {rest[:1]} command"}
        target, arguments = commands.choices[rest[0]], rest[1:]
        prefix = f"woof hex {rest[0]}"
    elif argv[0] == "mesh":
        from woof.mpas_mesh import build_parser as mesh_parser

        target, arguments, prefix = mesh_parser(), argv[1:], "woof mesh"
    else:
        return {"status": "command-not-in-this-build",
                "detail": f"no parser check for woof {argv[0]}"}
    _no_abbreviations(target)
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            _, unknown = target.parse_known_args(arguments)
    except SystemExit:
        message = stderr.getvalue().strip().splitlines()
        return {"status": "rejected", "detail": message[-1] if message else prefix}
    flags = [item for item in unknown if item.startswith("--")]
    if flags:
        return {"status": "flags-not-in-this-build",
                "detail": f"{prefix} does not know {flags}"}
    if unknown:
        return {"status": "rejected", "detail": f"{prefix}: stray {unknown}"}
    return {"status": "parsed"}


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
@dataclass
class AdaptRequest:
    history: Sequence[str]
    criteria: Sequence[str]
    out: Path
    fine_km: float
    background_km: float
    dt_seconds: float
    thresholds: Sequence[str] = ()
    weights: Sequence[str] = ()
    sites: Path | None = None
    assets: Path | None = None
    bbox: str | None = None
    margin_km: float = 30.0
    window_band_km: float | None = None
    raster_km: float | None = None
    max_gradient_percent: float | None = None
    generation: str = "regional"
    state: Path | None = None
    policy: RemeshPolicy = RemeshPolicy()
    fields: FieldOptions = FieldOptions()
    name: str | None = None
    cycle_index: int | None = None
    start_time: str | None = None
    hours: float = 6.0
    history_every_minutes: int = 30
    experimental_dt: bool = False
    vertical_spec: Path | None = None
    static_highres: bool = True
    from_grid: Path | None = None
    from_state: Path | None = None
    fallback_from_grid: Path | None = None
    fallback_from_state: Path | None = None
    met_dir: Path | None = None
    nfglevels: int | None = None
    extrap_airtemp: str | None = None
    use_spechumd: str | None = None
    les_model: str | None = None
    pbl: str | None = None
    max_cells: int | None = None


def _smoothness_limits() -> dict[str, float]:
    path = Path(__file__).resolve().parents[1] / "data" / "mpas" / "mesh-sizing.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    smoothness = document["smoothness"]
    return {
        "warn": float(smoothness["warn_above_percent_per_cell"]),
        "refuse": float(smoothness["refuse_above_percent_per_cell"]),
    }


def anchored_dt_floor() -> float:
    """The smallest dt the anchor table admits (read only, never edited)."""

    from .dt_admission import ADMITTED_TIMESTEPS

    return float(min(anchor.dt_seconds for anchor in ADMITTED_TIMESTEPS.values()))


def timestep_preflight(dt_seconds: float, min_spacing_km: float, *,
                       experimental: bool, hours: float,
                       history_every_minutes: int) -> dict[str, Any]:
    """Courant on the requested minimum spacing, the clock, and the anchor
    floor.  A dt under the anchored floor needs the explicit experimental
    lane and is labelled ``experimental-unanchored`` everywhere."""

    from .timestep_admission import CourantPolicy

    policy = CourantPolicy()
    if not math.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise AdaptRefusal(f"--dt-seconds {dt_seconds} must be finite and positive")
    maximum = (policy.safety_factor * min_spacing_km * 1000.0
               / policy.max_characteristic_speed_m_s)
    if dt_seconds > maximum:
        raise AdaptRefusal(
            f"--dt-seconds {dt_seconds:g} exceeds the Courant limit "
            f"{maximum:.3f} s at the requested minimum spacing "
            f"{min_spacing_km:g} km ({policy.max_characteristic_speed_m_s:g} m/s, "
            f"safety {policy.safety_factor:g}).  The delivered mesh's dcEdge is "
            f"shorter than the requested spacing, so a dt that fails here "
            f"fails at the forecast door too"
        )
    for what, seconds in (("--hours", hours * 3600.0),
                          ("--history-every-minutes", history_every_minutes * 60.0)):
        steps = seconds / dt_seconds
        if abs(steps - round(steps)) > 1.0e-9 * max(1.0, steps):
            raise AdaptRefusal(
                f"{what} is {seconds:g} s, not a whole number of {dt_seconds:g} s "
                f"steps; the forecast clock would land between steps"
            )
    floor = anchored_dt_floor()
    evidence = "anchor-table"
    if dt_seconds < floor:
        if not experimental:
            raise AdaptRefusal(
                f"--dt-seconds {dt_seconds:g} is below {floor:g} s, the smallest "
                f"dt the anchor table admits.  A sub-anchor timestep runs only on "
                f"the explicit experimental lane: pass --experimental-dt, and the "
                f"plan and the forecast carry timestep_evidence "
                f"{EXPERIMENTAL_DT_EVIDENCE!r}"
            )
        evidence = EXPERIMENTAL_DT_EVIDENCE
    return {
        "dt_seconds": dt_seconds,
        "courant_max_dt_seconds_at_requested_min_spacing": round(maximum, 6),
        "requested_min_spacing_km": min_spacing_km,
        "anchored_dt_floor_seconds": floor,
        "timestep_evidence": evidence,
        "authority": "preflight only; the forecast door's Courant, clock and "
                     "anchor/experimental gates on the delivered dcEdge decide",
    }


def _stamp(moment: datetime) -> str:
    return moment.strftime(_STAMP_FORMAT)


def _grid_box(path: Path) -> Box | None:
    import numpy as np

    lat, lon = _read_coordinates(path)
    lat_deg, lon_deg = np.degrees(lat), np.degrees(lon)
    if lat.size > 1000 and float(np.ptp(lat_deg)) > 170.0:
        return None  # global: covers everything
    return Box.around(lat_deg, lon_deg)


def _remap_source(request: AdaptRequest, cull_box: Box,
                  groups: Sequence[MeshGroup]) -> tuple[dict[str, Any] | None, str | None]:
    """Which grid/state the next init is remapped from, or why none."""

    candidates = []
    if request.from_grid is not None:
        candidates.append(("from", request.from_grid, request.from_state))
    if request.fallback_from_grid is not None:
        candidates.append(("fallback", request.fallback_from_grid,
                           request.fallback_from_state))
    if not candidates:
        return None, ("no --from-grid was given: the remap needs the grid the "
                      "previous state lives on")
    reasons = []
    for role, grid, state in candidates:
        if not Path(grid).is_file():
            reasons.append(f"{role} grid {grid} does not exist")
            continue
        if state is None:
            lat, _ = _read_coordinates(Path(grid))
            matching = [g for g in groups if g.n_cells == lat.size]
            if matching:
                state = matching[0].frames[-1].path
        if state is None:
            reasons.append(f"{role}: no --{'fallback-' if role == 'fallback' else ''}"
                           f"from-state and no history frame on {grid}")
            continue
        box = _grid_box(Path(grid))
        if box is not None and not box.contains(cull_box):
            reasons.append(
                f"{role} grid {Path(grid).name} covers {box.as_list()}, not the "
                f"next domain {cull_box.as_list()}; a remap from it would leave "
                f"cells with no source")
            continue
        return {"role": role, "grid": str(grid), "state": str(state)}, None
    return None, "; ".join(reasons)


def build_plan(request: AdaptRequest) -> dict[str, Any]:
    """Read history, form criteria, write the raster, decide, plan argv."""

    import numpy as np

    out = Path(request.out)
    criteria = resolve_criteria(request.criteria)
    ramps = criterion_ramps(criteria, request.thresholds, request.weights)
    if not (0.0 < request.fine_km < request.background_km):
        raise AdaptRefusal(
            f"--fine-km {request.fine_km:g} must be positive and finer than "
            f"--background-km {request.background_km:g}"
        )
    if request.generation not in ("regional", "global"):
        raise AdaptRefusal("--generation is regional or global")
    if "assets" in criteria and request.sites is None and request.assets is None:
        raise AdaptRefusal(
            "criterion assets needs --sites or --assets: a distance to nothing "
            "is no distance"
        )
    request.policy.validate()
    limits = _smoothness_limits()
    gradient_percent = (limits["warn"] if request.max_gradient_percent is None
                        else float(request.max_gradient_percent))
    if not 0.0 < gradient_percent <= limits["refuse"]:
        raise AdaptRefusal(
            f"--max-gradient-percent {gradient_percent:g} must lie in (0, "
            f"{limits['refuse']:g}]: past it `woof mesh` refuses the spec "
            f"(woof/data/mpas/mesh-sizing.json smoothness)"
        )
    gradient = gradient_percent / 100.0
    timestep = timestep_preflight(
        float(request.dt_seconds), float(request.fine_km),
        experimental=bool(request.experimental_dt), hours=float(request.hours),
        history_every_minutes=int(request.history_every_minutes),
    )

    # The state, and the raster its mesh was generated from, are verified
    # BEFORE anything is written: an -o that is the state's own folder would
    # otherwise overwrite that raster and every later comparison would be
    # against bytes no mesh was generated from.
    state = load_state(request.state)
    if state and state.get("mesh"):
        current_raster = Path(state["mesh"]["density"])
        if current_raster.resolve() == (out / "density.nc").resolve():
            raise AdaptRefusal(
                f"-o {out} is the folder the current mesh's raster lives in "
                f"({current_raster}); writing this cycle's raster there would "
                f"destroy the one the hysteresis compares against.  Use a fresh "
                f"-o per cycle"
            )
        if not current_raster.is_file():
            raise AdaptRefusal(
                f"the state's current mesh was generated from {current_raster}, "
                f"which no longer exists; the hysteresis compares against it"
            )
        if _sha256(current_raster) != state["mesh"].get("density_sha256"):
            raise AdaptRefusal(
                f"{current_raster} changed since the state recorded it; comparing "
                f"against other bytes than the mesh was generated from would "
                f"decide on a mesh nobody has"
            )

    paths = expand_history(request.history)
    groups = group_frames(paths)
    cell_fields: list[tuple[MeshGroup, dict[str, Any], dict[str, Any]]] = []
    weather = [name for name in criteria if name != "assets"]
    for group in groups:
        fields, notes = criterion_fields(group, weather, request.fields)
        cell_fields.append((group, fields, notes))
    if "precip" in weather and not any(
            np.isfinite(fields["precip"]).any() for _, fields, _ in cell_fields):
        raise AdaptRefusal(
            "criterion precip forms a RATE from consecutive accumulations and "
            "their valid times, and no mesh in --history has two timed frames.  "
            "A rate from one accumulation would be a total divided by a guess"
        )

    raster_km = float(request.raster_km if request.raster_km is not None
                      else max(request.fine_km, 1.0))
    densify = max(raster_km / 2.0, 0.05)
    points_lat, points_lon, asset_receipt = asset_points(
        request.sites, request.assets, densify_km=densify)

    # THE FORECAST DOMAIN: an explicit box, else the assets plus a margin,
    # else wherever a weather criterion fired plus a margin.
    if request.bbox is not None:
        domain = parse_bbox(request.bbox)
        domain_basis = "--bbox"
    elif points_lat.size:
        domain = Box.around(points_lat, points_lon).expanded(request.margin_km)
        domain_basis = f"assets + {request.margin_km:g} km"
    else:
        fired_lat: list[Any] = []
        fired_lon: list[Any] = []
        for group, fields, _ in cell_fields:
            demand = np.zeros(group.n_cells)
            for name in weather:
                r = ramps[name]
                demand = np.maximum(demand, ramp(fields[name], r["lo"], r["hi"]))
            fired = demand > 0.0
            fired_lat.append(np.degrees(group.lat[fired]))
            fired_lon.append(np.degrees(group.lon[fired]))
        lat_all = np.concatenate(fired_lat) if fired_lat else np.array([])
        if lat_all.size == 0:
            raise AdaptRefusal(
                "no criterion fired anywhere and no --bbox, --sites or --assets "
                "names a domain.  There is nothing to refine and nowhere to put "
                "a mesh; pass --bbox to plan a background mesh over a fixed area"
            )
        domain = Box.around(lat_all, np.concatenate(fired_lon)).expanded(request.margin_km)
        domain_basis = f"criteria + {request.margin_km:g} km"
    if domain.lon1 - domain.lon0 > 180.0:
        raise AdaptRefusal(
            f"the domain {domain.as_list()} spans more than 180 degrees of "
            f"longitude; a domain across the antimeridian is not planned here"
        )
    band = (float(request.window_band_km) if request.window_band_km is not None
            else (BOUNDARY_RINGS + 3) * float(request.background_km))
    raster_box = domain.expanded(band)
    lat, lon = raster_axes(raster_box, raster_km)
    areas = pixel_areas_km2(lat, lon)

    # Per-criterion demand on the raster: max over the meshes that carry it.
    demands: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    shape = (lat.size, lon.size)
    for name in weather:
        r = ramps[name]
        combined = np.zeros(shape)
        field_max = -np.inf
        covered_any = np.zeros(shape, dtype=bool)
        cells_over = 0
        for group, fields, _ in cell_fields:
            values = fields[name]
            finite = np.isfinite(values)
            if finite.any():
                field_max = max(field_max, float(np.nanmax(values)))
            cells_over += int(np.count_nonzero(values[finite] > r["lo"]))
            on_raster, covered = rasterize_cells(group, ramp(values, r["lo"], r["hi"]),
                                                 lat, lon)
            covered_any |= covered
            combined = np.fmax(combined, np.where(np.isfinite(on_raster), on_raster, 0.0))
        demands[name] = combined
        stats[name] = {
            "units": CRITERIA[name].units, "lo": r["lo"], "hi": r["hi"],
            "weight": r["weight"],
            "field_max": None if not math.isfinite(field_max) else round(field_max, 9),
            "cells_above_lo": cells_over,
            "raster_area_km2_with_demand": round(float(areas[combined > 0.0].sum()), 3),
            "raster_area_km2_full_demand": round(float(areas[combined >= 1.0].sum()), 3),
            "raster_fraction_on_a_history_mesh": round(float(
                areas[covered_any].sum() / areas.sum()), 6),
            "how": next((notes.get(name) for _, _, notes in cell_fields
                         if notes.get(name)), None),
        }
    if "assets" in criteria:
        r = ramps["assets"]
        distance = asset_distance_km(points_lat, points_lon, lat, lon)
        demands["assets"] = ramp(distance, r["lo"], r["hi"], inverted=True)
        stats["assets"] = {
            "units": "km", "lo": r["lo"], "hi": r["hi"], "weight": r["weight"],
            "points": int(points_lat.size), "densify_km": densify,
            "raster_area_km2_with_demand": round(float(
                areas[demands["assets"] > 0.0].sum()), 3),
            "raster_area_km2_full_demand": round(float(
                areas[demands["assets"] >= 1.0].sum()), 3),
            **asset_receipt,
        }

    demand = np.zeros(shape)
    for name in criteria:
        demand = np.maximum(demand, ramps[name]["weight"] * demands[name])
    demand = np.clip(demand, 0.0, 1.0)
    target = request.background_km * (request.fine_km / request.background_km) ** demand
    limited, sweeps = limit_gradient(target, lat, lon, gradient)
    worst = delivered_gradient(limited, lat, lon)
    if worst > gradient * (1.0 + 1.0e-9):
        raise AdaptRefusal(
            f"the limited raster's steepest gradient {worst * 100:.4f} %/cell is "
            f"past the {gradient_percent:g} %/cell it was limited to"
        )

    # DENSITY: woof.hex.density when it is here, else the internal writer.
    out.mkdir(parents=True, exist_ok=True)
    density_path = out / "density.nc"
    attributes = {
        "background_km": float(request.background_km),
        "fine_km": float(request.fine_km),
        "max_gradient_percent_per_cell": gradient_percent,
        "criteria": ",".join(criteria),
        "producer": "woof hex adapt",
    }
    external = external_density_writer()
    if external is not None:
        external(lat=lat, lon=lon, spacing_km=limited, out=density_path)
        writer = "woof.hex.density.density_from_fields"
    else:
        write_density_raster(density_path, lat, lon, limited, attributes=attributes)
        writer = "woof.hex.adapt.write_density_raster (internal; replace with woof.hex.density)"
    # Whoever wrote it, the contract is checked on the bytes.
    lat, lon, limited = read_density_raster(density_path)
    density_sha = _sha256(density_path)

    in_domain = ((lat[:, None] >= domain.lat0) & (lat[:, None] <= domain.lat1)
                 & (lon[None, :] >= domain.lon0) & (lon[None, :] <= domain.lon1))
    window_cells = estimated_cells(limited, areas)
    domain_cells = estimated_cells(limited[in_domain], areas[in_domain])
    sphere_km2 = 4.0 * math.pi * EARTH_RADIUS_KM ** 2
    outside_cells = ((sphere_km2 - float(areas.sum()))
                     / (HEX_AREA_FACTOR * request.background_km ** 2))
    generation_cells = (window_cells if request.generation == "regional"
                        else window_cells + outside_cells)
    edge_max = float(max(limited[0].max(), limited[-1].max(),
                         limited[:, 0].max(), limited[:, -1].max()))
    raster_summary = {
        "path": str(density_path), "sha256": density_sha, "writer": writer,
        "schema": DENSITY_SCHEMA,
        "shape": [int(lat.size), int(lon.size)], "raster_km": raster_km,
        "bbox": raster_box.as_list(),
        "min_spacing_km": float(limited.min()), "max_spacing_km": float(limited.max()),
        "max_spacing_on_raster_edge_km": edge_max,
        "max_gradient_percent_per_cell_requested": gradient_percent,
        "max_gradient_percent_per_cell_delivered": round(worst * 100.0, 6),
        "limiter_sweeps": sweeps,
        "fine_area_km2": round(float(areas[limited <= request.fine_km * 1.000001].sum()), 3),
    }
    if request.generation == "global" and edge_max < request.background_km * 0.999:
        raise AdaptRefusal(
            f"--generation global: the raster edge reaches only {edge_max:.3f} km "
            f"against the {request.background_km:g} km background outside it, a "
            f"jump the generator would see as a cliff.  Widen --window-band-km to "
            f"at least {(request.background_km - request.fine_km) / gradient:.0f} km "
            f"or use --generation regional"
        )
    estimate = {
        "basis": "area integral over hexagons of the requested spacing",
        "generation_cells": round(generation_cells, 1),
        "window_cells": round(window_cells, 1),
        "forecast_domain_cells": round(domain_cells, 1),
        "outside_window_background_cells": (
            round(outside_cells, 1) if request.generation == "global" else 0.0),
    }
    refusals: list[str] = []
    if request.max_cells is not None and domain_cells > request.max_cells:
        refusals.append(
            f"the forecast domain is an estimated {domain_cells:,.0f} cells, past "
            f"--max-cells {request.max_cells:,}"
        )

    # HYSTERESIS against the raster the CURRENT mesh was generated from
    # (loaded and verified above, before anything was written).
    cycle_index = int(request.cycle_index if request.cycle_index is not None
                      else (int(state.get("cycle_index", 0)) + 1 if state else 1))
    comparison = None
    domain_inside = None
    if state and state.get("mesh"):
        current = state["mesh"]
        current_raster = Path(current["density"])
        comparison = compare_targets(
            read_density_raster(current_raster), (lat, lon, limited),
            tolerance_ratio=request.policy.tolerance_ratio,
            background_km=float(current.get("background_km", request.background_km)),
        )
        domain_inside = Box(*current["domain"]).contains(domain)
    decision = decide(state, comparison, policy=request.policy,
                      domain_inside=domain_inside)

    # WHAT THE NEXT CYCLE RUNS.
    regenerate = decision["action"] in ("generate", "remesh")
    if regenerate:
        name = request.name or f"adapt-c{cycle_index:02d}-{density_sha[:10]}"
        work = out / "mesh"
        mesh = {
            "name": name,
            "parent_row": f"{name}-parent",
            "parent_grid": str(work / f"{name}-parent.grid.nc"),
            "parent_static": str(work / f"{name}-parent.static.nc"),
            "parent_static_highres": str(work / f"{name}-parent.static.highres.nc"),
            "parent_vertical": str(work / f"{name}-parent.vertical.nc"),
            "grid": str(work / f"{name}.grid.nc"),
            "static": str(work / f"{name}.static.nc"),
            "cull_receipt": str(work / f"{name}.grid.nc.cull-receipt.json"),
            "rows": str(work / "mesh-rows.json"),
            "density": str(density_path), "density_sha256": density_sha,
            "domain": domain.as_list(),
            "background_km": float(request.background_km),
            "generated_cycle": cycle_index, "cycles_held": 1,
            "dt_seconds": float(request.dt_seconds),
            "estimated_cells": round(domain_cells, 1),
        }
    else:
        mesh = dict(state["mesh"])
        if not math.isclose(float(mesh.get("dt_seconds", request.dt_seconds)),
                            float(request.dt_seconds)):
            raise AdaptRefusal(
                f"the hysteresis keeps mesh {mesh['name']}, whose row was "
                f"registered at dt {mesh.get('dt_seconds')} s, and --dt-seconds "
                f"is {request.dt_seconds:g}.  A kept row keeps its timestep; "
                f"pass the registered dt or let a remesh register a new row"
            )
        mesh["cycles_held"] = int(mesh.get("cycles_held", 0)) + 1
    init = out / "init" / f"{mesh['name']}.c{cycle_index:02d}.init.nc"
    lbc_dir = out / "lbc"
    forecast_dir = out / "forecast"

    window = {"kind": "polygon", "vertices_deg": raster_box.ring()}
    cull = {"kind": "polygon", "vertices_deg": domain.ring()}
    (out / "regional-window.json").write_text(
        json.dumps(window, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out / "cull-region.json").write_text(
        json.dumps(cull, indent=2) + "\n", encoding="utf-8", newline="\n")
    spec = {"name": mesh["name"], "background_km": float(request.background_km),
            "regions": [{"shape": "raster", "path": str(density_path)}]}
    (out / "mesh_spec.json").write_text(
        json.dumps(spec, indent=2) + "\n", encoding="utf-8", newline="\n")

    commands: list[dict[str, Any]] = []
    blocked: list[dict[str, str]] = []

    def add(stage: str, argv: list[Any], owner: str, env: Mapping[str, str] | None = None):
        argv = [str(item) for item in argv]
        commands.append({"stage": stage, "argv": argv, "owner": owner,
                         "env": dict(env or {}), "contract": check_argv(argv)})

    rows_env = {MESH_ROWS_ENVIRONMENT: mesh["rows"]}
    if regenerate:
        mesh_argv: list[Any] = [
            "mesh", "--density-raster", density_path,
            "--background-km", repr(float(request.background_km)),
            "--cells", str(int(math.ceil(generation_cells))),
            "--out", mesh["parent_grid"], "--static-out", mesh["parent_static"],
            "--name", mesh["parent_row"],
        ]
        if request.generation == "regional":
            mesh_argv += ["--regional-window", out / "regional-window.json"]
        add("mesh", mesh_argv, "unit 7 (--density-raster) / unit 9 (--regional-window)")
        static = mesh["parent_static"]
        if request.static_highres:
            add("static-highres", ["hex", "static-highres", "--static", static,
                                   "-o", mesh["parent_static_highres"],
                                   "--terrain", "glo30"], "unit 13")
            static = mesh["parent_static_highres"]
        add("register-parent", ["hex", "register", "--grid", mesh["parent_grid"],
                                "--static", static, "--name", mesh["parent_row"],
                                "--dt-seconds", repr(float(request.dt_seconds)),
                                "--rows", mesh["rows"]], "unit 10")
        cull_argv: list[Any] = [
            "hex", "cull", "--parent-grid", mesh["parent_grid"],
            "--parent-static", static, "--region", out / "cull-region.json",
            "--out-dir", work, "--name", mesh["name"], "--clobber",
        ]
        if request.vertical_spec is not None:
            add("vertical", ["hex", "vertical", "--grid", mesh["parent_grid"],
                             "--static", static, "--vertical-spec",
                             request.vertical_spec, "-o", mesh["parent_vertical"]],
                "unit 10")
            cull_argv += ["--parent-vertical", mesh["parent_vertical"]]
        else:
            blocked.append({"stage": "vertical",
                            "blocked_by": "no --vertical-spec: the regenerated mesh "
                                          "has no vertical artifact to cull"})
        add("cull", cull_argv, "existing door (+ unit 10 --parent-vertical)")
        add("register", ["hex", "register", "--grid", mesh["grid"],
                         "--static", mesh["static"], "--parent-row", mesh["parent_row"],
                         "--cull-receipt", mesh["cull_receipt"], "--name", mesh["name"],
                         "--dt-seconds", repr(float(request.dt_seconds)),
                         "--rows", mesh["rows"]], "unit 10")
    source, why_not = _remap_source(request, domain, groups)
    if source is not None:
        add("remap", ["hex", "remap", "--from-grid", source["grid"],
                      "--from-state", source["state"], "--to-grid", mesh["grid"],
                      "-o", init], "unit 14")
    else:
        blocked.append({"stage": "remap", "blocked_by": str(why_not)})

    start = None
    if request.start_time is not None:
        try:
            start = datetime.strptime(str(request.start_time)[:19], _STAMP_FORMAT)
        except ValueError as error:
            raise AdaptRefusal(
                f"--start-time {request.start_time!r} is not YYYY-MM-DD_HH:MM:SS"
            ) from error
    else:
        timed = [frame.valid for group in groups for frame in group.frames
                 if frame.valid is not None]
        if timed:
            start = max(timed)
    stop = None if start is None else start + timedelta(hours=float(request.hours))
    lbc_switches = (request.met_dir, request.nfglevels, request.extrap_airtemp,
                    request.use_spechumd)
    if start is None:
        blocked.append({"stage": "lbc", "blocked_by": "no --start-time and no "
                        "frame carries a valid time"})
        blocked.append({"stage": "forecast", "blocked_by": "no start time"})
    else:
        if all(item is not None for item in lbc_switches):
            add("lbc", ["hex", "lbc", "--grid", init, "--met-dir", request.met_dir,
                        "--out-dir", lbc_dir, "--start-time", _stamp(start),
                        "--stop-time", _stamp(stop),
                        "--nfglevels", str(int(request.nfglevels)),
                        "--extrap-airtemp", request.extrap_airtemp,
                        "--use-spechumd", request.use_spechumd], "existing door")
        else:
            blocked.append({
                "stage": "lbc",
                "blocked_by": "no --met-dir with --nfglevels/--extrap-airtemp/"
                              "--use-spechumd; `woof hex cycle run --adaptive` "
                              "drives the boundaries from its coarse parent with "
                              "rw_mpas_lbc --source unstructured-port-stream "
                              f"into {lbc_dir}",
            })
        forecast_argv: list[Any] = [
            "hex", "forecast", "--mesh", mesh["name"], "--grid", mesh["grid"],
            "--static", mesh["static"], "--init", init, "--lbc-dir", lbc_dir,
            "--init-source",
            f"woof hex adapt cycle {cycle_index}: remapped onto {mesh['name']}",
            "--start-time", _stamp(start), "--hours", repr(float(request.hours)),
            "--history-every-minutes", str(int(request.history_every_minutes)),
            "--out", forecast_dir, "--scratch", out / "scratch",
            "--receipt", out / "forecast-receipt.json",
            "--case-label", f"adapt-c{cycle_index:02d}",
        ]
        if timestep["timestep_evidence"] == EXPERIMENTAL_DT_EVIDENCE:
            forecast_argv.append("--experimental-dt")
        if request.les_model is not None:
            forecast_argv += ["--les-model", request.les_model]
        if request.pbl is not None:
            forecast_argv += ["--pbl", request.pbl]
        add("forecast", forecast_argv, "existing door (+ units 1/5/6 flags)",
            env=rows_env)

    # PROPOSED, not committed: the mesh this state names does not exist
    # until the plan's stages have run.  commit_state writes the
    # adapt-state.json a next cycle may continue from.
    new_state = {
        "schema": STATE_SCHEMA,
        "cycle_index": cycle_index,
        "mesh": mesh,
        "previous_decision": decision["action"],
        "committed": False,
    }
    (out / PROPOSED_STATE_NAME).write_text(
        json.dumps(new_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")

    not_parsed = [c["stage"] for c in commands if c["contract"]["status"] != "parsed"]
    plan = {
        "schema": PLAN_SCHEMA,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cycle_index": cycle_index,
        "experimental": timestep["timestep_evidence"] == EXPERIMENTAL_DT_EVIDENCE,
        "timestep_evidence": timestep["timestep_evidence"],
        "runnable": not refusals and not blocked and not not_parsed,
        "refusals": refusals,
        "stages_not_in_this_build": not_parsed,
        "blocked_stages": blocked,
        "commands": commands,
        "program": "python -m woof",
        "decision": decision,
        "policy": request.policy.as_dict(),
        "criteria": stats,
        "criteria_order": list(criteria),
        "combine": "weighted maximum of per-criterion demand; spacing = "
                   "background * (fine / background) ** demand",
        "raster": raster_summary,
        "estimate": estimate,
        "timestep": timestep,
        "domain": {"bbox": domain.as_list(), "basis": domain_basis,
                   "cull_region": str(out / "cull-region.json"),
                   "regional_window": str(out / "regional-window.json"),
                   "window_band_km": band, "generation": request.generation},
        "history": {
            "files": [str(frame.path) for group in groups for frame in group.frames],
            "meshes": [{"key": group.key, "cells": group.n_cells,
                        "frames": len(group.frames)} for group in groups],
        },
        "remap_source": source,
        "mesh": {**mesh, "init": str(init)},
        "paths": {"lbc_dir": str(lbc_dir), "forecast_dir": str(forecast_dir),
                  "init": str(init), "state": str(out / STATE_NAME),
                  "proposed_state": str(out / PROPOSED_STATE_NAME),
                  "density": str(density_path), "mesh_spec": str(out / "mesh_spec.json")},
        "start_time": None if start is None else _stamp(start),
        "stop_time": None if stop is None else _stamp(stop),
        "approximations": [
            "gradients: tangent-plane least squares over 7 nearest cells",
            "rasterisation: nearest cell within one cell spacing",
            "Courant: requested minimum spacing, not delivered dcEdge",
            "cells: area integral over hexagons of the requested spacing",
        ],
    }
    (out / "adapt-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    return plan


# ---------------------------------------------------------------------------
# the door
# ---------------------------------------------------------------------------
def _request(arguments: argparse.Namespace) -> AdaptRequest:
    return AdaptRequest(
        history=list(arguments.history),
        criteria=list(arguments.criteria),
        out=Path(arguments.out),
        fine_km=float(arguments.fine_km),
        background_km=float(arguments.background_km),
        dt_seconds=float(arguments.dt_seconds),
        thresholds=list(arguments.threshold or ()),
        weights=list(arguments.weight or ()),
        sites=arguments.sites, assets=arguments.assets, bbox=arguments.bbox,
        margin_km=float(arguments.margin_km),
        window_band_km=arguments.window_band_km,
        raster_km=arguments.raster_km,
        max_gradient_percent=arguments.max_gradient_percent,
        generation=arguments.generation,
        state=arguments.state,
        policy=RemeshPolicy(
            tolerance_ratio=float(arguments.tolerance_ratio),
            remesh_under_fraction=float(arguments.remesh_under_fraction),
            remesh_over_fraction=float(arguments.remesh_over_fraction),
            force_under_fraction=float(arguments.force_under_fraction),
            minimum_dwell_cycles=int(arguments.minimum_dwell_cycles),
        ),
        fields=FieldOptions(
            wind_level=int(arguments.wind_level),
            shear_level=int(arguments.shear_level),
            icing_levels=int(arguments.icing_levels),
        ),
        name=arguments.name,
        cycle_index=arguments.cycle_index,
        start_time=arguments.start_time,
        hours=float(arguments.hours),
        history_every_minutes=int(arguments.history_every_minutes),
        experimental_dt=bool(arguments.experimental_dt),
        vertical_spec=arguments.vertical_spec,
        static_highres=bool(arguments.static_highres),
        from_grid=arguments.from_grid, from_state=arguments.from_state,
        fallback_from_grid=arguments.fallback_from_grid,
        fallback_from_state=arguments.fallback_from_state,
        met_dir=arguments.met_dir, nfglevels=arguments.nfglevels,
        extrap_airtemp=arguments.extrap_airtemp,
        use_spechumd=arguments.use_spechumd,
        les_model=arguments.les_model, pbl=arguments.pbl,
        max_cells=arguments.max_cells,
    )


REQUIRED_FLAGS = (("history", "--history"), ("criteria", "--criteria"),
                  ("out", "-o"), ("fine_km", "--fine-km"),
                  ("background_km", "--background-km"),
                  ("dt_seconds", "--dt-seconds"))


def run_adapt(arguments: argparse.Namespace) -> int:
    if arguments.commit is not None:
        print(f"committed {commit_state(arguments.commit)}")
        return 0
    missing = [flag for name, flag in REQUIRED_FLAGS if getattr(arguments, name) is None]
    if missing:
        raise AdaptRefusal(f"woof hex adapt needs {', '.join(missing)} (or --commit)")
    plan = build_plan(_request(arguments))
    decision = plan["decision"]
    estimate = plan["estimate"]
    raster = plan["raster"]
    lines = [
        f"cycle {plan['cycle_index']}  decision {decision['action'].upper()}: "
        f"{decision['reason']}",
        f"raster {raster['shape'][0]}x{raster['shape'][1]} @ {raster['raster_km']:g} km  "
        f"spacing {raster['min_spacing_km']:.4g}..{raster['max_spacing_km']:.4g} km  "
        f"gradient {raster['max_gradient_percent_per_cell_delivered']:.3f} %/cell",
        f"cells: forecast domain ~{estimate['forecast_domain_cells']:,.0f}, "
        f"generation ~{estimate['generation_cells']:,.0f}",
        f"timestep {plan['timestep']['dt_seconds']:g} s  evidence "
        f"{plan['timestep_evidence']}",
    ]
    for name in plan["criteria_order"]:
        row = plan["criteria"][name]
        lines.append(
            f"  {name:<15} max {row.get('field_max', '-')!s:<12} "
            f"demand area {row['raster_area_km2_with_demand']:,.1f} km2"
        )
    for command in plan["commands"]:
        head = command["argv"][:2] if command["argv"][0] == "hex" else command["argv"][:1]
        lines.append(f"  {command['stage']:<15} woof {' '.join(head):<22} "
                     f"[{command['contract']['status']}]")
    for row in plan["blocked_stages"]:
        lines.append(f"  [blocked] {row['stage']}: {row['blocked_by']}")
    for refusal in plan["refusals"]:
        lines.append(f"  REFUSED: {refusal}")
    print("\n".join(lines))
    print(f"plan {Path(arguments.out) / 'adapt-plan.json'}")
    return 1 if plan["refusals"] else 0


def add_adapt_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "adapt",
        help="adapt the next cycle's mesh to this cycle's forecast: criteria -> "
             "density raster -> hysteresis -> the next cycle's commands",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--history", nargs="+", metavar="GLOB",
                        help="hex history frames (cuda-history.*.nc); globs are "
                             "expanded here, so quote them")
    parser.add_argument("--criteria", action="append",
                        metavar="LIST",
                        help="comma list of " + ",".join(
                            sorted(CRITERIA) + sorted(CRITERIA_ALIASES))
                        + "; repeatable")
    parser.add_argument("-o", "--out", type=Path, metavar="DIR",
                        help="next-spec folder: density.nc, adapt-plan.json, "
                             "adapt-state.proposed.json, cull/window shapes")
    parser.add_argument("--fine-km", type=float, metavar="KM",
                        help="spacing where a criterion's demand is full")
    parser.add_argument("--background-km", type=float, metavar="KM",
                        help="spacing where no criterion asks for anything")
    parser.add_argument("--dt-seconds", type=float, metavar="S",
                        help="the next forecast's timestep (registered with the row)")
    parser.add_argument(
        "--commit", type=Path, default=None, metavar="NEXT_SPEC",
        help="after a plan's stages have succeeded, promote NEXT_SPEC's "
             "adapt-state.proposed.json to the adapt-state.json --state reads, "
             "and exit.  --history, --criteria, -o, --fine-km, "
             "--background-km and --dt-seconds are required otherwise")
    parser.add_argument("--experimental-dt", action="store_true",
                        help="EXPERIMENTAL: admit a dt under the anchor table's "
                             "floor; plan and forecast carry timestep_evidence "
                             "experimental-unanchored")
    parser.add_argument("--threshold", action="append", metavar="NAME=LO:HI",
                        help="a criterion's demand ramp; repeatable")
    parser.add_argument("--weight", action="append", metavar="NAME=W",
                        help="a criterion's weight in (0, 1]; repeatable")
    parser.add_argument("--sites", type=Path, default=None, metavar="FILE",
                        help="woof-energy.sites.v1 document (criterion assets)")
    parser.add_argument("--assets", type=Path, default=None, metavar="FILE",
                        help="woof-energy.assets.v1 GeoJSON (criterion assets)")
    parser.add_argument("--bbox", default=None, metavar="LAT0,LAT1,LON0,LON1",
                        help="the forecast domain; default the assets (or where "
                             "a criterion fired) plus --margin-km")
    parser.add_argument("--margin-km", type=float, default=30.0, metavar="KM")
    parser.add_argument("--window-band-km", type=float, default=None, metavar="KM",
                        help="raster beyond the forecast domain (default "
                             f"{BOUNDARY_RINGS + 3} x --background-km)")
    parser.add_argument("--raster-km", type=float, default=None, metavar="KM",
                        help="raster pixel size (default max(--fine-km, 1))")
    parser.add_argument("--max-gradient-percent", type=float, default=None,
                        metavar="P",
                        help="limiter bound, %%/cell (default the published "
                             "reference in woof/data/mpas/mesh-sizing.json)")
    parser.add_argument("--generation", choices=("regional", "global"),
                        default="regional",
                        help="regional passes --regional-window to woof mesh")
    parser.add_argument("--state", type=Path, default=None, metavar="FILE",
                        help="the previous cycle's adapt-state.json")
    parser.add_argument("--cycle-index", type=int, default=None, metavar="N")
    parser.add_argument("--name", default=None, metavar="TEXT",
                        help="mesh row name if this cycle regenerates")
    parser.add_argument("--tolerance-ratio", type=float, default=1.25, metavar="X")
    parser.add_argument("--remesh-under-fraction", type=float, default=0.05, metavar="F")
    parser.add_argument("--remesh-over-fraction", type=float, default=0.30, metavar="F")
    parser.add_argument("--force-under-fraction", type=float, default=0.20, metavar="F")
    parser.add_argument("--minimum-dwell-cycles", type=int, default=2, metavar="N")
    parser.add_argument("--wind-level", type=int, default=0, metavar="K")
    parser.add_argument("--shear-level", type=int, default=10, metavar="K")
    parser.add_argument("--icing-levels", type=int, default=4, metavar="N")
    parser.add_argument("--start-time", default=None, metavar="YYYY-MM-DD_HH:MM:SS",
                        help="next forecast start; default the latest frame")
    parser.add_argument("--hours", type=float, default=6.0, metavar="H")
    parser.add_argument("--history-every-minutes", type=int, default=30, metavar="M")
    parser.add_argument("--vertical-spec", type=Path, default=None, metavar="FILE")
    parser.add_argument("--no-static-highres", dest="static_highres",
                        action="store_false", default=True,
                        help="skip woof hex static-highres on the regenerated static")
    parser.add_argument("--from-grid", type=Path, default=None, metavar="FILE",
                        help="grid of the state the next init is remapped from")
    parser.add_argument("--from-state", type=Path, default=None, metavar="FILE",
                        help="the state itself (default the latest frame on "
                             "--from-grid)")
    parser.add_argument("--fallback-from-grid", type=Path, default=None, metavar="FILE",
                        help="used when --from-grid does not cover the next domain")
    parser.add_argument("--fallback-from-state", type=Path, default=None, metavar="FILE")
    parser.add_argument("--met-dir", type=Path, default=None, metavar="DIR",
                        help="WPS intermediates for woof hex lbc")
    parser.add_argument("--nfglevels", type=int, default=None)
    parser.add_argument("--extrap-airtemp", choices=("constant", "linear", "lapse-rate"),
                        default=None)
    parser.add_argument("--use-spechumd", choices=("yes", "no"), default=None)
    parser.add_argument("--les-model", choices=("off", "3d_smagorinsky", "prognostic_tke"),
                        default=None, help="passed to woof hex forecast")
    parser.add_argument("--pbl", choices=("ysu", "off"), default=None,
                        help="passed to woof hex forecast")
    parser.add_argument("--max-cells", type=int, default=None, metavar="N",
                        help="refuse a forecast domain estimated past N cells")
    parser.set_defaults(handler=run_adapt)


__all__ = [
    "AdaptRefusal",
    "AdaptRequest",
    "CRITERIA",
    "DENSITY_SCHEMA",
    "PLAN_SCHEMA",
    "RemeshPolicy",
    "STATE_SCHEMA",
    "add_adapt_parser",
    "build_plan",
    "commit_state",
    "check_argv",
    "compare_targets",
    "decide",
    "limit_gradient",
    "read_density_raster",
    "write_density_raster",
]
