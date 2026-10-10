"""Sample wrfout history at sites and heights.

The heavy lifting lives in the Rust ``rw-sitesample`` library
(:mod:`woof.energy.sample_bridge`); this module is the Python face.

Output keys (both WRF and MPAS samplers use the same names):

profile, shape (time, site, height):
    ``U``, ``V``   earth-relative wind at mass points, m/s
    ``W``          vertical velocity destaggered to mass levels, m/s
    ``THETA``      potential temperature (T + 300), K
    ``PRES``       full pressure (P + PB), Pa
    ``QVAPOR``, ``QCLOUD``, ``QRAIN``, ``QICE``, ``QSNOW``, ``QGRAUP``  kg/kg
surface, shape (time, site):
    ``U10``, ``V10`` earth-relative; ``T2``, ``Q2``, ``PSFC``, ``SWDOWN``,
    ``SWDDNI``, ``SWDDIF``, ``RAINNC``, ``RAINC``, ``COSZEN`` as written.

Heights are metres above model terrain, interpolated linearly in height
between mass levels (height AGL from ``(PH + PHB) / g - HGT``, destaggered).
Horizontal interpolation is bilinear on mass points.  Sites outside the
grid's interior are NaN with ``inside == False``; nothing is extrapolated.

How a call runs
---------------
* The files are one domain: every file must carry the same projection
  attributes and grid dimensions, or the call is refused.  Their records
  are put in time order from ``Times``; a time that appears twice is
  refused (two runs mixed into one list), never deduplicated.
* Each site's fractional mass-point index comes from the file's own
  projection attributes (``MAP_PROJ``, ``TRUELAT1/2``, ``STAND_LON``,
  ``CEN_LAT/LON``, ``DX/DY``) through WOOF's WPS-transcribed projection
  classes (:func:`woof.static.projection.projection_class`).  At every
  record that index is checked against ``XLAT``/``XLONG`` at the site's
  nearest mass point; a disagreement of more than
  :data:`MAX_REGISTRATION_CELLS` (a quarter of a cell: wrong attributes,
  a half-cell stagger slip, a moved nest) is refused.
* A site is inside when its bilinear stencil lies on the mass grid
  (``0 <= i <= west_east - 1``, likewise ``j``).  The lateral boundary
  zone is not excluded here: keeping sites away from a domain's edges is
  the planner's job.
* Only the window covering the inside sites' stencils plus a two-cell
  margin is read from each record, and of the volumes only the lowest
  levels that bracket the highest requested height at every site (found
  from ``PH + PHB`` first, which is itself read to the previous record's
  depth plus two levels when that reaches every site's highest height;
  identical results, a fraction of the I/O on a 115-level run).  Files are opened one at a time.  Every per-cell operation
  (destaggering, bilinear and vertical interpolation, wind rotation) is
  done by the Rust library on that window.  The per-site work here (the
  projection, the cross-check, the window bounds) is small numpy vectors,
  which ``docs/dev/static-rust-port.md`` leaves to Python.
* ``g`` is WRF's ``9.81`` (``module_model_constants``), the constant the
  model's geopotential was built with.
* A requested height outside a site's column (below the lowest mass level
  or above the highest) is NaN, and the count is reported in ``notes``.

Requested keys the history cannot supply (``SWDDNI`` from a run without
that output, say) are omitted from the result and named in ``notes``; call
:func:`available_variables` first to refuse up front instead.  A key that is
not an output key at all is a :class:`ValueError`, and a call none of whose
requested keys can be supplied is refused with :class:`SampleUnavailable`.

Values are float32 (the history's own precision); the arithmetic is f64.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

PROFILE_VARS = ("U", "V", "W", "THETA", "PRES", "QVAPOR", "QCLOUD", "QRAIN",
                "QICE", "QSNOW", "QGRAUP")
SURFACE_VARS = ("U10", "V10", "T2", "Q2", "PSFC", "SWDOWN", "SWDDNI",
                "SWDDIF", "RAINNC", "RAINC", "COSZEN")

#: WRF's gravitational acceleration (``module_model_constants``).
WRF_GRAVITY = 9.81

#: Cells of margin read around the sites' stencils.
WINDOW_MARGIN = 2

#: Largest disagreement, in cells, between a site's projected index and
#: XLAT/XLONG at its nearest mass point.  float32 coordinates agree to a
#: few thousandths of a cell; a half-cell slip (mass and staggered grids
#: confused) must not pass.
MAX_REGISTRATION_CELLS = 0.25

#: wrfout ``MAP_PROJ`` codes WOOF's projection classes implement.
_MAP_PROJ = {1: "lambert", 2: "polar", 3: "mercator"}

_MASS3 = ("Time", "bottom_top", "south_north", "west_east")
_PLANE = ("Time", "south_north", "west_east")
#: Dimensions each source variable must have to be read.
_SOURCE_DIMS = {
    "U": ("Time", "bottom_top", "south_north", "west_east_stag"),
    "V": ("Time", "bottom_top", "south_north_stag", "west_east"),
    "W": ("Time", "bottom_top_stag", "south_north", "west_east"),
    "PH": ("Time", "bottom_top_stag", "south_north", "west_east"),
    "PHB": ("Time", "bottom_top_stag", "south_north", "west_east"),
    "T": _MASS3, "P": _MASS3, "PB": _MASS3,
    **{name: _MASS3 for name in ("QVAPOR", "QCLOUD", "QRAIN", "QICE",
                                 "QSNOW", "QGRAUP")},
    **{name: _PLANE for name in ("XLAT", "XLONG", "HGT", "SINALPHA",
                                 "COSALPHA", *SURFACE_VARS)},
    "Times": ("Time", "DateStrLen"),
}
#: Locating sites, terrain and time: without these nothing is sampled.
_BASE_SOURCES = ("Times", "XLAT", "XLONG", "HGT")
#: Every profile key also needs the geopotential for its heights.
_PROFILE_BASE = ("PH", "PHB")
_ROTATION = ("SINALPHA", "COSALPHA")
#: Source variables behind each output key (besides the bases above).
_KEY_SOURCES: dict[str, tuple[str, ...]] = {
    "U": ("U", "V", *_ROTATION),
    "V": ("U", "V", *_ROTATION),
    "W": ("W",),
    "THETA": ("T",),
    "PRES": ("P", "PB"),
    **{name: (name,) for name in ("QVAPOR", "QCLOUD", "QRAIN", "QICE",
                                  "QSNOW", "QGRAUP")},
    "U10": ("U10", "V10", *_ROTATION),
    "V10": ("U10", "V10", *_ROTATION),
    **{name: (name,) for name in ("T2", "Q2", "PSFC", "SWDOWN", "SWDDNI",
                                  "SWDDIF", "RAINNC", "RAINC", "COSZEN")},
}
#: Mass-field profile keys: key -> (source, added source, offset).
_SCALAR_PROFILES = {
    "W": ("W", None, 0.0),
    "THETA": ("T", None, 300.0),
    "PRES": ("P", "PB", 0.0),
    **{name: (name, None, 0.0) for name in ("QVAPOR", "QCLOUD", "QRAIN",
                                            "QICE", "QSNOW", "QGRAUP")},
}
#: Global attributes that must agree across the files of one domain.
_GRID_ATTRS = ("MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON", "CEN_LAT",
               "CEN_LON", "MOAD_CEN_LAT", "DX", "DY",
               "WEST-EAST_GRID_DIMENSION", "SOUTH-NORTH_GRID_DIMENSION")
_GRID_DIMS = ("west_east", "south_north", "bottom_top")


class SampleUnavailable(RuntimeError):
    """The history cannot supply a requested variable or site."""


@dataclass
class SampleResult:
    times: np.ndarray                 # datetime64[s], (T,)
    heights_m: np.ndarray             # (H,)
    profile: dict[str, np.ndarray]    # name -> (T, S, H)
    surface: dict[str, np.ndarray]    # name -> (T, S)
    inside: np.ndarray                # bool, (S,)
    terrain_m: np.ndarray             # model terrain height at sites, (S,)
    dx_m: float
    source: str = ""
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# File inspection
# ---------------------------------------------------------------------------

def _paths(paths: Sequence[Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    resolved = [Path(p) for p in paths]
    if not resolved:
        raise SampleUnavailable("no wrfout files were given")
    missing = [str(p) for p in resolved if not p.is_file()]
    if missing:
        raise SampleUnavailable(f"wrfout file(s) not found: {', '.join(missing)}")
    return resolved


def _dataset(path: Path):
    """One history file, opened without masking; unreadable is refused."""
    import netCDF4

    try:
        ds = netCDF4.Dataset(path)
    except OSError as error:
        raise SampleUnavailable(f"{path} is not a readable netCDF file: "
                                f"{error}") from error
    ds.set_auto_mask(False)
    return ds


def _sources(ds) -> set[str]:
    """Source variables present with the dimensions this sampler reads."""
    return {name for name, dims in _SOURCE_DIMS.items()
            if name in ds.variables
            and tuple(ds.variables[name].dimensions) == dims}


def _keys(sources: set[str]) -> set[str]:
    if not set(_BASE_SOURCES) <= sources:
        return set()
    keys = set()
    for key, needs in _KEY_SOURCES.items():
        if key in PROFILE_VARS:
            needs = needs + _PROFILE_BASE
        if set(needs) <= sources:
            keys.add(key)
    return keys


def available_variables(paths: Sequence[Path]) -> set[str]:
    """Output keys (see module docstring) the history files can supply.

    A key is available when every file carries all of its source variables
    with the expected dimensions (``U`` needs ``U``, ``V``, ``SINALPHA``,
    ``COSALPHA``, ``PH``, ``PHB``; ``THETA`` needs ``T``, ``PH``, ``PHB``;
    and so on), plus ``Times``, ``XLAT``, ``XLONG`` and ``HGT``, without
    which no key is available.
    """

    common: set[str] | None = None
    for path in _paths(paths):
        with _dataset(path) as ds:
            found = _sources(ds)
        common = found if common is None else common & found
    return _keys(common or set())


@dataclass(frozen=True)
class _FileMeta:
    path: Path
    attrs: dict[str, float]
    dims: tuple[int, int, int]        # west_east, south_north, bottom_top
    times: tuple[np.datetime64, ...]
    sources: frozenset[str]


@dataclass(frozen=True)
class _Grid:
    attrs: dict[str, float]
    nx: int
    ny: int
    nz: int


def _attr(ds, name: str, path: Path) -> float:
    if name not in ds.ncattrs():
        raise SampleUnavailable(f"{path} has no global attribute {name}; "
                                "cannot place sites on its grid")
    value = np.asarray(ds.getncattr(name)).ravel()
    if value.size != 1 or not np.isfinite(value[0]):
        raise SampleUnavailable(f"{path} attribute {name} = {value!r} is not "
                                "one finite number")
    return float(value[0])


def _times(path: Path, ds) -> tuple[np.datetime64, ...]:
    if "Times" not in ds.variables:
        raise SampleUnavailable(f"{path} has no Times variable")
    raw = np.asarray(ds.variables["Times"][:])
    if raw.ndim != 2:
        raise SampleUnavailable(f"{path} Times has shape {raw.shape}")
    times = []
    for row in raw:
        text = b"".join(bytes(c) for c in row).decode("ascii", "replace")
        text = text.strip().strip("\x00")
        try:
            times.append(np.datetime64(text.replace("_", "T"), "s"))
        except ValueError as error:
            raise SampleUnavailable(
                f"{path} Times entry {text!r} is not YYYY-MM-DD_HH:MM:SS"
            ) from error
    return tuple(times)


def _inspect(path: Path) -> _FileMeta:
    """Grid attributes, dimensions, times and variables of one file (the
    file is closed again: a long run is hundreds of files)."""
    with _dataset(path) as ds:
        missing = [d for d in _GRID_DIMS if d not in ds.dimensions]
        if missing:
            raise SampleUnavailable(
                f"{path} lacks dimension(s) {', '.join(missing)}; not a "
                "wrfout history file")
        return _FileMeta(
            path=path,
            attrs={name: _attr(ds, name, path) for name in _GRID_ATTRS},
            dims=tuple(len(ds.dimensions[d]) for d in _GRID_DIMS),
            times=_times(path, ds),
            sources=frozenset(_sources(ds)))


def _one_domain(metas: list[_FileMeta]) -> _Grid:
    first = metas[0]
    for meta in metas[1:]:
        if meta.attrs != first.attrs or meta.dims != first.dims:
            changed = [k for k in _GRID_ATTRS
                       if meta.attrs[k] != first.attrs[k]]
            if meta.dims != first.dims:
                changed.append(f"dimensions {meta.dims} vs {first.dims}")
            raise SampleUnavailable(
                f"{meta.path} is not the same domain as {first.path} "
                f"({', '.join(changed)}); sample one domain per call")
    nx, ny, nz = first.dims
    if (int(first.attrs["WEST-EAST_GRID_DIMENSION"]) != nx + 1
            or int(first.attrs["SOUTH-NORTH_GRID_DIMENSION"]) != ny + 1):
        raise SampleUnavailable(
            f"{first.path} grid-dimension attributes disagree with its "
            f"west_east={nx} / south_north={ny} dimensions")
    if nx < 2 or ny < 2:
        raise SampleUnavailable(
            f"{first.path} has a {ny}x{nx} mass grid; bilinear sampling "
            "needs at least 2x2")
    return _Grid(first.attrs, nx, ny, nz)


def _records(metas: list[_FileMeta]) -> list[tuple[np.datetime64, int, int]]:
    """``(time, file index, record)`` in time order; duplicates refused."""
    records = [(time, d, r) for d, meta in enumerate(metas)
               for r, time in enumerate(meta.times)]
    if not records:
        raise SampleUnavailable("the wrfout files hold no time records")
    records.sort(key=lambda item: (item[0], item[1], item[2]))
    for (t0, d0, _), (t1, d1, _) in zip(records, records[1:]):
        if t0 == t1:
            where = (str(metas[d0].path) if d0 == d1
                     else f"{metas[d0].path} and {metas[d1].path}")
            raise SampleUnavailable(
                f"time {t0} appears twice ({where}); the files must be one "
                "run's history with each time once")
    return records


def _projection(grid: _Grid, path: Path):
    from woof.static.projection import projection_class

    a = grid.attrs
    code = int(a["MAP_PROJ"])
    if code not in _MAP_PROJ:
        raise SampleUnavailable(
            f"{path} MAP_PROJ={code} is not a projection the site sampler "
            f"places sites on (implemented: "
            f"{', '.join(f'{k} {v}' for k, v in _MAP_PROJ.items())})")
    try:
        return projection_class(_MAP_PROJ[code])(
            a["CEN_LAT"], a["CEN_LON"], a["TRUELAT1"], a["TRUELAT2"],
            a["STAND_LON"], a["DX"], a["DY"],
            int(a["WEST-EAST_GRID_DIMENSION"]),
            int(a["SOUTH-NORTH_GRID_DIMENSION"]),
            moad_cen_lat=a["MOAD_CEN_LAT"])
    except (ValueError, NotImplementedError) as error:
        raise SampleUnavailable(
            f"{path}: its projection attributes do not define a grid "
            f"({error})") from error


# ---------------------------------------------------------------------------
# Argument checks
# ---------------------------------------------------------------------------

def _site_arrays(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if lat.ndim != 1 or lat.shape != lon.shape:
        raise ValueError(f"lat and lon must be 1-D arrays of one length, got "
                         f"{lat.shape} and {lon.shape}")
    if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
        raise ValueError("site coordinates must be finite")
    if np.any(np.abs(lat) > 90.0) or np.any(np.abs(lon) > 360.0):
        raise ValueError("site latitude must be within [-90, 90] and "
                         "longitude within [-360, 360] degrees")
    return lat, lon


def _heights(heights_m: Sequence[float]) -> np.ndarray:
    heights = np.asarray(list(heights_m), dtype=np.float64)
    if heights.ndim != 1:
        raise ValueError("heights_m must be a flat sequence of metres")
    if not np.all(np.isfinite(heights)) or np.any(heights < 0.0):
        raise ValueError(f"heights_m must be finite metres above ground, got "
                         f"{heights.tolist()}")
    return heights


def _requested(names: Sequence[str], allowed: tuple[str, ...],
               what: str) -> list[str]:
    if isinstance(names, str):
        names = [names]
    names = list(dict.fromkeys(names))
    unknown = [n for n in names if n not in allowed]
    if unknown:
        raise ValueError(f"{', '.join(unknown)} not {what} output key(s); "
                         f"choose from {', '.join(allowed)}")
    return names


# ---------------------------------------------------------------------------
# Window reads
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Window:
    j0: int
    j1: int
    i0: int
    i1: int          # exclusive ends, mass points


def _window(fi: np.ndarray, fj: np.ndarray, inside: np.ndarray,
            grid: _Grid) -> _Window | None:
    if not inside.any():
        return None
    i_lo = int(np.floor(fi[inside].min())) - WINDOW_MARGIN
    i_hi = int(np.floor(fi[inside].max())) + 2 + WINDOW_MARGIN
    j_lo = int(np.floor(fj[inside].min())) - WINDOW_MARGIN
    j_hi = int(np.floor(fj[inside].max())) + 2 + WINDOW_MARGIN
    return _Window(max(0, j_lo), min(grid.ny, j_hi),
                   max(0, i_lo), min(grid.nx, i_hi))


def _read(ds, name: str, record: int, w: _Window,
          levels: int | None = None) -> np.ndarray:
    """The window of one record; ``levels`` keeps the lowest mass levels
    (one more for a ``bottom_top_stag`` field)."""
    var = ds.variables[name]
    dims = var.dimensions
    sy = 1 if dims[-2] == "south_north_stag" else 0
    sx = 1 if dims[-1] == "west_east_stag" else 0
    rows = slice(w.j0, w.j1 + sy)
    cols = slice(w.i0, w.i1 + sx)
    if len(dims) == 4:
        top = None
        if levels is not None:
            top = levels + (1 if dims[1] == "bottom_top_stag" else 0)
        data = var[record, :top, rows, cols]
    else:
        data = var[record, rows, cols]
    data = np.ascontiguousarray(data, dtype=np.float32)
    for marker in ("_FillValue", "missing_value"):
        if marker in var.ncattrs():
            fill = np.float32(np.asarray(var.getncattr(marker)).ravel()[0])
            data = np.where(data == fill, np.float32(np.nan), data)
    return data


def _levels_needed(zagl: np.ndarray, top: float) -> int:
    """Mass levels (from the bottom) that bracket every height up to ``top``
    at every site: the first level at or above ``top`` per site, the deepest
    of those, plus the levels under it.  Interpolating on that truncated
    column gives the same values as the full one, so the volumes above it
    are never read."""
    nz = zagl.shape[1]
    if zagl.shape[0] == 0:
        return nz
    reach = zagl >= top                  # NaN compares False
    if not reach.any(axis=1).all():
        return nz
    return min(nz, int(reach.argmax(axis=1).max()) + 1)


def _cross_check(projection, xlat: np.ndarray, xlong: np.ndarray,
                 fi: np.ndarray, fj: np.ndarray, inside: np.ndarray,
                 w: _Window, grid: _Grid, label: str) -> float:
    """Largest |projected index - nearest mass index| over the inside sites,
    in cells; refuse above :data:`MAX_REGISTRATION_CELLS` or where the
    coordinates are not finite."""
    ii = np.clip(np.rint(fi[inside]).astype(np.int64), 0, grid.nx - 1)
    jj = np.clip(np.rint(fj[inside]).astype(np.int64), 0, grid.ny - 1)
    lat = xlat[jj - w.j0, ii - w.i0].astype(np.float64)
    lon = xlong[jj - w.j0, ii - w.i0].astype(np.float64)
    x, y = projection.latlon_to_ij(lat, lon)
    deviation = np.maximum(np.abs(np.asarray(x) - 1.0 - ii),
                           np.abs(np.asarray(y) - 1.0 - jj))
    if deviation.size == 0:
        return 0.0
    bad = ~np.isfinite(deviation)
    if bad.any():
        k = int(np.argmax(bad))
        raise SampleUnavailable(
            f"{label}: XLAT/XLONG at mass point (j={jj[k]}, i={ii[k]}) is "
            f"({lat[k]}, {lon[k]}), not a finite position; the site there "
            "cannot be checked against the file's coordinates")
    k = int(np.argmax(deviation))
    worst = float(deviation[k])
    if worst > MAX_REGISTRATION_CELLS:
        raise SampleUnavailable(
            f"{label}: XLAT/XLONG at mass point (j={jj[k]}, i={ii[k]}) is "
            f"({lat[k]:.6f}, {lon[k]:.6f}), which the file's projection "
            f"attributes place {worst:.3f} cells away (more than "
            f"{MAX_REGISTRATION_CELLS}); the attributes and the coordinates "
            "disagree (or the nest moved), so no site position can be "
            "trusted")
    return worst


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_wrfout(paths: Sequence[Path], lat: np.ndarray, lon: np.ndarray,
                  heights_m: Sequence[float],
                  profile_vars: Sequence[str] = PROFILE_VARS,
                  surface_vars: Sequence[str] = SURFACE_VARS
                  ) -> SampleResult:
    """Sample wrfout files (one domain, time-ordered) at the sites."""

    lat, lon = _site_arrays(lat, lon)
    heights = _heights(heights_m)
    profile_req = _requested(profile_vars, PROFILE_VARS, "profile")
    surface_req = _requested(surface_vars, SURFACE_VARS, "surface")
    files = _paths(paths)
    notes: list[str] = []
    metas = [_inspect(path) for path in files]
    grid = _one_domain(metas)
    records = _records(metas)
    common = set.intersection(*(set(meta.sources) for meta in metas))
    absent = [name for name in _BASE_SOURCES if name not in common]
    if absent:
        raise SampleUnavailable(
            f"the history lacks {', '.join(absent)} (with wrfout "
            "dimensions); sites cannot be placed or heights measured")
    available = _keys(common)
    profile_keys = [k for k in profile_req if k in available]
    surface_keys = [k for k in surface_req if k in available]
    for key in [*profile_req, *surface_req]:
        if key not in available:
            needs = _KEY_SOURCES[key] + (
                _PROFILE_BASE if key in PROFILE_VARS else ())
            lacking = [n for n in dict.fromkeys(needs) if n not in common]
            notes.append(f"omitted {key}: the history lacks "
                         f"{', '.join(lacking)}")
    if (profile_req or surface_req) and not (profile_keys or surface_keys):
        raise SampleUnavailable(
            "none of the requested keys can be supplied by this history: "
            + "; ".join(notes))

    projection = _projection(grid, files[0])
    x, y = projection.latlon_to_ij(lat, lon)
    fi = np.asarray(x, dtype=np.float64) - 1.0
    fj = np.asarray(y, dtype=np.float64) - 1.0

    from woof.energy import sample_bridge as bridge

    inside = bridge.inside(fi, fj, grid.ny, grid.nx)
    window = _window(fi, fj, inside, grid)
    ns, nt, nh = lat.size, len(records), heights.size
    profile = {k: np.full((nt, ns, nh), np.nan, dtype=np.float32)
               for k in profile_keys}
    surface = {k: np.full((nt, ns), np.nan, dtype=np.float32)
               for k in surface_keys}
    terrain = np.full(ns, np.nan, dtype=np.float32)
    outside = int(ns - inside.sum())
    if outside:
        notes.append(f"{outside} of {ns} site(s) outside the grid interior: "
                     "NaN, inside=False")
    sample_profiles = bool(profile_keys) and nh > 0
    worst = 0.0
    deepest = 0
    missed = np.zeros((int(inside.sum()), nh), dtype=bool)
    if window is not None:
        fiw = np.where(inside, fi - window.i0, np.nan)
        fjw = np.where(inside, fj - window.j0, np.nan)
        top = float(heights.max()) if nh else 0.0
        open_index, ds = -1, None
        guess: int | None = None
        try:
            for t, (time, d, r) in enumerate(records):
                if d != open_index:
                    if ds is not None:
                        ds.close()
                    ds, open_index = None, -1
                    ds, open_index = _dataset(files[d]), d
                label = f"{files[d]} at {time}"
                planes: dict[str, np.ndarray] = {}

                def plane(name: str) -> np.ndarray:
                    # rotation and terrain planes are reused within a record
                    if name not in planes:
                        planes[name] = _read(ds, name, r, window)
                    return planes[name]

                worst = max(worst, _cross_check(
                    projection, plane("XLAT"), plane("XLONG"), fi, fj,
                    inside, window, grid, label))
                hgt = plane("HGT")
                if t == 0:
                    terrain[:] = bridge.surface(hgt, fiw, fjw)
                if sample_profiles:
                    def heights_agl(levels: int | None) -> np.ndarray:
                        return bridge.heights_agl(
                            _read(ds, "PH", r, window, levels=levels),
                            _read(ds, "PHB", r, window, levels=levels),
                            hgt, fiw, fjw, WRF_GRAVITY)

                    # The geopotential is read as deep as the previous
                    # record needed plus two levels, and again in full
                    # only when that does not reach the highest height
                    # at every site.
                    zagl = None
                    if guess is not None and guess < grid.nz:
                        zagl = heights_agl(guess)
                        if not np.all(zagl[inside, -1] >= top):
                            zagl = None
                    if zagl is None:
                        zagl = heights_agl(None)
                    lo, hi = zagl[inside, :1], zagl[inside, -1:]
                    missed |= ~((heights[None, :] >= lo)
                                & (heights[None, :] <= hi))
                    nk = _levels_needed(zagl[inside], top)
                    deepest = max(deepest, nk)
                    guess = min(grid.nz, nk + 2)
                    zagl = np.ascontiguousarray(zagl[:, :nk])

                    def volume(name: str) -> np.ndarray:
                        return _read(ds, name, r, window, levels=nk)

                    if "U" in profile or "V" in profile:
                        ue, ve = bridge.wind_profile(
                            volume("U"), volume("V"), plane("SINALPHA"),
                            plane("COSALPHA"), fiw, fjw, zagl, heights)
                        if "U" in profile:
                            profile["U"][t] = ue
                        if "V" in profile:
                            profile["V"][t] = ve
                    for key in profile_keys:
                        if key not in _SCALAR_PROFILES:
                            continue
                        source, plus, offset = _SCALAR_PROFILES[key]
                        stagger = (bridge.STAGGER_Z
                                   if _SOURCE_DIMS[source][1] == "bottom_top_stag"
                                   else bridge.STAGGER_MASS)
                        profile[key][t] = bridge.profile(
                            volume(source), fiw, fjw, zagl, heights,
                            stagger=stagger,
                            plus=None if plus is None else volume(plus),
                            offset=offset)
                if "U10" in surface or "V10" in surface:
                    ue, ve = bridge.wind_surface(
                        plane("U10"), plane("V10"), plane("SINALPHA"),
                        plane("COSALPHA"), fiw, fjw)
                    if "U10" in surface:
                        surface["U10"][t] = ue
                    if "V10" in surface:
                        surface["V10"][t] = ve
                for key in surface_keys:
                    if key not in ("U10", "V10"):
                        surface[key][t] = bridge.surface(plane(key), fiw, fjw)
        finally:
            if ds is not None:
                ds.close()
        levels = (f", lowest {deepest} of {grid.nz} levels"
                  if sample_profiles else "")
        notes.append(
            f"window j[{window.j0}:{window.j1}] i[{window.i0}:{window.i1}] "
            f"of {grid.ny}x{grid.nx} mass points{levels} read per record")
        notes.append(
            f"projection (MAP_PROJ={int(grid.attrs['MAP_PROJ'])}) agrees "
            f"with XLAT/XLONG within {worst:.3g} cells at the sites' "
            "nearest mass points")
    if profile_keys:
        notes.append(f"heights above model terrain: (PH + PHB) / "
                     f"{WRF_GRAVITY} - HGT at mass levels, linear in "
                     "height, NaN outside the column")
        for h, misses in zip(heights, missed.sum(axis=0)):
            if misses:
                notes.append(
                    f"{h:g} m is outside the mass-level column at "
                    f"{int(misses)} inside site(s) at one or more times "
                    "(NaN there)")

    times = np.array([rec[0] for rec in records], dtype="datetime64[s]")
    first, last = files[records[0][1]], files[records[-1][1]]
    source = (f"wrfout {first.name}" if first == last
              else f"wrfout {first.name} .. {last.name}")
    source += f" ({len(files)} file(s), {nt} time(s))"
    return SampleResult(
        times=times, heights_m=heights, profile=profile, surface=surface,
        inside=inside.astype(bool), terrain_m=terrain,
        dx_m=float(grid.attrs["DX"]), source=source, notes=notes)
