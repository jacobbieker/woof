"""Fire-behaviour fuel statics: NFUEL_CAT, and terrain slope on the fire mesh.

A WRF-SFIRE-style fire model reads one static fuel category per fire-mesh
cell, and the terrain slope it spreads over.  This module builds both for
a domain that asks for them (:func:`with_fuel_statics`).  No shipped
configuration asks, so every other domain builds exactly the field set it
always built.

The categories are Anderson (1982) 1-13, plus 14 for no fuel, as
WRF-SFIRE's ``NFUEL_CAT`` numbers them.  They come from one of two
sources:

* a LANDFIRE FBFM13 or FBFM40 GeoTIFF, when the request names one.  It is
  a local file, read with rasterio.  Each fire-mesh cell centre is placed
  through the grid's own projection (WRF's module_llxy words, for every
  projection and nest a woof grid can have), into the raster's CRS, and the
  cell takes the pixel it lands in, by nearest neighbour.
  FBFM40 codes map to Anderson-13 through the crosswalk table.  Pixels
  that are no data, and cells outside the raster, take the land-use
  crosswalk below.  The receipt counts them, so a raster that misses the
  domain cannot pass silently as a fuel map;
* otherwise the domain's ``LU_INDEX``, through the crosswalk table for the
  land-use legend the build's own dataset declares (``MMINLU``: USGS,
  MODIFIED_IGBP_MODIS_NOAH or NLCD40).  Water, lakes, urban, ice and
  barren classes map to 14.  The legend's own ``ISWATER``, ``ISLAKE``,
  ``ISICE`` and ``ISURBAN`` must map to 14 in the table, or the build
  refuses: a table read against the wrong legend would burn the sea.

The crosswalk is data, ``woof/data/fire/landuse_to_fuel.json``.  Its
SHA-256 goes into the fuel receipt.

The fields:

* ``NFUEL_CAT`` ``(ny, nx)``, float64 like every atmospheric static field.
  On the land-use route it is the crosswalk of ``LU_INDEX``.  On the
  LANDFIRE route it is the dominant fire-mesh category of the cell's
  ``sr_y * sr_x`` sub-cells (lowest category wins ties, as
  :func:`woof.static.build.dominant_category` breaks them);
* ``NFUEL_CAT_FIRE`` ``(ny * sr_y, nx * sr_x)``, uint8: the refined fire
  mesh.  Sub-cell ``(J, I)`` is centred at 1-based mass coordinate
  ``(0.5 + (J + 0.5) / sr_y, 0.5 + (I + 0.5) / sr_x)``, WRF's alignment,
  so a cell's sub-cells tile it exactly.  Sampling is nearest neighbour.
  From ``LU_INDEX``, every sub-cell takes its parent cell's value, because
  the nearest mass point to a sub-cell centre is always the parent;
* ``ZSF``, float32: ``HGT_M`` refined onto the fire mesh as WRF v4.7.1's
  interpolate_z2fire refines atmospheric fields (module_fr_fire_util.F):
  the terrain continued one cell past the domain, then bilinear, in REAL(4)
  and WRF's order of evaluation;
* ``DZDXF``, ``DZDYF``, float32: the fire-mesh slope as
  module_fr_fire_model.F forms it from ZSF, a centred difference over
  ``2 * dx / sr_x`` with ZSF continued one node past the mesh.  Like
  WRF-SFIRE's, on the map plane, without the map factor.

The fire mesh is built on the GPU when one is usable (NVRTC kernels,
:mod:`woof.static.fire_mesh_gpu`) and on NumPy otherwise.  Both evaluate
WRF's statements in WRF's order, so they give the same words, and both are
held to WRF's own Fortran by tools/fire_mesh_wrf471_oracle.
:func:`build_fuel_fields` can leave the fire-mesh fields on the device
(``resident=True``) for a consumer that integrates them there.

The fire-mesh names are WRF-SFIRE's geo_em names.  The mesh covers the mass
grid only: WRF-SFIRE's ``west_east_subgrid`` is ``e_we * sr_x``, with the
extra staggered strip, and a geo_em writer pads that strip itself.

A selection carries a request through the static build
(:func:`woof.static.build.build_static` builds the fields last, after any
static source's overlay).  A route that overlays ``[static.highres]`` land
use or terrain afterwards must call :func:`refresh_fuel_fields`, so that
the fuel and slope come from the fields the run integrates.  The root
routes do (:func:`woof.runtime.case_static_fields`, ``woof static``,
``woof fuel preview``).  :func:`with_fuel_statics` is therefore not
chained into the nest and child static routes, which overlay without that
refresh.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Mapping, MutableMapping

import numpy as np

#: The crosswalk table's schema string; anything else is refused.
CROSSWALK_SCHEMA = "woof-landuse-to-fuel-v1"
#: The packaged crosswalk table.
CROSSWALK_PATH = (Path(__file__).resolve().parent.parent / "data" / "fire"
                  / "landuse_to_fuel.json")
#: WRF-SFIRE's no-fuel category.
NO_FUEL = 14
#: The categories a fuel field may hold: Anderson 1-13 and no fuel.
FUEL_CATEGORIES = tuple(range(1, NO_FUEL + 1))
#: How a LANDFIRE raster's codes are read.  ``auto`` reads FBFM13 when
#: every burnable code is 1-13, and FBFM40 when every one is 101-204.
LANDFIRE_KINDS = ("auto", "fbfm13", "fbfm40")
#: Where the fire-mesh work runs.  ``auto`` takes the GPU when one is usable
#: (:func:`gpu_available`); the result is the same word for word either way.
DEVICES = ("auto", "cpu", "gpu")
#: The field a request adds on the atmospheric grid.
ATMOSPHERIC_FIELDS = ("NFUEL_CAT",)
#: The fields a request adds on the fire mesh, WRF-SFIRE's geo_em names.
FIRE_MESH_FIELDS = ("NFUEL_CAT_FIRE", "ZSF", "DZDXF", "DZDYF")
FUEL_FIELDS = ATMOSPHERIC_FIELDS + FIRE_MESH_FIELDS


def _refusal(action: str, why: str) -> ValueError:
    from woof.explain import layered
    return ValueError(layered(action, why))


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise _refusal(
            f"fire-mesh refinement {name} must be a whole number of fire "
            f"cells per atmospheric cell (1 or more), got {value!r}.",
            "The fire mesh tiles each atmospheric cell exactly, as "
            "WRF-SFIRE's sr_x and sr_y do.  A fractional or rounded ratio "
            "would place sub-cells across cell edges.")
    if int(value) < 1:
        raise _refusal(
            f"fire-mesh refinement {name} must be 1 or more, got {value}.  "
            "Set sr_x and sr_y to the fire cells per atmospheric cell.",
            "WRF-SFIRE refuses a fire run with no fire mesh.  Zero is its "
            "default for sr_x and sr_y, so an unset ratio means the mesh "
            "was never declared.")
    return int(value)


@dataclass(frozen=True)
class FuelStaticsRequest:
    """What a domain's fuel statics are built from and at what refinement.

    ``sr_x``/``sr_y`` are fire cells per atmospheric cell (WRF-SFIRE's
    ``&domains sr_x, sr_y``).  ``landfire`` is a local LANDFIRE FBFM13 or
    FBFM40 GeoTIFF, or ``None`` to use the land-use crosswalk.  ``device``
    is where the fire mesh is built (:data:`DEVICES`).  It is left out of
    equality and hashing, because both devices give the same words, so a
    memo built on one serves the other.
    """

    sr_x: int
    sr_y: int
    landfire: Path | None = None
    landfire_kind: str = "auto"
    device: str = field(default="auto", compare=False)

    def __post_init__(self):
        object.__setattr__(self, "sr_x", _positive_int("sr_x", self.sr_x))
        object.__setattr__(self, "sr_y", _positive_int("sr_y", self.sr_y))
        if self.landfire is not None:
            object.__setattr__(self, "landfire", Path(self.landfire))
        if self.landfire_kind not in LANDFIRE_KINDS:
            raise _refusal(
                f"landfire_kind must be one of {list(LANDFIRE_KINDS)}, got "
                f"{self.landfire_kind!r}.",
                "FBFM13 and FBFM40 rasters share the non-burnable codes "
                "91-99 and differ in every burnable one.  auto reads the "
                "kind from the codes the domain samples.")
        if self.device not in DEVICES:
            raise _refusal(
                f"device must be one of {list(DEVICES)}, got "
                f"{self.device!r}.",
                "auto builds the fire mesh on the GPU when one is usable "
                "and on the CPU otherwise; both give the same words.")
        if self.landfire is None and self.landfire_kind != "auto":
            raise _refusal(
                f"landfire_kind={self.landfire_kind!r} names a raster kind, "
                "but the request names no LANDFIRE raster.  Give the raster "
                "path, or leave landfire_kind at 'auto'.",
                "A kind with no raster would be dropped in silence, and the "
                "land-use crosswalk would build the map in its place.")


def fuel_request_for(cfg) -> FuelStaticsRequest | None:
    """The fuel statics ``cfg`` asks for, or ``None``.

    ``cfg`` is a :class:`FuelStaticsRequest`, or a configuration that
    carries WRF-SFIRE's namelist names: ``ifire`` > 0 turns the fire model
    on, and ``sr_x``/``sr_y`` set the mesh.  A configuration without
    ``ifire`` (every configuration woof ships) asks for nothing.
    """
    if cfg is None:
        return None
    if isinstance(cfg, FuelStaticsRequest):
        return cfg
    if int(getattr(cfg, "ifire", 0) or 0) <= 0:
        return None
    return FuelStaticsRequest(sr_x=getattr(cfg, "sr_x", 0),
                              sr_y=getattr(cfg, "sr_y", 0))


def with_fuel_statics(selection, cfg):
    """A GEOG selection also requesting this configuration's fuel statics."""
    request = fuel_request_for(cfg)
    if request is None:
        return selection
    return replace(selection, fuel=request)


# --- the crosswalk table -----------------------------------------------------

@lru_cache(maxsize=1)
def crosswalk_table() -> dict:
    """The packaged crosswalk table, validated, with its SHA-256."""
    payload = CROSSWALK_PATH.read_bytes()
    table = json.loads(payload)
    if table.get("schema") != CROSSWALK_SCHEMA:
        raise ValueError(
            f"{CROSSWALK_PATH} has schema {table.get('schema')!r}, expected "
            f"{CROSSWALK_SCHEMA!r}")
    rows = [(f"landuse {legend} {code}", row["fuel"])
            for legend, spec in table["landuse"].items()
            for code, row in spec["categories"].items()]
    rows += [(f"FBFM40 {code}", row["fuel"])
             for code, row in table["landfire"]["fbfm40"].items()]
    bad = [(where, fuel) for where, fuel in rows
           if isinstance(fuel, bool) or fuel not in FUEL_CATEGORIES]
    if bad:
        raise ValueError(
            f"{CROSSWALK_PATH} maps outside the fuel categories 1-14: {bad}")
    table["sha256"] = hashlib.sha256(payload).hexdigest()
    return table


def _legend(mminlu) -> tuple[str, dict]:
    table = crosswalk_table()
    name = str(mminlu).strip().upper()
    name = table["landuse_aliases"].get(name, name)
    spec = table["landuse"].get(name)
    if spec is None:
        raise _refusal(
            f"no land-use-to-fuel crosswalk for the land-use legend "
            f"MMINLU={mminlu!r}; the table covers "
            f"{sorted(table['landuse'])}.  Give the request a LANDFIRE "
            "FBFM13 or FBFM40 raster, or add the legend's rows to "
            f"{CROSSWALK_PATH.name}.",
            "The crosswalk is keyed by legend because category numbers "
            "mean different classes in each one: 16 is water in USGS and "
            "barren in MODIS.  Reading a legend through another's rows "
            "would map the sea to fuel.")
    return name, spec


def landuse_fuel_lookup(mminlu) -> np.ndarray:
    """``lookup[category]`` = fuel for one legend; 0 where it has no row."""
    _, spec = _legend(mminlu)
    codes = {int(code): int(row["fuel"])
             for code, row in spec["categories"].items()}
    lookup = np.zeros(max(codes) + 1, dtype=np.int64)
    for code, fuel in codes.items():
        lookup[code] = fuel
    return lookup


def _require_legend_no_fuel(name: str, lookup: np.ndarray,
                            landuse_attrs: Mapping[str, object]) -> None:
    """The legend's own water, lake, ice and urban classes map to 14."""
    wrong = []
    for key in ("ISWATER", "ISLAKE", "ISICE", "ISURBAN"):
        value = landuse_attrs.get(key)
        if value is None:
            continue
        category = int(value)
        if category < 1:
            continue  # WPS spells "no such class" as -1.
        fuel = int(lookup[category]) if category < lookup.size else 0
        if fuel != NO_FUEL:
            wrong.append(f"{key}={category} maps to {fuel or 'nothing'}")
    if wrong:
        raise _refusal(
            f"the {name} land-use-to-fuel crosswalk does not match the "
            f"land-use dataset's own legend: {'; '.join(wrong)} (expected "
            f"{NO_FUEL}, no fuel).  Fix the {name} rows of "
            f"{CROSSWALK_PATH.name}, or check that the dataset's MMINLU "
            "names the legend it is written in.",
            "The dataset's index names its water, lake, ice and urban "
            "categories.  When the table disagrees with them, the table is "
            "being read against a legend it was not written for.")


def nfuel_from_lu_index(lu_index, landuse_attrs: Mapping[str, object]
                        ) -> np.ndarray:
    """Anderson-13 fuel categories (14 = no fuel) from ``LU_INDEX``."""
    name, _ = _legend(landuse_attrs.get("MMINLU"))
    lookup = landuse_fuel_lookup(name)
    _require_legend_no_fuel(name, lookup, landuse_attrs)
    lu = np.asarray(lu_index, dtype=np.float64)
    if not np.isfinite(lu).all() or np.any(lu != np.floor(lu)):
        raise ValueError("LU_INDEX must hold whole land-use categories")
    categories = lu.astype(np.int64)
    known = (categories >= 1) & (categories < lookup.size)
    fuel = np.zeros(categories.shape, dtype=np.int64)
    fuel[known] = lookup[categories[known]]
    unmapped = sorted({int(c) for c in np.unique(categories[fuel == 0])})
    if unmapped:
        raise _refusal(
            f"LU_INDEX holds {name} categories {unmapped} that the "
            f"land-use-to-fuel crosswalk has no row for.  Add them to "
            f"{CROSSWALK_PATH.name}.",
            "A category with no row has no fuel the table vouches for.  "
            "Leaving it at no fuel would stop a fire at a class boundary "
            "nobody chose, and guessing one would let it burn there.")
    return fuel.astype(np.float64)


# --- devices -----------------------------------------------------------------

#: NFUEL_CAT_FIRE's storage: categories 1-14 fit a byte.
FIRE_CATEGORY_DTYPE = np.uint8
#: ZSF, DZDXF and DZDYF: WRF's fire model computes them in REAL(4).
FIRE_FLOAT_DTYPE = np.float32
#: A fire cell the LANDFIRE raster gave no pixel for.
MISSING_CODE = np.int32(np.iinfo(np.int32).min)


def gpu_available() -> bool:
    """Whether a CUDA device may be used here (GPUWM_NO_LOCAL_GPU says no)."""
    from woof.local_gpu import no_local_gpu
    if no_local_gpu():
        return False
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def array_module(device: str = "auto"):
    """NumPy or CuPy for ``device``."""
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
    if device == "cpu":
        return np
    usable = gpu_available()
    if device == "auto" and not usable:
        return np
    if not usable:
        raise _refusal(
            "device='gpu' was asked for the fire-mesh build, and no CUDA "
            "device is usable here (no cupy, no card, or GPUWM_NO_LOCAL_GPU "
            "is set).  Use device='auto' or 'cpu'.",
            "The CPU and GPU bodies give the same words, so 'auto' loses "
            "nothing but time.")
    import cupy
    return cupy


def _is_device(array) -> bool:
    from woof.spectral_ops.backend import is_cupy_array
    return is_cupy_array(array)


def _xp_of(array):
    from woof.spectral_ops.backend import namespace
    return namespace(array)


def _host(array) -> np.ndarray:
    from woof.spectral_ops.backend import asnumpy
    return asnumpy(array)


def to_device(fields: Mapping[str, object]) -> dict:
    """``fields`` with the fire-mesh fields CuPy arrays on the current device.

    For a consumer (the fire model) that keeps the fire mesh resident.  The
    layout is :func:`build_fuel_fields` ``resident=True``'s: the fire-mesh
    fields move, and ``NFUEL_CAT`` and every other field stay as they are.
    """
    import cupy
    return {name: (cupy.asarray(value) if name in FIRE_MESH_FIELDS
                   else value)
            for name, value in fields.items()}


def _gpu_failure_types() -> tuple[type, ...]:
    """The device failures auto steps back to the CPU body from."""
    from cupy.cuda import compiler, driver, memory, runtime
    return (runtime.CUDARuntimeError, driver.CUDADriverError,
            memory.OutOfMemoryError, compiler.CompileException)


# --- LANDFIRE ----------------------------------------------------------------

def _fbfm_lookup() -> tuple[np.ndarray, frozenset, dict]:
    table = crosswalk_table()["landfire"]
    nonburnable = frozenset(int(code) for code in table["nonburnable"])
    fbfm40 = {int(code): int(row["fuel"])
              for code, row in table["fbfm40"].items()}
    lookup = np.zeros(max(fbfm40) + 1, dtype=FIRE_CATEGORY_DTYPE)
    lookup[1:14] = np.arange(1, 14)
    for code in nonburnable:
        lookup[code] = NO_FUEL
    for code, fuel in fbfm40.items():
        lookup[code] = fuel
    return lookup, nonburnable, fbfm40


def landfire_codes_to_fuel(codes, kind: str = "auto"):
    """``(fuel, valid, kind)`` for raw LANDFIRE FBFM13/FBFM40 codes.

    ``codes`` is an integer array (:data:`MISSING_CODE` where the raster
    gave no pixel), or a float array with NaN there; NumPy or CuPy.  Codes
    1-13 are FBFM13 and stay themselves.  Codes 101-204 are FBFM40 and go
    through the crosswalk.  The non-burnable codes 91-99 are 14 in both.
    ``valid`` is False on missing cells and codes <= 0 (LANDFIRE's -9999
    fill).  Any other code is refused, as is a mix of FBFM13 and FBFM40
    codes.  ``fuel`` is uint8, 0 where not valid, on the input's device.
    """
    if kind not in LANDFIRE_KINDS:
        raise ValueError(f"landfire kind must be one of {LANDFIRE_KINDS}")
    xp = _xp_of(codes)
    lookup, nonburnable, fbfm40 = _fbfm_lookup()
    values = xp.asarray(codes)
    if values.dtype.kind == "f":
        valid = xp.isfinite(values) & (values > 0.0)
        raw = xp.where(valid, values, 0.0)
        if bool(xp.any(raw != xp.floor(raw))):
            raise ValueError("LANDFIRE fuel codes must be whole numbers")
        raw = raw.astype(xp.int64)
    else:
        raw = values.astype(xp.int64)
        valid = raw > 0
        raw = xp.where(valid, raw, 0)
    present = {int(code) for code in _host(xp.unique(raw[valid]))}
    has13 = sorted(present & set(range(1, 14)))
    has40 = sorted(present & set(fbfm40))
    unknown = sorted(present - set(range(1, 14)) - set(fbfm40) - nonburnable)
    if unknown:
        raise _refusal(
            f"the LANDFIRE raster holds codes {unknown[:20]} that are "
            "neither FBFM13 (1-13), FBFM40 (101-204) nor non-burnable "
            "(91-99).  Check that the file is an FBFM13 or FBFM40 product "
            "and not another LANDFIRE layer.",
            "Other LANDFIRE layers (canopy cover, EVT) are also integer "
            "rasters.  Read as fuel models, their codes would make a fuel "
            "map that looks plausible and is wrong everywhere.")
    if kind == "auto":
        if has13 and has40:
            raise _refusal(
                f"the LANDFIRE raster mixes FBFM13 codes {has13} with "
                f"FBFM40 codes {has40[:10]}.  Name the kind "
                "(landfire_kind = 'fbfm13' or 'fbfm40').",
                "One product holds one code set.  A mix means two rasters "
                "were mosaicked, and auto cannot tell which code set each "
                "pixel uses.")
        kind = "fbfm40" if has40 else "fbfm13"
    if kind == "fbfm13" and has40:
        raise ValueError(
            f"landfire_kind='fbfm13' but the raster holds FBFM40 codes "
            f"{has40[:10]}")
    if kind == "fbfm40" and has13:
        raise ValueError(
            f"landfire_kind='fbfm40' but the raster holds FBFM13 codes "
            f"{has13}")
    fuel = xp.where(valid, xp.asarray(lookup)[raw], 0).astype(
        FIRE_CATEGORY_DTYPE)
    return fuel, valid, kind


def fire_mesh_axis(n: int, sr: int) -> np.ndarray:
    """1-based mass coordinates of the ``n * sr`` fire sub-cell centres.

    WRF's alignment: interpolate_z2fire lines mass point ``ids`` up with
    fire node ``ifds + (ir - 1) * 0.5``.
    """
    return 0.5 + (np.arange(int(n) * int(sr), dtype=np.float64) + 0.5) / sr


def _geog_unavailable() -> str:
    from .geog_stack import geog_unavailable_detail
    return geog_unavailable_detail()


@lru_cache(maxsize=8)
def _file_sha256(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def landfire_sha256(path) -> str:
    """The raster's SHA-256, hashed once per file version per process."""
    path = Path(path).resolve()
    stat = path.stat()
    return _file_sha256(str(path), stat.st_size, stat.st_mtime_ns)


def _geo_stack():
    try:
        import rasterio
        from pyproj import Transformer
        from rasterio.windows import Window
    except ImportError as exc:
        raise RuntimeError(_geog_unavailable()) from exc
    return rasterio, Transformer, Window


def fire_mesh_latlon(grid, sr_x: int, sr_y: int):
    """``(lat, lon)``: the fire-cell centres through the grid's projection.

    ``grid.ij_to_latlon`` is woof's transcription of WRF's module_llxy
    ``ij_to_latlon``, for every projection a woof grid can have (Lambert,
    Mercator, polar stereographic) and for nests and moved nests, which
    carry their own placement.  At the fire-cell centres its words are
    WRF's own: tests/test_fire_mesh_wrf471_parity.py holds them to WRF
    v4.7.1 module_llxy at 0 ULP in both hemispheres.  float64 ``(ny * sr_y,
    nx * sr_x)``.
    """
    nx, ny = int(grid.e_we) - 1, int(grid.e_sn) - 1
    x, y = np.meshgrid(fire_mesh_axis(nx, sr_x), fire_mesh_axis(ny, sr_y))
    lat, lon = grid.ij_to_latlon(x, y)
    return (np.asarray(lat, dtype=np.float64),
            np.asarray(lon, dtype=np.float64))


def landfire_pixel_indices(dataset, grid, sr_x: int, sr_y: int):
    """``(row, col)``: each fire cell's raster pixel, -1 outside the raster.

    As WPS geogrid samples a source: the cell centre's latitude and
    longitude (:func:`fire_mesh_latlon`, WRF's words) go into the raster's
    CRS (PROJ, from geographic WGS84 coordinates, the datum convention of
    the ``[static.highres]`` overlay), then through the raster's inverse
    affine to fractional pixel coordinates, and the cell takes the pixel
    those fall in (floor).  int64 ``(ny * sr_y, nx * sr_x)``.
    """
    _, Transformer, _ = _geo_stack()
    if dataset.crs is None:
        raise ValueError(
            f"LANDFIRE raster {dataset.name} has no CRS; LANDFIRE products "
            "are delivered with one, so this is not a LANDFIRE GeoTIFF")
    lat, lon = fire_mesh_latlon(grid, sr_x, sr_y)
    source_x, source_y = Transformer.from_crs(
        "EPSG:4326", dataset.crs, always_xy=True).transform(lon, lat)
    source_x, source_y = np.asarray(source_x), np.asarray(source_y)
    inverse = ~dataset.transform
    col_f = inverse.a * source_x + inverse.b * source_y + inverse.c
    row_f = inverse.d * source_x + inverse.e * source_y + inverse.f
    inside = (np.isfinite(col_f) & np.isfinite(row_f)
              & (col_f >= 0.0) & (col_f < dataset.width)
              & (row_f >= 0.0) & (row_f < dataset.height))
    col = np.full(col_f.shape, -1, dtype=np.int64)
    row = np.full(row_f.shape, -1, dtype=np.int64)
    col[inside] = np.floor(col_f[inside]).astype(np.int64)
    row[inside] = np.floor(row_f[inside]).astype(np.int64)
    return row, col


def sample_landfire(path, grid, sr_x: int, sr_y: int, *, xp=np):
    """Raw raster codes at every fire sub-cell centre, nearest neighbour.

    int32 ``(ny * sr_y, nx * sr_x)`` on ``xp``'s device,
    :data:`MISSING_CODE` where the cell falls outside the raster or on its
    no-data value.  Only the window the domain covers is read, so a
    CONUS-wide product costs the domain's footprint.  The pixel indices
    are computed on the host (:func:`landfire_pixel_indices`); the gather
    runs on the device when ``xp`` is CuPy.
    """
    rasterio, _, Window = _geo_stack()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"LANDFIRE fuel raster missing: {path}")
    with rasterio.open(path) as dataset:
        row, col = landfire_pixel_indices(dataset, grid, sr_x, sr_y)
        inside = row >= 0
        if not inside.any():
            return xp.full(row.shape, MISSING_CODE, dtype=xp.int32)
        c0, c1 = int(col[inside].min()), int(col[inside].max())
        r0, r1 = int(row[inside].min()), int(row[inside].max())
        block = dataset.read(1, window=Window(c0, r0, c1 - c0 + 1,
                                              r1 - r0 + 1))
        nodata = dataset.nodata
        missing = np.zeros(block.shape, dtype=bool)
        if nodata is not None:
            missing |= (np.isnan(block) if np.isnan(nodata)
                        else block == nodata)
        if block.dtype.kind == "f":
            missing |= ~np.isfinite(block)
        present = block[~missing]
        limits = np.iinfo(np.int32)
        if present.size and (
                (block.dtype.kind == "f"
                 and bool(np.any(present != np.floor(present))))
                or float(present.max()) > limits.max
                or float(present.min()) <= limits.min):
            raise _refusal(
                f"LANDFIRE raster {path} holds {block.dtype} pixels that "
                "are not whole fuel-model codes (or lie outside int32).  "
                "Use the FBFM13 or FBFM40 product as LANDFIRE publishes it.",
                "The fuel models are integer codes.  A resampled or "
                "rescaled copy has values between them that name no model.")
        block = np.where(missing, 0, block).astype(np.int32)
        block[missing] = MISSING_CODE
    local_row = np.where(inside, row - r0, -1)
    local_col = np.where(inside, col - c0, 0)
    if xp is np:
        codes = np.full(row.shape, MISSING_CODE, dtype=np.int32)
        codes[inside] = block[local_row[inside], local_col[inside]]
        return codes
    from . import fire_mesh_gpu
    return fire_mesh_gpu.gather(block, local_row, local_col, MISSING_CODE)


# --- the fire mesh, as WRF v4.7.1 refines it ---------------------------------

def _ex(a, b):
    """continue_at_boundary's EX, bias 0, in REAL(4) and WRF's order."""
    f32 = np.float32
    bias = f32(0.0)
    t = f32(2.0) * a - b
    return (f32(1.0) - bias) * t + bias * np.maximum(np.maximum(t, a), b)


def continue_at_boundary(z) -> np.ndarray:
    """``z`` (ny, nx) float32 with WRF's one-cell linear continuation.

    module_fr_fire_util.F continue_at_boundary(1, 1, 0., ...) on a tile
    that is the whole domain: the x halo on the domain's rows, the y halo
    on its columns, then each corner from the diagonal.
    """
    z = np.asarray(z, dtype=FIRE_FLOAT_DTYPE)
    ny, nx = z.shape
    out = np.zeros((ny + 2, nx + 2), dtype=FIRE_FLOAT_DTYPE)
    out[1:-1, 1:-1] = z
    out[1:-1, 0] = _ex(z[:, 0], z[:, 1])
    out[1:-1, -1] = _ex(z[:, -1], z[:, -2])
    out[0, 1:-1] = _ex(z[0, :], z[1, :])
    out[-1, 1:-1] = _ex(z[-1, :], z[-2, :])
    out[0, 0] = _ex(z[0, 0], z[1, 1])
    out[-1, 0] = _ex(z[-1, 0], z[-2, 1])
    out[0, -1] = _ex(z[0, -1], z[1, -2])
    out[-1, -1] = _ex(z[-1, -1], z[-2, -2])
    return out


def wrf_refinement_axis(n: int, r: int) -> tuple[np.ndarray, np.ndarray]:
    """``(cell, weight)`` for fire nodes ``1..n*r`` along one axis.

    interpolate_2d as interpolate_z2fire calls it, in REAL(4): cell ``i2``
    of the halo-extended axis (0 is the continued cell before the domain)
    starts at fire coordinate ``rio = rip1 + ir*(i2 - rip2)``, with
    ``rip1 = ifds + (ir-1)*0.5`` and ``rip2 = ids``, and covers nodes
    ``ceiling(rio) .. floor(rio) + ir``.  A node takes the LAST cell WRF's
    loop writes it from, the highest covering ``i2``, with weight
    ``tx = (i1 - rio) * (1./ir)``.
    """
    f32 = np.float32
    n, r = int(n), int(r)
    nf = n * r
    rip2 = f32(1)
    rip1 = f32(1) + f32(r - 1) * f32(0.5)
    rx = f32(1.0) / f32(r)
    lo, hi = 0, nf + 1  # snode: the fire tile plus a node past each end
    cell = np.full(nf, -1, dtype=np.int32)
    weight = np.zeros(nf, dtype=FIRE_FLOAT_DTYPE)
    for i2 in range(0, n + 1):  # its2 = ids - 1 .. ite2 - 1 = ide
        rio = rip1 + f32(r) * (f32(i2) - rip2)
        start = max(lo, int(np.ceil(rio)))
        end = min(hi, int(np.floor(rio)) + r)
        for i1 in range(max(start, 1), min(end, nf) + 1):
            cell[i1 - 1] = i2
            weight[i1 - 1] = (f32(i1) - rio) * rx
    if (cell < 0).any():  # pragma: no cover - every node is covered
        raise AssertionError("fire node outside every atmospheric cell")
    return cell, weight


def _require_mesh(ny: int, nx: int) -> None:
    if nx < 2 or ny < 2:
        raise _refusal(
            f"the fire mesh needs an atmospheric grid of at least 2 x 2 "
            f"cells, got {nx} x {ny}.",
            "WRF continues the terrain one cell past the domain from the "
            "two cells inside each edge (continue_at_boundary).")


def refine_zsf(hgt, sr_x: int, sr_y: int, *, xp=np):
    """``HGT_M`` on the fire mesh, as WRF's interpolate_z2fire refines it.

    float32 ``(ny * sr_y, nx * sr_x)``, on ``xp``'s device.  The edge band
    is WRF's linear continuation, so terrain falling to the sea at the
    domain edge continues below zero there, as WRF's does (interpolate_z2fire
    clamps only roughness, flag_z0 = 1, and the terrain is called with 0).
    """
    z = np.asarray(_host(hgt), dtype=np.float64).astype(FIRE_FLOAT_DTYPE)
    ny, nx = z.shape
    _require_mesh(ny, nx)
    za = continue_at_boundary(z)
    i2, tx = wrf_refinement_axis(nx, sr_x)
    j2, ty = wrf_refinement_axis(ny, sr_y)
    if xp is not np:
        from . import fire_mesh_gpu
        return fire_mesh_gpu.refine_zsf(za, i2, tx, j2, ty)
    f32 = np.float32
    x = tx[None, :]
    y = ty[:, None]
    rows, rows1 = j2[:, None], j2[:, None] + 1
    v00 = za[rows, i2[None, :]]
    v01 = za[rows1, i2[None, :]]
    v10 = za[rows, i2[None, :] + 1]
    v11 = za[rows1, i2[None, :] + 1]
    v = (f32(1.0) - x) * (f32(1.0) - y) * v00
    v = v + (f32(1.0) - x) * y * v01
    v = v + x * (f32(1.0) - y) * v10
    v = v + x * y * v11
    return v


def fire_mesh_spacing(d: float, r: int) -> np.float32:
    """WRF's fire-mesh spacing ``dxf = dx/ir`` (module_fr_fire_driver.F:566)."""
    return np.float32(d) / np.float32(int(r))


def fire_mesh_slope(zsf, dxf, dyf):
    """``(dz/dx, dz/dy)`` as module_fr_fire_model.F:194-208 forms them.

    ZSF continued one node past the mesh, then
    ``(zsf(i+1,j)-zsf(i-1,j))/(2.*fdx)``; float32, on ``zsf``'s device.
    """
    f32 = np.float32
    twodx, twody = f32(2.0) * f32(dxf), f32(2.0) * f32(dyf)
    _require_mesh(*zsf.shape)
    if _is_device(zsf):
        from . import fire_mesh_gpu
        return fire_mesh_gpu.slope(zsf, twodx, twody)
    z = np.asarray(zsf, dtype=FIRE_FLOAT_DTYPE)
    ze = continue_at_boundary(z)
    dzdx = (ze[1:-1, 2:] - ze[1:-1, :-2]) / twodx
    dzdy = (ze[2:, 1:-1] - ze[:-2, 1:-1]) / twody
    return dzdx, dzdy


def refine_nearest(field, sr_x: int, sr_y: int):
    """Each fire sub-cell takes its parent atmospheric cell's category.

    uint8 on ``field``'s device (the nearest mass point to a sub-cell
    centre is always its parent).
    """
    if _is_device(field):
        from . import fire_mesh_gpu
        return fire_mesh_gpu.repeat(field, sr_x, sr_y)
    values = np.asarray(field).astype(FIRE_CATEGORY_DTYPE)
    return np.repeat(np.repeat(values, int(sr_y), axis=0), int(sr_x), axis=1)


def dominant_fuel(fire, sr_x: int, sr_y: int) -> np.ndarray:
    """Each atmospheric cell's most common sub-cell category (lowest wins).

    The counts are integers, so their order of summation cannot move them;
    the argmax is :func:`woof.static.build.dominant_category`'s, on the host.
    """
    xp = _xp_of(fire)
    ny, nx = fire.shape[0] // sr_y, fire.shape[1] // sr_x
    blocks = fire.reshape(ny, sr_y, nx, sr_x)
    counts = xp.stack([(blocks == category).sum(axis=(1, 3))
                       for category in FUEL_CATEGORIES])
    from .build import dominant_category
    return dominant_category(_host(counts))


# --- the build ---------------------------------------------------------------

def build_fuel_fields(grid, fields: Mapping[str, np.ndarray],
                      request: FuelStaticsRequest, *,
                      landuse_attrs: Mapping[str, object],
                      coverage_report: MutableMapping[str, object]
                      | None = None, device: str | None = None,
                      resident: bool = False) -> dict:
    """The fuel statics of ``request`` for ``grid``, from built fields.

    ``fields`` must hold the domain's ``LU_INDEX`` and ``HGT_M`` as the
    build finished them (overlay included).  ``landuse_attrs`` is the
    land-use legend of the same build
    (:meth:`woof.static.build.GeogSelection.landuse_global_attrs`).

    The fire mesh is built on ``device`` (:func:`array_module`), by
    default the request's.  On ``auto``, a device that fails (an NVRTC or
    driver error, an out-of-memory card) is reported once as a WORKAROUND
    and the CPU body builds the same words instead.  The result is host
    NumPy arrays unless ``resident`` asks to keep the fire-mesh fields on
    the device the work ran on (CuPy on the GPU route), for a consumer that
    integrates them there.  ``NFUEL_CAT`` is float64 on the host either
    way, like every atmospheric static field.
    """
    device = request.device if device is None else device
    xp = array_module(device)
    if xp is not np and device == "auto":
        try:
            return _build_fuel_fields(grid, fields, request, xp,
                                      landuse_attrs=landuse_attrs,
                                      coverage_report=coverage_report,
                                      resident=resident)
        except _gpu_failure_types() as error:
            import sys
            print(f"[fuel] WORKAROUND: the fire mesh is built on the CPU "
                  f"({type(error).__name__}: {error}); the GPU and CPU "
                  "bodies give the same words", file=sys.stderr)
            xp = np
    return _build_fuel_fields(grid, fields, request, xp,
                              landuse_attrs=landuse_attrs,
                              coverage_report=coverage_report,
                              resident=resident)


def _build_fuel_fields(grid, fields, request, xp, *, landuse_attrs,
                       coverage_report, resident) -> dict:
    missing = [name for name in ("LU_INDEX", "HGT_M") if name not in fields]
    if missing:
        raise KeyError(f"fuel statics need the built fields {missing}")
    sr_x, sr_y = request.sr_x, request.sr_y
    shape = (int(grid.e_sn) - 1, int(grid.e_we) - 1)
    for name in ("LU_INDEX", "HGT_M"):
        if np.shape(fields[name]) != shape:
            raise ValueError(
                f"{name} has shape {np.shape(fields[name])}, the grid's "
                f"mass points are {shape}")
    _require_mesh(*shape)
    table = crosswalk_table()
    receipt: dict[str, object] = {
        "source": "landuse",
        "mminlu": str(landuse_attrs.get("MMINLU")),
        "crosswalk": CROSSWALK_PATH.name,
        "crosswalk_sha256": table["sha256"],
        "sr_x": sr_x, "sr_y": sr_y,
        "fire_mesh_shape": [shape[0] * sr_y, shape[1] * sr_x],
        "device": "cpu" if xp is np else "gpu",
    }

    def landuse_fuel() -> np.ndarray:
        receipt["mminlu"] = _legend(landuse_attrs.get("MMINLU"))[0]
        return nfuel_from_lu_index(fields["LU_INDEX"], landuse_attrs)

    def on_mesh(atmospheric):
        return refine_nearest(xp.asarray(atmospheric.astype(
            FIRE_CATEGORY_DTYPE)), sr_x, sr_y)

    if request.landfire is None:
        atmospheric = landuse_fuel()
        fire = on_mesh(atmospheric)
    else:
        codes = xp.asarray(
            sample_landfire(request.landfire, grid, sr_x, sr_y, xp=xp))
        mapped, valid, kind = landfire_codes_to_fuel(
            codes, request.landfire_kind)
        if not bool(valid.any()):
            raise _refusal(
                f"the LANDFIRE raster {request.landfire} has no fuel code "
                "anywhere on this domain's fire mesh.  Check that it "
                "covers the domain, or drop it to use the land-use "
                "crosswalk.",
                "Every cell would take the land-use crosswalk, so the "
                "result would be the land-use map under the raster's name.")
        sampled = int(valid.sum())
        # The land-use crosswalk only fills the raster's gaps, so a
        # raster that covers the mesh needs no table for the legend.
        fire = (mapped if sampled == valid.size else xp.where(
            valid, mapped, on_mesh(landuse_fuel())))
        atmospheric = dominant_fuel(fire, sr_x, sr_y)
        receipt.update({
            "source": "landfire",
            "landfire": {
                "path": str(request.landfire.resolve()),
                "bytes": request.landfire.stat().st_size,
                "sha256": landfire_sha256(request.landfire),
                "kind": kind,
                "sampled_cells": sampled,
                "landuse_filled_cells": int(valid.size) - sampled,
            },
        })
    zsf = refine_zsf(fields["HGT_M"], sr_x, sr_y, xp=xp)
    dzdx, dzdy = fire_mesh_slope(zsf, fire_mesh_spacing(grid.dx, sr_x),
                                 fire_mesh_spacing(grid.dy, sr_y))
    if coverage_report is not None and "fuel" in coverage_report:
        raise ValueError("duplicate static source-coverage field 'fuel'")
    out = {"NFUEL_CAT": np.asarray(atmospheric, dtype=np.float64),
           "NFUEL_CAT_FIRE": fire, "ZSF": zsf, "DZDXF": dzdx, "DZDYF": dzdy}
    if not resident:
        out = {name: _host(value) for name, value in out.items()}
    # Last, so a device failure part way leaves no receipt behind for the
    # CPU body's to collide with.
    if coverage_report is not None:
        coverage_report["fuel"] = receipt
    return out


def refresh_fuel_fields(fields: Mapping[str, np.ndarray], grid, selection,
                        *, landuse_attrs: Mapping[str, object],
                        coverage_report: MutableMapping[str, object]
                        | None = None) -> dict:
    """``fields`` with the fuel statics rebuilt from its final land use.

    For a route that overlays ``[static.highres]`` land use or terrain
    after :func:`woof.static.build.build_static`: the fuel map and slope
    must come from the land use and terrain the run integrates.  A
    selection without a fuel request returns ``fields`` itself.
    """
    request = getattr(selection, "fuel", None)
    if request is None:
        return fields
    out = dict(fields)
    out.update(build_fuel_fields(grid, fields, request,
                                 landuse_attrs=landuse_attrs,
                                 coverage_report=coverage_report))
    return out


def fuel_summary(fields: Mapping[str, object]) -> dict[str, object]:
    """Category histograms and slope statistics of built fuel statics."""
    names = crosswalk_table()["fuel_categories"]
    host = {name: _host(fields[name]) for name in FUEL_FIELDS}

    def histogram(values) -> list[dict[str, object]]:
        total = int(values.size)
        counts = np.bincount(values.astype(np.int64).ravel(),
                             minlength=NO_FUEL + 1)
        return [{"category": category, "name": names[str(category)],
                 "cells": int(counts[category]),
                 "percent": 100.0 * int(counts[category]) / total}
                for category in FUEL_CATEGORIES if counts[category]]

    dzdx = host["DZDXF"].astype(np.float64)
    dzdy = host["DZDYF"].astype(np.float64)
    slope = np.hypot(dzdx, dzdy)
    zsf = host["ZSF"]
    return {
        "atmospheric_shape": list(host["NFUEL_CAT"].shape),
        "fire_mesh_shape": list(host["NFUEL_CAT_FIRE"].shape),
        "atmospheric": histogram(host["NFUEL_CAT"]),
        "fire_mesh": histogram(host["NFUEL_CAT_FIRE"]),
        "slope": {
            "dzdx_min": float(dzdx.min()), "dzdx_max": float(dzdx.max()),
            "dzdy_min": float(dzdy.min()), "dzdy_max": float(dzdy.max()),
            "magnitude_mean": float(slope.mean()),
            "magnitude_p95": float(np.percentile(slope, 95.0)),
            "magnitude_max": float(slope.max()),
            "max_degrees": float(np.degrees(np.arctan(slope.max()))),
        },
        "terrain_m": {"min": float(zsf.min()), "max": float(zsf.max())},
    }
