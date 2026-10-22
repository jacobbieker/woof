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
  a local file, read with rasterio.  Each fire-mesh cell centre is
  projected through the grid's own CRS (:func:`woof.static.highres._grid_crs`,
  the one the ``[static.highres]`` reprojection uses) into the raster's
  CRS, and the cell takes the pixel it lands in, by nearest neighbour.
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

The fields, all float64 like every other static field:

* ``NFUEL_CAT`` ``(ny, nx)``: the atmospheric grid.  On the land-use route
  it is the crosswalk of ``LU_INDEX``.  On the LANDFIRE route it is the
  dominant fire-mesh category of the cell's ``sr_y * sr_x`` sub-cells
  (lowest category wins ties, as :func:`woof.static.build.dominant_category`
  breaks them);
* ``NFUEL_CAT_FIRE`` ``(ny * sr_y, nx * sr_x)``: the refined fire mesh.
  Sub-cell ``(J, I)`` is centred at 1-based mass coordinate
  ``(0.5 + (J + 0.5) / sr_y, 0.5 + (I + 0.5) / sr_x)``, so a cell's
  sub-cells tile it exactly.  Sampling is nearest neighbour.  From
  ``LU_INDEX``, every sub-cell takes its parent cell's value, because the
  nearest mass point to a sub-cell centre is always the parent;
* ``ZSF``: ``HGT_M`` refined bilinearly onto the fire mesh.  The outer
  half-cell band is extrapolated linearly from the edge interval, so the
  edge slope is the edge cell's slope and not a flat shelf;
* ``DZDXF``, ``DZDYF``: the gradient of ``ZSF`` on the fire mesh.
  Differences are centred inside the mesh and one-sided at its edge, at
  spacings ``dx / sr_x`` and ``dy / sr_y``.  Like WRF-SFIRE's fire mesh
  and :mod:`woof.static.terrain_slope`, they are taken on the map plane,
  without the map factor.

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
from dataclasses import dataclass, replace
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
    FBFM40 GeoTIFF, or ``None`` to use the land-use crosswalk.
    """

    sr_x: int
    sr_y: int
    landfire: Path | None = None
    landfire_kind: str = "auto"

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


# --- LANDFIRE ----------------------------------------------------------------

def landfire_codes_to_fuel(codes, kind: str = "auto"
                           ) -> tuple[np.ndarray, np.ndarray, str]:
    """``(fuel, valid, kind)`` for raw LANDFIRE FBFM13/FBFM40 codes.

    ``codes`` is float with NaN where the raster gave no data.  Codes 1-13
    are FBFM13 and stay themselves.  Codes 101-204 are FBFM40 and go
    through the crosswalk.  The non-burnable codes 91-99 are 14 in both.
    ``valid`` is False on NaN and on codes <= 0 (LANDFIRE's -9999 fill).
    Any other code is refused, as is a mix of FBFM13 and FBFM40 codes.
    """
    if kind not in LANDFIRE_KINDS:
        raise ValueError(f"landfire kind must be one of {LANDFIRE_KINDS}")
    table = crosswalk_table()["landfire"]
    nonburnable = {int(code) for code in table["nonburnable"]}
    fbfm40 = {int(code): int(row["fuel"])
              for code, row in table["fbfm40"].items()}
    values = np.asarray(codes, dtype=np.float64)
    valid = np.isfinite(values) & (values > 0.0)
    raw = np.where(valid, values, 0.0)
    if np.any(raw != np.floor(raw)):
        raise ValueError("LANDFIRE fuel codes must be whole numbers")
    raw = raw.astype(np.int64)
    present = {int(code) for code in np.unique(raw[valid])}
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
    lookup = np.zeros(max(fbfm40) + 1, dtype=np.int64)
    lookup[1:14] = np.arange(1, 14)
    for code in nonburnable:
        lookup[code] = NO_FUEL
    for code, fuel in fbfm40.items():
        lookup[code] = fuel
    fuel = np.where(valid, lookup[raw], 0).astype(np.float64)
    return fuel, valid, kind


def fire_mesh_axis(n: int, sr: int) -> np.ndarray:
    """1-based mass coordinates of the ``n * sr`` fire sub-cell centres."""
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


def sample_landfire(path, grid, sr_x: int, sr_y: int) -> np.ndarray:
    """Raw raster codes at every fire sub-cell centre, nearest neighbour.

    float64 ``(ny * sr_y, nx * sr_x)``, NaN where the cell falls outside
    the raster or on its no-data value.  Only the window the domain covers
    is read, so a CONUS-wide product costs the domain's footprint.
    """
    try:
        import rasterio
        from pyproj import Transformer
        from rasterio.windows import Window
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise RuntimeError(_geog_unavailable()) from exc
    from .highres import _grid_crs

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"LANDFIRE fuel raster missing: {path}")
    nx, ny = int(grid.e_we) - 1, int(grid.e_sn) - 1
    try:
        grid_crs = _grid_crs(grid)
    except NotImplementedError:
        raise _refusal(
            f"a LANDFIRE raster cannot be placed on a "
            f"{getattr(grid, 'map_proj', '?')!r} grid; drop the raster to "
            "build the fuel map from the land-use crosswalk.",
            "The raster is reprojected through the grid's PROJ CRS, which "
            "woof defines for its lambert, mercator and polar grids "
            "only.") from None
    ref_x, ref_y = Transformer.from_crs(
        "EPSG:4326", grid_crs, always_xy=True).transform(
            grid.ref_lon, grid.ref_lat)
    xs = ref_x + (fire_mesh_axis(nx, sr_x) - float(grid.known_x)) * grid.dx
    ys = ref_y + (fire_mesh_axis(ny, sr_y) - float(grid.known_y)) * grid.dy
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    codes = np.full(mesh_x.shape, np.nan, dtype=np.float64)
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise ValueError(
                f"LANDFIRE raster {path} has no CRS; LANDFIRE products are "
                "delivered with one, so this is not a LANDFIRE GeoTIFF")
        source_x, source_y = Transformer.from_crs(
            grid_crs, dataset.crs, always_xy=True).transform(mesh_x, mesh_y)
        inverse = ~dataset.transform
        source_x, source_y = np.asarray(source_x), np.asarray(source_y)
        col_f = inverse.a * source_x + inverse.b * source_y + inverse.c
        row_f = inverse.d * source_x + inverse.e * source_y + inverse.f
        finite = np.isfinite(col_f) & np.isfinite(row_f)
        col = np.full(codes.shape, -1, dtype=np.int64)
        row = np.full(codes.shape, -1, dtype=np.int64)
        col[finite] = np.floor(col_f[finite]).astype(np.int64)
        row[finite] = np.floor(row_f[finite]).astype(np.int64)
        inside = (finite & (col >= 0) & (col < dataset.width)
                  & (row >= 0) & (row < dataset.height))
        if not inside.any():
            return codes
        c0, c1 = int(col[inside].min()), int(col[inside].max())
        r0, r1 = int(row[inside].min()), int(row[inside].max())
        window = Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
        block = dataset.read(1, window=window).astype(np.float64)
        nodata = dataset.nodata
        if nodata is not None:
            block[block == nodata] = np.nan
        codes[inside] = block[row[inside] - r0, col[inside] - c0]
    return codes


# --- refinement onto the fire mesh -------------------------------------------

def refine_nearest(field, sr_x: int, sr_y: int) -> np.ndarray:
    """Each fire sub-cell takes its parent atmospheric cell's value."""
    values = np.asarray(field)
    return np.repeat(np.repeat(values, int(sr_y), axis=0), int(sr_x), axis=1)


def _bilinear_axis(n: int, sr: int):
    x = fire_mesh_axis(n, sr) - 1.0  # 0-based mass index
    if n == 1:
        zeros = np.zeros(x.shape, dtype=np.int64)
        return zeros, zeros, np.zeros(x.shape)
    i0 = np.clip(np.floor(x).astype(np.int64), 0, n - 2)
    return i0, i0 + 1, x - i0


def refine_bilinear(field, sr_x: int, sr_y: int) -> np.ndarray:
    """``field`` on the fire mesh: bilinear inside, linear past the edge."""
    z = np.asarray(field, dtype=np.float64)
    ny, nx = z.shape
    y0, y1, wy = _bilinear_axis(ny, int(sr_y))
    x0, x1, wx = _bilinear_axis(nx, int(sr_x))
    wy = wy[:, None]
    south = z[y0][:, x0] * (1.0 - wx) + z[y0][:, x1] * wx
    north = z[y1][:, x0] * (1.0 - wx) + z[y1][:, x1] * wx
    return south * (1.0 - wy) + north * wy


def fire_mesh_slope(zsf, dxf: float, dyf: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """``(dz/dx, dz/dy)`` of ``zsf`` at fire-mesh spacings ``dxf``, ``dyf``."""
    z = np.asarray(zsf, dtype=np.float64)
    dzdx = (np.gradient(z, float(dxf), axis=1) if z.shape[1] > 1
            else np.zeros_like(z))
    dzdy = (np.gradient(z, float(dyf), axis=0) if z.shape[0] > 1
            else np.zeros_like(z))
    return dzdx, dzdy


def dominant_fuel(fire, sr_x: int, sr_y: int) -> np.ndarray:
    """Each atmospheric cell's most common sub-cell category (lowest wins)."""
    values = np.asarray(fire)
    ny, nx = values.shape[0] // sr_y, values.shape[1] // sr_x
    blocks = values.reshape(ny, sr_y, nx, sr_x)
    counts = np.stack([(blocks == category).sum(axis=(1, 3))
                       for category in FUEL_CATEGORIES])
    from .build import dominant_category
    return dominant_category(counts)


# --- the build ---------------------------------------------------------------

def build_fuel_fields(grid, fields: Mapping[str, np.ndarray],
                      request: FuelStaticsRequest, *,
                      landuse_attrs: Mapping[str, object],
                      coverage_report: MutableMapping[str, object]
                      | None = None) -> dict[str, np.ndarray]:
    """The fuel statics of ``request`` for ``grid``, from built fields.

    ``fields`` must hold the domain's ``LU_INDEX`` and ``HGT_M`` as the
    build finished them (overlay included).  ``landuse_attrs`` is the
    land-use legend of the same build
    (:meth:`woof.static.build.GeogSelection.landuse_global_attrs`).
    """
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
    table = crosswalk_table()
    receipt: dict[str, object] = {
        "source": "landuse",
        "mminlu": str(landuse_attrs.get("MMINLU")),
        "crosswalk": CROSSWALK_PATH.name,
        "crosswalk_sha256": table["sha256"],
        "sr_x": sr_x, "sr_y": sr_y,
        "fire_mesh_shape": [shape[0] * sr_y, shape[1] * sr_x],
    }

    def landuse_fuel() -> np.ndarray:
        receipt["mminlu"] = _legend(landuse_attrs.get("MMINLU"))[0]
        return nfuel_from_lu_index(fields["LU_INDEX"], landuse_attrs)

    if request.landfire is None:
        atmospheric = landuse_fuel()
        fire = refine_nearest(atmospheric, sr_x, sr_y)
    else:
        codes = sample_landfire(request.landfire, grid, sr_x, sr_y)
        mapped, valid, kind = landfire_codes_to_fuel(
            codes, request.landfire_kind)
        if not valid.any():
            raise _refusal(
                f"the LANDFIRE raster {request.landfire} has no fuel code "
                "anywhere on this domain's fire mesh.  Check that it "
                "covers the domain, or drop it to use the land-use "
                "crosswalk.",
                "Every cell would take the land-use crosswalk, so the "
                "result would be the land-use map under the raster's name.")
        # The land-use crosswalk only fills the raster's gaps, so a
        # raster that covers the mesh needs no table for the legend.
        fire = (mapped if valid.all() else np.where(
            valid, mapped, refine_nearest(landuse_fuel(), sr_x, sr_y)))
        atmospheric = dominant_fuel(fire, sr_x, sr_y)
        receipt.update({
            "source": "landfire",
            "landfire": {
                "path": str(request.landfire.resolve()),
                "bytes": request.landfire.stat().st_size,
                "sha256": landfire_sha256(request.landfire),
                "kind": kind,
                "sampled_cells": int(valid.sum()),
                "landuse_filled_cells": int((~valid).sum()),
            },
        })
    zsf = refine_bilinear(fields["HGT_M"], sr_x, sr_y)
    dzdx, dzdy = fire_mesh_slope(zsf, grid.dx / sr_x, grid.dy / sr_y)
    if coverage_report is not None:
        if "fuel" in coverage_report:
            raise ValueError("duplicate static source-coverage field 'fuel'")
        coverage_report["fuel"] = receipt
    return {"NFUEL_CAT": atmospheric, "NFUEL_CAT_FIRE": fire, "ZSF": zsf,
            "DZDXF": dzdx, "DZDYF": dzdy}


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


def fuel_summary(fields: Mapping[str, np.ndarray]) -> dict[str, object]:
    """Category histograms and slope statistics of built fuel statics."""
    names = crosswalk_table()["fuel_categories"]

    def histogram(values) -> list[dict[str, object]]:
        values = np.asarray(values)
        total = int(values.size)
        rows = []
        for category in FUEL_CATEGORIES:
            count = int(np.count_nonzero(values == category))
            if count:
                rows.append({"category": category,
                             "name": names[str(category)], "cells": count,
                             "percent": 100.0 * count / total})
        return rows

    slope = np.hypot(np.asarray(fields["DZDXF"]), np.asarray(fields["DZDYF"]))
    return {
        "atmospheric_shape": list(np.shape(fields["NFUEL_CAT"])),
        "fire_mesh_shape": list(np.shape(fields["NFUEL_CAT_FIRE"])),
        "atmospheric": histogram(fields["NFUEL_CAT"]),
        "fire_mesh": histogram(fields["NFUEL_CAT_FIRE"]),
        "slope": {
            "dzdx_min": float(np.min(fields["DZDXF"])),
            "dzdx_max": float(np.max(fields["DZDXF"])),
            "dzdy_min": float(np.min(fields["DZDYF"])),
            "dzdy_max": float(np.max(fields["DZDYF"])),
            "magnitude_mean": float(np.mean(slope)),
            "magnitude_p95": float(np.percentile(slope, 95.0)),
            "magnitude_max": float(np.max(slope)),
            "max_degrees": float(np.degrees(np.arctan(np.max(slope)))),
        },
        "terrain_m": {"min": float(np.min(fields["ZSF"])),
                      "max": float(np.max(fields["ZSF"]))},
    }
