"""Build a ``woof-buildings.v1`` product for one WRF domain (``woof buildings``).

The door from real data to the resolved obstacles (``obstacle_opt = 1``)
and to the gridded urban morphology: it takes one domain of a WPS namelist,
range-reads the GEM-15 ``dsm``/``dtm`` blocks under it (cached per user, the
same cache the terrain uses), reads any footprint files the caller names,
and aggregates them onto the domain's own mass cells
(:func:`woof.static.buildings.aggregate` through
:class:`woof.static.buildings.ProjectionGrid`).

Two ways to get building heights, chosen by what the caller gives:

* GEM-15 only: the nDSM (``dsm - dtm``) filtered by
  :func:`woof.static.buildings.ndsm_heights` (height window, minimum object
  area), on GEM-15's own 15 m UTM lattice;
* footprints (with or without GEM-15): the footprint polygons are burned on
  one UTM lattice over the domain (``resolution_m``) with their own height,
  else the GEM-15 nDSM median over their own pixels, else storeys x storey
  height, else ``default_height_m``.  Trees and other non-buildings in the
  nDSM never become obstacles on this route: only footprints are burned.

Nothing here knows a place: the domain comes from the namelist and the
footprints from files the caller names or from the Overture fetch.  Every contributing source's
licence and attribution go into the product (``BUILDING_SOURCES``).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from woof.static.buildings import (BUILDING_SOURCES, HeightRaster,
                                   ProjectionGrid, aggregate,
                                   gem15_building_rasters,
                                   rasterize_footprints, read_footprints,
                                   utm_raster, write_product)

#: Default footprint lattice: a quarter of the grid spacing, but never
#: coarser than this, so a house-sized footprint still covers pixels and
#: the wall and frontal areas are not lost to the pixel size.
MAX_DEFAULT_RESOLUTION_M = 5.0
#: Largest footprint lattice built in one piece (pixels); a bigger domain
#: names a coarser ``resolution_m`` (``--resolution``).
MAX_LATTICE_PIXELS = 150_000_000

def domain_bbox(grid, *, halo_cells: float = 2.0, samples: int = 64):
    """Lon/lat box around a domain's mass cells plus ``halo_cells``.

    The perimeter of the cell-edge rectangle (WRF's 1-based mass points sit
    at x = i + 1, so the cells span x in [0.5, nx + 0.5]) is sampled
    densely and mapped through the domain's own ``ij_to_latlon``, so a
    curved projected edge is enclosed.
    """
    from woof.static.highres_fetch import FootprintBBox

    nx, ny = int(grid.e_we) - 1, int(grid.e_sn) - 1
    lo, hx, hy = 0.5 - halo_cells, nx + 0.5 + halo_cells, ny + 0.5 + halo_cells
    t = np.linspace(0.0, 1.0, samples)
    xs = np.concatenate([lo + (hx - lo) * t, np.full(samples, hx),
                         hx - (hx - lo) * t, np.full(samples, lo)])
    ys = np.concatenate([np.full(samples, lo), lo + (hy - lo) * t,
                         np.full(samples, hy), hy - (hy - lo) * t])
    lat, lon = grid.ij_to_latlon(xs, ys)
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    if float(np.ptp(lon)) > 180.0:
        raise ValueError("the domain crosses the antimeridian or contains "
                         "a pole; build its buildings product in pieces "
                         "that do neither")
    return FootprintBBox(float(lat.min()), float(lat.max()),
                         float(lon.min()), float(lon.max()))


def parse_footprint_spec(spec: str) -> tuple[Path, str]:
    """``PATH`` or ``PATH:SOURCE`` (a :data:`BUILDING_SOURCES` key)."""
    path, sep, source = str(spec).rpartition(":")
    if sep and source in BUILDING_SOURCES:
        return Path(path), source
    if sep and source and "/" not in source and "\\" not in source:
        raise ValueError(f"unknown footprint source {source!r} in {spec!r}; "
                         f"choose one of {sorted(BUILDING_SOURCES)}")
    return Path(spec), "user"


def _lonlat_crs(crs) -> bool:
    if crs in ("OGC:CRS84", "EPSG:4326"):
        return True
    from pyproj import CRS
    return bool(CRS.from_user_input(crs).is_geographic)


def _read_in_box(path: Path, source: str, bbox):
    """A footprint file's footprints that touch ``bbox`` (lon/lat)."""
    from woof.static.buildings import _bbox_hit

    footprints, crs = read_footprints(path, source=source)
    if _lonlat_crs(crs):
        box = (bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max)
    else:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        # Densified edges: projected parallels curve, so the corners alone
        # can miss part of the box.
        box = tuple(float(v) for v in tr.transform_bounds(
            bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max,
            densify_pts=64))
    return [fp for fp in footprints if _bbox_hit(fp, box)], crs


def _sub_window(window: HeightRaster, bounds, pad: int = 1):
    """The part of ``window`` under ``bounds`` (x0, y0, x1, y1 in its
    CRS) plus ``pad`` pixels, or None when they do not meet."""
    rows, cols = window.shape
    minx, miny, maxx, maxy = bounds
    c0 = max(0, int(math.floor((minx - window.x0) / window.px)) - pad)
    c1 = min(cols, int(math.ceil((maxx - window.x0) / window.px)) + pad)
    r0 = max(0, int(math.floor((window.y0 - maxy) / window.py)) - pad)
    r1 = min(rows, int(math.ceil((window.y0 - miny) / window.py)) + pad)
    if c1 <= c0 or r1 <= r0:
        return None
    return HeightRaster(window.height[r0:r1, c0:c1],
                        window.x0 + c0 * window.px,
                        window.y0 - r0 * window.py, window.px, window.py,
                        window.crs, window.sources)


def _ndsm_medians_for(footprints, crs, windows, resolution_m):
    """GEM-15 nDSM median over each height-less footprint's own pixels,
    from the first window that has one (NaN elsewhere).

    The median is read around each footprint on the window's lattice split
    by up to 4 (nearest, so no height is invented), so a footprint smaller
    than a 15 m pixel still finds its pixels; only the few pixels under
    the footprint are split, never the whole window.
    """
    from woof.static.buildings import (_footprint_median,
                                       _transform_footprints)

    medians = np.full(len(footprints), np.nan)
    need = [n for n, fp in enumerate(footprints) if math.isnan(fp.height)]
    for window in windows:
        todo = [n for n in need if math.isnan(medians[n])]
        if not todo:
            break
        factor = min(4, max(1, int(math.ceil(window.px / resolution_m))))
        moved = _transform_footprints([footprints[n] for n in todo], crs,
                                      window.crs)
        for n, fp in zip(todo, moved):
            sub = _sub_window(window, fp.bounds())
            if sub is None:
                continue
            fine = sub.supersampled(factor)
            medians[n] = _footprint_median(
                fp, np.asarray(fine.height, np.float64), fine)
    return medians


def _one_lattice(windows: Sequence[HeightRaster], bbox):
    """GEM-15 windows as rasters :func:`aggregate` can mosaic.

    GEM-15 cells share one 15 m lattice within a UTM zone, but cells either
    side of a zone edge are on different lattices and overlap; such windows
    are resampled (nearest, the taller value where they overlap) onto one
    15 m UTM lattice over the domain.  Windows of one zone pass unchanged.
    """
    from woof.static.buildings import _same_crs
    if all(_same_crs(w.crs, windows[0].crs) for w in windows[1:]):
        return list(windows)
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    like = utm_raster(bbox.lon_min, bbox.lat_min, bbox.lon_max,
                      bbox.lat_max, windows[0].px)
    out = np.full(like.shape, np.nan, np.float32)
    for w in windows:
        part = np.full(like.shape, np.nan, np.float32)
        reproject(np.asarray(w.height, np.float32), part,
                  src_transform=from_origin(w.x0, w.y0, w.px, w.py),
                  src_crs=w.crs, src_nodata=np.nan,
                  dst_transform=from_origin(like.x0, like.y0, like.px,
                                            like.py),
                  dst_crs=like.crs, dst_nodata=np.nan,
                  resampling=Resampling.nearest)
        out = np.fmax(out, part)
    sources = tuple(sorted({s for w in windows for s in w.sources}))
    return [HeightRaster(out, like.x0, like.y0, like.px, like.py, like.crs,
                         sources)]


def _default_resolution(dx: float, dy: float) -> float:
    return max(1.0, min(min(dx, dy) / 4.0, MAX_DEFAULT_RESOLUTION_M))


def build_domain_buildings(grid, output, *, footprints: Sequence = (),
                           gem15: bool = True, cache_root=None,
                           offline: bool = False,
                           resolution_m: float | None = None,
                           storey_height_m: float = 3.0,
                           default_height_m: float | None = None,
                           min_area_m2: float = 200.0,
                           grid_note: str = "", urlopen=None, index=None,
                           overture: str | None = None,
                           overture_filesystem=None
                           ) -> tuple[Path, dict]:
    """Write the buildings product of one WRF domain (``grid``, a
    :class:`woof.static.projection.ProjectedGrid`) to ``output``.

    ``footprints`` holds ``(path, source)`` pairs.  ``resolution_m`` is
    the footprint lattice spacing (default a quarter of the smaller grid
    spacing, between 1 m and :data:`MAX_DEFAULT_RESOLUTION_M`); a lattice
    over :data:`MAX_LATTICE_PIXELS` refuses and asks for a coarser one.  Returns ``(path, receipt)``; the receipt
    (also stored in the product) names the box, the GEM-15 windows, the
    footprint counts and where each building's height came from.
    ``overture`` (a release id or ``"latest"``) first fetches the Overture
    Maps buildings under the domain plus a halo into ``cache_root`` (see
    :mod:`woof.static.overture`) and adds them as an ``overture`` file;
    its provenance sidecar goes into the receipt.
    """
    footprints = list(footprints)
    if not gem15 and not footprints and overture is None:
        raise ValueError("nothing to build from: give footprint files "
                         "(--footprints, --overture), or leave GEM-15 on")
    for name, value in (("storey_height_m", storey_height_m),
                        ("default_height_m", default_height_m)):
        if value is not None and not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be > 0, got {value!r}")
    model = ProjectionGrid(grid)
    if resolution_m is None:
        resolution_m = _default_resolution(model.dx, model.dy)
    if not (math.isfinite(resolution_m) and resolution_m > 0):
        raise ValueError(f"resolution_m must be > 0, got {resolution_m!r}")
    bbox = domain_bbox(grid)
    receipt: dict = {"bbox": dict(lat_min=bbox.lat_min,
                                  lat_max=bbox.lat_max,
                                  lon_min=bbox.lon_min,
                                  lon_max=bbox.lon_max),
                     "grid": dict(nx=model.nx, ny=model.ny, dx=model.dx,
                                  dy=model.dy),
                     "footprint_resolution_m": float(resolution_m)}
    from woof.static.highres_production import default_highres_cache_root
    root = Path(cache_root) if cache_root else default_highres_cache_root()
    if overture is not None:
        from woof.static.overture import fetch_overture_buildings
        # Four cells plus ~100 m (0.001 degree) of halo.
        fetch_box = domain_bbox(grid, halo_cells=4.0).padded(0.001)
        got, sidecar = fetch_overture_buildings(
            fetch_box, root / "overture", release=overture, offline=offline,
            filesystem=overture_filesystem)
        footprints.append((got, "overture"))
        receipt["overture"] = sidecar
    windows: list[HeightRaster] = []
    if gem15:
        from woof.static import gem15 as g15
        fetched, gem = {}, {}
        for source_id in ("gem15-dsm", "gem15-dtm"):
            files, manifest = g15.fetch_gem15_windows(
                bbox, root, source_id, urlopen=urlopen, offline=offline,
                index=index)
            fetched[source_id] = [item.path for item in files]
            gem[source_id] = dict(
                windows=[item.receipt() for item in files],
                unpublished_squares=[
                    f"{lat}/{lon}" for lat, lon
                    in manifest.get("gem15_unpublished_squares", ())])
        # Alone, the nDSM drops objects under min_area_m2; with footprints
        # (which decide what is a building) it only supplies heights.
        windows, gem["ndsm"] = gem15_building_rasters(
            fetched["gem15-dsm"], fetched["gem15-dtm"],
            min_area_m2=0.0 if footprints else min_area_m2)
        receipt["gem15"] = gem
    if not footprints:
        if not windows:
            raise ValueError("GEM-15 publishes no data under this domain and "
                             "no footprints were given")
        rasters = _one_lattice(windows, bbox)
    else:
        like = utm_raster(bbox.lon_min, bbox.lat_min, bbox.lon_max,
                          bbox.lat_max, resolution_m)
        if like.height.size > MAX_LATTICE_PIXELS:
            raise ValueError(
                f"a {resolution_m:g} m footprint lattice over this domain "
                f"has {like.height.size:,} pixels (most "
                f"{MAX_LATTICE_PIXELS:,}); pass a coarser --resolution "
                "(footprints much smaller than it are lost or grown to a "
                "pixel) or build the product for a smaller domain")
        rasters, files = [], []
        for path, source in footprints:
            fps, crs = _read_in_box(Path(path), source, bbox)
            medians = (_ndsm_medians_for(fps, crs, windows, resolution_m)
                       if windows else None)
            # The nDSM source joins only when a building whose height it
            # gave shows on the lattice (rasterize_footprints decides).
            burnt, burn = rasterize_footprints(
                fps, crs, like, storey_height_m=storey_height_m,
                default_height_m=default_height_m, ndsm_medians=medians)
            rasters.append(burnt)
            files.append(dict(burn, file=Path(path).name, source=source,
                              crs=str(crs)))
        receipt["footprint_files"] = files
    morph = aggregate(rasters, model)
    path = write_product(output, morph, grid_note=grid_note,
                         extra_receipt=receipt)
    return path, receipt


def _arguments(parser) -> None:
    parser.add_argument("--namelist", type=Path, required=True,
                        metavar="NAMELIST_WPS",
                        help="WPS namelist whose domain gets the product")
    parser.add_argument("--domain", type=int, default=None, metavar="N",
                        help="1-based domain number (default: the "
                             "innermost, the last one in the namelist)")
    parser.add_argument("--output", type=Path, required=True, metavar="NC",
                        help="buildings product to write (NetCDF); name "
                             "it in [[domain]] obstacle_file")
    parser.add_argument("--footprints", action="append", default=[],
                        metavar="PATH[:SOURCE]",
                        help="footprint file (GeoJSON or GeoParquet) and "
                             "its source, one of "
                             + ", ".join(sorted(BUILDING_SOURCES))
                             + " (default user); repeat for more files")
    parser.add_argument("--overture", nargs="?", const="latest", default=None,
                        metavar="RELEASE",
                        help="fetch the Overture Maps buildings under the "
                             "domain plus a halo (anonymous S3, ODbL 1.0) "
                             "into --cache; RELEASE is a release id or "
                             "latest (the default when given bare); "
                             "combines with --footprints")
    parser.add_argument("--no-gem15", action="store_true",
                        help="do not read GEM-15: heights then come from "
                             "the footprints' own attributes or storeys "
                             "only")
    parser.add_argument("--cache", type=Path, default=None, metavar="DIR",
                        help="GEM-15 and Overture cache root (default: "
                             "the high-resolution terrain cache)")
    parser.add_argument("--offline", action="store_true",
                        help="read GEM-15 and Overture from the cache "
                             "only")
    parser.add_argument("--resolution", type=float, default=None,
                        metavar="M",
                        help="footprint lattice spacing in metres "
                             "(default: a quarter of the grid spacing, "
                             "at most 5)")
    parser.add_argument("--storey-height", type=float, default=3.0,
                        metavar="M",
                        help="metres per storey for footprints with only a "
                             "storey count (default 3)")
    parser.add_argument("--default-height", type=float, default=None,
                        metavar="M",
                        help="height of footprints with no height, storeys "
                             "or GEM-15 value (default: such footprints "
                             "are left out)")
    parser.add_argument("--min-area", type=float, default=200.0,
                        metavar="M2",
                        help="GEM-15 alone: drop nDSM objects smaller than "
                             "this (default 200)")


def _run(args) -> int:
    import json
    import sys

    from woof.static.highres_refusal import HighresRefusal
    from woof.static.projection import grids_from_wps_namelist

    try:
        grids = grids_from_wps_namelist(args.namelist)
    except (OSError, ValueError, KeyError) as exc:
        print(f"woof buildings: cannot read {args.namelist}: {exc}",
              file=sys.stderr)
        return 2
    number = len(grids) if args.domain is None else int(args.domain)
    if not 1 <= number <= len(grids):
        print(f"woof buildings: --domain {number} is not in "
              f"{args.namelist} ({len(grids)} domains)", file=sys.stderr)
        return 2
    if (args.no_gem15 and args.overture is None
            and (args.offline or args.cache is not None)):
        print("woof buildings: --cache and --offline only apply to GEM-15 "
              "and Overture, and are ignored with --no-gem15 alone",
              file=sys.stderr)
    try:
        footprints = [parse_footprint_spec(s) for s in args.footprints]
        for path, _ in footprints:
            if not path.is_file():
                raise FileNotFoundError(f"footprint file {path} not found")
        path, receipt = build_domain_buildings(
            grids[number - 1], args.output, footprints=footprints,
            gem15=not args.no_gem15, cache_root=args.cache,
            offline=args.offline, resolution_m=args.resolution,
            storey_height_m=args.storey_height,
            default_height_m=args.default_height,
            min_area_m2=args.min_area, overture=args.overture,
            grid_note=f"{args.namelist.name} d{number:02d}")
    except (ValueError, KeyError, OSError, HighresRefusal) as exc:
        # PreparationRefusal (a ValueError) carries its own remedy line.
        print(f"woof buildings: {exc}", file=sys.stderr)
        remedy = getattr(exc, "remedy", None)
        if remedy:
            print(remedy, file=sys.stderr)
        return 2
    print(f"woof buildings: wrote {path} for d{number:02d}")
    print(json.dumps(receipt, indent=1, sort_keys=True, default=str))
    return 0


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "buildings", help="build the buildings product (resolved "
                          "obstacles, urban morphology) for one domain "
                          "from GEM-15 and footprint files")
    _arguments(parser)
    parser.set_defaults(func=_run)


__all__ = ["build_domain_buildings", "domain_bbox", "parse_footprint_spec",
           "register_cli"]
