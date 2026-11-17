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
footprints from files the caller downloaded.  Every contributing source's
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
        raise ValueError("the domain crosses the antimeridian; build its "
                         "buildings product in two halves")
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
        x, y = tr.transform(
            np.array([bbox.lon_min, bbox.lon_max, bbox.lon_max,
                      bbox.lon_min]),
            np.array([bbox.lat_min, bbox.lat_min, bbox.lat_max,
                      bbox.lat_max]))
        box = (float(np.min(x)), float(np.min(y)), float(np.max(x)),
               float(np.max(y)))
    return [fp for fp in footprints if _bbox_hit(fp, box)], crs


def _ndsm_heights_for(footprints, crs, windows, resolution_m):
    """Footprints without a height attribute get the GEM-15 nDSM median
    over their own pixels, from the first window that has one.

    The median is read on the window's lattice split by up to 4 (nearest,
    so no height is invented), so a footprint smaller than a 15 m pixel
    still finds its pixels.  Returns ``(footprints, n_from_ndsm)``.
    """
    from woof.static.buildings import (Footprint, _footprint_median,
                                       _transform_footprints)

    need = [n for n, fp in enumerate(footprints) if math.isnan(fp.height)]
    heights = {}
    for window in windows:
        todo = [n for n in need if n not in heights]
        if not todo:
            break
        factor = min(4, max(1, int(math.ceil(window.px / resolution_m))))
        fine = window.supersampled(factor)
        values = np.asarray(fine.height, np.float64)
        moved = _transform_footprints([footprints[n] for n in todo], crs,
                                      window.crs)
        for n, fp in zip(todo, moved):
            median = _footprint_median(fp, values, fine)
            if not math.isnan(median):
                heights[n] = median
    out = [Footprint(fp.polygons, heights[n], fp.levels, fp.source)
           if n in heights else fp for n, fp in enumerate(footprints)]
    return out, len(heights)


def build_domain_buildings(grid, output, *, footprints: Sequence = (),
                           gem15: bool = True, cache_root=None,
                           offline: bool = False,
                           resolution_m: float | None = None,
                           storey_height_m: float = 3.0,
                           default_height_m: float | None = None,
                           min_area_m2: float = 200.0,
                           grid_note: str = "", urlopen=None, index=None
                           ) -> tuple[Path, dict]:
    """Write the buildings product of one WRF domain (``grid``, a
    :class:`woof.static.projection.ProjectedGrid`) to ``output``.

    ``footprints`` holds ``(path, source)`` pairs.  ``resolution_m`` is
    the footprint lattice spacing (default a quarter of the smaller grid
    spacing, at least 1 m).  Returns ``(path, receipt)``; the receipt
    (also stored in the product) names the box, the GEM-15 windows, the
    footprint counts and where each building's height came from.
    """
    if not gem15 and not footprints:
        raise ValueError("nothing to build from: give footprint files, or "
                         "leave GEM-15 on")
    model = ProjectionGrid(grid)
    if resolution_m is None:
        resolution_m = max(1.0, min(model.dx, model.dy) / 4.0)
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
    windows: list[HeightRaster] = []
    if gem15:
        from woof.static import gem15 as g15
        from woof.static.highres_production import default_highres_cache_root
        root = Path(cache_root) if cache_root else default_highres_cache_root()
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
        rasters = windows
    else:
        like = utm_raster(bbox.lon_min, bbox.lat_min, bbox.lon_max,
                          bbox.lat_max, resolution_m)
        rasters, files = [], []
        for path, source in footprints:
            fps, crs = _read_in_box(Path(path), source, bbox)
            n_ndsm = 0
            if windows:
                fps, n_ndsm = _ndsm_heights_for(fps, crs, windows,
                                                resolution_m)
            burnt, burn = rasterize_footprints(
                fps, crs, like, storey_height_m=storey_height_m,
                default_height_m=default_height_m)
            # Heights the nDSM supplied arrive as attributes; name them.
            burn["height_from"]["attribute"] -= n_ndsm
            burn["height_from"]["gem15_ndsm_median"] = n_ndsm
            sources = set(burnt.sources)
            if n_ndsm:
                sources.add("gem15-ndsm")
            rasters.append(HeightRaster(burnt.height, burnt.x0, burnt.y0,
                                        burnt.px, burnt.py, burnt.crs,
                                        tuple(sorted(sources))))
            files.append(dict(burn, file=Path(path).name, source=source,
                              crs=str(crs), sources=sorted(sources)))
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
    parser.add_argument("--no-gem15", action="store_true",
                        help="do not read GEM-15: heights then come from "
                             "the footprints' own attributes or storeys "
                             "only")
    parser.add_argument("--cache", type=Path, default=None, metavar="DIR",
                        help="GEM-15 cache root (default: the high-"
                             "resolution terrain cache)")
    parser.add_argument("--offline", action="store_true",
                        help="read GEM-15 from the cache only")
    parser.add_argument("--resolution", type=float, default=None,
                        metavar="M",
                        help="footprint lattice spacing in metres "
                             "(default: a quarter of the grid spacing)")
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

    from woof.static.projection import grids_from_wps_namelist

    grids = grids_from_wps_namelist(args.namelist)
    number = len(grids) if args.domain is None else int(args.domain)
    if not 1 <= number <= len(grids):
        print(f"woof buildings: --domain {number} is not in "
              f"{args.namelist} ({len(grids)} domains)", file=sys.stderr)
        return 2
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
            min_area_m2=args.min_area,
            grid_note=f"{args.namelist.name} d{number:02d}")
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"woof buildings: {exc}", file=sys.stderr)
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
