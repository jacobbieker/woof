"""Radar ducting diagnostic from WRF/WOOF history files.

``woof ducting PATH... --out DIR`` reads ``wrfout`` frames and writes, per
frame, a NetCDF of 2-D duct fields and, for the whole run, a CSV time series
of how much of a region is ducted.

Modified refractivity on the model's mass levels::

    M = 77.6 / T * (P + 4810 e / T) + 0.157 z       (P, e hPa; T K; z m AGL)

A trapping layer is a run of levels where dM/dz < 0.  Its top is the local M
minimum that ends the run.  The duct is SURFACE-BASED when M at that top is
at or below M at the lowest model level (base 0), otherwise ELEVATED, with its
base where M, followed down from the top, first equals M_top (linear
interpolation between levels).  The strength is the largest M between base and
top minus M_top.  Per column the strongest duct (largest strength) is
reported, with the number of trapping layers and the base of the lowest one.
Only levels at or below ``max_height`` are searched; a trapping layer still
descending at that height ends there.

Evaporation ducts (typically under about 40 m over the sea) lie below what a
model's lowest levels resolve and are not diagnosed.
"""
import csv
import json
from pathlib import Path
import re

import numpy as np

from woof.core import constants as c

DUCT_NONE, DUCT_SURFACE, DUCT_ELEVATED = 0, 1, 2

_NAME = re.compile(r'wrfout_d(\d+)_(\d{4}-\d\d-\d\d)_(\d\d)[:_](\d\d)[:_](\d\d)')

_ATTRS = dict(
    duct_type=('1', '0 none, 1 surface-based, 2 elevated (strongest duct)'),
    duct_base=('m', 'duct base height above ground'),
    duct_top=('m', 'duct top height above ground'),
    duct_thickness=('m', 'duct top minus base'),
    duct_strength=('M-units', 'largest M between base and top minus M at the top'),
    n_trapping_layers=('1', 'trapping layers (dM/dz < 0) below the search height'),
    trap_base_lowest=('m', 'start height of the lowest trapping layer'))


def modified_refractivity(theta_pert, p_pert, p_base, qvapor, ph_pert, ph_base, hgt):
    """Return ``(M, z)`` on mass levels, shape ``(nz, ny, nx)``.

    ``theta_pert`` is WRF's ``T`` (potential temperature minus 300 K), the
    pressures are in Pa, ``ph_*`` are on w-levels (``nz + 1``), ``hgt`` is the
    terrain height in metres; ``z`` is metres above ground.
    """
    theta = np.asarray(theta_pert, dtype=np.float64) + 300.0
    p = np.asarray(p_pert, dtype=np.float64) + np.asarray(p_base, dtype=np.float64)
    q = np.maximum(np.asarray(qvapor, dtype=np.float64), 0.0)
    phi = (np.asarray(ph_pert, dtype=np.float64) + np.asarray(ph_base, dtype=np.float64)) / c.G
    z = 0.5 * (phi[1:] + phi[:-1]) - np.asarray(hgt, dtype=np.float64)[None]
    temp = theta * (p / c.P0) ** c.RCP
    e = q * p / (0.622 + q) / 100.0
    m = 77.6 / temp * (p / 100.0 + 4810.0 * e / temp) + 0.157 * z
    return m, z


def diagnose_ducts(m, z, max_height=3000.0):
    """Diagnose ducts in each column of ``m`` and ``z`` (shape ``(nz, ...)``).

    Returns a dict of arrays shaped like one level: ``duct_type`` (int8; 0
    none, 1 surface-based, 2 elevated), ``duct_base``, ``duct_top``,
    ``duct_thickness`` and ``duct_strength`` (all NaN without a duct),
    ``n_trapping_layers`` and ``trap_base_lowest`` (start height of the lowest
    trapping layer, NaN without one).
    """
    m = np.asarray(m, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    nz = m.shape[0]
    shape = m.shape[1:]
    m = m.reshape(nz, -1)
    z = z.reshape(nz, -1)
    ncol = m.shape[1]
    cols = np.arange(ncol)
    # trap[k]: the layer from level k to k+1 is trapping and within max_height.
    trap = np.zeros((nz + 1, ncol), bool)
    if nz > 1:
        trap[:nz - 1] = (m[1:] < m[:-1]) & (z[1:] <= max_height)
    count = np.zeros(ncol, np.int32)
    best = np.full(ncol, -np.inf)
    dtype = np.zeros(ncol, np.int8)
    base = np.full(ncol, np.nan)
    top = np.full(ncol, np.nan)
    low_base = np.full(ncol, np.nan)
    for k in range(nz):
        began = trap[k] & ~trap[k - 1] if k else trap[0].copy()
        count += began
        low_base = np.where(began & np.isnan(low_base), z[k], low_base)
        if not k:
            continue
        idx = cols[trap[k - 1] & ~trap[k]]
        if not idx.size:
            continue
        mtop = m[k, idx]
        surface = mtop <= m[0, idx]
        # Highest level below the top whose M is at or below M_top; the base
        # is the crossing between it and the level above it.
        below = m[:k, idx] <= mtop[None]
        j = np.where(below.any(axis=0), k - 1 - np.argmax(below[::-1], axis=0), 0)
        m_lo, m_hi = m[j, idx], m[j + 1, idx]
        z_lo, z_hi = z[j, idx], z[j + 1, idx]
        rise = np.where(m_hi > m_lo, m_hi - m_lo, 1.0)
        z_base = np.where(surface, 0.0, z_lo + (mtop - m_lo) / rise * (z_hi - z_lo))
        inside = (z[:k + 1, idx] > z_base[None]) | surface[None]
        peak = np.where(inside, m[:k + 1, idx], -np.inf).max(axis=0)
        strength = peak - mtop
        better = strength > best[idx]
        sel = idx[better]
        best[sel] = strength[better]
        dtype[sel] = np.where(surface[better], DUCT_SURFACE, DUCT_ELEVATED)
        base[sel] = z_base[better]
        top[sel] = z[k, sel]
    ducted = dtype > 0
    return dict(
        duct_type=dtype.reshape(shape),
        duct_base=base.reshape(shape),
        duct_top=top.reshape(shape),
        duct_thickness=(top - base).reshape(shape),
        duct_strength=np.where(ducted, best, np.nan).reshape(shape),
        n_trapping_layers=count.reshape(shape),
        trap_base_lowest=low_base.reshape(shape))


def _ring_inside(lon, lat, ring):
    ring = np.asarray(ring, dtype=np.float64)[:, :2]
    x, y = lon.ravel(), lat.ravel()
    inside = np.zeros(x.shape, bool)
    xj, yj = ring[-1]
    for xi, yi in ring:
        if yi != yj:
            inside ^= ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        xj, yj = xi, yi
    return inside.reshape(lon.shape)


def _polygons(geojson):
    kind = geojson.get('type')
    if kind == 'FeatureCollection':
        for feature in geojson.get('features', []):
            yield from _polygons(feature)
    elif kind == 'Feature':
        yield from _polygons(geojson['geometry'])
    elif kind == 'Polygon':
        yield geojson['coordinates']
    elif kind == 'MultiPolygon':
        yield from geojson['coordinates']
    else:
        raise ValueError(f'region GeoJSON must hold Polygon or MultiPolygon geometry, not {kind!r}')


def region_mask(lon, lat, geojson=None, landmask=None, sea_only=False):
    """Boolean mask of columns inside the polygon(s) (all, if none given) and,
    with ``sea_only``, over sea (``landmask < 0.5``)."""
    lon, lat = np.asarray(lon), np.asarray(lat)
    mask = np.ones(lon.shape, bool)
    if geojson is not None:
        mask = np.zeros(lon.shape, bool)
        for rings in _polygons(geojson):
            poly = _ring_inside(lon, lat, rings[0])
            for hole in rings[1:]:
                poly &= ~_ring_inside(lon, lat, hole)
            mask |= poly
    if sea_only:
        if landmask is None:
            raise ValueError('--sea-only needs LANDMASK in the history file')
        mask &= np.asarray(landmask) < 0.5
    return mask


def region_summary(fields, mask):
    """Coverage (%) of any / surface-based / elevated ducts, and the median
    top and strength of the ducted columns, inside ``mask``."""
    kind = fields['duct_type'][mask]
    n = kind.size
    ducted = kind > 0
    nan = float('nan')
    return dict(
        n_columns=int(n),
        any_duct_pct=100.0 * ducted.sum() / n if n else nan,
        surface_duct_pct=100.0 * (kind == DUCT_SURFACE).sum() / n if n else nan,
        elevated_duct_pct=100.0 * (kind == DUCT_ELEVATED).sum() / n if n else nan,
        median_top_m=float(np.median(fields['duct_top'][mask][ducted])) if ducted.any() else nan,
        median_strength_M=float(np.median(fields['duct_strength'][mask][ducted])) if ducted.any() else nan)


def parse_frame_name(path):
    """``(domain, 'YYYY-MM-DDTHH:MM:SS')`` from a ``wrfout_dNN_...`` name, with
    ``:`` or ``_`` between the time fields; ``None`` if it does not parse."""
    found = _NAME.search(Path(path).name)
    if not found:
        return None
    dom, day, hh, mm, ss = found.groups()
    return int(dom), f'{day}T{hh}:{mm}:{ss}'


def find_frames(paths, pattern='wrfout_d*'):
    """History frames under folders (searched recursively, skipping ``*.json``
    sidecars) and files named directly, ordered by domain then time."""
    files = set()
    for entry in map(Path, paths):
        if entry.is_dir():
            files.update(p.resolve() for p in entry.rglob(pattern) if p.is_file() and p.suffix != '.json')
        elif entry.is_file():
            files.add(entry.resolve())
        else:
            raise ValueError(f'{entry} does not exist')
    return sorted(files, key=lambda p: (parse_frame_name(p) or (0, ''), str(p)))


def process_frame(path, out_dir, max_height=3000.0, geojson=None, sea_only=False):
    """Write one frame's duct fields; return its summary row and file."""
    from netCDF4 import Dataset
    parsed = parse_frame_name(path)
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        if parsed:
            domain, when = parsed
        elif 'Times' in ds.variables:
            domain = int(getattr(ds, 'GRID_ID', 1))
            when = b''.join(ds['Times'][0]).decode().replace('_', 'T')
        else:
            raise ValueError(f'{path}: no time in the file name or a Times variable')
        lat, lon = np.asarray(ds['XLAT'][0]), np.asarray(ds['XLONG'][0])
        landmask = np.asarray(ds['LANDMASK'][0]) if 'LANDMASK' in ds.variables else None
        hgt = np.asarray(ds['HGT'][0])
        phi = (np.asarray(ds['PH'][0]) + np.asarray(ds['PHB'][0])) / c.G - hgt[None]
        median_z = np.median(phi.reshape(phi.shape[0], -1), axis=1)
        nk = min(int(np.searchsorted(median_z, max_height + 500.0)) + 1, phi.shape[0] - 1)
        m, z = modified_refractivity(
            ds['T'][0, :nk], ds['P'][0, :nk], ds['PB'][0, :nk], ds['QVAPOR'][0, :nk],
            ds['PH'][0, :nk + 1], ds['PHB'][0, :nk + 1], hgt)
    fields = diagnose_ducts(m, z, max_height)
    mask = region_mask(lon, lat, geojson, landmask, sea_only)
    target = Path(out_dir) / f'ducts_d{domain:02d}_{when.replace("-", "").replace(":", "")}.nc'
    with Dataset(target, 'w') as out:
        out.createDimension('south_north', lat.shape[0])
        out.createDimension('west_east', lat.shape[1])
        dims = ('south_north', 'west_east')
        for name, arr, units in (('XLAT', lat, 'degrees_north'), ('XLONG', lon, 'degrees_east')):
            var = out.createVariable(name, 'f4', dims, zlib=True)
            var[:] = arr
            var.units = units
        for name, arr in fields.items():
            var = out.createVariable(name, arr.dtype, dims, zlib=True)
            var[:] = arr
            var.units, var.description = _ATTRS[name]
        var = out.createVariable('region_mask', 'i1', dims, zlib=True)
        var[:] = mask
        var.description = '1 where the column is in the summary region'
        out.valid_time = when
        out.max_height_m = float(max_height)
        out.source_file = Path(path).name
        out.description = ('Radar ducting from modified refractivity; evaporation ducts '
                           '(below about 40 m) are not resolved and not diagnosed')
    row = dict(time=when, domain=domain, file=Path(path).name)
    row.update(region_summary(fields, mask))
    return row, target


def register_cli(sub):
    parser = sub.add_parser(
        'ducting', help='radar ducting (modified refractivity) fields and regional coverage from wrfout frames')
    parser.add_argument('paths', nargs='+', type=Path, metavar='PATH', help='run folders (searched recursively) or wrfout files')
    parser.add_argument('--out', type=Path, required=True, help='directory for the per-frame NetCDF files and ducting_summary.csv')
    parser.add_argument('--max-height', type=float, default=3000.0, help='search ducts up to this height above ground in metres')
    parser.add_argument('--region', type=Path, help='GeoJSON Polygon/MultiPolygon (lon/lat) limiting the summary region')
    parser.add_argument('--sea-only', action='store_true', help='summarise sea columns only (LANDMASK < 0.5)')
    parser.add_argument('--glob', default='wrfout_d*', help='file-name pattern searched under folders')
    parser.set_defaults(func=main)


def main(args):
    if args.max_height <= 0:
        raise ValueError('--max-height must be positive')
    geojson = json.loads(args.region.read_text()) if args.region else None
    files = find_frames(args.paths, args.glob)
    if not files:
        raise ValueError('no wrfout frames found under ' + ', '.join(map(str, args.paths)))
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in files:
        row, target = process_frame(path, args.out, args.max_height, geojson, args.sea_only)
        rows.append(row)
        print(f'{target.name}: any {row["any_duct_pct"]:.1f}% surface {row["surface_duct_pct"]:.1f}% '
              f'elevated {row["elevated_duct_pct"]:.1f}%')
    fields = ['time', 'domain', 'file', 'n_columns', 'any_duct_pct', 'surface_duct_pct',
              'elevated_duct_pct', 'median_top_m', 'median_strength_M']
    with (args.out / 'ducting_summary.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {len(rows)} frames and {args.out / "ducting_summary.csv"}')
