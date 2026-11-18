"""Table-driven lake state: Rust CF reads/remapping, CUDA column binding.

Automatic sources are public and keyless. Licensed sources require a local
file and exact licence acceptance. Credentials are never discovered.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import tomllib
from urllib.request import urlopen
import subprocess

import numpy as np

TABLE = Path(__file__).resolve().parents[1] / 'data/surface_obs/lake-state-sources.v1.toml'
FIELDS = ('water_temperature', 'bottom_temperature', 'mixed_layer_depth',
          'temperature_shape_factor', 'ice_thickness', 'ice_fraction', 'ice_temperature')


def source_rows(path=TABLE):
    doc = tomllib.loads(Path(path).read_text())
    if doc.get('schema') != 'gpuwm-lake-state-sources-v1':
        raise ValueError('unrecognized lake-state source schema')
    rows = {r['id']: r for r in doc['source']}
    if len(rows) != len(doc['source']):
        raise ValueError('duplicate lake-state source ids')
    return rows


def utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def sample_field(latitude, longitude, values, target_latitude, target_longitude, *, scale=1., offset=0., zero_roundoff_tolerance=0.):
    """Marshal arrays only; interpolation and SI conversion run in Rust."""
    from woof.ingest.cpu_backend import masked_fields_cpu_backend, host_step_workers
    backend = masked_fields_cpu_backend()
    try:
        entry = backend._library.gpuwm_lake_state_sample_f64
    except AttributeError as e:
        raise RuntimeError(f'{backend.path} lacks gpuwm_lake_state_sample_f64; rebuild tools/grib1_bridge') from e
    la, lo, data, tla, tlo = [np.ascontiguousarray(v, dtype=np.float64) for v in
              (latitude, longitude, values, target_latitude, target_longitude)]
    if la.ndim != 1 or lo.ndim != 1 or data.shape != (la.size, lo.size) or tla.shape != tlo.shape:
        raise ValueError('lake source axes/field or target coordinates have incompatible dimensions')
    out = np.empty(tla.shape, np.float64)
    ptr = ctypes.c_void_p
    entry.argtypes = [ptr, ctypes.c_size_t, ptr, ctypes.c_size_t, ptr, ctypes.c_double,
                      ctypes.c_double, ptr, ptr, ctypes.c_size_t, ptr, ctypes.c_size_t]
    entry.restype = ctypes.c_int32
    rc = entry(ptr(la.ctypes.data), la.size, ptr(lo.ctypes.data), lo.size, ptr(data.ctypes.data),
               scale, offset, ptr(tla.ctypes.data), ptr(tlo.ctypes.data), tla.size, ptr(out.ctypes.data), host_step_workers(backend))
    if rc:
        raise ValueError(f'lake-state sampler refused malformed coordinates (native status {rc})')
    if zero_roundoff_tolerance:
        normalize = backend._library.gpuwm_lake_state_zero_roundoff_f64
        normalize.argtypes = [ptr, ctypes.c_size_t, ctypes.c_double, ctypes.c_size_t]
        normalize.restype = ctypes.c_int32
        if normalize(ptr(out.ctypes.data), out.size, zero_roundoff_tolerance, host_step_workers(backend)):
            raise ValueError('lake source zero-roundoff tolerance must be finite and nonnegative')
    return out


def cyclic_longitude(longitude, values):
    """Close a whole-globe longitude axis with its first column one period on.

    A global 0.25-degree axis holds 1440 columns, so the last column sits a
    step short of the first one's twin and the sampler (no extrapolation)
    left the cells in that one-column seam unbound.  A regional axis is
    returned unchanged.
    """
    lon = np.asarray(longitude, np.float64)
    data = np.asarray(values, np.float64)
    if lon.ndim != 1 or lon.size < 2 or data.ndim != 2 or data.shape[1] != lon.size:
        return lon, data
    step = np.diff(lon)
    if not np.all(np.abs(step - step[0]) <= 1e-9) or abs(abs(step[0]) * lon.size - 360.) > 1e-6:
        return lon, data
    return np.append(lon, lon[-1] + step[0]), np.concatenate([data, data[:, :1]], axis=1)


def decode_field(path, row, field, valid_time, lat, lon):
    from woof.netcdf_bridge import Dataset
    with Dataset(path) as ds:
        times = tuple(utc(v) for v in ds.variables[row['time']].times())
        admitted = [(t, i) for i, t in enumerate(times) if t <= utc(valid_time)]
        if not admitted:
            raise ValueError(f'{path}: no lake analysis at or before {valid_time}')
        selected, index = max(admitted)
        age = (utc(valid_time)-selected).total_seconds()/3600
        if age > row['max_age_hours']:
            raise ValueError(f'{path}: lake analysis is {age:g} h old, row limit {row["max_age_hours"]} h')
        variable = ds.variables[field['variable']]
        axes = (row['latitude'], row['longitude'])
        expected = (row['time'], *axes)
        if variable.dimensions != expected:
            raise ValueError(f'{path}: {variable.name} axes {variable.dimensions} != {expected}')
        if variable.units not in (field['units'], *field.get('unit_aliases', ())):
            raise ValueError(f'{path}: {variable.name} units {variable.units!r} != {field["units"]!r}')
        longitude, grid = ds.variables[axes[1]][:], variable[index]
        if row.get('acquisition') == 'zarr':
            longitude, grid = cyclic_longitude(longitude, grid)
        result = sample_field(ds.variables[axes[0]][:], longitude,
            grid, lat, lon, scale=field.get('scale', 1.), offset=field.get('offset', 0.),
            zero_roundoff_tolerance=field.get('zero_roundoff_tolerance', 0.))
    return result, dict(field=field['name'], variable=field['variable'], file=str(path), sha256=sha256(path),
                        valid_time=selected.isoformat(), age_hours=age, zero_roundoff_tolerance=field.get('zero_roundoff_tolerance', 0.), covered_targets=int(verification_scores(result, result)['count']))


def acquire_field(row, field, valid_time, bounds, cache):
    stamp = utc(valid_time).replace(hour=row.get('analysis_hour', 12), minute=0, second=0, microsecond=0)
    if stamp > utc(valid_time):
        stamp -= timedelta(days=1)
    url = field['url'].format(**field, time=stamp.strftime('%Y-%m-%dT%H:%M:%SZ'), **bounds)
    path = cache / (hashlib.sha256(url.encode()).hexdigest()+'.nc')
    if not path.exists():
        partial = path.with_suffix(f'.{os.getpid()}.part')
        try:
            with urlopen(url, timeout=30) as response, partial.open('wb') as out:
                for block in iter(lambda: response.read(1024*1024), b''):
                    out.write(block)
            with partial.open('rb') as f:
                if f.read(3) not in (b'CDF', b'\x89HD'):
                    raise ValueError(f'{url}: response is not NetCDF')
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    return path, url



def acquire_zarr(row, valid_time, cache):
    """One native Zarr extraction of every field of a ``zarr`` row.

    The time is the latest hour at or before initialization that the store
    has actually written.  The store's declared stop runs ahead of its chunks
    (ARCO declared ERA5T through 2026-11-10 while 2026-11-09 00Z had no chunk
    yet; rw_zarr then serves the fill value), so an hour where any field is
    entirely missing is passed over: the search steps back a day at a time,
    within the row's max_age_hours, to a written hour, then bisects the hours
    between it and the last unwritten one (the store fills in time order), so
    the state is the latest written hour, not up to 23 h older.  The request
    is always the whole globe from 180 W, so a lake cell is sampled off the
    same window whichever other cells are asked for (the X12 rule below);
    decode_field closes the cyclic seam with a wrapped column.  A full ARCO
    field is one chunk, so this costs no more than a crop.
    """
    from woof.fetch import ERA5_PRESSURE_LEVELS_HPA
    from woof.netcdf_bridge import Dataset
    from woof.zarr_bridge import extract_regular_zarr, zarr_coverage
    coverage = zarr_coverage(row['store'])
    valid = utc(valid_time)
    selected = min(valid.replace(minute=0, second=0, microsecond=0), utc(coverage['accepted_stop']))

    def attempt(when):
        request = {'store': row['store'], 'times': [when.strftime('%Y-%m-%dT%H:%M:%SZ')],
                   'area': [-90., -180., 90., 180.], 'expected_levels_hpa': list(ERA5_PRESSURE_LEVELS_HPA),
                   'fields': [{'source': f['variable'], 'output': f['variable'], 'units': f['units'],
                               'pressure': False} for f in row['field']]}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = cache / f'{row["id"]}-{key}.nc'
        ident = f'{row["store"]}@{request["times"][0]}'
        if path.exists():
            return path, ident
        stage = cache / f'.{key}.{os.getpid()}'
        stage.mkdir(parents=True, exist_ok=True)
        try:
            staged = stage / path.name
            extract_regular_zarr(request, request_path=stage / 'request.json', output=staged,
                                 progress=lambda line: None)
            # Every field: a partly written hour cached here would bind
            # all-missing fields for every later run at this time.
            with Dataset(staged) as ds:
                written = all(bool(np.isfinite(np.asarray(ds.variables[f['variable']][0])).any())
                              for f in row['field'])
            if written:
                staged.replace(path)
                return path, ident
        finally:
            for leftover in stage.iterdir():
                leftover.unlink()
            stage.rmdir()
        return None

    tried, missed, found = [], None, None
    while (valid - selected).total_seconds() / 3600 <= row['max_age_hours']:
        found = attempt(selected)
        if found:
            break
        tried.append(selected.strftime('%Y-%m-%dT%H:%M:%SZ'))
        missed = selected
        selected -= timedelta(days=1)
    if not found:
        raise ValueError(f'{row["store"]}: no hour within {row["max_age_hours"]} h before {valid.isoformat()} '
                         f'is written (tried {", ".join(tried) or "none"})')
    if missed is not None:
        written_hour = selected
        while (gap := int((missed - written_hour).total_seconds() // 3600)) > 1:
            middle = written_hour + timedelta(hours=gap // 2)
            try:
                hit = attempt(middle)
            except SOURCE_FAILURES:
                # A refinement probe that fails keeps the written hour in hand
                # rather than refuse a run that already has a valid state.
                break
            if hit:
                written_hour, found = middle, hit
            else:
                missed = middle
    return found


def apply_ice_authority(values, forcing_ice):
    """A model source fills lake temperatures; the forcing's ice decides frozen cells.

    Where the forcing has ice (an observed analysis such as IMS on the GFS
    route) the cell keeps it: the source's water profile is dropped there and
    its ice thickness and temperature are kept only where the source is frozen
    too, at the forcing's concentration.  Where the forcing is open water the
    source's ice is dropped, so a model never freezes or thaws a lake that the
    forcing's ice analysis decided.  Returns the per-rule cell counts.
    """
    icy = np.asarray(forcing_ice, np.float64) > 0
    for name in ('water_temperature', 'bottom_temperature', 'mixed_layer_depth', 'temperature_shape_factor'):
        if name in values:
            values[name] = np.where(icy, np.nan, values[name])
    depth = values.get('ice_thickness')
    frozen = icy & np.isfinite(depth) & (depth > 0) if depth is not None else np.zeros_like(icy)
    if depth is not None:
        values['ice_thickness'] = np.where(frozen, depth, np.nan)
    if 'ice_temperature' in values:
        values['ice_temperature'] = np.where(frozen, values['ice_temperature'], np.nan)
    values['ice_fraction'] = np.where(frozen, np.asarray(forcing_ice, np.float64), np.nan)
    return dict(forcing_ice_cells=int(icy.sum()), source_ice_kept_cells=int(frozen.sum()))


def coordinate_coverage(latitude, longitude, row):
    """Rust request envelope and in-source count; Python only marshals arrays."""
    from woof.ingest.cpu_backend import masked_fields_cpu_backend, host_step_workers
    backend = masked_fields_cpu_backend()
    arrays = [np.ascontiguousarray(v, np.float64) for v in (latitude, longitude)]
    if arrays[0].shape != arrays[1].shape:
        raise ValueError('lake request coordinates must have identical shapes')
    entry = backend._library.gpuwm_lake_state_coverage_f64
    ptr = ctypes.c_void_p
    entry.argtypes = [ptr, ptr, ctypes.c_size_t] + [ctypes.c_double]*4 + [ptr, ctypes.c_size_t]
    entry.restype = ctypes.c_int32
    out = np.empty(5, np.float64)
    rc = entry(*(ptr(a.ctypes.data) for a in arrays), arrays[0].size,
        row.get('south', -90.), row.get('north', 90.), row.get('west', -180.), row.get('east', 180.),
        ptr(out.ctypes.data), host_step_workers(backend))
    if rc:
        raise ValueError(f'lake request has no finite geographic coordinates (native status {rc})')
    return dict(zip(('south', 'north', 'west', 'east', 'covered'), map(float, out)))


#: ``lake_state_source`` chains: each source fills only the lake cells the
#: earlier ones left without a water temperature.
CHAINS = {'auto+era5': ('auto', 'era5')}

#: Physical admission ranges of the canonical fields (SI), judged in float32
#: as the surface carriers hold them.
RANGES = {'water_temperature': (270., 330.), 'bottom_temperature': (270., 330.),
          'mixed_layer_depth': (0., 2000.), 'temperature_shape_factor': (0.001, .999),
          'ice_thickness': (0., 20.), 'ice_fraction': (0., 1.), 'ice_temperature': (170., 273.16)}


#: The failures a source acquisition may raise (a store timeout is a
#: subprocess.TimeoutExpired, a SubprocessError).
SOURCE_FAILURES = (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError)


def out_of_range(name, data):
    """Finite values outside the field's admission range (float32 comparison)."""
    lower, upper = RANGES[name]
    donor = np.asarray(data, np.float32)
    return np.isfinite(donor) & ((donor < lower) | (donor > upper))


def load_chain(sources, valid_time, lat, lon, *, forcing_ice=None, **kwargs):
    """Observed sources first, then a model fallback for the cells they miss.

    A later source fills only the slots an earlier one left missing, so an
    earlier finite value (GLSEA ice on a cell whose water temperature is
    missing) is never overwritten.  The ``auto`` step keeps auto's tolerant
    contract: its out-of-range values are dropped with a warning, and the
    cells stay pending for the next source.  Every other step is explicit: its
    failure (a store timeout, a network error, no written hour) refuses the
    run with the chain and the step named.
    """
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    values, receipts = {}, []
    pending = np.ones(lat.shape, bool)
    for source in sources:
        if not pending.any():
            break
        ice = None if forcing_ice is None else np.asarray(forcing_ice)[pending]
        try:
            part, receipt = load_state(source, valid_time, lat[pending], lon[pending],
                                       forcing_ice=ice, **kwargs)
        except SOURCE_FAILURES as e:
            if source == 'auto':
                raise
            raise ValueError(
                f'lake_state_source={"+".join(sources)}: the {source} step failed for '
                f'{int(pending.sum())} of {lat.size} lake cells the earlier steps left unbound '
                f'({type(e).__name__}: {e}); an explicit source refuses rather than keep the forcing. '
                f'Use lake_state_source=auto to keep the forcing there') from e
        receipt['requested_cells'] = int(pending.sum())
        receipts.append(receipt)
        if not part:
            continue
        if source == 'auto':
            for name in list(part):
                bad = out_of_range(name, part[name]) if name in RANGES else None
                if bad is not None and bad.any():
                    lower, upper = RANGES[name]
                    # Dropped and passed on to the next step, which may bind
                    # them: whether the forcing is kept is unbound_cells' to say.
                    receipt.setdefault('dropped', []).append(
                        f'auto {name}: {int(bad.sum())} source values outside [{lower}, {upper}] '
                        'dropped and passed to the next step')
                    receipt['status'] = 'partial_forcing_fallback'
                    part[name] = np.where(bad, np.nan, np.asarray(part[name], np.float64))
        index = np.flatnonzero(pending)
        for name, data in part.items():
            merged = values.setdefault(name, np.full(lat.shape, np.nan))
            merged[index] = np.where(np.isfinite(merged[index]), merged[index], data)
        bound = np.zeros(lat.shape, bool)
        bound[index] = np.isfinite(part.get('water_temperature', np.full(index.size, np.nan)))
        if forcing_ice is not None and 'water_temperature' in part:
            bound[index] |= np.isfinite(part.get('ice_thickness', np.full(index.size, np.nan)))
        pending &= ~bound
    errors = [e for r in receipts for e in r.get('errors', ())]
    dropped = [e for r in receipts for e in r.get('dropped', ())]
    status = ('bound' if not pending.any() else 'partial_forcing_fallback') if values else 'forcing_fallback'
    return values, dict(schema='gpuwm.lake-init.v1', source='+'.join(sources), status=status,
                        chain=receipts, errors=errors, dropped=dropped, unbound_cells=int(pending.sum()),
                        target_cells=int(lat.size))


def load_state(source, valid_time, lat, lon, *, path='', accept_licence='', cache=None, forcing_ice=None):
    """Sample canonical fields and return provenance, ages, coverage and errors.

    ``forcing_ice`` (the forcing's ice concentration at the targets) is
    required by a row with ``ice_authority = "forcing"``.
    """
    started = time.monotonic()
    rows = source_rows()
    if source == 'auto':
        row = next((r for r in rows.values() if r.get('automatic', False)
                    and coordinate_coverage(lat, lon, r)['covered'] > 0), None)
        if row is None:
            return {}, dict(schema='gpuwm.lake-init.v1', status='outside_source_coverage')
    else:
        if source not in rows:
            raise ValueError(f'unknown lake-state source {source!r}; choices: auto, forcing, {", ".join(rows)}')
        row = rows[source]
    if row['research_only'] and accept_licence != row['licence']:
        raise ValueError(f'lake source {row["id"]} requires lake_state_accept_licence={row["licence"]!r}; its data cannot be a default')
    zarr = row.get('acquisition') == 'zarr'
    if row.get('ice_authority') == 'forcing' and forcing_ice is None:
        raise ValueError(f'lake source {row["id"]} needs the forcing ice concentration: its ice is subordinate to the forcing\'s')
    if not path and not zarr and any('url' not in f for f in row['field']):
        raise ValueError(f'lake source {row["id"]} requires lake_state_path to an acquired CF NetCDF file')
    cache = Path(cache or os.environ.get('WOOF_LAKE_STATE_CACHE', str(Path.home()/'.woof/cache/lake-state')))
    cache.mkdir(parents=True, exist_ok=True)
    # THE REQUEST IS THE SOURCE'S WHOLE DECLARED EXTENT, not the targets'
    # envelope.  Cropped to the targets, the same lake cell was sampled from
    # a different grid window depending on which other cells were asked for
    # in the same call, and the bilinear weights moved by a last-place bit:
    # the resident 5 km CONUS run and its slab-built streamed twin
    # disagreed by up to 3e-8 in 13 Great Lakes ice-fraction cells (X12).
    # A row with no declared extent keeps the envelope crop.
    if all(key in row for key in ('south', 'north', 'west', 'east')):
        bounds = {key: row[key] for key in ('south', 'north', 'west', 'east')}
    else:
        envelope = coordinate_coverage(lat, lon, row)
        bounds = dict(south=max(row.get('south', -90.), envelope['south']-.04),
                      north=min(row.get('north', 90.), envelope['north']+.04),
                      west=max(row.get('west', -180.), envelope['west']-.04),
                      east=min(row.get('east', 180.), envelope['east']+.04))
    date = utc(valid_time).date().isoformat()
    fields = [f for f in row['field'] if f.get('start', '0000') <= date <= f.get('end', '9999')]
    shared = None
    if zarr and not path:
        try:
            shared = acquire_zarr(row, valid_time, cache)
        except SOURCE_FAILURES as e:
            raise ValueError(f'lake source {row["id"]}: {row["store"]} unavailable '
                             f'({type(e).__name__}: {e})') from e
    def read(field):
        p, url = ((Path(path), None) if path else shared if shared
                  else acquire_field(row, field, valid_time, bounds, cache))
        data, evidence = decode_field(p, row, field, valid_time, lat, lon)
        evidence['url'] = url
        return field['name'], data, evidence
    values, records, errors = {}, [], []
    with ThreadPoolExecutor(max_workers=max(1, min(4, len(fields)))) as executor:
        futures = [(f, executor.submit(read, f)) for f in fields]
        for f, future in futures:
            try:
                name, data, record = future.result()
                values[name] = data
                records.append(record)
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                errors.append(f'{f["name"]}: {type(e).__name__}: {e}')
    uncovered = {r['field']: int(lat.size)-r['covered_targets'] for r in records if r['covered_targets'] < lat.size}
    receipt = dict(schema='gpuwm.lake-init.v1', source=row['id'], licence=row['licence'],
        licence_url=row['licence_url'], citation=row['citation'], records=records, errors=errors,
        status=('partial_forcing_fallback' if errors or uncovered else 'bound') if values else 'forcing_fallback', uncovered_targets=uncovered, wall_seconds=time.monotonic()-started,
        target_cells=int(lat.size), interpolation='Rust local bilinear, finite donors only, no extrapolation',
        profile_rule='CLM lakeini surface-constrained profile; supplied bottom/mixed-layer/shape fields use an integral-shape power profile',
        ice_rule='thickness times horizontal concentration distributed top-down as equivalent ice volume; concentration alone retains the lakeini thickness prior, reported as unobserved')
    if errors and source != 'auto':
        raise ValueError('explicit lake-state source incomplete: '+'; '.join(errors))
    if row.get('ice_authority') == 'forcing' and values:
        receipt['ice_authority'] = dict(rule='forcing', **apply_ice_authority(values, forcing_ice))
    return values, receipt


def initialize_surface(fields, cfg, *, valid_time, latitude, longitude, cold_start=True):
    """Bind observations before either the LSM or lake cold start."""
    import cupy as cp
    source = getattr(cfg, 'lake_state_source', 'auto')
    if not cold_start or source == 'forcing':
        return None, dict(schema='gpuwm.lake-init.v1', status='retained_forcing', source=source)
    if valid_time is None or latitude is None or longitude is None:
        # A chain is explicit here: its model step cannot run without them.
        if source != 'auto':
            raise ValueError('explicit lake-state binding requires initialization time and surface coordinates')
        return None, dict(schema='gpuwm.lake-init.v1', status='missing_time_or_coordinates', source=source)
    shape = fields['tsk'].shape
    mask = fields['lakemask'] == 1
    if not bool(cp.any(mask).item()):
        return None, dict(schema='gpuwm.lake-init.v1', status='no_lake_cells')
    la, lo = cp.broadcast_arrays(cp.asarray(latitude), cp.asarray(longitude))
    if la.shape != shape:
        raise ValueError('lake initialization coordinates must match the surface grid')
    loader = (lambda *a, **kw: load_chain(CHAINS[source], *a, **kw)) if source in CHAINS else (
        lambda *a, **kw: load_state(source, *a, **kw))
    values, receipt = loader(valid_time, cp.asnumpy(la[mask]), cp.asnumpy(lo[mask]),
        path=getattr(cfg, 'lake_state_path', ''), accept_licence=getattr(cfg, 'lake_state_accept_licence', ''),
        # No xice: None, so a row whose ice is subordinate to the forcing's
        # refuses by name instead of reading an open-water forcing.
        forcing_ice=cp.asnumpy(fields['xice'][mask]) if 'xice' in fields else None)
    for error in receipt.get('errors', ()):
        print(f'WARNING lake initialization retained forcing: {error}', file=sys.stderr, flush=True)
    for note in receipt.get('dropped', ()):
        print(f'WARNING lake initialization: {note}', file=sys.stderr, flush=True)
    if receipt.get('uncovered_targets'):
        print(f'WARNING lake initialization retained forcing for uncovered targets: {receipt["uncovered_targets"]}', file=sys.stderr, flush=True)
    if receipt.get('unbound_cells'):
        print(f'WARNING lake initialization retained forcing for {receipt["unbound_cells"]} of '
              f'{receipt["target_cells"]} lake cells no {source} step bound', file=sys.stderr, flush=True)
    mapped = {}
    for name, data in values.items():
        target = cp.full(shape, cp.nan, cp.float32)
        donor = cp.asarray(data, cp.float32)
        lower, upper = RANGES[name]
        bad = cp.isfinite(donor) & ((donor < lower) | (donor > upper))
        count = int(bad.sum().item())
        if count:
            message = f'{name}: {count} source values outside [{lower}, {upper}]'
            # A chain's auto step already dropped its own out-of-range values
            # (load_chain), so what remains here came from an explicit step.
            if source != 'auto':
                raise ValueError(message)
            receipt.setdefault('errors', []).append(message)
            receipt['status'] = 'partial_forcing_fallback'
            print(f'WARNING lake initialization retained forcing: {message}', file=sys.stderr, flush=True)
        target[mask] = cp.where(bad, cp.nan, donor)
        mapped[name] = target
    water, fraction, depth = (mapped.get(k) for k in ('water_temperature', 'ice_fraction', 'ice_thickness'))
    if fraction is None and depth is not None:
        fraction = cp.where(cp.isfinite(depth), (depth > 0).astype(cp.float32), cp.nan)
        mapped['ice_fraction'] = fraction
        receipt['fraction_rule'] = 'source lake column with positive ice depth is ice-covered'
    if water is not None:
        valid = mask & cp.isfinite(water) & (water >= 270.) & (water <= 330.)
        fields['sst'][valid] = water[valid]
        fields['tsk'][valid] = water[valid]
        receipt['temperature_bound_cells'] = int(valid.sum().item())
    if fraction is not None:
        valid = mask & cp.isfinite(fraction) & (fraction >= 0) & (fraction <= 1)
        for name in ('xice', 'xicem'):
            fields[name][valid] = fraction[valid]
        icy = valid & (fraction > 0)
        ice_temp = mapped.get('ice_temperature')
        if ice_temp is not None:
            usable = icy & cp.isfinite(ice_temp) & (ice_temp <= 273.16) & (ice_temp >= 170)
            fields['tsk'][usable] = ice_temp[usable]
        fields['tsk'][icy] = cp.minimum(fields['tsk'][icy], cp.float32(273.15))
        receipt['ice_fraction_bound_cells'] = int(valid.sum().item())
        receipt['ice_thickness_unobserved_cells'] = int((icy & (~cp.isfinite(depth) if depth is not None else True)).sum().item())
    receipt_dir = Path(os.environ.get('WOOF_LAKE_RECEIPT_DIR', os.environ.get('WOOF_LAKE_STATE_CACHE', str(Path.home()/'.woof/cache/lake-state'))))
    receipt_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()[:16]
    receipt_path = receipt_dir/f'lake-init-{key}.json'
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(f'lake initialization: {receipt["status"]}; receipt={receipt_path}', file=sys.stderr, flush=True)
    return mapped or None, receipt


def decode_buoy_temperature(text: bytes, valid_time, *, tolerance_seconds=3600):
    """Rust decodes WTMP, sentinel values and nearest report time, never ATMP."""
    from woof.ingest.cpu_backend import masked_fields_cpu_backend
    entry = masked_fields_cpu_backend()._library.gpuwm_lake_ndbc_temperature_f64
    entry.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64,
                      ctypes.c_int64, ctypes.c_void_p]
    entry.restype = ctypes.c_int32
    raw = ctypes.create_string_buffer(text)
    result = np.empty(2, np.float64)
    rc = entry(ctypes.cast(raw, ctypes.c_void_p), len(text), int(utc(valid_time).timestamp()),
               tolerance_seconds, ctypes.c_void_p(result.ctypes.data))
    if rc:
        raise ValueError(f'NDBC WTMP unavailable or malformed in requested window (native status {rc})')
    return dict(valid_time=datetime.fromtimestamp(result[0], timezone.utc).isoformat(),
                water_temperature_k=float(result[1]))


def nearest_lake_value(latitude, longitude, values, mask, target_latitude, target_longitude, *, max_distance_km):
    """Match a buoy only to a nearby wet model cell, in Rust."""
    from woof.ingest.cpu_backend import masked_fields_cpu_backend
    arrays = [np.ascontiguousarray(v, np.float64) for v in (latitude, longitude, values, mask)]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('lake verification coordinates, mask and values must have identical shapes')
    entry = masked_fields_cpu_backend()._library.gpuwm_lake_state_nearest_f64
    ptr = ctypes.c_void_p
    entry.argtypes = [ptr]*4 + [ctypes.c_size_t] + [ctypes.c_double]*3 + [ptr]
    entry.restype = ctypes.c_int32
    out = np.empty(3, np.float64)
    rc = entry(*(ptr(a.ctypes.data) for a in arrays), arrays[0].size, target_latitude,
               target_longitude, max_distance_km, ptr(out.ctypes.data))
    if rc:
        raise ValueError(f'no finite lake value within {max_distance_km:g} km (native status {rc})')
    return dict(flat_index=int(out[0]), distance_km=float(out[1]), value=float(out[2]))


def diagnostic_values(values, mask, *, scale=1., offset=0.):
    """Rust wet-cell masking and affine display conversion, without a second decoder."""
    from woof.ingest.cpu_backend import masked_fields_cpu_backend, host_step_workers
    backend = masked_fields_cpu_backend()
    values, mask = [np.ascontiguousarray(v, np.float64) for v in (values, mask)]
    if values.shape != mask.shape:
        raise ValueError('lake diagnostic values and mask must have identical shapes')
    output = np.empty_like(values)
    entry = backend._library.gpuwm_lake_state_diagnostic_f64
    ptr = ctypes.c_void_p
    entry.argtypes = [ptr, ptr, ctypes.c_size_t, ctypes.c_double, ctypes.c_double, ptr, ctypes.c_size_t]
    entry.restype = ctypes.c_int32
    rc = entry(ptr(values.ctypes.data), ptr(mask.ctypes.data), values.size, scale, offset,
               ptr(output.ctypes.data), host_step_workers(backend))
    if rc:
        raise ValueError(f'lake diagnostic conversion failed (native status {rc})')
    return output


def verification_scores(prediction, reference):
    """Rust parallel paired finite-value reductions; missing pairs are excluded."""
    from woof.ingest.cpu_backend import masked_fields_cpu_backend, host_step_workers
    backend = masked_fields_cpu_backend()
    arrays = [np.ascontiguousarray(v, np.float64) for v in (prediction, reference)]
    if arrays[0].shape != arrays[1].shape:
        raise ValueError('lake verification arrays must have identical shapes')
    output = np.empty(5, np.float64)
    entry = backend._library.gpuwm_lake_state_scores_f64
    ptr = ctypes.c_void_p
    entry.argtypes = [ptr, ptr, ctypes.c_size_t, ptr, ctypes.c_size_t]
    entry.restype = ctypes.c_int32
    rc = entry(*(ptr(a.ctypes.data) for a in arrays), arrays[0].size, ptr(output.ctypes.data), host_step_workers(backend))
    if rc:
        raise ValueError(f'lake verification failed (native status {rc})')
    scores = dict(zip(('count', 'bias', 'mae', 'rmse', 'max_absolute_error'), map(float, output)))
    if scores['count'] == 0:
        scores.update({name: None for name in ('bias', 'mae', 'rmse', 'max_absolute_error')})
    return scores


def write_diagnostic_cf(path, *, temperature, ice_fraction, lake_mask, latitude, longitude, valid_time, source):
    """Rust-written CF product for the production renderer, wet cells only.

    Fahrenheit uses the renderer's production temperature ladder. This file
    is a diagnostic product, never an initialization input or a restart.
    """
    from woof.io.classic_product import ClassicProduct
    shape = np.shape(temperature)
    if len(shape) != 2 or any(np.shape(v) != shape for v in (ice_fraction, lake_mask, latitude, longitude)):
        raise ValueError('lake diagnostic fields must share a two-dimensional surface grid')
    temperature_f = diagnostic_values(temperature, lake_mask, scale=1.8, offset=-459.67)
    fraction = diagnostic_values(ice_fraction, lake_mask)
    with ClassicProduct(path) as ds:
        for name, size in (('time', 1), ('y', shape[0]), ('x', shape[1])):
            ds.createDimension(name, size)
        ds.setncatts({'Conventions': 'CF-1.8', 'source': source})
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.units = 'seconds since 1970-01-01 00:00:00'
        time_var.standard_name = 'time'
        time_var[:] = np.asarray([utc(valid_time).timestamp()], np.float64)
        for name, values, units, standard in (
                ('latitude', latitude, 'degrees_north', 'latitude'),
                ('longitude', longitude, 'degrees_east', 'longitude'),
                ('lake_temperature_f', temperature_f, 'degF', ''),
                ('lake_ice_fraction', fraction, '1', 'lake_ice_area_fraction')):
            var = ds.createVariable(name, 'f8', ('y', 'x'))
            var.units = units
            if standard:
                var.standard_name = standard
            var[:] = values
    return Path(path)
