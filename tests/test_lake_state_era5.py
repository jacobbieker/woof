"""ERA5 (ARCO) FLake lake state: request, ice authority and the observed-first chain."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import woof.ingest.lake_state as obs
from woof.ingest.lake_state import source_rows


def test_era5_row_is_a_keyless_model_row_with_the_era5_land_fields():
    rows = source_rows()
    row = rows['era5']
    assert row['acquisition'] == 'zarr' and row['store'].startswith('https://')
    assert not row.get('automatic', False) and not row['research_only'] and row['citation']
    assert row['ice_authority'] == 'forcing'
    assert ({f['name'] for f in row['field']}
            == {f['name'] for f in rows['era5-land']['field']})
    # The default stays GLSEA-only: no other row joins the automatic chain.
    assert [r['id'] for r in rows.values() if r.get('automatic')] == ['glsea']


def fake_bridge(monkeypatch, stop, requests, unwritten=(), missing=()):
    """rw_zarr stand-in: an unwritten hour comes back as the fill value.

    ``unwritten`` is a collection of hours or a predicate on the hour text;
    ``missing`` names fields that stay fill on every hour ``unwritten``
    rejects, the others being written (a partly written hour)."""
    import woof.netcdf_bridge as nb
    import woof.zarr_bridge as zb
    monkeypatch.setattr(zb, 'zarr_coverage', lambda store: dict(accepted_stop=stop))
    def extract(request, *, request_path, output, progress):
        requests.append(request)
        Path(output).write_text(request['times'][0])
        return {}
    gone = unwritten if callable(unwritten) else (lambda hour: hour in unwritten)
    class Fake:
        def __init__(self, path):
            hour = Path(path).read_text()
            def fill(name):
                return np.nan if gone(hour) and (not missing or name in missing) else 280.
            self.variables = {f['source']: np.full((1, 2, 2), fill(f['source'])) for f in requests[-1]['fields']}
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
    monkeypatch.setattr(zb, 'extract_regular_zarr', extract)
    monkeypatch.setattr(nb, 'Dataset', Fake)


def test_zarr_request_is_global_latest_published_hour_and_cached(monkeypatch, tmp_path):
    requests = []
    fake_bridge(monkeypatch, '2026-11-10T23:00:00Z', requests)
    row = source_rows()['era5']
    valid = datetime(2026, 11, 9, 12, 30, tzinfo=timezone.utc)
    path, ident = obs.acquire_zarr(row, valid, tmp_path)
    assert requests[-1]['times'] == ['2026-11-09T12:00:00Z'] and ident.endswith('2026-11-09T12:00:00Z')
    assert requests[-1]['area'] == [-90., -180., 90., 180.]
    assert {f['source'] for f in requests[-1]['fields']} == {f['variable'] for f in row['field']}
    assert not any(f['pressure'] for f in requests[-1]['fields'])
    # The same request is served from the cache, whatever the targets.
    assert obs.acquire_zarr(row, valid, tmp_path)[0] == path and len(requests) == 1
    # Past the store's last published hour: that hour (decode_field judges its age).
    obs.acquire_zarr(row, datetime(2026, 11, 16, 12, tzinfo=timezone.utc), tmp_path)
    assert requests[-1]['times'] == ['2026-11-10T23:00:00Z']
    # Targets at the antimeridian ask for the same window (decode_field
    # closes the cyclic seam), so the request is served from the cache.
    assert obs.acquire_zarr(row, valid, tmp_path)[0] == path
    assert all(r['area'] == [-90., -180., 90., 180.] for r in requests)
    assert not list(tmp_path.glob('.*'))


def test_an_unwritten_declared_hour_steps_back_a_day_within_the_age_limit(monkeypatch, tmp_path):
    """ARCO declared ERA5T through 2026-11-10 while 2026-11-09 had no chunk."""
    requests = []
    fake_bridge(monkeypatch, '2026-11-10T00:00:00Z', requests,
                unwritten=('2026-11-09T12:00:00Z', '2026-11-08T12:00:00Z'))
    row = source_rows()['era5']
    path, ident = obs.acquire_zarr(row, datetime(2026, 11, 9, 12, tzinfo=timezone.utc), tmp_path)
    assert [r['times'][0] for r in requests[:3]] == ['2026-11-09T12:00:00Z', '2026-11-08T12:00:00Z', '2026-11-07T12:00:00Z']
    # Then the hours between the written and the last unwritten one are
    # bisected: here every one of them is written, so the latest is taken.
    assert ident.endswith('2026-11-08T11:00:00Z') and path.exists()
    requests.clear()
    fake_bridge(monkeypatch, '2026-11-10T00:00:00Z', requests,
                unwritten=tuple(f'2026-{m:02d}-{d:02d}T12:00:00Z' for m, d in
                                ((10, 30), (10, 31), *((11, d) for d in range(1, 10)))))
    with pytest.raises(ValueError, match='no hour within 240 h'):
        obs.acquire_zarr(row, datetime(2026, 11, 9, 12, tzinfo=timezone.utc), tmp_path / 'x')


def test_a_failed_refinement_probe_keeps_the_written_hour(monkeypatch, tmp_path):
    """Review (#587): a probe failing during the hourly bisection must not
    refuse a run that already found a written hour."""
    requests = []
    fake_bridge(monkeypatch, '2026-11-10T00:00:00Z', requests, unwritten=('2026-11-09T12:00:00Z',))
    import woof.zarr_bridge as zb
    extract = zb.extract_regular_zarr
    def flaky(request, **kw):
        if len(requests) >= 2:
            requests.append(request)
            raise OSError('store timed out')
        return extract(request, **kw)
    monkeypatch.setattr(zb, 'extract_regular_zarr', flaky)
    row = source_rows()['era5']
    path, ident = obs.acquire_zarr(row, datetime(2026, 11, 9, 12, tzinfo=timezone.utc), tmp_path)
    assert ident.endswith('2026-11-08T12:00:00Z') and path.exists() and len(requests) == 3


def test_forcing_ice_decides_frozen_cells():
    values = dict(water_temperature=np.array([280., 275., 274., 273.2]),
                  bottom_temperature=np.full(4, 277.), mixed_layer_depth=np.full(4, 3.),
                  temperature_shape_factor=np.full(4, .6),
                  ice_thickness=np.array([0., .3, 0., .2]), ice_temperature=np.array([273.16, 268., 273.16, 270.]))
    forcing_ice = np.array([0., 0., 1., .7])
    counts = obs.apply_ice_authority(values, forcing_ice)
    assert counts == dict(forcing_ice_cells=2, source_ice_kept_cells=1)
    np.testing.assert_array_equal(values['water_temperature'][:2], [280., 275.])
    assert np.isnan(values['water_temperature'][2:]).all() and np.isnan(values['bottom_temperature'][2:]).all()
    # Open forcing: the source never freezes it.  Icy forcing, open source: forcing kept.
    assert np.isnan(values['ice_thickness'][:3]).all() and np.isnan(values['ice_fraction'][:3]).all()
    assert values['ice_thickness'][3] == .2 and values['ice_fraction'][3] == .7
    assert values['ice_temperature'][3] == 270. and np.isnan(values['ice_temperature'][:3]).all()


def test_ice_authority_row_refuses_without_forcing_ice(tmp_path):
    with pytest.raises(ValueError, match='forcing ice concentration'):
        obs.load_state('era5', datetime(2026, 11, 9, 12), np.array([46.4]), np.array([6.5]), cache=tmp_path)


def test_chain_fills_only_cells_the_observed_source_missed(monkeypatch):
    calls = []
    def fake(source, valid_time, lat, lon, *, forcing_ice=None, **kw):
        calls.append((source, lat.copy()))
        if source == 'auto':
            water = np.where(lat > 41., 280., np.nan)
            return dict(water_temperature=water, ice_fraction=np.where(lat > 41., 0., np.nan)), dict(status='partial_forcing_fallback', errors=[])
        return dict(water_temperature=np.full(lat.size, 285.)), dict(status='bound', errors=[])
    monkeypatch.setattr(obs, 'load_state', fake)
    lat = np.array([42., 46., 40.])
    values, receipt = obs.load_chain(obs.CHAINS['auto+era5'], None, lat, np.zeros(3), forcing_ice=np.zeros(3))
    assert [c[0] for c in calls] == ['auto', 'era5']
    np.testing.assert_array_equal(calls[1][1], [40.])
    np.testing.assert_array_equal(values['water_temperature'], [280., 280., 285.])
    np.testing.assert_array_equal(values['ice_fraction'][:2], [0., 0.])
    assert np.isnan(values['ice_fraction'][2])
    assert receipt['status'] == 'bound' and receipt['unbound_cells'] == 0 and receipt['source'] == 'auto+era5'


def test_chain_value_is_a_valid_configuration():
    from woof.config import ConfigurationRefused, validate_run_config
    from test_ruc_mosaic_options import _config
    validate_run_config(_config(lake_state_source='auto+era5'))
    validate_run_config(_config(lake_state_source='era5', sf_lake_physics=1))
    with pytest.raises(ConfigurationRefused, match='lake_state_source'):
        validate_run_config(_config(lake_state_source='era5+auto'))


def test_the_latest_written_hour_not_a_day_older(monkeypatch, tmp_path):
    """Review 8: written through 11-08 05Z; a day stride alone took 11-07 12Z."""
    requests = []
    fake_bridge(monkeypatch, '2026-11-10T00:00:00Z', requests,
                unwritten=lambda hour: hour > '2026-11-08T05:00:00Z')
    row = source_rows()['era5']
    _, ident = obs.acquire_zarr(row, datetime(2026, 11, 9, 12, tzinfo=timezone.utc), tmp_path)
    assert ident.endswith('2026-11-08T05:00:00Z')
    assert len(requests) <= 3 + 5


def test_a_partly_written_hour_is_not_cached(monkeypatch, tmp_path):
    """Review 3: the first field written, a later one still fill."""
    requests = []
    fake_bridge(monkeypatch, '2026-11-10T00:00:00Z', requests,
                unwritten=('2026-11-09T12:00:00Z',), missing=('lake_ice_temperature',))
    row = source_rows()['era5']
    path, ident = obs.acquire_zarr(row, datetime(2026, 11, 9, 12, tzinfo=timezone.utc), tmp_path)
    assert not ident.endswith('2026-11-09T12:00:00Z') and path.exists()
    assert len(list(tmp_path.glob('*.nc'))) == len([r for r in requests if r['times'][0] != '2026-11-09T12:00:00Z'])


def test_the_cyclic_seam_is_closed_for_a_global_axis():
    """Review 2: ARCO's whole-globe axis from 180 W ends at 179.5 E, so a lake
    between 179.5 E and 179.75 E (or 0 E with the 0 E seam) kept the forcing."""
    lon = np.arange(-180.25, 179.75, .25)
    assert lon.size == 1440
    data = np.tile(np.cos(np.radians(lon)), (2, 1))
    closed_lon, closed = obs.cyclic_longitude(lon, data)
    assert closed_lon[-1] == 179.75 and closed.shape == (2, 1441)
    np.testing.assert_array_equal(closed[:, -1], data[:, 0])
    targets = np.array([179.6, -179.9, 6.6, -0.1])
    got = obs.sample_field([1., 0.], closed_lon, closed, np.full(4, .5), targets)
    assert np.isfinite(got).all()
    # The interior is sampled exactly as before.
    np.testing.assert_array_equal(got[2:], obs.sample_field([1., 0.], lon, data, np.full(2, .5), targets[2:]))
    assert np.isnan(obs.sample_field([1., 0.], lon, data, [.5], [179.6]))[0]
    regional = np.arange(-92., -75., .25)
    assert obs.cyclic_longitude(regional, np.ones((2, regional.size)))[0].size == regional.size


def test_chain_never_overwrites_an_earlier_finite_value(monkeypatch):
    """Review 1: GLSEA ice on a cell whose water temperature is missing."""
    def fake(source, valid_time, lat, lon, *, forcing_ice=None, **kw):
        if source == 'auto':
            return (dict(water_temperature=np.array([280., np.nan]), ice_fraction=np.array([0., .8])),
                    dict(status='partial_forcing_fallback', errors=[]))
        assert lat.size == 1
        return (dict(water_temperature=np.array([276.]), ice_fraction=np.array([np.nan])),
                dict(status='bound', errors=[]))
    monkeypatch.setattr(obs, 'load_state', fake)
    values, receipt = obs.load_chain(obs.CHAINS['auto+era5'], None, np.array([42., 46.]), np.zeros(2),
                                     forcing_ice=np.array([0., .8]))
    np.testing.assert_array_equal(values['water_temperature'], [280., 276.])
    np.testing.assert_array_equal(values['ice_fraction'], [0., .8])


def test_chain_auto_step_stays_tolerant_of_out_of_range_values(monkeypatch):
    """Review 4: auto drops its out-of-range values; the model step fills them."""
    def fake(source, valid_time, lat, lon, *, forcing_ice=None, **kw):
        if source == 'auto':
            return (dict(water_temperature=np.array([280., 340.])), dict(status='bound', errors=[]))
        return dict(water_temperature=np.full(lat.size, 277.)), dict(status='bound', errors=[])
    monkeypatch.setattr(obs, 'load_state', fake)
    values, receipt = obs.load_chain(obs.CHAINS['auto+era5'], None, np.array([42., 46.]), np.zeros(2),
                                     forcing_ice=np.zeros(2))
    np.testing.assert_array_equal(values['water_temperature'], [280., 277.])
    # A value the next step filled is a note, not a retained-forcing error.
    assert receipt['status'] == 'bound' and not receipt['errors']
    assert any('outside [270.0, 330.0]' in e for e in receipt['dropped'])
    assert receipt['chain'][0]['status'] == 'partial_forcing_fallback'


def test_chain_model_step_failure_is_a_named_refusal(monkeypatch, tmp_path):
    """Review 5: the store's coverage query timing out refuses by name."""
    import subprocess
    import woof.zarr_bridge as zb
    def timeout(store):
        raise subprocess.TimeoutExpired(['rw_zarr', 'coverage', store], 600)
    monkeypatch.setattr(zb, 'zarr_coverage', timeout)
    with pytest.raises(ValueError, match=r'lake source era5: .* unavailable \(TimeoutExpired'):
        obs.load_state('era5', datetime(2026, 11, 9, 12), np.array([46.4]), np.array([6.5]),
                       cache=tmp_path, forcing_ice=np.zeros(1))
    real = obs.load_state
    def auto_misses(source, *a, **kw):
        if source == 'auto':
            return {}, dict(schema='gpuwm.lake-init.v1', status='outside_source_coverage')
        return real(source, *a, **kw)
    monkeypatch.setattr(obs, 'load_state', auto_misses)
    with pytest.raises(ValueError, match=r'lake_state_source=auto\+era5: the era5 step failed for 1 of 1'):
        obs.load_chain(obs.CHAINS['auto+era5'], datetime(2026, 11, 9, 12), np.array([46.4]),
                       np.array([6.5]), cache=tmp_path, forcing_ice=np.zeros(1))
