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


def fake_bridge(monkeypatch, stop, requests, unwritten=()):
    """rw_zarr stand-in: an unwritten hour comes back as the fill value."""
    import woof.netcdf_bridge as nb
    import woof.zarr_bridge as zb
    monkeypatch.setattr(zb, 'zarr_coverage', lambda store: dict(accepted_stop=stop))
    def extract(request, *, request_path, output, progress):
        requests.append(request)
        Path(output).write_text(request['times'][0])
        return {}
    class Fake:
        def __init__(self, path):
            fill = np.nan if Path(path).read_text() in unwritten else 280.
            self.variables = {f['source']: np.full((1, 2, 2), fill) for f in requests[-1]['fields']}
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
    path, ident = obs.acquire_zarr(row, valid, np.array([6.6, 9.4]), tmp_path)
    assert requests[-1]['times'] == ['2026-11-09T12:00:00Z'] and ident.endswith('2026-11-09T12:00:00Z')
    assert requests[-1]['area'] == [-90., -180., 90., 180.]
    assert {f['source'] for f in requests[-1]['fields']} == {f['variable'] for f in row['field']}
    assert not any(f['pressure'] for f in requests[-1]['fields'])
    # The same request is served from the cache, whatever the targets.
    assert obs.acquire_zarr(row, valid, np.array([7.0]), tmp_path)[0] == path and len(requests) == 1
    # Past the store's last published hour: that hour (decode_field judges its age).
    obs.acquire_zarr(row, datetime(2026, 11, 16, 12, tzinfo=timezone.utc), np.array([7.]), tmp_path)
    assert requests[-1]['times'] == ['2026-11-10T23:00:00Z']
    # Targets at the antimeridian move the cyclic seam to 0 E.
    obs.acquire_zarr(row, valid, np.array([179.6]), tmp_path)
    assert requests[-1]['area'] == [-90., 0., 90., 360.]
    assert not list(tmp_path.glob('.*'))


def test_an_unwritten_declared_hour_steps_back_a_day_within_the_age_limit(monkeypatch, tmp_path):
    """ARCO declared ERA5T through 2026-11-10 while 2026-11-09 had no chunk."""
    requests = []
    fake_bridge(monkeypatch, '2026-11-10T00:00:00Z', requests,
                unwritten=('2026-11-09T12:00:00Z', '2026-11-08T12:00:00Z'))
    row = source_rows()['era5']
    path, ident = obs.acquire_zarr(row, datetime(2026, 11, 9, 12, tzinfo=timezone.utc), np.array([7.]), tmp_path)
    assert [r['times'][0] for r in requests] == ['2026-11-09T12:00:00Z', '2026-11-08T12:00:00Z', '2026-11-07T12:00:00Z']
    assert ident.endswith('2026-11-07T12:00:00Z') and path.exists() and len(list(tmp_path.glob('*.nc'))) == 1
    requests.clear()
    fake_bridge(monkeypatch, '2026-11-10T00:00:00Z', requests,
                unwritten=tuple(f'2026-{m:02d}-{d:02d}T12:00:00Z' for m, d in
                                ((10, 30), (10, 31), *((11, d) for d in range(1, 10)))))
    with pytest.raises(ValueError, match='no hour within 240 h'):
        obs.acquire_zarr(row, datetime(2026, 11, 9, 12, tzinfo=timezone.utc), np.array([7.]), tmp_path / 'x')


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
