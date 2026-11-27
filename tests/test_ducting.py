"""woof ducting: modified refractivity, duct diagnosis, regions, the CLI."""
import csv
import json

import numpy as np
import pytest

from woof.ducting import (diagnose_ducts, find_frames, modified_refractivity,
                          parse_frame_name, region_mask)


def _column(values, heights):
    return np.array(values, float)[:, None], np.array(heights, float)[:, None]


Z = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0]


def test_standard_atmosphere_has_no_duct():
    m, z = _column([300, 301, 302, 303, 304, 305, 306, 307], Z)
    out = diagnose_ducts(m, z)
    assert out['duct_type'][0] == 0
    assert out['n_trapping_layers'][0] == 0
    assert np.isnan(out['duct_top'][0]) and np.isnan(out['trap_base_lowest'][0])


def test_surface_based_duct():
    m, z = _column([350, 345, 340, 345, 360, 361, 362, 363], Z)
    out = diagnose_ducts(m, z)
    assert out['duct_type'][0] == 1
    assert out['duct_base'][0] == 0.0
    assert out['duct_top'][0] == 200.0
    assert out['duct_thickness'][0] == 200.0
    assert out['duct_strength'][0] == pytest.approx(10.0)
    assert out['n_trapping_layers'][0] == 1
    assert out['trap_base_lowest'][0] == 0.0


def test_elevated_duct_base_is_interpolated():
    m, z = _column([300, 310, 320, 308, 303, 315, 316, 317], Z)
    out = diagnose_ducts(m, z)
    assert out['duct_type'][0] == 2
    # Going down from the top (M = 303 at 400 m) M first reaches 303 between
    # 0 m (300) and 100 m (310): 30 m.
    assert out['duct_base'][0] == pytest.approx(30.0)
    assert out['duct_top'][0] == 400.0
    assert out['duct_thickness'][0] == pytest.approx(370.0)
    assert out['duct_strength'][0] == pytest.approx(17.0)
    assert out['trap_base_lowest'][0] == 200.0


def test_strongest_of_several_trapping_layers():
    m, z = _column([300, 310, 305, 306, 330, 320, 315, 330], Z)
    out = diagnose_ducts(m, z)
    assert out['n_trapping_layers'][0] == 2
    assert out['trap_base_lowest'][0] == 100.0
    # Layer one: 310 -> 305 (M_top 305, base 50 m, strength 5).
    # Layer two: 330 -> 315 (M_top 315, base 0.. interpolated, strength 15).
    assert out['duct_top'][0] == 600.0
    assert out['duct_strength'][0] == pytest.approx(15.0)
    assert out['duct_type'][0] == 2


def test_max_height_limits_search():
    m, z = _column([300, 301, 302, 303, 304, 330, 320, 310], Z)
    assert diagnose_ducts(m, z, max_height=3000.0)['duct_type'][0] == 2
    assert diagnose_ducts(m, z, max_height=450.0)['duct_type'][0] == 0


def test_modified_refractivity_hand_value():
    shape = (1, 1, 1)
    # theta = 300 K at 1e5 Pa gives T = 300 K; mid-level height 100 m.
    m, z = modified_refractivity(
        np.zeros(shape), np.zeros(shape), np.full(shape, 1.0e5), np.full(shape, 0.01),
        np.array([0.0, 2 * 9.81 * 100.0]).reshape(2, 1, 1), np.zeros((2, 1, 1)), np.zeros((1, 1)))
    e = 0.01 * 1.0e5 / (0.622 + 0.01) / 100.0
    expected = 77.6 / 300.0 * (1000.0 + 4810.0 * e / 300.0) + 0.157 * 100.0
    assert z[0, 0, 0] == pytest.approx(100.0)
    assert m[0, 0, 0] == pytest.approx(expected)
    assert 335.0 < expected < 345.0


def test_negative_humidity_is_clipped():
    shape = (1, 1, 1)
    args = (np.zeros(shape), np.zeros(shape), np.full(shape, 1.0e5))
    ph = np.zeros((2, 1, 1))
    dry, _ = modified_refractivity(*args, np.zeros(shape), ph, ph, np.zeros((1, 1)))
    neg, _ = modified_refractivity(*args, np.full(shape, -1e-3), ph, ph, np.zeros((1, 1)))
    assert dry[0, 0, 0] == neg[0, 0, 0]


def test_vectorised_columns_match_single_columns():
    cols = [[300, 301, 302, 303, 304, 305, 306, 307],
            [350, 345, 340, 345, 360, 361, 362, 363],
            [300, 310, 320, 308, 303, 315, 316, 317],
            [300, 310, 305, 306, 330, 320, 315, 330]]
    m = np.array(cols, float).T.reshape(8, 2, 2)
    z = np.broadcast_to(np.array(Z)[:, None, None], m.shape)
    out = diagnose_ducts(m, z)
    assert out['duct_type'].shape == (2, 2)
    assert out['duct_type'].ravel().tolist() == [0, 1, 2, 2]
    for i, col in enumerate(cols):
        one = diagnose_ducts(*_column(col, Z))
        for key in one:
            np.testing.assert_array_equal(out[key].ravel()[i], one[key][0])


def test_region_mask_polygon_multipolygon_and_sea():
    lon, lat = np.meshgrid(np.arange(5.0), np.arange(4.0))
    square = [[0.5, 0.5], [2.5, 0.5], [2.5, 2.5], [0.5, 2.5], [0.5, 0.5]]
    inside = region_mask(lon, lat, {'type': 'Polygon', 'coordinates': [square]})
    assert inside.sum() == 4
    assert inside[1:3, 1:3].all()
    far = [[3.5, -0.5], [4.5, -0.5], [4.5, 0.5], [3.5, 0.5], [3.5, -0.5]]
    multi = {'type': 'Feature', 'geometry': {'type': 'MultiPolygon', 'coordinates': [[square], [far]]}}
    assert region_mask(lon, lat, multi).sum() == 5
    land = np.zeros(lon.shape)
    land[1, 1] = 1.0
    sea = region_mask(lon, lat, {'type': 'Polygon', 'coordinates': [square]}, land, True)
    assert sea.sum() == 3 and not sea[1, 1]
    assert region_mask(lon, lat, None, land, True).sum() == lon.size - 1
    with pytest.raises(ValueError):
        region_mask(lon, lat, None, None, True)
    with pytest.raises(ValueError):
        region_mask(lon, lat, {'type': 'Point', 'coordinates': [0, 0]})


def test_frame_name_parsing_both_forms(tmp_path):
    assert parse_frame_name('wrfout_d01_2000-01-02_03_04_05') == (1, '2000-01-02T03:04:05')
    assert parse_frame_name('/x/wrfout_d02_2000-01-02_03:04:05') == (2, '2000-01-02T03:04:05')
    assert parse_frame_name('wrfinput_d01') is None
    for name in ('wrfout_d01_2000-01-01_01_00_00', 'wrfout_d01_2000-01-01_00:00:00',
                 'wrfout_d01_2000-01-01_01_00_00.json'):
        deep = tmp_path / 'run-1' / 'chain' / 'run' / 'wrfout'
        deep.mkdir(parents=True, exist_ok=True)
        (deep / name).write_bytes(b'')
    found = find_frames([tmp_path])
    assert [p.name for p in found] == ['wrfout_d01_2000-01-01_00:00:00', 'wrfout_d01_2000-01-01_01_00_00']


def _write_wrfout(path, theta_pert, nz=6, ny=3, nx=4):
    from netCDF4 import Dataset
    z_w = np.arange(nz + 1) * 100.0
    with Dataset(path, 'w') as ds:
        ds.createDimension('Time', 1)
        ds.createDimension('bottom_top', nz)
        ds.createDimension('bottom_top_stag', nz + 1)
        ds.createDimension('south_north', ny)
        ds.createDimension('west_east', nx)

        def put(name, dims, value):
            ds.createVariable(name, 'f8', ('Time',) + dims)[0] = value

        put('T', ('bottom_top', 'south_north', 'west_east'), theta_pert)
        put('P', ('bottom_top', 'south_north', 'west_east'), 0.0)
        pb = 1.0e5 * np.exp(-np.arange(nz) * 100.0 / 8000.0)
        put('PB', ('bottom_top', 'south_north', 'west_east'), pb[:, None, None])
        put('QVAPOR', ('bottom_top', 'south_north', 'west_east'), 0.004)
        put('PH', ('bottom_top_stag', 'south_north', 'west_east'), 0.0)
        put('PHB', ('bottom_top_stag', 'south_north', 'west_east'), (z_w * 9.81)[:, None, None])
        put('HGT', ('south_north', 'west_east'), 0.0)
        lon, lat = np.meshgrid(np.arange(nx) + 10.0, np.arange(ny) + 20.0)
        put('XLAT', ('south_north', 'west_east'), lat)
        put('XLONG', ('south_north', 'west_east'), lon)
        land = np.zeros((ny, nx))
        land[:, 0] = 1.0
        put('LANDMASK', ('south_north', 'west_east'), land)


def test_cli_end_to_end(tmp_path, capsys):
    from netCDF4 import Dataset
    from woof.cli import main
    run = tmp_path / 'run-1' / 'chain' / 'run' / 'wrfout'
    run.mkdir(parents=True)
    # Strong warm, dry layer aloft in every column of the second frame only.
    base = np.zeros((6, 3, 4))
    warm = base.copy()
    warm[3:] += 25.0
    _write_wrfout(run / 'wrfout_d01_2000-01-01_00_00_00', base)
    _write_wrfout(run / 'wrfout_d01_2000-01-01_01:00:00', warm)
    (run / 'wrfout_d01_2000-01-01_00_00_00.json').write_text('{}')
    region = tmp_path / 'region.geojson'
    region.write_text(json.dumps({'type': 'Polygon', 'coordinates': [
        [[10.5, 19.5], [12.5, 19.5], [12.5, 22.5], [10.5, 22.5], [10.5, 19.5]]]}))
    out = tmp_path / 'out'
    code = main(['ducting', str(tmp_path), '--out', str(out), '--region', str(region),
                 '--sea-only', '--max-height', '550'])
    assert code in (0, None)
    files = sorted(p.name for p in out.glob('ducts_*.nc'))
    assert files == ['ducts_d01_20000101T000000.nc', 'ducts_d01_20000101T010000.nc']
    with Dataset(out / files[1]) as ds:
        assert ds['duct_type'].shape == (3, 4)
        assert ds['duct_type'].units == '1'
        assert 'XLAT' in ds.variables and 'XLONG' in ds.variables
        assert ds['region_mask'][:].sum() == 6    # two columns east of the land column x 3 rows
        assert ds.valid_time == '2000-01-01T01:00:00'
    rows = list(csv.DictReader((out / 'ducting_summary.csv').open()))
    assert [r['time'] for r in rows] == ['2000-01-01T00:00:00', '2000-01-01T01:00:00']
    assert [r['n_columns'] for r in rows] == ['6', '6']
    assert float(rows[0]['any_duct_pct']) == 0.0
    assert float(rows[1]['any_duct_pct']) == 100.0
    assert float(rows[1]['median_top_m']) > 0.0


def test_cli_refuses_when_no_frames(tmp_path, capsys):
    from woof.cli import main
    assert main(['ducting', str(tmp_path), '--out', str(tmp_path / 'o')]) != 0
    assert 'no wrfout frames' in capsys.readouterr().err
