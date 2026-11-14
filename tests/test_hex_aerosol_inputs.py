"""aer_opt = 2 on hex: the auxinput15 records remapped to the mesh cells (host)."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import aerosol_inputs as ai
from woof.hex.errors import ConfigurationRefusal
from woof.hex.hydro_remap import grid_cells

START = datetime(2026, 10, 3, 18, 0)


def _cfg(tmp_path, interval=3600):
    return SimpleNamespace(
        aer_aod550_opt=2, aer_angexp_opt=1, aer_ssa_opt=1, aer_asy_opt=1,
        aer_auxinput15=str(tmp_path / "aux_d<domain>_<date>.nc"),
        aer_auxinput15_interval=interval, grid_id=1)


def _write(path, aod, *, corners=True, lat0=30.0, lon0=-100.0, n=6):
    import netCDF4

    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("south_north", n)
        ds.createDimension("west_east", n)
        ds.createDimension("south_north_stag", n + 1)
        ds.createDimension("west_east_stag", n + 1)
        ds.createVariable("AOD5502D", "f4", ("Time", "south_north", "west_east"))[:] = aod
        lat_c, lon_c = np.meshgrid(lat0 + np.arange(n + 1.0), lon0 + np.arange(n + 1.0), indexing="ij")
        if corners:
            ds.createVariable("XLAT_C", "f8", ("south_north_stag", "west_east_stag"))[:] = lat_c
            ds.createVariable("XLONG_C", "f8", ("south_north_stag", "west_east_stag"))[:] = lon_c
        else:
            ds.createVariable("XLAT", "f8", ("Time", "south_north", "west_east"))[:] = lat_c[:-1, :-1] + 0.5
            ds.createVariable("XLONG", "f8", ("Time", "south_north", "west_east"))[:] = lon_c[:-1, :-1] + 0.5


def _mesh_cells():
    # 3 x 2 cells of 1 x 2 degrees inside the grid (lat 31-34, lon -99..-95)
    lat_c, lon_c = np.meshgrid(31.0 + np.arange(4.0), -99.0 + 2.0 * np.arange(3.0), indexing="ij")
    return grid_cells(lat_c, lon_c)


def test_the_records_the_run_reads_follow_the_file_interval(tmp_path):
    cfg = _cfg(tmp_path)
    assert ai.run_offsets(cfg, 3600.0) == [0.0]
    assert ai.run_offsets(cfg, 3601.0) == [0.0, 3600.0]
    assert ai.run_offsets(cfg, 3 * 3600.0) == [0.0, 3600.0, 7200.0]
    with pytest.raises(ConfigurationRefusal, match="positive aer_auxinput15_interval"):
        ai.run_offsets(_cfg(tmp_path, 0), 3600.0)


@pytest.mark.parametrize("corners", [True, False])
def test_a_record_is_remapped_conservatively_onto_the_cells(tmp_path, corners):
    pytest.importorskip("woof.netcdf_bridge")
    rng = np.random.default_rng(1)
    for hour in (0, 1):
        stamp = (START + timedelta(hours=hour)).strftime("%Y-%m-%d_%H:%M:%S")
        _write(str(tmp_path / f"aux_d01_{stamp}.nc"),
               rng.uniform(0.1, 0.5, (1, 6, 6)) * (1 + hour), corners=corners)
    cfg = _cfg(tmp_path)
    try:
        frames, offsets = ai.stage_user_aerosol(
            cfg, START, mesh=None, run_seconds=7200.0, cells=_mesh_cells())
    except ModuleNotFoundError:
        pytest.skip("the Rust NetCDF bridge is not built")
    assert frames.shape == (2, 1, 6) and frames.dtype == np.float32
    assert offsets.tolist() == [0.0, 3600.0]
    stamp = START.strftime("%Y-%m-%d_%H:%M:%S")
    import netCDF4

    grid = np.asarray(netCDF4.Dataset(str(tmp_path / f"aux_d01_{stamp}.nc"))["AOD5502D"][0], float)
    # Mesh cell (r, c) holds grid row r+1 and columns 2c+1..2c+2 (aligned on meridians).
    want = np.array([grid[r + 1, 1 + 2 * c: 3 + 2 * c].mean()
                     for r in range(3) for c in range(2)])
    got = frames[0, 0]
    # A 2-degree great-circle edge sags off the two 1-degree parallels it spans,
    # so the cell takes a sliver of its neighbours' rows: agreement to 1e-3.
    np.testing.assert_allclose(got, want, rtol=1e-3)
    assert np.all(np.isfinite(frames))


def test_a_mesh_larger_than_the_file_grid_is_refused(tmp_path):
    pytest.importorskip("woof.netcdf_bridge")
    _write(str(tmp_path / f"aux_d01_{START.strftime('%Y-%m-%d_%H:%M:%S')}.nc"),
           np.full((1, 6, 6), 0.2))
    lat_c, lon_c = np.meshgrid(28.0 + np.arange(4.0), -99.0 + 2.0 * np.arange(3.0), indexing="ij")
    try:
        with pytest.raises(ConfigurationRefusal, match="does not cover the mesh"):
            ai.stage_user_aerosol(_cfg(tmp_path), START, mesh=None, run_seconds=3600.0,
                                  cells=grid_cells(lat_c, lon_c))
    except ModuleNotFoundError:
        pytest.skip("the Rust NetCDF bridge is not built")


def test_a_constant_record_remaps_to_the_same_constant(tmp_path):
    pytest.importorskip("woof.netcdf_bridge")
    _write(str(tmp_path / f"aux_d01_{START.strftime('%Y-%m-%d_%H:%M:%S')}.nc"),
           np.full((1, 6, 6), 0.37))
    try:
        frames, _ = ai.stage_user_aerosol(_cfg(tmp_path), START, mesh=None,
                                          run_seconds=3600.0, cells=_mesh_cells())
    except ModuleNotFoundError:
        pytest.skip("the Rust NetCDF bridge is not built")
    np.testing.assert_allclose(frames, np.float32(0.37), rtol=1e-6)
