"""The fuel statics' GPU body against its CPU body: the same words.

Categories must be equal and ZSF, DZDXF and DZDYF must be equal as raw
float32 words (max ULP 0), on the land-use route and the LANDFIRE route.
Both bodies are held to WRF itself by tests/test_fire_mesh_wrf471_parity.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_gpu

MODIS = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17, "ISLAKE": 21,
         "ISICE": 15, "ISURBAN": 13, "NUM_LAND_CAT": 21}


def _domain(nx=23, ny=17, dx=750.0):
    from woof.static.lambert import LambertGrid
    grid = LambertGrid(ref_lat=44.1, ref_lon=-121.6, truelat1=33.0,
                       truelat2=45.0, stand_lon=-121.0, dx=dx, dy=dx,
                       e_we=nx + 1, e_sn=ny + 1)
    rng = np.random.default_rng(20261022)
    j, i = np.mgrid[0:ny, 0:nx].astype(np.float64)
    hgt = (900.0 + 400.0 * np.sin(0.4 * i) * np.cos(0.3 * j)
           + rng.normal(0.0, 30.0, (ny, nx)))
    hgt[:, 0] = 0.0  # a shore on the west edge
    lu = rng.choice([1, 2, 5, 7, 10, 12, 13, 16, 17], size=(ny, nx))
    lu[:, :2] = 17
    return grid, {"HGT_M": hgt, "LU_INDEX": lu.astype(np.float64)}


def _assert_same(cpu: dict, gpu: dict) -> None:
    from woof.core.fp32_ulp import max_ulp
    import cupy as cp
    for name, value in gpu.items():
        host = value.get() if isinstance(value, cp.ndarray) else value
        assert host.dtype == cpu[name].dtype, name
        if host.dtype == np.float32:
            assert max_ulp(host, cpu[name]) == 0, name
            np.testing.assert_array_equal(host.view(np.uint32),
                                          cpu[name].view(np.uint32))
        else:
            np.testing.assert_array_equal(host, cpu[name])


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("sr", [(1, 1), (3, 3), (4, 2), (10, 10)])
def test_landuse_route_gpu_is_the_cpu_words(sr):
    from woof.static.fuel import FuelStaticsRequest, build_fuel_fields
    grid, fields = _domain()
    request = FuelStaticsRequest(sr_x=sr[0], sr_y=sr[1])
    reports = {}, {}
    cpu = build_fuel_fields(grid, fields, request, landuse_attrs=MODIS,
                            device="cpu", coverage_report=reports[0])
    gpu = build_fuel_fields(grid, fields, request, landuse_attrs=MODIS,
                            device="gpu", coverage_report=reports[1])
    assert all(isinstance(value, np.ndarray) for value in gpu.values())
    _assert_same(cpu, gpu)
    assert reports[0]["fuel"]["device"] == "cpu"
    assert reports[1]["fuel"]["device"] == "gpu"
    if sr[0] > 1:  # the edge band is WRF's continuation past the shore
        assert float(cpu["ZSF"].min()) < 0.0


@pytest.mark.gpu
@requires_gpu
def test_landfire_route_gpu_is_the_cpu_words(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("pyproj")
    from rasterio.transform import from_origin
    from woof.static.fuel import (FuelStaticsRequest, build_fuel_fields,
                                  fire_mesh_axis)
    grid, fields = _domain()
    lat, lon = grid.ij_to_latlon(*np.meshgrid(fire_mesh_axis(23, 4),
                                              fire_mesh_axis(17, 4)))
    res = 0.00031
    west, north = lon.min() - 0.01, lat.max() + 0.01
    width = int((lon.max() - 0.03 - west) / res)  # leaves an east strip
    height = int((north - lat.min() + 0.01) / res)
    codes = np.array([101, 122, 145, 165, 183, 188, 98, 91, 202, -9999],
                     dtype=np.int16)
    data = codes[(np.arange(width * height) // 37) % codes.size]
    path = tmp_path / "fbfm40.tif"
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height,
                       count=1, dtype="int16", crs="EPSG:4326", nodata=-9999,
                       transform=from_origin(west, north, res, res)) as out:
        out.write(data.reshape(height, width), 1)
    request = FuelStaticsRequest(sr_x=4, sr_y=4, landfire=path)
    reports = {}, {}
    cpu = build_fuel_fields(grid, fields, request, landuse_attrs=MODIS,
                            device="cpu", coverage_report=reports[0])
    gpu = build_fuel_fields(grid, fields, request, landuse_attrs=MODIS,
                            device="gpu", coverage_report=reports[1])
    _assert_same(cpu, gpu)
    filled = reports[0]["fuel"]["landfire"]["landuse_filled_cells"]
    assert filled > 0
    assert reports[1]["fuel"]["landfire"]["landuse_filled_cells"] == filled


@pytest.mark.gpu
@requires_gpu
def test_fire_mesh_can_stay_on_the_device():
    import cupy as cp
    from woof.static.fuel import (FIRE_MESH_FIELDS, FuelStaticsRequest,
                                  build_fuel_fields, to_device)
    grid, fields = _domain()
    request = FuelStaticsRequest(sr_x=3, sr_y=3)
    resident = build_fuel_fields(grid, fields, request, landuse_attrs=MODIS,
                                 device="gpu", resident=True)
    for name in FIRE_MESH_FIELDS:
        assert isinstance(resident[name], cp.ndarray), name
    assert isinstance(resident["NFUEL_CAT"], np.ndarray)
    host = build_fuel_fields(grid, fields, request, landuse_attrs=MODIS,
                             device="cpu")
    moved = to_device({**host, "HGT_M": fields["HGT_M"]})
    assert isinstance(moved["HGT_M"], np.ndarray)
    assert isinstance(moved["NFUEL_CAT"], np.ndarray)  # resident's layout
    _assert_same(host, {name: moved[name] for name in host})
    _assert_same(host, resident)


@pytest.mark.gpu
@requires_gpu
def test_a_failing_device_falls_back_on_auto_and_refuses_on_gpu(
        monkeypatch, capsys):
    import cupy as cp
    from woof.static import fire_mesh_gpu, fuel
    from woof.static.fuel import FuelStaticsRequest, build_fuel_fields
    monkeypatch.setattr(fuel, "_GPU_FAILED", None)
    grid, fields = _domain()
    request = FuelStaticsRequest(sr_x=2, sr_y=2)
    cpu = build_fuel_fields(grid, fields, request, landuse_attrs=MODIS,
                            device="cpu")

    def broken(*args, **kwargs):
        raise cp.cuda.runtime.CUDARuntimeError(2)  # out of memory

    monkeypatch.setattr(fire_mesh_gpu, "refine_zsf", broken)
    report = {}
    auto = build_fuel_fields(grid, fields, request, landuse_attrs=MODIS,
                             device="auto", coverage_report=report)
    assert "WORKAROUND" in capsys.readouterr().err
    assert report["fuel"]["device"] == "cpu"
    _assert_same(cpu, auto)
    # Remembered: auto does not retry the failed device, nor say so again.
    calls = []
    monkeypatch.setattr(fire_mesh_gpu, "refine_zsf",
                        lambda *a, **k: calls.append(a))
    again = build_fuel_fields(grid, fields, request, landuse_attrs=MODIS,
                              device="auto")
    assert not calls and "WORKAROUND" not in capsys.readouterr().err
    _assert_same(cpu, again)
    monkeypatch.setattr(fire_mesh_gpu, "refine_zsf", broken)
    with pytest.raises(cp.cuda.runtime.CUDARuntimeError):
        build_fuel_fields(grid, fields, request, landuse_attrs=MODIS,
                          device="gpu")


@pytest.mark.gpu
@requires_gpu
def test_a_one_node_device_mesh_is_refused():
    import cupy as cp
    from woof.static.fuel import fire_mesh_slope
    for shape in ((1, 4), (4, 1)):
        with pytest.raises(ValueError, match="at least 2 x 2"):
            fire_mesh_slope(cp.ones(shape, cp.float32), 10.0, 10.0)



@pytest.mark.gpu
@requires_gpu
def test_a_host_result_build_leaves_the_forecasts_pool_alone():
    """A nest spawned mid-forecast builds its fuel in a private pool: the
    default pool (the forecast's) neither grows nor is emptied."""
    import cupy as cp
    from woof.static.fuel import FuelStaticsRequest, build_fuel_fields
    grid, fields = _domain()
    pool = cp.get_default_memory_pool()
    held = cp.zeros(1 << 20, dtype=cp.float32)  # the forecast's own block
    del held  # cached in the default pool, as a forecast's would be
    before = (pool.used_bytes(), pool.total_bytes())
    build_fuel_fields(grid, fields, FuelStaticsRequest(sr_x=6, sr_y=6),
                      landuse_attrs=MODIS, device="gpu")
    assert (pool.used_bytes(), pool.total_bytes()) == before
