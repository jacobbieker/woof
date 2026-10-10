"""MPAS history sampling at sites: interpolation, mapping and ``inside``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from woof.energy.sample import SampleResult, SampleUnavailable
from woof.energy.sample_hex import (
    available_variables,
    horizontal_weights,
    sample_mpas,
)

LAT0, LON0 = 51.6, -3.9
STEP = 0.05                       # degrees between cell centres
INTERFACES_AGL = np.array([0.0, 20.0, 60.0, 120.0])   # midpoints 10, 40, 90


def _grid():
    """A 3 x 3 patch of cells and the 8 triangles between them."""

    lat = np.repeat(LAT0 + STEP * np.arange(3), 3)
    lon = np.tile(LON0 + STEP * np.arange(3), 3)
    triangles = []
    for j in range(2):
        for i in range(2):
            a, b = 3 * j + i, 3 * j + i + 1
            c, d = 3 * (j + 1) + i, 3 * (j + 1) + i + 1
            triangles += [(a, b, d), (a, d, c)]
    cov = np.array(triangles) + 1          # one-based
    cov = np.vstack([cov, [[1, 2, 0]]])    # a vertex on the open edge
    return lat, lon, cov


def _xyz(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)


def _linear(lat_deg, lon_deg, scale=1.0):
    """A field linear in unit-sphere xyz: barycentric weights reproduce it."""

    x, y, z = _xyz(lat_deg, lon_deg)
    return scale * (3000.0 * x - 2000.0 * y + 500.0 * z)


def _terrain(lat_deg, lon_deg):
    return 100.0 + 0.0 * lat_deg


def _write_mesh(path: Path, *, bdy=None, with_cov=True, with_zgrid=True,
                n_extra_cells=0):
    lat, lon, cov = _grid()
    if n_extra_cells:
        lat = np.concatenate([lat, np.full(n_extra_cells, LAT0 - 1.0)])
        lon = np.concatenate([lon, np.full(n_extra_cells, LON0 - 1.0)])
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("nCells", len(lat))
        ds.createDimension("nVertices", len(cov))
        ds.createDimension("vertexDegree", 3)
        ds.createDimension("nVertLevelsP1", len(INTERFACES_AGL))
        ds.createVariable("latCell", "f8", ("nCells",))[:] = np.radians(lat)
        ds.createVariable("lonCell", "f8", ("nCells",))[:] = np.radians(lon)
        if with_cov:
            ds.createVariable("cellsOnVertex", "i4",
                              ("nVertices", "vertexDegree"))[:] = cov
        if with_zgrid:
            terrain = _terrain(lat, lon)
            ds.createVariable("zgrid", "f8", ("nCells", "nVertLevelsP1"))[:] = \
                terrain[:, None] + INTERFACES_AGL[None, :]
        if bdy is not None:
            ds.createVariable("bdyMaskCell", "i4", ("nCells",))[:] = bdy
    return lat, lon


def _write_history(path: Path, *, stamp="2026-10-10_06:00:00",
                   dialect="native", offset=0.0, with_xtime=True,
                   n_extra_cells=0):
    lat, lon, _ = _grid()
    if n_extra_cells:
        lat = np.concatenate([lat, np.full(n_extra_cells, LAT0 - 1.0)])
        lon = np.concatenate([lon, np.full(n_extra_cells, LON0 - 1.0)])
    n = len(lat)
    levels = len(INTERFACES_AGL) - 1
    base = _linear(lat, lon)[:, None]
    level = np.arange(levels)[None, :]
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("nCells", n)
        ds.createDimension("nVertLevels", levels)
        ds.createDimension("nVertLevelsP1", levels + 1)
        ds.createDimension("StrLen", 64)
        if with_xtime:
            xtime = ds.createVariable("xtime", "S1", ("Time", "StrLen"))
            xtime.set_auto_chartostring(False)
            chars = np.zeros(64, dtype="S1")
            chars[:len(stamp)] = list(stamp)
            xtime[0, :] = chars

        def put(name, values, dims):
            ds.createVariable(name, "f4", ("Time",) + dims)[0] = values

        u_name = "uReconstructZonal" if dialect == "native" else "u_zonal"
        v_name = "uReconstructMeridional" if dialect == "native" \
            else "v_meridional"
        put(u_name, base + 10.0 * level + offset, ("nCells", "nVertLevels"))
        put(v_name, -base, ("nCells", "nVertLevels"))
        put("theta", 290.0 + level + 0.0 * base, ("nCells", "nVertLevels"))
        put("w", np.tile(np.arange(levels + 1.0), (n, 1)),
            ("nCells", "nVertLevelsP1"))
        put("qv", 0.005 + 0.0 * base, ("nCells", "nVertLevels"))
        if dialect == "native":
            put("pressure_p", 100.0 + 0.0 * base, ("nCells", "nVertLevels"))
            put("pressure_base", 90000.0 - 1000.0 * level + 0.0 * base,
                ("nCells", "nVertLevels"))
            put("t2m", _linear(lat, lon, 0.01), ("nCells",))
            put("swdnb", 400.0 + 0.0 * lat, ("nCells",))
            put("coszr", 0.5 + 0.0 * lat, ("nCells",))
        else:
            put("pressure", 90000.0 - 1000.0 * level + 0.0 * base,
                ("nCells", "nVertLevels"))
            put("t2", _linear(lat, lon, 0.01), ("nCells",))
            put("surface_pressure", 95000.0 + 0.0 * lat, ("nCells",))
        put("u10", _linear(lat, lon, 0.001), ("nCells",))
        put("rainnc", 1.5 + 0.0 * lat, ("nCells",))


SITES_LAT = np.array([LAT0 + 0.03, LAT0 + 0.07, LAT0 + 0.5])
SITES_LON = np.array([LON0 + 0.02, LON0 + 0.06, LON0])
HEIGHTS = (10.0, 25.0, 40.0, 200.0)


def test_barycentric_interpolation_and_heights(tmp_path):
    _write_mesh(tmp_path / "mesh.nc")
    _write_history(tmp_path / "history.nc")
    result = sample_mpas([tmp_path / "history.nc"], SITES_LAT, SITES_LON,
                         HEIGHTS, mesh_path=tmp_path / "mesh.nc")
    assert isinstance(result, SampleResult)
    assert result.inside.tolist() == [True, True, False]
    truth = _linear(SITES_LAT[:2], SITES_LON[:2])
    u = result.profile["U"]
    assert u.shape == (1, 3, 4)
    # mass level 0 sits at 10 m AGL, level 1 at 40 m: U = base + 10 * level
    np.testing.assert_allclose(u[0, :2, 0], truth, rtol=0, atol=2e-2)
    np.testing.assert_allclose(u[0, :2, 2], truth + 10.0, atol=2e-2)
    np.testing.assert_allclose(u[0, :2, 1], truth + 5.0, atol=2e-2)
    assert np.all(np.isnan(u[0, :2, 3]))           # above the top midpoint
    assert np.all(np.isnan(u[0, 2]))               # outside the mesh
    np.testing.assert_allclose(result.terrain_m[:2], 100.0)
    assert np.isnan(result.terrain_m[2])
    np.testing.assert_allclose(result.heights_m, HEIGHTS)
    assert result.dx_m == pytest.approx(STEP * np.pi / 180 * 6371229 * 0.62,
                                        rel=0.1)
    assert any("barycentric" in note for note in result.notes)
    assert any("not extrapolated" in note for note in result.notes)


def test_variable_mapping_native(tmp_path):
    _write_mesh(tmp_path / "mesh.nc")
    _write_history(tmp_path / "history.nc")
    result = sample_mpas([tmp_path / "history.nc"], SITES_LAT, SITES_LON,
                         HEIGHTS, mesh_path=tmp_path / "mesh.nc")
    assert set(result.profile) == {"U", "V", "W", "THETA", "PRES", "QVAPOR"}
    assert set(result.surface) == {"U10", "T2", "SWDOWN", "COSZEN", "RAINNC"}
    np.testing.assert_allclose(result.profile["V"][0, :2],
                               -result.profile["U"][0, :2]
                               + np.array([0.0, 5.0, 10.0, np.nan]),
                               atol=2e-2)
    # w interfaces 0,1,2,3 destagger to mass levels 0.5, 1.5, 2.5
    np.testing.assert_allclose(result.profile["W"][0, 0, :3], [0.5, 1.0, 1.5])
    # pressure_p + pressure_base
    np.testing.assert_allclose(result.profile["PRES"][0, 0, :3],
                               [90100.0, 89600.0, 89100.0])
    np.testing.assert_allclose(result.profile["THETA"][0, 0, 2], 291.0)
    np.testing.assert_allclose(result.surface["T2"][0, :2],
                               _linear(SITES_LAT[:2], SITES_LON[:2], 0.01),
                               atol=1e-3)
    assert result.times.tolist() == [np.datetime64("2026-10-10T06:00:00")]
    assert any("not sampled" in note for note in result.notes)
    available = available_variables([tmp_path / "history.nc"],
                                    mesh_path=tmp_path / "mesh.nc")
    assert available == set(result.profile) | set(result.surface)


def test_cuda_dialect_time_from_file_name(tmp_path):
    _write_mesh(tmp_path / "corridor.init.nc")
    for hour, offset in ((7, 1.0), (6, 0.0)):
        _write_history(tmp_path / f"cuda-history.2026-10-10_0{hour}.00.00.nc",
                       dialect="cuda", with_xtime=False, offset=offset)
    paths = sorted(tmp_path.glob("cuda-history.*.nc"), reverse=True)
    result = sample_mpas(paths, SITES_LAT, SITES_LON, HEIGHTS,
                         mesh_path=tmp_path / "corridor.init.nc")
    assert result.times.tolist() == [np.datetime64("2026-10-10T06:00:00"),
                                     np.datetime64("2026-10-10T07:00:00")]
    assert any("sorted" in note for note in result.notes)
    u = result.profile["U"]
    np.testing.assert_allclose(u[1, :2, 0] - u[0, :2, 0], 1.0, atol=1e-3)
    assert "PSFC" in result.surface and "T2" in result.surface
    np.testing.assert_allclose(result.profile["PRES"][0, 0, 0], 90000.0)


def test_boundary_cells_are_outside(tmp_path):
    bdy = np.zeros(9, dtype=int)
    bdy[0] = 1                         # the corner the first site touches
    _write_mesh(tmp_path / "mesh.nc", bdy=bdy)
    _write_history(tmp_path / "history.nc")
    result = sample_mpas([tmp_path / "history.nc"], SITES_LAT, SITES_LON,
                         HEIGHTS, mesh_path=tmp_path / "mesh.nc")
    assert result.inside.tolist() == [False, True, False]
    assert np.all(np.isnan(result.profile["U"][0, 0]))
    assert np.isnan(result.surface["T2"][0, 0])
    assert np.isfinite(result.surface["T2"][0, 1])


def test_open_vertices_are_ignored(tmp_path):
    lat, lon, cov = _grid()
    _write_mesh(tmp_path / "mesh.nc")
    from woof.energy.sample_hex import _read_mesh

    mesh = _read_mesh([tmp_path / "mesh.nc"], None)
    cells, weights, inside, method = horizontal_weights(
        mesh, SITES_LAT, SITES_LON)
    assert inside.tolist() == [True, True, False]
    np.testing.assert_allclose(weights.sum(axis=1), [1.0, 1.0, 0.0])
    assert "barycentric" in method


def test_nearest_cell_fallback_without_cells_on_vertex(tmp_path):
    _write_mesh(tmp_path / "mesh.nc", with_cov=False)
    _write_history(tmp_path / "history.nc")
    result = sample_mpas([tmp_path / "history.nc"], [LAT0 + 0.051],
                         [LON0 + 0.049], HEIGHTS,
                         mesh_path=tmp_path / "mesh.nc")
    assert result.inside.tolist() == [True]
    np.testing.assert_allclose(result.surface["T2"][0, 0],
                               _linear(LAT0 + STEP, LON0 + STEP, 0.01),
                               atol=1e-3)
    assert any("nearest cell" in note for note in result.notes)


def test_no_zgrid_means_no_profiles(tmp_path):
    _write_mesh(tmp_path / "mesh.nc", with_zgrid=False)
    _write_history(tmp_path / "history.nc")
    available = available_variables([tmp_path / "history.nc"],
                                    mesh_path=tmp_path / "mesh.nc")
    assert not available & {"U", "V", "W", "THETA", "PRES"}
    result = sample_mpas([tmp_path / "history.nc"], SITES_LAT, SITES_LON,
                         HEIGHTS, mesh_path=tmp_path / "mesh.nc")
    assert result.profile == {}
    assert "T2" in result.surface
    assert any("no zgrid" in note for note in result.notes)


def test_mesh_mismatch_is_refused(tmp_path):
    _write_mesh(tmp_path / "mesh.nc")
    _write_history(tmp_path / "history.nc", n_extra_cells=2)
    with pytest.raises(SampleUnavailable, match="not this history's mesh"):
        sample_mpas([tmp_path / "history.nc"], SITES_LAT, SITES_LON, HEIGHTS,
                    mesh_path=tmp_path / "mesh.nc")


def test_unknown_valid_time_is_refused(tmp_path):
    _write_mesh(tmp_path / "mesh.nc")
    _write_history(tmp_path / "run.nc", with_xtime=False)
    with pytest.raises(SampleUnavailable, match="valid time is unknown"):
        sample_mpas([tmp_path / "run.nc"], SITES_LAT, SITES_LON, HEIGHTS,
                    mesh_path=tmp_path / "mesh.nc")


def test_duplicate_times_and_unknown_keys_are_refused(tmp_path):
    _write_mesh(tmp_path / "mesh.nc")
    _write_history(tmp_path / "a.nc")
    _write_history(tmp_path / "b.nc")
    with pytest.raises(SampleUnavailable, match="share a valid time"):
        sample_mpas([tmp_path / "a.nc", tmp_path / "b.nc"], SITES_LAT,
                    SITES_LON, HEIGHTS, mesh_path=tmp_path / "mesh.nc")
    with pytest.raises(SampleUnavailable, match="unknown sample keys"):
        sample_mpas([tmp_path / "a.nc"], SITES_LAT, SITES_LON, HEIGHTS,
                    profile_vars=("TKE",), mesh_path=tmp_path / "mesh.nc")


def test_no_paths_and_no_coordinates_are_refused(tmp_path):
    with pytest.raises(SampleUnavailable, match="no MPAS history"):
        available_variables([])
    path = tmp_path / "bare.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("nCells", 3)
        ds.createVariable("t2m", "f4", ("nCells",))[:] = 1.0
    with pytest.raises(SampleUnavailable, match="latCell/lonCell"):
        sample_mpas([path], SITES_LAT, SITES_LON, HEIGHTS)


def test_zero_weight_slots_never_poison_a_site(tmp_path):
    _write_mesh(tmp_path / "mesh.nc", with_cov=False)
    _write_history(tmp_path / "history.nc")
    with netCDF4.Dataset(tmp_path / "history.nc", "a") as ds:
        values = ds.variables["t2m"][0]
        values[0] = np.nan                 # the placeholder slots' cell
        ds.variables["t2m"][0] = values
    result = sample_mpas([tmp_path / "history.nc"], [LAT0 + 0.051],
                         [LON0 + 0.049], HEIGHTS,
                         mesh_path=tmp_path / "mesh.nc")
    assert np.isfinite(result.surface["T2"][0, 0])
    assert np.all(np.isfinite(result.profile["U"][0, 0, :3]))


def test_keys_must_match_their_kind(tmp_path):
    _write_mesh(tmp_path / "mesh.nc")
    _write_history(tmp_path / "history.nc")
    with pytest.raises(SampleUnavailable, match="unknown sample keys"):
        sample_mpas([tmp_path / "history.nc"], SITES_LAT, SITES_LON, HEIGHTS,
                    surface_vars=("U10", "THETA"),
                    mesh_path=tmp_path / "mesh.nc")
    with pytest.raises(SampleUnavailable, match="unknown sample keys"):
        sample_mpas([tmp_path / "history.nc"], SITES_LAT, SITES_LON, HEIGHTS,
                    profile_vars=("U10",), mesh_path=tmp_path / "mesh.nc")
