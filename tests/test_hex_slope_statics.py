"""slope_rad on the hex mesh: SLOPE/SLP_AZI from each cell's own neighbours.

woof.hex.slope_statics replaces WRF's grid stencil (start_em.F) with a
least-squares tangent-plane fit over cellsOnCell; these tests hold the fit to
planar terrain (which both reproduce exactly), to WRF's aspect convention and
flat threshold, and the surface-input contract and engine to their refusals.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex import slope_statics as ss
from woof.hex import surface_inputs as si

RADIUS = 6371229.0
#: Axial offsets of a hexagonal lattice's six neighbours.
_AXIAL = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))


def _hex_patch(rows=7, cols=7, spacing=3000.0, lat0=0.0, lon0=0.3):
    """A hexagonal-lattice patch on the sphere: ``(lat, lon, cellsOnCell,
    nEdgesOnCell, east_m, north_m)``; edge cells have missing (-1) neighbours."""
    index = {}
    xy = []
    for r in range(rows):
        for q in range(cols):
            index[(q, r)] = len(xy)
            xy.append((spacing * (q + 0.5 * r), spacing * (np.sqrt(3.0) / 2.0) * r))
    xy = np.asarray(xy) - np.mean(xy, axis=0)
    p0 = np.array([np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)])
    east = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
    north = np.array([-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)])
    p = p0 + (xy[:, :1] * east + xy[:, 1:] * north) / RADIUS
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    lat = np.arcsin(p[:, 2])
    lon = np.arctan2(p[:, 1], p[:, 0])
    cells = np.full((len(xy), 6), -1, dtype=np.int64)
    counts = np.full(len(xy), 6, dtype=np.int64)
    for (q, r), i in index.items():
        for slot, (dq, dr) in enumerate(_AXIAL):
            cells[i, slot] = index.get((q + dq, r + dr), -1)
    return lat, lon, cells, counts, xy[:, 0], xy[:, 1]


def _wrf_slope_aspect(hx, hy):
    slope = np.arctan(np.hypot(hx, hy))
    return slope, np.arctan2(hx, hy) + np.pi


@pytest.mark.parametrize("hx,hy", [(0.05, 0.0), (0.0, 0.05), (-0.03, 0.04), (0.2, -0.1)])
def test_planar_terrain_is_reproduced_on_every_cell_boundary_included(hx, hy):
    lat, lon, cells, counts, x, y = _hex_patch()
    height = 500.0 + hx * x + hy * y
    slope, azimuth = ss.slope_aspect(lat, lon, cells, counts, height, radius_m=RADIUS)
    want_slope, want_azimuth = _wrf_slope_aspect(hx, hy)
    assert slope.dtype == np.float32 and azimuth.dtype == np.float32
    np.testing.assert_allclose(slope, want_slope, rtol=0, atol=2e-6)
    np.testing.assert_allclose(azimuth, want_azimuth, rtol=0, atol=2e-5)


def test_the_azimuth_is_each_cells_own_geographic_north():
    """Away from the equator a plane in the patch centre's tangent frame turns
    against each cell's own north by the meridian convergence (dlon sin lat);
    the centre cell reads it exactly, the edges within that angle."""
    lat0 = 0.7
    lat, lon, cells, counts, x, y = _hex_patch(lat0=lat0)
    slope, azimuth = ss.slope_aspect(lat, lon, cells, counts, 0.05 * x, radius_m=RADIUS)
    centre = int(np.argmin(np.hypot(x, y)))
    assert abs(azimuth[centre] - 1.5 * np.pi) < 2e-6
    convergence = np.abs(lon - lon[centre]) * np.sin(lat0)
    np.testing.assert_allclose(azimuth, 1.5 * np.pi + np.sign(lon - lon[centre]) * convergence,
                               atol=2e-5)


def test_the_aspect_is_wrfs_downslope_facing_direction():
    lat, lon, cells, counts, x, y = _hex_patch()
    # Terrain rising toward the east faces west (3 pi / 2); toward the north, south (pi).
    _, east = ss.slope_aspect(lat, lon, cells, counts, 0.1 * x, radius_m=RADIUS)
    _, north = ss.slope_aspect(lat, lon, cells, counts, 0.1 * y, radius_m=RADIUS)
    np.testing.assert_allclose(east, 1.5 * np.pi, atol=2e-5)
    np.testing.assert_allclose(north, np.pi, atol=2e-5)


def test_flat_and_sub_threshold_terrain_is_zero_slope_zero_azimuth():
    lat, lon, cells, counts, x, _ = _hex_patch()
    for height in (np.full(x.size, 812.0), 812.0 + 0.5 * ss.FLAT_SLOPE * x):
        slope, azimuth = ss.slope_aspect(lat, lon, cells, counts, height, radius_m=RADIUS)
        assert not slope.any() and not azimuth.any()


def test_a_cell_without_two_independent_neighbours_gets_no_gradient():
    lat, lon, cells, counts, x, _ = _hex_patch(rows=1, cols=4)
    # A single row: every neighbour lies on one line, so no plane is fitted.
    slope, azimuth = ss.slope_aspect(lat, lon, cells, counts, 0.1 * x, radius_m=RADIUS)
    assert not slope.any() and not azimuth.any()
    # A neighbour past nEdgesOnCell, or outside the mesh, is not read.
    lat, lon, cells, counts, x, y = _hex_patch()
    noisy = cells.copy()
    noisy[noisy < 0] = lat.size  # MPAS's nCells+1 "outside" cell, zero-based
    a = ss.slope_aspect(lat, lon, noisy, counts, 0.1 * x + 0.05 * y, radius_m=RADIUS)
    b = ss.slope_aspect(lat, lon, cells, counts, 0.1 * x + 0.05 * y, radius_m=RADIUS)
    np.testing.assert_array_equal(a[0], b[0])
    # Past nEdgesOnCell: an interior cell counting five edges ignores its sixth
    # slot, whatever cell sits there.
    centre = int(np.argmin(np.hypot(x, y)))
    short = counts.copy()
    short[centre] = 5
    far = int(np.argmax(np.hypot(x, y)))
    other = cells.copy()
    other[centre, 5] = far
    height = 0.1 * x + 0.05 * y + 1e-3 * (x * y)
    a = ss.slope_aspect(lat, lon, cells, short, height, radius_m=RADIUS)
    b = ss.slope_aspect(lat, lon, other, short, height, radius_m=RADIUS)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_the_fit_refuses_disagreeing_arrays_and_non_finite_terrain():
    lat, lon, cells, counts, x, _ = _hex_patch()
    with pytest.raises(ValueError, match="do not agree"):
        ss.slope_aspect(lat, lon, cells[:-1], counts, x, radius_m=RADIUS)
    with pytest.raises(ValueError, match="finite"):
        ss.slope_aspect(lat, lon, cells, counts, np.where(x > 0, np.nan, x), radius_m=RADIUS)


def _mesh(planar=False):
    lat, lon, cells, counts, x, y = _hex_patch()
    return SimpleNamespace(latCell=lat, lonCell=lon, cellsOnCell=cells.T.copy(),
                           nEdgesOnCell=counts, sphere_radius=RADIUS,
                           on_a_sphere="NO" if planar else "YES"), x, y


def test_the_mesh_entry_reads_a_hex_mesh_and_refuses_a_planar_one():
    mesh, x, y = _mesh()
    slope, _ = ss.mesh_slope_aspect(mesh, 0.1 * x)
    np.testing.assert_allclose(slope, np.arctan(0.1), atol=2e-6)
    planar, _, _ = _mesh(planar=True)
    with pytest.raises(ValueError, match="planar"):
        ss.mesh_slope_aspect(planar, 0.1 * x)


def _slope_physics(**extra):
    from woof.core.mpas_column_suite import MPAS_DEFAULT_PHYSICS

    return dict(MPAS_DEFAULT_PHYSICS, **extra)


def test_the_contract_reads_slope_statics_only_when_slope_rad_adjusts_the_surface():
    def keywords(**extra):
        return [need.input.keyword for need in si.required_inputs(
            si.plan_config(_slope_physics(**extra)))]

    assert "topo_slope" not in keywords()
    assert {"topo_slope", "topo_slope_azimuth"} <= set(keywords(slope_rad=1))
    # WRF's own no-op: no land surface to hand the adjusted flux to.
    assert "topo_slope" not in keywords(slope_rad=1, sf_surface_physics=0)


def test_bind_derives_the_statics_from_the_runs_mesh_and_terrain():
    mesh, x, y = _mesh()
    height = 300.0 + 0.1 * x - 0.02 * y
    physics = _slope_physics(slope_rad=1)
    inputs, receipt = si.bind(physics, None, latitude_rad=mesh.latCell,
                              longitude_rad=mesh.lonCell, context={
                                  "ivgtyp": np.ones(x.size, np.int32),
                                  "latitude_deg": np.degrees(mesh.latCell),
                                  "start_time": None, "mminlu": "MODIFIED_IGBP_MODIS_NOAH",
                                  "aerosol": None,
                                  "slope_mesh": {"mesh": mesh, "terrain_height_m": height}})
    want = ss.mesh_slope_aspect(mesh, height)
    np.testing.assert_array_equal(inputs["topo_slope"], want[0])
    np.testing.assert_array_equal(inputs["topo_slope_azimuth"], want[1])
    assert "topo_slope" in str(receipt)


def test_derive_refuses_a_context_without_the_mesh():
    with pytest.raises(ValueError, match="mesh and terrain"):
        ss.derive(None, {}, 0)


@requires_gpu
def test_the_fit_matches_wrfs_stencil_on_a_plane():
    """E1: on planar terrain WRF's start_em stencil and the mesh fit agree."""
    from woof.core.topo_radiation import slope_geometry

    import cupy as cp

    hx, hy, dx = 0.07, -0.04, 3000.0
    j, i = np.mgrid[0:6, 0:8].astype(np.float64)
    ht = (400.0 + hx * dx * i + hy * dx * j).astype(np.float32)
    ones, zeros = np.ones_like(ht), np.zeros_like(ht)
    slope, azi = slope_geometry(ht, ones, ones, zeros, ones, dx=dx, dy=dx)
    lat, lon, cells, counts, x, y = _hex_patch()
    fit = ss.slope_aspect(lat, lon, cells, counts, 400.0 + hx * x + hy * y, radius_m=RADIUS)
    mine = [value[int(np.argmin(np.hypot(x, y)))] for value in fit]
    np.testing.assert_allclose(cp.asnumpy(slope)[1:-1, 1:-1], mine[0], atol=2e-6)
    np.testing.assert_allclose(cp.asnumpy(azi)[1:-1, 1:-1], mine[1], atol=2e-5)


@requires_gpu
def test_the_engine_holds_caller_statics_to_its_shape_and_refusals():
    import cupy as cp

    from woof.core.topo_radiation import TopoShortwave

    shape = (1, 4)
    state = SimpleNamespace(ht=cp.zeros(shape, cp.float32))
    cfg = SimpleNamespace(nested=False, topo_shading=0, dx=3000.0, dy=3000.0,
                          shadlen=25000.0, ra_lw_physics=1, ra_sw_physics=1)
    lat = np.full(shape, 40.0, np.float32)
    good = (np.full(shape, 0.1, np.float32), np.full(shape, 3.0, np.float32))

    def build(statics, **overrides):
        return TopoShortwave(state=state, cfg=SimpleNamespace(**{**vars(cfg), **overrides}),
                             fields={}, latitude=lat, longitude=lat, start_time=None,
                             slope_statics=statics)

    carrier = build(good)
    np.testing.assert_array_equal(cp.asnumpy(carrier.fields["slope"]), good[0])
    np.testing.assert_array_equal(cp.asnumpy(carrier.fields["slp_azi"]), good[1])
    with pytest.raises(ValueError, match="topo_shading"):
        build(good, topo_shading=1)
    with pytest.raises(ValueError, match=r"\[0, pi/2\)"):
        build((np.full(shape, 1.6, np.float32), good[1]))
    with pytest.raises(ValueError, match="finite"):
        build((good[0], np.full(shape, np.nan, np.float32)))


def test_a_terrain_of_another_size_and_a_missing_mesh_are_refused():
    mesh, x, _ = _mesh()
    with pytest.raises(ValueError, match="terrain has"):
        ss.mesh_slope_aspect(mesh, np.append(x, 0.0))
    with pytest.raises(ValueError, match="mesh and terrain"):
        ss.derive(None, {"slope_mesh": {"mesh": None, "terrain_height_m": x}}, 0)


def test_preflight_names_a_planar_grid(tmp_path):
    import netCDF4

    for flag, refused in (("YES", False), ("NO", True)):
        path = tmp_path / f"{flag}.grid.nc"
        with netCDF4.Dataset(path, "w") as dataset:
            dataset.on_a_sphere = flag
        problem = ss.planar_mesh_problem(path)
        assert (problem is not None) == refused
        if refused:
            assert "slope_rad = 0" in problem
