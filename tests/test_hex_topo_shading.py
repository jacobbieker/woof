"""topo_shading = 1 on a hex mesh: the statics, the contract and the shadow search.

WRF's toposhad marches grid rows (woof.core.topo_radiation.terrain_shadow); the mesh march
(mesh_terrain_shadow, woof.hex.topo_shading) samples the same sun azimuth over Voronoi cells.  On a
plane both read the plane exactly, so their masks agree with each other and with the plane's own
answer: shadowed exactly when the terrain toward the sun rises faster than the sun.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from test_hex_slope_statics import RADIUS, _hex_patch, _mesh, _slope_physics
from woof.hex import surface_inputs as si
from woof.hex import topo_shading as ts

# A sun well up the sky in the afternoon at the patch (lat0 0.6 rad, lon0 0.3 rad).
LAT0, LON0 = 0.6, 0.3
SUN = dict(xtime_minutes=0.0, gmt=14.0, radt_minutes=0.0, declin=0.3)


def _sun(lat_deg, lon_deg, *, xtime_minutes, gmt, radt_minutes, declin):
    """toposhad's csza and sol_azi in float64 (module_radiation_driver.F), for picking test planes."""
    degrad = np.pi / 180.0
    xt24 = np.mod(xtime_minutes + radt_minutes * 0.5, 1440.0)
    hrang = 15.0 * (gmt + xt24 / 60.0 + lon_deg / 15.0 - 12.0) * degrad
    lat = lat_deg * degrad
    csza = np.sin(lat) * np.sin(declin) + np.cos(lat) * np.cos(declin) * np.cos(hrang)
    argu = np.clip((csza * np.sin(lat) - np.sin(declin)) / (np.sin(np.arccos(csza)) * np.cos(lat)), -1, 1)
    return csza, np.copysign(np.arccos(argu), np.sin(hrang)) + np.pi


def _patch(rows=15, cols=15, spacing=3000.0):
    lat, lon, cells, counts, x, y = _hex_patch(rows, cols, spacing, lat0=LAT0, lon0=LON0)
    return lat, lon, cells, counts, x, y


def test_the_statics_carry_neighbours_positions_and_spacing():
    lat, lon, cells, counts, x, y = _patch(rows=5, cols=5)
    table, geometry = ts.shadow_statics(lat, lon, cells, counts, radius_m=RADIUS)
    n = lat.size
    assert table.dtype == np.int32 and table.shape == (1, 6, n)
    assert geometry.dtype == np.float32 and geometry.shape == (1, 7, n)
    # the patch's edge cells have neighbours outside the mesh: -2, never -1 within nEdgesOnCell
    assert set(np.unique(table)) <= set(range(n)) | {ts.OUTSIDE}
    assert np.array_equal(table[0].T >= 0, cells >= 0)
    position = geometry[0, :3].astype(np.float64) + geometry[0, 3:6].astype(np.float64)
    want = RADIUS * np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    assert np.max(np.abs(position - want)) < 1e-3
    np.testing.assert_allclose(geometry[0, 6], 3000.0, rtol=1e-5)
    short = counts.copy()
    short[0] = 3
    table, _ = ts.shadow_statics(lat, lon, cells, short, radius_m=RADIUS)
    assert np.all(table[0, 3:, 0] == ts.UNUSED)


def test_a_cell_without_neighbours_and_a_planar_mesh_are_refused():
    lat, lon, cells, counts, *_ = _patch(rows=3, cols=3)
    lonely = cells.copy()
    lonely[4] = -1
    with pytest.raises(ValueError, match="no neighbour"):
        ts.shadow_statics(lat, lon, lonely, counts, radius_m=RADIUS)
    with pytest.raises(ValueError, match="planar"):
        ts.mesh_shadow_statics(_mesh(planar=True)[0])


def test_the_contract_reads_the_shadow_statics_only_with_topo_shading():
    def keywords(**extra):
        cfg = si.plan_config(_slope_physics(**extra))
        return [need.input.keyword for need in si.required_inputs(cfg)]

    shadow = ["topo_shadow_cells", "topo_shadow_geometry"]
    assert not set(shadow) & set(keywords(slope_rad=1))
    assert keywords(slope_rad=1, topo_shading=1)[-4:] == ["topo_slope", "topo_slope_azimuth", *shadow]
    # WRF reads the shadow only in the slope adjustment: without slope_rad nothing is staged
    assert not set(shadow) & set(keywords(topo_shading=1))


def test_bind_derives_the_shadow_statics_from_the_runs_mesh():
    mesh, x, y = _mesh()
    physics = _slope_physics(slope_rad=1, topo_shading=1)
    bound, receipt = si.bind(physics, None, latitude_rad=mesh.latCell,
                             longitude_rad=mesh.lonCell, context={
                                 "ivgtyp": np.ones(x.size, np.int32),
                                 "latitude_deg": np.degrees(mesh.latCell),
                                 "start_time": None, "mminlu": "MODIFIED_IGBP_MODIS_NOAH",
                                 "aerosol": None,
                                 "slope_mesh": {"mesh": mesh, "terrain_height_m": 500.0 + 0.01 * x}})
    want = ts.mesh_shadow_statics(mesh)
    np.testing.assert_array_equal(bound["topo_shadow_cells"], want[0])
    np.testing.assert_array_equal(bound["topo_shadow_geometry"], want[1])
    sealed = si.seal(si.plan_config(physics), bound, n_columns=x.size)
    assert {"topo_slope", "topo_slope_azimuth", "topo_shadow_cells",
            "topo_shadow_geometry"} <= set(sealed)
    assert "topo_shadow_cells" in str(receipt)


def _plane(x, y, azimuth, elevation_rad, h0=800.0):
    """A plane rising toward ``azimuth`` (clockwise from north) at ``elevation_rad``."""
    return h0 + np.tan(elevation_rad) * (x * np.sin(azimuth) + y * np.cos(azimuth))


def _mesh_shadow(height, lat, lon, cells, counts, shadlen=12000.0, sun=SUN):
    import cupy as cp

    from woof.core.topo_radiation import mesh_terrain_shadow

    table, geometry = ts.shadow_statics(lat, lon, cells, counts, radius_m=RADIUS)
    mask, shad = mesh_terrain_shadow(
        cp.asarray(height.astype(np.float32)[None, :]), np.degrees(lat)[None, :].astype(np.float32),
        np.degrees(lon)[None, :].astype(np.float32), table[0], geometry[0], shadlen=shadlen, **sun)
    return cp.asnumpy(mask)[0], cp.asnumpy(shad)[0]


@requires_gpu
@pytest.mark.parametrize("margin_deg,shadowed", [(5.0, True), (-5.0, False)])
def test_a_plane_is_shadowed_exactly_when_it_rises_faster_than_the_sun(margin_deg, shadowed):
    lat, lon, cells, counts, x, y = _patch()
    centre = np.hypot(x, y) < 3.5 * 3000.0
    csza, azimuth = _sun(np.degrees(LAT0), np.degrees(LON0), **SUN)
    sun_elevation = np.arcsin(csza)
    assert 0.2 < csza < 0.8
    height = _plane(x, y, azimuth, sun_elevation + np.radians(margin_deg))
    mask, shad = _mesh_shadow(height, lat, lon, cells, counts)
    assert np.all(mask[centre] == int(shadowed)), mask[centre]
    assert np.all(shad >= height.astype(np.float32) - np.float32(0.001))
    if not shadowed:
        np.testing.assert_array_equal(shad, (height.astype(np.float32) - np.float32(0.001)))


@requires_gpu
def test_the_mesh_mask_is_wrfs_grid_mask_on_a_plane():
    """E1 on planar terrain: WRF's toposhad on a grid and the mesh march agree."""
    import cupy as cp

    from woof.core.topo_radiation import terrain_shadow

    csza, azimuth = _sun(np.degrees(LAT0), np.degrees(LON0), **SUN)
    for margin in (4.0, -4.0):
        rise = np.arcsin(csza) + np.radians(margin)
        dx = 3000.0
        j, i = np.mgrid[0:15, 0:15].astype(np.float64)
        grid_height = _plane((i - 7) * dx, (j - 7) * dx, azimuth, rise).astype(np.float32)
        ones, zeros = np.ones_like(grid_height), np.zeros_like(grid_height)
        lat_grid = np.full_like(grid_height, np.degrees(LAT0))
        lon_grid = np.full_like(grid_height, np.degrees(LON0))
        grid_mask, _ = terrain_shadow(grid_height, lat_grid, lon_grid, zeros, ones, dx=dx, dy=dx,
                                      shadlen=12000.0, **SUN)
        lat, lon, cells, counts, x, y = _patch()
        mask, _ = _mesh_shadow(_plane(x, y, azimuth, rise), lat, lon, cells, counts)
        centre = int(np.argmin(np.hypot(x, y)))
        assert int(cp.asnumpy(grid_mask)[7, 7]) == int(mask[centre]) == int(margin > 0)


@requires_gpu
def test_a_hill_shadows_the_cells_behind_it_and_a_night_sky_shadows_nothing():
    lat, lon, cells, counts, x, y = _patch(rows=21, cols=21)
    low = dict(SUN, gmt=16.5)
    csza, azimuth = _sun(np.degrees(LAT0), np.degrees(LON0), **low)
    assert 0.2 < csza < 0.3   # the sun 14 degrees up
    # a 1500 m hill 4.5 km toward the sun from the patch centre: 18 degrees up from there
    hx, hy = 4500.0 * np.sin(azimuth), 4500.0 * np.cos(azimuth)
    height = 300.0 + 1500.0 * np.exp(-((x - hx) ** 2 + (y - hy) ** 2) / (2 * 2500.0 ** 2))
    mask, shad = _mesh_shadow(height, lat, lon, cells, counts, shadlen=25000.0, sun=low)
    behind = int(np.argmin(np.hypot(x, y)))
    sunny = int(np.argmin(np.hypot(x - 2 * hx, y - 2 * hy)))
    assert mask[behind] == 1 and shad[behind] > height[behind]
    assert mask[sunny] == 0
    import cupy as cp

    from woof.core.topo_radiation import mesh_terrain_shadow

    table, geometry = ts.shadow_statics(lat, lon, cells, counts, radius_m=RADIUS)
    night = dict(SUN, gmt=2.0)
    assert _sun(np.degrees(LAT0), np.degrees(LON0), **night)[0] < 1e-2
    mask, shad = mesh_terrain_shadow(
        cp.asarray(height.astype(np.float32)[None, :]), np.degrees(lat)[None, :].astype(np.float32),
        np.degrees(lon)[None, :].astype(np.float32), table[0], geometry[0], shadlen=25000.0, **night)
    assert not cp.asnumpy(mask).any()
    np.testing.assert_array_equal(cp.asnumpy(shad)[0], height.astype(np.float32) - np.float32(0.001))


@requires_gpu
def test_a_mesa_casts_its_analytic_shadow_edge_cells_included():
    """E1 against a closed form: flat ground before a ramp up to a plateau H higher, toward a sun at
    elevation e.  A ground cell is shadowed when the plateau's near rim is within L = H / tan(e) of it.
    The march samples every dxc, so the computed shadow ends between L - dxc and L.  Every flat cell whose
    ray reaches the rim inside the mesh is held to that, the cells on the mesh's outer edge included
    (their scan reads the on-mesh triangles and goes on along and into the mesh)."""
    from scipy.spatial import Delaunay

    lat, lon, cells, counts, x, y = _patch(rows=31, cols=31)
    low = dict(SUN, gmt=16.5)
    csza, azimuth = _sun(np.degrees(LAT0), np.degrees(LON0), **low)
    tan_sun = np.tan(np.arcsin(csza))
    along = x * np.sin(azimuth) + y * np.cos(azimuth)   # metres toward the sun
    rise, ramp, dxc = 2000.0, 3000.0, 3000.0
    height = 300.0 + rise * np.clip(along / ramp, 0.0, 1.0)
    reach = rise / tan_sun
    assert 6000.0 < reach < 9000.0
    mask, shad = _mesh_shadow(height, lat, lon, cells, counts, shadlen=25000.0, sun=low)
    rim = ramp - along                                   # distance to the rim along the ray
    hull = Delaunay(np.c_[x, y])
    steps = np.arange(1, int(25000.0 / dxc) + 2)[None, :] * dxc
    on_mesh = np.all((hull.find_simplex(np.stack(
        [x[:, None] + steps * np.sin(azimuth), y[:, None] + steps * np.cos(azimuth)], -1)) >= 0)
        | (steps > rim[:, None] + dxc), axis=1)
    flat = along <= 0.0
    edge = np.any(cells < 0, axis=1)
    shadowed = flat & on_mesh & (rim <= reach - dxc - 100.0)
    sunlit = (flat & (rim > reach + 100.0)) | (along >= ramp)
    assert shadowed.sum() > 15 and (shadowed & edge).sum() > 0 and sunlit.sum() > 100
    assert np.all(mask[shadowed] == 1), np.flatnonzero(shadowed & (mask == 0))
    assert np.all(mask[sunlit] == 0), np.flatnonzero(sunlit & (mask == 1))
    # HT_SHAD: the height the sun must clear, the rim's line of sight at the cell
    np.testing.assert_array_less(height[shadowed], shad[shadowed])


def _reference_mask(lat, lon, height, x, y, *, shadlen, spacing, sun):
    """An independent float64 march: samples k * spacing along the great circle at the sun azimuth,
    HT linear on scipy's Delaunay of the patch (whose x, y are the gnomonic plane the patch is built
    on), ending outside the hull.  Returns WRF's test's largest margin sin(topoelev) - csza per cell
    (NaN where the sun is down)."""
    from scipy.interpolate import LinearNDInterpolator

    interp = LinearNDInterpolator(np.c_[x, y], height)
    p0 = np.array([np.cos(LAT0) * np.cos(LON0), np.cos(LAT0) * np.sin(LON0), np.sin(LAT0)])
    east = np.array([-np.sin(LON0), np.cos(LON0), 0.0])
    north = np.array([-np.sin(LAT0) * np.cos(LON0), -np.sin(LAT0) * np.sin(LON0), np.cos(LAT0)])
    margin = np.full(lat.size, np.nan)
    for o in range(lat.size):
        csza, azimuth = _sun(np.degrees(lat[o]), np.degrees(lon[o]), **sun)
        if csza < 1e-2:
            continue
        best = -np.inf
        for k in range(1, int(shadlen / spacing + 1) + 1):
            angle = k * spacing / RADIUS
            la = np.arcsin(np.sin(lat[o]) * np.cos(angle)
                           + np.cos(lat[o]) * np.sin(angle) * np.cos(azimuth))
            lo = lon[o] + np.arctan2(np.sin(azimuth) * np.sin(angle) * np.cos(lat[o]),
                                     np.cos(angle) - np.sin(lat[o]) * np.sin(la))
            v = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
            v = v / (v @ p0) * RADIUS
            h = float(interp(v @ east, v @ north))
            if not np.isfinite(h):
                break
            best = max(best, np.sin(np.arctan((h - height[o]) / (k * spacing))) - csza)
        margin[o] = best
    return margin


@requires_gpu
@pytest.mark.parametrize("gmt", [17.0, 17.25])
def test_the_mesh_mask_is_an_independent_float64_march_on_rough_terrain(gmt):
    """E1 on terrain with no closed form: the kernel's mask against a float64 march written apart from
    it (great-circle samples, scipy's linear interpolant), cell for cell, except where the float64
    margin is within 1e-4 of WRF's threshold, where float32 may fall either side."""
    lat, lon, cells, counts, x, y = _patch(rows=25, cols=25)
    rng = np.random.default_rng(7)
    height = 600.0 + np.zeros_like(x)
    for _ in range(12):
        cx, cy = rng.uniform(-30000.0, 30000.0, 2)
        amp, width = rng.uniform(800.0, 2500.0), rng.uniform(2500.0, 6000.0)
        height += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * width ** 2))
    sun = dict(SUN, gmt=gmt)
    mask, _ = _mesh_shadow(height, lat, lon, cells, counts, shadlen=15000.0, sun=sun)
    margin = _reference_mask(lat, lon, height.astype(np.float32).astype(np.float64), x, y,
                             shadlen=15000.0, spacing=3000.0, sun=sun)
    clear = np.abs(margin) > 1e-4
    assert clear.sum() > 0.97 * x.size
    want = (margin >= 0).astype(np.int32)
    assert 20 < want[clear].sum() < clear.sum() - 20
    np.testing.assert_array_equal(mask[clear], want[clear])


@requires_gpu
def test_the_engine_runs_the_mesh_search_only_with_the_mesh_statics():
    import cupy as cp

    from woof.core.topo_radiation import TopoShortwave

    lat, lon, cells, counts, x, y = _patch(rows=3, cols=3)
    n = x.size
    shape = (1, n)
    table, geometry = ts.shadow_statics(lat, lon, cells, counts, radius_m=RADIUS)
    state = SimpleNamespace(ht=cp.zeros(shape, cp.float32))
    cfg = SimpleNamespace(nested=False, topo_shading=1, dx=3000.0, dy=3000.0,
                          shadlen=25000.0, ra_lw_physics=1, ra_sw_physics=1)
    statics = (np.full(shape, 0.1, np.float32), np.full(shape, 3.0, np.float32))
    latitude = np.degrees(lat)[None, :].astype(np.float32)

    def build(shadow, **overrides):
        return TopoShortwave(state=state, cfg=SimpleNamespace(**{**vars(cfg), **overrides}),
                             fields={}, latitude=latitude, longitude=latitude, start_time=None,
                             slope_statics=statics, shadow_mesh=shadow)

    carrier = build((table, geometry))
    assert carrier.shadow_mesh is not None and "shadowmask" in carrier.fields
    with pytest.raises(ValueError, match="topo_shadow_cells"):
        build(None)
    with pytest.raises(ValueError, match="without a parent"):
        build((table, geometry), nested=True)
    with pytest.raises(ValueError, match="topo_shading = 1"):
        build((table, geometry), topo_shading=0)
    bad = table.copy()
    bad[0, 0, 0] = n
    with pytest.raises(ValueError, match="index the mesh"):
        build((bad, geometry))
