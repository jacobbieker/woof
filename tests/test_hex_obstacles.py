"""woof's opt-in porous immersed obstacles on the hex C-grid (obstacle_opt = 1).

Host: the face coefficients from a hex buildings product (woof.hex.obstacles),
the Voronoi aggregation (woof.static.buildings.VoronoiGrid) and the door's
flags and refusals.  Device: the two kernels against a float32 numpy replay,
and off-is-identical in child processes under each numerics mode.

Nothing here imports woof.hex.obstacles at module level: the identity test
checks that a default run never loads it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _numerics_child  # noqa: E402


def _mesh(level=2, radius=None):
    from woof.verify.cases import hex_obstacles as case

    return case.sphere_mesh(level, 1.0, **({} if radius is None else {"radius": radius}))


def _host(mesh, nz=10, ztop=50.0):
    from woof.hex.obstacles import HostMesh

    a = mesh.arrays
    nc = int(np.asarray(a["areaCell"]).size)
    z_w = np.repeat(np.linspace(0.0, ztop, nz + 1)[:, None], nc, axis=1)
    return HostMesh(
        cells_on_edge=np.asarray(a["cellsOnEdge"], np.int64),
        edges_on_cell=np.asarray(a["edgesOnCell"], np.int64),
        n_edges_on_cell=np.asarray(a["nEdgesOnCell"], np.int64),
        dc_edge=np.asarray(a["dcEdge"], np.float64), dv_edge=np.asarray(a["dvEdge"], np.float64),
        area_cell=np.asarray(a["areaCell"], np.float64), angle_edge=np.asarray(a["angleEdge"], np.float64),
        z_w=z_w, valid_cell=np.ones(nc, bool))


def _column(mesh, cell, height):
    """A product with one solid column of ``height`` metres at ``cell``."""
    from woof.verify.cases import hex_obstacles as case

    morph = case.canyon_morphology(mesh, lat_extent=-1.0)
    morph.lambda_p[0, cell] = morph.lambda_b[0, cell] = 1.0
    morph.mean_height[0, cell] = height
    morph.bldg_hist[0, 0, cell] = 1.0
    morph.bldg_hbin[0, 0, cell] = height
    return morph


def test_settings_take_phase_ones_defaults_and_refuse_bad_values():
    from woof.hex.obstacles import DEFAULTS, ObstacleSettings

    got = ObstacleSettings().as_dict()
    assert got["obstacle_opt"] == 1
    assert {k: got[k] for k in DEFAULTS} == DEFAULTS
    for bad in (dict(obstacle_solid_fraction=0.0), dict(obstacle_solid_fraction=1.5),
                dict(obstacle_relax_factor=-1.0), dict(obstacle_wall_z0=float("nan")),
                dict(obstacle_drag_coefficient=0.0)):
        with pytest.raises(ValueError):
            ObstacleSettings(**bad)


def test_a_solid_column_relaxes_its_own_faces_and_walls_its_neighbours():
    from woof.hex.obstacles import ObstacleSettings, build_coefficients

    mesh = _mesh()
    host = _host(mesh)
    cell, layers = 7, 4                                     # 20 m on 5 m layers
    settings = ObstacleSettings()
    edges, wfaces, receipt = build_coefficients(host, _column(mesh, cell, 20.0), settings)
    ne, nc = host.n_edges, host.n_cells
    assert receipt["solid_cells"] == layers and receipt["porous_cells"] == 0
    a = np.zeros(10 * ne)
    b = np.zeros(10 * ne)
    a[edges[0]], b[edges[0]] = edges[1], edges[2]
    a, b = a.reshape(10, ne), b.reshape(10, ne)
    own = host.edges_on_cell[cell, :host.n_edges_on_cell[cell]]
    # Every face of the column's solid layers is relaxed at phase 1's rate, and only those.
    assert np.all(a[:layers, own] == settings.obstacle_relax_factor)
    assert np.count_nonzero(a) == layers * own.size
    # Above the roof the column's faces carry the roof wall law, not the relaxation.
    assert np.all(a[layers, own] == 0) and np.all(b[layers, own] > 0)
    # Faces of the neighbours that touch the column's side walls carry a wall term.
    assert np.count_nonzero(b[:layers]) > 0
    assert np.all(b[layers + 1:] == 0)
    # w faces: the interfaces inside and on top of the column are relaxed.
    wa = np.zeros(11 * nc)
    wa[wfaces[0]] = wfaces[1]
    wa = wa.reshape(11, nc)
    assert np.flatnonzero(wa[:, cell]).tolist() == list(range(1, layers + 1))
    assert np.count_nonzero(wa[:, np.arange(nc) != cell]) == 0


def test_no_buildings_means_no_faces_and_the_ground_wall_is_case_only():
    from woof.hex.obstacles import ObstacleSettings, build_coefficients
    from woof.verify.cases import hex_obstacles as case

    mesh = _mesh()
    host = _host(mesh)
    empty = case.canyon_morphology(mesh, lat_extent=-1.0)
    edges, wfaces, receipt = build_coefficients(host, empty, ObstacleSettings())
    assert edges[0].size == 0 and wfaces[0].size == 0 and receipt["faces"] == [0, 0]
    edges, _, _ = build_coefficients(host, empty, ObstacleSettings(), ground_wall=True)
    assert edges[0].size == host.n_edges and np.all(edges[0] < host.n_edges)


def test_a_product_for_another_mesh_or_a_coarse_mesh_is_refused():
    from woof.hex.obstacles import ObstacleSettings, build_coefficients

    mesh = _mesh()
    host = _host(mesh)
    wrong = _column(_mesh(level=1), 3, 20.0)
    with pytest.raises(ValueError, match="needs a hex product"):
        build_coefficients(host, wrong, ObstacleSettings())
    coarse = _mesh(radius=6.371229e6)
    with pytest.raises(ValueError, match="urban canopy"):
        build_coefficients(_host(coarse), _column(coarse, 7, 20.0), ObstacleSettings())


def test_wall_laws_are_evaluated_only_where_a_wall_stands():
    """A padded column's zero dz, or a thin layer far from any building, is
    never put through the wall law's z0 check."""
    from woof.hex.obstacles import ObstacleSettings, build_coefficients

    mesh = _mesh()
    host = _host(mesh)
    xyz = np.stack([np.asarray(mesh.arrays[k], np.float64) for k in ("xCell", "yCell", "zCell")], 1)
    far = int(np.argmax(np.linalg.norm(xyz - xyz[7], axis=1)))
    host.z_w[:, far] = 0.0                                  # a padded column: dz = 0
    host.valid_cell[far] = False
    edges, _, _ = build_coefficients(host, _column(mesh, 7, 20.0), ObstacleSettings())
    assert edges[0].size > 0
    # z0 = 1 m is too rough for the 5 m layers, but only where a wall stands.
    with pytest.raises(ValueError, match="too rough"):
        build_coefficients(host, _column(mesh, 7, 20.0), ObstacleSettings(obstacle_wall_z0=1.0))
    empty = _column(mesh, 7, 0.0)
    edges, wfaces, _ = build_coefficients(host, empty, ObstacleSettings(obstacle_wall_z0=1.0))
    assert edges[0].size == 0 and wfaces[0].size == 0


def test_voronoi_grid_neighbours_and_outside_points():
    from woof.static.buildings import VoronoiGrid

    mesh = _mesh()
    a = mesh.arrays
    grid = VoronoiGrid.from_mesh(mesh)
    one = np.zeros((1, grid.n_cells), bool)
    one[0, 7] = True
    grown = grid.dilate(one)[0]
    coc = np.asarray(a["cellsOnCell"], np.int64)[7, :int(a["nEdgesOnCell"][7])]
    assert set(np.flatnonzero(grown).tolist()) == {7, *coc.tolist()}
    # The Delaunay fallback finds the same neighbours.
    bare = VoronoiGrid(a["latCell"], a["lonCell"], a["areaCell"], radius=grid.radius)
    assert np.array_equal(bare.dilate(one), grid.dilate(one))
    # A limited-area grid (one cell and its ring) puts far points outside.
    keep = np.flatnonzero(grown)
    patch = VoronoiGrid(np.asarray(a["latCell"])[keep], np.asarray(a["lonCell"])[keep],
                        np.asarray(a["areaCell"])[keep], radius=grid.radius)
    lat = np.asarray(a["latCell"], np.float64)
    lon = np.asarray(a["lonCell"], np.float64)
    got = patch.cells(np.degrees(lon), np.degrees(lat), "EPSG:4326")
    assert np.array_equal(got[keep], np.arange(keep.size))
    far = np.arccos(np.clip(np.sin(lat) * np.sin(lat[7])
                            + np.cos(lat) * np.cos(lat[7]) * np.cos(lon - lon[7]), -1, 1)) > 1.0
    assert far.any() and np.all(got[far] == -1)


def test_voronoi_aggregation_puts_the_cube_in_the_cells_under_it():
    from woof.verify.cases import hex_obstacles as case
    from woof.verify.cases.hex_obstacles import H, RADIUS

    mesh = _mesh(level=4)
    morph = case.cube_morphology(mesh)
    assert morph.shape == (1, mesh.dimensions["nCells"])
    a = mesh.arrays
    x = np.asarray(a["lonCell"], np.float64) * RADIUS
    y = np.asarray(a["latCell"], np.float64) * RADIUS
    phi = morph.solid_fraction(np.zeros((1, x.size)), np.full((1, x.size), 0.5 * H))[0]
    core = (np.abs(x) < 0.3 * H) & (np.abs(y) < 0.3 * H)
    far = np.hypot(x, y) > 1.5 * H
    assert core.any() and np.all(phi[core] > 0.99)
    assert np.all(phi[far] == 0)
    above = morph.solid_fraction(np.full((1, x.size), 1.05 * H), np.full((1, x.size), 2 * H))[0]
    assert np.all(above == 0)
    # The footprint area is the cube's, to the raster's resolution.
    area = float(np.sum(np.asarray(morph.lambda_p[0], np.float64) * np.asarray(a["areaCell"], np.float64)))
    assert abs(area - H * H) < 0.05 * H * H


# --------------------------------------------------------------------------- door


def _args(**kw):
    values = dict(obstacle_opt=None, obstacle_file=None, obstacle_solid_fraction=None,
                  obstacle_relax_factor=None, obstacle_drag_coefficient=None, obstacle_wall_z0=None)
    values.update(kw)
    return SimpleNamespace(**values)


def test_the_door_is_off_by_default_and_refuses_keys_without_the_switch(tmp_path):
    from woof.hex import forecast_door as door

    assert door._resolve_obstacles(_args()) is None
    assert door._resolve_obstacles(_args(obstacle_opt=0)) is None
    with pytest.raises(Exception, match="never read"):
        door._resolve_obstacles(_args(obstacle_file=tmp_path / "b.nc"))
    with pytest.raises(Exception, match="--obstacle-relax-factor"):
        door._resolve_obstacles(_args(obstacle_relax_factor=5.0))


def test_the_door_resolves_the_settings_and_refuses_what_it_cannot_honour(tmp_path):
    from woof.hex import forecast_door as door

    product = tmp_path / "buildings.nc"
    product.write_bytes(b"x")
    with pytest.raises(Exception, match="needs --obstacle-file"):
        door._resolve_obstacles(_args(obstacle_opt=1))
    with pytest.raises(Exception, match="does not exist"):
        door._resolve_obstacles(_args(obstacle_opt=1, obstacle_file=tmp_path / "missing.nc"))
    with pytest.raises(Exception, match="only 0"):
        door._resolve_obstacles(_args(obstacle_opt=2, obstacle_file=product))
    with pytest.raises(Exception, match="must lie in"):
        door._resolve_obstacles(_args(obstacle_opt=1, obstacle_file=product, obstacle_solid_fraction=2.0))
    for flag in ("local_timestep", "fullstep_local_timestep", "partial_nest"):
        with pytest.raises(Exception, match="never applied"):
            door._resolve_obstacles(_args(obstacle_opt=1, obstacle_file=product, **{flag: True}))
    got = door._resolve_obstacles(_args(obstacle_opt=1, obstacle_file=product, obstacle_relax_factor=4.0))
    assert got["obstacle_opt"] == 1 and got["obstacle_relax_factor"] == 4.0
    assert got["obstacle_file"] == str(product.resolve())
    assert got["obstacle_solid_fraction"] == 0.5


def test_the_forecast_parser_and_the_partitioned_executor():
    from woof.hex import forecast_door as door

    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    assert vars(parser.parse_args([]))["obstacle_opt"] is None
    got = parser.parse_args(["--obstacle-opt", "1", "--obstacle-file", "b.nc", "--obstacle-wall-z0", "0.1"])
    assert got.obstacle_opt == 1 and got.obstacle_file == Path("b.nc") and got.obstacle_wall_z0 == 0.1
    request = SimpleNamespace(nudging=None, obs_nudging=None, tc_climate=None, w_damping=None,
                              inflow_perturbation=None, obstacles=None)
    assert door._partitioned_samehost_refusal(request) is None
    request.obstacles = {"obstacle_opt": 1}
    assert "--obstacle-opt" in door._partitioned_samehost_refusal(request)


# --------------------------------------------------------------------------- device


def _cp():
    return pytest.importorskip("cupy")


def _replay(face, a, b, dt, speed, momentum, tend):
    f32 = np.float32
    out = tend.copy().ravel()
    rate = (a / f32(dt)).astype(f32) + (b * speed).astype(f32)
    keff = (rate / (f32(1) + (rate * f32(dt)).astype(f32))).astype(f32)
    hit = rate > 0
    out[face[hit]] = (out[face[hit]] - (keff[hit] * momentum.ravel()[face[hit]]).astype(f32)).astype(f32)
    return out.reshape(tend.shape)


def test_the_kernels_match_a_float32_replay():
    cp = _cp()
    from woof.hex.cuda_backend import KernelCache
    from woof.hex.obstacles import HexObstacles

    f32 = np.float32
    rng = np.random.default_rng(1)
    mesh = _mesh(level=3)
    host = _host(mesh)
    nlev, ne, nc = 10, host.n_edges, host.n_cells
    mx = host.edges_on_cell.shape[1]
    coe = host.cells_on_edge.astype(np.int32)
    used = np.arange(mx)[None] < host.n_edges_on_cell[:, None]
    slot = np.where(used, host.edges_on_cell, 0).astype(np.int32)
    weight = np.where(used, 0.5 * host.dc_edge[slot] * host.dv_edge[slot] / host.area_cell[:, None], 0).astype(f32)
    eface = np.sort(rng.choice(nlev * ne, 900, replace=False)).astype(np.int32)
    wface = np.sort(rng.choice(np.arange(nc, nlev * nc), 300, replace=False)).astype(np.int32)
    ea, eb = (rng.uniform(0, 10, eface.size) * (rng.random(eface.size) < 0.5)).astype(f32), \
        rng.uniform(0, 0.5, eface.size).astype(f32)
    wa, wb = (rng.uniform(0, 10, wface.size) * (rng.random(wface.size) < 0.5)).astype(f32), \
        rng.uniform(0, 0.5, wface.size).astype(f32)
    u, v = (rng.normal(0, 4, (nlev, ne)).astype(f32) for _ in range(2))
    w = rng.normal(0, 1, (nlev + 1, nc)).astype(f32)
    ru, rw = (u * f32(1.2)).astype(f32), (w * f32(1.2)).astype(f32)
    tu, tw = rng.normal(0, 1, (nlev, ne)).astype(f32), rng.normal(0, 1, (nlev + 1, nc)).astype(f32)
    dt = 0.0333
    obstacles = HexObstacles(
        edges=tuple(cp.asarray(x) for x in (eface, ea, eb)),
        wfaces=tuple(cp.asarray(x) for x in (wface, wa, wb)),
        cells_on_edge=cp.asarray(coe), edges_on_cell=cp.asarray(slot),
        n_edges_on_cell=cp.asarray(host.n_edges_on_cell.astype(np.int32)),
        ke_weight=cp.asarray(weight), shape=(nlev, ne, nc, mx), cache=KernelCache())
    dtu, dtw = cp.asarray(tu), cp.asarray(tw)
    obstacles.apply_edges(dtu, cp.asarray(ru), cp.asarray(u), cp.asarray(v), cp.asarray(w), dt=dt)
    obstacles.apply_wfaces(dtw, cp.asarray(rw), cp.asarray(u), cp.asarray(w), dt=dt)
    assert obstacles.launches == 2

    k, e = eface // ne, eface % ne
    c1, c2 = coe[e, 0], coe[e, 1]
    ww = (f32(0.25) * ((w[k, c1] + w[k, c2]).astype(f32) + (w[k + 1, c1] + w[k + 1, c2]).astype(f32))).astype(f32)
    uu, vv = u.ravel()[eface], v.ravel()[eface]
    speed = np.sqrt(((uu * uu).astype(f32) + (vv * vv).astype(f32)).astype(f32) + (ww * ww).astype(f32))
    np.testing.assert_array_equal(cp.asnumpy(dtu), _replay(eface, ea, eb, dt, speed.astype(f32), ru, tu))

    k, c = wface // nc, wface % nc
    below = np.zeros(wface.size, f32)
    above = np.zeros(wface.size, f32)
    for s in range(mx):
        live = s < host.n_edges_on_cell[c]
        edge = slot[c, s]
        wt = weight[c, s]
        ul, uh = u[k - 1, edge], u[k, edge]
        below = np.where(live, (below + (wt * (ul * ul).astype(f32)).astype(f32)).astype(f32), below)
        above = np.where(live, (above + (wt * (uh * uh).astype(f32)).astype(f32)).astype(f32), above)
    horizontal = (f32(0.5) * (below + above).astype(f32)).astype(f32)
    wv = w.ravel()[wface]
    speed = np.sqrt((horizontal + (wv * wv).astype(f32)).astype(f32)).astype(f32)
    np.testing.assert_array_equal(cp.asnumpy(dtw), _replay(wface, wa, wb, dt, speed, rw, tw))


def _run(morph_kind, steps=6):
    from woof.verify.cases import hex_obstacles as case

    mesh, driver, _ = case.build(2, 1.0, nz=10, ztop=50.0, dt=0.2, sub_steps=4)
    receipt = None
    if morph_kind is not None:
        from woof.hex import obstacles

        morph = (case.canyon_morphology(mesh, lat_extent=-1.0) if morph_kind == "empty"
                 else _column(mesh, 7, 20.0))
        receipt = obstacles.attach(driver, obstacles.ObstacleSettings(), morph)
    for _ in range(steps):
        driver.atmosphere = driver.step_device().atmosphere
    return case._state_sha256(driver), driver, receipt


def test_off_is_byte_identical_and_never_loads_the_module():
    """Off is the default driver: no import, no attribute, no launch.  An
    attached product with no building is the same bytes (nothing attaches),
    and a building changes them, twice the same."""
    _cp()
    sys.modules.pop("woof.hex.obstacles", None)
    default, driver, _ = _run(None)
    assert "woof.hex.obstacles" not in sys.modules
    assert driver.obstacles is None
    empty, driver, receipt = _run("empty")
    assert receipt["faces"] == [0, 0] and driver.obstacles is None
    assert empty == default
    built, driver, receipt = _run("column")
    again, _, _ = _run("column")
    assert receipt["solid_cells"] == 4 and driver.obstacles.launches > 0
    assert built != default and built == again


@pytest.mark.parametrize(*_numerics_child.MODES)
def test_identity_holds_under_each_numerics(numerics, exact):
    _cp()
    _numerics_child.run(__file__, ["test_off_is_byte_identical_and_never_loads_the_module",
                                   "test_the_kernels_match_a_float32_replay"], numerics, exact)
