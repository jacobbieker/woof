"""WRF's wind-farm scheme on hex's unstructured columns (woof.hex.windfarm).

* Placement: the nearest cell centre (MPAS's rule for a point), ties to the
  lowest column, refused outside the mesh and in boundary columns; a cell's
  turbines grouped in table order exactly as on a WRF grid.
* The seam: ``windfarm_opt = 1`` on the engine's generic column seam adds
  ``fitch_column_cells`` into the PBL slot's A-grid rates and MYNN's QKE, word
  for word what the launcher adds to the same rates, and writes POWER.
* Momentum: the cell-centre drag hex projects to edges (MPAS's
  ``tend_u_physics`` form, ``project_edges_v841_f32``) carries the cells'
  total momentum, exactly on a regular hexagonal mesh.
* Off: no key, no placement, no observer, no export -- the seam is built
  with exactly the keywords it was built with before.
* Refused by name: the MPAS suite route, a non-MYNN PBL, ``windfarm_ij = 1``.
"""
from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from tests.conftest import requires_gpu

F = np.float32
CURVE = "V126/3450"


def _lattice(n=9, dx=3000.0, lat0=54.0, lon0=3.0):
    """An n x n patch of cell centres ``dx`` apart (a hex-like row offset)."""
    lats, lons = [], []
    for j in range(n):
        for i in range(n):
            y = j * dx * math.sqrt(3) / 2
            x = (i + 0.5 * (j % 2)) * dx
            lats.append(lat0 + y / 111195.0)
            lons.append(lon0 + x / (111195.0 * math.cos(math.radians(lat0))))
    return np.array(lats), np.array(lons)


def _placement(n=9, dx=3000.0, interior=None, **kw):
    from woof.windfarm.cells import CellPlacement

    lat, lon = _lattice(n, dx, **kw)
    area = np.full(lat.size, dx * dx * math.sqrt(3) / 2)
    if interior is None:
        interior = np.ones(lat.size, bool)
    return CellPlacement(latitude_deg=lat, longitude_deg=lon, area_m2=area,
                         interior=interior, nominal_dx_m=dx)


def _farm_rows(placement, cells, per_cell=1, curve=CURVE):
    rows = []
    for n, c in enumerate(cells):
        for m in range(per_cell):
            rows.append({"turbine_id": f"t{n}_{m}",
                         "lat": float(placement.latitude_deg[c]) + 1e-4 * m,
                         "lon": float(placement.longitude_deg[c]),
                         "hub_height_m": 117.0, "curve": curve})
    return rows


def _write(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# placement


def test_the_tree_agrees_with_brute_force_on_random_points():
    from woof.windfarm.cells import _unit_vectors, nearest_cells

    p = _placement(15, 2000.0)
    rng = np.random.default_rng(9)
    lat = rng.uniform(p.latitude_deg.min(), p.latitude_deg.max(), 400)
    lon = rng.uniform(p.longitude_deg.min(), p.longitude_deg.max(), 400)
    cols, _ = nearest_cells(p, lat, lon)
    brute = np.argmax(_unit_vectors(lat, lon) @ p._xyz.T, axis=1)
    assert np.array_equal(cols, brute)


def test_a_point_goes_to_the_nearest_cell_centre_and_ties_to_the_lowest():
    from woof.windfarm.cells import nearest_cells

    p = _placement()
    cols, dist = nearest_cells(p, p.latitude_deg[[5, 40]] + 1e-5,
                               p.longitude_deg[[5, 40]])
    assert list(cols) == [5, 40] and (dist < 5.0).all()
    # A placement with a duplicated centre: the tie goes to the lower column.
    from woof.windfarm.cells import CellPlacement
    lat = np.append(p.latitude_deg, p.latitude_deg[7])
    lon = np.append(p.longitude_deg, p.longitude_deg[7])
    twin = CellPlacement(latitude_deg=lat, longitude_deg=lon,
                         area_m2=np.append(p.area_m2, p.area_m2[7]),
                         interior=np.ones(lat.size, bool))
    assert nearest_cells(twin, [lat[7]], [lon[7]])[0][0] == 7


def test_outside_and_boundary_turbines_are_refused_by_name():
    from woof.windfarm.cells import CellPlacementRefused, place_on_cells

    p = _placement(interior=np.arange(81) != 3)
    with pytest.raises(CellPlacementRefused, match="outside this mesh: far"):
        place_on_cells(p, [10.0], [10.0], ["far"])
    with pytest.raises(CellPlacementRefused, match="boundary columns: edge"):
        place_on_cells(p, [p.latitude_deg[3]], [p.longitude_deg[3]], ["edge"])


def test_cell_layout_groups_a_cell_in_table_order(tmp_path):
    from woof.windfarm.layout import build_cell_layout, read_turbines

    p = _placement()
    rows = _farm_rows(p, [10, 10, 20], per_cell=1)
    rows[1]["curve"] = "V117/3450"          # A, B in one cell
    rows.insert(2, dict(rows[0], turbine_id="t0_again"))  # then A again
    layout = build_cell_layout(read_turbines(_write(tmp_path / "f.csv", rows)),
                               p)
    assert (layout.ny, layout.nx) == (1, 81) and layout.projection is None
    assert list(layout.turbine_i) == [10, 10, 10, 20]
    assert list(layout.group_cell) == [10, 10, 10, 20]   # A, B, A, A
    assert list(layout.group_turbines) == [1, 1, 1, 1]
    kernel = layout.kernel_groups()
    assert list(kernel["cell"]) == [10, 10, 10, 20]


def test_the_coupling_places_tables_and_wrf_lists_and_refuses_ij(tmp_path):
    from woof.core.windfarm_coupling import (WindFarmRefused,
                                             fitch_layout_for_file)

    p = _placement()
    table = _write(tmp_path / "farm.csv", _farm_rows(p, [12, 30], per_cell=2))
    layout = fitch_layout_for_file(table, shape=p.shape, grid=p)
    assert layout.shape == (1, 81)
    assert list(layout.cell) == [12, 30] and list(layout.count) == [2, 2]
    with pytest.raises(WindFarmRefused, match="no i/j"):
        fitch_layout_for_file(table, ij=1, shape=p.shape, grid=p)
    # WRF's own list: lat lon type, with wind-turbine-1.tbl beside it.
    (tmp_path / "wind-turbine-1.tbl").write_text(
        "3\n100. 90. 0.1 2.0\n4. 0.8 100.\n12. 0.7 2000.\n25. 0.1 2000.\n")
    wrf = tmp_path / "windturbines.txt"
    wrf.write_text(f"{p.latitude_deg[40]:.6f} {p.longitude_deg[40]:.6f} 1\n"
                   f"{p.latitude_deg[40]:.6f} {p.longitude_deg[40]:.6f} 1\n")
    layout = fitch_layout_for_file(wrf, shape=p.shape, grid=p)
    assert list(layout.cell) == [40] and list(layout.count) == [2]
    with pytest.raises(WindFarmRefused, match="no i/j"):
        fitch_layout_for_file(wrf, ij=1, shape=p.shape, grid=p)


def test_interior_columns_masks_the_boundary_and_the_garbage_column():
    from types import SimpleNamespace
    from woof.hex.windfarm import interior_columns

    mesh = SimpleNamespace(latCell=np.zeros(5),
                           bdyMaskCell=np.array([0, 0, 1, 7, 0]))
    assert list(interior_columns(mesh, is_regional=True)) == [
        True, True, False, False, True, False]
    assert interior_columns(mesh, is_regional=False).all()


# ---------------------------------------------------------------------------
# configuration


def test_the_keys_reach_the_generic_seam_and_ysu_is_refused_by_name():
    from woof.hex.physics_suites import (PhysicsSelectionRefusal,
                                         parse_override, resolve_physics)

    keys = dict(parse_override(t) for t in (
        "windfarm_opt=1", "windfarm_turbine_file=/x/farm.csv",
        "windfarm_tke_factor=0.25"))
    resolved = resolve_physics("rrfs_v2_dev", keys)
    assert resolved.seam == "engine"
    engine = dict(resolved.engine_physics)
    assert engine["windfarm_opt"] == 1 and engine["bl_pbl_physics"] == 5
    assert engine["windfarm_turbine_file"] == "/x/farm.csv"
    with pytest.raises(PhysicsSelectionRefusal, match="needs MYNN"):
        resolve_physics(None, keys)          # the default suite runs YSU


def test_the_mpas_route_refuses_the_keys_by_name():
    from woof.hex.windfarm import mpas_route_refusal

    assert mpas_route_refusal(["config_pbl_scheme"]) is None
    text = mpas_route_refusal(["WINDFARM_OPT", "config_dt"])
    assert "windfarm_opt" in text and "no wind-farm scheme" in text


def test_the_constructor_carries_the_interior_mask_only_with_the_farm_on():
    from woof.hex import cuda_suite_physics_v841 as adapter

    rows = [r for r in adapter.load_physics_suites()]
    assert rows      # the table loads; the mapping rule is below
    values = {"n_columns": 4, "xice_threshold": 0.5}
    suite = {"selection": {"land_surface_scheme": "ruc"},
             "physics": {"bl_pbl_physics": 5}, "fractional_seaice": 1}
    original = adapter._suite_for_row
    try:
        adapter._suite_for_row = lambda name: suite
        off = adapter.constructor_values_for(
            values, row_name="x", orographic_statistics=None,
            cell_area_m2=np.ones(4), interior_column=np.zeros(4, bool))
        assert "interior_column" not in off
        suite["physics"] = {"bl_pbl_physics": 5, "windfarm_opt": 1}
        with pytest.raises(ValueError, match="needs interior_column"):
            adapter.constructor_values_for(
                values, row_name="x", orographic_statistics=None,
                cell_area_m2=np.ones(4))
        on = adapter.constructor_values_for(
            values, row_name="x", orographic_statistics=None,
            cell_area_m2=np.ones(4),
            interior_column=np.array([1, 1, 0, 0], bool))
        assert on["interior_column"].dtype == np.bool_
        assert list(on["interior_column"]) == [True, True, False, False]
    finally:
        adapter._suite_for_row = original


# ---------------------------------------------------------------------------
# momentum through the edge projection


def _regular_hex_mesh(n=12, d=1.0):
    """A doubly periodic regular hexagonal Voronoi mesh: centres on a
    triangular lattice, each edge between neighbours, its normal from the
    first cell to the second, length d/sqrt(3), centre spacing d."""
    centres = {}
    for j in range(n):
        for i in range(n):
            centres[(i, j)] = len(centres)
    edges = []
    for (i, j), c in centres.items():
        for di, dj in ((1, 0), (0, 1), (-1, 1)):
            other = centres[((i + di) % n, (j + dj) % n)]
            # lattice vectors a1 = (1, 0), a2 = (1/2, sqrt(3)/2)
            vec = np.array([di + 0.5 * dj, dj * math.sqrt(3) / 2])
            edges.append((c, other, vec / np.linalg.norm(vec)))
    return len(centres), edges, d / math.sqrt(3), d


def _project(du, dv, edges):
    """hex's project_edges_v841_f32 (rho_edge = 1) in binary64: the normal
    component of the mean of the two cells' vectors (east = x, north = y)."""
    out = np.empty((du.shape[0], len(edges)))
    for e, (c1, c2, normal) in enumerate(edges):
        out[:, e] = (0.5 * du[:, c1] * normal[0] + 0.5 * dv[:, c1] * normal[1]
                     + 0.5 * du[:, c2] * normal[0] + 0.5 * dv[:, c2] * normal[1])
    return out


def test_the_edge_projection_conserves_the_farm_momentum_on_a_regular_mesh():
    """sum_e (l_e d_e) F_e n_e == sum_c A_c F_c for any cell field F on a
    regular hexagonal mesh (Perot's identity: sum over a cell's edges of
    (l_e d_e / 2) n_e n_e^T = A_c I), so the momentum the turbines remove at
    cell centres is the momentum the edge tendencies remove."""
    ncell, edges, length, d = _regular_hex_mesh()
    area = math.sqrt(3) / 2 * d * d
    rng = np.random.default_rng(5)
    du = np.zeros((3, ncell))
    dv = np.zeros((3, ncell))
    farm = rng.choice(ncell, 25, replace=False)
    du[:, farm] = -rng.uniform(1e-3, 5e-3, (3, farm.size))
    dv[:, farm] = rng.uniform(-1e-3, 1e-3, (3, farm.size))
    edge = _project(du, dv, edges)
    weight = length * d
    normals = np.array([n for _, _, n in edges])
    via_edges = np.stack([(edge * weight * normals[:, 0]).sum(axis=1),
                          (edge * weight * normals[:, 1]).sum(axis=1)])
    via_cells = np.stack([du.sum(axis=1) * area, dv.sum(axis=1) * area])
    assert np.allclose(via_edges, via_cells, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# the seam (GPU)


def _seam(tmp_path, *, farm: bool, ncol=8):
    from tests import mpas_column_physics_cases as cases
    from woof.core import mpas_column_suite as suite

    physics = dict(cases.selections()["pbl.mynn"])
    lat, lon = _lattice(3, 3000.0)
    lat, lon = lat[:ncol], lon[:ncol]
    if farm:
        p = _placement(3)
        rows = _farm_rows(p, [2, 4, 4], per_cell=1)
        path = _write(tmp_path / "farm.csv", rows)
        physics.update(windfarm_opt=1, windfarm_turbine_file=str(path),
                       windfarm_tke_factor=0.25)
    plan = suite.column_physics_plan(
        physics, n_levels=cases.NZ, dt=cases.DT, radiation_seconds=120.0,
        surface_pbl_seconds=cases.DT)
    rng = np.random.default_rng(2)
    area = np.linspace(7.0e6, 8.5e6, ncol)
    seam = suite.run_mpas_column_physics(
        physics=physics, n_levels=cases.NZ, n_columns=ncol, dt=cases.DT,
        radiation_seconds=120.0, surface_pbl_seconds=cases.DT,
        start_time=cases.START, latitude_deg=lat, longitude_deg=lon,
        terrain_height_m=rng.uniform(0.0, 50.0, ncol).astype(F),
        z_interface_nominal_m=np.linspace(0.0, cases.ZTOP, cases.NZ + 1),
        p_top_pa=cases.P_TOP, dx_m=3000.0, cell_area_m2=area,
        surface=cases.surface(plan, ncol), statics=cases.statics(plan, ncol))
    return seam, area


def _phase1(seam):
    import cupy as cp
    from tests import mpas_column_physics_cases as cases

    fields, species = cases.profile(cp, seam.plan, seam._ncol)
    # The case's 30 levels are 667 m deep, so the hub sits below the first
    # mass level and its wind is scaled down by hub/z1 (WRF's arm): a wind
    # strong enough to turn the rotors there.
    fields["u"] = fields["u"] * cp.float32(4.0)
    fields["v"] = fields["v"] * cp.float32(4.0)
    out = seam.run_phase1(
        dt=cases.DT, u=fields["u"], v=fields["v"], theta=fields["theta"],
        pressure=fields["pressure"],
        pressure_interface=fields["pressure_interface"],
        z_interface=fields["z_interface"], w=fields["w"],
        rho_dry=fields["rho_dry"], species=species)
    return out, fields


@requires_gpu
@pytest.mark.gpu
def test_the_seam_adds_exactly_the_cells_kernel_into_mynn_rates(tmp_path):
    import cupy as cp
    from woof.core.windfarm_fitch import cell_wfdensity, launch_fitch

    off, _ = _seam(tmp_path, farm=False)
    on, area = _seam(tmp_path, farm=True)
    assert getattr(off._driver, "windfarm", None) is None
    assert "power" not in off._driver.fields
    assert off.windfarm_placement is None and off.phase2_observers == ()
    qke_before = on._driver.fields["qke"].copy()
    r_off, fields = _phase1(off)
    r_on, _ = _phase1(on)
    nz, ncol = r_off.du.shape
    # The drag the launcher adds to the MYNN-only rates, on the same columns.
    du = r_off.du.reshape(nz, 1, ncol).copy()
    dv = r_off.dv.reshape(nz, 1, ncol).copy()
    qke = off._driver.fields["qke"].copy()
    power = launch_fitch(
        fields["u"].reshape(nz, 1, ncol), fields["v"].reshape(nz, 1, ncol),
        fields["z_interface"].reshape(nz + 1, 1, ncol),
        on._driver.windfarm.layout, du, dv,
        wfdensity=cp.asarray(cell_wfdensity(area)), qke=qke, dt=60.0,
        tke_factor=0.25)
    for got, want in ((r_on.du, du), (r_on.dv, dv),
                      (on._driver.fields["qke"], qke),
                      (on._driver.fields["power"], power)):
        assert np.array_equal(cp.asnumpy(got).reshape(-1).view(np.uint32),
                              cp.asnumpy(want).reshape(-1).view(np.uint32))
    p = cp.asnumpy(on._driver.fields["power"]).reshape(-1)
    assert p[2] > 0 and p[4] > p[2] * 1.5 and p[[0, 1, 3]].sum() == 0
    assert bool(cp.any(on._driver.fields["qke"] != qke_before))


@requires_gpu
@pytest.mark.gpu
def test_the_seam_refuses_a_farm_without_cell_areas(tmp_path):
    from tests import mpas_column_physics_cases as cases
    from woof.core import mpas_column_suite as suite

    p = _placement(3)
    physics = dict(cases.selections()["pbl.mynn"], windfarm_opt=1,
                   windfarm_turbine_file=str(_write(tmp_path / "f.csv",
                                                    _farm_rows(p, [2]))))
    plan = suite.column_physics_plan(
        physics, n_levels=cases.NZ, dt=cases.DT, radiation_seconds=120.0,
        surface_pbl_seconds=cases.DT)
    lat, lon = _lattice(3, 3000.0)
    with pytest.raises(ValueError, match="needs cell_area_m2"):
        suite.run_mpas_column_physics(
            physics=physics, n_levels=cases.NZ, n_columns=9, dt=cases.DT,
            radiation_seconds=120.0, surface_pbl_seconds=cases.DT,
            start_time=cases.START, latitude_deg=lat, longitude_deg=lon,
            terrain_height_m=np.zeros(9, F),
            z_interface_nominal_m=np.linspace(0.0, cases.ZTOP, cases.NZ + 1),
            p_top_pa=cases.P_TOP, dx_m=3000.0,
            surface=cases.surface(plan, 9), statics=cases.statics(plan, 9))


# ---------------------------------------------------------------------------
# the per-turbine power window (GPU)


def _fake_seam(p, nz=20, seed=4):
    import cupy as cp
    from types import SimpleNamespace

    rng = np.random.default_rng(seed)
    n = p.n_columns
    dz = rng.uniform(30.0, 80.0, (nz, n))
    zw = np.concatenate([np.zeros((1, n)), np.cumsum(dz, axis=0)]) + \
        rng.uniform(0.0, 20.0, n)
    f = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=F))  # noqa: E731
    state = SimpleNamespace(
        php=f((zw * 9.81).reshape(nz + 1, 1, n)),
        thp=f(rng.uniform(285, 300, (nz, 1, n))),
        p=f(rng.uniform(8.5e4, 1.01e5, (nz, 1, n))),
        qv=f(rng.uniform(1e-3, 9e-3, (nz, 1, n))),
        ht=f(zw[0].reshape(1, n)))
    seam = SimpleNamespace(
        _nz=nz, _ncol=n, _dt=20.0, _state=state,
        _in={"u": f(rng.normal(9, 3, (nz, 1, n))),
             "v": f(rng.normal(-1, 3, (nz, 1, n)))})
    return seam


@requires_gpu
@pytest.mark.gpu
def test_the_power_window_reads_each_turbine_column_word_for_word(tmp_path):
    """The pseudo-grid gather hands the WRF-like core's kernel exactly the
    turbine's own column: its evaluation equals the host road's on the
    seam's columns, every word, and the window folds and restarts."""
    import cupy as cp
    from woof.core.windfarm_power_diag import G, evaluate_turbines
    from woof.windfarm.layout import build_cell_layout, read_turbines
    from woof.hex.windfarm import HexWindfarmPower

    p = _placement()
    rows = _farm_rows(p, [10, 10, 40, 77], per_cell=2)
    layout = build_cell_layout(read_turbines(_write(tmp_path / "f.csv", rows)),
                               p)
    power = HexWindfarmPower(layout, n_columns=p.n_columns)
    power.bind_cells(p)
    seam = _fake_seam(p)
    pseudo = power._gather(seam)
    got = power.diagnostic.evaluate(pseudo)
    nz, cols = seam._nz, power.columns
    host = {k: cp.asnumpy(v) for k, v in vars(seam._state).items()}
    zw = (host["php"].reshape(nz + 1, -1)[:, cols] + F(0.0)) / F(G)
    z = F(0.5) * (zw[:-1] + zw[1:]) - host["ht"].reshape(-1)[cols]
    columns = {"z": z,
               "u": cp.asnumpy(seam._in["u"]).reshape(nz, -1)[:, cols],
               "v": cp.asnumpy(seam._in["v"]).reshape(nz, -1)[:, cols],
               "theta": host["thp"].reshape(nz, -1)[:, cols],
               "p": host["p"].reshape(nz, -1)[:, cols],
               "qv": host["qv"].reshape(nz, -1)[:, cols]}
    d = power.diagnostic
    want = evaluate_turbines(
        columns, d.targets_m, d.weights, curves=d.distinct_curves,
        curve_index=d.curve_index, rated_w=d.rated_w,
        half_rho_area=d.half_rho_area, cut_out_ms=d.cut_out_ms)
    for key in ("rews", "rho", "v_eq", "power", "ct"):
        assert np.array_equal(np.asarray(got[key]).view(np.uint64),
                              np.asarray(want[key]).view(np.uint64)), key
    assert (want["power"] > 0).all()
    # The window: two folds, then a take restarts it.
    power.observe(seam)
    power.commit()
    power.observe(seam)
    power.discard()                  # an aborted step never enters
    power.observe(seam)
    power.commit()
    window = power.take_window()
    assert window["steps"] == 2 and window["seconds"] == 40.0
    assert np.allclose(window["sums"]["power"], 40.0 * want["power"])
    assert power.take_window() == {"steps": 0, "seconds": 0.0}
    # The sidecar names each turbine's own column.
    path = power.write_frame(want, window, valid_time="2020-07-10_01:00:00",
                             output_dir=tmp_path)
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        assert list(ds["grid_i"][:]) == [10] * 4 + [40] * 2 + [77] * 2
        assert ds["POWER_FARM"][...] > 0
