"""The hex cell <-> LSM grid exchange on the device (woof.hex.hydro_remap_gpu), the opt-in route.

The device operator keeps the overlap weights resident as CSR in both directions and sums each output in the
host scipy's order with the host's own contraction (hydro_remap_gpu.host_fuses), so its words are the host
route's.  Checked here, on a uniform spherical Voronoi mesh and a variable-resolution one (refined around the
grid), with WRF-like soil and runoff values:

* the forward mean, the increment return, the back-map and the category majority are the host's, byte for
  byte (0 ULP), under strict numerics and, in a child process, fast numerics;
* conservation: sum_c A_c q_c = sum_g A_g q_g for a returned field, to 1e-13 relative (binary64 rounding of
  the two area-weighted sums).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import SphericalVoronoi

RADIUS = 6371229.0


def _mesh(refined: bool, n=20000, seed=4):
    """A spherical Voronoi mesh: ``n`` uniform points, or (``refined``) those plus a dense cluster over the
    grid, so cells shrink about tenfold towards it (a variable-resolution mesh)."""
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n, 3))
    if refined:
        lat, lon = np.radians(rng.uniform(9.0, 20.5, 6000)), np.radians(rng.uniform(19.0, 33.0, 6000))
        points = np.vstack([points, np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                                              np.sin(lat)], axis=1)])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    sv = SphericalVoronoi(points)
    sv.sort_vertices_of_regions()
    counts = np.array([len(r) for r in sv.regions])
    on_cell = np.full((len(points), counts.max()), -1, np.int64)
    for i, region in enumerate(sv.regions):
        on_cell[i, :len(region)] = region
    lat = lambda xyz: np.arcsin(np.clip(xyz[:, 2], -1, 1))  # noqa: E731
    lon = lambda xyz: np.arctan2(xyz[:, 1], xyz[:, 0])  # noqa: E731
    arrays = {"latCell": lat(points), "lonCell": lon(points), "latVertex": lat(sv.vertices),
              "lonVertex": lon(sv.vertices), "nEdgesOnCell": counts, "verticesOnCell": on_cell}
    return SimpleNamespace(arrays=arrays, n_cells=len(points), attrs={"sphere_radius_m": RADIUS})


def _exchange(refined):
    from woof.hex.hydro_remap import CellGridExchange, grid_cells, mesh_cells
    ny, nx, d = 7, 9, 1.25
    lat, lon = np.meshgrid(10.0 + d * np.arange(ny + 1), 20.0 + d * np.arange(nx + 1), indexing="ij")
    cells = mesh_cells(_mesh(refined))
    return CellGridExchange.build(cells, grid_cells(lat, lon, radius_m=RADIUS), (ny, nx)), cells


def _operator_words(refined: bool) -> dict:
    """Every device output against the host's (bytes); returns the conservation residuals."""
    import cupy as cp
    from woof.hex.hydro_remap_gpu import DeviceExchange
    ex, cells = _exchange(refined)
    dev = DeviceExchange(ex)
    nc, ng, f32 = dev.cells, dev.grid, np.float32
    rng = np.random.default_rng(11)
    columns = nc + 1                                          # a regional run's trailing garbage column
    soil = rng.uniform(0.05, 0.45, (4, columns)).astype(f32)
    soil[:, -1] = np.nan
    # forward
    exact, out = cp.full((4, ng), np.nan, cp.float64), cp.full((4, ng), np.nan, cp.float32)  # poisoned
    dev.to_grid(cp.asarray(soil), fields=4, stride=columns, exact=exact, out=out)
    want = ex.to_grid(soil[:, :nc].astype(np.float64)).reshape(4, ng)
    assert cp.asnumpy(exact).tobytes() == want.tobytes(), "to_grid binary64"
    assert cp.asnumpy(out).tobytes() == want.astype(f32).tobytes(), "to_grid binary32"
    # increment: the routing's change on the grid returned to the cells, the state updated in place
    before = want.astype(f32).reshape(4, *ex.shape)
    after = (before + rng.uniform(-0.02, 0.03, before.shape).astype(f32)).astype(f32)
    state, prior, inc_exact = cp.asarray(soil), cp.full((4, nc), np.nan, cp.float32), cp.full((4, nc), np.nan,
                                                                                                 cp.float64)
    dev.increment(cp.asarray(before), cp.asarray(after), state, fields=4, stride=columns, prior=prior,
                  exact=inc_exact)
    host_exact = soil[:, :nc].astype(np.float64) + ex.increment(before, after)
    assert cp.asnumpy(inc_exact).tobytes() == host_exact.tobytes(), "increment binary64"
    got = cp.asnumpy(state)
    assert got[:, :nc].tobytes() == host_exact.astype(f32).tobytes(), "increment binary32"
    assert np.isnan(got[:, -1]).all() and cp.asnumpy(prior).tobytes() == soil[:, :nc].tobytes()
    # back-map (the ponded head), with ponding on some grid cells only
    head = np.where(rng.uniform(size=ex.shape) < 0.4, rng.uniform(0.0, 30.0, ex.shape), 0.0).astype(f32)
    back64, plane = cp.full(nc, np.nan, cp.float64), cp.full((1, columns), 7.0, cp.float32)
    dev.to_cells(cp.asarray(head), exact=back64, out=plane)
    host_back = ex.to_cells(head.astype(np.float64))
    assert cp.asnumpy(back64).tobytes() == host_back.tobytes(), "to_cells binary64"
    assert cp.asnumpy(plane)[0, :nc].tobytes() == host_back.astype(f32).tobytes()
    assert float(plane[0, -1]) == 7.0
    # majority: two classes split along a latitude, a patchwork and one class, as the host picks them
    lat = np.degrees(np.arcsin(cells.centers[:, 2]))
    for cats in (np.where(lat > 14.0, 7, 2), rng.integers(1, 21, nc), np.full(nc, 5)):
        got = cp.asnumpy(dev.dominant(cp.asarray(cats.astype(np.int32))))
        assert np.array_equal(got, ex.dominant(cats)), "dominant"
    # conservation of what returns to the cells
    residual = abs(float(ex.integral_cells(host_back[None])[0]) - float(ex.integral_grid(head[None].astype(
        np.float64))[0])) / float(ex.integral_grid(head[None].astype(np.float64))[0])
    increment = cp.asnumpy(inc_exact) - soil[:, :nc]
    change = ex.integral_grid(after.astype(np.float64) - before)
    inc_residual = float(np.max(np.abs(ex.integral_cells(increment) - change) / np.abs(change)))
    return {"head": residual, "increment": inc_residual}


@pytest.mark.parametrize("refined", [False, True], ids=["uniform", "variable"])
def test_the_device_operator_is_the_host_operator_word_for_word(refined):
    pytest.importorskip("cupy")
    residual = _operator_words(refined)
    assert residual["head"] <= 1e-13 and residual["increment"] <= 1e-12, residual


def test_the_words_hold_under_fast_numerics():
    """tests/conftest.py pins strict numerics; production runs fast.  On Metal the fast translation computes
    binary64 through df64 (woof.backend.metal.loader.f64_mode), a different code path from strict's soft-float."""
    pytest.importorskip("cupy")
    env = dict(os.environ, GPUWM_NUMERICS="fast", WOOF_HEX_NUMERICS="fast", GPUWM_WRF_EXACT="0",
               PYTHONPATH=os.pathsep.join(filter(None, (str(Path(__file__).parents[1]), os.environ.get(
                   "PYTHONPATH")))))
    code = ("import test_hex_hydro_remap_gpu as t; from woof import numerics; assert numerics.mode() == 'fast';"
            "[t._operator_words(r) for r in (False, True)]; print('ok')")
    run = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent, env=env, capture_output=True,
                         text=True, timeout=900)
    assert run.returncode == 0 and run.stdout.strip().endswith("ok"), run.stderr[-4000:]


def test_the_host_probe_names_one_loop():
    from woof.hex.hydro_remap_gpu import host_fuses
    assert all(isinstance(f, bool) for f in host_fuses())


def test_subnormal_words_are_kept_as_numpy_keeps_them():
    """The binary32 casts keep subnormals as numpy's astype does (no FTZ in the module's compile)."""
    cp = pytest.importorskip("cupy")
    from woof.hex.hydro_remap_gpu import DeviceExchange
    ex, _ = _exchange(True)
    dev = DeviceExchange(ex)
    nc, ng = dev.cells, dev.grid
    tiny = np.random.default_rng(5).uniform(1e-44, 1e-39, (1, nc)).astype(np.float32)
    exact, out = cp.full((1, ng), np.nan, cp.float64), cp.full((1, ng), np.nan, cp.float32)
    dev.to_grid(cp.asarray(tiny), fields=1, stride=nc, exact=exact, out=out)
    want = ex.to_grid(tiny.astype(np.float64)).reshape(1, ng)
    assert np.count_nonzero(want.astype(np.float32)) > 0
    assert cp.asnumpy(exact).tobytes() == want.tobytes()
    assert cp.asnumpy(out).tobytes() == want.astype(np.float32).tobytes()


def test_majority_ties_and_arealess_rows_follow_numpy():
    """Exact ties go to the smaller class; a row whose overlaps hold no area takes the mesh's smallest class,
    as np.argmax over every class does."""
    cp = pytest.importorskip("cupy")
    import scipy.sparse as sp
    from woof.hex.hydro_remap import CellGridExchange
    from woof.hex.hydro_remap_gpu import DeviceExchange
    # rows: equal halves of classes 9 and 4; 3 beats 7 by area; zero-area overlaps of classes 8 and 6
    w = np.array([2.5, 2.5, 1.0, 3.0, 0.0, 0.0])
    m = sp.csr_matrix((w, [0, 1, 2, 3, 4, 5], [0, 2, 4, 6]), shape=(3, 6))
    ex = CellGridExchange(m, np.full(6, 5.0), np.asarray(m.sum(axis=1)).ravel(), (1, 3), {})
    cats = np.array([9, 4, 7, 3, 8, 6])
    assert np.array_equal(cp.asnumpy(DeviceExchange(ex).dominant(cp.asarray(cats))), ex.dominant(cats))
    assert ex.dominant(cats).tolist() == [[4, 3, 3]]


def test_mismatched_arguments_are_refused():
    cp = pytest.importorskip("cupy")
    from woof.hex.hydro_remap_gpu import DeviceExchange
    ex, _ = _exchange(True)
    dev = DeviceExchange(ex)
    nc, ng = dev.cells, dev.grid
    exact, out = cp.zeros((1, ng), cp.float64), cp.zeros((1, ng), cp.float32)
    with pytest.raises(ValueError, match="float32"):
        dev.to_grid(cp.zeros((1, nc), cp.float64), fields=1, stride=nc, exact=exact, out=out)
    with pytest.raises(ValueError, match="stride"):
        dev.to_grid(cp.zeros((2, nc), cp.float32), fields=2, stride=nc - 1, exact=exact, out=out)
    with pytest.raises(ValueError, match="one category"):
        dev.dominant(cp.zeros(nc + 1, cp.int32))
    with pytest.raises(ValueError, match="integer"):
        dev.dominant(cp.zeros(nc, cp.float32))
