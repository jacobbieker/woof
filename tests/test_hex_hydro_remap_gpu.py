"""The hex cell <-> LSM grid exchange on the device (woof.hex.hydro_remap_gpu), the default route on a GPU.

The device operator keeps the overlap weights resident as CSR in both directions and sums each output in the
host scipy's order with the host's own contraction (hydro_remap_gpu.host_fuses), so its words are the host
route's.  Checked here, on a uniform spherical Voronoi mesh and a variable-resolution one (refined around the
grid), with WRF-like soil and runoff values:

* the forward mean, the increment return, the back-map and the category majority are the host's, byte for
  byte (0 ULP), under strict numerics and, in a child process, fast numerics;
* conservation: sum_c A_c q_c = sum_g A_g q_g for a returned field, to 1e-13 relative (binary64 rounding of
  the two area-weighted sums);
* the coupled seam (with the WRF-Hydro opt-in) attached by default runs on the device and gives the host
  route's soil, head and routing state byte for byte over four cycles, and its device-side budget closes as
  the host's does; run this file with GPUWM_NUMERICS=fast for the fast words;
* the switch: on unless turned off (``--no-wrf-hydro-gpu-exchange``, ``[wrf_hydro] gpu_exchange = false``,
  ``WOOF_WRF_HYDRO_GPU_EXCHANGE=0``), refused without ``--wrf-hydro``; with no GPU backend, or a host
  scipy that mixes loops, the host route runs and the log says why.
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

from woof import optin

RADIUS = 6371229.0
REASON = optin.unavailable_reason("wrf_hydro")


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


# -- the switch ---------------------------------------------------------------------------------------------
@pytest.fixture
def clean_env(monkeypatch):
    from woof import wrf_hydro_route as route
    for key in (route.ENV_NAMELIST, route.ENV_OUTPUT, route.ENV_GPU_EXCHANGE):
        monkeypatch.delenv(key, raising=False)
    return route


def test_the_gpu_exchange_is_on_unless_turned_off(clean_env, tmp_path, monkeypatch):
    route = clean_env
    assert route.gpu_exchange() is True
    with route.launch_scope(namelist=tmp_path / "hydro.namelist"):
        assert route.configured() and route.gpu_exchange() is True
    with route.launch_scope(namelist=tmp_path / "hydro.namelist", gpu_exchange=False):
        assert route.gpu_exchange() is False
    assert route.gpu_exchange() is True and route.configured() is None
    toml = tmp_path / "case.toml"
    toml.write_text('[wrf_hydro]\nnamelist = "hydro.namelist"\ngpu_exchange = false\n')
    with route.launch_scope(config=toml):
        assert route.gpu_exchange() is False
    with route.launch_scope(config=toml, gpu_exchange=True):          # the flag overrides the table
        assert route.gpu_exchange() is True
    assert route.gpu_exchange() is True
    monkeypatch.setenv(route.ENV_GPU_EXCHANGE, "0")
    assert route.gpu_exchange() is False
    monkeypatch.delenv(route.ENV_GPU_EXCHANGE)
    for asked in (True, False):
        with pytest.raises(ValueError, match="without --wrf-hydro"):
            with route.launch_scope(gpu_exchange=asked):
                pass
    toml.write_text('[wrf_hydro]\nnamelist = "hydro.namelist"\ngpu_exchange = "yes"\n')
    with pytest.raises(ValueError, match="true or false"):
        route.read_table(toml)


def test_the_cli_flag_reaches_the_launch():
    import argparse

    from woof.wrf_hydro_route import add_arguments
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert parser.parse_args([]).wrf_hydro_gpu_exchange is None
    assert parser.parse_args(["--wrf-hydro-gpu-exchange"]).wrf_hydro_gpu_exchange is True
    assert parser.parse_args(["--no-wrf-hydro-gpu-exchange"]).wrf_hydro_gpu_exchange is False


def test_the_host_route_runs_and_says_why_without_a_device(monkeypatch, caplog):
    from woof import backend
    from woof.hex import hydro_remap_gpu, wrf_hydro_hex
    exchange, _ = _exchange(False)
    assert wrf_hydro_hex._device_exchange(exchange, False) == (None, "--no-wrf-hydro-gpu-exchange")
    monkeypatch.setattr(backend, "active", lambda: "cpu")
    device, reason = wrf_hydro_hex._device_exchange(exchange, True)
    assert device is None and "no GPU backend" in reason
    monkeypatch.setattr(backend, "active", lambda: "cuda")

    def mixed(*_):
        raise RuntimeError("this scipy mixes fused and plain loops")
    monkeypatch.setattr(hydro_remap_gpu.DeviceExchange, "__init__", mixed)
    device, reason = wrf_hydro_hex._device_exchange(exchange, True)
    assert device is None and "mixes" in reason


# -- the coupled seam (WRF-Hydro opt-in) -------------------------------------------------------------------
@pytest.mark.skipif(REASON is not None, reason=f"opt-in rule: {REASON}")
def test_the_default_seam_is_the_host_seam(tmp_path):
    cp = pytest.importorskip("cupy")
    import datetime

    import test_hex_wrf_hydro_gpu as h
    from woof import backend
    from woof.hex.hydro_remap import mesh_cells
    from woof.hex.wrf_hydro_hex import DEVICE_BACKENDS, attach
    if backend.active() not in DEVICE_BACKENDS:
        pytest.skip(f"no GPU backend ({backend.active()})")
    namelist = h._domain(tmp_path / "dom")
    cells = mesh_cells(mesh := h._hex_mesh())
    nc = len(cells.area)
    water = np.degrees(mesh.arrays["lonCell"]) > h.LON0 + h.DLON * (h.NX - 1.5)
    start = datetime.datetime(2020, 1, 1)

    def seam(tag, **kw):
        fields = {"smois": cp.zeros((4, 1, nc), cp.float32), "sh2o": cp.zeros((4, 1, nc), cp.float32),
                  "ivgtyp": cp.asarray(np.where(water, 17, 10).reshape(1, nc), dtype=cp.int32),
                  "isltyp": cp.asarray(np.where(water, 14, 6).reshape(1, nc), dtype=cp.int32)}
        driver = SimpleNamespace(fields=fields, bldt_seconds=60.0)
        return attach(driver, cells, namelist=namelist, start_time=start, dzs=h.DZS, lsm=2,
                      output_dir=str(tmp_path / tag), **kw), driver

    (host, hd), (dev, dd) = seam("host", gpu_exchange=False), seam("dev")
    assert host.device is None and dev.device is not None
    rng = np.random.default_rng(3)
    f32 = np.float32
    for cycle in range(4):
        smois = rng.uniform(0.2, 0.4, (4, 1, nc)).astype(f32)
        infx, sold = rng.uniform(0.0, 6.0, (1, nc)).astype(f32), rng.uniform(0.0, 0.2, (1, nc)).astype(f32)
        for s, d in ((host, hd), (dev, dd)):
            d.fields["smois"][...] = cp.asarray(smois)
            d.fields["sh2o"][...] = cp.asarray((smois * f32(0.9)).astype(f32))
            s.planes["infxsrt"][...] = cp.asarray(infx)
            s.planes["soldrain"][...] = cp.asarray(sold)
            s.exchange(d.fields)
        for name in ("smois", "sh2o"):
            assert cp.asnumpy(hd.fields[name]).tobytes() == cp.asnumpy(dd.fields[name]).tobytes(), (cycle, name)
        got, want = dev.digest_arrays(), host.digest_arrays()
        assert set(got) == set(want)
        for name in want:
            assert cp.asnumpy(got[name]).tobytes() == cp.asnumpy(want[name]).tobytes(), (cycle, name)
    assert len(dev.budget) == len(host.budget) == 4
    for b, ref in zip(dev.budget, host.budget):
        total = b["soil_water_cells_m3"]
        assert abs(b["soil_change_cells_exact_m3"] - b["soil_change_grid_m3"]) <= 1e-12 * total
        assert abs(b["runoff_cells_m3"] - b["runoff_grid_exact_m3"]) <= 1e-12 * b["runoff_cells_m3"]
        assert abs(b["head_cells_m3"] - b["head_grid_m3"]) <= 1e-12 * max(abs(b["head_grid_m3"]), 1e-9)
        for key, value in ref.items():
            assert abs(b[key] - value) <= 1e-12 * max(abs(value), total), key
    assert dev.receipt()["exchange_route"]["device"] is True
    assert host.receipt()["exchange_route"] == {"device": False, "reason": "--no-wrf-hydro-gpu-exchange"}
    assert any(b["head_grid_m3"] > 0.0 for b in dev.budget) and abs(dev.budget[-1]["soil_change_grid_m3"]) > 0
