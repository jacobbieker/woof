"""The opt-in porous immersed obstacles (obstacle_opt = 1, woof.core.obstacles).

Host checks: the configuration refusals, the face coefficients of a resolved
street canyon, and the boundary ring.  GPU checks, each in a child process
under the default (fast) and strict numerics (and run the file again with
``GPUWM_WRF_EXACT=1``): a domain with ``obstacle_opt = 0``, one with
``obstacle_opt = 1`` and no buildings, and one whose face list carries only
zero coefficients (the kernel launches and must write nothing) step to the
same words.  A one-minute canyon checks the sign of the in-canyon vortex;
the full canyon and cube benchmarks are ``slow``.
"""
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import (ConfigurationRefused, OBSTACLE_RUN_DEFAULTS,
                         RunConfig, validate_run_config)


def _les(**overrides):
    values = dict(nx=20, ny=4, nz=20, dx=2.0, dy=2.0, ztop=40.0, dt=0.1,
                  run_seconds=0.0, km_opt=3, bl_pbl_physics=0,
                  sf_sfclay_physics=0, moist=False, mp_physics=0, isfflx=0)
    values.update(overrides)
    return RunConfig(**values)


def test_defaults_are_off():
    cfg = _les()
    for name, default in OBSTACLE_RUN_DEFAULTS.items():
        assert getattr(cfg, name) == default
    validate_run_config(cfg)


@pytest.mark.parametrize("overrides, match", [
    (dict(obstacle_opt=2, obstacle_file="b.nc"), "obstacle_opt must be"),
    (dict(obstacle_opt=True, obstacle_file="b.nc"), "obstacle_opt must be"),
    (dict(obstacle_file="b.nc"), "obstacle_opt = 0"),
    (dict(obstacle_opt=1), "needs obstacle_file"),
    (dict(obstacle_opt=1, obstacle_file="b.nc", dx=1000.0, dy=1000.0),
     "dx <= 500"),
    (dict(obstacle_solid_fraction=0.0), "obstacle_solid_fraction"),
    (dict(obstacle_solid_fraction=1.5), "obstacle_solid_fraction"),
    (dict(obstacle_relax_factor=-1.0), "obstacle_relax_factor"),
    (dict(obstacle_drag_coefficient=float("nan")), "obstacle_drag_coefficient"),
    (dict(obstacle_wall_z0=0.0), "obstacle_wall_z0"),
])
def test_refusals(overrides, match):
    with pytest.raises(ConfigurationRefused, match=match):
        validate_run_config(_les(**overrides))


def test_selected_obstacles_validate():
    validate_run_config(_les(obstacle_opt=1, obstacle_file="b.nc"))


def _canyon_coefficients(ring=0, **cfg_overrides):
    from woof.core.obstacles import build_coefficients
    from woof.verify.cases import obstacle_canyon as oc

    cfg = dataclasses.replace(oc.default_config(), **cfg_overrides)
    morph = oc.morphology(cfg)
    z_w = np.broadcast_to(np.linspace(0.0, cfg.ztop, cfg.nz + 1)[:, None, None],
                          (cfg.nz + 1, cfg.ny, cfg.nx))
    periodic = ring == 0
    return cfg, build_coefficients(morph, z_w, cfg, periodic_x=periodic,
                                   periodic_y=periodic, ring=ring)


def test_canyon_faces():
    """Buildings fill x in [20, 40) m (cells 10..19) up to 20 m (layers
    0..9) of a 20 x 4 x 50 periodic box with 2 m cells."""
    cfg, (fu, fv, fw, receipt) = _canyon_coefficients()
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    assert receipt["solid_cells"] == 10 * ny * 10
    assert receipt["porous_cells"] == 0
    relax = np.float32(cfg.obstacle_relax_factor)
    # u: faces i = 10..20 and the periodic face 0 are blocked below the roof;
    # the faces just above the roof carry the roof's wall stress.
    u_a = np.zeros((nz, ny, nx + 1), np.float32)
    u_a.ravel()[fu[0]] = fu[1]
    u_b = np.zeros_like(u_a)
    u_b.ravel()[fu[0]] = fu[2]
    blocked = np.zeros(nx + 1, bool)
    blocked[10:] = blocked[0] = True
    assert np.all(u_a[:10][..., blocked] == relax)
    assert np.all(u_a[:10][..., ~blocked] == 0)
    assert np.all(u_a[10:] == 0)
    assert np.all(u_b[10][..., blocked] > 0)
    assert np.all(u_b[11:] == 0)
    assert fu[0].size == 12 * ny * 10 + 12 * ny
    # v: the canyon is uniform in y, so v faces are blocked over the
    # building cells only and carry facade stress beside them.
    v_a = np.zeros((nz, ny + 1, nx), np.float32)
    v_a.ravel()[fv[0]] = fv[1]
    assert np.all(v_a[:10, :, 10:] == relax)
    assert np.all(v_a[:10, :, :10] == 0)
    v_b = np.zeros_like(v_a)
    v_b.ravel()[fv[0]] = fv[2]
    assert np.all(v_b[:10, :, (9, 0)] > 0) and np.all(v_b[:10, :, 1:9] == 0)
    # w: the roof face (k = 10) and every face inside is blocked; the
    # surface and the lid (k = 0, nz) never appear.
    w_a = np.zeros((nz + 1, ny, nx), np.float32)
    w_a.ravel()[fw[0]] = fw[1]
    assert np.all(w_a[1:11, :, 10:] == relax)
    assert np.all(w_a[0] == 0) and np.all(w_a[nz] == 0)
    assert np.all(w_a[11:] == 0) and np.all(w_a[:, :, :10] == 0)


def test_boundary_ring_drops_edge_buildings():
    cfg, (fu, fv, fw, receipt) = _canyon_coefficients(ring=12)
    # A 12-cell ring leaves no interior column (nx = 20, ny = 4).
    assert receipt["solid_cells"] == 0
    assert fu[0].size == fv[0].size == fw[0].size == 0


def test_empty_morphology_attaches_nothing(monkeypatch):
    pytest.importorskip("cupy")
    from woof.core import obstacles
    from woof.static.buildings import CartesianGrid, aggregate, block_raster

    cfg = _les(obstacle_opt=1, obstacle_file="unused.nc")
    morph = aggregate([block_raster(cfg.nx, cfg.ny, cfg.dx, cfg.dy, [])],
                      CartesianGrid(cfg.nx, cfg.ny, cfg.dx, cfg.dy))
    z_w = np.broadcast_to(np.linspace(0.0, cfg.ztop, cfg.nz + 1)[:, None, None],
                          (cfg.nz + 1, cfg.ny, cfg.nx))
    fu, fv, fw, receipt = obstacles.build_coefficients(
        morph, z_w, cfg, periodic_x=True, periodic_y=True)
    assert receipt["solid_cells"] == 0
    assert fu[0].size + fv[0].size + fw[0].size == 0


# ---- GPU: the off path is the code that ran before --------------------------

def _run(route, steps=12):
    """Child-process body: one canyon-like LES stepped under ``route``."""
    import cupy as cp

    from woof.core import obstacles
    from woof.core.dycore import run_steps
    from woof.static.buildings import CartesianGrid, aggregate, block_raster
    from woof.verify.cases.obstacle_canyon import build

    cfg = _les(obstacle_opt=0 if route == "off" else 1,
               obstacle_file="" if route == "off" else "unused.nc",
               time_step_sound=40, mix_isotropic=1, c_s=0.18)
    blocks = [] if route == "empty" else [(20.0, 30.0, 0.0, 8.0, 10.0)]
    morph = aggregate([block_raster(cfg.nx, cfg.ny, cfg.dx, cfg.dy, blocks)],
                      CartesianGrid(cfg.nx, cfg.ny, cfg.dx, cfg.dy))
    state = build(cfg, morph if route != "off" else None)
    if route == "empty":
        assert state._obstacles is None
    if route == "zeroed":
        faces = state._obstacles
        assert faces is not None and faces.total > 0
        faces.a[...] = 0
        faces.b[...] = 0
    rng = np.random.default_rng(3)
    for name in ("u", "v", "w"):
        field = getattr(state, name)
        field += cp.asarray(0.5 * rng.standard_normal(field.shape),
                            dtype=cp.float32)
    run_steps(state, cfg, n=steps)
    words = {}
    for name in ("u", "v", "w", "t", "ph", "mu", "p"):
        value = getattr(state, name, None)
        if value is not None:
            words[name] = cp.asnumpy(value).view(np.uint32)
    return words


def _identity():
    off = _run("off")
    for route in ("empty", "zeroed"):
        other = _run(route)
        assert off.keys() == other.keys()
        for name in off:
            assert np.array_equal(off[name], other[name]), (route, name)
    assert np.all(np.isfinite(off["u"].view(np.float32)))
    print("IDENTICAL")


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("numerics", ["fast", "strict"])
def test_no_obstacles_step_the_words_the_off_path_steps(numerics):
    if numerics == "fast" and os.environ.get("GPUWM_WRF_EXACT", "0") == "1":
        pytest.skip("GPUWM_WRF_EXACT=1 is strict numerics")
    here = Path(__file__).resolve().parent
    code = (f"import os, sys; sys.path.insert(0, {str(here)!r}); "
            "import test_obstacles as t; "
            f"os.environ['GPUWM_NUMERICS'] = {numerics!r}; "
            "from woof import numerics; "
            f"assert numerics.mode() == {numerics!r}, numerics.mode(); "
            "t._identity()")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, env=dict(os.environ), timeout=900)
    assert result.returncode == 0 and "IDENTICAL" in result.stdout, (
        result.stdout[-4000:] + result.stderr[-4000:])


def _within(value, band):
    low, high = band
    return ((low is None or value > low) and (high is None or value < high))


@requires_gpu
@pytest.mark.gpu
def test_one_minute_canyon_turns_the_right_way():
    from woof.verify.cases import obstacle_canyon as oc

    m = oc.run(minutes=1.0, average_minutes=0.5)
    assert not m["nan"]
    assert m["floor_u"] < 0 < m["roof_u"], m
    assert m["leak"] < 0.05, m


@requires_gpu
@pytest.mark.gpu
@pytest.mark.slow
def test_canyon_benchmark():
    from woof.verify.cases import obstacle_canyon as oc

    m = oc.run()
    assert not m["nan"]
    for name, band in oc.GATES.items():
        assert _within(m[name], band), (name, m[name], band)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.slow
def test_cube_benchmark():
    from woof.verify.cases import obstacle_cube as oc

    m = oc.run()
    assert not m["nan"]
    for name, band in oc.GATES.items():
        assert _within(m[name], band), (name, m[name], band)
