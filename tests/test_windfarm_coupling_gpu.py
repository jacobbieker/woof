"""windfarm_opt = 1 in the running model, on the device.

The seam (woof.core.physics.PhysicsDriver._couple_pbl_slot) must hand the
Fitch kernel exactly what WRF's PBL driver hands dragforce
(module_pbl_driver.F:2053-2063) -- phy_prep's mass-point winds, z_at_w, the
domain's dx, the MODEL dt, MYNN's QKE as MYNN left it and the scheme's own
RUBLTEN/RVBLTEN -- and couple what comes back.  These tests hold one PBL
step of the driver word for word to the pure kernel and to its binary32 CPU
mirror on the same inputs, a checkpoint round trip bit for bit, and the
off path to the run without the fields.  Synthetic farms only.
"""
from __future__ import annotations

import csv
import dataclasses
import math

import numpy as np
import pytest

from conftest import requires_gpu

CURVE = "V112/3000"
N = 24


def _grid():
    from woof.static.lambert import LambertGrid

    return LambertGrid(ref_lat=35.2, ref_lon=-97.9, truelat1=30.0,
                       truelat2=60.0, stand_lon=-97.9, dx=1000.0, dy=1000.0,
                       e_we=N + 1, e_sn=N + 1)


def _farm_csv(path, *, size=3, spacing=600.0):
    grid = _grid()
    lat0, lon0 = (float(v) for v in grid.ij_to_latlon(N / 2.0 - 2.0,
                                                       N / 2.0 - 2.0))
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["turbine_id", "lat", "lon", "hub_height_m",
                         "curve"])
        for a in range(size):
            for b in range(size):
                lat = lat0 + a * spacing / 111195.0
                lon = lon0 + b * spacing / (111195.0
                                            * math.cos(math.radians(lat0)))
                writer.writerow([f"wt{a}{b}", lat, lon, 94.0, CURVE])
    return path


def _cfg(**over):
    from woof.config import RunConfig

    base = dict(nx=N, ny=N, nz=40, dx=1000.0, dy=1000.0, ztop=16000.0,
                dt=6.0, run_seconds=0.0, time_step_sound=4, moist=True,
                mp_physics=8, sf_sfclay_physics=5, sf_surface_physics=2,
                bl_pbl_physics=5, bldt=0.0)
    base.update(over)
    return RunConfig(**base)


def _state(cfg):
    import cupy as cp

    from woof.config import validate_run_config
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics

    validate_run_config(cfg)

    def theta(z):
        z = np.asarray(z, np.float64)
        return 298.0 + 0.0035 * z

    def qvapor(z):
        z = np.asarray(z, np.float64)
        return 0.012 * np.exp(-z / 2500.0)

    coord = make_vertical_coord(cfg.nz, stretch=1.6)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, qvapor)
    jj, ii = np.meshgrid(np.arange(cfg.ny), np.arange(cfg.nx + 1),
                         indexing="ij")
    state.u[...] = cp.asarray(9.0 + 0.3 * np.sin(ii / 3.0) + 0.1 * jj,
                              dtype=cp.float32)[None]
    state.v[...] = cp.float32(2.5)
    tsk = np.full((cfg.ny, cfg.nx), 302.0)
    soil_t = np.stack([tsk - 0.5, tsk - 1.0, tsk - 1.5, tsk - 2.0])
    soil_m = np.full((4, cfg.ny, cfg.nx), 0.30)
    initialize_physics(
        state, cfg, landmask=np.ones_like(tsk), tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m, ivgtyp=10, isltyp=6, vegfra=55.0, tmn=287.0,
        swdown=650.0, glw=330.0, pblh=600.0, windfarm_grid=_grid())
    return state


def _on(tmp_path, **over):
    return _cfg(windfarm_opt=1,
                windfarm_turbine_file=str(_farm_csv(tmp_path / "farm.csv")),
                **over)


def _words(a):
    import cupy as cp

    a = cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)
    return np.ascontiguousarray(a).view(np.uint32)


def _prognostics(state):
    import cupy as cp

    out = {}
    for name in ("u", "v", "w", "thp", "php", "mup", "qv", "qc", "qr",
                 "qi", "qs", "qg"):
        value = getattr(state, name, None)
        if value is not None:
            out[name] = cp.asnumpy(value).copy()
    for name, value in state.physics.fields.items():
        out[f"fields/{name}"] = cp.asnumpy(value).copy()
    return out


@pytest.mark.gpu
@requires_gpu
def test_one_pbl_step_is_the_pure_kernel_on_the_seams_inputs(
        tmp_path, monkeypatch):
    """The driver's PBL slot == launch_fitch on WRF's arguments, word for
    word, and the MYNN rates it adds into are the off run's own."""
    import cupy as cp

    import woof.core.physics as physics
    from woof.core import windfarm_coupling as wc
    from woof.core.dycore import step
    from woof.core.windfarm_fitch import launch_fitch
    from woof.verify.windfarm_fitch_ref import f32_fitch_column

    coupled, seen = [], []
    original_couple = physics.couple_ysu_tendencies
    original_apply = wc.WindFarm.apply

    def couple(state, cfg, rates):
        coupled.append({k: cp.asnumpy(v).copy() for k, v in rates.items()
                        if k in ("du", "dv")}
                       | {"qke": cp.asnumpy(
                           state.physics.fields["qke"]).copy()})
        return original_couple(state, cfg, rates)

    def apply(self, atmosphere, du, dv, *, qke, power, dx, dt):
        seen.append({"u": cp.asnumpy(atmosphere["u"]).copy(),
                     "v": cp.asnumpy(atmosphere["v"]).copy(),
                     "z_w": cp.asnumpy(atmosphere["z_interface"]).copy(),
                     "du": cp.asnumpy(du).copy(), "dv": cp.asnumpy(dv).copy(),
                     "qke": cp.asnumpy(qke).copy(), "dx": dx, "dt": dt,
                     "layout": self.layout, "factor": self.tke_factor})
        original_apply(self, atmosphere, du, dv, qke=qke, power=power,
                       dx=dx, dt=dt)
        seen[-1]["power"] = cp.asnumpy(power).copy()

    monkeypatch.setattr(physics, "couple_ysu_tendencies", couple)
    monkeypatch.setattr(wc.WindFarm, "apply", apply)

    cfg = _on(tmp_path, windfarm_tke_factor=0.5)
    state = _state(cfg)
    before = physics._prepare_atmosphere(state)
    expected_u = cp.asnumpy(before["u"]).copy()
    expected_zw = cp.asnumpy(before["z_interface"]).copy()
    step(state, cfg)
    assert len(seen) == 1 and len(coupled) == 1
    got = seen[0]
    # WRF's arguments: phy_prep's winds and z_at_w at the step's start, the
    # grid spacing, the model dt (not dtbl), the configured factor.
    assert np.array_equal(_words(got["u"]), _words(expected_u))
    assert np.array_equal(_words(got["z_w"]), _words(expected_zw))
    assert (got["dx"], got["dt"], got["factor"]) == (1000.0, 6.0, 0.5)
    # The pure kernel on the same inputs, from MYNN's own rates and QKE.
    du, dv, qke = (cp.asarray(got[k]) for k in ("du", "dv", "qke"))
    power = launch_fitch(cp.asarray(got["u"]), cp.asarray(got["v"]),
                         cp.asarray(got["z_w"]), got["layout"], du, dv,
                         dx=1000.0, qke=qke, dt=6.0, tke_factor=0.5)
    for name, value in (("du", du), ("dv", dv), ("qke", qke)):
        assert np.array_equal(_words(coupled[0][name]), _words(value)), name
    assert np.array_equal(_words(got["power"]), _words(power))
    assert float(power.sum()) > 0.0
    assert np.array_equal(_words(state.physics.fields["power"]),
                          _words(power))
    # And the binary32 CPU mirror, word for word.
    cdu, cdv, cqke = got["du"].copy(), got["dv"].copy(), got["qke"].copy()
    cpower, _, _ = f32_fitch_column(got["u"], got["v"], got["z_w"],
                                    got["layout"], cdu, cdv, cqke, dx=1000.0,
                                    dt=6.0, tke_factor=0.5)
    for name, value in (("du", cdu), ("dv", cdv), ("qke", cqke)):
        assert np.array_equal(_words(coupled[0][name]), _words(value)), name
    assert np.array_equal(_words(got["power"]), _words(cpower))
    # The rates the farm added into are MYNN's own: the off run's.
    coupled.clear()
    seen.clear()
    off = _state(_cfg())
    step(off, _cfg())
    for name in ("du", "dv", "qke"):
        assert np.array_equal(_words(coupled[0][name]), _words(got[name]))


@pytest.mark.gpu
@requires_gpu
def test_the_farm_slows_and_stirs_its_own_columns(tmp_path):
    import cupy as cp

    from woof.core.dycore import step

    cfg = _on(tmp_path)
    on, off = _state(cfg), _state(_cfg())
    for _ in range(4):
        step(on, cfg)
        step(off, _cfg())
    layout = on.physics.windfarm.layout
    cells = np.flatnonzero(np.diff(layout.cell_offsets) > 0)
    j, i = np.divmod(cells, N)
    qke_on = cp.asnumpy(on.physics.fields["qke"])[:8, j, i]
    qke_off = cp.asnumpy(off.physics.fields["qke"])[:8, j, i]
    # After a few steps MYNN redistributes the source, so the rotor layers
    # as a whole carry more turbulence, not every word.
    assert qke_on[1:6].sum() > 1.5 * qke_off[1:6].sum()
    speed = lambda s: np.hypot(  # noqa: E731
        0.5 * (cp.asnumpy(s.u)[:, :, :-1] + cp.asnumpy(s.u)[:, :, 1:]),
        0.5 * (cp.asnumpy(s.v)[:, :-1] + cp.asnumpy(s.v)[:, 1:]))
    assert speed(on)[2:6, j, i].mean() < speed(off)[2:6, j, i].mean()
    assert float(cp.asnumpy(on.physics.fields["power"]).sum()) > 1.0e6


@pytest.mark.gpu
@requires_gpu
def test_off_is_the_run_without_the_fields(tmp_path):
    """Off -- even with a shared turbine file and factor set -- builds no
    carrier, adds no field and integrates bit for bit as the default."""
    from woof.core.dycore import step

    path = _farm_csv(tmp_path / "farm.csv")
    plain_cfg = _cfg()
    inert_cfg = _cfg(windfarm_turbine_file=str(path), windfarm_tke_factor=1.0)
    plain, inert = _state(plain_cfg), _state(inert_cfg)
    assert plain.physics.windfarm is None and inert.physics.windfarm is None
    assert "power" not in inert.physics.fields
    for _ in range(3):
        step(plain, plain_cfg)
        step(inert, inert_cfg)
    a, b = _prognostics(plain), _prognostics(inert)
    assert a.keys() == b.keys()
    for name in a:
        assert np.array_equal(_words(a[name]), _words(b[name])), name


@pytest.mark.gpu
@requires_gpu
def test_a_checkpoint_round_trip_is_bit_exact(tmp_path):
    from woof.core.dycore import step
    from woof.io import restart

    cfg = _on(tmp_path)
    straight = _state(cfg)
    for _ in range(2):
        step(straight, cfg)
    path = restart.write_restart(tmp_path / "ck.npz", straight, cfg)
    for _ in range(3):
        step(straight, cfg)
    resumed = _state(cfg)
    restart.restore_restart(path, resumed, cfg)
    for _ in range(3):
        step(resumed, cfg)
    a, b = _prognostics(straight), _prognostics(resumed)
    assert "fields/power" in a
    assert a.keys() == b.keys()
    for name in a:
        assert np.array_equal(_words(a[name]), _words(b[name])), name


@pytest.mark.gpu
@requires_gpu
def test_a_resume_refuses_another_turbine_file(tmp_path):
    from woof.core.dycore import step
    from woof.io import restart

    cfg = _on(tmp_path)
    state = _state(cfg)
    step(state, cfg)
    path = restart.write_restart(tmp_path / "ck.npz", state, cfg)
    other = _farm_csv(tmp_path / "other.csv", size=2)
    moved = dataclasses.replace(cfg, windfarm_turbine_file=str(other))
    with pytest.raises(Exception, match="windfarm"):
        restart.restore_restart(path, _state(moved), moved)


@pytest.mark.gpu
@requires_gpu
def test_the_farm_adds_no_device_to_host_copy_per_step(tmp_path,
                                                       monkeypatch):
    """The layout is uploaded once; a step reads nothing back."""
    import cupy as cp

    from woof.core.dycore import step

    counts = {"n": 0}

    def counting(original):
        def wrapper(*args, **kwargs):
            counts["n"] += 1
            return original(*args, **kwargs)
        return wrapper

    def per_step(cfg):
        state = _state(cfg)
        step(state, cfg)
        counts["n"] = 0
        for _ in range(3):
            step(state, cfg)
        return counts["n"]

    for name in ("get", "item", "__float__", "__int__", "__bool__",
                 "tolist"):
        monkeypatch.setattr(cp.ndarray, name,
                            counting(getattr(cp.ndarray, name)))
    monkeypatch.setattr(cp, "asnumpy", counting(cp.asnumpy))
    assert per_step(_on(tmp_path)) == per_step(_cfg())


@pytest.mark.gpu
@requires_gpu
def test_a_wrf_list_and_its_woof_table_run_the_same_words(tmp_path):
    """The two turbine-file roads hand the kernel the same farm."""
    import cupy as cp

    from woof.core import windfarm_coupling as wc
    from woof.core.windfarm_fitch import launch_fitch
    from woof.windfarm import layout as wl

    grid = _grid()
    path = _farm_csv(tmp_path / "farm.csv")
    built = wl.build_layout(wl.read_turbines(path), grid)
    wrf = built.write_wrf_inputs(tmp_path / "wrf") / "windturbines.txt"
    rng = np.random.default_rng(7)
    nz = 30
    dz = np.linspace(20.0, 400.0, nz)
    z_w = np.concatenate([[0.0], np.cumsum(dz)]).astype(np.float32)
    z_w = np.broadcast_to(z_w[:, None, None] + 350.0, (nz + 1, N, N)).copy()
    u = (8.0 + rng.standard_normal((nz, N, N))).astype(np.float32)
    v = (2.0 + rng.standard_normal((nz, N, N))).astype(np.float32)
    results = []
    for source in (path, wrf):
        layout = wc.fitch_layout_for_file(source, shape=(N, N), grid=grid)
        du = cp.zeros((nz, N, N), cp.float32)
        dv = cp.zeros_like(du)
        qke = cp.full_like(du, 0.1)
        power = launch_fitch(u, v, z_w, layout, du, dv, dx=1000.0, qke=qke,
                             dt=6.0)
        results.append([_words(a) for a in (du, dv, qke, power)])
    for a, b in zip(*results):
        assert np.array_equal(a, b)
    assert results[0][3].view(np.float32).sum() > 0
