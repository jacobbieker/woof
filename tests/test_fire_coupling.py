# tests/test_fire_coupling.py
"""WRF's coupled fire (RunConfig.ifire, woof/fire/coupling.py) in the model.

CPU half: the configuration surface (field order, refusals, inert-when-off
identities), fire_ignition_convert's rules, set_nfuel_cat by altitude, the
fuel-statics request ifire makes, the registration tables and the history
routing.  GPU half: an idealised LES grass fire (WRF test/em_fire's shape)
heats and lifts the air above it and its smoke rises; the smoke flux and
active-area kernels equal their NumPy twins word for word; a run resumed
from a checkpoint is bit-identical to one that never stopped; an off run
allocates nothing and is byte-identical whatever the inert fire keys say;
and a fire step reads nothing back to the host.  The WRF parity of the
driver itself is tests/test_fire_coupling_wrf471_parity.py.
"""
from dataclasses import fields, replace

import numpy as np
import pytest
from conftest import requires_gpu

from woof.config import (FIRE_RUN_DEFAULTS, RunConfig, drop_inert_fire_keys,
                         validate_run_config)

F = np.float32
IGNITION = (300.0, 250.0, 300.0, 400.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.5,
            1.0, 4.0)


def _cfg(**kw):
    base = dict(nx=24, ny=20, nz=16, dx=50.0, dy=50.0, ztop=1600.0, dt=0.5,
                run_seconds=0.0, km_opt=2, bl_pbl_physics=0,
                sf_sfclay_physics=0, moist=True, mp_physics=0, isfflx=0,
                time_step_sound=6, open_x=True, open_y=True,
                p_surf=101710.0)
    base.update(kw)
    return RunConfig(**base)


def _fire(**kw):
    base = dict(ifire=2, sr_x=4, sr_y=4, fire_fuel_read=0, fire_fuel_cat=1,
                fire_ignitions=(IGNITION,))
    base.update(kw)
    return _cfg(**base)


# --------------------------------------------------------------- config ---

def test_fields_appended_last_and_default_off():
    names = [f.name for f in fields(RunConfig)]
    tail = names[-len(FIRE_RUN_DEFAULTS):]
    assert tail == list(FIRE_RUN_DEFAULTS)
    assert names[-len(FIRE_RUN_DEFAULTS) - 1] == "smoke_emission_file"
    cfg = _cfg()
    for key, default in FIRE_RUN_DEFAULTS.items():
        assert getattr(cfg, key) == default, key
    validate_run_config(cfg)
    validate_run_config(_fire())


@pytest.mark.parametrize("kw, match", [
    ({"ifire": 1}, "not a WRF v4.7.1 fire model"),
    ({"sr_x": 0}, "needs sr_x >= 1"),
    ({"moist": False}, "needs moist = true"),
    ({"use_adaptive_time_step": True}, "use_adaptive_time_step"),
    ({"fire_fuel_read": 3}, "fire_fuel_read = 3"),
    ({"fire_fuel_cat": 15}, "not an Anderson fuel model"),
    ({"fire_fuel_table": "x"}, "fire_fuel_table"),
    ({"fire_fmc_read": 0}, "fire_fmc_read = 0"),
    ({"fire_wind_height": 0.0}, "fire_wind_height"),
    ({"fire_ext_grnd": -1.0}, "fire_ext_grnd"),
    ({"fire_atm_feedback": -0.5}, "fire_atm_feedback"),
    ({"fire_smoke_factor": "co2"}, "fire_smoke_factor"),
    ({"fire_advection": 2}, "fire_advection"),
    ({"smoke_tracer": True, "smoke_emission_file": "e.npy"},
     "smoke_emission_file"),
    ({"fire_ignitions": ((1.0, 1.0) + (0.0,) * 2 + (2.0, 2.0)
                         + (0.0,) * 6,)}, "Only one of the ideal or real"),
])
def test_fire_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        validate_run_config(_fire(**kw))


def test_small_grid_refused():
    with pytest.raises(ValueError, match="4 x 4"):
        validate_run_config(_fire(nx=3))


def test_inert_keys_drop_from_every_identity():
    from woof.io import restart

    off = restart.configuration_echo(_cfg(fire_wind_height=9.0, sr_x=3))
    assert not set(FIRE_RUN_DEFAULTS) & set(off)
    assert off == restart.configuration_echo(_cfg())
    on = restart.configuration_echo(_fire())
    assert on["ifire"] == 2 and on["sr_x"] == 4
    assert on["fire_ignitions"] == [list(IGNITION)]
    doc = drop_inert_fire_keys({"ifire": 0, "fire_viscosity": 0.3, "u": 1})
    assert doc == {"u": 1}


def test_prepared_identity_tables_and_per_domain():
    from woof.experiment import _DOMAIN_RUN_OVERRIDES
    from woof.ingest.prepared_cache import (DEFAULT_TOLERANT_IDENTITY_FIELDS,
                                            PREPARATION_INERT_RUN_FIELDS)

    for key in FIRE_RUN_DEFAULTS:
        assert key in _DOMAIN_RUN_OVERRIDES, key
        path = f"run.{key}"
        if key in ("ifire", "sr_x", "sr_y"):
            assert path in DEFAULT_TOLERANT_IDENTITY_FIELDS
            assert path not in PREPARATION_INERT_RUN_FIELDS
        else:
            assert path in PREPARATION_INERT_RUN_FIELDS, key


def test_toml_ignitions_normalize():
    from woof.config import normalize_fire_ignitions

    out = normalize_fire_ignitions([{"start_x": 10, "start_y": 20,
                                     "radius": 5.0}])
    assert out == ((10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.01,
                    0.0, 0.0),)
    with pytest.raises(ValueError, match="unknown ignition keys"):
        normalize_fire_ignitions([{"start_q": 1.0}])


# ------------------------------------------------- WRF's driver rules ---

def test_ignition_convert_counts_and_expands_like_wrf():
    from woof.fire.coupling import convert_ignitions, ignition_record

    lines = [ignition_record({"start_x": 100.0, "start_y": 50.0,
                              "radius": 10.0, "start_time": 5.0}),
             ignition_record({"start_x": 7.0, "radius": 0.0})]
    plan = convert_ignitions(lines)
    # the last line with a positive radius is the first: one ignition
    assert len(plan.lines) == 1 and not plan.longlat
    line = plan.lines[0]
    assert line.end == (F(100.0), F(50.0))       # zero end -> start
    assert line.end_time == F(5.0)               # zero end time -> start
    assert plan.unit_x == plan.unit_y == F(1.0)
    real = convert_ignitions([ignition_record({
        "start_lon": -105.0, "start_lat": 40.0, "radius": 10.0})],
        cen_lat=40.0)
    assert real.longlat
    # unit_fxlat = pi2/(360*reradius) in REAL(4): ~111 km per degree
    assert 111000.0 < float(real.unit_y) < 111300.0
    assert 0.76 < float(real.unit_x) / float(real.unit_y) < 0.77
    with pytest.raises(ValueError, match="no positive radius"):
        convert_ignitions([ignition_record({"start_x": 1.0}),
                           ignition_record({"start_x": 2.0, "radius": 3.0})])
    with pytest.raises(ValueError, match="at most 5"):
        convert_ignitions([lines[0]] * 6)


def test_nfuel_by_altitude_is_wrfs_chain():
    from woof.fire.coupling import nfuel_by_altitude

    z = np.array([0.0, 1524.0, 1600.0, 2073.0, 2200.0, 2438.0, 2500.0,
                  3354.0, 3400.0, 3658.0, 3700.0], dtype=F)
    assert nfuel_by_altitude(z).tolist() == [3, 3, 2, 2, 8, 8, 10, 10, 1, 1,
                                             14]


def test_refinement_axis_matches_the_statics_terrain_alignment():
    """interpolate_z2fire's alignment is one case of interpolate_2d's."""
    from woof.fire.coupling import refinement_axis
    from woof.static.fuel import wrf_refinement_axis

    for n, r in ((5, 1), (7, 3), (6, 4), (9, 5)):
        cell, w = refinement_axis(n * r, r, F(1.0) + F(r - 1) * F(0.5), 0,
                                  n + 1)
        ref_cell, ref_w = wrf_refinement_axis(n, r)
        np.testing.assert_array_equal(cell, ref_cell)
        np.testing.assert_array_equal(w.view(np.uint32),
                                      ref_w.view(np.uint32))


def test_ifire_requests_the_domain_fuel_statics():
    from woof.static.fuel import fuel_for, resolve_domain_fuel

    raw = {"shared": {"ifire": 0},
           "domain": [{"grid_id": 1},
                      {"grid_id": 2, "ifire": 2, "sr_x": 4, "sr_y": 5}]}
    carrier = resolve_domain_fuel(raw, None, source="t.toml")
    assert fuel_for(carrier, 1) is None
    request = fuel_for(carrier, 2)
    assert (request.sr_x, request.sr_y) == (4, 5)
    raw["domain"][1]["static"] = {"fuel": {"sr_x": 2, "sr_y": 2}}
    with pytest.raises(ValueError, match="same refinement"):
        resolve_domain_fuel(raw, None, source="t.toml")
    assert resolve_domain_fuel({"shared": {}, "domain": [{"grid_id": 1}]},
                               None, source="t.toml") is None


def test_moving_and_spawned_nests_are_refused():
    from types import SimpleNamespace

    from woof.experiment import _refuse_moving_fire

    fixed = SimpleNamespace(grid_id=2, run=_fire(), follow=None, spawn=None)
    _refuse_moving_fire([fixed], None, "t.toml")
    for kind in ("follow", "spawn"):
        moving = SimpleNamespace(grid_id=2, run=_fire(), follow=None,
                                 spawn=None)
        setattr(moving, kind, object())
        with pytest.raises(ValueError, match="moves or is spawned"):
            _refuse_moving_fire([moving], None, "t.toml")


# ------------------------------------------------------- registrations ---

def test_inventory_and_restart_classification():
    from woof.core.device_inventory import state_array_shapes
    from woof.fire.coupling import REBUILT_ATTRS, STATE_ATTRS
    from woof.io.restart import classify_state_attr

    off = state_array_shapes(_cfg())
    assert not (set(STATE_ATTRS) | set(REBUILT_ATTRS)) & set(off)
    on = state_array_shapes(_fire())
    assert on["fire_lfn"] == on["fire_tign"] == (82, 98)
    assert on["fire_fuel_frac"] == on["fire_uf"] == (80, 96)
    assert on["fire_grnhfx"] == on["fire_area"] == (20, 24)
    assert on["fire_rthfrten"] == (16, 20, 24)
    for name in STATE_ATTRS:
        assert classify_state_attr(name) == "serialize", name
    for name in REBUILT_ATTRS:
        assert classify_state_attr(name) == "rebuild", name
    assert classify_state_attr("_fire") == "infra"


def test_health_rules():
    from woof.core.health import rule_for_field
    from woof.fire.coupling import STATE_ATTRS

    for name in STATE_ATTRS:
        rule = rule_for_field(name)
        assert rule is not None
    frac = rule_for_field("fire_fuel_frac")
    assert (frac.lower, frac.upper) == (0.0, 1.0)


def test_history_inventory_and_subgrid_dims(tmp_path):
    from woof.io.history_layout import produced_history_shapes
    from woof.io.wrf_output_schema import (FIRE_MESH_DIMENSIONED_FIELDS,
                                           REGISTRY_VAR_META)

    off = produced_history_shapes(_cfg())
    on = produced_history_shapes(_fire())
    fire_names = set(on) - set(off)
    assert fire_names == (FIRE_MESH_DIMENSIONED_FIELDS
                          | {"GRNHFX", "GRNQFX", "CANHFX", "CANQFX",
                             "AVG_FUEL_FRAC"})
    for name in FIRE_MESH_DIMENSIONED_FIELDS:
        assert on[name] == (80, 96), name
        assert name in REGISTRY_VAR_META
    assert on["GRNHFX"] == (20, 24)


def test_writer_routes_the_fire_mesh_to_subgrid_axes(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    from woof.io.wrfout import WrfoutWriter

    path = tmp_path / "w.nc"
    with WrfoutWriter(path, nx=24, ny=20, nz=16, dx=50.0, dy=50.0) as w:
        with pytest.raises(ValueError, match="not a refinement"):
            w._dims_for("ROS", (81, 96))
        w.write_frame("2026-10-23_00:00:00", {
            "FGRNHFX": np.full((80, 96), 2.5e4, F),
            "TIGN_G": np.zeros((80, 96), F),
            "GRNHFX": np.full((20, 24), 3.0e3, F)})
    with netCDF4.Dataset(path) as ds:
        assert ds["FGRNHFX"].dimensions == (
            "Time", "south_north_subgrid", "west_east_subgrid")
        assert ds.dimensions["west_east_subgrid"].size == 96
        assert ds.dimensions["south_north_subgrid"].size == 80
        assert ds["FGRNHFX"].units == "W/m^2"
        assert ds["GRNHFX"].dimensions == ("Time", "south_north",
                                           "west_east")


# ------------------------------------------------------------------ GPU ---

SOUNDING = np.array([(0.0, 288.0, 9.0), (1000.0, 295.0, 7.4),
                     (2000.0, 302.5, 5.5), (3000.0, 308.79, 4.0)])


def _state(cfg, u0=3.0):
    import cupy as cp

    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced

    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: np.interp(np.asarray(z, float), SOUNDING[:, 0],
                                   SOUNDING[:, 1]),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, coord, base,
        lambda z: 1e-3 * np.interp(np.asarray(z, float), SOUNDING[:, 0],
                                   SOUNDING[:, 2]))
    state.u[...] = cp.float32(u0)
    return state


def _words(a):
    a = a.get() if hasattr(a, "get") else np.asarray(a)
    return np.ascontiguousarray(a).view(np.uint8)


def _serialized(state):
    from woof.state_serialization_contract import STATE_SERIALIZED_ATTRS

    return {name: _words(getattr(state, name))
            for name in STATE_SERIALIZED_ATTRS
            if getattr(state, name, None) is not None
            and hasattr(getattr(state, name), "shape")}


@pytest.mark.gpu
@requires_gpu
def test_grass_fire_heats_lifts_and_lofts_smoke():
    import cupy as cp

    from woof.core.dycore import step

    cfg = _fire(smoke_tracer=True)
    state = _state(cfg)
    theta0 = cp.asnumpy(state.thp).copy()
    for _ in range(80):                          # 40 s
        step(state, cfg)
    assert float(cp.max(state.fire_fgrnhfx)) > 1.0e4
    assert float(cp.max(state.fire_grnhfx)) > 1.0e3
    assert float(cp.max(state.fire_area)) > 0.0
    burning = cp.asnumpy(state.fire_grnhfx) > 0
    dtheta = cp.asnumpy(state.thp) - theta0
    assert dtheta[0][burning].max() > 0.1           # the fire heats the air
    assert float(cp.max(state.w)) > 0.05            # and lifts it
    smoke = cp.asnumpy(state.smoke)
    assert smoke.min() >= 0.0 and smoke.max() > 0.0
    assert smoke[2:].sum() > 0.0                    # smoke above layer 2
    emis = cp.asnumpy(state.smoke_emis)
    assert emis.max() > 0.0
    assert np.all(emis[[0, -1], :] == 0) and np.all(emis[:, [0, -1]] == 0)


@pytest.mark.gpu
@requires_gpu
def test_smoke_flux_and_active_area_kernels_match_numpy():
    import cupy as cp

    from woof.fire import coupling as c

    rng = np.random.default_rng(7)
    ny, nx, ir, jr = 6, 7, 3, 4
    burnt = rng.uniform(0, 0.3, (ny * jr, nx * ir)).astype(F)
    fgip = rng.uniform(0, 1.5, burnt.shape).astype(F)
    factor = rng.uniform(0, 0.02, burnt.shape).astype(F)
    for fts in (factor, F(0.02)):
        want = c.smoke_flux_np(burnt, fgip, fts, F(0.7), ir=ir, jr=jr,
                               ny=ny, nx=nx)
        out = cp.full((ny, nx), -1.0, dtype=F)
        c.smoke_flux_gpu(cp.asarray(burnt), cp.asarray(fgip),
                         cp.asarray(factor) if np.ndim(fts) else None,
                         F(0.02), F(0.7), out, ir=ir, jr=jr)
        assert np.array_equal(_words(out), _words(want))
    area = rng.uniform(-0.2, 1.0, burnt.shape).astype(F).clip(0, None)
    tign = rng.uniform(0, 60, (ny * jr + 2, nx * ir + 2)).astype(F)
    fuel_time = rng.uniform(5, 50, burnt.shape).astype(F)
    want = c.active_area_np(area, tign, fuel_time, F(55.0), F(12.5),
                            ir=ir, jr=jr, ny=ny, nx=nx)
    out = cp.empty((ny, nx), dtype=F)
    c.active_area_gpu(cp.asarray(area), cp.asarray(tign),
                      cp.asarray(fuel_time), F(55.0), F(12.5), out, ir=ir,
                      jr=jr)
    assert np.array_equal(_words(out), _words(want))
    alt = rng.uniform(0.8, 1.2, (3, ny, nx)).astype(F)
    qv = rng.uniform(0, 0.02, alt.shape).astype(F)
    rho = cp.empty_like(cp.asarray(alt))
    c.rho_gpu(cp.asarray(alt), cp.asarray(qv), rho)
    assert np.array_equal(_words(rho), _words(c.rho_np(alt, qv)))


@pytest.mark.gpu
@requires_gpu
def test_restart_resumes_bit_identical(tmp_path):
    from woof.core.dycore import step
    from woof.io import restart

    cfg = _fire(smoke_tracer=True)
    straight = _state(cfg)
    for _ in range(100):                        # 50 s, the fire burning
        step(straight, cfg)
    first = _state(cfg)
    for _ in range(60):
        step(first, cfg)
    path = tmp_path / "fire.npz"
    restart.write_restart(path, first, cfg)
    resumed = _state(cfg)
    restart.restore_restart(path, resumed, cfg)
    import cupy as cp
    # The fire is burning at the checkpoint, and nothing has attached the
    # resumed state's runtime yet: its first step attaches lazily and must
    # keep the restored fire rather than re-initialising it.
    assert bool(cp.any(resumed.fire_lfn < 0))
    assert getattr(resumed, "_fire", None) is None
    for _ in range(40):
        step(resumed, cfg)
    a, b = _serialized(straight), _serialized(resumed)
    assert set(a) == set(b)
    assert {"fire_lfn", "fire_tign", "smoke"} <= set(a)
    bad = [name for name in a if not np.array_equal(a[name], b[name])]
    assert not bad, bad


@pytest.mark.gpu
@requires_gpu
def test_off_run_is_inert_whatever_the_fire_keys_say():
    from woof.core.dycore import step
    from woof.fire.coupling import STATE_ATTRS

    plain = _cfg()
    keyed = _cfg(fire_wind_height=9.0, sr_x=4, sr_y=4,
                 fire_ignitions=(IGNITION,), fire_fuel_read=0)
    a, b = _state(plain), _state(keyed)
    for name in STATE_ATTRS:
        assert not hasattr(a, name) and not hasattr(b, name)
    for _ in range(10):
        step(a, plain)
        step(b, keyed)
    sa, sb = _serialized(a), _serialized(b)
    assert set(sa) == set(sb)
    assert all(np.array_equal(sa[k], sb[k]) for k in sa)


@pytest.mark.gpu
@requires_gpu
def test_fire_step_reads_nothing_back(monkeypatch):
    """A fire step adds no device-to-host read to the step it runs in."""
    import cupy as cp

    from woof.core.dycore import step

    counts = {"n": 0}

    def counting(original):
        def wrapper(*args, **kwargs):
            counts["n"] += 1
            return original(*args, **kwargs)
        return wrapper

    for name in ("get", "item", "__float__", "__int__", "__bool__",
                 "tolist"):
        monkeypatch.setattr(cp.ndarray, name,
                            counting(getattr(cp.ndarray, name)))
    monkeypatch.setattr(cp, "asnumpy", counting(cp.asnumpy))

    def per_step(cfg):
        state = _state(cfg)
        step(state, cfg)                    # attaches the fire once
        counts["n"] = 0
        for _ in range(3):
            step(state, cfg)
        return counts["n"]

    assert per_step(_fire(smoke_tracer=True)) == per_step(_cfg())


def _chaparral_case(xp):
    from woof.fire import levelset as ls

    mesh = ls.FireMesh(12, 10, 30.0, 30.0, 4, 4)
    chap = np.zeros(mesh.shape, F)
    chap[:, mesh.nx // 2:] = 1.0
    params = ls.FireParams.build(
        mesh, xp=xp, vx=4.0, vy=1.0, dzdxf=0.05, r_0=10.0, bbb=1.5,
        phiwc=0.02, betafl=0.02, fgip=0.5, ischap=xp.asarray(chap))
    state = ls.initial_state(mesh, xp=xp)
    ignite = ls.Ignition(start=(180.0, 150.0), radius=40.0, ros=1.0)
    for _ in range(6):
        state, out = ls.step(state, mesh, params, 0.5, ignitions=(ignite,))
    return chap, state, out


def test_chaparral_runs_the_wind_only_rate_with_ros_base_zero():
    """fire_ros's chaparral arm: max(1.2974 spdms**1.41, .03333), base 0.

    WRF leaves ros_base undefined on this arm (the held-out word, as in
    woof.fire.wrf_sfire); the rest is WRF's statement.  A front cell's
    rate there is never below the backing floor and differs from the
    Rothermel half's.
    """
    from woof.fire import levelset as ls
    from woof.fire.wrf_sfire import r4

    chap, state, out = _chaparral_case(np)
    ros = np.asarray(out.ros)
    assert (ros[chap > 0] > 0).any() and (ros[chap == 0] > 0).any()
    # The arm itself, for a front normal along x (fire_advection = 1).
    mesh = ls.FireMesh(4, 4, 30.0, 30.0, 1, 1)
    p = ls.FireParams.build(mesh, vx=1.0, vy=0.5, dzdxf=0.05, r_0=10.0,
                            bbb=1.5, phiwc=0.02, betafl=0.02, ischap=1.0)
    one, zero = np.ones(mesh.shape, F), np.zeros(mesh.shape, F)
    rb, rw, rs = ls._fire_ros_cpu(one, zero, p, 1)
    speed = np.sqrt(F(1.0) + F(0.25)) + ls.TINY
    cor = np.fmax(F(0.0), F(1.0) / speed)
    want = np.fmax(r4("1.2974") * ls._powf(speed, r4("1.41")),
                   r4(".03333")) * cor
    assert np.all(rb == 0) and np.all(rs == 0)
    np.testing.assert_array_equal(rw.view(np.uint32),
                                  np.broadcast_to(want, rw.shape)
                                  .astype(F).view(np.uint32))


@pytest.mark.gpu
@requires_gpu
def test_chaparral_device_matches_host():
    import cupy as cp

    _, host_state, host_out = _chaparral_case(np)
    _, dev_state, dev_out = _chaparral_case(cp)
    for a, b in ((host_out.ros, dev_out.ros),
                 (host_state.lfn_m, dev_state.lfn_m),
                 (host_state.tign_m, dev_state.tign_m),
                 (host_out.fgrnhfx, dev_out.fgrnhfx)):
        assert np.array_equal(_words(a), _words(b))


# ------------------------------------------------------- kernel registry ---

def test_the_recorded_frames_are_bound_to_the_unit_that_compiles():
    import hashlib
    import re

    from woof.core import kernel_frame_recordings as recordings
    from woof.fire import coupling

    source = coupling.module_source()
    assert hashlib.sha256(source.encode()).hexdigest() == \
        recordings.FIRE_COUPLING_COMPOSED_SOURCE_SHA256
    assert coupling.MODULE_OPTIONS == \
        recordings.FIRE_COUPLING_COMPOSED_OPTIONS
    exports = set(re.findall(r"__global__ void (fc_\w+)", source))
    for row in recordings.FIRE_COUPLING_COMPOSED_FRAME_READINGS.values():
        assert set(row) == exports


def test_a_fire_domain_prices_both_fire_units():
    from types import SimpleNamespace

    from woof.core.preflight import (CHAINED_TRANSLATION_UNIT_FRAMES,
                                     domain_kernel_modules)

    on = domain_kernel_modules(SimpleNamespace(run=_fire(), grid_id=1),
                               prices_refl=False)
    off = domain_kernel_modules(SimpleNamespace(run=_cfg(), grid_id=1),
                                prices_refl=False)
    units = {"fire_levelset_composed", "fire_coupling_composed"}
    assert units <= set(on)
    assert not units & set(off)
    assert CHAINED_TRANSLATION_UNIT_FRAMES[
        "fire_coupling_composed"].covers == {"fire_coupling"}


@pytest.mark.gpu
@requires_gpu
def test_the_coupling_unit_frames_match_the_recorded_unit():
    import cupy as cp

    from woof.certify.compile_platform import compile_platform_fingerprint
    from woof.core import kernel_frame_recordings as recordings
    from woof.fire import coupling

    rows = recordings.FIRE_COUPLING_COMPOSED_FRAME_READINGS
    module = coupling._module(cp.cuda.Device().id)
    names = next(iter(rows.values()))
    got = {name: int(cp.cuda.driver.funcGetAttribute(
        cp.cuda.driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        module.get_function(name).ptr)) for name in names}
    fingerprint = compile_platform_fingerprint()
    key = (fingerprint["device_compute_capability"],
           fingerprint["nvrtc_build"])
    if key in rows:
        assert got == dict(rows[key]), (key, got)
    else:
        ceiling = max(max(row.values()) for row in rows.values())
        assert max(got.values()) <= ceiling, (key, got, ceiling)
