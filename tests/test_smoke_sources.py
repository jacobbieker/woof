# tests/test_smoke_sources.py
"""The smoke tracer's sources and sinks (woof/core/smoke_sources.py).

CPU half: the configuration surface (field order, refusals, identity and
checkpoint inertness at the defaults), the registration tables, WRF's
plumerise schedule and Zhang's season.  GPU half: every new glue kernel
equals its NumPy twin word for word; one step's injection and removal equal
the merged woof.smoke library routines called on the same inputs word for
word (PBL profile, plume rise per column on its own heat flux, Zhang
deposition, GOCART settling); the mass budget closes; a checkpoint resumes
bit for bit; and nothing is read back to the host in a step.
"""
from dataclasses import fields, replace
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

from woof.config import RunConfig, validate_run_config
from woof.core import smoke_sources as ss

NEW_FIELDS = ("smoke_injection", "smoke_plumerise_interval_s",
              "smoke_plume_heat_flux", "smoke_plume_fire_area",
              "smoke_plume_flaming_fraction", "smoke_drydep",
              "smoke_aer_drydep_opt", "smoke_settling",
              "smoke_particle_radius", "smoke_particle_density")


def _cfg(**kw):
    kw.setdefault("nx", 12)
    kw.setdefault("ny", 8)
    kw.setdefault("nz", 30)
    kw.setdefault("dx", 1000.0)
    kw.setdefault("dy", 1000.0)
    kw.setdefault("ztop", 20000.0)
    kw.setdefault("dt", 10.0)
    kw.setdefault("run_seconds", 0.0)
    kw.setdefault("km_opt", 0)
    if kw["km_opt"] == 0:
        from woof.config import KM_OPT_ZERO_ACK
        kw.setdefault("km_opt_zero_acknowledgement", KM_OPT_ZERO_ACK)
    return RunConfig(**kw)


# --------------------------------------------------------------- config ---

def test_fields_appended_last_with_their_defaults():
    names = [f.name for f in fields(RunConfig)]
    assert tuple(names[-len(NEW_FIELDS):]) == NEW_FIELDS
    cfg = _cfg()
    for name in NEW_FIELDS:
        assert getattr(cfg, name) == ss.SMOKE_SOURCE_DEFAULTS[name], name
    assert set(ss.SMOKE_SOURCE_DEFAULTS) == set(NEW_FIELDS)
    # The particle defaults are GOCART's finest dust bin, read from WRF.
    assert ss.GOCART_PM25_RADIUS == 0.73e-6
    assert ss.GOCART_PM25_DENSITY == 2500.0


@pytest.mark.parametrize("kw, match", [
    ({"smoke_injection": "plume"}, "must be one of surface, pbl, plumerise"),
    ({"smoke_injection": "pbl"}, "set smoke_tracer = true"),
    ({"smoke_drydep": True}, "set smoke_tracer = true"),
    ({"smoke_tracer": True, "smoke_drydep": 1}, "must be true or false"),
    ({"smoke_tracer": True, "smoke_aer_drydep_opt": 101},
     "must be one of 301, 302, 311, 312"),
    ({"smoke_tracer": True, "smoke_plume_flaming_fraction": 1.5},
     r"must be in \[0, 1\]"),
    ({"smoke_tracer": True, "smoke_particle_radius": 0.0},
     "must be a positive number"),
    ({"smoke_tracer": True, "smoke_plumerise_interval_s": -1.0},
     "must be a finite number"),
    ({"smoke_tracer": True, "smoke_injection": "pbl"},
     "bl_pbl_physics = 0 computes none"),
    ({"smoke_tracer": True, "smoke_drydep": True},
     "sf_sfclay_physics = 0 computes none"),
    ({"smoke_tracer": True, "smoke_injection": "plumerise",
      "smoke_emission_file": "e.npy"}, "with a .npy emission"),
    ({"smoke_tracer": True, "smoke_injection": "plumerise", "nz": 200},
     "takes 3 to 198 levels"),
    ({"smoke_tracer": True, "smoke_injection": "plumerise",
      "smoke_plume_heat_flux": 3.0e4}, "set both positive"),
])
def test_config_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        validate_run_config(_cfg(**kw))


def test_config_admits_every_supported_selection():
    validate_run_config(_cfg(smoke_tracer=True))
    validate_run_config(_cfg(smoke_tracer=True, smoke_injection="plumerise"))
    validate_run_config(_cfg(smoke_tracer=True, smoke_injection="plumerise",
                             smoke_emission_file="e.npy",
                             smoke_plume_heat_flux=3.0e4,
                             smoke_plume_fire_area=1.0e4))
    validate_run_config(_cfg(smoke_tracer=True, smoke_injection="pbl",
                             bl_pbl_physics=1, sf_sfclay_physics=1))
    for opt in ss.AER_DRYDEP_OPTS:
        validate_run_config(_cfg(smoke_tracer=True, smoke_drydep=True,
                                 smoke_aer_drydep_opt=opt,
                                 sf_sfclay_physics=1, bl_pbl_physics=1))
    validate_run_config(_cfg(smoke_tracer=True, smoke_settling=True))


def test_default_keys_drop_from_checkpoint_echo_and_identity():
    from woof.io import restart

    on = _cfg(smoke_tracer=True)
    echo = restart.configuration_echo(on)
    for name in NEW_FIELDS:
        assert name not in echo, name
    moved = restart.configuration_echo(replace(
        on, smoke_settling=True, smoke_particle_radius=1.0e-6))
    assert moved["smoke_settling"] is True
    assert moved["smoke_particle_radius"] == 1.0e-6
    assert "smoke_injection" not in moved


def test_identity_tables():
    from woof.ingest.prepared_cache import (DEFAULT_TOLERANT_IDENTITY_FIELDS,
                                            PREPARATION_INERT_RUN_FIELDS)

    allocating = {"smoke_injection", "smoke_drydep", "smoke_settling"}
    for name in NEW_FIELDS:
        key = "run." + name
        if name in allocating:
            assert key in DEFAULT_TOLERANT_IDENTITY_FIELDS, name
            assert key not in PREPARATION_INERT_RUN_FIELDS, name
        else:
            assert key in PREPARATION_INERT_RUN_FIELDS, name


# ------------------------------------------------------- registrations ---

def test_inventory_and_restart_classification():
    from woof.core.device_inventory import state_array_shapes
    from woof.io.restart import (DRIVER_REBUILT_ATTRS, classify_state_attr)

    new = {"smoke_inj_frac", "smoke_inj_top", "smoke_drydep_vel",
           "smoke_deposited", "smoke_graset"}
    assert not new & set(state_array_shapes(_cfg(smoke_tracer=True)))
    plume = state_array_shapes(_cfg(smoke_tracer=True,
                                    smoke_injection="plumerise"))
    assert plume["smoke_inj_frac"] == (30, 8, 12)
    assert plume["smoke_inj_top"] == (8, 12)
    pbl = state_array_shapes(_cfg(smoke_tracer=True, smoke_injection="pbl"))
    assert "smoke_inj_frac" not in pbl and pbl["smoke_inj_top"] == (8, 12)
    sinks = state_array_shapes(_cfg(smoke_tracer=True, smoke_drydep=True,
                                    smoke_settling=True))
    assert {"smoke_drydep_vel", "smoke_deposited", "smoke_graset"} <= set(sinks)
    settle = state_array_shapes(_cfg(smoke_tracer=True, smoke_settling=True))
    assert "smoke_drydep_vel" not in settle and "smoke_deposited" in settle
    for name in new:
        assert classify_state_attr(name) == "serialize", name
    assert classify_state_attr("_smoke_plume_bound") == "infra"
    assert "smoke_surface" in DRIVER_REBUILT_ATTRS


def test_health_rules():
    from woof.core.health import rule_for_field

    assert rule_for_field("smoke_drydep_vel").upper == 0.5
    for name in ("smoke_inj_frac", "smoke_inj_top", "smoke_drydep_vel",
                 "smoke_deposited", "smoke_graset"):
        rule = rule_for_field(name)
        assert rule.status_class == "tracer" and rule.lower == 0.0, name


def test_history_rows_and_planner():
    from woof.io.history_layout import (live_state_history_fields,
                                        produced_history_shapes)
    from woof.io.wrf_output_schema import REGISTRY_VAR_META

    assert REGISTRY_VAR_META["SMOKE_DRYDEP_VEL"][1] == "m s-1"
    assert REGISTRY_VAR_META["SMOKE_DEPOSITED"][1] == "ug m-2"
    assert REGISTRY_VAR_META["SMOKE_GRASET"][1] == "ug m-2"
    assert REGISTRY_VAR_META["SMOKE_INJ_TOP"][1] == "m"
    state = SimpleNamespace(smoke_deposited=np.zeros((3, 4)))
    assert live_state_history_fields(state)["SMOKE_DEPOSITED"] is \
        state.smoke_deposited
    base = produced_history_shapes(_cfg(smoke_tracer=True))
    on = produced_history_shapes(_cfg(smoke_tracer=True,
                                      smoke_injection="plumerise",
                                      smoke_drydep=True, smoke_settling=True))
    assert set(on) - set(base) == {"SMOKE_INJ_TOP", "SMOKE_DRYDEP_VEL",
                                   "SMOKE_DEPOSITED", "SMOKE_GRASET"}


def test_preflight_slots_lifetimes_transients_and_modules():
    from woof.core import preflight as pf

    tracer = pf.scratch_slot_registry(_cfg(smoke_tracer=True))
    assert not any(name.startswith("smoke_src_") for name in tracer)
    for kw in ({"smoke_injection": "plumerise"},
               {"smoke_injection": "pbl", "bl_pbl_physics": 1},
               {"smoke_drydep": True, "smoke_settling": True}):
        slots = pf.scratch_slot_registry(_cfg(smoke_tracer=True, **kw))
        mine = [name for name in slots if name.startswith("smoke_src_")]
        assert mine, kw
        for name in mine:
            assert pf.scratch_slot_lifetime(name) is not None, name
    assert pf.smoke_sources_transient_shapes(_cfg(smoke_tracer=True)) == {}
    assert pf.smoke_sources_transient_shapes(
        _cfg(smoke_tracer=True, smoke_settling=True))
    assert "smoke_sources" in pf.UNMEASURED_KERNEL_MODULES
    frame = pf.CHAINED_TRANSLATION_UNIT_FRAMES["smoke_sources_composed"]
    assert frame.covers == frozenset({"smoke_sources"})


# ------------------------------------------------------------ schedule ---

def test_plume_schedule_is_wrfs_and_a_function_of_the_clock():
    cfg = _cfg(smoke_tracer=True, smoke_injection="plumerise", dt=60.0,
               smoke_plumerise_interval_s=600.0)
    due = [ss.plume_due(SimpleNamespace(elapsed_seconds=60.0 * n), cfg)
           for n in range(25)]
    # ktau == 1, then every step that ends at or past a multiple of frq
    # (emissions_driver.F:687-694): curr_secs + dt + 0.01 >= next multiple.
    assert [n for n, d in enumerate(due) if d] == [0, 9, 19]
    # A nest born at 1234 s calls on its own first step.
    nest = SimpleNamespace(elapsed_seconds=1234.0, domain_start_offset=1234.0)
    assert ss.plume_due(nest, cfg)
    nest.elapsed_seconds = 1294.0
    assert not ss.plume_due(nest, cfg)
    always = replace(cfg, smoke_plumerise_interval_s=0.0)
    assert ss.plume_due(SimpleNamespace(elapsed_seconds=420.0), always)


def test_zhang_season_and_variant():
    start = datetime(2026, 1, 15)
    assert ss.zhang_season(301, start, 0.0) == 2
    assert ss.zhang_season(302, start, 0.0) == 1
    assert ss.zhang_season(311, datetime(2026, 7, 1), 0.0) == 1
    assert ss.zhang_season(301, datetime(2026, 9, 27), 2 * 86400.0) == 2
    assert ss.zhang_optaa(301) == ss.zhang_optaa(302) == 1
    assert ss.zhang_optaa(311) == ss.zhang_optaa(312) == 2
    with pytest.raises(RuntimeError, match="no start time"):
        ss.zhang_season(301, None, 0.0)


def test_plume_fields_from_the_emission_file(tmp_path):
    from netCDF4 import Dataset

    def write(path, names):
        with Dataset(path, "w") as ds:
            ds.createDimension("south_north", 4)
            ds.createDimension("west_east", 5)
            for name in ("smoke_emission", *names):
                var = ds.createVariable(name, "f4",
                                        ("south_north", "west_east"))
                var[:] = np.full((4, 5), 2.0, np.float32)

    write(tmp_path / "both.nc", ("fire_heat_flux", "fire_area"))
    heat, area = ss.load_plume_fields(tmp_path / "both.nc", (4, 5))
    assert heat.dtype == np.float32 and float(area[0, 0]) == 2.0
    write(tmp_path / "none.nc", ())
    assert ss.load_plume_fields(tmp_path / "none.nc", (4, 5)) is None
    with Dataset(tmp_path / "kw.nc", "w") as ds:
        ds.createDimension("south_north", 4)
        ds.createDimension("west_east", 5)
        for name, units in (("fire_heat_flux", "kW m-2"),
                            ("fire_area", "m2")):
            var = ds.createVariable(name, "f4", ("south_north", "west_east"))
            var.units = units
            var[:] = np.full((4, 5), 3.0, np.float32)
    heat, _ = ss.load_plume_fields(tmp_path / "kw.nc", (4, 5))
    assert float(heat[0, 0]) == 3000.0
    with Dataset(tmp_path / "frac.nc", "w") as ds:
        ds.createDimension("south_north", 4)
        ds.createDimension("west_east", 5)
        for name, units in (("fire_heat_flux", "W m-2"), ("fire_area", "1")):
            var = ds.createVariable(name, "f4", ("south_north", "west_east"))
            var.units = units
            var[:] = np.full((4, 5), 0.5, np.float32)
    with pytest.raises(ValueError, match="fire_area in '1'"):
        ss.load_plume_fields(tmp_path / "frac.nc", (4, 5))
    write(tmp_path / "one.nc", ("fire_area",))
    with pytest.raises(ValueError, match="not its partner"):
        ss.load_plume_fields(tmp_path / "one.nc", (4, 5))
    assert ss.load_plume_fields(tmp_path / "e.npy", (4, 5)) is None


def test_the_driver_attachment_refuses_at_setup():
    on = _cfg(smoke_tracer=True, smoke_drydep=True)
    start = datetime(2026, 5, 1)
    assert ss.smoke_surface(_cfg(), "USGS", None) is None
    assert ss.smoke_surface(on, "USGS", start).dataset == "USGS"
    with pytest.raises(ValueError, match="USGS land-use classes"):
        ss.smoke_surface(on, "NLCD40", start)
    # A dated season with no date fails at setup, not at step 1.
    with pytest.raises(ValueError, match="no start time"):
        ss.smoke_surface(on, "USGS", None)
    summer = replace(on, smoke_aer_drydep_opt=302)
    assert ss.smoke_surface(summer, "USGS", None).start_time is None


# ------------------------------------------------------------------ GPU ---

def _words(a):
    a = np.ascontiguousarray(np.asarray(a))
    return a.view(np.uint64 if a.dtype == np.float64 else np.uint32)


def _same(actual, expected, what=""):
    import cupy as cp

    actual = cp.asnumpy(actual) if hasattr(actual, "get") else actual
    expected = cp.asnumpy(expected) if hasattr(expected, "get") else expected
    actual = np.asarray(actual)
    expected = np.asarray(expected, dtype=actual.dtype)
    assert actual.shape == expected.shape, what
    np.testing.assert_array_equal(_words(actual), _words(expected),
                                  err_msg=what)


def _state(cfg, u0=0.0, seed=0, noise=False):
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest

    vc = make_vertical_coord(cfg.nz)
    base = make_base_state(
        vc, lambda z: np.where(z < 11000.0, 288.0 + 0.0045 * z,
                               337.5 + 0.02 * (z - 11000.0)),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_at_rest(cfg, vc, base)
    rng = np.random.default_rng(seed)
    if u0 and noise:
        # Unbalanced winds for a single evaluation of the ambient column
        # (they are not stepped).
        state.u[...] = cp.asarray(
            u0 + rng.normal(0.0, 1.0, state.u.shape), dtype=cp.float32)
        state.v[...] = cp.asarray(
            rng.normal(0.0, 1.0, state.v.shape), dtype=cp.float32)
    elif u0:
        state.u[...] = cp.float32(u0)
        state.v[...] = cp.float32(0.3 * u0)
    update_diagnostics(state, cfg.hypsometric_opt)
    return state


def _fake_physics(state, *, seed=1, dataset="USGS", pblh=1500.0):
    """The surface fields the deposition reads, as a driver publishes them."""
    import cupy as cp

    rng = np.random.default_rng(seed)
    ny, nx = state.mup.shape
    classes = 21 if dataset != "USGS" else 25
    fields = {
        "ust": cp.asarray(rng.uniform(0.05, 0.8, (ny, nx)), cp.float32),
        "znt": cp.asarray(rng.uniform(0.001, 1.0, (ny, nx)), cp.float32),
        "rmol": cp.asarray(rng.uniform(-0.05, 0.05, (ny, nx)), cp.float32),
        "ivgtyp": cp.asarray(rng.integers(1, classes + 1, (ny, nx)),
                             cp.int32),
        "pblh": cp.asarray(pblh * rng.uniform(0.5, 1.5, (ny, nx)),
                           cp.float32),
    }
    state.physics = SimpleNamespace(
        fields=fields,
        smoke_surface=ss.SmokeSurface(dataset, datetime(2026, 8, 1, 12)))
    return fields


def _emission(path, ny, nx, cells, flux=1.0e-8):
    field = np.zeros((ny, nx))
    for j, i in cells:
        field[j, i] = flux
    np.save(path, field)
    return field


@pytest.mark.gpu
@requires_gpu
def test_the_sources_unit_compiles_within_its_priced_frame():
    import re

    import cupy as cp

    from woof.core import preflight as pf

    priced = pf.CHAINED_TRANSLATION_UNIT_FRAMES["smoke_sources_composed"]
    module = ss._module(cp.cuda.Device().id)
    names = set(re.findall(r'extern "C" __global__ void (\w+)',
                           ss.module_source()))
    frames = {name: int(cp.cuda.driver.funcGetAttribute(
        cp.cuda.driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        module.get_function(name).ptr)) for name in names}
    assert max(frames.values()) <= priced.max_local_size_bytes, frames


@pytest.mark.gpu
@requires_gpu
def test_ambient_column_kernel_matches_its_twin_word_for_word():
    import cupy as cp

    cfg = _cfg(smoke_tracer=True, smoke_injection="plumerise", moist=True)
    state = _state(cfg, u0=6.0, noise=True)
    rng = np.random.default_rng(3)
    state.thp[...] += cp.asarray(rng.normal(0, 0.5, state.thp.shape),
                                 cp.float32)
    state.qv[...] = cp.asarray(rng.uniform(0, 0.012, state.qv.shape),
                               cp.float32)
    env = ss._env(state, plume=True, sinks=True)
    host = {name: cp.asnumpy(getattr(state, name)) for name in (
        "u", "v", "thb", "thp", "p", "qv", "phb", "php", "alt")}
    ref = ss.env_reference(**host)
    for name, value in env.items():
        _same(value, ref[name], name)


@pytest.mark.gpu
@requires_gpu
def test_small_glue_kernels_match_their_twins_word_for_word():
    import cupy as cp

    from woof.core.smoke_tracer import UG_G

    rng = np.random.default_rng(11)
    nz, ny, nx = 9, 5, 7
    n = ny * nx
    grid = ((n + 127) // 128,)

    # the plume profile's normalization, one refused column
    eburn = rng.uniform(0, 3e-3, (nz, ny, nx)).astype(np.float32)
    eburn[0] = 1.0
    eburn[:, 0, 0] = 0.0
    eburn[0, 0, 0] = 1.0
    dz = rng.uniform(20, 400, (nz, ny, nx)).astype(np.float32)
    zw = np.concatenate([np.zeros((1, ny, nx)), np.cumsum(dz, axis=0)]
                        ).astype(np.float32)
    status = np.zeros((ny, nx), np.int32)
    status[2, 3] = 3
    frac = cp.empty((nz, ny, nx), cp.float32)
    top = cp.empty((ny, nx), cp.float32)
    ss._kernel("smoke_plume_fraction")(
        grid, (128,), (cp.asarray(eburn), cp.asarray(dz), cp.asarray(zw),
                       cp.asarray(status), frac, top, np.int32(nz),
                       np.int32(n)))
    rf, rt = ss.plume_fraction_reference(eburn, dz, zw, status)
    _same(frac, rf, "frac")
    _same(top, rt, "inj_top")
    np.testing.assert_allclose(cp.asnumpy(frac).sum(axis=0), 1.0, rtol=1e-6)

    # the PBL profile's fractions
    source = rng.uniform(0, 1e-3, (nz, ny, nx)).astype(np.float32)
    source[5:] = 0.0
    ss._kernel("smoke_pbl_fraction")(
        grid, (128,), (cp.asarray(source), cp.asarray(zw), frac, top,
                       np.int32(nz), np.int32(n)))
    rf, rt = ss.pbl_fraction_reference(source, zw)
    _same(frac, rf, "pbl frac")
    _same(top, rt, "pbl top")

    # the profile source, with an all-zero (not yet computed) column
    f = rng.uniform(0, 0.3, (nz, ny, nx)).astype(np.float32)
    f[3:] = 0.0
    f[:, 1, 1] = 0.0
    tend = rng.normal(0, 1, (nz, ny, nx)).astype(np.float32)
    emis = rng.uniform(0, 1e-7, (ny, nx)).astype(np.float32)
    dnw = -rng.uniform(0.01, 0.05, nz).astype(np.float32)
    held = cp.asarray(tend)
    ss._kernel("smoke_profile_source")(
        grid, (128,), (held, cp.asarray(emis), cp.asarray(f),
                       cp.asarray(dnw), UG_G, np.int32(nz), np.int32(n)))
    _same(held, ss.profile_source_reference(tend, emis, f, dnw), "source")

    # deposition inputs (MODIS mapped), the cap, the apply, settling apply
    t0 = rng.uniform(250, 310, (ny, nx)).astype(np.float32)
    rho0 = rng.uniform(0.9, 1.3, (ny, nx)).astype(np.float32)
    rmol = rng.uniform(-0.1, 0.1, (ny, nx)).astype(np.float32)
    rmol[0, 0] = 0.0
    lu = rng.integers(0, 23, (ny, nx)).astype(np.int32)
    outs = {name: cp.empty((ny, nx), cp.float32) for name in (
        "dg_num", "lnsig", "dens", "airtemp", "airdens", "obukhov",
        "kinvisc", "freepath")}
    ints = {name: cp.empty((ny, nx), cp.int32)
            for name in ("season", "surftype", "wetsurf")}
    ss._kernel("smoke_drydep_prepare")(
        grid, (128,), (cp.asarray(t0), cp.asarray(rho0), cp.asarray(rmol),
                       cp.asarray(lu), np.int32(1), np.int32(2),
                       np.float32(1.46e-6), np.float32(2500.0),
                       outs["dg_num"], outs["lnsig"], outs["dens"],
                       outs["airtemp"], outs["airdens"], outs["obukhov"],
                       ints["season"], ints["surftype"], ints["wetsurf"],
                       outs["kinvisc"], outs["freepath"], np.int32(n)))
    ref = ss.drydep_prepare_reference(t0, rho0, rmol, lu, modis=True,
                                      iseason=2, diameter=1.46e-6,
                                      density=2500.0)
    for name, value in {**outs, **ints}.items():
        _same(value, ref[name], name)

    vel = rng.uniform(0, 0.9, (ny, nx)).astype(np.float32)
    capped = cp.empty((ny, nx), cp.float32)
    ss._kernel("smoke_drydep_cap")(grid, (128,),
                                   (cp.asarray(vel), capped, np.int32(n)))
    _same(capped, ss.drydep_cap_reference(vel), "cap")

    c1h = rng.uniform(0.5, 1, nz).astype(np.float32)
    c2h = rng.uniform(0, 5000, nz).astype(np.float32)
    mub = rng.uniform(8e4, 9e4, (ny, nx)).astype(np.float32)
    mup = rng.normal(0, 300, (ny, nx)).astype(np.float32)
    smoke = rng.uniform(0, 500, (nz, ny, nx)).astype(np.float32)
    tendency = -rng.uniform(0, 10, (ny, nx)).astype(np.float32)
    tendency[0, 1] = -1.0e6
    dep = rng.uniform(0, 100, (ny, nx)).astype(np.float32)
    dev_smoke, dev_dep = cp.asarray(smoke), cp.asarray(dep)
    ss._kernel("smoke_drydep_apply")(
        grid, (128,), (dev_smoke, cp.asarray(tendency), cp.asarray(c1h),
                       cp.asarray(c2h), cp.asarray(dnw), cp.asarray(mub),
                       cp.asarray(mup), dev_dep, np.float32(10.0),
                       ss._G, np.int32(n)))
    q0, rdep = ss.drydep_apply_reference(smoke[0], tendency, c1h, c2h, dnw,
                                         mub, mup, dep, 10.0)
    _same(dev_smoke[0], q0, "drydep q0")
    _same(dev_smoke[1:], smoke[1:], "drydep aloft untouched")
    _same(dev_dep, rdep, "drydep booked")

    tc = (smoke.astype(np.float64)
          * rng.uniform(0.98, 1.0, smoke.shape))
    graset = rng.uniform(0, 5, (ny, nx)).astype(np.float32)
    dev_smoke, dev_dep, dev_gr = (cp.asarray(smoke), cp.asarray(dep),
                                  cp.asarray(graset))
    ss._kernel("smoke_settling_apply")(
        grid, (128,), (dev_smoke, cp.asarray(tc), cp.asarray(c1h),
                       cp.asarray(c2h), cp.asarray(dnw), cp.asarray(mub),
                       cp.asarray(mup), dev_dep, dev_gr, ss._G,
                       np.int32(nz), np.int32(n)))
    rs, rd, rg = ss.settling_apply_reference(smoke, tc, c1h, c2h, dnw, mub,
                                             mup, dep, graset)
    _same(dev_smoke, rs, "settled")
    _same(dev_dep, rd, "settling booked")
    _same(dev_gr, rg, "graset")


@pytest.mark.gpu
@requires_gpu
def test_surface_profile_is_the_surface_source_word_for_word():
    """A (1, 0, ..., 0) profile adds smoke_tracer's own words."""
    import cupy as cp

    from woof.core import smoke_tracer

    cfg = _cfg(smoke_tracer=True, smoke_injection="plumerise")
    state = _state(cfg)
    rng = np.random.default_rng(5)
    state.smoke_emis[...] = cp.asarray(rng.uniform(0, 1e-7, (8, 12)),
                                       cp.float32)
    tend = cp.asarray(rng.normal(0, 1, state.p.shape), cp.float32)
    expected = tend.copy()
    smoke_tracer.add_surface_source(state, expected)
    state.smoke_inj_frac[...] = 0
    state.smoke_inj_frac[0] = 1.0
    ss._kernel("smoke_profile_source")(
        ss._blocks(96), (128,),
        (tend, state.smoke_emis, state.smoke_inj_frac, state.dnw,
         smoke_tracer.UG_G, np.int32(cfg.nz), np.int32(96)))
    _same(tend, expected, "surface profile")


@pytest.mark.gpu
@requires_gpu
def test_pbl_profile_is_the_library_profile_word_for_word():
    import cupy as cp

    from woof.smoke.injection import pbl_uniform_profile

    cfg = _cfg(smoke_tracer=True, smoke_injection="pbl", bl_pbl_physics=1)
    state = _state(cfg)
    fields = _fake_physics(state)
    frac = ss.run_pbl_profile(state, cfg)
    zw = state.scratch((cfg.nz + 1, 8, 12), "smoke_src_zw")
    ones = cp.ones(state.p.shape, cp.float32)
    library = pbl_uniform_profile(zw, ones, 1.0, fields["pblh"])
    _same(state.scratch(state.p.shape, "smoke_src_pblsrc"), library,
          "pbl source")
    rf, rt = ss.pbl_fraction_reference(cp.asnumpy(library), cp.asnumpy(zw))
    _same(frac, rf, "pbl frac")
    _same(state.smoke_inj_top, rt, "pbl top")
    np.testing.assert_allclose(cp.asnumpy(frac).sum(axis=0), 1.0, rtol=2e-6)


def _plume_tables(heat):
    table = np.full((2, 4), np.float32(heat) / np.float32(1000.0),
                    dtype=np.float32)
    return table


def _library_plume_column(env, j, i, heat, area, flaming, *, numpy=False):
    import cupy as cp

    from woof.smoke.plume_rise import plumerise, plumerise_np

    nz = env["u"].shape[0]
    cols = [cp.asnumpy(env[name][:, j:j + 1, i:i + 1]) for name in (
        "u", "v", "theta", "pi", "qv", "zt")]
    zm = cp.asnumpy(env["zw"][:nz, j:j + 1, i:i + 1])
    firesize = np.zeros((4, 1, 1), np.float32)
    firesize[3] = area
    mean_fct = np.zeros((4, 1, 1), np.float32)
    mean_fct[3] = flaming
    eburn_in = np.ones((1, 1), np.float32)
    run = plumerise_np if numpy else plumerise
    out = run(*cols, zm, firesize, mean_fct, eburn_in,
              heat_flux=_plume_tables(heat))
    return out


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("source", ["scalars", "fire_seam", "file"])
def test_plume_profile_is_the_library_plumerise_word_for_word(source,
                                                             tmp_path):
    """Each burning column's injection is WRF-Chem's plumerise, on its own
    heat flux, and the cached profile is WRF's scale_fire_emiss conversion
    of that output."""
    import cupy as cp

    burning = [(2, 3), (5, 9), (6, 1)]
    heat = {(2, 3): 3.0e4, (5, 9): 8.0e4, (6, 1): 4.4e3}
    area = {(2, 3): 2.0e4, (5, 9): 5.0e5, (6, 1): 1.0e3}
    kw = dict(smoke_tracer=True, smoke_injection="plumerise", moist=True,
              smoke_plume_flaming_fraction=0.7)
    if source == "scalars":
        _emission(tmp_path / "e.npy", 8, 12, burning)
        kw.update(smoke_emission_file=str(tmp_path / "e.npy"),
                  smoke_plume_heat_flux=3.0e4, smoke_plume_fire_area=2.0e4)
        heat = {c: 3.0e4 for c in burning}
        area = {c: 2.0e4 for c in burning}
    elif source == "file":
        from netCDF4 import Dataset
        with Dataset(tmp_path / "e.nc", "w") as ds:
            ds.createDimension("south_north", 8)
            ds.createDimension("west_east", 12)
            planes = {"smoke_emission": np.zeros((8, 12)),
                      "fire_heat_flux": np.zeros((8, 12)),
                      "fire_area": np.zeros((8, 12))}
            for c in burning:
                planes["smoke_emission"][c] = 1e-8
                planes["fire_heat_flux"][c] = heat[c]
                planes["fire_area"][c] = area[c]
            for name, value in planes.items():
                var = ds.createVariable(name, "f4",
                                        ("south_north", "west_east"))
                var[:] = value
        kw.update(smoke_emission_file=str(tmp_path / "e.nc"))
    cfg = _cfg(**kw)
    state = _state(cfg, u0=5.0, noise=True)
    state.qv[...] = cp.float32(0.006) * cp.exp(
        -cp.arange(cfg.nz, dtype=cp.float32) / 8.0)[:, None, None]
    if source == "fire_seam":
        # The fire model writes the emission and its fire beside it.
        for c in burning:
            state.smoke_emis[c] = cp.float32(1.0e-8)
        hf = np.zeros((8, 12), np.float32)
        ar = np.zeros((8, 12), np.float32)
        for c in burning:
            hf[c], ar[c] = heat[c], area[c]
        state.fire_heat_flux = cp.asarray(hf)
        state.fire_area = cp.asarray(ar)
    from woof.core import smoke_tracer
    smoke_tracer.bind_emission(state, cfg)
    ss.run_plume(state, cfg)
    env = {name: state.scratch(shape, f"smoke_src_{name}") for name, shape in
           (("u", state.p.shape), ("v", state.p.shape),
            ("theta", state.p.shape), ("pi", state.p.shape),
            ("qv", state.p.shape), ("zt", state.p.shape),
            ("dz", state.p.shape), ("zw", (cfg.nz + 1, 8, 12)))}
    eburn = cp.asnumpy(state.scratch(state.p.shape, "smoke_src_eburn"))
    k2 = cp.asnumpy(state.scratch((8, 12), "smoke_src_k2", dtype=np.int32))
    for c in burning:
        lib = _library_plume_column(env, *c, heat[c], area[c], 0.7)
        _same(eburn[:, c[0], c[1]], cp.asnumpy(lib.eburn_out)[:, 0, 0],
              f"eburn {c}")
        assert k2[c] == int(cp.asnumpy(lib.k2)[3, 0, 0])
    # The CPU reference of the library, for the first column.
    ref = _library_plume_column(env, *burning[0], heat[burning[0]],
                                area[burning[0]], 0.7, numpy=True)
    _same(eburn[:, burning[0][0], burning[0][1]], ref.eburn_out[:, 0, 0],
          "eburn vs plumerise_np")
    # Every other column keeps the smouldering (surface) profile.
    mask = np.ones((8, 12), bool)
    for c in burning:
        mask[c] = False
    assert (eburn[0][mask] == 1.0).all() and (eburn[1:][:, mask] == 0).all()
    rf, rt = ss.plume_fraction_reference(
        eburn, cp.asnumpy(env["dz"]), cp.asnumpy(env["zw"]),
        np.zeros((8, 12), np.int32))
    _same(state.smoke_inj_frac, rf, "frac")
    _same(state.smoke_inj_top, rt, "inj_top")
    assert np.isfinite(rt).all()
    top = cp.asnumpy(state.smoke_inj_top)
    assert min(top[c] for c in burning) > top[mask].max()
    if source != "scalars":
        assert top[burning[1]] > top[burning[0]]


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("opt,dataset", [(301, "USGS"), (312, "MODIS")])
def test_dry_deposition_is_the_library_routines_word_for_word(opt, dataset):
    import cupy as cp

    from woof.smoke.deposition import (aerodynamic_resistance,
                                       surface_deposition_tendency,
                                       zhang2001_depvel)

    cfg = _cfg(smoke_tracer=True, smoke_drydep=True, sf_sfclay_physics=1,
               smoke_aer_drydep_opt=opt, moist=True,
               smoke_particle_radius=0.3e-6, smoke_particle_density=1350.0)
    state = _state(cfg)
    rng = np.random.default_rng(9)
    state.smoke[...] = cp.asarray(rng.uniform(0, 300, state.smoke.shape),
                                  cp.float32)
    state.qv[...] = cp.float32(0.004)
    name = "USGS" if dataset == "USGS" else "MODIFIED_IGBP_MODIS_NOAH"
    f = _fake_physics(state, dataset=name)
    before = cp.asnumpy(state.smoke).copy()
    env = ss._env(state, sinks=True)
    ss._apply_drydep(state, cfg, env)
    host = lambda a: cp.asnumpy(a)
    lu = host(f["ivgtyp"])
    if dataset != "USGS":
        lu = ss.drydep_prepare_reference(
            host(env["t"][0]), host(env["rho"][0]), None, lu, modis=True,
            iseason=1, diameter=0.0, density=0.0)["surftype"]
    ra = aerodynamic_resistance(f["ust"], f["znt"], env["zref"],
                                obukhov_length=1.0 / f["rmol"])
    _same(state.scratch((8, 12), "smoke_src_resist_a"), ra, "resist_a")
    season = ss.zhang_season(opt, datetime(2026, 8, 1, 12),
                             state.elapsed_seconds)
    z = zhang2001_depvel(np.float32(0.6e-6), 0.0, 1350.0, env["t"][0],
                         env["rho"][0], f["ust"], ra, isurftype=lu,
                         iseason=season, iwetsurf=0, modesurftype=2,
                         zhang_optaa=ss.zhang_optaa(opt))
    _same(state.scratch((8, 12), "smoke_src_depvel3"), z.depvel_3, "depvel3")
    capped = ss.drydep_cap_reference(host(z.depvel_3))
    _same(state.smoke_drydep_vel, capped, "SMOKE_DRYDEP_VEL")
    lib = surface_deposition_tendency(before[0], env["dz"][0], capped,
                                      env["rho"][0])
    _same(state.scratch((8, 12), "smoke_src_drydep_tend"), lib.tendency,
          "tendency")
    q0, dep = ss.drydep_apply_reference(
        before[0], host(lib.tendency), host(state.c1h), host(state.c2h),
        host(state.dnw), host(state.mub2d), host(state.mup),
        np.zeros((8, 12), np.float32), cfg.dt)
    _same(state.smoke[0], q0, "lowest layer")
    _same(state.smoke[1:], before[1:], "aloft")
    _same(state.smoke_deposited, dep, "SMOKE_DEPOSITED")
    assert float(host(state.smoke_deposited).min()) >= 0.0
    assert float(host(state.smoke_deposited).max()) > 0.0


@pytest.mark.gpu
@requires_gpu
def test_settling_is_the_library_gocart_settling_word_for_word():
    import cupy as cp

    from woof.smoke.deposition import gocart_settling

    cfg = _cfg(smoke_tracer=True, smoke_settling=True, moist=True,
               smoke_particle_radius=5.0e-6)
    state = _state(cfg)
    rng = np.random.default_rng(13)
    state.smoke[...] = cp.asarray(rng.uniform(0, 300, state.smoke.shape),
                                  cp.float32)
    state.qv[...] = cp.float32(0.003)
    before = cp.asnumpy(state.smoke).copy()
    env = ss._env(state, sinks=True)
    ss._apply_settling(state, cfg, env)
    lib = gocart_settling(before, env["t"], state.p, env["dz"], env["rho"],
                          cfg.dt, 5.0e-6, 2500.0)
    host = lambda a: cp.asnumpy(a)
    _same(state.smoke, host(lib.tc).astype(np.float32), "settled")
    rs, rd, rg = ss.settling_apply_reference(
        before, host(lib.tc), host(state.c1h), host(state.c2h),
        host(state.dnw), host(state.mub2d), host(state.mup),
        np.zeros((8, 12), np.float32), np.zeros((8, 12), np.float32))
    _same(state.smoke_deposited, rd, "SMOKE_DEPOSITED")
    _same(state.smoke_graset, rg, "SMOKE_GRASET")
    assert float(host(state.smoke_graset).min()) > 0.0


def _budget(state, emitted_ug_m2):
    import cupy as cp

    col = cp.asnumpy(state.smoke_col).astype(np.float64).sum()
    dep = cp.asnumpy(state.smoke_deposited).astype(np.float64).sum()
    return col, dep, (col + dep) / emitted_ug_m2


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("u0", [0.0, 8.0])
def test_mass_budget_closes(tmp_path, u0):
    """emitted = airborne + deposited (+ outflow, zero on a periodic
    domain), each step, through plume-rise injection, transport, dry
    deposition and settling."""
    cells = [(2, 3), (5, 9)]
    field = _emission(tmp_path / "e.npy", 8, 12, cells, flux=2.0e-7)
    cfg = _cfg(smoke_tracer=True, smoke_emission_file=str(tmp_path / "e.npy"),
               smoke_injection="plumerise", smoke_plume_heat_flux=2.0e4,
               smoke_plume_fire_area=5.0e4, smoke_plumerise_interval_s=40.0,
               smoke_drydep=True, smoke_settling=True,
               smoke_aer_drydep_opt=302, smoke_particle_radius=4.0e-6,
               km_opt=4 if u0 else 0)
    state = _state(cfg, u0=u0)
    _fake_physics(state)
    from woof.core.dycore import step
    for n in range(1, 13):
        step(state, cfg)
        emitted = field.sum() * 1.0e9 * n * cfg.dt
        col, dep, ratio = _budget(state, emitted)
        assert abs(ratio - 1.0) < (2e-5 if u0 else 2e-6), (n, ratio)
    assert dep > 0.0 and col > 0.0


@pytest.mark.gpu
@requires_gpu
def test_checkpoint_resumes_bit_for_bit(tmp_path):
    import cupy as cp

    from woof.core.dycore import step
    from woof.io.restart import restore_restart, write_restart

    _emission(tmp_path / "e.npy", 8, 12, [(3, 4), (6, 8)], flux=1.0e-7)
    cfg = _cfg(smoke_tracer=True, smoke_emission_file=str(tmp_path / "e.npy"),
               smoke_injection="plumerise", smoke_plume_heat_flux=3.0e4,
               smoke_plume_fire_area=1.0e5, smoke_plumerise_interval_s=70.0,
               smoke_drydep=True, smoke_settling=True,
               smoke_aer_drydep_opt=302, km_opt=4)

    def fresh():
        state = _state(cfg, u0=4.0)
        _fake_physics(state)
        return state

    straight = fresh()
    for _ in range(9):
        step(straight, cfg)
    split = fresh()
    for _ in range(4):
        step(split, cfg)
    physics = split.physics
    split.physics = None
    path = write_restart(tmp_path / "split.npz", split, cfg)
    resumed = fresh()
    resumed.physics = None
    restore_restart(path, resumed, cfg)
    resumed.physics = physics
    assert not ss.plume_due(resumed, cfg)        # resumes between calls
    for _ in range(5):
        step(resumed, cfg)
    for name in ("smoke", "smoke_col", "smoke_inj_frac", "smoke_inj_top",
                 "smoke_drydep_vel", "smoke_deposited", "smoke_graset"):
        _same(getattr(resumed, name), cp.asnumpy(getattr(straight, name)),
              name)


@pytest.mark.gpu
@requires_gpu
def test_sources_and_sinks_add_no_device_to_host_copy(tmp_path, monkeypatch):
    import cupy as cp

    from woof.core.dycore import step

    _emission(tmp_path / "e.npy", 8, 12, [(3, 4)], flux=1.0e-8)
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

    def per_step(**kw):
        cfg = _cfg(smoke_tracer=True,
                   smoke_emission_file=str(tmp_path / "e.npy"), **kw)
        state = _state(cfg, u0=3.0)
        _fake_physics(state)
        step(state, cfg)
        counts["n"] = 0
        for _ in range(4):
            step(state, cfg)
        return counts["n"]

    off = per_step()
    on = per_step(smoke_injection="plumerise", smoke_plume_heat_flux=3.0e4,
                  smoke_plume_fire_area=1.0e5, smoke_plumerise_interval_s=20.0,
                  smoke_drydep=True, smoke_settling=True,
                  smoke_aer_drydep_opt=302)
    # (Unvalidated: the stand-in physics publishes the PBL height.)
    pbl = per_step(smoke_injection="pbl")
    assert on == off == pbl, (on, off, pbl)


@pytest.mark.gpu
@requires_gpu
def test_the_coupled_fire_drives_the_plume_through_its_own_fields():
    """ifire = 2 writes state.fire_heat_flux, state.fire_area and
    state.smoke_emis every step; the plume runs on exactly those words,
    each burning column equal to the library's plumerise on its own heat
    flux and area, and the emitted mass enters the tracer conserved."""
    import cupy as cp

    from test_fire_coupling import _fire, _state as _fire_state
    from woof.core.dycore import step

    cfg = _fire(smoke_tracer=True, smoke_injection="plumerise",
                smoke_plumerise_interval_s=0.0, nz=30, ztop=4000.0)
    validate_run_config(cfg)
    state = _fire_state(cfg)
    emitted = 0.0
    burned = 0
    for n in range(60):                           # 30 s, every step a plume
        step(state, cfg)
        emitted += float(cp.asnumpy(state.smoke_emis).astype(np.float64)
                         .sum()) * 1.0e9 * cfg.dt
        if n < 40:
            continue
        heat = cp.asnumpy(state.fire_heat_flux)
        area = cp.asnumpy(state.fire_area)
        emis = cp.asnumpy(state.smoke_emis)
        fire = np.argwhere((heat > 0) & (area >= 1e-6) & (emis > 0))
        nz, ny, nx = state.p.shape
        env = {name: state.scratch((nz + 1 if name == "zw" else nz, ny, nx),
                                   f"smoke_src_{name}")
               for name in ("u", "v", "theta", "pi", "qv", "zt", "zw")}
        eburn = cp.asnumpy(state.scratch((nz, ny, nx), "smoke_src_eburn"))
        status = cp.asnumpy(state.scratch((ny, nx), "smoke_src_status",
                                          dtype=np.int32))
        assert not status.any()
        for j, i in fire[:3]:
            lib = _library_plume_column(env, j, i, heat[j, i], area[j, i],
                                        1.0)
            _same(eburn[:, j, i], cp.asnumpy(lib.eburn_out)[:, 0, 0],
                  f"fire column {(j, i)} step {n}")
            burned += 1
        quiet = (heat == 0) | (emis == 0)
        assert (eburn[0][quiet] == 1.0).all()
    assert burned > 0
    top = cp.asnumpy(state.smoke_inj_top)
    assert np.isfinite(top).all()

    def airborne(s):
        layer = ((s.c1h[:, None, None] * (s.mub2d + s.mup)[None]
                  + s.c2h[:, None, None]) * (-s.dnw[:, None, None]))
        return float(cp.asnumpy((s.smoke.astype(cp.float64)
                                 * layer.astype(cp.float64)).sum())) / 9.81

    # The same fire with the surface profile carries the same smoke mass:
    # the plume moves the emission up, it does not make or lose any.  (Both
    # carry ~8% more than the emission tally on this open LES box, a
    # property of the fire coupling's source or the open-boundary
    # transport, not of the profile.)
    surface = _fire_state(replace(cfg, smoke_injection="surface"))
    for _ in range(60):
        step(surface, replace(cfg, smoke_injection="surface"))
    ratio = airborne(state) / airborne(surface)
    assert abs(ratio - 1.0) < 1e-3, ratio
    assert emitted > 0.0


@pytest.mark.gpu
@requires_gpu
def test_a_plume_with_no_fire_is_refused_by_name(tmp_path):
    from netCDF4 import Dataset

    with Dataset(tmp_path / "e.nc", "w") as ds:
        ds.createDimension("south_north", 8)
        ds.createDimension("west_east", 12)
        ds.createVariable("smoke_emission", "f4",
                          ("south_north", "west_east"))[:] = 1e-8
    cfg = _cfg(smoke_tracer=True, smoke_injection="plumerise",
               smoke_emission_file=str(tmp_path / "e.nc"))
    validate_run_config(cfg)
    state = _state(cfg)
    with pytest.raises(RuntimeError, match="no fire to raise a plume"):
        ss.run_plume(state, cfg)
    # The file was read once and remembered as carrying no plume fields.
    assert getattr(state, "_smoke_plume_bound") == (str(tmp_path / "e.nc"),
                                                    False)


@pytest.mark.gpu
@requires_gpu
def test_a_column_the_plume_model_refuses_is_named_and_emits_at_the_surface():
    import cupy as cp

    from woof.core.health import rule_for_field

    status = np.zeros((2, 2), np.int32)
    status[1, 0] = 1
    eburn = np.zeros((4, 2, 2), np.float32)
    eburn[0] = 1.0
    eburn[2, 1, 0] = 0.01
    frac, top = ss.plume_fraction_reference(
        eburn, np.full((4, 2, 2), 100.0, np.float32),
        np.arange(5, dtype=np.float32)[:, None, None] * np.ones((5, 2, 2),
                                                                np.float32)
        * 100.0, status)
    assert frac[0, 1, 0] == 1.0 and frac[1:, 1, 0].sum() == 0.0
    assert np.isnan(top[1, 0])
    rule = rule_for_field("smoke_inj_top")
    assert rule.status_class == "tracer"
    del cp


# ------------------------------------------------------------- namelist ---

def _namelist_pair_texts():
    """The importer battery's synthetic two-domain pair, read from
    tests/test_namelist_import.py without importing it (its module pulls
    in the reference bundle's case module)."""
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).parent
                      / "test_namelist_import.py").read_text())
    texts = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("WPS_TEXT", "INPUT_TEXT")):
            texts[node.targets[0].id] = ast.literal_eval(node.value)
    return texts["WPS_TEXT"], texts["INPUT_TEXT"]


def _import_with(tmp_path, extra_input="", extra_dynamics=""):
    from woof.namelist_import import import_namelists

    wps, inp = _namelist_pair_texts()
    if extra_dynamics:
        inp = inp.replace(" hybrid_opt = 2,",
                          " hybrid_opt = 2,\n" + extra_dynamics)
    (tmp_path / "namelist.wps").write_text(wps)
    (tmp_path / "namelist.input").write_text(inp + extra_input)
    return import_namelists(tmp_path / "namelist.wps",
                            tmp_path / "namelist.input")


def _chem_import(tmp_path, body):
    """``body``'s &chem keys, with its tracer_opt line in &dynamics, where
    WRF declares it (Registry.EM_COMMON)."""
    lines = body.splitlines(keepends=True)
    tracer = "".join(line for line in lines if "tracer_opt" in line)
    chem = "".join(line for line in lines if "tracer_opt" not in line)
    return _import_with(tmp_path, extra_input="&chem\n" + chem + "/\n",
                        extra_dynamics=tracer)


def test_wrf_chem_keys_import(tmp_path):
    from woof.experiment import load_experiment

    toml_text, report = _chem_import(tmp_path, (
        " chem_opt = 0, 0,\n tracer_opt = 1, 1,\n"
        " biomass_burn_opt = 1, 1,\n plumerisefire_frq = 30, 30,\n"
        " scale_fire_emiss = .true., .true.,\n aer_drydep_opt = 302, 302,\n"
        " chemdt = 10, 10,\n vertmix_onoff = 1, 1,\n"))
    assert toml_text.count("smoke_tracer = true") == 1
    for line in ('smoke_tracer = true', 'smoke_injection = "plumerise"',
                 'smoke_plumerise_interval_s = 1800.0',
                 'smoke_drydep = true', 'smoke_aer_drydep_opt = 302'):
        assert line in toml_text, line
    assert {d.key for d in report.dropped if d.section == "chem"} == {
        "chemdt", "scale_fire_emiss"}
    assert any("aer_drydep_opt = 302" in n for n in report.notices)
    (tmp_path / "chem.toml").write_text(toml_text)
    experiment = load_experiment(tmp_path / "chem.toml")
    for dc in experiment.domains:
        assert dc.run.smoke_injection == "plumerise"
        assert dc.run.smoke_plumerise_interval_s == 1800.0
        assert dc.run.smoke_drydep and dc.run.smoke_aer_drydep_opt == 302


def test_a_namelist_without_chem_imports_unchanged(tmp_path):
    toml_text, _ = _import_with(tmp_path)
    assert "smoke_" not in toml_text
    # aer_drydep_opt at its Registry default is inert under chem_opt = 0.
    toml_text, report = _chem_import(tmp_path, " aer_drydep_opt = 1, 1,\n")
    assert "smoke_" not in toml_text
    assert "aer_drydep_opt" in {d.key for d in report.dropped}


@pytest.mark.parametrize("body, key", [
    (" chem_opt = 1, 1,\n", "chem_opt"),
    (" tracer_opt = 2, 2,\n", "tracer_opt"),
    (" tracer_opt = 1, 1,\n aer_drydep_opt = 101, 101,\n", "aer_drydep_opt"),
    # An omitted tail keeps the Registry default (0 for tracer_opt), so a
    # single value is d01's alone, and [shared] settings are refused.
    (" tracer_opt = 1,\n", "tracer_opt"),
    (" tracer_opt = 1, 1,\n biomass_burn_opt = 1, 1,\n", "scale_fire_emiss"),
    (" tracer_opt = 1, 1,\n biomass_burn_opt = 1, 1,\n"
     " scale_fire_emiss = .false., .false.,\n", "scale_fire_emiss"),
    (" biomass_burn_opt = 1, 1,\n scale_fire_emiss = .true., .true.,\n",
     "biomass_burn_opt"),
    (" tracer_opt = 1, 1,\n vertmix_onoff = 0, 0,\n", "vertmix_onoff"),
    (" tracer_opt = 1, 1,\n kemit = 19,\n", "kemit"),
])
def test_wrf_chem_keys_refused_by_name(tmp_path, body, key):
    from woof.namelist_import import NamelistRefusal

    with pytest.raises(NamelistRefusal, match=key):
        _chem_import(tmp_path, body)
