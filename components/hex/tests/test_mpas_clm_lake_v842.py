"""WRF's CLM lake on the MPAS-A v8.4.2 path (woof.hex.mpasphys.clm_lake_v842).

* the physics driver runs ``clm_lake_init`` once, at the start of the first step
  before any surface scheme, and ``clm_lake`` on every land-surface step after the
  land surface, sea ice and surface diagnostics and before the PBL, only when
  ``config_clm_lake_backend = 'hex-cuda'``; without it nothing runs and the
  receipt is unchanged;
* the configurations whose lake words the route does not carry are refused by name;
* the glue reads MPAS's own words on R2-shaped pools (owned cells first, halo
  after) and every lake call IS WRF's lake kernel on the same inputs: the
  service's columns are replayed through woof.core.lake (lakeini, then Lake) on a
  WRF-shaped (2, 4) grid from inputs formed here independently of the service
  (MPAS's lsm_from_MPAS SWDOWN and RAINBL), and the initial lake state, every
  step's lake state, every surface output and the lakeini water hand-back agree
  bit for bit; halo cells and non-lake cells are never written.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys.driver_v842 import (PhysicsDriver, PhysicsManager,
                                           SlotRegistry, resolve_namelist)

ON = "\n&hex_mpasphys\nconfig_clm_lake_backend = 'hex-cuda'\n/\n"


def deck(suite='mesoscale_reference', **options):
    values = {'config_physics_suite': suite, **options}
    return '&physics\n' + '\n'.join(f'{k} = {v!r}' for k, v in values.items()) + '\n/'


def _driver(text):
    config = resolve_namelist(text)

    class Hooks:
        def __init__(self):
            self.operations = []

        def invoke(self, operation, call):
            self.operations.append(operation)

        def restart_state(self):
            return {}

        def restore_restart_state(self, payload):
            pass
    hooks = Hooks()
    registry = SlotRegistry()
    for slot in config.selectors:
        if config.active(slot):
            registry.register(slot, config.selectors[slot], config.backends[slot],
                              lambda call, h=hooks: h.operations.append(call.slot))
    driver = PhysicsDriver(config, PhysicsManager(config, 360), registry, hooks)
    driver.initialize(None)
    hooks.operations.clear()
    return config, driver, hooks


def test_driver_runs_the_lake_after_the_land_surface_and_before_the_pbl():
    config, driver, hooks = _driver(deck() + ON)
    for step, time in ((1, 0), (2, 360), (3, 720)):
        hooks.operations.clear()
        driver.begin_step(itimestep=step, time_seconds=time, pools=None)
        ops = hooks.operations
        assert ops.count("clm_lake_init") == (step == 1)
        if step == 1:
            assert ops.index("clm_lake_init") < ops.index("allocate_forall_physics")
            assert ops.index("clm_lake_init") < ops.index("surface_layer")
        assert ops.count("clm_lake") == 1
        at = ops.index("clm_lake")
        assert ops[at - 1] == "atmphys_sfc_diagnostics"
        assert ops[at + 1] == "allocate_pbl"
        assert ops.index("land_surface") < at
        for rk in (1, 2, 3):
            driver.physics_get_tend(rk_step=rk, dynamics_substep=1, pools=None)
        driver.finish_step(pools=None)
    call = next(c for c in driver.calls if c["operation"] == "clm_lake")
    assert call["backend"] == "hex-cuda"
    assert call["time_seconds"] == 720
    assert config.receipt()["service_backends"]["clm_lake"] == "hex-cuda"


def test_without_the_service_nothing_runs_and_the_receipt_is_unchanged():
    config, driver, hooks = _driver(deck())
    driver.begin_step(itimestep=1, time_seconds=0, pools=None)
    assert not {"clm_lake", "clm_lake_init"} & set(hooks.operations)
    assert "service_backends" not in config.receipt()
    with pytest.raises(ValueError, match="unsupported clm_lake service backend"):
        resolve_namelist(deck() + "\n&hex_mpasphys\nconfig_clm_lake_backend = 'native-cpu'\n/\n")


@pytest.mark.parametrize("options, match", [
    (dict(config_lsm_scheme='off'), "no land surface"),
    (dict(config_sst_update=True), "config_sst_update"),
    (dict(config_sstdiurn_update=True), "config_sstdiurn_update"),
    (dict(config_oml1d=True), "ocean mixed layer"),
    (dict(config_pbl_interval='00:30:00'), "config_pbl_interval"),
])
def test_configurations_the_lake_does_not_carry_are_refused_by_name(options, match):
    resolve_namelist(deck(**options))      # each runs without the lake
    with pytest.raises(ValueError, match=match):
        resolve_namelist(deck(**options) + ON)


def test_routes_without_a_lake_carrier_are_refused_by_name():
    from woof.hex.mpasphys.clm_lake_v842 import refusal
    config = resolve_namelist(deck() + ON)
    layout = SimpleNamespace(configs={"config_do_restart": False})
    plain = SimpleNamespace(samehost_devices=None)
    assert refusal(plain, config, layout, moving=None, execution=None) is None
    restart = SimpleNamespace(configs={"config_do_restart": True})
    assert "native-file restart" in refusal(plain, config, restart, moving=None, execution=None)
    assert "moving" in refusal(plain, config, layout, moving=object(), execution=None)
    assert "same-host" in refusal(plain, config, layout, moving=None, execution=object())
    assert "same-host" in refusal(SimpleNamespace(samehost_devices="0,1"), config, layout,
                                  moving=None, execution=None)


def _sidecar(path, lat, lon, mask, depth):
    import netCDF4
    from woof.hex import surface_inputs as si
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("nCells", lat.size)
        dataset.createVariable("LAKEMASK", "i4", ("nCells",))[:] = mask
        dataset.createVariable("LAKE_DEPTH", "f4", ("nCells",))[:] = depth
        dataset.setncattr("LAKE_DEPTH_FLAG", np.int32(1))
        dataset.setncattr("LAKEFLAG", np.int32(1))
        dataset.setncattr("schema", si.SCHEMA)
        dataset.createVariable("latCell", "f8", ("nCells",))[:] = lat
        dataset.createVariable("lonCell", "f8", ("nCells",))[:] = np.mod(lon, 2 * np.pi)
        dataset.setncattr("provenance_json", "{}")
    return path


def test_statics_come_from_the_mesh_sidecar_and_are_refused_without_it(tmp_path):
    from woof.hex.mpasphys.clm_lake_v842 import bind_statics
    lat = np.radians(np.linspace(40.0, 41.0, 6))
    lon = np.radians(np.linspace(-84.0, -83.0, 6))
    mesh = SimpleNamespace(arrays={"latCell": lat, "lonCell": lon})
    layout = SimpleNamespace(dimensions={"nCellsSolve": 5})
    native = SimpleNamespace()
    arguments = SimpleNamespace(surface_statics=None, grid=str(tmp_path / "m.grid.nc"), mesh="m",
                                static=None, init=None)
    with pytest.raises(ValueError, match="woof hex surface-statics"):
        bind_statics(native, mesh, layout, arguments)
    _sidecar(tmp_path / "m.surface_statics.nc", lat, lon, [1, 0, 1, 0, 0, 1], [12.5, 10, 3.25, 10, 10, 9])
    receipt = bind_statics(native, mesh, layout, arguments)
    assert receipt["lake_cells"] == 2           # owned cells only
    np.testing.assert_array_equal(native.clm_lake_statics["lakemask"], np.float32([1, 0, 1, 0, 0]))
    np.testing.assert_array_equal(native.clm_lake_statics["lake_depth"], np.float32([12.5, 10, 3.25, 10, 10]))
    assert native.clm_lake_statics["lake_depth_flag"] == 1
    other = SimpleNamespace(arrays={"latCell": lat + 0.01, "lonCell": lon})
    with pytest.raises(Exception, match="other cells"):
        bind_statics(native, other, layout, arguments)


# ---------------------------------------------------------------------------------------------------------------
# E1: every lake call on R2-shaped pools is WRF's lake kernel on the same inputs.
# ---------------------------------------------------------------------------------------------------------------
GRID = (2, 4)
NL, NC, HALO = 6, 8, 3
LAKES = np.array([1, 0, 0, 1, 0, 1, 1, 0], dtype=np.float32)
DEPTH = np.array([12.5, 10.0, 10.0, 3.25, 10.0, 41.0, 160.0, 10.0], np.float32)
#: one frozen lake (MPAS's init made it land ice) that lakeini hands back to lake water.
FROZEN = 5
ISWATER, ISICE = 17, 15
OUTPUT_FIELDS = (("diag_physics", "hfx"), ("diag_physics", "lh"), ("diag_physics", "grdflx"),
                 ("sfc_input", "skintemp"), ("diag_physics", "qfx"), ("diag_physics", "t2m"),
                 ("diag_physics", "th2m"), ("diag_physics", "q2"), ("diag_physics", "sfc_albedo"))


class _Provider:
    """The device pool views' surface the lake glue reads: owned cells first, then halo (NaN, never read)."""

    def __init__(self, cp, rng):
        self.cp = cp
        n = NC + HALO

        def vec(lo, hi, dtype=np.float32):
            a = np.full(n, np.nan, np.float32) if dtype == np.float32 else np.full(n, -9, dtype)
            a[:NC] = rng.uniform(lo, hi, NC).astype(dtype) if dtype == np.float32 else rng.integers(lo, hi, NC)
            return cp.asarray(a.astype(dtype))
        self.sfc = {"skintemp": vec(275.0, 295.0), "snow": vec(0.0, 0.0), "xice": vec(0.0, 0.0),
                    "xland": vec(1.0, 1.0), "isltyp": vec(1, 15, np.int32), "ivgtyp": vec(1, 15, np.int32)}
        host = {k: cp.asnumpy(v) for k, v in self.sfc.items()}
        lake = LAKES == 1
        host["xland"][:NC][lake] = 2.0
        host["ivgtyp"][:NC][lake] = ISWATER
        host["isltyp"][:NC][lake] = 14
        # MPAS's own sea-ice initialization turns frozen water into land ice (xland = 1, ivgtyp = isice).
        host["xice"][FROZEN], host["xland"][FROZEN], host["ivgtyp"][FROZEN] = 0.9, 1.0, ISICE
        host["isltyp"][FROZEN] = 16
        host["snow"][FROZEN] = 3.0
        self.sfc = {k: cp.asarray(v) for k, v in host.items()}
        self.diag = {}
        self.mesh = {"latCell": cp.asarray(np.radians(rng.uniform(20.0, 60.0, n)).astype(np.float32))}
        self.rng = rng
        self.refresh()
        self.session = SimpleNamespace(
            layout=SimpleNamespace(dimensions={"nVertLevels": NL, "nCellsSolve": NC}),
            native=SimpleNamespace(clm_lake_statics=dict(lakemask=LAKES.copy(), lake_depth=DEPTH.copy(),
                                                         lake_depth_flag=1, lake_mask_flag=1)))
        self.native = SimpleNamespace(get_field=lambda pool, name: np.array(ISWATER, np.int32))
        self.scalar_state = {"xice_threshold": np.float32(0.5)}

    def refresh(self):
        """A new step's words: what MPAS's schemes left in the pools before the lake glue."""
        cp, rng, n = self.cp, self.rng, NC + HALO

        def vec(lo, hi):
            a = np.full(n, np.nan, np.float32)
            a[:NC] = rng.uniform(lo, hi, NC)
            return cp.asarray(a)

        def prof(lo, hi, levels=NL):
            a = np.full((levels, n), np.nan, np.float32)
            a[:, :NC] = rng.uniform(lo, hi, (levels, NC))
            return cp.asarray(a)
        p = np.full((NL + 1, n), np.nan, np.float32)
        p[:, :NC] = np.linspace(1.0e5, 2.0e4, NL + 1, dtype=np.float32)[:, None] - rng.uniform(0, 800, NC)
        self.prepared = {"t_p": prof(270.0, 300.0), "qv_p": prof(1e-3, 1.2e-2), "u_p": prof(-12.0, 12.0),
                         "v_p": prof(-12.0, 12.0), "dz_p": prof(40.0, 80.0), "pres2_hyd_p": cp.asarray(p)}
        self.diag.update(gsw=vec(0.0, 700.0), sfc_albedo=vec(0.06, 0.2), glw=vec(250.0, 400.0),
                         sfc_emiss=vec(0.95, 0.99), rainncv=vec(0.0, 2.0), raincv=vec(0.0, 1.0),
                         hfx=vec(-50.0, 200.0), lh=vec(0.0, 300.0), grdflx=vec(-50.0, 50.0),
                         qfx=vec(0.0, 1e-4), t2m=vec(270.0, 300.0), th2m=vec(270.0, 300.0),
                         q2=vec(1e-3, 1e-2))

    def __getitem__(self, key):
        return {"prepared": self.prepared, "diag_physics": self.diag, "sfc_input": self.sfc,
                "mesh": self.mesh, "scalar_state": self.scalar_state}[key]


def _host(cp, value):
    return np.array(cp.asnumpy(cp.asarray(value)), copy=True)


def _words(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


@pytest.mark.parametrize("micro, convection", [("mp_thompson", "cu_ntiedtke"), ("mp_wsm6", "off")])
def test_every_lake_call_is_wrf_lake_kernel_on_the_same_inputs(micro, convection):
    import cupy as cp
    from woof.core.lake import initialize_lake
    from woof.hex.mpasphys.clm_lake_v842 import lake_init_step
    provider = _Provider(cp, np.random.default_rng(7))
    pools = {"diag_physics": provider.diag, "sfc_input": provider.sfc}
    initial = {name: _host(cp, provider.sfc[name]) for name in provider.sfc}
    lat = _host(cp, provider.mesh["latCell"])
    call = SimpleNamespace(dt=60.0, options={"config_microp_scheme": micro,
                                             "config_convection_scheme": convection})
    columns = lake_init_step(provider, call)
    assert columns.column_count == int(LAKES.sum())

    # WRF: lakeini on a (2, 4) grid of the same owned cells, from MPAS's initial surface.
    def grid(values):
        values = np.asarray(values)
        return cp.asarray(np.ascontiguousarray(values[..., :NC]).reshape(values.shape[:-1] + GRID))
    wrf = {"tsk": grid(initial["skintemp"]), "isltyp": grid(initial["isltyp"]), "snow": grid(initial["snow"]),
           "xice": grid(initial["xice"]), "xland": grid(initial["xland"]), "ivgtyp": grid(initial["ivgtyp"]),
           "lakemask": grid(LAKES)}
    replay = initialize_lake(
        wrf, latitude=grid(np.degrees(lat.astype(np.float64)).astype(np.float32)), lake_depth=grid(DEPTH),
        lake_depth_flag=1, use_lakedepth=1, lakedepth_default=50.0, iswater=ISWATER, xice_threshold=0.5)
    for name in ("lake_columns", "lake_static", "lake_latitude"):
        got = _host(cp, columns.fields[name])
        np.testing.assert_array_equal(_words(_host(cp, wrf[name])),
                                      _words(got.reshape(got.shape[:-1] + GRID)), err_msg=name)
    # lakeini's hand-back of the frozen lake to water reaches MPAS's pools, owned cells only.
    for name in ("xice", "xland", "ivgtyp"):
        got = _host(cp, provider.sfc[name])
        np.testing.assert_array_equal(got[:NC], _host(cp, wrf[name]).reshape(-1), err_msg=name)
        np.testing.assert_array_equal(got[NC:], initial[name][NC:], err_msg=name)
    assert _host(cp, provider.sfc["xland"])[FROZEN] == 2.0
    assert _host(cp, provider.sfc["ivgtyp"])[FROZEN] == ISWATER

    for index in range(4):
        provider.refresh()
        before = {(pool, name): _host(cp, pools[pool][name]) for pool, name in OUTPUT_FIELDS}
        diag = {name: _host(cp, provider.diag[name]) for name in provider.diag}
        prepared = {name: _host(cp, provider.prepared[name]) for name in provider.prepared}
        columns.step(provider, call)
        after = {(pool, name): _host(cp, pools[pool][name]) for pool, name in OUTPUT_FIELDS}

        # MPAS's lsm_from_MPAS words, formed here: SWDOWN = gsw/(1-albedo), RAINBL = 0 + rainncv + raincv.
        swdown = (diag["gsw"] / (np.float32(1.0) - diag["sfc_albedo"])).astype(np.float32)
        rain = np.zeros(NC + HALO, np.float32)
        if micro != "off":
            rain = rain + diag["rainncv"]
        if convection != "off":
            rain = rain + diag["raincv"]
        for (pool, name), value in before.items():
            wrf[{"skintemp": "tsk", "t2m": "t2", "th2m": "th2", "sfc_albedo": "albedo"}.get(name, name)] = grid(value)
        wrf.update(glw=grid(diag["glw"]), emiss=grid(diag["sfc_emiss"]), swdown=grid(swdown))
        atmosphere = {"temperature": grid(prepared["t_p"][:1]), "p_interface": grid(prepared["pres2_hyd_p"][:2]),
                      "dz": grid(prepared["dz_p"][:1]), "qv": grid(prepared["qv_p"][:1]),
                      "u": grid(prepared["u_p"][:1]), "v": grid(prepared["v_p"][:1])}
        replay.step(wrf, atmosphere, dt=60.0, precipitation=grid(rain))
        np.testing.assert_array_equal(
            _words(_host(cp, wrf["lake_columns"])),
            _words(_host(cp, columns.fields["lake_columns"]).reshape(-1, *GRID)), err_msg=f"call {index}")
        lake = LAKES == 1
        for (pool, name), value in after.items():
            wrf_name = {"skintemp": "tsk", "t2m": "t2", "th2m": "th2", "sfc_albedo": "albedo"}.get(name, name)
            np.testing.assert_array_equal(_words(value[:NC]), _words(_host(cp, wrf[wrf_name]).reshape(-1)),
                                          err_msg=f"call {index}: {name}")
            # non-lake cells keep MPAS's words; halo cells are never written
            np.testing.assert_array_equal(_words(value[:NC][~lake]), _words(before[pool, name][:NC][~lake]))
            np.testing.assert_array_equal(_words(value[NC:]), _words(before[pool, name][NC:]))
            assert (_words(value[:NC][lake]) != _words(before[pool, name][:NC][lake])).any() or name == "sfc_albedo"


def test_the_lake_refuses_a_lake_cell_this_run_holds_as_land():
    import cupy as cp
    from woof.hex.mpasphys.clm_lake_v842 import lake_init_step
    provider = _Provider(cp, np.random.default_rng(3))
    provider.sfc["xland"][0] = 1.0
    provider.sfc["xice"][0] = 0.0
    with pytest.raises(ValueError, match="LAKEMASK marks 1 cells"):
        lake_init_step(provider, SimpleNamespace(dt=60.0, options={}))
