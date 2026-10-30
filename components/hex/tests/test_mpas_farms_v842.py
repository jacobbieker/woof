"""WRF-Solar FARMS on the MPAS-A v8.4.2 path (woof.hex.mpasphys.farms_v842).

* the MPAS-layout entry point (``farms_mpas``: [level, cell] arrays, the
  owned cells first, halo/dummy cells after) returns compiled WRF v4.7.1's
  FARMS words exactly, on the tools/farms_wrf471_oracle columns;
* the physics driver calls ``farms_sw`` on every step, radiation steps
  included, right after the radiation slots and before the surface layer,
  only when ``config_farms_sw_backend = 'hex-cuda'``; the receipt of a run
  without it is unchanged;
* the glue operation reads MPAS's own words (prepared arrays, last
  radiation call's cldfrac, the radii exactly when the MPAS shortwave reads
  them) and overwrites swdnb/swddir/swddni/swddif on owned cells only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys.driver_v842 import (PhysicsDriver, PhysicsManager,
                                           SlotRegistry, resolve_namelist)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "tests" / "data" / "oracles" / "farms"
OUTPUTS = ("swdown2", "swddir2", "swddni2", "swddif2", "swdownc2", "swddnic2")
ON = "\n&hex_mpasphys\nconfig_farms_sw_backend = 'hex-cuda'\n/\n"


def deck(suite='mesoscale_reference', **options):
    values = {'config_physics_suite': suite, **options}
    return '&physics\n' + '\n'.join(f'{k} = {v!r}' for k, v in values.items()) + '\n/'


@pytest.fixture(scope="module")
def oracle():
    path = DATA / "farms_wrf471.npz"
    provenance = json.loads((DATA / "provenance.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["npz_sha256"]
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.parametrize("scen", ["req1_summer", "req0_jan", "req1_dec", "req0_feb"])
def test_mpas_layout_matches_wrf471_word_for_word(oracle, scen):
    import cupy as cp
    from woof.hex.mpasphys.farms_v842 import farms_mpas
    g = lambda n: oracle[f"{scen}/{n}"]
    nl = g("rho").shape[0] - 1
    nc = g("albedo").size
    halo = 7   # unowned cells after the owned ones: never read

    def profile(name):
        a = np.full((nl + 1, nc + halo), np.nan, np.float32)
        a[:, :nc] = g(name)
        return cp.asarray(a)

    def surface(a):
        b = np.full(nc + halo, np.nan, np.float32)
        b[:nc] = a
        return cp.asarray(b)
    rc, ri, rs, _ = (int(v) for v in g("scalars"))
    assert not (g("aod5502d")[g("coszen_loc") > 0] > 0).all()
    zero_aerosol = g("aod5502d") == 0
    out = farms_mpas(
        cp, nc=nc, nl=nl, p8w=surface(g("p8w")[0]), rho=profile("rho"),
        dz=profile("dz8w"), qv=profile("qv"), qc=profile("qc"), qi=profile("qi"),
        qs=profile("qs"), cldfrac=profile("cldfra"), re_cloud=profile("re_cloud"),
        re_ice=profile("re_ice"), re_snow=profile("re_snow"),
        albedo=surface(g("albedo")), coszen=surface(g("coszen_loc")),
        has_req=(rc, ri, rs), juday=int(np.trunc(g("julian")[0])))
    # MPAS carries no aerosol to FARMS (WRF aer_opt = 0): the columns whose
    # fixture aerosol is zero are WRF's words exactly
    for name in OUTPUTS:
        got = cp.asnumpy(out[name]).view(np.uint32)[zero_aerosol]
        want = g(f"out/{name}").view(np.uint32)[zero_aerosol]
        assert (got == want).all(), (name, int((got != want).sum()))
    assert zero_aerosol.sum() > 700


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


def test_driver_runs_farms_every_step_after_the_radiation_slots():
    config, driver, hooks = _driver(deck() + ON)
    for step, time in ((1, 0), (2, 360)):
        hooks.operations.clear()
        driver.begin_step(itimestep=step, time_seconds=time, pools=None)
        ops = hooks.operations
        assert ops.count("farms_sw") == 1
        at = ops.index("farms_sw")
        assert ops[at + 1] == "allocate_sfclayer"
        assert ops[at - 1] == "deallocate_radiation_lw"
        assert ("radiation_sw" in ops) == (step == 1)
        for rk in (1, 2, 3):
            driver.physics_get_tend(rk_step=rk, dynamics_substep=1, pools=None)
        driver.finish_step(pools=None)
    call = next(c for c in driver.calls if c["operation"] == "farms_sw")
    assert call["backend"] == "hex-cuda"
    assert config.receipt()["service_backends"]["farms_sw"] == "hex-cuda"


def test_without_the_service_nothing_runs_and_the_receipt_is_unchanged():
    config, driver, hooks = _driver(deck())
    driver.begin_step(itimestep=1, time_seconds=0, pools=None)
    assert "farms_sw" not in hooks.operations
    assert "service_backends" not in config.receipt()
    # no shortwave to sit beside: FARMS does not run either
    config, driver, hooks = _driver(deck(config_radt_sw_scheme='off') + ON)
    driver.begin_step(itimestep=1, time_seconds=0, pools=None)
    assert "farms_sw" not in hooks.operations
    with pytest.raises(ValueError, match="unsupported farms_sw service backend"):
        resolve_namelist(deck() + "\n&hex_mpasphys\nconfig_farms_sw_backend = 'native-cpu'\n/\n")


class _Provider:
    """The device pool views' surface the glue operation reads."""

    def __init__(self, cp, nl, nc, halo, rng, *, microp_re):
        self.cp = cp
        n = nc + halo
        prof = lambda lo, hi: cp.asarray(rng.uniform(lo, hi, (nl, n)).astype(np.float32))
        self.prepared = {
            "pres2_hyd_p": cp.asarray(np.linspace(1.0e5, 2.0e3, nl + 1, dtype=np.float32)[:, None]
                                      * np.ones((1, n), np.float32)),
            "rho_p": prof(0.2, 1.2), "dz_p": prof(50, 800), "qv_p": prof(1e-5, 1e-2),
            "qc_p": prof(0, 1e-4), "qi_p": prof(0, 5e-5), "qs_p": prof(0, 5e-5)}
        self.diag = {"cldfrac": prof(0, 1), "re_cloud": prof(3e-6, 2e-5),
                     "re_ice": prof(1e-5, 6e-5), "re_snow": prof(2e-5, 3e-4),
                     "sfc_albedo": cp.asarray(rng.uniform(0.1, 0.3, n).astype(np.float32))}
        for name in ("swdnb", "swddir", "swddni", "swddif"):
            self.diag[name] = cp.full(n, -7.0, dtype=cp.float32)
        self.mesh = {"latCell": cp.asarray(rng.uniform(-1.2, 1.2, n).astype(np.float32)),
                     "lonCell": cp.asarray(rng.uniform(0, 6.28, n).astype(np.float32))}
        self.session = SimpleNamespace(layout=SimpleNamespace(
            dimensions={"nVertLevels": nl, "nCellsSolve": nc}))
        tables = {"mpas_atmphys_manager.curr_julday": 171.5, "mpas_atmphys_manager.gmt": 0.0}
        self.native = SimpleNamespace(get_table=lambda name: (np.array(np.float32(tables[name])), ()))
        self.values = {"constants": {"degrad": np.float32(0.017453292), "dpd": np.float32(360.0 / 365.0)},
                       "scalar_state": {"has_reqc": 1, "has_reqi": 1, "has_reqs": 1}}
        self.microp_re = microp_re

    def __getitem__(self, key):
        return {"prepared": self.prepared, "diag_physics": self.diag, "mesh": self.mesh,
                **self.values}[key]


@pytest.mark.parametrize("microp_re", [True, False])
def test_glue_reads_mpas_words_and_overwrites_owned_cells(microp_re):
    import cupy as cp
    from woof.hex.mpasphys.farms_v842 import farms_mpas, farms_sw_step, mpas_coszen
    nl, nc, halo = 12, 40, 5
    p = _Provider(cp, nl, nc, halo, np.random.default_rng(3), microp_re=microp_re)
    halo_before = {n: cp.asnumpy(p.diag[n][nc:]) for n in ("swdnb", "swddir", "swddni", "swddif")}
    call = SimpleNamespace(time_seconds=43200.0, options={
        "config_microp_re": microp_re, "config_microp_scheme": "mp_thompson"})
    farms_sw_step(p, call)
    cz = mpas_coszen(cp, lat_cell=p.mesh["latCell"][:nc], lon_cell=p.mesh["lonCell"][:nc],
                     julian=171.5, xtime_min=np.float32(720.0), gmt=0.0,
                     degrad=np.float32(0.017453292), dpd=np.float32(360.0 / 365.0))
    zeros = cp.zeros((nl, nc), cp.float32)
    want = farms_mpas(
        cp, nc=nc, nl=nl, p8w=p.prepared["pres2_hyd_p"][0], rho=p.prepared["rho_p"],
        dz=p.prepared["dz_p"], qv=p.prepared["qv_p"], qc=p.prepared["qc_p"],
        qi=p.prepared["qi_p"], qs=p.prepared["qs_p"], cldfrac=p.diag["cldfrac"],
        re_cloud=p.diag["re_cloud"] if microp_re else zeros,
        re_ice=p.diag["re_ice"] if microp_re else zeros,
        re_snow=p.diag["re_snow"] if microp_re else zeros,
        albedo=p.diag["sfc_albedo"], coszen=cz,
        has_req=(1, 1, 1) if microp_re else (0, 0, 0), juday=171)
    for source, target in (("swdown2", "swdnb"), ("swddir2", "swddir"),
                           ("swddni2", "swddni"), ("swddif2", "swddif")):
        assert (cp.asnumpy(p.diag[target][:nc]).view(np.uint32)
                == cp.asnumpy(want[source]).view(np.uint32)).all(), target
        assert (cp.asnumpy(p.diag[target][nc:]) == halo_before[target]).all()
    assert (cp.asnumpy(cz) > 0).any() and (cp.asnumpy(cz) <= 0).any()


def test_mpas_coszen_is_mpas_radiation_geometry_at_the_step():
    """The statements of mpas_sw_prep's coszr with radt = 0 and of
    mpas_sw_radconst, against a binary64 evaluation of the same formulas."""
    import cupy as cp
    from woof.hex.mpasphys.farms_v842 import mpas_coszen
    rng = np.random.default_rng(5)
    lat = rng.uniform(-1.5, 1.5, 4000).astype(np.float32)
    lon = rng.uniform(-3.1, 6.2, 4000).astype(np.float32)
    degrad, dpd = np.float32(0.017453292), np.float32(360.0 / 365.0)
    for julian, xtime in ((10.25, 15.0), (171.5, 725.0), (300.9, 2000.0)):
        got = cp.asnumpy(mpas_coszen(cp, lat_cell=cp.asarray(lat), lon_cell=cp.asarray(lon),
                                     julian=julian, xtime_min=xtime, gmt=6.0,
                                     degrad=degrad, dpd=dpd))
        d = np.float64
        sx = d(dpd) * ((julian - 80.0) if julian >= 80.0 else (julian + 285.0)) * d(degrad)
        declin = np.arcsin(np.sin(23.5 * d(degrad)) * np.sin(sx))
        local = 6.0 + np.fmod(xtime, 1440.0) / 60.0 + (lon / d(degrad)) / 15.0
        angle = 15.0 * (local - 12.0) * d(degrad)
        want = np.sin(lat) * np.sin(declin) + np.cos(lat) * np.cos(declin) * np.cos(angle)
        assert np.abs(got - want).max() < 2.0e-5
