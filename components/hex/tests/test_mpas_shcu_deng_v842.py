"""Deng shallow cumulus and cu_rad_feedback on the MPAS-A v8.4.2 path
(woof.hex.mpasphys.shcu_deng_v842; WOOF-only services).

* the MPAS entry point of the Deng column body (``deng_shcu_mpas_column``:
  a DX and a DSIGMA per column) returns compiled WRF v4.7.1's words on every
  graded word of the tools/shcu_deng_wrf471_oracle fixture;
* the ``shcu_deng`` glue hands that entry point MPAS's own words: the
  prepared arrays, MYNN's qke/kzh/el_pbl, hpbl, zgrid(1), lonCell/degrad,
  the radiation heating, mavail (updated on owned cells only), the step's
  minutes, MPAS's grid length (len_disp/meshDensity**0.25, glibc powf) and
  the eta increment of the dry hydrostatic pressure;
* the coupling adds the tendencies to physics_get_tend's sources in
  mpas_atmphys_todynamics.F's arithmetic on owned cells only;
* the cloudiness slot merges the convective cloud in WRF's order with
  WRF-graded statements (woof.core.cumulus_radiation), and changes no word
  with both services off; FARMS reads the same condensate;
* the driver order and cadence, the refusals, and the restart carry.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys.driver_v842 import (PhysicsDriver, PhysicsManager,
                                           SlotRegistry, resolve_namelist)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "data" / "oracles" / "shcu_deng"
F = np.float32
DEG = F(0.017453292)
DENG_ON = "config_shcu_deng_backend = 'hex-cuda'\n"
FEEDBACK_ON = "config_cu_rad_feedback_backend = 'hex-cuda'\n"


def deck(suite="mesoscale_reference", backends="", **options):
    values = {"config_physics_suite": suite, **options}
    text = "&physics\n" + "\n".join(f"{k} = {v!r}" for k, v in values.items()) + "\n/"
    return text + "\n&hex_mpasphys\n" + backends + "/\n"


GPU_BACKENDS = "config_radt_cld_backend = 'hex-cuda'\n"


def _harness():
    import sys
    sys.path.insert(0, str(ROOT))
    from tools.shcu_deng_wrf471_oracle import compare_woof, deng_columns
    return deng_columns, compare_woof


def test_mpas_entry_point_matches_wrf471_word_for_word():
    _, compare_woof = _harness()
    rows = compare_woof.run(str(FIXTURE / "columns-structured.bin.gz"),
                            str(FIXTURE / "oracle-structured-O0.out.gz"),
                            mpas_entry=True)
    summary = compare_woof.summarize(rows)
    bad = [r for r in rows if r["off"] or r["bit_unequal"]]
    assert summary["off"] == 0 and summary["bit_unequal"] == 0, json.dumps(bad[:10])
    assert summary["max_ulp"] == 0 and summary["status_columns"] == 0
    assert summary["words"] == 1_737_808


# ------------------------------------------------------------------ driver
def _driver(text):
    config = resolve_namelist(text)

    class Hooks:
        def __init__(self):
            self.operations = []

        def invoke(self, operation, call):
            self.operations.append(operation)
            if operation == "shcu_deng_tend":
                return ("coupled", call.pools["sources"])
            if operation == "physics_get_tend":
                return "sources"

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


def test_driver_runs_deng_every_step_after_convection_and_couples_once():
    config, driver, hooks = _driver(deck(
        config_pbl_scheme="bl_mynn", config_convection_scheme="cu_grell_freitas",
        backends=GPU_BACKENDS + DENG_ON + FEEDBACK_ON))
    for step, time in ((1, 0), (2, 360)):
        hooks.operations.clear()
        driver.begin_step(itimestep=step, time_seconds=time, pools=None)
        ops = hooks.operations
        assert ops.count("shcu_deng") == 1
        at = ops.index("shcu_deng")
        assert ops[at - 1] == "update_convection_step2"
        assert ops[at + 1] == "deallocate_forall_physics"
        for rk in (1, 2, 3):
            got = driver.physics_get_tend(rk_step=rk, dynamics_substep=1, pools=None)
            assert got == ("coupled", "sources")
        assert hooks.operations.count("shcu_deng_tend") == 1
        assert hooks.operations.index("shcu_deng_tend") == hooks.operations.index("physics_get_tend") + 1
        driver.finish_step(pools=None)
    backends = config.receipt()["service_backends"]
    assert backends["shcu_deng"] == backends["cu_rad_feedback"] == "hex-cuda"
    call = next(c for c in driver.calls if c["operation"] == "shcu_deng")
    assert call["backend"] == "hex-cuda"


def test_without_the_services_nothing_runs_and_the_receipt_is_unchanged():
    config, driver, hooks = _driver(deck(config_pbl_scheme="bl_mynn"))
    driver.begin_step(itimestep=1, time_seconds=0, pools=None)
    driver.physics_get_tend(rk_step=1, dynamics_substep=1, pools=None)
    assert "shcu_deng" not in hooks.operations and "shcu_deng_tend" not in hooks.operations
    assert "service_backends" not in config.receipt()


@pytest.mark.parametrize("text, match", [
    (deck(config_pbl_scheme="bl_ysu", backends=GPU_BACKENDS + DENG_ON), "MYNN"),
    (deck(config_pbl_scheme="bl_mynn", backends=DENG_ON), "cloudiness slot"),
    (deck(config_pbl_scheme="bl_mynn", config_radt_lw_scheme="cam_lw",
          config_radt_sw_scheme="cam_sw", backends=GPU_BACKENDS + DENG_ON), "RRTMG"),
    (deck(config_pbl_scheme="bl_mynn", config_microp_scheme="off",
          backends=GPU_BACKENDS + DENG_ON), "microphysics"),
    (deck(config_convection_scheme="cu_kain_fritsch",
          backends=GPU_BACKENDS + FEEDBACK_ON), "cu_grell_freitas"),
    (deck(backends=GPU_BACKENDS + "config_shcu_deng_backend = 'native-cpu'\n"),
     "unsupported shcu_deng service backend"),
])
def test_the_services_refuse_what_they_cannot_honour(text, match):
    with pytest.raises(ValueError, match=match):
        resolve_namelist(text)


# --------------------------------------------------------- glue operation
class _Provider:
    """The device pool views' surface the glue reads, MPAS [level, cell]
    arrays with ``halo`` unowned cells after the owned ones."""

    def __init__(self, cp, case, step, halo, rng):
        self.cp = cp
        nl, nc = case["nz"], case["ncol"]
        n = nc + halo

        def lev(a, rows):
            out = np.full((rows, n), np.nan, np.float32)
            out[:a.shape[0], :nc] = a
            return cp.asarray(out)

        def surf(a):
            out = np.full(n, np.nan, np.float32)
            out[:nc] = a
            return cp.asarray(out)
        s = case["steps"][step]
        # the prepared globals carry kme = nl+1 rows
        self.prepared = {f"{k}_p": lev(s[w], nl + 1) for k, w in (
            ("u", "u"), ("v", "v"), ("t", "t"), ("qv", "qv"), ("qc", "qc"),
            ("qr", "qr"), ("pres", "p"), ("pi", "pi"), ("rho", "rho"), ("dz", "dz8w"),
            ("z", "z_at_w"), ("w", "w"))}
        ps = rng.uniform(8.0e4, 1.02e5, nc).astype(np.float32)
        znw = np.concatenate(([1.0], 1.0 + np.cumsum(case["dnw"]))).astype(np.float32)
        pd = (F(5000.0) + znw[:, None] * (ps[None, :] - F(5000.0))).astype(np.float32)
        self.prepared["pres2_hydd_p"] = lev(pd, nl + 1)
        self.diag = {"qke": lev(s["qke"], nl), "kzh": lev(s["exch_h"], nl),
                     "el_pbl": lev(s["el_pbl"], nl), "hpbl": surf(s["pblh"]),
                     "mavail": surf(case["state"]["mavail"])}
        self.tend = {"rthratenlw": lev(s["rthratenlw"], nl),
                     "rthratensw": lev(s["rthratensw"], nl)}
        zgrid = np.full((nl + 1, n), np.nan, np.float32)
        zgrid[0, :nc] = case["ht"]
        self.mesh = {"zgrid": cp.asarray(zgrid),
                     "lonCell": surf((case["xlong"] * DEG).astype(np.float32)),
                     "meshDensity": surf(rng.uniform(0.01, 1.0, nc).astype(np.float32))}
        self.session = SimpleNamespace(layout=SimpleNamespace(
            dimensions={"nVertLevels": nl, "nCellsSolve": nc}))
        gmt = case["gmt"]
        self.native = SimpleNamespace(get_table=lambda name: (np.array(F(gmt)), ()))
        self.values = {"constants": {"degrad": DEG}, "configs": {"config_len_disp": 3000.0}}

    def __getitem__(self, key):
        return {"prepared": self.prepared, "diag_physics": self.diag,
                "tend_physics": self.tend, "mesh": self.mesh, **self.values}[key]


def _seeded_state(case):
    """The case's carried Deng state (the oracle's injected histories)."""
    import cupy as cp
    from woof.core.shcu_deng import DengState
    D, _ = _harness()
    nz, ncol, st = case["nz"], case["ncol"], case["state"]
    state = DengState.cold(nz, 1, ncol)
    for k in ("w0avg", "tkeavg", "cldareaa", "cldareab", "cldliqa", "cldliqb", "wub"):
        getattr(state, k)[...] = cp.asarray(st[k].reshape(nz, 1, ncol))
    for k in ("rainsh", "rainshv", "rainshvb", "pblmax", "capesave", "xtime1",
              "radsave", "clddpthb", "cldtopb", "pblhavg"):
        getattr(state, k)[...] = cp.asarray(st[k].reshape(1, ncol))
    for k in D.STATE_INT:
        getattr(state, k)[...] = cp.asarray(st[k].astype(np.int32).reshape(1, ncol))
    state.ainckfsa[...] = cp.asarray(st["ainckfsa"].reshape(D.NSH, 1, ncol))
    return state


def _mynn_case():
    D, _ = _harness()
    cases = D.read_input(FIXTURE / "columns-structured.bin.gz")
    return next(c for c in cases if c["bl_pbl"] == 5 and not c["adapt"])


def test_glue_hands_the_kernel_mpas_words():
    import cupy as cp
    from woof.core.noahmp_libm import powf
    from woof.core.shcu_deng import launch_deng
    from woof.hex.mpasphys.shcu_deng_v842 import DengService, deng_step
    case = _mynn_case()
    nl, nc, halo = case["nz"], case["ncol"], 6
    rng = np.random.default_rng(11)
    p = _Provider(cp, case, 0, halo, rng)
    service = DengService(p.session)
    service.state = _seeded_state(case)
    dt = float(case["dt"])
    want_state = _seeded_state(case)
    mavail_ref = cp.asarray(case["state"]["mavail"].reshape(1, nc))
    for step in range(2):
        p = _Provider(cp, case, step, halo, np.random.default_rng(11))
        p.diag["mavail"][:nc] = mavail_ref.reshape(nc)
        xtime = F(case["steps"][step]["xtime"])
        call = SimpleNamespace(dt=dt, time_seconds=float(xtime * F(60.0)))
        out = deng_step(p, call, service)
        # the reference: WRF-layout arrays built here from the same words
        s = case["steps"][step]
        sh = (nl, 1, nc)
        d = lambda a, shape=sh: cp.asarray(np.ascontiguousarray(np.asarray(a, np.float32).reshape(shape)))
        density = cp.asnumpy(p.mesh["meshDensity"][:nc])
        dx = np.array([F(3000.0) / powf(x, F(0.25)) for x in density], np.float32)
        lon = cp.asnumpy(p.mesh["lonCell"][:nc])
        pd = cp.asnumpy(p.prepared["pres2_hydd_p"])[:, :nc]
        dnw = ((pd[1:] - pd[:-1]) / (pd[0] - pd[nl])[None, :]).astype(np.float32).T
        ref = launch_deng(
            u=d(s["u"]), v=d(s["v"]), t=d(s["t"]), qv=d(s["qv"]), qc=d(s["qc"]),
            qr=d(s["qr"]), p=d(s["p"]), pii=d(s["pi"]), rho=d(s["rho"]),
            dz8w=d(s["dz8w"]), tke=d(s["qke"]), exch_h=d(s["exch_h"]),
            el_pbl=d(s["el_pbl"]), rthratenlw=d(s["rthratenlw"]),
            rthratensw=d(s["rthratensw"]), z_at_w=d(s["z_at_w"], (nl + 1, 1, nc)),
            w=d(s["w"], (nl + 1, 1, nc)), pblh=d(s["pblh"], (1, nc)),
            ht=d(case["ht"], (1, nc)), xlong=d(lon / DEG, (1, nc)),
            dnw=d(dnw, (1, nc, nl)), mavail=mavail_ref, state=want_state, dt=dt,
            dx=d(dx, (1, nc)), xtime=float(F(call.time_seconds) / F(60.0)),
            gmt=float(case["gmt"]), adaptive=False, bl_pbl_physics=5)
        for k in ("rthshten", "rqvshten", "rqcshten", "rqrshten", "rdcashten", "rqcdcshten"):
            assert (cp.asnumpy(out[k]).view(np.uint32) == cp.asnumpy(ref[k]).view(np.uint32)).all(), k
        for k, a in want_state.arrays().items():
            got = cp.asnumpy(getattr(service.state, k))
            assert (got.view(np.uint32) == cp.asnumpy(a).view(np.uint32)).all(), k
        assert (cp.asnumpy(p.diag["mavail"][:nc]).view(np.uint32)
                == cp.asnumpy(mavail_ref).reshape(nc).view(np.uint32)).all()
        assert np.isnan(cp.asnumpy(p.diag["mavail"][nc:])).all()
        assert set(service.tendencies) == {"rthshten", "rqvshten", "rqcshten", "rqrshten"}
    # the case made clouds and rain: the gate above is not graded on zeros
    assert float(cp.abs(cp.asarray(out["rthshten"])).max()) > 0
    # DNW: the eta increments of the column's dry pressure reproduce WRF's
    # eta levels to rounding (pd was built from them)
    assert np.allclose(dnw, np.tile(case["dnw"], (nc, 1)), rtol=0, atol=2e-6)


def test_coupling_is_todynamics_arithmetic_on_owned_cells():
    import cupy as cp
    from woof.hex.cuda_physics_v841 import RV_OVER_RD_F32 as rvord
    from woof.hex.mpasphys.shcu_deng_v842 import couple_deng
    rng = np.random.default_rng(4)
    nl, nc, halo, ns = 9, 30, 4, 5
    n = nc + halo
    g = lambda *shape, lo=-1.0, hi=1.0: rng.uniform(lo, hi, shape).astype(np.float32)
    rho_theta, scalars = g(nl, n), g(ns, nl, n)
    mass, theta_m, qv = g(nl, n, lo=0.1, hi=1.3), g(nl, n, lo=250, hi=400), g(nl, n, lo=0, hi=0.02)
    tend = {k: g(nl, nc, lo=-1e-3, hi=1e-3) for k in ("rthshten", "rqvshten", "rqcshten", "rqrshten")}
    rt, sc = cp.asarray(rho_theta), cp.asarray(scalars)
    couple_deng(cp, rho_theta=rt, scalars=sc, mass=cp.asarray(mass), theta_m=cp.asarray(theta_m),
                qv=cp.asarray(qv), tendencies={k: cp.asarray(v) for k, v in tend.items()},
                index_qv=0, index_qc=1, index_qr=3, nc=nc)
    m = mass[:, :nc]
    th = tend["rthshten"] * m
    vq = tend["rqvshten"] * m
    coeff = F(1.0) + rvord * qv[:, :nc]
    want_rt = rho_theta.copy()
    want_rt[:, :nc] = rho_theta[:, :nc] + (coeff * th + ((rvord * theta_m[:, :nc]) * vq) / coeff)
    want_sc = scalars.copy()
    want_sc[0, :, :nc] = scalars[0, :, :nc] + vq
    want_sc[1, :, :nc] = scalars[1, :, :nc] + tend["rqcshten"] * m
    want_sc[3, :, :nc] = scalars[3, :, :nc] + tend["rqrshten"] * m
    assert (cp.asnumpy(rt).view(np.uint32) == want_rt.view(np.uint32)).all()
    assert (cp.asnumpy(sc).view(np.uint32) == want_sc.view(np.uint32)).all()


# ------------------------------------------------------- cloudiness merge
def _cloud_pools(cp, rng, nz, nc, halo, convective=None):
    n = nc + halo
    lev = lambda lo, hi: cp.asarray(rng.uniform(lo, hi, (nz, n)).astype(np.float32))
    prepared = {"qv": lev(1e-5, 1e-2), "qc": lev(0, 1e-4), "qi": lev(0, 5e-5), "qs": lev(0, 5e-5),
                "t": lev(200, 300), "pres": lev(2e4, 1e5), "pres_hyd": lev(2e4, 1e5),
                "rho": lev(0.3, 1.2)}
    for name in ("qc", "qi"):
        prepared[name][:, ::3] = 0.0     # clear columns the convective cloud can fill
    pools = {"prepared": prepared,
             "mesh": {"meshDensity": cp.asarray(rng.uniform(0.1, 1, n).astype(np.float32))},
             "sfc_input": {"xland": cp.asarray(rng.choice([1.0, 2.0], n).astype(np.float32))},
             "diag_physics": {"cldfrac": cp.full((nz, n), -3.0, dtype=cp.float32)},
             "nCellsSolve": nc, "config_len_disp": 3000.0, "f_qc": 1, "f_qi": 1,
             "radiation_prepared": {k: cp.full((nz, n), -5.0, dtype=cp.float32)
                                    for k in ("qvrad", "qcrad", "qirad", "qsrad", "cldfrac")}}
    for k in ("dx", "xland"):
        pools["radiation_prepared"][k] = cp.zeros(n, dtype=cp.float32)
    if convective is not None:
        pools["convective_cloud"] = convective
    return pools


@pytest.mark.parametrize("scheme", ["cld_incidence", "cld_fraction", "cld_fraction_thompson"])
def test_cloudiness_merges_the_convective_cloud_in_wrf_order(scheme):
    import cupy as cp
    from woof.core.cumulus_radiation import add_convective_condensate, merge_deng_cloud
    from woof.hex.mpasphys import radshared_v842 as rs
    nz, nc, halo = 12, 40, 5
    rng = np.random.default_rng(8)
    sh = cp.asarray(rng.uniform(0, 1, (nz, nc)).astype(np.float32))
    cw = cp.asarray(rng.uniform(0, 2e-4, (nz, nc)).astype(np.float32))
    qc_cu = cp.asarray(rng.uniform(0, 3e-4, (nz, nc + halo)).astype(np.float32))
    qi_cu = cp.asarray(rng.uniform(0, 1e-4, (nz, nc + halo)).astype(np.float32))
    cloud = {"deng": (sh, cw), "feedback": True, "qc_cu": qc_cu, "qi_cu": qi_cu}
    off = _cloud_pools(cp, np.random.default_rng(1), nz, nc, halo)
    on = _cloud_pools(cp, np.random.default_rng(1), nz, nc, halo, cloud)
    plain = _cloud_pools(cp, np.random.default_rng(1), nz, nc, halo, {"deng": None, "feedback": False})
    call = lambda pools: SimpleNamespace(scheme=scheme, pools=pools)
    base = rs.cloudiness_slot(call(off))
    rs.cloudiness_slot(call(plain))
    rs.cloudiness_slot(call(on))
    # both services off: not a word moves
    for pool, name in (("diag_physics", "cldfrac"),):
        assert (cp.asnumpy(plain[pool][name]).view(np.uint32)
                == cp.asnumpy(off[pool][name]).view(np.uint32)).all()
    for name in ("qcrad", "qirad", "cldfrac"):
        assert (cp.asnumpy(plain["radiation_prepared"][name]).view(np.uint32)
                == cp.asnumpy(off["radiation_prepared"][name]).view(np.uint32)).all()
    # on: ICLOUD_CU = 1 before the fraction, then Deng's merge after it
    profiles = {n: rs._from_public(off["prepared"][n], nc) for n in
                ("qv", "qc", "qi", "qs", "t", "pres", "pres_hyd", "rho")}
    profiles["qc"], profiles["qi"] = add_convective_condensate(
        profiles["qc"], profiles["qi"], rs._from_public(qc_cu, nc), rs._from_public(qi_cu, nc),
        f_qc=True, f_qi=True)
    want = rs.driver_cloudiness(scheme=scheme, **profiles,
                                meshDensity=off["mesh"]["meshDensity"][:nc],
                                xland=off["sfc_input"]["xland"][:nc], len_disp=3000.0,
                                f_qc=1, f_qi=1)
    cf, qcrad = merge_deng_cloud(want["cldfrac"], want["qcrad"],
                                 cldfra_sh=cp.ascontiguousarray(sh.T), cw_rad=cp.ascontiguousarray(cw.T))
    got_cf = cp.asnumpy(on["diag_physics"]["cldfrac"][:, :nc]).T
    assert (got_cf.view(np.uint32) == cp.asnumpy(cf).view(np.uint32)).all()
    got_qc = cp.asnumpy(on["radiation_prepared"]["qcrad"][:nz, :nc]).T
    assert (got_qc.view(np.uint32) == cp.asnumpy(qcrad).view(np.uint32)).all()
    got_qi = cp.asnumpy(on["radiation_prepared"]["qirad"][:nz, :nc]).T
    assert (got_qi.view(np.uint32) == cp.asnumpy(want["qirad"]).view(np.uint32)).all()
    # the merge moved words, and the halos are untouched
    assert (cp.asnumpy(on["diag_physics"]["cldfrac"]) != cp.asnumpy(off["diag_physics"]["cldfrac"])).any()
    assert (cp.asnumpy(on["diag_physics"]["cldfrac"][:, nc:]) == -3.0).all()
    assert (cp.asnumpy(base["cldfrac"]) <= 1.0).all()


def test_farms_reads_the_merged_condensate():
    import cupy as cp
    import woof.hex.mpasphys.farms_v842 as fm
    from woof.hex.mpasphys.shcu_deng_v842 import merged_condensate
    seen = {}

    def spy(cp_, **kw):
        seen.update(kw)
        return {k: cp.zeros(kw["nc"], cp.float32) for k, _ in fm.OVERWRITES}
    nl, nc, n = 6, 10, 13
    rng = np.random.default_rng(2)
    lev = lambda: cp.asarray(rng.uniform(0, 1e-4, (nl, n)).astype(np.float32))
    sh, cw = (cp.asarray(rng.uniform(0, 1, (nl, nc)).astype(np.float32)) for _ in range(2))
    service = SimpleNamespace(state=SimpleNamespace(), _ensure_state=lambda: None)
    service.state.cldfra_sh, service.state.cw_rad = sh.reshape(nl, 1, nc), cw.reshape(nl, 1, nc)
    config = SimpleNamespace(service_backends={"shcu_deng": "hex-cuda", "cu_rad_feedback": "hex-cuda"})
    session = SimpleNamespace(config=config, shcu_deng_service=service, layout=SimpleNamespace(
        dimensions={"nVertLevels": nl, "nCellsSolve": nc}))
    prepared = {k: lev() for k in ("qv_p", "qc_p", "qi_p", "qs_p", "rho_p", "dz_p")}
    prepared["pres2_hyd_p"] = cp.ones((nl + 1, n), cp.float32)
    diag = {k: lev() for k in ("cldfrac", "qc_cu", "qi_cu", "re_cloud", "re_ice", "re_snow")}
    diag.update({k: cp.zeros(n, cp.float32) for k in ("sfc_albedo", "swdnb", "swddir", "swddni", "swddif")})
    values = {"prepared": prepared, "diag_physics": diag,
              "mesh": {"latCell": cp.zeros(n, cp.float32), "lonCell": cp.zeros(n, cp.float32)},
              "constants": {"degrad": DEG, "dpd": F(360.0 / 365.0)},
              "scalar_state": {"has_reqc": 0, "has_reqi": 0, "has_reqs": 0, "f_qi": 1}}
    provider = type("P", (), {"__getitem__": lambda self, k: values[k]})()
    provider.cp, provider.session = cp, session
    provider.native = SimpleNamespace(get_table=lambda name: (np.array(F(100.0)), ()))
    call = SimpleNamespace(time_seconds=0.0, options={"config_microp_re": False,
                                                      "config_microp_scheme": "mp_wsm6"})
    original = fm.farms_mpas
    fm.farms_mpas = spy
    try:
        fm.farms_sw_step(provider, call)
    finally:
        fm.farms_mpas = original
    own = lambda a: a[:nl, :nc]
    qc, qi = merged_condensate(cp, qc=own(prepared["qc_p"]), qi=own(prepared["qi_p"]),
                               qc_cu=own(diag["qc_cu"]), qi_cu=own(diag["qi_cu"]),
                               cw_rad=cw, f_qc=True, f_qi=True)
    assert (cp.asnumpy(seen["qc"]).view(np.uint32) == cp.asnumpy(qc).view(np.uint32)).all()
    assert (cp.asnumpy(seen["qi"]).view(np.uint32) == cp.asnumpy(qi).view(np.uint32)).all()
    want_qc = (cp.asnumpy(own(prepared["qc_p"])) + cp.asnumpy(own(diag["qc_cu"])))
    want_qc = cp.asnumpy(cw) + want_qc
    assert (cp.asnumpy(qc).view(np.uint32) == want_qc.view(np.uint32)).all()
    assert seen["cldfrac"] is diag["cldfrac"]


# ---------------------------------------------------------------- restart
def test_restart_carries_the_deng_state_bit_for_bit():
    import cupy as cp
    from woof.hex.mpasphys.shcu_deng_v842 import DengService, deng_step
    case = _mynn_case()
    nc = case["ncol"]
    dt = float(case["dt"])

    def run(service, step):
        p = _Provider(cp, case, step, 3, np.random.default_rng(11))
        xtime = F(case["steps"][step]["xtime"])
        return deng_step(p, SimpleNamespace(dt=dt, time_seconds=float(xtime * F(60.0))), service)
    session = SimpleNamespace(layout=SimpleNamespace(
        dimensions={"nVertLevels": case["nz"], "nCellsSolve": nc}))
    straight = DengService(session)
    straight.state = _seeded_state(case)
    run(straight, 0)
    saved = straight.restart_state()
    want = run(straight, 1)
    resumed = DengService(session)
    resumed.restore_restart_state(saved)
    got = run(resumed, 1)
    for k in ("rthshten", "rqvshten", "rqcshten", "rqrshten"):
        assert (cp.asnumpy(got[k]).view(np.uint32) == cp.asnumpy(want[k]).view(np.uint32)).all(), k
    for k, a in straight.state.arrays().items():
        assert (cp.asnumpy(getattr(resumed.state, k)).view(np.uint32)
                == cp.asnumpy(a).view(np.uint32)).all(), k
    bad = dict(saved)
    bad.pop("w0avg")
    with pytest.raises(ValueError, match="restart state differs"):
        DengService(session).restore_restart_state(bad)
