"""use_theta_m = 1 and do_avgflx_em = 1 in the whole model, on the idealized
moist bubble (woof/verify/cases/moist_bubble.py: WK82 sounding, 3 K thermal,
periodic x-z channel, km_opt = 4 + monotonic sixth order, Rayleigh w top).

* With microphysics off the dycore conserves the mass-coupled prognostic it
  integrates: sum((c1h*mu + c2h)*theta*(-dnw)) holds to float32 roundoff for
  theta_m under use_theta_m = 1 exactly as it does for dry theta under 0.
* The theta_m run and the dry run are the same storm (the prognostic
  changes, the physics does not).
* use_theta_m = 0 and do_avgflx_em = 0 launch nothing new: every theta_m and
  avgflx kernel is unreachable, so the dry run's words are the words it had.
* History: THM is the prognostic, T the dry theta; AVGFLX_* are the
  history-interval means of the last RK stage's sumflux fluxes, restarted
  after a history step, and switching them on moves no forecast word.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from conftest import requires_gpu

def _strict() -> bool:
    from woof.wrf_exact import ENABLED
    return bool(ENABLED)


# Strict arithmetic reads WRF's t = theta - 300 and needs thb = 300
# (woof.core.state.adopt_wrf_theta_carrier), which the idealized initializer
# does not adopt -- for dry theta as much as for theta_m.  The strict words of
# every theta_m statement are graded by tests/test_bigstep_thetam_wrf471_parity.py.
pytestmark = [pytest.mark.gpu, requires_gpu,
              pytest.mark.skipif(_strict(), reason="idealized base state is not "
                                 "the strict thb = 300 carrier")]

NX = 48
STEPS = 12


def _build(**changes):
    from woof.config import validate_run_config
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.verify.cases import moist_bubble as mb
    from woof.verify.cases.wk82 import wk82_sounding
    cfg = validate_run_config(dataclasses.replace(
        mb.default_config(), nx=NX, **changes))
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: wk82_sounding(z)[0],
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    return mb.build(cfg, coord, base), cfg


def _coupled_total(state) -> float:
    import cupy as cp
    f8 = cp.float64
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    theta = thb.astype(f8) + state.thp.astype(f8)
    chm = (state.c1h[:, None, None].astype(f8)
           * (state.mub2d + state.mup)[None].astype(f8)
           + state.c2h[:, None, None].astype(f8))
    return float(cp.sum(theta * chm * (-state.dnw[:, None, None].astype(f8))))


def test_the_moist_bubble_starts_from_real_exes_theta_m():
    import cupy as cp
    from woof.core import constants as c
    moist, _ = _build(use_theta_m=1)
    dry, _ = _build(use_theta_m=0)
    assert moist.theta_m and not dry.theta_m
    f = np.float32
    thb = cp.asnumpy(dry.thb)[:, None, None].astype(f)
    want = ((cp.asnumpy(dry.thp) + thb)
            * (f(1.0) + f(c.RVOVRD) * cp.asnumpy(dry.qv)) - thb).astype(f)
    assert np.array_equal(cp.asnumpy(moist.thp).view(np.uint32), want.view(np.uint32))
    # Same dry mass, vapour and geopotential: only the prognostic changed.
    for name in ("mup", "qv", "php", "u", "w"):
        assert np.array_equal(cp.asnumpy(getattr(moist, name)),
                              cp.asnumpy(getattr(dry, name))), name
    # And the same pressure, to float32: calc_p_rho_phi's theta_m branch.
    rel = np.abs(cp.asnumpy(moist.p).astype(np.float64)
                 / cp.asnumpy(dry.p).astype(np.float64) - 1.0)
    assert rel.max() < 2e-6


@pytest.mark.parametrize("use_theta_m", (0, 1))
def test_the_dycore_conserves_its_coupled_prognostic(use_theta_m):
    from woof.core.dycore import run_steps
    state, cfg = _build(use_theta_m=use_theta_m, mp_physics=0)
    before = _coupled_total(state)
    run_steps(state, cfg, n=STEPS)
    drift = abs(_coupled_total(state) - before) / abs(before)
    assert drift < 2e-7, drift


def test_theta_m_and_dry_theta_integrate_the_same_storm():
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.core.theta_m import dry_theta
    runs = {}
    for flag in (0, 1):
        state, cfg = _build(use_theta_m=flag)
        run_steps(state, cfg, n=STEPS)
        runs[flag] = state
    dry, moist = runs[0], runs[1]
    assert moist.theta_m
    w0 = float(cp.abs(dry.w).max())
    w1 = float(cp.abs(moist.w).max())
    assert abs(w1 - w0) <= 0.02 * w0 + 1e-3
    theta_dry = cp.asnumpy(dry_theta(moist))
    assert np.abs(theta_dry - cp.asnumpy(dry.thp)).max() < 0.05
    assert np.abs(cp.asnumpy(moist.qv) - cp.asnumpy(dry.qv)).max() < 1e-5


def test_dry_theta_and_no_avgflx_launch_nothing_new(monkeypatch):
    """use_theta_m = 0 / do_avgflx_em = 0: no new kernel is reachable."""
    import woof.core.kernels as kernels
    from woof.core.dycore import run_steps
    real = kernels.get_kernel

    def guarded(name, func, *args, **kwargs):
        if name in ("theta_m", "avgflx"):
            raise AssertionError(f"{name}.{func} launched on a dry-theta run")
        return real(name, func, *args, **kwargs)

    monkeypatch.setattr(kernels, "get_kernel", guarded)
    state, cfg = _build()
    assert cfg.use_theta_m == 0 and cfg.do_avgflx_em == 0
    run_steps(state, cfg, n=2)
    assert not state.theta_m
    assert state.existing_scratch("theta_m_stash") is None
    assert state.existing_scratch("avgflx_rum") is None


def test_history_writes_thm_as_the_prognostic_and_t_as_dry_theta():
    import cupy as cp
    from woof.core.theta_m import dry_theta
    from woof.io.wrfout import state_frame
    from woof.wrf_exact import ENABLED
    moist, _ = _build(use_theta_m=1)
    frame = state_frame(moist)
    thb = cp.asnumpy(moist.thb)[:, None, None]
    thm = cp.asnumpy(moist.thp) if ENABLED else cp.asnumpy(moist.thp) + thb - np.float32(300.0)
    dry = cp.asnumpy(dry_theta(moist))
    dry = dry if ENABLED else dry + thb - np.float32(300.0)
    assert np.array_equal(frame["THM"], thm.astype(np.float32))
    assert np.array_equal(frame["T"], dry.astype(np.float32))
    plain, _ = _build(use_theta_m=0)
    frame = state_frame(plain)
    assert np.array_equal(frame["T"], frame["THM"])


def test_avgflx_means_restart_after_history_and_move_no_forecast_word():
    import cupy as cp
    from woof.core import dycore
    from woof.core.avgflx import history_fields
    on, cfg_on = _build(do_avgflx_em=1)
    off, cfg_off = _build()
    fluxes = []
    history_steps = {3}
    for n in range(1, 7):
        dycore.step(on, cfg_on, avgflx_history_due=n in history_steps)
        dycore.step(off, cfg_off)
        fluxes.append(tuple(cp.asnumpy(on.existing_scratch(slot)).copy()
                            for slot in ("rk_ru_m", "rk_rv_m", "rk_ww_m")))
        if n == 3:
            at_history = {k: cp.asnumpy(v).copy()
                          for k, v in history_fields(on).items()}
    for name in ("u", "v", "w", "thp", "php", "mup", "qv", "qc", "qr", "p"):
        assert np.array_equal(cp.asnumpy(getattr(on, name)),
                              cp.asnumpy(getattr(off, name))), name

    def mean(window):
        out = []
        for index in range(3):
            avg = np.zeros_like(window[0][index])
            for count, sample in enumerate(window):
                avg = ((np.float32(count) * avg + sample[index])
                       / np.float32(count + 1)).astype(np.float32)
            out.append(avg)
        return out

    names = ("AVGFLX_RUM", "AVGFLX_RVM", "AVGFLX_WWM")
    first = mean(fluxes[0:3])
    second = mean(fluxes[3:6])
    final = {k: cp.asnumpy(v) for k, v in history_fields(on).items()}
    assert sorted(final) == sorted(names)
    for index, name in enumerate(names):
        assert np.array_equal(at_history[name], first[index]), name
        assert np.array_equal(final[name], second[index]), name


def test_avgflx_history_and_restart_classification():
    from woof.io.restart import classify_scratch_slot
    from woof.io.wrf_output_schema import REGISTRY_VAR_META
    assert REGISTRY_VAR_META["AVGFLX_RUM"] == (
        "hist-time-averaged mu-coupled u", "Pa m s-1")
    assert REGISTRY_VAR_META["AVGFLX_WWM"] == (
        "hist-time-averaged mu-coupled eta-dot", "Pa s-1")
    for slot in ("avgflx_rum", "avgflx_rvm", "avgflx_wwm"):
        assert classify_scratch_slot(slot) == "serialize"
    assert classify_scratch_slot("avgflx_count") == "rebuild"
    for slot in ("theta_m_stash", "theta_m_dry", "theta_m_rtheta",
                 "theta_m_qv_diabatic"):
        assert classify_scratch_slot(slot) == "rebuild"


# --- The real-data-like configuration (woof/verify/cases/theta_m_real_like.py):
# Thompson (mp 8) + RRTMG + YSU or MYNN + Grell-Freitas on a 12 km patch.

REAL_LIKE_STEPS = 20


def _dry_total(state) -> float:
    """sum((c1h*mu + c2h)*theta*(-dnw)) of the DRY theta, whichever the
    state integrates."""
    import cupy as cp
    from woof.core.theta_m import dry_theta
    f8 = cp.float64
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    theta = thb.astype(f8) + dry_theta(state).astype(f8)
    chm = (state.c1h[:, None, None].astype(f8)
           * (state.mub2d + state.mup)[None].astype(f8)
           + state.c2h[:, None, None].astype(f8))
    return float(cp.sum(theta * chm * (-state.dnw[:, None, None].astype(f8))))


@pytest.mark.parametrize("pbl", ("ysu", "mynn"))
def test_real_like_thompson_gf_runs_theta_m_stably_and_conservatively(pbl):
    """Both runs are stable, and the theta_m run's dry-theta budget is the
    dry run's: the coupled dry-theta total moves by what the physics puts in
    (radiation, surface flux, GF, Thompson), identically in both modes to
    the float32 envelope the dycore conserves to (2e-7; measured on the RTX
    A6000: YSU 5.9e-8, MYNN 1.1e-8 after 30 steps)."""
    import cupy as cp
    from woof.core.dycore import run_steps, stability_report
    from woof.core.theta_m import dry_theta
    from woof.verify.cases import theta_m_real_like as case
    drift, states = {}, {}
    for flag in (0, 1):
        cfg = case.config(use_theta_m=flag, pbl=pbl)
        state, _driver = case.build(cfg)
        assert state.theta_m == bool(flag)
        before = _dry_total(state)
        run_steps(state, cfg, n=REAL_LIKE_STEPS)
        report = stability_report(state, cfg)
        assert not report["nan"], report
        assert report["w_max"] < 1.0 and report["u_max"] < 5.0, report
        drift[flag] = (_dry_total(state) - before) / before
        states[flag] = state
    # The physics heats the patch (a nonzero budget to agree on) ...
    assert drift[0] > 1e-5
    # ... and the theta_m run's dry-theta budget is the dry run's.
    assert abs(drift[1] - drift[0]) < 2e-7, drift
    theta = cp.asnumpy(dry_theta(states[1]))
    assert np.abs(theta - cp.asnumpy(states[0].thp)).max() < 0.05
    assert float(cp.abs(states[1].qv - states[0].qv).max()) < 1e-4


def test_real_like_dry_run_launches_no_theta_m_kernel(monkeypatch):
    """use_theta_m = 0 on the real-like configuration reaches none of the
    theta_m units (the conversions, the RTHFTEN block, the hflux variant)."""
    import woof.core.kernels as kernels
    from woof.core.dycore import run_steps
    from woof.verify.cases import theta_m_real_like as case
    real = kernels.get_kernel
    new = ("theta_m", "theta_m_cumulus", "smag2d_theta_m")

    def guarded(name, func, *args, **kwargs):
        if name in new:
            raise AssertionError(f"{name}.{func} launched on a dry-theta run")
        return real(name, func, *args, **kwargs)

    monkeypatch.setattr(kernels, "get_kernel", guarded)
    cfg = case.config(use_theta_m=0)
    state, _driver = case.build(cfg)
    run_steps(state, cfg, n=2)
    assert not state.theta_m
    for slot in ("theta_m_stash", "theta_m_h_pre", "theta_m_th_phy"):
        assert state.existing_scratch(slot) is None, slot
