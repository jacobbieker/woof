"""QNSE (bl_pbl_physics=4, with its mass flux) and BouLac (8) on 256 levels.

``QNSE_KMAX`` / ``BOULAC_KMAX`` size per-thread column arrays, so the
launchers compile one of two tiers (woof.core.pbl_level_tiers): the
unchanged source, whose define is 128, for every column of at most 128
levels, and a 256 tier above.  Three claims per scheme:

1. A 256-level column to a 1 hPa top runs through the SHIPPED launcher and
   every output is finite and non-trivial (the kernels' own ``nz > KMAX``
   guards return without writing, which "finite and moved" would catch).
2. At nz in {49, 128} the shipped launcher takes the unchanged-source tier,
   and forcing the 256 tier onto the same column reproduces every output
   word (raw bits): the tier changes where the column lives, not what is
   computed.
3. Optional CUDA-vs-Metal agreement: with ``WOOF_PBL256_XBACKEND_DIR`` set
   each backend's 256-level outputs are written there and, when the other
   backend's file is present, compared.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_physics_256_levels_gpu import (  # noqa: E402
    _assert_bitwise, _assert_ran, _atmosphere, _bits, _f32)
from woof.core import pbl_level_tiers  # noqa: E402

pytestmark = pytest.mark.gpu

DEEP = 256
SHIPPED = 128
NCOL = 48


def test_the_tier_ladder_and_contract_agree():
    from woof.physics_vertical_contract import (
        BOULAC_VERTICAL_LEVEL_BOUNDS, QNSE_VERTICAL_LEVEL_BOUNDS)

    assert [pbl_level_tiers.kmax_tier(n) for n in (4, 49, 128, 129, 256)] == [
        128, 128, 128, 256, 256]
    with pytest.raises(ValueError):
        pbl_level_tiers.kmax_tier(257)
    assert QNSE_VERTICAL_LEVEL_BOUNDS == (4, DEEP)
    assert BOULAC_VERTICAL_LEVEL_BOUNDS == (3, DEEP)


def _qnse_inputs(cp, nz, ncol=NCOL):
    atm = _atmosphere(nz, ncol)
    rng = np.random.default_rng(100 + nz)
    t = atm["temperature"]
    rho = atm["pressure"] / (287.0 * t)
    lev = lambda a: _f32(cp, a)  # noqa: E731

    def sfc(lo, hi):
        return _f32(cp, rng.uniform(lo, hi, (1, ncol)))

    columns = dict(dz=lev(atm["dz"]), u=lev(atm["u"]), v=lev(atm["v"]),
                   t=lev(t), th=lev(atm["theta"]), exner=lev(atm["exner"]),
                   qv=lev(atm["qv"]),
                   qc=lev(rng.uniform(0, 3e-5, (nz, 1, ncol))),
                   p=lev(atm["pressure"]), rho=lev(rho))
    surface = dict(ht=sfc(0, 0), psfc=sfc(1e5, 1e5), ust=sfc(0.1, 0.7),
                   chklowq=sfc(1, 1), xland=_f32(cp, rng.choice(
                       [1.0, 2.0], (1, ncol))), sice=sfc(0, 0),
                   snow=sfc(0, 0), akhs=sfc(0.01, 0.06),
                   akms=sfc(0.02, 0.1), elflx=sfc(10, 200),
                   hfx=sfc(-30, 250), uz0=sfc(0, 0), vz0=sfc(0, 0),
                   corf=sfc(1e-4, 1e-4))
    state = dict(thz0=sfc(298, 302), qz0=sfc(0.012, 0.018),
                 qsfc=sfc(0.012, 0.018), ct=sfc(0, 0))
    tke = lev(rng.uniform(0.05, 0.6, (nz, 1, ncol)))
    qfx = sfc(1e-5, 2e-4)
    return columns, surface, state, tke, qfx


def _run_qnse(cp, nz, mfshconv):
    from woof.core.myjpbl import (MFSHCONV_DIAGNOSTICS_3D,
                                  MFSHCONV_OUTPUTS_3D, qnse_pbl_step)

    columns, surface, state, tke, qfx = _qnse_inputs(cp, nz)
    surface["qfx"] = qfx
    mf_state = {"rc_mf": cp.zeros((nz, 1, NCOL), dtype=cp.float32)}
    mf_outputs = {n: cp.zeros((nz, 1, NCOL), dtype=cp.float32)
                  for n in MFSHCONV_OUTPUTS_3D}
    diag = {n: cp.zeros((nz, 1, NCOL), dtype=cp.float32)
            for n in MFSHCONV_DIAGNOSTICS_3D}
    exch = lambda: cp.zeros((nz, 1, NCOL), dtype=cp.float32)  # noqa: E731
    out = qnse_pbl_step(columns, surface, state, tke, dtturbl=30.0,
                        exch_h=exch(), exch_m=exch(), mfshconv=mfshconv,
                        mf_state=mf_state, mf_outputs=mf_outputs,
                        diagnostics=diag)
    result = _bits(out)
    result["tke"] = np.asarray(cp.asnumpy(tke)).copy()
    for name, array in state.items():
        result["state_" + name] = np.asarray(cp.asnumpy(array)).copy()
    if mfshconv:
        for name, array in {**mf_outputs, **mf_state, **diag}.items():
            result["mf_" + name] = np.asarray(cp.asnumpy(array)).copy()
    return result


def _boulac_inputs(cp, nz, ncol=NCOL):
    atm = _atmosphere(nz, ncol)
    rng = np.random.default_rng(200 + nz)
    t = atm["temperature"]
    rho = atm["pressure"] / (287.0 * t)
    lev = lambda a: _f32(cp, a)  # noqa: E731

    def sfc(lo, hi):
        return _f32(cp, rng.uniform(lo, hi, (1, ncol)))

    columns = dict(dz=lev(atm["dz"]), u=lev(atm["u"]), v=lev(atm["v"]),
                   th=lev(atm["theta"]), rho=lev(rho), qv=lev(atm["qv"]),
                   qc=lev(rng.uniform(0, 3e-5, (nz, 1, ncol))))
    surface = dict(hfx=sfc(-30, 250), qfx=sfc(1e-5, 2e-4), ust=sfc(0.1, 0.7))
    tke = lev(rng.uniform(1e-4, 0.6, (nz, 1, ncol)))
    return columns, surface, tke


def _run_boulac(cp, nz):
    from woof.core.boulac import launch_boulac

    columns, surface, tke = _boulac_inputs(cp, nz)
    out = launch_boulac(columns, surface, tke, dt=30.0, cp_air=1004.5,
                        g=9.81)
    result = _bits(out)
    result["tke"] = np.asarray(cp.asnumpy(tke)).copy()
    return result


def _check_xbackend(label, result):
    """Write this backend's words; compare with the other backend's file."""
    directory = os.environ.get("WOOF_PBL256_XBACKEND_DIR")
    if not directory:
        return
    from woof import backend

    os.makedirs(directory, exist_ok=True)
    mine = backend.active()
    np.savez(os.path.join(directory, f"{label}.{mine}.npz"), **result)
    other = "cuda" if mine != "cuda" else "metal"
    path = os.path.join(directory, f"{label}.{other}.npz")
    if not os.path.exists(path):
        return
    with np.load(path) as theirs:
        for name in sorted(result):
            a, b = result[name], theirs[name]
            if a.dtype.kind != "f":
                continue
            scale = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-30)
            err = float(np.abs(a.astype(np.float64) - b).max()) / scale
            assert err < 5e-3, f"{label}/{name}: CUDA vs Metal {err:.3e}"


@pytest.mark.parametrize("mfshconv", [False, True])
def test_qnse_runs_a_256_level_column_to_a_1_hpa_top(mfshconv):
    import cupy as cp

    out = _run_qnse(cp, DEEP, mfshconv)
    assert out["rublten"].shape == (DEEP, 1, NCOL)
    _assert_ran(out, ("rublten", "rthblten", "exch_h", "el_myj", "tke"))
    assert (out["pblh"] > 0).all()
    assert ((out["kpbl"] >= 1) & (out["kpbl"] < DEEP)).all()
    if mfshconv:
        assert np.any(out["mf_massflux_edkf"] != 0)
    _check_xbackend(f"qnse_mf{int(mfshconv)}", out)


@pytest.mark.parametrize("nz", [49, SHIPPED])
@pytest.mark.parametrize("mfshconv", [False, True])
def test_qnse_256_tier_reproduces_the_unchanged_kernel(nz, mfshconv,
                                                      monkeypatch):
    import cupy as cp

    from woof.core import myjpbl

    calls = []
    real = myjpbl.kmax_tier
    monkeypatch.setattr(myjpbl, "kmax_tier",
                        lambda n: calls.append(real(n)) or real(n))
    shipped = _run_qnse(cp, nz, mfshconv)
    assert set(calls) == {SHIPPED}, "nz <= 128 must take the unchanged tier"
    _assert_ran(shipped, ("rublten", "rthblten", "exch_h"))
    monkeypatch.setattr(myjpbl, "kmax_tier", lambda n: DEEP)
    deep = _run_qnse(cp, nz, mfshconv)
    _assert_bitwise(deep, shipped, f"QNSE_KMAX 128 -> 256 at nz={nz}")


def test_boulac_runs_a_256_level_column_to_a_1_hpa_top():
    import cupy as cp

    out = _run_boulac(cp, DEEP)
    assert out["rublten"].shape == (DEEP, 1, NCOL)
    _assert_ran(out, ("rublten", "rthblten", "tke"))
    assert (out["pblh"] > 0).all()
    _check_xbackend("boulac", out)


@pytest.mark.parametrize("nz", [49, SHIPPED])
def test_boulac_256_tier_reproduces_the_unchanged_kernel(nz, monkeypatch):
    import cupy as cp

    from woof.core import boulac

    calls = []
    real = boulac.kmax_tier
    monkeypatch.setattr(boulac, "kmax_tier",
                        lambda n: calls.append(real(n)) or real(n))
    shipped = _run_boulac(cp, nz)
    assert set(calls) == {SHIPPED}, "nz <= 128 must take the unchanged tier"
    _assert_ran(shipped, ("rublten", "rthblten"))
    monkeypatch.setattr(boulac, "kmax_tier", lambda n: DEEP)
    deep = _run_boulac(cp, nz)
    _assert_bitwise(deep, shipped, f"BOULAC_KMAX 128 -> 256 at nz={nz}")
