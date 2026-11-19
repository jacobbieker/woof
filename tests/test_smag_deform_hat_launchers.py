"""The stored-hat deformation (dycore.launch_wrf_deform; default on Metal)
in every closure that computes WRF's deformation: km_opt = 1 (isotropic,
hats in the u/v carrying buffers), 2 (TKE), 3 (3-D Smagorinsky), 4 (2-D
Smagorinsky) and 5 (SMS-3DTKE), hats in the diff6 face workspaces.  A few
whole-model steps with WOOF_SMAG_DEFORM_HAT=0 and =1 leave every prognostic
word the same, on non-unity map factors (so the stored division is not the
exact divide-by-one) with random winds, in child processes under fast,
strict and WRF-exact numerics.  A hat slot still live when the closure ran
(a clobbered carrying buffer or BN2) would change the words.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import _numerics_child
from conftest import requires_gpu

FIELDS = ("u", "v", "w", "thp", "php", "mup", "qv", "tke")


def _state(km_opt, diff_opt):
    import cupy as cp
    import test_sms3dtke_model as sms

    extra = {"khdif": 40.0, "kvdif": 3.0} if km_opt == 1 else {}
    base = dict(sms.BASE)
    try:
        sms.BASE.update(diff_opt=diff_opt, **extra)
        state, cfg = sms._state(cp, km_opt)
    finally:
        sms.BASE.clear()
        sms.BASE.update(base)
    ny, nx = cfg.ny, cfg.nx
    xx = 2.0 * np.pi * np.arange(nx) / nx
    yy = 2.0 * np.pi * np.arange(ny) / ny
    msft = 1.04 + 0.03 * np.sin(xx)[None, :] + 0.02 * np.cos(yy)[:, None]
    msfu = 0.5 * (msft + np.roll(msft, 1, axis=1))
    msfv = 0.5 * (msft + np.roll(msft, 1, axis=0))
    state.set_map_coriolis(
        msft=msft, msfu=np.concatenate([msfu, msfu[:, :1]], axis=1),
        msfv=np.concatenate([msfv, msfv[:1, :]], axis=0))
    rng = np.random.default_rng(7 + km_opt)
    state.u[...] += cp.asarray(2.0 * rng.standard_normal(state.u.shape), dtype=cp.float32)
    state.v[...] += cp.asarray(2.0 * rng.standard_normal(state.v.shape), dtype=cp.float32)
    if getattr(state, "tke", None) is not None:
        state.tke[...] = cp.asarray(0.5 * np.abs(rng.standard_normal(state.tke.shape)),
                                    dtype=cp.float32)
    return state, cfg


def _words(km_opt, diff_opt, route):
    import cupy as cp
    from woof.core import dycore

    os.environ["WOOF_SMAG_DEFORM_HAT"] = route
    state, cfg = _state(km_opt, diff_opt)
    seen = []
    tiered = dycore.get_kernel_int_defines

    def spy(module, name, defines):
        seen.append(name)
        return tiered(module, name, defines)
    dycore.get_kernel_int_defines = spy
    try:
        dycore.run_steps(state, cfg, 3)
    finally:
        dycore.get_kernel_int_defines = tiered
    # Each step's closure took the route asked for.
    assert seen.count("wrf_smag_deform_hat") == (3 if route == "1" else 0), seen
    out = {}
    for name in FIELDS:
        a = getattr(state, name, None)
        if a is not None:
            out[name] = cp.asnumpy(a).view(np.uint32).copy()
    return out


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("km_opt,diff_opt", [(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (2, 1)])
def test_stored_hats_keep_the_model_words(km_opt, diff_opt):
    off = _words(km_opt, diff_opt, "0")
    on = _words(km_opt, diff_opt, "1")
    assert np.all(np.isfinite(off["u"].view(np.float32)))
    for name in off:
        assert np.array_equal(off[name], on[name]), (km_opt, diff_opt, name)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize(*_numerics_child.MODES)
def test_stored_hat_model_words_hold_under_each_numerics(numerics, exact):
    """The battery runs strict; production runs fast.  Each mode in its own child."""
    _numerics_child.run(__file__, ["test_stored_hats_keep_the_model_words"], numerics, exact)
