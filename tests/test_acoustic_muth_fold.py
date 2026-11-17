"""advance_mu_th_msf's default arm (kernels/acoustic.cu) saves th_pp_old and
p_pp_old inside its theta'' update loops instead of a separate copy pass
first; against that copy pass (MUTH_SEPARATE_COPY=1), word for word over
whole substeps, in each numerics mode.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import _numerics_child
from conftest import requires_gpu


def _substep_words(get_mu, *, nested, damp, top_lid, moist, shape):
    import cupy as cp

    from woof.core import acoustic, dycore
    from woof.verify.npref import random_acoustic_state

    nz, ny, nx = shape
    # Stretched hybrid levels: WRF-like fnm != fnp and c1/c2 weights, so a
    # contraction the pass adds or drops cannot hide behind 0.5 products.
    state, cfg = random_acoustic_state(seed=11, nz=nz, ny=ny, nx=nx, stretch=1.2,
                                       hybrid_opt=2, msf_amp=0.1, hill_height=300.0,
                                       moist=moist, mp_physics=8 if moist else 0)
    assert not np.all(cp.asnumpy(state.fnm)[1:] == np.float32(0.5))
    rng = np.random.default_rng(12)
    if moist:
        for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
            array = getattr(state, name, None)
            if array is not None:
                array[...] = cp.asarray(1e-3 * rng.random(array.shape), dtype=cp.float32)
    cfg = dataclasses.replace(cfg, specified=True, spec_zone=1, relax_zone=4,
                              spec_bdy_width=5, nested=nested, emdiv=0.01, epssm=0.1,
                              top_lid=top_lid, damp_opt=3 if damp else 0, dampcoef=0.2)
    if damp:
        # A damping layer through the upper third of the column.
        cfg = dataclasses.replace(cfg, zdamp=float(cp.asnumpy(state.phb).max() / 9.81 / 3.0))
    nz, ny, nx = state.p.shape
    mudf = state.scratch(state.mup.shape, "acoustic_mudf")
    mudf[...] = cp.asarray(rng.standard_normal(mudf.shape), dtype=cp.float32)
    sums = tuple(state.scratch(shape, name) for shape, name in (
        ((nz, ny, nx + 1), "rk_ru_m"), ((nz, ny + 1, nx), "rk_rv_m"),
        ((nz + 1, ny, nx), "rk_ww_m")))
    histories = (state.scratch((ny, nx), "acoustic_mu_pp_old"),
                 state.scratch((nz, ny, nx), "acoustic_th_pp_old"))
    for array in histories:     # poisoned: every substep overwrites them
        array[...] = cp.asarray(rng.standard_normal(array.shape), dtype=cp.float32)
    state.p_pp_old[...] = cp.asarray(rng.standard_normal(state.p_pp_old.shape),
                                     dtype=cp.float32)
    for array in sums:          # poisoned: the first substep zeroes them
        array[...] = cp.asarray(rng.standard_normal(array.shape), dtype=cp.float32)
    dtau = 1.25
    real_get = acoustic.get_kernel
    picked = []

    def get_kernel(name, func, shape=()):
        if func == "advance_mu_th_msf":
            picked.append(func)
            return get_mu(name, func, shape)
        return real_get(name, func, shape)

    acoustic.get_kernel = get_kernel
    try:
        coefficients = acoustic.prepare_acoustic_coefficients(state, cfg, dtau)
        emdiv = dycore._prepare_emdiv_filter_launch(state, cfg, mudf)
        launch = acoustic.prepare_acoustic_substep_launch(
            state, cfg, dtau, coefficients, mudf=mudf, emdiv_args=emdiv.args, sumflux=sums)
    finally:
        acoustic.get_kernel = real_get
    assert picked, "the substep did not take advance_mu_th_msf"
    absorbed = launch.absorbs
    zero = dycore._prepare_sumflux_launch("zero_sumflux", sums)
    accumulate = dycore._prepare_sumflux_launch(
        "accumulate_sumflux", sums, (state.u_pp, state.v_pp, state.ww_pp))
    for index in range(3):
        first = index == 0
        if "emdiv" not in absorbed:
            emdiv()
        if first and "sumflux" not in absorbed:
            zero()
        launch(first=first)
        if "sumflux" not in absorbed:
            accumulate()
    cp.cuda.runtime.deviceSynchronize()
    words = {name: cp.asnumpy(value).view(np.uint32) for name, value in vars(state).items()
             if isinstance(value, cp.ndarray) and value.dtype == cp.float32}
    for name, array in zip(("ru_m", "rv_m", "ww_m"), sums):
        words[name] = cp.asnumpy(array).view(np.uint32)
    for name, array in zip(("mu_pp_old", "th_pp_old"), histories):
        words[name] = cp.asnumpy(array).view(np.uint32)
    return words


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("moist", [False, True])
@pytest.mark.parametrize("shape,specialized", [((20, 9, 13), True), ((34, 11, 14), False)])
def test_folded_copy_writes_the_separate_copy_words(nested, moist, shape, specialized):
    from woof.core import acoustic
    from woof.core.kernels import get_kernel, get_kernel_int_defines

    if acoustic.WRF_EXACT:
        pytest.skip("the WRF-exact arm keeps its own copy")

    def folded(name, func, shape=()):
        return get_kernel(name, func, shape if specialized else ())

    def separate(name, func, shape=()):
        return get_kernel_int_defines(name, func, (("MUTH_SEPARATE_COPY", 1),),
                                      shape if specialized else ())

    kw = dict(nested=nested, damp=True, top_lid=0, moist=moist, shape=shape)
    reference = _substep_words(separate, **kw)
    candidate = _substep_words(folded, **kw)
    assert reference.keys() == candidate.keys()
    assert {"th_pp", "th_pp_old", "p_pp_old"} <= reference.keys()
    for name in reference:
        differ = int(np.count_nonzero(reference[name] != candidate[name]))
        assert differ == 0, f"{name}: {differ} words differ"


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize(*_numerics_child.MODES)
def test_folded_copy_words_hold_under_each_numerics(numerics, exact):
    """The battery runs strict; production runs fast, where NVRTC (and
    Metal's fp contract(fast)) may contract the rewritten update differently."""
    _numerics_child.run(__file__, ["test_folded_copy_writes_the_separate_copy_words"],
                        numerics, exact)
