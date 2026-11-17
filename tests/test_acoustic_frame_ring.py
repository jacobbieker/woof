"""advance_frame_phi_w_ring (kernels/acoustic.cu): the specified/nested
frame's phi'', w'' and p''/alpha'' update over the ring's columns with
level lanes, against advance_specified_phi_w / advance_nested_phi_w (one
thread per column of the whole domain), word for word, in each numerics
mode; and the launcher's selection (acoustic._frame_ring_launch).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import _numerics_child
from conftest import requires_gpu

_WRITTEN = ("ph_pp", "w_pp", "p_pp", "al_pp")
_POISON = np.uint32(0x7FC0DEAD)     # a quiet NaN no kernel computes


def _frame_words(ring, *, nested, spec_zone, shape, seed):
    import cupy as cp

    from woof.core import acoustic
    from woof.core.kernels import get_kernel
    from woof.verify.npref import random_acoustic_state

    nz, ny, nx = shape
    # Stretched hybrid levels: WRF-like c1/c2 and fnm != fnp weights.
    state, cfg = random_acoustic_state(seed=seed, nz=nz, ny=ny, nx=nx, stretch=1.2,
                                       hybrid_opt=2, msf_amp=0.1, hill_height=300.0)
    cfg = dataclasses.replace(cfg, specified=True, spec_zone=spec_zone,
                              relax_zone=spec_zone + 3, spec_bdy_width=spec_zone + 4,
                              nested=nested)
    assert acoustic._frame_takes_table_w(cfg) == nested
    rng = np.random.default_rng(seed + 1)
    for name in ("ph_pp", "w_pp", "rph_t", "rw_t", "rmu_t", "mu_pp", "th_pp"):
        array = getattr(state, name)
        array[...] = cp.asarray(rng.standard_normal(array.shape) * 0.1
                                + cp.asnumpy(array), dtype=cp.float32)
    for name in ("p_pp", "al_pp"):     # poisoned: the frame rewrites its ring
        getattr(state, name).view(cp.uint32)[...] = _POISON
    dtau = 1.25
    c2a = acoustic.prepare_acoustic_coefficients(state, cfg, dtau)[0]
    nz, ny, nx = state.p.shape
    if ring:
        launch = acoustic._frame_ring_launch(state, cfg, acoustic.DTYPE(dtau), c2a, nz, ny, nx)
        assert launch is not None
        launch()
    else:
        common = (state.ph_pp, state.w_pp, state.p_pp, state.al_pp, state.rph_t)
        rest = (state.th_pp, state.mup, state.mu_pp, state.mub2d, state.rmu_t,
                state.php, state.thp, state.thb, state.alt, c2a, state.rdnw,
                state.c1h, state.c2h, state.c1f, state.c2f, acoustic.DTYPE(dtau),
                np.int32(spec_zone), acoustic._base3d(state),
                np.int32(nz), np.int32(ny), np.int32(nx))
        if nested:
            kernel = get_kernel("acoustic", "advance_nested_phi_w", (nz, ny, nx))
            args = common + (state.rw_t,) + rest
        else:
            kernel = get_kernel("acoustic", "advance_specified_phi_w")
            args = common + rest
        kernel(((ny * nx + 255) // 256,), (256,), args)
    cp.cuda.runtime.deviceSynchronize()
    return {name: cp.asnumpy(getattr(state, name)).view(np.uint32) for name in _WRITTEN}


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("spec_zone", [1, 3])
@pytest.mark.parametrize("shape", [(10, 9, 13), (34, 16, 12), (49, 20, 23)])
def test_ring_writes_the_frame_kernels_words(nested, spec_zone, shape):
    from woof.core import acoustic

    if acoustic.WRF_EXACT:
        pytest.skip("the WRF-exact build keeps the column frame kernels")
    reference = _frame_words(False, nested=nested, spec_zone=spec_zone, shape=shape, seed=7)
    candidate = _frame_words(True, nested=nested, spec_zone=spec_zone, shape=shape, seed=7)
    nz, ny, nx = shape
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    frame = ((i < spec_zone) | (i >= nx - spec_zone)
             | (j < spec_zone) | (j >= ny - spec_zone))
    for name in _WRITTEN:
        differ = int(np.count_nonzero(reference[name] != candidate[name]))
        assert differ == 0, f"{name}: {differ} words differ"
    # The ring's p''/alpha'' were all written; the interior's left alone.
    for name in ("p_pp", "al_pp"):
        assert not np.any(candidate[name][:, frame] == _POISON), name
        assert np.all(candidate[name][:, ~frame] == _POISON), name


def test_a_ring_as_wide_as_half_the_domain_keeps_the_column_kernels():
    from types import SimpleNamespace

    from woof.core import acoustic

    cfg = SimpleNamespace(spec_zone=5)
    assert acoustic._frame_ring_launch(None, cfg, None, None, 10, 10, 30) is None
    assert acoustic._frame_ring_launch(None, cfg, None, None, 10, 30, 10) is None


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize(*_numerics_child.MODES)
def test_ring_words_hold_under_each_numerics(numerics, exact):
    """The battery runs strict; production runs fast, where NVRTC (and
    Metal's fp contract(fast)) may contract the per-level diagnosis
    differently in the lane loop."""
    _numerics_child.run(__file__, ["test_ring_writes_the_frame_kernels_words"],
                        numerics, exact)
