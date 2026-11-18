"""wrf_smag_uvhat + wrf_smag_deform_hat (kernels/smag2d.cu): u/msfu and
v/msfv divided once per face and stored, then read by the deformation
kernel in place of the division at every stencil point; word for word the
wrf_smag_deform tensors (dycore.launch_wrf_smag2d_km; default on Metal
only), over periodic and bounded axes, 1-D and 3-D base geopotential, moist
and dry density, under fast, strict and WRF-exact numerics.
"""

from __future__ import annotations

import numpy as np
import pytest

import _numerics_child
from conftest import requires_gpu
from test_smag_multi import _grid


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("phb3d,moist,boundary", [
    (0, 1, 1), (1, 1, 0), (1, 0, 1), (0, 0, 0),
])
def test_stored_hats_keep_the_deformation_words(phb3d, moist, boundary):
    import cupy as cp
    from woof.core.kernels import get_kernel

    nz, ny, nx = 11, 13, 17
    tpb = 64
    rng = np.random.default_rng(41 + phb3d + 2 * moist + 4 * boundary)
    common, tail = _grid(rng, nz, ny, nx, phb3d=phb3d, moist=moist, boundary=boundary)
    # WRF-like non-zero third surface weight (_grid's cf3 = 0 would hide
    # the third level's hats in the k = 0 extrapolation).
    common[20] = np.float32(0.12)
    u, v, msfu, msfv = common[0], common[1], common[8], common[9]
    grid = ((nx + tpb - 1) // tpb, ny, nz)

    def tensors():
        return [cp.full((nz, ny, nx), np.nan, cp.float32) for _ in range(3)]

    ref = tensors()
    get_kernel("smag2d", "wrf_smag_deform")(grid, (tpb, 1, 1), tuple(common + ref + tail))
    got = tensors()
    # Poisoned hats: every face the deformation reads must be written.
    uhat = cp.full((nz, ny, nx + 1), np.nan, cp.float32)
    vhat = cp.full((nz, ny + 1, nx), np.nan, cp.float32)
    get_kernel("smag2d", "wrf_smag_uvhat")(
        ((nx + tpb) // tpb, ny + 1, nz), (tpb, 1, 1),
        (u, v, msfu, msfv, uhat, vhat, np.int32(nz), np.int32(ny), np.int32(nx)))
    get_kernel("smag2d", "wrf_smag_deform_hat")(
        grid, (tpb, 1, 1), tuple(common + got + tail + [uhat, vhat]))
    for name, a, b in zip(("d11", "d22", "d12"), ref, got):
        a = cp.asnumpy(a)
        assert np.all(np.isfinite(a)), name
        assert np.array_equal(a.view(np.uint32), cp.asnumpy(b).view(np.uint32)), name


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize(*_numerics_child.MODES)
def test_stored_hat_words_hold_under_each_numerics(numerics, exact):
    """The battery runs strict; production runs fast (contraction allowed).
    Each mode in its own child process."""
    _numerics_child.run(__file__, ["test_stored_hats_keep_the_deformation_words"],
                        numerics, exact)


def test_stored_hats_are_metal_only_by_default(monkeypatch):
    """Metal takes the stored-hat pair by default; CUDA keeps
    wrf_smag_deform (the extra pass did not pay there); the variable
    forces either on any backend and refuses anything but 0/1."""
    import woof.backend as wb
    from woof.core import dycore

    monkeypatch.delenv("WOOF_SMAG_DEFORM_HAT", raising=False)
    monkeypatch.setattr(wb, "active", lambda: "mlx")
    assert dycore._smag_deform_hat_enabled()
    monkeypatch.setattr(wb, "active", lambda: "cuda")
    assert not dycore._smag_deform_hat_enabled()
    monkeypatch.setenv("WOOF_SMAG_DEFORM_HAT", "1")
    assert dycore._smag_deform_hat_enabled()
    monkeypatch.setattr(wb, "active", lambda: "mlx")
    monkeypatch.setenv("WOOF_SMAG_DEFORM_HAT", "0")
    assert not dycore._smag_deform_hat_enabled()
    for bad in ("", "true", "2"):
        monkeypatch.setenv("WOOF_SMAG_DEFORM_HAT", bad)
        with pytest.raises(ValueError, match="WOOF_SMAG_DEFORM_HAT"):
            dycore._smag_deform_hat_enabled()
