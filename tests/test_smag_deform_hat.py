"""wrf_smag_uvhat + wrf_smag_deform_hat (kernels/smag2d.cu's
GPUWM_SMAG_STORED_HAT=1 tier): u/msfu and v/msfv divided once per face and
stored, then read by the deformation kernel in place of the division at
every stencil point; word for word the wrf_smag_deform tensors
(dycore.launch_wrf_smag2d_km; default on Metal only), over periodic,
bounded and mixed axes, one and several launch blocks, 1-D and 3-D base
geopotential, moist and dry density, under fast, strict and WRF-exact
numerics.
"""

from __future__ import annotations

import numpy as np
import pytest

import _numerics_child
from conftest import requires_gpu
from test_smag_multi import _grid


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("phb3d,moist,bx,by,nx", [
    (0, 1, 1, 1, 17), (1, 1, 0, 0, 17), (1, 0, 1, 1, 17), (0, 0, 0, 0, 17),
    (1, 1, 1, 0, 17), (0, 1, 0, 1, 17), (1, 1, 1, 0, 63), (0, 1, 0, 1, 64),
])
def test_stored_hats_keep_the_deformation_words(phb3d, moist, bx, by, nx):
    import cupy as cp
    from woof.core.dycore import _SMAG_HAT_DEFINES
    from woof.core.kernels import get_kernel, get_kernel_int_defines

    nz, ny = 11, 13
    tpb = 64
    rng = np.random.default_rng(41 + phb3d + 2 * moist + 4 * bx + 8 * by + nx)
    common, tail = _grid(rng, nz, ny, nx, phb3d=phb3d, moist=moist, boundary=bx)
    tail[-1] = np.int32(by)
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
    get_kernel_int_defines("smag2d", "wrf_smag_uvhat", _SMAG_HAT_DEFINES)(
        ((nx + tpb) // tpb, ny + 1, nz), (tpb, 1, 1),
        (u, v, msfu, msfv, uhat, vhat, np.int32(nz), np.int32(ny), np.int32(nx)))
    get_kernel_int_defines("smag2d", "wrf_smag_deform_hat", _SMAG_HAT_DEFINES)(
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
    forces either on any backend, and unset or empty keeps the default."""
    import woof.backend as wb
    from woof.core import dycore

    for unset in (None, "", "  "):
        if unset is None:
            monkeypatch.delenv("WOOF_SMAG_DEFORM_HAT", raising=False)
        else:
            monkeypatch.setenv("WOOF_SMAG_DEFORM_HAT", unset)
        monkeypatch.setattr(wb, "active", lambda: "mlx")
        assert dycore._smag_deform_hat_enabled()
        monkeypatch.setattr(wb, "active", lambda: "cuda")
        assert not dycore._smag_deform_hat_enabled()
    for backend in ("mlx", "cuda"):
        monkeypatch.setattr(wb, "active", lambda: backend)
        for on in ("1", "true", "On ", "yes"):
            monkeypatch.setenv("WOOF_SMAG_DEFORM_HAT", on)
            assert dycore._smag_deform_hat_enabled(), on
        for off in ("0", "false", " OFF", "no"):
            monkeypatch.setenv("WOOF_SMAG_DEFORM_HAT", off)
            assert not dycore._smag_deform_hat_enabled(), off


def test_overlapping_hat_backings_are_refused():
    """The hats need two separate backings (diff6_x and diff6_y)."""
    from woof.core.dycore import _disjoint

    class _Buf:
        def __init__(self, ptr, nbytes):
            self.data = type("D", (), {"ptr": ptr})()
            self.nbytes = nbytes

    assert _disjoint(_Buf(0, 64), _Buf(64, 64))
    assert _disjoint(_Buf(128, 64), _Buf(0, 128))
    assert not _disjoint(_Buf(0, 65), _Buf(64, 64))
    assert not _disjoint(_Buf(0, 64), _Buf(0, 64))


def test_default_smag2d_text_has_no_stored_hat_code():
    """Every stored-hat line sits under GPUWM_SMAG_STORED_HAT, so the
    default smag2d module and the units composing smag2d.cu compile the
    same text as before the tier existed."""
    import re
    from pathlib import Path
    import woof.core.kernels as kernels

    text = (Path(kernels.__file__).parent / "smag2d.cu").read_text()
    stripped = re.sub(r"#ifdef GPUWM_SMAG_STORED_HAT\n.*?#endif\n", "", text,
                      flags=re.S)
    for word in ("uhat", "vhat", "wrf_smag_uvhat", "wrf_smag_deform_hat"):
        assert not re.search(rf"\b{word}\b|q\.{word}", stripped), word
