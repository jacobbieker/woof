"""WRF's w_damping limiter as an MPAS vertical-momentum tendency term.

WRF ``w_damp`` (dyn_em/module_big_step_utilities_em.F, ``w_damping = 1``) pushes
the coupled w tendency against the vertical motion where the vertical Courant
number exceeds an onset::

    vert_cfl = |w * rdz * dt|
    if vert_cfl > w_damp_on:  tend -= sign(1, w) * w_alpha * (vert_cfl - w_crit_cfl) * mass

WRF forms ``vert_cfl`` from the eta mass flux and ``rdnw``; the MPAS height
coordinate has the physical ``w`` and the layer thickness, so ``vert_cfl`` is
``|w| dt / dz`` with ``dz`` the thickness of the layer above the interface
(``rdzw``).  ``mass`` is WRF's ``c1f*mut + c2f`` (the column mass that couples
its tendency); MPAS couples its ``w`` tendency by ``rho_zz``, interpolated to the
interface with ``fzm``/``fzp`` as the dynamics does.  ``w_alpha`` = 0.3,
``w_beta`` = 1 (the onset, WRF's value when zadvect_implicit is off) and the
operation order are WRF's (woof/core/kernels/openbc.cu ``w_damp``, the kernel
the WRF path runs), each operation a single rounding.  Interior w levels only.
Off by default: nothing here runs unless ``--w-damping`` is given.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: WRF share/module_model_constants.F:88-89.
W_ALPHA = 0.3
W_BETA = 1.0

_CUDA_SOURCE = r'''
// Each operation is one explicit round-to-nearest intrinsic, as WRF's own w_damp
// kernel spells them (woof/core/kernels/openbc.cu): an FMA contraction here would
// move the Courant number across the onset and turn the limiter on or off.
extern "C" __global__ void w_damping_f32(
    const int nlev, const int ncells, const float dt, const float w_damp_on,
    const float w_crit_cfl, const float w_alpha, const float *w,
    const float *rho, const float *fzm, const float *fzp, const float *rdzw,
    float *tend)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= (nlev - 1) * ncells) return;
    const int k = index / ncells + 1;
    const int cell = index - (k - 1) * ncells;
    const int at = k * ncells + cell;
    const float vert_cfl = fabsf(__fmul_rn(__fmul_rn(w[at], rdzw[k]), dt));
    if (vert_cfl > w_damp_on) {
        const float mass = __fadd_rn(__fmul_rn(fzm[k], rho[k * ncells + cell]),
                                     __fmul_rn(fzp[k], rho[(k - 1) * ncells + cell]));
        float damp = __fmul_rn(copysignf(1.0f, w[at]), w_alpha);
        damp = __fmul_rn(damp, __fsub_rn(vert_cfl, w_crit_cfl));
        tend[at] = __fsub_rn(tend[at], __fmul_rn(damp, mass));
    }
}
'''


@dataclass(frozen=True)
class WDamping:
    """The run's w_damping switch: WRF's ``w_crit_cfl`` and the onset."""

    w_crit_cfl: float = 1.0
    w_damp_on: float = W_BETA

    def __post_init__(self):
        for name in ("w_crit_cfl", "w_damp_on"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")


def apply_w_damping(tend_rw, w, rho, vertical, *, dt, damping: WDamping, cache):
    """Add the limiter to ``tend_rw`` (``[nlev + 1, nCells]``) in place."""
    from .cuda_backend import require_resident_array

    tend_rw = require_resident_array("w tendency", tend_rw, dtype=np.float32)
    nlev1, ncells = tend_rw.shape
    nlev = nlev1 - 1
    for name, field, rows in (("w", w, nlev1), ("rho", rho, nlev)):
        field = require_resident_array(f"w damping {name}", field, dtype=np.float32)
        if tuple(field.shape) != (rows, ncells):
            raise ValueError(f"w damping {name} has shape {tuple(field.shape)}, "
                             f"not ({rows}, {ncells})")
    kernel = cache.raw_kernel("w_damping_f32", _CUDA_SOURCE,
                              module_key="hexcore.cuda_w_damping")
    kernel((((nlev - 1) * ncells + 127) // 128,), (128,),
           (np.int32(nlev), np.int32(ncells), np.float32(dt),
            np.float32(damping.w_damp_on), np.float32(damping.w_crit_cfl),
            np.float32(W_ALPHA), w, rho, vertical.fzm, vertical.fzp,
            vertical.rdzw, tend_rw))
    return tend_rw
