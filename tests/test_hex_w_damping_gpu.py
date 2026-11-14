"""The hex w_damping term IS WRF's w_damp on the same columns.

``woof.hex.cuda_w_damping`` writes WRF's limiter (module_big_step_utilities_em.F
``w_damp``) into MPAS's w tendency.  WRF's kernel (woof/core/kernels/openbc.cu
``w_damp``, the one the WRF path runs) is replayed on the same columns with a
unit column mass (``c1f = 0``, ``c2f = 1``, so ``ww/m = w`` and the coupled term
is the bare one); hex's term must equal it times the interface density,
bit for bit, on every level of every column.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

NLEV, NCELLS, DT = 12, 64, 30.0


def _columns(seed=0):
    rng = np.random.default_rng(seed)
    dz = rng.uniform(40.0, 900.0, NLEV).astype(np.float32)
    rdzw = (np.float32(1.0) / dz).astype(np.float32)
    w = rng.normal(0.0, 12.0, (NLEV + 1, NCELLS)).astype(np.float32)
    w[0] = 0.0
    w[-1] = 0.0
    rho = rng.uniform(0.2, 1.2, (NLEV, NCELLS)).astype(np.float32)
    fzm = rng.uniform(0.3, 0.7, NLEV).astype(np.float32)
    fzp = (np.float32(1.0) - fzm).astype(np.float32)
    tend = rng.normal(0.0, 1.0, (NLEV + 1, NCELLS)).astype(np.float32)
    return rdzw, w, rho, fzm, fzp, tend


def _hex(tend, w, rho, fzm, fzp, rdzw, crit, onset=1.0):
    from woof.hex.cuda_backend import KernelCache
    from woof.hex.cuda_w_damping import WDamping, apply_w_damping

    out = cp.asarray(tend).copy()
    vertical = SimpleNamespace(fzm=cp.asarray(fzm), fzp=cp.asarray(fzp),
                               rdzw=cp.asarray(rdzw))
    apply_w_damping(out, cp.asarray(w), cp.asarray(rho), vertical, dt=DT,
                    damping=WDamping(w_crit_cfl=crit, w_damp_on=onset),
                    cache=KernelCache())
    return cp.asnumpy(out)


def _wrf_unit_mass(w, rdzw, crit, onset):
    from woof.core.kernels import get_kernel

    plane = (1, NCELLS)
    rw_t = cp.zeros((NLEV + 1,) + plane, cp.float32)
    ww = cp.asarray(w).reshape((NLEV + 1,) + plane)
    zeros = cp.zeros(plane, cp.float32)
    c1f = cp.zeros(NLEV + 1, cp.float32)
    c2f = cp.ones(NLEV + 1, cp.float32)
    kernel = get_kernel("openbc", "w_damp")
    blocks = ((NLEV - 1) * NCELLS + 127) // 128
    kernel((blocks,), (128,), (rw_t, ww, ww, zeros, zeros, c1f, c2f,
                               cp.asarray(rdzw), np.float32(DT), np.float32(onset),
                               np.float32(crit), np.int32(NLEV), np.int32(1), np.int32(NCELLS)))
    return cp.asnumpy(rw_t).reshape(NLEV + 1, NCELLS)


@pytest.mark.parametrize("crit,onset", [(1.0, 1.0), (1.2, 1.0), (0.8, 0.5)])
def test_hex_w_damping_is_wrf_w_damp_on_the_same_columns(crit, onset):
    rdzw, w, rho, fzm, fzp, tend = _columns()
    got = _hex(tend, w, rho, fzm, fzp, rdzw, crit, onset)
    wrf = _wrf_unit_mass(w, rdzw, crit, onset)
    mass = np.zeros((NLEV + 1, NCELLS), np.float32)
    mass[1:NLEV] = (fzm[1:NLEV, None] * rho[1:NLEV]) + (fzp[1:NLEV, None] * rho[0:NLEV - 1])
    want = (tend + (wrf * mass)).astype(np.float32)
    acted = wrf != 0
    assert acted.any() and (~acted[1:NLEV]).any()
    np.testing.assert_array_equal(got.view(np.uint32)[acted], want.view(np.uint32)[acted])
    # Where the limiter does not act, and on the end levels, the tendency is untouched.
    np.testing.assert_array_equal(got[~acted], tend[~acted])
    np.testing.assert_array_equal(got[[0, NLEV]], tend[[0, NLEV]])


def test_the_limiter_pushes_against_the_motion_above_the_critical_courant_number():
    rdzw, w, rho, fzm, fzp, tend = _columns(3)
    zero = np.zeros_like(tend)
    got = _hex(zero, w, rho, fzm, fzp, rdzw, crit=0.5, onset=1.0)
    cfl = np.abs(w[1:NLEV] * rdzw[1:NLEV, None] * np.float32(DT))
    live = cfl > 1.0
    assert live.any()
    assert np.all(np.sign(got[1:NLEV][live]) == -np.sign(w[1:NLEV][live]))
    assert np.all(got[1:NLEV][~live] == 0)
