"""The flux-divergence launches' 32 x 4 CUDA block (woof/core/advection.py
_FLUX_BLOCK_CUDA) writes the words the 128 x 1 row block writes: every
thread computes its own point, so only the thread a point lands on moves.
Each launcher, on the specified/open and periodic paths, at row lengths
that leave a partial block, in each numerics mode.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

import _numerics_child
from conftest import requires_gpu


def _tendencies(*, shape, open_bc, horder, weno, zero_first):
    import cupy as cp

    from woof.core import advection as adv

    nz, ny, nx = shape
    rng = np.random.default_rng(3)

    def field(*s, scale=1.0, offset=0.0):
        return cp.asarray((offset + scale * rng.standard_normal(s)).astype(np.float32))

    q = field(nz, ny, nx, offset=300.0)
    u = field(nz, ny, nx + 1, scale=10.0)
    v = field(nz, ny + 1, nx, scale=10.0)
    w = field(nz + 1, ny, nx)
    ru = field(nz, ny, nx + 1, scale=1e4)
    rv = field(nz, ny + 1, nx, scale=1e4)
    rw = field(nz + 1, ny, nx, scale=1e3)
    # WRF-like stretched weights: fnm != fnp, so no product hides behind 0.5.
    fnm = rng.uniform(0.4, 0.6, nz).astype(np.float32)
    coord = types.SimpleNamespace(
        rdnw=rng.uniform(-60, -40, nz).astype(np.float32),
        rdn=rng.uniform(-60, -40, nz).astype(np.float32),
        fnm=fnm, fnp=(1 - fnm).astype(np.float32))
    msft = cp.asarray(rng.uniform(0.95, 1.05, (ny, nx)).astype(np.float32))
    msfu = cp.asarray(rng.uniform(0.95, 1.05, (ny, nx + 1)).astype(np.float32))
    msfv = cp.asarray(rng.uniform(0.95, 1.05, (ny + 1, nx)).astype(np.float32))
    # Seeded (not zero) tendencies: the accumulating launches add onto them.
    tq, tu, tv, tw = (field(*a.shape, scale=1e-3) for a in (q, u, v, w))
    kw = dict(open_x=open_bc, open_y=open_bc, spec=open_bc, vorder=3)
    adv.launch_flux_div_scalar(q, ru, rv, rw, tq, coord, 333.0, 333.0, msf=msft,
                               horder=horder, zero_first=zero_first, **kw)
    adv.launch_flux_div_u(u, ru, rv, rw, tu, coord, 333.0, 333.0, msf=msfu,
                          horder=horder, weno=weno, **kw)
    adv.launch_flux_div_v(v, ru, rv, rw, tv, coord, 333.0, 333.0, msf=msfv,
                          horder=horder, weno=weno, **kw)
    adv.launch_flux_div_w(w, ru, rv, rw, tw, coord, 333.0, 333.0, msf=msft,
                          horder=horder, weno=weno, **kw)
    cp.cuda.runtime.deviceSynchronize()
    return {name: cp.asnumpy(t).view(np.uint32)
            for name, t in (("q", tq), ("u", tu), ("v", tv), ("w", tw))}


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(12, 13, 150), (9, 10, 37), (7, 21, 8)])
@pytest.mark.parametrize("open_bc", [True, False])
@pytest.mark.parametrize("horder,weno,zero_first", [(5, False, False), (5, False, True),
                                                    (3, False, False), (5, True, False)])
def test_flux_block_writes_the_row_block_words(monkeypatch, shape, open_bc, horder,
                                               weno, zero_first):
    from woof.core import advection as adv

    kw = dict(shape=shape, open_bc=open_bc, horder=horder, weno=weno,
              zero_first=zero_first)
    # The row block against the CUDA block, on whichever backend runs
    # (Metal launches the row block in production).
    monkeypatch.setattr(adv, "_flux_block", lambda: adv._FLUX_BLOCK_ROW)
    reference = _tendencies(**kw)
    monkeypatch.setattr(adv, "_flux_block", lambda: adv._FLUX_BLOCK_CUDA)
    candidate = _tendencies(**kw)
    for name in reference:
        differ = int(np.count_nonzero(reference[name] != candidate[name]))
        assert differ == 0, f"{name}: {differ} words differ"


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize(*_numerics_child.MODES)
def test_flux_block_words_hold_under_each_numerics(numerics, exact):
    _numerics_child.run(__file__, ["test_flux_block_writes_the_row_block_words"],
                        numerics, exact)
