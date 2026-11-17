"""The positive-definite advection launches' 64 x 4 CUDA block
(woof/core/moist.py _PD_BLOCK_CUDA) writes the words the 128 x 1 row block
writes.  Every thread computes its own point; pd_renorm_apply's warp
exchange of a neighbour's cell scale sees a warp of 32 consecutive i of one
row under either block, with its warp edges at the same i, so the same cells
exchange and the same cells evaluate their own.  Both launchers, periodic and
specified/open, vertical orders 3 and 5, horizontal orders 5 and 3, the
zero-first fold, row lengths that leave partial blocks and partial warps, in
each numerics mode.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

import _numerics_child
from conftest import requires_gpu


def _outputs(*, shape, open_bc, vorder, horder, zero_first):
    import cupy as cp

    from woof.core import moist

    nz, ny, nx = shape
    rng = np.random.default_rng(7)

    def field(*s, lo=0.0, hi=1.0):
        return cp.asarray(rng.uniform(lo, hi, s).astype(np.float32))

    q = field(nz, ny, nx, hi=1e-3)
    q0 = field(nz, ny, nx, hi=1e-3)
    # empty and near-empty cells, so the limiter scales below one
    q0[:, ::3, ::4] = 0.0
    q[:, 1::5, ::3] = 0.0
    ru = field(nz, ny, nx + 1, lo=-4e4, hi=4e4)
    rv = field(nz, ny + 1, nx, lo=-4e4, hi=4e4)
    rw = field(nz + 1, ny, nx, lo=-2.0, hi=2.0)
    rw[0] = 0.0
    rw[-1] = 0.0
    mut = field(ny, nx, lo=9.0e4, hi=1.0e5)
    msft = field(ny, nx, lo=0.95, hi=1.05)
    fnm = rng.uniform(0.4, 0.6, nz).astype(np.float32)
    coord = types.SimpleNamespace(
        c1h=rng.uniform(0.9, 1.0, nz).astype(np.float32),
        c2h=rng.uniform(0.0, 50.0, nz).astype(np.float32),
        rdnw=rng.uniform(-60, -40, nz).astype(np.float32),
        fnm=fnm, fnp=(1 - fnm).astype(np.float32),
        mub2d=field(ny, nx, lo=8.0e4, hi=9.0e4), mup0=field(ny, nx, lo=-500.0, hi=500.0))
    bufs = [cp.full(s, np.nan, dtype=cp.float32) for s in
            ((nz, ny, nx + 1),) * 2 + ((nz, ny + 1, nx),) * 2 + ((nz + 1, ny, nx),) * 2]
    tend = cp.asarray(rng.normal(0.0, 1e-6, (nz, ny, nx)).astype(np.float32))
    moist.launch_pd_fluxes(q, q0, ru, rv, rw, mut, coord, 3000.0, 3000.0, 12.0, *bufs,
                           msft=msft, open_x=open_bc, open_y=open_bc, vorder=vorder,
                           horder=horder)
    moist.launch_pd_renorm_apply(q0, mut, *bufs, tend=tend, coord=coord, dx=3000.0,
                                 dy=3000.0, dt=12.0, msft=msft, open_x=open_bc,
                                 open_y=open_bc, zero_first=zero_first)
    cp.cuda.runtime.deviceSynchronize()
    names = ("fxl", "fxc", "fyl", "fyc", "fzl", "fzc", "tend")
    return {name: cp.asnumpy(a).view(np.uint32) for name, a in zip(names, bufs + [tend])}


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(12, 13, 150), (9, 10, 37), (7, 21, 8), (5, 9, 64)])
@pytest.mark.parametrize("open_bc", [True, False])
@pytest.mark.parametrize("vorder,horder,zero_first", [(3, 5, False), (3, 5, True),
                                                      (5, 5, False), (3, 3, False)])
def test_pd_block_writes_the_row_block_words(monkeypatch, shape, open_bc, vorder, horder,
                                             zero_first):
    from woof.core import moist

    if open_bc and min(shape[1:]) < moist.FIFTH_ORDER_STENCIL_AXIS:
        pytest.skip("an open axis needs 7 cells")
    kw = dict(shape=shape, open_bc=open_bc, vorder=vorder, horder=horder,
              zero_first=zero_first)
    # The row block against the CUDA block, on whichever backend runs
    # (Metal launches the row block in production).
    monkeypatch.setattr(moist, "_pd_block", lambda: moist._PD_BLOCK_ROW)
    reference = _outputs(**kw)
    monkeypatch.setattr(moist, "_pd_block", lambda: moist._PD_BLOCK_CUDA)
    candidate = _outputs(**kw)
    for name in reference:
        differ = int(np.count_nonzero(reference[name] != candidate[name]))
        assert differ == 0, f"{name}: {differ} words differ"
    # not vacuous: the limiter ran and moved the tendency
    assert np.count_nonzero(candidate["tend"] != 0)


def test_pd_block_override_keeps_whole_warps_on_one_row(monkeypatch):
    from woof.core import moist

    monkeypatch.setenv("WOOF_PD_BLOCK", "32x8")
    assert moist._pd_block() == (32, 8, 1)
    monkeypatch.setenv("WOOF_PD_BLOCK", "16x8")
    with pytest.raises(ValueError, match="multiple of 32"):
        moist._pd_block()
    assert moist._PD_BLOCK_CUDA[0] % 32 == 0 and moist._PD_BLOCK_ROW[0] % 32 == 0


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize(*_numerics_child.MODES)
def test_pd_block_words_hold_under_each_numerics(numerics, exact):
    _numerics_child.run(__file__, ["test_pd_block_writes_the_row_block_words"],
                        numerics, exact)
