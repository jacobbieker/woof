"""wrf_smag_flux_s_multi / wrf_smag_hd_s_multi (kernels/smag2d.cu): several
scalar rows per launch, word for word the per-row wrf_smag_flux_s and
wrf_smag_hd_s launches (dycore.launch_wrf_smag2d_hd, stagger "").

The face metrics (density and K averages, slope, face rdz, map factor) and
the cell metrics (map, slope averages, rdzw) are formed once per thread and
every row then evaluates the per-row statements; the words must match over
a theta row (full_theta, 1-D and 3-D base theta), moist and dry density,
periodic and bounded axes, 3-D base geopotential, and both address widths.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

NROWS = 5


def _grid(rng, nz, ny, nx, *, phb3d, moist, boundary):
    import cupy as cp

    f = lambda lo, hi, *shape: cp.asarray(rng.uniform(lo, hi, shape).astype(np.float32))
    z = (np.cumsum(np.full(nz + 1, 300.0, np.float32)) - 300.0) * np.float32(9.81)
    phb = (np.broadcast_to(z[:, None, None], (nz + 1, ny, nx))
           + rng.normal(0, 200.0, (nz + 1, ny, nx))).astype(np.float32) if phb3d else z
    eta = np.linspace(1.0, 0.0, nz + 1, dtype=np.float32)
    dnw = np.diff(eta).astype(np.float32)
    dn = np.concatenate([[dnw[0]], 0.5 * (dnw[1:] + dnw[:-1])]).astype(np.float32)
    fnm = np.full(nz, 0.5, np.float32); fnp = np.full(nz, 0.5, np.float32)
    args = dict(
        u=f(-10, 10, nz, ny, nx + 1), v=f(-10, 10, nz, ny + 1, nx),
        w=f(-1, 1, nz + 1, ny, nx), php=f(-50, 50, nz + 1, ny, nx),
        phb=cp.asarray(phb), alt=f(0.8, 1.2, nz, ny, nx), qv=f(0, 0.02, nz, ny, nx),
        msft=f(0.95, 1.05, ny, nx), msfu=f(0.95, 1.05, ny, nx + 1),
        msfv=f(0.95, 1.05, ny + 1, nx), fnm=cp.asarray(fnm), fnp=cp.asarray(fnp),
        dn=cp.asarray(dn), dnw=cp.asarray(dnw))
    scalars = (np.float32(1 / 3000.0), np.float32(1 / 3000.0), np.float32(3000.0),
               np.float32(3000.0), np.float32(1.5), np.float32(-0.5), np.float32(0.0),
               np.int32(moist))
    common = [args[k] for k in ("u", "v", "w", "php", "phb", "alt", "qv", "msft",
                                 "msfu", "msfv", "fnm", "fnp", "dn", "dnw")] + list(scalars)
    tail = [np.int32(nz), np.int32(ny), np.int32(nx), np.int32(phb3d),
            np.int32(boundary), np.int32(boundary)]
    return common, tail


def _kernels(nz, ny, nx, index32):
    from woof.core.kernels import get_kernel, get_kernel_int_defines
    names = ("wrf_smag_flux_s", "wrf_smag_hd_s", "wrf_smag_flux_s_multi", "wrf_smag_hd_s_multi")
    if index32:
        return [get_kernel_int_defines("smag2d", n, (("GPUWM_SMAG_INDEX32", 1),)) for n in names]
    return [get_kernel("smag2d", n) for n in names]


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("phb3d,moist,boundary,thb3d,index32", [
    (0, 1, 1, 0, True), (1, 1, 0, 1, True), (1, 0, 1, 0, False), (0, 1, 1, 1, False),
])
def test_multi_row_launches_match_the_per_row_words(phb3d, moist, boundary, thb3d, index32):
    import cupy as cp

    nz, ny, nx = 11, 13, 17
    tpb = 64
    rng = np.random.default_rng(29 + phb3d + 2 * moist + 4 * boundary)
    common, tail = _grid(rng, nz, ny, nx, phb3d=phb3d, moist=moist, boundary=boundary)
    u = lambda lo, hi, *shape: cp.asarray(rng.uniform(lo, hi, shape).astype(np.float32))
    kh = u(0, 500.0, nz, ny, nx)
    thb = (u(280, 320, nz, ny, nx) if thb3d
           else cp.asarray(np.linspace(290.0, 320.0, nz, dtype=np.float32)))
    fields = [u(-5, 5, nz, ny, nx) for _ in range(NROWS)]
    fields[0] = u(-3, 3, nz, ny, nx)             # the theta row (thp)
    full = [1, 0, 0, 0, 0]
    tend0 = [u(-1e-3, 1e-3, nz, ny, nx) for _ in range(NROWS)]
    flux_s, hd_s, flux_multi, hd_multi = _kernels(nz, ny, nx, index32)
    flux_grid = ((nx + 1 + tpb - 1) // tpb, ny + 1, nz)
    grid = ((nx + tpb - 1) // tpb, ny, nz)

    # per row
    ref_fx, ref_fy, ref_tend = [], [], []
    for r in range(NROWS):
        fx = cp.empty((nz, ny, nx + 1), cp.float32); fy = cp.empty((nz, ny + 1, nx), cp.float32)
        flux_s(flux_grid, (tpb, 1, 1), tuple(common + [fields[r], kh, thb, np.int32(full[r]),
                                                       np.int32(thb3d), fx, fy] + tail))
        tend = tend0[r].copy()
        hd_s(grid, (tpb, 1, 1), tuple(common + [fx, fy, tend] + tail))
        ref_fx.append(cp.asnumpy(fx).view(np.uint32)); ref_fy.append(cp.asnumpy(fy).view(np.uint32))
        ref_tend.append(cp.asnumpy(tend).view(np.uint32))

    # multi: two chunks (3 + 2 rows), by-value tables
    got_fx, got_fy, got_tend = [None] * NROWS, [None] * NROWS, [None] * NROWS
    for start in (0, 3):
        rows = list(range(start, min(start + 3, NROWS)))
        pairs = [(cp.empty((nz, ny, nx + 1), cp.float32), cp.empty((nz, ny + 1, nx), cp.float32))
                 for _ in rows]
        tends = [tend0[r].copy() for r in rows]
        table = np.zeros((8, 4), dtype=np.uint64)
        for k, r in enumerate(rows):
            table[k] = (fields[r].data.ptr, pairs[k][0].data.ptr, pairs[k][1].data.ptr, full[r])
        blob = table.reshape(-1).view(np.dtype(('V', table.nbytes)))
        flux_multi(flux_grid, (tpb, 1, 1), tuple(common + [kh, thb, np.int32(thb3d), blob,
                                                           np.int32(len(rows))] + tail))
        table = np.zeros((8, 4), dtype=np.uint64)
        for k, r in enumerate(rows):
            table[k] = (pairs[k][0].data.ptr, pairs[k][1].data.ptr, tends[k].data.ptr, 0)
        blob = table.reshape(-1).view(np.dtype(('V', table.nbytes)))
        hd_multi(grid, (tpb, 1, 1), tuple(common + [blob, np.int32(len(rows))] + tail))
        for k, r in enumerate(rows):
            got_fx[r] = cp.asnumpy(pairs[k][0]).view(np.uint32)
            got_fy[r] = cp.asnumpy(pairs[k][1]).view(np.uint32)
            got_tend[r] = cp.asnumpy(tends[k]).view(np.uint32)
    for r in range(NROWS):
        assert np.array_equal(ref_fx[r], got_fx[r]), ("fx", r)
        assert np.array_equal(ref_fy[r], got_fy[r]), ("fy", r)
        assert np.array_equal(ref_tend[r], got_tend[r]), ("tend", r)
        assert not np.array_equal(ref_tend[r], cp.asnumpy(tend0[r]).view(np.uint32)), r


def test_row_eligibility_and_chunking():
    """Host arrays, a second K or a shared tendency buffer take the per-row
    path; the chunk size is bounded by the kernel's table."""
    import woof.core.dycore as dy

    assert dy.SMAG_MULTI_ROWS <= 8
    a = np.zeros((2, 3, 4), np.float32)
    rows = [(a, a, a, False), (a, a, a.copy(), False)]
    assert dy._smag_multi_rows(rows, 2, 3, 4) is False      # host arrays
    assert dy._smag_multi_rows([], 2, 3, 4) is False
