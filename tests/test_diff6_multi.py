"""diff6_multi: several mass-point rows in one launch, word for word the
per-field diff6 launch (kernels/diff6.cu, dycore.launch_diff6_multi).

The multi kernel forms the field-independent face masses, slope tapers and
map factor once per point and then evaluates diff6's variant-0 statements
per row, so every word it writes must equal the one launch_diff6 writes.
Checked over the monotonic switch, the slope taper, forced and periodic
loop bounds, a held incoming tendency, and more rows than one table holds.
"""
import numpy as np
import pytest

from conftest import requires_gpu


def _fields(rng, nrows, nlev, ny, nx):
    f = [rng.standard_normal((nlev, ny, nx)).astype(np.float32)
         for _ in range(nrows)]
    held = [rng.standard_normal((nlev, ny, nx)).astype(np.float32) * 1e-3
            for _ in range(nrows)]
    mut = (1.0e5 + 500.0 * rng.standard_normal((ny, nx))).astype(np.float32)
    c1 = np.linspace(1.0, 0.0, nlev).astype(np.float32)
    c2 = np.linspace(0.0, 5000.0, nlev).astype(np.float32)
    phb = np.cumsum(
        200.0 + 50.0 * rng.standard_normal((nlev, ny, nx)), axis=0
    ).astype(np.float32) * np.float32(9.81)
    msfu = (1.0 + 0.01 * rng.standard_normal((ny, nx + 1))).astype(np.float32)
    msfv = (1.0 + 0.01 * rng.standard_normal((ny + 1, nx))).astype(np.float32)
    msft = (1.0 + 0.01 * rng.standard_normal((ny, nx))).astype(np.float32)
    return f, held, mut, c1, c2, phb, msfu, msfv, msft


@requires_gpu
@pytest.mark.parametrize("opt,slopeopt,bnd", [
    (1, 0, False), (2, 0, False), (2, 1, True), (1, 1, True), (2, 0, True),
])
def test_multi_launch_matches_per_field_words(opt, slopeopt, bnd):
    import cupy as cp
    from woof.core.dycore import launch_diff6, launch_diff6_multi

    rng = np.random.default_rng(7)
    nrows, nlev, ny, nx = 5, 6, 14, 17
    f, held, mut, c1, c2, phb, msfu, msfv, msft = _fields(
        rng, nrows, nlev, ny, nx)
    factors = [0.12, 0.25, 0.25, 0.4, 0.12]
    dts = [15.0, 5.0, 5.0, 5.0, 7.5]
    dev = lambda a: cp.asarray(a, dtype=cp.float32)
    mut_d, c1_d, c2_d = dev(mut), dev(c1), dev(c2)
    phb_d, msfu_d, msfv_d, msft_d = dev(phb), dev(msfu), dev(msfv), dev(msft)
    maps = dict(phb=phb_d, msfu=msfu_d, msfv=msfv_d, msft=msft_d,
                slopeopt=slopeopt, thresh=0.1, dx=3000.0, dy=3000.0,
                bnd_x=bnd, bnd_y=bnd)
    single = []
    for row in range(nrows):
        tend = dev(held[row])
        launch_diff6(dev(f[row]), tend, mut_d, c1_d, c2_d, factors[row],
                     dts[row], opt, stagger="", **maps)
        single.append(cp.asnumpy(tend))
    multi = [dev(h) for h in held]
    rows = [(dev(f[row]), multi[row], factors[row], dts[row])
            for row in range(nrows)]
    launch_diff6_multi(rows, mut_d, c1_d, c2_d, opt, **maps)
    for row in range(nrows):
        got = cp.asnumpy(multi[row])
        assert got.dtype == np.float32
        assert np.array_equal(got.view(np.uint32),
                              single[row].view(np.uint32)), row
        # The filter changed something: the check is not vacuous.
        assert not np.array_equal(got, held[row])


@requires_gpu
def test_more_rows_than_one_table_take_further_launches():
    import cupy as cp
    from woof.core.dycore import (DIFF6_MULTI_MAX_ROWS, launch_diff6,
                                  launch_diff6_multi)

    rng = np.random.default_rng(11)
    nrows, nlev, ny, nx = DIFF6_MULTI_MAX_ROWS + 3, 3, 9, 11
    f, held, mut, c1, c2, phb, msfu, msfv, msft = _fields(
        rng, nrows, nlev, ny, nx)
    dev = lambda a: cp.asarray(a, dtype=cp.float32)
    mut_d, c1_d, c2_d = dev(mut), dev(c1), dev(c2)
    maps = dict(phb=None, msfu=None, msfv=None, msft=dev(msft),
                slopeopt=0, bnd_x=True, bnd_y=False)
    single = []
    for row in range(nrows):
        tend = dev(held[row])
        launch_diff6(dev(f[row]), tend, mut_d, c1_d, c2_d, 0.12, 20.0, 2,
                     stagger="", **maps)
        single.append(cp.asnumpy(tend))
    multi = [dev(h) for h in held]
    rows = [(dev(f[row]), multi[row], 0.12, 20.0) for row in range(nrows)]
    launch_diff6_multi(rows, mut_d, c1_d, c2_d, 2, **maps)
    for row in range(nrows):
        assert np.array_equal(cp.asnumpy(multi[row]).view(np.uint32),
                              single[row].view(np.uint32)), row


@requires_gpu
def test_rows_sharing_a_target_or_shapes_fall_back(monkeypatch):
    """Two rows into one buffer, or a strided view, keep the per-field path."""
    import cupy as cp
    from woof.core import dycore
    from woof.core.dycore import _diff6_multi_rows

    monkeypatch.setattr(dycore, "DIFF6_MULTI_ENABLED", True)
    a = cp.zeros((2, 4, 5), dtype=cp.float32)
    b = cp.zeros((2, 4, 5), dtype=cp.float32)
    t = cp.zeros((2, 4, 5), dtype=cp.float32)
    mut = cp.ones((4, 5), dtype=cp.float32)
    c = cp.ones((2,), dtype=cp.float32)
    planes = (None, None, None, None)
    assert _diff6_multi_rows([(a, t, 0.1, 1.0)], c, c, mut, planes)
    assert not _diff6_multi_rows([(a, t, 0.1, 1.0), (b, t, 0.1, 1.0)],
                                 c, c, mut, planes)
    assert not _diff6_multi_rows([(a[:, ::2], t[:, ::2], 0.1, 1.0)],
                                 c, c, mut, planes)
    assert not _diff6_multi_rows([], c, c, mut, planes)
    monkeypatch.setattr(dycore, "DIFF6_MULTI_ENABLED", False)
    assert not _diff6_multi_rows([(a, t, 0.1, 1.0)], c, c, mut, planes)
