"""Kernels that reach buffers through flat address tables launch on Metal
without a synchronize (woof/backend/metal/loader.py, WOOF_BACKEND=mlx).

The Smagorinsky row kernels (wrf_smag_flux_s_multi / wrf_smag_hd_s_multi)
take a by-value table of row addresses BESIDE their own const grid
pointers, and mp_ring_copy reads its section addresses from a DEVICE table.
Both used to synchronize before and after every launch (about 75 ms of a
233 ms Edinburgh 3 km step on an M3 Max, 2026-11-16).  The loader now
resolves the table words to the storages they name -- a device table's
words read in place from unified memory, once per value the table takes --
and binds those storages for MLX to order.  These tests hold the words
(against the host model or the per-row kernels) while in-place writers and
MLX readers crowd each launch, and count the synchronizes.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")

_BUMP = r'''extern "C" __global__ void bump(float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; for (int k = 0; k < 32; ++k) v = v * 0.5f + s; x[i] = v; } }'''


def _host_bump(v, s):
    for _ in range(32):
        v = (v * np.float32(0.5) + np.float32(s)).astype(np.float32)
    return v


def _mlx_or_skip():
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")


@pytest.fixture
def synchronizes(monkeypatch):
    """The mx.synchronize / mx.eval calls made while a test runs."""
    import mlx.core as mx
    calls = []
    for name in ("synchronize", "eval"):
        fn = getattr(mx, name)

        def counted(*a, _fn=fn, _name=name, **k):
            calls.append(_name)
            return _fn(*a, **k)
        monkeypatch.setattr(mx, name, counted)
    return calls


def test_a_device_address_table_launch_is_ordered_without_a_synchronize(synchronizes):
    """mp_ring_copy gathers ring sections of ``arr`` into ``buf`` through a
    device table of their addresses: each gather reads what the bump before
    it wrote, an MLX reader taken before it keeps buf's old words, and the
    bump after it does not overwrite its source under it."""
    _mlx_or_skip()
    import cupy as cp
    from woof.core.kernels import get_kernel

    ring = get_kernel("microphysics_validation", "mp_ring_copy")
    bump = cp.RawKernel(_BUMP, "bump")
    nz, ny, nx = 6, 40, 50
    n = nz * ny * nx
    rng = np.random.default_rng(5)
    arr_h = rng.standard_normal(n).astype(np.float32)
    arr = cp.asarray(arr_h)
    # two sections: west columns 0..2 and north rows ny-3..ny-1
    sections = [(0, ny, 0, 3), (ny - 3, 3, 0, nx)]
    bufs = [cp.zeros(nz * nj * ni, cp.float32) for _j0, nj, _i0, ni in sections]
    rows = [(arr.data.ptr, b.data.ptr, nz, j0, nj, i0, ni, nx, ny * nx)
            for b, (j0, nj, i0, ni) in zip(bufs, sections)]
    table = cp.asarray(np.asarray(rows, dtype=np.int64).reshape(-1))

    def host_gather(a):
        cube = a.reshape(nz, ny, nx)
        return [cube[:, j0:j0 + nj, i0:i0 + ni].reshape(-1).copy() for j0, nj, i0, ni in sections]

    grid = ((n + 255) // 256,)
    taken, want = [], []
    buf_h = [np.zeros(nz * nj * ni, np.float32) for _j0, nj, _i0, ni in sections]
    ring((1, 2), (256,), (table, np.int32(2), np.int32(0)))      # the table's first read
    buf_h = host_gather(arr_h)
    del synchronizes[:]
    for step in range(8):
        bump(grid, (256,), (arr, np.float32(step), np.int32(n)))
        arr_h = _host_bump(arr_h, step)
        taken.append([b * cp.float32(1.0) for b in bufs])
        want.append([b.copy() for b in buf_h])
        ring((1, 2), (256,), (table, np.int32(2), np.int32(0)))
        buf_h = host_gather(arr_h)
        taken.append([b + cp.float32(0.0) for b in bufs])
        want.append([b.copy() for b in buf_h])
        bump(grid, (256,), (arr, np.float32(-step), np.int32(n)))
        arr_h = _host_bump(arr_h, -step)
    assert synchronizes == [], synchronizes
    for got, ref in zip(taken, want):
        for g, r in zip(got, ref):
            assert np.array_equal(g.get().view(np.uint32), r.view(np.uint32))
    assert np.array_equal(arr.get().view(np.uint32), arr_h.view(np.uint32))


def test_a_new_device_table_value_is_read_again():
    """A table rewritten in place names other sections: the words cached
    for its old value must not be used for the new one."""
    _mlx_or_skip()
    import cupy as cp
    from woof.core.kernels import get_kernel

    ring = get_kernel("microphysics_validation", "mp_ring_copy")
    n = 64
    a, b = cp.asarray(np.arange(n, dtype=np.float32)), cp.asarray(np.arange(n, dtype=np.float32) + 100)
    out_a, out_b = cp.zeros(n, cp.float32), cp.zeros(n, cp.float32)
    row = lambda src, dst: (src.data.ptr, dst.data.ptr, 1, 0, 1, 0, n, n, n)
    table = cp.asarray(np.asarray([row(a, out_a)], dtype=np.int64).reshape(-1))
    ring((1, 1), (256,), (table, np.int32(1), np.int32(0)))
    table[...] = cp.asarray(np.asarray([row(b, out_b)], dtype=np.int64).reshape(-1))
    reader = out_b * cp.float32(1.0)              # taken before the second gather
    ring((1, 1), (256,), (table, np.int32(1), np.int32(0)))
    assert np.array_equal(out_a.get(), np.arange(n, dtype=np.float32))
    assert np.array_equal(reader.get(), np.zeros(n, np.float32))
    assert np.array_equal(out_b.get(), np.arange(n, dtype=np.float32) + 100)


def test_row_tables_beside_grid_pointers_launch_without_a_synchronize(synchronizes):
    """The Smagorinsky multi-row kernels, with each row's field written in
    place just before the launch pair and its tendency overwritten just
    after, give the per-row kernels' words and never synchronize."""
    _mlx_or_skip()
    import cupy as cp
    from test_smag_multi import _grid, _kernels

    nz, ny, nx, tpb, nrows = 9, 12, 14, 64, 3
    rng = np.random.default_rng(41)
    common, tail = _grid(rng, nz, ny, nx, phb3d=1, moist=1, boundary=1)
    u = lambda lo, hi, *shape: cp.asarray(rng.uniform(lo, hi, shape).astype(np.float32))
    kh = u(0, 500.0, nz, ny, nx)
    thb = cp.asarray(np.linspace(290.0, 320.0, nz, dtype=np.float32))
    fields = [u(-5, 5, nz, ny, nx) for _ in range(nrows)]
    tend0 = [u(-1e-3, 1e-3, nz, ny, nx) for _ in range(nrows)]
    flux_s, hd_s, flux_multi, hd_multi = _kernels(nz, ny, nx, True)
    bump = cp.RawKernel(_BUMP, "bump")
    size = nz * ny * nx
    flux_grid = ((nx + 1 + tpb - 1) // tpb, ny + 1, nz)
    grid = ((nx + tpb - 1) // tpb, ny, nz)

    ref = []
    for r in range(nrows):
        f = fields[r].copy()
        bump(((size + 255) // 256,), (256,), (f, np.float32(r), np.int32(size)))
        fx = cp.empty((nz, ny, nx + 1), cp.float32); fy = cp.empty((nz, ny + 1, nx), cp.float32)
        flux_s(flux_grid, (tpb, 1, 1), tuple(common + [f, kh, thb, np.int32(0), np.int32(0),
                                                       fx, fy] + tail))
        tend = tend0[r].copy()
        hd_s(grid, (tpb, 1, 1), tuple(common + [fx, fy, tend] + tail))
        ref.append(cp.asnumpy(tend).view(np.uint32))

    pairs = [(cp.empty((nz, ny, nx + 1), cp.float32), cp.empty((nz, ny + 1, nx), cp.float32))
             for _ in range(nrows)]
    tends = [t.copy() for t in tend0]
    flux_table = np.zeros((8, 4), dtype=np.uint64)
    hd_table = np.zeros((8, 4), dtype=np.uint64)
    for k in range(nrows):
        flux_table[k] = (fields[k].data.ptr, pairs[k][0].data.ptr, pairs[k][1].data.ptr, 0)
        hd_table[k] = (pairs[k][0].data.ptr, pairs[k][1].data.ptr, tends[k].data.ptr, 0)
    flux_blob = flux_table.reshape(-1).view(np.dtype(("V", flux_table.nbytes)))
    hd_blob = hd_table.reshape(-1).view(np.dtype(("V", hd_table.nbytes)))
    del synchronizes[:]
    for k in range(nrows):
        bump(((size + 255) // 256,), (256,), (fields[k], np.float32(k), np.int32(size)))
    before = [t * cp.float32(1.0) for t in tends]
    flux_multi(flux_grid, (tpb, 1, 1), tuple(common + [kh, thb, np.int32(0), flux_blob,
                                                       np.int32(nrows)] + tail))
    hd_multi(grid, (tpb, 1, 1), tuple(common + [hd_blob, np.int32(nrows)] + tail))
    after = [t + cp.float32(0.0) for t in tends]
    for k in range(nrows):
        bump(((size + 255) // 256,), (256,), (tends[k], np.float32(7), np.int32(size)))
    assert synchronizes == [], synchronizes
    for k in range(nrows):
        assert np.array_equal(before[k].get(), tend0[k].get()), k
        assert np.array_equal(after[k].get().view(np.uint32), ref[k]), k


_REWRITE = r'''extern "C" __global__ void rewrite_table(unsigned long long table,
                                              long long src, long long dst) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        long long* t = (long long*)table; t[0] = src; t[1] = dst; } }'''


def test_a_table_rewritten_through_its_address_is_read_again(monkeypatch):
    """A kernel that writes the table through a raw address leaves the
    table's storage value as it was: the words read for the old value must
    not be used to order the next gather (it would bind a and out_a, not
    b and out_b, as the gather's hazards)."""
    _mlx_or_skip()
    import cupy as cp
    from woof.backend.metal import loader
    from woof.core.kernels import get_kernel

    used = []
    words = loader._device_table_words
    monkeypatch.setattr(loader, "_device_table_words",
                        lambda view: used.append(words(view)) or used[-1])

    ring = get_kernel("microphysics_validation", "mp_ring_copy")
    rewrite = cp.RawKernel(_REWRITE, "rewrite_table")
    n = 64
    a, b = cp.asarray(np.arange(n, dtype=np.float32)), cp.asarray(np.arange(n, dtype=np.float32) + 100)
    out_a, out_b = cp.zeros(n, cp.float32), cp.zeros(n, cp.float32)
    row = lambda src, dst: (src.data.ptr, dst.data.ptr, 1, 0, 1, 0, n, n, n)
    table = cp.asarray(np.asarray([row(a, out_a)], dtype=np.int64).reshape(-1))
    ring((1, 1), (256,), (table, np.int32(1), np.int32(0)))      # the words are cached
    rewrite((1,), (32,), (np.uint64(table.data.ptr), np.int64(b.data.ptr), np.int64(out_b.data.ptr)))
    reader = out_b * cp.float32(1.0)              # taken before the second gather
    ring((1, 1), (256,), (table, np.int32(1), np.int32(0)))
    assert [int(w) for w in used[-1][:2]] == [b.data.ptr, out_b.data.ptr]
    assert np.array_equal(out_a.get(), np.arange(n, dtype=np.float32))
    assert np.array_equal(reader.get(), np.zeros(n, np.float32))
    assert np.array_equal(out_b.get(), np.arange(n, dtype=np.float32) + 100)


def test_a_table_rewritten_through_an_alias_is_read_again(monkeypatch):
    """A write through another storage over the table's buffer (an alias,
    as hostAlloc memory has) gives only the alias a new value: the words
    read before it must not be used for the next gather."""
    _mlx_or_skip()
    import cupy as cp
    from woof.backend.facade._array import ndarray
    from woof.backend.metal import loader
    from woof.core.kernels import get_kernel

    used = []
    words = loader._device_table_words
    monkeypatch.setattr(loader, "_device_table_words",
                        lambda view: used.append(words(view)) or used[-1])
    ring = get_kernel("microphysics_validation", "mp_ring_copy")
    n = 64
    a, b = cp.asarray(np.arange(n, dtype=np.float32)), cp.asarray(np.arange(n, dtype=np.float32) + 100)
    out_a, out_b = cp.zeros(n, cp.float32), cp.zeros(n, cp.float32)
    row = lambda src, dst: np.asarray([(src.data.ptr, dst.data.ptr, 1, 0, 1, 0, n, n, n)],
                                      dtype=np.int64).reshape(-1)
    table = cp.asarray(row(a, out_a))
    ring((1, 1), (256,), (table, np.int32(1), np.int32(0)))      # the words are cached
    alias = table._st.alias(np.int32)
    assert alias is not table._st
    as_int32 = ndarray((2 * table.size,), np.int32, _storage=alias, _offset=0)
    as_int32[...] = cp.asarray(row(b, out_b).view(np.int32))
    ring((1, 1), (256,), (table, np.int32(1), np.int32(0)))
    assert [int(w) for w in used[-1][:2]] == [b.data.ptr, out_b.data.ptr]
    assert np.array_equal(out_b.get(), np.arange(n, dtype=np.float32) + 100)
