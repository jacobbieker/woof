"""Split exact reductions on the Metal facade (WOOF_METAL_SPLIT_REDUCTIONS).

The facade's ``ReductionKernel`` folds each output serially on one thread.
An exact, order-free, idempotent fold (``a | b``, ``a || b``, ``a & b``,
``a && b``) with a plain ``post_map`` is folded in slices and then across the
slices instead: the same value, from many threads.  These tests compare the
split against the serial fold and NumPy.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from woof.backend.facade import _elementwise as metal


@pytest.mark.parametrize("reduce_expr, post_map, splits", [
    ("a | b", "y = a", True), ("a||b", "y = a;", True), ("b && a", "y=a", True),
    ("a + b", "y = a", False), ("max(a, b)", "y = a", False), ("a | b", "y = a + 1", False),
])
def test_only_exact_idempotent_folds_split(reduce_expr, post_map, splits):
    kernel = metal.ReductionKernel("int32 x", "int32 y", "x", reduce_expr, post_map, "0", "k")
    assert kernel.splittable() is splits


def test_generic_output_without_reduce_type_stays_serial():
    kernel = metal.ReductionKernel("T x", "T y", "x", "a | b", "y = a", "0", "k")
    assert not kernel.splittable()


def test_slice_counts(monkeypatch):
    monkeypatch.delenv("WOOF_METAL_SPLIT_REDUCTIONS", raising=False)
    assert metal._slices(1, metal.SPLIT_MIN_ELEMENTS - 1) == 1
    assert metal._slices(1, 400_000) == 1024
    assert metal._slices(3, 8192) == 8192 // metal.SPLIT_SLICE_ELEMENTS
    assert metal._slices(metal.SPLIT_MAX_OUTPUTS, 1 << 20) == 1
    assert metal._slices(metal.SPLIT_MAX_OUTPUTS - 1, 1 << 20) == metal.SPLIT_MAX_THREADS // (metal.SPLIT_MAX_OUTPUTS - 1)
    monkeypatch.setenv("WOOF_METAL_SPLIT_REDUCTIONS", "0")
    assert metal._slices(1, 400_000) == 1


def _metal():
    if importlib.util.find_spec("mlx") is None:
        pytest.skip("needs mlx (Apple silicon)")
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import cupy as cp
    return cp


@pytest.mark.parametrize("n", [1, 4095, 4096, 5000, 400_000, 1_000_003])
def test_split_matches_serial_and_numpy(n, monkeypatch):
    cp = _metal()
    rng = np.random.default_rng(n)
    equal = cp.ReductionKernel(
        "uint8 v0, uint8 v1, uint8 p0, uint8 p1", "bool out",
        "(v0 != p0) || (v1 != p1)", "a || b", "out = a", "false", "split_equal_bytes")
    orred = cp.ReductionKernel("int32 x", "uint32 y", "(unsigned int)x", "a | b", "y = a",
                               "0", "split_status_or")
    anded = cp.ReductionKernel("int32 x", "int32 y", "x", "a & b", "y = a", "-1", "split_and")
    nonfinite = cp.ReductionKernel("T x", "int32 y", "isfinite(x) ? 0 : 1", "a | b", "y = a",
                                   "0", "split_nonfinite")
    v = [rng.integers(0, 256, n, dtype=np.uint8) for _ in range(2)]
    x = rng.integers(0, 1 << 20, n, dtype=np.int32)
    x[rng.integers(0, n)] |= 1 << 30
    # sparse, distinct bits per row and position: a slice that reads another
    # output's row, or skips or repeats an element, changes the value
    m = np.zeros((3, n), dtype=np.int32)
    w = np.full(n, -1, dtype=np.int32)
    for row in range(3):
        for bit, at in enumerate(rng.integers(0, n, 10)):
            m[row, at] |= np.int32(1 << (3 * bit + row))
    for bit, at in enumerate(rng.integers(0, n, 12)):
        w[at] &= ~np.int32(1 << (2 * bit + 1))
    f = rng.standard_normal(n).astype(np.float32)
    f[-1] = np.inf

    def run():
        cases = []
        for flip in (None, 0, n - 1, n // 2):
            p = [a.copy() for a in v]
            if flip is not None:
                p[1][flip] ^= 1
            cases.append(bool(equal(*map(cp.asarray, v), *map(cp.asarray, p))))
        cases.append(int(orred(cp.asarray(x))))
        cases.append(cp.asnumpy(orred(cp.asarray(m), axis=1)).tolist())
        cases.append(int(anded(cp.asarray(w))))
        cases.append(int(nonfinite(cp.asarray(f))))
        return cases

    split = run()
    monkeypatch.setenv("WOOF_METAL_SPLIT_REDUCTIONS", "0")
    serial = run()
    want = [False, True, True, True, int(np.bitwise_or.reduce(x.astype(np.uint32))),
            np.bitwise_or.reduce(m.astype(np.uint32), axis=1).tolist(),
            int(np.bitwise_and.reduce(w)), 1]
    assert split == serial == want
