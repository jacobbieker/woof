"""Column-major arrays on the MLX backend (WOOF_BACKEND=mlx).

CuPy follows NumPy's memory-order rules: ``reshape(..., order="F")`` reads
indices column-major, and ``astype``, ``copy(order="K")`` and the ``*_like``
constructors keep a column-major prototype's layout.  Kernels index those
buffers directly, so a layout the facade dropped is a scrambled field.
Breakage it prevents: the SATMEDMF driver packs its columns with
``reshape(nc, -1, order="F")``; read row-major, the humidity slots held other
tracers and the step stopped with "CCPP humidity increment reached unit
specific humidity".
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")

_COLUMN_MAJOR_INDEX = r'''
extern "C" __global__ void column_major_index(double* x, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) x[i] = (double)((i % rows) * 100 + i / rows);
}
'''


@pytest.fixture
def cp():
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import cupy
    return cupy


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fortran_order_reshape_ravel_and_copy_match_numpy(cp, dtype):
    host = np.arange(2 * 3 * 4, dtype=dtype).reshape(2, 3, 4)
    for source in (host, np.asfortranarray(host)):
        a = cp.asarray(source)
        for order in ("C", "F", "A", "K"):
            if order != "K":
                assert np.array_equal(a.reshape(2, -1, order=order).get(),
                                      source.reshape(2, -1, order=order))
            assert np.array_equal(a.ravel(order=order).get(), source.ravel(order=order))
            assert np.array_equal(a.flatten(order=order).get(), source.flatten(order=order))
            assert a.copy(order=order).strides == source.copy(order=order).strides
        both = cp.concatenate([a.reshape(2, -1, order="F")] * 2, axis=1)
        assert np.array_equal(cp.asfortranarray(both).get(),
                              np.concatenate([source.reshape(2, -1, order="F")] * 2, axis=1))


def test_astype_and_like_constructors_keep_column_major_layout(cp):
    f = cp.asfortranarray(cp.asarray(np.arange(12.0).reshape(3, 4)))
    for result in (f.astype(cp.float32), cp.empty_like(f), cp.zeros_like(f),
                   cp.ones_like(f), cp.full_like(f, 2.0), f.copy(order="K"), cp.array(f, order="F")):
        assert result.is_f_contiguous() and not result.is_c_contiguous()
    assert np.array_equal(f.astype(cp.float32).get(), np.arange(12.0, dtype=np.float32).reshape(3, 4))
    assert cp.empty_like(f, order="C").is_c_contiguous()


def test_kernel_writes_an_empty_like_buffer_column_major(cp):
    rows, cols = 3, 5
    prototype = cp.asfortranarray(cp.zeros((rows, cols), dtype=cp.float64))
    out = cp.empty_like(prototype)
    kernel = cp.RawModule(code=_COLUMN_MAJOR_INDEX).get_function("column_major_index")
    kernel((1,), (32,), (out, np.int32(rows), np.int32(cols)))
    expected = np.add.outer(np.arange(rows) * 100, np.arange(cols)).astype(np.float64)
    assert np.array_equal(out.get(), expected)
