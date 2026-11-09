"""complex128 arrays and cuFFT plans, as woof's FFT callers use them.

On CUDA these are CuPy's own; on the mlx backend the facade stores
complex128 as (real, imaginary) binary64 word pairs and runs a plan's
transform on the host through NumPy's pocketfft (woof/backend/facade/cufft.py).
The reference is NumPy's FFT with cuFFT's scaling: the facade must match it
bitwise (same algorithm), and cuFFT within a few binary64 ulps of the
signal's magnitude, which is the bound the facade's FFT claims against CUDA.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _facade():
    from woof import backend
    return backend.active() == "mlx"


#: cuFFT vs pocketfft: both are O(log n) binary64 algorithms; their results
#: differ by rounding, bounded relative to the spectrum's largest mode.
REL = 64 * np.finfo(np.float64).eps


def _close(got, want):
    if _facade():
        assert np.array_equal(got.view(np.uint64) if got.dtype == np.float64 else got,
                              want.view(np.uint64) if want.dtype == np.float64 else want)
    else:
        scale = float(np.max(np.abs(want))) or 1.0
        assert float(np.max(np.abs(got - want))) <= REL * scale


def test_complex128_arrays_hold_their_words():
    rng = np.random.default_rng(7)
    host = rng.standard_normal((5, 6)) + 1j * rng.standard_normal((5, 6))
    a = cp.asarray(host)
    assert a.dtype == cp.complex128 and a.shape == (5, 6) and a.nbytes == host.nbytes
    assert np.array_equal(a.get(), host)
    assert np.array_equal(a[1:4, ::2].get(), host[1:4, ::2])
    assert np.array_equal(a.T.copy().get(), host.T)
    assert np.array_equal(a.real.get(), host.real) and np.array_equal(a.imag.get(), host.imag)
    assert np.array_equal(a.view(cp.float64).get(), host.view(np.float64))
    assert np.array_equal((a * 2 + 1j).get(), host * 2 + 1j)
    assert np.allclose(cp.abs(a).get(), np.abs(host), rtol=1e-15, atol=0)
    b = cp.zeros((3, 4), cp.complex128)
    assert np.array_equal(b.get(), np.zeros((3, 4), np.complex128))
    b.fill(1 - 2j)
    b[1] = cp.asarray(host[0, :4])
    b[cp.asarray([0, 2]), 3] = 5j
    want = np.full((3, 4), 1 - 2j)
    want[1] = host[0, :4]
    want[[0, 2], 3] = 5j
    assert np.array_equal(b.get(), want)
    assert np.array_equal(b[cp.asarray([2, 0])].get(), want[[2, 0]])
    assert np.isclose(complex(cp.sum(a).get()), complex(np.sum(host)), rtol=1e-14, atol=0)


@pytest.mark.parametrize("nx", [32, 33])
def test_plan1d_d2z_and_z2d_are_cufft_transforms(nx):
    from cupy.cuda import cufft
    rng = np.random.default_rng(nx)
    batch = 7
    work_host = rng.standard_normal((batch, nx)) * 300.0
    work = cp.asarray(work_host)
    spectrum = cp.empty((batch, nx // 2 + 1), cp.complex128)
    forward = cufft.Plan1d(nx, cufft.CUFFT_D2Z, batch)
    backward = cufft.Plan1d(nx, cufft.CUFFT_Z2D, batch)
    forward.fft(work, spectrum, cufft.CUFFT_FORWARD)
    want = np.fft.rfft(work_host, axis=-1)
    _close(spectrum.get(), want)
    # cuFFT's inverse is unnormalised: n times NumPy's
    spectrum.set(want)
    back = cp.empty_like(work)
    backward.fft(spectrum, back, cufft.CUFFT_INVERSE)
    _close(back.get(), np.fft.irfft(want, n=nx, axis=-1) * nx)


def test_single_precision_and_c2c_plans():
    from cupy.cuda import cufft
    rng = np.random.default_rng(3)
    host = (rng.standard_normal((4, 16)) + 1j * rng.standard_normal((4, 16))).astype(np.complex64)
    a = cp.asarray(host)
    out = cp.empty_like(a)
    plan = cufft.Plan1d(16, cufft.CUFFT_C2C, 4)
    plan.fft(a, out, cufft.CUFFT_FORWARD)
    want = np.fft.fft(host, axis=-1)
    assert out.dtype == cp.complex64
    assert np.allclose(out.get(), want, rtol=1e-5, atol=1e-4)
    plan.fft(out, a, cufft.CUFFT_INVERSE)
    assert np.allclose(a.get(), host * 16, rtol=1e-5, atol=1e-3)


def test_nd_plans_expose_a_work_area():
    from cupy.cuda import cufft
    from cupy.fft._fft import _get_cufft_plan_nd
    plan = _get_cufft_plan_nd((3, 8, 10), cufft.CUFFT_C2C, axes=(-2, -1), order="C", to_cache=False)
    assert int(plan.work_area.mem.size) >= 0
    assert int(cufft.Plan1d(10, cufft.CUFFT_C2C, 24).work_area.mem.size) >= 0


def _kernel(ctype, body):
    return cp.RawKernel(f"""extern "C" __global__ void woof_test_step({ctype}* x, int n) {{
        int i = threadIdx.x; if (i < n) {{ {body} }} }}""", "woof_test_step")


def test_a_captured_plan_reruns_on_each_graph_launch():
    """A plan executed inside a CUDA-graph capture transforms the input's
    values at its place in the stream, on every launch (the polar filter's
    captured global step)."""
    from cupy.cuda import cufft
    nx, batch = 16, 3
    work = cp.zeros((batch, nx), cp.float64)
    spectrum = cp.empty((batch, nx // 2 + 1), cp.complex128)
    plan = cufft.Plan1d(nx, cufft.CUFFT_D2Z, batch)
    scale = _kernel("double", "x[i] = x[i] * 2.0 + 1.0;")
    plan.fft(work, spectrum, cufft.CUFFT_FORWARD)          # warm, as callers do
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        scale((1,), (work.size,), (work, np.int32(work.size)))
        plan.fft(work, spectrum, cufft.CUFFT_FORWARD)
        graph = stream.end_capture()
    expected = np.zeros((batch, nx))
    for _ in range(3):
        graph.launch(stream)
        stream.synchronize()
        expected = expected * 2.0 + 1.0
        _close(spectrum.get(), np.fft.rfft(expected, axis=-1))


def test_a_captured_cupy_fft_reruns_on_each_graph_launch():
    """cupy.fft inside a capture (spectral nudging's operator): the result
    array is the graph's own, rewritten by each launch."""
    host = np.arange(48, dtype=np.float32).reshape(6, 8)
    work = cp.asarray(host)
    double = _kernel("float", "x[i] = x[i] * 2.0f;")
    cp.fft.fft2(work, axes=(-2, -1))
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        double((1,), (work.size,), (work, np.int32(work.size)))
        spectrum = cp.fft.fft2(work, axes=(-2, -1))
        graph = stream.end_capture()
    for k in range(1, 3):
        graph.launch(stream)
        stream.synchronize()
        want = np.fft.fft2(host * 2 ** k, axes=(-2, -1))
        assert spectrum.dtype == cp.complex64
        assert np.allclose(spectrum.get(), want, rtol=1e-5, atol=1e-2)
