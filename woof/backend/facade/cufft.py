"""``cupy.cuda.cufft`` on the mlx backend: plans over NumPy's FFT.

MLX's FFT runs in single precision only, and woof's cuFFT callers are
binary64 (the global polar filter's D2Z/Z2D) or must match NumPy's
``cupy.fft`` semantics, so a plan's transform runs on the host through
NumPy's pocketfft and writes its result into the output array in place --
the buffer the plan's caller and its kernels hold, as cuFFT writes it.

Semantics follow cuFFT's, not NumPy's:

* the inverse (C2R, Z2D, and C2C with ``CUFFT_INVERSE``) is unnormalised:
  NumPy's inverse times ``nx``;
* C2R/Z2D read the ``nx // 2 + 1`` Hermitian half; the imaginary parts of
  the zero (and, for even ``nx``, the Nyquist) mode are ignored, as both
  cuFFT and pocketfft ignore them;
* single-precision types (R2C, C2R, C2C) compute in single precision.

The transform is not cuFFT's algorithm, so values agree with CUDA's to
rounding, not bitwise (``tests/test_facade_cufft.py`` states the bound).

During a CUDA-graph capture a plan's ``fft`` is recorded (``_capture.defer``)
(:mod:`woof.backend.facade._capture`): the replay runs the transform on the
arrays' values at that point in the stream, as the captured cuFFT node does.
"""

from __future__ import annotations

import types

import numpy as np

CUFFT_R2C = 0x2a
CUFFT_C2R = 0x2c
CUFFT_C2C = 0x29
CUFFT_D2Z = 0x6a
CUFFT_Z2D = 0x6c
CUFFT_Z2Z = 0x69
CUFFT_FORWARD = -1
CUFFT_INVERSE = 1

#: fft type -> (input dtype, output dtype, kind)
_TYPES = {
    CUFFT_R2C: (np.float32, np.complex64, "r2c"),
    CUFFT_C2R: (np.complex64, np.float32, "c2r"),
    CUFFT_C2C: (np.complex64, np.complex64, "c2c"),
    CUFFT_D2Z: (np.float64, np.complex128, "r2c"),
    CUFFT_Z2D: (np.complex128, np.float64, "c2r"),
    CUFFT_Z2Z: (np.complex128, np.complex128, "c2c"),
}


class CuFFTError(RuntimeError):
    pass


def _work_area():
    """A plan's work area: none on the device (the transform runs on the host)."""
    return types.SimpleNamespace(mem=types.SimpleNamespace(size=0, ptr=0), ptr=0, size=0)


def transform(host: np.ndarray, kind: str, axes, shape, direction: int, out_dtype) -> np.ndarray:
    """cuFFT's transform of ``host`` over ``axes`` (logical sizes ``shape``)."""
    axes, shape = tuple(axes), tuple(shape)
    if kind == "r2c":
        r = np.fft.rfftn(host, s=shape, axes=axes)
    elif kind == "c2r":
        r = np.fft.irfftn(host, s=shape, axes=axes, norm="forward")
    elif direction == CUFFT_FORWARD:
        r = np.fft.fftn(host, s=shape, axes=axes)
    elif direction == CUFFT_INVERSE:
        r = np.fft.ifftn(host, s=shape, axes=axes, norm="forward")
    else:
        raise CuFFTError(f"invalid cuFFT direction {direction}")
    return np.ascontiguousarray(r.astype(out_dtype, copy=False))


class _Plan:
    """What Plan1d and PlanNd share: a transform of ``batch`` signals of
    logical size ``shape`` over the trailing axes of packed arrays."""

    def __init__(self, shape, fft_type, batch):
        if fft_type not in _TYPES:
            raise CuFFTError(f"unknown cuFFT type {fft_type:#x}")
        self.shape = tuple(int(n) for n in shape)
        self.fft_type = fft_type
        self.batch = int(batch)
        in_dtype, out_dtype, self.kind = _TYPES[fft_type]
        self.in_dtype, self.out_dtype = np.dtype(in_dtype), np.dtype(out_dtype)
        self.work_area = _work_area()
        self.gpus = None

    def _sizes(self):
        last = self.shape[-1] // 2 + 1
        half = self.shape[:-1] + (last,)
        if self.kind == "r2c":
            return self.shape, half
        if self.kind == "c2r":
            return half, self.shape
        return self.shape, self.shape

    def _check(self, a, out):
        from woof.backend.facade._array import ndarray
        n_in, n_out = (int(np.prod(s)) * self.batch for s in self._sizes())
        for what, x, dtype, n in (("input", a, self.in_dtype, n_in), ("output", out, self.out_dtype, n_out)):
            if not isinstance(x, ndarray):
                raise CuFFTError(f"the plan's {what} must be a device array")
            if x.dtype != dtype:
                raise CuFFTError(f"the plan's {what} is {x.dtype}, not {dtype}")
            if x.size != n or not x.is_c_contiguous():
                raise CuFFTError(f"the plan's {what} must be {n} packed elements, not {x.shape}")

    def fft(self, a, out, direction):
        """Execute the plan from ``a`` into ``out`` (both packed device arrays)."""
        from woof.backend.facade import _capture
        self._check(a, out)
        if _capture.defer(self._execute, a, out, direction):
            return
        self._execute(a, out, direction)

    def _execute(self, a, out, direction):
        src, dst = self._sizes()
        host = a.get().reshape((self.batch,) + src)
        axes = tuple(range(1, len(self.shape) + 1))
        r = transform(host, self.kind, axes, self.shape, direction, self.out_dtype)
        out.set(r.reshape(out.shape))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Plan1d(_Plan):
    """``cupy.cuda.cufft.Plan1d(nx, fft_type, batch)``."""

    def __init__(self, nx, fft_type, batch, devices=None, out=None):
        super().__init__((nx,), fft_type, batch)
        self.nx = int(nx)


class PlanNd(_Plan):
    """``cupy.cuda.cufft.PlanNd`` for packed C-order batches over the
    trailing axes (``cupy.fft._fft._get_cufft_plan_nd``'s plans)."""

    def __init__(self, shape, fft_type, batch=1):
        super().__init__(shape, fft_type, batch)


def get_cufft_plan_nd(shape, fft_type, axes=None, order="C", out_size=None, to_cache=True):
    """``cupy.fft._fft._get_cufft_plan_nd``: a plan over ``axes`` (the
    trailing ones) of a C-order array of ``shape``."""
    shape = tuple(int(n) for n in shape)
    nd = len(shape)
    axes = tuple(range(nd)) if axes is None else tuple(sorted(int(a) % nd for a in np.atleast_1d(axes)))
    if order != "C" or axes != tuple(range(nd - len(axes), nd)):
        raise NotImplementedError("the mlx backend's cuFFT plans cover trailing axes of C-order arrays")
    logical = list(shape[nd - len(axes):])
    if fft_type in (CUFFT_C2R, CUFFT_Z2D):
        # the C2R plan's input shape holds the Hermitian half on the last axis
        logical[-1] = (int(out_size) if out_size is not None else 2 * (logical[-1] - 1))
    batch = int(np.prod(shape[:nd - len(axes)]))
    return PlanNd(tuple(logical), fft_type, batch)
