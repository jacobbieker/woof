"""E3SM v3.2.0's ocean-input accumulation on the GPU, opt-in.

The coupler runs the ocean less often than it merges x2o: each merge's x2o
is summed into an accumulator (``prep_ocn_accum``) and, before the ocean
runs, the sum is averaged and copied back into x2o (``prep_ocn_accum_avg``),
word for word (``driver-mct/main/prep_ocn_mod.F90`` over
``share/util/mct_mod.F90``'s ``mct_avect_accum`` and ``mct_aVect_avg``):

- :meth:`OceanAccumulator.accum`: with a count of 0 the accumulator becomes
  x2o (a copy), otherwise accumulator + x2o; the count goes up by 1;
- :meth:`OceanAccumulator.accum_avg`: with a count above 1 the accumulator
  is multiplied by ``1/count`` (rounded once: ``mct_aVect_avg``'s ``ravg``,
  not a division); then x2o becomes the accumulator, whatever the count
  (with none, the last average again), and the count goes back to 0.

The accumulator is set up as ``prep_ocn_init`` sets it up: x2o's fields,
zeroed, a count of 0. One ocean instance (``num_inst_ocn = 1``), one task,
``real(r8)``. This is not called by any default path. The kernels
(``ocn_accum_e3sm.cu``) are held word for word to the routines cut verbatim
and run on the pinned MCT, built with gfortran ``-O0 -ffp-contract=off``
(``tools/e3sm_coupler_oracle``, ``tests/test_e3sm_ocn_accum_parity.py``).
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

E3SM_TAG = "v3.2.0"
E3SM_COMMIT = "648b167b7f95267a159a9fdee25faf604c2bfc4f"
MCT_COMMIT = "82b0071e69d14330b75d23b0bc68543ebea9aadc"
SOURCE_FILES = ("driver-mct/main/prep_ocn_mod.F90:prep_ocn_accum,prep_ocn_accum_avg",
                "share/util/mct_mod.F90:mct_avect_accum,mct_aVect_avg")
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
_BLOCK = 256


def module_source() -> str:
    return files("woof.hex.coupled").joinpath("ocn_accum_e3sm.cu").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(), options, None, "ocn_accum_e3sm.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _options() -> tuple[str, ...]:
    from woof.hex import numerics as _hex_numerics
    return _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True, numerics=_hex_numerics._bind())


def _device_of(a) -> int:
    return int(a.device.id)


def mct_avg_factor(count: int) -> float:
    """``mct_aVect_avg``'s ``ravg = 1.0_R8/real(counter,R8)``."""
    return 1.0 / float(count)


class OceanAccumulator:
    """``x2oacc_ox`` and ``x2oacc_ox_cnt`` for one ocean instance on the
    flux grid: ``(lsize, nflds)`` float64, x2o's fields in x2o's order."""

    def __init__(self, lsize: int, x2o_fields, *, device: int | None = None):
        import cupy as cp
        fields = tuple(x2o_fields)
        if len(set(fields)) != len(fields):
            raise ValueError("x2o's field list must name each field once")
        if isinstance(lsize, bool) or not isinstance(lsize, (int, np.integer)) or lsize < 0:
            raise ValueError("lsize must be a non-negative integer")
        self.fields = fields
        self.device = cp.cuda.Device().id if device is None else int(device)
        with cp.cuda.Device(self.device):
            self.acc = cp.zeros((int(lsize), len(fields)), dtype=np.float64)
        self.count = 0

    def _check(self, x2o):
        import cupy as cp
        if not isinstance(x2o, cp.ndarray):
            raise TypeError(f"x2o must be a cupy (device) array; it is {type(x2o).__name__}")
        if x2o.dtype != np.float64 or not x2o.flags.c_contiguous or x2o.shape != self.acc.shape:
            raise ValueError(f"x2o must be a C-contiguous float64 array of shape {self.acc.shape}")
        if _device_of(x2o) != self.device or cp.cuda.Device().id != self.device:
            raise ValueError(f"x2o is on device {_device_of(x2o)} and the current device is "
                             f"{cp.cuda.Device().id}; the accumulator is on device {self.device}")
        if x2o.size and int(x2o.data.ptr) == int(self.acc.data.ptr):
            raise ValueError("x2o is the accumulator")

    def _launch(self, kernel: str, args) -> None:
        n = int(self.acc.size)
        if n:
            _module_on(self.device, _options()).get_function(kernel)(
                ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), (self.acc,) + tuple(args) + (np.int64(n),))

    def accum(self, x2o) -> None:
        """``prep_ocn_accum``: called after each merge with the merged x2o."""
        self._check(x2o)
        if self.count == 0:
            self.acc[...] = x2o
        else:
            self._launch("ocn_accum_add", (x2o,))
        self.count += 1

    def accum_avg(self, x2o) -> None:
        """``prep_ocn_accum_avg``: x2o, in place, becomes the average."""
        self._check(x2o)
        if self.count > 1:
            self._launch("ocn_accum_scale", (np.float64(mct_avg_factor(self.count)),))
        x2o[...] = self.acc
        self.count = 0
