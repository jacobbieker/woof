"""cupy.power on binary32 under GPUWM_WRF_EXACT=1 gives CUDA's powf words.

Every WRF-exact microphysics adapter forms the Exner function as
``cupy.power(p / P0, RCP)``.  On the MLX backend that was Metal's pow, which
is not CUDA's powf: on the NSSL column oracle's 4800 cells 1601 Exner words
differed by 1 ULP, so ``nssl2_sediment_all_parallel_64`` was handed a
different temperature and the production step left WRF.  The facade now
runs a transcription of CUDA's powf (woof/backend/translate/cuda_powf.cuh).

The expected words are CUDA's: the oracle fixture's ``in_exner`` is the
CUDA adapter's own ``pii`` (tools/nssl2_wrf461_column_oracle/
run_gpu_columns.py), and the table below was measured with strict CuPy on
an RTX A6000 (each row: x, y, CuPy's word; rows where Metal's pow differs,
subnormal and negative bases, and IEEE special cases).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wrf_exact_process import wrf_exact_process

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "tests" / "data" / "oracles" / "nssl2_column" / "nssl2-column-oracle-v2.npz"

_CUDA_WORDS = (
    (0x68809DB9, 0xBFDD89A7, 0x00000086),
    (0x186E5ACD, 0x3FEB42C0, 0x0000002C),
    (0x6B1A0140, 0xBFBC9B1E, 0x00155069),
    (0x35F96013, 0xC0706001, 0x63357243),
    (0x39B7791B, 0x408D41C7, 0x261FF4BD),
    (0x64865018, 0xBFBE2694, 0x087A076F),
    (0xBFD4A249, 0x40A00000, 0xC14A6913),
    (0xBFFC022F, 0xC0A00000, 0xBD0A7684),
    (0xBFD02641, 0xC0C00000, 0x3D5D7F4E),
    (0x402A336D, 0xC0C00000, 0x3B39435A),
    (0x3F27EDC1, 0x40800000, 0x3E3D99CB),
    (0x3FBF53A8, 0xC0A00000, 0x3E093CA7),
    (0x7F7FC99E, 0xBF800000, 0x002006CE),
    (0x006CE3EE, 0xBF800000, 0x7E96769A),
    (0x00000001, 0x3F800000, 0x00000001),
    (0x00000001, 0xBF000000, 0x64B504F3),
    (0x40000000, 0x3F000000, 0x3FB504F3),
    (0x40000000, 0xC0400000, 0x3E000000),
    (0x80000001, 0x3F000000, 0x7FFFFFFF),
    (0x80000001, 0xBF800000, 0xFF800000),
    (0xBF800000, 0x00000001, 0x7FFFFFFF),
    (0xBF800000, 0xBF800000, 0xBF800000),
    (0xFF800000, 0x00000001, 0x7F800000),
    (0xFF800000, 0xFF800000, 0x00000000),
    (0x7F800000, 0x7F800000, 0x7F800000),
    (0x7F7FC99E, 0x7F800000, 0x7F800000),
    (0x006CE3EE, 0xFF800000, 0x7F800000),
    (0x7FC00000, 0x7FC00000, 0x7FFFFFFF),
    (0x7FC00000, 0x40400000, 0x7FFFFFFF),
    (0x00000000, 0xC0400000, 0x7F800000),
    (0x80000000, 0x3F000000, 0x00000000),
    (0x80000000, 0x80000000, 0x3F800000),
    (0x3F800000, 0x7FC00000, 0x3F800000),
)


def _words(a):
    return np.asarray(a, dtype=np.float32).view(np.uint32)


@pytest.mark.gpu
@wrf_exact_process
def test_exner_power_matches_cuda_words():
    import cupy as cp

    from woof.core import constants as c

    data = np.load(_FIXTURE, allow_pickle=False)
    pressure = cp.asarray(data["in_pressure"], dtype=cp.float32)
    pii = cp.power(pressure / np.float32(c.P0), np.float32(c.RCP))
    differ = int(np.count_nonzero(_words(cp.asnumpy(pii)) != _words(data["in_exner"])))
    assert differ == 0, f"{differ} of {pii.size} Exner words differ from CUDA's"


@pytest.mark.gpu
@wrf_exact_process
def test_power_matches_cuda_powf_words():
    import cupy as cp

    import woof  # noqa: F401  (strict NVRTC options for CuPy's own kernels)

    table = np.array(_CUDA_WORDS, dtype=np.uint32)
    x, y = table[:, 0].view(np.float32), table[:, 1].view(np.float32)
    got = _words(cp.asnumpy(cp.power(cp.asarray(x), cp.asarray(y))))
    bad = [(f"{a:08X}", f"{b:08X}", f"{w:08X}", f"{g:08X}")
           for (a, b, w), g in zip(_CUDA_WORDS, got) if w != g]
    assert not bad, bad
