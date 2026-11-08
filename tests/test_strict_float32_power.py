"""cupy.power on binary32 under GPUWM_WRF_EXACT=1: CUDA's words on CUDA,
correctly rounded words on Metal by default, CUDA's words with the opt-in.

Every WRF-exact microphysics adapter forms the Exner function as
``cupy.power(p / P0, RCP)``.  On the MLX backend that was Metal's pow, which
is neither correctly rounded nor CUDA's powf.  The facade's strict power is
now WOOF's correctly rounded powf (woof/backend/translate/strict_powf.cuh),
or, with ``WOOF_STRICT_POWF=cuda`` in a source checkout, the opt-in
transcription of CUDA's powf (tools/cuda_powf/cuda_powf.cuh; NOTICE).

CUDA's words: the oracle fixture's ``in_exner`` is the CUDA adapter's own
``pii`` (tools/nssl2_wrf461_column_oracle/run_gpu_columns.py), and the
table below was measured with strict CuPy on an RTX A6000 (each row: x, y,
CuPy's word).  Correctly rounded words come from tools/strict_powf/
reference.py.  Where the two differ is the parity ledger's
"independent-powf" gap (woof/verify/backend_parity_ledger.json).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from wrf_exact_process import wrf_exact_process

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "strict_powf"))

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
    """The binary32 words of ``a``."""
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def _expects_cuda_words() -> bool:
    """CUDA, or Metal with the opt-in: CuPy's powf words."""
    from woof import backend
    if backend.active() != "mlx":
        return True
    from woof.backend.facade._ops import strict_powf_choice
    return strict_powf_choice() == "cuda"


def _differing(got, want):
    """Positions whose words differ (any NaN matches any NaN: payloads are
    the ledger's nan-payload gap)."""
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    return np.flatnonzero((_words(got) != _words(want)) & ~(np.isnan(got) & np.isnan(want)))


@pytest.mark.gpu
@wrf_exact_process
def test_exner_power_words():
    """(p/P0)**RCP on the oracle's pressures: the CUDA adapter's pii, or
    the correctly rounded words."""
    import woof  # noqa: F401  (the backend and strict options before CuPy)
    import cupy as cp

    from reference import powf
    from woof.core import constants as c

    data = np.load(_FIXTURE, allow_pickle=False)
    ratio = cp.asarray(data["in_pressure"], dtype=cp.float32) / np.float32(c.P0)
    pii = cp.asnumpy(cp.power(ratio, np.float32(c.RCP)))
    if _expects_cuda_words():
        want = data["in_exner"]
    else:
        want, _ = powf(cp.asnumpy(ratio), np.full(pii.shape, c.RCP, np.float32))
    differ = _differing(pii, want).size
    assert differ == 0, f"{differ} of {pii.size} Exner words differ"


@pytest.mark.gpu
@wrf_exact_process
def test_power_rows():
    """CuPy's powf words, subnormal and special cases among them, or the
    correctly rounded words of the same rows."""
    import woof  # noqa: F401  (the backend and strict options before CuPy)
    import cupy as cp

    from reference import powf

    table = np.array(_CUDA_WORDS, dtype=np.uint32)
    x, y = table[:, 0].view(np.float32), table[:, 1].view(np.float32)
    got = cp.asnumpy(cp.power(cp.asarray(x), cp.asarray(y)))
    want = table[:, 2].view(np.float32) if _expects_cuda_words() else powf(x, y)[0]
    bad = [(f"{_words(x)[i]:08X}", f"{_words(y)[i]:08X}", f"{_words(want)[i]:08X}",
            f"{_words(got)[i]:08X}") for i in _differing(got, want)]
    assert not bad, bad
    # two scalars give a 0-d array, as CuPy's ufunc does
    root = cp.power(np.float32(2.0), np.float32(0.5))
    assert root.shape == () and _words(cp.asnumpy(root)) == 0x3FB504F3


@pytest.mark.gpu
@wrf_exact_process
def test_exact_midpoints_round_to_even():
    """WOOF's powf on the binary32 midpoints x**y can land on exactly
    (4097^2 = 2^24 + 2^13 + 1, 66049^1.5 = 257^3): ties to even."""
    import woof  # noqa: F401  (the backend and strict options before CuPy)
    import cupy as cp

    if _expects_cuda_words():
        pytest.skip("CUDA's powf is not correctly rounded")
    from reference import powf

    xs = [4097.0, 4099.0, 66049.0, 66049.0 / 2 ** 20, 8191.0 / 4096, 3.0, 5.0, 625.0]
    ys = [2.0, 2.0, 1.5, 1.5, 2.0, 15.0, 11.0, 2.75]
    x, y = np.array(xs, np.float32), np.array(ys, np.float32)
    got = cp.asnumpy(cp.power(cp.asarray(x), cp.asarray(y)))
    want, _ = powf(x, y)
    assert _differing(got, want).size == 0, (got, want)
    assert got[0] == np.float32(16785408.0)
