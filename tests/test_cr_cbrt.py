"""The correctly rounded binary64 cube root, host and device.

``woof.core.cr_cbrt.cbrt`` (NumPy) and ``uw_cbrt``
(woof/core/kernels/cr_cbrt_flt64.cuh) are transcriptions of CORE-MATH's
``cr_cbrt``; ``woof.core.correctly_rounded_libm.cbrt_cr`` is an exact
integer reference that shares no code with either.  These tests grade a
sweep on every run; the 2^28-argument proof (and the upstream C) is
tools/uwpbl_wrf471_oracle/libm64/sweep_cbrt.py, recorded in gpu-cbrt.log (which
also counts how often the C library's cbrt misses the correctly rounded root).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core.correctly_rounded_libm import cbrt_cr
from woof.core.cr_cbrt import cbrt

REPO = Path(__file__).resolve().parents[1]
LIBM64 = REPO / "tools/uwpbl_wrf471_oracle/libm64"
UPSTREAM_SHA256 = (
    "0a86b7a8f433dfa4610023767102ea19eb3642ea6982871c61c956148837dde4")
#: The device transcription's bytes, which every kernel that prepends it
#: (the wind-farm power kernel) compiles.  Changing them is a decision, not
#: an edit: re-run tools/uwpbl_wrf471_oracle/libm64/sweep_cbrt.py (2^28
#: arguments, every one equal to upstream and the host) and re-pin.
HEADER_SHA256 = (
    "543414e82e9042820f6fb386a29bea072b9995ab8046e2c8d1d32e7aff7c2691")


def _arguments(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    part = count // 6
    cubes = (rng.integers(1, 2**17, part).astype(np.float64) ** 3
             * np.ldexp(1.0, 3 * rng.integers(-300, 300, part)))
    near = (cubes.view(np.int64)
            + rng.integers(-3, 4, part)).view(np.float64)
    x = np.concatenate([
        rng.integers(0, 2**64 - 1, part, dtype=np.uint64,
                     endpoint=True).view(np.float64),
        rng.uniform(0.5, 1.5, part), rng.uniform(0.0, 1e5, part),
        np.ldexp(rng.uniform(0.5, 1.0, part), rng.integers(-1074, 1025, part)),
        rng.integers(1, 2**52, part, dtype=np.uint64).view(np.float64),
        near])
    specials = [0.0, -0.0, math.inf, -math.inf, math.nan, 5e-324, -5e-324,
                2.2250738585072014e-308, 1.7976931348623157e308, 1.0, 8.0,
                27.0, -64.0, 0.001,
                float.fromhex("0x1.9b78223aa307cp+1"),
                float.fromhex("0x1.a202bfc89ddffp+2")]
    return np.concatenate([x, np.array(specials)])


def _words(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)


def test_the_host_cbrt_is_the_exact_reference_word_for_word():
    x = _arguments(120_000, 7)
    want = np.array([cbrt_cr(float(v)) for v in x])
    got = cbrt(x)
    finite = np.isfinite(x)
    assert np.count_nonzero(_words(got)[finite] != _words(want)[finite]) == 0
    # NaN in, NaN out; the signs of the zeros and infinities kept.
    np.testing.assert_array_equal(np.isnan(got), np.isnan(x))
    for v in (0.0, -0.0, math.inf, -math.inf):
        assert _words(cbrt(v)) == _words(v)
    assert cbrt(np.float64(27.0)) == 3.0
    assert cbrt(x.reshape(-1, 2)[:10]).shape == (10, 2)


def test_the_exact_reference_is_exact_on_its_own_terms():
    """Independent of any libm: the result's cube brackets x, and x sits on
    the correct side of the midpoint cube, in exact rational arithmetic."""
    from fractions import Fraction

    rng = np.random.default_rng(9)
    for v in np.concatenate([rng.uniform(1e-300, 1e300, 300),
                             rng.uniform(0.5, 1.5, 300)]):
        y = cbrt_cr(float(v))
        lo, hi = math.nextafter(y, 0.0), math.nextafter(y, math.inf)
        exact = Fraction(float(v))
        low_mid = (Fraction(lo) + Fraction(y)) / 2
        high_mid = (Fraction(y) + Fraction(hi)) / 2
        assert low_mid ** 3 < exact < high_mid ** 3, v


def test_the_upstream_source_is_pinned_and_recorded():
    upstream = LIBM64 / "upstream" / "cbrt.c"
    digest = hashlib.sha256(upstream.read_bytes()).hexdigest()
    assert digest == UPSTREAM_SHA256
    recorded = json.loads((LIBM64 / "upstream-sha256.json").read_text())
    assert recorded["upstream\\cbrt.c"] == UPSTREAM_SHA256
    header_path = REPO / "woof/core/kernels/cr_cbrt_flt64.cuh"
    assert hashlib.sha256(header_path.read_bytes()).hexdigest() == HEADER_SHA256
    header = header_path.read_text()
    assert UPSTREAM_SHA256 in header and "Alexei Sibidanov" in header
    assert "Permission is hereby granted" in header


@pytest.mark.gpu
@requires_gpu
def test_the_device_cbrt_is_the_host_and_the_reference_word_for_word():
    import cupy as cp
    from woof.core.cr_cbrt import device_cbrt

    x = _arguments(1 << 22, 11)
    gpu = cp.asnumpy(device_cbrt(cp.asarray(x)))
    assert np.count_nonzero(_words(gpu) != _words(cbrt(x))) == 0
    sample = x[::64]
    want = np.array([cbrt_cr(float(v)) for v in sample])
    finite = np.isfinite(sample)
    assert np.count_nonzero(
        _words(gpu[::64])[finite] != _words(want)[finite]) == 0
