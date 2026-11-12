"""accumulate_scaled (woof/core/bandwidth_glue.py) stores the words of
``dst += src * scale`` done as CuPy's multiply then add."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def test_accumulate_scaled_matches_the_two_ufuncs_word_for_word():
    from woof.core.bandwidth_glue import accumulate_scaled
    rng = np.random.default_rng(13)
    for scale in (np.float32(24.0), np.float32(7.5), np.float32(1e-3)):
        src_h = (rng.standard_normal((120, 150)) * 300).astype(np.float32)
        dst_h = (rng.standard_normal((120, 150)) * 1e6).astype(np.float32)
        src_h[:3] = np.float32(1e-39)               # subnormal products and sums
        dst_h[:2] = np.float32(-0.0)
        want_dev = cp.asarray(dst_h)
        want_dev += cp.asarray(src_h) * scale
        got = cp.asarray(dst_h)
        assert accumulate_scaled(got, cp.asarray(src_h), scale)
        assert np.array_equal(got.get().view(np.uint32), want_dev.get().view(np.uint32))


def test_accumulate_scaled_leaves_unsupported_operands_to_the_caller():
    from woof.core.bandwidth_glue import accumulate_scaled
    a = cp.ones((4, 6), dtype=cp.float32)
    assert not accumulate_scaled(a, cp.ones((4, 6), dtype=cp.float64), np.float32(2))
    assert not accumulate_scaled(a[:, ::2], cp.ones((4, 3), dtype=cp.float32), np.float32(2))
    assert not accumulate_scaled(a[:, 1:], a[:, :-1], np.float32(2))
    assert bool((a == 1).all())
