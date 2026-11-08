"""cupy.random, divmod, argwhere and angle on the MLX facade.

They raised BackendUnsupported (the global ideal start's random field, the
structured-grid wave backend's divmod, the vortex CLI's argwhere).  The
facade computes them on the host and uploads the result; CuPy's own values
are the reference where they are deterministic.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")


@pytest.fixture
def cp():
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import cupy
    return cupy


def test_divmod_argwhere_and_angle_match_numpy(cp):
    a = np.arange(-7, 13, dtype=np.int32)
    q, r = cp.divmod(cp.asarray(a), 4)
    nq, nr = np.divmod(a, 4)
    assert cp.asnumpy(q).tolist() == nq.tolist() and cp.asnumpy(r).tolist() == nr.tolist()
    m = (a.reshape(4, 5) % 3) == 0
    assert cp.asnumpy(cp.argwhere(cp.asarray(m))).tolist() == np.argwhere(m).tolist()
    z = np.array([1 + 1j, -1 + 0j, 0 - 2j], dtype=np.complex64)
    assert np.allclose(cp.asnumpy(cp.angle(cp.asarray(z))), np.angle(z))


def test_random_draws_have_the_shape_dtype_and_range_asked_for(cp):
    x = cp.random.random((3, 4, 5), dtype=cp.float32)
    assert x.shape == (3, 4, 5) and x.dtype == np.float32
    host = cp.asnumpy(x)
    assert host.min() >= 0.0 and host.max() < 1.0 and np.unique(host).size > 50
    a = cp.asnumpy(cp.random.RandomState(7).random_sample(16))
    b = cp.asnumpy(cp.random.RandomState(7).random_sample(16))
    assert a.tobytes() == b.tobytes() and a.dtype == np.float64
    n = cp.asnumpy(cp.random.standard_normal((2000,), dtype=cp.float32))
    assert abs(float(n.mean())) < 0.1 and 0.9 < float(n.std()) < 1.1
