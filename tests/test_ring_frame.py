"""zero_frame / copy_frame store at exactly the elements the slice
assignments they replace store at (woof/core/ring_frame.py), word for word."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _slices_zero(a, x_lo, x_hi, y_lo, y_hi):
    if x_lo:
        a[..., :x_lo] = 0
    if x_hi:
        a[..., a.shape[-1] - min(x_hi, a.shape[-1]):] = 0
    if y_lo:
        a[..., :y_lo, :] = 0
    if y_hi:
        a[..., a.shape[-2] - min(y_hi, a.shape[-2]):, :] = 0


@pytest.mark.parametrize("shape", [(5, 9, 11), (3, 12, 13), (7, 4), (2, 3, 6, 5)])
@pytest.mark.parametrize("widths", [(1, 1, 1, 1), (3, 4, 3, 4), (2, 0, 0, 2),
                                    (0, 0, 1, 1), (1, 1, 0, 0), (5, 5, 5, 5), (20, 1, 1, 20)])
def test_zero_frame_matches_the_slice_assignments(shape, widths):
    from woof.core.ring_frame import zero_frame
    rng = np.random.default_rng(3)
    host = rng.standard_normal(shape).astype(np.float32)
    host[host > 1.5] = -0.0                    # signed zeros stay as they were
    want = host.copy()
    _slices_zero(want, *widths)
    dev = cp.asarray(host)
    assert zero_frame(dev, *widths)
    assert np.array_equal(cp.asnumpy(dev).view(np.uint32), want.view(np.uint32))


@pytest.mark.parametrize("shape,sz", [((4, 10, 12), 1), ((3, 9, 9), 4), ((2, 16, 7), 3)])
def test_copy_frame_matches_the_held_ring_copy(shape, sz):
    from woof.core.ring_frame import copy_frame
    rng = np.random.default_rng(5)
    tend = rng.standard_normal(shape).astype(np.float32)
    held = rng.standard_normal(shape).astype(np.float32)
    nz, ny, nx = shape
    want = tend.copy()
    want[:, :sz, :] = held[:, :sz, :]
    want[:, ny - sz:, :] = held[:, ny - sz:, :]
    want[:, sz:ny - sz, :sz] = held[:, sz:ny - sz, :sz]
    want[:, sz:ny - sz, nx - sz:] = held[:, sz:ny - sz, nx - sz:]
    dev = cp.asarray(tend)
    assert copy_frame(dev, cp.asarray(held), sz, sz, sz, sz)
    assert np.array_equal(cp.asnumpy(dev).view(np.uint32), want.view(np.uint32))


def test_a_buffer_the_kernel_does_not_cover_is_left_to_the_caller():
    from woof.core.ring_frame import copy_frame, zero_frame
    a = cp.ones((4, 6, 8), dtype=cp.float32)
    view = a[:, :, ::2]
    assert not zero_frame(view, 1, 1, 1, 1)
    assert not copy_frame(a, cp.ones((4, 6, 7), dtype=cp.float32), 1, 1, 1, 1)
    assert not zero_frame(cp.ones((5,), dtype=cp.float32), 1, 1, 1, 1)
    assert bool((a == 1).all())
