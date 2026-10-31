"""In-place Metal launches are ordered after earlier readers (WOOF_BACKEND=mlx).

MLX fences one command encoder after another only for read-after-write: it
never writes a buffer in place, so an earlier encoder that merely READS a
buffer is not waited for.  The translated kernels write in place, so without
woof's own ordering (woof/backend/metal/ext/launch.cpp: each launch is its
own pass, ending with wait_in_flight) a launch could overwrite a buffer a
still-running MLX op was reading.  Breakage it prevents: the UW PBL driver
seam's 10 steps gave a different state on every repeat and, one run in two,
"the UW PBL returned non-finite du" (2026-10-29).
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")

_SCALE = r'''
extern "C" __global__ void scale_in_place(float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; for (int k = 0; k < 64; ++k) v = v * s + 1.0f; x[i] = v; }
}
'''


def test_a_launch_waits_for_earlier_readers_of_what_it_overwrites():
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import cupy as cp
    import mlx.core as mx

    kernel = cp.RawKernel(_SCALE, "scale_in_place")
    n = 1 << 20
    want = None
    for _ in range(8):
        x = cp.asarray(np.linspace(0.0, 1.0, n, dtype=np.float32))
        snapshots = []
        for step in range(12):
            # an MLX-native reader of x, committed (async) before the write
            snap = x * cp.float32(2.0)
            mx.async_eval(snap._st.arr)
            snapshots.append(snap)
            kernel(((n + 255) // 256,), (256,), (x, np.float32(0.5), np.int32(n)))
        words = np.concatenate([s.get().view(np.uint32) for s in snapshots] + [x.get().view(np.uint32)])
        if want is None:
            want = words
        assert np.array_equal(words, want), "a reader saw a later in-place write"


def test_values_taken_before_an_in_place_launch_keep_their_words():
    """``before = x.copy()`` or ``x * 2``, then a kernel writing ``x`` in
    place: the copy and the expression hold x's old words (CUDA's stream
    order), a view of x shows the new ones.  Breakage it prevents: MLX
    computed the lazy copy after the launch, so Lin's h_diabatic -- theta
    after minus a copy from before -- was 0 everywhere (2026-10-29)."""
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import cupy as cp

    bump = cp.RawKernel(r'''extern "C" __global__ void bump(float* x, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] += 1.0f; }''', "bump")
    x = cp.zeros(64, dtype=cp.float32)
    before, doubled, view = x.copy(), x * cp.float32(2.0), x[2:5]
    bump((1,), (64,), (x, np.int32(64)))
    assert not before.get().any()
    assert not doubled.get().any()
    assert (view.get() == 1.0).all()
