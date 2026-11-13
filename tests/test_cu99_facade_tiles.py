"""CU99 off CUDA: the facade path submits the Rust scheduler's tiles.

gpuwm_cu99_launch_tiles hands cuLaunchKernel 256-thread blocks over
``tile``-column spans with the span's first column as the last argument.
Metal and the host backend have no cuLaunchKernel, so
woof.core.cu99_launch._launch_tiles_facade issues the same spans through the
kernel object; the WRF 4.6.1 word-for-word test (test_kf_old_wrf461_parity)
grades its arithmetic on Metal.
"""

import numpy as np
import pytest

from woof.core import cu99_launch


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, grid, block, args):
        self.calls.append((grid, block, args))


def _options(ny, nx, nz=10):
    return (0, 20, 0, 1, 1, 1, 0, nz, ny, nx)


def test_tiles_match_the_rust_loop():
    kernel = _Recorder()
    arrays = tuple(object() for _ in range(28))
    cu99_launch._launch_tiles_facade(kernel, arrays, (60.0, 3000.0, 5.0),
                                     _options(26, 50), tile=512)
    spans = [(c[0], c[2][-1]) for c in kernel.calls]
    assert spans == [((2,), np.int32(0)), ((2,), np.int32(512)), ((2,), np.int32(1024))]
    assert all(c[1] == (256,) for c in kernel.calls)
    grid, block, args = kernel.calls[0]
    assert len(args) == 42 and args[:28] == arrays
    assert [type(a) for a in args[28:31]] == [np.float32] * 3
    assert [type(a) for a in args[31:]] == [np.int32] * 11


@pytest.mark.parametrize("scalars, options, tile", [
    ((60.0, 3000.0, 5.0), _options(26, 50), 300),          # a non-block-aligned inner tile
    ((0.0, 3000.0, 5.0), _options(26, 50), 512),           # dt <= 0
    ((60.0, float("nan"), 5.0), _options(26, 50), 512),    # non-finite dx
    ((60.0, 3000.0, 5.0), (2, 20, 0, 1, 1, 1, 0, 10, 26, 50), 512),  # a flag not 0/1
])
def test_metadata_the_rust_side_rejects_is_rejected(scalars, options, tile):
    kernel = _Recorder()
    with pytest.raises(ValueError, match="rejected metadata"):
        cu99_launch._launch_tiles_facade(kernel, tuple(object() for _ in range(28)),
                                         scalars, options, tile=tile)
    assert kernel.calls == []
