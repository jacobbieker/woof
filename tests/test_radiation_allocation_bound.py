"""The batched SW chain fits the MLX facade's largest allocation.

A full-physics hex forecast on Metal stopped at step 0: spcvmc's wk
workspace for a 2,048-column chunk at 151 levels was one 2,216,689,664-byte
buffer, and MLX sizes an array with int32.  The chunk is narrowed so wk fits
cupy.MAX_ALLOCATION_BYTES where the backend publishes it (the facade); CUDA's
cupy publishes none and keeps its width.  Chunk width is workspace shape
only (per-column results are bitwise identical at any width).
"""

import importlib.util

import numpy as np
import pytest

from woof.core.rrtmg_sw import NGPTSW, SPCVMC_WK_ARRAYS, sw_allocation_column_bound


def test_the_bound_fits_the_largest_allocation_and_is_absent_without_one():
    limit = 2**31 - 1
    for nlay in (40, 55, 150, 255):
        nc = sw_allocation_column_bound(nlay, limit)
        per_column = NGPTSW * SPCVMC_WK_ARRAYS * (nlay + 1) * 4
        assert nc * per_column <= limit < (nc + 1) * per_column
    assert sw_allocation_column_bound(150, 2**31 - 1) < 2048     # the hex case narrows
    assert sw_allocation_column_bound(150, None) >= 2**40          # CUDA: no bound


@pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="needs mlx (Apple silicon)")
def test_the_facade_names_its_allocation_limit():
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import cupy as cp
    assert cp.MAX_ALLOCATION_BYTES == 2**31 - 1
    with pytest.raises(MemoryError, match="element limit per allocation"):
        cp.empty(2**31, dtype=np.uint8)
