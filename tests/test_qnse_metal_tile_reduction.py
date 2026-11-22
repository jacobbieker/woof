"""QNSE's EDMF mass-flux kernel compiles and loads on every backend.

``mfshconv_column``'s first tile pass folds the deepest processed level of
every column into ``*tile_top``.  CUDA does it as a warp reduction
(``__activemask`` + ``__shfl_down_sync``) and one ``atomicMax`` per warp;
Metal has no ``__activemask``, and until the ``__WOOF_METAL__`` branch the
translation of ``kernels/qnsepbl.cu`` failed, so QNSE never ran on Metal.
The value is an integer maximum, which no order of lanes changes, so the
Metal branch takes each SIMD group's maximum with ``__reduce_max_sync``
(``simd_max``) and its lowest live thread posts it with one ``atomicMax``.  The WRF word grade in
tests/test_qnse_wrf461_parity.py runs one-column tiles (``native_tile=False``,
tile_mode 0), so this file also runs the production two-pass tile path and
checks that the reduced ``tile_top`` over all 72 fixture columns (two full
SIMD groups and a partial one) is the maximum of the 72 one-column values.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

SOURCE = Path(__file__).resolve().parents[1] / "woof/core/kernels/qnsepbl.cu"


def _tile_branch():
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("    if (tile_mode == 1) {\n")
    end = text.index("#endif\n    }\n", start)
    return text[start:end]


def test_the_cuda_arm_keeps_the_warp_reduction():
    """Outside ``__WOOF_METAL__`` the preprocessed CUDA source is the
    pre-change warp reduction, so CUDA compiles the same code."""
    branch = _tile_branch()
    metal, cuda = branch.split("#else\n", 1)
    assert "#ifdef __WOOF_METAL__" in metal
    assert "__reduce_max_sync(0xffffffffu, processed)" in metal
    assert "__reduce_min_sync(0xffffffffu, (int)threadIdx.x)" in metal
    assert "if ((int)threadIdx.x == first) atomicMax(tile_top, group_top);" in metal
    code = re.sub(r"//[^\n]*", "", metal)
    assert "__activemask" not in code and "__shfl" not in code
    assert re.search(r"const unsigned mask = __activemask\(\);", cuda)
    assert "__shfl_down_sync(mask, processed, offset)" in cuda
    assert "if (lane == 0) atomicMax(tile_top, processed);" in cuda


def _record_tile_tops(monkeypatch):
    """Force the production tile path; collect every launch's tile_top."""
    import cupy as cp

    from woof.core import myjpbl

    tops = []
    real = myjpbl.qnse_pbl_step

    def tiled(*args, **kwargs):
        kwargs["native_tile"] = True
        out = real(*args, **kwargs)
        tops.append(int(cp.asnumpy(kwargs["mf_state"]["_tile_top"])[0]))
        return out

    monkeypatch.setattr(myjpbl, "qnse_pbl_step", tiled)
    return tops


@requires_gpu
@pytest.mark.gpu
def test_the_tile_reduction_posts_the_deepest_column(monkeypatch):
    """The kernels translate, launch, and the cross-lane reduction is the
    maximum of the one-column answers."""
    import test_qnse_wrf461_parity as parity

    tops = _record_tile_tops(monkeypatch)
    fx = parity.load("mf1")
    ncol = int(fx["meta_ncol"])
    parity._run(fx, 1, parity._state_from_oracle(fx, 1))
    (whole,) = tops
    for i in range(ncol):
        one = {k: (v if k.startswith("meta_") else v[..., i:i + 1])
               for k, v in fx.items()}
        one["meta_ncol"] = np.array(1)
        parity._run(one, 1, parity._state_from_oracle(one, 1))
    single = tops[1:]
    assert len(single) == ncol
    assert whole == max(single) > 0
