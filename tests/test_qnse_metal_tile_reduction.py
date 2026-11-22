"""QNSE's EDMF mass-flux kernel compiles and loads on every backend.

``mfshconv_column``'s first tile pass folds the deepest processed level of
every column into ``*tile_top``.  CUDA does it as a warp reduction
(``__activemask`` + ``__shfl_down_sync``) and one ``atomicMax`` per warp;
Metal has no ``__activemask``, and until the ``__WOOF_METAL__`` branch the
translation of ``kernels/qnsepbl.cu`` failed, so QNSE never ran on Metal.
The value is an integer maximum, which no order of lanes changes, so the
Metal branch posts each lane's level with ``atomicMax`` directly.  The word
grade of the scheme itself is tests/test_qnse_wrf461_parity.py on each
backend; this file holds the load and the CUDA source arm.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

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
    assert "atomicMax(tile_top, processed);" in metal
    code = re.sub(r"//[^\n]*", "", metal)
    assert "__activemask" not in code and "__shfl" not in code
    assert re.search(r"const unsigned mask = __activemask\(\);", cuda)
    assert "__shfl_down_sync(mask, processed, offset)" in cuda
    assert "if (lane == 0) atomicMax(tile_top, processed);" in cuda


@pytest.mark.gpu
@pytest.mark.parametrize("entry", ["mfshconv_column", "qnsepbl_column"])
def test_qnse_kernels_load_on_the_active_backend(entry):
    from woof.core.kernels import get_kernel

    assert get_kernel("qnsepbl", entry) is not None
