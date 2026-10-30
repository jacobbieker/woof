"""radtran_disort.cu's host build reproduces the committed output words.

tests/data/radtran/kernel_words.npz holds fixed binary64 solver inputs and
the words the host build produced (tools/uvspec_oracle/kernel_words.py).
The device builds are held to the same words in
tests/test_radtran_kernel_words_gpu.py, so CUDA and Metal agree word for
word with each other through these fixtures.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

WORDS = Path(__file__).parent / "data" / "radtran" / "kernel_words.npz"


def _have_cxx():
    return any(shutil.which(c) for c in ("clang++", "g++", "c++"))


@pytest.mark.skipif(not _have_cxx(), reason="needs a host C++ compiler")
def test_host_build_words():
    from tools.uvspec_oracle import kernel_words as kw
    d = np.load(WORDS)
    inp = {k[3:]: d[k] for k in d.files if k.startswith("in_")}
    for label, nstr, flags, angles in kw.RUNS:
        f, r = kw.run(inp, label, nstr, flags, angles, "host")
        assert np.array_equal(f.view(np.uint64), d[f"{label}_flux"].view(np.uint64)), label
        assert np.array_equal(r.view(np.uint64), d[f"{label}_rad"].view(np.uint64)), label


def test_work_size_matches_kernel():
    from woof.radtran import _host, solver
    if not _have_cxx():
        pytest.skip("needs a host C++ compiler")
    lib = _host.library()
    for nlyr in (1, 7, 60):
        for nstr in (2, 4, 8, 16, 32):
            assert lib.radtran_work_size_host(nlyr, nstr // 2, nstr) == solver.work_size(nlyr, nstr)
