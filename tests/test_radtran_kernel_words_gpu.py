"""The device build of radtran_disort.cu (CUDA or Metal) reproduces the host words.

Both device backends are held to the same committed words
(tests/data/radtran/kernel_words.npz), so CUDA and MLX agree word for word.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import cupy  # noqa: F401  (marks this module gpu; the facade on MLX)

from conftest import requires_gpu

WORDS = Path(__file__).parent / "data" / "radtran" / "kernel_words.npz"


@requires_gpu
def test_device_words_equal_host_words():
    from tools.uvspec_oracle import kernel_words as kw
    d = np.load(WORDS)
    inp = {k[3:]: d[k] for k in d.files if k.startswith("in_")}
    bad = []
    for label, nstr, flags, angles in kw.RUNS:
        f, r = kw.run(inp, label, nstr, flags, angles, "gpu")
        if not np.array_equal(f.view(np.uint64), d[f"{label}_flux"].view(np.uint64)):
            bad.append((label, "flux", float(np.max(np.abs(f - d[f"{label}_flux"])))))
        if not np.array_equal(r.view(np.uint64), d[f"{label}_rad"].view(np.uint64)):
            bad.append((label, "rad", float(np.max(np.abs(r - d[f"{label}_rad"])))))
    assert not bad, bad
