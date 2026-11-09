"""The WRF-Hydro and NEMO opt-ins compile with IEEE subnormals kept.

CuPy appends ``-ftz=true`` after a RawModule's own options, so a unit that
asked for ``--ftz=false`` used to flush binary32 subnormals anyway, while
the gfortran oracles these units are graded against keep them.  Each opt-in
compiles through its ``_compile.raw_module``; this halves the smallest
normal float32 on device and wants the subnormal 2**-127, not zero.  The
WRF-Hydro gw2d fixture carries a real subnormal (its ebot), graded in
``tests/test_wrf_hydro_gw2d_parity.py``.  Each case skips, naming the
opt-in rule, unless its opt-in is enabled.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof import optin

SOURCE = 'extern "C" __global__ void halve(float* a) { a[1] = __fmul_rn(a[0], 0.5f); }'


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", ["wrf_hydro", "nemo"])
def test_optin_units_keep_subnormals(name):
    reason = optin.unavailable_reason(name)
    if reason is not None:
        pytest.skip(f"opt-in rule: {reason}")
    import cupy as cp

    mod = optin.load(name)
    compile_unit = __import__(mod.__name__ + "._compile", fromlist=["raw_module"]).raw_module
    unit = compile_unit(SOURCE, ("--fmad=false", "--ftz=true"))
    a = cp.asarray(np.array([np.finfo(np.float32).tiny, 0.0], np.float32))
    unit.get_function("halve")((1,), (1,), (a,))
    assert a.get().view(np.uint32)[1] == 0x00400000      # 2**-127, not a flushed zero
