"""Noah-MP glacier columns as WRF builds them with -DWRF_HYDRO.

Under WRF_HYDRO, WATER_GLACIER (module_sf_noahmp_glacier.F, WRF 4.7.1's Noah-MP e5c0859, :2144-2146) sends
the ponded head WRF-Hydro returns straight to surface runoff,
RUNSRF = RUNSRF + sfcheadrt/DT, and ERROR_GLACIER's water-budget stop is
compiled out (:3031-3047).  The glacier port is graded, as without
WRF_HYDRO (tests/test_noahmp_glacier_gpu.py), by bitwise CUDA/CPython
parity over chained steps of a multi-regime battery; here every column also
carries a ponded head.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from test_noahmp_glacier_gpu import DT, _battery, _chain
from woof.core.noahmp_glacier_gpu import (GLACIER_OUTPUT_ARRAYS, GLACIER_OUTPUT_SCALARS,
                                          evaluate_glacier_columns,
                                          evaluate_glacier_columns_on_host)


def _heads(count):
    return (np.arange(count, dtype=np.float32) % 7) * np.float32(0.75)


def test_the_head_leaves_as_surface_runoff_on_the_host():
    inputs, count = _battery()
    base = evaluate_glacier_columns_on_host(inputs, count, DT)
    hydro = evaluate_glacier_columns_on_host(dict(inputs, sfcheadrt=_heads(count)), count, DT)
    want = (base["runsrf"] + _heads(count) / np.float32(DT)).astype(np.float32)
    moved = _heads(count) > 0
    assert (hydro["runsrf"][moved] != base["runsrf"][moved]).all()
    assert np.allclose(hydro["runsrf"], want, rtol=1e-6, atol=0)
    for name in GLACIER_OUTPUT_SCALARS:
        if name != "runsrf":
            assert (hydro[name].view(np.uint32) == base[name].view(np.uint32)).all(), name


@pytest.mark.gpu
@requires_gpu
def test_cuda_and_cpython_glacier_columns_with_a_ponded_head_are_bitwise_twins():
    inputs, count = _battery()
    dev_in = host_in = dict(inputs, sfcheadrt=_heads(count))
    for step in range(12):
        dev = evaluate_glacier_columns(dev_in, count, DT)
        host = evaluate_glacier_columns_on_host(host_in, count, DT)
        for name in GLACIER_OUTPUT_SCALARS:
            np.testing.assert_array_equal(dev[name].view(np.uint32), host[name].view(np.uint32),
                                          err_msg=f"step {step} field {name}")
        for name, _w in GLACIER_OUTPUT_ARRAYS:
            np.testing.assert_array_equal(dev[name].view(np.uint32), host[name].view(np.uint32),
                                          err_msg=f"step {step} field {name}")
        np.testing.assert_array_equal(dev["isnow"], host["isnow"])
        dev_in = dict(_chain(dev_in, dev), sfcheadrt=_heads(count))
        host_in = dict(_chain(host_in, host), sfcheadrt=_heads(count))
