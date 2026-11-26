"""KF-CuP never publishes negative convective precipitation.

When KF_cup_PARA caps the downdraft (DDINC = TRPPT/TDER), binary32
TDER*DDINC can round one ulp above TRPPT, so WRF's bare TRPPT-TDER is about
-2^-15 and RAINCV is negative for the whole NCA window; a hex forecast's
RAINC then accumulates below zero.  kfcup.cu floors PPTFLX at zero, as kf.cu
and mskf.cu do (the float64 answer there is 0).

The fixture is one 149-level column of a hex forecast whose call took the
capped branch and published RAINCV = -2.3e-10 mm before the floor
(tests/fixtures/kfcup/kfcup-capped-downdraft-149.npz; model state only).
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).resolve().parent / "fixtures/kfcup/kfcup-capped-downdraft-149.npz"

pytestmark = pytest.mark.skipif(os.environ.get("GPUWM_NO_LOCAL_GPU") == "1",
                                reason="runs the KF-CuP kernel")


def _run():
    import cupy as cp
    from woof.core.kf import kfcup_workspace_floats, launch_kfcup_columns

    z = np.load(FIXTURE)
    params = ast.literal_eval(str(z["params"][0]))
    params.pop("shallowcu_forced_ra", None)
    buffers = {k: cp.asarray(np.ascontiguousarray(z["pre_" + k]))
               for k in ("lev_in", "sca_in", "kpbl", "held", "iheld", "lev_out")}
    nz, ncol = buffers["lev_in"].shape[1:]
    workspace = cp.empty(kfcup_workspace_floats(nz, ncol, params["numbins"]), dtype=cp.float32)
    status = cp.zeros(1, dtype=cp.int32)
    launch_kfcup_columns(buffers, dt=float(z["dt"]), dx=float(z["dx"]), ntst=int(z["ntst"]),
                         flags=int(z["flags"]), workspace=workspace, status=status, **params)
    return nz, cp.asnumpy(buffers["held"]), int(cp.asnumpy(status)[0])


def test_the_capped_downdraft_column_publishes_zero_not_negative_rain():
    nz, held, status = _run()
    assert nz == 149 and status == 0
    nca, raincv = float(held[0, 0]), float(held[1, 0])
    assert nca > 0.0, "the column must convect for the capped branch to be exercised"
    assert raincv == 0.0, raincv
