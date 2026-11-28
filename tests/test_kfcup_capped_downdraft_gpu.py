"""KF-CuP's capped-downdraft rain flux, corrected and exact (correct-mode FQ11).

When KF_cup_PARA caps the downdraft (DDINC = TRPPT/TDER), binary32
TDER*DDINC can round one ulp above TRPPT, so WRF's bare TRPPT-TDER
(module_cu_kfcup.F:2711-2721) is about -2^-15 and RAINCV is negative for the
whole NCA window; a hex forecast's RAINC then accumulates below zero.
RunConfig.kf_capped_downdraft_rain_floor (on in correct mode) compiles
kfcup.cu with KF_CAPPED_DOWNDRAFT_RAIN_FLOOR, which floors the flux at zero
(the float64 answer there); exact mode keeps WRF's bare subtraction.

The fixture is one 149-level column of a hex forecast whose call took the
capped branch and published RAINCV = -2.3e-10 mm under the default
arithmetic before any floor
(tests/fixtures/kfcup/kfcup-capped-downdraft-149.npz; model state only).

WRF's own Fortran on this column (KF_cup_CPS from WRF 4.7.1's
module_cu_kfcup.F, built as WRF builds it: gfortran -O2 -fno-fast-math
-ffp-contract=off) gives TDER = 4605.73047 (0x458FEDD8), DDINC = 0.256296217
(0x3E833942), TRPPT = 307.795532 (0x4399E5D4); TDER*DDINC rounds to TRPPT
exactly, so PPTFLX = 0 and RAINCV = 0 (0x00000000), NCA = 750.  The port's
strict arithmetic (GPUWM_WRF_EXACT=1) without the floor gives the same 0;
the negative word is the default arithmetic's (contracted, CUDA libm).
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).resolve().parent / "fixtures/kfcup/kfcup-capped-downdraft-149.npz"

#: RAINCV of the exact (unfloored) column: WRF's own word, which the strict
#: arithmetic reproduces, and each backend's default-arithmetic negative word.
EXACT_WRF_RAINCV = 0x00000000
EXACT_DEFAULT_RAINCV = {"cuda": 0xAF7E486D,     # -2.3126896e-10 mm
                        "mlx": 0xAF7E2EFC}      # Metal, -2.3117858e-10 mm

pytestmark = pytest.mark.skipif(os.environ.get("GPUWM_NO_LOCAL_GPU") == "1",
                                reason="runs the KF-CuP kernel")


def _run(rain_floor: bool):
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
                         flags=int(z["flags"]), workspace=workspace, status=status,
                         rain_floor=rain_floor, **params)
    return nz, cp.asnumpy(buffers["held"]), int(cp.asnumpy(status)[0])


def test_the_corrected_column_publishes_zero_not_negative_rain():
    nz, held, status = _run(rain_floor=True)
    assert nz == 149 and status == 0
    nca, raincv = float(held[0, 0]), held[1, 0]
    assert nca == 750.0, "the column must convect for the capped branch to be exercised"
    assert int(raincv.view(np.uint32)) == 0, raincv


def test_the_exact_column_is_wrfs_bare_subtraction():
    """Without the define the kernel is WRF's TRPPT-TDER: in a strict process
    WRF's own 0 word, in a default one the negative word the floor exists for."""
    from woof import backend
    from woof.wrf_exact import ENABLED
    nz, held, status = _run(rain_floor=False)
    assert nz == 149 and status == 0
    assert float(held[0, 0]) == 750.0
    word = int(held[1, 0].view(np.uint32))
    if ENABLED:
        assert word == EXACT_WRF_RAINCV, hex(word)
        return
    name = backend.active()
    if name not in EXACT_DEFAULT_RAINCV:
        pytest.skip(f"no recorded default-arithmetic word for backend {name!r}")
    assert word == EXACT_DEFAULT_RAINCV[name], hex(word)
