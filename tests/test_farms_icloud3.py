"""swint_opt = 2 (FARMS) with icloud = 3 (WRF-Solar cal_cldfra3).

In WRF v4.7.1 the combination is valid: the radiation driver's Solar_step
runs cal_cldfra3 on every step (module_radiation_driver.F:1462-1510) and
Farms_driver (:3064-3068) reads its CLDFRA and the qc/qi it rewrote, before
the driver restores the prognostic qc/qi (:3137-3155).  cal_cldfra3 itself is
graded word for word by tests/test_cldfra3_wrf471_oracle.py and FARMS by
tests/test_farms_wrf471_parity.py; this file holds the composition:

* on a radiation step the CLDFRA FARMS reads is the radiation call's own;
* every step, FARMS's CLDFRA and qc/qi are cal_cldfra3's of the step's state;
* with solar_diagnostics on, its per-step Solar_step cloud is FARMS's;
* the prognostic state is untouched by the made-up condensate.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

NSTEPS = 6


def _cfg(**extra):
    from tilestream import harness
    from woof.config import validate_run_config
    return validate_run_config(harness.make_config(
        16, 12, 20, moist=True, mp_physics=8, ra_physics=4,
        ra_rrtmg_variant="rrtmg_legacy", dt=1.0, time_step_sound=4,
        ztop=20000.0, radt_minutes=0.05, swint_opt=2, icloud=3,
        bl_pbl_physics=1, sf_sfclay_physics=91, sf_surface_physics=2, **extra))


def test_config_admits_farms_with_icloud3():
    cfg = _cfg()
    assert cfg.icloud == 3 and cfg.swint_opt == 2

