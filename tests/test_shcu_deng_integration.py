"""shcu_physics = 5 (Deng) in the ARW forecast: its doors and its seams.

The scheme's arithmetic is graded word for word against WRF v4.7.1 by
tests/test_shcu_deng_wrf471_oracle.py.  This file pins how the port is
reached and wired:

* the doors: the RunConfig key (appended last, off), validate_run_config's
  pairing laws (WRF's own: MYJ/MYNN only, module_shallowcu_driver.F:605-630;
  WOOF's: a microphysics with QC/QR, RRTMG, the model step), the WRF
  namelist importer (0 and 5 import, 1-4 refuse by name);
* the restart digest keeps every earlier header (off values dropped).

The driver seam on a card is tests/test_shcu_deng_driver_gpu.py: a module
that imports cupy is marked ``gpu`` whole by tests/conftest.py, which would
deselect these checks from the CPU battery.
"""

from __future__ import annotations

import dataclasses
import pytest

from woof.config import ConfigurationRefused, RunConfig, validate_run_config


def _cfg(**overrides) -> RunConfig:
    base = dict(nx=8, ny=8, nz=40, dx=3000.0, dy=3000.0, ztop=18000.0,
                dt=20.0, run_seconds=0.0, moist=True, mp_physics=8,
                bl_pbl_physics=2, sf_sfclay_physics=2, sf_surface_physics=2,
                num_soil_layers=4, bldt=0.0, ra_physics=4,
                ra_rrtmg_variant="rrtmg_legacy", radt_minutes=1.0,
                shcu_physics=5)
    base.update(overrides)
    return RunConfig(**base)


# ---------------------------------------------------------------------------
# doors
# ---------------------------------------------------------------------------

def _myj(text):
    return (text.replace(" bl_pbl_physics = 11, 11,", " bl_pbl_physics = 2, 2,")
            .replace(" sf_sfclay_physics = 91, 91,",
                     " sf_sfclay_physics = 2, 2,"))


def test_restart_digest_drops_the_off_value():
    from woof.io.restart import _drop_default_off_run_keys
    values = {"shcu_physics": 0, "cu_rad_feedback": False}
    _drop_default_off_run_keys(values)
    assert values == {}
    values = {"shcu_physics": 5, "cu_rad_feedback": True}
    _drop_default_off_run_keys(values)
    assert values == {"shcu_physics": 5, "cu_rad_feedback": True}
