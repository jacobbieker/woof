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

def test_deng_is_admitted_beside_myj_and_mynn():
    validate_run_config(_cfg())
    validate_run_config(_cfg(bl_pbl_physics=5, sf_sfclay_physics=5,
                             sf_surface_physics=2))
    # beside Kain-Fritsch, which shares WRF's one W0AVG with it
    validate_run_config(_cfg(cu_physics=1, cudt_minutes=5.0))


@pytest.mark.parametrize("overrides, fragment", [
    (dict(shcu_physics=1), "is not ported"),
    (dict(shcu_physics=4), "is not ported"),
    (dict(bl_pbl_physics=1, sf_sfclay_physics=91, sf_surface_physics=2),
     "module_shallowcu_driver.F:605-630"),
    (dict(mp_physics=0), "QC and QR"),
    (dict(ra_physics=1), "RRTMG"),
])
def test_each_pairing_law_refuses_by_name(overrides, fragment):
    with pytest.raises(ConfigurationRefused) as caught:
        validate_run_config(_cfg(**overrides))
    assert fragment in str(caught.value)


def test_the_selector_is_appended_off_and_keyword_only():
    fields = RunConfig.__dataclass_fields__
    assert fields["shcu_physics"].default == 0 and fields["shcu_physics"].kw_only
    names = [f.name for f in dataclasses.fields(RunConfig)]
    # appended after them: the output-only WRF-Solar solar_diagnostics and
    # the radtran diagnostic's radtran_aerosol
    assert names[-4:] == ["shcu_physics", "cu_rad_feedback", "solar_diagnostics", "radtran_aerosol"]


def _myj(text):
    return (text.replace(" bl_pbl_physics = 11, 11,", " bl_pbl_physics = 2, 2,")
            .replace(" sf_sfclay_physics = 91, 91,",
                     " sf_sfclay_physics = 2, 2,"))


def test_the_namelist_importer_carries_five_and_refuses_one_to_four(tmp_path):
    from test_namelist_import import _import_with
    toml, report = _import_with(
        tmp_path, extra_physics=" shcu_physics = 5,\n", inp_filter=_myj)
    fixed = {(f.section, f.key): f for f in report.fixed}
    # max_dom = 2: the omitted second domain keeps WRF's default 0.
    assert fixed[("physics", "shcu_physics")].fixed_value == [5, 0]
    assert "shcu_physics = 5" in toml and "shcu_physics = 0" in toml
    toml0, _ = _import_with(tmp_path, extra_physics=" shcu_physics = 0,\n",
                            inp_filter=_myj)
    assert "shcu_physics" not in toml0
    for value in (1, 2, 3):
        with pytest.raises(ValueError, match="shcu_physics"):
            _import_with(tmp_path, extra_physics=f" shcu_physics = {value},\n",
                         inp_filter=_myj)


def test_restart_digest_drops_the_off_value():
    from woof.io.restart import _drop_default_off_run_keys
    values = {"shcu_physics": 0, "cu_rad_feedback": False}
    _drop_default_off_run_keys(values)
    assert values == {}
    values = {"shcu_physics": 5, "cu_rad_feedback": True}
    _drop_default_off_run_keys(values)
    assert values == {"shcu_physics": 5, "cu_rad_feedback": True}
