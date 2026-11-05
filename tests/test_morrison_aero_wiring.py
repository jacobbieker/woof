"""mp_physics=40 rides the six-mass (moist:qv,qc,qr,qi,qs,qg) tables of
mp_physics=10: PBL ice tendencies, legacy-RRTMG ice flags and graupel."""
from __future__ import annotations

import dataclasses

from woof.config import RunConfig
from woof.core import physics_inventory as pi
from woof.core import preflight


def _cfg(**kw):
    return dataclasses.replace(
        RunConfig(nx=8, ny=8, nz=10, dx=3000.0, dy=3000.0, ztop=20000.0, dt=10.0,
                  run_seconds=60.0, moist=True),
        mp_physics=40, **kw)


def test_the_pbl_ice_tendency_is_carried_and_priced():
    assert 40 in pi.PBL_RQI_MICROPHYSICS
    cfg = _cfg(bl_pbl_physics=1, bldt=5.0)
    assert "dqi" in pi.pbl_raw_rate_names(cfg)
    shapes = preflight.physics_array_shapes(cfg)
    assert any(name.endswith("/rqi") for name in shapes), sorted(shapes)


def test_mp10_and_mp40_share_the_six_mass_answers():
    from woof.core.rrtmg_legacy import _LEGACY_ICE_ACTIVE_MICROPHYSICS
    from woof.lightning_contract import GRAUPEL_PROVIDERS
    for table in (pi.PBL_RQI_MICROPHYSICS, _LEGACY_ICE_ACTIVE_MICROPHYSICS,
                  GRAUPEL_PROVIDERS, pi.REFL_10CM_MICROPHYSICS):
        assert (10 in table) and (40 in table)
