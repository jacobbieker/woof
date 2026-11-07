"""shcu_physics = 5 (Deng) on a card: the driver seam in a short forecast.

* Deng runs on every model step after the cumulus slot;
* the driver's call IS ``launch_deng`` on the driver's own fields (the
  glue is held word for word, the carried state included);
* its coupled tendencies reach the RK step (the forecast moves with them
  and does not without them), and the history's RAINSH is Deng's field;
* a checkpoint carries every Deng field and a resumed run is bit-identical
  to the uninterrupted one;
* default off: no Deng field is allocated and nothing launches.

The doors (config, namelist, digest) are tests/test_shcu_deng_integration.py.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import validate_run_config
from test_shcu_deng_integration import _cfg

# ---------------------------------------------------------------------------
# on a card
# ---------------------------------------------------------------------------

def _make_state(cfg):
    """A capped, moist mixed layer under a warm surface: Deng's regime."""
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced

    coord = make_vertical_coord(cfg.nz)

    def theta(z):
        z = np.asarray(z)
        return 300.0 + np.where(z < 1200.0, 0.0, 1.5 + 0.0045 * (z - 1200.0))

    def qv(z):
        z = np.asarray(z)
        return np.where(z < 1200.0, 0.0145,
                        0.006 * np.exp(-(z - 1200.0) / 2500.0))

    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    return init_moist_balanced(cfg, coord, base, qv)


def _build(cfg):
    from woof.core.physics import initialize_physics

    state = _make_state(cfg)
    lat = np.full((cfg.ny, cfg.nx), 35.0, np.float32)
    lon = np.full((cfg.ny, cfg.nx), -97.0, np.float32)
    driver = initialize_physics(
        state, cfg, landmask=1.0, tsk=306.0, pblh=1200.0,
        radiation_start_time=datetime(2024, 7, 1, 18),
        radiation_latitude=lat, radiation_longitude=lon)
    if getattr(driver, "shallow_cumulus", None) is not None:
        _seed_cloud_history(driver)
    return state, driver


def _seed_cloud_history(driver):
    """A non-buoyant cloud (NBC) left by an earlier updraft near the capping
    inversion -- what Deng's own NBC leapfrog carries between steps -- so the
    dissipation code (edge evaporation, in-cloud and exposed-base diffusion,
    CTEI, drizzle) runs from the first step.  Varies by column."""
    import cupy as cp
    f = driver.fields
    nz, ny, nx = f["cldareab"].shape
    j, i = np.indices((ny, nx), dtype=np.float32)
    weight = cp.asarray(np.float32(0.3) + np.float32(0.05) * i
                        + np.float32(0.02) * j)
    for k, area in ((10, 0.2), (11, 0.6), (12, 0.4)):
        f["cldareab"][k] = cp.float32(area) * weight
        f["cldareaa"][k] = cp.float32(area) * weight
        f["cldliqb"][k] = cp.float32(6.0e-4) * weight
        f["cldliqa"][k] = cp.float32(5.0e-4) * weight
    f["clddpthb"][...] = cp.float32(600.0)
    f["capesave"][...] = cp.float32(40.0)
    f["radsave"][...] = cp.float32(400.0)


def _snapshot(state, driver):
    """Every carrier of the state and its driver, the streaming inventory's
    own list, plus every driver field by name."""
    import cupy as cp
    from tilestream.physics_inventory import carrier_inventory
    out = {key: cp.asnumpy(value)
           for key, value in carrier_inventory(state).items()}
    out.update({f"fields/{k}": cp.asnumpy(v)
                for k, v in driver.fields.items() if hasattr(v, "shape")})
    return out

