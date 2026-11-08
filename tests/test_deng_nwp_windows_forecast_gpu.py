"""Deng (shcu_physics = 5) and the NWP history windows (nwp_diagnostics = 1)
in a short forecast on the active GPU backend.

Both are 1.1.6's own units (woof/core/kernels/deng.cu, nwp_windows.cu).  On
the MLX backend each is translated to Metal: deng's Fortran gotos cross
(``IF (..) GOTO 255`` / ``175 ...`` / ``GOTO 255`` / ``255 ...`` /
``GOTO 175``) and are dispatched, and nwp_windows hands a local array of
device pointers to a device function.  Neither built on Metal before; here
they launch, Deng moves the forecast, and the windows hold the step's
extremes.  Run the file on CUDA and on MLX with WOOF_LAUNCH_TRACE_DIR set
and compare the launches with tools/backend_receipts.py.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from conftest import requires_gpu
from test_shcu_deng_integration import _cfg

STEPS = 4


def _build(cfg):
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics

    coord = make_vertical_coord(cfg.nz)

    def theta(z):
        z = np.asarray(z)
        return 300.0 + np.where(z < 1200.0, 0.0, 1.5 + 0.0045 * (z - 1200.0))

    def qv(z):
        z = np.asarray(z)
        return np.where(z < 1200.0, 0.0145, 0.006 * np.exp(-(z - 1200.0) / 2500.0))

    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, qv)
    lat = np.full((cfg.ny, cfg.nx), 35.0, np.float32)
    lon = np.full((cfg.ny, cfg.nx), -97.0, np.float32)
    driver = initialize_physics(
        state, cfg, landmask=1.0, tsk=306.0, pblh=1200.0,
        radiation_start_time=datetime(2024, 7, 1, 18),
        radiation_latitude=lat, radiation_longitude=lon)
    _seed_cloud_history(driver)
    return state, driver


def _seed_cloud_history(driver):
    """A non-buoyant cloud left by an earlier updraft near the capping
    inversion (what Deng's own leapfrog carries between steps), so the
    dissipation code runs from the first step.  Varies by column."""
    import cupy as cp
    f = driver.fields
    nz, ny, nx = f["cldareab"].shape
    j, i = np.indices((ny, nx), dtype=np.float32)
    weight = cp.asarray(np.float32(0.3) + np.float32(0.05) * i + np.float32(0.02) * j)
    for k, area in ((10, 0.2), (11, 0.6), (12, 0.4)):
        f["cldareab"][k] = cp.float32(area) * weight
        f["cldareaa"][k] = cp.float32(area) * weight
        f["cldliqb"][k] = cp.float32(6.0e-4) * weight
        f["cldliqa"][k] = cp.float32(5.0e-4) * weight
    f["clddpthb"][...] = cp.float32(600.0)
    f["capesave"][...] = cp.float32(40.0)
    f["radsave"][...] = cp.float32(400.0)


@pytest.mark.gpu
@requires_gpu
def test_deng_and_the_nwp_windows_run_in_a_forecast():
    import cupy as cp
    from tilestream import harness
    from woof.config import validate_run_config
    from woof.core import nwp_diagnostics as nwp

    cfg = validate_run_config(_cfg(nwp_diagnostics=1, icloud_bl=0, open_x=True, open_y=True))
    state, driver = _build(cfg)
    harness.run_steps(state, cfg, STEPS)
    assert driver.call_counts.get("shallow_cumulus") == STEPS
    for name in ("rushten", "rthshten", "rqvshten"):
        assert bool(cp.isfinite(driver.fields[name]).all()), name
    # the scheme acts on this sounding: it is not grading zeros
    assert max(float(cp.abs(driver.fields[name]).max())
               for name in ("rthshten", "rqvshten", "rqcshten")) > 0.0
    up = cp.asnumpy(state.existing_scratch("w_up_max"))
    down = cp.asnumpy(state.existing_scratch("w_dn_max"))
    for name in nwp.PERSISTENT:
        assert np.isfinite(cp.asnumpy(state.existing_scratch(name))).all(), name
    # the window's largest updraft is no smaller than its largest downdraft
    assert (up >= down).all()
    # and they are written: not every window is still its zero
    assert any(float(np.abs(cp.asnumpy(state.existing_scratch(name))).max()) > 0.0
               for name in nwp.PERSISTENT)
