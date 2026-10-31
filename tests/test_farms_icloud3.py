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


@pytest.mark.parametrize("solar", [0, 1])
def test_farms_reads_cal_cldfra3(solar):
    from tilestream.physics_inventory import default_builder
    from woof.core import dycore
    from woof.core.cldfra3 import cal_cldfra3, cldfra3_gridkm

    cfg = _cfg(solar_diagnostics=solar)
    state, drv = default_builder(
        cfg, 4242, start_time=datetime(2024, 6, 21, 17, tzinfo=timezone.utc))
    # moisten a layer so cal_cldfra3 has partial cloud to fill
    state.qv[3:6] *= cp.float32(1.5)
    farms_cloud = drv.farms.cloud_fraction
    seen = []

    def spy(atmosphere, fields, *, itimestep):
        cf, qc, qi = farms_cloud(atmosphere, fields, itimestep=itimestep)
        nz = atmosphere["qv"].shape[0]
        tr = lambda a: cp.ascontiguousarray(cp.ascontiguousarray(a).reshape(nz, -1).T)
        c3, q3c, q3i = cal_cldfra3(
            tr(atmosphere["qv"]), tr(atmosphere["qc"]), tr(atmosphere["qi"]),
            tr(atmosphere["qs"]), tr(atmosphere["dz"]), tr(atmosphere["pressure"]),
            tr(atmosphere["temperature"]), fields["xland"].reshape(-1),
            cldfra3_gridkm(cfg.dx, cfg.dy))
        for got, want in ((cf, c3), (qc, q3c), (qi, q3i)):
            assert np.array_equal(cp.asnumpy(tr(got)).view(np.uint32),
                                  cp.asnumpy(want).view(np.uint32))
        seen.append((cp.asnumpy(cf), cp.asnumpy(qc)))
        return cf, qc, qi

    drv.farms.cloud_fraction = spy
    for _ in range(NSTEPS):
        before = drv.call_counts.get("radiation", 0)
        qc_before = cp.asnumpy(state.qc).copy()
        dycore.step(state, cfg)
        cf, qc_rad = seen[-1]
        assert ((cf > 0) & (cf < 1)).any()
        # the made-up condensate is FARMS's, not the state's
        assert (qc_rad != qc_before).any()
        if drv.call_counts.get("radiation", 0) > before:
            np.testing.assert_array_equal(
                cp.asnumpy(drv.cldfra).view(np.uint32), cf.view(np.uint32))
        if solar:
            np.testing.assert_array_equal(
                cp.asnumpy(state.existing_scratch("solar_step_cldfra")).view(np.uint32),
                cf.view(np.uint32))
    for name in ("swdown2", "swddni2", "swddif2"):
        assert np.isfinite(cp.asnumpy(drv.fields[name])).all()
