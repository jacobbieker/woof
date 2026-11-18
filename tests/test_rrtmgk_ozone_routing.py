"""RRTMG-K (14/14) on the nested CAM-ozone carrier, like legacy RRTMG.

K's adapter retains the o33d it radiated with legacy RRTMG's contract, so a
14/14 root with o3input = 2 feeds the carrier from that field
("legacy-root") instead of evaluating CAM a second time, and a 14/14 nest
is a parent-interpolated consumer.
"""
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import RunConfig
from woof.core.cam_ozone import (ROUTING_PARENT_INTERPOLATED, cam_ozone_domain_ids,
                                 cam_ozone_setup)
from woof.experiment import experiment_from_run_config


def _k_tree():
    run = RunConfig(nx=28, ny=28, nz=12, dx=3000., dy=3000., ztop=16000.,
                    dt=3., run_seconds=30., ra_lw_physics=14, ra_sw_physics=14,
                    o3input=2)
    exp = experiment_from_run_config(run, datetime(2000, 6, 1, 12))
    child = replace(exp.root, grid_id=2, parent_id=1, parent_grid_ratio=3,
                    i_parent_start=8, j_parent_start=8,
                    run=replace(run, nx=16, ny=16, dx=1000., dy=1000., dt=1.))
    return replace(exp, domains=(exp.root, child))


def test_a_k_root_feeds_its_k_nest_from_its_own_retained_field():
    exp = _k_tree()
    assert cam_ozone_domain_ids(exp) == {1, 2}
    modes = {}
    for dc in exp.domains:
        n = dc.run.ny, dc.run.nx
        grid = SimpleNamespace(latlon_mass=lambda n=n: (np.zeros(n), np.zeros(n)))
        modes[dc.grid_id] = cam_ozone_setup(exp=exp, dc=dc, grid=grid).mode
    assert modes == {1: "legacy-root", 2: ROUTING_PARENT_INTERPOLATED}


@pytest.mark.gpu
def test_a_k_root_carrier_holds_the_ozone_k_radiated():
    import cupy as cp
    from tilestream import harness
    from woof.core.cam_ozone import CamOzoneState, attach_cam_ozone
    from woof.core.radiation_composition import retained_ozone_adapter
    from woof.core.rrtmgk import RRTMGKRadiation
    cfg = replace(_k_tree().root.run, nx=12, ny=12, moist=True, mp_physics=10,
                  radt=.2, dt=3.)
    state, driver = harness.make_physics_state(cfg, 104)
    reference, control = harness.make_physics_state(cfg, 104)
    geo = harness.make_geography(cfg)
    owner = CamOzoneState(datetime(2011, 4, 27, 18), geo.lat, geo.lon, "legacy-root")
    attach_cam_ozone(state, cfg, owner)
    adapter = retained_ozone_adapter(driver.radiation_callable)
    assert isinstance(adapter, RRTMGKRadiation)
    for elapsed, expected_calls in ((0., 1), (3., 1), (12., 2)):
        state.elapsed_seconds = reference.elapsed_seconds = elapsed
        driver.compute(state, cfg)
        control.compute(reference, cfg)
        expected = retained_ozone_adapter(control.radiation_callable)._o33d_grid
        np.testing.assert_array_equal(cp.asnumpy(driver.o3rad), expected)
        np.testing.assert_array_equal(cp.asnumpy(driver.rthratenlw),
                                      cp.asnumpy(control.rthratenlw))
        np.testing.assert_array_equal(cp.asnumpy(driver.rthratensw),
                                      cp.asnumpy(control.rthratensw))
        assert driver.call_counts["cam_ozone"] == expected_calls
