"""--w-damping and --inflow-perturbation on the MPAS suite route (matrix 4.2 / 5.7 follow-ups), host side.

The suite route (mpasphys.runtime_v842) steps the same resident dycore and regional runtime as the WOOF-physics
route, so both terms attach to the same seats (``CudaDryDycoreDriver.w_damping``, ``regional_v841.
inflow_perturbation``); only the PBLH source differs (the suite's own ``diag_physics/hpbl``).  This file holds the
seams: off builds nothing, the door resolves instead of refusing, and every place the terms cannot act refuses by
name.  The forecast evidence (off bit-identical, on applied) is the gated GPU run.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import cuda_w_damping, inflow_perturbation
from woof.hex import forecast_door as door
from woof.hex.cuda_driver import CudaDryDycoreDriver


def test_w_damping_off_attaches_nothing():
    dycore = SimpleNamespace()
    assert cuda_w_damping.attach(dycore, None) is None
    assert not hasattr(dycore, "w_damping")


def test_w_damping_refuses_a_dycore_that_does_not_carry_it():
    with pytest.raises(ValueError, match="never applied"):
        cuda_w_damping.attach(SimpleNamespace(), 1.0)


def test_w_damping_attaches_to_the_resident_driver():
    dycore = CudaDryDycoreDriver.__new__(CudaDryDycoreDriver)
    receipt = cuda_w_damping.attach(dycore, 1.2)
    assert isinstance(dycore.w_damping, cuda_w_damping.WDamping)
    assert dycore.w_damping.w_crit_cfl == 1.2
    assert receipt["w_crit_cfl"] == 1.2 and receipt["w_alpha"] == cuda_w_damping.W_ALPHA


def test_inflow_off_attaches_nothing():
    assert inflow_perturbation.attach_v842(SimpleNamespace(), None, pbl_height=None, n_cells=0,
                                           pbl_active=True) is None


def test_inflow_refuses_a_global_mesh():
    with pytest.raises(ValueError, match="this mesh is global"):
        inflow_perturbation.attach_v842(SimpleNamespace(regional_v841=None), {"seed": 0}, pbl_height=None,
                                        n_cells=0, pbl_active=True)


def test_inflow_refuses_a_partitioned_regional_run():
    runtime = SimpleNamespace(padded=SimpleNamespace(partition_layout_v841=object()))
    with pytest.raises(ValueError, match="one card"):
        inflow_perturbation.attach_v842(SimpleNamespace(regional_v841=runtime), {"seed": 0}, pbl_height=None,
                                        n_cells=0, pbl_active=True)


def _regional_runtime(n):
    return SimpleNamespace(padded=SimpleNamespace(), masks_host=SimpleNamespace(bdy_mask_cell=np.arange(n) % 3))


def test_inflow_refuses_a_suite_whose_pbl_slot_resolves_off():
    runtime = _regional_runtime(4)
    assert inflow_perturbation.pblh_from_suite(lambda: np.zeros(4), runtime, 4, pbl_active=False) is None
    with pytest.raises(ValueError, match="pbl slot resolves to off"):
        inflow_perturbation.attach_v842(SimpleNamespace(regional_v841=runtime), {"seed": 0},
                                        pbl_height=lambda: np.zeros(4), n_cells=4, pbl_active=False)


@pytest.mark.parametrize("suite, pbl, active", [("none", "suite", False), ("mesoscale_reference", "suite", True),
                                                ("convection_permitting", "bl_mynn", True),
                                                ("mesoscale_reference", "off", False)])
def test_the_pbl_slot_is_the_resolved_selector_not_the_namelist_string(suite, pbl, active):
    """THE BREAKAGE THIS PREVENTS (review of this change): reading config_pbl_scheme's raw string took 'suite'
    under config_physics_suite = 'none' for a PBL, so the seeding was accepted and never scaled by a depth."""
    from woof.hex.mpasphys.driver_v842 import resolve_namelist
    config = resolve_namelist(f"&physics\n config_physics_suite='{suite}'\n config_pbl_scheme='{pbl}'\n/\n",
                              default_backend="hex-cuda")
    assert bool(config.active("pbl")) is active


def test_the_host_hpbl_is_averaged_on_the_host_over_the_boundary_cells():
    runtime = _regional_runtime(9)
    hpbl = np.arange(10, dtype=np.float32) * 100.0          # one padding word past n_cells
    read = inflow_perturbation.pblh_from_suite(lambda: hpbl, runtime, 9, pbl_active=True)
    ring = np.arange(9) % 3
    assert read() == float(np.mean(hpbl[:9][ring >= 1].astype(np.float64)))


def _resolved(**kw):
    return SimpleNamespace(**{"w_damping_crit_cfl": None, "inflow_perturbation_options": None,
                              "samehost_devices": None, "hosts": None, **kw})


def test_the_door_admits_one_card_and_off_runs():
    assert door.suite_route_seat_refusal(_resolved(samehost_devices="0,1"), environ={}) is None
    assert door.suite_route_seat_refusal(_resolved(w_damping_crit_cfl=1.0), environ={}) is None


@pytest.mark.parametrize("where, kw, environ", [
    ("--samehost-devices", {"samehost_devices": "0,1"}, {}),
    ("--hosts", {"hosts": "a,b"}, {}),
    ("--hosts", {}, {"WOOF_HEX_HOSTS": "a,b"}),
])
def test_the_door_refuses_runs_off_the_one_card_seat_naming_every_flag(where, kw, environ):
    refusal = door.suite_route_seat_refusal(
        _resolved(w_damping_crit_cfl=1.0, inflow_perturbation_options={"seed": 0}, **kw), environ=environ)
    assert where in refusal and "--w-damping and --inflow-perturbation" in refusal and "never applied" in refusal


def test_the_door_refuses_a_moving_region(monkeypatch):
    from woof.hex import fullstep_lts_admission
    switch = fullstep_lts_admission.MOVING_LAUNCH_VARIABLES[0]
    refusal = door.suite_route_seat_refusal(_resolved(inflow_perturbation_options={"seed": 0}),
                                            environ={switch: "1"})
    assert "moving region" in refusal and switch in refusal


def test_the_old_suite_route_refusals_are_gone():
    source = inspect.getsource(door._run_forecast)
    assert "--w-damping is carried by" not in source and "--inflow-perturbation is carried by" not in source


def test_the_help_names_both_routes():
    import argparse

    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    helps = {action.dest: action.help for action in parser._actions}
    for dest in ("w_damping", "inflow_perturbation"):
        assert "WOOF-physics route only" not in helps[dest] and "both routes" in helps[dest]
