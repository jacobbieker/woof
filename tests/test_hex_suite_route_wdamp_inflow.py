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
    assert cuda_w_damping.attach_v842(dycore, None) is None
    assert not hasattr(dycore, "w_damping")


def test_w_damping_refuses_a_dycore_that_does_not_carry_it():
    with pytest.raises(ValueError, match="never applied"):
        cuda_w_damping.attach_v842(SimpleNamespace(), 1.0)


def test_w_damping_attaches_to_the_resident_driver():
    dycore = CudaDryDycoreDriver.__new__(CudaDryDycoreDriver)
    receipt = cuda_w_damping.attach_v842(dycore, 1.2)
    assert isinstance(dycore.w_damping, cuda_w_damping.WDamping)
    assert dycore.w_damping.w_crit_cfl == 1.2
    assert receipt["w_crit_cfl"] == 1.2 and receipt["w_alpha"] == cuda_w_damping.W_ALPHA


def test_inflow_off_attaches_nothing():
    assert inflow_perturbation.attach_v842(SimpleNamespace(), None, pbl_height=None, n_cells=0,
                                           pbl_scheme="bl_mynn") is None


def test_inflow_refuses_a_global_mesh():
    with pytest.raises(ValueError, match="this mesh is global"):
        inflow_perturbation.attach_v842(SimpleNamespace(regional_v841=None), {"seed": 0}, pbl_height=None,
                                        n_cells=0, pbl_scheme="bl_mynn")


def test_inflow_refuses_a_partitioned_regional_run():
    runtime = SimpleNamespace(padded=SimpleNamespace(partition_layout_v841=object()))
    with pytest.raises(ValueError, match="one card"):
        inflow_perturbation.attach_v842(SimpleNamespace(regional_v841=runtime), {"seed": 0}, pbl_height=None,
                                        n_cells=0, pbl_scheme="bl_mynn")


@pytest.mark.parametrize("scheme", [None, "off", "'off'", "none", ""])
def test_inflow_refuses_a_suite_without_a_pbl_scheme(scheme):
    runtime = SimpleNamespace(padded=SimpleNamespace(), masks_host=SimpleNamespace(bdy_mask_cell=np.zeros(4)))
    assert inflow_perturbation.pblh_from_suite(lambda: np.zeros(4), runtime, 4, pbl_scheme=scheme) is None
    with pytest.raises(ValueError, match="config_pbl_scheme is off"):
        inflow_perturbation.attach_v842(SimpleNamespace(regional_v841=runtime), {"seed": 0},
                                        pbl_height=lambda: np.zeros(4), n_cells=4, pbl_scheme=scheme)


def test_the_door_resolves_both_flags_on_the_suite_route():
    source = inspect.getsource(door._run_forecast)
    block = source[source.index("if native_config is not None:\n            if getattr(arguments, \"obs_nudging\""):]
    block = block[:block.index("ported_route_refusal")]
    assert "arguments.w_damping_crit_cfl = _resolve_w_damping(arguments)" in block
    assert "arguments.inflow_perturbation_options = _resolve_inflow_perturbation(arguments)" in block
    assert "--w-damping is carried by" not in block and "--inflow-perturbation is carried by" not in block


def test_the_suite_runtime_attaches_both_and_refuses_moving_or_split_runs():
    from woof.hex.mpasphys import runtime_v842
    source = inspect.getsource(runtime_v842._run_native_dynamics)
    assert "getattr(arguments, 'w_damping_crit_cfl', None)" in source
    assert "getattr(arguments, 'inflow_perturbation_options', None)" in source
    assert "attach_w_damping(dycore, w_crit_cfl)" in source
    assert "pbl_height=lambda: session.field('diag_physics', 'hpbl')" in source
    assert "if moves or rank_execution:" in source


def test_the_help_names_both_routes():
    import argparse

    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    helps = {action.dest: action.help for action in parser._actions}
    for dest in ("w_damping", "inflow_perturbation"):
        assert "WOOF-physics route only" not in helps[dest] and "both routes" in helps[dest]
