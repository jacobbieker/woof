"""The compensated large-step recovery attachment, without a device.

The GPU half (parity with the CPU authority, the stepping comparison) is
``test_cuda_compensated_recovery_gpu.py``.  These gates hold the parts that
decide WHETHER the lane runs and that it leaves the minted bytes alone: the
gate is the anchor table's floor, anchored runs rebind nothing, the module is
outside the regional kernel set and pinned as an execution source, and the
forecast stack attaches and labels it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _layout import PACKAGE_DIR

from woof.hex.cuda_backend.regional_admission import (
    MINTED_KERNEL_SET_SHA256,
    REGIONAL_KERNEL_SOURCES,
    kernel_set_sha256,
)
from woof.hex.cuda_compensated_recovery import (
    CARRIED_SUMS,
    RECOVERY_MODE_COMPENSATED,
    RECOVERY_MODE_KEY,
    attach_compensated_recovery,
    compensated_recovery_floor_seconds,
    detach_compensated_recovery,
    stamp_large_step_recovery_history,
)
from woof.hex.dt_admission import ADMITTED_TIMESTEPS
from woof.hex.errors import ConfigurationRefusal
from woof.hex.integration import COMPENSATED_RECOVERY_FIELDS


class _Driver:
    def __init__(self, dt: float) -> None:
        self.config = SimpleNamespace(config_dt=dt)
        self.cache = None
        self.calls = 0

    def _recover_candidate(self, *args):
        self.calls += 1
        return args


def test_the_kernel_set_digest_is_the_minted_one():
    """Adding the lane moved no regional kernel-set byte."""

    assert kernel_set_sha256() == MINTED_KERNEL_SET_SHA256
    assert not any("compensated" in name for name in REGIONAL_KERNEL_SOURCES)


@pytest.mark.parametrize(
    "dt", sorted({float(anchor.dt_seconds) for anchor in ADMITTED_TIMESTEPS.values()})
)
def test_every_anchored_timestep_rebinds_nothing(dt):
    driver = _Driver(dt)
    assert attach_compensated_recovery(driver) is None
    assert "_recover_candidate" not in vars(driver)
    assert not hasattr(driver, "large_step_recovery")


def test_the_floor_itself_is_anchored_and_stays_uncompensated():
    floor = compensated_recovery_floor_seconds()
    assert floor == min(float(a.dt_seconds) for a in ADMITTED_TIMESTEPS.values())
    assert attach_compensated_recovery(_Driver(floor)) is None


def test_a_sub_anchor_timestep_wraps_the_recovery_and_detach_restores_it():
    driver = _Driver(0.25)
    attachment = attach_compensated_recovery(driver)
    assert attachment is not None
    assert driver.large_step_recovery is attachment
    assert driver._recover_candidate == attachment.recover_candidate
    with pytest.raises(ConfigurationRefusal):
        attach_compensated_recovery(driver)
    detach_compensated_recovery(attachment)
    assert "_recover_candidate" not in vars(driver)
    assert driver.large_step_recovery is None


def test_a_stage_before_any_residual_passes_the_perturbations_through_untouched():
    """No residual is carried yet, so nothing launches and the stock
    recovery receives the caller's own acoustic object."""

    driver = _Driver(0.5)
    attachment = attach_compensated_recovery(driver)
    acoustic = SimpleNamespace(rho_pp=1, rtheta_pp=2, ru_p=3, rw_p=4)
    out = driver._recover_candidate("state", "diag", acoustic, 1, 2, "scalars")
    assert out == ("state", "diag", acoustic, 1, 2, "scalars")
    assert attachment.stage_recoveries == 1 and attachment.residual_updates == 0


def test_the_carried_sums_are_the_cpu_authoritys():
    assert [name for name, *_ in CARRIED_SUMS] == [
        recovered for _saved, _pert, recovered in COMPENSATED_RECOVERY_FIELDS
    ]
    assert [pert for _n, _o, _a, pert, _ro, _ra in CARRIED_SUMS] == [
        pert for _saved, pert, _recovered in COMPENSATED_RECOVERY_FIELDS
    ]


def test_the_history_label(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    path = tmp_path / "history.nc"
    with netcdf4.Dataset(path, "w") as dataset:
        dataset.createDimension("nCells", 1)
    attachment = attach_compensated_recovery(_Driver(0.25))
    stamp_large_step_recovery_history(path, attachment)
    with netcdf4.Dataset(path) as dataset:
        assert dataset.getncattr(RECOVERY_MODE_KEY) == RECOVERY_MODE_COMPENSATED
        assert "experimental" in dataset.getncattr(RECOVERY_MODE_KEY + "_reason")


def test_the_module_is_a_pinned_execution_source():
    import importlib.util
    import sys

    path = PACKAGE_DIR / "drivers" / "run_cuda_v841_full_physics_x4.py"
    name = "_test_compensated_recovery_runner"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        assert "src/hexcore/cuda_compensated_recovery.py" in module.EXECUTION_SOURCE_PINS
        module.require_frozen_execution_sources()
    finally:
        sys.modules.pop(name, None)


def test_the_device_stack_attaches_it_and_the_forecast_labels_it():
    stack_source = (PACKAGE_DIR / "drivers" / "run_cuda_v841_full_physics_x4.py").read_text(
        encoding="utf-8"
    )
    assert "recovery_attachment = attach_compensated_recovery(driver)" in stack_source
    assert '{"large_step_recovery": recovery_attachment}' in stack_source
    forecast = (PACKAGE_DIR / "drivers" / "run_cuda_v841_forecast.py").read_text(
        encoding="utf-8"
    )
    assert 'recovery_attachment = stack.get("large_step_recovery")' in forecast
    assert "stamp_large_step_recovery_history(destination, recovery_attachment)" in forecast
    # The run receipt is labelled from the attachment that executed, so a
    # preflight (no stack) never claims a recovery mode.
    assert 'payload["large_step_recovery"] = forecast["large_step_recovery"]' in forecast
    assert 'provenance["large_step_recovery"]' not in forecast


def test_the_partitioned_driver_is_refused():
    driver = _Driver(0.25)
    driver.halo_exchanger_v841 = object()
    with pytest.raises(ConfigurationRefusal):
        attach_compensated_recovery(driver)
    assert "_recover_candidate" not in vars(driver)


def test_both_admitted_rk_orders_end_on_stage_three():
    for order in (2, 3):
        driver = _Driver(0.25)
        driver.config.config_time_integration_order = order
        assert attach_compensated_recovery(driver) is not None


def test_detach_refuses_while_a_later_wrapper_is_outermost():
    driver = _Driver(0.25)
    attachment = attach_compensated_recovery(driver)
    later = lambda *args: attachment.recover_candidate(*args)  # noqa: E731
    driver._recover_candidate = later  # local time stepping, attached after
    with pytest.raises(RuntimeError, match="outermost"):
        detach_compensated_recovery(attachment)
    assert driver._recover_candidate is later


class _SteppingDriver(_Driver):
    def __init__(self, dt: float) -> None:
        super().__init__(dt)
        self.atmosphere = SimpleNamespace(state=SimpleNamespace(time_seconds=0.0))
        self.steps = 0

    def _step_device_v841(self, *args, **kwargs):
        self.steps += 1
        return "stepped"


def test_a_retried_step_clears_the_residual_and_an_advancing_one_keeps_it():
    driver = _SteppingDriver(0.25)
    attachment = attach_compensated_recovery(driver)
    attachment.residual["ru"] = "carried"
    attachment.anchor["ru"] = "anchor"
    assert driver._step_device_v841() == "stepped"
    driver.atmosphere.state.time_seconds = 0.25  # committed: advanced
    driver._step_device_v841()
    assert attachment.residual and attachment.residual_resets == 0
    driver._step_device_v841()  # the same start time again: a retry
    assert not attachment.residual and not attachment.anchor
    assert attachment.residual_resets == 1 and driver.steps == 3
    detach_compensated_recovery(attachment)
    assert "_step_device_v841" not in vars(driver)
