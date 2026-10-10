"""The compensated large-step recovery on the GPU against its CPU authority.

``woof.hex.cuda_compensated_recovery`` wraps the production
``CudaDryDycoreDriver._recover_candidate``; these gates run that production
method and its shipped recovery kernels on the synthetic patches of
``woof.hex.precision_probe`` (through ``stock_recovery_driver``) and hold the
result to ``woof.hex.integration.recover_large_step_variables_compensated``.
"""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

from woof.hex.cuda_acoustic import CudaAcousticState
from woof.hex.cuda_backend.containers import (
    DevicePrognosticState,
    DeviceSavedDiagnostics,
    TransferStats,
)
from woof.hex.cuda_compensated_recovery import (
    MODULE_KEY,
    CompensatedRecoveryKernels,
    attach_compensated_recovery,
    compensated_recovery_floor_seconds,
    detach_compensated_recovery,
)
from woof.hex.integration import (
    RecoveryBackground,
    RecoveryResidual,
    RecoveryState,
    recover_large_step_variables_compensated,
)
from woof.hex.precision_probe import (
    SyntheticPatchMesh,
    device_large_step_increment_loss,
    large_step_increment_loss,
    stock_recovery_driver,
    synthetic_hex_patch,
)

NLEV = 6
SUB_ANCHOR_DT = 0.25


@pytest.fixture(scope="module")
def kernel_cache():
    from woof.hex.cuda_backend import KernelCache

    return KernelCache()


def _case(*, seed=11):
    """A float32 recovery case on a two-ring synthetic patch, flat terrain."""

    patch = synthetic_hex_patch(100.0, rings=2)
    coe = patch.arrays["cellsOnEdge"].copy()
    coe[:, 1] = np.where(coe[:, 1] < 0, coe[:, 0], coe[:, 1])
    mesh = SyntheticPatchMesh(arrays=dict(patch.arrays, cellsOnEdge=coe), attrs=patch.attrs)
    ncells = patch.arrays["nEdgesOnCell"].size
    nedges = coe.shape[0]
    rng = np.random.default_rng(seed)
    f = np.float32

    def cell(scale, offset=0.0):
        return (offset + scale * rng.standard_normal((NLEV, ncells))).astype(f)

    def edge(scale):
        return (scale * rng.standard_normal((NLEV, nedges))).astype(f)

    def iface(scale):
        out = (scale * rng.standard_normal((NLEV + 1, ncells))).astype(f)
        out[0] = 0.0
        out[-1] = 0.0
        return out

    state = RecoveryState(
        ww_avg=iface(0.0), rw_save=iface(0.3), w=iface(0.0), rw=iface(0.0),
        rw_p=iface(1.0e-5), rtheta_p=cell(0.0), rtheta_pp=cell(3.0e-5),
        rtheta_p_save=cell(20.0), rho_p=cell(0.0), rho_p_save=cell(0.01),
        rho_pp=cell(1.0e-7), rho_zz=cell(0.0, 1.0), ru_avg=edge(0.0),
        ru_save=edge(12.0), ru_p=edge(3.0e-5), u=edge(0.0), ru=edge(0.0),
        exner=cell(0.0, 1.0), pressure_p=cell(0.0), theta_m=cell(0.0, 300.0),
    )
    background = RecoveryBackground(
        rho_base=np.abs(cell(0.05, 1.0)),
        exner_base=cell(0.0, 0.9),
        rtheta_base=cell(0.0, 300.0),
        zz=cell(0.0, 1.0),
    )
    residual = RecoveryResidual(
        rho_p=cell(1.0e-10),
        rtheta_p=cell(1.0e-6),
        ru=edge(4.0e-7),
        rw=iface(1.0e-9),
    )
    fz = np.full(NLEV + 1, 0.5, dtype=f)
    zb = np.zeros((NLEV + 1, ncells, 6), dtype=f)
    kwargs = dict(
        dt=SUB_ANCHOR_DT, acoustic_steps=2, rk_step=3,
        rt_diabatic_tendency=np.zeros((NLEV, ncells), dtype=f), fzm=fz, fzp=fz,
        cf1=f(1.0), cf2=f(0.0), cf3=f(0.0), zb_cell=zb, zb3_cell=zb,
        boundary_mask_cell=np.zeros(ncells, dtype=np.int32),
    )
    return mesh, coe, state, background, residual, kwargs


def _device_inputs(state, background):
    zero = TransferStats(0, 0.0)
    f = np.dtype(np.float32)
    ncells = state.rho_p.shape[1]
    rho = state.rho_p_save + background.rho_base
    saved_state = DevicePrognosticState(
        cp.asarray(rho),
        cp.asarray(state.rtheta_p_save + background.rtheta_base),
        cp.asarray(state.ru_save),
        cp.asarray(state.rw_save),
        cp.zeros((0, NLEV, ncells), dtype=cp.float32),
        0.0, f, zero,
    )
    saved_diag = DeviceSavedDiagnostics(
        cp.asarray(state.theta_m), cp.asarray(state.exner),
        cp.asarray(state.rho_p_save), cp.asarray(state.rtheta_p_save),
        cp.asarray(state.pressure_p), cp.asarray(state.u), cp.asarray(state.w),
        f, zero,
    )
    acoustic = CudaAcousticState(
        ru_p=cp.asarray(state.ru_p), rw_p=cp.asarray(state.rw_p),
        rtheta_pp=cp.asarray(state.rtheta_pp),
        rtheta_pp_old=cp.zeros_like(cp.asarray(state.rtheta_pp)),
        rho_pp=cp.asarray(state.rho_pp), ru_avg=cp.asarray(state.ru_avg),
        ww_avg=cp.asarray(state.ww_avg),
    )
    return saved_state, saved_diag, acoustic


def _driver(coe, background, kwargs, kernel_cache):
    return stock_recovery_driver(
        rho_base=background.rho_base, rtheta_base=background.rtheta_base,
        exner_base=background.exner_base, zz=background.zz,
        fzm=kwargs["fzm"], fzp=kwargs["fzp"], cells_on_edge=coe,
        kernel_cache=kernel_cache,
    )


def _seed_residual(attachment, state, residual):
    """Carry ``residual`` into the attachment, recorded against the saved values."""

    for name, saved in (
        ("rho_p", state.rho_p_save), ("rtheta_p", state.rtheta_p_save),
        ("ru", state.ru_save), ("rw", state.rw_save),
    ):
        attachment.residual[name] = cp.asarray(getattr(residual, name))
        attachment.anchor[name] = cp.asarray(saved)


@pytest.mark.parametrize("stage", [1, 3])
def test_gpu_matches_the_cpu_authority_on_a_synthetic_patch(stage, kernel_cache):
    mesh, coe, state, background, residual, kwargs = _case()
    kwargs = dict(kwargs, rk_step=stage)
    cpu_out, cpu_residual = recover_large_step_variables_compensated(
        mesh, state, background, residual, final_stage=(stage == 3), **kwargs
    )
    driver = _driver(coe, background, kwargs, kernel_cache)
    attachment = attach_compensated_recovery(driver, dt_seconds=SUB_ANCHOR_DT)
    assert attachment is not None
    _seed_residual(attachment, state, residual)
    saved_state, saved_diag, acoustic = _device_inputs(state, background)
    gpu_state, gpu_diag, _fu, _fw = driver._recover_candidate(
        saved_state, saved_diag, acoustic, stage, kwargs["acoustic_steps"],
        saved_state.scalars,
    )
    # The carried sums are the same binary32 additions on both sides.
    for cpu_name, device in (
        ("rho_p", gpu_diag.density_perturbation),
        ("rtheta_p", gpu_diag.rho_theta_perturbation),
        ("ru", gpu_state.rho_u),
        ("rw", gpu_state.rho_w),
    ):
        np.testing.assert_array_equal(cp.asnumpy(device), getattr(cpu_out, cpu_name), err_msg=cpu_name)
    # Read-outs through the production kernels, within binary32 rounding.
    for cpu_name, device, rtol in (
        ("theta_m", gpu_diag.theta_m, 2.0e-7),
        ("u", gpu_diag.normal_velocity, 2.0e-7),
        ("w", gpu_diag.vertical_velocity, 4.0e-7),
    ):
        np.testing.assert_allclose(
            cp.asnumpy(device), getattr(cpu_out, cpu_name), rtol=rtol, atol=1e-30,
            err_msg=cpu_name,
        )
    if stage == 3:
        np.testing.assert_allclose(cp.asnumpy(gpu_diag.exner), cpu_out.exner, rtol=1e-6)
        np.testing.assert_allclose(
            cp.asnumpy(gpu_diag.pressure_perturbation), cpu_out.pressure_p,
            rtol=1e-5, atol=1e-3,
        )
    # The residual: rewritten (exactly, as TwoSum is exact) on the final
    # stage, untouched on an intermediate one.
    for name in ("rho_p", "rtheta_p", "ru", "rw"):
        np.testing.assert_array_equal(
            cp.asnumpy(attachment.residual[name]), getattr(cpu_residual, name), err_msg=name
        )
    assert attachment.residual_updates == (1 if stage == 3 else 0)
    # The caller's acoustic perturbations are not folded in place.
    np.testing.assert_array_equal(cp.asnumpy(acoustic.rtheta_pp), state.rtheta_pp)
    np.testing.assert_array_equal(cp.asnumpy(acoustic.ru_p), state.ru_p)


def test_a_zero_residual_is_the_stock_recovery_bit_for_bit(kernel_cache):
    mesh, coe, state, background, residual, kwargs = _case(seed=5)
    stock = _driver(coe, background, kwargs, kernel_cache)
    wrapped = _driver(coe, background, kwargs, kernel_cache)
    attach_compensated_recovery(wrapped, dt_seconds=SUB_ANCHOR_DT)
    for stage in (1, 2, 3):
        inputs = _device_inputs(state, background)
        a = stock._recover_candidate(*inputs, stage, 2, inputs[0].scalars)
        inputs = _device_inputs(state, background)
        b = wrapped._recover_candidate(*inputs, stage, 2, inputs[0].scalars)
        for owner_a, owner_b in ((a[0], b[0]), (a[1], b[1])):
            for name in owner_a.__slots__:
                left, right = getattr(owner_a, name), getattr(owner_b, name)
                if isinstance(left, cp.ndarray):
                    assert cp.asnumpy(left).tobytes() == cp.asnumpy(right).tobytes(), (stage, name)
        for left, right in ((a[2], b[2]), (a[3], b[3])):
            assert cp.asnumpy(left).tobytes() == cp.asnumpy(right).tobytes()


def test_detach_restores_the_stock_method(kernel_cache):
    _mesh, coe, _state, background, _residual, kwargs = _case()
    driver = _driver(coe, background, kwargs, kernel_cache)
    attachment = attach_compensated_recovery(driver, dt_seconds=SUB_ANCHOR_DT)
    assert "_recover_candidate" in vars(driver)
    detach_compensated_recovery(attachment)
    assert "_recover_candidate" not in vars(driver)
    assert driver.large_step_recovery is None


def test_the_fold_guard_drops_a_residual_whose_value_was_rewritten():
    kernels = CompensatedRecoveryKernels()
    pert = cp.asarray(np.full(8, 3.0e-5, dtype=np.float32))
    residual = cp.asarray(np.full(8, 1.0e-12, dtype=np.float32))
    saved = cp.asarray(np.arange(8, dtype=np.float32) + 10.0)
    anchor = saved.copy()
    anchor[[2, 5]] += np.float32(1.0)  # something rewrote these two values
    folded, error = kernels.fold(pert, residual, saved=saved, anchor=anchor)
    host = cp.asnumpy(folded)
    expected = np.float32(3.0e-5) + np.float32(1.0e-12)
    assert np.all(host[[2, 5]] == np.float32(3.0e-5))
    assert np.all(np.delete(host, [2, 5]) == expected)
    assert np.all(cp.asnumpy(error)[[2, 5]] == 0.0)


def test_the_residual_is_kept_only_where_the_recovered_value_is_the_sum():
    kernels = CompensatedRecoveryKernels()
    saved = cp.asarray(np.full((3, 4), 300.0, dtype=np.float32))
    folded = cp.asarray(np.full((3, 4), 1.0e-5, dtype=np.float32))
    recovered = saved + folded
    recovered[1, 2] = np.float32(42.0)  # a specified-zone overwrite, say
    residual, anchor = kernels.residual(saved, folded, None, recovered, rows=(0, 2))
    host = cp.asnumpy(residual)
    exact = np.float64(300.0) + np.float64(np.float32(1.0e-5))
    expected = np.float32(exact - np.float64(np.float32(300.0) + np.float32(1.0e-5)))
    assert host[1, 2] == 0.0
    assert np.all(host[2] == 0.0)  # outside the row range
    keep = np.ones((3, 4), dtype=bool)
    keep[1, 2] = False
    keep[2] = False
    assert np.all(host[keep] == expected) and expected != 0.0
    assert cp.asnumpy(anchor).tobytes() == cp.asnumpy(recovered).tobytes()


def test_the_kernels_compile_through_the_cache_under_their_own_module_key(kernel_cache):
    kernels = CompensatedRecoveryKernels(kernel_cache)
    pert = cp.zeros(4, dtype=cp.float32)
    kernels.fold(pert, pert)
    manifest = kernel_cache.compile_manifest()
    assert MODULE_KEY in str(manifest)


@pytest.mark.parametrize("dt", [0.5, 0.25])
def test_gpu_stepping_reproduces_the_cpu_probe_and_recovers_the_loss(dt, kernel_cache):
    """The GPU loop is the CPU probe's binary32 loop, recovered through the
    production method: the measured losses agree, and compensation removes
    the momentum increment loss as unit 4 measured on CPU."""

    seconds = 15.0
    results = {}
    for compensated in (False, True):
        device = device_large_step_increment_loss(
            dt, simulated_seconds=seconds, compensated=compensated,
            kernel_cache=kernel_cache,
        )["float32_device"]
        host = large_step_increment_loss(
            dt, simulated_seconds=seconds, compensated=compensated
        )["float32"]
        for key, value in host.items():
            assert device[key] == pytest.approx(value, rel=1e-6, abs=1e-9), (compensated, key)
        results[compensated] = device
    plain, comp = results[False], results[True]
    assert abs(comp["u_lost_fraction"]) < 0.1 * abs(plain["u_lost_fraction"])
    assert comp["u_rms_relative_error"] < 0.25 * plain["u_rms_relative_error"]
    assert abs(comp["theta_lost_fraction"]) < abs(plain["theta_lost_fraction"])


def test_intermediate_stages_leave_the_carried_state_unchanged(kernel_cache):
    final_only = device_large_step_increment_loss(
        0.5, simulated_seconds=3.0, compensated=True, kernel_cache=kernel_cache
    )
    all_stages = device_large_step_increment_loss(
        0.5, simulated_seconds=3.0, compensated=True, all_rk_stages=True,
        kernel_cache=kernel_cache,
    )
    for name, array in final_only["final"].items():
        assert array.tobytes() == all_stages["final"][name].tobytes(), name
    assert all_stages["attachment"]["residual_updates"] == final_only["attachment"]["residual_updates"]
    assert all_stages["attachment"]["stage_recoveries"] == 3 * final_only["attachment"]["stage_recoveries"]


def test_the_floor_is_the_anchor_table_floor():
    from woof.hex.dt_admission import ADMITTED_TIMESTEPS

    assert compensated_recovery_floor_seconds() == min(
        float(anchor.dt_seconds) for anchor in ADMITTED_TIMESTEPS.values()
    )
