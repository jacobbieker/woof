"""Small-timestep rounding of the carried large-step state, and its remedy.

At dt 0.5/0.25 s each timestep's theta and momentum increment is only a few
binary32 units in the last place of the carried ``rtheta_p``/``ru``, so a
weak tendency is partly rounded away every step.  The probe measures that
through the CPU recovery authority; the compensated wrapper carries the exact
TwoSum residual across timesteps and is gated to the sub-anchor lane.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.hex.dt_admission import ADMITTED_TIMESTEPS
from woof.hex.integration import (
    RecoveryBackground,
    RecoveryResidual,
    RecoveryState,
    compensated_recovery_required,
    recover_large_step_variables,
    recover_large_step_variables_compensated,
)
from woof.hex.precision_probe import (
    SyntheticPatchMesh,
    large_step_increment_loss,
    synthetic_hex_patch,
)


def _case(dtype, *, seed=7):
    patch = synthetic_hex_patch(100.0, rings=1)
    coe = patch.arrays["cellsOnEdge"].copy()
    coe[:, 1] = np.where(coe[:, 1] < 0, coe[:, 0], coe[:, 1])
    mesh = SyntheticPatchMesh(arrays=dict(patch.arrays, cellsOnEdge=coe), attrs=patch.attrs)
    nlev = 4
    ncells = patch.arrays["nEdgesOnCell"].size
    nedges = coe.shape[0]
    rng = np.random.default_rng(seed)

    def cell(scale, offset=0.0):
        return (offset + scale * rng.standard_normal((nlev, ncells))).astype(dtype)

    def edge(scale):
        return (scale * rng.standard_normal((nlev, nedges))).astype(dtype)

    def iface(scale):
        out = (scale * rng.standard_normal((nlev + 1, ncells))).astype(dtype)
        out[0] = 0.0
        out[-1] = 0.0
        return out

    rho_base = cell(0.0, 1.0)
    state = RecoveryState(
        ww_avg=iface(0.0), rw_save=iface(0.3), w=iface(0.0), rw=iface(0.0),
        rw_p=iface(1.0e-5), rtheta_p=cell(0.0), rtheta_pp=cell(3.0e-5),
        rtheta_p_save=cell(20.0), rho_p=cell(0.0), rho_p_save=cell(0.01),
        rho_pp=cell(1.0e-7), rho_zz=cell(0.0, 1.0), ru_avg=edge(0.0),
        ru_save=edge(12.0), ru_p=edge(3.0e-5), u=edge(0.0), ru=edge(0.0),
        exner=cell(0.0, 1.0), pressure_p=cell(0.0), theta_m=cell(0.0, 300.0),
    )
    background = RecoveryBackground(
        rho_base=rho_base,
        exner_base=cell(0.0, 0.9),
        rtheta_base=cell(0.0, 300.0),
        zz=cell(0.0, 1.0),
    )
    fz = np.full(nlev + 1, 0.5, dtype=dtype)
    zb = np.zeros((nlev + 1, ncells, 6), dtype=dtype)
    kwargs = dict(
        dt=0.25, acoustic_steps=2, rk_step=3,
        rt_diabatic_tendency=cell(1.0e-4), fzm=fz, fzp=fz,
        cf1=dtype(1.0), cf2=dtype(0.0), cf3=dtype(0.0), zb_cell=zb, zb3_cell=zb,
        boundary_mask_cell=np.full(ncells, 99, dtype=np.int32),
    )
    return mesh, state, background, kwargs


def test_the_gate_is_the_anchor_table_floor():
    floor = min(anchor.dt_seconds for anchor in ADMITTED_TIMESTEPS.values())
    assert compensated_recovery_required(floor * 0.999)
    assert compensated_recovery_required(min(0.25, floor * 0.5))
    assert not compensated_recovery_required(floor)
    assert not compensated_recovery_required(max(20.0, floor * 4.0))


def test_zero_residual_reproduces_the_authority_bit_for_bit():
    """The wrapper adds an exact zero to every perturbation, so with no
    carried residual the recovered state is the authority's own."""

    mesh, state, background, kwargs = _case(np.float32)
    plain = recover_large_step_variables(mesh, state, background, **kwargs)
    comp, _ = recover_large_step_variables_compensated(
        mesh, state, background, RecoveryResidual.zeros_like(state),
        final_stage=True, **kwargs,
    )
    for name in RecoveryState.__slots__:
        assert np.asarray(getattr(comp, name)).tobytes() == np.asarray(getattr(plain, name)).tobytes(), name


def test_the_residual_is_the_exact_rounding_error_including_stage_three():
    mesh, state, background, kwargs = _case(np.float32)
    out, residual = recover_large_step_variables_compensated(
        mesh, state, background, RecoveryResidual.zeros_like(state),
        final_stage=True, **kwargs,
    )
    f64 = np.float64
    c = ((np.float32(kwargs["dt"]) * out.rho_zz) * kwargs["rt_diabatic_tendency"]).astype(f64)
    exact = {
        "rho_p": state.rho_p_save.astype(f64) + state.rho_pp.astype(f64),
        "rtheta_p": state.rtheta_p_save.astype(f64) + state.rtheta_pp.astype(f64) - c,
        "ru": state.ru_save.astype(f64) + state.ru_p.astype(f64),
    }
    for name, value in exact.items():
        recovered = np.asarray(getattr(out, name), dtype=f64)
        carried = np.asarray(getattr(residual, name), dtype=f64)
        # recovered + residual holds the exact sum to binary32-of-residual.
        gap = np.abs((recovered + carried) - value)
        assert np.all(gap <= np.abs(np.spacing(np.float32(np.abs(carried) + 1e-30))) + 1e-12 * np.abs(value)), name
        assert np.any(carried != 0.0), name
    rw_exact = state.rw_save.astype(f64)[1:-1] + state.rw_p.astype(f64)[1:-1]
    rw_gap = np.abs(out.rw.astype(f64)[1:-1] + residual.rw.astype(f64)[1:-1] - rw_exact)
    assert np.all(rw_gap <= 1e-12)
    assert np.all(residual.rw[0] == 0.0) and np.all(residual.rw[-1] == 0.0)


def test_the_caller_perturbations_come_back_unfolded():
    mesh, state, background, kwargs = _case(np.float32)
    carried = RecoveryResidual.zeros_like(state)
    carried.rtheta_p[...] = np.float32(1.0e-6)
    carried.ru[...] = np.float32(1.0e-6)
    out, _ = recover_large_step_variables_compensated(
        mesh, state, background, carried, final_stage=True, **kwargs
    )
    for name in ("rho_pp", "rtheta_pp", "ru_p", "rw_p"):
        assert out.__getattribute__(name).tobytes() == getattr(state, name).tobytes(), name


def test_the_fold_error_is_carried_near_a_zero_crossing():
    """saved ~ 0 and |pert| ~ |saved|: folding the residual into the
    perturbation rounds at the residual's own size, and that error must be
    carried too, so saved + pert + residual_in == recovered + residual_out
    to within binary32-of-the-residual."""

    mesh, state, background, kwargs = _case(np.float32)
    state.ru_save[...] = np.float32(1.0e-3)
    state.ru_p[...] = np.float32(-0.9e-3)
    carried = RecoveryResidual.zeros_like(state)
    carried.ru[...] = np.float32(3.3e-11)
    out, after = recover_large_step_variables_compensated(
        mesh, state, background, carried, final_stage=True, **kwargs
    )
    f64 = np.float64
    exact = state.ru_save.astype(f64) + state.ru_p.astype(f64) + carried.ru.astype(f64)
    held = out.ru.astype(f64) + after.ru.astype(f64)
    assert np.max(np.abs(held - exact)) <= 1.0e-17


def test_intermediate_stages_leave_the_residual_alone():
    mesh, state, background, kwargs = _case(np.float32)
    carried = RecoveryResidual.zeros_like(state)
    carried.rtheta_p[...] = np.float32(1.0e-6)
    _, after = recover_large_step_variables_compensated(
        mesh, state, background, carried, final_stage=False, **dict(kwargs, rk_step=1)
    )
    assert after is carried


def test_binary64_is_a_no_op_wrapper():
    mesh, state, background, kwargs = _case(np.float64)
    plain = recover_large_step_variables(mesh, state, background, **kwargs)
    comp, residual = recover_large_step_variables_compensated(
        mesh, state, background, RecoveryResidual.zeros_like(state),
        final_stage=True, **kwargs,
    )
    for name in RecoveryState.__slots__:
        assert np.asarray(getattr(comp, name)).tobytes() == np.asarray(getattr(plain, name)).tobytes()
    assert not np.any(residual.rtheta_p)


def test_a_mismatched_residual_is_refused():
    mesh, state, background, kwargs = _case(np.float32)
    residual = RecoveryResidual.zeros_like(state)
    residual.ru = residual.ru.astype(np.float64)
    with pytest.raises(ValueError, match="recovery residual ru"):
        recover_large_step_variables_compensated(
            mesh, state, background, residual, final_stage=True, **kwargs
        )


@pytest.mark.parametrize("dt", [0.5, 0.25])
def test_small_timesteps_lose_increment_and_compensation_restores_it(dt):
    plain = large_step_increment_loss(dt, simulated_seconds=30.0)
    comp = large_step_increment_loss(dt, simulated_seconds=30.0, compensated=True)
    coarse = large_step_increment_loss(5.0, simulated_seconds=30.0)
    # float64 holds the intended increment.
    assert abs(plain["float64"]["theta_lost_fraction"]) < 1.0e-8
    # float32 without compensation: percent-level systematic theta loss and
    # RMS error well above what the 5 s step carries.
    assert plain["float32"]["theta_lost_fraction"] > 1.0e-2
    assert plain["float32"]["theta_rms_relative_error"] > 2.0 * coarse["float32"]["theta_rms_relative_error"]
    # Compensated: no worse than the 5 s step's representation floor.
    for key in ("theta_rms_relative_error", "u_rms_relative_error"):
        assert comp["float32"][key] <= 1.05 * coarse["float32"][key] + 1.0e-6, key
    assert abs(comp["float32"]["u_lost_fraction"]) < 1.0e-3
