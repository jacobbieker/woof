"""WRF diffusion and advection settings on hex: the same operator passes, the rest is refused by name.

Hex integrates with the MPAS-A dycore.  A WRF dynamics setting maps only where
MPAS runs exactly that operator (v_mom_adv_order=3, mix_full_fields=true, a
zero constant K); every other value is refused with the reason the operators
differ and MPAS's own option for it.  Until this table, h_mom_adv_order,
v_mom_adv_order, v_sca_adv_order and mix_full_fields resolved as engine
settings: they moved the run onto the generic column seam and were recorded in
its receipt, and nothing read them.
"""
from __future__ import annotations

import pytest

from woof.hex import physics_suites as ps

# (overrides, knob, words of the reason, words of the alternative)
REFUSED = [
    ({"diff_opt": 0}, "diff_opt", "vector Laplacian", "--horiz-mixing off"),
    ({"diff_opt": 1}, "diff_opt", "stress", "--horiz-mixing"),
    ({"diff_opt": 2}, "diff_opt", "stress", "--horiz-mixing"),
    ({"khdif": 100.0}, "khdif", "xkhh = khdif/prandtl", "2d_fixed"),
    ({"kvdif": 1.0}, "kvdif", "Prandtl", "config_v_mom_eddy_visc2"),
    ({"c_s": 0.25}, "c_s", "Smagorinsky", "config_smagorinsky_coef"),
    ({"mix_full_fields": False}, "mix_full_fields", "t_base", "config_mix_full"),
    ({"h_mom_adv_order": 5}, "h_mom_adv_order", "vector-invariant", "config_coef_3rd_order"),
    ({"h_mom_adv_order": 3}, "h_mom_adv_order", "vector-invariant", "config_coef_3rd_order"),
    ({"h_sca_adv_order": 3}, "h_sca_adv_order", "zb3_cell", "config_coef_3rd_order"),
    ({"h_sca_adv_order": 5}, "h_sca_adv_order", "zb3_cell", "config_coef_3rd_order"),
    ({"v_sca_adv_order": 3}, "v_sca_adv_order", "same config_coef_3rd_order", "config_coef_3rd_order"),
    ({"v_mom_adv_order": 5}, "v_mom_adv_order", "no MPAS counterpart", "v_mom_adv_order=3"),
    ({"v_mom_adv_order": 2}, "v_mom_adv_order", "no MPAS counterpart", "v_mom_adv_order=3"),
]


@pytest.mark.parametrize("overrides,knob,reason,alternative", REFUSED,
                         ids=[f"{item[1]}-{i}" for i, item in enumerate(REFUSED)])
def test_a_wrf_operator_mpas_does_not_run_is_refused_by_name(overrides, knob, reason, alternative):
    with pytest.raises(ps.PhysicsSelectionRefusal) as caught:
        ps.resolve_physics("arwen_mpas_column", overrides)
    mine = [item for item in caught.value.refusals if item.knob == knob]
    assert mine, [item.line() for item in caught.value.refusals]
    assert reason in mine[0].reason
    assert alternative in mine[0].follow_up
    # the operator's own reason, not the generic "WOOF's ARW dycore reads it"
    assert "ARW dycore reads it" not in mine[0].reason


MATCHED = [
    {"v_mom_adv_order": 3},
    {"mix_full_fields": True},
    {"mix_full_fields": "true"},
    {"mix_full_fields": ".true."},
    {"khdif": 0},
    {"kvdif": 0.0},
]


@pytest.mark.parametrize("overrides", MATCHED, ids=[str(item) for item in MATCHED])
def test_the_operator_mpas_runs_passes_recorded_and_not_handed_to_the_engine(overrides):
    plain = ps.resolve_physics("arwen_mpas_column", {})
    resolved = ps.resolve_physics("arwen_mpas_column", overrides)
    # The run is the one without the setting: same seam, same engine settings.
    assert resolved.seam == plain.seam
    assert resolved.settings == plain.settings
    (key,) = overrides
    assert key not in resolved.settings


def test_every_dynamics_row_cites_both_codes():
    rows = ps.load_table()["refused_settings"]["settings"]
    for key in ("diff_opt", "khdif", "kvdif", "c_s", "mix_full_fields", "h_mom_adv_order",
                "h_sca_adv_order", "v_sca_adv_order", "v_mom_adv_order"):
        reason = rows[key]["reason"]
        assert "MPAS" in reason and ".F" in reason, key
