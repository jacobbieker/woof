"""use_theta_m = 1 and do_avgflx_em = 1 against compiled WRF v4.7.1, word for word.

tests/data/wrf471_thetam is tools/bigstep_wrf471_oracle/thetam_build.sh's
record: WRF v4.7.1's use_theta_m statements, extracted unchanged from
dyn_em/module_big_step_utilities_em.F (calc_p_rho_phi, phy_prep,
moist_physics_prep_em, moist_physics_finish_em, initialize_moist_old,
conv_t_tendf_to_moist) and dyn_em/module_initialize_real.F (real.exe's
theta -> theta_m block), with dyn_em/module_avgflx_em.F compiled whole, at
gfortran -O0 -ffp-contract=off against glibc 2.43, on the real WRF patch of
tests/data/wrf471_bigstep/state-real.npz (12 x 10 x 49).

Every conversion record (real.exe's theta_m, phy_prep's th_phy and
th_phy_m_t0, moist_physics_prep_em's h_diabatic/qv_diabatic,
moist_physics_finish_em's t_new/th_phy_m_t0/h_diabatic with and without the
clamp, conv_t_tendf_to_moist) and every upd_avgflx record (the zero_avgflx
reset, counts 1, 5 and 359) must be WRF's word in both arithmetic modes.
calc_p_rho_phi's theta_m branch (hypsometric_opt 1 and 2) and the dry
control are WRF's words under strict diagnostics (GPUWM_WRF_EXACT=1, which
selects the WRF operation order); the default EOS keeps woof's documented
spelling (woof/core/kernels/diagnostics.cu), and there the theta_m branch
and the dry branch must diagnose the same pressure.
"""
from __future__ import annotations

import hashlib
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify import thetam_oracle as oracle

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "bigstep_wrf471_oracle"


def _cases_module():
    spec = importlib.util.spec_from_file_location(
        "thetam_cases", TOOLS / "thetam_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _receipt() -> str:
    return (oracle.ORACLE_DIR / "receipt.txt").read_text()


def test_record_is_the_pinned_wrf_source_on_glibc_243():
    receipt = _receipt()
    assert "commit f52c197ed39d12e087d02c50f412d90d418f6186" in receipt
    assert "GLIBC 2.43" in receipt
    assert "-O0 -ffp-contract=off" in receipt
    spec = importlib.util.spec_from_file_location(
        "thetam_build", TOOLS / "thetam_build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    for relative, digest in build.SOURCES.items():
        assert f"{digest}  {relative}" in receipt, relative
    outputs = hashlib.sha256(
        (oracle.ORACLE_DIR / "thetam_outputs.bin").read_bytes()).hexdigest()
    assert f"{outputs}  thetam_outputs.bin" in receipt


def test_the_inputs_are_the_record_inputs_and_the_real_patch():
    """The committed npz writes exactly the inputs.bin the record read, and
    its thermodynamic fields are the bigstep oracle's real WRF patch."""
    cases = _cases_module()
    inputs = oracle.load_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        cases.write(inputs, Path(tmp))
        digest = hashlib.sha256(
            (Path(tmp) / "thetam_inputs.bin").read_bytes()).hexdigest()
    assert f"{digest}  thetam_inputs.bin" in _receipt()
    rebuilt = cases.build(ROOT / "tests" / "data" / "wrf471_bigstep" / "state-real.npz")
    assert sorted(rebuilt) == sorted(inputs)
    for name, value in rebuilt.items():
        assert np.array_equal(value.view(np.uint32), inputs[name].view(np.uint32)), name


def test_the_record_exercises_both_finish_arms_and_the_signed_zero():
    inputs = oracle.load_inputs()
    record = oracle.load_outputs()
    lim_b = np.float32(np.float32(0.07) * np.float32(oracle.DT))
    raw = record["finish_b_in_th_phy"] - record["finish_b_in_h_diabatic"]
    assert (np.abs(raw) > lim_b).any() and (np.abs(raw) <= lim_b).any()
    assert (np.signbit(inputs["ru_m"]) & (inputs["ru_m"] == 0)).any()
    # use_theta_m = 1 moves the prognostic: theta_m' is not dry theta'.
    assert not np.array_equal(record["init_t_m"], inputs["t_dry"])


_EOS = ("eos_hypso1_al", "eos_hypso1_p", "eos_hypso2_al", "eos_hypso2_p",
        "eos_dry_al", "eos_dry_p")


@pytest.mark.gpu
@requires_gpu
def test_every_theta_m_and_avgflx_word_is_compiled_wrfs():
    inputs = oracle.load_inputs()
    record = oracle.load_outputs(inputs=inputs)
    port = oracle.port_outputs(inputs, record)
    rows = oracle.compare(port, record)
    from woof.wrf_exact import DIAGNOSTICS_ENABLED
    graded = {name: row for name, row in rows.items()
              if DIAGNOSTICS_ENABLED or name not in _EOS}
    assert sum(row["words"] for row in graded.values()) >= 111720
    assert {name: row["differ"] for name, row in graded.items()
            if row["differ"]} == {}


@pytest.mark.gpu
@requires_gpu
def test_default_eos_theta_m_branch_diagnoses_the_dry_branchs_pressure():
    """moist = 0 on theta_m and moist = 1 on dry theta are the same physical
    equation of state; in woof's default spelling they agree to float32."""
    inputs = oracle.load_inputs()
    record = oracle.load_outputs(inputs=inputs)
    port = oracle.port_outputs(inputs, record)
    pb = inputs["pb"].astype(np.float64)
    moist = port["eos_hypso1_p"].astype(np.float64) + pb
    dry = port["eos_dry_p"].astype(np.float64) + pb
    assert np.max(np.abs(moist - dry) / dry) < 2e-6
    assert np.array_equal(port["eos_hypso1_al"], port["eos_dry_al"])
