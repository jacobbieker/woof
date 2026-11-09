"""Kain-Fritsch under a varying model step, graded against WRF v4.7.1.

WRF's per-step cumulus clock (``advance_ppt``: RAINC += PRATEC*DT, the KF
NCA expiry and NCA -= DT) runs on EVERY model step.  woof's
``PhysicsDriver._advance_cumulus_clock`` ran it only when the step ended on a
multiple of ``clock_dt``, a test meant for the compatibility integrator's
internal substeps.  With no substeps ``clock_dt`` is the live ``dt``, and an
adaptive clock's elapsed time is a sum of unequal steps, so the clock was
skipped on nearly every adaptive step: convective rain stopped accumulating
and NCA stopped counting down, so a triggered column held its tendencies
far past its convective time scale and never retriggered.  Measured on this
case before the fix: 0.26 mm of RAINC where WRF has 76 mm.

The fixture under tests/data/oracles/kf_adaptive is WRF's answer on a frozen
three-column state over 110 adaptive-style steps (30-96 s, one CFL cut):
the byte-unmodified ``phys/module_cu_kfeta.F`` with the adaptive run_param
test of ``module_cumulus_driver.F`` and ``advance_ppt``
(``module_physics_addtendc.F``) transcribed around it.
tools/kf_adaptive_wrf471_oracle builds and runs it; its README has the
recipe.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "oracles" / "kf_adaptive"
CASE = FIXTURE / "case.bin"
ORACLE = FIXTURE / "oracle-O0.out"
FIXTURE_SHA256 = {
    "case.bin":
        "19a4daf2e0b8a898cad37c50269aaeea398d8e1312b6d9013cac5daff2c71693",
    "oracle-O0.out":
        "81fc9a31ed64397f36146628ec15e33fa1795a4d084ddbf09f9b874fe41cd0a2",
}


def test_the_fixture_is_the_recorded_wrf_run():
    for name, digest in FIXTURE_SHA256.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest


def test_the_case_exercises_what_a_varying_step_changes():
    from tools.kf_adaptive_wrf471_oracle import case

    dts, words = case.read_case(CASE)
    assert dts == case.dt_sequence()
    # Almost no step ends on a multiple of its own dt: the state the old
    # phase test misread as "an internal substep".
    elapsed, aligned = 0.0, 0
    for dt in dts:
        remainder = math.fmod(elapsed + dt, dt)
        aligned += min(remainder, dt - remainder) <= 1.0e-6 * dt
        elapsed += dt
    assert aligned <= 2 and len(set(dts)) > 30
    oracle = case.read_oracle(ORACLE)
    assert not oracle["fatal"].any()
    assert int(oracle["run_param"].sum()) == 25
    # Every column convects, expires (NCA crosses zero) and retriggers.
    for c in range(oracle["nca"].shape[1]):
        nca = oracle["nca"][:, c]
        assert (oracle["rainc"][-1, c] > 10.0)
        assert (np.diff(nca) > 0).sum() >= 3
        assert (nca <= 0).any()


@pytest.mark.skipif(not os.environ.get("WRF_SOURCE_ROOT")
                    or shutil.which("gfortran") is None,
                    reason="WRF_SOURCE_ROOT (v4.7.1) and gfortran rebuild the oracle")
def test_the_fortran_rebuild_reproduces_the_fixture(tmp_path):
    out = tmp_path / "oracle.out"
    subprocess.run(
        ["bash", str(ROOT / "tools/kf_adaptive_wrf471_oracle/build.sh"),
         os.environ["WRF_SOURCE_ROOT"], str(tmp_path / "build"), str(CASE),
         str(out)], check=True)
    assert out.read_bytes() == ORACLE.read_bytes()


@pytest.mark.gpu
@requires_gpu
def test_the_driver_matches_wrf_over_a_varying_step():
    from tools.kf_adaptive_wrf471_oracle import case
    from woof.core.physics import WRF_EXACT

    dts, words = case.read_case(CASE)
    oracle = case.read_oracle(ORACLE)
    cfg, state = case.build_case()
    woof = case.run_woof(cfg, state, words, dts, oracle["run_param"])

    np.testing.assert_array_equal(woof["run_param"], oracle["run_param"])
    # The NCA timer is integral seconds of DT words: exact on every step.
    np.testing.assert_array_equal(woof["nca"], oracle["nca"])
    if WRF_EXACT:
        # Strict arithmetic: the rain words are WRF's, step for step.
        for name in ("rainc", "pratec", "raincv"):
            np.testing.assert_array_equal(woof[name], oracle[name],
                                          err_msg=name)
        tolerance = 2.5e-7      # the held rates' mass couple/decouple
    else:
        tolerance = 2.0e-5      # the KF kernel's default arithmetic
    for name in ("rainc", "pratec", "raincv", "rth", "rqv", "rth_rk"):
        scale = max(float(np.abs(oracle[name]).max()), 1.0e-30)
        error = float(np.abs(woof[name] - oracle[name]).max()) / scale
        assert error <= tolerance, (name, error)
    # Expiry is a discrete decision: where WRF holds zero, so does woof.
    np.testing.assert_array_equal(woof["rth"] == 0.0, oracle["rth"] == 0.0)


@pytest.mark.parametrize("cu_physics", [2, 14])
def test_bmj_and_ksas_run_every_step_under_an_adaptive_clock(cu_physics):
    """WRF resets cudt to 0 for BMJ, every SAS and Tiedtke when the step is
    adaptive (share/module_check_a_mundo.F:2155-2172 in v4.7.1): their
    closures span DT*STEPCU of the call's step.  BMJ was the one member
    woof still ran on a positive cudt."""
    from woof.config import (RunConfig, cumulus_adaptive_cudt_zero_site,
                             validate_run_config)

    assert cumulus_adaptive_cudt_zero_site(cu_physics) is not None

    def run(**kw):
        return RunConfig(nx=30, ny=30, nz=45, dx=12000.0, dy=12000.0,
                         ztop=18000.0, dt=60.0, run_seconds=600.0,
                         moist=True, cu_physics=cu_physics, **kw)

    with pytest.raises(ValueError, match="closure is formed over"):
        validate_run_config(run(use_adaptive_time_step=True,
                                cudt_minutes=5.0))
    validate_run_config(run(use_adaptive_time_step=True, cudt_minutes=0.0))
    validate_run_config(run(cudt_minutes=5.0))
