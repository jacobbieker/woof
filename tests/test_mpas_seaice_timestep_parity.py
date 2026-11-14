"""MPAS-Seaice's forward time step (dynamics only) over N steps, word for word against v8.4.2 at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's forward model as
``tools/mpas_seaice_oracle/run_seaice_step.F90`` drives it, compiled with
gfortran -O0 -ffp-contract=off at RKIND 4 and 8:
- ``seaice_init_evp`` and the velocity solver's time steps;
- ``seaice_initialize_time_diagnostics``;
- per step: the clock, ``seaice_timestep`` and ``seaice_timestep_finalize``.
The step routines, the column wrappers and the time diagnostics are cut
verbatim, and ``mpas_seaice_velocity_solver.F`` is compiled verbatim. The
column package, advection and forcing are off. The meshes are culled
hexagonal tori as one block of a decomposition. The outputs are committed
(tests/data/oracles/mpas_seaice_step, PROVENANCE.md). When ``GFORTRAN`` (or
gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` are set, the fixture itself is
rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A time-step driver that drifts from MPAS over steps or dynamics subcycles:
   the time diagnostics, the step counter, the velocity solver's composition,
   the fluxes re-initialised at the step's end.
2. MPAS's missing-cell element treated as a cell: the loops v8.4.2 bounds by
   ``nCells`` stop there, its whole-array operations do not.
3. A restart that does not continue the run bitwise.
4. A gate that cannot fail: mutated counts, a skipped finalize or a wrong
   subcycle count must produce mismatches.
5. A configuration this driver does not carry, run silently: refused by name.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.verify import mpas_seaice_step_oracle as oo

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": oo.input_digest(case) for p in BOTH for case in oo.cases(p)}


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_branches(precision):
    fixture = oo.load_fixture(precision)
    cases = {case["name"]: case for case in oo.cases(precision)}
    assert set(fixture) == set(cases)
    for name, out in fixture.items():
        case, config = cases[name], cases[name]["config"]
        m = case["pools"]["mesh"]
        nv, nc = case["dims"]["nVertices"], case["dims"]["nCells"]
        assert m["nCellsSolve"] < nc and m["nVerticesSolve"] < nv
        steps = config["oracle_n_steps"]
        assert int(out["mesh/itimestep"][0]) == steps
        days = steps * config["config_dt"] / 86400.0
        assert abs(float(out["diagnostics/daysSinceStartOfSim"]) - days) < 1e-5
        assert abs(float(out["diagnostics/timeAverageTestVariable"]) - days) < 1e-5
        assert (out["velocity_solver/airStressCellU"] == 0).all()
        assert (out["atmos_coupling/atmosReferenceSpeed10m"] == 0).all()
        u = out["velocity_solver/uVelocity"][:nv]
        assert np.isfinite(u).all() and np.count_nonzero(u) > nv // 4
        assert 0 < np.count_nonzero(out["velocity_solver/solveVelocity"][:nv]) < nv
    assert {c["config"]["config_revised_evp"] for c in cases.values()} == {True, False}
    assert any(c["config"]["config_dynamics_subcycle_number"] > 1 for c in cases.values())
    assert any(not c["config"]["config_use_ocean_stress"] for c in cases.values())


@pytest.mark.parametrize("knob, value", [
    ("config_use_column_package", True), ("config_use_advection", True), ("config_use_forcing", True),
    ("config_perform_unit_test", True), ("config_do_restart", True), ("config_use_halo_exch", True),
    ("config_stress_divergence_scheme", "weak"),
])
def test_what_the_driver_does_not_carry_is_refused_by_name(knob, value):
    from woof.hex.mpas_seaice import timestep_v842 as ts
    config = oo.cases()[0]["config"]
    ts.check_config(config)
    with pytest.raises(ConfigurationRefusal, match=knob):
        ts.check_config(dict(config, **{knob: value}))


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word_over_the_steps(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_a_restart_continues_the_run_bitwise(tmp_path, precision):
    """One step, a restart file, a fresh read, the rest: the oracle's words."""
    import cupy as cp
    from woof.hex.mpas_seaice import timestep_v842 as ts
    from woof.verify import mpas_ocean_stream as stream
    case = next(c for c in oo.cases(precision) if c["config"]["oracle_n_steps"] >= 3)
    config = case["config"]
    n = int(config["oracle_n_steps"])
    dev = stream.on_device(oo.FIELDS, case)
    dev["diagnostics"]["xtime"] = config["diagnostics:xtime"]
    dev["diagnostics"]["simulationStartTime"] = config["diagnostics:simulationStartTime"]
    clock = ts.initialize(pools=dev, configs=config)
    next_step = ts.advance(pools=dev, configs=config, clock=clock, n_steps=1)
    dev["diagnostics"]["oracle_next_step"] = next_step
    ts.write_restart(tmp_path / "restart.npz", pools=dev, clock=clock)
    pools, clock = ts.read_restart(tmp_path / "restart.npz", like=dev)
    del dev
    ts.advance(pools=pools, configs=config, clock=clock, n_steps=n - 1,
               first_step=int(pools["diagnostics"]["oracle_next_step"]))
    want = oo.load_fixture(precision)[case["name"]]
    bad = {key: m for key, words in want.items()
           if (m := oo.word_mismatches(cp.asnumpy(pools[key.split("/")[0]][key.split("/", 1)[1]]), words))}
    assert not bad, bad


@requires_gpu
@pytest.mark.parametrize("mutation", ["no_block_counts", "no_finalize", "one_dynamics_subcycle"])
def test_a_mutated_driver_is_caught(monkeypatch, mutation):
    import contextlib
    from woof.hex.mpas_seaice import evp_v842
    from woof.hex.mpas_seaice import timestep_v842 as ts
    case = next(c for c in oo.cases("r8") if c["config"]["config_dynamics_subcycle_number"] > 1)
    if mutation == "no_block_counts":
        monkeypatch.setattr(evp_v842, "block_counts", lambda **kw: contextlib.nullcontext())
        field = "velocity_solver/divergence"
    elif mutation == "no_finalize":
        case["pools"]["velocity_solver"]["airStressCellU"][:] = 0.25
        original = ts.advance

        def advance_without_finalize(**kw):
            pools = kw["pools"]
            keep = pools["velocity_solver"]["airStressCellU"].copy()
            out = original(**kw)
            pools["velocity_solver"]["airStressCellU"][...] = keep
            return out
        monkeypatch.setattr(ts, "advance", advance_without_finalize)
        field = "velocity_solver/airStressCellU"
    else:
        case["config"] = dict(case["config"], config_dynamics_subcycle_number=1)
        original_init = ts.initialize

        def init_with_two(**kw):
            clock = original_init(pools=kw["pools"], configs=dict(kw["configs"], config_dynamics_subcycle_number=2))
            return clock
        monkeypatch.setattr(ts, "initialize", init_with_two)
        field = "velocity_solver/uVelocity"
    got = oo.run_gpu(case)
    assert oo.word_mismatches(got[field], oo.load_fixture("r8")[case["name"]][field]) > 0


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_seaice").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MPAS-Seaice oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_step_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
