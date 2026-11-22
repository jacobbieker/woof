"""MPAS-Seaice's forward time step with incremental-remap advection, word for word against v8.4.2 (RKIND 4 and 8).

The reference is MPAS-Model v8.4.2's forward model as
``tools/mpas_seaice_oracle/run_seaice_step.F90`` drives it with
``config_use_advection``: ``seaice_init_advection`` (the remap's init,
verbatim) and, each dynamics subcycle, ``seaice_run_advection`` (compiled
verbatim) after the velocity solver, over N steps, on planar hexagon and
quadrilateral blocks. The init's geometry is the GPU's input. The outputs are
committed (tests/data/oracles/mpas_seaice_step_remap, PROVENANCE.md).

THE BREAKAGES THESE PREVENT
---------------------------
1. Advection out of its place in the step (after each dynamics subcycle's
   velocity solve, at dynamicsTimeStep), or remapped ice that does not feed
   the next subcycle.
2. A restart that does not continue the remapped run bitwise.
3. A gate that cannot fail: a step without advection, or advection at the
   step's dt instead of the subcycle's, must produce mismatches.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.verify import mpas_seaice_remap_oracle as ro
from woof.verify import mpas_seaice_step_remap_oracle as sr

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (sr.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": sr.input_digest(case) for p in BOTH for case in sr.cases(p)}


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_branches(precision):
    fixture = sr.load_fixture(precision)
    cases = {case["name"]: case for case in sr.cases(precision)}
    assert set(fixture) == set(cases)
    for name, out in fixture.items():
        case = cases[name]
        nc = case["dims"]["nCells"]
        area = out["tracers/iceAreaCategory"][:nc]
        assert np.isfinite(area).all() and (area >= 0).all()
        assert ro.word_mismatches(area, case["pools"]["tracers"]["iceAreaCategory"][:nc]) > nc // 4
        assert "tracer_edge_fluxes/iceEnthalpyEdgeFlux" in out
        assert int(out["mesh/itimestep"][0]) == case["config"]["oracle_n_steps"]
    assert {c["mesh_kind"] for c in cases.values()} == {"hex", "quad"}
    assert any(c["config"]["config_dynamics_subcycle_number"] > 1 for c in cases.values())


def test_upwind_in_the_step_is_refused_by_name():
    from woof.hex.mpas_seaice import timestep_v842 as ts
    config = sr.cases("r8")[0]["config"]
    ts.check_config(config)
    with pytest.raises(ConfigurationRefusal, match="undeclared"):
        ts.check_config(dict(config, config_advection_type="upwind"))


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word_over_the_steps(precision):
    report = sr.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_a_restart_continues_the_remapped_run_bitwise(tmp_path, precision):
    import cupy as cp
    from woof.hex.mpas_seaice import timestep_v842 as ts
    case = next(c for c in sr.cases(precision) if c["config"]["oracle_n_steps"] >= 3)
    fixture = sr.load_fixture(precision)[case["name"]]
    config = case["config"]
    dev = sr.device_pools(case, fixture)
    clock = ts.initialize(pools=dev, configs=config, remap_geometry="input", basis="input")
    next_step = ts.advance(pools=dev, configs=config, clock=clock, n_steps=1)
    dev["diagnostics"]["oracle_next_step"] = next_step
    ts.write_restart(tmp_path / "restart.npz", pools=dev, clock=clock)
    pools, clock = ts.read_restart(tmp_path / "restart.npz", like=dev, configs=case["config"])
    del dev
    ts.advance(pools=pools, configs=config, clock=clock, n_steps=int(config["oracle_n_steps"]) - 1,
               first_step=int(pools["diagnostics"]["oracle_next_step"]))
    want = ro.expected_outputs(case, fixture)
    bad = {key: m for key, words in want.items() if ro.graded(case, key)
           and (m := sr.word_mismatches(cp.asnumpy(pools[key.split("/")[0]][key.split("/", 1)[1]]), words))}
    assert not bad, bad


@requires_gpu
@pytest.mark.parametrize("mutation", ["no_advection", "step_dt"])
def test_a_mutated_step_is_caught(monkeypatch, mutation):
    from woof.hex.mpas_seaice import remap_v842 as rv
    from woof.hex.mpas_seaice import timestep_v842 as ts
    case = next(c for c in sr.cases("r8") if c["config"]["config_dynamics_subcycle_number"] > 1)
    fixture = sr.load_fixture("r8")[case["name"]]
    if mutation == "no_advection":
        monkeypatch.setattr(ts, "advection_tree", lambda pools, configs: None)
    else:
        original = rv.remap_block

        def at_step_dt(tree, *, pools, mesh, configs, dt):
            return original(tree, pools=pools, mesh=mesh, configs=configs, dt=float(configs["config_dt"]))
        monkeypatch.setattr(rv, "remap_block", at_step_dt)
    got = sr.run_gpu(case, fixture)
    field = "tracers/iceAreaCategory"
    assert sr.word_mismatches(got[field], fixture[field]) > 0


def _oracle_toolchain():
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_seaice").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MPAS-Seaice oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
@pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64"),
                    reason="the fixture is x86-64 gfortran's: arm64 gfortran's MAX/MIN keep another operand "
                           "on signed-zero ties")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_step_remap_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(sr.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert sr.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
