"""MPAS-Ocean's forward mode with GM and Redi on, word for word against v8.4.2 at RKIND 4 and 8.

The reference is the forward oracle (``tools/mpas_ocean_oracle/run_forward.F90``)
with ``config_use_standardGM`` on and GM compiled verbatim: the cold start's
boundary layer depth, GM's bolus velocity added to the transport velocity and
its vectors (``mpas_ocn_init_routines.F``), then each step with GM and Redi
(``tests/test_mpas_ocean_split_step_gm_parity.py``) and the time-level shift.
The outputs are committed (tests/data/oracles/mpas_ocean_forward_gm,
PROVENANCE.md).

THE BREAKAGES THESE PREVENT
---------------------------
1. The cold start without GM: the first step's transport velocity without the
   bolus velocity, or the surface tapering without the boundary layer depth.
2. GM and Redi composed over several steps and both integrators, through the
   time-level shift.
3. A port exact only under strict numerics: the gate runs in fast and strict
   child processes too.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _numerics_child import MODES, run as run_child
from conftest import requires_gpu
from woof.verify import mpas_ocean_forward_gm_oracle as oo

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
        case = cases[name]
        nc, ne = case["dims"]["nCells"], case["dims"]["nEdges"]
        assert case["config"]["config_use_standardGM"]
        # single blocks: v8.4.2's slopes read GM's huge fill on the outermost halo edges
        assert len(set(case["pools"]["mesh"]["nCellsArray"].tolist())) == 1
        assert np.isfinite(out["state/ssh"][:nc]).all()
        assert np.count_nonzero(out["diagnostics/normalGMBolusVelocity"][:ne])
        # the cold start's boundary layer depth: half the top layer
        h = case["pools"]["state"]["layerThickness"][:nc, 0]
        assert (out["diagnostics/boundaryLayerDepth"][:nc] == h * h.dtype.type(0.5)).all()
    configs = [c["config"] for c in cases.values()]
    assert {c["config_time_integrator"] for c in configs} == {"split_explicit", "unsplit_explicit"}
    assert any(c["config_use_Redi_surface_layer_tapering"] for c in configs)
    assert any(c["oracle_n_steps"] % 2 for c in configs) and any(c["oracle_n_steps"] % 2 == 0 for c in configs)


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word_over_the_steps(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_v842_oracle_word_for_word_over_the_steps"], numerics, exact)


@requires_gpu
@pytest.mark.parametrize("mutation", ["no_cold_start_gm", "no_boundary_layer_depth"])
def test_a_mutated_cold_start_is_caught(monkeypatch, mutation):
    from woof.hex.mpas_ocean import gm_v842
    real, calls = gm_v842.compute_bolus_velocity, []

    def first_call_mutated(**kw):
        calls.append(1)
        if len(calls) > 1:
            real(**kw)
        elif mutation == "no_boundary_layer_depth":
            # the input's noise in place of the cold start's half top layer
            kw["diagnostics"]["boundaryLayerDepth"][...] = noise
            real(**kw)
    monkeypatch.setattr(gm_v842, "compute_bolus_velocity", first_call_mutated)
    if mutation == "no_cold_start_gm":
        name, fields = "gm_split_three_steps", ("state/layerThickness", "state/layerThickness@2")
    else:
        name, fields = "gm_split_jm_rich_tapering", ("diagnostics/relativeSlopeTapering",)
    case = next(c for c in oo.cases("r8") if c["name"] == name)
    import cupy as cp
    noise = cp.asarray(case["pools"]["diagnostics"]["boundaryLayerDepth"])
    got = oo.run_gpu(case)
    want = oo.load_fixture("r8")[name]
    assert sum(oo.word_mismatches(got[f], want[f]) for f in fields) > 0


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_ocean").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MPAS-Ocean oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_ocean_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_ocean_oracle" / "make_fixture.py"),
                    str(tmp_path / "build"), "--piece", "forward_gm", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
