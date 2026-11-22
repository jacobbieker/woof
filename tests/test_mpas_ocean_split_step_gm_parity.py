"""One MPAS-Ocean split-explicit step with GM and Redi on, word for word against v8.4.2 at RKIND 4 and 8.

The reference is the split step oracle (``tools/mpas_ocean_oracle/run_split_step.F90``)
with ``config_use_standardGM`` on and GM compiled verbatim (``mpas_ocn_gm.F``,
``ocn_reconstruct_gm_vectors`` cut out of ``mpas_ocn_diagnostics.F``): stage
2's transport velocity adds the input's bolus velocity, the tracer tendency
runs Redi on the input's slopes, GM is computed before the implicit vertical
mixing (which adds its ``k33``) and after the last diagnostics, and the
step ends with GM's vectors on cells. The outputs are committed
(tests/data/oracles/mpas_ocean_split_step_gm, PROVENANCE.md).

THE BREAKAGES THESE PREVENT
---------------------------
1. GM or Redi run at the wrong point of the step, on the wrong time level or
   not at all: either GM call, Redi in the tracer tendency, and the vectors
   at the end are each held by a mutation.
2. GM's ``nVertLevels+1`` fields reconstructed with the wrong level stride.
3. A port exact only under strict numerics: the gate runs in fast and strict
   child processes too.
4. A case generator whose bytes drift. The input digests are pinned.
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
from woof.verify import mpas_ocean_split_step_gm_oracle as oo

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
        config, d0 = case["config"], case["pools"]["diagnostics"]
        nc, ne = case["dims"]["nCells"], case["dims"]["nEdges"]
        assert config["config_use_standardGM"]
        assert np.isfinite(out["state/normalVelocity@2"][:ne]).all()
        assert np.isfinite(out["tracers/activeTracers@2"][:nc]).all()
        bolus = out["diagnostics/normalGMBolusVelocity"][:ne]
        assert np.count_nonzero(bolus) and not (bolus == d0["normalGMBolusVelocity"][:ne]).all()
        # the step's GM rewrites the slopes; the vectors are made from them
        assert not (out["diagnostics/relativeSlopeTopOfEdge"][:ne] == d0["relativeSlopeTopOfEdge"][:ne]).all()
        for vector in ("transportVelocityX", "GMBolusVelocityZonal", "relativeSlopeTopOfCellY", "GMStreamFuncZ"):
            assert not (out[f"diagnostics/{vector}"][:nc] == d0[vector][:nc]).all(), vector
        k33 = out["diagnostics/k33"][:nc]
        assert (np.count_nonzero(k33) == 0) == config["config_disable_redi_k33"]
    configs = [c["config"] for c in cases.values()]
    assert {c["config_time_integrator"] for c in configs} == {"split_explicit", "unsplit_explicit"}
    assert any(c["config_use_Redi_surface_layer_tapering"] and c["config_use_Redi_bottom_layer_tapering"]
               for c in configs)
    assert any(c["config_disable_redi_horizontal_term1"] for c in configs)


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_v842_oracle_word_for_word"], numerics, exact)


def _skip_call(module, name: str, which: int, monkeypatch) -> None:
    """``module.name`` with its ``which``-th call (1-based) a no-op."""
    real, calls = getattr(module, name), []

    def wrapped(**kw):
        calls.append(1)
        if len(calls) != which:
            real(**kw)
    monkeypatch.setattr(module, name, wrapped)


@requires_gpu
@pytest.mark.parametrize("mutation, field", [
    ("no_gm_before_vmix", "tracers/activeTracers@2"),
    ("no_gm_after_diagnostics", "diagnostics/normalGMBolusVelocity"),
    ("no_redi", "tracers/activeTracers@2"),
    ("no_gm_vectors", "diagnostics/GMBolusVelocityX"),
    ("p1_vectors_at_nl_stride", "diagnostics/GMStreamFuncX"),
])
def test_a_mutated_step_is_caught(monkeypatch, mutation, field):
    from woof.hex.mpas_ocean import gm_v842, split_step_v842, tracer_hmix_v842
    if mutation == "no_gm_before_vmix":
        _skip_call(gm_v842, "compute_bolus_velocity", 1, monkeypatch)
    elif mutation == "no_gm_after_diagnostics":
        _skip_call(gm_v842, "compute_bolus_velocity", 2, monkeypatch)
    elif mutation == "no_redi":
        monkeypatch.setattr(tracer_hmix_v842, "redi_on", lambda configs: False)
    elif mutation == "no_gm_vectors":
        monkeypatch.setattr(split_step_v842, "reconstruct_gm_vectors", lambda **kw: None)
    else:
        real = split_step_v842._launch

        def launch(precision, name, n, args):
            if name == "ocn_ss_reconstruct_v842":
                args = (*args[:-2], args[12], args[12])  # both strides nVertLevels
            real(precision, name, n, args)
        monkeypatch.setattr(split_step_v842, "_launch", launch)
    case = oo.cases("r8")[0]
    got = oo.run_gpu(case)
    want = oo.load_fixture("r8")[case["name"]]
    assert oo.word_mismatches(got[field], want[field]) > 0


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
                    str(tmp_path / "build"), "--piece", "split_step_gm", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
