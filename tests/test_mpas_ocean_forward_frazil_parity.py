"""MPAS-Ocean's forward mode with frazil ice formation, word for word against v8.4.2 at RKIND 4 and 8.

The reference is the forward oracle (``tools/mpas_ocean_oracle/run_forward.F90``)
with ``config_use_frazil_ice_formation`` on: per step the absorbed fractions,
``ocn_frazil_forcing_build_arrays`` (``mpas_ocn_frazil_forcing.F`` compiled
verbatim), ``ocn_timestep`` (whose tendencies add the frazil terms, and whose
diagnostics add the frazil's surface pressure) and the time-level shift, the
accumulated frazil fields shifted with the state. The outputs are committed
(tests/data/oracles/mpas_ocean_forward_frazil, PROVENANCE.md).

THE BREAKAGES THESE PREVENT
---------------------------
1. Frazil built at the wrong point of the step, from the wrong time level, or
   its accumulated fields left out of the cold start or the level shift.
2. A port exact only under strict numerics: the gate runs in fast and strict
   child processes too.
3. A gate that cannot fail: the build skipped, or the frazil fields not
   shifted, must produce mismatches.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _numerics_child import MODES, run as run_child
from conftest import requires_gpu
import _oracle_toolchains
from woof.verify import mpas_ocean_forward_frazil_oracle as oo

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
    land_ice = False
    for name, out in fixture.items():
        case = cases[name]
        nc = case["dims"]["nCells"]
        assert case["config"]["config_use_frazil_ice_formation"]
        mass = np.concatenate([out["state/accumulatedFrazilIceMass"][:nc],
                               out["state/accumulatedFrazilIceMass@2"][:nc]])
        assert (mass > case["pools"]["state"]["accumulatedFrazilIceMass"][:nc].min()).any()
        assert (out["forcing/frazilLayerThicknessTendency"] != 0).any()
        assert np.isfinite(out["state/ssh"][:nc]).all()
        land_ice |= case["pools"]["forcing"]["landIceMask"] is not None
    assert land_ice
    assert any(c["config"]["oracle_n_steps"] % 2 for c in cases.values())
    assert any(c["config"]["oracle_n_steps"] % 2 == 0 for c in cases.values())


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
@pytest.mark.parametrize("mutation", ["no_build", "frazil_not_shifted"])
def test_a_mutated_driver_is_caught(monkeypatch, mutation):
    from woof.hex.mpas_ocean import forward_v842 as fw
    from woof.hex.mpas_ocean import frazil_v842
    if mutation == "no_build":
        monkeypatch.setattr(frazil_v842, "build_arrays", lambda **kw: None)
        fields = ("state/layerThickness", "state/accumulatedFrazilIceMass", "state/accumulatedFrazilIceMass@2")
    else:
        monkeypatch.setattr(fw, "_two_level", lambda pools: dict(fw.TWO_LEVEL))
        fields = ("state/accumulatedFrazilIceMass", "state/accumulatedFrazilIceMass@2")
    case = oo.cases("r8")[0]
    got = oo.run_gpu(case)
    want = oo.load_fixture("r8")[case["name"]]
    assert sum(oo.word_mismatches(got[f], want[f]) for f in fields) > 0


def _oracle_toolchain() -> tuple[str, Path] | None:
    return _oracle_toolchains.mpas_v842("core_ocean")


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MPAS-Ocean oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_ocean_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_ocean_oracle" / "make_fixture.py"),
                    str(tmp_path / "build"), "--piece", "forward_frazil", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
