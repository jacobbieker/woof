"""MPAS-Ocean's forward mode with CVMix (and KPP) on over N steps, word for word against v8.4.2 at RKIND 8,
on uniform and variable-resolution meshes.

The reference is the forward oracle (``tools/mpas_ocean_oracle/run_forward.F90``)
linked with CVMix v0.84-beta and ``mpas_ocn_vmix_cvmix.F`` verbatim
(``run_forward_cvmix_r8``): the cold start's boundary layer depth, then
each step's KPP input fields, non-local flux and CVMix coefficients, KPP's
boundary layer and non-local flux handed from step to step. The outputs are
committed (tests/data/oracles/mpas_ocean_forward_cvmix, PROVENANCE.md).

THE BREAKAGES THESE PREVENT
---------------------------
1. A port exact for one step that drifts over several, on a uniform or a
   variable-resolution mesh, on one block or on a decomposition's block
   (nested halos: KPP's boundary-layer smoothing over the halo).
2. KPP's input fields read on the wrong time level: held by a mutation.
3. A fixture that misses a branch: uniform and variable resolution, nested
   halos, the non-local flux in the tendency and in the implicit solve, the
   unsplit integrator without KPP, the wind-driven friction velocity (behind
   ``WOOF_GLIBC239_LIBM64=1`` at RKIND 8) and the calm one.
4. A port exact only under strict numerics: the gate runs in fast and strict
   child processes too.
5. A case generator whose bytes drift. The input digests are pinned.
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
from woof.verify import mpas_ocean_forward_cvmix_oracle as oo

ROOT = Path(__file__).resolve().parents[1]


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/r8": oo.input_digest(case) for case in oo.cases("r8")}


def test_the_meshes_are_uniform_or_vary_in_resolution():
    for name, _, _, _, _, _, _, overrides in oo.CASES:
        case = next(c for c in oo.cases("r8") if c["name"] == name)
        mesh, nc, ne = case["pools"]["mesh"], case["dims"]["nCells"], case["dims"]["nEdges"]
        spread = mesh["dcEdge"][:ne].max() / mesh["dcEdge"][:ne].min()
        area = mesh["areaCell"][:nc].max() / mesh["areaCell"][:nc].min()
        if overrides["_resolution"] == "uniform":
            assert spread == 1.0 and area == 1.0, name
        else:
            assert spread > 2.5 and area > 6.0, name
        assert np.allclose(mesh["dvEdge"][:ne] * np.sqrt(3.0), mesh["dcEdge"][:ne])


def test_fixture_covers_the_branches():
    fixture = oo.load_fixture("r8")
    cases = {case["name"]: case for case in oo.cases("r8")}
    assert set(fixture) == set(cases)
    seen = set()
    for name, out in fixture.items():
        case = cases[name]
        c = case["config"]
        nc, ne = case["dims"]["nCells"], case["dims"]["nEdges"]
        assert c["oracle_n_steps"] >= 2 and c["config_use_cvmix"]
        assert np.isfinite(out["state/normalVelocity"][:ne]).all()
        seen.add(case["name"].split("_")[0])
        if len(set(case["pools"]["mesh"]["nCellsArray"].tolist())) > 1:
            seen.add("nested")
        if not c["config_use_cvmix_kpp"]:
            continue
        assert np.count_nonzero(out["diagnostics/vertNonLocalFlux"][:nc]), name
        friction = out["diagnostics/surfaceFrictionVelocity"][:int(case["pools"]["mesh"]["nCellsArray"][0])]
        assert (friction > 0).all() == c.get("config_use_bulk_wind_stress", False), name
        seen.add("wind" if c.get("config_use_bulk_wind_stress", False) else "calm")
        seen.add("implicit" if c["config_cvmix_kpp_nonlocal_with_implicit_mix"] else "tendency")
    assert seen == {"kpp", "cvmix", "nested", "wind", "calm", "implicit", "tendency"}
    assert {o["_resolution"] for *_, o in oo.CASES} == {"uniform", "variable"}


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word():
    report = oo.measure("r8")
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_v842_oracle_word_for_word"], numerics, exact)


@requires_gpu
def test_kpp_inputs_on_the_wrong_time_level_are_caught(monkeypatch):
    from woof.hex.mpas_ocean import tend_tracer_v842
    real = tend_tracer_v842.tend_tracer

    def tend_tracer(**kw):
        real(**dict(kw, tracers_level1=kw["tracers"]))  # time level 2 for the surface fluxes
    monkeypatch.setattr(tend_tracer_v842, "tend_tracer", tend_tracer)
    case = next(c for c in oo.cases("r8") if c["name"] == "kpp_forward_variable_jm_bulk")
    got = oo.run_gpu(case)
    want = oo.load_fixture("r8")[case["name"]]
    field = "tracers/activeTracers"
    assert oo.word_mismatches(got[field], want[field]) > 0


def _oracle_toolchain() -> tuple[str, Path, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    cvmix = os.environ.get("CVMIX_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not cvmix or not gfortran or not Path(source, "src", "core_ocean").is_dir():
        return None
    return gfortran, Path(source), Path(cvmix)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the CVMix oracle build needs gfortran (pixi -e mpas-oracle), WOOF_MPAS_V842_SOURCE "
                           "naming an MPAS-Model v8.4.2 tree and CVMIX_SOURCE naming CVMix v0.84-beta's src/shared")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source, cvmix = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_ocean_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True,
                   env={**os.environ, "GFORTRAN": gfortran, "CVMIX_SOURCE": str(cvmix)})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_ocean_oracle" / "make_fixture.py"),
                    str(tmp_path / "build"), "--piece", "forward_cvmix", "--out", str(tmp_path / "fixture")],
                   check=True)
    fresh = np.load(tmp_path / "fixture" / "oracle-r8.npz")
    pinned = np.load(oo.FIXTURE_DIR / "oracle-r8.npz")
    assert sorted(fresh.files) == sorted(pinned.files)
    for key in pinned.files:
        assert oo.word_mismatches(fresh[key], pinned[key]) == 0, key
