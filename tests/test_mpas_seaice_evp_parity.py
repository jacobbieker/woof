"""The MPAS-Seaice variational EVP internal stress, word for word against v8.4.2.

The reference is MPAS-Model v8.4.2's own
``mpas_seaice_velocity_solver_constitutive_relation.F`` and
``mpas_seaice_velocity_solver_variational.F``, compiled verbatim with gfortran
-O0 -ffp-contract=off at MPAS's default single precision and driven by
``tools/mpas_seaice_oracle/run_oracle.F90``. That driver runs
``seaice_init_evp``, then nIter calls of ``seaice_internal_stress_variational``,
then ``seaice_final_divergence_shear_variational``. The oracle's outputs are
committed (tests/data/oracles/mpas_seaice_evp, PROVENANCE.md), so the GPU
comparison runs without gfortran. When the oracle can be built here (gfortran
on PATH or named by ``GFORTRAN``, e.g. the mpas-oracle pixi environment's, and
``WOOF_MPAS_V842_SOURCE`` naming the MPAS-Model v8.4.2 tree), the fixture
itself is regenerated and compared too.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance. Every output word must be
   bit-identical, including the entries the routines leave alone.
2. A case generator whose bytes drift, so the fixture no longer describes
   the inputs the GPU is fed. The input digests are pinned.
3. A gate that cannot fail. Mutating one evaluation order in the kernel must
   produce mismatches.
4. The weak-form scheme silently running the variational kernels. It is
   refused by name.
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
from woof.verify import mpas_seaice_evp_oracle as so

ROOT = Path(__file__).resolve().parents[1]


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (so.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {case["name"]: so.input_digest(case) for case in so.cases()}


def test_fixture_covers_the_branches():
    fixture = so.load_fixture()
    assert set(fixture) == {row[0] for row in so.CASES}
    for case in so.cases():
        # unsolved cells and vertices exist, and so do cells at rest (Delta below puny)
        assert (case["solveStress"] == 0).any() and (case["solveVelocity"] == 0).any()
        n = case["nEdgesOnCell"]
        assert set(np.unique(n)) >= ({5, 6, 7} if case["maxEdges"] == 7 else {5, 6})
        at_rest = [c for c in range(case["nCells"])
                   if not case["uVelocity"][case["verticesOnCell"][c, :n[c]] - 1].any()
                   and not case["vVelocity"][case["verticesOnCell"][c, :n[c]] - 1].any()]
        assert at_rest
        out = fixture[case["name"]]
        assert all(np.isfinite(v).all() for v in out.values())


def test_weak_stress_divergence_is_refused_by_name():
    from woof.hex.mpas_seaice import evp_v842
    evp_v842.require_scheme("variational")
    with pytest.raises(ConfigurationRefusal) as exc:
        evp_v842.require_scheme("weak")
    assert "config_stress_divergence_scheme" in str(exc.value)


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word():
    report = so.measure()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
def test_the_seaice_door_verify_grades_the_ported_piece():
    from woof.hex.mpas_seaice.verify import verify
    report = verify()
    assert report["passed"] and report["status"] == "stub"
    assert [p["name"] for p in report["pieces"]][0] == "evp_variational_internal_stress"


@requires_gpu
@pytest.mark.parametrize("original, mutated, field", [
    # (d*100)*86400 folded to d*8640000
    ("divergence[iCell] = M(M(div, 100.0f), 86400.0f);",
     "divergence[iCell] = M(div, 8640000.0f);", "divergence"),
    # the stress-divergence normal terms summed in the other order
    ("cellU = S(S(cellU, M(s11, bU)), M(s12, bV));",
     "cellU = S(cellU, A(M(s11, bU), M(s12, bV)));", "stressDivergenceU"),
])
def test_a_mutated_evaluation_order_is_caught(monkeypatch, original, mutated, field):
    from woof.hex.mpas_seaice import evp_v842
    source = evp_v842.module_source()
    assert source.count(original) == 1
    monkeypatch.setattr(evp_v842, "module_source", lambda: source.replace(original, mutated))
    evp_v842._module_on.cache_clear()
    try:
        case = so.cases()[0]
        got = so.run_gpu(case)
        assert so.bit_mismatches(got[field], so.load_fixture()[case["name"]][field]) > 0
    finally:
        evp_v842._module_on.cache_clear()


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
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    fresh = np.load(tmp_path / "fixture" / "oracle-r4.npz")
    pinned = np.load(so.FIXTURE_DIR / "oracle-r4.npz")
    assert sorted(fresh.files) == sorted(pinned.files)
    for key in pinned.files:
        assert so.bit_mismatches(fresh[key], pinned[key]) == 0, key
