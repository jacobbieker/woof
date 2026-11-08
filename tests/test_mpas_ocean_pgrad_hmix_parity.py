"""MPAS-Ocean pressure gradient and hmix, word for word against v8.4.2, at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's own ``ocn_vel_pressure_grad_tend`` (every
``config_pressure_gradient_type``) and ``ocn_vel_hmix_tend`` (del2, Leith,
del4), compiled verbatim with gfortran -O0 -ffp-contract=off at MPAS's default
single precision and at its double precision, and driven by
``tools/mpas_ocean_oracle/run_pgrad_hmix.F90``. The outputs are committed
(tests/data/oracles/mpas_ocean_pgrad_hmix, PROVENANCE.md). When ``GFORTRAN``
(or gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` are set, the fixture itself
is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance, in either precision.
2. A case generator whose bytes drift. The input digests are pinned.
3. A gate that cannot fail. A mutated evaluation order, or the Jacobian
   branches' layer-1 term skipped on a land edge, must produce mismatches.
4. An unported option (another pressure-gradient type, the tensor hmix forms)
   silently running as something else. Each is refused by name.
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
from woof.verify import mpas_ocean_pgrad_hmix_oracle as oo

ROOT = Path(__file__).resolve().parents[1]


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": oo.input_digest(case, p)
                        for p in oo.PRECISIONS for case in oo.cases()}


def test_fixture_covers_the_branches():
    all_cases = oo.cases()
    assert {case["pgradType"] for case in all_cases} == set(oo.PGRAD_TYPES)
    assert {case["reals"]["config_common_level_weight"] for case in all_cases} >= {0.0, 0.5, 1.0}
    for flag in oo.FLAGS:
        assert {case["flags"][flag] for case in all_cases} == {True, False}, flag
    for case in all_cases:
        land = case["maxLevelEdgeTop"] == 0
        assert (land & (case["edgeMask"][:, 0] == 1)).any()
        inside = np.arange(case["nVertLevels"])[None, :] < case["maxLevelEdgeTop"][:, None]
        assert (inside & (case["edgeMask"] == 0)).any()
        assert (case["maxLevelVertexTop"] < case["nVertLevels"]).any()
        assert {-1, 1} <= set(np.unique(case["edgeSignOnVertex"]))
        assert (case["dvEdge"] < 0.25 * case["dcEdge"]).any()
    r8 = oo.load_fixture("r8")
    # Leith viscosities on both sides of config_Leith_visc2_max
    leith = [c for c in all_cases if c["flags"]["config_use_Leith_del2"] and not c["flags"]["config_use_mom_del2"]]
    for case in leith:
        visc = r8[case["name"]]["viscosity"]
        cap = case["reals"]["config_Leith_visc2_max"]
        assert (visc == cap).any() and ((visc > 0) & (visc < cap)).any()
    for precision in oo.PRECISIONS:
        fixture = oo.load_fixture(precision)
        assert set(fixture) == {row[0] for row in oo.CASES}
        assert all(np.isfinite(v).all() for c in fixture.values() for v in c.values())


def test_unported_options_are_refused_by_name():
    from woof.hex.mpas_ocean import pgrad_hmix_v842 as ph
    assert [ph.pgrad_type_code(t) for t in oo.PGRAD_TYPES] == [1, 2, 3, 4, 5]
    with pytest.raises(ConfigurationRefusal) as exc:
        ph.pgrad_type_code("Jacobian_POM_SCRUM")
    assert "config_pressure_gradient_type" in str(exc.value)
    for knob in ("config_use_mom_del2_tensor", "config_use_mom_del4_tensor"):
        with pytest.raises(ConfigurationRefusal) as exc:
            ph.refuse_hmix_tensor(**{"config_use_mom_del2_tensor": False, "config_use_mom_del4_tensor": False,
                                     knob: True})
        assert knob in str(exc.value)


def test_pgrad_constants_are_the_rkind_words():
    from woof.hex.mpas_ocean import pgrad_hmix_v842 as ph
    d0, g0 = ph.pgrad_constants(1026.0, "r4")
    assert d0.dtype == np.float32 and d0 == np.float32(1.0) / np.float32(1026.0)
    assert g0 == np.float32(9.80616) / np.float32(1026.0)
    d0, g0 = ph.pgrad_constants(1026.0, "r8")
    assert g0.dtype == np.float64 and g0 == 9.80616 / 1026.0


@requires_gpu
@pytest.mark.parametrize("precision", sorted(oo.PRECISIONS))
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", sorted(oo.PRECISIONS))
@pytest.mark.parametrize("original, mutated, case_name, field", [
    # the Jacobian's zStar denominator reassociated
    ("D(S(M(z2m, z1m), M(z2k, z1k)), S(A(S(z2m, z2k), z1m), z1k))",
     "D(S(M(z2m, z1m), M(z2k, z1k)), S(S(z2m, z2k), S(z1k, z1m)))",
     "jdens_all", "tendPgrad"),
    # the layer-1 term skipped where maxLevelEdgeTop is 0
    ("rk pGrad = M(M((rk)edgeMask[base], invdcEdge), x);\n        tend[base] = A(tend[base], pGrad);",
     "rk pGrad = M(M((rk)edgeMask[base], invdcEdge), x);\n        if (kmax > 0) tend[base] = A(tend[base], pGrad);",
     "jts_gamma1", "tendPgrad"),
    # the Leith viscosity's factors regrouped
    ("visc2 = M(M(M(M(p, M(p, p)), ABS(dRv)), invLength1), (rk)SQRT3);",
     "visc2 = M(M(p, M(p, p)), M(ABS(dRv), M(invLength1, (rk)SQRT3)));",
     "mont_leith", "tendHmix"),
    # del4's vertex sum term regrouped
    ("M(M(M(sign, dcEdge[iEdge]), delsqU[iEdge * nl + k]), invAreaTri1)",
     "M(M(sign, dcEdge[iEdge]), M(delsqU[iEdge * nl + k], invAreaTri1))",
     "montd_del4", "tendHmix"),
])
def test_a_mutated_port_is_caught(monkeypatch, precision, original, mutated, case_name, field):
    from woof.hex.mpas_ocean import pgrad_hmix_v842
    real = pgrad_hmix_v842.module_source
    assert real("r4").count(original) == 1
    monkeypatch.setattr(pgrad_hmix_v842, "module_source", lambda p="r4": real(p).replace(original, mutated))
    pgrad_hmix_v842._module_on.cache_clear()
    try:
        case = next(c for c in oo.cases() if c["name"] == case_name)
        got = oo.run_gpu(case, precision)
        assert oo.bit_mismatches(got[field], oo.load_fixture(precision)[case_name][field]) > 0
    finally:
        pgrad_hmix_v842._module_on.cache_clear()


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
                    str(tmp_path / "build"), "--piece", "pgrad_hmix", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in oo.PRECISIONS:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.bit_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
