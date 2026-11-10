"""MALI's thickness and tracer advection, word for word against v8.4.2.

The reference is MALI v8.4.2's own land-ice modules, compiled verbatim with
gfortran -O0 -ffp-contract=off at RKIND 4. ``tools/mali_oracle/run_advection.F90``
makes a restart's ``li_thermal_init`` and then one
``li_advection_thickness_tracers`` call: the layer thicknesses, the tracer
setup (temperature or enthalpy), the first-order upwind advection, the
dynamic thickening, the masks, the surface and basal mass balance, the
vertical remap and the tracer finish. The oracle's outputs are committed
(tests/data/oracles/mali_advection, PROVENANCE.md), so the GPU comparison runs
without gfortran. When the oracle can be built here (``GFORTRAN`` or gfortran
on PATH, and ``WOOF_MPAS_V842_SOURCE`` naming the MPAS-Model v8.4.2 tree),
the fixture itself is regenerated and compared too.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MALI only to a tolerance. Every output word, the
   masks included, must be bit-identical, halo cells too.
2. A case generator whose bytes drift. The input digests are pinned.
3. A gate that cannot fail. Mutating one evaluation order in the kernels must
   produce mismatches.
4. Outputs that depend on the model's uninitialised scratch arrays: the
   oracle fills them with random words, and the port never reads them.
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
from woof.verify import mali_advection_oracle as ao

ROOT = Path(__file__).resolve().parents[1]


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (ao.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {**{case["name"]: ao.input_digest(case) for case in ao.cases()},
                        **{case["name"] + "/r8": ao.input_digest(case) for case in ao.cases("r8")}}


def test_fixture_covers_the_branches():
    fixture = ao.load_fixture()
    cases = {case["name"]: case for case in ao.cases()}
    assert set(fixture) == set(cases)
    assert {c["thermalSolver"] for c in cases.values()} == set(ao.THERMAL)
    assert {c["advectTracers"] for c in cases.values()} == {False, True}
    assert {c["thicknessAdvection"] for c in cases.values()} == {"fo", "none"}
    assert {c["velocitySolver"] for c in cases.values()} == {"sia", "none"}
    for name, out in fixture.items():
        case = cases[name]
        # the oracle ran clean: no error flag, no logged error
        assert out["errInit"][0] == 0 and out["errAdvect"][0] == 0 and out["logErrors"][0] == 0
        assert all(np.isfinite(v).all() for v in out.values())
        if case["thicknessAdvection"] == "fo":
            # whole layers melted, ice-free cells, both mass-balance signs on ice
            assert (out["layerThickness"] == 0).any() and (out["thickness"] == 0).any()
            assert (out["thickness"] > 0).any()
            assert (case["sfcMassBal"] > 0).any() and (case["sfcMassBal"] < 0).any()
        v = case["layerNormalVelocity"]
        assert (np.signbit(v) & (v == 0)).any() and (~np.signbit(v) & (v == 0)).any()
        assert case["nCellsSolve"] < case["nCells"]
    # air temperatures on both sides of the freezing point reach min(T, 273.15)
    t = cases["fo_enthalpy"]["surfaceAirTemperature"]
    assert (t > 273.15).any() and (t < 273.15).any()


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = ao.measure(precision=precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
@pytest.mark.parametrize("original, mutated, field, case", [
    # the edge flux's velocity, length and thickness multiplied in another order
    ("const rk flux = M(M(vel, dvEdge[iEdge]), layerThicknessEdge[iEdge * nLevels + k]);",
     "const rk flux = M(vel, M(dvEdge[iEdge], layerThicknessEdge[iEdge * nLevels + k]));", "thickness",
     "fo_temperature"),
    # the dynamic thickening scaled to years before the division by dt
    ("M(D(S(total, thickness[iCell]), dt), LI_SCYR)", "D(M(S(total, thickness[iCell]), LI_SCYR), dt)",
     "dynamicThickening", "fo_thickness_only"),
    # the surface accumulation's dt/rhoi formed first
    ("const rk accum = D(M(sfc, dt), rhoi);", "const rk accum = M(sfc, D(dt, rhoi));", "layerThickness",
     "fo_enthalpy"),
    # the basal accumulation's likewise
    # (at RKIND 8 the difference reaches the remapped temperature, not the layer thicknesses)
    ("const rk accum = D(M(basal, dt), rhoi);", "const rk accum = M(basal, D(dt, rhoi));",
     ("layerThickness", "temperature"), "fo_temperature"),
    # the remapped tracer taken through a reciprocal
    ("t[k] = D(hTsum[k], h[k]);", "t[k] = M(hTsum[k], D((rk)1.0, h[k]));", "temperature", "fo_temperature"),
    # the new ice's enthalpy as T * (rhoi * cp)
    ("surfaceTracers[iCell] = M(M(surface, rhoi), LI_CP_ICE);",
     "surfaceTracers[iCell] = M(surface, M(rhoi, LI_CP_ICE));", "enthalpy", "fo_enthalpy"),
])
def test_a_mutated_evaluation_order_is_caught(monkeypatch, precision, original, mutated, field, case):
    from woof.hex.mpas_ice import advection_v842
    sources = {p: advection_v842.module_source(p) for p in ("r4", "r8")}
    assert sources[precision].count(original) == 1
    monkeypatch.setattr(advection_v842, "module_source", lambda p="r4": sources[p].replace(original, mutated))
    advection_v842._module_on.cache_clear()
    try:
        row = next(c for c in ao.cases(precision) if c["name"] == case)
        got = ao.run_gpu(row)
        want = ao.load_fixture(precision)[case]
        assert sum(ao.word_mismatches(got[f], want[f]) for f in ((field,) if isinstance(field, str) else field)) > 0
    finally:
        advection_v842._module_on.cache_clear()


def test_unknown_advection_is_refused():
    from woof.hex.mpas_ice import advection_v842
    with pytest.raises(ValueError, match="not a valid option"):
        advection_v842.advect_thickness_tracers(
            mesh={}, geometry={}, velocity={}, thermal={}, dt=1.0, advect_tracers=True,
            config_thickness_advection="fct", config_thermal_solver="temperature", config_ice_density=910.0,
            config_ocean_density=1028.0, config_sea_level=0.0, config_dynamic_thickness=10.0)


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_landice").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MALI oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mali_oracle" / "build.sh"), str(source), str(tmp_path / "build")],
                   check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mali_oracle" / "make_advection_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    for precision in ("r4", "r8"):
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(ao.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert ao.word_mismatches(fresh[key], pinned[key]) == 0, key
