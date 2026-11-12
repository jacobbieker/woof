"""MALI's thermal solver and basal melt, word for word against v8.4.2.

The reference is MALI v8.4.2's own land-ice modules, compiled verbatim with
gfortran -O0 -ffp-contract=off at RKIND 4. ``tools/mali_oracle/run_thermal_solver.F90``
makes a restart's ``li_thermal_init``, then ``li_basal_melt_floating_ice`` and
``li_thermal_solver`` as ``li_time_integration_fe`` does: the masks, the SIA
heat dissipation, the basal friction, the temperature or enthalpy column
solve (matrix elements and ``tridiag_solver``), the conductive fluxes, the
grounded and floating basal melt. The oracle's outputs are committed
(tests/data/oracles/mali_thermal_solver, PROVENANCE.md), so the GPU comparison
runs without gfortran. When the oracle can be built here (``GFORTRAN`` or
gfortran on PATH, and ``WOOF_MPAS_V842_SOURCE`` naming the MPAS-Model v8.4.2
tree), the fixture itself is regenerated and compared too.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MALI only to a tolerance. Every output word, the
   masks included, must be bit-identical, halo cells too.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: floating and grounded columns, each of
   the bed's boundary rows, melt and freeze-on, drainage, each floating
   melt option.
4. A gate that cannot fail. Mutating one evaluation order in the kernels must
   produce mismatches.
5. Outputs that depend on the model's uninitialised scratch fields: the
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
from woof.verify import mali_thermal_solver_oracle as to

ROOT = Path(__file__).resolve().parents[1]
F = np.float32


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (to.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {**{case["name"]: to.input_digest(case) for case in to.cases()},
                        **{case["name"] + "/r8": to.input_digest(case) for case in to.cases("r8")},
                        **{case["name"] + "/" + to.R8_GLIBC239: to.input_digest(case)
                           for case in to.cases(to.R8_GLIBC239)}}


def test_fixture_covers_the_branches():
    fixture = to.load_fixture()
    cases = {case["name"]: case for case in to.cases()}
    assert set(fixture) == set(cases)
    assert {c["thermalSolver"] for c in cases.values()} == set(to.THERMAL)
    assert {c["floatMelt"] for c in cases.values()} == set(to.FLOAT_MELT)
    assert {c["velocitySolver"] for c in cases.values()} == {"sia", "none"}
    assert {c["airSource"] for c in cases.values()} == {"file", "lapse"}
    assert {c["calculateBmb"] for c in cases.values()} == {False, True}
    for name, out in fixture.items():
        case = cases[name]
        # the oracle ran clean: no error flag, no logged error
        assert not any(out[k][0] for k in ("errInit", "errFloat", "errSolver", "logErrors"))
        assert all(np.isfinite(v).all() for v in out.values())
        assert case["nCellsSolve"] < case["nCells"]
        ncs = case["nCellsSolve"]
        h = case["thickness"][:ncs]
        active = h > case["thermalThickness"]
        floating = (out["cellMask"][:ncs] & 4) == 4
        assert (active & floating).any() and (active & ~floating).any() and (~active).any()
        if case["thermalSolver"] == "none":
            continue
        rhoi = case["iceDensity"]
        pmp_bed = -((F(9.7456e-8) * rhoi) * F(9.80616)) * h
        bottom = case["temperature"][:ncs, -1] - F(273.15)
        temperate = np.abs(bottom - to._pmp_column(rhoi, h, case["layerCenterSigma"][-1])) < F(0.001)
        melting = np.abs((case["basalTemperature"][:ncs] - F(273.15)) - pmp_bed) < F(0.001)
        grounded = active & ~floating
        # every boundary row of the grounded bed: melting, frozen, and (enthalpy) a temperate bottom layer
        assert (grounded & melting & ~temperate).any() and (grounded & ~melting & ~temperate).any()
        if case["thermalSolver"] == "enthalpy":
            assert (grounded & temperate).any()
            # drainage of water above 0.01, and temperate layers below it
            wf = out["waterfrac"][:ncs]
            assert (wf == F(0.01)).any() and ((wf > 0) & (wf < F(0.01))).any()
        # freeze-on onto a dry bed resets the basal temperature below the melting point
        reset = out["basalTemperature"][:ncs] == (pmp_bed - F(0.01)) + F(273.15)
        assert (reset & grounded).any()
        g = out["groundedBasalMassBal"]
        if case["calculateBmb"]:
            assert (g > 0).any() and (g < 0).any()
        else:
            assert not g.any()
    # the floating melt options produce melt; Seroussi's cap binds
    for name in ("temperature_sia_constant", "enthalpy_sia_mismip", "temperature_none_seroussi", "no_thermal_mismip"):
        assert (fixture[name]["floatingBasalMassBal"] < 0).any()
    rhoi = cases["temperature_none_seroussi"]["iceDensity"]
    cap = (F(-1.0) * F(50.0)) * rhoi / F(31536000.0)
    assert (fixture["temperature_none_seroussi"]["floatingBasalMassBal"] == cap).any()
    # air temperatures on both sides of the freezing point reach min(0, T)
    t = cases["temperature_sia_constant"]["surfaceAirTemperature"]
    assert (t > 273.15).any() and (t < 273.15).any()


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = to.measure(precision=precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
@pytest.mark.parametrize("original, mutated, field, case", [
    # the dissipation's base multiplied in another order
    ("M(M(M(M(layerCenterSigma[k], rhoi), LI_GRAVITY), thicknessEdge), slope)",
     "M(M(M(layerCenterSigma[k], M(rhoi, LI_GRAVITY)), thicknessEdge), slope)", "heatDissipation",
     "temperature_sia_constant"),
    # the edge weight's area division taken first
    ("D(M(M(LI_C(0.25), dcEdge[iEdge]), dvEdge[iEdge]), areaCell[iCell])",
     "M(M(LI_C(0.25), dcEdge[iEdge]), D(dvEdge[iEdge], areaCell[iCell]))", "heatDissipation",
     "enthalpy_sia_mismip"),
    # the temperature matrix factor's divisions merged
    ("D(D(M(deltat, LI_ICE_CONDUCTIVITY), M(rhoi, LI_CP_ICE)), M(h, h))",
     "D(M(deltat, LI_ICE_CONDUCTIVITY), M(M(rhoi, LI_CP_ICE), M(h, h)))", ("temperature", "basalConductiveFlux"),
     None),   # every case: at RKIND 8 the change shows in the temperature_none case only
    # the tridiagonal back substitution's product reassociated
    ("bb[i] = D(S(y[i], M(a[i], bb[i - 1])), S(b[i], M(a[i], aa[i - 1])));",
     "bb[i] = M(S(y[i], M(a[i], bb[i - 1])), D((rk)1.0, S(b[i], M(a[i], aa[i - 1]))));", "temperature",
     "temperature_none_seroussi"),
    # the enthalpy diffusivity's average in another order
    ("diffusivity[k] = A(M(avgFactor, diffusivityCold), M(S((rk)1.0, avgFactor), diffusivityTemperate));",
     "diffusivity[k] = S(A(M(avgFactor, diffusivityCold), diffusivityTemperate), "
     "M(avgFactor, diffusivityTemperate));", "enthalpy", "enthalpy_sia_mismip"),
    # the internal melt's energy formed as rhoi * (cp * dT)
    ("M(M(M(rhoi, LI_CP_ICE), S(T[k], pmp)), layerThickness)",
     "M(M(rhoi, M(LI_CP_ICE, S(T[k], pmp))), layerThickness)", "groundedBasalMassBal", "temperature_sia_constant"),
    # MISMIP's cavity scale taken as a reciprocal product
    ("LI_TANH(D(hCavity, LI_C(75.0)))", "LI_TANH(M(hCavity, D((rk)1.0, LI_C(75.0))))", "floatingBasalMassBal",
     "no_thermal_mismip"),
    # the basal friction's speed squared after the scale
    ("M(M(betaSolve[iCell], M(basalSpeed[iCell], basalSpeed[iCell])), LI_SCYR)",
     "M(M(M(betaSolve[iCell], basalSpeed[iCell]), basalSpeed[iCell]), LI_SCYR)", "basalFrictionFlux",
     "enthalpy_none_file"),
])
def test_a_mutated_evaluation_order_is_caught(monkeypatch, precision, original, mutated, field, case):
    from woof.hex.mpas_ice import thermal_solver_v842
    sources = {p: thermal_solver_v842.module_source(p) for p in ("r4", "r8")}
    assert sources[precision].count(original) == 1
    monkeypatch.setattr(thermal_solver_v842, "module_source", lambda p="r4": sources[p].replace(original, mutated))
    thermal_solver_v842._module_on.cache_clear()
    try:
        # the case by its row: at RKIND 8 the refused floating melts run under other names
        rows = to.cases(precision)
        if case is not None:
            rows = [rows[[r[0] for r in to.CASES].index(case)]]
        fields = (field,) if isinstance(field, str) else field
        fixture = to.load_fixture(precision)
        total = sum(to.word_mismatches(got[f], fixture[row["name"]][f])
                    for row in rows for got in (to.run_gpu(row),) for f in fields)
        assert total > 0
    finally:
        thermal_solver_v842._module_on.cache_clear()


@requires_gpu
def test_the_platform_tanh_is_caught_at_rkind8(monkeypatch):
    # CUDA's (or Metal's) own binary64 tanh is not glibc's: the fixture tells them apart
    from woof.hex.mpas_ice import thermal_solver_v842
    source = thermal_solver_v842.module_source("r8")
    original = "#define LI_TANH(x) g64_tanh::cr_tanh(x)"
    assert source.count(original) == 1
    monkeypatch.setattr(thermal_solver_v842, "module_source",
                        lambda p="r4": source.replace(original, "#define LI_TANH(x) tanh(x)"))
    thermal_solver_v842._module_on.cache_clear()
    try:
        row = next(c for c in to.cases("r8") if c["name"] == "no_thermal_mismip")
        got = to.run_gpu(row)
        assert to.word_mismatches(got["floatingBasalMassBal"],
                                  to.load_fixture("r8")[row["name"]]["floatingBasalMassBal"]) > 0
    finally:
        thermal_solver_v842._module_on.cache_clear()


def test_unknown_options_are_refused():
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex.mpas_ice import thermal_solver_v842 as ts
    with pytest.raises(ConfigurationRefusal):
        ts.basal_melt_floating_ice(mesh={}, geometry={}, config_basal_mass_bal_float="ismip6",
                                   config_thermal_thickness=1.0, config_ice_density=910.0,
                                   config_ocean_density=1028.0, config_sea_level=0.0, config_dynamic_thickness=10.0)
    with pytest.raises(ValueError, match="seroussi_period"):
        ts.basal_melt_floating_ice(mesh={}, geometry={}, config_basal_mass_bal_float="seroussi",
                                   config_basal_mass_bal_seroussi_period=0.0, config_thermal_thickness=1.0,
                                   config_ice_density=910.0, config_ocean_density=1028.0, config_sea_level=0.0,
                                   config_dynamic_thickness=10.0)
    with pytest.raises(ConfigurationRefusal):
        ts.thermal_solver(mesh={}, geometry={}, velocity={}, thermal={}, deltat=1.0, config_thermal_solver="sia",
                          config_thermal_thickness=1.0, config_ice_density=910.0, config_ocean_density=1028.0,
                          config_sea_level=0.0, config_dynamic_thickness=10.0)
    with pytest.raises(ValueError, match="invalid deltat"):
        ts.thermal_solver(mesh={}, geometry={}, velocity={}, thermal={}, deltat=0.0,
                          config_thermal_solver="temperature", config_thermal_thickness=1.0,
                          config_ice_density=910.0, config_ocean_density=1028.0, config_sea_level=0.0,
                          config_dynamic_thickness=10.0)


@requires_gpu
def test_rkind8_refuses_the_floating_melt_without_a_matched_libm(monkeypatch):
    from woof.hex.errors import ConfigurationRefusal
    monkeypatch.delenv("WOOF_GLIBC239_LIBM64", raising=False)
    row = next(c for c in to.cases("r8") if c["name"] == "enthalpy_sia_mismip")
    with pytest.raises(ConfigurationRefusal, match="RKIND 8.*WOOF_GLIBC239_LIBM64=1"):
        to.run_gpu(dict(row, floatMelt="seroussi"))


@pytest.fixture
def glibc239(monkeypatch):
    """The opt-in glibc 2.39 binary64 sin/cos/atan transcription (LGPL, source checkout only)."""
    from woof.hex.mpas_ice import thermal_solver_v842
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")
    thermal_solver_v842._module_on.cache_clear()
    yield
    thermal_solver_v842._module_on.cache_clear()


def test_the_glibc239_fixture_runs_seroussi_at_rkind8():
    fixture = to.load_fixture(to.R8_GLIBC239)
    cases = to.cases(to.R8_GLIBC239)
    assert {c["name"] for c in cases} == set(fixture) == {"temperature_none_seroussi"}
    assert all(c["precision"] == "r8" and c["floatMelt"] == "seroussi" for c in cases)
    melt = fixture["temperature_none_seroussi"]["floatingBasalMassBal"]
    assert melt.dtype == np.float64 and np.count_nonzero(melt) > 0


@requires_gpu
def test_rkind8_seroussi_matches_the_oracle_under_the_glibc239_opt_in(glibc239):
    report = to.measure(precision=to.R8_GLIBC239)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("original, mutated", [
    # glibc's cos in place of its sin: the r8 seroussi path runs, and is held
    ("#define LI_SIN(x) g239::g239_sin(x)", "#define LI_SIN(x) g239::g239_cos(x)"),
    # the phase left out of the sine's argument
    ("D(daysSinceStart, LI_C(365.0))), M(twoPi, serPhase)", "D(daysSinceStart, LI_C(365.0))), (rk)0.0"),
])
def test_a_mutated_rkind8_seroussi_is_caught(monkeypatch, glibc239, original, mutated):
    from woof.hex.mpas_ice import thermal_solver_v842
    source = thermal_solver_v842.module_source("r8")
    assert source.count(original) == 1
    monkeypatch.setattr(thermal_solver_v842, "module_source", lambda p="r4": source.replace(original, mutated))
    thermal_solver_v842._module_on.cache_clear()
    row = to.cases(to.R8_GLIBC239)[0]
    got = to.run_gpu(row)
    want = to.load_fixture(to.R8_GLIBC239)[row["name"]]["floatingBasalMassBal"]
    assert to.word_mismatches(got["floatingBasalMassBal"], want) > 0


def test_rkind8_runs_mismip_with_the_correctly_rounded_tanh():
    # 'mismip' is an RKIND 8 case of its own (glibc 2.44's tanh is CORE-MATH's), and
    # the r8 unit calls that transcription, not the platform tanh
    from woof.hex.mpas_ice import thermal_solver_v842
    assert "enthalpy_sia_mismip" in {c["name"] for c in to.cases("r8")}
    assert "mismip" not in thermal_solver_v842.R8_REFUSED_FLOAT_MELT
    source = thermal_solver_v842.module_source("r8")
    assert "#define LI_TANH(x) g64_tanh::cr_tanh(x)" in source and "namespace g64_tanh" in source


@requires_gpu
def test_a_column_past_the_temperature_thresholds_raises():
    row = next(c for c in to.cases() if c["name"] == "temperature_sia_constant")
    # a column at 150 K (-123 deg C) stays below -100 deg C through a month
    row = dict(row, temperature=np.full_like(row["temperature"], 150.0),
               surfaceAirTemperature=np.full_like(row["surfaceAirTemperature"], 150.0))
    with pytest.raises(RuntimeError, match="100"):
        to.run_gpu(row)


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
    subprocess.run([sys.executable, str(ROOT / "tools" / "mali_oracle" / "make_thermal_solver_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    for precision in ("r4", "r8", to.R8_GLIBC239):
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(to.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert to.word_mismatches(fresh[key], pinned[key]) == 0, key
