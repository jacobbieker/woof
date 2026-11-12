"""MPAS-Ocean's equation of state and freezing temperature, word for word against v8.4.2, at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's own ``ocn_equation_of_state_density``
('linear' and 'jm', each displacement type MPAS-Ocean calls it with, with and
without the expansion coefficients), ``ocn_freezing_temperature`` and its
salinity derivative, compiled verbatim with gfortran -O0 -ffp-contract=off at
MPAS's default single precision and at its double precision, and driven by
``tools/mpas_ocean_oracle/run_eos.F90``. The outputs are committed
(tests/data/oracles/mpas_ocean_eos, PROVENANCE.md). When ``GFORTRAN`` (or
gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` are set, the fixture itself is
rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance, in either precision.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: JM's clamps on both sides, a one-level
   column, the displaced reference levels, short columns whose lower levels
   keep their words, pressures on both sides of the freezing reference.
4. A gate that cannot fail. A mutated evaluation order, a dropped clamp or
   an off-by-one reference level must produce mismatches.
5. An equation of state v8.4.2 does not carry silently running as another:
   it is refused by name.
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
from woof.verify import mpas_ocean_eos_oracle as oo

ROOT = Path(__file__).resolve().parents[1]


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": oo.input_digest(case, p)
                        for p in oo.PRECISIONS for case in oo.cases()}


def test_fixture_covers_the_branches():
    all_cases = oo.cases()
    assert {case["eosType"] for case in all_cases} == set(oo.EOS_TYPES)
    assert {(k, kind) for k, kind, _, _ in oo.CALLS} >= {(0, "relative"), (1, "absolute"), (1, "relative"),
                                                         (0, "surfaceDisplaced")}
    assert any(case["nVertLevels"] == 1 for case in all_cases)
    clamped = next(c for c in all_cases if c["name"] == "jm_clamped")
    t = clamped["activeTracers"][:, :, clamped["indexT"] - 1]
    s = clamped["activeTracers"][:, :, clamped["indexS"] - 1]
    assert (t < -2.0).any() and (t > 40.0).any() and (s < 0.0).any() and (s > 42.0).any()
    for precision in oo.PRECISIONS:
        fixture = oo.load_fixture(precision)
        assert set(fixture) == {row[0] for row in oo.CASES}
        for case in all_cases:
            out = fixture[case["name"]]
            assert all(np.isfinite(v).all() for v in out.values())
            below = np.arange(case["nVertLevels"])[None, :] >= case["maxLevelCell"][:, None]
            sentinel = case["sentinel"].astype(oo.PRECISIONS[precision])
            if below.any():
                # below maxLevelCell the Fortran writes nothing: the sentinel words stay
                for name in ("density", "thermalExpansionCoeff", "densityDeepest"):
                    assert (out[name][below] == sentinel[below]).all(), (case["name"], name)
            active = ~below
            assert ((out["density"][active] > 990.0) & (out["density"][active] < 1060.0)).all()
            p = case["freezeP"]
            for where in ("open_ocean", "land_ice_cavity"):
                reference = case["configs"][f"config_{where}_freezing_temperature_reference_pressure"]
                assert (p < reference).any() and (p > reference).any() and (p == reference).any()


def test_unported_options_are_refused_by_name():
    from woof.hex.mpas_ocean import eos_v842
    assert eos_v842.eos_type({"config_eos_type": "jm"}) == "jm"
    assert eos_v842.eos_type(None) == "linear"
    with pytest.raises(ConfigurationRefusal) as exc:
        eos_v842.eos_type({"config_eos_type": "wright"})
    assert "config_eos_type" in str(exc.value)


@requires_gpu
def test_an_absolute_displacement_outside_the_column_raises():
    import cupy as cp
    from woof.hex.mpas_ocean import eos_v842
    case = oo.cases()[2]
    mesh = {"maxLevelCell": cp.asarray(case["maxLevelCell"]), "refBottomDepth": cp.asarray(case["refBottomDepth"])}
    with pytest.raises(ValueError, match="k_displaced"):
        eos_v842.equation_of_state_density(
            mesh=mesh, active_tracers=cp.asarray(case["activeTracers"]),
            tracers_surface_value=cp.asarray(case["tracersSurfaceValue"]), index_temperature=1, index_salinity=2,
            density=cp.asarray(case["sentinel"]), k_displaced=0, displacement_type="absolute",
            configs={"config_eos_type": "jm"})


@requires_gpu
@pytest.mark.parametrize("precision", sorted(oo.PRECISIONS))
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", sorted(oo.PRECISIONS))
@pytest.mark.parametrize("original, mutated, case_name, field", [
    # the bulk modulus's salinity term distributed
    ("M(SQ, A(WORK3, WORK4)));", "A(M(SQ, WORK3), M(SQ, WORK4)));", "jm_ocean", "density"),
    # the density's reciprocal replaced by a division
    ("const rk rho = M(M(A(unt0, RHO_S), BULK_MOD), DENOMK);",
     "const rk rho = D(M(A(unt0, RHO_S), BULK_MOD), S(BULK_MOD, p));", "jm_ocean", "density"),
    # exp's argument rounded as a division (the thin surface layers carry exp's bits)
    ("EXP(-M(C(0.025), depth))", "EXP(-D(depth, C(40.0)))", "jm_deep_short_columns", "density"),
    # the thermal expansion's correction regrouped
    ("M(M(M(p, A(unt0, RHO_S)), DKDT), DENOMK)", "M(M(p, A(unt0, RHO_S)), M(DKDT, DENOMK))", "jm_ocean",
     "thermalExpansionCoeff"),
    # JM's upper salinity clamp dropped
    ("eos_max(eos_min(tracerSalt, C(42.0)), C(0.0))", "eos_max(tracerSalt, C(0.0))", "jm_clamped", "density"),
    # the displaced reference level one too shallow
    ("int kTest = type == EOS_RELATIVE ? k + 1 + kDisplaced : kDisplaced;",
     "int kTest = type == EOS_RELATIVE ? k + kDisplaced : kDisplaced;", "jm_ocean", "displacedDensity"),
    # the linear terms added in the other order
    ("const rk rho = A(S(densityref, M(alpha, S(T, Tref))), M(beta, S(Sa, Sref)));",
     "const rk rho = S(A(densityref, M(beta, S(Sa, Sref))), M(alpha, S(T, Tref)));", "linear_configured",
     "density"),
    # the freezing temperature's pS term regrouped
    ("M(M(coeffPS, pressureOffset), salinity[i])", "M(coeffPS, M(pressureOffset, salinity[i]))", "jm_ocean",
     "freezingTemperatureCavity"),
    # the pressure offset's floor at zero dropped
    ("eos_max(S(pressure[i], referencePressure), C(0.0))", "S(pressure[i], referencePressure)", "jm_ocean",
     "freezingTemperatureCavity"),
])
def test_a_mutated_port_is_caught(monkeypatch, precision, original, mutated, case_name, field):
    from woof.hex.mpas_ocean import eos_v842
    real = eos_v842.module_source
    assert real(precision).count(original) == 1
    monkeypatch.setattr(eos_v842, "module_source", lambda p="r4": real(p).replace(original, mutated))
    eos_v842._module_on.cache_clear()
    try:
        case = next(c for c in oo.cases() if c["name"] == case_name)
        got = oo.run_gpu(case, precision)
        assert oo.bit_mismatches(got[field], oo.load_fixture(precision)[case_name][field]) > 0
    finally:
        eos_v842._module_on.cache_clear()


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
                    str(tmp_path / "build"), "--piece", "eos", "--out", str(tmp_path / "fixture")], check=True)
    for precision in oo.PRECISIONS:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.bit_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
