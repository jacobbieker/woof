"""MPAS-Seaice's coupling preparation, word for word against v8.4.2, at RKIND 4 and 8.

The reference is ``seaice_column_coupling_prep`` and
``seaice_column_scale_fluxes``, cut verbatim out of v8.4.2's
``mpas_seaice_column.F`` and compiled with gfortran -O0 -ffp-contract=off at
MPAS's default single precision and at its double precision. The driver is
``tools/mpas_seaice_oracle/run_coupling_prep.F90``; the fixture is
tests/data/oracles/mpas_seaice_coupling_prep (see its PROVENANCE.md). When
``GFORTRAN`` (or gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` are set, the
fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance, in either precision, or
   only under strict numerics: the word-exact gate also runs in fast and
   strict child processes.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch. It must cover:
   - ice and open water (the open-water longwave);
   - the sun up, down, exactly at the horizon, exactly at seaicePuny and
     between the two;
   - the pond freshwater feedback on and off;
   - one category and five;
   - halo cells left alone.
4. A gate that cannot fail. Each of these mutations must produce mismatches:
   - a mutated sun test (three ways), shortwave scaling factor or pond feedback;
   - scaling by division instead of the inverse;
   - x**4 evaluated left to right.
5. An unported option run silently. The ocean mixed layer (the Registry
   default for stand-alone runs) and the column biogeochemistry are refused
   by name.
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
from woof.verify import mpas_seaice_coupling_prep_oracle as oo

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": oo.input_digest(case) for p in BOTH for case in oo.cases(p)}


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_branches(precision):
    from woof.hex.mpas_seaice import coupling_prep_v842 as cpl
    fixture = oo.load_fixture(precision)
    cases = {case["name"]: case for case in oo.cases(precision)}
    assert set(fixture) == set(cases)
    seen = dict.fromkeys(("open", "ice", "sun", "dark", "horizon", "at_puny", "below_puny", "feedback", "halo",
                          "one"), 0)
    for name, out in fixture.items():
        case = cases[name]
        pools, cfg = case["pools"], case["config"]
        solved, nc = int(pools["tracers"]["nCellsSolve"]), case["dims"]["nCells"]
        real = np.dtype(oo.PRECISIONS[precision]).type
        area = pools["tracers_aggregate"]["iceAreaCell"][:solved]
        coszen = pools["shortwave"]["solarZenithAngleCosine"][:solved]
        for key, words in out.items():
            assert np.isfinite(words).all(), (name, key)
            pool, field = key.split("/")
            # halo cells beyond nCellsSolve are left as they came
            assert oo.word_mismatches(words[solved:], pools[pool][field][solved:]) == 0, (name, key)
        seen["halo"] += int(solved < nc)
        seen["one"] += int(case["dims"]["nCategories"] == 1)
        open_water = area == 0
        seen["open"] += int(open_water.sum())
        seen["ice"] += int((~open_water).sum())
        # the open-water longwave, one IEEE rounding per operation as numpy does them
        t = pools["ocean_coupling"]["seaFreezingTemperature"][:solved] + real(cpl.FRESH_WATER_FREEZING_POINT)
        t2 = t * t
        want = -(real(cpl.STEFAN_BOLTZMANN) * (t2 * t2))
        assert oo.word_mismatches(out["atmos_fluxes/longwaveUp"][:solved][open_water], want[open_water]) == 0
        seen["sun"] += int((coszen > real(cpl.PUNY)).sum())
        seen["dark"] += int((coszen <= real(cpl.PUNY)).sum())
        ice = area > 0
        seen["horizon"] += int(((coszen == 0) & ice).sum())
        seen["at_puny"] += int(((coszen == real(cpl.PUNY)) & ice).sum())
        seen["below_puny"] += int(((coszen > 0) & (coszen < real(cpl.PUNY)) & ice).sum())
        bare = out["shortwave/bareIceAlbedoCell"][:solved]
        assert (bare[coszen <= real(cpl.PUNY)] == 0).all()
        moved = out["ponds/pondFreshWaterFlux"][:solved] != pools["ponds"]["pondFreshWaterFlux"][:solved]
        if cfg["config_include_pond_freshwater_feedback"]:
            seen["feedback"] += 1
            assert moved.any()
        else:
            assert not moved.any()
    assert all(seen.values()), seen


def test_unported_options_are_refused_by_name(monkeypatch):
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex.mpas_seaice import coupling_prep_v842 as cpl
    assert cpl.DEFAULT_CONFIGS["config_use_ocean_mixed_layer"] is True
    # the mixed layer (the Registry default) needs glibc's atan: refused without the opt-in
    monkeypatch.delenv("WOOF_GLIBC239_LIBM64", raising=False)
    with pytest.raises(ConfigurationRefusal, match="config_use_ocean_mixed_layer"):
        cpl.check_config({})
    with pytest.raises(ConfigurationRefusal, match="config_atmos_boundary_method"):
        cpl.check_config({"config_atmos_boundary_method": "constant"})
    with pytest.raises(ConfigurationRefusal, match="config_use_column_biogeochemistry"):
        cpl.check_config({"config_use_ocean_mixed_layer": False, "config_use_column_biogeochemistry": True})
    cpl.check_config(dict(oo.BASE_CONFIG))


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert len(report) == len(oo.CASES) and not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_v842_oracle_word_for_word"], numerics, exact)


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
@pytest.mark.parametrize("original, mutated, field", [
    # the sun counted as up at the horizon
    ("solarZenithAngleCosine[iCell] > puny;", "solarZenithAngleCosine[iCell] > -puny;",
     "shortwave/bareIceAlbedoCell"),
    # the sun counted as up at exactly seaicePuny
    ("solarZenithAngleCosine[iCell] > puny;", "solarZenithAngleCosine[iCell] >= puny;",
     "shortwave/bareIceAlbedoCell"),
    # the sun counted as up below seaicePuny
    ("solarZenithAngleCosine[iCell] > puny;", "solarZenithAngleCosine[iCell] > ZERO;",
     "shortwave/bareIceAlbedoCell"),
    # the shortwave scaling factor summed from the infrared end
    ("A(A(A(M(shortwaveVisibleDirectDown[iCell], S(ONE, visDir)),\n"
     "              M(shortwaveVisibleDiffuseDown[iCell], S(ONE, visDif))),\n"
     "            M(shortwaveIRDirectDown[iCell], S(ONE, irDir))),\n"
     "          M(shortwaveIRDiffuseDown[iCell], S(ONE, irDif)));",
     "A(M(shortwaveVisibleDirectDown[iCell], S(ONE, visDir)),\n"
     "              A(M(shortwaveVisibleDiffuseDown[iCell], S(ONE, visDif)),\n"
     "            A(M(shortwaveIRDirectDown[iCell], S(ONE, irDir)),\n"
     "          M(shortwaveIRDiffuseDown[iCell], S(ONE, irDif)))));",
     "shortwave/shortwaveScalingFactor"),
    # the pond feedback's density divided by the step first
    ("D(M(pondFreshWaterFlux[iCell], densityFreshwater), dt)", "M(pondFreshWaterFlux[iCell], D(densityFreshwater, dt))",
     "ponds/pondFreshWaterFlux"),
    # the fluxes divided by the ice area rather than multiplied by its inverse
    ("#define CP_SCALE(f) f[iCell] = M(f[iCell], iceAreaInverse);",
     "#define CP_SCALE(f) f[iCell] = D(f[iCell], iceAreaCell[iCell]);", "ocean_fluxes/oceanHeatFlux"),
    # x**4 evaluated left to right
    ("longwaveUp[iCell] = -M(stefanBoltzmann, M(t2, t2));", "longwaveUp[iCell] = -M(stefanBoltzmann, M(M(t2, t), t));",
     "atmos_fluxes/longwaveUp"),
])
def test_a_mutated_kernel_is_caught(monkeypatch, precision, original, mutated, field):
    from woof.hex.mpas_seaice import coupling_prep_v842 as cpl
    real = cpl.module_source
    assert real(precision).count(original) == 1
    monkeypatch.setattr(cpl, "module_source", lambda p="r4": real(p).replace(original, mutated))
    cpl._module_on.cache_clear()
    try:
        want = oo.load_fixture(precision)
        mismatches = sum(oo.word_mismatches(oo.run_gpu(case)[field], want[case["name"]][field])
                         for case in oo.cases(precision))
        assert mismatches > 0
    finally:
        cpl._module_on.cache_clear()


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
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_coupling_prep_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
