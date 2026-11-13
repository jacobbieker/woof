"""MPAS-Ocean's ocn_tend_tracer (active tracers), word for word against v8.4.2, at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's own ``ocn_tend_tracer``, with
``mpas_ocn_tendency.F`` compiled verbatim against the real tracer modules
(advection, mixing, surface flux, short-wave absorption, non-local flux),
bulk forcing, diagnostics and equation of state, with gfortran -O0
-ffp-contract=off at MPAS's default single precision and at its double
precision, driven by ``tools/mpas_ocean_oracle/run_tend_tracer.F90``. The
outputs are committed (tests/data/oracles/mpas_ocean_tend_tracer,
PROVENANCE.md). When ``GFORTRAN`` (or gfortran on PATH) and
``WOOF_MPAS_V842_SOURCE`` are set, the fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance, in either precision.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: the bulk forcing with and without the
   bulk thickness flux, the sea-ice salt flux limited, the surface flux's
   remainder sent to the bottom level, Jerlov absorption with the
   buoyancy-depth search ending inside and below the column, the budgets,
   the surface flux, the active tracers or the whole tendency switched off.
4. A gate that cannot fail: a mutated evaluation order or branch must produce
   mismatches.
5. An option this port does not carry silently skipped: refused by name.
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
from woof.verify import mpas_ocean_tend_tracer_oracle as oo

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")
TEND = "tracersTend/activeTracersTend"


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": oo.input_digest(case) for p in BOTH for case in oo.cases(p)}


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_branches(precision):
    fixture = oo.load_fixture(precision)
    cases = {case["name"]: case for case in oo.cases(precision)}
    assert set(fixture) == set(cases)
    limited = deep_search = shallow_search = remainder = 0
    for name, out in fixture.items():
        case = cases[name]
        c = case["config"]
        tend_before = case["pools"]["tracersTend"]["activeTracersTend"]
        changed = oo.word_mismatches(out[TEND], tend_before)
        if c["config_disable_tr_all_tend"] or not c["config_use_activeTracers"]:
            assert changed == 0, name
            continue
        assert changed > 0 and np.isfinite(out[TEND]).all(), name
        nc = case["dims"]["nCells"]
        removed = out["tracersSurfaceFlux/activeTracersSurfaceFluxRemoved"][:nc, 1]
        if c["config_use_activeTracers_surface_bulk_forcing"]:
            limited += int((removed != 0).sum())
        fraction = case["pools"]["forcing"]["fractionAbsorbed"][:nc]
        mlc = case["pools"]["mesh"]["maxLevelCell"][:nc]
        remainder += int(sum(1.0 - fraction[i, :mlc[i]].sum() > 0 for i in range(nc)))
        if c["config_sw_absorption_type"] == "jerlov":
            h = case["pools"]["state"]["layerThickness"][:nc].astype(np.float64)
            column = np.array([h[i, :mlc[i]].sum() for i in range(nc)])
            depth = abs(c["config_surface_buoyancy_depth"])
            deep_search += int((column <= depth).sum())
            shallow_search += int((column > depth).sum())
        if c["config_compute_active_tracer_budgets"]:
            for field in oo.BUDGET_FIELDS:
                assert np.isfinite(out[f"diagnostics/{field}"]).all()
    assert limited > 0 and remainder > 0 and deep_search > 0 and shallow_search > 0


def test_options_not_ported_are_refused_by_name():
    from woof.hex.mpas_ocean import tend_tracer_v842 as tt
    base = dict(oo.BASE_CONFIG)
    tt.check_config(base)
    for knob, value in (("config_use_cvmix_kpp", True), ("config_use_activeTracers_surface_restoring", True),
                        ("config_use_activeTracers_interior_restoring", True),
                        ("config_use_activeTracers_exponential_decay", True),
                        ("config_use_activeTracers_idealAge_forcing", True),
                        ("config_use_activeTracers_ttd_forcing", True), ("config_land_ice_flux_mode", "coupled"),
                        ("config_use_frazil_ice_formation", True), ("config_sw_absorption_type", "ohlmann00"),
                        ("config_use_standardGM", True), ("config_monotonic", False)):
        with pytest.raises(ConfigurationRefusal) as exc:
            tt.check_config(dict(base, **{knob: value}))
        assert knob in str(exc.value)
    with pytest.raises(ConfigurationRefusal, match="tracer group"):
        tt.check_config(base, groups=("activeTracers", "debugTracers"))
    with pytest.raises(ValueError, match="bulk forcing off"):
        tt.check_config(dict(base, config_sw_absorption_type="jerlov"))


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
@pytest.mark.parametrize("original, mutated, case_name, field", [
    # the heat fluxes summed in the other order (the long-wave terms cancel)
    ("A(A(A(A(latentHeatFlux[iCell], sensibleHeatFlux[iCell]), longWaveHeatFluxUp[iCell]),\n"
     "                  longWaveHeatFluxDown[iCell]), seaIceHeatFlux[iCell])",
     "A(latentHeatFlux[iCell], A(sensibleHeatFlux[iCell], A(longWaveHeatFluxUp[iCell],\n"
     "                  A(longWaveHeatFluxDown[iCell], seaIceHeatFlux[iCell]))))",
     "bulk_jerlov_del2_budgets", "tracersSurfaceFlux/activeTracersSurfaceFlux"),
    # the removed salt flux as the flux less the allowed part
    ("M(M(S(C(1.0), D(allowedSalt, requiredSalt)), seaIceSalinityFlux[iCell]), sfluxFactor)",
     "S(seaIceSalinityFlux[iCell], M(D(allowedSalt, requiredSalt), seaIceSalinityFlux[iCell]))",
     "bulk_salt_limit_del4", "tracersSurfaceFlux/activeTracersSurfaceFluxRemoved"),
    # the runoff flux divided by rho_sw first
    ("D(M(riverRunoffFlux[iCell], tt_max(T1, C(0.0))), rhoSw)",
     "M(riverRunoffFlux[iCell], D(tt_max(T1, C(0.0)), rhoSw))", "bulk_jerlov_del2_budgets",
     "tracersSurfaceFlux/activeTracersSurfaceFluxRunoff"),
    # the surface flux's remainder kept from the bottom level
    ("    if (top > 0 && remaining > C(0.0)) {", "    if (false) {", "advection_only", TEND),
    # Jerlov's second band weighted as the whole less the first band's weight
    ("M(S(C(1.0), rfac), EXP(-D(depth, depth2)))", "S(EXP(-D(depth, depth2)), M(rfac, EXP(-D(depth, depth2))))",
     "bulk_jerlov_del2_budgets", "diagnostics/penetrativeTemperatureFluxOBL"),
    # the buoyancy-depth level taken where the search ends, without v8.4.2's special cases
    ("const int depLev = (kf == top || kf == 1) ? 2 : kf;", "const int depLev = kf > top ? top : kf;",
     "bulk_jerlov_del2_budgets", "diagnostics/penetrativeTemperatureFluxOBL"),
])
def test_a_mutated_port_is_caught(monkeypatch, precision, original, mutated, case_name, field):
    from woof.hex.mpas_ocean import tend_tracer_v842
    real = tend_tracer_v842.module_source
    assert real(precision).count(original) == 1
    monkeypatch.setattr(tend_tracer_v842, "module_source", lambda p="r4": real(p).replace(original, mutated))
    tend_tracer_v842._module_on.cache_clear()
    try:
        case = next(c for c in oo.cases(precision) if c["name"] == case_name)
        got = oo.run_gpu(case)
        assert oo.word_mismatches(got[field], oo.load_fixture(precision)[case_name][field]) > 0
    finally:
        tend_tracer_v842._module_on.cache_clear()


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
                    str(tmp_path / "build"), "--piece", "tend_tracer", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
