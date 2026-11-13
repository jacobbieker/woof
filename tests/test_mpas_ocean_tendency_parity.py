"""MPAS-Ocean full momentum and thickness tendencies, word for word against v8.4.2.

The reference is MPAS-Model v8.4.2's own ``ocn_tend_vel`` and ``ocn_tend_thick``
(``mpas_ocn_tendency.F``, with every term they call compiled verbatim:
Coriolis, vertical advection, pressure gradient, hmix, bulk forcing, surface
stress, explicit bottom drag, Rayleigh friction, thickness advection and
surface flux), and ``ocn_vel_forcing_tend`` and ``ocn_thick_surface_flux_tend``
on their own, built with gfortran -O0 -ffp-contract=off at RKIND 4 and 8 and
driven by ``tools/mpas_ocean_oracle/run_tendency.F90``. The outputs are
committed (tests/data/oracles/mpas_ocean_tendency, PROVENANCE.md). When
``GFORTRAN`` (or gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` are set, the
fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A composition that adds the terms in another order, or a term that agrees
   with MPAS only to a tolerance, in either precision.
2. A case generator whose bytes drift. The input digests are pinned.
3. A gate that cannot fail. A mutated forcing term or a reordered composition
   must produce mismatches.
4. An unported switch (land-ice fluxes, frazil, the RKIND 8 bulk wind stress
   without the glibc 2.39 opt-in,
   the bottom-level terms on a mesh with a dry edge) silently running as
   something else. Each is refused by name.
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
from woof.verify import mpas_ocean_tendency_oracle as oo

ROOT = Path(__file__).resolve().parents[1]


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": oo.input_digest(case, p)
                        for p in (*oo.PRECISIONS, *oo.EXTRA_PRECISIONS) for case in oo.cases()
                        if p in case["precisions"]}


def test_fixture_covers_the_branches():
    all_cases = oo.cases()
    for flag in oo.FLAGS:
        assert {case["configs"][flag] for case in all_cases} == {True, False}, flag
    assert {case["rk4"] for case in all_cases} == {True, False}
    for case in all_cases:
        z = -np.cumsum(case["layerThicknessEdge"], axis=1)
        a = case["surfaceFluxAttenuationCoefficient"][case["cellsOnEdge"] - 1].mean(axis=1)
        # the transmission's exp(-100) floor is reached
        assert (z / a[:, None] < -100).any()
        column = [case["fractionAbsorbed"][i, :case["maxLevelCell"][i]].sum() for i in range(case["nCells"])]
        assert min(column) < 1 < max(column)
        if case["configs"]["config_use_explicit_bottom_drag"] or case["configs"]["config_Rayleigh_bottom_friction"]:
            assert (case["maxLevelEdgeTop"] >= 1).all()
    # the bulk wind stress at RKIND 8 is the opt-in's fixture, not the default one
    assert any("r8" not in case["precisions"] and oo.R8_GLIBC239 in case["precisions"]
               and case["configs"]["config_use_bulk_wind_stress"] for case in all_cases)
    for precision in (*oo.PRECISIONS, *oo.EXTRA_PRECISIONS):
        fixture = oo.load_fixture(precision)
        assert set(fixture) == {c["name"] for c in all_cases if precision in c["precisions"]}
        assert all(np.isfinite(v).all() for c in fixture.values() for v in c.values())


def test_unported_switches_are_refused_by_name(monkeypatch):
    from woof.hex.mpas_ocean import forcing_v842 as fv
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "0")
    base = dict(fv.DEFAULT_CONFIGS)
    for knob, value, precision in (("config_land_ice_flux_mode", "standalone", "r4"),
                                   ("config_use_frazil_ice_formation", True, "r4"),
                                   ("config_use_bulk_wind_stress", True, "r8")):
        with pytest.raises(ConfigurationRefusal) as exc:
            fv.refuse_unported({**base, knob: value}, precision)
        assert knob in str(exc.value)
    fv.refuse_unported({**base, "config_use_bulk_wind_stress": True}, "r4")
    with pytest.raises(ValueError):
        fv._configs({"config_not_read": 1})


@requires_gpu
def test_a_dry_edge_refuses_the_bottom_terms():
    import cupy as cp
    from woof.hex.mpas_ocean import forcing_v842 as fv
    case = next(c for c in oo.cases() if c["name"] == "zmid_del2_split")
    assert (case["maxLevelEdgeTop"] == 0).any()
    dev = {k: cp.asarray(np.ascontiguousarray(case[k], np.int32)) for k in ("maxLevelEdgeTop", "cellsOnEdge",
                                                                            "edgeMask")}
    real = {k: cp.asarray(case[k].astype(np.float32)) for k in ("normalVelocity", "kineticEnergyCell",
                                                                "layerThicknessEdge", "surfaceStressIn",
                                                                "surfaceFluxAttenuationCoefficient",
                                                                "tendVelForcing")}
    for knob in ("config_use_explicit_bottom_drag", "config_Rayleigh_bottom_friction"):
        configs = {knob: True, "config_Rayleigh_friction": True}
        with pytest.raises(ConfigurationRefusal) as exc:
            fv.vel_forcing_tend(mesh=dev, normal_velocity=real["normalVelocity"],
                                surface_flux_attenuation_coefficient=real["surfaceFluxAttenuationCoefficient"],
                                surface_stress=real["surfaceStressIn"], kinetic_energy_cell=real["kineticEnergyCell"],
                                layer_thickness_edge=real["layerThicknessEdge"], tend=real["tendVelForcing"],
                                configs=configs)
        assert knob in str(exc.value)


def test_the_glibc_opt_in_lifts_the_rkind_8_bulk_wind_stress(monkeypatch):
    from woof.hex.mpas_ocean import forcing_v842 as fv
    from woof.hex.mpas_ice import rkind
    configs = {**fv.DEFAULT_CONFIGS, "config_use_bulk_wind_stress": True}
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")
    if not rkind.trig64():
        pytest.skip("the opt-in transcription is not in this checkout")
    fv.refuse_unported(configs, "r8")
    assert "g239_cos" in fv.module_source("r8") and "BW_COS(angleEdge" in fv.module_source("r8")


@requires_gpu
@pytest.mark.parametrize("precision", sorted(oo.PRECISIONS) + [oo.R8_GLIBC239])
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    from woof.hex.mpas_ice import rkind
    if precision == oo.R8_GLIBC239 and not rkind.trig64():
        pytest.skip("the RKIND 8 bulk wind stress needs WOOF_GLIBC239_LIBM64=1")
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", sorted(oo.PRECISIONS))
@pytest.mark.parametrize("module_name, original, mutated, case_name, field", [
    # the surface-stress term's divisions regrouped
    ("forcing_v842",
     "D(D(M(M((rk)edgeMask[base + k], ss), S(tTop, tBot)), rhoSw), h)",
     "D(M(M((rk)edgeMask[base + k], ss), S(tTop, tBot)), M(rhoSw, h))",
     "jdens_drag_rayleigh", "tendVelForcing"),
    # the bottom drag's KE sum regrouped with the coefficient
    ("forcing_v842",
     "tend[p] = S(tend[p], D(M(M(coef, drag), normalVelocity[p]), layerThicknessEdge[p]));",
     "tend[p] = S(tend[p], M(coef, D(M(drag, normalVelocity[p]), layerThicknessEdge[p])));",
     "jdens_drag_rayleigh", "tendVelForcing"),
    # the bulk freshwater flux summed in another order
    ("forcing_v842",
     "D(A(A(A(A(snowFlux[iCell], rainFlux[iCell]), evaporationFlux[iCell]), seaIceFreshWaterFlux[iCell]),",
     "D(A(A(A(A(rainFlux[iCell], evaporationFlux[iCell]), snowFlux[iCell]), seaIceFreshWaterFlux[iCell]),",
     "zmid_del2_split", "surfaceThicknessFlux"),
])
def test_a_mutated_port_is_caught(monkeypatch, precision, module_name, original, mutated, case_name, field):
    import importlib
    module = importlib.import_module(f"woof.hex.mpas_ocean.{module_name}")
    real = module.module_source
    assert real(precision).count(original) == 1
    monkeypatch.setattr(module, "module_source", lambda p="r4": real(p).replace(original, mutated))
    module._module_on.cache_clear()
    try:
        case = next(c for c in oo.cases() if c["name"] == case_name)
        got = oo.run_gpu(case, precision)
        assert oo.bit_mismatches(got[field], oo.load_fixture(precision)[case_name][field]) > 0
    finally:
        module._module_on.cache_clear()


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", oo.R8_GLIBC239])
def test_a_swapped_wind_rotation_is_caught(monkeypatch, precision):
    """The edge's cos and sin swapped in the bulk wind stress (RKIND 8 under the opt-in)."""
    from woof.hex.mpas_ice import rkind
    from woof.hex.mpas_ocean import forcing_v842 as fv
    if precision == oo.R8_GLIBC239 and not rkind.trig64():
        pytest.skip("the RKIND 8 bulk wind stress needs WOOF_GLIBC239_LIBM64=1")
    base = "r8" if precision == oo.R8_GLIBC239 else precision
    real = fv.module_source
    original = "M(BW_COS(angleEdge[iEdge]), zonalAverage)"
    assert real(base).count(original) == 1
    monkeypatch.setattr(fv, "module_source",
                        lambda p="r4": real(p).replace(original, "M(BW_SIN(angleEdge[iEdge]), zonalAverage)"))
    fv._module_on.cache_clear()
    try:
        case = next(c for c in oo.cases() if c["name"] == "jts_rk4_bulk_drag")
        got = oo.run_gpu(case, precision)
        assert oo.bit_mismatches(got["surfaceStress"], oo.load_fixture(precision)[case["name"]]["surfaceStress"]) > 0
    finally:
        fv._module_on.cache_clear()


@requires_gpu
def test_a_reordered_composition_is_caught(monkeypatch):
    """Adding the pressure gradient before the vertical advection changes words."""
    from woof.hex.mpas_ocean import forcing_v842 as fv
    from woof.hex.mpas_ocean import pgrad_hmix_v842, tend_v842
    real_vadv, real_pgrad = tend_v842.vel_vadv_tend, pgrad_hmix_v842.vel_pressure_grad_tend
    held = {}
    monkeypatch.setattr(tend_v842, "vel_vadv_tend", lambda **kw: held.setdefault("vadv", kw))

    def pgrad_then_vadv(**kw):
        real_pgrad(**kw)
        real_vadv(**held["vadv"])
    monkeypatch.setattr(pgrad_hmix_v842, "vel_pressure_grad_tend", pgrad_then_vadv)
    case = next(c for c in oo.cases() if c["name"] == "zmid_del2_split")
    got = oo.run_gpu(case, "r4")
    assert fv.tend_v842 is tend_v842
    assert oo.bit_mismatches(got["tendNormalVelocity"],
                             oo.load_fixture("r4")[case["name"]]["tendNormalVelocity"]) > 0


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
                    str(tmp_path / "build"), "--piece", "tendency", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in oo.PRECISIONS:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.bit_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
