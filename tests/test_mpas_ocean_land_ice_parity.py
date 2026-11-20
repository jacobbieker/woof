"""MPAS-Ocean's land-ice fluxes ('standalone', 'coupled') and the effective
density in land ice, word for word against v8.4.2, at RKIND 4 and 8.

Two references, both MPAS-Model v8.4.2 compiled verbatim with gfortran -O0
-ffp-contract=off at MPAS's default single precision and at its double
precision:

* ``tools/mpas_ocean_oracle/run_land_ice.F90`` drives
  ``mpas_ocn_surface_land_ice_fluxes.F`` (the build of the fluxes, then the
  thickness, velocity and tracer forcing they add) and
  ``mpas_ocn_effective_density_in_land_ice.F``
  (tests/data/oracles/mpas_ocean_land_ice);
* ``run_diagnostics.F90`` drives ``ocn_diagnostic_solve``, whose
  ``ocn_compute_land_ice_flux_input_fields`` computes the top drag, the
  friction velocity, the boundary-layer tracers, the Jenkins transfer
  velocities and the land-ice attenuation
  (tests/data/oracles/mpas_ocean_land_ice_diagnostics).

When ``GFORTRAN`` (or gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` are set,
both fixtures are rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance, in either precision, or
   only under strict numerics: the word-exact gates also run in fast and
   strict child processes.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: ISOMIP and Jenkins, the Jenkins interface
   salinity at its 0.001 floor, open-ocean cells and halo cells beyond
   nCellsSolve left alone, 'pressure_only' writing nothing, the coupled
   effective density, swapped var_array indices.
4. A gate that cannot fail: a mutated freezing point, ISOMIP flux, salinity
   floor, accumulation, thickness, tracer or velocity forcing, effective
   density, top drag, neighbour weighting or attenuation must produce
   mismatches.
5. The default changed: with ``config_land_ice_flux_mode`` 'off' (the
   Registry default) nothing is written.
6. An unported formulation run silently: 'HollandJenkins' and
   ``useHollandJenkinsAdvDiff`` are refused by name.
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
from woof.verify import mpas_ocean_land_ice_diagnostics_oracle as do
from woof.verify import mpas_ocean_land_ice_oracle as oo

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")
ORACLES = {"fluxes": oo, "diagnostics": do}


@pytest.mark.parametrize("which", sorted(ORACLES))
def test_case_inputs_are_the_recorded_bytes(which):
    m = ORACLES[which]
    recorded = dict(reversed(line.split()) for line in
                    (m.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": m.input_digest(case) for p in BOTH for case in m.cases(p)}


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_flux_branches(precision):
    fixture = oo.load_fixture(precision)
    cases = {case["name"]: case for case in oo.cases(precision)}
    assert set(fixture) == set(cases)
    seen = {"ISOMIP": 0, "Jenkins": 0, "floor": 0, "halo": 0, "coupled": 0, "pressure_only": 0, "swapped": 0}
    for name, out in fixture.items():
        case = cases[name]
        pools, cfg = case["pools"], case["config"]
        f, mode = pools["forcing"], cfg["config_land_ice_flux_mode"]
        for key, words in out.items():
            assert np.isfinite(words).all(), (name, key)
        if mode == "pressure_only":
            # landIceFluxesPKG is off: the routines leave every field as it came
            seen["pressure_only"] += 1
            for key, words in out.items():
                pool, field = key.split("/")
                assert oo.word_mismatches(words, pools[pool][field]) == 0, (name, key)
            continue
        solved, nc = int(pools["mesh"]["nCellsSolve"]), case["dims"]["nCells"]
        mask = f["landIceMask"][:solved] == 1
        fw, given = out["forcing/landIceFreshwaterFlux"], f["landIceFreshwaterFlux"]
        seen["swapped"] += int(f["index_landIceInterfaceSalinity"] == 1)
        if mode == "coupled":
            # the coupler's fluxes are kept; the effective density is updated everywhere
            seen["coupled"] += 1
            assert (fw == given).all()
            assert (out["state/effectiveDensityInLandIce@2"][:nc]
                    != pools["state"]["effectiveDensityInLandIce@2"][:nc]).all()
            continue
        formulation = cfg["config_land_ice_flux_formulation"]
        seen[formulation] += 1
        assert (fw[:solved][mask] != given[:solved][mask]).any(), name
        assert (fw[:solved][~mask] == given[:solved][~mask]).all(), name
        for field in ("landIceFreshwaterFlux", "landIceHeatFlux", "heatFluxToLandIce"):
            assert (out[f"forcing/{field}"][solved:] == f[field][solved:]).all(), (name, field)
        seen["halo"] += int(solved < nc)
        if formulation == "Jenkins":
            isl = f["index_landIceInterfaceSalinity"] - 1
            salinity = out["forcing/landIceInterfaceTracers"][:solved][mask, isl]
            seen["floor"] += int((salinity == salinity.dtype.type(0.001)).sum())
        else:
            assert (out["forcing/heatFluxToLandIce"][:solved][mask] == 0).all()
    assert all(seen.values()), seen


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_diagnostic_branches(precision):
    fixture = do.load_fixture(precision)
    cases = {case["name"]: case for case in do.cases(precision)}
    assert set(fixture) == set(cases)
    formulations, weighted, attenuated = set(), 0, 0
    for name, out in fixture.items():
        case = cases[name]
        pools, cfg = case["pools"], case["config"]
        d = pools["diagnostics"]
        for key in ("diagnostics/landIceBoundaryLayerTracers", "diagnostics/landIceFrictionVelocity",
                    "diagnostics/topDrag", "diagnostics/topDragMagnitude"):
            assert np.isfinite(out[key]).all() and (out[key] != d[key.split("/")[1]]).any(), (name, key)
        formulation = cfg["config_land_ice_flux_formulation"]
        formulations.add(formulation)
        tv_moved = (out["diagnostics/landIceTracerTransferVelocities"] != d["landIceTracerTransferVelocities"]).any()
        assert tv_moved == (formulation == "Jenkins"), name
        weighted += int(cfg["config_land_ice_flux_boundaryLayerNeighborWeight"] > 0)
        mask = out["forcing/landIceMask"] == 1
        coefficient = out["diagnostics/surfaceFluxAttenuationCoefficient"]
        want = coefficient.dtype.type(cfg["config_land_ice_flux_attenuation_coefficient"])
        attenuated += int(mask.any() and (coefficient[mask] == want).all())
    assert formulations == {"ISOMIP", "Jenkins"} and weighted >= 2 and attenuated == len(fixture)


def test_unported_formulations_are_refused_by_name():
    from woof.hex.mpas_ocean import land_ice_v842 as li
    from woof.hex.mpas_ocean.refusal import ConfigurationRefusal
    base = {"config_land_ice_flux_mode": "standalone"}
    with pytest.raises(ConfigurationRefusal, match="HollandJenkins"):
        li.check_config({**base, "config_land_ice_flux_formulation": "HollandJenkins"})
    with pytest.raises(ConfigurationRefusal, match="useHollandJenkinsAdvDiff"):
        li.check_config({**base, "config_land_ice_flux_useHollandJenkinsAdvDiff": True})
    with pytest.raises(ConfigurationRefusal, match="config_land_ice_flux_formulation"):
        li.check_config({**base, "config_land_ice_flux_formulation": "Holland"})
    with pytest.raises(ConfigurationRefusal, match="config_land_ice_flux_mode"):
        li.check_config({"config_land_ice_flux_mode": "data"})
    # the advection-diffusion switch is read only by standalone Jenkins (v8.4.2's build_arrays)
    li.check_config({"config_land_ice_flux_mode": "coupled", "config_land_ice_flux_useHollandJenkinsAdvDiff": True})
    li.check_config({**base, "config_land_ice_flux_formulation": "ISOMIP",
                     "config_land_ice_flux_useHollandJenkinsAdvDiff": True})
    for mode in li.MODES:
        li.check_config({"config_land_ice_flux_mode": mode})


@requires_gpu
def test_land_ice_off_writes_nothing():
    """The Registry default ('off'): every routine returns at once."""
    import cupy as cp
    from woof.hex.mpas_ocean import land_ice_v842 as li
    from woof.verify import mpas_ocean_stream as stream
    assert li.DEFAULT_CONFIGS["config_land_ice_flux_mode"] == "off" and not li.fluxes_on({})
    assert li.two_level({}) == ()
    case = next(c for c in oo.cases("r4") if c["name"] == "coupled")
    config = dict(case["config"], config_land_ice_flux_mode="off")
    dev = stream.on_device(oo.FIELDS, case)
    m, f, d, st = dev["mesh"], dev["forcing"], dev["diagnostics"], dev["state"]
    before = {(p, n): cp.asnumpy(dev[p][n]) for p, n, k, _ in oo.FIELDS
              if k in (oo.REAL, oo.INT) and dev[p][n] is not None}
    nc, ne = int(m["nCells"]), int(m["nEdges"])
    li.build_arrays(dt=case["dt"], mesh=m, state=st, diagnostics=d, forcing=f, configs=config)
    li.input_fields(mesh=m, state=st, tracers={}, diagnostics=d, forcing=f, configs=config)
    li.thick_tend(forcing=f, surface_thickness_flux=f["surfaceThicknessFlux"], n_cells=nc, configs=config)
    li.vel_tend(diagnostics=d, surface_stress=f["surfaceStress"], surface_stress_magnitude=f["surfaceStressMagnitude"],
                n_edges=ne, n_cells=nc, configs=config)
    li.tracers_tend(forcing=f, tracers_surface_flux=dev["tracersSurfaceFlux"]["activeTracersSurfaceFlux"],
                    n_cells=nc, configs=config)
    li.effective_density_update(mesh=m, forcing=f, state=st, configs=config)
    li.coupled_means(diagnostics=d, forcing=f, effective_density=st["effectiveDensityInLandIce@2"], n_cells=nc,
                     n_accumulated=3, configs=config)
    for (p, n), words in before.items():
        assert oo.word_mismatches(cp.asnumpy(dev[p][n]), words) == 0, (p, n)


@requires_gpu
@pytest.mark.parametrize("which", sorted(ORACLES))
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(which, precision):
    report = ORACLES[which].measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert report and not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_v842_oracle_word_for_word"], numerics, exact)


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
@pytest.mark.parametrize("which, original, mutated, field", [
    # the freezing point's pressure-salinity term dropped
    ("fluxes", "A(A(A(c[0], M(c[1], salinity)), M(c[2], pressureOffset)), M(M(c[3], pressureOffset), salinity))",
     "A(A(c[0], M(c[1], salinity)), M(c[2], pressureOffset))", "forcing/landIceInterfaceTracers"),
    # ISOMIP's freshwater flux with its sign flipped
    ("fluxes", "const rk freshwaterFlux = -M(M(M(rhoSw, gammaT), D(cpSw, latent)), S(it, blt));",
     "const rk freshwaterFlux = M(M(M(rhoSw, gammaT), D(cpSw, latent)), S(it, blt));",
     "forcing/landIceFreshwaterFlux"),
    # the Jenkins interface salinity without its floor
    ("fluxes", "A(b, SQRT(S(M(b, b), M(M(C(4.0), a), c))))), C(0.001));",
     "A(b, SQRT(S(M(b, b), M(M(C(4.0), a), c))))), C(0.0));", "forcing/landIceInterfaceTracers"),
    # the melted mass accumulated with the wrong sign
    ("fluxes", "S(accumulatedLandIceMassOld[iCell], M(dt, landIceFreshwaterFlux[iCell]))",
     "A(accumulatedLandIceMassOld[iCell], M(dt, landIceFreshwaterFlux[iCell]))", "state/accumulatedLandIceMass@2"),
    # the thickness flux in kg rather than m
    ("fluxes", "D(landIceFreshwaterFlux[iCell], rhoSw)", "landIceFreshwaterFlux[iCell]",
     "forcing/surfaceThicknessFlux"),
    # the heat flux added to the salinity's surface flux
    ("fluxes", "f[0] = A(f[0], D(landIceHeatFlux[iCell], M(rhoSw, cpSw)));",
     "f[1] = A(f[1], D(landIceHeatFlux[iCell], M(rhoSw, cpSw)));", "tracersSurfaceFlux/activeTracersSurfaceFlux"),
    # the top drag left out of the velocity forcing's magnitude
    ("fluxes", "A(surfaceStressMagnitude[iCell], topDragMagnitude[iCell])", "surfaceStressMagnitude[iCell]",
     "forcing/surfaceStressMagnitude"),
    # the effective density without gravity
    ("fluxes", "D(-landIcePressure[iCell], M(sshNew[iCell], GRAVITY))", "D(-landIcePressure[iCell], sshNew[iCell])",
     "state/effectiveDensityInLandIce@2"),
    # the effective density's neighbours unweighted
    ("fluxes", "weightSum = A(weightSum, mask);", "", "state/effectiveDensityInLandIce@2"),
    # the top drag on the mean fraction's half
    ("diagnostics", "M(C(0.5), A(landIceFraction[cell1], landIceFraction[cell2]))", "landIceFraction[cell1]",
     "diagnostics/topDrag"),
    # the friction velocity without the tidal velocity
    ("diagnostics", "A(M(C(2.0), ke), M(rmsTidalVelocity, rmsTidalVelocity))", "M(C(2.0), ke)",
     "diagnostics/landIceFrictionVelocity"),
    # the boundary layer's neighbour weights not summed
    ("diagnostics", "weightSum = A(weightSum, M(mask, neighborWeight));", "",
     "diagnostics/landIceBoundaryLayerTracers"),
    # the boundary layer one level shallower
    ("diagnostics", "for (int k = 0; k < maxLevelCell[iCell]; ++k) {", "for (int k = 1; k < maxLevelCell[iCell]; ++k) {",
     "diagnostics/landIceBoundaryLayerTracers"),
    # the land-ice attenuation not applied
    ("diagnostics", "if (landIceMask[iCell] == 1) surfaceFluxAttenuationCoefficient[iCell] = attenuation;", "",
     "diagnostics/surfaceFluxAttenuationCoefficient"),
])
def test_a_mutated_kernel_is_caught(monkeypatch, precision, which, original, mutated, field):
    from woof.hex.mpas_ocean import land_ice_v842
    m = ORACLES[which]
    real = land_ice_v842.module_source
    assert real(precision).count(original) == 1
    monkeypatch.setattr(land_ice_v842, "module_source", lambda p="r4": real(p).replace(original, mutated))
    land_ice_v842._module_on.cache_clear()
    try:
        want = m.load_fixture(precision)
        mismatches = sum(m.word_mismatches(m.run_gpu(case)[field], want[case["name"]][field])
                         for case in m.cases(precision) if field in want[case["name"]])
        assert mismatches > 0
    finally:
        land_ice_v842._module_on.cache_clear()


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_ocean").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MPAS-Ocean oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
def test_the_oracle_rebuilt_here_writes_the_fixtures(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_ocean_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    for piece, m in (("land_ice", oo), ("land_ice_diagnostics", do)):
        out = tmp_path / piece
        subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_ocean_oracle" / "make_fixture.py"),
                        str(tmp_path / "build"), "--piece", piece, "--out", str(out)], check=True)
        for precision in BOTH:
            fresh = np.load(out / f"oracle-{precision}.npz")
            pinned = np.load(m.FIXTURE_DIR / f"oracle-{precision}.npz")
            assert sorted(fresh.files) == sorted(pinned.files)
            for key in pinned.files:
                assert m.word_mismatches(fresh[key], pinned[key]) == 0, (piece, precision, key)
