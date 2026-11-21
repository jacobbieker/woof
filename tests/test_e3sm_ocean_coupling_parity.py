"""E3SM v3.2.0's MPAS-Ocean coupler import and export, word for word.

The reference is E3SM v3.2.0's own ``ocn_import_mct`` and ``ocn_export_mct``
(cut verbatim out of ``components/mpas-ocean/driver/ocn_comp_mct.F``, with
``mpaso_cpl_indices_set``, ``ocn_constants_init`` and
``ocn_freezing_temperature`` as E3SM builds them) over the MPAS pool stubs,
built with gfortran -O0 -ffp-contract=off and driven by
``tools/e3sm_coupler_oracle/run_ocean_coupling.F90``. The outputs are
committed (tests/data/oracles/e3sm_ocean_coupling, PROVENANCE.md). When
``GFORTRAN`` (or gfortran on PATH), ``WOOF_E3SM_V320_SOURCE`` and
``WOOF_MCT_SOURCE`` are set, the fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with E3SM only to a tolerance, or only under strict
   numerics: the word-exact gate also runs in fast and strict child
   processes, over every o2x column and every array, halo cells included.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: absent forcing arrays, the removed
   runoff on its latitudes, signed zeros into gfortran's MAX and MIN, the
   frazil's three branches with a land-ice mask and without, the land-ice
   modes, the thermal forcing's mask threshold, the waves two ways.
4. An index plan that drifts from E3SM's: each stop must be refused with
   E3SM's message; what E3SM gets wrong past bounds, and the BGC tracers,
   are refused by name.
5. A gate that cannot fail: mutated kernels and constants must mismatch.
6. The default changed: nothing in woof/ calls this import or export.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _numerics_child import MODES, run as run_child
from conftest import requires_gpu
from woof.hex.coupled import ocn_coupling_e3sm as occ
from woof.verify import e3sm_ocean_coupling_oracle as oc

ROOT = Path(__file__).resolve().parents[1]


def _case(name):
    return next(c for c in oc.cases() if c["name"] == name)


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oc.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {case["name"]: oc.input_digest(case) for case in oc.cases()}


def test_fixture_covers_the_branches():
    outputs, aborts = oc.load_fixture()
    cases = {c["name"]: c for c in oc.cases()}
    assert set(outputs) | set(aborts) == set(cases)
    assert set(aborts) == {n for n, c in cases.items() if c["aborts"]}

    def col(case, name):
        return outputs[case]["o2x"][:, cases[case]["lists"]["o2x"].index(name)]

    def before(case, pool, name):
        return cases[case]["pools"][pool][name][1]

    # signed zeros: MAX(-0.0, 0.0) keeps -0.0; MIN(-0.0, 0.0) keeps -0.0 (frazil row 1), +0.0 stays (row 2)
    sw = outputs["gcase"]["forcing/shortWaveHeatFlux"]
    assert np.signbit(sw[0]) and sw[0] == 0 and not np.signbit(sw[1]) and sw[2] == 0
    q = col("gcase", "Fioo_q")
    assert q[1] == 0 and np.signbit(q[1]) and q[2] == 0 and not np.signbit(q[2])
    # the sea-ice pressure cap, with mpas_constants' float32 gravity
    cap = (1026.0 * oc.GRAVITY) * 5.0
    assert (outputs["gcase"]["forcing/seaIcePressure"] == cap).any()
    # removed runoff: south of 60S moved (strictly), the halo zeroed
    st = cases["standalone"]
    lat = before("standalone", "mesh", "latCell")
    rofl, removed = outputs["standalone"]["forcing/riverRunoffFlux"], outputs["standalone"]["forcing/removedRiverRunoffFlux"]
    ns = st["lsize"]
    south = lat[:ns] < -1.04719666667
    assert not south[0] and not south[2] and south[3] and (~south).any()  # on, just north of, south of 60S
    assert (rofl[:ns][south] == 0).all() and (removed[ns:] == 0).all()
    # the land-ice frazil: mask 1 cells send no sea-ice frazil but the land-ice heat
    mask = before("standalone", "forcing", "landIceMask")[:ns]
    assert (mask == 1).any() and (mask == 2).any()
    assert (col("standalone", "Fioo_q")[mask == 1] == 0).all() and (col("standalone", "Foxo_q_li")[mask == 1] != 0).any()
    # the frazil arrays reset
    ng = cases["gcase"]["lsize"]
    assert (outputs["gcase"]["state/accumulatedFrazilIceMass"][:ng] == 0).all()
    # untouched where E3SM leaves them: absent fields' packages, the halo, o2x columns not written
    for case in ("gcase", "waves_passive"):
        for key, value in outputs[case].items():
            if key == "o2x" or not key.startswith("forcing/") or cases[case]["pools"]["forcing"][key[8:]][0] not in (5, 6):
                continue
            n = cases[case]["lsize"]
            assert oc.word_mismatches(value[n:], before(case, "forcing", key[8:])[n:]) == 0, key
    assert oc.word_mismatches(col("gcase", "So_blt"), cases["gcase"]["o2x"][:, cases["gcase"]["lists"]["o2x"].index("So_blt")]) == 0
    assert (col("coupled_tf3d", "So_rhoeff") == 0).all()
    # the thermal forcing: a mask of exactly 0.9 is not above it; levels past nGlcZLevels untouched
    tfm = col("coupled_tf3d", "So_tf3d_mask01")
    assert tfm[0] == 0 and set(np.unique(tfm)) == {0.0, 1.0}
    c = cases["coupled_tf3d"]
    assert oc.word_mismatches(col("coupled_tf3d", "So_tf3d04"), c["o2x"][:, c["lists"]["o2x"].index("So_tf3d04")]) == 0


def test_the_indices_refuse_what_e3sm_stops_on():
    _, aborts = oc.load_fixture()
    for case in oc.cases():
        if not case["aborts"]:
            oc.indices_for(case)
            continue
        with pytest.raises(occ.CouplingAbort) as stopped:
            oc.indices_for(case)
        assert str(stopped.value).split(":: FATAL")[-1] in aborts[case["name"]]


def _refusal_setup(name="gcase", **changes):
    case = _case(name)
    return case, oc.indices_for(case), dataclasses.replace(oc.options_for(case), **changes)


@pytest.mark.parametrize("change, match", [
    (dict(config_use_ecosysTracers=True), "config_use_ecosysTracers"),
    (dict(config_use_DMSTracers=True), "config_use_DMSTracers"),
    (dict(config_use_MacroMoleculesTracers=True), "config_use_MacroMoleculesTracers"),
    (dict(config_use_CFCTracers=True), "config_use_CFCTracers"),
])
def test_the_bgc_tracers_are_refused_by_name(change, match):
    _, _, opts = _refusal_setup(**change)
    with pytest.raises(NotImplementedError, match=match):
        opts.check()


def test_what_e3sm_gets_wrong_past_bounds_is_refused():
    case, idx, opts = _refusal_setup(config_momentum_use_active_wave=True)
    present = [n for n, (k, _) in case["pools"]["forcing"].items() if k in (5, 6)]
    with pytest.raises(ValueError, match="column 0"):
        occ.import_rows(idx, opts, present)
    _, _, opts = _refusal_setup(config_remove_ais_river_runoff=True)
    with pytest.raises(ValueError, match="removedRiverRunoffFlux"):
        occ.import_rows(idx, opts, present)
    _, _, opts = _refusal_setup(config_eke_correction_enable=True)
    with pytest.raises(ValueError, match="windSpeed10m"):
        occ.import_rows(idx, opts, [n for n in present if n != "windSpeed10m"])
    lists = [n for n in case["lists"]["o2x"] if n != "Faoo_h2otemp"]
    idx2 = occ.indices(case["lists"]["x2o"], lists)
    with pytest.raises(ValueError, match="Faoo_h2otemp"):
        occ.export_o2x(idx2, oc.options_for(case), np.zeros((case["lsize"], len(lists))),
                       n_cells_solve=case["lsize"], forcing={}, state={}, dims={}, diagnostics={})
    tf = _case("coupled_tf3d")
    with pytest.raises(ValueError, match="glc_nzoc"):
        occ.export_o2x(oc.indices_for(tf), oc.options_for(tf), np.zeros((tf["lsize"], len(tf["lists"]["o2x"]))),
                       n_cells_solve=tf["lsize"], forcing={}, state={}, dims={"nGlcZLevels": 5}, diagnostics={})


def _export_args(name):
    """The export's arguments for a case, as host arrays (refused before any launch)."""
    case = _case(name)
    pools = {"forcing": {}, "state": {}}
    for pool, key, kind, value in oc._entries(case):
        if pool in pools and kind in (5, 6):
            pools[pool][key] = np.ascontiguousarray(value, dtype=np.float64)
        elif pool in pools and kind == 7:
            pools[pool][key] = np.ascontiguousarray(value, dtype=np.int32)
    dims = {n: v for n, (k, v) in case["pools"]["forcing"].items() if k == 4}
    diagnostics = {"landIceBoundaryLayerTracers": case["blt"].astype(np.float64),
                   "landIceTracerTransferVelocities": case["tv"].astype(np.float64),
                   "indexBLT": 1, "indexBLS": 2, "indexHeatTrans": 1, "indexSaltTrans": 2}
    o2x = case["o2x"].astype(np.float64)
    return case, dict(idx=oc.indices_for(case), opts=oc.options_for(case), o2x=o2x), dict(
        n_cells_solve=case["lsize"], forcing=pools["forcing"], state=pools["state"], dims=dims,
        diagnostics=diagnostics)


@requires_gpu
@pytest.mark.parametrize("name, change, match", [
    ("gcase", lambda a, n: a["forcing"].update(avgTracersSurfaceValue=a["forcing"]["avgTracersSurfaceValue"][:n - 1]
                                               .copy()), "at least nCellsSolve"),
    ("gcase", lambda a, n: a["dims"].update(index_avgSalinitySurfaceValue=4), "index 4 is outside"),
    ("gcase", lambda a, n: a["dims"].update(index_avgSSHGradientZonal=0), "index 0 is outside"),
    ("gcase", lambda a, n: a["state"].update(layerThickness=a["state"]["layerThickness"][:, 0].copy()), "2-D"),
    ("gcase", lambda a, n: a["state"].update(ssh=a["state"]["ssh"][:n - 1].copy()), "ssh"),
    ("standalone", lambda a, n: a["forcing"].update(landIceMask=a["forcing"]["landIceMask"][:n - 1].copy()),
     "landIceMask"),
    ("coupled_tf3d", lambda a, n: a["diagnostics"].update(indexHeatTrans=3), "index 3 is outside"),
    ("coupled_tf3d", lambda a, n: a["diagnostics"].update(landIceBoundaryLayerTracers=a["diagnostics"][
        "landIceBoundaryLayerTracers"][:n - 1].copy()), "landIceBoundaryLayerTracers"),
    ("coupled_tf3d", lambda a, n: a["forcing"].update(avgThermalForcingAtZLevels=np.ascontiguousarray(
        a["forcing"]["avgThermalForcingAtZLevels"][:, :2])), "index 3 is outside"),
    ("coupled_tf3d", lambda a, n: a["forcing"].update(avgThermalForcingAtZLevelsMask=np.ascontiguousarray(
        a["forcing"]["avgThermalForcingAtZLevelsMask"][:, :2])), "index 3 is outside"),
])
def test_the_export_refuses_arrays_it_would_read_past(name, change, match):
    case, head, args = _export_args(name)
    change(args, case["lsize"])
    with pytest.raises(ValueError, match=match):
        occ.export_o2x(head["idx"], head["opts"], head["o2x"], **args)


@pytest.mark.parametrize("change, match", [
    (dict(config_land_ice_flux_mode="Standalone"), "config_land_ice_flux_mode"),
    (dict(config_land_ice_flux_mode="diagnostic"), "config_land_ice_flux_mode"),
    (dict(config_land_ice_flux_mode=" coupled"), "config_land_ice_flux_mode"),
    (dict(config_glc_thermal_forcing_coupling_mode="3D"), "config_glc_thermal_forcing_coupling_mode"),
    (dict(config_glc_thermal_forcing_coupling_mode="2d"), "config_glc_thermal_forcing_coupling_mode"),
])
def test_unknown_modes_are_refused(change, match):
    _, _, opts = _refusal_setup(**change)
    with pytest.raises(ValueError, match=match):
        opts.check()
    if "land_ice" in match:
        with pytest.raises(ValueError, match=match):
            occ.export_columns(oc.indices_for(_case("gcase")), opts)


def test_the_known_modes_are_taken():
    for mode, li in (("off", 0), ("pressure_only", 0), ("standalone", 1), ("data", 2), ("coupled   ", 3)):
        _, _, opts = _refusal_setup(config_land_ice_flux_mode=mode)
        opts.check()
        assert occ._land_ice_mode(mode) == li
    assert occ._glc_3d("3d ") and not occ._glc_3d("off")


def test_the_default_path_does_not_call_the_coupling():
    """Opt-in: nothing under woof/ imports it but its own oracle module and
    the opt-in hybrid (MPAS v8.4.2 under E3SM v3.2.0's coupler)."""
    users = [p for p in (ROOT / "woof").rglob("*.py")
             if "ocn_coupling_e3sm" in p.read_text(encoding="utf-8")
             and p.name not in ("ocn_coupling_e3sm.py", "e3sm_ocean_coupling_oracle.py", "hybrid_ocean.py")]
    assert users == []
    # and importing the package and its siblings does not load it
    siblings = sorted(p.stem for p in (ROOT / "woof" / "hex" / "coupled").glob("*.py")
                      if p.stem not in ("__init__", "ocn_coupling_e3sm"))
    code = ("import sys, importlib, woof, woof.hex, woof.hex.coupled\n"
            f"for m in {siblings!r}: importlib.import_module('woof.hex.coupled.' + m)\n"
            "assert 'woof.hex.coupled.ocn_coupling_e3sm' not in sys.modules, 'loaded'\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


@requires_gpu
def test_gpu_matches_the_e3sm_oracle_word_for_word():
    report = oc.measure()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_e3sm_oracle_word_for_word"], numerics, exact)


def _mismatches(case, key, run=None):
    outputs, _ = oc.load_fixture()
    c = _case(case)
    got = (run or oc.run_gpu)(c)
    want = outputs[case]
    if key.startswith("o2x/"):
        k = c["lists"]["o2x"].index(key[4:])
        return oc.word_mismatches(got["o2x"][:, k], want["o2x"][:, k])
    return oc.word_mismatches(got[key], want[key])


@requires_gpu
@pytest.mark.parametrize("original, mutated, key, case", [
    # gfortran's MAX keeps the first argument on a tie (-0.0 stays)
    ("b > a ? __double_as_longlong(b) : __double_as_longlong(a)",
     "a > b ? __double_as_longlong(a) : __double_as_longlong(b)", "forcing/shortWaveHeatFlux", "gcase"),
    # the frazil energy's MIN(.., 0) taken as MAX (NVRTC lowers either MIN tie form to min.f64, which
    # agrees with gfortran wherever the second argument is +0.0 or the positive cap, as it always is here)
    ("seaIceEnergy[i] = fmin_first(", "seaIceEnergy[i] = fmax_first(", "o2x/Fioo_q", "gcase"),
    # the removal's latitude strictly south
    ("latCell[i] < (code[j]", "latCell[i] <= (code[j]", "forcing/riverRunoffFlux", "standalone"),
    # the precipitation factor dropped
    ("case R_RAIN: *dst = M(x[c1[j]], precFactor);", "case R_RAIN: *dst = x[c1[j]];", "forcing/rainFlux",
     "standalone"),
    # the active-wave stress reassociated, (taux - Tawx) + Twox
    ("S(x[c1[j]], S(x[c2[j]], x[c3[j]]))", "A(S(x[c1[j]], x[c2[j]]), x[c3[j]])", "forcing/windStressZonal",
     "waves_active"),
    # the halo not zeroed
    ("if (zeroFirst[j]) buf[(size_t)j * nCells + i] = 0.0;", "", "forcing/removedRiverRunoffFlux", "standalone"),
    # any nonzero land-ice mask taken as land ice
    ("landIceMask[i] == 1", "landIceMask[i] != 0", "o2x/Fioo_q", "standalone"),
    # the accumulated frazil not reset
    ("        accumulated[i] = 0.0;\n", "", "state/accumulatedFrazilIceMass", "gcase"),
    # the freezing point's mushy denominator dropped
    ("D(M(coeff_mushy, salinity), S(1.0, D(salinity, 1e3)))", "M(coeff_mushy, salinity)", "o2x/Fioo_q", "gcase"),
    # the thermal-forcing mask threshold inclusive
    ("if (m > 0.9) {", "if (m >= 0.9) {", "o2x/So_tf3d_mask01", "coupled_tf3d"),
    # the solid runoff removed south of 60S, not 57S
    ("-0.99483767345", "-1.04719666667", "forcing/iceRunoffFlux", "standalone"),
    # the sea-ice pressure not capped
    ("fmin_first(x[c1[j]], M(M(density0, gravity), 5.0))", "x[c1[j]]", "forcing/seaIcePressure", "gcase"),
    # the 10 m wind not square-rooted
    ("case R_SQRT: *dst = __dsqrt_rn(x[c1[j]]);", "case R_SQRT: *dst = x[c1[j]];", "forcing/windSpeed10m",
     "standalone"),
    # Faoo_h2otemp without cp_sw
    ("M(M(fwtf[i], rho_sw), cp_sw)", "M(fwtf[i], rho_sw)", "o2x/Faoo_h2otemp", "gcase"),
    # the accumulated frazil's latent heat dropped
    ("seaIceEnergy[i] = M(accumulated[i], heatOfFusion);", "seaIceEnergy[i] = accumulated[i];", "o2x/Fioo_q",
     "gcase"),
    # land ice data sends the boundary layer and not its own fluxes (modes 2 and 3 swapped)
    ("bool standaloneOrData = liMode == 1 || liMode == 2;", "bool standaloneOrData = liMode == 1 || liMode == 3;",
     "o2x/Foxo_ismw", "data"),
    ("if (liMode == 1 || liMode == 3) {", "if (liMode == 1 || liMode == 2) {", "o2x/So_blt", "coupled_tf3d"),
    # the land-ice frazil's heat without the latent heat
    ("o[col[O_Q_LI]] = D(M(accumulated[i], heatOfFusion), dt);", "o[col[O_Q_LI]] = D(accumulated[i], dt);",
     "o2x/Foxo_q_li", "standalone"),
])
def test_a_mutated_kernel_is_caught(monkeypatch, original, mutated, key, case):
    real = occ.module_source
    assert real().count(original) == 1
    monkeypatch.setattr(occ, "module_source", lambda: real().replace(original, mutated))
    occ._module_on.cache_clear()
    try:
        assert _mismatches(case, key) > 0
    finally:
        occ._module_on.cache_clear()


@requires_gpu
@pytest.mark.parametrize("name, value, key, case", [
    ("GRAVITY", 9.80616, "forcing/seaIcePressure", "gcase"),  # not float32-rounded
    ("RHO_SW", 1027.25, "o2x/Faoo_h2otemp", "standalone"),  # config_density0, not shr_const's
    ("T0_KELVIN", 273.16, "o2x/Fioo_q", "gcase"),  # ocn_constants' own, not shr_const's
])
def test_a_mutated_constant_is_caught(monkeypatch, name, value, key, case):
    monkeypatch.setattr(occ, name, value)
    assert _mismatches(case, key) > 0


def _oracle_toolchain():
    e3sm, mct = os.environ.get("WOOF_E3SM_V320_SOURCE"), os.environ.get("WOOF_MCT_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not e3sm or not mct or not gfortran or not Path(e3sm, "components", "mpas-ocean", "driver").is_dir():
        return None
    return gfortran, Path(e3sm), Path(mct)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the E3SM coupler oracle build needs gfortran (pixi -e mpas-oracle), "
                           "WOOF_E3SM_V320_SOURCE (an E3SM v3.2.0 tree with components/mpas-ocean and "
                           "mpas-framework) and WOOF_MCT_SOURCE (MCT at 82b0071)")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, e3sm, mct = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "e3sm_coupler_oracle" / "build.sh"), str(e3sm), str(mct),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "e3sm_coupler_oracle" / "make_fixture.py"),
                    str(tmp_path / "build"), "--oracle", "coupling", "--out", str(tmp_path / "fixture")], check=True)
    for name in ("oracle-r8.npz", "aborts.json", "inputs.sha256"):
        if name.endswith(".npz"):
            fresh, pinned = np.load(tmp_path / "fixture" / name), np.load(oc.FIXTURE_DIR / name)
            assert sorted(fresh.files) == sorted(pinned.files)
            for key in pinned.files:
                assert oc.word_mismatches(fresh[key], pinned[key]) == 0, key
        else:
            assert (tmp_path / "fixture" / name).read_text() == (oc.FIXTURE_DIR / name).read_text()
