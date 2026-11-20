"""E3SM v3.2.0's MPAS-Seaice coupler import and export, word for word.

The reference is E3SM v3.2.0's own ``ice_import_mct`` and ``ice_export_mct``
(cut verbatim out of ``components/mpas-seaice/driver/ice_comp_mct.F`` with
``basal_pressure`` and ``frazil_mass``, over ``mpassi_cpl_indices_set``,
``seaice_constants`` under ``CCSMCOUPLED`` and the pinned Icepack's column
physics) on the MPAS pool stubs, built with gfortran -O0 -ffp-contract=off
and driven by ``tools/e3sm_coupler_oracle/run_seaice_coupling.F90``. The
outputs are committed (tests/data/oracles/e3sm_seaice_coupling,
PROVENANCE.md). When ``GFORTRAN`` (or gfortran on PATH),
``WOOF_E3SM_V320_SOURCE``, ``WOOF_MCT_SOURCE`` and ``WOOF_ICEPACK_SOURCE``
are set, the fixture itself is rebuilt and compared.

The stress rotation's sine, cosine, arcsine and arctangent are glibc's in
the oracle; the port takes their words as an input (``trig``), and these
tests feed the words the oracle recorded, so the parity claim does not rest
on the host's libm. ``rotation_trig`` (Python's ``math``) is checked against
them where the host libm is glibc.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with E3SM only to a tolerance, or only under strict
   numerics: the word-exact gate also runs in fast and strict child
   processes, over every i2x column and every array, halo cells included.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: every tfrz_option and thermodynamics,
   the frazil mass (mushy, both Si0new branches, both liquidus branches,
   BL99, none), the slab ocean, ice fractions of 0, -0, below and above 1,
   the rotated and unrotated grids, ponds, column shortwave, icebergs, the
   aerosols bulk and modal, the z-aerosols, the waves two ways.
4. An index plan that drifts from E3SM's: each stop must be refused with
   MCT's message; what E3SM gets wrong (quiet lookups written at column 0,
   reads past bounds, a pool pointer never set) is refused by name.
5. A gate that cannot fail: mutated kernels and constants must mismatch.
6. The default changed: nothing in woof/ calls this import or export.

QUIRKS (E3SM v3.2.0, ice_comp_mct.F / mpassi_cpl_indices.F)
-----------------------------------------------------------
- ice_export_mct writes Faii_evap twice and Faii_swnet twice (once more
  under the column shortwave): harmless, the same word.
- Si_u10withgusts is Si_u10 (atmosReferenceSpeed10m).
- Under an ocean surface that is not "free" the import leaves
  frazilMassAdjust alone and the export still adds it to Fioi_meltw and
  Fioi_salt (kept: the case som_linear_salt).
- tfrz_option "mushy" under non-mushy thermodynamics is the linear
  liquidus (Icepack's icepack_liquidus_temperature; kept: bl99_mushy_tfrz).
- Under the waves two ways with the column waves the import reads
  index_x2i_Sw_wavespec(1:nFrequencies), allocated (1:wav_nfreq): refused
  when nFrequencies > wav_nfreq.
- The biogeochemistry i2x/x2i lookups are quiet (index 0 if absent): the
  coupled column biogeochemistry is refused.
- The export reads oceanDustIronFlux (z-aerosols coupled) through the
  biogeochemistry pool pointer, set only under the coupled column
  biogeochemistry: E3SM faults (the oracle segfaults); refused.
- Si_ifloe must be in the list under the waves two ways but is written only
  with the floe size distribution (kept: waves_twoway_off).
- The wave spectrum's names are 'Sw_wavespec' // i2.2: past 99, "**".
"""
from __future__ import annotations

import dataclasses
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _numerics_child import MODES, run as run_child
from conftest import requires_gpu
from woof.hex.coupled import ice_coupling_e3sm as icc
from woof.verify import e3sm_seaice_coupling_oracle as si

ROOT = Path(__file__).resolve().parents[1]


def _case(name):
    return next(c for c in si.cases() if c["name"] == name)


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (si.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {case["name"]: si.input_digest(case) for case in si.cases()}


def test_fixture_covers_the_branches():
    outputs, aborts = si.load_fixture()
    cases = {c["name"]: c for c in si.cases()}
    assert set(outputs) | set(aborts) == set(cases)
    assert set(aborts) == {n for n, c in cases.items() if c["aborts"]}
    run = [c["spec"] for c in cases.values() if not c["aborts"]]
    assert {s["tfrz"] for s in run} == set(icc.TFRZ)
    assert {s["thermo"] for s in run} == set(icc.KTHERM)
    assert {(s["thermo"], s["tfrz"]) for s in run} >= {("mushy", "mushy"), ("BL99", "mushy")}
    for key in ("rotate", "ponds", "column_sw", "bergs", "aerosols", "modal", "zaer", "column_bgc", "fsd",
                "column_waves"):
        assert {bool(s[key]) for s in run} == {True, False}, key
    assert {s["surface"] for s in run} == {"free", "mixed_layer"}
    assert {s["wav"] for s in run} == {"none", "twoway"}
    assert any(s["halo"] == 0 for s in run) and any(s["halo"] > 0 for s in run)

    def col(case, lst, name):
        c = cases[case]
        return (outputs[case]["i2x"] if lst == "i2x" else c["x2i"])[:, c["lists"][lst].index(name)]

    # the frazil mass: mushy, both Si0new branches (S > 6 and S <= 6), both liquidus branches (S/0.75 > Sb)
    c = cases["mushy_free"]
    q, s = col("mushy_free", "x2i", "Fioo_q"), col("mushy_free", "x2i", "So_s")
    hot = q > 0
    assert (hot & (s > 6)).any() and (hot & (s <= 6)).any() and (hot & ((s - 3) / 0.75 > 123.67)).any()
    assert (~hot).any() and ((q == 0) & np.signbit(q)).any()
    adjust = outputs["mushy_free"]["ocean_coupling/frazilMassAdjust"][:c["lsize"]]
    assert (adjust[hot] != col("mushy_free", "x2i", "Fioo_frazil")[hot]).all()
    # the slab ocean leaves frazilMassAdjust as it was
    som = cases["som_linear_salt"]
    assert si.word_mismatches(outputs["som_linear_salt"]["ocean_coupling/frazilMassAdjust"],
                              som["pools"]["ocean_coupling"]["frazilMassAdjust"][1]) == 0
    # the ice fraction: 0, -0, 1, above 1, below 0, tiny, interior; only a positive one writes the rest
    for name in ("mushy_free", "bl99_mushy_tfrz"):
        ifrac, t = col(name, "i2x", "Si_ifrac"), col(name, "i2x", "Si_t")
        assert ifrac[3] == 1.0 and ifrac[1] == 0 and np.signbit(ifrac[1]) and ifrac[4] < 0
        assert (t[ifrac <= 0] == 0).all() and (t[ifrac > 0] != 0).all()
    # icebergs and floes written regardless of the fraction; the floe only with the distribution
    bergs = col("bl99_mushy_tfrz", "i2x", "PFioi_bergw")
    assert bergs[0] != 0
    assert (col("waves_twoway", "i2x", "Si_ifloe") != 0).all()
    assert (col("waves_twoway_off", "i2x", "Si_ifloe") == 0).all()
    # the halo is untouched by the import
    for name, case in cases.items():
        if case["aborts"]:
            continue
        n = case["lsize"]
        for key, value in outputs[name].items():
            pool, _, array = key.partition("/")
            if pool in case["pools"] and array in case["pools"][pool]:
                assert si.word_mismatches(value[n:], case["pools"][pool][array][1][n:]) == 0, (name, key)
    # the waves two ways with the column waves read the spectrum; without them, nothing
    w = cases["waves_twoway"]
    spec = outputs["waves_twoway"]["wave_coupling/waveSpectra"][:w["lsize"]]
    assert si.word_mismatches(spec[:, 6], col("waves_twoway", "x2i", "Sw_wavespec07")) == 0
    co = cases["waves_column_only"]
    assert si.word_mismatches(outputs["waves_column_only"]["wave_coupling/waveSpectra"],
                              co["pools"]["wave_coupling"]["waveSpectra"][1]) == 0


def test_the_indices_refuse_what_e3sm_stops_on():
    _, aborts = si.load_fixture()
    for case in si.cases():
        if not case["aborts"]:
            si.indices_for(case)
            continue
        with pytest.raises(icc.CouplingAbort) as stopped:
            si.indices_for(case)
        assert str(stopped.value).split(":: FATAL")[-1] in aborts[case["name"]]


def test_the_wavespec_names_are_e3sms():
    assert icc.wavespec_name(7) == "Sw_wavespec07"
    assert icc.wavespec_name(99) == "Sw_wavespec99"
    assert icc.wavespec_name(100) == "Sw_wavespec**"
    case = _case("waves_twoway")
    with pytest.raises(icc.CouplingAbort, match=r"Sw_wavespec\*\*"):
        icc.indices(case["lists"]["x2i"], case["lists"]["i2x"], wav_ice_coup="twoway", wav_nfreq=100)
    idx = icc.indices(case["lists"]["x2i"], case["lists"]["i2x"], wav_ice_coup="twoway   ", wav_nfreq=7)
    assert idx.wav_ice_coup == "twoway" and len(idx.wavespec) == 7


def _refusal_setup(name="mushy_free", **changes):
    case = _case(name)
    return case, si.indices_for(case), dataclasses.replace(si.options_for(case), **changes)


@pytest.mark.parametrize("change, match", [
    (dict(config_use_column_biogeochemistry=True, config_couple_biogeochemistry_fields=True), "column 0"),
    (dict(config_use_zaerosols=True, config_couple_biogeochemistry_fields=True), "never sets"),
    (dict(config_thermodynamics_type="Mushy"), "config_thermodynamics_type"),
    (dict(config_thermodynamics_type="zero_layer"), "config_thermodynamics_type"),
    (dict(config_sea_freezing_temperature_type="linear"), "tfrz_option"),
    (dict(config_sea_freezing_temperature_type=" mushy"), "tfrz_option"),
])
def test_what_e3sm_gets_wrong_or_aborts_on_is_refused(change, match):
    _, _, opts = _refusal_setup(**change)
    with pytest.raises(ValueError, match=match):
        opts.check()


def test_the_known_types_are_taken():
    for thermo, k in (("mushy", 2), ("BL99 ", 1), ("zero layer", 0)):
        _, _, opts = _refusal_setup(config_thermodynamics_type=thermo)
        opts.check()
        assert opts.ktherm == k
    for tfrz, k in (("mushy", 0), ("linear_salt", 1), ("constant  ", 2), ("minus1p8", 3)):
        _, _, opts = _refusal_setup(config_sea_freezing_temperature_type=tfrz)
        opts.check()
        assert opts.tfrz == k
    # column BGC alone and the z-aerosols alone are taken
    _refusal_setup(config_use_column_biogeochemistry=True)[2].check()
    _refusal_setup(config_use_zaerosols=True)[2].check()


def test_the_constants_are_ccsmcoupled():
    """seaice_constants under CCSMCOUPLED takes shr_const_mod's; Icepack's Lfresh is Lsub - Lvap."""
    assert (icc.RHOI, icc.RHOW, icc.CP_ICE, icc.CP_OCN, icc.LATICE) == (917.0, 1026.0, 2117.27, 3996.0, 333700.0)
    assert icc.LFRESH == (3.337e5 + 2.501e6) - 2.501e6


def test_the_default_path_does_not_call_the_coupling():
    """Opt-in: nothing under woof/ imports it but its own oracle module."""
    users = [p for p in (ROOT / "woof").rglob("*.py")
             if "ice_coupling_e3sm" in p.read_text(encoding="utf-8")
             and p.name not in ("ice_coupling_e3sm.py", "e3sm_seaice_coupling_oracle.py")]
    assert users == []
    siblings = sorted(p.stem for p in (ROOT / "woof" / "hex" / "coupled").glob("*.py")
                      if p.stem not in ("__init__", "ice_coupling_e3sm"))
    code = ("import sys, importlib, woof, woof.hex, woof.hex.coupled\n"
            f"for m in {siblings!r}: importlib.import_module('woof.hex.coupled.' + m)\n"
            "assert 'woof.hex.coupled.ice_coupling_e3sm' not in sys.modules, 'loaded'\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


@pytest.mark.skipif(platform.system() != "Linux" or platform.libc_ver()[0] != "glibc",
                    reason="rotation_trig is the platform libm; its words are E3SM's only on glibc")
def test_rotation_trig_is_the_oracles_on_glibc():
    outputs, _ = si.load_fixture()
    for case in si.cases():
        if case["aborts"]:
            continue
        m = case["pools"]["mesh"]
        got = icc.rotation_trig(m["latCell"][1], m["lonCell"][1], m["xCell"][1], m["yCell"][1], m["zCell"][1],
                                m["sphere_radius"][1], case["spec"]["rotate"], case["lsize"])
        assert si.word_mismatches(got, outputs[case["name"]]["trig"]) == 0, case["name"]


@requires_gpu
def test_gpu_matches_the_e3sm_oracle_word_for_word():
    report = si.measure()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_e3sm_oracle_word_for_word"], numerics, exact)


@requires_gpu
def test_the_import_refuses_more_frequencies_than_coupled():
    import cupy as cp
    case = _case("waves_twoway")
    idx, opts = si.indices_for(case), si.options_for(case)
    with pytest.raises(ValueError, match="wav_nfreq"):
        icc.import_x2i(idx, opts, cp.asarray(case["x2i"]), n_cells_solve=case["lsize"], pools={},
                       n_frequencies=case["nfreq"] + 1)


@requires_gpu
@pytest.mark.parametrize("pool, name, cut, match", [
    ("tracers_aggregate", "iceAreaCell", 1, "at least nCellsSolve"),
    ("ocean_fluxes", "oceanSaltFlux", 1, "at least nCellsSolve"),
    ("atmos_coupling", "airDensity", 1, "over the"),
])
def test_arrays_too_short_are_refused(pool, name, cut, match):
    import cupy as cp
    case = _case("mushy_free")
    idx, opts = si.indices_for(case), si.options_for(case)
    pools = {p: {n: cp.asarray(np.ascontiguousarray(v, dtype=np.float64)) for n, (k, v) in arrays.items()
                 if k in (5, 6)} for p, arrays in case["pools"].items()}
    a = case["pools"][pool][name][1]
    pools[pool][name] = cp.asarray(np.ascontiguousarray(a[:case["lsize"] - cut] if pool != "atmos_coupling"
                                                        else a[:-cut]))
    trig = cp.asarray(si.load_fixture()[0]["mushy_free"]["trig"])
    with pytest.raises(ValueError, match=match):
        if pool == "atmos_coupling":
            icc.import_x2i(idx, opts, cp.asarray(case["x2i"]), n_cells_solve=case["lsize"], pools=pools)
        else:
            icc.export_i2x(idx, opts, cp.asarray(case["i2x"]), n_cells_solve=case["lsize"], pools=pools, trig=trig)


def _mismatches(case, key):
    outputs, _ = si.load_fixture()
    c = _case(case)
    got, want = si.run_gpu(c), outputs[case]
    if key.startswith("i2x/"):
        k = c["lists"]["i2x"].index(key[4:])
        return si.word_mismatches(got["i2x"][:, k], want["i2x"][:, k])
    return si.word_mismatches(got[key], want[key])


@requires_gpu
@pytest.mark.parametrize("original, mutated, key, case", [
    # the fraction not capped at 1
    ("fmin_first(AR(A_AREA), 1.0)", "AR(A_AREA)", "i2x/Si_ifrac", "mushy_free"),
    # a zero fraction taken as ice
    ("    if (ailohi > 0.0) {\n        o[col[I_T]]", "    if (ailohi >= 0.0) {\n        o[col[I_T]]", "i2x/Si_t",
     "mushy_free"),
    # the grid's rotation flag inverted
    ("if (rotate) {", "if (!rotate) {", "i2x/Faii_taux", "mushy_free"),
    # the ocean stress's sign
    ("-AR(A_OCNU), -AR(A_OCNV)", "AR(A_OCNU), AR(A_OCNV)", "i2x/Fioi_tauy", "mushy_free"),
    # the ponds' mass dropped
    ("if (ponds)", "if (0)", "i2x/Si_bpress", "bl99_mushy_tfrz"),
    # the salt's frazil term reassociated
    ("D(M(M(referenceSalinity, 0.001), AR(A_FRZADJ)), ailohi)", "D(M(referenceSalinity, M(0.001, AR(A_FRZADJ))), "
     "ailohi)", "i2x/Fioi_salt", "mushy_free"),
    # the surface temperature left in Celsius
    ("A(freshWaterFreezingPoint, AR(A_TSFC))", "AR(A_TSFC)", "i2x/Si_t", "mushy_free"),
    # the albedos written without the column shortwave
    ("if (columnSW) {", "if (1) {", "i2x/Si_avsdr", "bl99_mushy_tfrz"),
    # the floe size written without the distribution
    ("if (floe) o[col[I_IFLOE]]", "if (1) o[col[I_IFLOE]]", "i2x/Si_ifloe", "waves_twoway_off"),
    # the meltwater without the frazil adjustment
    ("A(AR(A_MELTW), D(AR(A_FRZADJ), ailohi))", "AR(A_MELTW)", "i2x/Fioi_meltw", "som_linear_salt"),
    # the SST left in Kelvin
    ("S(x[cj[0]], freshWaterFreezingPoint)", "x[cj[0]]", "ocean_coupling/seaSurfaceTemperature", "mushy_free"),
    # linear_salt's slope
    ("tfrz == 1 ? -M(depressT, sss)", "tfrz == 1 ? -M(0.0545, sss)", "ocean_coupling/seaFreezingTemperature",
     "som_linear_salt"),
    # tfrz 'mushy' under BL99 taken as the mush liquidus
    ("return ktherm == 2 ?", "return ktherm >= 1 ?", "ocean_coupling/seaFreezingTemperature", "bl99_mushy_tfrz"),
    # 'constant' taken as -1.8 (Tocnfrz is -1.8 too: the mutation is the minus1p8 literal)
    (": tfrz == 2 ? Tocnfrz : -1.8;", ": tfrz == 2 ? Tocnfrz : -1.9;", "ocean_coupling/seaFreezingTemperature",
     "zero_layer_minus1p8"),
    # the low-salinity frazil's denominator
    ("D(M(sss, sss), M(4.0, frazilSalinityReduction))", "D(M(sss, sss), M(2.0, frazilSalinityReduction))",
     "ocean_coupling/frazilMassAdjust", "mushy_free"),
    # the liquidus's high-salinity branch never taken
    ("Sbr <= SB_LIQ ? 1.0 : 0.0", "1.0", "ocean_coupling/frazilMassAdjust", "mushy_free"),
    # the brine salinity's cold branch never taken
    ("zTin > TB_LIQ ? 1.0 : 0.0", "1.0", "ocean_coupling/frazilMassAdjust", "mushy_free"),
    # the enthalpy's latent term
    ("M(M(S(1.0, phi), c.rhoi), c.Lfresh)", "M(M(phi, c.rhoi), c.Lfresh)", "ocean_coupling/frazilMassAdjust",
     "mushy_free"),
    # BL99's frazil enthalpy without the density
    ("qi0new = -M(densityIce, latentHeatMelting);", "qi0new = -latentHeatMelting;",
     "ocean_coupling/frazilMassAdjust", "bl99_constant"),
    # the frazil's sign
    ("*dst = S(x[cj[0]], rev);", "*dst = A(x[cj[0]], rev);", "ocean_coupling/frazilMassAdjust", "bl99_constant"),
    # a sum's first term dropped
    ("for (int m = 1; m < nc[j]; ++m)", "for (int m = 2; m < nc[j]; ++m)", "aerosols/atmosAerosolFlux",
     "aerosols_bulk"),
])
def test_a_mutated_kernel_is_caught(monkeypatch, original, mutated, key, case):
    real = icc.module_source
    assert real().count(original) == 1
    monkeypatch.setattr(icc, "module_source", lambda: real().replace(original, mutated))
    icc._module_on.cache_clear()
    try:
        assert _mismatches(case, key) > 0
    finally:
        icc._module_on.cache_clear()


@requires_gpu
@pytest.mark.parametrize("name, value, key, case", [
    ("TFFRESH", 273.16, "i2x/Si_t", "mushy_free"),
    ("RHOS", 300.0, "i2x/Si_bpress", "mushy_free"),
    ("GRAVITY", 9.81, "i2x/Si_bpress", "mushy_free"),
    ("LATICE", 3.34e5, "ocean_coupling/frazilMassAdjust", "bl99_constant"),  # seaice_constants' non-CCSM value
    ("CP_ICE", 2106.0, "ocean_coupling/frazilMassAdjust", "mushy_free"),
    ("REF_SALINITY", 5.0, "i2x/Fioi_salt", "mushy_free"),
    ("DEPRESS_T", 0.0545, "ocean_coupling/seaFreezingTemperature", "som_linear_salt"),
    ("FRAZIL_ICE_POROSITY", 0.7, "ocean_coupling/frazilMassAdjust", "mushy_free"),
])
def test_a_mutated_constant_is_caught(monkeypatch, name, value, key, case):
    monkeypatch.setattr(icc, name, value)
    assert _mismatches(case, key) > 0


def _oracle_toolchain():
    e3sm, mct = os.environ.get("WOOF_E3SM_V320_SOURCE"), os.environ.get("WOOF_MCT_SOURCE")
    icepack = os.environ.get("WOOF_ICEPACK_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if (not e3sm or not mct or not icepack or not gfortran
            or not Path(e3sm, "components", "mpas-seaice", "driver").is_dir()
            or not Path(icepack, "columnphysics").is_dir()):
        return None
    return gfortran, Path(e3sm), Path(mct), Path(icepack)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the E3SM coupler oracle build needs gfortran (pixi -e mpas-oracle), "
                           "WOOF_E3SM_V320_SOURCE (an E3SM v3.2.0 tree with components/mpas-seaice and "
                           "mpas-framework), WOOF_MCT_SOURCE (MCT at 82b0071) and WOOF_ICEPACK_SOURCE "
                           "(E3SM-Project/Icepack at cf1d460)")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, e3sm, mct, icepack = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "e3sm_coupler_oracle" / "build.sh"), str(e3sm), str(mct),
                    str(tmp_path / "build"), str(icepack)], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "e3sm_coupler_oracle" / "make_fixture.py"),
                    str(tmp_path / "build"), "--oracle", "seaice", "--out", str(tmp_path / "fixture")], check=True)
    for name in ("oracle-r8.npz", "aborts.json", "inputs.sha256"):
        if name.endswith(".npz"):
            fresh, pinned = np.load(tmp_path / "fixture" / name), np.load(si.FIXTURE_DIR / name)
            assert sorted(fresh.files) == sorted(pinned.files)
            for key in pinned.files:
                assert si.word_mismatches(fresh[key], pinned[key]) == 0, key
        else:
            assert (tmp_path / "fixture" / name).read_text() == (si.FIXTURE_DIR / name).read_text()
