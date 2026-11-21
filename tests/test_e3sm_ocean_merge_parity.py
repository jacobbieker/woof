"""E3SM v3.2.0's ocean import merge (prep_ocn_merge), word for word.

The reference is E3SM v3.2.0's own ``prep_ocn_merge`` (cut verbatim out of
``driver-mct/main/prep_ocn_mod.F90``) on the MCT commit v3.2.0 pins, built
with gfortran -O0 -ffp-contract=off and driven by
``tools/e3sm_coupler_oracle/run_ocean_merge.F90``. The outputs are committed
(tests/data/oracles/e3sm_ocean_merge, PROVENANCE.md). When ``GFORTRAN`` (or
gfortran on PATH), ``WOOF_E3SM_V320_SOURCE`` and ``WOOF_MCT_SOURCE`` are
set, the fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with E3SM only to a tolerance, or only under strict
   numerics: the word-exact gate also runs in fast and strict child
   processes.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: land points (both fractions zero),
   fraction sums other than one, the wave states copied only when coupled two
   ways, the nutrients, the per-category fields, a state copied and then
   merged onto.
4. A plan that drifts from E3SM's: each abort case must be refused with
   E3SM's message, and the isotopes are refused by name.
5. A gate that cannot fail: a mutated kernel must produce mismatches.
6. The default changed: nothing in the coupled step calls this merge.
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
from woof.hex.coupled import prep_ocn_merge_e3sm as pom
from woof.verify import e3sm_ocean_merge_oracle as oo

ROOT = Path(__file__).resolve().parents[1]


def _case(name):
    return next(c for c in oo.cases() if c["name"] == name)


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {case["name"]: oo.input_digest(case) for case in oo.cases()}


def test_fixture_covers_the_branches():
    outputs, aborts = oo.load_fixture()
    cases = {c["name"]: c for c in oo.cases()}
    assert set(outputs) | set(aborts) == set(cases)
    assert set(aborts) == {n for n, c in cases.items() if c["aborts"]}
    for name, calls in outputs.items():
        case = cases[name]
        assert len(calls) == len(case["calls"]) and all(np.isfinite(x).all() for x in calls)
        fr = case["calls"][0]["fractions"]
        names = case["lists"]["fractions"]
        i, o = fr[:, names.index("ifrac")], fr[:, names.index("ofrac")]
        assert ((i + o) == 0).any() and ((i + o) != 1).any() and ((i + o) == 1).any()
    branches, oneway = outputs["branches"], outputs["branches_wave_oneway"]
    x2o = cases["branches"]["lists"]["x2o"]
    wave = x2o.index("Sw_lamult")
    w2x = cases["branches"]["calls"][0]["w2x"][:, 0]
    assert oo.word_mismatches(branches[0][:, wave], w2x) == 0
    assert (oneway[0][:, wave] == 0).all()
    # Sx_shared: copied from the ice, then the ice's share added onto it
    # (numpy's float64 is IEEE, one rounding per operation, as the Fortran's)
    call = cases["branches"]["calls"][0]
    fn = cases["branches"]["lists"]["fractions"]
    i, o = call["fractions"][:, fn.index("ifrac")], call["fractions"][:, fn.index("ofrac")]
    total = i + o
    ifrac = np.where(total != 0, i / np.where(total != 0, total, 1.0), i)
    v = call["i2x"][:, cases["branches"]["lists"]["i2x"].index("Sx_shared")]
    assert oo.word_mismatches(branches[0][:, x2o.index("Sx_shared")], v + v * ifrac) == 0
    # a field two senders carry: the later copy (xao after a2x) lands
    xao = call["xao"][:, cases["branches"]["lists"]["xao"].index("Sa_pbot")]
    assert oo.word_mismatches(branches[0][:, x2o.index("Sa_pbot")], xao) == 0


def test_the_plan_refuses_what_e3sm_stops_on():
    _, aborts = oo.load_fixture()
    for case in oo.cases():
        if not case["aborts"]:
            oo.plan_for(case)
            continue
        with pytest.raises(pom.MergeAbort) as stopped:
            oo.plan_for(case)
        assert str(stopped.value).split(":: FATAL")[-1] in aborts[case["name"]]


def test_the_isotopes_and_unsplittable_names_are_refused():
    lists = dict(oo.GCASE, x2o=list(oo.GCASE["x2o"]) + ["Faxa_rain_16O"])
    with pytest.raises(NotImplementedError, match="isotopes"):
        pom.plan(lists)
    with pytest.raises(ValueError, match="without '_'"):
        pom.plan(dict(oo.GCASE, r2x=list(oo.GCASE["r2x"]) + ["volr"]))
    with pytest.raises(ValueError, match="ofrad"):
        pom.plan(dict(oo.GCASE, fractions=["afrac", "ifrac", "ofrac", "ifrad"]))


def test_the_coupled_step_does_not_call_the_merge():
    """Opt-in: nothing under woof/ imports the merge but its own oracle module."""
    users = [p for p in (ROOT / "woof").rglob("*.py")
             if "prep_ocn_merge_e3sm" in p.read_text(encoding="utf-8")
             and p.name not in ("prep_ocn_merge_e3sm.py", "e3sm_ocean_merge_oracle.py")
             and p.relative_to(ROOT).as_posix() not in ("woof/hex/coupled/interval_e3sm.py",
                                                    "woof/verify/e3sm_coupler_interval_oracle.py")]
    assert users == []


@requires_gpu
def test_gpu_matches_the_e3sm_oracle_word_for_word():
    report = oo.measure()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_e3sm_oracle_word_for_word"], numerics, exact)


@requires_gpu
@pytest.mark.parametrize("original, mutated, field, case", [
    # the fractions not normalised
    ("    if (frac_sum != 0.0) {", "    if (false) {", "Foxx_taux", "gcase"),
    # the shortwave weighted by afrac instead of the radiative afracr
    ("o[ix[X_O_SWNET]] = A(M(A(fswabsv, fswabsi), afracr)", "o[ix[X_O_SWNET]] = A(M(A(fswabsv, fswabsi), afrac)",
     "Foxx_swnet", "gcase"),
    # flux_epbalfact dropped from the snow
    ("o[ix[X_O_SNOW]] = M(o[ix[X_O_SNOW]], flux_epbalfact);", "", "Faxa_snow", "gcase_epbal"),
    # the flood left out of the liquid runoff
    ("A(A(r[ix[X_R_ROFL]], r[ix[X_R_FLOOD]]), g[ix[X_G_ROFL]])", "A(r[ix[X_R_ROFL]], g[ix[X_G_ROFL]])",
     "Foxx_rofl", "gcase"),
    # a replacing merge that adds
    ("else o[ko] = M(a[aindx[ko]], afrac);", "else o[ko] = A(o[ko], M(a[aindx[ko]], afrac));", "Faxa_lwdn",
     "gcase"),
    # x2o not zeroed first
    ("for (int k = 0; k < no; ++k) o[k] = 0.0;", "", "Foxx_taux", "gcase"),
    # the ice's shared-name copies dropped
    ("o[copyTo[c]] = src[copyFrom[c]];", "if (s != 1) o[copyTo[c]] = src[copyFrom[c]];", "Si_ifrac", "gcase"),
    # the per-category fields never written
    ("    if (perCat) {", "    if (false) {", "Sf_afracr", "branches"),
])
def test_a_mutated_kernel_is_caught(monkeypatch, original, mutated, field, case):
    real = pom.module_source
    assert real().count(original) == 1
    monkeypatch.setattr(pom, "module_source", lambda: real().replace(original, mutated))
    pom._module_on.cache_clear()
    try:
        outputs, _ = oo.load_fixture()
        c = _case(case)
        k = c["lists"]["x2o"].index(field)
        got = oo.run_gpu(c)
        assert sum(oo.word_mismatches(g[:, k], w[:, k]) for g, w in zip(got, outputs[case])) > 0
    finally:
        pom._module_on.cache_clear()


def _run_with_plan(case, the_plan):
    import cupy as cp
    out = []
    for call in case["calls"]:
        x2o = cp.asarray(call["x2o"])
        pom.merge(the_plan, {v: cp.asarray(call[v]) for v in pom.SOURCES if case["lists"][v]}, x2o,
                  flux_epbalfact=case["switches"]["flux_epbalfact"])
        out.append(cp.asnumpy(x2o))
    return out


@requires_gpu
@pytest.mark.parametrize("change, field, case", [
    ("ice_atmosphere_flux_merged", "Foxx_lat", "gcase"),  # Faii_lat not skipped
    ("state_replaces", "Sx_shared", "branches"),  # a matched state replacing, not adding
    ("wave_copied_one_way", "Sw_lamult", "branches_wave_oneway"),
])
def test_a_mutated_plan_is_caught(change, field, case):
    """The host plan: the ice/atmosphere fluxes (Faii_*) are skipped, only a
    flux sender's own field replaces, and the waves are copied only when
    coupled two ways."""
    import dataclasses
    c = _case(case)
    honest = oo.plan_for(c)
    k = c["lists"]["x2o"].index(field)
    if change == "ice_atmosphere_flux_merged":
        iindx = list(honest.iindx)
        iindx[k] = c["lists"]["i2x"].index("Faii_lat")
        mutated = dataclasses.replace(honest, iindx=tuple(iindx))
    elif change == "state_replaces":
        imerge = list(honest.imerge)
        imerge[k] = False
        mutated = dataclasses.replace(honest, imerge=tuple(imerge))
    else:
        mutated = pom.plan(c["lists"], wav_ocn_coup="twoway")
    outputs, _ = oo.load_fixture()
    got = _run_with_plan(c, mutated)
    assert sum(oo.word_mismatches(g[:, k], w[:, k]) for g, w in zip(got, outputs[case])) > 0


def _oracle_toolchain():
    e3sm, mct = os.environ.get("WOOF_E3SM_V320_SOURCE"), os.environ.get("WOOF_MCT_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not e3sm or not mct or not gfortran or not Path(e3sm, "driver-mct").is_dir():
        return None
    return gfortran, Path(e3sm), Path(mct)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the E3SM coupler oracle build needs gfortran (pixi -e mpas-oracle), "
                           "WOOF_E3SM_V320_SOURCE (an E3SM v3.2.0 tree) and WOOF_MCT_SOURCE (MCT at 82b0071)")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, e3sm, mct = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "e3sm_coupler_oracle" / "build.sh"), str(e3sm), str(mct),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "e3sm_coupler_oracle" / "make_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    for name in ("oracle-r8.npz", "aborts.json", "inputs.sha256"):
        if name.endswith(".npz"):
            fresh, pinned = np.load(tmp_path / "fixture" / name), np.load(oo.FIXTURE_DIR / name)
            assert sorted(fresh.files) == sorted(pinned.files)
            for key in pinned.files:
                assert oo.word_mismatches(fresh[key], pinned[key]) == 0, key
        else:
            assert (tmp_path / "fixture" / name).read_text() == (oo.FIXTURE_DIR / name).read_text()
