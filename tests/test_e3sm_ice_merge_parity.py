"""E3SM v3.2.0's sea-ice import merge (prep_ice_merge), word for word.

The reference is E3SM v3.2.0's own ``prep_ice_merge`` (cut verbatim out of
``driver-mct/main/prep_ice_mod.F90``) on the MCT commit v3.2.0 pins, built
with gfortran -O0 -ffp-contract=off and driven by
``tools/e3sm_coupler_oracle/run_ice_merge.F90``. The outputs are committed
(tests/data/oracles/e3sm_ice_merge, PROVENANCE.md). When ``GFORTRAN`` (or
gfortran on PATH), ``WOOF_E3SM_V320_SOURCE`` and ``WOOF_MCT_SOURCE`` are
set, the fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with E3SM only to a tolerance, or only under strict
   numerics: the word-exact gate also runs in fast and strict child
   processes.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: the waves copied only when coupled two
   ways, a field two senders carry (the later copy lands), a sender's own
   Faxa_rain overwritten by the sum, an x2i field no sender carries (kept,
   as the merge does not zero x2i), signed zeros in the sums.
4. A plan that drifts from E3SM's: each stop must be MCT's for the same
   field, and the isotopes are refused by name.
5. A gate that cannot fail: mutated kernels and plans must mismatch.
6. The default changed: nothing calls this merge.

QUIRKS (E3SM v3.2.0, prep_ice_mod.F90)
--------------------------------------
- x2i is not zeroed before the copies: a field no sender carries keeps the
  previous call's word (kept; Sx_nobody).
- The sums are scaled by flux_epbalfact in a second statement: two
  roundings, (a + b) then * f (kept).
- An x2i with Faxa_rain_16O but an a2x without the isotope's fields reads
  a2x's index 0 (quiet lookups, out of bounds); the isotopes are refused.
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
from woof.hex.coupled import prep_ice_merge_e3sm as pim
from woof.verify import e3sm_ice_merge_oracle as im

ROOT = Path(__file__).resolve().parents[1]


def _case(name):
    return next(c for c in im.cases() if c["name"] == name)


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (im.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {case["name"]: im.input_digest(case) for case in im.cases()}


def test_fixture_covers_the_branches():
    outputs, aborts = im.load_fixture()
    cases = {c["name"]: c for c in im.cases()}
    assert set(outputs) | set(aborts) == set(cases)
    assert set(aborts) == {n for n, c in cases.items() if c["aborts"]}
    for name, calls in outputs.items():
        assert len(calls) == len(cases[name]["calls"]) and all(np.isfinite(x).all() for x in calls)
    c = cases["branches_wave_twoway"]
    x2i, call = c["lists"]["x2i"], c["calls"][0]
    two, one = outputs["branches_wave_twoway"][0], outputs["branches_wave_oneway"][0]
    # the waves: copied two ways only
    w = call["w2x"][:, c["lists"]["w2x"].index("Sw_ustokes")]
    assert im.word_mismatches(two[:, x2i.index("Sw_ustokes")], w) == 0
    o1 = cases["branches_wave_oneway"]
    assert im.word_mismatches(one[:, x2i.index("Sw_ustokes")],
                              o1["calls"][0]["x2i"][:, x2i.index("Sw_ustokes")]) == 0
    # w2x's So_u is shared too, and lands after o2x's
    assert im.word_mismatches(two[:, x2i.index("So_u")], call["w2x"][:, c["lists"]["w2x"].index("So_u")]) == 0
    # a field two senders carry: a2x's copy after o2x's
    assert im.word_mismatches(two[:, x2i.index("Sx_shared")],
                              call["a2x"][:, c["lists"]["a2x"].index("Sx_shared")]) == 0
    # no sender: x2i's own word
    assert im.word_mismatches(two[:, x2i.index("Sx_nobody")], call["x2i"][:, x2i.index("Sx_nobody")]) == 0
    # rain: the sum, then the factor (numpy's float64 rounds each operation, as the Fortran)
    a = c["lists"]["a2x"]
    rain = (call["a2x"][:, a.index("Faxa_rainc")] + call["a2x"][:, a.index("Faxa_rainl")]) * 1.0371
    assert im.word_mismatches(two[:, x2i.index("Faxa_rain")], rain) == 0
    # signed zeros: -0 + -0 is -0 (snow), -0 + 0 is +0 (rain)
    g = outputs["gcase"][0]
    gx = cases["gcase"]["lists"]["x2i"]
    assert np.signbit(g[1, gx.index("Faxa_snow")]) and not np.signbit(g[1, gx.index("Faxa_rain")])


def test_the_plan_refuses_what_mct_stops_on():
    _, aborts = im.load_fixture()
    for case in im.cases():
        if not case["aborts"]:
            im.plan_for(case)
            continue
        with pytest.raises(pim.MergeAbort) as stopped:
            im.plan_for(case)
        assert str(stopped.value).split(":: FATAL")[-1] in aborts[case["name"]]


def test_several_missing_fields_stop_at_the_fortrans_first():
    def drop(lists, **gone):
        return {k: [f for f in v if f not in gone.get(k, ())] for k, v in lists.items()}
    for gone, first in ((dict(a2x=("Faxa_snowl",), x2i=("Fixx_rofi",)), "Faxa_snowl"),
                        (dict(a2x=("Faxa_rainl",), g2x=("Figg_rofi",)), "Faxa_rainl"),
                        (dict(r2x=("Firr_rofi",), g2x=("Figg_rofi",)), "Figg_rofi"),
                        (dict(x2i=("Faxa_snow", "Faxa_rain")), "Faxa_rain")):
        with pytest.raises(pim.MergeAbort, match=f'"{first}"'):
            pim.plan(drop(im.GCASE, **gone))


def test_the_isotopes_and_bad_lists_are_refused():
    with pytest.raises(NotImplementedError, match="isotopes.*Faxa_rain_HDO"):
        pim.plan(dict(im.GCASE, x2i=list(im.GCASE["x2i"]) + ["Faxa_snow_HDO", "Faxa_rain_HDO"]))
    # without Faxa_rain_<iso>, E3SM's isotope branch is off: Faxa_snow_<iso> is only copied
    snow = pim.plan(dict(im.GCASE, a2x=list(im.GCASE["a2x"]) + ["Faxa_snow_18O"],
                         x2i=list(im.GCASE["x2i"]) + ["Faxa_snow_18O"]))
    assert (1, len(im.GCASE["a2x"]), len(im.GCASE["x2i"])) in snow.copies
    with pytest.raises(ValueError, match="repeats"):
        pim.plan(dict(im.GCASE, o2x=list(im.GCASE["o2x"]) + ["So_t"]))
    with pytest.raises(ValueError, match="wav_ice_coup"):
        pim.plan(im.GCASE, wav_ice_coup="both")


def test_nothing_calls_the_merge():
    users = [p for p in (ROOT / "woof").rglob("*.py")
             if "prep_ice_merge_e3sm" in p.read_text(encoding="utf-8")
             and p.name not in ("prep_ice_merge_e3sm.py", "e3sm_ice_merge_oracle.py", "interval_e3sm.py",
                                "e3sm_coupler_interval_oracle.py")]
    assert users == []


@requires_gpu
def test_gpu_matches_the_e3sm_oracle_word_for_word():
    report = im.measure()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_e3sm_oracle_word_for_word"], numerics, exact)


@requires_gpu
def test_bad_arrays_are_refused():
    import cupy as cp
    c = _case("gcase")
    the_plan = im.plan_for(c)
    call = c["calls"][0]
    vectors = {v: cp.asarray(call[v]) for v in pim.SOURCES if c["lists"][v]}
    x2i = cp.asarray(call["x2i"])
    with pytest.raises(TypeError, match="cupy"):
        pim.merge(the_plan, vectors, call["x2i"], flux_epbalfact=1.0)
    with pytest.raises(ValueError, match="a2x must be"):
        pim.merge(the_plan, dict(vectors, a2x=vectors["a2x"][:-1]), x2i, flux_epbalfact=1.0)
    with pytest.raises(ValueError, match="C-contiguous"):
        pim.merge(the_plan, vectors, cp.asfortranarray(x2i), flux_epbalfact=1.0)
    with pytest.raises(TypeError, match="plan"):
        pim.merge(c["lists"], vectors, x2i, flux_epbalfact=1.0)


@requires_gpu
def test_aliased_arrays_are_refused():
    import cupy as cp
    c = _case("gcase")
    the_plan = im.plan_for(c)
    call = c["calls"][0]
    nx, no = len(c["lists"]["x2i"]), len(c["lists"]["o2x"])
    vectors = {v: cp.asarray(call[v]) for v in pim.SOURCES if c["lists"][v]}
    flat = cp.zeros(c["lsize"] * (nx + no), dtype=np.float64)
    x2i_alias = flat[:c["lsize"] * nx].reshape(c["lsize"], nx)
    o2x_alias = flat[c["lsize"] * nx - 1:c["lsize"] * nx - 1 + c["lsize"] * no].reshape(c["lsize"], no)
    with pytest.raises(ValueError, match="share memory"):
        pim.merge(the_plan, dict(vectors, o2x=o2x_alias), x2i_alias, flux_epbalfact=1.0)


@requires_gpu
@pytest.mark.parametrize("original, mutated, field, case", [
    # flux_epbalfact dropped from the snow
    ("x[ix[X_I_SNOW]] = M(x[ix[X_I_SNOW]], flux_epbalfact);", "", "Faxa_snow", "gcase_epbal"),
    # the factor applied to one term only
    ("x[ix[X_I_RAIN]] = A(a[ix[X_A_RAINC]], a[ix[X_A_RAINL]]);",
     "x[ix[X_I_RAIN]] = A(a[ix[X_A_RAINC]], M(a[ix[X_A_RAINL]], flux_epbalfact));", "Faxa_rain", "gcase_epbal"),
    # the runoff from the land ice only
    ("A(g2x[(size_t)n * ng + ix[X_G_ROFI]], r2x[(size_t)n * nr + ix[X_R_ROFI]])",
     "g2x[(size_t)n * ng + ix[X_G_ROFI]]", "Fixx_rofi", "gcase"),
    # the ocean's copies dropped
    ("x[copyTo[c]] = src[(size_t)n * width + copyFrom[c]];",
     "if (copySource[c] != 0) x[copyTo[c]] = src[(size_t)n * width + copyFrom[c]];", "So_t", "gcase"),
    # the copies in reverse order
    ("for (int c = 0; c < nCopies; ++c) {", "for (int c = nCopies - 1; c >= 0; --c) {", "Sx_shared",
     "branches_wave_twoway"),
    # x2i zeroed first
    ("    for (int c = 0; c < nCopies; ++c) {", "    for (int k = 0; k < ni; ++k) x[k] = 0.0;\n"
     "    for (int c = 0; c < nCopies; ++c) {", "Sx_nobody", "branches_wave_twoway"),
])
def test_a_mutated_kernel_is_caught(monkeypatch, original, mutated, field, case):
    real = pim.module_source
    assert real().count(original) == 1
    monkeypatch.setattr(pim, "module_source", lambda: real().replace(original, mutated))
    pim._module_on.cache_clear()
    try:
        outputs, _ = im.load_fixture()
        c = _case(case)
        k = c["lists"]["x2i"].index(field)
        got = im.run_gpu(c)
        assert sum(im.word_mismatches(g[:, k], w[:, k]) for g, w in zip(got, outputs[case])) > 0
    finally:
        pim._module_on.cache_clear()


@requires_gpu
def test_the_waves_copied_one_way_are_caught():
    import cupy as cp
    c = _case("branches_wave_oneway")
    mutated = pim.plan(c["lists"], wav_ice_coup="twoway")
    outputs, _ = im.load_fixture()
    k = c["lists"]["x2i"].index("Sw_ustokes")
    mismatches = 0
    for call, want in zip(c["calls"], outputs[c["name"]]):
        x2i = cp.asarray(call["x2i"])
        pim.merge(mutated, {v: cp.asarray(call[v]) for v in pim.SOURCES if c["lists"][v]}, x2i,
                  flux_epbalfact=c["flux_epbalfact"])
        mismatches += im.word_mismatches(cp.asnumpy(x2i)[:, k], want[:, k])
    assert mismatches > 0


def _oracle_toolchain():
    e3sm, mct = os.environ.get("WOOF_E3SM_V320_SOURCE"), os.environ.get("WOOF_MCT_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not e3sm or not mct or not gfortran or not Path(e3sm, "driver-mct", "main", "prep_ice_mod.F90").is_file():
        return None
    return gfortran, Path(e3sm), Path(mct)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the E3SM coupler oracle build needs gfortran (pixi -e mpas-oracle), "
                           "WOOF_E3SM_V320_SOURCE (an E3SM v3.2.0 tree) and WOOF_MCT_SOURCE (MCT at 82b0071)")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, e3sm, mct = _oracle_toolchain()
    tools = ROOT / "tools" / "e3sm_coupler_oracle"
    subprocess.run(["bash", str(tools / "build.sh"), str(e3sm), str(mct), str(tmp_path / "build")], check=True,
                   env={**os.environ, "GFORTRAN": gfortran}, capture_output=True)
    subprocess.run([sys.executable, str(tools / "make_fixture.py"), str(tmp_path / "build"), "--oracle", "icemerge",
                    "--out", str(tmp_path / "fixture")], check=True, cwd=ROOT)
    for name in ("oracle-r8.npz", "aborts.json", "inputs.sha256"):
        if name.endswith(".npz"):
            with np.load(tmp_path / "fixture" / name) as fresh, np.load(im.FIXTURE_DIR / name) as pinned:
                assert sorted(fresh.files) == sorted(pinned.files)
                for key in pinned.files:
                    assert im.word_mismatches(fresh[key], pinned[key]) == 0, key
        else:
            assert (tmp_path / "fixture" / name).read_text() == (im.FIXTURE_DIR / name).read_text()
