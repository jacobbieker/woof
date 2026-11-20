"""MPAS-Seaice's delta-Eddington column shortwave with a computed surface
temperature (colpkg stage 5b), word for word against v8.4.2.

The reference is MPAS-Model v8.4.2's ``column_radiation`` and
``seaice_init_column_shortwave`` (``mpas_seaice_column.F``) and the column
package under them (``colpkg_init_orbit``, ``colpkg_step_radiation`` with
``run_dEdd``: ``compute_coszen``, ``shortwave_dEdd``, ``compute_dEdd``,
``solution_dEdd``), compiled with gfortran -O0 -ffp-contract=off at RKIND 8.
``tools/mpas_seaice_oracle/run_column.F90`` (stages ``radiation``, which calls
``colpkg_init_orbit`` first as ``seaice_init_column_shortwave`` does, and
``init_shortwave``) runs them. The oracle's outputs are committed
(tests/data/oracles/mpas_seaice_column_dedd, PROVENANCE.md), so the GPU
comparison runs without gfortran. The orbit and the solar zenith angle call
glibc's sin, cos, atan and asin, LGPL and not correctly rounded, so every case
runs only with the opt-in transcription (``WOOF_GLIBC239_LIBM64=1``, source
checkouts only), and the default build refuses the delta-Eddington shortwave.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the routines may write must be bit-identical, in fast and in strict
   numerics.
2. v8.4.2's quirks "fixed": the solar zenith cosine raised to puny and
   written back; zero-heat-capacity ice folding its top layers' absorption
   into the surface per surface type; categories without ice keeping their
   albedo words.
3. A gate that cannot fail: a mutated evaluation order must produce
   mismatches.
4. glibc's trig running silently in the default build: it is refused by name.
5. A case generator whose bytes drift: the input digests are pinned.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_seaice import column_radiation_v842 as crad
from woof.verify import mpas_seaice_column_oracle as co

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = co.DEDD_FIXTURE_DIR
PUNY = 1.0e-11


def _case(name):
    return co.make_case(*next(row for row in co.DEDD_CASES if row[0] == name))


@pytest.fixture
def glibc239(monkeypatch):
    """The opt-in glibc 2.39 binary64 sin/cos/atan/asin transcription (LGPL, source checkout only)."""
    from woof.core import glibc239_gpu
    if not glibc239_gpu.header_path().is_file():
        pytest.skip("the opt-in glibc 2.39 transcription is only in a source checkout")
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")
    crad._module_on.cache_clear()
    yield
    crad._module_on.cache_clear()


@pytest.fixture
def default_build(monkeypatch):
    monkeypatch.delenv("WOOF_GLIBC239_LIBM64", raising=False)
    crad._module_on.cache_clear()
    yield
    crad._module_on.cache_clear()


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (FIXTURE / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", co.DEDD_CASES)}


def test_fixture_covers_the_branches():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", co.DEDD_CASES)}
    assert set(logs) == set(cases)
    assert not any(log["log"] for log in logs.values())
    for name, out in fixture.items():
        for key, value in out.items():
            assert np.isfinite(value).all(), (name, key)
    n_solve = co.MESH["nCellsSolve"]

    def got(name, key):
        return fixture[name][key]

    # compute_coszen in every owned cell; shortwave_dEdd raises it to puny under ice with sun
    coszen = np.concatenate([got(n, "shortwave/solarZenithAngleCosine")[:n_solve] for n in cases])
    assert (coszen == PUNY).any() and (coszen < 0).any() and (coszen > 0.1).any()
    for name in cases:
        halo = cases[name]["pools"]["shortwave"]["solarZenithAngleCosine"][n_solve:]
        assert (got(name, "shortwave/solarZenithAngleCosine")[n_solve:] == halo).all()
    # no sun (row 7): ice-covered categories get albedo 1
    alb = got("dedd_default_level_ponds", "shortwave/albedoVisibleDirectCategory")
    area = cases["dedd_default_level_ponds"]["pools"]["tracers"]["iceAreaCategory"][:, :, 0]
    assert (alb[7][area[7] > PUNY] == 1.0).all()
    assert ((alb[:n_solve] > 0) & (alb[:n_solve] < 1)).any()
    # the snow fraction: whole with hs0 = 0, partial with a transition depth, 0 below hs_min
    snowfrac = got("dedd_no_ponds_snow_transition", "shortwave/snowFractionCategory")[:n_solve]
    assert ((snowfrac > 0) & (snowfrac < 1)).any() and (snowfrac == 1).any() and (snowfrac == 0).any()
    # bare, snow and pond albedos
    for name in ("dedd_default_level_ponds", "dedd_cesm_ponds_snow_transition", "dedd_topo_ponds"):
        for key in ("bareIceAlbedoCategory", "snowAlbedoCategory", "pondAlbedoCategory",
                    "effectivePondAreaCategory"):
            assert (got(name, f"shortwave/{key}")[:n_solve] > 0).any(), (name, key)
    # no ponds: set_pond's effective area (fT > 0 near the melting point) and no pond albedo under ice
    nopond = "dedd_no_ponds_snow_transition"
    assert (got(nopond, "shortwave/effectivePondAreaCategory")[:n_solve] > 0).any()
    ice = cases[nopond]["pools"]["tracers"]["iceAreaCategory"][:n_solve, :, 0] > PUNY
    assert (got(nopond, "shortwave/pondAlbedoCategory")[:n_solve][ice] == 0).all()
    # level ponds write their snow depth difference
    assert "ponds/pondSnowDepthDifference" in fixture["dedd_default_level_ponds"]
    # zero-layer ice: the top layers' absorption folded into the surface
    for name in ("dedd_zero_layer", "dedd_zero_layer_topo_ponds"):
        assert (got(name, "shortwave/interiorShortwaveFlux")[:n_solve] == 0).all()
        assert (got(name, "shortwave/absorbedShortwaveIceLayer")[:n_solve, :, 0] == 0).all()
        assert (got(name, "shortwave/surfaceShortwaveFlux")[:n_solve] > 0).any()
    # with heat capacity every ice layer absorbs, down to the twelfth
    iswabs = got("dedd_bl99_twelve_layers", "shortwave/absorbedShortwaveIceLayer")[:n_solve]
    assert (iswabs[:, :, -1] > 0).any() and (iswabs[:, :, 0] > 0).any()
    assert (got("dedd_bl99_twelve_layers", "shortwave/absorbedShortwaveSnowLayer")[:n_solve] > 0).any()
    # bioTracerShortwave zeroed in the owned cells
    bio = got("dedd_default_level_ponds", "biogeochemistry/bioTracerShortwave")
    assert (bio[:n_solve] == 0).all()


def test_the_default_build_refuses_glibc_s_trig(default_build):
    case = _case("dedd_default_level_ponds")
    with pytest.raises(ConfigurationRefusal, match="sin, cos, atan and asin.*WOOF_GLIBC239_LIBM64=1"):
        crad.refuse_unported(case["config"], case["dims"])


def test_the_opt_in_carries_the_delta_eddington_shortwave(glibc239):
    case = _case("dedd_default_level_ponds")
    crad.refuse_unported(case["config"], case["dims"])
    # more categories than ice layers is only shortwave_ccsm3's problem
    crad.refuse_unported(case["config"], {**case["dims"], "nCategories": 8})
    for overrides, match in (({"config_use_aerosols": True}, "aerosols"),
                             ({"config_use_shortwave_bioabsorption": True}, "bioabsorption"),
                             ({"config_use_column_biogeochemistry": True}, "biogeochemistry")):
        with pytest.raises(ConfigurationRefusal, match=match):
            crad.refuse_unported({**case["config"], **overrides}, case["dims"])


def test_the_default_build_refuses_the_orbit(default_build):
    with pytest.raises(ConfigurationRefusal, match="sin, cos and atan.*WOOF_GLIBC239_LIBM64=1"):
        crad._orbit()


def test_the_day_of_the_year_follows_the_calendar():
    assert crad.day_of_year("2000-03-01_00:00:00", "gregorian") == 61.0
    assert crad.day_of_year("2000-03-01_00:00:00", "gregorian_noleap") == 60.0
    assert crad.day_of_year("2001-12-31_23:59:59", "gregorian_noleap") == 365.0
    assert crad.day_of_year("2000-02-29_07:00:00", "gregorian") == 60.0
    with pytest.raises(ConfigurationRefusal, match="29 February"):
        crad.day_of_year("2000-02-29_07:00:00", "gregorian_noleap")
    with pytest.raises(ConfigurationRefusal, match="config_calendar_type"):
        crad.day_of_year("2001-06-21_12:00:00", "360_day")


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word(glibc239):
    report = co.measure_dedd()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


_CHILD = """import json
from woof.hex import numerics
from woof.verify import mpas_seaice_column_oracle as co
report = co.measure_dedd()
print(json.dumps({"mode": numerics.mode(), "mismatches": sum(sum(f.values()) for f in report.values()),
                  "cases": len(report)}))
"""


@requires_gpu
def test_fast_and_strict_numerics_both_match_the_oracle(glibc239):
    rows = []
    for mode in ("fast", "strict"):
        env = {**os.environ, "WOOF_HEX_NUMERICS": mode, "WOOF_GLIBC239_LIBM64": "1",
               "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH"))))}
        out = subprocess.check_output([sys.executable, "-c", _CHILD], env=env, cwd=ROOT)
        rows.append(json.loads(out.decode().strip().splitlines()[-1]))
    assert [row["mode"] for row in rows] == ["fast", "strict"]
    assert all(row["mismatches"] == 0 and row["cases"] == len(co.DEDD_CASES) for row in rows), rows


@requires_gpu
@pytest.mark.parametrize("original, mutated, field, case", [
    # shr_orb_decl's mean longitude through 2 pi / 365 first
    ("const double lambm = A(lambm0, D(M(M(S(ydayp1, 80.5), 2.0), PI), 365.0));",
     "const double lambm = A(lambm0, M(S(ydayp1, 80.5), D(M(2.0, PI), 365.0)));",
     "shortwave/solarZenithAngleCosine", "dedd_cesm_ponds_snow_transition"),
    # column_radiation's longitude wrap dropped (cos is periodic, its words are not)
    ("if (tlon > DEDD_PII) tlon = S(tlon, M(2.0, DEDD_PII));", "",
     "shortwave/solarZenithAngleCosine", "dedd_bl99_level_ponds_noon"),
    # the Gaussian integration's weight applied first
    ("smr = A(smr, M(M(mu, rdr), gwt));", "smr = A(smr, M(mu, M(rdr, gwt)));",
     "shortwave/albedoVisibleDiffuseCategory", "dedd_default_level_ponds"),
    # the snow grain table interpolated as an increment
    ("Qs = A(M(DEDD_QS_TAB[nr - 1][ns], S(1.0, delr)), M(DEDD_QS_TAB[nr][ns], delr));",
     "Qs = A(DEDD_QS_TAB[nr - 1][ns], M(S(DEDD_QS_TAB[nr][ns], DEDD_QS_TAB[nr - 1][ns]), delr));",
     "shortwave/shortwaveLayerPenetration", "dedd_default_level_ponds"),
    # the negative tuning's scattering clamp dropped
    ("if (clamp) sigp = sw_max(sigp, 0.0);", "", "shortwave/pondAlbedoCategory", "dedd_negative_tuning"),
    # zero heat capacity: the fold reassociated
    ("*fswsfc = A(A(*fswsfc, Iswabs[0]), Sswabs[0]);", "*fswsfc = A(*fswsfc, A(Iswabs[0], Sswabs[0]));",
     "shortwave/surfaceShortwaveFlux", "dedd_zero_layer"),
    # the level ponds' lid melt fraction as a difference
    ("fpn = M(S(1.0, ffracnAll[cn + n]), fpn);", "fpn = S(fpn, M(ffracnAll[cn + n], fpn));",
     "shortwave/effectivePondAreaCategory", "dedd_default_level_ponds"),
    # set_pond's product reassociated
    ("fpn = M(M(0.3, fT), S(1.0, fsn));", "fpn = M(0.3, M(fT, S(1.0, fsn)));",
     "shortwave/effectivePondAreaCategory", "dedd_no_ponds_snow_transition"),
    # the solar zenith cosine not written back
    ("coszenAll[c] = coszen;", "", "shortwave/solarZenithAngleCosine", "dedd_default_level_ponds"),
])
def test_a_mutated_evaluation_order_is_caught(monkeypatch, glibc239, original, mutated, field, case):
    source = crad.module_source
    assert source().count(original) == 1
    monkeypatch.setattr(crad, "module_source", lambda: source().replace(original, mutated))
    crad._module_on.cache_clear()
    try:
        got, _ = co.run_gpu(_case(case))
        want = co.load_fixture(FIXTURE)[case][field]
        assert co.word_mismatches(got[field], want) > 0
    finally:
        crad._module_on.cache_clear()


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_seaice").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None or platform.machine() != "x86_64"
                    or platform.libc_ver() != ("glibc", "2.44"),
                    reason="the fixture is the x86-64 glibc 2.44 host's (its exp, sin, cos, atan and asin, "
                           "PROVENANCE.md); it needs gfortran (pixi -e mpas-oracle) and WOOF_MPAS_V842_SOURCE")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_column_fixture.py"),
                    str(tmp_path / "build"), "--cases", "dedd", "--out", str(tmp_path / "fixture")],
                   check=True)
    fresh = np.load(tmp_path / "fixture" / "oracle-r8.npz")
    pinned = np.load(FIXTURE / "oracle-r8.npz")
    assert sorted(fresh.files) == sorted(pinned.files)
    for key in pinned.files:
        assert co.word_mismatches(fresh[key], pinned[key]) == 0, key
    for name in ("column-logs.json", "inputs.sha256"):
        assert (tmp_path / "fixture" / name).read_text() == (FIXTURE / name).read_text()
