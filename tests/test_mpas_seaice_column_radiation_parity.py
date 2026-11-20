"""MPAS-Seaice's column shortwave without the delta-Eddington solution
(colpkg stage 5a), word for word against v8.4.2.

The reference is MPAS-Model v8.4.2's ``column_prep_radiation``,
``column_radiation`` and ``seaice_init_column_shortwave``
(``mpas_seaice_column.F``) and the column package under them
(``colpkg_prep_radiation``, ``colpkg_step_radiation`` with
``shortwave_ccsm3``), compiled with gfortran -O0 -ffp-contract=off at RKIND 8.
``tools/mpas_seaice_oracle/run_column.F90`` (stages ``prep_radiation``,
``radiation`` and ``init_shortwave``) runs them. The oracle's outputs are
committed (tests/data/oracles/mpas_seaice_column_radiation, PROVENANCE.md),
so the GPU comparison runs without gfortran. ``absorbed_solar``'s exp is
glibc's, which the GPU transcribes under MIT (glibc_flt64.cuh); the ccsm3
albedos' atan is glibc's, LGPL and not correctly rounded, so those cases run
only with the opt-in transcription (``WOOF_GLIBC239_LIBM64=1``, source
checkouts only), and the default build refuses them.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the routines may write must be bit-identical.
2. v8.4.2's quirks "fixed": absorbed_solar gets ncat as its layer count
   (the layers below the ncat-th keep 0); shortwave_ccsm3 sets the solar
   zenith cosine to 0.5; colpkg_step_radiation zeroes bioTracerShortwave;
   seaice_init_column_shortwave adds the bare-ice, snow and pond albedos
   onto their words without resetting them.
3. A gate that cannot fail: a mutated evaluation order must produce
   mismatches.
4. A configuration the port does not carry running silently: each is
   refused by name, glibc's atan in the default build included.
5. A case generator whose bytes drift: the input digests are pinned.
"""
from __future__ import annotations

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
FIXTURE = co.RADIATION_FIXTURE_DIR


def _case(name):
    return co.make_case(*next(row for row in co.RADIATION_CASES if row[0] == name))


@pytest.fixture
def glibc239(monkeypatch):
    """The opt-in glibc 2.39 binary64 atan transcription (LGPL, source checkout only)."""
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
    assert recorded == {f"{case['name']}/r8": co.input_digest(case)
                        for case in co.cases("r8", co.RADIATION_CASES)}


def test_fixture_covers_the_branches():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", co.RADIATION_CASES)}
    assert set(logs) == set(cases)
    assert not any(log["log"] for log in logs.values())
    for name, out in fixture.items():
        for key, value in out.items():
            assert np.isfinite(value).all(), (name, key)
    n_solve = co.MESH["nCellsSolve"]

    def got(name, key):
        return fixture[name][key]

    # shortwave_ccsm3 sets the solar zenith cosine to 0.5 in every owned cell
    for name in ("rad_ccsm3_constant_albedos", "rad_ccsm3_albedos", "init_shortwave_ccsm3_constant"):
        coszen = got(name, "shortwave/solarZenithAngleCosine")
        assert (coszen[:n_solve] == 0.5).all()
        assert (coszen[n_solve:] == cases[name]["pools"]["shortwave"]["solarZenithAngleCosine"][n_solve:]).all()
    # ...and leaves it alone with a coupled surface temperature
    assert "shortwave/solarZenithAngleCosine" not in fixture["rad_coupled_dedd_topo_ponds"]
    # absorbed_solar's layer count is ncat: five categories light five of seven layers
    iswabs = got("rad_ccsm3_constant_albedos", "shortwave/absorbedShortwaveIceLayer")
    assert (iswabs[:n_solve, :, 5:] == 0).all() and (iswabs[:n_solve, :, :5] > 0).any()
    penetration = got("rad_ccsm3_constant_albedos", "shortwave/shortwaveLayerPenetration")
    assert (penetration[:n_solve, :, 6:] == 0).all()
    # with as many categories as layers every layer is lit
    assert (got("rad_ccsm3_constant_mushy_five_layers", "shortwave/absorbedShortwaveIceLayer")[0, -1] > 0).any()
    # zero-layer ice absorbs nothing in its interior
    zero = "rad_ccsm3_constant_zero_layer"
    assert (got(zero, "shortwave/interiorShortwaveFlux")[:n_solve] == 0).all()
    assert (got(zero, "shortwave/absorbedShortwaveIceLayer")[:n_solve, :, 0] == 0).all()
    # the constant albedos: warm snow at the melting point (row 0), ocean albedo with no ice (row 1)
    constant = got("rad_ccsm3_constant_albedos", "shortwave/albedoVisibleDiffuseCategory")
    assert set(np.unique(constant[:n_solve])) <= {0.06, 0.68, 0.70, 0.77, 0.81}
    assert {0.77, 0.81} <= set(np.unique(constant[:n_solve]))
    assert (constant[1] == 0.06).all()
    # the ccsm3 albedos vary with thickness and temperature
    assert len(np.unique(got("rad_ccsm3_albedos", "shortwave/albedoVisibleDiffuseCategory")[:n_solve])) > 20
    # bioTracerShortwave zeroed in the owned cells
    for name in ("rad_ccsm3_albedos", "rad_coupled_dedd_topo_ponds"):
        bio = got(name, "biogeochemistry/bioTracerShortwave")
        assert (bio[:n_solve] == 0).all()
    # the topo ponds' effective area with a coupled surface temperature, and no shortwave
    ponds = "rad_coupled_dedd_topo_ponds"
    assert (got(ponds, "shortwave/effectivePondAreaCategory")[:n_solve] > 0).any()
    assert (got(ponds, "shortwave/surfaceShortwaveFlux")[:n_solve] == 0).all()
    # prep_radiation: the scaling factor is 1 without ice (rows 1, 8) or a previous net shortwave (row 6)
    scale = got("prep_radiation_default", "shortwave/shortwaveScalingFactor")
    assert (scale[[1, 6, 8]] == 1.0).all() and (scale[[0, 2, 3, 4, 5, 7]] != 1.0).all()
    # init_shortwave adds onto the cells' bare-ice albedo (not reset): an ice-free cell keeps its word
    for name in ("init_shortwave_ccsm3_constant", "init_shortwave_coupled_topo_ponds"):
        start = cases[name]["pools"]["shortwave"]["bareIceAlbedoCell"]
        assert got(name, "shortwave/bareIceAlbedoCell")[1] == start[1]


@pytest.mark.parametrize("overrides, match", [
    ({"config_use_aerosols": True}, "aerosols"),
    ({"config_use_shortwave_bioabsorption": True}, "bioabsorption"),
    ({"config_use_column_biogeochemistry": True}, "biogeochemistry"),
])
def test_unported_configurations_are_refused_by_name(overrides, match):
    case = _case("rad_ccsm3_constant_albedos")
    with pytest.raises(ConfigurationRefusal, match=match):
        crad.refuse_unported({**case["config"], **overrides}, case["dims"])


def test_more_categories_than_ice_layers_are_refused():
    case = _case("rad_ccsm3_constant_albedos")
    with pytest.raises(ConfigurationRefusal, match="ncat as absorbed_solar's nilyr"):
        crad.refuse_unported(case["config"], {**case["dims"], "nCategories": 8})
    # no shortwave with a coupled surface temperature: nothing to refuse
    coupled = _case("rad_coupled_dedd_topo_ponds")
    crad.refuse_unported(coupled["config"], {**coupled["dims"], "nCategories": 8})


def test_a_state_built_from_other_configs_is_refused():
    from types import SimpleNamespace
    case = _case("rad_ccsm3_constant_albedos")
    calc = bool(case["config"]["config_calc_surface_temperature"])
    crad.refuse_state_mismatch(case["config"], SimpleNamespace(calc_tsfc=calc))
    with pytest.raises(ConfigurationRefusal, match="ColumnState was initialised"):
        crad.refuse_state_mismatch(case["config"], SimpleNamespace(calc_tsfc=not calc))


def test_the_default_build_refuses_glibc_s_atan(default_build):
    case = _case("rad_ccsm3_albedos")
    with pytest.raises(ConfigurationRefusal, match="atan.*WOOF_GLIBC239_LIBM64=1"):
        crad.refuse_unported(case["config"], case["dims"])


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word_in_the_default_build(default_build):
    rows = tuple(row for row in co.RADIATION_CASES if row[0] not in co.RADIATION_TRIG_CASES)
    report = co.measure_radiation(rows=rows)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word_with_glibc_s_atan(glibc239):
    report = co.measure_radiation()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("original, mutated, field, case", [
    # absorbed_solar's layer count "fixed" to nilyr
    ("sw_absorbed_solar(heat_capacity, ncat,", "sw_absorbed_solar(heat_capacity, nilyr,",
     "shortwave/absorbedShortwaveIceLayer", "rad_ccsm3_constant_albedos"),
    # compute_albedos' intent(out) snow albedos on bare ice "fixed" to 0 (v8.4.2 keeps the caller's albocn)
    ("double alvdrns = SW_ALBOCN, alidrns = SW_ALBOCN, alvdfns = SW_ALBOCN, alidfns = SW_ALBOCN;",
     "double alvdrns = 0.0, alidrns = 0.0, alvdfns = 0.0, alidfns = 0.0;",
     "shortwave/snowAlbedoCategory", "rad_ccsm3_albedos"),
    # the snow-ice albedo blend as an increment
    ("*alvdfn = A(M(*alvdfni, S(1.0, asnow)), M(*alvdfns, asnow));",
     "*alvdfn = A(*alvdfni, M(S(*alvdfns, *alvdfni), asnow));",
     "shortwave/albedoVisibleDiffuseCategory", "rad_ccsm3_albedos"),
    # the penetrating visible flux's product reassociated
    ("double fswpenvdr = M(M(M(swvdr, S(1.0, alvdrni)), S(1.0, asnow)), SW_I0VIS);",
     "double fswpenvdr = M(M(swvdr, S(1.0, alvdrni)), M(S(1.0, asnow), SW_I0VIS));",
     "shortwave/shortwaveLayerPenetration", "rad_ccsm3_constant_albedos"),
    # Beer's law exponent reassociated
    ("tranbot = glibc_exp(-M(M(SW_KAPPAV, hilyr), (double)k));",
     "tranbot = glibc_exp(-M(SW_KAPPAV, M(hilyr, (double)k)));",
     "shortwave/absorbedShortwaveIceLayer", "rad_ccsm3_constant_albedos"),
    # zero-layer ice: the interior absorption not moved to the surface
    ("*fswsfc = A(*fswsfc, *fswint);", "*fswsfc = *fswsfc;",
     "shortwave/surfaceShortwaveFlux", "rad_ccsm3_constant_zero_layer"),
    # the warm snow threshold at 0 instead of -2 puny
    ("if (Tsfcn >= -2.0 * PUNY) {      // -c2*puny, folded", "if (Tsfcn >= 0.0) {",
     "shortwave/albedoVisibleDiffuseCategory", "rad_ccsm3_constant_albedos"),
    # the scaling factor through the reciprocal
    ("scale_factor = D(netsw, scale_factor);", "scale_factor = M(netsw, D(1.0, scale_factor));",
     "shortwave/surfaceShortwaveFlux", "prep_radiation_default"),
    # the topo ponds' lid at the critical thickness counted
    ("ipndn[n] < hp1", "ipndn[n] <= hp1", "shortwave/effectivePondAreaCategory", "rad_coupled_dedd_topo_ponds"),
    # the bare-ice albedo aggregated with the sun below the horizon
    ("if (coszenAll[c] > PUNY) {", "if (true) {", "shortwave/bareIceAlbedoCell",
     "init_shortwave_coupled_topo_ponds"),
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
                    reason="the fixture is the x86-64 glibc 2.44 host's (its exp and atan, PROVENANCE.md); "
                           "it needs gfortran (pixi -e mpas-oracle) and WOOF_MPAS_V842_SOURCE")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_column_fixture.py"),
                    str(tmp_path / "build"), "--cases", "radiation", "--out", str(tmp_path / "fixture")],
                   check=True)
    fresh = np.load(tmp_path / "fixture" / "oracle-r8.npz")
    pinned = np.load(FIXTURE / "oracle-r8.npz")
    assert sorted(fresh.files) == sorted(pinned.files)
    for key in pinned.files:
        assert co.word_mismatches(fresh[key], pinned[key]) == 0, key
    for name in ("column-logs.json", "inputs.sha256"):
        assert (tmp_path / "fixture" / name).read_text() == (FIXTURE / name).read_text()
