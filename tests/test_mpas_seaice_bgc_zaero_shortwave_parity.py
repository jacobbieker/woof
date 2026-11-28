"""MPAS-Seaice's z-aerosols in the delta-Eddington shortwave (the bioabsorption), word for word against v8.4.2.

The reference is MPAS-Model v8.4.2 with ``config_use_column_biogeochemistry``,
``config_use_brine``, ``config_use_vertical_tracers``,
``config_use_zaerosols`` (alone and with ``config_use_nonreactive``) and
``config_use_shortwave_bioabsorption`` (with and without
``config_use_modal_aerosols``), compiled with gfortran -O0
-ffp-contract=off at RKIND 8 on an x86-64 glibc 2.44 host, driven by
``tools/mpas_seaice_oracle/run_column.F90``'s ``radiation`` stage, its
``column_step`` and its coupled step. Its outputs are committed
(tests/data/oracles/mpas_seaice_bgc_zaero_shortwave, PROVENANCE.md), so the
GPU comparison runs without gfortran. The delta-Eddington shortwave needs the
LGPL opt-in ``WOOF_GLIBC239_LIBM64=1`` (set here).

``compute_shortwave_trcr`` remaps each aerosol's ice layers onto the
shortwave grid into ``bioTracerShortwave`` (below its two snow levels), and
``compute_dEdd`` folds the aerosols' optics into the snow's and the ice's
layers: the bulk tables, or the modal black carbon's (the within-ice one's
absorption enhanced by the snow grain size).

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the stages may write must be bit-identical.
2. A shortwave that drops the aerosols' optics, the modal tables or the
   within-ice black carbon's enhancement (the mutations below).
3. A stage that refuses the bioabsorption it now carries, or a port that
   reads past ``verticalGrid`` with fewer ice layers than biology layers.
4. A correction that does not do what it says.
5. A case generator whose bytes drift: the input digests are pinned.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_seaice import column_bgc_v842 as cbgc
from woof.hex.mpas_seaice import column_radiation_v842 as crad
from woof.hex.mpas_seaice import column_step_v842 as cstep
from woof.hex.mpas_seaice import column_therm_v842 as ct
from woof.hex.mpas_seaice import column_v842 as cv
from woof.hex.mpas_seaice import coupling_prep_v842 as cpl
from woof.verify import mpas_seaice_column_oracle as co

FIXTURE = co.BGC_ZAERO_SW_FIXTURE_DIR
ROWS = co.BGC_ZAERO_SW_CASES
BIO = "biogeochemistry/bioTracerShortwave"
CONC = "tracers/verticalAerosolsConc"
BULK = "rad_zaero_bioabsorption_level_ponds"
MODAL = "rad_zaero_nonreactive_modal_cesm_ponds_noon"
NINE = "rad_zaero_modal_topo_ponds_nine_layers"
STEP = "step_zaero_modal_bl99_dedd"
RADIATION = (BULK, MODAL, NINE)
SHORTWAVE = ("shortwave/albedoVisibleDirectCategory", "shortwave/surfaceShortwaveFlux",
             "shortwave/absorbedShortwaveIceLayer", "shortwave/shortwaveLayerPenetration", BIO)


def _row(name):
    return next(row for row in ROWS if row[0] == name)


def _case(name):
    return co.make_case(*_row(name))


@pytest.fixture
def glibc239(monkeypatch):
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (FIXTURE / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", ROWS)}


def test_fixture_writes_the_aerosols_on_the_shortwave_grid():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", ROWS)}
    assert set(logs) == set(cases) == set(fixture)
    for name, case in cases.items():
        d = case["dims"]
        assert logs[name].get("crash") is None and logs[name]["tracer_object"].get("IERR", [0]) == [0], name
        assert BIO in fixture[name], name
        n, nbl, L = d["nCellsSolve"], d["nBioLayers"], d["nIceLayers"] + d["nSnowLayers"] + 2
        bio = fixture[name][BIO][:n]
        # three aerosols' levels, the fourth slot (the Registry's chlorophyll's) zero
        assert bio.shape[2] == 4 * L and np.any(bio[:, :, :3 * L]) and not np.any(bio[:, :, 3 * L:]), name
        if case["stage"] != "radiation":
            continue
        conc, area, fbri = (case["pools"]["tracers"][k] for k in ("verticalAerosolsConc", "iceAreaCategory",
                                                                  "brineFraction"))
        for c in range(n):
            stopped = False
            for k in range(d["nCategories"]):
                # remap_zbgc's stop (cell 2's negative brine height in its first category) sticks
                # for the cell's later categories: colpkg_step_radiation never resets l_stop
                stopped = stopped or (area[c, k, 0] > 1.0e-11 and fbri[c, k, 0] < 0.0)
                if not area[c, k, 0] > 1.0e-11 or stopped:
                    assert not np.any(bio[c, k]), (name, c, k)
                    continue
                for m in range(3):
                    # the snow levels are the aerosol's two snow layers' words
                    assert bio[c, k, m * L] == conc[c, k, m * (nbl + 3) + nbl + 1], (name, c, k, m)
                    assert bio[c, k, m * L + 1] == conc[c, k, m * (nbl + 3) + nbl + 2], (name, c, k, m)
    assert cases[NINE]["dims"]["nIceLayers"] == 9 > cases[NINE]["dims"]["nBioLayers"]


@pytest.mark.parametrize("refuse", [
    lambda c: cstep.check_config(c["config"]),
    lambda c: cpl.check_config(c["config"]),
    lambda c: ct.refuse_unported(c["config"], c["dims"]),
    lambda c: crad.refuse_unported(c["config"], c["dims"]),
    lambda c: cbgc.refuse_unported(c["config"], c["dims"]),
])
def test_the_step_s_stages_take_the_bioabsorption(refuse, glibc239):
    refuse(_case(STEP))


def test_fewer_ice_layers_than_biology_layers_are_refused_unless_corrected(glibc239):
    case = _case(BULK)
    dims = {**case["dims"], "nIceLayers": 4}
    with pytest.raises(ConfigurationRefusal, match="verticalGrid"):
        crad.refuse_unported(case["config"], dims)
    crad.refuse_unported({**case["config"], "woof_v842_corrections": ("shortwave_bio_grid",)}, dims)
    # the umbrella turns it on as well; there the correction without the bioabsorption is not refused
    crad.refuse_unported({**case["config"], "woof_quirk_corrections": "all"}, dims)
    crad.refuse_unported({**case["config"], "config_use_shortwave_bioabsorption": False,
                          "woof_quirk_corrections": "all"}, case["dims"])
    # the correction means nothing without the bioabsorption of the z-aerosols
    plain = {**case["config"], "config_use_shortwave_bioabsorption": False,
             "woof_v842_corrections": ("shortwave_bio_grid",)}
    with pytest.raises(ConfigurationRefusal, match="shortwave_bio_grid"):
        crad.refuse_unported(plain, case["dims"])
    # the bioabsorption without the column biogeochemistry: v8.4.2 stops
    radiation_only = co.make_case(*co.DEDD_CASES[0])
    with pytest.raises(ConfigurationRefusal, match="bioabsorption"):
        crad.refuse_unported({**radiation_only["config"], "config_use_shortwave_bioabsorption": True},
                             radiation_only["dims"])


def test_the_bioabsorption_needs_the_z_aerosols():
    case = _case(MODAL)
    assert crad.shortwave_zaerosols(case["config"])
    assert not crad.shortwave_zaerosols({**case["config"], cv.ZAERO_FLAG: False})
    assert not crad.shortwave_zaerosols({**case["config"], "config_use_shortwave_bioabsorption": False})


@requires_gpu
def test_gpu_matches_the_oracle_word_for_word(glibc239):
    report = co.measure_carry(rows=ROWS, fixture_dir=FIXTURE)
    assert set(report) == {row[0] for row in ROWS}
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


def _mismatches(names, fields=SHORTWAVE):
    report = co.measure_carry(rows=tuple(_row(name) for name in names), fixture_dir=FIXTURE)
    return sum(row.get(f, 0) for row in report.values() for f in fields)


@requires_gpu
@pytest.mark.parametrize("names, original, mutated", [
    # the aerosols' optics never folded into the layers
    ((BULK,), "const bool zaer = za != nullptr && za->nz > 0;", "const bool zaer = false;"),
    # the bulk tables for the modal black carbon
    ((MODAL, NINE), "DeddZaero za = {nz, (flags & DEDD_MODAL) != 0,", "DeddZaero za = {nz, false,"),
    # the within-ice black carbon's absorption not enhanced
    ((MODAL, NINE), "kt = m == 0 ? kx : M(kx, za->bcenh[(k_bcini[k] * 10 + 1) * 3 + ns]);", "kt = kx;"),
    # remap_zbgc's stop not held for the cell's later categories
    (RADIATION, "if (aicenAll[cn + n] > PUNY && !l_stop) {", "if (aicenAll[cn + n] > PUNY) {"),
    # the remap's top concentration from the wrong level
    ((BULK,), "const double top_conc = M(c[0], DEDD_MIN_BGC);", "const double top_conc = M(c[1], DEDD_MIN_BGC);"),
])
def test_a_mutated_shortwave_is_caught(monkeypatch, glibc239, names, original, mutated):
    source = crad.module_source
    assert source().count(original) == 1
    monkeypatch.setattr(crad, "module_source", lambda: source().replace(original, mutated))
    crad._module_on.cache_clear()
    crad._orbit_on.cache_clear()
    try:
        assert _mismatches(names) > 0
    finally:
        crad._module_on.cache_clear()
        crad._orbit_on.cache_clear()


@requires_gpu
def test_the_grid_correction_moves_only_the_shortwave(glibc239):
    plain = co.run_gpu(_case(BULK))[0]
    case = _case(BULK)
    case["config"]["woof_v842_corrections"] = ("shortwave_bio_grid",)
    fixed = co.run_gpu(case)[0]
    moved = {k for k in plain if not np.array_equal(plain[k], fixed[k], equal_nan=True)}
    assert BIO in moved and moved <= {BIO} | {k for k in plain if k.startswith("shortwave/")}
    n = case["dims"]["nCellsSolve"]
    assert np.isfinite(fixed[BIO][:n]).all()
