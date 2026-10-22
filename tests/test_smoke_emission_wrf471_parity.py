"""WRF-SFIRE fire smoke emission: the port against WRF v4.7.1 itself.

The fixture is ``add_fire_tracer_emissions`` (``phys/module_fr_fire_atm.F``,
byte-unmodified, pinned by sha256) compiled at -O0 by
``tools/smoke_wrf471_oracle/build.sh`` and driven by ``run_emission.F90``
once per interior column on one-column tiles (WRF's own loop emits into
one column per tile; ``woof.smoke.emission``).  Four cases: square and
uneven fire-grid refinements (4x4, 3x2, 10x10, 1x1), zero and non-zero
initial tracer, burnt fractions from none to whole cells, fuel loads over
six decades.  Every word of the tracer, the touched and the untouched,
is compared: NumPy reference vs WRF and CuPy kernel vs WRF at zero ULP.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.smoke import emission as em
from woof.verify import smoke_oracle as so

CASES = so.case_names("emission")


def _case(name):
    raw = so.load_case(f"emission/{name}")

    def kji(a):                       # Fortran (i, k, j) -> (k, j, i)
        return np.ascontiguousarray(np.asarray(a, np.float32)
                                    .transpose(1, 2, 0))

    def fire(a):                      # Fortran (i_f, j_f) -> (j_f, i_f)
        return np.ascontiguousarray(np.asarray(a, np.float32).T)

    return dict(
        tracer=kji(raw["tracer_in"]), rho=kji(raw["rho"]),
        dz8w=kji(raw["dz8w"]), burnt=fire(raw["burnt_area_dt"]),
        fgip=fire(raw["fgip"]), fts=np.float32(raw["fire_tracer_smoke"]),
        refinement=(int(raw["ir"]), int(raw["jr"])),
        want=kji(raw["tracer_out"]))


def test_the_cases_exist_and_emit():
    assert CASES == ["case01", "case02", "case03", "case04"]
    for name in CASES:
        c = _case(name)
        changed = c["want"] != c["tracer"]
        # Only level 1 of interior columns moves; something always does.
        assert changed.any() and not changed[1:].any()
        assert not changed[0, 0].any() and not changed[0, -1].any()
        assert not changed[0, :, 0].any() and not changed[0, :, -1].any()


@pytest.mark.parametrize("name", CASES)
def test_reference_is_wrfs_word_for_word(name):
    c = _case(name)
    got = em.fire_tracer_emissions_np(
        c["tracer"], c["rho"], c["dz8w"], c["burnt"], c["fgip"], c["fts"],
        refinement=c["refinement"])
    so.assert_words_equal(got, c["want"], f"{name} tracer (CPU vs WRF)")


def test_a_field_factor_equal_to_the_scalar_is_the_scalar():
    c = _case("case02")
    field = np.full(c["burnt"].shape, c["fts"], np.float32)
    got = em.fire_tracer_emissions_np(
        c["tracer"], c["rho"], c["dz8w"], c["burnt"], c["fgip"], field,
        refinement=c["refinement"])
    so.assert_words_equal(got, c["want"], "field factor")


def test_emission_factors_map_the_anderson_groups():
    assert [em.ANDERSON_FUEL_CLASS[k] for k in range(1, 15)] == (
        ["grass"] * 3 + ["shrub"] * 4 + ["timber"] * 3 + ["slash"] * 3
        + ["none"])
    table = em.emission_factor_table()
    assert np.isnan(table[0]) and table[em.NO_FUEL] == 0.0
    assert table[1] == np.float32(8.0e-3) and table[8] == np.float32(15e-3)
    with pytest.raises(ValueError):
        em.emission_factor(15)
    with pytest.raises(ValueError, match="shrub"):
        em.emission_factor_field_np(
            np.array([1, 4]), {"grass": em.EMISSION_FACTORS["grass"]})
    flux = em.pm25_emission_flux_np(np.float32(2.5e-3), np.array(10))
    assert flux == np.float32(2.5e-3) * np.float32(15e-3)
    with pytest.raises(ValueError):
        em.pm25_emission_flux_np(-1.0, np.array(1))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", CASES)
def test_device_is_wrfs_word_for_word(name):
    import cupy as cp

    c = _case(name)
    got = em.fire_tracer_emissions(
        cp.asarray(c["tracer"]), cp.asarray(c["rho"]), cp.asarray(c["dz8w"]),
        cp.asarray(c["burnt"]), cp.asarray(c["fgip"]), c["fts"],
        refinement=c["refinement"])
    so.assert_words_equal(got.get(), c["want"], f"{name} tracer (GPU vs WRF)")
    ref = em.fire_tracer_emissions_np(
        c["tracer"], c["rho"], c["dz8w"], c["burnt"], c["fgip"], c["fts"],
        refinement=c["refinement"])
    so.assert_words_equal(got.get(), ref, f"{name} tracer (GPU vs CPU)")


@pytest.mark.gpu
@requires_gpu
def test_device_factor_field_and_pm25_flux_match_the_reference():
    import cupy as cp

    c = _case("case03")
    rng = np.random.default_rng(4)
    fuel = rng.integers(1, 15, size=c["burnt"].shape)
    field_np = em.emission_factor_field_np(fuel)
    field = em.emission_factor_field(cp.asarray(fuel))
    so.assert_words_equal(field.get(), field_np, "factor field")
    got = em.fire_tracer_emissions(
        cp.asarray(c["tracer"]), cp.asarray(c["rho"]), cp.asarray(c["dz8w"]),
        cp.asarray(c["burnt"]), cp.asarray(c["fgip"]), field,
        refinement=c["refinement"])
    ref = em.fire_tracer_emissions_np(
        c["tracer"], c["rho"], c["dz8w"], c["burnt"], c["fgip"], field_np,
        refinement=c["refinement"])
    so.assert_words_equal(got.get(), ref, "field factor (GPU vs CPU)")
    rate = (rng.random(fuel.shape) * 1e-2).astype(np.float32)
    so.assert_words_equal(em.pm25_emission_flux(cp.asarray(rate),
                                                cp.asarray(fuel)).get(),
                          em.pm25_emission_flux_np(rate, fuel), "pm25 flux")
