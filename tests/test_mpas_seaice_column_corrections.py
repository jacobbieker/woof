"""The column port's opt-in corrections of v8.4.2 bugs (``column_v842.CORRECTIONS``).

The default (no ``woof_v842_corrections``) keeps v8.4.2's words, and the
parity tests hold it to the oracle. Each correction here is tested on its own
for physical sanity: what the bug breaks, and that the corrected path does the
physics (an independent numpy reference, a conservation law or a symmetry).
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_seaice import column_radiation_v842 as crad
from woof.hex.mpas_seaice import column_therm_v842 as ct
from woof.hex.mpas_seaice import column_v842 as cv
from woof.verify import mpas_seaice_column_oracle as co

PUNY = 1.0e-11
KEY = "woof_v842_corrections"


def _case(rows, name, **overrides):
    name_, seed, stage, config = next(row for row in rows if row[0] == name)
    case = co.make_case(name_, seed, stage, {**config, **overrides})
    return case


@pytest.fixture
def glibc239(monkeypatch):
    from woof.core import glibc239_gpu
    if not glibc239_gpu.header_path().is_file():
        pytest.skip("the opt-in glibc 2.39 transcription is only in a source checkout")
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")
    for module in (ct, crad):
        module._module_on.cache_clear()
    yield
    for module in (ct, crad):
        module._module_on.cache_clear()


def test_every_correction_is_named_and_unknown_names_are_refused():
    # none named: every one in correct mode, none in exact mode (this
    # module's default: tests/conftest.py)
    assert cv.corrections({"config_physics_mode": "correct"}) == frozenset(cv.CORRECTIONS)
    assert cv.corrections({"config_physics_mode": "exact"}) == frozenset()
    assert cv.corrections({}) == frozenset()
    assert cv.corrections({KEY: ("set_sfcflux_order",)}) == {"set_sfcflux_order"}
    assert cv.corrections({KEY: "dedd_zero_volume_ice"}) == {"dedd_zero_volume_ice"}
    with pytest.raises(ConfigurationRefusal, match="not corrections"):
        cv.corrections({KEY: ("set_sfcflux_order", "no_such_fix")})
    for name, why in cv.CORRECTIONS.items():
        assert "v8.4.2" in why or ".F90" in why or ".F:" in why, name


@pytest.mark.parametrize("value", [None, 3, ("set_sfcflux_order", None), (["set_sfcflux_order"],)])
def test_corrections_that_are_not_names_are_refused(value):
    with pytest.raises(ConfigurationRefusal, match="a name or a sequence of names"):
        cv.corrections({KEY: value})


@pytest.mark.parametrize("name, shortwave, calc_tsfc", [
    ("ccsm3_absorbed_solar_layers", "dEdd", True), ("ccsm3_absorbed_solar_layers", "ccsm3", False),
    ("dedd_zero_volume_ice", "ccsm3", True), ("dedd_zero_volume_ice", "dEdd", False)])
def test_a_shortwave_correction_for_a_scheme_that_does_not_run_is_refused(name, shortwave, calc_tsfc, glibc239):
    case = _case(co.RADIATION_CASES, "rad_ccsm3_constant_albedos")
    config = {**case["config"], "config_shortwave_type": shortwave, "config_calc_surface_temperature": calc_tsfc,
              KEY: (name,)}
    with pytest.raises(ConfigurationRefusal, match="does not run"):
        crad.refuse_unported(config, case["dims"])


@requires_gpu
def test_ice_velocity_at_cell_without_the_vertex_fields_is_refused(glibc239):
    case = _velocity_case(0.5, -0.25, 0.5, -0.25, corrected=True)
    case["pools"]["velocity_solver"]["vVelocity"] = None
    with pytest.raises(ConfigurationRefusal, match="velocity_solver/vVelocity"):
        co.run_gpu(case)


def test_ccsm3_with_more_categories_than_layers_is_refused_unless_corrected():
    case = _case(co.RADIATION_CASES, "rad_ccsm3_constant_albedos")
    dims = {**case["dims"], "nCategories": case["dims"]["nIceLayers"] + 1}
    with pytest.raises(ConfigurationRefusal, match="ccsm3_absorbed_solar_layers"):
        crad.refuse_unported(case["config"], dims)
    crad.refuse_unported({**case["config"], KEY: ("ccsm3_absorbed_solar_layers",)}, dims)


# --- set_sfcflux_order -------------------------------------------------------

@requires_gpu
def test_set_sfcflux_order_reads_each_coupled_flux_as_what_it_is(glibc239):
    """The coupler's surface flux is the surface flux. The case's forcing is
    built for v8.4.2's swap (its surfaceConductiveFluxCouple, read as the
    surface flux, is at least its surfaceHeatFluxCouple, so the surface melts);
    handing the corrected run the same two fluxes under their own names gives
    v8.4.2's run word for word, without an abort, and the surface flux each
    category keeps is the coupled surface flux."""
    name = "vt_bl99_coupled_surface_temperature"
    n_solve = co.MESH["nCellsSolve"]
    v842_case = _case(co.VT_CASES, name)
    v842, v842_state = co.run_gpu(v842_case)
    fixed_case = _case(co.VT_CASES, name, **{KEY: ("set_sfcflux_order",)})
    alt = fixed_case["pools"]["alternative_atmos_forcing"]
    surface = v842_case["pools"]["alternative_atmos_forcing"]["surfaceConductiveFluxCouple"].copy()
    conductive = v842_case["pools"]["alternative_atmos_forcing"]["surfaceHeatFluxCouple"].copy()
    assert not np.array_equal(surface, conductive)
    alt["surfaceHeatFluxCouple"][:] = surface
    alt["surfaceConductiveFluxCouple"][:] = conductive
    fixed, fixed_state = co.run_gpu(fixed_case)
    assert v842_state.abort[0] == 0 and fixed_state.abort[0] == 0
    iced = fixed_case["pools"]["tracers"]["iceAreaCategory"][:n_solve, :, 0] > PUNY
    assert iced.any()
    assert np.array_equal(fixed["atmos_fluxes/surfaceHeatFluxCategory"][:n_solve][iced], surface[:n_solve][iced])
    inputs = {"alternative_atmos_forcing/surfaceHeatFluxCouple", "alternative_atmos_forcing/surfaceConductiveFluxCouple"}
    for key in fixed.keys() - inputs:
        assert np.array_equal(fixed[key], v842[key], equal_nan=True), key


# --- ice_velocity_at_cell ----------------------------------------------------

def _velocity_case(u_ice, v_ice, u_air, v_air, corrected, seed=7):
    case = _case(co.VT_CASES, "vt_bl99_mu71_highfreq", **({KEY: ("ice_velocity_at_cell",)} if corrected else {}))
    pools, dims = case["pools"], case["dims"]
    rs = np.random.RandomState(seed)
    mesh = pools["mesh"]
    n_vertices = dims["nVertices"]
    mesh["verticesOnCell"][:] = rs.randint(1, n_vertices + 1, mesh["verticesOnCell"].shape).astype(np.int32)
    mesh["nEdgesOnCell"][:] = rs.randint(3, mesh["verticesOnCell"].shape[1] + 1,
                                         mesh["nEdgesOnCell"].shape).astype(np.int32)
    pools["boundary"]["interiorVertex"][:] = 1
    mesh["areaTriangle"][:] = rs.uniform(0.1, 1.0, mesh["areaTriangle"].shape)
    vs, ac = pools["velocity_solver"], pools["atmos_coupling"]
    for field, value in (("uVelocity", u_ice), ("vVelocity", v_ice)):
        vs[field][:] = value
    for field in ("airStressCellU", "airStressCellV"):  # the column adds each category's stress to these
        vs[field][:] = 0.0
    for field, value in (("uAirVelocity", u_air), ("vAirVelocity", v_air)):
        ac[field][:] = value
    return case


@requires_gpu
def test_ice_velocity_at_cell_makes_the_stress_depend_on_the_relative_wind_only(glibc239):
    """High-frequency coupling's stress is rhoa*rd^2*|du|*du with du the wind
    relative to the ice. Ice drifting with the wind feels none: the corrected
    run with ice and air at (0.5, -0.25) m/s is the run with both at rest, word
    for word, and adds no stress; v8.4.2 (another vertex's u, no v) adds one."""
    n_solve = co.MESH["nCellsSolve"]
    keys = ("velocity_solver/airStressCellU", "velocity_solver/airStressCellV")
    case = _velocity_case(0.5, -0.25, 0.5, -0.25, corrected=True)
    iced = case["pools"]["tracers"]["iceAreaCategory"][:n_solve, :, 0].max(axis=1) > PUNY
    assert iced.any()
    drifting, _ = co.run_gpu(case)
    rest, _ = co.run_gpu(_velocity_case(0.0, 0.0, 0.0, 0.0, corrected=True))
    v842, _ = co.run_gpu(_velocity_case(0.5, -0.25, 0.5, -0.25, corrected=False))
    inputs = {"velocity_solver/uVelocity", "velocity_solver/vVelocity", "atmos_coupling/uAirVelocity",
              "atmos_coupling/vAirVelocity"}
    for key in drifting.keys() - inputs:
        assert np.array_equal(drifting[key], rest[key]), key
    assert (drifting[keys[0]][:n_solve][iced] == 0).all() and (drifting[keys[1]][:n_solve][iced] == 0).all()
    assert (v842[keys[1]][:n_solve][iced] != 0).any()


@requires_gpu
def test_ice_velocity_at_cell_is_the_area_weighted_vertex_mean(glibc239):
    """With varying vertex velocities and a uniform wind, each cell's stress
    points along the wind relative to the cell's ice velocity, the
    areaTriangle-weighted mean of its interior vertices (numpy reference)."""
    n_solve = co.MESH["nCellsSolve"]
    rs = np.random.RandomState(11)
    n_vertices = co.MESH["nVertices"]
    u_ice = np.append(rs.uniform(-0.3, 0.3, n_vertices), 0.0)
    v_ice = np.append(rs.uniform(-0.3, 0.3, n_vertices), 0.0)
    case = _velocity_case(u_ice, v_ice, 4.0, -3.0, corrected=True)
    mesh = case["pools"]["mesh"]
    area = mesh["areaTriangle"]
    out, _ = co.run_gpu(case)
    iced = case["pools"]["tracers"]["iceAreaCategory"][:n_solve, :, 0].max(axis=1) > PUNY
    assert iced.any()
    for c in np.flatnonzero(iced):
        vertices = mesh["verticesOnCell"][c, :mesh["nEdgesOnCell"][c]] - 1
        weights = area[vertices]
        u_cell = np.sum(weights * u_ice[vertices]) / np.sum(weights)
        v_cell = np.sum(weights * v_ice[vertices]) / np.sum(weights)
        strx, stry = out["velocity_solver/airStressCellU"][c], out["velocity_solver/airStressCellV"][c]
        assert np.hypot(strx, stry) > 0
        np.testing.assert_allclose(np.arctan2(stry, strx), np.arctan2(-3.0 - v_cell, 4.0 - u_cell), rtol=0,
                                   atol=1e-12)


# --- ccsm3_absorbed_solar_layers -----------------------------------------------

@requires_gpu
@pytest.mark.parametrize("ncat, nilyr", [(5, 7), (4, 2)])
def test_ccsm3_absorbed_solar_layers_follows_beer_s_law_through_every_layer(ncat, nilyr):
    """absorbed_solar with the category's nilyr layers: layer k of n absorbs
    fswpen*(exp(-kappav*hi*(k-1)/n) - exp(-kappav*hi*k/n)), every layer
    absorbs, and the layers' sum is the interior flux (numpy reference).
    v8.4.2 spreads it over ncat layers instead."""
    n_solve = co.MESH["nCellsSolve"]
    overrides = {"config_nCategories": ncat, "config_nIceLayers": nilyr}
    case = _case(co.RADIATION_CASES, "rad_ccsm3_constant_albedos", **overrides,
                 **{KEY: ("ccsm3_absorbed_solar_layers",)})
    out, _ = co.run_gpu(case)
    t = case["pools"]["tracers"]
    aice, vice = t["iceAreaCategory"][:n_solve, :, 0], t["iceVolumeCategory"][:n_solve, :, 0]
    iswabs = out["shortwave/absorbedShortwaveIceLayer"][:n_solve]
    fswpenl = out["shortwave/shortwaveLayerPenetration"][:n_solve]
    fswint = out["shortwave/interiorShortwaveFlux"][:n_solve]
    fswthru = out["shortwave/penetratingShortwaveFlux"][:n_solve]
    lit = (aice > PUNY) & (vice > 0) & (fswpenl[:, :, 0] > 0)
    assert lit.any()
    hi = vice[lit] / aice[lit]
    fswpen = fswpenl[:, :, 0][lit]
    depth = np.arange(nilyr + 1) / nilyr
    transmitted = np.exp(-1.4 * hi[:, None] * depth[None, :])
    np.testing.assert_allclose(iswabs[lit], fswpen[:, None] * -np.diff(transmitted, axis=1), rtol=1e-10, atol=0)
    assert (iswabs[lit] > 0).all()
    np.testing.assert_allclose(iswabs[lit].sum(axis=1), fswint[lit], rtol=1e-12)
    np.testing.assert_allclose(fswint[lit] + fswthru[lit], fswpen, rtol=1e-14)
    if ncat < nilyr:
        v842, _ = co.run_gpu(_case(co.RADIATION_CASES, "rad_ccsm3_constant_albedos", **overrides))
        assert (v842["shortwave/absorbedShortwaveIceLayer"][:n_solve][lit][:, ncat:] == 0).all()


# --- dedd_zero_volume_ice --------------------------------------------------------

LIT = 1  # a cell of dedd_default_level_ponds with the sun up (coszen 0.36) and shortwave down


def _zero_volume_case(corrected, kalg):
    overrides = {KEY: ("dedd_zero_volume_ice",)} if corrected else {}
    if kalg is not None:
        overrides["config_algae_absorption_coefficient"] = kalg
    case = _case(co.DEDD_CASES, "dedd_default_level_ponds", **overrides)
    t = case["pools"]["tracers"]
    t["iceAreaCategory"][LIT, -1, 0] = 0.3
    t["iceVolumeCategory"][LIT, -1, 0] = 0.0
    t["snowVolumeCategory"][LIT, -1, 0] = 0.0
    return case


@requires_gpu
@pytest.mark.parametrize("kalg", [None, 0.0], ids=["default_kalg", "no_algae"])
def test_dedd_zero_volume_ice_is_transparent_finite_and_conserves_energy(glibc239, kalg):
    """Area without volume: v8.4.2's bottom layer, of no thickness, has an
    infinite chlorophyll extinction, so with algae the visible light reaching
    it is absorbed in ice that is not there, and without algae (kalg = 0, so
    0*Inf) the category's shortwave is NaN. Corrected, the layers without
    thickness absorb nothing, every output is finite, albedos lie in [0, 1],
    and the category's absorbed shortwave (surface + interior + transmitted)
    is the incoming minus the reflected, as delta-Eddington conserves energy."""
    n_solve = co.MESH["nCellsSolve"]
    v842, _ = co.run_gpu(_zero_volume_case(False, kalg))
    assert v842["shortwave/solarZenithAngleCosine"][LIT] > 0.3
    bottom = v842["shortwave/absorbedShortwaveIceLayer"][LIT, -1, -1]
    if kalg is None:
        assert bottom > 1.0
    else:
        assert np.isnan(v842["shortwave/albedoVisibleDirectCategory"][LIT, -1])
    case = _zero_volume_case(True, kalg)
    out, _ = co.run_gpu(case)
    for key, value in out.items():
        if key.startswith("shortwave/") and value.ndim >= 2:
            assert np.isfinite(value[:n_solve]).all(), key
    assert np.allclose(out["shortwave/absorbedShortwaveIceLayer"][LIT, -1], 0.0, rtol=0, atol=1e-12)
    assert abs(out["shortwave/interiorShortwaveFlux"][LIT, -1]) < 1e-12
    ac = case["pools"]["atmos_coupling"]
    down = [ac[k][:n_solve, None] for k in ("shortwaveVisibleDirectDown", "shortwaveVisibleDiffuseDown",
                                            "shortwaveIRDirectDown", "shortwaveIRDiffuseDown")]
    albedo = [out[f"shortwave/{k}"][:n_solve] for k in ("albedoVisibleDirectCategory", "albedoVisibleDiffuseCategory",
                                                        "albedoIRDirectCategory", "albedoIRDiffuseCategory")]
    aice = case["pools"]["tracers"]["iceAreaCategory"][:n_solve, :, 0]
    coszen = out["shortwave/solarZenithAngleCosine"][:n_solve, None]
    lit = (aice > PUNY) & (coszen > PUNY) & (sum(down) > PUNY)
    assert lit[LIT, -1]
    for a in albedo:
        assert ((a[lit] >= 0) & (a[lit] <= 1)).all()
    absorbed = sum(d * (1.0 - a) for d, a in zip(down, albedo))
    budget = sum(out[f"shortwave/{k}"][:n_solve] for k in ("surfaceShortwaveFlux", "interiorShortwaveFlux",
                                                           "penetratingShortwaveFlux"))
    np.testing.assert_allclose(budget[lit], absorbed[lit], rtol=1e-9)
