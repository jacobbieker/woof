"""The MPAS-A v8.4.1 LES closure: CPU authority, configuration, and CUDA parity.

CPU tests run on a pure-Python 162-cell spherical Voronoi mesh scaled to a
~450 m spacing (``_les_mesh``); the GPU tests (auto-marked ``gpu`` by the
conftest because they import cupy) hold the binary32 kernels of
``woof.hex.cuda_les_v841`` to the float32 CPU authority on a synthetic 3-D
patch.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pytest

from _les_mesh import build_tiny_mesh, edge_normal_wind, edge_tangent_wind, flat_vertical
from woof.hex import les_v841 as les
from woof.hex.config_v841 import (
    V841DryDycoreConfig,
    V841MpasColumnPhysicsSmagorinskyGwdoConfig,
)
from woof.hex.errors import ConfigurationRefusal


@pytest.fixture(scope="module")
def mesh():
    return build_tiny_mesh(2)


@pytest.fixture(scope="module")
def geom64(mesh):
    return les.les_geometry_from_mesh(mesh, dtype=np.float64)


@pytest.fixture(scope="module")
def geom32(mesh):
    return les.les_geometry_from_mesh(mesh, dtype=np.float32)


NLEV = 10
DZ = 20.0


def _column_inputs(geom, *, shear=0.01, n2=0.0, dtype=np.float64):
    nc, ne = geom.n_cells, geom.n_edges
    vert = flat_vertical(NLEV, nc, DZ)
    zc = (np.arange(NLEV) + 0.5) * DZ
    return dict(
        u=np.zeros((NLEV, ne), dtype=dtype),
        v=np.zeros((NLEV, ne), dtype=dtype),
        ur_cell=np.repeat((shear * zc)[:, None], nc, axis=1).astype(dtype),
        vr_cell=np.zeros((NLEV, nc), dtype=dtype),
        w=np.zeros((NLEV + 1, nc), dtype=dtype),
        bn2=np.full((NLEV, nc), n2, dtype=dtype),
        zgrid=vert["zgrid"].astype(dtype),
        rho_zz=np.ones((NLEV, nc), dtype=dtype),
    )


def _config(geom, model, **kw):
    return les.LesV841Config(les_model=model, len_disp=geom.nominal_min_dc, **kw)


# ---------------------------------------------------------------------------
# analytic flows
# ---------------------------------------------------------------------------
def test_uniform_shear_gives_the_native_smagorinsky_viscosity(geom64):
    shear = 0.01
    cfg = _config(geom64, les.LES_MODEL_3D_SMAGORINSKY)
    visc = les.les_models_v841(geom64, cfg, dt=1.0, **_column_inputs(geom64, shear=shear))
    c_s = cfg.smagorinsky_coef
    # Native interior du/dz is (u(k+1)-u(k-1))/(z(k+2)+z(k+1)-z(k)-z(k-1)):
    # half the centred gradient on uniform levels; the end levels take a
    # one-sided difference over one layer (les_models lines 335-349).
    expected = np.full(NLEV, shear / 2.0)
    expected[0] = expected[-1] = shear
    kh = (c_s * cfg.len_disp) ** 2 * expected
    kv = (c_s * DZ) ** 2 * expected
    np.testing.assert_allclose(visc.eddy_visc_horz[:, 7], kh, rtol=1e-12)
    np.testing.assert_allclose(visc.eddy_visc_vert[:, 7], kv, rtol=1e-12)
    assert visc.tend_tke is None


def test_the_horizontal_smagorinsky_viscosity_is_capped(geom64):
    cfg = _config(geom64, les.LES_MODEL_3D_SMAGORINSKY)
    visc = les.les_models_v841(geom64, cfg, dt=1000.0, **_column_inputs(geom64, shear=10.0))
    ceiling = 0.01 * cfg.len_disp**2 / 1000.0
    assert np.all(visc.eddy_visc_horz <= ceiling * (1 + 1e-12))
    assert np.any(visc.eddy_visc_horz == pytest.approx(ceiling))


def test_stable_stratification_switches_the_smagorinsky_viscosity_off(geom64):
    cfg = _config(geom64, les.LES_MODEL_3D_SMAGORINSKY)
    inputs = _column_inputs(geom64, shear=0.01, n2=1.0e-3)
    visc = les.les_models_v841(geom64, cfg, dt=1.0, **inputs)
    assert np.all(visc.eddy_visc_horz == 0.0)
    assert np.all(visc.eddy_visc_vert == 0.0)


def test_stable_stratification_suppresses_tke(geom64):
    cfg = _config(geom64, les.LES_MODEL_PROGNOSTIC_15_ORDER)
    tke = np.full((NLEV, geom64.n_cells), 0.2)
    neutral = les.les_models_v841(
        geom64, cfg, dt=1.0, tke=tke, **_column_inputs(geom64, shear=0.02, n2=0.0)
    )
    stable = les.les_models_v841(
        geom64, cfg, dt=1.0, tke=tke, **_column_inputs(geom64, shear=0.02, n2=4.0e-4)
    )
    assert np.all(stable.eddy_visc_vert <= neutral.eddy_visc_vert)
    assert np.all(stable.tend_tke < neutral.tend_tke)
    assert np.all(stable.tend_tke < 0.0)
    # Integrate a column: under the stable profile the TKE decays below the
    # neutral one.
    e_n, e_s = tke.copy(), tke.copy()
    for _ in range(20):
        n = les.les_models_v841(geom64, cfg, dt=1.0, tke=e_n, **_column_inputs(geom64, shear=0.02))
        s = les.les_models_v841(
            geom64, cfg, dt=1.0, tke=e_s, **_column_inputs(geom64, shear=0.02, n2=4.0e-4)
        )
        e_n = np.maximum(0.0, e_n + 1.0 * n.tend_tke)
        e_s = np.maximum(0.0, e_s + 1.0 * s.tend_tke)
    assert e_s.mean() < e_n.mean()
    assert np.all(e_s <= e_n)


def test_tke_viscosity_is_ck_l_sqrt_e(geom64):
    cfg = _config(geom64, les.LES_MODEL_PROGNOSTIC_15_ORDER)
    tke = np.full((NLEV, geom64.n_cells), 0.25)
    visc = les.les_models_v841(geom64, cfg, dt=1.0, tke=tke, **_column_inputs(geom64, n2=0.0))
    # Neutral: l_h = config_len_disp, l_v = min(dz, delta_s) = dz on these levels.
    np.testing.assert_allclose(visc.eddy_visc_horz, les.C_K * cfg.len_disp * 0.5, rtol=1e-12)
    np.testing.assert_allclose(visc.eddy_visc_vert, les.C_K * DZ * 0.5, rtol=1e-12)
    np.testing.assert_allclose(visc.prandtl_3d_inv, 3.0, rtol=1e-12)


def test_negative_tke_is_bounded_at_zero(geom64):
    cfg = _config(geom64, les.LES_MODEL_PROGNOSTIC_15_ORDER)
    tke = np.full((NLEV, geom64.n_cells), -1.0)
    visc = les.les_models_v841(geom64, cfg, dt=1.0, tke=tke, **_column_inputs(geom64))
    assert np.all(visc.tke == 0.0)
    assert np.all(visc.eddy_visc_horz == 0.0)


def test_the_tke_source_is_formed_on_dynamics_substep_one_only(geom64):
    cfg = _config(geom64, les.LES_MODEL_PROGNOSTIC_15_ORDER)
    tke = np.full((NLEV, geom64.n_cells), 0.2)
    visc = les.les_models_v841(
        geom64, cfg, dt=1.0, tke=tke, dynamics_substep=2, **_column_inputs(geom64)
    )
    assert visc.tend_tke is None


def _full_inputs(mesh, geom, *, dtype, seed=1, uniform=False):
    rng = np.random.default_rng(seed)
    nc, ne, nv = geom.n_cells, geom.n_edges, geom.n_vertices
    vert = flat_vertical(NLEV, nc, DZ)
    if uniform:
        u = np.zeros((NLEV, ne))
        w = np.zeros((NLEV + 1, nc))
        theta = np.full((NLEV, nc), 300.0)
    else:
        u = rng.normal(0.0, 1.0, (NLEV, ne))
        w = rng.normal(0.0, 0.3, (NLEV + 1, nc))
        w[0] = w[-1] = 0.0
        theta = 300.0 + rng.normal(0.0, 0.5, (NLEV, nc))
    signs_c = les._cell_edge_sign(geom)
    signs_v = les._vertex_edge_sign(geom)
    div = np.zeros((NLEV, nc))
    vort = np.zeros((NLEV, nv))
    for slot in range(geom.max_edges):
        active = slot < geom.n_edges_on_cell
        edge = np.where(active, geom.edges_on_cell[:, slot], 0)
        div += np.where(active, signs_c[:, slot] * geom.dv_edge[edge] * u[:, edge], 0.0)
    div /= geom.area_cell
    for slot in range(3):
        edge = geom.edges_on_vertex[:, slot]
        vort += signs_v[:, slot] * geom.dc_edge[edge] * u[:, edge]
    vort /= geom.area_triangle
    cast = lambda a: np.asarray(a, dtype=dtype)  # noqa: E731
    return dict(
        u=cast(u),
        v=cast(np.zeros((NLEV, ne)) if uniform else rng.normal(0.0, 1.0, (NLEV, ne))),
        ur_cell=cast(np.zeros((NLEV, nc))),
        vr_cell=cast(np.zeros((NLEV, nc))),
        w=cast(w),
        theta_m=cast(theta),
        rho_edge=cast(np.ones((NLEV, ne))),
        rho_zz=cast(np.ones((NLEV, nc))),
        divergence=cast(div),
        vorticity=cast(vort),
        exner=cast(np.full((NLEV, nc), 0.98)),
        pressure_base=cast(np.full((NLEV, nc), 9.0e4)),
        pressure_p=cast(np.zeros((NLEV, nc))),
        zgrid=cast(vert["zgrid"]),
        zz=cast(vert["zz"]),
        rdzu=cast(vert["rdzu"]),
        rdzw=cast(vert["rdzw"]),
        fzm=cast(vert["fzm"]),
        fzp=cast(vert["fzp"]),
    )


@pytest.mark.parametrize(
    "model", [les.LES_MODEL_3D_SMAGORINSKY, les.LES_MODEL_PROGNOSTIC_15_ORDER]
)
def test_zero_deformation_gives_zero_tendency(mesh, geom64, model):
    cfg = _config(geom64, model)
    inputs = _full_inputs(mesh, geom64, dtype=np.float64, uniform=True)
    tke = np.full((NLEV, geom64.n_cells), 0.2)
    out = les.compute_les_tendencies_v841(
        geom64, cfg, dt=5.0, tke=tke if cfg.prognostic else None, **inputs
    )
    assert np.all(out.tend_u_euler == 0.0)
    assert np.all(out.tend_w_euler == 0.0)
    assert np.all(out.tend_theta_euler == 0.0)
    if cfg.prognostic:
        # only the dissipation term remains, and it removes energy
        assert np.all(out.tend_tke < 0.0)


@pytest.mark.parametrize(
    "model", [les.LES_MODEL_3D_SMAGORINSKY, les.LES_MODEL_PROGNOSTIC_15_ORDER]
)
def test_the_closure_dissipates_kinetic_energy_and_theta_variance(mesh, geom64, model):
    cfg = _config(geom64, model)
    inputs = _full_inputs(mesh, geom64, dtype=np.float64, seed=4)
    # Vertically uniform fields: the vertical fluxes vanish and what remains
    # is the horizontal del2 + del4 filter, which must be dissipative.
    inputs["u"] = np.repeat(inputs["u"][:1], NLEV, axis=0)
    inputs["divergence"] = np.repeat(inputs["divergence"][:1], NLEV, axis=0)
    inputs["vorticity"] = np.repeat(inputs["vorticity"][:1], NLEV, axis=0)
    inputs["theta_m"] = np.repeat(inputs["theta_m"][:1], NLEV, axis=0)
    tke = np.full((NLEV, geom64.n_cells), 0.3)
    out = les.compute_les_tendencies_v841(
        geom64, cfg, dt=5.0, tke=tke if cfg.prognostic else None, **inputs
    )
    edge_area = 0.5 * geom64.dc_edge * geom64.dv_edge
    ke_rate = np.sum(inputs["u"] * out.tend_u_euler * edge_area)
    anomaly = inputs["theta_m"] - np.average(inputs["theta_m"][0], weights=geom64.area_cell)
    var_rate = np.sum(anomaly * out.tend_theta_euler * geom64.area_cell)
    assert ke_rate < 0.0
    assert var_rate < 0.0
    assert np.all(out.eddy_visc_horz >= 0.0) and np.all(out.eddy_visc_vert >= 0.0)


def test_specified_surface_heat_flux_warms_the_lowest_layer(mesh, geom64):
    cfg = _config(
        geom64, les.LES_MODEL_3D_SMAGORINSKY, les_surface=les.LES_SURFACE_SPECIFIED,
        surface_heat_flux=0.1,
    )
    inputs = _full_inputs(mesh, geom64, dtype=np.float64, uniform=True)
    out = les.compute_les_tendencies_v841(geom64, cfg, dt=5.0, **inputs)
    # flux(1) = 0.1 * rho, flux(2) = 0 (no gradient) -> + rdzw * 0.1 * rho
    np.testing.assert_allclose(out.tend_theta_euler[0], 0.1 / DZ, rtol=1e-12)
    assert np.all(out.tend_theta_euler[1:] == 0.0)


def test_specified_surface_drag_decelerates_the_lowest_layer(mesh, geom64):
    cfg = _config(
        geom64, les.LES_MODEL_3D_SMAGORINSKY, les_surface=les.LES_SURFACE_SPECIFIED,
        surface_drag_coefficient=0.002,
    )
    inputs = _full_inputs(mesh, geom64, dtype=np.float64, uniform=True)
    inputs["u"] = np.full_like(inputs["u"], 5.0)
    inputs["divergence"] = np.zeros_like(inputs["divergence"])
    inputs["vorticity"] = np.zeros_like(inputs["vorticity"])
    out = les.compute_les_tendencies_v841(geom64, cfg, dt=5.0, **inputs)
    np.testing.assert_allclose(out.tend_u_euler[0], -0.002 * 25.0 / DZ, rtol=1e-12)


def test_gradient_weights_are_the_unsigned_partners_of_the_deformation_triple(geom64):
    # c2*s and cs*c are both sgn*dl^2*cos^2*sin/area^2.
    np.testing.assert_allclose(
        geom64.coef_c2 * geom64.coef_s, geom64.coef_cs * geom64.coef_c, atol=1e-15
    )
    # A closed polygon: the outward normals integrate to zero.
    active = np.arange(geom64.max_edges)[None, :] < geom64.n_edges_on_cell[:, None]
    scale = np.max(np.abs(geom64.coef_c))
    assert np.max(np.abs(np.sum(np.where(active, geom64.coef_c, 0.0), axis=1))) < 1e-9 * scale
    assert np.max(np.abs(np.sum(np.where(active, geom64.coef_s, 0.0), axis=1))) < 1e-9 * scale


def test_tke_transport_preserves_a_uniform_field(mesh, geom64):
    rng = np.random.default_rng(2)
    e = np.full((NLEV, geom64.n_cells), 0.3)
    out = les.advance_tke_v841(
        geom64,
        tke=e,
        tend_tke=np.zeros_like(e),
        rho_zz=np.ones_like(e),
        rho_u=rng.normal(0, 3, (NLEV, geom64.n_edges)),
        rho_w=rng.normal(0, 1, (NLEV + 1, geom64.n_cells)),
        rdzw=np.full(NLEV, 1.0 / DZ),
        dt=1.0,
    )
    np.testing.assert_allclose(out, e, rtol=1e-14)


def test_tke_transport_is_bounded_by_its_neighbours(mesh, geom64):
    rng = np.random.default_rng(5)
    e = rng.uniform(0.1, 1.0, (NLEV, geom64.n_cells))
    out = les.advance_tke_v841(
        geom64,
        tke=e,
        tend_tke=np.zeros_like(e),
        rho_zz=np.ones_like(e),
        rho_u=rng.normal(0, 1, (NLEV, geom64.n_edges)),
        rho_w=np.zeros((NLEV + 1, geom64.n_cells)),
        rdzw=np.full(NLEV, 1.0 / DZ),
        dt=1.0,
    )
    assert out.min() >= e.min() - 1e-12 and out.max() <= e.max() + 1e-12


# ---------------------------------------------------------------------------
# configuration refusals
# ---------------------------------------------------------------------------
def _dry(**kw):
    base = dict(
        config_horiz_mixing="2d_smagorinsky",
        config_visc4_2dsmag=0.05,
        config_smagorinsky_coef=0.125,
    )
    base.update(kw)
    return V841DryDycoreConfig(**base)


def test_the_dry_lane_admits_both_ported_les_models():
    for model in (les.LES_MODEL_3D_SMAGORINSKY, les.LES_MODEL_PROGNOSTIC_15_ORDER):
        _dry(config_les_model=model).validate()
    _dry(
        config_les_model=les.LES_MODEL_3D_SMAGORINSKY,
        config_les_surface="specified",
        config_surface_heat_flux=0.1,
        config_surface_drag_coefficient=0.001,
    ).validate()


@pytest.mark.parametrize(
    "kw, knob",
    [
        ({"config_les_model": "1.5_order"}, "config_les_model"),
        ({"config_les_surface": "specified"}, "config_les_surface"),
        ({"config_les_model": "3d_smagorinsky", "config_les_surface": "wet"}, "config_les_surface"),
        ({"config_les_model": "3d_smagorinsky", "config_horiz_mixing": "off",
          "config_visc4_2dsmag": 0.0, "config_smagorinsky_coef": 0.0}, "config_horiz_mixing"),
        ({"config_les_model": "3d_smagorinsky", "config_surface_heat_flux": 0.1},
         "config_surface_heat_flux"),
        ({"config_surface_drag_coefficient": 0.1}, "config_surface_drag_coefficient"),
        ({"config_les_model": "3d_smagorinsky", "config_mix_scalars": True}, "config_mix_scalars"),
        ({"config_les_model": "3d_smagorinsky", "config_les_surface": "specified",
          "config_surface_drag_coefficient": -0.1}, "config_surface_drag_coefficient"),
    ],
)
def test_unported_les_selections_refuse_by_name(kw, knob):
    with pytest.raises(ConfigurationRefusal) as caught:
        _dry(**kw).validate()
    assert caught.value.knob == knob


def test_les_with_a_running_pbl_scheme_refuses_naming_pbl_off():
    config = replace(
        V841MpasColumnPhysicsSmagorinskyGwdoConfig(),
        config_les_model=les.LES_MODEL_3D_SMAGORINSKY,
    )
    with pytest.raises(ConfigurationRefusal) as caught:
        config.validate()
    assert caught.value.knob == "config_pbl_scheme"
    assert "--pbl off" in str(caught.value)


def test_a_physics_suite_without_a_pbl_field_refuses_naming_pbl_off():
    probe = argparse.Namespace(
        config_les_model=les.LES_MODEL_3D_SMAGORINSKY,
        config_les_surface="none",
        config_physics_suite="mesoscale_reference",
        config_horiz_mixing="2d_smagorinsky",
    )
    with pytest.raises(ConfigurationRefusal) as caught:
        les.validate_les_selection(probe)
    assert "--pbl off" in str(caught.value)


def test_a_pbl_field_that_is_off_is_admitted():
    probe = argparse.Namespace(
        config_les_model=les.LES_MODEL_PROGNOSTIC_15_ORDER,
        config_les_surface="none",
        config_pbl_scheme="off",
        config_physics_suite="arwen_mpas_column",
        config_horiz_mixing="2d_smagorinsky",
    )
    les.validate_les_selection(probe)


def test_the_default_configurations_are_unchanged_by_the_port():
    V841DryDycoreConfig().validate()
    V841MpasColumnPhysicsSmagorinskyGwdoConfig().validate()
    assert V841MpasColumnPhysicsSmagorinskyGwdoConfig().config_les_model == "none"


def test_the_cpu_dycore_integrator_refuses_an_les_configuration():
    from woof.hex.driver import DryDycoreDriver

    config = _dry(config_les_model=les.LES_MODEL_3D_SMAGORINSKY)
    with pytest.raises(ConfigurationRefusal) as caught:
        DryDycoreDriver(object(), object(), object(), config)
    assert caught.value.knob == "config_les_model"


def test_cli_spellings_map_to_native_values():
    assert les.les_model_from_cli("off") == "none"
    assert les.les_model_from_cli("prognostic_tke") == "prognostic_1.5_order"
    with pytest.raises(ConfigurationRefusal):
        les.les_model_from_cli("tke")
    from woof.hex import forecast_door

    assert forecast_door.LES_MODEL_NATIVE == les.LES_MODEL_CLI
    assert forecast_door.LES_SURFACE_CHOICES == les.LES_SURFACE_CLI_CHOICES


def test_forecast_driver_config_is_untouched_without_les_and_carries_it_with():
    from woof.hex.drivers import run_cuda_v841_forecast as driver

    plain = driver.build_forecast_config(dt_seconds=120.0)
    same = driver.build_forecast_config(dt_seconds=120.0, les=None)
    assert plain == same
    assert plain.config_les_model == "none"
    selected = driver.build_forecast_config(
        dt_seconds=120.0,
        les={"les_model": les.LES_MODEL_3D_SMAGORINSKY, "les_surface": "none"},
    )
    assert selected.config_les_model == les.LES_MODEL_3D_SMAGORINSKY
    # YSU is still the PBL in this tree, so the selection is refused by name.
    with pytest.raises(ConfigurationRefusal) as caught:
        selected.validate()
    assert "--pbl off" in str(caught.value)


def test_driver_les_request_is_none_by_default():
    from woof.hex.drivers import run_cuda_v841_forecast as driver

    args = driver.parse_args(
        ["--grid", "g", "--static", "s", "--init", "i", "--hours", "1",
         "--history-every-minutes", "30", "--preflight-only", "--init-source", "x"]
    )
    assert driver._les_request(args) is None
    assert driver._les_provenance(None) == {}
    base = ["--grid", "g", "--static", "s", "--init", "i", "--hours", "1",
            "--history-every-minutes", "30", "--preflight-only", "--init-source", "x"]
    # This build has no PBL selection, so the driver refuses LES at parse
    # time with the door's own sentence, and refuses ignored LES flags.
    for extra in (
        ["--les-model", "prognostic_tke"],
        ["--les-initial-tke", "0.2"],
        ["--les-heat-flux", "0.1"],
    ):
        with pytest.raises(SystemExit):
            driver.parse_args(base + extra)
    args.les_model = "prognostic_tke"
    args.les_initial_tke = 0.2
    args.pbl = "off"
    assert driver._les_flag_problem(args) is None
    args.les_initial_tke = 0.0
    assert "positive" in driver._les_flag_problem(args)
    args.les_initial_tke = 0.2
    request = driver._les_request(args)
    assert request["pbl_scheme"] == "off"
    assert request["les_model"] == les.LES_MODEL_PROGNOSTIC_15_ORDER
    assert request["initial_tke"] == 0.2
    assert driver._les_provenance(request)["les_label"] == "les_model=prognostic_1.5_order"


# ---------------------------------------------------------------------------
# CUDA parity (auto-marked gpu)
# ---------------------------------------------------------------------------
def _gpu_case(mesh, geom32, model, surface):
    import cupy as cp

    from woof.hex.cuda_les_v841 import les_device_from_host
    from woof.hex.vector import reconstruct_2d

    rng = np.random.default_rng(7)
    nc, ne = geom32.n_cells, geom32.n_edges
    inputs = _full_inputs(mesh, geom32, dtype=np.float32, seed=9)
    zc = (np.arange(NLEV) + 0.5) * DZ
    inputs["u"] = (edge_normal_wind(mesh, 0.02 * zc, 0.01 * zc) + inputs["u"]).astype(np.float32)
    inputs["v"] = (edge_tangent_wind(mesh, 0.02 * zc, 0.01 * zc) + inputs["v"]).astype(np.float32)
    inputs["theta_m"] = (inputs["theta_m"] + 0.004 * zc[:, None]).astype(np.float32)
    rec = reconstruct_2d(mesh, inputs["u"])
    inputs["ur_cell"] = rec.zonal.astype(np.float32)
    inputs["vr_cell"] = rec.meridional.astype(np.float32)
    qv = np.full((NLEV, nc), 0.008, dtype=np.float32)
    qc = np.where(rng.random((NLEV, nc)) > 0.8, 2e-5, 0.0).astype(np.float32)
    extra = {}
    if surface == "specified":
        extra = dict(surface_heat_flux=0.1, surface_moisture_flux=1e-4, surface_drag_coefficient=0.002)
    cfg = _config(geom32, model, les_surface=surface, **extra)
    tke = rng.uniform(0.05, 0.5, (NLEV, nc)).astype(np.float32)
    cpu = les.compute_les_tendencies_v841(
        geom32, cfg, dt=5.0, qv=qv, qc=qc, qtot=qv + qc,
        tke=tke if cfg.prognostic else None, **inputs,
    )
    vert = {name: inputs[name] for name in ("zgrid", "zz", "rdzu", "rdzw", "fzm", "fzp")}
    device = les_device_from_host(
        geom32,
        lat_cell=mesh.arrays["latCell"],
        lon_cell=mesh.arrays["lonCell"],
        coeffs_reconstruct=mesh.arrays["coeffs_reconstruct"],
        vertical=vert,
        pressure_base=inputs["pressure_base"],
        config=cfg,
    )
    if device.tke is not None:
        device.tke = cp.asarray(tke)
    names = ("u", "v", "w", "theta_m", "rho_edge", "rho_zz", "divergence", "vorticity",
             "exner", "pressure_p")
    result = device.compute(
        dt=5.0, qv=cp.asarray(qv), qc=cp.asarray(qc), qtot=cp.asarray(qv + qc),
        **{name: cp.asarray(inputs[name]) for name in names},
    )
    return cp, cfg, cpu, device, result, rec, inputs, tke


def _close(cpu, gpu, cp, rtol=5e-5):
    host = cp.asnumpy(gpu).astype(np.float64)
    ref = np.asarray(cpu, dtype=np.float64)
    scale = max(float(np.max(np.abs(ref))), 1e-30)
    assert np.all(np.isfinite(host))
    assert float(np.max(np.abs(host - ref))) <= rtol * scale


@pytest.mark.parametrize("surface", ["none", "specified"])
@pytest.mark.parametrize(
    "model", [les.LES_MODEL_3D_SMAGORINSKY, les.LES_MODEL_PROGNOSTIC_15_ORDER]
)
def test_cuda_les_matches_the_float32_cpu_authority(mesh, geom32, model, surface):
    pytest.importorskip("cupy")
    cp, cfg, cpu, device, result, rec, inputs, tke = _gpu_case(mesh, geom32, model, surface)
    _close(rec.zonal, device._ur, cp)
    _close(rec.meridional, device._vr, cp)
    _close(cpu.bn2, result.bn2, cp)
    _close(cpu.eddy_visc_horz, result.kdiff, cp)
    _close(cpu.eddy_visc_vert, result.eddy_visc_vert, cp)
    _close(cpu.tend_u_euler, result.tend_u_euler, cp)
    _close(cpu.tend_w_euler, result.tend_w_euler, cp)
    _close(cpu.tend_theta_euler, result.tend_theta_euler, cp)
    if cfg.prognostic:
        _close(cpu.tend_tke, result.tend_tke, cp)
        rho_u = (inputs["rho_edge"] * inputs["u"]).astype(np.float32)
        rho_w = (1.1 * inputs["w"]).astype(np.float32)
        expected = les.advance_tke_v841(
            geom32, tke=tke, tend_tke=cpu.tend_tke, rho_zz=inputs["rho_zz"],
            rho_u=rho_u, rho_w=rho_w, rdzw=inputs["rdzw"], dt=5.0,
        )
        device.advance_tke(
            rho_zz=cp.asarray(inputs["rho_zz"]), rho_u=cp.asarray(rho_u),
            rho_w=cp.asarray(rho_w), dt=5.0,
        )
        _close(expected, device.tke, cp)
        assert device.pending_tend_tke is None


def test_cuda_les_refuses_a_zero_cold_start_for_the_prognostic_closure(mesh, geom32):
    pytest.importorskip("cupy")
    from woof.hex.cuda_les_v841 import les_device_from_host

    cfg = _config(geom32, les.LES_MODEL_PROGNOSTIC_15_ORDER)
    vert = flat_vertical(NLEV, geom32.n_cells, DZ)
    with pytest.raises(ConfigurationRefusal):
        les_device_from_host(
            geom32, lat_cell=mesh.arrays["latCell"], lon_cell=mesh.arrays["lonCell"],
            coeffs_reconstruct=mesh.arrays["coeffs_reconstruct"], vertical=vert,
            pressure_base=np.full((NLEV, geom32.n_cells), 9.0e4), config=cfg,
            initial_tke=0.0,
        )


def test_attach_is_a_no_op_without_an_les_model():
    from woof.hex.cuda_les_v841 import attach_les_v841

    class _Driver:
        config = V841MpasColumnPhysicsSmagorinskyGwdoConfig()

    driver = _Driver()
    assert attach_les_v841(driver, host_mesh=object()) is None
    assert not hasattr(driver, "les_v841")


def test_attachment_drives_the_closure_through_the_subcycle_call(mesh, geom32):
    """The rebinding on a stand-in driver whose subcycle makes the real call.

    Three subcycles per step: the TKE source is formed on the first, the
    TKE is advanced once after the third, and the mixing tendencies the
    driver receives are the LES ones.
    """

    cp = pytest.importorskip("cupy")
    from types import SimpleNamespace

    from woof.hex.cuda_les_v841 import attach_les_v841, detach_les_v841
    from woof.hex.mixing_v841 import initialize_deformation_weights_v841

    a = mesh.arrays
    i32 = lambda x: cp.asarray(np.asarray(x, dtype=np.int32))  # noqa: E731
    f32 = lambda x: cp.asarray(np.asarray(x, dtype=np.float32))  # noqa: E731
    nc, ne = geom32.n_cells, geom32.n_edges
    vert = flat_vertical(NLEV, nc, DZ)
    weights = initialize_deformation_weights_v841(mesh, dtype=np.float32)
    config = V841DryDycoreConfig(
        config_horiz_mixing="2d_smagorinsky", config_visc4_2dsmag=0.05,
        config_smagorinsky_coef=0.125, config_les_model=les.LES_MODEL_PROGNOSTIC_15_ORDER,
        config_dynamics_split_steps=3, config_split_dynamics_transport=True,
    )
    config.validate()
    original = lambda *args, **kwargs: "smagorinsky-2d"  # noqa: E731

    class StandIn:
        def __init__(self):
            self.config = config
            self.v841_context = object()
            self.mixing_config_v841 = object()
            self.cache = None
            self.horizontal = SimpleNamespace(
                mesh=SimpleNamespace(
                    cells_on_edge=i32(a["cellsOnEdge"]), vertices_on_edge=i32(a["verticesOnEdge"]),
                    edges_on_cell=i32(a["edgesOnCell"]), n_edges_on_cell=i32(a["nEdgesOnCell"]),
                    edges_on_vertex=i32(a["edgesOnVertex"]), dc_edge=f32(a["dcEdge"]),
                    dv_edge=f32(a["dvEdge"]), area_cell=f32(a["areaCell"]),
                    area_triangle=f32(a["areaTriangle"]), lat_cell=f32(a["latCell"]),
                    lon_cell=f32(a["lonCell"]), mesh_density=f32(a["meshDensity"]),
                    nominal_min_dc=float(geom32.nominal_min_dc),
                ),
                ncells=nc, nedges=ne, nvertices=geom32.n_vertices,
                max_edges=geom32.max_edges, vertex_degree=3, nlev=NLEV,
                _deformation_v841={
                    k: f32(getattr(weights, k)) for k in ("coef_c2", "coef_s2", "coef_cs")
                },
                compute_dry_mixing_tendencies_v841=original,
            )
            self.atmosphere = SimpleNamespace(
                vertical=SimpleNamespace(
                    **{k: f32(vert[k]) for k in ("zgrid", "zz", "rdzu", "rdzw", "fzm", "fzp")}
                ),
                reference=SimpleNamespace(
                    pressure_base=cp.full((NLEV, nc), 9.0e4, dtype=cp.float32)
                ),
            )

        def _advance_dynamics_subcycle_v841(self, state, *, time_level_one, diag, outer_dt):
            return self.horizontal.compute_dry_mixing_tendencies_v841(
                time_level_one.normal_velocity, diag.tangential_velocity,
                time_level_one.vertical_velocity, time_level_one.theta_m, diag.h_edge,
                diag.divergence, diag.vorticity, dt=outer_dt, config=None,
            )

    driver = StandIn()
    attachment = attach_les_v841(
        driver, host_mesh=mesh, scalar_names=("qv", "qc"), initial_tke=0.2
    )
    assert driver.les_v841 is attachment
    inputs = _full_inputs(mesh, geom32, dtype=np.float32, seed=3)
    zc = (np.arange(NLEV) + 0.5) * DZ
    u = f32(edge_normal_wind(mesh, 0.03 * zc, 0 * zc) + inputs["u"])
    state = SimpleNamespace(
        rho=f32(inputs["rho_zz"]), rho_u=u, rho_w=f32(inputs["w"]),
        scalars=cp.zeros((2, NLEV, nc), dtype=cp.float32),
    )
    saved = SimpleNamespace(
        normal_velocity=u, vertical_velocity=f32(inputs["w"]), theta_m=f32(inputs["theta_m"]),
        exner=f32(inputs["exner"]), pressure_perturbation=f32(inputs["pressure_p"]),
    )
    diag = SimpleNamespace(
        tangential_velocity=f32(inputs["v"]), h_edge=f32(inputs["rho_edge"]),
        divergence=f32(inputs["divergence"]), vorticity=f32(inputs["vorticity"]),
    )
    closure = attachment.closure
    for substep in range(1, 4):
        result = driver._advance_dynamics_subcycle_v841(
            state, time_level_one=saved, diag=diag, outer_dt=5.0
        )
        assert result != "smagorinsky-2d"
        assert np.all(np.isfinite(cp.asnumpy(result.tend_u_euler)))
        assert closure.tke_updates == (1 if substep == 3 else 0)
        assert (closure.pending_tend_tke is None) == (substep == 3)
    assert closure.calls == 3
    summary = attachment.summary()
    assert summary["label"] == "les_model=prognostic_1.5_order"
    assert summary["tke"]["finite"] and summary["tke"]["max"] > 0.0
    detach_les_v841(attachment)
    assert driver.horizontal.compute_dry_mixing_tendencies_v841 is original
    assert "_advance_dynamics_subcycle_v841" not in vars(driver)
    assert driver.les_v841 is None

    # A driver with an outer step method (the real CUDA driver's shape): the
    # substep count restarts per step and the TKE advances after the step.
    class StandInWithStep(StandIn):
        def _step_device_v841(self, fail_after=None):
            for substep in range(1, 4):
                self._advance_dynamics_subcycle_v841(
                    state, time_level_one=saved, diag=diag, outer_dt=5.0
                )
                assert attachment.closure.tke_updates == updates_before
                if fail_after == substep:
                    raise RuntimeError("refused step")
            return "stepped"

    stepped = StandInWithStep()
    attachment = attach_les_v841(
        stepped, host_mesh=mesh, scalar_names=("qv", "qc"), initial_tke=0.2
    )
    updates_before = 0
    with pytest.raises(RuntimeError):
        stepped._step_device_v841(fail_after=2)
    assert attachment.closure.tke_updates == 0
    assert stepped._step_device_v841() == "stepped"
    assert attachment.closure.tke_updates == 1
    updates_before = 1
    assert stepped._step_device_v841() == "stepped"
    assert attachment.closure.tke_updates == 2
    assert attachment.closure.calls == 2 + 3 + 3
    detach_les_v841(attachment)
    assert "_step_device_v841" not in vars(stepped)


def test_every_kernel_is_declared_and_uses_the_grid_stride_loop():
    import re

    from woof.hex.cuda_les_v841 import CUDA_LES_SOURCE, KERNEL_NAMES

    defined = re.findall(r'extern "C" __global__ void (\w+)\(', CUDA_LES_SOURCE)
    assert sorted(defined) == sorted(KERNEL_NAMES)
    for name in KERNEL_NAMES:
        body = CUDA_LES_SOURCE.split(f"void {name}(", 1)[1].split("extern", 1)[0]
        assert "LES_ELEMENT_LOOP(" in body, name
