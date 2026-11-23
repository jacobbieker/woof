"""The corrected opt-ins for v8.4.2's ocean quirks, each tested on its own.

The defaults keep v8.4.2's words (or its refusal) and stay gated word for word
against the oracles (test_mpas_ocean_land_ice_parity.py,
test_mpas_ocean_tendency_parity.py, test_mpas_ocean_split_step_parity.py).
These tests cover the corrected paths, each against a numpy reference written
from the physics, evaluated in float64 in the kernels' operation order (one
IEEE rounding per operation, as the kernels' explicit round-to-nearest
macros), so the comparison is word for word:

* ``woof_land_ice_isomip_freezing_salinity = 'boundary_layer_salinity'``: the
  ISOMIP interface temperature is the freezing point at the boundary layer's
  salinity (v8.4.2 passes its temperature);
* ``woof_bottom_drag_dry_edges = 'skip'``: the explicit bottom drag and the
  Rayleigh bottom friction leave a dry edge (maxLevelEdgeTop = 0) alone
  (v8.4.2 reads level 0, outside the array; the default refuses the mesh);
* ``woof_rayleigh_bottom_friction = 'corrected'``: the bottom friction takes
  ``config_Rayleigh_bottom_damping_coeff`` and runs whenever
  ``config_Rayleigh_bottom_friction`` is on (v8.4.2 takes the interior
  coefficient, and only with the interior friction on).

THE BREAKAGES THESE PREVENT
---------------------------
1. A corrected path that is wrong, or that leaks into the default: each is
   checked against its reference, and the default against v8.4.2's words.
2. A corrected path that differs between numerics modes: the GPU tests also
   run in fast, strict and wrf_exact child processes.
3. An unknown opt-in value run silently: it is refused by name.
"""
from __future__ import annotations

import numpy as np
import pytest

from _numerics_child import MODES, run as run_child
from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal

GPU_TESTS = ["test_isomip_freezing_point_at_the_boundary_layer_salinity", "test_dry_edges_skipped_on_request",
             "test_rayleigh_bottom_friction_corrected"]


# ------------------------------------------------------------ ISOMIP freezing point

def _isomip_run(salinity_opt_in):
    from woof.hex.mpas_ocean import eos_v842
    from woof.verify import mpas_ocean_land_ice_oracle as oo
    case = next(c for c in oo.cases("r8") if c["name"] == "standalone_isomip_halo_configured")
    if salinity_opt_in is not None:
        case = dict(case, config=dict(case["config"], woof_land_ice_isomip_freezing_salinity=salinity_opt_in))
    return case, oo.run_gpu(case), eos_v842._configs(case["config"])


def _freezing(salinity, pressure, c):
    p = "config_land_ice_cavity_freezing_temperature_"
    off = np.maximum(pressure - c[p + "reference_pressure"], 0.0)
    return ((c[p + "coeff_0"] + c[p + "coeff_S"] * salinity) + c[p + "coeff_p"] * off) + (c[p + "coeff_pS"] * off) * salinity


@requires_gpu
def test_isomip_freezing_point_at_the_boundary_layer_salinity():
    from woof.hex.mpas_ocean import land_ice_v842 as li
    case, default, eos = _isomip_run(None)
    _, named, _ = _isomip_run("boundary_layer_temperature")
    _, fixed, _ = _isomip_run("boundary_layer_salinity")
    for key in default:  # naming the default changes nothing
        assert default[key].tobytes() == named[key].tobytes(), key
    v, c = case["pools"], case["config"]
    it, isl = v["diagnostics"]["index_landIceBoundaryLayerTemperature"], \
        v["diagnostics"]["index_landIceBoundaryLayerSalinity"]
    iit = v["forcing"]["index_landIceInterfaceTemperature"]
    bl = v["diagnostics"]["landIceBoundaryLayerTracers"]
    f = v["forcing"]
    n = next(int(p["nCellsSolve"]) for p in v.values() if "nCellsSolve" in p)
    on = np.zeros(bl.shape[0], bool)
    on[:n] = f["landIceMask"][:n] != 0
    blt, bls = bl[:, it - 1], bl[:, isl - 1]
    rho_gamma = np.float64(c["config_density0"]) * np.float64(c["config_land_ice_flux_ISOMIP_gammaT"])
    cp, latent = np.float64(li.CP_SW), np.float64(li.LATENT_HEAT_FUSION_MKS)
    for out, salinity in ((default, blt), (fixed, bls)):
        tf = _freezing(salinity, f["landIcePressure"], eos)
        fw = -((rho_gamma * (cp / latent)) * (tf - blt))
        heat = cp * (fw * tf + rho_gamma * (tf - blt))
        got_it = out["forcing/landIceInterfaceTracers"][:, iit - 1]
        assert (got_it[on] == tf[on]).all()
        assert (out["forcing/landIceFreshwaterFlux"][on] == (f["landIceFraction"] * fw)[on]).all()
        assert (out["forcing/landIceHeatFlux"][on] == (f["landIceFraction"] * heat)[on]).all()
        # cells off the mask or beyond nCellsSolve are left as they came
        assert (got_it[~on] == f["landIceInterfaceTracers"][:, iit - 1][~on]).all()
    # physical: at sea-water salinity (30-35 psu) the freezing point lies
    # between -1.6 C (the surface) and -4.5 C (8 MPa), below v8.4.2's reading
    sea = on & (bls > 30.0)
    tf_fixed = fixed["forcing/landIceInterfaceTracers"][sea, iit - 1]
    assert sea.any() and ((tf_fixed < -1.6) & (tf_fixed > -4.5)).all()
    assert (tf_fixed < default["forcing/landIceInterfaceTracers"][sea, iit - 1]).all()
    # only the interface temperature and the fluxes it drives change
    changed = {k for k in default if default[k].tobytes() != fixed[k].tobytes()}
    assert {"forcing/landIceInterfaceTracers", "forcing/landIceFreshwaterFlux", "forcing/landIceHeatFlux"} <= changed
    # (and what they feed: the accumulated mass and the surface fluxes); the
    # mesh and the boundary layer are inputs
    assert not {k for k in changed if k.split("/")[0] in ("mesh", "diagnostics")}


def test_isomip_opt_in_values_are_named():
    from woof.hex.mpas_ocean import land_ice_v842 as li
    # Corrected in correct mode (the default), MPAS's line in exact mode.
    assert li.isomip_salinity_corrected({})
    assert not li.isomip_salinity_corrected({"config_physics_mode": "exact"})
    assert li.isomip_salinity_corrected({"woof_land_ice_isomip_freezing_salinity": "boundary_layer_salinity"})
    with pytest.raises(ConfigurationRefusal):
        li.isomip_salinity_corrected({"woof_land_ice_isomip_freezing_salinity": "salinity"})
    # check_config refuses a typo outside the ISOMIP branch too (Jenkins, and no fluxes at all)
    jenkins = {"config_land_ice_flux_mode": "standalone", "config_land_ice_flux_formulation": "Jenkins",
               "config_land_ice_flux_useHollandJenkinsAdvDiff": False}
    for configs in (jenkins, {"config_land_ice_flux_mode": "off"}):
        li.check_config(configs)
        li.check_config(dict(configs, woof_land_ice_isomip_freezing_salinity="boundary_layer_salinity"))
        with pytest.raises(ConfigurationRefusal, match="woof_land_ice_isomip_freezing_salinity"):
            li.check_config(dict(configs, woof_land_ice_isomip_freezing_salinity="boundary_layer_salnity"))


# ------------------------------------------------------------ bottom drag and Rayleigh friction

N_EDGES, N_CELLS, N_LEVELS = 97, 41, 6


def _mesh_and_fields(dry: bool, seed: int = 20261120):
    rng = np.random.default_rng(seed)
    top = rng.integers(1, N_LEVELS + 1, N_EDGES).astype(np.int32)
    if dry:
        top[rng.random(N_EDGES) < 0.2] = 0
        top[0] = 0  # the first edge: v8.4.2 would read before the array
    cells = rng.integers(1, N_CELLS + 1, (N_EDGES, 2)).astype(np.int32)
    mask = (np.arange(N_LEVELS)[None, :] < top[:, None]).astype(np.int32)
    fields = {
        "normal_velocity": rng.random((N_EDGES, N_LEVELS)) * 2.0 - 1.0,
        "kinetic_energy_cell": rng.random((N_CELLS, N_LEVELS)) * 0.5,
        "layer_thickness_edge": rng.random((N_EDGES, N_LEVELS)) * 40.0 + 5.0,
        "tend": rng.random((N_EDGES, N_LEVELS)) * 2.0e-5 - 1.0e-5,
    }
    return {"maxLevelEdgeTop": top, "cellsOnEdge": cells, "edgeMask": mask}, fields


def _run(mesh, fields, configs):
    import cupy as cp
    from woof.hex.mpas_ocean import forcing_v842 as fo
    dev = {k: cp.asarray(v) for k, v in fields.items()}
    tend = dev.pop("tend")
    fo.vel_forcing_tend(mesh={k: cp.asarray(v) for k, v in mesh.items()},
                        surface_flux_attenuation_coefficient=cp.ones(N_CELLS, dtype=cp.float64),
                        surface_stress=cp.zeros(N_EDGES, dtype=cp.float64), tend=tend,
                        configs={"config_disable_vel_surface_stress": True, **configs}, **dev)
    return cp.asnumpy(tend)


DRAG = {"config_use_explicit_bottom_drag": True, "config_explicit_bottom_drag_coeff": 2.5e-3}


@requires_gpu
def test_dry_edges_skipped_on_request():
    mesh, f = _mesh_and_fields(dry=True)
    for configs in (DRAG, {"config_Rayleigh_friction": True, "config_Rayleigh_bottom_friction": True}):
        with pytest.raises(ConfigurationRefusal):  # the default refuses the mesh
            _run(mesh, f, configs)
    got = _run(mesh, f, {**DRAG, "woof_bottom_drag_dry_edges": "skip"})
    want = f["tend"].copy()
    top = mesh["maxLevelEdgeTop"]
    for e in np.flatnonzero(top):
        k = top[e] - 1
        c1, c2 = mesh["cellsOnEdge"][e] - 1
        drag = np.sqrt(f["kinetic_energy_cell"][c1, k] + f["kinetic_energy_cell"][c2, k])
        want[e, k] = want[e, k] - ((DRAG["config_explicit_bottom_drag_coeff"] * drag) * f["normal_velocity"][e, k]) \
            / f["layer_thickness_edge"][e, k]
    assert got.tobytes() == want.tobytes()
    assert (got[top == 0] == f["tend"][top == 0]).all()
    # drag opposes the flow
    assert ((got - f["tend"]) * f["normal_velocity"] <= 0).all()
    # with no dry edge, 'skip' is the default word for word
    wet_mesh, wet = _mesh_and_fields(dry=False)
    assert _run(wet_mesh, wet, {**DRAG, "woof_bottom_drag_dry_edges": "skip"}).tobytes() == \
        _run(wet_mesh, wet, DRAG).tobytes()


def _rayleigh_reference(mesh, f, interior_coef, bottom_coef):
    want = f["tend"].copy()
    top = mesh["maxLevelEdgeTop"]
    u = f["normal_velocity"]
    for e in range(N_EDGES):
        if interior_coef is not None:
            for k in range(top[e]):
                want[e, k] = want[e, k] - interior_coef * u[e, k]
        if bottom_coef is not None and top[e] > 0:
            k = top[e] - 1
            want[e, k] = want[e, k] - bottom_coef * u[e, k]
    return want


@requires_gpu
def test_rayleigh_bottom_friction_corrected():
    mesh, f = _mesh_and_fields(dry=False)
    coefs = {"config_Rayleigh_damping_coeff": 3.0e-5, "config_Rayleigh_bottom_damping_coeff": 7.0e-4}
    both = {"config_Rayleigh_friction": True, "config_Rayleigh_bottom_friction": True, **coefs}
    only_bottom = {"config_Rayleigh_friction": False, "config_Rayleigh_bottom_friction": True, **coefs}
    fixed = {"woof_rayleigh_bottom_friction": "corrected"}
    # v8.4.2 (the default): the interior coefficient at the bottom, and nothing
    # without the interior friction
    assert _run(mesh, f, both).tobytes() == _rayleigh_reference(mesh, f, 3.0e-5, 3.0e-5).tobytes()
    assert _run(mesh, f, only_bottom).tobytes() == f["tend"].tobytes()
    assert _run(mesh, f, {**both, "woof_rayleigh_bottom_friction": "v842"}).tobytes() == \
        _run(mesh, f, both).tobytes()
    # corrected: the bottom coefficient, whenever the bottom friction is on
    got = _run(mesh, f, {**both, **fixed})
    assert got.tobytes() == _rayleigh_reference(mesh, f, 3.0e-5, 7.0e-4).tobytes()
    got_bottom = _run(mesh, f, {**only_bottom, **fixed})
    assert got_bottom.tobytes() == _rayleigh_reference(mesh, f, None, 7.0e-4).tobytes()
    assert ((got_bottom - f["tend"]) * f["normal_velocity"] <= 0).all()
    # the interior friction alone is not touched by the opt-in
    interior = {"config_Rayleigh_friction": True, **coefs}
    assert _run(mesh, f, {**interior, **fixed}).tobytes() == _run(mesh, f, interior).tobytes()
    # with dry edges skipped, the corrected bottom friction leaves them alone
    dmesh, df = _mesh_and_fields(dry=True)
    got = _run(dmesh, df, {**only_bottom, **fixed, "woof_bottom_drag_dry_edges": "skip"})
    assert got.tobytes() == _rayleigh_reference(dmesh, df, None, 7.0e-4).tobytes()


def test_forcing_opt_in_values_are_named():
    from woof.hex.mpas_ocean import forcing_v842 as fo
    # Corrected in correct mode (the default), v8.4.2's arithmetic in exact mode.
    assert fo._configs(None)["woof_bottom_drag_dry_edges"] == "skip"
    assert fo._configs(None)["woof_rayleigh_bottom_friction"] == "corrected"
    exact = fo._configs({"config_physics_mode": "exact"})
    assert exact["woof_bottom_drag_dry_edges"] == "refuse"
    assert exact["woof_rayleigh_bottom_friction"] == "v842"
    for knob in fo.WOOF_OPT_INS:
        assert knob not in fo.DEFAULT_CONFIGS  # the oracle streams carry MPAS's configs only
        with pytest.raises(ConfigurationRefusal):
            fo._configs({knob: "corrrected"})


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_corrected_paths_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, GPU_TESTS, numerics, exact)
