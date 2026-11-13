"""Cases and the stream layout for the MPAS-Ocean split-explicit step oracle.

``tools/mpas_ocean_oracle/run_split_step.F90`` drives one step of MPAS-Model
v8.4.2's ``ocn_time_integrator_split`` (cut verbatim out of
``mpas_ocn_time_integration_split.F``) with every module it reaches compiled
verbatim (the tendencies, diagnostics, equation of state, ALE thickness,
tracer advection and mixing, vertical mixing, time averaging), with gfortran
-O0 -ffp-contract=off at RKIND 4 and 8. The stream is
:mod:`woof.verify.mpas_ocean_stream`'s; a field's time level 2 is the entry
``NAME@2``.

The field table is the union of the pieces' tables (diagnostics, tracer
tendency, vertical mixing, vertical transport, barotropic stage) and what the
momentum tendency, the vector reconstruction and the time averaging add.
The meshes are random connectivity on one block with every halo extent the
block's count, about one edge in twelve on a boundary (a missing cell);
``maxLevelCell`` is at least 2 (monotone advection needs two levels).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_btr_subcycle_oracle as _btr
from woof.verify import mpas_ocean_diagnostics_oracle as _diag
from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify import mpas_ocean_tend_tracer_oracle as _tt
from woof.verify import mpas_ocean_vert_transport_oracle as _vt
from woof.verify import mpas_ocean_vmix_oracle as _vm
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_split_step")
CASES_PER_PRECISION = True
REAL, INT, DIM, DIMS, SCRATCH = _stream.REAL, _stream.INT, _stream.DIM, _stream.DIMS, _stream.SCRATCH
#: state fields with two time levels (level 2 is NAME@2)
TWO_LEVEL = {"state": ("layerThickness", "normalVelocity", "ssh", "normalBarotropicVelocity",
                       "normalBaroclinicVelocity", "sshSubcycle", "normalBarotropicVelocitySubcycle"),
             "tracers": ("activeTracers",)}
#: packages that are off: the fields are absent at both levels
ABSENT_FIELDS = (("state", "highFreqThickness", ("L", "C1")), ("state", "lowFreqDivergence", ("L", "C1")),
                 ("tend", "highFreqThickness", ("L", "C1")), ("tend", "lowFreqDivergence", ("L", "C1")))
_EXTRA = (
    ("tracers", "activeGRP_start", DIM, ()), ("tracers", "activeGRP_end", DIM, ()),
    ("mesh", "nEdgesSolve", DIM, ()), ("mesh", "nVerticesSolve", DIM, ()),
    ("mesh", "fEdge", REAL, ("E1",)), ("mesh", "meshScaling", REAL, ("E1",)), ("mesh", "angleEdge", REAL, ("E1",)),
    ("mesh", "maxLevelVertexTop", INT, ("V1",)), ("mesh", "lonCell", REAL, ("C1",)),
    ("mesh", "coeffs_reconstruct", REAL, (3, "ME", "C1")),
    ("diagnostics", "viscosity", REAL, ("L", "E1")),
    ("diagnostics", "velocityX", REAL, ("L", "C1")), ("diagnostics", "velocityY", REAL, ("L", "C1")),
    ("diagnostics", "velocityZ", REAL, ("L", "C1")), ("diagnostics", "velocityZonal", REAL, ("L", "C1")),
    ("diagnostics", "velocityMeridional", REAL, ("L", "C1")),
    ("diagnostics", "gradSSHX", REAL, ("C1",)), ("diagnostics", "gradSSHY", REAL, ("C1",)),
    ("diagnostics", "gradSSHZ", REAL, ("C1",)), ("diagnostics", "gradSSHZonal", REAL, ("C1",)),
    ("diagnostics", "gradSSHMeridional", REAL, ("C1",)),
    ("diagnostics", "index_surfaceVelocityZonal", DIM, ()), ("diagnostics", "index_surfaceVelocityMeridional", DIM, ()),
    ("diagnostics", "index_SSHGradientZonal", DIM, ()), ("diagnostics", "index_SSHGradientMeridional", DIM, ()),
    ("diagnostics", "surfaceVelocity", REAL, (2, "C1")), ("diagnostics", "SSHGradient", REAL, (2, "C1")),
    ("forcing", "nCells", DIM, ()), ("forcing", "nAccumulatedCoupled", DIM, ()),
    ("forcing", "index_avgTemperatureSurfaceValue", DIM, ()), ("forcing", "index_avgSSHGradientZonal", DIM, ()),
    ("forcing", "index_avgSSHGradientMeridional", DIM, ()),
    ("forcing", "avgTracersSurfaceValue", REAL, ("T", "C1")), ("forcing", "avgSurfaceVelocity", REAL, (2, "C1")),
    ("forcing", "avgSSHGradient", REAL, (2, "C1")),
    ("forcing", "surfaceStress", REAL, ("E1",)), ("forcing", "surfaceStressMagnitude", REAL, ("C1",)),
    ("forcing", "windStressZonal", REAL, ("C1",)), ("forcing", "windStressMeridional", REAL, ("C1",)),
    ("forcing", "surfaceThicknessFlux", REAL, ("C1",)), ("forcing", "surfaceThicknessFluxRunoff", REAL, ("C1",)),
    ("tend", "normalVelocity", REAL, ("L", "E1")), ("tend", "layerThickness", REAL, ("L", "C1")),
)


def _field_table():
    rows, seen = [], set()
    for table in (_diag.FIELDS, _tt.FIELDS, _vm.FIELDS, _vt.FIELDS, _btr.FIELDS, _EXTRA):
        for pool, name, kind, fshape in table:
            base = name.split("@")[0]
            if (pool, base) in seen:
                continue
            seen.add((pool, base))
            rows.append((pool, base, kind, fshape))
            if base in TWO_LEVEL.get(pool, ()):
                rows.append((pool, f"{base}@2", kind, fshape))
    for pool, name, fshape in ABSENT_FIELDS:
        rows += [(pool, name, REAL, fshape)]
        if pool == "state":
            rows += [(pool, f"{name}@2", REAL, fshape)]
    return tuple(rows)


FIELDS = _field_table()
ABSENT = {(pool, name) for pool, name, _ in ABSENT_FIELDS} | {(p, f"{n}@2") for p, n, _ in ABSENT_FIELDS
                                                             if p == "state"}

#: Registry.xml (v8.4.2) defaults of every other config the step's modules read
REGISTRY_DEFAULTS = {
    'config_Redi_bottom_layer_tapering_depth': 0.0, 'config_Redi_surface_layer_tapering_extent': 0.0,
    'config_btr_dt': '0000_00:00:15', 'config_check_ssh_consistency': True, 'config_check_zlevel_consistency':
    False, 'config_cvmix_background_diffusion': 1e-05, 'config_cvmix_background_viscosity': 0.0001,
    'config_cvmix_convective_basedOnBVF': True, 'config_cvmix_convective_diffusion': 1.0,
    'config_cvmix_convective_triggerBVF': 0.0, 'config_cvmix_convective_viscosity': 1.0,
    'config_cvmix_kpp_EkmanOBL': False, 'config_cvmix_kpp_MonObOBL': False,
    'config_cvmix_kpp_boundary_layer_depth': 30.0, 'config_cvmix_kpp_criticalBulkRichardsonNumber': 0.25,
    'config_cvmix_kpp_interpolationOMLType': 'quadratic', 'config_cvmix_kpp_matching': 'SimpleShapes',
    'config_cvmix_kpp_stop_OBL_search': 100.0, 'config_cvmix_kpp_surface_layer_extent': 0.1,
    'config_cvmix_kpp_use_enhanced_diff': True, 'config_cvmix_num_ri_smooth_loops': 2,
    'config_cvmix_prandtl_number': 1.0, 'config_cvmix_shear_KPP_Ri_zero': 0.7, 'config_cvmix_shear_KPP_exp':
    3.0, 'config_cvmix_shear_KPP_nu_zero': 0.005, 'config_cvmix_shear_PP_alpha': 5.0,
    'config_cvmix_shear_PP_exp': 2.0, 'config_cvmix_shear_PP_nu_zero': 0.005,
    'config_cvmix_shear_mixing_scheme': 'PP', 'config_cvmix_use_BLD_smoothing': True,
    'config_disable_redi_horizontal_term1': False, 'config_disable_redi_horizontal_term2': False,
    'config_disable_redi_horizontal_term3': False, 'config_disable_redi_k33': False,
    'config_disable_tr_nonlocalflux': False, 'config_do_restart': False, 'config_dt': '00:05:00',
    'config_frazil_fractional_thickness_limit': 0.1, 'config_frazil_heat_of_fusion': 334000.0,
    'config_frazil_ice_density': 1000.0, 'config_frazil_in_open_ocean': True,
    'config_frazil_land_ice_reference_salinity': 0.0, 'config_frazil_maximum_depth': 100.0,
    'config_frazil_maximum_freezing_temperature': 0.0, 'config_frazil_sea_ice_reference_salinity': 4.0,
    'config_frazil_under_land_ice': True, 'config_frazil_use_surface_pressure': True,
    'config_gravWaveSpeed_trunc': 0.3, 'config_highFreqThick_del2': 100.0, 'config_highFreqThick_restore_time':
    30.0, 'config_hmix_scaleWithMesh': False, 'config_land_ice_cavity_freezing_temperature_coeff_0': 0.0622,
    'config_land_ice_cavity_freezing_temperature_coeff_S': -0.0563,
    'config_land_ice_cavity_freezing_temperature_coeff_p': -7.43e-08,
    'config_land_ice_cavity_freezing_temperature_coeff_pS': -1.74e-10,
    'config_land_ice_cavity_freezing_temperature_reference_pressure': 0.0, 'config_land_ice_flux_ISOMIP_gammaT':
    0.0001, 'config_land_ice_flux_attenuation_coefficient': 10.0,
    'config_land_ice_flux_boundaryLayerNeighborWeight': 0.0, 'config_land_ice_flux_boundaryLayerThickness':
    10.0, 'config_land_ice_flux_cp_ice': 2009.0, 'config_land_ice_flux_formulation': 'Jenkins',
    'config_land_ice_flux_jenkins_heat_transfer_coefficient': 0.011,
    'config_land_ice_flux_jenkins_salt_transfer_coefficient': 0.00031, 'config_land_ice_flux_rho_ice': 918.0,
    'config_land_ice_flux_rms_tidal_velocity': 0.05, 'config_land_ice_flux_topDragCoeff': 0.0025,
    'config_land_ice_flux_useHollandJenkinsAdvDiff': False, 'config_maxMeshDensity': -1.0,
    'config_max_diff_tanh': 0.025, 'config_max_relative_slope': 0.01, 'config_max_visc_tanh': 0.25,
    'config_min_diff_tanh': 1e-05, 'config_min_visc_tanh': 0.0001, 'config_mom_del4_tensor': 50000000000000.0,
    'config_specific_heat_sea_water': 3985.0, 'config_standardGM_tracer_kappa': 0.0,
    'config_tensor_test_function': 'sph_uCosCos', 'config_test_tensors': False,
    'config_thickness_filter_timescale': 5.0, 'config_use_Redi_bottom_layer_tapering': False,
    'config_use_Redi_surface_layer_tapering': False, 'config_use_cvmix_background': False,
    'config_use_cvmix_convection': False, 'config_use_cvmix_double_diffusion': False,
    'config_use_cvmix_fixed_boundary_layer': False, 'config_use_cvmix_shear': False,
    'config_use_cvmix_tidal_mixing': False, 'config_use_highFreqThick_del2': False,
    'config_use_highFreqThick_restore': False, 'config_zMid_tanh': -100.0, 'config_zWidth_tanh': 100.0
}
BASE_CONFIG = {
    **{k: v for k, v in _diag.BASE_CONFIG.items()},
    **{k: v for k, v in _tt.BASE_CONFIG.items() if k != "oracle_active_tracers_only"},
    **_vm.BASE_CONFIG, **{k: v for k, v in _vt.BASE_CONFIG.items()},
    **{k: v for k, v in _btr.BASE_CONFIG.items()},
    # the momentum tendency's terms (forcing_v842.DEFAULT_CONFIGS)
    "config_vert_coord_movement": "uniform_stretching", "config_disable_vel_coriolis": False,
    "config_disable_vel_vadv": False, "config_disable_thick_hadv": False, "config_disable_thick_vadv": False,
    "config_common_level_weight": 0.5, "config_disable_vel_pgrad": False, "config_disable_vel_hmix": False,
    "config_use_mom_del2": True, "config_mom_del2": 10.0, "config_use_Leith_del2": False,
    "config_Leith_parameter": 1.0, "config_Leith_dx": 15000.0, "config_Leith_visc2_max": 2.5e3,
    "config_use_mom_del4": False, "config_mom_del4": 5.0e13, "config_mom_del4_div_factor": 1.0,
    "config_use_mom_del2_tensor": False, "config_use_mom_del4_tensor": False,
    "config_disable_vel_surface_stress": False, "config_use_explicit_bottom_drag": False,
    "config_explicit_bottom_drag_coeff": 1.0e-3, "config_disable_vel_explicit_bottom_drag": False,
    "config_Rayleigh_friction": False, "config_Rayleigh_damping_coeff": 1.0e-4,
    "config_Rayleigh_bottom_friction": False, "config_Rayleigh_bottom_damping_coeff": 1.0e-4,
    "config_disable_thick_sflux": False,
    # the step's own
    "config_time_integrator": "split_explicit", "config_n_ts_iter": 2, "config_n_bcl_iter_beg": 1,
    "config_n_bcl_iter_mid": 2, "config_n_bcl_iter_end": 2, "config_prescribe_velocity": False,
    "config_prescribe_thickness": False, "config_use_ecosysTracers": False, "config_use_DMSTracers": False,
    "config_use_MacroMoleculesTracers": False, "mesh:on_a_sphere": False,
    "config_use_cvmix_kpp": False, "config_include_KE_vertex": False, "config_land_ice_flux_mode": "off",
    "config_use_frazil_ice_formation": False, "nBtrSubcycles": 2, "config_btr_subcycle_loop_factor": 2,
    **REGISTRY_DEFAULTS,
}
#: (name, seed, nCells, nVertLevels, config overrides)
CASES = (
    ("split_defaults", 20261401, 40, 8, {}),
    ("split_jm_rich_bulk_budgets", 20261402, 44, 10, {
        "config_eos_type": "jm", "config_pressure_gradient_type": "Jacobian_from_TS",
        "config_use_rich_visc": True, "config_use_rich_diff": True,
        "config_use_activeTracers_surface_bulk_forcing": True, "config_use_bulk_thickness_flux": True,
        "config_sw_absorption_type": "jerlov", "config_compute_active_tracer_budgets": True,
        "config_use_tracer_del2": True, "config_tracer_del2": 50.0, "config_n_ts_iter": 3,
        "config_btr_solve_SSH2": True, "config_n_btr_cor_iter": 0, "nBtrSubcycles": 1, "_fresh": True}),
    ("unsplit_const_vmix_prescribed", 20261403, 32, 6, {
        "config_time_integrator": "unsplit_explicit", "config_use_const_visc": True, "config_use_const_diff": True,
        "config_vel_correction": False, "config_prescribe_thickness": True, "config_n_bcl_iter_mid": 1}),
    ("split_sphere", 20261404, 36, 8, {"mesh:on_a_sphere": True, "config_apvm_scale_factor": 0.5}),
)


def make_case(name: str, seed: int, n_cells: int, n_levels: int, overrides: dict, precision: str = "r4") -> dict:
    """One deterministic case: the diagnostics oracle's random mesh (missing
    neighbours, vertices and kites), the tracer advection stencils, an ocean's
    magnitudes, no flow through land edges; ``maxLevelCell`` at least 2.
    ``_fresh`` makes a quarter of the columns nearly fresh."""
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    nc, nl = n_cells, n_levels
    ne, nv, me, me2, vd, nt, na = 3 * nc, 2 * nc, 7, 14, 3, 2, 14
    config = dict(BASE_CONFIG, **{k: v for k, v in overrides.items() if not k.startswith("_")})

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    def ints(lo, hi, shp):
        return rs.randint(lo, hi, shp).astype(np.int32)

    mlc = np.zeros(nc + 1, np.int32)
    mlc[:nc] = ints(2, nl + 1, nc)
    mlc[: min(nc, 3)] = nl
    cells_on_edge = ints(1, nc + 1, (ne + 1, 2))
    cells_on_edge[rs.random_sample(ne + 1) < 1.0 / 12.0, 1] = nc + 1
    cells_on_edge[ne] = nc + 1
    mlet = np.minimum(mlc[cells_on_edge[:, 0] - 1], mlc[cells_on_edge[:, 1] - 1]).astype(np.int32)
    mleb = np.maximum(mlc[cells_on_edge[:, 0] - 1], mlc[cells_on_edge[:, 1] - 1]).astype(np.int32)
    mlet[ne] = mleb[ne] = 0
    cells_on_vertex = ints(1, nc + 1, (nv + 1, vd))
    cells_on_vertex[rs.random_sample((nv + 1, vd)) < 1.0 / 15.0] = nc + 1
    vertices_on_edge = ints(1, nv + 1, (ne + 1, 2))
    n_edges_on_cell = np.array([5, 6, 6, 6, 7], np.int32)[rs.randint(0, 5, nc + 1)]
    n_edges_on_cell[nc] = 0
    edges_on_cell = np.full((nc + 1, me), ne + 1, np.int32)
    cells_on_cell = np.full((nc + 1, me), nc + 1, np.int32)
    vertices_on_cell = np.full((nc + 1, me), nv + 1, np.int32)
    kite_index = np.ones((nc + 1, me), np.int32)
    edge_sign_on_cell = np.zeros((nc + 1, me), np.int32)
    for c in range(nc):
        k = n_edges_on_cell[c]
        edges_on_cell[c, :k] = ints(1, ne + 1, k)
        cells_on_cell[c, :k] = ints(1, nc + 1, k)
        vertices_on_cell[c, :k] = ints(1, nv + 1, k)
        kite_index[c, :k] = ints(1, vd + 1, k)
        edge_sign_on_cell[c, :k] = 2 * ints(0, 2, k) - 1
    n_edges_on_edge = ints(6, 11, ne + 1)
    n_edges_on_edge[ne] = 0
    edges_on_edge = np.full((ne + 1, me2), ne + 1, np.int32)
    for e in range(ne):
        edges_on_edge[e, :n_edges_on_edge[e]] = ints(1, ne + 1, n_edges_on_edge[e])
    edges_on_vertex = ints(1, ne + 1, (nv + 1, vd))
    mlvb = np.zeros(nv + 1, np.int32)
    mlvb[:nv] = ints(0, nl + 1, nv)
    for v_col in (vertices_on_edge[:ne, 0], vertices_on_edge[:ne, 1]):
        np.maximum.at(mlvb, v_col - 1, mleb[:ne])
    for c in range(nc):
        np.maximum.at(mlvb, vertices_on_cell[c, :n_edges_on_cell[c]] - 1, mlc[c])
    mlvb[nv] = 0
    mlvt = np.minimum(mlvb, ints(0, nl + 1, nv + 1)).astype(np.int32)
    mlvt[nv] = 0
    n_adv = ints(6, na + 1, ne + 1)
    n_adv[ne] = 0
    adv_cells = np.full((ne + 1, na), nc + 1, np.int32)
    for e in range(ne):
        adv_cells[e, :n_adv[e]] = ints(1, nc + 1, n_adv[e])
    level = np.arange(nl)[None, :]
    edge_mask = (level < mlet[:, None]).astype(np.int32)

    resting = uniform(10.0, 80.0, (nc + 1, nl))
    thickness = resting * uniform(0.97, 1.03, (nc + 1, nl))
    bottom_depth = np.array([thickness[c, :mlc[c]].sum() for c in range(nc + 1)])
    ref_bottom = np.cumsum(resting.mean(axis=0))
    velocity = uniform(-0.3, 0.3, (ne + 1, nl)) * (level < mlet[:, None])
    # no flow through a land edge (maxLevelEdgeTop 0), as a real state has
    btr = uniform(-0.05, 0.05, ne + 1) * (mlet > 0)
    tracers = np.empty((nc + 1, nl, nt))
    tracers[:, :, 0] = uniform(2.0, 25.0, (nc + 1, nl))
    tracers[:, :, 1] = uniform(33.5, 36.0, (nc + 1, nl))
    if overrides.get("_fresh"):
        # near-fresh cells, whose salinity the step's floor (0.001) holds
        fresh = rs.random_sample(nc + 1) < 0.25
        tracers[fresh, :, 1] = uniform(0.0, 2.0e-3, (int(fresh.sum()), nl))
    fraction = uniform(0.0, 1.0, (nc + 1, nl))
    fraction /= fraction.sum(axis=1, keepdims=True) * uniform(0.9, 1.3, (nc + 1, 1))
    ssh = uniform(-0.5, 0.5, nc + 1)
    dims = {"nCells": nc, "nEdges": ne, "nVertices": nv, "nVertLevels": nl, "nTracers": nt, "maxEdges": me,
            "maxEdges2": me2, "vertexDegree": vd, "nAdvectionCells": na}
    case = {"name": name, "precision": precision, "dims": dims, "config": config, "dt": 120.0}
    values = {
        "nVertLevels": nl, "vertexDegree": vd, "nCells": nc, "nEdges": ne, "nVertices": nv, "nCellsSolve": nc,
        "nEdgesSolve": ne, "nVerticesSolve": nv, "maxEdges": me,
        "nCellsArray": np.array([nc] * 4, np.int32), "nEdgesArray": np.array([ne] * 4, np.int32),
        "nVerticesArray": np.array([nv] * 4, np.int32),
        "maxLevelCell": mlc, "maxLevelEdgeTop": mlet, "maxLevelEdgeBot": mleb, "maxLevelVertexBot": mlvb,
        "maxLevelVertexTop": mlvt,
        "cellsOnEdge": cells_on_edge, "cellsOnVertex": cells_on_vertex, "verticesOnEdge": vertices_on_edge,
        "nEdgesOnCell": n_edges_on_cell, "nEdgesOnEdge": n_edges_on_edge, "edgesOnCell": edges_on_cell,
        "cellsOnCell": cells_on_cell, "edgesOnEdge": edges_on_edge, "edgesOnVertex": edges_on_vertex,
        "kiteIndexOnCell": kite_index, "verticesOnCell": vertices_on_cell,
        "boundaryCell": np.zeros((nc + 1, nl), np.int32),
        "edgeSignOnVertex": (2 * ints(0, 2, (nv + 1, vd)) - 1).astype(np.int32),
        "edgeSignOnCell": edge_sign_on_cell, "edgeMask": edge_mask,
        "weightsOnEdge": uniform(-0.3, 0.3, (ne + 1, me2)), "kiteAreasOnVertex": uniform(1.0e7, 1.5e8, (nv + 1, vd)),
        "dcEdge": uniform(1.0e4, 3.0e4, ne + 1), "dvEdge": uniform(5.0e3, 2.0e4, ne + 1),
        "areaCell": uniform(1.0e8, 8.0e8, nc + 1), "areaTriangle": uniform(5.0e7, 4.0e8, nv + 1),
        "bottomDepth": bottom_depth, "fVertex": uniform(-1.4e-4, 1.4e-4, nv + 1),
        "fEdge": uniform(-1.4e-4, 1.4e-4, ne + 1), "derivTwo": np.zeros((ne + 1, 2, 15)),
        "refBottomDepth": ref_bottom, "refBottomDepthTopOfCell": np.concatenate([[0.0], ref_bottom]),
        "latCell": uniform(-1.4, 1.4, nc + 1), "lonCell": uniform(0.0, 6.28, nc + 1),
        "angleEdge": uniform(-3.1, 3.1, ne + 1), "meshScaling": uniform(0.5, 1.0, ne + 1),
        "meshScalingDel2": uniform(0.3, 1.0, ne + 1), "meshScalingDel4": uniform(0.3, 1.0, ne + 1),
        "coeffs_reconstruct": uniform(-0.5, 0.5, (nc + 1, me, 3)),
        "advCoefs": uniform(-0.5, 0.5, (ne + 1, na)), "advCoefs3rd": uniform(-0.2, 0.2, (ne + 1, na)),
        "highOrderAdvectionMask": (rs.random_sample((ne + 1, nl)) < 0.8).astype(np.int32),
        "nAdvCellsForEdge": n_adv, "advCellsForEdge": adv_cells, "vertCoordMovementWeights": np.ones(nl),
        "restingThickness": resting,
        "layerThickness": thickness, "layerThickness@2": thickness, "normalVelocity": (velocity + btr[:, None]) * (level < mlet[:, None]),
        "normalVelocity@2": velocity, "ssh": ssh, "ssh@2": ssh,
        "normalBarotropicVelocity": btr, "normalBarotropicVelocity@2": btr,
        "normalBaroclinicVelocity": velocity, "normalBaroclinicVelocity@2": velocity,
        "sshSubcycle": ssh, "sshSubcycle@2": ssh, "normalBarotropicVelocitySubcycle": btr,
        "normalBarotropicVelocitySubcycle@2": btr,
        "index_temperature": 1, "index_salinity": 2, "activeGRP_start": 1, "activeGRP_end": nt,
        "activeTracers": tracers, "activeTracers@2": tracers,
        "index_temperatureSurfaceFlux": 1, "index_salinitySurfaceFlux": 2,
        "fractionAbsorbed": fraction, "fractionAbsorbedRunoff": fraction[::-1].copy(),
        "seaIcePressure": uniform(0.0, 2.0e4, nc + 1), "atmosphericPressure": uniform(9.8e4, 1.03e5, nc + 1),
        "latentHeatFlux": uniform(-150.0, 0.0, nc + 1), "sensibleHeatFlux": uniform(-50.0, 50.0, nc + 1),
        "longWaveHeatFluxUp": uniform(-420.0, -300.0, nc + 1), "longWaveHeatFluxDown": uniform(250.0, 400.0, nc + 1),
        "seaIceHeatFlux": uniform(-20.0, 0.0, nc + 1), "rainFlux": uniform(0.0, 1.0e-4, nc + 1),
        "snowFlux": uniform(0.0, 2.0e-5, nc + 1), "shortWaveHeatFlux": uniform(0.0, 300.0, nc + 1),
        "evaporationFlux": uniform(-5.0e-5, 0.0, nc + 1), "seaIceFreshWaterFlux": uniform(-1.0e-5, 1.0e-5, nc + 1),
        "seaIceSalinityFlux": uniform(-1.0e-6, 1.0e-6, nc + 1), "iceRunoffFlux": uniform(0.0, 1.0e-6, nc + 1),
        "riverRunoffFlux": uniform(0.0, 5.0e-6, nc + 1), "penetrativeTemperatureFlux": uniform(0.0, 0.05, nc + 1),
        "windStressZonal": uniform(-0.2, 0.2, nc + 1), "windStressMeridional": uniform(-0.2, 0.2, nc + 1),
        "index_surfaceVelocityZonal": 1, "index_surfaceVelocityMeridional": 2, "index_SSHGradientZonal": 1,
        "index_SSHGradientMeridional": 2, "nAccumulatedCoupled": 2, "index_avgTemperatureSurfaceValue": 1,
        "index_avgSSHGradientZonal": 1, "index_avgSSHGradientMeridional": 2,
        "avgTracersSurfaceValue": uniform(0.0, 300.0, (nc + 1, nt)),
        "avgSurfaceVelocity": uniform(-0.3, 0.3, (nc + 1, 2)), "avgSSHGradient": uniform(-1.0e-5, 1.0e-5, (nc + 1, 2)),
        "vertNonLocalFlux": np.zeros((nc + 1, nl + 1, nt)), "nonLocalSurfaceTracerFlux": np.zeros((nc + 1, nt)),
        "activeTracersSurfaceFluxRunoff": uniform(-1.0e-6, 1.0e-6, (nc + 1, nt)),
        "normalGMBolusVelocity": np.zeros((ne + 1, nl)), "k33": np.zeros((nc + 1, nl + 1)),
        "relativeSlopeTopOfEdge": np.zeros((ne + 1, nl)), "relativeSlopeTapering": np.zeros((ne + 1, nl)),
        "relativeSlopeTaperingCell": np.zeros((nc + 1, nl)), "barotropicForcing": np.zeros(ne + 1),
    }
    values["nCells"] = nc
    pools: dict[str, dict] = {}
    for pool, field, kind, fshape in FIELDS:
        if kind in (DIM, DIMS):
            value = values[field]
        elif kind == SCRATCH or (pool, field) in ABSENT or not _diag.present(case, field):
            value = None
        elif field in values:
            value = values[field]
        else:
            # a field the step writes before it reads: prior words it must overwrite or keep
            value = uniform(-1.0, 1.0, _stream.shape(case, fshape))
        if kind == REAL and value is not None:
            value = np.asarray(value).astype(real)
        pools.setdefault(pool, {})[field] = value
    case["pools"] = pools
    return case


def cases(precision: str = "r4") -> list[dict]:
    return [make_case(*row, precision=precision) for row in CASES]


def input_bytes(case: dict) -> bytes:
    return _stream.input_bytes(FIELDS, case)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    return _stream.read_outputs(FIELDS, raw, case)


word_mismatches = _stream.word_mismatches


def load_fixture(precision: str) -> dict[str, dict[str, np.ndarray]]:
    return _tend.load_fixture(precision, FIXTURE_DIR)


def run_gpu(case: dict) -> dict[str, np.ndarray]:
    """Run woof.hex.mpas_ocean.split_step_v842 on ``case`` as run_split_step.F90
    drives the Fortran: the starting diagnostics, then the step."""
    import cupy as cp
    from woof.hex.mpas_ocean import diagnostics_v842 as dg
    from woof.hex.mpas_ocean import split_step_v842 as ss

    dev = _stream.on_device(FIELDS, case)
    config = case["config"]
    st, tr = dev["state"], dev["tracers"]
    dg.diagnostic_solve(dt=case["dt"], mesh=dev["mesh"],
                        state={n: st[n] for n in ("layerThickness", "normalVelocity", "ssh")},
                        tracers={n: tr[n] for n in ("activeTracers", "index_temperature", "index_salinity")},
                        diagnostics=dev["diagnostics"], forcing=dev["forcing"], configs=config)
    ss.split_step(dt=case["dt"], n_btr_subcycles=int(config["nBtrSubcycles"]), pools=dev, configs=config)
    return {f"{pool}/{name}": cp.asnumpy(dev[pool][name]) for pool, name, kind, _ in FIELDS
            if kind in (REAL, INT) and dev[pool][name] is not None}


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision):
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: word_mismatches(got[name], want[name]) for name in want}
    return report
