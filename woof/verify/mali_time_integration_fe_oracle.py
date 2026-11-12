"""Cases and the stream layout for the MALI forward-Euler time-step oracle.

``tools/mali_oracle`` compiles MALI v8.4.2's land-ice modules verbatim with
gfortran -O0 -ffp-contract=off at RKIND 4 and 8, and
``run_time_integration_fe.F90`` makes a restart's ``li_thermal_init`` and
then one ``li_time_integrator_forwardeuler``. Its input stream is
self-describing (the clock, every config by name, every field by pool and
name), so this module owns the field table: :data:`FIELDS` names each field's
pool, type and shape, and a case is one dict per pool plus the configs.

The meshes are random connectivity (1-based, every index the routines read
inside the arrays). Fields are drawn in float64 from numpy's legacy
``RandomState`` with ``random_sample`` only and rounded once to the case's
precision, so the bytes do not depend on the platform. ``input_digest`` is
pinned beside the fixture.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mali_time_integration_fe"
PRECISIONS = {"r4": np.float32, "r8": np.float64}
STRKIND = 512
POOLS = ("mesh", "geometry", "thermal", "velocity", "hydro", "scratch")

HYDRO_CELL = ("basalMeltInput", "externalWaterInput", "waterThickness", "waterThicknessOld", "waterThicknessTendency",
              "tillWaterThickness", "tillWaterThicknessOld", "waterVelocityCellX", "waterVelocityCellY",
              "hydropotentialBase", "hydropotential", "waterPressure", "waterPressureOld", "waterPressureTendency",
              "effectivePressure", "divergence", "divergenceChannel", "channelAreaChangeCell", "openingRate",
              "closingRate", "zeroOrderSum")
HYDRO_EDGE = ("waterFlux", "waterFluxAdvec", "waterFluxDiffu", "waterVelocity", "waterThicknessEdge",
              "waterThicknessEdgeUpwind", "hydropotentialBaseSlopeNormal", "hydropotentialBaseSlopeTangent",
              "waterPressureSlopeNormal", "gradMagPhiEdge", "effectiveConducEdge", "diffusivity", "channelArea",
              "channelDischarge", "channelVelocity", "channelDiffusivity", "channelMelt", "channelPressureFreeze",
              "channelOpeningRate", "channelClosingRate", "channelChangeRate", "flowParamAChannel",
              "channelEffectivePressure")
HYDRO_SCALAR = ("deltatSGH", "deltatSGHadvec", "deltatSGHdiffu", "deltatSGHpressure", "deltatSGHadvecChannel",
                "deltatSGHdiffuChannel")

# (pool, name, type, shape in dimension names): type 'r' real, 'i' int32,
# 'd' a dimension. Shapes are C-order (the Fortran's reversed).
_C, _E, _V, _L, _I = "nCells", "nEdges", "nVertices", "nVertLevels", "nVertInterfaces"
FIELDS = (
    *(("mesh", d, "d", ()) for d in ("nCells", "nEdges", "nVertices", "nVertLevels", "nVertInterfaces",
                                    "nCellsSolve", "nEdgesSolve", "vertexDegree", "maxEdges", "maxEdges2")),
    ("geometry", "nCells", "d", ()),
    ("mesh", "nEdgesOnCell", "i", (_C,)), ("mesh", "cellsOnCell", "i", (_C, "maxEdges")),
    ("mesh", "cellsOnVertex", "i", (_V, 3)), ("mesh", "cellsOnEdge", "i", (_E, 2)),
    ("mesh", "edgesOnCell", "i", (_C, "maxEdges")), ("mesh", "edgeSignOnCell", "i", (_C, "maxEdges")),
    ("mesh", "verticesOnEdge", "i", (_E, 2)), ("mesh", "baryCellsOnVertex", "i", (_V, 3)),
    ("mesh", "nEdgesOnEdge", "i", (_E,)), ("mesh", "edgesOnEdge", "i", (_E, "maxEdges2")),
    ("mesh", "indexToCellID", "i", (_C,)), ("mesh", "indexToEdgeID", "i", (_E,)),
    ("mesh", "layerThicknessFractions", "r", (_L,)), ("mesh", "layerCenterSigma", "r", (_L,)),
    ("mesh", "layerInterfaceSigma", "r", (_I,)), ("mesh", "dvEdge", "r", (_E,)), ("mesh", "dcEdge", "r", (_E,)),
    ("mesh", "areaCell", "r", (_C,)), ("mesh", "xCell", "r", (_C,)), ("mesh", "latCell", "r", (_C,)),
    ("mesh", "lonCell", "r", (_C,)), ("mesh", "baryWeightsOnVertex", "r", (_V, 3)),
    ("mesh", "kiteAreasOnVertex", "r", (_V, 3)), ("mesh", "weightsOnEdge", "r", (_E, "maxEdges2")),
    ("mesh", "edgeNormalVectors", "r", (_E, 3)), ("mesh", "coeffs_reconstruct", "r", (_C, "maxEdges", 3)),
    ("mesh", "deltat", "r", ()), ("mesh", "daysSinceStart", "r", ()), ("mesh", "allowableDtACFL", "r", ()),
    ("mesh", "allowableDtDCFL", "r", ()),
    *(("geometry", n, "r", (_C,)) for n in (
        "thickness", "thicknessOld", "dHdt", "bedTopography", "upperSurface", "lowerSurface", "sfcMassBal",
        "basalMassBal", "groundedBasalMassBal", "floatingBasalMassBal", "dynamicThickening", "calvingThickness",
        "basalWaterThickness", "apparentDiffusivity", "restoreThickness", "eigencalvingParameter",
        "requiredCalvingVolumeRate", "calvingVelocity", "uncalvedVolume")),
    *(("geometry", n, "r", (_E,)) for n in ("normalSlopeEdge", "slopeEdge", "tangentSlopeEdge")),
    ("geometry", "upperSurfaceVertex", "r", (_V,)),
    ("geometry", "layerThickness", "r", (_C, _L)), ("geometry", "layerThicknessEdge", "r", (_E, _L)),
    ("geometry", "cellMask", "i", (_C,)), ("geometry", "vertexMask", "i", (_V,)), ("geometry", "edgeMask", "i", (_E,)),
    ("velocity", "normalVelocity", "r", (_E, _I)), ("velocity", "layerNormalVelocity", "r", (_E, _L)),
    *(("velocity", n, "r", (_C, _I)) for n in ("uReconstructX", "uReconstructY", "uReconstructZ",
                                               "uReconstructZonal", "uReconstructMeridional")),
    ("velocity", "flowParamA", "r", (_C, _L)),
    *(("velocity", n, "r", (_C,)) for n in ("surfaceSpeed", "basalSpeed", "betaSolve", "exx", "eyy", "exy", "eyx",
                                            "eTheta", "eMax", "eMin")),
    *(("thermal", n, "r", (_C, _L)) for n in ("temperature", "waterfrac", "enthalpy", "heatDissipation",
                                              "pmpTemperature")),
    *(("thermal", n, "r", (_C,)) for n in ("surfaceTemperature", "basalTemperature", "surfaceAirTemperature",
                                           "basalHeatFlux", "basalFrictionFlux", "surfaceConductiveFlux",
                                           "basalConductiveFlux", "basalPmpTemperature")),
    # the subglacial hydrology's pool
    *(("hydro", n, "r", (_C,)) for n in HYDRO_CELL),
    ("hydro", "hydropotentialBaseVertex", "r", (_V,)),
    *(("hydro", n, "r", (_E,)) for n in HYDRO_EDGE),
    ("hydro", "waterFluxMask", "i", (_E,)),
    *(("hydro", n, "r", ()) for n in HYDRO_SCALAR),
    # the model's scratch fields: uninitialised in the model, random words here
    *(("scratch", n, "r", (_C,)) for n in ("workCell", "workCell2", "workCell3", "slopeCellX", "slopeCellY")),
    ("scratch", "workLevelCell", "r", (_C, _L)), ("scratch", "workTracerCell", "r", (_C, 1)),
    ("scratch", "workTracerCell2", "r", (_C, 1)), ("scratch", "workTracerLevelCell", "r", (_C, _L, 1)),
    ("scratch", "workTracerLevelCell2", "r", (_C, _L, 1)), ("scratch", "workLevelEdge", "r", (_E, _L)),
    ("scratch", "iceCellMask", "i", (_C,)), ("scratch", "iceCellMask2", "i", (_C,)),
    ("scratch", "iceCellMask3", "i", (_C,)),
)
#: the fields a step may change, compared word for word (the mesh's own words and the scratch are not)
OUTPUTS = tuple((pool, name) for pool, name, kind, _ in FIELDS
                if pool in ("geometry", "thermal", "velocity", "hydro") and kind != "d") + (
    ("mesh", "deltat"), ("mesh", "allowableDtACFL"), ("mesh", "allowableDtDCFL"))
#: refused at RKIND 8 (glibc's binary64 atan): not in the r8 fixture
R8_REFUSED = (("velocity", "eTheta"),)
#: The fixture "precision" of the RKIND 8 cases run under the opt-in glibc
#: 2.39 sin/cos/atan transcription (WOOF_GLIBC239_LIBM64=1): RKIND 8 with
#: nothing refused, its words in oracle-r8-glibc239.npz
R8_GLIBC239 = "r8-glibc239"
#: the cases whose RKIND 8 configuration the opt-in lifts: the strain-rate
#: angle (every SIA case), the sphere, 'seroussi', the SHMIP C and D forcings
GLIBC239_CASES = ("sia_temperature_fo", "none_temperature_no_tracers", "sia_no_thermal_sphere",
                  "sgh_overburden_shmip", "sgh_shmip_d")

#: (name, seed, nCells, nVertLevels, overrides of BASE_CONFIG)
CASES = (
    ("sia_temperature_fo", 20261161, 96, 8,
     {"config_thermal_solver": "temperature", "config_basal_mass_bal_float": "constant"}),
    ("sia_enthalpy_adaptive", 20261162, 96, 10,
     {"config_thermal_solver": "enthalpy", "config_basal_mass_bal_float": "mismip",
      "config_adaptive_timestep": True, "config_adaptive_timestep_include_DCFL": True,
      "config_sia_tangent_slope_calculation": "from_vertex_barycentric_kiteareas",
      "config_flowParamA_calculation": "PB1982", "config_surface_air_temperature_source": "lapse"}),
    ("none_temperature_no_tracers", 20261163, 80, 6,
     {"config_velocity_solver": "none", "config_thermal_solver": "temperature", "config_tracer_advection": "none",
      "config_basal_mass_bal_float": "seroussi", "config_adaptive_timestep": True,
      "config_max_adaptive_timestep": 1234567.8}),
    ("sia_no_thermal_sphere", 20261164, 72, 5,
     {"config_thermal_solver": "none", "on_a_sphere": True, "config_flowParamA_calculation": "constant",
      "config_sia_tangent_slope_calculation": "from_normal_slope"}),
    ("sia_no_advection", 20261165, 64, 6,
     {"config_thickness_advection": "none", "config_thermal_solver": "enthalpy",
      "config_basal_mass_bal_float": "file", "config_thermal_calculate_bmb": False}),
    ("calving_thickness_threshold", 20261171, 96, 6,
     {"config_calving": "thickness_threshold", "config_calving_thickness": 300.0,
      "config_calving_timescale": 8.0e6}),
    ("calving_floating_data", 20261172, 80, 6,
     {"config_calving": "floating", "config_data_calving": True, "config_thermal_solver": "enthalpy"}),
    ("calving_topographic", 20261173, 80, 5,
     {"config_calving": "topographic_threshold", "config_calving_topography": -400.0,
      "config_velocity_solver": "none", "config_calving_timescale": 1.0e6}),
    ("calving_eigen_scalar", 20261174, 200, 6,
     {"config_calving": "eigencalving", "config_calving_thickness": 80.0, "config_thermal_solver": "none",
      "config_calving_eigencalving_parameter_scalar_value": 3.0e9, "_bed": (-2500.0, -200.0), "_speed": 0.01}),
    ("calving_eigen_data", 20261175, 200, 6,
     {"config_calving": "eigencalving", "config_calving_eigencalving_parameter_source": "data",
      "config_thermal_solver": "none",
      "config_calving_thickness": 40.0, "_bed": (-2500.0, 0.0), "_speed": 0.01}),
    ("calving_restore_front", 20261176, 96, 6,
     {"config_restore_calving_front": True, "config_calving": "floating", "config_thermal_solver": "enthalpy"}),
    ("sgh_sheet_cavity", 20261181, 96, 6,
     {"config_SGH": True, "config_thermal_solver": "none", "config_SGH_max_adaptive_timestep": 1500.0, "_speed": 0.01, "_bed": (-200.0, 1500.0), "_deltat": 3600.0,
      "config_SGH_englacial_porosity": 1.0, "config_SGH_conduc_coeff": 1.0e-4}),
    ("sgh_channels_thermal", 20261182, 96, 6,
     {"config_SGH": True, "config_SGH_chnl_active": True, "config_SGH_basal_melt": "thermal",
      "config_SGH_tangent_slope_calculation": "from_vertex_barycentric", "config_thermal_solver": "temperature",
      "config_thermal_calculate_bmb": False, "_speed": 0.01, "_bed": (-200.0, 1500.0), "_deltat": 3600.0,
      "config_SGH_englacial_porosity": 1.0, "config_SGH_conduc_coeff": 1.0e-4}),
    ("sgh_overburden_shmip", 20261183, 80, 5,
     {"config_SGH": True, "config_SGH_pressure_calc": "overburden", "config_SGH_basal_melt": "basal_heat",
      "config_SGH_tangent_slope_calculation": "from_vertex_barycentric_kiteareas",
      "config_SGH_shmip_forcing": "C2", "config_SGH_chnl_active": True, "config_SGH_include_pressure_melt": False,
      "config_thermal_solver": "none", "_speed": 0.01, "_bed": (-200.0, 1500.0), "_deltat": 3600.0,
      "config_SGH_englacial_porosity": 1.0, "config_SGH_conduc_coeff": 1.0e-4}),
    ("sgh_shmip_d", 20261184, 80, 5,
     {"config_SGH": True, "config_SGH_shmip_forcing": "D3", "config_thermal_solver": "none",
      "config_SGH_max_adaptive_timestep": 900.0, "_speed": 0.01, "_bed": (-200.0, 1500.0), "_deltat": 3600.0,
      "config_SGH_englacial_porosity": 1.0, "config_SGH_conduc_coeff": 1.0e-4}),
)
#: at RKIND 8 the SHMIP forcings (glibc's binary64 sin, cos) are refused: those cases run 'none'
R8_SHMIP = "none"
#: at RKIND 8 the cases run planar and without the floating melt whose sin is refused
R8_OVERRIDES = {"on_a_sphere": False}
R8_FLOAT_MELT = {"seroussi": "none"}

BASE_CONFIG = {
    "config_ice_density": 910.0, "config_ocean_density": 1028.0, "config_sea_level": 0.0,
    "config_dynamic_thickness": 10.0, "config_velocity_solver": "sia", "config_thickness_advection": "fo",
    "config_tracer_advection": "fo", "config_thermal_solver": "temperature", "config_temperature_init": "file",
    "config_surface_air_temperature_source": "file", "config_surface_air_temperature_value": 268.0,
    "config_surface_air_temperature_lapse_rate": 0.006, "config_basal_heat_flux_source": "file",
    "config_basal_heat_flux_value": 0.0, "config_do_restart": True, "config_thermal_thickness": 1.0,
    "config_thermal_calculate_bmb": True, "config_flowLawExponent": 3.0, "config_enhancementFactor": 1.0,
    "config_default_flowParamA": 3.1709792e-24, "config_flowParamA_calculation": "CP2010",
    "config_sia_tangent_slope_calculation": "from_vertex_barycentric", "config_basal_mass_bal_float": "none",
    "config_bmlt_float_flux": 100.0, "config_bmlt_float_xlimit": 1.0e5,
    "config_basal_mass_bal_seroussi_amplitude": 50.0, "config_basal_mass_bal_seroussi_period": 2.0,
    "config_basal_mass_bal_seroussi_phase": 0.7, "config_adaptive_timestep": False,
    "config_adaptive_timestep_include_DCFL": False, "config_adaptive_timestep_CFL_fraction": 0.25,
    "config_max_adaptive_timestep": 3.1536e7, "config_min_adaptive_timestep": 60.0,
    "config_restore_calving_front": False, "config_calving": "none", "config_calving_timescale": 0.0,
    "config_calving_thickness": 100.0, "config_calving_topography": -500.0,
    "config_calving_eigencalving_parameter_source": "scalar",
    "config_calving_eigencalving_parameter_scalar_value": 1.0e9,
    "config_data_calving": False, "config_do_velocity_reconstruction_for_external_dycore": False,
    "config_SGH": False, "config_SGH_adaptive_timestep_fraction": 1.0, "config_SGH_max_adaptive_timestep": 3.15e9,
    "config_SGH_tangent_slope_calculation": "from_normal_slope", "config_SGH_pressure_calc": "cavity",
    "config_SGH_alpha": 1.25, "config_SGH_beta": 1.5, "config_SGH_conduc_coeff": 0.001,
    "config_SGH_till_drainage": 3.1709792e-11, "config_SGH_bed_roughness": 0.5, "config_SGH_bed_roughness_max": 0.1,
    "config_SGH_creep_coefficient": 0.04, "config_SGH_englacial_porosity": 0.01, "config_SGH_till_max": 2.0,
    "config_SGH_chnl_active": False, "config_SGH_chnl_alpha": 1.25, "config_SGH_chnl_beta": 1.5,
    "config_SGH_chnl_conduc_coeff": 0.1, "config_SGH_chnl_creep_coefficient": 0.04,
    "config_SGH_incipient_channel_width": 2.0, "config_SGH_include_pressure_melt": True,
    "config_SGH_shmip_forcing": "none", "config_SGH_basal_melt": "file",
    "config_print_thickness_advection_info": False, "config_print_thermal_info": False,
    "config_print_calving_info": False, "config_print_velocity_cleanup_details": False, "config_stats_cell_ID": 0,
    "on_a_sphere": False,
}


def field_shape(case: dict, shape: tuple) -> tuple[int, ...]:
    return tuple(case["dims"][s] if isinstance(s, str) else s for s in shape)


def make_case(name: str, seed: int, n_cells: int, n_levels: int, overrides: dict, precision: str = "r4",
              glibc239: bool = False) -> dict:
    """One deterministic case: grounded and floating ice, thin ice and open
    ocean on a random mesh, with an ice sheet's magnitudes.

    - ice up to 2500 m on four fifths of the cells over beds from -1200 m to
      900 m, so the masks find grounded and floating ice, margins and open
      ocean, and SIA flow fast enough that floating ice past 100 km/yr is
      removed;
    - prior velocities slow enough for the advective CFL limit of a 30-day
      step; reconstruction coefficients of a 2-20 km mesh;
    - column temperatures from 240 K to 273 K and water fractions up to 0.03,
      so the thermal solver's branches all run (tests/test_mali_thermal_solver_parity.py
      holds those in detail);
    - prior words in every output, so the step must write what it owns.

    nCellsSolve = nCells: li_calculate_apparent_diffusivity reads the
    reconstructed slope on every cell, which mpas_reconstruct sets only on
    the owned cells, so in a halo the model reads its uninitialised scratch.
    """
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    nc, nl = n_cells, n_levels
    ne, nv, me, me2 = 3 * nc, 2 * nc, 7, 10
    config = dict(BASE_CONFIG, **{k: v for k, v in overrides.items() if not k.startswith("_")})
    bed = overrides.get("_bed", (-1200.0, 900.0))
    if precision == "r8" and not glibc239:
        config.update(R8_OVERRIDES)
        config["config_basal_mass_bal_float"] = R8_FLOAT_MELT.get(config["config_basal_mass_bal_float"],
                                                                   config["config_basal_mass_bal_float"])
        config["config_SGH_shmip_forcing"] = R8_SHMIP
    dims = {"nCells": nc, "nEdges": ne, "nVertices": nv, "nVertLevels": nl, "nVertInterfaces": nl + 1,
            "nCellsSolve": nc, "nEdgesSolve": ne - 5, "vertexDegree": 3, "maxEdges": me, "maxEdges2": me2}

    def uniform(lo, hi, shape):
        return (lo + (hi - lo) * rs.random_sample(shape)).astype(real)

    def ints(lo, hi, shape):
        return (lo + np.floor((hi - lo) * rs.random_sample(shape))).astype(np.int32)

    year = 365.0 * 86400.0
    thickness = uniform(1.0, overrides.get("_max_thickness", 2500.0), nc)
    thickness[rs.random_sample(nc) < 0.2] = 0.0
    thin = rs.random_sample(nc) < 0.08
    thickness[thin] = uniform(0.05, 1.0, int(thin.sum()))
    fractions = uniform(0.5, 1.5, nl).astype(np.float64)
    fractions = (fractions / fractions.sum()).astype(real)
    interfaces = np.concatenate([[0.0], np.cumsum(fractions.astype(np.float64))]).astype(real)
    interfaces[-1] = 1.0
    centres = (0.5 * (interfaces[:-1].astype(np.float64) + interfaces[1:])).astype(real)
    n_edges_on_edge = ints(6, me2 + 1, ne)
    edges_on_edge = np.full((ne, me2), ne + 5, np.int32)
    for e in range(ne):
        edges_on_edge[e, :n_edges_on_edge[e]] = ints(1, ne + 1, int(n_edges_on_edge[e]))
    velocity_scale = overrides.get("_speed", 30.0) / year
    pools = {
        "mesh": {
            "nEdgesOnCell": np.array([5, 6, 6, 6, 7], np.int32)[ints(0, 5, nc)],
            "cellsOnCell": ints(1, nc + 1, (nc, me)), "cellsOnVertex": ints(1, nc + 1, (nv, 3)),
            "cellsOnEdge": ints(1, nc + 1, (ne, 2)), "edgesOnCell": ints(1, ne + 1, (nc, me)),
            "edgeSignOnCell": np.where(rs.random_sample((nc, me)) < 0.5, -1, 1).astype(np.int32),
            "verticesOnEdge": ints(1, nv + 1, (ne, 2)), "baryCellsOnVertex": ints(1, nc + 1, (nv, 3)),
            "nEdgesOnEdge": n_edges_on_edge, "edgesOnEdge": edges_on_edge,
            "indexToCellID": np.arange(1, nc + 1, dtype=np.int32),
            "indexToEdgeID": np.arange(1, ne + 1, dtype=np.int32),
            "layerThicknessFractions": fractions, "layerCenterSigma": centres, "layerInterfaceSigma": interfaces,
            "dvEdge": uniform(2.0e3, 2.0e4, ne), "dcEdge": uniform(2.0e3, 2.0e4, ne),
            "areaCell": uniform(4.0e6, 4.0e8, nc), "xCell": uniform(-5.0e5, 5.0e5, nc),
            "latCell": uniform(-1.4, -1.0, nc), "lonCell": uniform(-3.1, 3.1, nc),
            "baryWeightsOnVertex": uniform(-0.1, 0.8, (nv, 3)), "kiteAreasOnVertex": uniform(1.0e6, 5.0e7, (nv, 3)),
            "weightsOnEdge": uniform(-0.5, 0.5, (ne, me2)), "edgeNormalVectors": uniform(-1.0, 1.0, (ne, 3)),
            "coeffs_reconstruct": uniform(-0.6, 0.6, (nc, me, 3)),
            "deltat": np.array(overrides.get("_deltat", 30.0 * 86400.0), real), "daysSinceStart": np.array(400.5, real),
            "allowableDtACFL": uniform(-1.0, 1.0, ()), "allowableDtDCFL": uniform(-1.0, 1.0, ()),
        },
        "geometry": {
            "thickness": thickness, "thicknessOld": uniform(0.0, 2500.0, nc), "dHdt": uniform(-1.0, 1.0, nc),
            "bedTopography": uniform(*bed, nc), "upperSurface": uniform(0.0, 3000.0, nc),
            "lowerSurface": uniform(-900.0, 0.0, nc), "sfcMassBal": uniform(-3.0e-5, 3.0e-5, nc),
            "basalMassBal": uniform(-1.0, 1.0, nc), "groundedBasalMassBal": uniform(-3.0e-5, 3.0e-5, nc),
            "floatingBasalMassBal": uniform(-3.0e-5, 0.0, nc), "dynamicThickening": uniform(-1.0, 1.0, nc),
            "calvingThickness": uniform(0.0, 5.0, nc),
            "basalWaterThickness": np.where(rs.random_sample(nc) < 0.5, uniform(0.0, 1.0, nc), real(0.0)).astype(real),
            "apparentDiffusivity": uniform(-1.0, 1.0, nc), "restoreThickness": uniform(-1.0, 1.0, nc),
            "eigencalvingParameter": uniform(1.0e8, 1.0e10, nc),
            "requiredCalvingVolumeRate": uniform(-1.0, 1.0, nc), "calvingVelocity": uniform(-1.0, 1.0, nc),
            "uncalvedVolume": uniform(-1.0, 1.0, nc),
            "normalSlopeEdge": uniform(-0.03, 0.03, ne), "slopeEdge": uniform(-1.0, 1.0, ne),
            "tangentSlopeEdge": uniform(-1.0, 1.0, ne), "upperSurfaceVertex": uniform(-1.0, 1.0, nv),
            "layerThickness": uniform(0.0, 300.0, (nc, nl)), "layerThicknessEdge": uniform(-1.0, 1.0, (ne, nl)),
            "cellMask": ints(0, 512, nc), "vertexMask": ints(0, 512, nv), "edgeMask": ints(0, 512, ne),
        },
        "velocity": {
            "normalVelocity": uniform(-velocity_scale, velocity_scale, (ne, nl + 1)),
            # the edges past nEdgesSolve keep these words, and advection reads them
            "layerNormalVelocity": uniform(-velocity_scale, velocity_scale, (ne, nl)),
            **{n: uniform(-velocity_scale, velocity_scale, (nc, nl + 1))
               for n in ("uReconstructX", "uReconstructY", "uReconstructZ", "uReconstructZonal",
                         "uReconstructMeridional")},
            "flowParamA": uniform(1.0e-25, 1.0e-23, (nc, nl)),
            "surfaceSpeed": uniform(-1.0, 1.0, nc), "basalSpeed": uniform(0.0, 300.0 / year, nc),
            "betaSolve": uniform(0.0, 1.0e3, nc),
            **{n: uniform(-1.0, 1.0, nc) for n in ("exx", "eyy", "exy", "eyx", "eTheta", "eMax", "eMin")},
        },
        "thermal": {
            "temperature": uniform(240.0, 273.0, (nc, nl)), "waterfrac": uniform(0.0, 0.03, (nc, nl)),
            "enthalpy": uniform(-6.0e8, -4.0e8, (nc, nl)), "heatDissipation": uniform(0.0, 1.0e-8, (nc, nl)),
            "pmpTemperature": uniform(-1.0, 1.0, (nc, nl)),
            "surfaceTemperature": uniform(230.0, 273.0, nc), "basalTemperature": uniform(250.0, 273.0, nc),
            "surfaceAirTemperature": uniform(230.0, 285.0, nc), "basalHeatFlux": uniform(0.03, 0.12, nc),
            "basalFrictionFlux": uniform(-1.0, 1.0, nc), "surfaceConductiveFlux": uniform(-1.0, 1.0, nc),
            "basalConductiveFlux": uniform(-1.0, 1.0, nc), "basalPmpTemperature": uniform(-1.0, 1.0, nc),
        },
    }
    case = {"name": name, "precision": precision, "glibc239": glibc239, "dims": dims, "config": config,
            "clock": (int(400 * 86400), int(400 * 86400 + 20 * 86400)), "pools": pools}
    scratch = {}
    for pool, field, kind, shape in FIELDS:
        if pool == "scratch":
            s = field_shape(case, shape)
            scratch[field] = ints(-5, 6, s) if kind == "i" else uniform(-1.0e3, 1.0e3, s)
    pools["scratch"] = scratch
    # the hydro pool, drawn last so the other fields' words do not depend on it
    hydro = {name: uniform(-1.0, 1.0, nc) for name in HYDRO_CELL}
    hydro.update({
        "basalMeltInput": uniform(0.0, 2.0e-5, nc), "externalWaterInput": uniform(0.0, 1.0e-5, nc),
        "waterThickness": uniform(0.0, 0.3, nc), "tillWaterThickness": uniform(0.0, 2.0, nc),
        "tillWaterThicknessOld": uniform(0.0, 2.0, nc), "waterPressure": uniform(0.0, 1.5e7, nc),
        "hydropotentialBase": uniform(-5.0e6, 3.0e7, nc), "effectivePressure": uniform(0.0, 3.0e6, nc)})
    hydro["hydropotentialBaseVertex"] = uniform(-5.0e6, 3.0e7, nv)
    hydro.update({name: uniform(-1.0, 1.0, ne) for name in HYDRO_EDGE})
    hydro.update({"channelArea": uniform(0.0, 20.0, ne), "flowParamAChannel": uniform(1.0e-25, 1.0e-23, ne),
                  "channelEffectivePressure": uniform(0.0, 3.0e6, ne)})
    hydro["waterFluxMask"] = np.where(rs.random_sample(ne) < 0.05, 2, 0).astype(np.int32)
    hydro.update({name: uniform(1.0e3, 1.0e5, ()) for name in HYDRO_SCALAR})
    hydro["deltatSGH"] = np.array(overrides.get("_sgh_dt", 3600.0), real)
    pools["hydro"] = hydro
    return case


def cases(precision: str = "r4") -> list[dict]:
    """The cases at 'r4', 'r8' or :data:`R8_GLIBC239` (RKIND 8 under the opt-in)."""
    if precision == R8_GLIBC239:
        return [make_case(*row, precision="r8", glibc239=True) for row in CASES if row[0] in GLIBC239_CASES]
    return [make_case(*row, precision=precision) for row in CASES]


def refused(case: dict) -> tuple:
    """The outputs ``case`` leaves out: the RKIND 8 refusals, unless it runs under the opt-in."""
    return R8_REFUSED if case["precision"] == "r8" and not case.get("glibc239") else ()


def _real(case: dict) -> np.dtype:
    return np.dtype(PRECISIONS[case["precision"]]).newbyteorder("<")


def _text(value: str, width: int) -> bytes:
    raw = value.encode("ascii")
    if len(raw) > width:
        raise ValueError(f"{value!r} is longer than {width} characters")
    return raw.ljust(width, b" ")


def input_bytes(case: dict) -> bytes:
    """run_time_integration_fe.F90's input stream for ``case``."""
    real = _real(case)
    config = case["config"]
    parts = [np.array(case["clock"], "<i8").tobytes(), np.array([len(config), len(FIELDS)], "<i4").tobytes()]
    for name in sorted(config):
        value = config[name]
        if isinstance(value, bool):
            parts += [_text(name, 64), np.array([3, int(value)], "<i4").tobytes()]
        elif isinstance(value, int):
            parts += [_text(name, 64), np.array([2, value], "<i4").tobytes()]
        elif isinstance(value, float):
            parts += [_text(name, 64), np.array([1], "<i4").tobytes(), np.array(value, real).tobytes()]
        else:
            parts += [_text(name, 64), np.array([4], "<i4").tobytes(), _text(value, STRKIND)]
    for pool, name, kind, shape in FIELDS:
        s = field_shape(case, shape)
        code = {"r": 1, "i": 2, "d": 3}[kind]
        dims = list(reversed(s)) + [1] * (3 - len(s))
        parts += [_text(pool, 16), _text(name, 64), np.array([code, len(s), *dims], "<i4").tobytes()]
        if kind == "d":
            parts.append(np.array(case["dims"][name], "<i4").tobytes())
        else:
            value = case["pools"][pool][name]
            if tuple(np.shape(value)) != s:
                raise ValueError(f"{pool}/{name} has shape {np.shape(value)}, not {s}")
            parts.append(np.ascontiguousarray(value, real if kind == "r" else "<i4").tobytes())
    return b"".join(parts)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    """run_time_integration_fe.F90's output stream -> ``pool/name`` arrays
    (:data:`OUTPUTS`, without the RKIND 8 refusals), plus the error words."""
    head = np.frombuffer(raw, "<i4", 3, 0)
    out = {"errInit": head[0:1].copy(), "errStep": head[1:2].copy(), "logErrors": head[2:3].copy()}
    off, keep = 12, set(OUTPUTS) - set(refused(case))
    for pool, name, kind, shape in FIELDS:
        s = field_shape(case, shape) if kind != "d" else ()
        dtype = np.dtype("<i4") if kind in "id" else _real(case)
        n = int(np.prod(s))
        if (pool, name) in keep:
            out[f"{pool}/{name}"] = np.frombuffer(raw, dtype, n, off).reshape(s).astype(dtype.newbyteorder("="))
        off += dtype.itemsize * n
    if off != len(raw):
        raise ValueError(f"oracle output: {len(raw) - off} trailing bytes")
    return out


def load_fixture(precision: str = "r4") -> dict[str, dict[str, np.ndarray]]:
    """The committed oracle outputs at ``precision``, keyed by case then ``pool/field``."""
    with np.load(FIXTURE_DIR / f"oracle-{precision}.npz") as data:
        out: dict[str, dict[str, np.ndarray]] = {}
        for key in data.files:
            case, field = key.split("/", 1)
            out.setdefault(case, {})[field] = data[key]
    return out


def word_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    """Words that differ as bit patterns (so -0.0 != 0.0)."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape or a.dtype.itemsize != b.dtype.itemsize:
        return int(max(a.size, b.size))
    word = np.dtype(f"u{a.dtype.itemsize}")
    return int(np.count_nonzero(a.view(word) != b.view(word)))


def run_gpu(case: dict) -> dict[str, np.ndarray]:
    """Run the GPU port on ``case`` as run_time_integration_fe.F90 drives the
    Fortran: a restart's thermal init, then
    :func:`woof.hex.mpas_ice.time_integration_fe_v842.forward_euler_step`.
    The same ``pool/name`` outputs, without the error words (the port raises
    instead)."""
    import cupy as cp
    from woof.hex.mpas_ice import thermal_init_v842, time_integration_fe_v842 as fe

    c = case["config"]
    dev = {pool: {name: cp.asarray(np.ascontiguousarray(value).reshape(np.shape(value) or (1,)))
                  for name, value in case["pools"][pool].items()} for pool in POOLS}
    for pool, name in refused(case):
        del dev[pool][name]
    mesh = dict(dev["mesh"], nCellsSolve=case["dims"]["nCellsSolve"], nEdgesSolve=case["dims"]["nEdgesSolve"])
    t, g = dev["thermal"], dev["geometry"]
    # li_thermal_init on a restart: with 'lapse', the surface air temperature
    thermal_init_v842.thermal_init(
        thermal=t, thickness=g["thickness"], upper_surface=g["upperSurface"],
        layer_center_sigma=mesh["layerCenterSigma"], n_cells_solve=case["dims"]["nCellsSolve"],
        config_ice_density=c["config_ice_density"], config_temperature_init=c["config_temperature_init"],
        config_surface_air_temperature_source=c["config_surface_air_temperature_source"],
        config_surface_air_temperature_value=c["config_surface_air_temperature_value"],
        config_surface_air_temperature_lapse_rate=c["config_surface_air_temperature_lapse_rate"],
        config_basal_heat_flux_source=c["config_basal_heat_flux_source"],
        config_basal_heat_flux_value=c["config_basal_heat_flux_value"], config_do_restart=True)
    now, ring = case["clock"]
    fe.forward_euler_step(mesh=mesh, geometry=g, thermal=t, velocity=dev["velocity"], hydro=dev["hydro"], config=c,
                          seconds_to_next_force_time=ring - now)
    out = {}
    for pool, name in OUTPUTS:
        if (pool, name) in refused(case):
            continue
        source = mesh if pool == "mesh" else dev[pool]
        value = cp.asnumpy(source[name])
        out[f"{pool}/{name}"] = value.reshape(np.shape(case["pools"][pool][name]))
    return out


def measure(reference: dict | None = None, precision: str = "r4") -> dict[str, dict[str, int]]:
    """Per case and field, the GPU words at ``precision`` that differ from the oracle's."""
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision):
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: word_mismatches(got[name], want[name]) for name in got}
    return report
