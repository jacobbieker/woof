"""Cases and the stream layout for the MALI forward-mode oracle.

``tools/mali_oracle/build_forward.sh`` compiles MALI v8.4.2's land-ice core
(``mpas_li_core.F``, ``mpas_li_time_integration.F`` and every module they
reach) with v8.4.2's own ``mpas_timekeeping.F`` and ESMF time library,
verbatim, with gfortran -O0 -ffp-contract=off at RKIND 4 and 8, and
``run_forward.F90`` runs a whole cold start: ``li_simulation_clock_init``,
``li_core_init`` and ``li_core_run`` to the clock's stop time. Its input
stream is self-describing (every config by name, every field by pool and
name, text fields and second time levels included), so this module owns the
field table: :data:`FIELDS` names each field's pool, type and shape, and a
case is one dict per pool plus the configs.

The cases are the MALI test cases' physics on the periodic hexagonal meshes
of :mod:`woof.verify.mali_hexmesh`, each on a uniform and a variable
resolution mesh: the dome (fixed and adaptive time steps, and with calving of
its floating margin), EISMINT2 experiment A with the temperature and the
enthalpy solver, the radial subglacial hydrology test (and a marine
variant, where grounded ice lies below sea level), and the enthalpy
benchmark. Every word is formed from correctly rounded float64 operations
(and numpy's legacy ``RandomState`` with ``random_sample`` only, for the
prior words of the diagnostics) and rounded once to the case's precision, so
the bytes do not depend on the platform. ``input_digest`` is pinned beside
the fixture.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from woof.hex.mpas_ice import analysis_v842
from woof.verify import mali_hexmesh
from woof.verify import mali_time_integration_fe_oracle as fe_oracle

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mali_forward"
PRECISIONS = {"r4": np.float32, "r8": np.float64}
STRKIND = 512
POOLS = ("mesh", "geometry", "thermal", "velocity", "hydro", "scratch")
HYDRO_CELL, HYDRO_EDGE, HYDRO_SCALAR = fe_oracle.HYDRO_CELL, fe_oracle.HYDRO_EDGE, fe_oracle.HYDRO_SCALAR
SECONDS_PER_YEAR = 365.0 * 86400.0

_C, _E, _V, _L, _I = "nCells", "nEdges", "nVertices", "nVertLevels", "nVertInterfaces"
#: the forward mode's fields beyond the time step's (pool, name, type, shape):
#: 'c' is text (StrKIND characters), and NAME@2 a field's second time level
_FORWARD_FIELDS = (
    ("mesh", "xtime", "c", ()), ("mesh", "simulationStartTime", "c", ()), ("mesh", "timestepNumber", "i", ()),
    ("mesh", "cellProcID", "i", (_C,)), ("mesh", "verticesOnCell", "i", (_C, "maxEdges")),
    ("mesh", "yCell", "r", (_C,)), ("mesh", "zCell", "r", (_C,)),
    ("mesh", "xVertex", "r", (_V,)), ("mesh", "yVertex", "r", (_V,)), ("mesh", "zVertex", "r", (_V,)),
    ("geometry", "vertexMask@2", "i", (_V,)),
    ("scratch", "vertexIndices", "i", (_V,)),
)
FIELDS = fe_oracle.FIELDS + _FORWARD_FIELDS
#: what the 'simple' velocity solver adds (its cases only, so the other cases'
#: streams are unchanged): the edge points, the prescribed velocity, and the
#: higher-order package's Dirichlet mask, which 'simple' allocates but never
#: reads (carried, and compared, word for word)
SIMPLE_FIELDS = (
    ("mesh", "xEdge", "r", (_E,)), ("mesh", "yEdge", "r", (_E,)),
    ("velocity", "normalVelocityInitial", "r", (_E, _I)),
    ("velocity", "dirichletVelocityMask", "i", (_C, _I)),
)


#: what li_compute_statistics adds (its cases only): its edge scratch mask
STATS_FIELDS = (("scratch", "iceEdgeMask", "i", (_E,)),)


#: what the analysis members add (their cases only): both members' pools, the
#: regions and their dimensions
AM_FIELDS = (("mesh", "nRegions", "d", ()), ("mesh", "nRegionGroups", "d", ()),
             *(("globalStatsAM", name, "r", ()) for name in analysis_v842.GLOBAL_FIELDS),
             *(("regionalStatsAM", name, "r", ("nRegions",)) for name in analysis_v842.REGIONAL_FIELDS),
             ("regions", "regionCellMasks", "i", (_C, "nRegions")))
AM_POOLS = ("globalStatsAM", "regionalStatsAM", "regions")


def am_on(config: dict) -> bool:
    return any(k.startswith("config_AM_") for k in config)


def streams_of(config: dict) -> dict[str, str] | None:
    """The streams a case declares ('stream:NAME' configs: their reference times)."""
    return {k[7:]: v for k, v in config.items() if k.startswith("stream:")} or None


def stats_on(config: dict) -> bool:
    return int(config.get("config_stats_interval", 0)) > 0 or bool(config.get("config_write_stats_on_startup", False))


def fields_for(config: dict) -> tuple:
    """The field table of a run with ``config``."""
    return (FIELDS + (SIMPLE_FIELDS if config.get("config_velocity_solver") == "simple" else ())
            + (STATS_FIELDS if stats_on(config) else ()) + (AM_FIELDS if am_on(config) else ()))


def fields(case: dict) -> tuple:
    return fields_for(case["config"])


def outputs(case: dict) -> tuple:
    """Every word compared for ``case``: :data:`OUTPUTS`, plus 'simple''s."""
    return tuple((pool, name) for pool, name, kind, _ in fields(case)
                 if pool not in ("scratch", "regions") and kind != "d")

#: the mesh's own words: the run must leave them as they are (compared too)
MESH_INPUTS = ("nEdgesOnCell", "cellsOnCell", "cellsOnVertex", "cellsOnEdge", "edgesOnCell", "verticesOnEdge",
               "nEdgesOnEdge", "edgesOnEdge", "indexToCellID", "indexToEdgeID", "dvEdge", "dcEdge", "areaCell",
               "xCell", "yCell", "zCell", "xVertex", "yVertex", "zVertex", "latCell", "lonCell",
               "kiteAreasOnVertex", "weightsOnEdge", "edgeNormalVectors", "coeffs_reconstruct", "verticesOnCell",
               "xEdge", "yEdge")
#: every word compared (the cases without 'simple'): every field but the dimensions and the scratch
OUTPUTS = tuple((pool, name) for pool, name, kind, _ in FIELDS if pool != "scratch" and kind != "d")
#: refused at RKIND 8 (glibc's binary64 atan): not in the r8 fixture
R8_REFUSED = (("velocity", "eTheta"),)

#: the time step's configs as the forward mode starts them: a cold start
BASE_CONFIG = dict(
    fe_oracle.BASE_CONFIG,
    config_do_restart=False, config_temperature_init="linear", config_surface_air_temperature_source="constant",
    config_basal_heat_flux_source="constant", config_basal_heat_flux_value=0.042,
    config_surface_air_temperature_value=253.15, config_flowParamA_calculation="constant",
    config_thermal_solver="none", config_tracer_advection="none",
    config_dt="0000-00-30_00:00:00", config_start_time="0001-01-01_00:00:00",
    config_run_duration="0000-00-120_00:00:00", config_stop_time="none",
    config_restart_timestamp_name="restart_timestamp",
    config_adaptive_timestep_force_interval="1000-00-00_00:00:00", config_year_digits=4,
    config_calendar_type="gregorian_noleap", config_time_integration="forward_euler", config_stats_interval=0,
    config_write_output_on_startup=True, config_write_stats_on_startup=False,
    config_write_albany_ascii_mesh=False, config_num_halos=2, config_SGH_till_max=0.1, sphere_radius=0.0,
)

_DOME = {"_physics": "dome"}


def _am(member: str, interval: str, on_startup: bool) -> dict:
    """An enabled analysis member's configs."""
    return {f"config_AM_{member}_enable": True, f"config_AM_{member}_compute_interval": interval,
            f"config_AM_{member}_stream_name": f"{member}Output",
            f"config_AM_{member}_compute_on_startup": on_startup, f"config_AM_{member}_write_on_startup": on_startup}

#: (name, nVertLevels, overrides of BASE_CONFIG); '_' keys shape the fields
CASES = (
    ("dome", 5, dict(_DOME)),
    ("dome_adaptive", 5, dict(_DOME, config_adaptive_timestep=True, config_adaptive_timestep_CFL_fraction=0.004,
                              config_adaptive_timestep_force_interval="0000-00-20_00:00:00",
                              config_max_adaptive_timestep=8.0 * 86400.0,
                              config_run_duration="0000-00-40_00:00:00",
                              config_sia_tangent_slope_calculation="from_vertex_barycentric_kiteareas")),
    ("dome_calving", 5, dict(_DOME, _bed="marine", config_calving="floating", config_dt="10_00:00:00",
                             config_run_duration="40_00:00:00",
                             config_sia_tangent_slope_calculation="from_normal_slope")),
    ("eismint2_a_temperature", 10, {"_physics": "eismint2", "config_thermal_solver": "temperature",
                                    "config_flowParamA_calculation": "PB1982",
                                    "config_surface_air_temperature_source": "file",
                                    "config_calendar_type": "gregorian", "config_start_time": "2000-02-20_00:00:00",
                                    "config_dt": "0000-00-05_00:00:00", "config_run_duration": "20_00:00:00"}),
    ("eismint2_a_enthalpy", 10, {"_physics": "eismint2", "config_thermal_solver": "enthalpy",
                                 "config_flowParamA_calculation": "CP2010",
                                 "config_surface_air_temperature_source": "file"}),
    ("hydro_radial", 5, {"_physics": "hydro", "config_SGH": True, "config_velocity_solver": "none",
                         "config_thickness_advection": "none", "config_dt": "06:00:00",
                         "config_run_duration": "0000-00-01_00:00:00", "config_SGH_englacial_porosity": 1.0,
                         "config_SGH_conduc_coeff": 1.0e-4}),
    ("hydro_marine", 5, {"_physics": "hydro", "_bed": "marine", "config_SGH": True,
                         "config_velocity_solver": "none", "config_thickness_advection": "none",
                         "config_dt": "06:00:00", "config_run_duration": "0000-00-00_12:00:00",
                         "config_SGH_englacial_porosity": 1.0, "config_SGH_conduc_coeff": 1.0e-4}),
    # the 'simple' velocity solver: a prescribed field, with the thermal solver; radial on the
    # marine bed with the circular-shelf test's spike ('floating') and rim ('topographic_threshold')
    ("simple_uniform", 5, dict(_DOME, config_velocity_solver="simple", config_simple_velocity_type="uniform",
                               config_stats_cell_ID=1, config_thermal_solver="temperature")),
    ("simple_radial_shelf", 5, dict(_DOME, _bed="marine", config_velocity_solver="simple",
                                    config_simple_velocity_type="radial", config_stats_cell_ID=1,
                                    config_calving="floating", config_dt="10_00:00:00",
                                    config_run_duration="40_00:00:00")),
    ("simple_rim", 5, dict(_DOME, _bed="marine", config_velocity_solver="simple",
                           config_simple_velocity_type="uniform", config_stats_cell_ID=1,
                           config_calving="topographic_threshold", config_dt="10_00:00:00",
                           config_run_duration="40_00:00:00")),
    # li_compute_statistics: on startup and every step or every other, the diagnostic column at
    # the dome's summit ('_stats_cell'), on no cell (ID 0), with the thermal solver, the
    # 'simple' solver's signed velocities and the zero velocity of 'none'
    ("stats_dome", 5, dict(_DOME, _stats_cell="centre", config_stats_interval=2,
                           config_write_stats_on_startup=True)),
    ("stats_eismint2", 10, {"_physics": "eismint2", "config_thermal_solver": "temperature",
                            "config_flowParamA_calculation": "PB1982",
                            "config_surface_air_temperature_source": "file", "config_dt": "0000-00-10_00:00:00",
                            "config_run_duration": "30_00:00:00", "config_stats_interval": 1,
                            "config_stats_cell_ID": 0}),
    ("stats_simple_shelf", 5, dict(_DOME, _bed="marine", _stats_cell="centre", config_velocity_solver="simple",
                                   config_simple_velocity_type="radial", config_calving="floating",
                                   config_dt="10_00:00:00", config_run_duration="30_00:00:00",
                                   config_stats_interval=1, config_write_stats_on_startup=True)),
    ("stats_benchmark", 10, {"_physics": "benchmark", "_stats_cell": "centre", "config_thermal_solver": "enthalpy",
                             "config_velocity_solver": "none", "config_thickness_advection": "none",
                             "config_surface_air_temperature_value": 243.15,
                             "config_dt": "0001-00-00_00:00:00", "config_run_duration": "0002-00-00_00:00:00",
                             "config_stats_interval": 2, "config_write_stats_on_startup": True}),
    # the analysis members: globalStats every 70 days on a stream whose reference time is 15 days
    # in (computes at days 30 and 90, and on startup), regionalStats every step from startup off,
    # over an inner disc, a half plane and no cell; on the marine bed with its grounding line,
    # globalStats at 'output_interval' (on startup only) and regionalStats every 20 days, a region
    # weighted 2
    ("am_dome", 5, dict(_DOME, _region3="empty", **_am("globalStats", "0000-00-70_00:00:00", True),
                        **_am("regionalStats", "dt", False),
                        **{"stream:globalStatsOutput": "0001-01-16_00:00:00",
                           "stream:regionalStatsOutput": "0001-01-01_00:00:00"})),
    ("am_marine", 5, dict(_DOME, _bed="marine", _region3="double", config_calving="floating",
                          config_dt="10_00:00:00", config_run_duration="40_00:00:00",
                          config_sia_tangent_slope_calculation="from_normal_slope",
                          **_am("globalStats", "output_interval", True),
                          **_am("regionalStats", "0000-00-20_00:00:00", True),
                          **{"stream:globalStatsOutput": "0001-01-01_00:00:00",
                             "stream:regionalStatsOutput": "0000-12-27_00:00:00"})),
    ("enthalpy_benchmark", 10, {"_physics": "benchmark", "config_thermal_solver": "enthalpy",
                                "config_velocity_solver": "none", "config_thickness_advection": "none",
                                "config_surface_air_temperature_value": 243.15,
                                "config_dt": "0001-00-00_00:00:00", "config_run_duration": "0004-00-00_00:00:00"}),
)
#: (suffix, stretch): every case on the uniform and the variable resolution mesh
MESHES = (("uniform", 0.0), ("varres", 1.2))
MESH_N, MESH_DC = 20, 4000.0


@lru_cache(maxsize=None)
def mesh_of(stretch: float) -> dict:
    return mali_hexmesh.torus(MESH_N, MESH_DC, stretch)


@lru_cache(maxsize=None)
def edge_points_of(stretch: float) -> tuple[np.ndarray, np.ndarray]:
    return mali_hexmesh.edge_points(MESH_N, MESH_DC, stretch)


def field_shape(case: dict, shape: tuple) -> tuple[int, ...]:
    return tuple(case["dims"][s] if isinstance(s, str) else s for s in shape)


def _dome(d: np.ndarray, h0: float, r0: float) -> np.ndarray:
    inside = d < r0
    q = np.where(inside, 1.0 - (d / r0) * (d / r0), 0.0)
    return np.where(inside, h0 * np.sqrt(q), 0.0)


def make_case(name: str, n_levels: int, overrides: dict, stretch: float, suffix: str,
              precision: str = "r4") -> dict:
    """One case on the ``stretch`` mesh at ``precision`` (see the module docstring)."""
    real = PRECISIONS[precision]
    mesh = mesh_of(stretch)
    nc, ne, nv, nl = mesh["nCells"], mesh["nEdges"], mesh["nVertices"], n_levels
    config = dict(BASE_CONFIG, **{k: v for k, v in overrides.items() if not k.startswith("_")})
    physics, bed_kind = overrides["_physics"], overrides.get("_bed", "flat")
    dims = {"nCells": nc, "nEdges": ne, "nVertices": nv, "nVertLevels": nl, "nVertInterfaces": nl + 1,
            "nCellsSolve": nc, "nEdgesSolve": ne, "vertexDegree": 3, "maxEdges": mesh["maxEdges"],
            "maxEdges2": mesh["maxEdges2"]}
    rs = np.random.RandomState(20261121 + 7 * len(name) + int(stretch * 10))

    def r(values) -> np.ndarray:
        return np.asarray(values, np.float64).astype(real)

    def prior(shape) -> np.ndarray:
        return r(-1.0 + 2.0 * rs.random_sample(shape))

    x0, y0 = mali_hexmesh.centre(mesh)
    d = mali_hexmesh.distance(mesh, mesh["xCell"], mesh["yCell"], x0, y0)
    if physics == "benchmark":
        thickness = np.full(nc, 1000.0)
    elif physics == "hydro":
        thickness = _dome(d, 800.0, 30000.0)
    else:
        thickness = _dome(d, 900.0 if physics == "dome" else 600.0, 28000.0)
    bed = np.zeros(nc) if bed_kind == "flat" else 300.0 - 700.0 * (d / 28000.0)
    smb = np.zeros(nc)
    air = np.full(nc, config["config_surface_air_temperature_value"])
    if physics == "eismint2":
        # EISMINT2 A scaled to the mesh: M = min(0.5, S_b (R_el - d)) m/yr, T = T_min + S_T d
        smb = np.minimum(0.5, 2.0e-5 * (30000.0 - d)) * config["config_ice_density"] / SECONDS_PER_YEAR
        air = 238.15 + 1.67e-4 * d
    fractions = np.full(nl, 1.0 / nl)
    speed = np.where(d < 30000.0, 30.0 / SECONDS_PER_YEAR * d / 30000.0, 0.0)
    safe = np.where(d > 0.0, d, 1.0)
    ux = np.zeros(nc)
    uy = np.zeros(nc)
    if physics == "hydro":
        dx = mesh["xCell"] - x0
        dy = mesh["yCell"] - y0
        ux, uy = speed * dx / safe, speed * dy / safe
    pools = {
        "mesh": {
            **{k: np.asarray(mesh[k]).astype(np.int32) for k in ("nEdgesOnCell", "cellsOnCell", "cellsOnVertex",
                                                                "cellsOnEdge", "edgesOnCell", "verticesOnEdge",
                                                                "nEdgesOnEdge", "edgesOnEdge", "verticesOnCell")},
            "edgeSignOnCell": np.zeros((nc, dims["maxEdges"]), np.int32),
            "baryCellsOnVertex": np.asarray(mesh["cellsOnVertex"], np.int32).copy(),
            "indexToCellID": np.arange(1, nc + 1, dtype=np.int32),
            "indexToEdgeID": np.arange(1, ne + 1, dtype=np.int32),
            "layerThicknessFractions": r(fractions), "layerCenterSigma": prior(nl),
            "layerInterfaceSigma": prior(nl + 1),
            **{k: r(mesh[k]) for k in ("dvEdge", "dcEdge", "areaCell", "xCell", "yCell", "zCell", "xVertex",
                                       "yVertex", "zVertex", "kiteAreasOnVertex", "weightsOnEdge",
                                       "edgeNormalVectors", "coeffs_reconstruct")},
            "latCell": r(np.zeros(nc)), "lonCell": r(np.zeros(nc)),
            "baryWeightsOnVertex": prior((nv, 3)),
            "deltat": prior(()), "daysSinceStart": prior(()), "allowableDtACFL": prior(()),
            "allowableDtDCFL": prior(()),
            "xtime": "no_date_available", "simulationStartTime": "no_date_available",
            "timestepNumber": np.array(-3, np.int32), "cellProcID": np.full(nc, 7, np.int32),
        },
        "geometry": {
            "thickness": r(thickness), "thicknessOld": prior(nc), "dHdt": prior(nc), "bedTopography": r(bed),
            "upperSurface": prior(nc), "lowerSurface": prior(nc), "sfcMassBal": r(smb),
            "basalMassBal": prior(nc), "groundedBasalMassBal": r(np.zeros(nc)),
            "floatingBasalMassBal": r(np.zeros(nc)), "dynamicThickening": prior(nc),
            "calvingThickness": r(np.zeros(nc)), "basalWaterThickness": r(np.zeros(nc)),
            "apparentDiffusivity": prior(nc), "restoreThickness": r(thickness),
            "eigencalvingParameter": r(np.zeros(nc)), "requiredCalvingVolumeRate": prior(nc),
            "calvingVelocity": prior(nc), "uncalvedVolume": prior(nc),
            "normalSlopeEdge": prior(ne), "slopeEdge": prior(ne), "tangentSlopeEdge": prior(ne),
            "upperSurfaceVertex": prior(nv), "layerThickness": prior((nc, nl)),
            "layerThicknessEdge": prior((ne, nl)),
            "cellMask": (rs.random_sample(nc) * 512).astype(np.int32),
            "vertexMask": (rs.random_sample(nv) * 512).astype(np.int32),
            "edgeMask": (rs.random_sample(ne) * 512).astype(np.int32),
            "vertexMask@2": (rs.random_sample(nv) * 512).astype(np.int32),
        },
        "velocity": {
            "normalVelocity": prior((ne, nl + 1)), "layerNormalVelocity": prior((ne, nl)),
            "uReconstructX": r(np.repeat(ux[:, None], nl + 1, 1)),
            "uReconstructY": r(np.repeat(uy[:, None], nl + 1, 1)),
            **{k: prior((nc, nl + 1)) for k in ("uReconstructZ", "uReconstructZonal", "uReconstructMeridional")},
            "flowParamA": prior((nc, nl)), "surfaceSpeed": prior(nc), "basalSpeed": prior(nc),
            "betaSolve": r(np.zeros(nc)),
            **{k: prior(nc) for k in ("exx", "eyy", "exy", "eyx", "eTheta", "eMax", "eMin")},
        },
        "thermal": {
            "temperature": r(np.full((nc, nl), 263.15)), "waterfrac": r(np.zeros((nc, nl))),
            # the enthalpy is a state the cold start reads as it is (li_thermal_init sets the
            # temperature only): the column's at 263.15 K, rhoi cp_ice T
            "enthalpy": r(np.full((nc, nl), config["config_ice_density"] * 2009.0 * 263.15)), "heatDissipation": r(np.zeros((nc, nl))),
            "pmpTemperature": prior((nc, nl)), "surfaceTemperature": prior(nc), "basalTemperature": prior(nc),
            "surfaceAirTemperature": r(air), "basalHeatFlux": prior(nc), "basalFrictionFlux": r(np.zeros(nc)),
            "surfaceConductiveFlux": prior(nc), "basalConductiveFlux": prior(nc), "basalPmpTemperature": prior(nc),
        },
    }
    overburden = config["config_ice_density"] * 9.80616 * thickness
    hydro = {k: prior(nc) for k in HYDRO_CELL}
    hydro.update({
        "basalMeltInput": r(np.where(thickness > 0.0, 0.05 * 1000.0 / SECONDS_PER_YEAR, 0.0)),
        "externalWaterInput": r(np.zeros(nc)), "waterThickness": r(np.where(d < 20000.0, 0.05, -0.01)),
        "tillWaterThickness": r(np.where(d < 15000.0, 0.2, -0.02)),
        "waterPressure": r(np.where(d < 25000.0, 0.6 * overburden, -1.0e3)),
        "waterPressureOld": prior(nc), "effectivePressure": r(0.4 * overburden)})
    hydro["hydropotentialBaseVertex"] = prior(nv)
    hydro.update({k: prior(ne) for k in HYDRO_EDGE})
    hydro.update({"channelArea": r(np.zeros(ne)), "flowParamAChannel": prior(ne),
                  "channelEffectivePressure": prior(ne)})
    hydro["waterFluxMask"] = np.zeros(ne, np.int32)
    hydro.update({k: prior(()) for k in HYDRO_SCALAR})
    pools["hydro"] = hydro
    case = {"name": f"{name}_{suffix}", "precision": precision, "dims": dims, "config": config, "pools": pools}
    scratch = {}
    for pool, field, kind, shape in FIELDS:
        if pool == "scratch":
            s = field_shape(case, shape)
            scratch[field] = ((rs.random_sample(s) * 11).astype(np.int32) - 5 if kind == "i"
                              else r(-1.0e3 + 2.0e3 * rs.random_sample(s)))
    pools["scratch"] = scratch
    if overrides.get("_stats_cell") == "centre":
        config["config_stats_cell_ID"] = int(np.argmin(d)) + 1
    if config["config_velocity_solver"] == "simple":
        # drawn last, so the other cases' words are as they were
        x_edge, y_edge = edge_points_of(stretch)
        pools["mesh"].update(xEdge=r(x_edge), yEdge=r(y_edge))
        pools["velocity"]["normalVelocityInitial"] = prior((ne, nl + 1))
        pools["velocity"]["dirichletVelocityMask"] = (rs.random_sample((nc, nl + 1)) < 0.5).astype(np.int32)
    if stats_on(config):
        scratch["iceEdgeMask"] = (rs.random_sample(ne) * 11).astype(np.int32) - 5
    if am_on(config):
        dims.update(nRegions=3, nRegionGroups=1)
        regions = np.zeros((nc, 3), np.int32)
        regions[:, 0] = d < 15000.0
        regions[:, 1] = np.asarray(mesh["xCell"]) > x0
        if overrides["_region3"] == "double":
            regions[:, 2] = 2 * (d < 10000.0)
        pools["regions"] = {"regionCellMasks": regions}
        pools["globalStatsAM"] = {k: prior(()) for k in analysis_v842.GLOBAL_FIELDS}
        pools["regionalStatsAM"] = {k: prior(3) for k in analysis_v842.REGIONAL_FIELDS}
    return case


def cases(precision: str = "r4") -> list[dict]:
    """Every case on both meshes at 'r4' or 'r8'."""
    return [make_case(name, nl, overrides, stretch, suffix, precision)
            for name, nl, overrides in CASES for suffix, stretch in MESHES]


def refused(case: dict) -> tuple:
    return R8_REFUSED if case["precision"] == "r8" else ()


def _real(case: dict) -> np.dtype:
    return np.dtype(PRECISIONS[case["precision"]]).newbyteorder("<")


def _text(value: str, width: int) -> bytes:
    raw = value.encode("ascii")
    if len(raw) > width:
        raise ValueError(f"{value!r} is longer than {width} characters")
    return raw.ljust(width, b" ")


def input_bytes(case: dict) -> bytes:
    """run_forward.F90's input stream for ``case``."""
    real = _real(case)
    config = case["config"]
    table = fields(case)
    parts = [np.array([len(config), len(table)], "<i4").tobytes()]
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
    for pool, name, kind, shape in table:
        s = field_shape(case, shape)
        code = {"r": 1, "i": 2, "d": 3, "c": 4}[kind]
        dims = list(reversed(s)) + [1] * (3 - len(s))
        parts += [_text(pool, 16), _text(name, 64), np.array([code, len(s), *dims], "<i4").tobytes()]
        if kind == "d":
            parts.append(np.array(case["dims"][name], "<i4").tobytes())
        elif kind == "c":
            parts.append(_text(case["pools"][pool][name], STRKIND))
        else:
            value = case["pools"][pool][name]
            if tuple(np.shape(value)) != s:
                raise ValueError(f"{pool}/{name} has shape {np.shape(value)}, not {s}")
            parts.append(np.ascontiguousarray(value, real if kind == "r" else "<i4").tobytes())
    return b"".join(parts)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    """run_forward.F90's output stream -> ``pool/name`` arrays (:data:`OUTPUTS`
    without the RKIND 8 refusals; text fields as 0-d unicode arrays), the
    error words and the clock's final time (``clock/now``); with the
    statistics on, li_compute_statistics's reduction words (``stats/real``,
    ``stats/int``) and its reports (``stats/log``, :func:`stats_log`)."""
    head = np.frombuffer(raw, "<i4", 4, 0)
    out = {"errClock": head[0:1].copy(), "errInit": head[1:2].copy(), "errRun": head[2:3].copy(),
           "logErrors": head[3:4].copy(),
           "clock/now": np.array(raw[16:16 + STRKIND].decode("ascii").rstrip())}
    off, keep = 16 + STRKIND, set(outputs(case)) - set(refused(case))
    for pool, name, kind, shape in fields(case):
        if kind == "c":
            if (pool, name) in keep:
                out[f"{pool}/{name}"] = np.array(raw[off:off + STRKIND].decode("ascii").rstrip())
            off += STRKIND
            continue
        s = field_shape(case, shape) if kind != "d" else ()
        dtype = np.dtype("<i4") if kind in "id" else _real(case)
        n = int(np.prod(s))
        if (pool, name) in keep:
            out[f"{pool}/{name}"] = np.frombuffer(raw, dtype, n, off).reshape(s).astype(dtype.newbyteorder("="))
        off += dtype.itemsize * n
    n_real, n_int, n_lines = (int(n) for n in np.frombuffer(raw, "<i4", 3, off))
    off += 12
    real_words = np.frombuffer(raw, _real(case), n_real, off).astype(_real(case).newbyteorder("="))
    off += _real(case).itemsize * int(n_real)
    int_words = np.frombuffer(raw, "<i4", n_int, off).astype(np.int32)
    off += 4 * int(n_int)
    lines = [raw[off + i * STRKIND:off + (i + 1) * STRKIND].decode("ascii") for i in range(int(n_lines))]
    off += STRKIND * int(n_lines)
    n_computes, n_words = (int(n) for n in np.frombuffer(raw, "<i4", 2, off))
    off += 8
    members = np.frombuffer(raw, "<i4", n_computes, off).astype(np.int32)
    off += 4 * n_computes
    times = [raw[off + i * STRKIND:off + (i + 1) * STRKIND].decode("ascii").rstrip() for i in range(n_computes)]
    off += STRKIND * n_computes
    am_words = np.frombuffer(raw, _real(case), n_words, off).astype(_real(case).newbyteorder("="))
    off += _real(case).itemsize * n_words
    if off != len(raw):
        raise ValueError(f"oracle output: {len(raw) - off} trailing bytes")
    if stats_on(case["config"]):
        out["stats/real"], out["stats/int"] = real_words, int_words
        out["stats/log"] = np.array(stats_log(lines))
    elif n_real or n_int:
        raise ValueError("oracle output: statistics words from a run without statistics")
    if am_on(case["config"]):
        out["am/member"], out["am/xtime"], out["am/words"] = members, np.array(times), am_words
    elif n_computes:
        raise ValueError("oracle output: analysis-member computes from a run without them")
    # The driver writes back the arrays it put, but the stub's
    # mpas_pool_shift_time_levels trades them between NAME and NAME@2 (once in
    # the initial solve, once a step): after an odd number of shifts the array
    # put as level 1 holds level 2. Label the words by time level, as MPAS's
    # pool (and the port) does.
    if "mesh/timestepNumber" in out and (1 + int(out["mesh/timestepNumber"][()])) % 2 == 1:
        for key in [k for k in out if k.endswith("@2")]:
            out[key], out[key[:-2]] = out[key[:-2]], out[key]
    return out


def stats_log(lines) -> str:
    """Every li_compute_statistics report in a log, in order, one blank
    line between."""
    from woof.hex.mpas_ice import statistics_v842
    return "\n\n".join(statistics_v842.reports(lines))


def load_fixture(precision: str = "r4") -> dict[str, dict[str, np.ndarray]]:
    """The committed oracle outputs at ``precision``, keyed by case then ``pool/field``."""
    with np.load(FIXTURE_DIR / f"oracle-{precision}.npz") as data:
        out: dict[str, dict[str, np.ndarray]] = {}
        for key in data.files:
            case, field = key.split("/", 1)
            out.setdefault(case, {})[field] = data[key]
    return out


def word_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    """Words that differ as bit patterns (so -0.0 != 0.0); text compares as text."""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "U" or b.dtype.kind == "U":
        return int(str(a) != str(b))
    return fe_oracle.word_mismatches(a, b)


def device_pools(case: dict) -> dict:
    """The case's pools as the port takes them: device arrays, host text and integers."""
    import cupy as cp
    pools = {}
    for pool in POOLS + (AM_POOLS if am_on(case["config"]) else ()):
        pools[pool] = {}
        for name, value in case["pools"][pool].items():
            if isinstance(value, str):
                pools[pool][name] = value
            else:
                pools[pool][name] = cp.asarray(np.ascontiguousarray(value).reshape(np.shape(value) or (1,)))
    for pool, name in refused(case):
        del pools[pool][name]
    return pools


def outputs_of(case: dict, pools: dict, clock) -> dict[str, np.ndarray]:
    """The port's pools as the oracle's ``pool/name`` outputs."""
    import cupy as cp
    out = {"clock/now": np.array(clock.time_stamp)}
    for pool, name in outputs(case):
        if (pool, name) in refused(case):
            continue
        value = pools[pool][name]
        if isinstance(value, str):
            out[f"{pool}/{name}"] = np.array(value)
        else:
            out[f"{pool}/{name}"] = cp.asnumpy(value).reshape(np.shape(case["pools"][pool][name]))
    return out


def run_gpu(case: dict, config: dict | None = None) -> dict[str, np.ndarray]:
    """Run the GPU port's forward mode on ``case`` as run_forward.F90 drives the
    Fortran (:func:`woof.hex.mpas_ice.forward_v842.run`): the same
    ``pool/name`` outputs and ``clock/now``, without the error words (the
    port raises instead)."""
    from woof.hex.mpas_ice import forward_v842

    pools = device_pools(case)
    stats: list = []
    computes: list = []
    run_config = {k: v for k, v in dict(case["config"], **(config or {})).items() if not k.startswith("stream:")}
    clock = forward_v842.run(pools=pools, config=run_config, on_statistics=stats.append, on_analysis=computes.append,
                             streams=streams_of(case["config"]))
    out = outputs_of(case, pools, clock)
    if am_on(case["config"]):
        out["am/member"] = np.array([1 + analysis_v842.MEMBERS.index(c.member) for c in computes], np.int32)
        out["am/xtime"] = np.array([c.xtime for c in computes])
        out["am/words"] = (np.concatenate([c.words for c in computes]) if computes
                           else np.zeros(0, PRECISIONS[case["precision"]]))
    if stats_on(case["config"]):
        out["stats/real"] = np.concatenate([s.real_words for s in stats])
        out["stats/int"] = np.concatenate([s.int_words for s in stats])
        out["stats/log"] = np.array(stats_log([line for s in stats for line in s.lines]))
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
