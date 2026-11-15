"""Cases and the stream layout for the MPAS-Seaice column-package oracle.

``tools/mpas_seaice_oracle/run_column.F90`` runs MPAS-Model v8.4.2's
``seaice_init_column_physics_package_parameters`` on one block, then the
stage a case names (``oracle_stage``):

- ``init``: ``initialize_coupler_fields``,
  ``seaice_init_column_physics_package_variables``, ``init_ice_state`` and
  ``seaice_reset_coupler_fluxes``, in ``seaice_init``'s order;
- ``aggregate``: ``seaice_column_aggregate``;
- ``update_state``: ``seaice_column_update_state`` (``oracle_update_type``,
  the stream's dt, ``oracle_ice_age_time_offset``);
- ``vertical_thermodynamics``: ``colpkg_init_thermo`` (the column package's
  ``heat_capacity`` and ``l_brine``, as seaice_init sets them), then
  ``column_vertical_thermodynamics`` at the clock's start
  (``config_start_time``), its ierr logged as ``TOBJ IERR``. Its cases carry
  a physically consistent BL99 column state (:func:`_thermo_state`).

``mpas_seaice_column.F``, ``mpas_seaice_initialize.F``'s and
``mpas_seaice_forcing.F``'s routines and the column package are compiled
verbatim with gfortran -O0 -ffp-contract=off at RKIND 8 (the column package
is dbl_kind; the wrappers do not compile at RKIND 4). The stream is
:mod:`woof.verify.mpas_ocean_stream`'s. The field table is the Registry's
(``woof/hex/mpas_seaice/data/registry-fields.json``): every real and integer
field of the pools the routines read, a field whose packages are off absent,
as MPAS allocates them.

The fixture keeps, per case, every output field the oracle changed and the
tracer object the oracle logs (``column-logs.json``).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from woof.hex.mpas_seaice import column_therm_v842 as ct
from woof.hex.mpas_seaice import column_v842 as cv
from woof.verify import mpas_ocean_stream as _stream

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_seaice_column"
#: the vertical thermodynamics' fixture (VT_CASES)
VT_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_column_therm"
PRECISIONS = {"r8": np.float64}
REAL, INT, DIM = _stream.REAL, _stream.INT, _stream.DIM
#: the Registry structs the column routines read and write, as oracle_stream pools
POOLS = ("mesh", "tracers", "tracers_aggregate", "icestate", "boundary", "velocity_solver", "atmos_coupling",
         "atmos_forcing", "ocean_coupling", "ridging", "melt_growth_rates",
         "atmos_fluxes", "ocean_fluxes", "ocean_atmosphere", "shortwave", "drag", "ponds", "aerosols",
         "biogeochemistry", "initial", "diagnostics")
#: and the vertical thermodynamics' coupled surface fluxes (a pool only its cases carry,
#: so the other cases' bytes are unchanged)
VT_POOLS = POOLS + ("alternative_atmos_forcing",)
#: the mesh block: one block's cells, owned and halo
MESH = {"nCells": 12, "nCellsSolve": 9, "nEdges": 30, "nEdgesSolve": 24, "nVertices": 20, "nVerticesSolve": 16,
        "maxEdges": 6, "maxEdges2": 12, "vertexDegree": 3}
_EXTENTS = {"nCells": "C1", "nEdges": "E1", "nVertices": "V1"}
BASE_CONFIG = {"config_dt": 3600.0, "oracle_clock_step": "0000_01:00:00", "oracle_update_type": "thermodynamics",
               "oracle_ice_age_time_offset": 0.0}
_BL99 = {"config_thermodynamics_type": "BL99", "config_sea_freezing_temperature_type": "linear_salt"}
_NO_PONDS = {"config_use_level_meltponds": False}
#: (name, seed, stage, config overrides)
CASES = (
    ("init_default", 20261901, "init", {}),
    ("init_bl99", 20261902, "init", _BL99),
    ("init_bl99_fixed_surface_temperature", 20261903, "init", {**_BL99,
                                                               "config_calc_surface_temperature": False}),
    ("init_zero_layer", 20261904, "init", {"config_thermodynamics_type": "zero layer",
                                           "config_sea_freezing_temperature_type": "linear_salt"}),
    ("init_bounds_new", 20261905, "init", {"config_category_bounds_type": "new"}),
    ("init_bounds_wmo6", 20261906, "init", {"config_category_bounds_type": "WMO", "config_nCategories": 6}),
    ("init_bounds_asymptotic", 20261907, "init", {"config_category_bounds_type": "asymptotic",
                                                  "config_nCategories": 4}),
    ("init_original_delta_function", 20261908, "init", {"config_itd_conversion_type": "delta function",
                                                        "config_nCategories": 3}),
    ("init_single_category", 20261909, "init", {"config_category_bounds_type": "single category",
                                                "config_itd_conversion_type": "delta function",
                                                "config_nCategories": 1}),
    ("init_bl99_three_layers", 20261910, "init", {**_BL99, "config_nIceLayers": 3}),
    ("init_bl99_sixteen_layers", 20261911, "init", {**_BL99, "config_nIceLayers": 16}),
    ("init_uniform", 20261912, "init", {"config_initial_condition_type": "uniform",
                                        "config_initial_velocity_type": "none", "config_initial_ice_area": 0.25,
                                        "config_initial_ice_volume": 0.5, "config_initial_snow_volume": 0.1}),
    ("init_no_ice_topo_ponds", 20261913, "init", {"config_initial_condition_type": "no_ice",
                                                  "config_use_level_meltponds": False,
                                                  "config_use_topo_meltponds": True}),
    ("init_cesm_ponds_aerosols", 20261914, "init", {"config_use_level_meltponds": False,
                                                    "config_use_cesm_meltponds": True,
                                                    "config_pond_refreezing_type": "cesm",
                                                    "config_use_aerosols": True}),
    ("aggregate_default", 20261921, "aggregate", {}),
    ("aggregate_cesm_ponds", 20261922, "aggregate", {"config_use_level_meltponds": False,
                                                     "config_use_cesm_meltponds": True,
                                                     "config_pond_refreezing_type": "cesm"}),
    ("aggregate_topo_ponds", 20261923, "aggregate", {"config_use_level_meltponds": False,
                                                     "config_use_topo_meltponds": True}),
    ("aggregate_bare", 20261924, "aggregate", {**_NO_PONDS, "config_use_ice_age": False,
                                               "config_use_first_year_ice": False, "config_use_level_ice": False,
                                               "config_nCategories": 3, "config_nIceLayers": 2}),
    ("update_state_thermodynamics", 20261931, "update_state", {}),
    ("update_state_transport_age_offset", 20261932, "update_state", {
        "oracle_update_type": "transport", "oracle_ice_age_time_offset": 1800.0}),
    ("update_state_no_age", 20261933, "update_state", {"config_use_ice_age": False}),
)
_VT = {**_BL99, **_NO_PONDS}
#: the vertical thermodynamics (colpkg stage 2): name, seed, stage, overrides
VT_CASES = (
    ("vt_bl99_bubbly", 20261941, "vertical_thermodynamics", _VT),
    ("vt_bl99_mu71_highfreq", 20261942, "vertical_thermodynamics", {
        **_VT, "config_heat_conductivity_type": "MU71", "config_use_high_frequency_coupling": True, "config_boundary_layer_iteration_number": 3,
        "config_nIceLayers": 4, "config_nCategories": 3}),
    ("vt_bl99_coupled_surface_temperature", 20261943, "vertical_thermodynamics", {
        **_VT, "config_calc_surface_temperature": False}),
    ("vt_bl99_melting", 20261944, "vertical_thermodynamics", {**_VT, "config_start_time": "2000-07-10_00:00:00"}),
    ("vt_bl99_first_year_north", 20261945, "vertical_thermodynamics", {
        **_VT, "config_start_time": "2000-09-15_00:00:00"}),
    ("vt_bl99_first_year_south", 20261946, "vertical_thermodynamics", {
        **_VT, "config_start_time": "2001-03-16_00:00:00", "config_nIceLayers": 9}),
    ("vt_bl99_one_layer_bare", 20261947, "vertical_thermodynamics", {
        **_VT, "config_nIceLayers": 1, "config_nCategories": 1, "config_category_bounds_type": "single category",
        "config_itd_conversion_type": "delta function", "config_use_ice_age": False,
        "config_use_first_year_ice": False, "config_use_level_ice": False}),
    ("vt_bl99_abort_salinity", 20261948, "vertical_thermodynamics", _VT),
)


def configs_of(overrides: dict, stage: str) -> dict:
    return {**cv.default_configs(), **BASE_CONFIG, **overrides, "oracle_stage": stage}


def field_rows(config: dict) -> tuple[list, dict[str, int], dict[str, bool]]:
    """The stream's field table, the dimensions and the packages for ``config``."""
    dims = cv.dimensions(config, {k: MESH[k] for k in ("nCells", "nEdges", "maxEdges", "maxEdges2", "nVertices",
                                                        "vertexDegree")})
    active = cv.packages(config)
    rows = [("mesh", name, DIM, ()) for name in sorted(dims)]
    rows += [("mesh", name, DIM, ()) for name in ("nCellsSolve", "nEdgesSolve", "nVerticesSolve")]
    rows += [("tracers_aggregate", "nCellsSolve", DIM, ())]
    pools = VT_POOLS if config.get("oracle_stage") == "vertical_thermodynamics" else POOLS
    for struct in cv.registry_fields()["var_structs"]:
        if struct["name"] not in pools:
            continue
        struct_on = all(active[f"{p}Active"] for p in struct["packages"])
        for name, kind, dimension_text, package_text in struct["vars"]:
            if kind not in ("real", "integer"):
                continue
            names = [d for d in dimension_text.split() if d != "Time"]
            fshape = tuple(_EXTENTS.get(d, dims.get(d)) for d in names)
            own = [p for p in package_text.split(";") if p]
            on = struct_on and (not own or any(active[f"{p}Active"] for p in own))
            rows.append((struct["name"], name, REAL if kind == "real" else INT, fshape, on))
    return rows, dims, active


def _uniform(rs):
    """``rs.uniform`` with the scaling done in numpy: RandomState.uniform's C
    ``low + scale * r`` is contracted into an FMA on arm64 builds and not on
    x86-64, so its words differ by host; separate ufuncs round each step."""
    def u(low, high, size=None):
        r = rs.random_sample(size)
        return np.add(np.multiply(np.subtract(high, low), r), low)
    return u


def _values(rs, name: str, shp: tuple, dims: dict, stage: str):
    """Plausible values per field; the rest uniform in [-1, 1)."""
    nc = dims["nCells"]
    u = _uniform(rs)
    if name == "latCell":
        # mostly polar, a temperate cell, and cells exactly on the bounds' degrees
        lat = np.radians(np.where(u(0.0, 1.0, shp) < 0.5, u(70.5, 89.0, shp), u(-89.0, -60.5, shp)))
        lat[:4] = np.radians([75.0, -65.0, 10.0, 70.0])
        return lat
    if name == "seaSurfaceSalinity":
        s = u(28.0, 36.0, shp)
        s[2], s[5] = 130.0, 0.0
        return s
    if name == "seaSurfaceTemperature":
        # mostly at or below freezing, two warm cells
        t = u(-2.2, -1.6, shp)
        t[[4, 8]] = 0.5
        return t
    if name == "airTemperature":
        t = u(240.0, 280.0, shp)
        t[0] = 278.0
        return t
    if name in ("uAirVelocity", "vAirVelocity"):
        return u(-20.0, 20.0, shp)
    if name in ("landIceMask", "landIceMaskVertex"):
        m = np.zeros(shp, np.int32)
        m[min(6, shp[0] - 1)] = 1
        return m
    if name == "interiorVertex":
        m = np.ones(shp, np.int32)
        m[::7] = 0
        return m
    return None


def _tracers(rs, pools: dict, dims: dict):
    """The category state: some ice-free cells and categories, a cell of
    sub-puny area, ice without volume, ice ages and level ice that are zero."""
    nc, ncat = dims["nCells"], dims["nCategories"]
    t = pools["tracers"]
    u = _uniform(rs)
    area = u(0.0, 1.0 / ncat, (nc + 1, ncat, 1))
    area[1] = 0.0
    area[3] = 2.0e-12 / ncat
    area[4, 0] = 0.0
    volume = area * u(0.0, 4.0, area.shape)
    volume[5, -1] = 0.0
    snow = volume * u(0.0, 0.3, area.shape)
    t["iceAreaCategory"], t["iceVolumeCategory"], t["snowVolumeCategory"] = area, volume, snow
    ranges = {"surfaceTemperature": (-30.0, 0.0), "iceEnthalpy": (-3.3e8, -1.0e8),
              "snowEnthalpy": (-1.2e8, -1.0e8), "iceSalinity": (0.0, 10.0), "iceAge": (0.0, 3.0e7),
              "firstYearIceArea": (0.0, 1.0), "levelIceArea": (0.0, 1.0), "levelIceVolume": (0.0, 1.0),
              "pondArea": (0.0, 1.0), "pondDepth": (0.0, 0.5), "pondLidThickness": (0.0, 0.1)}
    for name, (lo, hi) in ranges.items():
        if t.get(name) is not None:
            t[name] = u(lo, hi, t[name].shape)
    if t.get("iceAge") is not None:
        t["iceAge"][7] = 0.0
    if t.get("levelIceArea") is not None:
        t["levelIceArea"][8, :] = 0.0
    if t.get("pondArea") is not None:
        t["pondArea"][6, 1 % ncat] = 0.0


_RHOI, _RHOS, _CP_ICE, _CP_OCN, _LFRESH, _DEPRESST = 917.0, 330.0, 2106.0, 4218.0, 2.835e6 - 2.501e6, 0.054


def _thermo_state(rs, case: dict) -> None:
    """A BL99 column state and forcing the solver converges on: category
    thicknesses 0.2-4 m, snow up to 0.4 m (a snow-free cell and one below
    hs_min), salinities above min_salin, temperatures from the surface's down
    to the freezing point and below each layer's melting point, enthalpies
    consistent with them, an ice-free cell and a sub-puny one, day and night
    shortwave. Every step is a separate numpy ufunc (no reductions), so the
    words are the same on every host."""
    pools, dims, name = case["pools"], case["dims"], case["name"]
    nc, ncat, nil, nsl = dims["nCells"], dims["nCategories"], dims["nIceLayers"], dims["nSnowLayers"]
    n1 = nc + 1
    u = _uniform(rs)
    melting = "melting" in name
    t = pools["tracers"]
    share = u(0.05, 1.0, (n1, ncat))
    total = np.zeros((n1, 1))
    for n in range(ncat):
        total[:, 0] = total[:, 0] + share[:, n]
    area = share / total * u(0.3, 0.99, (n1, 1))
    area[1] = 0.0
    area[3] = 2.0e-12 / ncat
    area[4, 0] = 0.0
    hin = u(0.2, 4.0, (n1, ncat))
    hsn = u(0.0, 0.4, (n1, ncat))
    hsn[2] = 0.0
    hsn[6] = 5.0e-5
    hin[7], hsn[7] = 0.3, 0.35           # snow below the freeboard: snow-ice forms
    tsfc = u(-0.6, 0.0, (n1, ncat)) if melting else u(-30.0, -2.0, (n1, ncat))
    sal = u(0.5, 4.0, (n1, ncat, nil))
    tmlt = -sal * _DEPRESST
    tin = np.empty((n1, ncat, nil))
    top = tsfc * 0.8 if not melting else tsfc - 1.5
    for k in range(nil):
        z = (k + 0.5) / nil
        tin[:, :, k] = top + (-1.85 - top) * z
    tin = np.minimum(tin, tmlt - 0.02)
    zqin = -_RHOI * (_CP_ICE * (tmlt - tin) + _LFRESH * (1.0 - tmlt / tin) - _CP_OCN * tmlt)
    tsn = np.minimum(tsfc * 0.6 + tin[:, :, :1][:, :, 0] * 0.4, -0.05)
    zqsn = np.empty((n1, ncat, nsl))
    for k in range(nsl):
        zqsn[:, :, k] = -_RHOS * (_LFRESH - _CP_ICE * tsn)
    if "abort" in name:
        sal[5, 0, nil - 1] = 0.05          # below min_salin: init_vertical_profile stops
        sal[7, 0, 0] = 0.02                # a later cell, which MPAS never reaches
    t["iceAreaCategory"] = area[:, :, None]
    t["iceVolumeCategory"] = (area * hin)[:, :, None]
    t["snowVolumeCategory"] = (area * hsn)[:, :, None]
    t["surfaceTemperature"] = tsfc[:, :, None]
    t["iceSalinity"], t["iceEnthalpy"], t["snowEnthalpy"] = sal, zqin, zqsn
    if t.get("iceAge") is not None:
        t["iceAge"] = u(0.0, 3.0e7, t["iceAge"].shape)
    if t.get("firstYearIceArea") is not None:
        t["firstYearIceArea"] = u(0.0, 1.0, t["firstYearIceArea"].shape)
    aggregate = np.zeros(n1)
    for n in range(ncat):
        aggregate = aggregate + area[:, n]
    pools["tracers_aggregate"]["iceAreaCell"] = aggregate
    mesh = pools["mesh"]
    mesh["indexToCellID"] = np.arange(1, n1 + 1, dtype=np.int32)
    ac, af = pools["atmos_coupling"], pools["atmos_forcing"]
    ac["airPotentialTemperature"] = u(271.0, 278.0, (n1,)) if melting else u(240.0, 272.0, (n1,))
    ac["airTemperature"] = ac["airPotentialTemperature"] - 0.1
    ac["airSpecificHumidity"] = u(2.0e-4, 3.0e-3, (n1,))
    ac["airDensity"] = u(1.2, 1.4, (n1,))
    ac["airLevelHeight"] = u(5.0, 30.0, (n1,))
    ac["uAirVelocity"], ac["vAirVelocity"] = u(-15.0, 15.0, (n1,)), u(-15.0, 15.0, (n1,))
    ac["uAirVelocity"][0], ac["vAirVelocity"][0] = 0.3, -0.2      # below umin
    af["windSpeed"] = np.sqrt(ac["uAirVelocity"] * ac["uAirVelocity"] + ac["vAirVelocity"] * ac["vAirVelocity"])
    ac["longwaveDown"] = u(150.0, 320.0, (n1,))
    snowfall = u(0.0, 2.0e-5, (n1,))
    snowfall[::3] = 0.0
    ac["snowfallRate"] = snowfall
    sw = pools["shortwave"]
    day = np.ones((n1, ncat))
    if not melting:
        day[::2] = 0.0
    sw["surfaceShortwaveFlux"] = u(0.0, 300.0 if melting else 150.0, (n1, ncat)) * day
    sw["penetratingShortwaveFlux"] = u(0.0, 5.0, (n1, ncat)) * day
    # the interior flux is what the layers absorb (the solver's energy balance)
    layers = u(0.0, 30.0 / nil, (n1, ncat, nil)) * day[:, :, None]
    snow_layers = u(0.0, 10.0, (n1, ncat, nsl)) * day[:, :, None]
    snow_layers[hsn <= 1.0e-4] = 0.0     # no snow layer below hs_min absorbs (l_snow)
    interior = np.zeros((n1, ncat))
    for k in range(nsl):
        interior = interior + snow_layers[:, :, k]
    for k in range(nil):
        interior = interior + layers[:, :, k]
    sw["interiorShortwaveFlux"] = interior
    sw["absorbedShortwaveIceLayer"] = layers
    sw["absorbedShortwaveSnowLayer"] = snow_layers
    oc = pools["ocean_coupling"]
    oc["seaSurfaceSalinity"] = u(30.0, 35.0, (n1,))
    oc["seaFreezingTemperature"] = -_DEPRESST * oc["seaSurfaceSalinity"]
    oc["seaSurfaceTemperature"] = oc["seaFreezingTemperature"] + u(0.0, 1.5, (n1,))
    frzmlt = u(-200.0, 50.0, (n1,))
    frzmlt[8] = 0.0
    oc["freezingMeltingPotential"] = frzmlt
    vs = pools["velocity_solver"]
    vs["oceanStressCellU"], vs["oceanStressCellV"] = u(-0.3, 0.3, (n1,)), u(-0.3, 0.3, (n1,))
    vs["uVelocity"] = u(-0.5, 0.5, vs["uVelocity"].shape)
    pools["drag"]["oceanDragCoefficient"] = u(0.004, 0.007, (n1,))
    dg = pools["diagnostics"]
    for key in ("meltOnset", "freezeOnset"):
        onset = u(0.0, 365.0, (n1,))
        onset[::2] = 0.0
        dg[key] = onset
    alt = pools["alternative_atmos_forcing"]
    alt["latentHeatFluxCouple"] = u(-20.0, 5.0, (n1, ncat))
    alt["sensibleHeatFluxCouple"] = u(-30.0, 30.0, (n1, ncat))
    # colpkg_step_therm1 hands set_sfcflux the two swapped (fsurfn is the coupled
    # conductive flux, fcondtopn the coupled surface flux): fsurfn >= fcondtopn here
    alt["surfaceHeatFluxCouple"] = u(-30.0, 10.0, (n1, ncat))
    alt["surfaceConductiveFluxCouple"] = alt["surfaceHeatFluxCouple"] + u(0.0, 5.0, (n1, ncat))


def make_case(name, seed, stage, overrides) -> dict:
    rs = np.random.RandomState(seed)
    config = configs_of(overrides, stage)
    rows, dims, active = field_rows(config)
    case = {"name": name, "precision": "r8", "stage": stage, "config": config,
            "dt": 1800.0 if stage == "update_state" else float(config["config_dt"]),
            "dims": {**dims, "nCells": MESH["nCells"], "nCellsSolve": MESH["nCellsSolve"],
                     "nVertLevels": 0}, "pools": {}}
    for pool, fname, kind, fshape, *on in rows:
        p = case["pools"].setdefault(pool, {})
        if kind == DIM:
            p[fname] = MESH.get(fname, dims.get(fname))
            continue
        if not on[0]:
            p[fname] = None
            continue
        shp = _stream.shape(case, fshape)
        value = _values(rs, fname, shp, case["dims"], stage)
        if value is None:
            value = rs.randint(0, 2, shp).astype(np.int32) if kind == INT else _uniform(rs)(-1.0, 1.0, shp)
        p[fname] = value
    _tracers(rs, case["pools"], case["dims"])
    if stage == "vertical_thermodynamics":
        _thermo_state(rs, case)
    for key, on in active.items():
        config[f"package:{key}"] = on
    case["fields"] = [r[:4] for r in rows]
    return case


def cases(precision: str = "r8", rows=CASES) -> list[dict]:
    if precision != "r8":
        raise ValueError("the column oracle is RKIND 8 only")
    return [make_case(*row) for row in rows]


def input_bytes(case: dict) -> bytes:
    return _stream.input_bytes(case["fields"], case)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    return _stream.read_outputs(case["fields"], raw, case)


def parse_log(text: str) -> dict:
    """The tracer object (``TOBJ`` rows, a repeated name's rows concatenated)
    and every other line the routines logged."""
    tobj: dict[str, list[int]] = {}
    other = []
    for line in text.splitlines():
        if line.startswith("TOBJ "):
            parts = line.split()
            tobj.setdefault(parts[1], []).extend(int(x) for x in parts[2:])
        elif line.strip():
            other.append(line.rstrip())
    return {"tracer_object": tobj, "log": other}


def run_oracle(build_dir: Path, case: dict) -> tuple[dict[str, np.ndarray], dict]:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, log = Path(tmp) / "in.bin", Path(tmp) / "out.bin", Path(tmp) / "log.txt"
        src.write_bytes(input_bytes(case))
        subprocess.run([str(Path(build_dir) / "run_column_r8"), str(src), str(dst), str(log)], check=True)
        return read_outputs(dst.read_bytes(), case), parse_log(log.read_text(encoding="utf-8"))


word_mismatches = _stream.word_mismatches


def changed_fields(case: dict, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    changed = {}
    for key, value in outputs.items():
        pool, name = key.split("/", 1)
        if word_mismatches(value, np.asarray(case["pools"][pool][name])):
            changed[key] = value
    return changed


def load_fixture(fixture_dir: Path = FIXTURE_DIR) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(fixture_dir / "oracle-r8.npz")
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        case, field = key.split("/", 1)
        out.setdefault(case, {})[field] = data[key]
    return out


def load_logs(fixture_dir: Path = FIXTURE_DIR) -> dict:
    return json.loads((fixture_dir / "column-logs.json").read_text(encoding="utf-8"))


def expected_outputs(case: dict, fixture: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for pool, name, kind, _ in case["fields"]:
        if kind not in (REAL, INT) or case["pools"][pool][name] is None:
            continue
        key = f"{pool}/{name}"
        out[key] = fixture[key] if key in fixture else np.asarray(case["pools"][pool][name])
    return out


def on_device(case: dict) -> dict[str, dict]:
    return _stream.on_device(case["fields"], case)


def run_gpu(case: dict) -> tuple[dict[str, np.ndarray], "cv.ColumnState"]:
    """Run woof.hex.mpas_seaice.column_v842 on ``case`` as run_column.F90 drives the Fortran."""
    import cupy as cp

    dev = on_device(case)
    config = case["config"]
    state = cv.init_parameters(dev, config)
    stage = case["stage"]
    if stage == "init":
        cv.initialize_coupler_fields(dev, config, state)
        cv.init_variables(dev, config, state)
        cv.init_ice_state(dev, config, state)
        cv.reset_coupler_fluxes(dev, config, state)
    elif stage == "aggregate":
        cv.aggregate(dev, config, state)
    elif stage == "update_state":
        cv.update_state(dev, config, state, config["oracle_update_type"], case["dt"],
                        config["oracle_ice_age_time_offset"])
    elif stage == "vertical_thermodynamics":
        state.abort = ct.vertical_thermodynamics(dev, config, state)
    else:
        raise ValueError(f"unknown stage {stage}")
    outputs = {f"{pool}/{name}": cp.asnumpy(dev[pool][name]) for pool, name, kind, _ in case["fields"]
               if kind in (REAL, INT) and dev[pool][name] is not None}
    return outputs, state


def measure(reference: dict | None = None, logs: dict | None = None, *, rows=CASES,
            fixture_dir: Path = FIXTURE_DIR) -> dict[str, dict[str, int]]:
    """Per case and field, the words the GPU gets wrong, and the tracer
    object's integers that differ from the oracle's."""
    reference = load_fixture(fixture_dir) if reference is None else reference
    logs = load_logs(fixture_dir) if logs is None else logs
    report: dict[str, dict[str, int]] = {}
    for case in cases("r8", rows):
        got, state = run_gpu(case)
        want = expected_outputs(case, reference.get(case["name"], {}))
        report[case["name"]] = {key: word_mismatches(got[key], want[key]) for key in want}
        oracle = logs[case["name"]]["tracer_object"]
        mine = state.tracers.log_rows()
        report[case["name"]]["tracer_object"] = sum(int(mine[k] != oracle.get(k)) for k in mine)
    return report


def oracle_abort(log: dict) -> tuple[int, int | None, str | None]:
    """The oracle's ``column_vertical_thermodynamics`` outcome: ``(ierr, cell,
    message)`` as :func:`woof.hex.mpas_seaice.column_therm_v842.vertical_thermodynamics`
    returns it (the cell is ``indexToCellID``, the message the logged error)."""
    ierr = log["tracer_object"]["IERR"][0]
    if not ierr:
        return 0, None, None
    lines = log["log"]
    prefix = "ERR column_vertical_thermodynamics: "
    message = next(line for line in lines if line.startswith(prefix))[len("ERR "):]
    cell = int(lines[lines.index("ERR iCell: $i") + 1].split()[1])
    return ierr, cell, message


def measure_vt(reference: dict | None = None, logs: dict | None = None, *, rows=VT_CASES,
               fixture_dir: Path = VT_FIXTURE_DIR) -> dict[str, dict[str, int]]:
    """Per vertical-thermodynamics case: the words the GPU gets wrong per
    field (cases the oracle completes), whether its abort (ierr, first
    aborting cell, message) differs from the oracle's, and the tracer
    object's integers that differ."""
    reference = load_fixture(fixture_dir) if reference is None else reference
    logs = load_logs(fixture_dir) if logs is None else logs
    report: dict[str, dict[str, int]] = {}
    for case in cases("r8", rows):
        got, state = run_gpu(case)
        log = logs[case["name"]]
        oracle = oracle_abort(log)
        row = {"abort": int(tuple(state.abort) != oracle)}
        if not oracle[0]:
            want = expected_outputs(case, reference.get(case["name"], {}))
            row.update({key: word_mismatches(got[key], want[key]) for key in want})
        mine = state.tracers.log_rows()
        row["tracer_object"] = sum(int(mine[k] != log["tracer_object"].get(k)) for k in mine)
        report[case["name"]] = row
    return report


#: colpkg_init_thermo's profiles are graded for every ice-layer count up to this
THERMO_LAYERS = cv.MAX_ICE_LAYERS


def oracle_thermo_profiles(build_dir: Path, n_max: int = THERMO_LAYERS) -> np.ndarray:
    """``run_init_thermo_r8``'s profiles for 1..n_max ice layers, concatenated."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "thermo.bin"
        subprocess.run([str(Path(build_dir) / "run_init_thermo_r8"), str(n_max), str(out)], check=True)
        return np.frombuffer(out.read_bytes(), "<f8").astype(np.float64)


def gpu_thermo_profiles(n_max: int = THERMO_LAYERS) -> np.ndarray:
    """:mod:`~woof.hex.mpas_seaice.column_v842`'s ``colpkg_init_thermo`` (BL99) for 1..n_max layers."""
    return np.concatenate([cv.init_thermo_profile(n, ktherm=1) for n in range(1, n_max + 1)])


def thermo_profile_mismatches(want: np.ndarray, n_max: int = THERMO_LAYERS) -> dict[int, int]:
    """Per ice-layer count, the profile words the GPU gets wrong."""
    got = gpu_thermo_profiles(n_max)
    out, off = {}, 0
    for n in range(1, n_max + 1):
        out[n] = word_mismatches(got[off:off + n + 1], want[off:off + n + 1])
        off += n + 1
    return out


def load_thermo_profiles(fixture_dir: Path = FIXTURE_DIR) -> np.ndarray:
    return np.load(fixture_dir / "init-thermo-r8.npy")
