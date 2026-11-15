"""Cases and the stream layout for the MPAS-Seaice column-package oracle.

``tools/mpas_seaice_oracle/run_column.F90`` runs MPAS-Model v8.4.2's
``seaice_init_column_physics_package_parameters`` on one block, then the
stage a case names (``oracle_stage``):

- ``init``: ``initialize_coupler_fields``,
  ``seaice_init_column_physics_package_variables``, ``init_ice_state`` and
  ``seaice_reset_coupler_fluxes``, in ``seaice_init``'s order;
- ``aggregate``: ``seaice_column_aggregate``;
- ``update_state``: ``seaice_column_update_state`` (``oracle_update_type``,
  the stream's dt, ``oracle_ice_age_time_offset``).

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

from woof.hex.mpas_seaice import column_v842 as cv
from woof.verify import mpas_ocean_stream as _stream

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_seaice_column"
PRECISIONS = {"r8": np.float64}
REAL, INT, DIM = _stream.REAL, _stream.INT, _stream.DIM
#: the Registry structs the column routines read and write, as oracle_stream pools
POOLS = ("mesh", "tracers", "tracers_aggregate", "icestate", "boundary", "velocity_solver", "atmos_coupling",
         "atmos_forcing", "ocean_coupling", "ridging", "melt_growth_rates",
         "atmos_fluxes", "ocean_fluxes", "ocean_atmosphere", "shortwave", "drag", "ponds", "aerosols",
         "biogeochemistry", "initial", "diagnostics")
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
    for struct in cv.registry_fields()["var_structs"]:
        if struct["name"] not in POOLS:
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
