"""Cases and the stream layout for the MPAS-Seaice incremental-remap oracle.

``tools/mpas_seaice_oracle/run_incremental_remap.F90`` runs MPAS-Model
v8.4.2's ``seaice_init_advection_incremental_remap`` (verbatim) on one block,
then the stage a case names (``oracle_stage``):

- 0: nothing more: the tracer tree (from the packages) and the geometry;
- 1: the volume-to-thickness conversion of ``seaice_run_advection_incremental_remap``
  (and, under ``oracle_round_trip``, back to volume).

Everything is built with gfortran -O0 -ffp-contract=off at RKIND 4 and 8. The
stream is :mod:`woof.verify.mpas_ocean_stream`'s, with the remap's pools.

The meshes are :func:`woof.verify.mpas_seaice_planar_mesh.planar_mesh`'s
blocks (hexagons and quadrilaterals, with land and halo layers), so the
geometry v8.4.2 derives is real. ``coeffs_reconstruct`` (MPAS's RBF
reconstruction, an input here) is the least-squares map from the edges'
normal components to the cell's vector.

The fixture keeps, per case, every output field the oracle changed (the
geometry included: the GPU remap takes it as input) and lists the rest as
unchanged; ``remap-logs.json`` keeps the tracer list and the logged warnings.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from woof.hex.mpas_seaice.remap_v842 import PACKAGES, TRACER_TABLE
from woof.verify import mpas_ocean_stream as _stream
from woof.verify.mpas_seaice_planar_mesh import ordered_sum, planar_mesh

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_seaice_remap"
PRECISIONS = {"r4": np.float32, "r8": np.float64}
REAL, INT, DIM = _stream.REAL, _stream.INT, _stream.DIM
_C, _V, _E = ("C1",), ("V1",), ("E1",)

#: a tracer's first extent: one layer, the ice layers or the snow layers
TRACER_LAYERS = {"iceEnthalpy": "NIL", "iceSalinity": "NIL", "snowEnthalpy": "NSL"}
#: the per-tracer fields of the remap's pools (tracers.F:967-1100): (pool, suffix)
TRACER_AUX = (("tracer_masks", "Mask"), ("tracer_edge_fluxes", "EdgeFlux"), ("tracer_products", "Product"),
              ("tracer_triangles", "Triangle"), ("tracer_barycenter", "Barycenterx"),
              ("tracer_barycenter", "Barycentery"), ("tracer_reconstruction", "Center"),
              ("tracer_reconstruction", "Gradx"), ("tracer_reconstruction", "Grady"),
              ("tracer_conservation", "Cons"), ("tracer_conservation", "Cons@2"),
              ("tracer_monotonicity", "LocalMin"), ("tracer_monotonicity", "LocalMax"))
GEOMETRY = ("xVertexOnCell", "yVertexOnCell", "xVertexOnEdge", "yVertexOnEdge", "remapEdge", "cellsOnEdgeRemap",
            "edgesOnEdgeRemap", "minLengthEdgesOnVertex",
            *(f"{m}AvgCell" for m in ("x", "y", "xx", "xy", "yy", "xxx", "xxy", "xyy", "yyy", "xxxx", "xxxy",
                                      "xxyy", "xyyy", "yyyy")))
BASE_CONFIG = {
    "config_rotate_cartesian_grid": True, "config_use_level_meltponds": True, "config_use_halo_exch": False,
    "config_aggregate_halo_exch": False, "config_reuse_halo_exch": False, "config_conservation_check": False,
    "config_monotonicity_check": False, "config_recover_tracer_means_check": False, "oracle_n_steps": 1,
    "oracle_round_trip": False, "mesh:on_a_sphere": False,
}
_STANDARD = ("iceAreaCategory", "iceVolumeCategory", "snowVolumeCategory", "surfaceTemperature", "iceEnthalpy",
             "iceSalinity", "snowEnthalpy")
_ALL_PACKAGES = {p: True for p in PACKAGES}
#: (name, seed, stage, mesh kind, owned radius, packages on, config overrides, field overrides)
#: field overrides: "absent": tracers whose field is a package that is off
CASES = (
    ("tree_standard", 20261701, 0, "hex", 1, {}, {}, {}),
    ("tree_column_tracers", 20261702, 0, "hex", 1, {
        "pkgColumnTracerIceAgeActive": True, "pkgColumnTracerFirstYearIceActive": True,
        "pkgColumnTracerLevelIceActive": True, "pkgColumnTracerPondsActive": True,
        "pkgColumnTracerLidThicknessActive": True, "pkgColumnTracerAerosolsActive": True}, {}, {}),
    ("tree_cesm_ponds", 20261703, 0, "quad", 1, {
        "pkgColumnTracerPondsActive": True, "pkgColumnTracerLidThicknessActive": True},
     {"config_use_level_meltponds": False}, {}),
    ("tree_every_package", 20261704, 0, "hex", 1, _ALL_PACKAGES, {}, {}),
    ("tree_inactive_leaf", 20261705, 0, "hex", 1, {
        "pkgColumnTracerPondsActive": True, "pkgColumnTracerLidThicknessActive": True,
        "pkgColumnTracerIceAgeActive": True}, {}, {"absent": ("pondLidThickness", "iceAge")}),
    ("convert_hex", 20261711, 1, "hex", 3, {"pkgColumnTracerLevelIceActive": True}, {}, {}),
    ("convert_quad_round_trip", 20261712, 1, "quad", 3, {}, {"oracle_round_trip": True}, {}),
)


def _tracer_rows(names, with_aux: bool) -> list:
    rows = []
    for name in names:
        layers = TRACER_LAYERS.get(name, 1)
        rows.append(("tracers", name, REAL, (layers, "NCAT", "C1")))
        if not with_aux:
            continue
        for pool, suffix in TRACER_AUX:
            if pool == "tracer_masks":
                rows.append((pool, name + suffix, INT, (layers, "NCAT", "C1")))
            elif pool == "tracer_edge_fluxes":
                rows.append((pool, name + suffix, REAL, (layers, "NCAT", "E1")))
            elif pool == "tracer_triangles":
                rows.append((pool, name + suffix, REAL, (layers, "NCAT", 6, 6, "E1")))
            elif pool == "tracer_conservation":
                rows.append((pool, name + suffix, REAL, (layers, "NCAT")))
            else:
                rows.append((pool, name + suffix, REAL, (layers, "NCAT", "C1")))
    return rows


def field_rows(case: dict) -> list:
    """The stream's field table for ``case``: the mesh, the velocity, the
    remap pool and each tracer (with its per-tracer pools from stage 1 on)."""
    mesh = [("mesh", n, DIM, ()) for n in ("nCells", "nEdges", "nVertices", "nCellsSolve", "nEdgesSolve",
                                           "nVerticesSolve", "maxEdges", "vertexDegree", "nQuadPoints",
                                           "nTriPerEdgeRemap")]
    mesh += [("mesh", "nEdgesOnCell", INT, _C), ("mesh", "edgesOnCell", INT, ("ME", "C1")),
             ("mesh", "cellsOnCell", INT, ("ME", "C1")), ("mesh", "verticesOnCell", INT, ("ME", "C1")),
             ("mesh", "verticesOnEdge", INT, (2, "E1")), ("mesh", "cellsOnEdge", INT, (2, "E1")),
             ("mesh", "cellsOnVertex", INT, ("VD", "V1")), ("mesh", "edgesOnVertex", INT, ("VD", "V1")),
             ("mesh", "indexToCellID", INT, _C), ("mesh", "indexToVertexID", INT, _V),
             ("mesh", "indexToEdgeID", INT, _E)]
    mesh += [("mesh", n, REAL, _C) for n in ("areaCell", "xCell", "yCell", "zCell")]
    mesh += [("mesh", n, REAL, _V) for n in ("xVertex", "yVertex", "zVertex")]
    mesh += [("mesh", n, REAL, _E) for n in ("dcEdge", "dvEdge", "xEdge", "yEdge", "zEdge")]
    mesh += [("mesh", "coeffs_reconstruct", REAL, (3, "ME", "C1"))]
    velocity = [("velocity_solver", "uVelocity", REAL, _V), ("velocity_solver", "vVelocity", REAL, _V),
                ("velocity_solver", "dynamicsTimeStep", REAL, ())]
    remap = [("incremental_remap", "xTriangle", REAL, (6, 6, "E1")),
             ("incremental_remap", "yTriangle", REAL, (6, 6, "E1")),
             ("incremental_remap", "iCellTriangle", INT, (6, "E1")),
             ("incremental_remap", "triangleArea", REAL, (6, "E1")),
             ("incremental_remap", "departurePoint", REAL, (2, "V1")),
             ("incremental_remap", "remapEdge", INT, _E),
             ("incremental_remap", "cellsOnEdgeRemap", INT, (6, "E1")),
             ("incremental_remap", "edgesOnEdgeRemap", INT, (6, "E1")),
             ("incremental_remap", "xVertexOnCell", REAL, ("ME", "C1")),
             ("incremental_remap", "yVertexOnCell", REAL, ("ME", "C1")),
             ("incremental_remap", "xVertexOnEdge", REAL, (8, "E1")),
             ("incremental_remap", "yVertexOnEdge", REAL, (8, "E1")),
             *(("incremental_remap", n, REAL, (3, 3, "C1")) for n in ("transCellToGlobal", "transGlobalToCell")),
             *(("incremental_remap", n, REAL, (3, 3, "V1")) for n in ("transVertexToGlobal",
                                                                      "transGlobalToVertex")),
             *(("incremental_remap", n, REAL, (3, 3, "E1")) for n in ("transEdgeToGlobal", "transGlobalToEdge")),
             ("incremental_remap", "minLengthEdgesOnVertex", REAL, _V),
             *(("incremental_remap", n, REAL, _C) for n in GEOMETRY[8:]),
             ("incremental_remap", "maskEdge", INT, _E), ("incremental_remap", "maskCell", INT, _C),
             ("incremental_remap", "maskCategoryCell", INT, ("NCAT", "C1")),
             ("incremental_remap", "workCategoryCell", REAL, ("NCAT", "C1"))]
    return mesh + velocity + remap + _tracer_rows(case["tracer_fields"], case["stage"] >= 1)


def tracer_fields(packages: dict) -> list[str]:
    """The tracer fields a case carries: every tracer its packages add."""
    names = []
    for name, _, package in TRACER_TABLE:
        if package is None or packages.get(package, False):
            names.append(name)
    return names


def reconstruction_coefficients(m: dict) -> np.ndarray:
    """``coeffs_reconstruct``: per cell the least-squares map from the edges'
    normal components (cellsOnEdge(1) towards cellsOnEdge(2)) to the vector,
    in closed form."""
    nc, me = m["nCells"], m["maxEdges"]
    coeffs = np.zeros((nc + 1, me, 3))
    for c in range(nc):
        n = m["nEdgesOnCell"][c]
        normals = np.zeros((n, 2))
        for i in range(n):
            e = m["edgesOnCell"][c, i] - 1
            v1, v2 = m["verticesOnEdge"][e] - 1
            tx, ty = m["xVertex"][v2] - m["xVertex"][v1], m["yVertex"][v2] - m["yVertex"][v1]
            length = math.sqrt(tx * tx + ty * ty)
            normals[i] = (ty / length, -tx / length)   # to the right of V1 -> V2: cell 1 -> cell 2
        # (N^T N)^-1 N^T in closed form, summed in edge order (no LAPACK: the same words on every host)
        sxx = sxy = syy = 0.0
        for nx, ny in normals:
            sxx += nx * nx
            sxy += nx * ny
            syy += ny * ny
        det = sxx * syy - sxy * sxy
        for i, (nx, ny) in enumerate(normals):
            coeffs[c, i, 0] = (syy * nx - sxy * ny) / det
            coeffs[c, i, 1] = (sxx * ny - sxy * nx) / det
    return coeffs


def make_case(name, seed, stage, kind, owned, packages, config_overrides, field_overrides,
              precision="r4") -> dict:
    """One deterministic case: five categories' worth of ice (three here) with
    ice-free cells, tiny areas below the masks' and the zap's thresholds,
    snow-free ice, a 10 km mesh and drift up to 0.3 m/s over an hour."""
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    m = planar_mesh(kind, rs, owned=owned)
    nc, ne, nv, me, vd = m["nCells"], m["nEdges"], m["nVertices"], m["maxEdges"], m["vertexDegree"]
    ncat, nil, nsl = 3, 3, int(field_overrides.get("nSnowLayers", 1))
    dims = {"nCells": nc, "nEdges": ne, "nVertices": nv, "maxEdges": me, "vertexDegree": vd, "nCategories": ncat,
            "nIceLayers": nil, "nSnowLayers": nsl, "nVertLevels": 1}
    config = {**BASE_CONFIG, "config_num_halos": m["nHalos"], **config_overrides, "oracle_stage": stage}
    for package in PACKAGES:
        config[f"package:{package}"] = bool(packages.get(package, False))
    case = {"name": name, "precision": precision, "dims": dims, "config": config, "dt": 3600.0, "stage": stage,
            "mesh_kind": kind, "packages": {p: bool(packages.get(p, False)) for p in PACKAGES},
            "tracer_fields": tracer_fields(packages), "absent": tuple(field_overrides.get("absent", ()))}

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    values = {k: m[k] for k in ("nCells", "nEdges", "nVertices", "maxEdges", "vertexDegree", "nEdgesOnCell",
                                "edgesOnCell", "cellsOnCell", "verticesOnCell", "verticesOnEdge", "cellsOnEdge",
                                "cellsOnVertex", "edgesOnVertex", "areaCell", "xCell", "yCell", "zCell",
                                "xVertex", "yVertex", "zVertex", "dcEdge", "dvEdge", "xEdge", "yEdge", "zEdge")}
    values.update({"nCellsSolve": int(m["nCellsArray"][0]), "nEdgesSolve": int(m["nEdgesArray"][0]),
                   "nVerticesSolve": int(m["nVerticesArray"][0]), "nQuadPoints": 6, "nTriPerEdgeRemap": 6,
                   "indexToCellID": np.arange(1, nc + 2), "indexToVertexID": np.arange(1, nv + 2),
                   "indexToEdgeID": np.arange(1, ne + 2), "coeffs_reconstruct": reconstruction_coefficients(m),
                   "dynamicsTimeStep": 3600.0})
    # drift at the vertices: a smooth field up to about 0.3 m/s
    xv, yv = m["xVertex"], m["yVertex"]
    # a smooth bilinear-plus-quadratic drift (no libm: the same words on every host)
    a = uniform(-1.0, 1.0, (2, 4)) / 4.0
    speed = float(field_overrides.get("speed", 0.3))
    sx = (xv - ordered_sum(xv[:nv].tolist()) / nv) / (xv[:nv].max() - xv[:nv].min())
    sy = (yv - ordered_sum(yv[:nv].tolist()) / nv) / (yv[:nv].max() - yv[:nv].min())
    u = speed * (a[0, 0] + a[0, 1] * sx + a[0, 2] * sy + a[0, 3] * sx * sy) * 4.0 / 1.5
    v = speed * (a[1, 0] + a[1, 1] * sx + a[1, 2] * sy + a[1, 3] * sx * sy) * 4.0 / 1.5
    # still over the block's middle (owned cells there keep their values, tiny areas among them)
    cx, cy = ordered_sum(m["xCell"][:nc].tolist()) / nc, ordered_sum(m["yCell"][:nc].tolist()) / nc
    still = (xv - cx) ** 2 + (yv - cy) ** 2 < 1.6e4 ** 2
    u[still] = 0.0
    v[still] = 0.0
    values["uVelocity"], values["vVelocity"] = u, v
    # the ice: areas per category with an ice-free run of cells and tiny areas
    area = uniform(0.0, 0.35, (nc + 1, ncat))
    area[rs.random_sample((nc + 1, ncat)) < 0.12] = 0.0
    free = (m["xCell"] - m["xCell"][:nc].max()) ** 2 + (m["yCell"] - m["yCell"][:nc].max()) ** 2 < 3.0e4 ** 2
    area[free] = 0.0
    tiny = rs.random_sample((nc + 1, ncat))
    area[tiny < 0.03] = 1.0e-12
    area[(tiny >= 0.03) & (tiny < 0.05)] = 1.0e-23
    middle = (m["xCell"] - cx) ** 2 + (m["yCell"] - cy) ** 2 < 0.7e4 ** 2
    area[middle, 0] = 1.0e-23
    thickness = uniform(0.2, 3.0, (nc + 1, ncat))
    snow = uniform(0.0, 0.4, (nc + 1, ncat))
    snow[rs.random_sample((nc + 1, ncat)) < 0.15] = 0.0
    tracer_values = {
        "iceAreaCategory": area, "iceVolumeCategory": area * thickness, "snowVolumeCategory": area * snow,
        "surfaceTemperature": uniform(-20.0, 0.0, (nc + 1, ncat)),
        "iceEnthalpy": uniform(-3.3e8, -1.0e8, (nc + 1, ncat, nil)),
        "iceSalinity": uniform(0.0, 10.0, (nc + 1, ncat, nil)),
        "snowEnthalpy": uniform(-1.2e8, -1.0e8, (nc + 1, ncat, nsl)),
        "iceAge": uniform(0.0, 3.0e7, (nc + 1, ncat)), "pondDepth": uniform(0.0, 0.5, (nc + 1, ncat)),
        "pondLidThickness": uniform(0.0, 0.1, (nc + 1, ncat)),
        "levelIceArea": uniform(0.3, 1.0, (nc + 1, ncat)), "levelIceVolume": uniform(0.3, 1.0, (nc + 1, ncat)),
        "pondArea": uniform(0.0, 0.4, (nc + 1, ncat)), "brineFraction": uniform(0.0, 1.0, (nc + 1, ncat)),
    }
    for name in tracer_values:
        tracer_values[name][rs.random_sample(tracer_values[name].shape) < 0.05] = 0.0
    pools: dict[str, dict] = {}
    for pool, fname, fkind, fshape in field_rows(case):
        shp = _stream.shape(case, fshape)
        if fname in values:
            value = values[fname]
        elif pool == "tracers":
            value = tracer_values.get(fname, uniform(0.0, 1.0, shp))
            if fname in case["absent"]:
                value = None
        elif pool == "tracer_triangles":
            value = np.zeros(shp)
        elif fkind == INT:
            value = rs.randint(-3, 4, shp)
        else:
            value = uniform(-1.0, 1.0, shp)
        if value is not None:
            value = np.asarray(value).reshape(shp) if fkind != DIM else value
            if fkind == REAL:
                value = value.astype(real)
            elif fkind == INT:
                value = value.astype(np.int32)
        pools.setdefault(pool, {})[fname] = value
    case["pools"] = pools
    case["fields"] = field_rows(case)
    return case


def cases(precision: str = "r4") -> list[dict]:
    return [make_case(*row, precision=precision) for row in CASES]


def input_bytes(case: dict) -> bytes:
    return _stream.input_bytes(case["fields"], case)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    return _stream.read_outputs(case["fields"], raw, case)


word_mismatches = _stream.word_mismatches


def parse_log(text: str) -> dict:
    """The oracle's log: the ordered tracers, the warnings (message, ints,
    reals as floats), and stage 4's error code."""
    tree, warnings, ierr = [], [], None
    for line in text.splitlines():
        if line.startswith("TRACER "):
            _, name, parent, n_parents, has_child = line.split()
            tree.append([name, parent, int(n_parents), has_child == "T"])
        elif line.startswith("WARN "):
            warnings.append({"message": line[5:], "ints": [], "reals": []})
        elif line.startswith("INTS"):
            warnings[-1]["ints"] = [int(x) for x in line.split()[1:]]
        elif line.startswith("REALS"):
            warnings[-1]["reals"] = [float(x) for x in line.split()[1:]]
        elif line.startswith("IERR "):
            ierr = int(line.split()[1])
    return {"tree": tree, "warnings": warnings, "ierr": ierr}


def run_oracle(build_dir: Path, case: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Run ``run_incremental_remap_<precision>`` on ``case``: its outputs and parsed log."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, log = Path(tmp) / "in.bin", Path(tmp) / "out.bin", Path(tmp) / "log.txt"
        src.write_bytes(input_bytes(case))
        subprocess.run([str(Path(build_dir) / f"run_incremental_remap_{case['precision']}"), str(src), str(dst),
                        str(log)], check=True)
        return read_outputs(dst.read_bytes(), case), parse_log(log.read_text(encoding="utf-8"))


def graded(case: dict, key: str) -> bool:
    """Whether a field is graded: everything but the geometry (the GPU's
    input) and, over whole remap steps, the triangle values (scratch whose
    words reach the fluxes, which are graded)."""
    pool, name = key.split("/", 1)
    if name in GEOMETRY:
        return False
    return not (case["stage"] == 4 and pool == "tracer_triangles")


def changed_fields(case: dict, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The output fields whose words differ from the case's input (the
    triangle values of whole remap steps left out)."""
    changed = {}
    for key, value in outputs.items():
        pool, name = key.split("/", 1)
        if case["stage"] == 4 and pool == "tracer_triangles":
            continue
        if word_mismatches(value, np.asarray(case["pools"][pool][name])):
            changed[key] = value
    return changed


def load_fixture(precision: str) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(FIXTURE_DIR / f"oracle-{precision}.npz")
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        case, field = key.split("/", 1)
        out.setdefault(case, {})[field] = data[key]
    return out


def load_logs() -> dict:
    return json.loads((FIXTURE_DIR / "remap-logs.json").read_text(encoding="utf-8"))


def expected_outputs(case: dict, fixture: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every output field's oracle words: the fixture's, or the input's where unchanged."""
    out = {}
    for pool, name, kind, _ in case["fields"]:
        if kind not in (REAL, INT) or case["pools"][pool][name] is None:
            continue
        key = f"{pool}/{name}"
        out[key] = fixture[key] if key in fixture else np.asarray(case["pools"][pool][name])
    return out


def is_active(case: dict):
    """``isActive`` of a tracer field: absent fields are packages that are off."""
    return lambda name: name not in case["absent"]


def gpu_tree(case: dict):
    from woof.hex.mpas_seaice import remap_v842 as rv
    return rv.tracer_tree(case["packages"], case["config"], is_active(case))


def tree_rows(tree) -> list:
    """The tracer list as the oracle logs it: name, parent name, nParents, hasChild."""
    return [[t.name, t.parent_name, t.n_parents, t.has_child] for t in tree]


def on_device(case: dict, fixture: dict[str, np.ndarray] | None = None) -> dict[str, dict]:
    """The case's pools on the device, with the init's geometry (the fixture's
    remap-pool words where the oracle's init changed them) in place."""
    dev = _stream.on_device(case["fields"], case)
    if fixture is not None:
        import cupy as cp
        for name in GEOMETRY:
            key = f"incremental_remap/{name}"
            if key in fixture:
                dev["incremental_remap"][name] = cp.asarray(np.ascontiguousarray(fixture[key]))
    return dev


def run_gpu(case: dict, fixture: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Run woof.hex.mpas_seaice.remap_v842 on ``case`` as run_incremental_remap.F90 drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_seaice import remap_v842 as rv

    dev = on_device(case, fixture)
    tree = gpu_tree(case)
    stage = case["stage"]
    if stage == 1:
        rv.volume_to_thickness(tree, tracers=dev["tracers"], remap=dev["incremental_remap"])
        if case["config"]["oracle_round_trip"]:
            rv.thickness_to_volume(tree, tracers=dev["tracers"], remap=dev["incremental_remap"])
    elif stage != 0:
        raise ValueError(f"stage {stage} is not ported")
    return {f"{pool}/{name}": cp.asnumpy(dev[pool][name]) for pool, name, kind, _ in case["fields"]
            if kind in (REAL, INT) and dev[pool][name] is not None}


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    """Per case and field, the words the GPU gets wrong (the geometry is the input's, so not graded)."""
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision):
        fixture = reference.get(case["name"], {})
        got = run_gpu(case, fixture)
        want = expected_outputs(case, fixture)
        report[case["name"]] = {key: word_mismatches(got[key], want[key]) for key in want if graded(case, key)}
    return report
