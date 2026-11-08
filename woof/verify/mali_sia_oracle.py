"""Cases and the stream layout for the MALI SIA velocity-solver oracle.

``tools/mali_oracle`` compiles MPAS-Model v8.4.2's land-ice modules
(``li_constants``, ``li_setup``, ``li_mask``, ``li_thermal``,
``li_diagnostic_vars``, ``li_sia``) and the framework operators they call
(``mpas_constants``, ``mpas_vector_operations``, ``mpas_matrix_operations``,
``mpas_geometry_utils``), all verbatim, with gfortran -O0 -ffp-contract=off
at RKIND 4. ``run_oracle.F90`` makes one ``li_sia_solve`` call. This module
writes the driver's input stream, reads its output, builds the cases and runs
the GPU port the same way.

The array convention, the RandomState discipline and the random-connectivity
meshes are those of :mod:`woof.verify.mpas_seaice_evp_oracle`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mali_sia")

INT_FIELDS = ("cellsOnEdge", "verticesOnEdge", "baryCellsOnVertex", "cellsOnVertex", "nEdgesOnEdge",
              "edgesOnEdge", "edgeMask", "cellMask")
REAL_FIELDS = ("layerInterfaceSigma", "layerCenterSigma", "dvEdge", "baryWeightsOnVertex",
               "kiteAreasOnVertex", "weightsOnEdge", "thickness", "upperSurface", "normalSlopeEdge",
               "temperature")
OUTPUT_FIELDS = ("normalVelocity", "flowParamA", "slopeEdge", "tangentSlopeEdge", "upperSurfaceVertex")
TANGENT = {"from_vertex_barycentric": 1, "from_vertex_barycentric_kiteareas": 2, "from_normal_slope": 3}
FLOW_A = {"constant": 1, "PB1982": 2, "CP2010": 3}

#: (name, seed, nCells, nVertLevels, tangent method, flowParamA method, n, enhancement)
CASES = (
    ("bary_pb1982", 20261112, 80, 10, "from_vertex_barycentric", "PB1982", 3.0, 1.0),
    ("kite_cp2010", 20261113, 64, 12, "from_vertex_barycentric_kiteareas", "CP2010", 3.0, 2.5),
    ("normal_constant_n35", 20261114, 48, 8, "from_normal_slope", "constant", 3.5, 1.0),
)


def make_case(name: str, seed: int, n_cells: int, n_levels: int, tangent: str, flow: str,
              n: float, enhancement: float) -> dict:
    """One deterministic case with the magnitudes of an ice sheet on a 2-20 km mesh.

    The case covers:

    - cells without ice (thickness 0, not dynamic);
    - non-dynamic edges, which get zero velocity;
    - edges with only one dynamic neighbour, where flwaLevelEdge takes one
      cell's A;
    - temperatures either side of the 263.15 K branch and beyond the 223.15 K
      and 273.15 K clamps;
    - unsolved edges past nEdgesSolve, which keep their prior words;
    - non-mask bits set in the masks, which the bit tests ignore.

    A dynamic edge always has a dynamic neighbour, as li_calculate_mask
    guarantees; otherwise flwaLevelEdge would divide by zero.
    """
    rs = np.random.RandomState(seed)
    nc, nl = n_cells, n_levels
    ne, nv, me2 = 3 * nc, 2 * nc, 10

    def uniform(lo, hi, shape):
        return (lo + (hi - lo) * rs.random_sample(shape)).astype(np.float32)

    ice = rs.random_sample(nc) < 0.85
    thickness = np.where(ice, uniform(10.0, 3200.0, nc), np.float32(0.0)).astype(np.float32)
    dynamic_cell = ice & (rs.random_sample(nc) < 0.9)
    other_bits = (rs.randint(0, 16, nc) * 4 + rs.randint(0, 2, nc)).astype(np.int32)  # bits other than 2
    cell_mask = (other_bits | np.where(dynamic_cell, 2, 0)).astype(np.int32)
    cells_on_edge = rs.randint(1, nc + 1, (ne, 2)).astype(np.int32)
    any_dynamic = dynamic_cell[cells_on_edge[:, 0] - 1] | dynamic_cell[cells_on_edge[:, 1] - 1]
    dynamic_edge = any_dynamic & (rs.random_sample(ne) < 0.92)
    edge_mask = ((rs.randint(0, 16, ne) * 4 + rs.randint(0, 2, ne)) | np.where(dynamic_edge, 2, 0)).astype(np.int32)
    interfaces = np.sort(rs.random_sample(nl - 1))
    sigma_i = np.concatenate(([0.0], interfaces, [1.0])).astype(np.float32)
    sigma_c = ((sigma_i[:-1].astype(np.float64) + sigma_i[1:].astype(np.float64)) * 0.5).astype(np.float32)
    n_edges_on_edge = rs.randint(6, me2 + 1, ne).astype(np.int32)
    edges_on_edge = np.full((ne, me2), ne + 5, np.int32)
    for e in range(ne):
        edges_on_edge[e, :n_edges_on_edge[e]] = rs.randint(1, ne + 1, n_edges_on_edge[e])
    case = {
        "name": name, "nCells": nc, "nEdges": ne, "nVertices": nv, "nVertLevels": nl, "maxEdges2": me2,
        "nEdgesSolve": ne - 7, "tangentSlopeCalculation": tangent, "flowParamACalculation": flow,
        "iceDensity": np.float32(910.0), "flowLawExponent": np.float32(n),
        "enhancementFactor": np.float32(enhancement), "defaultFlowParamA": np.float32(3.1709792e-24),
        "dynamicThickness": np.float32(10.0),
        "cellsOnEdge": cells_on_edge,
        "verticesOnEdge": rs.randint(1, nv + 1, (ne, 2)).astype(np.int32),
        "baryCellsOnVertex": rs.randint(1, nc + 1, (nv, 3)).astype(np.int32),
        "cellsOnVertex": rs.randint(1, nc + 1, (nv, 3)).astype(np.int32),
        "nEdgesOnEdge": n_edges_on_edge, "edgesOnEdge": edges_on_edge,
        "edgeMask": edge_mask, "cellMask": cell_mask,
        "layerInterfaceSigma": sigma_i, "layerCenterSigma": sigma_c,
        "dvEdge": uniform(2.0e3, 2.0e4, ne),
        "baryWeightsOnVertex": uniform(-0.1, 0.8, (nv, 3)),
        "kiteAreasOnVertex": uniform(1.0e6, 5.0e7, (nv, 3)),
        "weightsOnEdge": uniform(-0.5, 0.5, (ne, me2)),
        "thickness": thickness,
        "upperSurface": uniform(0.0, 3500.0, nc),
        "normalSlopeEdge": uniform(-0.03, 0.03, ne),
        "temperature": uniform(215.0, 276.0, (nc, nl)),
        # prior words of every output
        "normalVelocity": uniform(-1.0, 1.0, (ne, nl + 1)),
        "flowParamA": uniform(-1.0, 1.0, (nc, nl)),
        "slopeEdge": uniform(-1.0, 1.0, ne),
        "tangentSlopeEdge": uniform(-1.0, 1.0, ne),
        "upperSurfaceVertex": uniform(-1.0, 1.0, nv),
    }
    return case


def cases() -> list[dict]:
    return [make_case(*row) for row in CASES]


def _shape(name: str, case: dict) -> tuple[int, ...]:
    return {"normalVelocity": (case["nEdges"], case["nVertLevels"] + 1),
            "flowParamA": (case["nCells"], case["nVertLevels"]),
            "slopeEdge": (case["nEdges"],), "tangentSlopeEdge": (case["nEdges"],),
            "upperSurfaceVertex": (case["nVertices"],)}[name]


def input_bytes(case: dict) -> bytes:
    head = np.array([case["nCells"], case["nEdges"], case["nVertices"], case["nVertLevels"], case["maxEdges2"],
                     case["nEdgesSolve"], TANGENT[case["tangentSlopeCalculation"]], FLOW_A[case["flowParamACalculation"]]], "<i4")
    reals = np.array([case["iceDensity"], case["flowLawExponent"], case["enhancementFactor"],
                      case["defaultFlowParamA"], case["dynamicThickness"]], "<f4")
    parts = [head.tobytes(), reals.tobytes()]
    for name in INT_FIELDS:
        parts.append(np.ascontiguousarray(case[name], "<i4").tobytes())
    for name in REAL_FIELDS + OUTPUT_FIELDS:
        parts.append(np.ascontiguousarray(case[name], "<f4").tobytes())
    return b"".join(parts)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    err, log_errors = np.frombuffer(raw, "<i4", 2, 0)
    out = {"err": np.array([err, log_errors], np.int32)}
    off = 8
    for name in OUTPUT_FIELDS:
        shape = _shape(name, case)
        count = int(np.prod(shape))
        out[name] = np.frombuffer(raw, "<f4", count, off).reshape(shape).copy()
        off += 4 * count
    if off != len(raw):
        raise ValueError(f"oracle output: {len(raw) - off} trailing bytes")
    return out


def load_fixture() -> dict[str, dict[str, np.ndarray]]:
    with np.load(FIXTURE_DIR / "oracle-r4.npz") as data:
        out: dict[str, dict[str, np.ndarray]] = {}
        for key in data.files:
            case, field = key.split("/", 1)
            out.setdefault(case, {})[field] = data[key]
    return out


def bit_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(a, np.float32).view(np.uint32)
                                != np.asarray(b, np.float32).view(np.uint32)))


def run_gpu(case: dict) -> dict[str, np.ndarray]:
    """One woof.hex.mpas_ice.sia_v842.sia_solve call, as run_oracle.F90 makes
    the li_sia_solve call."""
    import cupy as cp
    from woof.hex.mpas_ice import sia_v842 as sia

    dev = {name: cp.asarray(np.ascontiguousarray(case[name])) for name in INT_FIELDS + REAL_FIELDS + OUTPUT_FIELDS}
    sia.sia_solve(mesh=dev, geometry=dev, temperature=dev["temperature"], velocity=dev,
                  n_edges_solve=case["nEdgesSolve"], config_ice_density=case["iceDensity"],
                  config_flowLawExponent=case["flowLawExponent"],
                  config_enhancementFactor=case["enhancementFactor"],
                  config_default_flowParamA=case["defaultFlowParamA"],
                  config_flowParamA_calculation=case["flowParamACalculation"],
                  config_sia_tangent_slope_calculation=case["tangentSlopeCalculation"])
    return {name: cp.asnumpy(dev[name]) for name in OUTPUT_FIELDS}


def measure(reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture() if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases():
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: bit_mismatches(got[name], want[name]) for name in OUTPUT_FIELDS}
    return report
