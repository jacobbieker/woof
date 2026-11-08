"""Cases and the stream layout for the MPAS-Ocean tendency oracle.

``tools/mpas_ocean_oracle`` compiles MPAS-Model v8.4.2's
``mpas_ocn_vel_coriolis.F``, ``mpas_ocn_vel_vadv.F``, ``mpas_ocn_thick_hadv.F``
and ``mpas_ocn_thick_vadv.F`` verbatim (gfortran -O0 -ffp-contract=off) at
RKIND 4 and 8, and drives them with ``run_oracle.F90``. This module writes
the driver's input stream, reads its output stream, builds the cases and runs
the GPU port the same way.

The array convention, the RandomState discipline (``randint`` and
``random_sample`` only, so the bytes do not depend on the platform) and the
random-connectivity meshes are those of :mod:`woof.verify.mpas_seaice_evp_oracle`.
A case draws its reals in float64. The r4 run takes them rounded to float32,
and the r8 run takes them as drawn.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data"
               / "oracles" / "mpas_ocean_tend")
PRECISIONS = {"r4": np.float32, "r8": np.float64}

INT_FIELDS = ("maxLevelEdgeTop", "maxLevelEdgeBot", "maxLevelCell", "nEdgesOnEdge", "nEdgesOnCell",
              "cellsOnEdge", "edgesOnEdge", "edgeMask", "edgesOnCell", "edgeSignOnCell")
MESH_REAL_FIELDS = ("dcEdge", "dvEdge", "areaCell", "weightsOnEdge")
STATE_FIELDS = ("normalizedRelativeVorticityEdge", "normalizedPlanetaryVorticityEdge", "layerThicknessEdge",
                "normalVelocity", "kineticEnergyCell", "vertAleTransportTop")
TEND_FIELDS = ("tendVelCoriolis", "tendVelVadv", "tendThickHadv", "tendThickVadv")

#: (name, seed, nCells, nVertLevels, RK4 integrator)
CASES = (
    ("split", 20261110, 64, 24, False),
    ("rk4_deep", 20261111, 40, 60, True),
)


def make_case(name: str, seed: int, n_cells: int, n_levels: int, rk4: bool) -> dict:
    """One deterministic case, with magnitudes of a 10-30 km MPAS-Ocean mesh.

    The case covers:

    - edges with ``maxLevelEdgeTop`` = 0 (next to land), and edges whose
      ``maxLevelEdgeBot`` is deeper than their top;
    - cells with ``maxLevelCell`` short of ``nVertLevels``;
    - ``edgeMask`` zeros inside an edge's active column;
    - both signs of ``edgeSignOnCell``;
    - padding past ``nEdgesOnEdge`` and ``nEdgesOnCell``, filled with
      out-of-range indices that a correct port never reads.
    """
    rs = np.random.RandomState(seed)
    nc, nl = n_cells, n_levels
    ne = 3 * nc
    me, me2 = 7, 14

    def uniform(lo, hi, shape):
        return lo + (hi - lo) * rs.random_sample(shape)

    max_level_cell = rs.randint(max(1, nl // 3), nl + 1, nc).astype(np.int32)
    max_level_edge_top = rs.randint(0, nl + 1, ne).astype(np.int32)
    max_level_edge_top[rs.random_sample(ne) < 0.08] = 0
    max_level_edge_bot = np.minimum(nl, max_level_edge_top + rs.randint(0, 4, ne)).astype(np.int32)
    n_edges_on_edge = rs.randint(8, me2 + 1, ne).astype(np.int32)
    edges_on_edge = np.full((ne, me2), ne + 9, np.int32)
    for e in range(ne):
        edges_on_edge[e, :n_edges_on_edge[e]] = rs.randint(1, ne + 1, n_edges_on_edge[e])
    cells_on_edge = rs.randint(1, nc + 1, (ne, 2)).astype(np.int32)
    edge_mask = np.zeros((ne, nl), np.int32)
    for e in range(ne):
        edge_mask[e, :max_level_edge_top[e]] = 1
    edge_mask[rs.random_sample((ne, nl)) < 0.03] = 0
    n_edges_on_cell = np.array([5, 6, 6, 6, 7], np.int32)[rs.randint(0, 5, nc)]
    edges_on_cell = np.full((nc, me), ne + 9, np.int32)
    edge_sign_on_cell = np.zeros((nc, me), np.int32)
    for c in range(nc):
        edges_on_cell[c, :n_edges_on_cell[c]] = rs.randint(1, ne + 1, n_edges_on_cell[c])
        edge_sign_on_cell[c, :n_edges_on_cell[c]] = 2 * rs.randint(0, 2, n_edges_on_cell[c]) - 1
    case = {
        "name": name, "nCells": nc, "nEdges": ne, "maxEdges": me, "maxEdges2": me2,
        "nVertLevels": nl, "rk4": bool(rk4),
        "maxLevelEdgeTop": max_level_edge_top, "maxLevelEdgeBot": max_level_edge_bot,
        "maxLevelCell": max_level_cell, "nEdgesOnEdge": n_edges_on_edge, "nEdgesOnCell": n_edges_on_cell,
        "cellsOnEdge": cells_on_edge, "edgesOnEdge": edges_on_edge, "edgeMask": edge_mask,
        "edgesOnCell": edges_on_cell, "edgeSignOnCell": edge_sign_on_cell,
        "dcEdge": uniform(1.0e4, 3.0e4, ne), "dvEdge": uniform(5.0e3, 2.0e4, ne),
        "areaCell": uniform(1.0e8, 8.0e8, nc), "weightsOnEdge": uniform(-0.5, 0.5, (ne, me2)),
        "normalizedRelativeVorticityEdge": uniform(-2.0e-7, 2.0e-7, (ne, nl)),
        "normalizedPlanetaryVorticityEdge": uniform(-1.5e-6, 1.5e-6, (ne, nl)),
        "layerThicknessEdge": uniform(1.0, 250.0, (ne, nl)),
        "normalVelocity": uniform(-1.2, 1.2, (ne, nl)),
        "kineticEnergyCell": uniform(0.0, 0.8, (nc, nl)),
        "vertAleTransportTop": uniform(-2.0e-3, 2.0e-3, (nc, nl + 1)),
        "tendVelCoriolis": uniform(-1.0e-4, 1.0e-4, (ne, nl)),
        "tendVelVadv": uniform(-1.0e-4, 1.0e-4, (ne, nl)),
        "tendThickHadv": uniform(-1.0e-3, 1.0e-3, (nc, nl)),
        "tendThickVadv": uniform(-1.0e-3, 1.0e-3, (nc, nl)),
    }
    return case


def cases() -> list[dict]:
    return [make_case(*row) for row in CASES]


def _shape(name: str, case: dict) -> tuple[int, ...]:
    if name in ("tendThickHadv", "tendThickVadv", "kineticEnergyCell"):
        return (case["nCells"], case["nVertLevels"])
    return (case["nEdges"], case["nVertLevels"])


def input_bytes(case: dict, precision: str) -> bytes:
    """run_oracle.F90's input stream for ``case`` at ``precision``."""
    real = np.dtype(PRECISIONS[precision]).newbyteorder("<")
    head = np.array([case["nCells"], case["nEdges"], case["maxEdges"], case["maxEdges2"],
                     case["nVertLevels"], int(case["rk4"])], "<i4")
    parts = [head.tobytes()]
    for name in INT_FIELDS:
        parts.append(np.ascontiguousarray(case[name], "<i4").tobytes())
    for name in MESH_REAL_FIELDS + STATE_FIELDS + TEND_FIELDS:
        parts.append(np.ascontiguousarray(case[name]).astype(real).tobytes())
    return b"".join(parts)


def input_digest(case: dict, precision: str) -> str:
    return hashlib.sha256(input_bytes(case, precision)).hexdigest()


def read_outputs(raw: bytes, case: dict, precision: str) -> dict[str, np.ndarray]:
    real = np.dtype(PRECISIONS[precision]).newbyteorder("<")
    out, off = {}, 0
    for name in TEND_FIELDS:
        shape = _shape(name, case)
        n = int(np.prod(shape))
        out[name] = np.frombuffer(raw, real, n, off).reshape(shape).astype(PRECISIONS[precision])
        off += real.itemsize * n
    if off != len(raw):
        raise ValueError(f"oracle output: {len(raw) - off} trailing bytes")
    return out


def load_fixture(precision: str, fixture_dir: Path = FIXTURE_DIR) -> dict[str, dict[str, np.ndarray]]:
    """``<case>/<field>`` of a fixture's ``oracle-<precision>.npz`` as nested dicts."""
    with np.load(fixture_dir / f"oracle-{precision}.npz") as data:
        out: dict[str, dict[str, np.ndarray]] = {}
        for key in data.files:
            case, field = key.split("/", 1)
            out.setdefault(case, {})[field] = data[key]
    return out


def bit_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    """Words that differ as bit patterns (-0.0 != 0.0)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype:
        raise ValueError(f"{a.dtype} against {b.dtype}")
    view = np.uint32 if a.dtype == np.float32 else np.uint64
    return int(np.count_nonzero(a.view(view) != b.view(view)))


def run_gpu(case: dict, precision: str) -> dict[str, np.ndarray]:
    """Run woof.hex.mpas_ocean.tend_v842 on ``case`` as run_oracle.F90 drives
    the Fortran: each routine into its own tendency array."""
    import cupy as cp
    from woof.hex.mpas_ocean import tend_v842 as tend

    dtype = PRECISIONS[precision]
    dev = {name: cp.asarray(np.ascontiguousarray(case[name], np.int32)) for name in INT_FIELDS}
    for name in MESH_REAL_FIELDS + STATE_FIELDS + TEND_FIELDS:
        dev[name] = cp.asarray(np.ascontiguousarray(case[name]).astype(dtype))
    integrator = "RK4" if case["rk4"] else "split_explicit"
    tend.vel_coriolis_tend(
        mesh=dev, normalized_relative_vorticity_edge=dev["normalizedRelativeVorticityEdge"],
        normalized_planetary_vorticity_edge=dev["normalizedPlanetaryVorticityEdge"],
        layer_thickness_edge=dev["layerThicknessEdge"], normal_velocity=dev["normalVelocity"],
        kinetic_energy_cell=dev["kineticEnergyCell"], tend=dev["tendVelCoriolis"],
        config_time_integrator=integrator)
    tend.vel_vadv_tend(mesh=dev, normal_velocity=dev["normalVelocity"],
                       layer_thickness_edge=dev["layerThicknessEdge"],
                       vert_ale_transport_top=dev["vertAleTransportTop"], tend=dev["tendVelVadv"],
                       config_vert_coord_movement="uniform_stretching")
    tend.thick_hadv_tend(mesh=dev, normal_velocity=dev["normalVelocity"],
                         layer_thickness_edge=dev["layerThicknessEdge"], tend=dev["tendThickHadv"])
    tend.thick_vadv_tend(mesh=dev, vert_ale_transport_top=dev["vertAleTransportTop"],
                         tend=dev["tendThickVadv"])
    return {name: cp.asnumpy(dev[name]) for name in TEND_FIELDS}


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases():
        got = run_gpu(case, precision)
        want = reference[case["name"]]
        report[case["name"]] = {name: bit_mismatches(got[name], want[name]) for name in want}
    return report
