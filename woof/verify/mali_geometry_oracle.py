"""Cases and the stream layout for the MALI geometry-diagnostics oracle.

``tools/mali_oracle/run_geometry.F90`` makes, from MPAS-Model v8.4.2 compiled
verbatim (gfortran -O0 -ffp-contract=off, RKIND 4), the calls the model makes
before thickness advection: ``li_calculate_mask``, ``li_update_geometry``,
``li_calculate_layerThickness`` and ``li_layer_normal_velocity``. This module
writes the driver's input stream, reads its output, builds the cases and runs
the GPU port the same way. The RandomState discipline is that of
:mod:`woof.verify.mpas_seaice_evp_oracle`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mali_geometry")

INT_FIELDS = ("nEdgesOnCell", "cellsOnCell", "cellsOnVertex", "cellsOnEdge", "cellMask", "vertexMask", "edgeMask")
REAL_FIELDS = ("thickness", "bedTopography", "layerThicknessFractions", "dcEdge", "normalVelocity")
REAL_OUTPUTS = ("upperSurface", "lowerSurface", "layerThickness", "layerNormalVelocity")
INT_OUTPUTS = ("cellMask", "vertexMask", "edgeMask")

#: (name, seed, nCells, nVertLevels, velocity solver, sea level, dynamic thickness)
CASES = (
    ("sia_shelf", 20261115, 120, 10, "sia", 0.0, 10.0),
    ("none_thin", 20261116, 80, 6, "none", 50.0, 100.0),
)


def make_case(name: str, seed: int, n_cells: int, n_levels: int, solver: str, sea_level: float,
              dynamic_thickness: float) -> dict:
    """One deterministic case: an ice sheet with grounded and floating ice and
    open ocean on a random-connectivity mesh.

    The case covers ice-free cells, ice thinner than the dynamic threshold,
    floating margins next to open ocean (the calving-front rule), grounding
    lines, layer velocities that average to exactly zero (the CFL bigNumber
    branch), unsolved edges, and initial masks carrying the initial-extent bit
    and stale bits that li_calculate_mask clears.
    """
    rs = np.random.RandomState(seed)
    nc, nl = n_cells, n_levels
    ne, nv, me = 3 * nc, 2 * nc, 7

    def uniform(lo, hi, shape):
        return (lo + (hi - lo) * rs.random_sample(shape)).astype(np.float32)

    n_edges_on_cell = np.array([5, 6, 6, 6, 7], np.int32)[rs.randint(0, 5, nc)]
    cells_on_cell = np.full((nc, me), nc + 3, np.int32)
    for c in range(nc):
        cells_on_cell[c, :n_edges_on_cell[c]] = rs.randint(1, nc + 1, n_edges_on_cell[c])
    thickness = uniform(1.0, 2500.0, nc)
    thickness[rs.random_sample(nc) < 0.25] = 0.0
    thin = rs.random_sample(nc) < 0.1
    thickness[thin] = uniform(0.5, 1.5 * dynamic_thickness, int(thin.sum()))
    fractions = uniform(0.5, 1.5, nl)
    fractions = (fractions / fractions.astype(np.float64).sum()).astype(np.float32)
    velocity = uniform(-1.0e-5, 1.0e-5, (ne, nl + 1))
    zero = rs.random_sample(ne) < 0.1
    velocity[zero, :2] = 0.0
    return {
        "name": name, "nCells": nc, "nEdges": ne, "nVertices": nv, "nVertLevels": nl, "maxEdges": me,
        "velocitySolver": solver, "nEdgesSolve": ne - 5,
        "iceDensity": np.float32(910.0), "oceanDensity": np.float32(1028.0),
        "seaLevel": np.float32(sea_level), "dynamicThickness": np.float32(dynamic_thickness),
        "nEdgesOnCell": n_edges_on_cell, "cellsOnCell": cells_on_cell,
        "cellsOnVertex": rs.randint(1, nc + 1, (nv, 3)).astype(np.int32),
        "cellsOnEdge": rs.randint(1, nc + 1, (ne, 2)).astype(np.int32),
        "cellMask": rs.randint(0, 512, nc).astype(np.int32),
        "vertexMask": rs.randint(0, 512, nv).astype(np.int32),
        "edgeMask": rs.randint(0, 512, ne).astype(np.int32),
        "thickness": thickness, "bedTopography": uniform(-1200.0, 900.0, nc),
        "layerThicknessFractions": fractions, "dcEdge": uniform(1.0e3, 2.0e4, ne),
        "normalVelocity": velocity,
        "upperSurface": uniform(-1.0, 1.0, nc), "lowerSurface": uniform(-1.0, 1.0, nc),
        "layerThickness": uniform(-1.0, 1.0, (nc, nl)),
        "layerNormalVelocity": uniform(-1.0, 1.0, (ne, nl)),
    }


def cases() -> list[dict]:
    return [make_case(*row) for row in CASES]


def _shape(name: str, case: dict) -> tuple[int, ...]:
    return {"cellMask": (case["nCells"],), "vertexMask": (case["nVertices"],), "edgeMask": (case["nEdges"],),
            "upperSurface": (case["nCells"],), "lowerSurface": (case["nCells"],),
            "layerThickness": (case["nCells"], case["nVertLevels"]),
            "layerNormalVelocity": (case["nEdges"], case["nVertLevels"])}[name]


def input_bytes(case: dict) -> bytes:
    head = np.array([case["nCells"], case["nEdges"], case["nVertices"], case["nVertLevels"], case["maxEdges"],
                     1 if case["velocitySolver"] == "sia" else 2, case["nEdgesSolve"]], "<i4")
    reals = np.array([case["iceDensity"], case["oceanDensity"], case["seaLevel"], case["dynamicThickness"]], "<f4")
    parts = [head.tobytes(), reals.tobytes()]
    for name in INT_FIELDS:
        parts.append(np.ascontiguousarray(case[name], "<i4").tobytes())
    for name in REAL_FIELDS + REAL_OUTPUTS:
        parts.append(np.ascontiguousarray(case[name], "<f4").tobytes())
    return b"".join(parts)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    out = {"err": np.frombuffer(raw, "<i4", 3, 0).copy(),
           "minOfMaxAllowableDt": np.frombuffer(raw, "<f4", 1, 12).copy()}
    off = 16
    for name in INT_OUTPUTS:
        count = int(np.prod(_shape(name, case)))
        out[name] = np.frombuffer(raw, "<i4", count, off).reshape(_shape(name, case)).copy()
        off += 4 * count
    for name in REAL_OUTPUTS:
        count = int(np.prod(_shape(name, case)))
        out[name] = np.frombuffer(raw, "<f4", count, off).reshape(_shape(name, case)).copy()
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


def word_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))


def run_gpu(case: dict) -> dict[str, np.ndarray]:
    import cupy as cp
    from woof.hex.mpas_ice import geometry_v842 as geo

    dev = {name: cp.asarray(np.ascontiguousarray(case[name])) for name in INT_FIELDS + REAL_FIELDS + REAL_OUTPUTS}
    geo.calculate_mask(mesh=dev, geometry=dev, config_ice_density=case["iceDensity"],
                       config_ocean_density=case["oceanDensity"], config_sea_level=case["seaLevel"],
                       config_dynamic_thickness=case["dynamicThickness"],
                       config_velocity_solver=case["velocitySolver"])
    geo.update_geometry_and_layer_thickness(mesh=dev, geometry=dev, config_ice_density=case["iceDensity"],
                                            config_ocean_density=case["oceanDensity"],
                                            config_sea_level=case["seaLevel"])
    min_dt = geo.layer_normal_velocity(normal_velocity=dev["normalVelocity"],
                                       layer_normal_velocity=dev["layerNormalVelocity"], dc_edge=dev["dcEdge"],
                                       n_edges_solve=case["nEdgesSolve"])
    out = {name: cp.asnumpy(dev[name]) for name in INT_OUTPUTS + REAL_OUTPUTS}
    out["minOfMaxAllowableDt"] = cp.asnumpy(min_dt)
    return out


def measure(reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture() if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases():
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: word_mismatches(got[name], want[name])
                                for name in INT_OUTPUTS + REAL_OUTPUTS + ("minOfMaxAllowableDt",)}
    return report
