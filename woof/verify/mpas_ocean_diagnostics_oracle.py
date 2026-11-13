"""Cases and the stream layout for the MPAS-Ocean ``ocn_diagnostic_solve`` oracle.

``tools/mpas_ocean_oracle/run_diagnostics.F90`` drives MPAS-Model v8.4.2's
``ocn_diagnostic_solve`` (cut verbatim out of ``mpas_ocn_diagnostics.F`` with
``ocn_diagnostics_init`` and ``ocn_compute_land_ice_flux_input_fields``),
with the real equation of state and ``ocn_relativeVorticity_circulation``,
compiled with gfortran -O0 -ffp-contract=off at RKIND 4 and 8. Its input
stream is self-describing (dt, every config by name, every field by pool and
name), so this module owns the field table: :data:`FIELDS` names each
field's pool, kind and Fortran shape.

MPAS allocates every cell, edge and vertex field with one more element than
the block has (``nCells+1`` ...): the index a missing neighbour points to.
The routine writes ``-1e34`` sentinels there, so the arrays here have it too.

The meshes are random connectivity, built so that the routine reads no
scratch word it has not written (``maxLevelVertexBot`` covers every level an
edge or cell reads at its vertices; ``normalVelocity`` is zero below
``maxLevelEdgeTop``, where APVM multiplies a gradient v8.4.2 computes only
down to ``maxLevelEdgeTop`` by it). Fields are drawn in float64 from legacy
``RandomState`` and rounded once to the case's precision.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS, bit_mismatches

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_diagnostics")
STRKIND = 512
#: Cases are built at a precision (their reals rounded once to it)
CASES_PER_PRECISION = True
#: Field kinds of the stream: real array, integer array, dimension, dimension
#: array, absent (a package that is off), scratch field
REAL, INT, DIM, DIMS, ABSENT, SCRATCH = 1, 2, 3, 4, 5, 6
#: (pool, name, kind, Fortran shape in the case's dimension names). C1, E1 and
#: V1 are nCells+1, nEdges+1 and nVertices+1; LP1 is nVertLevels+1.
FIELDS = (
    ("mesh", "nVertLevels", DIM, ()), ("mesh", "vertexDegree", DIM, ()), ("mesh", "nCells", DIM, ()),
    ("mesh", "nEdges", DIM, ()), ("mesh", "nVertices", DIM, ()),
    ("mesh", "nCellsArray", DIMS, (4,)), ("mesh", "nEdgesArray", DIMS, (4,)), ("mesh", "nVerticesArray", DIMS, (4,)),
    ("mesh", "maxLevelCell", INT, ("C1",)), ("mesh", "maxLevelEdgeTop", INT, ("E1",)),
    ("mesh", "maxLevelEdgeBot", INT, ("E1",)), ("mesh", "maxLevelVertexBot", INT, ("V1",)),
    ("mesh", "cellsOnEdge", INT, (2, "E1")), ("mesh", "cellsOnVertex", INT, ("VD", "V1")),
    ("mesh", "verticesOnEdge", INT, (2, "E1")), ("mesh", "nEdgesOnCell", INT, ("C1",)),
    ("mesh", "nEdgesOnEdge", INT, ("E1",)), ("mesh", "edgesOnCell", INT, ("ME", "C1")),
    ("mesh", "edgesOnEdge", INT, ("ME2", "E1")), ("mesh", "edgesOnVertex", INT, ("VD", "V1")),
    ("mesh", "kiteIndexOnCell", INT, ("ME", "C1")), ("mesh", "verticesOnCell", INT, ("ME", "C1")),
    ("mesh", "boundaryCell", INT, ("L", "C1")), ("mesh", "edgeSignOnVertex", INT, ("VD", "V1")),
    ("mesh", "edgeSignOnCell", INT, ("ME", "C1")), ("mesh", "edgeMask", INT, ("L", "E1")),
    ("mesh", "weightsOnEdge", REAL, ("ME2", "E1")), ("mesh", "kiteAreasOnVertex", REAL, ("VD", "V1")),
    ("mesh", "dcEdge", REAL, ("E1",)), ("mesh", "dvEdge", REAL, ("E1",)), ("mesh", "areaCell", REAL, ("C1",)),
    ("mesh", "areaTriangle", REAL, ("V1",)), ("mesh", "bottomDepth", REAL, ("C1",)),
    ("mesh", "fVertex", REAL, ("V1",)), ("mesh", "derivTwo", REAL, (15, 2, "E1")),
    ("mesh", "refBottomDepth", REAL, ("L",)),
    ("state", "layerThickness", REAL, ("L", "C1")), ("state", "normalVelocity", REAL, ("L", "E1")),
    ("state", "ssh", REAL, ("C1",)),
    ("tracers", "index_temperature", DIM, ()), ("tracers", "index_salinity", DIM, ()),
    ("tracers", "activeTracers", REAL, ("T", "L", "C1")),
    ("diagnostics", "zMid", REAL, ("L", "C1")), ("diagnostics", "zTop", REAL, ("L", "C1")),
    ("diagnostics", "divergence", REAL, ("L", "C1")), ("diagnostics", "circulation", REAL, ("L", "V1")),
    ("diagnostics", "relativeVorticity", REAL, ("L", "V1")),
    ("diagnostics", "relativeVorticityCell", REAL, ("L", "C1")),
    ("diagnostics", "normalizedPlanetaryVorticityEdge", REAL, ("L", "E1")),
    ("diagnostics", "normalizedRelativeVorticityEdge", REAL, ("L", "E1")),
    ("diagnostics", "normalizedRelativeVorticityCell", REAL, ("L", "C1")),
    ("diagnostics", "density", REAL, ("L", "C1")), ("diagnostics", "displacedDensity", REAL, ("L", "C1")),
    ("diagnostics", "potentialDensity", REAL, ("L", "C1")),
    ("diagnostics", "montgomeryPotential", REAL, ("L", "C1")), ("diagnostics", "pressure", REAL, ("L", "C1")),
    ("diagnostics", "BruntVaisalaFreqTop", REAL, ("L", "C1")),
    ("diagnostics", "tangentialVelocity", REAL, ("L", "E1")),
    ("diagnostics", "layerThicknessEdge", REAL, ("L", "E1")),
    ("diagnostics", "kineticEnergyCell", REAL, ("L", "C1")),
    ("diagnostics", "vertVelocityTop", REAL, ("LP1", "C1")),
    ("diagnostics", "vertTransportVelocityTop", REAL, ("LP1", "C1")),
    ("diagnostics", "vertGMBolusVelocityTop", REAL, ("LP1", "C1")),
    ("diagnostics", "normalGMBolusVelocity", REAL, ("L", "E1")),
    ("diagnostics", "normalTransportVelocity", REAL, ("L", "E1")),
    ("diagnostics", "gradSSH", REAL, ("E1",)), ("diagnostics", "RiTopOfCell", REAL, ("LP1", "C1")),
    ("diagnostics", "pressureAdjustedSSH", REAL, ("C1",)),
    ("diagnostics", "tracersSurfaceValue", REAL, ("T", "C1")),
    ("diagnostics", "tracersSurfaceLayerValue", REAL, ("T", "C1")),
    ("diagnostics", "normalVelocitySurfaceLayer", REAL, ("E1",)),
    ("diagnostics", "indexSurfaceLayerDepth", REAL, ("C1",)),
    ("diagnostics", "surfaceFluxAttenuationCoefficient", REAL, ("C1",)),
    ("diagnostics", "surfaceFluxAttenuationCoefficientRunoff", REAL, ("C1",)),
    ("diagnostics", "inSituThermalExpansionCoeff", REAL, ("L", "C1")),
    ("diagnostics", "inSituSalineContractionCoeff", REAL, ("L", "C1")),
    ("forcing", "seaIcePressure", REAL, ("C1",)), ("forcing", "atmosphericPressure", REAL, ("C1",)),
    # landIcePressurePKG (config_land_ice_flux_mode 'pressure_only') and frazilIce
    ("forcing", "landIcePressure", REAL, ("C1",)), ("forcing", "landIceDraft", REAL, ("C1",)),
    ("forcing", "landIceFraction", REAL, ("C1",)), ("forcing", "landIceMask", INT, ("C1",)),
    ("forcing", "frazilSurfacePressure", REAL, ("C1",)),
    ("scratch", "kineticEnergyVertex", SCRATCH, ("L", "V1")),
    ("scratch", "kineticEnergyVertexOnCells", SCRATCH, ("L", "C1")),
    ("scratch", "normalizedRelativeVorticityVertex", SCRATCH, ("L", "V1")),
    ("scratch", "normalizedPlanetaryVorticityVertex", SCRATCH, ("L", "V1")),
    ("scratch", "vorticityGradientNormalComponent", SCRATCH, ("L", "E1")),
    ("scratch", "vorticityGradientTangentialComponent", SCRATCH, ("L", "E1")),
)
LAND_ICE_FIELDS = ("landIcePressure", "landIceDraft", "landIceFraction", "landIceMask")
#: Every real and integer array the stream carries, written back in the input's order
ARRAYS = tuple((pool, name) for pool, name, kind, _ in FIELDS if kind in (REAL, INT))
PGRAD_TYPES = ("pressure_and_zmid", "MontgomeryPotential", "MontgomeryPotential_and_density",
               "Jacobian_from_density", "Jacobian_from_TS")

BASE_CONFIG = {
    "config_density0": 1026.0, "config_apvm_scale_factor": 0.0, "config_pressure_gradient_type": "pressure_and_zmid",
    "config_coef_3rd_order": 0.25, "config_cvmix_kpp_surface_layer_averaging": 5.0, "config_use_cvmix_kpp": False,
    "config_flux_attenuation_coefficient": 0.001, "config_flux_attenuation_coefficient_runoff": 10.0,
    "config_include_KE_vertex": False, "config_time_integrator": "split_explicit",
    "config_land_ice_flux_mode": "off", "config_eos_type": "linear", "config_eos_linear_densityref": 1000.0,
    "config_eos_linear_alpha": 0.2, "config_eos_linear_beta": 0.8, "config_eos_linear_Tref": 5.0,
    "config_eos_linear_Sref": 35.0, "config_use_frazil_ice_formation": False,
}

#: (name, seed, nCells, nVertLevels, nested halos, config overrides)
CASES = (
    ("zmid_linear", 20261221, 48, 16, False, {}),
    ("jts_jm_apvm_kpp_land_ice", 20261222, 56, 24, True,
     {"config_pressure_gradient_type": "Jacobian_from_TS", "config_eos_type": "jm", "config_apvm_scale_factor": 0.5,
      "config_use_cvmix_kpp": True, "config_include_KE_vertex": True, "config_land_ice_flux_mode": "pressure_only",
      "config_use_frazil_ice_formation": True}),
    ("montgomery_jm_rk4_kpp_deep", 20261223, 40, 20, True,
     {"config_pressure_gradient_type": "MontgomeryPotential", "config_eos_type": "jm",
      "config_time_integrator": "RK4", "config_include_KE_vertex": True, "config_use_cvmix_kpp": True,
      "config_cvmix_kpp_surface_layer_averaging": 4000.0}),
    ("jdens_linear_apvm", 20261224, 44, 30, False,
     {"config_pressure_gradient_type": "Jacobian_from_density", "config_apvm_scale_factor": 0.25,
      "config_eos_linear_densityref": 1027.0}),
    ("one_level_kpp", 20261225, 32, 1, True,
     {"config_eos_type": "jm", "config_use_cvmix_kpp": True, "config_cvmix_kpp_surface_layer_averaging": 30.0}),
)


def shape(case: dict, fortran_shape: tuple) -> tuple[int, ...]:
    """A field's C-order shape: the Fortran shape's dimension names resolved, reversed."""
    d = case["dims"]
    sizes = {"C1": d["nCells"] + 1, "E1": d["nEdges"] + 1, "V1": d["nVertices"] + 1, "L": d["nVertLevels"],
             "LP1": d["nVertLevels"] + 1, "T": d["nTracers"], "ME": d["maxEdges"], "ME2": d["maxEdges2"],
             "VD": d["vertexDegree"]}
    return tuple(sizes[s] if isinstance(s, str) else s for s in reversed(fortran_shape))


def present(case: dict, name: str) -> bool:
    """Whether a forcing field's package is on (MPAS allocates it only then)."""
    if name in LAND_ICE_FIELDS:
        return case["config"]["config_land_ice_flux_mode"] == "pressure_only"
    if name == "frazilSurfacePressure":
        return case["config"]["config_use_frazil_ice_formation"]
    return True


def make_case(name: str, seed: int, n_cells: int, n_levels: int, nested: bool, overrides: dict,
              precision: str = "r4") -> dict:
    """One deterministic case on a random mesh with an ocean's magnitudes.

    - about one edge in twelve and one vertex corner in fifteen point at the
      missing cell ``nCells+1``, as a boundary does;
    - ``maxLevelCell`` from 1 to ``nVertLevels``, ``maxLevelEdgeTop``/``Bot``
      the min/max of an edge's cells, ``maxLevelVertexBot`` covering every
      level read at a vertex;
    - nested halo extents (``nCellsArray(1) < (2) < (3) < (4)``, the same for
      edges) where ``nested``: each loop's extent shows;
    - thin and thick layers, so the KPP surface layer ends inside a column,
      at its bottom or below it.
    """
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    nc, nl = n_cells, n_levels
    ne, nv, me, me2, vd, nt = 3 * nc, 2 * nc, 7, 14, 3, 3
    config = dict(BASE_CONFIG, **overrides)

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    def ints(lo, hi, shp):
        return rs.randint(lo, hi, shp).astype(np.int32)

    if nested:
        cells_array = [nc - 9, nc - 6, nc - 3, nc]
        edges_array = [ne - 20, ne - 13, ne - 6, ne]
    else:
        cells_array, edges_array = [nc] * 4, [ne] * 4
    vertices_array = [nv - 4, nv - 2, nv, nv]
    mlc = np.zeros(nc + 1, np.int32)
    mlc[:nc] = ints(1, nl + 1, nc)
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
    vertices_on_cell = np.full((nc + 1, me), nv + 1, np.int32)
    kite_index = np.ones((nc + 1, me), np.int32)
    edge_sign_on_cell = np.zeros((nc + 1, me), np.int32)
    for c in range(nc):
        k = n_edges_on_cell[c]
        edges_on_cell[c, :k] = ints(1, ne + 1, k)
        vertices_on_cell[c, :k] = ints(1, nv + 1, k)
        kite_index[c, :k] = ints(1, vd + 1, k)
        edge_sign_on_cell[c, :k] = 2 * ints(0, 2, k) - 1
    n_edges_on_edge = ints(6, 11, ne + 1)
    n_edges_on_edge[ne] = 0
    edges_on_edge = np.full((ne + 1, me2), ne + 1, np.int32)
    for e in range(ne):
        edges_on_edge[e, :n_edges_on_edge[e]] = ints(1, ne + 1, n_edges_on_edge[e])
    edges_on_vertex = ints(1, ne + 1, (nv + 1, vd))
    # maxLevelVertexBot: at least every level an edge or a cell reads at the vertex
    mlvb = np.zeros(nv + 1, np.int32)
    mlvb[:nv] = ints(0, nl + 1, nv)
    for v_col, levels in ((vertices_on_edge[:ne, 0], mleb[:ne]), (vertices_on_edge[:ne, 1], mleb[:ne])):
        np.maximum.at(mlvb, v_col - 1, levels)
    for c in range(nc):
        np.maximum.at(mlvb, vertices_on_cell[c, :n_edges_on_cell[c]] - 1, mlc[c])
    mlvb[nv] = 0
    level = np.arange(nl)[None, :]
    edge_mask = (level < mlet[:, None]).astype(np.int32)

    thickness = uniform(0.5, 120.0, (nc + 1, nl))
    thickness[rs.random_sample((nc + 1, nl)) < 0.2] *= 0.02
    bottom_depth = np.array([thickness[c, :mlc[c]].sum() for c in range(nc + 1)]) * uniform(0.98, 1.02, nc + 1)
    normal_velocity = uniform(-1.0, 1.0, (ne + 1, nl)) * (level < mlet[:, None])
    tracers = uniform(0.0, 1.0, (nc + 1, nl, nt))
    tracers[:, :, 0] = uniform(-1.8, 28.0, (nc + 1, nl))
    tracers[:, :, 1] = uniform(33.0, 37.0, (nc + 1, nl))
    dims = {"nCells": nc, "nEdges": ne, "nVertices": nv, "nVertLevels": nl, "nTracers": nt, "maxEdges": me,
            "maxEdges2": me2, "vertexDegree": vd}
    case = {"name": name, "precision": precision, "dims": dims, "config": config,
            "dt": 300.0 if config["config_time_integrator"] == "split_explicit" else 60.0}
    values = {
        "nVertLevels": nl, "vertexDegree": vd, "nCells": nc, "nEdges": ne, "nVertices": nv,
        "nCellsArray": np.array(cells_array, np.int32), "nEdgesArray": np.array(edges_array, np.int32),
        "nVerticesArray": np.array(vertices_array, np.int32),
        "maxLevelCell": mlc, "maxLevelEdgeTop": mlet, "maxLevelEdgeBot": mleb, "maxLevelVertexBot": mlvb,
        "cellsOnEdge": cells_on_edge, "cellsOnVertex": cells_on_vertex, "verticesOnEdge": vertices_on_edge,
        "nEdgesOnCell": n_edges_on_cell, "nEdgesOnEdge": n_edges_on_edge, "edgesOnCell": edges_on_cell,
        "edgesOnEdge": edges_on_edge, "edgesOnVertex": edges_on_vertex, "kiteIndexOnCell": kite_index,
        "verticesOnCell": vertices_on_cell, "boundaryCell": np.zeros((nc + 1, nl), np.int32),
        "edgeSignOnVertex": (2 * ints(0, 2, (nv + 1, vd)) - 1).astype(np.int32),
        "edgeSignOnCell": edge_sign_on_cell, "edgeMask": edge_mask,
        "weightsOnEdge": uniform(-0.5, 0.5, (ne + 1, me2)), "kiteAreasOnVertex": uniform(1.0e7, 1.5e8, (nv + 1, vd)),
        "dcEdge": uniform(1.0e4, 3.0e4, ne + 1), "dvEdge": uniform(5.0e3, 2.0e4, ne + 1),
        "areaCell": uniform(1.0e8, 8.0e8, nc + 1), "areaTriangle": uniform(5.0e7, 4.0e8, nv + 1),
        "bottomDepth": bottom_depth, "fVertex": uniform(-1.4e-4, 1.4e-4, nv + 1),
        "derivTwo": np.zeros((ne + 1, 2, 15)),
        "refBottomDepth": np.cumsum(uniform(1.0, 200.0, nl)),
        "layerThickness": thickness, "normalVelocity": normal_velocity, "ssh": uniform(-2.0, 2.0, nc + 1),
        "index_temperature": 1, "index_salinity": 2, "activeTracers": tracers,
        "normalTransportVelocity": normal_velocity * uniform(0.9, 1.1, (ne + 1, nl)),
        "normalGMBolusVelocity": uniform(-1.0e-3, 1.0e-3, (ne + 1, nl)) * (level < mlet[:, None]),
        "seaIcePressure": uniform(0.0, 2.0e4, nc + 1), "atmosphericPressure": uniform(9.8e4, 1.03e5, nc + 1),
        "landIcePressure": uniform(0.0, 5.0e6, nc + 1), "landIceDraft": uniform(-500.0, 0.0, nc + 1),
        "landIceFraction": uniform(0.0, 1.0, nc + 1), "landIceMask": ints(-3, 4, nc + 1),
        "frazilSurfacePressure": uniform(0.0, 1.0e3, nc + 1),
    }
    pools = {}
    for pool, field, kind, fshape in FIELDS:
        if kind in (DIM, DIMS):
            value = values[field]
        elif kind == SCRATCH or not present(case, field):
            value = None
        elif field in values:
            value = values[field]
        else:
            # an output the routine writes (some only in part): prior words it must keep or overwrite
            value = uniform(-1.0e3, 1.0e3, shape(case, fshape))
        if kind == REAL:
            value = np.asarray(value).astype(real)
        pools.setdefault(pool, {})[field] = value
    case["pools"] = pools
    return case


def cases(precision: str = "r4") -> list[dict]:
    return [make_case(*row, precision=precision) for row in CASES]


def _text(value: str, width: int) -> bytes:
    raw = value.encode("ascii")
    if len(raw) > width:
        raise ValueError(f"{value!r} is longer than {width} characters")
    return raw.ljust(width, b" ")


def _real(case: dict) -> np.dtype:
    return np.dtype(PRECISIONS[case["precision"]]).newbyteorder("<")


def input_bytes(case: dict) -> bytes:
    """run_diagnostics.F90's input stream for ``case``."""
    real = _real(case)
    config = case["config"]
    parts = [np.array(case["dt"], real).tobytes(), np.array([len(config), len(FIELDS)], "<i4").tobytes()]
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
    for pool, name, kind, fshape in FIELDS:
        value = case["pools"][pool][name]
        stream_kind = kind
        if kind in (REAL, INT) and value is None:
            stream_kind = ABSENT
        rank = len(fshape)
        dims = [shape(case, fshape)[::-1][i] if i < rank else 1 for i in range(3)] if rank else [1, 1, 1]
        code_rank = 4 if (stream_kind == ABSENT and kind == INT) else rank
        parts += [_text(pool, 16), _text(name, 64), np.array([stream_kind, code_rank, *dims], "<i4").tobytes()]
        if stream_kind in (REAL, INT):
            if tuple(np.shape(value)) != shape(case, fshape):
                raise ValueError(f"{pool}/{name} has shape {np.shape(value)}, not {shape(case, fshape)}")
            parts.append(np.ascontiguousarray(value, real if kind == REAL else "<i4").tobytes())
        elif stream_kind == DIM:
            parts.append(np.array(value, "<i4").tobytes())
        elif stream_kind == DIMS:
            parts.append(np.ascontiguousarray(value, "<i4").tobytes())
    return b"".join(parts)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    """run_diagnostics.F90's output stream -> ``pool/name`` arrays (present fields only)."""
    out, off = {}, 0
    kinds = {(pool, name): (kind, fshape) for pool, name, kind, fshape in FIELDS}
    for pool, name in ARRAYS:
        kind, fshape = kinds[(pool, name)]
        if case["pools"][pool][name] is None:
            continue
        dtype = _real(case) if kind == REAL else np.dtype("<i4")
        shp = shape(case, fshape)
        n = int(np.prod(shp))
        out[f"{pool}/{name}"] = np.frombuffer(raw, dtype, n, off).reshape(shp).astype(dtype.newbyteorder("="))
        off += dtype.itemsize * n
    if off != len(raw):
        raise ValueError(f"oracle output: {len(raw) - off} trailing bytes")
    return out


def load_fixture(precision: str) -> dict[str, dict[str, np.ndarray]]:
    return _tend.load_fixture(precision, FIXTURE_DIR)


def run_gpu(case: dict) -> dict[str, np.ndarray]:
    """Run woof.hex.mpas_ocean.diagnostics_v842 on ``case`` as run_diagnostics.F90
    drives the Fortran; the same ``pool/name`` outputs."""
    import cupy as cp
    from woof.hex.mpas_ocean import diagnostics_v842 as dg

    dev = {}
    for pool, name, kind, _ in FIELDS:
        value = case["pools"][pool][name]
        if kind in (REAL, INT) and value is not None:
            dev.setdefault(pool, {})[name] = cp.asarray(np.ascontiguousarray(value))
        elif kind in (DIM, DIMS):
            dev.setdefault(pool, {})[name] = value
        else:
            dev.setdefault(pool, {})[name] = None
    dg.diagnostic_solve(dt=case["dt"], mesh=dev["mesh"], state=dev["state"], tracers=dev["tracers"],
                        diagnostics=dev["diagnostics"], forcing=dev["forcing"], configs=case["config"])
    return {f"{pool}/{name}": cp.asnumpy(dev[pool][name]) for pool, name in ARRAYS
            if dev[pool][name] is not None}


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision):
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: bit_mismatches(got[name], want[name]) for name in want}
    return report
