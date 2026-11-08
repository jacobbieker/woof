"""Cases and the stream layout for the MPAS-Ocean pressure-gradient and hmix oracle.

``tools/mpas_ocean_oracle/run_pgrad_hmix.F90`` drives MPAS-Model v8.4.2's
``ocn_vel_pressure_grad_tend`` and ``ocn_vel_hmix_tend`` (del2, Leith, del4),
compiled verbatim (gfortran -O0 -ffp-contract=off) at RKIND 4 and 8. This
module writes the driver's input stream, reads its output stream, builds the
cases and runs the GPU port the same way.

The conventions (Fortran-order arrays read as C-order transposes, legacy
``RandomState`` with ``randint`` and ``random_sample`` only, random
connectivity, reals drawn in float64 and rounded to float32 for r4) are those
of :mod:`woof.verify.mpas_ocean_tend_oracle`, whose fixture loader and word
comparison this module reuses.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS, bit_mismatches

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data"
               / "oracles" / "mpas_ocean_pgrad_hmix")
PGRAD_TYPES = ("pressure_and_zmid", "MontgomeryPotential", "MontgomeryPotential_and_density",
               "Jacobian_from_density", "Jacobian_from_TS")
FLAGS = ("config_disable_vel_pgrad", "config_disable_vel_hmix", "config_use_mom_del2",
         "config_use_Leith_del2", "config_use_mom_del4")
REALS = ("config_density0", "config_common_level_weight", "config_mom_del2", "config_mom_del4",
         "config_mom_del4_div_factor", "config_Leith_parameter", "config_Leith_dx", "config_Leith_visc2_max")
INT_FIELDS = ("maxLevelEdgeTop", "maxLevelCell", "maxLevelVertexTop", "nEdgesOnCell", "cellsOnEdge",
              "verticesOnEdge", "edgeMask", "edgesOnVertex", "edgesOnCell", "edgeSignOnVertex", "edgeSignOnCell")
MESH_REAL_FIELDS = ("dcEdge", "dvEdge", "areaCell", "areaTriangle", "meshScalingDel2", "meshScalingDel4",
                    "meshScaling")
STATE_FIELDS = ("pressure", "montgomeryPotential", "zMid", "density", "potentialDensity",
                "inSituThermalExpansionCoeff", "inSituSalineContractionCoeff", "tracers", "divergence",
                "relativeVorticity", "normalVelocity", "tangentialVelocity")
OUT_FIELDS = ("tendPgrad", "tendHmix", "viscosity")

_BASE = {"config_density0": 1026.0, "config_common_level_weight": 0.5, "config_mom_del2": 10.0,
         "config_mom_del4": 5.0e9, "config_mom_del4_div_factor": 1.0, "config_Leith_parameter": 1.0,
         "config_Leith_dx": 15000.0, "config_Leith_visc2_max": 25.0}
_OFF = {"config_disable_vel_pgrad": False, "config_disable_vel_hmix": False, "config_use_mom_del2": False,
        "config_use_Leith_del2": False, "config_use_mom_del4": False}

#: (name, seed, nCells, nVertLevels, pressure-gradient type, (indexT, indexS),
#:  flag overrides, real overrides)
CASES = (
    ("zmid_del2", 20261120, 40, 20, "pressure_and_zmid", (1, 2),
     {"config_use_mom_del2": True}, {}),
    ("mont_leith", 20261121, 36, 24, "MontgomeryPotential", (1, 2),
     {"config_use_Leith_del2": True}, {}),
    ("montd_del4", 20261122, 40, 18, "MontgomeryPotential_and_density", (1, 2),
     {"config_use_mom_del4": True}, {"config_mom_del4_div_factor": 0.3}),
    ("jdens_all", 20261123, 48, 30, "Jacobian_from_density", (1, 2),
     {"config_use_mom_del2": True, "config_use_Leith_del2": True, "config_use_mom_del4": True},
     {"config_density0": 1027.5}),
    ("jts_gamma1", 20261124, 40, 26, "Jacobian_from_TS", (2, 3),
     {"config_use_mom_del2": True, "config_use_mom_del4": True}, {"config_common_level_weight": 1.0}),
    # hmix disabled: viscosity keeps its sentinel words
    ("jts_gamma0_hmix_off", 20261125, 32, 40, "Jacobian_from_TS", (3, 1),
     {"config_disable_vel_hmix": True, "config_use_mom_del2": True}, {"config_common_level_weight": 0.0}),
    # pgrad disabled; del4 asked for with config_mom_del4 = 0 stays off
    ("pgrad_off_leith", 20261126, 32, 16, "Jacobian_from_density", (1, 2),
     {"config_disable_vel_pgrad": True, "config_use_Leith_del2": True, "config_use_mom_del4": True},
     {"config_mom_del4": 0.0}),
)


def make_case(name, seed, n_cells, n_levels, pgrad_type, tracer_index, flags, reals) -> dict:
    """One deterministic case, with magnitudes of a 10-30 km MPAS-Ocean mesh.

    The case covers:

    - edges with ``maxLevelEdgeTop`` 0, some of them with ``edgeMask(1)`` 1,
      so the Jacobian branches' unconditional layer-1 term shows;
    - ``edgeMask`` zeros inside an edge's active column;
    - short ``maxLevelCell`` and ``maxLevelVertexTop`` (del4's sums);
    - both signs of ``edgeSignOnCell`` and ``edgeSignOnVertex``;
    - ``dvEdge`` below ``0.25*dcEdge`` on some edges (del4's max);
    - Leith viscosities on both sides of ``config_Leith_visc2_max``;
    - padding past ``nEdgesOnCell`` that a correct port never reads.
    """
    rs = np.random.RandomState(seed)
    nc, nl = n_cells, n_levels
    ne, nv = 3 * nc, 2 * nc
    me, vd, nt = 7, 3, 3

    def uniform(lo, hi, shape):
        return lo + (hi - lo) * rs.random_sample(shape)

    max_level_cell = rs.randint(max(2, nl // 3), nl + 1, nc).astype(np.int32)
    max_level_vertex_top = rs.randint(0, nl + 1, nv).astype(np.int32)
    max_level_edge_top = rs.randint(0, nl + 1, ne).astype(np.int32)
    max_level_edge_top[rs.random_sample(ne) < 0.08] = 0
    cells_on_edge = rs.randint(1, nc + 1, (ne, 2)).astype(np.int32)
    vertices_on_edge = rs.randint(1, nv + 1, (ne, 2)).astype(np.int32)
    edge_mask = np.zeros((ne, nl), np.int32)
    for e in range(ne):
        edge_mask[e, :max_level_edge_top[e]] = 1
    edge_mask[rs.random_sample((ne, nl)) < 0.03] = 0
    land = np.flatnonzero(max_level_edge_top == 0)
    edge_mask[land[: (len(land) + 1) // 2], 0] = 1
    edges_on_vertex = rs.randint(1, ne + 1, (nv, vd)).astype(np.int32)
    edge_sign_on_vertex = (2 * rs.randint(0, 2, (nv, vd)) - 1).astype(np.int32)
    n_edges_on_cell = np.array([5, 6, 6, 6, 7], np.int32)[rs.randint(0, 5, nc)]
    edges_on_cell = np.full((nc, me), ne + 9, np.int32)
    edge_sign_on_cell = np.zeros((nc, me), np.int32)
    for c in range(nc):
        edges_on_cell[c, :n_edges_on_cell[c]] = rs.randint(1, ne + 1, n_edges_on_cell[c])
        edge_sign_on_cell[c, :n_edges_on_cell[c]] = 2 * rs.randint(0, 2, n_edges_on_cell[c]) - 1
    dc_edge = uniform(1.0e4, 3.0e4, ne)
    dv_edge = uniform(5.0e3, 2.0e4, ne)
    dv_edge[rs.random_sample(ne) < 0.1] *= 0.2
    # zMid: layer midpoints of a column of positive thicknesses, falling with k
    thickness = uniform(2.0, 150.0, (nc, nl))
    z_top = -np.cumsum(thickness, axis=1) + thickness
    z_mid = z_top - 0.5 * thickness
    density = uniform(1020.0, 1030.0, (nc, nl))
    case = {
        "name": name, "nCells": nc, "nEdges": ne, "nVertices": nv, "maxEdges": me, "vertexDegree": vd,
        "nVertLevels": nl, "nTracers": nt, "indexT": tracer_index[0], "indexS": tracer_index[1],
        "pgradType": pgrad_type, "flags": {**_OFF, **flags}, "reals": {**_BASE, **reals},
        "maxLevelEdgeTop": max_level_edge_top, "maxLevelCell": max_level_cell,
        "maxLevelVertexTop": max_level_vertex_top, "nEdgesOnCell": n_edges_on_cell,
        "cellsOnEdge": cells_on_edge, "verticesOnEdge": vertices_on_edge, "edgeMask": edge_mask,
        "edgesOnVertex": edges_on_vertex, "edgesOnCell": edges_on_cell,
        "edgeSignOnVertex": edge_sign_on_vertex, "edgeSignOnCell": edge_sign_on_cell,
        "dcEdge": dc_edge, "dvEdge": dv_edge, "areaCell": uniform(1.0e8, 8.0e8, nc),
        "areaTriangle": uniform(5.0e7, 4.0e8, nv), "meshScalingDel2": uniform(0.3, 1.0, ne),
        "meshScalingDel4": uniform(0.3, 1.0, ne), "meshScaling": uniform(0.3, 1.0, ne),
        "pressure": 1025.0 * 9.8 * -z_mid + uniform(-50.0, 50.0, (nc, nl)),
        "montgomeryPotential": uniform(-1.0e3, 1.0e3, (nc, nl)),
        "zMid": z_mid, "density": density,
        "potentialDensity": uniform(1020.0, 1030.0, (nc, nl)),
        "inSituThermalExpansionCoeff": uniform(0.5e-7, 3.0e-7, (nc, nl)),
        "inSituSalineContractionCoeff": uniform(7.0e-7, 8.0e-7, (nc, nl)),
        # tracers(nTracers, nVertLevels, nCells): C-order (nCells, nVertLevels, nTracers)
        "tracers": np.stack([uniform(-1.8, 28.0, (nc, nl)), uniform(33.0, 37.0, (nc, nl)),
                             uniform(0.0, 1.0, (nc, nl))], axis=2),
        "divergence": uniform(-1.0e-6, 1.0e-6, (nc, nl)),
        "relativeVorticity": uniform(-2.0e-5, 2.0e-5, (nv, nl)),
        "normalVelocity": uniform(-1.2, 1.2, (ne, nl)),
        "tangentialVelocity": uniform(-1.2, 1.2, (ne, nl)),
        "tendPgrad": uniform(-1.0e-4, 1.0e-4, (ne, nl)),
        "tendHmix": uniform(-1.0e-8, 1.0e-8, (ne, nl)),
        "viscosity": uniform(-1.0, 1.0, (ne, nl)),
    }
    return case


def cases() -> list[dict]:
    return [make_case(*row) for row in CASES]


def input_bytes(case: dict, precision: str) -> bytes:
    """run_pgrad_hmix.F90's input stream for ``case`` at ``precision``."""
    real = np.dtype(PRECISIONS[precision]).newbyteorder("<")
    head = np.array([case["nCells"], case["nEdges"], case["nVertices"], case["maxEdges"], case["vertexDegree"],
                     case["nVertLevels"], case["nTracers"], case["indexT"], case["indexS"],
                     PGRAD_TYPES.index(case["pgradType"]) + 1], "<i4")
    flags = np.array([int(case["flags"][k]) for k in FLAGS] + [0], "<i4")
    reals = np.array([case["reals"][k] for k in REALS]).astype(real)
    parts = [head.tobytes(), flags.tobytes(), reals.tobytes()]
    for name in INT_FIELDS:
        parts.append(np.ascontiguousarray(case[name], "<i4").tobytes())
    for name in MESH_REAL_FIELDS + STATE_FIELDS + OUT_FIELDS:
        parts.append(np.ascontiguousarray(case[name]).astype(real).tobytes())
    return b"".join(parts)


def input_digest(case: dict, precision: str) -> str:
    return hashlib.sha256(input_bytes(case, precision)).hexdigest()


def read_outputs(raw: bytes, case: dict, precision: str) -> dict[str, np.ndarray]:
    real = np.dtype(PRECISIONS[precision]).newbyteorder("<")
    shape = (case["nEdges"], case["nVertLevels"])
    n = int(np.prod(shape))
    out, off = {}, 0
    for name in OUT_FIELDS:
        out[name] = np.frombuffer(raw, real, n, off).reshape(shape).astype(PRECISIONS[precision])
        off += real.itemsize * n
    if off != len(raw):
        raise ValueError(f"oracle output: {len(raw) - off} trailing bytes")
    return out


def load_fixture(precision: str) -> dict[str, dict[str, np.ndarray]]:
    return _tend.load_fixture(precision, FIXTURE_DIR)


def run_gpu(case: dict, precision: str) -> dict[str, np.ndarray]:
    """Run woof.hex.mpas_ocean.pgrad_hmix_v842 on ``case`` as run_pgrad_hmix.F90
    drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_ocean import pgrad_hmix_v842 as ph

    dtype = PRECISIONS[precision]
    dev = {name: cp.asarray(np.ascontiguousarray(case[name], np.int32)) for name in INT_FIELDS}
    for name in MESH_REAL_FIELDS + STATE_FIELDS + OUT_FIELDS:
        dev[name] = cp.asarray(np.ascontiguousarray(case[name]).astype(dtype))
    flags, reals = case["flags"], case["reals"]
    ph.vel_pressure_grad_tend(
        mesh=dev, pressure=dev["pressure"], montgomery_potential=dev["montgomeryPotential"], z_mid=dev["zMid"],
        density=dev["density"], potential_density=dev["potentialDensity"], tracers=dev["tracers"],
        index_temperature=case["indexT"], index_salinity=case["indexS"], tend=dev["tendPgrad"],
        config_pressure_gradient_type=case["pgradType"],
        config_common_level_weight=reals["config_common_level_weight"],
        config_density0=reals["config_density0"], config_disable_vel_pgrad=flags["config_disable_vel_pgrad"],
        in_situ_thermal_expansion_coeff=dev["inSituThermalExpansionCoeff"],
        in_situ_saline_contraction_coeff=dev["inSituSalineContractionCoeff"])
    ph.vel_hmix_tend(
        mesh=dev, divergence=dev["divergence"], relative_vorticity=dev["relativeVorticity"],
        viscosity=dev["viscosity"], tend=dev["tendHmix"],
        config_disable_vel_hmix=flags["config_disable_vel_hmix"],
        config_use_mom_del2=flags["config_use_mom_del2"], config_mom_del2=reals["config_mom_del2"],
        config_use_Leith_del2=flags["config_use_Leith_del2"],
        config_Leith_parameter=reals["config_Leith_parameter"], config_Leith_dx=reals["config_Leith_dx"],
        config_Leith_visc2_max=reals["config_Leith_visc2_max"],
        config_use_mom_del4=flags["config_use_mom_del4"], config_mom_del4=reals["config_mom_del4"],
        config_mom_del4_div_factor=reals["config_mom_del4_div_factor"])
    return {name: cp.asnumpy(dev[name]) for name in OUT_FIELDS}


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases():
        got = run_gpu(case, precision)
        want = reference[case["name"]]
        report[case["name"]] = {name: bit_mismatches(got[name], want[name]) for name in want}
    return report
