"""Write the oracle's inputs, run the compiled MPAS routines, pack the fixture.

    pixi run -e default python tools/partial_nest_mpas_oracle/make_fixture.py BUILD_DIR [OUT]

BUILD_DIR is build.sh's (it holds run_oracle_r8 / run_oracle_r4).  The
inputs are the D5 envelope's own meshes (woof.verify.hex_partial_nest_envelope:
the 2562-cell parent and the child culled from the 10242-cell mesh, on the
30 km planet):

* ``bary``: the parent's cells and vertices in binary64 (``cellsOnVertex``
  counterclockwise, :func:`woof.hex.partial_nest_coupling.
  counterclockwise_cells_on_vertex`), the child's cell centres and edge
  midpoints as the points, each point's nearest parent vertex as the search
  origin;
* ``apply``: those cells and weights (binary32) over a two-level random
  field on the parent;
* ``relax``: the child mesh, four levels of random state, driving values and
  tendencies, every cell and edge of bdyMask <= 4 active.

OUT (default tests/data/oracles/partial_nest_mpas/fixture.npz) holds every
input and MPAS's output words; tests/test_hex_partial_nest_oracle.py reruns
woof's operators on the inputs.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _write(d: Path, name: str, a: np.ndarray) -> None:
    np.ascontiguousarray(a).tofile(d / f"{name}.bin")


def _f(a) -> np.ndarray:
    """Fortran 1-based connectivity, invalid entries -> 1 (never read)."""
    a = np.asarray(a, np.int64)
    return np.where((a >= 0), a + 1, 1).astype(np.int32)


def main(argv: list[str]) -> int:
    build = Path(argv[1]).resolve()
    out = Path(argv[2]) if len(argv) > 2 else ROOT / "tests/data/oracles/partial_nest_mpas/fixture.npz"
    sys.path.insert(0, str(ROOT))
    from woof.hex.partial_nest_coupling import CellGeometry, nearest_vertices
    from woof.verify import hex_partial_nest_envelope as E
    paths = E.ensure_meshes()
    pmesh, ppos = E.load_mesh(paths["parent"])
    cmesh, cpos = E.load_mesh(paths["child"], culled_from=paths["fine"])
    pg = CellGeometry.from_mesh(pmesh, ppos)
    fix: dict[str, np.ndarray] = {}

    # -- bary (RKIND 8)
    d = build / "case_bary"
    d.mkdir(exist_ok=True)
    points = np.concatenate([np.stack([cpos[n + "Cell"] for n in "xyz"], 1),
                             np.stack([cpos[n + "Edge"] for n in "xyz"], 1)])
    search = nearest_vertices(pg, points)
    voc = np.asarray(pmesh.arrays["verticesOnCell"])
    noc = np.asarray(pmesh.arrays["nEdgesOnCell"], np.int32)
    (d / "dims.txt").write_text(f"{len(pg.x_cell)} {len(pg.x_vertex)} {voc.shape[1]} {len(points)}\n")
    for i, n in enumerate("xyz"):
        _write(d, n + "Cell", pg.x_cell[:, i])
        _write(d, n + "Vertex", pg.x_vertex[:, i])
        _write(d, n + "Point", points[:, i])
    _write(d, "cellsOnVertex", _f(pg.cells_on_vertex))
    _write(d, "verticesOnCell", _f(voc))
    _write(d, "nEdgesOnCell", noc)
    _write(d, "searchVertex", (search + 1).astype(np.int32))
    _write(d, "sphere_radius", np.array([pg.radius], np.float64))
    subprocess.run([str(build / "run_oracle_r8"), "bary", str(d)], check=True)
    cells = np.fromfile(d / "out_baryCells.bin", np.int32).reshape(-1, 3) - 1
    weights = np.fromfile(d / "out_baryWeights.bin", np.float64).reshape(-1, 3)
    fix.update({"pg_" + n: np.asarray(getattr(pg, n)) for n in (
        "x_cell", "x_vertex", "cells_on_vertex", "vertices_on_cell", "n_edges_on_cell",
        "area_cell")})
    fix["pg_radius"] = np.array([pg.radius])
    fix.update(bary_points=points, bary_search=search, bary_cells=cells, bary_weights=weights)

    # -- apply (RKIND 4)
    rng = np.random.default_rng(20261031)
    d = build / "case_apply"
    d.mkdir(exist_ok=True)
    field = (280.0 + 20.0 * rng.random((2, len(pg.x_cell)))).astype(np.float32)
    w32 = weights.astype(np.float32)
    (d / "dims.txt").write_text(f"{len(pg.x_cell)} {len(points)} 2\n")
    _write(d, "baryCells", (cells + 1).astype(np.int32))
    _write(d, "baryWeights", w32)
    _write(d, "field", field)
    subprocess.run([str(build / "run_oracle_r4"), "apply", str(d)], check=True)
    fix.update(apply_field=field, apply_weights=w32,
               apply_out=np.fromfile(d / "out_fieldPoints.bin", np.float32).reshape(2, -1))

    # -- relax (RKIND 4) on the child mesh
    d = build / "case_relax"
    d.mkdir(exist_ok=True)
    a = cmesh.arrays
    nc, ne, nv = (cmesh.dimensions[k] for k in ("nCells", "nEdges", "nVertices"))
    me, vd = int(np.asarray(a["edgesOnCell"]).shape[1]), int(np.asarray(a["edgesOnVertex"]).shape[1])
    nlev = 4
    coe = np.asarray(a["cellsOnEdge"], np.int64)
    eoc = np.asarray(a["edgesOnCell"], np.int64)
    eov = np.asarray(a["edgesOnVertex"], np.int64)
    voe = np.asarray(a["verticesOnEdge"], np.int64)
    noc_c = np.asarray(a["nEdgesOnCell"], np.int32)
    sign_c = np.zeros((nc, me), np.float32)
    for c in range(nc):
        for s in range(noc_c[c]):
            e = eoc[c, s]
            sign_c[c, s] = 1.0 if coe[e, 0] == c else -1.0
    sign_v = np.zeros((nv, vd), np.float32)
    for v in range(nv):
        for s in range(vd):
            e = eov[v, s]
            if 0 <= e < ne:
                sign_v[v, s] = 1.0 if voe[e, 0] == v else -1.0
    f32 = lambda x: np.asarray(x, np.float32)
    dc, dv = f32(a["dcEdge"]), f32(a["dvEdge"])
    geo = dict(
        dcEdge=dc, dvEdge=dv,
        invDcEdge=(np.float32(1.0) / dc).astype(np.float32),
        invDvEdge=(np.float32(1.0) / dv).astype(np.float32),
        invAreaCell=(np.float32(1.0) / f32(a["areaCell"])).astype(np.float32),
        invAreaTriangle=(np.float32(1.0) / f32(a["areaTriangle"])).astype(np.float32),
        meshScalingRegionalCell=(1.0 + rng.random(nc)).astype(np.float32),
        meshScalingRegionalEdge=(1.0 + rng.random(ne)).astype(np.float32),
        edgesOnCell_sign=sign_c, edgesOnVertex_sign=sign_v)
    conn = dict(cellsOnEdge=coe, verticesOnEdge=voe, edgesOnCell=eoc, edgesOnVertex=eov)
    active_c = (np.asarray(a["bdyMaskCell"]) <= 4).astype(np.int32)
    active_e = (np.asarray(a["bdyMaskEdge"]) <= 4).astype(np.int32)
    state = dict(
        ru=f32(rng.normal(0.0, 5.0, (nlev, ne))),
        theta_m=f32(300.0 + rng.normal(0.0, 1.0, (nlev, nc))),
        rho_zz=f32(1.0 + 0.05 * rng.random((nlev, nc))),
        tend_ru=f32(rng.normal(0.0, 1e-3, (nlev, ne))),
        tend_rt=f32(rng.normal(0.0, 1e-2, (nlev, nc))),
        tend_rho=f32(rng.normal(0.0, 1e-5, (nlev, nc))),
        ru_driving=f32(rng.normal(0.0, 5.0, (nlev, ne))),
        rt_driving=f32(300.0 + rng.normal(0.0, 1.0, (nlev, nc))),
        rho_driving=f32(1.0 + 0.05 * rng.random((nlev, nc))))
    scalars = np.array([6.0, 6.0], np.float32)          # dt, divdamp coef
    (d / "dims.txt").write_text(f"{nc} {ne} {nv} {me} {vd} {nlev}\n")
    for k, v in geo.items():
        _write(d, k, v)
    for k, v in conn.items():
        _write(d, k, _f(v))
    _write(d, "nEdgesOnCell", noc_c)
    _write(d, "activeCell", active_c)
    _write(d, "activeEdge", active_e)
    for k, v in state.items():
        _write(d, k, v.T)      # Fortran (k, i)
    _write(d, "scalars", scalars)
    subprocess.run([str(build / "run_oracle_r4"), "relax", str(d)], check=True)
    for name, n in (("tend_ru", ne), ("tend_rt", nc), ("tend_rho", nc)):
        fix["relax_out_" + name] = np.fromfile(d / f"out_{name}.bin", np.float32).reshape(n, nlev).T.copy()
    fix.update({"relax_" + k: v for k, v in geo.items()})
    fix.update({"relax_" + k: np.asarray(v, np.int32) for k, v in conn.items()})
    fix.update({"relax_" + k: v for k, v in state.items()})
    fix.update(relax_nEdgesOnCell=noc_c, relax_activeCell=active_c, relax_activeEdge=active_e,
               relax_scalars=scalars)
    # -- the fine-to-coarse remap table (gate b': no MPAS routine; the test
    #    holds the device kernel to the binary64 reference and the reference
    #    to conservation)
    from woof.hex.partial_nest_coupling import FeedbackRemap
    cg = CellGeometry.from_mesh(cmesh, cpos)
    interior = np.flatnonzero(np.asarray(cmesh.arrays["bdyMaskCell"]) == 0)
    remap = FeedbackRemap.build(pg, cg, interior)
    fix.update(remap_targets=remap.targets, remap_row_ptr=remap.row_ptr, remap_cols=remap.cols,
               remap_overlap=remap.overlap, remap_coverage=remap.coverage,
               remap_rho=f32(1.0 + 0.1 * rng.random((3, nc))),
               remap_q=f32(300.0 + rng.normal(0.0, 2.0, (3, nc))),
               remap_parent_cells=np.array([len(pg.x_cell)], np.int64))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **fix)
    (out.parent / "TOOLCHAIN.txt").write_text((build / "TOOLCHAIN.txt").read_text())
    print(out, hashlib.sha256(out.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
