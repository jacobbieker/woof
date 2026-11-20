"""Porous immersed obstacles on the hex dycore (``--obstacle-opt 1``, opt-in).

The phase-1 method of :mod:`woof.core.obstacles` (resolved buildings on the
WRF route) moved onto the MPAS C-grid: the edge-normal momentum ``rho_u`` on
the Voronoi edges and the vertical momentum ``rho_w`` on the cells' w faces.
It is woof's own option, beyond WRF and MPAS parity, and never a default.

GEOMETRY.  A ``woof-buildings.v1`` morphology aggregated onto the mesh's own
cells (:class:`woof.static.buildings.VoronoiGrid`, shape ``(1, nCells)``).
Each layer's solid fraction comes from the building height histogram and the
cell's level heights above ground (``zgrid - zgrid[0]``); a layer is solid at
``obstacle_solid_fraction``.  An edge takes its values from the two cells
sharing it (the C-grid blocked-face rule of phase 1).

RATE.  Every face touching an obstacle gets ``k = a/dt + b |U|``:

* ``a = obstacle_relax_factor`` where either cell (edge) or either layer
  (w face) is solid: Brinkman volume penalisation;
* ``b = Cd a_n + W``.  ``a_n`` is the frontal area density for flow along
  the edge normal, ``|cos t| lambda_fx + |sin t| lambda_fy`` (``t`` the
  edge's ``angleEdge``, x/y local east/north) spread over the cell's
  building volume profile, mean of the two cells, open layers only.  ``W``
  is the log-law wall stress of the walls of the two cells:
  ``(kappa/ln(d/z0))^2 dvEdge / areaCell |sin(t - t')|`` per solid
  neighbour across edge ``t'`` (``d = dcEdge/2``; the wall parallel to the
  flow drags it, the facing wall is the penalisation's), plus
  ``(kappa/ln(dz/2z0))^2 / dz`` over a solid layer below.  w faces take the
  side walls only (vertical flow is parallel to all of them).

The RK-stage tendency takes ``k/(1 + k dt)`` times the stage's coupled
momentum, after the stage's own slow terms (edges: after the saved Euler
pool, physics and Held-Suarez; w: after the saved Euler w and before the
omega conversion), every RK stage of every dynamics subcycle, with ``dt``
that subcycle's step.  The acoustic steps carry it as held slow forcing; no
acoustic kernel changes.  ``|U|`` on an edge is ``sqrt(u^2 + v^2 + w^2)``
from the stage's normal velocity, MPAS's reconstructed tangential velocity
and the four w values around the edge; on a w face it is ``w`` and the two
layers' cell speed from MPAS's kinetic-energy form.

OFF.  ``CudaDryDycoreDriver.obstacles`` is None: the stage loop tests one
attribute, nothing is imported, allocated or launched.

The ground is not an obstacle (the surface layer or the LES surface owns it);
the idealised cases (``woof.verify.cases.hex_obstacles``) ask for the same
wall law on the lowest layer with ``ground_wall=True``.  Scalars are not
masked and buildings exchange no heat or moisture (phase 1's limitations).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

#: Phase-1 key names and defaults (woof.config.RunConfig obstacle_*).
DEFAULTS = dict(obstacle_solid_fraction=0.5, obstacle_relax_factor=10.0,
                obstacle_drag_coefficient=0.4, obstacle_wall_z0=0.01)
#: As phase 1: above this spacing the urban canopy schemes are the tool.
MAX_SPACING_M = 500.0
_BLOCK = 128

_CUDA_SOURCE = r'''
// woof's opt-in porous immersed obstacles on the MPAS C-grid (not MPAS).
// Sparse face lists; k = a/dt + b|U|; tend -= k/(1+k dt) * momentum.
// Scalar operation order of woof/core/kernels/obstacle_drag.cu, one
// round-to-nearest intrinsic per operation, no contraction.
extern "C" __global__ void hex_obstacle_edges_f32(
    const int nlev, const int nedges, const int ncells, const int nfaces,
    const float dt, const int *face, const float *a, const float *b,
    const int *cells_on_edge, const float *u, const float *v,
    const float *w, const float *momentum, float *tend)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= nfaces) return;
    const int idx = face[n];
    const int k = idx / nedges;
    const int e = idx - k * nedges;
    const int c1 = cells_on_edge[2 * e];
    const int c2 = cells_on_edge[2 * e + 1];
    const float uu = u[idx];
    const float vv = v[idx];
    const float ww = __fmul_rn(0.25f, __fadd_rn(
        __fadd_rn(w[k * ncells + c1], w[k * ncells + c2]),
        __fadd_rn(w[(k + 1) * ncells + c1], w[(k + 1) * ncells + c2])));
    const float speed = __fsqrt_rn(__fadd_rn(__fadd_rn(__fmul_rn(uu, uu),
        __fmul_rn(vv, vv)), __fmul_rn(ww, ww)));
    const float rate = __fadd_rn(__fdiv_rn(a[n], dt), __fmul_rn(b[n], speed));
    if (!(rate > 0.0f)) return;
    const float keff = __fdiv_rn(rate, __fadd_rn(1.0f, __fmul_rn(rate, dt)));
    tend[idx] = __fsub_rn(tend[idx], __fmul_rn(keff, momentum[idx]));
}

extern "C" __global__ void hex_obstacle_wfaces_f32(
    const int nlev, const int nedges, const int ncells, const int max_edges,
    const int nfaces, const float dt, const int *face, const float *a,
    const float *b, const int *n_edges_on_cell, const int *edges_on_cell,
    const float *ke_weight, const float *u, const float *w,
    const float *momentum, float *tend)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= nfaces) return;
    const int idx = face[n];
    const int k = idx / ncells;              // 1 .. nlev-1
    const int c = idx - k * ncells;
    float below = 0.0f;
    float above = 0.0f;
    for (int s = 0; s < n_edges_on_cell[c]; ++s) {
        const int e = edges_on_cell[c * max_edges + s];
        const float weight = ke_weight[c * max_edges + s];
        const float ul = u[(k - 1) * nedges + e];
        const float uh = u[k * nedges + e];
        below = __fadd_rn(below, __fmul_rn(weight, __fmul_rn(ul, ul)));
        above = __fadd_rn(above, __fmul_rn(weight, __fmul_rn(uh, uh)));
    }
    const float ww = w[idx];
    const float horizontal = __fmul_rn(0.5f, __fadd_rn(below, above));
    const float speed = __fsqrt_rn(__fadd_rn(horizontal, __fmul_rn(ww, ww)));
    const float rate = __fadd_rn(__fdiv_rn(a[n], dt), __fmul_rn(b[n], speed));
    if (!(rate > 0.0f)) return;
    const float keff = __fdiv_rn(rate, __fadd_rn(1.0f, __fmul_rn(rate, dt)));
    tend[idx] = __fsub_rn(tend[idx], __fmul_rn(keff, momentum[idx]));
}
'''


@dataclass(frozen=True)
class ObstacleSettings:
    """The run's obstacle switches: phase 1's keys, phase 1's defaults."""

    obstacle_file: str = ""
    obstacle_solid_fraction: float = DEFAULTS["obstacle_solid_fraction"]
    obstacle_relax_factor: float = DEFAULTS["obstacle_relax_factor"]
    obstacle_drag_coefficient: float = DEFAULTS["obstacle_drag_coefficient"]
    obstacle_wall_z0: float = DEFAULTS["obstacle_wall_z0"]

    def __post_init__(self):
        s = float(self.obstacle_solid_fraction)
        if not (math.isfinite(s) and 0.0 < s <= 1.0):
            raise ValueError(f"obstacle_solid_fraction {s!r} must lie in (0, 1]")
        for name in ("obstacle_relax_factor", "obstacle_drag_coefficient",
                     "obstacle_wall_z0"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} {value!r} must be finite and positive")

    def as_dict(self) -> dict:
        return dict(obstacle_opt=1, obstacle_file=str(self.obstacle_file),
                    obstacle_solid_fraction=float(self.obstacle_solid_fraction),
                    obstacle_relax_factor=float(self.obstacle_relax_factor),
                    obstacle_drag_coefficient=float(self.obstacle_drag_coefficient),
                    obstacle_wall_z0=float(self.obstacle_wall_z0))


@dataclass
class HostMesh:
    """The few mesh arrays the coefficients read (host copies, 0-based)."""

    cells_on_edge: np.ndarray       # (nEdges, 2)
    edges_on_cell: np.ndarray       # (nCells, maxEdges)
    n_edges_on_cell: np.ndarray     # (nCells,)
    dc_edge: np.ndarray
    dv_edge: np.ndarray
    area_cell: np.ndarray
    angle_edge: np.ndarray
    z_w: np.ndarray                 # (nlev + 1, nCells) m above ground
    valid_cell: np.ndarray          # (nCells,) bool: may hold obstacles

    @property
    def n_cells(self) -> int:
        return int(self.area_cell.size)

    @property
    def n_edges(self) -> int:
        return int(self.dc_edge.size)


def _wall(spacing, z0):
    from woof.core.obstacles import _wall_coefficient
    return _wall_coefficient(spacing, z0)


def _wall_law(d_spacing, z0):
    """``(kappa/ln((spacing/2)/z0))^2`` (phase 1's law, without the 1/spacing)."""
    return _wall(d_spacing, z0) * np.asarray(d_spacing, np.float64)


def build_coefficients(mesh: HostMesh, morph, settings: ObstacleSettings, *,
                       ground_wall: bool = False):
    """Host face coefficients.

    Returns ``(edges, wfaces, receipt)``; ``edges`` is ``(flat k*nEdges+e,
    a, b)`` and ``wfaces`` ``(flat k*nCells+c, a, b)``, only faces with
    ``a > 0`` or ``b > 0``.
    """
    nc, ne = mesh.n_cells, mesh.n_edges
    if tuple(morph.shape) != (1, nc):
        raise ValueError(f"the buildings product is {tuple(morph.shape)}; this mesh "
                         f"needs a hex product of shape (1, {nc}) "
                         "(woof.static.buildings.VoronoiGrid)")
    z_w = np.asarray(mesh.z_w, np.float64)
    nlev = z_w.shape[0] - 1
    dz = np.diff(z_w, axis=0)
    valid = np.asarray(mesh.valid_cell, bool)
    phi = np.zeros((nlev, nc))
    for k in range(nlev):
        phi[k] = morph.solid_fraction(z_w[k][None], z_w[k + 1][None])[0]
    phi = np.clip(phi * valid[None], 0.0, 1.0)
    solid = phi >= float(settings.obstacle_solid_fraction)
    volume = (phi * dz).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(volume[None] > 0, phi / volume[None], 0.0)
    open_cell = ~solid
    z0 = float(settings.obstacle_wall_z0)
    relax = float(settings.obstacle_relax_factor)
    cd = float(settings.obstacle_drag_coefficient)

    coe = np.asarray(mesh.cells_on_edge, np.int64)
    eoc = np.asarray(mesh.edges_on_cell, np.int64)
    neoc = np.asarray(mesh.n_edges_on_cell, np.int64)
    angle = np.asarray(mesh.angle_edge, np.float64)
    inside = (coe >= 0) & (coe < nc)
    c1 = np.where(inside[:, 0], coe[:, 0], 0)
    c2 = np.where(inside[:, 1], coe[:, 1], 0)
    edge_ok = inside.all(axis=1) & valid[c1] & valid[c2]

    # Per (cell, slot): the wall across that slot's edge, per unit volume of
    # the cell, and the cell beyond it.
    max_edges = eoc.shape[1]
    slot_used = np.arange(max_edges)[None, :] < neoc[:, None]
    slot_edge = np.where(slot_used, eoc, 0)
    other = np.where(coe[slot_edge, 0] == np.arange(nc)[:, None],
                     coe[slot_edge, 1], coe[slot_edge, 0])
    other_ok = slot_used & (other >= 0) & (other < nc)
    other = np.where(other_ok, other, 0)
    dc = np.asarray(mesh.dc_edge, np.float64)
    dv = np.asarray(mesh.dv_edge, np.float64)
    area = np.asarray(mesh.area_cell, np.float64)
    # Wall laws only where a wall stands (and z0 is checked only there): an
    # edge to a cell that is solid in some layer, a layer over a solid one,
    # and the ground row when asked.  A padded column's zero dz, or a thin
    # layer or short edge far from any building, is never evaluated.
    any_solid = solid.any(axis=0)
    touched = np.unique(slot_edge[other_ok & valid[:, None] & any_solid[other]])
    law = np.zeros(ne)
    law[touched] = _wall_law(dc[touched], z0)
    wall = np.where(other_ok, law[slot_edge] * dv[slot_edge] / area[:, None], 0.0)
    needs_wz = np.zeros((nlev, nc), bool)              # (nlev, nc), per metre
    needs_wz[1:] = solid[:-1] & valid[None]
    if ground_wall:
        needs_wz[0] = valid
    wz = np.zeros((nlev, nc))
    wz[needs_wz] = _wall(dz[needs_wz], z0)
    slot_angle = angle[slot_edge]

    frontal = []
    for side in (c1, c2):
        frontal.append(np.abs(np.cos(angle)) * np.asarray(morph.frontal_x, np.float64)[0][side]
                       + np.abs(np.sin(angle)) * np.asarray(morph.frontal_y, np.float64)[0][side])
    # |sin(t_e - t_e')| for each edge against each slot of each side cell.
    sin_side = [np.abs(np.sin(angle[:, None] - slot_angle[side])) for side in (c1, c2)]

    edge_a = np.zeros((nlev, ne))
    edge_b = np.zeros((nlev, ne))
    w_a = np.zeros((nlev + 1, nc))
    w_b = np.zeros((nlev + 1, nc))
    lateral = np.zeros((nlev, nc))
    for k in range(nlev):
        s = solid[k]
        nb_solid = s[other] & other_ok
        lateral[k] = (wall * nb_solid).sum(axis=1)
        chi = (s[c1] | s[c2]) & edge_ok
        edge_a[k] = relax * chi
        porous = 0.5 * cd * (frontal[0] * share[k, c1] * open_cell[k, c1]
                             + frontal[1] * share[k, c2] * open_cell[k, c2])
        side_walls = 0.5 * ((wall[c1] * nb_solid[c1] * sin_side[0]).sum(axis=1)
                            + (wall[c2] * nb_solid[c2] * sin_side[1]).sum(axis=1))
        if k > 0:
            below = solid[k - 1]
            roof = 0.5 * (below[c1] * wz[k, c1] + below[c2] * wz[k, c2])
        elif ground_wall:
            roof = 0.5 * (wz[0, c1] + wz[0, c2])
        else:
            roof = 0.0
        edge_b[k] = np.where(chi | ~edge_ok, 0.0, porous + side_walls + roof)
    w_a[1:nlev] = relax * (solid[:-1] | solid[1:]) * valid[None]
    w_b[1:nlev] = (w_a[1:nlev] == 0) * 0.5 * (lateral[:-1] + lateral[1:]) * valid[None]

    if (edge_a > 0).any() or (edge_b > 0).any():
        used = (edge_a > 0) | (edge_b > 0)
        spacing = float(dc[used.any(axis=0)].max())
        if spacing > MAX_SPACING_M:
            raise ValueError(
                f"obstacles on edges with dcEdge up to {spacing:.0f} m: resolved obstacles "
                f"need a mesh finer than {MAX_SPACING_M:.0f} m where they stand; use the "
                "urban canopy schemes (sf_urban_physics) on coarser meshes")

    def listed(a, b):
        hit = np.flatnonzero((a.ravel() > 0) | (b.ravel() > 0))
        return (hit.astype(np.int32), a.ravel()[hit].astype(np.float32),
                b.ravel()[hit].astype(np.float32))

    edges = listed(edge_a, edge_b)
    wfaces = listed(w_a, w_b)
    receipt = dict(
        solid_cells=int(solid.sum()),
        porous_cells=int(((phi > 0) & ~solid).sum()),
        building_volume_column_sum_m=float(volume.sum()),
        faces=[int(edges[0].size), int(wfaces[0].size)],
        max_layer_fraction=float(phi.max()) if phi.size else 0.0,
        ground_wall=bool(ground_wall))
    return edges, wfaces, receipt


@dataclass
class HexObstacles:
    """Device face lists and the mesh tables the two kernels read."""

    edges: tuple
    wfaces: tuple
    cells_on_edge: object
    edges_on_cell: object
    n_edges_on_cell: object
    ke_weight: object
    shape: tuple                   # (nlev, nEdges, nCells, maxEdges)
    cache: object
    receipt: dict = field(default_factory=dict)
    #: host launch calls; a captured subcycle graph replays them uncounted.
    launches: int = 0

    def _kernel(self, name):
        return self.cache.raw_kernel(name, _CUDA_SOURCE, module_key="hexcore.obstacles")

    def apply_edges(self, tend_ru, rho_u, normal_velocity, tangential_velocity, w, *, dt):
        face, a, b = self.edges
        if face.size == 0:
            return
        nlev, ne, nc, _ = self.shape
        self._kernel("hex_obstacle_edges_f32")(
            ((face.size + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
            (np.int32(nlev), np.int32(ne), np.int32(nc), np.int32(face.size),
             np.float32(dt), face, a, b, self.cells_on_edge, normal_velocity,
             tangential_velocity, w, rho_u, tend_ru))
        self.launches += 1

    def apply_wfaces(self, tend_rw, rho_w, normal_velocity, w, *, dt):
        face, a, b = self.wfaces
        if face.size == 0:
            return
        nlev, ne, nc, max_edges = self.shape
        self._kernel("hex_obstacle_wfaces_f32")(
            ((face.size + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
            (np.int32(nlev), np.int32(ne), np.int32(nc), np.int32(max_edges),
             np.int32(face.size), np.float32(dt), face, a, b,
             self.n_edges_on_cell, self.edges_on_cell, self.ke_weight,
             normal_velocity, w, rho_w, tend_rw))
        self.launches += 1


def host_mesh_from_driver(driver) -> HostMesh:
    """Host copies of the resident mesh and level heights (once, at attach)."""
    cp = driver.cp
    mesh = driver.atmosphere.mesh
    get = lambda value: np.asarray(cp.asnumpy(value))
    nc = int(mesh.n_cells)
    zgrid = get(driver.atmosphere.vertical.zgrid).astype(np.float64)
    if zgrid.ndim != 2 or zgrid.shape[1] != nc:
        raise ValueError(f"zgrid {zgrid.shape} does not hold the mesh's {nc} cells")
    valid = np.ones(nc, bool)
    regional = getattr(driver, "regional_v841", None)
    if regional is not None:
        # A limited-area mesh keeps every boundary ring (bdyMaskCell > 0),
        # the padded column and one more ring free of obstacles, as phase 1
        # keeps the relaxation zone plus one.
        solve = int(regional.n_cells_solve)
        valid[solve:] = False
        mask = get(regional.masks.bdy_mask_cell).ravel()
        valid[:mask.size] &= mask[:nc] == 0
    eoc = get(mesh.edges_on_cell)
    if eoc.shape[0] != nc:
        raise ValueError("the resident edgesOnCell is not (nCells, maxEdges)")
    coe = get(mesh.cells_on_edge).reshape(-1, 2)
    if regional is not None:
        neoc = get(mesh.n_edges_on_cell)
        used = np.arange(eoc.shape[1])[None, :] < neoc[:, None]
        pair = coe[np.where(used, eoc, 0)]                       # (nc, maxEdges, 2)
        inside = (pair >= 0) & (pair < nc)
        near_invalid = (inside & ~valid[np.clip(pair, 0, nc - 1)]) | ~inside
        valid &= ~(near_invalid.any(axis=2) & used).any(axis=1)
    return HostMesh(
        cells_on_edge=coe,
        edges_on_cell=eoc, n_edges_on_cell=get(mesh.n_edges_on_cell),
        dc_edge=get(mesh.dc_edge), dv_edge=get(mesh.dv_edge),
        area_cell=get(mesh.area_cell), angle_edge=get(mesh.angle_edge),
        z_w=zgrid - zgrid[0][None], valid_cell=valid)


def refuse_driver(driver) -> None:
    """Name a driver whose RK stage loop is not the shared one."""
    from .cuda_driver import CudaDryDycoreDriver

    if not isinstance(driver, CudaDryDycoreDriver):
        raise ValueError(f"--obstacle-opt acts in the resident hex dycore driver; "
                         f"{type(driver).__name__} is not one")
    if (type(driver)._advance_dynamics_subcycle_v841
            is not CudaDryDycoreDriver._advance_dynamics_subcycle_v841
            and not getattr(type(driver), "obstacles_subcycle", False)):
        raise ValueError(f"--obstacle-opt: the {type(driver).__name__} dycore runs its own RK "
                         "stage loop, which does not carry the obstacle tendency (the RRFS fork "
                         "dycore is a follow-up); run the MPAS-A dycore")
    if driver.source_release == "v8.2.3":
        raise ValueError("--obstacle-opt needs the v8.4.1+ dycore lane")


def attach(driver, settings: ObstacleSettings, morph=None, *, ground_wall=False,
           host_mesh: HostMesh | None = None):
    """Build the faces and set ``driver.obstacles``; returns the receipt.

    ``morph`` defaults to ``settings.obstacle_file``.  A product with no
    obstacle face attaches nothing (``driver.obstacles`` stays None).
    """
    refuse_driver(driver)
    digest = None
    if morph is None:
        from pathlib import Path

        from woof.file_digest import sha256_file
        from woof.static.buildings import read_product
        path = Path(settings.obstacle_file)
        if not path.is_file():
            raise FileNotFoundError(f"--obstacle-file {str(path)!r} does not exist")
        digest = sha256_file(path, block_bytes=1024 * 1024)
        morph = read_product(path)
        if sha256_file(path, block_bytes=1024 * 1024) != digest:
            raise RuntimeError(f"obstacle_file {str(path)!r} changed while it was read")
    mesh = host_mesh_from_driver(driver) if host_mesh is None else host_mesh
    edges, wfaces, receipt = build_coefficients(mesh, morph, settings,
                                                ground_wall=ground_wall)
    receipt.update(settings.as_dict(), product_sha256=digest,
                   sources=list(getattr(morph, "sources", ())))
    if edges[0].size + wfaces[0].size == 0:
        driver.obstacles = None
        return receipt
    cp = driver.cp
    eoc = np.ascontiguousarray(mesh.edges_on_cell, np.int32)
    neoc = np.ascontiguousarray(mesh.n_edges_on_cell, np.int32)
    used = np.arange(eoc.shape[1])[None, :] < neoc[:, None]
    slot = np.where(used, eoc, 0)
    weight = np.where(used, 0.5 * np.asarray(mesh.dc_edge, np.float64)[slot]
                      * np.asarray(mesh.dv_edge, np.float64)[slot]
                      / np.asarray(mesh.area_cell, np.float64)[:, None], 0.0)
    nlev = mesh.z_w.shape[0] - 1
    driver.obstacles = HexObstacles(
        edges=tuple(cp.asarray(x) for x in edges),
        wfaces=tuple(cp.asarray(x) for x in wfaces),
        cells_on_edge=cp.asarray(np.ascontiguousarray(
            np.clip(mesh.cells_on_edge, 0, mesh.n_cells - 1), np.int32)),
        edges_on_cell=cp.asarray(np.ascontiguousarray(slot, np.int32)),
        n_edges_on_cell=cp.asarray(neoc),
        ke_weight=cp.asarray(np.ascontiguousarray(weight, np.float32)),
        shape=(nlev, mesh.n_edges, mesh.n_cells, int(eoc.shape[1])),
        cache=driver.cache, receipt=receipt)
    return receipt


__all__ = ["DEFAULTS", "HexObstacles", "HostMesh", "ObstacleSettings", "attach",
           "build_coefficients", "host_mesh_from_driver", "refuse_driver"]
