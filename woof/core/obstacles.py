"""Porous immersed obstacles: resolved buildings on the WRF route (opt-in).

``obstacle_opt = 1`` represents buildings and other solid obstacles at LES and
high resolution. It is woof's own option, beyond WRF parity, and never a
default. The geometry is a ``woof-buildings.v1`` morphology
(:mod:`woof.static.buildings`), rasterised once onto the domain's own cells.
The solid volume fraction of each layer comes from the building height
histogram and the heights of the base-state levels above ground
(``phb/g - ht``); the geometry is static, so a restart rebuilds the same
coefficients.

Every momentum face touching an obstacle gets a rate ``k = a/dt + b |U|``:

* ``a = obstacle_relax_factor`` where either mass cell sharing the face is
  solid (solid fraction >= ``obstacle_solid_fraction``). This is a Brinkman
  volume penalisation (Angot, Bruneau and Fabrie 1999) that blocks flow
  through facades and roofs, the C-grid "blocked face" rule;
* ``b = Cd a_f + W``. ``Cd a_f`` is canopy form drag in partly built cells
  (``a_f`` the frontal area density, from the cell's frontal area index
  spread over its building volume profile; Cd = ``obstacle_drag_coefficient``,
  BEP's 0.4). ``W`` is log-law wall stress, ``(kappa/ln(d/z0))^2 / spacing``
  per solid neighbour beside or below an open face, with d half the
  spacing and z0 = ``obstacle_wall_z0``.

The kernel subtracts ``k/(1 + k dt)`` times the coupled momentum from that
face's RK-stage tendency, after the stage's own slow tendencies, every RK
stage. That bounded implicit-rate form is unconditionally stable inside
RK3, and the acoustic steps carry it as part of the held slow forcing, as
they carry advection and the pressure-gradient terms. No acoustic kernel
changes. Without obstacles nothing is built and nothing launches, so the off
path is the code that ran before. Scalars are not masked: advection through
a blocked face vanishes with its velocity, but sub-grid diffusion across it
does not, and buildings exchange no heat or moisture (documented
limitations).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Von Karman constant of the wall law.
KARMAN = 0.4
OBSTACLE_OPTIONS = (0, 1)
_BLOCK = 128


@dataclass
class ObstacleFaces:
    """Device face lists and the host receipt of one domain's obstacles."""

    face: object
    a: object
    b: object
    counts: tuple[int, int, int]
    receipt: dict = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts)


def _wall_coefficient(spacing, z0):
    """Wall-law drag per metre for a wall ``spacing/2`` from the face."""
    d = np.maximum(0.5 * np.asarray(spacing, np.float64), 1.01 * z0)
    return (KARMAN / np.log(d / z0)) ** 2 / np.asarray(spacing, np.float64)


def _shift(field3, offset, axis, periodic):
    """``field3`` moved by ``offset`` cells along ``axis`` (0 beyond an
    open edge)."""
    if periodic:
        return np.roll(field3, -offset, axis=axis)
    out = np.zeros_like(field3)
    n = field3.shape[axis]
    src = [slice(None)] * field3.ndim
    dst = [slice(None)] * field3.ndim
    if offset > 0:
        src[axis], dst[axis] = slice(offset, n), slice(0, n - offset)
    else:
        src[axis], dst[axis] = slice(0, n + offset), slice(-offset, n)
    out[tuple(dst)] = field3[tuple(src)]
    return out


def layer_heights(state) -> np.ndarray:
    """(nz + 1, ny, nx) base-state w-level heights above ground (m)."""
    from woof.core import constants as c
    import cupy as cp

    nz, ny, nx = state.p.shape
    phb = np.asarray(cp.asnumpy(state.phb), np.float64)
    ht = np.asarray(cp.asnumpy(state.ht), np.float64)
    if phb.ndim == 1:
        phb = np.broadcast_to(phb[:, None, None], (nz + 1, ny, nx))
    return phb / c.G - ht[None]


def build_coefficients(morph, z_w, cfg, *, periodic_x: bool,
                       periodic_y: bool, ring: int = 0):
    """Host face coefficients from a morphology and level heights.

    Returns ``(faces_u, faces_v, faces_w, receipt)``; each faces tuple is
    ``(flat_index, a, b)`` over that component's own array, only for faces
    with ``a > 0`` or ``b > 0``.
    """
    nzp1, ny, nx = z_w.shape
    nz = nzp1 - 1
    if morph.shape != (ny, nx):
        raise ValueError(f"buildings product is {morph.shape}, the domain "
                         f"mass grid is {(ny, nx)}")
    dz = np.diff(z_w, axis=0)
    phi = np.zeros((nz, ny, nx))
    for k in range(nz):
        phi[k] = morph.solid_fraction(z_w[k], z_w[k + 1])
    if ring > 0:
        keep = np.zeros((ny, nx), bool)
        keep[ring:ny - ring, ring:nx - ring] = True
        phi *= keep[None]
    phi = np.clip(phi, 0.0, 1.0)
    solid = (phi >= float(cfg.obstacle_solid_fraction)).astype(np.float64)
    # Frontal area density of the open (porous) part: the cell's frontal
    # area index spread over its building volume profile.
    volume = (phi * dz).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(volume[None] > 0, phi / volume[None], 0.0)
    open_cell = 1.0 - solid
    fx = np.asarray(morph.frontal_x, np.float64)
    fy = np.asarray(morph.frontal_y, np.float64)
    if ring > 0:
        fx = fx * keep
        fy = fy * keep
    cd = float(cfg.obstacle_drag_coefficient)
    ax = cd * fx[None] * share * open_cell
    ay = cd * fy[None] * share * open_cell
    z0 = float(cfg.obstacle_wall_z0)
    relax = float(cfg.obstacle_relax_factor)
    wx = float(_wall_coefficient(cfg.dx, z0))
    wy = float(_wall_coefficient(cfg.dy, z0))
    wz = _wall_coefficient(dz, z0)

    def sx(f, o):
        return _shift(f, o, 2, periodic_x)

    def sy(f, o):
        return _shift(f, o, 1, periodic_y)

    def sz(f, o):
        return _shift(f, o, 0, False)

    # u faces: face i lies between cells i-1 (L) and i (R).
    def u_faces(cellf):
        left = sx(cellf, -1)                     # cell i-1 at index i
        right = cellf
        out = np.empty((nz, ny, nx + 1))
        out[..., :nx] = np.maximum(left, right) if cellf is solid else \
            0.5 * (left + right)
        out[..., nx] = out[..., 0] if periodic_x else 0.0
        if not periodic_x:
            out[..., 0] = 0.0
        return out

    def v_faces(cellf):
        south = sy(cellf, -1)
        north = cellf
        out = np.empty((nz, ny + 1, nx))
        out[:, :ny] = np.maximum(south, north) if cellf is solid else \
            0.5 * (south + north)
        out[:, ny] = out[:, 0] if periodic_y else 0.0
        if not periodic_y:
            out[:, 0] = 0.0
        return out

    # Wall neighbours of each cell pair, per open face.
    side_y = 0.5 * (sy(solid, 1) + sy(solid, -1))       # per cell, y sides
    side_x = 0.5 * (sx(solid, 1) + sx(solid, -1))
    below = sz(solid, -1)                               # cell k-1 at k
    u_a = relax * u_faces(solid)
    u_open = u_a == 0
    u_b = u_faces(ax) + u_open * (wy * u_faces(side_y)
                                  + u_faces(below * wz))
    v_a = relax * v_faces(solid)
    v_open = v_a == 0
    v_b = v_faces(ay) + v_open * (wx * v_faces(side_x)
                                  + v_faces(below * wz))
    # w faces k = 1..nz-1 between cells k-1 and k; no porous drag.
    w_a = np.zeros((nz + 1, ny, nx))
    w_b = np.zeros((nz + 1, ny, nx))
    w_a[1:nz] = relax * np.maximum(solid[:-1], solid[1:])
    lateral = (wx * 0.5 * (side_x[:-1] + side_x[1:])
               + wy * 0.5 * (side_y[:-1] + side_y[1:]))
    w_b[1:nz] = (w_a[1:nz] == 0) * lateral

    def listed(a, b):
        hit = np.flatnonzero((a.ravel() > 0) | (b.ravel() > 0))
        return (hit.astype(np.int32), a.ravel()[hit].astype(np.float32),
                b.ravel()[hit].astype(np.float32))

    faces = (listed(u_a, u_b), listed(v_a, v_b), listed(w_a, w_b))
    receipt = dict(
        solid_cells=int(solid.sum()), porous_cells=int(
            ((phi > 0) & (solid == 0)).sum()),
        building_volume_column_sum_m=float(volume.sum()),
        faces=[int(f[0].size) for f in faces],
        max_layer_fraction=float(phi.max()) if phi.size else 0.0)
    return (*faces, receipt)


def _boundary_ring(cfg) -> int:
    if cfg.specified or cfg.nested:
        return int(cfg.spec_bdy_width) + 1
    if cfg.open_x or cfg.open_y:
        return 2
    return 0


def attach(state, cfg, morph=None) -> ObstacleFaces | None:
    """Build and keep the domain's obstacle faces (``state._obstacles``).

    ``morph`` defaults to ``cfg.obstacle_file``.  A morphology with no
    obstacles attaches ``None``: nothing launches.
    """
    import cupy as cp

    if not getattr(cfg, "obstacle_opt", 0):
        state._obstacles = None
        return None
    if morph is None:
        from woof.static.buildings import read_product
        morph = read_product(cfg.obstacle_file)
    periodic_x = not (cfg.open_x or cfg.specified or cfg.nested)
    periodic_y = not (cfg.open_y or cfg.specified or cfg.nested)
    fu, fv, fw, receipt = build_coefficients(
        morph, layer_heights(state), cfg, periodic_x=periodic_x,
        periodic_y=periodic_y, ring=_boundary_ring(cfg))
    counts = (fu[0].size, fv[0].size, fw[0].size)
    receipt.update(sources=list(getattr(morph, "sources", ())),
                   option=int(cfg.obstacle_opt))
    if sum(counts) == 0:
        state._obstacles = None
        state._obstacles_receipt = receipt
        return None
    cat = lambda n: cp.asarray(np.concatenate([fu[n], fv[n], fw[n]]))
    faces = ObstacleFaces(face=cat(0), a=cat(1), b=cat(2), counts=counts,
                          receipt=receipt)
    faces.periodic = (periodic_x, periodic_y)
    state._obstacles = faces
    state._obstacles_receipt = receipt
    return faces


_KERNEL = []


def _kernel():
    if not _KERNEL:
        from woof.core.kernels import load_module_no_ftz
        _KERNEL.append(load_module_no_ftz("obstacle_drag", no_fmad=True)
                       .get_function("obstacle_drag"))
    return _KERNEL[0]


def apply(state, cfg, ru, rv) -> None:
    """Add the obstacle tendency to this RK stage's ``ru_t/rv_t/rw_t``."""
    if not getattr(cfg, "obstacle_opt", 0):
        return
    faces = getattr(state, "_obstacles", False)
    if faces is False:
        faces = attach(state, cfg)
    if faces is None:
        return
    from woof.wrf_exact import ENABLED as WRF_EXACT

    nz, ny, nx = state.p.shape
    nu, nv, nw = faces.counts
    total = nu + nv + nw
    mut = state.total_mu()
    divides = bool(state.has_msf and not WRF_EXACT)
    msft = state.msft if state.has_msf else mut
    periodic_x, periodic_y = faces.periodic
    _kernel()(((total + _BLOCK - 1) // _BLOCK,), (_BLOCK,), (
        np.int32(nx), np.int32(ny), np.int32(nz), np.int32(nu),
        np.int32(nv), np.int32(nw), np.int32(periodic_x),
        np.int32(periodic_y), np.float32(cfg.dt), faces.face, faces.a,
        faces.b, state.u, state.v, state.w, ru, rv, mut, state.c1f,
        state.c2f, msft, np.int32(divides), state.ru_t, state.rv_t,
        state.rw_t))


__all__ = ["KARMAN", "OBSTACLE_OPTIONS", "ObstacleFaces", "apply", "attach",
           "build_coefficients", "layer_heights"]
