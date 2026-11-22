"""MALI v8.4.2's 'simple' velocity solver: a prescribed velocity field.

:func:`block_init` is ``li_velocity_simple_block_init``
(``mpas_li_velocity_simple.F``), which ``landice_init_block`` calls once,
after the edge signs and before ``li_calculate_mask_init``:

- the cell velocity: with ``config_simple_velocity_type`` 'uniform', 1000
  m/yr along x (``flowTheta`` 0) on every cell; with 'radial', ``1.2e-3 /
  yr`` times the plane distance from (0, 0) (no periodic image), outward;
  any other value leaves it 0;
- ``normalVelocityInitial`` on the owned edges (0 elsewhere, and on every
  level alike): the two cells' mean velocity dotted with the unit vector from
  ``cellsOnEdge(1)``'s centre to the edge's ``(xEdge, yEdge)``. Where that is
  longer than ``dcEdge`` (the edge point stored across a periodic seam) the
  vector into ``cellsOnEdge(2)``'s centre is taken, negated;
- with ``config_calving`` 'topographic_threshold' the bed beyond 0.9 x 21 km
  of (0, 0) is deepened by ``radius / 18900 m``; with 'floating' the bed
  inside it is set to -880 m. The model writes both into ``bedTopography``
  here, in the velocity solver's initialisation (its circular-shelf test).

That last step is test-case code inside the model: any run with 'simple'
and either calving option has its input bed overwritten around (0, 0),
whatever the domain. 'v842' (the default, ``woof_simple_velocity_bed``)
keeps the model's words; 'corrected' leaves the input bed as it is.

The radial-melting and reconstruction-check branches sit behind ``.false.``
parameters in v8.4.2 and are not carried. Each step (``li_velocity_solve``,
:func:`woof.hex.mpas_ice.time_integration_fe_v842.velocity_solve`) copies
``normalVelocityInitial`` into ``normalVelocity`` and zeroes the owned edges
that are not dynamic ice; the reconstruction follows as for 'sia'.

The initialisation runs on the host at RKIND, one correctly rounded operation
at a time in the Fortran's order (elementwise, no sums), as the forward
mode's other mesh-scale set-up does. ``x**2`` is ``x*x`` (gfortran's
integer power); the parameters (``1000/scyr``, ``1.2e-3/scyr``, ``cos(0)``,
``sin(0)``, ``0.9 * 21000``) are folded at RKIND as gfortran folds them.

The higher-order package's ``dirichletVelocityMask`` is allocated for
'simple' (any solver but 'sia' turns the package on) and read from the
input, but only the higher-order solvers ('L1L2', 'FO', 'Stokes', refused
by name) read its words: ``li_calculate_mask`` takes the pointer and uses
it only in their branches, and no pool shift touches it. It is carried
unchanged.
"""
from __future__ import annotations

import numpy as np

from woof.hex.errors import ConfigurationRefusal
from woof import quirk_corrections

from . import rkind

SOURCE_FILES = ("src/core_landice/mode_forward/mpas_li_velocity_simple.F:66-628",
                "src/core_landice/mode_forward/mpas_li_velocity.F:184-208,505-516,604-612")
#: li_constants' scyr (a 365-day year)
SCYR = 31536000.0
SIMPLE_VELOCITY_TYPES = ("uniform", "radial")
#: woof_simple_velocity_bed: 'v842' (the circular-shelf bed) or 'corrected' (the input bed kept)
SIMPLE_VELOCITY_BED = ("v842", "corrected")
#: the circular-shelf test's parameters
MAX_RADIUS = 21000.0
SPIKE_TOPOGRAPHY = -880.0
#: the fields the solver reads beyond the step's
MESH_FIELDS = ("xEdge", "yEdge")
VELOCITY_FIELDS = ("normalVelocityInitial",)
#: li_mask_ValueDynamicIce
MASK_DYNAMIC_ICE = 2


def check_config(config: dict) -> None:
    """Refuse an unknown ``woof_simple_velocity_bed`` by name."""
    value = quirk_corrections.option(config, "woof_simple_velocity_bed")
    if value not in SIMPLE_VELOCITY_BED:
        raise ConfigurationRefusal(knob="woof_simple_velocity_bed", value=value, reason="not an option of this port",
                                   declaration="woof_simple_velocity_bed = 'v842' | 'corrected'")


def require_fields(*, mesh: dict | None = None, velocity: dict | None = None) -> None:
    """Refuse pools without the fields 'simple' reads, by name (the pools given)."""
    for pool, names, held in (("mesh", MESH_FIELDS, mesh), ("velocity", VELOCITY_FIELDS, velocity)):
        if held is None:
            continue
        for name in names:
            if name not in held:
                raise ConfigurationRefusal(
                    knob=name, value="missing", reason=f"config_velocity_solver = 'simple' reads {pool}/{name}",
                    declaration=f"a {pool} pool with {name}")


def cell_velocity(x_cell: np.ndarray, y_cell: np.ndarray, simple_type: str, real) -> tuple[np.ndarray, np.ndarray]:
    """``uVelocityX``, ``uVelocityY`` on every cell (m/s, RKIND)."""
    r = real
    x, y = np.asarray(x_cell, r), np.asarray(y_cell, r)
    ux, uy = np.zeros(x.shape, r), np.zeros(x.shape, r)
    if simple_type.strip() == "uniform":
        flow_speed = r(1000.0) / r(SCYR)
        ux[:] = flow_speed * r(1.0)      # cos(flowTheta = 0)
        uy[:] = flow_speed * r(0.0)      # sin(flowTheta = 0)
    elif simple_type.strip() == "radial":
        gradient = r(1.2e-3) / r(SCYR)
        dx = x - r(0.0)
        dy = y - r(0.0)
        radius = np.sqrt(dx * dx + dy * dy)
        inside = radius > r(0.0)
        safe = np.where(inside, radius, r(1.0))
        speed = gradient * radius
        ux[:] = np.where(inside, (speed * dx) / safe, r(0.0))
        uy[:] = np.where(inside, (speed * dy) / safe, r(0.0))
    return ux, uy


def normal_velocity_initial(*, ux, uy, x_cell, y_cell, x_edge, y_edge, dc_edge, cells_on_edge, n_edges_solve: int,
                            n_levels: int, real) -> np.ndarray:
    """``normalVelocityInitial`` as ``(nEdges, nVertInterfaces)``."""
    r = real
    n_edges = cells_on_edge.shape[0]
    out = np.zeros((n_edges, n_levels), r)
    c1 = cells_on_edge[:n_edges_solve, 0].astype(np.int64) - 1
    c2 = cells_on_edge[:n_edges_solve, 1].astype(np.int64) - 1
    xe, ye = np.asarray(x_edge, r)[:n_edges_solve], np.asarray(y_edge, r)[:n_edges_solve]
    xc, yc = np.asarray(x_cell, r), np.asarray(y_cell, r)
    half = r(0.5)
    u_edge_x = half * (ux[c1] + ux[c2])
    u_edge_y = half * (uy[c1] + uy[c2])
    n1 = xe - xc[c1]
    n2 = ye - yc[c1]
    magnitude = np.sqrt(n1 * n1 + n2 * n2)
    far = magnitude > np.asarray(dc_edge, r)[:n_edges_solve]
    m1 = -(xe - xc[c2])
    m2 = -(ye - yc[c2])
    n1 = np.where(far, m1, n1)
    n2 = np.where(far, m2, n2)
    magnitude = np.where(far, np.sqrt(m1 * m1 + m2 * m2), magnitude)
    n1 = n1 / magnitude
    n2 = n2 / magnitude
    out[:n_edges_solve, :] = (u_edge_x * n1 + u_edge_y * n2)[:, None]
    return out


def topography(bed: np.ndarray, x_cell, y_cell, config_calving: str, real) -> np.ndarray:
    """The block init's bed for the circular-shelf test (a copy)."""
    r = real
    bed = np.array(bed, r)
    option = config_calving.strip()
    if option not in ("topographic_threshold", "floating"):
        return bed
    x, y = np.asarray(x_cell, r), np.asarray(y_cell, r)
    dx = x - r(0.0)
    dy = y - r(0.0)
    radius = np.sqrt(dx * dx + dy * dy)
    rim = r(0.9) * r(MAX_RADIUS)
    if option == "topographic_threshold":
        beyond = radius > rim
        bed[beyond] = (bed[beyond] * radius[beyond]) / rim
    else:
        bed[radius < rim] = r(SPIKE_TOPOGRAPHY)
    return bed


def block_init(*, mesh: dict, geometry: dict, velocity: dict, config: dict, n_edges_solve: int | None = None) -> None:
    """``li_velocity_simple_block_init`` for one block, in place on device
    pools: ``velocity["normalVelocityInitial"]`` and (calving 'floating' or
    'topographic_threshold') ``geometry["bedTopography"]``."""
    import cupy as cp
    require_fields(mesh=mesh, velocity=velocity)
    precision = rkind.precision_of(geometry["thickness"])
    real = rkind.PRECISIONS[precision]
    host = {k: cp.asnumpy(mesh[k]) for k in ("xCell", "yCell", "xEdge", "yEdge", "dcEdge", "cellsOnEdge")}
    n_edges = host["cellsOnEdge"].reshape(-1, 2).shape[0]
    nes = n_edges if n_edges_solve is None else int(n_edges_solve)
    target = velocity["normalVelocityInitial"]
    n_levels = target.shape[1]
    ux, uy = cell_velocity(host["xCell"], host["yCell"], config.get("config_simple_velocity_type", "uniform"), real)
    words = normal_velocity_initial(
        ux=ux, uy=uy, x_cell=host["xCell"], y_cell=host["yCell"], x_edge=host["xEdge"], y_edge=host["yEdge"],
        dc_edge=host["dcEdge"], cells_on_edge=host["cellsOnEdge"].reshape(-1, 2), n_edges_solve=nes,
        n_levels=n_levels, real=real)
    target[...] = cp.asarray(words)
    check_config(config)
    if quirk_corrections.option(config, "woof_simple_velocity_bed") == "corrected":
        return
    bed = topography(cp.asnumpy(geometry["bedTopography"]), host["xCell"], host["yCell"],
                     config.get("config_calving", "none"), real)
    geometry["bedTopography"][...] = cp.asarray(bed)


def solve(*, velocity: dict, edge_mask, n_edges_solve: int) -> None:
    """``li_velocity_solve``'s 'simple' case, in place: ``normalVelocity`` is
    ``normalVelocityInitial``, zeroed on the owned edges that are not dynamic
    ice."""
    import cupy as cp
    v = velocity
    v["normalVelocity"][...] = v["normalVelocityInitial"]
    dynamic = (edge_mask[:n_edges_solve] & MASK_DYNAMIC_ICE) == MASK_DYNAMIC_ICE
    head = v["normalVelocity"][:n_edges_solve]
    head[...] = cp.where(dynamic[:, None], head, head.dtype.type(0))
