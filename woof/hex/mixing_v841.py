"""MPAS-A v8.4.1 deformation-based 2-D Smagorinsky horizontal mixing.

CPU authority for the v8.4.1 mixing formulation, mirrored from the MPAS-A
v8.4.1 native reference source tree that produced the 2026-08-17 reference
control (byte-identical across the gnu and intel build copies of that tree);
all source paths below are relative to that tree:

* deformation weights: ``src/core_atmosphere/mpas_atm_core.F:1620-1850``
  (``atm_initialize_deformation_weights``), spherical branch only, with the
  called helpers ``mpas_sphere_angle`` and ``mpas_arc_length`` from
  ``src/operators/mpas_geometry_utils.F:27-72,131-154``.  The block at core
  lines 1802-1812 (``mpas_plane_angle`` accumulation) is dead code -- every
  ``thetat`` entry it writes is overwritten by the ``atan2`` assignment inside
  the area loop at lines 1814-1822 before any use -- so it is not mirrored;
  this has no floating-point effect.
* eddy viscosity: ``src/core_atmosphere/dynamics/mpas_atm_dissipation_models.F
  :119-204`` (``smagorinsky_2d``), called from
  ``mpas_atm_time_integration.F:6346-6352`` on RK step 1 of every dynamics
  substep with the edge normal velocity ``u`` and the reconstructed edge
  tangential velocity ``v``.
* the u/w/theta applications (``u_dissipation_3d`` at dissipation-models
  lines 577-945, ``w_dissipation_3d`` at 949-1151, and the theta branch of
  ``scalar_dissipation_3d_les`` at 1155-1330) are, for the non-LES
  ``les_model_opt == LES_MODEL_NONE`` / ``v_*_eddy_visc2 == 0`` /
  ``config_mix_scalars = false`` regime this lane runs, term-for-term the
  stencils already mirrored by :mod:`woof.hex.mixing` for v8.2.3, so those
  authorities are reused here.  Two verified-inert differences: (a) the v8.4.1
  ``u_diffusion_les`` extra divergence-gradient term carries
  ``tau_12_factor = 0`` outside LES (dissipation-models line 684-685,711), an
  exact multiply-by-zero add of zero; (b) the theta application multiplies by
  ``prandtl_inv`` (lines 1280,1310) where ``prandtl = 1.0_RKIND``
  (``src/framework/mpas_constants.F:56``), an exact multiply by one.  Neither
  can change any binary32 bit.

The reference native build runs single precision (``RKIND =
selected_real_kind(6)``; both 24-h reference logs print ``Default real
precision: single``), so the execution dtype of this authority is float32 with
identical operation order; the float64 mirror of the same code path is the
pinning scaffold, exactly as elsewhere in the port.

Fine meshes are the exception to that rule for the deformation WEIGHTS
only.  binary32 holds an Earth-centred coordinate to 0.5 m, a fixed quantum
that is 1 % of a 50 m edge, and the native weights built from such
coordinates are off by up to 18 % at 50 m.  Below 500 m spacing the weights
are therefore evaluated in binary64 about each cell centre and cast to the
execution dtype once (``geometry="local64"``, selected automatically; see
:data:`LOCAL64_GEOMETRY_QUANTUM_RATIO`).  The kernels that consume them, and
every coarser mesh, are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import ConfigurationRefusal
from .mixing import (
    DryMixingTendencies,
    _finite_scalar,
    _float_field,
    _mesh_array,
    _same_dtype,
    compute_mesh_mixing_scaling,
    momentum_horizontal_filter_tendency,
    resolve_config_len_disp,
    theta_horizontal_filter_tendency,
    vertical_momentum_horizontal_filter_tendency,
)

FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class V841MixingConfig:
    """Exact works-or-refuses contract for the v8.4.1 Smagorinsky branch.

    Defaults are the native Registry values the natB-24h reference
    integrated, verbatim from that run's ``namelist.atmosphere``.
    """

    config_horiz_mixing: str = "2d_smagorinsky"
    config_len_disp: float = 0.0
    config_visc4_2dsmag: float = 0.05
    config_smagorinsky_coef: float = 0.125
    config_del4u_div_factor: float = 10.0
    config_h_ScaleWithMesh: bool = True
    config_mpas_cam_coef: float = 0.0

    def validate(self) -> None:
        if self.config_horiz_mixing != "2d_smagorinsky":
            raise ConfigurationRefusal(
                "config_horiz_mixing",
                self.config_horiz_mixing,
                "this authority admits the v8.4.1 2-D Smagorinsky branch",
                "config_horiz_mixing='2d_smagorinsky'",
            )
        length = _finite_scalar("config_len_disp", self.config_len_disp)
        if length < 0.0:
            raise ConfigurationRefusal(
                "config_len_disp",
                self.config_len_disp,
                "a negative filter length is not an MPAS-admitted value",
                "config_len_disp>=0 (zero selects nominalMinDc)",
            )
        visc4 = _finite_scalar("config_visc4_2dsmag", self.config_visc4_2dsmag)
        if visc4 < 0.0:
            raise ConfigurationRefusal(
                "config_visc4_2dsmag",
                self.config_visc4_2dsmag,
                "the Registry admits only non-negative fourth-order scaling",
                "config_visc4_2dsmag>=0",
            )
        smag = _finite_scalar(
            "config_smagorinsky_coef", self.config_smagorinsky_coef
        )
        if smag < 0.0:
            raise ConfigurationRefusal(
                "config_smagorinsky_coef",
                self.config_smagorinsky_coef,
                "the admitted Smagorinsky coefficient is non-negative",
                "config_smagorinsky_coef>=0",
            )
        div_factor = _finite_scalar(
            "config_del4u_div_factor", self.config_del4u_div_factor
        )
        if div_factor <= 0.0:
            raise ConfigurationRefusal(
                "config_del4u_div_factor",
                self.config_del4u_div_factor,
                "the Registry requires a positive divergent hyperdiffusion factor",
                "config_del4u_div_factor>0",
            )
        cam = _finite_scalar("config_mpas_cam_coef", self.config_mpas_cam_coef)
        if cam != 0.0:
            raise ConfigurationRefusal(
                "config_mpas_cam_coef",
                self.config_mpas_cam_coef,
                "the CAM-SE upper-level coefficient floor is not ported",
                "config_mpas_cam_coef=0.0",
            )
        if not isinstance(self.config_h_ScaleWithMesh, (bool, np.bool_)):
            raise ConfigurationRefusal(
                "config_h_ScaleWithMesh",
                self.config_h_ScaleWithMesh,
                "the Registry option is logical",
                "config_h_ScaleWithMesh=True or False",
            )


#: Geometry evaluation modes for :func:`initialize_deformation_weights_v841`.
#:
#: ``native``  -- the per-operation mirror of ``atm_initialize_deformation_weights``
#:   evaluated in the requested dtype from the Earth-centred coordinates divided
#:   by the sphere radius, exactly as the single-precision reference build does.
#: ``local64`` -- the same weights, mathematically identical, evaluated in
#:   binary64 from coordinates differenced about each cell centre (a local
#:   tangent plane) and cast to the requested dtype once at the end.
#: ``auto``    -- ``native`` unless the binary32 Earth-centred evaluation is
#:   numerically inadequate for this mesh (see
#:   :data:`LOCAL64_GEOMETRY_QUANTUM_RATIO`), then ``local64``.
DEFORMATION_GEOMETRY_MODES: tuple[str, ...] = ("auto", "native", "local64")

#: ``auto`` switches a binary32 build to ``local64`` when the binary32 spacing
#: of the sphere radius (0.5 m at Earth radius) exceeds this fraction of the
#: mesh's finest spacing (``min(dcEdge)``, the regional class key's own
#: measurement; see :func:`finest_cell_spacing_m`), i.e. below 500 m on Earth.
#:
#: Why there: binary32 Earth-centred coordinates are quantised at 0.5 m
#: whatever the cell size, so the native evaluation's relative weight error
#: grows roughly as 1/dc.  Measured against the closed-form regular hexagon
#: on synthetic equatorial patches (``woof.hex.precision_probe``; held by
#: ``components/hex/tests/test_deformation_geometry_precision.py``), the
#: largest native binary32 weight error relative to the largest weight is
#: 2.5e-3 at 3 km, 1.7e-2 at 711 m, 3.4e-2 at 250 m, 1.0e-1 at 100 m and
#: 1.8e-1 at 50 m; ``local64`` holds 3-9e-8 (binary32 storage of the
#: binary64 value) at every one of those spacings.
#:
#: The switch sits below the finest geometry any regional class was minted
#: on (a 711 m cull), so every minted, anchored and coarse configuration
#: keeps the exact bytes its evidence was measured on, at the native error
#: listed above -- that is the reference build's own error and the mints
#: are evidence about it.  Every sub-500 m mesh (the 50/100/250/380 m lane),
#: where nothing has been minted and the native error reaches 3-18 %, gets
#: binary64 geometry.
LOCAL64_GEOMETRY_QUANTUM_RATIO = 1.0e-3


@dataclass(frozen=True, slots=True)
class DeformationWeightsV841:
    """``deformation_coef_{c2,s2,cs}`` with shape ``(nCells, maxEdges)``.

    ``geometry`` records which evaluation produced the weights (``native`` or
    ``local64``; see :data:`DEFORMATION_GEOMETRY_MODES`) and
    ``finest_spacing_m`` the finest spacing (:func:`finest_cell_spacing_m`)
    the gate read; :func:`initialize_deformation_weights_v841` records it in
    every mode.
    """

    coef_c2: FloatArray
    coef_s2: FloatArray
    coef_cs: FloatArray
    geometry: str = "native"
    finest_spacing_m: float | None = None

    def validate(self, *, n_cells: int, max_edges: int) -> None:
        shape = (n_cells, max_edges)
        for name in ("coef_c2", "coef_s2", "coef_cs"):
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} != {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")


def _sphere_arc_length(
    ax: Any, ay: Any, az: Any, bx: Any, by: Any, bz: Any, one: Any
) -> Any:
    """``mpas_arc_length`` (mpas_geometry_utils.F:131-154)."""

    cx = bx - ax
    cy = by - ay
    cz = bz - az
    r = np.sqrt(ax * ax + ay * ay + az * az)
    c = np.sqrt(cx * cx + cy * cy + cz * cz)
    two = one + one
    return r * two * np.arcsin(c / (two * r))


def _sphere_angle(
    ax: Any, ay: Any, az: Any,
    bx: Any, by: Any, bz: Any,
    cx: Any, cy: Any, cz: Any,
    one: Any,
) -> Any:
    """``mpas_sphere_angle`` (mpas_geometry_utils.F:27-72)."""

    zero = one - one
    half = one / (one + one)
    two = one + one
    a = _sphere_arc_length(bx, by, bz, cx, cy, cz, one)
    b = _sphere_arc_length(ax, ay, az, cx, cy, cz, one)
    c = _sphere_arc_length(ax, ay, az, bx, by, bz, one)
    ab_x = bx - ax
    ab_y = by - ay
    ab_z = bz - az
    ac_x = cx - ax
    ac_y = cy - ay
    ac_z = cz - az
    d_x = (ab_y * ac_z) - (ab_z * ac_y)
    d_y = -((ab_x * ac_z) - (ab_z * ac_x))
    d_z = (ab_x * ac_y) - (ab_y * ac_x)
    s = half * (a + b + c)
    ratio = (np.sin(s - b) * np.sin(s - c)) / (np.sin(b) * np.sin(c))
    sin_angle = np.sqrt(np.minimum(one, np.maximum(zero, ratio)))
    magnitude = two * np.arcsin(np.maximum(np.minimum(sin_angle, one), -one))
    if (d_x * ax + d_y * ay + d_z * az) >= zero:
        return magnitude
    return -magnitude


def _deformation_weights_native(
    mesh: object,
    *,
    dtype: np.dtype[Any] | type[Any] = np.float32,
) -> DeformationWeightsV841:
    """Mirror ``atm_initialize_deformation_weights`` (mpas_atm_core.F:1620-1850).

    Spherical branch only (``on_a_sphere`` true); planar meshes are refused
    fail-closed.  Executed in the requested dtype throughout: float32 is the
    execution mirror of the single-precision native reference build, float64
    is the pinning scaffold.
    """

    out_dtype = np.dtype(dtype)
    if out_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("deformation weights dtype must be float32 or float64")

    radius = out_dtype.type(_require_spherical(mesh))

    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    edges_on_cell = np.asarray(_mesh_array(mesh, "edgesOnCell"), dtype=np.int64)
    cells_on_cell = np.asarray(_mesh_array(mesh, "cellsOnCell"), dtype=np.int64)
    vertices_on_cell = np.asarray(
        _mesh_array(mesh, "verticesOnCell"), dtype=np.int64
    )
    cells_on_edge = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
    x_cell = np.asarray(_mesh_array(mesh, "xCell"), dtype=out_dtype)
    y_cell = np.asarray(_mesh_array(mesh, "yCell"), dtype=out_dtype)
    z_cell = np.asarray(_mesh_array(mesh, "zCell"), dtype=out_dtype)
    x_vertex = np.asarray(_mesh_array(mesh, "xVertex"), dtype=out_dtype)
    y_vertex = np.asarray(_mesh_array(mesh, "yVertex"), dtype=out_dtype)
    z_vertex = np.asarray(_mesh_array(mesh, "zVertex"), dtype=out_dtype)

    n_cells = int(counts.size)
    max_edges = int(edges_on_cell.shape[1])
    if edges_on_cell.shape[0] != n_cells or vertices_on_cell.shape != edges_on_cell.shape:
        raise ValueError("edgesOnCell/verticesOnCell shapes are inconsistent")
    if cells_on_cell.shape != edges_on_cell.shape:
        raise ValueError("cellsOnCell shape is inconsistent with edgesOnCell")

    one = out_dtype.type(1.0)
    zero = out_dtype.type(0.0)
    quarter = out_dtype.type(0.25)
    half = one / (one + one)
    # pii = 2.*asin(1.0), evaluated in the working precision (core line 1698).
    pii = (one + one) * np.arcsin(one)

    coef_c2 = np.zeros((n_cells, max_edges), dtype=out_dtype)
    coef_s2 = np.zeros((n_cells, max_edges), dtype=out_dtype)
    coef_cs = np.zeros((n_cells, max_edges), dtype=out_dtype)

    for cell in range(n_cells):
        count = int(counts[cell])
        if count < 3 or count > max_edges:
            raise ValueError(f"cell {cell} has invalid nEdgesOnCell {count}")
        # Halo guard (core lines 1702-1716): native builds cell_list from
        # the cell and its neighbours and CYCLES -- leaving this cell's
        # weights at their zero initialization -- when any entry reaches
        # outside nCells.  On the serial global mesh that can only mean a
        # corrupt table; on a regional cull it is the ring-7 rows, whose
        # absent-neighbour slots map to the garbage cell, so those cells
        # keep zero weights exactly as the reference executable leaves them
        # (their specified-zone tendencies are overwritten regardless).
        neighbors = cells_on_cell[cell, :count]
        if np.any((neighbors < 0) | (neighbors >= n_cells)):
            continue
        verts = vertices_on_cell[cell, :count]
        if np.any((verts < 0) | (verts >= x_vertex.size)):
            raise ValueError(f"verticesOnCell reaches outside the mesh at cell {cell}")
        slot_edges = edges_on_cell[cell, :count]
        if np.any((slot_edges < 0) | (slot_edges >= cells_on_edge.shape[0])):
            raise ValueError(f"edgesOnCell reaches outside the mesh at cell {cell}")

        # Normalized Cartesian points (core lines 1725-1734).
        cx = x_cell[cell] / radius
        cy = y_cell[cell] / radius
        cz = z_cell[cell] / radius
        vx = x_vertex[verts] / radius
        vy = y_vertex[verts] / radius
        vz = z_vertex[verts] / radius

        # theta_abs (core lines 1742-1750).
        if cz == one:
            theta_abs = pii / (one + one)
        else:
            theta_abs = pii / (one + one) - _sphere_angle(
                cx, cy, cz, vx[0], vy[0], vz[0], zero, zero, one, one
            )

        # thetav / dl_sphere / thetat accumulation (core lines 1760-1772).
        thetat = np.zeros(count, dtype=out_dtype)
        dl_sphere = np.zeros(count, dtype=out_dtype)
        thetav = np.zeros(count, dtype=out_dtype)
        for j in range(count):
            jp1 = (j + 1) % count
            thetav[j] = _sphere_angle(
                cx, cy, cz,
                vx[j], vy[j], vz[j],
                vx[jp1], vy[jp1], vz[jp1],
                one,
            )
            dl_sphere[j] = radius * _sphere_arc_length(
                cx, cy, cz, vx[j], vy[j], vz[j], one
            )
        thetat[0] = theta_abs
        for j in range(1, count):
            thetat[j] = thetat[j - 1] + thetav[j - 1]

        # Tangent-plane vertices (core lines 1776-1779).
        xp = np.cos(thetat) * dl_sphere
        yp = np.sin(thetat) * dl_sphere

        # Cell area and edge-normal angles (core lines 1814-1822).  The
        # preceding mpas_plane_angle block (1802-1812) is dead code -- see the
        # module docstring.
        area_cell = zero
        theta_edge = np.zeros(count, dtype=out_dtype)
        for j in range(count):
            jp1 = (j + 1) % count
            dx = xp[jp1] - xp[j]
            dy = yp[jp1] - yp[j]
            area_cell = (
                area_cell
                + quarter * (xp[j] + xp[jp1]) * (yp[jp1] - yp[j])
                - quarter * (yp[j] + yp[jp1]) * (xp[jp1] - xp[j])
            )
            theta_edge[j] = np.arctan2(dy, dx) - pii / (one + one)

        # Coefficients (core lines 1826-1846).
        for j in range(count):
            jp1 = (j + 1) % count
            dx = xp[jp1] - xp[j]
            dy = yp[jp1] - yp[j]
            dl = np.sqrt(dx * dx + dy * dy)
            sin_t = np.sin(theta_edge[j])
            cos_t = np.cos(theta_edge[j])
            sint2 = sin_t * sin_t
            cost2 = cos_t * cos_t
            sint_cost = sin_t * cos_t
            c2 = dl * cost2 / area_cell
            s2 = dl * sint2 / area_cell
            cs = dl * sint_cost / area_cell
            if int(cells_on_edge[slot_edges[j], 0]) != cell:
                c2 = -c2
                s2 = -s2
                cs = -cs
            coef_c2[cell, j] = c2
            coef_s2[cell, j] = s2
            coef_cs[cell, j] = cs

    _ = half  # parity with the Fortran locals; no further use
    result = DeformationWeightsV841(
        coef_c2=coef_c2, coef_s2=coef_s2, coef_cs=coef_cs
    )
    result.validate(n_cells=n_cells, max_edges=max_edges)
    return result


def _require_spherical(mesh: object) -> float:
    attrs = getattr(mesh, "attrs", {})
    on_sphere = str(attrs.get("on_a_sphere", "NO")).strip().upper() == "YES"
    if not on_sphere:
        raise ConfigurationRefusal(
            "on_a_sphere",
            attrs.get("on_a_sphere"),
            "only the spherical deformation-weight branch is ported",
            "a spherical MPAS mesh",
        )
    radius = float(attrs["sphere_radius"])
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("sphere_radius must be finite and positive")
    return radius


def _xyz64(mesh: object, family: str) -> NDArray[np.float64]:
    return np.stack(
        [
            np.asarray(_mesh_array(mesh, axis + family), dtype=np.float64)
            for axis in "xyz"
        ],
        axis=-1,
    )


def finest_cell_spacing_m(mesh: object) -> float:
    """The mesh's finest cell spacing in metres, as the regional class key
    measures it: ``min(dcEdge)``.

    The same quantity, read the same way, that keys a regional class
    (``_measured_finest_edge_m`` in ``cuda_regional_forecast_v841``), so the
    geometry gate and the class key cannot classify one cull two ways.
    Mesh validation already holds ``dcEdge`` to the coordinates
    (:func:`woof.hex.mesh.spherical_arc_tolerance`).  Only a mesh carrying
    no ``dcEdge`` falls back to the binary64 cell-to-neighbour chord, each
    pair visited once through ``cellsOnEdge`` (regional sentinel slots
    skipped).  Returns ``inf`` when nothing can be measured.
    """

    try:
        dc_edge = np.asarray(_mesh_array(mesh, "dcEdge"), dtype=np.float64)
    except AttributeError:
        dc_edge = None
    if dc_edge is not None and dc_edge.size:
        return float(np.min(dc_edge))
    cells_on_edge = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
    cell_xyz = _xyz64(mesh, "Cell")
    n_cells = cell_xyz.shape[0]
    valid = np.all((cells_on_edge >= 0) & (cells_on_edge < n_cells), axis=1)
    if not np.any(valid):
        return float("inf")
    pairs = cells_on_edge[valid]
    chord = np.linalg.norm(cell_xyz[pairs[:, 1]] - cell_xyz[pairs[:, 0]], axis=-1)
    return float(np.min(chord))


def _deformation_weights_local64(
    mesh: object,
    *,
    dtype: np.dtype[Any],
    radius: float,
) -> DeformationWeightsV841:
    """``atm_initialize_deformation_weights`` in binary64 local coordinates.

    The same weights as the native mirror in exact arithmetic, evaluated so
    that no step differences two Earth-radius-sized numbers in a short
    format:

    * every vertex is taken RELATIVE to its cell centre in binary64
      (``d = x_vertex - x_cell``), so a 50 m offset keeps ~1e-9 m of
      resolution rather than the 0.5 m binary32 spacing of the absolute
      components;
    * the spherical angle at the cell centre between two great-circle arcs
      is the angle between their tangent vectors, which are the projections
      of ``d`` onto the tangent plane; it is evaluated as ``atan2`` of the
      cross and dot products (exact for the same quantity the native
      half-angle formula computes, and well conditioned at any size);
    * ``theta_abs + sum(thetav)`` -- the azimuth of each vertex measured
      counter-clockwise from local east -- is evaluated directly per vertex
      against the local east/north basis rather than accumulated;
    * the arc length from centre to vertex is ``r * 2 * asin(|d| / (2 r))``
      exactly as ``mpas_arc_length`` writes it.

    The polygon area, edge-normal angles and coefficients then follow core
    lines 1814-1846 term for term in binary64, and the three arrays are cast
    to ``dtype`` once at the end.  Vectorised over cells.
    """

    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    edges_on_cell = np.asarray(_mesh_array(mesh, "edgesOnCell"), dtype=np.int64)
    cells_on_cell = np.asarray(_mesh_array(mesh, "cellsOnCell"), dtype=np.int64)
    vertices_on_cell = np.asarray(
        _mesh_array(mesh, "verticesOnCell"), dtype=np.int64
    )
    cells_on_edge = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
    cell_xyz = _xyz64(mesh, "Cell")
    vertex_xyz = _xyz64(mesh, "Vertex")

    n_cells = int(counts.size)
    max_edges = int(edges_on_cell.shape[1])
    if edges_on_cell.shape[0] != n_cells or vertices_on_cell.shape != edges_on_cell.shape:
        raise ValueError("edgesOnCell/verticesOnCell shapes are inconsistent")
    if cells_on_cell.shape != edges_on_cell.shape:
        raise ValueError("cellsOnCell shape is inconsistent with edgesOnCell")
    bad_count = np.flatnonzero((counts < 3) | (counts > max_edges))
    if bad_count.size:
        cell = int(bad_count[0])
        raise ValueError(f"cell {cell} has invalid nEdgesOnCell {int(counts[cell])}")

    slots = np.arange(max_edges)[None, :]
    used = slots < counts[:, None]
    # Halo guard (core lines 1702-1716), as in the native mirror: a cell with
    # any neighbour outside nCells keeps zero weights.
    neighbour_ok = (cells_on_cell >= 0) & (cells_on_cell < n_cells)
    active = np.all(neighbour_ok | ~used, axis=1)
    vert_bad = used & ((vertices_on_cell < 0) | (vertices_on_cell >= vertex_xyz.shape[0]))
    edge_bad = used & ((edges_on_cell < 0) | (edges_on_cell >= cells_on_edge.shape[0]))
    for name, bad in (("verticesOnCell", vert_bad), ("edgesOnCell", edge_bad)):
        offenders = np.flatnonzero(active & np.any(bad, axis=1))
        if offenders.size:
            raise ValueError(
                f"{name} reaches outside the mesh at cell {int(offenders[0])}"
            )

    coef_c2 = np.zeros((n_cells, max_edges), dtype=np.float64)
    coef_s2 = np.zeros_like(coef_c2)
    coef_cs = np.zeros_like(coef_c2)
    rows = np.flatnonzero(active)
    if rows.size:
        cnt = counts[rows]
        mask = used[rows]
        nxt = np.where(slots + 1 < cnt[:, None], slots + 1, 0)
        verts = np.where(mask, vertices_on_cell[rows], 0)
        centre = cell_xyz[rows]  # (n, 3)
        r_c = np.linalg.norm(centre, axis=-1)
        normal = centre / r_c[:, None]
        delta = vertex_xyz[verts] - centre[:, None, :]  # (n, m, 3)

        # Local east/north basis (east x north = outward normal).
        z_hat = np.array([0.0, 0.0, 1.0])
        north = z_hat[None, :] - normal[:, 2:3] * normal
        north_norm = np.linalg.norm(north, axis=-1)
        tangent = delta - np.einsum("nmk,nk->nm", delta, normal)[..., None] * normal[:, None, :]
        # At a pole the native code sets theta_abs = pi/2 for the first
        # vertex; the equivalent basis takes that vertex's tangent as north.
        pole = north_norm <= 1.0e-12
        if np.any(pole):
            first = tangent[pole, 0, :]
            north[pole] = first
            north_norm[pole] = np.linalg.norm(first, axis=-1)
        north = north / north_norm[:, None]
        east = np.cross(north, normal)
        azimuth = np.arctan2(
            np.einsum("nmk,nk->nm", tangent, north),
            np.einsum("nmk,nk->nm", tangent, east),
        )

        # mpas_arc_length on the radius-normalised points, with the chord
        # taken from the binary64 local offset.
        r_unit = r_c / radius
        chord = np.linalg.norm(delta, axis=-1) / radius
        dl_sphere = radius * (
            r_unit[:, None] * 2.0 * np.arcsin(chord / (2.0 * r_unit[:, None]))
        )
        xp = np.where(mask, np.cos(azimuth) * dl_sphere, 0.0)
        yp = np.where(mask, np.sin(azimuth) * dl_sphere, 0.0)
        xp1 = np.take_along_axis(xp, nxt, axis=1)
        yp1 = np.take_along_axis(yp, nxt, axis=1)

        # Core lines 1814-1822 and 1826-1846, term for term.
        dx = xp1 - xp
        dy = yp1 - yp
        area_terms = 0.25 * (xp + xp1) * (yp1 - yp) - 0.25 * (yp + yp1) * (xp1 - xp)
        area_cell = np.sum(np.where(mask, area_terms, 0.0), axis=1)
        theta_edge = np.arctan2(dy, dx) - 0.5 * np.pi
        dl = np.sqrt(dx * dx + dy * dy)
        sin_t = np.sin(theta_edge)
        cos_t = np.cos(theta_edge)
        scale = dl / area_cell[:, None]
        owner = np.where(mask, edges_on_cell[rows], 0)
        sign = np.where(cells_on_edge[owner, 0] == rows[:, None], 1.0, -1.0)
        coef_c2[rows] = np.where(mask, sign * scale * (cos_t * cos_t), 0.0)
        coef_s2[rows] = np.where(mask, sign * scale * (sin_t * sin_t), 0.0)
        coef_cs[rows] = np.where(mask, sign * scale * (sin_t * cos_t), 0.0)

    result = DeformationWeightsV841(
        coef_c2=coef_c2.astype(dtype),
        coef_s2=coef_s2.astype(dtype),
        coef_cs=coef_cs.astype(dtype),
        geometry="local64",
    )
    result.validate(n_cells=n_cells, max_edges=max_edges)
    return result


def select_deformation_geometry(
    dtype: np.dtype[Any] | type[Any], radius: float, finest_spacing_m: float
) -> str:
    """The ``auto`` decision: ``native`` or ``local64`` for this mesh.

    binary64 execution is always ``native`` (the Earth-centred binary64
    evaluation already holds ~1e-11 relative at 50 m).  binary32 execution
    is ``local64`` exactly when ``spacing(float32(radius)) / finest_spacing``
    exceeds :data:`LOCAL64_GEOMETRY_QUANTUM_RATIO`.
    """

    if np.dtype(dtype) == np.dtype(np.float64):
        return "native"
    if not np.isfinite(finest_spacing_m) or finest_spacing_m <= 0.0:
        return "native"
    quantum = float(np.spacing(np.float32(radius)))
    if quantum / finest_spacing_m > LOCAL64_GEOMETRY_QUANTUM_RATIO:
        return "local64"
    return "native"


def initialize_deformation_weights_v841(
    mesh: object,
    *,
    dtype: np.dtype[Any] | type[Any] = np.float32,
    geometry: str = "auto",
) -> DeformationWeightsV841:
    """v8.4.1 deformation weights (``atm_initialize_deformation_weights``).

    ``geometry`` selects the evaluation (:data:`DEFORMATION_GEOMETRY_MODES`):
    ``native`` is the per-operation mirror of the single-precision reference
    build, ``local64`` the binary64 local-tangent-plane evaluation cast to
    ``dtype``, and ``auto`` (the default) keeps ``native`` on every mesh at
    or above 500 m spacing -- every minted, anchored and coarse configuration
    keeps its exact bytes -- and moves sub-500 m binary32 meshes to
    ``local64``, where the native evaluation's 0.5 m coordinate quantum is
    0.1-1 % of an edge and the weights it produces are off by 3-18 %
    (:data:`LOCAL64_GEOMETRY_QUANTUM_RATIO` records the measurements).
    The returned weights record the evaluation used in ``geometry``.
    """

    out_dtype = np.dtype(dtype)
    if out_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("deformation weights dtype must be float32 or float64")
    if geometry not in DEFORMATION_GEOMETRY_MODES:
        raise ConfigurationRefusal(
            "deformation_geometry",
            geometry,
            "the deformation weights know native, local64 and auto",
            "geometry in " + ", ".join(DEFORMATION_GEOMETRY_MODES),
        )
    radius = _require_spherical(mesh)
    spacing = finest_cell_spacing_m(mesh)
    chosen = (
        select_deformation_geometry(out_dtype, radius, spacing)
        if geometry == "auto"
        else geometry
    )
    if chosen == "native":
        weights = _deformation_weights_native(mesh, dtype=out_dtype)
    else:
        weights = _deformation_weights_local64(
            mesh, dtype=out_dtype, radius=radius
        )
    return replace(weights, finest_spacing_m=spacing)


@dataclass(frozen=True, slots=True)
class SmagorinskyCoefficientsV841:
    kdiff: FloatArray
    h_mom_eddy_visc4: np.floating[Any]
    h_theta_eddy_visc4: np.floating[Any]
    config_len_disp: np.floating[Any]


def compute_smagorinsky_coefficients_v841(
    mesh: object,
    normal_velocity: object,
    tangential_velocity: object,
    weights: DeformationWeightsV841,
    *,
    dt: float,
    config: V841MixingConfig | None = None,
) -> SmagorinskyCoefficientsV841:
    """Mirror ``smagorinsky_2d`` (mpas_atm_dissipation_models.F:119-204).

    ``u`` is the edge normal velocity and ``v`` the reconstructed edge
    tangential velocity, as at the native call site
    (mpas_atm_time_integration.F:6348-6352).
    """

    cfg = V841MixingConfig() if config is None else config
    cfg.validate()
    u = _float_field("normal_velocity", normal_velocity, ndim=2)
    v = _same_dtype(u, "tangential_velocity", tangential_velocity, ndim=2)
    if v.shape != u.shape:
        raise ValueError("normal_velocity and tangential_velocity shapes differ")
    timestep = _finite_scalar("config_dt", dt)
    if timestep <= 0.0:
        raise ConfigurationRefusal(
            "config_dt",
            dt,
            "the Smagorinsky stability ceiling requires a positive timestep",
            "config_dt>0",
        )

    edges = np.asarray(_mesh_array(mesh, "edgesOnCell"), dtype=np.int64)
    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    dtype = u.dtype
    coef_c2 = np.asarray(weights.coef_c2, dtype=dtype)
    coef_s2 = np.asarray(weights.coef_s2, dtype=dtype)
    coef_cs = np.asarray(weights.coef_cs, dtype=dtype)
    if edges.ndim != 2 or counts.shape != (edges.shape[0],):
        raise ValueError("edgesOnCell/nEdgesOnCell shapes are inconsistent")
    if coef_c2.shape != edges.shape:
        raise ValueError("deformation weights must have shape edgesOnCell")
    used = np.arange(edges.shape[1])[None, :] < counts[:, None]
    if np.any((edges[used] < 0) | (edges[used] >= u.shape[1])):
        raise ValueError("edgesOnCell contains an invalid used edge")

    length = resolve_config_len_disp(mesh, cfg.config_len_disp, dtype=dtype)
    cs_coef = dtype.type(cfg.config_smagorinsky_coef)
    # (c_s * config_len_disp)**2  (dissipation-models line 189)
    strain_scale = (cs_coef * length) * (cs_coef * length)
    # invDt = 1.0/dt; ceiling = (0.01*config_len_disp**2) * invDt
    # (time-integration line 6323; dissipation-models line 190)
    inv_dt = dtype.type(1.0) / dtype.type(timestep)
    ceiling = (dtype.type(0.01) * (length * length)) * inv_dt

    nlev = u.shape[0]
    n_cells = edges.shape[0]
    kdiff = np.zeros((nlev, n_cells), dtype=dtype)
    two = dtype.type(2.0)
    quarter = dtype.type(0.25)
    for cell in range(n_cells):
        dudx = np.zeros(nlev, dtype=dtype)
        dudy = np.zeros(nlev, dtype=dtype)
        dvdx = np.zeros(nlev, dtype=dtype)
        dvdy = np.zeros(nlev, dtype=dtype)
        for slot in range(int(counts[cell])):
            edge = int(edges[cell, slot])
            c2 = coef_c2[cell, slot]
            s2 = coef_s2[cell, slot]
            ccs = coef_cs[cell, slot]
            dudx += c2 * u[:, edge] - ccs * v[:, edge]
            dudy += ccs * u[:, edge] - s2 * v[:, edge]
            dvdx += ccs * u[:, edge] + c2 * v[:, edge]
            dvdy += s2 * u[:, edge] + ccs * v[:, edge]
        d_11 = two * dudx
        d_22 = two * dvdy
        d_12 = dudy + dvdx
        diff = d_11 - d_22
        strain = np.sqrt(quarter * (diff * diff) + d_12 * d_12)
        kdiff[:, cell] = np.minimum(strain_scale * strain, ceiling)

    visc4 = dtype.type(cfg.config_visc4_2dsmag)
    # h_mom_eddy_visc4 = config_visc4_2dsmag * config_len_disp**3;
    # h_theta_eddy_visc4 = h_mom_eddy_visc4 (dissipation-models lines 199-200)
    h4 = visc4 * ((length * length) * length)
    return SmagorinskyCoefficientsV841(
        kdiff=kdiff,
        h_mom_eddy_visc4=h4,
        h_theta_eddy_visc4=h4,
        config_len_disp=length,
    )


def compute_dry_mixing_tendencies_v841(
    mesh: object,
    weights: DeformationWeightsV841,
    *,
    normal_velocity: object,
    tangential_velocity: object,
    vertical_velocity: object,
    theta_m: object,
    rho_edge: object,
    divergence: object,
    vorticity: object,
    dt: float,
    config: V841MixingConfig | None = None,
) -> DryMixingTendencies:
    """v8.4.1 RK-stage-one horizontal mixing increments (saved for RK2/RK3).

    kdiff comes from the v8.4.1 ``smagorinsky_2d`` mirror; the u/w/theta
    applications reuse the :mod:`woof.hex.mixing` authorities, which are the
    same non-LES stencils as ``u_dissipation_3d`` / ``w_dissipation_3d`` /
    ``scalar_dissipation_3d_les`` (see the module docstring for the two
    exact-inert differences).
    """

    cfg = V841MixingConfig() if config is None else config
    cfg.validate()
    coefficients = compute_smagorinsky_coefficients_v841(
        mesh,
        normal_velocity,
        tangential_velocity,
        weights,
        dt=dt,
        config=cfg,
    )
    dtype = coefficients.kdiff.dtype
    scaling = compute_mesh_mixing_scaling(
        mesh,
        config_h_ScaleWithMesh=cfg.config_h_ScaleWithMesh,
        dtype=dtype,
    )
    # Regional culls carry stored-0 (negative-sentinel) cellsOnEdge slots on
    # ring-7 rows.  Native runs these filters over the explicit garbage
    # elements -- delsq scratch garbage columns are zeroed by atm_srk3 and
    # theta_m's garbage cell is zeroed by the rk setup -- so the filters run
    # here on the same padded memory model and the pads are stripped after.
    # Ring-6/7 filter lanes are dead regardless: the specified-zone tendency
    # overwrite replaces them before anything reads them.
    filter_mesh: object = mesh
    kdiff_arg: object = coefficients.kdiff
    div_arg: object = divergence
    vort_arg: object = vorticity
    theta_arg: object = theta_m
    w_arg: object = vertical_velocity
    raw_coe = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
    regional = bool(np.any(raw_coe < 0))
    n_cells = int(np.asarray(_mesh_array(mesh, "areaCell")).size)
    if regional:
        def _pad(value: object, fill: float = 0.0) -> FloatArray:
            data = np.asarray(value)
            pad = np.full(data.shape[:-1] + (1,), fill, dtype=data.dtype)
            return np.concatenate([data, pad], axis=-1)

        arrays: dict[str, np.ndarray] = {
            "cellsOnEdge": np.where(raw_coe < 0, n_cells, raw_coe),
        }
        for name in (
            "verticesOnEdge",
            "edgesOnVertex",
            "dcEdge",
            "dvEdge",
            "areaTriangle",
            "meshDensity",
            "nominalMinDc",
        ):
            try:
                arrays[name] = np.asarray(_mesh_array(mesh, name))
            except AttributeError:
                pass
        # The garbage cell: no edges, unit area (native never divides by its
        # area -- its loops exclude it -- so the pad only has to be inert).
        eoc = np.asarray(_mesh_array(mesh, "edgesOnCell"), dtype=np.int64)
        arrays["edgesOnCell"] = np.concatenate(
            [eoc, np.zeros((1, eoc.shape[1]), dtype=np.int64)], axis=0
        )
        counts_real = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
        arrays["nEdgesOnCell"] = np.concatenate(
            [counts_real, np.zeros(1, dtype=np.int64)]
        )
        area_real = np.asarray(_mesh_array(mesh, "areaCell"))
        arrays["areaCell"] = np.concatenate(
            [area_real, np.ones(1, dtype=area_real.dtype)]
        )

        class _RegionalFilterMesh:
            def __init__(self, table: dict[str, np.ndarray]) -> None:
                self.arrays = table

        filter_mesh = _RegionalFilterMesh(arrays)
        kdiff_arg = _pad(coefficients.kdiff)
        div_arg = _pad(divergence)
        vort_arg = vorticity
        theta_arg = _pad(theta_m)
        w_arg = _pad(vertical_velocity)
    momentum = momentum_horizontal_filter_tendency(
        filter_mesh,
        rho_edge=rho_edge,
        divergence=div_arg,
        vorticity=vort_arg,
        kdiff=kdiff_arg,
        h_mom_eddy_visc4=coefficients.h_mom_eddy_visc4,
        config_del4u_div_factor=cfg.config_del4u_div_factor,
        mesh_scaling_del2=scaling.del2,
        mesh_scaling_del4=scaling.del4,
    )
    w_filter = vertical_momentum_horizontal_filter_tendency(
        filter_mesh,
        vertical_velocity=w_arg,
        rho_edge=rho_edge,
        kdiff=kdiff_arg,
        h_mom_eddy_visc4=coefficients.h_mom_eddy_visc4,
        mesh_scaling_del2=scaling.del2,
        mesh_scaling_del4=scaling.del4,
    )
    theta_filter = theta_horizontal_filter_tendency(
        filter_mesh,
        theta_m=theta_arg,
        rho_edge=rho_edge,
        kdiff=kdiff_arg,
        h_theta_eddy_visc4=coefficients.h_theta_eddy_visc4,
        mesh_scaling_del2=scaling.del2,
        mesh_scaling_del4=scaling.del4,
    )

    def _strip(value: FloatArray, count: int) -> FloatArray:
        return value[:, :count] if regional else value

    return DryMixingTendencies(
        kdiff=coefficients.kdiff,
        h_mom_eddy_visc4=coefficients.h_mom_eddy_visc4,
        h_theta_eddy_visc4=coefficients.h_theta_eddy_visc4,
        tend_u_euler=momentum.tendency,
        tend_w_euler=_strip(w_filter.tendency, n_cells),
        tend_theta_euler=_strip(theta_filter.tendency, n_cells),
        delsq_u=momentum.delsq_u,
        delsq_divergence=_strip(momentum.delsq_divergence, n_cells),
        delsq_vorticity=momentum.delsq_vorticity,
        delsq_w=_strip(w_filter.laplacian, n_cells),
        delsq_theta=_strip(theta_filter.laplacian, n_cells),
    )


__all__ = [
    "DEFORMATION_GEOMETRY_MODES",
    "DeformationWeightsV841",
    "LOCAL64_GEOMETRY_QUANTUM_RATIO",
    "SmagorinskyCoefficientsV841",
    "V841MixingConfig",
    "compute_dry_mixing_tendencies_v841",
    "compute_smagorinsky_coefficients_v841",
    "finest_cell_spacing_m",
    "initialize_deformation_weights_v841",
    "select_deformation_geometry",
]
