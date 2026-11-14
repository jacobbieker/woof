"""WRF's inflow perturbation on a regional hex mesh's relaxation rings.

The WRF path (``woof.core.inflow_perturbation``) seeds a one-way child's
inflow: cell-blocked potential-temperature increments added to the theta
relaxation TARGET on the inflow faces' relax-zone rows, so the Davies
relaxation nudges the relax zone toward a perturbed boundary and advection
carries the blocks inward.  A regional hex mesh is driven the same way: the
MPAS relaxation (``regional_relaxzone_rayleigh/filter_cell``) pulls the
cells with ``1 <= bdyMaskCell <= nRelaxZone`` toward the boundary series.
This module adds WRF's increment to that target and to nothing else; every
formula and constant is WRF's own, imported from the shared module:

* draws: :func:`draw_unit_pattern` keyed on (seed, grid_id, face, refresh
  index), uniform in [-1, 1), one per (mass level, :data:`BLOCK_CELLS`-cell
  along-face block), held :data:`REFRESH_SECONDS` of model time
  (:func:`refresh_ticks` on the run's configured step);
* faces: :func:`select_faces` on the signed face-mean inward-normal wind;
* amplitude: :func:`face_amplitude`, the Eckert-number law
  ``scale * U**2 / (Ec * cp)``;
* depth: :func:`perturbed_level_count` of the domain-mean half-level heights
  against the PBLH (WRF reads the parent's PBLH over the child's footprint; a
  regional hex run has no parent model, so it reads its own physics PBLH over
  the boundary zone, and :func:`parent_pbl_refusal` refuses a run with no PBL);
* coupling: ``delta(rtheta_m) = rho_zz * (1 + rvord*qv) * delta_theta``,
  the target's own moist mass coupling (``rtheta_m / theta`` of the boundary
  state), so ``delta_theta`` Kelvin shift the relaxation equilibrium by
  ``delta_theta`` Kelvin, as ``add_coupled_theta`` does in WRF's coupled units.

The mesh analogue of WRF's four faces: each boundary cell belongs to the face
its outward direction from the domain centre points to (the larger of the
east and north components in the centre's tangent plane), and a block is
:data:`BLOCK_CELLS` mean cell spacings of the along-face coordinate, so every
ring of the relax zone at one along-face position shares the draw, as every
relax row of one WRF column does.  The specified zone, the other fields and
the boundary tendencies are untouched.  Off (the default) builds nothing.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from woof.core.inflow_perturbation import (BLOCK_CELLS, ECKERT, FACE_CODES, REFRESH_SECONDS,
                                           draw_unit_pattern, face_amplitude,
                                           parent_pbl_refusal, perturbed_level_count,
                                           refresh_ticks, select_faces)

FACES = ("west", "east", "south", "north")
#: The hex regional domain's grid id in the draw key (WRF's d01).
GRID_ID = 1


def _host(mesh, name):
    value = getattr(mesh, name, None)
    if value is None:
        arrays = getattr(mesh, "arrays", None)
        if isinstance(arrays, Mapping):
            value = arrays.get(name)
    if value is None:
        raise ValueError(f"inflow perturbation needs the mesh's {name}")
    return np.asarray(value)


def face_geometry(mesh, bdy_mask_cell, bdy_mask_edge, *, n_relax_zone: int):
    """Faces, along-face blocks and inward-flow edges of a regional mesh.

    Returns ``{face: {"cells", "blocks", "n_blocks", "edges", "signs"}}``:
    the relax-zone cells of the face with their block numbers, and the edges
    crossing a ring boundary on the face with the sign that turns their
    normal velocity into the inward-normal component.
    """
    lat = _host(mesh, "latCell").astype(np.float64)
    lon = _host(mesh, "lonCell").astype(np.float64)
    ncells = int(np.asarray(bdy_mask_cell).size)
    lat, lon = lat[:ncells], lon[:ncells]
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1)
    ring = np.asarray(bdy_mask_cell, dtype=np.int64)
    centre = xyz[ring == 0].mean(axis=0)
    centre /= np.linalg.norm(centre)
    east = np.cross([0.0, 0.0, 1.0], centre)
    if np.linalg.norm(east) < 1e-12:
        east = np.array([1.0, 0.0, 0.0])
    east /= np.linalg.norm(east)
    north = np.cross(centre, east)

    def plane(points):
        return points @ east, points @ north

    coe = _host(mesh, "cellsOnEdge").astype(np.int64)
    nedges = int(np.asarray(bdy_mask_edge).size)
    coe = coe[:nedges]
    if coe.min() >= 1 and coe.max() > ncells - 1:
        coe = coe - 1                          # 1-based file connectivity
    # Metres: MPAS spherical meshes carry dcEdge in metres on sphere_radius.
    radius = float((getattr(mesh, "attrs", None) or {}).get("sphere_radius", 6371229.0))
    spacing = float(np.mean(_host(mesh, "dcEdge")[:nedges][np.asarray(bdy_mask_edge) > 0]))
    width = BLOCK_CELLS * spacing

    def faces_of(e, n):
        return np.where(np.abs(e) >= np.abs(n), np.where(e < 0, 0, 1), np.where(n < 0, 2, 3))

    e, n = plane(xyz)
    cell_face = faces_of(e, n)
    relax = np.nonzero((ring >= 1) & (ring <= n_relax_zone))[0]
    out = {}
    for code, face in enumerate(FACES):
        cells = relax[cell_face[relax] == code]
        # The along-face coordinate, metres in the centre's tangent plane.
        along = (n if face in ("west", "east") else e)[cells] * radius
        blocks = (np.floor((along - along.min()) / width).astype(np.int64)
                  if cells.size else np.zeros(0, np.int64))
        out[face] = {"cells": cells, "blocks": blocks,
                     "n_blocks": int(blocks.max()) + 1 if blocks.size else 0}
    valid = (coe >= 0).all(axis=1) & (coe < ncells).all(axis=1)
    c1, c2 = np.where(valid, coe[:, 0], 0), np.where(valid, coe[:, 1], 0)
    crossing = valid & (ring[c1] != ring[c2]) & (np.minimum(ring[c1], ring[c2]) >= 1)
    mid = xyz[c1] + xyz[c2]
    me, mn = plane(mid)
    edge_face = faces_of(me, mn)
    # MPAS's normal points from cellsOnEdge(1) to cellsOnEdge(2): inward when
    # the second cell sits on an inner ring.
    sign = np.where(ring[c2] < ring[c1], 1.0, -1.0).astype(np.float32)
    for code, face in enumerate(FACES):
        edges = np.nonzero(crossing & (edge_face == code))[0]
        out[face]["edges"], out[face]["signs"] = edges, sign[edges]
    return out


class HexInflowPerturbation:
    """WRF's inflow seeding on one regional hex run (see the module docstring)."""

    def __init__(self, runtime, *, seed: int = 0, amplitude_scale: float = 1.0,
                 faces: str = "inflow", pblh_source=None, n_relax_zone: int = 5):
        import cupy as cp
        if pblh_source is None:
            raise ValueError(parent_pbl_refusal(None))
        if seed < 0:
            raise ValueError("inflow perturbation seed must be non-negative")
        if not math.isfinite(amplitude_scale) or amplitude_scale < 0:
            raise ValueError("inflow perturbation amplitude scale must be finite and non-negative")
        select_faces({f: 0.0 for f in FACES}, faces)        # validates the mode
        self.runtime = runtime
        self.seed, self.amplitude_scale, self.faces_mode = int(seed), float(amplitude_scale), str(faces)
        self.pblh_source = pblh_source
        mesh = getattr(runtime.padded, "base", runtime.padded)
        masks = runtime.masks_host
        self.geometry = face_geometry(mesh, masks.bdy_mask_cell, masks.bdy_mask_edge,
                                      n_relax_zone=n_relax_zone)
        self.nlev = int(runtime.nlev)
        self.relax_cells = cp.asarray(np.concatenate(
            [self.geometry[f]["cells"] for f in FACES]).astype(np.int64))
        zgrid = cp.asnumpy(runtime.atmosphere.vertical.zgrid).astype(np.float64)
        interior = np.asarray(masks.bdy_mask_cell) == 0
        z = zgrid[:, :interior.size][:, interior]
        # WRF: the domain-mean half-level height above ground (the base state).
        self.z_half_agl = (0.5 * (z[:-1] + z[1:]) - z[0]).mean(axis=1)
        self.start = runtime.clock
        self._ticks = refresh_ticks(1, float(runtime.outer_dt))
        self._key = None
        self._delta = None
        self._last_refresh = None
        self.totals = {}

    def _refresh(self, step: int) -> int:
        refresh = int(step) // self._ticks
        if self._last_refresh is not None and refresh < self._last_refresh:
            raise ValueError("inflow perturbation refresh index went backward")
        self._last_refresh = refresh
        return refresh

    def _inward_means(self, n_levels: int) -> dict:
        import cupy as cp
        u = self.runtime.driving_values("u", 1)
        means = {}
        for face in FACES:
            g = self.geometry[face]
            if not g["edges"].size:
                means[face] = 0.0
                continue
            edges = cp.asarray(g["edges"])
            signs = cp.asarray(g["signs"])
            means[face] = float(cp.mean(u[:n_levels, edges].astype(cp.float64) * signs[None, :]))
        return means

    def delta(self):
        """This step's (nlev, ncells) Kelvin increment, or None."""
        import cupy as cp
        # Model seconds since the simulation start (the runtime's lbc clock,
        # which advances once per model step; ``start`` is set at attach).
        elapsed = (self.runtime.clock - self.start).total_seconds()
        if elapsed == self._key:
            return self._delta
        self._key = elapsed
        self._delta = None
        step = int(round(elapsed / float(self.runtime.outer_dt)))
        pblh = float(self.pblh_source())
        n_levels = perturbed_level_count(self.z_half_agl, pblh)
        if n_levels == 0:
            return None
        inward = self._inward_means(n_levels)
        refresh = self._refresh(step)
        ncells = int(self.runtime.atmosphere.vertical.zgrid.shape[1])
        out = None
        for face in select_faces(inward, self.faces_mode):
            g = self.geometry[face]
            theta_max = face_amplitude(inward[face], self.amplitude_scale)
            total = self.totals.setdefault(face, {"forces": 0, "theta_max_k_max": 0.0})
            total["forces"] += 1
            total["theta_max_k_max"] = max(total["theta_max_k_max"], theta_max)
            if theta_max == 0.0 or not g["cells"].size:
                continue
            pattern = draw_unit_pattern(self.seed, GRID_ID, face, refresh, self.nlev, g["n_blocks"])
            values = (theta_max * pattern[:, g["blocks"]]).astype(np.float32)
            values[n_levels:, :] = np.float32(0.0)
            if out is None:
                out = cp.zeros((self.nlev, ncells), dtype=cp.float32)
            out[:, cp.asarray(g["cells"])] = cp.asarray(values)
        self._delta = out
        return out

    def target(self, values_rt, rk_step: int):
        """The rtheta_m relaxation target with this step's increment added."""
        import cupy as cp
        delta = self.delta()
        if delta is None:
            return values_rt
        theta = self.runtime.driving_values("theta", rk_step)
        cells = self.relax_cells
        out = values_rt.copy()
        out[:, cells] = values_rt[:, cells] + (values_rt[:, cells] / theta[:, cells]) * delta[:, cells]
        return out

    def receipt(self) -> dict:
        return {"seed": self.seed, "grid_id": GRID_ID, "amplitude_scale": self.amplitude_scale,
                "faces_mode": self.faces_mode, "eckert": ECKERT, "refresh_seconds": REFRESH_SECONDS,
                "block_cells": BLOCK_CELLS,
                "faces": {f: {"relax_cells": int(self.geometry[f]["cells"].size),
                              "blocks": self.geometry[f]["n_blocks"],
                              "crossing_edges": int(self.geometry[f]["edges"].size)} for f in FACES},
                "totals": {f: dict(v) for f, v in self.totals.items()}}


def pblh_from_backend(backend, runtime, n_relax_zone: int = 5):
    """The run's own physics PBLH, averaged over the boundary zone's cells."""
    import cupy as cp
    driver = getattr(getattr(backend, "_seam", None), "_driver", None)
    fields = getattr(driver, "fields", None) or {}
    if "pblh" not in fields or not int(getattr(getattr(driver, "refl_config", None), "bl_pbl_physics", 1) or 0):
        return None
    ring = np.asarray(runtime.masks_host.bdy_mask_cell)
    cells = cp.asarray(np.nonzero(ring >= 1)[0])

    def read():
        pblh = fields["pblh"].reshape(-1)
        return float(cp.mean(pblh[cells].astype(cp.float64)))
    return read


def attach(driver, backend, options: Mapping | None):
    """Bind the seeding to a regional run's relaxation; None (nothing built) when off."""
    if not options:
        return None
    runtime = getattr(driver, "regional_v841", None)
    if runtime is None:
        raise ValueError("--inflow-perturbation seeds a regional mesh's relaxation zone; this mesh is global")
    if getattr(runtime.padded.base if hasattr(runtime.padded, "base") else runtime.padded,
               "partition_layout_v841", None) is not None:
        raise ValueError("--inflow-perturbation needs the whole boundary on one card (face means); "
                         "a partitioned regional run would see only its own part")
    source = pblh_from_backend(backend, runtime)
    if source is None:
        raise ValueError(parent_pbl_refusal(0).replace("parent-diagnosed", "run's own")
                         + " (the hex run's own PBL scheme)")
    seeding = HexInflowPerturbation(
        runtime, seed=int(options.get("seed", 0)),
        amplitude_scale=float(options.get("amplitude_scale", 1.0)),
        faces=str(options.get("faces", "inflow")), pblh_source=source)
    # The simulation start: a resumed run's clock is already past it.
    from datetime import timedelta
    seeding.start = runtime.clock - timedelta(seconds=float(driver.atmosphere.state.time_seconds))
    runtime.inflow_perturbation = seeding
    return runtime.inflow_perturbation.receipt()


__all__ = ["FACES", "GRID_ID", "HexInflowPerturbation", "attach", "face_geometry",
           "pblh_from_backend", "FACE_CODES"]
