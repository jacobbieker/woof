"""WRF's wind-farm scheme (``windfarm_opt = 1``, Fitch et al. 2012) on hex.

hex runs WOOF's physics through the engine's generic column seam
(:class:`woof.core.mpas_column_suite.MpasColumnPhysics`), whose phase one is
the very ``PhysicsDriver.compute`` the WRF-like core steps.  So the wind
farm reaches hex through the same three keys the WRF-like core reads, given
as ``--physics`` overrides (``woof hex forecast --physics-suite rrfs_v2_dev
--physics windfarm_opt=1 --physics windfarm_turbine_file=farm.csv``, and
``windfarm_tke_factor``): they are RunConfig fields, so
:mod:`woof.hex.physics_suites` hands them to the generic seam and the
engine's own ``validate_run_config`` judges them (MYNN is required, by
name: ``bl_pbl_physics = 5``).  What is hex's own:

* PLACEMENT.  A turbine goes to the column whose cell centre is nearest
  (:mod:`woof.windfarm.cells`, the rule MPAS applies to a point), and a
  cell's turbines group in table order exactly as on a WRF grid
  (:func:`woof.windfarm.layout.build_cell_layout`; a WRF
  ``windturbines.txt`` lat/lon list is read by
  :mod:`woof.core.windfarm_coupling` and placed by the same rule).
  :func:`interior_columns` says which physics columns are modelled: a
  limited-area mesh's boundary cells (``bdyMaskCell > 0``) and its padded
  garbage column are not, and a turbine there is refused by name.
* THE DRAG.  WRF divides the rotor thrust by the cell volume
  ``dx*dx*dz``; on hex each column's own ``areaCell`` takes ``dx*dx``
  (``fitch_column_cells``, :func:`woof.core.windfarm_fitch.launch_fitch`
  with ``wfdensity``).  It runs in the PBL slot right after MYNN, adding
  into the slot's A-grid ``du``/``dv`` and MYNN's ``qke`` exactly as on the
  WRF-like core; hex then projects the cell-centre momentum rate onto the
  edges as it projects every PBL rate (``project_edges_v841_f32``, MPAS's
  ``tend_u_physics``: the normal component of the mean of the edge's two
  cell vectors, times the edge density).
* OUTPUT.  WRF's ``POWER`` (W per cell, the turbines' summed power at the
  hub-height wind) is the history field ``power``; and
  :class:`HexWindfarmPower` writes the per-turbine rotor-equivalent wind,
  power and thrust coefficient of a woof turbine table beside each history
  frame (``windfarm_d01_<valid time>.nc``), with the WRF-like core's own
  arithmetic (:mod:`woof.core.windfarm_power_diag`).

REFUSED BY NAME.  The MPAS suite route (``--physics-backend mpasphys``,
MPAS-A v8.4.2's own physics): MPAS has no wind-farm scheme, so its
namelist has no key for one (:func:`mpas_route_refusal`).  A PBL other
than MYNN (the engine's refusal).  ``windfarm_ij = 1`` (an unstructured
mesh has no i/j).  A turbine outside the mesh or in its boundary columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

#: The keys the WRF-like core reads, and hex hands its generic seam.
WINDFARM_KEYS = ("windfarm_opt", "windfarm_ij", "windfarm_tke_factor",
                 "windfarm_turbine_file")


def mpas_route_refusal(keys) -> str | None:
    """The refusal of a wind-farm key on the MPAS suite route, or None."""
    from woof.explain import layered

    named = sorted({str(key).strip().casefold() for key in keys}
                   & set(WINDFARM_KEYS))
    if not named:
        return None
    return layered(
        f"{', '.join(named)} on the MPAS suite route (--physics-backend "
        "mpasphys): MPAS-A v8.4.2 has no wind-farm scheme, so neither its "
        "Registry nor its physics driver has anything to run them with.\n"
        "  remedy: run the farm on WOOF's physics (drop --physics-backend "
        "mpasphys; e.g. --physics-suite rrfs_v2_dev, whose PBL is MYNN), "
        "where windfarm_opt = 1 is WRF v4.7.1's Fitch scheme",
        "WRF calls the Fitch drag from its PBL driver right after MYNN; "
        "hex carries it on the engine's generic column seam, which runs "
        "that driver.  Writing the keys into an MPAS namelist would record "
        "a wind farm that never acts.")


def interior_columns(mesh: Any, *, is_regional: bool) -> np.ndarray:
    """One bool per physics column: may a turbine stand there?

    A global mesh: every cell.  A limited-area mesh: the cells with
    ``bdyMaskCell == 0`` (the specified and relaxation zones are forced by
    the lateral boundary), then ``False`` for the padded garbage column the
    physics seam carries after them.
    """
    from woof.hex.drivers.run_cuda_v841_full_physics_x4 import _mesh_value

    n_cells = int(np.asarray(_mesh_value(mesh, "latCell")).size)
    if not is_regional:
        return np.ones(n_cells, dtype=np.bool_)
    mask = np.asarray(_mesh_value(mesh, "bdyMaskCell")).reshape(-1)
    if mask.size != n_cells:
        raise ValueError(
            f"bdyMaskCell carries {mask.size} cells against {n_cells}")
    return np.append(mask == 0, False).astype(np.bool_)


# --------------------------------------------------------------------------
# the per-turbine power diagnostic


@dataclass(frozen=True)
class _Request:
    """What :class:`woof.core.windfarm_power_diag.WindfarmPowerDiagnostic`
    reads off an ``[output] windfarm_turbines`` request."""

    turbines: tuple
    source: str


def power_turbines(layout) -> tuple:
    """A placed woof turbine table as the power diagnostic's turbine rows.

    Each turbine keeps its id, float32 lat/lon (as a float), its group's
    hub height and rotor (the words the drag runs) and its library curve
    id, which the diagnostic resolves to the same curve the layout did.
    """
    from woof.core.windfarm_power_diag import Turbine

    rows = []
    for n, name in enumerate(layout.turbine_ids):
        g = int(layout.turbine_group[n])
        rows.append(Turbine(
            name=str(name), lat=float(layout.turbine_lat[n]),
            lon=float(layout.turbine_lon[n]),
            hub_height_m=float(layout.group_hub_height_m[g]),
            rotor_diameter_m=float(layout.group_rotor_diameter_m[g]),
            turbine_type=str(layout.curve_ids[int(layout.turbine_curve[n])])))
    return tuple(rows)


class HexWindfarmPower:
    """The WRF-like core's wind-farm power window, on hex's columns.

    The arithmetic is :mod:`woof.core.windfarm_power_diag`'s, word for word
    (its kernel folds the window on the device; its host road evaluates the
    frame): this class only puts hex's columns where that diagnostic reads a
    WRF state.  Each turbine's column is gathered into a compact pseudo-grid
    of ``2n`` mass columns (turbine ``k`` at ``i = 2k``), whose staggered
    ``u`` holds the column's wind on both faces and whose ``v`` holds it on
    both rows, so the diagnostic's destaggering ``0.5*(a + a)`` returns the
    column's own word.

    WHEN IT FOLDS.  Its columns are gathered in phase two of every step,
    after the seam has copied the post-RK state in and before microphysics
    adjusts it (:meth:`observe`, a ``phase2_observers`` hook of the column
    seam): the WRF-like core's dycore-epilogue instant.  They enter the
    window when the backend commits the step (:meth:`commit`); an aborted
    step is dropped (:meth:`discard`).  The frame's own values are
    evaluated from the history frame's arrays (:meth:`frame_values`).  The
    window restarts at every frame (:meth:`take_window`); a resumed run
    starts a fresh window, as on the WRF-like core.
    """

    def __init__(self, layout, *, n_columns: int, grid_id: int = 1,
                 source: str = "", device_id: int | None = None):
        import cupy as cp
        from woof.core.windfarm_power_diag import WindfarmPowerDiagnostic

        self.layout = layout
        self.turbines = power_turbines(layout)
        self.columns = np.asarray(layout.turbine_i, dtype=np.int64)
        self.n = len(self.turbines)
        self.n_columns = int(n_columns)
        self.grid_id = int(grid_id)
        request = _Request(turbines=self.turbines, source=str(source))
        pseudo_rows = [(t.name, t.lat, t.lon, 0, 2 * k, t.lat, t.lon, 0.0)
                       for k, t in enumerate(self.turbines)]
        self.device_id = (int(cp.cuda.Device().id) if device_id is None
                          else int(device_id))
        with cp.cuda.Device(self.device_id):
            self.diagnostic = WindfarmPowerDiagnostic(
                request, pseudo_rows, ny=1, nx=2 * self.n, xp=cp,
                device_id=self.device_id, grid_id=self.grid_id)
        self._pseudo = None
        self._maps = None
        self._pending_dt = None
        #: The geometry the sidecar records: the column itself, its centre.
        self.cell_rows = None

    def bind_cells(self, placement) -> None:
        """Record each turbine's column and centre for the sidecar rows."""
        from woof.windfarm.cells import SPHERE_RADIUS_M, _unit_vectors

        cols = self.columns
        points = _unit_vectors([t.lat for t in self.turbines],
                               [t.lon for t in self.turbines])
        dots = np.clip(np.einsum("ij,ij->i", points, placement._xyz[cols]),
                       -1.0, 1.0)
        distance = SPHERE_RADIUS_M * np.arccos(dots)
        self.cell_rows = tuple(
            (t.name, t.lat, t.lon, 0, int(c),
             float(placement.latitude_deg[c]),
             float(placement.longitude_deg[c]), float(d))
            for t, c, d in zip(self.turbines, cols, distance))
        self.diagnostic.rows = self.cell_rows

    # -- the per-step fold --------------------------------------------------

    def _gather(self, seam):
        """The turbines' columns of the seam's state as a WRF-shaped state.

        One ``take`` per field, through flat index maps built once: every
        element of a pseudo-grid field names the seam element it copies (the
        odd pseudo columns repeat the even ones), so a step costs seven
        gather launches and nothing else."""
        import cupy as cp

        nz, ncol = int(seam._nz), int(seam._ncol)
        if ncol != self.n_columns:
            raise RuntimeError(
                f"the seam carries {ncol} columns and the farm was placed on "
                f"{self.n_columns}")
        state, buf = seam._state, seam._in
        if self._pseudo is None or self._pseudo.nz != nz:
            self._pseudo, self._maps = self._build_pseudo(nz, ncol)
        s, m = self._pseudo, self._maps
        for name, source in (("u", buf["u"]), ("v", buf["v"]),
                             ("php", state.php), ("thp", state.thp),
                             ("p", state.p), ("qv", state.qv),
                             ("ht", state.ht)):
            target = getattr(s, name)
            cp.take(source.reshape(-1), m[name], out=target.reshape(-1))
        return s

    def _build_pseudo(self, nz: int, ncol: int):
        """The pseudo-grid arrays and, per field, the flat source index of
        each of their elements (turbine k's column at pseudo i = 2k, 2k+1)."""
        import cupy as cp

        n = self.n
        col = np.repeat(self.columns, 2)                      # (2n,)
        u_col = np.append(col, col[-1])                       # (2n + 1,)

        def plane(rows, cols):
            return (np.arange(rows)[:, None] * ncol + cols[None, :]).reshape(-1)

        maps = {
            "u": plane(nz, u_col),                            # (nz, 1, 2n+1)
            "v": np.concatenate([plane(nz, col).reshape(nz, 1, -1)] * 2,
                                axis=1).reshape(-1),          # (nz, 2, 2n)
            "php": plane(nz + 1, col),                        # (nz+1, 1, 2n)
            "thp": plane(nz, col), "p": plane(nz, col), "qv": plane(nz, col),
            "ht": col.copy(),                                 # (1, 2n)
        }
        f32 = cp.float32
        pseudo = _PseudoState(
            nz=nz,
            u=cp.zeros((nz, 1, 2 * n + 1), f32),
            v=cp.zeros((nz, 2, 2 * n), f32),
            php=cp.zeros((nz + 1, 1, 2 * n), f32),
            phb=cp.zeros((nz + 1,), f32),
            ht=cp.zeros((1, 2 * n), f32),
            thp=cp.zeros((nz, 1, 2 * n), f32),
            thb=cp.zeros((nz,), f32),
            p=cp.zeros((nz, 1, 2 * n), f32),
            qv=cp.zeros((nz, 1, 2 * n), f32))
        return pseudo, {name: cp.asarray(index.astype(np.int64))
                        for name, index in maps.items()}

    def observe(self, seam) -> None:
        """Gather one step's post-RK columns (a ``phase2_observers`` hook).

        The fold waits for :meth:`commit`: a step the backend aborts never
        enters the window (:meth:`discard`)."""
        import cupy as cp

        with cp.cuda.Device(self.device_id):
            self._gather(seam)
        self._pending_dt = float(seam._dt)

    def commit(self) -> None:
        """Fold the gathered step into the window (the step committed)."""
        import cupy as cp

        if self._pending_dt is None:
            return
        with cp.cuda.Device(self.device_id):
            self.diagnostic.on_step(self._pseudo, self._pending_dt)
        self._pending_dt = None

    def discard(self) -> None:
        """Drop the gathered step (the step was rolled back)."""
        self._pending_dt = None

    def take_window(self) -> dict:
        """The window since the last frame (host), and restart it."""
        return self.diagnostic.take_window()

    # -- the frame ------------------------------------------------------------

    def frame_values(self, arrays: Mapping[str, Any], *, zgrid, terrain
                     ) -> dict:
        """Steps 3-6 of the diagnostic at the frame, on the host road.

        ``arrays`` are the history frame's ``(nVertLevels, nCells)``
        ``u_zonal``, ``v_meridional``, ``theta``, ``pressure`` and ``qv``;
        ``zgrid`` the init's ``(nCells, nVertLevelsP1)`` interface heights
        and ``terrain`` the cells' ``ter``: the mass-level heights above
        ground are ``0.5*(z_k + z_k+1) - ter`` in float32, the diagnostic's
        own formula.
        """
        from woof.core.windfarm_power_diag import evaluate_turbines

        f32 = np.float32
        cols = self.columns
        zw = np.asarray(zgrid, dtype=f32)[cols].T          # (nz+1, n)
        ter = np.asarray(terrain, dtype=f32).reshape(-1)[cols]
        z = f32(0.5) * (zw[:-1] + zw[1:]) - ter[None, :]
        columns = {"z": np.ascontiguousarray(z)}
        for key, name in (("u", "u_zonal"), ("v", "v_meridional"),
                          ("theta", "theta"), ("p", "pressure"),
                          ("qv", "qv")):
            if name not in arrays:
                if name == "qv":
                    continue
                raise ValueError(f"the history frame carries no {name}")
            columns[key] = np.ascontiguousarray(
                np.asarray(arrays[name], dtype=f32)[:, cols])
        diag = self.diagnostic
        return evaluate_turbines(
            columns, diag.targets_m, diag.weights,
            curves=diag.distinct_curves, curve_index=diag.curve_index,
            rated_w=diag.rated_w, half_rho_area=diag.half_rho_area,
            cut_out_ms=diag.cut_out_ms)

    def write_frame(self, now: Mapping[str, Any], window: Mapping[str, Any],
                    *, valid_time: str, output_dir) -> Path:
        """The frame's sidecar, as the WRF-like core writes it."""
        from woof.core.windfarm_power_diag import write_windfarm_sidecar

        # No step behind this frame (the start, or the first frame after a
        # resume): the window is the frame's own instant.
        stats = self.diagnostic.window_statistics(now, window)
        seconds = float(window["seconds"])
        when = datetime.strptime(valid_time, "%Y-%m-%d_%H:%M:%S")
        return write_windfarm_sidecar(
            self.diagnostic, now=now, window=stats, window_s=seconds,
            valid_time=when, output_dir=output_dir, grid_id=self.grid_id)


@dataclass
class _PseudoState:
    """The WRF state fields :mod:`woof.core.windfarm_power_diag` reads."""

    nz: int
    u: Any
    v: Any
    php: Any
    phb: Any
    ht: Any
    thp: Any
    thb: Any
    p: Any
    qv: Any


def attach_power(seam, *, turbine_file, placement, grid_id: int = 1):
    """Build the farm's :class:`HexWindfarmPower` and hook it to ``seam``.

    Only a woof turbine table carries the library curves the diagnostic
    integrates; a WRF ``windturbines.txt`` list (WRF tables only) gets no
    per-turbine window, and ``None`` is returned.
    """
    from woof.core.windfarm_coupling import turbine_file_kind
    from woof.windfarm.layout import build_cell_layout, read_turbines

    path = Path(turbine_file)
    if turbine_file_kind(path) == "wrf":
        return None
    layout = build_cell_layout(read_turbines(path), placement,
                               grid_id=grid_id)
    power = HexWindfarmPower(layout, n_columns=placement.n_columns,
                             grid_id=grid_id, source=str(path))
    power.bind_cells(placement)
    hook(seam, power)
    return power


def hook(seam, power) -> None:
    """Hand ``seam`` the power window's observer (once per seam: the backend
    builds a fresh seam on a resume and on a rollback)."""
    observers = tuple(getattr(seam, "phase2_observers", ()))
    if power.observe not in observers:
        seam.phase2_observers = observers + (power.observe,)


__all__ = ["HexWindfarmPower", "WINDFARM_KEYS", "attach_power", "hook",
           "interior_columns", "mpas_route_refusal", "power_turbines"]
