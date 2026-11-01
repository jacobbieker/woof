"""Opt-in Fitch turbine forcing on the model's spherical Voronoi cells.

Two schemes reach hex, both through upstream's switch:

* ``--wind-farms`` (upstream's ``fitch``): :class:`WindFarm` on the MPAS
  suite route (:func:`attach_native`, MPAS's MYNN) and the ArWen route
  (:func:`attach_arwen`), the turbines mapped to Voronoi cells once on the
  GPU (woof.core.wind_farm.attach_voronoi_wind_farm).
* ``wind_farm_scheme = "fitch_wrf"`` (WRF v4.7.1's windfarm_opt = 1 word
  for word), given as ``--physics`` overrides on WOOF's physics (the
  engine's generic column seam, :class:`woof.core.mpas_column_suite.
  MpasColumnPhysics`, whose phase one is the very ``PhysicsDriver.compute``
  the WRF-like core steps): ``woof hex forecast --physics-suite rrfs_v2_dev
  --physics wind_farm_scheme=fitch_wrf --physics wind_farms=farm.csv``
  (and ``windfarm_tke_factor``).  They are RunConfig fields, so
  :mod:`woof.hex.physics_suites` hands them to the generic seam and the
  engine's own ``validate_run_config`` judges them (MYNN is required, by
  name: ``bl_pbl_physics = 5``).  What is hex's own:

  - PLACEMENT.  A turbine goes to the column whose cell centre is nearest
    (:mod:`woof.windfarm.cells`, the rule MPAS applies to a point), and a
    cell's turbines group in table order exactly as on a WRF grid
    (:func:`woof.windfarm.layout.build_cell_layout`; a WRF
    ``windturbines.txt`` lat/lon list is read by
    :mod:`woof.core.windfarm_coupling` and placed by the same rule).
    :func:`interior_columns` says which physics columns are modelled: a
    limited-area mesh's boundary cells (``bdyMaskCell > 0``) and its padded
    garbage column are not, and a turbine there is refused by name.
  - THE DRAG.  WRF divides the rotor thrust by the cell volume
    ``dx*dx*dz``; on hex each column's own ``areaCell`` takes ``dx*dx``
    (``fitch_column_cells``, :func:`woof.core.windfarm_fitch.launch_fitch`
    with ``wfdensity``).  It runs in the PBL slot right after MYNN, adding
    into the slot's A-grid ``du``/``dv`` and MYNN's ``qke`` exactly as on
    the WRF-like core; hex then projects the cell-centre momentum rate onto
    the edges as it projects every PBL rate (``project_edges_v841_f32``,
    MPAS's ``tend_u_physics``).
  - OUTPUT.  WRF's ``POWER`` (W per cell) is the history field ``power``;
    and :class:`HexWindfarmPower` writes the per-turbine rotor-equivalent
    wind, power and thrust coefficient of a woof turbine table beside each
    history frame (``windfarm_d01_<valid time>.nc``), with the WRF-like
    core's own arithmetic (:mod:`woof.core.windfarm_power_diag`).

  REFUSED BY NAME.  The wind-farm keys as MPAS namelist / ``--physics``
  entries on the MPAS suite route (:func:`mpas_route_refusal`: MPAS's
  Registry has no such key; ``--wind-farms`` is that route's switch).  A
  PBL other than MYNN (the engine's refusal).  A windturbines-ij.txt list
  (an unstructured mesh has no i/j).  A turbine outside the mesh or in its
  boundary columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

#: The wind-farm keys an MPAS suite route has no Registry row for: WRF's
#: &physics names and WOOF's RunConfig switch (fitch_wrf's carrier).
WINDFARM_KEYS = ("windfarm_opt", "windfarm_ij", "windfarm_tke_factor",
                 "wind_farm_scheme", "wind_farms", "wind_farm_table")


def mpas_route_refusal(keys) -> str | None:
    """The refusal of a wind-farm key on the MPAS suite route, or None."""
    from woof.explain import layered

    named = sorted({str(key).strip().casefold() for key in keys}
                   & set(WINDFARM_KEYS))
    if not named:
        return None
    return layered(
        f"{', '.join(named)} on the MPAS suite route (--physics-backend "
        "mpasphys): MPAS-A v8.4.2's Registry has no wind-farm key, so a "
        "namelist entry would record a wind farm nothing reads.\n"
        "  remedy: on this route give --wind-farms (upstream's Fitch "
        "turbines in MPAS's MYNN); for WRF v4.7.1's Fitch scheme word for "
        "word, run WOOF's physics (drop --physics-backend mpasphys; e.g. "
        "--physics-suite rrfs_v2_dev, whose PBL is MYNN) with --physics "
        "wind_farm_scheme=fitch_wrf --physics wind_farms=FILE",
        "WRF calls the Fitch drag from its PBL driver right after MYNN; "
        "hex carries fitch_wrf on the engine's generic column seam, which "
        "runs that driver.")


def wrf_fitch_physics(physics) -> bool:
    """Whether a hex ``--physics`` mapping selects ``fitch_wrf``."""
    from woof.core.windfarm_coupling import wrf_fitch_selected
    return wrf_fitch_selected(dict(physics or {}))


def wrf_fitch_turbine_file(physics) -> str:
    """The ``fitch_wrf`` turbine file a hex ``--physics`` mapping names."""
    from woof.core.windfarm_coupling import turbine_file_of
    return turbine_file_of(dict(physics or {}))


@dataclass(frozen=True)
class WindFarmConfig:
    wind_farms: str = "off"
    windfarm_tke_factor: float = .25


def source_table(choice, *, year=0):
    if choice in (None, "", "off"):
        return None
    if str(choice) == "uswtdb":
        from woof.wind_inputs import inventory_table
        return str(inventory_table(year))
    return str(Path(choice).expanduser().resolve())


def require_mynn(pbl, *, interval=None):
    if pbl not in (5, "bl_mynn"):
        raise ValueError(f"Fitch turbines require MYNN; PBL {pbl!r} has no coupled turbine TKE source")
    if interval not in (None, "none", "", 0, 0.0):
        raise ValueError("Fitch turbines require an every-step PBL: a held momentum tendency would repeat drag without its matching TKE source")


class WindFarm:
    def __init__(self, choice, *, dt, latitude, longitude, area, interfaces, factor=.25):
        from woof.core.wind_farm import attach_voronoi_wind_farm
        table = source_table(choice)
        self.farm = attach_voronoi_wind_farm(
            SimpleNamespace(wind_farm_table=table, dt=dt, windfarm_tke_factor=factor),
            latitude=latitude, longitude=longitude, area=area, z_interface=interfaces)

    def history_fields(self):
        if self.farm is None:
            return {}
        import cupy as cp
        from woof.io.wrf_output_schema import REGISTRY_VAR_META
        out = {}
        for name, value in self.farm.history_fields().items():
            if name in ("POWER", "WIND_TKE_SOURCE_MAX"):
                value = value.reshape(-1)
                dims = ("nCells",)
            elif name == "WIND_TKE_SOURCE":
                value = value.reshape(self.farm.shape[0], -1)
                dims = ("nVertLevels", "nCells")
            else:
                dims = ("wind_turbine" if name.startswith("TURBINE_") else "wind_farm",)
            out[name] = (cp.asnumpy(value), dims, REGISTRY_VAR_META[name][1])
        return out

    def metadata(self):
        if self.farm is None:
            return json.dumps({"enabled": False, "reason": "empty turbine table"})
        farm = self.farm
        return json.dumps({"schema": "gpuwm-hex.wind-farms/v1", "scheme": "Fitch",
                           "turbine_count": len(farm.ids), "farm_count": len(farm.farm_ids),
                           "table_sha256": farm.table_sha256, "options": farm.options,
                           **getattr(self, '_metadata_reference', {})}, sort_keys=True)

    def write_metadata(self, directory):
        """Keep inventory-sized IDs outside HDF5's 65535-byte compact attributes."""
        if self.farm is None:
            return
        farm = self.farm
        document = {"schema": "gpuwm-hex.wind-farm-index/v1",
                    "turbine_ids": farm.ids, "farm_ids": farm.farm_ids,
                    "table_sha256": farm.table_sha256, "options": farm.options,
                    "input_provenance": farm.data.get('provenance', {})}
        payload = (json.dumps(document, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8')
        path = Path(directory) / 'wind-farms.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_bytes(payload)
        temporary.replace(path)
        self._metadata_reference = {'index_file': path.name,
                                    'index_sha256': hashlib.sha256(payload).hexdigest()}


def attach_native(arguments, session, mesh, output, *, dt, resident, coupling, halo=None):
    choice = getattr(arguments, "wind_farms", None)
    if choice in (None, '', 'off'):
        return None
    require_mynn(session.config.selectors.get("pbl"),
                 interval=session.config.options.get("config_pbl_interval"))
    if not resident or halo is not None:
        raise ValueError("Fitch turbines require resident single-rank cells; partition ownership has not been assigned and would duplicate turbine forcing")
    if session.layout.configs.get("config_apply_lbcs", False):
        raise ValueError("Fitch turbine mapping requires the complete Voronoi globe; a culled boundary needs explicit outside-domain ownership to avoid placing distant turbines on edge cells")
    p = session.device_pools
    nc = session.layout.dimensions['nCellsSolve']
    table = source_table(choice, year=int(str(session.layout.configs['config_start_time'])[:4]))
    farm = WindFarm(table, dt=dt, latitude=p.field('mesh', 'latCell')[:nc],
                    longitude=p.field('mesh', 'lonCell')[:nc], area=p.field('mesh', 'areaCell')[:nc],
                    interfaces=p.field('mesh', 'zgrid')[:, :nc])
    p.wind_farm = farm.farm
    if farm.farm is not None:
        from .cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841
        columns = prepare_mpas_to_phys_cuda_v841(coupling.dycore.atmosphere,
            scalar_names=coupling.names, geometry=coupling.geometry, kernel_cache=coupling.physics_cache)
        farm.farm.sample({'z_interface':columns.z_p, 'u':columns.u_p,
                          'v':columns.v_p, 'rho':columns.rho_p}, {})
    previous = getattr(output, 'extra_history', None)
    farm.write_metadata(output.output)
    output.extra_history = lambda: {**(previous() if previous else {}), **farm.history_fields()}
    output.extra_history_attrs = {**(getattr(output, 'extra_history_attrs', None) or {}),
                                  'WIND_FARMS': farm.metadata()}
    return farm


def apply_native(provider, fields):
    farm = getattr(provider, 'wind_farm', None)
    if farm is None:
        return
    nz = farm.shape[0]
    atmosphere = {name: fields[key][:, :nz].T for name, key in
                  (('u', 'u'), ('v', 'v'), ('rho', 'rho'))}
    atmosphere['z_interface'] = provider.prepared('z_p')[:nz+1]
    farm.apply_mynn(atmosphere,
        {'qke': fields['qke'][:, :nz].T, 'qke_adv': fields['qke_adv'][:, :nz].T},
        {'du': fields['rublten'][:, :nz].T, 'dv': fields['rvblten'][:, :nz].T},
        step_seconds=fields['delt'])


def attach_arwen(backend, host, driver, *, output_root=None):
    choice = host.get('wind_farms')
    if choice in (None, '', 'off'):
        return
    seam = backend._seam
    require_mynn(seam._cfg.bl_pbl_physics, interval=seam._cfg.bldt)
    if host.get('regional') is not None:
        raise ValueError("Fitch turbine mapping requires complete Voronoi ownership; a culled boundary could assign outside-domain turbines to edge cells")
    mesh = host['prepared'].mesh
    arrays = getattr(mesh, 'arrays', mesh)
    from woof.hex.cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841
    columns = prepare_mpas_to_phys_cuda_v841(driver.atmosphere,
        scalar_names=host['scalar_names'], geometry=backend._prep_geometry,
        kernel_cache=backend._kernel_cache)
    table = source_table(choice, year=int(host['start_time_text'][:4]))
    wind = WindFarm(table, dt=seam._cfg.dt, latitude=arrays['latCell'],
                    longitude=arrays['lonCell'], area=arrays['areaCell'], interfaces=columns.z_p)
    if wind.farm is not None:
        wind.farm.sample({'z_interface':columns.z_p, 'u':columns.u_p,
                          'v':columns.v_p, 'rho':columns.rho_p}, {})
    seam._driver.wind_farm = wind.farm
    if output_root is not None:
        wind.write_metadata(output_root)
    backend.wind_farms = wind


# --------------------------------------------------------------------------
# fitch_wrf on hex


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


__all__ = ["HexWindfarmPower", "WINDFARM_KEYS", "WindFarm", "WindFarmConfig",
           "apply_native", "attach_arwen", "attach_native", "attach_power",
           "hook", "interior_columns", "mpas_route_refusal", "power_turbines",
           "require_mynn", "source_table", "wrf_fitch_physics",
           "wrf_fitch_turbine_file"]
