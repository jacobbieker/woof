"""Partial-depth bands on MPAS-A (Phase D5): running two-way children.

plans/sorted-mapping-hummingbird.md, Phase D5 -- the hex counterpart of
:mod:`woof.core.partial_nest_run` (WRF D3/D4).  A child is a regional MPAS
mesh cut at a flat ``zgrid`` face (:mod:`woof.hex.partial_nest`); above it its
parent alone carries the atmosphere.  The child steps with the ORDINARY
v8.4.1 regional CUDA driver; the band adds new kernels
(:mod:`woof.hex.cuda_partial_nest_v841`) at fixed places in that step and
changes no byte of the dycore's units (the regional kernel set digest
``MINTED_KERNEL_SET_SHA256`` is untouched):

* the driver's own per-stage helpers -- the child driver is a
  :class:`PartialTopDriver`, a subclass that wraps ``_vertical_coefficients``,
  ``_vertical_u``, ``_launch`` (after the theta / u / w vertical-flux kernels)
  and ``_recover_candidate``;
* the kernels the acoustic step and the scalar transport resolve through the
  child's own :class:`PartialTopKernelCache`, which interposes on two
  launches: before the column solve (``pn_column_top_v841``) and after the
  scalar vertical flux (``pn_scalar_flux_top_v841``);
* the regional runtime's tendency adjustment (the top zone,
  :class:`PartialChildRuntime`).

Parent to child
---------------
At the start and end of each parent step the parent's interface planes --
``rw`` and w at the interface face ``k_i``, w at ``k_i+1``, theta_m at the
half levels ``k_i`` and ``k_i+1``, each scalar there, and u there (MPAS's
``mpas_reconstruct_2d`` zonal/meridional winds at the parent's cells,
interpolated to the child's edges and turned to the edge normal as
init_atmosphere does) -- are interpolated with MPAS's barycentric weights
(:mod:`woof.hex.partial_nest_coupling`) and linearly in time across the
parent step: the acoustic substeps take ``rw`` at each substep's end (the
specified top of the implicit w solve, ``rw_p(k_i) = rw(t) - rw_save``), the
big step takes the stage reference's values, the recovery takes w at the
stage's end.

The interface face carries the parent's own SECOND-ORDER face value,
``fzm(k_i)*q_above + fzp(k_i)*q(k_i-1)`` -- for theta in the acoustic
``coftz(k_i)`` and in the big step, for u, w and the scalars (an outgoing
scalar face capped at the donor's value, D4's cap) -- and the face below it
takes the parent's third-order interior flux, whose stencil now reaches the
parent's level above.  NOT upwinding: D4 measured that an upwinded face
carries the colder value up and the warmer down in stratification and,
two-way, ratchets heat into the top layer (``face_scheme`` "upwind" keeps it
to measure; "parent" is the parent's third-order flux at the interface
itself, which with the parent's own planes reproduces the parent).

Lateral zones: filled online through the lbc.py machinery -- each parent
step's end becomes the far end of the child's ``LbcPool`` interval
(:class:`OnlineDrivingState`: the parent's state interpolated onto every
child cell and edge below the interface, derived and differenced by the
regional unit's own ``regional_lbc_*`` kernels), so the specified and
relaxation zones run MPAS's regional arithmetic unchanged.

Top zone: the relaxation zone turned on its side.  The ``relax_levels``
layers below the interface relax ``rho_zz``, ``rho*theta_m`` and ``ru``
toward the same pool's values with ``atm_bdy_adjust_dynamics_relaxzone_tend``'s
statements (the Rayleigh terms, the Laplacian filters and the u filter's
divergence damping, ``config_relax_zone_divdamp_coef``) at ring strength
``bdyMask = 5 - d`` for the layer ``d`` below the interface, on every column
the lateral zones do not own.

Mass budget: barycentric interpolation is not conservative, so the child's
interpolated interface flux summed over its cells is not the parent's flux
through the area it covers; the uniform residual (one fixed-order binary64
sum over the overlap areas, ``pn_budget_residual_v841``) is added to the
interface ``rw`` itself -- a face-flux correction, as D4's ``mass_route
"face"`` (the column-source route tripled the bubble's error in D3).

Child to parent: theta_m and the scalars only, below the interface, on the
parent cells the child's interior covers wholly, by the conservative overlap
remap (mass-weighted by the child's ``rho_zz``; binary64,
``pn_remap_rows_v841``).  The fed-back theta enters the parent as an
``rtheta`` increment through the dycore's own ``recover_cells_f32`` (theta,
exner and p' in the dycore's arithmetic).  Never rho_zz, w or the coordinate.

Chains: band ``n``'s parent is band ``n-1``.  MPAS's ``rw`` is prognostic, so
a partial parent's ``rw`` below its own interface is already the solution
closed on that parent's interface flux; each band's snapshot reads it as it
stands.

Scope, refused by name (:func:`woof.hex.partial_nest.refuse_unsupported_config`):
the v8.4.1 split-three dry or moist dycore without physics callbacks, the
unlimited scalar transport, no upper u Rayleigh / CAM damping / w damping
below the interface, no 3-D mixing or LES through the interface, no fused
acoustic unit.  Radiation on the extended column: :mod:`woof.hex.partial_nest_physics`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from .cuda_backend import KernelCache
from .cuda_driver import CudaDryDycoreDriver
from .cuda_partial_nest_v841 import PartialNestKernels
from .cuda_regional_forecast_v841 import (CudaRegionalRuntimeV841,
                                          PaddedRegionalDrivingState)
from .partial_nest import (FACE_SCHEMES, HexPartialNestBand, PartialNestRefused,
                           refuse_damping_below, refuse_unsupported_config)
from .regional_v841 import N_RELAX_ZONE

RVORD = np.float32(461.6 / 287.0)
I32, F32 = np.int32, np.float32


def _cp():
    import cupy
    return cupy


# ---------------------------------------------------------------------------
# kernel interposition
# ---------------------------------------------------------------------------

class _HookedKernel:
    """A resolved kernel with the band's work before and after each launch."""

    def __init__(self, kernel: Any, name: str, cache: "PartialTopKernelCache") -> None:
        self._kernel = kernel
        self._name = name
        self._cache = cache

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kernel, name)

    def __call__(self, grid: Any, block: Any, args: Any, **kwargs: Any) -> Any:
        hooks = self._cache.pn_hooks
        if hooks is not None:
            hooks.before(self._name, args)
        result = self._kernel(grid, block, args, **kwargs)
        if hooks is not None:
            hooks.after(self._name, args)
        return result


class PartialTopKernelCache(KernelCache):
    """The child driver's kernel cache: every kernel resolves exactly as in
    :class:`KernelCache`; the ones the interface needs to see are wrapped."""

    HOOKED = frozenset({
        "acoustic_prepare_v841", "acoustic_column_solve_v841",
        "acoustic_column_pre_v841", "acoustic_column_solve_regional_v841",
        "transport_vertical_flux",
    })

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pn_hooks: Any = None
        self._pn_wrapped: dict[str, Any] = {}

    def _observe(self, kernel_key: str, kernel: Any, name: str = "", module_key: str = "") -> Any:
        inner = super()._observe(kernel_key, kernel, name, module_key)
        if name not in self.HOOKED:
            return inner
        wrapped = self._pn_wrapped.get(kernel_key)
        if wrapped is None or wrapped._kernel is not inner:
            wrapped = _HookedKernel(inner, name, self)
            self._pn_wrapped[kernel_key] = wrapped
        return wrapped


# ---------------------------------------------------------------------------
# the online lateral-boundary pool (lbc.py's machinery fed from memory)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class OnlineFrame:
    """One boundary time from the parent, device arrays in the child's padded
    (nlev, n+1) layout: u (edges), w (faces), rho, theta (dry), scalars."""

    valid_time: datetime
    fields: dict


class OnlinePool:
    """``LbcPool``'s admission rules over frames held in memory:
    ``start`` takes the latest frame at or before the time,
    ``advance`` the earliest strictly after the interval's end."""

    def __init__(self) -> None:
        self.frames: dict[datetime, OnlineFrame] = {}
        self._state: OnlineFrame | None = None
        self._interval_start: datetime | None = None

    def push(self, frame: OnlineFrame) -> None:
        self.frames[frame.valid_time] = frame
        # Keep what an interval can still need: the current end and later.
        if self._state is not None:
            for when in [t for t in self.frames if t < self._state.valid_time]:
                del self.frames[when]

    def start(self, when: datetime) -> OnlineFrame:
        times = [t for t in self.frames if t <= when]
        if not times:
            raise PartialNestRefused(f"no parent frame at or before {when}")
        self._state = self.frames[max(times)]
        return self._state

    def advance(self, when: datetime | None = None) -> OnlineFrame:
        clock = self._state.valid_time if when is None else when
        times = [t for t in self.frames if t > clock]
        if not times:
            raise PartialNestRefused(
                f"the child reached {clock} with no parent frame after it: the "
                "parent steps (and FORCE runs) before its child")
        self._interval_start = self._state.valid_time
        self._state = self.frames[min(times)]
        return self._state

    @property
    def interval_start(self) -> datetime:
        return self._interval_start

    @property
    def interval_end(self) -> datetime:
        return self._state.valid_time


class OnlineDrivingState(PaddedRegionalDrivingState):
    """The regional driving state over :class:`OnlinePool`: frames arrive as
    device arrays already in the padded layout, so ``_derive`` skips only
    the file upload and runs the regional unit's own derivation kernels."""

    def _derive(self, admitted: OnlineFrame) -> tuple[dict[str, Any], Any]:
        cp = _cp()
        f = admitted.fields
        u, w, rho, theta = f["u"], f["w"], f["rho"], f["theta"]
        scalars = cp.ascontiguousarray(f["scalars"])
        qv = cp.ascontiguousarray(scalars[0])
        rho_zz = cp.empty(self._cell_shape, dtype=cp.float32)
        rtheta_m = cp.empty(self._cell_shape, dtype=cp.float32)
        self._kernels.launch(
            "regional_lbc_derive_v841", self.nlev * self.ncells,
            (I32(self.nlev * self.ncells), F32(RVORD), self._zz, rho, theta, qv,
             rho_zz, rtheta_m))
        rho_edge = cp.array(self._rho_edge_zero, copy=True)
        ru = cp.empty(self._edge_shape, dtype=cp.float32)
        self._kernels.launch(
            "regional_lbc_rho_edge_v841", self.nedges,
            (I32(self.nlev), I32(self.ncells), I32(self.nedges), self._cells_on_edge,
             rho_zz, u, rho_edge, ru))
        return ({"u": u, "ru": ru, "rho_edge": rho_edge, "w": w, "rho": rho,
                 "rho_zz": rho_zz, "theta": theta, "rtheta_m": rtheta_m}, scalars)


class _OnlineInventory:
    """Stands in for ``LbcInventory`` while the runtime is constructed; the
    pool is replaced before any admission."""

    def __init__(self, paths: Any) -> None:
        self.paths = paths


@contextlib.contextmanager
def _online_construction():
    from . import cuda_regional_forecast_v841 as regional
    saved = regional.LbcInventory
    regional.LbcInventory = _OnlineInventory
    try:
        yield
    finally:
        regional.LbcInventory = saved


# ---------------------------------------------------------------------------
# the child driver and runtime
# ---------------------------------------------------------------------------

class PartialTopDriver(CudaDryDycoreDriver):
    """The child's v8.4.1 driver with its interface (module docstring).  An
    instance is a :class:`CudaDryDycoreDriver` built as usual whose class is
    then set to this one (:func:`make_partial_driver`); ``pn`` is its
    :class:`ChildTop`."""

    pn: "ChildTop"

    def _step_device_v841(self, *args: Any, **kwargs: Any) -> Any:
        self.pn.begin_step()
        return super()._step_device_v841(*args, **kwargs)

    def _advance_dynamics_subcycle_v841(self, state: Any, **kwargs: Any) -> Any:
        self.pn.begin_subcycle(state, kwargs["schedule"], int(kwargs.get("dynamics_substep", 1)))
        return super()._advance_dynamics_subcycle_v841(state, **kwargs)

    def _vertical_coefficients(self, state: Any, dts: float, theta: Any, *args: Any,
                               **kwargs: Any) -> Any:
        coefficients = super()._vertical_coefficients(state, dts, theta, *args, **kwargs)
        self.pn.coefficients_top(coefficients, dts, theta)
        return coefficients

    def _vertical_u(self, u: Any, rw: Any) -> Any:
        self.pn.begin_stage()
        return super()._vertical_u(u, rw)

    def _launch(self, name: str, count: int, args: tuple) -> None:
        super()._launch(name, count, args)
        if name in ("vertical_u_flux_f32", "theta_vertical_flux_f32", "w_vertical_flux_f32"):
            self.pn.big_step_flux(name, args)

    def _recover_candidate(self, saved_state: Any, saved_diag: Any, acoustic: Any,
                           stage: int, acoustic_steps: int, scalars: Any, **kwargs: Any):
        result = super()._recover_candidate(saved_state, saved_diag, acoustic, stage,
                                            acoustic_steps, scalars, **kwargs)
        state, diag, flux_u, flux_w = result
        self.pn.recover_top(saved_state, acoustic, acoustic_steps, state, diag, flux_w)
        return result


class PartialChildRuntime(CudaRegionalRuntimeV841):
    """The child's regional runtime with the top zone after MPAS's own
    tendency adjustment (see :class:`PartialTopDriver` for how it is made)."""

    pn: "ChildTop"

    def adjust_dynamics_tendencies(self, **kwargs: Any) -> None:
        super().adjust_dynamics_tendencies(**kwargs)
        self.pn.top_zone(kwargs)


# ---------------------------------------------------------------------------
# the interface, as the child's hooks
# ---------------------------------------------------------------------------

#: Cell planes of a snapshot, stacked: rw(k_i), w(k_i), w(k_i+1), theta_m at
#: the half levels k_i and k_i+1.
CELL_PLANES = ("rw", "w", "w1", "th", "th1")


@dataclasses.dataclass
class Planes:
    cells: Any       # (len(CELL_PLANES), ncells_pad)
    edges: Any       # (2, nedges_pad): u at k_i, k_i+1
    scalars: Any     # (2, nscalars, ncells_pad): q at k_i, k_i+1

    def plane(self, name: str) -> Any:
        return self.cells[CELL_PLANES.index(name)]


class ChildTop:
    """The child's interface: the per-step clock, the stage forcing and the
    launches every hook makes (module docstring)."""

    def __init__(self, nest: "PartialNestHex") -> None:
        cp = _cp()
        self.nest = nest
        self.k = nest.k
        c = nest.child_driver
        self.nc, self.ne = int(c.ncells), int(c.nedges)
        self.kernels = nest.child_kernels
        z = lambda *s: cp.zeros(s, dtype=cp.float32)
        self.stage_planes = z(len(CELL_PLANES), self.nc)
        self.stage_edges = z(2, self.ne)
        self.saved_th = z(self.nc)               # theta_m(k_i) at the subcycle start
        self.ns = int(c.atmosphere.state.scalars.shape[0])
        self.stage_q = z(2, max(self.ns, 1), self.nc)
        self.scheme = I32(FACE_SCHEMES[nest.band.face_scheme])
        self.step_frac0 = 0.0      # fraction of the parent step at the child step's start
        self.step_span = 1.0       # child dt / parent dt
        self.sub_frac0 = 0.0
        self.schedule = None
        self.stage = 0
        self.coefficient_calls = 0
        self.small_step = 1
        self.scalar_stage = 0
        self.rw_save = None
        self.cache = nest.child_cache
        #: Hooks switched off by name (measurement and diagnosis only):
        #: coefficients, column, flux, scalar_flux, recover, top_zone.
        self.disabled: set[str] = set()

    # -- the clock -----------------------------------------------------------
    def _frac(self, seconds_into_step: float) -> float:
        return self.step_frac0 + self.step_span * seconds_into_step / float(
            self.nest.child_config.config_dt)

    def begin_step(self) -> None:
        self.scalar_stage = 0
        self.cache.pn_hooks = self

    def begin_subcycle(self, state: Any, schedule: Any, substep: int) -> None:
        self.schedule = schedule
        dt_sub = float(self.nest.child_config.config_dt) / int(schedule.dynamics_splits)
        self.sub_start = (substep - 1) * dt_sub
        self.stage = 0
        self.coefficient_calls = 0
        self.rw_save = state.rho_w
        # theta_m above at the subcycle start (the saved theta's partner).
        self._interp_planes(self._frac(self.sub_start), cells=("th",), out_cells=None)
        cp = _cp()
        self.saved_th[...] = self._tmp_cells[CELL_PLANES.index("th")]

    def _stage_reference_seconds(self, stage: int) -> float:
        stages = self.schedule.stages
        return self.sub_start + (0.0 if stage == 1 else float(stages[stage - 2].large_timestep))

    def _interp_planes(self, frac: float, *, cells=CELL_PLANES, out_cells=None) -> None:
        """Every plane at ``frac`` into the stage buffers (one launch per
        stacked group)."""
        a, b = self.nest.a, self.nest.b
        if out_cells is None:
            out_cells = self.stage_planes
        self._tmp_cells = out_cells
        k = self.kernels
        n = int(a.cells.size)
        k.launch("pn_lerp_v841", n, (I32(n), F32(frac), a.cells, b.cells, out_cells))
        n = int(a.edges.size)
        k.launch("pn_lerp_v841", n, (I32(n), F32(frac), a.edges, b.edges, self.stage_edges))
        if self.ns:
            n = int(a.scalars.size)
            k.launch("pn_lerp_v841", n, (I32(n), F32(frac), a.scalars, b.scalars, self.stage_q))

    def begin_stage(self) -> None:
        self.stage += 1
        self._interp_planes(self._frac(self._stage_reference_seconds(self.stage)))

    # -- the acoustic step -------------------------------------------------
    def coefficients_top(self, coefficients: Any, dts: float, theta: Any) -> None:
        """After acoustic_coefficients_v841: the interface rows with theta
        above at the time ``theta`` stands for (the subcycle start, or the
        stage-2 reference on RK3's second solve)."""
        if "coefficients" in self.disabled:
            return
        self.coefficient_calls += 1
        cp = _cp()
        if self.coefficient_calls == 1:
            above = self.saved_th
        else:
            buffer = cp.empty_like(self.stage_planes)
            self._interp_planes(self._frac(self._stage_reference_seconds(2)), out_cells=buffer)
            above = buffer[CELL_PLANES.index("th")]
        c = self.nest.child_driver
        ctx = c.v841_context
        v = c.atmosphere.vertical
        self.kernels.launch("pn_coefficients_top_v841", self.nc, (
            I32(self.k), I32(self.nc), F32(dts), self.nest.fzm_top, self.nest.fzp_top,
            theta, cp.ascontiguousarray(above), v.zz, v.rdzw, ctx.etp, ctx.ewp,
            coefficients.cofwz, coefficients.cofwr, coefficients.cofwt, coefficients.cofrz,
            coefficients.alpha_tri, coefficients.coftz, coefficients.gamma_tri))

    def before(self, name: str, args: tuple) -> None:
        if name == "acoustic_prepare_v841":
            self.small_step = int(args[2])
        elif "column" in self.disabled:
            return
        elif name in ("acoustic_column_solve_v841", "acoustic_column_solve_regional_v841",
                      "acoustic_column_pre_v841"):
            dts = float(args[2])
            stage = self.schedule.stages[self.stage - 1]
            frac = self._frac(self.sub_start + self.small_step * dts)
            ctx = self.nest.child_driver.v841_context
            rw_p, ww_avg = args[-4], args[-1]
            a, b = self.nest.a, self.nest.b
            self.kernels.launch("pn_column_top_v841", self.nc, (
                I32(self.k), I32(self.nc), F32(frac), a.plane("rw"), b.plane("rw"),
                self.rw_save, ctx.ewm, ctx.ewp, rw_p, ww_avg))
            del stage

    def after(self, name: str, args: tuple) -> None:
        if name == "transport_vertical_flux":
            self.scalar_stage += 1
            if "scalar_flux" in self.disabled:
                return
            ntracers, nlev = int(args[0]), int(args[1])
            if nlev != self.k or ntracers != self.ns:
                return
            dt = float(self.nest.child_config.config_dt)
            offsets = (0.0, dt / 3.0, dt / 2.0)
            frac = self._frac(offsets[min(self.scalar_stage, 3) - 1])
            q = _cp().empty_like(self.stage_q)
            a, b = self.nest.a, self.nest.b
            n = int(a.scalars.size)
            self.kernels.launch("pn_lerp_v841", n, (I32(n), F32(frac), a.scalars, b.scalars, q))
            self.kernels.launch("pn_scalar_flux_top_v841", ntracers * self.nc, (
                I32(ntracers), I32(self.k), I32(self.nc), F32(args[3]), self.scheme,
                self.nest.fzm_top, self.nest.fzp_top, args[4], args[5],
                q[0], q[1], args[8]))

    # -- the big step ---------------------------------------------------------
    def big_step_flux(self, name: str, args: tuple) -> None:
        if "flux" in self.disabled:
            return
        p = self.stage_planes
        th, th1 = p[CELL_PLANES.index("th")], p[CELL_PLANES.index("th1")]
        if name == "theta_vertical_flux_f32":
            self.kernels.launch("pn_theta_flux_top_v841", self.nc, (
                I32(self.k), I32(self.nc), F32(args[2]), self.scheme,
                self.nest.fzm_top, self.nest.fzp_top, args[3], args[4], args[5], args[6],
                args[7], args[8], th, th1, self.saved_th, args[9]))
        elif name == "vertical_u_flux_f32":
            self.kernels.launch("pn_u_flux_top_v841", self.ne, (
                I32(self.k), I32(self.nc), I32(self.ne), self.scheme,
                self.nest.fzm_top, self.nest.fzp_top, args[3], args[4], args[5],
                self.stage_edges[0], self.stage_edges[1], args[8]))
        elif name == "w_vertical_flux_f32":
            self.kernels.launch("pn_w_flux_top_v841", self.nc, (
                I32(self.k), I32(self.nc), self.scheme, args[2], args[3],
                p[CELL_PLANES.index("w1")], args[4]))

    def recover_top(self, saved_state: Any, acoustic: Any, acoustic_steps: int,
                    state: Any, diag: Any, flux_w: Any) -> None:
        if "recover" in self.disabled:
            return
        stage = self.schedule.stages[self.stage - 1]
        frac = self._frac(self.sub_start + float(stage.large_timestep))
        a, b = self.nest.a, self.nest.b
        self.kernels.launch("pn_recover_top_v841", self.nc, (
            I32(self.k), I32(self.nc), I32(acoustic_steps), F32(frac),
            saved_state.rho_w, acoustic.rw_p, acoustic.ww_avg,
            a.plane("w"), b.plane("w"), state.rho_w, flux_w, diag.vertical_velocity))

    # -- the top zone --------------------------------------------------------
    def top_zone(self, kw: dict) -> None:
        if "top_zone" in self.disabled:
            return
        runtime = self.nest.child_driver.regional_v841
        rk_step = int(kw["rk_step"])
        values_ru = runtime.driving_values("ru", rk_step)
        values_rt = runtime.driving_values("rtheta_m", rk_step)
        values_rho = runtime.driving_values("rho_zz", rk_step)
        dt = np.float32(runtime.outer_dt)
        fifty_dt = F32(F32(50.0) * dt)
        ten_dt = F32(F32(10.0) * dt)
        relax_zone = F32(N_RELAX_ZONE)
        levels = I32(self.nest.band.relax_levels)
        cols_c, cols_e = self.nest.top_cells, self.nest.top_edges
        tend_rho, tend_rt, tend_ru = kw["tend_rho"], kw["tend_rt"], kw["tend_ru"]
        rho_zz, theta_m, rho_u = kw["rho_zz"], kw["theta_m"], kw["rho_u"]
        nc, ne = int(tend_rho.shape[-1]), int(tend_ru.shape[-1])
        mesh = runtime.atmosphere.mesh
        ctx = runtime.context
        k = self.kernels
        k.launch("pn_topzone_rayleigh_cell_v841", int(levels) * len(cols_c), (
            I32(self.k), I32(nc), levels, I32(len(cols_c)), fifty_dt, relax_zone, cols_c,
            runtime.mesh_scaling_cell, rho_zz, theta_m, values_rho, values_rt,
            tend_rho, tend_rt))
        k.launch("pn_topzone_rayleigh_edge_v841", int(levels) * len(cols_e), (
            I32(self.k), I32(ne), levels, I32(len(cols_e)), fifty_dt, relax_zone, cols_e,
            runtime.mesh_scaling_edge, rho_u, values_ru, tend_ru))
        k.launch("pn_topzone_filter_cell_v841", int(levels) * len(cols_c), (
            I32(self.k), I32(nc), I32(mesh.max_edges), levels, I32(len(cols_c)), ten_dt,
            relax_zone, cols_c, runtime.mesh_scaling_cell, mesh.n_edges_on_cell,
            mesh.edges_on_cell, mesh.cells_on_edge, mesh.edge_sign_on_cell, mesh.dv_edge,
            ctx.inv_dc_edge, rho_zz, theta_m, values_rho, values_rt, tend_rho, tend_rt))
        k.launch("pn_topzone_filter_edge_v841", int(levels) * len(cols_e), (
            I32(self.k), I32(ne), I32(mesh.max_edges), I32(mesh.vertex_degree), levels,
            I32(len(cols_e)), ten_dt, F32(runtime.config_relax_zone_divdamp_coef),
            relax_zone, cols_e, runtime.mesh_scaling_edge, mesh.n_edges_on_cell,
            mesh.edges_on_cell, mesh.cells_on_edge, mesh.vertices_on_edge,
            mesh.edges_on_vertex, mesh.edge_sign_on_cell, runtime.edge_sign_on_vertex,
            mesh.dc_edge, mesh.dv_edge, ctx.inv_dc_edge, ctx.inv_dv_edge,
            ctx.inv_area_cell, ctx.inv_area_triangle, rho_u, values_ru, tend_ru))


# ---------------------------------------------------------------------------
# the coupled pair
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ChildBuild:
    """Everything the host builds for a child before the coupler: its regional
    mesh, its float64 geometry, and its truncated host fields."""

    mesh: Any
    geometry: Any                 # partial_nest_coupling.CellGeometry
    edge_xyz: np.ndarray          # (nEdges, 3) float64 edge midpoints
    state: Any
    vertical: Any
    reference: Any
    saved: Any
    terrain: Any
    wind: Any
    config: Any
    start_time: datetime


def make_partial_driver(build: ChildBuild, *, kernel_cache: PartialTopKernelCache):
    """The child's regional v8.4.1 driver with the band's classes."""
    from .cuda_regional_forecast_v841 import assemble_regional_driver_v841
    with _online_construction():
        driver = assemble_regional_driver_v841(
            build.mesh, build.state, build.vertical, build.reference, build.saved,
            build.terrain, build.config, reference_wind_profiles=build.wind,
            lbc_paths=["online"], start_time=build.start_time, kernel_cache=kernel_cache,
            scalar_names=("lbc_qv",) + tuple(
                f"lbc_q{n}" for n in ()))
    driver.__class__ = PartialTopDriver
    runtime = driver.regional_v841
    runtime.__class__ = PartialChildRuntime
    old = runtime.driving
    pool = OnlinePool()
    online = OnlineDrivingState.__new__(OnlineDrivingState)
    online.__dict__.update(old.__dict__)
    online._pool = pool
    runtime.driving = online
    return driver, pool


class PartialNestHex:
    """One parent driver and one partial child, coupled (module docstring).

    ``parent_driver``: a global v8.4.1 driver or a :class:`PartialTopDriver`
    (a chain); ``parent_geometry``: its cells in binary64
    (:class:`~woof.hex.partial_nest_coupling.CellGeometry`); ``parent_mesh``:
    its host mesh (for the reconstruction coefficients and the trig of its
    cells); ``build``: the child (:class:`ChildBuild`)."""

    def __init__(self, parent_driver: Any, parent_mesh: Any, parent_geometry: Any,
                 build: ChildBuild, band: HexPartialNestBand, *,
                 parent_nest: "PartialNestHex | None" = None,
                 parent_vertical: Any = None) -> None:
        cp = _cp()
        from .partial_nest_coupling import BudgetCover, FeedbackRemap, bary_table
        from .rust_geometry import reconstruction_coefficients
        self.band = band
        self.k = k = int(band.interface_level)
        self.parent_driver = parent_driver
        self.parent_nest = parent_nest
        pcfg = parent_driver.config
        ccfg = build.config
        refuse_unsupported_config(pcfg, role="parent")
        refuse_unsupported_config(ccfg, role="child")
        if int(parent_driver.nlev) <= k + 1:
            raise PartialNestRefused(
                f"the interface (face {k}) needs two parent layers above it; the parent "
                f"has {parent_driver.nlev}")
        if int(build.state.rho.shape[0]) != k:
            raise PartialNestRefused(
                f"the child has {build.state.rho.shape[0]} layers; the band's interface "
                f"is face {k}")
        zg = cp.asnumpy(parent_driver.atmosphere.vertical.zgrid)
        n_parent = int(getattr(getattr(parent_driver, "regional_v841", None), "n_cells_solve",
                               zg.shape[1]))
        from .partial_nest import interface_height
        self.z_i = interface_height(zg[:, :n_parent], k, what="parent")
        cz = np.asarray(build.vertical.zgrid)
        if cz.shape[0] != k + 1 or np.any(cz[k] != self.z_i):
            raise PartialNestRefused(
                "the child's top face is not the parent's flat interface face "
                f"({self.z_i:.0f} m)")
        refuse_damping_below(pcfg, self.z_i)
        refuse_damping_below(ccfg, self.z_i)
        ratio = int(band.parent_time_step_ratio)
        if abs(float(ccfg.config_dt) * ratio - float(pcfg.config_dt)) > 1e-9 * float(pcfg.config_dt):
            raise PartialNestRefused("child config_dt times the ratio is not the parent's")
        fzm = np.asarray(cp.asnumpy(parent_driver.atmosphere.vertical.fzm), np.float32)
        fzp = np.asarray(cp.asnumpy(parent_driver.atmosphere.vertical.fzp), np.float32)
        self.fzm_top, self.fzp_top = F32(fzm[k]), F32(fzp[k])
        self.child_cache = PartialTopKernelCache()
        self.child_driver, self.pool = make_partial_driver(build, kernel_cache=self.child_cache)
        self.child_config = ccfg
        self.child_kernels = PartialNestKernels(self.child_cache)
        self.parent_kernels = PartialNestKernels(parent_driver.cache)
        self.start_time = build.start_time
        self.parent_steps_done = 0
        runtime = self.child_driver.regional_v841
        self.nc_solve = int(runtime.n_cells_solve)
        self.ne_solve = int(runtime.n_edges_solve)
        self.nc = int(self.child_driver.ncells)
        self.ne = int(self.child_driver.nedges)

        # -- barycentric tables (child cells and edge midpoints on the parent)
        t0 = time.perf_counter()
        self.bary_cells = bary_table(parent_geometry, build.geometry.x_cell)
        self.bary_edges = bary_table(parent_geometry, build.edge_xyz)
        self.d_cells = (cp.asarray(self.bary_cells.cells32), cp.asarray(self.bary_cells.weights32))
        self.d_edges = (cp.asarray(self.bary_edges.cells32), cp.asarray(self.bary_edges.weights32))
        # -- the parent's reconstruction and the child's edge directions
        pm = parent_mesh.arrays
        coeffs = np.asarray(pm["coeffs_reconstruct"] if "coeffs_reconstruct" in pm
                            else reconstruction_coefficients(parent_mesh), np.float32)
        npc = len(np.asarray(pm["nEdgesOnCell"]))
        self.p_coeffs = cp.asarray(np.ascontiguousarray(coeffs.reshape(npc, -1, 3)))
        lat = np.asarray(pm["latCell"], np.float64)
        lon = np.asarray(pm["lonCell"], np.float64)
        trig = np.stack([np.cos(lat), np.sin(lat), np.cos(lon), np.sin(lon)], axis=1)
        self.p_trig = cp.asarray(np.ascontiguousarray(trig.astype(np.float32)))
        self.p_eoc = cp.asarray(np.ascontiguousarray(np.maximum(
            np.asarray(pm["edgesOnCell"], np.int64), 0).astype(np.int32)))
        self.p_noc = cp.asarray(np.asarray(pm["nEdgesOnCell"], np.int32))
        self.p_max_edges = int(np.asarray(pm["edgesOnCell"]).shape[1])
        angle = np.asarray(build.mesh.arrays["angleEdge"], np.float64)
        self.c_cos = cp.asarray(np.cos(angle).astype(np.float32))
        self.c_sin = cp.asarray(np.sin(angle).astype(np.float32))
        # -- feedback and budget tables
        bmask = np.asarray(build.mesh.arrays["bdyMaskCell"])
        interior = np.flatnonzero(bmask == 0)
        self.remap = FeedbackRemap.build(parent_geometry, build.geometry, interior)
        self.cover = BudgetCover.build(parent_geometry, build.geometry)
        cover_full = np.zeros(npc, np.float64)
        cover_full[self.cover.parents] = self.cover.cover
        self.d_cover = cp.asarray(cover_full)
        self.d_child_area = cp.asarray(self.cover.child_area)
        self.d_remap = (cp.asarray(self.remap.row_ptr.astype(np.int32)),
                        cp.asarray(self.remap.cols.astype(np.int32)),
                        cp.asarray(self.remap.overlap),
                        cp.asarray(self.remap.targets.astype(np.int32)))
        # -- the top zone's columns: those no lateral zone owns
        emask = np.asarray(build.mesh.arrays["bdyMaskEdge"])
        self.top_cells = cp.asarray(np.flatnonzero(bmask <= 1).astype(np.int32))
        self.top_edges = cp.asarray(np.flatnonzero(emask <= 1).astype(np.int32))
        self.table_seconds = time.perf_counter() - t0

        self.ns = int(self.child_driver.atmosphere.state.scalars.shape[0])
        if int(parent_driver.atmosphere.state.scalars.shape[0]) != self.ns:
            raise PartialNestRefused("parent and child carry different scalar sets")
        mk = lambda: Planes(cp.zeros((len(CELL_PLANES), self.nc), cp.float32),
                            cp.zeros((2, self.ne), cp.float32),
                            cp.zeros((2, max(self.ns, 1), self.nc), cp.float32))
        self.a, self.b = mk(), mk()
        self._residual = cp.zeros((1,), cp.float32)
        self._max_residual = cp.zeros((1,), cp.float32)
        self.top = ChildTop(self)
        self.child_driver.pn = self.top
        self.child_driver.regional_v841.pn = self.top
        self.stats = {"forces": 0, "child_steps": 0, "feedbacks": 0, "parent_seconds": 0.0,
                      "child_seconds": 0.0, "coupling_seconds": 0.0}
        # The start: planes and the boundary frame at t0, and the child's own
        # interface row set to them.
        self.snapshot(self.a)
        self.pool.push(self.frame(self.start_time))
        st, sv = self.child_driver.atmosphere.state, self.child_driver.atmosphere.saved
        for field, plane in ((st.rho_w, "rw"), (sv.vertical_velocity, "w")):
            self.child_kernels.launch("pn_set_top_v841", self.nc, (
                I32(k), I32(self.nc), F32(0.0), self.a.plane(plane), self.a.plane(plane), field))

    # -- parent -> child -----------------------------------------------------
    def _bary(self, table, src: Any, k0: int, nlev: int, npoints: int, stride: int, out: Any):
        cells, weights = table
        self.parent_kernels.launch("pn_bary_apply_v841", nlev * npoints, (
            I32(nlev), I32(npoints), I32(src.shape[-1]), I32(k0), I32(stride),
            cells, weights, src, out))

    def _parent_winds(self, levels: slice):
        """The parent's zonal / meridional winds at its cells for ``levels``."""
        cp = _cp()
        p = self.parent_driver.atmosphere
        u = p.saved.normal_velocity
        nlev = levels.stop - levels.start
        npc = int(self.p_noc.shape[0])
        src = cp.ascontiguousarray(u[levels])
        uz = cp.empty((nlev, int(u.shape[1]) if False else npc), cp.float32)
        um = cp.empty_like(uz)
        self.parent_kernels.launch("pn_reconstruct_v841", nlev * npc, (
            I32(nlev), I32(npc), I32(src.shape[1]), I32(self.p_max_edges), self.p_noc,
            self.p_eoc, self.p_coeffs, self.p_trig, src, uz, um))
        return uz, um

    def _child_edge_normal(self, uz: Any, um: Any, nlev: int, out: Any) -> None:
        cp = _cp()
        ze = cp.zeros((nlev, self.ne), cp.float32)
        me = cp.zeros_like(ze)
        self._bary(self.d_edges, uz, 0, nlev, self.ne_solve, self.ne, ze)
        self._bary(self.d_edges, um, 0, nlev, self.ne_solve, self.ne, me)
        self.parent_kernels.launch("pn_edge_normal_v841", nlev * self.ne, (
            I32(nlev), I32(self.ne), self.c_cos_pad, self.c_sin_pad, ze, me, out))

    @property
    def c_cos_pad(self):
        cp = _cp()
        if not hasattr(self, "_c_cos_pad"):
            self._c_cos_pad = cp.zeros((self.ne,), cp.float32)
            self._c_cos_pad[:self.ne_solve] = self.c_cos
            self._c_sin_pad = cp.zeros((self.ne,), cp.float32)
            self._c_sin_pad[:self.ne_solve] = self.c_sin
        return self._c_cos_pad

    @property
    def c_sin_pad(self):
        self.c_cos_pad
        return self._c_sin_pad

    def snapshot(self, dst: Planes) -> None:
        """The parent's interface planes now, on the child grid."""
        cp = _cp()
        k = self.k
        p = self.parent_driver.atmosphere
        nc_s = self.nc_solve
        stack = [(p.state.rho_w, k), (p.saved.vertical_velocity, k),
                 (p.saved.vertical_velocity, k + 1), (p.saved.theta_m, k),
                 (p.saved.theta_m, k + 1)]
        for n, (field, level) in enumerate(stack):
            self._bary(self.d_cells, field, level, 1, nc_s, self.nc, dst.cells[n])
        uz, um = self._parent_winds(slice(k, k + 2))
        self._child_edge_normal(uz, um, 2, dst.edges)
        for s in range(self.ns):
            for j in range(2):
                self._bary(self.d_cells, p.state.scalars[s], k + j, 1, nc_s, self.nc,
                           dst.scalars[j, s])
        if self.band.mass_correction:
            row = cp.ascontiguousarray(p.state.rho_w[k])
            self.parent_kernels.launch("pn_budget_residual_v841", 1, (
                I32(int(self.d_cover.shape[0])), I32(nc_s), self.d_cover, row,
                self.d_child_area, cp.ascontiguousarray(dst.cells[0, :nc_s]), self._residual))
            self.parent_kernels.launch("pn_add_scalar_v841", nc_s, (
                I32(nc_s), self._residual, dst.cells[0]))
            cp.maximum(self._max_residual, cp.abs(self._residual), out=self._max_residual)

    @property
    def max_abs_mass_residual(self) -> float:
        return float(_cp().asnumpy(self._max_residual)[0])

    def frame(self, when: datetime) -> OnlineFrame:
        """The parent below the interface on every child cell and edge: the
        lateral pool's frame for ``when``."""
        cp = _cp()
        k = self.k
        p = self.parent_driver.atmosphere
        nlp, npc = int(p.state.rho.shape[0]), int(p.state.rho.shape[1])
        rho_phys = cp.empty((nlp, npc), cp.float32)
        theta_dry = cp.empty_like(rho_phys)
        qv = p.state.scalars[0] if self.ns else cp.zeros_like(rho_phys)
        n = nlp * npc
        self.parent_kernels.launch("pn_frame_cells_v841", n, (
            I32(n), F32(RVORD), p.state.rho, p.vertical.zz, p.saved.theta_m,
            cp.ascontiguousarray(qv), rho_phys, theta_dry))
        nc, ne, ncs = self.nc, self.ne, self.nc_solve
        rho = cp.zeros((k, nc), cp.float32)
        theta = cp.zeros((k, nc), cp.float32)
        w = cp.zeros((k + 1, nc), cp.float32)
        self._bary(self.d_cells, rho_phys, 0, k, ncs, nc, rho)
        self._bary(self.d_cells, theta_dry, 0, k, ncs, nc, theta)
        self._bary(self.d_cells, p.saved.vertical_velocity, 0, k + 1, ncs, nc, w)
        scalars = cp.zeros((max(self.ns, 1), k, nc), cp.float32)
        for s in range(self.ns):
            self._bary(self.d_cells, p.state.scalars[s], 0, k, ncs, nc, scalars[s])
        uz, um = self._parent_winds(slice(0, k))
        u = cp.zeros((k, ne), cp.float32)
        self._child_edge_normal(uz, um, k, u)
        return OnlineFrame(when, {"u": u, "w": w, "rho": rho, "theta": theta,
                                  "scalars": scalars})

    def force(self) -> None:
        """After the parent's step: the end-of-step planes and the frame."""
        t0 = time.perf_counter()
        self.parent_steps_done += 1
        self.snapshot(self.b)
        when = self.start_time + timedelta(
            seconds=float(self.parent_driver.config.config_dt) * self.parent_steps_done)
        self.pool.push(self.frame(when))
        self.stats["forces"] += 1
        self.stats["coupling_seconds"] += time.perf_counter() - t0

    def child_step(self, j: int) -> None:
        t0 = time.perf_counter()
        ratio = int(self.band.parent_time_step_ratio)
        self.top.step_frac0 = j / ratio
        self.top.step_span = 1.0 / ratio
        advance(self.child_driver)
        self.stats["child_steps"] += 1
        self.stats["child_seconds"] += time.perf_counter() - t0

    def roll(self) -> None:
        """The end planes become the next parent step's start planes."""
        self.a, self.b = self.b, self.a

    # -- child -> parent -----------------------------------------------------
    def feedback(self) -> None:
        if not self.band.feedback:
            return
        cp = _cp()
        t0 = time.perf_counter()
        k = self.k
        c = self.child_driver.atmosphere
        pd = self.parent_driver
        p = pd.atmosphere
        nlp, npc = int(p.state.rho.shape[0]), int(p.state.rho.shape[1])
        row_ptr, cols, overlap, targets = self.d_remap
        nrows = int(targets.shape[0])
        if nrows == 0:
            return
        theta_new = cp.array(p.saved.theta_m, copy=True)
        self.parent_kernels.launch("pn_remap_rows_v841", k * nrows, (
            I32(k), I32(nrows), I32(self.nc), I32(npc), row_ptr, cols, overlap, targets,
            c.state.rho, c.saved.theta_m, theta_new))
        delta = cp.zeros((nlp, npc), cp.float32)
        self.parent_kernels.launch("pn_feedback_delta_v841", k * nrows, (
            I32(k), I32(nrows), I32(npc), targets, theta_new, p.state.rho,
            p.saved.rho_theta_perturbation, p.reference.rho_theta_base, delta))
        rho = cp.empty_like(p.state.rho)
        rtheta = cp.empty_like(p.state.rho)
        rho_p = cp.empty_like(p.state.rho)
        rtheta_p = cp.empty_like(p.state.rho)
        theta = cp.empty_like(p.state.rho)
        exner = cp.empty_like(p.saved.exner)
        pressure_p = cp.empty_like(p.saved.pressure_perturbation)
        zero = cp.zeros_like(p.state.rho)
        pd._launch("recover_cells_f32", npc * nlp, (
            I32(nlp), I32(npc), I32(3), I32(pd.source_release == "v8.4.1"),
            F32(287.0), F32(1004.5), F32(100_000.0),
            p.saved.density_perturbation, p.saved.rho_theta_perturbation, zero, delta,
            I32(0), F32(0.0), p.saved.density_perturbation,
            p.reference.rho_base, p.reference.rho_theta_base, p.reference.exner_base,
            p.vertical.zz, p.saved.exner, p.saved.pressure_perturbation,
            rho, rtheta, rho_p, rtheta_p, theta, exner, pressure_p))
        p.state.rho_theta[...] = rtheta
        p.saved.theta_m[...] = theta
        p.saved.rho_theta_perturbation[...] = rtheta_p
        p.saved.exner[...] = exner
        p.saved.pressure_perturbation[...] = pressure_p
        for s in range(self.ns):
            q = cp.array(p.state.scalars[s], copy=True)
            self.parent_kernels.launch("pn_remap_rows_v841", k * nrows, (
                I32(k), I32(nrows), I32(self.nc), I32(npc), row_ptr, cols, overlap, targets,
                c.state.rho, cp.ascontiguousarray(c.state.scalars[s]), q))
            p.state.scalars[s] = q
        self.stats["feedbacks"] += 1
        self.stats["coupling_seconds"] += time.perf_counter() - t0


def advance(driver: Any) -> None:
    """One step of a hex CUDA driver, published: ``step_device`` returns the
    new atmosphere rather than filling the resident one."""
    driver.atmosphere = driver.step_device().atmosphere


def run(nests, parent_steps: int, *, on_child_step=None, on_parent_step=None) -> dict:
    """Drive a root and its chain (coarse to fine; band n's parent is band
    n-1's child): each parent step first, FORCE, the child ``ratio`` times
    (recursively for a chain), feedback last."""
    nests = (nests,) if isinstance(nests, PartialNestHex) else tuple(nests)
    for n in range(1, len(nests)):
        if nests[n].parent_driver is not nests[n - 1].child_driver:
            raise PartialNestRefused("a chain's band n must be band n-1's child")

    def cycle(level: int) -> None:
        nest = nests[level]
        nest.force()
        for j in range(int(nest.band.parent_time_step_ratio)):
            nest.child_step(j)
            if on_child_step is not None:
                on_child_step(nest, j)
            if level + 1 < len(nests):
                cycle(level + 1)
        nest.feedback()
        nest.roll()

    root = nests[0]
    for _ in range(int(parent_steps)):
        t0 = time.perf_counter()
        advance(root.parent_driver)
        root.stats["parent_seconds"] += time.perf_counter() - t0
        if on_parent_step is not None:
            on_parent_step(root)
        cycle(0)
    return {"parent_steps": int(parent_steps),
            "child_steps": [n.stats["child_steps"] for n in nests]}


__all__ = ["ChildBuild", "OnlineDrivingState", "OnlinePool", "PartialChildRuntime",
           "PartialNestHex", "PartialTopDriver", "PartialTopKernelCache", "run"]
