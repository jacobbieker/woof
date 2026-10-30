"""Partial-depth nest, Phase D3: a running two-way child below an interface.

plans/sorted-mapping-hummingbird.md, Phase D.  The child is its parent's
discrete column cut at a pure-pressure full level ``k_top`` (D1,
:mod:`woof.core.partial_nest`) with a mass flux through its top (D2,
:mod:`woof.core.partial_nest_dyn`, ``kernels/nest_top.cu``).  This module
runs it inside its parent; the new arithmetic is ``kernels/partial_nest.cu``.

Parent to child, every child small step
---------------------------------------
At the start and at the end of each parent step the parent's interface
planes -- the eta mass flux Omega at ``k_top`` and ``k_top+1``, w and the
total geopotential at ``k_top``, phi at ``k_top+1``, p and theta at the half
level above, u and v there -- are SINT-interpolated onto the child grid
(WRF ``sint`` through :func:`woof.core.nest_interp.sint`, with the
bdy-wrapper registrations the lateral tables use).  Each child acoustic
substep then takes, by linear interpolation in time across the parent step
(``nest_top_substep_forcing``, ``nest_top_time_interp``'s words) and after
subtracting the child's own RK-stage reference, ``ww_pp``, ``w_top``,
``ph_top``, ``ww_above``, ``p_above`` and ``p_above_old``; ``ww_rf``,
``t_above`` and ``pht_above`` are set once per RK stage
(``nest_top_stage_forcing``).  The child's big step calls ``calc_ww_top`` and
``add_top_face_flux`` (upwinded) for theta, u and v.

The interface's phi'' predictor is ``nest_top_specified_predictor``: with
both w'' and phi'' specified there, the rhs the implicit w solve sees at
the interface is the one that closes on the specified phi''.  D2's
continuation predictor rebuilds it from the child's own big-step phi
tendency at its top level, which the existing rhs_ph kernel forms as a
lid's (one-sided, no vertical advection through the interface).

Mass budget (``mass_correction``)
---------------------------------
SINT is not conservative: the child's interpolated interface flux, summed
over its cells, is not the parent's flux through the footprint.  The
uniform residual (``nest_top_budget_residual``, one fixed-order binary64
sum) closes the budget.  ``mass_route = "face"`` (the default) adds it to
the interface flux itself, so the face flux theta, u and v see and the
column mass agree; ``mass_route = "adj"`` hands it to D2's ``dmdt_adj``
hook instead (column mass only).  MEASURED (warm bubble, receipt variant
``bubble_mass_route_adj``): through ``dmdt_adj`` the child's theta' error
against all-fine roughly triples, because the D2 kernels keep the face flux
at ``ww_top`` while the column sum carries ``adj``, so the top layer's
theta flux no longer telescopes against its mass; a per-parent-cell
residual through ``dmdt_adj`` made the rest state unstable outright.

Top relaxation zone
-------------------
WRF's ``spec_bdytend`` + ``relax_bdytend_core`` on a TOP STRIP
(``nest_top_relaxation``: the lateral routine's south side turned on its
side, every operation rounded once in WRF's order; held to the compiled
routine by ``tools/partial_nest_wrf471_oracle``).  Row 0, the specified
row, is the INTERFACE: for w and phi the child's own top level, which the
acoustic top entry points specify; for u, v and theta the parent's half
level above it, which enters only as the first relaxation row's target.
``relax_levels`` relaxation rows lie below it.  The value/tendency tables
are built at FORCE in ``bdy_interp1``'s own arithmetic
(``nest_top_strip_tables``).  As in WRF's relax_bdy_dry the increment is
formed once per step from the time-t state (stage 1) and held for all three
stages (``nest_top_apply_held``).  u, v, theta, w and phi are relaxed; mu
is not.  (A first design made the child's top HALF level the specified row:
the acoustic pressure gradient still acts there, so the row was half
specified, and the u/v rows went unstable within a minute.)

Lateral frame and corners
-------------------------
The existing nest machinery, unchanged: ``couple_nest_field`` on both
grids, ``bdy_interp1`` and ``attach_nest_boundaries`` over the child's
levels (the parent's coupled fields sliced at the interface), the acoustic
frame kernel ``advance_nested_phi_w`` and ``apply_state_lateral_boundaries``.
The constant column-mass offset ``p_i - p_top`` lives in the child's base
state (``mub2d``): ``(c1*mu' + c2')*q`` is the parent's ``(c1*mu + c2)*q`` on
every shared level and ``mu'`` the parent's perturbation, so the coupled
fields carry over word for word.  Corners: the lateral frame owns every
column within its relaxation width at every level; the top zone takes the
rest.  The two sets of tables are SINT of one coupled parent field through
one registration in one arithmetic, so where they overlap their words are
the same (tests/test_partial_nest_run.py).

Feedback below the interface
----------------------------
WRF's ``copy_fcn`` and ``sm121``/``smdsm`` smoother exactly as the full nest
uses them (:mod:`woof.core.nest_interp`) on the parent's levels below the
interface, for mu, u, v and theta.  The column mass: the child's ``mu'`` is
the parent's perturbation, so the restricted perturbation IS the
restricted total mass with the constant offset added back, ``mu_parent =
R(mu_child) + (p_i - p_top)``, up to the two base states' own difference,
measured at construction and carried (zero on flat ground).  The
interface's w and phi are not fed back, and -- MEASURED on the rest case --
neither are w and phi below it: restricting the child's geopotential under
the parent's own interface geopotential leaves a thickness jump in the
parent's top child layer, and with w and phi fed back (at every level, or
below the top zone only) the coupled rest state grew without bound after
~1 h; with mu, u, v and theta it stays at its noise floor for the 2 h
tested.  The parent's acoustic step re-derives its w and phi from the
restricted mass, theta and winds.

Scheduling
----------
:func:`run` drives one parent and one partial child through
:func:`woof.core.clock.execute_schedule` with the child's
``parent_time_step_ratio``: the parent steps first, the child is forced and
steps ``ratio`` times, feedback last.

Scope (refused by name in :func:`woof.core.partial_nest.validate_partial_nest`
and :func:`refuse_child`): one band; the default arithmetic build; dry
dynamics without physics or moist scalars (radiation's extended column and
scalars are D4); no km_opt mixing on the child (the sixth-order horizontal
filter is carried); no damp_opt=3; no map factors; no emdiv; a periodic or
specified parent.
"""

from __future__ import annotations

import dataclasses
import time
from fractions import Fraction

import numpy as np

from woof.config import RunConfig
from woof.core.kernels import get_kernel
from woof.core.partial_nest import (PartialNestBand, PartialNestRefused,
                                    interface_pressure, validate_partial_nest)
from woof.core.partial_nest_dyn import (Interface, TopForcing, add_top_face_flux,
                                        calc_ww_top, prepare_top_substep)
from woof.core.state import DTYPE, DomainState

_THREADS = 256
_MODULE = "partial_nest"
_WRF_T0 = np.float32(300.0)

#: Mass-grid interface planes, stacked for one SINT launch.
_MASS_PLANES = ("om_k", "om_kp1", "w_k", "ph_k", "ph_kp1", "p_k", "th_k")
#: Relaxed in the top zone (WRF relax_bdy_dry's nested rows); mu is not.
_TOP_KINDS = ("u", "v", "t", "ph", "w")
#: The lateral frame (attach_nest_boundaries' inventory).
_FRAME_KINDS = ("u", "v", "w", "t", "ph", "mu")
#: Fed back below the interface (module docstring, "Feedback").
FEEDBACK_KINDS = ("mu", "u", "v", "t")
_STAGGER = {"u": "x", "v": "y"}
_FULL = frozenset({"w", "ph"})
_APPLICATION_NAME = {"t": "theta", "ph": "phi"}


def _xp():
    import cupy as cp
    return cp


def _blocks(n: int) -> tuple[int]:
    return ((int(n) + _THREADS - 1) // _THREADS,)


def _kernel(name: str):
    return get_kernel(_MODULE, name)


def refuse_child(cfg: RunConfig, state: DomainState) -> None:
    """What the running child does not carry (beyond the band's D0 rules)."""
    from woof.config import constant_k_mixing_active, wrf_mixing_package_active
    from woof.core.physics import physics_enabled
    if not cfg.nested:
        raise PartialNestRefused("the partial child must be configured nested=True")
    if state.qv is not None or getattr(cfg, "moist", False):
        raise PartialNestRefused(
            "moist scalars through the interface are not carried in D3 (the "
            "child is dry dynamics; physics and the extended radiation column are D4)")
    if physics_enabled(cfg) or cfg.mp_physics:
        raise PartialNestRefused("physics on the partial child is D4")
    if cfg.emdiv > 0.0:
        raise PartialNestRefused("emdiv is not carried on the partial child")
    if cfg.damp_opt == 3:
        raise PartialNestRefused("damp_opt=3 on the partial child")
    if state.has_msf:
        raise PartialNestRefused("map factors other than 1 on the partial child")
    if wrf_mixing_package_active(cfg) or constant_k_mixing_active(cfg):
        raise PartialNestRefused(
            "km_opt mixing on the partial child is not carried through its top "
            "(its vertical fluxes close on a lid); the horizontal sixth-order "
            "filter (diff_6th_opt) is")


@dataclasses.dataclass
class _Planes:
    """One time level of the parent's interface planes on the child grid."""

    mass: object        # (len(_MASS_PLANES), ny, nx)
    u_k: object         # (1, ny, nx+1)
    v_k: object         # (1, ny+1, nx)
    adj: object         # (ny, nx) mass-budget residual (mass_route "adj")

    def __getattr__(self, name):
        if name in _MASS_PLANES:
            return self.mass[_MASS_PLANES.index(name)]
        raise AttributeError(name)


class PartialNest:
    """One parent and one partial-depth child, coupled (module docstring).

    ``parent``/``pcfg``/``coord``: the parent's state, config and finalized
    float64 vertical coordinate; ``child``/``ccfg``: the child's state on
    D1's truncated coordinate (``partial_nest_dyn.truncate_state`` with
    ``coord``) and its nested config; ``band``: the D0 entry."""

    def __init__(self, parent: DomainState, pcfg: RunConfig, coord,
                 child: DomainState, ccfg: RunConfig, band: PartialNestBand):
        cp = _xp()
        validate_partial_nest(band, pcfg, coord, has_msf=parent.has_msf)
        refuse_child(ccfg, child)
        k = int(band.k_top)
        if ccfg.nz != k:
            raise PartialNestRefused(f"the child has {ccfg.nz} levels, the band's "
                                     f"interface is level {k}")
        if (ccfg.ny, ccfg.nx) != (band.ny, band.nx):
            raise PartialNestRefused("child extent differs from the band's")
        r = int(band.parent_grid_ratio)
        if abs(ccfg.dx * r - pcfg.dx) > 1e-6 * pcfg.dx:
            raise PartialNestRefused("child dx times the ratio is not the parent's dx")
        self.parent, self.pcfg, self.coord = parent, pcfg, coord
        self.child, self.ccfg, self.band = child, ccfg, band
        self.k = k
        self.p_i = interface_pressure(coord, k)
        self.shift = self.p_i - float(coord.p_top)
        from woof.core.nest_interp import bdy_width, register_nest
        self.regs = {
            s: register_nest(nri=r, nrj=r, i_parent_start=band.i_parent_start,
                             j_parent_start=band.j_parent_start,
                             child_nx=band.nx, child_ny=band.ny,
                             parent_nx=pcfg.nx, parent_ny=pcfg.ny,
                             stagger=s, wrapper="bdy")
            for s in ("", "x", "y")}
        self.iface = Interface.from_parent(parent, k)
        self.c_top = DTYPE(cp.asnumpy(child.c2f)[k])
        ny, nx = ccfg.ny, ccfg.nx
        self.top = TopForcing.zeros(ny, nx)
        z = lambda *s: cp.zeros(s, dtype=DTYPE)
        mk = lambda: _Planes(z(len(_MASS_PLANES), ny, nx), z(1, ny, nx + 1),
                             z(1, ny + 1, nx), z(ny, nx))
        self.a, self.b = mk(), mk()
        self.w_star, self.ph_star = z(ny, nx), z(ny, nx)
        self.u_above, self.v_above = z(ny, nx + 1), z(ny + 1, nx)
        self.face_flux = {"": z(ny, nx), "x": z(ny, nx + 1), "y": z(ny + 1, nx)}
        width = band.strip_width
        self.strip, self.held = {}, {}
        for kind in _TOP_KINDS:
            shp = self._child_shape(kind)
            self.strip[kind] = (z(width, *shp[1:]), z(width, *shp[1:]))
            self.held[kind] = z(*shp)
        from woof.ingest.lateral_bc import _weights
        fcx, gcx = _weights(width, band.spec_levels, band.relax_zone, float(ccfg.dt),
                            0.0, wrf_real=True)
        self.fcx, self.gcx = cp.asarray(fcx), cp.asarray(gcx)
        self.frame_width = bdy_width(ccfg.spec_zone, ccfg.relax_zone, ccfg.spec_bdy_width)
        self.lateral = {}
        self._mu_base_residual = self._base_mass_residual()
        # The interface planes read the parent's p (and the child's stage
        # reference reads its own) from the diagnosed EOS: an initialized
        # state carries none until its first stage, so diagnose both now --
        # the same words the first stage's update_diagnostics computes.
        from woof.core.diagnostics import update_diagnostics
        update_diagnostics(parent, pcfg.hypsometric_opt)
        update_diagnostics(child, ccfg.hypsometric_opt)
        self.parent_clock = None
        #: Where the interface budget enters (module docstring, "Mass budget").
        self.mass_route = "face"
        self.feedback_kinds = FEEDBACK_KINDS
        self.stats = {"forces": 0, "child_steps": 0, "feedbacks": 0,
                      "parent_seconds": 0.0, "child_seconds": 0.0,
                      "coupling_seconds": 0.0, "max_abs_adj": 0.0}

    # ------------------------------------------------------------------
    # shapes and coupled fields
    # ------------------------------------------------------------------
    def _child_shape(self, kind):
        nz, ny, nx = self.ccfg.nz, self.ccfg.ny, self.ccfg.nx
        s = _STAGGER.get(kind, "")
        return (nz + (kind in _FULL), ny + (s == "y"), nx + (s == "x"))

    def _parent_shape(self, kind):
        nz, ny, nx = self.pcfg.nz, self.pcfg.ny, self.pcfg.nx
        if kind == "mu":
            return (1, ny, nx)
        s = _STAGGER.get(kind, "")
        return (nz + (kind in _FULL), ny + (s == "y"), nx + (s == "x"))

    def _coupled(self, state, kind, shape, slot):
        from woof.ingest.lateral_bc import couple_nest_field
        out = state.scratch(shape, slot)
        couple_nest_field(state, kind, out=out)
        return out

    def _base_mass_residual(self):
        """``R(mub_child) + (p_i - p_top) - mub_parent`` on the feedback
        rectangle, float64 on the host: the two grids' base states' own
        difference (zero on flat ground).  ``None`` when it is zero."""
        cp = _xp()
        from woof.core.nest_interp import feedback_parent_bounds
        reg = self.regs[""]
        cm = cp.asnumpy(self.child.mub2d).astype(np.float64)
        pm = cp.asnumpy(self.parent.mub2d).astype(np.float64)
        r = reg.nri
        i_lo, i_hi, j_lo, j_hi = feedback_parent_bounds(reg, spec_zone=self.ccfg.spec_zone)
        res = np.zeros_like(pm)
        for pj in range(j_lo, j_hi + 1):
            for pi in range(i_lo, i_hi + 1):
                c0 = (pj - (reg.j_parent_start - 1)) * r
                i0 = (pi - (reg.i_parent_start - 1)) * r
                res[pj, pi] = cm[c0:c0 + r, i0:i0 + r].mean() + self.shift - pm[pj, pi]
        tol = 2.0 * np.spacing(np.float32(np.abs(pm).max()))
        return None if np.abs(res).max() <= tol else cp.asarray(res, dtype=DTYPE)

    # ------------------------------------------------------------------
    # parent -> child: interface planes, lateral and top tables
    # ------------------------------------------------------------------
    def snapshot(self, dst: _Planes) -> None:
        """The parent's interface planes now, on the child grid."""
        cp = _xp()
        from woof.core.dycore import stage_fluxes
        from woof.core.nest_interp import sint
        p, k = self.parent, self.k
        _, _, ww = stage_fluxes(p, self.pcfg)
        phb, thb = p.phb, p.thb          # profiles (flat) or 3-D: [k] is either
        stack = p.scratch((len(_MASS_PLANES), self.pcfg.ny, self.pcfg.nx),
                          "partial_nest_planes")
        stack[0] = ww[k]
        stack[1] = ww[k + 1]
        stack[2] = p.w[k]
        stack[3] = phb[k] + p.php[k]
        stack[4] = phb[k + 1] + p.php[k + 1]
        stack[5] = p.p[k]
        stack[6] = thb[k] + p.thp[k]
        sint(stack, self.regs[""], out=dst.mass)
        sint(cp.ascontiguousarray(p.u[k:k + 1]), self.regs["x"], out=dst.u_k)
        sint(cp.ascontiguousarray(p.v[k:k + 1]), self.regs["y"], out=dst.v_k)
        dst.adj[...] = 0.0
        if not self.band.mass_correction:
            return
        reg = self.regs[""]
        res = p.scratch((1,), "partial_nest_budget")
        _kernel("nest_top_budget_residual")(
            (1,), (1,),
            (cp.ascontiguousarray(ww[k]), dst.mass[0], res,
             np.int32(reg.i_parent_start), np.int32(reg.j_parent_start),
             np.int32(reg.nri), np.int32(reg.nrj),
             np.int32(self.pcfg.ny), np.int32(self.pcfg.nx),
             np.int32(self.ccfg.ny), np.int32(self.ccfg.nx)))
        self.stats["max_abs_adj"] = max(self.stats["max_abs_adj"], float(cp.abs(res)[0]))
        if self.mass_route == "adj":
            dst.adj[...] = res[0]
        elif self.mass_route == "face":
            dst.mass[0] += res[0]
        else:
            raise ValueError(f"mass_route must be 'adj' or 'face', not {self.mass_route!r}")

    def force(self, child_clock, parent_clock) -> None:
        """FORCE (after the parent's step): the end-of-step interface planes,
        the lateral frame's rolling tables and the top strip's tables."""
        cp = _xp()
        from woof.core.nest_interp import bdy_interp1, sint
        from woof.ingest.lateral_bc import attach_nest_boundaries
        t0 = time.perf_counter()
        self.snapshot(self.b)
        c, ccfg, k = self.child, self.ccfg, self.k
        cdt = DTYPE(parent_clock.dt_fp32)
        width = self.band.strip_width
        fields = {}
        for kind in _FRAME_KINDS:
            reg = self.regs[_STAGGER.get(kind, "")]
            pfull = self._coupled(self.parent, kind, self._parent_shape(kind),
                                  "partial_nest_parent_" + kind)
            nlev = 1 if kind == "mu" else k + (kind in _FULL)
            plow = cp.ascontiguousarray(pfull[:nlev])
            cshape = (1, ccfg.ny, ccfg.nx) if kind == "mu" else self._child_shape(kind)
            cfull = self._coupled(c, kind, cshape, "partial_nest_child_" + kind)
            name = _APPLICATION_NAME.get(kind, kind)
            fields[name] = bdy_interp1(
                plow, cfull, reg, parent_dt_fp32=cdt,
                parent_interval_ticks=parent_clock.step_ticks,
                spec_zone=ccfg.spec_zone, relax_zone=ccfg.relax_zone,
                spec_bdy_width=ccfg.spec_bdy_width, out=self.lateral.get(kind))
            self.lateral[kind] = fields[name]
            if kind in _TOP_KINDS:
                # Row 0 is the parent's level k_top for either staggering:
                # the interface (full levels) or the half level above it.
                pstrip = cp.ascontiguousarray(pfull[k - width + 1:k + 1][::-1])
                psca = sint(pstrip, reg)
                cstrip = c.scratch((width,) + cshape[1:], "partial_nest_cstrip_" + kind)
                n = cstrip.size
                _kernel("nest_top_strip_gather")(
                    _blocks(n), (_THREADS,),
                    (cfull, psca, cstrip, np.int32(width),
                     np.int32(kind not in _FULL), np.int32(cshape[0]),
                     np.int32(cshape[1]), np.int32(cshape[2])))
                val, tend = self.strip[kind]
                _kernel("nest_top_strip_tables")(
                    _blocks(n), (_THREADS,), (psca, cstrip, val, tend, cdt, np.int32(n)))
        attach_nest_boundaries(c, fields, clock=child_clock,
                               spec_bdy_width=ccfg.spec_bdy_width,
                               spec_zone=ccfg.spec_zone, relax_zone=ccfg.relax_zone)
        self.parent_clock = parent_clock
        self.stats["forces"] += 1
        self.stats["coupling_seconds"] += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # the child's step
    # ------------------------------------------------------------------
    def _fraction(self, child_clock, offset_s: float) -> float:
        """Fraction of the parent step at child time + ``offset_s``."""
        pc = self.parent_clock
        into = Fraction(child_clock.ticks - (pc.ticks - pc.step_ticks), pc.step_ticks)
        return float(into) + float(offset_s) / float(self.pcfg.dt)

    def _top_zone(self, istage: int, dtbc) -> None:
        """Stage 1: the held relaxation increments from the time-t state;
        every stage: specified rows replaced, relaxation rows added."""
        c, band = self.child, self.band
        width = band.strip_width
        frame = self.frame_width
        tends = {"u": c.ru_t, "v": c.rv_t, "t": c.rth_t, "ph": c.rph_t, "w": c.rw_t}
        for kind in _TOP_KINDS:
            shp = self._child_shape(kind)
            held = self.held[kind]
            n = int(np.prod(shp[1:]))
            half = np.int32(kind not in _FULL)
            if istage == 0:
                held[...] = 0.0
                field = self._coupled(c, kind, shp, "partial_nest_child_" + kind)
                val, tend = self.strip[kind]
                _kernel("nest_top_relaxation")(
                    _blocks(band.relax_zone * n), (_THREADS,),
                    (field, held, val, tend, self.fcx, self.gcx, DTYPE(dtbc),
                     np.int32(width), np.int32(band.spec_levels),
                     np.int32(band.relax_zone), np.int32(frame), half,
                     np.int32(shp[0]), np.int32(shp[1]), np.int32(shp[2])))
            _kernel("nest_top_apply_held")(
                _blocks(band.relax_zone * n), (_THREADS,),
                (tends[kind], held, np.int32(band.spec_levels),
                 np.int32(band.relax_zone), np.int32(frame), half,
                 np.int32(shp[0]), np.int32(shp[1]), np.int32(shp[2])))

    def _predictor(self, dtau):
        """The interface phi'' predictor consistent with the specified w''
        and phi'' there (``nest_top_specified_predictor``)."""
        c, cfg, top = self.child, self.ccfg, self.top
        n2 = cfg.ny * cfg.nx
        kernel = _kernel("nest_top_specified_predictor")
        args = (top.w_top, top.ph_top, c.mub2d, c.mup, c.mu_pp, c.c1f, c.c2f,
                DTYPE(dtau), DTYPE(cfg.epssm), np.int32(cfg.nz), np.int32(n2))

        def launch(rhs_top):
            kernel(_blocks(n2), (_THREADS,), (rhs_top,) + args)
        return launch

    def _frame_launch(self, dtau, c2a):
        """The nested frame's acoustic update (acoustic.py's
        ``advance_nested_phi_w`` launch, the same arguments)."""
        c, cfg = self.child, self.ccfg
        nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
        kernel = get_kernel("acoustic", "advance_nested_phi_w", (nz, ny, nx))
        args = (c.ph_pp, c.w_pp, c.p_pp, c.al_pp, c.rph_t, c.rw_t, c.th_pp, c.mup,
                c.mu_pp, c.mub2d, c.rmu_t, c.php, c.thp, c.thb, c.alt, c2a,
                c.rdnw, c.c1h, c.c2h, c.c1f, c.c2f, DTYPE(dtau),
                np.int32(cfg.spec_zone), np.int32(c.thb.ndim == 3),
                np.int32(nz), np.int32(ny), np.int32(nx))
        grid = _blocks(ny * nx)

        def launch():
            kernel(grid, (_THREADS,), args)
        return launch

    def child_step(self, child_clock) -> None:
        """One child step of ``ccfg.dt``: ``dycore.step``'s default acoustic
        RK3 order for a dry nested domain, with the top entry points, the
        top zone and the parent's interface forcing."""
        from woof.core.acoustic import prepare_acoustic_coefficients, prepare_moist_cq
        from woof.core.diagnostics import update_diagnostics
        from woof.core.dycore import (_add_slow_tendencies,
                                      _prepare_small_step_finish_launch,
                                      _prepare_small_step_init_launch,
                                      _prepare_tendency_zero, _save_time_t,
                                      add_fixed_dry_tendencies, apply_w_damping,
                                      prepare_fixed_tendencies, set_w_surface,
                                      stage_fluxes)
        from woof.core.partial_nest_dyn import time_interpolate
        from woof.ingest.lateral_bc import (apply_state_boundary_values,
                                            apply_state_lateral_boundaries)
        t0 = time.perf_counter()
        c, cfg, top, iface = self.child, self.ccfg, self.top, self.iface
        n2 = cfg.ny * cfg.nx
        dtbc = child_clock.dtbc_launch_fp32
        _save_time_t(c)
        zero = _prepare_tendency_zero(c)
        prepare_fixed_tendencies(c, cfg, None)          # diff_6th_opt, held
        ns = cfg.time_step_sound
        dt = float(cfg.dt)
        # (substeps, dtau, the stage reference's offset t* - t)
        stages = ((1, dt / 3.0, 0.0), (max(ns // 2, 1), dt / ns, dt / 3.0),
                  (ns, dt / ns, dt / 2.0))
        init = _prepare_small_step_init_launch(c, cfg)
        finish = _prepare_small_step_finish_launch(c, cfg)
        base3d = np.int32(c.thb.ndim == 3)
        a, b = self.a, self.b
        theta_total = c.scratch(c.thp.shape, "partial_nest_theta")
        thb = c.thb if c.thb.ndim == 3 else c.thb[:, None, None]
        for istage, (nsub, dtau, star) in enumerate(stages):
            zero()
            update_diagnostics(c, cfg.hypsometric_opt)
            fstar = self._fraction(child_clock, star)
            _kernel("nest_top_stage_forcing")(
                _blocks(n2), (_THREADS,),
                (a.om_k, b.om_k, a.adj, b.adj, a.th_k, b.th_k, a.ph_kp1, b.ph_kp1,
                 c.w, c.php, c.phb, base3d, np.int32(cfg.nz), DTYPE(fstar),
                 top.ww_rf, top.adj_rf, top.t_above, top.pht_above,
                 self.w_star, self.ph_star, np.int32(n2)))
            ru, rv, _ = stage_fluxes(c, cfg, first_stage=(istage == 0))
            ww = calc_ww_top(c, cfg, ru, rv, top.ww_rf, top.adj_rf, nested_frame=True)
            cq = prepare_moist_cq(c, cfg)
            _add_slow_tendencies(c, cfg, ru, rv, ww, cq=cq)
            add_fixed_dry_tendencies(c, cfg)
            time_interpolate(a.u_k[0], b.u_k[0], fstar, out=self.u_above)
            time_interpolate(a.v_k[0], b.v_k[0], fstar, out=self.v_above)
            theta_total[...] = thb + c.thp
            add_top_face_flux(c.rth_t, theta_total, top.t_above, top.ww_rf, c, cfg,
                              iface, "", nested_frame=True, flux=self.face_flux[""])
            add_top_face_flux(c.ru_t, c.u, self.u_above, top.ww_rf, c, cfg, iface,
                              "x", nested_frame=True, flux=self.face_flux["x"])
            add_top_face_flux(c.rv_t, c.v, self.v_above, top.ww_rf, c, cfg, iface,
                              "y", nested_frame=True, flux=self.face_flux["y"])
            self._top_zone(istage, dtbc)
            apply_w_damping(c, cfg, ww)
            apply_state_lateral_boundaries(c, cfg, rk_stage=istage)
            init()
            coef = prepare_acoustic_coefficients(c, cfg, dtau, cq=cq)
            launch = prepare_top_substep(c, cfg, dtau, coef, iface, top,
                                         theta_upwind=True, nested_frame=True,
                                         predictor=self._predictor(dtau))
            frame = self._frame_launch(dtau, coef[0])
            for n in range(nsub):
                # Every stage integrates its substeps from time t.
                f_new = self._fraction(child_clock, (n + 1) * dtau)
                f_old = self._fraction(child_clock, n * dtau)
                _kernel("nest_top_substep_forcing")(
                    _blocks(n2), (_THREADS,),
                    (a.om_k, b.om_k, a.w_k, b.w_k, a.ph_k, b.ph_k,
                     a.om_kp1, b.om_kp1, a.p_k, b.p_k, a.adj, b.adj,
                     top.ww_rf, top.adj_rf, self.w_star, self.ph_star,
                     self.c_top, DTYPE(f_new), DTYPE(f_old), DTYPE(fstar),
                     top.ww_pp, top.w_top, top.ph_top, top.ww_above,
                     top.p_above, top.p_above_old, top.adj_pp, np.int32(n2)))
                launch(first=(n == 0))
                frame()
            finish()
        apply_state_boundary_values(c, cfg, c.elapsed_seconds + cfg.dt)
        set_w_surface(c, cfg)
        update_diagnostics(c, cfg.hypsometric_opt)
        c.elapsed_seconds += cfg.dt
        self.stats["child_steps"] += 1
        self.stats["child_seconds"] += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # child -> parent
    # ------------------------------------------------------------------
    def feedback(self) -> None:
        """WRF copy_fcn (+ smoother) below the interface for
        :data:`FEEDBACK_KINDS` (module docstring, "Feedback")."""
        cp = _xp()
        from woof.core.diagnostics import update_diagnostics
        from woof.core.nest_interp import (copy_fcn, feedback_parent_bounds,
                                            smoother, smoother_parent_window)
        if self.band.feedback == 0:
            return
        t0 = time.perf_counter()
        p, c, k, spec = self.parent, self.child, self.k, self.ccfg.spec_zone
        sm = int(self.band.smooth_option)
        reg_m = self.regs[""]
        i_lo, i_hi, j_lo, j_hi = feedback_parent_bounds(reg_m, spec_zone=spec)
        if "mu" in self.feedback_kinds:
            # The perturbation, with the base states' residual (module doc).
            mu = p.mup[None].copy()
            copy_fcn(mu, c.mup[None], reg_m, spec_zone=spec)
            if self._mu_base_residual is not None:
                box = (slice(j_lo, j_hi + 1), slice(i_lo, i_hi + 1))
                mu[(0,) + box] += self._mu_base_residual[box]
            if sm:
                smoother(mu, reg_m, smooth_option=sm,
                         scratch=p.scratch(mu.shape, "partial_nest_smooth"))
            p.mup[...] = mu[0]
        for kind in self.feedback_kinds:
            if kind == "mu":
                continue
            # [:k]: below the interface (a w/phi row k_top is never written).
            reg =self.regs[_STAGGER.get(kind, "")]
            name = {"t": "thp", "ph": "php"}.get(kind, kind)
            target = getattr(p, name)
            lower = cp.ascontiguousarray(target[:k])
            if kind == "t":
                # WRF restricts t_2 = theta - 300 (nest.py _raw_child_field).
                src = c.scratch(c.thp.shape, "partial_nest_theta")
                src[...] = c.thp + (c.thb if c.thb.ndim == 3 else c.thb[:, None, None])
                src -= _WRF_T0
            else:
                src = cp.ascontiguousarray(getattr(c, name)[:k])
            copy_fcn(lower, src, reg, spec_zone=spec)
            if kind == "t":
                # Re-expressed against the parent's base theta over exactly
                # the cells copy_fcn wrote (nest.py _rebase_feedback_theta).
                ri_lo, ri_hi, rj_lo, rj_hi = feedback_parent_bounds(reg, spec_zone=spec)
                win = lower[:, rj_lo:rj_hi + 1, ri_lo:ri_hi + 1]
                win += _WRF_T0
                pthb = p.thb
                win -= (pthb[:k, rj_lo:rj_hi + 1, ri_lo:ri_hi + 1] if pthb.ndim == 3
                        else pthb[:k, None, None])
                lower[:, rj_lo:rj_hi + 1, ri_lo:ri_hi + 1] = win
            if sm:
                smoother(lower, reg, smooth_option=sm,
                         scratch=p.scratch(lower.shape, "partial_nest_smooth_" + kind))
            target[:k] = lower
        ci_lo, ci_hi, cj_lo, cj_hi = i_lo, i_hi, j_lo, j_hi
        if sm:
            i0, j0, niw, njw = smoother_parent_window(reg_m)
            if niw > 0 and njw > 0:
                ci_lo, ci_hi = min(ci_lo, i0), max(ci_hi, i0 + niw - 1)
                cj_lo, cj_hi = min(cj_lo, j0), max(cj_hi, j0 + njw - 1)
        update_diagnostics(p, self.pcfg.hypsometric_opt,
                           window=(cj_lo, ci_lo, cj_hi - cj_lo + 1, ci_hi - ci_lo + 1))
        self.stats["feedbacks"] += 1
        self.stats["coupling_seconds"] += time.perf_counter() - t0


# ---------------------------------------------------------------------------
# construction and scheduling
# ---------------------------------------------------------------------------

def child_config(pcfg: RunConfig, band: PartialNestBand, **overrides) -> RunConfig:
    """The child's full-depth nested config (before truncation at ``k_top``)."""
    r = int(band.parent_grid_ratio)
    rt = int(band.parent_time_step_ratio)
    values = dict(nx=band.nx, ny=band.ny, dx=pcfg.dx / r, dy=pcfg.dy / r,
                  dt=float(np.float32(pcfg.dt) / np.float32(rt)),
                  specified=False, nested=True, grid_id=2,
                  spec_zone=1, relax_zone=4, spec_bdy_width=5)
    values.update(overrides)
    return dataclasses.replace(pcfg, **values)


def two_domain_experiment(pcfg: RunConfig, ccfg: RunConfig, band: PartialNestBand,
                          *, run_seconds: float, time_step: int):
    """The minimal two-domain :class:`woof.experiment.ExperimentConfig`
    whose schedule the clock walks (the child's levels do not enter the
    schedule)."""
    from datetime import datetime
    from woof.experiment import DomainConfig, ExperimentConfig, VerticalConfig
    root_run = dataclasses.replace(pcfg, run_seconds=run_seconds, grid_id=1,
                                   output_interval_s=run_seconds,
                                   restart_interval_s=0.0)
    child_run = dataclasses.replace(ccfg, run_seconds=run_seconds, grid_id=2,
                                    output_interval_s=run_seconds,
                                    restart_interval_s=0.0)
    domains = (
        DomainConfig(grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
                     parent_grid_ratio=1, parent_time_step_ratio=1,
                     history_interval_s=run_seconds, run=root_run,
                     time_step=int(time_step)),
        DomainConfig(grid_id=2, parent_id=1, i_parent_start=band.i_parent_start,
                     j_parent_start=band.j_parent_start,
                     parent_grid_ratio=band.parent_grid_ratio,
                     parent_time_step_ratio=band.parent_time_step_ratio,
                     history_interval_s=run_seconds, run=child_run,
                     time_step=None))
    return ExperimentConfig(
        name="partial-nest-d3", start_time=datetime(2000, 1, 1),
        run_seconds=run_seconds,
        vertical=VerticalConfig(eta_levels=(), p_top=0.0, hybrid_opt=pcfg.hybrid_opt,
                                etac=pcfg.etac),
        projection=None, restart_interval_s=0.0, domains=domains)


def run(nest: PartialNest, *, run_seconds: float, time_step: int,
        on_parent_step=None, on_child_step=None):
    """Drive the parent and its partial child through
    :func:`woof.core.clock.execute_schedule` (the parent first, the child
    ``parent_time_step_ratio`` times per parent step, feedback last).
    Returns the executor's report."""
    from woof.core import dycore
    from woof.core.clock import build_schedule, execute_schedule
    exp = two_domain_experiment(nest.pcfg, nest.ccfg, nest.band,
                                run_seconds=run_seconds, time_step=time_step)
    schedule = build_schedule(exp)

    def on_step(gid, clock):
        if gid == 1:
            t0 = time.perf_counter()
            nest.snapshot(nest.a)
            dycore.step(nest.parent, nest.pcfg)
            nest.stats["parent_seconds"] += time.perf_counter() - t0
            if on_parent_step is not None:
                on_parent_step(nest, clock)
        else:
            nest.child_step(clock)
            if on_child_step is not None:
                on_child_step(nest, clock)

    def on_force(cid, pid, child_clock, parent_clock):
        nest.force(child_clock, parent_clock)

    def on_feedback(cid, pid, child_clock, parent_clock):
        nest.feedback()

    return execute_schedule(schedule, clocks=schedule.clock.clocks(), on_step=on_step,
                            on_force=on_force, on_feedback_commit=on_feedback)
