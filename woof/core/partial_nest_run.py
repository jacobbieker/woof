"""Partial-depth nest, Phases D3-D4: running two-way children below interfaces.

plans/sorted-mapping-hummingbird.md, Phase D.  The child is its parent's
discrete column cut at a pure-pressure full level ``k_top`` (D1,
:mod:`woof.core.partial_nest`) with a mass flux through its top (D2,
:mod:`woof.core.partial_nest_dyn`, ``kernels/nest_top.cu``).  This module
runs it inside its parent -- one band, or a chain of bands coarse to fine
(band ``n``'s parent is band ``n - 1``) -- and steps it with the dycore's
own RK loop (:func:`woof.core.dycore.step` with ``partial_top=``): the hooks
of :class:`ChildTop` run the interface at their places in that loop, so the
child carries moist scalars, physics and microphysics as any domain does.
The new arithmetic is ``kernels/partial_nest.cu`` (D3) and
``kernels/partial_nest_d4.cu`` (D4); the radiation column is
:mod:`woof.core.partial_nest_physics`.

Parent to child, every child small step
---------------------------------------
At the start and at the end of each parent step the parent's interface
planes -- the eta mass flux Omega at ``k_top`` and ``k_top+1``, w and the
total geopotential at ``k_top``, phi at ``k_top+1``, p and theta at the half
level above, u and v there, and each moist scalar there -- are
SINT-interpolated onto the child grid (WRF ``sint`` through
:func:`woof.core.nest_interp.sint`, with the bdy-wrapper registrations the
lateral tables use).  Each child acoustic substep then takes, by linear
interpolation in time across the parent step and after subtracting the
child's own RK-stage reference, ``ww_pp``, ``w_top``, ``ph_top``,
``ww_above``, ``p_above`` and ``p_above_old`` (``nest_top_substep_forcing``);
``ww_rf``, ``t_above``, ``pht_above``, u and v above and the child's top
theta are set once per RK stage (``nest_top_stage_forcing_fused``, one
launch where D3 took four), the moist scalars above in one more.  The
child's big step calls ``calc_ww_top`` and ``add_top_face_flux`` for theta,
u and v.

The top face's vertical flux (D4 change)
----------------------------------------
theta, u and v cross the interface with the parent's own second-order face
value ``fnm*q_above + fnp*q(nz-1)``, in the big step and in the acoustic
theta flux (``centered=True`` / ``theta_upwind=False`` of D2's entry
points).  D3 upwinded them.  MEASURED (gravity wave, receipts
``tests/data/receipts/partial-nest-d4``): in a stably stratified column an
upwinded face carries the colder value upward and the warmer value
downward, so the child's top layer gains heat at about ``|ww|*dtheta/2``
per unit mass whichever way the wave moves.  One-way, the top relaxation
zone removes that bias each step; two-way, the relaxation target is the
parent's restriction of the biased child, so it ratchets: +0.08 K at the top
child level after 1200 s and a block theta' RMSE of 0.023 K (parent alone
0.0075).  With the parent's face value the top level is unbiased
(``face_scheme = "upwind"`` keeps D3's for measurement).

Moist scalars through the interface (D4)
----------------------------------------
The scalars advect with the stage's acoustic time-averaged mass flux
(``sumflux``'s ``ww_m``), whose interface face the existing scalar kernels
treat as a lid.  :meth:`ChildTop.scalars_before` forms each species' face
flux ``ww_m[nz]*q_face`` (the same second-order face value, an outgoing
face capped at the donor's own value) from the stage-start scalar and the
parent's SINT-ed value above; :meth:`ChildTop.scalars_after` adds it to the
updated top level, ``dt*(-rdnw*F)/C(mu_new)`` (``nest_top_scalar_face_flux``,
``nest_top_scalar_apply``), so q = const telescopes against the mass the
acoustic loop moved through the interface.  The lateral frame carries every
moist scalar (:func:`woof.core.nest_fields.nest_field_kinds`); the top zone
relaxes the dynamics only.  NOT carried: hydrometeor sedimentation through
the interface (the child's microphysics sees a lid there).

Mass budget (``mass_correction``)
---------------------------------
SINT is not conservative: the child's interpolated interface flux, summed
over its cells, is not the parent's flux through the footprint.  The
uniform residual (``nest_top_budget_residual``, one fixed-order binary64
sum) closes the budget.  ``mass_route = "face"`` (the default) adds it to
the interface flux itself; ``"adj"`` hands it to D2's ``dmdt_adj`` hook
(column mass only; MEASURED in D3 to triple the bubble's theta' error).

Top relaxation zone
-------------------
WRF's ``spec_bdytend`` + ``relax_bdytend_core`` on a TOP STRIP
(``nest_top_relaxation``, held to the compiled routine by
``tools/partial_nest_wrf471_oracle``).  Row 0, the specified row, is the
INTERFACE: for w and phi the child's own top level, for u, v and theta the
parent's half level above it.  ``relax_levels`` relaxation rows lie below.
The increment is formed once per step from the time-t state (stage 1) and
held for all three stages.  u, v, theta, w and phi are relaxed; mu and the
scalars are not.

Lateral frame and corners
-------------------------
The existing nest machinery, unchanged: ``couple_nest_field`` on both
grids, ``bdy_interp1`` and ``attach_nest_boundaries`` over the child's
levels, the acoustic frame kernel ``advance_nested_phi_w`` and
``apply_state_lateral_boundaries``.  Corners: the lateral frame owns every
column within its relaxation width at every level; the top zone the rest.

Feedback below the interface
----------------------------
WRF's ``copy_fcn`` (and, for ``smooth_option`` 1/2, ``sm121``/``smdsm``) on
the parent's levels below the interface, for the band's
``feedback_fields`` -- theta by default, bringing every moist scalar the
child carries.  MEASURED on the gravity wave with the parent's face value
(block-mean RMSE against all-fine; parent alone: theta' 0.0075 K, u 0.0146
m/s):

* theta: theta' 0.0066, u 0.0122; warm bubble theta' 0.126 (D3 0.143);
* D3's mu, u, v and theta: theta' 0.0106;
* mu alone: u 0.10 -- the restricted column mass under the parent's own
  geopotential moves the parent's EOS pressure (alpha from phi and mu);
* u and v alone: theta' 0.011 -- the child's resolved short waves
  restricted onto a parent that cannot carry them;
* the smoother (smooth_option 1/2) makes each variant worse.

mu, u, v and w stay selectable (``feedback_fields``); restricted column mass
carries the constant offset back, ``mu_parent = R(mu_child) + (p_i -
p_top)``, up to the base states' own difference.

w and phi feedback
------------------
phi is refused by name (:func:`woof.core.partial_nest.validate_partial_nest`).
MEASURED at rest (the coupled noise floor is 1e-3 m/s): restricting the
child's geopotential onto the parent grows a mode at the interface with an
e-folding time of about 80 s (phi alone; 120-300 s with mu, theta, w
alongside); its w peaks at the child's top level and the parent's levels
k_top..k_top+1, and the column mass follows (mu' to 1e3 Pa).  It survives
the lateral exclusion widened to 2 or 3 parent cells, the top zone left out
of the restriction, the interface's thickness mismatch
``phi_parent - R(SINT(phi_parent))`` spread as a uniform stretch over the
column, and mu and theta fed back alongside; it grows faster at
parent_time_step_ratio 6 than 3.  The mechanism: below the interface the
parent's geopotential stops being its own prognostic variable.  Each parent
step it is replaced by the restriction of a child column that hangs from a
time-interpolated SINT copy of the parent's interface geopotential, while
the parent's w and everything above the interface evolve under the
parent's own implicit w-phi solve.  The parent's vertical acoustic and
hydrostatic adjustment of the layers at the interface is answered a parent
step late by a column the parent cannot move: a delayed feedback on an
oscillator, unstable for any lag of a parent step.  WRF feeds back w and ph
only in full-column nests, whose child owns the column to the model top and
has no interface.  w alone is stable (3 h at rest) and selectable.

Scheduling
----------
:func:`run` drives a root and its chain through
:func:`woof.core.clock.execute_schedule` with each band's
``parent_time_step_ratio``: a parent steps first, its child is forced and
steps ``ratio`` times, feedback last.  In a model run the same objects plug
into :func:`woof.core.model.execute_experiment` through its ``steppers``
mapping and the node coupler protocol (:class:`PartialNestCoupler`,
:func:`woof.core.model.partial_nest_steppers`).

Scope (refused by name in :func:`woof.core.partial_nest.validate_partial_nest`
and :func:`refuse_child`): the default arithmetic build; no km_opt mixing on
the child (the sixth-order horizontal filter is carried); no damp_opt=3; no
map factors; no emdiv; no IEVA; no TKE or chemistry scalars on the child; a
periodic or specified parent.
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
_MODULE_D4 = "partial_nest_d4"
_WRF_T0 = np.float32(300.0)

#: Mass-grid interface planes, stacked for one SINT launch.
_MASS_PLANES = ("om_k", "om_kp1", "w_k", "ph_k", "ph_kp1", "p_k", "th_k")
#: Relaxed in the top zone (WRF relax_bdy_dry's nested rows); mu is not.
_TOP_KINDS = ("u", "v", "t", "ph", "w")
#: The dynamics of the lateral frame (attach_nest_boundaries' inventory);
#: the moist scalars join from nest_field_kinds.
_FRAME_KINDS = ("u", "v", "w", "t", "ph", "mu")
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


def _kernel_d4(name: str):
    return get_kernel(_MODULE_D4, name)


def moist_species(cfg: RunConfig) -> tuple[str, ...]:
    """The scalars a moist child carries through its interface and frame:
    :func:`woof.core.nest_fields.nest_field_kinds` less the dynamics."""
    from woof.core.nest_fields import nest_field_kinds
    return tuple(k for k in nest_field_kinds(cfg) if k not in _FRAME_KINDS)


def refuse_child(cfg: RunConfig, state: DomainState) -> None:
    """What the running child does not carry (beyond the band's D0 rules)."""
    from woof.config import constant_k_mixing_active, wrf_mixing_package_active
    if not cfg.nested:
        raise PartialNestRefused("the partial child must be configured nested=True")
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
    if int(getattr(cfg, "zadvect_implicit", 0) or 0):
        raise PartialNestRefused("IEVA (zadvect_implicit) on the partial child")
    for name in ("tke", "qke_adv", "chem"):
        if getattr(state, name, None) is not None:
            raise PartialNestRefused(
                f"{name} on the partial child: only the moist scalars cross its "
                "interface")
    if bool(getattr(cfg, "moist", False)) != (state.qv is not None):
        raise PartialNestRefused("the child's moist flag and its state disagree")


@dataclasses.dataclass
class _Planes:
    """One time level of the parent's interface planes on the child grid."""

    mass: object        # (len(_MASS_PLANES), ny, nx)
    u_k: object         # (1, ny, nx+1)
    v_k: object         # (1, ny+1, nx)
    adj: object         # (ny, nx) mass-budget residual (mass_route "adj")
    q: object = None    # (nspecies, ny, nx) moist scalars above, or None

    def __post_init__(self):
        # Named views of the stacked mass planes, bound once.
        for n, name in enumerate(_MASS_PLANES):
            object.__setattr__(self, name, self.mass[n])


class PartialNest:
    """One parent and one partial-depth child, coupled (module docstring).

    ``parent``/``pcfg``/``coord``: the parent's state, config and finalized
    float64 vertical coordinate (for a band whose parent is itself a partial
    child, that child's truncated coordinate,
    :func:`woof.core.partial_nest.truncated_vertical_coord`);
    ``child``/``ccfg``: the child's state on D1's truncated coordinate
    (``partial_nest_dyn.truncate_state``) and its nested config; ``band``:
    the band entry; ``parent_nest``: the :class:`PartialNest` whose child is
    this one's parent (a chain), else ``None``."""

    def __init__(self, parent: DomainState, pcfg: RunConfig, coord,
                 child: DomainState, ccfg: RunConfig, band: PartialNestBand, *,
                 parent_nest: "PartialNest | None" = None):
        cp = _xp()
        validate_partial_nest(band, pcfg, coord, has_msf=parent.has_msf)
        refuse_child(ccfg, child)
        k = int(band.k_top)
        if k >= pcfg.nz:
            raise PartialNestRefused(
                f"the interface (level {k}) must be an interior level of the "
                f"parent's {pcfg.nz}-level column")
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
        self.parent_nest = parent_nest
        self.k = k
        self.p_i = interface_pressure(coord, k)
        self.shift = self.p_i - float(coord.p_top)
        self.species = moist_species(ccfg) if child.qv is not None else ()
        missing = [q for q in self.species if getattr(parent, q, None) is None]
        if missing:
            raise PartialNestRefused(f"a moist child needs its parent to carry {missing}")
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
        nq = len(self.species)
        mk = lambda: _Planes(z(len(_MASS_PLANES), ny, nx), z(1, ny, nx + 1),
                             z(1, ny + 1, nx), z(ny, nx),
                             z(nq, ny, nx) if nq else None)
        self.a, self.b = mk(), mk()
        self.w_star, self.ph_star = z(ny, nx), z(ny, nx)
        self.u_above, self.v_above = z(ny, nx + 1), z(ny + 1, nx)
        self.q_above = z(nq, ny, nx) if nq else None
        self.q_flux = z(nq, ny, nx) if nq else None
        # The face flux reads the top half level only; the fused stage
        # kernel writes that level (nest_top_stage_forcing_fused).
        self.theta_top = z(ccfg.nz, ny, nx)
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
        self.frame_kinds = _FRAME_KINDS + self.species
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
        self.frac_now = None
        #: Where the interface budget enters (module docstring, "Mass budget").
        self.mass_route = "face"
        #: The interface face's vertical flux: "centered" (the parent's own
        #: second-order face value, D4) or "upwind" (D3's; kept to measure).
        self.face_scheme = "centered"
        self.feedback_kinds = tuple(band.feedback_fields)
        self._max_adj = z(1)
        #: The radiation call's extended column (partial_nest_physics).
        self.extended = None
        self.stats = {"forces": 0, "child_steps": 0, "feedbacks": 0,
                      "parent_seconds": 0.0, "child_seconds": 0.0,
                      "coupling_seconds": 0.0}

    @property
    def max_abs_mass_residual(self) -> float:
        """The largest |interface budget residual| seen (one device read)."""
        return float(self._max_adj[0])

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
    def snapshot(self, dst: _Planes | None = None) -> None:
        """The parent's interface planes now, on the child grid (``dst``
        defaults to the start-of-parent-step planes ``a``)."""
        cp = _xp()
        from woof.core.dycore import stage_fluxes
        from woof.core.nest_interp import sint
        dst = self.a if dst is None else dst
        p, k = self.parent, self.k
        ru, rv, ww = stage_fluxes(p, self.pcfg)
        if self.parent_nest is not None:
            # A partial parent's Omega closes on ITS interface flux (now),
            # not on a lid: calc_ww_top with the parent band's own forcing
            # at its current fraction of its parent's step.
            pn = self.parent_nest
            from woof.core.partial_nest_dyn import time_interpolate
            frac = 0.0 if pn.frac_now is None else pn.frac_now
            top_now = p.scratch((2, self.pcfg.ny, self.pcfg.nx), "partial_nest_top_now")
            time_interpolate(pn.a.om_k, pn.b.om_k, frac, out=top_now[0])
            time_interpolate(pn.a.adj, pn.b.adj, frac, out=top_now[1])
            ww = calc_ww_top(p, self.pcfg, ru, rv, top_now[0], top_now[1],
                             nested_frame=True)
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
        if self.species:
            qs = p.scratch((len(self.species), self.pcfg.ny, self.pcfg.nx),
                           "partial_nest_q_planes")
            for n, name in enumerate(self.species):
                qs[n] = getattr(p, name)[k]
            sint(qs, self.regs[""], out=dst.q)
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
        # A device-side running maximum: no host read per snapshot.
        cp.maximum(self._max_adj, cp.abs(res), out=self._max_adj)
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
        for kind in self.frame_kinds:
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
        self.frac_now = 0.0          # the child stands at the parent step's start
        self.stats["forces"] += 1
        self.stats["coupling_seconds"] += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # the child's step
    # ------------------------------------------------------------------
    def _into(self, child_clock) -> float:
        """Fraction of the parent step at the child's time."""
        pc = self.parent_clock
        return float(Fraction(child_clock.ticks - (pc.ticks - pc.step_ticks),
                              pc.step_ticks))

    def _top_zone(self, istage: int, dtbc) -> None:
        """Stage 1: the held relaxation increments from the time-t state;
        every stage: specified rows replaced, relaxation rows added."""
        c, band = self.child, self.band
        width = band.strip_width
        frame = self.frame_width
        if istage == 0:
            for kind in _TOP_KINDS:
                shp = self._child_shape(kind)
                held = self.held[kind]
                n = int(np.prod(shp[1:]))
                held[...] = 0.0
                field = self._coupled(c, kind, shp, "partial_nest_child_" + kind)
                val, tend = self.strip[kind]
                _kernel("nest_top_relaxation")(
                    _blocks(band.relax_zone * n), (_THREADS,),
                    (field, held, val, tend, self.fcx, self.gcx, DTYPE(dtbc),
                     np.int32(width), np.int32(band.spec_levels),
                     np.int32(band.relax_zone), np.int32(frame),
                     np.int32(kind not in _FULL),
                     np.int32(shp[0]), np.int32(shp[1]), np.int32(shp[2])))
        # nest_top_apply_held for the five fields, one launch.
        cfg, h = self.ccfg, self.held
        ny, nx = cfg.ny, cfg.nx
        _kernel_d4("nest_top_apply_held_all")(
            _blocks(band.relax_zone * max(ny * (nx + 1), (ny + 1) * nx)), (_THREADS,),
            (c.ru_t, h["u"], c.rv_t, h["v"], c.rth_t, h["t"], c.rph_t, h["ph"],
             c.rw_t, h["w"], np.int32(band.spec_levels), np.int32(band.relax_zone),
             np.int32(frame), np.int32(cfg.nz), np.int32(ny), np.int32(nx)))

    def _predictor(self, dtau):
        """The interface phi'' predictor consistent with the specified w''
        and phi'' there (``nest_top_specified_predictor``)."""
        c, cfg, top = self.child, self.ccfg, self.top
        n2 = cfg.ny * cfg.nx
        kernel = _kernel("nest_top_specified_predictor")
        args = (top.w_top, top.ph_top, c.mub2d, c.mup, c.mu_pp, c.c1f, c.c2f,
                DTYPE(dtau), DTYPE(cfg.epssm), np.int32(cfg.nz), np.int32(n2))
        grid = _blocks(n2)

        def launch(rhs_top):
            kernel(grid, (_THREADS,), (rhs_top,) + args)
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

    def child_step(self, child_clock, **step_kwargs) -> None:
        """One child step of ``ccfg.dt``: :func:`woof.core.dycore.step` with
        this child's interface hooks (:class:`ChildTop`)."""
        from woof.core import dycore
        t0 = time.perf_counter()
        top = ChildTop(self, child_clock)
        self.frac_now = top.into
        dycore.step(self.child, self.ccfg, partial_top=top, **step_kwargs)
        self.frac_now = top.into + float(self.ccfg.dt) / float(self.pcfg.dt)
        self.stats["child_steps"] += 1
        self.stats["child_seconds"] += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # child -> parent
    # ------------------------------------------------------------------
    def feedback(self) -> None:
        """WRF copy_fcn (+ smoother) below the interface for the band's
        ``feedback_fields``, theta bringing the moist scalars (module
        docstring, "Feedback")."""
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
        kinds = tuple(self.feedback_kinds)
        if "mu" in kinds:
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
        if "t" in kinds:
            kinds = kinds + self.species
        for kind in kinds:
            if kind == "mu":
                continue
            # [:k]: below the interface (a w row k_top is never written).
            reg = self.regs[_STAGGER.get(kind, "")]
            name = {"t": "thp"}.get(kind, kind)
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


class ChildTop:
    """The child's interface as hooks of :func:`woof.core.dycore.step`
    (``partial_top=``; one object per child step)."""

    #: Stage reference offsets t* - t by RK stage, in units of the child dt.
    _STAR = (0.0, 1.0 / 3.0, 0.5)

    def __init__(self, nest: PartialNest, child_clock):
        if nest.parent_clock is None:
            raise PartialNestRefused("the partial child steps only after a FORCE")
        self.nest = nest
        self.into = nest._into(child_clock)
        self.pdt = float(nest.pcfg.dt)
        self.dt = float(nest.ccfg.dt)
        self.dtbc = child_clock.dtbc_launch_fp32
        self.fstar = 0.0

    def _fraction(self, offset_s: float) -> float:
        return self.into + float(offset_s) / self.pdt

    def admit(self, state, cfg, *, acoustic: bool) -> None:
        from woof import wrf_exact
        if state is not self.nest.child:
            raise PartialNestRefused("partial_top was handed another domain's state")
        if wrf_exact.ENABLED:
            raise PartialNestRefused("the partial child steps in the default build only")
        if not acoustic:
            raise PartialNestRefused("the partial child steps on the acoustic path")

    def stage_begin(self, istage: int) -> None:
        n = self.nest
        c, cfg, top, a, b = n.child, n.ccfg, n.top, n.a, n.b
        self.fstar = self._fraction(self._STAR[istage] * self.dt)
        ny, nx = cfg.ny, cfg.nx
        count = max(ny * (nx + 1), (ny + 1) * nx)
        _kernel_d4("nest_top_stage_forcing_fused")(
            _blocks(count), (_THREADS,),
            (a.om_k, b.om_k, a.adj, b.adj, a.th_k, b.th_k, a.ph_kp1, b.ph_kp1,
             a.u_k, b.u_k, a.v_k, b.v_k, c.w, c.php, c.phb, c.thp, c.thb,
             np.int32(c.thb.ndim == 3), np.int32(cfg.nz), np.int32(ny), np.int32(nx),
             DTYPE(self.fstar),
             top.ww_rf, top.adj_rf, top.t_above, top.pht_above,
             n.w_star, n.ph_star, n.u_above, n.v_above, n.theta_top))
        if n.species:
            from woof.core.partial_nest_dyn import time_interpolate
            time_interpolate(a.q, b.q, self.fstar, out=n.q_above)

    def stage_ww(self, ru, rv, istage: int):
        n = self.nest
        return calc_ww_top(n.child, n.ccfg, ru, rv, n.top.ww_rf, n.top.adj_rf,
                           nested_frame=True)

    def slow_tendencies(self, istage: int) -> None:
        n = self.nest
        c, cfg, top, iface = n.child, n.ccfg, n.top, n.iface
        cen = n.face_scheme != "upwind"
        # D2's add_top_face_flux for theta, u and v in one launch
        # (nest_top_face_flux_all; the child is nested: clamped edges).
        ny, nx = cfg.ny, cfg.nx
        _kernel_d4("nest_top_face_flux_all")(
            _blocks(max(ny * (nx + 1), (ny + 1) * nx)), (_THREADS,),
            (c.rth_t, n.theta_top, top.t_above, n.face_flux[""],
             c.ru_t, c.u, n.u_above, n.face_flux["x"],
             c.rv_t, c.v, n.v_above, n.face_flux["y"],
             top.ww_rf, c.rdnw, iface.fnm, iface.fnp, np.int32(cen),
             np.int32(1), np.int32(1), np.int32(cfg.nz), np.int32(ny), np.int32(nx)))
        n._top_zone(istage, self.dtbc)

    def acoustic_launch(self, dtau: float, coefficients, istage: int):
        n = self.nest
        c, cfg, top = n.child, n.ccfg, n.top
        n2 = cfg.ny * cfg.nx
        top_launch = prepare_top_substep(
            c, cfg, dtau, coefficients, n.iface, top,
            theta_upwind=(n.face_scheme == "upwind"), nested_frame=True,
            predictor=n._predictor(dtau))
        frame = n._frame_launch(dtau, coefficients[0])
        forcing = _kernel("nest_top_substep_forcing")
        a, b = n.a, n.b
        head = (a.om_k, b.om_k, a.w_k, b.w_k, a.ph_k, b.ph_k,
                a.om_kp1, b.om_kp1, a.p_k, b.p_k, a.adj, b.adj,
                top.ww_rf, top.adj_rf, n.w_star, n.ph_star, n.c_top)
        tail = (top.ww_pp, top.w_top, top.ph_top, top.ww_above,
                top.p_above, top.p_above_old, top.adj_pp, np.int32(n2))
        fstar = DTYPE(self.fstar)
        grid = _blocks(n2)
        counter = [0]

        def launch(*, first: bool) -> None:
            if first:
                counter[0] = 0
            sub = counter[0]
            # Every stage integrates its substeps from time t.
            f_new = DTYPE(self._fraction((sub + 1) * dtau))
            f_old = DTYPE(self._fraction(sub * dtau))
            forcing(grid, (_THREADS,), head + (f_new, f_old, fstar) + tail)
            top_launch(first=first)
            frame()
            counter[0] = sub + 1

        return launch

    def scalars_before(self, istage: int, ww_m) -> None:
        n = self.nest
        if not n.species:
            return
        c, cfg = n.child, n.ccfg
        n2 = cfg.ny * cfg.nx
        kernel = _kernel_d4("nest_top_scalar_face_flux")
        for i, name in enumerate(n.species):
            kernel(_blocks(n2), (_THREADS,),
                   (getattr(c, name), n.q_above[i], ww_m, n.iface.fnm, n.iface.fnp,
                    n.q_flux[i], np.int32(cfg.nz), np.int32(n2)))

    def scalars_after(self, istage: int, dt_stage: float) -> None:
        n = self.nest
        if not n.species:
            return
        c, cfg = n.child, n.ccfg
        n2 = cfg.ny * cfg.nx
        kernel = _kernel_d4("nest_top_scalar_apply")
        for i, name in enumerate(n.species):
            kernel(_blocks(n2), (_THREADS,),
                   (getattr(c, name), n.q_flux[i], c.mub2d, c.mup, c.c1h, c.c2h,
                    c.rdnw, DTYPE(dt_stage), np.int32(cfg.nz), np.int32(n2)))


# ---------------------------------------------------------------------------
# construction and scheduling
# ---------------------------------------------------------------------------

def child_config(pcfg: RunConfig, band: PartialNestBand, **overrides) -> RunConfig:
    """The child's full-depth nested config (before truncation at ``k_top``)."""
    r = int(band.parent_grid_ratio)
    rt = int(band.parent_time_step_ratio)
    values = dict(nx=band.nx, ny=band.ny, dx=pcfg.dx / r, dy=pcfg.dy / r,
                  dt=float(np.float32(pcfg.dt) / np.float32(rt)),
                  specified=False, nested=True, grid_id=int(pcfg.grid_id) + 1,
                  spec_zone=1, relax_zone=4, spec_bdy_width=5)
    values.update(overrides)
    return dataclasses.replace(pcfg, **values)


def chain_experiment(root_cfg: RunConfig, bands_and_cfgs, *, run_seconds: float,
                     time_step: int):
    """The minimal :class:`woof.experiment.ExperimentConfig` whose schedule
    the clock walks: the root and one domain per ``(band, ccfg)``, each the
    previous one's child (the children's levels do not enter the
    schedule)."""
    from datetime import datetime
    from woof.experiment import DomainConfig, ExperimentConfig, VerticalConfig
    root_run = dataclasses.replace(root_cfg, run_seconds=run_seconds, grid_id=1,
                                   output_interval_s=run_seconds,
                                   restart_interval_s=0.0)
    domains = [DomainConfig(grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
                            parent_grid_ratio=1, parent_time_step_ratio=1,
                            history_interval_s=run_seconds, run=root_run,
                            time_step=int(time_step))]
    for n, (band, ccfg) in enumerate(bands_and_cfgs):
        child_run = dataclasses.replace(ccfg, run_seconds=run_seconds, grid_id=n + 2,
                                        output_interval_s=run_seconds,
                                        restart_interval_s=0.0)
        domains.append(DomainConfig(
            grid_id=n + 2, parent_id=n + 1, i_parent_start=band.i_parent_start,
            j_parent_start=band.j_parent_start,
            parent_grid_ratio=band.parent_grid_ratio,
            parent_time_step_ratio=band.parent_time_step_ratio,
            history_interval_s=run_seconds, run=child_run, time_step=None))
    return ExperimentConfig(
        name="partial-nest-chain", start_time=datetime(2000, 1, 1),
        run_seconds=run_seconds,
        vertical=VerticalConfig(eta_levels=(), p_top=0.0, hybrid_opt=root_cfg.hybrid_opt,
                                etac=root_cfg.etac),
        projection=None, restart_interval_s=0.0, domains=tuple(domains))


def two_domain_experiment(pcfg: RunConfig, ccfg: RunConfig, band: PartialNestBand,
                          *, run_seconds: float, time_step: int):
    """:func:`chain_experiment` for one band (D3's entry)."""
    return chain_experiment(pcfg, ((band, ccfg),), run_seconds=run_seconds,
                            time_step=time_step)


def run(nests, *, run_seconds: float, time_step: int,
        on_parent_step=None, on_child_step=None, root_step=None):
    """Drive a root and its partial chain (one :class:`PartialNest` or a
    coarse-to-fine sequence, each band the next one's parent) through
    :func:`woof.core.clock.execute_schedule`: a parent first, its child
    ``parent_time_step_ratio`` times per parent step, feedback last.
    ``on_parent_step(nest, clock)`` fires after the root's step (with the
    first band), ``on_child_step(nest, clock)`` after each band's step.
    ``root_step(state, cfg)`` steps the root (default
    :func:`woof.core.dycore.step`).  Returns the executor's report."""
    from woof.core import dycore
    from woof.core.clock import build_schedule, execute_schedule
    nests = (nests,) if isinstance(nests, PartialNest) else tuple(nests)
    for n in range(1, len(nests)):
        if nests[n].parent is not nests[n - 1].child:
            raise PartialNestRefused("a chain's band n must be band n-1's child")
    root = nests[0]
    exp = chain_experiment(root.pcfg, [(n.band, n.ccfg) for n in nests],
                           run_seconds=run_seconds, time_step=time_step)
    schedule = build_schedule(exp)
    step_root = dycore.step if root_step is None else root_step

    def on_step(gid, clock):
        if gid == 1:
            t0 = time.perf_counter()
            root.snapshot(root.a)
            step_root(root.parent, root.pcfg)
            root.stats["parent_seconds"] += time.perf_counter() - t0
            if on_parent_step is not None:
                on_parent_step(root, clock)
            return
        nest = nests[gid - 2]
        if gid - 1 < len(nests):
            nests[gid - 1].snapshot(nests[gid - 1].a)
        nest.child_step(clock)
        if on_child_step is not None:
            on_child_step(nest, clock)

    def on_force(cid, pid, child_clock, parent_clock):
        nests[cid - 2].force(child_clock, parent_clock)

    def on_feedback(cid, pid, child_clock, parent_clock):
        nests[cid - 2].feedback()

    return execute_schedule(schedule, clocks=schedule.clock.clocks(), on_step=on_step,
                            on_force=on_force, on_feedback_commit=on_feedback)


class PartialNestCoupler:
    """The node coupler protocol of :func:`woof.core.model.execute_experiment`
    (``force``, the three-phase feedback, ``valid``) over one
    :class:`PartialNest` (:func:`woof.core.model.partial_nest_steppers`)."""

    def __init__(self, nest: PartialNest):
        self.nest = nest
        self.valid = False

    def force(self, node) -> None:
        self.nest.force(node.clock, node.parent.clock)
        node.clock.mark_force()
        self.valid = True

    def feedback_prepare(self, node, out) -> None:
        return None

    def feedback_commit(self, node) -> None:
        self.nest.feedback()

    def feedback_finalize(self, node) -> None:
        return None
