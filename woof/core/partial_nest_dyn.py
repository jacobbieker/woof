"""Partial-depth nest, Phase D2: the child's dynamics with a flux through its top.

The child is its parent's discrete column cut at a pure-pressure full level
``k_top`` (:mod:`woof.core.partial_nest`); its ``nz`` is ``k_top`` and its
full level ``nz`` is the interface.  Above it the parent alone carries the
atmosphere, so the child's top is not a lid: the parent's coupled eta mass
flux ``ww`` passes through it.  This module drives the new kernel entry
points of ``kernels/nest_top.cu`` (existing kernels are untouched):

* acoustic substep -- :func:`prepare_top_substep`: ``advance_uv_top`` (the
  interface pressure-gradient term sees the parent's p'' above),
  ``advance_mu_th_top`` (column mass tendency ``dmdt = sum_below(dnw*div) +
  ww_top``, the Omega'' recurrence up to the interface, the theta flux
  through it), ``nest_top_ph_predictor`` + ``advance_w_phi_top`` (the
  implicit w''/phi'' solve with w'' and phi'' specified at the interface);
  the implicit coefficients are the EXISTING ``calc_coefs`` run on the child
  (rows below the interface are exactly a full column's);
* big step -- :func:`calc_ww_top` (WRF ``calc_ww_cp`` with the interface
  flux) and :func:`add_top_face_flux` (upwinded scalar/momentum/theta flux
  through the top face, added to the existing advection tendencies, whose
  top-face flux on the child is zero);
* the parent-to-child top forcing -- :func:`interpolate_top_field` (WRF
  ``sint``) and :func:`time_interpolate` (across the parent step, one child
  small step at a time).

Sign: ``ww`` is Omega-signed, positive DOWNWARD (eta decreases upward), and
a positive interface flux adds mass to the child (``nest_top.cu`` header).

Mass scaling of ``ww_top``: the interface is pure pressure, so its full-level
mass coupling ``c1f*mu + c2f`` is ``c2f`` alone -- one constant, the same for
parent and child -- and the parent's coupled ``ww`` at ``k_top`` needs no
column-mass rescaling: ``sint`` it as it is.

Scope (D2): the default arithmetic build (strict ``GPUWM_WRF_EXACT*``
builds are refused by name), periodic lateral boundaries, no map factors, no
``damp_opt=3`` implicit damper (its layer is measured from the parent's
top).  The child's lateral frame (D3), the top relaxation zone (D3) and
feedback (D3) build on these entry points.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from woof.config import RunConfig
from woof.core.kernels import get_kernel
from woof.core.partial_nest import PartialNestRefused, interface_pressure
from woof.core.state import DTYPE, DomainState

_THREADS = 256
_MODULE = "nest_top"


def _xp():
    import cupy as cp
    return cp


def _blocks(n: int) -> tuple[int]:
    return ((n + _THREADS - 1) // _THREADS,)


def refuse_unsupported(cfg: RunConfig, state: DomainState | None = None, *,
                       nested_frame: bool = False) -> None:
    """Name every configuration the D2 entry points do not carry.

    ``nested_frame`` is D3's (:mod:`woof.core.partial_nest_run`): the child
    runs inside its parent's lateral frame (``cfg.nested``), which the entry
    points carry through the same ``spec_zone`` arguments as the existing
    acoustic kernels.  Every other lateral treatment stays refused."""
    from woof import wrf_exact
    if wrf_exact.ENABLED:
        raise PartialNestRefused(
            "the partial-depth nest's top entry points carry the default "
            "arithmetic build only; this process is strict (GPUWM_WRF_EXACT=1)")
    framed = (nested_frame and getattr(cfg, "nested", False)
              and not (cfg.open_x or cfg.open_y or getattr(cfg, "specified", False)))
    if not framed and (cfg.open_x or cfg.open_y or getattr(cfg, "specified", False)
                       or getattr(cfg, "nested", False)):
        raise PartialNestRefused(
            "the D2 top entry points are driven on periodic lateral boundaries; "
            "the child's lateral frame joins them in D3")
    if cfg.damp_opt == 3:
        raise PartialNestRefused(
            "damp_opt=3 measures its layer from the model top, which a partial-"
            "depth child does not have; the child's top zone is D3's")
    if state is not None and state.has_msf:
        raise PartialNestRefused("the D2 top entry points carry no map factors")
    if cfg.nz + 1 > 129:
        raise PartialNestRefused("the child's w''-phi'' column is limited to 128 levels")


@dataclass(frozen=True)
class Interface:
    """The parent's coordinate words at the interface full level ``k_top``."""

    k_top: int
    fnm: np.float32
    fnp: np.float32
    rdnw: np.float32

    @classmethod
    def from_parent(cls, parent: DomainState, k_top: int) -> "Interface":
        cp = _xp()
        if float(cp.asnumpy(parent.c3f)[k_top]) != 0.0:
            raise PartialNestRefused(
                f"full level {k_top} is terrain-following (c3f != 0); the interface "
                "must be a pure pressure level")
        return cls(int(k_top),
                   DTYPE(cp.asnumpy(parent.fnm)[k_top]),
                   DTYPE(cp.asnumpy(parent.fnp)[k_top]),
                   DTYPE(cp.asnumpy(parent.rdnw)[k_top]))


@dataclass
class TopForcing:
    """Everything specified at the interface, per child column ``(ny, nx)``.

    Acoustic perturbation variables are relative to the child's own RK-stage
    reference, like every ``*_pp`` field.  ``ww_pp``, ``w_top``, ``ph_top``,
    ``ww_above``, ``p_above`` and ``p_above_old`` change every small step
    (fill them in place, e.g. with :func:`time_interpolate`); ``ww_rf``,
    ``t_above`` and ``pht_above`` are fixed over an RK stage.
    """

    ww_pp: object        # Omega'' at the interface
    ww_rf: object        # stage-reference Omega at the interface
    adj_pp: object       # column-mass-tendency adjustment (zero for a free nest)
    adj_rf: object
    t_above: object      # theta_t* = thb + thp at the parent half level k_top
    p_above: object      # p'' at the parent half level k_top
    p_above_old: object  # its previous-substep value (divergence damping)
    ww_above: object     # Omega'' at the parent full level k_top + 1
    pht_above: object    # phi_t* = phb + php at the parent full level k_top + 1
    w_top: object        # solved w'' at the interface
    ph_top: object       # solved phi'' at the interface

    @classmethod
    def zeros(cls, ny: int, nx: int) -> "TopForcing":
        cp = _xp()
        return cls(**{f.name: cp.zeros((ny, nx), dtype=DTYPE)
                      for f in dataclasses.fields(cls)})


def truncate_state(parent: DomainState, cfg: RunConfig, k_top: int, *,
                   coord=None) -> tuple[DomainState, RunConfig]:
    """The parent's state below full level ``k_top`` as a child state.

    Every vertically staggered array keeps its levels ``< k_top`` (half) or
    ``<= k_top`` (full); 2-D fields and scalars are copied.  With ``coord``
    (the parent's float64 :class:`~woof.core.grid.VerticalCoord`) the child
    takes D1's truncated coordinate -- top ``p_i``, ``mu' = mu - (p_i -
    p_top)``, re-expressed ``c2``/``c4`` -- otherwise it keeps the parent's
    words for ``mub2d``, ``p_top`` and every coefficient (the same column in
    the parent's own variables, which is what a word-exact comparison with
    the parent needs).
    """
    cp = _xp()
    nz = cfg.nz
    if not 3 <= k_top < nz:
        raise PartialNestRefused(f"k_top {k_top} must be an interior level, 3..{nz - 1}")
    ccfg = dataclasses.replace(cfg, nz=int(k_top))
    child = DomainState(ccfg)
    for name, dev in vars(child).items():
        src = getattr(parent, name, None)
        if not isinstance(dev, cp.ndarray) or not isinstance(src, cp.ndarray):
            continue
        if src.ndim != dev.ndim:
            continue
        if dev.ndim >= 1 and src.shape[0] in (nz, nz + 1) \
                and dev.shape[0] == src.shape[0] - (nz - k_top) \
                and src.shape[1:] == dev.shape[1:]:
            dev[...] = src[:dev.shape[0]]
        elif src.shape == dev.shape:
            dev[...] = src
    for name in ("cf1", "cf2", "cf3", "cfn", "cfn1", "p_top", "mub", "has_msf"):
        if hasattr(parent, name):
            setattr(child, name, getattr(parent, name))
    if coord is not None:
        from woof.core.partial_nest import truncated_vertical_coord
        ccoord = truncated_vertical_coord(coord, k_top)
        for name in ("c1h", "c2h", "c1f", "c2f", "c3h", "c4h", "c3f", "c4f"):
            getattr(child, name)[...] = cp.asarray(getattr(ccoord, name), dtype=DTYPE)
        shift = interface_pressure(coord, k_top) - float(coord.p_top)
        child.mub2d[...] = parent.mub2d - DTYPE(shift)
        child.p_top = DTYPE(ccoord.p_top)
        child.mub = None if parent.mub is None else DTYPE(float(parent.mub) - shift)
    return child, ccfg


def column_sums(state: DomainState, cfg: RunConfig, klev: int, ru=None, rv=None):
    """``(6, ny, nx)`` column divergence sums in the dmdt arithmetic of the
    acoustic and big-step kernels: below ``klev`` and over the column, for
    the perturbation and reference acoustic parts and for calc_ww
    (``nest_top_column_sums``)."""
    cp = _xp()
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    sums = cp.zeros((6, ny, nx), dtype=DTYPE)
    want = ru is not None
    get_kernel(_MODULE, "nest_top_column_sums")(
        _blocks(ny * nx), (_THREADS,),
        (state.u_pp, state.v_pp, state.u, state.v, state.mup, state.mub2d,
         state.dnw, state.c1h, state.c2h,
         ru if want else state.u, rv if want else state.v, np.int32(want),
         DTYPE(1.0 / cfg.dx), DTYPE(1.0 / cfg.dy), np.int32(klev),
         np.int32(nz), np.int32(ny), np.int32(nx), np.int32(cfg.open_x),
         np.int32(cfg.open_y), sums))
    return sums


def prepare_top_substep(state: DomainState, cfg: RunConfig, dtau: float,
                        coefficients: tuple, iface: Interface, top: TopForcing,
                        *, mudf=None, theta_upwind: bool = False,
                        rigid_predictor: bool = False, nested_frame: bool = False,
                        predictor=None):
    """One child acoustic substep, bound once per RK stage (the counterpart
    of :func:`woof.core.acoustic.prepare_acoustic_substep_launch`).

    ``coefficients`` is :func:`woof.core.acoustic.prepare_acoustic_coefficients`
    on the child.  ``top`` is read by reference at every launch.
    ``rigid_predictor`` takes the interface phi'' predictor as zero (a rigid
    lid's), instead of :func:`nest_top_ph_predictor`'s continuation.
    ``nested_frame``: see :func:`refuse_unsupported`.  ``predictor``, a
    callable ``predictor(rhs_top)`` launched after ``advance_mu_th_top``,
    replaces :func:`nest_top_ph_predictor` (D3's specified-interface
    predictor, :mod:`woof.core.partial_nest_run`).
    """
    from woof.core.acoustic import (_base3d, _boundary_x, _boundary_y,
                                    _mass_w_boundary_zone, _spec_zone)
    refuse_unsupported(cfg, state, nested_frame=nested_frame)
    cp = _xp()
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    mu_old = state.scratch((ny, nx), "acoustic_mu_pp_old")
    th_old = state.scratch((nz, ny, nx), "acoustic_th_pp_old")
    rhs_top = state.scratch((ny, nx), "nest_top_rhs")
    c2a, a, alpha, gam = coefficients[:4]
    cq = (state.p, state.p, state.p, False) if len(coefficients) == 4 else coefficients[4:]
    cqu, cqv, cqw, use_cq = cq
    mudf_arg = state.mup if mudf is None else mudf
    write_mudf = np.int32(mudf is not None)
    rdx, rdy = DTYPE(1.0 / cfg.dx), DTYPE(1.0 / cfg.dy)
    dtau_arg = DTYPE(dtau)
    base3d = _base3d(state)
    nz_a, ny_a, nx_a = np.int32(nz), np.int32(ny), np.int32(nx)
    block = (_THREADS,)
    uv_grid = _blocks(nz * (ny + 1) * (nx + 1))
    col_grid = _blocks(ny * nx)

    uv = get_kernel(_MODULE, "advance_uv_top")
    uv_prefix = (
        state.u_pp, state.v_pp, state.ru_t, state.rv_t,
        state.p_pp, state.p_pp_old, state.ph_pp, state.php, state.phb,
        state.alt, state.al_pp, state.pb, state.mup, state.mu_pp, state.mub2d,
        state.c1h, state.c2h, state.fnm, state.fnp, state.rdnw,
        cqu, cqv, np.int32(use_cq), state.cf1, state.cf2, state.cf3,
        np.int32(cfg.top_lid), rdx, rdy, dtau_arg)
    uv_suffix = (np.int32(_spec_zone(cfg)), base3d, nz_a, ny_a, nx_a,
                 top.p_above, top.p_above_old, iface.fnm, iface.fnp)
    uv_first = uv_prefix + (DTYPE(0.0),) + uv_suffix
    uv_later = uv_prefix + (DTYPE(cfg.smdiv),) + uv_suffix

    mu = get_kernel(_MODULE, "advance_mu_th_top")
    mu_args = (
        state.u_pp, state.v_pp, state.u, state.v, state.mup, state.mu_pp,
        mu_old, state.rmu_t, mudf_arg, write_mudf, state.thp, state.thb,
        state.th_pp, th_old, state.rth_t, state.ww_pp, state.p_pp,
        state.p_pp_old, state.dnw, state.rdnw, state.fnm, state.fnp,
        state.c1h, state.c2h, state.mub2d, rdx, rdy, dtau_arg, base3d,
        nz_a, ny_a, nx_a, np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg)),
        np.int32(_mass_w_boundary_zone(cfg)),
        top.ww_pp, top.ww_rf, top.adj_pp, top.adj_rf, top.t_above,
        iface.fnm, iface.fnp, np.int32(bool(theta_upwind)))

    pred = get_kernel(_MODULE, "nest_top_ph_predictor")
    pred_args = (
        rhs_top, state.w_pp, state.ph_pp, state.rph_t, state.ww_pp,
        state.php, state.phb, state.mup, state.mub2d, state.rdnw,
        state.c1f, state.c2f, top.ww_above, top.pht_above,
        iface.fnm, iface.fnp, iface.rdnw, dtau_arg, DTYPE(cfg.epssm),
        base3d, nz_a, ny_a, nx_a)

    w = get_kernel(_MODULE, "advance_w_phi_top")
    w_args = (
        state.w_pp, state.ph_pp, state.rw_t, state.rph_t, state.ww_pp,
        state.mu_pp, mu_old, state.th_pp, th_old,
        state.thp, state.thb, state.php, state.phb, state.alt, c2a,
        a, alpha, gam, state.mup, state.u_pp, state.v_pp, state.ht,
        state.rdn, state.rdnw, state.fnm, state.fnp,
        state.c1h, state.c2h, state.c1f, state.c2f, state.mub2d,
        cqw, np.int32(use_cq), state.p_pp, state.al_pp,
        state.cf1, state.cf2, state.cf3, rdx, rdy, dtau_arg, DTYPE(cfg.epssm),
        np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg)),
        np.int32(_mass_w_boundary_zone(cfg)), base3d, nz_a, ny_a, nx_a,
        rhs_top, top.w_top, top.ph_top)

    def launch(*, first: bool) -> None:
        uv(uv_grid, block, uv_first if first else uv_later)
        mu(col_grid, block, mu_args)
        if rigid_predictor:
            rhs_top[...] = 0.0
        elif predictor is not None:
            predictor(rhs_top)
        else:
            pred(col_grid, block, pred_args)
        w(col_grid, block, w_args)

    launch.rhs_top = rhs_top
    return launch


def calc_ww_top(state: DomainState, cfg: RunConfig, ru, rv, ww_top, adj=None, *,
                nested_frame: bool = False):
    """WRF ``calc_ww_cp`` on the child with the interface flux ``ww_top``
    (``dycore._omega_ref``'s arithmetic, ``dmdt`` gaining ``ww_top`` and
    ``adj``).  Returns the ``(nz+1, ny, nx)`` Omega in the ``rk_ww`` slot."""
    refuse_unsupported(cfg, state, nested_frame=nested_frame)
    cp = _xp()
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    ww = state.scratch((nz + 1, ny, nx), "rk_ww")
    if adj is None:
        adj = cp.zeros((ny, nx), dtype=DTYPE)
    rdx = np.float32(1.0) / np.float32(cfg.dx)
    rdy = np.float32(1.0) / np.float32(cfg.dy)
    get_kernel(_MODULE, "calc_ww_top")(
        _blocks(ny * nx), (_THREADS,),
        (ru, rv, state.dnw, state.c1h, state.msft, np.int32(state.has_msf),
         rdx, rdy, np.int32(nz), np.int32(ny), np.int32(nx), ww_top, adj, ww))
    return ww


_STAGGER = {"": 0, "x": 1, "y": 2}


def add_top_face_flux(tend, q, q_above, ww_top, state: DomainState,
                      cfg: RunConfig, iface: Interface, stagger: str = "",
                      *, centered: bool = False, nested_frame: bool = False,
                      flux=None):
    """Add the flux of ``q`` through the child's top face to ``tend``'s top
    half level, upwinded on the face's Omega (``nest_top_face_flux``);
    returns the per-point face flux.  ``stagger`` "" (mass/theta/scalars),
    "x" (u) or "y" (v); ``q_above`` is the parent's value at half level
    ``k_top`` on the same staggering.  ``flux`` may pass the output plane
    (every point is written)."""
    refuse_unsupported(cfg, state, nested_frame=nested_frame)
    cp = _xp()
    ny, nx = cfg.ny, cfg.nx
    s = _STAGGER[stagger]
    shape = (ny + (s == 2), nx + (s == 1))
    if flux is None:
        flux = cp.zeros(shape, dtype=DTYPE)
    get_kernel(_MODULE, "nest_top_face_flux")(
        _blocks(shape[0] * shape[1]), (_THREADS,),
        (tend, q, q_above, ww_top, state.rdnw, flux, iface.fnm, iface.fnp,
         np.int32(bool(centered)), np.int32(s),
         np.int32(cfg.open_x or (nested_frame and cfg.nested)),
         np.int32(cfg.open_y or (nested_frame and cfg.nested)),
         np.int32(cfg.nz), np.int32(ny), np.int32(nx)))
    return flux


def time_interpolate(a, b, frac: float, out=None):
    """``a + frac*(b - a)``: a parent interface field at fraction ``frac``
    of the parent step, for one child small step."""
    cp = _xp()
    if out is None:
        out = cp.empty_like(a)
    n = int(a.size)
    get_kernel(_MODULE, "nest_top_time_interp")(
        _blocks(n), (_THREADS,), (out, a, b, DTYPE(frac), np.int32(n)))
    return out


def interpolate_top_field(parent_field, reg, *, out=None):
    """A parent interface field ``(ny_p, nx_p)`` on the child grid with WRF
    ``sint`` (:func:`woof.core.nest_interp.sint`).  ``ww`` at the
    pure-pressure interface needs no mass rescaling (module docstring)."""
    from woof.core.nest_interp import sint
    return sint(parent_field, reg, out=out)
