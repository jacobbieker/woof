"""WSPD10MAX, U10MAX, V10MAX, W_UP_MAX, W_DN_MAX and GUST: in-model wind maxima.

The wind members of WRF's ``nwp_diagnostics`` family plus a surface gust
estimate, each a per-step running maximum over the history interval, reset
after every history write.  Same lifecycle as ``UP_HELI_MAX``
(:mod:`woof.core.uh_diag`): eager allocation under ``nwp_diagnostics = 1``
(:class:`woof.core.state.DomainState`), a fold in the dycore epilogue, a
restart-serialized accumulator, and a zeroing after each durable frame
(:mod:`woof.core.interval_maxima`).  Trajectory-inert: nothing here writes a
model-evolution array.

WHAT IS PINNED, member by member
--------------------------------
WRF v4.6.1 ``phys/module_diag_nwp.F`` SUBROUTINE ``diagnostic_output_nwp``
(called every step from ``phys/module_diagnostics_driver.F`` under
``nwp_diagnostics == 1``, no ``diagflag`` gate; Registry.EM_COMMON:2079-2081,
IO "rh02", package ``nwp_output``):

* ``WSPD10MAX`` -- ``wind_vel = sqrt(u10*u10 + v10*v10)``; ``IF (wind_vel
  .GT. wspd10max) wspd10max = wind_vel``.  ``u10``/``v10`` are the
  surface-layer scheme's 10 m wind (``grid%u10``/``grid%v10``).
* ``U10MAX`` / ``V10MAX`` -- the 10 m components AT the instant
  ``WSPD10MAX`` last grew, i.e. the direction of the interval's peak wind.
  This is WRF's ``varstatisticsWIND`` rule (``phys/module_diag_cl.F``:
  ``IF (varuv > varuvmax) THEN varumax = varu; varvmax = varv``), NOT an
  independent max of each component.
* ``W_UP_MAX`` / ``W_DN_MAX`` -- ``DO k = kms, kme`` over every full level:
  ``IF (p8w(k) .GT. 40000. .AND. w(k) .GT. w_up_max) w_up_max = w(k)`` and
  the mirror with ``.LT.`` for ``w_dn_max``.  Both start the interval at 0,
  so a column with no updraft (downdraft) below 400 hPa reports 0.
  ``p8w`` is ``phy_prep``'s full-level pressure
  (``dyn_em/module_big_step_utilities_em.F``), transcribed in
  :func:`p8w_np`.

``GUST`` -- the per-step gust estimate of NOAA-EMC UPP ``CALGUST.f``
(Manikin 2003, with the RAP/HRRR/FV3R revision; the "mix down momentum from
the PBL" method of the operational RAP/HRRR post, Benjamin et al. 2016,
Mon. Wea. Rev. 144, 1669-1694), folded into a running max over the
history interval -- the same "max since the previous output" semantics as
ECMWF's ``10fg``.  Per column::

    SFCWIND = sqrt(u10**2 + v10**2)
    gust    = SFCWIND
    for k = 1 .. min(kpbl + 1, nz - 1):          # 0-based mass levels
        DZ   = zmid(k) - zsfc
        gust = max(gust, SFCWIND + (|V(k)| - SFCWIND)
                                   * (1 - min(0.5, DZ/2000)))

which is UPP's ``do K=LM-1,L-1,-1`` loop (UPP indexes top-down: ``LM-1`` is
the second mass level, ``L-1`` the level above the PBL-top level) with
``DELWIND = DELWIND*(1.0-MIN(0.5,DZ/2000.))``.  ``|V(k)|`` is the mass-point
(destaggered) model-level wind, ``zmid`` the mass-level height
``0.5*(z_at_w(k) + z_at_w(k+1))``, ``zsfc = z_at_w(0)``.  ``PBLH`` is the
height the run publishes as ``PBLH`` (the PBL scheme's field; SASE's
bulk-Richardson output height under ``bl_pbl_physics = 900``).

WOOF'S CHOICES WHERE THE SOURCES LEAVE ONE
------------------------------------------
1. ``kpbl`` (UPP's ``LPBL``, computed by UPP's caller as "the model level
   nearest the PBL top"): the LOWEST mass level whose height above ground
   is ``>= PBLH``; the top mass level when none is.  So the gust always
   samples every level inside the PBL plus one above it, and a PBLH of 0
   still reads level 1 (UPP's loop always reads ``LM-1``).
2. Without a surface-layer field (no physics driver, so no ``u10``/
   ``v10``), the 10 m members and ``GUST`` are not folded and stay 0.  For
   ``WSPD10MAX`` that is exactly WRF, whose ``u10`` is zero without
   ``sf_sfclay_physics``; for ``GUST`` it is a refusal to fabricate a gust
   from no surface wind.  A driver with a 10 m wind but no ``pblh`` field
   folds the 10 m members and leaves only ``GUST`` at 0.
   ``W_UP_MAX``/``W_DN_MAX`` are always folded.

RATIFIED DIVERGENCES FROM WRF (shared with UP_HELI_MAX where noted)
-------------------------------------------------------------------
1. Reset instant (shared, see ``uh_diag`` divergence 1).  WRF zeroes on the
   first step AFTER a history write (``MOD(itimestep-1, idump) == 0``) and
   then folds that step; woof zeroes immediately after the frame is durable
   and folds the completed step's state in the dycore epilogue.  Each
   frame's window is (T_prev, T_frame], including the frame's own instant.
2. ``p8w`` instant.  WRF's ``W_UP_MAX`` test reads the ``p8w`` ``phy_prep``
   built from the START-of-step state (first RK stage) against the
   end-of-step ``w``; woof evaluates both on the completed step.  Only the
   level mask near 400 hPa can differ, by one step's pressure change.
3. The 10 m wind is the surface-layer output of the step's physics call
   (start-of-step state), exactly WRF's ``grid%u10`` at diagnostic time;
   under ``bldt > dt`` both hold the last call's value.
4. ``U10MAX``/``V10MAX`` belong to WRF's ``output_diagnostics`` package
   (``clwrf`` climate statistics, reset on the auxhist3 interval and
   seeded from the first sample).  woof carries them in the
   ``nwp_diagnostics`` window beside ``WSPD10MAX`` (seeded at 0, reset at
   history writes), so the three always describe the same instant.
5. Sub-stepping (shared, ``uh_diag`` divergence 2): under a compatibility
   integrator running internal substeps the fold happens every internal
   step.

GPU/CPU: :func:`update_wind_maxima` launches ``wind_maxima_fold``
(``woof/core/kernels/wind_gust_diag.cu``) on device states and runs the
NumPy mirror below on host states.  The two agree bit for bit everywhere
except the TOP full level's ``p8w``, which is ``exp``/``log`` and therefore
libm-dependent by a few ULP; it can change a mask only for a model top
within those ULP of 400 hPa.

Streaming / tiles / multi-GPU: every member is column-local (the gust's u/v
destagger reads the column's own staggered faces), so no halo widening is
needed, unlike UP_HELI_MAX.  Nests: each domain owns its slots and is reset
by its own history write.
"""
from __future__ import annotations

import numpy as np

from woof.core.constants import G

DTYPE = np.float32

#: WRF's W_UP_MAX/W_DN_MAX pressure floor (module_diag_nwp.F), Pa.
W_EXTREMA_P8W_FLOOR_PA = DTYPE(40000.0)
#: UPP CALGUST's reduction depth and cap: (1 - min(0.5, dz/2000)).
GUST_REDUCTION_DEPTH_M = DTYPE(2000.0)
GUST_REDUCTION_CAP = DTYPE(0.5)

#: (wrfout name, serialized scratch slot), in emission order.  Every slot is
#: a restart-carried running max reset after each history write.
WIND_MAXIMA_FIELDS: tuple[tuple[str, str], ...] = (
    ("WSPD10MAX", "wspd10max"),
    ("U10MAX", "u10max"),
    ("V10MAX", "v10max"),
    ("W_UP_MAX", "w_up_max"),
    ("W_DN_MAX", "w_dn_max"),
    ("GUST", "gust_max"),
)
WIND_MAXIMA_SLOTS: tuple[str, ...] = tuple(s for _, s in WIND_MAXIMA_FIELDS)

_THREADS = 256
_F0 = DTYPE(0.0)
_F05 = DTYPE(0.5)
_F1 = DTYPE(1.0)
_G = DTYPE(G)


# ---------------------------------------------------------------------------
# NumPy mirror (identical per-element operation order to the CUDA kernel)
# ---------------------------------------------------------------------------

def _phb3(phb, nzf, ny, nx):
    phb = np.asarray(phb, dtype=DTYPE)
    if phb.ndim == 1:
        return np.broadcast_to(phb[:, None, None], (nzf, ny, nx))
    return phb


def z_at_w_np(ph, phb) -> np.ndarray:
    """``z_at_w = (phb + ph)/g`` on full levels, FP32."""
    ph = np.asarray(ph, dtype=DTYPE)
    phb3 = _phb3(phb, *ph.shape)
    return ((phb3 + ph) / _G).astype(DTYPE)


def z_mass_np(z_at_w) -> np.ndarray:
    """``0.5*(z_at_w(k) + z_at_w(k+1))`` on mass levels, FP32."""
    return (_F05 * (z_at_w[:-1] + z_at_w[1:])).astype(DTYPE)


def p8w_np(p, ph, phb, fnm, fnp) -> np.ndarray:
    """WRF ``phy_prep``'s full-level pressure from full mass-level ``p``.

    Interior ``fnm(k)*p_phy(k) + fnp(k)*p_phy(k-1)``; bottom
    ``w1*p_phy(1) + w2*p_phy(2)`` and top ``exp(w1*log(p_phy(kde-1)) +
    w2*log(p_phy(kde-2)))`` with ``w1 = (z0-z2)/(z1-z2)``, ``w2 = 1-w1``.
    """
    p = np.asarray(p, dtype=DTYPE)
    nz = p.shape[0]
    if nz < 2:
        raise ValueError(f"p8w needs at least two mass levels, got {nz}")
    fnm = np.asarray(fnm, dtype=DTYPE)
    fnp = np.asarray(fnp, dtype=DTYPE)
    zw = z_at_w_np(ph, phb)
    zm = z_mass_np(zw)
    out = np.empty((nz + 1,) + p.shape[1:], dtype=DTYPE)
    for kw in range(1, nz):
        out[kw] = fnm[kw] * p[kw] + fnp[kw] * p[kw - 1]
    w1 = (zw[0] - zm[1]) / (zm[0] - zm[1])
    w2 = _F1 - w1
    out[0] = w1 * p[0] + w2 * p[1]
    w1 = (zw[nz] - zm[nz - 2]) / (zm[nz - 1] - zm[nz - 2])
    w2 = _F1 - w1
    with np.errstate(divide="ignore", invalid="ignore"):
        out[nz] = np.exp(w1 * np.log(p[nz - 1]) + w2 * np.log(p[nz - 2]))
    return out


def fold_w_extrema_np(w, p8w, w_up_max, w_dn_max) -> None:
    """``W_UP_MAX``/``W_DN_MAX`` fold over every full level, in place."""
    floor = W_EXTREMA_P8W_FLOOR_PA
    for kw in range(w.shape[0]):
        below = p8w[kw] > floor
        wk = w[kw]
        np.copyto(w_up_max, wk, where=below & (wk > w_up_max))
        np.copyto(w_dn_max, wk, where=below & (wk < w_dn_max))


def fold_wind10_np(u10, v10, wspd10max, u10max, v10max) -> np.ndarray:
    """``WSPD10MAX`` and the components at its instant; returns |V10|."""
    u10 = np.asarray(u10, dtype=DTYPE)
    v10 = np.asarray(v10, dtype=DTYPE)
    sfc = np.sqrt(u10 * u10 + v10 * v10).astype(DTYPE)
    grow = sfc > wspd10max
    np.copyto(wspd10max, sfc, where=grow)
    np.copyto(u10max, u10, where=grow)
    np.copyto(v10max, v10, where=grow)
    return sfc


def pbl_top_level_np(z_mass_agl, pblh) -> np.ndarray:
    """woof's ``LPBL``: lowest mass level with height AGL ``>= PBLH``."""
    nz = z_mass_agl.shape[0]
    at_or_above = z_mass_agl >= np.asarray(pblh, dtype=DTYPE)[None]
    found = at_or_above.any(axis=0)
    return np.where(found, at_or_above.argmax(axis=0), nz - 1).astype(np.int32)


def gust_np(u, v, ph, phb, u10, v10, pblh) -> np.ndarray:
    """The per-step UPP ``CALGUST`` (RAPR/GFS/FV3R branch) gust, FP32."""
    u = np.asarray(u, dtype=DTYPE)
    v = np.asarray(v, dtype=DTYPE)
    nz = u.shape[0]
    u10 = np.asarray(u10, dtype=DTYPE)
    v10 = np.asarray(v10, dtype=DTYPE)
    sfc = np.sqrt(u10 * u10 + v10 * v10).astype(DTYPE)
    zw = z_at_w_np(ph, phb)
    zagl = (z_mass_np(zw) - zw[0]).astype(DTYPE)
    kpbl = pbl_top_level_np(zagl, pblh)
    ktop = np.minimum(kpbl + 1, nz - 1)
    gust = sfc.copy()
    for k in range(1, nz):
        um = _F05 * (u[k, :, :-1] + u[k, :, 1:])
        vm = _F05 * (v[k, :-1, :] + v[k, 1:, :])
        wind = np.sqrt(um * um + vm * vm).astype(DTYPE)
        q = (zagl[k] / GUST_REDUCTION_DEPTH_M).astype(DTYPE)
        reduce = np.where(q < GUST_REDUCTION_CAP, q, GUST_REDUCTION_CAP)
        cand = (sfc + (wind - sfc) * (_F1 - reduce)).astype(DTYPE)
        np.copyto(gust, cand, where=(k <= ktop) & (cand > gust))
    return gust


def mirror_wind_maxima_step_np(u, v, w, p, ph, phb, fnm, fnp,
                               u10, v10, pblh, maxima: dict) -> None:
    """One step of every member on host arrays; ``maxima`` maps slot -> plane.

    ``u10 is None`` skips the 10 m members and the gust, ``pblh is None``
    only the gust (woof choice 2).
    """
    fold_w_extrema_np(np.asarray(w, dtype=DTYPE), p8w_np(p, ph, phb, fnm, fnp),
                      maxima["w_up_max"], maxima["w_dn_max"])
    if u10 is None:
        return
    fold_wind10_np(u10, v10, maxima["wspd10max"], maxima["u10max"],
                   maxima["v10max"])
    if pblh is None:
        return
    gust = gust_np(u, v, ph, phb, u10, v10, pblh)
    np.copyto(maxima["gust_max"], gust, where=gust > maxima["gust_max"])


# ---------------------------------------------------------------------------
# Model-facing lifecycle
# ---------------------------------------------------------------------------

def _surface_inputs(state):
    """``(u10, v10, pblh)`` from the attached physics driver, or ``None``.

    ``None`` when the driver carries no 10 m wind (no surface layer); the
    PBL height alone may be ``None``, which skips only ``GUST``.  It is the
    one the history writer publishes as ``PBLH``: ``fields['pblh']``,
    except under SASE whose YSU slot never runs and whose published height
    is computed by ``_sase_output_pblh`` from the current state.  On SASE
    that is one bulk-Richardson pass per step, the price of a gust that
    tracks the boundary layer the run actually has (SASE runs only).
    """
    physics = getattr(state, "physics", None)
    fields = getattr(physics, "fields", None)
    if not fields:
        return None
    u10, v10 = fields.get("u10"), fields.get("v10")
    if u10 is None or v10 is None:
        return None
    if getattr(physics, "sase_active", False):
        pblh = physics._sase_output_pblh()
    else:
        pblh = fields.get("pblh")
    return u10, v10, pblh


def _to_host(arr):
    return arr if isinstance(arr, np.ndarray) else arr.get()


def update_wind_maxima(state, cfg=None) -> None:
    """Fold this completed step into every wind running max.

    Reads only completed-step model state and the physics driver's surface
    fields; writes only this module's own scratch slots.  Call once per
    dycore step when ``nwp_diagnostics == 1``.  The grid is read off the
    state's own ``w`` (``cfg`` is accepted for the dycore's call shape and
    is not otherwise consulted), so a registry hook that has no config in
    hand (:class:`WindMaximaIntervalDiagnostic`) folds the same way.
    """
    nzf, ny, nx = (int(n) for n in state.w.shape)
    nz = nzf - 1
    if nz < 2:
        raise ValueError(
            "the nwp_diagnostics wind maxima need nz >= 2 (phy_prep's p8w "
            f"extrapolates from two mass levels); got nz={nz}")
    # Literal slot names: the preflight completeness scanner classifies
    # every scratch(...) call site statically.
    maxima = {
        "wspd10max": state.scratch((ny, nx), "wspd10max"),
        "u10max": state.scratch((ny, nx), "u10max"),
        "v10max": state.scratch((ny, nx), "v10max"),
        "w_up_max": state.scratch((ny, nx), "w_up_max"),
        "w_dn_max": state.scratch((ny, nx), "w_dn_max"),
        "gust_max": state.scratch((ny, nx), "gust_max"),
    }
    surface = _surface_inputs(state)
    if isinstance(state.u, np.ndarray):
        u10, v10, pblh = ((None, None, None) if surface is None
                          else tuple(None if a is None else _to_host(a)
                                     for a in surface))
        mirror_wind_maxima_step_np(
            state.u, state.v, state.w, state.p, state.php, _to_host(state.phb),
            _to_host(state.fnm), _to_host(state.fnp), u10, v10, pblh, maxima)
        return
    device_wind_maxima_step(
        state.u, state.v, state.w, state.p, state.php, state.phb,
        state.fnm, state.fnp, surface, maxima)


def device_wind_maxima_step(u, v, w, p, ph, phb, fnm, fnp, surface,
                            maxima: dict) -> None:
    """One step of every member on device arrays (``wind_maxima_fold``)."""
    from woof.core.kernels import get_kernel

    nzf, ny, nx = w.shape
    nz = nzf - 1
    for name, arr in (("u", u), ("v", v), ("w", w), ("p", p), ("ph", ph),
                      ("phb", phb)):
        if arr.dtype != np.float32 or not arr.flags.c_contiguous:
            raise ValueError(f"wind_gust_diag requires contiguous FP32 {name}")
    for name, arr in (("fnm", fnm), ("fnp", fnp)):
        if (arr.dtype != np.float32 or not arr.flags.c_contiguous
                or tuple(arr.shape) != (nz,)):
            raise ValueError(
                f"wind_gust_diag requires a contiguous FP32 ({nz},) {name}, "
                f"got {arr.dtype} {tuple(arr.shape)}")
    gust_max = maxima["gust_max"]
    u10, v10, pblh = (None, None, None) if surface is None else surface
    for name, arr in (("u10", u10), ("v10", v10), ("pblh", pblh)):
        if arr is not None and (arr.dtype != np.float32
                                or not arr.flags.c_contiguous
                                or tuple(arr.shape) != (ny, nx)):
            raise ValueError(
                f"wind_gust_diag requires a contiguous FP32 ({ny}, {nx}) "
                f"{name}, got {arr.dtype} {tuple(arr.shape)}")
    has_wind10 = int(u10 is not None)
    has_pblh = int(has_wind10 and pblh is not None)
    # An absent input is never read (its flag is 0); any valid device
    # pointer stands in for it.
    u10 = gust_max if u10 is None else u10
    v10 = gust_max if v10 is None else v10
    pblh = gust_max if pblh is None else pblh
    blocks = ((nx * ny) + _THREADS - 1) // _THREADS
    get_kernel("wind_gust_diag", "wind_maxima_fold")(
        (blocks,), (_THREADS,),
        (u, v, w, p, ph, phb, np.int32(0 if phb.ndim == 1 else 1),
         fnm, fnp, u10, v10, pblh, np.int32(has_wind10), np.int32(has_pblh),
         _G,
         maxima["wspd10max"], maxima["u10max"], maxima["v10max"],
         maxima["w_up_max"], maxima["w_dn_max"], gust_max,
         np.int32(nz), np.int32(ny), np.int32(nx)))


def zero_domain_slot(state, slot: str) -> None:
    """Zero one whole-domain running max wherever the domain keeps it.

    The same two roads as ``uh_diag``'s reset, and for the same reason: on a
    streamed domain the accumulator lives in the tiling store, so the ranked
    road zeroes its slabs (``StreamedDomain.zero_scratch``) and every other
    road zeroes each view :func:`woof.core.streaming.live_scratch` returns
    (the state's copy and the store's).  A state that never allocated the
    slot, or a test double with no scratch pool, is a no-op.
    """
    from woof.core.streaming import live_scratch

    streamed = getattr(state, "_streamed_domain", None)
    zero = getattr(streamed, "zero_scratch", None)
    if zero is not None and zero(slot):
        existing = getattr(state, "existing_scratch", None)
        buf = existing(slot) if existing is not None else None
        if buf is not None:
            buf[...] = 0.0
        return
    for buf in live_scratch(state, slot):
        buf[...] = 0.0


def reset_wind_maxima(state) -> None:
    """Zero every wind running max after its history frame is durable.

    WRF's history-interval reset (module_diag_nwp.F ``reset_arrays``) moved
    to the write itself, ratified divergence 1.
    """
    for slot in WIND_MAXIMA_SLOTS:
        zero_domain_slot(state, slot)


class WindMaximaIntervalDiagnostic:
    """These six maxima in the shape of a per-history-interval registry member.

    For a registry that drives ``on_step`` / ``history_fields`` /
    ``close_interval`` per domain (the integration branch's
    ``woof.core.interval_diagnostics``): one object per domain state,
    delegating to the same fold, emission and reset as the direct hooks.
    Attach it INSTEAD OF, never beside, the direct hooks (the dycore
    epilogue fold under ``nwp_diagnostics == 1`` and the publishers'
    ``reset_history_interval_maxima``): both would fold every step twice
    (harmless for a max, but wasted) and reset twice.

    The accumulators stay serialized scratch slots, so a resumed run
    continues the open interval -- unlike a registry member whose buffers
    live outside the restart walk.  Every publisher can serve it: the
    reset goes through :func:`zero_domain_slot`, which
    reaches a streamed domain's store as well, so ``refuse_publisher``
    never refuses.
    """

    name = "wind_maxima"

    def on_step(self, state, dt) -> None:
        if getattr(state, "existing_scratch", lambda _s: None)(
                "wspd10max") is not None:
            update_wind_maxima(state)

    def history_fields(self, state) -> dict:
        existing = getattr(state, "existing_scratch", None)
        if existing is None:
            return {}
        return {name: existing(slot) for name, slot in WIND_MAXIMA_FIELDS
                if existing(slot) is not None}

    def close_interval(self, state, *, valid_time=None, output_dir=None,
                       grid_id=None) -> None:
        reset_wind_maxima(state)

    def refuse_publisher(self, reason) -> None:
        return None
