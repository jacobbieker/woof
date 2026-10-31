"""WRF ``use_theta_m = 1``: the moist potential temperature prognostic.

WRF v4 integrates ``t_2 = theta_m - T0`` with ``theta_m = theta*(1 +
Rv/Rd*qv)`` (Registry.EM_COMMON:2864, ``rconfig integer use_theta_m``, default
1 since V4.0).  The dycore carries the moisture in the prognostic instead of
in the equation of state, and the physics, which is written for dry theta,
sees dry theta through a handful of conversions.  Every ``use_theta_m``
branch of WRF v4.7.1's dyn_em/phys/share, and where woof answers it:

==========================================  ===================================
WRF branch                                  woof
==========================================  ===================================
module_initialize_real.F:4923 (and _ideal   :func:`convert_initial_state`, at the
:1726): ``t_2 = (t_2+T0)(1+Rv/Rd qv) - T0``  end of ``initialize_real`` and
at the end of initialization; the BC file   ``init_moist_balanced``; boundary
is built from that ``t_2``                  frames couple the converted state
calc_p_rho_phi :1061 (nonhydrostatic EOS    ``update_diagnostics``: the existing
``qvf = 1``); start_em :821-831             ``calc_p_alpha`` with ``moist = 0``
phy_prep :4830 (``th_phy`` dry)             :func:`physics_window`
conv_t_tendf_to_moist :6674 (first_rk_      :func:`convert_physics_tendencies`
step_part2.F:1005)
moist_physics_prep_em :5505 /               :func:`microphysics_window` and
moist_physics_finish_em :5735               :func:`finish_microphysics`
update_phys_fields (diagnostics driver      :func:`dry_view` / :func:`dry_theta`
:1252): history ``T`` = dry theta - T0      (history T, post-step diagnostics)
adjust_tempqv (nest_init_utils.F:852)       ``woof.ingest.nest_init`` passes
                                            ``use_theta_m`` from the state
nest_vertical / nest_interp                 their existing ``use_theta_m``
phy_prep_part2 :5323 (RTHFTEN for GD/GF/    refused by name (config)
G3/NTiedtke)
vertical_diffusion_2 :4294 / km_opt=5       refused by name unless the surface
implicit :8116 (diff_opt=2, no PBL)         heat and moisture fluxes are zero
dfi.F, wrf_timeseries.F, track_driver.F,    not features of woof (dfi, tslist,
firebrand_spotting.F:2248                   track) or refused by name
input_wrf.F:200 (file/namelist agreement)   the wrfinput doors (namelist_import)
==========================================  ===================================

The small step has no ``use_theta_m`` branch: its linearized pressure is
written in t_2 itself, so it integrates theta_m unchanged.

``state.theta_m`` records which theta the state's ``thp`` holds.  It is
False on every new state (every door builds dry theta) and becomes True only
where a door converts (:func:`convert_initial_state`) or a restart of a
theta_m run restores one.  The dycore refuses a step whose state and
configuration disagree, so a door that does not convert is refused by name
instead of integrating dry theta as theta_m.
"""

from __future__ import annotations

import copy
import weakref
from contextlib import contextmanager

import numpy as np

from woof.core import constants as c

_TPB = 256

#: Scratch slots (all rebuilt: written before every read within one call).
STASH_SLOT = "theta_m_stash"          # the theta_m' a window put aside
DRY_SLOT = "theta_m_dry"              # dry theta' (WRF th_phy_m_t0)
RTHETA_SLOT = "theta_m_rtheta"        # conv_t_tendf_to_moist's output
QV_DIABATIC_SLOT = "theta_m_qv_diabatic"  # moist_physics_prep_em's qv_diabatic
H_PRE_SLOT = "theta_m_h_pre"          # moist_physics_prep_em's h_diabatic
TH_PHY_SLOT = "theta_m_th_phy"        # the scheme's post-call dry th_phy
SCRATCH_SLOTS = (STASH_SLOT, DRY_SLOT, RTHETA_SLOT, QV_DIABATIC_SLOT,
                 H_PRE_SLOT, TH_PHY_SLOT)

#: The microphysics schemes whose theta_m finish is wired: every scheme
#: woof runs (woof.config.MP_PHYSICS_ACCEPTED).  Each adapter hands its
#: post-call full dry theta to :func:`finish_microphysics` -- through
#: :func:`woof.core.microphysics.moist_physics_finish` (Kessler, Lin, WSM3,
#: WSM5, Ferrier-Aligo, WSM6, WSM6R, Milbrandt-Yau, Morrison, WDM6, NSSL,
#: WSM7, Thompson aerosol-aware, fast SBM, P3, TEMPO, and MAD-WRF's
#: process-free mp_physics = 96, whose theta out is the theta in) or, for classic
#: Thompson's fused finishes (kernels/thompson.cu thompson_adapter_finish,
#: kernels/thompson_classic.cu tc_finish, the generated column), directly
#: after them -- and :func:`microphysics_window` makes WRF's theta_m finish
#: from it.  The scheme kernels and their dry finishes keep every byte.
GENERIC_FINISH_MICROPHYSICS = frozenset(
    {1, 2, 3, 4, 5, 6, 8, 9, 10, 16, 18, 24, 28, 30, 50, 96, 106, 900})

#: WRF phy_prep_part2's RTHFTEN branch (module_big_step_utilities_em.F:
#: 5309-5333): the cumulus schemes that read the advective theta forcing,
#: which WRF converts to dry theta under use_theta_m = 1
#: (:func:`convert_advective_forcing`).  Grell-Devenyi (93) is not ported
#: (woof.config refuses it for every run).
RTHFTEN_CUMULUS = {3: "Grell-Freitas", 5: "Grell 3D", 16: "New Tiedtke",
                   93: "Grell-Devenyi"}

_FINISH_STATE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()

#: The radar latent-heating forcing (woof.da.radar_tten, the HRRR fork's
#: mp_tend_radar on WRF V3.9) has no WRF v4.7.1 statement at all, under
#: either use_theta_m: its replacement of the microphysics increment is the
#: fork's dry-theta branch.
RADAR_REFUSAL = (
    "use_theta_m = 1 with a radar latent-heating forcing (mp_tend_radar) is "
    "not wired: the forcing is the HRRR fork's V3.9 dry-theta replacement of "
    "the microphysics increment, for which WRF v4.7.1 has no statement")


def run_theta_m(cfg) -> bool:
    """Does this configuration integrate theta_m?  WRF's branches test
    ``use_theta_m == 1 .AND. P_QV >= PARAM_FIRST_SCALAR``: a dry run has no
    vapour and integrates dry theta whatever the switch says."""
    return int(getattr(cfg, "use_theta_m", 0) or 0) == 1 and bool(
        getattr(cfg, "moist", False))


def state_theta_m(state) -> bool:
    """Does ``state.thp`` hold theta_m'?"""
    return bool(getattr(state, "theta_m", False))


def refusal(cfg) -> str | None:
    """Why this configuration cannot integrate theta_m, or ``None``."""
    value = getattr(cfg, "use_theta_m", 0)
    if isinstance(value, bool) or value not in (0, 1):
        return f"use_theta_m must be 0 or 1 (WRF Registry.EM_COMMON:2864), got {value!r}"
    if not run_theta_m(cfg):
        return None
    mp = int(cfg.mp_physics)
    if mp != 0 and mp not in GENERIC_FINISH_MICROPHYSICS:
        return (f"use_theta_m = 1 with mp_physics = {mp}: the theta_m finish "
                "(module_big_step_utilities_em.F:5735) is wired for "
                f"mp_physics in {sorted(GENERIC_FINISH_MICROPHYSICS)} only")
    if int(getattr(cfg, "km_opt", 0) or 0) == 5:
        return ("use_theta_m = 1 with km_opt = 5 (SMS-3DTKE) is not wired: "
                "WRF's km_opt = 5 path then forms th_phy as the dry theta of "
                "the moist carrier and adds the moist qfac and "
                "1.61*theta*QFX terms to its implicit heat solve "
                "(module_diffusion_em.F:8090-8116), and woof's km_opt = 5 "
                "operands and its WRF v4.7.1 oracle (tools/"
                "sms3dtke_wrf471_oracle) cover use_theta_m = 0 only")
    # Every cumulus scheme woof runs is admitted: KF, MSKF, Tiedtke and New
    # SAS read no theta forcing; GF, G3 and New Tiedtke read phy_prep_part2's
    # dry-theta RTHFTEN (convert_advective_forcing).  diff_opt = 2 without
    # a PBL takes vertical_diffusion_2's theta_m surface flux
    # (kernels/smag2d_theta_m.cu).
    return None


def _launch(state, func, args, n):
    from woof.core.kernels import get_kernel
    get_kernel("theta_m", func)(((n + _TPB - 1) // _TPB,), (_TPB,), args)


def _thb_args(state):
    thb = state.thb
    return thb, np.int32(1 if thb.ndim == 3 else 0)


def _numpy_state(state) -> bool:
    return isinstance(state.thp, np.ndarray)


def _base(state, xp):
    thb = xp.asarray(state.thb, dtype=np.float32)
    return thb if thb.ndim == 3 else thb[:, None, None]


def convert_initial_state(state) -> None:
    """real.exe / ideal.exe's last statement: ``t_2 = (t_2 + T0)*(1. +
    (R_v/R_d)*qv) - T0`` (module_initialize_real.F:4923-4931), then mark the
    state as holding theta_m.  A dry state (no vapour) is left as it is, as
    WRF's ``P_QV >= PARAM_FIRST_SCALAR`` test leaves it."""
    if state_theta_m(state):
        raise RuntimeError("the state already holds theta_m")
    if state.qv is None:
        return
    nz, ny, nx = state.thp.shape
    if _numpy_state(state):
        f = np.float32
        t0 = _base(state, np)
        qvf = f(1.0) + f(c.RVOVRD) * state.qv.astype(np.float32)
        state.thp[...] = (state.thp + t0) * qvf - t0
    else:
        thb, base3d = _thb_args(state)
        _launch(state, "theta_m_from_dry",
                (state.thp, state.qv, thb, base3d, np.int32(nz),
                 np.int32(ny * nx)), nz * ny * nx)
    state.theta_m = True


def dry_theta(state, out=None):
    """WRF's dry theta perturbation ``th_phy - T0`` from the state's theta_m
    (phy_prep :4830, update_phys_fields :1252), into ``out`` (a fresh
    ``theta_m_dry`` scratch by default).  For a state holding dry theta it
    is ``thp`` itself."""
    if not state_theta_m(state):
        return state.thp
    nz, ny, nx = state.thp.shape
    if _numpy_state(state):
        f = np.float32
        t0 = _base(state, np)
        qvf = f(1.0) + f(c.RVOVRD) * state.qv.astype(np.float32)
        result = (state.thp + t0) / qvf - t0
        if out is None:
            return result.astype(np.float32)
        out[...] = result
        return out
    if out is None:
        out = state.scratch((nz, ny, nx), DRY_SLOT)
    thb, base3d = _thb_args(state)
    _launch(state, "theta_dry_from_moist",
            (state.thp, state.qv, thb, base3d, np.int32(nz),
             np.int32(ny * nx), out), nz * ny * nx)
    return out


@contextmanager
def dry_view(state):
    """``state.thp`` reads as dry theta for the duration: the post-step
    diagnostics WRF's diagnostics driver runs on ``th_phy_m_t0``
    (update_phys_fields, module_diagnostics_driver.F:1252).  The block must
    only READ theta; the theta_m array is rebound, never overwritten."""
    if not state_theta_m(state):
        yield
        return
    moist = state.thp
    dry = dry_theta(state)
    state.thp = dry
    state.theta_m = False
    _WINDOW[state] = moist
    try:
        yield
    finally:
        _WINDOW.pop(state, None)
        state.thp = moist
        state.theta_m = True


def check_step(state, cfg) -> None:
    """The dycore's gate: the state's theta and the run's must agree."""
    wanted = run_theta_m(cfg)
    held = state_theta_m(state)
    if wanted and not held:
        raise RuntimeError(
            "use_theta_m = 1 but this state's theta is DRY: the door that "
            "built it does not convert to theta_m (module_initialize_real.F:"
            "4923).  Doors that convert: woof.ingest.real.initialize_real, "
            "woof.core.moist.init_moist_balanced and restarts of a theta_m "
            "run; set use_theta_m = 0 for any other")
    if held and not wanted:
        raise RuntimeError(
            "this state holds theta_m but the run integrates dry theta "
            "(use_theta_m = 0)")
    if wanted:
        for attribute, why in STEP_REFUSALS.items():
            if getattr(state, attribute, None) is not None:
                raise RuntimeError(why)


#: What a use_theta_m = 1 step still refuses, by the state attribute that
#: attaches it.  Admitted (no entry): stochastic perturbations -- WRF's
#: update_stoch_ten (SKEBS) and perturb_physics_tend (SPPT) act on the dry
#: physics tendencies before conv_t_tendf_to_moist (module_first_rk_step_
#: part2.F:839-1009), which is where woof's binding acts (dycore: after_
#: nonmicrophysics, then convert_physics_tendencies) -- and chemistry, whose
#: chem_prep reads grid%t_2 itself (chem/module_chem_utilities.F:80;
#: woof.core.chem_prep reads :func:`prognostic_theta`).
STEP_REFUSALS = {
    "_iau_forcing": (
        "use_theta_m = 1 with an IAU analysis increment is not wired: the "
        "data assimilation (woof.da: the observation operators, the analysis "
        "door, the field rules and the increment itself) reads and writes "
        "thp as dry theta, and WRF v4.7.1 has no IAU to follow"),
    "radar_tten_forcing": RADAR_REFUSAL,
    "_streamed_domain": (
        "use_theta_m = 1 on a streamed domain is not wired: the tile sweep "
        "(woof.core.streaming) steps tiles without the theta_m physics and "
        "microphysics windows and the phy_prep_part2 conversion"),
}


@contextmanager
def physics_window(state, cfg):
    """phy_prep's dry theta for the non-microphysics physics: ``thp`` holds
    ``th_phy - T0`` for the duration and the exact theta_m words return
    afterwards (they are the step-start ``t_1`` the tendency conversion
    reads, kept in :data:`STASH_SLOT`)."""
    if not state_theta_m(state):
        yield
        return
    stash = state.scratch(state.thp.shape, STASH_SLOT)
    stash[...] = state.thp
    dry_theta(state, out=state.thp)
    state.theta_m = False
    _WINDOW[state] = stash
    try:
        yield
    finally:
        _WINDOW.pop(state, None)
        state.thp[...] = stash
        state.theta_m = True


def convert_physics_tendencies(state, cfg, tendencies):
    """conv_t_tendf_to_moist on the step's held physics theta tendency.

    Returns a shallow copy of ``tendencies`` whose ``rtheta`` is the theta_m
    tendency (a scratch buffer): the driver's own held arrays stay dry, as
    WRF's RTHRATEN/RTHBLTEN/RTHCUTEN do, and are converted again each step
    with that step's vapour (moist_old) and theta_m (t_1).
    """
    if tendencies is None or not state_theta_m(state):
        return tendencies
    from woof.wrf_exact import ENABLED as WRF_EXACT
    rt = tendencies.rtheta
    nz, ny, nx = rt.shape
    out = state.scratch((nz, ny, nx), RTHETA_SLOT)
    stash = state.existing_scratch(STASH_SLOT)
    if stash is None:
        raise RuntimeError("conv_t_tendf_to_moist needs the step-start theta_m")
    thb, base3d = _thb_args(state)
    divide = bool(state.has_msf and not WRF_EXACT)
    msft = state.msft if divide else rt
    _launch(state, "conv_t_tendf_to_moist",
            (rt, tendencies.rqv, state.qv, stash, thb, base3d, msft,
             np.int32(divide), np.int32(nz), np.int32(ny * nx), out),
            nz * ny * nx)
    converted = copy.copy(tendencies)
    converted.rtheta = out
    return converted


@contextmanager
def microphysics_window(state, cfg):
    """moist_physics_prep_em / moist_physics_finish_em around the scheme.

    On entry ``thp`` takes the dry theta the scheme reads (prep :5505), the
    pre-scheme vapour is kept as ``qv_diabatic`` (:5539) and the pre-scheme
    full dry theta as h_diabatic (:5523, :data:`H_PRE_SLOT`, the sum every
    adapter's ``save_pre_mp_theta`` forms).  The scheme runs exactly as it
    runs on dry theta, dry finish included; its post-call full dry theta
    reaches :func:`finish_microphysics` (through
    :func:`woof.core.microphysics.moist_physics_finish`, or right after a
    fused finish), which keeps it.  On exit -- after the scheme, the
    specified-zone ring restore and ``mp_zero_out``, as WRF's
    moist_physics_finish_em follows microphysics_zero_out (solve_em.F:
    4018-4091) -- the theta_m finish (:5735-5741) writes ``thp`` (theta_m')
    and ``h_diabatic`` from h_diabatic, qv_diabatic, the scheme's th_phy and
    the post-call vapour, overwriting whatever the dry finish left.  A scheme
    that never hands its theta over is refused; the specified/nested ring the
    clipped WRF tiles never touch keeps its theta_m words and h_diabatic 0.
    """
    if not state_theta_m(state):
        yield
        return
    if getattr(state, "radar_tten_forcing", None) is not None:
        raise RuntimeError(RADAR_REFUSAL)
    shape = state.thp.shape
    stash = state.scratch(shape, STASH_SLOT)
    stash[...] = state.thp
    state.scratch(shape, QV_DIABATIC_SLOT)[...] = state.qv
    dry_theta(state, out=state.thp)
    _hold_pre_microphysics_theta(state)
    state.theta_m = False
    _WINDOW[state] = stash
    _FINISH_STATE[state] = ("armed",)
    captured = None
    unheated = False
    try:
        yield
        mark = _FINISH_STATE.get(state)
        captured = mark if mark is not None and mark[0] == "captured" else None
        unheated = mark is not None and mark[0] == "unheated"
    finally:
        _FINISH_STATE.pop(state, None)
        _WINDOW.pop(state, None)
        state.theta_m = True
        if captured is None:
            state.thp[...] = stash
    if captured is None and not unheated:
        raise RuntimeError(
            f"mp_physics = {cfg.mp_physics} did not hand its post-call theta "
            "to woof.core.theta_m.finish_microphysics, so its theta_m finish "
            "(module_big_step_utilities_em.F:5735) never ran")
    if captured is not None:
        _, lim, dt = captured
        _finish_theta_m(state, lim, dt)
    from woof.core.microphysics import _ring_guard_slices
    slices = _ring_guard_slices(state, cfg)
    if slices is not None:
        for slc in slices:
            state.thp[slc] = stash[slc]
            state.h_diabatic[slc] = 0.0


def _hold_pre_microphysics_theta(state) -> None:
    """moist_physics_prep_em's h_diabatic = th_phy (:5523), as the sum
    ``thb + thp`` every adapter's ``save_pre_mp_theta`` forms on the dry
    theta the window installed: thp = th_phy - thb is exact (the two are
    within a factor of two), so the sum is th_phy's own word."""
    h = state.scratch(state.thp.shape, H_PRE_SLOT)
    if _numpy_state(state):
        h[...] = _base(state, np) + state.thp
        return
    import cupy as cp
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    cp.add(thb, state.thp, out=h)


def finish_microphysics(state, cfg, th_phy, dt) -> bool:
    """Take the scheme's post-call full dry theta for moist_physics_finish_em's
    ``use_theta_m = 1`` branch; True when the state is inside
    :func:`microphysics_window`, which runs the finish on exit.

    ``th_phy`` is copied, so a caller may reuse its buffer.  With
    ``no_mp_heating = 1`` WRF leaves t_new and zeroes h_diabatic (:5770-5776):
    the window restores theta_m and h_diabatic is zeroed here.
    """
    mark = _FINISH_STATE.get(state)
    if mark is None:
        return False
    if mark[0] != "armed":
        raise RuntimeError("the microphysics finish ran twice in one window")
    if int(cfg.no_mp_heating) != 0:
        state.h_diabatic[...] = 0.0
        _FINISH_STATE[state] = ("unheated",)
        return True
    from woof.core.state import DTYPE
    state.scratch(state.thp.shape, TH_PHY_SLOT)[...] = th_phy
    # WRF multiplies the REAL mp_tend_lim by the REAL dt (:5706-5707).
    lim = DTYPE(DTYPE(cfg.mp_tend_lim) * DTYPE(dt))
    _FINISH_STATE[state] = ("captured", lim, DTYPE(dt))
    return True


def _finish_theta_m(state, lim, dt) -> None:
    """moist_physics_finish_em :5735-5741 over the whole field."""
    nz, ny, nx = state.thp.shape
    h = state.h_diabatic
    h[...] = state.scratch((nz, ny, nx), H_PRE_SLOT)
    th_phy = state.scratch((nz, ny, nx), TH_PHY_SLOT)
    qv_diabatic = state.scratch((nz, ny, nx), QV_DIABATIC_SLOT)
    dry = state.scratch((nz, ny, nx), DRY_SLOT)
    thb, base3d = _thb_args(state)
    _launch(state, "moist_physics_finish_theta_m",
            (th_phy, h, qv_diabatic, state.qv, thb, base3d,
             np.float32(lim), np.float32(dt), np.int32(nz),
             np.int32(ny * nx), state.thp, dry),
            nz * ny * nx)


#: The window's theta_m' while ``thp`` reads as dry theta (state -> array).
_WINDOW: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def prognostic_theta(state):
    """The state's prognostic theta' -- ``grid%t_2`` -- wherever the caller
    stands: theta_m' (the window's stash) inside a physics or microphysics
    window or a :func:`dry_view`, else ``thp``.  For code transcribing a WRF
    routine that reads ``t_2`` itself rather than phy_prep's th_phy
    (WRF-Chem's chem_prep, chem/module_chem_utilities.F:80)."""
    try:
        held = _WINDOW.get(state)
    except TypeError:
        # An object that cannot be weakly referenced (a bare namespace
        # standing in for a state) is never inside a window.
        held = None
    return state.thp if held is None else held


# ---------------------------------------------------------------------------
# phy_prep_part2's advective theta forcing (module_big_step_utilities_em.F:
# 5309-5333)
# ---------------------------------------------------------------------------

def forcing_already_decoupled(cfg) -> bool:
    """Strict Grell-Freitas keeps WRF's coupled RTHFTEN/RQVFTEN between the
    RK loop and the scheme and decouples them where the scheme reads them
    (woof.core.gf).  Under use_theta_m = 1 :func:`convert_advective_forcing`
    makes phy_prep_part2's decoupling itself, before its conversion, as WRF
    does, so the scheme must not divide a second time."""
    from woof.wrf_exact import ENABLED as WRF_EXACT
    return bool(WRF_EXACT) and int(cfg.cu_physics) == 3 and run_theta_m(cfg)


def convert_advective_forcing(state, cfg) -> None:
    """phy_prep_part2's ``use_theta_m == 1`` RTHFTEN block (:5323-5332) for
    the cumulus schemes that read it (GF, G3, New Tiedtke):

      th_phy  = (t_new + t0)/(1. + (R_v/R_d)*qv)
      rthften = th_phy/(t_new+t0)*(rthften - (R_v/R_d)*th_phy*rqvften)

    The advective forcing the dycore exported is the transport of the
    prognostic, theta_m; the scheme reads the dry-theta forcing.  WRF calls
    phy_prep_part2 after the RK loop and before moist_physics_prep_em
    (solve_em.F:3585), on that instant's t_2 and qv: both epilogues call this
    right before the microphysics.  The held arrays are converted in place,
    as WRF's grid%rthften is.  Strict Grell-Freitas's coupled pair takes
    phy_prep_part2's decoupling by c1h*muts + c2h first
    (:5299-5321, :func:`forcing_already_decoupled`); every other route
    exports the pair already decoupled (woof.core.dycore.
    capture_advective_theta_forcing).
    """
    if not state_theta_m(state) or int(cfg.cu_physics) not in RTHFTEN_CUMULUS:
        return
    rth = getattr(state, "rthften", None)
    rqv = getattr(state, "rqvften", None)
    if rth is None or rqv is None:
        raise RuntimeError(
            f"cu_physics = {cfg.cu_physics} under use_theta_m = 1 holds no "
            "RTHFTEN/RQVFTEN pair to convert")
    nz, ny, nx = rth.shape
    decouple = forcing_already_decoupled(cfg)
    if decouple:
        from woof.state_serialization_contract import WRF_MUTS_SLOT
        mut = state.existing_scratch(WRF_MUTS_SLOT)
        if mut is None:
            mut = state.total_mu()
    else:
        mut = state.mup          # unread
    thb, base3d = _thb_args(state)
    from woof.core.kernels import get_kernel
    n = nz * ny * nx
    get_kernel("theta_m_cumulus", "phy_prep_part2_rthften_theta_m")(
        ((n + _TPB - 1) // _TPB,), (_TPB,),
        (rth, rqv, state.thp, state.qv, thb, base3d, state.c1h, state.c2h,
         mut, np.int32(decouple), np.int32(nz), np.int32(ny * nx)))


