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
SCRATCH_SLOTS = (STASH_SLOT, DRY_SLOT, RTHETA_SLOT, QV_DIABATIC_SLOT)

#: The microphysics schemes whose finish goes through the shared
#: :func:`woof.core.microphysics.moist_physics_finish` (and so through
#: :func:`finish_microphysics`).  Every other scheme finishes theta inside
#: its own fused kernel with the dry branch, and is refused by name.
GENERIC_FINISH_MICROPHYSICS = frozenset({1})

#: WRF phy_prep_part2's RTHFTEN branch (module_big_step_utilities_em.F:
#: 5309-5333): the cumulus schemes that read the advective theta forcing,
#: which WRF converts to dry theta under use_theta_m = 1.
RTHFTEN_CUMULUS = {3: "Grell-Freitas", 5: "Grell 3D", 16: "New Tiedtke",
                   93: "Grell-Devenyi"}

_FINISH_STATE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


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
    if mp not in (0,) and mp not in GENERIC_FINISH_MICROPHYSICS:
        return (f"use_theta_m = 1 with mp_physics = {mp}: that scheme finishes "
                "theta in its own fused kernel with moist_physics_finish_em's "
                "dry (use_theta_m = 0) branch; the theta_m finish "
                "(module_big_step_utilities_em.F:5735) is wired for "
                f"mp_physics in {sorted(GENERIC_FINISH_MICROPHYSICS)} only")
    cu = int(cfg.cu_physics)
    if cu in RTHFTEN_CUMULUS:
        return (f"use_theta_m = 1 with cu_physics = {cu} ({RTHFTEN_CUMULUS[cu]}): "
                "phy_prep_part2's dry-theta RTHFTEN conversion "
                "(module_big_step_utilities_em.F:5323) is not wired")
    isfflx = int(cfg.isfflx)
    prescribed_heat = isfflx in (0, 2) and float(cfg.tke_heat_flux) != 0.0
    driver_fluxes = (isfflx in (1, 2)
                     and int(getattr(cfg, "sf_sfclay_physics", 0)) != 0)
    if (int(cfg.diff_opt) == 2 and int(cfg.bl_pbl_physics) == 0
            and (prescribed_heat or driver_fluxes)):
        return ("use_theta_m = 1 with diff_opt = 2, bl_pbl_physics = 0 and a "
                "nonzero surface heat or moisture flux: vertical_diffusion_2's "
                "theta_m surface flux (module_diffusion_em.F:4294-4318) is not "
                "wired; admitted are the configurations where both branches "
                "add zero (no surface layer and no prescribed tke_heat_flux)")
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
    try:
        yield
    finally:
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
        for attribute, what in (("_iau_forcing", "an IAU analysis increment"),
                                ("radar_tten_forcing", "a radar latent-heating forcing"),
                                ("chem", "chemistry (its preparation reads dry theta)"),
                                ("_streamed_domain", "a streamed domain"),
                                ("_ensemble_stochastic", "stochastic perturbations")):
            if getattr(state, attribute, None) is not None:
                raise RuntimeError(
                    f"use_theta_m = 1 with {what} is not wired: the theta it "
                    "adds or reads is dry theta")


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
    try:
        yield
    finally:
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

    On entry ``thp`` takes the dry theta the scheme reads (prep :5505) and
    the pre-scheme vapour is kept as ``qv_diabatic`` (:5539); the scheme's
    call of :func:`woof.core.microphysics.moist_physics_finish` lands in
    :func:`finish_microphysics`, which writes theta_m back.  A scheme that
    never calls it is refused, and the specified/nested ring the clipped
    WRF tiles never touch gets its theta_m words back.
    """
    if not state_theta_m(state):
        yield
        return
    if getattr(state, "radar_tten_forcing", None) is not None:
        raise RuntimeError("use_theta_m = 1 with a radar latent-heating "
                           "forcing is not wired")
    stash = state.scratch(state.thp.shape, STASH_SLOT)
    stash[...] = state.thp
    state.scratch(state.thp.shape, QV_DIABATIC_SLOT)[...] = state.qv
    dry_theta(state, out=state.thp)
    state.theta_m = False
    _FINISH_STATE[state] = "armed"
    finished = False
    try:
        yield
        finished = _FINISH_STATE.get(state) == "done"
    finally:
        _FINISH_STATE.pop(state, None)
        state.theta_m = True
        if not finished:
            state.thp[...] = stash
    if not finished and int(cfg.no_mp_heating) == 0:
        raise RuntimeError(
            f"mp_physics = {cfg.mp_physics} did not finish through "
            "moist_physics_finish, so its theta_m finish "
            "(module_big_step_utilities_em.F:5735) never ran")
    from woof.core.microphysics import _ring_guard_slices
    slices = _ring_guard_slices(state, cfg)
    if slices is not None:
        for slc in slices:
            state.thp[slc] = stash[slc]


def finish_microphysics(state, cfg, th_phy, dt) -> bool:
    """moist_physics_finish_em's ``use_theta_m = 1`` branch; True when it
    ran (the state is inside :func:`microphysics_window`)."""
    if _FINISH_STATE.get(state) != "armed":
        return False
    from woof.core.state import DTYPE
    nz, ny, nx = state.thp.shape
    if int(cfg.no_mp_heating) == 0:
        lim = DTYPE(DTYPE(cfg.mp_tend_lim) * DTYPE(dt))
        thb, base3d = _thb_args(state)
        _launch(state, "moist_physics_finish_theta_m",
                (th_phy, state.h_diabatic,
                 state.scratch((nz, ny, nx), QV_DIABATIC_SLOT), state.qv,
                 thb, base3d, np.float32(lim), np.float32(dt),
                 np.int32(nz), np.int32(ny * nx), state.thp,
                 state.scratch((nz, ny, nx), DRY_SLOT)),
                nz * ny * nx)
        _FINISH_STATE[state] = "done"
    else:
        state.h_diabatic[...] = 0.0
    return True
