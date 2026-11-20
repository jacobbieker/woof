"""One dry model step's tangent-linear and adjoint (P3a item 8), chained
from the ported operators in woof's strict order (GPUWM_WRF_EXACT=1,
woof/core/dycore.py ``_step_body`` and ``_wrf_step_epilogue``).

The step works on one dict of device arrays keyed by the strict primal's
names (:func:`run_state` builds it from a ``DomainState``), with the base
state of every operator held 3-D.  :func:`step_tl` advances the base state
(binary64 NL, or binary32 reproducing the primal's words) and maps a
perturbation; with a ``tape`` it records each operator's written arrays
before the operator runs, and :func:`step_ad` consumes that tape in
reverse (recompute, then tape: the step's own NL is rerun from its
start).

Covered: periodic axes, no map factors, Coriolis and curvature, dry, no physics or
mixing, ``rk_ord = 3``, ``h_*_adv_order`` 3-6, a 3-D base state (terrain)
and the w_damp limiter.
Everything else is refused by :func:`step_geometry`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acoustic import (CW_ACTIVE, CW_PASSIVE, CW_STATE, MT_ALIAS, MT_ACTIVE,
                       MT_PASSIVE, MT_STATE, SF_ACTIVE, SF_PASSIVE, SF_STATE,
                       SI_ACTIVE, SI_PASSIVE, SI_STATE, UV_TRAJECTORY,
                       W_STATE, W_TRAJECTORY, CWGeometry, MTGeometry,
                       SSGeometry, UVGeometry, WGeometry, acoustic_substep_ad,
                       acoustic_substep_tl, calc_coef_w_ad, calc_coef_w_tl,
                       small_step_finish_ad, small_step_finish_tl,
                       small_step_init_ad, small_step_init_tl)
from .advect import ADV_ACTIVE, ADV_PASSIVE, ADV_STATE, AdvGeometry, adv_ad, adv_tl
from .bigstep import (BUOY_ACTIVE, BUOY_PASSIVE, BUOY_STATE, CC_ACTIVE,
                      CC_PASSIVE, CC_STATE, GEO_ACTIVE, GEO_PASSIVE, GEO_STATE,
                      PGF_ACTIVE, PGF_PASSIVE, PGF_STATE, WD_ACTIVE, WD_PASSIVE,
                      WD_STATE, BuoyGeometry, CCGeometry, GeoGeometry,
                      PGFGeometry, WDampGeometry, buoy_ad, buoy_tl, cc_ad, cc_tl,
                      geo_ad, geo_tl, pgf_ad, pgf_tl, wd_ad, wd_tl)
from .stage import (PRP_ACTIVE, PRP_ALIAS, PRP_PASSIVE, PRP_STATE,
                    RSP_ACTIVE, RSP_ALIAS, RSP_PASSIVE, RSP_STATE,
                    SW_ACTIVE, SW_PASSIVE, PRPGeometry, RSPGeometry,
                    SWGeometry, _view, calc_p_rho_phi_ad, calc_p_rho_phi_tl,
                    rk_step_prep_ad, rk_step_prep_tl, set_w_surface_ad,
                    set_w_surface_tl)

#: The step's prognostics (dycore._PROGNOSTICS), their time-t copies and
#: the stage tendencies zeroed at each stage start (dycore._TENDENCIES).
PROGNOSTICS = ("u", "v", "w", "thp", "php", "mup")
TENDENCIES = ("ru_t", "rv_t", "rw_t", "rth_t", "rph_t", "rmu_t")

#: What a step reads from the previous one: the prognostics, and in the
#: strict build the previous step's diagnosis (p, al, alt and p') with
#: WRF's carried muts, which rk_step 1 reads as they are.
STEP_INPUTS = PROGNOSTICS + ("p", "al", "alt", "p_pert", "muts")

#: Each slow operator's names on the step dict.
_ADV_ALIAS = {
    "scalar": {"tend": "rth_t", "q": "thp", "rw": "ww_ref", "msf": "msft", "rdz": "rdnw"},
    "u": {"tend": "ru_t", "q": "u", "rw": "ww_ref", "msf": "msfu", "rdz": "rdnw"},
    "v": {"tend": "rv_t", "q": "v", "rw": "ww_ref", "msf": "msfv", "rdz": "rdnw"},
    "w": {"tend": "rw_t", "q": "w", "rw": "ww_ref", "msf": "msft", "rdz": "rdn"},
}
_PGF_ALIAS = {"p": "p_pert", "mub": "mub2d"}
_BUOY_ALIAS = {"p": "p_pert"}
_GEO_ALIAS = {"ww": "ww_ref", "mub": "mub2d"}
_WD_ALIAS = {"ww": "ww_ref", "mub": "mub2d"}

#: Arrays held as base state, made 3-D when the domain's are columns.
_BASE3D = ("phb", "pb", "alb", "thb")


@dataclass(frozen=True)
class StageSpec:
    """One RK stage: its operators' geometries.  ``substeps`` holds the
    first substep's (smdiv 0, as the strict launcher's first call) and the
    later ones'."""
    rk_step: int
    nsub: int
    diagnose: object          # PRPGeometry, or None on a reused rk_step 1
    rsp: RSPGeometry
    adv: tuple                # AdvGeometry per kind: scalar, u, v, w
    pgf: PGFGeometry
    buoy: BuoyGeometry
    geo: GeoGeometry
    wd: object                # WDampGeometry or None
    cc: object                # CCGeometry on rotational grids, else None
    ss: SSGeometry
    cw: CWGeometry
    first: tuple              # (UVGeometry, MTGeometry, WGeometry)
    later: tuple


@dataclass(frozen=True)
class StepGeometry:
    nz: int
    ny: int
    nx: int
    stages: tuple
    end_prp: PRPGeometry
    sw: SWGeometry


def _refuse(cfg, state) -> None:
    from woof.core.advection import scalar_horizontal_order
    checks = {
        "lateral boundaries other than periodic": bool(
            cfg.open_x or cfg.open_y or cfg.specified or getattr(cfg, "nested", False)),
        "map factors": bool(state.has_msf),
        "moisture": state.qv is not None,   # the primal's test (dycore._step_body)
        "microphysics": bool(cfg.mp_physics),
        "rk_ord 2": getattr(cfg, "rk_ord", 3) != 3,
        "emdiv": float(cfg.emdiv) > 0.0,
        "damp_opt 3": cfg.damp_opt == 3,
        "diff_6th_opt": bool(cfg.diff_6th_opt),
        "zadvect_implicit": bool(getattr(cfg, "zadvect_implicit", 0)),
        "momentum_adv_opt 3": getattr(cfg, "momentum_adv_opt", 1) == 3,
        "h_sca_adv_order outside 3-6": scalar_horizontal_order(cfg) not in (3, 4, 5, 6),
        "h_mom_adv_order outside 3-6": int(cfg.h_mom_adv_order) not in (3, 4, 5, 6),
        # The primal's PGF takes another arithmetic path on a 1-D base, so
        # the float step would not reproduce its words.
        "1-D base state (no terrain)": np.ndim(state.phb) != 3,
    }
    from woof.config import radiation_scheme_ids
    from woof.core.dycore import wrf_mixing_package_active
    from woof.core.physics_inventory import physics_enabled
    checks.update({
        "mixing package": bool(wrf_mixing_package_active(cfg)),
        # what else _step_body runs that the chain does not
        "physics": bool(physics_enabled(cfg)),
        "radiation selector 31 (idealized Rayleigh friction)":
            radiation_scheme_ids(cfg)[0] == 31,
        "use_theta_m": bool(cfg.use_theta_m),
        "moving frame": bool(getattr(cfg, "umove", 0.0) or getattr(cfg, "vmove", 0.0)
                             or getattr(state, "_moving_frame_velocity", None) is not None),
        "obstacle_opt": bool(cfg.obstacle_opt),
        "polar": bool(cfg.polar),
        "lightning_option": bool(cfg.lightning_option),
        "do_avgflx_em": bool(cfg.do_avgflx_em),
        "IAU forcing": getattr(state, "_iau_forcing", None) is not None,
        "spectral nudging": getattr(state, "_spectral_nudging_graph", None) is not None,
        "odd time_step_sound": int(cfg.time_step_sound) % 2 != 0,
    })
    bad = [k for k, v in checks.items() if v]
    if bad:
        raise NotImplementedError("the P3a step TL/AD does not cover: " + ", ".join(bad))


def step_geometries(state, cfg, nstep: int) -> tuple:
    """Geometries for ``nstep`` chained strict steps from ``state``.  The
    first follows the state (a cold state, without WRF's carried muts,
    diagnoses on rk_step 1); every later step follows a strict step, so it
    reuses that step's diagnosis and carried muts."""
    first = step_geometry(state, cfg)
    later = step_geometry(state, cfg, reuse_first=True, carried=True)
    return (first,) + (later,) * (nstep - 1)


def step_geometry(state, cfg, *, reuse_first: bool | None = None,
                  carried: bool | None = None) -> StepGeometry:
    """The operators' geometries for one strict step of ``state`` under
    ``cfg``, with each scalar rounded as its primal launcher rounds it.
    ``reuse_first``: rk_step 1 reads the previous step's diagnosis;
    ``carried``: the state holds WRF's muts.  Both default to what
    ``state`` holds now; chained steps take :func:`step_geometries`."""
    from woof.core import dycore as D
    from woof.core.acoustic import damp_magnitude
    from woof.core.advection import scalar_horizontal_order, vertical_orders
    _refuse(cfg, state)
    f32 = np.float32
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    dx, dy = float(cfg.dx), float(cfg.dy)
    rdx, rdy = float(f32(1.0 / dx)), float(f32(1.0 / dy))
    cf = dict(cf1=float(state.cf1), cf2=float(state.cf2), cf3=float(state.cf3))
    cfn, cfn1 = float(state.cfn), float(state.cfn1)
    top_lid = int(bool(cfg.top_lid))
    hypso = int(cfg.hypsometric_opt)
    p_top = float(f32(0.0 if state.p_top is None else state.p_top))
    if reuse_first is None:
        reuse_first = bool(D._wrf_reuse_diagnostics(state))
    if carried is None:
        carried = D._wrf_muts(state) is not None
    vsca, vmom = vertical_orders(cfg)
    hsca, hmom = scalar_horizontal_order(cfg), int(cfg.h_mom_adv_order)
    adv = tuple(AdvGeometry(kind, nz, ny, nx, rdx, rdy, vorder=v, horder=h)
                for kind, v, h in (("scalar", vsca, hsca), ("u", vmom, hmom),
                                   ("v", vmom, hmom), ("w", vsca, hsca)))
    pgf = PGFGeometry(nz, ny, nx, rdx, rdy, cfn=cfn, cfn1=cfn1, top_lid=top_lid, **cf)
    buoy = BuoyGeometry(nz, ny, nx)
    geo = GeoGeometry(nz, ny, nx, quarter_rdx=float(f32(0.25 / dx)),
                      quarter_rdy=float(f32(0.25 / dy)), cfn=cfn, cfn1=cfn1, order=hsca)
    wd = None
    if cfg.w_damping == 1:
        wd = WDampGeometry(nz, ny, nx, dt=float(f32(cfg.dt)),
                           w_damp_on=float(f32(D.w_damp_onset(cfg))),
                           w_crit_cfl=float(f32(cfg.w_crit_cfl)))
    rsp = RSPGeometry(nz, ny, nx, rdx=float(f32(1) / f32(dx)), rdy=float(f32(1) / f32(dy)))
    cc = CCGeometry(nz, ny, nx, rdx=rdx, rdy=rdy) if state.rotational else None
    stages = []
    for istage, (nsub, dtau) in enumerate(D._rk_acoustic_stages(cfg)):
        dtau = float(f32(dtau))
        prp = None
        if not (istage == 0 and reuse_first):
            prp = PRPGeometry(nz, ny, nx, hypso=hypso, carried=int(carried or istage > 0),
                              p_top=p_top)

        def sub(smdiv):
            return (UVGeometry(nz, ny, nx, rdx, rdy, dtau, smdiv=smdiv, top_lid=top_lid, **cf),
                    MTGeometry(nz, ny, nx, rdx, rdy, dtau, write_mudf=0),
                    WGeometry(nz, ny, nx, rdx, rdy, dtau, epssm=float(f32(cfg.epssm)),
                              top_lid=top_lid, dampmag=float(damp_magnitude(cfg, dtau)),
                              **cf))
        stages.append(StageSpec(
            rk_step=istage + 1, nsub=int(nsub), diagnose=prp, rsp=rsp, adv=adv, pgf=pgf,
            buoy=buoy, geo=geo, wd=wd, cc=cc,
            ss=SSGeometry(nz, ny, nx, boundary_x=0, boundary_y=0, rk_step=istage + 1),
            cw=CWGeometry(nz, ny, nx, dtau, epssm=float(f32(cfg.epssm)), top_lid=top_lid),
            first=sub(0.0), later=sub(float(f32(cfg.smdiv)))))
    return StepGeometry(
        nz, ny, nx, tuple(stages),
        end_prp=PRPGeometry(nz, ny, nx, hypso=hypso, carried=1, p_top=p_top),
        sw=SWGeometry(nz, ny, nx, half_rdx=float(f32(0.5 / dx)),
                      half_rdy=float(f32(0.5 / dy)), **cf))


def _names() -> tuple:
    """Every name the step's operators read or write."""
    def al(names, alias):
        return tuple(alias.get(n, n) for n in names)
    out = PROGNOSTICS + tuple(n + "0" for n in PROGNOSTICS) + TENDENCIES + ("muts",)
    out += al(PRP_STATE + PRP_ACTIVE + PRP_PASSIVE, PRP_ALIAS)
    out += al(RSP_STATE + RSP_ACTIVE + RSP_PASSIVE, RSP_ALIAS)
    for kind, alias in _ADV_ALIAS.items():
        out += al(ADV_STATE + ADV_ACTIVE + ADV_PASSIVE, alias)
    out += al(PGF_STATE + PGF_ACTIVE + PGF_PASSIVE, _PGF_ALIAS)
    out += al(BUOY_STATE + BUOY_ACTIVE + BUOY_PASSIVE, _BUOY_ALIAS)
    out += al(GEO_STATE + GEO_ACTIVE + GEO_PASSIVE, _GEO_ALIAS)
    out += al(WD_STATE + WD_ACTIVE + WD_PASSIVE, _WD_ALIAS)
    out += CC_STATE + CC_ACTIVE + CC_PASSIVE
    out += SI_STATE + SI_ACTIVE + SI_PASSIVE + CW_STATE + CW_ACTIVE + CW_PASSIVE
    out += SF_STATE + SF_ACTIVE + SF_PASSIVE
    out += ("u_pp", "v_pp") + UV_TRAJECTORY + W_STATE + W_TRAJECTORY
    out += al(MT_STATE + MT_ACTIVE + MT_PASSIVE, MT_ALIAS)
    out += ("w",) + SW_ACTIVE + SW_PASSIVE
    return tuple(dict.fromkeys(out))


def _shapes(nz, ny, nx) -> dict:
    m, u, v, w, c = (nz, ny, nx), (nz, ny, nx + 1), (nz, ny + 1, nx), (nz + 1, ny, nx), (ny, nx)
    return {"cqu": u, "cqv": v, "cqw": w, "mudf": c, "h_diabatic": m, "ru": u, "rv": v,
            "ww_ref": w, "mu_pp_old": c, "th_pp_old": m, "c2a": m, "a": w, "alpha": w,
            "gam": w, "muts": c, "p_pert": m, "mut": c}


def run_state(state, cfg, dtype=np.float64) -> dict:
    """The step dict of ``state``: device copies in ``dtype`` of every
    array the step's operators name, the base state 3-D, the strict
    scratch carriers under their operator names (Omega ww_ref, muts, the
    acoustic coefficients and old mass and theta), and zeros for the
    moist cq and mudf slots the dry step does not read."""
    import cupy as cp
    from woof.core import dycore as D
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    sh = _shapes(nz, ny, nx)
    scratch = {"ru": "rk_ru", "rv": "rk_rv", "ww_ref": "rk_ww",
               "mu_pp_old": "acoustic_mu_pp_old", "th_pp_old": "acoustic_th_pp_old",
               "c2a": "acoustic_c2a", "a": "acoustic_a", "alpha": "acoustic_alpha",
               "gam": "acoustic_gamma"}
    out = {}
    for n in _names():
        if n in ("cqu", "cqv", "cqw", "mudf", "h_diabatic", "mut"):
            a = cp.zeros(sh[n])
        elif n in scratch:
            a = state.scratch(sh[n], scratch[n])
        elif n == "muts":
            m = D._wrf_muts(state)
            a = m if m is not None else state.mub2d + state.mup
        elif n == "p_pert":
            a = state.p_perturbation
        else:
            a = getattr(state, n)
        a = cp.asarray(a)
        if n in _BASE3D and a.ndim == 1:
            lev = a.shape[0]
            a = cp.broadcast_to(a[:, None, None], (lev, ny, nx))
        out[n] = cp.ascontiguousarray(a.astype(dtype, copy=True))
    for n in ("fnm", "fnp"):  # the acoustic operators take nz + 1 levels
        out[n + "_w"] = cp.concatenate([out[n], cp.zeros(1, dtype=dtype)])
    return out


def _acoustic(d: dict) -> dict:
    """``d`` as the acoustic substep's operators name it (fnm and fnp on
    nz + 1 levels; the primal never reads the last)."""
    return {**d, "fnm": d["fnm_w"], "fnp": d["fnp_w"]}


def state_inputs(x: dict) -> dict:
    return {n: x[n] for n in STEP_INPUTS}


# ------------------------------------------------------------------ the step

def _op_table(spec: StageSpec):
    """The stage's operators from rk_step_prep to the slow tendencies:
    (name, tl, ad, alias, geometry, written names)."""
    ops = []
    if spec.diagnose is not None:
        ops.append(("prp", calc_p_rho_phi_tl, calc_p_rho_phi_ad, PRP_ALIAS, spec.diagnose,
                    PRP_STATE))
    ops.append(("rsp", rk_step_prep_tl, rk_step_prep_ad, RSP_ALIAS, spec.rsp, RSP_STATE))
    for g in spec.adv:
        ops.append(("adv_" + g.kind, adv_tl, adv_ad, _ADV_ALIAS[g.kind], g, ADV_STATE))
    ops.append(("pgf", pgf_tl, pgf_ad, _PGF_ALIAS, spec.pgf, PGF_STATE))
    ops.append(("buoy", buoy_tl, buoy_ad, _BUOY_ALIAS, spec.buoy, BUOY_STATE))
    ops.append(("geo", geo_tl, geo_ad, _GEO_ALIAS, spec.geo, GEO_STATE))
    if spec.wd is not None:
        ops.append(("wd", wd_tl, wd_ad, _WD_ALIAS, spec.wd, WD_STATE))
    if spec.cc is not None:
        ops.append(("mut", _mut_tl, _mut_ad, {}, None, ("mut",)))
        ops.append(("cc", cc_tl, cc_ad, {}, spec.cc, CC_STATE))
    return ops


def _mut_tl(state, pert_state, traj, pert, geom, *, nl=True):
    """coriolis_curvature's total dry mass (DomainState.total_mu)."""
    import cupy as cp
    if nl:
        cp.add(state["mub2d"], state["mup"], out=state["mut"])
    pert_state["mut"][...] = pert["mup"]


def _mut_ad(adj_state, traj, adj, geom):
    adj["mup"] += adj_state["mut"]
    adj_state["mut"][...] = 0


def _written(alias, names):
    return tuple(alias.get(n, n) for n in names)


def step_tl(x: dict, dx: dict, geom: StepGeometry, tape: list | None = None,
            observe=None) -> None:
    """One strict dry step: ``x`` advances (NL in its dtype) and ``dx``
    maps to the step's output perturbation, in place.  ``dx`` holds every
    name of ``x`` (zeros where unperturbed).  With a ``tape`` each
    operator's written arrays are appended before it runs.  ``observe``
    (a debugging aid) is called with a label and ``x`` after each part."""
    _step(x, dx, geom, tape, observe, tl=True)


def step_base(x: dict, geom: StepGeometry, tape: list | None = None,
              scratch: dict | None = None) -> dict:
    """The base state of :func:`step_tl` alone, through the base-only
    kernel build (no tangent arithmetic, loads or stores): ``x`` advances
    to the same words, and ``tape`` gets the same snapshots, so this is
    the recompute an adjoint step takes.  ``scratch`` (any dict of arrays
    shaped as ``x``'s, never read or written by the kernels) stands in
    for the perturbation arguments; one is allocated when not given, and
    returned for reuse."""
    import cupy as cp
    from ._mode import base_only
    if scratch is None:
        scratch = {n: cp.empty_like(v) for n, v in x.items()}
    with base_only():
        _step(x, scratch, geom, tape, None, tl=False)
    return scratch


def _step(x, dx, geom, tape, observe, *, tl):
    import cupy as cp
    seen = observe or (lambda label, d: None)

    def snap(names):
        if tape is not None:
            tape.append({n: x[n].copy() for n in names})

    def copy_names(dst, src):
        snap(dst)
        for d, s in zip(dst, src):
            x[d][...] = x[s]
            if tl:
                dx[d][...] = dx[s]

    copy_names(tuple(n + "0" for n in PROGNOSTICS), PROGNOSTICS)   # _save_time_t
    for spec in geom.stages:
        snap(TENDENCIES)                                          # zero_tendencies
        for n in TENDENCIES:
            x[n][...] = 0
            if tl:
                dx[n][...] = 0
        for name, op_tl, _ad, alias, g, state_names in _op_table(spec):
            snap(_written(alias, state_names))
            s, p = _view(x, alias), _view(dx, alias)
            op_tl(s, p, s, p, g, nl=True)
            seen(name, x)
        snap(SI_STATE)
        small_step_init_tl(x, dx, x, dx, spec.ss, nl=True)
        seen("si", x)
        snap(("muts",))
        src = "mup" if spec.rk_step == 1 else "mup0"
        cp.add(x["mub2d"], x[src], out=x["muts"])
        if tl:
            dx["muts"][...] = dx[src]
        snap(CW_STATE)
        calc_coef_w_tl(x, dx, x, dx, spec.cw, nl=True)
        seen("cw", x)
        for i in range(spec.nsub):
            acoustic_substep_tl(_acoustic(x), _acoustic(dx),
                                spec.first if i == 0 else spec.later, tape)
            seen("substep", x)
        snap(("muts",))
        cp.add(cp.add(x["mub2d"], x["mup"]), x["mu_pp"], out=x["muts"])
        if tl:
            cp.add(dx["mup"], dx["mu_pp"], out=dx["muts"])
        snap(SF_STATE)
        small_step_finish_tl(x, dx, x, dx, spec.ss, nl=True)
        seen("sf", x)
    snap(_written(PRP_ALIAS, PRP_STATE))                          # epilogue diagnosis
    s, p = _view(x, PRP_ALIAS), _view(dx, PRP_ALIAS)
    calc_p_rho_phi_tl(s, p, s, p, geom.end_prp, nl=True)
    snap(("w",))
    set_w_surface_tl(x, dx, x, dx, geom.sw, nl=True)
    snap(("u", "v"))                                              # close_periodic_alias
    for d in ((x, dx) if tl else (x,)):
        d["u"][..., -1] = d["u"][..., 0]
        d["v"][..., -1, :] = d["v"][..., 0, :]


def step_ad(ax: dict, x: dict, tape: list, geom: StepGeometry) -> dict:
    """Adjoint of :func:`step_tl`: ``ax`` (every name of the step dict)
    maps from the output adjoint to the input adjoint in place; ``x`` is
    the base after the step and ``tape`` what its TL appended (consumed).
    Returns the base at the step start."""
    cur = dict(x)

    def back():
        nonlocal cur
        cur = {**cur, **tape.pop()}
        return cur

    back()                                                        # periodic alias
    ax["u"][..., 0] += ax["u"][..., -1]
    ax["u"][..., -1] = 0
    ax["v"][..., 0, :] += ax["v"][..., -1, :]
    ax["v"][..., -1, :] = 0
    base = back()
    set_w_surface_ad(ax, base, ax, geom.sw)
    base = back()
    a = _view(ax, PRP_ALIAS)
    calc_p_rho_phi_ad(a, _view(base, PRP_ALIAS), a, geom.end_prp)
    for spec in reversed(geom.stages):
        base = back()                                             # sf
        small_step_finish_ad(ax, {n: base[n] for n in SF_STATE}, base, ax, spec.ss)
        back()                                                    # captured muts
        ax["mup"] += ax["muts"]
        ax["mu_pp"] += ax["muts"]
        ax["muts"][...] = 0
        for i in reversed(range(spec.nsub)):
            cur = acoustic_substep_ad(_acoustic(ax), _acoustic(cur), tape,
                                      spec.first if i == 0 else spec.later)
            cur.update(fnm=x["fnm"], fnp=x["fnp"])
        base = back()                                             # cw
        calc_coef_w_ad(ax, base, ax, spec.cw)
        back()                                                    # small_step_prep's muts
        src = "mup" if spec.rk_step == 1 else "mup0"
        ax[src] += ax["muts"]
        ax["muts"][...] = 0
        base = back()                                             # si
        small_step_init_ad(ax, base, ax, spec.ss)
        for _, _, ad, alias, g, _ in reversed(_op_table(spec)):
            base = back()
            a = _view(ax, alias)
            ad(a, _view(base, alias), a, g)
        back()                                                    # zero_tendencies
        for n in TENDENCIES:
            ax[n][...] = 0
    back()                                                        # _save_time_t
    for n in PROGNOSTICS:
        ax[n] += ax[n + "0"]
        ax[n + "0"][...] = 0
    assert not tape, len(tape)
    return cur
