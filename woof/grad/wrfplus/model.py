"""One model step's tangent-linear and adjoint (P3a item 8), chained
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

Covered: periodic axes or specified lateral boundaries (WRF's
relax_bdy_dry folded into rk_addtend_dry, spec_bdy_dry, spec_bdyupdate_ph
with zero_grad_bdy, spec_bdy_final and the carried mu halo, with the
boundary tables as active inputs), map factors, Coriolis and curvature,
dry or moist without microphysics (qv, qc and qr: calc_cq, the moist
pressure gradient, buoyancy, diagnosis and acoustic cq, sumflux and the
non-positive-definite scalar update, and on a specified domain qv's
held boundary tendency and spec_bdy_final with flow_dep_bdy for qc and
qr), on a dry state WRF's diff_opt 1 / km_opt 4 horizontal mixing (the
deformation, smag2d_km and the u, v, w and theta rows of the time-t
fields into the held tendencies, coupled to the map factors and added on
every stage), no physics, ``rk_ord = 3``, ``h_*_adv_order`` 3-6, a 3-D
base state (terrain) and the w_damp limiter.
Everything else is refused by :func:`step_geometry`.
"""

from __future__ import annotations

import dataclasses
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
from .lbc import (LB_PASSIVE, LB_TABLES, LF_PASSIVE, SP_ACTIVE, SP_PASSIVE,
                  SP_STATE, FinalGeometry, FlowGeometry, LBCGeometry,
                  SpecPhiGeometry, flow_ad, flow_tl, lbc_ad, lbc_tl, lbf_ad,
                  lbf_tl, sph_ad, sph_tl)
from .bigstep import (BUOY_ACTIVE, BUOY_PASSIVE, BUOY_STATE, CC_ACTIVE,
                      CC_PASSIVE, CC_STATE, CQ_STATE, GEO_ACTIVE, GEO_PASSIVE,
                      GEO_STATE, PGF_ACTIVE, PGF_PASSIVE, PGF_STATE, WD_ACTIVE,
                      WD_PASSIVE, WD_STATE, BuoyGeometry, CCGeometry, CQGeometry,
                      GeoGeometry, PGFGeometry, WDampGeometry, buoy_ad, buoy_tl,
                      cc_ad, cc_tl, cq_ad, cq_tl, geo_ad, geo_tl, pgf_ad, pgf_tl,
                      wd_ad, wd_tl)
from .diffusion import (DF_PASSIVE, DF_STATE, H2_PASSIVE, DFGeometry, H2Geometry,
                        HDGeometry, KCGeometry, KMGeometry, df_ad, df_tl, h2_active,
                        h2_ad, h2_passive, h2_tl, hd_ad, hd_tl, k2_ad, k2_tl, k3_ad,
                        k3_tl, km_ad, km_tl, n2_ad, n2_tl)
from .scalar import (SUMFLUX, ScalarGeometry, SumfluxGeometry, su_ad, su_tl,
                     sumflux_accumulate_ad, sumflux_accumulate_tl,
                     sumflux_finish_ad, sumflux_finish_tl, sumflux_zero_ad,
                     sumflux_zero_tl)
from .stage import (PRP_ACTIVE, PRP_ALIAS, PRP_PASSIVE, PRP_STATE,
                    RSP_ACTIVE, RSP_ALIAS, RSP_HALO, RSP_PASSIVE, RSP_STATE,
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

#: A moist state's species (moist.SPECIES with mp_physics 0): prognostics
#: a moist step also saves at time t, transports and reads from its
#: predecessor.
SPECIES = ("qv", "qc", "qr")

#: Specified lateral boundaries.  The step also carries WRF's mu halo
#: (dycore.WRF_MU_HALO_SLOT: the ring mass before spec_bdy_final, which
#: the next step's rk_step 1 calc_ww_cp reads), and takes the boundary
#: interval's tables as inputs it never writes: ``lbc_<field>_<side>``
#: (value) and ``lbc_<field>_<side>_t`` (tendency) for the dry fields,
#: coupled as woof's resident tables are (WRF's ``*_b``/``*_bt``).  Their
#: perturbation is the LBC perturbation, and their adjoint the LBC
#: gradient (WRFDA's var4d_lbc).
HALO = RSP_HALO
LBC_FIELDS = ("u", "v", "theta", "phi", "mu")
LBC_INPUTS = tuple(f"lbc_{f}_{t}" for f in LBC_FIELDS for t in LB_TABLES)
#: qv's tables, on a moist specified domain whose boundary supplies qv.
QV_INPUTS = tuple(f"lbc_qv_{t}" for t in LB_TABLES)

#: Each LBC field's tendency, state array and held buffer on the step
#: dict, and the mass its row couples with.  qv's row is the held capture
#: of moist.advance_scalars_stage on rk_step 1: the held tendency zeroed,
#: then the relaxation and specified rows from the time-t qv0 at the
#: time-t mass mup0 (apply_state_scalar_lateral_boundary's source_mup).
_LBC_TEND = {"u": "ru_t", "v": "rv_t", "theta": "rth_t", "phi": "rph_t", "mu": "rmu_t",
             "qv": "lbc_qv_held"}
_LBC_SRC = {"u": "u", "v": "v", "theta": "thp", "phi": "php", "mu": "lbc_mu_src",
            "qv": "qv0"}
_LBC_MU = {"qv": "mup0"}
_LBC_FINAL = {"u": "u", "v": "v", "theta": "thp", "phi": "php", "mu": "mup", "qv": "qv"}
#: The relaxation rows' passive names (state_relaxation_multi's header:
#: msfv is MSFV_INV in the strict build).
_LB_PASS = {"mub": "mub2d", "msfv": "msfv_inv", "fcx": "lbc_fcx", "gcx": "lbc_gcx"}
#: The held dry tendencies (prepare_fixed_tendencies' smag_r* slots, WRF's
#: *_tendf): zeroed at the step start, filled by the mixing rows and on a
#: specified domain the folded stage-1 relaxation, coupled to the map
#: factor once and added to the stage tendency on every stage
#: (add_fixed_dry_tendencies).
_HELD = {"u": "smag_ru", "v": "smag_rv", "w": "smag_rw", "theta": "smag_rth"}
_HELD_TEND = {"u": "ru_t", "v": "rv_t", "w": "rw_t", "theta": "rth_t"}
#: _couple_dry_mixing_map_factor's divisor per held row (v: times msfv_inv
#: under BIGSTEP, else divided by msfv).
_COUPLE = {"u": "msfu", "v": "msfv", "w": "msft", "theta": "msft"}
#: The map factor fold_lateral_relaxation scales each folded row by.
_FOLD_SCALE = {"u": "msfu", "v": "msfv", "theta": None}
#: _compute_coordinate_tendencies' rows at the step start (_smag2d_specs'
#: order): held kind, wrf_diff_opt1_horizontal row, the time-t field, the
#: coefficient, the hybrid mass weights and the row mass.  Theta mixes
#: thp0 (the strict basis) as its departure from diff1_theta_initial.
_MIX_ROWS = (("u", "u", "u0", "smag_km", "c1h", "c2h", "smag_mut"),
             ("v", "v", "v0", "smag_km", "c1h", "c2h", "smag_mut"),
             ("w", "w", "w0", "smag_km", "c1f", "c2f", "smag_mut"),
             ("theta", "t", "thp0", "smag_kh", "c1h", "c2h", "smag_mut"))
#: add_scalar_coordinate_wrf's moist rows after rk_step 1's
#: small_step_finish: each species' time-t field with the time-t Kh and the
#: acoustic loop's carried muts, into its held source smag_r<species>
#: (moist.fixed_scalar_tendencies), which every stage's update adds.
_MIX_Q = tuple((n, "s", n + "0", "smag_kh", "c1h", "c2h", "muts") for n in SPECIES)


def _q_slot(name):
    return "smag_r" + name


def step_inputs(geom) -> tuple:
    """What a step of ``geom`` reads from the previous one: STEP_INPUTS,
    the species on a moist state and the mu halo on a specified domain."""
    return (STEP_INPUTS + (SPECIES if geom.moist else ())
            + ((HALO,) if geom.specified else ()))


def step_parameters(geom) -> tuple:
    """The inputs a step of ``geom`` reads and never writes, whose
    perturbation and adjoint are the window's: the boundary tables."""
    if not geom.specified:
        return ()
    return LBC_INPUTS + (QV_INPUTS if geom.qv_table else ())

#: Each slow operator's names on the step dict.
_ADV_ALIAS = {
    "scalar": {"tend": "rth_t", "q": "thp", "rw": "ww_ref", "msf": "msft", "rdz": "rdnw"},
    "u": {"tend": "ru_t", "q": "u", "rw": "ww_ref", "msf": "msfu", "rdz": "rdnw"},
    "v": {"tend": "rv_t", "q": "v", "rw": "ww_ref", "msf": "msfv", "rdz": "rdnw"},
    "w": {"tend": "rw_t", "q": "w", "rw": "ww_ref", "msf": "msft", "rdz": "rdn"},
}
_PGF_ALIAS = {"p": "p_pert", "mub": "mub2d"}
#: The strict launchers pass MSFV_INV (DomainState.msfv_inv, WRF's
#: MF_VX_INV) where these operators name msfv: rk_step_prep's v coupling,
#: small_step_init, advance_uv and advance_mu_th.  small_step_finish, the
#: advection, slow_geopotential, coriolis_curvature and set_w_surface take
#: msfv itself.
_INV = {"msfv": "msfv_inv"}
_RSP_ALIAS = {**RSP_ALIAS, **_INV}
_BUOY_ALIAS = {"p": "p_pert", "mub": "mub2d"}   # mub: the moist buoyancy's base mass
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
    lbc: tuple = ()           # specified: (kind, LBCGeometry) rows after the held add
    sph: object = None        # specified: SpecPhiGeometry after each substep
    cq: object = None         # moist: CQGeometry before the slow tendencies
    sf: object = None         # moist: SumfluxGeometry
    su: tuple = ()            # moist: (species, ScalarGeometry) after the stage
    qv_capture: object = None  # moist, rk_step 1, qv tables: qv's held LBCGeometry
    flow: object = None       # moist, specified: FlowGeometry for qc and qr
    mix_q: tuple = ()         # moist mixing, rk_step 1: (_MIX_Q entry, HDGeometry)
    q_mix: bool = False       # moist mixing: the species' held sources are their rows


@dataclass(frozen=True)
class MixSpec:
    """diff_opt 1 / km_opt 4 or diff_opt 2 / km_opt 3 or 4 at the step
    start (prepare_fixed_tendencies): the cold step's diagnosis before it
    (None when rk_step 1 reuses the carried one), the deformation, the
    coefficients and the rows."""
    prp: object
    df: DFGeometry
    km: object                # KMGeometry (diff_opt 1); None under diff_opt 2
    rows: tuple               # (_MIX_ROWS entry, HDGeometry); diff_opt 2: _mix2_rows
    moist: tuple = ()         # the moist rows' held sources, zeroed at the start
    #: diff_opt 2 (dycore._compute_wrf_smag_tendencies): the coefficients'
    #: KCGeometry and km_opt (4: smag2d_km; 3: calculate_N2 and smag_km),
    #: and the vertical rows (launch_wrf_smag2d_vertical) after ``rows``
    kc: object = None
    km_opt: int = 4
    vrows: tuple = ()

    @property
    def diff2(self) -> bool:
        return self.kc is not None


@dataclass(frozen=True)
class StepGeometry:
    nz: int
    ny: int
    nx: int
    stages: tuple
    end_prp: PRPGeometry
    sw: SWGeometry
    specified: bool = False
    capture: tuple = ()       # stage 1's held relaxation: (kind, LBCGeometry)
    fold: tuple = ()          # the folded kinds (u, v, theta)
    fold_msf: bool = False    # the fold's map-factor scaling (state.has_msf)
    mix: object = None        # MixSpec: diff_opt 1 / km_opt 4
    held: tuple = ()          # the held rows added on every stage (_HELD)
    couple: tuple = ()        # the held rows divided by their map factor
    v_inv: bool = True        # v's coupling times msfv_inv (BIGSTEP)
    final: tuple = ()         # spec_bdy_final: (kind, FinalGeometry), mu first
    moist: bool = False       # qv, qc and qr (SPECIES)
    qv_table: bool = False    # specified, with qv's tables (held capture, final)


def _refuse(cfg, state) -> None:
    from woof.core.advection import scalar_horizontal_order
    checks = {
        "nested lateral boundaries": bool(getattr(cfg, "nested", False)),
        "open lateral boundaries": bool((cfg.open_x or cfg.open_y) and not cfg.specified),
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
    from woof.config import radiation_scheme_ids, wrf_mixing_package_active
    from woof.core.passive_tracer_contract import active_tracers
    from woof.core.physics_inventory import physics_enabled
    checks.update({
        "mixing package other than diff_opt 1 / km_opt 4 or diff_opt 2 / km_opt 3 or 4": bool(
            wrf_mixing_package_active(cfg) and not _mixes(cfg)),
        # named on their own: diff_opt 2's constant K, TKE and SFS closures
        "diff_opt 2 with km_opt 1, 2 or 5": bool(
            wrf_mixing_package_active(cfg) and int(cfg.diff_opt) == 2
            and int(cfg.km_opt) in (1, 2, 5)),
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
        # _step_body advances these after the species; the chain does not
        "TKE scalar": getattr(state, "tke", None) is not None,
        "chem scalars": getattr(state, "chem", None) is not None,
        # their transport (and, with the mixing, their smag rows)
        "passive tracers": bool(active_tracers(cfg)),
    })
    if state.qv is not None:   # the primal's test (dycore._step_body)
        checks.update(_refuse_moist(cfg, state))
    if _mixes(cfg):
        checks.update(_refuse_mixing(cfg, state))
    if cfg.specified:
        checks.update(_refuse_specified(cfg, state))
    bad = [k for k, v in checks.items() if v]
    if bad:
        raise NotImplementedError("the P3a step TL/AD does not cover: " + ", ".join(bad))


def _refuse_moist(cfg, state) -> dict:
    """What the moist binding does not cover: it takes qv, qc and qr
    (moist.SPECIES) through the non-positive-definite update, with
    calc_cq's n_mass 1 and the buoyancy's moist_mode 1 (mp_physics 0)."""
    from woof.core.moist import extra_moist_species
    species = tuple(getattr(state, "_scheme_base_fields", SPECIES))
    return {
        "moist_adv_opt (the positive-definite final stage)": int(cfg.moist_adv_opt) != 0,
        "moist species other than qv, qc and qr": bool(
            species != SPECIES or extra_moist_species(state)
            or getattr(state, "_registered_mass_slots", ())
            or any(getattr(state, n, None) is not None for n in ("qi", "qs", "qg", "qh"))
            or getattr(state, "etanew_restart_state", None) is not None
            or any(getattr(state, n, None) is None for n in SPECIES)),
        "moist_cq off": not bool(getattr(cfg, "moist_cq", True)),
    }


def _mixes(cfg) -> bool:
    """Whether the step runs a mixing package the binding covers (the
    package dycore.prepare_fixed_tendencies builds): diff_opt 1 / km_opt 4
    (the coordinate branch with smag2d_km's coefficients) or diff_opt 2
    with km_opt 4 (smag2d_km's slope limiter) or 3 (calculate_N2 and
    smag_km)."""
    # km_opt 3 and 4 run the package on their own (wrf_mixing_package_active)
    d, k = int(cfg.diff_opt), int(cfg.km_opt)
    return (d == 1 and k == 4) or (d == 2 and k in (3, 4))


def _diff2(cfg) -> bool:
    return int(cfg.diff_opt) == 2 and _mixes(cfg)


def _refuse_mixing(cfg, state) -> dict:
    """What the mixing binding does not cover."""
    from woof.core.dycore import coordinate_reference_missing
    out = {}
    if _diff2(cfg):
        from woof import wrf_exact
        out.update({
            # the rows' TL/AD port the GPUWM_WRF_EXACT_C_DIFFUSION statements
            "GPUWM_WRF_EXACT_DIFFUSION=0 (diff_opt 2's other arithmetic)":
                not wrf_exact.DIFFUSION_ENABLED,
            # the theta row mixes thp as WRF's t_2 = theta - 300, the strict
            # storage; elsewhere thp is theta - thb and the primal adds the base
            "diff_opt 2 outside strict mode (GPUWM_WRF_EXACT=1)": not wrf_exact.ENABLED,
            # calculate_N2's moist branches, the species' rows and the
            # moist density are not ported
            "moist diff_opt 2": state.qv is not None,
            # vertical_diffusion_2 runs only with bl_pbl_physics 0
            "diff_opt 2 with a PBL scheme": int(getattr(cfg, "bl_pbl_physics", 0)) != 0,
            "diff_opt 2 with a surface layer (USTM, HFX, QFX)":
                int(getattr(cfg, "sf_sfclay_physics", 0)) != 0,
            "isfflx outside 0-2": int(cfg.isfflx) not in (0, 1, 2),
        })
    out.update({
        # dycore.initialize_coordinate_reference would raise: a resumed
        # state cannot rebuild WRF's t_init from evolved theta
        "a resumed diff_opt 1 state without diff1_theta_initial": bool(
            coordinate_reference_missing(state, cfg)),
        "tc_horizontal_length_m": bool(getattr(cfg, "tc_horizontal_length_m", 0)),
        "m_opt (the SFS output capture)": bool(getattr(cfg, "m_opt", 0)),
    })
    return out


def _qv_table(state, cfg) -> bool:
    """Whether the boundary supplies qv on a moist specified domain:
    moist.advance_scalars_stage holds its tendency and
    apply_state_boundary_values forces it back."""
    from woof.boundary_fields import potential_external_scalar_fields
    from woof.ingest import lateral_bc as L
    if not (cfg.specified and state.qv is not None):
        return False
    supplied = L._active_device_interval(state, cfg)[0].fields
    return "qv" in supplied and "qv" in potential_external_scalar_fields(cfg)


def _refuse_specified(cfg, state) -> dict:
    """What the specified binding does not cover."""
    from woof.ingest import lateral_bc as L
    resident = getattr(state, "_lateral_boundary_device", None)
    bounds = state.lateral_boundaries
    out = {"specified lateral boundaries without attached tables":
           resident is None or bounds is None}
    if out["specified lateral boundaries without attached tables"]:
        return out
    laws = any(side.time_law is not None for iv in bounds.intervals
               for b in iv.fields.values()
               for side in (b.west, b.east, b.south, b.north))
    out.update({
        # the bound DomainClock's fp32 dtbc recurrence is not replayed
        "a bound boundary clock (DomainClock)": resident.clock is not None,
        "boundary time laws other than linear": laws,
        "relax_w (the table's w on a specified domain)": bool(L.specified_relaxes_w(cfg)),
        "DFI boundary forcing": getattr(state, "_dfi_boundary_start", None) is not None
                                or bool(getattr(cfg, "dfi_constant_bc", False)),
        "streamed tile seams": bool(getattr(bounds, "seam_sides", ()) or ()),
        "spec_zone 0": int(cfg.spec_zone) < 1,
        # the binding assumes both axes bounded (dycore._boundary_x/_y)
        "a specified domain with a periodic axis": not (
            _dycore_boundary(cfg, "x") and _dycore_boundary(cfg, "y")),
    })
    if state.qv is not None:
        from woof.boundary_fields import scalar_array_boundary_fields
        supplied = L._active_device_interval(state, cfg)[0].fields
        qv = _qv_table(state, cfg)
        out.update({
            # a supplied hydrometeor takes the held path instead of flow_dep_bdy
            "boundary tables for scalars other than qv": any(
                n in L.COUPLED_SCALAR_STATE_FIELDS and n != "qv" for n in supplied),
            "a qv table spec_bdy_final does not force back": qv and (
                "qv" not in L.COUPLED_SCALAR_STATE_FIELDS
                or "qv" in scalar_array_boundary_fields(cfg)),
        })
    return out


def _dycore_boundary(cfg, axis: str) -> bool:
    from woof.core import dycore as D
    return bool(D._boundary_x(cfg) if axis == "x" else D._boundary_y(cfg))


def step_geometries(state, cfg, nstep: int) -> tuple:
    """Geometries for ``nstep`` chained strict steps from ``state``.  The
    first follows the state (a cold state, without WRF's carried muts,
    diagnoses on rk_step 1); every later step follows a strict step, so it
    reuses that step's diagnosis and carried muts."""
    if cfg.specified:   # each step reads its own boundary time
        return tuple(step_geometry(state, cfg, step=k, **({} if k == 0 else dict(
            reuse_first=True, carried=True))) for k in range(nstep))
    first = step_geometry(state, cfg)
    later = step_geometry(state, cfg, reuse_first=True, carried=True)
    return (first,) + (later,) * (nstep - 1)


def _boundary_times(state, cfg, step: int) -> tuple:
    """WRF's dtbc for step ``step`` after ``state`` (the unbound clock,
    lateral_bc._served_external_interval): the stage rows' at the step
    start and spec_bdy_final's at its end (apply_state_boundary_values),
    elapsed time accumulated as dycore.step does.  Both must fall in the
    interval the window starts in: its tables are the step's inputs.  A
    step ending on the interval's end is accepted when the primal serves
    that time from the same interval (the last one, interval_index) and
    refused when it serves it from the next one."""
    bounds = state.lateral_boundaries
    first = bounds.interval_at(state.elapsed_seconds)
    t = state.elapsed_seconds
    for _ in range(step):
        t += cfg.dt
    end = t + cfg.dt
    def at(time):
        try:
            return bounds.interval_at(time)
        except ValueError:   # past the last table: another interval too
            return None
    if at(t) is not first or at(end) is not first:
        raise NotImplementedError(
            "the P3a step TL/AD does not cover: a window reaching another boundary "
            f"interval (step {step} ends at {end} s, which the primal serves from "
            f"another interval than {first.start_seconds}-{first.end_seconds} s)")
    return t - first.start_seconds, end - first.start_seconds


def step_geometry(state, cfg, *, reuse_first: bool | None = None,
                  carried: bool | None = None, step: int = 0) -> StepGeometry:
    """The operators' geometries for one strict step of ``state`` under
    ``cfg``, with each scalar rounded as its primal launcher rounds it.
    ``reuse_first``: rk_step 1 reads the previous step's diagnosis;
    ``carried``: the state holds WRF's muts (and, specified, the mu
    halo).  Both default to what ``state`` holds now; chained steps take
    :func:`step_geometries`.  ``step``: the step's index after ``state``
    (its boundary time, specified only)."""
    from woof.core import dycore as D
    from woof.core.acoustic import damp_magnitude
    from woof.core.advection import scalar_horizontal_order, vertical_orders
    _refuse(cfg, state)
    f32 = np.float32
    msf = int(bool(state.has_msf))
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
    spec = bool(cfg.specified)
    halo = 0
    if spec:
        from woof.state_serialization_contract import carried_wrf_mu_halo
        if step == 0:
            halo = int(carried_wrf_mu_halo(state) is not None
                       and getattr(state, "_wrf_mass_carriers_ready", True))
        else:
            halo = int(carried)
    bx, by = int(D._boundary_x(cfg)), int(D._boundary_y(cfg))
    szone = int(cfg.spec_zone) if spec else 0
    mzone = 1 if spec else 0     # acoustic._mass_w_boundary_zone
    vsca, vmom = vertical_orders(cfg)
    hsca, hmom = scalar_horizontal_order(cfg), int(cfg.h_mom_adv_order)
    adv = tuple(AdvGeometry(kind, nz, ny, nx, rdx, rdy, vorder=v, horder=h,
                            open_x=bx, open_y=by, spec=int(spec))
                for kind, v, h in (("scalar", vsca, hsca), ("u", vmom, hmom),
                                   ("v", vmom, hmom), ("w", vsca, hsca)))
    pgf = PGFGeometry(nz, ny, nx, rdx, rdy, cfn=cfn, cfn1=cfn1, top_lid=top_lid,
                      boundary_x=bx, boundary_y=by, **cf)
    buoy = BuoyGeometry(nz, ny, nx, has_msf=msf)
    geo = GeoGeometry(nz, ny, nx, quarter_rdx=float(f32(0.25 / dx)),
                      quarter_rdy=float(f32(0.25 / dy)), cfn=cfn, cfn1=cfn1, order=hsca,
                      has_msf=msf, specified=int(spec))
    wd = None
    if cfg.w_damping == 1:
        wd = WDampGeometry(nz, ny, nx, dt=float(f32(cfg.dt)),
                           w_damp_on=float(f32(D.w_damp_onset(cfg))),
                           w_crit_cfl=float(f32(cfg.w_crit_cfl)))
    rsp = RSPGeometry(nz, ny, nx, rdx=float(f32(1) / f32(dx)), rdy=float(f32(1) / f32(dy)),
                      has_msf=msf, boundary_x=bx, boundary_y=by)
    cc = (CCGeometry(nz, ny, nx, rdx=rdx, rdy=rdy, boundary_x=bx, boundary_y=by)
          if state.rotational else None)
    moist = state.qv is not None
    qvt = _qv_table(state, cfg)
    if moist:   # calc_cq n_mass 1 (mp_physics 0), the moist pgf and buoyancy
        pgf = dataclasses.replace(pgf, moist_cq=1)
        buoy = dataclasses.replace(buoy, moist=1)
    lb = _lbc_geometries(state, cfg, step, msf, qvt) if spec else None
    mixq = moist and _mixes(cfg)   # the moist rows' held sources
    stages = []
    for istage, (nsub, dtau_stage) in enumerate(D._rk_acoustic_stages(cfg)):
        dtau = float(f32(dtau_stage))
        prp = None
        if not (istage == 0 and reuse_first):
            prp = PRPGeometry(nz, ny, nx, hypso=hypso, carried=int(carried or istage > 0),
                              p_top=p_top, moist=int(moist))

        def sub(smdiv):
            return (UVGeometry(nz, ny, nx, rdx, rdy, dtau, smdiv=smdiv, top_lid=top_lid,
                               has_msf=msf, spec_zone=szone, moist_cq=int(moist), **cf),
                    MTGeometry(nz, ny, nx, rdx, rdy, dtau, write_mudf=0, mapped=msf,
                               spec_zone=mzone, open_x=bx, open_y=by),
                    WGeometry(nz, ny, nx, rdx, rdy, dtau, epssm=float(f32(cfg.epssm)),
                              top_lid=top_lid, dampmag=float(damp_magnitude(cfg, dtau)),
                              has_msf=msf, spec_zone=mzone, boundary_x=bx, boundary_y=by,
                              moist_cq=int(moist), **cf))
        moist_parts = {}
        if moist:
            # advance_scalars_stage's dt_eff = nsub*dtau, DTYPE-rounded by
            # the update; the advection's scalars are flux_div_scalar's
            sdt = float(f32(nsub * dtau_stage))
            moist_parts = dict(
                cq=CQGeometry(nz, ny, nx, n_mass=1),
                sf=SumfluxGeometry(nz, ny, nx, int(nsub)),
                su=tuple((n, ScalarGeometry(nz, ny, nx, rdx, rdy, sdt, vorder=vsca,
                                            horder=hsca, open_x=bx, open_y=by,
                                            spec=int(spec), spec_zone=szone,
                                            has_msf=msf,
                                            held=int((n == "qv" and qvt) or mixq)))
                         for n in SPECIES),
                qv_capture=lb["qv_capture"] if (qvt and istage == 0) else None,
                flow=FlowGeometry(nz, ny, nx, spec_zone=szone) if spec else None,
                mix_q=tuple((r, HDGeometry(nz, ny, nx, rdx, rdy, row=r[1], boundary_x=bx,
                                           boundary_y=by))
                            for r in _MIX_Q) if (mixq and istage == 0) else (),
                q_mix=mixq)
        stages.append(StageSpec(
            rk_step=istage + 1, nsub=int(nsub), diagnose=prp,
            rsp=dataclasses.replace(rsp, halo=halo) if istage == 0 else rsp,
            adv=adv, pgf=pgf, buoy=buoy, geo=geo, wd=wd, cc=cc,
            ss=SSGeometry(nz, ny, nx, has_msf=msf, boundary_x=bx, boundary_y=by,
                          rk_step=istage + 1),
            cw=CWGeometry(nz, ny, nx, dtau, epssm=float(f32(cfg.epssm)), top_lid=top_lid,
                          moist_cq=int(moist)),
            first=sub(0.0), later=sub(float(f32(cfg.smdiv))),
            lbc=lb["rows"][istage] if spec else (),
            sph=SpecPhiGeometry(nz, ny, nx, dtau, spec_zone=szone) if spec else None,
            **moist_parts))
    mix = None
    if _diff2(cfg):
        mix = _mix2_spec(state, cfg, stages[0].diagnose, rdx, rdy, bx, by)
        held = tuple(_HELD)
    elif _mixes(cfg):
        # launch_wrf_smag2d_km's and launch_coordinate_horizontal's scalars
        mix = MixSpec(
            prp=stages[0].diagnose,   # _step_body's cold diagnosis before the mixing
            df=DFGeometry(nz, ny, nx, rdx, rdy, cf1=float(f32(state.cf1)),
                          cf2=float(f32(state.cf2)), cf3=float(f32(state.cf3)),
                          boundary_x=bx, boundary_y=by,
                          # the primal's flag 2 (smag_boundary_flags)
                          seam_cold=int(cfg.diff_metrics_seam_slope is False)),
            km=KMGeometry(nz, ny, nx, dx=float(f32(dx)), dy=float(f32(dy)),
                          c_s=float(f32(cfg.c_s)), prandtl=float(f32(D._PRANDTL)),
                          boundary_x=bx, boundary_y=by),
            rows=tuple((r, HDGeometry(nz, ny, nx, rdx, rdy, row=r[1], boundary_x=bx,
                                      boundary_y=by))
                       for r in _MIX_ROWS),
            moist=tuple(_q_slot(n) for n in SPECIES) if mixq else ())
        held = tuple(_HELD)
    else:
        held = tuple(lb["fold"]) if spec else ()
    return StepGeometry(
        nz, ny, nx, tuple(stages),
        end_prp=PRPGeometry(nz, ny, nx, hypso=hypso, carried=1, p_top=p_top,
                            moist=int(moist)),
        sw=SWGeometry(nz, ny, nx, half_rdx=float(f32(0.5 / dx)),
                      half_rdy=float(f32(0.5 / dy)), mapped=msf, boundary_x=bx,
                      boundary_y=by, **cf),
        specified=spec, moist=moist, qv_table=qvt, mix=mix, held=held,
        couple=held if msf else (), v_inv=bool(D.BIGSTEP_ENABLED),
        **({} if not spec else dict(
            capture=lb["capture"], fold=lb["fold"], fold_msf=bool(msf),
            final=lb["final"])))


#: diff_opt 2's rows on the step dict (dycore._compute_wrf_smag_tendencies,
#: dry): each H2 row's names bound to the step's, the time-t fields
#: (_wrf_smag_grid_args with time_t), the diagnosed alt and the K the
#: primal hands the row.  ``kmv`` is the vertical momentum K (smag_km under
#: km_opt 4, where smag2d_km's xkmv is xkmh; smag_kmv under km_opt 3).
_MIX2_TIME_T = (("u", "u0"), ("v", "v0"), ("w", "w0"), ("php", "php0"))
_MIX2_HORIZONTAL = (("u", "u", "km"), ("v", "v", "km"), ("w", "w", "kmv"),
                    ("theta", "t", "kh"))
_MIX2_VERTICAL = (("u", "vu", "kmv"), ("v", "vv", "kmv"), ("w", "vw", "kmv"),
                  ("theta", "vt", "khv"))
_MIX2_K = {4: {"km": "smag_km", "kh": "smag_kh", "kmv": "smag_km", "khv": None},
           3: {"km": "smag_km", "kh": "smag_kh", "kmv": "smag_kmv", "khv": "smag_khv"}}


def _mix2_bind(kind, g, k):
    """(H2 name, step name) pairs of one diff_opt 2 row."""
    names = dict(_MIX2_TIME_T, field="thp0", xk=k)
    return tuple((n, names.get(n, n)) for n in h2_active(g) + h2_passive(g))


def _mix2_spec(state, cfg, prp, rdx, rdy, bx, by) -> MixSpec:
    """diff_opt 2's MixSpec: the deformation, the coefficients
    (launch_wrf_smag2d_km or launch_wrf_smag3d_km's scalars), the
    horizontal rows (u, v, w and theta, each followed by its open-strip
    zeroing) and launch_wrf_smag2d_vertical's rows with the surface terms
    of its isfflx case: without a surface layer the USTM stress and the
    HFX / QFX arms are cold (inactive), so isfflx 0 takes the cd0 stress,
    isfflx 0 and 2 the prescribed heat flux.  km_opt 4 leaves xkhv = 0, so
    its theta row is the heat flux alone (or none)."""
    from woof.core import dycore as D
    f32 = np.float32
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    km_opt, isfflx = int(cfg.km_opt), int(cfg.isfflx)
    seam = int(cfg.diff_metrics_seam_slope is False)   # the primal's flag 2
    cf = dict(cf1=float(f32(state.cf1)), cf2=float(f32(state.cf2)), cf3=float(f32(state.cf3)))
    common = dict(boundary_x=bx, boundary_y=by, seam_cold=seam)
    K = _MIX2_K[km_opt]
    rows = []
    for kind, row, k in _MIX2_HORIZONTAL:
        g = H2Geometry(nz, ny, nx, rdx, rdy, row=row, **cf, **common)
        rows.append((kind, g, _mix2_bind(kind, g, K[k])))
    heat = dict(heat="const", heat_flux=float(f32(cfg.tke_heat_flux))) \
        if isfflx in (0, 2) else {}
    vrows = []
    for kind, row, k in _MIX2_VERTICAL:
        extra = {}
        if row in ("vu", "vv") and isfflx == 0:
            extra = dict(surface="cd0", cd0=float(f32(cfg.tke_drag_coefficient)))
        elif row == "vt":
            if km_opt == 3:
                extra = dict(full_theta=1, **heat)
            elif heat:
                extra = dict(interior=0, **heat)
            else:
                continue
        g = H2Geometry(nz, ny, nx, rdx, rdy, row=row, **cf, **common, **extra)
        vrows.append((kind, g, _mix2_bind(kind, g, K[k])))
    return MixSpec(
        prp=prp,   # _step_body's cold diagnosis before the mixing
        df=DFGeometry(nz, ny, nx, rdx, rdy, **cf, **common),
        km=None, rows=tuple(rows), vrows=tuple(vrows), km_opt=km_opt,
        kc=KCGeometry(nz, ny, nx, rdx, rdy, dx=float(f32(cfg.dx)), dy=float(f32(cfg.dy)),
                      c_s=float(f32(cfg.c_s)), prandtl=float(f32(D._PRANDTL)),
                      dt=float(f32(cfg.dt)), mix_upper_bound=float(f32(cfg.mix_upper_bound)),
                      isotropic=int(bool(cfg.mix_isotropic)), **common))


def _lbc_geometries(state, cfg, step: int, msf: int, qvt: bool = False) -> dict:
    """The specified binding's rows (lateral_bc's strict
    capture_folded_relaxation, _apply_state_lateral_boundaries_wrf and
    apply_state_boundary_values) for step ``step``; with ``qvt`` qv's
    held capture (moist.advance_scalars_stage on rk_step 1) and its
    spec_bdy_final after phi's."""
    from woof.ingest import lateral_bc as L
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    dtbc, dtbc_end = _boundary_times(state, cfg, step)
    device, _common, _weights = L._relaxation_context(state, cfg)
    width = {f: int(device.fields[f].west.value.shape[-1])
             for f in LBC_FIELDS + (("qv",) if qvt else ())}
    if qvt and width["qv"] != width["theta"]:   # qv's row takes theta's Davies weights
        raise NotImplementedError("the P3a step TL/AD does not cover: a qv boundary table "
                                  f"{width['qv']} wide beside theta's {width['theta']}")
    zones = dict(spec_zone=int(cfg.spec_zone), relax_zone=int(cfg.relax_zone))

    def row(kind, **flags):
        return (kind, LBCGeometry(nz, ny, nx, kind, width=width[kind], dtbc=dtbc,
                                  has_msf=msf, **zones, **flags))
    folded = L.folded_relaxation_rows(cfg)
    capture = tuple(row(k, held_compute=1, held_store=1, target=0) for k in folded)
    rows = []
    for istage in range(3):
        r = [row(k) for k in ("u", "v", "theta")]
        if istage == 0:
            r.append(row("phi", held_compute=1, held_store=1, divide_msf=msf))
        else:
            r.append(row("phi", held_read=1))
        r.append(row("mu", apply_relax=int(istage == 0)))
        rows.append(tuple(r))
    final = tuple((k, FinalGeometry(nz, ny, nx, k, width=width[k], spec_zone=int(cfg.spec_zone),
                                    dtbc=dtbc_end, has_msf=msf))
                  for k in ("mu", "u", "v", "theta", "phi") + (("qv",) if qvt else ()))
    return {"rows": tuple(rows), "capture": capture, "fold": tuple(folded), "final": final,
            "qv_capture": row("qv", apply_relax=1)[1] if qvt else None}


def _names() -> tuple:
    """Every name the step's operators read or write."""
    def al(names, alias):
        return tuple(alias.get(n, n) for n in names)
    out = PROGNOSTICS + tuple(n + "0" for n in PROGNOSTICS) + TENDENCIES + ("muts",)
    out += al(PRP_STATE + PRP_ACTIVE + PRP_PASSIVE, PRP_ALIAS)
    out += al(RSP_STATE + RSP_ACTIVE + RSP_PASSIVE, _RSP_ALIAS) + ("msfv",)
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
    cq slots (written each stage on a moist state, unread on a dry one)
    and mudf.  A moist state adds its species, their time-t copies, the
    sumflux accumulators (the primal's, which the step zeroes first) and
    the update's tendency scratch."""
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
    if state.qv is not None:
        for n in SPECIES + tuple(n + "0" for n in SPECIES):
            out[n] = cp.ascontiguousarray(cp.asarray(getattr(state, n)).astype(dtype, copy=True))
        for n, shape, slot in zip(SUMFLUX, (sh["ru"], sh["rv"], sh["ww_ref"]),
                                  ("rk_ru_m", "rk_rv_m", "rk_ww_m")):
            out[n] = cp.ascontiguousarray(
                cp.asarray(state.scratch(shape, slot)).astype(dtype, copy=True))
        out["moist_tend"] = cp.zeros((nz, ny, nx), dtype=dtype)
    if cfg.specified or _mixes(cfg):   # the held tendencies (_HELD)
        for n, shape in zip(_HELD.values(), (sh["ru"], sh["rv"], sh["ww_ref"], sh["p_pert"])):
            out[n] = cp.zeros(shape, dtype=dtype)
    if _mixes(cfg):
        out.update(_mix_state(state, cfg, dtype))
        if state.qv is not None:   # the moist rows' held sources
            out.update({_q_slot(n): cp.zeros((nz, ny, nx), dtype=dtype) for n in SPECIES})
    if _diff2(cfg):
        # the base theta (calculate_N2, the theta row's full_theta), 3-D
        thb = cp.asarray(state.thb)
        if thb.ndim == 1:
            thb = cp.broadcast_to(thb[:, None, None], (nz, ny, nx))
        out["thb"] = cp.ascontiguousarray(thb.astype(dtype, copy=True))
        if int(cfg.km_opt) == 3:   # smag_km's vertical pair and BN2
            out.update({n: cp.zeros((nz, ny, nx), dtype=dtype)
                        for n in ("smag_kmv", "smag_khv", "bn2")})
    if cfg.specified:
        out.update(_lbc_state(state, cfg, dtype))
    return out


def _mix_state(state, cfg, dtype) -> dict:
    """The mixing's arrays: the deformation, the coefficients and the row
    mass (written each step), the vertical metrics the deformation reads
    and WRF's initial theta (dycore.initialize_coordinate_reference, which
    the primal's first step would take the same way, storing it on the
    state), passive as in WRFPLUS."""
    import cupy as cp
    from woof.core import dycore as D
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    out = {n: cp.zeros((nz, ny, nx), dtype=dtype)
           for n in DF_STATE + ("smag_km", "smag_kh")}
    out["smag_mut"] = cp.zeros((ny, nx), dtype=dtype)
    arrays = [("dn", state.dn), ("dnw", state.dnw)]
    if not _diff2(cfg):
        D.initialize_coordinate_reference(state, cfg)
        arrays.append(("diff1_theta_initial", state._scratch["diff1_theta_initial"]))
    for n, a in arrays:
        out[n] = cp.ascontiguousarray(cp.asarray(a).astype(dtype, copy=True))
    return out


def _lbc_state(state, cfg, dtype) -> dict:
    """The specified binding's arrays: the interval's tables and Davies
    weights (lateral_bc._relaxation_context), the carried mu halo (mup
    where none is carried: the first step does not read it), the held
    relaxation rows and the mu row's unread source;
    with qv's tables, those and qv's held tendency (and its row's unread
    held slot)."""
    import cupy as cp
    from woof.ingest import lateral_bc as L
    from woof.state_serialization_contract import carried_wrf_mu_halo
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    device, _common, (fcx, gcx) = L._relaxation_context(state, cfg)
    qvt = _qv_table(state, cfg)
    out = {}
    for f in LBC_FIELDS + (("qv",) if qvt else ()):
        b = device.fields[f]
        for side in ("west", "east", "south", "north"):
            sd = getattr(b, side)
            out[f"lbc_{f}_{side}"] = sd.value
            out[f"lbc_{f}_{side}_t"] = sd.tendency
    out["lbc_fcx"], out["lbc_gcx"] = fcx, gcx
    halo = carried_wrf_mu_halo(state)
    out[HALO] = state.mup if halo is None else halo
    out = {n: cp.ascontiguousarray(cp.asarray(a).astype(dtype, copy=True))
           for n, a in out.items()}
    u, v, m, w = (nz, ny, nx + 1), (nz, ny + 1, nx), (nz, ny, nx), (nz + 1, ny, nx)
    for n, sh in (("lbc_held_u", u), ("lbc_held_v", v), ("lbc_held_theta", m),
                  ("lbc_held_phi", w), ("lbc_held_mu", (1, ny, nx)),
                  ("lbc_mu_src", (1, ny, nx))) + (
                      (("lbc_qv_held", m), ("lbc_held_qv", m)) if qvt else ()):
        out[n] = cp.zeros(sh, dtype=dtype)
    return out


def _3d(a):
    return a if a.ndim == 3 else a[None]


def _lb_views(x, dx, kind):
    """lbc_tl/lbc_ad's dicts for a ``kind`` row on the step dicts: the
    state (tendency, held buffer), its trajectory and the active names."""
    def act(d):
        out = {"src": _3d(d[_LBC_SRC[kind]]), "mup": d[_LBC_MU.get(kind, "mup")]}
        out.update({t: d[f"lbc_{kind}_{t}"] for t in LB_TABLES})
        return out
    st = lambda d: {"tend": _3d(d[_LBC_TEND[kind]]), "held": d["lbc_held_" + kind]}
    traj = {**act(x), **{n: x[_LB_PASS.get(n, n)] for n in LB_PASSIVE}}
    return st, traj, act


def _lf_views(x, kind):
    def act(d):
        out = {"muts": d["muts"]}
        out.update({t: d[f"lbc_{kind}_{t}"] for t in LB_TABLES})
        return out
    st = lambda d: {"target": _3d(d[_LBC_FINAL[kind]])}
    traj = {**act(x), **{n: x[n] for n in LF_PASSIVE}}
    return st, traj, act


def _acoustic(d: dict) -> dict:
    """``d`` as the acoustic substep's operators name it (fnm and fnp on
    nz + 1 levels, the primal never reading the last; msfv is MSFV_INV,
    as advance_uv and advance_mu_th take it)."""
    return {**d, "fnm": d["fnm_w"], "fnp": d["fnp_w"], "msfv": d["msfv_inv"]}


def _inv(d: dict) -> dict:
    """``d`` with msfv as MSFV_INV (small_step_init's argument)."""
    return {**d, "msfv": d["msfv_inv"]}


# ------------------------------------------------------------------ the step

def _op_table(spec: StageSpec):
    """The stage's operators from rk_step_prep to the slow tendencies:
    (name, tl, ad, alias, geometry, written names)."""
    ops = []
    if spec.diagnose is not None:
        ops.append(("prp", calc_p_rho_phi_tl, calc_p_rho_phi_ad, PRP_ALIAS, spec.diagnose,
                    PRP_STATE))
    ops.append(("rsp", rk_step_prep_tl, rk_step_prep_ad, _RSP_ALIAS, spec.rsp, RSP_STATE))
    if spec.cq is not None:   # prepare_moist_cq, after stage_fluxes
        ops.append(("cq", cq_tl, cq_ad, {}, spec.cq, CQ_STATE))
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
    """One strict step (dry, or moist with qv, qc and qr): ``x`` advances (NL in its dtype) and ``dx``
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

    saved = PROGNOSTICS + (SPECIES if geom.moist else ())
    copy_names(tuple(n + "0" for n in saved), saved)               # _save_time_t
    _held_start_tl(x, dx, geom, snap, tl)
    seen("mixing", x)
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
        if geom.specified or geom.held:
            _lbc_stage_tl(x, dx, geom, spec, snap, tl)
            seen("lbc", x)
        snap(SI_STATE)
        small_step_init_tl(_inv(x), _inv(dx), _inv(x), _inv(dx), spec.ss, nl=True)
        seen("si", x)
        snap(("muts",))
        src = "mup" if spec.rk_step == 1 else "mup0"
        cp.add(x["mub2d"], x[src], out=x["muts"])
        if tl:
            dx["muts"][...] = dx[src]
        snap(CW_STATE)
        calc_coef_w_tl(x, dx, x, dx, spec.cw, nl=True)
        seen("cw", x)
        if spec.sf is not None:                                   # zero_sumflux
            snap(SUMFLUX)
            sumflux_zero_tl(x, dx, spec.sf, nl=True)
        for i in range(spec.nsub):
            acoustic_substep_tl(_acoustic(x), _acoustic(dx),
                                spec.first if i == 0 else spec.later, tape)
            if spec.sph is not None:   # advance_specified_phi_w
                snap(SP_STATE)
                sph_tl(x, dx, x, dx, spec.sph, nl=True)
            if spec.sf is not None:                               # accumulate_sumflux
                snap(SUMFLUX)
                sumflux_accumulate_tl(x, dx, x, dx, spec.sf, nl=True)
            seen("substep", x)
        if not geom.specified:
            # _capture_wrf_muts; with a boundary zone it keeps the
            # acoustic carrier the substeps advanced
            snap(("muts",))
            cp.add(cp.add(x["mub2d"], x["mup"]), x["mu_pp"], out=x["muts"])
            if tl:
                cp.add(dx["mup"], dx["mu_pp"], out=dx["muts"])
        snap(SF_STATE)
        small_step_finish_tl(x, dx, x, dx, spec.ss, nl=True)
        seen("sf", x)
        # add_scalar_coordinate_wrf: each row adds into its own held
        # source, zeroed at the step start and read by none: no snapshot
        for row, g in spec.mix_q:
            hd_tl({"tend": x[_q_slot(row[0])]}, {"tend": dx[_q_slot(row[0])]},
                  _hd_traj(x, row, g), _hd_act(dx, row), g, nl=True)
        if geom.moist:
            _scalars_tl(x, dx, spec, snap, tl)
            seen("scalars", x)
    snap(_written(PRP_ALIAS, PRP_STATE))                          # epilogue diagnosis
    s, p = _view(x, PRP_ALIAS), _view(dx, PRP_ALIAS)
    calc_p_rho_phi_tl(s, p, s, p, geom.end_prp, nl=True)
    if geom.specified:
        snap((HALO,))                                             # WRF_MU_HALO_SLOT
        x[HALO][...] = x["mup"]
        if tl:
            dx[HALO][...] = dx["mup"]
        for kind, g in geom.final:                                # spec_bdy_final
            snap((_LBC_FINAL[kind],))
            st, traj, act = _lf_views(x, kind)
            lbf_tl(st(x), st(dx), traj, act(dx), g, nl=True)
        seen("final", x)
    snap(("w",))
    set_w_surface_tl(x, dx, x, dx, geom.sw, nl=True)
    if not geom.specified:
        snap(("u", "v"))                                          # close_periodic_alias
        for d in ((x, dx) if tl else (x,)):
            d["u"][..., -1] = d["u"][..., 0]
            d["v"][..., -1, :] = d["v"][..., 0, :]


#: sumflux's stage fluxes on the step dict (finish_sumflux's ru, rv, ww).
_SF_ALIAS = {"ww": "ww_ref"}
#: The flow-dependent species (moist._apply_specified_scalar_flow_boundaries:
#: every species but qv, which never takes flow_dep_bdy).
_FLOW_SPECIES = ("qc", "qr")


def _su_view(d, name, spec):
    """One species' update dicts on the step dict (scalar.su_tl/_ad): the
    held source is qv's captured boundary tendency where its tables are
    attached, else, with the moist mixing (``spec.q_mix``), the species'
    row (moist._wrf_scalar_source)."""
    out = {"q": d[name], "q0": d[name + "0"], "tend": d["moist_tend"]}
    if name == "qv" and "lbc_qv_held" in d:
        out["held"] = d["lbc_qv_held"]
    elif spec.q_mix:
        out["held"] = d[_q_slot(name)]
    for n in SUMFLUX + ("mup0", "mup", "mub2d", "c1h", "c2h", "msft", "rdnw", "fnm", "fnp"):
        out[n] = d[n]
    return out


def _flow_traj(d):
    """flow_dep_bdy's fluxes: the stage's sumflux ones, as
    advance_scalars_stage passes them."""
    return {"ru": d["ru_m"], "rv": d["rv_m"]}


def _scalars_tl(x, dx, spec, snap, tl=True):
    """A moist stage's scalar work after small_step_finish, in
    dycore._step_body's strict order: finish_sumflux, then
    moist.advance_scalars_stage (qv's held capture on rk_step 1, the
    qv, qc and qr updates, flow_dep_bdy for qc and qr on a specified
    domain)."""
    snap(SUMFLUX)
    sumflux_finish_tl(_view(x, _SF_ALIAS), _view(dx, _SF_ALIAS), _view(x, _SF_ALIAS),
                      _view(dx, _SF_ALIAS), spec.sf, nl=True)
    if spec.qv_capture is not None:
        snap(("lbc_qv_held",))
        for d in (x, dx) if tl else (x,):   # moist._wrf_scalar_source: the mixing
            if spec.q_mix:
                d["lbc_qv_held"][...] = d[_q_slot("qv")]
            else:
                d["lbc_qv_held"][...] = 0
        st, traj, act = _lb_views(x, dx, "qv")
        lbc_tl(st(x), st(dx), traj, act(dx), spec.qv_capture, nl=True)
    for name, g in spec.su:
        snap((name, "moist_tend"))
        su_tl(_su_view(x, name, spec), _su_view(dx, name, spec), _su_view(x, name, spec),
              _su_view(dx, name, spec),
              g, nl=True)
    if spec.flow is not None:
        for name in _FLOW_SPECIES:
            snap((name,))
            flow_tl({"q": x[name]}, {"q": dx[name]}, _flow_traj(x), {}, spec.flow, nl=True)


def _scalars_ad(ax, back, spec):
    """Adjoint of :func:`_scalars_tl`."""
    if spec.flow is not None:
        for name in reversed(_FLOW_SPECIES):
            base = back()
            flow_ad({"q": ax[name]}, _flow_traj(base), {}, spec.flow)
    for name, g in reversed(spec.su):
        base = back()
        su_ad(_su_view(ax, name, spec), _su_view(base, name, spec), _su_view(ax, name, spec), g)
    if spec.qv_capture is not None:
        base = back()
        st, traj, act = _lb_views(base, None, "qv")
        lbc_ad(st(ax), traj, _lb_adj(ax, "qv"), spec.qv_capture)
        if spec.q_mix:
            ax[_q_slot("qv")] += ax["lbc_qv_held"]
        ax["lbc_qv_held"][...] = 0
    back()
    sumflux_finish_ad(_view(ax, _SF_ALIAS), None, _view(ax, _SF_ALIAS), spec.sf)


def _couple(d, geom, names=None):
    """_couple_dry_mixing_map_factor on the held rows of ``d`` (the msf
    factors read from ``names``, ``d`` itself by default): divided by the
    row's map factor, v times msfv_inv under BIGSTEP.  Its own adjoint."""
    names = d if names is None else names
    for kind in geom.couple:
        slot = d[_HELD[kind]]
        if kind == "v" and geom.v_inv:
            slot *= names["msfv_inv"][None]
        else:
            slot /= names[_COUPLE[kind]][None]


def _held_start_tl(x, dx, geom, snap, tl):
    """prepare_fixed_tendencies' dry work at the step start: the held
    rows (and with the moist mixing the species' held sources) zeroed;
    with the mixing (a cold step's diagnosis first) the deformation and smag2d_km of the time-t winds and geopotential, the
    rows' mass mub2d + mup0, the u, v, w and theta rows into the held rows
    and, unless a specified domain's fold takes it, the map-factor
    coupling."""
    import cupy as cp
    m = geom.mix
    if m is not None and m.prp is not None:     # _step_body's cold diagnosis
        snap(_written(PRP_ALIAS, PRP_STATE))
        s, p = _view(x, PRP_ALIAS), _view(dx, PRP_ALIAS)
        calc_p_rho_phi_tl(s, p, s, p, m.prp, nl=True)
    if geom.held:
        names = _held_zeroed(geom)
        snap(names)
        for n in names:
            x[n][...] = 0
            if tl:
                dx[n][...] = 0
    if m is None:
        return
    snap(DF_STATE)
    df_tl({n: x[n] for n in DF_STATE}, {n: dx[n] for n in DF_STATE}, _df_traj(x),
          _df_act(dx), m.df, nl=True)
    if m.diff2:
        _mix2_tl(x, dx, geom, snap, tl)
        return
    snap(("smag_km", "smag_kh"))
    km_tl(_km_out(x), _km_out(dx), {**{n: x[n] for n in DF_STATE}, "msft": x["msft"]},
          {n: dx[n] for n in DF_STATE}, m.km, nl=True)
    snap(("smag_mut",))
    cp.add(x["mub2d"], x["mup0"], out=x["smag_mut"])
    if tl:
        dx["smag_mut"][...] = dx["mup0"]
    # each row adds into its own slot, zeroed above: no snapshot needed
    for row, g in m.rows:
        kind = row[0]
        hd_tl({"tend": x[_HELD[kind]]}, {"tend": dx[_HELD[kind]]}, _hd_traj(x, row, g),
              _hd_act(dx, row), g, nl=True)
    if not geom.specified and geom.couple:
        snap(tuple(_HELD[k] for k in geom.couple))
        _couple(x, geom)
        if tl:
            _couple(dx, geom, x)


#: diff_opt 2's coefficients on the step dict.
_K2_OUT = {"xkmh": "smag_km", "xkhh": "smag_kh"}
_K3_OUT = {**_K2_OUT, "xkmv": "smag_kmv", "xkhv": "smag_khv"}


def _kc_names(m) -> list:
    """diff_opt 2's coefficient operators in order: (name, outputs,
    active inputs {operator name: step name}, passive names)."""
    t = dict(_MIX2_TIME_T)
    d = {n: n for n in DF_STATE}
    if m.km_opt == 4:
        return [("k2", _K2_OUT, {**d, "php": "php0"}, ("phb", "msft"))]
    return [("n2", {"bn2": "bn2"}, {"thp": "thp0", "php": "php0"}, ("thb", "phb")),
            ("k3", _K3_OUT, {**d, **t, "bn2": "bn2"}, ("phb", "msft", "msfu", "msfv"))]


_KC_OPS = {"k2": (k2_tl, k2_ad), "n2": (n2_tl, n2_ad), "k3": (k3_tl, k3_ad)}


def _bound(d, pairs):
    return {a: d[b] for a, b in pairs}


def _strip(d, names, geom):
    """dycore._zero_open_strips(width 1) on ``d[names]``: the outer
    row of each bounded axis.  Linear and its own adjoint."""
    bx, by = geom.mix.df.boundary_x, geom.mix.df.boundary_y
    for n in names:
        a = d[n]
        if bx:
            a[..., :1] = 0
            a[..., -1:] = 0
        if by:
            a[..., :1, :] = 0
            a[..., -1:, :] = 0


def _mix2_tl(x, dx, geom, snap, tl):
    """diff_opt 2 after the deformation (dycore._compute_wrf_smag_tendencies
    on a dry state, bl_pbl_physics 0): the coefficients; the u, v, w and
    theta rows, each into its zeroed held slot and then its open strips
    cleared; the vertical rows and surface terms; every slot's open strips
    cleared; _restore_open_ring_surface_fluxes (a no-op here: the binding
    takes periodic or specified axes); then the coupling as diff_opt 1."""
    m = geom.mix
    for op, outs, act, pas in _kc_names(m):
        snap(tuple(outs.values()))
        traj = {**{k: x[v] for k, v in act.items()}, **{n: x[n] for n in pas}}
        _KC_OPS[op][0]({k: x[v] for k, v in outs.items()}, {k: dx[v] for k, v in outs.items()},
                       traj, {k: dx[v] for k, v in act.items()}, m.kc, nl=True)
    # each row adds into its own slot, zeroed above: no snapshot needed
    pairs = [(kind, g, b, True) for kind, g, b in m.rows]
    pairs += [(kind, g, b, False) for kind, g, b in m.vrows]
    for kind, g, b, clear in pairs:
        act = h2_active(g)
        h2_tl({"tend": x[_HELD[kind]]}, {"tend": dx[_HELD[kind]]}, _bound(x, b),
              {a: dx[v] for a, v in b if a in act}, g, nl=True)
        if clear:
            for d in ((x, dx) if tl else (x,)):
                _strip(d, (_HELD[kind],), geom)
    for d in ((x, dx) if tl else (x,)):
        _strip(d, tuple(_HELD.values()), geom)
    if not geom.specified and geom.couple:
        snap(tuple(_HELD[k] for k in geom.couple))
        _couple(x, geom)
        if tl:
            _couple(dx, geom, x)


def _mix2_ad(ax, back, peek, geom):
    """Adjoint of :func:`_mix2_tl`."""
    m = geom.mix
    if not geom.specified and geom.couple:
        base = back()
        _couple(ax, geom, base)
    _strip(ax, tuple(_HELD.values()), geom)
    base = peek()   # the rows read no slot they write
    pairs = [(kind, g, b, True) for kind, g, b in m.rows]
    pairs += [(kind, g, b, False) for kind, g, b in m.vrows]
    for kind, g, b, clear in reversed(pairs):
        if clear:
            _strip(ax, (_HELD[kind],), geom)
        act = h2_active(g)
        h2_ad({"tend": ax[_HELD[kind]]}, _bound(base, b),
              {a: ax[v] for a, v in b if a in act}, g)
    for op, outs, act, pas in reversed(_kc_names(m)):
        base = back()
        traj = {**{k: base[v] for k, v in act.items()}, **{n: base[n] for n in pas}}
        _KC_OPS[op][1]({k: ax[v] for k, v in outs.items()}, traj,
                       {k: ax[v] for k, v in act.items()}, m.kc)


def _held_zeroed(geom):
    """The slots prepare_fixed_tendencies zeroes: the held rows and, with
    the moist mixing, the species' held sources."""
    return (tuple(_HELD[k] for k in geom.held)
            + (geom.mix.moist if geom.mix is not None else ()))


def _held_start_ad(ax, back, peek, geom):
    """Adjoint of :func:`_held_start_tl`."""
    m = geom.mix
    if m is not None and m.diff2:
        _mix2_ad(ax, back, peek, geom)
        base = back()
        df_ad({n: ax[n] for n in DF_STATE}, _df_traj(base), _df_act(ax), m.df)
    elif m is not None:
        if not geom.specified and geom.couple:
            base = back()
            _couple(ax, geom, base)
        base = peek()   # the rows read no slot they write: no snapshot
        for row, g in reversed(m.rows):
            kind = row[0]
            hd_ad({"tend": ax[_HELD[kind]]}, _hd_traj(base, row, g), _hd_act(ax, row), g)
        back()
        ax["mup0"] += ax["smag_mut"]
        ax["smag_mut"][...] = 0
        base = back()
        km_ad(_km_out(ax), {**{n: base[n] for n in DF_STATE}, "msft": base["msft"]},
              {n: ax[n] for n in DF_STATE}, m.km)
        base = back()
        df_ad({n: ax[n] for n in DF_STATE}, _df_traj(base), _df_act(ax), m.df)
    if geom.held:
        back()
        for n in _held_zeroed(geom):
            ax[n][...] = 0
    if m is not None and m.prp is not None:
        base = back()
        a = _view(ax, PRP_ALIAS)
        calc_p_rho_phi_ad(a, _view(base, PRP_ALIAS), a, m.prp)


def _df_act(d):
    """The deformation's active inputs: the time-t winds and geopotential
    (_wrf_smag_grid_args with time_t)."""
    return {"u": d["u0"], "v": d["v0"], "php": d["php0"]}


def _df_traj(x):
    return {**_df_act(x), **{n: x[n] for n in DF_PASSIVE}}


def _km_out(d):
    return {"xkmh": d["smag_km"], "xkhh": d["smag_kh"]}


def _hd_act(d, row):
    _kind, _row, field, xk, _c1, _c2, mut = row
    return {"field": d[field], "xk": d[xk], "mut": d[mut]}


def _hd_traj(x, row, g):
    _kind, _row, _field, _xk, c1, c2, _mut = row
    out = {**_hd_act(x, row), "c1": x[c1], "c2": x[c2], "msft": x["msft"],
           "msfu": x["msfu"], "msfv": x["msfv"]}
    if g.perturb:
        out["base"] = x["diff1_theta_initial"]
    return out


def _lbc_stage_tl(x, dx, geom, spec, snap, tl):
    """A stage's held and specified work after the slow tendencies, in
    dycore._step_body's strict order: on a specified domain's stage 1 the
    held relaxation of u, v and theta, folded into the held rows (times
    the map factor) and the rows' coupling (fold_lateral_relaxation);
    every stage the held rows' add (add_fixed_dry_tendencies), then the
    boundary rows (_apply_state_lateral_boundaries_wrf)."""
    if geom.specified and spec.rk_step == 1:
        for kind, g in geom.capture:
            snap(("lbc_held_" + kind,))
            st, traj, act = _lb_views(x, dx, kind)
            lbc_tl(st(x), st(dx), traj, act(dx), g, nl=True)
        snap(tuple(dict.fromkeys(_HELD[k] for k in geom.fold + geom.couple)))
        for d in ((x, dx) if tl else (x,)):   # the msf factors are passive
            for kind in geom.fold:
                scale = _FOLD_SCALE[kind]
                h = d["lbc_held_" + kind]
                d[_HELD[kind]] += h * x[scale][None] if (geom.fold_msf and scale) else h
            _couple(d, geom, x)
    if geom.held:
        snap(tuple(_HELD_TEND[k] for k in geom.held))
        for d in ((x, dx) if tl else (x,)):
            for kind in geom.held:
                d[_HELD_TEND[kind]] += d[_HELD[kind]]
    if geom.specified:
        for kind, g in spec.lbc:
            snap((_LBC_TEND[kind], "lbc_held_" + kind))
            st, traj, act = _lb_views(x, dx, kind)
            lbc_tl(st(x), st(dx), traj, act(dx), g, nl=True)


def _lbc_stage_ad(ax, back, geom, spec):
    """Adjoint of :func:`_lbc_stage_tl` (``back`` pops one snapshot and
    returns the base before that operation)."""
    if geom.specified:
        for kind, g in reversed(spec.lbc):
            base = back()
            st, traj, act = _lb_views(base, None, kind)
            lbc_ad(st(ax), traj, _lb_adj(ax, kind), g)
    if geom.held:
        back()
        for kind in geom.held:
            ax[_HELD[kind]] += ax[_HELD_TEND[kind]]
    if geom.specified and spec.rk_step == 1:
        base = back()
        _couple(ax, geom, base)
        for kind in reversed(geom.fold):
            scale = _FOLD_SCALE[kind]
            a = ax[_HELD[kind]]
            ax["lbc_held_" + kind] += (a * base[scale][None] if (geom.fold_msf and scale)
                                       else a)
        for kind, g in reversed(geom.capture):
            base = back()
            st, traj, act = _lb_views(base, None, kind)
            lbc_ad(st(ax), traj, _lb_adj(ax, kind), g)


def _lb_adj(ax, kind):
    """lbc_ad's active adjoints on ``ax``: the mu row's source is never
    read, so its adjoint lands in the unread scratch."""
    out = {"src": _3d(ax[_LBC_SRC[kind]]), "mup": ax[_LBC_MU.get(kind, "mup")]}
    out.update({t: ax[f"lbc_{kind}_{t}"] for t in LB_TABLES})
    return out


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

    if not geom.specified:
        back()                                                    # periodic alias
        ax["u"][..., 0] += ax["u"][..., -1]
        ax["u"][..., -1] = 0
        ax["v"][..., 0, :] += ax["v"][..., -1, :]
        ax["v"][..., -1, :] = 0
    base = back()
    set_w_surface_ad(ax, base, ax, geom.sw)
    if geom.specified:
        for kind, g in reversed(geom.final):                      # spec_bdy_final
            base = back()
            st, traj, act = _lf_views(base, kind)
            lbf_ad(st(ax), traj, act(ax), g)
        back()                                                    # the halo copy
        ax["mup"] += ax[HALO]
        ax[HALO][...] = 0
    base = back()
    a = _view(ax, PRP_ALIAS)
    calc_p_rho_phi_ad(a, _view(base, PRP_ALIAS), a, geom.end_prp)
    for spec in reversed(geom.stages):
        if geom.moist:
            _scalars_ad(ax, back, spec)
        for row, g in reversed(spec.mix_q):                       # moist mixing
            hd_ad({"tend": ax[_q_slot(row[0])]}, _hd_traj(cur, row, g), _hd_act(ax, row), g)
        base = back()                                             # sf
        small_step_finish_ad(ax, {n: base[n] for n in SF_STATE}, base, ax, spec.ss)
        if not geom.specified:
            back()                                                # captured muts
            ax["mup"] += ax["muts"]
            ax["mu_pp"] += ax["muts"]
            ax["muts"][...] = 0
        for i in reversed(range(spec.nsub)):
            if spec.sf is not None:                               # accumulate_sumflux
                back()
                sumflux_accumulate_ad(ax, None, ax, spec.sf)
            if spec.sph is not None:
                base = back()
                sph_ad(ax, base, base, ax, spec.sph)
            cur = acoustic_substep_ad(_acoustic(ax), _acoustic(cur), tape,
                                      spec.first if i == 0 else spec.later)
            cur.update(fnm=x["fnm"], fnp=x["fnp"], msfv=x["msfv"])
        if spec.sf is not None:                                   # zero_sumflux
            back()
            sumflux_zero_ad(ax, spec.sf)
        base = back()                                             # cw
        calc_coef_w_ad(ax, base, ax, spec.cw)
        back()                                                    # small_step_prep's muts
        src = "mup" if spec.rk_step == 1 else "mup0"
        ax[src] += ax["muts"]
        ax["muts"][...] = 0
        base = back()                                             # si
        small_step_init_ad(_inv(ax), _inv(base), _inv(ax), spec.ss)
        if geom.specified or geom.held:
            _lbc_stage_ad(ax, back, geom, spec)
        for _, _, ad, alias, g, _ in reversed(_op_table(spec)):
            base = back()
            a = _view(ax, alias)
            ad(a, _view(base, alias), a, g)
        back()                                                    # zero_tendencies
        for n in TENDENCIES:
            ax[n][...] = 0
    _held_start_ad(ax, back, lambda: cur, geom)
    back()                                                        # _save_time_t
    for n in PROGNOSTICS + (SPECIES if geom.moist else ()):
        ax[n] += ax[n + "0"]
        ax[n + "0"][...] = 0
    assert not tape, len(tape)
    return cur
