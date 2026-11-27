"""Correct mode: the switch between matched arithmetic and corrected arithmetic.

WHAT THIS IS.  ``physics.mode`` (the key shared by the engine, woof hex and
the MPAS namelist, where it is ``config_physics_mode``) takes two values:

``"correct"``
    THE DEFAULT from 2.9.0 (fixed means default).  Every registered
    correction below is applied, unless the run names it in
    ``physics.correct_skip`` (the per-correction A/B arm).
``"exact"``
    The matched arithmetic: WRF v4.6.1 / MPAS-A v8.4.2 and its pinned
    MMM-physics, defects included, byte-identical to the code before each
    correction existed.  Every strict, parity and identity proof names it
    explicitly, because the default no longer is.

Strict WRF mode (``GPUWM_WRF_EXACT=1``, :mod:`woof.wrf_exact`) is a 0 ULP
proof of WRF's own arithmetic, so it FORCES exact: a strict process resolves
an unnamed mode to exact and records ``forced_by`` in the receipt, and it
refuses a named ``correct`` (a strict label on corrected arithmetic is a proof
of nothing).  Like :mod:`woof.wrf_exact`, this is read once, at import.

Each correction is one or more mechanism-named :class:`woof.config.RunConfig`
keys (the same pattern ``woof.physics_mode``'s divergence ledger uses), so
a WOOF-ArWen TOML can also set one key alone.  The RunConfig defaults of those
keys are :data:`CORRECT_KEYS_DEFAULT`: on in a default process, off in a
strict one.  The mode is a resolver onto those keys and nothing else, and it
authors EVERY key in both modes, so a resolution never depends on a default.

WHY A REGISTRY.  A run in correct mode must say which findings it changed,
and an A/B arm must be able to drop one correction while keeping the others.
Both need the list to be data.

THE FINDINGS (all class A: unintended, the code contradicts itself):

WB1  Thompson's cloud-water sedimentation computes the flux out of the
     lowest level and books it nowhere, so that water leaves the model's
     budget (0.12 kg m-2 over 60 steps of fog on the MPAS test columns).
     Correct mode books it in its own total, CLOUDNC; rain and the
     atmosphere are unchanged (woof/core/thompson_cloud_booking.py).
     Classic (mp 8) and aerosol-aware (mp 28, the HRRR suite's scheme)
     Thompson both carry it, and both are corrected.

WB4  New Tiedtke water tendencies are per unit of the scheme's moist-air
     frame mass (dp/g) but are added to mixing ratios on fixed dry mass;
     about 1% of convective precipitation is not taken out of the column.
WB5  WSM6 sedimentation books the below-ground part of the straddling cell
     at the flat cell mean while the remap that keeps the rest aloft uses
     the sloped piecewise-linear profile; the column + surface water budget
     is off by either sign (0.0024 kg m-2 per call measured).  WDM6 carries
     the same routine (mp_wdm6.F nislfv_rain_plm6/plmr) and the same defect;
     its key is the second key of WB5.  WSM3, WSM5 and WSM7 carry their own
     copies (module_mp_wsm3.F, module_mp_wsm5.F, module_mp_wsm7.F
     nislfv_rain_plm) and the same defect (up to 0.0082 kg m-2 per call on
     the same columns); from 2.9.1 their keys are WB5's third to fifth.

ML1  MYNN's BouLac downward length (boulac_length, used by mym_length
     CASE(1), bl_mynn_mixlength = 1) solves for the distance a parcel sinks
     into the layer where its energy runs out.  Where that layer has exactly
     constant theta (bbb = 0) the linear branch divides by
     beta*(theta(izz) - theta(iz)) where the bbb -> 0 limit of the quadratic
     branch is beta*(theta(iz) - theta(izz)): the last step comes out negative
     and the downward length is short by twice it.  NCAR/MYNN-EDMF fixed it in
     399f50f (PR 94, issue 93; v6.1.1); correct mode compiles the same sign
     into the wrf_461 and gsd_41 length kernels (kernels/mynn_pbl.cu,
     MYNN_BOULAC_DOWN_SIGN).  The WRF 4.7.1 and 4.8.0 MYNN sources carry it
     too and are listed as uncorrected (:data:`KEY_UNREAD`).  WRF's own
     BouLac (8) and k-epsilon (17) PBL schemes carry the same loop with the
     same sign (module_bl_boulac.F:606, module_bl_keps.F boulac_length;
     fixed by WRF in wrf-model/WRF PR 2431, release-v4.8.1, not yet
     released); ML1's second key, ``pbl_boulac_down_sign``, compiles the
     same sign into both kernels (kernels/boulac.cu, kernels/keps.cu,
     BOULAC_DOWN_SIGN).

HN1  A hydrometeor mass handed from one model to another (prep from a
     hex or other parent, ``woof downscale``, a specified ring) arrives
     with number concentration 0, which a two-moment scheme sizes at its
     slope limit (Morrison: 2.8 mm rain; empty first maps, NaN reflectivity,
     40-55 dBZ edge strips).  Correct mode diagnoses the number from the mass
     with the child scheme's relations and forces a ring number back with
     its mass (woof/ingest/handover_numbers.py).  Exact mode keeps
     real.exe's zeros and WRF's tendency-only scalar ring.

MY1  WRF 4.6.1's MYNN (and every MYNN-EDMF before NCAR/MYNN-EDMF f3c9d9e,
     4 Nov 2026) limits a negative surface vapour flux with
     ``max(qvflux, min(0.9*sqv - 1e-8, 0)/dtz)``; for any sqv above 1.1e-8
     the MIN is 0, so dew and frost are deleted, not limited, which is too
     much moisture left in the lowest layer under a calm clear night (too
     much shallow nocturnal fog, upstream says).  The key is
     ``mynn_fixes_v61``: the upstream MYNN v6.1 fixes that apply to the WRF
     4.6.1 lineage (woof/core/mynn_pbl_v61.py: this dew limit, the surface
     density's virtual-temperature factor, ground-Exner surface theta, the
     PBL-height search start, and no surface-layer cold-start reset on a
     cycled start), read by the wrf_461 MYNN generation only.  WRF 4.7.1's
     and 4.8.0's MYNN (bl_mynn_version wrf_471, wrf_480) carry the same
     limiter and run it as WRF ships it, so a correct-mode run on them lists
     MY1 as uncorrected; GSD MYNN 4.1 (gsd_41) adds the flux with its sign
     and does not carry it.

FQ1-FQ12 are the fork's quirk ledger on the WRF route (woof.quirk_corrections,
``woof_quirk_corrections``): corrections of WRF 4.7.1 arithmetic that
contradicts itself, each with its own key.  Like the findings above they are
on in correct mode and off in exact mode (and in a strict WRF process).

FQ1  Grell-Freitas (cu_physics = 3) deep rain also counts the downdraft
     evaporation it returns to vapour; the key books it once
     (``gf_deep_rain_evaporation_booking``).
FQ2  YSU (bl_pbl_physics = 1) adds HFX as a theta increment and converts the
     result with pi, so the column receives HFX*pi; the key takes HFX/pi(1)
     (``ysu_surface_heat_exner``).  With BEP/BEM (sf_urban_physics = 2, 3)
     YSU takes its heat from the urban sources and the key is not read.
FQ3  The wrf_461 MYNN surface layer (sf_sfclay_physics = 5) integrates the
     2 m / 10 m heat profile from z0/L while its log terms start at zt; the
     key takes zt/L (``mynn_sfclay_psih_zolzt``).
FQ4  Noah (sf_surface_physics = 2) raises soil moisture to its 0.02 floor
     without taking the water from anywhere; the key books it from the layer
     below or RUNOFF3 (``noah_smc_floor_booking``).  A WRF-Hydro coupling
     does not read it.
FQ5  Thompson (mp_physics = 8, 28, 38) re-pairs the rain-snow collection
     only above 0 C wet bulb; below it the graupel gain keeps its
     pre-limiter value while the rain and snow limiters shrink their losses,
     so the exchange creates graupel.  The key sets prg_rcs = -(prr_rcs +
     prs_rcs) after the limiters, floored at zero (snow then takes only
     what rain lost) (``thompson_rain_snow_closure``).
FQ6  WRF's RRTMG longwave (ra_lw_physics = 4 with ra_rrtmg_variant =
     rrtmg_legacy) matches its standard-profile temperature to the column
     starting at kte+1, which overwrites the model-top interface temperature
     and the top layer's; the key keeps both and anchors the profile at the
     top interface (``rrtmg_lw_model_top``).  The RTE+RRTMGP adapter does not
     carry it.
FQ7  Thompson's cloud-water mass/number balance forms the cloud slope lamc
     from the entry cloud mass rc(k) rather than the updated xrc (WRF
     4.7.1 :3003); the key forms it from xrc, which changes the cloud
     number tendency of the aerosol-aware schemes (mp_physics = 28, 38;
     the classic scheme's fixed droplet number never leaves the call)
     (``thompson_lamc_updated_rc``).
FQ8  Noah (sf_surface_physics = 2) on a land-ice point keeps the liquid rain
     in PRCPF and never reads it, and books only the snowmelt as runoff, so
     the rain leaves the water budget; the key books it as surface runoff
     (``noah_land_ice_rain_runoff``).  A WRF-Hydro coupling carries it (the
     glacial arm writes SFCRUNOFF, never the routing planes).
FQ9  Noah (sf_surface_physics = 2) on a snow-covered land point computes the
     series (harmonic) snow-soil conductivity DF1H and never reads it; the
     soil heat flux blends the thickness-weighted arithmetic mean DF1A, a
     parallel-flow mean of layers the heat crosses in series, which
     overstates the conductivity of a thin snowpack over soil.  The key
     blends DF1H (``noah_snow_series_conductivity``); a snow-free column and
     a fully snow-covered one (SNCOVR > 0.97, to rounding) are unchanged.
FQ10 The wrf_461 MYNN surface layer (sf_sfclay_physics = 5) caps its 2 m
     mixing ratio at 1.05*qv(1) at every point, water included, where the
     surface driver's own cap of that limit runs over land only; the key
     caps over land only (``mynn_sfclay_q2_land_cap``).
FQ11 The Kain-Fritsch family (cu_physics = 1, 10, 11) caps the downdraft
     with DDINC = TRPPT/TDER and forms the rain flux TRPPT - TDER with no
     floor; in binary32 TDER*DDINC can round above TRPPT, so RAINCV and
     RAINC go a few ulp negative.  The key floors the flux at zero, the
     float64 answer (``kf_capped_downdraft_rain_floor``).
FQ12 WRF's compute_diff_metrics (diff_opt = 1, 2) fills a periodic axis's
     seam slope (zx at ids/ide, zy at jds/jde) for levels 1..ktf only, so the
     top w level at the seam keeps its cold zero and the top mass level's
     deformation reads it; the key takes the true slope there
     (``diff_metrics_seam_slope``).  Inert on a bounded axis.

WHICH SCHEMES CARRY EACH FINDING is data too (:attr:`Correction.carriers`),
separate from which schemes a key corrects (:data:`KEY_SCOPE`).  A domain
whose schemes carry a finding that no key in scope corrects is listed as
``uncorrected`` in the run receipt and named in one run-log line
(:func:`uncorrected`, :func:`uncorrected_notice`), so a receipt can never read
as a finding applied to a scheme that still runs the defect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

#: ``physics.mode`` values.  Anything else is refused by name: a typo must
#: never silently mean either mode.
PHYSICS_MODE_EXACT = "exact"
PHYSICS_MODE_CORRECT = "correct"
PHYSICS_MODES = (PHYSICS_MODE_EXACT, PHYSICS_MODE_CORRECT)

#: What forces exact in a strict process, as the receipt names it.
STRICT_FORCED_BY = "GPUWM_WRF_EXACT=1"


def _strict() -> bool:
    from woof import wrf_exact
    return bool(wrf_exact.ENABLED)


#: Whether this process is a strict WRF proof (read once, at import, as
#: :mod:`woof.wrf_exact` itself is).
STRICT = _strict()

#: The mode an unnamed ``physics.mode`` resolves to in this process.
PHYSICS_MODE_DEFAULT = PHYSICS_MODE_EXACT if STRICT else PHYSICS_MODE_CORRECT

#: The RunConfig default of every correction key in this process.
CORRECT_KEYS_DEFAULT = PHYSICS_MODE_DEFAULT == PHYSICS_MODE_CORRECT


@dataclass(frozen=True)
class Correction:
    """One registered correction: its finding id and the keys it sets."""

    finding: str
    keys: tuple[str, ...]
    summary: str
    #: ``((selector, values), ...)``: the scheme selections whose arithmetic
    #: carries the defect, whether or not a key corrects it there.
    carriers: tuple[tuple[str, tuple[int, ...]], ...] = ()

    @property
    def key(self) -> str:
        """The first (historically the only) key, kept for callers that
        read one key per finding."""
        return self.keys[0]


#: The Thompson schemes whose cloud number leaves the call (aerosol-aware 28
#: and hail-aware 38); RCON (29) is in THOMPSON_AEROSOL_FAMILY but not here.
#: RunConfig.thompson_lamc_updated_rc (FQ7) is read under these.
THOMPSON_CLOUD_BALANCE_MP = (28, 38)

#: Every correction this engine carries.  Order is the receipt order.
CORRECTIONS: tuple[Correction, ...] = (
    Correction(
        finding="WB1",
        keys=("thompson_cloud_surface_booking",),
        summary=(
            "Thompson cloud water settled out of the lowest level booked in "
            "its own accumulated total (CLOUDNC) instead of vanishing; rain "
            "and the atmosphere unchanged"),
        # Classic Thompson (8) and aerosol-aware Thompson (28) both drop the
        # cloud water their fallout takes out of the lowest level.
        carriers=(("mp_physics", (8, 28, 38)),),
    ),
    Correction(
        finding="WB4",
        keys=("ntiedtke_dry_mass_coupling",),
        summary=(
            "New Tiedtke water tendencies mapped from the scheme's moist "
            "frame mass dp/g = m_dry*(1+qv) onto dry mass (x(1+qv)); vapour "
            "forcing converted mixing ratio -> specific humidity "
            "(/(1+qv)^2)"),
        carriers=(("cu_physics", (16,)),),
    ),
    Correction(
        finding="WB5",
        keys=("wsm6_conservative_surface", "wdm6_conservative_surface",
              "wsm3_conservative_surface", "wsm5_conservative_surface",
              "wsm7_conservative_surface"),
        summary=(
            "WSM3, WSM5, WSM6, WSM7 and WDM6 below-ground partial cell booked "
            "with the remap's own piecewise-linear profile instead of the "
            "flat cell mean"),
        # Every scheme with a port of WRF's nislfv_rain_plm.  WSM6R (106)
        # falls out by explicit flux, and TEMPO books the straddling cell
        # with its own profile already: neither carries it.
        carriers=(("mp_physics", (3, 4, 6, 16, 24)),),
    ),
    Correction(
        finding="HN1",
        keys=("hydrometeor_handover_numbers",),
        summary=(
            "Hydrometeor number handed over with its mass: a number at zero "
            "(or, for Morrison, far under its limiter) beside a handed-over "
            "mass is diagnosed from the mass with the child scheme's "
            "relations, and a specified ring forces the number back with "
            "its mass"),
        # Thompson (8, 28): the ring force-back and the downscale closure;
        # Morrison (10): the prep, downscale and ring closure.
        carriers=(("mp_physics", (8, 10, 28)),),
    ),
    Correction(
        finding="SN1",
        keys=("noahmp_melt_reset_esat",),
        summary=(
            "Noah-MP land snowmelt reset (snow deeper than 5 cm, ground "
            "solved above freezing, TG set to 273.16 K) evaluates the ground "
            "evaporation, and bare-ground QSFC, at the reset temperature, as "
            "WRF's own glacier reset does, instead of at the discarded warmer "
            "TG; that stale value overstated evaporation from sunlit melting "
            "snow and drove the vegetated 2 m mixing ratio to zero or below"),
        carriers=(("sf_surface_physics", (4,)),),
    ),
    Correction(
        finding="MY1",
        keys=("mynn_fixes_v61",),
        summary=(
            "MYNN dew and frost limited, not deleted: a downward surface "
            "vapour flux is kept unless it would take the lowest layer below "
            "1e-8 kg/kg (NCAR/MYNN-EDMF v6.1.1 f3c9d9e), with the other "
            "upstream MYNN v6.1 fixes of the WRF 4.6.1 lineage (surface "
            "density virtual-temperature factor, ground-Exner surface theta, "
            "PBL-height search start, no surface-layer cold-start reset on a "
            "cycled start)"),
        # Every MYNN generation but GSD 4.1 limits the flux this way
        # (CARRIER_GENERATIONS below).
        carriers=(("bl_pbl_physics", (5,)),),
    ),
    Correction(
        finding="ML1",
        keys=("mynn_boulac_down_sign", "pbl_boulac_down_sign"),
        summary=(
            "BouLac downward mixing length (MYNN, the BouLac PBL and the "
            "k-epsilon PBL): in a layer of exactly constant theta the last "
            "step of the sinking parcel takes the sign of the bbb -> 0 limit, "
            "(e - zdo_sup)/(beta*(theta(iz) - theta(izz))), as NCAR/MYNN-EDMF "
            "v6.1.1 (399f50f) and wrf-model/WRF PR 2431 do, instead of WRF's "
            "opposite sign, which made it negative and the downward length "
            "short by twice it"),
        # MYNN (5) through bl_mynn_mixlength = 1 (first key); WRF's BouLac
        # (8) and k-epsilon (17) PBL schemes carry the same downward loop
        # with the same sign (second key).
        carriers=(("bl_pbl_physics", (5, 8, 17)),),
    ),
    # The fork's quirk ledger on the WRF route (woof.quirk_corrections).
    Correction(
        finding="FQ1",
        keys=("gf_deep_rain_evaporation_booking",),
        summary=(
            "Grell-Freitas deep rain no longer counts the downdraft "
            "evaporation it returns to vapour"),
        carriers=(("cu_physics", (3,)),),
    ),
    Correction(
        finding="FQ2",
        keys=("ysu_surface_heat_exner",),
        summary=(
            "YSU takes the surface heat flux as HFX/pi(1), so the column "
            "gets HFX rather than HFX*pi"),
        carriers=(("bl_pbl_physics", (1,)),),
    ),
    Correction(
        finding="FQ3",
        keys=("mynn_sfclay_psih_zolzt",),
        summary=(
            "the MYNN surface layer's 2 m / 10 m heat profile is integrated "
            "from zt/L, as its log term is"),
        carriers=(("sf_sfclay_physics", (5,)),),
    ),
    Correction(
        finding="FQ4",
        keys=("noah_smc_floor_booking",),
        summary=(
            "Noah's 0.02 soil-moisture floor takes the water it adds from "
            "the layer below, or RUNOFF3"),
        carriers=(("sf_surface_physics", (2,)),),
    ),
    Correction(
        finding="FQ5",
        keys=("thompson_rain_snow_closure",),
        summary=(
            "Thompson's rain-snow collection below 0 C makes graupel only "
            "from what rain and snow lose after their limiters (floored at "
            "zero)"),
        carriers=(("mp_physics", (8, 28, 38)),),
    ),
    Correction(
        finding="FQ6",
        keys=("rrtmg_lw_model_top",),
        summary=(
            "WRF's RRTMG longwave keeps the model-top interface and top "
            "layer temperatures when it matches the standard profile above "
            "the model top"),
        carriers=(("ra_lw_physics", (4,)),),
    ),
    Correction(
        finding="FQ7",
        keys=("thompson_lamc_updated_rc",),
        summary=(
            "Thompson's cloud-water mass/number balance forms the cloud "
            "slope from the updated cloud mass, not the entry mass"),
        carriers=(("mp_physics", THOMPSON_CLOUD_BALANCE_MP),),
    ),
    Correction(
        finding="FQ8",
        keys=("noah_land_ice_rain_runoff",),
        summary=(
            "Noah books the liquid rain that falls on land ice as surface "
            "runoff, where WRF drops it"),
        carriers=(("sf_surface_physics", (2,)),),
    ),
    Correction(
        finding="FQ9",
        keys=("noah_snow_series_conductivity",),
        summary=(
            "Noah blends the series snow-soil conductivity into the soil "
            "heat flux, where WRF computes it and uses the arithmetic mean"),
        carriers=(("sf_surface_physics", (2,)),),
    ),
    Correction(
        finding="FQ10",
        keys=("mynn_sfclay_q2_land_cap",),
        summary=(
            "the MYNN surface layer caps its 2 m mixing ratio at 1.05 times "
            "the lowest level over land only, as the surface driver does, "
            "not over water"),
        carriers=(("sf_sfclay_physics", (5,)),),
    ),
    Correction(
        finding="FQ11",
        keys=("kf_capped_downdraft_rain_floor",),
        summary=(
            "Kain-Fritsch, KF-CuP and MSKF floor the capped downdraft's "
            "precipitation flux at zero, so convective rain is never a few "
            "ulp negative"),
        carriers=(("cu_physics", (1, 10, 11)),),
    ),
    Correction(
        finding="FQ12",
        keys=("diff_metrics_seam_slope",),
        summary=(
            "the diffusion metrics take the true terrain slope at a periodic "
            "axis's seam on the top w level, where WRF keeps a cold zero"),
        carriers=(("diff_opt", (1, 2)),),
    ),
)

#: Finding -> ``(selector, absent value, values)``: a carrier that is a
#: scheme generation, not a scheme number.  The finding is carried only when
#: the run's generation selector is one of ``values`` (an absent selector
#: reads as ``absent value``, its RunConfig default, since the checkpoint
#: echo drops it there).
CARRIER_GENERATIONS: Mapping[str, tuple[str, str, tuple[str, ...]]] = {
    # MY1: wrf_461 (kernels/mynn_pbl.cu), wrf_471 and wrf_480
    # (data/mynn471, data/mynn480 model.cu) all hold the limiter;
    # gsd_41 adds dtz*flqv with its sign (kernels/mynn_pbl.cu MYNN_GSD41).
    "MY1": ("bl_mynn_version", "wrf_461", ("wrf_461", "wrf_471", "wrf_480")),
    # FQ3: the gsl_wrf39 surface layer integrates from z0/L throughout.
    "FQ3": ("mynn_sfclay_variant", "wrf_461", ("wrf_461",)),
    # FQ10: the gsl_wrf39 surface layer is another source.
    "FQ10": ("mynn_sfclay_variant", "wrf_461", ("wrf_461",)),
    # FQ6: only the legacy RRTMG longwave (rrtmg_legacy_prep) holds the
    # profile match; RTE+RRTMGP is another implementation.
    "FQ6": ("ra_rrtmg_variant", "rte-rrtmgp", ("rrtmg_legacy",)),
}

#: Key -> ``(selector, absent value, values)``: a key read only under one
#: scheme generation as well as under its :data:`KEY_SCOPE` scheme.
KEY_GENERATIONS: Mapping[str, tuple[str, str, tuple[str, ...]]] = {
    # The v6.1 fixes are marked edits of the wrf_461 unit; the other
    # generations are other sources and never read the key.
    "mynn_fixes_v61": ("bl_mynn_version", "wrf_461", ("wrf_461",)),
    "mynn_sfclay_psih_zolzt": ("mynn_sfclay_variant", "wrf_461", ("wrf_461",)),
    "rrtmg_lw_model_top": ("ra_rrtmg_variant", "rte-rrtmgp", ("rrtmg_legacy",)),
    "mynn_sfclay_q2_land_cap": ("mynn_sfclay_variant", "wrf_461", ("wrf_461",)),
}

#: Key -> ``(selector, values)``: the scheme selection under which the key is
#: read at all.  Outside it the key reaches nothing, so the checkpoint echo
#: and the configuration digest drop it (woof/io/restart.py) and a run whose
#: schemes never read it echoes and digests as it did before the key.
KEY_SCOPE: Mapping[str, tuple[str, tuple[int, ...]]] = {
    "ntiedtke_dry_mass_coupling": ("cu_physics", (16,)),
    "wsm6_conservative_surface": ("mp_physics", (6,)),
    "wdm6_conservative_surface": ("mp_physics", (16,)),
    "wsm3_conservative_surface": ("mp_physics", (3,)),
    "wsm5_conservative_surface": ("mp_physics", (4,)),
    "wsm7_conservative_surface": ("mp_physics", (24,)),
    # Classic (8), aerosol-aware (28), and graupel-hail (38) Thompson
    # all book settled cloud. Scope must include every writer so restart
    # allocates and restores the carried total before the first step.
    "thompson_cloud_surface_booking": ("mp_physics", (8, 28, 38)),
    # HN1: the schemes whose number moments the handover closure writes.
    "hydrometeor_handover_numbers": ("mp_physics", (8, 10, 28)),
    # SN1: VEGE_FLUX/BARE_FLUX (and GECROS's VEGE_FLUX) of Noah-MP.
    "noahmp_melt_reset_esat": ("sf_surface_physics", (4,)),
    # MY1: the MYNN PBL, at the wrf_461 generation (KEY_GENERATIONS).  The
    # surface-layer part (S1) runs only beside it.
    "mynn_fixes_v61": ("bl_pbl_physics", (5,)),
    # ML1: the two mixing-length kernels of the wrf_461/gsd_41 MYNN unit.
    "mynn_boulac_down_sign": ("bl_pbl_physics", (5,)),
    # ML1's second key: the BouLac (8) and k-epsilon (17) length kernels.
    "pbl_boulac_down_sign": ("bl_pbl_physics", (8, 17)),
    # FQ1-FQ12, the fork's quirk ledger (woof.quirk_corrections).
    "gf_deep_rain_evaporation_booking": ("cu_physics", (3,)),
    "ysu_surface_heat_exner": ("bl_pbl_physics", (1,)),
    "mynn_sfclay_psih_zolzt": ("sf_sfclay_physics", (5,)),
    "noah_smc_floor_booking": ("sf_surface_physics", (2,)),
    "thompson_rain_snow_closure": ("mp_physics", (8, 28, 38)),
    # FQ6: the longwave scheme (the aggregate ra_physics resolves to it,
    # _selector_int), at the legacy variant (KEY_GENERATIONS).
    "rrtmg_lw_model_top": ("ra_lw_physics", (4,)),
    # FQ7: the aerosol-aware Thompson, whose cloud number leaves the call.
    "thompson_lamc_updated_rc": ("mp_physics", THOMPSON_CLOUD_BALANCE_MP),
    # FQ8: Noah's glacial arm (sf_surface_physics = 2, land-ice points).
    "noah_land_ice_rain_runoff": ("sf_surface_physics", (2,)),
    # FQ9: Noah's land SFLX under snow (sf_surface_physics = 2).
    "noah_snow_series_conductivity": ("sf_surface_physics", (2,)),
    # FQ10: the MYNN surface layer, at wrf_461 (KEY_GENERATIONS), as FQ3.
    "mynn_sfclay_q2_land_cap": ("sf_sfclay_physics", (5,)),
    # FQ11: KF-eta (kf.cu), KF-CuP (kfcup.cu) and MSKF (mskf.cu); the old
    # KF (99) has no capped-downdraft subtraction.
    "kf_capped_downdraft_rain_floor": ("cu_physics", (1, 10, 11)),
    # FQ12: compute_diff_metrics runs under either diffusion option.
    "diff_metrics_seam_slope": ("diff_opt", (1, 2)),
}

#: A launch setting, not a RunConfig field, that :data:`KEY_UNREAD` reads
#: as a selector: whether this process couples WRF-Hydro
#: (woof.wrf_hydro_route.configured).  It is stated only in the values
#: :func:`launch_values` builds (what ran: run_reads, :func:`uncorrected`);
#: identities and echoes read the configuration alone.
WRF_HYDRO_SELECTOR = "wrf_hydro_coupled"

#: Key -> ``(selector, values)``: selections inside :data:`KEY_SCOPE` under
#: which the key is still not read, because that code path has no build of
#: the correction.  A finding whose carrier runs there is listed as
#: uncorrected, never as applied.
KEY_UNREAD: Mapping[str, tuple[str, tuple[object, ...]]] = {
    "mynn_boulac_down_sign": ("bl_mynn_version", ("wrf_471", "wrf_480")),
    # With BEP/BEM, WRF's YSU takes its surface heat from the urban sources
    # (bepswitch = 1), so the HFX/pi(1) correction would reach only the PBL
    # diagnosis: the key is not read there.
    "ysu_surface_heat_exner": ("sf_urban_physics", (2, 3)),
    # A WRF-Hydro coupling does not carry Noah's floor booking (its
    # drainage may go negative): woof.core.physics keeps FQ4 off there, so
    # the receipt lists the finding as uncorrected and the echo does not
    # bind the key.
    "noah_smc_floor_booking": (WRF_HYDRO_SELECTOR, (True,)),
}


def _key_unread(key: str, values) -> bool:
    unread = KEY_UNREAD.get(key)
    if unread is None:
        return False
    selector, names = unread
    value = (values.get(selector) if isinstance(values, Mapping)
             else getattr(values, selector, None))
    return value in names


def _generation_matches(rule, values: Mapping) -> bool:
    selector, absent, allowed = rule
    value = values.get(selector, absent)
    return (absent if value is None else value) in allowed


#: Selectors whose absent value is not 0: an echo omits diff_opt at its
#: default 2 (woof.io.restart reads an absent diff_opt as 2).
_ABSENT_SELECTOR = {"diff_opt": 2}

#: The fields :func:`_diff_metrics_seam_read` reads, beside diff_opt.
_SEAM_FIELDS = ("km_opt", "khdif", "kvdif", "bl_pbl_physics", "isfflx",
                "tke_heat_flux", "tke_drag_coefficient", "sf_sfclay_physics",
                "open_x", "open_y", "specified", "nested")


class _FieldView:
    """A mapping read as a RunConfig: an absent field is its default."""

    def __init__(self, values: Mapping):
        self._values = values

    def __getattr__(self, name):
        if name in self._values:
            return self._values[name]
        from woof.config import RunConfig
        return RunConfig.__dataclass_fields__[name].default


def _diff_metrics_seam_read(values: Mapping) -> bool:
    """FQ12 is read where the dycore builds WRF's mixing package
    (woof.config.wrf_mixing_package_active) on a periodic axis: a bounded
    axis never takes the seam slope."""
    from woof.config import wrf_mixing_package_active
    view = _FieldView(values)
    forced = bool(view.specified or view.nested)
    periodic = not (view.open_x or forced) or not (view.open_y or forced)
    return periodic and wrf_mixing_package_active(view)


#: Key -> predicate over the selection: a key read only where the predicate
#: holds, inside its :data:`KEY_SCOPE` scheme.
KEY_READ_WHEN = {"diff_metrics_seam_slope": _diff_metrics_seam_read}


def _selector_int(selector: str, values: Mapping) -> int:
    """``values[selector]`` as an int.  ``ra_lw_physics`` of -1 (or absent)
    is "not stated": the aggregate ``ra_physics`` names the longwave scheme
    (woof.config.radiation_scheme_ids)."""
    value = values.get(selector, _ABSENT_SELECTOR.get(selector, 0))
    if selector == "ra_lw_physics" and (value is None or value == -1):
        value = values.get("ra_physics", 0)
    return int(value or 0)


def key_in_scope(key: str, values: Mapping) -> bool:
    """Is correction ``key`` read under the scheme selection in ``values``
    (a configuration echo or ``dataclasses.asdict`` of a RunConfig)?"""
    selector, schemes = KEY_SCOPE[key]
    try:
        if _selector_int(selector, values) not in schemes:
            return False
    except (TypeError, ValueError):
        return True
    if _key_unread(key, values):
        return False
    read_when = KEY_READ_WHEN.get(key)
    if read_when is not None and not read_when(values):
        return False
    rule = KEY_GENERATIONS.get(key)
    return rule is None or _generation_matches(rule, values)


#: Every selector :func:`key_in_scope` and the carrier rules read.
SCOPE_SELECTORS: tuple[str, ...] = tuple(dict.fromkeys(
    [selector for selector, _ in KEY_SCOPE.values()]
    + [rule[0] for rule in KEY_GENERATIONS.values()]
    + [rule[0] for rule in CARRIER_GENERATIONS.values()]
    + [selector for selector, _ in KEY_UNREAD.values()]
    # the aggregate radiation selection stands for ra_lw_physics = -1
    + ["ra_physics"]
    # FQ12's read predicate (KEY_READ_WHEN)
    + list(_SEAM_FIELDS)))


def launch_values(cfg) -> dict:
    """:func:`_scope_values` with this launch's settings that decide what
    runs (:data:`WRF_HYDRO_SELECTOR`): for the questions "is the correction
    applied" (woof.quirk_corrections.run_reads) and the receipt
    (:func:`uncorrected`), never for a configuration's identity."""
    from woof.wrf_hydro_route import configured
    return {**_scope_values(cfg), WRF_HYDRO_SELECTOR: configured() is not None}


def _scope_values(cfg) -> dict:
    """The scope selectors of a RunConfig, or of a mapping, as a mapping
    (an absent one is left out, so it reads as its absent value)."""
    if isinstance(cfg, Mapping):
        return {name: cfg[name] for name in SCOPE_SELECTORS if name in cfg}
    return {name: getattr(cfg, name) for name in SCOPE_SELECTORS
            if hasattr(cfg, name)}


#: Correction keys that change no prognostic word, only what the run books
#: (WB1 adds the CLOUDNC total; rain and the atmosphere are bit-identical).
#: They are output-only switches like woof.checkpoint_identity's
#: CONFIG_DIAGNOSTIC_FIELDS, of which they are members: a resume may change
#: them, they bind no checkpoint, digest, fingerprint or prepared cache, and
#: they leave the checkpoint echo and the public documents, so the 2.9.0
#: default moves no Thompson run's identity.  The run receipt's correct_mode
#: block still records them.
OUTPUT_ONLY_KEYS: tuple[str, ...] = ("thompson_cloud_surface_booking",)

#: Correct-mode keys whose scheme already carried the correction before the
#: key existed, by the selector values that did: a checkpoint header,
#: fingerprint or document without the key ran it ON there, and OFF under
#: every other scheme.  FQ11: kf.cu floored the capped-downdraft rain flux
#: unconditionally from d547dc8c3 and mskf.cu from 695ddf87a, while
#: kfcup.cu ran WRF's bare subtraction until the key gated all three.
KEY_ABSENT_ON: Mapping[str, tuple[str, tuple[int, ...]]] = {
    "kf_capped_downdraft_rain_floor": ("cu_physics", (1, 11)),
}


def absent_value(key: str, values: Mapping) -> bool:
    """What a header, fingerprint or document without ``key`` ran under the
    scheme selection in ``values``: False (the matched arithmetic) except
    where :data:`KEY_ABSENT_ON` says the scheme carried the correction."""
    rule = KEY_ABSENT_ON.get(key)
    if rule is None:
        return False
    selector, schemes = rule
    try:
        return _selector_int(selector, values) in schemes
    except (TypeError, ValueError):
        return False


def key_echoes(key: str, values: Mapping) -> bool:
    """Is correction ``key`` written into a checkpoint echo, digest,
    fingerprint or experiment document?  Only where a selected scheme reads
    it and its value is not what absence means (:func:`absent_value`), so
    every record written before the key keeps its bytes."""
    if key in OUTPUT_ONLY_KEYS or not key_in_scope(key, values):
        return False
    absent = absent_value(key, values)
    return bool(values.get(key, absent)) != absent


CORRECTIONS_BY_FINDING: Mapping[str, Correction] = {
    c.finding: c for c in CORRECTIONS}
CORRECTION_KEYS: tuple[str, ...] = tuple(
    key for c in CORRECTIONS for key in c.keys)
#: Key -> the finding it belongs to.
FINDING_BY_KEY: Mapping[str, str] = {
    key: c.finding for c in CORRECTIONS for key in c.keys}


@dataclass(frozen=True)
class CorrectModeResolution:
    """The mode as one run resolved it."""

    mode: str
    skipped: tuple[str, ...]
    applied: tuple[str, ...]
    #: ``STRICT_FORCED_BY`` when a strict process forced exact; else None.
    forced_by: str | None = None
    #: The keys that are on, when read back from a configuration
    #: (:func:`resolution_of`); None means every key of every applied finding.
    keys_on: tuple[str, ...] | None = None

    @property
    def settings(self) -> dict[str, bool]:
        """RunConfig assignments for EVERY correction key: True for an
        applied finding, False otherwise.  Exact mode authors them all False,
        so the resolution is the same whatever the RunConfig defaults are."""
        return {key: c.finding in self.applied
                for c in CORRECTIONS for key in c.keys}

    def receipt(self) -> dict:
        keys_on = (self.keys_on if self.keys_on is not None else tuple(
            key for f in self.applied
            for key in CORRECTIONS_BY_FINDING[f].keys))
        receipt = {
            "schema": "gpuwm-correct-mode-v1",
            "physics.mode": self.mode,
            "physics.correct_skip": list(self.skipped),
            "applied_findings": list(self.applied),
            "keys": {key: True for key in keys_on},
        }
        if self.forced_by is not None:
            receipt["forced_by"] = self.forced_by
        return receipt


EXACT = CorrectModeResolution(mode=PHYSICS_MODE_EXACT, skipped=(), applied=())


def resolve(mode: str | None = None,
            correct_skip: Iterable[str] = (), *,
            source: str = "physics.mode") -> CorrectModeResolution:
    """Resolve ``physics.mode`` + ``physics.correct_skip``.

    ``mode=None`` is the default: correct, or exact (forced, recorded) in a
    strict process.  Refused (by name): an unknown mode, a skip list in exact
    mode (it would read as an arm that does something), an unknown finding
    id, a repeat, and a named ``correct`` in a strict process.
    """

    forced_by = None
    if mode is None:
        mode = PHYSICS_MODE_DEFAULT
        if STRICT:
            forced_by = STRICT_FORCED_BY
    if not isinstance(mode, str) or mode not in PHYSICS_MODES:
        raise ValueError(
            f"{source} = {mode!r} is not a physics mode; known modes: "
            f"{list(PHYSICS_MODES)}")
    if STRICT and mode == PHYSICS_MODE_CORRECT:
        raise ValueError(
            f"{source} = 'correct' in a strict WRF process "
            f"({STRICT_FORCED_BY}): strict mode is the 0 ULP proof of the "
            "matched arithmetic and forces physics.mode = 'exact', so a "
            "corrected run under it would carry a strict label it does not "
            "earn.  Name 'exact', or run without GPUWM_WRF_EXACT.")
    if isinstance(correct_skip, str):
        correct_skip = (correct_skip,)
    skipped: list[str] = []
    for finding in correct_skip:
        if finding not in CORRECTIONS_BY_FINDING:
            raise ValueError(
                f"physics.correct_skip names {finding!r}, which is not a "
                "correction this engine carries; known: "
                f"{sorted(CORRECTIONS_BY_FINDING)}")
        if finding in skipped:
            raise ValueError(
                f"physics.correct_skip names {finding!r} twice")
        skipped.append(finding)
    if mode == PHYSICS_MODE_EXACT:
        if skipped:
            raise ValueError(
                "physics.correct_skip is set while physics.mode = 'exact', "
                "which applies no correction; name the skip list on a "
                "correct-mode arm")
        return (EXACT if forced_by is None else CorrectModeResolution(
            mode=PHYSICS_MODE_EXACT, skipped=(), applied=(),
            forced_by=forced_by))
    applied = tuple(c.finding for c in CORRECTIONS if c.finding not in skipped)
    return CorrectModeResolution(
        mode=mode, skipped=tuple(skipped), applied=applied)


def resolution_of(cfg) -> CorrectModeResolution:
    """The mode a :class:`woof.config.RunConfig` (or any object carrying
    the keys) resolves to, read back from its keys: the run receipt's record
    of what actually ran.  A finding with any key on is applied (``keys``
    in the receipt names exactly which keys were on, so a single-key arm of
    a finding that spans several schemes reads as what it is); a finding
    with every key off is skipped.  No finding applied is exact, with
    ``forced_by`` in a strict process."""

    applied: list[str] = []
    skipped: list[str] = []
    keys_on: list[str] = []
    for c in CORRECTIONS:
        on = [key for key in c.keys if getattr(cfg, key, False) is True]
        if on:
            applied.append(c.finding)
            keys_on.extend(on)
        else:
            skipped.append(c.finding)
    if not applied:
        return (CorrectModeResolution(
            mode=PHYSICS_MODE_EXACT, skipped=(), applied=(),
            forced_by=STRICT_FORCED_BY) if STRICT else EXACT)
    return CorrectModeResolution(
        mode=PHYSICS_MODE_CORRECT, skipped=tuple(skipped),
        applied=tuple(applied), keys_on=tuple(keys_on))


def _selector_value(cfg, selector: str) -> int | None:
    values = (cfg if isinstance(cfg, Mapping) else
              {name: getattr(cfg, name, _ABSENT_SELECTOR.get(name, 0))
               for name in (selector, "ra_physics")})
    try:
        return _selector_int(selector, values)
    except (TypeError, ValueError):
        return None


def uncorrected(cfg) -> list[str]:
    """Findings this configuration's schemes carry
    (:attr:`Correction.carriers`) that no key in scope for those schemes
    corrects, while the run is in correct mode and the finding is not
    skipped.

    Empty for every shipped scheme once each carrier has a key; a finding
    listed here ran its defect although the mode is correct.  An exact run,
    and a finding named in ``physics.correct_skip``, list nothing: there the
    defect is what the run asked for."""

    resolution = resolution_of(cfg)
    if resolution.mode != PHYSICS_MODE_CORRECT:
        return []
    out: list[str] = []
    values = launch_values(cfg)
    for c in CORRECTIONS:
        if c.finding in resolution.skipped:
            continue
        generation = CARRIER_GENERATIONS.get(c.finding)
        if generation is not None and not _generation_matches(generation, values):
            continue
        # A carrier whose defect is reached only under a read predicate
        # (FQ12: a periodic axis) carries nothing outside it.
        if any(key in KEY_READ_WHEN and not KEY_READ_WHEN[key](values)
               for key in c.keys):
            continue
        for selector, schemes in c.carriers:
            value = _selector_value(cfg, selector)
            if value not in schemes:
                continue
            if not any(KEY_SCOPE[key][0] == selector
                       and value in KEY_SCOPE[key][1]
                       and key_in_scope(key, values) for key in c.keys):
                out.append(c.finding)
                break
    return out


def uncorrected_notice(configs: Mapping[int, object]) -> str | None:
    """One run-log line naming every correct-mode domain whose schemes carry
    a finding with no correction (:func:`uncorrected`), or None."""

    parts = []
    for grid_id, cfg in sorted(configs.items()):
        findings = uncorrected(cfg)
        if findings:
            parts.append(f"d{int(grid_id):02d} {'/'.join(findings)}")
    if not parts:
        return None
    return ("correct mode: no correction yet for " + ", ".join(parts)
            + " under the selected schemes; those findings run the matched "
            "WRF/MPAS arithmetic, and the run receipt lists them as "
            "uncorrected")


def in_effect(cfg) -> dict:
    """The corrections that reached this configuration's arithmetic or
    output: keys that are on AND read by its schemes (:data:`KEY_SCOPE`),
    with their findings.  A correct-mode run whose schemes read no key
    (no WSM3/5/6/7, WDM6, Thompson mp 8 or 28 or New Tiedtke) has none in
    effect and ran the matched arithmetic, though its mode is correct."""

    selectors = {selector: getattr(cfg, selector, 0)
                 for selector, _ in KEY_SCOPE.values()}
    selectors.update({name: value for name, value in _scope_values(cfg).items()
                      if name not in selectors})
    keys = [key for c in CORRECTIONS for key in c.keys
            if getattr(cfg, key, False) is True and key_in_scope(key, selectors)]
    return {"findings": [c.finding for c in CORRECTIONS
                         if any(key in keys for key in c.keys)],
            "keys": keys}


def run_receipt(configs: Mapping[int, object]) -> dict:
    """The run receipt's ``correct_mode`` block: per domain, the mode its
    RunConfig resolved to (:func:`resolution_of`), the corrections in
    effect under its schemes (:func:`in_effect`), plus this process's
    default.  Every forecast receipt carries it, so a run says which
    arithmetic it ran rather than leaving it to the defaults of the build
    that reads it later (and a strict run says exact was forced)."""

    return {
        "schema": "gpuwm-correct-mode-run-v2",
        "default": PHYSICS_MODE_DEFAULT,
        **({"forced_by": STRICT_FORCED_BY} if STRICT else {}),
        "domains": {str(int(grid_id)): _domain_receipt(cfg)
                    for grid_id, cfg in sorted(configs.items())},
    }


def _domain_receipt(cfg) -> dict:
    """One domain's block.  ``applied_findings`` names only the findings
    whose correction reached this domain's schemes (v2; v1 listed every
    finding with a key on, so a WSM5 domain read as WB1/WB4/WB5 applied),
    and ``uncorrected`` the findings its schemes carry with no correction."""

    effect = in_effect(cfg)
    block = {**resolution_of(cfg).receipt(), "in_effect": effect,
             "uncorrected": uncorrected(cfg)}
    block["applied_findings"] = [f for f in block["applied_findings"]
                                 if f in effect["findings"]]
    return block


__all__ = [
    "CARRIER_GENERATIONS", "CORRECTIONS", "CORRECTIONS_BY_FINDING",
    "CORRECTION_KEYS",
    "CORRECT_KEYS_DEFAULT", "Correction", "CorrectModeResolution", "EXACT",
    "FINDING_BY_KEY", "KEY_ABSENT_ON", "KEY_GENERATIONS", "KEY_SCOPE",
    "KEY_UNREAD",
    "OUTPUT_ONLY_KEYS", "PHYSICS_MODES", "SCOPE_SELECTORS",
    "PHYSICS_MODE_CORRECT",
    "PHYSICS_MODE_DEFAULT", "PHYSICS_MODE_EXACT", "STRICT",
    "STRICT_FORCED_BY", "absent_value", "in_effect", "key_echoes",
    "key_in_scope", "resolution_of",
    "resolve",
    "run_receipt", "uncorrected", "uncorrected_notice",
]
