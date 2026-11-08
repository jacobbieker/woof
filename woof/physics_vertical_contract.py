"""Lightweight first-call vertical bounds for physics front doors.

This module deliberately lives at package top level and imports no forecast
executor.  The standalone preprocessing distribution needs to reject an
impossible vertical grid without staging CUDA-backed component modules.
Component tests bind these values and layer-count helpers to their runtime
counterparts.
"""

from __future__ import annotations

import math

import numpy as np

KFCUP_VERTICAL_LEVEL_BOUNDS = (8, 128)

# The two component bounds that are OWNED elsewhere: WSM6's and WDM6's
# ceilings are the 256-level refusal ceilings their constants modules
# declare (both kernels keep their column arrays in a runtime-sized
# workspace, so nothing is compiled per level count).  Those modules are
# CuPy-free and staged wherever this one is, so the values are imported
# rather than transcribed.  Re-exported under their own names below.
from woof.core.wdm6_constants import WDM5_VERTICAL_LEVEL_BOUNDS, WDM6_VERTICAL_LEVEL_BOUNDS
# WSM3's ceiling is owned the same way: the deepest WSM3_KMAX tier of the
# CuPy-free constants leaf.
from woof.core.wsm3_constants import WSM3_VERTICAL_LEVEL_BOUNDS
# WSM6's 256-level workspace ceiling and WSM6R's window are owned by the
# WSM6 family's CuPy-free constants module, which their launchers enforce;
# imported for the same reason.
from woof.core.wsm6_constants import (WSM6R_VERTICAL_LEVEL_BOUNDS,
                                       WSM6_VERTICAL_LEVEL_BOUNDS)
from woof.core.wsm_family_constants import vertical_level_bounds as _wsm_family_bounds


class PhysicsVerticalPreflightError(ValueError):
    """Resolved physics cannot execute on the requested vertical grid.

    Lives here rather than beside the preflight because the LAUNCHERS raise
    it too: the preparation-time gate and the first-call rejection inside a
    physics module are the same refusal met at two moments, and a user who
    reaches the second one has already paid for fetch and preparation.  A
    ``ValueError`` subclass so every existing ``except ValueError`` around a
    launcher keeps working.
    """


def vertical_bounds_wording(bounds: tuple[int | None, int | None]) -> str:
    """The one spelling of a level-count bound, shared by every refusal."""

    minimum, maximum = bounds
    if minimum is None and maximum is None:
        raise ValueError("a vertical bound must constrain at least one end")
    if minimum is None:
        return f"nz <= {maximum}"
    if maximum is None:
        return f"nz >= {minimum}"
    return f"{minimum} <= nz <= {maximum}"


def outside_vertical_bounds(nz: int,
                            bounds: tuple[int | None, int | None]) -> bool:
    """True when ``nz`` violates either end of a component's bound."""

    minimum, maximum = bounds
    return ((minimum is not None and nz < minimum)
            or (maximum is not None and nz > maximum))


def refuse_vertical_levels(label: str,
                           bounds: tuple[int | None, int | None],
                           nz: int, *, breakage: str,
                           ) -> PhysicsVerticalPreflightError:
    """Build the launcher-side refusal in the preflight's own grammar.

    ``breakage`` names what would go wrong and what to do instead, because a
    refusal that only restates the number tells a user nothing they can act
    on.  The sentence before it is byte-identical to the line
    ``validate_resolved_physics_vertical_levels`` emits, so the two surfaces
    read as one gate rather than two.
    """

    return PhysicsVerticalPreflightError(
        f"{label} requires {vertical_bounds_wording(bounds)}, got nz={nz}: "
        f"{breakage}  `woof check` refuses this configuration before a run "
        "starts; reaching it here means the preparation gate was bypassed.")


FAST_SBM_VERTICAL_LEVEL_BOUNDS = (2, None)

ETANEW_VERTICAL_LEVEL_BOUNDS = (1, None)
MYNN_VERTICAL_LEVEL_BOUNDS = (5, None)
#: YSU.  The ceiling is ``YSU_KMAX`` in ``kernels/ysu_default.cu`` and
#: ``kernels/ysu_480.cu``, which is now a REFUSAL
#: ceiling and nothing else: the column arrays live in a global workspace
#: whose per-slot extent is the runtime ``nz``, so no array is sized by it.
#: It sits at the dynamical core's own 256-level ceiling.  The floor is the
#: launcher's own ``nz >= 4`` -- the counter-gradient and entrainment layers
#: the scheme indexes need four levels to exist.
YSU_VERTICAL_LEVEL_BOUNDS = (4, 256)
#: Shin-Hong, the same pair for the same two reasons (``SHINHONG_KMAX`` is
#: likewise only a refusal ceiling over a runtime-sized workspace, and the
#: scheme shares YSU's column indexing).  Held as its own name rather than
#: aliased to YSU's: the two kernels are separate translation units a future
#: change could size differently.
SHINHONG_VERTICAL_LEVEL_BOUNDS = (4, 256)
#: MYJ (and MYJURB, its urban companion).  The ceiling is the ``MYJ_KMAX``
#: refusal guard in ``kernels/myjpbl.cu`` / ``kernels/myjurb.cu``: since
#: 2026-10-12 neither kernel sizes an array by it -- the column lives in a
#: global workspace the launcher sizes to the runtime nz
#: (docs/kernel_local_memory_bounds.md section 12) -- so 256 is the level
#: count the end-to-end target runs at, not a frame budget.  The floor is
#: where VDIFQ's tridiagonal chain still has interior rows and MIXLEN's
#: profile smoothing can read ``REL(K-1)``/``REL(K+1)``.
MYJ_VERTICAL_LEVEL_BOUNDS = (4, 256)
#: QNSE (bl_pbl_physics=4) and its mass flux.  ``QNSE_KMAX`` in
#: ``kernels/qnsepbl.cu`` for the ceiling; the floor is MYJ's for MYJ's
#: reasons (VDIFQ's interior rows, MIXLEN's REL(K-1)/REL(K+1)), which the
#: mass flux's IKE-1 updraft loop also needs.
QNSE_VERTICAL_LEVEL_BOUNDS = (4, 128)
#: SASE.  The deepest ``SASE_KMAX`` tier ``woof/core/sase.py`` compiles
#: ``kernels/sase.cu`` at (``woof.core.sase_limits.KMAX_TIERS``; the run's
#: own tier is the smallest power of two from 32 that holds ``nz``), which is
#: the same number as ``woof.core.sase_limits.MAX_COLUMN_LEVELS`` and
#: ``woof.config.SASE_MAX_NZ``: the implicit vertical solve keeps FP64
#: columns of that depth in per-thread local memory.  256 is the dycore's own
#: ceiling.  The floor is where the solve stops being an identity -- one
#: interior face needs two levels.
SASE_VERTICAL_LEVEL_BOUNDS = (2, 256)
#: UW moist turbulence (bl_pbl_physics=9).  No ceiling: the CAM automatic
#: arrays live in a per-column global workspace sized from nz at launch
#: (woof/core/uwpbl.py), so no level count is compiled in.  The floor is
#: where the implicit diffusion has an interior interface (compute_vdiff's
#: k = 2..pver rows) and a Richardson number exists above the surface.
UWPBL_VERTICAL_LEVEL_BOUNDS = (2, None)
# MRF uses a dynamic global workspace and needs one interior interface.
MRF_VERTICAL_LEVEL_BOUNDS = (2, None)
SATMEDMF_VERTICAL_LEVEL_BOUNDS = (8, None)
# The second turbulence level reads the third mass/updraft level.
TEMF_VERTICAL_LEVEL_BOUNDS = (3, None)
#: GBM moist TKE (bl_pbl_physics=12).  No ceiling: GBM2D's column arrays
#: live in a global workspace sized from nz at launch (woof/core/gbmpbl.py),
#: which also retires WRF's own tridag bound gam(nmax=100).  The floor is
#: where the TKE equation still has an interior full level to solve on
#: (module_bl_gbmpbl.F:770-809 runs over kte-1 rows).
GBMPBL_VERTICAL_LEVEL_BOUNDS = (2, None)
#: EEPS (bl_pbl_physics=16).  No ceiling: the column's work arrays are
#: sized by a compile-time EEPS_NZ equal to the run's own level count
#: (woof/core/eeps_pbl.py), so no depth is compiled in.  The floor is
#: where the TKE/epsilon solves over rows 1..km-1 have an interior row.
EEPS_VERTICAL_LEVEL_BOUNDS = (3, None)
#: GFS nonlocal PBL (bl_pbl_physics=3).  No ceiling: the column arrays
#: live in a binary64 global workspace sized from nz at launch
#: (woof/core/gfs_pbl.py).  The floor is MONINP's own: its PBL-top
#: search runs over k = 2..KM/2 and interpolates against level k-1, so
#: below four levels the search is empty and the height reads ZL(0).
GFS_PBL_VERTICAL_LEVEL_BOUNDS = (4, None)

#: BouLac (bl_pbl_physics=8).  ``BOULAC_KMAX`` in ``kernels/boulac.cu`` for
#: the ceiling: one thread holds the column in per-thread local memory, the
#: MYJ shape.  The floor is where SHEAR and BUOY, which live on the interior
#: levels kts+1..kte-1 (module_bl_boulac.F:787, :820), have a level at all.
BOULAC_VERTICAL_LEVEL_BOUNDS = (3, 128)

#: ACM2 (bl_pbl_physics=7).  No ceiling: the column arrays live in a
#: global workspace sized from nz at launch (woof/core/acm2pbl.py), so no
#: level count is compiled in.  The floor is where ACM's two-level stencils
#: exist: AI(2) and DZFI(KTE-1) are read on every column and the
#: free-convection test compares THETAV(1) with THETAV(2) under KLPBL > 3.
ACM2_VERTICAL_LEVEL_BOUNDS = (4, None)
#: Kain-Fritsch.  ``KF_KMAX`` in ``kernels/kf.cu`` is the refusal ceiling;
#: it sizes only the two stack-resident arrays ``tv_env`` and
#: ``positive_energy`` (2 x 4 x 256 = 2,048 B declared at 256, kept on the
#: stack for bit-identity -- docs/kernel_local_memory_bounds.md section 9).
#: Every other column array is in a runtime-``nz`` workspace.  The floor is
#: the kernel's own ``nz >= 8`` refusal.
KF_VERTICAL_LEVEL_BOUNDS = (8, 256)
MSKF_VERTICAL_LEVEL_BOUNDS = (8, 160)
# Grell-Freitas: the inversion-layer search clamps kend to ktf-8 and its
# second-derivative stencil then reaches ktf-1, so a column shorter than 12
# levels leaves no search window at all; the certified fixture ran nz=40 and
# the kernel recompiles its level bound above GF_KMAX=40 through the
# integer-define tier, so there is no fixed upper bound to declare.
GF_VERTICAL_LEVEL_BOUNDS = (12, None)
KESSLER_VERTICAL_LEVEL_BOUNDS = (None, 256)
LIN_VERTICAL_LEVEL_BOUNDS = (2, 512)
#: Ferrier-Aligo (mp_physics=5): the kernel streams the column top-down and
#: keeps no level array (woof/core/kernels/fer_hires.cu), so there is no
#: ceiling; two levels are the least EGCP01DRV's top-level override and one
#: level below it need.
FERRIER_VERTICAL_LEVEL_BOUNDS = (2, None)
from woof.core.wsm6_constants import WSM5_VERTICAL_LEVEL_BOUNDS
#: WSM6 (mp_physics=6): imported above from ``woof.core.wsm6_constants``
#: (the 256-level workspace ceiling).
# Sedimentation uses adjacent level geometry; workspace is sized from nz.
GSFCGCE4ICE_VERTICAL_LEVEL_BOUNDS = (2, None)
THOMPSON_VERTICAL_LEVEL_BOUNDS = (2, 256)
#: Aerosol-aware Thompson (mp_physics=28).  Identical to the classic
#: Thompson bound and NOT a copy of a number: the mp=28 sedimentation
#: translation unit declares the same ``THOMPSON_AA_KMAX_GENERIC 256``
#: per-thread column bound as ``kernels/thompson.cu``, and the runtime
#: counterpart ``woof.core.thompson_aerosol.VERTICAL_LEVEL_BOUNDS`` is
#: bound to this constant by
#: ``tests/test_mp28_runnable.py::test_mp28_vertical_bounds_are_the_runtime_bounds``
#: so the two cannot drift.  Held as its own name rather than aliased to
#: THOMPSON_VERTICAL_LEVEL_BOUNDS because the two schemes' kernels are
#: separate translation units that a future change could size differently.
THOMPSON_AEROSOL_VERTICAL_LEVEL_BOUNDS = (2, 256)
UDM_VERTICAL_LEVEL_BOUNDS = (2, 256)
RCON_VERTICAL_LEVEL_BOUNDS = (2, 256)
MORRISON_VERTICAL_LEVEL_BOUNDS = (2, 256)
MORRISON_AERO_VERTICAL_LEVEL_BOUNDS = (2, 256)
NSSL2_VERTICAL_LEVEL_BOUNDS = (3, 256)
#: Milbrandt-Yau two-moment (mp_physics=9).  The same pair
#: ``woof.core.milbrandt2.VERTICAL_LEVEL_BOUNDS`` enforces at the first
#: call (``MY2_KMAX`` per-thread column bound, floor where the sedimentation
#: walk still has an interior level).  Held here because that module imports
#: CuPy at module scope and the preparation-time gate must price a
#: configuration on a host with no device -- and because until this row
#: existed the gate SKIPPED mp=9 silently, so a 300-level Milbrandt-Yau run
#: passed ``woof check`` and died on its first microphysics call.
MILBRANDT2_VERTICAL_LEVEL_BOUNDS = (3, 256)
#: TEMPO (named scheme, woof.microphysics_schemes): its sedimentation
#: reads k-2..k+1 (semi-Lagrangian wi stencil) and needs nz >= 3; the GPU
#: runtime compiles its column arrays for the domain's nz, capped here.
TEMPO_VERTICAL_LEVEL_BOUNDS = (3, 256)
#: WDM6 (mp_physics=16): the 256-level refusal ceiling, imported from the
#: CuPy-free constants module at the top of this file rather than retyped
#: (``WDM6_VERTICAL_LEVEL_BOUNDS``).
#: P3 one-category (mp_physics=50 and 51).  No per-thread column ceiling: the port
#: works on global ``(ncol, nk)`` workspaces (kbot=0, ktop=nk-1) and compiles
#: no level tier, so like Grell-Freitas the row is a floor with no declared
#: ceiling.  The floor is the dycore's own: one level is a column.
P3_VERTICAL_LEVEL_BOUNDS = (1, None)
#: WSM7 (mp_physics=24): the floor the PLM remap needs (a layer above and
#: below) and the deepest compiled ``WSM7_KMAX`` tier, read from the
#: scheme's row in woof/data/wsm_family/schemes.json rather than retyped.
WSM7_VERTICAL_LEVEL_BOUNDS = _wsm_family_bounds(24)
#: Goddard GCE (mp_physics=97): the row's own pair
#: (woof.core.mp_column_schemes), read rather than retyped.  Floor 2: the
#: column kernel reads the second level's density (module_mp_gsfcgce.F:
#: 1560); ceiling 256: the reflectivity unit's REFL_KMAX.
from woof.core.mp_column_schemes import (  # noqa: E402
    MP_COLUMN_SCHEMES as _TABLE_ROWS)
GSFCGCE_VERTICAL_LEVEL_BOUNDS = _TABLE_ROWS[97].vertical_level_bounds
#: SBU-YLin (mp_physics=13).  ``kernels/sbu_ylin.cu`` keeps ten per-level
#: working arrays in a fixed local frame compiled at the tiers the
#: registry's ``column_kernel`` row lists (64, 128, 256); a deeper column
#: is refused by the launcher.  One level is a column (the sedimentation
#: loops simply have no interior face).
SBU_YLIN_VERTICAL_LEVEL_BOUNDS = (1, 256)
#: NTU multi-moment (mp_physics=56): two levels for the sedimentation
#: flux, and the deepest compiled column tier of the kernel's per-column
#: work arrays (``woof.core.ntu.KMAX_TIERS``); a deeper column would
#: overrun them and is refused by ``woof.core.ntu.kmax_for``.
NTU_VERTICAL_LEVEL_BOUNDS = (2, 256)
#: New Tiedtke (cu_physics=16).  ``kernels/ntiedtke.cu`` skips ``k >= nz-2``
#: and loops ``k = nz-2 .. 1`` (the transcription of ``jk = klevm1, 2, -1``),
#: so a column needs four levels before the cloud-base search has one
#: interior row; no fixed ceiling, the workspaces are ``(ncol, nz)``.
NEW_TIEDTKE_VERTICAL_LEVEL_BOUNDS = (4, None)
#: Orographic gravity-wave drag (``gwd_opt`` 1 = KIM, 3 = GSL), launched
#: inside the PBL driver by ``woof/core/terrain_drag.py``.  ``TD_KMAX`` in
#: ``kernels/terrain_drag.cu`` sizes the per-thread stress profiles
#: ``taup[TD_KMAX + 1]`` (and GSL's ``taufb``), so it is a real per-thread
#: column ceiling; it sits at the dycore's 256.  The launchers declare no
#: floor, so none is stated here.  ``gwd_opt`` is a run setting rather than a registry component,
#: so ``validate_resolved_physics_vertical_levels`` reads this row directly
#: whenever ``gwd_opt`` is 1 or 3.
TERRAIN_DRAG_VERTICAL_LEVEL_BOUNDS = (None, 256)
# CU_SAS forms ZL(KTE) from ZL(KTE-1), module_cu_sas.F:354.
SAS_VERTICAL_LEVEL_BOUNDS = (2, None)
# Original cu6 reads the same interior cloud-base levels; global arenas
# carry the full vertical dimension without a compiled level ceiling.
TIEDTKE_VERTICAL_LEVEL_BOUNDS = (4, None)
#: Grell 3D (cu_physics=5).  ``kernels/g3.cu`` starts the k22 search at
#: level 3 (cup_MAXIMI(..,3,..)), stops the cloud-top search at ktf-1 and
#: smooths k = 3 .. nz-2 in conv_grell_spread3d, so a column needs six
#: levels for every one of those ranges to hold an interior row; no fixed
#: ceiling, the workspace is (nz+1) words per slot at run time.
GRELL_3D_VERTICAL_LEVEL_BOUNDS = (6, None)
#: New SAS (cu_physics=96).  nsas2d reads zi(3)-zi(2) for the bottom
#: mass-flux level (module_cu_nsas.F:909) and needs kb < kbcon < kmax for
#: a cloud base, and nscv2d's PBL and cloud searches run k = 2 .. kte-1,
#: so four levels are the least with one interior row; no ceiling, the
#: workspace is sized from nz at run time (woof.core.column_cumulus).
NEW_SAS_VERTICAL_LEVEL_BOUNDS = (4, None)
# KSAS reads zi(3) and xlamb(1..2), and needs an interior cloud layer.
KSAS_VERTICAL_LEVEL_BOUNDS = (4, None)
#: Betts-Miller-Janjic (cu_physics=2).  ``kernels/bmj.cu`` reads the cloud
#: base's lower neighbour ``LBOT+1`` with ``LBOT <= LMH-1`` and requires
#: ``LTOP <= LBOT-2`` for any cloud, so a column with fewer than four
#: levels has no admissible cloud and the shallow arm's ``LTOP-1`` would
#: leave it; no fixed ceiling, the workspace is ``(slots * nz, columns)``.
#: The same number is the ``vertical_minimum`` of the scheme's registry
#: ``column_kernel`` row, which the launcher enforces.
BMJ_VERTICAL_LEVEL_BOUNDS = (4, None)

# SHALCV reads level 2 of the lower KM-1 levels (WRF module_cu_osas.F:2438).
OSAS_VERTICAL_LEVEL_BOUNDS = (3, None)
#: Grell-Devenyi (cu_physics=93).  ``kernels/gd.cu`` (CUP_enss) starts the
#: originating-level search at level 3 (cup_MAXIMI(HEO_CUP,3,...)), searches
#: kbmax over ``k = 1..ktf-2`` and scans cloud base up to kbmax+2, and the
#: downdraft needs jmin > 3: below six levels those windows overlap and the
#: maximum search reads he_cup(3) of a column that may not have it.  No
#: ceiling: the workspace is sized at run time to (nz+2) per slot.
GD_VERTICAL_LEVEL_BOUNDS = (6, None)
#: The dynamical core's own vertical bound, from the top ``WPHI_MAX_LEV``
#: tier ``woof.core.acoustic`` compiles the implicit w''-phi'' solve at.
#: Bound to the launcher's ladder by
#: ``tests/test_acoustic_nz_tiers.py::test_the_contract_bound_is_the_launcher_bound``.
ACOUSTIC_VERTICAL_LEVEL_BOUNDS = (1, 256)

RRTMGP_TOA_PRESSURE_PA = 1.005183574463
WRF_LW_UPPER_DELTA_P_PA = 400.0
#: RTE+RRTMGP's ceiling is on MODEL levels, not on radiation layers.  No
#: RRTMGP kernel holds a fixed per-layer array any more: the RTE solvers
#: compile ``RRTMGP_MAX_LAYERS`` to each call's own layer count and the
#: Planck-source kernel walks a one-layer window (it held ``pfrac[128]``,
#: the old 128-layer cap, until 2026-10).  The radiation column is
#: therefore ``nz`` plus WRF's above-model layers
#: (:func:`rrtmgp_above_model_layer_counts`) at ANY model top -- up to
#: 256 + nint(p_top / 400 Pa) longwave and 256 + 1 shortwave -- and the
#: bound that remains is the depth the solvers are validated and priced
#: at, which is the dynamical core's own (``ACOUSTIC_VERTICAL_LEVEL_BOUNDS``).
#: The local frame that grows with the column is priced per layer
#: (``woof.core.preflight.RRTMGP_RTE_LAYER_FRAME``).
MAX_RRTMGP_MODEL_LEVELS = 256
#: Legacy RRTMG longwave (ra_lw_physics = 4, ra_rrtmg_variant =
#: "rrtmg_legacy") is bounded by the MODEL depth, not by a layer count.
#: The engine's one per-thread column (``rlw_rtrn_march``'s
#: atrans/atot/bbugas/bbutot, ``RLW_MAXLAY`` floats each) is compiled per
#: run at :func:`legacy_longwave_layer_tier` of the run's layer count, so
#: it holds whatever WRF builds; the ceiling is the dynamical core's own
#: ``nz`` bound, and the radiation column may then carry WRF's 4 hPa
#: buffer layers above the model top on top of it -- up to
#: :func:`max_legacy_longwave_layers` ``(p_top)`` = 256 + nint(p_top/400 Pa)
#: layers.  (Until 2026-10 this was a flat 128-layer
#: ``MAX_LEGACY_LONGWAVE_LAYERS``, the then-fixed ``RLW_MAXLAY``.)
MAX_LEGACY_LONGWAVE_MODEL_LEVELS = ACOUSTIC_VERTICAL_LEVEL_BOUNDS[1]
#: ``RLW_MAXLAY`` as ``kernels/rrtmg_lw_chain.cu`` declares it by default:
#: every run of at most this many LW layers compiles the unchanged source.
LEGACY_LONGWAVE_SHIPPED_LAYER_TIER = 128
#: Deeper runs round their layer count up to a multiple of this, so the
#: few depths a deployment uses share a handful of compiled units (and
#: their NVRTC cache entries) rather than one per layer count.
LEGACY_LONGWAVE_LAYER_TIER_STEP = 64
MAX_LEGACY_SHORTWAVE_LAYERS = None  # runtime-sized workspace; no fixed SW layer ceiling


def legacy_longwave_layer_tier(nlayers: int) -> int:
    """``RLW_MAXLAY`` the legacy LW chain is compiled at for ``nlayers``.

    The shipped 128 for every column of at most 128 layers (the unchanged
    translation unit, so those runs are bit-for-bit the pre-tier engine),
    else ``nlayers`` rounded up to a multiple of
    :data:`LEGACY_LONGWAVE_LAYER_TIER_STEP` (192, 256, 320, ...).
    """
    if isinstance(nlayers, bool) or int(nlayers) != nlayers or int(nlayers) < 1:
        raise ValueError(
            f"legacy LW layer count must be an integer >= 1, got {nlayers!r}")
    count = int(nlayers)
    if count <= LEGACY_LONGWAVE_SHIPPED_LAYER_TIER:
        return LEGACY_LONGWAVE_SHIPPED_LAYER_TIER
    step = LEGACY_LONGWAVE_LAYER_TIER_STEP
    return -(-count // step) * step


def max_legacy_longwave_layers(p_top: float) -> int:
    """Deepest legacy LW radiation column a run at ``p_top`` [Pa] can build:
    :data:`MAX_LEGACY_LONGWAVE_MODEL_LEVELS` model layers plus WRF's
    ``nint(p_top/400 Pa)`` buffer layers above the model top."""
    return legacy_radiation_layer_counts(
        MAX_LEGACY_LONGWAVE_MODEL_LEVELS, p_top)[0]


def rrtmgp_above_model_layer_counts(
        p_top: float, *,
        pressure_floor: float = RRTMGP_TOA_PRESSURE_PA,
) -> tuple[int, int]:
    """Return WRF's RTE+RRTMGP ``(LW, SW)`` cap-layer counts."""

    if (not math.isfinite(p_top) or not math.isfinite(pressure_floor)
            or p_top < 0.0 or pressure_floor <= 0.0):
        raise ValueError("radiation top pressures must be finite and nonnegative")
    if p_top <= pressure_floor:
        return 0, 0
    lw = int(np.floor(p_top / WRF_LW_UPPER_DELTA_P_PA + 0.5))
    sw = int(0.5 * p_top >= pressure_floor)
    return max(0, lw), sw


def legacy_radiation_layer_counts(
        nz: int, p_top: float,
) -> tuple[int, int]:
    """Return exact first-call WRF legacy-RRTMG ``(LW, SW)`` counts."""

    value = np.float32(
        np.float32(np.float32(p_top) * np.float32("0.01"))
        / np.float32("4.0"))
    raw = float(value)
    rounded = int(raw)
    if raw - rounded >= 0.5:
        rounded += 1
    elif raw - rounded <= -0.5:
        rounded -= 1
    # The runtime keeps one clear buffer layer above a top below 2 hPa,
    # where the rounded 4-hPa extension is zero
    # (woof.core.rrtmg_legacy_prep.compute_lw_nlayers).
    return int(nz) + max(1, rounded), int(nz) + 1


# GFDL Eta owns 81-layer work arrays and 82 interface slots. Larger grids
# would index past the explicit CUDA layout before a radiation call completes.
GFDLETA_VERTICAL_LEVEL_BOUNDS = (1, 81)

#: K-epsilon (bl_pbl_physics=17).  No ceiling: the column arrays live in a
#: global workspace sized from nz at launch (woof/core/keps.py).  The floor
#: is GET_PBLH's: its theta-v search starts at level 3 at the earliest and
#: its exit safeguard fires at kte-1, so below four levels the search has no
#: level to test.
KEPS_VERTICAL_LEVEL_BOUNDS = (4, None)

# CAM column pools scale dynamically with vertical extent.
CAMMGMP_VERTICAL_LEVEL_BOUNDS = (2, 512)
CAMZM_VERTICAL_LEVEL_BOUNDS = (4, None)
UWSHCU_VERTICAL_LEVEL_BOUNDS = (3, None)
