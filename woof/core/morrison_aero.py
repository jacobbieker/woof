"""Aerosol-aware Morrison two-moment microphysics (WRF ``mp_physics=40``).

WRF v4.7.1 ``phys/module_mp_morr_two_moment_aero.F`` (package
``morr_tm_aero``) is the mp_physics = 10 module with four init words
changed (DCS 350 um and the three words derived from it), constant
no-hydrometeor radii for RRTMG (2.49 um cloud, 4.99 um ice), the droplet
number returned through the advected scalar QNC, and a prescribed-aerosol
path (``aercu_opt = 2``): droplet activation (Abdul-Razzak and Ghan 2000)
and ice nucleation (Liu and Penner 2005) on ten CESM aerosol modes, with
WVAR from the PBL's heat diffusivity.  ``kernels/morrison.cu`` compiled as
the ``morrison_aero`` module carries every line of it (``MORR_AERO``
blocks); graded word for word against WRF by
``tools/morrison_aero_wrf471_oracle``.

What woof runs: ``aercu_opt = 0`` (WRF's default), the constant-droplet
path.  WRF refuses ``aercu_opt > 0`` unless ``cu_physics = 11`` (MSKF,
check_a_mundo), which woof does not port, and the prescribed aerosol is
read from WRF's ``CESM_RCP4.5_Aerosol_Data.dat`` by the cumulus driver;
the kernel's aercu_opt = 2 path is held to WRF's words through
:func:`launch_morrison_aero` for the day MSKF lands.
"""

from __future__ import annotations

import numpy as np

import cupy as cp

from woof.core.kernels import get_kernel
from woof.core.morrison import (_CELL_TPB, _COLUMN_TPB, _SEDIMENT_TPB,
                                _SHALLOW_KMAX, VERTICAL_LEVEL_BOUNDS, _check,
                                _prepare_fields, bound_snow_ratio,
                                launch_exner, launch_refl10cm_wrf)
from woof.core.morrison_constants import (NWORK, SCRATCH_WORK_SLOT, WORK_T3D,
                                           rimed_ice_constants,
                                           work_chunk_columns,
                                           work_slot_shape)
from woof.core.state import DTYPE

MODULE = "morrison_aero"
#: WRF's aerosol species (aerocu) count, no_src_types_cu.
NO_SRC_TYPES_CU = 10
#: CCN1_GS..CCN7_GS supersaturations (%), Registry.EM_COMMON:1679-1685.
CCN_SUPERSATURATIONS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
AERCU_OPTIONS = (0, 1, 2)
LAUNCHERS = ("launch_morrison_aero",)


def launch_morrison_aero(theta, qv, qc, qr, qi, qs, qg,
                         nc, nr, ni, ns, ng, rho, pii, pressure, dz, w, kzh,
                         rainnc, rainncv, snownc, snowncv,
                         graupelnc, graupelncv, sr, dt: float, *,
                         aercu_opt: int = 0, aercu_fct: float = 1.0,
                         aerocu=None, ccn=None, efcg=None, efig=None,
                         efsg=None, wact=None, refl_10cm=None,
                         mskf_refl_10cm=None,
                         effc=None, effr=None, effi=None, effs=None,
                         qrcuten=None, qscuten=None, qicuten=None,
                         morr_rimed_ice: int = 1, columns_per_call=None,
                         t1d_out=None, _work_scratch=None,
                         _ltrue_scratch=None):
    """One WRF MP_MORR_TWO_MOMENT_AERO call over an FP32 column batch.

    As :func:`woof.core.morrison.launch_morrison`, plus: ``nc`` is WRF's
    NC (QNC), read when ``aercu_opt == 2`` and returned in every mode;
    ``w`` and ``kzh`` are ``(nz + 1, ny, nx)`` w-level fields (WRF reads
    W(K) and KZH(K+1)); ``aerocu`` is ``(10, nz, ny, nx)`` CESM aerosol mass
    (ug m-3, WRF's species order), read when ``aercu_opt == 2``; ``ccn``
    ``(7, nz, ny, nx)``, when given, receives CCN1_GS..CCN7_GS (WRF's
    units; zero unless aercu_opt == 2); ``efcg/efig/efsg/wact`` are written when
    ``aercu_opt > 0``.  ``refl_10cm`` and ``mskf_refl_10cm``, when given,
    receive the driver's two reflectivities; with no MSKF the cumulus
    hydrometeors WRF adds to the second are zero, so it is the first's
    words (refl10cm_hm of the same arguments, NR + 0*0).
    """
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"Morrison fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    minimum, maximum = VERTICAL_LEVEL_BOUNDS
    if not minimum <= nz <= maximum:
        raise ValueError(f"nz={nz} outside {VERTICAL_LEVEL_BOUNDS}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if aercu_opt not in AERCU_OPTIONS:
        raise ValueError(f"aercu_opt must be one of {AERCU_OPTIONS}")
    rimed_ice_constants(morr_rimed_ice)
    for name, value in (
            ("theta", theta), ("qv", qv), ("qc", qc), ("qr", qr),
            ("qi", qi), ("qs", qs), ("qg", qg), ("nc", nc), ("nr", nr),
            ("ni", ni), ("ns", ns), ("ng", ng), ("rho", rho),
            ("pii", pii), ("pressure", pressure), ("dz", dz)):
        _check(name, value, shape)
    staggered = (nz + 1, ny, nx)
    _check("w", w, staggered)
    _check("kzh", kzh, staggered)
    prescribed = aercu_opt == 2
    if prescribed:
        if aerocu is None:
            raise ValueError("aercu_opt=2 needs the prescribed aerosol aerocu")
        _check("aerocu", aerocu, (NO_SRC_TYPES_CU,) + shape)
    cu_values = (qrcuten, qscuten, qicuten)
    has_cu = any(value is not None for value in cu_values)
    if has_cu and not all(value is not None for value in cu_values):
        raise ValueError("qrcuten, qscuten, and qicuten must be supplied "
                         "together")
    if has_cu:
        for name, value in zip(("qrcuten", "qscuten", "qicuten"), cu_values):
            _check(name, value, shape)
        qrcu, qscu, qicu = cu_values
    else:
        qrcu = qscu = qicu = rho
    for name, value in (
            ("rainnc", rainnc), ("rainncv", rainncv),
            ("snownc", snownc), ("snowncv", snowncv),
            ("graupelnc", graupelnc), ("graupelncv", graupelncv),
            ("sr", sr)):
        _check(name, value, (ny, nx))
    effective = {"effc": effc, "effr": effr, "effi": effi, "effs": effs}
    for name, value in effective.items():
        if value is None:
            effective[name] = cp.empty_like(theta)
        else:
            _check(name, value, shape)
    if ccn is not None:
        _check("ccn", ccn, (7,) + shape)
    radii = {"efcg": efcg, "efig": efig, "efsg": efsg, "wact": wact}
    for name, value in radii.items():
        if value is None:
            if aercu_opt > 0:
                raise ValueError(f"aercu_opt={aercu_opt} writes {name}")
            radii[name] = theta      # never written when aercu_opt == 0
        else:
            _check(name, value, shape)
    ncol = ny * nx
    chunk = ncol if columns_per_call is None else int(columns_per_call)
    if chunk < 1:
        raise ValueError(f"columns_per_call must be positive, got {chunk}")
    chunk = min(chunk, ncol)
    if _work_scratch is None:
        work = cp.empty((NWORK * nz * chunk,), dtype=DTYPE)
    else:
        work = _work_scratch
        if (work.dtype != DTYPE or not work.flags.c_contiguous
                or work.size < NWORK * nz * chunk):
            raise ValueError("_work_scratch is too small")
    flat_work = work.reshape(-1)
    refl_due = refl_10cm is not None or mskf_refl_10cm is not None
    if refl_due and t1d_out is None:
        t1d_out = cp.empty_like(theta)
    if t1d_out is not None:
        _check("t1d_out", t1d_out, shape)
    ltrue = (cp.empty((ny, nx), dtype=cp.int32)
             if _ltrue_scratch is None else _ltrue_scratch)
    _check("_ltrue_scratch", ltrue, (ny, nx), cp.int32)
    ltrue.fill(0)
    stop = cp.zeros((1,), dtype=cp.int32)
    aero_in = aerocu if prescribed else rho

    rimed = np.int32(morr_rimed_ice)
    process = get_kernel(MODULE, "mo_process")
    sediment = get_kernel(MODULE, "mo_sediment_64" if nz <= _SHALLOW_KMAX
                          else "mo_sediment_256")
    finalize = get_kernel(MODULE, "mo_finalize")
    ncell = np.int64(nz * ncol)
    for col0 in range(0, ncol, chunk):
        ccols = min(chunk, ncol - col0)
        nlocal = nz * ccols
        cell_blocks = (nlocal + _CELL_TPB - 1) // _CELL_TPB
        column_blocks = (ccols + _COLUMN_TPB - 1) // _COLUMN_TPB
        span = (np.int64(nlocal), np.int32(ncol), np.int32(col0),
                np.int32(ccols))
        process(
            (cell_blocks,), (_CELL_TPB,),
            (theta, pii, pressure, qv, qc, qr, qi, qs, qg, nc, nr, ni, ns,
             ng, qrcu, qscu, qicu, np.int32(has_cu), rho, flat_work, ltrue,
             DTYPE(dt), rimed) + span
            + (w, kzh, aero_in, rho if ccn is None else ccn,
               np.int32(ccn is not None), np.int32(aercu_opt),
               DTYPE(aercu_fct), stop, ncell))
        sediment(
            (column_blocks,), (_SEDIMENT_TPB,),
            (pressure, dz, rho, qc, qr, qi, qs, qg, nc, nr, ni, ns, ng,
             flat_work, ltrue, rainnc, rainncv, snownc, snowncv, graupelnc,
             graupelncv, sr, DTYPE(dt), rimed, np.int32(nz), np.int32(ncol),
             np.int32(col0), np.int32(ccols), np.int32(aercu_opt)))
        finalize(
            (cell_blocks,), (_CELL_TPB,),
            (theta, pii, pressure, rho, qv, qc, qr, qi, qs, qg, nc, nr, ni,
             ns, ng, flat_work, ltrue, effective["effc"], effective["effi"],
             effective["effs"], effective["effr"], DTYPE(dt), rimed) + span
            + (w, kzh, radii["efcg"], radii["efig"], radii["efsg"],
               radii["wact"], np.int32(aercu_opt)))
        if t1d_out is not None:
            plane = flat_work[WORK_T3D * nlocal:(WORK_T3D + 1) * nlocal]
            t1d_out.reshape(nz, ncol)[:, col0:col0 + ccols] = (
                plane.reshape(nz, ccols))
    if prescribed and int(stop.get()[0]):
        # module_mp_morr_two_moment_aero.F:5946-5951: WRF writes
        # 'incorrect ice nucleation number' and STOPs.
        raise RuntimeError("mp_physics=40: incorrect ice nucleation number "
                           "(WRF stops: 'nuclei prbolem?')")
    if refl_due:
        refl = refl_10cm if refl_10cm is not None else cp.empty_like(theta)
        launch_refl10cm_wrf(qv, qr, nr, qs, ns, qg, ng, t1d_out, pressure,
                            refl, morr_rimed_ice=morr_rimed_ice)
        if mskf_refl_10cm is not None:
            _check("mskf_refl_10cm", mskf_refl_10cm, shape)
            cp.copyto(mskf_refl_10cm, refl)
    return work


def kzh_on_w_levels(state):
    """WRF's KZH = exch_h on the nz + 1 w levels, or zeros with no PBL.

    woof keeps exch_h on the nz lower interfaces (k = 0 is the surface);
    the model top's interface is WRF's never-written zero.
    """
    nz, ny, nx = state.p.shape
    kzh = state.scratch((nz + 1, ny, nx), "morr_aero_kzh")
    kzh.fill(0)
    fields = getattr(getattr(state, "physics", None), "fields", None) or {}
    exch_h = fields.get("exch_h")
    if exch_h is not None:
        kzh[:nz] = exch_h
    return kzh


def apply(state, cfg, dt: float, *, refl_10cm_due: bool = False):
    """Prepare WRF fields and apply mp_physics = 40 to ``state`` in place.

    aercu_opt = 0 only (WRF refuses the rest without MSKF): constant
    droplet number, no prescribed aerosol.  The package's CCN1_GS..CCN7_GS
    (WRF's zeros here), EFCG/EFIG/EFSG/WACT (written by WRF for aercu_opt >
    0 only) and MSKF_REFL_10CM (REFL_10CM's words with no MSKF) are not
    kept.
    """
    nz, ny, nx = state.p.shape
    required = ("qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng")
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=40 state lacks Morrison fields: "
                         + ", ".join(missing))
    aercu_opt = int(getattr(cfg, "aercu_opt", 0))
    if aercu_opt != 0:
        raise ValueError("mp_physics=40 runs aercu_opt=0 only")
    from woof.core import constants as c
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    theta = state.scratch((nz, ny, nx), "morr_theta")
    rho = state.scratch((nz, ny, nx), "morr_rho")
    pii = state.scratch((nz, ny, nx), "morr_pii")
    dz = state.scratch((nz, ny, nx), "morr_dz")
    work = state.scratch(work_slot_shape(nz, ny * nx), SCRATCH_WORK_SLOT)
    ltrue = state.scratch((ny, nx), "morr_ltrue", dtype=np.int32)
    _prepare_fields(thb, phb, state.thp, state.php,
                    np.int32(ny * nx), np.int32(state.thb.ndim == 3),
                    np.int32(state.phb.ndim == 3), DTYPE(c.G), theta, dz)
    launch_exner(state.p, pii)
    surface = (ny, nx)
    rainnc = state.scratch(surface, "mp_rainnc")
    rainncv = state.scratch(surface, "mp_rainncv")
    snownc = state.scratch(surface, "mp_snownc")
    snowncv = state.scratch(surface, "mp_snowncv")
    graupelnc = state.scratch(surface, "mp_graupelnc")
    graupelncv = state.scratch(surface, "mp_graupelncv")
    sr = state.scratch(surface, "mp_sr")
    from woof.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta)
    save_pre_mp_theta(state)
    cu_rates = (getattr(getattr(state, "physics", None), "cu_rates", None)
                if getattr(cfg, "cu_physics", 0) else None)
    refl = t1d = None
    if refl_10cm_due:
        refl = state.scratch((nz, ny, nx), "refl_10cm")
        t1d = state.scratch((nz, ny, nx), "refl_t")
    launch_morrison_aero(
        theta, state.qv, state.qc, state.qr, state.qi, state.qs, state.qg,
        state.nc, state.nr, state.ni, state.ns, state.ng,
        rho, pii, state.p, dz, state.w, kzh_on_w_levels(state),
        rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr, dt,
        aercu_opt=0,
        effc=state.effc, effr=state.effr, effi=state.effi, effs=state.effs,
        refl_10cm=refl,
        qrcuten=None if cu_rates is None else cu_rates["rqrcuten"],
        qscuten=None if cu_rates is None else cu_rates["rqscuten"],
        qicuten=None if cu_rates is None else cu_rates["rqicuten"],
        morr_rimed_ice=cfg.morr_rimed_ice,
        columns_per_call=work_chunk_columns(ny * nx, nz), t1d_out=t1d,
        _work_scratch=work, _ltrue_scratch=ltrue)
    if refl_10cm_due:
        from woof.core.refl import stash_refl_10cm
        stash_refl_10cm(state, refl)
    moist_physics_finish(state, cfg, theta, dt)
    bound_snow_ratio(sr)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)
