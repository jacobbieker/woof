"""MAD-WRF: WRF v4.7.1's solar-irradiance nowcast configuration.

What WRF v4.7.1 does (doc/README.madwrf; phys/module_madwrf.F;
Registry.EM_COMMON:2677-2680, :3052, :3071-3072):

* ``mp_physics = 96`` (package ``madwrf_mp``, moist qv, qc, qi, qs): the
  microphysics driver's ``CASE (MADWRF_MP)`` does nothing
  (module_microphysics_driver.F:2895), so the hydrometeors are advected and
  diffused with no microphysical process.  ``madwrf_opt = 1`` requires it
  and ``use_mp_re = 0`` (module_check_a_mundo.F:375-381).
* ``madwrf_opt = 2`` (package ``madwrf``, tracer TR_QC, TR_QI, TR_QS):
  real.exe sets the tracers to qc, qi, qs (Init_madwrf_tracers,
  module_initialize_real.F:4868-4874); they are advected and diffused as
  WRF tracers; every step, before moist_physics_prep and the microphysics,
  Cloud_tracer_nudge relaxes qc, qi, qs toward them, q += (tr - q)*dt /
  madwrf_dt_relax, while xtime < madwrf_dt_nudge (minutes;
  solve_em.F:3646-3662, module_big_step_utilities_em.F:6711-6756), over the
  scalar tile that leaves the specified/nest boundary zone out.  It needs a
  scheme with qc, qi and qs (module_physics_init.F:4693).
* ``madwrf_cldinit = 1``: real.exe's Init_madwrf_clouds
  (module_initialize_real.F:4861-4866) rebuilds qv, qc, qi, qs and CLDFRA
  column by column with a variant of G. Thompson's cal_cldfra3
  (modify_qvapor = .true.): RH-based cloud fraction, made-up condensate
  where the fraction is partial, and -- when metgrid supplies them --
  satellite constraints: CLDMASK removes cloud from clear columns, BRTEMP or
  CLDTOPZ moves the cloud top (BRTEMP through Calc_cldtopz_from_brtemp, from
  the tropopause down to the first level warmer than the brightness
  temperature), CLDBASEZ the base.

Ported here: the column routines (woof/core/kernels/madwrf.cu, graded word
for word against compiled WRF by tools/madwrf_wrf471_oracle) and their
drivers: :func:`cloud_init` for a real.exe-style initialization,
:func:`nudge` for the step.
"""

from __future__ import annotations

import math

import numpy as np

F = np.float32
MISSING = F(-999.9)
_BLOCK = 64


class MadwrfFatal(RuntimeError):
    """WRF's wrf_error_fatal('FATAL ERROR, problem in temperature profile.')."""


def _kernel(name):
    from woof.core.kernels import get_kernel
    return get_kernel("madwrf", name)


def insert_clouds(flag_cldmask: int, flag_cldtopz: int, flag_cldbasez: int,
                  flag_brtemp: int) -> int:
    """Init_madwrf_clouds' method from the met_em flags (:104-107)."""
    mode = 0
    if flag_cldmask == 1 and flag_cldtopz == 0 and flag_brtemp == 0:
        mode = 1
    if (flag_cldmask == 1 and flag_brtemp == 1) or flag_cldtopz == 1:
        mode = 2
    if mode == 2 and flag_cldbasez == 1:
        mode = 3
    return mode


def cloud_gridkm(dx, dy) -> np.float32:
    """``gridkm = sqrt(dx*dy)*CONVERT_M_TO_KM`` (:115)."""
    return F(F(np.sqrt(F(F(dx) * F(dy)))) * F(0.001))


def cloud_init(*, t2, p, ph, phb, qv, qc, qi, qs, xland, ht, dx, dy,
               cldmask=None, cldtopz=None, cldbasez=None, brtemp=None,
               flags=(0, 0, 0, 0), p00=1.0e5, check=True):
    """Init_madwrf_clouds over ``(ncol, nz)`` float32 columns (level 1 at
    the surface): ``t2`` perturbation theta (theta - 300), ``p`` the
    hydrostatic pressure, ``ph``/``phb`` (ncol, nz+1) geopotential.  qv, qc,
    qi, qs are rewritten in place; returns ``(cldfra (ncol, nz+1), cldmask,
    cldtopz)`` (the last two updated as real.exe updates them).  ``flags``
    is (flag_cldmask, flag_cldtopz, flag_cldbasez, flag_brtemp)."""
    import cupy as cp

    def col(a, name, width):
        a = cp.asarray(a)
        if a.dtype != cp.float32 or a.ndim != 2 or a.shape[1] != width:
            raise TypeError(f"madwrf.cloud_init: {name} must be float32 (ncol, {width})")
        if not a.flags.c_contiguous:
            raise ValueError(f"madwrf.cloud_init: {name} must be C-contiguous")
        return a

    ncol, nz = cp.asarray(t2).shape
    t2, p = col(t2, "t2", nz), col(p, "p", nz)
    ph, phb = col(ph, "ph", nz + 1), col(phb, "phb", nz + 1)
    q = [col(a, n, nz) for a, n in ((qv, "qv"), (qc, "qc"), (qi, "qi"), (qs, "qs"))]
    if nz < 6:
        raise ValueError("Init_madwrf_clouds needs at least six levels")

    def surf(a, fill):
        if a is None:
            return cp.full(ncol, fill, dtype=cp.float32)
        return cp.ascontiguousarray(cp.asarray(a, dtype=cp.float32).reshape(-1))
    xland, ht = surf(xland, 1.0), surf(ht, 0.0)
    cldmask, cldtopz = surf(cldmask, MISSING).copy(), surf(cldtopz, MISSING).copy()
    cldbasez, brtemp = surf(cldbasez, MISSING), surf(brtemp, MISSING)
    cldfra = cp.zeros((ncol, nz + 1), dtype=cp.float32)
    if ncol == 0:
        return cldfra, cldmask, cldtopz
    work = cp.zeros(8 * (nz + 1) * ncol, dtype=cp.float32)
    status = cp.zeros(ncol, dtype=cp.int32)
    fm, ft, fb, fbr = (int(v) for v in flags)
    _kernel("madwrf_cloud_init_columns")(
        ((ncol + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
        (np.int32(ncol), np.int32(nz), F(p00), cloud_gridkm(dx, dy),
         np.int32(insert_clouds(fm, ft, fb, fbr)), np.int32(ft), np.int32(fbr),
         t2, p, ph, phb, *q, xland, ht, cldmask, cldtopz, cldbasez, brtemp,
         cldfra, work, status))
    if check:
        bad = int(cp.count_nonzero(status))
        if bad:
            raise MadwrfFatal(
                f"madwrf_cldinit: in {bad} column(s) the -12 C or -40 C level "
                "lies within three levels of the model top -- WRF's "
                "find_cloudLayers stops ('FATAL ERROR, problem in temperature "
                "profile.')")
    return cldfra, cldmask, cldtopz


def nudge_due(xtime_minutes: float, dt_nudge: float) -> bool:
    """``if (xtime < dt_nudge)`` in float32 (WRF's REAL xtime)."""
    return bool(F(xtime_minutes) < F(dt_nudge))


def nudge(qc, qi, qs, trqc, trqi, trqs, *, dt, dt_relax, xtime_minutes,
          dt_nudge, tile=None):
    """Cloud_tracer_nudge in place on (nz, ny, nx) arrays.  ``tile`` is
    ``(i0, i1, j0, j1)`` (half-open, 0-based), the scalar tile; None is the
    whole plane.  Returns whether it nudged."""
    if not nudge_due(xtime_minutes, dt_nudge):
        return False
    nz, ny, nx = qc.shape
    i0, i1, j0, j1 = (0, nx, 0, ny) if tile is None else tile
    n = max(i1 - i0, 0) * max(j1 - j0, 0) * nz
    if n == 0:
        return True
    for a in (qc, qi, qs, trqc, trqi, trqs):
        if a.shape != qc.shape or not a.flags.c_contiguous or a.dtype != np.float32:
            raise ValueError("madwrf.nudge: six C-contiguous float32 arrays of one shape")
    _kernel("madwrf_nudge")(
        ((n + 255) // 256,), (256,),
        (np.int32(nz), np.int32(ny), np.int32(nx), np.int32(i0), np.int32(i1),
         np.int32(j0), np.int32(j1), F(dt), F(dt_relax), qc, qi, qs,
         trqc, trqi, trqs))
    return True


#: The chem species set that carries TR_QC/TR_QI/TR_QS (WRF's tracer array,
#: package madwrf): advected and diffused, zero inflow.
TRACER_SET = "madwrf"
TRACERS = ("tr_qc", "tr_qi", "tr_qs")


def scalar_tile(cfg, ny: int, nx: int):
    """solve_em.F:3632-3643's scalar tile in 0-based half-open form: the
    specified/nest ring of width ``spec_zone`` left out (periodic x keeps
    its full rows)."""
    specified = bool(getattr(cfg, "specified", False)) or bool(getattr(cfg, "nested", False))
    sz = int(getattr(cfg, "spec_zone", 1)) if specified else 0
    if getattr(cfg, "periodic_x", False):
        i0, i1 = 0, nx
    else:
        i0, i1 = sz, nx - sz
    return i0, i1, sz, ny - sz


def finite_positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) and value > 0


def apply(state, cfg, dt, *, refl_10cm_due=False):
    """mp_physics = 96 (CASE (MADWRF_MP), module_microphysics_driver.F:2895):
    no microphysical process.  The driver's prep and finish run around it as
    around every scheme (the theta it hands back is the theta it took, so
    h_diabatic is zero); RAINNC/RAINNCV and SR stay zero."""
    import cupy as cp
    from woof.core.microphysics import (MicrophysicsDiagnostics,
                                        moist_physics_finish, save_pre_mp_theta)
    nz, ny, nx = state.p.shape
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    th = state.scratch((nz, ny, nx), "mp_th")
    th[...] = thb + state.thp
    save_pre_mp_theta(state)
    if refl_10cm_due:
        # WRF computes none for MADWRF_MP; woof's history diagnostic is the
        # Kessler operator on the zero rain (woof.core.refl).
        from woof.core.microphysics import prepare_microphysics_exner
        from woof.core.refl import compute_and_stash_refl_10cm
        pii = state.scratch((nz, ny, nx), "mp_pii")
        prepare_microphysics_exner(state.p, pii)
        refl_t = state.scratch((nz, ny, nx), "refl_t")
        refl_t[...] = th * pii
        compute_and_stash_refl_10cm(state, cfg, refl_t, state.p)
    moist_physics_finish(state, cfg, th, dt)
    rainnc = state.scratch((ny, nx), "mp_rainnc")
    rainncv = state.scratch((ny, nx), "mp_rainncv")
    rainncv[...] = 0.0
    sr = state.scratch((ny, nx), "mp_sr")
    sr[...] = 0.0
    return MicrophysicsDiagnostics(rainnc=rainnc, rainncv=rainncv, sr=sr)


def tracer_fields(state):
    """(tr_qc, tr_qi, tr_qs) arrays of the domain's chem arena."""
    from woof.core.chem_state import chem_state_of
    chem = chem_state_of(state)
    if chem is None:
        raise ValueError("madwrf_opt = 2 carries TR_QC/TR_QI/TR_QS in the chem "
                         "arena, and this state has none")
    rows = {row.name: row for row in chem.transported}
    missing = [name for name in TRACERS if name not in rows]
    if missing:
        raise ValueError(f"madwrf_opt = 2: the chem arena lacks {missing}")
    return tuple(getattr(state, rows[name].state_attr) for name in TRACERS)


def seed_tracers(state) -> None:
    """Init_madwrf_tracers (module_madwrf.F:206-237): tr = q on every mass
    point and level, as real.exe leaves them in wrfinput."""
    for tr, q in zip(tracer_fields(state), (state.qc, state.qi, state.qs)):
        tr[...] = q


def step_nudge(state, cfg) -> bool:
    """solve_em.F:3646-3662 for this step: Cloud_tracer_nudge over the
    scalar tile while xtime (minutes since the domain started, at the
    step's start) is below madwrf_dt_nudge.  Returns whether it nudged."""
    if int(getattr(cfg, "madwrf_opt", 0)) != 2:
        return False
    epoch = float(getattr(state, "domain_start_offset", 0.0) or 0.0)
    xtime = F(F(float(state.elapsed_seconds) - epoch) / F(60.0))
    nz, ny, nx = state.qc.shape
    trqc, trqi, trqs = tracer_fields(state)
    return nudge(state.qc, state.qi, state.qs, trqc, trqi, trqs,
                 dt=float(cfg.dt), dt_relax=float(cfg.madwrf_dt_relax),
                 xtime_minutes=xtime, dt_nudge=float(cfg.madwrf_dt_nudge),
                 tile=scalar_tile(cfg, ny, nx))


#: The metgrid fields Init_madwrf_clouds reads (Registry.EM_COMMON:3071,
#: package cldinit, state:brtemp,cldmask,cldtopz,cldbasez), with the
#: FLAG_* attributes module_initialize_real.F passes beside them.
OBSERVATION_FIELDS = ("CLDMASK", "CLDTOPZ", "CLDBASEZ", "BRTEMP")


def observations_from_metem(case):
    """``{name: (ny, nx) array, "flags": (cldmask, cldtopz, cldbasez,
    brtemp)}`` from one met_em case; absent fields are WRF's missing
    value with a zero flag."""
    out = {}
    flags = []
    for name in OBSERVATION_FIELDS:
        field = case.statics.get(name)
        raw = case.attributes.get(f"FLAG_{name}")
        flag = int(float(raw)) if raw is not None and field is not None else 0
        out[name] = None if field is None else np.asarray(field, dtype=np.float32)
        flags.append(flag)
    cm, ct, cb, br = flags
    out["flags"] = (cm, ct, cb, br)
    return out


def cloud_init_state(state, cfg, *, p_hyd, terrain, landmask=None,
                     observations=None, p00=1.0e5) -> dict:
    """real.exe's Init_madwrf_clouds on a freshly initialized state, after
    its geopotential and hydrostatic pressure are final and with no
    rebalance after it (module_initialize_real.F:4861-4866): qv, qc, qi and
    qs are rewritten in place.  ``p_hyd`` is the (nz, ny, nx) hydrostatic
    pressure, ``terrain`` HGT, ``landmask`` 1 over land (XLAND = 2 -
    LANDMASK).  Returns a small receipt."""
    import cupy as cp
    nz, ny, nx = state.qv.shape
    ncol = ny * nx

    def cols(a, width):
        a = cp.asarray(np.asarray(cp.asnumpy(a), dtype=np.float32)
                       if not isinstance(a, cp.ndarray) else a, dtype=cp.float32)
        if a.ndim == 1:
            a = cp.broadcast_to(a[:, None, None], (a.shape[0], ny, nx))
        return cp.ascontiguousarray(a.reshape(width, ncol).T)

    theta = state.total_theta()
    t2 = cols(cp.asarray(theta, dtype=cp.float32) - cp.float32(300.0), nz)
    p = cols(p_hyd, nz)
    ph = cols(state.php, nz + 1)
    phb = cols(state.phb, nz + 1)
    species = [cols(getattr(state, n), nz) for n in ("qv", "qc", "qi", "qs")]
    surf = lambda a, fill: (cp.full(ncol, fill, dtype=cp.float32) if a is None else
                            cp.asarray(np.asarray(cp.asnumpy(a) if isinstance(a, cp.ndarray) else a,
                                                  dtype=np.float32).reshape(-1)))
    xland = (cp.ones(ncol, dtype=cp.float32) if landmask is None else
             cp.float32(2.0) - surf(landmask, 1.0))
    obs = observations or {"flags": (0, 0, 0, 0)}
    cf, _, _ = cloud_init(
        t2=t2, p=p, ph=ph, phb=phb, qv=species[0], qc=species[1], qi=species[2],
        qs=species[3], xland=xland, ht=surf(terrain, 0.0), dx=cfg.dx, dy=cfg.dy,
        cldmask=None if obs.get("CLDMASK") is None else surf(obs["CLDMASK"], MISSING),
        cldtopz=None if obs.get("CLDTOPZ") is None else surf(obs["CLDTOPZ"], MISSING),
        cldbasez=None if obs.get("CLDBASEZ") is None else surf(obs["CLDBASEZ"], MISSING),
        brtemp=None if obs.get("BRTEMP") is None else surf(obs["BRTEMP"], MISSING),
        flags=obs["flags"], p00=p00)
    for name, column in zip(("qv", "qc", "qi", "qs"), species):
        target = getattr(state, name)
        grid = cp.ascontiguousarray(column.T).reshape(nz, ny, nx)
        if isinstance(target, cp.ndarray):
            target[...] = grid
        else:
            target[...] = cp.asnumpy(grid)
    return {"madwrf_cldinit": {"insert_clouds": insert_clouds(*obs["flags"]),
                               "flags": dict(zip(OBSERVATION_FIELDS, obs["flags"])),
                               "cloudy_columns": int(cp.count_nonzero(cf.max(axis=1) > 0))}}
