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
