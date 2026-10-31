"""WRF-Solar cloud fraction, ``icloud = 3`` (G. Thompson's cal_cldfra3).

The radiation driver of WRF v4.7.1 (phys/module_radiation_driver.F
:1462-1510) builds CLDFRA from relative humidity, scale-aware in the grid
spacing, and invents liquid and ice where the fraction is partial
(find_cloudLayers, adjust_cloudIce/H2O, adjust_cloudFinal).  The invented
condensate is radiation's only: the driver saves qc/qi on entry
(:1210-1230) and restores them after the schemes ran (:3137-3155), so the
prognostic mixing ratios never see it.  CLDFRA itself is the driver's
grid%cldfra, the one written to history.

:func:`cal_cldfra3` runs the kernel (woof/core/kernels/cldfra3.cu) over
column-major ``(ncol, nz)`` copies; tools/cldfra3_wrf471_oracle is the
gfortran -O0 reference it is graded against word for word.

WRF-Solar EPS (``multi_perturb = 1`` with ``pert_cld3``) perturbs the
temperature and vapour cal_cldfra3 reads, ``t*(1 + perts_th*pert_cld3_t)``
and ``max(0, qv*(1 + perts_qvapor*pert_cld3_qv))`` (:1472-1474).  That is
the ``perturbation`` hook below: the inputs are perturbed before the call
and nothing inside the routine changes.  ``perturbation`` carries the two
pert3d fields in the caller's column layout and the (pert_cld3_qv,
pert_cld3_t) factors; woof.core.wrfsolar_eps.cld3_inputs applies the two
statements word for word (tests/test_wrfsolar_eps_wrf471_oracle.py).
"""

from __future__ import annotations

import numpy as np

__all__ = ["cal_cldfra3", "cldfra3_gridkm", "Cldfra3Fatal"]


class Cldfra3Fatal(RuntimeError):
    """WRF's wrf_error_fatal('FATAL ERROR, problem in temperature profile.')."""


def cldfra3_gridkm(dx, dy):
    """The driver's ``gridkm`` (module_radiation_driver.F:1055-1056).

    ``SQRT(dx*0.001*dx*0.001 + dy*0.001*dy*0.001)`` in default REAL, left to
    right; WRF's ARW step passes dx and dy, never ``dxkm``.
    """
    f = np.float32
    dx, dy, m = f(dx), f(dy), f(0.001)
    with np.errstate(all="raise"):
        return f(np.sqrt(f(f(f(dx * m) * dx) * m) + f(f(f(dy * m) * dy) * m)))


def cal_cldfra3(qv, qc, qi, qs, dz8w, p, t, xland, gridkm, *,
                perturbation=None, check=True):
    """Return ``(cldfra, qc_rad, qi_rad)`` for ``(ncol, nz)`` float32 columns.

    Level 1 is the lowest model level.  ``qc``/``qi`` are not modified: the
    radiation copies are returned.  ``xland`` is ``(ncol,)``.  ``check``
    reads the per-column fatal flag back (one small device read) and raises
    :class:`Cldfra3Fatal` where WRF stops.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel
    if perturbation is not None:
        # WRF-Solar EPS (multi_perturb = 1, pert_cld3): t_1d and qv_1d are
        # perturbed copies (module_radiation_driver.F:1472-1474).
        from woof.core.wrfsolar_eps import cld3_inputs
        f_qv, f_t = perturbation["factors"]
        t, qv = cld3_inputs(cp.asarray(t), cp.asarray(qv),
                            perturbation["pert_th"], perturbation["pert_qv"],
                            f_qv, f_t)

    def col(a, name):
        a = cp.asarray(a)
        if a.dtype != cp.float32 or a.ndim != 2:
            raise TypeError(f"cal_cldfra3: {name} must be float32 (ncol, nz)")
        return cp.ascontiguousarray(a)

    qv = col(qv, "qv")
    ncol, nz = qv.shape
    arrays = {}
    for name, a in (("qs", qs), ("dz8w", dz8w), ("p", p), ("t", t)):
        arrays[name] = col(a, name)
        if arrays[name].shape != qv.shape:
            raise ValueError(f"cal_cldfra3: {name} shape {arrays[name].shape} != {qv.shape}")
    qc_rad = col(qc, "qc").copy()
    qi_rad = col(qi, "qi").copy()
    if qc_rad.shape != qv.shape or qi_rad.shape != qv.shape:
        raise ValueError("cal_cldfra3: qc/qi shape mismatch")
    xland = cp.ascontiguousarray(cp.asarray(xland, dtype=cp.float32).reshape(-1))
    if xland.size != ncol:
        raise ValueError("cal_cldfra3: xland must hold one value per column")
    if nz < 6:
        raise ValueError("cal_cldfra3 needs at least six levels (find_cloudLayers' tropopause search)")
    cldfra = cp.zeros((ncol, nz), dtype=cp.float32)
    if ncol == 0:
        return cldfra, qc_rad, qi_rad
    work = cp.empty(4 * ncol * nz, dtype=cp.float32)
    status = cp.zeros(ncol, dtype=cp.int32)
    block = 64
    get_kernel("cldfra3", "cldfra3_columns")(
        ((ncol + block - 1) // block,), (block,),
        (np.int32(ncol), np.int32(nz), qv, qc_rad, qi_rad, arrays["qs"],
         arrays["dz8w"], arrays["p"], arrays["t"], xland,
         np.float32(gridkm), cldfra, work, status))
    if check:
        bad = int(cp.count_nonzero(status))
        if bad:
            raise Cldfra3Fatal(
                f"icloud=3: in {bad} column(s) the -12 C level lies within "
                "three levels of the model top -- WRF's find_cloudLayers "
                "stops here ('FATAL ERROR, problem in temperature profile.')")
    return cldfra, qc_rad, qi_rad
