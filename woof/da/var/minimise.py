"""WRFDA's conjugate-gradient minimisation, statement by statement
(docs/dev/WRFDA-3DVAR-PLAN.md, sections 1.3 and 4.7; plan item V1.4).

:func:`minimise_cg` is ``da_minimise_cg``: the gradient at the start,
``phat = -precon * ghat``, per iteration one Hessian-vector product
``fhat`` (:meth:`~woof.da.var.cost.VarCost.calculate_gradj` without a
residual), ``step = rrmold / <fhat, phat>``, ``ghat += step fhat``,
``xhat += step phat``, the optional modified Gram-Schmidt
re-orthonormalisation of ``ghat`` against every stored normalised gradient
(``orthonorm_gradient``, newest first), ``ratio = rrmnew / rrmold``,
``phat = -precon * ghat + ratio * phat``, and ``J`` from the quadratic
identity ``J0 + 0.5 <ghat0, xhat>`` (or recomputed with
``calculate_cg_cost_fn``).  It stops when ``|g| < eps * |g0|`` or after
``ntmax`` iterations.  Inner products are the cost's (``da_dot_cv``: the
serial ``da_dot`` order on the reference and oracle tiers).

Each step prints WRFDA's lines (``1PD22.15``), so a run's log compares with
WRFDA's ``rsl``/stdout as text.  :func:`minimise` is the outer loop
(``max_ext_its``) of ``da_solve``.

Derived from WRFDA 4.7.1 (``var/da/da_minimisation/da_minimise_cg.inc``,
``var/da/da_main/da_solve.inc``), public domain; see
``licenses/LICENSE-WRF-public-domain.txt``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = ["CGResult", "fortran_d", "minimise", "minimise_cg"]

_RULE = "-" * 70


def fortran_d(x: float, w: int = 22, d: int = 15) -> str:
    """Fortran ``1PDw.d``: one digit before the point, ``D`` exponent (gfortran)."""
    x = float(x)
    if not math.isfinite(x):
        s = "NaN" if math.isnan(x) else ("-Infinity" if x < 0 else "Infinity")
        return s.rjust(w) if len(s) <= w else "*" * w
    mant, exp = f"{x:.{d}E}".split("E")
    e = int(exp)
    s = mant + (f"D{e:+03d}" if abs(e) <= 99 else f"{e:+04d}")
    return s.rjust(w) if len(s) <= w else "*" * w


def _line12(it: int, iteration: int, j: float, g: float, step: float) -> str:
    # format 12: (a13,1x,i3,1x,i4,5x,1PD22.15,5x,1PD22.15,5x,1PD22.15)
    return (f"{' minimize_cg ':13s} {it:3d} {iteration:4d}     {fortran_d(j)}     {fortran_d(g)}     "
            f"{fortran_d(step)}")


@dataclass
class CGResult:
    """One outer iteration: per inner iteration ``J``, ``|g|`` and the step, and ``xhat``."""

    it: int
    xhat: object
    j: list = field(default_factory=list)
    g: list = field(default_factory=list)
    step: list = field(default_factory=list)
    iterations: int = 0
    j_final: float = math.nan
    g_final: float = math.nan
    j_grad_norm_target: float = math.nan
    lines: list = field(default_factory=list)

    @property
    def log(self) -> str:
        return "\n".join(self.lines) + "\n"


def _precon(cost, ghat, j0_total: float, precondition_factor: float):
    """``precondition_cg``'s diagonal by control-variable level block (no ensemble: blocks v1..v5)."""
    ops = cost.ops
    cv_u = getattr(cost, "cv_u", cost.cv_size)
    gu = ghat if cv_u == cost.cv_size else ghat[:cv_u]  # U's levels only; VarBC blocks keep 1
    g_total = ops.dot(gu, gu)
    mz = list(cost.u.cv_mz[:5])
    nn = cost.nx * cost.ny
    values = np.ones(cost.cv_size, dtype=np.float64)
    iend = 0
    for nv in range(len(mz)):
        for _ in range(mz[nv]):
            istart = iend
            iend = istart + nn
            g_partial = ops.dot(ghat[istart:iend], ghat[istart:iend])
            jo_partial = j0_total / sum(mz)
            values[istart:iend] = 1.0 / (1.0 + precondition_factor * (g_partial / g_total) / (jo_partial / j0_total))
    return ops.asarray(values)


def minimise_cg(cost, it: int = 1, xhat=None, cv=None, *, ntmax: int = 75, eps: float = 0.01,
                orthonorm_gradient: bool = False, precondition_cg: bool = False,
                precondition_factor: float = 1.0, calculate_cg_cost_fn: bool = False,
                write: Callable[[str], None] | None = None) -> CGResult:
    """``da_minimise_cg`` for outer iteration ``it``; ``xhat`` and ``cv`` default to zero.

    ``write`` receives each stdout line (none when ``None``); the lines are
    also kept in :attr:`CGResult.lines`.
    """
    ops = cost.ops
    xhat = cost.zeros() if xhat is None else ops.asarray(xhat)
    cv = cost.zeros() if cv is None else ops.asarray(cv)
    res = CGResult(it=it, xhat=xhat)

    def out(s: str) -> None:
        res.lines.append(s)
        if write is not None:
            write(s)

    out("Minimize cost function using CG method")
    out("  ")
    j_cost = cost.calculate_j(xhat, cv, 0)
    xc = ops.copy(xhat)
    ops.axpy(1.0, cv, xc)  # xhat + cv
    ghat = cost.calculate_gradj(xc, re=cost.re)
    j0_total = j_cost.total
    if j0_total == 0.0:
        res.j_final = 0.0
        return res
    ghat0 = ops.copy(ghat)
    precon = _precon(cost, ghat, j0_total, precondition_factor) if precondition_cg else None

    def pg():  # precon * ghat (precon = 1.0 is the identity, word for word)
        return ghat if precon is None else ops.mul(precon, ghat)

    t = pg()
    phat = cost.zeros()
    ops.neg_axpby(phat, t, 0.0)  # phat = -precon * ghat
    rrmold = ops.dot(t, ghat)  # da_dot_cv(-phat, ghat); -phat = precon * ghat exactly
    j_grad_norm_target = math.sqrt(rrmold)
    res.j_grad_norm_target = j_grad_norm_target
    qhat = []
    if orthonorm_gradient:
        qhat.append(ops.div(ghat, math.sqrt(rrmold)))
    out(f"Starting outer iteration : {it:3d}")
    out(f"Starting cost function: {fortran_d(j0_total)}, Gradient= {fortran_d(math.sqrt(rrmold))}")
    out(f"For this outer iteration gradient target is:       {fortran_d(eps * j_grad_norm_target)}")
    out(_RULE)
    out("              Loop Iter     Cost Function              Gradient                   Step")
    out(_line12(it, 0, j0_total, math.sqrt(rrmold), 0.0))
    res.j.append(j0_total)
    res.g.append(math.sqrt(rrmold))
    res.step.append(0.0)

    iteration = 0
    for iteration in range(1, ntmax + 1):
        if rrmold == 0.0:
            break
        fhat = cost.calculate_gradj(phat)
        apdotp = ops.dot(fhat, phat)
        step = 0.0
        if apdotp > 0.0:
            step = rrmold / apdotp
        ops.axpy(step, fhat, ghat)
        ops.axpy(step, phat, xhat)
        if orthonorm_gradient:
            for i in range(iteration - 1, -1, -1):
                gdot = ops.dot_device(pg(), qhat[i])  # stays on the device: no sync per vector
                ops.axpy_neg(gdot, qhat[i], ghat)  # ghat - gdot * qhat(i)
        rrmnew = ops.dot(pg(), ghat)
        rrmnew_norm = math.sqrt(rrmnew)
        ratio = 0.0
        if rrmold > 0.0:
            ratio = rrmnew / rrmold
        if orthonorm_gradient:
            qhat.append(ops.div(ghat, rrmnew_norm))
        ops.neg_axpby(phat, pg(), ratio)  # phat = -precon * ghat + ratio * phat
        rrmold = rrmnew
        if calculate_cg_cost_fn:
            j_total = cost.calculate_j(xhat, cv, iteration).total
        else:
            j_total = j0_total + 0.5 * ops.dot(ghat0, xhat)
        out(_line12(it, iteration, j_total, rrmnew_norm, step))
        res.j.append(j_total)
        res.g.append(rrmnew_norm)
        res.step.append(step)
        if rrmnew_norm < eps * j_grad_norm_target:
            break
    else:
        iteration = ntmax + 1
    iteration = min(iteration, ntmax)
    res.iterations = iteration
    out(_RULE)
    out(" ")
    out(f"Inner iteration stopped after {iteration:4d} iterations")
    out(" ")
    j_cost = cost.calculate_j(xhat, cv, iteration)
    xc = ops.copy(xhat)
    ops.axpy(1.0, cv, xc)
    ghat = cost.calculate_gradj(xc, re=cost.re)
    rrmnew_norm = math.sqrt(ops.dot(ghat, ghat))
    res.j_final, res.g_final = j_cost.total, rrmnew_norm
    out(f"Final: {iteration:3d} iter, J={fortran_d(j_cost.total)}, g={fortran_d(rrmnew_norm)}")
    out(_RULE)
    res.xhat = xhat
    return res


def _per_it(value, it: int):
    if isinstance(value, (list, tuple)):
        return value[it - 1]
    return value


def minimise(cost, *, max_ext_its: int = 1, ntmax: int | Sequence[int] = 75, eps: float | Sequence[float] = 0.01,
             relinearise: Callable | None = None, **kw) -> tuple[list[CGResult], object]:
    """``da_solve``'s outer loop around :func:`minimise_cg`; returns the results and ``cvt``.

    Each outer loop starts from ``xhat = 0`` with ``cv = cvt`` (the sum of
    the previous loops' ``xhat``) and ends with ``cvt = cvt + xhat``.  Before
    loop ``it > 1`` the innovations are renewed: ``relinearise(it, cost,
    cvt)`` when given (a nonlinear ``H`` re-run on the updated background),
    else, for a linear ``H'``, ``d = d - H' U xhat`` of the previous loop.
    ``ntmax`` and ``eps`` may be per-loop sequences, as WRFDA's namelist.
    """
    cvt = cost.zeros()
    results = []
    for it in range(1, max_ext_its + 1):
        if it > 1:
            prev = results[-1].xhat
            if relinearise is not None:
                relinearise(it, cost, cvt)
            else:
                ys = cost.vtoy(prev)
                cost.set_innovations([d - y for d, y in zip(cost.d, ys)])
        r = minimise_cg(cost, it, None, cvt, ntmax=_per_it(ntmax, it), eps=_per_it(eps, it), **kw)
        cost.ops.axpy(1.0, r.xhat, cvt)  # cvt = cvt + xhat
        results.append(r)
    return results, cvt
