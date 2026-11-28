"""WRFDA 4D-Var's outer loops (EXPERIMENTAL; docs/dev/WRFDA-4DVAR-PLAN.md, PR 4D.4).

``da_solve`` with ``var4d = .true.`` and ``max_ext_its > 1`` (WRFDA 4.7.1
``var/da/da_main/da_solve.inc``, section [8]), loop ``it``:

1. ``xhat = 0`` (``da_initialize_cv``, line 805); ``cvt`` carries the earlier loops'
   ``xhat`` summed (zero in the first loop).
2. For ``it > 1``, ``da_nl_model(it)`` reruns the nonlinear model over the
   window from ``grid``, the analysis the previous loop left there
   (lines 827-922: ``grid`` copied into ``model_grid``, its lateral
   boundaries included, then ``da_nl_model(it)``).  The TL and AD models are linearised about that
   run (WRFPLUS reads its states every time step).
3. ``da_get_innov_vector(it)`` (line 933): every slot's innovations against that
   run's state at the slot time.  The var4d branch does not restore ``fg``
   after the slot loop (``da_get_innov_vector.inc`` line 254), so the
   background ``U``, the transfer and the analysis are linearised about
   slot 1's state: the rerun's start, the previous loop's analysis.
4. ``da_minimise_cg(it, xhat, cvt)``: ``Jb = 0.5 (cvt + xhat)^T (cvt +
   xhat)`` (``da_calculate_j.inc`` line 160), so ``Jb`` stays measured
   from the FIRST guess, while ``Jo`` is the new innovations' minus ``H'
   M' U xhat``.  ``J`` and the gradient target restart in every loop
   (``da_minimise_cg.inc`` line 138); each loop logs its own
   ``Starting cost function`` and ``Final`` ``J``.
5. ``cvt = cvt + xhat`` (line 995); the increment ``xa = U xhat`` is THIS
   loop's ``xhat`` only (line 1032, the var4d branch), transferred with
   ``da_transfer_xatowrf`` and added to ``grid``, which already holds the
   earlier loops' increments (``da_transfer_xatoanalysis``, line 1102).

:func:`outer_loops` is that loop over pluggable pieces: ``setup(it,
trajectory)`` builds loop ``it``'s cost about the run (step 3 and the cost
of step 4), ``update(it, context, xhat)`` the analysis after it (step 5),
and ``nl_model(it, state)`` the rerun (step 2).  It relinearises ``model``
(``model.relinearise(trajectory)``) before each loop after the first.
:func:`woof.da.var.analysis.analyse` drives it with WRFDA's pieces; the
tests drive it with toys and with P3a's real window.

Derived from WRFDA 4.7.1 (``var/da/da_main/da_solve.inc``,
``da_minimisation/da_calculate_j.inc``, ``da_minimise_cg.inc``,
``da_get_innov_vector.inc``, ``da_transfer_model/da_transfer_xatoanalysis.inc``),
public domain; see ``licenses/LICENSE-WRF-public-domain.txt``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = ["Trajectory", "outer_loops", "relinearise"]


@dataclass
class Trajectory:
    """One nonlinear run over the window, as an ``nl_model`` returns it.

    ``slots``: the run's state at each slot time, ``slots[0]`` the state it
    started from (for :func:`~woof.da.var.analysis.analyse`, a
    :class:`~woof.da.var.analysis.FirstGuess` or a WRF file per slot, as
    ``fgat=`` takes them: WRFDA's ``nl_d01_<date>``).

    ``states``: the run's model state at every time step, what
    ``Window.forward(trajectory=...)`` takes (``nsteps`` or ``nsteps + 1``
    STEP_INPUTS dicts) to relinearise a
    :class:`~woof.da.var.window4d.WindowModel`; ``None`` for a TL model
    without a base state (:class:`~woof.da.var.fourd.IdentityModel`).

    ``start``: instead of ``states``, the state the run started from (a
    STEP_INPUTS dict) when the run is the window's own model: the
    WindowModel reruns it under its checkpoint policy rather than holding
    every step's state.
    """

    slots: Sequence
    states: Sequence | None = None
    start: dict | None = None


def relinearise(model, trajectory: Trajectory) -> None:
    """Linearise ``model`` about ``trajectory`` (``model.relinearise``), refusing a model that cannot be."""
    if not isinstance(trajectory, Trajectory):
        raise TypeError(f"nl_model returned {type(trajectory).__name__}, not a Trajectory")
    fn = getattr(model, "relinearise", None)
    if fn is None:
        raise TypeError(f"{type(model).__name__} has no relinearise(trajectory): an outer loop would keep "
                        "its TL and AD linearised about the first loop's trajectory")
    fn(trajectory)


def _per_it(value, it: int):
    return value[it - 1] if isinstance(value, (list, tuple)) else value


def outer_loops(setup: Callable, update: Callable, first: Trajectory, *, max_ext_its: int = 1,
                nl_model: Callable | None = None, model=None, ntmax=75, eps=0.01,
                write: Callable[[str], None] | None = None, **cg_kw):
    """``da_solve``'s var4d outer loop (module docstring).

    ``setup(it, trajectory) -> (cost, context)``: loop ``it``'s
    :class:`~woof.da.var.cost.VarCost` about ``trajectory``;
    ``update(it, context, xhat) -> state``: the analysis after loop ``it``
    (its ``xhat`` added to the state ``context`` was linearised about);
    ``nl_model(it, state) -> Trajectory``: the rerun from that state.
    ``first`` is loop 1's trajectory.  ``ntmax`` and ``eps`` may be per-loop
    sequences; ``cg_kw`` goes to
    :func:`~woof.da.var.minimise.minimise_cg`.

    Returns ``(results, cvt, state, context)``: the per-loop
    :class:`~woof.da.var.minimise.CGResult` (each loop's ``J`` is its own,
    ``results[-1].j_final`` the reported one), ``cvt`` (every loop's
    ``xhat`` summed), the final analysis and the last loop's context.
    """
    from .minimise import minimise_cg

    if max_ext_its < 1:
        raise ValueError(f"max_ext_its must be >= 1, not {max_ext_its}")
    if max_ext_its > 1 and nl_model is None:
        raise NotImplementedError("4D-Var with max_ext_its > 1 needs nl_model (WRFDA reruns the nonlinear model "
                                  "from the updated analysis in every outer loop)")
    trajectory, cvt, state, results = first, None, None, []
    context = None
    for it in range(1, max_ext_its + 1):
        if it > 1:
            trajectory = nl_model(it, state)
            relinearise(model, trajectory)
        cost, context = setup(it, trajectory)
        if cvt is None:
            cvt = cost.zeros()
        elif int(cvt.size) != cost.cv_size:
            raise ValueError(f"outer loop {it}: the control vector has {cost.cv_size} elements, loop 1's "
                             f"{int(cvt.size)}")
        r = minimise_cg(cost, it, None, cvt, ntmax=_per_it(ntmax, it), eps=_per_it(eps, it), write=write,
                        **cg_kw)
        cost.ops.axpy(1.0, r.xhat, cvt)  # cvt = cvt + xhat
        results.append(r)
        state = update(it, context, r.xhat)
    return results, cvt, state, context
