"""A window of strict dry steps: the forward model, its tangent-linear and
its adjoint over N steps with checkpoints (P3a item 8, for 4D-Var).

States in and out are dicts over :data:`model.STEP_INPUTS` (the
prognostics and the previous step's diagnosis).  State ``k`` is the state
after ``k`` steps, ``0 <= k <= nsteps``.

- :meth:`Window.forward` runs the base and keeps a checkpoint every
  ``checkpoint_every`` steps.
- :meth:`Window.tl` maps a perturbation of state 0 through the window and
  returns the perturbation at the requested states.  With
  ``partial=True`` the perturbation may name any subset of STEP_INPUTS
  and the rest start at zero: WRFDA's TL start, which passes the
  prognostics and p' with g_ph = g_al = 0 (:data:`WRFDA_TL_START`).
  Names outside STEP_INPUTS are refused on every call.  The
  adjoint of that zero fill is a restriction, so :meth:`Window.ad` needs
  no counterpart: take the same names from its result.
- :meth:`Window.ad` takes the adjoint at the final state and adjoint
  forcing at any states (the observation terms of a 4D-Var cost), and
  returns the adjoint of state 0.

The adjoint rebuilds each step's tape from the checkpoint before it,
through the base-only kernel build (:func:`model.step_base`): each segment
of ``checkpoint_every`` steps is rerun once to recover its step-start
states, then each step is taped and swept in reverse.  A checkpoint is a
state's STEP_INPUTS alone: a step reads nothing else its predecessor
wrote (the scratch it uses it writes first, which
test_grad_wrfplus_window.py pins by poisoning it), so one work dict
serves every step.  Memory: one checkpoint per segment, one segment's
step-start states, the work dict and one step's tape.

Everything :func:`model.step_geometry` refuses is refused here, by name.
"""

from __future__ import annotations

import numpy as np

from . import model as M


#: The names WRFDA's TL start perturbs (da_transfer_xatowrftl): the
#: prognostics but phi' (g_ph_2 = 0), and g_p.  woof carries WRF's
#: perturbation pressure as ``p_pert`` and the full pressure as ``p``;
#: both have the tangent g_p, so both take it.  ``muts`` is WRF's carried
#: ``MUT + MU''``; WRFDA's window starts where its nonlinear run does,
#: from start_em's ``muts = mub + mu'``, so its tangent there is g_mu:
#: pass ``dx0["muts"] = dx0["mup"]``.  With ``Window.tl(dx0,
#: partial=True)`` the rest (phi', al and alt) start at zero.
#:
#: So on input the ``p`` and ``p_pert`` tangents are one array (g_p), and
#: ``muts`` and ``mup`` are one (g_mu); the adjoint of that copy is a sum:
#: g_p's adjoint is ``adj["p"] + adj["p_pert"]`` and g_mu's
#: ``adj["mup"] + adj["muts"]``.  :func:`wrfda_tl_start` and
#: :func:`wrfda_tl_start_ad` build and restrict them.
WRFDA_TL_START = ("u", "v", "w", "thp", "mup", "p", "p_pert", "muts")

#: What :func:`wrfda_tl_start` takes: WRFDA's g_u, g_v, g_w, g_t (as
#: theta'), g_mu and g_p, by woof's names.
WRFDA_TL_FIELDS = ("u", "v", "w", "thp", "mup", "p")

#: The copies WRFDA's start makes: target <- source.
_WRFDA_COPIES = {"p_pert": "p", "muts": "mup"}


def wrfda_tl_start(g: dict) -> dict:
    """WRFDA's TL start for ``Window.tl(..., partial=True)`` from
    ``g`` over :data:`WRFDA_TL_FIELDS` (no other name): g_p into both
    pressure carriers and g_mu into ``muts`` too (the arrays are shared,
    not copied; ``Window.tl`` only reads them)."""
    if set(g) != set(WRFDA_TL_FIELDS):
        raise ValueError(f"wrfda_tl_start takes {list(WRFDA_TL_FIELDS)}, got {sorted(g)}")
    return {**g, **{t: g[src] for t, src in _WRFDA_COPIES.items()}}


def wrfda_tl_start_ad(adj: dict) -> dict:
    """The adjoint of :func:`wrfda_tl_start`: :data:`WRFDA_TL_FIELDS` of
    ``adj`` (a ``Window.ad`` result), with each copy's adjoint summed back
    into its source (``p + p_pert``, ``mup + muts``); new arrays."""
    out = {n: adj[n].copy() for n in WRFDA_TL_FIELDS}
    for t, src in _WRFDA_COPIES.items():
        out[src] += adj[t]
    return out


def _copy(d: dict, names=None) -> dict:
    return {n: d[n].copy() for n in (d if names is None else names)}


class Window:
    """``nsteps`` strict steps of ``cfg`` from ``state`` (a DomainState at
    the window start; a cold state diagnoses on its first step), in
    ``dtype`` (binary64 for the 4D-Var lane, binary32 to reproduce the
    primal's words)."""

    def __init__(self, state, cfg, nsteps: int, *, dtype=np.float64,
                 checkpoint_every: int = 1):
        if nsteps < 1:
            raise ValueError(f"nsteps must be >= 1, got {nsteps}")
        if checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
        self.nsteps = int(nsteps)
        self.every = int(checkpoint_every)
        self.geoms = M.step_geometries(state, cfg, self.nsteps)
        self._start = M.run_state(state, cfg, dtype)
        self._scratch = None
        self._mutable = self._written_names()
        missing = set(M.STEP_INPUTS) - set(self._mutable)
        if missing:   # the work dict would write into the shared start arrays
            raise AssertionError(f"a step does not write {sorted(missing)}")
        # the one dict every step runs in: the window's constant arrays
        # (shared) and its own copies of what a step writes
        self._work = {**self._start, **_copy(self._start, self._mutable)}
        self._checkpoints: dict[int, dict] | None = None

    # ---------------------------------------------------------------- helpers

    def _written_names(self) -> tuple:
        """Every name a step writes (its tape's keys and the time-t
        copies): what a checkpoint must hold.  Found by taping one step on
        a copy; the rest of the dict is constant over the window."""
        import cupy as cp
        x = _copy(self._start)
        tape = []
        self._scratch = M.step_base(x, self.geoms[0], tape)
        names = set()
        for snap in tape:
            names.update(snap)
        del x, tape
        cp.get_default_memory_pool().free_all_blocks()
        return tuple(n for n in self._start if n in names)

    def _check(self, d: dict, what: str, partial: bool = False) -> None:
        """``d`` holds every STEP_INPUTS name (any subset if ``partial``,
        and no other name), shaped and typed as the window's (no silent
        cast or broadcast)."""
        unknown = sorted(set(d) - set(M.STEP_INPUTS))
        if unknown:
            raise ValueError(f"{what} names {unknown}, not STEP_INPUTS {list(M.STEP_INPUTS)}")
        missing = [n for n in M.STEP_INPUTS if n not in d]
        if missing and not partial:
            raise ValueError(f"{what} lacks {missing}; pass partial=True to start "
                             "them at zero")
        for n in M.STEP_INPUTS:
            if n not in d:
                continue
            a, ref = d[n], self._start[n]
            if a.shape != ref.shape or a.dtype != ref.dtype:
                raise ValueError(f"{what}[{n!r}] is {a.dtype}{tuple(a.shape)}, "
                                 f"the window's is {ref.dtype}{tuple(ref.shape)}")

    def _load(self, ck: dict) -> dict:
        """The work dict holding state ``ck`` (STEP_INPUTS)."""
        for n in M.STEP_INPUTS:
            self._work[n][...] = ck[n]
        return self._work

    def _inputs(self, d: dict) -> dict:
        return {n: d[n] for n in M.STEP_INPUTS}

    # ---------------------------------------------------------------- forward

    def forward(self, x0: dict | None = None) -> dict:
        """Run the base from state 0 (the window's start state, or
        ``x0``'s STEP_INPUTS) and keep the checkpoints the adjoint uses.
        Returns the final state's STEP_INPUTS."""
        if x0 is not None:
            self._check(x0, "x0")
        x = self._load(self._inputs(self._start) if x0 is None else x0)
        self._checkpoints = {}
        for k, g in enumerate(self.geoms):
            if k % self.every == 0:
                self._checkpoints[k] = _copy(self._inputs(x))
            self._scratch = M.step_base(x, g, None, self._scratch)
        return _copy(self._inputs(x))

    # --------------------------------------------------------------------- TL

    def tl(self, dx0: dict, save_at=(), *, partial: bool = False) -> dict:
        """Map ``dx0`` (STEP_INPUTS of state 0) through the window, from
        the base the last :meth:`forward` ran (the start state if none).
        With ``partial`` ``dx0`` may hold any subset of STEP_INPUTS and
        the names it omits start at zero (WRFDA's start,
        :data:`WRFDA_TL_START`).  Returns ``{k: dx_k}`` (STEP_INPUTS) for
        every ``k`` in ``save_at`` and for ``nsteps``."""
        import cupy as cp
        if partial and not dx0:
            raise ValueError("dx0 is empty: partial=True still needs at least "
                             "one STEP_INPUTS name")
        self._check(dx0, "dx0", partial)
        save = set(int(k) for k in save_at) | {self.nsteps}
        bad = [k for k in save if not 0 <= k <= self.nsteps]
        if bad:
            raise ValueError(f"save_at states outside 0..{self.nsteps}: {sorted(bad)}")
        start = self._checkpoints.get(0) if self._checkpoints else None
        x = self._load(start if start is not None else self._inputs(self._start))
        dx = {n: cp.zeros_like(v) for n, v in x.items()}
        for n in M.STEP_INPUTS:
            if n in dx0:
                dx[n][...] = dx0[n]
        out = {}
        if 0 in save:
            out[0] = _copy(self._inputs(dx))
        for k, g in enumerate(self.geoms):
            M.step_tl(x, dx, g)
            if k + 1 in save:
                out[k + 1] = _copy(self._inputs(dx))
        return out

    # --------------------------------------------------------------------- AD

    def ad(self, ax_final: dict | None = None, forcing: dict | None = None) -> dict:
        """The adjoint of the window: ``ax_final`` (STEP_INPUTS of state
        ``nsteps``; zero if None) plus ``forcing`` ``{k: STEP_INPUTS}``
        added at state ``k`` (``0 <= k <= nsteps``), mapped back to state
        0.  Needs :meth:`forward` first.  Returns state 0's STEP_INPUTS."""
        import cupy as cp
        if self._checkpoints is None:
            raise RuntimeError("Window.ad needs Window.forward first (its checkpoints)")
        forcing = {int(k): v for k, v in (forcing or {}).items()}
        for k, v in forcing.items():
            self._check(v, f"forcing[{k}]")
        if ax_final is not None:
            self._check(ax_final, "ax_final")
        bad = [k for k in forcing if not 0 <= k <= self.nsteps]
        if bad:
            raise ValueError(f"forcing at states outside 0..{self.nsteps}: {sorted(bad)}")
        ax = {n: cp.zeros_like(v) for n, v in self._start.items()}

        def add(k, a):
            if a is not None:
                for n in M.STEP_INPUTS:
                    ax[n] += a[n]

        add(self.nsteps, ax_final)
        add(self.nsteps, forcing.get(self.nsteps))
        for c in sorted(self._checkpoints, reverse=True):
            seg = range(c, min(c + self.every, self.nsteps))
            # the segment's step-start states, from its checkpoint
            # (the first is the checkpoint itself, which _load only reads)
            starts = {c: self._checkpoints[c]}
            if len(seg) > 1:
                x = self._load(self._checkpoints[c])
                for k in seg[:-1]:
                    self._scratch = M.step_base(x, self.geoms[k], None, self._scratch)
                    starts[k + 1] = _copy(self._inputs(x))
            for k in reversed(seg):
                x = self._load(starts.pop(k))
                tape = []
                self._scratch = M.step_base(x, self.geoms[k], tape, self._scratch)
                M.step_ad(ax, x, tape, self.geoms[k])
                del tape
                add(k, forcing.get(k))
        return _copy(self._inputs(ax))
