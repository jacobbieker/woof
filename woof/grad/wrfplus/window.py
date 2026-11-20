"""A window of strict dry steps: the forward model, its tangent-linear and
its adjoint over N steps with checkpoints (P3a item 8, for 4D-Var).

States in and out are dicts over :data:`model.STEP_INPUTS` (the
prognostics and the previous step's diagnosis).  State ``k`` is the state
after ``k`` steps, ``0 <= k <= nsteps``.

- :meth:`Window.forward` runs the base and keeps a checkpoint every
  ``checkpoint_every`` steps.
- :meth:`Window.tl` maps a perturbation of state 0 through the window and
  returns the perturbation at the requested states.
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

    def _check(self, d: dict, what: str) -> None:
        """``d`` holds every STEP_INPUTS name, shaped and typed as the
        window's (no silent cast or broadcast)."""
        for n in M.STEP_INPUTS:
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

    def tl(self, dx0: dict, save_at=()) -> dict:
        """Map ``dx0`` (STEP_INPUTS of state 0) through the window, from
        the base the last :meth:`forward` ran (the start state if none).
        Returns ``{k: dx_k}`` (STEP_INPUTS) for every ``k`` in ``save_at``
        and for ``nsteps``."""
        import cupy as cp
        self._check(dx0, "dx0")
        save = set(int(k) for k in save_at) | {self.nsteps}
        bad = [k for k in save if not 0 <= k <= self.nsteps]
        if bad:
            raise ValueError(f"save_at states outside 0..{self.nsteps}: {sorted(bad)}")
        start = self._checkpoints.get(0) if self._checkpoints else None
        x = self._load(start if start is not None else self._inputs(self._start))
        dx = {n: cp.zeros_like(v) for n, v in x.items()}
        for n in M.STEP_INPUTS:
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
            starts = {}
            x = self._load(self._checkpoints[c])
            for k in seg:
                starts[k] = _copy(self._inputs(x))
                if k + 1 < seg.stop:
                    self._scratch = M.step_base(x, self.geoms[k], None, self._scratch)
            for k in reversed(seg):
                x = self._load(starts.pop(k))
                tape = []
                self._scratch = M.step_base(x, self.geoms[k], tape, self._scratch)
                M.step_ad(ax, x, tape, self.geoms[k])
                del tape
                add(k, forcing.get(k))
        return _copy(self._inputs(ax))
