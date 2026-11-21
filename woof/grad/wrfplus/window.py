"""A window of strict dry steps: the forward model, its tangent-linear and
its adjoint over N steps with checkpoints (P3a item 8, for 4D-Var).

States in and out are dicts over :data:`model.STEP_INPUTS` (the
prognostics and the previous step's diagnosis; :func:`model.step_inputs`
adds the carried mu halo on a specified domain).  State ``k`` is the state
after ``k`` steps, ``0 <= k <= nsteps``.  A specified domain's boundary
tables (:data:`model.LBC_INPUTS`, ``Window.params``) are inputs no step
writes: :meth:`Window.tl` takes their perturbation in ``dx0`` (zero where
absent) and :meth:`Window.ad` returns their adjoint with state 0's.

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


def _wrfda_copies(specified: bool) -> dict:
    # On a specified domain WRFPLUS's TL start also fills the mu halo
    # from g_mu: wrf_run_tl calls start_domain, whose g_start_domain_em
    # (wrftladj/start_em_tl.F:706) runs g_set_physical_bc2d on mu_2 with
    # 't', and for a specified domain that copies g_mu_2's edge row into
    # the halo rows (wrftladj/module_bc_tl.F, open_xs and the other
    # sides).  woof's halo (model.HALO) is a copy of mup whose ring is
    # what rk_step 1 reads, so its tangent is g_mu.
    return {**_WRFDA_COPIES, M.HALO: "mup"} if specified else _WRFDA_COPIES


def wrfda_tl_start(g: dict, *, specified: bool = False) -> dict:
    """WRFDA's TL start for ``Window.tl(..., partial=True)`` from
    ``g`` over :data:`WRFDA_TL_FIELDS` (no other name): g_p into both
    pressure carriers and g_mu into ``muts`` too, and with ``specified``
    (a window on specified boundaries) into the mu halo as well (the
    arrays are shared, not copied; ``Window.tl`` only reads them)."""
    if set(g) != set(WRFDA_TL_FIELDS):
        raise ValueError(f"wrfda_tl_start takes {list(WRFDA_TL_FIELDS)}, got {sorted(g)}")
    return {**g, **{t: g[src] for t, src in _wrfda_copies(specified).items()}}


def wrfda_tl_start_ad(adj: dict, *, specified: bool = False) -> dict:
    """The adjoint of :func:`wrfda_tl_start` (the same ``specified``):
    :data:`WRFDA_TL_FIELDS` of ``adj`` (a ``Window.ad`` result), with each
    copy's adjoint summed back into its source (``p + p_pert``, ``mup +
    muts`` and, with ``specified``, ``+ adj[HALO]``); new arrays."""
    out = {n: adj[n].copy() for n in WRFDA_TL_FIELDS}
    for t, src in _wrfda_copies(specified).items():
        out[src] += adj[t]
    return out


def lbc_from_ends(start: dict, end: dict, seconds: float) -> dict:
    """WRFDA's boundary perturbation under ``var4d_lbc``: the window's
    tables (``Window.params``) from the perturbation of each value table
    at the interval's start (``start``) and end (``end``), both dicts over
    the value-table names (``lbc_<field>_<side>``, in the tables' stored
    form): the value is ``start`` and the tendency ``(end - start) /
    seconds``, as WRF's ``bt`` ties to the next record.  ``seconds`` is
    the interval's length (``Window.lbc_seconds``).  Without ``var4d_lbc``
    WRFDA perturbs the start only and keeps the tendency: pass the value
    tables alone in ``dx0`` (the ``_t`` tables then start at zero)."""
    if set(start) != set(end):
        raise ValueError(f"start and end name other tables: {sorted(set(start) ^ set(end))}")
    out = {}
    for n, a in start.items():
        out[n] = a.copy()
        out[n + "_t"] = (end[n] - a) / seconds
    return out


def lbc_from_ends_ad(adj: dict, names, seconds: float) -> tuple:
    """The adjoint of :func:`lbc_from_ends` over the value tables
    ``names``: ``adj`` (a ``Window.ad`` result) folded into the start's
    and the end's adjoints, ``(start_adj, end_adj)``; new arrays."""
    start, end = {}, {}
    for n in names:
        tend = adj[n + "_t"] / seconds
        start[n] = adj[n] - tend
        end[n] = tend.copy()
    return start, end


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
        self.inputs = M.step_inputs(self.geoms[0])
        self.params = M.step_parameters(self.geoms[0])
        #: the boundary interval's length (lbc_from_ends), specified only
        self.lbc_seconds = None
        if self.params:
            iv = state.lateral_boundaries.interval_at(state.elapsed_seconds)
            self.lbc_seconds = float(iv.end_seconds - iv.start_seconds)
        self._start = M.run_state(state, cfg, dtype)
        self._scratch = None
        self._mutable = self._written_names()
        missing = set(self.inputs) - set(self._mutable)
        if missing:   # the work dict would write into the shared start arrays
            raise AssertionError(f"a step does not write {sorted(missing)}")
        # the one dict every step runs in: the window's constant arrays
        # (shared) and its own copies of what a step writes
        self._work = {**self._start, **_copy(self._start, self._mutable)}
        self._checkpoints: dict[int, dict] | None = None
        self._external = False   # the base comes from forward(trajectory=)

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

    def _check(self, d: dict, what: str, partial: bool = False,
               tables: bool = False) -> None:
        """``d`` holds every state input (any subset if ``partial``) and,
        if ``tables``, any of the boundary tables (``params``), and no
        other name, shaped and typed as the window's (no silent cast or
        broadcast)."""
        allowed = self.inputs + (self.params if tables else ())
        unknown = sorted(set(d) - set(allowed))
        if unknown:
            raise ValueError(f"{what} names {unknown}, not the window's inputs "
                             f"{list(allowed)}")
        missing = [n for n in self.inputs if n not in d]
        if missing and not partial:
            raise ValueError(f"{what} lacks {missing}; pass partial=True to start "
                             "them at zero")
        for n in allowed:
            if n not in d:
                continue
            a, ref = d[n], self._start[n]
            if a.shape != ref.shape or a.dtype != ref.dtype:
                raise ValueError(f"{what}[{n!r}] is {a.dtype}{tuple(a.shape)}, "
                                 f"the window's is {ref.dtype}{tuple(ref.shape)}")

    def _load(self, ck: dict) -> dict:
        """The work dict holding state ``ck`` (STEP_INPUTS)."""
        for n in self.inputs:
            self._work[n][...] = ck[n]
        return self._work

    def _inputs(self, d: dict) -> dict:
        return {n: d[n] for n in self.inputs}

    # ---------------------------------------------------------------- forward

    def forward(self, x0: dict | None = None, *, trajectory=None) -> dict:
        """Run the base from state 0 (the window's start state, or
        ``x0``'s STEP_INPUTS) and keep the checkpoints the adjoint uses.
        Returns the final state's STEP_INPUTS.

        With ``trajectory`` (a sequence of ``nsteps`` or ``nsteps + 1``
        STEP_INPUTS dicts, state ``k`` at index ``k``; see
        :meth:`trajectory_state`) the window runs no base of its own:
        :meth:`tl` and :meth:`ad` linearise step ``k`` about
        ``trajectory[k]``, as WRFPLUS linearises about the nonlinear run's
        states (a forecast with physics, or WRF's own).  Each step's
        geometry is still the window's (``cfg`` at construction), and
        each state must hold what that step reads: the previous step's
        diagnosis and carried muts where the step reuses them.  Returns
        state ``nsteps`` (given, or one window step from the last)."""
        if trajectory is not None:
            if x0 is not None:
                raise ValueError("forward takes x0 or trajectory, not both")
            return self._use_trajectory(trajectory)
        if x0 is not None:
            # no tables: the base runs on the tables the window was built with
            self._check(x0, "x0")
        self._external = False
        x = self._load(self._inputs(self._start) if x0 is None else x0)
        self._checkpoints = {}
        for k, g in enumerate(self.geoms):
            if k % self.every == 0:
                self._checkpoints[k] = _copy(self._inputs(x))
            self._scratch = M.step_base(x, g, None, self._scratch)
        return _copy(self._inputs(x))

    def _use_trajectory(self, trajectory) -> dict:
        states = list(trajectory)
        if len(states) not in (self.nsteps, self.nsteps + 1):
            raise ValueError(f"trajectory holds {len(states)} states; the window "
                             f"needs {self.nsteps} (states 0..{self.nsteps - 1}) "
                             f"or {self.nsteps + 1}")
        for k, st in enumerate(states):
            self._check(st, f"trajectory[{k}]")
        # copies: a step reads its state through the work dict only
        checkpoints = {k: _copy(self._inputs(st)) for k, st in enumerate(states)}
        if len(states) > self.nsteps:
            end = checkpoints.pop(self.nsteps)   # already a copy
        else:
            x = self._load(checkpoints[self.nsteps - 1])
            self._scratch = M.step_base(x, self.geoms[-1], None, self._scratch)
            end = _copy(self._inputs(x))
        # switch the base only once the end state exists: a failed step
        # leaves the window on its previous base
        self._checkpoints = checkpoints
        self._external = True
        return end

    def trajectory_state(self, state, cfg) -> dict:
        """A trajectory entry from a DomainState (a woof forecast's state
        after some steps), in the window's dtype: its STEP_INPUTS as
        :func:`model.run_state` reads them."""
        x = M.run_state(state, cfg, self._start["u"].dtype)
        return _copy(self._inputs(x))

    # --------------------------------------------------------------------- TL

    def tl(self, dx0: dict, save_at=(), *, partial: bool = False) -> dict:
        """Map ``dx0`` (STEP_INPUTS of state 0) through the window, from
        the base the last :meth:`forward` ran (the start state if none).
        ``dx0`` may also hold boundary tables (``params``; zero where
        absent).  With ``partial`` ``dx0`` may hold any subset of the state
        inputs and the names it omits start at zero (WRFDA's start,
        :data:`WRFDA_TL_START`).  Returns ``{k: dx_k}`` (state inputs) for
        every ``k`` in ``save_at`` and for ``nsteps``."""
        import cupy as cp
        if partial and not dx0:
            raise ValueError("dx0 is empty: partial=True still needs at least "
                             "one STEP_INPUTS name")
        self._check(dx0, "dx0", partial, tables=True)
        save = set(int(k) for k in save_at) | {self.nsteps}
        bad = [k for k in save if not 0 <= k <= self.nsteps]
        if bad:
            raise ValueError(f"save_at states outside 0..{self.nsteps}: {sorted(bad)}")
        start = self._checkpoints.get(0) if self._checkpoints else None
        x = self._load(start if start is not None else self._inputs(self._start))
        dx = {n: cp.zeros_like(v) for n, v in x.items()}
        for n in self.inputs + self.params:
            if n in dx0:
                dx[n][...] = dx0[n]
        out = {}
        if 0 in save:
            out[0] = _copy(self._inputs(dx))
        for k, g in enumerate(self.geoms):
            if self._external and k:   # the supplied base, not this one's
                x = self._load(self._checkpoints[k])
            M.step_tl(x, dx, g)
            if k + 1 in save:
                out[k + 1] = _copy(self._inputs(dx))
        return out

    # --------------------------------------------------------------------- AD

    def ad(self, ax_final: dict | None = None, forcing: dict | None = None) -> dict:
        """The adjoint of the window: ``ax_final`` (STEP_INPUTS of state
        ``nsteps``; zero if None) plus ``forcing`` ``{k: STEP_INPUTS}``
        added at state ``k`` (``0 <= k <= nsteps``), mapped back to state
        0.  Needs :meth:`forward` first.  Returns state 0's inputs and the
        boundary tables' adjoint (``params``)."""
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
                for n in self.inputs:
                    ax[n] += a[n]

        add(self.nsteps, ax_final)
        add(self.nsteps, forcing.get(self.nsteps))
        every = 1 if self._external else self.every   # every state is given
        for c in sorted(self._checkpoints, reverse=True):
            seg = range(c, min(c + every, self.nsteps))
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
        return _copy(ax, self.inputs + self.params)
