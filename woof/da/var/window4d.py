"""P3a's dry WRF window as the 4D-Var tangent-linear model (EXPERIMENTAL;
docs/dev/WRFDA-4DVAR-PLAN.md, PR 4D.3).

:class:`WindowModel` is a :class:`~woof.da.var.fourd.TLModel` over a
:class:`woof.grad.wrfplus.window.Window`:

- the TL state :func:`~woof.da.var.transfer4d.xa_to_tl` writes (WRFDA's
  ``g_*``, Fortran ``(i, j, k)``) goes into the window's layout
  (``(k, j, i)``, the window's device and type) and starts the window as
  WRFDA's TL model starts (:func:`~woof.grad.wrfplus.window.wrfda_tl_start`:
  ``g_p`` into both pressure carriers, ``g_mu`` into ``muts`` too, ``al``
  and ``alt`` zero).  ``g_ph_2`` goes into ``php`` (zero unless
  ``ph="hydrostatic"``);
- slot ``n`` is the window's state ``n * steps_per_slot`` (``var4d_bin`` over
  the time step), read back into WRFDA's names; ``g_p`` is WRF's
  perturbation pressure, the window's ``p_pert``;
- the adjoint puts each slot's forcing on the same names, runs
  :meth:`Window.ad` and restricts with
  :func:`~woof.grad.wrfplus.window.wrfda_tl_start_ad`.

**Moisture.**  On a moist window (``Window.inputs`` holds qv, qc and qr:
P3a's moist binding, mp_physics 0) ``moisture="advect"`` (the default there)
starts the window's ``qv`` from ``g_qv`` and reads each slot's ``g_qv``
back, as WRFDA's TL model advects ``g_moist``; qc and qr start at zero
(``cloud_cv_options = 0``: WRFDA's transfer gives them no perturbation) and
nothing observes them.  On a dry window ``moisture="persist"`` (the default
there, and the only option) holds the ``qv`` perturbation constant over the
window (its adjoint sums the slots' forcings): not WRFDA's moist TL, meant
for the dry OSSE (L2).  Each option is refused on the other kind of
window.

**Two trajectories.**  The window is linearised about ITS OWN base (its
start state run forward by :meth:`Window.forward`), but each slot's ``H'``
and ``wrftltoxa`` are linearised about THAT SLOT'S first guess ``xb``
(:class:`~woof.da.var.fourd.FourDObsTerm`).  Nothing here makes the two
the same: if the window's base at slot ``n`` is not ``fgNN``, the 4D-Var
gradient mixes two trajectories and is not WRFDA's.  WRFDA makes them one
(WRFPLUS writes the trajectory the slots read).  Pass ``base=`` (the slots'
``(xb, grid)``) to compare the window's base with each slot's ``grid`` at
construction and refuse a mismatch over ``base_rtol``; the L1/L0 lanes
should turn it on.

The window must be binary64: a binary32 window's adjoint is only its
transpose to about 1e-6, and the minimiser needs 1e-12.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import transfer4d as t4

__all__ = ["WindowModel", "to_window", "from_window"]

#: WRFDA's TL names -> the window's names (the slot read-back; the start
#: goes through ``wrfda_tl_start``).
_NAMES = {"u_2": "u", "v_2": "v", "w_2": "w", "t_2": "thp", "mu_2": "mup", "ph_2": "php"}

#: ``moisture``: ``advect`` (a moist window carries g_qv) or ``persist`` (a dry
#: window: g_qv held over the window).
_MOISTURE = ("advect", "persist")

#: ``halo_start``: ``wrfplus`` (WRFPLUS's start_domain: the primal mu halo
#: reset to the edge row, its tangent seeded from g_mu) or ``carried`` (the
#: window's carried halo, the TL halo starting at zero; woof's own window).
_HALO_STARTS = ("wrfplus", "carried")


def _halo_name():
    from woof.grad.wrfplus import model as M

    return getattr(M, "HALO", None)


#: WRF's state names (``wrftoxb``'s ``grid``) -> the window's, for the base check.
_BASE = {**_NAMES, "p": "p_pert"}


def to_window(a: np.ndarray) -> np.ndarray:
    """Fortran ``(i, j[, k])`` to the window's ``([k,] j, i)``."""
    return np.ascontiguousarray(np.asarray(a).transpose())


def from_window(a: np.ndarray) -> np.ndarray:
    """The window's ``([k,] j, i)`` to Fortran ``(i, j[, k])``, binary64."""
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64).transpose())


def _step_inputs() -> tuple:
    from woof.grad.wrfplus import model as M

    return M.STEP_INPUTS


def _refuse_metal() -> None:
    """The window on Metal: refused, as Metal has no binary64 (the 4D-Var
    window is binary64 only; PERF-QUEUE holds the Metal port)."""
    from woof import backend

    if backend.active() == "mlx":
        raise NotImplementedError("4D-Var's window model on Metal: the window must be binary64 and Metal has no "
                                  "binary64; run it on CUDA or the host backend (WOOF_BACKEND=host)")


def _default_to_device(a: np.ndarray, dtype):
    import cupy as cp

    return cp.asarray(np.ascontiguousarray(a, dtype=dtype))


def _default_to_host(a) -> np.ndarray:
    return a.get() if hasattr(a, "get") else np.asarray(a)


class WindowModel:
    """WRFDA's TL and AD model over ``window`` (a forwarded
    :class:`~woof.grad.wrfplus.window.Window`), one slot every
    ``steps_per_slot`` steps.

    ``to_device``/``to_host`` move arrays between the host and the window
    (cupy, or woof's facade, by default); tests pass their own.

    ``base`` (optional): the slots' ``(xb, grid)`` pairs, as
    :class:`~woof.da.var.fourd.FourDObsTerm` takes them; the window's base
    at each slot is compared with that slot's ``grid`` (see the module
    docstring) and a relative difference over ``base_rtol`` is refused.
    ``base_end``: the end state for a window on an external trajectory
    (:meth:`check_base`).

    The adjoint forces every state input the window has
    (``Window.inputs`` where it has one: a specified domain adds the
    carried mu halo, forced with zero, as nothing observes it).

    ``halo_start`` (specified windows only): ``"wrfplus"`` (default) starts
    as WRFPLUS's ``start_domain`` does, the primal mu halo reset to the edge
    row and the TL halo seeded from ``g_mu``
    (``wrfda_tl_start(..., specified=True)``); ``"carried"`` keeps the
    window's carried halo and starts the TL halo at zero (woof's own
    window, not WRFPLUS's: a warm window then differs from WRFPLUS's base).  The
    boundary tables are not perturbed (WRFDA's ``var4d_lbc = .false.``)
    and their adjoint is dropped.

    :meth:`relinearise` moves the base to a new nonlinear run (an outer
    loop's :class:`~woof.da.var.outer4d.Trajectory`, through
    ``Window.forward(trajectory=...)``).  Binary64 only: refused on Metal
    when ``to_device`` is the default.
    """

    def __init__(self, window, steps_per_slot: int, *, moisture: str | None = None, to_device=None, to_host=None,
                 base: Sequence | None = None, base_rtol: float = 1e-5, base_end: dict | None = None,
                 halo_start: str = "wrfplus"):
        moist = "qv" in tuple(getattr(window, "inputs", None) or ())
        if moisture is None:
            moisture = "advect" if moist else "persist"
        if moisture not in _MOISTURE:
            raise ValueError(f"moisture must be one of {_MOISTURE}, not {moisture!r}")
        if moisture == "advect" and not moist:
            raise NotImplementedError("moisture='advect': the window is dry (no qv among its inputs); "
                                      "only 'persist' holds on a dry window")
        if moisture == "persist" and moist:
            raise ValueError("moisture='persist' on a moist window: its qv tangent moves with the flow, "
                             "so the slots' g_qv are the window's ('advect')")
        self.moisture = moisture
        if to_device is None:
            _refuse_metal()
        if steps_per_slot < 1:
            raise ValueError(f"steps_per_slot must be >= 1, not {steps_per_slot}")
        self.window = window
        self.sps = int(steps_per_slot)
        self.dtype = np.dtype(str(window._start["u"].dtype))
        if self.dtype != np.float64:
            raise TypeError(f"the window is {self.dtype}; 4D-Var needs a binary64 window "
                            "(a binary32 window's adjoint is its transpose to only ~1e-6)")
        self._dev = to_device or (lambda a: _default_to_device(a, self.dtype))
        self._host = to_host or _default_to_host
        if halo_start not in _HALO_STARTS:
            raise ValueError(f"halo_start must be one of {_HALO_STARTS}, not {halo_start!r}")
        self.halo = _halo_name() if _halo_name() in self._inputs() else None
        self.halo_start = halo_start
        if self.halo is not None and halo_start == "wrfplus":
            self._halo_to_edge()
        if getattr(window, "_checkpoints", None) is None:
            window.forward()
        self.base_diff = None
        self._end = base_end
        if base is not None:
            self.base_diff = self.check_base(base, rtol=base_rtol, end=base_end)

    def relinearise(self, trajectory) -> dict:
        """An outer loop's new nonlinear run (:class:`~woof.da.var.outer4d.Trajectory`):
        the window linearises about its per-step ``states``
        (``Window.forward(trajectory=...)``, as WRFPLUS reads the rerun's
        states).  Returns the run's end state, which :meth:`check_base`
        then uses by default.  A trajectory without ``states`` is refused:
        the window would stay on the previous loop's base."""
        states = getattr(trajectory, "states", None)
        if states is None:
            raise ValueError("the window model needs the rerun's per-step states (Trajectory.states) to "
                             "relinearise; without them the TL and AD stay on the previous loop's trajectory")
        self._end = self.window.forward(trajectory=states)
        if self.halo is not None and self.halo_start == "wrfplus":
            self._halo_to_edge()   # an external trajectory: checked, not changed
        self.base_diff = None
        return self._end

    def check_base(self, slots: Sequence, *, rtol: float = 1e-5, end: dict | None = None) -> dict:
        """The window's base at each slot against that slot's ``grid``:
        ``{(n, name): max|base - grid| / max|grid|}``; refuses any over
        ``rtol``.

        A window running its own base: reruns :meth:`Window.forward` (the
        base the TL and AD then use is unchanged) and needs a checkpoint at
        every slot state.  A window on an external trajectory
        (``Window.forward(trajectory=...)``, which holds every state but the
        last): reads the trajectory, never reruns ``forward`` (that would
        replace it), and compares the last slot with ``end``, the state
        ``forward(trajectory=...)`` returned."""
        states = self._slots(len(slots))
        nsteps = self.window.nsteps
        if getattr(self.window, "_external", False):
            end = end if end is not None else self._end
            if end is None:
                raise ValueError("the window runs on an external trajectory: pass end= (the state "
                                 "forward(trajectory=...) returned) for the last slot")
            final = end
        else:
            every = int(getattr(self.window, "every", 1))
            if self.sps % every:
                raise ValueError(f"the base check needs a checkpoint at every slot: checkpoint_every={every} "
                                 f"does not divide steps_per_slot={self.sps}")
            final = self.window.forward()
        out, bad = {}, []
        for n, (k, (_, grid)) in enumerate(zip(states, slots)):
            st = final if k == nsteps else self.window._checkpoints[k]
            for g, w in _BASE.items():
                ref = np.asarray(grid[g], dtype=np.float64)
                got = from_window(self._host(st[w]))
                if got.shape != ref.shape:
                    raise ValueError(f"slot {n}: the window's {w} is {got.shape[::-1]} (k, j, i), "
                                     f"the slot's {g} {ref.shape} (i, j, k)")
                scale = float(np.max(np.abs(ref))) or 1.0
                r = float(np.max(np.abs(got - ref))) / scale
                out[(n, g)] = r
                if not r <= rtol:
                    bad.append(f"slot {n} {g}: {r:.3g}")
        if bad:
            raise ValueError("the window's base is not the slots' first guesses (two trajectories): "
                             + "; ".join(bad) + f" > rtol {rtol:g}")
        return out

    def _halo_to_edge(self) -> None:
        """WRFPLUS's ``start_domain`` on a specified window: the primal mu
        halo starts as the edge row, which in woof's form (model.HALO, a copy
        of ``mup`` whose ring rk_step 1 reads) is ``mup`` itself.  Resets the
        window's start and reruns its base if it already ran one; a window
        on an external trajectory is checked, not changed (its state 0 is
        the caller's)."""
        w, h = self.window, self.halo
        if getattr(w, "_external", False):
            st = w._checkpoints[0]
            if not bool(np.array_equal(self._host(st[h]), self._host(st["mup"]))):
                raise ValueError("halo_start='wrfplus': the external trajectory's state 0 carries a mu halo "
                                 "that is not mup (WRFPLUS's start resets it to the edge row); reset it, or "
                                 "pass halo_start='carried'")
            return
        w._start[h][...] = w._start["mup"]
        if getattr(w, "_checkpoints", None) is not None:
            w.forward()

    def _tl_start(self, d: dict) -> dict:
        from woof.grad.wrfplus.window import wrfda_tl_start

        if self.halo is not None and self.halo_start == "wrfplus":
            return wrfda_tl_start(d, specified=True)
        return wrfda_tl_start(d)

    def _tl_start_ad(self, a: dict) -> dict:
        from woof.grad.wrfplus.window import wrfda_tl_start_ad

        if self.halo is not None and self.halo_start == "wrfplus":
            return wrfda_tl_start_ad(a, specified=True)
        return wrfda_tl_start_ad(a)

    def _inputs(self) -> tuple:
        """The window's state inputs: ``Window.inputs`` (STEP_INPUTS plus the
        mu halo on a specified domain) where the window has it."""
        return tuple(getattr(self.window, "inputs", None) or _step_inputs())

    def _slots(self, nslots: int) -> list:
        if nslots < 2:
            raise ValueError(f"{nslots} slot(s): 4D-Var over a window needs at least 2 slots "
                             "(the start and the end); one slot is 3D-Var")
        if self.window.nsteps != self.sps * (nslots - 1):
            raise ValueError(f"{nslots} slots {self.sps} steps apart need a window of "
                             f"{self.sps * (nslots - 1)} steps; it has {self.window.nsteps}")
        return [n * self.sps for n in range(nslots)]

    def tl(self, g0: dict, nslots: int) -> list:
        missing = [k for k in t4.TL_KEYS if k not in g0]
        if missing:
            raise ValueError(f"the TL state lacks {missing}")
        states = self._slots(nslots)
        dev = {"u": g0["u_2"], "v": g0["v_2"], "w": g0["w_2"], "thp": g0["t_2"], "mup": g0["mu_2"],
               "p": g0["p"]}
        dx0 = self._tl_start({k: self._dev(to_window(v)) for k, v in dev.items()})
        dx0["php"] = self._dev(to_window(g0["ph_2"]))
        if self.moisture == "advect":
            dx0["qv"] = self._dev(to_window(g0["qv"]))
        out = self.window.tl(dx0, save_at=states, partial=True)
        res = []
        for k in states:
            d = out[k]
            g = {tl: from_window(self._host(d[w])) for tl, w in _NAMES.items()}
            g["p"] = from_window(self._host(d["p_pert"]))
            g["qv"] = (from_window(self._host(d["qv"])) if self.moisture == "advect"
                       else np.array(g0["qv"], dtype=np.float64))
            res.append(g)
        return res

    def ad(self, forcings: Sequence) -> dict:
        forcings = list(forcings)
        states = self._slots(len(forcings))
        start = self.window._start
        forcing = {}
        qv = None
        for k, f in zip(states, forcings):
            dk = {}
            for n in self._inputs():
                dk[n] = self._dev(np.zeros(tuple(start[n].shape)))
            for tl, w in _NAMES.items():
                if tl in f:
                    dk[w] = self._dev(to_window(f[tl]))
            if "p" in f:
                dk["p_pert"] = self._dev(to_window(f["p"]))
            if self.moisture == "advect" and "qv" in f:
                dk["qv"] = self._dev(to_window(f["qv"]))
            forcing[k] = dk
            if self.moisture == "persist" and "qv" in f:
                qv = np.array(f["qv"], dtype=np.float64) if qv is None else qv + f["qv"]
        a = self.window.ad(None, forcing)
        a0 = self._tl_start_ad(a)
        # php is not one of wrfda_tl_start's fields (WRFDA starts g_ph at
        # zero; ph="hydrostatic" sets it here, outside that map), so a0 has
        # no php: its adjoint is the window's own, a["php"]
        out = {"u_2": a0["u"], "v_2": a0["v"], "w_2": a0["w"], "t_2": a0["thp"], "mu_2": a0["mup"],
               "p": a0["p"], "ph_2": a["php"]}
        if self.moisture == "advect":
            out["qv"] = a["qv"]   # not one of wrfda_tl_start's fields either
        out = {k: from_window(self._host(v)) for k, v in out.items()}
        if qv is not None:
            out["qv"] = qv
        return out

