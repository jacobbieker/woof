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

The window is dry: it carries no moisture.  The ``qv`` perturbation is held
constant over the window (``moisture="persist"``, the only option; its
adjoint sums the slots' forcings).  WRFDA's TL model advects ``g_qv``, so
this is not the L1 lane for moisture; it is meant for the dry OSSE (L2)
until the window carries moisture.

The window is linearised about its own base (its start state and
:meth:`Window.forward`); the slots' ``H'`` about the slots' first guesses
(:class:`~woof.da.var.fourd.FourDObsTerm`).  Making the two the same
trajectory is the caller's (the L2 OSSE's) business.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import transfer4d as t4

__all__ = ["WindowModel", "to_window", "from_window"]

#: WRFDA's TL names -> the window's names (the slot read-back; the start
#: goes through ``wrfda_tl_start``).
_NAMES = {"u_2": "u", "v_2": "v", "w_2": "w", "t_2": "thp", "mu_2": "mup", "ph_2": "php"}


def to_window(a: np.ndarray) -> np.ndarray:
    """Fortran ``(i, j[, k])`` to the window's ``([k,] j, i)``."""
    return np.ascontiguousarray(np.asarray(a).transpose())


def from_window(a: np.ndarray) -> np.ndarray:
    """The window's ``([k,] j, i)`` to Fortran ``(i, j[, k])``, binary64."""
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64).transpose())


def _step_inputs() -> tuple:
    from woof.grad.wrfplus import model as M

    return M.STEP_INPUTS


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
    """

    def __init__(self, window, steps_per_slot: int, *, moisture: str = "persist", to_device=None, to_host=None):
        if moisture != "persist":
            raise NotImplementedError(f"moisture={moisture!r}: the window is dry; only 'persist' is ported")
        if steps_per_slot < 1:
            raise ValueError(f"steps_per_slot must be >= 1, not {steps_per_slot}")
        self.window = window
        self.sps = int(steps_per_slot)
        self.dtype = window._start["u"].dtype
        self._dev = to_device or (lambda a: _default_to_device(a, self.dtype))
        self._host = to_host or _default_to_host
        if getattr(window, "_checkpoints", None) is None:
            window.forward()

    def _slots(self, nslots: int) -> list:
        if nslots < 1:
            raise ValueError("no slots")
        if self.window.nsteps != self.sps * (nslots - 1):
            raise ValueError(f"{nslots} slots {self.sps} steps apart need a window of "
                             f"{self.sps * (nslots - 1)} steps; it has {self.window.nsteps}")
        return [n * self.sps for n in range(nslots)]

    def tl(self, g0: dict, nslots: int) -> list:
        from woof.grad.wrfplus.window import wrfda_tl_start

        missing = [k for k in t4.TL_KEYS if k not in g0]
        if missing:
            raise ValueError(f"the TL state lacks {missing}")
        states = self._slots(nslots)
        dev = {"u": g0["u_2"], "v": g0["v_2"], "w": g0["w_2"], "thp": g0["t_2"], "mup": g0["mu_2"],
               "p": g0["p"]}
        dx0 = wrfda_tl_start({k: self._dev(to_window(v)) for k, v in dev.items()})
        dx0["php"] = self._dev(to_window(g0["ph_2"]))
        out = self.window.tl(dx0, save_at=states, partial=True)
        res = []
        for k in states:
            d = out[k]
            g = {tl: from_window(self._host(d[w])) for tl, w in _NAMES.items()}
            g["p"] = from_window(self._host(d["p_pert"]))
            g["qv"] = np.array(g0["qv"], dtype=np.float64)
            res.append(g)
        return res

    def ad(self, forcings: Sequence) -> dict:
        from woof.grad.wrfplus.window import wrfda_tl_start_ad

        forcings = list(forcings)
        states = self._slots(len(forcings))
        start = self.window._start
        forcing = {}
        qv = None
        for k, f in zip(states, forcings):
            dk = {}
            for n in _step_inputs():
                dk[n] = self._dev(np.zeros(tuple(start[n].shape)))
            for tl, w in _NAMES.items():
                if tl in f:
                    dk[w] = self._dev(to_window(f[tl]))
            if "p" in f:
                dk["p_pert"] = self._dev(to_window(f["p"]))
            forcing[k] = dk
            if "qv" in f:
                qv = np.array(f["qv"], dtype=np.float64) if qv is None else qv + f["qv"]
        a = self.window.ad(None, forcing)
        a0 = wrfda_tl_start_ad(a)
        out = {"u_2": a0["u"], "v_2": a0["v"], "w_2": a0["w"], "t_2": a0["thp"], "mu_2": a0["mup"],
               "p": a0["p"], "ph_2": a["php"]}
        out = {k: from_window(self._host(v)) for k, v in out.items()}
        if qv is not None:
            out["qv"] = qv
        return out

