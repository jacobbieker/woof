"""WRFDA 4D-Var's observation term: ``H'`` at every time slot through a
tangent-linear model (EXPERIMENTAL; docs/dev/WRFDA-4DVAR-PLAN.md, PR 4D.2).

WRFDA 4.7.1 with ``var4d = .true.`` (``da_transform_vtoy``'s ``var4d``
branch) maps the analysis increment at the window's start to every slot::

    xa_0 --da_transfer_xatowrftl--> g_0 --TL model--> g_1 .. g_N
    for each slot n, last to first:
        da_read_basicstates (slot n's first guess)
        xa_n = da_transfer_wrftltoxa(g_n)   (about slot n's state)
        da_transform_xtoxa: xa_n%rho from xa_n%p, xa_n%t (p is the model's)
        y_n  = H'_n xa_n                     (slot n's observations, about slot n's xb)

and the adjoint (``da_transform_vtoy_adj``) the reverse: each slot's
``H'_n^T``, ``xtoxa_adj``'s ``rho`` part and ``da_transfer_wrftltoxa_adj``
give that slot's adjoint forcing; the adjoint model runs them back to the
start, and ``da_transfer_xatowrftl_adj`` gives ``xa_0``'s gradient.

:class:`FourDObsTerm` is that composition as a
:class:`~woof.da.var.cost.ObsTerm`; the model is pluggable
(:class:`TLModel`).  :class:`IdentityModel` is WRFPLUS's
``enable_identity``: every slot's perturbation is the initial one, the
adjoint sums the forcings.  P3a's ``Window`` (PR 4D.3) is the next model.

The observations and innovations are FGAT's (:mod:`woof.da.var.fgat`):
one file and one first guess per slot, ``plocal`` slot boundaries,
each slot's innovations against its own first guess.  ``y`` is the slots'
flat vectors one after the other; :meth:`FourDObsTerm.jo_split` gives
``da_jo_and_grady``'s groups over all slots (each type's reports in
``iv`` order, which is slot order).

Derived from WRFDA 4.7.1 (``var/da/da_minimisation/da_transform_vtoy.inc``,
``da_transform_vtoy_adj.inc``, ``var/da/da_vtox_transforms/da_transform_xtoxa.inc``,
``da_transform_xtoxa_adj.inc``, ``var/da/da_physics/da_pt_to_rho_lin.inc``,
``da_pt_to_rho_adj.inc``), public domain; see
``licenses/LICENSE-WRF-public-domain.txt``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from . import transfer4d as t4
from .cost import XA_KEYS

__all__ = ["FourDObsTerm", "IdentityModel", "TLModel", "Var4D", "pt_to_rho_lin", "pt_to_rho_adj"]


class TLModel(Protocol):
    """A tangent-linear model over the window's slots and its adjoint.

    ``tl(g0, nslots) -> [g_1, ..., g_N]``
        the perturbation at each slot time (slot 1 is the window's start,
        so ``g_1`` is ``g0``'s state at time zero), keys
        :data:`~woof.da.var.transfer4d.TL_KEYS`.
    ``ad(forcings) -> a_g0``
        the adjoint: ``forcings[n]`` is added at slot ``n``'s time while
        the adjoint runs from the last slot back to the start.
    """

    def tl(self, g0: dict, nslots: int) -> list: ...

    def ad(self, forcings: Sequence) -> dict: ...


class IdentityModel:
    """WRFPLUS's ``enable_identity``: ``M = I`` between every pair of slots."""

    def tl(self, g0: dict, nslots: int) -> list:
        return [{k: np.array(v, dtype=np.float64) for k, v in g0.items()} for _ in range(nslots)]

    def ad(self, forcings: Sequence) -> dict:
        out: dict = {}
        for f in reversed(list(forcings)):  # the adjoint runs from the last slot to the first
            for k, v in f.items():
                out[k] = np.array(v, dtype=np.float64) if k not in out else out[k] + v
        return out


@dataclass
class Var4D:
    """The 4D-Var options :func:`~woof.da.var.analysis.analyse` takes (``var4d=``).

    ``model`` is a :class:`TLModel` (the identity by default); ``press`` and
    ``ph`` are :func:`~woof.da.var.transfer4d.xa_to_tl`'s word-exact
    defaults or their opt-in corrections.
    """

    model: TLModel = field(default_factory=IdentityModel)
    press: str = "wrfda"
    ph: str = "zero"


def pt_to_rho_lin(xa: dict, xb: dict) -> np.ndarray:
    """``da_pt_to_rho_lin``: ``rho' = rho_b (p'/p_b - t'/t_b)``."""
    return xb["rho"] * (xa["p"] / xb["p"] - xa["t"] / xb["t"])


def pt_to_rho_adj(a: dict, xb: dict) -> dict:
    """``da_pt_to_rho_adj``: ``rho``'s gradient into ``p`` and ``t`` (``rho`` consumed)."""
    out = dict(a)
    temp = a["rho"] * xb["rho"]
    out["p"] = a["p"] + temp / xb["p"]
    out["t"] = a["t"] - temp / xb["t"]
    out["rho"] = np.zeros_like(temp)
    return out


def _host64(v):
    v = v.get() if hasattr(v, "get") else v
    return np.asarray(v, dtype=np.float64)


class FourDObsTerm:
    """``H' M`` over the window's slots as a :class:`~woof.da.var.cost.ObsTerm`.

    ``iv`` is FGAT's innovation vector (``plocal`` set, one slot per first
    guess); ``slots`` the ``(xb, grid)`` pairs of
    :func:`~woof.da.var.analysis.wrftoxb` for ``fg01`` .. ``fgNN``;
    ``start`` the pair the increment is transferred about (the analysis
    first guess, at the window's start).  ``x`` is ``u v t q psfc`` at the
    start (``p`` and ``rho`` are not read: WRFDA's transfer recomputes
    ``p`` and the TL model carries it); :meth:`ad` returns zero for them.
    """

    def __init__(self, iv, slots: Sequence, start: tuple, *, model: TLModel | None = None, press: str = "wrfda",
                 ph: str = "zero", libm: str = "correctly-rounded"):
        from . import fgat as fm
        from .obsop_wrfda import FlatObsTerm

        if iv.plocal is None:
            raise ValueError("FourDObsTerm needs FGAT's slotted innovations (plocal is None)")
        self.nslots = len(next(iter(iv.plocal.values()))) - 1
        if len(slots) != self.nslots:
            raise ValueError(f"{self.nslots} observation slots but {len(slots)} first guesses")
        self.slots = list(slots)
        self.start = start
        self.model = model if model is not None else IdentityModel()
        self.press, self.ph = press, ph
        xb0, grid0 = start
        t4.check_supported(xb0, grid0, press=press, ph=ph)
        for xb, grid in self.slots:
            t4.check_supported(xb, grid)
            if xb["t"].shape != xb0["t"].shape:
                raise ValueError(f"a slot is on {xb['t'].shape}, the start on {xb0['t'].shape}")
        kte = xb0["t"].shape[2]
        self.terms = [FlatObsTerm(fm.concat([fm.slot(iv, n)], kte=kte), self.slots[n][0], libm=libm)
                      for n in range(self.nslots)]
        self.offsets = np.concatenate([[0], np.cumsum([t.size for t in self.terms])]).astype(np.int64)
        self.size = int(self.offsets[-1])
        cat = (lambda name, dt: np.concatenate([getattr(t, name) for t in self.terms]).astype(dt)
               if self.terms else np.zeros(0, dt))
        self.d = cat("d", np.float64)
        self.err = cat("err", np.float64)
        self.mask = cat("mask", bool)
        self._groups = self._join_groups()

    def _join_groups(self):
        from . import innov_wrfda as iw

        per = {}
        for n, t in enumerate(self.terms):
            for kind, variables in t.jo_split():
                d = per.setdefault(kind, {})
                for var, idx in variables:
                    d.setdefault(var, []).append(self.offsets[n] + np.asarray(idx, dtype=np.int64))
        out = []
        for kind in iw.JO_ORDER:
            if kind not in per:
                continue
            out.append((kind, [(var, np.concatenate(per[kind][var])) for var in iw.JO_FIELDS[kind]
                               if var in per[kind]]))
        return out

    def jo_split(self):
        return self._groups

    def tl(self, x: dict):
        xb0, grid0 = self.start
        x0 = {k: _host64(x[k]) for k in ("u", "v", "t", "q", "psfc")}
        g0 = t4.xa_to_tl(x0, xb0, grid0, press=self.press, ph=self.ph)
        gs = self.model.tl(g0, self.nslots)
        out = np.zeros(self.size, dtype=np.float64)
        for n in range(self.nslots - 1, -1, -1):  # WRFDA's slot loop, last to first
            xb, grid = self.slots[n]
            xa = t4.tl_to_xa(gs[n], xb, grid)
            xa["rho"] = pt_to_rho_lin(xa, xb)
            out[self.offsets[n]:self.offsets[n + 1]] = self.terms[n].tl(xa)
        return out

    def ad(self, y) -> dict:
        y = _host64(y)
        forcings = [None] * self.nslots
        for n in range(self.nslots):  # da_transform_vtoy_adj: first to last
            xb, grid = self.slots[n]
            a = self.terms[n].ad(y[self.offsets[n]:self.offsets[n + 1]])
            nx, ny, nz = xb["t"].shape
            a = {k: _host64(a[k]) if k in a else np.zeros((nx, ny) if k == "psfc" else (nx, ny, nz))
                 for k in ("u", "v", "t", "q", "psfc", "p", "rho")}
            a = pt_to_rho_adj(a, xb)
            del a["rho"]
            forcings[n] = t4.tl_to_xa_adj(a, xb, grid)
        a0 = self.model.ad(forcings)
        xb0, grid0 = self.start
        ax = t4.xa_to_tl_adj(a0, xb0, grid0, press=self.press, ph=self.ph)
        nx, ny, nz = xb0["t"].shape
        out = {k: ax[k] for k in ("u", "v", "t", "q", "psfc")}
        out["p"] = np.zeros((nx, ny, nz))
        out["rho"] = np.zeros((nx, ny, nz))
        assert set(out) == set(XA_KEYS)
        return out
