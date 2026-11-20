"""WRFDA's linearised observation operators H' and their adjoints (EXPERIMENTAL).

:class:`ObsStencil` is fixed at the innovation step
(:func:`woof.da.var.innov_wrfda.innovations`): per type it holds each
report's horizontal weights ``(i, j, dx, dy, dxm, dym)``, each level's
vertical ``(k, dz, dzm)`` and, for the nonlinear types, the background
linearisation. :meth:`ObsStencil.tl` is a gather; :meth:`ObsStencil.ad` is a
scatter in WRFDA's own loop order (``da_transform_xtoy_adj``'s type order,
each type's field order, reports in order, ``da_interp_lin_3d_adj``'s
``fmz`` accumulation then column scatter). Reports whose four columns are
disjoint are scattered together ("waves"), which keeps every cell's
sequence of additions exactly as the serial loop has it: no unordered
atomics, the same words every run.

The x-space increment is WRFDA's ``grid%xa`` after ``da_transform_xtoxa``:
``u v t q`` and ``p rho`` ``(ix, jy, kz)``, ``psfc`` ``(ix, jy)``. Surface
types read ``psfc`` for pressure (``sfc_assi_options = 1``); ``tpw``
(``da_transform_xtotpw``) and ``ref`` (``da_transform_xtogpsref_lin``) are
derived inside :meth:`~ObsStencil.tl`, and their adjoints are applied at the
end of :meth:`~ObsStencil.ad` as ``da_transform_xtoxa_adj`` does before its
``rho``/``p`` part. :class:`ObsTerm` composes with
:func:`woof.da.var.reference.xtoxa` when the caller's x is ``u v t q psfc``.

Derived from WRFDA 4.7.1 (``da_transform_xtoy_*``, ``da_interp_lin_*``,
``da_transform_xtotpw*``, ``da_transform_xtogpsref*``, ``da_tv_profile*``,
``da_find_layer*``, ``da_thickness*``), public domain; see
``licenses/LICENSE-WRF-public-domain.txt``.
"""

from __future__ import annotations

import numpy as np

from . import innov_wrfda as iw

__all__ = ["ObsStencil", "ObsTerm", "AD_ORDER", "xtotpw", "xtotpw_adj",
           "xtogpsref_lin", "xtogpsref_adj"]

#: ``da_transform_xtoy_adj``'s type order.
AD_ORDER = ("sound", "sonde_sfc", "synop", "geoamv", "polaramv", "airep", "metar", "ships",
            "gpspw", "pilot", "satem", "qscat", "profiler", "buoy", "gpsref")
_LINEAR = {
    "sound": ("u", "v", "t", "q"), "sonde_sfc": ("u", "v", "t", "q"), "synop": ("u", "v", "t", "q"),
    "geoamv": ("u", "v"), "polaramv": ("u", "v"), "airep": ("u", "v", "t", "q"),
    "metar": ("u", "v", "t", "q"), "ships": ("u", "v", "t", "q"), "pilot": ("u", "v"),
    "qscat": ("u", "v"), "profiler": ("u", "v"), "buoy": ("u", "v", "t", "q"),
}
X_FIELDS = ("u", "v", "t", "q", "psfc", "p", "rho")


def _waves(i0, j0):
    """Group reports so that no two in a wave share any of the four columns,
    and every column sees its reports in their original order."""
    last: dict = {}
    wave = np.empty(len(i0), dtype=np.int64)
    for n, (a, b) in enumerate(zip(i0.tolist(), j0.tolist())):
        cells = ((a, b), (a + 1, b), (a, b + 1), (a + 1, b + 1))
        w = 1 + max(last.get(c, -1) for c in cells)
        wave[n] = w
        for c in cells:
            last[c] = w
    return [np.flatnonzero(wave == w) for w in range(int(wave.max()) + 1)] if len(wave) else []


class _Interp:
    """One type's ``da_interp_lin_3d``/``_2d`` and their adjoints."""

    def __init__(self, geo: iw.Geometry, kte: int):
        self.geo = geo
        self.kte = kte
        i0, j0, dx, dy, dxm, dym = geo.h
        self.waves = _waves(i0, j0)
        # adjoint weights, as WRFDA forms them: dym*dxm, dym*dx, dy*dxm, dy*dx
        self.w4 = (dym * dxm, dym * dx, dy * dxm, dy * dx)
        lev = np.bincount(geo.rep, minlength=geo.nrep)
        off = np.concatenate([[0], np.cumsum(lev)])
        self.pos = []
        for p in range(int(lev.max()) if len(lev) else 0):
            reps = np.flatnonzero(lev > p)
            rows = off[reps] + p
            ok = geo.k[rows] > 0
            self.pos.append((reps[ok], rows[ok]))

    def tl3(self, f):
        return iw.interp_lin_3d(f, self.geo)

    def tl2(self, f2):
        return iw.interp_lin_2d(f2, self.geo)

    def _scatter(self, g, val):
        i0, j0 = self.geo.h[0], self.geo.h[1]
        for r in self.waves:
            a, b = i0[r], j0[r]
            for (di, dj), w in zip(((0, 0), (1, 0), (0, 1), (1, 1)), self.w4):
                ww = w[r][:, None] if val.ndim == 2 else w[r]
                g[a + di, b + dj] = ww * val[r] + g[a + di, b + dj]

    def ad3(self, g, fo):
        """``da_interp_lin_3d_adj``: accumulate ``fo`` (per level) into ``g``."""
        geo = self.geo
        fmz = np.zeros((geo.nrep, self.kte))
        for reps, rows in self.pos:
            k0 = geo.k[rows] - 1
            fmz[reps, k0] = geo.dzm[rows] * fo[rows] + fmz[reps, k0]
            fmz[reps, k0 + 1] = geo.dz[rows] * fo[rows] + fmz[reps, k0 + 1]
        self._scatter(g, fmz)

    def ad2(self, g2, fo):
        """``da_interp_lin_2d_adj`` at level 1 (``fo`` per report)."""
        self._scatter(g2, np.asarray(fo, dtype=np.float64))


class ObsStencil:
    """H' and H'^T for the innovation vector ``iv`` at the background ``xb``."""

    def __init__(self, iv, xb: dict):
        self.xb = {k: (np.asarray(v, dtype=np.float64) if not np.isscalar(v) else v)
                   for k, v in xb.items()}
        self.kte = self.xb["u"].shape[2]
        self.types = [n for n in AD_ORDER if n in iv.types and n in iw.IMPLEMENTED
                      and "geometry" in iv.types[n].extra]
        self.interp = {n: _Interp(iv.types[n].extra["geometry"], self.kte) for n in self.types}
        self.sizes = {n: len(iv.types[n].extra["geometry"].rep) for n in self.types}

    # ------------------------------------------------------------------ TL
    def tl(self, x: dict) -> dict:
        """``da_transform_xtoy``: ``{type: {field: H'x per level}}``."""
        y = {}
        for n in self.types:
            it = self.interp[n]
            out = {}
            if n in _LINEAR:
                for v in _LINEAR[n]:
                    out[v] = it.tl3(x[v])
                if n in iw.SURFACE_TYPES:
                    out["p"] = it.tl2(x["psfc"])
            y[n] = out
        return y

    # ------------------------------------------------------------------ AD
    def ad(self, y: dict) -> dict:
        """``da_transform_xtoy_adj``: H'^T y as fresh x-space arrays."""
        nx, ny, nz = self.xb["u"].shape
        g = {v: np.zeros((nx, ny, nz)) for v in ("u", "v", "t", "q", "p", "rho")}
        g["psfc"] = np.zeros((nx, ny))
        for n in self.types:
            it = self.interp[n]
            yy = y[n]
            if n in _LINEAR:
                for v in _LINEAR[n]:
                    it.ad3(g[v], np.asarray(yy[v], dtype=np.float64))
                if n in iw.SURFACE_TYPES:
                    it.ad2(g["psfc"], yy["p"])
        return g

    def zeros_y(self) -> dict:
        return {n: {v: np.zeros(self.sizes[n]) for v in self.fields(n)} for n in self.types}

    @staticmethod
    def fields(n: str) -> tuple:
        if n in _LINEAR:
            return _LINEAR[n] + (("p",) if n in iw.SURFACE_TYPES else ())
        return {"gpspw": ("tpw",), "gpsref": ("ref",), "satem": ("thickness",)}[n]


class ObsTerm:
    """The observation term for a variational cost: ``tl``, ``ad``, the
    innovations ``d``, errors ``err``, ``mask`` (``qc >= 0``) and
    :meth:`jo_split`, all as ``{type: {field: per-level array}}``.

    With ``xtoxa=(xb_fields, c1h, c2h)`` x is ``u v t q psfc`` and
    :func:`woof.da.var.reference.xtoxa` / ``xtoxa_adj`` are composed in;
    without it x already holds ``p`` and ``rho``.
    """

    def __init__(self, iv, xb: dict, *, xtoxa=None):
        self.iv = iv
        self.stencil = ObsStencil(iv, xb)
        self.xtoxa = xtoxa
        self.d, self.err, self.mask = {}, {}, {}
        for n in self.stencil.types:
            t = iv.types[n]
            fs = ObsStencil.fields(n)
            self.d[n] = {v: t.fields[v].inv.copy() for v in fs}
            self.err[n] = {v: t.fields[v].error.copy() for v in fs}
            self.mask[n] = {v: t.fields[v].qc >= 0 for v in fs}

    def tl(self, x: dict) -> dict:
        if self.xtoxa is not None:
            from .reference import xtoxa

            xb, c1h, c2h = self.xtoxa
            x = xtoxa(x, xb, c1h, c2h)
        return self.stencil.tl(x)

    def ad(self, y: dict) -> dict:
        g = self.stencil.ad(y)
        if self.xtoxa is not None:
            from .reference import xtoxa_adj

            xb, c1h, c2h = self.xtoxa
            g = xtoxa_adj(g, xb, c1h, c2h)
            g = {k: g[k] for k in ("u", "v", "t", "q", "psfc")}
        return g

    def jo_split(self, y: dict | None = None) -> dict:
        """Per-type ``Jo`` (``da_jo_and_grady``) at ``H'x = y``."""
        return iw.jo(self.iv, y)
