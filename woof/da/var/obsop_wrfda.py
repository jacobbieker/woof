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

from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from . import innov_wrfda as iw

__all__ = ["ObsStencil", "ObsTerm", "FlatObsTerm", "AD_ORDER", "xtotpw", "xtotpw_adj",
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


# --------------------------------------------------------------------------- tpw, ref
COEFF = (3.73e-3 * 1.0e8) / 77.6


def xtotpw(x, xb):
    """``da_transform_xtotpw``: ``tpw = 0.1 * sum_k dhf (q' rho_b + q_b rho')``."""
    hf = xb["hf"]
    pw = np.zeros(hf.shape[:2])
    for k in range(hf.shape[2] - 1):
        pw = pw + (hf[:, :, k + 1] - hf[:, :, k]) * (x["q"][:, :, k] * xb["rho"][:, :, k]
                                                    + xb["q"][:, :, k] * x["rho"][:, :, k])
    return 0.1 * pw


def xtotpw_adj(tpw, g, xb):
    """``da_transform_xtotpw_adj``: accumulate into ``g["q"]``, ``g["rho"]``."""
    hf = xb["hf"]
    pw = 0.1 * tpw
    for k in range(hf.shape[2] - 1):
        dzpw = (hf[:, :, k + 1] - hf[:, :, k]) * pw
        g["q"][:, :, k] = g["q"][:, :, k] + dzpw * xb["rho"][:, :, k]
        g["rho"][:, :, k] = g["rho"][:, :, k] + dzpw * xb["q"][:, :, k]


def xtogpsref_lin(x, xb):
    """``da_transform_xtogpsref_lin`` (``xa%p`` does not enter, as in WRFDA)."""
    pb, tb, qb, ta, qa = xb["p"], xb["t"], xb["q"], x["t"], x["q"]
    partone = -(0.776 * pb * ta / (tb * tb))
    partone9 = 0.776 * pb / tb
    dividnd = ta * (0.622 + 0.378 * qb) + tb * 0.378 * qa
    dividnd9 = tb * (0.622 + 0.378 * qb)
    parttwo = COEFF * qa / dividnd9 - COEFF * qb * dividnd / (dividnd9 * dividnd9)
    parttwo9 = 1.0 + COEFF * qb / dividnd9
    return partone9 * parttwo + partone * parttwo9


def xtogpsref_adj(ref, g, xb):
    """``da_transform_xtogpsref_adj``: accumulate into ``g["q"]``, ``g["t"]``."""
    pb, tb, qb = xb["p"], xb["t"], xb["q"]
    partone9 = 0.776 * pb / tb
    dividnd9 = tb * (0.622 + 0.378 * qb)
    parttwo9 = 1.0 + COEFF * qb / dividnd9
    partone = ref * parttwo9
    parttwo = ref * partone9
    g["q"] = g["q"] + COEFF * parttwo / dividnd9
    dividnd = -(COEFF * qb * parttwo / (dividnd9 * dividnd9))
    g["t"] = g["t"] + dividnd * (0.622 + 0.378 * qb)
    g["q"] = g["q"] + tb * 0.378 * dividnd
    g["t"] = g["t"] - 0.776 * pb * partone / (tb * tb)


# --------------------------------------------------------------------------- gpspw
class _Gpspw:
    """``da_transform_xtoy_gpspw`` and ``_adj``; the station-height term's
    background pieces are fixed here."""

    def __init__(self, t, xb):
        self.w = list(zip(*iw._w(t.info)))
        self.pieces = [iw._gpspw_dpw(w, float(e), xb) for w, e in zip(self.w, t.info["elv"])]

    def tl(self, x, tpw):
        y = np.empty(len(self.w))
        for n, (w, (branch, _, terms)) in enumerate(zip(self.w, self.pieces)):
            dpw = 0.0
            for k, rk, qk, dz in terms:
                dq = iw._hcol(x["q"][:, :, k], *w)
                drho = iw._hcol(x["rho"][:, :, k], *w)
                ddpw = (rk * dq + drho * qk) * dz
                dpw = ddpw if branch == 1 else dpw + ddpw
            y[n] = iw._hcol(tpw, *w) + 0.1 * dpw
        return y

    def ad(self, g, gtpw, yv):
        for n, (w, (branch, _, terms)) in enumerate(zip(self.w, self.pieces)):
            i0, j0, dx, dy, dxm, dym = w
            gy = yv[n]
            dpw = 0.1 * gy
            gtpw[i0, j0] = gtpw[i0, j0] + dxm * dym * gy
            gtpw[i0 + 1, j0] = gtpw[i0 + 1, j0] + dym * dx * gy
            gtpw[i0, j0 + 1] = gtpw[i0, j0 + 1] + dxm * dy * gy
            gtpw[i0 + 1, j0 + 1] = gtpw[i0 + 1, j0 + 1] + dx * dy * gy
            for k, rk, qk, dz in terms:
                mdq = rk * dz * dpw
                mdrho = qk * dz * dpw
                for f, val in (("q", mdq), ("rho", mdrho)):
                    a = g[f]
                    a[i0, j0, k] = a[i0, j0, k] + dym * dxm * val
                    a[i0 + 1, j0, k] = a[i0 + 1, j0, k] + dym * dx * val
                    a[i0, j0 + 1, k] = a[i0, j0 + 1, k] + dy * dxm * val
                    a[i0 + 1, j0 + 1, k] = a[i0 + 1, j0 + 1, k] + dy * dx * val


# --------------------------------------------------------------------------- satem
class _Satem:
    """``da_transform_xtoy_satem`` and ``_adj`` (``da_tv_profile_tl/_adj``,
    ``da_find_layer_tl/_adj``, ``da_thickness_tl/_adj``), transcribed as
    WRFDA has them. Two WRFDA properties carry over: the levels run while
    ``p`` is not missing (the innovation stops at ``ptop``), and ``TGL_tv2``
    stays the reference level's for every level, although the nonlinear
    thickness takes ``tv2`` from the level below (so H' is the derivative of
    the innovation's thickness on each report's first level only; the
    adjoint is the exact adjoint of this H')."""

    def __init__(self, t, xb, lm):
        # Every log argument here is a background ratio, the same each call:
        # memoising the (correctly rounded, decimal) log returns the same words
        # and takes it out of the per-iteration cost.
        self.lm = SimpleNamespace(log=lru_cache(maxsize=None)(lm.log))
        self.xb = xb
        self.w = list(zip(*iw._w(t.info)))
        self.off = t.offsets
        self.p = t.coords["p"]
        self.ref_p = t.extra["ref_p"]
        self.n = len(self.p)

    def _levels(self, n):
        return [lv for lv in range(self.off[n], self.off[n + 1]) if abs(self.p[lv] - iw.MISSING_R) > 1.0]

    def _coefs(self, pre, pre_ma, tv_ma):
        """find_layer's branch and its linearisation (and the nonlinear update)."""
        lm = self.lm
        layer, tv, a = iw.find_layer(pre, pre_ma, tv_ma, lm)
        b = a + 1
        l2 = lm.log(pre_ma[a] / pre_ma[b])
        coef1 = lm.log(pre / pre_ma[b]) / (pre_ma[a] * (l2 * l2))
        coef2 = lm.log(pre_ma[a] / pre) / (pre_ma[b] * (l2 * l2))
        alpha = lm.log(pre_ma[a] / pre) / l2
        ghost = 0 if layer == 1 else (len(pre_ma) - 1 if layer == len(pre_ma) - 1 else None)
        return layer, tv, (a, b, coef1, coef2, alpha, ghost)

    def tl(self, x):
        lm, xb = self.lm, self.xb
        y = np.zeros(self.n)
        tvx = x["t"] * (1.0 + 0.61 * xb["q"]) + 0.61 * xb["t"] * x["q"]
        for n, w in enumerate(self.w):
            levels = self._levels(n)
            if self.off[n + 1] == self.off[n]:
                continue
            pre_ma, tv_ma = iw.tv_profile(xb, w)
            i0, j0 = w[0], w[1]
            ww = (0, 0, *w[2:])
            kte = len(pre_ma) - 2
            tp = np.zeros(kte + 2)
            ttv = np.zeros(kte + 2)
            tp[1:kte + 1] = iw._hcol(x["p"][i0:i0 + 2, j0:j0 + 2], *ww)
            ttv[1:kte + 1] = iw._hcol(tvx[i0:i0 + 2, j0:j0 + 2], *ww)

            def find_tl(pre):
                layer, tv, (a, b, c1, c2, al, ghost) = self._coefs(pre, pre_ma, tv_ma)
                talpha = c1 * tp[a] + c2 * tp[b]
                ttv_ = (1.0 - al) * ttv[a] + (tv_ma[b] - tv_ma[a]) * talpha + al * ttv[b]
                if ghost is not None:
                    tp[ghost] = 0.0
                return layer, tv, ttv_

            layer2, tv2, ttv2 = find_tl(self.ref_p[n])
            pres2 = self.ref_p[n]
            for lv in levels:
                pre1 = self.p[lv]
                layer1, tv1, ttv1 = find_tl(pre1)
                thk = (ttv1 * lm.log(pre_ma[layer1 - 1] / pre1) + tp[layer1 - 1] * tv1 / pre_ma[layer1 - 1]
                       + ttv2 * lm.log(pres2 / pre_ma[layer2]) - tp[layer2] * tv2 / pre_ma[layer2])
                thk = iw.THK * thk
                p_tmp, tp_tmp = {layer1: pre1}, {layer1: 0.0}
                p_tmp[layer2 - 1], tp_tmp[layer2 - 1] = pres2, 0.0
                for k in range(layer2, layer1):
                    tp_tmp[k], p_tmp[k] = tp[k], pre_ma[k]
                for k in range(layer2, layer1):
                    thk = thk + iw.THK * (ttv[k] * lm.log(p_tmp[k - 1] / p_tmp[k + 1])
                                          + tp_tmp[k - 1] * tv_ma[k] / p_tmp[k - 1]
                                          - tp_tmp[k + 1] * tv_ma[k] / p_tmp[k + 1])
                y[lv] = thk
                pres2, layer2, tv2 = pre1, layer1, tv1
        return y

    def ad(self, g, yv):
        lm, xb = self.lm, self.xb
        kte = xb["p"].shape[2]
        ap = np.zeros(kte + 2)
        atv = np.zeros(kte + 2)

        def find_adj(pre, pre_ma, tv_ma, a_tv):
            _, _, (a, b, c1, c2, al, ghost) = self._coefs(pre, pre_ma, tv_ma)
            if ghost is not None:
                ap[ghost] = 0.0
            atv[a] = atv[a] + (1.0 - al) * a_tv
            aalpha = (tv_ma[b] - tv_ma[a]) * a_tv
            atv[b] = atv[b] + al * a_tv
            ap[a] = ap[a] + c1 * aalpha
            ap[b] = ap[b] + c2 * aalpha

        for n, w in enumerate(self.w):
            if self.off[n + 1] == self.off[n]:
                continue
            pre_ma, tv_ma = iw.tv_profile(xb, w)
            layer2, tv2, _ = iw.find_layer(self.ref_p[n], pre_ma, tv_ma, lm)
            pres2 = self.ref_p[n]
            atv1 = atv2 = 0.0
            for lv in self._levels(n):
                pre1 = self.p[lv]
                layer1, tv1, _ = iw.find_layer(pre1, pre_ma, tv_ma, lm)
                # da_thickness_adj
                athk = yv[lv]
                p_tmp = {layer1: pre1}
                p_tmp[layer2 - 1] = pres2
                for k in range(layer2, layer1):
                    p_tmp[k] = pre_ma[k]
                ap_tmp = {k: 0.0 for k in range(layer2 - 1, layer1 + 1)}
                for k in range(layer2, layer1):
                    ap_tmp[k + 1] = ap_tmp[k + 1] - iw.THK * athk * tv_ma[k] / p_tmp[k + 1]
                    ap_tmp[k - 1] = ap_tmp[k - 1] + iw.THK * athk * tv_ma[k] / p_tmp[k - 1]
                    atv[k] = atv[k] + iw.THK * athk * lm.log(p_tmp[k - 1] / p_tmp[k + 1])
                for k in range(layer2, layer1):
                    ap[k] = ap[k] + ap_tmp[k]
                athk = iw.THK * athk
                ap[layer2] = ap[layer2] - athk * tv2 / pre_ma[layer2]
                atv2 = atv2 + athk * lm.log(pres2 / pre_ma[layer2])
                ap[layer1 - 1] = ap[layer1 - 1] + athk * tv1 / pre_ma[layer1 - 1]
                atv1 = atv1 + athk * lm.log(pre_ma[layer1 - 1] / pre1)
                # da_find_layer_adj for this level (re-runs the nonlinear find_layer)
                find_adj(pre1, pre_ma, tv_ma, atv1)
                atv1 = 0.0
                pres2, layer2, tv2 = pre1, layer1, tv1
            find_adj(self.ref_p[n], pre_ma, tv_ma, atv2)
            atv2 = 0.0
            # da_tv_profile_adj
            i0, j0, dx, dy, dxm, dym = w
            s = slice(1, kte + 1)
            atm = {(0, 0): dym * dxm * atv[s], (1, 0): dym * dx * atv[s],
                   (0, 1): dy * dxm * atv[s], (1, 1): dy * dx * atv[s]}
            gp = g["p"]
            gp[i0, j0] = gp[i0, j0] + dym * dxm * ap[s]
            gp[i0 + 1, j0] = gp[i0 + 1, j0] + dym * dx * ap[s]
            gp[i0, j0 + 1] = gp[i0, j0 + 1] + dy * dxm * ap[s]
            gp[i0 + 1, j0 + 1] = gp[i0 + 1, j0 + 1] + dy * dx * ap[s]
            atv[s] = 0.0
            ap[s] = 0.0
            for di in (0, 1):
                for dj in (0, 1):
                    a, b = i0 + di, j0 + dj
                    g["t"][a, b] = g["t"][a, b] + atm[(di, dj)] * (1.0 + 0.61 * xb["q"][a, b])
                    g["q"][a, b] = g["q"][a, b] + 0.61 * xb["t"][a, b] * atm[(di, dj)]


class ObsStencil:
    """H' and H'^T for the innovation vector ``iv`` at the background ``xb``."""

    def __init__(self, iv, xb: dict, *, libm: str = "correctly-rounded"):
        self.xb = {k: (np.asarray(v, dtype=np.float64) if not np.isscalar(v) else v)
                   for k, v in xb.items()}
        self.kte = self.xb["u"].shape[2]
        self.types = [n for n in AD_ORDER if n in iv.types and n in iw.IMPLEMENTED
                      and "geometry" in iv.types[n].extra]
        self.interp = {n: _Interp(iv.types[n].extra["geometry"], self.kte) for n in self.types}
        self.sizes = {n: len(iv.types[n].extra["geometry"].rep) for n in self.types}
        self.gpspw = _Gpspw(iv.types["gpspw"], self.xb) if "gpspw" in self.types else None
        self.satem = (_Satem(iv.types["satem"], self.xb, iw._libm(libm))
                      if "satem" in self.types else None)
        self.use_tpw = "gpspw" in self.types or "gpsref" in self.types
        self.use_ref = "gpsref" in self.types

    # ------------------------------------------------------------------ TL
    def tl(self, x: dict) -> dict:
        """``da_transform_xtoy``: ``{type: {field: H'x per level}}``."""
        y = {}
        tpw = xtotpw(x, self.xb) if self.use_tpw else None
        ref = xtogpsref_lin(x, self.xb) if self.use_ref else None
        for n in self.types:
            it = self.interp[n]
            out = {}
            if n in _LINEAR:
                for v in _LINEAR[n]:
                    out[v] = it.tl3(x[v])
                if n in iw.SURFACE_TYPES:
                    out["p"] = it.tl2(x["psfc"])
            elif n == "gpspw":
                out["tpw"] = self.gpspw.tl(x, tpw)
            elif n == "gpsref":
                out["ref"] = it.tl3(ref)
            elif n == "satem":
                out["thickness"] = self.satem.tl(x)
            y[n] = out
        return y

    # ------------------------------------------------------------------ AD
    def ad(self, y: dict) -> dict:
        """``da_transform_xtoy_adj``: H'^T y as fresh x-space arrays."""
        nx, ny, nz = self.xb["u"].shape
        g = {v: np.zeros((nx, ny, nz)) for v in ("u", "v", "t", "q", "p", "rho")}
        g["psfc"] = np.zeros((nx, ny))
        gtpw = np.zeros((nx, ny))
        gref = np.zeros((nx, ny, nz))
        for n in self.types:
            it = self.interp[n]
            yy = y[n]
            if n in _LINEAR:
                for v in _LINEAR[n]:
                    it.ad3(g[v], np.asarray(yy[v], dtype=np.float64))
                if n in iw.SURFACE_TYPES:
                    it.ad2(g["psfc"], yy["p"])
            elif n == "gpspw":
                self.gpspw.ad(g, gtpw, np.asarray(yy["tpw"], dtype=np.float64))
            elif n == "gpsref":
                it.ad3(gref, np.asarray(yy["ref"], dtype=np.float64))
            elif n == "satem":
                self.satem.ad(g, np.asarray(yy["thickness"], dtype=np.float64))
        # da_transform_xtoxa_adj's observation part: gpsref, then tpw
        if self.use_ref:
            xtogpsref_adj(gref, g, self.xb)
        if self.use_tpw:
            xtotpw_adj(gtpw, g, self.xb)
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

    def __init__(self, iv, xb: dict, *, xtoxa=None, libm: str = "correctly-rounded"):
        self.iv = iv
        self.stencil = ObsStencil(iv, xb, libm=libm)
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


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


class FlatObsTerm:
    """:class:`ObsTerm` as :class:`woof.da.var.cost.ObsTerm`: one flat vector.

    ``y`` is every type's fields concatenated (types in :data:`AD_ORDER`,
    each type's fields in :meth:`ObsStencil.fields`'s order, levels report
    by report).  :meth:`jo_split` gives ``da_jo_and_grady``'s groups: per
    type its fields in ``JO_FIELDS`` order, each over the ``proc_domain``
    reports' levels.  ``tl``/``ad`` take host or device arrays and return
    host binary64 arrays; the stencil runs on the host in binary64, so its
    words do not depend on the backend.
    """

    def __init__(self, iv, xb: dict, *, xtoxa=None, libm: str = "correctly-rounded"):
        self.term = ObsTerm(iv, xb, xtoxa=xtoxa, libm=libm)
        st = self.term.stencil
        self.slots = []  # (type, field, start, stop)
        off = 0
        for n in st.types:
            for v in ObsStencil.fields(n):
                m = len(self.term.d[n][v])
                self.slots.append((n, v, off, off + m))
                off += m
        self.size = off

        def cat(dd):
            if not self.slots:
                return np.zeros(0)
            return np.concatenate([np.asarray(dd[n][v]) for n, v, _, _ in self.slots])

        self.d = cat(self.term.d).astype(np.float64)
        self.err = cat(self.term.err).astype(np.float64)
        self.mask = cat(self.term.mask).astype(bool)
        where = {(n, v): a for n, v, a, _ in self.slots}
        self._groups = []
        for n in iw.JO_ORDER:
            if n not in st.types or iv.types[n].nlocal == 0:
                continue
            t = iv.types[n]
            pd = np.repeat(np.asarray(t.info["proc_domain"], dtype=bool), t.info["levels"])
            self._groups.append((n, [(v, where[(n, v)] + np.flatnonzero(pd)) for v in iw.JO_FIELDS[n]]))

    def jo_split(self):
        return self._groups

    def tl(self, x: dict):
        y = self.term.tl({k: np.asarray(_host(v), dtype=np.float64) for k, v in x.items()})
        out = np.zeros(self.size, dtype=np.float64)
        for n, v, a, b in self.slots:
            out[a:b] = y[n][v]
        return out

    def ad(self, y) -> dict:
        y = np.asarray(_host(y), dtype=np.float64)
        yy: dict = {}
        for n, v, a, b in self.slots:
            yy.setdefault(n, {})[v] = y[a:b]
        return self.term.ad(yy)
