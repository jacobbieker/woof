"""WRFDA's H' and H'^T on the device, the host operators' words (EXPERIMENTAL;
docs/dev/WRFDA-3DVAR-PLAN.md, sections 4.5 and 4.6; plan item V1.6).

:class:`DeviceObsTerm` is :class:`~woof.da.var.obsop_wrfda.FlatObsTerm` with
its interpolation types on the device (``kernels/var_obs.cu``):

- ``H'`` is a gather: one thread per observation level evaluates
  ``da_interp_lin_3d`` (``_2d`` for the surface pressure) in the host's
  operation order; ``gpsref`` first maps ``t``, ``q`` to refractivity
  elementwise (``da_transform_xtogpsref_lin``).
- ``H'^T`` is the host's scatter rewritten as a gather: each report's
  ``da_interp_lin_3d_adj`` profile is formed by one thread, then one thread
  per ``(column, level)`` adds every contribution to its point in the order
  the serial loops add them (types in ``da_transform_xtoy_adj``'s order,
  each type's fields, reports, the four corners), from a list fixed once at
  construction.  No atomics.

The station-dependent ``gpspw`` and ``satem`` operators (a few hundred
reports with per-report branches) stay on the host
(:class:`~woof.da.var.obsop_wrfda.ObsStencil`'s own code): their ``H'``
reads the fields it needs from the device, and their ``H'^T`` hands its
per-report values to the same ordered gather, so every point still sees its
additions in the serial order.  Arithmetic is binary64 throughout, so ``tl``
and ``ad`` return the host term's words on CUDA and on Metal; the tier's
real type only enters where the fields do (binary32 fields are exact in
binary64, and ``ad``'s fields are rounded once to the tier's type, as the
host's cast).
"""

from __future__ import annotations

import numpy as np

from . import innov_wrfda as iw
from .obsop_wrfda import _LINEAR, COEFF, FlatObsTerm, xtotpw
from .transforms import PRECISIONS, blocks

__all__ = ["DeviceObsTerm"]

_THREADS = 256
#: the adjoint's 3-D outputs (``gref`` is the gpsref refractivity adjoint).
_FIELDS3 = ("u", "v", "t", "q", "p", "rho", "gref")


class _Block:
    """One run of adjoint contributions, in emission order."""

    def __init__(self, field, col, src, w, lev=None, flag=None):
        n = len(col)
        self.field = field
        self.col = np.asarray(col, dtype=np.int64)
        self.src = np.asarray(src, dtype=np.int64)
        self.w = np.asarray(w, dtype=np.float64)
        self.lev = np.full(n, -1, dtype=np.int32) if lev is None else np.asarray(lev, dtype=np.int32)
        self.flag = np.zeros(n, dtype=np.int32) if flag is None else np.asarray(flag, dtype=np.int32)


def _corners(i0, j0, ny):
    """The four columns of each report, (0,0) (1,0) (0,1) (1,1), report-major."""
    i0 = np.asarray(i0, dtype=np.int64)
    j0 = np.asarray(j0, dtype=np.int64)
    return np.stack([i0 * ny + j0, (i0 + 1) * ny + j0, i0 * ny + j0 + 1, (i0 + 1) * ny + j0 + 1], axis=-1)


class DeviceObsTerm:
    """:class:`FlatObsTerm`'s ``tl``/``ad`` on the device for one precision tier.

    ``d``, ``err``, ``mask`` and :meth:`jo_split` are the flat term's.
    ``tl(x)`` takes the tier's device fields and returns ``y`` as a binary64
    device vector; ``ad(y)`` takes a device vector and returns the seven
    fields (``u v t q p rho`` ``(nx, ny, nz)``, ``psfc`` ``(nx, ny)``) in the
    tier's real type.
    """

    def __init__(self, flat: FlatObsTerm, precision: str):
        import cupy as cp

        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, not {precision!r}")
        if flat.term.xtoxa is not None:
            raise NotImplementedError("DeviceObsTerm composes with DeviceU's xa (p, rho), not xtoxa")
        self.flat = flat
        self.precision = precision
        self.dtype = np.float64 if precision == "oracle" else np.float32
        self.d, self.err, self.mask, self.size = flat.d, flat.err, flat.mask, flat.size
        st = flat.term.stencil
        self.st = st
        xb = st.xb
        nx, ny, nz = xb["u"].shape
        self.nx, self.ny, self.nz = nx, ny, nz
        self.ncol = nx * ny
        i32 = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.int32))  # noqa: E731
        i64 = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.int64))  # noqa: E731
        f64 = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.float64))  # noqa: E731

        # per type: geometry on the device
        self.geo = {}
        for n in st.types:
            g = st.interp[n].geo
            nrep, nlev = g.nrep, g.nlev
            off = np.concatenate([[0], np.cumsum(np.bincount(g.rep, minlength=nrep))]).astype(np.int64)
            self.geo[n] = {"nrep": nrep, "nlev": nlev, "i0": i32(g.h[0]), "j0": i32(g.h[1]),
                           "dx": f64(g.h[2]), "dy": f64(g.h[3]), "dxm": f64(g.h[4]), "dym": f64(g.h[5]),
                           "rep": i32(g.rep), "k": i32(g.k), "dz": f64(g.dz), "dzm": f64(g.dzm), "off": i64(off)}

        # the adjoint's contribution list, in the serial loops' order
        where = {(n, v): a for n, v, a, _ in flat.slots}
        self.host_types = [n for n in st.types if n in ("gpspw", "satem")]
        rows = sum(st.interp[n].geo.nrep * len(_LINEAR[n] if n in _LINEAR else ("ref",))
                   for n in st.types if n in _LINEAR or n == "gpsref")
        self.rows_device = rows
        gp_rows = 2 * sum(len(t) for _, _, t in st.gpspw.pieces) if "gpspw" in st.types else 0
        satem_reps = []
        if "satem" in st.types:
            sa = st.satem
            satem_reps = [n for n in range(len(sa.w)) if sa.off[n + 1] != sa.off[n]]
        self.host_rows = {"satem": (gp_rows, len(satem_reps))}
        self.rows_host = gp_rows + 2 * len(satem_reps)
        blocks_: list[_Block] = []
        self.fmz_slots = []  # (type, y0, row0): device-formed profiles
        r0 = 0
        for n in st.types:
            geo = st.interp[n].geo
            nrep = geo.nrep
            cols = _corners(geo.h[0], geo.h[1], ny)
            w4 = np.stack(st.interp[n].w4, axis=-1)
            if n in _LINEAR or n == "gpsref":
                for v in (_LINEAR[n] if n in _LINEAR else ("ref",)):
                    self.fmz_slots.append((n, where[(n, v)], r0))
                    src = np.repeat(r0 + np.arange(nrep), 4)
                    blocks_.append(_Block("gref" if v == "ref" else v, cols.ravel(), src, w4.ravel()))
                    r0 += nrep
                if n in iw.SURFACE_TYPES:
                    if geo.nlev != nrep:
                        raise ValueError(f"{n}: a surface type has one level per report")
                    src = np.repeat(where[(n, "p")] + np.arange(nrep), 4)
                    blocks_.append(_Block("psfc", cols.ravel(), src, w4.ravel()))
            elif n == "gpspw":
                # per report, per term: q's four corners then rho's, at level k
                h = rows
                for (i0, j0, dx, dy, dxm, dym), (_, _, terms) in zip(st.gpspw.w, st.gpspw.pieces):
                    cc = _corners(i0, j0, ny)
                    ww = (dym * dxm, dym * dx, dy * dxm, dy * dx)
                    for k, _rk, _qk, _dz in terms:
                        for f in ("q", "rho"):
                            blocks_.append(_Block(f, cc, np.full(4, h), ww, np.full(4, k)))
                            h += 1
            elif n == "satem":
                nr = len(satem_reps)
                if nr:
                    w = [st.satem.w[m] for m in satem_reps]
                    cc = np.stack([_corners(a[0], a[1], ny) for a in w])
                    ww = np.array([(dym * dxm, dym * dx, dy * dxm, dy * dx) for _, _, dx, dy, dxm, dym in w])
                    ap = np.repeat(rows + gp_rows + np.arange(nr), 4)
                    atv = ap + nr
                    blocks_.append(_Block("p", cc.ravel(), ap, ww.ravel()))
                    blocks_.append(_Block("t", cc.ravel(), atv, ww.ravel(), flag=np.ones(4 * nr)))
                    blocks_.append(_Block("q", cc.ravel(), atv, ww.ravel(), flag=np.ones(4 * nr)))
        hrows = self.rows_host

        self.lists = {}
        for f in _FIELDS3 + ("psfc",):
            bs = [b for b in blocks_ if b.field == f]
            if not bs:
                continue
            col = np.concatenate([b.col for b in bs])
            order = np.argsort(col, kind="stable")
            row = np.zeros(self.ncol + 1, dtype=np.int64)
            row[1:] = np.cumsum(np.bincount(col, minlength=self.ncol))
            if row[-1] >= 2 ** 31 or rows + hrows >= 2 ** 31:
                raise ValueError("observation adjoint list too long for int32 indices")
            cat = lambda name: np.concatenate([getattr(b, name) for b in bs])[order]  # noqa: E731
            self.lists[f] = {"row": i32(row), "src": i32(cat("src")), "w": f64(cat("w")),
                             "lev": i32(cat("lev")), "flag": i32(cat("flag"))}

        # background pieces
        self.bg = {}
        if "satem" in st.types:
            self.bg["t"] = f64(1.0 + 0.61 * xb["q"])
            self.bg["q"] = f64(0.61 * xb["t"])
        if st.use_ref:
            pb, tb, qb = xb["p"], xb["t"], xb["q"]
            a = 0.776 * pb
            c = 0.622 + 0.378 * qb
            e = tb * c
            f = COEFF * qb
            self.ref_bg = tuple(f64(z) for z in (a, tb * tb, a / tb, c, tb * 0.378, e, f, e * e, 1.0 + f / e))
        if st.use_tpw:
            hf = xb["hf"]
            self.tpw_bg = (f64(hf[:, :, 1:] - hf[:, :, :-1]), f64(xb["rho"]), f64(xb["q"]))
        self._dummy = cp.zeros(1, dtype=cp.float64)
        self._coeff = f64([COEFF])

    # ---------------------------------------------------------------- helpers

    def _go(self, name, n, *args):
        from .transforms import kernel

        kernel(self.precision, name, "var_obs.cu")((blocks(n),), (_THREADS,), args)

    def jo_split(self):
        return self.flat.jo_split()

    def _host_fields(self, x, keys):
        return {k: np.asarray(x[k].get(), dtype=np.float64) for k in keys}

    # ---------------------------------------------------------------- H'

    def tl(self, x: dict):
        """``H' x`` for the tier's device fields; a binary64 device vector."""
        import cupy as cp

        st = self.st
        ny, nz = self.ny, self.nz
        x = {k: cp.ascontiguousarray(v) for k, v in x.items()}
        y = cp.zeros(self.size, dtype=cp.float64)
        ref = None
        host = {}
        for n, v, a, b in self.flat.slots:
            g = self.geo[n]
            h = (g["i0"], g["j0"], g["dx"], g["dy"], g["dxm"], g["dym"])
            if n in st.types and (n in _LINEAR or n == "gpsref"):
                if v == "p" and n in iw.SURFACE_TYPES:
                    if g["nrep"]:
                        self._go("var_obs_tl2", g["nrep"], x["psfc"].reshape(-1), np.int32(ny), *h,
                                 np.int64(g["nrep"]), y, np.int64(a))
                    continue
                if v == "ref":
                    if ref is None:
                        ref = cp.empty(self.nx * ny * nz, dtype=cp.float64)
                        self._go("var_obs_gpsref_tl", ref.size, x["t"], x["q"], *self.ref_bg, self._coeff,
                                 np.int64(ref.size), ref)
                    f = ref
                else:
                    f = x[v]
                if g["nlev"]:
                    self._go("var_obs_tl3_f64" if v == "ref" else "var_obs_tl3", g["nlev"], f, np.int32(ny), np.int32(nz), *h, g["rep"], g["k"],
                             g["dz"], g["dzm"], np.int64(g["nlev"]), y, np.int64(a))
            else:
                host[(n, v)] = (a, b)
        if host:
            need = set()
            if "gpspw" in self.host_types:
                need |= {"q", "rho"}
            if "satem" in self.host_types:
                need |= {"p", "t", "q"}
            xh = self._host_fields(x, sorted(need))
            for (n, v), (a, b) in host.items():
                if n == "gpspw":
                    vals = st.gpspw.tl(xh, xtotpw(xh, st.xb))
                else:
                    vals = st.satem.tl(xh)
                y[a:b] = cp.asarray(np.asarray(vals, dtype=np.float64))
        return y

    # ---------------------------------------------------------------- H'^T

    def ad(self, y) -> dict:
        """``H'^T y`` into fresh fields of the tier's real type."""
        import cupy as cp

        st = self.st
        nx, ny, nz = self.nx, self.ny, self.nz
        n3 = nx * ny * nz
        yd = cp.ascontiguousarray(cp.asarray(y).astype(cp.float64).ravel())
        s = cp.zeros(max(1, (self.rows_device + self.rows_host) * nz), dtype=cp.float64)
        for n, y0, r0 in self.fmz_slots:
            g = self.geo[n]
            if g["nrep"]:
                self._go("var_obs_fmz", g["nrep"], yd, np.int64(y0), g["off"], g["k"], g["dz"], g["dzm"],
                         np.int64(g["nrep"]), np.int32(nz), s, np.int64(r0))
        gtpw = None
        if self.host_types:
            yh = yd.get()
            hrows = np.zeros((self.rows_host, nz))
            where = {(n, v): (a, b) for n, v, a, b in self.flat.slots}
            if "gpspw" in self.host_types:
                a, b = where[("gpspw", "tpw")]
                gtpw = np.zeros((nx, ny))
                r = 0
                for _w, k, mdq, mdrho in st.gpspw.ad_terms(gtpw, yh[a:b]):
                    hrows[r, k] = mdq
                    hrows[r + 1, k] = mdrho
                    r += 2
            if "satem" in self.host_types:
                a, b = where[("satem", "thickness")]
                h0, nr = self.host_rows["satem"]
                for r, (_w, ap_s, atv_s) in enumerate(st.satem.ad_reports(yh[a:b])):
                    hrows[h0 + r] = ap_s
                    hrows[h0 + nr + r] = atv_s
            if self.rows_host:
                s[self.rows_device * nz:] = cp.asarray(hrows.ravel())
        out = {}
        for f in _FIELDS3:
            if f == "gref" and not st.use_ref:
                continue
            o = cp.zeros(n3, dtype=cp.float64)
            lst = self.lists.get(f)
            if lst is not None:
                self._go("var_obs_gather3", n3, o, lst["row"], lst["src"], lst["w"], lst["lev"], lst["flag"], s,
                         self.bg.get(f, self._dummy), np.int64(self.ncol), np.int32(nz))
            out[f] = o
        ps = cp.zeros(nx * ny, dtype=cp.float64)
        lst = self.lists.get("psfc")
        if lst is not None:
            self._go("var_obs_gather2", nx * ny, ps, lst["row"], lst["src"], lst["w"], yd, np.int64(self.ncol))
        out["psfc"] = ps
        # da_transform_xtoxa_adj's observation part: gpsref, then tpw
        if st.use_ref:
            self._go("var_obs_gpsref_adj", n3, out.pop("gref"), out["q"], out["t"], *self.ref_bg, self._coeff,
                     np.int64(n3))
        if st.use_tpw:
            gt = cp.zeros(nx * ny, dtype=cp.float64) if gtpw is None else cp.asarray(np.ascontiguousarray(gtpw))
            self._go("var_obs_tpw_adj", nx * ny, gt, out["q"], out["rho"], *self.tpw_bg, np.int64(nx * ny),
                     np.int32(nz))
        res = {}
        for k in ("u", "v", "t", "q", "p", "rho", "psfc"):
            a = out[k]
            if self.dtype != np.float64:
                r = cp.empty(a.size, dtype=self.dtype)
                self._go("var_obs_to_real", a.size, a, r, np.int64(a.size))
                a = r
            res[k] = a.reshape((nx, ny) if k == "psfc" else (nx, ny, nz))
        return res
