"""``U`` and ``U^T`` on the device (docs/dev/WRFDA-3DVAR-PLAN.md, section 4.5).

:class:`DeviceU` runs the control-variable transform of
:mod:`woof.da.var.reference` as kernel launches (``kernels/var_u.cu``),
statement for statement in the reference's order, at one of two precisions:

``oracle``
    binary64 storage and arithmetic.  Every operation is a correctly rounded
    intrinsic, so the result is the reference's, word for word, on CUDA and
    on Metal's soft binary64; ``tests/test_var_transforms_gpu.py`` checks
    the words under both fast and strict numerics.
``production``
    binary32 storage and arithmetic in the same order; CUDA and Metal give
    the same words under strict numerics.

Arrays are in the reference's index order: 3-D fields ``(nx, ny, nz)``
C-contiguous, ``psfc`` ``(nx, ny)``, the control vector WRFDA's
(Fortran-ordered ``v1..v5`` blocks).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = ["DeviceU", "PRECISIONS", "psichi_adj_plan"]

PRECISIONS = ("oracle", "production")
_KERNELS = Path(__file__).with_name("kernels")
#: the translation units: U, U^T and the control-vector operations; H' and H'^T
UNITS = ("var_u.cu", "var_obs.cu")
_THREADS = 256
_MAX_BLOCKS = 65535
#: CUDA keeps binary32 subnormals for this unit (woof.cuda_subnormals), and
#: the Metal translation follows the marker, so both round them the same way.
_MARKER = "#define GPUWM_RETAIN_SUBNORMALS 1\n"
XA_KEYS = ("u", "v", "t", "q", "psfc", "p", "rho")


def _source(precision: str, unit: str = "var_u.cu") -> str:
    if unit not in UNITS:
        raise ValueError(f"unknown woof var kernel unit {unit!r}")
    head = _MARKER + ("#define VAR_F64 1\n" if precision == "oracle" else "")
    return head + (_KERNELS / unit).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _module(precision: str, unit: str = "var_u.cu"):
    import cupy as cp

    from woof import backend

    if backend.active() == "cuda":
        from woof import cuda_subnormals

        cuda_subnormals.install()
    return cp.RawModule(code=_source(precision, unit), options=("-std=c++17",))


def kernel(precision: str, name: str, unit: str = "var_u.cu"):
    return _module(precision, unit).get_function(name)


def blocks(total: int) -> int:
    return max(1, min(_MAX_BLOCKS, (int(total) + _THREADS - 1) // _THREADS))


@lru_cache(maxsize=None)
def psichi_adj_plan(nx: int, ny: int):
    """The gather lists of ``da_psichi_to_uv_adj``'s scatter, in its order.

    Returns ``(row, src, op)`` for psi and for chi (CSR over the target
    points ``i * ny + j``): each target's contributions in the order the
    serial scatter adds them.  ``op`` bit 0 is the coefficient (0 cx,
    1 cy), bit 1 the field (0 u, 1 v), bit 2 the sign (1 subtracts).
    """
    if nx < 4 or ny < 4:
        raise ValueError(f"psi/chi to u/v needs nx, ny >= 4, got {nx}, {ny}")
    ie, je = nx - 1, ny - 1
    psi: list[list] = [[] for _ in range(nx * ny)]
    chi: list[list] = [[] for _ in range(nx * ny)]
    CX, CY, U, V, NEG = 0, 1, 0, 2, 4

    def scatter(i, j, pxp, pxm, cyp, cym, pyp, pym, cxp, cxm):
        s = i * ny + j
        t = lambda p: p[0] * ny + p[1]  # noqa: E731
        psi[t(pxp)].append((s, CX | V))
        psi[t(pxm)].append((s, CX | V | NEG))
        chi[t(cyp)].append((s, CY | V))
        chi[t(cym)].append((s, CY | V | NEG))
        psi[t(pyp)].append((s, CY | U | NEG))
        psi[t(pym)].append((s, CY | U))
        chi[t(cxp)].append((s, CX | U))
        chi[t(cxm)].append((s, CX | U | NEG))

    j = je
    for i in range(ie - 1, 0, -1):
        scatter(i, j, (i + 1, j), (i - 1, j), (i, j), (i, j - 2), (i, j), (i, j - 2), (i + 1, j), (i - 1, j))
    j = 0
    for i in range(ie - 1, 0, -1):
        scatter(i, j, (i + 1, j), (i - 1, j), (i, j + 2), (i, j), (i, j + 2), (i, j), (i + 1, j), (i - 1, j))
    i = ie
    for j in range(je - 1, 0, -1):
        scatter(i, j, (i, j), (i - 2, j), (i, j + 1), (i, j - 1), (i, j + 1), (i, j - 1), (i, j), (i - 2, j))
    i = 0
    for j in range(je - 1, 0, -1):
        scatter(i, j, (i + 2, j), (i, j), (i, j + 1), (i, j - 1), (i, j + 1), (i, j - 1), (i + 2, j), (i, j))
    for j in range(je - 1, 0, -1):
        for i in range(ie - 1, 0, -1):
            scatter(i, j, (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1), (i, j + 1), (i, j - 1),
                    (i + 1, j), (i - 1, j))

    def csr(lists):
        row = np.zeros(len(lists) + 1, dtype=np.int32)
        row[1:] = np.cumsum([len(x) for x in lists])
        flat = [e for x in lists for e in x]
        return row, np.array([e[0] for e in flat], np.int32), np.array([e[1] for e in flat], np.int32)

    return csr(psi), csr(chi)


class DeviceU:
    """``U`` and ``U^T`` for one :class:`~woof.da.var.be_dat.RegionalB` and background.

    ``xb`` holds the reference's keys: grid_box_area, coefx, coefy (nx, ny);
    qs, q, rho, p, t (nx, ny, nz); psac (nx, ny); dnw, c1h, c2h (nz).
    """

    def __init__(self, b, xb: dict, precision: str = "oracle"):
        import cupy as cp

        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, not {precision!r}")
        if b.cv_options not in (5, 7):
            raise NotImplementedError(f"woof var: cv_options = {b.cv_options} transforms are not implemented")
        self.precision = precision
        self.dtype = np.float64 if precision == "oracle" else np.float32
        self.b = b
        self.nx, self.ny = (int(n) for n in np.shape(xb["grid_box_area"]))
        self.nz = int(b.kz)
        self.cv_mz = tuple(int(m) for m in b.cv_mz[:5])
        self.cv_size = self.nx * self.ny * sum(self.cv_mz)
        dev = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=self.dtype))  # noqa: E731
        self.xb = {k: dev(xb[k]) for k in ("coefx", "coefy", "qs", "q", "psac", "dnw", "rho", "p", "t",
                                             "c1h", "c2h")}
        gba = dev(xb["grid_box_area"])
        self.px = cp.empty_like(gba)
        self._k("var_sqrt")((blocks(gba.size),), (_THREADS,), (gba, self.px, np.int64(gba.size)))
        self.v = []
        for s in b.v[:5]:
            if s.mz > 0:
                self.v.append({"mz": int(s.mz), "evec": dev(s.evec), "val": dev(s.val),
                               "alpha": dev(np.asarray(s.rf_alpha)[: s.mz])})
            else:
                self.v.append({"mz": 0})
        self.reg = {k: dev(a) for k, a in b.reg.items() if k in ("psi_t", "psi_ps", "psi_chi")}
        plans = psichi_adj_plan(self.nx, self.ny)
        self.plan = [tuple(cp.asarray(a) for a in p) for p in plans]

    # ------------------------------------------------------------ helpers

    def _k(self, name):
        return kernel(self.precision, name)

    def _launch(self, name, total, *args):
        self._k(name)((blocks(total),), (_THREADS,), args)

    def _i(self, *v):
        return tuple(np.int32(x) for x in v)

    def _zeros(self, *shape):
        import cupy as cp

        return cp.zeros(shape, dtype=self.dtype)

    def _rf_all(self, fields, adjoint):
        """:meth:`_rf` for every variable, each sweep one launch over all their lines."""
        nx, ny = self.nx, self.ny
        npass = self.b.rf_passes // 2
        live = [(f, s) for f, s in zip(fields, self.v) if s["mz"]]
        if not live:
            return
        pad = live + [(live[0][0], {"mz": 0, "alpha": live[0][1]["alpha"]})] * (5 - len(live))
        ptrs = [f for f, _ in pad] + [s["alpha"] for _, s in pad]
        mzs = [s["mz"] for _, s in pad]
        # / sqrt(area) before the first sweep and * after the second (U^T: the reverse)
        first, last = (1, 0) if adjoint else (0, 1)
        for n, axis in enumerate((1, 0) if adjoint else (0, 1)):
            lines = (ny if axis == 0 else nx) * sum(mzs)
            pre, post = (first, -1) if n == 0 else (-1, last)
            self._launch("var_rf_lines5", lines, *ptrs, *self._i(*mzs, nx, ny, axis, npass, 1 if adjoint else 0),
                         self.px, *self._i(pre, post))

    # ------------------------------------------------------------ forward

    def apply(self, cv, stages: bool = False, with_xtoxa: bool = True) -> dict:
        """``x = U v``; returns ``{"xa": {...}}`` and, with ``stages``, vv, vp, vp_bal.

        ``with_xtoxa=False`` stops after ``da_transform_vtox`` (no p, rho).
        """
        import cupy as cp

        nx, ny, nz = self.nx, self.ny, self.nz
        cv = cp.ascontiguousarray(cp.asarray(cv, dtype=self.dtype)).ravel()
        if cv.size != self.cv_size:
            raise ValueError(f"cv has {cv.size} values; nx*ny*sum(cv_mz[:5]) is {self.cv_size}")
        vv, off = {}, 0
        for n, s in enumerate(self.v, start=1):
            mz = s["mz"]
            f = self._zeros(nx, ny, mz)
            if mz:
                size = nx * ny * mz
                block = cp.ascontiguousarray(cv[off: off + size])
                self._launch("var_cv_block", size, block, f, *self._i(nx, ny, mz, 0))
                off += size
            vv[f"v{n}"] = f
        self._rf_all([vv[f"v{n}"] for n in range(1, 6)], adjoint=False)
        out = {"vv": {k: a.copy() for k, a in vv.items()}} if stages else {}
        vp = {}
        for n, s in enumerate(self.v, start=1):
            lev = 1 if n == 5 else nz
            f = self._zeros(nx, ny, lev)
            if s["mz"]:
                e = s["evec"]
                self._launch("var_vvtovp", nx * ny * lev, e, s["val"], vv[f"v{n}"], f,
                             *self._i(nx, ny, lev, s["mz"], e.shape[1], e.shape[2], s["val"].shape[1]))
            vp[f"v{n}"] = f
        if stages:
            out["vp"] = {k: a.copy() for k, a in vp.items()}
        xa = self._vptox(vp)
        if stages:
            out["vp_bal"] = vp
        if with_xtoxa:
            self._xtoxa(xa)
        out["xa"] = xa
        return out

    def _vptox(self, vp) -> dict:
        nx, ny, nz = self.nx, self.ny, self.nz
        n3 = nx * ny * nz
        xa = {}
        if self.b.cv_options != 7:
            self._launch("var_reg_psi_t", n3, vp["v3"], self.reg["psi_t"], vp["v1"], *self._i(nx, ny, nz, 0))
        xa["t"] = vp["v3"].copy()
        if self.b.cv_options != 7:
            self._launch("var_reg_psi_ps", nx * ny, vp["v5"], self.reg["psi_ps"], vp["v1"], *self._i(nx, ny, nz))
        xa["psfc"] = vp["v5"].reshape(nx, ny).copy()
        if self.b.cv_options != 7:
            self._launch("var_add_reg_prod", n3, vp["v2"], self.reg["psi_chi"], vp["v1"], *self._i(nx, ny, nz, 0))
            u, v = self._zeros(nx, ny, nz), self._zeros(nx, ny, nz)
            self._launch("var_psichi_to_uv", n3, vp["v1"], vp["v2"], self.xb["coefx"], self.xb["coefy"], u, v,
                         *self._i(nx, ny, nz))
            self._launch("var_uv_corners", nz, u, v, *self._i(nx, ny, nz))
            xa["u"], xa["v"] = u, v
        else:
            xa["u"], xa["v"] = vp["v1"].copy(), vp["v2"].copy()
        q = self._zeros(nx, ny, nz)
        self._launch("var_mul", n3, q, vp["v4"], self.xb["qs"], np.int64(n3))
        xa["q"] = q
        return xa

    def _xtoxa(self, xa):
        nx, ny, nz = self.nx, self.ny, self.nz
        xa["p"], xa["rho"] = self._zeros(nx, ny, nz), self._zeros(nx, ny, nz)
        x = self.xb
        self._launch("var_xtoxa", nx * ny, xa["q"], xa["t"], xa["psfc"], x["q"], x["psac"], x["dnw"], x["c1h"],
                     x["c2h"], x["rho"], x["p"], x["t"], xa["p"], xa["rho"], *self._i(nx, ny, nz))

    # ------------------------------------------------------------ adjoint

    def adjoint(self, xa: dict, stages: bool = False, with_xtoxa: bool = True) -> dict:
        """``v = U^T x``; returns ``{"cv": ...}`` and, with ``stages``, xa, vp, vv_eof, vv.

        ``with_xtoxa=False`` starts at ``da_transform_vtox_adj`` (``xa`` is
        already through ``da_transform_xtoxa_adj``).
        """
        import cupy as cp

        nx, ny, nz = self.nx, self.ny, self.nz
        n3 = nx * ny * nz
        a = {k: cp.array(xa[k], dtype=self.dtype, copy=True, order="C") for k in XA_KEYS}
        a["psfc"] = cp.ascontiguousarray(a["psfc"].reshape(nx, ny))
        if with_xtoxa:
            self._xtoxa_adj(a)
        out = {"xa": {k: v.copy() for k, v in a.items()}} if stages else {}
        vp = self._vptox_adj(a)
        if stages:
            out["vp"] = {k: v.copy() for k, v in vp.items()}
        vv = {}
        for n, s in enumerate(self.v, start=1):
            mz = s["mz"]
            f = self._zeros(nx, ny, mz)
            if mz:
                lev = 1 if n == 5 else nz
                e = s["evec"]
                self._launch("var_vvtovp_adj", nx * ny * mz, e, s["val"], vp[f"v{n}"], f,
                             *self._i(nx, ny, lev, mz, e.shape[1], e.shape[2], s["val"].shape[1]))
            vv[f"v{n}"] = f
        if stages:
            out["vv_eof"] = {k: v.copy() for k, v in vv.items()}
        parts = []
        self._rf_all([vv[f"v{n}"] for n in range(1, 6)], adjoint=True)
        for n, s in enumerate(self.v, start=1):
            mz = s["mz"]
            if not mz:
                continue
            f = vv[f"v{n}"]
            size = nx * ny * mz
            block = self._zeros(size)
            self._launch("var_cv_block", size, block, f, *self._i(nx, ny, mz, 1))
            parts.append(block)
        if stages:
            out["vv"] = vv
        out["cv"] = cp.concatenate(parts) if parts else self._zeros(0)
        return out

    def _xtoxa_adj(self, a):
        nx, ny, nz = self.nx, self.ny, self.nz
        x = self.xb
        mra = self._zeros(nx, ny, nz)
        self._launch("var_xtoxa_adj", nx * ny, a["p"], a["t"], a["rho"], a["psfc"], a["q"], mra, x["q"],
                     x["psac"], x["dnw"], x["c1h"], x["c2h"], x["rho"], x["p"], x["t"], *self._i(nx, ny, nz))

    def _vptox_adj(self, a) -> dict:
        nx, ny, nz = self.nx, self.ny, self.nz
        n3 = nx * ny * nz
        vp = {f"v{n}": self._zeros(nx, ny, nz) for n in range(1, 5)}
        vp["v5"] = self._zeros(nx, ny, 1)
        self._launch("var_add_prod", n3, vp["v4"], a["q"], self.xb["qs"], np.int64(n3))
        if self.b.cv_options != 7:
            u, v = a["u"].copy(), a["v"].copy()
            self._launch("var_uv_corners_adj", nz, u, v, *self._i(nx, ny, nz))
            for target, (row, src, op) in zip((vp["v1"], vp["v2"]), self.plan):
                self._launch("var_gather_adj", n3, target, row, src, op, self.xb["coefx"], self.xb["coefy"],
                             u, v, *self._i(nx * ny, nz))
        else:
            self._launch("var_add", n3, vp["v1"], a["u"], np.int64(n3))
            self._launch("var_add", n3, vp["v2"], a["v"], np.int64(n3))
        if self.b.cv_options != 7:
            self._launch("var_add_reg_prod", n3, vp["v1"], self.reg["psi_ps"], a["psfc"], *self._i(nx, ny, nz, 1))
        self._launch("var_add", nx * ny, vp["v5"], a["psfc"], np.int64(nx * ny))
        if self.b.cv_options != 7:
            self._launch("var_reg_psi_t", n3, vp["v1"], self.reg["psi_t"], a["t"], *self._i(nx, ny, nz, 1))
        self._launch("var_add", n3, vp["v3"], a["t"], np.int64(n3))
        if self.b.cv_options != 7:
            self._launch("var_add_reg_prod", n3, vp["v1"], self.reg["psi_chi"], vp["v2"], *self._i(nx, ny, nz, 0))
        return vp
