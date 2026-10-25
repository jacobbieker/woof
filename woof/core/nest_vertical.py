"""WRF vertical grid nesting: a live child on its own eta ladder.

WRF v4.7.1's ``vert_refine_method = 2`` lets a nest carry its own
``e_vert``/``eta_levels`` under the parent's ``p_top``, ``hybrid_opt`` and
``etac``.  Everything the parent hands down to such a child passes through
two operators, both ported here, both held word for word to WRF's own
Fortran (``tools/vert_nest_wrf471_oracle``,
``tests/test_nest_vertical_wrf471_parity.py``):

* **the vertical interpolation** (``share/interp_fcn.F``
  ``vert_interp_vert_nesting``/``_w`` through ``inter_wrf_copy`` and
  ``coeff_mon_wrf_copy``): every parent column, onto the child's levels,
  by a monotone cubic Hermite in a reference log-pressure height
  (:func:`reference_heights`, ``force_domain_em_part2.F:73-109``).  Half
  levels carry a surface and a top point extrapolated with the parent's
  ``cf1..cfn1``; full levels do not.
* **the rebalance** (``force_domain_em_part2.F:139-300``): after the
  interpolation, ``t_2`` and QV are uncoupled with the child's column mass,
  the child's base state is rebuilt on the parent's columns, the
  perturbation pressure and inverse density are integrated down from the
  top, ``ph_2`` is rebuilt hydrostatically, and the three are coupled again
  with the child's coefficients.

WRF interpolates vertically on the parent's columns FIRST and horizontally
second (the intermediate grid has the parent's spacing and the child's
levels), so a coupler runs these before its SINT.  Birth
(``interp_domain_em_part2``) runs the interpolation alone, on the
uncoupled fields; forcing (``force_domain_em_part2``) runs both, on the
coupled fields.

Two implementations, one result: :class:`VerticalNestPlan` launches the
CUDA kernels (``kernels/nest_vertical.cu``, compiled -fmad=false
--ftz=false), and its ``*_cpu`` methods are the binary32 NumPy twin
(glibc's logf/powf through :mod:`woof.core.noahmp_libm`), word for word
the same.  Per-column.  The forcing path's working slabs are the parent's
donor columns at the child's level count (a few per forced kind, plus
four for the rebalance), taken from the CuPy pool each parent step:
about (nz_child + 1) x donor columns x 4 B each, small beside the arena.

The base-state constants are WRF's real-data defaults (``p00`` 1e5 Pa,
``tlp`` 50 K, ``tiso`` 200 K, no stratospheric lapse) with the run's
``base_temp``, the constants woof's own real initialization uses
(``woof/ingest/real.py``).

Declared divergence: where WRF's segment search finds no interval for a
child point (only possible if the two reference columns did not share
their end points), WRF leaves the value undefined; the plan refuses such
a pair of ladders by name instead (:func:`_segments`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

F = np.float32

#: module_model_constants.F, folded in real(4) as gfortran folds them.
G = F(9.81)
R_D = F(287.0)
CP = F(F(F(7.0) * R_D) / F(2.0))
R_V = F(461.6)
CV = F(CP - R_D)
CVPM = F(-F(CV / CP))
P1000MB = F(100000.0)
T0 = F(300.0)
RVOVRD = F(R_V / R_D)
RCP = F(R_D / CP)
RDP = F(R_D / P1000MB)

#: force_domain_em_part2.F:82-84, the reference column.
HSCA_M = F(6.7)
P_SURF_M = F(1.0e5)

#: WRF's real-data base state defaults (Registry.EM_COMMON: base_pres,
#: base_lapse, iso_temp, base_pres_strat, base_lapse_strat); ``t00`` is
#: the run's ``base_temp``.
P00 = F(100000.0)
TLP = F(50.0)
TISO = F(200.0)
P_STRAT = F(0.0)
TLP_STRAT = F(-11.0)

MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "woof.core.nest_vertical:nest_vertical"
_SOURCES = ("glibc_flt32.cuh", "nest_vertical.cu")
_BLOCK = 128


class VerticalNestRefused(ValueError):
    """A parent/child ladder pair WRF's vertical nesting cannot carry."""


def _f32(value) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value), dtype=F)


def _logf(x) -> np.ndarray:
    from woof.core.noahmp_libm import logf_array

    return logf_array(np.asarray(x, dtype=F))


def _powf(x, y) -> np.ndarray:
    from woof.core.noahmp_libm import powf_array

    return powf_array(np.asarray(x, dtype=F), np.asarray(y, dtype=F))


def reference_heights(*, p_top, parent_c3f, parent_c4f, parent_c3h,
                      parent_c4h, child_c3f, child_c4f, child_c3h,
                      child_c4h):
    """``force_domain_em_part2.F:82-107``: the reference log-pressure
    heights of both ladders, ``(alt_w_c, alt_u_c, alt_w_n, alt_u_n)``.

    One column, ``p_surf = 1e5`` Pa: ``alt = -6.7*log(p/p_surf)`` with
    ``p = mu*c3 + p_top + c4``.  Half-level heights carry the full-level
    surface and top beside them (``alt_u(1) = alt_w(1)``,
    ``alt_u(kde+1) = alt_w(kde)``).  ``p_top`` is the CHILD's (WRF reads
    ``ngrid%p_top``), shared with the parent by construction.
    """

    p_top_m = F(p_top)
    mu_m = F(P_SURF_M - p_top_m)

    def heights(c3, c4):
        c3, c4 = _f32(c3), _f32(c4)
        pre = (F(mu_m) * c3 + p_top_m) + c4
        return F(-HSCA_M) * _logf(pre / P_SURF_M)

    alt_w_c = heights(parent_c3f, parent_c4f)
    alt_w_n = heights(child_c3f, child_c4f)
    alt_u_c = np.concatenate([alt_w_c[:1], heights(parent_c3h, parent_c4h),
                              alt_w_c[-1:]]).astype(F)
    alt_u_n = np.concatenate([alt_w_n[:1], heights(child_c3h, child_c4h),
                              alt_w_n[-1:]]).astype(F)
    return alt_w_c, alt_u_c, alt_w_n, alt_u_n


def _segments(alt_c: np.ndarray, alt_n: np.ndarray) -> np.ndarray:
    """``inter_wrf_copy``'s search (:2088-2099): for every fine point but
    the last, the coarse interval ``j`` with ``alt_c[j] <= x < alt_c[j+1]``
    (the FIRST such, as WRF's ``goto`` takes it)."""

    seg = np.full(alt_n.size, -1, dtype=np.int32)
    for i in range(alt_n.size - 1):
        x = alt_n[i]
        for j in range(alt_c.size - 1):
            if x >= alt_c[j] and x < alt_c[j + 1]:
                seg[i] = j
                break
        if seg[i] < 0:
            raise VerticalNestRefused(
                f"child reference height {float(x)!r} at point {i} lies in "
                "no parent interval; the two ladders do not share their end "
                "points (p_top, hybrid_opt and etac must match), and WRF "
                "would leave this level undefined")
    return seg


@dataclass(frozen=True, eq=False)
class VerticalNestPlan:
    """Both ladders' constants for one parent->child edge, on the host.

    Built once per edge (:meth:`from_coefficients` or
    :meth:`from_states`); the per-column work is :meth:`interp` /
    :meth:`rebalance` on the device and their ``_cpu`` twins.
    """

    nz_parent: int
    nz_child: int
    alt_w_c: np.ndarray
    alt_u_c: np.ndarray
    alt_w_n: np.ndarray
    alt_u_n: np.ndarray
    seg_mass: np.ndarray
    seg_w: np.ndarray
    parent_cf: tuple
    child: dict
    p_top: np.float32
    base: dict
    hypsometric_opt: int
    use_theta_m: int = 0
    _device: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_coefficients(cls, *, parent: dict, child: dict, p_top,
                          hypsometric_opt: int, base_temp,
                          use_theta_m: int = 0, base: dict | None = None):
        """``parent``/``child`` carry WRF's arrays, 0-based as woof stores
        them: ``c1f..c4f`` (nz+1), ``c1h..c4h``, ``dnw``, ``rdnw``, ``rdn``
        (nz); the parent also ``cf`` = (cf1, cf2, cf3, cfn, cfn1)."""

        nzp = int(np.asarray(parent["c3h"]).size)
        nzn = int(np.asarray(child["c3h"]).size)
        if nzp < 3:
            raise VerticalNestRefused(
                "a parent needs at least three mass levels: the surface "
                "point extrapolates from the lowest three (cf1..cf3)")
        alt_w_c, alt_u_c, alt_w_n, alt_u_n = reference_heights(
            p_top=p_top,
            parent_c3f=parent["c3f"], parent_c4f=parent["c4f"],
            parent_c3h=parent["c3h"], parent_c4h=parent["c4h"],
            child_c3f=child["c3f"], child_c4f=child["c4f"],
            child_c3h=child["c3h"], child_c4h=child["c4h"])
        names = ("c1h", "c2h", "c3h", "c4h", "c1f", "c2f", "c3f", "c4f",
                 "dnw", "rdnw", "rdn")
        values = {name: _f32(child[name]) for name in names}
        for name in names:
            want = nzn + 1 if name.endswith("f") else nzn
            if values[name].size != want:
                raise VerticalNestRefused(
                    f"child {name} has {values[name].size} values, expected "
                    f"{want}")
        merged = {"p00": P00, "tlp": TLP, "tiso": TISO, "p_strat": P_STRAT,
                  "tlp_strat": TLP_STRAT, "t00": F(base_temp)}
        merged.update({k: F(v) for k, v in (base or {}).items()})
        if int(hypsometric_opt) not in (1, 2):
            raise VerticalNestRefused(
                f"hypsometric_opt {hypsometric_opt!r}: WRF's rebalance "
                "takes 1 or 2")
        return cls(
            nz_parent=nzp, nz_child=nzn, alt_w_c=alt_w_c, alt_u_c=alt_u_c,
            alt_w_n=alt_w_n, alt_u_n=alt_u_n,
            seg_mass=_segments(alt_u_c, alt_u_n),
            seg_w=_segments(alt_w_c, alt_w_n),
            parent_cf=tuple(F(v) for v in parent["cf"]),
            child=values, p_top=F(p_top), base=merged,
            hypsometric_opt=int(hypsometric_opt),
            use_theta_m=int(use_theta_m))

    @classmethod
    def from_states(cls, parent_state, child_state, child_cfg):
        """The plan for a live parent/child pair (their loaded bases)."""

        def host(value):
            try:
                import cupy as cp
                if isinstance(value, cp.ndarray):
                    return cp.asnumpy(value)
            except ImportError:
                pass
            return np.asarray(value)

        def ladder(state, *, with_cf):
            out = {name: host(getattr(state, name)) for name in (
                "c1h", "c2h", "c3h", "c4h", "c1f", "c2f", "c3f", "c4f",
                "dnw", "rdnw", "rdn")}
            if with_cf:
                out["cf"] = tuple(F(getattr(state, name)) for name in (
                    "cf1", "cf2", "cf3", "cfn", "cfn1"))
            return out

        if float(parent_state.p_top) != float(child_state.p_top):
            raise VerticalNestRefused(
                f"vertical nesting needs one model top: parent p_top "
                f"{float(parent_state.p_top)!r}, child "
                f"{float(child_state.p_top)!r}")
        return cls.from_coefficients(
            parent=ladder(parent_state, with_cf=True),
            child=ladder(child_state, with_cf=False),
            p_top=child_state.p_top,
            hypsometric_opt=int(getattr(child_cfg, "hypsometric_opt", 2)),
            base_temp=float(getattr(child_cfg, "base_temp", 290.0)))

    # -- shapes ---------------------------------------------------------------

    def levels_out(self, nlev_in: int) -> int:
        """Child levels for a parent field of ``nlev_in`` levels."""

        if nlev_in == self.nz_parent:
            return self.nz_child
        if nlev_in == self.nz_parent + 1:
            return self.nz_child + 1
        raise VerticalNestRefused(
            f"a parent field of {nlev_in} levels is neither half "
            f"({self.nz_parent}) nor full ({self.nz_parent + 1})")

    # -- CPU twin -------------------------------------------------------------

    def _pro_cpu(self, src: np.ndarray, wstag: bool) -> np.ndarray:
        if wstag:
            return src
        cf1, cf2, cf3, cfn, cfn1 = self.parent_cf
        bottom = (cf1 * src[0] + cf2 * src[1]) + cf3 * src[2]
        top = cfn * src[-1] + cfn1 * src[-2]
        return np.concatenate([bottom[None], src, top[None]]).astype(F)

    def interp_cpu(self, field) -> np.ndarray:
        """The binary32 twin of ``nest_vert_interp``: ``field`` is
        ``(nlev, ...)`` on the parent's half (nz) or full (nz+1) levels."""

        src = _f32(field)
        shape = src.shape
        nlev = shape[0]
        ndst = self.levels_out(nlev)
        wstag = nlev == self.nz_parent + 1
        src = src.reshape(nlev, -1)
        y = self._pro_cpu(src, wstag)
        x = self.alt_w_c if wstag else self.alt_u_c
        xn = self.alt_w_n if wstag else self.alt_u_n
        seg = self.seg_w if wstag else self.seg_mass
        n = x.size
        h = (x[1:] - x[:-1]).astype(F)
        s = ((y[1:] - y[:-1]) / h[:, None]).astype(F)
        yp = np.empty_like(y)
        yp[0] = s[0]
        yp[n - 1] = s[n - 2]
        hm, hp = h[:-1, None], h[1:, None]
        sm, sp = s[:-1], s[1:]
        p = ((sm * hp + sp * hm) / (hm + hp)).astype(F)
        sg = (np.copysign(F(1.0), sm) + np.copysign(F(1.0), sp)).astype(F)
        mn = np.minimum(np.minimum(np.abs(sm), np.abs(sp)),
                        (F(0.5) * np.abs(p)).astype(F))
        yp[1:n - 1] = sg * mn
        out = np.empty((ndst, src.shape[1]), dtype=F)
        n_n = xn.size
        for k in range(ndst):
            i = k if wstag else k + 1
            if i == n_n - 1:
                out[k] = y[n - 1]
                continue
            j = int(seg[i])
            hj = h[j]
            sj = s[j]
            a = ((yp[j] + yp[j + 1]) - F(2.0) * sj) / (hj * hj)
            b = ((F(3.0) * sj - F(2.0) * yp[j]) - yp[j + 1]) / hj
            pj = F(xn[i] - x[j])
            out[k] = pj * (pj * (a * pj + b) + yp[j]) + y[j]
        return out.reshape((ndst,) + shape[1:])

    def rebalance_cpu(self, t, qv, ph, mub, mu, ht, *, scratch=None):
        """The binary32 twin of ``nest_vert_rebalance``.  ``t``/``qv``
        ``(nz, ...)`` and ``ph`` ``(nz+1, ...)`` are rebalanced in place
        (as float32 copies, returned); ``scratch``, when a dict, receives
        ``pb``, ``alb``, ``phb`` and ``al``."""

        c = self.child
        b = self.base
        nz = self.nz_child
        t = _f32(t).copy()
        qv = _f32(qv).copy()
        ph = _f32(ph).copy()
        shape_h, shape_f = t.shape, ph.shape
        t = t.reshape(nz, -1)
        qv = qv.reshape(nz, -1)
        ph = ph.reshape(nz + 1, -1)
        MUB = _f32(mub).reshape(-1)
        MU2 = _f32(mu).reshape(-1)
        HT = _f32(ht).reshape(-1)
        p_top = self.p_top

        def col_mass(k, which):
            c1, c2 = (c["c1h"], c["c2h"]) if which == "h" else (c["c1f"],
                                                                 c["c2f"])
            return (c1[k] * MUB + c2[k]) + (c1[k] * MU2)

        for k in range(nz):
            m = col_mass(k, "h")
            t[k] = t[k] / m
            qv[k] = qv[k] / m
        pb = np.empty_like(t)
        alb = np.empty_like(t)
        for k in range(nz):
            pbk = (c["c3h"][k] * MUB + c["c4h"][k]) + p_top
            pb[k] = pbk
            temp = np.maximum(b["tiso"],
                              b["t00"] + b["tlp"] * _logf(pbk / b["p00"]))
            low = pbk < b["p_strat"]
            if low.any():
                strat = b["tiso"] + b["tlp_strat"] * _logf(
                    pbk / (b["p_strat"] if b["p_strat"] != 0 else F(1.0)))
                temp = np.where(low, strat, temp)
            t_init = temp * _powf(b["p00"] / pbk, RCP) - T0
            alb[k] = (RDP * (t_init + T0)) * _powf(pbk / P1000MB, CVPM)
        phb = np.empty_like(ph)
        phb[0] = HT * G
        if self.hypsometric_opt == 1:
            for kk in range(1, nz + 1):
                k = kk - 1
                w = c["dnw"][k] * (c["c1h"][k] * MUB + c["c2h"][k])
                phb[kk] = phb[k] - w * alb[k]
        else:
            for k in range(1, nz + 1):
                pfu = (c["c3f"][k] * MUB + c["c4f"][k]) + p_top
                pfd = (c["c3f"][k - 1] * MUB + c["c4f"][k - 1]) + p_top
                phm = (c["c3h"][k - 1] * MUB + c["c4h"][k - 1]) + p_top
                phb[k] = phb[k - 1] + (alb[k - 1] * phm) * _logf(pfd / pfu)
        al = np.empty_like(t)

        def inverse_density(kk, pk):
            qvf = (F(1.0) + RVOVRD * qv[kk]) if self.use_theta_m == 0 \
                else np.ones_like(qv[kk])
            tk = (RDP * (t[kk] + T0)) * qvf
            return tk * _powf((pk + pb[kk]) / P1000MB, CVPM) - alb[kk]

        kk = nz - 1
        k = kk + 1
        qvf1 = F(0.5) * (qv[kk] + qv[kk])
        qvf2 = F(1.0) / (F(1.0) + qvf1)
        qvf1 = qvf1 * qvf2
        a = c["c1f"][k] * MU2
        bb = qvf1 * (c["c1f"][k] * MUB + c["c2f"][k])
        pk = -(((F(0.5) * (a + bb)) / c["rdnw"][kk]) / qvf2)
        al[kk] = inverse_density(kk, pk)
        pnext = pk
        for kk in range(nz - 2, -1, -1):
            k = kk + 1
            qvf1 = F(0.5) * (qv[kk] + qv[kk + 1])
            qvf2 = F(1.0) / (F(1.0) + qvf1)
            qvf1 = qvf1 * qvf2
            a = c["c1f"][k] * MU2
            bb = qvf1 * (c["c1f"][k] * MUB + c["c2f"][k])
            pk = pnext - (((a + bb) / qvf2) / c["rdn"][kk + 1])
            al[kk] = inverse_density(kk, pk)
            pnext = pk
        if self.hypsometric_opt == 1:
            for kk in range(1, nz + 1):
                k = kk - 1
                m = col_mass(k, "h")
                first = m * al[kk - 1]
                second = (c["c1h"][k] * MU2) * alb[kk - 1]
                ph[kk] = ph[kk - 1] - c["dnw"][kk - 1] * (first + second)
        else:
            ph[0] = phb[0]
            mt = MUB + MU2
            for k in range(1, nz + 1):
                pfu = (c["c3f"][k] * mt + c["c4f"][k]) + p_top
                pfd = (c["c3f"][k - 1] * mt + c["c4f"][k - 1]) + p_top
                phm = (c["c3h"][k - 1] * mt + c["c4h"][k - 1]) + p_top
                ph[k] = ph[k - 1] + ((alb[k - 1] + al[k - 1]) * phm) \
                    * _logf(pfd / pfu)
            for k in range(nz + 1):
                ph[k] = ph[k] - phb[k]
        for k in range(nz + 1):
            ph[k] = ph[k] * col_mass(k, "f")
        for k in range(nz):
            m = col_mass(k, "h")
            t[k] = t[k] * m
            qv[k] = qv[k] * m
        if scratch is not None:
            scratch.update({"pb": pb.reshape(shape_h),
                            "alb": alb.reshape(shape_h),
                            "phb": phb.reshape(shape_f),
                            "al": al.reshape(shape_h)})
        return t.reshape(shape_h), qv.reshape(shape_h), ph.reshape(shape_f)

    # -- device ---------------------------------------------------------------

    def _dev(self, name: str):
        import cupy as cp

        key = (cp.cuda.runtime.getDevice(), name)
        if key not in self._device:
            if name.startswith("seg_") or name.startswith("alt_"):
                value = getattr(self, name)
            else:
                value = self.child[name]
            self._device[key] = cp.asarray(value)
        return self._device[key]

    def interp(self, field, *, out=None):
        """``nest_vert_interp`` on a device field ``(nlev, ...)``; returns
        ``(nlev_child, ...)`` (into ``out`` when given)."""

        import cupy as cp

        src = cp.ascontiguousarray(field, dtype=cp.float32)
        nlev = int(src.shape[0])
        ndst = self.levels_out(nlev)
        wstag = nlev == self.nz_parent + 1
        ncol = int(src.size // nlev)
        shape = (ndst,) + tuple(src.shape[1:])
        if out is None:
            out = cp.empty(shape, dtype=cp.float32)
        elif tuple(out.shape) != shape or out.dtype != cp.float32 \
                or not out.flags.c_contiguous:
            raise ValueError(f"interp out must be a contiguous float32 "
                             f"{shape} array")
        alt_c = self._dev("alt_w_c" if wstag else "alt_u_c")
        alt_n = self._dev("alt_w_n" if wstag else "alt_u_n")
        seg = self._dev("seg_w" if wstag else "seg_mass")
        cf1, cf2, cf3, cfn, cfn1 = self.parent_cf
        if ncol:
            _kernel("nest_vert_interp")(
                ((ncol + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
                (src, out, alt_c, alt_n, seg, np.int32(nlev),
                 np.int32(ndst), np.int32(ncol), np.int32(1 if wstag else 0),
                 cf1, cf2, cf3, cfn, cfn1))
        return out

    def rebalance(self, t, qv, ph, mub, mu, ht, *, scratch=None):
        """``nest_vert_rebalance`` in place on device ``t``/``qv``
        ``(nz, ...)`` and ``ph`` ``(nz+1, ...)`` (contiguous float32)."""

        import cupy as cp

        nz = self.nz_child
        for name, array, levels in (("t", t, nz), ("qv", qv, nz),
                                    ("ph", ph, nz + 1)):
            if (array.dtype != cp.float32 or not array.flags.c_contiguous
                    or int(array.shape[0]) != levels):
                raise ValueError(f"rebalance {name} must be a contiguous "
                                 f"float32 array of {levels} levels")
        ncol = int(t.size // nz)
        mub = cp.ascontiguousarray(mub, dtype=cp.float32)
        mu = cp.ascontiguousarray(mu, dtype=cp.float32)
        ht = cp.ascontiguousarray(ht, dtype=cp.float32)
        if any(int(a.size) != ncol for a in (mub, mu, ht)):
            raise ValueError("rebalance mub/mu/ht must hold one value per "
                             "column")
        work = scratch if scratch is not None else {}
        for name, levels in (("pb", nz), ("alb", nz), ("phb", nz + 1),
                             ("al", nz)):
            if name not in work:
                work[name] = cp.empty((levels,) + tuple(t.shape[1:]),
                                      dtype=cp.float32)
        b = self.base
        c = self.child
        if ncol:
            _kernel("nest_vert_rebalance")(
                ((ncol + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
                (t, qv, ph, work["pb"], work["alb"], work["phb"], work["al"],
                 mub, mu, ht,
                 self._dev("c1h"), self._dev("c2h"), self._dev("c3h"),
                 self._dev("c4h"), self._dev("c1f"), self._dev("c2f"),
                 self._dev("c3f"), self._dev("c4f"), self._dev("dnw"),
                 self._dev("rdnw"), self._dev("rdn"),
                 np.int32(nz), np.int32(ncol),
                 np.int32(self.hypsometric_opt), np.int32(self.use_theta_m),
                 np.int32(0), self.p_top, b["tiso"], b["t00"], b["tlp"],
                 b["p00"], b["p_strat"], b["tlp_strat"], G, T0, P1000MB,
                 RCP, RDP, CVPM, RVOVRD))
        return t, qv, ph


def ladders_differ(parent_state, child_state) -> bool:
    """Whether two live states sit on different eta ladders."""

    def host(value):
        try:
            import cupy as cp
            if isinstance(value, cp.ndarray):
                return cp.asnumpy(value)
        except ImportError:
            pass
        return np.asarray(value)

    a, b = host(parent_state.znw), host(child_state.znw)
    return a.shape != b.shape or not np.array_equal(a, b)


def interp_any(plan: VerticalNestPlan, field):
    """:meth:`VerticalNestPlan.interp` on a device array, its CPU twin on a
    host one; the same words either way."""

    if isinstance(field, np.ndarray):
        return plan.interp_cpu(field)
    return plan.interp(field)


#: The parent fields a nest's initialization reads through the vertical
#: interpolation (woof.ingest.nest_init.parent_only_init and its base):
#: every prognostic and scalar on the parent's half or full levels, the
#: dynamics' forcing pair and the base state.  Anything else (2-D fields,
#: category arrays, the coordinate itself) is the parent's own.
VERTICAL_PARENT_FIELDS = frozenset({
    "u", "v", "w", "thp", "php", "qv", "qc", "qr", "qi", "qs", "qg",
    "nc", "nr", "ni", "ns", "ng", "nwfa", "nifa", "qh", "qndrop", "qnr",
    "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh", "qir", "qib",
    "h_diabatic", "rthften", "rqvften", "pb", "alb", "thb", "phb",
})


class VerticallyNestedParent:
    """A parent state, read through the vertical interpolation onto a
    child's ladder: WRF's intermediate grid (the parent's columns, the
    child's levels) for an initialization from the parent
    (``interp_domain_em_part2``).

    The fields of :data:`VERTICAL_PARENT_FIELDS` come back interpolated,
    once (cached), whether 3-D or a flat parent's 1-D column; everything
    else is the parent's own.  Theta is interpolated as WRF holds it,
    ``t_2 = theta - 300``: ``thp`` is then ``(t_2 + 300) - thb``, beside
    ``thb`` interpolated (in the parent's own dimensionality), so the two
    add back to the interpolated total theta.
    """

    def __init__(self, parent, plan: VerticalNestPlan):
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_plan", plan)
        object.__setattr__(self, "_cache", {})

    @staticmethod
    def _xp(value):
        if isinstance(value, np.ndarray):
            return np
        import cupy as cp
        return cp

    def _interp(self, value):
        """Interpolate a 3-D field, or a 1-D column as one column."""
        if getattr(value, "ndim", 0) == 1:
            xp = self._xp(value)
            column = xp.ascontiguousarray(value[:, None], dtype=xp.float32)
            return interp_any(self._plan, column)[:, 0]
        xp = self._xp(value)
        return interp_any(self._plan, xp.ascontiguousarray(value,
                                                           dtype=xp.float32))

    def _theta(self):
        cache = self._cache
        if "thp" not in cache:
            parent = self._parent
            thp = parent.thp
            xp = self._xp(thp)
            thb = parent.thb
            thb3 = thb if thb.ndim == 3 else xp.broadcast_to(
                thb[:, None, None], thp.shape)
            t300 = (thp + thb3) - xp.float32(300.0)
            t_child = self._interp(xp.ascontiguousarray(t300))
            thb_child = self._interp(thb)
            thb_child3 = thb_child if thb_child.ndim == 3 else \
                thb_child[:, None, None]
            cache["thb"] = thb_child
            cache["thp"] = (t_child + xp.float32(300.0)) - thb_child3
        return cache["thp"], cache["thb"]

    def __getattr__(self, name):
        if name in ("thp", "thb"):
            thp, thb = self._theta()
            return thp if name == "thp" else thb
        value = getattr(self._parent, name)
        if name not in VERTICAL_PARENT_FIELDS or value is None:
            return value
        shape = getattr(value, "shape", None)
        levels = (self._plan.nz_parent, self._plan.nz_parent + 1)
        if shape is None or len(shape) not in (1, 3) \
                or int(shape[0]) not in levels:
            return value
        cache = self._cache
        if name not in cache:
            cache[name] = self._interp(value)
        return cache[name]

    def __setattr__(self, name, value):
        raise AttributeError("the vertically nested parent view is "
                             "read-only")


def module_source() -> str:
    """The exact source string NVRTC compiles."""
    from woof.core.kernels import _preamble

    root = Path(__file__).parent / "kernels"
    return _preamble() + "".join(
        (root / name).read_text(encoding="utf-8") for name in _SOURCES)


@lru_cache(maxsize=None)
def _module(device: int):
    """NVRTC directly, -fmad=false and --ftz=false, one module per device
    (not ``cupy.RawModule``, which appends ``-ftz=true``; see
    ``woof.core.terrain_drag._module``)."""
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler

    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile

    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(
            source, MODULE_OPTIONS, None, "nest_vertical.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS,
                  module=None)
    return module


def _kernel(name: str):
    import cupy as cp

    return _module(cp.cuda.runtime.getDevice()).get_function(name)


__all__ = ["VerticalNestPlan", "VerticalNestRefused",
           "VerticallyNestedParent", "interp_any",
           "ladders_differ", "module_source", "reference_heights"]
