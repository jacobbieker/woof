"""The 3D-Var cost function and its gradient in WRFDA's operation order
(docs/dev/WRFDA-3DVAR-PLAN.md, sections 1.2, 4.6 and 4.7; plan item V1.4).

``J(v) = Jb + Jo`` with ``Jb = jb_factor * 0.5 * v^T v`` (cv5/cv7, no
ensemble, VarBC or other terms) and ``Jo = 0.5 * sum((d - H' U v) / err)^2``
over the accepted observations, and the gradient ``v - U^T H'^T R^-1 (d - H'
U v)``.  The routines follow WRFDA 4.7.1's statement by statement:

- :meth:`VarCost.calculate_j` is ``da_calculate_j``: ``y = H' U xhat`` when
  ``iter > 0`` (``da_transform_vtoy``), ``re = inv - y`` where ``qc >= 0``
  else 0 (``da_calculate_residual``/``da_residual``), Jo by
  ``da_jo_and_grady`` (per type, per variable ``jo = jo - re * g`` with
  ``g = -re / (error * error)`` over reports then levels, halved, the
  variables summed in the type's order and the types in ``jo%total``'s
  order), and ``Jb = jb_factor * 0.5 * da_dot_cv(cv + xhat, cv + xhat)``.
- :meth:`VarCost.calculate_gradj` is ``da_calculate_gradj``: with a residual
  (``present(re)``) ``grad = vtoy_adj(-re / err^2) + jb_factor * cv``;
  without one it is the Hessian-vector product ``grad = -vtoy_adj(-H' U cv
  / err^2) + jb_factor * cv``, ``da_calculate_grady`` masking ``y`` first.

Three tiers:

``reference``
    NumPy binary64 on the host: :mod:`woof.da.var.reference`'s ``U`` and
    ``U^T``, serial (``da_dot``) inner products.
``oracle``
    :class:`~woof.da.var.transforms.DeviceU` in binary64 and the
    ``wrfda-serial`` reduction: the reference's words on CUDA and Metal.
``production``
    ``DeviceU`` in binary32 and the ``fixed-tree`` reduction.

Observation space (``m`` values, tens of thousands) is small: the residual,
``g`` and Jo are computed on the host in the tier's real type (NumPy
element-wise operations round exactly as the scalar Fortran statements), Jo
accumulated in binary64 left to right.  That makes them independent of the
backend by construction; only ``H'`` and ``H'^T`` themselves (the
:class:`ObsTerm`) run where the term puts them.

Derived from WRFDA 4.7.1 (``var/da/da_minimisation``: ``da_calculate_j``,
``da_calculate_gradj``, ``da_calculate_residual``, ``da_calculate_grady``,
``da_jo_and_grady``, ``da_transform_vtoy(_adj)``, ``da_dot``, ``da_dot_cv``;
``var/da/da_test/da_check_gradient``), public domain; see
``licenses/LICENSE-WRF-public-domain.txt``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from woof.da.var import reference as ref

__all__ = [
    "JO_ORDER",
    "TIERS",
    "JCost",
    "ObsTerm",
    "ReferenceU",
    "StencilAdapter",
    "SparseObsTerm",
    "VarCost",
    "as_obs_term",
    "check_gradient",
]

TIERS = ("reference", "oracle", "production")
XA_KEYS = ("u", "v", "t", "q", "psfc", "p", "rho")

#: ``jo%total``'s summation order (da_jo_and_grady.inc); a type not listed
#: here is summed after these, in the order the terms give it.
JO_ORDER = ("sound", "sonde_sfc", "geoamv", "polaramv", "synop", "satem", "pilot", "airep", "metar", "ships",
            "gpspw", "ssmi_tb", "ssmi_rv", "ssmt1", "ssmt2", "pseudo", "qscat", "profiler", "buoy", "radar",
            "gpsref", "gpseph", "bogus", "radiance", "airsr", "mtgirs", "tamdar", "tamdar_sfc", "rain",
            "lightning")


# --------------------------------------------------------------- protocol


@runtime_checkable
class ObsTerm(Protocol):
    """The observation side of the cost function, one flat vector of ``m`` values.

    ``d``, ``err`` and ``mask`` are host arrays of length ``m``: the innovation
    (``iv%..%inv``), the observation error and ``qc >= obs_qc_pointer``.

    ``tl(x) -> y``
        ``H' dx`` for ``x`` a dict of the analysis-increment fields
        (:data:`XA_KEYS`, the reference's ``(nx, ny, nz)`` layout, ``psfc``
        ``(nx, ny)``), host or device arrays of the tier's real type; ``y``
        length ``m`` (host or device).
    ``ad(y) -> x``
        ``H'^T dy`` into zeroed fields (``da_zero_x`` then the
        ``da_transform_xtoy_<type>_adj`` scatters in their order); missing
        keys are zero.  ``y`` arrives as the tier's array type.
    ``jo_split() -> [(type, [(variable, index), ...]), ...]``
        the Jo groups: per observation type its variables in
        ``da_jo_and_grady_<type>``'s order, each with the flat indices of
        ``y`` that type's Jo sums over, in its loop order (reports, then
        levels; ``proc_domain`` reports only).

    Optional control block (VarBC): a term with ``control_size > 0`` owns
    that many control variables after ``U``'s (``cv = [v_U, v_1, ...]`` in
    term order).  Its ``tl`` then also reads ``x["varbc"]`` (host
    binary64), its ``ad`` also returns ``"varbc"``, and ``jp(v)`` and
    ``jp_grad(v)`` give its background term and gradient; ``Jb`` stays over
    ``v_U``.  Without such a term nothing changes, word for word.
    """

    d: np.ndarray
    err: np.ndarray
    mask: np.ndarray

    def tl(self, x: dict): ...

    def ad(self, y) -> dict: ...

    def jo_split(self) -> list: ...


class StencilAdapter:
    """Adapter point for an operator object with ``tl``/``ad`` and per-observation
    ``inv``, ``error`` and ``qc`` (the V1.2 ``ObsStencil``).

    ``mask`` is ``qc >= 0``; ``jo_split`` is the object's own when it has one,
    otherwise one group (``kind``, ``"y"``) over every observation.
    """

    def __init__(self, op, kind: str = "obs"):
        self.op = op
        self.kind = kind
        self.d = np.asarray(op.inv, dtype=np.float64)
        self.err = np.asarray(op.error, dtype=np.float64)
        self.mask = np.asarray(op.qc) >= 0

    def tl(self, x):
        return self.op.tl(x)

    def ad(self, y):
        return self.op.ad(y)

    def jo_split(self):
        if hasattr(self.op, "jo_split"):
            return self.op.jo_split()
        return [(self.kind, [("y", np.arange(self.d.size))])]


def as_obs_term(obj) -> ObsTerm:
    """``obj`` itself when it has the :class:`ObsTerm` attributes, else a :class:`StencilAdapter`."""
    if all(hasattr(obj, a) for a in ("d", "err", "mask", "tl", "ad", "jo_split")):
        return obj
    if all(hasattr(obj, a) for a in ("inv", "error", "qc", "tl", "ad")):
        return StencilAdapter(obj)
    raise TypeError(f"{type(obj).__name__} is not an ObsTerm (d, err, mask, tl, ad, jo_split)")


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _on_device(term, tier: str):
    """A :class:`~woof.da.var.obsop_wrfda.FlatObsTerm` as its device form (the same words), else ``term``."""
    from woof.da.var.obsop_wrfda import FlatObsTerm

    if isinstance(term, FlatObsTerm) and term.term.xtoxa is None:
        from woof.da.var.obsop_device import DeviceObsTerm

        return DeviceObsTerm(term, tier)
    return term


class SparseObsTerm:
    """A linear ``H'``: observation ``i`` is ``sum_k w[i, k] * x[field[i, k]].flat[index[i, k]]``.

    ``tl`` sums the ``k`` terms left to right; ``ad`` scatters ``w * y`` into
    zeroed fields observation by observation, ``k`` in order.  Arithmetic is
    on the host in the input's real type, so the words are the same whatever
    the backend.  It is the synthetic term of the tests and a generic stencil
    (horizontal corners times vertical pair) for linear operators.
    """

    def __init__(self, shapes: dict, field_: np.ndarray, index: np.ndarray, weight: np.ndarray, d, err,
                 qc=None, groups=None):
        self.shapes = {k: tuple(v) for k, v in shapes.items()}
        self.field = np.asarray(field_)
        self.index = np.asarray(index, dtype=np.int64)
        self.weight = np.asarray(weight, dtype=np.float64)
        self.d = np.asarray(d, dtype=np.float64)
        self.err = np.asarray(err, dtype=np.float64)
        self.mask = np.ones(self.d.size, dtype=bool) if qc is None else np.asarray(qc) >= 0
        self.groups = groups or [("obs", [("y", np.arange(self.d.size))])]

    def jo_split(self):
        return self.groups

    def tl(self, x):
        h = {k: _host(v).ravel() for k, v in x.items()}
        dt = next(iter(h.values())).dtype
        w = self.weight.astype(dt)
        y = np.zeros(self.d.size, dtype=dt)
        for k in range(self.index.shape[1]):
            col = np.empty(self.d.size, dtype=dt)
            for name in np.unique(self.field[:, k]):
                sel = self.field[:, k] == name
                col[sel] = h[str(name)][self.index[sel, k]]
            y = y + w[:, k] * col
        return y

    def ad(self, y):
        y = _host(y)
        dt = y.dtype
        out = {k: np.zeros(int(np.prod(s)), dtype=dt) for k, s in self.shapes.items()}
        prod = (self.weight.astype(dt) * y[:, None]).ravel()  # w * y, one rounding each
        fields, index = self.field.ravel(), self.index.ravel()  # (i, k) in scatter order
        for name, f in out.items():
            sel = fields == name
            np.add.at(f, index[sel], prod[sel])  # unbuffered, in order: f[j] = f[j] + w y
        return {k: v.reshape(self.shapes[k]) for k, v in out.items()}


# --------------------------------------------------------------- U adapters


class ReferenceU:
    """:mod:`woof.da.var.reference`'s ``U`` and ``U^T`` with :class:`DeviceU`'s interface."""

    precision = "reference"
    dtype = np.float64

    def __init__(self, b, xb: dict):
        self.b, self.xb = b, xb
        self.nx, self.ny = (int(n) for n in np.shape(xb["grid_box_area"]))
        self.nz = int(b.kz)
        self.cv_mz = tuple(int(m) for m in b.cv_mz[:5])
        self.cv_size = self.nx * self.ny * sum(self.cv_mz)

    def apply(self, cv) -> dict:
        return {"xa": ref.transform_u(np.asarray(cv, dtype=np.float64), self.b, self.xb)["xa"]}

    def adjoint(self, xa: dict) -> dict:
        return {"cv": ref.transform_u_adj(xa, self.b, self.xb)["cv"]}


# --------------------------------------------------------------- vector ops


class _HostOps:
    """Binary64 host mirrors of the kernels, the serial dot of ``da_dot``."""

    def __init__(self):
        self.dtype = np.float64

    def asarray(self, a):
        return np.ascontiguousarray(_host(a), dtype=np.float64)

    def zeros(self, n):
        return np.zeros(n, dtype=np.float64)

    def copy(self, x):
        return np.array(x, dtype=np.float64, copy=True)

    def dot(self, x, y) -> float:
        p = np.asarray(x, dtype=np.float64).ravel() * np.asarray(y, dtype=np.float64).ravel()
        return float(np.add.accumulate(p)[-1]) if p.size else 0.0

    def axpy(self, a, x, y):
        y[...] = y + np.float64(a) * x

    def dot_device(self, x, y) -> float:
        return self.dot(x, y)

    def axpy_neg(self, s, x, y):
        self.axpy(-s, x, y)

    def div(self, x, a):
        return x / np.float64(a)

    def mul(self, a, b):
        return a * b

    def neg_axpby(self, p, t, b):
        p[...] = -t + np.float64(b) * p

    def add_into(self, out, a):
        out[...] = out + a


class _DeviceOps:
    """The same operations as ``kernels/var_u.cu`` launches."""

    def __init__(self, precision: str, reduction: str):
        self.precision = precision
        self.reduction = reduction
        self.dtype = np.float64 if precision == "oracle" else np.float32

    def _k(self, name):
        from woof.da.var.transforms import kernel

        return kernel(self.precision, name)

    def _go(self, name, n, *args):
        from woof.da.var.transforms import blocks

        self._k(name)((blocks(n),), (256,), args)

    def asarray(self, a):
        import cupy as cp

        return cp.ascontiguousarray(cp.asarray(a, dtype=self.dtype)).ravel()

    def zeros(self, n):
        import cupy as cp

        return cp.zeros(n, dtype=self.dtype)

    def copy(self, x):
        return x.copy()

    def dot(self, x, y) -> float:
        from woof.da.var.control import dot

        return dot(x, y, self.reduction)

    def axpy(self, a, x, y):
        from woof.da.var.control import axpy

        axpy(a, x, y)

    def dot_device(self, x, y):
        """The dot left on the device (no host sync), for :meth:`axpy_neg`."""
        from woof.da.var.control import dot_device

        return dot_device(x, y, self.reduction)

    def axpy_neg(self, s, x, y):
        """``y = y + (-s) x`` with ``s`` a :meth:`dot_device` value."""
        from woof.da.var.control import axpy_neg_device

        axpy_neg_device(s, x, y)

    def div(self, x, a):
        out = self.zeros(x.size)
        self._go("var_div_scalar", x.size, out, x, self.dtype(a), np.int64(x.size))
        return out

    def mul(self, a, b):
        out = self.zeros(a.size)
        self._go("var_mul", a.size, out, a, b, np.int64(a.size))
        return out

    def neg_axpby(self, p, t, b):
        self._go("var_neg_axpby", p.size, p, t, self.dtype(b), np.int64(p.size))

    def add_into(self, out, a):
        self._go("var_add", out.size, out, a, np.int64(out.size))


# --------------------------------------------------------------- the cost


@dataclass
class JCost:
    """``j_type``: the total and the parts this port has (Jb, Jo and Jo per type)."""

    total: float
    jb: float
    jo: float
    jo_types: dict = field(default_factory=dict)
    #: VarBC's background term (0 without a control block).
    jp: float = 0.0


class VarCost:
    """``J`` and ``grad J`` over control space for one ``U`` and a list of :class:`ObsTerm`.

    Build with :meth:`reference` (NumPy binary64) or :meth:`device` (``oracle``
    or ``production``).  Vectors are host arrays for the reference tier and
    device arrays otherwise; :attr:`ops` holds the vector operations
    (``dot`` in the tier's reduction) that :mod:`woof.da.var.minimise` uses.
    """

    def __init__(self, u, obs, *, tier: str = "reference", jb_factor: float = 1.0, reduction: str | None = None,
                 device_obs: bool = True):
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, not {tier!r}")
        self.u = u
        self.tier = tier
        self.jb_factor = float(jb_factor)
        terms = [obs] if not isinstance(obs, (list, tuple)) else list(obs)
        self.terms = [as_obs_term(t) for t in terms]
        if tier != "reference" and device_obs:
            self.terms = [_on_device(t, tier) for t in self.terms]
        if tier == "reference":
            self.ops = _HostOps()
        else:
            if reduction is None:
                reduction = "wrfda-serial" if tier == "oracle" else "fixed-tree"
            self.ops = _DeviceOps(tier, reduction)
        self.dtype = self.ops.dtype
        self.cv_size = int(u.cv_size)
        # terms with a block of their own in the control vector (VarBC), after U's
        self.cv_u = self.cv_size
        self._blocks = []
        for n, t in enumerate(self.terms):
            size = int(getattr(t, "control_size", 0) or 0)
            if size:
                self._blocks.append((n, self.cv_size, size))
                self.cv_size += size
        self.nx, self.ny, self.nz = int(u.nx), int(u.ny), int(u.nz)
        self.set_innovations([t.d for t in self.terms])
        self._groups = self._jo_groups()
        self.re = None

    @classmethod
    def reference(cls, b, xb, obs, **kw) -> "VarCost":
        return cls(ReferenceU(b, xb), obs, tier="reference", **kw)

    @classmethod
    def device(cls, b, xb, obs, precision: str = "oracle", **kw) -> "VarCost":
        from woof.da.var.transforms import DeviceU

        return cls(DeviceU(b, xb, precision), obs, tier=precision, **kw)

    # ------------------------------------------------------------ setup

    def set_innovations(self, ds: Sequence) -> None:
        """Replace ``iv%..%inv`` (a new outer loop's innovations), one array per term."""
        if len(ds) != len(self.terms):
            raise ValueError(f"{len(ds)} innovation vectors for {len(self.terms)} terms")
        self.d = [np.asarray(d, dtype=np.float64).astype(self.dtype) for d in ds]
        self.err = [np.asarray(t.err, dtype=np.float64).astype(self.dtype) for t in self.terms]
        self.mask = [np.asarray(t.mask, dtype=bool) for t in self.terms]
        self._dev_grady = None
        if self.tier != "reference" and self.terms and all(hasattr(t, "grady") for t in self.terms):
            import cupy as cp

            self._dev_grady = [(cp.asarray(e), cp.asarray(m.astype(np.int32))) for e, m in zip(self.err, self.mask)]

    def _jo_groups(self):
        groups = []
        for n, t in enumerate(self.terms):
            for kind, variables in t.jo_split():
                groups.append((kind, n, [(v, np.asarray(i, dtype=np.int64)) for v, i in variables]))
        rank = {k: i for i, k in enumerate(JO_ORDER)}
        order = sorted(range(len(groups)), key=lambda g: (rank.get(groups[g][0], len(JO_ORDER)), g))
        return [groups[g] for g in order]

    def zeros(self):
        return self.ops.zeros(self.cv_size)

    def asarray(self, v):
        return self.ops.asarray(v)

    # ------------------------------------------------------------ transforms

    def _fields(self, xa: dict) -> dict:
        out = {}
        for k in XA_KEYS:
            shape = (self.nx, self.ny) if k == "psfc" else (self.nx, self.ny, self.nz)
            a = xa.get(k)
            if a is None:
                a = np.zeros(shape, dtype=self.dtype) if self.tier == "reference" else None
                if a is None:
                    import cupy as cp

                    a = cp.zeros(shape, dtype=self.dtype)
            elif self.tier != "reference":
                import cupy as cp

                a = cp.ascontiguousarray(cp.asarray(a, dtype=self.dtype).reshape(shape))
            else:
                a = np.ascontiguousarray(_host(a), dtype=np.float64).reshape(shape)
            out[k] = a
        return out

    def vtoy(self, cv) -> list:
        """``da_transform_vtoy``: ``y = H' U cv`` per term, host arrays of the tier's type."""
        if not self._blocks:
            xa = self.u.apply(cv)["xa"]
            return [np.asarray(_host(t.tl(xa)), dtype=self.dtype).ravel() for t in self.terms]
        xa = self.u.apply(cv[:self.cv_u])["xa"]
        block = {n: np.asarray(_host(cv[o:o + k]), dtype=np.float64) for n, o, k in self._blocks}
        return [np.asarray(_host(t.tl(dict(xa, varbc=block[n]) if n in block else xa)), dtype=self.dtype).ravel()
                for n, t in enumerate(self.terms)]

    def vtoy_adj(self, gs: Sequence):
        """``da_transform_vtoy_adj``: ``U^T H'^T g``, the terms' scatters summed in list order."""
        xa = None
        blocks = {}
        for n, (t, g) in enumerate(zip(self.terms, gs)):
            gd = g if self.tier == "reference" or not isinstance(g, np.ndarray) else self._to_device(g)
            ad = t.ad(gd)
            if self._blocks and "varbc" in ad:
                blocks[n] = np.asarray(_host(ad["varbc"]), dtype=np.float64)
            part = self._fields(ad)
            if xa is None:
                xa = part
            else:
                for k in XA_KEYS:
                    self.ops.add_into(xa[k].reshape(-1), part[k].reshape(-1))
        if not self._blocks:
            return self.ops.asarray(self.u.adjoint(xa)["cv"])
        cv_u = _host(self.u.adjoint(xa)["cv"])
        parts = [blocks.get(n, np.zeros(k)) for n, _, k in self._blocks]
        return self.ops.asarray(np.concatenate([np.asarray(cv_u, dtype=self.dtype)]
                                               + [p.astype(self.dtype) for p in parts]))

    def _jp(self, cvx) -> float:
        """``j % jp``: the terms' background terms over their blocks of ``cv + xhat``, in term order."""
        jp = 0.0
        for n, o, k in self._blocks:
            jp = jp + self.terms[n].jp(np.asarray(_host(cvx[o:o + k]), dtype=np.float64))
        return jp

    def _set_block_gradient(self, grad, adj, cv, sign: float):
        """Replace ``jb_factor * cv`` in each block of ``grad`` by ``grad Jp``: ``sign * adj + grad_jp(cv)``."""
        if not self._blocks:
            return grad
        host = np.array(_host(grad), dtype=self.dtype)
        for n, o, k in self._blocks:
            a = np.asarray(_host(adj[o:o + k]), dtype=np.float64)
            g = self.terms[n].jp_grad(np.asarray(_host(cv[o:o + k]), dtype=np.float64))
            host[o:o + k] = (sign * a + g).astype(self.dtype)
        return self.ops.asarray(host)

    def _to_device(self, a):
        import cupy as cp

        return cp.asarray(np.ascontiguousarray(a, dtype=self.dtype))

    # ------------------------------------------------------------ obs space

    def residual(self, ys) -> list:
        """``da_calculate_residual``: ``re = inv - y`` where ``qc >= 0``, else 0."""
        zero = self.dtype(0.0)
        return [np.where(m, d - y, zero) for d, y, m in zip(self.d, ys, self.mask)]

    def grady(self, res) -> list:
        """``da_calculate_grady``: ``re = 0`` where rejected, ``g = -re / (error * error)``."""
        zero = self.dtype(0.0)
        out = []
        for r, e, m in zip(res, self.err, self.mask):
            r = np.where(m, r, zero)
            out.append(-r / (e * e))
        return out

    def jo(self, res) -> tuple[float, dict]:
        """``da_jo_and_grady``: Jo per type and ``jo%total``, in WRFDA's order."""
        gs = []
        for r, e in zip(res, self.err):
            gs.append(-r / (e * e))  # da_jo_and_grady_<type>: no qc test, re is already 0 there
        per_type: dict = {}
        total = None
        for kind, n, variables in self._groups:
            r, g = res[n].astype(np.float64), gs[n].astype(np.float64)
            per = {}
            for v, idx in variables:
                p = r[idx] * g[idx]
                acc = float(np.add.accumulate(-p)[-1]) if p.size else 0.0
                per[v] = 0.5 * acc
            tot = None
            for v, _ in variables:
                tot = per[v] if tot is None else tot + per[v]
            per["_"] = 0.0 if tot is None else tot
            per_type[kind] = per
            total = per["_"] if total is None else total + per["_"]
        return (0.0 if total is None else total), per_type

    # ------------------------------------------------------------ J and grad J

    def calculate_j(self, xhat, cv=None, iter_: int = 1, y=None) -> JCost:
        """``da_calculate_j``; keeps the residual in :attr:`re` for :meth:`calculate_gradj`.

        ``iter_ > 0`` evaluates ``y = H' U xhat``; ``iter_ == 0`` uses ``y``
        (WRFDA's freshly allocated, zero ``y`` when ``None``).
        """
        if iter_ > 0:
            ys = self.vtoy(xhat)
        elif y is None:
            ys = [np.zeros(d.size, dtype=self.dtype) for d in self.d]
        else:
            ys = [np.asarray(a, dtype=self.dtype) for a in y]
        self.re = self.residual(ys)
        jo, per_type = self.jo(self.re)
        cvx = self.ops.copy(xhat)
        if cv is not None:
            self.ops.axpy(1.0, cv, cvx)  # cv + xhat: 1.0 * cv is exact
        if not self._blocks:
            jb = self.jb_factor * 0.5 * self.ops.dot(cvx, cvx)
            return JCost(total=jb + jo, jb=jb, jo=jo, jo_types=per_type)
        u = cvx[:self.cv_u]
        jb = self.jb_factor * 0.5 * self.ops.dot(u, u)
        jp = self._jp(cvx)
        return JCost(total=jb + jo + jp, jb=jb, jo=jo, jo_types=per_type, jp=jp)

    def calculate_gradj(self, cv, re=None):
        """``da_calculate_gradj``: with ``re`` the gradient, without it ``(I + U^T H'^T R^-1 H' U) cv``."""
        if re is not None:
            adj = self.vtoy_adj(self.grady(re))
            grad = self.ops.copy(adj)
            self.ops.axpy(self.jb_factor, cv, grad)  # grad_jo + jb_factor * cv
            return self._set_block_gradient(grad, adj, cv, 1.0)
        if self._dev_grady is not None:
            # H' U cv, grady and H'^T without leaving the device (the host's words)
            xa = self.u.apply(cv)["xa"]
            gs = [t.grady(t.tl(xa), e, m) for t, (e, m) in zip(self.terms, self._dev_grady)]
            adj = self.vtoy_adj(gs)
        else:
            ys = self.vtoy(cv)
            adj = self.vtoy_adj(self.grady(ys))
        grad = self.ops.copy(cv)
        self.ops.neg_axpby(grad, adj, self.jb_factor)  # -grad_jo + jb_factor * cv
        return self._set_block_gradient(grad, adj, cv, -1.0)

    # ------------------------------------------------------------ conveniences

    def value_and_gradient(self, v):
        """``J(v)`` and ``grad J(v)`` with ``cv = 0`` (as ``da_check_gradient`` calls them)."""
        j = self.calculate_j(v, None, iter_=1)
        return j, self.calculate_gradj(v, re=self.re)


def check_gradient(cost: VarCost, xhat, pdx: float = 1.0e-10, itertest: int = 8) -> dict:
    """``da_check_gradient``: the Taylor test along ``-grad J``.

    ``xhat`` is the test point (WRFDA draws ``random_number - 0.5``); returns
    the per-``a`` values (``alpha``, ``F(a)``, ``T1 = (F(a) - F(0)) / (a
    DF(0))``) and ``best``, ``ibest`` and ``digits`` as WRFDA's conclusion
    computes them.
    """
    ops = cost.ops
    xhat = ops.asarray(xhat)
    j, grad = cost.value_and_gradient(ops.copy(xhat))
    zf0 = j.total
    xdir = ops.zeros(cost.cv_size)
    ops.axpy(-1.0, grad, xdir)  # xdir = -1.0 * grad
    zdf0 = ops.dot(grad, xdir)
    rows = []
    for jj in range(1, itertest + 1):
        za = pdx * float(10 ** (jj - 1))
        yhat = ops.copy(xhat)
        ops.axpy(za, xdir, yhat)
        ja, grada = cost.value_and_gradient(yhat)
        zfa = ja.total
        zdfa = ops.dot(grada, xdir)
        zt1 = (zfa - zf0) / (za * zdf0)
        rows.append({"alpha": za, "f": zfa, "df": zdfa, "t1": zt1})
    zermin, ibest = math.inf, -1
    for n, r in enumerate(rows):
        t = abs((r["f"] - zf0) / (r["alpha"] * zdf0) - 1.0)
        if t < zermin:
            ibest, zermin = n, t
    digits = int(-math.log(zermin + np.finfo(np.float64).tiny) / math.log(10.0))
    return {"f0": zf0, "df0": zdf0, "rows": rows, "ibest": ibest, "best": rows[ibest]["t1"],
            "residual": zermin, "digits": digits}
