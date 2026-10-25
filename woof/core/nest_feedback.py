"""Two-way feedback from a nest on its own eta ladder, below a height.

WRF v4.7.1 has no such operator: ``share/module_check_a_mundo.F:2719-2737``
stops a run that asks for feedback with ``vert_refine_method /= 0``
("vert_refine_method=2 only works with feedback = 0").  So this is woof's
own, built around the horizontal step WRF does have, and held to its own
NumPy twin rather than to an oracle:

1. **Child ladder -> parent ladder, per child column**
   (:func:`remap_layers`, kernel ``nest_fb_remap``).  Both ladders are
   placed in the CHILD column's own dry-pressure coordinate
   ``pd = c3f * MU + c4f`` (``MU = mub + mup``; the shared ``p_top``
   cancels, the bottom edge is ``MU`` and the top 0 on both).  A
   mass-point or face field becomes the mass-coupled content of each
   parent layer, ``Q_k = sum_j overlap(j, k) * v_j`` in Pa x units: every
   child layer's mass is split exactly across the parent layers it
   overlaps, so the column's content is conserved to rounding.  ``w`` and
   ``ph`` live on interfaces and are interpolated linearly in ``pd``.
2. **The horizontal restriction** is WRF's ``copy_fcn``, unchanged, on
   those parent-ladder child columns, then a division by a layer mass.
   ``t``, ``u`` and ``v`` keep the child's mass-weighted MEAN: divided by
   the child columns' own masses in the parent's layers (mode 2 of the
   same kernel), restricted alike, so a constant comes back as itself
   whatever the two grids' terrain.  Moisture, number concentrations and
   other scalars keep the child's CONTENT: divided by the parent's own
   layer mass, so the parent's water in each layer over the restriction
   rectangle is the child's, to rounding.  ``ph`` is carried as the total
   geopotential, interpolated in ``pd``, restricted, and re-expressed
   against the parent's own ``phb``.
3. **The parent smoother and the clamp** run on a copy of the parent field,
   exactly as WRF's feedback runs them on the parent.
4. **The height limit** (:func:`level_weights`, ``nest_fb_blend``): the
   copy is blended into the parent level by level, ``p + w_k (r - p)``.
   ``w = 1`` on the parent layers wholly below ``feedback_top_m`` (above
   the parent's ground, on its base geopotential, in every column the
   edge writes), falling linearly to zero over the top
   ``feedback_taper_levels`` of them, and 0 above, where the parent's
   words are not touched.  Interfaces take the mean of the two layers
   they separate.  ``ph`` is the exception (:func:`ph_column`,
   ``nest_fb_ph_column``): geopotential is a vertical integral, so the
   column is rebuilt upward from the ground on total geopotential, the
   fed layers taking the child's thickness and the untouched ones their
   own.  The parent keeps its ``mu`` under a limit (below), so above the
   limit every layer keeps its mass and inverse density, hence its
   pressure, potential temperature and moisture; its geopotential moves
   only by what the column below it now weighs.

``mu`` is a column quantity: with the whole column fed back it is fed
back whole, as WRF feeds it, and each layer's dry mass is then the child's
wherever the parent's ``mub`` is the mean of the child's under it.  With a
limit below the model top the parent keeps its own ``mu``: the column above
keeps the parent's winds, and the child's column mass written under them
adds or removes mass aloft that no flux carried (measured on a three-tier
real-data run: the parent's ``w`` noise above the limit grew 4-10x).  The
parent's mass then follows its own continuity equation, driven by the
child's winds below the limit.

On an edge inside one ladder step 1 is skipped (the identity), so a nest
on the parent's own levels with no height limit is today's feedback word
for word, and a limit above the model top (every weight 1) is too.
"""

from __future__ import annotations

import numpy as np

from woof.core.nest_vertical import _BLOCK, _kernel

F = np.float32
D = np.float64

#: woof/core/nest_vertical.py's g, the base geopotential's.
_G = 9.81

#: Fields that live on interfaces (nz + 1 levels): interpolated, not
#: integrated, and never uncoupled.
INTERFACE_KINDS = frozenset({"w", "ph"})


def _is_host(value) -> bool:
    return isinstance(value, np.ndarray)


def _ladder(state):
    return np.asarray(_host(state.c3f), dtype=F), \
        np.asarray(_host(state.c4f), dtype=F)


def _host(value):
    if _is_host(value):
        return value
    import cupy as cp
    return cp.asnumpy(value)


# -- CPU twins ---------------------------------------------------------------


def _mu_cpu(mub, mup, nyf, nxf, xstag, ystag):
    m = np.asarray(mub, dtype=F).astype(D) + np.asarray(mup, dtype=F).astype(D)
    ny, nx = m.shape
    if xstag:
        il = np.clip(np.arange(nxf) - 1, 0, nx - 1)
        ir = np.clip(np.arange(nxf), 0, nx - 1)
        a, b = m[:, il], m[:, ir]
        return np.where(il == ir, a, 0.5 * (a + b))
    if ystag:
        jl = np.clip(np.arange(nyf) - 1, 0, ny - 1)
        jr = np.clip(np.arange(nyf), 0, ny - 1)
        a, b = m[jl, :], m[jr, :]
        return np.where((jl == jr)[:, None], a, 0.5 * (a + b))
    return m


def _edge_cpu(c3, c4, k, nz, mu):
    if k <= 0:
        return mu
    if k >= nz:
        return np.zeros_like(mu)
    a = D(c3[k]) * mu
    return a + D(c4[k])


def remap_layers_cpu(src, mub, mup, child_ladder, parent_ladder, *,
                     xstag=0, ystag=0, interface=False, mass=False):
    """The binary64 twin of ``nest_fb_remap`` (see the module docstring).

    ``src`` is ``(nzc, nyf, nxf)`` (``nzc + 1`` for an interface field) on
    the child's ladder; returns ``(nzp, nyf, nxf)`` (``nzp + 1``).  With
    ``mass`` the column's mass in each parent layer is returned instead
    (``src`` gives only the shape)."""

    c3c, c4c = child_ladder
    c3p, c4p = parent_ladder
    nzc, nzp = c3c.size - 1, c3p.size - 1
    src = np.asarray(src, dtype=F)
    nyf, nxf = src.shape[1:]
    mu = _mu_cpu(mub, mup, nyf, nxf, xstag, ystag)
    if mass:
        out = np.empty((nzp, nyf, nxf), dtype=F)
        for k in range(nzp):
            out[k] = (_edge_cpu(c3p, c4p, k, nzp, mu)
                      - _edge_cpu(c3p, c4p, k + 1, nzp, mu)).astype(F)
        return out
    if not interface:
        if src.shape[0] != nzc:
            raise ValueError(f"a layer field has {nzc} levels, got "
                             f"{src.shape[0]}")
        out = np.empty((nzp, nyf, nxf), dtype=F)
        ce = [_edge_cpu(c3c, c4c, j, nzc, mu) for j in range(nzc + 1)]
        for k in range(nzp):
            bot = _edge_cpu(c3p, c4p, k, nzp, mu)
            top = _edge_cpu(c3p, c4p, k + 1, nzp, mu)
            acc = np.zeros_like(mu)
            for j in range(nzc):
                ov = np.minimum(bot, ce[j]) - np.maximum(top, ce[j + 1])
                term = ov * src[j].astype(D)
                acc = acc + np.where(ov > 0.0, term, 0.0)
            out[k] = acc.astype(F)
        return out
    if src.shape[0] != nzc + 1:
        raise ValueError(f"an interface field has {nzc + 1} levels, got "
                         f"{src.shape[0]}")
    out = np.empty((nzp + 1, nyf, nxf), dtype=F)
    ce = np.stack([_edge_cpu(c3c, c4c, j, nzc, mu) for j in range(nzc + 1)])
    for k in range(nzp + 1):
        x = _edge_cpu(c3p, c4p, k, nzp, mu)
        j = (ce[1:nzc] >= x).sum(axis=0) if nzc > 1 else \
            np.zeros(x.shape, dtype=np.int64)
        j = j[None]
        lo = np.take_along_axis(ce, j, 0)[0]
        hi = np.take_along_axis(ce, j + 1, 0)[0]
        f = (lo - x) / (lo - hi)
        a = np.take_along_axis(src, j, 0)[0].astype(D)
        b = np.take_along_axis(src, j + 1, 0)[0].astype(D)
        d = b - a
        s = f * d
        out[k] = (a + s).astype(F)
    return out


def blend_cpu(parent, fed, weights, window):
    """The twin of ``nest_fb_blend``, in place on ``parent``."""

    i0, j0, ni, nj = window
    if ni <= 0 or nj <= 0:
        return parent
    for k, wk in enumerate(np.asarray(weights, dtype=F)):
        p = parent[k, j0:j0 + nj, i0:i0 + ni]
        r = fed[k, j0:j0 + nj, i0:i0 + ni]
        if wk == F(1.0):
            p[...] = r
        elif wk != F(0.0):
            d = (r - p).astype(F)
            s = (wk * d).astype(F)
            p[...] = (p + s).astype(F)
    return parent


# -- device ------------------------------------------------------------------


def _blocks(n: int):
    return ((int(n) + _BLOCK - 1) // _BLOCK,)


def _dev_ladder(ladder):
    import cupy as cp
    return tuple(cp.asarray(v, dtype=cp.float32) for v in ladder)


def remap_layers(src, mub, mup, child_ladder, parent_ladder, *, xstag=0,
                 ystag=0, interface=False, mass=False, out=None):
    """``nest_fb_remap`` on device arrays; :func:`remap_layers_cpu` on host
    ones.  ``mub``/``mup`` are the child's mass-point ``(ny, nx)``."""

    if _is_host(src):
        return remap_layers_cpu(src, mub, mup, child_ladder, parent_ladder,
                                xstag=xstag, ystag=ystag,
                                interface=interface, mass=mass)
    import cupy as cp

    c3c, c4c = _dev_ladder(child_ladder)
    c3p, c4p = _dev_ladder(parent_ladder)
    nzc, nzp = int(c3c.size) - 1, int(c3p.size) - 1
    src = cp.ascontiguousarray(src, dtype=cp.float32)
    interface = interface and not mass
    want_in = nzc + 1 if interface else nzc
    if int(src.shape[0]) != want_in:
        raise ValueError(f"remap source has {int(src.shape[0])} levels, "
                         f"expected {want_in}")
    nyf, nxf = int(src.shape[1]), int(src.shape[2])
    nlev = nzp + 1 if interface else nzp
    if out is None:
        out = cp.empty((nlev, nyf, nxf), dtype=cp.float32)
    mub = cp.ascontiguousarray(mub, dtype=cp.float32)
    mup = cp.ascontiguousarray(mup, dtype=cp.float32)
    ny, nx = int(mub.shape[0]), int(mub.shape[1])
    ncol = nyf * nxf
    if ncol:
        _kernel("nest_fb_remap")(
            _blocks(ncol), (_BLOCK,),
            (src, out, mub, mup, c3c, c4c, c3p, c4p, np.int32(nzc),
             np.int32(nzp), np.int32(ny), np.int32(nx), np.int32(nyf),
             np.int32(nxf), np.int32(xstag), np.int32(ystag),
             np.int32(2 if mass else 1 if interface else 0)))
    return out


def blend(parent, fed, weights, window):
    """``nest_fb_blend`` in place on ``parent`` (or its twin)."""

    if _is_host(parent):
        return blend_cpu(parent, fed, weights, window)
    import cupy as cp

    i0, j0, ni, nj = (int(v) for v in window)
    if ni <= 0 or nj <= 0:
        return parent
    for name, a in (("parent", parent), ("fed", fed)):
        if not a.flags.c_contiguous or a.dtype != cp.float32:
            raise ValueError(f"blend {name} must be contiguous float32")
    if parent.shape != fed.shape:
        raise ValueError("blend operands differ in shape")
    w = cp.asarray(weights, dtype=cp.float32)
    nlev = int(parent.shape[0])
    if int(w.size) != nlev:
        raise ValueError(f"{int(w.size)} weights for {nlev} levels")
    _kernel("nest_fb_blend")(
        _blocks(nlev * ni * nj), (_BLOCK,),
        (parent, fed, w, np.int32(nlev), np.int32(parent.shape[1]),
         np.int32(parent.shape[2]), np.int32(i0), np.int32(j0),
         np.int32(ni), np.int32(nj)))
    return parent


def ph_column_cpu(ph, fed, phb, w_layer, w_iface, window):
    """The twin of ``nest_fb_ph_column``, in place on ``ph``.  ``fed`` is
    read only where a layer weight is nonzero (the levels the transaction
    restricted)."""

    i0, j0, ni, nj = window
    if ni <= 0 or nj <= 0:
        return ph
    nz = ph.shape[0] - 1
    sl = (slice(j0, j0 + nj), slice(i0, i0 + ni))
    phb = np.asarray(phb, dtype=F)

    def base(k):
        return D(phb[k]) if phb.ndim == 1 else phb[(k,) + sl].astype(D)

    wl = np.asarray(w_layer, dtype=F)
    wi = np.asarray(w_iface, dtype=F)
    p0 = ph[(0,) + sl].copy()
    r0 = fed[(0,) + sl]
    if wi[0] == F(1.0):
        g0 = r0.copy()
    elif wi[0] == F(0.0):
        g0 = p0.copy()
    else:
        g0 = (p0 + (wi[0] * (r0 - p0).astype(F)).astype(F)).astype(F)
    tot = base(0) + g0.astype(D)
    old_lo = base(0) + p0.astype(D)
    fed_lo = base(0) + r0.astype(D)
    ph[(0,) + sl] = g0
    for k in range(nz):
        old_hi = base(k + 1) + ph[(k + 1,) + sl].astype(D)
        d_old = old_hi - old_lo
        w = wl[k]
        d = d_old
        if w != F(0.0):
            fed_hi = base(k + 1) + fed[(k + 1,) + sl].astype(D)
            d_fed = fed_hi - fed_lo
            d = d_fed if w == F(1.0) else d_old + D(w) * (d_fed - d_old)
            fed_lo = fed_hi
        tot = tot + d
        ph[(k + 1,) + sl] = (tot - base(k + 1)).astype(F)
        old_lo = old_hi
    return ph


def ph_column(ph, fed, phb, w_layer, w_iface, window):
    """``nest_fb_ph_column`` in place on ``ph`` (or its twin)."""

    if _is_host(ph):
        return ph_column_cpu(ph, fed, phb, w_layer, w_iface, window)
    import cupy as cp

    i0, j0, ni, nj = (int(v) for v in window)
    if ni <= 0 or nj <= 0:
        return ph
    for name, a in (("ph", ph), ("fed", fed)):
        if not a.flags.c_contiguous or a.dtype != cp.float32:
            raise ValueError(f"ph_column {name} must be contiguous float32")
    nz = int(ph.shape[0]) - 1
    phb = cp.ascontiguousarray(phb, dtype=cp.float32)
    _kernel("nest_fb_ph_column")(
        _blocks(ni * nj), (_BLOCK,),
        (ph, fed, phb, cp.asarray(w_layer, dtype=cp.float32),
         cp.asarray(w_iface, dtype=cp.float32), np.int32(nz),
         np.int32(ph.shape[1]), np.int32(ph.shape[2]),
         np.int32(1 if phb.ndim == 1 else 0), np.int32(i0), np.int32(j0),
         np.int32(ni), np.int32(nj)))
    return ph


# -- the height limit ----------------------------------------------------------


class FeedbackTopRefused(ValueError):
    """A feedback height limit the parent's levels cannot honour."""


def levels_below(phb, top_m: float, window) -> int:
    """How many of the parent's layers lie wholly below ``top_m`` metres
    above its ground in every column of ``window = (i0, j0, ni, nj)``,
    read on the base geopotential ``phb`` (``(nz+1, ny, nx)`` or a flat
    parent's ``(nz+1,)``)."""

    phb = np.asarray(_host(phb), dtype=D)
    if phb.ndim == 1:
        cols = phb[:, None]
    else:
        i0, j0, ni, nj = window
        cols = phb[:, j0:j0 + nj, i0:i0 + ni].reshape(phb.shape[0], -1)
    height = (cols[1:] - cols[:1]) / _G      # top of layer k, AGL
    below = np.all(height <= float(top_m), axis=1)
    count = 0
    for ok in below:
        if not ok:
            break
        count += 1
    return count


def level_weights(nz: int, below: int, taper: int):
    """``(layer weights (nz,), interface weights (nz+1,))`` in float32.

    ``below`` parent layers are fed back, the top ``taper`` of them at
    ``taper/(taper+1)`` .. ``1/(taper+1)``; with every layer below the
    limit nothing tapers (that IS full feedback)."""

    nz, below, taper = int(nz), int(below), int(taper)
    w = np.zeros(nz, dtype=D)
    if below >= nz:
        w[:] = 1.0
    else:
        n = min(taper, below)
        for k in range(below):
            w[k] = 1.0 if k < below - n else (below - k) / (n + 1)
    wi = np.empty(nz + 1, dtype=D)
    wi[0] = w[0]
    wi[nz] = w[nz - 1]
    wi[1:nz] = 0.5 * (w[:-1] + w[1:])
    return w.astype(F), wi.astype(F)


__all__ = [
    "FeedbackTopRefused", "INTERFACE_KINDS", "blend", "blend_cpu",
    "level_weights", "levels_below", "ph_column", "ph_column_cpu",
    "remap_layers", "remap_layers_cpu",
]
