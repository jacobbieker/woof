"""NumPy binary64 mirrors of WRFDA 4.7.1 routines, in WRFDA's operation order.

Each function here follows one WRFDA subroutine statement by statement, so
that, given the same binary64 inputs, it rounds exactly as a serial WRFDA
build does (gfortran ``-O2 -ftree-vectorize -funroll-loops`` on baseline
x86-64: no FMA contraction, no reassociation, ``x**3`` evaluated as
``(x*x)*x``).  Loops that WRFDA runs over an independent index are
vectorised over that index with element-wise NumPy operations, which round
each element exactly as the scalar loop would; reductions are written as
explicit left-to-right loops, never ``np.sum`` (pairwise).

These are the reference the GPU kernels and the oracle gates are checked
against, not a production path.

Derived from WRFDA 4.7.1 (``var/da/da_recursive_filter``), public domain;
see ``licenses/LICENSE-WRF-public-domain.txt``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "calculate_rf_factors",
    "recursive_filter_1d",
    "recursive_filter_1d_adj",
    "seq_sum",
]


def seq_sum(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """Left-to-right sum along ``axis`` (Fortran ``sum`` as gfortran -O2 does it)."""
    a = np.moveaxis(np.asarray(a, dtype=np.float64), axis, 0)
    acc = np.zeros(a.shape[1:], dtype=np.float64)
    for row in a:
        acc = acc + row
    return acc


def _pow3(x):
    return (x * x) * x


def recursive_filter_1d(pass_: int, alpha, field: np.ndarray) -> np.ndarray:
    """``da_recursive_filter_1d``: one forward pass along the last axis.

    ``alpha`` broadcasts against ``field[..., 0]`` (one alpha per line).
    Returns a new array.
    """
    a = np.asarray(field, dtype=np.float64)
    n = a.shape[-1]
    alpha = np.asarray(alpha, dtype=np.float64)
    one_alpha = 1.0 - alpha
    b = np.empty_like(a)
    if pass_ == 1:
        b[..., 0] = one_alpha * a[..., 0]
    elif pass_ == 2:
        b[..., 0] = a[..., 0] / (1.0 + alpha)
    else:
        den = 1.0 - alpha * alpha
        b[..., 0] = one_alpha * (a[..., 0] - _pow3(alpha) * a[..., 1]) / (den * den)
    for j in range(1, n):
        b[..., j] = alpha * b[..., j - 1] + one_alpha * a[..., j]
    c = np.empty_like(a)
    if pass_ == 1:
        c[..., n - 1] = b[..., n - 1] / (1.0 + alpha)
    else:
        den = 1.0 - alpha * alpha
        c[..., n - 1] = one_alpha * (b[..., n - 1] - _pow3(alpha) * b[..., n - 2]) / (den * den)
    for j in range(n - 2, -1, -1):
        c[..., j] = alpha * c[..., j + 1] + one_alpha * b[..., j]
    return c


def recursive_filter_1d_adj(pass_: int, alpha, field: np.ndarray) -> np.ndarray:
    """``da_recursive_filter_1d_adj``: the adjoint of one pass, last axis."""
    c = np.array(field, dtype=np.float64, copy=True)
    n = c.shape[-1]
    alpha = np.asarray(alpha, dtype=np.float64)
    one_alpha = 1.0 - alpha
    b = np.zeros_like(c)
    for j in range(0, n - 1):
        c[..., j + 1] = c[..., j + 1] + alpha * c[..., j]
        b[..., j] = one_alpha * c[..., j]
    if pass_ == 1:
        b[..., n - 1] = b[..., n - 1] + c[..., n - 1] / (1.0 + alpha)
    else:
        den = 1.0 - alpha * alpha
        den2 = den * den
        b[..., n - 1] = b[..., n - 1] + one_alpha * c[..., n - 1] / den2
        b[..., n - 2] = b[..., n - 2] - one_alpha * _pow3(alpha) * c[..., n - 1] / den2
    a = np.zeros_like(c)
    for j in range(n - 1, 0, -1):
        b[..., j - 1] = b[..., j - 1] + alpha * b[..., j]
        a[..., j] = a[..., j] + one_alpha * b[..., j]
    if pass_ == 1:
        a[..., 0] = a[..., 0] + one_alpha * b[..., 0]
    elif pass_ == 2:
        a[..., 0] = a[..., 0] + b[..., 0] / (1.0 + alpha)
    else:
        den = 1.0 - alpha * alpha
        den2 = den * den
        a[..., 0] = a[..., 0] + one_alpha * b[..., 0] / den2
        a[..., 1] = a[..., 1] - one_alpha * _pow3(alpha) * b[..., 0] / den2
    return a


def calculate_rf_factors(rf_lengthscale, rf_passes: int):
    """``da_calculate_rf_factors``: ``(rf_alpha, rf_scale_factor)`` per mode.

    ``rf_lengthscale`` is non-dimensional (grid lengths).  The scale factor
    is the inverse of the filtered delta function's peak, filtered on
    ``-500..500`` by ``rf_passes/2`` adjoint then ``rf_passes/2`` forward
    passes, as WRFDA does.
    """
    s = np.asarray(rf_lengthscale, dtype=np.float64)
    rf_e = 0.25 * rf_passes / (s * s)
    rf_alpha = 1.0 + rf_e - np.sqrt(rf_e * (rf_e + 2.0))
    n = 500
    field = np.zeros(s.shape + (2 * n + 1,), dtype=np.float64)
    field[..., n] = 1.0
    for p in range(1, rf_passes // 2 + 1):
        field = recursive_filter_1d_adj(p, rf_alpha, field)
    for p in range(1, rf_passes // 2 + 1):
        field = recursive_filter_1d(p, rf_alpha, field)
    return rf_alpha, 1.0 / field[..., n]


# ------------------------------------------------------------------ U, U^T
#
# Arrays are in WRFDA's index order, 0-based: a 3-D field is (nx, ny, nz)
# over the tile (its:ite, jts:jte, kts:kte); vv/vp blocks are (nx, ny, mz).


def transform_through_rf(field, rf_alpha, grid_box_area, rf_passes):
    """``da_transform_through_rf`` (``vert_corr = 2``: no ``val`` scaling)."""
    f = np.asarray(field, dtype=np.float64)
    p_x = np.sqrt(np.asarray(grid_box_area, dtype=np.float64))
    v = f / p_x[:, :, None]
    alpha = np.asarray(rf_alpha, dtype=np.float64)
    # x sweeps: lines along i for each (j, m)
    lines = np.transpose(v, (1, 2, 0))  # (ny, mz, nx)
    for p in range(1, rf_passes // 2 + 1):
        lines = recursive_filter_1d(p, alpha[None, :], lines)
    v = np.transpose(lines, (2, 0, 1))
    # y sweeps: lines along j for each (i, m)
    lines = np.transpose(v, (0, 2, 1))  # (nx, mz, ny)
    for p in range(1, rf_passes // 2 + 1):
        lines = recursive_filter_1d(p, alpha[None, :], lines)
    v = np.transpose(lines, (0, 2, 1))
    return v * p_x[:, :, None]


def transform_through_rf_adj(field, rf_alpha, grid_box_area, rf_passes):
    """``da_transform_through_rf_adj``: y sweeps, then x, passes descending."""
    f = np.asarray(field, dtype=np.float64)
    p_x = np.sqrt(np.asarray(grid_box_area, dtype=np.float64))
    v = f * p_x[:, :, None]
    alpha = np.asarray(rf_alpha, dtype=np.float64)
    lines = np.transpose(v, (0, 2, 1))
    for p in range(rf_passes // 2, 0, -1):
        lines = recursive_filter_1d_adj(p, alpha[None, :], lines)
    v = np.transpose(lines, (0, 2, 1))
    lines = np.transpose(v, (1, 2, 0))
    for p in range(rf_passes // 2, 0, -1):
        lines = recursive_filter_1d_adj(p, alpha[None, :], lines)
    v = np.transpose(lines, (2, 0, 1))
    return v / p_x[:, :, None]


def vvtovp(evec, val, vv, levels):
    """``da_transform_vvtovp`` (``vertical_ip = 0``); evec (ny, nk, mz), val (ny, mz)."""
    nx, ny, mz = vv.shape
    vp = np.zeros((nx, ny, levels))
    for k in range(levels):
        acc = np.zeros((nx, ny))
        for m in range(mz):
            temp = evec[:, k, m] * val[:, m]
            acc = acc + temp[None, :] * vv[:, :, m]
        vp[:, :, k] = acc
    return vp


def vvtovp_adj(evec, val, vp, mz):
    """``da_transform_vvtovp_adj``."""
    nx, ny, levels = vp.shape
    vv = np.zeros((nx, ny, mz))
    for m in range(mz):
        acc = np.zeros((nx, ny))
        for k in range(levels):
            temp = evec[:, k, m] * val[:, m]
            acc = acc + temp[None, :] * vp[:, :, k]
        vv[:, :, m] = acc
    return vv


def psichi_to_uv(psi, chi, coefx, coefy):
    """``da_psichi_to_uv`` (regional, no A2C): A-grid u, v from psi, chi."""
    nx, ny, nz = psi.shape
    cx = coefx[:, :, None]
    cy = coefy[:, :, None]
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    I = slice(1, nx - 1)
    J = slice(1, ny - 1)
    u[I, J] = -(cy[I, J] * (psi[I, 2:] - psi[I, :-2])) + cx[I, J] * (chi[2:, J] - chi[:-2, J])
    v[I, J] = cx[I, J] * (psi[2:, J] - psi[:-2, J]) + cy[I, J] * (chi[I, 2:] - chi[I, :-2])
    i = 0
    u[i, J] = -(cy[i, J] * (psi[i, 2:] - psi[i, :-2])) + cx[i, J] * (chi[i + 2, J] - chi[i, J])
    v[i, J] = cx[i, J] * (psi[i + 2, J] - psi[i, J]) + cy[i, J] * (chi[i, 2:] - chi[i, :-2])
    i = nx - 1
    u[i, J] = -(cy[i, J] * (psi[i, 2:] - psi[i, :-2])) + cx[i, J] * (chi[i, J] - chi[i - 2, J])
    v[i, J] = cx[i, J] * (psi[i, J] - psi[i - 2, J]) + cy[i, J] * (chi[i, 2:] - chi[i, :-2])
    j = 0
    u[I, j] = -(cy[I, j] * (psi[I, j + 2] - psi[I, j])) + cx[I, j] * (chi[2:, j] - chi[:-2, j])
    v[I, j] = cx[I, j] * (psi[2:, j] - psi[:-2, j]) + cy[I, j] * (chi[I, j + 2] - chi[I, j])
    j = ny - 1
    u[I, j] = -(cy[I, j] * (psi[I, j] - psi[I, j - 2])) + cx[I, j] * (chi[2:, j] - chi[:-2, j])
    v[I, j] = cx[I, j] * (psi[2:, j] - psi[:-2, j]) + cy[I, j] * (chi[I, j] - chi[I, j - 2])
    for a in (u, v):
        a[0, 0] = 0.5 * (a[1, 0] + a[0, 1])
        a[nx - 1, 0] = 0.5 * (a[nx - 2, 0] + a[nx - 1, 1])
        a[0, ny - 1] = 0.5 * (a[1, ny - 1] + a[0, ny - 2])
        a[nx - 1, ny - 1] = 0.5 * (a[nx - 2, ny - 1] + a[nx - 1, ny - 2])
    return u, v


def psichi_to_uv_adj(u, v, coefx, coefy, psi=None, chi=None):
    """``da_psichi_to_uv_adj``: serial scatter in WRFDA's order (k vectorised).

    Accumulates into ``psi``, ``chi`` (zeros if None); ``u``, ``v`` are
    copied first (WRFDA updates its own u, v at the corners).
    """
    u = np.array(u, dtype=np.float64)
    v = np.array(v, dtype=np.float64)
    nx, ny, nz = u.shape
    psi = np.zeros_like(u) if psi is None else np.array(psi, dtype=np.float64)
    chi = np.zeros_like(u) if chi is None else np.array(chi, dtype=np.float64)
    ie, je = nx - 1, ny - 1
    for a in (u, v):
        a[1, 0] = a[1, 0] + 0.5 * a[0, 0]
        a[0, 1] = a[0, 1] + 0.5 * a[0, 0]
    for a in (u, v):
        a[ie - 1, 0] = a[ie - 1, 0] + 0.5 * a[ie, 0]
        a[ie, 1] = a[ie, 1] + 0.5 * a[ie, 0]
    for a in (u, v):
        a[1, je] = a[1, je] + 0.5 * a[0, je]
        a[0, je - 1] = a[0, je - 1] + 0.5 * a[0, je]
    for a in (u, v):
        a[ie - 1, je] = a[ie - 1, je] + 0.5 * a[ie, je]
        a[ie, je - 1] = a[ie, je - 1] + 0.5 * a[ie, je]
    cx, cy = coefx, coefy

    def scatter(i, j, pxp, pxm, cyp, cym, pyp, pym, cxp, cxm):
        # psi(pxp) += cx v; psi(pxm) -= cx v; chi(cyp) += cy v; chi(cym) -= cy v
        # psi(pyp) -= cy u; psi(pym) += cy u; chi(cxp) += cx u; chi(cxm) -= cx u
        vv = v[i, j]
        uu = u[i, j]
        psi[pxp] = psi[pxp] + cx[i, j] * vv
        psi[pxm] = psi[pxm] - cx[i, j] * vv
        chi[cyp] = chi[cyp] + cy[i, j] * vv
        chi[cym] = chi[cym] - cy[i, j] * vv
        psi[pyp] = psi[pyp] - cy[i, j] * uu
        psi[pym] = psi[pym] + cy[i, j] * uu
        chi[cxp] = chi[cxp] + cx[i, j] * uu
        chi[cxm] = chi[cxm] - cx[i, j] * uu

    # WRFDA loops k outermost; k-slices are independent, so k is vectorised.
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
            scatter(i, j, (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1), (i, j + 1), (i, j - 1), (i + 1, j), (i - 1, j))
    return psi, chi


def vptox(vp, reg, cv_options, coefx, coefy, qs):
    """``da_transform_vptox`` for cv_options 5 and 7 (``jb_factor > 0``).

    ``vp`` is a dict v1..v5 (v5 is (nx, ny, 1)); it is updated as WRFDA
    updates it (balanced parts added).  Returns ``(vp, xa)`` with xa keys
    u, v, t, q, psfc.
    """
    vp = {k: np.array(a, dtype=np.float64) for k, a in vp.items()}
    nz = vp["v1"].shape[2]
    if cv_options == 6:
        raise NotImplementedError("woof var: cv_options = 6 transforms are not implemented")
    if cv_options != 7:
        rt = reg["psi_t"]  # (ny, nz, nz): reg_psi_t(j, k, k1)
        for k1 in range(nz):
            for k in range(nz):
                vp["v3"][:, :, k] = vp["v3"][:, :, k] + rt[None, :, k, k1] * vp["v1"][:, :, k1]
    xa = {"t": vp["v3"].copy()}
    if cv_options != 7:
        rp = reg["psi_ps"]  # (ny, nz)
        for k in range(nz):
            vp["v5"][:, :, 0] = vp["v5"][:, :, 0] + rp[None, :, k] * vp["v1"][:, :, k]
    xa["psfc"] = vp["v5"][:, :, 0].copy()
    if cv_options != 7:
        rc = reg["psi_chi"]  # (ny, nz)
        vp["v2"] = vp["v2"] + rc[None, :, :] * vp["v1"]
        xa["u"], xa["v"] = psichi_to_uv(vp["v1"], vp["v2"], coefx, coefy)
    else:
        xa["u"], xa["v"] = vp["v1"].copy(), vp["v2"].copy()
    xa["q"] = vp["v4"] * qs
    return vp, xa


def vptox_adj(xa, reg, cv_options, coefx, coefy, qs):
    """``da_transform_vptox_adj`` (cv 5/7) from zeroed ``vp``; returns vp dict."""
    nx, ny, nz = xa["t"].shape
    z = np.zeros((nx, ny, nz))
    vp = {"v1": z.copy(), "v2": z.copy(), "v3": z.copy(), "v4": z.copy(), "v5": np.zeros((nx, ny, 1))}
    vp["v4"] = vp["v4"] + xa["q"] * qs
    if cv_options != 7:
        vp["v1"], vp["v2"] = psichi_to_uv_adj(xa["u"], xa["v"], coefx, coefy, vp["v1"], vp["v2"])
    else:
        vp["v1"] = xa["u"] + vp["v1"]
        vp["v2"] = xa["v"] + vp["v2"]
    psfc = xa["psfc"]
    if cv_options != 7:
        rp = reg["psi_ps"]
        vp["v1"] = vp["v1"] + rp[None, :, :] * psfc[:, :, None]
    vp["v5"][:, :, 0] = vp["v5"][:, :, 0] + psfc
    if cv_options != 7:
        rt = reg["psi_t"]
        for k1 in range(nz):
            for k in range(nz):
                vp["v1"][:, :, k1] = vp["v1"][:, :, k1] + rt[None, :, k, k1] * xa["t"][:, :, k]
    vp["v3"] = vp["v3"] + xa["t"]
    if cv_options != 7:
        vp["v1"] = vp["v1"] + reg["psi_chi"][None, :, :] * vp["v2"]
    return vp


def xtoxa(xa, xb, c1h, c2h):
    """``da_transform_xtoxa`` (regional WRF, 3D-Var): adds xa p and rho.

    ``xb`` keys q, psac (2-D), dnw (nz), rho, p, t.  tpw / gpsref are the
    observation operators' business, not here.
    """
    q_a, q_b = xa["q"], xb["q"]
    nx, ny, nz = q_a.shape
    dnw = xb["dnw"]
    psac = xb["psac"]
    one_m_q = 1.0 - q_b
    mr_a = q_a / (one_m_q * one_m_q)
    mr_b = q_b / one_m_q
    sdmd = np.zeros((nx, ny))
    s1md = np.zeros((nx, ny))
    for k in range(nz):
        sdmd = sdmd + mr_a[:, :, k] * dnw[k]
        s1md = s1md + (1.0 + mr_b[:, :, k]) * dnw[k]
    mu = -(xa["psfc"] + psac * sdmd) / s1md
    p_hi = np.zeros((nx, ny))
    pa = np.empty((nx, ny, nz))
    for k in range(nz - 1, -1, -1):
        pk = p_hi - (c1h[k] * mu * (1.0 + mr_b[:, :, k]) + (c1h[k] * psac + c2h[k]) * mr_a[:, :, k]) * dnw[k]
        pa[:, :, k] = 0.5 * (pk + p_hi)
        p_hi = pk
    out = dict(xa)
    out["p"] = pa
    out["rho"] = xb["rho"] * (pa / xb["p"] - xa["t"] / xb["t"])
    return out


def xtoxa_adj(xa, xb, c1h, c2h):
    """``da_transform_xtoxa_adj`` (test_transforms path: no tpw/gpsref).

    Accumulates into copies of xa's p, t, psfc, q; rho is consumed (zeroed).
    """
    out = {k: np.array(v, dtype=np.float64) for k, v in xa.items()}
    temp = out["rho"] * xb["rho"]
    out["p"] = out["p"] + temp / xb["p"]
    out["t"] = out["t"] - temp / xb["t"]
    out["rho"] = np.zeros_like(out["rho"])
    q_b = xb["q"]
    nx, ny, nz = q_b.shape
    dnw = xb["dnw"]
    psac = xb["psac"]
    mu = np.zeros((nx, ny))
    s1md = np.zeros((nx, ny))
    p = np.zeros((nx, ny, nz + 1))
    mr_a = np.empty((nx, ny, nz))
    for k in range(nz):
        mr_b = q_b[:, :, k] / (1.0 - q_b[:, :, k])
        s1md = s1md + (1.0 + mr_b) * dnw[k]
        p[:, :, k] = p[:, :, k] + 0.5 * out["p"][:, :, k]
        p[:, :, k + 1] = p[:, :, k + 1] + 0.5 * out["p"][:, :, k]
        mu = mu - c1h[k] * p[:, :, k] * (1.0 + mr_b) * dnw[k]
        mr_a[:, :, k] = -(p[:, :, k] * (c1h[k] * psac + c2h[k]) * dnw[k])
        p[:, :, k + 1] = p[:, :, k + 1] + p[:, :, k]
    out["psfc"] = out["psfc"] - mu / s1md
    sdmd = -(mu * psac / s1md)
    for k in range(nz):
        mr_a[:, :, k] = mr_a[:, :, k] + sdmd * dnw[k]
        d = 1.0 - q_b[:, :, k]
        out["q"][:, :, k] = out["q"][:, :, k] + mr_a[:, :, k] / (d * d)
    return out


def cv_to_vv(cv, nx, ny, cv_mz):
    """``da_cv_to_vv``: blocks v1..v5, each (nx, ny, mz) in Fortran order.

    ``cv`` must hold exactly those blocks: alpha or VarBC control variables
    are not part of this transform, and a size mismatch is an error.
    """
    want = nx * ny * sum(int(m) for m in cv_mz[:5])
    if len(cv) != want:
        raise ValueError(f"cv has {len(cv)} values; nx*ny*sum(cv_mz[:5]) is {want}")
    out, off = {}, 0
    for n, mz in enumerate(cv_mz[:5], start=1):
        size = nx * ny * mz
        out[f"v{n}"] = np.asarray(cv[off : off + size], dtype=np.float64).reshape((nx, ny, mz), order="F")
        off += size
    return out


def vv_to_cv(vv, cv_mz):
    """``da_vv_to_cv``."""
    parts = [np.asarray(vv[f"v{n}"]).reshape(-1, order="F") for n, mz in enumerate(cv_mz[:5], start=1) if mz > 0]
    return np.concatenate(parts) if parts else np.zeros(0)


def transform_u(cv, b, xb):
    """``U v``: vtovv (RF), vertical EOFs, vptox, xtoxa.  Returns dict of stages.

    ``b`` is a :class:`woof.da.var.be_dat.RegionalB`; ``xb`` holds
    grid_box_area, coefx, coefy, qs, q, psac, dnw, rho, p, t, c1h, c2h.
    """
    nx, ny = xb["grid_box_area"].shape
    nz = b.kz
    rf_passes = b.rf_passes  # the passes setup_be_regional normalised rf_alpha and val for
    vv = cv_to_vv(cv, nx, ny, b.cv_mz)
    for n, s in enumerate(b.v, start=1):
        if s.mz > 0:
            vv[f"v{n}"] = transform_through_rf(vv[f"v{n}"], s.rf_alpha, xb["grid_box_area"], rf_passes)
    vp = {}
    for n, s in enumerate(b.v, start=1):
        lev = 1 if n == 5 else nz
        vp[f"v{n}"] = vvtovp(s.evec, s.val, vv[f"v{n}"], lev) if s.mz > 0 else np.zeros((nx, ny, lev))
    vp_bal, xa = vptox(vp, b.reg, b.cv_options, xb["coefx"], xb["coefy"], xb["qs"])
    xa = xtoxa(xa, xb, xb["c1h"], xb["c2h"])
    return {"vv": vv, "vp": vp, "vp_bal": vp_bal, "xa": xa}


def transform_u_adj(xa, b, xb):
    """``U^T x``: xtoxa_adj, vptox_adj, vertical EOF adjoint, RF adjoint, vv_to_cv."""
    nx, ny, nz = xa["t"].shape
    rf_passes = b.rf_passes
    xa1 = xtoxa_adj(xa, xb, xb["c1h"], xb["c2h"])
    vp = vptox_adj(xa1, b.reg, b.cv_options, xb["coefx"], xb["coefy"], xb["qs"])
    vv = {}
    for n, s in enumerate(b.v, start=1):
        vv[f"v{n}"] = vvtovp_adj(s.evec, s.val, vp[f"v{n}"], s.mz) if s.mz > 0 else np.zeros((nx, ny, 0))
    vv_eof = dict(vv)
    for n, s in enumerate(b.v, start=1):
        if s.mz > 0:
            vv[f"v{n}"] = transform_through_rf_adj(vv[f"v{n}"], s.rf_alpha, xb["grid_box_area"], rf_passes)
    return {"xa": xa1, "vp": vp, "vv_eof": vv_eof, "vv": vv, "cv": vv_to_cv(vv, b.cv_mz)}
