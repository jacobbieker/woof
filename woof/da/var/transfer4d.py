"""WRFDA 4D-Var's transfers between ``xa`` and the WRFPLUS tangent-linear state
(docs/dev/WRFDA-4DVAR-PLAN.md, PR 4D.1).

- :func:`xa_to_tl` is ``da_transfer_xatowrftl``: the analysis increment at
  the start of the window (WRFDA's A grid, specific humidity, ``psfc``) to
  the initial perturbation of the tangent-linear model (C-grid ``u``/``v``,
  ``w``, ``theta``, ``mu``, the mixing ratio, the perturbation pressure and
  ``ph``).
- :func:`tl_to_xa` is ``da_transfer_wrftltoxa``: the tangent-linear state at
  one time slot back to ``xa``, linearised about that slot's first guess.
- :func:`xa_to_tl_adj` and :func:`tl_to_xa_adj` are WRFDA's adjoints
  (``da_transfer_xatowrftl_adj``, ``da_transfer_wrftltoxa_adj``), statement
  by statement in the serial build's order.

Every function takes the ``(xb, grid)`` pair :func:`~woof.da.var.analysis.wrftoxb`
returns for the time it is linearised about: WRFDA reads each slot's first
guess (``da_read_basicstates``) before ``da_transfer_wrftltoxa``, so the
reverse transfer at slot ``n`` reads ``xb``/``grid`` of slot ``n``, not of
the analysis time.  Arrays are in Fortran order ``(i, j, k)``, binary64.

Two things in ``da_transfer_xatowrftl`` are not physical; each is kept word
for word as the default, with a named opt-in correction
(docs/dev/WRFDA-4DVAR-PLAN.md):

``press="wrfda"`` (default) / ``press="dnw"``
    WRFDA integrates the full-level pressure perturbation down the column
    with ``dn`` (the spacing of the half levels: ``dn(1) = 0``, and on the
    tutorial ``sum(dn) = -0.9936`` against ``sum(dnw) = -1``).  With ``dnw``
    the surface full-level perturbation equals ``xa%psfc`` and the column is
    the dry hydrostatic one.
``ph="zero"`` (default) / ``ph="hydrostatic"``
    WRFDA starts the tangent-linear model with no geopotential perturbation
    (``g_ph_2 = 0``) whatever ``mu`` and ``theta`` do; the model then
    adjusts through sound waves.  ``"hydrostatic"`` integrates the linearised
    hypsometric equation (WRF's ``hypsometric_opt`` 1 or 2, as the first
    guess has it) from the increments, with the inverse density from the
    linearised equation of state.

Not ported: hydrometeor control variables (``cloud_cv_options >= 1``),
non-terrain-following coordinates (``hybrid_opt > 0``, which the transfer
ignores in WRFDA: it reads ``dnw`` and ``mub`` only), and the lateral
boundary perturbation ``da_transfer_wrftl_lbc_t0`` (P3a's window is
periodic).  ``xa%rho`` and the other derived fields come from
``da_transform_xtoxa`` after :func:`tl_to_xa` (PR 4D.2).

Derived from WRFDA 4.7.1 (``var/da/da_transfer_model/da_transfer_xatowrftl.inc``,
``da_transfer_xatowrftl_adj.inc``, ``da_transfer_wrftltoxa.inc``,
``da_transfer_wrftltoxa_adj.inc``), public domain; see
``licenses/LICENSE-WRF-public-domain.txt``.
"""

from __future__ import annotations

import numpy as np

from .analysis import CP, GAS_CONSTANT, KAPPA, RD_OVER_RV, T0

__all__ = ["TL_KEYS", "XA4D_KEYS", "check_supported", "hydrostatic_ph", "hydrostatic_ph_adj", "tl_to_xa",
           "tl_to_xa_adj", "xa_to_tl", "xa_to_tl_adj"]

#: The tangent-linear state the transfers write and read (WRFPLUS ``g_*``).
TL_KEYS = ("u_2", "v_2", "w_2", "t_2", "ph_2", "mu_2", "qv", "p")
#: The ``xa`` fields the transfers write and read.
XA4D_KEYS = ("u", "v", "w", "t", "q", "psfc", "p")

_PRESS = ("wrfda", "dnw")
_PH = ("zero", "hydrostatic")
CVPM = -(1.0 - GAS_CONSTANT / CP)
RVOVRD = 1.0 / RD_OVER_RV


def check_supported(xb: dict, grid: dict, *, cloud_cv_options: int = 0, press: str = "wrfda",
                    ph: str = "zero") -> None:
    """Refuse what the port does not cover (see the module docstring)."""
    if cloud_cv_options:
        raise NotImplementedError(f"cloud_cv_options = {cloud_cv_options}: the hydrometeor transfers are not ported")
    if press not in _PRESS:
        raise ValueError(f"press must be one of {_PRESS}, not {press!r}")
    if ph not in _PH:
        raise ValueError(f"ph must be one of {_PH}, not {ph!r}")
    c1h, c2h = grid.get("c1h"), grid.get("c2h")
    if c1h is not None and (np.any(np.asarray(c1h) != 1.0) or np.any(np.asarray(c2h) != 0.0)):
        raise NotImplementedError("a hybrid vertical coordinate (hybrid_opt > 0): WRFDA's transfer assumes "
                                  "terrain-following eta (c1h = 1, c2h = 0)")
    if "dn" not in xb:
        raise ValueError("the first guess has no DN, which da_transfer_xatowrftl integrates with")
    nx, ny, _ = xb["t"].shape
    if nx < 2 or ny < 2:
        raise ValueError("the A- to C-grid wind transfer needs at least 2 points in each direction")


def _column_weights(xb, grid, press):
    return np.asarray(xb["dn"] if press == "wrfda" else grid["dnw"], dtype=np.float64)


def _a_to_c(a, axis):
    """``g_u_2`` from ``xa%u`` (serial branch): the average inside, ``(3a - b)/2`` at the two edges."""
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    c = np.empty((n + 1,) + a.shape[1:])
    c[1:n] = 0.5 * (a[:-1] + a[1:])
    c[n] = (3.0 * a[n - 1] - a[n - 2]) / 2.0
    c[0] = (3.0 * a[0] - a[1]) / 2.0
    return np.moveaxis(c, 0, axis)


def _a_to_c_adj(c, axis):
    """``da_transfer_xatowrftl_adj``'s ``utmp`` form of the adjoint of :func:`_a_to_c`."""
    c = np.moveaxis(c, axis, 0)
    n = c.shape[0] - 1
    tmp = 0.5 * (c[1:] + c[:-1])
    lo = 0.5 * c[0]
    hi = 0.5 * c[n]
    a = np.zeros((n,) + c.shape[1:])
    a[0] = a[0] + 2.0 * lo
    a[1] = a[1] - lo
    a[n - 1] = a[n - 1] + 2.0 * hi
    a[n - 2] = a[n - 2] - hi
    a = a + tmp
    return np.moveaxis(a, 0, axis)


def _c_to_a(c, axis):
    c = np.moveaxis(c, axis, 0)
    return np.moveaxis(0.5 * (c[1:] + c[:-1]), 0, axis)


def _c_to_a_adj(a, axis):
    """``da_transfer_wrftltoxa_adj``'s serial branch."""
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    c = np.zeros((n + 1,) + a.shape[1:])
    c[1:n] = c[1:n] + 0.5 * (a[:-1] + a[1:])
    c[0] = c[0] + a[0] / 2.0
    c[n] = c[n] + a[n - 1] / 2.0
    return np.moveaxis(c, 0, axis)


def _s1md(grid):
    qv, dnw = grid["qv"], grid["dnw"]
    s1md = np.zeros(qv.shape[:2])
    for k in range(qv.shape[2]):
        s1md = s1md + (1.0 + qv[..., k]) * dnw[k]
    return s1md


# --------------------------------------------------------------------------- hydrostatic ph (opt-in)


def _alt(xb, grid):
    """The total inverse density WRF's equation of state gives, and its parts' derivatives."""
    theta = T0 + grid["t_2"]
    qvf = 1.0 + RVOVRD * grid["qv"]
    alt = (GAS_CONSTANT / grid["base_pres"]) * theta * qvf * (xb["p"] / grid["base_pres"]) ** CVPM
    return alt, theta, qvf


def _hyps(grid, mut, k):
    c3f, c4f, c3h, c4h, ptop = grid["c3f"], grid["c4f"], grid["c3h"], grid["c4h"], grid["p_top"]
    pfd = mut * c3f[k] + c4f[k] + ptop
    pfu = mut * c3f[k + 1] + c4f[k + 1] + ptop
    phm = mut * c3h[k] + c4h[k] + ptop
    return pfd, pfu, phm


def hydrostatic_ph(g: dict, xb: dict, grid: dict) -> np.ndarray:
    """The linearised hydrostatic ``g_ph_2`` from ``g_t_2``, ``g_qv``, ``g_p`` and ``g_mu_2`` (``ph="hydrostatic"``).

    ``hypsometric_opt = 1``: ``ph(k+1) = ph(k) - dnw(k) (mu alt - mub alb)``;
    ``2``: ``phi(k+1) = phi(k) + alt(k) phm(k) log(pfd/pfu)``, both with the
    surface fixed and ``alt`` from the equation of state.
    """
    alt, theta, qvf = _alt(xb, grid)
    g_alt = alt * (g["t_2"] / theta + RVOVRD * g["qv"] / qvf + CVPM * g["p"] / xb["p"])
    mut = grid["mu_2"] + grid["mub"]
    g_mu = g["mu_2"]
    nx, ny, nz = alt.shape
    out = np.zeros((nx, ny, nz + 1))
    for k in range(nz):
        if grid["hypsometric_opt"] == 1:
            d = -grid["dnw"][k] * (g_mu * alt[..., k] + mut * g_alt[..., k])
        else:
            pfd, pfu, phm = _hyps(grid, mut, k)
            lg = np.log(pfd / pfu)
            g_lg = g_mu * (grid["c3f"][k] / pfd - grid["c3f"][k + 1] / pfu)
            d = g_alt[..., k] * phm * lg + alt[..., k] * (grid["c3h"][k] * g_mu) * lg + alt[..., k] * phm * g_lg
        out[..., k + 1] = out[..., k] + d
    return out


def hydrostatic_ph_adj(a_ph: np.ndarray, xb: dict, grid: dict) -> dict:
    """The adjoint of :func:`hydrostatic_ph`: gradients for ``t_2``, ``qv``, ``p`` and ``mu_2``."""
    alt, theta, qvf = _alt(xb, grid)
    mut = grid["mu_2"] + grid["mub"]
    nx, ny, nz = alt.shape
    a_ph = np.array(a_ph, dtype=np.float64)
    a_alt = np.zeros((nx, ny, nz))
    a_mu = np.zeros((nx, ny))
    for k in range(nz - 1, -1, -1):
        a_d = a_ph[..., k + 1]
        a_ph[..., k] = a_ph[..., k] + a_d
        if grid["hypsometric_opt"] == 1:
            a_mu = a_mu - grid["dnw"][k] * alt[..., k] * a_d
            a_alt[..., k] = a_alt[..., k] - grid["dnw"][k] * mut * a_d
        else:
            pfd, pfu, phm = _hyps(grid, mut, k)
            lg = np.log(pfd / pfu)
            a_alt[..., k] = a_alt[..., k] + phm * lg * a_d
            a_mu = a_mu + (alt[..., k] * grid["c3h"][k] * lg
                           + alt[..., k] * phm * (grid["c3f"][k] / pfd - grid["c3f"][k + 1] / pfu)) * a_d
    return {"t_2": alt * a_alt / theta, "qv": alt * a_alt * RVOVRD / qvf, "p": alt * a_alt * CVPM / xb["p"],
            "mu_2": a_mu}


# --------------------------------------------------------------------------- xa -> tangent-linear model


def xa_to_tl(xa: dict, xb: dict, grid: dict, *, press: str = "wrfda", ph: str = "zero",
             cloud_cv_options: int = 0) -> dict:
    """``da_transfer_xatowrftl``: ``xa`` at the window start to the TL model's ``g_*`` (keys :data:`TL_KEYS`).

    ``xa["p"]`` is not read: WRFDA overwrites it with the integrated column
    (returned as ``g["p"]``).  ``xa["w"]`` is optional (zero if absent).
    """
    check_supported(xb, grid, cloud_cv_options=cloud_cv_options, press=press, ph=ph)
    nx, ny, nz = xb["t"].shape
    xq, xbq = xa["q"], xb["q"]
    g_qv = np.empty((nx, ny, nz))
    for k in range(nz):
        g_qv[..., k] = xq[..., k] / (1.0 - xbq[..., k]) ** 2
    qv, dnw = grid["qv"], grid["dnw"]
    sdmd = np.zeros((nx, ny))
    s1md = np.zeros((nx, ny))
    for k in range(nz):
        sdmd = sdmd + g_qv[..., k] * dnw[k]
        s1md = s1md + (1.0 + qv[..., k]) * dnw[k]
    g_mu = -(xa["psfc"] + xb["psac"] * sdmd) / s1md
    dn = _column_weights(xb, grid, press)
    mut = grid["mu_2"] + grid["mub"]
    g_press = np.zeros((nx, ny, nz + 1))
    g_p = np.empty((nx, ny, nz))
    for k in range(nz - 1, -1, -1):
        g_press[..., k] = g_press[..., k + 1] - (g_mu * (1.0 + qv[..., k]) + mut * g_qv[..., k]) * dn[k]
        g_p[..., k] = 0.5 * (g_press[..., k] + g_press[..., k + 1])
    g_t = (T0 + grid["t_2"]) * (xa["t"] / xb["t"] - KAPPA * g_p / xb["p"])
    g = {"u_2": _a_to_c(xa["u"], 0), "v_2": _a_to_c(xa["v"], 1),
         "w_2": np.array(xa["w"], dtype=np.float64) if "w" in xa else np.zeros((nx, ny, nz + 1)),
         "t_2": g_t, "mu_2": g_mu, "qv": g_qv, "p": g_p}
    g["ph_2"] = hydrostatic_ph(g, xb, grid) if ph == "hydrostatic" else np.zeros((nx, ny, nz + 1))
    return g


def xa_to_tl_adj(a: dict, xb: dict, grid: dict, *, press: str = "wrfda", ph: str = "zero",
                 cloud_cv_options: int = 0) -> dict:
    """``da_transfer_xatowrftl_adj``: the TL model's adjoint at the window start to ``xa``'s gradient.

    Missing keys of ``a`` are zero.  The returned ``p`` is zero (the forward
    transfer does not read ``xa%p``).
    """
    check_supported(xb, grid, cloud_cv_options=cloud_cv_options, press=press, ph=ph)
    nx, ny, nz = xb["t"].shape

    def get(key, shape):
        return np.array(a[key], dtype=np.float64) if key in a else np.zeros(shape)

    a_t2 = get("t_2", (nx, ny, nz))
    a_qv = get("qv", (nx, ny, nz))
    a_p = get("p", (nx, ny, nz))
    a_mu = get("mu_2", (nx, ny))
    if ph == "hydrostatic" and "ph_2" in a:
        h = hydrostatic_ph_adj(a["ph_2"], xb, grid)
        a_t2, a_qv, a_p, a_mu = a_t2 + h["t_2"], a_qv + h["qv"], a_p + h["p"], a_mu + h["mu_2"]
    out = {"w": get("w_2", (nx, ny, nz + 1))}
    out["u"] = _a_to_c_adj(get("u_2", (nx + 1, ny, nz)), 0)
    out["v"] = _a_to_c_adj(get("v_2", (nx, ny + 1, nz)), 1)
    xp = a_p
    theta = T0 + grid["t_2"]
    xp = xp - theta * KAPPA * a_t2 / xb["p"]
    out["t"] = theta * a_t2 / xb["t"]
    dn = _column_weights(xb, grid, press)
    qv = grid["qv"]
    mut = grid["mu_2"] + grid["mub"]
    a_press = np.zeros((nx, ny, nz + 1))
    for k in range(nz):
        a_press[..., k + 1] = a_press[..., k + 1] + 0.5 * xp[..., k]
        a_press[..., k] = a_press[..., k] + 0.5 * xp[..., k]
        a_qv[..., k] = a_qv[..., k] - mut * a_press[..., k] * dn[k]
        a_mu = a_mu - a_press[..., k] * (1.0 + qv[..., k]) * dn[k]
        a_press[..., k + 1] = a_press[..., k + 1] + a_press[..., k]
    s1md = _s1md(grid)
    sdmd = -xb["psac"] * a_mu / s1md
    out["psfc"] = -a_mu / s1md
    for k in range(nz):
        a_qv[..., k] = a_qv[..., k] + sdmd * grid["dnw"][k]
    out["q"] = a_qv / (1.0 - xb["q"]) ** 2
    out["p"] = np.zeros((nx, ny, nz))
    return out


# --------------------------------------------------------------------------- tangent-linear model -> xa


def tl_to_xa(g: dict, xb: dict, grid: dict, *, cloud_cv_options: int = 0) -> dict:
    """``da_transfer_wrftltoxa``: the TL state at a slot to ``xa`` about that slot's ``(xb, grid)``.

    ``g_ph_2`` is not read (WRFDA does not).
    """
    check_supported(xb, grid, cloud_cv_options=cloud_cv_options)
    nx, ny, nz = xb["t"].shape
    g_qv = g["qv"]
    xa = {"q": g_qv * (1.0 - xb["q"]) ** 2}
    qv, dnw = grid["qv"], grid["dnw"]
    sdmd = np.zeros((nx, ny))
    s1md = np.zeros((nx, ny))
    for k in range(nz):
        sdmd = sdmd + g_qv[..., k] * dnw[k]
        s1md = s1md + (1.0 + qv[..., k]) * dnw[k]
    xa["psfc"] = -xb["psac"] * sdmd - g["mu_2"] * s1md
    xa["p"] = np.array(g["p"], dtype=np.float64)
    xa["t"] = xb["t"] * (g["t_2"] / (T0 + grid["t_2"]) + KAPPA * xa["p"] / xb["p"])
    xa["u"] = _c_to_a(g["u_2"], 0)
    xa["v"] = _c_to_a(g["v_2"], 1)
    xa["w"] = np.array(g["w_2"], dtype=np.float64)
    return xa


def tl_to_xa_adj(a: dict, xb: dict, grid: dict, *, cloud_cv_options: int = 0) -> dict:
    """``da_transfer_wrftltoxa_adj``: ``xa``'s gradient at a slot to the adjoint forcing ``a_*`` (keys :data:`TL_KEYS`).

    Missing keys of ``a`` are zero; ``ph_2`` of the result is zero.
    """
    check_supported(xb, grid, cloud_cv_options=cloud_cv_options)
    nx, ny, nz = xb["t"].shape

    def get(key, shape):
        return np.array(a[key], dtype=np.float64) if key in a else np.zeros(shape)

    out = {"w_2": get("w", (nx, ny, nz + 1))}
    out["u_2"] = _c_to_a_adj(get("u", (nx, ny, nz)), 0)
    out["v_2"] = _c_to_a_adj(get("v", (nx, ny, nz)), 1)
    out["ph_2"] = np.zeros((nx, ny, nz + 1))
    xt = get("t", (nx, ny, nz))
    xp = get("p", (nx, ny, nz)) + xb["t"] * KAPPA * xt / xb["p"]
    out["t_2"] = xb["t"] * xt / (T0 + grid["t_2"])
    out["p"] = xp
    psfc = get("psfc", (nx, ny))
    s1md = _s1md(grid)
    out["mu_2"] = -psfc * s1md
    sdmd = -xb["psac"] * psfc
    g_qv = np.zeros((nx, ny, nz))
    for k in range(nz):
        g_qv[..., k] = g_qv[..., k] + sdmd * grid["dnw"][k]
    out["qv"] = g_qv + get("q", (nx, ny, nz)) * (1.0 - xb["q"]) ** 2
    return out
