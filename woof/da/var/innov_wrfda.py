"""WRFDA's innovation step for the conventional types (EXPERIMENTAL).

A port of ``da_get_innov_vector`` in WRFDA 4.7.1 for the types
:mod:`woof.da.var.obs_wrfda` reads, on a regional ARW first guess held as
one tile, with the tutorial's namelist (``sfc_assi_options = 1``,
``check_max_iv = .true.`` with every ``max_error_*`` 5,
``max_stheight_diff = 100``, ``qc_rej_both``, ``wind_sd*``,
``obs_err_inflate``, ``psfc_from_slp``, ``sfcht_adjust_q`` and
``position_lev_dependant`` all ``.false.``, first outer loop):

* the vertical position: ``da_to_zk`` (pressure or height) on the
  horizontally interpolated column, ``da_convert_zk`` to ``(k, dz, dzm)``;
* the surface types (synop, metar, ships, buoy, sonde_sfc) with
  ``da_obs_sfc_correction`` (``da_mo_correction``, ``da_intpsfc_prs``,
  ``da_intpsfc_tem``) for stations below the model surface;
* ``H(x_b)`` by ``da_interp_lin_3d`` / ``da_interp_lin_2d``, the innovation
  ``O - H(x_b)`` with each type's own missing-value rules (which levels are
  reset to 0 and which keep the observed value);
* the per-type ``da_check_max_iv_*`` (``da_max_error_qc``);
* the nonlinear types: GNSS precipitable water (``gpspw``, with the
  station-height correction ``dpw``), GNSS-RO refractivity (``gpsref``, with
  ``da_qc_gpsref``) and satem thickness (``da_tv_profile``,
  ``da_find_layer``, ``da_thickness``, the land-mask rule);
* ``Jo`` at the background as ``da_jo_and_grady`` sums it (:func:`jo`).

Every arithmetic step is binary64 in WRFDA's order (the oracle build is
``-fdefault-real-8``, gfortran ``-O2``, no FMA). Element-wise NumPy over an
independent index rounds exactly as the Fortran loop does; reductions are
left-to-right loops. ``log``, ``exp`` and real ``**`` are correctly rounded
by default (:mod:`woof.da.var.crmath`); ``libm="host"`` uses the host C
library, which on a glibc host is the oracle's. Constant expressions that
gfortran folds at compile time (``log(40.0/0.05)``) are always correctly
rounded, as MPFR folds them.

Against the oracle build's ``woof_iv.bin`` (``tests/test_var_innov_wrfda.py``)
the qc flags agree everywhere and the innovations to 1e-10 relative; the
residuals are libm ulps in the surface correction's ``**``/``exp``/``log``
and the GNSS-RO height (see :mod:`woof.da.var.obs_wrfda`).

Derived from WRFDA 4.7.1 (``var/da/da_synop``, ``da_metar``, ``da_ships``,
``da_buoy``, ``da_sound``, ``da_airep``, ``da_pilot``, ``da_profiler``,
``da_geoamv``, ``da_polaramv``, ``da_qscat``, ``da_gpspw``, ``da_gpsref``,
``da_satem``, ``da_interpolation``, ``da_tools``, ``da_minimisation``),
public domain; see ``licenses/LICENSE-WRF-public-domain.txt``.
"""

from __future__ import annotations

import copy
import math
import types
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import crmath
from .obs_wrfda import MISSING_DATA, MISSING_R, WrfdaObs

__all__ = ["Options", "innovations", "jo", "jo_total", "to_zk", "convert_zk",
           "interp_lin_3d", "interp_lin_2d", "JO_ORDER", "JO_FIELDS"]

GRAVITY = 9.81
GAS_CONSTANT = 287.0
CP = 7.0 * GAS_CONSTANT / 2.0
RCP = GAS_CONSTANT / CP
PI = 3.1415926
RADIAN = PI / 180.0
FAILS_ERROR_MAX = -3
SURFACE_CORRECTION = 2
ABOVE_MODEL_LID = -8
SURFACE_TYPES = ("synop", "metar", "ships", "buoy", "sonde_sfc")

#: ``da_jo_and_grady``'s per-type field sums, in its order.
JO_FIELDS = {
    "sound": ("u", "v", "t", "q"), "sonde_sfc": ("u", "v", "t", "q", "p"),
    "synop": ("u", "v", "t", "p", "q"), "geoamv": ("u", "v"), "polaramv": ("u", "v"),
    "airep": ("u", "v", "t", "q"), "pilot": ("u", "v"), "satem": ("thickness",),
    "metar": ("u", "v", "t", "p", "q"), "ships": ("u", "v", "t", "p", "q"),
    "gpspw": ("tpw",), "qscat": ("u", "v"), "profiler": ("u", "v"),
    "buoy": ("u", "v", "t", "p", "q"), "gpsref": ("ref",),
}
#: ``jo%total``'s summation order.
JO_ORDER = ("sound", "sonde_sfc", "geoamv", "polaramv", "synop", "satem", "pilot", "airep",
            "metar", "ships", "gpspw", "qscat", "profiler", "buoy", "gpsref")


@dataclass(frozen=True)
class Options:
    """The namelist values this port follows (WRFDA's defaults/tutorial)."""

    check_max_iv: bool = True
    max_error_t: float = 5.0
    max_error_uv: float = 5.0
    max_error_pw: float = 5.0
    max_error_ref: float = 5.0
    max_error_q: float = 5.0
    max_error_p: float = 5.0
    max_error_thickness: float = 5.0
    max_stheight_diff: float = 100.0
    # da_qc_gpsref (registry defaults)
    gpsref_qc_dndz_opt: int = 1
    gpsref_qc_dndz_thresh: float = -50.0
    gpsref_qc_dndz2_opt: int = 1
    gpsref_qc_dndz2_thresh: float = 100.0
    gpsref_qc_gsi_opt: int = 1
    gpsref_qc_pcnt_opt: int = 1
    gpsref_qc_pcnt_h1: float = 7000.0
    gpsref_qc_pcnt_h2: float = 25000.0
    gpsref_qc_pcnt_below: float = 0.05
    gpsref_qc_pcnt_middle: float = 0.04
    gpsref_qc_pcnt_above: float = 0.10


@lru_cache(maxsize=2)
def _libm(name: str = "correctly-rounded"):
    if name == "correctly-rounded":
        return crmath
    if name != "host":
        raise ValueError(f"libm {name!r}: expected 'correctly-rounded' or 'host'")
    from woof.core.host_libm import scalar

    return types.SimpleNamespace(exp=scalar("exp"), log=scalar("log"), pow_=scalar("pow"),
                                 cos=scalar("cos"), sqrt=math.sqrt)


# --------------------------------------------------------------------------- interpolation
def _w(info, rows=None):
    """``(i0, j0, dx, dy, dxm, dym)`` per report (0-based i, j), or per row."""
    i0 = np.asarray(info["i"], dtype=np.int64) - 1
    j0 = np.asarray(info["j"], dtype=np.int64) - 1
    w = [np.asarray(info[k], dtype=np.float64) for k in ("dx", "dy", "dxm", "dym")]
    if rows is not None:
        i0, j0 = i0[rows], j0[rows]
        w = [a[rows] for a in w]
    return (i0, j0, *w)


def _hcol(f, i0, j0, dx, dy, dxm, dym):
    """``dym*(dxm*f(i,j)+dx*f(i+1,j)) + dy*(dxm*f(i,j+1)+dx*f(i+1,j+1))``."""
    if f.ndim == 3 and np.ndim(i0) > 0:
        dx, dy, dxm, dym = (a[:, None] for a in (dx, dy, dxm, dym))
    return (dym * (dxm * f[i0, j0] + dx * f[i0 + 1, j0])
            + dy * (dxm * f[i0, j0 + 1] + dx * f[i0 + 1, j0 + 1]))


def interp_lin_3d(f, geo):
    """``da_interp_lin_3d`` per level; ``geo`` is a :class:`Geometry`."""
    out = np.zeros(geo.nlev, dtype=np.float64)
    m = geo.k > 0
    if not m.any():
        return out
    r = geo.rep[m]
    i0, j0, dx, dy, dxm, dym = (a[r] for a in geo.h)
    k0 = geo.k[m] - 1

    def fmz(kk):
        return (dym * (dxm * f[i0, j0, kk] + dx * f[i0 + 1, j0, kk])
                + dy * (dxm * f[i0, j0 + 1, kk] + dx * f[i0 + 1, j0 + 1, kk]))

    out[m] = geo.dzm[m] * fmz(k0) + geo.dz[m] * fmz(k0 + 1)
    return out


def interp_lin_2d(f2, geo):
    """``da_interp_lin_2d`` at level 1, per report."""
    return _hcol(f2, *geo.h)


@dataclass
class Geometry:
    """One type's horizontal weights (per report) and vertical ``(k, dz, dzm)``
    (per level, report by report), as WRFDA's ``infa_type`` holds them after
    ``da_convert_zk``."""

    h: tuple
    rep: np.ndarray
    k: np.ndarray
    dz: np.ndarray
    dzm: np.ndarray
    zk: np.ndarray

    @property
    def nlev(self) -> int:
        return len(self.rep)

    @property
    def nrep(self) -> int:
        return len(self.h[0])

    @classmethod
    def of(cls, t, kte: int) -> "Geometry":
        lev = np.asarray(t.info["levels"], dtype=np.int64)
        rep = np.repeat(np.arange(len(lev)), lev)
        zk = t.extra["zk"]
        k, dz, dzm = convert_zk(zk, kte)
        return cls(_w(t.info), rep, k, dz, dzm, zk)


# --------------------------------------------------------------------------- vertical
def to_zk(obs_v: float, mdl_v: np.ndarray, by_pressure: bool) -> float:
    """``da_to_zk`` (``anal_type_verify = .false.``); 1-based fractional level."""
    kte = len(mdl_v)
    if by_pressure:
        if obs_v > mdl_v[0] or obs_v < mdl_v[kte - 1]:
            return MISSING_R
        hit = (obs_v <= mdl_v[:-1]) & (obs_v >= mdl_v[1:])
    else:
        if obs_v < mdl_v[0] or obs_v > mdl_v[kte - 1]:
            return MISSING_R
        hit = (obs_v >= mdl_v[:-1]) & (obs_v <= mdl_v[1:])
    if not hit.any():
        return MISSING_R
    k = int(np.argmax(hit))
    return float(k + 1) + (mdl_v[k] - obs_v) / (mdl_v[k] - mdl_v[k + 1])


def convert_zk(zk: np.ndarray, kte: int):
    """``da_convert_zk``: ``(k, dz, dzm)`` (k 1-based, 0 for missing; ``k``
    clipped to ``1..kte-1`` with ``kte`` the first guess's level count)."""
    zk = np.asarray(zk, dtype=np.float64)
    ok = (zk > 0.0) & (zk != MISSING_R)
    k = np.zeros(zk.shape, dtype=np.int64)
    k[ok] = np.clip(np.trunc(zk[ok]).astype(np.int64), 1, kte - 1)
    dz = np.zeros(zk.shape)
    dzm = np.zeros(zk.shape)
    dz[ok] = zk[ok] - k[ok].astype(np.float64)
    dzm[ok] = 1.0 - dz[ok]
    return k, dz, dzm


# --------------------------------------------------------------------------- check_max_iv
def _max_error_qc(f, rows, max_error):
    """``da_max_error_qc`` on ``rows`` (a boolean mask)."""
    err_max = f.error * max_error
    fail = rows & (np.abs(f.inv) > err_max)
    f.qc[fail] = FAILS_ERROR_MAX
    f.inv[fail] = 0.0
    return fail


def _check_uv(t, o):
    for v in ("u", "v"):
        f = t.fields[v]
        _max_error_qc(f, f.qc >= 0, o.max_error_uv)


def _check_q(t, o, depends):
    q = t.fields["q"]
    on = q.qc >= 0
    dep = np.zeros(len(q), dtype=bool)
    for v in depends:
        dep |= t.fields[v].qc == FAILS_ERROR_MAX
    q.qc[on & dep] = FAILS_ERROR_MAX
    q.inv[on & dep] = 0.0
    _max_error_qc(q, on & ~dep, o.max_error_q)


def _check_surface(t, o):
    _check_uv(t, o)
    for v, me in (("t", o.max_error_t), ("p", o.max_error_p)):
        f = t.fields[v]
        _max_error_qc(f, f.qc >= 0, me)
    _check_q(t, o, ("t", "p"))


def _check_uvtq(t, o):
    _check_uv(t, o)
    f = t.fields["t"]
    _max_error_qc(f, f.qc >= 0, o.max_error_t)
    _check_q(t, o, ("t",))


# --------------------------------------------------------------------------- surface
def _mo_correction(ho, po, to, qo, hm, pm, tm, qm, um, vm, roughness, lm):
    correc = 1.0
    hdif = hm - ho
    if hdif <= 10.0:
        return correc
    zint = roughness
    if zint < 0.0001:
        zint = 0.0001
    thm = tm * lm.pow_(1000.0 / (pm / 100.0), RCP)
    tho = to * lm.pow_(1000.0 / (po / 100.0), RCP)
    tvm = tm * (1.0 + 0.608 * qm)
    tvo = to * (1.0 + 0.608 * qo)
    thvm = tvm * lm.pow_(1000.0 / (pm / 100.0), RCP)
    thvo = tvo * lm.pow_(1000.0 / (po / 100.0), RCP)
    za = hm - ho
    if za > 40.0:
        za = 40.0
    va2 = um * um + vm * vm
    vc2 = 4.0 * max(tho - thm, 0.0)
    v2 = va2 + vc2
    rib = (GRAVITY * za / thm) * (thvm - thvo) / v2
    hll = abs(rib * lm.log(za / zint))
    correc = 1.0
    if rib < 0.0 and hll > 1.5:
        if zint < 0.2:
            correc = 1.000 + 0.320 * lm.pow_(zint, 0.200)
        else:
            correc = 1.169 + 0.315 * zint
        # log(40.0/0.05) and log(40.0) are folded at compile time
        winfac = 0.5 * (lm.log(za / 0.05) / _LOG_800 + lm.log(za) / _LOG_40)
        correc = correc * winfac
    return correc


# gfortran folds log(40.0/0.05) and log(40.0) at compile time (MPFR, correctly rounded)
_LOG_800 = crmath.log(40.0 / 0.05)
_LOG_40 = crmath.log(40.0)


def _intpsfc_prs(ho, po, hm, tm, qm, to, qo, lm):
    tvm = tm * (1.0 + 0.608 * qm)
    if to is not None and qo is not None:
        tvo = to * (1.0 + 0.608 * qo)
    elif to is not None:
        tvo = to
    else:
        tvo = tvm
    tv = 0.5 * (tvm + tvo)
    dz = hm - ho
    arg = dz * GRAVITY / GAS_CONSTANT
    arg = arg / tv
    return po * lm.exp(-arg)


def _intpsfc_tem(ho, po, to, height, pressure, temp, lm):
    prs_mb = pressure / 100.0
    kte = len(prs_mb)
    k100 = k200 = 0
    for jk in range(1, kte):
        if prs_mb[0] - prs_mb[jk] >= 100.0:
            k100 = jk
            break
    for jk in range(1, kte):
        if prs_mb[0] - prs_mb[jk] >= 200.0:
            k200 = jk
            break
    if k100 <= 0 or k200 <= 0 or k200 < k100:
        return MISSING_R
    th100 = temp[k100] * lm.pow_(1000.0 / prs_mb[k100], RCP)
    th200 = temp[k200] * lm.pow_(1000.0 / prs_mb[k200], RCP)
    th_obs = to * lm.pow_(1000.0 / (po / 100.0), RCP)
    dth_21 = th100 - th200
    dhe_sfc1 = height[k100] - height[0]
    dhe_obs1 = height[k100] - ho
    dhe_12 = height[k200] - height[k100]
    dhe_sfc2 = height[k200] - height[0]
    dhe_obs2 = height[k200] - ho
    th_sfc_int = th100 + (dth_21 / dhe_12) * dhe_sfc1
    th_obs_int = th100 + (dth_21 / dhe_12) * dhe_obs1
    dth_obs = th_obs_int - th_obs
    dth_sfc = (dhe_sfc2 / dhe_obs2) * dth_obs
    th_sfc = th_sfc_int - dth_sfc
    return th_sfc * lm.pow_(prs_mb[0] / 1000.0, RCP)


def _obs_sfc_correction(t, n, xb, lm):
    """``da_obs_sfc_correction`` (``psfc_from_slp = .false.``) on report ``n``."""
    w = tuple(a[n] for a in _w(t.info))
    rough = float(_hcol(xb["rough"], *w))
    pressure = _hcol(xb["p"], *w)
    height = _hcol(xb["h"], *w)
    t_mdl = _hcol(xb["t"], *w)
    q_mdl = _hcol(xb["q"], *w)
    u_mdl = _hcol(xb["u"], *w)
    v_mdl = _hcol(xb["v"], *w)
    F = t.fields
    ho = t.coords["h"][n]
    to = F["t"].inv[n]
    qo = F["q"].inv[n]
    if F["p"].inv[n] < 1.0:
        F["p"].qc[n] = MISSING_DATA
    po = F["p"].inv[n]
    if abs(po - MISSING_R) < 1.0 or abs(ho - MISSING_R) < 1.0:
        return
    if not ho < height[0]:
        return
    um, vm, tm, pm, qm, hm = u_mdl[0], v_mdl[0], t_mdl[0], pressure[0], q_mdl[0], height[0]
    if abs(F["u"].inv[n] - MISSING_R) > 1.0 and abs(F["v"].inv[n] - MISSING_R) > 1.0:
        tt = tm if abs(to - MISSING_R) < 1.0 else to
        correc = _mo_correction(ho, po, tt, qo, hm, pm, tm, qm, um, vm, rough, lm)
        for v in ("u", "v"):
            F[v].inv[n] = correc * F[v].inv[n]
            if abs(correc - 1.0) > 0.0:
                F[v].qc[n] = SURFACE_CORRECTION
    if F["p"].qc[n] >= 0:
        if abs(to - MISSING_R) > 1.0 and abs(qo - MISSING_R) > 1.0:
            val = _intpsfc_prs(ho, po, hm, tm, qm, to, qo, lm)
        elif abs(to - MISSING_R) > 1.0:
            val = _intpsfc_prs(ho, po, hm, tm, qm, to, None, lm)
        else:
            val = _intpsfc_prs(ho, po, hm, tm, qm, None, None, lm)
        F["p"].inv[n] = val
        F["p"].qc[n] = SURFACE_CORRECTION
    if abs(F["t"].inv[n] - MISSING_R) > 1.0:
        F["t"].inv[n] = _intpsfc_tem(ho, po, to, height, pressure, t_mdl, lm)
        F["t"].qc[n] = SURFACE_CORRECTION


def _surface(t, ob, xb, o, lm):
    """``da_get_innov_vector_{synop,metar,ships,buoy,sonde_sfc}``,
    ``sfc_assi_options = 1``."""
    n_rep = t.nlocal
    zk = np.full(n_rep, MISSING_R)
    h = t.coords["h"]
    W = _w(t.info)
    F = t.fields
    for n in range(n_rep):
        w = tuple(a[n] for a in W)
        if h[n] > MISSING_R:
            v_h = _hcol(xb["h"], *w)
            hd = v_h[0] - h[n]
            if abs(hd) <= o.max_stheight_diff:
                if h[n] < v_h[0]:
                    zk[n] = 1.0 + 1.0e-6
                    _obs_sfc_correction(t, n, xb, lm)
                else:
                    zk[n] = to_zk(h[n], v_h, False)
        elif ob["p"][n] > 1.0:
            v_p = _hcol(xb["p"], *w)
            zk[n] = to_zk(ob["p"][n], v_p, True)
            if zk[n] < 0.0:
                F["p"].inv[n] = MISSING_R
                F["p"].qc[n] = MISSING_DATA
                zk[n] = 1.0 + 1.0e-6
    t.extra["zk"] = zk
    geo = Geometry.of(t, xb["p"].shape[2])
    for v in ("u", "v", "t", "q", "p"):
        F[v].qc[zk < 0.0] = MISSING_DATA
    model = {v: interp_lin_3d(xb[v], geo) for v in ("u", "v", "t", "q", "p")}
    for v in ("u", "v"):
        ok = (ob[v] > MISSING_R) & (F[v].qc >= 0)
        F[v].inv[:] = np.where(ok, F[v].inv - model[v], 0.0)
    ok = F["p"].qc >= 0
    F["p"].inv[:] = np.where(ok, F["p"].inv - model["p"], 0.0)
    ok = (ob["t"] > 0.0) & (F["t"].qc >= 0)
    F["t"].inv[:] = np.where(ok, F["t"].inv - model["t"], 0.0)
    ok = (ob["q"] > 0.0) & (F["q"].qc >= 0)
    qsrc = ob["q"] if t.name == "synop" else F["q"].inv
    F["q"].inv[:] = np.where(ok, qsrc - model["q"], 0.0)
    if o.check_max_iv:
        _check_surface(t, o)
    return geo


# --------------------------------------------------------------------------- upper air
def _columns(t, xb, names):
    W = _w(t.info)
    return {v: _hcol(xb[v], *W) for v in names}


def _upper(t, ob, xb, o, *, kind):
    """sound, airep, pilot, profiler, geoamv, polaramv."""
    off = t.offsets
    lev = len(t.fields["u"])
    zk = np.full(lev, MISSING_R)
    F = t.fields
    cols = _columns(t, xb, ("h", "p") if kind != "amv" else ("p",))
    p = t.coords["p"]
    h = t.coords.get("h")
    for n in range(t.nlocal):
        for lv in range(off[n], off[n + 1]):
            if p[lv] > 1.0:
                zk[lv] = to_zk(p[lv], cols["p"][n], True)
            elif kind == "sound" and h[lv] > 0.0:
                zk[lv] = to_zk(h[lv], cols["h"][n], False)
            elif kind == "pilot" and h[lv] > MISSING_R:
                zk[lv] = to_zk(h[lv], cols["h"][n], False)
    t.extra["zk"] = zk
    geo = Geometry.of(t, xb["p"].shape[2])
    fields = [v for v in ("u", "v", "t", "q") if v in F]
    bad = zk < 0.0
    for v in fields:
        F[v].qc[bad] = MISSING_DATA
    model = {v: interp_lin_3d(xb[v], geo) for v in fields}
    reset = t.name not in ("profiler", "polaramv")
    for v in fields:
        ok = (ob[v] > MISSING_R) & (F[v].qc >= 0)
        if reset:
            F[v].inv[:] = np.where(ok, ob[v] - model[v], 0.0)
        else:
            F[v].inv[ok] = ob[v][ok] - model[v][ok]
    if o.check_max_iv:
        if "t" in F:
            _check_uvtq(t, o)
        else:
            _check_uv(t, o)
    return geo


def _qscat(t, ob, xb, o):
    zk = np.full(t.nlocal, MISSING_R)
    h = t.coords["h"]
    cols = _columns(t, xb, ("h",))
    for n in range(t.nlocal):
        if h[n] > MISSING_R:
            zk[n] = to_zk(h[n], cols["h"][n], False)
            if zk[n] < 1.0:
                zk[n] = 1.0
    t.extra["zk"] = zk
    geo = Geometry.of(t, xb["p"].shape[2])
    F = t.fields
    for v in ("u", "v"):
        F[v].qc[zk < 0.0] = MISSING_DATA
    for v in ("u", "v"):
        m = interp_lin_3d(xb[v], geo)
        ok = (ob[v] > MISSING_R) & (F[v].qc >= 0)
        F[v].inv[ok] = ob[v][ok] - m[ok]
    if o.check_max_iv:
        _check_uv(t, o)
    return geo


# --------------------------------------------------------------------------- gpspw
def _gpspw_dpw(w, elv, xb):
    """The station-height correction's pieces (``da_get_innov_vector_gpspw``):
    ``(branch, model_terr, [(k, rho_k, q_k, dz_k)])`` with 0-based ``k``;
    branch 1 is ``obs_terr <= model_terr`` (``dz`` is ``obs - model``)."""
    obs_terr = elv
    model_terr = float(_hcol(xb["terr"], *w))
    if obs_terr <= model_terr:
        q1 = float(_hcol(xb["q"][:, :, 0], *w))
        rho1 = float(_hcol(xb["rho"][:, :, 0], *w))
        return 1, model_terr, [(0, rho1, q1, obs_terr - model_terr)]
    kte = xb["q"].shape[2]
    z = [float(_hcol(xb["hf"][:, :, 0], *w))]
    terms = []
    for k in range(kte):
        if z[k] >= obs_terr:
            break
        z.append(float(_hcol(xb["hf"][:, :, k + 1], *w)))
        qk = float(_hcol(xb["q"][:, :, k], *w))
        rk = float(_hcol(xb["rho"][:, :, k], *w))
        dz = (z[k + 1] - z[k]) if z[k + 1] <= obs_terr else (obs_terr - z[k])
        terms.append((k, rk, qk, dz))
    return 2, model_terr, terms


def _gpspw(t, ob, xb, o):
    t.extra["zk"] = np.full(t.nlocal, MISSING_R)
    geo = Geometry.of(t, xb["p"].shape[2])
    F = t.fields["tpw"]
    W = _w(t.info)
    model_tpw = _hcol(xb["tpw"], *W)
    F.inv[:] = 0.0
    for n in range(t.nlocal):
        if ob["tpw"][n] > MISSING_R and F.qc[n] >= 0:
            w = tuple(a[n] for a in W)
            branch, _, terms = _gpspw_dpw(w, float(t.info["elv"][n]), xb)
            if branch == 1:
                _, rho1, q1, dz = terms[0]
                dpw = rho1 * q1 * dz
            else:
                dpw = 0.0
                for _, rk, qk, dz in terms:
                    dpw = dpw + rk * qk * dz
            F.inv[n] = ob["tpw"][n] - model_tpw[n] + 0.1 * dpw
    if o.check_max_iv:
        _max_error_qc(F, F.qc >= 0, o.max_error_pw)
    return geo


# --------------------------------------------------------------------------- gpsref
QCFLAG_PCNT_BELOW, QCFLAG_PCNT_MIDDLE, QCFLAG_PCNT_ABOVE = -31, -32, -33
QCFLAG_DNDZ, QCFLAG_DNDZ2, QCFLAG_CUTOFF = -34, -35, -36


def _gpsref(t, ob, xb, o, lm):
    off = t.offsets
    h = t.coords["h"]
    F = t.fields
    ref, pf = F["ref"], F["p"]
    zk = np.full(len(ref), MISSING_R)
    cols = _columns(t, xb, ("h", "p"))
    kte = xb["p"].shape[2]
    ob = dict(ob)
    ob["p"] = ob["p"].copy()
    for n in range(t.nlocal):
        v_h, v_p = cols["h"][n], cols["p"][n]
        for lv in range(off[n], off[n + 1]):
            if h[lv] > 0.0:
                zk[lv] = to_zk(h[lv], v_h, False)
            if zk[lv] < 0.0:
                ref.qc[lv] = MISSING_DATA
        for lv in range(off[n], off[n + 1]):
            kk = int(zk[lv])
            if kk >= 1 and kk + 1 <= kte:
                dz = zk[lv] - float(kk)
                dzm = 1.0 - dz
                pf.inv[lv] = v_p[kk - 1] * dzm + v_p[kk] * dz
                ob["p"][lv] = pf.inv[lv]
                pf.qc[lv] = -5
    t.extra["zk"] = zk
    geo = Geometry.of(t, xb["p"].shape[2])
    model_ref = np.array([lm.exp(x) for x in interp_lin_3d(xb["reflog"], geo)])
    ref.inv[:] = 0.0
    ok = (ob["ref"] > MISSING_R) & (ref.qc != MISSING_DATA)
    ref.inv[ok] = ob["ref"][ok] - model_ref[ok]
    _qc_gpsref(t, ob["ref"], xb, geo, o, lm)
    return geo


def _qc_gpsref(t, obref, xb, geo, o, lm):
    """``da_qc_gpsref`` (first outer loop)."""
    off = t.offsets
    lev = t.info["levels"]
    h = t.coords["h"]
    f = t.fields["ref"]
    model_ref = np.where(f.qc != MISSING_DATA, obref - f.inv, 0.0)
    if o.check_max_iv:
        thr = f.error * o.max_error_ref
        f.qc[(f.qc >= 0) & (np.abs(f.inv) > thr)] = FAILS_ERROR_MAX
    if o.gpsref_qc_dndz_opt == 1 or o.gpsref_qc_dndz2_opt == 1:
        for n in range(t.nlocal):
            if lev[n] <= 2:
                continue
            s = off[n]
            top = 0
            for k in range(lev[n]):
                if model_ref[s + k] > 0.0:
                    top = k + 1
            if top <= 2:
                continue
            hh, ob_, mo = h[s:s + top], obref[s:s + top], model_ref[s:s + top]
            zd = np.empty(top)
            zd[0] = hh[1] - hh[0]
            zd[top - 1] = hh[top - 1] - hh[top - 2]
            zd[1:top - 1] = hh[2:top] - hh[0:top - 2]
            zd = zd * 0.001

            def deriv(a):
                d = np.empty(top)
                d[0] = (a[1] - a[0]) / zd[0]
                d[top - 1] = (a[top - 1] - a[top - 2]) / zd[top - 1]
                d[1:top - 1] = (a[2:top] - a[0:top - 2]) / zd[1:top - 1]
                return d

            dno, dnm = deriv(ob_), deriv(mo)
            qc = f.qc[s:s + top]
            if o.gpsref_qc_dndz_opt == 1:
                bad = (qc != MISSING_DATA) & ((dno < o.gpsref_qc_dndz_thresh) | (dnm < o.gpsref_qc_dndz_thresh))
                qc[bad] = QCFLAG_DNDZ
            if o.gpsref_qc_dndz2_opt == 1:
                d2o, d2m = deriv(dno), deriv(dnm)
                th = o.gpsref_qc_dndz2_thresh
                bad = (qc != MISSING_DATA) & ((np.abs(d2o) > th) | (np.abs(d2m) > th))
                qc[bad] = QCFLAG_DNDZ2
    if o.gpsref_qc_gsi_opt == 1:
        model_t = interp_lin_3d(xb["t"], geo)
        cutoff = stddev = 0.0  # Fortran locals carry over between levels
        for n in range(t.nlocal):
            g_lat = lm.cos(RADIAN * float(t.info["lat"][n]))
            for lv in range(off[n], off[n + 1]):
                g = h[lv]
                mt = model_t[lv]
                gsi_t = 0.5 if mt <= 240.0 else 0.001 * mt * mt - 0.455 * mt + 52.075
                if 0.0 <= g < 4000.0:
                    cutoff, stddev = 1.0, 1.0 + 2.5 * g_lat
                elif 4000.0 <= g < 6000.0:
                    cutoff = 3.0
                    stddev = (6000.0 - g) / 2000.0 * (1.0 + 2.5 * g_lat) + (g - 4000.0) / 2000.0 * gsi_t
                elif 6000.0 <= g < 9000.0:
                    cutoff, stddev = 3.0, gsi_t
                elif 9000.0 <= g < 11000.0:
                    cutoff = 3.0
                    stddev = (11000.0 - g) / 2000.0 * gsi_t + (g - 9000.0) / 2000.0 * (0.25 + 0.5 * g_lat)
                elif 11000.0 <= g < 30000.0:
                    cutoff, stddev = 3.0, 0.25 + 0.5 * g_lat
                elif g >= 30000.0:
                    cutoff = 0.0
                if f.qc[lv] != MISSING_DATA and abs(f.inv[lv]) >= stddev * cutoff:
                    f.qc[lv] = QCFLAG_CUTOFF
    if o.gpsref_qc_pcnt_opt == 1:
        h1, h2 = o.gpsref_qc_pcnt_h1, o.gpsref_qc_pcnt_h2
        # transcribed as WRFDA has them: pcnt2 is the "above" value, used in
        # the middle band, and pcnt3 the "middle" value, used above h2
        pcnt1, pcnt2, pcnt3 = o.gpsref_qc_pcnt_below, o.gpsref_qc_pcnt_above, o.gpsref_qc_pcnt_middle
        with np.errstate(divide="ignore", invalid="ignore"):
            percnt = 2.0 * np.abs(f.inv / (obref + model_ref))
        on = f.qc != MISSING_DATA
        f.qc[on & (h < h1) & (percnt > pcnt1)] = QCFLAG_PCNT_BELOW
        f.qc[on & (h > h2) & (percnt > pcnt3)] = QCFLAG_PCNT_ABOVE
        f.qc[on & (h >= h1) & (h <= h2) & (percnt > pcnt2)] = QCFLAG_PCNT_MIDDLE
        names = [str(x) for x in t.info["name"]]
        if lev.max(initial=0) > 1:
            below = []
            for n in range(t.nlocal):
                hb = 0.0
                for lv in range(off[n], off[n + 1]):
                    if f.qc[lv] == QCFLAG_PCNT_BELOW:
                        hb = max(h[lv], hb)
                if hb != 0.0:
                    below.append((names[n], hb))
            for n in range(t.nlocal):
                for nm, hb in below:
                    if names[n] == nm:
                        for lv in range(off[n], off[n + 1]):
                            if h[lv] < hb and f.qc[lv] != MISSING_DATA:
                                f.qc[lv] = QCFLAG_PCNT_BELOW
                        break
        else:
            ids = [str(x) for x in t.info["id"]]
            for na in range(t.nlocal):
                if f.qc[off[na]] == QCFLAG_PCNT_BELOW:
                    for nb in range(t.nlocal):
                        if (ids[nb] == ids[na] and names[nb] == names[na]
                                and f.qc[off[nb]] != MISSING_DATA and h[off[nb]] <= h[off[na]]):
                            f.qc[off[nb]] = QCFLAG_PCNT_BELOW


# --------------------------------------------------------------------------- satem
THK = 0.5 * GAS_CONSTANT / GRAVITY


def tv_profile(xb, w):
    """``da_tv_profile``: ``pre_ma``, ``tv_ma`` indexed ``kts-1..kte+1`` (0..kte+1)."""
    kte = xb["p"].shape[2]
    pre_ma = np.full(kte + 2, np.nan)
    tv_ma = np.full(kte + 2, np.nan)
    i0, j0 = w[0], w[1]
    tv = xb["t"][i0:i0 + 2, j0:j0 + 2] * (1.0 + 0.61 * xb["q"][i0:i0 + 2, j0:j0 + 2])
    ww = (0, 0, *w[2:])
    pre_ma[1:kte + 1] = _hcol(xb["p"][i0:i0 + 2, j0:j0 + 2], *ww)
    tv_ma[1:kte + 1] = _hcol(tv, *ww)
    return pre_ma, tv_ma


def find_layer(pre, pre_ma, tv_ma, lm):
    """``da_find_layer`` (Fortran indices; mutates ``pre_ma``'s ghost cells).
    Returns ``(layer, tv, k0)``: ``k0`` is the bracketing level used."""
    ks, ke = 1, len(pre_ma) - 2
    if pre >= pre_ma[ks]:
        a, b, layer = ks, ks + 1, ks
        alpha = lm.log(pre_ma[a] / pre) / lm.log(pre_ma[a] / pre_ma[b])
        tv = tv_ma[a] * (1.0 - alpha) + tv_ma[b] * alpha
        pre_ma[ks - 1] = pre
        return layer, tv, a
    if pre <= pre_ma[ke]:
        a, b, layer = ke - 1, ke, ke + 1
        alpha = lm.log(pre_ma[a] / pre) / lm.log(pre_ma[a] / pre_ma[b])
        tv = tv_ma[a] * (1.0 - alpha) + tv_ma[b] * alpha
        pre_ma[ke + 1] = pre
        return layer, tv, a
    for k in range(ks, ke):
        if pre >= pre_ma[k + 1] and pre < pre_ma[k]:
            alpha = lm.log(pre_ma[k] / pre) / lm.log(pre_ma[k] / pre_ma[k + 1])
            return k + 1, tv_ma[k] * (1.0 - alpha) + tv_ma[k + 1] * alpha, k
    raise ValueError(f"da_find_layer: no layer for {pre!r}")


def thickness(pre_ma, tv_ma, tv1, tv2, layer1, layer2, pre1, pre2, lm):
    """``da_thickness``."""
    thk = THK * (tv1 * lm.log(pre_ma[layer1 - 1] / pre1) + tv2 * lm.log(pre2 / pre_ma[layer2]))
    p_tmp = {layer1: pre1}
    p_tmp[layer2 - 1] = pre2
    for k in range(layer2, layer1):
        p_tmp[k] = pre_ma[k]
    for k in range(layer2, layer1):
        thk = thk + THK * tv_ma[k] * lm.log(p_tmp[k - 1] / p_tmp[k + 1])
    return thk


def _satem(t, ob, xb, o, lm):
    t.extra["zk"] = np.full(len(t.fields["thickness"]), MISSING_R)
    geo = Geometry.of(t, xb["p"].shape[2])
    off = t.offsets
    F = t.fields["thickness"]
    p = t.coords["p"]
    ref_p = t.extra["ref_p"]
    W = _w(t.info)
    ptop = float(xb["ptop"])
    for n in range(t.nlocal):
        if off[n + 1] - off[n] < 1:
            continue
        pre_ma, tv_ma = tv_profile(xb, tuple(a[n] for a in W))
        layer2, tv2, _ = find_layer(ref_p[n], pre_ma, tv_ma, lm)
        pres2 = ref_p[n]
        for lv in range(off[n], off[n + 1]):
            F.inv[lv] = 0.0
            if p[lv] >= ptop:
                layer1, tv1, _ = find_layer(p[lv], pre_ma, tv_ma, lm)
                thk = thickness(pre_ma, tv_ma, tv1, tv2, layer1, layer2, p[lv], pres2, lm)
                if abs(ob["thickness"][lv] - MISSING_R) > 1.0 and F.qc[lv] != MISSING_DATA:
                    F.inv[lv] = ob["thickness"][lv] - thk
                pres2, layer2, tv2 = p[lv], layer1, tv1
            else:
                F.qc[lv] = ABOVE_MODEL_LID
    if o.check_max_iv:
        _max_error_qc(F, F.qc >= 0, o.max_error_thickness)
    lmask = xb["landmask"]
    for n in range(t.nlocal):
        if off[n + 1] == off[n]:
            continue  # no level: off[n] is the next report's first
        i0, j0 = W[0][n], W[1][n]
        if (lmask[i0, j0] != 0.0 or lmask[i0 + 1, j0] != 0.0 or lmask[i0, j0 + 1] != 0.0
                or lmask[i0 + 1, j0 + 1] != 0.0):
            F.inv[off[n]] = 0.0
    return geo


# --------------------------------------------------------------------------- driver
IMPLEMENTED = SURFACE_TYPES + ("qscat", "sound", "airep", "pilot", "profiler", "geoamv", "polaramv",
                               "gpspw", "gpsref", "satem")
_UPPER = {"sound": "sound", "airep": "sound", "pilot": "pilot", "profiler": "pilot",
          "geoamv": "amv", "polaramv": "amv"}


def innovations(obs: WrfdaObs, xb: dict, *, libm: str = "correctly-rounded",
                options: Options | None = None, use=None) -> WrfdaObs:
    """``da_get_innov_vector`` (first outer loop) for every type in ``obs``.

    ``obs`` is :func:`woof.da.var.obs_wrfda.read_obs_ascii`'s output (``inv``
    holds the observed values); ``xb`` holds the first guess in Fortran index
    order on one tile (``woof.da.var.oracle.read_dump`` of ``woof_xb.bin``
    has the layout): ``u v t q p h hf rho ref reflog`` ``(ix, jy, kz)``
    (``hf`` ``kz+1``), ``psfc terr rough tpw landmask`` ``(ix, jy)`` and
    ``ptop``. Returns a deep copy whose fields hold ``(O-B, qc, error)`` and
    whose ``extra["zk"]`` holds each level's ``zk`` (``extra["geometry"]`` the
    :class:`Geometry`, ``extra["lin"]`` the nonlinear types' linearisation).
    """
    o = options or Options()
    lm = _libm(libm)
    iv = copy.deepcopy(obs)
    xb = {k: (np.asarray(v, dtype=np.float64) if not np.isscalar(v) else v) for k, v in xb.items()}
    names = use if use is not None else [n for n in iv.types if n in IMPLEMENTED]
    for name in names:
        t = iv.types[name]
        ob = {k: f.inv.copy() for k, f in obs.types[name].fields.items()}
        if t.nlocal == 0:
            t.extra["zk"] = np.zeros(0)
            t.extra["geometry"] = Geometry.of(t, xb["p"].shape[2])
            continue
        if name in SURFACE_TYPES:
            geo = _surface(t, ob, xb, o, lm)
        elif name in _UPPER:
            geo = _upper(t, ob, xb, o, kind=_UPPER[name])
        elif name == "qscat":
            geo = _qscat(t, ob, xb, o)
        elif name == "gpspw":
            geo = _gpspw(t, ob, xb, o)
        elif name == "gpsref":
            geo = _gpsref(t, ob, xb, o, lm)
        elif name == "satem":
            geo = _satem(t, ob, xb, o, lm)
        else:
            raise NotImplementedError(f"{name}: not ported")
        t.extra["geometry"] = geo
    return iv


# --------------------------------------------------------------------------- Jo
def jo(iv: WrfdaObs, y: dict | None = None) -> dict:
    """``da_jo_and_grady_<type>``: per type ``{field: Jo_field}`` and the
    type's total (``"_"``), at ``H'(dx) = y`` (``None``: the background).

    ``re = inv - y`` where ``qc >= 0`` else 0; ``jo_field`` is
    ``-sum re * (-re / error**2)`` over reports then levels, halved.
    """
    out = {}
    for name in JO_ORDER:
        t = iv.types.get(name)
        if t is None or t.nlocal == 0:
            continue
        pd = np.repeat(np.asarray(t.info["proc_domain"], dtype=bool), t.info["levels"])
        per = {}
        for v in JO_FIELDS[name]:
            f = t.fields[v]
            yy = np.zeros(len(f)) if y is None else y[name][v]
            re = np.where(f.qc >= 0, f.inv - yy, 0.0)
            g = -re / (f.error * f.error)
            acc = 0.0
            for a, b, use_ in zip(re, g, pd):
                if use_:
                    acc = acc - a * b
            per[v] = 0.5 * acc
        tot = 0.0
        for k, v in enumerate(JO_FIELDS[name]):
            tot = per[v] if k == 0 else tot + per[v]
        per["_"] = tot
        out[name] = per
    return out


def jo_total(per_type: dict) -> float:
    """``jo%total``: the per-type totals in ``da_jo_and_grady``'s order."""
    tot = 0.0
    for name in JO_ORDER:
        if name in per_type:
            tot = tot + per_type[name]["_"]
    return tot
