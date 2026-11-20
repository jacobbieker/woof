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
    def of(cls, t) -> "Geometry":
        lev = np.asarray(t.info["levels"], dtype=np.int64)
        rep = np.repeat(np.arange(len(lev)), lev)
        zk = t.extra["zk"]
        k, dz, dzm = convert_zk(zk)
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


def convert_zk(zk: np.ndarray, kte: int = 40):
    """``da_convert_zk``: ``(k, dz, dzm)`` (k 1-based, 0 for missing)."""
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
    geo = Geometry.of(t)
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
    geo = Geometry.of(t)
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
    geo = Geometry.of(t)
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


# --------------------------------------------------------------------------- driver
IMPLEMENTED = SURFACE_TYPES + ("qscat", "sound", "airep", "pilot", "profiler", "geoamv", "polaramv")
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
            t.extra["geometry"] = Geometry.of(t)
            continue
        if name in SURFACE_TYPES:
            geo = _surface(t, ob, xb, o, lm)
        elif name in _UPPER:
            geo = _upper(t, ob, xb, o, kind=_UPPER[name])
        elif name == "qscat":
            geo = _qscat(t, ob, xb, o)
        else:
            raise NotImplementedError(f"{name}: not ported yet")
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
