"""WRF v4.7.1 fire spread (WRF-SFIRE physics), bit for bit, on CPU and GPU.

A transcription of ``phys/module_fr_fire_phys.F`` from WRF v4.7.1 (tag
v4.7.1, commit f52c197), the WRF tree's home of the WRF-SFIRE fuel and
spread physics:

* :func:`set_fire_params` -- ``set_fire_params``: the per-cell Rothermel
  (1972) coefficients of WRF's one-particle fuel categories, from the fuel
  category and surface fuel moisture, with the tables ``init_fuel_cats``
  loads (the pinned ``test/em_fire/namelist.fire`` or the module's
  compiled DATA);
* :func:`fire_ros` -- ``fire_ros``: the spread rate normal to the fire line,
  split into a base term, a wind term in ``wind . n`` and a slope term in
  ``grad(z) . n``, which is what the level set advances;
* :func:`fire_ros_total` -- the sum the level set takes
  (``module_fr_fire_core.F:2049-2051``): ``ros_base + ros_wind +
  fire_slope_factor * ros_slope``, floored at 0 under ``fire_grows_only``.

WRF builds this physics in real(4), so everything here is float32.  The
NumPy path is the reference: IEEE float32 arithmetic in WRF's operation
order, with glibc's ``powf``/``expf`` (``woof.core.noahmp_libm``).  The
CuPy path is ``kernels/fire_spread.cu`` through direct NVRTC
(``woof.fire._device``), the same operations with ``glibc_flt32.cuh``.
Both equal WRF's own Fortran, compiled at -O0 by
``tools/fire_ros_wrf471_oracle``, raw word for raw word
(``tests/test_fire_ros_wrf471_parity.py``), with one word held out: the
chaparral branch of ``fire_ros`` never assigns ``ros_base`` (WRF reads the
caller's value), and the port returns 0 there (PROVENANCE.md).

WRF's choices are kept, including the ones a physical model would not
make: ``fgi`` is treated as wet mass (``fuelloadm = (1 - bmst) fgi``), the
moisture damping is not clipped (above extinction ``r_0`` goes negative and
the level set's ``max(ros, 0)`` stops the fire), the normal is not
normalised, and the 6 m/s cap takes its excess out of the wind and slope
terms one after the other.  :mod:`woof.fire.rothermel` is the full
multi-class model.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from woof.fire import _device
from woof.fire import fuels as _fuels

#: The two table configurations ``init_fuel_cats`` can load.
TABLES = ("namelist", "defaults")
_FIELDS = ("fgi", "fgi_lh", "fueldepthm", "savr", "fuelmce", "fueldens",
           "st", "se", "weight", "ichap")


def r4(text: str) -> np.float32:
    """The real(4) value of a Fortran literal: the nearest float, ties even.

    ``np.float32(float(text))`` rounds twice (decimal to double to float)
    and can land one float off; this rounds the exact decimal once.
    """
    exact = Fraction(text.replace("d", "e").replace("D", "e"))
    guess = np.float32(float(exact))
    best = None
    for cand in (np.nextafter(guess, np.float32(-np.inf)), guess,
                 np.nextafter(guess, np.float32(np.inf))):
        err = abs(Fraction(float(cand)) - exact)
        even = (int(cand.view(np.uint32)) & 1) == 0
        key = (err, not even)
        if best is None or key < best[0]:
            best = (key, cand)
    return np.float32(best[1])


def r4_value(value) -> np.float32:
    """A runtime real(4) argument: any real number, rounded as a literal."""
    return r4(repr(float(value)))


_F = np.float32


@functools.lru_cache(maxsize=None)
def _scalars() -> dict[str, np.float32]:
    s = {k: r4(v) for k, v in _fuels.WRF471_SCALARS.items()}
    # init_fuel_cats: fuelheat = cmbcnst * 4.30e-04, in real(4).
    s["fuelheat"] = _F(s["cmbcnst"] * r4("4.30e-04"))
    return s


@functools.lru_cache(maxsize=None)
def _table_host(table: str) -> np.ndarray:
    """(field, category) float32 table, field order ``_FIELDS``."""
    if table not in TABLES:
        raise ValueError(f"table must be one of {TABLES}")
    columns = {k: list(v) for k, v in _fuels.WRF471_DATA.items()}
    if table == "namelist":
        for k, v in _fuels.WRF471_NAMELIST.items():
            columns[k][:len(v)] = v
    out = np.empty((len(_FIELDS), 54), dtype=np.float32)
    for f, name in enumerate(_FIELDS):
        out[f] = [r4(x) for x in columns[name]]
    return out


@functools.lru_cache(maxsize=None)
def _table_on(key, table: str):
    return key[0].asarray(_table_host(table))


def _table_device(xp, table: str):
    """The table on ``xp``'s current device, uploaded once per device."""
    return _table_on(_device.device_key(xp), table)


@functools.lru_cache(maxsize=None)
def _ksb() -> np.ndarray:
    """set_fire_params' ksb: fuel number (index) -> WRF category, 0 unused."""
    out = np.full(_fuels.WRF471_KSB_SIZE + 1, _fuels.WRF471_NO_FUEL_CAT,
                  dtype=np.int32)
    out[0] = -1
    for number, k in _fuels.WRF471_KSB.items():
        out[number] = k
    return out


@functools.lru_cache(maxsize=None)
def _ksb_on(key):
    return key[0].asarray(_ksb())


_xp = _device.array_namespace


def category_rows(nfuel_cat, *, table: str = "namelist"):
    """Table row (0-based) of each cell's fuel, -1 for no fuel.

    ``nfuel_cat`` is WRF's per-cell fuel number: 1-13 (Anderson), 101-204
    (Scott & Burgan) or any other number in 1..204 for no fuel; it is
    truncated to an integer as WRF's ``int(nfuel_cat(i,j))`` does.  A number
    outside 1..204 reads outside WRF's ksb array and is refused here; a
    category above the table's ``nfuelcats`` is WRF's "fuel category out of
    bounds" crash and raises the same way.  The compiled defaults declare
    ``nfuelcats = 53`` but put SB4 (204) at category 54, so WRF v4.7.1
    rejects SB4 in that configuration, and so does this.
    """
    xp = _xp(nfuel_cat)
    raw = xp.asarray(nfuel_cat)
    if raw.dtype.kind == "f":
        if not bool(xp.all(xp.isfinite(raw))):
            raise ValueError("nfuel_cat must be finite")
        raw = xp.trunc(raw)
    number = raw.astype(xp.int64)
    if bool(xp.any((number < 1) | (number > _fuels.WRF471_KSB_SIZE))):
        raise ValueError("nfuel_cat must lie in 1..204 (WRF's ksb map)")
    k = _ksb_on(_device.device_key(xp))[number]
    no_fuel = k == _fuels.WRF471_NO_FUEL_CAT
    if bool(xp.any(~no_fuel & (k > _fuels.WRF471_NFUELCATS[table]))):
        raise ValueError(
            "set_fire_params: fuel category out of bounds for the "
            f"{table!r} table (nfuelcats = {_fuels.WRF471_NFUELCATS[table]})")
    return xp.where(no_fuel, -1, k - 1).astype(xp.int32)


@dataclass(frozen=True)
class FireParams:
    """``set_fire_params`` output per cell, float32, WRF's names and units.

    ``r_0`` no-wind no-slope rate (ft min-1), ``phiwc`` wind coefficient
    ``C (beta/beta_op)^-E``, ``bbb`` wind exponent B, ``betafl`` packing
    ratio, ``ischap`` 1 for chaparral, ``fgip`` initial fuel mass (kg m-2),
    ``fuel_time`` burn-out time constant (s), ``iboros`` I_R x t_r (kJ m-2).
    """

    fuel_time: Any
    fgip: Any
    ischap: Any
    betafl: Any
    bbb: Any
    phiwc: Any
    r_0: Any
    iboros: Any


_PARAM_OUT = ("fuel_time", "fgip", "ischap", "betafl", "bbb", "phiwc", "r_0",
              "iboros")


def _expf(x):
    """glibc's ``expf`` over an array: the core of ``noahmp_libm.expf``
    (e_expf.c) in NumPy float64, the same operations in the same order;
    the few arguments on its special-case paths (|x| >= 88, inf, nan) go
    through the scalar transcription.  ``tests/test_fire_ros_wrf471_parity``
    holds the two equal word for word."""
    from woof.core import noahmp_libm as L
    x = np.asarray(x, dtype=np.float32)
    flat = x.reshape(-1)
    # The core runs over every element and the special-case elements are
    # then replaced, so its overflow on them is never seen.
    with np.errstate(over="ignore", invalid="ignore"):
        xd = flat.astype(np.float64)
        z = L._EXP2F_INVLN2_SCALED * xd
        kd = z + L._EXP2F_SHIFT
        ki = kd.view(np.uint64)
        kd = kd - L._EXP2F_SHIFT
        r = z - kd
        tab = L._EXP2F_TAB_U64[(ki % np.uint64(L._EXP2F_N)).astype(np.intp)]
        s = (tab + (ki << np.uint64(52 - 5))).view(np.float64)
        poly = L._EXP2F_POLY_SCALED
        zz = poly[0] * r + poly[1]
        r2 = r * r
        y = poly[2] * r + 1.0
        y = zz * r2 + y
        out = (y * s).astype(np.float32)
        top = (flat.view(np.uint32) >> np.uint32(20)) & np.uint32(0x7FF)
        special = top >= (np.float32(88.0).view(np.uint32) >> np.uint32(20))
        for i in np.flatnonzero(special):
            out[i] = L.expf(flat[i])
    return out.reshape(x.shape)


def _powf(x, y):
    from woof.core import noahmp_libm
    return noahmp_libm.powf_array(np.asarray(x, np.float32),
                                  np.asarray(y, np.float32))


def _set_fire_params_host(rows, fmc, tab, s) -> dict[str, np.ndarray]:
    """set_fire_params in float32 NumPy, WRF's operation order line by line."""
    no_fuel = rows < 0
    col = np.where(no_fuel, 0, rows)
    fgi, fgi_lh, depthm, savr, mce, dens, st, se, weight, ichap = (
        tab[f][col] for f in range(len(_FIELDS)))
    one = _F(1.0)
    fuel_time = weight / r4("0.85")
    lh = s["fuelmc_g_lh"]
    if lh > r4("0.3") and lh < r4("1.2"):
        fgip = fgi + (one - (lh - r4("0.3")) / r4("0.9")) * fgi_lh
    elif lh <= r4("0.3"):
        fgip = fgi + fgi_lh
    else:
        fgip = fgi.copy()
    bmst = fmc / (one + fmc)
    fuelloadm = (one - bmst) * fgip
    ft2 = _F(r4(".3048") * r4(".3048"))
    fuelload = (fuelloadm * ft2) * r4("2.205")
    fueldepth = depthm / r4("0.3048")
    betafl = fuelload / (fueldepth * dens)
    betaop = r4("3.348") * _powf(savr, r4("-0.8189"))
    qig = r4("250.") + r4("1116.") * fmc
    epsilon = _expf(r4("-138.") / savr)
    rhob = fuelload / fueldepth
    c = r4("7.47") * _expf(r4("-0.133") * _powf(savr, r4("0.55")))
    bbb = r4("0.02526") * _powf(savr, r4("0.54"))
    e = r4("0.715") * _expf(r4("-3.59e-4") * savr)
    phiwc = c * _powf(betafl / betaop, -e)
    rtemp2 = _powf(savr, r4("1.5"))
    gammax = rtemp2 / (r4("495.") + r4("0.0594") * rtemp2)
    a = one / (r4("4.774") * _powf(savr, r4("0.1")) - r4("7.27"))
    ratio = betafl / betaop
    gamma = (gammax * _powf(ratio, a)) * _expf(a * (one - ratio))
    wn = fuelload / (one + st)
    r1 = fmc / mce
    r1sq = r1 * r1
    etam = ((one - r4("2.59") * r1) + r4("5.11") * r1sq) - r4("3.52") * (r1sq * r1)
    etas = r4("0.174") * _powf(se, r4("-0.19"))
    ir = (((gamma * wn) * s["fuelheat"]) * etam) * etas
    k1 = _F(ft2 * r4("60."))
    k2 = _F(r4("60.") * r4("12.6"))
    iboros = (((ir * r4("1055.")) / k1) * r4("1.e-3")) * (k2 / savr)
    xifr = _expf((r4("0.792") + r4("0.681") * _powf(savr, r4("0.5")))
                 * (betafl + r4("0.1"))) / (r4("192.") + r4("0.2595") * savr)
    r_0 = (ir * xifr) / ((rhob * epsilon) * qig)
    out = dict(fuel_time=fuel_time, fgip=fgip, ischap=ichap, betafl=betafl,
               bbb=bbb, phiwc=phiwc, r_0=r_0, iboros=iboros)
    nofuel_value = dict(fuel_time=_F(r4("7.") / r4("0.85")), fgip=_F(0),
                        ischap=_F(0), betafl=_F(1), bbb=_F(1), phiwc=_F(0),
                        r_0=_F(0), iboros=_F(0))
    return {k: np.where(no_fuel, nofuel_value[k], v).astype(np.float32)
            for k, v in out.items()}


def set_fire_params(nfuel_cat, fmc_g=None, *, table: str = "namelist"
                    ) -> FireParams:
    """WRF v4.7.1 ``set_fire_params`` over a fire grid.

    ``nfuel_cat`` is the fuel number of each cell (see :func:`category_rows`)
    and ``fmc_g`` the surface fuel moisture (fraction), WRF's ``fuelmc_g``
    (0.08) when omitted.  ``table`` picks the categories ``init_fuel_cats``
    loads: ``"namelist"`` (the pinned ``test/em_fire/namelist.fire``, the 13
    Anderson categories) or ``"defaults"`` (the compiled DATA, which also
    carry the Scott & Burgan 40 as one-particle categories).
    """
    xp = _xp(nfuel_cat, fmc_g)
    s = _scalars()
    rows = category_rows(nfuel_cat, table=table)
    fmc = xp.asarray(s["fuelmc_g"] if fmc_g is None else fmc_g,
                     dtype=xp.float32)
    shape = np.broadcast_shapes(rows.shape, fmc.shape)
    if xp is np:
        rows, fmc = (np.broadcast_to(a, shape) for a in (rows, fmc))
        return FireParams(**_set_fire_params_host(
            rows, fmc, _table_host(table), s))
    rows = _device.device_array(xp, rows, xp.int32, shape)
    fmc = _device.device_array(xp, fmc, xp.float32, shape)
    out = {k: xp.empty(shape, dtype=xp.float32) for k in _PARAM_OUT}
    n = int(np.prod(shape))
    _device.launch("wrf_fire_params", n, (
        rows, fmc, np.int64(n), _table_device(xp, table), np.int32(54),
        s["fuelheat"], s["fuelmc_g_lh"], *(out[k] for k in _PARAM_OUT)))
    return FireParams(**out)


def _fmax(a, b):
    """gfortran MAX(a, b): b if b > a, else a (the first wins a tie).

    The tie rule decides the sign of a zero result, and the oracle sweep
    reaches it: ``max(0., -0.)`` at a calm or perpendicular wind in
    ``fire_advection = 1``, where the port equals WRF.  NaN arguments are
    outside the contract (gfortran does not promise which operand MAX
    returns then) and the inputs must be finite.
    """
    return np.where(b > a, b, a)


def _fmin(a, b):
    return np.where(b < a, b, a)


def _fire_ros_host(r_0, phiwc, bbb, betafl, ischap, vx, vy, dzdx, dzdy,
                   nvx, nvy, advection, ros_max, dtype, pow_, literal, tiny):
    """fire_ros in NumPy, ``dtype`` float32 (WRF) or float64 (full model).

    ``literal`` turns a WRF literal into a ``dtype`` constant; ``pow_`` is
    the C library's pow for ``dtype``.
    """
    L = literal
    zero = dtype(0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if advection:
            speed = np.sqrt(vx * vx + vy * vy) + tiny
            tanphi = np.sqrt(dzdx * dzdx + dzdy * dzdy) + tiny
            cor_wind = _fmax(zero, (vx * nvx + vy * nvy) / speed)
            cor_slope = _fmax(zero, (dzdx * nvx + dzdy * nvy) / tanphi)
        else:
            speed = vx * nvx + vy * nvy
            tanphi = dzdx * nvx + dzdy * nvy
            cor_wind = cor_slope = dtype(1)
        chap = ischap > zero
        spdms = _fmax(speed, zero)
        umid = _fmin(spdms, L("30.")) * L("196.850")
        phiw = pow_(umid, bbb) * phiwc
        tpos = tanphi > zero
        phis = np.where(tpos, (L("5.275") * pow_(betafl, L("-0.3")))
                        * (tanphi * tanphi), zero)
        rb = r_0 * L(".00508")
        rw = rb * phiw
        rs = rb * phis
        chap_back = L(".03333")
        chap_wind = _fmax(L("1.2974") * pow_(spdms, L("1.41")), chap_back)
        rb = np.where(chap, zero, rb)
        rw = np.where(chap, chap_wind, rw) * cor_wind
        rs = np.where(chap, zero, rs) * cor_slope
        excess = ((rb + rw) + rs) - ros_max
        over = excess > zero
        rw2 = rw - (excess * rw) / (rw + rs)
        rw = np.where(over, rw2, rw)
        rs2 = rs - (excess * rs) / (rw + rs)
        rs = np.where(over, rs2, rs)
    return (rb.astype(dtype), rw.astype(dtype), rs.astype(dtype))


def fire_ros(params: FireParams, vx, vy, dzdxf, dzdyf, propx, propy, *,
             fire_advection: int = 0, ros_max: float = 6.0):
    """WRF v4.7.1 ``fire_ros``: ``(ros_base, ros_wind, ros_slope)``, m s-1.

    ``vx``/``vy`` is the fire-grid wind (m s-1, at ``fire_wind_height``),
    ``dzdxf``/``dzdyf`` the terrain gradient, ``propx``/``propy`` the
    fire-line normal, used as given (WRF v4.7.1 sets ``scale = 1``).  With
    ``fire_advection = 0`` the wind and slope enter through their
    components along the normal; with 1, through their magnitudes times the
    positive part of the cosine to the normal.  Wind or slope against the
    normal contributes nothing, so a backing or flanking line spreads at
    ``ros_base``.  ``ros_max`` is WRF's fixed 6 m s-1 cap.  float32 in and
    out; CuPy inputs run the NVRTC kernel.

    On chaparral cells (``ischap``) WRF never assigns ``ros_base``; the port
    returns 0 there, so ``ros_wind`` carries the whole Clark et al. (2004)
    rate ``max(1.2974 U^1.41, 0.03333)``.  Inputs must be finite.
    """
    p = params
    return fire_ros_dispatch(p, vx, vy, dzdxf, dzdyf, propx, propy,
                             fire_advection=fire_advection,
                             ros_max=r4_value(ros_max), dtype=np.float32,
                             kernel="wrf_fire_ros", pow_=_powf, literal=r4)


def fire_ros_dispatch(p, vx, vy, dzdxf, dzdyf, propx, propy, *,
                      fire_advection, ros_max, dtype, kernel, pow_, literal):
    """``fire_ros`` in ``dtype`` on NumPy (the reference) or CuPy (NVRTC).

    Shared by :func:`fire_ros` (float32, WRF) and
    :func:`woof.fire.rothermel.rothermel_fire_ros` (float64).
    """
    xp = _xp(p.r_0, vx, vy, dzdxf, dzdyf, propx, propy)
    ins = (p.r_0, p.phiwc, p.bbb, p.betafl, p.ischap, vx, vy, dzdxf, dzdyf,
           propx, propy)
    shape = np.broadcast_shapes(*(np.shape(a) for a in ins))
    if xp is np:
        arrays = [np.broadcast_to(np.asarray(a, dtype), shape) for a in ins]
        return _fire_ros_host(*arrays, bool(fire_advection), ros_max, dtype,
                              pow_, literal, dtype(np.finfo(dtype).tiny))
    arrays = [_device.device_array(xp, a, dtype, shape) for a in ins]
    out = [xp.empty(shape, dtype=dtype) for _ in range(3)]
    n = int(np.prod(shape))
    _device.launch(kernel, n, (*arrays, np.int64(n),
                               np.int32(1 if fire_advection else 0),
                               dtype(ros_max), *out))
    return tuple(out)


def fire_ros_total(params: FireParams, vx, vy, dzdxf, dzdyf, propx, propy, *,
                   fire_advection: int = 0, fire_slope_factor: float = 1.0,
                   fire_grows_only: bool = True, ros_max: float = 6.0):
    """The level set's rate, ``module_fr_fire_core.F:2047-2051``, float32.

    ``rr = ros_base + ros_wind + fire_slope_factor * ros_slope``, and
    ``max(rr, 0)`` when ``fire_grows_only`` (WRF's default 1).
    """
    rb, rw, rs = fire_ros(params, vx, vy, dzdxf, dzdyf, propx, propy,
                          fire_advection=fire_advection, ros_max=ros_max)
    xp = _xp(rb)
    factor = r4_value(fire_slope_factor)
    if xp is np:
        rr = (rb + rw) + factor * rs
        if fire_grows_only:
            rr = _fmax(rr, _F(0.0))
        return rr.astype(np.float32)
    rr = xp.empty(rb.shape, dtype=xp.float32)
    n = int(rb.size)
    _device.launch("wrf_fire_ros_total", n, (rb, rw, rs, np.int64(n), factor,
                                         np.int32(1 if fire_grows_only else 0),
                                         rr))
    return rr


# --------------------------------------------------------------------------
# helpers the coupled model needs around fire_ros
# --------------------------------------------------------------------------

def category_field(nfuel_cat, name: str, *, table: str = "namelist"):
    """One per-category quantity gathered onto a grid of fuel numbers.

    ``name`` is a ``set_fire_params`` table field (``"weight"``,
    ``"fgi"``, ``"savr"`` ...), float32 as WRF holds it; no-fuel cells take
    category 14's value.
    """
    xp = _xp(nfuel_cat)
    if name not in _FIELDS:
        raise KeyError(f"unknown table field {name!r}; one of {_FIELDS}")
    rows = category_rows(nfuel_cat, table=table)
    rows = xp.where(rows < 0, _fuels.WRF471_NO_FUEL_CAT - 1, rows)
    return _table_device(xp, table)[_FIELDS.index(name)][rows]


def midflame_wind(wind, nfuel_cat):
    """Mid-flame wind from the 20 ft (6.096 m) wind: ``windrf * U``.

    ``windrf`` is the per-category wind reduction factor of Baughman &
    Albini (1980) in ``namelist.fire`` (Anderson 13 only; WRF v4.7.1 reads
    it but no longer applies it, interpolating the wind to
    ``fire_wind_height`` instead, and openwfm WRF-SFIRE multiplies the wind
    at ``fcwh`` by it).  No-fuel cells use 1e-7.
    """
    xp = _xp(wind, nfuel_cat)
    k = xp.asarray(nfuel_cat).astype(xp.int64)
    if bool(xp.any((k < 1) | (k > 14))):
        raise ValueError("windrf is defined for categories 1..14")
    windrf = xp.asarray(np.asarray(
        [_fuels.WRF_SFIRE_FUEL_CATEGORIES[i].windrf for i in range(1, 15)]))
    return xp.asarray(wind, dtype=xp.float64) * windrf[k - 1]
