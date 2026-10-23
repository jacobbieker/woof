"""Rothermel (1972) surface fire spread with the Albini (1976) corrections.

The heterogeneous-fuel spread model exactly as Andrews (2018) RMRS-GTR-371
Table 6 lists it and BehavePlus implements it (firelab/behave
``surfaceFuelbedIntermediates.cpp``, ``surfaceFireReactionIntensity.cpp``,
``surfaceFire.cpp``): six particle classes (1-h, 10-h, 100-h and dead
herbaceous; live herbaceous and live woody), surface-area weighting, the
Albini size-subclass weights for net load, the live moisture of
extinction, the dynamic live-to-dead herbaceous transfer of Scott & Burgan
(2005), and an optional wind limit -- Rothermel's ``U <= 0.9 I_R`` or the
Andrews, Cruz & Rothermel (2013) correction ``U <= 96.8 I_R^(1/3)``.
Neither is imposed by default, which is BehavePlus's default and the
recommendation of Andrews et al. (2013).

* :func:`rothermel_surface_fire` -- head-fire behaviour (rate, intensity,
  flame length, the fuel-bed intermediates the published tables print).
* :func:`rothermel_fire_params` / :func:`rothermel_fire_ros` -- the same
  model in the form WRF's level set consumes: the coefficients
  ``set_fire_params`` stores (``r_0``, ``phiwc``, ``bbb``, ``betafl``) and
  WRF v4.7.1's ``fire_ros`` evaluated on them in float64, so a backing or
  flanking line spreads at ``r_0`` and the wind and slope terms enter
  through ``wind . n`` and ``grad(z) . n``.  :mod:`woof.fire.wrf_sfire` is
  WRF's own float32 one-particle version of both, bit for bit.

Units.  Rothermel's equations are dimensional in US customary units (lb,
ft, min, Btu); every function converts its SI inputs once on the way in,
evaluates the published equations in the published units, and converts
once on the way out.  Inputs and outputs are SI (kg m-2, m, m-1, m s-1,
kW m-2, kW m-1, kJ m-2, s; moisture as a fraction of oven-dry mass) except
the WRF-form coefficients, which keep WRF's units.

CPU and GPU.  Everything is float64.  The NumPy path is the reference: the
operations of ``kernels/fire_spread.cu`` in the same order, each class sum
added in class order from 0, with the C library's ``exp`` and ``pow``
(``woof.core.host_libm``; NumPy's own vector loops round differently on an
AVX-512 host).  The CuPy path is that kernel through direct NVRTC with
``-fmad=false --ftz=false`` and ``glibc_flt64.cuh``, one thread per cell
over the whole fire mesh, reading fuel tables uploaded to the device once
per process.  The two agree raw word for raw word
(``tests/test_fire_rothermel.py``).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np

from woof.fire import _device
from woof.fire import fuels as _fuels
from woof.fire.fuels import (
    BTU_PER_FT2, BTU_PER_FT2_MIN, BTU_PER_FT_S, BTU_PER_LB, FT, FT_PER_MIN,
    LB_PER_FT2, LB_PER_FT3, PER_FT, FuelModel, fuel_model,
)

#: m s-1 -> ft min-1, spelled as the kernel spells it (fr_div(60.0, 0.3048)).
_MS_TO_FT_MIN = 60.0 / 0.3048
#: Wind-limit choices accepted by :func:`rothermel_surface_fire`.
WIND_LIMITS = (None, "rothermel1972", "andrews2013")
_WIND_LIMIT_CODE = {None: 0, "rothermel1972": 1, "andrews2013": 2}
#: The defined non-burnable model numbers (NB1-NB3, NB8, NB9): zero spread.
NON_BURNABLE = frozenset(_fuels.NON_BURNABLE_FUEL_MODELS)


_xp = _device.array_namespace


def _exp(x):
    from woof.core import host_libm
    return host_libm.exp(x)


def _pow(x, y):
    from woof.core import host_libm
    return host_libm.power(x, y)


# --------------------------------------------------------------------------
# standard fuel model table, uploaded once per namespace
# --------------------------------------------------------------------------

# Class order everywhere: 0 1-h, 1 10-h, 2 100-h, 3 dead herbaceous (dead);
# 4 live herbaceous, 5 live woody (live).
_FIELDS = ("w1", "w10", "w100", "wherb", "wwoody", "s1", "s10", "s100",
           "sherb", "swoody", "depth", "mxd", "hd", "hl", "rhop", "st", "se",
           "dynamic")
_HEAD_OUT = ("ros", "r0", "phi_wind", "phi_slope", "effective_wind", "ir",
             "hpa", "fli", "flame", "residence", "sigma", "bulk_density",
             "beta", "rpr", "mxl", "wind_c", "wind_b", "wind_e")


def _us_row(m: FuelModel) -> tuple[float, ...]:
    """A :class:`FuelModel` in the US units the equations are written in."""
    return (m.load_1h / LB_PER_FT2, m.load_10h / LB_PER_FT2,
            m.load_100h / LB_PER_FT2, m.load_live_herb / LB_PER_FT2,
            m.load_live_woody / LB_PER_FT2,
            m.sav_1h / PER_FT, m.sav_10h / PER_FT, m.sav_100h / PER_FT,
            m.sav_live_herb / PER_FT, m.sav_live_woody / PER_FT,
            m.depth / FT, m.moisture_extinction_dead,
            m.heat_content_dead / BTU_PER_LB, m.heat_content_live / BTU_PER_LB,
            m.particle_density / LB_PER_FT3, m.total_mineral_content,
            m.effective_mineral_content, float(m.dynamic))


def _standard_table(xp) -> tuple[Any, Any]:
    """(columns, lookup) on ``xp``'s current device, uploaded once each."""
    return _standard_table_on(_device.device_key(xp))


@functools.lru_cache(maxsize=None)
def _standard_table_on(key) -> tuple[Any, Any]:
    """(columns, lookup) for one (namespace, device) key.

    ``columns`` is (nfield, nmodel) in US units, one column per standard
    or non-burnable model; ``lookup[number]`` is that model's column, -1
    for any number that is not a model.  The CuPy copy is a few kB and
    lives on the device current at first use.
    """
    models = sorted({**_fuels.STANDARD_FUEL_MODELS,
                     **_fuels.NON_BURNABLE_FUEL_MODELS}.values(),
                    key=lambda m: m.number)
    rows = np.asarray([_us_row(m) for m in models], dtype=np.float64)
    lookup = np.full(256, -1, dtype=np.int64)
    for row, m in enumerate(models):
        lookup[m.number] = row
    xp = key[0]
    return xp.asarray(np.ascontiguousarray(rows.T)), xp.asarray(lookup)


@functools.lru_cache(maxsize=64)
def _single_table_on(key, model: FuelModel):
    row = np.asarray(_us_row(model), dtype=np.float64)[:, None]
    return key[0].asarray(np.ascontiguousarray(row))


def _single_table(xp, model: FuelModel):
    """A one-column table for a single (possibly custom) model, cached
    per (namespace, device)."""
    return _single_table_on(_device.device_key(xp), model)


def _model_index(xp, fuel):
    """(columns, index): the device table and each cell's column in it."""
    if isinstance(fuel, (FuelModel, str)) or (
            np.ndim(fuel) == 0 and not hasattr(fuel, "__cuda_array_interface__")):
        return _single_table(xp, fuel_model(fuel)), xp.zeros((), dtype=xp.int32)
    columns, lookup = _standard_table(xp)
    numbers = xp.asarray(fuel)
    if numbers.dtype.kind == "f":
        # A resampled float grid holding 101.5 is not GR1; refuse it.
        if not bool(xp.all(xp.isfinite(numbers) & (numbers == xp.trunc(numbers)))):
            raise ValueError("fuel model numbers must be finite integers")
        numbers = numbers.astype(xp.int64)
    elif numbers.dtype.kind not in "iu":
        raise TypeError(f"fuel model numbers must be integers, got {numbers.dtype}")
    inside = (numbers >= 0) & (numbers < lookup.size)
    index = xp.where(inside, lookup[xp.where(inside, numbers, 0)], -1)
    if bool(xp.any(index < 0)):
        bad = numbers[index < 0]
        bad = np.unique(bad.get() if xp is not np else bad)
        raise KeyError(f"unknown fuel model numbers {bad.tolist()}")
    return columns, index.astype(xp.int32)


def _inputs(xp, fuel, moistures, curing, extra=()):
    """The table, each cell's column, and every per-cell input, broadcast."""
    columns, index = _model_index(xp, fuel)
    if curing is None:
        cure = xp.asarray(-1.0, dtype=xp.float64)       # moisture decides
    else:
        cure = xp.asarray(curing, dtype=xp.float64)
        if bool(xp.any(~xp.isfinite(cure))):
            raise ValueError("curing must be finite")
        cure = xp.where(cure < 0.0, 0.0, xp.where(cure > 1.0, 1.0, cure))
    fields = [xp.asarray(v, dtype=xp.float64) for v in (*moistures, *extra)]
    shape = np.broadcast_shapes(index.shape, cure.shape,
                                *(f.shape for f in fields))
    # The kernels need contiguous full-shape buffers; the NumPy reference
    # reads broadcast views.
    expand = ((lambda a: xp.broadcast_to(a, shape).copy()) if xp is not np
              else (lambda a: np.broadcast_to(a, shape)))
    return columns, expand(index), expand(cure), [expand(f) for f in fields], shape


# --------------------------------------------------------------------------
# the NumPy reference, operation for operation with fr_fuel_bed
# --------------------------------------------------------------------------

def _div0(num, den, threshold):
    """``num / den`` where ``den > threshold``, else 0, without a warning."""
    ok = den > threshold
    return np.where(ok, num / np.where(ok, den, 1.0), 0.0)


def _fuel_bed_host(columns, index, m1, m10, m100, mh, mw, curing):
    T = {k: columns[j][index] for j, k in enumerate(_FIELDS)}
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cured_m = np.where(mh < 0.30, 1.0,
                           np.where(mh <= 1.20, 1.333 - 1.11 * mh, 0.0))
        cured = np.where(curing < 0.0, cured_m, curing)
        cured = np.where(T["dynamic"] > 0.5, cured, 0.0)
        wherb = T["wherb"]
        wdh = wherb * cured
        w = [T["w1"], T["w10"], T["w100"], wdh, wherb - wdh, T["wwoody"]]
        s = [T["s1"], T["s10"], T["s100"], T["sherb"], T["sherb"], T["swoody"]]
        m = [m1, m10, m100, m1, mh, mw]
        hd, hl = T["hd"], T["hl"]
        depth, mxd, rhop = T["depth"], T["mxd"], T["rhop"]
        st, se = T["st"], T["se"]

        area = [(w[j] * s[j]) / rhop for j in range(6)]
        ex138 = [_exp(-138.0 / s[j]) for j in range(6)]
        a_dead = ((area[0] + area[1]) + area[2]) + area[3]
        a_live = area[4] + area[5]
        f = [_div0(area[j], a_dead if j < 4 else a_live, 1.0e-7)
             for j in range(6)]
        a_total = a_dead + a_live
        f_dead = _div0(a_dead, a_total, 0.0)
        f_live = np.where(a_total > 0.0, 1.0 - f_dead, 0.0)

        def subclass(x):
            return np.where(x >= 1200.0, 0, np.where(x >= 192.0, 1, np.where(
                x >= 96.0, 2, np.where(x >= 48.0, 3, np.where(x >= 16.0, 4, 5)))))

        sub = [subclass(s[j]) for j in range(6)]
        zero = np.zeros(np.shape(index))
        wn_d = wn_l = h_d = h_l = se_d = se_l = zero
        m_d = m_l = s_d = s_l = load = bsum = zero
        for j in range(6):
            g = zero
            for q in range(6):
                if (q < 4) == (j < 4):
                    g = np.where(sub[q] == sub[j], g + f[q], g)
            g = np.where(sub[j] < 5, g, 0.0)
            wn = (g * w[j]) * (1.0 - st)
            if j < 4:
                wn_d = wn_d + wn
                h_d = h_d + f[j] * hd
                se_d = se_d + f[j] * se
                m_d = m_d + f[j] * m[j]
                s_d = s_d + f[j] * s[j]
            else:
                wn_l = wn_l + wn
                h_l = h_l + f[j] * hl
                se_l = se_l + f[j] * se
                m_l = m_l + f[j] * m[j]
                s_l = s_l + f[j] * s[j]
            load = load + w[j]
            bsum = bsum + w[j] / rhop
        sig_raw = f_dead * s_d + f_live * s_l
        bulk_density = load / depth
        beta = bsum / depth
        burnable = (sig_raw > 1.0e-7) & (load > 0.0)
        sig = np.where(burnable, sig_raw, 1.0)
        beta_op = 3.348 * _pow(sig, -0.8189)
        rpr = beta / beta_op

        fine_d = fine_dm = fine_l = zero
        for j in range(4):
            fw = w[j] * ex138[j]
            fine_d = fine_d + fw
            fine_dm = fine_dm + fw * m[j]
        for j in (4, 5):
            fine_l = fine_l + w[j] * _exp(-500.0 / s[j])
        fdm = _div0(fine_dm, fine_d, 1.0e-7)
        ratio_w = _div0(fine_d, fine_l, 1.0e-7)
        mxl = 2.9 * ratio_w * (1.0 - fdm / mxd) - 0.226
        mxl = np.where(mxl > mxd, mxl, mxd)

        def damping(r):
            cubic = 1.0 - 2.59 * r + 5.11 * (r * r) - 3.52 * ((r * r) * r)
            return np.where(r >= 1.0 - 1e-12, 0.0, cubic)

        etam_d = damping(m_d / mxd)
        etam_l = damping(m_l / mxl)

        def mineral(x):
            v = np.where(x > 0.0, 0.174 * _pow(np.where(x > 0.0, x, 1.0), -0.19),
                         0.0)
            return np.where(v < 1.0, v, 1.0)

        etas_d, etas_l = mineral(se_d), mineral(se_l)
        sig15 = _pow(sig, 1.5)
        gamma_max = sig15 / (495.0 + 0.0594 * sig15)
        a_coef = 133.0 * _pow(sig, -0.7913)
        gamma = (gamma_max * _pow(rpr, a_coef)) * _exp(a_coef * (1.0 - rpr))
        ir = gamma * (((wn_d * h_d) * etam_d) * etas_d
                      + ((wn_l * h_l) * etam_l) * etas_l)
        ir = np.where(burnable, ir, 0.0)
        xi = _exp((0.792 + 0.681 * np.sqrt(sig)) * (beta + 0.1)) / (
            192.0 + 0.2595 * sig)
        hs = zero
        for j in range(6):
            fc = f_dead if j < 4 else f_live
            hs = hs + ((fc * f[j]) * ex138[j]) * (250.0 + 1116.0 * m[j])
        hs = bulk_density * hs
        r0 = _div0(ir * xi, hs, 1.0e-7)
        r0 = np.where(burnable, r0, 0.0)
        return dict(
            r0=r0, ir=ir, sigma=np.where(burnable, sig_raw, 0.0),
            bulk_density=bulk_density, beta=beta,
            beta_safe=np.where(beta > 0.0, beta, 1.0),
            rpr=np.where(burnable, rpr, 0.0),
            rpr_safe=np.where(burnable, rpr, 1.0), mxl=mxl,
            wind_c=7.47 * _exp(-0.133 * _pow(sig, 0.55)),
            wind_b=0.02526 * _pow(sig, 0.54),
            wind_e=0.715 * _exp(-3.59e-4 * sig), burnable=burnable)


def _head_fire_host(bed, wind, slope, limit_code):
    c, b, e, rp = bed["wind_c"], bed["wind_b"], bed["wind_e"], bed["rpr_safe"]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u = np.where(wind > 0.0, wind, 0.0) * _MS_TO_FT_MIN
        phi_w = np.where(u >= 1.0e-7, (c * _pow(np.where(u >= 1.0e-7, u, 1.0), b))
                         * _pow(rp, -e), 0.0)
        t = np.abs(slope)
        phi_s = (5.275 * _pow(bed["beta_safe"], -0.3)) * (t * t)
        phi_w = np.where(bed["burnable"], phi_w, 0.0)
        phi_s = np.where(bed["burnable"], phi_s, 0.0)
        phi_e = phi_w + phi_s
        u_eff = _pow((phi_e * _pow(rp, e)) / c, 1.0 / b)
        limited = np.zeros(np.shape(phi_e), dtype=bool)
        if limit_code:
            ir = bed["ir"]
            u_lim = 0.9 * ir if limit_code == 1 else 96.8 * _pow(ir, 1.0 / 3.0)
            limited = (u_eff > u_lim) & bed["burnable"] & (ir > 0.0)
            phi_lim = (c * _pow(u_lim, b)) * _pow(rp, -e)
            phi_e = np.where(limited, phi_lim, phi_e)
            u_eff = np.where(limited, u_lim, u_eff)
        ros = bed["r0"] * (1.0 + phi_e)
        residence = _div0(384.0, bed["sigma"], 1.0e-7)
        hpa = bed["ir"] * residence
        fli = (hpa * ros) / 60.0
        flame = np.where(fli < 1.0e-7, 0.0,
                         0.45 * _pow(np.where(fli < 1.0e-7, 1.0, fli), 0.46))
    out = dict(ros=ros, r0=bed["r0"], phi_wind=phi_w, phi_slope=phi_s,
               effective_wind=u_eff, ir=bed["ir"], hpa=hpa, fli=fli,
               flame=flame, residence=residence, sigma=bed["sigma"],
               bulk_density=bed["bulk_density"], beta=bed["beta"],
               rpr=bed["rpr"], mxl=bed["mxl"], wind_c=c, wind_b=b, wind_e=e)
    return out, limited


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SurfaceFire:
    """Head-fire behaviour of the full model, SI units, one value per cell.

    ``ros`` is the head-fire rate of spread with wind blowing upslope (m
    s-1) and ``ros_no_wind_no_slope`` the base rate R0.  ``phi_wind`` and
    ``phi_slope`` are the dimensionless wind and slope factors, so
    ``ros = R0 (1 + phi_wind + phi_slope)`` unless the wind limit binds.
    ``effective_wind`` is the mid-flame wind that alone would give the
    combined factor (m s-1); ``wind_limited`` flags cells where the limit
    was imposed.  ``reaction_intensity`` (kW m-2), ``heat_per_unit_area``
    (kJ m-2), ``fireline_intensity`` (Byram, kW m-1), ``flame_length``
    (Byram, m) and ``residence_time`` (Anderson 1969, s) are the
    associated models of Andrews (2018) Table 7.  The fuel bed's
    characteristic SAV (m-1), bulk density (kg m-3), packing ratio,
    relative packing ratio and live moisture of extinction (fraction) are
    the intermediates the published tables tabulate.  ``wind_c``,
    ``wind_b``, ``wind_e`` are Rothermel's wind-factor constants for U in
    ft min-1, kept in those units because they are only meaningful there.
    """

    ros: Any
    ros_no_wind_no_slope: Any
    phi_wind: Any
    phi_slope: Any
    effective_wind: Any
    wind_limited: Any
    reaction_intensity: Any
    heat_per_unit_area: Any
    fireline_intensity: Any
    flame_length: Any
    residence_time: Any
    characteristic_sav: Any
    bulk_density: Any
    packing_ratio: Any
    relative_packing_ratio: Any
    moisture_extinction_live: Any
    wind_c: Any
    wind_b: Any
    wind_e: Any


def rothermel_surface_fire(fuel, m_1h, m_10h, m_100h, m_live_herb,
                           m_live_woody, midflame_wind, slope, *,
                           curing=None, wind_limit: str | None = None
                           ) -> SurfaceFire:
    """Head-fire spread of the full Rothermel-Albini model, float64.

    ``fuel`` is a :class:`~woof.fire.fuels.FuelModel`, a standard model
    number or code, or an integer array of model numbers (one per cell;
    91-93, 98, 99 are non-burnable).  Moisture contents are fractions of
    oven-dry mass for the 1-h, 10-h, 100-h, live herbaceous and live woody
    classes; ``midflame_wind`` is the mid-flame wind speed in m s-1 (a
    negative value is treated as calm), and ``slope`` is the slope
    steepness ``tan(phi)`` along the spread direction -- only its
    magnitude matters, because the head fire runs upslope.

    ``curing`` overrides the moisture-derived live-to-dead herbaceous
    transfer of dynamic models with a fixed cured fraction, as BehavePlus's
    direct-curing option does.  ``wind_limit`` is ``None`` (no limit),
    ``"rothermel1972"`` (``U_E <= 0.9 I_R``) or ``"andrews2013"``
    (``U_E <= 96.8 I_R^(1/3)``), both in ft min-1 and Btu ft-2 min-1 and
    applied to the effective wind, after which the rate is
    ``R0 (1 + C U_lim^B (beta/beta_op)^-E)``.  All array arguments
    broadcast together; CuPy inputs run one NVRTC kernel.
    """
    if wind_limit not in WIND_LIMITS:
        raise ValueError(f"wind_limit must be one of {WIND_LIMITS}")
    code = _WIND_LIMIT_CODE[wind_limit]
    xp = _xp(fuel, m_1h, m_10h, m_100h, m_live_herb, m_live_woody,
             midflame_wind, slope, curing)
    columns, index, cure, fields, shape = _inputs(
        xp, fuel, (m_1h, m_10h, m_100h, m_live_herb, m_live_woody), curing,
        (midflame_wind, slope))
    if xp is np:
        bed = _fuel_bed_host(columns, index, *fields[:5], cure)
        us, limited = _head_fire_host(bed, fields[5], fields[6], code)
    else:
        n = int(np.prod(shape))
        out = xp.empty((len(_HEAD_OUT), max(n, 1)), dtype=xp.float64)
        flag = xp.empty(max(n, 1), dtype=xp.uint8)
        _device.launch("rothermel_head_fire", n, (
            index, *fields[:5], cure, fields[5], fields[6], np.int64(n), columns,
            np.int32(columns.shape[1]), np.int32(code), out, np.int64(out.shape[1]),
            flag))
        us = {k: out[i, :n].reshape(shape) for i, k in enumerate(_HEAD_OUT)}
        limited = flag[:n].reshape(shape).astype(bool)
    return SurfaceFire(
        ros=us["ros"] * FT_PER_MIN,
        ros_no_wind_no_slope=us["r0"] * FT_PER_MIN,
        phi_wind=us["phi_wind"], phi_slope=us["phi_slope"],
        effective_wind=us["effective_wind"] * FT_PER_MIN,
        wind_limited=limited,
        reaction_intensity=us["ir"] * BTU_PER_FT2_MIN,
        heat_per_unit_area=us["hpa"] * BTU_PER_FT2,
        fireline_intensity=us["fli"] * BTU_PER_FT_S,
        flame_length=us["flame"] * FT,
        residence_time=us["residence"] * 60.0,
        characteristic_sav=us["sigma"] * PER_FT,
        bulk_density=us["bulk_density"] * LB_PER_FT3,
        packing_ratio=us["beta"], relative_packing_ratio=us["rpr"],
        moisture_extinction_live=us["mxl"],
        wind_c=us["wind_c"], wind_b=us["wind_b"], wind_e=us["wind_e"])


@dataclass(frozen=True)
class RothermelFireParams:
    """The full model as WRF ``set_fire_params`` coefficients, float64.

    WRF's names and units: ``r_0`` no-wind no-slope rate (ft min-1),
    ``phiwc = C (beta/beta_op)^-E``, ``bbb = B``, ``betafl`` the packing
    ratio (the slope factor is ``5.275 betafl^-0.3 tan^2``), ``ischap`` 0.
    """

    r_0: Any
    phiwc: Any
    bbb: Any
    betafl: Any
    ischap: Any


def rothermel_fire_params(fuel, m_1h, m_10h, m_100h, m_live_herb,
                          m_live_woody, *, curing=None) -> RothermelFireParams:
    """The coefficients :func:`rothermel_fire_ros` needs, per cell."""
    xp = _xp(fuel, m_1h, m_10h, m_100h, m_live_herb, m_live_woody, curing)
    columns, index, cure, fields, shape = _inputs(
        xp, fuel, (m_1h, m_10h, m_100h, m_live_herb, m_live_woody), curing)
    if xp is np:
        bed = _fuel_bed_host(columns, index, *fields, cure)
        r_0 = bed["r0"]
        phiwc = bed["wind_c"] * _pow(bed["rpr_safe"], -bed["wind_e"])
        bbb, betafl = bed["wind_b"], bed["beta_safe"]
    else:
        n = int(np.prod(shape))
        r_0, phiwc, bbb, betafl = (xp.empty(shape, dtype=xp.float64)
                                   for _ in range(4))
        _device.launch("rothermel_coefficients", n, (
            index, *fields, cure, np.int64(n), columns,
            np.int32(columns.shape[1]), r_0, phiwc, bbb, betafl))
    return RothermelFireParams(r_0=r_0, phiwc=phiwc, bbb=bbb, betafl=betafl,
                               ischap=xp.zeros(shape, dtype=xp.float64))


def rothermel_fire_ros(params: RothermelFireParams, vx, vy, dzdxf, dzdyf,
                       propx, propy, *, fire_advection: int = 0,
                       ros_max: float = 6.0):
    """WRF v4.7.1 ``fire_ros`` on the full model's coefficients, float64.

    The same transcription as :func:`woof.fire.wrf_sfire.fire_ros`, in
    double precision: ``(ros_base, ros_wind, ros_slope)`` in m s-1 for a
    fire-line normal ``(propx, propy)``, wind ``(vx, vy)`` (m s-1, mid-
    flame) and terrain gradient ``(dzdxf, dzdyf)``.  For wind and slope
    along the normal the sum is the head-fire rate of
    :func:`rothermel_surface_fire` up to WRF's literal unit factors
    (``196.850``, ``.00508``); against the normal it is ``r_0``.
    """
    from woof.fire.wrf_sfire import fire_ros_dispatch
    return fire_ros_dispatch(params, vx, vy, dzdxf, dzdyf, propx, propy,
                             fire_advection=fire_advection,
                             ros_max=float(ros_max), dtype=np.float64,
                             kernel="rothermel_fire_ros", pow_=_pow,
                             literal=float)


# --------------------------------------------------------------------------
# wind at mid-flame height
# --------------------------------------------------------------------------

def unsheltered_wind_adjustment_factor(depth):
    """Albini & Baughman (1979) unsheltered 20-ft to mid-flame factor.

    ``WAF = 1.83 / ln((20 + 0.36 H) / (0.13 H))`` with fuel bed depth H in
    ft (Andrews 2012, RMRS-GTR-266, eq. 6; BehavePlus
    ``windAdjustmentFactor.cpp``), for fuels with no overstory.  ``depth``
    is in m, NumPy (a per-model quantity, evaluated on the host with the C
    library's ``log``).  WRF-SFIRE's ``windrf`` for the Anderson 13 is this
    factor to two decimals (except the litter models 8 and 9); it has none
    for the Scott & Burgan 40, and this is the published way to get one.
    """
    from woof.core import host_libm
    h = np.asarray(depth, dtype=np.float64) / FT
    return 1.83 / host_libm.log((20.0 + 0.36 * h) / (0.13 * h))


def log_profile_wind(wind, z_from, z_to, z0):
    """Move a neutral log-profile wind from height ``z_from`` to ``z_to``.

    ``U(z_to) = U(z_from) ln(z_to/z0) / ln(z_from/z0)``, the vertical
    interpolation WRF uses to bring the lowest-level wind to
    ``fire_wind_height``; heights and roughness length in m.  NumPy, with
    the C library's ``log`` (the factor is per cell but cheap; move it to
    the device with ``cupy.asarray``).
    """
    from woof.core import host_libm
    z0 = np.asarray(z0, dtype=np.float64)
    return np.asarray(wind, dtype=np.float64) * (
        host_libm.log(np.asarray(z_to, dtype=np.float64) / z0)
        / host_libm.log(np.asarray(z_from, dtype=np.float64) / z0))
