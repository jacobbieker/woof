"""WRF's p_lev / z_lev diagnostics at user-given levels on the MPAS-A v8.4.2 suite route.

A WOOF-only output.  MPAS-A v8.4.2's own isobaric diagnostics (isobaric.py) write a FIXED set of levels under
fixed Registry names (``temperature_850hPa`` ...), word for word against native MPAS; WRF's ``p_lev_diags`` and
``z_lev_diags`` take the levels from the namelist.  This module is that: the WRF 4.7.1 phys/module_diag_pld.F
(``pld``) and phys/module_diag_zld.F (``zld``) column interpolation, on mesh columns (one cell is one WRF column;
both are column local), written as ``diag`` fields [nPLevels|nZLevels, nCells].  With neither option on nothing here
is allocated, constructed or added to the Registry, so a run without it keeps its registry and stream digests.

The WOOF namelist group ``&diagnostics`` (the group config_solar_diagnostics and config_nwp_diagnostics share)
takes, because the namelist reader is scalar only, the level lists as quoted comma separated character scalars::

    config_p_lev_diags      logical   WRF p_lev_diags (default false)
    config_press_levels     character WRF press_levels, Pa, e.g. '100000,85000,50000' (WRF order is kept: the
                                      search state carried from one level to the next makes the order matter)
    config_extrap_below_grnd integer  1 = missing below ground (default), 2 = WRF's Shuell extrapolation
    config_p_lev_missing    real      WRF p_lev_missing (default -999.)
    config_z_lev_diags      logical   WRF z_lev_diags (default false)
    config_z_levels         character WRF z_levels, m; a level < 1 is metres ABOVE GROUND, |z| + terrain height
    config_z_lev_missing    real      WRF z_lev_missing (default -999.)

    config_use_tot_or_hyd_p = 2 (WRF's hydrostatic pressure) is refused by name: MPAS carries no hydrostatic
    pressure column, only total pressure (pressure_base + pressure_p), WRF's use_tot_or_hyd_p = 1.

Fields (WRF's names in lower case)::

    p_pl  [nPLevels]        the levels                        u_pl v_pl t_pl rh_pl ght_pl s_pl td_pl q_pl
    z_zl  [nZLevels]        the levels                        u_zl v_zl t_zl rh_zl ght_zl s_zl td_zl q_zl p_zl
                                                              [nPLevels|nZLevels, nCells]

Where the WRF route and this one differ, and why:

* WRF's staggered winds ``0.5 * (u(i) + u(i+1))`` are the cell-centre reconstructed ``uReconstructZonal/Meridional``
  (MPAS has no C-grid x/y faces on a mesh column); ``s`` is the speed of those.
* ``t`` is the dry potential temperature ``diag.theta`` less WRF's base ``t0 = 300``, so ``(t + t0)`` returns it.
* the geopotential is ``PH = 0``, ``PHB = 9.81 * zgrid`` (as nwp_diag_v842 and solar_diagnostics_v842), so WRF's
  ``(zp + zb) / g`` heights are ``zgrid``.
* ``pw``, the pressure at the interfaces WRF's ``pld`` searches for the geopotential height, is the interface
  pressure the MPAS isobaric module builds for its own ``height_*hPa`` (mpas_isobaric_diagnostics.F:596-628,
  the log-pressure blend of the neighbouring mass levels), not WRF's hydrostatic ``p_hyd_w``.

The interpolation is WRF's word for word, search state included: ``ke_h``/``ke_f`` are carried from one level to the
next, a level found at layer ``ke`` makes the next search start at ``ke``, and the below-ground branch (which
does not move ``ke_h``) is tried first at the layer the search starts on.  It is evaluated for every cell at once
(numpy or cupy arrays), in float32 with WRF's operation order.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .utils import Direction

GROUP = "diagnostics"
P_OPTION = "config_p_lev_diags"
Z_OPTION = "config_z_lev_diags"
OPTIONS = ("config_p_lev_diags", "config_press_levels", "config_extrap_below_grnd", "config_p_lev_missing",
           "config_z_lev_diags", "config_z_levels", "config_z_lev_missing", "config_use_tot_or_hyd_p")
P_PACKAGE = "p_lev_diags"
Z_PACKAGE = "z_lev_diags"
#: Prefix of nothing: these diagnostics hold no running state (each frame is a pure function of the state).
P_FIELDS_2D = ("u_pl", "v_pl", "t_pl", "rh_pl", "ght_pl", "s_pl", "td_pl", "q_pl")
Z_FIELDS_2D = ("u_zl", "v_zl", "t_zl", "rh_zl", "ght_zl", "s_zl", "td_zl", "q_zl", "p_zl")
_DESCRIPTION = {
    "u": ("m s-1", "u wind at the {kind} level"), "v": ("m s-1", "v wind at the {kind} level"),
    "t": ("K", "Temperature at the {kind} level"), "rh": ("%", "Relative humidity at the {kind} level"),
    "ght": ("m", "Geopotential height at the {kind} level"), "s": ("m s-1", "Wind speed at the {kind} level"),
    "td": ("K", "Dewpoint temperature at the {kind} level"), "q": ("kg kg-1", "Water vapor mixing ratio at the {kind} level"),
    "p": ("Pa", "Pressure at the {kind} level"),
}


def _row(name: str, dimension: str, kind: str) -> dict[str, Any]:
    units, text = _DESCRIPTION[name.split("_")[0]]
    return {"type": "real", "dimensions": [dimension, "nCells", "Time"],
            "attrs": {"units": units, "long_name": text.format(kind=kind), "wrf_name": name.upper()},
            "gates": [P_PACKAGE if name.endswith("_pl") else Z_PACKAGE]}


FIELDS: dict[str, dict[str, Any]] = {
    **{name: _row(name, "nPLevels", "pressure") for name in P_FIELDS_2D},
    **{name: _row(name, "nZLevels", "height") for name in Z_FIELDS_2D},
    "p_pl": {"type": "real", "dimensions": ["nPLevels"], "attrs": {"units": "Pa", "long_name": "Pressure levels",
             "wrf_name": "P_PL"}, "gates": [P_PACKAGE]},
    "z_zl": {"type": "real", "dimensions": ["nZLevels"], "attrs": {"units": "m", "long_name": "Height levels",
             "wrf_name": "Z_ZL"}, "gates": [Z_PACKAGE]},
}
NAMES = tuple(FIELDS)
DIMENSIONS = {"nPLevels": P_PACKAGE, "nZLevels": Z_PACKAGE}


@dataclass(frozen=True)
class LevelConfig:
    """The resolved ``&diagnostics`` level options (either list may be empty = that diagnostic off)."""

    press_levels: tuple[float, ...] = ()
    z_levels: tuple[float, ...] = ()
    extrap_below_grnd: int = 1
    p_missing: float = -999.0
    z_missing: float = -999.0


def field_definition(name: str) -> dict[str, Any]:
    row = FIELDS[name]
    return {"type": row["type"], "dimensions": list(row["dimensions"]),
            "attrs": dict(row["attrs"]), "gates": list(row["gates"])}


def _logical(key: str, value: Any) -> bool:
    if isinstance(value, str) and value.strip().strip(".").casefold() in ("true", "t", "false", "f"):
        return value.strip().strip(".").casefold() in ("true", "t")
    if not isinstance(value, bool):
        raise ValueError(f"{key} is logical (.true./.false.), not {value!r}")
    return value


def _levels(key: str, value: Any, *, positive: bool) -> tuple[float, ...]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        raise ValueError(f"{key} is a quoted comma separated list of levels, e.g. '100000,85000', not {value!r}")
    try:
        levels = tuple(float(np.float32(item)) for item in text.replace(";", ",").split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{key} {value!r} is not a list of numbers") from error
    if not levels:
        raise ValueError(f"{key} names no levels")
    if not all(np.isfinite(levels)):
        raise ValueError(f"{key} holds a level that is not finite")
    if positive and min(levels) <= 0:
        raise ValueError(f"{key} levels are pressures in Pa and must be positive")
    return levels


def resolve(groups: Mapping[str, Mapping[str, Any]]) -> LevelConfig | None:
    """The level options from the namelist's &diagnostics group (None when neither diagnostic is on)."""
    supplied = {str(k).casefold(): v for k, v in groups.get(GROUP, {}).items()}
    p_on = _logical(P_OPTION, supplied.get(P_OPTION, False))
    z_on = _logical(Z_OPTION, supplied.get(Z_OPTION, False))
    hyd = supplied.get("config_use_tot_or_hyd_p", 1)
    if hyd != 1:
        raise ValueError(
            "config_use_tot_or_hyd_p = 2 (WRF's hydrostatic pressure) is not available: MPAS carries total pressure "
            "(pressure_base + pressure_p) only, which is WRF's use_tot_or_hyd_p = 1")
    if not (p_on or z_on):
        stray = sorted(key for key in supplied if key in OPTIONS and key not in (P_OPTION, Z_OPTION, "config_use_tot_or_hyd_p"))
        if stray:
            raise ValueError(f"&{GROUP} {stray[0]} is set while neither {P_OPTION} nor {Z_OPTION} is on; it would be read by nothing")
        return None
    extrap = supplied.get("config_extrap_below_grnd", 1)
    if type(extrap) is not int or extrap not in (1, 2):
        raise ValueError(f"config_extrap_below_grnd is 1 (missing below ground) or 2 (WRF's extrapolation), not {extrap!r}")
    p_levels = z_levels = ()
    if p_on:
        if "config_press_levels" not in supplied:
            raise ValueError(f"{P_OPTION} needs config_press_levels (Pa, a quoted comma separated list)")
        p_levels = _levels("config_press_levels", supplied["config_press_levels"], positive=True)
    elif "config_press_levels" in supplied:
        raise ValueError(f"config_press_levels is set while {P_OPTION} is off; it would be read by nothing")
    if z_on:
        if "config_z_levels" not in supplied:
            raise ValueError(f"{Z_OPTION} needs config_z_levels (m, a quoted comma separated list)")
        z_levels = _levels("config_z_levels", supplied["config_z_levels"], positive=False)
    elif "config_z_levels" in supplied:
        raise ValueError(f"config_z_levels is set while {Z_OPTION} is off; it would be read by nothing")
    missing = {}
    for key in ("config_p_lev_missing", "config_z_lev_missing"):
        value = supplied.get(key, -999.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} is real, not {value!r}")
        missing[key] = float(np.float32(value))
    return LevelConfig(p_levels, z_levels, extrap, missing["config_p_lev_missing"], missing["config_z_lev_missing"])


def packages(config: Any) -> set[str]:
    levels = getattr(config, "plevel_diagnostics", None)
    if levels is None:
        return set()
    return ({P_PACKAGE} if levels.press_levels else set()) | ({Z_PACKAGE} if levels.z_levels else set())


def extend_registry(root, config: Any) -> None:
    """Add the dimensions, the ``diag`` fields and their ``diagnostics`` stream members of the active diagnostics
    to a parsed Registry tree; leave the tree as parsed when neither is on."""
    import xml.etree.ElementTree as ET
    active = packages(config)
    if not active:
        return
    levels = config.plevel_diagnostics
    sizes = {"nPLevels": len(levels.press_levels), "nZLevels": len(levels.z_levels)}
    dims = root.find("./dims")
    struct = root.find("./var_struct[@name='diag']")
    stream = root.find("./streams/stream[@name='diagnostics']")
    for dim, package in DIMENSIONS.items():
        if package in active and dims.find(f"./dim[@name='{dim}']") is None:
            ET.SubElement(dims, "dim", {"name": dim, "definition": str(sizes[dim])})
    for name, row in FIELDS.items():
        package = row["gates"][0]
        if package not in active:
            continue
        if struct.find(f"./var[@name='{name}']") is None:
            ET.SubElement(struct, "var", {
                "name": name, "type": "real", "dimensions": " ".join(row["dimensions"]),
                "units": row["attrs"]["units"], "description": row["attrs"]["long_name"],
                "default_value": "0.0", "packages": package})
        if stream is not None and stream.find(f"./var[@name='{name}']") is None:
            ET.SubElement(stream, "var", {"name": name, "packages": package})


# ---------------------------------------------------------------------------
# the WRF interpolation
# ---------------------------------------------------------------------------
F32 = np.float32
_G, _R_D, _T0, _P1000 = F32(9.81), F32(287.0), F32(300.0), F32(100000.0)
_RCP = F32(_R_D / F32(F32(7.0) * _R_D / F32(2.0)))
_EPS, _T_KELVIN = F32(0.622), F32(273.15)
_S1, _S2, _S3, _S4, _S5 = F32(243.5), F32(17.67), F32(F32(0.6112) * F32(10.0)), F32(611.0), F32(5418.12)
_ZSHUL, _TVSHUL = F32(75.0), F32(290.66)


def interface_pressure(xp, p, zgrid):
    """mpas_isobaric_diagnostics.F:596-628 (the kernel prepare_interfaces): pressure at the nl+1 interfaces from the nl
    mass-level pressures ``p`` (nl, nc) and interface heights ``zgrid`` (nl+1, nc); the log-pressure blend."""
    nl = p.shape[0]
    h = zgrid.astype(F32)
    half = F32(0.5)

    def blend(w1, pa, pb):
        return xp.exp(w1 * xp.log(pa) + (F32(1) - w1) * xp.log(pb)).astype(F32)

    out = [None] * (nl + 1)
    for k in (0, nl):
        near, far = (k + 1, k + 2) if k == 0 else (k - 1, k - 2)
        a, b = (0, 1) if k == 0 else (nl - 1, nl - 2)
        z0 = h[k]
        z1 = half * (h[k] + h[near])
        z2 = half * (h[near] + h[far])
        w1 = (z0 - z2) / (z1 - z2)
        out[k] = blend(w1, p[a], p[b])
    for k in range(1, nl):
        den = h[k + 1] - h[k - 1]
        w1 = (h[k] - h[k - 1]) / den
        w2 = (h[k + 1] - h[k]) / den
        out[k] = xp.exp(w1 * xp.log(p[k]) + w2 * xp.log(p[k - 1])).astype(F32)
    return xp.stack(out)


def _first_at_or_after(xp, condition, start):
    """Per column the first row ``>= start`` of ``condition`` (rows, cols): (found, row)."""
    rows = xp.arange(condition.shape[0], dtype=xp.int32)[:, None]
    ok = condition & (rows >= start[None, :])
    return ok.any(axis=0), ok.argmax(axis=0).astype(xp.int32)


def _gather(xp, array, row):
    return array[row, xp.arange(array.shape[1])]


def _column_state(xp, u, v, t, qv, p):
    """Per mass level: temperature, wind speed, mixing ratio, dewpoint, from WRF's own expressions."""
    temp = (t + _T0) * (p / _P1000) ** _RCP
    q = xp.maximum(qv, F32(0))
    ev = xp.maximum(q * p * F32(0.01) / (_EPS + q), F32(0.001))
    dew = _T_KELVIN + (_S1 / ((_S2 / xp.log(ev / _S3)) - F32(1.0)))
    speed = F32(0.5) * xp.sqrt((u + u) ** 2 + (v + v) ** 2)
    return temp, speed, q, dew


def _qs(xp, t, p):
    """The saturation mixing ratio WRF's ``qs = eps * es / (p - es)`` with ``es = s4 * exp(s5 * (1/273 - 1/t))``."""
    es = _S4 * xp.exp(_S5 * (F32(1.0) / F32(273.0) - F32(1.0) / t))
    return _EPS * es / (p - es)


def _rh(xp, q, t, pm):
    return q / _qs(xp, t, pm) * F32(100.0)


def _quiet(xp):
    """Columns that select no layer evaluate their unused words on a dummy layer: no numpy warnings for those."""
    return np.errstate(all="ignore") if xp is np else contextlib.nullcontext()


def pressure_levels(xp, **arguments):
    """WRF 4.7.1 module_diag_pld.F ``pld`` for every column; see :func:`_pressure_levels`."""
    with _quiet(xp):
        return _pressure_levels(xp, **arguments)


def height_levels(xp, **arguments):
    """WRF 4.7.1 module_diag_zld.F ``zld`` for every column; see :func:`_height_levels`."""
    with _quiet(xp):
        return _height_levels(xp, **arguments)


def _pressure_levels(xp, *, u, v, t, qv, zp, zb, pp, pb, pw, levels, extrap_below_grnd=1, missing=-999.0):
    """WRF 4.7.1 module_diag_pld.F ``pld`` for every column.

    ``u v t qv pp pb`` are (nl, nc) mass-level arrays (``t`` is WRF's theta - t0), ``zp zb pw`` (nl+1, nc)
    interface arrays; ``levels`` are the Pa levels in WRF's order.  Returns ``{name: (len(levels), nc)}`` for
    u v t rh ght s td q (the WRF ``*_pl`` fields).
    """
    nl, nc = pp.shape
    missing = F32(missing)
    p = pp + pb
    temp, speed, q, dew = _column_state(xp, u, v, t, qv, p)
    zmass = (zp + zb)
    out = {name: xp.full((len(levels), nc), missing, dtype=F32)
           for name in ("u", "v", "t", "rh", "ght", "s", "td", "q")}
    ke_h = xp.zeros(nc, dtype=xp.int32)
    ke_f = xp.zeros(nc, dtype=xp.int32)
    pu, pd = p[1:], p[:-1]
    cols = xp.arange(nc)
    for kp, level in enumerate(levels):
        pm = F32(level)
        # --- the layers of the mass levels (ke_loop_half)
        inside = (pd >= pm) & (pu < pm)
        found, ke = _first_at_or_after(xp, inside, ke_h)
        below = xp.zeros(nc, dtype=bool)
        if extrap_below_grnd == 2:
            start = xp.minimum(ke_h, nl - 2)
            pd0 = p[start, cols]
            below = (ke_h <= nl - 2) & (pm > pd0)
        if bool(below.any()):
            k = xp.minimum(ke_h, nl - 2)
            pdk = p[k, cols]
            zu = F32(0.5) * (((zp[k, cols] + zb[k, cols]) + zp[k + 1, cols]) + zb[k + 1, cols]) / _G
            tu = temp[k, cols]
            tvu = tu * (F32(1.0) + F32(0.608) * q[k, cols])
            tvd = tvu + zu * F32(6.5e-3)
            hot = tvd > _TVSHUL
            tvd = xp.where(hot, xp.where(tvu > _TVSHUL, _TVSHUL - F32(5.0e-3) * (tvu - _TVSHUL) ** 2, _TVSHUL), tvd)
            gammas = xp.where(zu > _ZSHUL, (tvu - tvd) / zu, F32(0.0))
            part = (_R_D / _G) * (np.log(pm) - xp.log(pdk))
            ght = zu - tvu * part / (F32(1.0) + F32(0.5) * gammas * part)
            tpl = tu + (zu - ght) * F32(6.5e-3)
            rh = q[k, cols] / _qs(xp, tu, pdk) * F32(100.0)
            qpl = rh * _qs(xp, tpl, pm) / F32(100.0)
            ed = xp.maximum(qpl * pm * F32(0.01) / (_EPS + qpl), F32(0.001))
            tdpl = _T_KELVIN + (_S1 / ((_S2 / xp.log(ed / _S3)) - F32(1.0)))
            for name, value in (("ght", ght), ("t", tpl), ("s", speed[k, cols]), ("u", u[k, cols]), ("v", v[k, cols]),
                                ("rh", rh), ("q", qpl), ("td", tdpl)):
                out[name][kp] = xp.where(below, value.astype(F32), out[name][kp])
        use = found & ~below
        k = xp.minimum(ke, nl - 2)
        den = xp.where(use, pu[k, cols] - pd[k, cols], F32(1.0))
        pdk, puk = pd[k, cols], pu[k, cols]
        w_u, w_d = pm - pdk, puk - pm

        def blend(up, down):
            return (up * w_u + down * w_d) / den

        tpl = blend(temp[k + 1, cols], temp[k, cols])
        qpl = blend(q[k + 1, cols], q[k, cols])
        tdpl = blend(dew[k + 1, cols], dew[k, cols])
        values = {"t": tpl, "s": blend(speed[k + 1, cols], speed[k, cols]), "u": blend(u[k + 1, cols], u[k, cols]),
                  "v": blend(v[k + 1, cols], v[k, cols]), "q": qpl, "td": tdpl,
                  "rh": _rh(xp, qpl, tpl, pm)}
        for name, value in values.items():
            out[name][kp] = xp.where(use, value.astype(F32), out[name][kp])
        ke_h = xp.where(use, ke, ke_h)
        # --- the full levels (ke_loop_full): the geopotential height between the interface pressures
        crossing = (pw[:-1] >= pm) & (pw[1:] < pm)
        found_f, kf = _first_at_or_after(xp, crossing, ke_f)
        pu_l, pd_l, pm_l = xp.log(pw[1:][kf, cols]), np.log(pm), xp.log(pw[:-1][kf, cols])
        zu_f = zmass[1:][kf, cols] / _G
        zd_f = zmass[:-1][kf, cols] / _G
        den_f = xp.where(found_f, pu_l - pm_l, F32(1.0))
        ght = (zu_f * (pd_l - pm_l) + zd_f * (pu_l - pd_l)) / den_f
        out["ght"][kp] = xp.where(found_f, ght.astype(F32), out["ght"][kp])
        ke_f = xp.where(found_f, kf, ke_f)
    return out


def _height_levels(xp, *, u, v, t, qv, zp, zb, pp, pb, ht, levels, missing=-999.0):
    """WRF 4.7.1 module_diag_zld.F ``zld`` for every column.

    Same inputs as :func:`pressure_levels` plus the terrain height ``ht`` (nc,); ``levels`` are heights in m in
    WRF's order, a level below 1 being metres above ground.  Returns ``{name: (len(levels), nc)}`` for
    u v t rh ght s td q p (the WRF ``*_zl`` fields).
    """
    nl, nc = pp.shape
    missing = F32(missing)
    p = pp + pb
    temp, speed, q, dew = _column_state(xp, u, v, t, qv, p)
    zi = zp + zb
    zmid = ((zp[:-1] + zb[:-1]) + zp[1:]) + zb[1:]
    zmid = zmid / F32(2.0) / _G
    out = {name: xp.full((len(levels), nc), missing, dtype=F32)
           for name in ("u", "v", "t", "rh", "ght", "s", "td", "q", "p")}
    ke_h = xp.zeros(nc, dtype=xp.int32)
    ke_f = xp.zeros(nc, dtype=xp.int32)
    cols = xp.arange(nc)
    zd_all, zu_all = zmid[:-1], zmid[1:]
    zface_d, zface_u = zi[:-1] / _G, zi[1:] / _G
    for kz, level in enumerate(levels):
        level = F32(level)
        zm = xp.where(level < F32(1.0), xp.abs(level) + ht, level + xp.zeros(nc, dtype=F32)).astype(F32)
        inside = (zd_all <= zm[None, :]) & (zu_all > zm[None, :])
        found, ke = _first_at_or_after(xp, inside, ke_h)
        k = xp.minimum(ke, nl - 2)
        zd, zu = zd_all[k, cols], zu_all[k, cols]
        den = xp.where(found, zu - zd, F32(1.0))
        pu, pd = p[k + 1, cols], p[k, cols]
        w_u, w_d = zm - zd, zu - zm

        def blend(up, down):
            return (up * w_u + down * w_d) / den

        pm = xp.exp((xp.log(pu) * w_u + xp.log(pd) * w_d) / den)
        tzl = blend(temp[k + 1, cols], temp[k, cols])
        qzl = blend(q[k + 1, cols], q[k, cols])
        values = {"p": pm, "t": tzl, "s": blend(speed[k + 1, cols], speed[k, cols]),
                  "u": blend(u[k + 1, cols], u[k, cols]), "v": blend(v[k + 1, cols], v[k, cols]),
                  "q": qzl, "td": blend(dew[k + 1, cols], dew[k, cols])}
        values["rh"] = qzl / _qs(xp, tzl, pm) * F32(100.0)
        for name, value in values.items():
            out[name][kz] = xp.where(found, value.astype(F32), out[name][kz])
        ke_h = xp.where(found, ke, ke_h)
        crossing = (zface_d <= zm[None, :]) & (zface_u > zm[None, :])
        found_f, kf = _first_at_or_after(xp, crossing, ke_f)
        zu_f, zd_f = zface_u[kf, cols], zface_d[kf, cols]
        den_f = xp.where(found_f, zu_f - zd_f, F32(1.0))
        ght = (zu_f * (zm - zd_f) + zd_f * (zu_f - zm)) / den_f
        out["ght"][kz] = xp.where(found_f, ght.astype(F32), out["ght"][kz])
        ke_f = xp.where(found_f, kf, ke_f)
    return out


# ---------------------------------------------------------------------------
# the route's module
# ---------------------------------------------------------------------------
class LevelDiagnosticsModule:
    """Publish the p_lev / z_lev fields into the diag buffers a due stream will write.

    ``owner`` is the route's ResidentOutputDiagnostics (its device pools, buffers and stream views).  Nothing here
    accumulates: each frame is a function of the end-of-step state, so there is nothing to reset or to checkpoint.
    """

    def __init__(self, owner, config: LevelConfig):
        self.owner = owner
        self.config = config

    def setup(self, utils):
        session = self.owner.session
        dims = session.layout.dimensions
        self.cp = self.owner.cp
        self.nc, self.nl = int(dims["nCellsSolve"]), int(dims["nVertLevels"])
        self.utils = utils
        self.names = tuple(name for name in NAMES if name in {row["name"] for row in session.layout.fields
                                                              if row["pool"] == "diag"})

    def _scalar(self, name):
        indices = self.owner.session.layout.indices
        key = ("state", "index_" + name)
        if key not in indices or int(indices[key]) < 1:
            return None
        return self.owner.authority("state", "scalars")[int(indices[key]) - 1, :self.nl, :self.nc]

    def _need(self):
        return [name for name in self.names if self.utils.field_will_be_written(name)]

    def compute(self):
        wanted = self._need()
        if not wanted or not self.nc:
            return
        cp, nc, nl = self.cp, self.nc, self.nl
        f32 = cp.float32
        get = self.owner.authority

        def mass(pool, name):
            return cp.ascontiguousarray(get(pool, name)[:nl, :nc], dtype=f32)

        qv = self._scalar("qv")
        if qv is None:
            raise ValueError("p_lev/z_lev diagnostics need the water vapour scalar (qv)")
        zgrid = cp.ascontiguousarray(get("mesh", "zgrid")[:nl + 1, :nc], dtype=f32)
        common = dict(u=mass("diag", "uReconstructZonal"), v=mass("diag", "uReconstructMeridional"),
                      t=mass("diag", "theta") - f32(300.0), qv=cp.ascontiguousarray(qv, dtype=f32),
                      zp=cp.zeros_like(zgrid), zb=zgrid * f32(9.81),
                      pp=mass("diag", "pressure_p"), pb=mass("diag", "pressure_base"))
        buffers = self.owner.buffer
        if any(name in P_FIELDS_2D or name == "p_pl" for name in wanted) and self.config.press_levels:
            pw = interface_pressure(cp, common["pp"] + common["pb"], zgrid)
            result = pressure_levels(cp, pw=pw, levels=self.config.press_levels,
                                     extrap_below_grnd=self.config.extrap_below_grnd,
                                     missing=self.config.p_missing, **common)
            for name in P_FIELDS_2D:
                if name in wanted:
                    buffers("diag", name)[:, :nc] = result[name.split("_")[0]]
            if "p_pl" in wanted:
                buffers("diag", "p_pl")[...] = cp.asarray(self.config.press_levels, dtype=f32)
        if any(name in Z_FIELDS_2D or name == "z_zl" for name in wanted) and self.config.z_levels:
            ht = cp.ascontiguousarray(get("mesh", "zgrid")[0, :nc], dtype=f32)
            result = height_levels(cp, ht=ht, levels=self.config.z_levels, missing=self.config.z_missing, **common)
            for name in Z_FIELDS_2D:
                if name in wanted:
                    buffers("diag", name)[:, :nc] = result[name.split("_")[0]]
            if "z_zl" in wanted:
                buffers("diag", "z_zl")[...] = cp.asarray(self.config.z_levels, dtype=f32)
