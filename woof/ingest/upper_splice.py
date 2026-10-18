"""Stratospheric top-up: a second, isobaric source above a regional column.

HRRR's pressure product stops at 50 hPa and its native hybrid column at a
15 hPa lid, while a stratospheric run wants a model top near 10 hPa or
above.  GFS publishes T, RH, U, V and geopotential up to 1 hPa on the
0.25 degree grid, so this module carries the levels the LOWER source does
not have from an UPPER source mapped onto the same target grid, and hands
:func:`woof.ingest.real.initialize_real` one composite column whose top
reaches the model top.

The splice is done on the horizontally interpolated snapshot ("option A"),
never inside the mapped composition: every forcing time -- the initial
state and each boundary time -- is mapped once from each source onto the
same target points, and the two columns are joined here.  Nothing below
names HRRR or GFS; the lower snapshot is anything
:func:`woof.ingest.real.initialize_real` takes (an isobaric ladder, or
hybrid levels carrying a 3-D ``PRES``), and the upper snapshot is any
isobaric snapshot carrying ``TT``, ``UU``, ``VV``, ``GHT`` and ``RH`` or
``SPFH``.

The join, per column:

* below the SEAM (default 100 hPa .. 50 hPa) the lower source is kept
  unchanged;
* on every lower level inside the seam the value becomes
  ``(1 - w) * lower + w * upper(p)`` with ``w`` linear in ln p, 0 at the
  seam bottom and 1 at the seam top, and ``upper(p)`` the upper column
  interpolated linearly in ln p to that level's own pressure;
* lower levels above the seam top take the upper column outright (``w``
  is 1 there), so a lower source whose top lies above the seam -- native
  HRRR's -- is overwritten by the upper source in its damping layer
  rather than spliced at its lid;
* the upper source's levels above the whole lower column are appended.

The seam is checked before anything is returned: where both sources hold
the seam, temperature, geopotential height and wind must agree within
bounds a real analysis pair meets, the composite height must increase
strictly upward in every column and its pressure decrease strictly.  A
pair that does not is refused, loudly, rather than handed to the
initializer as a column with a step in it.

Moisture is carried in the lower source's own representation: an upper
RH column becomes specific humidity through the Bolton liquid-water
relation WRF's ``rh_to_mxrat1`` uses, with that routine's stratospheric
qv cap (``qv_max_p_safe``), so the upper levels hold exactly the vapour
the RH lane of the initializer would have made from them.
"""
from __future__ import annotations

import math
from dataclasses import fields as dataclass_fields, is_dataclass, replace
from types import SimpleNamespace
from typing import Mapping

import numpy as np

from woof.explain import layered
from woof.ingest.source_coverage import PreparationRefusal


SCHEMA = "woof-upper-source-splice-v1"

#: The default seam, Pa.  The coordinator's brief asked for "~70-50 hPa";
#: it is widened to 100 hPa so a lower ladder on HRRR's 25 hPa pressure
#: spacing still has an interior blended level (75 hPa, w = 0.415) rather
#: than a bare switch between two adjacent levels.  Both bounds are GFS
#: pgrb2 levels.
DEFAULT_SEAM_BOTTOM_PA = 10000.0
DEFAULT_SEAM_TOP_PA = 5000.0

#: Seam agreement bounds, measured as the largest |lower - upper| over
#: every column and every lower level inside the seam.  Two analyses of
#: the same lower stratosphere differ by a few K and some tens of metres;
#: these bounds are several times that and still far below what a wrong
#: field, a wrong unit or a wrong valid time produces (a 6 h timing error
#: in a jet moves winds by tens of m/s; a geopotential read as m2 s-2
#: differs by kilometres).
MAX_SEAM_TEMPERATURE_DIFFERENCE_K = 15.0
MAX_SEAM_HEIGHT_DIFFERENCE_M = 400.0
MAX_SEAM_WIND_DIFFERENCE_MS = 40.0

#: An upper level is appended only when it sits at least this far (in
#: ln p, about 2 %) above the highest lower level in EVERY column.  A
#: closer one would make a near-degenerate layer; dropping it loses
#: nothing, because the lower level just beneath it already holds the
#: upper value (``w`` is 1 above the seam top).
MIN_APPENDED_LOG_SEPARATION = 0.02

#: The physical range an appended temperature must lie in.
#: Wide enough for a summer mesopause near GFS's 1 Pa reach.
APPENDED_TEMPERATURE_RANGE_K = (110.0, 360.0)

#: Hydrometeor masses and their number concentrations: tapered to zero
#: across the seam and zero above it.  The upper source carries none, and
#: the stratosphere holds none an initial state needs.
_ZERO_ABOVE = frozenset({"QC", "QR", "QI", "QS", "QG", "QH",
                         "QNC", "QNR", "QNI", "QNS", "QNG", "QNH"})
#: Aerosol number concentrations: held at the lower column's top value
#: above it.  Zero would be a cloud-condensation-free stratosphere the
#: aerosol-aware scheme would then advect down through the boundaries.
_HOLD_ABOVE = frozenset({"QNWFA", "QNIFA", "QNBCA"})
#: Columns blended across the seam.  ``HGT`` is the native HRRR spelling
#: of the same geopotential height ``GHT`` carries.
_BLENDED = ("TT", "UU", "VV", "GHT", "HGT")

#: WRF ``rh_to_mxrat1`` (module_initialize_real.F:7490-7498, Registry
#: defaults): where p < qv_max_p_safe and qv > qv_max_flag, qv becomes
#: qv_max_value.
_QV_MAX_P_SAFE_PA = 10000.0
_QV_MAX_FLAG = 1.0e-5
_QV_MAX_VALUE = 3.0e-6


class UpperSpliceRefusal(PreparationRefusal):
    """The two sources do not join into one column."""

    remedy = (
        "remedy: check that the upper source was fetched for the same valid "
        "times as the lower one (`woof fetch --source gfs` with the run's "
        "cycle and --p-top-pa), or opt out with [fetch] upper_source = "
        "\"none\" and raise [shared].p_top to the lower source's own top.")


def _array_module(value):
    if hasattr(value, "__cuda_array_interface__"):
        import cupy as cp
        return cp
    return np


def _host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def _refuse(action: str, why: str) -> None:
    raise UpperSpliceRefusal(layered(action, why))


def _staggered(pressure, shape, xp):
    """Mass-point pressure carried to a C-grid point set of ``shape``.

    U points sit between two mass columns in x, V points in y; the edge
    points take their one neighbour.  Arithmetic mean, as WRF's real.exe
    averages ``pres`` onto the momentum points for its own vertical
    interpolation of U and V.
    """
    ny, nx = pressure.shape[-2:]
    if tuple(shape[-2:]) == (ny, nx):
        return pressure
    if tuple(shape[-2:]) == (ny, nx + 1):
        out = xp.empty(pressure.shape[:-1] + (nx + 1,), dtype=pressure.dtype)
        out[..., 1:-1] = 0.5 * (pressure[..., :-1] + pressure[..., 1:])
        out[..., 0] = pressure[..., 0]
        out[..., -1] = pressure[..., -1]
        return out
    if tuple(shape[-2:]) == (ny + 1, nx):
        out = xp.empty(pressure.shape[:-2] + (ny + 1, nx), dtype=pressure.dtype)
        out[..., 1:-1, :] = 0.5 * (pressure[..., :-1, :] + pressure[..., 1:, :])
        out[..., 0, :] = pressure[..., 0, :]
        out[..., -1, :] = pressure[..., -1, :]
        return out
    raise ValueError(f"no C-grid staggering maps mass {pressure.shape[-2:]} "
                     f"to {tuple(shape[-2:])}")


def _upper_at(values, log_levels, log_target, xp):
    """``values`` (ascending-pressure 1-D ladder) at ``log_target``, linear in ln p.

    ``log_levels`` is ascending; targets outside it take the end value
    (callers only ask inside it).
    """
    nlev = log_levels.size
    index = xp.searchsorted(log_levels, log_target.ravel(), side="right") - 1
    index = xp.clip(index, 0, nlev - 2).reshape(log_target.shape)
    lo = log_levels[index]
    hi = log_levels[index + 1]
    weight = xp.clip((log_target - lo) / (hi - lo), 0.0, 1.0)
    below = xp.take_along_axis(values, index, axis=0)
    above = xp.take_along_axis(values, index + 1, axis=0)
    return below + weight * (above - below)


def _saturation_vapour_pressure_pa(temperature, xp):
    # Bolton/WRF liquid-water relation (woof.core.constants SVP1..SVPT0).
    return 611.2 * xp.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))


def specific_humidity_from_rh(relative_humidity, temperature, pressure, xp=np):
    """WRF ``rh_to_mxrat1`` vapour, capped as it caps the stratosphere, as q.

    RH is clipped to 0..100 % and the vapour pressure to 0.99 p (a column
    so warm and so thin that saturation exceeds the pressure itself would
    otherwise divide by a non-positive dry pressure); the stratospheric
    cap then applies exactly as the initializer's RH lane applies it.
    """
    rh = xp.clip(relative_humidity, 0.0, 100.0)
    vapour = xp.minimum(0.01 * rh * _saturation_vapour_pressure_pa(temperature, xp),
                        0.99 * pressure)
    mixing = 0.622 * vapour / (pressure - vapour)
    mixing = xp.where((pressure < _QV_MAX_P_SAFE_PA) & (mixing > _QV_MAX_FLAG),
                      _QV_MAX_VALUE, mixing)
    return mixing / (1.0 + mixing)


def relative_humidity_from_specific(specific, temperature, pressure, xp=np):
    """The inverse of :func:`specific_humidity_from_rh` before its clipping."""
    mixing = specific / (1.0 - specific)
    vapour = mixing * pressure / (mixing + 0.622)
    return 100.0 * vapour / _saturation_vapour_pressure_pa(temperature, xp)


def _lower_pressure(lower, nlev, mass_shape, xp):
    pres = lower.fields.get("PRES")
    if pres is not None and getattr(pres, "ndim", 0) == 3:
        if tuple(pres.shape) != (nlev, *mass_shape):
            _refuse("The lower source's PRES does not match its level count.",
                    f"PRES has shape {tuple(pres.shape)}; TT has "
                    f"{(nlev, *mass_shape)}.")
        return xp.asarray(pres, dtype=xp.float64), True
    levels = xp.asarray(np.asarray(lower.levels_hpa, dtype=np.float64) * 100.0)
    return xp.broadcast_to(levels[:, None, None], (nlev, *mass_shape)).copy(), False


def _is_index_ladder(levels: np.ndarray) -> bool:
    """True when ``levels_hpa`` numbers hybrid levels rather than pressures.

    Native HRRR snapshots carry ``1..50`` there and their pressure in a
    3-D ``PRES`` (:data:`woof.ingest.hrrr.HRRR_HYBRID_LEVELS`); the
    initializer reads only the count.
    """
    return bool(np.array_equal(levels, np.arange(1.0, levels.size + 1.0)))


def _upper_ladder(upper):
    """The upper snapshot as an ascending-pressure 1-D ladder (Pa) + permutation."""
    levels = np.asarray(upper.levels_hpa, dtype=np.float64) * 100.0
    pres = upper.fields.get("PRES")
    if pres is not None and getattr(pres, "ndim", 0) == 3:
        host = _host(pres)
        spread = np.ptp(host.reshape(host.shape[0], -1), axis=1)
        if np.any(spread > 1e-3 * np.abs(host.reshape(host.shape[0], -1)).max(axis=1)):
            _refuse("The upper source is not on pressure levels.",
                    "The splice appends the upper source's levels as "
                    "isobaric levels; this one carries a PRES field that "
                    "varies horizontally on a level.")
    if levels.ndim != 1 or levels.size < 2 or np.any(levels <= 0.0):
        _refuse("The upper source has no usable pressure ladder.",
                f"levels_hpa is {upper.levels_hpa!r}.")
    order = np.argsort(levels)
    if np.any(np.diff(levels[order]) <= 0.0):
        _refuse("The upper source repeats a pressure level.",
                f"levels_hpa is {upper.levels_hpa!r}.")
    return levels[order], order


def splice_upper_snapshot(lower, upper, *, upper_label: str = "upper",
                          seam_bottom_pa: float = DEFAULT_SEAM_BOTTOM_PA,
                          seam_top_pa: float = DEFAULT_SEAM_TOP_PA,
                          upper_receipt: Mapping[str, object] | None = None):
    """``lower`` with ``upper`` carried above it; returns ``(snapshot, receipt)``.

    Both snapshots must be on the same target points (one grid, or one
    boundary strip of it) and valid at the same time.  The returned
    snapshot has the lower snapshot's type -- a
    :class:`woof.ingest.horiz.HorizontalSnapshot` stays one, the
    pickle-safe boundary namespaces stay namespaces -- with its level
    axis extended and ``upper_source_receipt`` set.
    """
    seam_bottom_pa = float(seam_bottom_pa)
    seam_top_pa = float(seam_top_pa)
    if not (math.isfinite(seam_bottom_pa) and math.isfinite(seam_top_pa)
            and seam_bottom_pa > seam_top_pa > 0.0):
        raise ValueError("the seam must satisfy bottom > top > 0 Pa")
    if lower.valid_time != upper.valid_time:
        _refuse(f"The {upper_label} source is valid at {upper.valid_time}, "
                f"the lower source at {lower.valid_time}.",
                "Each forcing time is joined from the two sources' analyses "
                "of the SAME instant; a column joined across times would "
                "carry the stratosphere of another hour.")
    for name in ("TT", "UU", "VV", "GHT"):
        for label, snapshot in (("lower", lower), (upper_label, upper)):
            if name not in snapshot.fields:
                _refuse(f"The {label} source carries no {name}.",
                        "The splice blends TT, UU, VV and GHT across the seam.")
    lower_moisture = "SPFH" if "SPFH" in lower.fields else "RH"
    if lower_moisture not in lower.fields:
        _refuse("The lower source carries neither SPFH nor RH.",
                "The composite column needs a moisture field.")
    if not ("RH" in upper.fields or "SPFH" in upper.fields):
        _refuse(f"The {upper_label} source carries neither RH nor SPFH.",
                "The composite column needs a moisture field.")

    tt = lower.fields["TT"]
    xp = _array_module(tt)
    nlev = int(np.asarray(lower.levels_hpa).size)
    if tt.ndim != 3 or tt.shape[0] != nlev:
        raise ValueError("lower TT must be (level, y, x) on levels_hpa")
    mass_shape = tuple(tt.shape[1:])
    upper_tt = upper.fields["TT"]
    if tuple(upper_tt.shape[1:]) != mass_shape:
        _refuse(f"The {upper_label} source was mapped onto other points.",
                f"Its TT has horizontal shape {tuple(upper_tt.shape[1:])}; "
                f"the lower source's is {mass_shape}.  Both must be "
                "interpolated onto the same target grid (or boundary strip).")

    p_lower, hybrid = _lower_pressure(lower, nlev, mass_shape, xp)
    if not bool(xp.isfinite(p_lower).all()) or bool((p_lower <= 0.0).any()):
        _refuse("The lower source's pressure is not finite and positive.", "")
    mean_profile = _host(p_lower.reshape(nlev, -1).mean(axis=1))
    bottom_first = bool(mean_profile[0] > mean_profile[-1])
    step = xp.diff(p_lower, axis=0)
    if bool((step >= 0.0).any() if bottom_first else (step <= 0.0).any()):
        _refuse("The lower source's pressure is not monotonic in every column.",
                "The splice reads each column's top from its pressure.")
    top_index = nlev - 1 if bottom_first else 0
    lower_top = p_lower[top_index]
    lower_top_max = float(_host(lower_top.max()))
    lower_top_min = float(_host(lower_top.min()))
    if lower_top_max > seam_top_pa * (1.0 + 1e-6):
        _refuse(f"The lower source stops at {lower_top_max:g} Pa, below the "
                f"{seam_top_pa:g} Pa seam top.",
                "The seam blends the two sources where BOTH hold the "
                "column; lower the seam or use a lower source reaching it.")

    p_upper, order = _upper_ladder(upper)
    if p_upper[-1] < seam_bottom_pa * (1.0 - 1e-6):
        _refuse(f"The {upper_label} source starts at {p_upper[-1]:g} Pa, above "
                f"the {seam_bottom_pa:g} Pa seam bottom.",
                "The seam needs the upper source from its bottom up.")
    if p_upper[0] > lower_top_min * (1.0 + 1e-6):
        _refuse(f"The {upper_label} source stops at {p_upper[0]:g} Pa, below "
                f"the lower source's own top ({lower_top_min:g} Pa).",
                "Above the seam the upper source replaces the lower one, so "
                "it has to reach at least as high.")
    log_upper = xp.asarray(np.log(p_upper))
    order_device = xp.asarray(order)

    def upper_field(name):
        value = xp.asarray(upper.fields[name], dtype=xp.float64)
        return value[order_device]

    upper_tt64 = upper_field("TT")
    p_upper_3d = xp.asarray(p_upper)[:, None, None]
    if "RH" in upper.fields and lower_moisture == "SPFH":
        upper_moisture = specific_humidity_from_rh(
            upper_field("RH"), upper_tt64, p_upper_3d, xp)
        moisture_rule = ("upper RH -> SPFH: Bolton liquid saturation "
                         "(WRF rh_to_mxrat1) with its stratospheric qv cap")
    elif "SPFH" in upper.fields and lower_moisture == "RH":
        upper_moisture = relative_humidity_from_specific(
            upper_field("SPFH"), upper_tt64, p_upper_3d, xp)
        moisture_rule = "upper SPFH -> RH: inverse Bolton liquid saturation"
    else:
        upper_moisture = upper_field(lower_moisture)
        moisture_rule = f"{lower_moisture} carried unchanged"

    upper_columns = {"TT": upper_tt64, "UU": upper_field("UU"),
                     "VV": upper_field("VV"), "GHT": upper_field("GHT"),
                     lower_moisture: upper_moisture}
    upper_columns["HGT"] = upper_columns["GHT"]

    # Lower levels reaching the seam: everything above its bottom anywhere.
    log_bottom = math.log(seam_bottom_pa)
    log_span = math.log(seam_bottom_pa / seam_top_pa)
    reached = _host((p_lower <= seam_bottom_pa).reshape(nlev, -1).any(axis=1))
    touched = np.flatnonzero(reached)
    in_seam = _host(((p_lower <= seam_bottom_pa) & (p_lower >= seam_top_pa))
                    .reshape(nlev, -1).any(axis=1))
    if not in_seam.any():
        _refuse("No lower level lies inside the seam, so the two sources "
                "cannot be compared there.",
                f"The seam is {seam_bottom_pa:g}..{seam_top_pa:g} Pa.")

    # Appended levels: the upper ladder above the whole lower column.
    keep = p_upper < lower_top_min * math.exp(-MIN_APPENDED_LOG_SEPARATION)
    appended_index = np.flatnonzero(keep)
    appended_p = p_upper[appended_index]
    if appended_index.size and bottom_first:
        # Highest pressure first, continuing the bottom-first ladder.
        appended_index = appended_index[::-1]
        appended_p = appended_p[::-1]
    appended_device = xp.asarray(appended_index)

    def weights(pressure):
        log_p = xp.log(pressure)
        return xp.clip((log_bottom - log_p) / log_span, 0.0, 1.0), log_p

    statistics: dict[str, float] = {}

    def compare(name, lower_values, upper_values, pressure):
        seam = (pressure <= seam_bottom_pa) & (pressure >= seam_top_pa)
        difference = xp.where(seam, lower_values - upper_values, 0.0)
        count = int(_host(seam.sum()))
        if count == 0:
            return
        worst = float(_host(xp.abs(difference).max()))
        mean = float(_host(difference.sum())) / count
        previous = statistics.get(f"max_abs_{name}", 0.0)
        statistics[f"max_abs_{name}"] = max(previous, worst)
        statistics.setdefault(f"mean_{name}", mean)

    new_fields: dict[str, object] = {}
    zeroed: list[str] = []
    held: list[str] = []
    composite_count = nlev + appended_index.size

    def assemble(lower_values, appended_values):
        if appended_values is None or appended_index.size == 0:
            return lower_values
        parts = ((lower_values, appended_values) if bottom_first
                 else (appended_values, lower_values))
        return xp.concatenate(parts, axis=0)

    for name, value in lower.fields.items():
        ndim = getattr(value, "ndim", 0)
        if ndim != 3 or value.shape[0] != nlev:
            new_fields[name] = value
            continue
        dtype = value.dtype
        if name in _BLENDED or name == lower_moisture:
            pressure = _staggered(p_lower, value.shape, xp)
            blended = xp.asarray(value, dtype=xp.float64).copy()
            if touched.size:
                sub_p = pressure[touched]
                w, log_p = weights(sub_p)
                upper_here = _upper_at(_staggered_upper(upper_columns[name],
                                                        value.shape, xp),
                                       log_upper, log_p, xp)
                lower_here = blended[touched]
                if name in ("TT", "GHT", "UU", "VV"):
                    compare({"TT": "temperature_K", "GHT": "height_m",
                             "UU": "u_wind_ms", "VV": "v_wind_ms"}[name],
                            lower_here, upper_here, sub_p)
                blended[touched] = (1.0 - w) * lower_here + w * upper_here
            appended = (_staggered_upper(upper_columns[name], value.shape, xp)
                        [appended_device] if appended_index.size else None)
            new_fields[name] = assemble(blended, appended).astype(dtype)
        elif name == "PRES":
            appended = (xp.broadcast_to(
                xp.asarray(appended_p)[:, None, None],
                (appended_p.size, *value.shape[1:])) if appended_index.size
                else None)
            new_fields[name] = assemble(
                xp.asarray(value, dtype=xp.float64), appended).astype(dtype)
        elif name in _ZERO_ABOVE:
            pressure = _staggered(p_lower, value.shape, xp)
            w, _ = weights(pressure)
            tapered = xp.asarray(value, dtype=xp.float64) * (1.0 - w)
            appended = (xp.zeros((appended_index.size, *value.shape[1:]))
                        if appended_index.size else None)
            new_fields[name] = assemble(tapered, appended).astype(dtype)
            zeroed.append(name)
        elif name in _HOLD_ABOVE:
            top = xp.asarray(value[top_index], dtype=xp.float64)
            appended = (xp.broadcast_to(top, (appended_index.size, *top.shape))
                        if appended_index.size else None)
            new_fields[name] = assemble(
                xp.asarray(value, dtype=xp.float64), appended).astype(dtype)
            held.append(name)
        else:
            _refuse(f"The lower source carries a column field {name!r} the "
                    "upper-source splice has no rule for.",
                    "Every column field must be blended, tapered to zero, "
                    "held above, or carried as pressure; extending an "
                    "unknown one with invented values is refused.  Add it "
                    "to woof/ingest/upper_splice.py with its rule.")

    # The seam itself, then the composite column.
    limits = (("max_abs_temperature_K", MAX_SEAM_TEMPERATURE_DIFFERENCE_K,
               "temperature", "K"),
              ("max_abs_height_m", MAX_SEAM_HEIGHT_DIFFERENCE_M,
               "geopotential height", "m"),
              ("max_abs_u_wind_ms", MAX_SEAM_WIND_DIFFERENCE_MS,
               "grid-relative u wind", "m/s"),
              ("max_abs_v_wind_ms", MAX_SEAM_WIND_DIFFERENCE_MS,
               "grid-relative v wind", "m/s"))
    for key, bound, label, unit in limits:
        observed = statistics.get(key)
        if observed is not None and not observed <= bound:
            _refuse(f"The lower and {upper_label} sources disagree in the "
                    f"{seam_bottom_pa / 100:g}..{seam_top_pa / 100:g} hPa seam: "
                    f"{label} differs by up to {observed:.1f} {unit} "
                    f"(bound {bound:g} {unit}).",
                    "Two analyses of the same lower stratosphere agree to a "
                    "few units; a difference this large is a wrong valid "
                    "time, a wrong unit, a wrong field or a wind left "
                    "earth-relative on one side.  Blending across it would "
                    "put a step into the initial state and every boundary.")
    height = new_fields["GHT"]
    composite_p = new_fields.get("PRES")
    if composite_p is None or getattr(composite_p, "ndim", 0) != 3:
        composite_p = assemble(p_lower, (xp.broadcast_to(
            xp.asarray(appended_p)[:, None, None],
            (appended_p.size, *mass_shape)) if appended_index.size else None))
    dz = xp.diff(xp.asarray(height, dtype=xp.float64), axis=0)
    dp = xp.diff(xp.asarray(composite_p, dtype=xp.float64), axis=0)
    if not bottom_first:
        dz, dp = -dz, -dp
    if bool((dz <= 0.0).any()):
        _refuse("The composite geopotential height does not increase upward "
                "in every column.",
                f"{int(_host((dz <= 0.0).sum()))} layer(s) of the joined "
                f"lower+{upper_label} column fold back on themselves.")
    if bool((dp >= 0.0).any()):
        _refuse("The composite pressure does not decrease upward in every "
                "column.", "")
    if appended_index.size:
        appended_tt = _host(upper_columns["TT"][appended_device])
        low, high = APPENDED_TEMPERATURE_RANGE_K
        if not (np.isfinite(appended_tt).all()
                and appended_tt.min() >= low and appended_tt.max() <= high):
            _refuse(f"The {upper_label} source's temperature above the lower "
                    f"column leaves {low:g}..{high:g} K.",
                    f"Observed {appended_tt.min():.1f}..{appended_tt.max():.1f} K.")

    levels = np.asarray(lower.levels_hpa, dtype=np.float64)
    if appended_index.size:
        if hybrid and _is_index_ladder(levels):
            # An index ladder stays an index ladder: the initializer reads
            # only its length, and the pressure rides PRES.
            levels = np.arange(1.0, composite_count + 1.0)
        else:
            # ``appended_p`` is already in the lower ladder's order.
            extra = appended_p / 100.0
            levels = (np.concatenate((levels, extra)) if bottom_first
                      else np.concatenate((extra, levels)))
    receipt = {
        "schema": SCHEMA,
        "upper_source": upper_label,
        "valid_time": lower.valid_time.isoformat(),
        "seam_pa": [seam_bottom_pa, seam_top_pa],
        "blend": "linear in ln p: w = 0 at the seam bottom, 1 at the seam top",
        "lower_levels": nlev,
        "lower_top_pa": [lower_top_min, lower_top_max],
        "lower_levels_in_seam": int(in_seam.sum()),
        "appended_levels_hpa": [float(value) / 100.0 for value in appended_p],
        "composite_top_pa": float(min(lower_top_min, *appended_p)
                                  if appended_p.size else lower_top_min),
        "moisture": moisture_rule,
        "zeroed_above": sorted(zeroed),
        "held_above": sorted(held),
        "seam_statistics": {key: round(value, 4)
                            for key, value in sorted(statistics.items())},
        "seam_bounds": {"temperature_K": MAX_SEAM_TEMPERATURE_DIFFERENCE_K,
                        "height_m": MAX_SEAM_HEIGHT_DIFFERENCE_M,
                        "wind_ms": MAX_SEAM_WIND_DIFFERENCE_MS},
        **({"upper": dict(upper_receipt)} if upper_receipt else {}),
    }
    return _rebuilt(lower, levels, new_fields, receipt), receipt


def _staggered_upper(values, shape, xp):
    """The upper column on the lower field's point set (they share the grid)."""
    if tuple(values.shape[1:]) != tuple(shape[1:]):
        raise ValueError(
            f"upper field shape {tuple(values.shape)} differs from the lower "
            f"field's {tuple(shape)} on the same points")
    return values


def _rebuilt(lower, levels, fields, receipt):
    if is_dataclass(lower):
        names = {item.name for item in dataclass_fields(lower)}
        extra = ({"upper_source_receipt": receipt}
                 if "upper_source_receipt" in names else {})
        return replace(lower, levels_hpa=levels, fields=fields, **extra)
    rebuilt = SimpleNamespace(**vars(lower))
    rebuilt.levels_hpa = levels
    rebuilt.fields = fields
    rebuilt.upper_source_receipt = receipt
    return rebuilt


def receipt_sentence(receipt: Mapping[str, object]) -> str:
    """One log line naming what the splice did."""
    stats = receipt.get("seam_statistics", {})
    appended = receipt.get("appended_levels_hpa", [])
    seam = receipt["seam_pa"]
    return (
        f"upper source: {receipt['upper_source']} carries the column above "
        f"{seam[1] / 100:g} hPa (blended in ln p from {seam[0] / 100:g} hPa"
        + (f", +{len(appended)} level(s) to {min(appended):g} hPa"
           if appended else "")
        + f"); seam max|dT| {stats.get('max_abs_temperature_K', float('nan')):.2f} K, "
        f"max|dZ| {stats.get('max_abs_height_m', float('nan')):.1f} m, "
        f"max|dU,dV| {max(stats.get('max_abs_u_wind_ms', 0.0), stats.get('max_abs_v_wind_ms', 0.0)):.1f} m/s")
