"""Wind-farm power at listed turbines: rotor-equivalent wind, power, thrust.

A wind-farm forecast is graded on GENERATION, and generation is a
non-linear function of the wind over the whole rotor disk: the cube of a
speed averaged over a history interval is not the average of its cubes,
and a hub-height plane says nothing about the shear across a 150 m rotor.
So with ``[output] windfarm_turbines = "turbines.csv"`` this diagnostic
evaluates, after EVERY completed model step and at each listed turbine,

``REWS_TURBINE``
    the IEC 61400-12-1:2017 rotor-equivalent wind speed over the disk;
``POWER_TURBINE``
    electrical power (W) from the turbine's own
    :class:`woof.energy.turbine_library.TurbineCurve` at the
    density-corrected REWS;
``CT_TURBINE``
    the thrust coefficient from the same curve
    (:meth:`TurbineCurve.thrust_coefficient`) at the same speed;
``POWER_FARM`` / ``POWER_FARM_MEAN`` / ``WINDFARM_WINDOW_S``
    the domain's summed power at the frame, its time-weighted mean over
    the window ``(T_prev_frame, T_frame]``, and how many model seconds the
    window covers,

and flushes the interval mean and max of every per-turbine quantity to a
sidecar per history write (``windfarm_d0N_<valid time>.nc``), the way
``[output] hub_sites`` flushes the hub-wind series.  It reads the
RESOLVED model winds, so it works on any run, with or without a turbine
drag parameterisation feeding back on those winds: with one, the winds
it reads are already waked; without one, it is a free-stream estimate.

THE ARITHMETIC, PINNED, AND THE SAME WORDS ON BOTH ROADS.  On the device
it is ONE inline CUDA kernel, one thread per turbine, and a one-thread
kernel that folds the farm sum; the NumPy functions below are its
operation-for-operation mirror and the CPU road.  The kernel is compiled
straight through NVRTC with ``-fmad=false --ftz=false`` (no contraction,
no flush).  Its transcendentals are ``pow`` from the glibc-word device
libm ``woof/core/kernels/glibc_flt64.cuh`` (the host takes the C
library's ``pow`` through :mod:`woof.core.host_libm`) and the correctly
rounded ``cbrt`` of ``woof/core/kernels/cr_cbrt_flt64.cuh`` (the host
takes its NumPy twin, :mod:`woof.core.cr_cbrt`); every sum runs in a
fixed index order.  So every output word is equal on the two roads (max
ULP 0, ``tests/test_windfarm_power_diag_gpu.py``), and equal to the
independent oracles each step names:

1. each turbine maps to its NEAREST mass cell (``locate_sites`` of
   :mod:`woof.core.hub_wind_diag`, so a turbine is inside or outside a
   domain by exactly the hub-site rule);
2. that column's mass-level heights above ground and destaggered winds,
   formed exactly as the hub-wind diagnostic forms them (steps 1-2 of
   :mod:`woof.core.hub_wind_diag`), in float32;
3. the disk is cut into ``N_ROTOR_SEGMENTS`` equal-height horizontal
   strips (:func:`woof.energy.ramps.rotor_segments`: centre heights, and
   weights that are each strip's exact share of the disk area); u, v at
   every strip centre and at the hub, and theta, full pressure and vapour
   at the hub, are interpolated linear in z-AGL between the bracketing
   mass levels with the hub-wind hold rule (step 3 there: below the
   lowest mass level holds level 0, above the top holds the top), in
   float32 -- the words of
   :func:`woof.core.hub_wind_diag.interp_to_heights_np` (and so of the
   hub-wind kernel: the interpolated hub u, v give ``WSPD_HUB``'s words,
   the hub pressure and vapour ``P_HUB``'s and ``QV_HUB``'s);
4. in float64 from here on: ``REWS = (sum_i w_i (s_i cos phi_i)^3)^(1/3)``
   summed strip 0 first, with ``s_i`` the strip speed and ``phi_i`` its
   direction relative to the hub wind, ``cos phi`` clipped to ``[0, 1]``
   (1 where either wind is calm) --
   :func:`woof.energy.ramps.rotor_equivalent_wind_speed`'s rule, whose
   strips are interpolated in ln(height) from a forecast's heights, so its
   REWS is not this one's; the root is the correctly rounded cbrt
   (:func:`woof.core.cr_cbrt.cbrt`, ``uw_cbrt`` on the device);
5. hub moist density ``rho = p / (R_d (T (1 + 0.608 q)))``,
   ``T = theta (p / P0)^(R_d/c_p)``, ``q = qv / (1 + qv)``: the words of
   :func:`woof.energy.extract.air_density` of ``air_temperature`` (``qv =
   0`` on a dry state); and the IEC pitch-regulated density correction
   ``v_eq = REWS (rho / 1.225)^(1/3)``: the words of
   :func:`woof.energy.ramps.density_corrected_speed` and of
   :func:`woof.energy.power_farm.density_corrected_speed`, which take the
   same correctly rounded cube root (CORE-MATH's, transcribed for the host
   and the device alike);
6. ``POWER = fraction(v_eq) * (rated_kw * 1000)`` W and ``CT = Ct(v_eq)``,
   both ZERO above the curve's ``cut_out_ms`` (no restart hysteresis: a
   diagnostic carries no running state across a cut-out, where
   ``power_farm.running_state`` does).  ``fraction`` is the words of
   :meth:`TurbineCurve.fraction` (the sub-step ``woof energy power``
   shares) and ``Ct`` the words of :meth:`TurbineCurve.thrust_coefficient`
   (its 40-step bisection, transcribed).

WHEN IT FOLDS, AND WHEN IT RESETS.  Folded in the dycore epilogue of
every completed step through :mod:`woof.core.interval_diagnostics`, the
instant the hub-wind window folds; the frame's own values are evaluated
AT THE FRAME, from the state the frame is written from, because the
post-RK microphysics adjustment moves theta and pressure (and so the
density) after the epilogue.  The window restarts immediately after each
history frame is snapshotted.  A frame with no step behind it (the cold
start) reports MEAN = MAX = the frame's own value and a zero window.

RESTART SEMANTICS (documented, not checkpointed), as hub-height wind: the
window is output, not model state, binds no restart identity, and a
resumed run starts a fresh window at the restart instant.

NOT SUPPORTED, AND REFUSED BY NAME (:class:`WindfarmPowerUnsupportedRoad`):
``[tiles]``-streamed domains, ensemble member capture and nest
relocation -- the roads that bypass the dycore epilogue or move the
turbines' cells.

COST.  Per step, two kernel launches and no host synchronisation: the
turbine indices, rotor-strip heights, curve points and the window all
live on the device, and nothing crosses to the host until a frame is
written, when one packed ``11 n_turbine + 2`` float64 buffer is copied
(the frame's values and the window).  A NumPy-composed version of the
same arithmetic -- dozens of small CuPy launches per step -- measured
6-9 ms per step for nine turbines on an RTX A6000; the fused kernel is
the reason it is not that.  Device memory is ``O(n_turbine)`` plus each
distinct curve's points.
"""
from __future__ import annotations

import csv
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path

import numpy as np

from woof.core.constants import CP, G, P0, RD
from woof.core.cr_cbrt import cbrt as _cbrt
from woof.core.hub_wind_diag import (_array_module, _epoch_seconds,
                                     locate_sites)
from woof.io.wrf_output_schema import WINDFARM_POWER_OUTPUT_FIELDS

DTYPE = np.float32

#: The registry name (:mod:`woof.core.interval_diagnostics`).
DIAGNOSTIC_NAME = "windfarm_power"

#: The turbine CSV's required columns, in the order they are documented.
TURBINE_FILE_COLUMNS: tuple[str, ...] = (
    "name", "lat", "lon", "hub_height_m", "rotor_diameter_m", "turbine_type")
#: Columns a row MAY carry: ``rated_kw`` is required for, and only read
#: for, a ``generic`` turbine.
OPTIONAL_TURBINE_FILE_COLUMNS: tuple[str, ...] = ("rated_kw",)
#: The ``turbine_type`` that asks for a parametric curve at the row's own
#: rated power and rotor diameter (:func:`woof.energy.turbine_library.
#: generic_curve`) instead of a library entry.
GENERIC_TURBINE_TYPE = "generic"
#: Bounds that keep a typo from becoming a silent allocation or a rotor
#: that is not a rotor.
MAX_TURBINES = 5000
MAX_HUB_HEIGHT_M = 2000.0
MAX_ROTOR_DIAMETER_M = 400.0
#: A library curve's rotor and the file's may differ by this much (the
#: library's own fuzzy-match rotor tolerance); more is refused, because
#: the disk integrated and the curve read would be two different machines.
ROTOR_TOLERANCE_M = 1.5

#: Strips the rotor disk is cut into (``woof.energy.ramps``'s default).
N_ROTOR_SEGMENTS = 10

_G = DTYPE(G)
_F05 = DTYPE(0.5)
#: IEC 61400-12-1 reference density of every library curve, kg m-3.
REFERENCE_DENSITY = 1.225
_VIRT = 0.608


class WindfarmPowerError(RuntimeError):
    """The wind-farm power diagnostic cannot run as configured."""


class WindfarmPowerUnsupportedRoad(WindfarmPowerError):
    """A run road that would bypass the window (streamed, captured, moved)."""


class TurbineFileError(ValueError):
    """A turbine CSV row or header the diagnostic cannot honour."""


# ---------------------------------------------------------------------------
# The turbine file
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Turbine:
    """One validated row of the turbine CSV.

    PINNED: ``lon`` is normalised to ``[-180, 180)``; ``hub_height_m``
    exceeds half the rotor diameter (the lower tip is above ground);
    ``turbine_type`` is a turbine-library curve id (any library name or
    alias is resolved to its id when the file is read) or
    :data:`GENERIC_TURBINE_TYPE`, and ``rated_kw`` is set exactly when it
    is generic.
    """

    name: str
    lat: float
    lon: float
    hub_height_m: float
    rotor_diameter_m: float
    turbine_type: str
    rated_kw: float | None = None


def _number(text: str, what: str, where: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise TurbineFileError(
            f"{where}: {what} must be a number, got {text!r}.") from None
    if not math.isfinite(value):
        raise TurbineFileError(f"{where}: {what} must be finite, got {text!r}.")
    return value


def _library():
    from woof.energy.turbine_library import load_turbine_library
    return load_turbine_library()


def _resolve_type(text: str, library, where: str) -> str:
    """A library curve id for ``text``, or :data:`GENERIC_TURBINE_TYPE`."""
    if text == GENERIC_TURBINE_TYPE:
        return text
    if text in library.curves:
        return text
    from woof.energy.turbine_library import compact
    ids = library._exact.get(compact(text), [])
    if len(ids) == 1:
        return ids[0]
    if len(ids) > 1:
        raise TurbineFileError(
            f"{where}: turbine_type {text!r} names {len(ids)} library curves "
            f"({', '.join(sorted(ids))}); write the curve id you mean.")
    import difflib
    close = difflib.get_close_matches(text, sorted(library.curves), n=3)
    hint = f"  Did you mean {', '.join(close)}?" if close else ""
    raise TurbineFileError(
        f"{where}: turbine_type {text!r} is not a curve of the turbine "
        f"library {library.path}.{hint}  Use a library curve id, or "
        f"{GENERIC_TURBINE_TYPE!r} with a rated_kw column for a parametric "
        "curve at the row's rated power and rotor diameter.")


def read_turbine_file(path, *, library=None) -> tuple[Turbine, ...]:
    """Every row of a turbine CSV, validated, in file order.

    The header names :data:`TURBINE_FILE_COLUMNS` (any order) and may add
    :data:`OPTIONAL_TURBINE_FILE_COLUMNS`; blank lines and lines starting
    with ``#`` are skipped.  Every contradiction is refused by row.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as err:
        raise TurbineFileError(f"{path} cannot be read: {err}") from None
    numbered = [(index + 1, line) for index, line in
                enumerate(text.splitlines())
                if line.strip() and not line.lstrip().startswith("#")]
    if not numbered:
        raise TurbineFileError(f"{path} names no turbines.")
    rows = list(csv.reader([line for _, line in numbered],
                           skipinitialspace=True))
    header = [cell.strip() for cell in rows[0]]
    missing = [c for c in TURBINE_FILE_COLUMNS if c not in header]
    allowed = set(TURBINE_FILE_COLUMNS) | set(OPTIONAL_TURBINE_FILE_COLUMNS)
    unknown = [c for c in header if c not in allowed]
    if missing or unknown or len(set(header)) != len(header):
        raise TurbineFileError(
            f"{path} line {numbered[0][0]}: the header must name "
            f"{', '.join(TURBINE_FILE_COLUMNS)} once each (and may add "
            f"{', '.join(OPTIONAL_TURBINE_FILE_COLUMNS)}); got {header}"
            + (f", missing {missing}" if missing else "")
            + (f", unknown {unknown}" if unknown else "") + ".")
    if len(rows) - 1 > MAX_TURBINES:
        raise TurbineFileError(
            f"{path} lists {len(rows) - 1} turbines; at most {MAX_TURBINES} "
            "are carried.")
    library = _library() if library is None else library
    turbines, seen = [], set()
    for (lineno, _), cells in zip(numbered[1:], rows[1:]):
        where = f"{path} line {lineno}"
        if len(cells) != len(header):
            raise TurbineFileError(
                f"{where} has {len(cells)} fields; the header has "
                f"{len(header)}.")
        row = dict(zip(header, (cell.strip() for cell in cells)))
        name = row["name"]
        if not name:
            raise TurbineFileError(f"{where}: name is empty.")
        if name in seen:
            raise TurbineFileError(f"{where} repeats turbine name {name!r}.")
        seen.add(name)
        lat = _number(row["lat"], "lat", where)
        lon = _number(row["lon"], "lon", where)
        hub = _number(row["hub_height_m"], "hub_height_m", where)
        rotor = _number(row["rotor_diameter_m"], "rotor_diameter_m", where)
        if not -90.0 <= lat <= 90.0:
            raise TurbineFileError(f"{where}: lat {lat} is outside [-90, 90].")
        if not -180.0 <= lon < 360.0:
            raise TurbineFileError(
                f"{where}: lon {lon} is outside [-180, 360).")
        if not 0.0 < rotor <= MAX_ROTOR_DIAMETER_M:
            raise TurbineFileError(
                f"{where}: rotor_diameter_m {rotor:g} must be in "
                f"(0, {MAX_ROTOR_DIAMETER_M:g}] m.")
        if not 0.5 * rotor < hub <= MAX_HUB_HEIGHT_M:
            raise TurbineFileError(
                f"{where}: hub_height_m {hub:g} must exceed half the rotor "
                f"diameter ({0.5 * rotor:g} m: the lower tip is above "
                f"ground) and be at most {MAX_HUB_HEIGHT_M:g} m.")
        kind = _resolve_type(row["turbine_type"], library, where)
        rated_text = row.get("rated_kw", "")
        rated = None
        if kind == GENERIC_TURBINE_TYPE:
            if not rated_text:
                raise TurbineFileError(
                    f"{where}: a {GENERIC_TURBINE_TYPE!r} turbine needs "
                    "rated_kw (the parametric curve is built from the rated "
                    "power and the rotor diameter).")
            rated = _number(rated_text, "rated_kw", where)
            if not rated > 0.0:
                raise TurbineFileError(
                    f"{where}: rated_kw must be positive, got {rated:g}.")
        else:
            if rated_text:
                raise TurbineFileError(
                    f"{where}: rated_kw is read only for a "
                    f"{GENERIC_TURBINE_TYPE!r} turbine; library curve "
                    f"{kind!r} carries its own rating.  Delete the value.")
            curve_rotor = library.curves[kind].rotor_diameter_m
            if abs(curve_rotor - rotor) > ROTOR_TOLERANCE_M:
                raise TurbineFileError(
                    f"{where}: rotor_diameter_m {rotor:g} disagrees with "
                    f"library curve {kind!r}'s {curve_rotor:g} m.  Correct "
                    f"the diameter, or use {GENERIC_TURBINE_TYPE!r} with "
                    "rated_kw for a machine the library does not carry.")
        # Only the [180, 360) half moves, so a longitude already in range
        # keeps its exact float (the hub_sites rule).
        turbines.append(Turbine(
            name=name, lat=lat, lon=lon - 360.0 if lon >= 180.0 else lon,
            hub_height_m=hub, rotor_diameter_m=rotor, turbine_type=kind,
            rated_kw=rated))
    if not turbines:
        raise TurbineFileError(f"{path} names no turbines.")
    return tuple(turbines)


def turbine_curve(turbine: Turbine, library=None):
    """The :class:`TurbineCurve` a validated row names."""
    from woof.energy.turbine_library import generic_curve
    if turbine.turbine_type == GENERIC_TURBINE_TYPE:
        return generic_curve(
            float(turbine.rated_kw), rotor_m=float(turbine.rotor_diameter_m),
            label=(f"generic_{turbine.rated_kw:g}kW_"
                   f"{turbine.rotor_diameter_m:g}m"))
    library = _library() if library is None else library
    return library.get(turbine.turbine_type)


# ---------------------------------------------------------------------------
# The arithmetic: the CPU road (NumPy, with the C library's pow through
# woof.core.host_libm), which the device kernel below mirrors word for word
# ---------------------------------------------------------------------------

#: The rotor-equivalent and density-correction cube root is CORE-MATH's
#: correctly rounded cbrt: :func:`woof.core.cr_cbrt.cbrt` here and
#: ``uw_cbrt`` (woof/core/kernels/cr_cbrt_flt64.cuh) in the kernel, the
#: same algorithm and, being correctly rounded, the same word as every
#: other correctly rounded cbrt -- the one the energy products' density
#: correction takes too.  Neither ``pow(x, 1/3)`` nor the C library's
#: ``cbrt`` is correctly rounded (tools/uwpbl_wrf471_oracle/libm64/README.md
#: measures both).  Imported at the top of the module as ``_cbrt``.

#: Bisection steps of :meth:`TurbineCurve.thrust_coefficient` (its own 40).
CT_BISECTION_STEPS = 40


def _pow(base, exponent):
    """``base ** exponent`` through the C library (woof.core.host_libm)."""
    from woof.core import host_libm
    base = np.asarray(base, dtype=np.float64)
    return host_libm.power(base, np.full(base.shape, float(exponent)))


def sequential_sum(values) -> float:
    """``((0 + v0) + v1) + ...`` in index order: the farm sum, the one
    order both roads use (the device's farm kernel is one thread)."""
    total = 0.0
    for value in np.asarray(values, dtype=np.float64).ravel():
        total = total + float(value)
    return total


def interp_columns(xp, fields, z, targets):
    """Linear-in-z of ``(nz, n)`` ``fields`` to per-column ``targets``.

    ``z`` is ``(nz, n)`` metres above ground, increasing with ``k``;
    ``targets`` is ``(k, n)``.  The hub-wind rule, operation for operation
    (:func:`woof.core.hub_wind_diag.interp_to_heights_np`): a target below
    ``z[0]`` holds level 0, one at or above ``z[nz-1]`` the top.
    """
    nz = int(z.shape[0])
    kb = xp.sum(z[None, :, :] <= targets[:, None, :], axis=1) - 1
    below = kb < 0
    above = kb >= nz - 1
    lo = xp.clip(kb, 0, nz - 2)
    hi = lo + 1
    z_lo = xp.take_along_axis(z, lo, axis=0)
    z_hi = xp.take_along_axis(z, hi, axis=0)
    with np.errstate(all="ignore"):
        w = (targets - z_lo) / (z_hi - z_lo)
    out = []
    for field in fields:
        f_lo = xp.take_along_axis(field, lo, axis=0)
        f_hi = xp.take_along_axis(field, hi, axis=0)
        value = f_lo + w * (f_hi - f_lo)
        value = xp.where(below, field[0][None, :], value)
        value = xp.where(above, field[nz - 1][None, :], value)
        out.append(value)
    return out


def turbine_columns(xp, state, sj, si) -> dict:
    """The ``(nz, n)`` float32 columns of the turbines' cells.

    Heights and destaggered winds formed exactly as
    :func:`woof.core.hub_wind_diag.mass_heights_agl_np` and
    :func:`~woof.core.hub_wind_diag.destagger_np` form them.
    """
    u, v = state.u, state.v
    um = _F05 * (u[:, sj, si] + u[:, sj, si + 1])
    vm = _F05 * (v[:, sj, si] + v[:, sj + 1, si])
    phb = state.phb[:, None] if state.phb.ndim == 1 else state.phb[:, sj, si]
    z_w = (state.php[:, sj, si] + phb) / _G
    z = _F05 * (z_w[:-1] + z_w[1:]) - state.ht[sj, si]
    thb = state.thb[:, None] if state.thb.ndim == 1 else state.thb[:, sj, si]
    columns = {"z": z, "u": um, "v": vm,
               "theta": state.thp[:, sj, si] + thb, "p": state.p[:, sj, si]}
    qv = getattr(state, "qv", None)
    if qv is not None:
        columns["qv"] = qv[:, sj, si]
    return columns


def curve_fraction(speed, curves, curve_index) -> np.ndarray:
    """:meth:`TurbineCurve.fraction` of each turbine's own curve, verbatim:
    ``np.interp(v, speeds, power_kw, left=0) / rated_kw``, NaN -> NaN."""
    speed = np.asarray(speed, dtype=np.float64)
    curve_index = np.asarray(curve_index)
    out = np.empty_like(speed)
    # One vectorised interp per distinct curve: the same per-element
    # arithmetic as the library's own call on that curve.
    for index, curve in enumerate(curves):
        mine = curve_index == index
        if mine.any():
            out[mine] = np.interp(speed[mine], curve.speeds, curve.power_kw,
                                  left=0.0) / curve.rated_kw
    return np.where(np.isfinite(speed), out, np.nan)


def thrust_coefficient(speed, curves, curve_index, half_rho_area) -> np.ndarray:
    """:meth:`TurbineCurve.thrust_coefficient`, operation for operation, with
    its ``v ** 3`` through the C library's pow (host_libm)."""
    from woof.energy.turbine_library import (DRIVETRAIN_EFFICIENCY,
                                             ROTOR_EFFICIENCY)
    v = np.asarray(speed, dtype=np.float64)
    rated_kw = np.array([c.rated_kw for c in curves],
                        dtype=np.float64)[np.asarray(curve_index)]
    power_w = curve_fraction(np.maximum(v, 0.0), curves,
                             curve_index) * rated_kw * 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        cp = power_w / (half_rho_area * _pow(np.maximum(v, 1e-6), 3.0)
                        * DRIVETRAIN_EFFICIENCY * ROTOR_EFFICIENCY)
    cp = np.clip(np.nan_to_num(cp, nan=0.0), 0.0, 16.0 / 27.0 * 0.999)
    lo = np.zeros_like(cp)
    hi = np.full_like(cp, 1.0 / 3.0)
    for _ in range(CT_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        above = 4.0 * mid * (1.0 - mid) ** 2 > cp
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    a = 0.5 * (lo + hi)
    return np.where(cp > 0.0, 4.0 * a * (1.0 - a), 0.0)


def evaluate_turbines(columns, targets, weights, *, curves, curve_index,
                      rated_w, half_rho_area, cut_out_ms) -> dict:
    """Steps 3-6 of the module docstring for every turbine, on the host.

    ``targets`` is ``(N_ROTOR_SEGMENTS + 1, n)`` float32: the strip centres
    then the hub.  ``curves`` are the distinct :class:`TurbineCurve`
    objects and ``curve_index`` the one each turbine reads; ``rated_w`` is
    ``rated_kw * 1000`` and ``half_rho_area`` ``0.5 * 1.225 * area`` of
    the curve, per turbine.  Returns float64 ``(n,)`` arrays ``rews``,
    ``rho``, ``v_eq``, ``power`` (W) and ``ct``, plus the float32 hub
    interpolants ``u_hub``/``v_hub``/``theta_hub``/``p_hub``/``qv_hub``.
    """
    names = ["u", "v", "theta", "p"] + (["qv"] if "qv" in columns else [])
    at = dict(zip(names, interp_columns(np, [columns[n] for n in names],
                                        columns["z"], targets)))
    f64 = {name: value.astype(np.float64) for name, value in at.items()}
    uh, vh = f64["u"][-1], f64["v"][-1]
    hub_norm = np.sqrt(uh * uh + vh * vh)
    cubed = np.zeros_like(uh)
    for s in range(N_ROTOR_SEGMENTS):
        us, vs = f64["u"][s], f64["v"][s]
        speed = np.sqrt(us * us + vs * vs)
        norm = speed * hub_norm
        with np.errstate(all="ignore"):
            cosine = (us * uh + vs * vh) / np.where(norm > 0.0, norm, 1.0)
        cosine = np.where(norm > 0.0,
                          np.minimum(np.maximum(cosine, 0.0), 1.0), 1.0)
        along = np.maximum(speed * cosine, 0.0)
        cubed = cubed + float(weights[s]) * (along * along * along)
    rews = _cbrt(cubed)
    theta, p = f64["theta"][-1], f64["p"][-1]
    q = f64["qv"][-1] if "qv" in f64 else np.zeros_like(p)
    temperature = theta * _pow(p / P0, RD / CP)
    rho = p / (RD * (temperature * (1.0 + _VIRT * (q / (1.0 + q)))))
    v_eq = rews * _cbrt(rho / REFERENCE_DENSITY)
    fraction = curve_fraction(v_eq, curves, curve_index)
    ct = thrust_coefficient(v_eq, curves, curve_index, half_rho_area)
    finite = np.isfinite(v_eq)
    running = finite & (v_eq <= cut_out_ms)
    power = np.where(finite, np.where(running, fraction * rated_w, 0.0),
                     np.nan)
    ct = np.where(finite, np.where(running, ct, 0.0), np.nan)
    out = {"rews": rews, "rho": rho, "v_eq": v_eq, "power": power, "ct": ct}
    out.update({f"{name}_hub": at[name][-1] for name in names})
    return out


# ---------------------------------------------------------------------------
# The CUDA kernels.  Inline, the hub-wind precedent (woof/core/hub_wind_diag.py
# says why a woof/core/kernels/*.cu unit is not used), prefixed by the
# device libm LIBM_HEADERS (glibc-word pow from glibc_flt64.cuh, CORE-MATH's
# correctly rounded cbrt from cr_cbrt_flt64.cuh) and compiled straight
# through NVRTC with --fmad=false --ftz=false (the terrain-drag route).  The
# kernel text is pinned by SHA-256 in tests/test_windfarm_power_diag.py and
# each header by its own pin (tests/test_kernel_source_freeze_per_module.py,
# tests/test_cr_cbrt.py).
# ---------------------------------------------------------------------------

#: Slots of the packed float64 output buffer, each ``n_turbine`` long:
#: the evaluation, then the window's time-weighted sums and its maxima of
#: :data:`WINDOW_QUANTITIES`.  Two farm values (sum, max) follow.
NOW_SLOTS = ("rews", "rho", "v_eq", "power", "ct")
#: The per-turbine quantities the window folds (mean and max of each).
WINDOW_QUANTITIES = ("power", "ct", "rews")
_N_SLOTS = len(NOW_SLOTS) + 2 * len(WINDOW_QUANTITIES)

_THREADS = 128

KERNEL_SOURCE = r"""
extern "C" {

__device__ __forceinline__ float wf_zw(const float* ph, const float* phb,
                                       int phb3d, long long plane, long long c,
                                       int k, float grav)
{
    const float base = phb3d ? phb[(long long)k * plane + c] : phb[k];
    return (ph[(long long)k * plane + c] + base) / grav;
}

__device__ __forceinline__ float wf_z(const float* ph, const float* phb,
                                      int phb3d, long long plane, long long c,
                                      int k, float hgt, float grav)
{
    return 0.5f * (wf_zw(ph, phb, phb3d, plane, c, k, grav)
                   + wf_zw(ph, phb, phb3d, plane, c, k + 1, grav)) - hgt;
}

// One mass-level value of field f at the turbine's cell: 0 u (destaggered),
// 1 v (destaggered), 2 total theta, 3 full pressure, 4 vapour.
__device__ __forceinline__ float wf_field(
    int f, int k, const float* u, const float* v, const float* thp,
    const float* thb, int thb3d, const float* p, const float* qv,
    int ny, int nx, long long plane, long long c, int j, int i)
{
    const long long m = (long long)k * plane + c;
    if (f == 0) {
        const long long row = ((long long)k * ny + j) * (long long)(nx + 1);
        return 0.5f * (u[row + i] + u[row + i + 1]);
    }
    if (f == 1) {
        const long long lo = ((long long)k * (ny + 1) + j) * (long long)nx;
        return 0.5f * (v[lo + i] + v[lo + nx + i]);
    }
    if (f == 2) return thp[m] + (thb3d ? thb[m] : thb[k]);
    if (f == 3) return p[m];
    return qv[m];
}

// numpy.maximum / numpy.minimum: a NaN in either argument is the answer.
__device__ __forceinline__ double wf_max(double a, double b)
{
    return (a > b || isnan(a)) ? a : b;
}
__device__ __forceinline__ double wf_min(double a, double b)
{
    return (a < b || isnan(a)) ? a : b;
}

// numpy.interp(x, xp, fp, left=0.0) on points [a0, a1], as numpy's
// compiled arr_interp does it (binary search, slope * (x - xp[j]) + fp[j]).
__device__ __forceinline__ double wf_interp(double x, const double* xp,
                                            const double* fp, int a0, int a1)
{
    if (isnan(x)) return x;
    if (x < xp[a0]) return 0.0;
    if (x > xp[a1]) return fp[a1];
    if (x == xp[a1]) return fp[a1];
    int lo = a0, hi = a1;              // xp[lo] <= x < xp[hi]
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (xp[mid] <= x) lo = mid; else hi = mid;
    }
    if (xp[lo] == x) return fp[lo];
    const double slope = (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo]);
    return slope * (x - xp[lo]) + fp[lo];
}

__global__ void windfarm_turbines(
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ ph, const float* __restrict__ phb,
    const int phb3d, const float* __restrict__ ht,
    const float* __restrict__ thp, const float* __restrict__ thb,
    const int thb3d, const float* __restrict__ p,
    const float* __restrict__ qv, const int has_q,
    const int nz, const int ny, const int nx, const float grav,
    const int n, const int* __restrict__ sj, const int* __restrict__ si,
    const float* __restrict__ targets, const double* __restrict__ weights,
    const int* __restrict__ curve_index, const int* __restrict__ curve_offset,
    const double* __restrict__ curve_x, const double* __restrict__ curve_kw,
    const double* __restrict__ curve_rated_kw,
    const double* __restrict__ rated_w,
    const double* __restrict__ half_rho_area,
    const double* __restrict__ cut_out, const double eta_d,
    const double eta_r, double* out, float* hub, const double dt,
    const int mode)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int j = sj[t];
    const int i = si[t];
    const long long plane = (long long)ny * nx;
    const long long c = (long long)j * nx + i;
    const float hgt = ht[c];

    // Steps 2-3 for every target, the hub first (target row WF_NSEG), then
    // strips 0..WF_NSEG-1: the hub-wind bracketing, count(z <= h) - 1,
    // with its hold below and above.
    // One walk of the column counts every target's levels at once.
    float h_row[WF_NSEG + 1];
    int count[WF_NSEG + 1];
    #pragma unroll
    for (int r = 0; r <= WF_NSEG; ++r) {
        h_row[r] = targets[(long long)r * n + t];
        count[r] = 0;
    }
    for (int k = 0; k < nz; ++k) {
        const float zk = wf_z(ph, phb, phb3d, plane, c, k, hgt, grav);
        #pragma unroll
        for (int r = 0; r <= WF_NSEG; ++r)
            if (zk <= h_row[r]) ++count[r];
    }
    double cubed = 0.0;
    double uh = 0.0, vh = 0.0, hub_norm = 0.0;
    float th_h = 0.0f, p_h = 0.0f, q_h = 0.0f;
    for (int pass = 0; pass <= WF_NSEG; ++pass) {
        const int row = pass == 0 ? WF_NSEG : pass - 1;
        const float h = h_row[row];
        const int kb = count[row] - 1;
        const int lo = kb < 0 ? 0 : (kb > nz - 2 ? nz - 2 : kb);
        const float z_lo = wf_z(ph, phb, phb3d, plane, c, lo, hgt, grav);
        const float z_hi = wf_z(ph, phb, phb3d, plane, c, lo + 1, hgt, grav);
        const float w = (h - z_lo) / (z_hi - z_lo);
        const int nf = pass == 0 ? (has_q ? 5 : 4) : 2;
        float val[5];
        for (int f = 0; f < nf; ++f) {
            if (kb < 0) {
                val[f] = wf_field(f, 0, u, v, thp, thb, thb3d, p, qv, ny, nx,
                                  plane, c, j, i);
            } else if (kb >= nz - 1) {
                val[f] = wf_field(f, nz - 1, u, v, thp, thb, thb3d, p, qv, ny,
                                  nx, plane, c, j, i);
            } else {
                const float f_lo = wf_field(f, lo, u, v, thp, thb, thb3d, p,
                                            qv, ny, nx, plane, c, j, i);
                const float f_hi = wf_field(f, lo + 1, u, v, thp, thb, thb3d,
                                            p, qv, ny, nx, plane, c, j, i);
                val[f] = f_lo + w * (f_hi - f_lo);
            }
        }
        if (pass == 0) {
            for (int f = 0; f < 5; ++f)
                hub[(long long)f * n + t] = f < nf ? val[f] : 0.0f;
            uh = (double)val[0];
            vh = (double)val[1];
            hub_norm = sqrt(uh * uh + vh * vh);
            th_h = val[2];
            p_h = val[3];
            q_h = has_q ? val[4] : 0.0f;
            continue;
        }
        // Step 4: the strip's veer-weighted speed, cubed, area weighted,
        // summed strip 0 first.
        const double us = (double)val[0];
        const double vs = (double)val[1];
        const double speed = sqrt(us * us + vs * vs);
        const double norm = speed * hub_norm;
        double cosine = 1.0;
        if (norm > 0.0)
            cosine = wf_min(wf_max((us * uh + vs * vh) / norm, 0.0), 1.0);
        const double along = wf_max(speed * cosine, 0.0);
        cubed = cubed + weights[row] * (along * along * along);
    }
    const double rews = uw_cbrt(cubed);

    // Step 5: hub moist density (woof.energy.extract.air_density of
    // air_temperature) and the IEC density correction (ramps').
    const double theta = (double)th_h;
    const double pres = (double)p_h;
    const double q = (double)q_h;
    const double temperature = theta * glibc_pow(pres / WF_P0, WF_RCP);
    const double rho = pres / (WF_RD * (temperature
                                        * (1.0 + WF_VIRT * (q / (1.0 + q)))));
    const double v_eq = rews * uw_cbrt(rho / WF_RHO0);

    // Step 6: TurbineCurve.fraction and .thrust_coefficient, verbatim.
    const int ci = curve_index[t];
    const int a0 = curve_offset[ci];
    const int a1 = curve_offset[ci + 1] - 1;
    const double nan = __longlong_as_double(0x7ff8000000000000ULL);
    const int finite = isfinite(v_eq);
    double fraction = wf_interp(v_eq, curve_x, curve_kw, a0, a1)
                      / curve_rated_kw[ci];
    if (!finite) fraction = nan;
    const double v_pos = wf_max(v_eq, 0.0);
    double fraction_ct = wf_interp(v_pos, curve_x, curve_kw, a0, a1)
                         / curve_rated_kw[ci];
    if (!isfinite(v_pos)) fraction_ct = nan;
    const double power_w = fraction_ct * curve_rated_kw[ci] * 1000.0;
    const double floor_v = wf_max(v_eq, 1.0e-6);
    double cp = power_w / (half_rho_area[t] * glibc_pow(floor_v, 3.0)
                           * eta_d * eta_r);
    if (isnan(cp)) cp = 0.0;
    cp = cp < 0.0 ? 0.0 : (cp > WF_CP_CAP ? WF_CP_CAP : cp);
    double lo_a = 0.0, hi_a = WF_ONE_THIRD;
    for (int it = 0; it < WF_CT_STEPS; ++it) {
        const double mid = 0.5 * (lo_a + hi_a);
        const double one_minus = 1.0 - mid;
        if (4.0 * mid * (one_minus * one_minus) > cp) hi_a = mid;
        else lo_a = mid;
    }
    const double a = 0.5 * (lo_a + hi_a);
    const double ct_curve = cp > 0.0 ? 4.0 * a * (1.0 - a) : 0.0;
    const int running = finite && v_eq <= cut_out[t];
    const double power = finite ? (running ? fraction * rated_w[t] : 0.0)
                                : nan;
    const double ct = finite ? (running ? ct_curve : 0.0) : nan;

    out[0LL * n + t] = rews;
    out[1LL * n + t] = rho;
    out[2LL * n + t] = v_eq;
    out[3LL * n + t] = power;
    out[4LL * n + t] = ct;
    if (mode == 0) return;
    // The window over WINDOW_QUANTITIES = (power, ct, rews).
    const double q3[3] = {power, ct, rews};
    for (int s = 0; s < 3; ++s) {
        double* sum = out + (long long)(5 + s) * n + t;
        double* mx = out + (long long)(8 + s) * n + t;
        if (mode == 1) {
            *sum = q3[s] * dt;
            *mx = q3[s];
        } else {
            *sum = *sum + q3[s] * dt;
            *mx = fmax(*mx, q3[s]);
        }
    }
}

// The farm's summed power this step, ((0 + p0) + p1) + ... in turbine
// order -- one thread, so the order is the host's -- folded into (sum, max).
__global__ void windfarm_farm(const double* __restrict__ power, const int n,
                              double* farm, const double dt, const int mode)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double total = 0.0;
    for (int t = 0; t < n; ++t) total = total + power[t];
    if (mode == 1) {
        farm[0] = total * dt;
        farm[1] = total;
    } else {
        farm[0] = farm[0] + total * dt;
        farm[1] = fmax(farm[1], total);
    }
}

}
"""
for _name, _value in (
        ("WF_NSEG", str(N_ROTOR_SEGMENTS)),
        ("WF_CT_STEPS", str(CT_BISECTION_STEPS)),
        ("WF_ONE_THIRD", repr(1.0 / 3.0)),
        ("WF_P0", repr(float(P0))), ("WF_RCP", repr(RD / CP)),
        ("WF_RD", repr(float(RD))), ("WF_VIRT", repr(_VIRT)),
        ("WF_RHO0", repr(REFERENCE_DENSITY)),
        ("WF_CP_CAP", repr(16.0 / 27.0 * 0.999))):
    KERNEL_SOURCE = KERNEL_SOURCE.replace(_name, _value)
del _name, _value

#: No FMA contraction and no flush to zero (the terrain-drag options).
MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "woof.core.windfarm_power_diag:windfarm_power"
#: The device libm prepended to the kernel, read from the kernel directory:
#: glibc-word pow, then the correctly rounded cbrt.
LIBM_HEADERS = ("glibc_flt64.cuh", "cr_cbrt_flt64.cuh")

#: SHA-256 of the kernel text (after the LIBM_HEADERS), pinned by
#: tests/test_windfarm_power_diag.py.
KERNEL_SOURCE_SHA256 = hashlib.sha256(KERNEL_SOURCE.encode("utf-8")).hexdigest()


def module_source() -> str:
    """The exact source string NVRTC compiles: the libm, then the kernels."""
    root = Path(__file__).parent / "kernels"
    return "".join((root / name).read_text(encoding="utf-8")
                   for name in LIBM_HEADERS) + KERNEL_SOURCE


@lru_cache(maxsize=None)
def _module(device_id: int):
    """NVRTC directly with :data:`MODULE_OPTIONS`, one module per device.

    Not ``cupy.RawModule``, which appends ``-ftz=true`` after the caller's
    options (woof/core/terrain_drag.py measured what that costs).
    """
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler
    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile

    source = module_source()
    with cp.cuda.Device(device_id), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(
            source, MODULE_OPTIONS, None, "windfarm_power.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS,
                  module=None)
    return module


@lru_cache(maxsize=None)
def _kernel(device_id: int, name: str):
    return _module(device_id).get_function(name)


# ---------------------------------------------------------------------------
# The diagnostic
# ---------------------------------------------------------------------------

def _host(array):
    """A host NumPy view of a device array, NumPy array or NumPy scalar."""
    return np.asarray(array.get() if hasattr(array, "get") else array)


_INPUT_SHAPES = ("u", "v", "php", "phb", "ht", "thp", "thb", "p")


class WindfarmPowerDiagnostic:
    """One domain's wind-farm power window (see the module docstring).

    Construct through :func:`attach_windfarm_power`.  ``located`` is the
    ``inside`` list of :func:`woof.core.hub_wind_diag.locate_sites`, whose
    order is the turbine axis of every frame field and of the sidecar.
    On a device state every evaluation is the kernel; on a host state it
    is :func:`evaluate_turbines`.
    """

    name = DIAGNOSTIC_NAME

    def __init__(self, request, located, *, ny: int, nx: int, xp=np,
                 device_id: int | None = None, outside=(), grid_id: int = 1,
                 library=None):
        from woof.energy.ramps import rotor_segments
        from woof.energy.turbine_library import (DRIVETRAIN_EFFICIENCY,
                                                 ROTOR_EFFICIENCY)

        self.request = request
        self.ny, self.nx = int(ny), int(nx)
        self.xp = xp
        self.device_id = device_id
        self.grid_id = int(grid_id)
        self.outside = tuple(outside)
        by_name = {t.name: t for t in request.turbines}
        self.rows = tuple(located)
        self.turbines = tuple(by_name[row[0]] for row in self.rows)
        self.n = len(self.turbines)
        if not self.n:
            raise WindfarmPowerError(
                f"d{self.grid_id:02d}: a wind-farm power window needs at "
                "least one turbine inside the domain")
        library = _library() if library is None else library
        distinct, keys, index = [], {}, []
        for turbine in self.turbines:
            key = (turbine.turbine_type, turbine.rated_kw,
                   turbine.rotor_diameter_m
                   if turbine.turbine_type == GENERIC_TURBINE_TYPE else None)
            if key not in keys:
                keys[key] = len(distinct)
                distinct.append(turbine_curve(turbine, library))
            index.append(keys[key])
        self.distinct_curves = tuple(distinct)
        self.curve_index = np.array(index, dtype=np.int32)
        self.curves = tuple(distinct[i] for i in index)
        centres, weights = rotor_segments(
            [t.hub_height_m for t in self.turbines],
            [t.rotor_diameter_m for t in self.turbines], N_ROTOR_SEGMENTS)
        self.weights = np.asarray(weights, dtype=np.float64)
        targets = np.concatenate(
            [centres.T, np.array([[t.hub_height_m for t in self.turbines]])])
        self.targets_m = np.ascontiguousarray(targets.astype(DTYPE))
        self.rated_w = np.array([c.rated_kw * 1000.0 for c in self.curves])
        self.cut_out_ms = np.array([c.cut_out_ms for c in self.curves])
        # TurbineCurve.thrust_coefficient's ``0.5 * REFERENCE_DENSITY *
        # area``, area = math.pi * D ** 2 / 4.0 of the CURVE, as it forms it.
        from woof.energy.turbine_library import REFERENCE_DENSITY as RHO0
        self.half_rho_area = np.array(
            [0.5 * RHO0 * (math.pi * c.rotor_diameter_m ** 2 / 4.0)
             for c in self.curves])
        self._eta = (float(DRIVETRAIN_EFFICIENCY), float(ROTOR_EFFICIENCY))
        self._sj = np.array([r[3] for r in self.rows], dtype=np.int64)
        self._si = np.array([r[4] for r in self.rows], dtype=np.int64)
        #: Host-road window state (the device keeps its own in ``_out``).
        zero = np.zeros(self.n, dtype=np.float64)
        self._sum = {q: zero.copy() for q in WINDOW_QUANTITIES}
        self._max = {q: zero.copy() for q in WINDOW_QUANTITIES}
        self._farm = [0.0, 0.0]
        if xp is not np:
            offsets = np.cumsum([0] + [len(c.speeds) for c in distinct])
            with self._device():
                self._dev = dict(
                    sj=xp.asarray(self._sj.astype(np.int32)),
                    si=xp.asarray(self._si.astype(np.int32)),
                    targets=xp.asarray(self.targets_m),
                    weights=xp.asarray(self.weights),
                    curve_index=xp.asarray(self.curve_index),
                    curve_offset=xp.asarray(offsets.astype(np.int32)),
                    curve_x=xp.asarray(np.concatenate(
                        [np.asarray(c.speeds, np.float64) for c in distinct])),
                    curve_kw=xp.asarray(np.concatenate(
                        [np.asarray(c.power_kw, np.float64)
                         for c in distinct])),
                    curve_rated_kw=xp.asarray(np.array(
                        [float(c.rated_kw) for c in distinct])),
                    rated_w=xp.asarray(self.rated_w),
                    half_rho_area=xp.asarray(self.half_rho_area),
                    cut_out=xp.asarray(self.cut_out_ms))
                #: ONE packed buffer: the evaluation, the window, the farm.
                self._out = xp.zeros(_N_SLOTS * self.n + 2, dtype=np.float64)
                self._hub = xp.zeros(5 * self.n, dtype=DTYPE)
                self._spare = xp.zeros(1, dtype=DTYPE)
        #: The frame's own evaluation (host float64), set by
        #: :meth:`history_fields` and read by :meth:`close_interval`.
        self._frame = None
        self.window_s = 0.0
        self.window_steps = 0

    # -- plumbing ----------------------------------------------------------

    def _device(self):
        if self.xp is np or self.device_id is None:
            return nullcontext()
        import cupy as cp
        return cp.cuda.Device(self.device_id)

    def _check(self, state) -> None:
        arrays = [(name, getattr(state, name)) for name in _INPUT_SHAPES]
        if getattr(state, "qv", None) is not None:
            arrays.append(("qv", state.qv))
        for label, array in arrays:
            if array.dtype != DTYPE or not array.flags.c_contiguous:
                raise WindfarmPowerError(
                    f"wind-farm power needs contiguous float32 {label}, got "
                    f"{array.dtype} (contiguous={array.flags.c_contiguous})")
        nz = int(state.u.shape[0])
        if state.u.shape != (nz, self.ny, self.nx + 1):
            raise WindfarmPowerError(
                f"U has shape {state.u.shape}; this diagnostic was built for "
                f"a {(self.ny, self.nx)} mass grid")

    def _launch(self, state, mode: int, dt: float) -> None:
        """The turbine kernel (and, folding, the farm kernel) on ``state``."""
        self._check(state)
        d = self._dev
        qv = getattr(state, "qv", None)
        nz = int(state.u.shape[0])
        device = int(state.u.device.id)
        blocks = (self.n + _THREADS - 1) // _THREADS
        _kernel(device, "windfarm_turbines")(
            (blocks,), (_THREADS,),
            (state.u, state.v, state.php, state.phb,
             np.int32(0 if state.phb.ndim == 1 else 1), state.ht,
             state.thp, state.thb, np.int32(0 if state.thb.ndim == 1 else 1),
             state.p, self._spare if qv is None else qv,
             np.int32(0 if qv is None else 1), np.int32(nz),
             np.int32(self.ny), np.int32(self.nx), _G, np.int32(self.n),
             d["sj"], d["si"], d["targets"], d["weights"], d["curve_index"],
             d["curve_offset"], d["curve_x"], d["curve_kw"],
             d["curve_rated_kw"], d["rated_w"], d["half_rho_area"],
             d["cut_out"], np.float64(self._eta[0]), np.float64(self._eta[1]),
             self._out, self._hub, np.float64(dt), np.int32(mode)))
        if mode:
            _kernel(device, "windfarm_farm")(
                (1,), (1,),
                (self._out[3 * self.n:4 * self.n], np.int32(self.n),
                 self._out[_N_SLOTS * self.n:], np.float64(dt),
                 np.int32(mode)))

    def _evaluate_host(self, state) -> dict:
        columns = turbine_columns(np, state, self._sj, self._si)
        for label, array in columns.items():
            if array.dtype != DTYPE:
                raise WindfarmPowerError(
                    f"wind-farm power needs float32 {label}, got "
                    f"{array.dtype}")
        return evaluate_turbines(
            columns, self.targets_m, self.weights,
            curves=self.distinct_curves, curve_index=self.curve_index,
            rated_w=self.rated_w, half_rho_area=self.half_rho_area,
            cut_out_ms=self.cut_out_ms)

    def evaluate(self, state) -> dict:
        """Steps 1-6 at every turbine, on ``state`` (host arrays out)."""
        if self.xp is np:
            return self._evaluate_host(state)
        with self._device():
            self._launch(state, 0, 0.0)
            packed = self._out.get()
            hub = self._hub.get()
        out = {name: packed[k * self.n:(k + 1) * self.n].copy()
               for k, name in enumerate(NOW_SLOTS)}
        names = ["u", "v", "theta", "p"] + (
            ["qv"] if getattr(state, "qv", None) is not None else [])
        out.update({f"{name}_hub": hub[k * self.n:(k + 1) * self.n].copy()
                    for k, name in enumerate(names)})
        return out

    # -- the registry protocol ----------------------------------------------

    def on_step(self, state, dt: float) -> None:
        """Fold one completed step of length ``dt`` into the window."""
        dt = float(dt)
        start = self.window_steps == 0
        if self.xp is not np:
            with self._device():
                self._launch(state, 1 if start else 2, dt)
        else:
            now = self._evaluate_host(state)
            farm = sequential_sum(now["power"])
            for q in WINDOW_QUANTITIES:
                if start:
                    self._sum[q] = now[q] * dt
                    self._max[q] = now[q].copy()
                else:
                    self._sum[q] = self._sum[q] + now[q] * dt
                    self._max[q] = np.fmax(self._max[q], now[q])
            self._farm = ([farm * dt, farm] if start else
                          [self._farm[0] + farm * dt,
                           float(np.fmax(self._farm[1], farm))])
        self.window_s += dt
        self.window_steps += 1

    def _window_host(self) -> tuple[dict, dict, list]:
        """The window's sums, maxima and farm pair, on the host."""
        if self.xp is np:
            return self._sum, self._max, self._farm
        with self._device():
            packed = self._out.get()
        n = self.n
        sums = {q: packed[(5 + k) * n:(6 + k) * n].copy()
                for k, q in enumerate(WINDOW_QUANTITIES)}
        maxima = {q: packed[(8 + k) * n:(9 + k) * n].copy()
                  for k, q in enumerate(WINDOW_QUANTITIES)}
        return sums, maxima, [float(packed[-2]), float(packed[-1])]

    def take_window(self) -> dict:
        """The window folded since it last restarted, on the host, and
        restart it: ``{"steps", "seconds"}`` plus, after a step, the
        ``"sums"``/``"max"`` per quantity and the ``"farm"`` pair.  A caller
        that closes the window itself (hex's per-frame sidecar,
        :mod:`woof.hex.windfarm`) reads it here rather than the internals."""
        if not self.window_steps:
            out = {"steps": 0, "seconds": 0.0}
        else:
            sums, maxima, farm = self._window_host()
            out = {"steps": int(self.window_steps),
                   "seconds": float(self.window_s),
                   "sums": sums, "max": maxima, "farm": list(farm)}
        self.window_s = 0.0
        self.window_steps = 0
        return out

    @staticmethod
    def window_statistics(now, window) -> dict:
        """The sidecar's window statistics from :meth:`take_window`'s
        record, or the frame's own instant when no step is behind it."""
        if window["steps"]:
            seconds = window["seconds"]
            return {"mean": {q: window["sums"][q] / seconds
                             for q in WINDOW_QUANTITIES},
                    "max": window["max"],
                    "farm_mean": window["farm"][0] / seconds,
                    "farm_max": window["farm"][1]}
        farm = sequential_sum(now["power"])
        return {"mean": {q: now[q] for q in WINDOW_QUANTITIES},
                "max": {q: now[q] for q in WINDOW_QUANTITIES},
                "farm_mean": farm, "farm_max": farm}

    def history_fields(self, state) -> dict:
        """The frame's turbine fields: evaluated NOW, plus the farm mean."""
        now = self.evaluate(state)
        farm_now = sequential_sum(now["power"])
        if self.window_steps:
            sums, maxima, farm = self._window_host()
            window = {"mean": {q: sums[q] / self.window_s
                               for q in WINDOW_QUANTITIES},
                      "max": maxima,
                      "farm_mean": farm[0] / self.window_s,
                      "farm_max": farm[1]}
        else:
            # No step behind this frame (the cold start, the first frame
            # after a resume): the window is the frame's own instant.
            window = {"mean": {q: now[q] for q in WINDOW_QUANTITIES},
                      "max": {q: now[q] for q in WINDOW_QUANTITIES},
                      "farm_mean": farm_now, "farm_max": farm_now}
        self._frame = (now, window, self.window_s)
        fields = {
            "WINDFARM_WINDOW_S": np.asarray(self.window_s, dtype=DTYPE),
            "REWS_TURBINE": now["rews"].astype(DTYPE),
            "POWER_TURBINE": now["power"].astype(DTYPE),
            "CT_TURBINE": now["ct"].astype(DTYPE),
            "POWER_FARM": np.asarray(farm_now, dtype=DTYPE),
            "POWER_FARM_MEAN": np.asarray(window["farm_mean"], dtype=DTYPE),
        }
        assert tuple(fields) == WINDFARM_POWER_OUTPUT_FIELDS
        return fields

    def close_interval(self, state, *, valid_time, output_dir,
                       grid_id: int) -> Path | None:
        """Flush this window's per-turbine sidecar and restart the window."""
        if self._frame is None:
            # A publisher that snapshotted no frame through history_fields
            # (it would have refused first) still gets the instant it closes.
            self.history_fields(state)
        now, window, window_s = self._frame
        path = write_windfarm_sidecar(
            self, now=now, window=window, window_s=window_s,
            valid_time=valid_time, output_dir=output_dir,
            grid_id=int(grid_id))
        self._frame = None
        self.window_s = 0.0
        self.window_steps = 0
        return path

    def refuse_publisher(self, reason: str) -> None:
        from woof.explain import layered
        raise WindfarmPowerUnsupportedRoad(layered(
            f"[output] windfarm_turbines is configured on d{self.grid_id:02d}, "
            f"but {reason}.  Remove windfarm_turbines from this domain's "
            "[output], or run it on the resident road.",
            "The wind-farm power diagnostic folds every step in the dycore "
            "epilogue and restarts at every history write; this road would "
            "publish frames without it."))


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------

def sidecar_filename(valid_time: datetime, grid_id: int) -> str:
    """``windfarm_d0N_YYYY-MM-DD_HH:MM:SS.nc``, the wrfout time spelling."""
    return f"windfarm_d{int(grid_id):02d}_{valid_time:%Y-%m-%d_%H:%M:%S}.nc"


_SIDECAR_SERIES = (
    # name, quantity, statistic, units, long name
    ("REWS", "rews", None, "m s-1", "rotor-equivalent wind speed at the frame"),
    ("REWS_MEAN", "rews", "mean", "m s-1",
     "time-mean rotor-equivalent wind speed over the window"),
    ("REWS_MAX", "rews", "max", "m s-1",
     "max rotor-equivalent wind speed over the window"),
    ("RHO_HUB", "rho", None, "kg m-3", "moist air density at hub height"),
    ("POWER", "power", None, "W", "electrical power at the frame"),
    ("POWER_MEAN", "power", "mean", "W",
     "time-mean electrical power over the window"),
    ("POWER_MAX", "power", "max", "W", "max electrical power over the window"),
    ("CT", "ct", None, "1", "thrust coefficient at the frame"),
    ("CT_MEAN", "ct", "mean", "1", "time-mean thrust coefficient over the window"),
    ("CT_MAX", "ct", "max", "1", "max thrust coefficient over the window"),
)


def write_windfarm_sidecar(diagnostic: WindfarmPowerDiagnostic, *, now,
                           window, window_s: float, valid_time: datetime,
                           output_dir, grid_id: int) -> Path:
    """One window's per-turbine statistics, published atomically."""
    import netCDF4

    from woof.io.netcdf_serialization import NETCDF4_IO_LOCK

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / sidecar_filename(valid_time, grid_id)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    rows, turbines = diagnostic.rows, diagnostic.turbines
    with NETCDF4_IO_LOCK:
        with netCDF4.Dataset(temp, "w", format="NETCDF4") as ds:
            ds.createDimension("turbine", diagnostic.n)
            for name, values in (
                    ("turbine_name", [t.name for t in turbines]),
                    ("turbine_type", [t.turbine_type for t in turbines]),
                    ("curve_id", [c.curve_id for c in diagnostic.curves])):
                var = ds.createVariable(name, str, ("turbine",))
                for index, value in enumerate(values):
                    var[index] = value
            for name, values, dtype, units, long_name in (
                    ("turbine_lat", [t.lat for t in turbines], "f8",
                     "degree_north", "listed turbine latitude"),
                    ("turbine_lon", [t.lon for t in turbines], "f8",
                     "degree_east", "listed turbine longitude"),
                    ("grid_j", [r[3] for r in rows], "i4", "1",
                     "0-based south_north index of the sampled mass cell"),
                    ("grid_i", [r[4] for r in rows], "i4", "1",
                     "0-based west_east index of the sampled mass cell"),
                    ("grid_lat", [r[5] for r in rows], "f8", "degree_north",
                     "sampled mass cell latitude"),
                    ("grid_lon", [r[6] for r in rows], "f8", "degree_east",
                     "sampled mass cell longitude"),
                    ("turbine_distance_m", [r[7] for r in rows], "f8", "m",
                     "great-circle distance turbine to cell centre"),
                    ("hub_height_m", [t.hub_height_m for t in turbines], "f8",
                     "m", "hub height above ground"),
                    ("rotor_diameter_m", [t.rotor_diameter_m for t in turbines],
                     "f8", "m", "rotor diameter"),
                    ("rated_power", diagnostic.rated_w, "f8", "W",
                     "rated electrical power of the curve"),
                    ("cut_out_ms", diagnostic.cut_out_ms, "f8", "m s-1",
                     "cut-out wind speed of the curve")):
                var = ds.createVariable(name, dtype, ("turbine",))
                var.units = units
                var.long_name = long_name
                var[:] = np.asarray(values)
            for name, quantity, statistic, units, long_name in _SIDECAR_SERIES:
                values = (now[quantity] if statistic is None
                          else window[statistic][quantity])
                var = ds.createVariable(name, "f8", ("turbine",))
                var.units = units
                var.long_name = long_name
                var[:] = np.asarray(values, dtype=np.float64)
            for name, value, long_name in (
                    ("POWER_FARM", sequential_sum(now["power"]),
                     "summed electrical power at the frame"),
                    ("POWER_FARM_MEAN", window["farm_mean"],
                     "time-mean summed electrical power over the window"),
                    ("POWER_FARM_MAX", window["farm_max"],
                     "max summed electrical power over the window")):
                var = ds.createVariable(name, "f8", ())
                var.units = "W"
                var.long_name = long_name
                var.assignValue(value)
            ds.setncatts({
                "title": "woof wind-farm power at listed turbines",
                "grid_id": np.int32(grid_id),
                "valid_time": f"{valid_time:%Y-%m-%d_%H:%M:%S}",
                "valid_time_epoch_s": _epoch_seconds(valid_time),
                "window_seconds": float(window_s),
                "turbines_source": str(diagnostic.request.source or ""),
                "turbines_outside_domain": ",".join(diagnostic.outside),
                "rotor_segments": np.int32(N_ROTOR_SEGMENTS),
                "reference_density": REFERENCE_DENSITY,
                "method": (
                    "nearest mass cell; IEC 61400-12-1 rotor-equivalent "
                    "wind over equal-height strips, linear in z-AGL between "
                    "mass levels from (PH+PHB)/g-HGT, veer-weighted by the "
                    "cosine to the hub wind; v_eq = REWS (rho_hub / "
                    "1.225)^(1/3); power and thrust coefficient from the "
                    "turbine library curve at v_eq, zero above cut-out; "
                    "time-weighted mean and max over every model step of "
                    "the window"),
            })
    os.replace(temp, path)
    return path


# ---------------------------------------------------------------------------
# Attachment and disk pricing
# ---------------------------------------------------------------------------

def attach_windfarm_power(state, selection, *, nx: int, ny: int, dx: float,
                          dy: float, grid_id: int, latlon=None,
                          streamed: bool = False):
    """Attach the window ``selection`` asks for to ``state`` (or nothing).

    ``selection`` is the domain's resolved
    :class:`woof.io.history_selection.HistorySelection`; ``latlon`` is a
    zero-argument callable returning the mass-point ``(XLAT, XLONG)``
    planes.  Refuses a streamed domain by name.  Returns the attached
    diagnostic, or ``None`` when the option is off or no listed turbine
    lies inside this domain.
    """
    from woof.core import interval_diagnostics
    from woof.explain import layered

    request = getattr(selection, "windfarm_power", None)
    if request is None:
        return None
    if streamed or getattr(state, "_streamed_domain", None) is not None:
        raise WindfarmPowerUnsupportedRoad(layered(
            f"[output] windfarm_turbines is configured on d{int(grid_id):02d}, "
            "which runs [tiles]-streamed.  Run the domain resident (its own "
            "tiles = { mode = \"off\" }) or remove windfarm_turbines.",
            "The step is a tile sweep that never reaches the dycore "
            "epilogue the window folds in, and the frame is built from the "
            "host store."))
    existing = interval_diagnostics.get(state, DIAGNOSTIC_NAME)
    if existing is not None:
        if existing.request == request:
            return existing
        raise WindfarmPowerError(
            f"d{int(grid_id):02d} already carries a wind-farm power window "
            "built from a different [output] request")
    if latlon is None:
        raise WindfarmPowerError(layered(
            f"d{int(grid_id):02d}: [output] windfarm_turbines needs the "
            "domain's mass-point latitude/longitude to place the turbines, "
            "and this road supplied none.",
            "Each turbine samples its nearest mass cell; without the "
            "georeference no cell can be chosen."))
    lat, lon = (_host(plane) for plane in latlon())
    located = locate_sites([(t.name, t.lat, t.lon) for t in request.turbines],
                           lat, lon, dx=dx, dy=dy)
    if not located["inside"]:
        from woof.explain import warn
        warn(f"d{int(grid_id):02d}: none of the {len(request.turbines)} "
             "[output] windfarm_turbines lies inside this domain, so it "
             "writes no turbine fields and no windfarm sidecar",
             why="a turbine is diagnosed by every domain that contains it "
                 "and by no other")
        return None
    xp = _array_module(state)
    device_id = None if xp is np else int(state.u.device.id)
    diagnostic = WindfarmPowerDiagnostic(
        request, located["inside"], ny=ny, nx=nx, xp=xp, device_id=device_id,
        outside=located["outside"], grid_id=grid_id)
    interval_diagnostics.attach(state, diagnostic)
    return diagnostic


def windfarm_history_shapes(selection) -> dict[str, tuple]:
    """On-disk shapes the diagnostic adds to a frame (disk pricing).

    An upper bound: every LISTED turbine is counted, because which of
    them a domain contains is known only once its grid is placed.
    """
    request = getattr(selection, "windfarm_power", None)
    if request is None:
        return {}
    n = len(request.turbines)
    return {name: ((n,) if name.endswith("_TURBINE") else ())
            for name in WINDFARM_POWER_OUTPUT_FIELDS}


__all__ = [
    "DIAGNOSTIC_NAME",
    "GENERIC_TURBINE_TYPE",
    "N_ROTOR_SEGMENTS",
    "TURBINE_FILE_COLUMNS",
    "Turbine",
    "TurbineFileError",
    "WindfarmPowerDiagnostic",
    "WindfarmPowerError",
    "WindfarmPowerUnsupportedRoad",
    "attach_windfarm_power",
    "curve_fraction",
    "evaluate_turbines",
    "interp_columns",
    "read_turbine_file",
    "sidecar_filename",
    "turbine_columns",
    "thrust_coefficient",
    "turbine_curve",
    "windfarm_history_shapes",
    "write_windfarm_sidecar",
]
