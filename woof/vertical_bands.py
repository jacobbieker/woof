"""Height-band-biased eta ladders: put the levels where the targets are.

WRF's ``compute_eta`` (:mod:`woof.ingest.eta`) knows two stretch factors
and a deepest layer; it cannot be told "at least 100 levels between 15 and
25 km".  A column built around a stratospheric layer of interest needs
exactly that, so this module builds ``nz + 1`` full eta levels from:

* ``dzbot``: the first layer's depth (m), pinned;
* ``max_stretch``: the largest adjacent layer-depth ratio, in either
  direction (default 1.08), held at every interface by construction;
* the model top, as ``p_top_pa`` or ``ztop_m``;
* explicit bands (:class:`HeightBand`, ``LO:HI:N[:msl|agl]``) that must
  each hold at least ``N`` mass levels;
* a target level density: a uniform baseline, coarsening by
  ``damping_coarsening`` across the top ``damping_depth_m`` (the
  ``damp_opt = 3`` sponge, so the levels spent there are few), plus
  ``bias_weight`` times a smoothed histogram of target heights (for
  example the height column of a 3-D (time, lat, lon, height) track).

Heights: every height here is METRES ABOVE MEAN SEA LEVEL unless a band
says ``agl`` (then it is metres above ``surface_elevation_m``, the
reference column's ground).  GPS altitudes are MSL, which is why
MSL is the default.  The model's heights are geopotential metres; at
25 km a geometric height is ~100 m above its geopotential one, which is
inside one layer of the columns this exists for and is not corrected.

Heights <-> eta.  A height becomes a dry hydrostatic pressure in a
reference atmosphere, and the pressure becomes eta by inverting WRF v4's
hybrid full-level dry pressure ``pf = B(eta)*(ps - pt) +
(eta - B(eta))*(P0 - pt) + pt`` (:func:`woof.core.grid.compute_hybrid_coeffs`)
for the reference column's surface pressure ``ps``.  Above ``etac``
``B = 0``, so every level there is ISOBARIC: where a stratospheric level
sits in metres is decided by the real atmosphere's temperature, not by
the terrain or the coordinate.  Two reference atmospheres are offered:

* ``standard`` (default): the U.S. Standard Atmosphere 1976, to
  84.852 km.  It places a 15-25 km band where a mid-latitude
  stratosphere actually has those heights.
* ``wrf-base``: the model's own analytic dry base state
  (``module_initialize_real.F``: ``T = max(iso_temp, t00 + A*ln(p/P0))``,
  ``A`` = 50 K, iso_temp = 200 K, exactly what
  :func:`woof.ingest.real._make_real_base_serial` builds), whose surface
  relation is :func:`woof.core.grid.analytic_base_pressure_field`.  Its
  stratosphere is a 200 K isotherm, ~15 K colder than the standard one,
  so its 25 km sits ~1.2 km LOWER in pressure-height than the real
  atmosphere's: a band laid out in it lands its levels ~10% too deep
  once the real stratosphere fills them.  It is here for columns that
  must be described in the model's own reference state.

This module deliberately does not route through
:mod:`woof.energy.vertical`, whose preset refuses model tops above 20 km.

How the ladder is solved.  Each band contributes a hard cap on layer
depth inside it (initially ``(hi - lo) / N``) and ``max_dz`` a global one;
the target density contributes a soft cap ``s / density(z)``.  The caps
are smoothed into an envelope that the stretch limit can actually follow
(a layer may grow by ``(r - 1)`` metres per metre of ascent and must
start shrinking ``(r - 1)/r`` metres per metre before a tighter cap), the
layers are laid greedily up the envelope from ``dzbot``, and the scale
``s`` is bisected until exactly ``nz`` layers reach the model top.  Each
band's cap is then searched, solve by solve, for the DEEPEST layers that
still meet its count: a band is a minimum, and every level it does not
need goes back to the density.  Every result is re-measured through :func:`dz_profile`, the
same eta -> height path a reader of the ladder uses, so the counts it
reports are the counts the ladder has.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from woof.core import constants as c
from woof.explain import layered

#: Reference atmospheres heights are converted through (module docstring).
REFERENCE_ATMOSPHERES = ("standard", "wrf-base")
DEFAULT_REFERENCE = "standard"

#: Generator defaults.  ``nz`` is the dynamical core's ceiling
#: (``woof.physics_vertical_contract.ACOUSTIC_VERTICAL_LEVEL_BOUNDS``):
#: a column built to put levels where they matter has no reason to stop
#: short of it.
DEFAULT_NZ = 256
DEFAULT_DZBOT_M = 10.0
DEFAULT_MAX_STRETCH = 1.08
#: The wizard's ``zdamp`` (``[shared].zdamp``, damp_opt 3): the top layer
#: the sponge absorbs in, where resolution buys nothing.
DEFAULT_DAMPING_DEPTH_M = 5000.0
#: Baseline layer depth at the model top relative to below the sponge.
DEFAULT_DAMPING_COARSENING = 3.0
DEFAULT_BIAS_WEIGHT = 1.0
#: Gaussian smoothing of the target-height histogram.
DEFAULT_BIAS_BANDWIDTH_M = 500.0

#: WRF's analytic base state (``module_initialize_real.F``; Registry
#: ``base_lapse`` 50 K, ``iso_temp`` 200 K, ``base_temp`` 290 K).
WRF_BASE_LAPSE_K = 50.0
WRF_ISO_TEMP_K = 200.0
WRF_BASE_TEMP_K = 290.0

#: U.S. Standard Atmosphere 1976 below 84.852 km (geopotential): layer
#: base heights (m) and lapse rates (K/m), with its own constants.
_US76_BASES_M = (0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0)
_US76_LAPSE = (-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002)
_US76_TOP_M = 84852.0
_US76_G0 = 9.80665
_US76_R = 287.05287
_US76_T0 = 288.15
_US76_P0 = 101325.0

#: Height-grid spacing (m) the caps and envelope are sampled on.
_GRID_M = 2.0


class BandColumnRefused(ValueError):
    """The requested band-biased column cannot be built as asked."""


class _Unreachable(Exception):
    """``nz`` layers under the current caps fall short of the model top."""


# --------------------------------------------------------------------------
# Inputs


@dataclass(frozen=True)
class HeightBand:
    """At least ``min_levels`` mass levels between ``lo_m`` and ``hi_m``.

    ``datum`` is ``"msl"`` (metres above mean sea level, the default and
    what GPS altitudes are) or ``"agl"`` (metres above the
    reference column's surface).
    """

    lo_m: float
    hi_m: float
    min_levels: int
    datum: str = "msl"

    def msl(self, surface_elevation_m: float) -> tuple[float, float]:
        """``(lo, hi)`` in metres above MSL."""

        offset = float(surface_elevation_m) if self.datum == "agl" else 0.0
        return self.lo_m + offset, self.hi_m + offset

    def label(self) -> str:
        return f"{self.lo_m:g}-{self.hi_m:g} m {self.datum.upper()}"


def parse_band(text: str) -> HeightBand:
    """``--eta-band LO:HI:N[:msl|agl]`` -> :class:`HeightBand`.

    argparse type: raises :class:`argparse.ArgumentTypeError` with the
    expected spelling on anything else.
    """

    spelling = ("LO:HI:N[:msl|agl] -- heights in metres (MSL unless "
                "':agl'), N the minimum mass levels between them, e.g. "
                "15000:25000:100 or 0:2000:41:agl")
    parts = [part.strip() for part in str(text).split(":")]
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(f"{text!r} is not {spelling}")
    datum = parts[3].lower() if len(parts) == 4 else "msl"
    if datum not in ("msl", "agl"):
        raise argparse.ArgumentTypeError(
            f"{text!r}: datum {parts[3]!r} must be 'msl' or 'agl'; "
            f"expected {spelling}")
    try:
        lo, hi = float(parts[0]), float(parts[1])
        count = int(parts[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not {spelling} ({error})") from error
    if not (math.isfinite(lo) and math.isfinite(hi)) or not hi > lo \
            or lo < (0.0 if datum == "agl" else -500.0):
        raise argparse.ArgumentTypeError(
            f"{text!r}: need finite heights with LO < HI (and LO >= 0 for "
            f"agl); expected {spelling}")
    if count < 1:
        raise argparse.ArgumentTypeError(
            f"{text!r}: N must be a whole number >= 1")
    return HeightBand(lo, hi, count, datum)


def parse_bias_heights(value: str) -> tuple[float, ...]:
    """``--eta-bias-heights LIST|FILE``: heights in metres MSL.

    A comma list, or a file.  A file whose rows hold ONE number each (or
    a flat run of numbers) is read by the ``--eta-levels`` reader.  A file
    whose rows hold several fields is a TRACK -- for example
    ``time,lat,lon,height`` rows of a 3-D path, CSV or whitespace -- and
    its LAST field is the height; a first row whose last field is not a
    number is a header and is skipped.  ``#`` starts a comment either way.
    """

    import re

    from woof.cli_numbers import eta_levels_list

    path = Path(str(value)).expanduser()
    try:
        is_file = "," not in str(value) and path.is_file()
    except OSError:
        is_file = False
    if not is_file:
        return eta_levels_list(value)
    rows = [re.split(r"[\s,]+", line.split("#", 1)[0].strip())
            for line in path.read_text("utf-8").splitlines()]
    rows = [row for row in rows if row and row != [""]]
    def numeric(field: str) -> bool:
        try:
            float(field)
        except ValueError:
            return False
        return True

    # A track announces itself by a header row, or by every row carrying
    # the same three-plus fields (lat, lon, height at least); anything
    # else is a flat run of heights.
    if not rows:
        return eta_levels_list(value)
    header = len(rows[0]) > 1 and not numeric(rows[0][-1])
    widths = {len(row) for row in (rows[1:] if header else rows)}
    if not (header or (len(widths) == 1 and min(widths) >= 3)):
        return eta_levels_list(value)
    heights = []
    for index, row in enumerate(rows):
        try:
            height = float(row[-1])
        except ValueError:
            if index == 0:
                continue
            raise argparse.ArgumentTypeError(
                f"track file {str(value)!r} row {index + 1}: last field "
                f"{row[-1]!r} is not a height in metres") from None
        if not math.isfinite(height):
            raise argparse.ArgumentTypeError(
                f"track file {str(value)!r} row {index + 1}: height "
                f"{row[-1]!r} is not finite")
        heights.append(height)
    if not heights:
        raise argparse.ArgumentTypeError(
            f"track file {str(value)!r} holds no heights")
    return tuple(heights)


# --------------------------------------------------------------------------
# Reference atmospheres: height (m MSL) <-> dry pressure (Pa)


def _us76_layer_table():
    temps = [_US76_T0]
    pressures = [_US76_P0]
    for k in range(1, len(_US76_BASES_M)):
        dz = _US76_BASES_M[k] - _US76_BASES_M[k - 1]
        t0, lapse, p0 = temps[-1], _US76_LAPSE[k - 1], pressures[-1]
        t1 = t0 + lapse * dz
        if lapse == 0.0:
            p1 = p0 * math.exp(-_US76_G0 * dz / (_US76_R * t0))
        else:
            p1 = p0 * (t1 / t0) ** (-_US76_G0 / (_US76_R * lapse))
        temps.append(t1)
        pressures.append(p1)
    return np.array(temps), np.array(pressures)


_US76_T, _US76_P = _us76_layer_table()


def _standard_pressure(z):
    z = np.asarray(z, dtype=np.float64)
    k = np.clip(np.searchsorted(_US76_BASES_M, z, side="right") - 1,
                0, len(_US76_BASES_M) - 1)
    base = np.asarray(_US76_BASES_M)[k]
    lapse = np.asarray(_US76_LAPSE)[k]
    t0, p0 = _US76_T[k], _US76_P[k]
    dz = z - base
    t = t0 + lapse * dz
    with np.errstate(divide="ignore", invalid="ignore"):
        power = np.where(lapse == 0.0, 0.0,
                         -_US76_G0 / (_US76_R * np.where(lapse == 0.0, 1.0,
                                                          lapse)))
        graded = p0 * np.power(t / t0, power)
    iso = p0 * np.exp(-_US76_G0 * dz / (_US76_R * t0))
    return np.where(lapse == 0.0, iso, graded)


def _standard_height(p):
    p = np.asarray(p, dtype=np.float64)
    # Layer bases have DESCENDING pressure; index by the negated array.
    k = np.clip(np.searchsorted(-_US76_P, -p, side="right") - 1,
                0, len(_US76_BASES_M) - 1)
    base = np.asarray(_US76_BASES_M)[k]
    lapse = np.asarray(_US76_LAPSE)[k]
    t0, p0 = _US76_T[k], _US76_P[k]
    with np.errstate(divide="ignore", invalid="ignore"):
        iso = base + _US76_R * t0 / _US76_G0 * np.log(p0 / p)
        safe = np.where(lapse == 0.0, 1.0, lapse)
        graded = base + t0 / safe * (
            np.power(p / p0, -_US76_R * safe / _US76_G0) - 1.0)
    return np.where(lapse == 0.0, iso, graded)


def _wrf_iso_point(base_temp: float) -> tuple[float, float]:
    """``(p, z)`` where WRF's base-state temperature meets ``iso_temp``."""

    x_iso = (WRF_ISO_TEMP_K - float(base_temp)) / WRF_BASE_LAPSE_K
    p_iso = c.P0 * math.exp(x_iso)
    z_iso = -(c.RD / c.G) * (float(base_temp) * x_iso
                             + 0.5 * WRF_BASE_LAPSE_K * x_iso * x_iso)
    return p_iso, z_iso


def _wrf_base_pressure(z, base_temp: float):
    from woof.core.grid import analytic_base_pressure_field

    z = np.asarray(z, dtype=np.float64)
    p_iso, z_iso = _wrf_iso_point(base_temp)
    below = analytic_base_pressure_field(np.minimum(z, z_iso), base_temp)
    above = p_iso * np.exp(-(z - z_iso) * c.G / (c.RD * WRF_ISO_TEMP_K))
    return np.where(z <= z_iso, below, above)


def _wrf_base_height(p, base_temp: float):
    p = np.asarray(p, dtype=np.float64)
    p_iso, z_iso = _wrf_iso_point(base_temp)
    ratio = float(base_temp) / WRF_BASE_LAPSE_K
    # woof.core.grid.analytic_base_terrain_height, elementwise.
    inner = np.log(np.maximum(p, p_iso) / c.P0) + ratio
    graded = ((ratio ** 2 - inner ** 2) * WRF_BASE_LAPSE_K * c.RD
              / (2.0 * c.G))
    iso = z_iso + c.RD * WRF_ISO_TEMP_K / c.G * np.log(
        p_iso / np.minimum(p, p_iso))
    return np.where(p >= p_iso, graded, iso)


def reference_pressure(z, *, reference: str = DEFAULT_REFERENCE,
                       base_temp: float = WRF_BASE_TEMP_K):
    """Dry pressure (Pa) at height ``z`` (m MSL) in ``reference``."""

    _check_reference(reference, base_temp)
    if reference == "standard":
        if np.any(np.asarray(z) > _US76_TOP_M):
            raise BandColumnRefused(layered(
                f"height above {_US76_TOP_M:g} m is outside the standard "
                "atmosphere; lower the model top",
                "The U.S. Standard Atmosphere 1976 tables used here stop "
                "at 84.852 km geopotential."))
        return _standard_pressure(z)
    return _wrf_base_pressure(z, base_temp)


def reference_height(p, *, reference: str = DEFAULT_REFERENCE,
                     base_temp: float = WRF_BASE_TEMP_K):
    """Height (m MSL) of dry pressure ``p`` (Pa) in ``reference``."""

    _check_reference(reference, base_temp)
    if reference == "standard":
        if np.any(np.asarray(p) < _standard_pressure(_US76_TOP_M)):
            raise BandColumnRefused(layered(
                "pressure above the standard atmosphere's 84.852 km top; "
                "raise the model top pressure",
                "The U.S. Standard Atmosphere 1976 tables used here stop "
                "at 84.852 km geopotential."))
        return _standard_height(p)
    return _wrf_base_height(p, base_temp)


def _check_reference(reference: str, base_temp: float) -> None:
    if reference not in REFERENCE_ATMOSPHERES:
        raise BandColumnRefused(
            f"reference atmosphere {reference!r} is not one of "
            f"{', '.join(REFERENCE_ATMOSPHERES)}")
    if reference == "wrf-base" and not float(base_temp) > WRF_ISO_TEMP_K:
        raise BandColumnRefused(
            f"base_temp {base_temp!r} K must exceed WRF's iso_temp "
            f"{WRF_ISO_TEMP_K:g} K")


# --------------------------------------------------------------------------
# Pressure <-> eta through WRF v4's hybrid coordinate


def _hybrid_b(eta, hybrid_opt: int, etac: float):
    """``B(eta)`` (= ``c3f``) elementwise, as ``compute_hybrid_coeffs``."""

    eta = np.asarray(eta, dtype=np.float64)
    if hybrid_opt in (0, 1):
        return eta.copy()
    if hybrid_opt != 2:
        raise BandColumnRefused(
            f"hybrid_opt must be 0/1 (B=eta) or 2 (Klemp cubic), got "
            f"{hybrid_opt!r}")
    from woof.core.grid import hybrid_b_closed_form

    b1, b2, b3, b4, b5 = hybrid_b_closed_form(etac)
    eta2 = eta * eta
    b = (b1 + b2 * eta + b3 * eta2 + b4 * eta2 * eta) / b5
    return np.where(eta < etac, 0.0, b)


def eta_pressure(eta, *, surface_pressure_pa: float, p_top_pa: float,
                 hybrid_opt: int = 2, etac: float = 0.2):
    """Full-level dry pressure (Pa) of ``eta`` over a column with this
    dry surface pressure: ``B*(ps - pt) + (eta - B)*(P0 - pt) + pt``."""

    eta = np.asarray(eta, dtype=np.float64)
    b = _hybrid_b(eta, hybrid_opt, etac)
    pt = float(p_top_pa)
    return (b * (float(surface_pressure_pa) - pt)
            + (eta - b) * (c.P0 - pt) + pt)


def pressure_eta(p, *, surface_pressure_pa: float, p_top_pa: float,
                 hybrid_opt: int = 2, etac: float = 0.2):
    """Invert :func:`eta_pressure`: the eta whose pressure is ``p``.

    Bisection on ``[0, 1]`` (``eta_pressure`` is increasing in eta for any
    column whose ground lies in the base state's reach); 60 halvings take
    the bracket below float64 resolution.
    """

    p = np.asarray(p, dtype=np.float64)
    lo = np.zeros_like(p)
    hi = np.ones_like(p)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        above = eta_pressure(mid, surface_pressure_pa=surface_pressure_pa,
                             p_top_pa=p_top_pa, hybrid_opt=hybrid_opt,
                             etac=etac) > p
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Measuring a ladder


def dz_profile(eta, *, p_top_pa: float, surface_elevation_m: float = 0.0,
               reference: str = DEFAULT_REFERENCE,
               base_temp: float = WRF_BASE_TEMP_K,
               hybrid_opt: int = 2, etac: float = 0.2) -> dict:
    """Heights and layer depths of a full-level ``eta`` ladder.

    For the reference column with ground at ``surface_elevation_m`` (m
    MSL) in ``reference``.  Returns numpy arrays: ``z_full_m`` (``nz + 1``
    full-level heights, m MSL), ``z_mass_m`` (``nz`` mass-level heights at
    ``znu``, m MSL), ``dz_m`` (``nz`` layer depths), ``stretch`` (``nz - 1``
    adjacent ratios ``dz[k+1]/dz[k]``), plus scalars ``max_stretch`` (the
    largest ratio in either direction), ``surface_elevation_m``,
    ``surface_pressure_pa``, ``p_top_pa``, ``ztop_m``, ``reference``.
    """

    eta = np.asarray(eta, dtype=np.float64)
    z0 = float(surface_elevation_m)
    ps = float(reference_pressure(z0, reference=reference,
                                  base_temp=base_temp))
    common = dict(surface_pressure_pa=ps, p_top_pa=p_top_pa,
                  hybrid_opt=hybrid_opt, etac=etac)
    p_full = eta_pressure(eta, **common)
    z_full = reference_height(p_full, reference=reference,
                              base_temp=base_temp)
    z_full[0] = z0
    # The mass level lives at znu, whose pressure is c3h*(ps-pt)+c4h+pt:
    # the MEAN of the bracketing full-level B and eta.
    znu = 0.5 * (eta[:-1] + eta[1:])
    b_full = _hybrid_b(eta, hybrid_opt, etac)
    b_half = 0.5 * (b_full[:-1] + b_full[1:])
    pt = float(p_top_pa)
    p_mass = b_half * (ps - pt) + (znu - b_half) * (c.P0 - pt) + pt
    z_mass = reference_height(p_mass, reference=reference,
                              base_temp=base_temp)
    dz = np.diff(z_full)
    stretch = dz[1:] / dz[:-1]
    both = np.concatenate([stretch, 1.0 / stretch]) if stretch.size else \
        np.array([1.0])
    return {"z_full_m": z_full, "z_mass_m": z_mass, "dz_m": dz,
            "stretch": stretch, "max_stretch": float(both.max()),
            "surface_elevation_m": z0, "surface_pressure_pa": ps,
            "p_top_pa": pt, "ztop_m": float(z_full[-1]),
            "reference": reference}


def levels_in_band(profile: dict, band: HeightBand) -> int:
    """Mass levels of ``profile`` inside ``band`` (edges inclusive)."""

    lo, hi = band.msl(profile["surface_elevation_m"])
    z = profile["z_mass_m"]
    return int(np.count_nonzero((z >= lo) & (z <= hi)))


def band_dz_range(profile: dict, band: HeightBand) -> tuple[float, float]:
    """``(min, max)`` depth of the layers whose mass level is in ``band``."""

    lo, hi = band.msl(profile["surface_elevation_m"])
    z = profile["z_mass_m"]
    inside = (z >= lo) & (z <= hi)
    if not inside.any():
        return (math.nan, math.nan)
    dz = profile["dz_m"][inside]
    return float(dz.min()), float(dz.max())


def profile_lines(profile: dict, bands=(), *, slab_m: float = 2500.0,
                  max_stretch: float | None = None) -> list[str]:
    """The measured column in a few lines: top, first layer, stretch, each
    band's count and depths, and level counts per ``slab_m`` of height."""

    dz = profile["dz_m"]
    nz = dz.size
    deepest = int(np.argmax(dz))
    limit = (f" (limit {max_stretch:g})" if max_stretch is not None else "")
    lines = [
        f"band-biased eta: {nz} mass levels, ground "
        f"{profile['surface_elevation_m']:g} m MSL, top "
        f"{profile['ztop_m'] / 1000.0:.2f} km MSL at "
        f"{profile['p_top_pa']:g} Pa ({profile['reference']} atmosphere)",
        f"  first layer {dz[0]:.1f} m; max adjacent stretch "
        f"{profile['max_stretch']:.4f}{limit}; deepest layer "
        f"{dz[deepest]:.0f} m at "
        f"{profile['z_mass_m'][deepest] / 1000.0:.2f} km",
    ]
    for band in bands:
        low, high = band_dz_range(profile, band)
        lines.append(
            f"  band {band.label()}: {levels_in_band(profile, band)} levels "
            f"(min {band.min_levels}), dz {low:.0f}-{high:.0f} m")
    z_mass = profile["z_mass_m"]
    edges = np.arange(math.floor(profile["surface_elevation_m"] / slab_m)
                      * slab_m, profile["ztop_m"] + slab_m, slab_m)
    slabs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (z_mass >= lo) & (z_mass < hi)
        if inside.any():
            slabs.append(f"{lo / 1000:g}-{hi / 1000:g} km {int(inside.sum())} "
                         f"(dz {dz[inside].min():.0f}-{dz[inside].max():.0f})")
    for start in range(0, len(slabs), 3):
        lines.append("  " + "; ".join(slabs[start:start + 3]))
    return lines


def eta_file_text(eta, *, profile: dict | None = None, header=()) -> str:
    """``eta`` one level per line, readable by ``--eta-levels FILE``.

    ``header`` lines and, given ``profile``, a per-level table (full-level
    height, layer depth) are written as ``#`` comments, which the
    ``--eta-levels`` reader skips.
    """

    lines = [f"# {line}" for line in header]
    if profile is not None:
        lines.append(
            f"# heights: {profile['reference']} atmosphere over ground at "
            f"{profile['surface_elevation_m']:g} m MSL, p_top "
            f"{profile['p_top_pa']:g} Pa")
        lines.append("# k  eta  z_full_m_msl  dz_above_m")
        dz = profile["dz_m"]
        for k, (level, z) in enumerate(zip(eta, profile["z_full_m"])):
            above = f"{dz[k]:.1f}" if k < dz.size else "-"
            lines.append(f"# {k} {float(level)!r} {z:.1f} {above}")
    lines.extend(repr(float(level)) for level in eta)
    return "\n".join(lines) + "\n"


def write_eta_file(path, eta, *, profile: dict | None = None,
                   header=()) -> Path:
    """Write :func:`eta_file_text` to ``path``."""

    path = Path(path)
    path.write_text(eta_file_text(eta, profile=profile, header=header),
                    encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# The generator


@dataclass(frozen=True)
class BandColumn:
    """A generated ladder, the settings that built it, and its profile."""

    eta: np.ndarray
    p_top_pa: float
    profile: dict
    bands: tuple[HeightBand, ...]
    settings: dict = field(default_factory=dict)

    @property
    def nz(self) -> int:
        return int(self.eta.size - 1)

    def lines(self) -> list[str]:
        return profile_lines(self.profile, self.bands,
                             max_stretch=self.settings.get("max_stretch"))


def _float32_shortest(values) -> np.ndarray:
    """Each value as the shortest decimal of its float32 (the spelling the
    WRF generator path emits; exact round trip, ordering preserved)."""

    return np.array([float(str(np.float32(v))) for v in values])


def _bias_density(z, heights, bandwidth_m, column):
    """Smoothed histogram of ``heights`` on grid ``z``, mean 1 over
    ``column`` (a boolean mask of the grid inside the model column)."""

    heights = np.asarray(heights, dtype=np.float64)
    edges = np.concatenate([z - 0.5 * _GRID_M, [z[-1] + 0.5 * _GRID_M]])
    counts, _ = np.histogram(heights, bins=edges)
    sigma = float(bandwidth_m) / _GRID_M
    half = int(math.ceil(4.0 * sigma))
    taps = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    # "full" then the centred slice: mode="same" returns the LONGER of the
    # two inputs, which is the kernel's length on a shallow column.
    smooth = np.convolve(counts.astype(np.float64), taps / taps.sum(),
                         mode="full")[half:half + counts.size]
    mean = smooth[column].mean()
    return smooth / mean if mean > 0.0 else smooth


def _envelope(cap, z, r):
    """Smallest layer-depth ceiling under ``cap`` that a column stretching
    at most ``r`` per layer can follow: growth limited to ``(r - 1)`` m per
    metre after a tight cap, shrinking begun ``(r - 1)/r`` m per metre
    before one.  Both passes keep it continuous, so the solve is too."""

    up = r - 1.0
    down = (r - 1.0) / r
    forward = np.minimum.accumulate(cap - up * z) + up * z
    rev = (forward + down * z)[::-1]
    return np.minimum.accumulate(rev)[::-1] - down * z


def _lay(nz, dzbot, r, z, ceiling, z0):
    """Greedy layers up ``ceiling`` from ``dzbot``, stretch held to ``r``."""

    # ``z`` is the uniform _GRID_M grid from z0: interpolate by index
    # rather than np.interp, which costs more per call than the arithmetic.
    values = ceiling.tolist()
    last = len(values) - 1
    dz = [float(dzbot)]
    top = z0 + dz[0]
    for _ in range(1, nz):
        at = (top - z0) / _GRID_M
        i = min(max(int(at), 0), last - 1)
        frac = min(max(at - i, 0.0), 1.0)
        want = values[i] + frac * (values[i + 1] - values[i])
        prev = dz[-1]
        dz.append(min(max(want, prev / r), prev * r))
        top += dz[-1]
    return np.array(dz)


def band_biased_column(*, nz: int = DEFAULT_NZ,
                       dzbot: float = DEFAULT_DZBOT_M,
                       max_stretch: float = DEFAULT_MAX_STRETCH,
                       p_top_pa: float | None = None,
                       ztop_m: float | None = None,
                       bands=(),
                       bias_heights=(),
                       bias_weight: float = DEFAULT_BIAS_WEIGHT,
                       bias_bandwidth_m: float = DEFAULT_BIAS_BANDWIDTH_M,
                       max_dz: float | None = None,
                       surface_elevation_m: float = 0.0,
                       reference: str = DEFAULT_REFERENCE,
                       base_temp: float = WRF_BASE_TEMP_K,
                       hybrid_opt: int = 2, etac: float = 0.2,
                       damping_depth_m: float = DEFAULT_DAMPING_DEPTH_M,
                       damping_coarsening: float = DEFAULT_DAMPING_COARSENING,
                       ) -> BandColumn:
    """Build the band-biased ladder (module docstring) and measure it.

    Exactly one of ``p_top_pa`` / ``ztop_m`` (m MSL).  Refuses
    (:class:`BandColumnRefused`) inputs no ladder can honour: bands outside
    the column, more band levels than ``nz``, a column ``nz`` layers cannot
    span at this ``dzbot``/``max_stretch``/``max_dz``.
    """

    bands = tuple(bands)
    _check_reference(reference, base_temp)
    if isinstance(nz, bool) or int(nz) != nz or not 4 <= int(nz) <= 256:
        raise BandColumnRefused(
            f"nz must be a whole number in 4..256 (the vertical stencil "
            f"width and the dynamical core's ceiling); got {nz!r}")
    nz = int(nz)
    for name, value in (("dzbot", dzbot), ("bias_bandwidth_m",
                                           bias_bandwidth_m),
                        ("damping_coarsening", damping_coarsening)):
        if not (math.isfinite(float(value)) and float(value) > 0.0):
            raise BandColumnRefused(
                f"{name} must be a positive finite number; got {value!r}")
    if not (math.isfinite(float(max_stretch)) and float(max_stretch) > 1.0):
        raise BandColumnRefused(layered(
            f"max_stretch must be a finite ratio above 1; got "
            f"{max_stretch!r}",
            "It bounds dz[k+1]/dz[k] and its inverse; 1 would forbid the "
            "column from changing layer depth at all."))
    if max_dz is not None and not (math.isfinite(float(max_dz))
                                   and float(max_dz) > 0.0):
        raise BandColumnRefused(
            f"max_dz must be a positive finite number; got {max_dz!r}")
    if not (math.isfinite(float(bias_weight)) and float(bias_weight) >= 0.0):
        raise BandColumnRefused(
            f"bias_weight must be finite and >= 0; got {bias_weight!r}")
    if not (math.isfinite(float(damping_depth_m))
            and float(damping_depth_m) >= 0.0):
        raise BandColumnRefused(
            f"damping_depth_m must be finite and >= 0; got "
            f"{damping_depth_m!r}")
    if (p_top_pa is None) == (ztop_m is None):
        raise BandColumnRefused(
            "give exactly one model top: p_top_pa (Pa) or ztop_m (m MSL)")
    r = float(max_stretch)
    # Laid a hair inside the limit: the float32 spelling of eta moves a
    # 10 m layer by ~1e-4 of itself, which must not cross the limit.
    r_lay = 1.0 + (r - 1.0) * 0.995
    z0 = float(surface_elevation_m)
    ps = float(reference_pressure(z0, reference=reference,
                                  base_temp=base_temp))
    if p_top_pa is not None:
        p_top = float(p_top_pa)
        if not (math.isfinite(p_top) and 0.0 < p_top < min(ps, c.P0)):
            raise BandColumnRefused(
                f"p_top {p_top_pa!r} Pa must lie strictly between 0 and "
                f"the column's surface pressure {min(ps, c.P0):.0f} Pa")
        ztop = float(reference_height(p_top, reference=reference,
                                      base_temp=base_temp))
    else:
        ztop = float(ztop_m)
        if not (math.isfinite(ztop) and ztop > z0):
            raise BandColumnRefused(
                f"ztop {ztop_m!r} m must lie above the ground at {z0:g} m")
        p_top = float(reference_pressure(ztop, reference=reference,
                                         base_temp=base_temp))
    depth = ztop - z0

    # Bands, in MSL, refused if they leave the column or cannot fit.
    spans = []
    for band in bands:
        lo, hi = band.msl(z0)
        if lo < z0 - 1e-6 or hi > ztop + 1e-6:
            raise BandColumnRefused(layered(
                f"band {band.label()} reaches outside the column "
                f"{z0:.0f}-{ztop:.0f} m MSL; move it inside, or raise the "
                "model top (lower p_top)",
                "A band's heights are MSL unless it ends ':agl'; the "
                f"column's top is p_top {p_top:g} Pa in the {reference} "
                "atmosphere."))
        if lo < z0 + float(dzbot) and (hi - lo) / band.min_levels < dzbot:
            raise BandColumnRefused(layered(
                f"band {band.label()} needs layers of at most "
                f"{(hi - lo) / band.min_levels:g} m where the first layer is "
                f"pinned at dzbot {float(dzbot):g} m; lower --dzbot below "
                "that, or widen the band or lower its count",
                "The first layer is exactly dzbot deep and every layer above "
                "it can shrink only by the stretch limit."))
        spans.append((lo, hi))
    if sum(band.min_levels for band in bands) > nz:
        raise BandColumnRefused(
            f"the bands ask for {sum(b.min_levels for b in bands)} levels "
            f"but the column has nz = {nz}")
    heights = np.asarray(tuple(bias_heights), dtype=np.float64)
    if heights.size and not np.isfinite(heights).all():
        raise BandColumnRefused("every bias height must be finite")

    # Height grid (a margin above the top keeps the greedy laying defined
    # while the scale is still too large).
    z = np.arange(z0, ztop + 0.1 * depth + 1000.0, _GRID_M)
    column = z <= ztop
    density = np.ones_like(z)
    if damping_depth_m > 0.0:
        frac = np.clip((z - (ztop - damping_depth_m)) / damping_depth_m,
                       0.0, 1.0)
        density = 1.0 / (1.0 + (float(damping_coarsening) - 1.0) * frac)
    if heights.size and bias_weight > 0.0:
        inside = heights[(heights >= z0) & (heights <= ztop)]
        if inside.size == 0:
            raise BandColumnRefused(layered(
                f"none of the {heights.size} bias heights lies inside the "
                f"column {z0:.0f}-{ztop:.0f} m MSL",
                "Bias heights are metres above mean sea level (GPS track "
                "altitudes are); check their units and the model top."))
        density = density + float(bias_weight) * _bias_density(
            z, inside, bias_bandwidth_m, column)
    soft = 1.0 / density
    global_cap = math.inf if max_dz is None else float(max_dz)

    def solve(caps):
        hard = np.full_like(z, global_cap)
        for (lo, hi), cap in zip(spans, caps):
            hard[(z >= lo) & (z <= hi)] = np.minimum(
                hard[(z >= lo) & (z <= hi)], cap)

        def layers(scale):
            return _lay(nz, float(dzbot), r_lay, z,
                        _envelope(np.minimum(hard, scale * soft), z, r_lay),
                        z0)

        lo_s, hi_s = 1e-3, 1e7
        if layers(hi_s).sum() < depth:
            raise _Unreachable
        if layers(lo_s).sum() > depth:
            raise BandColumnRefused(
                f"dzbot {float(dzbot):g} m is too deep for {nz} layers in a "
                f"{depth:.0f} m column; lower dzbot or nz")
        for _ in range(100):
            mid = math.sqrt(lo_s * hi_s)
            if layers(mid).sum() < depth:
                lo_s = mid
            else:
                hi_s = mid
            if hi_s / lo_s < 1.0 + 1e-8:
                break
        dz = layers(hi_s)
        # The bisection leaves a residual of ~1e-8 of the column; a
        # uniform scale removes it without touching any ratio.
        return dz * (depth / dz.sum())

    def measure(caps):
        dz = solve(caps)
        z_full = z0 + np.concatenate([[0.0], np.cumsum(dz)])
        p_full = reference_pressure(np.minimum(z_full, ztop),
                                    reference=reference, base_temp=base_temp)
        eta = pressure_eta(p_full, surface_pressure_pa=ps, p_top_pa=p_top,
                           hybrid_opt=hybrid_opt, etac=etac)
        eta[0], eta[-1] = 1.0, 0.0
        eta = _float32_shortest(eta)
        if not np.all(np.diff(eta) < 0.0):
            raise BandColumnRefused(layered(
                "the generated ladder is not strictly decreasing in eta at "
                "float32 resolution; thicken dzbot or lower nz",
                "Layers thinner than float32 can tell apart near eta = 1 "
                "collapse onto one level."))
        profile = dz_profile(eta, p_top_pa=p_top, surface_elevation_m=z0,
                             reference=reference, base_temp=base_temp,
                             hybrid_opt=hybrid_opt, etac=etac)
        return eta, profile, [levels_in_band(profile, band) for band in bands]

    # Each band's cap is searched for the DEEPEST layers that still give
    # it its count: a band is a minimum, and every level it does not need
    # goes back to the target density.  ``ok``/``bad`` bracket the cap
    # (count met / count short); bands couple only through the shared
    # scale, so a bracket is re-opened if a neighbour's move breaks it.
    def unreachable():
        return BandColumnRefused(layered(
            f"{nz} layers from dzbot {float(dzbot):g} m stretching at most "
            f"{r:g} cannot reach the model top {ztop:.0f} m MSL while every "
            "band keeps its count; raise nz, dzbot, max_stretch or max_dz, "
            "lower the band counts, or lower the top",
            "Each band's layers are capped at the deepest depth that still "
            "gives it its count, max_dz caps every layer, and the stretch "
            "limit bounds how fast layers deepen between the caps; together "
            "they leave too few metres per level."))

    caps = [(hi - lo) / band.min_levels for band, (lo, hi)
            in zip(bands, spans)]
    ok: list = [None] * len(bands)
    bad: list = [None] * len(bands)
    best = None
    for _ in range(120):
        try:
            eta, profile, counts = measure(caps)
        except _Unreachable:
            # The caps are too tight to reach the top at all: every band
            # that has room to deepen its layers does, and only when none
            # has is the column refused.
            grown = False
            for i in range(len(bands)):
                if bad[i] is None:
                    caps[i] *= 1.25
                elif bad[i] / caps[i] >= 1.002:
                    caps[i] = math.sqrt(caps[i] * bad[i])
                else:
                    continue
                grown = True
            if grown:
                continue
            raise unreachable() from None
        met = all(have >= band.min_levels
                  for have, band in zip(counts, bands))
        if met:
            best = (eta, profile, list(caps))
        settled = met
        for i, (have, band) in enumerate(zip(counts, bands)):
            need = band.min_levels
            if have >= need:
                ok[i] = caps[i]
                if bad[i] is not None and bad[i] <= caps[i]:
                    bad[i] = None
                if have == need or (bad[i] is not None
                                    and bad[i] / caps[i] < 1.002):
                    continue
                if band_dz_range(profile, band)[1] < 0.99 * caps[i]:
                    # The cap no longer binds: the density alone puts
                    # more than the minimum here.
                    continue
                settled = False
                caps[i] = (math.sqrt(caps[i] * bad[i]) if bad[i] is not None
                           else caps[i] * min(2.0, have / need))
            else:
                bad[i] = caps[i]
                if ok[i] is not None and ok[i] >= caps[i]:
                    ok[i] = None
                caps[i] = (math.sqrt(caps[i] * ok[i]) if ok[i] is not None
                           else caps[i] * max(0.5, (have + 0.25)
                                              / (need + 0.25)))
        if settled:
            break
    if best is None:
        # No caps both reached the top and met every count: the same
        # refusal, found by exhausting the search rather than the brackets.
        raise unreachable()
    eta, profile, caps = best
    settings = {"nz": nz, "dzbot": float(dzbot), "max_stretch": r,
                "max_dz": max_dz, "surface_elevation_m": z0,
                "reference": reference, "base_temp": float(base_temp),
                "hybrid_opt": int(hybrid_opt), "etac": float(etac),
                "damping_depth_m": float(damping_depth_m),
                "damping_coarsening": float(damping_coarsening),
                "bias_heights": int(heights.size),
                "bias_weight": float(bias_weight),
                "bias_bandwidth_m": float(bias_bandwidth_m),
                "band_caps_m": tuple(float(cap) for cap in caps)}
    return BandColumn(eta=eta, p_top_pa=p_top, profile=profile, bands=bands,
                      settings=settings)


def band_biased_eta(**kwargs) -> np.ndarray:
    """The ``nz + 1`` full eta levels of :func:`band_biased_column`."""

    return band_biased_column(**kwargs).eta


# --------------------------------------------------------------------------
# ``python -m woof.vertical_bands``: build, report, optionally write


def main(argv=None) -> int:
    from woof.cli_numbers import positive_float

    parser = argparse.ArgumentParser(
        prog="python -m woof.vertical_bands",
        description="Build a height-band-biased eta ladder, print its dz "
                    "profile, and optionally write it for --eta-levels FILE.")
    parser.add_argument("--nz", type=int, default=DEFAULT_NZ)
    top = parser.add_mutually_exclusive_group(required=True)
    top.add_argument("--p-top-pa", type=positive_float)
    top.add_argument("--ztop-m", type=positive_float,
                     help="model top in m MSL")
    parser.add_argument("--dzbot", type=positive_float,
                        default=DEFAULT_DZBOT_M)
    parser.add_argument("--max-stretch", type=positive_float,
                        default=DEFAULT_MAX_STRETCH)
    parser.add_argument("--max-dz", type=positive_float, default=None)
    parser.add_argument("--band", type=parse_band, action="append",
                        default=[], metavar="LO:HI:N[:msl|agl]")
    parser.add_argument("--bias-heights", type=parse_bias_heights,
                        default=(), metavar="LIST|FILE")
    parser.add_argument("--bias-weight", type=float,
                        default=DEFAULT_BIAS_WEIGHT)
    parser.add_argument("--surface-elevation-m", type=float, default=0.0)
    parser.add_argument("--reference", choices=REFERENCE_ATMOSPHERES,
                        default=DEFAULT_REFERENCE)
    parser.add_argument("--write", metavar="FILE", default=None)
    parser.add_argument("--table", action="store_true",
                        help="print every level")
    args = parser.parse_args(argv)
    try:
        column = band_biased_column(
            nz=args.nz, dzbot=args.dzbot, max_stretch=args.max_stretch,
            p_top_pa=args.p_top_pa, ztop_m=args.ztop_m, bands=args.band,
            bias_heights=args.bias_heights, bias_weight=args.bias_weight,
            max_dz=args.max_dz, surface_elevation_m=args.surface_elevation_m,
            reference=args.reference)
    except BandColumnRefused as error:
        parser.exit(2, f"refused: {error}\n")
    for line in column.lines():
        print(line)
    if args.table:
        profile = column.profile
        for k in range(column.nz):
            print(f"{k:4d} z_mass {profile['z_mass_m'][k]:9.1f} m  dz "
                  f"{profile['dz_m'][k]:7.1f} m")
    if args.write:
        write_eta_file(args.write, column.eta, profile=column.profile,
                       header=column.lines())
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
