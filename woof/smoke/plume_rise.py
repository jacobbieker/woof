"""Smoke injection-height profiles onto a model column.

Every profile turns a surface emission flux ``F`` (kg m-2 s-1, e.g. from
:func:`woof.smoke.emission.pm25_emission_flux`) into a volume source
``S[k]`` (kg m-3 s-1) on the column's ``nz`` layers, normalised so the
column integral is the surface flux::

    sum_k S[k] * (z_w[k+1] - z_w[k]) == F

to rounding.  The tracer mixing-ratio tendency is then ``S[k] / rho[k]``
(kg kg-1 s-1).  ``z_w`` are the interface heights above ground
``(nz+1, ...)``, strictly increasing, ``z_w[0]`` the surface.

THE THREE PROFILES.

* :func:`lowest_layer_profile` -- the whole flux into layer 0, which is
  what a resolved fire-atmosphere model such as WRF-SFIRE does (the fire's
  own heat flux lifts the smoke on the model grid).
* :func:`pbl_uniform_profile` -- uniform in height from the ground to
  ``fraction * pbl_height``, the "well mixed in the boundary layer"
  assumption of many regional smoke systems; a layer the top cuts through
  receives the share of its depth below the top.
* :func:`plume_rise_profile` -- the injection layer from a 1-D plume-rise
  estimate (:func:`plume_rise`) against the column's own potential
  temperature and wind; the flux is spread uniformly between the plume's
  injection bottom and its top, with an optional share kept in layer 0
  (WRF-Chem's smouldering fraction).

All three go through one distribution rule, :func:`uniform_profile`:
uniform over ``[z_bot, z_top]`` clipped to the column, layer ``k`` taking
``overlap_k / (z_top - z_bot)`` of the flux.  A degenerate interval
(``z_top <= z_bot`` after clipping) puts the whole flux in the layer that
contains ``z_bot``, so no column ever loses its emission.

THE PLUME MODEL, PINNED.  A steady, Boussinesq, top-hat integral plume in
the spirit of the 1-D plume-rise model of Freitas et al. (2007, ACP 7,
3385-3398; the ``plumerise_opt`` of WRF-Chem) with the environmental wind
entrainment of Freitas et al. (2010, ACP 10, 585-594), integrated in height
on the column's own levels.  It is a simplification: no moisture, no
microphysics, no drag, and the fire's area enters only as the initial
plume radius.  In flux variables (the factor pi dropped throughout) with
plume radius ``R``, vertical velocity ``w``, potential temperature excess
``dth = theta_p - theta_e`` and horizontal velocity ``u_p``::

    m = R^2 w        p = m w        b = m dth        h = m u_p

    dm/dz = E = 2 R (alpha w + beta |U_e - u_p|)        entrainment
    dp/dz = (m / w) g (dth / theta_e) / (1 + gamma)     buoyancy
    db/dz = -m d(theta_e)/dz                            stratification
    dh/dz = U_e E                                       wind entrainment

``alpha`` is the Morton, Taylor and Turner (1956) vertical entrainment
coefficient, ``beta`` the cross-flow coefficient of bent-over plumes
(Hoult, Fay and Forney 1969), ``gamma = 0.5`` the virtual-mass factor
Freitas et al. (2007) take from Simpson and Wiggert (1969)
(:class:`PlumeRiseParameters`).  The plume starts at the ground from the
fire's convective heat flux ``Q`` (W m-2) and area ``A`` (m2)::

    R0   = sqrt(A / pi)
    w0   = (g Q R0 / (rho c_p theta_e(0)))^(1/3)     free-convection scale
    dth0 = Q / (rho c_p w0)                          so rho c_p w0 dth0 = Q
    u_p0 = 0

(the heated air leaves the combustion zone with no horizontal momentum and
is bent over by the entrained wind, which is what lets the wind lower the
plume) and rises until the momentum flux ``p`` reaches zero: that height is the
plume top ``z_top``, found by linear interpolation of ``p`` inside the
step that crosses zero (a plume still rising at the column top stops
there).  The injection bottom is ``injection_bottom_fraction * z_top``.

The environment is piecewise linear in height between the mass-level
heights ``z_m = (z_w[k] + z_w[k+1]) / 2``, and held constant below the
first and above the last; ``theta`` and ``wind`` (the horizontal wind
speed) are given on the mass levels.  The integration runs over the
``nz + 1`` segments ``[z_w[0], z_m[0]], [z_m[0], z_m[1]], ...,
[z_m[nz-1], z_w[nz]]``, each in ``substeps`` equal steps of Heun's
predictor-corrector; a step whose predictor or corrector momentum flux is
non-positive ends the plume there.  :func:`plume_rise_np` and
:func:`plume_rise` take the same steps in the same order, the first per
column in float64 scalars, the second vectorised over the columns.

ORDERS OF MAGNITUDE.  A 50 kW m-2 fire over one hectare (500 MW) under a
1.5 km neutral mixed layer capped by a 2 K, 100 m inversion (4 K km-1
above) in a 5 m s-1 wind tops out near 1.7 km with the default
parameters: just above the boundary layer, where the MISR plume-height
climatology (Val Martin et al. 2010, ACP 10, 1491-1510) finds most North
American fire plumes, and the same order as the Sofiev et al. (2012, ACP
12, 1995-2006) fit.  A stronger wind, a smaller fire or a weaker heat flux
lowers it; a larger fire lifts it (1 km2 at 50 kW m-2 reaches 2.3 km).

KNOWN LIMITATION.  In an exactly neutral layer the buoyancy flux is
conserved and a pure integral plume never stops, so every fire reaches at
least the capping inversion; a plume is limited by the stratification it
meets, not by the boundary layer's turbulence.  For small fires this
amounts to the "mixed through the boundary layer" assumption of
:func:`pbl_uniform_profile`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from woof.core import constants as c
from woof.smoke.emission import array_module

#: Reference air density for the plume's initial state (kg m-3).
AIR_DENSITY = 1.2


@dataclass(frozen=True)
class PlumeRiseParameters:
    """The plume model's coefficients and step control (module notes)."""

    #: Vertical (Morton-Taylor-Turner) entrainment coefficient.
    alpha: float = 0.1
    #: Cross-flow (bent-over plume) entrainment coefficient.
    beta: float = 0.5
    #: Virtual-mass factor: buoyancy accelerates by 1 / (1 + gamma).
    gamma: float = 0.5
    #: Heun steps per integration segment.
    substeps: int = 8
    #: Injection bottom as a fraction of the plume top.
    injection_bottom_fraction: float = 0.5
    #: Share of the flux kept in layer 0 (WRF-Chem's smouldering share).
    surface_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not (self.alpha > 0.0 and self.beta >= 0.0 and self.gamma >= 0.0):
            raise ValueError("alpha must be positive, beta and gamma "
                             "non-negative")
        if (not isinstance(self.substeps, (int, np.integer))
                or isinstance(self.substeps, bool) or self.substeps < 1):
            raise ValueError("substeps must be a positive integer")
        if not 0.0 <= self.injection_bottom_fraction <= 1.0:
            raise ValueError("injection_bottom_fraction must be in [0, 1]")
        if not 0.0 <= self.surface_fraction <= 1.0:
            raise ValueError("surface_fraction must be in [0, 1]")


DEFAULT_PARAMETERS = PlumeRiseParameters()


class PlumeRiseResult(NamedTuple):
    """Per-column plume heights (m above ground) and initial state."""

    top: Any
    bottom: Any
    #: Initial vertical velocity ``w0`` (m s-1); 0 where there is no fire.
    initial_velocity: Any
    #: Initial potential temperature excess ``dth0`` (K).
    initial_excess: Any


# --------------------------------------------------------------------------
# column checks


def _check_interfaces(z_w: Any, xp) -> None:
    if z_w.ndim < 1 or z_w.shape[0] < 2:
        raise ValueError("z_w needs at least two interface heights on axis 0")
    if not bool(xp.all(xp.isfinite(z_w))):
        raise ValueError("z_w must be finite")
    if not bool(xp.all(z_w[1:] > z_w[:-1])):
        raise ValueError("z_w must increase strictly along axis 0")


def _check_mass(name: str, field: Any, z_w: Any) -> None:
    if field.shape != (z_w.shape[0] - 1,) + z_w.shape[1:]:
        raise ValueError(f"{name} has shape {field.shape}; the mass levels "
                         f"of z_w {z_w.shape} are "
                         f"{(z_w.shape[0] - 1,) + z_w.shape[1:]}")


# --------------------------------------------------------------------------
# distribution rule


def _uniform_column(zw: np.ndarray, flux: float, bot: float,
                    top: float) -> np.ndarray:
    """One column of :func:`uniform_profile_np` in float64 scalars."""
    nz = zw.shape[0] - 1
    bot = min(max(bot, zw[0]), zw[nz])
    top = min(max(top, zw[0]), zw[nz])
    out = np.zeros(nz)
    depth = top - bot
    if depth > 0.0:
        for k in range(nz):
            overlap = min(zw[k + 1], top) - max(zw[k], bot)
            if overlap > 0.0:
                out[k] = flux * (overlap / depth) / (zw[k + 1] - zw[k])
        return out
    k = nz - 1
    for j in range(nz):
        if zw[j] <= bot < zw[j + 1]:
            k = j
            break
    out[k] = flux / (zw[k + 1] - zw[k])
    return out


def _columns(z_w: np.ndarray, *planes: Any) -> tuple:
    """Float64 views with the column axes broadcast to z_w's trailing shape."""
    cols = z_w.shape[1:]
    return tuple(np.broadcast_to(np.asarray(a, dtype=np.float64), cols)
                 for a in planes)


def uniform_profile_np(z_w: Any, flux: Any, z_bot: Any,
                       z_top: Any) -> np.ndarray:
    """Reference uniform injection over ``[z_bot, z_top]`` (kg m-3 s-1).

    ``flux`` (kg m-2 s-1), ``z_bot`` and ``z_top`` (m above ground) are
    per column (scalars broadcast).  Float64, column by column.
    """
    zw = np.asarray(z_w, dtype=np.float64)
    _check_interfaces(zw, np)
    flux, z_bot, z_top = _columns(zw, flux, z_bot, z_top)
    out = np.zeros((zw.shape[0] - 1,) + zw.shape[1:])
    for col in np.ndindex(zw.shape[1:]):
        sl = (slice(None),) + col
        out[sl] = _uniform_column(zw[sl], float(flux[col]),
                                  float(z_bot[col]), float(z_top[col]))
    return out


def uniform_profile(z_w: Any, flux: Any, z_bot: Any, z_top: Any) -> Any:
    """Uniform injection over ``[z_bot, z_top]`` on the caller's arrays.

    The vectorised :func:`uniform_profile_np`; keeps ``z_w``'s dtype.
    """
    xp = array_module(z_w, flux, z_bot, z_top)
    zw = xp.asarray(z_w)
    if zw.dtype.kind != "f":
        zw = zw.astype(np.float64)
    _check_interfaces(zw, xp)
    dtype = zw.dtype
    nz = zw.shape[0] - 1
    flux = xp.asarray(flux, dtype=dtype)
    lo, hi = zw[:-1], zw[1:]
    bot = xp.clip(xp.asarray(z_bot, dtype=dtype), zw[0], zw[-1])
    top = xp.clip(xp.asarray(z_top, dtype=dtype), zw[0], zw[-1])
    depth = top - bot
    thick = depth > 0
    overlap = xp.maximum(xp.minimum(hi, top) - xp.maximum(lo, bot), 0)
    share = overlap / xp.where(thick, depth, 1)
    # A degenerate interval: the layer containing z_bot (the top layer when
    # z_bot sits on the column top), exactly as the reference searches.
    level = xp.arange(nz).reshape((nz,) + (1,) * (zw.ndim - 1))
    contains = (lo <= bot) & (bot < hi)
    contains = contains | ((level == nz - 1) & ~xp.any(contains, axis=0))
    weight = xp.where(thick, share, contains.astype(dtype))
    return flux * weight / (hi - lo)


def lowest_layer_profile_np(z_w: Any, flux: Any) -> np.ndarray:
    """Reference: the whole flux into layer 0 (kg m-3 s-1), float64."""
    zw = np.asarray(z_w, dtype=np.float64)
    _check_interfaces(zw, np)
    (flux,) = _columns(zw, flux)
    out = np.zeros((zw.shape[0] - 1,) + zw.shape[1:])
    for col in np.ndindex(zw.shape[1:]):
        out[(0,) + col] = flux[col] / (zw[(1,) + col] - zw[(0,) + col])
    return out


def lowest_layer_profile(z_w: Any, flux: Any) -> Any:
    """The whole flux into layer 0, on the caller's arrays."""
    xp = array_module(z_w, flux)
    zw = xp.asarray(z_w)
    if zw.dtype.kind != "f":
        zw = zw.astype(np.float64)
    _check_interfaces(zw, xp)
    out = xp.zeros((zw.shape[0] - 1,) + zw.shape[1:], dtype=zw.dtype)
    out[0] = xp.asarray(flux, dtype=zw.dtype) / (zw[1] - zw[0])
    return out


def _pbl_top(pbl_height: Any, fraction: float, xp) -> Any:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    return xp.asarray(pbl_height) * fraction


def pbl_uniform_profile_np(z_w: Any, flux: Any, pbl_height: Any,
                           fraction: float = 1.0) -> np.ndarray:
    """Reference: uniform from the ground to ``fraction * pbl_height``."""
    zw = np.asarray(z_w, dtype=np.float64)
    top = _pbl_top(np.asarray(pbl_height, dtype=np.float64), fraction, np)
    return uniform_profile_np(zw, flux, zw[0], top)


def pbl_uniform_profile(z_w: Any, flux: Any, pbl_height: Any,
                        fraction: float = 1.0) -> Any:
    """Uniform from the ground to ``fraction * pbl_height``, vectorised."""
    xp = array_module(z_w, flux, pbl_height)
    zw = xp.asarray(z_w)
    return uniform_profile(zw, flux, zw[0], _pbl_top(pbl_height, fraction, xp))


# --------------------------------------------------------------------------
# plume rise: float64 reference, one column at a time


def _rhs_scalar(z, m, p, b, h, za, seglen, tha, thb, ua, ub, prm):
    """The four flux derivatives at height ``z`` (module notes)."""
    frac = (z - za) / seglen
    theta_e = tha + (thb - tha) * frac
    wind_e = ua + (ub - ua) * frac
    dthdz = (thb - tha) / seglen
    w = p / m
    radius = math.sqrt(m / w)
    entrain = 2.0 * radius * (prm.alpha * w + prm.beta * abs(wind_e - h / m))
    dp = (m / w) * c.G * ((b / m) / theta_e) / (1.0 + prm.gamma)
    return entrain, dp, -m * dthdz, wind_e * entrain


def _initial_state(heat_flux, area, theta0, rho):
    """``(w0, dth0)`` of a fire, or ``(0, 0)`` where there is none."""
    if not (heat_flux > 0.0 and area > 0.0):
        return 0.0, 0.0
    radius = math.sqrt(area / math.pi)
    w0 = (c.G * heat_flux * radius / (rho * c.CP * theta0)) ** (1.0 / 3.0)
    return w0, heat_flux / (rho * c.CP * w0)


def _plume_column(zw, theta, wind, heat_flux, area, rho, prm):
    """``(top, w0, dth0)`` of one column, float64 scalars."""
    nz = theta.shape[0]
    w0, dth0 = _initial_state(heat_flux, area, theta[0], rho)
    if w0 == 0.0:
        return 0.0, 0.0, 0.0
    zm = 0.5 * (zw[:-1] + zw[1:])
    nodes = [zw[0]] + list(zm) + [zw[nz]]
    th_nodes = [theta[0]] + list(theta) + [theta[nz - 1]]
    u_nodes = [wind[0]] + list(wind) + [wind[nz - 1]]
    radius0 = math.sqrt(area / math.pi)
    m = radius0 * radius0 * w0
    p = m * w0
    b = m * dth0
    h = 0.0
    for s in range(nz + 1):
        za, seglen = nodes[s], nodes[s + 1] - nodes[s]
        env = (za, seglen, th_nodes[s], th_nodes[s + 1], u_nodes[s],
               u_nodes[s + 1], prm)
        dz = seglen / prm.substeps
        for j in range(prm.substeps):
            z0 = za + j * dz
            k1 = _rhs_scalar(z0, m, p, b, h, *env)
            p1 = p + dz * k1[1]
            if p1 <= 0.0:
                return z0 + dz * (p / (p - p1)), w0, dth0
            k2 = _rhs_scalar(z0 + dz, m + dz * k1[0], p1, b + dz * k1[2],
                             h + dz * k1[3], *env)
            pn = p + (0.5 * dz) * (k1[1] + k2[1])
            if pn <= 0.0:
                return z0 + dz * (p / (p - pn)), w0, dth0
            m = m + (0.5 * dz) * (k1[0] + k2[0])
            b = b + (0.5 * dz) * (k1[2] + k2[2])
            h = h + (0.5 * dz) * (k1[3] + k2[3])
            p = pn
    return float(zw[nz]), w0, dth0


def _check_fire(heat_flux, fire_area, air_density, xp) -> None:
    for name, value in (("heat_flux", heat_flux), ("fire_area", fire_area)):
        value = xp.asarray(value)
        if not bool(xp.all((value >= 0) & xp.isfinite(value))):
            raise ValueError(f"{name} must be finite and non-negative")
    if not bool(xp.all(xp.asarray(air_density) > 0)):
        raise ValueError("air_density must be positive (kg m-3)")


def _plume_inputs(z_w, theta, wind, xp, dtype=None):
    zw = xp.asarray(z_w) if dtype is None else xp.asarray(z_w, dtype=dtype)
    if zw.dtype.kind != "f":
        zw = zw.astype(np.float64)
    _check_interfaces(zw, xp)
    th = xp.asarray(theta, dtype=zw.dtype)
    u = xp.asarray(wind, dtype=zw.dtype)
    _check_mass("theta", th, zw)
    _check_mass("wind", u, zw)
    if not bool(xp.all(th > 0)):
        raise ValueError("theta must be positive (K)")
    if not bool(xp.all(u >= 0)):
        raise ValueError("wind is the horizontal wind speed, >= 0 (m s-1)")
    return zw, th, u


def plume_rise_np(z_w: Any, theta: Any, wind: Any, heat_flux: Any,
                  fire_area: Any, air_density: Any = AIR_DENSITY,
                  params: PlumeRiseParameters = DEFAULT_PARAMETERS
                  ) -> PlumeRiseResult:
    """Reference plume top and injection bottom per column, float64.

    ``theta`` (K) and ``wind`` (horizontal speed, m s-1) are on the mass
    levels ``(nz, ...)``; ``heat_flux`` is the fire's convective heat flux
    into the air (W m-2), ``fire_area`` the burning area (m2) and
    ``air_density`` (kg m-3) the near-surface air density, all per column
    (scalars broadcast).  A column with no heat flux or no area has top and
    bottom 0.
    """
    zw, th, u = _plume_inputs(z_w, theta, wind, np, np.float64)
    q, area, rho = _columns(zw, heat_flux, fire_area, air_density)
    _check_fire(q, area, rho, np)
    cols = zw.shape[1:]
    top, w0, dth0 = np.zeros(cols), np.zeros(cols), np.zeros(cols)
    for col in np.ndindex(cols):
        sl = (slice(None),) + col
        top[col], w0[col], dth0[col] = _plume_column(
            zw[sl], th[sl], u[sl], float(q[col]), float(area[col]),
            float(rho[col]), params)
    return PlumeRiseResult(top, params.injection_bottom_fraction * top,
                           w0, dth0)


# --------------------------------------------------------------------------
# plume rise: vectorised over columns, on the caller's array module


def _rhs_vector(xp, z, m, p, b, h, za, seglen, tha, thb, ua, ub, prm):
    """:func:`_rhs_scalar`, operation for operation, over columns."""
    frac = (z - za) / seglen
    theta_e = tha + (thb - tha) * frac
    wind_e = ua + (ub - ua) * frac
    dthdz = (thb - tha) / seglen
    w = p / m
    radius = xp.sqrt(m / w)
    entrain = 2.0 * radius * (prm.alpha * w
                              + prm.beta * xp.abs(wind_e - h / m))
    dp = (m / w) * c.G * ((b / m) / theta_e) / (1.0 + prm.gamma)
    return entrain, dp, -m * dthdz, wind_e * entrain


def plume_rise(z_w: Any, theta: Any, wind: Any, heat_flux: Any,
               fire_area: Any, air_density: Any = AIR_DENSITY,
               params: PlumeRiseParameters = DEFAULT_PARAMETERS
               ) -> PlumeRiseResult:
    """:func:`plume_rise_np` vectorised over columns, in ``z_w``'s dtype.

    The same Heun steps in the same order; a column that has stopped is
    frozen (its state no longer advances) while the others go on, and the
    loop ends early once every column has stopped.
    """
    xp = array_module(z_w, theta, wind, heat_flux, fire_area, air_density)
    zw, th, u = _plume_inputs(z_w, theta, wind, xp)
    dtype, cols = zw.dtype, zw.shape[1:]
    nz = th.shape[0]
    q = xp.broadcast_to(xp.asarray(heat_flux, dtype=dtype), cols)
    area = xp.broadcast_to(xp.asarray(fire_area, dtype=dtype), cols)
    rho = xp.broadcast_to(xp.asarray(air_density, dtype=dtype), cols)
    _check_fire(q, area, rho, xp)

    fire = (q > 0) & (area > 0)
    safe_q = xp.where(fire, q, 1)
    radius0 = xp.sqrt(xp.where(fire, area, 1) / math.pi)
    w0 = (c.G * safe_q * radius0 / (rho * c.CP * th[0])) ** (1.0 / 3.0)
    dth0 = safe_q / (rho * c.CP * w0)
    m = radius0 * radius0 * w0
    p = m * w0
    b = m * dth0
    h = xp.zeros_like(m)

    zm = 0.5 * (zw[:-1] + zw[1:])
    nodes = xp.concatenate([zw[:1], zm, zw[-1:]])
    th_nodes = xp.concatenate([th[:1], th, th[-1:]])
    u_nodes = xp.concatenate([u[:1], u, u[-1:]])

    alive = fire.copy()
    top = xp.where(fire, zw[-1], 0).astype(dtype)
    for s in range(nz + 1):
        if not bool(xp.any(alive)):
            break
        za, seglen = nodes[s], nodes[s + 1] - nodes[s]
        env = (za, seglen, th_nodes[s], th_nodes[s + 1], u_nodes[s],
               u_nodes[s + 1], params)
        dz = seglen / params.substeps
        for j in range(params.substeps):
            z0 = za + j * dz
            k1 = _rhs_vector(xp, z0, m, p, b, h, *env)
            p1 = p + dz * k1[1]
            stop = alive & (p1 <= 0)
            top = xp.where(stop, z0 + dz * (p / xp.where(stop, p - p1, 1)),
                           top)
            alive = alive & ~stop
            # The corrector is evaluated everywhere; a column that stopped
            # (or never started) sees a harmless positive momentum flux.
            p1_safe = xp.where(alive, p1, p)
            k2 = _rhs_vector(xp, z0 + dz, m + dz * k1[0], p1_safe,
                             b + dz * k1[2], h + dz * k1[3], *env)
            pn = p + (0.5 * dz) * (k1[1] + k2[1])
            stop = alive & (pn <= 0)
            top = xp.where(stop, z0 + dz * (p / xp.where(stop, p - pn, 1)),
                           top)
            alive = alive & ~stop
            m = xp.where(alive, m + (0.5 * dz) * (k1[0] + k2[0]), m)
            b = xp.where(alive, b + (0.5 * dz) * (k1[2] + k2[2]), b)
            h = xp.where(alive, h + (0.5 * dz) * (k1[3] + k2[3]), h)
            p = xp.where(alive, pn, p)

    zero = xp.zeros(cols, dtype=dtype)
    return PlumeRiseResult(top, params.injection_bottom_fraction * top,
                           xp.where(fire, w0, zero),
                           xp.where(fire, dth0, zero))


# --------------------------------------------------------------------------
# plume-rise injection profile


def plume_rise_profile_np(z_w: Any, flux: Any, theta: Any, wind: Any,
                          heat_flux: Any, fire_area: Any,
                          air_density: Any = AIR_DENSITY,
                          params: PlumeRiseParameters = DEFAULT_PARAMETERS
                          ) -> tuple[np.ndarray, PlumeRiseResult]:
    """Reference plume-rise injection (kg m-3 s-1) and the plume heights.

    ``surface_fraction`` of the flux goes to layer 0, the rest uniformly
    over ``[bottom, top]`` of :func:`plume_rise_np`.
    """
    zw = np.asarray(z_w, dtype=np.float64)
    plume = plume_rise_np(zw, theta, wind, heat_flux, fire_area,
                          air_density, params)
    (flux,) = _columns(zw, flux)
    lofted = uniform_profile_np(
        zw, flux * (1.0 - params.surface_fraction), plume.bottom, plume.top)
    kept = lowest_layer_profile_np(zw, flux * params.surface_fraction)
    return lofted + kept, plume


def plume_rise_profile(z_w: Any, flux: Any, theta: Any, wind: Any,
                       heat_flux: Any, fire_area: Any,
                       air_density: Any = AIR_DENSITY,
                       params: PlumeRiseParameters = DEFAULT_PARAMETERS
                       ) -> tuple[Any, PlumeRiseResult]:
    """:func:`plume_rise_profile_np` on the caller's arrays."""
    xp = array_module(z_w, flux, theta, wind, heat_flux, fire_area,
                      air_density)
    zw = xp.asarray(z_w)
    if zw.dtype.kind != "f":
        zw = zw.astype(np.float64)
    plume = plume_rise(zw, theta, wind, heat_flux, fire_area, air_density,
                       params)
    flux = xp.asarray(flux, dtype=zw.dtype)
    lofted = uniform_profile(zw, flux * (1.0 - params.surface_fraction),
                             plume.bottom, plume.top)
    kept = lowest_layer_profile(zw, flux * params.surface_fraction)
    return lofted + kept, plume
