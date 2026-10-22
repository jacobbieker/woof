"""Removal of smoke particles: dry deposition and gravitational settling.

THE DEPOSITION VELOCITY, PINNED (Zhang, Gong, Padro and Barrie 2001,
Atmos. Environ. 35, 549-560, eqs. 1-7, with the Seinfeld and Pandis
(2016, ch. 9) particle mechanics).  For particles of diameter ``d`` and
density ``rho_p`` in air at temperature ``T`` and pressure ``p``::

    mu      = 1.458e-6 T^1.5 / (T + 110.4)            Sutherland viscosity
    lambda  = (mu / p) sqrt(pi R_d T / 2)             mean free path
    C_c     = 1 + (2 lambda / d) (1.257 + 0.4 exp(-0.55 d / lambda))
    v_g     = rho_p d^2 g C_c / (18 mu)               settling (Stokes)
    nu      = mu R_d T / p                            kinematic viscosity
    D_B     = k_B T C_c / (3 pi mu d)                 Brownian diffusivity

    R_a = (ln(z_r / z_0) - psi_h(z_r / L)) / (kappa u*),  held >= 0
    E_B = Sc^(-gamma),  Sc = nu / D_B                 Brownian diffusion
    St  = v_g u* / (g A)   (vegetated: collector radius A)
        = v_g u*^2 / nu    (smooth surfaces: water, desert, ice)
    E_IM = (St / (alpha + St))^2                      impaction
    E_IN = (d / A)^2 / 2   (vegetated only)           interception
    R_1  = exp(-sqrt(St))                             rebound
    R_s  = 1 / (eps_0 u* (E_B + E_IM + E_IN) R_1),  eps_0 = 3

    v_d = v_g + 1 / (R_a + R_s)

``psi_h`` is the Businger-Dyer stability correction for heat (Paulson
1970): ``-5 zeta`` for ``zeta >= 0`` and ``2 ln((1 + sqrt(1 - 16 zeta))/2)``
below; no Obukhov length is neutral (``psi_h = 0``).  ``z_r`` is the
height the resistance is taken to, normally the lowest mass level.  The
land classes carry Zhang et al.'s (2001, Table 3) midsummer ``alpha``,
``gamma`` and collector radius ``A`` (:data:`DEPOSITION_SURFACES`), with a
typical roughness length the caller may replace by the model's own field.

For accumulation-mode smoke (0.1-1 um) the model sits at its minimum,
where neither Brownian diffusion nor impaction is efficient:
``v_d`` of order 0.05-0.5 cm s-1 over vegetation, inside the 0.1-1 cm s-1
measured for fine particles over forests and grass (Zhang et al. 2001,
Fig. 4; Petroff et al. 2008, Atmos. Environ. 42, 3625-3653).

THE COLUMN, PINNED.  Settling moves the tracer down through every
interface at the settling speed of the layer above it, first-order upwind;
the surface interface carries the deposition velocity instead (which
already holds the settling term)::

    F[k]  = rho[k] v_g[k] q[k]    k = 1..nz-1     (kg m-2 s-1, downward)
    F[0]  = rho[0] v_d q[0]                       (the deposition flux)
    F[nz] = 0
    dq[k]/dt = (F[k+1] - F[k]) / (rho[k] dz[k])

so the column's tracer mass changes by exactly ``-F[0]``
(:func:`settling_tendency`).  The tendency is meant for a forward step
with ``v dt <= dz`` in every layer; the speeds here are mm s-1 and below,
so any model time step passes that by orders of magnitude.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple

import numpy as np

from woof.core import constants as c
from woof.smoke.emission import array_module

#: Boltzmann constant (J K-1), exact SI.
BOLTZMANN = 1.380649e-23
#: von Karman constant.
KARMAN = 0.4
#: Zhang et al. (2001) empirical surface-resistance constant ``eps_0``.
EPS0 = 3.0
#: Representative smoke PM2.5 particle: diameter (m) and density (kg m-3).
#: Fresh biomass-burning smoke has its volume (mass) median near 0.25-0.3
#: um dry (Reid et al. 2005, ACP 5, 799-825) and a density of 1.2-1.5
#: g cm-3; these are defaults only.
SMOKE_DIAMETER = 0.3e-6
SMOKE_DENSITY = 1400.0


@dataclass(frozen=True)
class DepositionSurface:
    """One land class's Zhang et al. (2001, Table 3) particle parameters."""

    name: str
    #: Typical roughness length (m); the model's own field may replace it.
    roughness_length: float
    #: Impaction parameter ``alpha``.
    alpha: float
    #: Brownian exponent ``gamma``.
    gamma: float
    #: Characteristic collector radius ``A`` (m); 0 for a smooth surface.
    collector_radius: float

    @property
    def vegetated(self) -> bool:
        return self.collector_radius > 0.0


#: Zhang et al. (2001) Table 3 land classes (midsummer collector radii).
DEPOSITION_SURFACES: Mapping[str, DepositionSurface] = MappingProxyType({
    "needleleaf_forest": DepositionSurface(
        "needleleaf_forest", 0.8, 1.0, 0.56, 2.0e-3),
    "broadleaf_forest": DepositionSurface(
        "broadleaf_forest", 1.0, 0.8, 0.56, 5.0e-3),
    "grass": DepositionSurface("grass", 0.05, 1.2, 0.54, 2.0e-3),
    "crops": DepositionSurface("crops", 0.1, 1.2, 0.54, 2.0e-3),
    "shrub": DepositionSurface("shrub", 0.1, 1.3, 0.54, 10.0e-3),
    "urban": DepositionSurface("urban", 1.0, 1.5, 0.56, 10.0e-3),
    "desert": DepositionSurface("desert", 0.04, 50.0, 0.54, 0.0),
    "water": DepositionSurface("water", 1.0e-4, 100.0, 0.50, 0.0),
})


class DepositionVelocity(NamedTuple):
    """Deposition velocity and its pieces (m s-1, s m-1)."""

    deposition: Any
    settling: Any
    aerodynamic_resistance: Any
    surface_resistance: Any


class ColumnTendency(NamedTuple):
    """Mixing-ratio tendency (q-units s-1) and surface deposition flux."""

    tendency: Any
    #: ``rho[0] v_d q[0]``, downward positive (kg m-2 s-1 per unit q).
    surface_flux: Any


def _surface(surface: DepositionSurface | str) -> DepositionSurface:
    if isinstance(surface, DepositionSurface):
        return surface
    if surface not in DEPOSITION_SURFACES:
        raise ValueError(f"unknown deposition surface {surface!r}; known: "
                         f"{sorted(DEPOSITION_SURFACES)}")
    return DEPOSITION_SURFACES[surface]


def _check_particle(diameter: float, density: float) -> None:
    if not (diameter > 0.0 and density > 0.0):
        raise ValueError("particle diameter and density must be positive")


# --------------------------------------------------------------------------
# float64 reference, one value at a time


def _air_scalar(temperature: float, pressure: float):
    """``(mu, lambda)``: Sutherland viscosity and the mean free path."""
    mu = 1.458e-6 * temperature ** 1.5 / (temperature + 110.4)
    mfp = (mu / pressure) * math.sqrt(math.pi * c.RD * temperature / 2.0)
    return mu, mfp


def _settling_scalar(d: float, rho_p: float, temperature: float,
                     pressure: float):
    """``(v_g, C_c, mu)`` for one particle size and air state."""
    mu, mfp = _air_scalar(temperature, pressure)
    cc = 1.0 + (2.0 * mfp / d) * (1.257 + 0.4 * math.exp(-0.55 * d / mfp))
    return rho_p * d * d * c.G * cc / (18.0 * mu), cc, mu


def _psi_h_scalar(zeta: float) -> float:
    if zeta >= 0.0:
        return -5.0 * zeta
    return 2.0 * math.log((1.0 + math.sqrt(1.0 - 16.0 * zeta)) / 2.0)


def _vd_scalar(d, rho_p, ustar, temperature, pressure, z_ref, z0, inv_l,
               surface):
    vg, cc, mu = _settling_scalar(d, rho_p, temperature, pressure)
    nu = mu * c.RD * temperature / pressure
    diffusivity = BOLTZMANN * temperature * cc / (3.0 * math.pi * mu * d)
    ra = (math.log(z_ref / z0) - _psi_h_scalar(z_ref * inv_l)) / (
        KARMAN * ustar)
    ra = max(ra, 0.0)
    e_b = (nu / diffusivity) ** (-surface.gamma)
    if surface.vegetated:
        stokes = vg * ustar / (c.G * surface.collector_radius)
        e_in = 0.5 * (d / surface.collector_radius) ** 2
    else:
        stokes = vg * ustar * ustar / nu
        e_in = 0.0
    e_im = (stokes / (surface.alpha + stokes)) ** 2
    r1 = math.exp(-math.sqrt(stokes))
    rs = 1.0 / (EPS0 * ustar * (e_b + e_im + e_in) * r1)
    return vg + 1.0 / (ra + rs), vg, ra, rs


def settling_velocity_np(temperature: Any, pressure: Any,
                         diameter: float = SMOKE_DIAMETER,
                         particle_density: float = SMOKE_DENSITY
                         ) -> np.ndarray:
    """Reference Stokes settling speed (m s-1, downward), float64.

    ``temperature`` (K) and ``pressure`` (Pa) of the air, any shape;
    ``diameter`` (m) and ``particle_density`` (kg m-3) of the particles.
    """
    _check_particle(diameter, particle_density)
    t, p = np.broadcast_arrays(np.asarray(temperature, dtype=np.float64),
                               np.asarray(pressure, dtype=np.float64))
    out = np.empty(t.shape)
    for i in np.ndindex(t.shape):
        out[i] = _settling_scalar(diameter, particle_density, float(t[i]),
                                  float(p[i]))[0]
    return out


def _inverse_obukhov(obukhov_length: Any, xp) -> Any:
    """``1/L``, with ``None`` (and an infinite L) neutral."""
    if obukhov_length is None:
        return xp.asarray(0.0)
    length = xp.asarray(obukhov_length)
    if length.dtype.kind != "f":
        length = length.astype(np.float64)
    if bool(xp.any(length == 0)):
        raise ValueError("an Obukhov length of 0 is undefined; pass inf for "
                         "neutral")
    return 1.0 / length


def _check_state(ustar, temperature, pressure, z_ref, z0, xp) -> None:
    for name, value in (("ustar", ustar), ("temperature", temperature),
                        ("pressure", pressure), ("z_ref", z_ref),
                        ("roughness_length", z0)):
        if not bool(xp.all(xp.asarray(value) > 0)):
            raise ValueError(f"{name} must be positive")
    if not bool(xp.all(xp.asarray(z_ref) > xp.asarray(z0))):
        raise ValueError("z_ref must lie above the roughness length")


def dry_deposition_velocity_np(ustar: Any, temperature: Any, pressure: Any,
                               z_ref: Any,
                               surface: DepositionSurface | str = "grass",
                               diameter: float = SMOKE_DIAMETER,
                               particle_density: float = SMOKE_DENSITY,
                               roughness_length: Any = None,
                               obukhov_length: Any = None
                               ) -> DepositionVelocity:
    """Reference Zhang et al. (2001) deposition velocity, float64.

    ``ustar`` friction velocity (m s-1), ``temperature`` (K) and
    ``pressure`` (Pa) of the lowest model layer, ``z_ref`` the height
    (m above ground) the aerodynamic resistance runs to, all broadcast
    together; ``roughness_length`` (m) defaults to the surface class's,
    ``obukhov_length`` (m) to neutral.
    """
    _check_particle(diameter, particle_density)
    surf = _surface(surface)
    z0 = surf.roughness_length if roughness_length is None else (
        roughness_length)
    _check_state(ustar, temperature, pressure, z_ref, z0, np)
    inv_l = _inverse_obukhov(obukhov_length, np)
    arrays = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (
        ustar, temperature, pressure, z_ref, z0, inv_l)))
    outs = [np.empty(arrays[0].shape) for _ in range(4)]
    for i in np.ndindex(arrays[0].shape):
        row = _vd_scalar(diameter, particle_density,
                         *(float(a[i]) for a in arrays), surf)
        for out, value in zip(outs, row):
            out[i] = value
    return DepositionVelocity(*outs)


def _column_arrays(q, z_w, v_settle, xp, rho, dtype=None):
    qa = xp.asarray(q) if dtype is None else xp.asarray(q, dtype=dtype)
    if qa.dtype.kind != "f":
        qa = qa.astype(np.float64)
    zw = xp.asarray(z_w, dtype=qa.dtype)
    if zw.shape[0] != qa.shape[0] + 1 or zw.shape[1:] != qa.shape[1:]:
        raise ValueError(f"z_w {zw.shape} must be q's interfaces, "
                         f"{(qa.shape[0] + 1,) + qa.shape[1:]}")
    if not bool(xp.all(zw[1:] > zw[:-1])):
        raise ValueError("z_w must increase strictly along axis 0")
    vs = xp.broadcast_to(xp.asarray(v_settle, dtype=qa.dtype), qa.shape)
    density = (xp.ones(qa.shape, dtype=qa.dtype) if rho is None else
               xp.broadcast_to(xp.asarray(rho, dtype=qa.dtype), qa.shape))
    if not (bool(xp.all(vs >= 0)) and bool(xp.all(xp.isfinite(vs)))
            and bool(xp.all(density > 0)) and bool(xp.all(xp.isfinite(qa)))):
        raise ValueError("q must be finite, settling speeds finite and "
                         ">= 0, and rho positive")
    return qa, zw, vs, density


def settling_tendency_np(q: Any, z_w: Any, v_settle: Any, v_deposit: Any,
                         rho: Any = None) -> ColumnTendency:
    """Reference settling and deposition tendency of a tracer, float64.

    ``q`` is the tracer mixing ratio ``(nz, ...)`` (kg kg-1, or any
    per-mass-of-air unit), ``z_w`` its interface heights ``(nz+1, ...)``,
    ``v_settle`` the settling speed per layer (m s-1, downward; broadcast
    to ``q``), ``v_deposit`` the deposition velocity per column (m s-1) and
    ``rho`` the air density per layer (kg m-3; ``None`` is uniform, when it
    cancels).  Column by column, interface by interface.
    """
    qa, zw, vs, density = _column_arrays(q, z_w, v_settle, np, rho,
                                         np.float64)
    vd = np.broadcast_to(np.asarray(v_deposit, dtype=np.float64),
                         qa.shape[1:])
    if not (np.all(vd >= 0) and np.all(np.isfinite(vd))):
        raise ValueError("deposition velocity must be finite and >= 0")
    nz = qa.shape[0]
    tendency = np.zeros(qa.shape)
    surface_flux = np.zeros(qa.shape[1:])
    for col in np.ndindex(qa.shape[1:]):
        flux = [0.0] * (nz + 1)
        flux[0] = density[(0,) + col] * vd[col] * qa[(0,) + col]
        for k in range(1, nz):
            flux[k] = density[(k,) + col] * vs[(k,) + col] * qa[(k,) + col]
        for k in range(nz):
            dz = zw[(k + 1,) + col] - zw[(k,) + col]
            tendency[(k,) + col] = (flux[k + 1] - flux[k]) / (
                density[(k,) + col] * dz)
        surface_flux[col] = flux[0]
    return ColumnTendency(tendency, surface_flux)


# --------------------------------------------------------------------------
# vectorised, on the caller's array module


def settling_velocity(temperature: Any, pressure: Any,
                      diameter: float = SMOKE_DIAMETER,
                      particle_density: float = SMOKE_DENSITY) -> Any:
    """:func:`settling_velocity_np` on the caller's arrays, their dtype."""
    _check_particle(diameter, particle_density)
    xp = array_module(temperature, pressure)
    t = xp.asarray(temperature)
    if t.dtype.kind != "f":
        t = t.astype(np.float64)
    p = xp.asarray(pressure, dtype=t.dtype)
    return _settling_vector(xp, diameter, particle_density, t, p)[0]


def _settling_vector(xp, d, rho_p, t, p):
    mu = 1.458e-6 * t ** 1.5 / (t + 110.4)
    mfp = (mu / p) * xp.sqrt(math.pi * c.RD * t / 2.0)
    cc = 1.0 + (2.0 * mfp / d) * (1.257 + 0.4 * xp.exp(-0.55 * d / mfp))
    return rho_p * d * d * c.G * cc / (18.0 * mu), cc, mu


def _psi_h_vector(xp, zeta):
    unstable = 2.0 * xp.log(
        (1.0 + xp.sqrt(1.0 - 16.0 * xp.minimum(zeta, 0))) / 2.0)
    return xp.where(zeta >= 0, -5.0 * zeta, unstable)


def dry_deposition_velocity(ustar: Any, temperature: Any, pressure: Any,
                            z_ref: Any,
                            surface: DepositionSurface | str = "grass",
                            diameter: float = SMOKE_DIAMETER,
                            particle_density: float = SMOKE_DENSITY,
                            roughness_length: Any = None,
                            obukhov_length: Any = None
                            ) -> DepositionVelocity:
    """:func:`dry_deposition_velocity_np` vectorised, in ``ustar``'s dtype."""
    _check_particle(diameter, particle_density)
    surf = _surface(surface)
    xp = array_module(ustar, temperature, pressure, z_ref, roughness_length,
                      obukhov_length)
    us = xp.asarray(ustar)
    if us.dtype.kind != "f":
        us = us.astype(np.float64)
    dtype = us.dtype
    t = xp.asarray(temperature, dtype=dtype)
    p = xp.asarray(pressure, dtype=dtype)
    zr = xp.asarray(z_ref, dtype=dtype)
    z0 = xp.asarray(surf.roughness_length if roughness_length is None
                    else roughness_length, dtype=dtype)
    _check_state(us, t, p, zr, z0, xp)
    inv_l = _inverse_obukhov(obukhov_length, xp).astype(dtype)

    vg, cc, mu = _settling_vector(xp, diameter, particle_density, t, p)
    nu = mu * c.RD * t / p
    diffusivity = BOLTZMANN * t * cc / (3.0 * math.pi * mu * diameter)
    ra = (xp.log(zr / z0) - _psi_h_vector(xp, zr * inv_l)) / (KARMAN * us)
    ra = xp.maximum(ra, 0)
    e_b = (nu / diffusivity) ** (-surf.gamma)
    if surf.vegetated:
        stokes = vg * us / (c.G * surf.collector_radius)
        e_in = 0.5 * (diameter / surf.collector_radius) ** 2
    else:
        stokes = vg * us * us / nu
        e_in = 0.0
    e_im = (stokes / (surf.alpha + stokes)) ** 2
    r1 = xp.exp(-xp.sqrt(stokes))
    rs = 1.0 / (EPS0 * us * (e_b + e_im + e_in) * r1)
    return DepositionVelocity(vg + 1.0 / (ra + rs), vg, ra, rs)


def settling_tendency(q: Any, z_w: Any, v_settle: Any, v_deposit: Any,
                      rho: Any = None) -> ColumnTendency:
    """:func:`settling_tendency_np` vectorised, in ``q``'s dtype."""
    xp = array_module(q, z_w, v_settle, v_deposit, rho)
    qa, zw, vs, density = _column_arrays(q, z_w, v_settle, xp, rho)
    vd = xp.asarray(v_deposit, dtype=qa.dtype)
    if not (bool(xp.all(vd >= 0)) and bool(xp.all(xp.isfinite(vd)))):
        raise ValueError("deposition velocity must be finite and >= 0")
    flux = xp.zeros((qa.shape[0] + 1,) + qa.shape[1:], dtype=qa.dtype)
    flux[0] = density[0] * vd * qa[0]
    flux[1:-1] = density[1:] * vs[1:] * qa[1:]
    tendency = (flux[1:] - flux[:-1]) / (density * (zw[1:] - zw[:-1]))
    return ColumnTendency(tendency, flux[0])


def deposited_fraction(v_deposit: Any, depth: Any, duration: Any) -> Any:
    """Share of a well-mixed layer's tracer deposited in ``duration`` (s).

    ``1 - exp(-v_d t / h)`` for a layer of depth ``h`` (m) with no source
    and nothing settling in from above: the closed form of the lowest
    layer's tendency above.
    """
    xp = array_module(v_deposit, depth, duration)
    return 1.0 - xp.exp(-xp.asarray(v_deposit) * duration / depth)
