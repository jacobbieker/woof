"""Fire smoke emission: WRF-SFIRE's smoke tracer source and PM2.5 factors.

THE SOURCE, PINNED (WRF v4.7.1 ``phys/module_fr_fire_atm.F:15-106``,
``add_fire_tracer_emissions``, ``tracer_opt = 3``, ``fire_smk_scheme =
0``).  WRF-SFIRE's fire model burns fuel on a refined fire grid, ``ir`` by
``jr`` fire cells under every atmospheric cell (``sr_x``, ``sr_y``).  In
each time step a fire cell burns the fraction ``burnt_area_dt`` of its area
and of its fuel load ``fgip`` (kg m-2), and every fire cell under an
interior atmospheric column adds, to the column's lowest level, in g of
smoke per kg of air::

    emis = avgw * fire_tracer_smoke * burnt_area_dt * fgip * 1000
               / (rho * dz8w)
    avgw = 1.0 / (ir * jr)

one cell after another, ``joff`` (the fire-grid row) outer and ``ioff``
inner, each added to the running tracer.  ``fire_tracer_smoke`` is the
mass of smoke per mass of fuel burned (``Registry/registry.fire:328``,
default 0.02); ``rho`` (kg m-3) and ``dz8w`` (m) are the lowest layer's
density and depth.  The edge columns (the first and last in each
direction) get nothing, as in WRF (``i_st = max(its, ids+1)``, ``i_en =
min(ite, ide-2)``).  :func:`fire_tracer_emissions` (CuPy, one kernel
launch) and :func:`fire_tracer_emissions_np` (the NumPy reference, real(4)
operation for operation) agree with the compiled WRF at zero ULP
(``tests/test_smoke_emission_wrf471_parity.py``).

TWO DEPARTURES FROM WRF, BOTH DELIBERATE.  (1) WRF's i loop is ``do
i=i_st,i_st`` (``module_fr_fire_atm.F:81``): on a tile it emits into the
first interior column only.  These functions apply the same statement to
every interior column; the oracle drives WRF's routine once per column,
on one-column tiles, so every column's words are WRF's own.  (2)
``fire_tracer_smoke`` may be a fire-grid field instead of one number, so
that the factor can follow the fuel (:func:`emission_factor_field`); a
scalar is WRF's form exactly.  Scheme 1 (the truncated-Gaussian release)
is not ported: its ``tg_dist`` reads ``z_w`` before setting it
(``module_fr_fire_atm.F:342``).

THE FACTORS.  WRF's default 0.02 is total smoke.  For PM2.5 the factor is
the mass of PM2.5 per mass of dry fuel, by fuel class
(:data:`EMISSION_FACTORS`, g kg-1):

========  =====  ==========================================================
class     EF     basis
========  =====  ==========================================================
grass     8.0    savanna and grassland (Andreae 2019, ACP 19, 8523-8546,
                 Table 1; Andreae and Merlet 2001, GBC 15, 955-966), and
                 the US grassland values of Urbanski (2014, For. Ecol.
                 Manage. 317, 51-60)
shrub     9.0    chaparral and other shrubland (Urbanski 2014)
timber    15.0   temperate and boreal forest (Andreae 2019, Table 1; the
                 western-US forest values of Urbanski 2014 bracket it)
slash     13.0   logging residue (Urbanski 2014, activity fuels)
========  =====  ==========================================================

Representative values inside the ranges those compilations report (which
scatter by about a factor of two), not a transcription of one table entry.
The thirteen Anderson (1982) fuel models map onto the classes by their own
groups (:data:`ANDERSON_FUEL_CLASS`); category 14 is WRF-SFIRE's "no
fuel", factor 0.  :func:`pm25_emission_flux` turns a burned fuel mass rate
(kg m-2 s-1) into a PM2.5 flux the same way, one product per cell.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

#: Low heat content of the Anderson fuel models, 8000 Btu/lb (J kg-1).
FUEL_HEAT_CONTENT = 8000.0 * 1055.05585 / 0.45359237
#: WRF-SFIRE's "no fuel" category, emission factor zero.
NO_FUEL = 14
#: Registry/registry.fire:328, fire_tracer_smoke's default.
WRF_FIRE_TRACER_SMOKE = 0.02

F = np.float32


@dataclass(frozen=True)
class EmissionFactor:
    """One fuel class's PM2.5 emission factor and where it comes from."""

    fuel_class: str
    pm25_g_per_kg: float
    basis: str

    @property
    def pm25_kg_per_kg(self) -> float:
        """The factor as a mass ratio (kg PM2.5 per kg dry fuel)."""
        return self.pm25_g_per_kg * 1.0e-3


#: The broad fuel classes, g PM2.5 per kg dry matter burned (module notes).
EMISSION_FACTORS: Mapping[str, EmissionFactor] = MappingProxyType({
    "grass": EmissionFactor(
        "grass", 8.0, "Andreae (2019) savanna/grassland; Urbanski (2014)"),
    "shrub": EmissionFactor(
        "shrub", 9.0, "Urbanski (2014) shrubland; Andreae (2019)"),
    "timber": EmissionFactor(
        "timber", 15.0, "Andreae (2019) temperate/boreal forest; "
                        "Urbanski (2014)"),
    "slash": EmissionFactor("slash", 13.0, "Urbanski (2014) activity fuels"),
    "none": EmissionFactor("none", 0.0, "no fuel"),
})

#: Anderson (1982) fuel model -> broad class; 14 is WRF-SFIRE's no fuel.
ANDERSON_FUEL_CLASS: Mapping[int, str] = MappingProxyType({
    1: "grass",     # short grass (1 ft)
    2: "grass",     # timber (grass and understory)
    3: "grass",     # tall grass (2.5 ft)
    4: "shrub",     # chaparral (6 ft)
    5: "shrub",     # brush (2 ft)
    6: "shrub",     # dormant brush, hardwood slash
    7: "shrub",     # southern rough
    8: "timber",    # closed timber litter
    9: "timber",    # hardwood litter
    10: "timber",   # timber (litter and understory)
    11: "slash",    # light logging slash
    12: "slash",    # medium logging slash
    13: "slash",    # heavy logging slash
    NO_FUEL: "none",
})


# --------------------------------------------------------------------------
# factors


def emission_factor(fuel: int | str,
                    factors: Mapping[str, EmissionFactor] = EMISSION_FACTORS
                    ) -> float:
    """PM2.5 emission factor (kg kg-1) of an Anderson category or a class.

    ``fuel`` is an Anderson fuel model number (1-13, or 14 for no fuel) or a
    class name from ``factors``; anything else is refused by name.  No fuel
    emits nothing whether or not ``factors`` lists it.
    """
    if isinstance(fuel, str):
        name = fuel
    else:
        number = int(fuel)
        if number != fuel or number not in ANDERSON_FUEL_CLASS:
            raise ValueError(
                f"fuel category {fuel!r} is not an Anderson fuel model "
                f"(1-13) or the no-fuel category {NO_FUEL}")
        name = ANDERSON_FUEL_CLASS[number]
    if name == "none" and name not in factors:
        return 0.0
    if name not in factors:
        raise ValueError(f"fuel class {name!r} has no emission factor; "
                         f"known classes: {sorted(factors)}")
    return factors[name].pm25_kg_per_kg


def emission_factor_table(
        factors: Mapping[str, EmissionFactor] = EMISSION_FACTORS
) -> np.ndarray:
    """Factors (kg kg-1) by Anderson category, float32, length 15.

    Entry 0, and every category whose class ``factors`` lacks, is NaN, so a
    lookup with an unvalidated index or an incomplete table cannot quietly
    emit nothing (:func:`emission_factor_field` refuses them by name).
    """
    table = np.full(NO_FUEL + 1, np.nan, dtype=np.float32)
    for number, name in ANDERSON_FUEL_CLASS.items():
        if name in factors or name == "none":
            table[number] = F(emission_factor(number, factors))
    return table


def burn_rate_from_heat_flux(heat_flux: float,
                             heat_content: float = FUEL_HEAT_CONTENT
                             ) -> float:
    """Dry fuel burned (kg m-2 s-1) from a fire heat flux (W m-2), a scalar.

    ``m_dot = Q / h`` with ``h`` the fuel's heat content (J kg-1, default
    8000 Btu/lb = 18.6 MJ kg-1, Rothermel 1972).  A Python-float helper for
    setting up cases, not a device path.
    """
    if not heat_content > 0.0:
        raise ValueError(f"heat_content must be positive, got {heat_content}")
    return float(heat_flux) / heat_content


def _categories(fuel, xp):
    category = xp.asarray(fuel)
    if category.dtype.kind not in "iu":
        raise ValueError("fuel categories must be an integer array")
    if category.size and (bool(xp.any(category < 1))
                          or bool(xp.any(category > NO_FUEL))):
        raise ValueError(f"fuel categories must be Anderson models 1-13 or "
                         f"the no-fuel category {NO_FUEL}")
    return category


def _gather(table: np.ndarray, category, xp, factors):
    """``table[category]``, refusing a category whose class has no factor."""
    if np.isnan(table[1:]).any():
        host = category.get() if xp is not np else category
        for number in np.unique(host):
            if np.isnan(table[number]):
                emission_factor(int(number), factors)   # the refusal, by name
    return xp.asarray(table)[category]


def emission_factor_field_np(fuel, factors: Mapping[str, EmissionFactor]
                             = EMISSION_FACTORS) -> np.ndarray:
    """Per-cell PM2.5 factor (kg kg-1, float32) from Anderson categories."""
    table = emission_factor_table(factors)
    return _gather(table, _categories(fuel, np), np, factors)


def emission_factor_field(fuel, factors: Mapping[str, EmissionFactor]
                          = EMISSION_FACTORS):
    """:func:`emission_factor_field_np` on the device (a gather, no maths).

    The result is a ``fire_tracer_smoke`` field for
    :func:`fire_tracer_emissions`.
    """
    import cupy as cp

    table = emission_factor_table(factors)
    return _gather(table, _categories(fuel, cp), cp, factors)


# --------------------------------------------------------------------------
# PM2.5 flux from a burned fuel mass rate


def _check_rate(rate, xp) -> None:
    if not bool(xp.all((rate >= 0) & xp.isfinite(rate))):
        raise ValueError("burn_rate must be finite and non-negative")


def pm25_emission_flux_np(burn_rate, fuel, factors: Mapping[str,
                          EmissionFactor] = EMISSION_FACTORS) -> np.ndarray:
    """Reference PM2.5 flux (kg m-2 s-1, float32): rate times factor.

    ``burn_rate`` is the dry fuel burned (kg m-2 s-1); ``fuel`` the Anderson
    category of each cell (broadcast to it).  One real(4) product per cell.
    """
    rate = np.asarray(burn_rate, dtype=np.float32)
    _check_rate(rate, np)
    factor = emission_factor_field_np(np.broadcast_to(fuel, rate.shape),
                                      factors)
    return rate * factor


def pm25_emission_flux(burn_rate, fuel, factors: Mapping[str,
                       EmissionFactor] = EMISSION_FACTORS):
    """:func:`pm25_emission_flux_np` on the device, one kernel launch."""
    import cupy as cp

    from woof.smoke import _kernels

    rate = cp.ascontiguousarray(cp.asarray(burn_rate, dtype=cp.float32))
    _check_rate(rate, cp)
    factor = cp.ascontiguousarray(emission_factor_field(
        cp.broadcast_to(cp.asarray(fuel), rate.shape), factors))
    flux = cp.empty_like(rate)
    _kernels.kernel("smoke_pm25_flux")(
        _kernels.grid(rate.size), (_kernels.BLOCK,),
        (rate, factor, flux, np.int32(rate.size)))
    return flux


# --------------------------------------------------------------------------
# WRF-SFIRE: add_fire_tracer_emissions, fire_smk_scheme = 0


def _fire_shapes(tracer, rho, dz8w, burnt_area_dt, fgip, refinement):
    if tracer.ndim != 3:
        raise ValueError(f"tracer must be (nz, ny, nx), got {tracer.shape}")
    _, ny, nx = tracer.shape
    ir, jr = (int(r) for r in refinement)
    if ir < 1 or jr < 1:
        raise ValueError(f"refinement must be positive, got {refinement}")
    for name, field in (("rho", rho), ("dz8w", dz8w)):
        if field.shape != (ny, nx):
            raise ValueError(f"{name} level 1 {field.shape} does not match "
                             f"tracer {tracer.shape}")
    fire = (ny * jr, nx * ir)
    for name, field in (("burnt_area_dt", burnt_area_dt), ("fgip", fgip)):
        if field.shape != fire:
            raise ValueError(f"{name} must be the fire grid {fire} for "
                             f"refinement (ir, jr) = {(ir, jr)}, got "
                             f"{field.shape}")
    return ny, nx, ir, jr


def _level1(field):
    return field[0] if field.ndim == 3 else field


def fire_tracer_emissions_np(tracer, rho, dz8w, burnt_area_dt, fgip,
                             fire_tracer_smoke=WRF_FIRE_TRACER_SMOKE, *,
                             refinement=(1, 1)) -> np.ndarray:
    """Reference ``add_fire_tracer_emissions`` (real(4)), a new tracer.

    ``tracer`` is the fire smoke mixing ratio (nz, ny, nx) in g kg-1;
    ``rho`` (kg m-3) and ``dz8w`` (m) are (nz, ny, nx) or the lowest level
    (ny, nx); ``burnt_area_dt`` (fraction burnt this step) and ``fgip``
    (fuel load, kg m-2) are on the fire grid (ny*jr, nx*ir) with
    ``refinement = (ir, jr)``; ``fire_tracer_smoke`` (kg kg-1) is a scalar
    (WRF's form) or a fire-grid field.  Only level 0 changes.
    """
    tracer = np.asarray(tracer, dtype=np.float32)
    rho, dz8w = (_level1(np.asarray(a, dtype=np.float32)) for a in (rho, dz8w))
    burnt = np.asarray(burnt_area_dt, dtype=np.float32)
    fuel = np.asarray(fgip, dtype=np.float32)
    ny, nx, ir, jr = _fire_shapes(tracer, rho, dz8w, burnt, fuel, refinement)
    field = np.ndim(fire_tracer_smoke) > 0
    fts = (np.asarray(fire_tracer_smoke, dtype=np.float32) if field
           else F(fire_tracer_smoke))
    if field and fts.shape != burnt.shape:
        raise ValueError("a fire_tracer_smoke field must be on the fire grid")
    out = tracer.copy()
    if nx < 3 or ny < 3:
        return out
    avgw = F(1.0) / F(ir * jr)
    inner = (slice(1, ny - 1), slice(1, nx - 1))
    plane = out[0][inner]
    denom = rho[inner] * dz8w[inner]
    for joff in range(jr):
        for ioff in range(ir):
            cells = (slice(jr + joff, (ny - 1) * jr, jr),
                     slice(ir + ioff, (nx - 1) * ir, ir))
            factor = fts[cells] if field else fts
            emis = avgw * factor
            emis = emis * burnt[cells]
            emis = emis * fuel[cells]
            emis = emis * F(1000.0)
            emis = emis / denom
            plane = plane + emis
    out[0][inner] = plane
    return out


def fire_tracer_emissions(tracer, rho, dz8w, burnt_area_dt, fgip,
                          fire_tracer_smoke=WRF_FIRE_TRACER_SMOKE, *,
                          refinement=(1, 1)):
    """:func:`fire_tracer_emissions_np` on the device, one kernel launch.

    CuPy arrays in (anything array-like is moved there), a new CuPy tracer
    out; the input is not modified.
    """
    import cupy as cp

    from woof.smoke import _kernels

    def f32(a):
        return cp.ascontiguousarray(cp.asarray(a, dtype=cp.float32))

    tracer = f32(tracer)
    rho, dz8w = (f32(_level1(cp.asarray(a))) for a in (rho, dz8w))
    burnt, fuel = f32(burnt_area_dt), f32(fgip)
    ny, nx, ir, jr = _fire_shapes(tracer, rho, dz8w, burnt, fuel, refinement)
    field = np.ndim(fire_tracer_smoke) > 0
    factor = f32(fire_tracer_smoke) if field else None
    if field and factor.shape != burnt.shape:
        raise ValueError("a fire_tracer_smoke field must be on the fire grid")
    scalar = F(0.0) if field else F(fire_tracer_smoke)
    out = tracer.copy()
    plane = out[0]
    _kernels.kernel("smoke_fire_tracer_emissions")(
        _kernels.grid(ny * nx), (_kernels.BLOCK,),
        (plane, rho, dz8w, burnt, fuel,
         factor if field else np.uintp(0), scalar, np.int32(nx),
         np.int32(ny), np.int32(ir), np.int32(jr)))
    return out
