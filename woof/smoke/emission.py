"""PM2.5 emission from burning fuel: emission factors and the surface flux.

THE APPROACH.  An emission factor EF (mass of species emitted per mass of
dry fuel consumed) times the fuel consumption rate is the emission rate.
That is how both fire-emission routes in the WRF family form a smoke
source: WRF-SFIRE multiplies the fuel mass burned in each fire-grid cell
per time step by a per-tracer factor (``fire_tracer_smoke``; Mandel et al.
2011, GMD 4, 591-610) and the WRF-SFIRE-CHEM coupling multiplies it by
per-species, per-fuel-category factors before passing the flux to the
chemistry emission arrays (Kochanski et al. 2016, GMD 9, 2853-2875);
WRF-Chem's biomass-burning preprocessor does the same with burned biomass
from satellite fire products and the Andreae and Merlet (2001) factors
(Freitas et al. 2011, GMD 4, 419-433)::

    E_PM2.5 [kg m-2 s-1] = EF_PM2.5 [kg kg-1] * m_dot [kg m-2 s-1]

with ``m_dot`` the dry fuel mass burned per unit area per second
(:func:`pm25_emission_flux`).

THE FACTORS.  Four broad fuel classes, each one rounded central value in
g of PM2.5 per kg of dry matter burned (:data:`EMISSION_FACTORS`):

========  =====  ==========================================================
class     EF     basis
========  =====  ==========================================================
grass     8.0    savanna and grassland (Andreae 2019, ACP 19, 8523-8546,
                 Table 1; Andreae and Merlet 2001, GBC 15, 955-966), and
                 the US grassland/rangeland values of Urbanski (2014, For.
                 Ecol. Manage. 317, 51-60)
shrub     9.0    chaparral and other shrubland fires (Urbanski 2014, the
                 south-western shrubland values; Andreae 2019 sits between
                 its grassland and forest categories)
timber    15.0   temperate and boreal forest (Andreae 2019, Table 1; the
                 western-US forest wildfire and prescribed fire values of
                 Urbanski 2014 bracket it)
slash     13.0   logging residue and piled slash (Urbanski 2014, the
                 prescribed fire values for activity fuels)
========  =====  ==========================================================

These are representative values chosen inside the ranges those
compilations report, not a transcription of any one table entry; the
compiled studies scatter by a factor of about two for every class (fire
phase, fuel moisture and measurement method all move the factor).  A
caller with a better local factor passes its own table
(:func:`emission_factor_table`).

The thirteen Anderson (1982, USDA INT-122) fire-behaviour fuel models map
onto the classes by Anderson's own groups (:data:`ANDERSON_FUEL_CLASS`):
grass 1-3, shrub (chaparral and brush) 4-7, timber litter 8-10 and logging
slash 11-13.  Category 14 is WRF-SFIRE's "no fuel" entry
(``namelist.fire``), whose factor is zero.

THE HEAT FLUX, WHEN THAT IS WHAT A FIRE MODEL GIVES.  A model that reports
the fire's heat flux instead of the fuel consumed recovers the latter with
the fuel heat content: ``m_dot = Q / h`` (:func:`burn_rate_from_heat_flux`)
with ``h = 8000 Btu/lb = 18.6 MJ kg-1``, the low heat content of
Rothermel (1972) and of the Anderson fuel models (:data:`FUEL_HEAT_CONTENT`).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

#: Low heat content of the Anderson fuel models, 8000 Btu/lb (J kg-1).
FUEL_HEAT_CONTENT = 8000.0 * 1055.05585 / 0.45359237

#: WRF-SFIRE's "no fuel" category (namelist.fire), emission factor zero.
NO_FUEL = 14


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
    "slash": EmissionFactor(
        "slash", 13.0, "Urbanski (2014) activity fuels"),
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


def array_module(*arrays: Any):
    """``cupy`` when any input is a CuPy array, else ``numpy``.

    CuPy is never imported here: an input can only be a CuPy array if the
    caller already imported it, so ``sys.modules`` is the whole question.
    (The ``woof.energy.icing_turbine`` idiom, restated so this package
    imports nothing from the energy products.)
    """
    cupy = sys.modules.get("cupy")
    if cupy is None:
        return np
    return cupy.get_array_module(*[a for a in arrays if a is not None])


def emission_factor(fuel: int | str,
                    factors: Mapping[str, EmissionFactor] = EMISSION_FACTORS
                    ) -> float:
    """PM2.5 emission factor (kg kg-1) of an Anderson category or a class.

    ``fuel`` is an Anderson fuel model number (1-13, or 14 for no fuel) or
    a class name from ``factors``.  Anything else is refused by name.  No
    fuel emits nothing whether or not ``factors`` lists it.
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
    """Factors (kg kg-1) indexed by Anderson category, float64, length 15.

    Entry 0 is NaN (there is no category 0), and so is every category
    whose class ``factors`` does not list, so a lookup with an unvalidated
    index or an incomplete table cannot quietly emit nothing.
    """
    table = np.full(NO_FUEL + 1, np.nan)
    for number, name in ANDERSON_FUEL_CLASS.items():
        if name in factors or name == "none":
            table[number] = emission_factor(number, factors)
    return table


def burn_rate_from_heat_flux(heat_flux: Any,
                             heat_content: float = FUEL_HEAT_CONTENT) -> Any:
    """Dry fuel burned (kg m-2 s-1) from the fire heat flux (W m-2).

    ``m_dot = Q / h``; ``heat_content`` in J kg-1.  Array in, array out, on
    the caller's array module.
    """
    if not heat_content > 0.0:
        raise ValueError(f"heat_content must be positive, got {heat_content}")
    return heat_flux / heat_content


def _validated_categories(fuel: Any, xp) -> Any:
    """Integer Anderson categories, every one in 1..14, else ValueError."""
    category = xp.asarray(fuel)
    if category.dtype.kind not in "iu":
        rounded = xp.rint(category)
        if bool(xp.any(rounded != category)):
            raise ValueError("fuel categories must be whole numbers")
        category = rounded.astype(np.int64)
    if category.size and (bool(xp.any(category < 1))
                          or bool(xp.any(category > NO_FUEL))):
        raise ValueError(f"fuel categories must be Anderson models 1-13 or "
                         f"the no-fuel category {NO_FUEL}")
    return category


def pm25_emission_flux_np(burn_rate: Any, fuel: Any,
                          factors: Mapping[str, EmissionFactor]
                          = EMISSION_FACTORS) -> np.ndarray:
    """Reference PM2.5 emission flux (kg m-2 s-1), float64, cell by cell.

    ``burn_rate`` is the dry fuel mass burned (kg m-2 s-1, >= 0); ``fuel``
    is an Anderson category array broadcastable against it, or one class
    name or category for every cell.
    """
    rate = np.asarray(burn_rate, dtype=np.float64)
    if not np.all(np.isfinite(rate)) or np.any(rate < 0.0):
        raise ValueError("burn_rate must be finite and non-negative")
    if isinstance(fuel, str):
        return rate * emission_factor(fuel, factors)
    category = _validated_categories(fuel, np)
    rate, category = np.broadcast_arrays(rate, category)
    out = np.empty(rate.shape)
    for index in np.ndindex(rate.shape):
        out[index] = rate[index] * emission_factor(int(category[index]),
                                                   factors)
    return out


def pm25_emission_flux(burn_rate: Any, fuel: Any,
                       factors: Mapping[str, EmissionFactor]
                       = EMISSION_FACTORS) -> Any:
    """PM2.5 emission flux (kg m-2 s-1) on the caller's array module.

    Same contract as :func:`pm25_emission_flux_np`, vectorised: the factor
    table is gathered by category on the device, and the flux keeps the
    burn rate's floating dtype.
    """
    xp = array_module(burn_rate, None if isinstance(fuel, str) else fuel)
    rate = xp.asarray(burn_rate)
    if rate.dtype.kind != "f":
        rate = rate.astype(np.float64)
    if (not bool(xp.all(xp.isfinite(rate)))) or bool(xp.any(rate < 0.0)):
        raise ValueError("burn_rate must be finite and non-negative")
    if isinstance(fuel, str):
        return rate * rate.dtype.type(emission_factor(fuel, factors))
    category = _validated_categories(fuel, xp)
    host = emission_factor_table(factors)
    if np.isnan(host[1:]).any():
        # An incomplete table: the reference's refusal, by name, for a
        # category whose class it lacks (one host copy, only in this case).
        present = category.get() if xp is not np else category
        for number in np.unique(present):
            if np.isnan(host[number]):
                emission_factor(int(number), factors)
    table = xp.asarray(host, dtype=rate.dtype)
    return rate * table[category]
