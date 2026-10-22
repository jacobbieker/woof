"""Smoke source physics: fire emission, injection height and dry deposition.

A stand-alone library of array-in/array-out functions, not wired into the
dycore.  A fire model supplies the burned fuel mass rate (and its heat
flux); these functions turn that into a PM2.5 source distributed over a
model column, and remove the tracer again by gravitational settling and
dry deposition at the surface:

* :mod:`woof.smoke.emission` -- PM2.5 emission factors by fuel (Anderson
  13 categories and broad classes), and the emission flux
  ``kg m-2 s-1`` from the burned fuel mass rate;
* :mod:`woof.smoke.plume_rise` -- injection-height profiles onto a column
  of model interface heights (lowest layer, uniform over a fraction of the
  PBL, and a Freitas-style 1-D plume-rise estimate), each normalised so the
  column integral is the surface flux;
* :mod:`woof.smoke.deposition` -- the Zhang et al. (2001) resistance
  model for the dry deposition velocity, gravitational settling, and the
  column tendencies they put on a tracer mixing ratio.

THE CONVENTIONS, PINNED.  Vertical is axis 0, as in the model's
``(nz, ny, nx)`` fields; any trailing shape is a set of independent
columns.  Interface (w-level) heights ``z_w`` are metres above ground,
shape ``(nz+1, ...)``, strictly increasing, ``z_w[0]`` the surface.
Mass-level quantities are ``(nz, ...)``.  SI units throughout.

Every physics function has a float64 NumPy reference (suffix ``_np``,
written column by column in scalar arithmetic so it reads like the
equations) and an array-module form (no suffix) that runs the same
discretisation vectorised over columns on NumPy or CuPy arrays, in the
input's own precision.  ``tests/test_smoke_physics.py`` holds the two to
each other.  CuPy is never imported here: the array module is the one the
caller's arrays already belong to, and the device forms are plain CuPy
array maths, so no kernel is compiled on this library's behalf.

The vectorised plume-rise estimate is not re-exported here, because its
name is its module's: call it as ``woof.smoke.plume_rise.plume_rise``.
"""
from __future__ import annotations

from woof.smoke.deposition import (DEPOSITION_SURFACES, DepositionSurface,
                                   DepositionVelocity,
                                   ColumnTendency, deposited_fraction,
                                   dry_deposition_velocity,
                                   dry_deposition_velocity_np,
                                   settling_tendency, settling_tendency_np,
                                   settling_velocity, settling_velocity_np)
from woof.smoke.emission import (ANDERSON_FUEL_CLASS, EMISSION_FACTORS,
                                 FUEL_HEAT_CONTENT, NO_FUEL, EmissionFactor,
                                 burn_rate_from_heat_flux, emission_factor,
                                 emission_factor_table, pm25_emission_flux,
                                 pm25_emission_flux_np)
from woof.smoke.plume_rise import (PlumeRiseParameters, PlumeRiseResult,
                                   lowest_layer_profile,
                                   lowest_layer_profile_np,
                                   pbl_uniform_profile, pbl_uniform_profile_np,
                                   plume_rise_np,
                                   plume_rise_profile, plume_rise_profile_np,
                                   uniform_profile, uniform_profile_np)

__all__ = [
    "ANDERSON_FUEL_CLASS", "ColumnTendency", "DEPOSITION_SURFACES",
    "DepositionSurface", "DepositionVelocity", "EMISSION_FACTORS",
    "EmissionFactor",
    "FUEL_HEAT_CONTENT", "NO_FUEL", "PlumeRiseParameters", "PlumeRiseResult",
    "burn_rate_from_heat_flux", "deposited_fraction",
    "dry_deposition_velocity", "dry_deposition_velocity_np",
    "emission_factor", "emission_factor_table", "lowest_layer_profile",
    "lowest_layer_profile_np", "pbl_uniform_profile",
    "pbl_uniform_profile_np", "plume_rise_np",
    "plume_rise_profile", "plume_rise_profile_np", "pm25_emission_flux",
    "pm25_emission_flux_np", "settling_tendency", "settling_tendency_np",
    "settling_velocity", "settling_velocity_np", "uniform_profile",
    "uniform_profile_np",
]
