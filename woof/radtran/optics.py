"""Layer optical properties: molecules, clouds, aerosol, combined for the solver.

Molecular part (per layer, top first):

* Rayleigh: sigma(lambda, CO2) of Bodhaine et al. (1999) times the air
  column (:func:`woof.radtran.atmosphere.layer_column`); the CO2 mixing ratio
  is the column's value at its lowest level.
* Absorption: for every species, the layer cross section is the mean of the
  cross sections at the layer's two levels (REPTRAN lookup or tabulated crs
  file, :mod:`woof.radtran.reptran`, :mod:`woof.radtran.crs`) times the gas
  column; O4 uses the column of n(O2)^2 (as the gas density) instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from woof.radtran import atmosphere, crs, rayleigh
from woof.radtran.reptran import SPECIES


@dataclass
class MolecularOptics:
    dtau_ray: np.ndarray     # (..., nlyr)
    dtau_abs: np.ndarray     # (..., nlyr)
    by_species: dict         # species -> (..., nlyr)
    chi2: np.ndarray         # (...,) Rayleigh second Legendre moment


def _mean_levels(x):
    return 0.5 * (x[..., :-1] + x[..., 1:])


def molecular(col: atmosphere.Column, rep, iw: int, lam_nm: float | None = None,
              *, co2_rayleigh=None) -> MolecularOptics:
    """Molecular optical depths of ``col`` at representative wavelength ``iw``."""
    lam = float(rep.wvl[iw]) if lam_nm is None else float(lam_nm)
    co2 = (col.vmr("CO2")[..., -1] if co2_rayleigh is None else np.asarray(co2_rayleigh))
    sig_r = rayleigh.cross_section(lam, co2)
    air = atmosphere.layer_column(col.z, col.n_air)
    dtau_ray = np.asarray(sig_r)[..., None] * air
    chi2 = rayleigh.second_moment(lam, co2)
    by = {}
    h2o_vmr = col.vmr("H2O")
    for sp in SPECIES:
        if sp not in col.dens or not rep.needs_lookup(iw, sp):
            continue
        lk = rep.lookup(sp)
        if not lk.has(iw):
            continue
        sig = lk.sigma(iw, col.p, col.t, h2o_vmr if sp == "H2O" else None)
        by[sp] = _mean_levels(sig) * atmosphere.layer_column(col.z, col.n_air, col.dens[sp])
    # tabulated cross sections where the lookup has none
    if "O3" not in by:
        s = crs.o3(lam, col.t)
        if np.any(s):
            by["O3"] = _mean_levels(s) * atmosphere.layer_column(col.z, col.n_air, col.dens["O3"])
    s = crs.no2(lam)
    if s and "NO2" in col.dens:
        by["NO2"] = s * atmosphere.layer_column(col.z, col.n_air, col.dens["NO2"])
    s = crs.o4(lam)
    if s:
        by["O4"] = s * atmosphere.layer_column(col.z, col.n_air, col.dens["O2"] ** 2)
    dtau_abs = sum(by.values()) if by else np.zeros_like(dtau_ray)
    return MolecularOptics(dtau_ray=dtau_ray, dtau_abs=dtau_abs, by_species=by, chi2=np.asarray(chi2))
