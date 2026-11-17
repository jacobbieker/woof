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


def _memo(cache, key, make):
    """``cache[key]``, made once; ``cache=None`` makes it every time."""
    if cache is None:
        return make()
    if key not in cache:
        cache[key] = make()
    return cache[key]


def molecular(col: atmosphere.Column, rep, iw: int, lam_nm: float | None = None,
              *, co2_rayleigh=None, cache: dict | None = None) -> MolecularOptics:
    """Molecular optical depths of ``col`` at representative wavelength ``iw``.

    ``cache`` (a dict the caller keeps for one ``col``, unchanged between
    calls) holds the wavelength-independent parts across calls: the layer
    air and gas columns, the H2O mixing ratio and the REPTRAN pressure,
    temperature and vmr brackets (one per lookup grid, shared by every
    species on it).  Each is the same expression on the same arrays, so the
    words are those of the uncached call.
    """
    lam = float(rep.wvl[iw]) if lam_nm is None else float(lam_nm)
    co2 = (_memo(cache, "co2", lambda: col.vmr("CO2")[..., -1]) if co2_rayleigh is None
           else np.asarray(co2_rayleigh))
    sig_r = rayleigh.cross_section(lam, co2)
    # the air factors are shared by the air column and every gas column
    air_f = _memo(cache, "air_f", lambda: atmosphere.layer_air(col.z, col.n_air))
    air = _memo(cache, "air", lambda: atmosphere.layer_column(col.z, col.n_air, air=air_f))

    def gas_column(name, dens):
        return _memo(cache, ("gas", name), lambda: atmosphere.layer_column(col.z, col.n_air, dens(), air=air_f))

    dtau_ray = np.asarray(sig_r)[..., None] * air
    chi2 = rayleigh.second_moment(lam, co2)
    by = {}
    for sp in SPECIES:
        if sp not in col.dens or not rep.needs_lookup(iw, sp):
            continue
        lk = rep.lookup(sp)
        if not lk.has(iw):
            continue
        h2o_vmr = _memo(cache, "h2o_vmr", lambda: col.vmr("H2O")) if sp == "H2O" else None
        br = _memo(cache, ("bracket", lk.bracket_key(h2o_vmr is not None)),
                   lambda: lk.bracket(col.p, col.t, h2o_vmr))
        sig = lk.sigma_at(iw, br)
        by[sp] = _mean_levels(sig) * gas_column(sp, lambda: col.dens[sp])
    # tabulated cross sections where the lookup has none
    if "O3" not in by:
        s = crs.o3(lam, col.t)
        if np.any(s):
            by["O3"] = _mean_levels(s) * gas_column("O3", lambda: col.dens["O3"])
    s = crs.no2(lam)
    if s and "NO2" in col.dens:
        by["NO2"] = s * gas_column("NO2", lambda: col.dens["NO2"])
    s = crs.o4(lam)
    if s:
        by["O4"] = s * gas_column("O4", lambda: col.dens["O2"] ** 2)
    dtau_abs = sum(by.values()) if by else np.zeros_like(dtau_ray)
    return MolecularOptics(dtau_ray=dtau_ray, dtau_abs=dtau_abs, by_species=by, chi2=np.asarray(chi2))
