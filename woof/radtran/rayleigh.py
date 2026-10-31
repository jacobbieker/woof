"""Rayleigh scattering of dry air after Bodhaine, Wood, Dutton & Slusser (1999).

J. Atmos. Oceanic Technol. 16, 1854-1861: the refractive index of dry air
with 300 ppm CO2 (Peck & Reeder 1972 form, their eq. 18) scaled to the CO2
mixing ratio (their eq. 19), the King factors of N2, O2, Ar and CO2 combined
by volume (their eqs. 22-23), and the cross section

    sigma = 24 pi^3 (n^2 - 1)^2 / (lambda^4 Ns^2 (n^2 + 2)^2) F_air

with Ns the molecular number density at 288.15 K and 1013.25 hPa.  The
depolarisation ratio follows from F_air = (6 + 3 rho) / (6 - 7 rho), and the
Rayleigh phase function p(mu) = a + b mu^2 with
a = 1.5 (1 + rho) / (2 + rho), b = 1.5 (1 - rho) / (2 + rho) has the single
non-trivial Legendre moment chi_2 = (1 - rho) / (5 (2 + rho)).
"""

from __future__ import annotations

import math

import numpy as np

#: Molecular number density of air at 288.15 K and 1013.25 hPa [cm^-3]
#: (Bodhaine et al. 1999, Table 1 note).
NS_CM3 = 2.546899e19


def refractive_index_minus_one(lam_um, co2_vmr):
    """(n - 1) of dry air at 288.15 K, 1013.25 hPa for CO2 volume mixing ratio."""
    l2 = 1.0 / (np.asarray(lam_um, dtype=np.float64) ** 2)
    n300 = (8060.51 + 2480990.0 / (132.274 - l2) + 17455.7 / (39.32957 - l2)) * 1e-8
    return n300 * (1.0 + 0.54 * (co2_vmr - 0.0003))


def king_factor(lam_um, co2_vmr):
    l2 = 1.0 / (np.asarray(lam_um, dtype=np.float64) ** 2)
    f_n2 = 1.034 + 3.17e-4 * l2
    f_o2 = 1.096 + 1.385e-3 * l2 + 1.448e-4 * l2 * l2
    f_ar = 1.00
    f_co2 = 1.15
    c = co2_vmr * 100.0                       # percent by volume
    return ((78.084 * f_n2 + 20.946 * f_o2 + 0.934 * f_ar + c * f_co2)
            / (78.084 + 20.946 + 0.934 + c))


def cross_section(lam_nm, co2_vmr):
    """Rayleigh scattering cross section per molecule [cm^2]."""
    lam_um = np.asarray(lam_nm, dtype=np.float64) * 1e-3
    n = refractive_index_minus_one(lam_um, co2_vmr) + 1.0
    lam_cm = lam_um * 1e-4
    n2 = n * n
    return (24.0 * math.pi ** 3 * (n2 - 1.0) ** 2
            / (lam_cm ** 4 * NS_CM3 ** 2 * (n2 + 2.0) ** 2) * king_factor(lam_um, co2_vmr))


def depolarization(lam_nm, co2_vmr):
    f = king_factor(np.asarray(lam_nm, dtype=np.float64) * 1e-3, co2_vmr)
    return 6.0 * (f - 1.0) / (3.0 + 7.0 * f)


def second_moment(lam_nm, co2_vmr):
    rho = depolarization(lam_nm, co2_vmr)
    return (1.0 - rho) / (5.0 * (2.0 + rho))
