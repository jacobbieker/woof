"""woof.radtran as a model diagnostic (shared by the WRF and MPAS adapters).

The model hands over its columns on its own levels (interfaces, surface
first, SI units) and receives band-averaged spectral irradiances and
radiances per REPTRAN band:

* ``edir``, ``edn``, ``eup`` at the surface and ``eup_toa`` at the top of
  the atmosphere [W m-2 nm-1]: direct (horizontal), diffuse downward and
  upward irradiance;
* ``rad`` [W m-2 nm-1 sr-1] for every requested direction (umu, phi):
  upwelling directions (umu > 0) at the top of the atmosphere (what a
  satellite sees), downwelling directions (umu < 0) at the surface (what a
  ground radiometer sees); phi is relative to the sun's azimuth.

Column assembly (documented choice): the model's interface levels are the
radiative-transfer levels and its mass layers the RT layers; temperature and
water vapour at an interior interface are the means of the two adjacent
mass levels, at the surface and model top the adjacent mass level's value;
the interface pressure and altitude are the model's own.  Above the model
top the U.S. Standard (AFGL) atmosphere is spliced
(:func:`woof.radtran.atmosphere.model_columns`); O3, CH4, N2O, CO, NO2, O2,
N2 come from the AFGL profile's mixing ratios, CO2 from the run's value.
Cloud water and ice per mass layer become Hu & Stamnes and Fu (1996) cloud
layers, with the microphysics' effective radii where the scheme provides
them (clipped to the parameterisations' ranges) and 10 um / 30 um otherwise.

Aerosol (``aerosol``, off by default) becomes OPAC species layers
(:mod:`woof.radtran.aerosol`; documented mapping, docs/dev/radtran.md):

* ``"thompson"``: the Thompson & Eidhammer (2014) aerosol-aware
  microphysics' number concentrations [kg-1] times the dry-air density give
  numbers per m3; water-friendly aerosol (nwfa: sulfates and organics)
  becomes OPAC water-soluble (waso) and ice-friendly aerosol (nifa: mineral
  dust) OPAC mineral, accumulation mode (miam), each at OPAC's mean particle
  mass (waso at its 50 % humidity node, the mass basis of uvspec's species
  profiles; miam dry).  Model layers only; none above the model top.
* an OPAC mixture name (``continental_average``, ``maritime_clean``,
  ``urban``, ...): libRadtran's OPAC mixture profile (Hess et al. 1998,
  masses at 50 % RH) by height above the model surface, on every layer
  including the spliced standard atmosphere.

The humidity node of each layer follows from the column's own temperature
and water vapour (uvspec's rule, :func:`woof.radtran.aerosol.select_nodes`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from woof.radtran import aerosol as opac
from woof.radtran import atmosphere, driver
from woof.radtran.options import Options

#: Molar-mass ratio dry air / water vapour (for qv -> volume mixing ratio).
AIR_OVER_H2O = 28.9644 / 18.01528
#: Effective radii used where the microphysics provides none [um].
DEFAULT_REFF_WATER = 10.0
DEFAULT_REFF_ICE = 30.0
#: Parameterisation ranges [um]: Hu & Stamnes 2.5-60; Fu (1996) with the
#: Yang definition maps r -> D = 2 r 4/(3 sqrt 3), valid for D 9.3-65 um.
REFF_WATER_RANGE = (2.5, 60.0)
REFF_ICE_RANGE = (6.06, 42.3)
#: radtran aerosol settings: off, Thompson aerosol-aware numbers, or an OPAC mixture
AEROSOL_THOMPSON = "thompson"
AEROSOL_CHOICES = ("none", AEROSOL_THOMPSON) + opac.MIXTURES
#: Thompson aerosol -> OPAC species (number -> mass at this humidity node)
THOMPSON_SPECIES = (("waso", 50.0), ("miam", 0.0))


@dataclass(frozen=True)
class Diagnostic:
    """The run's radtran settings (from RunConfig / the MPAS namelist)."""
    wavelengths: tuple
    interval_s: float = 3600.0
    umu: tuple = ()
    phi: tuple = ()
    mol_abs_param: str = "reptran coarse"
    solver: str = "disort"
    streams: int = 0
    pseudospherical: bool = False
    co2_vmr: float = 4.0e-4
    aerosol: str = "none"

    @property
    def enabled(self) -> bool:
        return bool(self.wavelengths)

    def options(self, day_of_year: int | None = None) -> Options:
        nstr = None if self.streams in (0, None) else int(self.streams)
        return Options(wavelength=(min(self.wavelengths), max(self.wavelengths)),
                       mol_abs_param=self.mol_abs_param, rte_solver=self.solver,
                       number_of_streams=nstr, umu=tuple(self.umu), phi=tuple(self.phi),
                       pseudospherical=bool(self.pseudospherical), zout=("sur", "toa"),
                       day_of_year=day_of_year).validate()

    def directions(self) -> list[tuple[float, float]]:
        phis = tuple(self.phi) if self.phi else (0.0,)
        return [(float(u), float(p)) for u in self.umu for p in phis]


def diagnostic_from_config(cfg) -> Diagnostic:
    """The radtran settings of a woof RunConfig (WRF side)."""
    return Diagnostic(
        wavelengths=tuple(float(w) for w in getattr(cfg, "radtran_wavelengths", ()) or ()),
        interval_s=float(getattr(cfg, "radtran_interval_s", 3600.0)),
        umu=tuple(float(u) for u in getattr(cfg, "radtran_umu", ()) or ()),
        phi=tuple(float(p) for p in getattr(cfg, "radtran_phi", ()) or ()),
        mol_abs_param=str(getattr(cfg, "radtran_mol_abs_param", "reptran coarse")),
        solver=str(getattr(cfg, "radtran_solver", "disort")),
        streams=int(getattr(cfg, "radtran_streams", 0) or 0),
        pseudospherical=bool(getattr(cfg, "radtran_pseudospherical", False)),
        aerosol=str(getattr(cfg, "radtran_aerosol", "none")))


def validate(diag: Diagnostic) -> None:
    """Refuse a configuration the solver cannot run (config-time check)."""
    if not diag.enabled:
        return
    if any((not math.isfinite(w)) or w < 120.0 or w > 5000.0 for w in diag.wavelengths):
        raise ValueError("radtran_wavelengths must lie within REPTRAN's solar range, 120-5000 nm")
    if not (diag.interval_s > 0 and math.isfinite(diag.interval_s)):
        raise ValueError("radtran_interval_s must be a positive number of seconds")
    if diag.solver not in ("disort", "twostr"):
        raise ValueError("radtran_solver must be \"disort\" or \"twostr\"")
    if diag.solver == "twostr" and diag.umu:
        raise ValueError("radtran_umu needs radtran_solver = \"disort\" (two-stream gives irradiances only)")
    if diag.phi and not diag.umu:
        raise ValueError("radtran_phi needs radtran_umu")
    if any(u == 0.0 or abs(u) > 1.0 for u in diag.umu):
        raise ValueError("radtran_umu entries must be cosines in [-1, 0) or (0, 1]")
    parts = diag.mol_abs_param.split()
    if parts[:1] != ["reptran"] or (len(parts) > 1 and parts[1] not in ("coarse", "medium", "fine")):
        raise ValueError("radtran_mol_abs_param must be \"reptran coarse|medium|fine\"")
    if diag.streams not in (0, None) and (int(diag.streams) % 2 or not 2 <= int(diag.streams) <= 32):
        raise ValueError("radtran_streams must be 0 (default) or even, 2..32")
    if diag.aerosol not in AEROSOL_CHOICES:
        raise ValueError("radtran_aerosol must be \"none\", \"thompson\" or an OPAC mixture ("
                         + ", ".join(opac.MIXTURES) + ")")
    if diag.aerosol != "none":
        lo = opac.LAM_MIN_UM * 1e3
        if any(w < lo for w in diag.wavelengths):
            raise ValueError(f"radtran_aerosol: the OPAC tables start at {lo:.0f} nm")
    diag.options()


def due(elapsed_s: float, dt: float, interval_s: float, start_s: float = 0.0) -> bool:
    """True on the first step and on every step that crosses a multiple of the
    interval (the model's hail-diagnostic rule, woof/core/hail.py)."""
    start = elapsed_s - start_s
    end = start + dt
    return start <= 0.0 or int(end / interval_s) > int(start / interval_s)


def band_table(diag: Diagnostic):
    """(band centre, wvlmin, wvlmax) [nm] of the band containing each requested
    wavelength, in the order requested (reads the REPTRAN band file)."""
    from woof.radtran import data
    from woof.radtran.reptran import open_reptran
    rep = open_reptran(str(data.reptran_file(diag.options().reptran_resolution)))
    b = np.array([rep.band_containing(float(w)) for w in diag.wavelengths], dtype=np.int64)
    lo, hi = rep.wvlmin[b], rep.wvlmax[b]
    return 0.5 * (lo + hi), lo.copy(), hi.copy()


def _interfaces(mass):
    """(ncol, nz) mass-level values -> (ncol, nz+1) interface values, surface first."""
    out = np.empty((mass.shape[0], mass.shape[1] + 1))
    out[:, 1:-1] = 0.5 * (mass[:, :-1] + mass[:, 1:])
    out[:, 0] = mass[:, 0]
    out[:, -1] = mass[:, -1]
    return out


def aerosol_layers(diag: Diagnostic, col, nz: int, *, rho_dry=None, nwfa=None, nifa=None):
    """OPAC species layers (top first, every RT layer) for the run's setting."""
    if diag.aerosol == "none":
        return None
    ncol, nlev = col.z.shape
    L = nlev - 1
    if diag.aerosol == AEROSOL_THOMPSON:
        if nwfa is None or nifa is None or rho_dry is None:
            raise ValueError("radtran_aerosol = \"thompson\" needs the aerosol-aware Thompson "
                             "microphysics' nwfa and nifa")
        mass = np.zeros((ncol, L, len(THOMPSON_SPECIES)))
        rho = np.asarray(rho_dry, dtype=np.float64)
        for k, (num, (sp, rh)) in enumerate(zip((nwfa, nifa), THOMPSON_SPECIES)):
            n = np.maximum(np.asarray(num, dtype=np.float64), 0.0) * rho        # m-3
            mass[:, L - nz:, k] = (n * opac.particle_mass(sp, rh))[:, ::-1]       # g m-3
        return opac.AerosolLayers(tuple(s for s, _ in THOMPSON_SPECIES), mass)
    zb = col.z[:, 1:] - col.z[:, -1:]
    return opac.mixture_layers(diag.aerosol, zb)


def compute(diag: Diagnostic, *, z_w_m, p_w_pa, t_mass, qv_mass, umu0, albedo,
            qc_mass=None, qi_mass=None, rho_dry=None, re_cloud_um=None, re_ice_um=None,
            nwfa=None, nifa=None, day_of_year: int | None = None, backend: str = "gpu") -> dict:
    """Spectral diagnostic on (ncol, ...) host arrays, surface first.

    z_w_m, p_w_pa: (ncol, nz+1) interface altitude above sea level [m] and
    pressure [Pa]; t_mass [K], qv_mass [kg/kg]: (ncol, nz); qc/qi [kg/kg]
    and rho_dry [kg m-3] per mass layer for clouds; nwfa/nifa [kg-1] per
    mass layer for ``aerosol = "thompson"``; umu0, albedo: (ncol,).
    Returns arrays shaped (nband, ncol) and (nband, ndir, ncol).
    """
    opts = diag.options(day_of_year)
    z = np.asarray(z_w_m, dtype=np.float64)[:, ::-1] * 1e-3            # top first, km
    p = np.asarray(p_w_pa, dtype=np.float64)[:, ::-1] * 1e-2
    t = _interfaces(np.asarray(t_mass, dtype=np.float64))[:, ::-1]
    qv = _interfaces(np.maximum(np.asarray(qv_mass, dtype=np.float64), 0.0))[:, ::-1]
    col = atmosphere.model_columns(z, p, t, h2o_vmr=qv * AIR_OVER_H2O, co2_vmr=diag.co2_vmr)
    ncol, nlev = col.z.shape
    nz = z.shape[1] - 1
    L = nlev - 1
    clouds = None
    if (qc_mass is not None or qi_mass is not None) and rho_dry is not None:
        rho = np.asarray(rho_dry, dtype=np.float64)
        pad = lambda a: np.concatenate((np.zeros((ncol, L - nz)), a[:, ::-1]), axis=1)
        kw = {}
        if qc_mass is not None:
            lwc = np.maximum(np.asarray(qc_mass, dtype=np.float64), 0.0) * rho * 1e3   # g m-3
            re = (np.full(lwc.shape, DEFAULT_REFF_WATER) if re_cloud_um is None
                  else np.where(np.asarray(re_cloud_um) > 0, re_cloud_um, DEFAULT_REFF_WATER))
            kw.update(lwc=pad(np.where(lwc > 1e-8, lwc, 0.0)), reff_wc=pad(np.clip(re, *REFF_WATER_RANGE)))
        if qi_mass is not None:
            iwc = np.maximum(np.asarray(qi_mass, dtype=np.float64), 0.0) * rho * 1e3
            re = (np.full(iwc.shape, DEFAULT_REFF_ICE) if re_ice_um is None
                  else np.where(np.asarray(re_ice_um) > 0, re_ice_um, DEFAULT_REFF_ICE))
            kw.update(iwc=pad(np.where(iwc > 1e-8, iwc, 0.0)), reff_ic=pad(np.clip(re, *REFF_ICE_RANGE)))
        clouds = driver.CloudLayers(**kw)
    aer = aerosol_layers(diag, col, nz, rho_dry=rho_dry, nwfa=nwfa, nifa=nifa)
    res = driver.compute(opts, col, clouds, aerosol=aer, backend=backend, umu0=umu0, albedo=albedo,
                         wavelengths=diag.wavelengths)
    mw = 1e-3                                                            # mW -> W
    # one row per requested wavelength: the band containing it
    centres = res.wavelength
    idx = np.array([int(np.nonzero((res.wvlmin <= w) & (res.wvlmax > w))[0][0]) for w in diag.wavelengths])
    out = {"wavelength": np.asarray(diag.wavelengths, dtype=np.float64), "band_centre": centres[idx],
           "band_min": res.wvlmin[idx], "band_max": res.wvlmax[idx],
           "edir": res.edir[:, 0, idx].T * mw, "edn": res.edn[:, 0, idx].T * mw,
           "eup": res.eup[:, 0, idx].T * mw, "eup_toa": res.eup[:, 1, idx].T * mw}
    if diag.umu:
        dirs = diag.directions()
        nphi = len(diag.phi) if diag.phi else 1
        rad = np.zeros((idx.size, len(dirs), ncol))
        for k, (u, _) in enumerate(dirs):
            iu, ip = divmod(k, nphi)
            lev = 1 if u > 0 else 0                                       # TOA up, surface down
            rad[:, k, :] = res.uu[:, lev, iu, ip, idx].T * mw
        out["rad"] = rad
    return out
