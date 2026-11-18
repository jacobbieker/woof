"""L2: ionisation of the middle atmosphere by solar protons.

During a solar proton event (SPE) protons of 1-500 MeV reach the polar caps
and ionise the stratosphere and mesosphere.  Every ion pair makes about 1.25
odd-nitrogen molecules (Porter et al. 1976, J. Chem. Phys. 65, 154) and, below
about 70 km, about 2 odd-hydrogen molecules (Solomon et al. 1981, Planet.
Space Sci. 29, 885); both destroy ozone.  This module computes the ion-pair
production rate from the proton flux, from first principles:

* the spectrum is an integral proton flux J(>E) (cm-2 s-1 sr-1): OMNI2's
  words 43-48 (>1, >2, >4, >10, >30 and >60 MeV, GOES since the 1990s; from
  about 2010 only >10, >30 and >60) or GOES in real time through NOAA SWPC
  (>1 to >500 MeV).  It is a power law between the channels present,
  extended past the last one with the last pair's slope but never flatter
  than J ~ E^-3 (:data:`TAIL_INDEX_MIN`, chosen against the CMIP7 ion-pair
  rates of the 2000-2005 events), up to :data:`E_MAX`;
* an hour counts as an event only while J(>10 MeV) is at least NOAA's S1
  threshold, 10 pfu (:data:`SPE_THRESHOLD_PFU`): below it the channels carry
  mostly galactic cosmic rays and detector background;
* the energy loss is the Bethe formula for protons in air (Z/A = 0.49919,
  I = 85.7 eV; PDG, "Passage of particles through matter"), whose range table
  gives each proton's residual energy after a slant column depth;
* the flux is isotropic over the downward hemisphere.  A layer between
  column depths X1 < X2 (g cm-2) takes, per unit area,
  ``2 pi int mu dmu int dE j(E) [E'(E, X1/mu) - E'(E, X2/mu)]``, which is
  exact energy bookkeeping (a proton stopping inside the layer leaves all of
  its residual energy there);
* one ion pair costs W = 35 eV (Porter et al. 1976);
* the protons reach the polar cap only: geomagnetic latitudes poleward of
  :data:`POLAR_CAP_DEG` (the open-field convention of Jackman et al. 2008,
  JGR 113, D17203), uniformly.

Auroral electrons (100 eV to 1 MeV) have their own ion-pair rates
(:func:`electron_absorbed`, :func:`electron_ion_pairs_per_gram`): the
parameterisation of Fang et al. 2010 (GRL 37, L22106), fitted to
first-principles transport models, for an isotropic monoenergetic or
Maxwellian flux such as OVATION Prime's energy flux and mean energy.  They
deposit at 70-150 km, above every WRF and MPAS model top, so they are a
diagnostic, not a source in the forecast.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

#: OMNI2 proton channels, MeV (words 43-48).
CHANNELS_MEV = (1.0, 2.0, 4.0, 10.0, 30.0, 60.0)
#: Energy per ion pair in air, MeV (35 eV).
W_ION_MEV = 35.0e-6
#: The spectrum's upper end, MeV.
E_MAX = 1000.0
#: Lowest energy of the range table, MeV.  A proton below it is taken as stopped.
E_MIN = 0.1
#: The least steep integral power-law index past the last channel.  OMNI2 stops
#: at 60 MeV, and the 30-60 MeV slope extended to 1 GeV makes the lower
#: stratosphere 5-40 times the CMIP7 rates (2000-2005 events); index 3 makes
#: them 0.6-1.0 times, and 0.4-0.6 times on the 2012-2017 events.
TAIL_INDEX_MIN = 3.0
#: Geomagnetic latitude of the polar cap's edge, degrees.
POLAR_CAP_DEG = 60.0
#: Odd nitrogen and odd hydrogen per ion pair.
NOX_PER_ION_PAIR = 1.25
HOX_PER_ION_PAIR = 2.0

_ME = 0.51099895        # electron rest energy, MeV
_MP = 938.27208816      # proton rest energy, MeV
_K = 0.307075           # 4 pi N_A r_e^2 m_e c^2, MeV cm2 mol-1
_Z_OVER_A = 0.49919     # dry air
_I_MEV = 85.7e-6        # dry air mean excitation energy
_G = 9.80665            # m s-2


class ParticleError(ValueError):
    pass


def stopping_power(energy_mev) -> np.ndarray:
    """Bethe mass stopping power of air for protons, MeV cm2 g-1 (no shell or
    density corrections: within a few per cent above 1 MeV and below 1 GeV)."""
    e = np.asarray(energy_mev, dtype=np.float64)
    gamma = 1.0 + e / _MP
    beta2 = 1.0 - 1.0 / (gamma * gamma)
    bg2 = beta2 * gamma * gamma
    ratio = _ME / _MP
    tmax = 2.0 * _ME * bg2 / (1.0 + 2.0 * gamma * ratio + ratio * ratio)
    log_term = 0.5 * np.log(2.0 * _ME * bg2 * tmax / (_I_MEV * _I_MEV))
    return _K * _Z_OVER_A / beta2 * np.maximum(log_term - beta2, 1.0e-3)


@dataclass(frozen=True)
class RangeTable:
    energy: np.ndarray      # MeV, ascending log grid from E_MIN to E_MAX
    range: np.ndarray       # g cm-2, the CSDA range at each energy

    def residual(self, energy, depth) -> np.ndarray:
        """Energy, MeV, left after a slant column depth (g cm-2); 0 once stopped."""
        r = np.interp(np.log(energy), np.log(self.energy), np.log(self.range))
        left = np.exp(r) - np.asarray(depth, dtype=np.float64)
        out = np.zeros(np.broadcast(left, energy).shape)
        ok = left > self.range[0]
        out[ok] = np.exp(np.interp(np.log(left[ok]), np.log(self.range), np.log(self.energy)))
        return out


def range_table(n: int = 400) -> RangeTable:
    """The CSDA range of protons in air, integrated from the Bethe formula
    (trapezoids in log E); below E_MIN the range is taken as E / S(E) / 2,
    the scaling of a stopping power falling as 1/E."""
    e = np.geomspace(E_MIN, E_MAX * 1.01, n)
    inv = 1.0 / stopping_power(e)
    le = np.log(e)
    seg = 0.5 * (inv[1:] * e[1:] + inv[:-1] * e[:-1]) * np.diff(le)
    r = np.concatenate([[0.5 * e[0] * inv[0]], 0.5 * e[0] * inv[0] + np.cumsum(seg)])
    return RangeTable(e, r)


def spectrum(integral_flux, channels=CHANNELS_MEV, *, e_max: float = E_MAX, n: int = 240,
             tail_index: float | None = None):
    """(energy edges, MeV; protons cm-2 s-1 sr-1 in each bin) from integral
    fluxes J(>E) at the channels (NaN where missing; at least two present):
    a power law between successive channels present, extended past the last
    one with the last pair's slope (never flatter than ``tail_index``), from
    the lowest channel present (nothing below it) to ``e_max``.  The last bin
    also holds every proton above ``e_max``: in the atmosphere a proton above
    1 GeV loses energy at close to the minimum-ionising rate, as one at 1 GeV
    does, so the tail is placed there rather than dropped."""
    tail_index = TAIL_INDEX_MIN if tail_index is None else tail_index
    j = np.asarray(integral_flux, dtype=np.float64)
    ch = np.asarray(channels, dtype=np.float64)
    if j.shape != ch.shape:
        raise ParticleError(f"{j.size} fluxes for {ch.size} channels")
    ok = np.isfinite(j) & (j > 0)
    if ok.sum() < 2:
        raise ParticleError("the proton spectrum needs at least two channels with a positive flux")
    le, lj = np.log(ch[ok]), np.log(j[ok])
    lj = np.minimum.accumulate(lj)         # J(>E) cannot rise with E: a noisy higher channel is held
    edges = np.geomspace(ch[ok][0], e_max, n + 1)
    lx = np.log(edges)
    last = min((lj[-1] - lj[-2]) / (le[-1] - le[-2]), -tail_index)
    big_j = np.exp(np.where(lx > le[-1], lj[-1] + last * (lx - le[-1]), np.interp(lx, le, lj)))
    big_j[-1] = 0.0                        # the last bin takes J(>edges[-2]): the tail above e_max too
    return edges, np.maximum(big_j[:-1] - big_j[1:], 0.0)


def _mu_nodes(n: int = 16):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w            # on (0, 1)


#: Column depths, g cm-2, of the absorption table: 1e-5 hPa to 1100 hPa.
DEPTH_GRID = np.geomspace(1.0e-5, 1.13e3, 241)


def absorbed(integral_flux, depth, channels=CHANNELS_MEV, table: RangeTable | None = None,
             n_mu: int = 16) -> np.ndarray:
    """Energy absorbed above each column depth (g cm-2), MeV cm-2 s-1, from a
    flux isotropic over the downward hemisphere: ``F(X) = 2 pi int mu dmu int
    dE j(E) [E - E'(E, X / mu)]``.  A layer's energy is the difference of F
    at its edges, so energy is conserved exactly."""
    table = table or range_table()
    x = np.asarray(depth, dtype=np.float64)
    edges, counts = spectrum(integral_flux, channels)
    e_mid = np.sqrt(edges[1:] * edges[:-1])
    mu, w = _mu_nodes(n_mu)
    out = np.zeros(x.shape)
    for m, wm in zip(mu, w):
        resid = table.residual(e_mid[:, None], x.ravel()[None, :] / m)     # (nE, nX)
        out += (2.0 * math.pi * wm * m * (counts @ (e_mid[:, None] - resid))).reshape(x.shape)
    return out


#: Fang et al. 2010 (GRL 37, L22106), Table 1: P_ij for isotropically incident
#: monoenergetic 100 eV to 1 MeV electrons; C_i(E) = exp(sum_j P_ij ln(E/keV)^j).
FANG2010_P = np.array([
    [1.24616e+0, 1.45903e+0, -2.42269e-1, 5.95459e-2],
    [2.23976e+0, -4.22918e-7, 1.36458e-2, 2.53332e-3],
    [1.41754e+0, 1.44597e-1, 1.70433e-2, 6.39717e-4],
    [2.48775e-1, -1.50890e-1, 6.30894e-9, 1.23707e-3],
    [-4.65119e-1, -1.05081e-1, -8.95701e-2, 1.22450e-2],
    [3.86019e-1, 1.75430e-3, -7.42960e-4, 4.60881e-4],
    [-6.45454e-1, 8.49555e-4, -4.28581e-2, -2.99302e-3],
    [9.48930e-1, 1.97385e-1, -2.50660e-3, -2.06938e-3]])
#: The energies the fit covers, keV.
ELECTRON_E_RANGE_KEV = (0.1, 1000.0)
#: keV per erg.
_KEV_PER_ERG = 1.0 / 1.602176634e-9


def electron_dissipation(y, e_kev) -> np.ndarray:
    """Fang et al. 2010 eq. 4: the normalised energy dissipation f(y, E) of
    monoenergetic electrons of ``e_kev`` keV at normalised depth ``y``."""
    e = float(e_kev)
    if not ELECTRON_E_RANGE_KEV[0] <= e <= ELECTRON_E_RANGE_KEV[1]:
        raise ParticleError(f"electron energy {e:g} keV is outside the 0.1-1000 keV of Fang et al. 2010")
    le = math.log(e)
    c = np.exp(FANG2010_P @ np.array([1.0, le, le * le, le ** 3]))
    y = np.asarray(y, dtype=np.float64)
    return (c[0] * y ** c[1] * np.exp(-c[2] * y ** c[3])
            + c[4] * y ** c[5] * np.exp(-c[6] * y ** c[7]))


def _electron_y(depth, e_kev):
    """Fang et al. 2010 eq. 1, y = (2 / E) (rho H / 6e-6)^0.7, with rho H the
    column depth in g cm-2."""
    return (2.0 / e_kev) * (np.asarray(depth, dtype=np.float64) / 6.0e-6) ** 0.7


def electron_absorbed(depth, energy_flux_erg, e_kev) -> np.ndarray:
    """Energy absorbed above each column depth (g cm-2), keV cm-2 s-1, from an
    isotropic monoenergetic flux of ``energy_flux_erg`` erg cm-2 s-1 at
    ``e_kev``: ``Q int_0^X f(y) dX' / X'``, since Fang's ionisation rate per
    unit mass is ``Q f / (D X)`` (eq. 3, D = 35 eV).  The total over the whole
    column is Q to within the fit's few per cent."""
    q = float(energy_flux_erg) * _KEV_PER_ERG
    ly = np.linspace(math.log(1.0e-6), math.log(1.0e4), 2001)
    # dX / X = dy / (0.7 y): trapezoids in ln y.
    g = electron_dissipation(np.exp(ly), e_kev) / 0.7
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(ly))])
    y = _electron_y(depth, e_kev)
    return q * np.interp(np.log(np.maximum(y, 1.0e-300)), ly, cum, left=0.0, right=cum[-1])


def maxwellian_bins(mean_energy_kev, n: int = 160):
    """(energies keV, energy-flux fractions) of a Maxwellian number flux
    ``phi ~ E exp(-E / E0)`` with mean energy 2 E0 = ``mean_energy_kev`` (the
    OVATION Prime eflux / nflux), on log bins over 0.1-1000 keV; the fractions
    sum to 1 (the part outside the range is dropped and the rest rescaled)."""
    mean = float(mean_energy_kev)
    if not math.isfinite(mean) or mean <= 0.0:
        raise ParticleError(f"electron mean energy {mean_energy_kev!r} keV must be positive")
    e0 = 0.5 * mean
    edges = np.geomspace(*ELECTRON_E_RANGE_KEV, n + 1)
    e = np.sqrt(edges[1:] * edges[:-1])
    w = e ** 3 * np.exp(-e / e0) * np.diff(np.log(edges))    # energy flux ~ E^2 exp(-E/E0) dE
    if not np.any(w > 0):
        raise ParticleError(f"a {mean:g} keV Maxwellian has no energy in 0.1-1000 keV")
    return e, w / w.sum()


def electron_ion_pairs_per_gram(energy_flux_erg, mean_energy_kev, p_edges_pa, *, spectrum="maxwellian"):
    """Layer-mean ion pairs g-1 s-1 between successive interface pressures (Pa)
    from auroral electrons of ``energy_flux_erg`` erg cm-2 s-1: a Maxwellian of
    that mean energy (keV) or, with ``spectrum="mono"``, monoenergetic."""
    x = depth_from_pressure(p_edges_pa)
    if spectrum not in ("maxwellian", "mono"):
        raise ParticleError(f"electron spectrum {spectrum!r} must be 'maxwellian' or 'mono'")
    if float(energy_flux_erg) == 0.0:                 # no aurora (OVATION's mean energy is 0 there)
        return np.zeros(x.shape[0] - 1)
    if spectrum == "mono":
        f = electron_absorbed(x, energy_flux_erg, mean_energy_kev)
    else:
        e, frac = maxwellian_bins(mean_energy_kev)
        f = sum(electron_absorbed(x, energy_flux_erg * w, ek) for ek, w in zip(e, frac) if w > 1.0e-12)
    dx = np.abs(np.diff(x))
    return np.where(dx > 0, np.abs(np.diff(f)) / np.where(dx > 0, dx, 1.0), 0.0) / (W_ION_MEV * 1.0e3)


def depth_from_pressure(p_pa) -> np.ndarray:
    """Column depth, g cm-2, above a pressure in Pa (hydrostatic, standard gravity)."""
    return np.asarray(p_pa, dtype=np.float64) / _G * 0.1


def ion_pairs_per_gram(table_f, p_edges_pa, depth_grid=None, xp=np):
    """Layer-mean ion pairs g-1 s-1 between successive interface pressures
    (Pa; any shape whose first axis runs over the interfaces, top or bottom
    first), from :func:`absorbed` on ``depth_grid`` (log-linear interpolation;
    no absorption above the grid's top)."""
    grid = DEPTH_GRID if depth_grid is None else depth_grid
    lx = xp.log(xp.asarray(grid))
    f = xp.asarray(table_f)
    x = depth_from_pressure(p_edges_pa) if xp is np else xp.asarray(p_edges_pa, dtype=xp.float64) / _G * 0.1
    lxe = xp.log(xp.clip(x, grid[0], grid[-1]))
    fe = xp.interp(lxe.ravel(), lx, f).reshape(x.shape)
    fe = xp.where(x <= grid[0], f[0] * x / grid[0], fe)
    dx = xp.abs(x[1:] - x[:-1])
    return xp.where(dx > 0, xp.abs(fe[1:] - fe[:-1]) / xp.where(dx > 0, dx, 1.0), 0.0) / W_ION_MEV


#: Air's molar mass over Avogadro's number, g per molecule.
_G_PER_MOLECULE = 28.9647 / 6.02214076e23


def vmr_tendency(q_per_gram, per_ion_pair):
    """Volume mixing ratio per second (mol mol-1 s-1) made by ``per_ion_pair``
    molecules per ion pair: ``yield * q * m_air / N_A`` (no temperature needed)."""
    return per_ion_pair * q_per_gram * _G_PER_MOLECULE


def ion_pair_rate(integral_flux, p_edges_pa, rho_kg_m3, channels=CHANNELS_MEV,
                  table: RangeTable | None = None) -> np.ndarray:
    """Ion pairs cm-3 s-1 in each layer of a polar-cap column, from interface
    pressures (Pa) and the layers' air density (kg m-3), computed directly."""
    f = absorbed(integral_flux, depth_from_pressure(p_edges_pa), channels, table)
    x = depth_from_pressure(p_edges_pa)
    per_gram = np.abs(np.diff(f)) / np.abs(np.diff(x)) / W_ION_MEV
    return np.asarray(rho_kg_m3, dtype=np.float64) * 1.0e-3 * per_gram


def energy_flux(integral_flux, channels=CHANNELS_MEV) -> float:
    """The downward energy flux of the spectrum, MeV cm-2 s-1 (isotropic: pi x intensity)."""
    edges, counts = spectrum(integral_flux, channels)
    return float(math.pi * np.dot(counts, np.sqrt(edges[1:] * edges[:-1])))


# ------------------------------------------------------------------ sources

SWPC_PROTONS_URL = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-3-day.json"
#: NOAA's S1 threshold: an event is on while the >10 MeV flux is at least 10 pfu.
SPE_THRESHOLD_PFU = 10.0


@dataclass(frozen=True)
class ProtonRecord:
    """Hourly integral proton fluxes, cm-2 s-1 sr-1 (NaN where missing)."""
    hours: np.ndarray        # datetime64[h], the start of each hour
    channels: tuple          # MeV
    flux: np.ndarray         # (n, len(channels))
    source: str
    sha256: str

    def at(self, when) -> np.ndarray:
        from woof.spacewx.solar_irradiance import naive_utc
        i = int((np.datetime64(naive_utc(when), "h") - self.hours[0]).astype(int))
        if not 0 <= i < self.hours.size:
            raise ParticleError(f"{when}: outside the proton record {self.hours[0]}..{self.hours[-1]} ({self.source})")
        return self.flux[i].copy()

    def active(self, flux) -> bool:
        """Whether an hour is inside a solar proton event (>10 MeV at or above the S1 threshold)."""
        k = self.channels.index(10.0)
        return bool(np.isfinite(flux[k]) and flux[k] >= SPE_THRESHOLD_PFU)


def from_solar_wind(sw) -> ProtonRecord:
    """OMNI2's six channels (words 43-48) out of a :class:`~woof.spacewx.solarwind.SolarWind`."""
    return ProtonRecord(sw.hours, CHANNELS_MEV, np.asarray(sw.protons, dtype=np.float64), sw.source, sw.sha256)


def parse_swpc(text: str, source: str = "swpc") -> ProtonRecord:
    """Hourly means of SWPC's GOES integral proton JSON (records of time_tag,
    energy '>=N MeV' and flux in pfu), at least 3 of 12 five-minute values."""
    import hashlib
    import json
    rows = json.loads(text)
    if not isinstance(rows, list):
        raise ParticleError(f"{source}: not a list of records")
    # One value per five-minute tag and channel: where two spacecraft report
    # the same tag (around a primary switch), the one with more records wins.
    count: dict = {}
    for row in rows:
        if isinstance(row, dict):
            count[row.get("satellite")] = count.get(row.get("satellite"), 0) + 1
    best: dict = {}
    for row in rows:
        try:
            e = float(str(row["energy"]).strip().lstrip(">=").split()[0])
            tag = str(row["time_tag"])
            np.datetime64(tag[:13], "h")
            v = float(row["flux"])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if not np.isfinite(v) or v <= 0:
            continue
        sat = row.get("satellite")
        held = best.get((tag, e))
        if held is None or count.get(sat, 0) > count.get(held[0], 0):
            best[(tag, e)] = (sat, v)
    bins: dict = {}
    channels = set()
    for (tag, e), (_sat, v) in best.items():
        channels.add(e)
        bins.setdefault(np.datetime64(tag[:13], "h"), {}).setdefault(e, []).append(v)
    if not bins or 10.0 not in channels:
        raise ParticleError(f"{source}: no >=10 MeV proton records")
    ch = tuple(sorted(channels))
    hours = np.arange(min(bins), max(bins) + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    flux = np.full((hours.size, len(ch)), np.nan)
    for t, by_e in bins.items():
        i = int((t - hours[0]).astype(int))
        for e, vals in by_e.items():
            if len(vals) >= 3:
                flux[i, ch.index(e)] = float(np.mean(vals))
    return ProtonRecord(hours, ch, flux, source, hashlib.sha256(text.encode()).hexdigest())


def fetch_swpc(*, opener=None) -> ProtonRecord:
    import urllib.request
    opener = opener or (lambda url: urllib.request.urlopen(url, timeout=120))
    with opener(SWPC_PROTONS_URL) as reply:
        return parse_swpc(reply.read().decode("utf-8", "replace"), "swpc")


def load(source: str, start, end) -> ProtonRecord:
    """``"omni"`` (OMNI2, fetched), ``"swpc"`` (GOES real time, the last 3 days)
    or OMNI2 path(s) joined by ``:``."""
    from woof.spacewx import solarwind
    text = str(source or "").strip()
    if text == "swpc":
        return fetch_swpc()
    try:
        return from_solar_wind(solarwind.load(text, start, end))
    except solarwind.SolarWindError as exc:
        raise ParticleError(str(exc)) from None


def polar_cap(mlat_deg, edge_deg: float = POLAR_CAP_DEG) -> np.ndarray:
    """1 where the protons reach (|geomagnetic latitude| >= the edge), else 0."""
    return (np.abs(np.asarray(mlat_deg, dtype=np.float64)) >= edge_deg).astype(np.float64)


__all__ = ["CHANNELS_MEV", "W_ION_MEV", "E_MAX", "POLAR_CAP_DEG", "NOX_PER_ION_PAIR", "HOX_PER_ION_PAIR",
           "DEPTH_GRID", "ParticleError", "stopping_power", "RangeTable", "range_table", "spectrum", "absorbed",
           "depth_from_pressure", "ion_pairs_per_gram", "FANG2010_P", "ELECTRON_E_RANGE_KEV",
           "electron_dissipation", "electron_absorbed", "maxwellian_bins", "electron_ion_pairs_per_gram", "vmr_tendency", "ion_pair_rate", "energy_flux",
           "polar_cap", "SPE_THRESHOLD_PFU", "ProtonRecord", "from_solar_wind", "parse_swpc", "fetch_swpc",
           "load"]
