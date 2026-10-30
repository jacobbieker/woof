"""OPAC aerosol for woof.radtran (Hess, Koepke & Schult 1998, BAMS 79, 831-844).

Species (OPAC's components; size distribution, density and refractive index
per relative-humidity node are read at run time from the user's libRadtran
data, ``aerosol/OPAC/size_distr.cfg`` and ``refractive_indices/``):

    inso insoluble            waso water soluble      soot soot
    ssam sea salt (acc.)      sscm sea salt (coarse)  minm mineral (nucl.)
    miam mineral (acc.)       micm mineral (coarse)   mitr mineral transported
    suso sulfate droplets

Optical properties are computed by woof's own Mie code
(:func:`woof.radtran.mie.lognormal_bulk`: lognormal number distribution
between OPAC's r_min and r_max, exact Legendre moments) at OPAC's tabulated
wavelengths (0.25-6 um) and humidity nodes, and cached on disk
(``$XDG_CACHE_HOME/woof/radtran-opac``).  libRadtran's ``mie`` tool, run as
a black box on the same OPAC inputs, agrees to <= 1.3e-5 in extinction
per mass in the visible and <= 2.7e-4 at 4 um, where the ``mie`` tool's own
size integration (its default size-parameter step) is the coarser of the two
(tools/uvspec_oracle/opac_library.py, docs/dev/radtran.md).

How a layer's optical properties are formed follows uvspec 2.0.6 with an
``aerosol_species_file``, established by probing that binary (never its
source):

* **Mass**: the profile gives each species' mass concentration [g m-3]
  per layer.  The extinction is mass x the species' extinction per unit
  mass *at the selected humidity node* (the Mie table's extinction per
  gram of the wet particle at that node); uvspec applies no further
  hygroscopic mass scaling.  The OPAC mixtures in libRadtran's data are
  masses "corresponding to OPAC at 50 % relative humidity", so a woof
  mapping from number concentrations uses the per-particle mass at the 50 %
  node (:func:`particle_mass`).
* **Relative humidity** at a level: 100 e / e_s(T), e = n_H2O k T, with the
  saturation pressure over water

      ln(e_s / hPa) = 19.1484007 - 4055.15160 / T - 185921.210 / T^2,

  a three-constant law fitted to uvspec's own ``output_user rh`` over
  182-335 K (residual 4e-7, its print precision).  A layer's relative
  humidity is the mean of its two levels' values.
* **Humidity node**: the nearest tabulated node, except that a layer whose
  nearest node is the first (dry, 0 %) one keeps the node of the layer
  above it (layers visited from the top, the top layer starting from the
  dry node).  So the dry node is used only while every layer above uses it
  too, and a dry layer below a humid one takes that humid layer's node.
  The search restarts for every wavelength.
* **Wavelength**: linear interpolation between the bracketing tabulated
  wavelengths of the extinction, scattering coefficient and scattering
  coefficient x Legendre moment (i.e. moments weighted by scattering).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from woof.radtran import data, mie

SPECIES = ("inso", "waso", "soot", "ssam", "sscm", "minm", "miam", "micm", "mitr", "suso")
_CFG_INDEX = {s: i + 1 for i, s in enumerate(SPECIES)}
#: OPAC mixtures shipped with libRadtran (aerosol/OPAC/standard_aerosol_files).
MIXTURES = ("continental_clean", "continental_average", "continental_polluted", "urban",
            "maritime_clean", "maritime_polluted", "maritime_tropical", "desert", "antarctic")
#: Highest Legendre moment kept (woof's maximum number of streams).
LMAX = 32
#: Tabulated wavelength range woof uses [um] (OPAC's grid, 0.25-6 um).
LAM_MIN_UM, LAM_MAX_UM = 0.25, 6.0
#: Size integration of the Mie tables: trapezoid in ln r, steps DLNR in ln r
#: and at most DX in size parameter (:func:`woof.radtran.mie.size_grid`).
DLNR, DX = 0.005, 0.05
#: Version of the table recipe (part of the cache key).
TABLE_VERSION = 1

KB = 1.380649e-23
#: ln(e_s/hPa) = ES_A + ES_B / T + ES_C / T^2 (fitted to uvspec's rh output)
ES_A, ES_B, ES_C = 19.1484007, -4055.15160, -185921.210


# ---------------------------------------------------------------------------
# humidity
# ---------------------------------------------------------------------------

def saturation_pressure_hpa(t_k):
    t = np.asarray(t_k, dtype=np.float64)
    return np.exp(ES_A + ES_B / t + ES_C / (t * t))


def relative_humidity(t_k, n_h2o_cm3):
    """Relative humidity over water [%] from temperature and H2O number density."""
    t = np.asarray(t_k, dtype=np.float64)
    e_hpa = np.asarray(n_h2o_cm3, dtype=np.float64) * 1e6 * KB * t * 1e-2
    return 100.0 * e_hpa / saturation_pressure_hpa(t)


def layer_rh(col) -> np.ndarray:
    """(..., nlyr) layer relative humidity, mean of the two levels' values."""
    rh = relative_humidity(col.t, col.dens["H2O"])
    return 0.5 * (rh[..., :-1] + rh[..., 1:])


def select_nodes(rh_layer, hum) -> np.ndarray:
    """Humidity-node index per layer (layers top first along the last axis).

    uvspec's rule (probed): the nearest node, except that where the nearest
    is the first (dry) node the layer keeps the node of the layer above;
    the top layer starts from the first node.  So the dry node is selected
    only while every layer above has selected it too.
    """
    rh = np.asarray(rh_layer, dtype=np.float64)
    hum = np.asarray(hum, dtype=np.float64)
    out = np.zeros(rh.shape, dtype=np.int64)
    if hum.size == 1:
        return out
    near = np.argmin(np.abs(rh[..., None] - hum), axis=-1)
    prev = np.zeros(rh.shape[:-1], dtype=np.int64)
    for k in range(rh.shape[-1]):
        prev = np.where(near[..., k] > 0, near[..., k], prev)
        out[..., k] = prev
    return out


# ---------------------------------------------------------------------------
# OPAC species inputs (user's libRadtran data)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Node:
    rh: float
    r_min: float
    r_max: float
    r_mod: float
    rho: float          # g cm-3
    sigma: float


@lru_cache(maxsize=None)
def species_nodes(name: str) -> tuple:
    """OPAC size-distribution rows of a species, one per humidity node."""
    if name not in _CFG_INDEX:
        raise ValueError(f"unknown OPAC species {name!r}; known: {', '.join(SPECIES)}")
    rows = []
    for line in open(data.path("aerosol", "OPAC", "size_distr.cfg")):
        s = line.split("#", 1)[0].split()
        if len(s) == 7 and int(s[0]) == _CFG_INDEX[name]:
            rh, rmin, rmax, rmod, rho, sig = (float(v) for v in s[1:])
            rows.append(Node(rh, rmin, rmax, rmod, rho, sig))
    return tuple(rows)


def _refr_path(name: str, rh: float) -> Path:
    return data.path("aerosol", "OPAC", "refractive_indices", f"{name}{int(round(rh)):02d}_refr.dat")


@lru_cache(maxsize=None)
def _refr_table(name: str, rh: float) -> np.ndarray:
    rows = [[float(v) for v in l.split()[:3]] for l in open(_refr_path(name, rh)) if l.strip()]
    return np.array(rows)


def refractive_index(name: str, rh: float, lam_um: float) -> complex:
    """Refractive index (absorption >= 0), linear in wavelength between the file's nodes."""
    t = _refr_table(name, rh)
    return complex(float(np.interp(lam_um, t[:, 0], t[:, 1])), abs(float(np.interp(lam_um, t[:, 0], t[:, 2]))))


@lru_cache(maxsize=None)
def wavelengths_um(name: str) -> np.ndarray:
    """Tabulated wavelengths of a species within LAM_MIN_UM..LAM_MAX_UM (its dry node's grid)."""
    t = _refr_table(name, species_nodes(name)[0].rh)
    lam = t[:, 0]
    return lam[(lam >= LAM_MIN_UM - 1e-9) & (lam <= LAM_MAX_UM + 1e-9)].copy()


def particle_mass(name: str, rh: float = 50.0) -> float:
    """Mean mass of one particle [g] at the species' humidity node nearest ``rh``
    (lognormal between r_min and r_max, the size integral the Mie tables use)."""
    nodes = species_nodes(name)
    nd = nodes[int(np.argmin([abs(n.rh - rh) for n in nodes]))]
    lnr = np.log(np.exp(np.linspace(math.log(nd.r_min), math.log(nd.r_max), 4001)))
    tw = np.full(lnr.size, lnr[1] - lnr[0])
    tw[0] *= 0.5
    tw[-1] *= 0.5
    w = np.exp(-0.5 * ((lnr - math.log(nd.r_mod)) / math.log(nd.sigma)) ** 2) * tw
    v_um3 = np.sum(w * 4.0 / 3.0 * math.pi * np.exp(3 * lnr)) / np.sum(w)
    return float(nd.rho * v_um3 * 1e-12)                    # g cm-3 * um3 -> g


# ---------------------------------------------------------------------------
# Mie tables (computed on demand, cached on disk)
# ---------------------------------------------------------------------------

def _cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "woof" / "radtran-opac"


def _node_key(name, nd: Node, lam_um, m) -> str:
    rec = dict(v=TABLE_VERSION, s=name, node=nd.__dict__, lam=repr(float(lam_um)),
               m=[repr(m.real), repr(m.imag)], dlnr=DLNR, dx=DX, lmax=LMAX)
    return hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:24]


_MEM: dict = {}

#: Scattering angles [deg] of the tabulated phase functions (TMS / exact
#: single scattering): 0.05 deg steps to 1 deg, 0.25 deg to 10 deg, then 1 deg.
PHASE_THETA = np.concatenate((np.linspace(0.0, 1.0, 21), np.linspace(1.25, 10.0, 36),
                              np.linspace(11.0, 180.0, 170)))


def _node_file(name, il, ih, kind):
    nd = species_nodes(name)[ih]
    lam = float(wavelengths_um(name)[il])
    m = refractive_index(name, nd.rh, lam)
    key = _node_key(name, nd, lam, m) if kind == "opt" else _node_key(name, nd, lam, m) + "-phase"
    return nd, lam, m, _cache_dir() / f"{name}-{key}.npy"


def _save(f, v):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(tmp, v)
    os.replace(tmp, f)


def node_optics(name: str, il: int, ih: int):
    """(ext [km-1 per g m-3], ssa, chi[0..LMAX]) of species ``name`` at
    tabulated wavelength index ``il`` and humidity node ``ih``."""
    key = (name, il, ih)
    if key in _MEM:
        return _MEM[key]
    nd, lam, m, f = _node_file(name, il, ih, "opt")
    if f.exists():
        v = np.load(f)
    else:
        o = mie.lognormal_bulk(lam, m, nd.r_mod, nd.sigma, nd.r_min, nd.r_max, nd.rho, lmax=LMAX, dlnr=DLNR, dx=DX)
        v = np.concatenate(([o["ext"], o["ssa"]], o["chi"]))
        _save(f, v)
    out = (float(v[0]), float(v[1]), v[2:].copy())
    _MEM[key] = out
    return out


def node_phase(name: str, il: int, ih: int) -> np.ndarray:
    """Phase function (normalised, (1/2) int p dmu = 1) at PHASE_THETA."""
    key = (name, il, ih, "phase")
    if key in _MEM:
        return _MEM[key]
    nd, lam, m, f = _node_file(name, il, ih, "phase")
    if f.exists():
        v = np.load(f)
    else:
        o = mie.lognormal_bulk(lam, m, nd.r_mod, nd.sigma, nd.r_min, nd.r_max, nd.rho, lmax=LMAX, dlnr=DLNR,
                               dx=DX, angles_cos=np.cos(np.radians(PHASE_THETA)))
        v = o["phase"]
        _save(f, v)
    _MEM[key] = v
    return v


@dataclass
class SpeciesAt:
    """One species' optics at one wavelength for every humidity node."""
    ext: np.ndarray      # (nhum,) km-1 per g m-3
    sca: np.ndarray      # (nhum,) ext * ssa
    sca_chi: np.ndarray  # (nhum, LMAX+1) sca * chi_l
    hum: np.ndarray


def species_at(name: str, lam_nm: float) -> SpeciesAt:
    """Linear interpolation in wavelength (of ext, sca, sca*chi) between the
    bracketing tabulated wavelengths (uvspec's rule, probed)."""
    lam = lam_nm * 1e-3
    grid = wavelengths_um(name)
    if not (grid[0] <= lam <= grid[-1]):
        raise ValueError(f"OPAC aerosol tables cover {grid[0] * 1e3:.0f}-{grid[-1] * 1e3:.0f} nm; got {lam_nm} nm")
    i = int(np.clip(np.searchsorted(grid, lam, side="right") - 1, 0, grid.size - 2))
    t = (lam - grid[i]) / (grid[i + 1] - grid[i])
    nodes = species_nodes(name)
    ext, sca, sc = [], [], []
    for ih in range(len(nodes)):
        e0, w0, c0 = node_optics(name, i, ih)
        if t == 0.0:
            e1, w1, c1 = e0, w0, c0
        else:
            e1, w1, c1 = node_optics(name, i + 1, ih)
        ext.append((1 - t) * e0 + t * e1)
        sca.append((1 - t) * e0 * w0 + t * e1 * w1)
        sc.append((1 - t) * e0 * w0 * c0 + t * e1 * w1 * c1)
    return SpeciesAt(np.array(ext), np.array(sca), np.array(sc), np.array([n.rh for n in nodes]))


def species_phase_at(name: str, lam_nm: float) -> np.ndarray:
    """(nhum, len(PHASE_THETA)) phase function at ``lam_nm``: the
    scattering-weighted linear interpolation in wavelength used for the moments."""
    lam = lam_nm * 1e-3
    grid = wavelengths_um(name)
    i = int(np.clip(np.searchsorted(grid, lam, side="right") - 1, 0, grid.size - 2))
    t = (lam - grid[i]) / (grid[i + 1] - grid[i])
    out = []
    for ih in range(len(species_nodes(name))):
        e0, w0, _ = node_optics(name, i, ih)
        p0 = node_phase(name, i, ih)
        if t == 0.0:
            out.append(p0)
            continue
        e1, w1, _ = node_optics(name, i + 1, ih)
        p1 = node_phase(name, i + 1, ih)
        s0, s1 = (1 - t) * e0 * w0, t * e1 * w1
        out.append((s0 * p0 + s1 * p1) / (s0 + s1))
    return np.array(out)


def phase_at_angles(table: np.ndarray, cos_scat) -> np.ndarray:
    """Interpolate tabulated phase functions (..., ntheta) to scattering-angle
    cosines: linear in angle of ln p."""
    th = np.degrees(np.arccos(np.clip(np.asarray(cos_scat, dtype=np.float64), -1.0, 1.0)))
    lp = np.log(table)
    j = np.clip(np.searchsorted(PHASE_THETA, th, side="right") - 1, 0, PHASE_THETA.size - 2)
    t = (th - PHASE_THETA[j]) / (PHASE_THETA[j + 1] - PHASE_THETA[j])
    return np.exp((1 - t) * lp[..., j] + t * lp[..., j + 1])


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------

@dataclass
class AerosolLayers:
    """Per-layer OPAC species mass concentrations [g m-3], layers top first.

    ``mass`` is (..., nlyr, nspecies) in the order of ``species``.
    """
    species: tuple
    mass: np.ndarray

    def __post_init__(self):
        for s in self.species:
            if s not in _CFG_INDEX:
                raise ValueError(f"unknown OPAC species {s!r}")
        self.mass = np.asarray(self.mass, dtype=np.float64)
        if self.mass.shape[-1] != len(self.species):
            raise ValueError("aerosol mass needs one column per species")


@dataclass
class LayerAerosol:
    tau_ext: np.ndarray       # (..., nlyr)
    tau_sca: np.ndarray
    sca_chi: np.ndarray       # (..., nlyr, LMAX+1) tau_sca * chi_l
    by_species: list = field(default_factory=list)   # per species (tau_sca, node index, SpeciesAt)


def layer_optics(aer: AerosolLayers, nodes: dict, lam_nm: float, dz_km) -> LayerAerosol:
    """Aerosol optical depths of every layer at wavelength ``lam_nm``.

    ``nodes``: species -> (..., nlyr) humidity-node index (:func:`select_nodes`).
    """
    dz = np.asarray(dz_km, dtype=np.float64)
    shape = np.broadcast_shapes(aer.mass.shape[:-1], dz.shape)
    text = np.zeros(shape)
    tsca = np.zeros(shape)
    sc = np.zeros(shape + (LMAX + 1,))
    per = []
    for k, s in enumerate(aer.species):
        m = np.broadcast_to(aer.mass[..., k], shape)
        if not np.any(m > 0):
            continue
        sa = species_at(s, lam_nm)
        ih = np.broadcast_to(nodes[s], shape)
        te = m * sa.ext[ih] * dz
        ts = m * sa.sca[ih] * dz
        text += te
        tsca += ts
        sc += (m * dz)[..., None] * sa.sca_chi[ih]
        per.append((s, ts, ih))
    return LayerAerosol(text, tsca, sc, per)


def layer_phase_sum(la: LayerAerosol, lam_nm: float, cosang, scale=1.0) -> np.ndarray:
    """sum over species of tau_sca * p(Theta) per layer and direction.

    ``cosang``: (ncol, 1, numu, nphi) scattering-angle cosines; returns
    (ncol, nlyr, numu, nphi).  ``scale`` multiplies every species' tau_sca
    (the driver's no_absorption / no_scattering switches)."""
    out = 0.0
    for s, ts, ih in la.by_species:
        tab = species_phase_at(s, lam_nm)                     # (nhum, ntheta)
        nh = tab.shape[0]
        p = phase_at_angles(tab, cosang)                      # (nhum?) broadcast below
        # p: (nhum, ncol, 1, numu, nphi) -> pick each layer's node
        p = np.moveaxis(p, 0, -1)                             # (ncol, 1, numu, nphi, nhum)
        idx = np.broadcast_to(ih[..., None, None, None], ih.shape + p.shape[2:4] + (1,))
        sel = np.take_along_axis(np.broadcast_to(p, ih.shape + p.shape[2:]), idx, axis=-1)[..., 0]
        out = out + (ts * scale)[..., None, None] * sel
    return out


def humidity_nodes(aer: AerosolLayers, col) -> dict:
    rh = layer_rh(col)
    return {s: select_nodes(rh, [n.rh for n in species_nodes(s)]) for s in aer.species}


# ---------------------------------------------------------------------------
# OPAC mixtures (libRadtran's standard_aerosol_files)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def mixture(name: str):
    """(z_km ascending, species tuple, mass[nz, nspecies] g m-3) of an OPAC mixture file."""
    if name not in MIXTURES:
        raise ValueError(f"unknown OPAC mixture {name!r}; known: {', '.join(MIXTURES)}")
    species = None
    rows = []
    for line in open(data.path("aerosol", "OPAC", "standard_aerosol_files", f"{name}.dat")):
        s = line.split()
        if not s:
            continue
        if s[0].startswith("#"):
            toks = [t.lower() for t in line.replace("#", " ").split()]
            if toks[:1] == ["z(km)"]:
                species = tuple(toks[1:])
            continue
        rows.append([float(v) for v in s])
    a = np.array(sorted(rows, key=lambda r: r[0]))
    if species is None or len(species) != a.shape[1] - 1:
        raise ValueError(f"cannot read the species header of OPAC mixture {name}")
    return a[:, 0], species, a[:, 1:]


def mixture_layers(name: str, z_bottom_km) -> AerosolLayers:
    """An OPAC mixture on layers whose bottoms lie ``z_bottom_km`` above the
    surface: a layer takes the row whose interval [z_k, z_k+1) holds its
    bottom (the mixture file's row-defines-the-layer-above convention)."""
    z, species, mass = mixture(name)
    zb = np.asarray(z_bottom_km, dtype=np.float64)
    k = np.searchsorted(z, zb + 1e-9, side="right") - 1
    inside = (k >= 0) & (k < z.size - 1)
    out = np.where(inside[..., None], mass[np.clip(k, 0, z.size - 1)], 0.0)
    return AerosolLayers(species=species, mass=out)
