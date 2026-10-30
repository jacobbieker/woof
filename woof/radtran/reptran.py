"""REPTRAN band parameterisation (Gasteiger et al. 2014, JQSRT 148, 99-115).

Reads libRadtran's REPTRAN netCDF data at run time from the user's own
libRadtran installation (:mod:`woof.radtran.data`); nothing is copied into
woof.  A band is a weighted sum over its representative wavelengths::

    <X>_band = sum_i w_i X(lambda_i)

and the extraterrestrial irradiance at each representative wavelength is in
the band file (variable ``extra``, mW m-2 nm-1).

Molecular absorption cross sections come from the per-species lookup tables
``<file>.lookup.<SPECIES>.cdf`` (ARTS-generated, Buehler et al. 2010):
``xsec(t_pert, vmr, wvl, pressure)`` with each pressure node carrying a
reference temperature ``t_ref`` and the temperature axis given as offsets
from it.  woof interpolates linearly in pressure, linearly in the
temperature offset from each pressure node's own reference, and (H2O only)
linearly in the H2O volume mixing ratio; the stored values times 1e-16 are
cross sections in cm^2.  (That reading of the tables was established by
probing uvspec as a black box with thin layers; see docs/dev/radtran.md.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

#: Species with lookup tables, in the band file's order.
SPECIES = ("H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2", "N2")
XSEC_TO_CM2 = 1e-16


@dataclass
class Band:
    index: int              # 0-based band index in the file
    wvlmin: float
    wvlmax: float
    integral: float         # wvl_integral [nm]
    iwvl: np.ndarray        # 0-based indices of representative wavelengths
    weight: np.ndarray


@dataclass
class Reptran:
    path: Path
    wvl: np.ndarray                   # representative wavelengths [nm]
    extra: np.ndarray                 # extraterrestrial irradiance [mW m-2 nm-1]
    wvlmin: np.ndarray
    wvlmax: np.ndarray
    integral: np.ndarray
    nwvl_in_band: np.ndarray
    iwvl: np.ndarray                  # (max_nwvl, nbands) 1-based in file -> 0-based here
    weight: np.ndarray
    species: tuple
    source: np.ndarray                # (nwvl, nspecies) cross_section_source
    _lookups: dict = field(default_factory=dict)

    def band(self, b: int) -> Band:
        n = int(self.nwvl_in_band[b])
        return Band(b, float(self.wvlmin[b]), float(self.wvlmax[b]), float(self.integral[b]),
                    self.iwvl[:n, b].copy(), self.weight[:n, b].copy())

    def bands_overlapping(self, lo: float, hi: float) -> list[int]:
        """Bands overlapping [lo, hi] nm (uvspec's wavelength selection)."""
        sel = np.nonzero((self.wvlmax > lo) & (self.wvlmin < hi))[0]
        if lo == hi:
            sel = np.nonzero((self.wvlmin <= lo) & (self.wvlmax > lo))[0][:1]
        return [int(b) for b in sel]

    def band_containing(self, lam: float) -> int:
        b = np.nonzero((self.wvlmin <= lam) & (self.wvlmax > lam))[0]
        if b.size == 0:
            raise ValueError(f"{lam} nm is outside the REPTRAN bands of {self.path.name}")
        return int(b[0])

    def needs_lookup(self, iw: int, species: str) -> bool:
        j = self.species.index(species)
        return bool(self.source[iw, j] == 1)

    def lookup(self, species: str) -> "Lookup":
        if species not in self._lookups:
            self._lookups[species] = Lookup.read(
                self.path.with_name(self.path.stem + f".lookup.{species}.cdf"))
        return self._lookups[species]


@dataclass
class Lookup:
    path: Path
    wvl_index: np.ndarray       # 0-based representative wavelength index per row
    pressure: np.ndarray        # Pa, decreasing
    t_ref: np.ndarray           # K per pressure node
    t_pert: np.ndarray          # K offsets
    vmrs: np.ndarray            # H2O vmr nodes (one node for other species)
    xsec: np.ndarray            # (t_pert, vmr, row, p) as stored (float32 -> float64)
    _row: dict = field(default_factory=dict)

    @classmethod
    def read(cls, path: Path) -> "Lookup":
        import netCDF4
        with netCDF4.Dataset(path) as d:
            wi = np.asarray(d["wvl_index"][:], dtype=np.int64) - 1
            return cls(path=path, wvl_index=wi,
                       pressure=np.asarray(d["pressure"][:], dtype=np.float64),
                       t_ref=np.asarray(d["t_ref"][:], dtype=np.float64),
                       t_pert=np.asarray(d["t_pert"][:], dtype=np.float64),
                       vmrs=np.asarray(d["vmrs"][:], dtype=np.float64),
                       xsec=np.asarray(d["xsec"][:], dtype=np.float32).astype(np.float64),
                       _row={int(w): r for r, w in enumerate(wi)})

    def has(self, iw: int) -> bool:
        return iw in self._row

    def sigma(self, iw: int, p_hpa, t, vmr=None) -> np.ndarray:
        """Cross section [cm^2] at representative wavelength ``iw`` for
        pressures [hPa], temperatures [K] and (H2O) vmr, any matching shape."""
        r = self._row[iw]
        x = self.xsec[:, :, r, :]                     # (nt, nv, np)
        p = np.asarray(p_hpa, dtype=np.float64) * 100.0
        t = np.asarray(t, dtype=np.float64)
        P = self.pressure
        # pressure nodes are decreasing: work in ascending order
        Pa = P[::-1]
        ip = np.clip(np.searchsorted(Pa, p) - 1, 0, Pa.size - 2)
        p0, p1 = Pa[ip], Pa[ip + 1]
        wp1 = (p - p0) / (p1 - p0)
        out = np.zeros(np.broadcast(p, t).shape)
        for node, wpn in ((P.size - 1 - ip, 1.0 - wp1), (P.size - 2 - ip, wp1)):
            dt = t - self.t_ref[node]
            it = np.clip(np.searchsorted(self.t_pert, dt) - 1, 0, self.t_pert.size - 2)
            t0, t1 = self.t_pert[it], self.t_pert[it + 1]
            wt1 = (dt - t0) / (t1 - t0)
            if self.vmrs.size > 1 and vmr is not None:
                v = np.asarray(vmr, dtype=np.float64)
                iv = np.clip(np.searchsorted(self.vmrs, v) - 1, 0, self.vmrs.size - 2)
                v0, v1 = self.vmrs[iv], self.vmrs[iv + 1]
                wv1 = (v - v0) / (v1 - v0)
                val = 0.0
                for ti, wt in ((it, 1.0 - wt1), (it + 1, wt1)):
                    for vi, wv in ((iv, 1.0 - wv1), (iv + 1, wv1)):
                        val = val + wt * wv * x[ti, vi, node]
            else:
                val = (1.0 - wt1) * x[it, 0, node] + wt1 * x[it + 1, 0, node]
            out = out + wpn * val
        return out * XSEC_TO_CM2


@lru_cache(maxsize=None)
def open_reptran(path: str) -> Reptran:
    import netCDF4
    p = Path(path)
    with netCDF4.Dataset(p) as d:
        names = tuple(b"".join(np.asarray(s).tobytes() for s in row).decode().strip("\x00 ")
                      for row in d["species_name"][:])
        names = tuple(n.strip() for n in names)
        return Reptran(
            path=p,
            wvl=np.asarray(d["wvl"][:], dtype=np.float64),
            extra=np.asarray(d["extra"][:], dtype=np.float64) if "extra" in d.variables else None,
            wvlmin=np.asarray(d["wvlmin"][:], dtype=np.float64),
            wvlmax=np.asarray(d["wvlmax"][:], dtype=np.float64),
            integral=np.asarray(d["wvl_integral"][:], dtype=np.float64),
            nwvl_in_band=np.asarray(d["nwvl_in_band"][:], dtype=np.int64),
            iwvl=np.asarray(d["iwvl"][:], dtype=np.int64) - 1,
            weight=np.asarray(d["iwvl_weight"][:], dtype=np.float64),
            species=names,
            source=np.asarray(d["cross_section_source"][:], dtype=np.int64))
