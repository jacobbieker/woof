"""WOOF's port of NCAR TUV-x v0.17.0 (ebec1cb4, Apache-2.0): photolysis rates for MICM on hex columns.

The port follows TUV-x's own configuration (tools/tuvx_oracle/photolysis.json, TUV 5.4 cut to the reactions
WOOF's MICM mechanisms use) and is graded piece by piece against the oracle's words
(tests/data/tuvx_oracle/fixtures.json, tools/tuvx_oracle).  The input data are fetched on demand
(woof.chem_tuvx_data).  This module holds the configuration-time pieces, built once per run on the host in
binary64 in TUV-x's own operation order:

* the wavelength grid from its CSV file (src/grids/from_csv_file.F90);
* the extraterrestrial flux on that grid (src/profiles/extraterrestrial_flux.F90): each solar file read,
  extended with TUV-x's zero points (src/util.F90 add_point) or, for neckel.flx, shifted and converted;
  interpolated onto the grid (src/interpolate.F90, the area-conserving and fractional-target
  interpolators); each file owns the bins whose lower edge lies in its band; then W m-2 nm-1 to
  photon cm-2 s-1 per bin.

Every expression keeps the Fortran's order and kinds (``1.0e-13`` in the unit conversion is a default-REAL
literal, so its single-precision value enters the product).
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np

#: src/constants.F90
DELTAX = 1.0e-5
HC = 6.626068e-34 * 2.99792458e8
#: src/profiles/extraterrestrial_flux.F90 bin_edge(0:4): file k owns the bins with bin_edge(k-1) <= lower edge
#: < bin_edge(k).
ETF_BAND_EDGES = (0.0, 150.01, 200.07, 1000.99, float(np.finfo(np.float64).max))
#: default-REAL 1.0e-13 (the unit conversion's literal), promoted to double
_E13_SINGLE = float(np.float32(1.0e-13))
_COMMENT = "#!$%*"


class TuvxPortError(ValueError):
    pass


class WavelengthGrid:
    """A TUV-x grid: ``edge`` (n + 1), ``mid`` and ``delta`` (n), binary64."""

    def __init__(self, edges):
        e = np.asarray(edges, dtype=np.float64)
        self.edge = e
        self.mid = 0.5 * (e[:-1] + e[1:])
        self.delta = e[1:] - e[:-1]

    @property
    def ncells(self):
        return self.edge.size - 1


def _values(line):
    """Fortran list-directed numbers of one record (blanks or commas separate; D exponents)."""
    fields = [f for f in re.split(r"[,\s]+", line.strip().split("/")[0]) if f]
    try:
        return [float(f.replace("D", "E").replace("d", "e")) for f in fields]
    except ValueError as error:
        raise TuvxPortError(f"not a list-directed number record: {line!r}") from error


def read_grid_csv(path) -> WavelengthGrid:
    """grid_from_csv_file_t: the first record is a count and is discarded; then one edge per record."""
    lines = Path(path).read_text().splitlines()
    edges = [_values(line)[0] for line in lines[1:] if line.strip()]
    return WavelengthGrid(edges)


def add_point(x, y, xnew, ynew):
    """src/util.F90 add_point: insert ``(xnew, ynew)`` into ascending ``x``."""
    x, y = list(x), list(y)
    if any(b <= a for a, b in zip(x, x[1:])):
        raise TuvxPortError("add_point: grid not monotonically increasing")
    if xnew in x:
        raise TuvxPortError("add_point: xnew exactly matches a grid x value")
    if xnew < x[0]:
        at = 0
    elif xnew > x[-1]:
        at = len(x)
    else:
        at = next(i for i in range(1, len(x)) if x[i] > xnew)
    x.insert(at, xnew)
    y.insert(at, ynew)
    return x, y


def interpolate_conserving(x_target, x_source, y_source):
    """src/interpolate.F90 interpolate_conserving (area-conserving linear), its 1-based walk kept literally."""
    xt = [float(v) for v in x_target]
    xs = [None] + [float(v) for v in x_source]          # 1-based
    ys = [None] + [float(v) for v in y_source]
    n, ng = len(xs) - 1, len(xt)
    if any(xs[i] >= xs[i + 1] for i in range(1, n)):
        raise TuvxPortError("source grid must be monotonically increasing")
    if any(xt[i] >= xt[i + 1] for i in range(ng - 1)):
        raise TuvxPortError("target grid must be monotonically increasing")
    if xs[1] > xt[0] or xs[n] < xt[ng - 1]:
        raise TuvxPortError("src and target grid do not overlap")
    out = [0.0] * (ng - 1)
    jstart = 1
    for i in range(ng - 1):
        area = 0.0
        xgl, xgu = xt[i], xt[i + 1]
        k = jstart
        if k < n:
            while True:
                if xs[k + 1] <= xgl:
                    jstart = k - 1
                    k = k + 1
                    if k >= n:
                        break
                else:
                    break
            while k < n and xs[k] < xgu:
                if k < 1:
                    raise TuvxPortError("conserving interpolation walked before the source grid")
                jstart = k - 1
                a1 = max(xs[k], xgl)
                a2 = min(xs[k + 1], xgu)
                if xs[k + 1] == xs[k]:
                    darea = 0.0
                else:
                    slope = (ys[k + 1] - ys[k]) / (xs[k + 1] - xs[k])
                    b1 = ys[k] + slope * (a1 - xs[k])
                    b2 = ys[k] + slope * (a2 - xs[k])
                    darea = 0.5 * (a2 - a1) * (b2 + b1)
                area = area + darea
                k = k + 1
        out[i] = area / (xgu - xgl)
    return np.asarray(out, dtype=np.float64)


def interpolate_fractional_target(x_target, x_source, y_source):
    """src/interpolate.F90 interpolate_fractional_target with its default ``fold_in_ = .false.``."""
    xt = [None] + [float(v) for v in x_target]
    xs = [None] + [float(v) for v in x_source]
    ys = [None] + [float(v) for v in y_source]
    n, ng = len(xs) - 1, len(xt) - 1
    out = [0.0] * (ng - 1)
    jstart = 1
    for i in range(1, ng):
        total = 0.0
        j = jstart
        if j < n:
            while True:
                if xs[j + 1] < xt[i]:
                    jstart = j
                    j = j + 1
                    if j >= n:
                        break
                else:
                    break
            while xs[j] <= xt[i + 1] and j < n:
                a1 = max(xs[j], xt[i])
                a2 = min(xs[j + 1], xt[i + 1])
                total = total + ys[j] * (a2 - a1)
                j = j + 1
            out[i - 1] = total / (xt[i + 1] - xt[i])
    return np.asarray(out, dtype=np.float64)


INTERPOLATORS = {"conserving": interpolate_conserving, "fractional target": interpolate_fractional_target}


def read_flux_file(path):
    """One solar file's (wavelength, value) records, TUV-x's comment rule: a record whose first non-blank
    character is one of ``#!$%*`` is skipped (a blank record is data, and fails, as in TUV-x)."""
    x, y = [], []
    for line in Path(path).read_text().splitlines():
        head = line[:132].lstrip()[:1] or " "
        if head in _COMMENT:
            continue
        values = _values(line[:132])
        if len(values) < 2:
            raise TuvxPortError(f"invalid data format in {path}: {line!r}")
        x.append(values[0])
        y.append(values[1])
    return x, y


def extraterrestrial_flux(files, interpolators, grid: WavelengthGrid, *, root=".") -> np.ndarray:
    """profile_extraterrestrial_flux_t: photon cm-2 s-1 per bin of ``grid`` from the solar ``files`` (paths
    relative to ``root``), each with its interpolator (``""`` takes ``"conserving"``)."""
    nbins = grid.ncells
    mid = np.zeros(nbins, dtype=np.float64)
    if len(files) > len(ETF_BAND_EDGES) - 1:
        raise TuvxPortError("more solar files than TUV-x's flux bands")
    for index, (name, kind) in enumerate(zip(files, interpolators), start=1):
        kind = kind or "conserving"
        if kind not in INTERPOLATORS:
            raise TuvxPortError(f"interpolator {kind!r} is not ported")
        x, y = read_flux_file(Path(root) / name)
        if "neckel.flx" in str(name):
            xa = np.asarray(x, dtype=np.float64)
            shifted = np.where(xa < 630.0, xa - 0.5, np.where(xa < 870.0, xa - 1.0, xa - 2.5))
            y = list((1.0e13 * HC) * np.asarray(y, dtype=np.float64) / xa)
            x = list(shifted)
            x.append(x[-1] + 2.5)
            y.append(0.0)
        else:
            x, y = add_point(x, y, (1.0 - DELTAX) * x[0], 0.0)
            x, y = add_point(x, y, 0.0, 0.0)
            x, y = add_point(x, y, (1.0 + DELTAX) * x[-1], 0.0)
            x, y = add_point(x, y, 1.0e38, 0.0)
        values = INTERPOLATORS[kind](grid.edge, x, y)
        lower = grid.edge[:nbins]
        band = (ETF_BAND_EDGES[index - 1] <= lower) & (lower < ETF_BAND_EDGES[index])
        mid = np.where(band, values, mid)
    return _E13_SINGLE * mid * grid.mid * grid.delta / HC


def configuration(config, *, root):
    """(wavelength grid, extraterrestrial flux) of a TUV-x configuration dict, its data paths under ``root``."""
    grids = {g["name"]: g for g in config["grids"]}
    wavelength = grids["wavelength"]
    if wavelength["type"] != "from csv file":
        raise TuvxPortError(f"wavelength grid type {wavelength['type']!r} is not ported")
    grid = read_grid_csv(Path(root) / wavelength["file path"])
    profiles = {p["name"]: p for p in config["profiles"]}
    etf = profiles["extraterrestrial flux"]
    files = etf["file path"]
    kinds = etf.get("interpolator", [""] * len(files))
    return grid, extraterrestrial_flux(files, kinds, grid, root=root)
