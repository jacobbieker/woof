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
    if len(interpolators) != len(files):
        raise TuvxPortError(f"{len(interpolators)} interpolators for {len(files)} solar files")
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


# ------------------------------------------------------------------------------------------------------------
# Cross sections and quantum yields (src/cross_section.F90, cross_sections/{o3_tint,no2_tint}.F90,
# src/quantum_yield.F90, quantum_yields/{o3-o2_o1d,o3-o2_o3p,no2_tint}.F90): (height edge, wavelength bin) tables
# from the column's edge temperatures.  The O2 Lyman-alpha / Schumann-Runge bands are applied in the rate
# calculation, not here (photolysis_rates.F90 "apply O2 bands").

def read_parameters(path, header):
    """tuvx_netcdf read_netcdf_file: ``(wavelength or None, temperature or None, parameters (nlambda, nparam))``."""
    from netCDF4 import Dataset
    with Dataset(path) as ds:
        params = np.asarray(ds.variables[header + "parameters"][:], dtype=np.float64).T
        wavelength = (np.asarray(ds.variables["wavelength"][:], dtype=np.float64)
                      if "wavelength" in ds.variables else None)
        temperature = (np.asarray(ds.variables["temperature"][:], dtype=np.float64)
                       if "temperature" in ds.variables else None)
    if params.ndim != 2 or params.shape[1] < 1:
        raise TuvxPortError(f"{path} contains no parameters")
    if wavelength is not None and wavelength.size != params.shape[0]:
        raise TuvxPortError(f"wavelength and parameters array size mismatch in {path}")
    return wavelength, temperature, params


def _extrapolation(config, key, x, y, *, qy_upper=False):
    entry = config.get(key)
    if entry is None:
        return 0.0
    kind = entry["type"]
    if kind == "boundary":
        return None                     # the caller reads its own boundary value at the point TUV-x does
    if kind == "constant":
        return float(entry["value"])
    raise TuvxPortError(f"bad extrapolation type {kind!r}")


def add_points_cross_section(config, x, y):
    """cross_section.F90 add_points: the extrapolation values are read from the data before any insertion."""
    x, y = list(x), list(y)
    lower_x, upper_x = x[0], x[-1]
    lower = _extrapolation(config, "lower extrapolation", x, y)
    lower = y[0] if lower is None else lower
    upper = _extrapolation(config, "upper extrapolation", x, y)
    upper = y[-1] if upper is None else upper
    x, y = add_point(x, y, (1.0 - DELTAX) * lower_x, lower)
    x, y = add_point(x, y, 0.0, lower)
    x, y = add_point(x, y, (1.0 + DELTAX) * upper_x, upper)
    x, y = add_point(x, y, 1.0e38, upper)
    return x, y


def add_points_quantum_yield(config, x, y):
    """quantum_yield.F90 add_points.  Its upper "boundary" value is read after the two lower points went in,
    so it is the original data's third-from-last value (data_parameter(nRows) on the grown array), kept."""
    x, y = list(x), list(y)
    n = len(x)
    lower_x, upper_x = x[0], x[-1]
    lower = _extrapolation(config, "lower extrapolation", x, y)
    lower = y[0] if lower is None else lower
    x, y = add_point(x, y, 0.0, lower)
    x, y = add_point(x, y, (1.0 - DELTAX) * lower_x, lower)
    upper = _extrapolation(config, "upper extrapolation", x, y)
    upper = y[n - 1] if upper is None else upper
    x, y = add_point(x, y, (1.0 + DELTAX) * upper_x, upper)
    x, y = add_point(x, y, 1.0e38, upper)
    return x, y


def _interpolator(entry):
    kind = (entry.get("interpolator") or {}).get("type", "conserving") if isinstance(entry, dict) else "conserving"
    if kind not in INTERPOLATORS:
        raise TuvxPortError(f"interpolator {kind!r} is not ported")
    return INTERPOLATORS[kind]


def _temperature_table(temperature, params):
    """The tint constructors' deltaT and the reversal of a decreasing temperature table (columns swapped)."""
    temperature = np.array(temperature, dtype=np.float64)
    params = np.array(params, dtype=np.float64)
    nparm = params.shape[1]
    delta = temperature[1:nparm] - temperature[:nparm - 1]
    if not np.all(delta > 0.0):
        if np.any(delta > 0.0):
            raise TuvxPortError("temperature array not monotonic")
        for lower in range(nparm // 2):
            upper = nparm - lower - 1
            temperature[lower], temperature[upper] = temperature[upper], temperature[lower]
            params[:, [lower, upper]] = params[:, [upper, lower]]
        delta = temperature[1:nparm] - temperature[:nparm - 1]
    return temperature, delta, params


def _on_grid(grid, config, entry, wavelength, params, add_points):
    """Each parameter column extended (add_points) and interpolated onto the grid's edges: (nbins, nparam)."""
    out = np.empty((grid.ncells, params.shape[1]), dtype=np.float64)
    interpolate = _interpolator(entry)
    for j in range(params.shape[1]):
        x, y = add_points(config, wavelength, params[:, j])
        out[:, j] = interpolate(grid.edge, x, y)
    return out


def refraction(wavelength, density):
    """cross_sections/o3_tint.F90 refraction: the refractive index of standard air at ``density``."""
    divisor = 2.69e19 * 273.15 / 288.15
    out = []
    for w in np.asarray(wavelength, dtype=np.float64):
        wrk = min(max(float(w), 200.0), 2000.0)
        sig = 1.0e3 / wrk
        sigsq = sig * sig
        wrk = 8342.13 + 2406030.0 / (130.0 - sigsq) + 15997.0 / (38.9 - sigsq)
        wrk = wrk * density / divisor
        out.append(1.0 + 1.0e-8 * wrk)
    return np.asarray(out, dtype=np.float64)


class BaseCrossSection:
    """cross_section.F90 base type from netcdf files (no "zero below/above", overrides or merged data)."""

    def __init__(self, config, grid, root):
        for key in ("override bands", "merge data", "data"):
            if key in config:
                raise TuvxPortError(f"base cross section {key!r} is not ported")
        self.parms = []
        for entry in config["netcdf files"]:
            if "zero below" in entry or "zero above" in entry:
                raise TuvxPortError("base cross section 'zero below/above' is not ported")
            wavelength, _temperature, params = read_parameters(Path(root) / entry["file path"], "cross_section_")
            if wavelength is None:
                self.parms.append(params)
                continue
            # parameters%array starts at zero and each file's columns are added in (0 + value); process_file
            # extends them with the FILE entry's extrapolation (the tint types read the cross section's)
            self.parms.append(0.0 + _on_grid(grid, entry, entry, wavelength, params,
                                             lambda _c, x, y: add_points_cross_section(entry, x, y)))

    def calculate(self, temperature_edges):
        n = np.asarray(temperature_edges).size
        return np.repeat(self.parms[0][:, 0][None, :], n, axis=0)


class O3TintCrossSection:
    """cross_sections/o3_tint.F90 (type "O3"): four files, refraction-corrected wavelengths, the band chosen by
    each bin's lower edge (file 3 below 185 nm, file 4 below 195, file 2 below 345, file 1 above)."""

    REFRACTION_DENSITY = 2.45e19

    def __init__(self, config, grid, root):
        files = config["netcdf files"]
        if len(files) < 2:
            raise TuvxPortError("O3 tint needs at least two netcdf files")
        self.grid = grid
        self.files = []
        for index, entry in enumerate(files, start=1):
            wavelength, temperature, params = read_parameters(Path(root) / entry["file path"], "cross_section_")
            if index == 2 and params.shape[1] < 4:
                raise TuvxPortError("O3 tint file 2 must have 4 or more parameters")
            if wavelength is None or temperature is None:
                raise TuvxPortError("O3 tint files need wavelengths and temperatures")
            wavelength = refraction(wavelength, self.REFRACTION_DENSITY) * wavelength
            delta = None
            if temperature.size > 1:
                temperature, delta, params = _temperature_table(temperature, params)
            array = _on_grid(grid, entry, entry, wavelength, params,
                             lambda _c, x, y: add_points_cross_section(config, x, y))
            self.files.append({"temperature": temperature, "delta": delta, "array": array})
        self.v185 = float(refraction([185.0], self.REFRACTION_DENSITY)[0] * 185.0)
        self.v195 = float(refraction([195.0], self.REFRACTION_DENSITY)[0] * 195.0)
        self.v345 = float(refraction([345.0], self.REFRACTION_DENSITY)[0] * 345.0)
        self.files[1]["array"][:, 3] = self.files[0]["array"][:, 0]

    def calculate(self, temperature_edges):
        temps = [float(t) for t in np.asarray(temperature_edges, dtype=np.float64)]
        edge = self.grid.edge
        out = np.zeros((len(temps), self.grid.ncells), dtype=np.float64)
        for k, model in enumerate(temps):
            for w in range(self.grid.ncells):
                lam = float(edge[w])
                if lam < self.v185:
                    f = 2
                elif lam < self.v195:
                    f = 3
                elif lam < self.v345:
                    f = 1
                else:
                    out[k, w] = 0.0 + self.files[0]["array"][w, 0]
                    continue
                table = self.files[f]
                data = table["temperature"]
                tadj = min(max(model, float(data[0])), float(data[-1]))
                t = next((i for i in range(1, data.size) if tadj <= data[i]), data.size)
                t -= 1
                tstar = (tadj - data[t]) / table["delta"][t]
                a = table["array"]
                out[k, w] = 0.0 + a[w, t] + tstar * (a[w, t + 1] - a[w, t])
        return out


class NO2TintCrossSection:
    """cross_sections/no2_tint.F90: linear in temperature between the file's tables, clamped to its range."""

    def __init__(self, config, grid, root):
        self.files = []
        for entry in config["netcdf files"]:
            wavelength, temperature, params = read_parameters(Path(root) / entry["file path"], "cross_section_")
            if params.shape[1] < 2 or temperature is None or temperature.size < max(2, params.shape[1]):
                raise TuvxPortError("NO2 tint needs two or more parameters and as many temperatures")
            temperature, delta, params = _temperature_table(temperature, params)
            array = (params if wavelength is None else
                     _on_grid(grid, entry, entry, wavelength, params,
                              lambda _c, x, y: add_points_cross_section(config, x, y)))
            self.files.append({"temperature": temperature, "delta": delta, "array": array})
        self.nbins = grid.ncells

    def calculate(self, temperature_edges):
        temps = [float(t) for t in np.asarray(temperature_edges, dtype=np.float64)]
        out = np.zeros((len(temps), self.nbins), dtype=np.float64)
        for table in self.files:
            data, a = table["temperature"], table["array"]
            for k, model in enumerate(temps):
                tadj = min(max(model, float(data[0])), float(data[-1]))
                # Fortran: tNdx exits where tadj <= T(tNdx), else ends at nTemp + 1; tNdx = min(nTemp, tNdx) - 1
                # (1-based), i.e. 0-based min(nTemp, i + 1) - 2 for the 0-based exit index i
                i = next((i for i in range(1, data.size) if tadj <= data[i]), data.size)
                t = min(data.size, i + 1) - 2
                tstar = (tadj - data[t]) / table["delta"][t]
                out[k] = out[k] + a[:, t] + tstar * (a[:, t + 1] - a[:, t])
        return out


class BaseQuantumYield:
    """quantum_yield.F90 base type with a ``constant value`` (the only form the configuration uses)."""

    def __init__(self, config, grid, root):
        for key in ("netcdf files", "override bands"):
            if key in config:
                raise TuvxPortError(f"base quantum yield {key!r} is not ported")
        self.value = float(config.get("constant value", 0.0))
        self.nbins = grid.ncells

    def calculate(self, temperature_edges):
        return np.full((np.asarray(temperature_edges).size, self.nbins), self.value, dtype=np.float64)


class O3QuantumYield:
    """quantum_yields/o3-o2_o1d.F90 (O(1D)); ``o3p`` is o3-o2_o3p.F90's one minus it.  glibc exp/pow words."""

    A = (0.8036, 8.9061, 0.1192)
    X = (304.225, 314.957, 310.737)
    OM = (5.576, 6.601, 2.187)

    def __init__(self, config, grid, root, *, o3p=False):
        self.mid = [float(w) for w in grid.mid]
        self.o3p = o3p

    def calculate(self, temperature_edges):
        import math
        a, x, om = self.A, self.X, self.OM
        out = np.zeros((np.asarray(temperature_edges).size, len(self.mid)), dtype=np.float64)
        for k, temp in enumerate(float(t) for t in np.asarray(temperature_edges, dtype=np.float64)):
            kt = 0.695 * temp
            q1 = 1.0
            q2 = math.exp(-825.518 / kt)
            qfac1 = q1 / (q1 + q2)
            qfac2 = q2 / (q1 + q2)
            t300 = temp / 300.0
            for w, lam in enumerate(self.mid):
                if lam <= 305.0:
                    value = 0.90
                elif 328.0 < lam <= 340.0:
                    value = 0.08
                elif 305.0 < lam <= 328.0:
                    y1 = (x[0] - lam) / om[0]
                    y1sq = y1 * y1
                    y2 = (x[1] - lam) / om[1]
                    y3 = (x[2] - lam) / om[2]
                    value = (0.0765 + a[0] * qfac1 * math.exp(-(y1sq * y1sq))
                             + a[1] * t300 * t300 * qfac2 * math.exp(-(y2 * y2))
                             + a[2] * math.pow(t300, 1.5) * math.exp(-(y3 * y3)))
                else:
                    value = 0.0
                out[k, w] = 1.0 - value if self.o3p else value
        return out


class NO2TintQuantumYield:
    """quantum_yields/no2_tint.F90: linear in temperature (not clamped), floored at zero."""

    def __init__(self, config, grid, root):
        self.files = []
        for name in config["netcdf files"]:
            wavelength, temperature, params = read_parameters(Path(root) / name, "quantum_yield_")
            if params.shape[1] < 2 or temperature is None or temperature.size < max(2, params.shape[1]):
                raise TuvxPortError("NO2 tint quantum yield needs two or more parameters and temperatures")
            temperature, delta, params = _temperature_table(temperature, params)
            array = (params if wavelength is None else
                     _on_grid(grid, {}, {}, wavelength, params,
                              lambda _c, x, y: add_points_quantum_yield(config, x, y)))
            self.files.append({"temperature": temperature, "delta": delta, "array": array})
        self.nbins = grid.ncells

    def calculate(self, temperature_edges):
        temps = [float(t) for t in np.asarray(temperature_edges, dtype=np.float64)]
        out = np.zeros((len(temps), self.nbins), dtype=np.float64)
        for table in self.files:
            data, a = table["temperature"], table["array"]
            for k, tadj in enumerate(temps):
                # Fortran: tNdx exits where tadj <= T(tNdx), else ends at nTemp + 1; tNdx = min(nTemp, tNdx) - 1
                # (1-based), i.e. 0-based min(nTemp, i + 1) - 2 for the 0-based exit index i
                i = next((i for i in range(1, data.size) if tadj <= data[i]), data.size)
                t = min(data.size, i + 1) - 2
                tstar = (tadj - data[t]) / table["delta"][t]
                out[k] = out[k] + a[:, t] + tstar * (a[:, t + 1] - a[:, t])
        return np.maximum(out, 0.0)


CROSS_SECTIONS = {"base": BaseCrossSection, "O3": O3TintCrossSection, "NO2 tint": NO2TintCrossSection}
QUANTUM_YIELDS = {"base": BaseQuantumYield, "NO2 tint": NO2TintQuantumYield,
                  "O3+hv->O2+O(1D)": O3QuantumYield,
                  "O3+hv->O2+O(3P)": lambda c, g, r: O3QuantumYield(c, g, r, o3p=True)}


class Photolysis:
    """Each configured reaction's cross section and quantum yield calculators on ``grid``."""

    def __init__(self, config, grid, *, root):
        self.reactions = []
        for reaction in config["photolysis"]["reactions"]:
            xs, qy = reaction["cross section"], reaction["quantum yield"]
            if xs["type"] not in CROSS_SECTIONS or qy["type"] not in QUANTUM_YIELDS:
                raise TuvxPortError(f"{reaction['name']}: {xs['type']!r} / {qy['type']!r} is not ported")
            self.reactions.append((reaction["name"], CROSS_SECTIONS[xs["type"]](xs, grid, root),
                                   QUANTUM_YIELDS[qy["type"]](qy, grid, root),
                                   bool(xs.get("apply O2 bands", False))))

    def tables(self, temperature_edges):
        """``{reaction: (cross section cm2, quantum yield)}``, each (height edge, wavelength bin)."""
        return {name: (xs.calculate(temperature_edges), qy.calculate(temperature_edges))
                for name, xs, qy, _bands in self.reactions}
