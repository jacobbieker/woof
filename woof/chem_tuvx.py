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


# ------------------------------------------------------------------------------------------------------------
# Spherical geometry (src/spherical_geometry.F90): the direct beam's path through each layer, per column.
# Columns are batched on a leading axis; every column's arithmetic is its own, in TUV-x's order, so a batch
# gives each column's words.  Indices follow TUV-x: an edge ``i`` counts from the top of the atmosphere
# (0 = top), a layer ``j`` (1-based in TUV-x, 0-based here) from the top as well.

#: src/constants.F90
PI = 3.1415926535898
RADIUS_KM = 6.371e3
LARGEST = 1.0e36
D2R = PI / 180.0


class Geometry:
    """spherical_geometry_t after set_parameters: ``nid`` (ncol, nlayer + 1), the number of layers the direct
    beam crosses to reach each edge (-1 when it never does), and ``dsdh`` (ncol, nlayer + 1, nlayer), its
    slant path over vertical depth in each layer crossed; ``sza`` (ncol,) in degrees."""

    def __init__(self, nid, dsdh, sza):
        self.nid, self.dsdh, self.sza = nid, dsdh, sza

    @property
    def nlayer(self):
        return self.dsdh.shape[2]


def _columns(values):
    a = np.asarray(values, dtype=np.float64)
    return a[None, :] if a.ndim == 1 else a


def spherical_geometry(edges_km, sza_deg) -> Geometry:
    """set_parameters (Dahlback and Stamnes 1991, appendix B) for each column's height edges (km, ascending,
    (ncol, nz) or (nz,)) and solar zenith angle (degrees)."""
    z = _columns(edges_km)
    zen = np.atleast_1d(np.asarray(sza_deg, dtype=np.float64))
    ncol, nz = z.shape
    if zen.shape != (ncol,):
        raise TuvxPortError(f"{zen.size} solar zenith angles for {ncol} columns")
    nlayer = nz - 1
    zenrad = zen * D2R
    re = RADIUS_KM + z[:, 0]
    zd = (z - z[:, :1])[:, ::-1]                 # zd(0:nlayer): heights above the surface, top first
    sinrad = np.sin(zenrad)
    twilight = zen > 90.0
    nid = np.zeros((ncol, nlayer + 1), dtype=np.int64)
    dsdh = np.zeros((ncol, nlayer + 1, nlayer), dtype=np.float64)
    for i in range(nlayer + 1):
        rpsinz = (re + zd[:, i]) * sinrad
        dark = twilight & (rpsinz < re)
        ident = np.where(twilight, -1, i)
        for j in range(1, nlayer + 1):           # the layer holding the screening height
            inside = (rpsinz < (zd[:, j - 1] + re)) & (rpsinz >= (zd[:, j] + re))
            ident = np.where(twilight & inside, j, ident)
        ident = np.where(dark, -1, ident)
        for j in range(1, nlayer + 1):
            active = j <= ident
            if not active.any():
                continue
            sm = np.where((j == ident) & (ident == i) & twilight, -1.0, 1.0)
            rj = re + zd[:, j - 1]
            rjp1 = re + zd[:, j]
            dhj = zd[:, j - 1] - zd[:, j]
            ga = np.maximum(0.0, rj * rj - rpsinz * rpsinz)
            gb = np.maximum(0.0, rjp1 * rjp1 - rpsinz * rpsinz)
            dsj = np.where((ident > i) & (j == ident), np.sqrt(ga), np.sqrt(ga) - sm * np.sqrt(gb))
            dsdh[:, i, j - 1] = np.where(active, dsj / dhj, 0.0)
        nid[:, i] = ident
    return Geometry(nid, dsdh, zen)


def air_mass(geometry: Geometry, column):
    """spherical_geometry_t%air_mass: ``(vertical, slant)`` columns of a species from its layer columns
    (ncol, nlayer + 1; the exo layer last, molecule cm-2).  ``vertical`` (ncol, nlayer) is the column above
    each layer's base, surface first; ``slant`` (ncol, nlayer + 1) the column along the direct beam above
    each edge, surface first (LARGEST where the beam does not reach)."""
    aircol = _columns(column)
    ncol, nz = aircol.shape
    nlayer = nz - 1
    if geometry.nlayer != nlayer:
        raise TuvxPortError(f"a {nz}-element column for a {geometry.nlayer}-layer geometry")
    vcol = np.empty((ncol, nlayer), dtype=np.float64)
    accum = aircol[:, nz - 1].copy()
    for ident in range(nlayer, 0, -1):
        accum = accum + aircol[:, ident - 1]
        vcol[:, ident - 1] = accum
    scol = np.empty((ncol, nz), dtype=np.float64)
    scol[:, nz - 1] = geometry.dsdh[:, 1, 0] * aircol[:, nz - 1]
    for ident in range(1, nlayer + 1):
        nid = geometry.nid[:, ident]
        single = np.minimum(nid, ident)
        accum = scol[:, nz - 1].copy()
        for j in range(1, nlayer + 1):
            d = geometry.dsdh[:, ident, j - 1]
            a = aircol[:, nz - 1 - j]
            term = np.where(j <= single, a * d, 2.0 * a * d)
            accum = np.where(j <= nid, accum + term, accum)
        scol[:, nz - 1 - ident] = np.where(nid < 0, LARGEST, accum)
    return vcol, scol


def exo_layer_columns(layer_columns):
    """A from-host profile's exo_layer_dens_ without a scale height or exo density: the host's layer columns
    and a zero exo layer (src/profiles/from_host.F90 update)."""
    lay = _columns(layer_columns)
    return np.concatenate([lay, np.zeros((lay.shape[0], 1))], axis=1)


# ------------------------------------------------------------------------------------------------------------
# O2 Lyman-alpha and Schumann-Runge bands (src/la_sr_bands.F90): the O2 optical depth (radiative transfer) and
# the O2 photolysis cross section in those bins, from the O2 slant column along the direct beam.  Level and
# layer arrays are surface first, columns on the leading axis.

#: the Lyman-alpha and Schumann-Runge wavelength grids [nm]
LA_EDGES = (121.4, 121.9)
SRB_EDGES = (175.4, 177.0, 178.6, 180.2, 181.8, 183.5, 185.2, 186.9, 188.7, 190.5, 192.3, 194.2, 196.1, 198.0,
             200.0, 202.0, 204.1, 206.2)
NSRB = len(SRB_EDGES) - 1
NPOLY = 20
PRECIS = 1.0e-7
#: lymana_OD / lymana_xs parameters
_LA_B = (6.8431e-01, 2.29841e-01, 8.65412e-02)
_LA_C = (8.22114e-21, 1.77556e-20, 8.22112e-21)
_LA_D = (6.0073e-21, 4.28569e-21, 1.28059e-20)
_LA_E = (8.21666e-21, 1.63296e-20, 4.85121e-17)
_LA_XSMIN, _LA_TINY, _LA_EXP_LIM, _LA_LARGE_OD = 1.0e-20, 1.0e-100, 100.0e8, 1000.0
#: schum_OD / schum_xs: the cross sections above the Chebyshev range
_SRB_XSLOD = (6.2180730e-21, 5.8473627e-22, 5.6996334e-22, 4.5627094e-22, 1.7668250e-22, 1.1178808e-22,
              1.2040544e-22, 4.0994668e-23, 1.8450616e-23, 1.5639540e-23, 8.7961075e-24, 7.6475608e-24,
              7.6260556e-24, 7.5565696e-24, 7.6334338e-24, 7.4371992e-24, 7.3642966e-24)


def _exp38():
    """``exp(38._dk)``, a constant gfortran folds with correct rounding."""
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    return float(Decimal(38).exp())


SRB_COLMIN = _exp38()


def read_srb_parameters(path):
    """init_srb_xs: (AC, BC), each (NPOLY, NSRB), from two blocks of format (/) then 20 records of
    17(E23.14, 1x)."""
    lines = Path(path).read_text().splitlines()

    def block(start):
        rows = []
        for line in lines[start:start + NPOLY]:
            rows.append([float(line[24 * k:24 * k + 23].replace("D", "E")) for k in range(NSRB)])
        return np.asarray(rows, dtype=np.float64)

    return block(2), block(2 + NPOLY + 2)


def _sum3(tau, coefficients):
    """dot_product of three: 0 + t1 c1 + t2 c2 + t3 c3, in order."""
    total = tau[0] * coefficients[0]
    total = total + tau[1] * coefficients[1]
    return total + tau[2] * coefficients[2]


class LaSrBands:
    """la_sr_bands_t on a wavelength grid (TUV-x's O2 profile form: no "O2 estimate")."""

    def __init__(self, config, grid: WavelengthGrid, *, root):
        if "O2 estimate" in config:
            raise TuvxPortError("la_sr_bands 'O2 estimate' is not ported")
        edge = grid.edge
        self.has_la = bool(edge[0] <= LA_EDGES[0] and edge[-1] >= LA_EDGES[-1])
        self.has_srb = bool(edge[0] <= SRB_EDGES[0] and edge[-1] >= SRB_EDGES[-1])
        self.ila = self.isrb = None
        if self.has_la:
            self.ila = self._start(edge, LA_EDGES, "Lyman alpha")
        if self.has_srb:
            self.isrb = self._start(edge, SRB_EDGES, "Schumann-Runge")
            self.ac, self.bc = read_srb_parameters(Path(root) / config["cross section parameters file"])

    @staticmethod
    def _start(edge, band, what):
        """The 0-based grid edge where the band starts; its edges must follow within 10 * PRECIS."""
        hits = [i for i, e in enumerate(edge) if abs(float(e) - band[0]) < 10.0 * PRECIS]
        if not hits:
            raise TuvxPortError(f"{what} grid mis-match: below 205.8 nm only the pre-specified grid is permitted")
        start = hits[0]
        for k, w in enumerate(band[1:], start=1):
            if start + k >= edge.size or abs(float(edge[start + k]) - w) > 10.0 * PRECIS:
                raise TuvxPortError(f"{what} grid mismatch at {w} nm")
        return start

    @property
    def bins(self):
        """The 0-based wavelength bins the bands set."""
        out = []
        if self.has_la:
            out.append(self.ila)
        if self.has_srb:
            out.extend(range(self.isrb, self.isrb + NSRB))
        return out

    # -- the shared front of la_srb_OD / la_srb_xs
    @staticmethod
    def _secchi(air_vcol, air_scol):
        nzm1 = air_vcol.shape[1]
        secchi = np.empty_like(air_scol)
        ok = air_scol[:, :nzm1] <= 0.1 * LARGEST
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            secchi[:, :nzm1] = np.where(ok, air_scol[:, :nzm1] / air_vcol, 2.0)
        secchi[:, nzm1] = secchi[:, nzm1 - 1]
        return secchi

    @staticmethod
    def _rm(o2col, coeff_c, coeff_b):
        taus = []
        for c in coeff_c:
            sigma = c * o2col
            taus.append(np.where(sigma < _LA_EXP_LIM, np.exp(-sigma), 0.0))
        return _sum3(taus, coeff_b)

    def _lyman_alpha_od(self, o2col, secchi):
        rm = self._rm(o2col, _LA_C, _LA_B)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(rm)
            od = lr[:, 1:] / secchi[:, 1:] - lr[:, :-1] / secchi[:, :-1]
        good = (rm[:, :-1] > _LA_TINY) & (rm[:, 1:] > 0.0)
        return np.where(good, od, _LA_LARGE_OD)

    def _lyman_alpha_xs(self, o2col):
        rm = self._rm(o2col, _LA_C, _LA_B)
        ro2 = self._rm(o2col, _LA_E, _LA_D)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = ro2 / rm
        out = np.where((rm > _LA_TINY) & (ro2 > _LA_TINY), ratio, _LA_XSMIN)
        # the top edge takes the ratio whenever rm is above the floor (ro2 is not checked there)
        out[:, -1] = np.where(rm[:, -1] > _LA_TINY, ratio[:, -1], _LA_XSMIN)
        return out

    def _chebyshev(self, coefficients, x):
        """chebyshev_evaluation for every band at once: coefficients (NPOLY, NSRB), x (ncol,)."""
        if np.any((x < 38.0) | (x > 56.0)):
            raise TuvxPortError("x out-of-bounds for Chebyshev polynomial")
        y = ((2.0 * x - (38.0 + 56.0)) / (56.0 - 38.0))[:, None]
        y2 = 2.0 * y
        di = np.zeros((x.size, NSRB))
        di1 = np.zeros((x.size, NSRB))
        for i in range(NPOLY - 1, 0, -1):
            dtemp = di
            di = y2 * di - di1 + coefficients[i]
            di1 = dtemp
        return y * di - di1 + 0.5 * coefficients[0]

    def _srb_xs(self, o2col, tlev):
        """schum_xs: the effective cross sections (ncol, nz, NSRB) and the clamped columns."""
        ncol, nz = o2col.shape
        xs = np.broadcast_to(np.asarray(_SRB_XSLOD), (ncol, nz, NSRB)).copy()
        o2col1 = np.maximum(o2col, SRB_COLMIN)
        ktop = np.full(ncol, nz)                 # 1-based, as TUV-x
        kbot = np.zeros(ncol, dtype=np.int64)
        for k in range(1, nz + 1):
            low = o2col[:, k - 1] < SRB_COLMIN
            ktop = np.where(low, np.minimum(k - 1, ktop), ktop)
            with np.errstate(divide="ignore"):
                x = np.log(o2col1[:, k - 1])
            above = ~low & (x > 56.0)
            kbot = np.where(above, k, kbot)
            fit = ~low & ~above
            if fit.any():
                xf = x[fit]
                a = self._chebyshev(self.ac, xf)
                b = self._chebyshev(self.bc, xf)
                xs[fit, k - 1] = np.exp(a * (tlev[fit, k - 1][:, None] - 220.0) + b)
        kbot = np.where(kbot == nz, nz - 1, kbot)
        for c in range(ncol):
            if kbot[c] > 0:
                xs[c, :kbot[c]] = xs[c, kbot[c]]
            if ktop[c] < nz:
                xs[c, ktop[c]:] = xs[c, ktop[c] - 1]
        return xs, o2col1

    def _srb_od(self, o2col, tlev, secchi):
        xs, o2col1 = self._srb_xs(o2col, tlev)
        nz = o2col.shape[1]
        norm = 1.0 / float(nz - 1)
        lo, hi = xs[:, :-1], xs[:, 1:]
        clo, chi = o2col1[:, :-1, None], o2col1[:, 1:, None]
        near = np.abs(1.0 - o2col1[:, 1:] / o2col1[:, :-1]) <= 2.0 * PRECIS
        with np.errstate(divide="ignore", invalid="ignore"):
            far = np.abs((hi * chi - lo * clo) / (1.0 + np.log(hi / lo) / np.log(chi / clo)))
            far = 2.0 * far / (secchi[:, :-1] + secchi[:, 1:])[:, :, None]
        return np.where(near[:, :, None], hi * chi * norm, far)

    def optical_depth(self, air_vcol, air_scol, o2_scol, temperature_edges):
        """la_srb_OD: ``{bin: O2 optical depth (ncol, nlayer)}`` for the band bins (0-based)."""
        secchi = self._secchi(_columns(air_vcol), _columns(air_scol))
        o2col, tlev = _columns(o2_scol), _columns(temperature_edges)
        out = {}
        if self.has_la:
            out[self.ila] = self._lyman_alpha_od(o2col, secchi)
        if self.has_srb:
            od = self._srb_od(o2col, tlev, secchi)
            for k in range(NSRB):
                out[self.isrb + k] = od[:, :, k]
        return out

    def cross_section(self, air_vcol, air_scol, o2_scol, temperature_edges):
        """la_srb_xs: ``{bin: O2 cross section (ncol, nedge)}`` for the band bins (0-based)."""
        o2col, tlev = _columns(o2_scol), _columns(temperature_edges)
        out = {}
        if self.has_la:
            out[self.ila] = self._lyman_alpha_xs(o2col)
        if self.has_srb:
            xs, _ = self._srb_xs(o2col, tlev)
            for k in range(NSRB):
                out[self.isrb + k] = xs[:, :, k]
        return out
