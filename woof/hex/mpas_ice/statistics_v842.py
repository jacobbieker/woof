"""MALI v8.4.2's global statistics: ``li_compute_statistics``.

``li_core_run`` calls it after every ``config_stats_interval``-th step (when
that is positive), and ``li_core_initial_solve`` once more on a cold start
with ``config_write_stats_on_startup`` (timestep 0, after the initial
velocity solve and diagnostics). It changes one word of the state:
``li_calculate_layerThickness`` from ``thickness`` (as the step does). The
rest is a report to the log:

- the ice area, volume and energy (``rhoi cp_ice`` times the
  layer-thickness-weighted temperature) over the owned ice cells, and the mean
  thickness and temperature formed from them;
- the largest and smallest column thickness, temperature, normal velocity
  magnitude (every interface) and basal normal velocity magnitude, each
  with the cell or edge ID (and level) it is first found at;
- the column at ``config_stats_cell_ID``: upper surface, thickness, bed,
  surface mass balance, surface and basal temperature, and the normal
  velocity on the cell's first edge and the temperature on every level
  (zeros when no owned cell has that ID).

:func:`compute_statistics` reproduces it on the host at RKIND, one correctly
rounded operation at a time in the Fortran's order: a column's ``sum`` from
level 1 down, the cell sums from cell 1 up (``np.add.accumulate``, which
adds in sequence; never a pairwise sum), the extrema with their first
locations by strict comparison from the Fortran's sentinels (``1.0e34``,
``huge``), and x86-64 gfortran's ``MAX`` (``maxss``: the first argument only when it
is the larger). The report's lines are formatted as gfortran writes them (``A``,
``I``, ``F`` and ``E`` edit descriptors). On one rank every reduction is
the identity; :attr:`Statistics.real_words` and :attr:`Statistics.int_words`
are the words each reduction and broadcast returns, in call order, as the
oracle records them (``tools/mali_oracle/stubs.F90``).

One v8.4.2 quirk has a named opt-in correction
(``woof_statistics_extrema``): the extrema take every owned cell and edge,
ice or not (only the sums are masked), so the minimum thickness is 0 and
the extreme temperatures and velocities are those of the ice-free cells'
words whenever any cell is ice free. 'v842' (the default) keeps the
model's report; 'ice' takes the extrema over the ice cells (for velocity,
the ice edges) alone, and reports 0 at ID 0 when there are none.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from woof.hex.errors import ConfigurationRefusal

from . import rkind

SOURCE_FILES = ("src/core_landice/mode_forward/mpas_li_statistics.F:76-660,684-922",
                "src/core_landice/mode_forward/mpas_li_core.F:385-392,727-731",
                "src/core_landice/shared/mpas_li_setup.F:553-609")
#: woof_statistics_extrema: 'v842' (every owned cell and edge) or 'ice' (the ice ones)
STATISTICS_EXTREMA = ("v842", "ice")
#: li_mask_ValueIce
MASK_ICE = 32
#: the separator line a report starts with
RULE = "-" * 60


@dataclass(frozen=True)
class Statistics:
    """One ``li_compute_statistics`` call: the step it reports
    (``itimestep``), the words its reductions return in call order, and
    the log lines (trailing blanks removed)."""
    itimestep: int
    real_words: np.ndarray
    int_words: np.ndarray
    lines: tuple[str, ...]

    @property
    def report(self) -> str:
        return "\n".join(self.lines)


def check_config(config: dict) -> None:
    value = config.get("woof_statistics_extrema", "v842")
    if value not in STATISTICS_EXTREMA:
        raise ConfigurationRefusal(knob="woof_statistics_extrema", value=value,
                                   reason="not an option of this port",
                                   declaration="woof_statistics_extrema = 'v842' | 'ice'")


def enabled(config: dict) -> bool:
    return int(config.get("config_stats_interval", 0)) > 0 or bool(config.get("config_write_stats_on_startup",
                                                                               False))


def due(config: dict, timestep_number: int) -> bool:
    """``li_core_run``'s test after a step."""
    interval = int(config.get("config_stats_interval", 0))
    return interval > 0 and timestep_number % interval == 0


# ---------------------------------------------------------------- RKIND literals and gfortran's edit descriptors


def literal(text: str, real):
    """A Fortran real literal at RKIND: the decimal rounded once (to nearest,
    ties to even), as gfortran reads it."""
    exact = Fraction(text)
    near = real(float(exact))
    best = near
    for candidate in (np.nextafter(near, real(-np.inf)), np.nextafter(near, real(np.inf))):
        if not np.isfinite(candidate):
            continue
        d_best, d_cand = abs(Fraction(float(best)) - exact), abs(Fraction(float(candidate)) - exact)
        if d_cand < d_best or (d_cand == d_best and int(np.asarray(candidate).view(
                np.uint32 if real is np.float32 else np.uint64)) % 2 == 0):
            best = candidate
    return real(best)


def _special(x: float, w: int) -> str | None:
    if np.isnan(x):
        return "NaN".rjust(w) if w >= 3 else "*" * w
    if np.isinf(x):
        text = ("-" if x < 0 else "") + ("Infinity" if w >= 8 + (x < 0) else "Inf")
        return text.rjust(w) if len(text) <= w else "*" * w
    return None


def fmt_a(text: str, w: int) -> str:
    """``Aw`` on output: right-justified, or the leftmost ``w`` characters."""
    return text[:w] if len(text) >= w else text.rjust(w)


def fmt_i(n: int, w: int) -> str:
    text = str(int(n))
    return text.rjust(w) if len(text) <= w else "*" * w


def fmt_f(x, w: int, d: int) -> str:
    """``Fw.d`` on output, as gfortran writes it: the value rounded to ``d``
    places (to nearest, ties to even, on its exact binary value), the sign
    of a negative value (and of -0) kept, the leading 0 dropped when the
    field is otherwise too narrow, asterisks when it still is."""
    x = float(x)
    special = _special(x, w)
    if special is not None:
        return special
    body = f"{abs(x):.{d}f}"
    sign = "-" if np.signbit(x) else ""
    text = sign + body
    if len(text) > w and body.startswith("0."):
        text = sign + body[1:]
    return text.rjust(w) if len(text) <= w else "*" * w


def fmt_e(x, w: int, d: int) -> str:
    """``Ew.d`` on output, as gfortran writes it: ``0.ddd...E+xx`` (``d``
    significant digits, rounded to nearest, ties to even), ``+xxx`` without
    the E for a three-digit exponent."""
    x = float(x)
    special = _special(x, w)
    if special is not None:
        return special
    sign = "-" if np.signbit(x) else ""
    if x == 0.0:
        digits, exponent = "0" * d, 0
    else:
        mantissa, exp10 = f"{abs(x):.{d - 1}e}".split("e")
        digits, exponent = mantissa.replace(".", ""), int(exp10) + 1
    if abs(exponent) <= 99:
        tail = f"E{'-' if exponent < 0 else '+'}{abs(exponent):02d}"
    elif abs(exponent) <= 999:
        tail = f"{'-' if exponent < 0 else '+'}{abs(exponent):03d}"
    else:
        return "*" * w
    text = f"{sign}0.{digits}{tail}"
    if len(text) > w:
        text = f"{sign}.{digits}{tail}"
    return text.rjust(w) if len(text) <= w else "*" * w


# ---------------------------------------------------------------- the reductions (host, RKIND, the Fortran's order)


def _column_sums(field: np.ndarray, real) -> np.ndarray:
    """``sum(field(:,i))`` for every i: level 1 first, from 0."""
    acc = np.zeros(field.shape[0], real)
    for k in range(field.shape[1]):
        acc = acc + field[:, k]
    return acc


def _sequential_sum(terms: np.ndarray, real):
    """``s = 0; do i: s = s + terms(i)``."""
    return np.add.accumulate(np.concatenate([np.zeros(1, real), np.asarray(terms, real)]))[-1]


def _first_extreme(values: np.ndarray, init, larger: bool, eligible: np.ndarray | None = None):
    """The running extremum of ``values`` (flattened in the Fortran's
    element order) from ``init`` by strict comparison, and its 1-based
    position (0 when no value beats ``init``). NaN never compares true;
    with ``eligible`` the other elements are skipped."""
    flat = np.asarray(values).ravel()
    usable = ~np.isnan(flat)
    if eligible is not None:
        usable &= np.asarray(eligible).ravel()
    if not usable.any():
        return init, 0
    fill = -np.inf if larger else np.inf
    probe = np.where(usable, flat, fill)
    at = int(np.argmax(probe) if larger else np.argmin(probe))
    best = flat[at]
    if (best > init) if larger else (best < init):
        return best, at + 1
    return init, 0


def _fmax(a, b):
    """x86-64 gfortran's ``MAX(a, b)`` (``maxss``/``maxsd``): ``a`` only when
    it is the larger, so a tie of signed zeros (or a NaN) gives ``b``."""
    return a if a > b else b


def compute_statistics(*, mesh: dict, geometry: dict, thermal: dict, velocity: dict, config: dict,
                       itimestep: int, xtime: str, apply_layer_thickness=None) -> Statistics:
    """``li_compute_statistics(domain, itimestep)`` on one block that owns
    every cell and edge. The arrays are host or device arrays (copied to
    the host); ``geometry["layerThickness"]`` is recomputed in place first
    (``apply_layer_thickness``, the device kernel, when given; else on the
    host array)."""
    check_config(config)
    host = _host
    thickness = host(geometry["thickness"])
    precision = rkind.precision_of(thickness)
    real = rkind.PRECISIONS[precision]
    if apply_layer_thickness is not None:
        apply_layer_thickness()
    else:
        geometry["layerThickness"][...] = thickness[:, None] * host(mesh["layerThicknessFractions"])[None, :]
    layer_thickness = host(geometry["layerThickness"])
    cell_mask, edge_mask = host(geometry["cellMask"]), host(geometry["edgeMask"])
    area = host(mesh["areaCell"])
    temperature, normal_velocity = host(thermal["temperature"]), host(velocity["normalVelocity"])
    cell_id, edge_id = host(mesh["indexToCellID"]), host(mesh["indexToEdgeID"])
    n_cells, n_levels = layer_thickness.shape
    ice_cell = ((cell_mask & MASK_ICE) != 0).astype(np.int32)
    ice_edge = ((edge_mask & MASK_ICE) != 0).astype(np.int32)
    masked = config.get("woof_statistics_extrema", "v842") == "ice"
    rhoi = real(config["config_ice_density"])
    cp_ice, kelvin, scyr = real(2009.0), literal("273.15", real), real(31536000.0)
    big, huge = literal("1.0e34", real), real(np.finfo(real).max)
    stats_cell = int(config["config_stats_cell_ID"])
    reals: list = []
    ints: list = []

    def at(ids: np.ndarray, position: int, what: str) -> int:
        if position == 0:
            if masked:
                return 0
            raise ValueError(f"li_compute_statistics: no {what} beats the Fortran's sentinel (a NaN state); "
                             "v8.4.2 would index position 0")
        return int(ids[position - 1])

    weight = ice_cell.astype(real) * area   # real(mask(i),RKIND) * areas(i)
    cells_ok = ice_cell.astype(bool) if masked else None

    # thickness: the column sums' extrema (li_compute_field_local_stats with the vertical-sum locations)
    column = _column_sums(layer_thickness, real)
    local_min, loc_min = _first_extreme(column, big, False, cells_ok)
    local_max, loc_max = _first_extreme(column, -big, True, cells_ok)
    thickness_min, thickness_min_cell = huge, 0
    thickness_max, thickness_max_cell = -huge, 0
    if local_min < thickness_min:
        thickness_min, thickness_min_cell = local_min, at(cell_id, loc_min, "column thickness")
    if local_max > thickness_max:
        thickness_max, thickness_max_cell = local_max, at(cell_id, loc_max, "column thickness")
    if masked and loc_min == 0:
        thickness_min = real(0.0)
    if masked and loc_max == 0:
        thickness_max = real(0.0)

    # area: workLevelCell is 1 on level 1 and 0 below, so the column sum is 1
    ones_column = _column_sums(np.concatenate([np.ones((n_cells, 1), real), np.zeros((n_cells, n_levels - 1), real)],
                                              axis=1), real)
    ice_area = real(0.0) + _sequential_sum(weight * ones_column, real)
    # volume: workLevelCell is 1 everywhere
    ice_volume = real(0.0) + _sequential_sum(weight * _column_sums(layer_thickness * real(1.0), real), real)

    # temperature: the extrema over every level of every cell, with their cell and level
    eligible = np.repeat(cells_ok[:, None], n_levels, 1) if masked else None
    t_min, t_min_at = _first_extreme(temperature, big, False, eligible)
    t_max, t_max_at = _first_extreme(temperature, -big, True, eligible)
    temperature_min, temperature_max = huge, -huge
    t_min_cell = t_min_level = t_max_cell = t_max_level = 0
    if t_min < temperature_min:
        temperature_min = t_min
        t_min_cell, t_min_level = at(cell_id, (t_min_at - 1) // n_levels + 1 if t_min_at else 0, "temperature"), \
            ((t_min_at - 1) % n_levels + 1 if t_min_at else 0)
    if t_max > temperature_max:
        temperature_max = t_max
        t_max_cell, t_max_level = at(cell_id, (t_max_at - 1) // n_levels + 1 if t_max_at else 0, "temperature"), \
            ((t_max_at - 1) % n_levels + 1 if t_max_at else 0)
    if masked and t_min_at == 0:
        temperature_min = real(0.0)
    if masked and t_max_at == 0:
        temperature_max = real(0.0)
    energy_local = _sequential_sum(weight * _column_sums(layer_thickness * temperature, real), real)
    ice_energy = real(0.0) + energy_local * rhoi * cp_ice

    # velocity: every interface of every owned edge, by magnitude
    n_interfaces = normal_velocity.shape[1]

    def speed_max(field: np.ndarray, levels: int, what: str):
        ok = np.repeat(ice_edge.astype(bool)[:, None], levels, 1) if masked else None
        lo, lo_at = _first_extreme(field, big, False, ok)
        hi, hi_at = _first_extreme(field, -big, True, ok)
        absolute = _fmax(hi, -lo)
        if hi > abs(lo):
            position = hi_at
        else:
            position = lo_at
        edge = at(edge_id, (position - 1) // levels + 1 if position else 0, what)
        level = (position - 1) % levels + 1 if position else 0
        if masked and position == 0:
            absolute = real(0.0)
        return absolute, edge, level

    velocity_max, velocity_max_edge, velocity_max_level = -huge, 0, 0
    absolute, edge, level = speed_max(normal_velocity, n_interfaces, "normal velocity")
    if absolute > velocity_max:
        velocity_max, velocity_max_edge, velocity_max_level = absolute, edge, level
    basal_max, basal_max_edge = -huge, 0
    absolute, edge, _ = speed_max(normal_velocity[:, n_interfaces - 1:], 1, "basal normal velocity")
    if absolute > basal_max:
        basal_max, basal_max_edge = absolute, edge

    # the diagnostic column (the last owned cell with the ID; v8.4.2's IDs are unique)
    diag = dict(cell=0, upper=real(0.0), thick=real(0.0), bed=real(0.0), smb=real(0.0), sfc_t=real(0.0),
                basal_t=real(0.0))
    diag_speed = np.zeros(n_interfaces, real)
    diag_temperature = np.zeros(n_levels, real)
    hits = np.flatnonzero(cell_id == stats_cell)
    if hits.size:
        i = int(hits[-1])
        edges_on_cell = host(mesh["edgesOnCell"])
        diag.update(cell=i + 1, upper=host(geometry["upperSurface"])[i], thick=thickness[i],
                    bed=host(geometry["bedTopography"])[i],
                    smb=host(geometry["sfcMassBal"])[i] * scyr / real(1000.0),
                    sfc_t=host(thermal["surfaceTemperature"])[i], basal_t=host(thermal["basalTemperature"])[i])
        diag_speed = normal_velocity[int(edges_on_cell[i, 0]) - 1, :] * scyr
        diag_temperature = temperature[i, :].copy()

    # the reductions, in call order (one rank: each returns its argument)
    reals += [ice_area, ice_volume, ice_energy]
    reals += [thickness_min, thickness_min]
    ints += [thickness_min_cell]
    reals += [thickness_max, thickness_max]
    ints += [thickness_max_cell]
    reals += [temperature_min, temperature_min]
    ints += [t_min_cell, t_min_level]
    reals += [temperature_max, temperature_max]
    ints += [t_max_cell, t_max_level]
    reals += [velocity_max, velocity_max]
    ints += [velocity_max_edge, velocity_max_level]
    reals += [basal_max, basal_max]
    ints += [basal_max_edge]
    ints += [diag["cell"], 0, 0]   # the cell, its block (localBlockID) and rank
    reals += [diag["upper"], diag["thick"], diag["bed"], diag["smb"], diag["sfc_t"], diag["basal_t"]]
    reals += list(diag_speed) + list(diag_temperature)

    thickness_mean = ice_volume / ice_area if ice_area > real(0.0) else real(0.0)
    temperature_mean = (ice_energy / (ice_volume * rhoi * cp_ice) if ice_volume > real(0.0) else real(0.0))
    sigma = host(mesh["layerCenterSigma"])
    lines = [
        " ", RULE, " ",
        fmt_a("Global statistics: time =", 25) + fmt_a(xtime.rstrip(), 20),
        fmt_a("               timestep =", 25) + fmt_i(itimestep, 8),
        " ",
        "Total ice area (km^2)           " + fmt_e(ice_area * literal("1.0e-6", real), 24, 16),
        "Total ice volume (km^3)         " + fmt_e(ice_volume * literal("1.0e-9", real), 24, 16),
        "Total ice energy (J)            " + fmt_e(ice_energy, 24, 16),
        "Max thickness (m), cell         " + fmt_f(thickness_max, 24, 16) + fmt_i(thickness_max_cell, 8),
        "Min thickness (m), cell         " + fmt_f(thickness_min, 24, 16) + fmt_i(thickness_min_cell, 8),
        "Mean thickness (m)              " + fmt_f(thickness_mean, 24, 16),
        "Max temperature (C), cell, level" + fmt_f(temperature_max - kelvin, 24, 16) + fmt_i(t_max_cell, 8)
        + fmt_i(t_max_level, 4),
        "Min temperature (C), cell, level" + fmt_f(temperature_min - kelvin, 24, 16) + fmt_i(t_min_cell, 8)
        + fmt_i(t_min_level, 4),
        "Mean temperature (C)            " + fmt_f(temperature_mean - kelvin, 24, 16),
        "Max velocity (m/yr), edge, level" + fmt_f(velocity_max * scyr, 24, 16) + fmt_i(velocity_max_edge, 8)
        + fmt_i(velocity_max_level, 4),
        "Max basal velo (m/yr), edge     " + fmt_f(basal_max * scyr, 24, 16) + fmt_i(basal_max_edge, 8),
        " ",
        "Column diagnostics: cell ID = " + fmt_i(stats_cell, 6),
        "Local cell ID, block, proc =  " + fmt_i(diag["cell"], 6) + fmt_i(0, 6) + fmt_i(0, 6),
        " ",
        "Upper surface (m)        " + fmt_f(diag["upper"], 24, 16),
        "Thickness (m)            " + fmt_f(diag["thick"], 24, 16),
        "Bed topography (m)       " + fmt_f(diag["bed"], 24, 16),
        "Sfc mass balance (m/yr)  " + fmt_f(diag["smb"], 24, 16),
        " ",
        "Sigma        Ice speed (m/yr)       Ice temperature (C)",
        fmt_f(real(0.0), 6, 4) + fmt_a("------", 25) + fmt_f(diag["sfc_t"] - kelvin, 24, 16),
        *(fmt_f(sigma[k], 6, 4) + fmt_f(diag_speed[k], 25, 16) + fmt_f(diag_temperature[k] - kelvin, 24, 16)
          for k in range(n_levels)),
        fmt_f(real(1.0), 6, 4) + fmt_a("------", 25) + fmt_f(diag["basal_t"] - kelvin, 24, 16),
        " ",
    ]
    return Statistics(itimestep=int(itimestep), real_words=np.array(reals, real),
                      int_words=np.array(ints, np.int32), lines=tuple(line.rstrip() for line in lines))


def _host(array) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    get = getattr(array, "get", None)
    return np.asarray(get() if get is not None else array)


def reports(lines) -> list[str]:
    """The statistics reports in a log: each from its rule line to its basal
    temperature line (``1.0000``), joined by newlines."""
    lines = [line.rstrip() for line in lines]
    out, i = [], 0
    while i < len(lines):
        if lines[i] == RULE and i + 2 < len(lines) and lines[i + 2].startswith("Global statistics: time ="):
            j = i
            while j < len(lines) and not lines[j].startswith(fmt_f(1.0, 6, 4)):
                j += 1
            out.append("\n".join(lines[i:j + 1]))
            i = j + 1
        else:
            i += 1
    return out
