"""The wind-energy model column: hub-resolving, model top at most 20 km.

``woof energy plan --topology wrf-nests|wrf-tiles`` reads the explicit-column
flags ``woof domain`` takes (``--p-top-pa``, ``--eta-levels``, ``--dzbot``,
``--dzstretch-s``, ``--dzstretch-u``, ``--max-dz``).  When any of them is
given the planner builds its column through :func:`wind_energy_column`; when
none is, the planner is unchanged (``--nz`` alone keeps its old meaning).

What is pinned here:

* **Model top.**  Refused above :data:`MAX_MODEL_TOP_M` (20 km), judged as
  the geopotential altitude of ``p_top`` in the 1976 US / ICAO standard
  atmosphere (:func:`isa_height_m`).  That puts the shallowest admissible
  top at :func:`isa_pressure_pa` ``(20 km)`` = 5474.9 Pa, so WRF's own
  50 hPa (~20.6 km) is refused here and the preset default is
  :data:`DEFAULT_P_TOP_PA` = 5500 Pa (~19.97 km).
* **Levels.**  WRF 4.6.1 ``compute_eta`` (``auto_levels_opt = 2``, through
  :func:`woof.ingest.eta.generate_wrf_eta` and its native bridge) with the
  preset controls :data:`DEFAULT_DZBOT_M` 12 m, :data:`DEFAULT_DZSTRETCH_S`
  1.15, :data:`DEFAULT_DZSTRETCH_U` 1.1, :data:`DEFAULT_MAX_DZ_M` 1000 m at
  :data:`DEFAULT_NZ` 70 mass levels; any of them may be overridden, or the
  whole ladder replaced with ``--eta-levels``.
* **Hindcast levels.**  ``woof energy hindcast`` runs :data:`HINDCAST_NZ`
  (49) levels by default: the smallest count whose column keeps the rotor
  layers above AND no layer deeper than ``max_dz``
  (:func:`lean_column_refusal`, :func:`smallest_column_nz`), 70% of the
  70-level column's cells at the same rotor resolution.
* **Rotor layers.**  At least :data:`DEFAULT_MIN_ROTOR_LAYERS` (8) whole
  model layers below :data:`ROTOR_TOP_M` (250 m) in the reference column, or
  the column is refused.  The defaults give 10 (12, 26, 41, 59, 80, 104,
  131, 162, 198, 239 m), WRF's own ``dzstretch_s = 1.3`` gives 7.
* **Reference column.**  Heights are for a flat column whose surface sits
  at the WRF reference pressure P0 = 1000 hPa, where the hybrid coordinate
  reduces exactly to ``p = eta (P0 - p_top) + p_top`` for every
  ``hybrid_opt``; pressure becomes height through the standard atmosphere
  and is reported above that surface (AGL).  Real columns differ with
  temperature and terrain by a few per cent; the numbers are a design
  statement, not a forecast.
* **Physics.**  The level count is held to the dynamical core's 256 by
  :func:`woof.domain_wizard.validate_eta_ladder`, and every emitted domain
  to its physics rows by :func:`check_physics_levels` (the
  ``woof.physics_compat`` gate ``woof check`` runs, with the radiation
  above-model layers at this ``p_top``).
* **Source.**  ``p_top`` above the source's reach is refused
  (:func:`woof.domain_wizard.source_model_top_refusal`).

The column record is ``woof-energy.vertical.v1`` (:meth:`WindEnergyColumn.
record`); planners write it as ``vertical.json`` beside ``plan.json``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from woof.energy.contracts import ContractError, _write_json

VERTICAL_SCHEMA = "woof-energy.vertical.v1"

#: The highest model top the wind-energy preset admits (m, standard
#: atmosphere geopotential altitude of ``p_top``).
MAX_MODEL_TOP_M = 20000.0
#: Preset model top (Pa): ~19.97 km in the standard atmosphere.
DEFAULT_P_TOP_PA = 5500.0
#: Preset mass-level count.
DEFAULT_NZ = 70
#: Preset WRF ``compute_eta`` controls (auto_levels_opt = 2).
DEFAULT_DZBOT_M = 12.0
DEFAULT_DZSTRETCH_S = 1.15
DEFAULT_DZSTRETCH_U = 1.1
DEFAULT_MAX_DZ_M = 1000.0
#: Top of the rotor layer the preset resolves (m AGL): a 150 m hub plus a
#: 100 m blade reaches it.
ROTOR_TOP_M = 250.0
#: Whole model layers the preset demands below :data:`ROTOR_TOP_M`.
DEFAULT_MIN_ROTOR_LAYERS = 8
#: The hindcast's level count: the smallest ``nz`` whose preset column
#: (12 m lowest layer, top 5500 Pa ~ 19.97 km) keeps at least
#: :data:`DEFAULT_MIN_ROTOR_LAYERS` layers below :data:`ROTOR_TOP_M` AND has
#: no layer deeper than :data:`DEFAULT_MAX_DZ_M` (:func:`lean_column_refusal`,
#: found by :func:`smallest_column_nz`).  Pinned by test: 49 gives 10 rotor
#: layers and a 966 m deepest layer; 48 still builds but WRF stretches its
#: top layer to 1091 m (compute_eta admits up to 1.5 x max_dz), 47 breaks
#: even that and 46 or fewer cannot reach the top at all.  Cost scales with ``nz``: 49 levels
#: are 70% of the :data:`DEFAULT_NZ` = 70 column's cells with the same
#: rotor-layer resolution (the extra 21 levels of the 70-level column all
#: sit above the rotor, thinning the free troposphere and stratosphere).
HINDCAST_NZ = 49

#: WRF's reference surface pressure (Pa), the column heights are quoted for.
REFERENCE_SURFACE_PA = 100000.0

# 1976 US / ICAO standard atmosphere, geopotential metres.  Base pressures
# are derived from the lapse rates below (not transcribed), so the table
# and the hydrostatic integral are one statement.
_ISA_G0 = 9.80665
_ISA_R = 287.05287
_ISA_SEA_LEVEL_PA = 101325.0
_ISA_SEA_LEVEL_K = 288.15
_ISA_LAYERS = ((0.0, -0.0065), (11000.0, 0.0), (20000.0, 0.001),
               (32000.0, 0.0028), (47000.0, 0.0), (51000.0, -0.0028),
               (71000.0, -0.002))
_ISA_TOP_M = 84852.0


def _isa_bases() -> tuple[tuple[float, float, float, float], ...]:
    bases = []
    temperature, pressure = _ISA_SEA_LEVEL_K, _ISA_SEA_LEVEL_PA
    for index, (z0, lapse) in enumerate(_ISA_LAYERS):
        bases.append((z0, lapse, temperature, pressure))
        z1 = (_ISA_LAYERS[index + 1][0] if index + 1 < len(_ISA_LAYERS)
              else _ISA_TOP_M)
        pressure = _isa_layer_pressure(z1 - z0, lapse, temperature, pressure)
        temperature += lapse * (z1 - z0)
    return tuple(bases)


def _isa_layer_pressure(dz: float, lapse: float, t0: float, p0: float
                        ) -> float:
    if lapse == 0.0:
        return p0 * math.exp(-_ISA_G0 * dz / (_ISA_R * t0))
    return p0 * ((t0 + lapse * dz) / t0) ** (-_ISA_G0 / (_ISA_R * lapse))


_ISA_BASES = _isa_bases()


class VerticalRefused(ContractError):
    """The wind-energy column cannot be built as asked."""


def isa_pressure_pa(height_m: float) -> float:
    """Standard-atmosphere pressure (Pa) at geopotential ``height_m``."""

    z = float(height_m)
    if not math.isfinite(z) or not 0.0 <= z <= _ISA_TOP_M:
        raise ValueError(f"standard-atmosphere height {height_m!r} m is "
                         f"outside 0..{_ISA_TOP_M:g} m")
    for z0, lapse, t0, p0 in reversed(_ISA_BASES):
        if z >= z0:
            return _isa_layer_pressure(z - z0, lapse, t0, p0)
    raise AssertionError("unreachable")


def isa_height_m(pressure_pa: float) -> float:
    """Standard-atmosphere geopotential altitude (m) of ``pressure_pa``."""

    p = float(pressure_pa)
    floor = isa_pressure_pa(_ISA_TOP_M)
    if not math.isfinite(p) or not floor <= p <= _ISA_SEA_LEVEL_PA * 1.2:
        raise ValueError(f"pressure {pressure_pa!r} Pa is outside the "
                         f"standard atmosphere table ({floor:.4g} Pa .. "
                         f"{_ISA_SEA_LEVEL_PA * 1.2:g} Pa)")
    for z0, lapse, t0, p0 in reversed(_ISA_BASES):
        if p <= p0 or z0 == 0.0:
            if lapse == 0.0:
                return z0 + _ISA_R * t0 / _ISA_G0 * math.log(p0 / p)
            return z0 + t0 / lapse * (
                (p / p0) ** (-_ISA_R * lapse / _ISA_G0) - 1.0)
    raise AssertionError("unreachable")


#: The shallowest ``p_top`` (Pa) the preset admits: the standard-atmosphere
#: pressure at :data:`MAX_MODEL_TOP_M` (5474.9 Pa).
MIN_P_TOP_PA = isa_pressure_pa(MAX_MODEL_TOP_M)


def column_heights(eta_levels: Sequence[float], p_top_pa: float, *,
                   surface_pa: float = REFERENCE_SURFACE_PA
                   ) -> dict[str, list[float]]:
    """Full- and mass-level heights (m above the surface) of a column.

    The reference column of the module docstring: ``p = eta (ps - p_top) +
    p_top`` with ``ps`` = ``surface_pa`` (exact for every ``hybrid_opt`` at
    ``ps = P0``), mass levels at WRF's half-level ``znu = (znw_k +
    znw_k+1) / 2``, heights from :func:`isa_height_m`.
    """

    eta = [float(level) for level in eta_levels]
    pt, ps = float(p_top_pa), float(surface_pa)
    z_surface = isa_height_m(ps)

    def height(level: float) -> float:
        return isa_height_m(level * (ps - pt) + pt) - z_surface

    full = [height(level) for level in eta]
    mass = [height(0.5 * (eta[k] + eta[k + 1])) for k in range(len(eta) - 1)]
    return {"full_level_m": full, "mass_level_m": mass,
            "layer_depth_m": [full[k + 1] - full[k]
                              for k in range(len(full) - 1)]}


def rotor_layers(full_level_m: Sequence[float], *,
                 rotor_top_m: float = ROTOR_TOP_M) -> int:
    """Whole model layers whose top lies at or below ``rotor_top_m``."""

    return sum(1 for z in list(full_level_m)[1:] if z <= rotor_top_m + 1e-9)


def model_top_refusal(p_top_pa: float) -> str | None:
    """Why ``p_top_pa`` is above the preset's 20 km ceiling, or None."""

    top = isa_height_m(float(p_top_pa))
    if top <= MAX_MODEL_TOP_M:
        return None
    return (f"p_top {float(p_top_pa):g} Pa is {top / 1000.0:.2f} km in the "
            f"standard atmosphere, above the wind-energy column's "
            f"{MAX_MODEL_TOP_M / 1000.0:g} km ceiling; use --p-top-pa >= "
            f"{math.ceil(MIN_P_TOP_PA):d} (the preset default is "
            f"{DEFAULT_P_TOP_PA:g} Pa)")


@dataclass(frozen=True)
class WindEnergyColumn:
    """One resolved wind-energy column and what was proven about it."""

    nz: int
    p_top_pa: float
    eta_levels: tuple[float, ...]
    origin: str
    controls: dict[str, float]
    model_top_m: float
    full_level_m: tuple[float, ...]
    mass_level_m: tuple[float, ...]
    rotor_layers: int
    min_rotor_layers: int
    receipt: dict[str, Any] | None = None
    wizard: dict[str, Any] = field(default_factory=dict, repr=False)

    def wizard_vertical(self) -> dict[str, Any]:
        """The ``vertical=`` argument :func:`woof.domain_wizard.render_config`
        takes, with the column's own header lines."""

        return {**self.wizard, "header": self.header_lines()}

    def header_lines(self) -> list[str]:
        lowest = ", ".join(f"{z:.0f}" for z in self.full_level_m[1:7])
        return [
            f"WIND-ENERGY COLUMN (woof.energy.vertical): model top "
            f"{self.model_top_m / 1000.0:.2f} km (standard atmosphere, "
            f"ceiling {MAX_MODEL_TOP_M / 1000.0:g} km);",
            f"  {self.rotor_layers} layers below {ROTOR_TOP_M:g} m "
            f"(>= {self.min_rotor_layers} required); lowest full levels "
            f"~{lowest} m AGL"]

    def summary(self) -> dict[str, Any]:
        """The plan's pointer to its ``vertical.json`` record (carried in
        the first plan domain's ``extra["vertical"]`` and the planner's
        JSON summary)."""

        return {"schema": VERTICAL_SCHEMA, "record": "vertical.json",
                "nz": self.nz, "p_top_pa": self.p_top_pa,
                "model_top_m": round(self.model_top_m, 1),
                "rotor_layers": self.rotor_layers, "origin": self.origin}

    def record(self) -> dict[str, Any]:
        """The ``woof-energy.vertical.v1`` document."""

        return {
            "schema": VERTICAL_SCHEMA,
            "nz": self.nz,
            "p_top_pa": self.p_top_pa,
            "model_top_m": round(self.model_top_m, 1),
            "max_model_top_m": MAX_MODEL_TOP_M,
            "origin": self.origin,
            "controls": dict(self.controls),
            "eta_levels": list(self.eta_levels),
            "heights_reference": (
                "flat column, surface at P0 = 100000 Pa, 1976 standard "
                "atmosphere, metres above the surface"),
            "full_level_m": [round(z, 1) for z in self.full_level_m],
            "mass_level_m": [round(z, 1) for z in self.mass_level_m],
            "rotor_top_m": ROTOR_TOP_M,
            "rotor_layers": self.rotor_layers,
            "min_rotor_layers": self.min_rotor_layers,
            "generator_receipt": self.receipt,
        }

    def write(self, path: str | Path) -> Path:
        """Write :meth:`record` atomically."""

        return _write_json(self.record(), path)


def wind_energy_column(nz: int | None = None,
                       p_top_pa: float | None = None,
                       dzbot_m: float | None = None, *,
                       dzstretch_s: float | None = None,
                       dzstretch_u: float | None = None,
                       max_dz_m: float | None = None,
                       eta_levels: Sequence[float] | None = None,
                       min_rotor_layers: int = DEFAULT_MIN_ROTOR_LAYERS,
                       source: str | None = None,
                       cpu_bridge=None) -> WindEnergyColumn:
    """Build and prove the wind-energy column (see the module docstring).

    ``None`` arguments take the preset defaults.  With ``eta_levels`` the
    ladder is used as given (``nz`` must agree; the generator controls
    must be absent); otherwise WRF ``compute_eta`` generates ``nz + 1``
    levels to ``p_top_pa``.  Refuses (:class:`VerticalRefused`) a top above
    20 km, a ladder that is not 1 -> 0 strictly decreasing, fewer than
    ``min_rotor_layers`` layers below 250 m, more levels than the dynamical
    core holds, and a top above the source's reach.
    """

    from woof.domain_wizard import (resolve_vertical_column,
                                    source_model_top_refusal)

    p_top = DEFAULT_P_TOP_PA if p_top_pa is None else float(p_top_pa)
    if isinstance(p_top_pa, bool) or not math.isfinite(p_top) or p_top <= 0:
        raise VerticalRefused(f"p_top {p_top_pa!r} must be a positive "
                              "pressure in Pa")
    refusal = model_top_refusal(p_top)
    if refusal is not None:
        raise VerticalRefused(refusal)
    if isinstance(min_rotor_layers, bool) or int(min_rotor_layers) < 0:
        raise VerticalRefused("min_rotor_layers must be a whole number >= 0")
    reach = (source_model_top_refusal(source, p_top) if p_top_pa is None
             else None)
    if reach is not None:
        # The preset's own top, not a flag the user gave: say so.
        raise VerticalRefused(
            f"the wind-energy preset top {DEFAULT_P_TOP_PA:g} Pa (no "
            "--p-top-pa given) cannot be served: " + reach)
    if eta_levels is not None:
        given = [name for name, value in (
            ("dzbot", dzbot_m), ("dzstretch_s", dzstretch_s),
            ("dzstretch_u", dzstretch_u), ("max_dz", max_dz_m))
            if value is not None]
        if given:
            raise VerticalRefused(
                f"eta_levels names the column outright; {', '.join(given)} "
                "would generate a different one.  Give one or the other")
        generator: dict[str, float | None] = {}
        levels_nz = nz
    else:
        generator = {
            "dzbot": DEFAULT_DZBOT_M if dzbot_m is None else dzbot_m,
            "dzstretch_s": (DEFAULT_DZSTRETCH_S if dzstretch_s is None
                            else dzstretch_s),
            "dzstretch_u": (DEFAULT_DZSTRETCH_U if dzstretch_u is None
                            else dzstretch_u),
            "max_dz": DEFAULT_MAX_DZ_M if max_dz_m is None else max_dz_m,
        }
        levels_nz = DEFAULT_NZ if nz is None else nz
    try:
        resolved = resolve_vertical_column(
            source=source, nz=levels_nz, p_top_pa=p_top,
            eta_levels=eta_levels, cpu_bridge=cpu_bridge, **generator)
    except ValueError as error:
        raise VerticalRefused(str(error)) from error
    assert resolved is not None
    heights = column_heights(resolved["eta_levels"], resolved["p_top"])
    rotor = rotor_layers(heights["full_level_m"])
    if rotor < int(min_rotor_layers):
        lowest = ", ".join(f"{z:.0f}" for z in heights["full_level_m"][1:12])
        remedy = ("give an --eta-levels ladder with more levels below "
                  f"{ROTOR_TOP_M:g} m" if eta_levels is not None else
                  "lower --dzbot or --dzstretch-s")
        raise VerticalRefused(
            f"the column has {rotor} whole layer(s) below {ROTOR_TOP_M:g} m "
            f"(full levels ~{lowest} m AGL); the wind-energy column needs "
            f">= {int(min_rotor_layers)} to resolve the rotor: {remedy}")
    return WindEnergyColumn(
        nz=int(resolved["nz"]), p_top_pa=float(resolved["p_top"]),
        eta_levels=tuple(resolved["eta_levels"]), origin=resolved["origin"],
        controls=dict(resolved["controls"]),
        model_top_m=isa_height_m(resolved["p_top"]),
        full_level_m=tuple(heights["full_level_m"]),
        mass_level_m=tuple(heights["mass_level_m"]),
        rotor_layers=rotor, min_rotor_layers=int(min_rotor_layers),
        receipt=resolved["receipt"], wizard=resolved)


def deepest_layer_m(column: WindEnergyColumn) -> float:
    """The deepest layer of ``column``'s reference heights (m)."""

    full = column.full_level_m
    return max(full[k + 1] - full[k] for k in range(len(full) - 1))


def lean_column_refusal(nz: int, *, p_top_pa: float | None = None,
                        max_dz_m: float | None = None,
                        source: str | None = None) -> str | None:
    """Why the preset column at ``nz`` levels is not a lean hindcast column,
    or ``None``.

    Lean means everything :func:`wind_energy_column` proves (top at most
    20 km, at least :data:`DEFAULT_MIN_ROTOR_LAYERS` layers below 250 m, a
    ladder WRF ``compute_eta`` can build) AND no layer deeper than
    ``max_dz_m`` (default :data:`DEFAULT_MAX_DZ_M`): WRF's generator admits
    a top layer up to 1.5 x ``max_dz``, which this check does not.
    """

    limit = DEFAULT_MAX_DZ_M if max_dz_m is None else float(max_dz_m)
    try:
        column = wind_energy_column(nz=int(nz), p_top_pa=p_top_pa,
                                    max_dz_m=max_dz_m, source=source)
    except VerticalRefused as error:
        return str(error)
    deepest = deepest_layer_m(column)
    if deepest > limit + 1e-6:
        return (f"nz {int(nz)}: the column's deepest layer is "
                f"{deepest:.0f} m, deeper than max_dz {limit:g} m (WRF "
                "stretched its upper layers to reach p_top); use more levels")
    return None


def smallest_column_nz(*, start: int = 20, stop: int = 256,
                       p_top_pa: float | None = None,
                       max_dz_m: float | None = None,
                       source: str | None = None) -> int:
    """The smallest ``nz`` in ``[start, stop]`` that passes
    :func:`lean_column_refusal`; refuses when none does.

    The preset defaults give :data:`HINDCAST_NZ` (pinned by test).  Each
    probe runs WRF ``compute_eta`` through the native bridge, so this is a
    planning-time search, not something a hot path calls.
    """

    last = None
    for nz in range(int(start), int(stop) + 1):
        last = lean_column_refusal(nz, p_top_pa=p_top_pa, max_dz_m=max_dz_m,
                                   source=source)
        if last is None:
            return nz
    raise VerticalRefused(f"no nz in [{start}, {stop}] gives a lean "
                          f"wind-energy column; the last refusal: {last}")


#: ``woof energy plan`` dests the column reads (``--nz`` is not one of
#: them: alone it keeps the planners' old meaning).
COLUMN_FLAGS = (("p_top_pa", "--p-top-pa"), ("eta_levels", "--eta-levels"),
                ("dzbot", "--dzbot"), ("dzstretch_s", "--dzstretch-s"),
                ("dzstretch_u", "--dzstretch-u"), ("max_dz", "--max-dz"))


def column_flags_given(args: Any) -> list[str]:
    """The explicit-column flags present on a parsed ``woof energy plan``."""

    return [flag for dest, flag in COLUMN_FLAGS
            if getattr(args, dest, None) is not None]


def column_request(args: Any) -> dict[str, Any]:
    """The explicit-column flags of a parsed ``woof energy plan``, as the
    planners' ``column_request`` (``--nz`` travels separately)."""

    return {dest: getattr(args, dest, None) for dest, _ in COLUMN_FLAGS}


def column_from_values(*, nz: int | None = None,
                       p_top_pa: float | None = None,
                       eta_levels: Sequence[float] | None = None,
                       dzbot: float | None = None,
                       dzstretch_s: float | None = None,
                       dzstretch_u: float | None = None,
                       max_dz: float | None = None,
                       source: str | None = None
                       ) -> WindEnergyColumn | None:
    """The planners' column: :func:`wind_energy_column` when any explicit-column
    value is set, else ``None``."""

    if all(value is None for value in (p_top_pa, eta_levels, dzbot,
                                       dzstretch_s, dzstretch_u, max_dz)):
        return None
    return wind_energy_column(
        nz=nz, p_top_pa=p_top_pa, dzbot_m=dzbot, dzstretch_s=dzstretch_s,
        dzstretch_u=dzstretch_u, max_dz_m=max_dz, eta_levels=eta_levels,
        source=source)


def check_physics_levels(settings: Mapping[str, Any] | Any, *,
                         p_top_pa: float, what: str) -> None:
    """Hold one domain's resolved physics to its vertical rows at this
    ``p_top`` (the gate ``woof check`` runs), as a :class:`VerticalRefused`.
    """

    from woof.physics_compat import validate_resolved_physics_vertical_levels

    try:
        validate_resolved_physics_vertical_levels(settings,
                                                  p_top=float(p_top_pa))
    except ValueError as error:
        raise VerticalRefused(f"{what}: {error}") from error


__all__ = [
    "VERTICAL_SCHEMA", "MAX_MODEL_TOP_M", "MIN_P_TOP_PA", "DEFAULT_P_TOP_PA",
    "DEFAULT_NZ", "DEFAULT_DZBOT_M", "DEFAULT_DZSTRETCH_S",
    "DEFAULT_DZSTRETCH_U", "DEFAULT_MAX_DZ_M", "ROTOR_TOP_M",
    "DEFAULT_MIN_ROTOR_LAYERS", "HINDCAST_NZ", "COLUMN_FLAGS",
    "VerticalRefused", "deepest_layer_m", "lean_column_refusal",
    "smallest_column_nz",
    "WindEnergyColumn", "isa_height_m", "isa_pressure_pa", "column_heights",
    "rotor_layers", "model_top_refusal", "wind_energy_column",
    "column_flags_given", "column_request", "column_from_values",
    "check_physics_levels",
]
