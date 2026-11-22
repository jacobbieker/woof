"""The umbrella over woof's opt-in corrections of WRF / MPAS quirks.

Every correction of a WRF 4.7.1 / MPAS v8.4.2 quirk that woof ports word
for word has its own named opt-in, off by default (the oracle's words).
``woof_quirk_corrections = 'all'`` turns every one of them on where its
scheme runs; ``'none'`` (the default) leaves each to its own key.

The rules, the same on every route:

* Default (``'none'``): nothing changes; each individual key keeps working.
* ``'all'``: each correction below whose scheme the run carries takes its
  corrected value. A correction whose scheme the run does not carry stays
  off (YSU's on a MYNN run, the MALI statistics' on a run without them).
* A key set explicitly to anything but its corrected value next to
  ``'all'`` is a conflict and is refused by name.
* A correction its own key refuses in some combination (YSU's with BEP/BEM,
  say) is refused there by name under the umbrella too: ``'all'`` never
  bypasses a refusal and never drops a correction silently. (The
  MPAS-Seaice ``ice_velocity_at_cell`` needs the velocity solver's
  ``vVelocity`` and ``interiorVertex`` pools: a column run without them
  does not carry it, so 'all' leaves it off there; named, it is refused.)
* On the WRF route RunConfig derives the correction keys under 'all'
  (on where the scheme runs, off elsewhere, again after a
  ``dataclasses.replace`` that changes the scheme); the TOML readers
  refuse an explicit conflicting key first. A RunConfig built in Python
  with 'all' takes the derived values over its correction keys.
* Each route's umbrella reaches the keys that route carries. The hex
  door's ``--quirk-corrections all`` sets its own flag only: the
  RunConfig corrections (GF's, YSU's) are not hex flags. Where a hex
  route's drivers do not carry the undriven-species rule (the MPAS suite
  route, local time stepping, partial nest, the partitioned same-host
  executor) it stays off there, as a scheme the run does not carry.

:data:`CORRECTIONS` is the ledger: one row per correction, with the route
and key that carries it. tests/test_quirk_corrections.py holds the code's
opt-ins, README.md and docs/public/CONFIGURATION.md to it, so a new
correction cannot be left out of the umbrella.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

#: The umbrella's key on every route (RunConfig field, MPAS config dicts,
#: and the hex door's ``--quirk-corrections``).
UMBRELLA = "woof_quirk_corrections"
#: Its values: 'none' (default) and 'all'.
UMBRELLA_VALUES = ("none", "all")

#: the routes, and where each carries its keys
ROUTES = {
    "wrf": "RunConfig field (TOML [shared] / [[domain]])",
    "hex": "woof.hex.forecast_door flag",
    "mpas_ocean": "MPAS-Ocean configs dict",
    "mali": "MALI (mpas_ice) config dict",
    "mpas_seaice": "MPAS-Seaice column configs['woof_v842_corrections'] name",
    "mpas_seaice_velocity": "MPAS-Seaice velocity solver set-up name (timestep basis_corrections / weak_corrections)",
}


@dataclass(frozen=True)
class QuirkCorrection:
    """One opt-in correction: ``key`` on ``route`` takes ``corrected``
    under the umbrella; ``default`` is the word-exact value.

    ``active`` (WRF route only) says whether a run carries the scheme:
    it takes ``get(name)``, the run's value of RunConfig field ``name``."""

    key: str
    route: str
    default: object
    corrected: object
    scheme: str
    summary: str
    active: Callable[[Callable[[str], object]], bool] | None = field(default=None, compare=False, repr=False)


CORRECTIONS: tuple[QuirkCorrection, ...] = (
    # WRF route (woof.config.RunConfig)
    QuirkCorrection(
        "gf_deep_rain_evaporation_booking", "wrf", False, True, "cu_physics = 3",
        "Grell-Freitas deep rain no longer counts the downdraft evaporation it returns to vapour",
        active=lambda get: int(get("cu_physics")) == 3),
    QuirkCorrection(
        "ysu_surface_heat_exner", "wrf", False, True, "bl_pbl_physics = 1",
        "YSU takes the surface heat flux as HFX/pi(1), so the column gets HFX rather than HFX*pi",
        active=lambda get: int(get("bl_pbl_physics")) == 1),
    QuirkCorrection(
        "mynn_sfclay_psih_zolzt", "wrf", False, True, "sf_sfclay_physics = 5, mynn_sfclay_variant = wrf_461",
        "the MYNN surface layer's 2 m / 10 m heat profile is integrated from zt/L, as its log term is",
        # the gsl_wrf39 fork integrates from z0/L throughout: nothing to correct
        active=lambda get: int(get("sf_sfclay_physics")) == 5 and get("mynn_sfclay_variant") == "wrf_461"),
    # hex (MPAS) regional route
    QuirkCorrection(
        "undriven_species_boundary", "hex", "free", "wrf-flow-dependent", "regional mesh (--lbc-dir)",
        "the microphysics species the boundary stream does not drive take WRF's flow-dependent "
        "specified zone"),
    # MPAS-Ocean v8.4.2
    QuirkCorrection(
        "woof_land_ice_isomip_freezing_salinity", "mpas_ocean", "boundary_layer_temperature",
        "boundary_layer_salinity", "ISOMIP land-ice fluxes",
        "the ISOMIP freezing point takes the boundary layer's salinity, not its temperature"),
    QuirkCorrection(
        "woof_bottom_drag_dry_edges", "mpas_ocean", "refuse", "skip", "bottom drag / Rayleigh bottom friction",
        "no bottom drag on a dry edge (maxLevelEdgeTop = 0) instead of v8.4.2's level-0 read (refused)"),
    QuirkCorrection(
        "woof_rayleigh_bottom_friction", "mpas_ocean", "v842", "corrected", "config_Rayleigh_bottom_friction",
        "Rayleigh bottom friction uses config_Rayleigh_bottom_damping_coeff whenever it is on"),
    QuirkCorrection(
        "woof_redi_k33", "mpas_ocean", "v842", "kappa", "config_use_Redi",
        "Redi's vertical diffusion adds config_Redi_kappa * k33 (a diffusivity), not the bare slope k33"),
    # MALI v8.4.2
    QuirkCorrection(
        "woof_sgh_init_mask", "mali", "v842", "corrected", "config_SGH",
        "the subglacial hydrology's initial ocean pressure reads the computed cell mask"),
    QuirkCorrection(
        "woof_enthalpy_advection_units", "mali", "v842", "corrected", "enthalpy solver with advection",
        "advected enthalpy converts temperatures in Celsius, as the thermal solver does"),
    QuirkCorrection(
        "woof_simple_velocity_bed", "mali", "v842", "corrected", "simple velocity solver",
        "the simple velocity solver keeps the input bed (no circular-shelf test-case bed)"),
    QuirkCorrection(
        "woof_statistics_extrema", "mali", "v842", "ice", "global statistics",
        "the statistics' extrema are taken over ice only"),
    QuirkCorrection(
        "woof_analysis_empty", "mali", "v842", "corrected", "global/regional stats analysis members",
        "means and minimum thickness over no ice report 0, not NaN or the sentinel"),
    # MPAS-Seaice v8.4.2 column (woof_v842_corrections names)
    QuirkCorrection(
        "set_sfcflux_order", "mpas_seaice", False, True, "config_calc_surface_temperature = .false.",
        "set_sfcflux gets its coupled fluxes in its own order"),
    QuirkCorrection(
        "ice_velocity_at_cell", "mpas_seaice", False, True, "column with the velocity solver",
        "the atmosphere's relative wind uses the cell's ice velocity, both components"),
    QuirkCorrection(
        "ccsm3_absorbed_solar_layers", "mpas_seaice", False, True, "ccsm3 shortwave",
        "the CCSM3 penetrating shortwave is spread over nilyr layers, not ncat"),
    QuirkCorrection(
        "dedd_zero_volume_ice", "mpas_seaice", False, True, "delta-Eddington shortwave",
        "layers without thickness absorb no chlorophyll light (no NaN)"),
    QuirkCorrection(
        "zbgc_c2n_proteins", "mpas_seaice", False, True, "column z-BGC",
        "the diatoms and the proteins get their own carbon to nitrogen ratios"),
    QuirkCorrection(
        "bgc_inactive_diagnostics", "mpas_seaice", False, True, "column biogeochemistry",
        "only allocated BGC diagnostics are zeroed (no write to address 0)"),
    QuirkCorrection(
        "zsal_solve", "mpas_seaice", False, True, "config_use_vertical_zsalinity",
        "the vertical salinity is solved, as config_use_vertical_zsalinity asks"),
    # MPAS-Seaice v8.4.2 velocity solver set-up (woof.hex.mpas_seaice.timestep_v842)
    QuirkCorrection(
        "planar_metric_terms", "mpas_seaice_velocity", False, True, "variational basis on a planar mesh",
        "the variational metric terms are zero on a plane (not the rotated z' or 0/0)"),
    QuirkCorrection(
        "periodic_local_coords", "mpas_seaice_velocity", False, True, "variational basis on a periodic plane",
        "the basis' vertex offsets are unwrapped across the period, as mpas_fix_periodicity does"),
    QuirkCorrection(
        "weak_ridge_delta", "mpas_seaice_velocity", False, True, "weak stress divergence with the column package",
        "each cell's ridgeShear takes its own Delta, not the last owned cell's"),
)

BY_KEY = {c.key: c for c in CORRECTIONS}
#: how each route spells "turn every correction on", for the refusals
UMBRELLA_SPELLING = {"hex": "--quirk-corrections all"}
#: how each route spells a key, for the refusals (default: the key itself)
KEY_SPELLING = {"undriven_species_boundary": "--undriven-species-boundary"}


def keys(route: str) -> tuple[str, ...]:
    """The correction keys ``route`` carries, in ledger order."""
    if route not in ROUTES:
        raise ValueError(f"route must be one of {sorted(ROUTES)}")
    return tuple(c.key for c in CORRECTIONS if c.route == route)


def _refusal(knob: str, value, reason: str, declaration: str):
    from woof.hex.errors import ConfigurationRefusal
    return ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def umbrella_value(value) -> str:
    """``value`` checked as an umbrella value (refused by name otherwise)."""
    if value not in UMBRELLA_VALUES:
        raise ValueError(f"{UMBRELLA} = {value!r} must be 'none' or 'all'")
    return value


def conflict_message(key: str, value) -> str:
    """Why ``key`` = ``value`` next to the umbrella's 'all' is refused, in
    the spelling of the route that carries ``key``."""
    c = BY_KEY[key]
    umbrella = UMBRELLA_SPELLING.get(c.route, f"{UMBRELLA} = 'all'")
    if c.route == "hex":
        name = KEY_SPELLING.get(key, key)
        setting, wanted = f"{name} {value}", f"{name} {c.corrected}"
        drop = f"--quirk-corrections none with {name}, or --quirk-corrections all without {name}"
    else:
        setting, wanted = f"{key} = {value!r}", f"{key} = {c.corrected!r}"
        drop = f"{UMBRELLA} = 'none' with the individual keys, or 'all' without {key}"
    return (f"{umbrella} turns on {wanted} ({c.summary}), but this run sets {setting}; drop one: {drop}")


def all_on(config: dict) -> bool:
    """True when the MPAS config dict ``config`` asks for every correction."""
    value = config.get(UMBRELLA, "none")
    try:
        return umbrella_value(value) == "all"
    except ValueError as error:
        raise _refusal(UMBRELLA, value, "the umbrella over the opt-in quirk corrections",
                       f"{UMBRELLA} = 'none' | 'all'") from error


def option(config: dict, key: str):
    """The value an MPAS config dict gives correction ``key``: its own key
    when set, else the corrected value under ``woof_quirk_corrections =
    'all'``, else the word-exact default. A set key that conflicts with
    the umbrella is refused by name."""
    c = BY_KEY[key]
    if c.route not in ("mpas_ocean", "mali"):
        raise ValueError(f"{key} is carried on the {c.route} route, not in an MPAS config dict")
    umbrella = all_on(config)
    if key in config:
        value = config[key]
        # (a value that is neither the default nor the corrected one is the
        # port's own to refuse as not an option, with or without the umbrella)
        if umbrella and value == c.default:
            raise _refusal(key, value, conflict_message(key, value),
                           f"{key} = {c.corrected!r} or unset, with {UMBRELLA} = 'all'")
        return value
    return c.corrected if umbrella else c.default


def with_options(config: dict, route: str) -> dict:
    """``config`` with every ``route`` correction key resolved by :func:`option`."""
    return {**config, **{key: option(config, key) for key in keys(route)}}


def apply_run_corrections(kw: dict) -> dict:
    """The WRF route's TOML readers: a RunConfig keyword dict with ``'all'``
    resolved.

    Under ``woof_quirk_corrections = 'all'`` every RunConfig correction
    whose scheme the dict selects is set true; one set explicitly to
    anything else is refused by name (woof.config.ConfigurationRefused).
    The returned dict is the input otherwise (``'none'``: unchanged).
    RunConfig.__post_init__ applies the same rule to a RunConfig built
    any other way (:func:`run_corrections_due`)."""
    from woof.config import ConfigurationRefused, RunConfig
    value = kw.get(UMBRELLA, "none")
    try:
        umbrella_value(value)
    except ValueError as error:
        raise ConfigurationRefused(str(error)) from error
    if value != "all":
        return kw
    out = dict(kw)
    for key in keys("wrf"):
        if not run_scheme_active(key, out, RunConfig):
            if out.get(key) is True:
                # what validate_run_config refuses without the umbrella;
                # under 'all' the key would be re-derived off, silently
                raise ConfigurationRefused(
                    f"{key} corrects {BY_KEY[key].scheme}, which this run does not select, so it would "
                    "never be read; drop it")
            continue
        if key in out and out[key] is not True:
            raise ConfigurationRefused(conflict_message(key, out[key]))
        out[key] = True
    return out


def run_scheme_active(key: str, cfg, defaults=None) -> bool:
    """Whether the run (a RunConfig, or a keyword dict over ``defaults``)
    carries the scheme RunConfig correction ``key`` corrects."""

    def get(name):
        if isinstance(cfg, dict):
            return cfg.get(name, getattr(defaults, name))
        return getattr(cfg, name)

    c = BY_KEY[key]
    if c.route != "wrf" or c.active is None:
        raise ValueError(f"{key} is not a RunConfig correction with a scheme predicate")
    return bool(c.active(get))


def run_corrections_due(cfg) -> tuple[str, ...]:
    """The RunConfig corrections ``'all'`` turns on for ``cfg``: those whose
    scheme it runs."""
    return tuple(key for key in keys("wrf") if run_scheme_active(key, cfg))


def run_corrections_under_all(cfg) -> dict:
    """Every RunConfig correction's value under ``'all'``: on where its
    scheme runs, off elsewhere (RunConfig.__post_init__ writes them)."""
    due = run_corrections_due(cfg)
    return {key: key in due for key in keys("wrf")}


def check_run_config(cfg) -> None:
    """validate_run_config's umbrella check: the value is 'none' or 'all'.
    (RunConfig.__post_init__ has already turned the due corrections on.)"""
    from woof.config import ConfigurationRefused
    try:
        umbrella_value(getattr(cfg, UMBRELLA))
    except ValueError as error:
        raise ConfigurationRefused(str(error)) from error


def seaice_names(configs: dict) -> tuple[str, ...]:
    """The column corrections the umbrella adds to ``woof_v842_corrections``."""
    return keys("mpas_seaice") if all_on(configs) else ()


def hex_undriven_species_boundary(arguments, *, carried: bool = True) -> str | None:
    """The hex door's ``--undriven-species-boundary`` with the umbrella:
    the explicit flag when given, else ``wrf-flow-dependent`` under
    ``--quirk-corrections all`` on a regional run (``--lbc-dir``) whose
    route ``carried`` the rule, else None (``free``). An explicit ``free``
    next to ``all`` on a regional run is refused by name."""
    mode = getattr(arguments, "undriven_species_boundary", None)
    umbrella = umbrella_value(getattr(arguments, "quirk_corrections", None) or "none") == "all"
    regional = getattr(arguments, "lbc_dir", None) is not None
    if not (umbrella and regional):
        return mode
    if not carried:
        # the umbrella leaves the rule off here; an explicit flag is the
        # caller's to judge (free is the only value such a route honours)
        return mode
    corrected = BY_KEY["undriven_species_boundary"].corrected
    if mode is not None and mode != corrected:
        raise ValueError(conflict_message("undriven_species_boundary", mode))
    return corrected
