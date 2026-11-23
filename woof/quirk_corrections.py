"""The umbrella over woof's corrections of WRF / MPAS quirks.

Every correction of a WRF 4.7.1 / MPAS v8.4.2 quirk that woof ports word
for word has its own named key. Since the upstream WOOF 1.1.7 sync the
corrections follow the physics mode (woof.correct_mode, ``physics.mode``):
in correct mode, the default, every one is on where its scheme runs; in
exact mode (``physics.mode = 'exact'``, the hex door's ``--physics-mode
exact``, an MPAS config dict's ``config_physics_mode = 'exact'``, or a
strict ``GPUWM_WRF_EXACT=1`` process) every one is off, so the run is
the oracle's arithmetic word for word. Parity gates run in exact mode.

The rules, the same on every route:

* Unset umbrella (the default): each correction takes the mode's value
  (correct: corrected; exact: WRF's / MPAS's), and a correction's own key,
  when set, wins: that is how one correction is turned off in correct
  mode, or on in exact mode.
* ``woof_quirk_corrections = 'all'`` / ``'none'``: every correction on /
  off whatever the mode. A key set the other way next to ``'all'`` is a
  conflict and is refused by name; next to ``'none'`` the key wins (that
  is how a single correction is run on its own).
* A correction is read only where its scheme runs (YSU's on a MYNN run,
  the MALI statistics' on a run without them, are inert). Where its
  scheme cannot carry it (YSU's with BEP/BEM, Noah's with a WRF-Hydro
  coupling, the MPAS-Seaice ``ice_velocity_at_cell`` without the
  velocity solver's pools) it stays off; the WRF route's run receipt
  lists such a finding as uncorrected (woof.correct_mode.uncorrected).
* On the WRF route the RunConfig keys are correct-mode findings FQ1-FQ6
  of woof.correct_mode: their defaults are ``CORRECT_KEYS_DEFAULT``,
  exact mode authors them off with every other correct-mode key, and the
  checkpoint echo binds one only where it is on and read.
* Each route's umbrella reaches the keys that route carries. The hex
  door's ``--quirk-corrections`` sets its own flag only: the RunConfig
  corrections (GF's, YSU's) are not hex flags. Where a hex route's
  drivers do not carry the undriven-species rule (the MPAS suite route,
  local time stepping, partial nest, the partitioned same-host executor)
  it stays off there, as a scheme the run does not carry. The MPAS-Seaice
  corrections are names, not keys: under ``'none'`` the ones named in
  ``woof_v842_corrections`` still apply, which is how a subset is run.

:data:`CORRECTIONS` is the ledger: one row per correction, with the route
and key that carries it. tests/test_quirk_corrections.py holds the code's
corrections, README.md and docs/public/CONFIGURATION.md to it, and checks
that exact mode turns every one off.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

#: The umbrella's key on every route (RunConfig field, MPAS config dicts,
#: and the hex door's ``--quirk-corrections``).
UMBRELLA = "woof_quirk_corrections"
#: Its values: 'none' and 'all' (unset, the default, follows the physics mode).
UMBRELLA_VALUES = ("none", "all")
#: The physics-mode key of the MPAS config dicts (MPAS's namelist spelling,
#: as MPAS-A's &nhyd_model config_physics_mode): 'correct' or 'exact'.
MODE_KEY = "config_physics_mode"

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
    """One correction: ``key`` on ``route`` takes ``corrected`` in correct
    mode (and under 'all'); ``exact`` is the word-exact value, taken in
    exact mode (and under 'none').

    ``active`` (WRF route only) says whether a run carries the scheme:
    it takes ``get(name)``, the run's value of RunConfig field ``name``."""

    key: str
    route: str
    exact: object
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
    QuirkCorrection(
        "noah_smc_floor_booking", "wrf", False, True, "sf_surface_physics = 2",
        "Noah's 0.02 soil-moisture floor takes the water it adds from the layer below, or RUNOFF3",
        active=lambda get: int(get("sf_surface_physics")) == 2),
    QuirkCorrection(
        "thompson_rain_snow_closure", "wrf", False, True, "mp_physics = 8, 28 or 38",
        "Thompson's rain-snow collection below 0 C makes graupel from what rain and snow lose after their limiters",
        active=lambda get: int(get("mp_physics")) in (8, 28, 38)),
    QuirkCorrection(
        "rrtmg_lw_model_top", "wrf", False, True, "ra_lw_physics = 4, ra_rrtmg_variant = rrtmg_legacy",
        "WRF's RRTMG longwave keeps the model-top interface and top layer temperatures when it matches the standard profile above the model top",
        # RTE+RRTMGP is another implementation: nothing to correct
        active=lambda get: (int(get("ra_physics") if int(get("ra_lw_physics")) == -1
                                else get("ra_lw_physics")) == 4
                            and get("ra_rrtmg_variant") == "rrtmg_legacy")),
    QuirkCorrection(
        "noah_land_ice_rain_runoff", "wrf", False, True, "sf_surface_physics = 2, land-ice points",
        "Noah books the liquid rain that falls on land ice as surface runoff, where WRF's glacial arm drops it",
        active=lambda get: int(get("sf_surface_physics")) == 2),
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
    QuirkCorrection(
        "bgc_snow_new_category", "mpas_seaice", False, True, "vertical tracers without the vertical thermodynamics",
        "a category without an initial area had no snow (no 0/0 NaN in oceanBioFluxes)"),
    QuirkCorrection(
        "subcycle_snow_ice_tracers", "mpas_seaice", False, True, "vertical tracers with dynamics subcycles",
        "the vertical tracers are separated after each subcycle, so the next combine keeps the ridging"),
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


def default_umbrella(mode=None) -> str:
    """The umbrella an unset ``woof_quirk_corrections`` stands for under
    physics mode ``mode`` (None: this process's default mode, which is
    exact in a strict WRF process): 'all' in correct mode, 'none' in exact."""
    from woof import correct_mode
    if mode is None:
        mode = correct_mode.PHYSICS_MODE_DEFAULT
    if mode not in correct_mode.PHYSICS_MODES:
        raise ValueError(f"physics mode {mode!r} must be one of {list(correct_mode.PHYSICS_MODES)}")
    return "all" if mode == correct_mode.PHYSICS_MODE_CORRECT else "none"


def _refusal(knob: str, value, reason: str, declaration: str):
    from woof.hex.errors import ConfigurationRefusal
    return ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def umbrella_value(value) -> str:
    """``value`` checked as an umbrella value (refused by name otherwise)."""
    if value not in UMBRELLA_VALUES:
        raise ValueError(f"{UMBRELLA} = {value!r} must be 'none' or 'all' (or unset, to follow the physics mode)")
    return value


def _wanted(c: QuirkCorrection, umbrella: str):
    return c.corrected if umbrella == "all" else c.exact


def conflict_message(key: str, value, umbrella: str = "all") -> str:
    """Why ``key`` = ``value`` next to an explicit ``umbrella`` ('all' or
    'none') is refused, in the spelling of the route that carries ``key``."""
    c = BY_KEY[key]
    other = "none" if umbrella == "all" else "all"
    verb = "turns on" if umbrella == "all" else "turns off"
    if c.route == "hex":
        name = KEY_SPELLING.get(key, key)
        spelled = f"--quirk-corrections {umbrella}"
        setting, wanted = f"{name} {value}", f"{name} {_wanted(c, umbrella)}"
        drop = f"--quirk-corrections unset or {other} with {name}, or --quirk-corrections {umbrella} without {name}"
    else:
        spelled = f"{UMBRELLA} = {umbrella!r}"
        setting, wanted = f"{key} = {value!r}", f"{key} = {_wanted(c, umbrella)!r}"
        drop = f"{UMBRELLA} unset or {other!r} with {key}, or {umbrella!r} without {key}"
    return (f"{spelled} {verb} {wanted} ({c.summary}), but this run sets {setting}; drop one: {drop}")


def _umbrella(config: dict) -> tuple[str, bool]:
    """``(umbrella, explicit)`` for an MPAS config dict: its own
    ``woof_quirk_corrections`` when set, else the one its physics mode
    (``config_physics_mode``, else the process default) stands for."""
    if config.get(UMBRELLA) is not None:
        value = config[UMBRELLA]
        try:
            return umbrella_value(value), True
        except ValueError as error:
            raise _refusal(UMBRELLA, value, "the umbrella over the quirk corrections",
                           f"{UMBRELLA} = 'none' | 'all', or unset") from error
    mode = config.get(MODE_KEY)
    try:
        return default_umbrella(mode), False
    except ValueError as error:
        raise _refusal(MODE_KEY, mode, "the physics mode", f"{MODE_KEY} = 'correct' | 'exact'") from error


def all_on(config: dict) -> bool:
    """True when the MPAS config dict ``config`` takes every correction by
    default: ``woof_quirk_corrections = 'all'``, or unset in correct mode."""
    return _umbrella(config)[0] == "all"


def option(config: dict, key: str):
    """The value an MPAS config dict gives correction ``key``: its own key
    when set, else the corrected value in correct mode (or under 'all'),
    else the word-exact value (exact mode, or 'none'). A set key that
    conflicts with an explicit 'all' is refused by name."""
    c = BY_KEY[key]
    if c.route not in ("mpas_ocean", "mali"):
        raise ValueError(f"{key} is carried on the {c.route} route, not in an MPAS config dict")
    umbrella, explicit = _umbrella(config)
    if key in config:
        value = config[key]
        # (a value that is neither the exact nor the corrected one is the
        # port's own to refuse as not an option, with or without the umbrella)
        other = c.exact if umbrella == "all" else c.corrected
        if explicit and umbrella == "all" and value == other:
            raise _refusal(key, value, conflict_message(key, value, umbrella),
                           f"{key} = {_wanted(c, umbrella)!r} or unset, with {UMBRELLA} = {umbrella!r}")
        return value
    return _wanted(c, umbrella)


def with_options(config: dict, route: str) -> dict:
    """``config`` with every ``route`` correction key resolved by :func:`option`."""
    return {**config, **{key: option(config, key) for key in keys(route)}}


def apply_run_corrections(kw: dict) -> dict:
    """The WRF route's TOML readers: a RunConfig keyword dict with an
    explicit umbrella resolved.

    Under ``woof_quirk_corrections = 'all'`` every RunConfig correction is
    set true, and one set explicitly false where its scheme runs is refused
    by name (woof.config.ConfigurationRefused).  Under ``'none'`` every one
    not set explicitly is set false; one set explicitly keeps its value
    (as before the mode switch: 'none' with the individual keys).
    The returned dict is the input otherwise (unset: each key keeps its
    own value, whose default follows the physics mode).
    RunConfig.__post_init__ applies the same rule to a RunConfig built
    any other way (:func:`run_corrections_under`)."""
    from woof.config import ConfigurationRefused, RunConfig
    value = kw.get(UMBRELLA)
    if value is None:
        return kw
    try:
        umbrella_value(value)
    except ValueError as error:
        raise ConfigurationRefused(str(error)) from error
    out = dict(kw)
    for key, wanted in run_corrections_under(value).items():
        # a key whose scheme the domain does not run is never read; the
        # experiment reader writes such keys off on those domains
        if key in out and value == "none":
            continue  # 'none' with an individual key: the key wins
        if key in out and out[key] is not wanted and run_scheme_active(key, out, RunConfig):
            raise ConfigurationRefused(conflict_message(key, out[key], value))
        out[key] = wanted
    if value == "none":
        # every key is now explicit; RunConfig.__post_init__ would otherwise
        # write 'none' over the individual keys that won
        out[UMBRELLA] = None
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
    """The RunConfig corrections ``cfg`` runs: on, and read by its schemes
    (:func:`run_reads`)."""
    return tuple(key for key in keys("wrf") if run_reads(cfg, key))


def run_corrections_under(umbrella: str) -> dict:
    """Every RunConfig correction's value under an explicit ``umbrella``:
    all true under 'all', all false under 'none' (RunConfig.__post_init__
    writes them for a RunConfig built in Python; the TOML readers resolve
    'none' first, so an individual key set next to it wins there; each is
    read only where its scheme runs)."""
    umbrella_value(umbrella)
    return {key: _wanted(BY_KEY[key], umbrella) for key in keys("wrf")}


def run_reads(cfg, key: str) -> bool:
    """Whether RunConfig correction ``key`` reaches ``cfg``'s arithmetic:
    on, and read by the schemes it selects (woof.correct_mode.KEY_SCOPE,
    KEY_GENERATIONS and KEY_UNREAD). Every consumer asks this, so a key
    that is on by default (correct mode) is inert where it is not read."""
    from woof import correct_mode
    if BY_KEY[key].route != "wrf":
        raise ValueError(f"{key} is not a RunConfig correction")
    return (getattr(cfg, key, False) is True
            and correct_mode.key_in_scope(key, correct_mode._scope_values(cfg)))


def check_run_config(cfg) -> None:
    """validate_run_config's umbrella check: unset, 'none' or 'all'.
    (RunConfig.__post_init__ has already applied an explicit one.)"""
    from woof.config import ConfigurationRefused
    value = getattr(cfg, UMBRELLA)
    if value is None:
        return
    try:
        umbrella_value(value)
    except ValueError as error:
        raise ConfigurationRefused(str(error)) from error


def seaice_names(configs: dict) -> tuple[str, ...]:
    """The column corrections the umbrella adds to ``woof_v842_corrections``:
    every one under an explicit 'all', or in correct mode when
    ``woof_v842_corrections`` is unset; none under 'none' or in exact mode.
    A ``woof_v842_corrections`` given under the mode's default is the whole
    set (so naming fewer, or none, turns the others off)."""
    umbrella, explicit = _umbrella(configs)
    if umbrella != "all" or (not explicit and "woof_v842_corrections" in configs):
        return ()
    return keys("mpas_seaice")


def hex_umbrella(arguments) -> tuple[str, bool]:
    """``(umbrella, explicit)`` on the hex door: ``--quirk-corrections``
    when given, else the one ``--physics-mode`` (else the door's default,
    correct) stands for."""
    value = getattr(arguments, "quirk_corrections", None)
    if value is not None:
        return umbrella_value(value), True
    # unset: this process's default mode (exact in a strict WRF process)
    return default_umbrella(getattr(arguments, "physics_mode", None)), False


def hex_undriven_species_boundary(arguments, *, carried: bool = True) -> str | None:
    """The hex door's ``--undriven-species-boundary`` with the umbrella:
    the explicit flag when given, else ``wrf-flow-dependent`` on a regional
    run (``--lbc-dir``) whose route ``carried`` the rule, in correct mode
    or under ``--quirk-corrections all``, else None (``free``). An explicit
    flag that contradicts an explicit ``--quirk-corrections all`` on a
    regional run is refused by name."""
    mode = getattr(arguments, "undriven_species_boundary", None)
    umbrella, explicit = hex_umbrella(arguments)
    regional = getattr(arguments, "lbc_dir", None) is not None
    if not regional:
        return mode
    c = BY_KEY["undriven_species_boundary"]
    if mode is not None:
        if explicit and umbrella == "all" and carried and mode != c.corrected:
            raise ValueError(conflict_message("undriven_species_boundary", mode, umbrella))
        return mode
    if umbrella != "all" or not carried:
        # the umbrella leaves the rule off here (exact mode, or a route
        # whose drivers do not carry it)
        return None
    return c.corrected
