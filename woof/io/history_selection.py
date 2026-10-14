"""Which variables the history tape carries: the ``[output]`` surface.

A woof run writes its whole history inventory every period, and on a
four-domain tree at forecast cadence that is the thing that fills a
disk.  WRF's own answer is ``iofields_filename``; this is woof's, and
it is deliberately a SELECTION over the inventory the run already
produces rather than a second inventory of its own:

* the vocabulary is the UNION of the wrfout schema tables, both of them
  in :mod:`woof.io.wrf_output_schema` (``HISTORY_FIELDS_BY_NETCDF_NAME``
  and ``REGISTRY_VAR_META``, which the writer binds as ``_VAR_META``),
  so a scheme that adds a history field joins the vocabulary with its
  schema row and no code here changes -- the arbitrary-model test
  applied to variables;
* the presets are TABLES of names, so a new preset, or a new member of
  one, is a row;
* the filter is one predicate applied to the frame the writer already
  assembled, so there is no per-variable code path to grow.

THE DEFAULT IS THE FULL SET.  ``[output]`` absent, or ``preset =
"full"``, writes exactly what every run before this feature wrote --
same inventory, same order, same header -- and stamps no attribute of
its own.  Trimming is a thing a user asks for, never something a
default does to them.

WHAT IS NOT DROPPABLE, and why.  :data:`STRUCTURAL_FIELDS` is the
georeference, the vertical coordinate and the time coordinate: without
them no reader can place a value in space or time, so a file missing
one is not a smaller wrfout, it is an unreadable one.  Naming one in
``history_drop`` is refused with that sentence rather than served.

THE CHECKPOINT STREAM IS UNAFFECTED.  Checkpoints are separate files
written by :mod:`woof.io.restart` from model state, not from the
history frame, so a run that trims its history restarts exactly as it
would have.  ``tests/test_restart.py`` proves it, and the selection is
deliberately kept out of the restart identity for the same reason
``[tiles]`` is: it changes no number the model computes.
"""
from __future__ import annotations

from dataclasses import dataclass
import difflib
import math
from pathlib import Path

from woof.explain import warn
from woof.io.wrf_output_schema import (HISTORY_FIELDS_BY_NETCDF_NAME,
                                        HUB_WIND_OUTPUT_FIELDS,
                                        REGISTRY_VAR_META)


#: The names the writer emits for every frame regardless of physics, plus
#: the time coordinate it derives.  Kept here rather than read off
#: ``state_frame`` because that function needs a live GPU state to answer
#: and this table has to be readable at config-resolution time.
_CORE_DYNAMICS_FIELDS = (
    "Times", "XTIME", "ITIMESTEP",
    "T", "U", "V", "W", "PH", "PHB", "MU", "MUB", "HGT",
    "P", "PB", "PSFC", "P_TOP", "ZNU", "ZNW",
    "QVAPOR", "QCLOUD", "QRAIN",
    "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V",
    "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E", "SINALPHA", "COSALPHA",
    "LANDMASK", "LU_INDEX",
)


def _writer_metadata_names() -> frozenset[str]:
    """Every name the writer carries Registry metadata for.

    Read from :mod:`woof.io.wrf_output_schema`, which OWNS the table,
    and never from :mod:`woof.io.wrfout`, which merely binds it.  This
    function is called at module scope to build
    :data:`HISTORY_VOCABULARY`, so importing the writer here imports the
    writer whenever anything reads a config -- and the writer imports
    :mod:`woof.supervisor`, ``netCDF4`` and the runtime behind them.
    MEASURED: ``import woof.era5_direct`` pulled the whole forecast
    output side in through case_data -> experiment -> here, and the
    RW-WPS preprocessing wheel's no-CuPy/no-executor import proof went
    red on it.  ``tests/test_history_selection.py`` pins the boundary.
    """

    return frozenset(REGISTRY_VAR_META)


#: Every variable name a woof wrfout can carry: the union of the two
#: schema tables plus the core dynamics/time names.  A ``[output]`` list
#: may only name members of this set, and an unknown name is refused with
#: the whole set printed -- there is no other place a user can read it.
HISTORY_VOCABULARY: frozenset[str] = frozenset(
    set(_CORE_DYNAMICS_FIELDS)
    | set(HISTORY_FIELDS_BY_NETCDF_NAME)
    | _writer_metadata_names()
)

#: The fields no selection may drop, and the sentence that says why.
#:
#: These are not "important" fields -- ``REFL_10CM`` is important and is
#: droppable.  These are the ones whose absence stops a reader from
#: placing ANY value: the horizontal georeference (``XLAT``/``XLONG``),
#: the terrain and mass/vertical coordinate the eta levels are defined
#: against (``HGT``, ``ZNU``, ``ZNW``, ``P_TOP``, ``MUB``, ``PHB``), and
#: the time coordinate WRF tooling reads (``Times``, ``XTIME``,
#: ``ITIMESTEP``).  A wrfout missing one of these is not a smaller
#: wrfout; it is a file no tool in the estate can open as one.
#:
#: ``T`` IS IN THIS SET, AND IT IS HERE BECAUSE IT WAS MEASURED.  A
#: two-domain proof run whose 1 km child took ``preset = "minimal"``
#: wrote a d02 tape that ``rw_wrfbatch`` refused outright -- "Open WRF
#: ... failed and the file is not a supported post-processed WRF
#: archive" -- not a product skipped, the whole FILE rejected.  The
#: ladder was then measured against the real binary on a real frame,
#: one field at a time (``rw_wrfbatch --list-products``):
#:
#:   minimal (57 vars, no 3-D)      rc 1, the file does not open
#:   minimal + T (58)               rc 0,  30 products renderable
#:   minimal + T,P,PB (60)          rc 0,  38
#:   minimal + T,P,PB,PH (61)       rc 0,  42
#:   minimal + ...,QVAPOR (62)      rc 0,  66
#:   minimal + ...,U,V,W (65)       rc 0, 160
#:   the full inventory (74)        rc 0, 162
#:
#: So ``T`` is the field that makes the difference between a readable
#: wrfout and a file the estate's own reader will not open, which is
#: exactly what this set is for.  It costs one 3-D volume, and a preset
#: that shed it would be selling a file nothing can read.
STRUCTURAL_FIELDS: frozenset[str] = frozenset({
    "Times", "XTIME", "ITIMESTEP",
    "XLAT", "XLONG",
    "HGT", "ZNU", "ZNW", "P_TOP", "MUB", "PHB",
    "T",
})

#: The ``minimal`` preset's science: the 2-D surface state and the
#: accumulators every surface product is drawn from.  One row per field;
#: adding a surface product's input to the preset is adding a row.
#:
#: It carries no 3-D volume of its own.  The one it does write --
#: ``T``, plus the ``PHB`` column -- comes from
#: :data:`STRUCTURAL_FIELDS`, which is where the measured reason lives:
#: without ``T`` the file is not a wrfout any reader in the estate will
#: open.
_MINIMAL_FIELDS: frozenset[str] = frozenset({
    # Surface state and near-surface diagnostics.
    "T2", "Q2", "TH2", "PSFC", "U10", "V10", "TSK", "MU",
    # Surface energy budget and boundary-layer depth.
    "HFX", "LH", "QFX", "GRDFLX", "SWDOWN", "GLW", "OLR", "PBLH",
    "UST", "PSIM", "PSIH",
    # Precipitation and snow accumulators.
    "RAINC", "RAINNC", "RAINSH", "SNOWNC", "GRAUPELNC", "HAILNC",
    "SNOW", "SNOWC", "SNOWH",
    # Column extremes a severe-weather surface panel reads.
    "UP_HELI_MAX",
    # Hub-height wind planes and their interval statistics: present only
    # when [output] asks for them (hub_heights_m), and 2-D per height, so
    # they belong with the surface panel rather than the volumes.
    *HUB_WIND_OUTPUT_FIELDS,
    # Grid metadata: 2-D, cheap, and what a staggered reader needs.
    "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V",
    "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E", "SINALPHA", "COSALPHA",
    # Land identity: what the ground IS, which every child grid needs.
    "LANDMASK", "LU_INDEX", "ISLTYP", "IVGTYP", "TMN", "VEGFRA", "SEAICE",
})

#: What ``severe`` adds to ``minimal``: the storm-scale volumes.  It is
#: the full inventory MINUS the restart-only scheme carriers no render
#: product reads -- stated as an explicit add-list rather than a
#: subtract-list so a new scheme's carriers do not silently join it.
_SEVERE_EXTRA_FIELDS: frozenset[str] = frozenset({
    # Dynamics volumes (``T`` is structural and rides along regardless).
    "U", "V", "W", "PH", "P", "PB",
    # Hydrometeors: mixing ratios and the number concentrations a
    # double-moment scheme carries.
    "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL",
    "QNCLOUD", "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL", "QNHAIL",
    # Simulated reflectivity, the volume every radar product is built on.
    "REFL_10CM",
})

#: The ``energy`` preset's science: what a forecast along power lines,
#: substations and renewable sites is sampled from (``woof energy
#: extract``).  Stated as its own add-list, not built from ``minimal``,
#: because its reader is a site sampler and not a render panel.
#:
#: * Winds and temperature through the column (``U``, ``V``, ``W``, ``T``
#:   and the height/pressure coordinate ``PH``/``PHB``/``P``/``PB``) for
#:   hub-height and conductor-height interpolation.
#: * Hydrometeors for icing and precipitation loading.
#: * The near-surface state and the surface shortwave TRIO a PV
#:   plane-of-array model needs: global ``SWDOWN``, direct-normal
#:   ``SWDDNI``, diffuse ``SWDDIF`` and the zenith cosine ``COSZEN``.
#: * ``SINALPHA``/``COSALPHA`` to rotate grid winds to earth-relative
#:   bearings against a line's azimuth, and the georeference.
#:
#: A member the run does not produce is simply absent from its tape, the
#: rule every preset already follows (a preset is a filter over what the
#: run produced, so ``severe`` on a warm-rain scheme has no ``QICE`` and
#: says nothing): ``SWDDNI``/``SWDDIF`` need a shortwave scheme that
#: computes a direct/diffuse split (``ra_sw_physics = 4``), ``COSZEN``
#: needs radiation on, ``QICE``/``QSNOW``/``QGRAUP`` need an ice scheme.  The
#: consumer reads that absence as "not produced"; nothing is zero-filled.
_ENERGY_FIELDS: frozenset[str] = frozenset({
    # Column dynamics and the vertical coordinate.
    "U", "V", "W", "T", "PH", "PHB", "HGT", "P", "PB",
    # Moisture and hydrometeors (icing, loading).  QNCLOUD is Thompson
    # mp 28's droplet number: turbine icing takes the droplet median volume
    # diameter from it (woof.energy.icing_turbine) and otherwise assumes one.
    "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNCLOUD",
    # Near-surface state.
    "T2", "Q2", "U10", "V10", "PSFC",
    # Surface shortwave: global, direct-normal, diffuse, zenith cosine.
    "SWDOWN", "SWDDNI", "SWDDIF", "COSZEN",
    # Precipitation accumulators.
    "RAINNC", "RAINC",
    # Grid rotation and georeference.
    "SINALPHA", "COSALPHA", "XLAT", "XLONG",
    # In-model hub-height wind and its per-interval MAX/MIN/MEAN
    # (woof/core/hub_wind_diag.py), when [output] hub_heights_m asks.
    *HUB_WIND_OUTPUT_FIELDS,
})

#: Named presets: name -> the set of droppable fields it keeps, or
#: ``None`` for "everything this run produces".  :data:`STRUCTURAL_FIELDS`
#: is added to every non-``None`` set at selection time.
HISTORY_PRESETS: dict[str, frozenset[str] | None] = {
    "full": None,
    "minimal": _MINIMAL_FIELDS,
    "severe": _MINIMAL_FIELDS | _SEVERE_EXTRA_FIELDS,
    "energy": _ENERGY_FIELDS,
}

#: What each named preset costs in RENDER PRODUCTS, not in variables.
#:
#: The ladder at the head of this module was measured one field at a
#: time against the real binary (``rw_wrfbatch --list-products``) on one
#: 1 km frame of the proof run; these are its rungs read as products
#: rather than as prose.  ``severe`` keeps every volume a render product
#: reads and sheds only the restart-only scheme carriers none of them
#: touches, so it prices at the full inventory's count.
#:
#: The figure is computed from THIS table inside
#: :meth:`HistorySelection.warn_lost_products`, never from what a caller
#: passes as ``produced``, so the two plan-review doors print the same
#: number for one configuration no matter what either of them hands in.
PRESET_PRODUCT_COUNTS: dict[str, int] = {
    "full": 162,
    "severe": 162,
    "minimal": 30,
}

#: Presets measured on a frame of their OWN rather than on the ladder above:
#: name -> (products kept, products the same frame's full inventory renders,
#: the basis clause).  Their figures share no basis with
#: :data:`PRESET_PRODUCT_COUNTS` -- a newer renderer build whose catalog is
#: larger, a different run -- so they are reported as their own pair and
#: never mixed into the ladder's 162, and the ladder's per-field
#: attribution (:data:`PRODUCT_LOSS_ATTRIBUTION`) is not applied to them:
#: ``energy`` KEEPS ``U,V,W``, ``QVAPOR``, ``P,PB`` and ``PH``.
#:
#: ``energy``: rw_wrfbatch --list-products on the 14Z frame of a 2-hour
#: 3 km HRRR-forced run (100x80x49, Thompson, RTE+RRTMGP), the full tape
#: (82 variables) against the same frame trimmed by this preset (37):
#: ``CATALOG ... renderable=207`` and ``renderable=174``.  What it sheds
#: that products read is ``REFL_10CM`` (radar products fall back to the
#: generic hydrometeor dBZ), ``MU``, ``UP_HELI_MAX``, ``OLR``
#: and the land-surface rows.
SEPARATELY_MEASURED_PRESET_PRODUCT_COUNTS: dict[str, tuple[int, int, str]] = {
    "energy": (174, 207,
               "measured with rw_wrfbatch --list-products on one 3 km "
               "frame (49 levels, Thompson, RTE+RRTMGP), full tape against "
               "the same frame trimmed to the preset"),
}

#: Which fields the loss is attributable to, largest first, read off the
#: same ladder by subtraction: 160 - 66 for ``U,V,W``, 66 - 42 for
#: ``QVAPOR``, 38 - 30 for ``P,PB``, 42 - 38 for ``PH``.  The remainder
#: is whatever the rest of the full inventory serves, and
#: :meth:`HistorySelection.warn_lost_products` names it as such rather
#: than leaving a reader to find that the figures do not add up.
PRODUCT_LOSS_ATTRIBUTION: tuple[tuple[tuple[str, ...], int], ...] = (
    (("U", "V", "W"), 94),
    (("QVAPOR",), 24),
    (("P", "PB"), 8),
    (("PH",), 4),
)

#: How the counts above were obtained, in one clause a warning can carry.
PRODUCT_COUNT_BASIS = (
    "measured with rw_wrfbatch --list-products on one 1 km frame of the "
    "proof run (49 levels, WSM6, RRTMG), one field at a time")

#: The keys ``[output]`` accepts.  The four ``hub_*`` keys configure the
#: in-model hub-height wind diagnostic (:class:`HubWindRequest`).
OUTPUT_KEYS: frozenset[str] = frozenset(
    {"preset", "history_vars", "history_drop",
     "hub_wind", "hub_heights_m", "hub_sample_interval_s", "hub_sites"})

#: ``hub_heights_m`` when ``hub_wind = true`` names none: the common
#: onshore/offshore turbine hub range, in m above ground level.
DEFAULT_HUB_HEIGHTS_M: tuple[float, ...] = (40.0, 80.0, 100.0, 120.0,
                                            150.0, 200.0)
#: ``hub_sample_interval_s`` when ``hub_sites`` is given without one.
DEFAULT_HUB_SAMPLE_INTERVAL_S = 60.0
#: Upper bounds that keep a typo from becoming a silent allocation.  A
#: wind 2 km above ground is not a hub height, and 32 levels is already
#: more than any turbine fleet spans.
MAX_HUB_HEIGHT_M = 2000.0
MAX_HUB_HEIGHTS = 32


class HubWindConfigError(ValueError):
    """An ``[output] hub_*`` key the diagnostic cannot honour as written."""


@dataclass(frozen=True)
class HubWindRequest:
    """The validated ``[output] hub_*`` keys of one domain.

    PINNED: ``heights_m`` is strictly increasing, every entry in
    ``(0, MAX_HUB_HEIGHT_M]``; ``sites`` is ``(site_id, lat, lon)``
    triples with unique ids, longitudes normalised to ``[-180, 180)``;
    ``sample_interval_s`` is ``None`` exactly when ``sites`` is empty.
    Everything here is plain tuples and floats so ``dataclasses.asdict``
    (the resolved-TOML and identity walks) serialises it unchanged.
    """

    heights_m: tuple[float, ...] = DEFAULT_HUB_HEIGHTS_M
    sample_interval_s: float | None = None
    sites: tuple[tuple[str, float, float], ...] = ()
    #: Where the sites came from (a sites.v1 path, or "inline").
    sites_source: str | None = None

    def spelling(self) -> str:
        parts = ["hub_heights_m=" + ",".join(f"{h:g}" for h in self.heights_m)]
        if self.sites:
            parts.append(f"hub_sites={len(self.sites)}")
            parts.append(f"hub_sample_interval_s={self.sample_interval_s:g}")
        return " ".join(parts)


def _finite_number(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HubWindConfigError(f"{what} must be a number, got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise HubWindConfigError(f"{what} must be finite, got {value!r}.")
    return number


def _resolve_sites_path(text: str, *, base_dir, source: str) -> Path:
    """A sites file named relative to the config, else the working directory.

    The config's own directory FIRST: a case folder that carries its sites
    file must sample those turbines from wherever it is launched, and an
    unrelated ``sites.json`` in the launch directory must not win.
    """
    path = Path(text).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = []
        if base_dir is not None:
            candidates.append(Path(base_dir) / path)
        candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise HubWindConfigError(
        f"[output] hub_sites of {source} names {text!r}, which is not a "
        f"file (looked at {[str(c) for c in candidates]}).  hub_sites is a "
        "sites.v1 JSON document (woof energy sites) or an inline list of "
        '{ site_id = "...", lat = ..., lon = ... } tables.')


def _parse_sites(value, *, base_dir, source: str
                 ) -> tuple[tuple[tuple[str, float, float], ...], str]:
    rows: list[tuple[str, float, float]] = []
    if isinstance(value, str):
        from woof.energy.contracts import ContractError, load_sites

        path = _resolve_sites_path(value, base_dir=base_dir, source=source)
        try:
            site_set = load_sites(path)
        except ContractError as err:
            raise HubWindConfigError(
                f"[output] hub_sites of {source}: {path} is not a valid "
                f"sites.v1 document: {err}") from None
        rows = [(site.site_id, float(site.lat), float(site.lon))
                for site in site_set.sites]
        origin = str(path)
    elif isinstance(value, (list, tuple)):
        for index, row in enumerate(value):
            if not isinstance(row, dict) or set(row) != {"site_id", "lat", "lon"}:
                raise HubWindConfigError(
                    f"[output] hub_sites[{index}] of {source} must be a table "
                    'with exactly site_id, lat and lon, e.g. { site_id = '
                    '"hornsea-1", lat = 53.88, lon = 1.92 }, got '
                    f"{row!r}.")
            site_id = row["site_id"]
            if not isinstance(site_id, str) or not site_id:
                raise HubWindConfigError(
                    f"[output] hub_sites[{index}] of {source}: site_id must "
                    f"be a non-empty string, got {site_id!r}.")
            rows.append((site_id,
                         _finite_number(row["lat"], f"hub_sites[{index}].lat"),
                         _finite_number(row["lon"], f"hub_sites[{index}].lon")))
        origin = "inline"
    else:
        raise HubWindConfigError(
            f"[output] hub_sites of {source} must be a sites.v1 file path or "
            f"a list of site tables, got {value!r}.")
    if not rows:
        raise HubWindConfigError(
            f"[output] hub_sites of {source} names no sites.  Delete the key "
            "to turn site sampling off.")
    seen: set[str] = set()
    normalised = []
    for site_id, lat, lon in rows:
        if site_id in seen:
            raise HubWindConfigError(
                f"[output] hub_sites of {source} repeats site_id {site_id!r}.")
        seen.add(site_id)
        if not -90.0 <= lat <= 90.0:
            raise HubWindConfigError(
                f"[output] hub_sites of {source}: {site_id} latitude {lat} "
                "is outside [-90, 90].")
        if not -180.0 <= lon < 360.0:
            raise HubWindConfigError(
                f"[output] hub_sites of {source}: {site_id} longitude {lon} "
                "is outside [-180, 360).")
        # Only the [180, 360) half moves, so a longitude already in range
        # keeps its exact float.
        normalised.append((site_id, lat, lon - 360.0 if lon >= 180.0 else lon))
    return tuple(normalised), origin


def output_key_rows():
    """Declared rows (type, default, one-line doc) for every ``[output]`` key.

    The whole table, not only the keys no typed field carries: a front end
    reading :func:`woof.config.declared_key_rows` learns ``[output]`` from
    here, and a partial table would read as "these are all the keys".
    """
    from woof.config_keys import KeyRow, key_rows

    return key_rows(
        KeyRow("preset", "string", "full",
               "Named history set: full (default), minimal, severe or "
               "energy."),
        KeyRow("history_vars", "array", None,
               "Explicit list of wrfout variables to keep; excludes "
               "history_drop and a non-full preset.", items="string"),
        KeyRow("history_drop", "array", None,
               "Wrfout variables to drop from the preset; structural "
               "fields refuse.", items="string"),
        KeyRow("hub_wind", "boolean", False,
               "true turns the in-model hub-height wind fields on at the "
               "default heights."),
        KeyRow("hub_heights_m", "array", list(DEFAULT_HUB_HEIGHTS_M),
               "Heights above ground (m, strictly increasing) of the "
               "in-model hub-height wind fields; setting it turns them on.",
               items="number"),
        KeyRow("hub_sample_interval_s", "number",
               DEFAULT_HUB_SAMPLE_INTERVAL_S,
               "Cadence in model seconds of per-site hub-wind sampling "
               "into the hubwind sidecar; needs hub_sites."),
        KeyRow("hub_sites", ("string", "array"), None,
               "Sites to sample every hub_sample_interval_s: a sites.v1 "
               "JSON path or a list of {site_id, lat, lon} tables."),
    )


def parse_hub_wind(table, *, source: str = "<config>", base_dir=None
                   ) -> HubWindRequest | None:
    """The ``hub_*`` keys of one ``[output]`` table, or ``None`` (off).

    Off unless asked: ``hub_wind = true`` turns the diagnostic on at
    :data:`DEFAULT_HUB_HEIGHTS_M`; ``hub_heights_m`` on its own turns it on
    at those heights.  ``hub_sites`` adds per-site sampling every
    ``hub_sample_interval_s`` (default :data:`DEFAULT_HUB_SAMPLE_INTERVAL_S`).
    Every contradiction is refused by name rather than resolved.
    """
    keys = {"hub_wind", "hub_heights_m", "hub_sample_interval_s",
            "hub_sites"} & set(table)
    if not keys:
        return None
    switch = table.get("hub_wind")
    if switch is not None and not isinstance(switch, bool):
        raise HubWindConfigError(
            f"[output] hub_wind of {source} must be true or false, got "
            f"{switch!r}.")
    if switch is False:
        others = sorted(keys - {"hub_wind"})
        if others:
            raise HubWindConfigError(
                f"[output] of {source} sets hub_wind = false together with "
                f"{others}.  Those keys configure the diagnostic the switch "
                "turns off -- delete them, or set hub_wind = true.")
        return None
    heights = DEFAULT_HUB_HEIGHTS_M
    if "hub_heights_m" in table:
        raw = table["hub_heights_m"]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
            raise HubWindConfigError(
                f"[output] hub_heights_m of {source} must be a LIST of "
                f"heights in m above ground, e.g. [80, 100, 120], got {raw!r}.")
        if not raw:
            raise HubWindConfigError(
                f"[output] hub_heights_m of {source} is an empty list.  Name "
                "the heights you want, or set hub_wind = true for the "
                f"defaults {list(DEFAULT_HUB_HEIGHTS_M)}.")
        heights = tuple(_finite_number(h, f"[output] hub_heights_m of {source}")
                        for h in raw)
    if len(heights) > MAX_HUB_HEIGHTS:
        raise HubWindConfigError(
            f"[output] hub_heights_m of {source} names {len(heights)} heights; "
            f"at most {MAX_HUB_HEIGHTS} are carried.")
    for height in heights:
        if not 0.0 < height <= MAX_HUB_HEIGHT_M:
            raise HubWindConfigError(
                f"[output] hub_heights_m of {source} contains {height:g} m; "
                f"every height must be in (0, {MAX_HUB_HEIGHT_M:g}] m above "
                "ground.")
    if list(heights) != sorted(set(heights)):
        raise HubWindConfigError(
            f"[output] hub_heights_m of {source} must be strictly increasing "
            f"with no repeats, got {list(heights)}.")
    sites: tuple[tuple[str, float, float], ...] = ()
    origin = None
    interval = None
    if "hub_sites" in table:
        sites, origin = _parse_sites(table["hub_sites"], base_dir=base_dir,
                                     source=source)
        interval = DEFAULT_HUB_SAMPLE_INTERVAL_S
    if "hub_sample_interval_s" in table:
        if not sites:
            raise HubWindConfigError(
                f"[output] of {source} sets hub_sample_interval_s without "
                "hub_sites: the interval is the cadence of per-site sampling "
                "and there are no sites to sample.  Add hub_sites or delete "
                "the key.")
        interval = _finite_number(table["hub_sample_interval_s"],
                                  f"[output] hub_sample_interval_s of {source}")
        if interval <= 0.0:
            raise HubWindConfigError(
                f"[output] hub_sample_interval_s of {source} must be "
                f"positive, got {interval:g}.")
    return HubWindRequest(heights_m=tuple(float(h) for h in heights),
                          sample_interval_s=interval, sites=sites,
                          sites_source=origin)

#: How many dropped names the plan-time warning spells out before it
#: switches to a count.  See :meth:`HistorySelection.warn_lost_products`.
_NAMED_DROP_LIMIT = 12


#: woof's render product catalog, as the wrfout variables each product
#: is drawn from.  ``A|B`` means either input satisfies the requirement.
#:
#: ONE table, read by three callers: the matplotlib engine's
#: availability listing and render pre-check
#: (:func:`woof.render.missing_declared_inputs`), the plan-time
#: dependency warning below, and the render front door's named refusal.
#: Two tables describing one dependency is how a listing and a render
#: come to disagree about whether a product can be drawn.
#:
#: The rust-catalog rows are the slugs
#: ``woof.render.RUST_PRODUCT_ALIASES`` maps the shared names onto, plus
#: the generic ``var:`` token for a stored variable the engine draws
#: directly; their inputs are the wrfout variables whose absence the
#: renderer itself reports as missing-fields.  Slugs beyond these are
#: still not claimed: the renderer owns its whole catalog, it answers
#: for their inputs against a real file through ``--list-products``, and
#: inventing rows here for recipes this module has not verified would be
#: the correct-looking-wrong-mapping defect that ``wind10`` already cost
#: once.  What a caller with a WIDER verified mapping does is pass it to
#: :func:`lost_products`, which is the seam that exists for exactly that
#: and keeps these rows as the answer when nobody has one.
PRODUCT_HISTORY_INPUTS: dict[str, tuple[str, ...]] = {
    # The shared product names (the matplotlib engine's whole catalog).
    "refl": ("REFL_10CM",),
    "t2": ("T2",),
    "wind10": ("U10", "V10"),
    "precip": ("RAINC|RAINNC",),
    # OLR is present exactly when the attached longwave scheme produced
    # it, so a radiation-off run lists this product as missing-fields
    # rather than failing -- which is the accurate report of that run.
    "olr": ("OLR",),
    # The GENERIC token for the same field.  The rust engine imports
    # WRF's OLR by default and draws any stored 2-D variable through
    # ``--products var:<name>``, so ``var:wrf_olr`` is how OLR is asked
    # for on that lane -- and without this row a run that shed OLR met a
    # bare engine skip after its run directory had been claimed instead
    # of the named [output] refusal at the door.
    "var:wrf_olr": ("OLR",),
    # The rust catalog slugs the shared names resolve to.
    "composite_reflectivity": ("REFL_10CM",),
    "2m_temperature": ("T2",),
    "10m_wind_speed_and_direction": ("U10", "V10"),
    "total_qpf": ("RAINC|RAINNC",),
}


def _requirement_lost(requirement: str, dropped: frozenset[str]) -> bool:
    """True when every option of one ``A|B`` requirement was dropped."""
    return all(option in dropped for option in requirement.split("|"))


def lost_products(dropped, requirements=None) -> dict[str, tuple[str, ...]]:
    """Which catalog products these dropped variables kill, and by what.

    The value is the requirement(s) that can no longer be met, spelled
    the way the table spells them (``RAINC or RAINNC`` for an either-or
    row), so the reader gets the mapping and not merely a count.

    ``requirements`` replaces :data:`PRODUCT_HISTORY_INPUTS` for a
    caller that HAS a wider verified mapping, in the same
    ``slug -> ("A|B", ...)`` grammar over WRFOUT VARIABLE NAMES.  It is
    deliberately not the renderer's own requirement rows: those are
    spelled in the store's selector vocabulary, and a selector is not a
    netCDF variable name -- the hop between them runs through wrf-core
    diagnostics that declare no input list at all, so joining them here
    would invent exactly the kind of correct-looking wrong mapping this
    table's own note refuses.  Absent, the verified rows answer, which
    is what every caller in the tree does today.
    """
    dropped = frozenset(dropped)
    table = PRODUCT_HISTORY_INPUTS if requirements is None else dict(requirements)
    lost: dict[str, tuple[str, ...]] = {}
    for product, requirements in table.items():
        missing = tuple(" or ".join(requirement.split("|"))
                        for requirement in requirements
                        if _requirement_lost(requirement, dropped))
        if missing:
            lost[product] = missing
    return lost


def _did_you_mean(name: str, known) -> str:
    close = difflib.get_close_matches(name, sorted(known), n=3, cutoff=0.6)
    return "" if not close else f"  Did you mean {', '.join(close)}?"


@dataclass(frozen=True)
class HistorySelection:
    """One resolved ``[output]`` block: which history variables to write.

    ``preset`` names the base set (``full`` -- the DEFAULT -- keeps
    everything the run produces).  ``history_vars`` replaces the base
    with an explicit include list; ``history_drop`` subtracts from it.
    The two lists are mutually exclusive, and so are ``preset`` and
    ``history_vars``: they are two spellings of the same include list
    and there is no rule for which wins.
    """

    preset: str = "full"
    history_vars: tuple[str, ...] = ()
    history_drop: tuple[str, ...] = ()
    #: Where this block was read from, for refusal text only.
    source: str = "<config>"
    #: The in-model hub-height wind diagnostic this domain asked for, or
    #: ``None`` (the default: no diagnostic, no hub fields, nothing moves).
    #: It lives here and not on RunConfig for the reason the rest of
    #: [output] does: it reads model state and writes only its own
    #: buffers, so it binds no restart identity.
    hub_wind: HubWindRequest | None = None

    # ``_include`` -- the resolved include set, or None for "everything
    # produced" -- is NOT a dataclass field, deliberately.
    # ``dataclasses.asdict`` walks every field, and both the
    # restart-identity payload and the resolved-TOML round trip walk the
    # experiment that way; a frozenset in that walk is not
    # JSON-serializable.  It is derived state, so __post_init__ sets it
    # as a plain attribute.

    def __post_init__(self) -> None:
        if self.preset not in HISTORY_PRESETS:
            raise ValueError(
                f"[output] preset = {self.preset!r} in {self.source} is not "
                f"a known history preset; choose one of "
                f"{sorted(HISTORY_PRESETS)}.  'full' is the DEFAULT and "
                "writes every variable the run produces; 'minimal' keeps "
                "the 2-D surface state and the accumulators; 'severe' adds "
                "the storm-scale volumes (winds, theta, hydrometeors, "
                "REFL_10CM); 'energy' keeps what a power-grid site forecast "
                "samples (column winds and theta, hydrometeors, the "
                "near-surface state, SWDOWN/SWDDNI/SWDDIF/COSZEN).")
        if self.history_vars and self.history_drop:
            raise ValueError(
                f"[output] of {self.source} sets both history_vars and "
                "history_drop.  They are an include list and an exclude "
                "list over the same inventory and there is no rule for "
                "which wins -- choose one: history_vars to name what the "
                "tape keeps, history_drop to name what it sheds.")
        if self.preset != "full" and self.history_vars:
            raise ValueError(
                f"[output] of {self.source} sets preset = "
                f"{self.preset!r} together with history_vars.  Both are "
                "include lists over the same inventory -- choose one: the "
                "preset for a named set, history_vars for your own.  A "
                "preset MAY be trimmed further with history_drop.")
        for key, names in (("history_vars", self.history_vars),
                           ("history_drop", self.history_drop)):
            for name in names:
                if name not in HISTORY_VOCABULARY:
                    raise ValueError(
                        f"[output] {key} of {self.source} names "
                        f"{name!r}, which is not a wrfout history "
                        f"variable.{_did_you_mean(name, HISTORY_VOCABULARY)}"
                        "  The valid set is: "
                        f"{', '.join(sorted(HISTORY_VOCABULARY))}.")
        for name in self.history_drop:
            if name in STRUCTURAL_FIELDS:
                raise ValueError(
                    f"[output] history_drop of {self.source} names "
                    f"{name!r}, which is structural: it is part of the "
                    "georeference, the vertical coordinate or the time "
                    "coordinate, and a wrfout without it is not a smaller "
                    "file but an unreadable one -- no reader can place a "
                    "value in space or time.  The structural set is "
                    f"{sorted(STRUCTURAL_FIELDS)}.  Drop the science "
                    "fields you do not need instead, or use preset = "
                    "'minimal'.")
        include: frozenset[str] | None
        if self.history_vars:
            include = frozenset(self.history_vars)
        else:
            include = HISTORY_PRESETS[self.preset]
        object.__setattr__(self, "_include", include)
        if self.hub_wind is not None and not any(
                self.keeps(name) for name in HUB_WIND_OUTPUT_FIELDS
                if name not in ("HUB_HEIGHT", "HUB_WINDOW_S")):
            raise HubWindConfigError(
                f"[output] of {self.source} asks for hub-height wind but its "
                "selection keeps none of "
                f"{list(HUB_WIND_OUTPUT_FIELDS[2:])}: the diagnostic would "
                "run every step and publish nothing.  Keep at least one of "
                "them, or delete the hub_* keys.")

    @classmethod
    def from_mapping(cls, table, *, source: str = "<config>", base_dir=None
                     ) -> "HistorySelection":
        """Validate one ``[output]`` table.  ``None`` gives :data:`FULL`.

        ``base_dir`` is the config file's directory: a relative
        ``hub_sites`` path resolves from there first, then from the
        working directory.
        """
        if table is None:
            return FULL
        if not isinstance(table, dict):
            raise ValueError(
                f"[output] of {source} must be a table, e.g. "
                '[output] preset = "minimal", got '
                f"{table!r}.")
        unknown = sorted(set(table) - OUTPUT_KEYS)
        if unknown:
            suggestions = "".join(
                _did_you_mean(key, OUTPUT_KEYS) for key in unknown)
            raise ValueError(
                f"unknown key(s) {unknown} in [output] of {source}; known "
                f"keys: {sorted(OUTPUT_KEYS)}.{suggestions}")
        for key in ("history_vars", "history_drop"):
            value = table.get(key)
            if value is None:
                continue
            if isinstance(value, str) or not isinstance(value, (list, tuple)):
                raise ValueError(
                    f"[output] {key} of {source} must be a LIST of wrfout "
                    f'variable names, e.g. {key} = ["QCLOUD", "QRAIN"], '
                    f"got {value!r}.")
            for name in value:
                if not isinstance(name, str):
                    raise ValueError(
                        f"[output] {key} of {source} must contain variable "
                        f"NAMES; {name!r} is not a string.")
            if not value:
                # WRITTEN AND EMPTY, which is not the same as absent: an
                # empty include list would write a bare structural
                # skeleton with no science in it, and an empty exclude
                # list is a key that does nothing.  Both are far more
                # likely to be a half-finished edit than an intent.
                raise ValueError(
                    f"[output] {key} of {source} is an empty list.  An "
                    "empty history_vars would write a structural skeleton "
                    "with no science in it, and an empty history_drop "
                    "sheds nothing -- name the variables you mean, or "
                    "delete the key.")
        return cls(
            preset=str(table.get("preset", "full")),
            history_vars=tuple(table.get("history_vars", ()) or ()),
            history_drop=tuple(table.get("history_drop", ()) or ()),
            source=source,
            hub_wind=parse_hub_wind(table, source=source, base_dir=base_dir),
        )

    @property
    def writes_everything(self) -> bool:
        """``True`` when this selection removes nothing from any frame."""
        return self._include is None and not self.history_drop

    def keeps(self, name: str) -> bool:
        """Whether one produced variable survives this selection."""
        if name in STRUCTURAL_FIELDS:
            return True
        if name in self.history_drop:
            return False
        return self._include is None or name in self._include

    def select(self, produced) -> tuple[str, ...]:
        """The kept subset of ``produced``, IN THE PRODUCED ORDER.

        Order is not cosmetic: HDF5 lays its name heap out in
        variable-creation order, so the same numbers written in a
        different order give a file that hashes differently while every
        variable in it compares equal.  The filter may only remove.
        """
        return tuple(name for name in produced if self.keeps(name))

    def dropped(self, produced) -> tuple[str, ...]:
        """The subset of ``produced`` this selection removes."""
        return tuple(name for name in produced if not self.keeps(name))

    def wrfout_attrs(self, produced) -> dict[str, str]:
        """Global attributes recording what this selection did.

        A trimmed file must SAY it was trimmed and name what it shed:
        that is what lets the render front door answer "this run dropped
        REFL_10CM" instead of the much weaker "this file has no
        REFL_10CM".  A full-default run stamps NOTHING, so every wrfout
        written before this feature keeps its exact header.
        """
        if self.writes_everything:
            return {}
        dropped = self.dropped(produced)
        return {
            "GPUWM_HISTORY_PRESET": self.preset,
            "GPUWM_HISTORY_SELECTION": self.spelling(),
            "GPUWM_HISTORY_DROPPED": ",".join(dropped),
        }

    def apply(self, frame) -> tuple[dict, dict[str, str]]:
        """(kept frame, history attributes) for an ALREADY-ASSEMBLED frame.

        The seam for a writer that has the whole host frame in hand --
        ``woof downscale``'s offline child, which builds one frame and
        writes it synchronously.  The asynchronous per-domain writer does
        NOT use this: it has to know the kept set BEFORE it stages, so a
        dropped volume never crosses the bus, and it resolves the same
        selection through its own ``_history_plan``.

        A full-default selection returns the frame object unchanged and
        an empty attribute set, so nothing about a default run moves.
        """
        if self.writes_everything:
            return frame, {}
        produced = tuple(frame)
        return ({name: frame[name] for name in self.select(produced)},
                self.wrfout_attrs(produced))

    def spelling(self) -> str:
        """The one-line form of this block, for receipts and messages."""
        parts = [f"preset={self.preset}"]
        if self.history_vars:
            parts.append("history_vars=" + ",".join(self.history_vars))
        if self.history_drop:
            parts.append("history_drop=" + ",".join(self.history_drop))
        if self.hub_wind is not None:
            parts.append(self.hub_wind.spelling())
        return " ".join(parts)

    def product_cost(self) -> tuple[int, int, bool]:
        """``(products kept, products in the full inventory, exact)``.

        Computed from :data:`PRESET_PRODUCT_COUNTS` INSIDE this object,
        never from what a caller passes as ``produced``: the two
        plan-review doors hand in different inventories, and a
        configuration that prints two different figures depending on
        which door read it is the disagreement this lane exists to end.

        ``exact`` is False for a selection that is not exactly a named
        preset.  ``history_vars`` and ``history_drop`` narrow the tape
        further than the preset they start from, so the preset's figure
        is an UPPER BOUND there, and the caller says so rather than
        printing a measured-looking number nobody measured.
        """

        exact = not (self.history_vars or self.history_drop)
        separate = SEPARATELY_MEASURED_PRESET_PRODUCT_COUNTS.get(self.preset)
        if separate is not None and not self.history_vars:
            return separate[0], separate[1], exact
        total = PRESET_PRODUCT_COUNTS["full"]
        kept = PRESET_PRODUCT_COUNTS.get(self.preset, total)
        return kept, total, exact

    def warn_lost_products(self, produced, *, where: str) -> None:
        """Name the render products this selection kills, at plan time.

        The user asked for storage back; what they get in exchange is a
        set of pictures they can no longer draw, and they should read
        that BEFORE the run rather than discover it at render time.  It
        is a warning and not a refusal: shedding a product deliberately
        is exactly what this feature is for.

        It LEADS WITH PRODUCTS.  A variable count answers "how much disk
        did I save"; the reader is choosing between pictures and disk,
        and the product figure is the other half of that trade.  It used
        to be absent entirely, so ``preset = "minimal"`` reported two
        products by name and a bare variable count while the measured
        ladder in this module recorded that the same preset leaves 30 of
        162 renderable.
        """
        dropped = self.dropped(produced)
        if not dropped:
            return
        lost = lost_products(dropped)
        if not lost:
            return
        kept, total, exact = self.product_cost()
        separate = (None if self.history_vars else
                    SEPARATELY_MEASURED_PRESET_PRODUCT_COUNTS.get(self.preset))
        if separate is not None:
            # Its own frame, its own basis: no ladder attribution, which
            # names fields this preset keeps.
            cost = (f"keeps {kept} of {total} render products, "
                    f"{total - kept} lost ({separate[2]})"
                    if exact else
                    f"keeps at most {kept} of {total} render products (an "
                    f"upper bound: this selection narrows "
                    f"preset={self.preset} further; {separate[2]})")
        elif kept < total:
            # The named rungs do not exhaust the loss: the ladder's own
            # 162 -> 160 step belongs to the rest of the inventory, and
            # a sentence whose figures do not add up reads as an error
            # in the measurement rather than as what it is.
            named = [f"{count} to {','.join(fields)}"
                     for fields, count in PRODUCT_LOSS_ATTRIBUTION]
            remainder = (total - kept) - sum(
                count for _, count in PRODUCT_LOSS_ATTRIBUTION)
            if remainder > 0:
                named.append(f"{remainder} to the rest of the full "
                             f"inventory")
            attribution = ", ".join(named)
            cost = (f"keeps {kept} of {total} render products, "
                    f"{total - kept} lost: {attribution}"
                    if exact else
                    f"keeps at most {kept} of {total} render products "
                    f"(an upper bound: this selection narrows "
                    f"preset={self.preset} further)")
        else:
            cost = (f"keeps all {total} render products"
                    if exact else
                    f"keeps at most {total} render products "
                    "(an upper bound: this selection narrows the preset "
                    "it starts from)")
        mapping = "; ".join(
            f"{product} needs {', '.join(missing)}"
            for product, missing in sorted(lost.items()))
        # NAME the drops while naming them is readable; COUNT them once
        # it is a wall.  A preset sheds well over a hundred scheme
        # carriers, and printing all of them (measured: >1,700 characters
        # on one real d02) buries the actionable half -- which product
        # dies of which variable -- under state nobody was reading.
        # Sorted, not in produced order: the plan-time caller passes the
        # VOCABULARY (a frozenset), whose iteration order varies per
        # process, and a warning that reorders itself between two runs of
        # the same config is a warning a reader cannot diff.
        shed = (", ".join(sorted(dropped))
                if len(dropped) <= _NAMED_DROP_LIMIT
                else f"{len(dropped)} variables")
        warn(
            f"{where}: [output] {self.spelling()} {cost}.  It drops "
            f"{shed}, so these render products can no longer be drawn "
            f"from this run's history: {mapping}.  Keep the variable(s) "
            "if you want the product(s); otherwise this is the storage "
            "you asked for.",
            "The product figure comes from this module's own measured "
            f"ladder ({PRODUCT_COUNT_BASIS}), read as products rather "
            "than as prose.  The NAMED mapping beside it is a narrower "
            "thing: woof.io.history_selection.PRODUCT_HISTORY_INPUTS "
            f"carries {len(PRODUCT_HISTORY_INPUTS)} verified rows, so of "
            f"the {PRESET_PRODUCT_COUNTS['full']} products the measured "
            "full inventory renders, roughly "
            f"{PRESET_PRODUCT_COUNTS['full'] - len(PRODUCT_HISTORY_INPUTS)} "
            "were NOT checked by name here.  The renderer answers for "
            "those itself, against a real file, through --list-products: "
            "a product named above will refuse by name at render time, "
            "and one that is not named may still be lost.")


#: The default: write everything the run produces, stamp nothing.
FULL = HistorySelection()


def resolve(tree, domain) -> HistorySelection:
    """One domain's selection: its own block, else the tree-wide one.

    A domain that says nothing takes the tree-wide table; a domain that
    speaks overrides it ENTIRELY rather than merging key by key, on the
    ``[tiles]`` precedent -- a half-inherited selection ("the preset from
    the tree, the drop list from the domain") is a configuration nobody
    can read off the file.
    """
    if domain is not None:
        return domain
    return FULL if tree is None else tree


__all__ = [
    "FULL",
    "HISTORY_PRESETS",
    "PRESET_PRODUCT_COUNTS",
    "PRODUCT_COUNT_BASIS",
    "PRODUCT_LOSS_ATTRIBUTION",
    "SEPARATELY_MEASURED_PRESET_PRODUCT_COUNTS",
    "HISTORY_VOCABULARY",
    "DEFAULT_HUB_HEIGHTS_M",
    "DEFAULT_HUB_SAMPLE_INTERVAL_S",
    "HistorySelection",
    "HubWindConfigError",
    "HubWindRequest",
    "OUTPUT_KEYS",
    "PRODUCT_HISTORY_INPUTS",
    "STRUCTURAL_FIELDS",
    "lost_products",
    "parse_hub_wind",
    "resolve",
]
