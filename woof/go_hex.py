"""``woof go`` on the hex model: ``[model] engine = "hex"`` in the experiment TOML.

WHY THIS ROUTE EXISTS.  ``woof go`` is the WRF path's end-to-end workflow
(fetch, prepare, forecast, render), and it could not run the hex model at all:
a hex forecast had only its own doors (``woof hex regional``/``global``).  This
module is the hex half of ``go``.  It reads the same experiment TOML, refuses
what hex cannot do yet by name, and drives the shipped hex regional door
(:mod:`woof.hex.regional_door`) with the region the TOML draws.  It changes no
hex door and adds no preparation of its own.

THE STAGES.  Each go stage maps onto the hex regional door's stages
(:data:`STAGE_MAP`):

* ``fetch``    -> ``intermediate``: the source's files through hex's source
                  table (``woof hex sources``; the table sources fetch through
                  woof's own acquisition), decoded to WPS intermediates;
* ``prepare``  -> ``mesh``, ``init``, ``lbc``, ``contract``: the WPS/real.exe
                  replacement (variable-resolution mesh and static over the
                  region, initial state, boundary series, and the regional
                  class admission, minted on first use);
* ``forecast`` -> ``forecast``;
* ``render``   -> ``products``.

WHAT THE TOML MEANS ON HEX.  The single ``[[domain]]`` on ``[projection]`` is
the region: its outline (the projected rectangle, through the same WRF grid
:mod:`woof.source_coverage` places for coverage) becomes the drawn fine core,
and ``dx`` the fine spacing.  ``[fetch]`` names the source and cycle, checked
by go's own :func:`woof.go_cli.checked_config_fetch_cycle`; ``run_seconds``
and ``history_interval_s`` are the forecast length and history cadence.
Physics selectors written in ``[shared]`` or the ``[[domain]]``
(``mp_physics``, ``cu_physics``, ... and the hex generation selectors) go to
``hex physics resolve`` as ``--physics KEY=VALUE``, so WRF namelist names work
unchanged; with none written the run takes the hex default (the MPAS v8.4.2
suite route, by mesh spacing and regional recipe).

WHAT IS REFUSED.  Every other key and table is refused by name, with the
parity-matrix item (``HEX-WRF-PARITY.md``) that will add it: a key hex does
not read would otherwise run the hex defaults under the name of your setting.
The default engine stays WRF: a TOML without ``[model]`` never reaches here.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

#: go stage -> the hex regional door's stages that do its work.
STAGE_MAP: Mapping[str, tuple[str, ...]] = {
    "fetch": ("intermediate",),
    "prepare": ("mesh", "init", "lbc", "contract"),
    "forecast": ("forecast",),
    "render": ("products",),
}

#: Tables the hex route reads.  ``[output]`` is read for its format keys only.
READ_TABLES = ("model", "experiment", "projection", "domain", "shared", "fetch", "output", "static")

#: Tables with no hex equivalent on the go route yet, each with the matrix item that adds it.
REFUSED_TABLES: Mapping[str, str] = {
    "ensemble": "go ensembles on hex are matrix item 3.3 (stream, multi-run and go-job for hex); "
                "`woof hex global --members` runs hex members today",
    "nudging": "grid and spectral nudging on hex are matrix item 5.1",
    "spectral_nudging": "grid and spectral nudging on hex are matrix item 5.1",
    "wrf_hydro": "WRF-Hydro on hex is the in-flight hex hydro item P7",
    "relocation": "a moving nest on hex is `woof hex moving` (matrix A.9); its go route is matrix item 3.3",
    "perturbation": "initial theta bubbles are WRF idealized-case inputs; hex ideal cases are matrix A.8 "
                    "(partial, no user door)",
    "tiles": "out-of-core tiles are N/A on the mesh (matrix A.17): the mesh partitions instead",
    "devices": "multi-card hex runs are `woof hex regional --samehost-devices`; the go mapping is matrix item 3.3",
    "simulated_radar": "the simulated-radar door for hex is matrix item 2.7",
    "radar_heating": "radar latent heating has no hex item yet (matrix A.10, missing)",
    "spectral_numerics": "WRF spectral numerics have no hex equivalent (hex runs MPAS dynamics, matrix A.8)",
    "physics_params": "per-scheme parameter tables have no hex go mapping yet (matrix item 1.3)",
    "case_data": "[case_data] is the WRF preparation's input table; hex prepares through its own "
                 "mesh/init doors from [fetch]",
    "ingest": "[ingest] is WRF ingest policy; hex ingests through its source table",
    "lineage": "cyclone lineage is a WRF go provenance table; hex storm workflows are `woof hex moving`",
}

#: [experiment] keys the hex route reads.
EXPERIMENT_KEYS = ("name", "start_time", "run_seconds")
#: [experiment] keys accepted only at the value that makes them inert on hex.
EXPERIMENT_INERT = {"restart_interval_s": 0.0}
#: [[domain]] keys the hex route reads.
DOMAIN_KEYS = ("nx", "ny", "dx", "dy", "history_interval_s")
#: [[domain]] keys accepted only at their root-domain values (one domain, no parent).
DOMAIN_ROOT = {"grid_id": 1, "parent_id": 0, "i_parent_start": 1, "j_parent_start": 1,
               "parent_grid_ratio": 1, "parent_time_step_ratio": 1, "specified": True, "nested": False}
#: [fetch] keys hex reads (source, cycle, forecast_start_hour) or that are WRF go's
#: advisory download hints: hex's own source door decides what to fetch.
FETCH_KEYS = ("source", "cycle", "forecast_start_hour", "hours", "area", "out", "cadence")
#: [projection] keys the region outline reads (the WRF root grid :mod:`woof.source_coverage` places).
PROJECTION_KEYS = ("map_proj", "ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon")
#: [output] keys the hex forecast door reads (``woof.output_formats``).
OUTPUT_KEYS = ("formats", "icechunk_store", "icechunk_delete_after_commit")

#: Keys refused by name with their reason, wherever they are written.
_VERTICAL = ("hex takes its vertical grid from the regional row's vertical_spec (countries/tc55-v1.json); "
             "a WRF eta ladder has no hex mapping yet (3.1 follow-up)")
_STEP = ("hex steps at the mesh row's dt (`--dt row`), the step its regional class is minted at; "
         "adaptive stepping is matrix item 4.1")
KEY_REASONS: Mapping[str, str] = {
    **{key: _VERTICAL for key in ("nz", "e_vert", "eta_levels", "ztop", "p_top", "hybrid_opt", "etac",
                                  "auto_levels_opt", "max_dz", "dzbot", "dzstretch_s", "dzstretch_u")},
    **{key: _STEP for key in ("time_step", "time_step_fract_num", "time_step_fract_den",
                              "use_adaptive_time_step")},
    "max_dom": "nests are N/A on the mesh (matrix A.9): one variable-resolution mesh",
    "partial_nest": "a partial nest on hex is matrix item 5.6",
    "spec_bdy_width": "hex's regional zones are MPAS's fixed nSpecZone=2/nRelaxZone=5 (matrix A.9)",
    "spec_zone": "hex's regional zones are MPAS's fixed nSpecZone=2/nRelaxZone=5 (matrix A.9)",
    "relax_zone": "hex's regional zones are MPAS's fixed nSpecZone=2/nRelaxZone=5 (matrix A.9)",
    "blend_width": "hex's regional zones are MPAS's fixed nSpecZone=2/nRelaxZone=5 (matrix A.9)",
    "feedback": "nest feedback is N/A on the mesh (matrix item 5.8)",
    "smooth_option": "nest feedback smoothing is N/A on the mesh (matrix item 5.8)",
    **{key: (f"SPP on hex is `woof hex forecast --physics {key}=1` on the generic column seam (matrix item 5.4); "
             "the go route has no mapping yet") for key in ("spp_conv", "spp_pbl", "spp_lsm")},
    "multi_perturb": "WRF-Solar EPS on hex is the MPAS suite route's &wrfsolar_eps (`woof hex forecast --physics config_multi_perturb=true ...`, global and regional, matrix item 5.5); the go route has no mapping yet",
    "grid_fdda": "grid and spectral nudging on hex are matrix item 5.1",
    "obs_nudge_opt": "obs nudging on hex is matrix item 5.2",
    "dfi_opt": "DFI on hex is matrix item 4.4 (IAU is the hex alternative)",
    "w_damping": "w_damping on hex is `woof hex forecast --w-damping` (matrix item 4.2); the go route has no mapping yet",
}


#: [static.highres] terrain_source values -> ``woof hex regional --terrain-source``.  The WRF registry ids
#: and the hex door's names are both accepted; "auto", 3DEP and SRTM have no hex fetcher.
TERRAIN_SOURCE_NAMES: Mapping[str, str] = {"glo30": "glo30", "copernicus-dem-glo30": "glo30",
                                           "gem15-dtm": "gem15-dtm", "gem15-dsm": "gem15-dsm"}
#: [static.highres] keys accepted only at the value that makes them inert when only terrain is replaced.
STATIC_INERT: Mapping[str, Any] = {"fields": "terrain", "landcover_source": "auto", "soil_source": "soilgrids",
                                   "lcz_pervious": "urban", "on_refuse": "error"}


class HexGoRefusal(ValueError):
    """A go request the hex route will not half-run; the message names the matrix item."""


# ---------------------------------------------------------------------------
# the engine switch
# ---------------------------------------------------------------------------
def read_document(config) -> dict | None:
    """The TOML at ``config``, or None when it is not a readable TOML file.

    None leaves the launch to the WRF path, which reports an unreadable file in
    its own words exactly as before this route existed."""

    if config is None:
        return None
    path = Path(config)
    if path.name == "namelist.input" or not path.is_file():
        return None
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None
    return document if isinstance(document, dict) else None


def configured_engine(config) -> str:
    """``"hex"`` when CONFIG's ``[model]`` table names hex; ``"wrf"`` otherwise (and when it has none)."""

    document = read_document(config)
    if document is None or "model" not in document:
        return "wrf"
    from woof.config import model_engine
    try:
        return model_engine(document["model"], where=f"[model] of {config}")
    except ValueError as error:
        raise HexGoRefusal(str(error)) from None


# ---------------------------------------------------------------------------
# the go flags
# ---------------------------------------------------------------------------
#: go flags the hex route reads.  ``prepare_only`` stops after the prepare stages;
#: ``render_products`` takes ``none`` (end at the history frames).
READ_FLAGS = frozenset({"config", "outdir", "dry_run", "cycle", "card", "prepare_only", "render_products",
                        "output_config", "output_format", "icechunk_store", "icechunk_delete_after_commit",
                        "explain", "command", "handler", "func"})

#: go flags with no hex route yet, by the matrix item that adds them.
FLAG_ITEMS: Mapping[str, str] = {
    "restart": "3.2 (`woof resume` / `woof branch` for hex checkpoints)",
    "members": "3.3 (go ensembles for hex; `woof hex global --members` today)",
    "storm_nests": "A.9 (nests are N/A on the mesh; `woof hex moving` follows storms)",
    "nest_at": "A.9 (nests are N/A on the mesh; `woof hex add-nest` hangs a WRF child)",
    "nudging": "5.1 (grid and spectral nudging on hex)",
    "hydro": "P7 (hex hydro)",
    "wind_farms": "3.1 follow-up (`woof hex forecast --wind-farms` exists; go does not forward it yet)",
    "tiles": "A.17 (out-of-core tiles are N/A on the mesh)",
    "smoke": "3.1 follow-up (smoke runs on the hrrrv5 suite through `woof hex regional`)",
    "prepared_root": "3.3 (go-job and prepared bundles for hex)",
    "branch_from": "3.2 (`woof branch` for hex)",
    "resume_ensemble": "3.3",
    "restart_roster": "3.3",
    "readiness": "3.3",
    "grib2": "2.8 (hex GRIB2 export)",
    "render_section": "2.7 / 3.1 follow-up (hex products draw no vertical section)",
}


def _go_defaults() -> dict[str, Any]:
    from woof.go_cli import register_cli

    parser = argparse.ArgumentParser(add_help=False)
    register_cli(parser.add_subparsers(dest="command"))
    go = parser._subparsers._group_actions[0].choices["go"]
    return {action.dest: action.default for action in go._actions if action.dest != "help"}


def refuse_flags(args) -> None:
    """Refuse, by name, every go flag set away from its default that the hex route does not read."""

    defaults = _go_defaults()
    named = []
    if getattr(args, "command", None) == "ensemble":
        named.append("woof ensemble (3.3: go ensembles for hex; `woof hex global --members` today)")
    for dest, default in defaults.items():
        if dest in READ_FLAGS or not hasattr(args, dest):
            continue
        value = getattr(args, dest)
        if value == default:
            continue
        flag = "--" + dest.replace("_", "-")
        item = FLAG_ITEMS.get(dest, "3.3 (the rest of go's workflow flags for hex)")
        named.append(f"{flag} ({item})")
    products = getattr(args, "render_products", None)
    if products is not None and str(products).strip().lower() != "none":
        named.append("--products other than 'none' (hex draws its own product set; "
                     "a go product list on hex is a 3.1 follow-up)")
    if named:
        raise HexGoRefusal("[model] engine = 'hex' does not run these woof go flags yet: "
                           + "; ".join(named) + ".  Drop them, or run the WRF engine.")


# ---------------------------------------------------------------------------
# the TOML
# ---------------------------------------------------------------------------
def _physics_keys() -> frozenset[str]:
    from woof.hex.physics_suites import engine_selector_aliases, generation_keys

    return frozenset(engine_selector_aliases()) | frozenset(generation_keys())


def _chem_reason(key: str) -> str | None:
    from woof.config import CHEM_RUN_FIELDS, FIRE_RUN_FIELDS

    if key in FIRE_RUN_FIELDS or key == "ifire":
        return "SFIRE on hex is matrix item 6.1"
    if key in CHEM_RUN_FIELDS or key.startswith(("chem", "emiss", "bio_emiss")):
        return ("chemistry from go on hex waits on the hex chem route on the default suite "
                "(P4, in flight) and GOCART oxidants on the mesh (matrix item 6.2)")
    return None


def _key_refusal(table: str, key: str) -> str:
    reason = KEY_REASONS.get(key) or _chem_reason(key) or (
        "it has no hex mapping on the go route (hex runs MPAS dynamics and its own namelist; "
        "a WRF dynamics or namelist key would be silently dropped)")
    return f"[{table}] {key}: {reason}"


def _terrain_request(static: Any, *, source: str, base_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """The ``[static.highres]`` block as a hex terrain request (or None), and what it asks that hex cannot do.

    The block is validated by the WRF path's own parser (:func:`woof.static.highres_production.
    parse_static_table`); hex then accepts the part its mesh stage runs (``woof hex static-highres``
    terrain: GLO-30, GEM-15) and refuses the rest by name."""

    if not isinstance(static, dict):
        return None, [f"[static]: must be a table, got {static!r}"]
    refusals = [f"[static] {key}: hex builds its static from WPS_GEOG in the mesh stage and takes no "
                "external static or geography root on the go route" for key in static if key != "highres"]
    table = static.get("highres")
    if table is None or not isinstance(table, dict):
        return None, refusals or ["[static] declares no [static.highres] table"]
    table = dict(table)
    wanted = table.get("terrain_source")
    if wanted == "glo30":
        table["terrain_source"] = "copernicus-dem-glo30"  # the WRF registry id, which its parser knows
    from woof.static.highres_production import parse_static_table
    try:
        config = parse_static_table({"highres": table}, source=source, base_dir=base_dir)
    except ValueError as error:
        return None, refusals + [str(error)]
    if refusals or config is None or not config.enabled:
        return None, refusals
    for key, inert in STATIC_INERT.items():
        value = table.get(key, inert)  # unwritten is the terrain-only run the hex mesh stage makes
        if value != inert:
            refusals.append(f"[static.highres] {key} = {value!r}: the hex mesh stage replaces terrain only "
                            f"(`woof hex static-highres --terrain`); {key} is accepted only as {inert!r}, "
                            "so it is not silently dropped (hex land use is `woof hex static-highres --landuse`)")
    if "terrain_source" not in table:
        refusals.append("[static.highres] terrain_source: name it (glo30, gem15-dtm, gem15-dsm); hex has no 'auto'")
    elif TERRAIN_SOURCE_NAMES.get(str(wanted)) is None:
        refusals.append(f"[static.highres] terrain_source = {wanted!r}: hex static-highres serves "
                        f"{', '.join(sorted(set(TERRAIN_SOURCE_NAMES.values())))} (3DEP, SRTM and 'auto' "
                        "have no hex fetcher)")
    if refusals:
        return None, refusals
    return {"source": TERRAIN_SOURCE_NAMES[str(wanted)], "cache_root": str(config.cache_root),
            "max_dx_m": config.max_dx_m}, []


def read_request(document: Mapping[str, Any], *, source: str = "CONFIG") -> dict[str, Any]:
    """The hex request a go TOML makes, every unread key refused by name (all at once)."""

    refusals: list[str] = []
    for name in document:
        if name in READ_TABLES:
            continue
        reason = REFUSED_TABLES.get(name, "it has no hex equivalent on the go route yet")
        refusals.append(f"[{name}]: {reason}")
    experiment = document.get("experiment")
    if not isinstance(experiment, dict):
        raise HexGoRefusal(f"{source} must carry an [experiment] table (name, start_time, run_seconds)")
    for key, value in experiment.items():
        if key in EXPERIMENT_KEYS or (key in EXPERIMENT_INERT and value == EXPERIMENT_INERT[key]):
            continue
        refusals.append(_key_refusal("experiment", key) if key not in EXPERIMENT_INERT else
                        f"[experiment] {key} = {value!r}: hex checkpoints from go are matrix item 3.2")
    domains = document.get("domain")
    if not isinstance(domains, list) or not domains:
        raise HexGoRefusal(f"{source} must carry one [[domain]] table: the region the hex mesh refines")
    if len(domains) > 1:
        refusals.append(f"{len(domains)} [[domain]] tables: nests are N/A on the mesh (matrix A.9, "
                        "'multi-domain trees in woof go'): one variable-resolution mesh carries the "
                        "refinement, so draw the fine region as the one [[domain]]; `woof hex add-nest` "
                        "hangs a WRF child under a hex run")
    domain = domains[0]
    physics: dict[str, Any] = {}
    physics_keys = _physics_keys()
    for key, value in domain.items():
        if key in DOMAIN_KEYS:
            continue
        if key in DOMAIN_ROOT:
            if value != DOMAIN_ROOT[key]:
                refusals.append(f"[[domain]] {key} = {value!r}: a hex go run is one root domain (matrix A.9)")
            continue
        if key in physics_keys:
            physics[key] = value
            continue
        refusals.append(_key_refusal("domain", key))
    shared = document.get("shared") or {}
    for key, value in shared.items():
        if key in physics_keys:
            if key in physics and physics[key] != value:
                refusals.append(f"{key} is {value!r} in [shared] and {physics[key]!r} in [[domain]]; "
                                "a hex mesh runs one physics selection")
            physics.setdefault(key, value)
            continue
        refusals.append(_key_refusal("shared", key))
    output = document.get("output") or {}
    for key in output:
        if key not in OUTPUT_KEYS:
            refusals.append(f"[output] {key}: hex history writes its own streams; a go field selection on "
                            "hex has no matrix item yet (3.1 follow-up)")
    terrain = None
    if "static" in document:
        terrain, static_refusals = _terrain_request(document["static"], source=source,
                                                    base_dir=Path(source).expanduser().absolute().parent)
        refusals.extend(static_refusals)
    fetch = document.get("fetch")
    if not isinstance(fetch, dict) or "source" not in fetch or "cycle" not in fetch:
        refusals.append("[fetch] source and cycle: the hex route fetches the named source's cycle and "
                        "starts from it")
        fetch = {}
    for key in fetch:
        if key not in FETCH_KEYS:
            refusals.append(f"[fetch] {key}: the hex source door has no such request "
                            "(hex fetches through `woof hex sources`; 3.1 follow-up)")
    projection = document.get("projection")
    if not isinstance(projection, dict):
        refusals.append("[projection]: the hex route draws its region from the [[domain]] on this projection")
    else:
        for key in projection:
            if key not in PROJECTION_KEYS:
                refusals.append(f"[projection] {key}: the hex region outline reads only "
                                f"{', '.join(PROJECTION_KEYS)}")
        for key in PROJECTION_KEYS:
            if key not in projection:
                refusals.append(f"[projection] {key}: the hex route needs it to place the region")
    for key in ("nx", "ny", "dx", "history_interval_s"):
        if key not in domain:
            refusals.append(f"[[domain]] {key}: the hex route needs it (region size, fine spacing, "
                            "history cadence)")
    if "dy" in domain and "dx" in domain and float(domain["dy"]) != float(domain["dx"]):
        refusals.append("[[domain]] dy differs from dx: the hex fine spacing is one number")
    for key in ("start_time", "run_seconds"):
        if key not in experiment:
            refusals.append(f"[experiment] {key}: the hex route needs it")
    if refusals:
        raise HexGoRefusal(f"{source} asks the hex engine for what it cannot run yet:\n  - "
                           + "\n  - ".join(refusals))
    start = experiment["start_time"]
    if not isinstance(start, datetime) or start.tzinfo is not None:
        raise HexGoRefusal(f"[experiment] start_time = {start!r}: write an offset-free TOML datetime "
                           "(UTC, e.g. 2025-05-30T12:00:00), as the WRF experiment loader requires")
    run_seconds = float(experiment["run_seconds"])
    if run_seconds <= 0 or run_seconds % 3600:
        raise HexGoRefusal(f"[experiment] run_seconds = {run_seconds:g}: a hex regional run lasts whole hours "
                           "(one boundary file per source hour)")
    history = float(domain["history_interval_s"])
    if history <= 0 or history % 60:
        raise HexGoRefusal(f"[[domain]] history_interval_s = {history:g}: hex history is in whole minutes")
    return {"name": str(experiment.get("name") or Path(source).stem),
            "start_time": experiment["start_time"], "hours": int(run_seconds // 3600),
            "history_every_minutes": int(history // 60), "projection": dict(projection),
            "nx": int(domain["nx"]), "ny": int(domain["ny"]), "dx_m": float(domain["dx"]),
            "fetch": dict(fetch), "physics": physics, "terrain": _scoped(terrain, float(domain["dx"])),
            "output_table": bool(output), "output": dict(output)}


def _scoped(terrain: dict[str, Any] | None, dx_m: float) -> dict[str, Any] | None:
    """``max_dx_m`` scopes the block to domains at or below that spacing, as it does on the WRF path."""

    if terrain is None:
        return None
    limit = terrain.get("max_dx_m")
    if limit is not None and dx_m > float(limit):
        return None
    return {key: value for key, value in terrain.items() if key != "max_dx_m"}


def region_polygon(projection: Mapping[str, Any], nx: int, ny: int, dx_m: float, *,
                   per_side: int = 16) -> dict[str, Any]:
    """The [[domain]]'s outline on its projection, as a hex drawn-shape polygon row.

    The grid is the one :func:`woof.source_coverage._root_grid` places for go's
    coverage checks (centred on ref_lat/ref_lon as WPS places a root), and the
    outline runs along its mass-cell edges, so the hex fine core covers every
    cell the WRF domain would."""

    from woof.static.projection import is_latlon_map_proj
    from woof.source_coverage import _root_grid

    if is_latlon_map_proj(projection.get("map_proj")):
        raise HexGoRefusal("[projection] map_proj is lat-lon: a global or lat-lon run on hex is "
                           "`woof hex global`; its go route is a 3.1 follow-up")
    grid = _root_grid(dict(projection), nx, ny, dx_m)
    lo_x, hi_x, lo_y, hi_y = 0.5, nx + 0.5, 0.5, ny + 0.5
    path = []
    for k in range(per_side):
        path.append((lo_x + (hi_x - lo_x) * k / per_side, lo_y))
    for k in range(per_side):
        path.append((hi_x, lo_y + (hi_y - lo_y) * k / per_side))
    for k in range(per_side):
        path.append((hi_x - (hi_x - lo_x) * k / per_side, hi_y))
    for k in range(per_side):
        path.append((lo_x, hi_y - (hi_y - lo_y) * k / per_side))
    vertices = []
    for x, y in path:
        lat, lon = grid.ij_to_latlon(x, y)
        vertices.append([round(float(lat), 6), round(((float(lon) + 180.0) % 360.0) - 180.0, 6)])
    return {"kind": "polygon", "vertices_deg": vertices}


def background_km(fine_dx_m: float, *, target_km: float = 120.0) -> float:
    """The halving ladder that holds ``fine_dx_m`` as a rung: fine * 2^k nearest the mesh door's 120 km."""

    fine_km = float(fine_dx_m) / 1000.0
    if fine_km <= 0:
        raise HexGoRefusal(f"[[domain]] dx = {fine_dx_m!r} is not a spacing")
    k = max(0, round(math.log2(target_km / fine_km)))
    return fine_km * 2 ** k


def _cycle_and_lead(request: Mapping[str, Any], args) -> tuple[datetime, int]:
    from woof.go_cli import GoRefusal, checked_config_fetch_cycle

    fetch = dict(request["fetch"])
    lead = fetch.get("forecast_start_hour", 0)
    if isinstance(lead, bool) or not isinstance(lead, int) or lead < 0:
        # Checked here for every run: with --cycle the shared check below skips its own lead check.
        raise HexGoRefusal(f"[fetch] forecast_start_hour = {lead!r} must be a nonnegative integer")
    flag = getattr(args, "cycle", None)
    if flag is not None:
        fetch["cycle"] = flag
    # A --cycle re-times the run as go's own --cycle does: the start becomes the cycle plus its
    # lead, so the saved start_time is checked against the saved cycle only.
    start = request["start_time"] if flag is None else None
    try:
        cycle = checked_config_fetch_cycle(fetch, start_time=start)
    except GoRefusal as error:
        raise HexGoRefusal(str(error)) from None
    return cycle, lead


def work_dir(args, config: Path) -> Path:
    """go's output root: ``--outdir``, else ``<config-stem>-go`` beside the config (go's own default)."""

    outdir = getattr(args, "outdir", None)
    return Path(outdir).expanduser().absolute() if outdir is not None else \
        config.expanduser().absolute().with_name(f"{config.stem}-go")


def regional_argv(request: Mapping[str, Any], args, config: Path, *, shape: Path) -> list[str]:
    """The ``woof hex regional`` command this go run is."""

    cycle, lead = _cycle_and_lead(request, args)
    argv = ["regional", "--shape", str(shape), "--cycle", cycle.strftime("%Y-%m-%dT%H"),
            "--source", str(request["fetch"]["source"]), "--hours", str(request["hours"]),
            "--history-every-minutes", str(request["history_every_minutes"]),
            "--fine-dx-m", repr(request["dx_m"]), "--background-km", repr(background_km(request["dx_m"])),
            "--work-dir", str(work_dir(args, config))]
    if lead:
        argv += ["--lead-hours", str(lead)]
    if getattr(args, "card", None):
        argv += ["--card", str(args.card)]
    terrain = request.get("terrain")
    if terrain is not None:
        argv += ["--terrain-source", terrain["source"], "--terrain-cache-root", terrain["cache_root"]]
    for key, value in request["physics"].items():
        text = str(value).lower() if isinstance(value, bool) else str(value)
        argv += [f"--forecast-arg=--physics={key}={text}"]
    from woof.output_formats import argument_values
    argv += argument_values(args)
    if request.get("output_table") and getattr(args, "output_config", None) is None:
        argv += ["--output-config", str(config)]
    products = getattr(args, "render_products", None)
    if getattr(args, "prepare_only", False):
        argv += ["--skip", "forecast", "--skip", "products"]
    elif products is not None and str(products).strip().lower() == "none":
        argv += ["--no-products"]
    else:
        deletes = (getattr(args, "icechunk_delete_after_commit", None)
                   or (request.get("output") or {}).get("icechunk_delete_after_commit") is True)
        if deletes:
            # The products stage reads the history frames after the forecast; deleting each frame
            # once the store commits it would leave the products nothing to draw.
            raise HexGoRefusal("icechunk_delete_after_commit deletes the history frames the hex products "
                               "stage reads after the forecast; add --products none, or keep the frames")
    return argv


def plan(args, *, config: Path | None = None, document: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The hex go plan: the stage map, the region and the regional command (no files written)."""

    config = Path(config if config is not None else args.config)
    document = document if document is not None else read_document(config)
    if document is None:
        raise HexGoRefusal(f"{config} is not a readable experiment TOML")
    refuse_flags(args)
    request = read_request(document, source=str(config))
    root = work_dir(args, config)
    shape = root / "go-region.json"
    polygon = region_polygon(request["projection"], request["nx"], request["ny"], request["dx_m"])
    argv = regional_argv(request, args, config, shape=shape)
    stages = {go: [hex_stage for hex_stage in hex_stages
                   if not (getattr(args, "prepare_only", False) and hex_stage in ("forecast", "products"))
                   and not ("--no-products" in argv and hex_stage == "products")]
              for go, hex_stages in STAGE_MAP.items()}
    return {"schema": "woof.go-hex-plan/v1", "engine": "hex", "config": str(config),
            "name": request["name"], "stages": stages, "region": polygon,
            "fine_dx_m": request["dx_m"], "hours": request["hours"], "physics": request["physics"],
            # Only a [static.highres] terrain block adds the key: every other go plan is unchanged.
            **({"terrain": request["terrain"]} if request.get("terrain") else {}),
            "work_dir": str(root), "regional_argv": argv,
            "command": "woof hex " + " ".join(shlex.quote(part) for part in argv)}


def launch(args, *, observer=None) -> int:
    """Run ``woof go CONFIG`` on the hex engine through the hex regional door."""

    if observer is not None:
        raise HexGoRefusal("the run-plan event stream for a hex go run is matrix item 3.3; "
                           "run `woof go CONFIG` directly")
    value = plan(args)
    if getattr(args, "dry_run", False):
        print(json.dumps(value, indent=2))
        return 0
    root = Path(value["work_dir"])
    root.mkdir(parents=True, exist_ok=True)
    shape = root / "go-region.json"
    shape.write_text(json.dumps(value["region"], indent=2) + "\n", encoding="utf-8")
    (root / "go-hex-plan.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(f"go: engine hex, {value['name']}: {value['command']}", flush=True)
    from woof.hex.cli import main as hex_main

    started = time.monotonic()
    code = int(hex_main(list(value["regional_argv"])) or 0)
    elapsed = time.monotonic() - started
    print(f"go: engine hex finished with exit {code} in {elapsed:.1f} s; receipt "
          f"{root / 'regional-receipt.json'}", file=sys.stderr if code else sys.stdout, flush=True)
    return code


__all__ = ["HexGoRefusal", "STAGE_MAP", "background_km", "configured_engine", "launch", "plan",
           "read_request", "refuse_flags", "region_polygon"]
