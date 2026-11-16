"""``woof go --lakes``: the lake model and lake initial state as one opt-in.

``--lakes clm`` turns on WRF's CLM lake model (``sf_lake_physics = 1``) with
its WPS bathymetry (``use_lakedepth = 1``; the ``lake_depth`` geography is
fetched on demand) and starts the lakes from ``lake_state_source =
"auto+era5"``: GLSEA where it covers, ERA5's FLake state elsewhere.  ``--lakes
ic`` keeps the lake model off and binds only the initial state; ``--lakes off``
writes the defaults.  Every domain gets the same choice: a lake model on one
grid and not its nest would hand the nest's lakes another surface scheme.

None of this is a default, because it changes the forecast wherever the
domain has lakes.  Like ``--wind-farms``, the switch writes an immutable copy
of the configuration beside the original and launches that.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

#: Mode -> the [shared] settings it writes.
MODES = {
    "clm": {"sf_lake_physics": 1, "use_lakedepth": 1, "lake_state_source": "auto+era5"},
    "ic": {"sf_lake_physics": 0, "lake_state_source": "auto+era5"},
    "off": {"sf_lake_physics": 0, "lake_state_source": "auto"},
}

#: Per-domain keys the switch owns, removed from every [[domain]] table.
OWNED = ("sf_lake_physics", "use_lakedepth", "lakedepth_default", "lake_min_elev",
         "lake_state_source", "lake_state_path", "lake_state_accept_licence")


def add_argument(parser):
    parser.add_argument(
        "--lakes", choices=tuple(MODES), default=None,
        help="opt-in lakes: clm = WRF's CLM lake model with WPS bathymetry, lakes started from "
             "GLSEA where it covers and ERA5's FLake state elsewhere; ic = that initial state only; "
             "off = the defaults (docs/lakes.md)")


def lake_settings(raw: dict, mode: str) -> dict:
    """The configuration document with the mode's settings on every domain."""
    if mode not in MODES:
        raise ValueError(f"--lakes {mode!r} is not one of {', '.join(MODES)}")
    engine = str(raw.get("model", {}).get("engine", "wrf"))
    if engine != "wrf" and mode != "off":
        raise ValueError(
            f"--lakes {mode} configures the WRF route; this configuration runs [model] engine = "
            f"{engine!r}. On hex the CLM lake is `woof hex forecast --physics sf_lake_physics=1` with "
            f"`--surface-statics` (docs/lakes.md)")
    shared = raw.setdefault("shared", {})
    for key in ("lake_state_path", "lake_state_accept_licence"):
        shared.pop(key, None)
    shared.update(MODES[mode])
    for domain in raw.get("domain", []):
        for key in OWNED:
            domain.pop(key, None)
    return raw


def prepare_launch(args):
    """Materialize the lake-configured copy beside its source and launch that."""
    mode = getattr(args, "lakes", None)
    if mode is None:
        return
    import tomllib
    from woof.config_authority import read_config_authority
    from woof.toml_document import emit_experiment_toml
    authority = read_config_authority(args.config)
    raw = lake_settings(tomllib.loads(authority.payload.decode()), mode)
    payload = emit_experiment_toml(raw).encode()
    target = authority.source.with_name(
        f"{authority.source.stem}.lakes-{mode}-{hashlib.sha256(payload).hexdigest()[:16]}.toml")
    if not target.exists():
        target.write_bytes(payload)
    elif target.read_bytes() != payload:
        raise ValueError(f"lake configuration identity conflict at {target}")
    args.config = target
