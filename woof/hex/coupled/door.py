"""The ``woof hex coupled`` door: the coupled MPAS run.

This landing is the coupling spine's front end. It reports the coupling state
(which cores are ported, which channels are active), plans a coupled clock
(validating the cadence against every component's timestep and the run length
against the cadence), and refuses by name any core that is not yet ported. The
run loop that steps the ported cores -- and the native A-O-Si bit-exact
reference that gates it -- lands with the first ported core, the sea-ice lane.
"""

from __future__ import annotations

import argparse
import json

from woof.hex.errors import ConfigurationRefusal

from . import exchange, registry
from .clock import CoupledClock

_DEFAULT_DT = 900.0


def add_coupled_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "coupled",
        help="the coupled MPAS run: the hex atmosphere with ocean, sea-ice and land-ice",
    )
    parser.add_argument(
        "--component",
        action="append",
        default=[],
        metavar="NAME",
        help="name a core to couple (repeatable); a core that is not yet ported is refused by name",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="plan the coupled clock (validate the cadence and timesteps) instead of only reporting the state",
    )
    parser.add_argument("--start", type=float, default=0.0, help="start time, seconds")
    parser.add_argument("--end", type=float, default=0.0, help="end time, seconds")
    parser.add_argument("--cadence", type=float, default=0.0, help="flux-exchange cadence, seconds")
    parser.add_argument(
        "--dt",
        action="append",
        default=[],
        metavar="NAME=SECONDS",
        help="a component timestep (repeatable); a present component with no --dt takes the default",
    )
    parser.set_defaults(handler=_coupled)


def _parse_dts(pairs: list[str]) -> dict[str, float]:
    dts: dict[str, float] = {}
    for pair in pairs:
        name, sep, text = pair.partition("=")
        if not sep or not name:
            raise ConfigurationRefusal(
                knob="coupled.dt",
                value=pair,
                reason="not NAME=SECONDS",
                declaration="each --dt is NAME=SECONDS",
            )
        try:
            dts[name] = float(text)
        except ValueError:
            raise ConfigurationRefusal(
                knob="coupled.dt",
                value=pair,
                reason=f"{text!r} is not a number of seconds",
                declaration="each --dt is NAME=SECONDS",
            ) from None
    return dts


def _coupled(arguments: argparse.Namespace) -> int:
    if arguments.component:
        present: set[str] = set()
        for name in arguments.component:
            registry.require_ported(name)
            present.add(name)
    else:
        present = {c.name for c in registry.all_components() if c.status == "ported"}

    state: dict[str, object] = {
        "components": [
            {"name": c.name, "core": c.core_dir, "status": c.status, "branch": c.branch}
            for c in registry.all_components()
        ],
        "present": sorted(present),
        "active_channels": [c.name for c in exchange.channels_for(present)],
    }

    if arguments.plan:
        if not arguments.cadence:
            raise ConfigurationRefusal(
                knob="coupled.cadence",
                value=arguments.cadence,
                reason="a planned clock needs a cadence",
                declaration="pass --cadence SECONDS with --plan",
            )
        dts = _parse_dts(arguments.dt)
        for name in present:
            dts.setdefault(name, _DEFAULT_DT)
        for name in dts:
            registry.require_ported(name)
        clock = CoupledClock(
            start=arguments.start,
            end=arguments.end,
            cadence=arguments.cadence,
            dts=dts,
        )
        state["clock"] = clock.receipt()

    print(json.dumps(state, indent=2))
    return 0
