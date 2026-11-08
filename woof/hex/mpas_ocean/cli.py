"""The ``woof hex ocean`` door: the MPAS-Ocean port (stub until its dycore lands)."""

from __future__ import annotations

from woof.hex.coupled import registry
from woof.hex.mpas_model_door import add_model_parser

_ROW = {
    "word": "ocean",
    "name": "ocean",
    "module": "woof.hex.mpas_ocean",
    "attr": "registry-options.json",
}


def add_ocean_parser(commands) -> None:
    info = registry.component(_ROW["name"])
    add_model_parser(
        commands,
        word="ocean",
        help_text=f"the MPAS-Ocean port (core {info.core_dir}; status: {info.status})",
        row=_ROW,
    )
