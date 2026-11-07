"""The ``woof hex ocean`` door: the MPAS-Ocean port (a stub until every piece is ported)."""

from __future__ import annotations

from woof.hex.coupled import registry
from woof.hex.mpas_model_door import add_model_parser

_ROW = {
    "word": "ocean",
    "name": "ocean",
    "module": "woof.hex.mpas_ocean",
    "attr": "registry-options.json",
    "verify": "woof.hex.mpas_ocean.verify:verify",
}


def add_ocean_parser(commands) -> None:
    info = registry.component(_ROW["name"])
    add_model_parser(
        commands,
        word="ocean",
        help_text=f"the MPAS-Ocean port (core {info.core_dir}; status: {info.status})",
        row=_ROW,
    )
