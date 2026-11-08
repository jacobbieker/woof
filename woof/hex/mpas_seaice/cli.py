"""The ``woof hex seaice`` door: the MPAS-Seaice port (stub until its dycore lands)."""

from __future__ import annotations

from woof.hex.coupled import registry
from woof.hex.mpas_model_door import add_model_parser

_ROW = {
    "word": "seaice",
    "name": "seaice",
    "module": "woof.hex.mpas_seaice",
    "attr": "registry-options.json",
}


def add_seaice_parser(commands) -> None:
    info = registry.component(_ROW["name"])
    add_model_parser(
        commands,
        word="seaice",
        help_text=f"the MPAS-Seaice port (core {info.core_dir}; status: {info.status})",
        row=_ROW,
    )
