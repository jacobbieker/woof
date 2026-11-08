"""The ``woof hex ice`` door: the MPAS-ICE (land-ice) port (stub until its dycore lands)."""

from __future__ import annotations

from woof.hex.coupled import registry
from woof.hex.mpas_model_door import add_model_parser

_ROW = {
    "word": "ice",
    "name": "landice",
    "module": "woof.hex.mpas_ice",
    "attr": "registry-options.json",
}


def add_ice_parser(commands) -> None:
    info = registry.component(_ROW["name"])
    add_model_parser(
        commands,
        word="ice",
        help_text=f"the MPAS-ICE (land-ice) port (core {info.core_dir}; status: {info.status})",
        row=_ROW,
    )
