"""OpenStreetMap power infrastructure via the Overpass API (unit 1, stub).

``woof energy fetch`` asks Overpass for ``power=line|minor_line|cable|
substation|plant|generator|tower`` inside a bbox or polygon and writes a
``woof-energy.assets.v1`` document.  OSM data is (c) OpenStreetMap
contributors, ODbL 1.0; the attribution travels in the document's
``sources`` records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from woof.energy.contracts import AssetCollection, EnergyNotImplemented

SOURCE = "osm"
ATTRIBUTION = "(c) OpenStreetMap contributors, ODbL 1.0"
LICENSE = "ODbL-1.0"


def fetch_assets(*, bbox: tuple[float, float, float, float] | None = None,
                 polygon: Path | None = None,
                 kinds: Sequence[str] = ("line", "cable", "substation",
                                         "plant", "generator"),
                 min_voltage_kv: float | None = None,
                 endpoint: str | None = None, refresh: bool = False,
                 offline: bool = False, timeout_s: float = 180.0
                 ) -> AssetCollection:
    """Fetch OSM power assets.  ``bbox`` is ``(west, south, east, north)``."""

    raise EnergyNotImplemented("woof energy fetch (OSM/Overpass)")


def main(args) -> int:
    raise EnergyNotImplemented("woof energy fetch (OSM/Overpass)")
