"""Grid and renewable asset importers (unit 2, stub).

``woof energy import`` reads PyPSA-Eur network CSVs, the UK Renewable Energy
Planning Database (REPD) CSV, GeoJSON and generic CSV, and writes a
``woof-energy.assets.v1`` document, optionally merged into an existing one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from woof.energy.contracts import AssetCollection, EnergyNotImplemented

FORMATS = ("pypsa-eur", "repd", "geojson", "csv")


def import_assets(paths: Sequence[Path], *, fmt: str,
                  column_map: Mapping[str, str] | None = None,
                  kind: str | None = None) -> AssetCollection:
    """Read ``paths`` in format ``fmt``.  ``column_map`` keys: lat, lon, id."""

    raise EnergyNotImplemented("woof energy import")


def merge_collections(base: AssetCollection, extra: AssetCollection
                      ) -> AssetCollection:
    """``base`` plus the assets of ``extra`` that duplicate none of it."""

    raise EnergyNotImplemented("woof energy import --merge")


def main(args) -> int:
    raise EnergyNotImplemented("woof energy import")
