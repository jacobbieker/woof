"""Assets -> forecast sites (unit 4, stub).

``woof energy sites`` samples lines and cables every ``--spacing-m`` with
the local conductor bearing, and adds substations, turbines (with hub
height), PV farms and optionally towers, writing ``woof-energy.sites.v1``.
"""

from __future__ import annotations

from typing import Sequence

from woof.energy.contracts import AssetCollection, EnergyNotImplemented, SiteSet


def build_sites(assets: AssetCollection, *, spacing_m: float = 100.0,
                kinds: Sequence[str] | None = None,
                min_voltage_kv: float | None = None,
                region: Sequence | None = None,
                heights_m: Sequence[float] = (10.0, 30.0, 100.0),
                include_towers: bool = False,
                pv_grid_m: float | None = None) -> SiteSet:
    """``region`` is a list of polygons as in :mod:`woof.energy.geometry`."""

    raise EnergyNotImplemented("woof energy sites")


def main(args) -> int:
    raise EnergyNotImplemented("woof energy sites")
