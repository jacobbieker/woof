"""Energy products from a ``forecast.v1`` file (unit 12, stub).

Dynamic line rating (IEEE 738), conductor icing (Makkonen / ISO 12494),
wind turbine power at hub height and PV power.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from woof.energy.contracts import EnergyNotImplemented

PRODUCTS = ("dlr", "icing", "wind-power", "pv-power")


def compute_products(forecast_path: Path, *, output: Path,
                     conductor: str = "auto",
                     conductor_table: Path | None = None,
                     products: Sequence[str] = PRODUCTS) -> Path:
    raise EnergyNotImplemented("woof energy rating")


def main(args) -> int:
    raise EnergyNotImplemented("woof energy rating")
