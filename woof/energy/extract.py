"""Plan runs -> ``woof-energy.forecast.v1`` (unit 11, stub)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from woof.energy.contracts import EnergyNotImplemented


def extract_forecast(plan_path: Path, *, output: Path,
                     sites_path: Path | None = None,
                     heights_m: Sequence[float] | None = None,
                     variables: Sequence[str] | None = None,
                     fmt: str = "netcdf") -> Path:
    raise EnergyNotImplemented("woof energy extract")


def main(args) -> int:
    raise EnergyNotImplemented("woof energy extract")
