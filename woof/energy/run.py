"""Run every domain of a plan (unit 8, stub).

``woof energy run`` drives the existing routes per plan domain -- ``woof go``
for WRF roots, ``woof downscale`` for ``wrf-tiles`` children after their
parent, the ``woof hex`` route for a ``hex-swath`` mesh -- and records a run
manifest (``runs/manifest.json`` next to the plan).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from woof.energy.contracts import EnergyNotImplemented

MANIFEST_SCHEMA = "woof-energy.run-manifest.v1"


def run_plan(plan_path: Path, *, dry_run: bool = False,
             only: Sequence[str] | None = None, resume: bool = False) -> dict:
    """Run the plan's domains in parent-first order; return the manifest."""

    raise EnergyNotImplemented("woof energy run")


def main(args) -> int:
    raise EnergyNotImplemented("woof energy run")
