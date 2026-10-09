"""Sample MPAS (hex) history at sites and heights (unit 7, stub).

Same output contract as :func:`woof.energy.sample.sample_wrfout`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from woof.energy.contracts import EnergyNotImplemented
from woof.energy.sample import PROFILE_VARS, SURFACE_VARS, SampleResult


def available_variables(paths: Sequence[Path], *,
                        mesh_path: Path | None = None) -> set[str]:
    raise EnergyNotImplemented("woof.energy.sample_hex.available_variables")


def sample_mpas(paths: Sequence[Path], lat: np.ndarray, lon: np.ndarray,
                heights_m: Sequence[float],
                profile_vars: Sequence[str] = PROFILE_VARS,
                surface_vars: Sequence[str] = SURFACE_VARS, *,
                mesh_path: Path | None = None) -> SampleResult:
    """Sample MPAS history files (time-ordered) at the sites."""

    raise EnergyNotImplemented("woof.energy.sample_hex.sample_mpas")
