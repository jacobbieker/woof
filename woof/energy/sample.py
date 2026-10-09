"""Sample wrfout history at sites and heights (unit 10, stub).

The heavy lifting lives in the Rust ``rw-sitesample`` library
(:mod:`woof.energy.sample_bridge`); this module is the Python face.

Output keys (both WRF and MPAS samplers use the same names):

profile, shape (time, site, height):
    ``U``, ``V``   earth-relative wind at mass points, m/s
    ``W``          vertical velocity destaggered to mass levels, m/s
    ``THETA``      potential temperature (T + 300), K
    ``PRES``       full pressure (P + PB), Pa
    ``QVAPOR``, ``QCLOUD``, ``QRAIN``, ``QICE``, ``QSNOW``, ``QGRAUP``  kg/kg
surface, shape (time, site):
    ``U10``, ``V10`` earth-relative; ``T2``, ``Q2``, ``PSFC``, ``SWDOWN``,
    ``SWDDNI``, ``SWDDIF``, ``RAINNC``, ``RAINC``, ``COSZEN`` as written.

Heights are metres above model terrain, interpolated linearly in height
between mass levels (height AGL from ``(PH + PHB) / g - HGT``, destaggered).
Horizontal interpolation is bilinear on mass points.  Sites outside the
grid's interior are NaN with ``inside == False``; nothing is extrapolated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from woof.energy.contracts import EnergyNotImplemented

PROFILE_VARS = ("U", "V", "W", "THETA", "PRES", "QVAPOR", "QCLOUD", "QRAIN",
                "QICE", "QSNOW", "QGRAUP")
SURFACE_VARS = ("U10", "V10", "T2", "Q2", "PSFC", "SWDOWN", "SWDDNI",
                "SWDDIF", "RAINNC", "RAINC", "COSZEN")


class SampleUnavailable(RuntimeError):
    """The history cannot supply a requested variable or site."""


@dataclass
class SampleResult:
    times: np.ndarray                 # datetime64[s], (T,)
    heights_m: np.ndarray             # (H,)
    profile: dict[str, np.ndarray]    # name -> (T, S, H)
    surface: dict[str, np.ndarray]    # name -> (T, S)
    inside: np.ndarray                # bool, (S,)
    terrain_m: np.ndarray             # model terrain height at sites, (S,)
    dx_m: float
    source: str = ""
    notes: list[str] = field(default_factory=list)


def available_variables(paths: Sequence[Path]) -> set[str]:
    """Output keys (see module docstring) the history files can supply."""

    raise EnergyNotImplemented("woof.energy.sample.available_variables")


def sample_wrfout(paths: Sequence[Path], lat: np.ndarray, lon: np.ndarray,
                  heights_m: Sequence[float],
                  profile_vars: Sequence[str] = PROFILE_VARS,
                  surface_vars: Sequence[str] = SURFACE_VARS
                  ) -> SampleResult:
    """Sample wrfout files (one domain, time-ordered) at the sites."""

    raise EnergyNotImplemented("woof.energy.sample.sample_wrfout")
