"""Wind farms inside the model: turbine layouts on a domain's grid.

:mod:`woof.windfarm.layout` reads a turbine table and places it on a
domain (WRF's ``windturbines.txt`` and ``windturbines-ij.txt``); ``woof
windfarm layout`` prints what it placed.  A forecast runs the turbines'
drag with ``wind_farm_scheme = "fitch_wrf"`` (WRF v4.7.1's windfarm_opt =
1, :mod:`woof.core.windfarm_coupling` behind :mod:`woof.core.wind_farm`),
which reads its ``wind_farms`` table through this package.
"""

from woof.windfarm.layout import (  # noqa: F401
    DeviceWindFarmLayout,
    Turbine,
    WindFarmLayout,
    WindFarmLayoutRefused,
    build_layout,
    domain_grid,
    layout_for_config,
    read_turbines,
)

__all__ = [
    "DeviceWindFarmLayout", "Turbine", "WindFarmLayout",
    "WindFarmLayoutRefused", "build_layout",
    "domain_grid", "layout_for_config", "read_turbines",
]
