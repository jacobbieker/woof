"""High-resolution forecasts along power lines, substations and renewable sites.

``woof energy`` turns grid topology into forecast domains and back again::

    woof energy fetch / import  ->  assets.geojson  (woof-energy.assets.v1)
    woof energy sites           ->  sites.json      (woof-energy.sites.v1)
    woof energy plan            ->  plan/plan.json  (woof-energy.plan.v1)
    woof energy run             ->  one run per plan domain
    woof energy extract         ->  forecast.nc     (woof-energy.forecast.v1)
    woof energy rating          ->  products.nc

Every stage reads and writes one of the versioned file contracts in
:mod:`woof.energy.contracts`, so each stage can be rerun, replaced or fed
from outside without touching the others.

Three domain topologies are planned:

* ``wrf-nests``: one configuration whose sibling nests sit over corridor
  clusters (at most ``MAX_DOMAINS - 1`` children);
* ``wrf-tiles``: one regional parent run plus any number of one-way offline
  child tiles (``woof downscale``) covering the corridors -- an irregular
  WRF topology with no domain-count ceiling;
* ``hex-swath``: an MPAS variable-resolution mesh refined along the
  corridors and culled to a limited area.
"""

from __future__ import annotations

from woof.energy.contracts import (
    ASSETS_SCHEMA,
    FORECAST_SCHEMA,
    PLAN_SCHEMA,
    SITES_SCHEMA,
    ContractError,
    EnergyNotImplemented,
)

__all__ = [
    "ASSETS_SCHEMA",
    "FORECAST_SCHEMA",
    "PLAN_SCHEMA",
    "SITES_SCHEMA",
    "ContractError",
    "EnergyNotImplemented",
]
