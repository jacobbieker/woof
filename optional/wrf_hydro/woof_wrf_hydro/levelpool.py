# WOOF-OPTIN: wrf-hydro
"""WRF-Hydro's level-pool reservoirs on device (``levelpool.cu``).

``LEVELPOOL_PHYSICS`` through ``run_levelpool_reservoir``
(hydro/Routing/Reservoirs/Level_Pool/module_levelpool.F90 in WRF 4.7.1's
hydro/).  ``lake_opt`` is hydro.namelist's lake_option as levelpool_init
receives it: 1 is a third-order Runge-Kutta step of the pool elevation under
the period's inflow ramp, with outflow through an orifice, over a weir and
over the dam; 2 is the pass-through (outflow = inflow).  lake_option 0
removes the lakes (module_RT: NLAKES = 0) and 3 is resolved by read_rt_nlst
before the lakes are built, so neither reaches a reservoir.  Each lake's properties are levelpool_init's (LAKEPARM: LkArea km2,
WeirE, WeirC, WeirL, DamL, OrificeE, OrificeC, OrificeA, LkMxE); the state
is the water elevation RESHT, the inflow QLAKEI and the outflow QLAKEO.
lake_option 3's persistence and RFC-forecast reservoirs are
:mod:`woof_wrf_hydro.reservoir_da`.
"""

from __future__ import annotations

import numpy as np

_PROPS = ("lake_area", "weir_elevation", "weir_coefficient", "weir_length", "dam_length",
          "orifice_elevation", "orifice_coefficient", "orifice_area", "max_depth")


class LevelPool:
    """``n`` level-pool lakes: device float32 properties and state."""

    def __init__(self, *, water_elevation, qlakei=None, qlakeo=None, lake_opt=1, **props):
        import cupy as cp

        missing = [k for k in _PROPS if k not in props]
        if missing or set(props) - set(_PROPS):
            raise ValueError(f"level-pool properties are {_PROPS}")
        water_elevation = np.asarray(water_elevation, np.float32).reshape(-1)
        self.n = len(water_elevation)
        table = np.stack([np.asarray(props[k], np.float32).reshape(-1) for k in _PROPS], axis=1)
        if table.shape != (self.n, 9):
            raise ValueError("one value per lake for every property")
        self.props = cp.asarray(np.ascontiguousarray(table if self.n else np.zeros((1, 9),
                                                                                   np.float32)))
        z = np.zeros(max(self.n, 1), np.float32)
        self.resht = cp.asarray(water_elevation if self.n else z)
        self.qlakei = cp.asarray(np.asarray(qlakei, np.float32) if qlakei is not None else z)
        self.qlakeo = cp.asarray(np.asarray(qlakeo, np.float32) if qlakeo is not None else z)
        opt = np.broadcast_to(np.asarray(lake_opt, np.int32), (self.n,)).copy()
        if not np.isin(opt, (1, 2)).all():
            raise ValueError("lake_opt is 1 (level pool) or 2 (pass-through); LEVELPOOL_PHYSICS "
                             "stops on anything else")
        self.lake_opt = cp.asarray(opt if self.n else np.ones(1, np.int32))

    @classmethod
    def empty(cls):
        return cls(water_elevation=np.zeros(0), **{k: np.zeros(0) for k in _PROPS})

    def run(self, *, previous_inflow, inflow, lateral_inflow, routing_period) -> None:
        """run_levelpool_reservoir on every lake: updates ``resht`` and ``qlakeo``."""
        import cupy as cp

        from .diffusive import _k

        _k("wrfhydro_levelpool")((((self.n + 127) // 128),), (128,), (
            cp.asarray(previous_inflow, cp.float32), cp.asarray(inflow, cp.float32),
            cp.asarray(lateral_inflow, cp.float32), self.resht, self.qlakeo, self.props,
            self.lake_opt, np.float32(routing_period), np.int32(self.n)))
