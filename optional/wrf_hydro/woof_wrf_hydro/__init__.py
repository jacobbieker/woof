# WOOF-OPTIN: wrf-hydro
"""WRF-Hydro routing for WOOF, an explicit opt-in (see ../LICENSE, ../NOTICE.md).

Loaded only through :func:`woof.optin.load` with ``WOOF_ENABLE_WRF_HYDRO=1``.
"""

from .subsurface import SubsurfaceRouting, SubsurfaceState  # noqa: F401
from . import grid_transform  # noqa: F401
from .overland import OverlandRouting, OverlandState  # noqa: F401
from .channel import ReachRouting  # noqa: F401
from .diffusive import DiffusiveWaveRouting  # noqa: F401
from .levelpool import LevelPool  # noqa: F401
from .baseflow import GroundwaterBucket  # noqa: F401
from .gw2d import Gw2d  # noqa: F401
from . import namelist  # noqa: F401

SOURCE = "WRF-Hydro as shipped in WRF 4.7.1 hydro/ (WRF f52c197ed39d12e087d02c50f412d90d418f6186)"
