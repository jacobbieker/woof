# WOOF-OPTIN: wrf-hydro
"""WRF-Hydro routing for WOOF, an explicit opt-in (see ../LICENSE, ../NOTICE.md).

Loaded only through :func:`woof.optin.load` with ``WOOF_ENABLE_WRF_HYDRO=1``.
"""

from .subsurface import SubsurfaceRouting, SubsurfaceState  # noqa: F401
from . import grid_transform  # noqa: F401

SOURCE = "WRF-Hydro as shipped in WRF 4.6.1 hydro/ (WRF d66e442fccc04111067e29274c9f9eaccc3cef28)"
