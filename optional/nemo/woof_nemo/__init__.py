# WOOF-OPTIN: nemo
"""NEMO kernels for WOOF, an explicit opt-in (see ../LICENSE, ../NOTICE.md).

Loaded only through :func:`woof.optin.load` with ``WOOF_ENABLE_NEMO=1``.
"""

from .traqsr import TraQsr2BD  # noqa: F401
from .eos import EquationOfState  # noqa: F401
from .tke import TkeMixing  # noqa: F401
from .trazdf import TracerVerticalDiffusion  # noqa: F401

SOURCE = "NEMO 5.0.2 (forge.nemo-ocean.eu tag 5.0.2, dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796)"
