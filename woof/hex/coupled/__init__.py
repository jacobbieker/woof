"""The coupling spine for a coupled MPAS run on the hex engine.

The hex core is MPAS-Atmosphere. This package is what the other MPAS cores
-- ocean, sea-ice and land-ice -- couple to: the shared clock that steps every
component on one wall clock, the matched flux exchange between components, and
the registry that names which cores are ported and which are still a refusal.

It is backend-agnostic on purpose: the clock, the exchange and the registry are
host-side, and the component kernels they schedule run through the same CUDA or
MLX (Metal) facade the hex core uses, so a coupled run is the same code on both
backends. Nothing here advances a component that is not registered, and a core
that is named but not yet ported is refused by name.
"""

from .clock import CoupledClock, CoupledInterval
from .exchange import CHANNELS, FluxChannel, channels_for, check_matched
from .registry import ComponentInfo, all_components, component, require_ported

__all__ = [
    "CHANNELS",
    "ComponentInfo",
    "CoupledClock",
    "CoupledInterval",
    "FluxChannel",
    "all_components",
    "channels_for",
    "check_matched",
    "component",
    "require_ported",
]
