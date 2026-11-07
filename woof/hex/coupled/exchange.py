"""The matched flux exchange between coupled components.

A channel is a named quantity one component sends to another, with its units.
The exchange is *matched*: at every exchange the sender's field and the
receiver's field are compared byte for byte, so a channel that silently drops,
reorders or rescales a flux is a refusal, not a drift. That is the conservation
gate for a coupled run -- what one core sends is, to the bit, what the other
receives.

The channel table is the native MPAS coupled exchange surface (atmosphere,
ocean, sea-ice and land-ice). A channel is active only when both of its
components are ported; the active set for a run is computed from the present
components, and a run that names an unported channel is refused by name.
"""

from __future__ import annotations

from dataclasses import dataclass

from woof.hex.errors import ConfigurationRefusal


@dataclass(frozen=True)
class FluxChannel:
    name: str
    source: str
    target: str
    quantity: str
    units: str


#: The native MPAS coupled exchange surface. Source and target are component
#: names from :mod:`woof.hex.coupled.registry`.
CHANNELS: tuple[FluxChannel, ...] = (
    FluxChannel("atm2ocn_momentum_x", "atmosphere", "ocean", "surface stress, x", "Pa"),
    FluxChannel("atm2ocn_momentum_y", "atmosphere", "ocean", "surface stress, y", "Pa"),
    FluxChannel("atm2ocn_heat", "atmosphere", "ocean", "surface heat flux", "W/m^2"),
    FluxChannel("atm2ocn_freshwater", "atmosphere", "ocean", "precipitation minus evaporation", "m/s"),
    FluxChannel("atm2ocn_sw", "atmosphere", "ocean", "absorbed shortwave", "W/m^2"),
    FluxChannel("ocn2atm_sst", "ocean", "atmosphere", "sea surface temperature", "K"),
    FluxChannel("ocn2atm_sss", "ocean", "atmosphere", "sea surface salinity", "PSU"),
    FluxChannel("atm2si_momentum_x", "atmosphere", "seaice", "surface stress, x", "Pa"),
    FluxChannel("atm2si_momentum_y", "atmosphere", "seaice", "surface stress, y", "Pa"),
    FluxChannel("atm2si_heat", "atmosphere", "seaice", "surface heat flux", "W/m^2"),
    FluxChannel("atm2si_freshwater", "atmosphere", "seaice", "precipitation minus evaporation", "m/s"),
    FluxChannel("si2atm_sst", "seaice", "atmosphere", "ice surface / ocean temperature", "K"),
    FluxChannel("si2atm_cover", "seaice", "atmosphere", "sea-ice fraction", "1"),
    FluxChannel("ocn2si_base_heat", "ocean", "seaice", "basal heat flux", "W/m^2"),
    FluxChannel("ocn2si_melt", "ocean", "seaice", "melt and freeze freshwater", "m/s"),
    FluxChannel("si2ocn_base_heat", "seaice", "ocean", "basal heat flux", "W/m^2"),
    FluxChannel("si2ocn_melt", "seaice", "ocean", "melt and freeze freshwater", "m/s"),
    FluxChannel("li2ocn_mass", "landice", "ocean", "melt and runoff mass flux", "kg/m^2/s"),
    FluxChannel("li2ocn_momentum_x", "landice", "ocean", "calving momentum, x", "Pa"),
    FluxChannel("li2ocn_momentum_y", "landice", "ocean", "calving momentum, y", "Pa"),
    FluxChannel("li2atm_mass", "landice", "atmosphere", "melt mass flux", "kg/m^2/s"),
)


def channels_for(present: set[str]) -> tuple[FluxChannel, ...]:
    """The channels whose source and target are both in the present set."""

    return tuple(c for c in CHANNELS if c.source in present and c.target in present)


def check_matched(channel: FluxChannel, sent, received) -> None:
    """Byte-for-byte equality of a channel's sent and received field.

    A mismatch is a refusal: a coupled run that does not conserve a channel to
    the bit is not a run, and naming the channel is the diagnosis.
    """

    import numpy as np

    sender = np.ascontiguousarray(sent)
    receiver = np.ascontiguousarray(received)
    if sender.shape != receiver.shape:
        raise ConfigurationRefusal(
            knob="coupled.channel",
            value=channel.name,
            reason=f"shape {sender.shape} does not match {receiver.shape}",
            declaration="a matched exchange carries the same shape both ways",
        )
    if sender.dtype != receiver.dtype:
        raise ConfigurationRefusal(
            knob="coupled.channel",
            value=channel.name,
            reason=f"dtype {sender.dtype} does not match {receiver.dtype}",
            declaration="a matched exchange carries the same dtype both ways",
        )
    # count differing words, not bytes: one word can differ in several bytes
    width = max(sender.dtype.itemsize, 1)
    diff = int(np.count_nonzero(np.any(
        sender.view(np.uint8).reshape(sender.size, width) != receiver.view(np.uint8).reshape(receiver.size, width),
        axis=1)))
    if diff:
        raise ConfigurationRefusal(
            knob="coupled.channel",
            value=channel.name,
            reason=(
                f"{diff} of {sender.size} words differ between what {channel.source} sent "
                f"and what {channel.target} received"
            ),
            declaration="a coupled exchange is matched to the bit",
        )
