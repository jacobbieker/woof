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
    #: the word the channel must carry ("int32" for a mask); None: whatever real the run uses
    dtype: str | None = None
    #: a diagnostic: sent only when a run asks for it, never part of the exchange's conservation
    diagnostic: bool = False


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
    # the land-ice/ocean seam, as agreed with the MALI port (woof/hex/mpas_ice/ocean_seam_v842.py):
    # the ocean reads the shelf's overburden, fraction and mask; runoff comes split by phase,
    # positive into the ocean; the ocean returns the floating ice's basal mass balance in
    # MALI's sign (negative is melt: the ocean's -landIceFreshwaterFlux)
    FluxChannel("landIcePressure", "landice", "ocean", "floating ice overburden", "Pa"),
    FluxChannel("landIceFraction", "landice", "ocean", "floating ice fraction", "1"),
    FluxChannel("landIceMask", "landice", "ocean", "floating ice mask", "1", dtype="int32"),
    FluxChannel("li2ocn_liquid_runoff", "landice", "ocean", "liquid runoff (melt and discharge)", "kg/m^2/s"),
    FluxChannel("li2ocn_solid_runoff", "landice", "ocean", "solid runoff (calving)", "kg/m^2/s"),
    FluxChannel("landIceDraft", "landice", "ocean", "floating ice draft", "m", diagnostic=True),
    FluxChannel("ocn2li_floating_basal_mass_bal", "ocean", "landice", "floating basal mass balance", "kg/m^2/s"),
    FluxChannel("li2atm_mass", "landice", "atmosphere", "melt mass flux", "kg/m^2/s"),
)

#: Channels named but not exchanged: the name, and why (with what replaces it).
REFUSED: dict[str, str] = {
    "li2ocn_momentum_x": "MPAS v8.4.2's land ice exports no calving momentum; the ocean has no import for it",
    "li2ocn_momentum_y": "MPAS v8.4.2's land ice exports no calving momentum; the ocean has no import for it",
    "li2ocn_mass": "the land ice's mass flux is split by phase: li2ocn_liquid_runoff and li2ocn_solid_runoff",
}


def channels_for(present: set[str], *, diagnostics: bool = False) -> tuple[FluxChannel, ...]:
    """The channels whose source and target are both in the present set
    (with the diagnostic channels only when asked for)."""

    return tuple(c for c in CHANNELS
                 if c.source in present and c.target in present and (diagnostics or not c.diagnostic))


def channel(name: str) -> FluxChannel:
    """The channel of that name; a refused or unknown name is refused by name."""

    for c in CHANNELS:
        if c.name == name:
            return c
    if name in REFUSED:
        raise ConfigurationRefusal(knob="coupled.channel", value=name, reason=REFUSED[name],
                                   declaration="the exchange carries " + ", ".join(c.name for c in CHANNELS))
    raise ConfigurationRefusal(knob="coupled.channel", value=name, reason="no such channel",
                               declaration="the exchange carries " + ", ".join(c.name for c in CHANNELS))


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
    if channel.dtype is not None and sender.dtype != np.dtype(channel.dtype):
        raise ConfigurationRefusal(
            knob="coupled.channel",
            value=channel.name,
            reason=f"dtype {sender.dtype} is not the channel's {channel.dtype}",
            declaration="a channel carries its declared word",
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
