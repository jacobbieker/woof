"""The coupled component registry.

Which MPAS components a run can couple, and which are actually ported to the
GPU. The atmosphere is the hex core and is always present. The ocean, sea-ice
and land-ice cores are named from the start so the coupling interface is stable
while they are ported, but a core that is not yet ported is a *stub*: the door
and its coupling channels exist, and asking to couple it is refused by name
rather than running something that looks right.

A component is ``ported`` only when its dynamical core and every scheme are on
the GPU and its landing gate is green; until then it is ``stub``.
"""

from __future__ import annotations

from dataclasses import dataclass

from woof.hex.errors import ConfigurationRefusal

#: component -> (the MPAS-Model core directory, port status, the port's branch)
_COMPONENTS: dict[str, tuple[str, str, str]] = {
    "atmosphere": ("core_atmosphere", "ported", "hex"),
    "ocean": ("core_ocean", "stub", "ocean"),
    "seaice": ("core_seaice", "stub", "seaice"),
    "landice": ("core_landice", "stub", "ice"),
}


@dataclass(frozen=True)
class ComponentInfo:
    name: str
    core_dir: str
    status: str
    branch: str


def all_components() -> tuple[ComponentInfo, ...]:
    return tuple(
        ComponentInfo(name, core_dir, status, branch)
        for name, (core_dir, status, branch) in _COMPONENTS.items()
    )


def component(name: str) -> ComponentInfo:
    try:
        core_dir, status, branch = _COMPONENTS[name]
    except KeyError:
        raise ConfigurationRefusal(
            knob="coupled.component",
            value=name,
            reason="not a coupled component",
            declaration="the coupled components are: " + ", ".join(_COMPONENTS),
        ) from None
    return ComponentInfo(name, core_dir, status, branch)


def require_ported(name: str) -> ComponentInfo:
    info = component(name)
    if info.status != "ported":
        raise ConfigurationRefusal(
            knob="coupled.component",
            value=name,
            reason=f"the {name} core is not yet ported to the GPU (status: {info.status})",
            declaration=(
                "woof hex coupled runs the ported cores only; the "
                f"{name} port lands on feat/mpas-{info.branch}"
            ),
        )
    return info
