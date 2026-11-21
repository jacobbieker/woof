"""The coupled component registry.

Which MPAS components a run can couple, and which are actually ported to the
GPU. The atmosphere is the hex core and is always present. The ocean, sea-ice
and land-ice cores are named from the start so the coupling interface is stable
while they are ported, but a core that is not yet ported is a *stub*: the door
and its coupling channels exist, and asking to couple it is refused by name
rather than running something that looks right.

A component is ``ported`` only when its dynamical core and every scheme are on
the GPU and its landing gate is green; until then it is ``stub``, or
``partial`` once its standalone forward mode runs on the GPU (word for word
against its oracle) while schemes are still missing. A ``partial`` core is
refused by a coupled step as a stub is.

A component may instead be ``ported_subset``: it declares, in ``_SUBSETS``, the
exact schemes a coupled step can run. A step names the schemes it asks for;
any scheme outside the declared subset is refused by name, never skipped. A
component that declares no subset keeps the strict rule above.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from woof.hex.errors import ConfigurationRefusal

#: component -> (the MPAS-Model core directory, port status, the port's branch)
_COMPONENTS: dict[str, tuple[str, str, str]] = {
    "atmosphere": ("core_atmosphere", "ported", "hex"),
    "ocean": ("core_ocean", "stub", "ocean"),
    "seaice": ("core_seaice", "stub", "seaice"),
    "landice": ("core_landice", "ported_subset", "ice"),
}

#: component -> the schemes a ``ported_subset`` component can run in a coupled step
_SUBSETS: dict[str, tuple[str, ...]] = {
    # MALI v8.4.2: the forward mode with these velocity solvers (Albany's
    # FO/L1L2/Stokes are refused by name)
    "landice": ("sia", "simple", "none"),
}


#: opt-in coupled configurations that are not a model's own, each named plainly:
#: name -> (what it is, the module that runs it, the components it carries)
_HYBRIDS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "e3sm-v3.2.0-coupler+mpas-v8.4.2": (
        "E3SM v3.2.0 coupler with MPAS v8.4.2 components, not an E3SM configuration",
        "woof.hex.coupled.hybrid_ocean",
        ("ocean",),
    ),
}


def hybrids() -> dict[str, tuple[str, str, tuple[str, ...]]]:
    """The opt-in hybrid configurations (none is on the default path). E3SM
    v3.2.0's own MPAS-Ocean and MPAS-Seaice are not ported
    (docs/mpas-coupled-roadmap.md)."""
    return dict(_HYBRIDS)


@dataclass(frozen=True)
class ComponentInfo:
    name: str
    core_dir: str
    status: str
    branch: str
    schemes: tuple[str, ...] = ()


def all_components() -> tuple[ComponentInfo, ...]:
    return tuple(
        ComponentInfo(name, core_dir, status, branch, _SUBSETS.get(name, ()))
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
    return ComponentInfo(name, core_dir, status, branch, _SUBSETS.get(name, ()))


def require_ported(name: str, schemes: Sequence[str] | None = None) -> ComponentInfo:
    """Refuse a component a coupled step cannot run.

    ``ported`` passes (``schemes`` is not consulted). ``ported_subset`` passes
    only when ``schemes`` names the schemes the step runs and every one is in
    the component's declared subset; the first one outside it is refused by
    name. Anything else is refused as not ported.
    """
    info = component(name)
    if isinstance(schemes, str):
        schemes = (schemes,)  # one scheme, not its characters
    if info.status == "ported_subset":
        declared = ", ".join(info.schemes)
        if not schemes:
            raise ConfigurationRefusal(
                knob="coupled.schemes",
                value=name,
                reason=f"the {name} core is ported for a declared subset of schemes only",
                declaration=f"name the schemes the step runs; the declared subset is: {declared}",
            )
        for scheme in schemes:
            if scheme not in info.schemes:
                raise ConfigurationRefusal(
                    knob="coupled.schemes",
                    value=scheme,
                    reason=f"the {name} scheme {scheme!r} is not in its declared ported subset",
                    declaration=f"the {name} declared subset is: {declared}",
                )
        return info
    if info.status != "ported":
        raise ConfigurationRefusal(
            knob="coupled.component",
            value=name,
            reason=f"the {name} core is not yet ported to the GPU (status: {info.status})",
            declaration=(
                "woof hex coupled runs the ported cores only; the "
                f"{name} port's progress and remaining work are in "
                "docs/mpas-coupled-roadmap.md"
            ),
        )
    return info
