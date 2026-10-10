"""PBL admission: ``woof hex forecast --pbl {ysu,off}``.

The default is YSU, the proven configuration, and a run that does not pass
``--pbl`` is byte-identical to one built before this module existed: the
config, the sealed constructor mapping, the driver argv and every receipt
are unchanged.

``--pbl off`` mirrors WRF ``bl_pbl_physics = 0``.  The engine seam's PBL
slot is switched off (``woof.core.mpas_column_batch`` ``pbl_scheme="off"``):
YSU is never called and its held ``du``/``dv``/``dtheta``/``dq*`` rates stay
zero, while the revised-MO surface layer and Noah-MP still run on the
surface/PBL cadence and publish their fluxes.  The external YSU gravity-wave
drag still runs; it adds its drag to a zero PBL momentum tendency.

WHAT PBL-OFF IS FOR.  An LES-class mesh resolves the energy-containing
boundary-layer eddies, and a 1-D PBL scheme on such a mesh double-counts
them.  With the PBL off, surface heat, moisture and momentum reach the
atmosphere only through a turbulence closure (``--les-model``, MPAS
``config_les_model`` with ``config_les_surface``).  Without one they reach
it by NO route: the fluxes are computed and published and the column never
feels them.  So the switch is guarded:

* it is admitted when ``--les-model`` selects a closure, or
* when the mesh's finest spacing is below :data:`GRAY_ZONE_BELOW_M` (the
  boundary-layer gray zone ends roughly where dx falls under the boundary
  layer depth), with a warning that no closure carries the surface fluxes
  unless ``--les-model`` is also given, and otherwise
* it is REFUSED by name, unless ``--allow-pbl-off-gray-zone`` overrides the
  refusal; an overridden run prints and records a gray-zone warning.

Grell-Freitas indexes the column at KPBL, which only a PBL scheme writes,
so ``--pbl off`` with a GF selection is refused whatever the spacing (on a
sub-3 km mesh the convection ruling already selects no cumulus scheme).

EVIDENCE.  Every timestep anchor in :mod:`woof.hex.dt_admission` was
measured with YSU.  A PBL-off run is a configuration none of them measured,
and its decision says so: ``"anchor_evidence": "unanchored-configuration"``.
This module never adds, edits or borrows an anchor.

This module is stdlib-only, like :mod:`woof.hex.convection_admission`, so the
forecast door can decide preflight on a box with no CUDA lane.
"""

from __future__ import annotations

import math
from typing import Any


class PblAdmissionError(RuntimeError):
    """A PBL selection is refused, by name."""


#: What ``--pbl`` accepts.  ``ysu`` is the default and the proven lane.
REQUESTS: tuple[str, ...] = ("ysu", "off")
DEFAULT_REQUEST = "ysu"

#: ``--pbl`` -> ``config_pbl_scheme`` (the MPAS Registry vocabulary).
CONFIG_SCHEMES: dict[str, str] = {"ysu": "bl_ysu", "off": "off"}

#: ``config_pbl_scheme`` values the frozen column-physics lane admits.
ADMITTED_PBL_SCHEMES: tuple[str, ...] = tuple(CONFIG_SCHEMES.values())

#: ``--pbl`` -> the engine seam's ``pbl_scheme`` and WRF ``bl_pbl_physics``.
ENGINE_SCHEMES: dict[str, str] = {"ysu": "ysu", "off": "off"}
BL_PBL_PHYSICS: dict[str, int] = {"ysu": 1, "off": 0}

#: The engine counter that says the surface/PBL stack ran a step: the PBL
#: slot's own with YSU, the surface layer's with the slot off (the stack
#: still runs on its cadence; it has no PBL member).  Mirrors the seam's
#: ``_surface_pbl_counter``.
SURFACE_PBL_COUNTER: dict[str, str] = {"ysu": "ysu", "off": "sfclay"}

#: Below this finest spacing a PBL-off run is admitted without a closure.
#: Roughly where dx falls under a convective boundary layer's depth and the
#: largest eddies start to be resolved; above it a 1-D PBL scheme is still
#: the only vertical route for the surface fluxes.
GRAY_ZONE_BELOW_M: float = 1_000.0

#: ``--les-model`` values that select no closure.
LES_OFF_VALUES: tuple[str, ...] = ("off", "none")

#: The PBL slot is part of the CONFIGURATION a dt anchor certifies.
ANCHOR_NOTE = (
    "every dt anchor in woof.hex.dt_admission was measured with bl_ysu; a "
    "PBL-off run is a configuration none of them measured"
)


def config_scheme(request: str) -> str:
    """``--pbl`` value -> ``config_pbl_scheme``."""

    if request not in CONFIG_SCHEMES:
        raise PblAdmissionError(
            f"--pbl {request!r} is not one of {list(REQUESTS)}"
        )
    return CONFIG_SCHEMES[request]


def request_for_config(scheme: str) -> str:
    """``config_pbl_scheme`` -> ``--pbl`` value."""

    for request, config in CONFIG_SCHEMES.items():
        if config == scheme:
            return request
    raise PblAdmissionError(
        f"config_pbl_scheme={scheme!r} is not one the frozen column-physics "
        f"lane admits ({', '.join(ADMITTED_PBL_SCHEMES)})"
    )


def les_closure_selected(les_model: str | None) -> bool:
    return les_model is not None and str(les_model).strip().lower() not in LES_OFF_VALUES


def pbl_decision(
    *,
    requested: str = DEFAULT_REQUEST,
    finest_spacing_m: float | None,
    les_model: str | None = "off",
    convection_scheme: str | None = None,
    allow_gray_zone: bool = False,
) -> dict[str, Any]:
    """Decide the run's PBL slot, and record why.

    ``finest_spacing_m`` is the smaller of the row's nominal spacing and its
    measured ``min(dcEdge)`` (the door has only the first; the bind has
    both).  ``convection_scheme`` is the config-level cumulus selection
    already decided (``cu_grell_freitas`` or ``off``), or ``None`` when it
    is not yet known.  Returns a JSON-ready mapping for the receipt, or
    raises :class:`PblAdmissionError` naming what would admit the run.
    """

    if requested not in REQUESTS:
        raise PblAdmissionError(
            f"--pbl {requested!r} is not one of {list(REQUESTS)}: 'ysu' is the "
            "proven configuration, 'off' switches the PBL scheme off (WRF "
            "bl_pbl_physics=0) for a run that resolves turbulence"
        )
    base = {
        "schema": "woof-hex.pbl-decision/v1",
        "requested": requested,
        "scheme": requested,
        "config_pbl_scheme": CONFIG_SCHEMES[requested],
        "engine_pbl_scheme": ENGINE_SCHEMES[requested],
        "bl_pbl_physics": BL_PBL_PHYSICS[requested],
        "les_model": None if les_model is None else str(les_model),
        "finest_spacing_m": (
            None if finest_spacing_m is None else float(finest_spacing_m)
        ),
        "gray_zone_below_m": GRAY_ZONE_BELOW_M,
    }
    if requested == "ysu":
        return {
            **base,
            "source": "default",
            "anchor_evidence": "the proven configuration",
            "warnings": [],
            "note": "YSU, the proven configuration; nothing changes",
        }

    if convection_scheme == "cu_grell_freitas":
        raise PblAdmissionError(
            "--pbl off with Grell-Freitas convection: GF indexes the column "
            "at KPBL, which only a PBL scheme writes, so the pair reads an "
            "uninitialised level.  Pass --convection off (the default 'auto' "
            "already selects no cumulus scheme below 3 km)"
        )
    if finest_spacing_m is not None:
        spacing = float(finest_spacing_m)
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise PblAdmissionError(
                f"finest_spacing_m={finest_spacing_m!r} is not a grid spacing"
            )
    closure = les_closure_selected(les_model)
    fine = finest_spacing_m is not None and float(finest_spacing_m) < GRAY_ZONE_BELOW_M
    warnings: list[str] = []
    if closure:
        source = "les-closure"
        note = (
            f"the PBL is off and the {les_model} closure carries the "
            "boundary-layer turbulence and the surface fluxes"
        )
    elif fine:
        source = "resolution"
        note = (
            f"the finest spacing {float(finest_spacing_m):g} m is below the "
            f"{GRAY_ZONE_BELOW_M:g} m gray-zone limit, so the PBL may be off"
        )
        warnings.append(
            "NO TURBULENCE CLOSURE: the PBL is off and --les-model is off, so "
            "the surface heat, moisture and momentum fluxes are computed and "
            "published but reach the atmosphere by no route.  Select "
            "--les-model to carry them"
        )
        warnings.append(
            "VARIABLE RESOLUTION: the decision is taken on the mesh's FINEST "
            "spacing and the PBL is off in every column; cells coarser than "
            f"{GRAY_ZONE_BELOW_M:g} m (a graded mesh's outer rings) have "
            "neither a PBL scheme nor resolved eddies"
        )
    elif allow_gray_zone:
        source = "override"
        spacing_text = (
            "unknown" if finest_spacing_m is None
            else f"{float(finest_spacing_m):g} m"
        )
        note = (
            f"--allow-pbl-off-gray-zone overrode the refusal at finest "
            f"spacing {spacing_text}"
        )
        warnings.append(
            f"GRAY ZONE: the PBL is off at finest spacing {spacing_text}, at "
            f"or above the {GRAY_ZONE_BELOW_M:g} m limit, with no closure "
            "selected.  The boundary layer is neither parameterised nor "
            "resolved, and the surface fluxes reach the atmosphere by no "
            "route.  This run is an experiment arm, never a forecast"
        )
    else:
        spacing_text = (
            "an unknown spacing" if finest_spacing_m is None
            else f"{float(finest_spacing_m):g} m"
        )
        raise PblAdmissionError(
            f"--pbl off at {spacing_text}: at or above "
            f"{GRAY_ZONE_BELOW_M:g} m the boundary-layer eddies are not "
            "resolved, and with the PBL scheme off and no --les-model closure "
            "the surface fluxes reach the atmosphere by no route.  Select a "
            "closure with --les-model, use a mesh finer than "
            f"{GRAY_ZONE_BELOW_M:g} m, or pass --allow-pbl-off-gray-zone for "
            "an experiment arm that records itself as one"
        )
    return {
        **base,
        "source": source,
        "anchor_evidence": "unanchored-configuration",
        "anchor_note": ANCHOR_NOTE,
        "warnings": warnings,
        "note": note,
    }


__all__ = [
    "ADMITTED_PBL_SCHEMES",
    "ANCHOR_NOTE",
    "BL_PBL_PHYSICS",
    "CONFIG_SCHEMES",
    "DEFAULT_REQUEST",
    "ENGINE_SCHEMES",
    "GRAY_ZONE_BELOW_M",
    "LES_OFF_VALUES",
    "PblAdmissionError",
    "REQUESTS",
    "SURFACE_PBL_COUNTER",
    "config_scheme",
    "les_closure_selected",
    "pbl_decision",
    "request_for_config",
]
