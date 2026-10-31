"""Height-dependent horizontal resolution on MPAS-A: the partial-depth band.

plans/sorted-mapping-hummingbird.md, Phase D5 -- the hex counterpart of
:mod:`woof.core.partial_nest` (WRF, D0-D4).  A band is a finer REGIONAL
MPAS mesh (any limited-area mesh with MPAS-Limited-Area's ``bdyMask`` rings:
``rw_mpas_mesh --spec ... --cull-parent``, a density raster from
:mod:`woof.hex.density` for a variable-resolution band) that carries the
atmosphere from the ground up to a FLAT height ``z_i``; above it the parent
alone does.  The interface is a ``zgrid`` face where every column of both
meshes stands at one height (the terrain influence of MPAS's smoothed
coordinate has decayed there), so the child is the parent's own discrete
column cut at that face:

* the parent's computational levels below the interface -- ``zw``, ``rdzw``,
  ``fzm``/``fzp``, the off-centring profile, the reference wind profiles --
  word for word (the child's coordinate is built full depth on its own mesh
  with the parent's levels and model top, then truncated, so its terrain
  following is the parent's);
* the interface face ``k_i`` (0-based; the child's ``nVertLevels``) is the
  child's top, with the parent's ``fzm(k_i)``/``fzp(k_i)`` there.

This module holds the band configuration, the refusals by name, the
truncation of a full-depth child onto its interface, and the WOOF-only
namelist group ``&woof_partial_nest``.  The coupled run is
:mod:`woof.hex.partial_nest_run`; its kernels are
:mod:`woof.hex.cuda_partial_nest_v841`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields, replace
from typing import Any, Mapping

import numpy as np


class PartialNestRefused(ValueError):
    """A partial-depth band the hex formulation cannot carry, named."""


#: What a band may restrict onto its parent below the interface: theta
#: (bringing every scalar the child carries).  The vertical coordinate and w
#: are refused: D4 measured a delayed-feedback instability for WRF's phi/w
#: (woof.core.partial_nest_run, "w and phi feedback"), and MPAS's rho_zz is its
#: coordinate's mass.
FEEDBACK_FIELDS = ("theta",)
_REFUSED_FEEDBACK = {
    "w": "w is not fed back below a partial interface: the parent's w-rho "
         "acoustic solve answered a parent step late by a restricted column is "
         "the delayed-feedback oscillator D4 measured unstable (phi/w in WRF)",
    "rho": "rho_zz is the MPAS coordinate's mass; restricting it moves the "
           "parent's EOS pressure under its own theta (D4 measured mu "
           "feedback worsening the gravity wave)",
    "rho_zz": "see rho",
    "zgrid": "the vertical coordinate is never fed back",
    "u": "u is not fed back: D4 measured restricted short waves hurting the "
         "parent (theta' block RMSE 0.011 against 0.0075 alone)",
}

#: Face schemes for the interface's vertical fluxes.  "second_order" is the
#: parent's own second-order face value (the D4 lesson: an upwinded face
#: ratchets heat into the top layer in stratification); "parent" uses the
#: parent's third-order interior flux with two parent levels above (the
#: degenerate gate: a child driven by its own parent reproduces it);
#: "upwind" is D3's, kept to measure.
FACE_SCHEMES = {"second_order": 0, "parent": 1, "upwind": 2}

#: MPAS's relaxation zone has nRelaxZone - 1 = 4 rings (bdyMask 2..5).
MAX_RELAX_LEVELS = 4


@dataclass(frozen=True)
class HexPartialNestBand:
    """One partial-depth band of a hex run.

    ``interface_level``: the 0-based ``zgrid`` face of the parent that is the
    child's top (the child has that many layers).  ``parent_time_step_ratio``:
    child steps per parent step.  ``relax_levels``: the top zone's depth in
    layers below the interface, each relaxed with MPAS's relaxation-zone
    arithmetic at the ring strength ``bdyMask = 5 - d`` (``d`` layers below the
    interface).  ``feedback`` 1 restricts ``feedback_fields`` onto the parent
    below the interface after each parent step.  ``mass_correction`` closes
    the parent's interface mass budget over the child as a face-flux
    correction.  ``child_mesh``: the band's regional grid file (the namelist
    route); the programmatic route passes a mesh object instead.
    """

    interface_level: int
    parent_time_step_ratio: int
    relax_levels: int = MAX_RELAX_LEVELS
    feedback: int = 1
    feedback_fields: tuple = FEEDBACK_FIELDS
    mass_correction: bool = True
    face_scheme: str = "second_order"
    child_mesh: str | None = None

    def __post_init__(self) -> None:
        validate_band(self)


def validate_band(band: HexPartialNestBand) -> HexPartialNestBand:
    """Refuse, by name, every band setting the formulation does not carry."""
    if int(band.interface_level) < 4:
        raise PartialNestRefused(
            f"interface_level {band.interface_level}: the child needs at least four "
            "layers (its face below the interface takes the parent's third-order "
            "interior flux, whose stencil reaches three layers down)")
    if int(band.parent_time_step_ratio) < 1:
        raise PartialNestRefused("parent_time_step_ratio must be >= 1")
    if not 1 <= int(band.relax_levels) <= MAX_RELAX_LEVELS:
        raise PartialNestRefused(
            f"relax_levels {band.relax_levels}: MPAS's relaxation zone has "
            f"{MAX_RELAX_LEVELS} ring strengths (bdyMask 2..5), one per top-zone layer")
    if int(band.relax_levels) + 2 > int(band.interface_level):
        raise PartialNestRefused("the top zone does not fit below the interface")
    if band.feedback not in (0, 1):
        raise PartialNestRefused("feedback must be 0 or 1")
    for name in band.feedback_fields:
        if name in _REFUSED_FEEDBACK:
            raise PartialNestRefused(f"feedback_fields {name!r}: {_REFUSED_FEEDBACK[name]}")
        if name not in FEEDBACK_FIELDS:
            raise PartialNestRefused(
                f"feedback_fields {name!r}: only {FEEDBACK_FIELDS} (with the scalars) "
                "is fed back")
    if band.face_scheme not in FACE_SCHEMES:
        raise PartialNestRefused(
            f"face_scheme {band.face_scheme!r}: one of {sorted(FACE_SCHEMES)}")
    return band


def refuse_unsupported_config(config: Any, *, role: str = "child") -> None:
    """What a partial child's dycore configuration may not carry, by name.

    Each refusal is a term whose layer or flux is measured from the MODEL
    TOP or crosses the interface through a path the band does not couple."""
    from woof.hex.cuda_acoustic_fusion_v841 import fusion_enabled
    if getattr(config, "source_release", None) != "v8.4.1":
        raise PartialNestRefused("the partial band runs the v8.4.1 CUDA dycore")
    if bool(getattr(config, "config_rayleigh_damp_u", False)):
        raise PartialNestRefused(
            "config_rayleigh_damp_u counts its levels from the model top, which "
            f"a partial {role} does not have")
    if float(getattr(config, "config_mpas_cam_coef", 0.0)) != 0.0:
        raise PartialNestRefused(
            "config_mpas_cam_coef damps the top config_number_cam_damping_levels, "
            f"measured from the model top a partial {role} does not have")
    if bool(getattr(config, "config_monotonic", False)) or bool(
            getattr(config, "config_positive_definite", False)):
        raise PartialNestRefused(
            "config_monotonic / config_positive_definite: the limited scalar "
            "transport's bounds and low-order flux through the interface are not "
            "carried; run the unlimited third-order transport")
    if getattr(config, "config_les_model", "none") != "none":
        raise PartialNestRefused("an LES closure's vertical fluxes close on a lid")
    if bool(getattr(config, "config_vertical_mixing", False)) or (
            getattr(config, "config_horiz_mixing", "off") != "off"
            and bool(getattr(config, "config_mix_full", False))):
        raise PartialNestRefused(
            "config_mix_full / vertical mixing: the 3-D mixing's vertical fluxes "
            "close on a lid at the interface (the 2-D horizontal mixing is carried)")
    if int(getattr(config, "config_dynamics_split_steps", 1)) != 3 or not bool(
            getattr(config, "config_split_dynamics_transport", False)):
        raise PartialNestRefused(
            "the band runs the v8.4.1 split-three dynamics/transport schedule")
    if fusion_enabled():
        raise PartialNestRefused(
            "WOOF_HEX_ACOUSTIC_FUSION=1 fuses the column solve the interface "
            "specifies its top face in; unset it")
    if bool(getattr(config, "config_apply_lbcs", True)) is False and role == "child":
        raise PartialNestRefused("the child's lateral zones need config_apply_lbcs")


def refuse_damping_below(config: Any, interface_height_m: float) -> None:
    """The w damping layer (``config_xnutr`` from ``config_zd``) is measured
    from the model top; it must lie wholly above the interface."""
    if float(getattr(config, "config_xnutr", 0.0)) != 0.0 and float(
            getattr(config, "config_zd", 0.0)) < float(interface_height_m):
        raise PartialNestRefused(
            f"config_zd {config.config_zd} m is below the interface "
            f"({interface_height_m:.0f} m): the w damping layer would reach into "
            "the band, where its profile is measured from a top the child lacks")


# ---------------------------------------------------------------------------
# the interface and truncation
# ---------------------------------------------------------------------------

def interface_height(zgrid: np.ndarray, level: int, *, what: str = "mesh") -> float:
    """The flat height of face ``level``, refused when any column differs."""
    z = np.asarray(zgrid)
    if not 1 <= int(level) < z.shape[0] - 1:
        raise PartialNestRefused(
            f"interface_level {level} must be an interior face (1..{z.shape[0] - 2}) "
            "with at least one parent level above it")
    row = z[int(level)]
    if np.any(row != row.flat[0]):
        spread = float(np.max(row) - np.min(row))
        raise PartialNestRefused(
            f"zgrid face {level} is not flat on the {what} (spread {spread:.3g} m): "
            "the interface must be a level where the terrain influence has decayed")
    return float(row.flat[0])


def _cut(value: Any, nlev: int, k: int) -> Any:
    if not isinstance(value, np.ndarray) or value.ndim == 0:
        return value
    if value.shape[0] == nlev + 1:
        return np.ascontiguousarray(value[:k + 1])
    if value.shape[0] == nlev:
        return np.ascontiguousarray(value[:k])
    if value.ndim == 3 and value.shape[1] == nlev:
        return np.ascontiguousarray(value[:, :k])
    return value


def truncate(obj: Any, nlev: int, k: int) -> Any:
    """A full-depth host object (``VerticalGrid``, ``PrognosticState``, a
    reference / saved / terrain / wind dataclass) cut below face ``k``."""
    if obj is None:
        return None
    if dataclasses.is_dataclass(obj):
        return replace(obj, **{f.name: _cut(getattr(obj, f.name), nlev, k)
                               for f in fields(obj)
                               if isinstance(getattr(obj, f.name), np.ndarray)})
    return _cut(obj, nlev, k)


# ---------------------------------------------------------------------------
# the WOOF-only namelist group
# ---------------------------------------------------------------------------

#: The group's name and its knobs (MPAS-style names).
NAMELIST_GROUP = "woof_partial_nest"
_KNOBS = {
    "config_partial_nest_mesh": ("child_mesh", str),
    "config_partial_nest_interface_level": ("interface_level", int),
    "config_partial_nest_time_step_ratio": ("parent_time_step_ratio", int),
    "config_partial_nest_relax_levels": ("relax_levels", int),
    "config_partial_nest_feedback": ("feedback", int),
    "config_partial_nest_feedback_fields": ("feedback_fields", str),
    "config_partial_nest_mass_correction": ("mass_correction", bool),
    "config_partial_nest_face_scheme": ("face_scheme", str),
}


def band_from_namelist(text: str) -> HexPartialNestBand | None:
    """The ``&woof_partial_nest`` group of a namelist as a band, or ``None``
    when the group is absent.  Unknown knobs and anything the band cannot
    carry are refused by name; the rest of the namelist is not read here."""
    from woof.hex.config import parse_fortran_namelist
    parsed = parse_fortran_namelist(text)
    group = None
    for name, assignments in parsed.items():
        if str(name).lower() == NAMELIST_GROUP:
            group = assignments
    if group is None:
        return None
    return band_from_mapping(group, where=f"&{NAMELIST_GROUP}")


def band_from_mapping(group: Mapping[str, Any], *, where: str = "partial nest") -> HexPartialNestBand:
    values: dict[str, Any] = {}
    for knob, raw in group.items():
        key = str(knob).lower()
        if key not in _KNOBS:
            raise PartialNestRefused(f"{where}: unknown knob {knob!r} (known: {sorted(_KNOBS)})")
        attribute, kind = _KNOBS[key]
        value = getattr(raw, "value", raw)
        if kind is bool:
            if isinstance(value, str):
                value = value.strip().strip(".").lower() in ("true", "t")
            value = bool(value)
        elif kind is int:
            value = int(value)
        else:
            value = str(value).strip().strip("'\"")
        if attribute == "feedback_fields":
            value = tuple(v.strip() for v in str(value).split(",") if v.strip())
        values[attribute] = value
    missing = {"interface_level", "parent_time_step_ratio"} - set(values)
    if missing:
        raise PartialNestRefused(f"{where} is missing {sorted(missing)}")
    return HexPartialNestBand(**values)


__all__ = [
    "FACE_SCHEMES", "FEEDBACK_FIELDS", "HexPartialNestBand", "MAX_RELAX_LEVELS",
    "NAMELIST_GROUP", "PartialNestRefused", "band_from_mapping", "band_from_namelist",
    "interface_height", "refuse_damping_below", "refuse_unsupported_config", "truncate",
    "validate_band",
]
