"""Height-dependent horizontal resolution: the partial-depth nest's column.

A finer horizontal grid covers a footprint of its parent from the surface up
to an interface level ``k_top`` only; above it the parent alone carries the
atmosphere (plans/sorted-mapping-hummingbird.md, Phase D).  The interface is
a level where WRF's hybrid coordinate is pure pressure -- at or above
``etac`` (``B = c3 = 0``) -- so its pressure ``p_i`` is one constant, and the
child is the parent's own discrete column cut there:

* the same eta levels below the interface, the same ``c1`` and ``c3``;
* its own top pressure ``p_i``, with the pressure-weighted coefficients
  re-expressed for it: ``c4' = c4 + c3*(p_i - p_top) + p_top - p_i`` and
  ``c2' = c2 + c1*(p_i - p_top)``;
* column mass ``mu' = mu - (p_i - p_top)`` (``p_s - p_i``).

Then ``c3'*mu' + c4' + p_i`` is the parent's ``c3*mu + c4 + p_top`` on every
shared level and ``c1*mu' + c2'`` is the parent's mass per unit eta
``c1*mu + c2``: below the interface the child's discrete equations are the
parent's, and the coordinate moves no word of a configuration that has no
partial nest.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from woof.core.grid import VerticalCoord


class PartialNestRefused(ValueError):
    """A partial-depth nest the formulation cannot carry, named."""


def interface_pressure(parent: VerticalCoord, k_top: int) -> float:
    """``p_i``, the dry pressure of full level ``k_top`` -- a constant
    because the level is pure pressure (``c3f[k_top] == 0``)."""
    if parent.p_top is None or parent.c3f is None:
        raise PartialNestRefused("the parent coordinate has no finalized hybrid coefficients "
                                 "(make_base_state sets them with p_top)")
    nz = len(parent.znu)
    if not 1 <= k_top < nz:
        raise PartialNestRefused(f"the interface must be an interior full level of the parent "
                                 f"(1..{nz - 1}), not {k_top}")
    if float(parent.c3f[k_top]) != 0.0:
        raise PartialNestRefused(
            f"full level {k_top} (eta {float(parent.znw[k_top]):.4f}) is terrain-following "
            f"(B = {float(parent.c3f[k_top]):.3g}); a partial-depth nest's interface must be a "
            f"pure pressure level, at or above etac = {parent.etac}: raise etac or the interface")
    return float(parent.c4f[k_top]) + float(parent.p_top)


def truncated_vertical_coord(parent: VerticalCoord, k_top: int) -> VerticalCoord:
    """The parent's coordinate below full level ``k_top`` as a column of its
    own, its top at the interface pressure (module docstring)."""
    p_i = interface_pressure(parent, k_top)
    shift = p_i - float(parent.p_top)
    full, half = slice(0, k_top + 1), slice(0, k_top)
    child = replace(
        parent,
        znw=parent.znw[full].copy(), znu=parent.znu[half].copy(),
        dnw=parent.dnw[half].copy(), rdnw=parent.rdnw[half].copy(),
        dn=parent.dn[half].copy(), rdn=parent.rdn[half].copy(),
        fnp=parent.fnp[half].copy(), fnm=parent.fnm[half].copy(),
        p_top=p_i,
        c1f=parent.c1f[full].copy(), c3f=parent.c3f[full].copy(),
        c1h=parent.c1h[half].copy(), c3h=parent.c3h[half].copy(),
        c2f=parent.c2f[full] + parent.c1f[full] * shift,
        c2h=parent.c2h[half] + parent.c1h[half] * shift,
        # (c4 + c3*shift + p_top) - p_i: zero at the interface itself,
        # where c3 = 0 and c4 + p_top is p_i.
        c4f=parent.c4f[full] + parent.c3f[full] * shift + float(parent.p_top) - p_i,
        c4h=parent.c4h[half] + parent.c3h[half] * shift + float(parent.p_top) - p_i,
    )
    return child


def child_column_mass(parent_mu, parent: VerticalCoord, k_top: int):
    """``mu' = mu - (p_i - p_top)``: the child's dry column mass over a
    column whose parent mass is ``parent_mu`` (any array module)."""
    return parent_mu - (interface_pressure(parent, k_top) - float(parent.p_top))


# ---------------------------------------------------------------------------
# D0: one partial child band, and everything it cannot carry, refused by name
# ---------------------------------------------------------------------------

#: Deep convection is refused while the interface sits below this height
#: (plans/sorted-mapping-hummingbird.md D0): a deep-cumulus column would
#: reach through the child's top into a parent column the child cannot see.
DEEP_CUMULUS_MIN_INTERFACE_M = 12000.0

_BAND_KEYS = frozenset({
    "k_top", "parent_grid_ratio", "parent_time_step_ratio",
    "i_parent_start", "j_parent_start", "nx", "ny", "spec_levels",
    "relax_levels", "feedback", "smooth_option", "mass_correction"})


@dataclass(frozen=True)
class PartialNestBand:
    """One partial-depth child band (the minimal D0 config entry).

    The child covers parent cells ``i_parent_start .. i_parent_start +
    nx/ratio - 1`` (1-based, WRF's namelist meaning) from the surface to the
    parent's full level ``k_top``, a pure-pressure level.  Its top zone is
    ``spec_levels`` specified levels plus ``relax_levels`` relaxation levels
    below the interface (WRF's spec_zone and relax_zone - spec_zone, turned
    on their side); ``feedback`` 1 restricts the child onto the parent below
    the interface each parent step with WRF's copy_fcn and, for
    ``smooth_option`` 1/2, its sm121/smdsm smoother.  ``mass_correction``
    hands the child the parent's interface budget through dmdt_adj.
    """

    k_top: int
    parent_grid_ratio: int
    parent_time_step_ratio: int
    i_parent_start: int
    j_parent_start: int
    nx: int
    ny: int
    spec_levels: int = 1
    relax_levels: int = 5
    feedback: int = 1
    smooth_option: int = 0
    mass_correction: bool = True

    @property
    def relax_zone(self) -> int:
        """WRF's relax_zone: the specified levels plus the relaxation levels."""
        return int(self.spec_levels) + int(self.relax_levels)

    @property
    def strip_width(self) -> int:
        """The top strip's depth, ``bdy_width``'s relax_zone + 1 (the inner
        row the relaxation stencil reads below its last row)."""
        return self.relax_zone + 1


def partial_nest_from_mapping(table) -> "PartialNestBand | None":
    """The ``[partial_nest]`` config entry, or ``None`` when absent.

    ``table`` is the parsed mapping (TOML or dict).  A list of tables is
    more than one band and is refused by name (D4); unknown keys are
    refused, so a misspelt key cannot silently fall back to a default.
    """
    if table is None:
        return None
    if isinstance(table, (list, tuple)):
        if len(table) != 1:
            raise PartialNestRefused(
                f"{len(table)} partial-nest bands are configured; D3 carries one "
                "band (multiple bands, coarsest first, are D4)")
        table = table[0]
    unknown = set(table) - _BAND_KEYS
    if unknown:
        raise PartialNestRefused(f"unknown [partial_nest] keys {sorted(unknown)}")
    missing = {"k_top", "parent_grid_ratio", "i_parent_start", "j_parent_start",
               "nx", "ny"} - set(table)
    if missing:
        raise PartialNestRefused(f"[partial_nest] is missing {sorted(missing)}")
    values = dict(table)
    values.setdefault("parent_time_step_ratio", values["parent_grid_ratio"])
    for key in _BAND_KEYS - {"mass_correction"}:
        if key in values:
            values[key] = int(values[key])
    if "mass_correction" in values:
        values["mass_correction"] = bool(values["mass_correction"])
    return PartialNestBand(**values)


def validate_partial_nest(band, parent_cfg, parent_coord: VerticalCoord, *,
                          interface_height_m: float | None = None,
                          has_msf: bool = False):
    """Refuse, by name, every configuration D3 does not carry.

    ``band`` may be a :class:`PartialNestBand`, a sequence of them (more
    than one is refused) or ``None`` (no partial nest: returns ``None`` and
    touches nothing).  ``interface_height_m`` is the interface's lowest
    height above sea level (the deep-cumulus rule needs it whenever
    cumulus is on).
    """
    from woof import wrf_exact
    if band is None:
        return None
    if isinstance(band, (list, tuple)):
        if len(band) != 1:
            raise PartialNestRefused(
                f"{len(band)} partial-nest bands are configured; D3 carries one "
                "band (multiple bands, coarsest first, are D4)")
        band = band[0]
    if wrf_exact.ENABLED:
        raise PartialNestRefused(
            "a partial-depth nest runs in the default arithmetic build only; this "
            "process is strict (GPUWM_WRF_EXACT=1)")
    k = int(band.k_top)
    interface_pressure(parent_coord, k)          # pure pressure, interior level
    if float(parent_coord.c1f[k]) != 0.0:
        raise PartialNestRefused(
            f"full level {k} is the etac level itself (c1 = dB/deta = "
            f"{float(parent_coord.c1f[k]):.3g} there): the interface's mass coupling "
            "must be the constant c2 alone, so it has to sit strictly above etac")
    if band.spec_levels < 1 or band.relax_levels < 1:
        raise PartialNestRefused("the top zone needs at least one specified and one "
                                 "relaxation level")
    if band.relax_zone + 2 > k:
        raise PartialNestRefused(
            f"the top zone ({band.spec_levels} specified + {band.relax_levels} "
            f"relaxation levels) does not fit below interface level {k}")
    if int(parent_cfg.damp_opt) == 3:
        raise PartialNestRefused(
            "damp_opt=3 measures its layer from the model top, which a partial-depth "
            "child does not have; its top zone is the relaxation zone")
    if int(getattr(parent_cfg, "cu_physics", 0)) != 0:
        if interface_height_m is None:
            raise PartialNestRefused(
                "deep cumulus is on and the interface height is unknown; the "
                f"interface must be above {DEEP_CUMULUS_MIN_INTERFACE_M:.0f} m")
        if interface_height_m < DEEP_CUMULUS_MIN_INTERFACE_M:
            raise PartialNestRefused(
                f"deep cumulus (cu_physics={parent_cfg.cu_physics}) with the "
                f"interface at {interface_height_m:.0f} m, below "
                f"{DEEP_CUMULUS_MIN_INTERFACE_M:.0f} m")
    if has_msf:
        raise PartialNestRefused("map factors other than 1 are not carried by the "
                                 "partial-depth nest's top entry points")
    r = int(band.parent_grid_ratio)
    if r < 2 or int(band.parent_time_step_ratio) < 1:
        raise PartialNestRefused("parent_grid_ratio must be >= 2 and "
                                 "parent_time_step_ratio >= 1")
    if band.nx % r or band.ny % r:
        raise PartialNestRefused(
            f"the child's extent {band.ny}x{band.nx} must be whole parent cells "
            f"at ratio {r} (the interface budget is kept per parent cell)")
    if band.feedback not in (0, 1) or band.smooth_option not in (0, 1, 2):
        raise PartialNestRefused("feedback must be 0/1 and smooth_option 0/1/2")
    if parent_cfg.open_x or parent_cfg.open_y:
        raise PartialNestRefused("radiative-open parent boundaries are not carried")
    return band
