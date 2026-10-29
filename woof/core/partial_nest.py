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

from dataclasses import replace

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
