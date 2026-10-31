"""Partial-depth nest, Phase D4: physics on a partial child.

plans/sorted-mapping-hummingbird.md, Phase D4.  A partial child (a
:class:`woof.core.partial_nest_run.PartialNest`) steps with
:func:`woof.core.dycore.step`, so its physics driver runs exactly as on any
domain -- surface, PBL and microphysics column by column below its
interface -- with one exception: radiation needs the whole atmosphere above
each column, and the child's column ends at ``k_top``.

The extended column
-------------------
Each radiation call builds the child's column up to the model top
(:class:`ExtendedColumn`): the child's own levels below the interface, and
above it the parent's levels SINT-interpolated onto the child grid (WRF
``sint`` with the band's bdy-wrapper registration, the interpolation the
interface planes take; for a band whose parent is itself partial, from that
parent's own extended column, so every band reaches the root's top).  The
column is a full-depth :class:`woof.core.state.DomainState` on the child's
grid with the root's vertical coordinate: its base state is the root's
(profiles, or SINT-ed 3-D fields over terrain), its column mass the child's
plus the constant mass above the interface (``mu' `` is shared), and its
perturbation theta, geopotential, EOS pressure, inverse density and moist
scalars are refreshed at each radiation call.  The radiation adapter
(:func:`woof.core.radiation_composition.make_radiation`, built for that
column with the root's ``p_top``) runs on it through the ordinary
``phy_prep`` (:func:`woof.core.physics._prepare_atmosphere`), and
:class:`ExtendedColumnRadiation` hands the driver the heating rates (and the
cloud fraction / aerosol optical depth profiles) of the child's levels only:
heating is applied below the interface; above it the parent's own radiation
heats the parent.  Surface fluxes are the extended column's.

What is not carried (refused upstream or documented): deep cumulus below
~12 km (D0), km_opt mixing through the interface, and hydrometeor
sedimentation through the interface (the child's microphysics sees a lid).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from woof.core.partial_nest import PartialNestRefused
from woof.core.state import DTYPE, DomainState

_THREADS = 256

#: Half-level fields refreshed above the interface at each radiation call.
_HALF_REFRESH = ("p", "alt")


def _xp():
    import cupy as cp
    return cp


def _blocks(n: int) -> tuple[int]:
    return ((int(n) + _THREADS - 1) // _THREADS,)


def chain_root(nest):
    """``(root_state, root_cfg)`` of a band's chain."""
    while nest.parent_nest is not None:
        nest = nest.parent_nest
    return nest.parent, nest.pcfg


class ExtendedColumn:
    """The child grid's column to the model top (module docstring)."""

    def __init__(self, nest):
        cp = _xp()
        self.nest = nest
        root, root_cfg = chain_root(nest)
        self.root, self.root_cfg = root, root_cfg
        c, ccfg, k = nest.child, nest.ccfg, nest.k
        self.k = k
        self.cfg = dataclasses.replace(ccfg, nz=int(root_cfg.nz))
        ext = DomainState(self.cfg)
        nz = self.cfg.nz
        # The root's vertical coordinate and 1-D base profiles.
        for name, dev in vars(ext).items():
            src = getattr(root, name, None)
            if (isinstance(dev, cp.ndarray) and isinstance(src, cp.ndarray)
                    and dev.ndim == 1 and src.shape == dev.shape
                    and dev.shape[0] in (nz, nz + 1)):
                dev[...] = src
        for name in ("cf1", "cf2", "cf3", "cfn", "cfn1", "p_top", "mub", "has_msf"):
            if hasattr(root, name):
                setattr(ext, name, getattr(root, name))
        if ext.has_msf:
            raise PartialNestRefused("map factors on the extended radiation column")
        self.mass_above = DTYPE(nest.p_i - float(root.p_top))
        ext.mub2d[...] = c.mub2d + self.mass_above
        if ext.mub is not None and c.mub is not None:
            ext.mub = DTYPE(float(c.mub) + float(self.mass_above))
        ext.ht[...] = c.ht
        self.state = ext
        self.parent_ext = (None if nest.parent_nest is None
                           else nest.parent_nest.extended)
        if nest.parent_nest is not None and self.parent_ext is None:
            raise PartialNestRefused(
                "a band's extended column needs its parent band's (attach the "
                "chain's physics coarse to fine)")
        # 3-D base state (terrain): the child's below, SINT of the parent's
        # column above.  Profiles were copied from the root above.
        src = self._source_state()
        for name in ("thb", "pb", "alb", "phb"):
            dev = getattr(ext, name)
            if dev.ndim != 3:
                continue
            full = name == "phb"
            self._fill(dev, getattr(c, name), getattr(src, name), full)

    def _source_state(self):
        """The full-depth state above this band: the root (first band) or
        the parent band's extended column."""
        if self.parent_ext is None:
            return self.root
        return self.parent_ext.state

    def _fill(self, out, below, source_full, full: bool) -> None:
        """``out`` = ``below`` on the child's levels and SINT(``source``)
        above (``nest_ext_column_fill``)."""
        cp = _xp()
        from woof.core.kernels import get_kernel
        from woof.core.nest_interp import sint
        k = self.k
        nb = k + 1 if full else k
        above = sint(cp.ascontiguousarray(source_full[k:]), self.nest.regs[""])
        n = int(np.prod(out.shape[1:]))
        get_kernel("partial_nest_d4", "nest_ext_column_fill")(
            _blocks(out.size), (_THREADS,),
            (cp.ascontiguousarray(below[:nb]), above, out, np.int32(nb),
             np.int32(out.shape[0]), np.int32(1 if full else 0), np.int32(n)))

    def refresh(self) -> DomainState:
        """The column now: the child below, the parent's SINT above."""
        cp = _xp()
        if self.parent_ext is not None:
            self.parent_ext.refresh()
        nest, ext = self.nest, self.state
        c = nest.child
        src = self._source_state()
        k = self.k
        # Totals above, perturbations against the column's own base.
        thb_s = src.thb if src.thb.ndim == 3 else src.thb[:, None, None]
        phb_s = src.phb if src.phb.ndim == 3 else src.phb[:, None, None]
        theta = ext.scratch(ext.thp.shape, "partial_ext_theta")
        tot_src = src.scratch(src.thp.shape, "partial_ext_src_theta")
        cp.add(src.thp, thb_s, out=tot_src)
        c_tot = c.scratch(c.thp.shape, "partial_ext_child_theta")
        cp.add(c.thp, c.thb if c.thb.ndim == 3 else c.thb[:, None, None], out=c_tot)
        self._fill(theta, c_tot, tot_src, False)
        cp.subtract(theta, ext.thb if ext.thb.ndim == 3 else ext.thb[:, None, None],
                    out=ext.thp)
        phi = ext.scratch(ext.php.shape, "partial_ext_phi")
        src_phi = src.scratch(src.php.shape, "partial_ext_src_phi")
        cp.add(src.php, phb_s, out=src_phi)
        c_phi = c.scratch(c.php.shape, "partial_ext_child_phi")
        cp.add(c.php, c.phb if c.phb.ndim == 3 else c.phb[:, None, None], out=c_phi)
        self._fill(phi, c_phi, src_phi, True)
        cp.subtract(phi, ext.phb if ext.phb.ndim == 3 else ext.phb[:, None, None],
                    out=ext.php)
        for name in _HALF_REFRESH + tuple(nest.species):
            self._fill(getattr(ext, name), getattr(c, name), getattr(src, name), False)
        ext.mup[...] = c.mup
        ext.elapsed_seconds = c.elapsed_seconds
        return ext


class ExtendedColumnRadiation:
    """A radiation adapter run on the child's extended column; the driver
    receives the child's levels (module docstring).  Every other attribute
    is the wrapped adapter's."""

    def __init__(self, inner, column: ExtendedColumn):
        self.inner = inner
        self.column = column

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, *, atmosphere, fields, state, cfg):
        from woof.core.physics import _prepare_atmosphere
        col = self.column
        ext = col.refresh()
        result = self.inner(atmosphere=_prepare_atmosphere(ext), fields=fields,
                            state=ext, cfg=col.cfg)
        k = col.k
        cp = _xp()
        cut = {}
        for name in ("rthratenlw", "rthratensw", "cldfra", "aod5503d"):
            value = getattr(result, name, None)
            if value is not None and getattr(value, "ndim", 0) == 3:
                cut[name] = cp.ascontiguousarray(value[:k])
        return dataclasses.replace(result, **cut)


def attach_child_physics(nest, *, start_time, latitude, longitude, **physics_kwargs):
    """The child's physics driver (``woof.core.physics.initialize_physics``
    on the child state and config) with its radiation on the extended
    column; returns the driver (also ``nest.child.physics``).  A chain
    attaches coarse to fine.  ``latitude``/``longitude``: the child's mass
    grid; ``physics_kwargs`` go to ``initialize_physics`` (surface fields
    and the rest of its documented neutral defaults)."""
    from woof.config import radiation_scheme_ids
    from woof.core.physics import initialize_physics
    from woof.core.radiation_composition import make_radiation
    radiation = None
    if any(radiation_scheme_ids(nest.ccfg)):
        column = ExtendedColumn(nest)
        nest.extended = column
        inner = make_radiation(column.cfg, start_time, latitude, longitude,
                               p_top=float(column.root.p_top))
        radiation = ExtendedColumnRadiation(inner, column)
    elif nest.parent_nest is not None or physics_kwargs.pop("extended", False):
        nest.extended = ExtendedColumn(nest)
    return initialize_physics(nest.child, nest.ccfg, radiation=radiation,
                              radiation_start_time=start_time,
                              radiation_latitude=latitude,
                              radiation_longitude=longitude, **physics_kwargs)
