# WOOF-OPTIN: wrf-hydro
"""WRF-Hydro's overland routing on device (``overland.cu``), rt_option = 1.

A port of ``OverlandRouting`` (hydro/Routing/Noah_distr_routing_overland.F90 in
WRF 4.7.1's hydro/): when any cell's infiltration excess exceeds its retention
depth, OV_RTNG sets the head to it and takes INT(DT/DTRT_TER) substeps, each
capping channel cells at 5 mm (water above goes to the channel) and lake cells
at their retention (water above goes to the lake), then ROUTE_OVERLAND1 moves
water down the steepest head-adjusted slope (GETMAX8DIR when the terrain
neighbour's head reverses it) by a Manning-type law and drains the domain
edges.  ``rt_option = 2`` runs ROUTE_OVERLAND2 instead: an x sweep then a
y sweep over each cell's east and north links on the routing
initialisation's slopes ``sox``/``soy`` (module_RT :1095-1098), with no
GETMAX8DIR and no edge drain.  The retention depth is the caller's, already
reduced by slope as module_RT does for its rt_option (:1099-1112 for 2,
:1186-1194 for 1).  The mass-balance diagnostics are not computed.

Planes are ``(jxrt, ixrt)``: float32, int32 for ch_netrt, lake_mask, sdir
(3, jxrt, ixrt); slope (8, ...), dist (9, ...) as in ``subsurface``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._compile import raw_module

_TPB = 128
_MODULE = None


def _k(name):
    global _MODULE
    if _MODULE is None:
        kernels = Path(__import__("woof.core.kernels", fromlist=["x"]).__file__).parent
        here = Path(__file__).resolve().parent
        source = ((kernels / "glibc_flt32.cuh").read_text(encoding="utf-8") + "\n"
                  + (here / "overland.cu").read_text(encoding="utf-8"))
        _MODULE = raw_module(source, ("--fmad=false", "--ftz=false"))
    return _MODULE.get_function(name)


@dataclass
class OverlandState:
    """The overland control and stream/lake fields WRF-Hydro carries."""

    head: object                 # surface_water_head_routing (mm)
    retention: object            # retention_depth (mm); channel cells become 5
    to_channel: object           # surface_water_to_channel (mm)
    to_lake: object              # surface_water_to_lake (mm)
    boundary_flux: object        # boundary_flux (mm)
    totals: object               # (3,) float32: accumulated to channel, to lake, boundary_flux_total


class OverlandRouting:
    def __init__(self, *, dist, slope, sdir, roughness, ch_netrt, lake_mask, rt_option=1,
                 sox=None, soy=None):
        import cupy as cp

        if rt_option not in (1, 2):
            raise ValueError("WRF-Hydro rt_option is 1 or 2")
        if rt_option == 2 and (sox is None or soy is None):
            raise ValueError("rt_option = 2 routes on the terrain slopes sox and soy")
        self.rt_option, self.sox, self.soy = int(rt_option), sox, soy
        self.jxrt, self.ixrt = map(int, roughness.shape)
        n = self.ixrt * self.jxrt
        self.dist, self.slope, self.sdir = dist, slope, sdir
        self.roughness, self.ch_netrt, self.lake_mask = roughness, ch_netrt, lake_mask
        z = lambda dt=cp.float32: cp.zeros((self.jxrt, self.ixrt), dt)  # noqa: E731
        self._dch, self._dlk = z(), z()
        self._adj, self._qq = z(), z()
        self._target, self._bnd = z(cp.int32), z(cp.int32)
        self._flag = cp.zeros((1,), cp.int32)
        self._grid = ((n + _TPB - 1) // _TPB,)

    def route(self, state: OverlandState, infiltration_excess, qx, qy, *, dt, dtrt_ter) -> None:
        """One ``OverlandRouting`` call."""
        import cupy as cp

        n, xx, yy = self.ixrt * self.jxrt, np.int32(self.ixrt), np.int32(self.jxrt)
        g, t = self._grid, (_TPB,)
        self._flag.fill(0)
        _k("ov_any_above")(g, t, (infiltration_excess, state.retention, self._flag, np.int32(n)))
        cp.copyto(state.head, infiltration_excess)
        if int(self._flag.get()[0]) == 0:
            return
        qx.fill(0)
        qy.fill(0)
        steps = int(np.float32(dt) / np.float32(dtrt_ter))
        acc = state.totals[0:2]
        for _ in range(steps):
            _k("ov_retention")(g, t, (self.ch_netrt, self.lake_mask, state.retention, state.head,
                                      state.to_channel, state.to_lake, self._dch, self._dlk,
                                      np.int32(n)))
            _k("ov_retention_total")((1,), (1,), (self.ch_netrt, self.lake_mask, self._dch,
                                                  self._dlk, acc, np.int32(n)))
            if self.rt_option == 2:
                self._overland2(state, qx, qy, np.float32(dtrt_ter), xx, yy, n)
                continue
            _k("ov_route_cell")(g, t, (np.float32(dtrt_ter), self.dist, state.head, state.retention,
                                       self.roughness, self.slope, self.sdir, qx, qy, self._adj,
                                       self._qq, self._target, self._bnd, xx, yy))
            _k("ov_gather")(g, t, (self._adj, self._qq, self._target, self._bnd, state.head,
                                   state.boundary_flux, xx, yy))
            _k("ov_total")((1,), (1,), (self._qq, self._target, self._bnd, state.totals[2:3], xx, yy))
            _k("ov_edge")(g, t, (state.head, state.boundary_flux, state.retention, xx, yy))

    def _overland2(self, state, qx, qy, dt, xx, yy, n):
        """ROUTE_OVERLAND2 (:666-929): the x sweep, H = H + DH, the y sweep, H = H + DH."""
        import cupy as cp

        if not hasattr(self, "_dh"):
            self._dh = cp.zeros((self.jxrt, self.ixrt), cp.float32)
            self._tb = cp.zeros((self.jxrt, self.ixrt), cp.float32)
            self._fb = cp.zeros((self.jxrt, self.ixrt), cp.int32)
        for d, so, q, lines in ((0, self.sox, qx, self.jxrt), (1, self.soy, qy, self.ixrt)):
            _k("ov2_sweep")(((lines + _TPB - 1) // _TPB,), (_TPB,), (
                np.int32(d), dt, self.dist, state.head, state.retention, self.roughness, so, q,
                self._dh, state.boundary_flux, self._tb, self._fb, xx, yy))
            _k("ov2_apply")(self._grid, (_TPB,), (state.head, self._dh, np.int32(n)))
            _k("ov2_total")((1,), (1,), (self._tb, self._fb, state.totals[2:3], np.int32(n)))
