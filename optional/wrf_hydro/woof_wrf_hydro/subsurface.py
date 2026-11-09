# WOOF-OPTIN: wrf-hydro
"""WRF-Hydro's subsurface routing on device (``subsurface.cu``).

A port of ``subsurfaceRouting`` (hydro/Routing/Noah_distr_routing_subsurface.F90
in WRF 4.7.1's hydro/ tree): FINDZWAT finds each column's saturated layers,
water-table depth and routable water; ROUTE_SUBSURFACE1 sends each interior
cell's saturated lateral flow to its steepest total-head neighbour (GETSUB8)
and books flow leaving the domain; SUBSFC_RTNG adds or drains that flow layer
by layer and spills what a full column cannot hold into the infiltration
excess.  ``rt_option = 2`` selects ROUTE_OVERLAND2 for overland flow only:
WRF-Hydro itself hard-codes ROUTE_SUBSURFACE1 here (subsurfaceRouting
:235-256, ROUTE_SUBSURFACE2 commented out), so both values route the same.

One :meth:`SubsurfaceRouting.route` is one ``subsurfaceRouting`` call; call
:meth:`SubsurfaceState.end_hydro_step` after the rest of the HYDRO step, as
HYDRO_exe zeroes QSUBRT (module_HYDRO_drv.F90:747).  WRF-Hydro stops the run
on five conditions; :meth:`route` raises ``RuntimeError`` naming them, after
one 4-byte read of the error word.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._compile import raw_module

_TPB = 128
_ERRORS = {1: "dist(i,j,index) <= 0", 2: "soldep == 0", 4: "hsub < 0",
           8: "qqsub > 0", 16: "SMCMAX exceeded in SUBSFC_RTNG"}
_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        here = Path(__file__).resolve().parent
        kernels = Path(__import__("woof.core.kernels", fromlist=["x"]).__file__).parent
        source = ((kernels / "glibc_flt32.cuh").read_text(encoding="utf-8")
                  + "\n" + (here / "subsurface.cu").read_text(encoding="utf-8"))
        _MODULE = raw_module(source, ("--fmad=false", "--ftz=false"))
    return _MODULE


def _k(name):
    return _module().get_function(name)


@dataclass
class SubsurfaceState:
    """WRF-Hydro's subsurface state (module_subsurface_state.F90) on device."""

    zwattablrt: object   # (jxrt, ixrt) float32, water-table depth (m)
    qsubrt: object       # (jxrt, ixrt) float32, net subsurface flow (m3/s)
    qsubbdryrt: object   # (jxrt, ixrt) float32, flow across the boundary (m3/s)
    qsubbdrytrt: object  # (1,) float32, domain total of the above

    @classmethod
    def zeros(cls, ixrt: int, jxrt: int) -> "SubsurfaceState":
        import cupy as cp

        z = lambda: cp.zeros((jxrt, ixrt), cp.float32)  # noqa: E731
        return cls(z(), z(), z(), cp.zeros((1,), cp.float32))

    def end_hydro_step(self) -> None:
        self.qsubrt.fill(0)


class SubsurfaceRouting:
    """The static part of ``subsurfaceRouting`` and its launches.

    Planes are float32 ``(jxrt, ixrt)``; ``slope`` is (8, jxrt, ixrt) in
    GETSUB8's N, NE, E, SE, S, SW, W, NW order; ``dist`` is (9, jxrt, ixrt)
    with the cell area last; ``sdir`` is int32 (3, jxrt, ixrt) holding WRF's
    1-based steepest-neighbour (i, j, direction).
    """

    def __init__(self, *, dt, sldpth, zsoil, lksat, nexp, soldep, slope, sdir,
                 dist, rt_option: int = 1):
        import cupy as cp

        if rt_option not in (1, 2):
            raise ValueError("WRF-Hydro rt_option is 1 or 2")
        self.jxrt, self.ixrt = map(int, lksat.shape)
        self.nsoil = len(sldpth)
        if len(zsoil) != self.nsoil:
            raise ValueError("zsoil and sldpth need one value per soil layer")
        n = self.ixrt * self.jxrt
        for name, a, shape, dtype in (
                ("lksat", lksat, (self.jxrt, self.ixrt), cp.float32),
                ("nexp", nexp, (self.jxrt, self.ixrt), cp.float32),
                ("soldep", soldep, (self.jxrt, self.ixrt), cp.float32),
                ("slope", slope, (8, self.jxrt, self.ixrt), cp.float32),
                ("dist", dist, (9, self.jxrt, self.ixrt), cp.float32),
                ("sdir", sdir, (3, self.jxrt, self.ixrt), cp.int32)):
            if tuple(a.shape) != shape or a.dtype != dtype:
                raise ValueError(f"{name}: expected {shape} {np.dtype(dtype)}")
        self.dt = np.float32(dt)
        self.sldpth = cp.asarray(np.asarray(sldpth, np.float32))
        self.zsoil = cp.asarray(np.asarray(zsoil, np.float32))
        self.lksat, self.nexp, self.soldep = lksat, nexp, soldep
        self.slope, self.sdir, self.dist = slope, sdir, dist
        self._cwat = cp.zeros((self.jxrt, self.ixrt), cp.float32)
        self._sat = cp.zeros((self.jxrt, self.ixrt), cp.int32)
        self._qq = cp.zeros((self.jxrt, self.ixrt), cp.float32)
        self._target = cp.zeros((self.jxrt, self.ixrt), cp.int32)
        self._bnd = cp.zeros((self.jxrt, self.ixrt), cp.int32)
        self._err = cp.zeros((1,), cp.int32)
        self._grid = ((n + _TPB - 1) // _TPB,)

    def route(self, state: SubsurfaceState, *, smc, smcmax, smcref, smcwlt,
              infiltration_excess) -> None:
        """One ``subsurfaceRouting`` call; ``smc`` and ``infiltration_excess``
        are updated in place, soil fields are (nsoil, jxrt, ixrt)."""
        n, xx, yy = self.ixrt * self.jxrt, self.ixrt, self.jxrt
        nsoil = np.int32(self.nsoil)
        self._err.fill(0)
        _k("hyd_findzwat")(self._grid, (_TPB,), (
            smc, smcref, smcwlt, self.zsoil, self.sldpth, state.zwattablrt,
            self._cwat, self._sat, nsoil, np.int32(n)))
        _k("hyd_route1_cell")(self._grid, (_TPB,), (
            self.dist, state.zwattablrt, self.lksat, self.nexp, self.soldep,
            self.slope, self.sdir, self._cwat, self.dt, state.qsubrt, self._qq,
            self._target, self._bnd, self._err, np.int32(xx), np.int32(yy)))
        _k("hyd_route1_gather")(self._grid, (_TPB,), (
            self._qq, self._target, self._bnd, state.qsubrt, state.qsubbdryrt,
            np.int32(xx), np.int32(yy)))
        _k("hyd_route1_total")((1,), (1,), (
            self._qq, self._target, self._bnd, state.qsubbdrytrt,
            np.int32(xx), np.int32(yy)))
        _k("hyd_subsfc_update")(self._grid, (_TPB,), (
            state.qsubrt, self.dist, self._sat, self.sldpth, smcmax, smcref, smc,
            infiltration_excess, self.dt, self._err, nsoil, np.int32(n)))
        code = int(self._err.get()[0])
        if code:
            names = [text for bit, text in _ERRORS.items() if code & bit]
            raise RuntimeError("WRF-Hydro subsurface routing stopped: "
                               + "; ".join(names))
