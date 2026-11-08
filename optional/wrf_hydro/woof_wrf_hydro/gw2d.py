# WOOF-OPTIN: wrf-hydro
"""WRF-Hydro's 2-D groundwater model on device (``gw2d.cu``), GWBASESWCRT = 3.

``hydro/Routing/module_gw_gw2d.F90`` in WRF 4.7.1's hydro/: :meth:`Gw2d.init`
is ``gw2d_ini`` (the hydraulic-conductivity sentinel 100 -> 5E-4, then
GwPreCycles calls of gwstep), and :meth:`Gw2d.hydro_step` is the
GWBASESWCRT = 3 arm of HYDRO_drv's groundwater call
(module_HYDRO_drv.F90:849-874): gwSoilFlux when gwsoilcpl > 0, gwstep, the
new head, the surface excess added to the routing grid's ponded head
(mm), aggregateQsgw, istep + 1.  gwstep is WRF-Hydro's MPP_LAND build on one
process: an implicit N-S line sweep then an E-W line sweep (sub_tri_solv),
the aquifer re-wetting fix, the ocean exchange and the surface excess.

Planes are device float32 (jxrt, ixrt) (Fortran's (ixrt, jxrt)); soil
moisture is (nsoil, jxrt, ixrt).  ``ltype`` is 1 land, 2 ocean, other values
inactive; ``ebot``/``eocn`` are gwstep's carried diagnostics, which WRF-Hydro
rescales by the land area and dt every call.  The groundwater-channel
exchange is the diffusive wave's (``DiffusiveWaveRouting`` with
``gwbaseswcrt=3``, which writes ``qgw_chanrt``) and
:meth:`Gw2d.channel_exchange`, HYDRO_drv's head correction after it.
"""

from __future__ import annotations

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
                  + (here / "gw2d.cu").read_text(encoding="utf-8"))
        _MODULE = raw_module(source, ("--fmad=false", "--ftz=false"))
    return _MODULE.get_function(name)


def _g(n):
    return ((n + _TPB - 1) // _TPB,)


class Gw2d:
    """One domain's groundwater: static aquifer, carried head and fluxes."""

    def __init__(self, *, ltype, elev, bot, hycond, poros, ho, dx, dt, aggfactrt):
        import cupy as cp

        f = lambda a: cp.ascontiguousarray(cp.asarray(a, dtype=cp.float32))  # noqa: E731
        self.ltype = cp.ascontiguousarray(cp.asarray(ltype, dtype=cp.int32))
        self.jx, self.ix = self.ltype.shape
        self.elev, self.bot, self.hycond, self.poros = f(elev), f(bot), f(hycond), f(poros)
        self.ho = f(ho)
        self.h = self.ho.copy()
        z = lambda: cp.zeros((self.jx, self.ix), cp.float32)  # noqa: E731
        self.convgw, self.excess, self.qsgwrt, self.qdarcyrt = z(), z(), z(), z()
        self.qgw_chanrt = z()
        self.dx, self.dt, self.agg = np.float32(dx), np.float32(dt), int(aggfactrt)
        if self.ix % self.agg or self.jx % self.agg:
            raise ValueError("the routing grid is AGGFACTRT times the LSM grid")
        self.qsgw = cp.zeros((self.jx // self.agg, self.ix // self.agg), cp.float32)
        self.e = cp.zeros(2, cp.float32)          # ebot, eocn
        self.istep = 0
        self._t1, self._t2, self._wk, self._x = z(), z(), z(), z()
        if self.ix > self.jx:
            # gwstep allocates zfac(1:ydim) and sub_tri_solv's N-S sweep
            # writes zfac(1:xdim): past the end when IX > JX.
            raise NotImplementedError("gw2d with IXRT > JXRT: WRF-Hydro's gwstep writes past "
                                      "its zfac work array (allocated JXRT long)")

    @property
    def ebot(self):
        return self.e[0]

    @property
    def eocn(self):
        return self.e[1]

    def init(self, *, gw_pre_cycles: int) -> None:
        """gw2d_ini (:29-86)."""
        import cupy as cp

        self.qsgwrt.fill(0)
        self.qdarcyrt.fill(0)
        self.qgw_chanrt.fill(0)
        self.excess.fill(0)
        self.istep = 0
        self.hycond[...] = cp.where(self.hycond == cp.float32(100), cp.float32(5e-4), self.hycond)
        for _ in range(int(gw_pre_cycles)):
            self.gwstep()
            self.ho[...] = self.h

    def gwstep(self) -> None:
        """gwstep (:154-913) on the current ho; writes h, convgw, excess, ebot, eocn."""
        ix, jx, n = np.int32(self.ix), np.int32(self.jx), self.ix * self.jx
        darea = np.float32(self.dx * self.dx)
        self.h[...] = self.ho
        _k("gw_trans")(_g(n), (_TPB,), (ix, jx, self.dx, self.h, self.elev, self.bot,
                                        self.hycond, self._t1, self._t2))
        _k("gw_sweep_ns")(_g(self.ix), (_TPB,), (ix, jx, self.dt, darea, self.poros, self.ho,
                                                 self.h, self._t1, self._t2, self._wk, self._x))
        self.h[...] = self._x
        _k("gw_trans")(_g(n), (_TPB,), (ix, jx, self.dx, self.h, self.elev, self.bot,
                                        self.hycond, self._t1, self._t2))
        _k("gw_sweep_ew")(_g(self.jx), (_TPB,), (ix, jx, self.dt, darea, self.poros, self.ho,
                                                 self.h, self._t1, self._t2, self._wk, self._x))
        self.h[...] = self._x
        _k("gw_finish")((1,), (1,), (ix, jx, self.dt, darea, self.ltype, self.poros, self.elev,
                                     self.bot, self.ho, self.h, self.convgw, self.excess, self.e))

    def soil_flux(self, *, elrt, soiltyp, smcrt, smcmaxrt, sldpth, zsoil) -> None:
        """gwSoilFlux (:965-1066): updates smcrt in place, qsgwrt, qdarcyrt and ho."""
        import cupy as cp

        nsoil = smcrt.shape[0]
        _k("gw_soilflux")(_g(self.ix * self.jx), (_TPB,), (
            np.int32(self.ix * self.jx), np.int32(nsoil), np.float32(zsoil[nsoil - 1]),
            self.dt, cp.asarray(elrt, cp.float32),
            cp.ascontiguousarray(cp.asarray(soiltyp, cp.int32)), smcrt, smcmaxrt,
            cp.asarray(np.asarray(sldpth, np.float32)), self.ho, self.bot, self.poros,
            self.qsgwrt, self.qdarcyrt))

    def hydro_step(self, *, surface_head, gwsoilcpl, elrt=None, soiltyp=None, smcrt=None,
                   smcmaxrt=None, sldpth=None, zsoil=None) -> None:
        """module_HYDRO_drv.F90:849-874; ``surface_head`` (mm) takes the excess."""
        import cupy as cp

        if gwsoilcpl > 0:
            self.soil_flux(elrt=elrt, soiltyp=soiltyp, smcrt=smcrt, smcmaxrt=smcmaxrt,
                           sldpth=sldpth, zsoil=zsoil)
        self.excess.fill(0)
        self.gwstep()
        self.ho[...] = self.h
        surface_head[...] = surface_head + self.excess * cp.float32(1000.0)
        jx, ix = self.qsgw.shape
        _k("gw_aggregate")(_g(ix * jx), (_TPB,), (np.int32(ix), np.int32(jx),
                                                  np.int32(self.agg), np.int32(self.ix),
                                                  self.qdarcyrt, self.qsgw))
        self.istep += 1

    def channel_exchange(self, *, gw_chan_cond_sw) -> None:
        """module_HYDRO_drv.F90:986-994: with gwChanCondSw = 1, the channel
        flux ``qgw_chanrt`` the diffusive wave wrote leaves the aquifer head."""
        if int(gw_chan_cond_sw) != 1:
            return
        n = self.ix * self.jx
        _k("gw_chan_head")(_g(n), (_TPB,), (np.int32(n), self.dt, self.dx, self.qgw_chanrt,
                                            self.poros, self.ho))
