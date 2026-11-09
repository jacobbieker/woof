# WOOF-OPTIN: wrf-hydro
"""WRF-Hydro's reach channel routing on device (``channel.cu``).

The reach branch of ``drive_CHANNEL`` (hydro/Routing/module_channel_routing.F90
in WRF 4.7.1's hydro/) for channel_option 1 (Muskingum, ``MUSKING``) and 2
(Muskingum-Cunge, ``SUBMUSKINGCUNGE``): each call gathers the lateral inflow
of every reach from its channel pixels (QSTRMVOLRT + QINFLOWBASE, or a lake
pixel's LAKEINFLORT + QINFLOWBASE, as m3/s), then takes INT((DT+0.5)/DTRT_CH)
substeps in which every reach routes its upstream reaches' previous-substep
flows.  A reservoir reach (TYPEL = 1) whose ``lake_lookup`` names a lake
routes it through the reservoir (``lakes``, a
:class:`~woof_wrf_hydro.levelpool.LevelPool`: level pool or pass-through) with
its upstream inflow and no lateral inflow, recording RESHT, QLAKEI and QLAKEO,
and then, as in WRF 4.7.1, takes the zero routed flow every reservoir reach
holds; a TYPEL = 2 reach keeps its flow (the commit skips it).
channel_option 3 (the diffusive wave on nodes) is
:mod:`woof_wrf_hydro.diffusive`.

The network is host data, read once to build the pixel and upstream lists:
``ch_lnkrt_sl`` and ``lake_mskrt`` (numpy (jxrt, ixrt)), ``chanxi``/``chanyj``
(1-based pixel positions of the NLINKS nodes), ``linkid``, ``to_node``.
Per-reach properties and the state are device arrays of length
max(NLINKS, NLINKSL), rt_allocate's nlinksize (the first NLINKSL are reaches).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._compile import raw_module

_TPB = 128
_MODULE = None
_ERRORS = {1: "Error in channel coefficients -> Muskingum cunge", 2: "D is 0 in MUSKINGCUNGE",
           4: "Musk Cunge: failure to converge"}


def _k(name):
    global _MODULE
    if _MODULE is None:
        kernels = Path(__import__("woof.core.kernels", fromlist=["x"]).__file__).parent
        here = Path(__file__).resolve().parent
        source = ((kernels / "glibc_flt32.cuh").read_text(encoding="utf-8") + "\n"
                  + (here / "levelpool.cu").read_text(encoding="utf-8") + "\n"
                  + (here / "channel.cu").read_text(encoding="utf-8"))
        _MODULE = raw_module(source, ("--fmad=false", "--ftz=false"))
    return _MODULE.get_function(name)


class ReachRouting:
    def __init__(self, *, channel_option, ch_lnkrt_sl, lake_mskrt, chanxi, chanyj, node_area,
                 linkid, to_node, nlinksl, typel, musk, musx, so, chanlen, mannn, chsslp,
                 bw, tw, tw_cc, n_cc, channk, lakes=None, lake_lookup=None):
        import cupy as cp

        from .levelpool import LevelPool

        if channel_option == 3:
            raise NotImplementedError("channel_option = 3 (the diffusive wave on nodes) is "
                                      "not reach routing: use DiffusiveWaveRouting")
        if channel_option not in (1, 2):
            raise ValueError("WRF-Hydro channel_option is 1, 2 or 3")
        self.option = int(channel_option)
        nnodes, self.nlinksl = len(chanxi), int(nlinksl)
        self.nlinks = max(nnodes, self.nlinksl)
        jxrt, ixrt = ch_lnkrt_sl.shape
        cell = (np.asarray(chanyj) - 1) * ixrt + (np.asarray(chanxi) - 1)
        lists = [[] for _ in range(self.nlinks)]
        for jj in range(nnodes):
            j, i = int(chanyj[jj]) - 1, int(chanxi[jj]) - 1
            if ch_lnkrt_sl[j, i] > 0:
                lists[int(ch_lnkrt_sl[j, i]) - 1].append((jj, 0))
            elif lake_mskrt[j, i] > 0:
                # a lake pixel's inflow lands on the reach numbered as its lake
                if lake_mskrt[j, i] > self.nlinks:
                    raise ValueError("a LAKEGRID lake number exceeds the link arrays "
                                     "(drive_CHANNEL indexes QLateral with it)")
                lists[int(lake_mskrt[j, i]) - 1].append((jj, 1))
        start = np.cumsum([0] + [len(x) for x in lists]).astype(np.int32)
        flat = [p for x in lists for p in x]
        ups = [[m for m in range(self.nlinksl) if to_node[m] == linkid[k]]
               for k in range(self.nlinksl)]
        ustart = np.cumsum([0] + [len(x) for x in ups]).astype(np.int32)
        d = cp.asarray
        self._start, self._ustart = d(start), d(ustart)
        self._pix = d(np.asarray([p for p, _ in flat] or [0], np.int32))
        self._lake = d(np.asarray([q for _, q in flat] or [0], np.int32))
        self._ups = d(np.asarray([m for x in ups for m in x] or [0], np.int32))
        self._cell = d(cell.astype(np.int32))
        self.node_area = node_area
        self.props = (typel, musk, musx, so, chanlen, mannn, chsslp, bw, tw, tw_cc, n_cc, channk)
        self.lakes = lakes if lakes is not None else LevelPool.empty()
        lake_of = (np.full(self.nlinks, -1, np.int64) if lake_lookup is None
                   else np.asarray(lake_lookup, np.int64).reshape(-1))
        if lake_of.shape != (self.nlinks,):
            raise ValueError("lake_lookup has one entry per link (0-based lake, < 0 for none)")
        named = lake_of[lake_of >= 0]
        if (named >= self.lakes.n).any() or len(set(named.tolist())) != len(named):
            raise ValueError("lake_lookup names each level-pool lake at most once")
        self._lake_of = d(np.where(lake_of >= 0, lake_of, -1).astype(np.int32))
        self._tmp = cp.zeros(self.nlinks, cp.float32)
        self._err = cp.zeros((1,), cp.int32)

    def route(self, *, qlink1, qlink2, hlink, qlateral, velocity, qloss, qstrmvolrt,
              qinflowbase, lakeinflort, dt, dtrt_ch) -> None:
        """One call of drive_CHANNEL's reach branch; state arrays in place."""
        g = lambda n: ((n + _TPB - 1) // _TPB,)  # noqa: E731
        self._err.fill(0)
        _k("ch_lateral")(g(self.nlinks), (_TPB,), (
            self._start, self._pix, self._lake, qstrmvolrt, qinflowbase, lakeinflort,
            self._cell, self.node_area, np.float32(dt), qlateral, np.int32(self.nlinks)))
        nsteps = int((np.float32(dt) + np.float32(0.5)) / np.float32(dtrt_ch))
        typel = self.props[0]
        for _ in range(nsteps):
            _k("ch_reach")(g(self.nlinksl), (_TPB,), (
                np.int32(self.option), self._ustart, self._ups, qlink1, qlink2, qlateral,
                *self.props, np.float32(dtrt_ch), self._tmp, velocity, qloss, hlink,
                self._err, self._lake_of, self.lakes.props, self.lakes.lake_opt,
                self.lakes.resht, self.lakes.qlakei, self.lakes.qlakeo, np.int32(self.nlinksl)))
            _k("ch_commit")(g(self.nlinksl), (_TPB,), (
                typel, self._tmp, qlink1, qlink2, np.int32(self.nlinksl)))
        code = int(self._err.get()[0])
        if code & 4:
            # WRF prints "Musk Cunge WARNING: Failure to converge" and goes on.
            import warnings
            warnings.warn(_ERRORS[4], RuntimeWarning, stacklevel=2)
        if code & 3:
            raise RuntimeError("WRF-Hydro channel routing stopped: "
                               + "; ".join(t for b, t in _ERRORS.items() if code & b & 3))
