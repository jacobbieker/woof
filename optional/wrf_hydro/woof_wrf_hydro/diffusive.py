# WOOF-OPTIN: wrf-hydro
"""WRF-Hydro's diffusive-wave channel routing on nodes (``diffusive.cu``).

The channel_option = 3 branch of ``drive_CHANNEL``
(hydro/Routing/module_channel_routing.F90 in WRF 4.6.1's hydro/): each call
doubles the routing step DTCT (bounded by 0.01 s and DTRT_CH), then takes
INT(DT/DTCT) sub-steps.  A sub-step forms every regular node's lateral inflow
from its pixel (QSTRMVOLRT + QINFLOWBASE, mm over DT, as m3/s), the
diffusive-wave conveyance across each node face (``DIFFUSION``, with
upstream weighting), the node mass balance, and, unless a node would lose
more than its volume (the Courant test, which restarts the interval with half
the step), the new volumes and depths (``HEAD``'s bisection; a pour point
takes its own depth as the critical depth, as does a node flowing into a
reservoir, TYPEL = 2).  Lakes are level-pool reservoirs
(:mod:`woof_wrf_hydro.levelpool`): their lake nodes' LAKEINFLORT, the face
flows of links into LAKENODE nodes, and an orifice/weir/overtop outflow that
the node draining each lake (CH_NETRT > 0) takes in the next step.  This is
WRF-Hydro's MPP_LAND build (the only one it has) on one process.

Refused by name: groundwater-channel exchange (GWBASESWCRT = 3), reservoir
types other than the level pool (persistence, RFC forecasts), and networks
whose CH_NETLNK numbering is not the node order.  The network is host data,
read once; the state is device float32.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_MODULE = None
_TPB = 256
_ERRORS = {1: "FATAL ERROR: node_area(i) is zero (drive_CHANNEL)",
           2: "Error in Diffusion function ->channel coefficients",
           4: "FATAL ERROR: This node does not have a type (TYPEL)"}


def _k(name):
    global _MODULE
    if _MODULE is None:
        import cupy as cp

        kernels = Path(__import__("woof.core.kernels", fromlist=["x"]).__file__).parent
        here = Path(__file__).resolve().parent
        source = ((kernels / "glibc_flt32.cuh").read_text(encoding="utf-8") + "\n"
                  + (here / "levelpool.cu").read_text(encoding="utf-8") + "\n"
                  + (here / "diffusive.cu").read_text(encoding="utf-8"))
        _MODULE = cp.RawModule(code=source, options=("--fmad=false", "--ftz=false"))
    return _MODULE.get_function(name)


class DiffusiveWaveRouting:
    def __init__(self, *, ch_netrt, lake_mskrt, ch_netlnk, chanxi, chanyj, node_area,
                 from_node, to_node, typel, zelev, chanlen, mannn, so, chsslp, bw,
                 retdep_chan, dtrt_ch, dtct=None, gwbaseswcrt=0, lakenode=None, lakes=None,
                 reservoir_type="level_pool"):
        import cupy as cp

        if gwbaseswcrt == 3:
            raise NotImplementedError("GWBASESWCRT = 3 (groundwater-channel exchange in the "
                                      "diffusive wave) is not ported")
        ch_netrt, lake_mskrt, ch_netlnk = (np.asarray(a) for a in (ch_netrt, lake_mskrt,
                                                                    ch_netlnk))
        chanxi, chanyj, typel = (np.asarray(a, np.int64) for a in (chanxi, chanyj, typel))
        from_node, to_node = np.asarray(from_node, np.int64), np.asarray(to_node, np.int64)
        n = len(chanxi)
        jxrt, ixrt = ch_netrt.shape
        j, i = chanyj - 1, chanxi - 1
        if reservoir_type != "level_pool":
            raise NotImplementedError(f"reservoir type {reservoir_type!r} (persistence, RFC "
                                      "forecasts) is not ported; the level pool is")
        from .levelpool import LevelPool

        self.lakes = lakes if lakes is not None else LevelPool.empty()
        nlakes = self.lakes.n
        lakenode = np.zeros(n, np.int64) if lakenode is None else np.asarray(lakenode, np.int64)
        lake_at = lake_mskrt[j, i]
        lake_node = (lake_at > 0) & (ch_netrt[j, i] <= 0)
        pour = np.where(ch_netrt[j, i] > 0, ch_netrt[j, i], 0)
        if (lake_at[lake_node] > nlakes).any() or (pour > nlakes).any() or (lakenode > nlakes).any():
            raise ValueError("a lake number exceeds the number of level-pool lakes")
        if (ch_netlnk[j, i] != np.arange(1, n + 1)).any():
            raise NotImplementedError("CH_NETLNK must number the nodes in node order")
        if ((typel == 0) & (to_node <= 0)).any() or not np.isin(typel, (0, 1, 2)).all():
            raise ValueError("TYPEL 0 nodes need a TO_NODE; TYPEL is 0, 1 or 2")
        if ((from_node < 1) | (from_node > n) | (to_node < 0) | (to_node > n)).any():
            raise ValueError("FROM_NODE/TO_NODE out of range")
        regular = (ch_netrt[j, i] == 0) & (lake_mskrt[j, i] < 0)
        tl = [[] for _ in range(n)]
        fl = [[] for _ in range(n)]
        for k in range(n):
            if typel[k] == 0:
                tl[to_node[k] - 1].append(k)
            fl[from_node[k] - 1].append(k)
        ln = [[k for k in range(n) if lake_node[k] and lake_at[k] == m + 1] for m in range(nlakes)]
        li = [[k for k in range(n) if to_node[k] > 0 and lakenode[to_node[k] - 1] == m + 1]
              for m in range(nlakes)]
        ts = np.cumsum([0] + [len(x) for x in tl]).astype(np.int32)
        fs = np.cumsum([0] + [len(x) for x in fl]).astype(np.int32)
        d = lambda a, t=np.int32: cp.asarray(np.ascontiguousarray(a, t))  # noqa: E731
        self.n = n
        self._net = dict(
            cell=d(j * ixrt + i), regular=d(regular), typel=d(typel), from_node=d(from_node),
            to_node=d(to_node), tstart=d(ts), tlist=d([k for x in tl for k in x] or [0]),
            fstart=d(fs), flist=d([k for x in fl for k in x] or [0]), pour=d(pour),
            lnstart=d(np.cumsum([0] + [len(x) for x in ln])),
            lnlist=d([k for x in ln for k in x] or [0]),
            listart=d(np.cumsum([0] + [len(x) for x in li])),
            lilist=d([k for x in li for k in x] or [0]))
        self._lakework = [cp.zeros(max(nlakes, 1), cp.float32) for _ in range(2)]
        self._props = {k: d(v, np.float32) for k, v in dict(
            node_area=node_area, zelev=zelev, chanlen=chanlen, mannn=mannn, so=so,
            chsslp=chsslp, bw=bw).items()}
        self.retdep_chan, self.dtrt_ch = np.float32(retdep_chan), np.float32(dtrt_ch)
        # module_namelist.F90:403: DTCT starts at DTRT_CH
        self.dtct = cp.asarray(np.array([dtrt_ch if dtct is None else dtct], np.float32))
        self._work = [cp.zeros(n, cp.float32) for _ in range(4)]
        self._err = cp.zeros(1, cp.int32)

    def route(self, *, dt, qstrmvolrt, qinflowbase, hlink, cvol, qlink1, qlateral,
              lakeinflort=None) -> None:
        """One drive_CHANNEL call over the model step ``dt``: updates hlink,
        cvol, qlink1 (QLINK(:,1)), qlateral, the routing step ``dtct`` and the
        lakes' state (``self.lakes``: resht, qlakei, qlakeo)."""
        import cupy as cp

        if lakeinflort is None:
            lakeinflort = cp.zeros_like(qstrmvolrt)
        for name, a in (("qstrmvolrt", qstrmvolrt), ("qinflowbase", qinflowbase),
                        ("lakeinflort", lakeinflort)):
            if a.dtype != cp.float32 or not a.flags.c_contiguous:
                raise ValueError(f"{name}: expected contiguous float32 (jxrt, ixrt)")
        for name, a in (("hlink", hlink), ("cvol", cvol), ("qlink1", qlink1),
                        ("qlateral", qlateral)):
            if a.shape != (self.n,) or a.dtype != cp.float32:
                raise ValueError(f"{name}: expected float32 ({self.n},)")
        g, p, L = self._net, self._props, self.lakes
        self._err[...] = 0
        _k("wrfhydro_diffusive")((1,), (_TPB,), (
            g["cell"], qstrmvolrt, qinflowbase, p["node_area"], g["from_node"], g["to_node"],
            g["typel"], p["zelev"], p["chanlen"], p["mannn"], p["so"], p["chsslp"], p["bw"],
            g["regular"], g["tstart"], g["tlist"], g["fstart"], g["flist"], hlink, cvol, qlink1,
            qlateral, *self._work, self.dtct, self._err, np.float32(dt), self.dtrt_ch,
            self.retdep_chan, np.int32(self.n), g["pour"], lakeinflort, g["lnstart"],
            g["lnlist"], g["listart"], g["lilist"], L.props, L.resht, L.qlakei, L.qlakeo,
            *self._lakework, np.int32(L.n)))
        code = int(self._err.get()[0])
        if code:
            raise RuntimeError("; ".join(m for b, m in _ERRORS.items() if code & b))
