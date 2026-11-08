# WOOF-OPTIN: wrf-hydro
"""WRF-Hydro's groundwater bucket on device (``baseflow.cu``).

``simp_gw_buck`` (hydro/Routing/module_GW_baseflow.F90 in WRF 4.7.1's hydro/):
each basin's bucket takes the mean LSM drainage (SOLDRAIN; plus INFXSRT when
overland routing is off), discharges by the exponential bucket
(GWBASESWCRT = 1, with overflow spill) or passes it through
(GWBASESWCRT = 2), and the outflow is spread over the basin's stream pixels
as QINFLOWBASE for the channel.  GWBASESWCRT = 3 is the 2-D groundwater
model (:class:`woof_wrf_hydro.gw2d.Gw2d`); 4 (the area-normalised bucket,
simp_gw_buck_nhd, UDMP) is refused by name.

Basin masks are host data, read once: ``gwsubbasmsk`` (jx, ix) of basin ids,
``basnsind`` (the ids, in bucket order), ``gw_strm_msk_lind`` (jxrt, ixrt) of
1-based bucket indices on stream pixels.  ``pixel_count`` reproduces
module_RT.F90's non-MPP stream-pixel count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_TPB = 128
_MODULE = None


def _k(name):
    global _MODULE
    if _MODULE is None:
        import cupy as cp

        kernels = Path(__import__("woof.core.kernels", fromlist=["x"]).__file__).parent
        here = Path(__file__).resolve().parent
        source = ((kernels / "glibc_flt64.cuh").read_text(encoding="utf-8") + "\n"
                  + (here / "baseflow.cu").read_text(encoding="utf-8"))
        _MODULE = cp.RawModule(code=source, options=("--fmad=false", "--ftz=false"))
    return _MODULE.get_function(name)


def pixel_count(gw_strm_msk, nb) -> np.ndarray:
    """module_RT.F90:1369-1380: REAL count of stream pixels per bucket."""
    counts = np.zeros(nb, np.float32)
    for b in np.asarray(gw_strm_msk).ravel():
        if b > 0:
            counts[b - 1] = np.float32(counts[b - 1] + np.float32(1.0))
    return counts


class GroundwaterBucket:
    def __init__(self, *, gwbaseswcrt, gwsubbasmsk, basnsind, gw_strm_msk_lind, basns_area,
                 c, ex, z_mx, pixct, dist9):
        import cupy as cp

        if gwbaseswcrt == 3:
            raise ValueError("GWBASESWCRT = 3 is the 2-D groundwater model, not a bucket: "
                             "use Gw2d")
        if gwbaseswcrt == 4:
            raise NotImplementedError("GWBASESWCRT = 4 (the area-normalised exponential "
                                      "bucket of simp_gw_buck_nhd, UDMP) is not ported")
        if gwbaseswcrt not in (1, 2):
            raise ValueError("the WRF-Hydro bucket is GWBASESWCRT 1 or 2 here")
        self.sw = int(gwbaseswcrt)
        jx, ix = gwsubbasmsk.shape
        lists = []
        for bid in basnsind:   # WRF's do i (outer), do j (inner)
            lists.append([j * ix + i for i in range(ix) for j in range(jx)
                          if gwsubbasmsk[j, i] == bid])
        d = cp.asarray
        self._start = d(np.cumsum([0] + [len(x) for x in lists]).astype(np.int32))
        self._cells = d(np.asarray([c for x in lists for c in x] or [0], np.int32))
        self.nb = len(basnsind)
        self.lind = d(np.ascontiguousarray(gw_strm_msk_lind, np.int32))
        self.area, self.c, self.ex, self.z_mx, self.pixct, self.dist9 = (
            basns_area, c, ex, z_mx, pixct, dist9)

    def step(self, *, z, qin, qout, qinflowbase, infxsrt, soldrain, dt, ovrtswcrt) -> None:
        """One simp_gw_buck call: z (REAL bucket levels, mm), qin, qout (m3/s)
        per bucket and qinflowbase (mm, routing grid) are written."""
        _k("gw_bucket")(((self.nb + _TPB - 1) // _TPB,), (_TPB,), (
            self._start, self._cells, infxsrt, soldrain, self.area, self.c, self.ex,
            self.z_mx, z, qin, qout, np.float32(dt), np.int32(self.sw),
            np.int32(ovrtswcrt), np.int32(self.nb)))
        nrt = int(self.lind.size)
        _k("gw_inflowbase")(((nrt + _TPB - 1) // _TPB,), (_TPB,), (
            self.lind, qout, self.pixct, self.dist9, qinflowbase, np.float32(dt), np.int32(nrt)))
