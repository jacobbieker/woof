# WOOF-OPTIN: wrf-hydro
"""WRF-Hydro's LSM <-> routing grid transforms on device (``grid_transform.cu``).

:func:`disaggregate` is ``disaggregateDomain`` (hydro/Routing/Noah_distr_routing.F90):
the LSM cell's infiltration excess, soil moisture and soil properties onto its
AGGFACTRT x AGGFACTRT routing cells, by the weights the last aggregation left,
with frozen-soil adjustment and excess water pushed up the profile.
:func:`aggregate` is ``aggregateDomain`` (hydro/HYDRO_drv/module_HYDRO_drv.F90):
the routed head and soil moisture back onto the LSM cell, and the new weights.
WRF-Hydro's fatal stops raise ``RuntimeError`` naming them after one 4-byte
read.  Arrays: LSM planes ``(jx, ix)``, routing planes ``(jxrt, ixrt)``,
soil fields ``(nsoil, ...)``; float32, integers int32.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_TPB = 64
_MODULE = None
_ERRORS = {1: "SMCMAX exceeded upon disaggregation", 2: "SMCRT depleted upon disaggregation",
           4: "SMCMAX exceeded upon aggregation", 8: "negative SH2OX upon aggregation"}


def _k(name):
    global _MODULE
    if _MODULE is None:
        import cupy as cp

        source = (Path(__file__).resolve().parent / "grid_transform.cu").read_text(encoding="utf-8")
        _MODULE = cp.RawModule(code=source, options=("--fmad=false", "--ftz=false"))
    return _MODULE.get_function(name)


def _check(err):
    code = int(err.get()[0])
    if code:
        raise RuntimeError("WRF-Hydro grid transform stopped: "
                           + "; ".join(t for b, t in _ERRORS.items() if code & b))


def disaggregate(*, agg, iswater, sldpth, sice, smc, sh2ox, infxsrt, area_lsm,
                 smcmax1, smcref1, smcwlt1, vegtyp, lksat, nexp, dist9, infxswgt,
                 lksatfac, ch_netrt, sh2owgt, elrt, soiltyp, lake_mskrt, out) -> None:
    """``disaggregateDomain``.  ``sice`` is written (and, for a layer frozen
    solid, rewritten as WRF does); ``lake_mskrt`` is updated in place;
    ``out`` maps smcrefrt, infxsubrt, smcmaxrt, smcwltrt, smcrt, lksatrt,
    nexprt and soiltyprt to preallocated device arrays."""
    import cupy as cp

    nsoil, jx, ix = map(int, smc.shape)
    if nsoil > 16:
        raise ValueError("the port carries at most 16 soil layers")
    err = cp.zeros((1,), cp.int32)
    sl = cp.asarray(np.asarray(sldpth, np.float32))
    _k("hyd_disaggregate")(((ix * jx + _TPB - 1) // _TPB,), (_TPB,), (
        np.int32(ix), np.int32(jx), np.int32(nsoil), np.int32(agg), np.int32(iswater),
        sice, smc, sh2ox, infxsrt, area_lsm, smcmax1, smcref1, smcwlt1, vegtyp, lksat,
        nexp, dist9, infxswgt, lksatfac, ch_netrt, sh2owgt, out["smcrefrt"],
        out["infxsubrt"], out["smcmaxrt"], out["smcwltrt"], out["smcrt"], lake_mskrt,
        out["lksatrt"], out["nexprt"], sl, out["soiltyprt"], soiltyp, err))
    nrt = ix * jx * agg * agg
    _k("hyd_disaggregate_ocean")(((nrt + 255) // 256,), (256,), (
        elrt, out["infxsubrt"], np.int32(nrt)))
    _check(err)


def aggregate(*, agg, head_rt, dist9, smcrt, smcmaxrt, sice, head_lsm, sh2ox, smc,
              infxswgt, sh2owgt) -> None:
    """``aggregateDomain``: writes head_lsm, sh2ox, smc, infxswgt, sh2owgt."""
    import cupy as cp

    nsoil, jx, ix = map(int, sh2ox.shape)
    if nsoil > 16:
        raise ValueError("the port carries at most 16 soil layers")
    err = cp.zeros((1,), cp.int32)
    _k("hyd_aggregate")(((ix * jx + _TPB - 1) // _TPB,), (_TPB,), (
        np.int32(ix), np.int32(jx), np.int32(nsoil), np.int32(agg), head_rt, dist9,
        smcrt, smcmaxrt, head_lsm, sh2ox, sice, smc, infxswgt, sh2owgt, err))
    _check(err)
