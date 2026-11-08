"""WRF-Hydro LSM <-> routing grid transforms (the opt-in), bitwise against WRF 4.7.1.

``disaggregateDomain`` and ``aggregateDomain`` as WRF 4.7.1's hydro/ ships
them, run by ``woof_wrf_hydro.grid_transform``.  The fixture is
``optional/wrf_hydro/oracle/build_grid_transform.sh``: both routines cut byte
for byte from the pinned source, a 6 x 5 LSM grid with AGGFACTRT = 4, frozen
soil, a layer frozen solid, over-filling weights, a lake cell and ocean cells.
Every test skips, naming the opt-in rule, unless ``WOOF_ENABLE_WRF_HYDRO=1``.
"""

from __future__ import annotations

import csv
import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from woof import optin

REASON = optin.unavailable_reason("wrf_hydro")
pytestmark = pytest.mark.skipif(REASON is not None, reason=f"opt-in rule: {REASON}")

FIX = optin.optional_root() / "wrf_hydro" / "oracle" / "fixtures" / "grid_transform"
IX, JX, AGG, NSOIL, ISWATER = 6, 5, 4, 4, 17
IXRT, JXRT = IX * AGG, JX * AGG
SLDPTH = (0.1, 0.3, 0.6, 1.0)


def _rows(name):
    with (FIX / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _plane(rows, key, shape, dtype=np.float32):
    a = np.zeros(shape, dtype)
    for r in rows:
        v = r[key]
        a[int(r["j"]) - 1, int(r["i"]) - 1] = np.float32(v) if dtype is np.float32 else int(v)
    return a


def _soil(rows, prefix, shape):
    return np.ascontiguousarray(np.stack([_plane(rows, f"{prefix}{k}", shape) for k in range(1, NSOIL + 1)]))


def _same(got, want, what):
    import cupy as cp

    g = cp.asnumpy(got)
    if g.dtype == np.float32:
        assert np.array_equal(g.view(np.uint32), np.ascontiguousarray(want, np.float32).view(np.uint32)), what
    else:
        assert np.array_equal(g, want), what


def test_fixtures_are_the_ones_the_build_hashed():
    rows = (FIX / "grid-transform-sha256sums.txt").read_text().splitlines()
    pinned = {line.split()[1]: line.split()[0] for line in rows}
    for name in ("disagg-lsm.csv", "disagg-rt.csv", "agg-rt.csv", "agg-lsm.csv"):
        assert pinned[name] == hashlib.sha256((FIX / name).read_bytes()).hexdigest()


@pytest.mark.gpu
@requires_gpu
def test_disaggregate_then_aggregate_are_bitwise_against_wrf_hydro():
    import cupy as cp

    gt = optin.load("wrf_hydro").grid_transform
    lsm, rt = _rows("disagg-lsm.csv"), _rows("disagg-rt.csv")
    L, R = (JX, IX), (JXRT, IXRT)
    d = cp.asarray
    sice = cp.zeros((NSOIL, *L), cp.float32)
    lake = d(_plane(rt, "lake_in", R, np.int32))
    out = {k: cp.zeros((NSOIL, *R), cp.float32) for k in ("smcrefrt", "smcmaxrt", "smcwltrt", "smcrt")}
    out.update({k: cp.zeros(R, cp.float32) for k in ("infxsubrt", "lksatrt", "nexprt")})
    out["soiltyprt"] = cp.zeros(R, cp.int32)
    gt.disaggregate(
        agg=AGG, iswater=ISWATER, sldpth=SLDPTH, sice=sice,
        smc=d(_soil(lsm, "smc", L)), sh2ox=d(_soil(lsm, "sh2o", L)),
        infxsrt=d(_plane(lsm, "infxsrt", L)), area_lsm=d(_plane(lsm, "area", L)),
        smcmax1=d(_plane(lsm, "smcmax", L)), smcref1=d(_plane(lsm, "smcref", L)),
        smcwlt1=d(_plane(lsm, "smcwlt", L)), vegtyp=d(_plane(lsm, "vegtyp", L, np.int32)),
        lksat=d(_plane(lsm, "lksat", L)), nexp=d(_plane(lsm, "nexp", L)),
        dist9=d(_plane(rt, "dist9", R)), infxswgt=d(_plane(rt, "infxswgt", R)),
        lksatfac=d(_plane(rt, "lksatfac", R)), ch_netrt=d(_plane(rt, "ch_netrt", R, np.int32)),
        sh2owgt=d(_soil(rt, "w", R)), elrt=d(_plane(rt, "elrt", R)),
        soiltyp=d(_plane(lsm, "soiltyp", L, np.int32)), lake_mskrt=lake, out=out)
    _same(sice, _soil(lsm, "sice", L), "sice")
    _same(lake, _plane(rt, "lake", R, np.int32), "lake_mskrt")
    _same(out["soiltyprt"], _plane(rt, "soiltyprt", R, np.int32), "soiltyprt")
    for key in ("infxsubrt", "lksatrt", "nexprt"):
        _same(out[key], _plane(rt, key, R), key)
    for key, prefix in (("smcmaxrt", "smcmax"), ("smcrefrt", "smcref"),
                        ("smcwltrt", "smcwlt"), ("smcrt", "smc")):
        _same(out[key], _soil(rt, prefix, R), key)

    art, alsm = _rows("agg-rt.csv"), _rows("agg-lsm.csv")
    head_lsm = cp.zeros(L, cp.float32)
    sh2ox = d(_soil(lsm, "sh2o", L))
    smc = d(_soil(lsm, "smc", L))
    infxswgt = d(_plane(rt, "infxswgt", R))
    sh2owgt = d(_soil(rt, "w", R))
    gt.aggregate(agg=AGG, head_rt=d(_plane(art, "head", R)), dist9=d(_plane(rt, "dist9", R)),
                 smcrt=d(_soil(art, "smc", R)), smcmaxrt=out["smcmaxrt"], sice=sice,
                 head_lsm=head_lsm, sh2ox=sh2ox, smc=smc, infxswgt=infxswgt, sh2owgt=sh2owgt)
    _same(head_lsm, _plane(alsm, "head_lsm", L), "head_lsm")
    _same(sh2ox, _soil(alsm, "sh2o", L), "sh2ox")
    _same(smc, _soil(alsm, "smc", L), "smc")
    _same(infxswgt, _plane(art, "infxswgt", R), "infxswgt")
    _same(sh2owgt, _soil(art, "w", R), "sh2owgt")
