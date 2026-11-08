"""WRF-Hydro groundwater bucket (the opt-in), bitwise against WRF 4.6.1's hydro/.

``simp_gw_buck`` run by ``woof_wrf_hydro.baseflow``.  The fixture is
``optional/wrf_hydro/oracle/build_baseflow.sh``: the routine cut byte for byte
from the pinned source, four basins, four exponential-bucket calls (one
bucket spilling; surface runoff counted while OVRTSWCRT = 0) and two
pass-through calls, with drainage and runoff values outside [0, 1000] mm.
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

FIX = optin.optional_root() / "wrf_hydro" / "oracle" / "fixtures" / "baseflow"
IX, JX, IXRT, JXRT, NB, DT = 6, 5, 24, 20, 4, 3600.0


def _csv(name):
    with (FIX / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def test_fixtures_are_the_ones_the_build_hashed():
    rows = (FIX / "baseflow-sha256sums.txt").read_text().splitlines()
    pinned = {line.split()[1]: line.split()[0] for line in rows}
    for name in ("baseflow-static.csv", "baseflow-grids.csv", "baseflow-forcing.csv",
                 "baseflow-steps.csv"):
        assert pinned[name] == hashlib.sha256((FIX / name).read_bytes()).hexdigest()


def test_unported_bucket_options_are_refused_by_name():
    mod = optin.load("wrf_hydro")
    for sw, word in ((3, "gw2d"), (4, "simp_gw_buck_nhd")):
        with pytest.raises(NotImplementedError, match=word):
            mod.GroundwaterBucket(gwbaseswcrt=sw, gwsubbasmsk=None, basnsind=None,
                                  gw_strm_msk_lind=None, basns_area=None, c=None, ex=None,
                                  z_mx=None, pixct=None, dist9=None)


@pytest.mark.gpu
@requires_gpu
def test_bucket_is_bitwise_against_wrf_hydro():
    import cupy as cp

    mod = optin.load("wrf_hydro")
    st = _csv("baseflow-static.csv")
    f = lambda k: np.array([np.float32(r[k]) for r in st], np.float32)  # noqa: E731
    grids = _csv("baseflow-grids.csv")
    lsm = np.zeros((JX, IX), np.int64)
    lind = np.zeros((JXRT, IXRT), np.int32)
    dist9 = np.zeros((JXRT, IXRT), np.float32)
    for r in grids:
        i, j = int(r["i"]) - 1, int(r["j"]) - 1
        if r["grid"] == "lsm":
            lsm[j, i] = int(r["mask"])
        else:
            lind[j, i] = int(r["mask"])
            dist9[j, i] = np.float32(r["dist9"])
    pixct = mod.baseflow.pixel_count(lind, NB)
    assert np.array_equal(pixct, f("pixct"))
    d = cp.asarray
    buckets = {sw: mod.GroundwaterBucket(
        gwbaseswcrt=sw, gwsubbasmsk=lsm, basnsind=[int(r["basnsind"]) for r in st],
        gw_strm_msk_lind=lind, basns_area=d(f("area")), c=d(f("c")), ex=d(f("ex")),
        z_mx=d(f("z_mx")), pixct=d(pixct), dist9=d(dist9)) for sw in (1, 2)}
    z = d(f("z0"))
    qin, qout = cp.zeros(NB, cp.float32), cp.zeros(NB, cp.float32)
    qinflowbase = cp.zeros((JXRT, IXRT), cp.float32)
    forcing = _csv("baseflow-forcing.csv")
    lines = (FIX / "baseflow-steps.csv").read_text().splitlines()
    steps = [ln.split(",") for ln in lines[1:] if ln and ln[0].isdigit()]
    qrows = [ln[1:].split(",") for ln in lines if ln.startswith("q")]
    for n in range(1, 7):
        rows = [r for r in forcing if r["call"] == str(n)]
        r1 = np.zeros((JX, IX), np.float32)
        r2 = np.zeros((JX, IX), np.float32)
        for r in rows:
            r1[int(r["j"]) - 1, int(r["i"]) - 1] = np.float32(r["runoff1"])
            r2[int(r["j"]) - 1, int(r["i"]) - 1] = np.float32(r["runoff2"])
        sw, ov = int(rows[0]["sw"]), int(rows[0]["ov"])
        buckets[sw].step(z=z, qin=qin, qout=qout, qinflowbase=qinflowbase, infxsrt=d(r1),
                         soldrain=d(r2), dt=DT, ovrtswcrt=ov)
        want = [s for s in steps if s[0] == str(n)]
        for col, arr in ((2, z), (3, qin), (4, qout)):
            assert np.array_equal(_bits(cp.asnumpy(arr)),
                                  _bits([np.float32(s[col]) for s in want])), (n, col)
        wq = np.zeros((JXRT, IXRT), np.float32)
        for q in qrows:
            if q[0] == str(n):
                wq[int(q[2]) - 1, int(q[1]) - 1] = np.float32(q[3])
        assert np.array_equal(_bits(cp.asnumpy(qinflowbase)), _bits(wq)), (n, "qinflowbase")
