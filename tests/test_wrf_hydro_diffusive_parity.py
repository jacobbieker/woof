"""WRF-Hydro diffusive-wave channel routing (the opt-in), bitwise against WRF 4.6.1's hydro/.

drive_CHANNEL's channel_option = 3 branch with DIFFUSION and HEAD, run by
``woof_wrf_hydro.diffusive``.  The fixture is
``optional/wrf_hydro/oracle/build_diffusive.sh``: the routines and the
diffusive block cut byte for byte from the pinned source, eight model steps
over a 20-node network (a main stem ending in a pour point and two
tributaries) with an adverse reach, dry nodes and a storm pulse; the first
step halves DTCT four times on the Courant test.  Every test skips, naming
the opt-in rule, unless ``WOOF_ENABLE_WRF_HYDRO=1``.
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

FIX = optin.optional_root() / "wrf_hydro" / "oracle" / "fixtures" / "diffusive"
IXRT, JXRT, N, DT, DTRT, RETDEP = 14, 10, 20, 300.0, 10.0, 0.001


def _rows(name):
    with (FIX / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def _network():
    rows = _rows("diffusive-network.csv")
    col = lambda k, t=np.float32: np.array([t(float(r[k])) for r in rows])  # noqa: E731
    ch_netrt = np.full((JXRT, IXRT), -9999, np.int32)
    lake = np.full((JXRT, IXRT), -9999, np.int32)
    netlnk = np.full((JXRT, IXRT), -9999, np.int64)
    xi, yj = col("chanxi", int), col("chanyj", int)
    for k in range(N):
        ch_netrt[yj[k] - 1, xi[k] - 1] = 0
        netlnk[yj[k] - 1, xi[k] - 1] = k + 1
    net = dict(ch_netrt=ch_netrt, lake_mskrt=lake, ch_netlnk=netlnk, chanxi=xi, chanyj=yj,
               from_node=col("from_node", int), to_node=col("to_node", int),
               typel=col("typel", int))
    for k in ("node_area", "zelev", "chanlen", "mannn", "so", "chsslp", "bw"):
        net[k] = col(k)
    return net, col("hlink0"), col("cvol0")


def test_fixtures_are_the_ones_the_build_hashed():
    rows = (FIX / "diffusive-sha256sums.txt").read_text().splitlines()
    pinned = {line.split()[1]: line.split()[0] for line in rows}
    for name in ("diffusive-network.csv", "diffusive-forcing.csv", "diffusive-steps.csv"):
        assert pinned[name] == hashlib.sha256((FIX / name).read_bytes()).hexdigest()


def test_lakes_and_gw_exchange_are_refused_by_name():
    mod = optin.load("wrf_hydro")
    net, _, _ = _network()
    with pytest.raises(NotImplementedError, match="GWBASESWCRT = 3"):
        mod.DiffusiveWaveRouting(**net, retdep_chan=RETDEP, dtrt_ch=DTRT, gwbaseswcrt=3)
    lake = dict(net)
    lake["lake_mskrt"] = net["lake_mskrt"].copy()
    lake["lake_mskrt"][net["chanyj"][3] - 1, net["chanxi"][3] - 1] = 1
    with pytest.raises(NotImplementedError, match="level-pool"):
        mod.DiffusiveWaveRouting(**lake, retdep_chan=RETDEP, dtrt_ch=DTRT)


@pytest.mark.gpu
@requires_gpu
def test_diffusive_wave_is_bitwise_against_wrf_hydro():
    import cupy as cp

    mod = optin.load("wrf_hydro")
    net, h0, v0 = _network()
    route = mod.DiffusiveWaveRouting(**net, retdep_chan=RETDEP, dtrt_ch=DTRT)
    forcing, steps = {}, {}
    for r in _rows("diffusive-forcing.csv"):
        forcing.setdefault(int(r["step"]), []).append(r)
    for r in _rows("diffusive-steps.csv"):
        steps.setdefault(int(r["step"]), []).append(r)
    hlink, cvol = cp.asarray(h0), cp.asarray(v0)
    qlink1, qlat = cp.zeros(N, cp.float32), cp.zeros(N, cp.float32)
    for it in sorted(forcing):
        qs = np.zeros((JXRT, IXRT), np.float32)
        qb = np.zeros((JXRT, IXRT), np.float32)
        for r in forcing[it]:
            k = int(r["node"]) - 1
            qs[net["chanyj"][k] - 1, net["chanxi"][k] - 1] = np.float32(r["qstrmvolrt"])
            qb[net["chanyj"][k] - 1, net["chanxi"][k] - 1] = np.float32(r["qinflowbase"])
        route.route(dt=DT, qstrmvolrt=cp.asarray(qs), qinflowbase=cp.asarray(qb), hlink=hlink,
                    cvol=cvol, qlink1=qlink1, qlateral=qlat)
        want = steps[it]
        assert _bits(route.dtct.get())[0] == _bits([np.float32(want[0]["dtct"])])[0], it
        for name, got in (("hlink", hlink), ("cvol", cvol), ("qlink1", qlink1),
                          ("qlateral", qlat)):
            w = np.array([np.float32(r[name]) for r in want])
            bad = _bits(got.get()) != _bits(w)
            assert not bad.any(), (f"step {it} {name}: {int(bad.sum())} differ, first node "
                                   f"{int(np.argmax(bad)) + 1}: {got.get()[bad][0]!r} != {w[bad][0]!r}")
