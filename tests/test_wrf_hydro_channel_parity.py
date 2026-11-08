"""WRF-Hydro reach channel routing (the opt-in), bitwise against WRF 4.7.1's hydro/.

drive_CHANNEL's reach branch with MUSKING and SUBMUSKINGCUNGE, run by
``woof_wrf_hydro.channel``.  The fixture is
``optional/wrf_hydro/oracle/build_channel.sh``: both routines and the reach
block cut byte for byte from the pinned source, a 12-reach network fed by 60
pixels (one in a lake), four Muskingum-Cunge calls then two Muskingum calls,
with a compound channel that floods, Bw = Tw, Bw > Tw, Cs = 0, channel loss,
two reservoir reaches routing a level pool and a pass-through lake, and a
TYPEL = 2 reach the commit skips.  Every test skips, naming the opt-in rule, unless
``WOOF_ENABLE_WRF_HYDRO=1``.
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

FIX = optin.optional_root() / "wrf_hydro" / "oracle" / "fixtures" / "channel"
IXRT, JXRT, NLINKS, NLINKSL, DT, DTRT = 20, 16, 60, 12, 300.0, 60.0


def _rows(name):
    with (FIX / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def test_fixtures_are_the_ones_the_build_hashed():
    rows = (FIX / "channel-sha256sums.txt").read_text().splitlines()
    pinned = {line.split()[1]: line.split()[0] for line in rows}
    for name in ("channel-network.csv", "channel-pixels.csv", "channel-forcing.csv",
                 "channel-steps.csv", "channel-lakes.csv"):
        assert pinned[name] == hashlib.sha256((FIX / name).read_bytes()).hexdigest()


def test_diffusive_channel_option_is_refused_by_name():
    mod = optin.load("wrf_hydro")
    with pytest.raises(NotImplementedError, match="channel_option = 3"):
        mod.ReachRouting(channel_option=3, **{k: None for k in (
            "ch_lnkrt_sl", "lake_mskrt", "chanxi", "chanyj", "node_area", "linkid", "to_node",
            "nlinksl", "typel", "musk", "musx", "so", "chanlen", "mannn", "chsslp", "bw", "tw",
            "tw_cc", "n_cc", "channk")})


@pytest.mark.gpu
@requires_gpu
def test_reach_routing_is_bitwise_against_wrf_hydro():
    import cupy as cp

    mod = optin.load("wrf_hydro")
    net, pix = _rows("channel-network.csv"), _rows("channel-pixels.csv")
    f32 = lambda key: np.array([np.float32(r[key]) for r in net] + [0.0] * (NLINKS - NLINKSL), np.float32)  # noqa: E731
    chanxi = np.array([int(r["chanxi"]) for r in pix])
    chanyj = np.array([int(r["chanyj"]) for r in pix])
    ch = np.zeros((JXRT, IXRT), np.int64)
    lake = np.full((JXRT, IXRT), -9999, np.int64)
    for r in pix:
        j, i = int(r["chanyj"]) - 1, int(r["chanxi"]) - 1
        ch[j, i] = int(r["link"])
        lake[j, i] = int(r["lake"])
    linkid = [int(r["linkid"]) for r in net]
    to_node = [int(r["to_node"]) for r in net]
    typel = np.array([int(r["typel"]) for r in net] + [0] * (NLINKS - NLINKSL), np.int32)
    d = cp.asarray
    steps, forcing = _rows("channel-steps.csv"), _rows("channel-forcing.csv")
    state = {k: cp.zeros(NLINKS, cp.float32) for k in ("qlink1", "qlink2", "qlateral", "velocity", "qloss")}
    state["hlink"] = d(f32("hlink0"))
    # run_channel.F90 starts QLINK(k,:) = 0.5*frac(k,41); read back from no row,
    # so rebuild it with the same integer hash.
    q0 = np.array([np.float32(0.5) * np.float32(np.float32((k * 41 * 7919) % 1009) / np.float32(1009.0))
                   for k in range(1, NLINKSL + 1)] + [0.0] * (NLINKS - NLINKSL), np.float32)
    state["qlink1"][:] = d(q0)
    state["qlink2"][:] = d(q0)
    lakes_rows = _rows("channel-lakes.csv")
    first = [r for r in lakes_rows if r["call"] == "0"]
    lf = lambda key: np.array([np.float32(r[key]) for r in first])  # noqa: E731
    pool = mod.LevelPool(water_elevation=lf("resht"), lake_area=lf("lake_area"),
                         weir_elevation=lf("weir_elev"), weir_coefficient=lf("weir_coef"),
                         weir_length=lf("weir_len"), dam_length=lf("dam_len"),
                         orifice_elevation=lf("orifice_elev"),
                         orifice_coefficient=lf("orifice_coef"), orifice_area=lf("orifice_area"),
                         max_depth=lf("max_depth"),
                         lake_opt=np.array([int(r["lake_opt"]) for r in first]))
    lake_lookup = np.full(NLINKS, -1, np.int64)
    for r in first:
        lake_lookup[int(r["k"]) - 1] = int(r["lake"]) - 1
    assert sorted(int(r["lake_opt"]) for r in first) == [1, 2]
    assert sorted(typel[:NLINKSL].tolist()).count(1) == 2 and 2 in typel
    routers = {}
    for n in range(1, 7):
        opt = 2 if n <= 4 else 1
        if opt not in routers:
            routers[opt] = mod.ReachRouting(
                channel_option=opt, ch_lnkrt_sl=ch, lake_mskrt=lake, chanxi=chanxi,
                chanyj=chanyj, node_area=d(np.array([np.float32(r["node_area"]) for r in pix])),
                linkid=linkid, to_node=to_node, nlinksl=NLINKSL, typel=d(typel),
                lakes=pool, lake_lookup=lake_lookup,
                **{k: d(f32(k)) for k in ("musk", "musx", "so", "chanlen", "mannn", "chsslp",
                                          "bw", "tw", "tw_cc", "n_cc", "channk")})
        planes = {k: np.zeros((JXRT, IXRT), np.float32) for k in ("qstrmvol", "qinflowbase", "lakeinflo")}
        for r in forcing:
            if r["call"] == str(n):
                jj = int(r["jj"]) - 1
                for k in planes:
                    planes[k][chanyj[jj] - 1, chanxi[jj] - 1] = np.float32(r[k])
        routers[opt].route(qstrmvolrt=d(planes["qstrmvol"]), qinflowbase=d(planes["qinflowbase"]),
                           lakeinflort=d(planes["lakeinflo"]), dt=DT, dtrt_ch=DTRT, **state)
        want = [r for r in steps if r["call"] == str(n)]
        for key, col in (("qlink1", "qlink1"), ("qlink2", "qlink2"), ("hlink", "hlink"),
                         ("qlateral", "qlateral"), ("velocity", "velocity"), ("qloss", "qloss")):
            got = cp.asnumpy(state[key])[:NLINKSL]
            assert np.array_equal(_bits(got), _bits([np.float32(r[col]) for r in want])), (n, key)
        lk = [r for r in lakes_rows if r["call"] == str(n)]
        for key, col in (("resht", "resht"), ("qlakei", "qlakei"), ("qlakeo", "qlakeo")):
            got = cp.asnumpy(getattr(pool, key))
            assert np.array_equal(_bits(got), _bits([np.float32(r[col]) for r in lk])), (n, key)
