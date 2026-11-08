"""NEMO FCT tracer advection (the opt-in), bitwise against NEMO 5.0.2.

tra_adv_fct run by ``woof_nemo.traadv``.  The fixture is
``optional/nemo/oracle/build_traadv.sh``: traadv_fct.F90 compiled whole and
unchanged against NEMO-named service modules (lbc_lnk filling the closed
halo with zero, as NEMO does on one process), three calls for each of the
four (nn_fct_h, nn_fct_v) pairs over a 10 x 9 x 15 grid with NEMO 5.0's
two-cell halo, a land column, a shelf and fronts the limiter acts on.
Every test skips, naming the opt-in rule, unless ``WOOF_ENABLE_NEMO=1``.
"""

from __future__ import annotations

import csv
import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from woof import optin

REASON = optin.unavailable_reason("nemo")
pytestmark = pytest.mark.skipif(REASON is not None, reason=f"opt-in rule: {REASON}")

FIX = optin.optional_root() / "nemo" / "oracle" / "fixtures" / "traadv"
NX, NY, NZ = 10, 9, 15
PAIRS = [(2, 2), (2, 4), (4, 2), (4, 4)]


def _rows(name):
    with (FIX / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _grid():
    tmask, e3t = np.zeros((NZ, NY, NX)), np.zeros((NZ, NY, NX))
    e1e2t = np.zeros((NY, NX))
    mbkt = np.zeros((NY, NX), np.int32)
    for r in _rows("traadv-grid.csv"):
        i, j, k = int(r["ji"]) - 1, int(r["jj"]) - 1, int(r["jk"]) - 1
        tmask[k, j, i], e3t[k, j, i] = float(r["tmask"]), float(r["e3t"])
        e1e2t[j, i] = float(r["e1e2t"])
        mbkt[j, i] = max(int(r["mbkt"]), 1)
    umask = tmask * np.concatenate([tmask[:, :, 1:], tmask[:, :, -1:]], axis=2)
    vmask = tmask * np.concatenate([tmask[:, 1:, :], tmask[:, -1:, :]], axis=1)
    wmask = tmask.copy()
    wmask[1:] = tmask[1:] * tmask[:-1]
    return dict(tmask=tmask, umask=umask, vmask=vmask, wmask=wmask, e3t=e3t, e1e2t=e1e2t,
                r1_e1e2t=1.0 / e1e2t, mbkt=mbkt, mikt=np.ones((NY, NX), np.int32))


def _calls():
    keys = ("pu", "pv", "pw", "tb", "sb", "tn", "sn", "trhs0", "srhs0", "trhs", "srhs")
    out = {}
    for r in _rows("traadv-calls.csv"):
        key = (int(r["kn_fct_h"]), int(r["kn_fct_v"]), int(r["call"]))
        d = out.setdefault(key, {k: np.zeros((NZ, NY, NX)) for k in keys})
        i, j, k = int(r["ji"]) - 1, int(r["jj"]) - 1, int(r["jk"]) - 1
        for name in keys:
            d[name][k, j, i] = float(r[name])
    return out


def test_fixtures_are_the_ones_the_build_hashed():
    rows = (FIX / "traadv-sha256sums.txt").read_text().splitlines()
    pinned = {line.split()[1]: line.split()[0] for line in rows}
    for name in ("traadv-grid.csv", "traadv-calls.csv"):
        assert pinned[name] == hashlib.sha256((FIX / name).read_bytes()).hexdigest()


@pytest.mark.parametrize("kw,word", [(dict(ln_zad_aimp=True), "ln_zad_Aimp"),
                                     (dict(ln_isfcav=True), "ln_isfcav"),
                                     (dict(trend_diagnostics=True), "l_trdtra"),
                                     (dict(tiling=True), "ln_tile"),
                                     (dict(rk3=True), "key_RK3")])
def test_unported_fct_options_are_refused_by_name(kw, word):
    mod = optin.load("nemo")
    with pytest.raises(NotImplementedError, match=word):
        mod.TracerAdvectionFct(**kw)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("pair", PAIRS)
def test_tra_adv_fct_is_bitwise_against_nemo(pair):
    import cupy as cp

    mod = optin.load("nemo")
    g = {k: cp.asarray(v) for k, v in _grid().items()}
    calls = _calls()
    adv = mod.TracerAdvectionFct(nn_fct_h=pair[0], nn_fct_v=pair[1])
    for ic in (1, 2, 3):
        c = calls[(*pair, ic)]
        for tr in ("t", "s"):
            rhs = cp.asarray(c[f"{tr}rhs0"])
            adv.step(p2dt=1800.0, pU=cp.asarray(c["pu"]), pV=cp.asarray(c["pv"]),
                     pW=cp.asarray(c["pw"]), tb=cp.asarray(c[f"{tr}b"]),
                     tn=cp.asarray(c[f"{tr}n"]), rhs=rhs, **g)
            got = rhs.get()[:, 2:-2, 2:-2]
            want = c[f"{tr}rhs"][:, 2:-2, 2:-2]
            bad = got.view(np.int64) != want.view(np.int64)
            assert not bad.any(), (f"{pair} call {ic} {tr}: {int(bad.sum())} words differ, first "
                                   f"{np.argwhere(bad)[0].tolist()}: {got[bad][0]!r} != "
                                   f"{want[bad][0]!r}")
