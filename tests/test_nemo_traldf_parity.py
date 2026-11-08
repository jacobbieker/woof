"""NEMO iso-level lateral tracer diffusion (the opt-in), bitwise against NEMO 5.0.2.

traldf_lev_lap and traldf_lev_blp run by ``woof_nemo.traldf``.  The fixture
is ``optional/nemo/oracle/build_traldf.sh``: traldf_lev.F90 compiled whole
and unchanged against NEMO-named service modules, three calls per operator
over a 10 x 9 x 15 grid with NEMO 5.0's two-cell halo, a land column, a
shelf and varying scale factors.  Every test skips, naming the opt-in rule,
unless ``WOOF_ENABLE_NEMO=1``.
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

FIX = optin.optional_root() / "nemo" / "oracle" / "fixtures" / "traldf"
NX, NY, NZ = 10, 9, 15


def _rows(name):
    with (FIX / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _grid():
    g3 = {k: np.zeros((NZ, NY, NX)) for k in ("e3t", "e3u", "e3v")}
    g2 = {k: np.zeros((NY, NX)) for k in ("e2_e1u", "e1_e2v", "r1_e1e2t")}
    for r in _rows("traldf-grid.csv"):
        i, j, k = int(r["ji"]) - 1, int(r["jj"]) - 1, int(r["jk"]) - 1
        for name in g3:
            g3[name][k, j, i] = float(r[name])
        for name in g2:
            g2[name][j, i] = float(r[name])
    return {**g3, **g2}


def _calls():
    keys = ("ahtu", "ahtv", "tb", "sb", "trhs0", "srhs0", "trhs", "srhs")
    out = {}
    for r in _rows("traldf-calls.csv"):
        d = out.setdefault((r["op"], int(r["call"])), {k: np.zeros((NZ, NY, NX)) for k in keys})
        i, j, k = int(r["ji"]) - 1, int(r["jj"]) - 1, int(r["jk"]) - 1
        for name in keys:
            d[name][k, j, i] = float(r[name])
    return out


def test_fixtures_are_the_ones_the_build_hashed():
    rows = (FIX / "traldf-sha256sums.txt").read_text().splitlines()
    pinned = {line.split()[1]: line.split()[0] for line in rows}
    for name in ("traldf-grid.csv", "traldf-calls.csv"):
        assert pinned[name] == hashlib.sha256((FIX / name).read_bytes()).hexdigest()


@pytest.mark.parametrize("kw,word", [(dict(direction="iso"), "ln_traldf_iso"),
                                     (dict(direction="triad"), "ln_traldf_triad"),
                                     (dict(transport_diagnostics=True), "l_diaptr")])
def test_unported_traldf_options_are_refused_by_name(kw, word):
    mod = optin.load("nemo")
    with pytest.raises(NotImplementedError, match=word):
        mod.LateralDiffusionLevel(**kw)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("op", ["lap", "blp"])
def test_traldf_lev_is_bitwise_against_nemo(op):
    import cupy as cp

    mod = optin.load("nemo")
    g = {k: cp.asarray(v) for k, v in _grid().items()}
    calls = _calls()
    ldf = mod.LateralDiffusionLevel(operator=op)
    for ic in (1, 2, 3):
        c = calls[(op, ic)]
        for tr in ("t", "s"):
            rhs = cp.asarray(c[f"{tr}rhs0"])
            ldf.step(ahtu=cp.asarray(c["ahtu"]), ahtv=cp.asarray(c["ahtv"]),
                     tb=cp.asarray(c[f"{tr}b"]), rhs=rhs, **g)
            got = rhs.get()[:, 2:-2, 2:-2]
            want = c[f"{tr}rhs"][:, 2:-2, 2:-2]
            bad = got.view(np.int64) != want.view(np.int64)
            assert not bad.any(), (f"{op} call {ic} {tr}: {int(bad.sum())} words differ, first "
                                   f"{np.argwhere(bad)[0].tolist()}: {got[bad][0]!r} != "
                                   f"{want[bad][0]!r}")
