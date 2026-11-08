"""NEMO implicit vertical diffusion of T and S (the opt-in), bitwise against NEMO 5.0.2.

tra_zdf run by ``woof_nemo.trazdf``.  The fixture is
``optional/nemo/oracle/build_trazdf.sh``: trazdf.F90 compiled whole and
unchanged against NEMO-named service modules, three calls over a 9 x 7 x 21
grid with a halo, land, a shelf and a barely mixed fresh column whose
salinity the solve drives below zero (NEMO resets it to 0.1).  Every test
skips, naming the opt-in rule, unless ``WOOF_ENABLE_NEMO=1``.
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

FIX = optin.optional_root() / "nemo" / "oracle" / "fixtures" / "trazdf"
NX, NY, NZ = 9, 7, 21


def _rows(name):
    with (FIX / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _f3(rows, key):
    a = np.zeros((NZ, NY, NX))
    for r in rows:
        a[int(r["jk"]) - 1, int(r["jj"]) - 1, int(r["ji"]) - 1] = float(r[key])
    return a


def test_fixtures_are_the_ones_the_build_hashed():
    rows = (FIX / "trazdf-sha256sums.txt").read_text().splitlines()
    pinned = {line.split()[1]: line.split()[0] for line in rows}
    for name in ("trazdf-grid.csv", "trazdf-steps.csv"):
        assert pinned[name] == hashlib.sha256((FIX / name).read_bytes()).hexdigest()


@pytest.mark.parametrize("kw,word", [(dict(ln_ldfslp=True), "l_ldfslp"),
                                     (dict(ln_zad_aimp=True), "ln_zad_Aimp"),
                                     (dict(ln_zdfmfc=True), "ln_zdfmfc"),
                                     (dict(ln_zdfddm=True), "ln_zdfddm")])
def test_unported_tra_zdf_options_are_refused_by_name(kw, word):
    mod = optin.load("nemo")
    with pytest.raises(NotImplementedError, match=word):
        mod.TracerVerticalDiffusion(rDt=1800.0, **kw)


@pytest.mark.gpu
@requires_gpu
def test_tra_zdf_is_bitwise_against_nemo():
    import cupy as cp

    mod = optin.load("nemo")
    g = _rows("trazdf-grid.csv")
    d = lambda a: cp.asarray(np.ascontiguousarray(a))  # noqa: E731
    f = {k: d(_f3(g, k)) for k in ("tmask", "e3t", "e3w", "avt", "t0", "s0", "trhs", "srhs")}
    zdf = mod.TracerVerticalDiffusion(rDt=1800.0)
    tb, sb = f["t0"].copy(), f["s0"].copy()
    ta, sa = f["t0"].copy(), f["s0"].copy()
    steps = _rows("trazdf-steps.csv")
    inner = (slice(None), slice(1, NY - 1), slice(1, NX - 1))
    for kt in (1, 2, 3):
        zdf.step(avt=f["avt"], e3t=f["e3t"], e3w=f["e3w"], tmask=f["tmask"], tb=tb, sb=sb,
                 trhs=f["trhs"], srhs=f["srhs"], ta=ta, sa=sa)
        rows = [r for r in steps if r["kt"] == str(kt)]
        for key, arr in (("t", ta), ("s", sa)):
            got = cp.asnumpy(arr)[inner]
            assert np.array_equal(got.view(np.uint64), _f3(rows, key)[inner].view(np.uint64)), (kt, key)
        tb, sb = ta.copy(), sa.copy()
