"""NEMO's equation of state (the opt-in), bitwise against NEMO 5.0.2.

eos_insitu_pot, rab_3d and bn2 for TEOS-10, EOS-80 and S-EOS, run by
``woof_nemo.eos``.  The fixture is ``optional/nemo/oracle/build_eos.sh``:
src/OCE/TRA/eosbn2.F90 compiled whole and unchanged against NEMO-named
service modules, with eos_init reading each nameos choice from namelist text.
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

FIX = optin.optional_root() / "nemo" / "oracle" / "fixtures" / "eos"
NX, NY, NZ = 7, 5, 31


def _field(rows, key):
    a = np.zeros((NZ, NY, NX))
    for r in rows:
        a[int(r["jk"]) - 1, int(r["jj"]) - 1, int(r["ji"]) - 1] = float(r[key])
    return a


def _rows(name):
    with (FIX / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_fixtures_are_the_ones_the_build_hashed():
    rows = (FIX / "eos-sha256sums.txt").read_text().splitlines()
    pinned = {line.split()[1]: line.split()[0] for line in rows}
    for name in ("eos-grid.csv", "eos-teos10.csv", "eos-eos80.csv", "eos-seos.csv"):
        assert pinned[name] == hashlib.sha256((FIX / name).read_bytes()).hexdigest()


def test_stochastic_eos_is_refused_by_name():
    mod = optin.load("nemo")
    with pytest.raises(NotImplementedError, match="ln_sto_eos"):
        mod.EquationOfState("teos10", stochastic=True)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("scheme", ["teos10", "eos80", "seos"])
def test_eos_rab_bn2_are_bitwise_against_nemo(scheme):
    import cupy as cp

    mod = optin.load("nemo")
    grid = _rows("eos-grid.csv")
    g = {k: cp.asarray(_field(grid, k)) for k in ("tmask", "wmask", "gdept", "gdepw", "e3w", "t", "s")}
    eos = mod.EquationOfState(scheme)
    z = lambda: cp.zeros((NZ, NY, NX), cp.float64)  # noqa: E731
    prd, prhop, alpha, beta, pn2 = z(), z(), z(), z(), z()
    eos.insitu_pot(g["t"], g["s"], g["gdept"], g["tmask"], prd, prhop)
    eos.rab(g["t"], g["s"], g["gdept"], g["tmask"], alpha, beta)
    eos.bn2(g["t"], g["s"], alpha, beta, g["gdept"], g["gdepw"], g["e3w"], g["wmask"], pn2)
    want = _rows(f"eos-{scheme}.csv")
    for key, got in (("prd", prd), ("prhop", prhop), ("alpha", alpha), ("beta", beta), ("n2", pn2)):
        assert np.array_equal(cp.asnumpy(got).view(np.uint64), _field(want, key).view(np.uint64)), key
