"""NEMO 2-band penetrative solar radiation (the opt-in), bitwise against NEMO 5.0.2.

The port lives in ``optional/nemo`` under CeCILL v2.0 and loads only through
:func:`woof.optin.load` with ``WOOF_ENABLE_NEMO=1``; every test here skips,
naming that rule, when the opt-in is off.  The fixture is
``optional/nemo/oracle/build.sh``: tra_qsr, qsr_2BD and qsr_ext_lev cut from
the pinned traqsr.F90 and compiled unchanged, three leapfrog steps over a
7 x 5 x 31 grid with a shelf column, a dark column and a full-sun column.
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

FIXTURES = optin.optional_root() / "nemo" / "oracle" / "fixtures"
NX, NY = 7, 5


def _grid():
    lines = (FIXTURES / "traqsr-grid.csv").read_text().splitlines()
    nk0, nkv, nksr, nz = (int(x) for x in lines[0].split(",")[-4:])
    rows = list(csv.DictReader(lines[1:]))
    fields = {k: np.zeros((nz, NY, NX)) for k in ("e3t", "gdepw", "tmask", "wmask")}
    for r in rows:
        for k in fields:
            fields[k][int(r["jk"]) - 1, int(r["jj"]) - 1, int(r["ji"]) - 1] = float(r[k])
    return (nk0, nkv, nksr, nz), fields


def _steps():
    with (FIXTURES / "traqsr-steps.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def _field(rows, key, nz):
    a = np.zeros((nz, NY, NX))
    for r in rows:
        a[int(r["jk"]) - 1, int(r["jj"]) - 1, int(r["ji"]) - 1] = float(r[key])
    return a


def test_fixtures_are_the_ones_build_sh_hashed():
    rows = (FIXTURES / "oracle-sha256sums.txt").read_text().splitlines()
    pinned = {line.split()[1]: line.split()[0] for line in rows}
    for name in ("traqsr-grid.csv", "traqsr-steps.csv"):
        assert pinned[name] == hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()


def test_unported_light_schemes_and_restart_are_refused_by_name():
    mod = optin.load("nemo")
    with pytest.raises(NotImplementedError, match="ln_qsr_2bd"):
        mod.TraQsr2BD(gdepw=None, e3t=None, tmask=None, wmask=None, rDt=1800.0,
                      scheme="rgb")
    with pytest.raises(NotImplementedError, match="restart"):
        mod.TraQsr2BD(gdepw=None, e3t=None, tmask=None, wmask=None, rDt=1800.0,
                      restart=True)


@pytest.mark.gpu
@requires_gpu
def test_traqsr_2bd_is_bitwise_against_nemo():
    import cupy as cp

    mod = optin.load("nemo")
    (nk0, nkv, nksr, nz), g = _grid()
    qsr_model = mod.TraQsr2BD(**{k: cp.asarray(v) for k, v in g.items()}, rDt=1800.0)
    assert (qsr_model.nk0, qsr_model.nkV, qsr_model.nksr) == (nk0, nkv, nksr)
    steps = _steps()
    for kt in sorted({int(r["kt"]) for r in steps}):
        rows = [r for r in steps if int(r["kt"]) == kt]
        qsr = np.zeros((NY, NX))
        for r in rows:
            qsr[int(r["jj"]) - 1, int(r["ji"]) - 1] = float(r["qsr"])
        ta = cp.zeros((nz, NY, NX), cp.float64)
        qsr_model.step(cp.asarray(qsr), ta)
        for key, got in (("ta_rhs", ta), ("qsr_hc", qsr_model.qsr_hc),
                         ("qsr_hc_b", qsr_model.qsr_hc_b)):
            want = _field(rows, key, nz)
            assert np.array_equal(cp.asnumpy(got).view(np.uint64), want.view(np.uint64)), (kt, key)
        fraq = _field(rows, "fraqsr_1lev", nz)[0]
        assert np.array_equal(cp.asnumpy(qsr_model.fraqsr_1lev).view(np.uint64),
                              fraq.view(np.uint64)), (kt, "fraqsr_1lev")
