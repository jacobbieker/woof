"""icloud = 3 (WRF-Solar cal_cldfra3) against compiled WRF v4.7.1, word for word.

tests/data/oracles/cldfra3 is tools/cldfra3_wrf471_oracle's record: the
unmodified routines at gfortran -O0 -ffp-contract=off against glibc 2.39's
libm, driven by 500 columns (make_columns.py).  Every output word -- gridkm,
CLDFRA and the radiation copies of qc and qi -- must be equal.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.core.cldfra3 import Cldfra3Fatal, cal_cldfra3, cldfra3_gridkm

DATA = Path(__file__).parent / "data" / "oracles" / "cldfra3"


def _read():
    receipt = (DATA / "receipt.txt").read_text()
    for name in ("columns.bin", "cldfra3.out.bin"):
        digest = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        assert f"{digest}  {name}" in receipt, name
    raw = np.fromfile(DATA / "columns.bin", dtype=np.uint8)
    out = np.fromfile(DATA / "cldfra3.out.bin", dtype=np.uint8)
    pos = opos = 0
    (ncol,) = raw[:4].view("<i4")
    pos = 4
    cols = []
    for _ in range(ncol):
        nz = int(raw[pos:pos + 4].view("<i4")[0])
        dx, dy, xland = raw[pos + 4:pos + 16].view("<f4")
        pos += 16
        arrays = raw[pos:pos + 28 * nz].view("<f4").reshape(7, nz)
        pos += 28 * nz
        res = out[opos:opos + 4 + 12 * nz].view("<f4")
        opos += 4 + 12 * nz
        cols.append(dict(nz=nz, dx=dx, dy=dy, xland=xland, arrays=arrays,
                         gridkm=res[0], cf=res[1:1 + nz], qc=res[1 + nz:1 + 2 * nz],
                         qi=res[1 + 2 * nz:]))
    assert pos == raw.size and opos == out.size
    return cols


@pytest.fixture(scope="module")
def columns():
    return _read()


def _groups(cols):
    groups = {}
    for i, c in enumerate(cols):
        key = (c["nz"], float(c["dx"]), float(c["dy"]))
        groups.setdefault(key, []).append(i)
    return groups


def test_fixture_exercises_every_arm(columns):
    """The record reaches the branches it claims to (make_columns.py)."""
    cf = np.concatenate([c["cf"] for c in columns])
    qc_in = np.concatenate([c["arrays"][1] for c in columns])
    qi_in = np.concatenate([c["arrays"][2] for c in columns])
    qc_out = np.concatenate([c["qc"] for c in columns])
    qi_out = np.concatenate([c["qi"] for c in columns])
    assert (cf == 1).sum() > 500 and ((cf > 0) & (cf < 1)).sum() > 2000
    assert (cf == 0).sum() > 2000
    log_arm = (qc_in + qi_in > 1e-10) & (qc_in + qi_in < 1e-6) & (qc_in <= 1e-6) & (qi_in < 1e-7)
    assert log_arm.sum() > 50
    assert (qc_out > qc_in).sum() > 200 and (qi_out > qi_in).sum() > 200
    # adjust_cloudFinal: a fractional-cloud liquid path above 1 kg m-2 is
    # multiplied by 1/path, which leaves exactly 1 kg m-2 behind.  The
    # stress columns reach it; the ice arm (iwp > 1) is out of reach of any
    # plausible column and stays unexercised.
    def path(c):
        qv, qc, qi, qs, dz, p, t = c["arrays"]
        frac = (c["cf"] > 0) & (c["cf"] < 1)
        return float((c["qc"] * (p / (287.0 * t)) * dz)[frac].sum())
    assert sum(abs(path(c) - 1.0) < 1e-3 for c in columns) >= 15
    assert {float(c["xland"]) for c in columns} == {1.0, 2.0}


@pytest.mark.parametrize("exact", ["0", "1"])
def test_cldfra3_every_word(columns, monkeypatch, exact):
    monkeypatch.setenv("GPUWM_WRF_EXACT", exact)
    total = 0
    for (nz, dx, dy), idx in _groups(columns).items():
        sub = [columns[i] for i in idx]
        gridkm = cldfra3_gridkm(dx, dy)
        for c in sub:
            assert np.float32(gridkm).view("u4") == c["gridkm"].view("u4")
        stack = lambda j: cp.asarray(np.stack([c["arrays"][j] for c in sub]))
        qv, qc, qi, qs, dz, p, t = (stack(j) for j in range(7))
        qc0, qi0 = qc.copy(), qi.copy()
        xland = cp.asarray(np.array([c["xland"] for c in sub], np.float32))
        cf, qc_rad, qi_rad = cal_cldfra3(qv, qc, qi, qs, dz, p, t, xland, gridkm)
        # the inputs are the prognostic fields: never written
        assert cp.array_equal(qc, qc0) and cp.array_equal(qi, qi0)
        for got, key in ((cf, "cf"), (qc_rad, "qc"), (qi_rad, "qi")):
            want = np.stack([c[key] for c in sub])
            got = cp.asnumpy(got)
            bad = np.flatnonzero(got.view("u4") != want.view("u4"))
            assert bad.size == 0, (key, nz, dx, bad[:5], got.ravel()[bad[:5]], want.ravel()[bad[:5]])
            total += want.size
    assert total == 3 * sum(c["nz"] for c in columns)


def test_fatal_profile_is_refused():
    """A column warmer than -12 C within three levels of its top is WRF's
    wrf_error_fatal; the port refuses it instead of inventing a cloud."""
    nz = 20
    p = np.linspace(100000.0, 20000.0, nz, dtype=np.float32)[None]
    t = np.full((1, nz), 280.0, np.float32)
    z = np.zeros((1, nz), np.float32)
    dz = np.full((1, nz), 300.0, np.float32)
    with pytest.raises(Cldfra3Fatal):
        cal_cldfra3(cp.asarray(z + 1e-3), cp.asarray(z), cp.asarray(z), cp.asarray(z),
                    cp.asarray(dz), cp.asarray(p), cp.asarray(t),
                    cp.ones(1, cp.float32), np.float32(3.0))


def test_perturbation_hook_refused():
    z = cp.zeros((1, 10), cp.float32)
    with pytest.raises(NotImplementedError, match="pert_cld3"):
        cal_cldfra3(z, z, z, z, z, z, z, cp.ones(1, cp.float32), 3.0,
                    perturbation={"pert_cld3_t": 0.1})
