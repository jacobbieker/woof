"""MAD-WRF's column routines against compiled WRF v4.7.1, word for word.

tests/data/oracles/madwrf is tools/madwrf_wrf471_oracle's record: real.exe's
Init_madwrf_clouds (module_madwrf.F, unmodified) on 250 columns under its
five cloud-initialization inputs, and solve_em's Cloud_tracer_nudge on six
blocks, gfortran -O0 -ffp-contract=off against glibc 2.43 on x86-64.  Every
output word -- qv, qc, qi, qs, CLDFRA, CLDMASK, CLDTOPZ and the nudged qc,
qi, qs -- must be equal (woof/core/madwrf.py, kernels/madwrf.cu).
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.core import madwrf

DATA = Path(__file__).parent / "data" / "oracles" / "madwrf"


def _read():
    receipt = (DATA / "receipt.txt").read_text()
    for name in ("madwrf_columns.bin", "madwrf.out.bin"):
        digest = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        assert f"{digest}  {name}" in receipt, name
    raw = np.fromfile(DATA / "madwrf_columns.bin", dtype=np.uint8)
    out = np.fromfile(DATA / "madwrf.out.bin", dtype="<f4")
    pos, opos = 4, 0
    (ncol,) = raw[:4].view("<i4")
    cols = []
    for _ in range(ncol):
        nz, fm, ft, fb, fbr = (int(v) for v in raw[pos:pos + 20].view("<i4"))
        pos += 20
        dx, dy, xland, ht, cm, ct, cb, br = raw[pos:pos + 32].view("<f4")
        pos += 32
        sizes = (nz, nz, nz + 1, nz + 1, nz, nz, nz, nz)
        arrs = []
        for n in sizes:
            arrs.append(raw[pos:pos + 4 * n].view("<f4"))
            pos += 4 * n
        res = out[opos:opos + 5 * nz + 3]
        opos += 5 * nz + 3
        cols.append(dict(nz=nz, flags=(fm, ft, fb, fbr), dx=dx, dy=dy, xland=xland,
                         ht=ht, cldmask=cm, cldtopz=ct, cldbasez=cb, brtemp=br,
                         arrays=arrs, out=res))
    (ncase,) = raw[pos:pos + 4].view("<i4")
    pos += 4
    cases = []
    for _ in range(ncase):
        ni, nk, nj, its, ite, jts, jte = (int(v) for v in raw[pos:pos + 28].view("<i4"))
        pos += 28
        dt, relax, nudge, xtime = raw[pos:pos + 16].view("<f4")
        pos += 16
        n = 6 * nj * (nk + 1) * ni
        q = raw[pos:pos + 4 * n].view("<f4").reshape(6, nj, nk + 1, ni)
        pos += 4 * n
        m = 3 * nj * (nk + 1) * ni
        res = out[opos:opos + m].reshape(3, nj, nk + 1, ni)
        opos += m
        cases.append(dict(ni=ni, nk=nk, nj=nj, tile=(its, ite, jts, jte), dt=dt,
                          relax=relax, nudge=nudge, xtime=xtime, q=q, out=res))
    assert pos == raw.size and opos == out.size
    return cols, cases


@pytest.fixture(scope="module")
def record():
    return _read()


def _bits(a):
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def test_the_record_reaches_every_arm(record):
    cols, cases = record
    modes = {madwrf.insert_clouds(*c["flags"]) for c in cols}
    assert modes == {0, 1, 2, 3}
    moved = sum(int(np.any(_bits(c["out"][:4 * c["nz"]])
                           != _bits(np.concatenate(c["arrays"][4:])))) for c in cols)
    assert moved > 150
    cf = np.concatenate([c["out"][4 * c["nz"]:5 * c["nz"] + 1] for c in cols])
    assert (cf == 0.5).sum() > 20 and ((cf > 0) & (cf < 1) & (cf != 0.5)).sum() > 500
    assert any(c["flags"][3] and c["out"][-1] > 0 for c in cols)     # BRTEMP cloud top
    assert {bool(madwrf.nudge_due(c["xtime"], c["nudge"])) for c in cases} == {True, False}


@pytest.mark.parametrize("exact", ["0", "1"])
def test_cloud_init_every_word(record, monkeypatch, exact):
    monkeypatch.setenv("GPUWM_WRF_EXACT", exact)
    cols, _ = record
    words = 0
    for i, c in enumerate(cols):
        nz = c["nz"]
        d = lambda a: cp.asarray(np.ascontiguousarray(a, np.float32)[None])
        t2, p, ph, phb, qv, qc, qi, qs = (d(a) for a in c["arrays"])
        cf, cm, ct = madwrf.cloud_init(
            t2=t2, p=p, ph=ph, phb=phb, qv=qv, qc=qc, qi=qi, qs=qs,
            xland=[c["xland"]], ht=[c["ht"]], dx=c["dx"], dy=c["dy"],
            cldmask=[c["cldmask"]], cldtopz=[c["cldtopz"]], cldbasez=[c["cldbasez"]],
            brtemp=[c["brtemp"]], flags=c["flags"])
        got = np.concatenate([cp.asnumpy(a).ravel() for a in (qv, qc, qi, qs, cf, cm, ct)])
        bad = np.flatnonzero(_bits(got) != _bits(c["out"]))
        assert bad.size == 0, (i, c["flags"], bad[:6], got[bad[:6]], c["out"][bad[:6]])
        words += got.size
    assert words == sum(5 * c["nz"] + 3 for c in cols)


@pytest.mark.parametrize("exact", ["0", "1"])
def test_nudge_every_word(record, monkeypatch, exact):
    monkeypatch.setenv("GPUWM_WRF_EXACT", exact)
    _, cases = record
    for c in cases:
        # Fortran (ni, nk+1, nj) is woof's (nj, nk+1, ni); move to (nk+1, nj, ni)
        # the mass levels (woof's state arrays): Fortran's k_end = kde-1
        arr = [cp.asarray(np.ascontiguousarray(c["q"][n].transpose(1, 0, 2)[:c["nk"]]))
               for n in range(6)]
        its, ite, jts, jte = c["tile"]
        did = madwrf.nudge(*arr, dt=c["dt"], dt_relax=c["relax"], xtime_minutes=c["xtime"],
                           dt_nudge=c["nudge"], tile=(its - 1, min(ite, c["ni"]),
                                                      jts - 1, min(jte, c["nj"])))
        assert did == madwrf.nudge_due(c["xtime"], c["nudge"])
        # only the mass levels move: k_end = min(kte, kde-1); the top is the input
        want = c["out"].transpose(0, 2, 1, 3)
        for n in range(3):
            got = cp.asnumpy(arr[n])
            got_mass, want_mass = got, want[n][:c["nk"]]
            assert np.array_equal(_bits(got_mass), _bits(want_mass)), (c["dt"], c["xtime"], n)
            assert np.array_equal(_bits(want[n][c["nk"]]), _bits(c["q"][n].transpose(1, 0, 2)[c["nk"]]))
