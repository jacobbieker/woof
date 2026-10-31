"""WRF-Solar EPS scheme hooks against compiled WRF v4.7.1, word for word.

tests/data/oracles/wrfsolar_eps is tools/wrfsolar_eps_wrf471_oracle's
record: the drivers' unmodified Add_/Remove_multi_perturb_* routines (FARMS,
Deng, Thompson) and the icloud = 3 input statements, gfortran -O0
-ffp-contract=off on x86-64, over 128 columns x 24 levels and six sets of
namelist factors (make_inputs.py).  The perturbation fields are inputs.
Every output word -- the perturbed fields, Add_'s saved copies and what
Remove_ leaves behind -- must be equal (woof/core/wrfsolar_eps.py,
woof/core/kernels/wrfsolar_eps.cu).
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.core import wrfsolar_eps as eps

DATA = Path(__file__).parent / "data" / "oracles" / "wrfsolar_eps"
PERTS = ("ALBEDO", "AOD", "ANGSTROM", "ASSYMFAC", "QVAPOR", "QCLOUD", "QICE",
         "QSNOW", "NI", "TH", "W")


def _read():
    receipt = (DATA / "receipt.txt").read_text()
    for name in ("inputs.bin", "eps.out.bin"):
        digest = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        assert f"{digest}  {name}" in receipt, name
    raw = np.fromfile(DATA / "inputs.bin", dtype=np.uint8)
    ni, nk, ncase = (int(v) for v in raw[:12].view("<i4"))
    pos = 12
    fac = raw[pos:pos + 4 * 18 * ncase].view("<f4").reshape(ncase, 18)
    pos += 4 * 18 * ncase
    kme = nk + 1

    def take(shape):
        nonlocal pos
        n = int(np.prod(shape))
        a = raw[pos:pos + 4 * n].view("<f4").reshape(shape)
        pos += 4 * n
        return a
    perts = {name: take((kme, ni)) for name in PERTS}
    state = {name: take((kme, ni)) for name in ("qv", "qc", "qi", "qs", "ni", "th", "t", "w")}
    two = {name: take((ni,)) for name in ("albedo", "aod", "angexp", "aerasy")}
    mp = {name: take((kme, ni)) for name in ("qv", "qc", "qi", "qs", "ni")}
    assert pos == raw.size
    out = np.fromfile(DATA / "eps.out.bin", dtype="<f4")
    return dict(ni=ni, nk=nk, ncase=ncase, fac=fac, perts=perts, state=state,
                two=two, mp=mp, out=out)


@pytest.fixture(scope="module")
def record():
    return _read()


class _Out:
    def __init__(self, out):
        self.out, self.pos = out, 0

    def take(self, n):
        a = self.out[self.pos:self.pos + n]
        self.pos += n
        return a


def _dev(a):
    return cp.asarray(np.ascontiguousarray(a, dtype=np.float32))


def _same(got, want, what, total):
    got = cp.asnumpy(got).ravel()
    want = np.asarray(want, dtype=np.float32).ravel()
    assert got.size == want.size, what
    bad = np.flatnonzero(got.view(np.uint32) != want.view(np.uint32))
    assert bad.size == 0, (what, bad.size, bad[:5], got[bad[:5]], want[bad[:5]])
    total[0] += want.size


def test_the_record_reaches_every_clamp(record):
    """The fixture moves inputs across the clamps it claims to."""
    p = record["perts"]["QVAPOR"][:record["nk"]]
    assert ((1 + p * 4.0) < 0).sum() > 0                 # 1 + pert*factor < 0
    qv = record["state"]["qv"][:record["nk"]]
    assert (qv < 0).sum() > 0 and (np.signbit(qv) & (qv == 0)).sum() > 0
    bits = qv.view(np.uint32) & 0x7fffffff
    assert ((bits > 0) & (bits < 0x00800000)).sum() > 0                  # subnormals
    alb = record["two"]["albedo"]
    assert (alb == 1.0).sum() > 0 and (alb == 0.0).sum() > 0
    assert (np.signbit(alb) & (alb == 0)).sum() > 0
    # signed-zero ties reach every zero-bounded clamp (the forms differ per
    # statement on x86-64; woof/core/wrfsolar_eps.py)
    out = record["out"]
    assert (np.signbit(out) & (out == 0)).sum() > 100


@pytest.mark.parametrize("exact", ["0", "1"])
def test_every_hook_word(record, monkeypatch, exact):
    monkeypatch.setenv("GPUWM_WRF_EXACT", exact)
    ni, nk = record["ni"], record["nk"]
    perts = {name: eps.Pattern(_dev(a[:nk].reshape(nk, 1, ni)))
             for name, a in record["perts"].items()}
    o = _Out(record["out"])
    m3, w3 = ni * nk, ni * (nk + 1)
    s, two, mp = record["state"], record["two"], record["mp"]
    total = [0]
    for c in range(record["ncase"]):
        f = record["fac"][c]
        # FARMS
        fields = {k: _dev(v.reshape(1, ni)) for k, v in two.items()}
        q = {k: _dev(s[k][:nk].reshape(nk, 1, ni)) for k in ("qv", "qc", "qs")}
        saved = eps.farms_add(perts, f[0:7], albedo=fields["albedo"], aod=fields["aod"],
                              angexp=fields["angexp"], aerasy=fields["aerasy"], **q)
        for k in ("albedo", "aod", "angexp", "aerasy"):
            _same(fields[k], o.take(ni), f"farms add {k} case {c}", total)
        for k in ("qv", "qc", "qs"):
            _same(q[k], o.take(m3), f"farms add {k} case {c}", total)
        for k, t in zip(("qv", "qc", "qs"), saved):
            _same(t, o.take(m3), f"farms add {k}_tmp case {c}", total)
        eps.farms_remove(perts, f[0:7], saved, albedo=fields["albedo"], aod=fields["aod"],
                         angexp=fields["angexp"], aerasy=fields["aerasy"], **q)
        for k in ("albedo", "aod", "angexp", "aerasy"):
            _same(fields[k], o.take(ni), f"farms remove {k} case {c}", total)
        for k in ("qv", "qc", "qs"):
            _same(q[k], o.take(m3), f"farms remove {k} case {c}", total)
        # Deng
        d = {k: _dev(s[k][:nk].reshape(nk, 1, ni)) for k in ("qv", "qc", "th")}
        d["w"] = _dev(s["w"].reshape(nk + 1, 1, ni))
        eps.deng_add(perts, f[7:11], **d)
        for k in ("qv", "qc", "th"):
            _same(d[k], o.take(m3), f"deng add {k} case {c}", total)
        _same(d["w"], o.take(w3), f"deng add w case {c}", total)
        eps.deng_remove(perts, f[7:11], **d)
        for k in ("qv", "qc", "th"):
            _same(d[k], o.take(m3), f"deng remove {k} case {c}", total)
        _same(d["w"], o.take(w3), f"deng remove w case {c}", total)
        # Thompson
        t = {k: _dev(s[k][:nk].reshape(nk, 1, ni)) for k in ("qv", "qc", "qi", "qs", "ni")}
        saved = eps.thompson_add(perts, f[11:16], **t)
        for k in ("qv", "qc", "qi", "qs", "ni"):
            _same(t[k], o.take(m3), f"thompson add {k} case {c}", total)
        for k, tmp in zip(("qv", "qc", "qi", "qs", "ni"), saved):
            _same(tmp, o.take(m3), f"thompson add {k}_tmp case {c}", total)
        post = {k: _dev(mp[k][:nk].reshape(nk, 1, ni)) for k in ("qv", "qc", "qi", "qs", "ni")}
        eps.thompson_remove(perts, f[11:16], saved, **post)
        for k in ("qv", "qc", "qi", "qs", "ni"):
            _same(post[k], o.take(m3), f"thompson remove {k} case {c}", total)
        # icloud = 3 inputs (cal_cldfra3's columns: any shared layout)
        tt, qq = eps.cld3_inputs(_dev(s["t"][:nk].T), _dev(s["qv"][:nk].T),
                                 _dev(record["perts"]["TH"][:nk].T),
                                 _dev(record["perts"]["QVAPOR"][:nk].T), f[16], f[17])
        _same(tt.T, o.take(m3), f"cld3 t case {c}", total)
        _same(qq.T, o.take(m3), f"cld3 qv case {c}", total)
    assert o.pos == record["out"].size
    assert total[0] == record["out"].size


def test_a_height_constant_pattern_reads_its_one_level(record):
    """STOCHPERT.TBL's vertstruc = 0 rows are stored as one level; the
    planar read gives the words of the same pattern repeated in height."""
    ni, nk = record["ni"], record["nk"]
    base = record["perts"]["QVAPOR"][0]
    full = {name: eps.Pattern(_dev(np.broadcast_to(base, (nk, ni)).reshape(nk, 1, ni)))
            for name in ("QVAPOR", "QCLOUD", "QICE", "QSNOW", "NI")}
    flat = {name: eps.Pattern(_dev(base.reshape(1, 1, ni))) for name in full}
    f = np.array([1.5, 2.0, 0.5, 3.0, 1.0], np.float32)
    outs = []
    for perts in (full, flat):
        t = {k: _dev(record["state"][k][:nk].reshape(nk, 1, ni))
             for k in ("qv", "qc", "qi", "qs", "ni")}
        saved = eps.thompson_add(perts, f, **t)
        eps.thompson_remove(perts, f, saved, **t)
        outs.append([cp.asnumpy(v) for v in t.values()])
    for a, b in zip(*outs):
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
