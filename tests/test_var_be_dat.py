"""woof.da.var.be_dat: WRFDA be.dat reader and da_setup_be_regional (plan V0.1, gate O1).

The synthetic tests run everywhere.  The O1 gate compares woof's set-up arrays
with WRFDA 4.7.1's own, word for word: against the committed digests in
``tests/data/wrfda471/manifest.json`` when the tutorial ``be.dat`` is
available (``WOOF_WRFDA471_DIR`` pointing at a directory holding it, as
unpacked from the pinned test-data tarball), and against the oracle build's
``woof_be.bin`` dump when that is in the same directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from woof.da.var import be_dat as bd
from woof.da.var.oracle import array_digest, read_dump
from woof.da.var.reference import (
    calculate_rf_factors,
    recursive_filter_1d,
    recursive_filter_1d_adj,
)

MANIFEST = Path(__file__).parent / "data" / "wrfda471" / "manifest.json"


def _synthetic(ni=7, nj=5, nk=6, nb2d=2, cv_options=5, eta=False, seed=3):
    rng = np.random.default_rng(seed)
    bins = np.ones((ni, nj, nk), dtype=np.int64)
    bins2d = np.ones((ni, nj), dtype=np.int64)
    bins2d[:, nj // 2 :] = 2 if nb2d > 1 else 1
    num_bins = nk * nb2d
    for j in range(nj):
        for k in range(nk):
            bins[:, j, k] = (bins2d[0, j] - 1) * nk + k + 1
    reg = {}
    if cv_options == 5:
        reg = {
            "psi_chi": rng.standard_normal(num_bins),
            "psi_ps": rng.standard_normal((nk, nb2d)),
            "psi_t": rng.standard_normal((nk, nk, nb2d)),
        }
    elif cv_options == 6:
        for key, rank in bd._REG_RANK.items():
            shape = {1: (num_bins,), 2: (nk, nb2d), 3: (nk, nk, nb2d)}[rank]
            reg[key] = rng.standard_normal(shape)

    def eig(name, n):
        q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        ev = np.sort(rng.uniform(0.1, 5.0, n))[::-1].copy()
        return bd.BeEigen(name, q, ev, np.repeat(q[:, :, None], nb2d, axis=2),
                          np.repeat(ev[:, None], nb2d, axis=1) * rng.uniform(0.5, 2, (n, nb2d)))

    names = ("psi", "chi_u", "t_u", "rh_u") if cv_options != 7 else ("u", "v", "t", "rh")
    eigs = tuple(eig(n, nk) for n in names)
    return bd.BeDat(
        ni=ni, nj=nj, nk=nk, bin_type=1 if nb2d > 1 else 5,
        eta_be=np.linspace(0.99, 0.01, nk) if eta else None,
        lat_min=-90.0, lat_max=90.0, binwidth_lat=10.0,
        hgt_min=0.0, hgt_max=20000.0, binwidth_hgt=1000.0,
        num_bins=num_bins, num_bins2d=nb2d, bin=bins, bin2d=bins2d,
        cv_options=cv_options, regcoeff=reg, eigen=eigs,
        eigen_ps=eig("ps_u" if cv_options != 7 else "ps", 1),
        rf_lengthscale=tuple(rng.uniform(1.0, 4.0, nk) for _ in range(4)),
        rf_lengthscale_ps=rng.uniform(1.0, 4.0, 1),
        endian=">", real_size=8, sha256="",
    )


def _same(a: bd.BeDat, b: bd.BeDat):
    for f in ("ni", "nj", "nk", "bin_type", "num_bins", "num_bins2d", "cv_options",
              "lat_min", "hgt_max", "binwidth_lat"):
        assert getattr(a, f) == getattr(b, f), f
    np.testing.assert_array_equal(a.bin, b.bin)
    np.testing.assert_array_equal(a.bin2d, b.bin2d)
    assert a.regcoeff.keys() == b.regcoeff.keys()
    for k in a.regcoeff:
        np.testing.assert_array_equal(a.regcoeff[k], b.regcoeff[k])
    for ea, eb in zip(list(a.eigen) + [a.eigen_ps], list(b.eigen) + [b.eigen_ps]):
        assert ea.name == eb.name
        for f in ("evec_glo", "eval_glo", "evec_loc", "eval_loc"):
            np.testing.assert_array_equal(getattr(ea, f), getattr(eb, f))
    for ra, rb in zip(a.rf_lengthscale, b.rf_lengthscale):
        np.testing.assert_array_equal(ra, rb)


@pytest.mark.parametrize("endian", [">", "<"])
@pytest.mark.parametrize("cv_options", [5, 6, 7])
def test_round_trip(tmp_path, endian, cv_options):
    be = _synthetic(cv_options=cv_options)
    p = tmp_path / "be.dat"
    sha = bd.write_be_dat(p, be, endian=endian)
    got = bd.read_be_dat(p, cv_options=cv_options)
    assert got.endian == endian and got.real_size == 8 and got.sha256 == sha
    assert got.eta_be is None
    _same(be, got)


def test_real4_and_eta_be(tmp_path):
    be = _synthetic(eta=True)
    # binary32 values, so the narrowing write is exact
    be = bd.BeDat(**{**be.__dict__, "rf_lengthscale": tuple(r.astype(np.float32).astype(np.float64)
                                                             for r in be.rf_lengthscale)})
    p = tmp_path / "be4.dat"
    bd.write_be_dat(p, be, real_size=4)
    got = bd.read_be_dat(p)
    assert got.real_size == 4
    np.testing.assert_array_equal(got.eta_be, be.eta_be.astype(np.float32))
    for ra, rb in zip(be.rf_lengthscale, got.rf_lengthscale):
        np.testing.assert_array_equal(ra, rb)
    with pytest.raises(NotImplementedError, match="eta_be"):
        bd.setup_be_regional(got, ix=be.ni, jy=be.nj, kz=be.nk, ds=1000.0)


def test_refusals(tmp_path):
    be = _synthetic()
    p = tmp_path / "be.dat"
    bd.write_be_dat(p, be)
    raw = bytearray(p.read_bytes())
    (tmp_path / "trail.dat").write_bytes(bytes(raw) + b"\0\0")
    with pytest.raises(bd.BeDatError, match="trailing"):
        bd.read_be_dat(tmp_path / "trail.dat")
    bad = bytearray(raw)
    bad[16 + 3] ^= 1  # first record's closing marker
    (tmp_path / "marker.dat").write_bytes(bytes(bad))
    with pytest.raises(bd.BeDatError, match="disagree|past the end"):
        bd.read_be_dat(tmp_path / "marker.dat")
    renamed = bd.BeDat(**{**be.__dict__, "eigen_ps": bd.BeEigen("tsk", *list(be.eigen_ps.__dict__.values())[1:])})
    bd.write_be_dat(tmp_path / "name.dat", renamed)
    with pytest.raises(bd.BeDatError, match="expected ps_u"):
        bd.read_be_dat(tmp_path / "name.dat")
    with pytest.raises(ValueError, match="do not match"):
        bd.setup_be_regional(bd.read_be_dat(p), ix=be.ni + 1, jy=be.nj, kz=be.nk, ds=1.0)
    with pytest.raises(NotImplementedError, match="interpolate_stats"):
        bd.setup_be_regional(bd.read_be_dat(p), ix=be.ni, jy=be.nj, kz=be.nk, ds=1.0,
                             config=bd.BeConfig(interpolate_stats=True))


@pytest.mark.parametrize("pass_", [1, 2, 3])
def test_rf_adjoint(pass_):
    rng = np.random.default_rng(pass_)
    x = rng.standard_normal((4, 37))
    y = rng.standard_normal((4, 37))
    alpha = rng.uniform(0.2, 0.9, 4)
    lhs = np.sum(recursive_filter_1d(pass_, alpha, x) * y, axis=1)
    rhs = np.sum(x * recursive_filter_1d_adj(pass_, alpha, y), axis=1)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-13)


def test_rf_factors_normalise_the_peak():
    s = np.array([1.5, 3.0, 7.25])
    alpha, sf = calculate_rf_factors(s, 6)
    e = 0.25 * 6 / (s * s)
    np.testing.assert_array_equal(alpha, 1.0 + e - np.sqrt(e * (e + 2.0)))
    for m in range(3):
        f = np.zeros(1001)
        f[500] = 1.0
        for p in (1, 2, 3):
            f = recursive_filter_1d_adj(p, alpha[m], f)
        for p in (1, 2, 3):
            f = recursive_filter_1d(p, alpha[m], f)
        assert f[500] * sf[m] == pytest.approx(1.0, rel=1e-15)


def _dump_bytes(endian, records):
    out = [b"WOOFDUMP"]
    for name, a in records:
        a = np.asarray(a)
        kind, dt = (b"r", endian + "f8") if a.dtype.kind == "f" else (b"i", endian + "i4")
        out += [name.ljust(32).encode(), kind, np.array([a.ndim], dtype=endian + "i4").tobytes(),
                np.array(a.shape, dtype=endian + "i4").tobytes(),
                np.asarray(a, dtype=dt).reshape(-1, order="F").tobytes()]
    return b"".join(out)


@pytest.mark.parametrize("endian", [">", "<"])
def test_read_dump_either_byte_order_with_a_scalar_first(tmp_path, endian):
    # A rank-0 first record reads the same in both byte orders, so the
    # order must come from the whole file (woof_be.bin starts with cv%size).
    m = np.arange(6, dtype=np.float64).reshape(2, 3) / 7.0
    path = tmp_path / "d.bin"
    path.write_bytes(_dump_bytes(endian, [("cv%size", np.int32(42)), ("v1%evec", m),
                                          ("cv%size", np.int32(7))]))
    d = read_dump(path)
    assert int(d["cv%size"]) == 42 and int(d["cv%size#2"]) == 7
    np.testing.assert_array_equal(d["v1%evec"], m)
    path.write_bytes(_dump_bytes(endian, [("x", m)])[:-3])
    with pytest.raises(ValueError, match="neither byte order"):
        read_dump(path)


def test_vertical_truncation():
    ev = np.array([5.0, 3.0, 1.0, 0.5, 0.0])
    assert bd._vertical_truncation(100.0, ev, 1) == 4  # stops at the first non-positive
    assert bd._vertical_truncation(99.0, ev, 1) == 4  # 9.0/9.5 < 0.99 until k = 4
    assert bd._vertical_truncation(50.0, ev, 1) == 1
    assert bd._vertical_truncation(0.0, ev, 1) == 1  # WRFDA's "avoid segfault"
    assert bd._vertical_truncation(99.0, np.array([1.0, 1e-13, 1.0]), 1) == 1


def test_setup_synthetic():
    be = _synthetic(nb2d=2)
    cfg = bd.BeConfig(var_scaling2=2.0, len_scaling3=0.5, max_vert_var1=100.0)
    b = bd.setup_be_regional(be, ix=be.ni, jy=be.nj, kz=be.nk, ds=3000.0, config=cfg)
    assert b.cv_mz[0] == be.nk and b.cv_mz[4] == 1
    # bin_type 1: regressions averaged over j, left to right
    raw = be.regcoeff["psi_ps"][:, be.bin2d[0, :] - 1].T
    acc = np.zeros(be.nk)
    for j in range(be.nj):
        acc = acc + raw[j]
    np.testing.assert_array_equal(b.reg["psi_ps"], np.broadcast_to(acc / be.nj, raw.shape))
    v2 = b.v[1]
    s = (be.rf_lengthscale[1][: v2.mz] * 3000.0) * 1.0 / 3000.0
    alpha, sf = calculate_rf_factors(s, 6)
    np.testing.assert_array_equal(v2.rf_alpha, alpha)
    np.testing.assert_array_equal(v2.val[0], sf * (2.0 * np.sqrt(be.eigen[1].eval_glo[: v2.mz])))
    v3 = b.v[2]
    s3 = (0.5 * (be.rf_lengthscale[2][: v3.mz] * 3000.0)) / 3000.0
    np.testing.assert_array_equal(v3.rf_lengthscale, s3)
    # vert_evalue = 2: per-latitude eigenvalues by bin2d
    b2 = bd.setup_be_regional(be, ix=be.ni, jy=be.nj, kz=be.nk, ds=3000.0,
                              config=bd.BeConfig(vert_evalue=2))
    j = be.nj - 1
    m = b2.v[0].mz
    _, sf0 = calculate_rf_factors(be.rf_lengthscale[0][:m] * 3000.0 / 3000.0, 6)
    np.testing.assert_array_equal(
        b2.v[0].val[j], sf0 * np.sqrt(be.eigen[0].eval_loc[:m, be.bin2d[0, j] - 1]))


# ---------------------------------------------------------------- gate O1

def _tutorial():
    d = os.environ.get("WOOF_WRFDA471_DIR")
    if not d or not (Path(d) / "be.dat").is_file():
        pytest.skip("WOOF_WRFDA471_DIR with the WRFDA tutorial be.dat is not set")
    man = json.loads(MANIFEST.read_text())
    be = bd.read_be_dat(Path(d) / "be.dat")
    assert be.sha256 == man["tutorial"]["be_dat_sha256"]
    t = man["tutorial"]
    b = bd.setup_be_regional(be, ix=t["ix"], jy=t["jy"], kz=t["kz"], ds=t["ds"])
    return Path(d), man, b


def _o1_arrays(b: bd.RegionalB) -> dict:
    out = {"cv_mz": np.array(b.cv_mz, dtype=np.int32)}
    for i, s in enumerate(b.v, start=1):
        if s.mz > 0:
            out[f"v{i}%val"] = s.val
            out[f"v{i}%evec"] = s.evec
            out[f"v{i}%rf_alpha"] = s.rf_alpha
            out[f"v{i}%val_g"] = s.val_g
            out[f"v{i}%evec_g"] = s.evec_g
    out["reg_psi_chi"] = b.reg["psi_chi"]
    out["reg_psi_t"] = b.reg["psi_t"]
    out["reg_psi_ps"] = b.reg["psi_ps"]
    return out


def test_o1_tutorial_digests():
    """O1: every set-up array equals WRFDA's (committed digests of its dump)."""
    _, man, b = _tutorial()
    want = man["o1_digests"]
    got = {k: array_digest(v) for k, v in _o1_arrays(b).items()}
    assert got == want


def test_o1_tutorial_dump():
    """O1 against the oracle build's dump itself (when it is beside be.dat)."""
    d, _, b = _tutorial()
    if not (d / "woof_be.bin").is_file():
        pytest.skip("woof_be.bin (oracle dump) not in WOOF_WRFDA471_DIR")
    dump = read_dump(d / "woof_be.bin")
    for k, v in _o1_arrays(b).items():
        ref = dump[k]
        if k == "cv_mz":  # WRFDA's cv_mz also carries the alpha and ensemble sizes
            ref = ref[:5]
        assert ref.shape == np.shape(v), k
        assert np.array_equal(ref, v), (k, np.max(np.abs(ref - v)))
