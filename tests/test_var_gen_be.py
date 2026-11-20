"""woof.da.var.gen_be: be.dat from forecast differences (plan V2, gate GV2).

The synthetic tests run everywhere: the Poisson solve inverts WRFDA's
5-point Laplacian, the winds' psi/chi round trip, the stage-4 pair sums and
counts against WRFDA's own pair loop, a planted balance regression, the
diagnostics' e20.8 round trip and smoothing, and a woof-made be.dat read back
by woof's 3D-Var set-up.

The oracle gate needs ``WOOF_GENBE_DIR`` holding WRFDA's gen_be sample
forecasts (``gen_be_fc/``, unpacked from ``gen_be_forecasts_200802.tar.gz``
on the WRFDA test-data page) and WRFDA 4.7.1's own ``oracle_cv5/be.dat`` and
``oracle_cv7/be.dat`` made from them (``tools/wrfda_oracle/run_gen_be.sh``;
their SHA-256 are in ``tests/data/wrfda471/manifest.json``).  woof's gen_be
on the same twenty forecasts (ten 24 h minus 12 h pairs, 2008-02-04 00Z to
02-08 12Z) must agree to GV2's tolerances: eigenvalues and regressions to
1e-6 relative, length scales to 1 %.  Measured on the Mac and on pb:
regressions within 1.0e-7, eigenvalues 5.5e-8, eigenvectors 8.2e-8 (absolute),
length scales 7.4e-8 for cv5, the residual being WRFDA's ``fft661`` trig
tables (``pi = 3.1415926``) in psi and chi; cv7 has no Poisson solve and
agrees to 3e-12 with identical length scales.  WRFDA's tutorial 3D-Var with
the woof-made file runs as with WRFDA's own (final J 21497.748 against
21497.749, 30 iterations each), and woof's 3D-Var with the woof-made file
reproduces WRFDA's log with it.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from woof.da.var import be_dat as bd
from woof.da.var import gen_be as gb

MANIFEST = Path(__file__).parent / "data" / "wrfda471" / "manifest.json"


def _laplacian(a, ds):
    out = np.zeros_like(a)
    out[1:-1, 1:-1] = ((a[2:, 1:-1] + a[:-2, 1:-1] + a[1:-1, 2:] + a[1:-1, :-2]) - 4.0 * a[1:-1, 1:-1]) / (ds * ds)
    return out


@pytest.mark.parametrize("shape", [(17, 11), (30, 21), (89, 59)])
def test_poisson_solve_inverts_the_five_point_laplacian(shape):
    rng = np.random.default_rng(1)
    ds = 3.0e4
    n1, n2 = gb._fft_size(shape[0]), gb._fft_size(shape[1])
    # WRFDA zeroes del2a outside 1..dim-2 and pads to (n1+1, n2+1) with a zero
    # boundary: a field whose Laplacian lies inside that window is recovered.
    big = np.zeros((n1 + 1, n2 + 1))
    big[1 : shape[0] - 2, 1 : shape[1] - 2] = rng.standard_normal((shape[0] - 3, shape[1] - 3))
    lap = _laplacian(big, ds)
    sol = gb._del2a_to_a(lap[: shape[0], : shape[1]], ds)
    # PI = 3.1415926 in the eigenvalues: a relative residual near 1e-7, as WRFDA's
    ref = big[: shape[0], : shape[1]]
    assert np.max(np.abs(sol - ref)) <= 1e-6 * np.max(np.abs(ref))


def test_fft_sizes_follow_da_fft_initialize1():
    assert [gb._fft_size(d) for d in (89, 90, 59, 60)] == [90, 90, 60, 60]
    assert gb._find_fft_factors(90) and gb._find_fft_factors(64) and not gb._find_fft_factors(58)
    assert not gb._find_fft_factors(0) and not gb._find_fft_factors(1) and not gb._find_fft_factors(-4)
    with pytest.raises(ValueError):
        gb._fft_size(2)


def test_nmc_perturbation_refuses_mismatched_cv_options():
    f = _synthetic_history()
    s5 = gb.standard_fields(f, 5)
    with pytest.raises(ValueError, match="lacks"):
        gb.nmc_perturbation(s5, s5, 7)
    with pytest.raises(ValueError, match="lacks"):
        gb.ensemble_perturbations([s5, s5], 7)
    with pytest.raises(NotImplementedError):
        gb.nmc_perturbation(f, f, 6)
    # raw fields are converted for the requested cv_options
    g = _synthetic_history(seed=1)
    d = gb.nmc_perturbation(f, g, 7)
    assert set(d) >= {"u", "v", "t", "rh", "ps", "height", "lat"}
    np.testing.assert_array_equal(d["height"], gb.standard_fields(g, 7)["height"])


def _synthetic_history(ni=24, nj=18, nk=6, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, ni + 1)
    y = np.linspace(0, 2 * np.pi, nj + 1)
    f = {
        "U": scale * (np.sin(x)[:, None, None] * np.cos(y[:-1])[None, :, None] + 0.1 * rng.standard_normal((ni + 1, nj, nk))),
        "V": scale * (np.cos(x[:-1])[:, None, None] * np.sin(y)[None, :, None] + 0.1 * rng.standard_normal((ni, nj + 1, nk))),
        "T": 5.0 * rng.standard_normal((ni, nj, nk)),
        "P": 100.0 * rng.standard_normal((ni, nj, nk)),
        "PB": np.broadcast_to(np.linspace(95000.0, 30000.0, nk), (ni, nj, nk)).copy(),
        "QVAPOR": np.abs(0.005 + 0.001 * rng.standard_normal((ni, nj, nk))),
        "PH": 10.0 * rng.standard_normal((ni, nj, nk + 1)),
        "PHB": np.broadcast_to(np.linspace(0.0, 9.81 * 12000.0, nk + 1), (ni, nj, nk + 1)).copy(),
        "PSFC": 100000.0 + 300.0 * rng.standard_normal((ni, nj)),
        "XLAT": np.broadcast_to(np.linspace(30.0, 45.0, nj), (ni, nj)).copy(),
        "MAPFAC_M": np.ones((ni, nj)),
        "MAPFAC_U": np.ones((ni + 1, nj)),
        "MAPFAC_V": np.ones((ni, nj + 1)),
        "ds": 30000.0,
    }
    return f


def test_standard_fields_shapes_and_cv7_winds():
    f = _synthetic_history()
    s5 = gb.standard_fields(f, 5)
    s7 = gb.standard_fields(f, 7)
    ni, nj, nk = f["T"].shape
    for n in ("psi", "chi", "t", "rh", "height"):
        assert s5[n].shape == (ni, nj, nk)
    assert s5["ps"].shape == (ni, nj)
    np.testing.assert_array_equal(s7["u"], 0.5 * (f["U"][:-1] + f["U"][1:]))
    np.testing.assert_array_equal(s7["t"], s5["t"])
    assert np.all(np.isfinite(s5["rh"])) and np.all(s5["rh"] > 0)


def test_psi_chi_reproduce_the_interior_vorticity_and_divergence():
    f = _synthetic_history(ni=30, nj=24, nk=1)
    ds = f["ds"]
    vor, div = gb._uv_to_vor_div(f["U"][:, :, 0], f["V"][:, :, 0], f["MAPFAC_U"], f["MAPFAC_V"], ds)
    chi = gb._del2a_to_a(div, ds)
    lap = _laplacian(chi, ds)
    # The solve holds div fixed on the interior (zero beyond, by WRFDA's padding).
    np.testing.assert_allclose(lap[2:-2, 2:-2], div[2:-2, 2:-2], rtol=0, atol=2e-6 * np.abs(div).max())


def test_vorticity_of_solid_body_rotation():
    # u = -a y, v = a x (C grid, unit map factors): corner vorticity is 2a everywhere inside
    ni, nj, a, ds = 12, 10, 1e-4, 3.0e4
    U = (-a * (np.arange(nj) + 0.5) * ds)[None, :] * np.ones((ni + 1, 1))
    V = (a * (np.arange(ni) + 0.5) * ds)[:, None] * np.ones((1, nj + 1))
    vor, div = gb._uv_to_vor_div(U, V, np.ones((ni + 1, nj)), np.ones((ni, nj + 1)), ds)
    np.testing.assert_allclose(vor[1:ni, 1:nj], 2 * a, rtol=1e-12)
    np.testing.assert_allclose(div, 0.0, atol=1e-15)


def test_gen_be_input_guards():
    samples, _, _ = _planted_samples(n=3)
    bad = [dict(s) for s in samples]
    bad[1]["ps"] = bad[1]["ps"][:-1]
    with pytest.raises(ValueError, match="shape"):
        gb.gen_be(bad)
    zero = [{**s, "psi": s["psi"] * 0.0} for s in samples]
    with pytest.raises(ValueError, match="no variance"):
        gb.gen_be(zero)
    # a one-level 3-D field keeps the EOF path (one mode field), not the 2-D bypass
    one = [{k: (v[:, :, :1] if np.ndim(v) == 3 else v) for k, v in s.items()} for s in samples]
    stats, modes = gb._stage3(one, "t", gb.GenBeConfig())
    assert modes is not None and modes[0].shape == one[0]["t"].shape


def _brute_pairs(f):
    """WRFDA's get_grid_info and get_covariance loops (stride 1, one bin), verbatim."""
    ni, nj = f.shape
    nn = ni * ni + nj * nj
    nr = np.zeros(nn + 1, dtype=np.int64)
    bb = np.zeros(nn + 1)
    for j in range(nj):
        for i in range(ni):
            for m in range(i, ni):
                d2 = (m - i) ** 2
                nr[d2] += 1
                bb[d2] += f[i, j] * f[m, j]
            for n in range(j + 1, nj):
                for m in range(ni):
                    d2 = (m - i) ** 2 + (n - j) ** 2
                    nr[d2] += 1
                    bb[d2] += f[i, j] * f[m, n]
    return nr, bb


def test_pair_sums_and_counts_match_wrfdas_pair_loop():
    rng = np.random.default_rng(5)
    f = rng.standard_normal((9, 7))
    nr, bb = _brute_pairs(f)
    nn = 9 * 9 + 7 * 7
    np.testing.assert_array_equal(gb._pair_counts(9, 7, nn), nr)
    np.testing.assert_allclose(gb._pair_sums(f, nn), bb, rtol=0, atol=1e-12 * np.abs(bb).max())


def test_scale_length_of_a_gaussian_correlated_field():
    # A field smoothed with a Gaussian of known scale: the fitted length is that scale.
    rng = np.random.default_rng(7)
    ni = nj = 64
    L = 4.0
    k = np.fft.fftfreq(2 * ni)[:, None] ** 2 + np.fft.fftfreq(2 * nj)[None, :] ** 2
    filt = np.exp(-((2 * np.pi) ** 2) * k * L * L / 4.0)
    fields = []
    for _ in range(6):
        w = rng.standard_normal((2 * ni, 2 * nj))
        fields.append(np.real(np.fft.ifft2(np.fft.fft2(w) * filt))[:ni, :nj])
    ml, sl = gb._scale_length(fields)
    # gen_be's fit: correlation exp(-d^2 / (8 s^2)) has yr = d/s.  The smoothed
    # white noise has correlation exp(-d^2 / (2 L^2)), i.e. s = L/2.
    assert sl == pytest.approx(L / 2.0, rel=0.1)


def test_diags_round_trip_spikes_and_smoothing():
    sl = np.array([2.123456789, 2.2, 9.0, 2.4, 2.5, 2.6])
    ml = 1.0 / sl
    out = gb._diags_scale(ml, sl, 0)
    assert out[0] == float("2.1234568e+00")
    assert out[2] == 0.5 * (2.2 + 2.4)  # the spike is replaced by its neighbours' mean
    sm = gb._diags_scale(ml, sl, 2)
    assert sm[0] == out[0] and sm[-1] == out[-1]
    one = out[1:-1] + 0.25 * ((out[:-2] + out[2:]) - 2.0 * out[1:-1])
    one = np.concatenate([[out[0]], one, [out[-1]]])
    two = one[1:-1] + 0.25 * ((one[:-2] + one[2:]) - 2.0 * one[1:-1])
    np.testing.assert_array_equal(sm[1:-1], two)


def _smooth(rng, ni, nj, L=3.0):
    k = np.fft.fftfreq(ni)[:, None] ** 2 + np.fft.fftfreq(nj)[None, :] ** 2
    return np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((ni, nj))) * np.exp(-((2 * np.pi) ** 2) * k * L * L / 4)))


def _planted_samples(n=8, ni=20, nj=16, nk=5, seed=11):
    """Smooth samples with chi = 0.3 psi, T = G psi and ps = w . psi plus small unbalanced parts.

    psi's vertical modes have variances 1, 0.3, 0.1, 1e-6, 1e-6: stage 2
    drops the mode at which the cumulative variance crosses 1 - 1e-6 and
    every one after it (here the two smallest), so the planted balance is
    recovered on the others.
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((nk, nk)) * 1e-6
    w = rng.standard_normal(nk) * 1e-4
    q, _ = np.linalg.qr(rng.standard_normal((nk, nk)))
    sd = np.sqrt(np.array([1.0, 0.3, 0.1, 1e-6, 1e-6]))

    def field3():
        return np.stack([_smooth(rng, ni, nj) for _ in range(nk)], axis=2)

    out = []
    for _ in range(n):
        psi = 1e6 * np.einsum("km,ijm->ijk", q * sd, field3())
        s = {
            "psi": psi,
            "chi": 0.3 * psi + 1e3 * field3(),
            "t": np.einsum("kl,ijl->ijk", G, psi) + 1e-4 * field3(),
            "rh": 0.1 * field3(),
            "ps": np.einsum("l,ijl->ij", w, psi) + 1e-3 * _smooth(rng, ni, nj),
            "height": np.zeros((ni, nj, nk)),
            "lat": np.zeros((ni, nj)),
        }
        out.append(s)
    return out, G, w


def test_stage2_recovers_a_planted_balance():
    samples, G, w = _planted_samples()
    pert = gb._stage1(samples, ("psi", "chi", "t", "rh", "ps"))
    reg = gb._stage2(pert, 5)
    np.testing.assert_allclose(reg["psi_chi"], 0.3, rtol=1e-2)
    for s in pert:
        t_fit = np.einsum("kl,ijl->ijk", reg["psi_t"][:, :, 0], s["psi"])
        ps_fit = np.einsum("l,ijl->ij", reg["psi_ps"][:, 0], s["psi"])
        t_bal = np.einsum("kl,ijl->ijk", G, s["psi"])
        ps_bal = np.einsum("l,ijl->ij", w, s["psi"])
        assert np.max(np.abs(t_fit - t_bal)) <= 0.02 * np.max(np.abs(t_bal))
        assert np.max(np.abs(ps_fit - ps_bal)) <= 0.02 * np.max(np.abs(ps_bal))


def test_gen_be_writes_a_file_woofs_3dvar_reads(tmp_path):
    samples, _, _ = _planted_samples(n=4, ni=18, nj=14, nk=5)
    for cv in (5, 7):
        if cv == 7:
            for s in samples:
                s["u"], s["v"] = s["psi"] * 1e-6, s["chi"] * 1e-6
        r = gb.gen_be(samples, gb.GenBeConfig(cv_options=cv))
        p = tmp_path / f"be{cv}.dat"
        sha = bd.write_be_dat(p, r.be)
        assert hashlib.sha256(p.read_bytes()).hexdigest() == sha
        back = bd.read_be_dat(p, cv)
        np.testing.assert_array_equal(back.eigen[0].eval_glo, r.be.eigen[0].eval_glo)
        assert back.eigen_ps.name == ("ps_u" if cv == 5 else "ps")
        assert np.all(np.diff(back.eigen[0].eval_glo) <= 0)
        b = bd.setup_be_regional(back, ix=18, jy=14, kz=5, ds=30000.0, config=bd.BeConfig(cv_options=cv))
        assert b.cv_mz[4] == 1 and all(1 <= m <= 5 for m in b.cv_mz)
        assert all(np.all(np.isfinite(s.rf_alpha)) for s in b.v)


def test_refusals():
    samples, _, _ = _planted_samples(n=2)
    for kw in ({"bin_type": 1}, {"cv_options": 6}, {"num_passes": 1}, {"stride": 2}):
        with pytest.raises(NotImplementedError):
            gb.gen_be(samples, gb.GenBeConfig(**kw))
    with pytest.raises(ValueError):
        gb.gen_be(samples[:1])
    with pytest.raises(NotImplementedError):
        gb.standard_fields(_synthetic_history(), 6)


# --------------------------------------------------------------------------- oracle gate (GV2)


def _close(a, b, rtol, atol, what):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"{what}: shape {a.shape} != {b.shape}"
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=what)


def _sign_aligned(a, b):
    """Columns of ``a`` with each sign set by the largest-|component| of the matching ``b`` column."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    if a.ndim == 3:  # (nk, nk, bins)
        return np.stack([_sign_aligned(a[:, :, i], b[:, :, i]) for i in range(a.shape[2])], axis=2)
    idx = np.argmax(np.abs(b), axis=0)
    cols = np.arange(b.shape[1])
    sign = np.sign(a[idx, cols] * b[idx, cols])
    sign[sign == 0] = 1.0
    return a * sign


def _oracle_dir() -> Path:
    d = os.environ.get("WOOF_GENBE_DIR")
    if not d:
        pytest.skip("WOOF_GENBE_DIR is not set (gen_be_fc/ plus WRFDA's cv5 and cv7 be.dat)")
    assert (Path(d) / "gen_be_fc").is_dir(), f"WOOF_GENBE_DIR={d} has no gen_be_fc/"
    return Path(d)


def _oracle_be(d: Path, cv: int) -> Path:
    """``oracle_cvN/be.dat`` (staged) or ``run/cvN/be.dat`` (run_gen_be.sh's own layout)."""
    for rel in (f"oracle_cv{cv}/be.dat", f"run/cv{cv}/be.dat"):
        if (d / rel).exists():
            return d / rel
    raise AssertionError(f"WOOF_GENBE_DIR={d} has neither oracle_cv{cv}/be.dat nor run/cv{cv}/be.dat")


@lru_cache(maxsize=None)
def _woof_be(d: str, cv: int):
    fc = Path(d) / "gen_be_fc"
    samples = []
    t = datetime(2008, 2, 4, 0)
    while t <= datetime(2008, 2, 8, 12):
        fn = t.strftime("wrfout_d01_%Y-%m-%d_%H:00:00")
        long = gb.read_wrf_history(fc / (t - timedelta(hours=24)).strftime("%Y%m%d%H") / fn)
        short = gb.read_wrf_history(fc / (t - timedelta(hours=12)).strftime("%Y%m%d%H") / fn)
        samples.append(gb.nmc_perturbation(long, short, cv))
        t += timedelta(hours=12)
    return gb.gen_be(samples, gb.GenBeConfig(cv_options=cv))


@pytest.mark.parametrize("cv", [5, 7])
def test_gv2_woof_gen_be_reproduces_wrfdas_be_dat(cv, tmp_path):
    d = _oracle_dir()
    oracle_path = _oracle_be(d, cv)
    pinned = json.loads(MANIFEST.read_text())["gen_be"][f"oracle_cv{cv}_be_dat_sha256"]
    assert hashlib.sha256(oracle_path.read_bytes()).hexdigest() == pinned
    o = bd.read_be_dat(oracle_path, cv)
    r = _woof_be(str(d), cv)
    bd.write_be_dat(tmp_path / "be.dat", r.be)
    w = bd.read_be_dat(tmp_path / "be.dat", cv)
    assert (w.ni, w.nj, w.nk, w.num_bins, w.num_bins2d) == (o.ni, o.nj, o.nk, o.num_bins, o.num_bins2d)
    np.testing.assert_array_equal(w.bin, o.bin)
    assert set(w.regcoeff) == set(o.regcoeff)
    # header fields
    for n in ("bin_type", "cv_options", "lat_min", "lat_max", "binwidth_lat", "hgt_min", "hgt_max", "binwidth_hgt"):
        assert getattr(w, n) == getattr(o, n), n
    assert (w.eta_be is None) == (o.eta_be is None)
    if o.eta_be is not None:
        assert w.eta_be.shape == o.eta_be.shape
        np.testing.assert_allclose(w.eta_be, o.eta_be, rtol=1e-12)
    np.testing.assert_array_equal(w.bin2d, o.bin2d)
    # regressions (GV2: 1e-6 of the array's peak, measured 1.0e-7): the peak-normalised bound as atol,
    # plus an elementwise rtol of 5e-6 (small entries differ by up to 1.9e-6 relative at 5e-8 of the peak)
    for k in o.regcoeff:
        _close(w.regcoeff[k], o.regcoeff[k], rtol=5e-6, atol=1e-7 * np.abs(o.regcoeff[k]).max(), what=f"regcoeff {k}")
    for eo, ew in zip(list(o.eigen) + [o.eigen_ps], list(w.eigen) + [w.eigen_ps], strict=True):
        assert ew.name == eo.name
        _close(ew.eval_glo, eo.eval_glo, rtol=1e-6, atol=0.0, what=f"{eo.name} eval_glo")
        _close(ew.eval_loc, eo.eval_loc, rtol=1e-6, atol=0.0, what=f"{eo.name} eval_loc")
        _close(_sign_aligned(ew.evec_glo, eo.evec_glo), eo.evec_glo, rtol=0.0, atol=1e-6, what=f"{eo.name} evec_glo")
        _close(_sign_aligned(ew.evec_loc, eo.evec_loc), eo.evec_loc, rtol=0.0, atol=1e-6, what=f"{eo.name} evec_loc")
    # stage 4: smoothed lengths to 1e-6 (measured 7.4e-8 for cv5, identical for cv7)
    for a, b in zip(list(w.rf_lengthscale) + [w.rf_lengthscale_ps], list(o.rf_lengthscale) + [o.rf_lengthscale_ps],
                    strict=True):
        _close(a, b, rtol=1e-6, atol=0.0, what="rf_lengthscale")
    # ... and the raw per-mode scale length, from the oracle's sl_print files when its run tree is present
    sl_dir = oracle_path.resolve().parent / "working" / "gen_be_stage4_regional.1"
    if sl_dir.is_dir():
        names = ("psi", "chi_u", "t_u", "rh", "ps_u") if cv == 5 else ("u", "v", "t", "rh", "ps")
        for name in names:
            raw = r.scale_length_raw[name]
            ref = []
            for kk in range(1, raw.size + 1):
                txt = (sl_dir / f"dir.{name}{kk}" / f"sl_print.{name}.{kk:02d}").read_text().split()
                assert int(txt[0]) == kk
                ref.append(float(txt[2]))
            # sl_print is written with e20.8: compare after the same 8-digit rounding
            _close(np.array([gb._fortran_e20_8(v) for v in raw]), np.array(ref), rtol=1e-6, atol=0.0,
                   what=f"raw sl {name}")
