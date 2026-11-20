"""woof.da.var.hybrid: WRFDA's alpha control variable, hybrid 3DEnVar (plan V3.1).

What is asserted, and why it is enough:

- **Weights:** ``jb_factor`` is ``da_setup_background_errors``'s in all three
  branches (hybrid ``je/(je-1)``, full ensemble ``-999`` with no static
  control variables, no ensemble ``1``), and ``J = Jb + Jo + Je`` with each
  term its weighted half square.
- **Vertical localisation (``alpha_vertloc_opt = 2``):** the modes
  reconstruct ``da_get_alpha_vertloc``'s log-p Gaussian correlation, unit
  diagonal, total variance kept.  ``be.vertloc.dat`` and ``ep`` files round
  trip (``ep_format`` 1 and 11).
- **Adjoint (O3) and gradient (O4):** ``<U v, x> = <v, U^T x>`` to 1e-13 and
  ``da_check_gradient``'s Taylor ratio within 1e-6 of 1, for hybrid and
  full-ensemble weights, with and without vertical localisation.
- **Single observation:** the minimiser's increment is the closed form
  ``B h d / (h^T B h + r^2)`` with ``B = U_s U_s^T / jb_factor + U_e U_e^T /
  je_factor``, and the ensemble part is the localised ensemble covariance:
  ``sum_n x'_n(x) x'_n(x0) C(x, x0)`` with ``C`` the alpha correlation, which
  is about 1 at the observation and decays with ``alpha_corr_scale``.
- **Device tiers:** the oracle tier's U, U^T, J and grad J are the
  reference's words on CUDA and Metal; the production tier is within
  binary32 of them and its words, in fast and strict child processes, hash
  to a constant both backends reproduce.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

from woof.da.var import hybrid as hy
from woof.da.var.cost import SparseObsTerm, check_gradient
from woof.da.var.minimise import minimise_cg

from test_var_cost import _shapes, case

REPO = Path(__file__).resolve().parents[1]
NX, NY, NZ, NE = 13, 11, 6, 5

#: SHA-256 of the production tier's xa (U v) and cv (U^T x) words for
#: :func:`_dev_case` (je_factor 2 and 1); measured on CUDA (A6000, fast and
#: strict) and Metal (strict).
PRODUCTION_DIGESTS = {
    "production2": "9ce5c74fca38e30e36bf2da6fd8b97fedbadba2a38b5b6999377f941dbeffc3f",
    "production1": "4023ced4bc5b6213c5e70c6403cd4d3719d2e420ac660d7287daa5cc50a4cbad",
}


def _znw(nz):
    return np.linspace(1.0, 0.0, nz + 1) ** 1.3


def _ensemble(scale: dict, seed=21, ne=NE):
    """Smooth random members with ``scale``'s standard deviation per field."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in hy.FIELDS:
        shape = (ne, NX, NY) if k == "ps" else (ne, NX, NY, NZ)
        a = rng.standard_normal(shape)
        for ax in (1, 2):
            a = (np.roll(a, 1, ax) + 2.0 * a + np.roll(a, -1, ax)) / 4.0
        # the spread rounded to 6 digits: the words cannot depend on the host's pairwise sum
        out[k] = scale[k] * a / float(f"{a.std():.6g}")
    return out


def _static_scale(b, xb):
    """The static ``U``'s increment spread per field, so the ensemble's is comparable."""
    from woof.da.var.cost import ReferenceU

    u = ReferenceU(b, xb)
    xa = u.apply(np.random.default_rng(1).standard_normal(u.cv_size))["xa"]
    sd = {"u": xa["u"].std(), "v": xa["v"].std(), "t": xa["t"].std(), "q": xa["q"].std(), "ps": xa["psfc"].std()}
    return {k: float(f"{v:.3g}") for k, v in sd.items()}


def _setup(je_factor=2.0, opt=2, scale=9.0, obs=None):
    b, xb, sobs = case()
    xb = dict(xb, ds=3000.0, znw=_znw(NZ), ptop=5000.0)
    ep = hy.Ensemble.from_members(_ensemble(_static_scale(b, xb)))
    cfg = hy.HybridConfig(je_factor=je_factor, alpha_vertloc_opt=opt, alpha_corr_scale=scale)
    return b, xb, ep, cfg, (sobs if obs is None else obs)


# ------------------------------------------------------------ weights, files


def test_jb_factor_branches():
    assert hy.jb_factor_for(0, 2.0) == 1.0
    assert hy.jb_factor_for(10, 2.0) == 2.0
    assert hy.jb_factor_for(10, 4.0) == 4.0 / 3.0
    assert 1.0 / hy.jb_factor_for(10, 4.0) + 1.0 / 4.0 == pytest.approx(1.0, abs=1e-15)
    assert hy.jb_factor_for(10, 1.0) == -999.0 and hy.jb_factor_for(10, 0.5) == -999.0


def test_alpha_vertloc_is_the_log_p_correlation():
    val, evec = hy.alpha_vertloc(_znw(20), 5000.0)
    rho = (evec * val**2) @ evec.T
    assert np.allclose(np.diag(rho), 1.0, atol=1e-12)
    assert np.allclose(rho, rho.T, atol=1e-12) and np.all(rho > 0.0)
    assert np.all(np.diff(rho[10, 10:]) < 0.0)  # decays away from the level
    assert np.all(np.diff(val) <= 0.0) and val.size == 20


def test_vertloc_and_ep_files_round_trip(tmp_path):
    val, evec = hy.alpha_vertloc(_znw(NZ), 5000.0)
    hy.write_vertloc(tmp_path / "be.vertloc.dat", val, evec)
    v2, e2 = hy.read_vertloc(tmp_path / "be.vertloc.dat")
    np.testing.assert_array_equal(v2, val)
    np.testing.assert_array_equal(e2, evec)
    perts = _ensemble(dict.fromkeys(hy.FIELDS, 1.0))
    hy.write_ep(tmp_path / "ep", perts, ep_format=1)
    back = hy.read_ep(tmp_path / "ep", NE, ep_format=1, shape=(NX, NY, NZ))
    for k in hy.FIELDS:
        np.testing.assert_array_equal(back[k], perts[k])
    hy.write_ep(tmp_path / "ep4", perts, ep_format=11, endian="<")
    back = hy.read_ep(tmp_path / "ep4", NE, ep_format=11)
    for k in hy.FIELDS:
        np.testing.assert_array_equal(back[k], perts[k].astype(np.float32))
    assert sorted(p.name for p in (tmp_path / "ep").iterdir())[:5] == ["ps.e001", "ps.e002", "ps.e003", "ps.e004",
                                                                      "ps.e005"]


def test_perturbations_are_scaled_anomalies():
    m = _ensemble(dict.fromkeys(hy.FIELDS, 1.0))
    ep = hy.Ensemble.from_members(m)
    assert ep.ne == NE and ep.shape == (NX, NY, NZ)
    mean = m["t"].mean(axis=0)
    np.testing.assert_allclose(ep.t[..., 2], (m["t"][2] - mean) / np.sqrt(NE - 1), atol=1e-12)
    assert np.abs(ep.u.sum(axis=-1)).max() < 1e-12


def test_setup_alpha_options():
    a0 = hy.setup_alpha(NE, jy=NY, kz=NZ, ds=3000.0, config=hy.HybridConfig(alpha_vertloc_opt=0))
    assert a0.mz == 1 and np.all(a0.evec == 1.0)
    a2 = hy.setup_alpha(NE, jy=NY, kz=NZ, ds=3000.0, config=hy.HybridConfig(), znw=_znw(NZ), ptop=5000.0)
    assert 1 <= a2.mz <= NZ
    np.testing.assert_allclose(a2.rf_lengthscale, 200.0 * 1000.0 / 3000.0)
    with pytest.raises(ValueError):
        hy.setup_alpha(NE, jy=NY, kz=NZ, ds=3000.0, config=hy.HybridConfig(alpha_vertloc_opt=1))
    with pytest.raises(NotImplementedError):
        hy.HybridConfig(alphacv_method=1).check()


# ------------------------------------------------------------ reference tier


def _xa(rng):
    return {k: rng.standard_normal((NX, NY) if k == "psfc" else (NX, NY, NZ))
            for k in ("u", "v", "t", "q", "psfc", "p", "rho")}


def _serial(a, b):
    acc = 0.0
    for x, y in zip(np.ravel(a), np.ravel(b)):
        acc = acc + float(x) * float(y)
    return acc


@pytest.mark.parametrize("je_factor,opt", [(2.0, 2), (1.0, 2), (4.0, 0)])
def test_hybrid_u_adjoint(je_factor, opt):
    b, xb, ep, cfg, obs = _setup(je_factor, opt)
    cost = hy.hybrid_cost(b, xb, obs, ep, cfg)
    u = cost.u
    assert u.cv_size == u.cv_size_jb + NX * NY * u.alpha.mz * NE
    assert (u.cv_size_jb == 0) == (je_factor <= 1.0)
    rng = np.random.default_rng(3)
    v = rng.standard_normal(u.cv_size)
    x = _xa(rng)
    fwd = u.apply(v)["xa"]
    lhs = sum(_serial(fwd[k], x[k]) for k in x)
    rhs = _serial(v, u.adjoint(x)["cv"])
    assert abs(lhs - rhs) <= 1e-13 * abs(lhs)


@pytest.mark.parametrize("je_factor,opt", [(2.0, 2), (1.0, 2), (4.0, 0)])
def test_hybrid_check_gradient_and_j_terms(je_factor, opt):
    b, xb, ep, cfg, obs = _setup(je_factor, opt)
    cost = hy.hybrid_cost(b, xb, obs, ep, cfg)
    rng = np.random.default_rng(7)
    v = rng.uniform(size=cost.cv_size) - 0.5
    j = cost.calculate_j(v, None, iter_=1)
    nb = cost.cv_size_jb
    jbf = hy.jb_factor_for(NE, je_factor)
    assert j.jb == (jbf * 0.5 * _serial(v[:nb], v[:nb]) if nb else 0.0)
    assert j.je == je_factor * 0.5 * _serial(v[nb:], v[nb:])
    assert j.total == j.jb + j.jo + j.je
    out = check_gradient(cost, v)
    print(f"je_factor {je_factor} opt {opt}: best T1 {out['best']!r} ({out['digits']} digits)")
    assert abs(out["best"] - 1.0) <= 1e-6, out


def _single_ob(i0, j0, k0, d=1.0, err=0.5):
    shapes = _shapes(NX, NY, NZ)
    idx = np.ravel_multi_index((i0, j0, k0), (NX, NY, NZ))
    return SparseObsTerm(shapes, np.array([["t"]], dtype=object), np.array([[idx]]), np.ones((1, 1)),
                         np.array([d]), np.array([err]), groups=[("sound", [("t", np.arange(1))])])


def test_single_ob_hybrid_increment_structure():
    i0, j0, k0 = 6, 5, 2
    obs = _single_ob(i0, j0, k0)
    b, xb, ep, cfg, _ = _setup(2.0, 2, scale=3.0, obs=obs)
    cost = hy.hybrid_cost(b, xb, obs, ep, cfg)
    u = cost.u
    r = minimise_cg(cost, ntmax=20, eps=1e-12)
    xa = u.apply(r.xhat)["xa"]
    # closed form: B h d / (h^T B h + r^2), B = U W^-1 U^T
    h = {k: np.zeros_like(v) for k, v in _xa(np.random.default_rng(0)).items()}
    h["t"][i0, j0, k0] = 1.0
    uth = u.adjoint(h)["cv"]
    nb = u.cv_size_jb
    w = np.concatenate([uth[:nb] / cost.jb_factor, uth[nb:] / cost.je_factor])
    bh = u.apply(w)["xa"]
    hbh = bh["t"][i0, j0, k0]
    gain = 1.0 / (hbh + 0.5**2)
    for k in ("u", "v", "t", "q", "psfc"):
        np.testing.assert_allclose(xa[k], gain * bh[k], rtol=0, atol=1e-9 * np.abs(bh[k]).max())
    # the ensemble part: sum_n x'_n(x) x'_n(x0) C(x, x0) / je_factor
    we = np.concatenate([np.zeros(nb), uth[nb:]])
    ens = u.apply(we)["xa"]
    ue = hy.HybridReferenceU(b, u.alpha, ep, u.xb, u.jb_factor)
    a = u.alpha
    delta = np.zeros((NX, NY, NZ))
    delta[i0, j0, k0] = 1.0
    vv = hy.ref.vvtovp_adj(a.evec, a.val, delta, a.mz)
    vv = hy.ref.transform_through_rf_adj(vv, a.rf_alpha, xb["grid_box_area"], a.rf_passes)
    vv = hy.ref.transform_through_rf(vv, a.rf_alpha, xb["grid_box_area"], a.rf_passes)
    corr = hy.ref.vvtovp(a.evec, a.val, vv, NZ)
    cov = np.zeros((NX, NY, NZ))
    for n in range(NE):
        cov = cov + ue.ep.t[..., n] * ue.ep.t[i0, j0, k0, n]
    np.testing.assert_allclose(ens["t"], cov * corr, rtol=0, atol=1e-12 * np.abs(cov * corr).max())
    # the alpha correlation: ~1 at the observation, decaying with alpha_corr_scale (1 grid length)
    c0 = corr[i0, j0, k0]
    far = corr[:, j0, k0][np.abs(np.arange(NX) - i0) >= 5]
    print(f"single ob: C(x0,x0) {c0:.4f}, C at 5+ grid lengths <= {np.abs(far).max():.2e}, "
          f"ensemble share of h^T B h {ens['t'][i0, j0, k0] / cost.je_factor / hbh:.3f}")
    assert 0.8 < c0 < 1.2
    assert np.abs(far).max() < 0.1 * c0
    # the static part fills the rest of h^T B h
    ws = np.concatenate([uth[:nb], np.zeros(uth.size - nb)])
    st = u.apply(ws)["xa"]["t"][i0, j0, k0]
    assert st / cost.jb_factor + ens["t"][i0, j0, k0] / cost.je_factor == pytest.approx(hbh, rel=1e-12)


def test_precondition_cg_blocks_include_alpha():
    b, xb, ep, cfg, obs = _setup(2.0, 2)
    cost = hy.hybrid_cost(b, xb, obs, ep, cfg)
    r = minimise_cg(cost, ntmax=5, precondition_cg=True)
    assert r.j_final < r.j[0]
    b, xb, ep, cfg, obs = _setup(1.0, 2)
    cost = hy.hybrid_cost(b, xb, obs, ep, cfg)
    with pytest.raises(NotImplementedError):
        minimise_cg(cost, ntmax=5, precondition_cg=True)


# ------------------------------------------------------------ device tiers


def _dev_case(je_factor):
    b, xb, ep, cfg, obs = _setup(je_factor, 2)
    rng = np.random.default_rng(5)
    ref_cost = hy.hybrid_cost(b, xb, obs, ep, cfg)
    v = rng.standard_normal(ref_cost.cv_size)
    x = _xa(rng)
    return b, xb, ep, cfg, obs, ref_cost, v, x


def _device_words(je_factor, precision):
    """Device U v, U^T x, J and grad J against the reference: (max rel diffs, sha of the words)."""
    import cupy as cp

    b, xb, ep, cfg, obs, rc, v, x = _dev_case(je_factor)
    dc = hy.hybrid_cost(b, xb, obs, ep, cfg, tier=precision)
    pairs = []
    rs, ds = rc.u.apply(v, stages=True), dc.u.apply(cp.asarray(v, dtype=dc.dtype), stages=True)
    for k in ("alpha_vv", "alpha_vp"):
        pairs.append((f"U.{k}", ds[k], rs[k].reshape(ds[k].shape) if k == "alpha_vp" else
                      np.moveaxis(rs[k], 3, 2).reshape(ds[k].shape)))
    pairs += [(f"U.xa.{k}", ds["xa"][k], rs["xa"][k]) for k in ("u", "v", "t", "q", "psfc", "p", "rho")]
    ra = rc.u.adjoint(x)
    da = dc.u.adjoint({k: cp.asarray(a, dtype=dc.dtype) for k, a in x.items()})
    pairs.append(("UT.cv", da["cv"], ra["cv"]))
    jr, gr = rc.value_and_gradient(v)
    jd, gd = dc.value_and_gradient(dc.asarray(v))
    pairs.append(("grad", gd, gr))
    report, words = {}, hashlib.sha256()
    for name, a, want in pairs:
        a = np.asarray(a.get())
        want = np.asarray(want)
        d = np.abs(a.astype(np.float64).reshape(want.shape) - want)
        report[name] = float(d.max()) / max(float(np.abs(want).max()), 1e-300)
        if name.startswith("U.xa.") or name == "UT.cv":  # J and grad also carry the host's Jo
            words.update(np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<")).tobytes())
    report["J"] = abs(jd.total - jr.total) / abs(jr.total)
    return report, words.hexdigest()


@requires_gpu
@pytest.mark.parametrize("je_factor", [2.0, 1.0])
def test_oracle_tier_is_the_reference_word_for_word(je_factor):
    report, _ = _device_words(je_factor, "oracle")
    bad = {k: v for k, v in report.items() if v != 0.0}
    assert not bad, bad


@requires_gpu
@pytest.mark.parametrize("je_factor", [2.0, 1.0])
def test_production_tier_is_within_binary32(je_factor):
    report, _ = _device_words(je_factor, "production")
    bad = {k: v for k, v in report.items() if not v <= 1e-5}
    assert not bad, bad


_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from test_var_hybrid import _device_words
out = {}
for je in (2.0, 1.0):
    r, _ = _device_words(je, "oracle")
    out[f"oracle{int(je)}"] = max(r.values())
    _, d = _device_words(je, "production")
    out[f"production{int(je)}"] = d
print("RESULT " + json.dumps(out))
"""


@requires_gpu
@pytest.mark.parametrize("mode", ["fast", "strict"])
def test_words_in_fast_and_strict_child_processes(mode):
    env = dict(os.environ, GPUWM_NUMERICS=mode)
    proc = subprocess.run([sys.executable, "-c", _CHILD, str(Path(__file__).parent)], env=env,
                          capture_output=True, text=True, timeout=1800, cwd=REPO)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = [s for s in proc.stdout.splitlines() if s.startswith("RESULT ")][-1]
    got = json.loads(line[len("RESULT "):])
    print(f"{mode}: {got}")
    assert got["oracle2"] == 0.0 and got["oracle1"] == 0.0, got
    from woof import backend

    if mode == "strict" or backend.active() == "cuda":
        assert {k: got[k] for k in PRODUCTION_DIGESTS} == PRODUCTION_DIGESTS, got
