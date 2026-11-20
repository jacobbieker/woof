"""woof.da.var.varbc: WRFDA's variational bias correction (plan V4.4, section 13; gates GR-V1, V2, V4, V5).

What is asserted, and why it is enough:

- **GR-V1:** ``<H' v, y> = <v, H'^T y>`` to 1e-13 through ``VarCost``'s
  whole chain with the VarBC block in ``cv`` (state part and bias part).
- **GR-V2:** ``da_check_gradient``'s Taylor ratio within 1e-6 of 1 for
  ``J = Jb + Jo + Jp``, for both ``Jp`` forms.
- **GR-V5:** with the state held still, a known bias (constant plus skin
  temperature) is recovered: CG's coefficients are the dense least-squares
  solution with the ``Jp`` prior, and that solution is the truth to within
  the noise; ``VARBC.out`` reads back as written.
- **GR-V4:** the production tier's CG words with radiances and VarBC are one
  digest under ``GPUWM_NUMERICS=fast`` and ``strict``, the same on Metal
  and CUDA (the radiance and VarBC arithmetic is host binary64).
- The pieces are WRFDA's: ``da_predictor_crtm`` on a hand column, the
  cold-start mode, the preconditioner ``vtox = H^-1/2`` (from a Jacobi that
  agrees with LAPACK to rounding), and a block-free ``VarCost`` unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu

from woof.da.var import radiance as rad
from woof.da.var import varbc as vbc
from woof.da.var.cost import VarCost, check_gradient
from woof.da.var.minimise import minimise, minimise_cg

from test_var_cost import case
from test_var_radiance import WHEN, _LinearCRTM, _batch, _proj, _surface, _xb_columns

REPO = Path(__file__).resolve().parents[1]

#: SHA-256 of the production tier's CG words (J, |g|, step per iteration,
#: final J and |g|, xhat as binary32) for the synthetic case of :func:`_digest_case`;
#: measured on Metal and CUDA (A6000), fast and strict.
PRODUCTION_VARBC_DIGEST = "9e248538167825e19b5e1d7d8deb998f8ed8a9b73d570b0b9f081e2159c52a39"


def _serial(a, b):
    acc = 0.0
    for x, y in zip(np.ravel(a), np.ravel(b)):
        acc = acc + float(x) * float(y)
    return acc


def _extras(b, rng, nz):
    """The columns and record ``predictors`` reads, for a synthetic batch."""
    n = b.n
    pf = np.tile(np.r_[20.0, np.geomspace(40.0, 1010.0, nz)], (n, 1)) * rng.uniform(0.97, 1.0, (n, 1))
    cols = SimpleNamespace(temperature_k=rng.uniform(210, 300, (n, nz)), mixing_ratio_gkg=rng.uniform(0.01, 15, (n, nz)),
                           skin_temperature_k=rng.uniform(270, 310, n), level_pressure_hpa=pf)
    b.extra = {"columns": cols, "record": SimpleNamespace(sensor_zenith=rng.uniform(0, 60, n), scanpos=np.zeros(n))}
    return b


def _varbc_term(rng, shape, nbatch_obs=(60, 25), jp_form="wrfda"):
    nx, ny, nz = shape
    batches = [_extras(_batch(rng, nx, ny, nz, n=nbatch_obs[0], nchan=3), rng, nz),
               _extras(_batch(rng, nx, ny, nz, n=nbatch_obs[1], nchan=1, sensor="ahi_himawari9"), rng, nz)]
    vb = vbc.VarBC({"abi_g16": vbc.template("abi_g16", range(7, 17)),
                    "ahi_himawari9": vbc.template("ahi_himawari9", range(7, 17))},
                   vbc.VarBCSettings(jp_form=jp_form))
    for b in batches:
        b.inv = b.inv - vb.bias(b)
    term = rad.RadianceTerm(batches, shape)
    return vb, vb.attach(term)


# ------------------------------------------------------------------ the pieces


def test_jacobi_agrees_with_lapack_and_is_a_decomposition():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((6, 6))
    a = a @ a.T + 0.1 * np.eye(6)
    lam, e = vbc.jacobi_eigh(a)
    np.testing.assert_allclose(np.sort(lam), np.linalg.eigvalsh(a), rtol=1e-12)
    np.testing.assert_allclose(e @ np.diag(lam) @ e.T, a, atol=1e-12)
    np.testing.assert_allclose(e.T @ e, np.eye(6), atol=1e-13)


def test_predictor_crtm_on_a_hand_column():
    pf = np.array([50.0, 200.0, 300.0, 700.0, 1000.0])
    t = np.array([220.0, 230.0, 260.0, 285.0])
    q = np.zeros(4)
    p = vbc.predictor_crtm(t, q, 295.0, pf)
    kth = 287.0 / 9.81
    assert p[0] == pytest.approx(kth * (260.0 * np.log(700 / 300) + 285.0 * np.log(1000 / 700)), rel=1e-14)
    assert p[1] == pytest.approx(kth * 220.0 * np.log(200 / 50), rel=1e-14)
    assert p[2] == 295.0 and p[3] == 0.0
    # below 1000 hPa missing: the hypsometric extension to 1000 hPa
    p2 = vbc.predictor_crtm(t, q, 295.0, pf * 0.95)
    assert p2[0] > kth * 285.0 * np.log(1000 / 950) > 0


def test_coldstart_mode_and_normalisation():
    rng = np.random.default_rng(2)
    b = _extras(_batch(rng, 6, 5, 7, n=200, nchan=3), rng, 7)
    b.inv = 2.03 + 0.01 * rng.standard_normal((200, 3))
    vb = vbc.VarBC({"abi_g16": vbc.template("abi_g16", range(7, 17))})
    bias = vb.bias(b)
    inst = vb.instruments["abi_g16"]
    for c in (8, 9, 10):
        assert inst.channel(c).param[0] == pytest.approx(2.0, abs=1e-12)  # the 0.1 K bin at 2.0
    np.testing.assert_allclose(bias, 2.0, atol=1e-12)  # cold: only the constant has a value
    pred = vb._pred[id(b)]
    assert pred[:, 0].tolist() == [1.0] * 200  # the constant has no spread and stays 1
    np.testing.assert_allclose(pred[:, 1:5].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(pred[:, 1:5].std(axis=0), 1.0, rtol=1e-12)


def test_no_blocks_leaves_varcost_unchanged():
    b, xb, conv = case()
    cost = VarCost.reference(b, xb, conv)
    term = rad.RadianceTerm([_batch(np.random.default_rng(3), *xb["t"].shape, n=10)], xb["t"].shape)
    with_rad = VarCost.reference(b, xb, [conv, term])
    assert cost.cv_size == with_rad.cv_size == cost.cv_u
    assert cost.calculate_j(np.ones(cost.cv_size)).jp == 0.0


def test_varbc_file_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    vb, term = _varbc_term(rng, (7, 6, 5))
    vb.update(term, rng.standard_normal(term.control_size))
    path = tmp_path / "VARBC.out"
    vb.write(path)
    back = vbc.read_varbc(path)
    for sensor, inst in vb.instruments.items():
        got = back[vbc.VARBC_IDS[sensor]]
        assert got.nchanl == inst.nchanl and got.nbgerr.tolist() == inst.nbgerr.tolist()
        for c in inst.channels:
            g = got.channel(c.channel)
            assert g.pred_use.tolist() == c.pred_use.tolist()
            np.testing.assert_allclose(g.param[c.pred_use[c.ipred] > 0], np.round(c.param, 3)[c.pred_use[c.ipred] > 0],
                                       atol=5e-4)
    # nobs >= varbc_nobsmin turned the observed cold predictors warm
    assert (vb.instruments["abi_g16"].channel(8).pred_use[:5] == 1).all()
    assert (vb.instruments["abi_g16"].channel(7).pred_use[:5] == 0).all()


# ------------------------------------------------------------------ GR-V1, GR-V2


@pytest.mark.parametrize("jp_form", ["wrfda", "matrix"])
def test_chain_adjoint_and_gradient_with_varbc(jp_form):
    b, xb, conv = case()
    rng = np.random.default_rng(6)
    vb, term = _varbc_term(rng, xb["t"].shape, jp_form=jp_form)
    cost = VarCost.reference(b, xb, [conv, term])
    assert cost.cv_size == cost.cv_u + term.control_size == cost.cv_u + 3 * 5 + 5
    # errors scaled to the increments U makes
    scale = float(np.std(term.tl(cost.u.apply(rng.standard_normal(cost.cv_u))["xa"])))
    for bt in term.batches:
        bt.error = bt.error * scale
    term.err = np.concatenate([bt.error.ravel() for bt in term.batches])
    cost = VarCost.reference(b, xb, [conv, term])
    v = rng.standard_normal(cost.cv_size)
    y = [rng.standard_normal(conv.d.size), rng.standard_normal(term.m)]
    hv = cost.vtoy(v)
    lhs = _serial(hv[0], y[0]) + _serial(hv[1], y[1])
    rhs = _serial(v, cost.vtoy_adj(y))
    rel = abs(lhs - rhs) / abs(lhs)
    print(f"GR-V1 ({jp_form}): rel={rel:.3e}")
    assert rel <= 1e-13
    j = cost.calculate_j(v, None, 1)
    assert j.jp > 0.0 and j.total == j.jb + j.jo + j.jp
    chk = check_gradient(cost, rng.uniform(-0.5, 0.5, cost.cv_size))
    print(f"GR-V2 ({jp_form}): best={chk['best']!r} digits={chk['digits']}")
    assert abs(chk["best"] - 1.0) < 1e-6


# ------------------------------------------------------------------ GR-V5


class _StillU:
    """``x = s v`` for ``t`` and ``q`` only: with a tiny ``s`` the state cannot absorb the bias."""

    def __init__(self, shape, s):
        self.nx, self.ny, self.nz = shape
        self.n = self.nx * self.ny * self.nz
        self.cv_size = 2 * self.n
        self.cv_mz = (self.nz, self.nz, 0, 0, 0)
        self.s = s

    def apply(self, cv):
        cv = np.asarray(cv, dtype=np.float64)
        shape = (self.nx, self.ny, self.nz)
        return {"xa": {"t": self.s * cv[:self.n].reshape(shape), "q": 1e-3 * self.s * cv[self.n:].reshape(shape)}}

    def adjoint(self, xa):
        return {"cv": np.concatenate([self.s * np.ravel(xa["t"]), 1e-3 * self.s * np.ravel(xa["q"])])}


def _recovery_case(rng, n=300, noise=0.2, jp_form="wrfda"):
    nx, ny, nz = 24, 20, 12
    proj = _proj(nx, ny)
    xb = _xb_columns(nx, ny, nz)
    ramp = np.linspace(0.0, 1.0, nz)
    xb["t"] = xb["t"] + rng.uniform(-6, 6, (nx, ny, 1)) * ramp + rng.uniform(-6, 6, (nx, ny, 1)) * ramp[::-1]
    xb["q"] = xb["q"] * rng.uniform(0.3, 1.2, (nx, ny, 1))
    xb["lat"] = np.linspace(25, 45, nx * ny).reshape(nx, ny)
    xb["lon"] = np.linspace(-110, -80, nx * ny).reshape(nx, ny)
    surface = _surface(nx, ny, nz)
    surface["TSK"] = rng.uniform(275, 310, (nx, ny))
    # observations at random grid positions (llxy inverted by Newton)
    lats, lons = [], []
    for x, y in zip(rng.uniform(2, nx - 2, n), rng.uniform(2, ny - 2, n)):
        la, lo = 35.0, -95.0
        for _ in range(12):
            fx, fy = proj.llxy(la, lo)
            ex, ey = proj.llxy(la + 1e-4, lo), proj.llxy(la, lo + 1e-4)
            jm = np.array([[ex[0] - fx, ey[0] - fx], [ex[1] - fy, ey[1] - fy]]) / 1e-4
            d = np.linalg.solve(jm, [x - fx, y - fy])
            la, lo = la + d[0], lo + d[1]
        lats.append(la)
        lons.append(lo)
    rec = rad.RadianceObs(sensor="abi_g16", channels=[8, 9, 10], bt=np.zeros((n, 3)), lat=lats, lon=lons,
                          sensor_zenith=rng.uniform(10, 60, n), sensor_azimuth=0.0, sensor_scan=0.0,
                          solar_zenith=100.0, solar_azimuth=0.0, time=WHEN)
    settings = rad.RadianceQC(thinning=False)
    clean, _ = rad.radiance_term([rec], xb, surface, proj, _LinearCRTM(), settings=settings)
    b0 = clean.batches[0]
    true_bias = 1.2 + 0.05 * (b0.extra["columns"].skin_temperature_k - 290.0)
    rec.bt = b0.hx + true_bias[:, None] + noise * rng.standard_normal(b0.hx.shape)
    vb = vbc.VarBC({"abi_g16": vbc.template("abi_g16", range(7, 17))},
                    vbc.VarBCSettings(varbc_nbgerr=5, jp_form=jp_form))  # a weak prior: one cycle
    term, _ = rad.radiance_term([rec], xb, surface, proj, _LinearCRTM(), settings=settings, bias=vb.bias)
    vb.attach(term)
    return vb, term, true_bias, (nx, ny, nz)


@pytest.mark.parametrize("jp_form", ["wrfda", "matrix"])
def test_known_bias_is_recovered(jp_form):
    rng = np.random.default_rng(7)
    vb, term, true_bias, shape = _recovery_case(rng, jp_form=jp_form)
    b = term.batches[0]
    assert term.mask.all()
    cost = VarCost(_StillU(shape, 1e-6), [term])
    results, cvt = minimise(cost, max_ext_its=1, ntmax=200, eps=1e-12)
    block = cvt[cost.cv_u:]
    # the dense solution of the bias problem alone (the state is held still)
    k = term.control_size
    hp = np.zeros((term.m, k))
    for i in range(k):
        e = np.zeros(k)
        e[i] = 1.0
        hp[:, i] = cost.vtoy(np.r_[np.zeros(cost.cv_u), e])[0]
    w = 1.0 / term.err ** 2
    a = hp.T @ (w[:, None] * hp) + np.stack([term.varbc.jp_grad(np.eye(k)[i]) for i in range(k)], axis=1)
    dense = np.linalg.solve(a, hp.T @ (w * term.d))
    np.testing.assert_allclose(block, dense, rtol=1e-6, atol=1e-8)
    vb.update(term, block)
    pred = vb._pred[id(b)]
    for c, ch in enumerate((8, 9, 10)):
        bc = vb.instruments["abi_g16"].channel(ch)
        bias = pred[:, bc.ipred] @ bc.param
        err = bias - true_bias
        print(f"GR-V5 ({jp_form}) channel {ch}: bias error mean={err.mean():+.4f} rms={np.sqrt((err ** 2).mean()):.4f} K")
        assert abs(err.mean()) < 0.05 and np.sqrt((err ** 2).mean()) < 0.08


# ------------------------------------------------------------------ GR-V4


def _digest_case(precision):
    b, xb, conv = case()
    rng = np.random.default_rng(9)
    vb, term = _varbc_term(rng, xb["t"].shape)
    probe = VarCost.reference(b, xb, [conv, term])
    scale = float(np.std(term.tl(probe.u.apply(rng.standard_normal(probe.cv_u))["xa"])))
    for bt in term.batches:
        bt.error = bt.error * scale
    term.err = np.concatenate([bt.error.ravel() for bt in term.batches])
    cost = VarCost.reference(b, xb, [conv, term]) if precision == "reference" else \
        VarCost.device(b, xb, [conv, term], precision)
    r = minimise_cg(cost, ntmax=40, eps=1e-6)
    xhat = r.xhat if precision == "reference" else r.xhat.get()
    return r, np.asarray(xhat)


def _words(r, xhat) -> str:
    h = hashlib.sha256()
    for a in (r.j, r.g, r.step, [r.j_final, r.g_final]):
        h.update(struct.pack(f"<{len(a)}d", *a))
    h.update(np.ascontiguousarray(xhat, dtype="<f4").tobytes())
    return h.hexdigest()


def test_reference_cg_with_varbc_converges():
    r, x = _digest_case("reference")
    assert r.g_final < 1e-6 * r.g[0] * 1.01 or r.iterations == 40
    assert r.j_final < r.j[0]


_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from test_var_varbc import _digest_case, _words
r, x = _digest_case("production")
print("RESULT " + json.dumps({"digest": _words(r, x), "j": [v.hex() for v in r.j]}))
"""


@requires_gpu
def test_production_words_with_varbc_in_fast_and_strict_child_processes():
    got = {}
    for mode in ("fast", "strict"):
        env = dict(os.environ, GPUWM_NUMERICS=mode)
        proc = subprocess.run([sys.executable, "-c", _CHILD, str(Path(__file__).parent)], env=env,
                              capture_output=True, text=True, timeout=1800, cwd=REPO)
        assert proc.returncode == 0, proc.stderr[-4000:]
        line = [s for s in proc.stdout.splitlines() if s.startswith("RESULT ")][-1]
        got[mode] = json.loads(line[len("RESULT "):])
    assert got["fast"]["j"] == got["strict"]["j"]
    assert got["fast"]["digest"] == got["strict"]["digest"] == PRODUCTION_VARBC_DIGEST, got["strict"]["digest"]


# ------------------------------------------------------------------ the door


def test_analyse_door_with_radiances_on_the_tutorial():
    """``analyse(radiances=...)`` on WRFDA's tutorial case with a linear stand-in for CRTM.

    The synthetic radiances carry a 1 K constant bias and a moist signal in
    one quadrant: VarBC takes the bias, the analysis fits the signal (WV
    O-A below O-B) and the conventional fit stays within 5 % of the
    conventional-only run's.
    """
    import os

    from woof.da.var.analysis import analyse, read_first_guess
    from woof.da.var.obs_wrfda import LambertWRFDA

    d = os.environ.get("WOOF_WRFDA471_DIR")
    if not d or not all((Path(d) / n).is_file() for n in ("be.dat", "ob.ascii", "fg")):
        pytest.skip("WOOF_WRFDA471_DIR with be.dat, ob.ascii and fg is not set")
    d = Path(d)
    fg = read_first_guess(d / "fg")
    lat, lon = fg["XLAT"], fg["XLONG"]
    nx, ny = lat.shape
    rng = np.random.default_rng(12)
    ii, jj = np.meshgrid(np.arange(3, nx - 3, 2), np.arange(3, ny - 3, 2), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    n = ii.size
    rec = rad.RadianceObs(sensor="abi_g16", channels=[8, 9, 10], bt=np.zeros((n, 3)), lat=lat[ii, jj],
                          lon=lon[ii, jj], sensor_zenith=40.0, sensor_azimuth=0.0, sensor_scan=0.0,
                          solar_zenith=100.0, solar_azimuth=0.0, time=WHEN)
    from woof.da.var.analysis import wrftoxb

    xb, _ = wrftoxb(fg)
    clean, _ = rad.radiance_term([rec], xb, rad.read_surface(d / "fg"), LambertWRFDA.from_attributes(fg.attrs),
                                 _LinearCRTM(), settings=rad.RadianceQC(thinning=False))
    b0 = clean.batches[0]
    loc = rad.locate(LambertWRFDA.from_attributes(fg.attrs), rec.lat, rec.lon)
    assert loc["inside"].all() and b0.n == n
    # the stand-in has one K for every channel, so the signal is the same in each
    moist = ((ii > nx // 2) & (jj > ny // 2))[:, None] * np.full(3, -0.3)
    rec.bt = b0.hx + 1.0 + moist + 0.05 * rng.standard_normal((n, 3))
    common = dict(tier="reference", ntmax=30, eps=0.01)
    base = analyse(d / "be.dat", d / "ob.ascii", d / "fg", **common)
    vb = vbc.VarBC({"abi_g16": vbc.template("abi_g16", range(7, 17))})
    qc = rad.RadianceQC(thinning=False, error_k={8: 0.3, 9: 0.3, 10: 0.3})
    res = analyse(d / "be.dat", d / "ob.ascii", d / "fg", radiances=[rec], crtm=_LinearCRTM(), radiance_qc=qc,
                  varbc=vb, **common)
    print(json.dumps({k: v for k, v in res.fit.items()}, indent=1))
    for ch in (8, 9, 10):
        f = res.fit[f"abi_g16:{ch}"]
        assert f["n"] == n and f["oma_rms"] < f["omb_rms"]
        assert vb.instruments["abi_g16"].channel(ch).param[0] == pytest.approx(1.0, abs=0.15)
    # the synthetic radiances agree with x_b, not with the real reports, so they pull
    # the conventional fit back a little; the real-case gate (GR-V3) asks for no loss
    conv, conv0 = res.fit["conventional"], base.fit["conventional"]
    assert conv["n"] == conv0["n"] and conv0["oma_rms"] < conv["oma_rms"] <= 1.05 * conv0["oma_rms"] < conv["omb_rms"]
    assert res.radiance["varbc_controls"] == 15
