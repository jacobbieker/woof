"""woof.da.var.cost: J, grad J and the vtoy chain (plan V1.4; gates O3 and O4).

What is asserted, and why it is enough:

- **Adjoint of the whole chain (O3):** ``<H' U v, y> = <v, U^T H'^T y>`` to
  1e-13 in binary64, with ``H'`` a random sparse stencil touching every
  analysis field (u, v, t, q, psfc, p, rho).
- **Gradient (O4):** ``da_check_gradient``'s Taylor ratio over 8 decades of
  ``a`` (1e-10 .. 1e-3) is within 1e-6 of 1 at the best ``a``.
- **Jo is WRFDA's:** per type and variable, ``0.5 sum(re^2 / err^2)`` over the
  accepted observations, summed in ``jo%total``'s type order; a rejected
  observation (qc < 0) contributes neither to Jo nor to the gradient.
- **Hessian product:** ``calculate_gradj`` without a residual is
  ``grad J(p) - grad J(0)`` for this quadratic.
- **Device tiers:** the oracle tier's J and gradient are the reference's
  words; the production tier is within binary32 of them and repeatable.
- **Tutorial (B only):** on WRFDA's tutorial ``be.dat`` and background, with
  no observations, ``J = 0.5 v^T v`` and its gradient is ``v`` word for word
  (``U`` is not exercised by Jb; the full O6 lane comes with the operators).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

from woof.da.var import be_dat as bd
from woof.da.var import reference as ref
from woof.da.var.cost import JO_ORDER, SparseObsTerm, VarCost, check_gradient

from test_var_be_dat import _synthetic
from test_var_transforms_reference import _xb

NX, NY, NZ = 13, 11, 6
FIELDS = ("u", "v", "t", "q", "psfc", "p", "rho")


def _shapes(nx, ny, nz):
    return {k: ((nx, ny) if k == "psfc" else (nx, ny, nz)) for k in FIELDS}


def synthetic_obs(b, xb, nobs=240, seed=11, kinds=("u", "v", "t", "q", "psfc"), reject=0.1, stencil=4):
    """A random linear stencil (``stencil`` grid points per observation) with
    errors scaled to the increments ``U`` makes, innovations from a truth
    ``v`` plus noise, ``reject`` of them qc < 0, and two Jo types."""
    rng = np.random.default_rng(seed)
    nx, ny = xb["grid_box_area"].shape
    nz = b.kz
    shapes = _shapes(nx, ny, nz)
    kind = rng.choice(np.array(kinds), nobs)
    field_ = np.repeat(kind[:, None], stencil, axis=1).astype(object)
    index = np.stack([rng.integers(0, int(np.prod(shapes[str(k)])), stencil) for k in kind])
    w = rng.uniform(0.0, 1.0, (nobs, stencil))
    w = w / w.sum(axis=1, keepdims=True)
    cv_size = nx * ny * sum(b.cv_mz)
    probe = SparseObsTerm(shapes, field_, index, w, np.zeros(nobs), np.ones(nobs))
    u = VarCost.reference(b, xb, probe).u
    y = probe.tl(u.apply(rng.standard_normal(cv_size))["xa"])
    scale = {k: float(np.std(y[kind == k])) or 1.0 for k in kinds}
    err = np.array([scale[str(k)] for k in kind]) * rng.uniform(0.5, 1.5, nobs)
    truth = probe.tl(u.apply(rng.standard_normal(cv_size))["xa"])
    d = truth + err * rng.standard_normal(nobs)
    qc = np.where(rng.uniform(size=nobs) < reject, -5, 0)
    # two report types, each with its variables in WRFDA's order; synop's
    # pressure is the psfc stencil
    owner = np.where((np.arange(nobs) % 3 == 0) | (kind == "psfc"), "synop", "sound")
    groups = []
    for t, names in (("synop", ("u", "v", "t", "psfc", "q")), ("sound", ("u", "v", "t", "q", "p", "rho"))):
        variables = [(n, np.flatnonzero((owner == t) & (kind == n))) for n in names]
        groups.append((t, variables))
    return SparseObsTerm(shapes, field_, index, w, d, err, qc=qc, groups=groups)


def case(cv_options=5, **kw):
    be = _synthetic(ni=NX, nj=NY, nk=NZ, nb2d=2, cv_options=cv_options)
    b = bd.setup_be_regional(be, ix=NX, jy=NY, kz=NZ, ds=3000.0, config=bd.BeConfig(cv_options=cv_options))
    xb = _xb(np.random.default_rng(5), NX, NY, NZ)
    return b, xb, synthetic_obs(b, xb, **kw)


@pytest.fixture(scope="module")
def setup():
    return case()


def _serial(a, b):
    acc = 0.0
    for x, y in zip(np.ravel(a), np.ravel(b)):
        acc = acc + float(x) * float(y)
    return acc


# ------------------------------------------------------------ reference tier


def test_vtoy_adjoint_to_1e13(setup):
    b, xb, obs = setup
    kinds = ("u", "v", "t", "q", "psfc", "p", "rho")
    obs = synthetic_obs(b, xb, kinds=kinds, reject=0.0)
    cost = VarCost.reference(b, xb, obs)
    rng = np.random.default_rng(1)
    v = rng.standard_normal(cost.cv_size)
    y = rng.standard_normal(obs.d.size)
    (hv,) = cost.vtoy(v)
    lhs = _serial(hv, y)
    rhs = _serial(v, cost.vtoy_adj([y]))
    rel = abs(lhs - rhs) / abs(lhs)
    print(f"vtoy adjoint: <HUv,y>={lhs!r} <v,U^T H^T y>={rhs!r} rel={rel:.3e}")
    assert rel <= 1e-13


def test_jo_is_wrfdas_sum_over_accepted_observations(setup):
    b, xb, obs = setup
    cost = VarCost.reference(b, xb, obs)
    v = np.random.default_rng(2).standard_normal(cost.cv_size)
    j = cost.calculate_j(v, None, 1)
    (hv,) = cost.vtoy(v)
    r = np.where(obs.mask, obs.d - hv, 0.0)
    want = 0.5 * float(np.sum((r / obs.err) ** 2))
    assert j.jo == pytest.approx(want, rel=1e-12)
    assert j.jb == pytest.approx(0.5 * float(np.dot(v, v)), rel=1e-13)
    assert j.total == j.jb + j.jo
    assert list(j.jo_types) == [t for t in JO_ORDER if t in j.jo_types] == ["sound", "synop"]
    syn = j.jo_types["synop"]
    assert syn["_"] == (((syn["u"] + syn["v"]) + syn["t"]) + syn["psfc"]) + syn["q"]
    assert j.jo == j.jo_types["sound"]["_"] + j.jo_types["synop"]["_"]


def test_rejected_observations_do_not_count(setup):
    b, xb, obs = setup
    cost = VarCost.reference(b, xb, obs)
    v = np.random.default_rng(3).standard_normal(cost.cv_size)
    j, g = cost.value_and_gradient(v)
    bumped = SparseObsTerm(obs.shapes, obs.field, obs.index, obs.weight,
                           np.where(obs.mask, obs.d, obs.d + 1e3), obs.err,
                           qc=np.where(obs.mask, 0, -5), groups=obs.groups)
    j2, g2 = VarCost.reference(b, xb, bumped).value_and_gradient(v)
    assert j2.total == j.total
    np.testing.assert_array_equal(g2, g)


def test_hessian_product_is_the_gradient_difference(setup):
    b, xb, obs = setup
    cost = VarCost.reference(b, xb, obs)
    p = np.random.default_rng(4).standard_normal(cost.cv_size)
    hp = cost.calculate_gradj(p)
    _, g0 = cost.value_and_gradient(np.zeros(cost.cv_size))
    _, gp = cost.value_and_gradient(p)
    np.testing.assert_allclose(hp, gp - g0, rtol=0, atol=1e-10 * float(np.max(np.abs(hp))))


@pytest.mark.parametrize("cv_options", [5, 7])
def test_check_gradient_taylor_ratio(cv_options):
    b, xb, obs = case(cv_options)
    cost = VarCost.reference(b, xb, obs)
    rng = np.random.default_rng(7)
    out = check_gradient(cost, rng.uniform(size=cost.cv_size) - 0.5)
    ratios = ", ".join(f"{r['alpha']:.0e}:{r['t1']:.12f}" for r in out["rows"])
    print(f"cv{cv_options} grtest T1 {ratios}; best {out['best']!r} ({out['digits']} digits)")
    assert len(out["rows"]) == 8 and out["df0"] < 0
    assert abs(out["best"] - 1.0) <= 1e-6, out


# ------------------------------------------------------------ device tiers


def _device_run(precision, cv_options=5):
    import cupy as cp

    b, xb, obs = case(cv_options)
    cost = VarCost.device(b, xb, obs, precision)
    v = np.random.default_rng(2).standard_normal(cost.cv_size)
    j, g = cost.value_and_gradient(cost.asarray(v))
    hp = cost.calculate_gradj(cost.asarray(v))
    return j, cp.asnumpy(g), cp.asnumpy(hp), (b, xb, obs, v)


@requires_gpu
@pytest.mark.parametrize("cv_options", [5, 7])
def test_oracle_tier_is_the_reference_word_for_word(cv_options):
    j, g, hp, (b, xb, obs, v) = _device_run("oracle", cv_options)
    cost = VarCost.reference(b, xb, obs)
    jr, gr = cost.value_and_gradient(v)
    assert (j.total, j.jb, j.jo) == (jr.total, jr.jb, jr.jo)
    np.testing.assert_array_equal(g, gr)
    np.testing.assert_array_equal(hp, cost.calculate_gradj(v))


@requires_gpu
def test_production_tier_is_close_and_repeatable():
    j, g, hp, (b, xb, obs, v) = _device_run("production")
    jr, gr = VarCost.reference(b, xb, obs).value_and_gradient(v)
    assert j.total == pytest.approx(jr.total, rel=1e-5)
    assert float(np.max(np.abs(g - gr))) <= 1e-4 * float(np.max(np.abs(gr)))
    j2, g2, hp2, _ = _device_run("production")
    assert j2.total == j.total
    np.testing.assert_array_equal(g2, g)
    np.testing.assert_array_equal(hp2, hp)


@requires_gpu
def test_device_vtoy_adjoint():
    b, xb, obs = case()
    for precision, tol in (("oracle", 1e-13), ("production", 1e-5)):
        cost = VarCost.device(b, xb, obs, precision)
        rng = np.random.default_rng(1)
        v = cost.asarray(rng.standard_normal(cost.cv_size))
        y = rng.standard_normal(obs.d.size).astype(cost.dtype)
        (hv,) = cost.vtoy(v)
        lhs = _serial(hv, y)
        rhs = cost.ops.dot(v, cost.vtoy_adj([y]))
        assert abs(lhs - rhs) <= tol * abs(lhs), (precision, lhs, rhs)


# ------------------------------------------------------------ tutorial, B only


def _tutorial():
    d = os.environ.get("WOOF_WRFDA471_DIR")
    if not d or not all((Path(d) / f).is_file() for f in ("be.dat", "woof_xb.bin")):
        pytest.skip("WOOF_WRFDA471_DIR with be.dat and woof_xb.bin is not set")
    from woof.da.var.oracle import read_dump

    dx = read_dump(Path(d) / "woof_xb.bin")
    nx, ny = dx["lat"].shape
    nz = dx["dnw"].size
    b = bd.setup_be_regional(bd.read_be_dat(Path(d) / "be.dat"), ix=nx, jy=ny, kz=nz, ds=float(dx["ds"]))
    xb = {k: dx[k] for k in ("grid_box_area", "coefx", "coefy", "qs", "q", "psac", "dnw", "rho", "p", "t",
                             "c1h", "c2h")}
    return b, xb


def test_tutorial_b_only():
    b, xb = _tutorial()
    nx, ny = xb["grid_box_area"].shape
    shapes = _shapes(nx, ny, b.kz)
    empty = SparseObsTerm(shapes, np.zeros((0, 1), dtype=object), np.zeros((0, 1), dtype=np.int64),
                          np.zeros((0, 1)), np.zeros(0), np.zeros(0), groups=[])
    cost = VarCost.reference(b, xb, empty)
    v = np.random.default_rng(0).uniform(size=cost.cv_size) - 0.5
    j = cost.calculate_j(v, None, 0)
    assert j.jo == 0.0 and j.total == j.jb == 0.5 * _serial(v, v)
    np.testing.assert_array_equal(cost.calculate_gradj(v, re=cost.re), v)
    hv = cost.calculate_gradj(v)  # (I + 0) v, through U and U^T with nothing observed
    np.testing.assert_array_equal(hv, v)
