"""woof.da.var.pairing: hybrid 3DEnVar paired with an LETKF cycle (plan V3.2, second half).

What is asserted, and why it is enough:

- **Layout:** the LETKF's ``(R, nz, ny, nx)`` stacks and the hybrid's
  ``(R, nx, ny, nz)`` ones round-trip exactly (``ps`` as 3-D or 4-D); the
  ensemble handed to the hybrid is :meth:`Ensemble.from_members` of the same
  members, word for word.
- **Recentring:** the analysis members' mean is the variational analysis
  (to rounding), the perturbations about it are the LETKF's unchanged,
  ``gamma = 0`` is the identity, ``gamma = 1/2`` moves the mean half way, and
  a wrong shape or ``gamma`` is refused.
- **One step end to end:** the real :func:`woof.da.letkf.analyze` on a tiny
  grid with the hybrid cost's own minimiser standing in for ``analyse``
  (``analyse`` itself is covered by test_var_hybrid_cli): the ensemble the
  hybrid receives is the LETKF prior's, the members come out centred on the
  variational analysis, and the LETKF's perturbations are kept.
- **Refusals:** an ``ensemble=`` keyword (the pairing builds it) and an LETKF
  that does not analyse u, v, t and q (the LETKF takes 3-D fields only, so ``ps``
  keeps the prior's perturbations).  ``ep_from_wrf`` is unchanged by the
  ``member_fields`` split (test_var_hybrid_cli).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from woof.da import letkf as lk
from woof.da.var import hybrid as hy
from woof.da.var import pairing as pr
from woof.da.var.cost import SparseObsTerm
from woof.da.var.minimise import minimise_cg

from test_var_cost import _shapes
from test_var_hybrid import NE, NX, NY, NZ, _ensemble, _setup, _static_scale, _znw

LK = {"u": "U", "v": "V", "t": "T", "q": "QV", "ps": "PS"}


def _prior(seed=3, ps4=False):
    """Members as the LETKF holds them: mean plus smooth perturbations, in (R, nz, ny, nx)."""
    scale = {"u": 1.5, "v": 1.5, "t": 0.8, "q": 4e-4, "ps": 60.0}
    pert = _ensemble(scale, seed=seed)
    mean = {"u": 5.0, "v": -3.0, "t": 270.0, "q": 4e-3, "ps": 9.0e4}
    var = {k: pert[k] + mean[k] for k in hy.FIELDS}
    like = {LK[k]: (np.ascontiguousarray(np.transpose(v, (0, 3, 2, 1))) if k != "ps"
                    else np.ascontiguousarray(np.transpose(v, (0, 2, 1)))) for k, v in var.items()}
    if ps4:
        like["PS"] = like["PS"][:, None]
    return var, like


def test_layout_round_trips_and_ensemble_is_from_members():
    for ps4 in (False, True):
        var, like = _prior(ps4=ps4)
        got = pr.to_var_layout(like, LK)
        for k in hy.FIELDS:
            np.testing.assert_array_equal(got[k], var[k])
        back = pr.from_var_layout(got, LK, like=like)
        for name in like:
            np.testing.assert_array_equal(back[name], like[name])
            assert back[name].shape == like[name].shape
    ens = pr.ensemble_from_letkf(like, LK)
    ref = hy.Ensemble.from_members(var)
    for a, b in zip(ens.arrays(), ref.arrays()):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="pairing field names"):
        pr.to_var_layout(like, {"w": "W"})
    with pytest.raises(ValueError, match="must be"):
        pr.to_var_layout(dict(like, T=like["T"][0]), LK)


def _target(var, shift):
    return {k: pr._mean(var[k]) + shift[k] for k in hy.FIELDS}


def test_recentre_moves_the_mean_and_keeps_the_perturbations():
    var, like = _prior()
    shift = {"u": 0.7, "v": -0.4, "t": 1.3, "q": 2e-4, "ps": -120.0}
    rng = np.random.default_rng(4)
    target = {k: pr._mean(var[k]) + shift[k] + 0.1 * shift[k] * rng.standard_normal(var[k].shape[1:])
              for k in hy.FIELDS}
    out = pr.to_var_layout(pr.recentre(like, target, names=LK), LK)
    for k in hy.FIELDS:
        np.testing.assert_allclose(pr._mean(out[k]), target[k], rtol=0, atol=1e-11 * np.abs(target[k]).max())
        np.testing.assert_allclose(out[k] - pr._mean(out[k])[None], var[k] - pr._mean(var[k])[None], rtol=0,
                                   atol=1e-9 * max(1.0, np.abs(var[k]).max()) * 1e-3)
    same = pr.recentre(like, target, names=LK, gamma=0.0)
    for name in like:
        np.testing.assert_array_equal(same[name], like[name])
    half = pr.to_var_layout(pr.recentre(like, target, names=LK, gamma=0.5), LK)
    for k in hy.FIELDS:
        np.testing.assert_allclose(pr._mean(half[k]), 0.5 * (pr._mean(var[k]) + target[k]), rtol=0,
                                   atol=1e-11 * np.abs(target[k]).max())
    with pytest.raises(ValueError, match="gamma"):
        pr.recentre(like, target, names=LK, gamma=1.5)
    with pytest.raises(ValueError, match="target t"):
        pr.recentre(like, dict(target, t=target["t"][:-1]), names=LK)


def _letkf_obs(like, k0=2):
    """One observation per column of a 3 x 3 block at level ``k0`` on T: an identity operator."""
    shape = like["T"].shape[1:]
    mask = np.zeros(shape, dtype=bool)
    mask[k0, 3:8, 3:8] = True
    truth = like["T"].mean(axis=0) + 0.8
    values = np.where(mask, truth, np.nan)
    return lk.GriddedObs(name="t", values=values, errors=0.5, simulated=like["T"].copy(), mask=mask)


def test_hybrid_step_end_to_end_with_the_letkf(monkeypatch):
    var, like = _prior()
    b, xb, _, cfg, _ = _setup(2.0, 2, scale=3.0)
    # the variational first guess: the ensemble mean in the hybrid's (nx, ny, nz) layout
    xbm = dict(xb, u=pr._mean(var["u"]), v=pr._mean(var["v"]), t=pr._mean(var["t"]), q=pr._mean(var["q"]),
               psfc=pr._mean(var["ps"]))
    seen = {}

    def fake_analyse(be, obs, first_guess, *, ensemble, hybrid, **kw):
        seen["ensemble"] = ensemble
        shapes = _shapes(NX, NY, NZ)
        idx = np.ravel_multi_index((5, 5, 2), (NX, NY, NZ))
        term = SparseObsTerm(shapes, np.array([["t"]], dtype=object), np.array([[idx]]), np.ones((1, 1)),
                             np.array([0.8]), np.array([0.5]), groups=[("sound", [("t", np.arange(1))])])
        cost = hy.hybrid_cost(b, dict(xb, ds=3000.0, znw=_znw(NZ), ptop=5000.0), term, ensemble, hybrid)
        r = minimise_cg(cost, ntmax=20, eps=1e-12)
        return SimpleNamespace(xb=xbm, xa=cost.u.apply(r.xhat)["xa"])

    monkeypatch.setattr("woof.da.var.analysis.analyse", fake_analyse)
    grid = lk.GridGeometry(dx_m=3000.0, dy_m=3000.0, heights_m=np.linspace(300.0, 9000.0, NZ))
    lcfg = lk.LetkfConfig(localization=lk.Localization(horizontal_m=9000.0, vertical_m=4000.0),
                          analysis_fields=tuple(v for k, v in LK.items() if k != "ps"), rtps_alpha=0.0)
    step = pr.hybrid_step("be", "obs", "fg", like, ([_letkf_obs(like)], grid, lcfg), hybrid=cfg, names=LK)
    ref = hy.Ensemble.from_members(var)
    for a, c in zip(seen["ensemble"].arrays(), ref.arrays()):
        np.testing.assert_array_equal(a, c)
    out = pr.to_var_layout(step.members, LK)
    target = pr.variational_state(step.var)
    inc = {k: np.asarray(like[LK[k]] + step.increments.get(LK[k], 0.0)) for k in hy.FIELDS}
    lan = pr.to_var_layout({LK[k]: inc[k] for k in hy.FIELDS}, LK)
    for k in hy.FIELDS:
        np.testing.assert_allclose(pr._mean(out[k]), target[k], rtol=0, atol=1e-10 * np.abs(target[k]).max())
        np.testing.assert_allclose(out[k] - pr._mean(out[k])[None], lan[k] - pr._mean(lan[k])[None], rtol=0,
                                   atol=1e-8 * max(1.0, np.abs(lan[k]).max()) * 1e-3)
    # the observation (+0.8 K at one point, +0.8 over the LETKF's block) moved the variational mean there
    assert step.stats["shift"]["t"] > 0.0
    assert step.stats["spread_post"]["t"] > 0.0 and step.stats["negative_q"] == 0
    with pytest.raises(ValueError, match="do not pass ensemble"):
        pr.hybrid_step("be", "obs", "fg", like, None, hybrid=cfg, names=LK, ensemble=ref)


def test_the_letkf_must_analyse_every_pairing_field(monkeypatch):
    var, like = _prior()
    _, _, _, cfg, _ = _setup(2.0, 2, scale=3.0)
    monkeypatch.setattr("woof.da.var.analysis.analyse", lambda *a, **k: SimpleNamespace(xb={}, xa={}))
    lcfg = lk.LetkfConfig(localization=lk.Localization(horizontal_m=9000.0, vertical_m=4000.0),
                          analysis_fields=("T", "U", "V"), rtps_alpha=0.0)
    with pytest.raises(ValueError, match="analysis_fields lack"):
        pr.hybrid_step("be", "obs", "fg", like, ([], None, lcfg), hybrid=cfg, names=LK)

