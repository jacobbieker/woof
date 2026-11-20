"""WRFDA 4D-Var's xa <-> tangent-linear transfers (woof.da.var.transfer4d, PR 4D.1).

Host binary64 on a synthetic eta column (and on the tutorial first guess
when ``WOOF_WRFDA471_DIR`` is set):

- each adjoint is the transpose of its tangent-linear transfer (dot-product
  tests, both ``press`` and both ``ph`` options, ``hypsometric_opt`` 1 and 2);
- the defaults are WRFDA's statements: ``dn`` in the pressure column and no
  ``ph`` perturbation, and the reverse transfer inverts ``q``, ``psfc`` and
  ``t`` about the same first guess;
- ``press="dnw"`` puts ``xa%psfc`` at the surface full level, which the
  default does not;
- ``ph="hydrostatic"`` is the derivative of WRF's hypsometric equation
  (central differences of the nonlinear column);
- what is not ported is refused.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from woof.da.var import transfer4d as t4
from woof.da.var.analysis import KAPPA, T0

NX, NY, NZ = 5, 4, 7
PTOP, P0 = 5000.0, 100000.0


def _column(hyps=2, seed=0):
    rng = np.random.default_rng(seed)
    znw = np.concatenate([[1.0], np.sort(rng.uniform(0.05, 0.95, NZ - 1))[::-1], [0.0]])
    znu = 0.5 * (znw[:-1] + znw[1:])
    dnw = np.diff(znw)
    dn = np.concatenate([[0.0], np.diff(znu)])
    mub = 95000.0 - PTOP + rng.uniform(-3000, 3000, (NX, NY))
    mu_2 = rng.uniform(-300, 300, (NX, NY))
    mut = mub + mu_2
    t_2 = np.linspace(-10.0, 60.0, NZ) + rng.uniform(-2, 2, (NX, NY, NZ))
    qv = 0.015 * np.exp(-6.0 * (1.0 - znu)) * (1.0 + 0.2 * rng.uniform(-1, 1, (NX, NY, NZ)))
    p = znu * mut[..., None] + PTOP + rng.uniform(-200, 200, (NX, NY, NZ))
    t = (T0 + t_2) * (p / P0) ** KAPPA
    grid = {"qv": qv, "mu_2": mu_2, "mub": mub, "t_2": t_2, "dnw": dnw, "base_pres": P0, "p_top": PTOP,
            "hypsometric_opt": hyps, "c1h": np.ones(NZ), "c2h": np.zeros(NZ), "c3h": znu, "c4h": np.zeros(NZ),
            "c3f": znw, "c4f": np.zeros(NZ + 1)}
    xb = {"t": t, "p": p, "q": qv / (1.0 + qv), "psac": mut, "dn": dn, "dnw": dnw}
    return xb, grid


def _xa(rng, w=True):
    xa = {"u": rng.normal(0, 2, (NX, NY, NZ)), "v": rng.normal(0, 2, (NX, NY, NZ)),
          "t": rng.normal(0, 1, (NX, NY, NZ)), "q": rng.normal(0, 1e-3, (NX, NY, NZ)),
          "psfc": rng.normal(0, 100, (NX, NY)), "p": rng.normal(0, 50, (NX, NY, NZ))}
    if w:
        xa["w"] = rng.normal(0, 0.1, (NX, NY, NZ + 1))
    return xa


def _g(rng):
    return {"u_2": rng.normal(0, 2, (NX + 1, NY, NZ)), "v_2": rng.normal(0, 2, (NX, NY + 1, NZ)),
            "w_2": rng.normal(0, 0.1, (NX, NY, NZ + 1)), "t_2": rng.normal(0, 1, (NX, NY, NZ)),
            "ph_2": rng.normal(0, 10, (NX, NY, NZ + 1)), "mu_2": rng.normal(0, 100, (NX, NY)),
            "qv": rng.normal(0, 1e-3, (NX, NY, NZ)), "p": rng.normal(0, 50, (NX, NY, NZ))}


def _dot(a, b, keys):
    return sum(float(np.sum(a[k] * b[k])) for k in keys)


def _dot_xa_to_tl(xb, grid, rng, **kw):
    x = _xa(rng)
    y = _g(rng)
    lhs = _dot(t4.xa_to_tl(x, xb, grid, **kw), y, t4.TL_KEYS)
    ax = t4.xa_to_tl_adj(y, xb, grid, **kw)
    assert not np.any(ax["p"])  # xa%p is overwritten, not read
    rhs = _dot(x, ax, t4.XA4D_KEYS)
    return lhs, rhs


@pytest.mark.parametrize("hyps", [1, 2])
@pytest.mark.parametrize("press", ["wrfda", "dnw"])
@pytest.mark.parametrize("ph", ["zero", "hydrostatic"])
def test_xa_to_tl_adjoint_is_the_transpose(hyps, press, ph):
    xb, grid = _column(hyps)
    lhs, rhs = _dot_xa_to_tl(xb, grid, np.random.default_rng(1), press=press, ph=ph)
    assert lhs == pytest.approx(rhs, rel=1e-13)


def test_tl_to_xa_adjoint_is_the_transpose():
    xb, grid = _column()
    rng = np.random.default_rng(2)
    g, y = _g(rng), _xa(rng)
    lhs = _dot(t4.tl_to_xa(g, xb, grid), y, t4.XA4D_KEYS)
    ag = t4.tl_to_xa_adj(y, xb, grid)
    assert not np.any(ag["ph_2"])
    assert lhs == pytest.approx(_dot(g, ag, t4.TL_KEYS), rel=1e-13)


def test_default_is_wrfda_statements():
    """``g_press`` with ``dn``, ``g_ph_2 = 0``, the extrapolated C-grid edges, ``w`` copied."""
    xb, grid = _column()
    xa = _xa(np.random.default_rng(3))
    g = t4.xa_to_tl(xa, xb, grid)
    g_qv = xa["q"] / (1.0 - xb["q"]) ** 2
    s1md = np.sum((1.0 + grid["qv"]) * grid["dnw"], axis=2)
    g_mu = -(xa["psfc"] + xb["psac"] * np.sum(g_qv * grid["dnw"], axis=2)) / s1md
    np.testing.assert_allclose(g["mu_2"], g_mu, rtol=1e-13)
    np.testing.assert_allclose(g["qv"], g_qv, rtol=0, atol=0)
    mut = grid["mu_2"] + grid["mub"]
    press = np.zeros((NX, NY, NZ + 1))
    for k in range(NZ - 1, -1, -1):
        press[..., k] = press[..., k + 1] - (g_mu * (1.0 + grid["qv"][..., k]) + mut * g_qv[..., k]) * xb["dn"][k]
    np.testing.assert_allclose(g["p"], 0.5 * (press[..., :-1] + press[..., 1:]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(g["t_2"], (T0 + grid["t_2"]) * (xa["t"] / xb["t"] - KAPPA * g["p"] / xb["p"]),
                               rtol=1e-13)
    assert not np.any(g["ph_2"])
    u = xa["u"]
    np.testing.assert_array_equal(g["u_2"][1:-1], 0.5 * (u[:-1] + u[1:]))
    np.testing.assert_array_equal(g["u_2"][0], (3.0 * u[0] - u[1]) / 2.0)
    np.testing.assert_array_equal(g["u_2"][-1], (3.0 * u[-1] - u[-2]) / 2.0)
    np.testing.assert_array_equal(g["v_2"][:, 0], (3.0 * xa["v"][:, 0] - xa["v"][:, 1]) / 2.0)
    np.testing.assert_array_equal(g["w_2"], xa["w"])
    assert not np.any(t4.xa_to_tl({k: v for k, v in xa.items() if k != "w"}, xb, grid)["w_2"])


def test_reverse_transfer_inverts_q_psfc_and_t():
    xb, grid = _column()
    xa = _xa(np.random.default_rng(4))
    g = t4.xa_to_tl(xa, xb, grid)
    back = t4.tl_to_xa(g, xb, grid)
    for k in ("q", "t", "psfc"):
        np.testing.assert_allclose(back[k], xa[k], rtol=1e-12, atol=1e-12 * np.abs(xa[k]).max())
    np.testing.assert_array_equal(back["p"], g["p"])
    np.testing.assert_array_equal(back["w"], xa["w"])
    u = xa["u"]
    np.testing.assert_allclose(back["u"][1:-1], 0.25 * (u[:-2] + 2.0 * u[1:-1] + u[2:]), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("press", ["wrfda", "dnw"])
def test_dnw_puts_psfc_at_the_surface_full_level(press):
    xb, grid = _column()
    xa = _xa(np.random.default_rng(5))
    g_p = t4.xa_to_tl(xa, xb, grid, press=press)["p"]
    full = np.zeros((NX, NY, NZ + 1))
    for k in range(NZ - 1, -1, -1):
        full[..., k] = 2.0 * g_p[..., k] - full[..., k + 1]
    err = np.abs(full[..., 0] - xa["psfc"]).max() / np.abs(xa["psfc"]).max()
    if press == "dnw":
        assert err < 1e-10
    else:
        assert err > 1e-3  # WRFDA's dn column misses psfc


def _phi_total(t_2, qv, p, mu_2, grid):
    """WRF's hydrostatic total geopotential from the surface (hypsometric_opt 1 or 2), nonlinear."""
    alt = (287.0 / P0) * (T0 + t_2) * (1.0 + qv / t4.RD_OVER_RV) * (p / P0) ** t4.CVPM
    mut = grid["mub"] + mu_2
    phi = np.zeros(t_2.shape[:2] + (NZ + 1,))
    for k in range(NZ):
        if grid["hypsometric_opt"] == 1:
            d = -grid["dnw"][k] * mut * alt[..., k]
        else:
            pfd = mut * grid["c3f"][k] + PTOP
            pfu = mut * grid["c3f"][k + 1] + PTOP
            d = alt[..., k] * (mut * grid["c3h"][k] + PTOP) * np.log(pfd / pfu)
        phi[..., k + 1] = phi[..., k] + d
    return phi


@pytest.mark.parametrize("hyps", [1, 2])
def test_hydrostatic_ph_is_the_derivative_of_the_hypsometric_column(hyps):
    xb, grid = _column(hyps)
    g = _g(np.random.default_rng(6))
    tl = t4.hydrostatic_ph(g, xb, grid)
    eps = 1e-3

    def f(s):
        return _phi_total(grid["t_2"] + s * g["t_2"], grid["qv"] + s * g["qv"], xb["p"] + s * g["p"],
                          grid["mu_2"] + s * g["mu_2"], grid)

    fd = (f(eps) - f(-eps)) / (2.0 * eps)
    np.testing.assert_allclose(tl, fd, rtol=1e-6, atol=1e-6 * np.abs(fd).max())


def test_refusals():
    xb, grid = _column()
    xa = _xa(np.random.default_rng(7))
    with pytest.raises(NotImplementedError, match="cloud_cv_options"):
        t4.xa_to_tl(xa, xb, grid, cloud_cv_options=1)
    with pytest.raises(NotImplementedError, match="hybrid"):
        t4.tl_to_xa(_g(np.random.default_rng(8)), xb, {**grid, "c1h": np.linspace(1.0, 0.0, NZ)})
    with pytest.raises(ValueError, match="press"):
        t4.xa_to_tl(xa, xb, grid, press="dn")
    with pytest.raises(ValueError, match="ph"):
        t4.xa_to_tl_adj({}, xb, grid, ph="balanced")
    with pytest.raises(ValueError, match="DN"):
        t4.xa_to_tl(xa, {k: v for k, v in xb.items() if k != "dn"}, grid)


def _tutorial():
    d = os.environ.get("WOOF_WRFDA471_DIR")
    if not d or not (Path(d) / "fg").exists():
        pytest.skip("WOOF_WRFDA471_DIR with fg is not set")
    from woof.da.var.analysis import read_first_guess, wrftoxb

    return wrftoxb(read_first_guess(Path(d) / "fg"))


def test_tutorial_first_guess():
    """The dn quirk on the real eta levels, and the dot-product tests on the tutorial state."""
    xb, grid = _tutorial()
    assert xb["dn"][0] == 0.0
    assert abs(float(np.sum(grid["dnw"])) + 1.0) < 1e-12
    assert abs(float(np.sum(xb["dn"])) + 1.0) > 1e-3
    nx, ny, nz = xb["t"].shape
    rng = np.random.default_rng(9)
    x = {"u": rng.normal(size=(nx, ny, nz)), "v": rng.normal(size=(nx, ny, nz)), "w": rng.normal(size=(nx, ny, nz + 1)),
         "t": rng.normal(size=(nx, ny, nz)), "q": 1e-3 * rng.normal(size=(nx, ny, nz)),
         "psfc": 100.0 * rng.normal(size=(nx, ny)), "p": np.zeros((nx, ny, nz))}
    for ph in ("zero", "hydrostatic"):
        g = t4.xa_to_tl(x, xb, grid, ph=ph)
        y = {k: rng.normal(size=v.shape) for k, v in g.items()}
        assert _dot(g, y, t4.TL_KEYS) == pytest.approx(_dot(x, t4.xa_to_tl_adj(y, xb, grid, ph=ph), t4.XA4D_KEYS),
                                                       rel=1e-12)
    g = {k: rng.normal(size=v.shape) for k, v in g.items()}
    y = {k: rng.normal(size=v.shape) for k, v in x.items()}
    assert _dot(t4.tl_to_xa(g, xb, grid), y, t4.XA4D_KEYS) == pytest.approx(
        _dot(g, t4.tl_to_xa_adj(y, xb, grid), t4.TL_KEYS), rel=1e-12)
