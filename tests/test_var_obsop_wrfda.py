"""woof.da.var.obsop_wrfda: WRFDA's H' and H'^T (plan V1.2/V1.3, gate O3).

O3: ``<H'dx, dy> = <dx, H'^T dy>`` to 1e-13 relative, per type and for all
types together, with fixed seeds; the synthetic grid runs everywhere, the
tutorial geometry needs ``WOOF_WRFDA471_DIR`` (see test_var_innov_wrfda.py).
The adjoint's wave-grouped scatter is checked word for word against a plain
serial transcription of ``da_interp_lin_3d_adj``.
"""

from __future__ import annotations

import numpy as np
import pytest

import _var_wrfda_synthetic as syn
from test_var_innov_wrfda import _tutorial
from woof.da.var import innov_wrfda as iw
from woof.da.var import obs_wrfda as ow
from woof.da.var import obsop_wrfda as oo

M = ow.MISSING_R
O3_RTOL = 1e-13


def _rand_x(shape, seed):
    rng = np.random.default_rng(seed)
    x = {v: rng.standard_normal(shape) for v in ("u", "v", "t", "q", "p", "rho")}
    x["psfc"] = rng.standard_normal(shape[:2])
    return x


def _rand_y(st, seed):
    rng = np.random.default_rng(seed)
    return {n: {v: rng.standard_normal(st.sizes[n]) for v in st.fields(n)} for n in st.types}


def _dot_y(a, b, only=None):
    return sum(float(np.dot(a[n][v], b[n][v])) for n in a if only in (None, n) for v in a[n])


def _dot_x(a, b):
    return sum(float(np.vdot(a[k], b[k])) for k in a)


def _o3(st, shape, seed):
    x = _rand_x(shape, seed)
    dy = _rand_y(st, seed + 1)
    y = st.tl(x)
    out = {}
    for n in st.types + [None]:
        sub = dy if n is None else {m: ({v: (a if m == n else np.zeros_like(a)) for v, a in f.items()})
                                    for m, f in dy.items()}
        lhs, rhs = _dot_y(y, sub), _dot_x(x, st.ad(sub))
        if lhs == 0.0 and rhs == 0.0:
            out[n] = 0.0
            continue
        out[n] = abs(lhs - rhs) / abs(lhs)
    return out


_SFC = {"u": [(2.0, 0, 1.0)], "v": [(1.0, 0, 1.0)], "t": [(285.0, 0, 2.0)],
        "p": [(99000.0, 0, 100.0)], "q": [(0.005, 0, 0.001)]}


def _synthetic_obs():
    xb = syn.background()
    reports = []
    for n, (i, j, dx, dy) in enumerate([(2, 2, 0.3, 0.6), (3, 2, 0.9, 0.1), (2, 2, 0.5, 0.5), (6, 5, 0.2, 0.7)]):
        col = iw._hcol(xb["p"], i - 1, j - 1, dx, dy, 1.0 - dx, 1.0 - dy)
        ps = [0.5 * (col[k] + col[k + 1]) for k in (0, 3, 3, 17, 38)] + [10.0]
        reports.append(dict(i=i, j=j, dx=dx, dy=dy, h=[M] * len(ps), p=ps,
                            fields={v: [(0.0, 0, 1.0)] * len(ps) for v in ("u", "v", "t", "q")}))
    hcol = iw._hcol(xb["h"], 1, 1, 0.5, 0.5, 0.5, 0.5)
    types = {
        "sound": syn.obs_type("sound", reports),
        "pilot": syn.obs_type("pilot", reports),
        "geoamv": syn.obs_type("geoamv", reports),
        "synop": syn.obs_type("synop", [dict(i=2, j=2, dx=0.5, dy=0.5, h=[hcol[0] + 30.0],
                                             fields=_SFC),
                                        dict(i=3, j=3, dx=0.1, dy=0.2, h=[hcol[0] - 20.0],
                                             fields=_SFC)]),
        "qscat": syn.obs_type("qscat", [dict(i=4, j=4, dx=0.4, dy=0.4, h=[hcol[0] + 10.0],
                                             fields={v: [(0.0, 0, 1.0)] for v in ("u", "v")})]),
    }
    return xb, iw.innovations(syn.obs(types), xb, options=iw.Options(check_max_iv=False))


def test_o3_synthetic_per_type_and_combined():
    xb, iv = _synthetic_obs()
    st = oo.ObsStencil(iv, xb)
    assert set(st.types) >= {"sound", "pilot", "geoamv", "synop", "qscat"}
    for n, r in _o3(st, xb["u"].shape, 11).items():
        assert r <= O3_RTOL, (n, r)


def _serial_ad3(g, geo, fo, kte):
    """da_interp_lin_3d_adj, one report and one level at a time."""
    i0, j0, dx, dy, dxm, dym = geo.h
    for n in range(geo.nrep):
        fmz = np.zeros(kte)
        for row in np.flatnonzero(geo.rep == n):
            k = geo.k[row]
            if k > 0:
                fmz[k - 1] = geo.dzm[row] * fo[row] + fmz[k - 1]
                fmz[k] = geo.dz[row] * fo[row] + fmz[k]
        a, b = i0[n], j0[n]
        for kk in range(kte):
            g[a, b, kk] = dym[n] * dxm[n] * fmz[kk] + g[a, b, kk]
            g[a + 1, b, kk] = dym[n] * dx[n] * fmz[kk] + g[a + 1, b, kk]
            g[a, b + 1, kk] = dy[n] * dxm[n] * fmz[kk] + g[a, b + 1, kk]
            g[a + 1, b + 1, kk] = dy[n] * dx[n] * fmz[kk] + g[a + 1, b + 1, kk]


def test_wave_scatter_is_the_serial_loop_word_for_word():
    xb, iv = _synthetic_obs()
    st = oo.ObsStencil(iv, xb)
    rng = np.random.default_rng(5)
    for n in ("sound", "synop"):
        it = st.interp[n]
        fo = rng.standard_normal(st.sizes[n])
        g0 = rng.standard_normal(xb["u"].shape)
        a, b = g0.copy(), g0.copy()
        it.ad3(a, fo)
        _serial_ad3(b, it.geo, fo, it.kte)
        assert np.array_equal(a, b), n


def test_adjoint_is_deterministic():
    xb, iv = _synthetic_obs()
    st = oo.ObsStencil(iv, xb)
    dy = _rand_y(st, 3)
    g1, g2 = st.ad(dy), st.ad(dy)
    assert all(np.array_equal(g1[k], g2[k]) for k in g1)


def test_obsterm_composes_xtoxa():
    xb, iv = _synthetic_obs()
    nz = xb["u"].shape[2]
    rng = np.random.default_rng(2)
    xbx = dict(q=xb["q"], psac=xb["psfc"] - 5000.0, dnw=np.full(nz, -1.0 / nz), rho=xb["rho"],
               p=xb["p"], t=xb["t"])
    c1h, c2h = np.linspace(1.0, 0.0, nz), np.linspace(0.0, 4000.0, nz)
    term = oo.ObsTerm(iv, xb, xtoxa=(xbx, c1h, c2h))
    x = {v: rng.standard_normal(xb["u"].shape) for v in ("u", "v", "t", "q")}
    x["psfc"] = rng.standard_normal(xb["psfc"].shape)
    dy = _rand_y(term.stencil, 8)
    lhs, rhs = _dot_y(term.tl(x), dy), _dot_x(x, term.ad(dy))
    assert abs(lhs - rhs) <= 1e-12 * abs(lhs)
    assert set(term.d) == set(term.err) == set(term.mask) == set(term.stencil.types)
    assert set(term.jo_split()) <= set(iw.JO_FIELDS)


def test_o3_tutorial_per_type_and_combined():
    obs, xb, dump, iv = _tutorial()
    st = oo.ObsStencil(iv, xb)
    res = _o3(st, xb["u"].shape, 2026)
    for n, r in res.items():
        assert r <= O3_RTOL, (n, r)


# --------------------------------------------------------------------------- nonlinear types
def _nonlinear_obs():
    """gpspw (both station-height branches), gpsref and satem on the synthetic grid."""
    xb = syn.background()
    terr = iw._hcol(xb["terr"], 1, 1, 0.5, 0.5, 0.5, 0.5)
    hcol = iw._hcol(xb["h"], 2, 3, 0.4, 0.3, 0.6, 0.7)
    pcol = iw._hcol(xb["p"], 3, 2, 0.2, 0.8, 0.8, 0.2)
    gpsref = [dict(i=3, j=4, dx=0.4, dy=0.3, h=[hcol[k] + 15.0 for k in (2, 9, 20, 33)],
                   fields={"ref": [(0.0, 0, 5.0)] * 4})]
    types = {
        "gpspw": syn.obs_type("gpspw", [dict(i=2, j=2, dx=0.5, dy=0.5, elv=terr - 25.0,
                                             fields={"tpw": [(0.0, 0, 0.2)]}),
                                        dict(i=5, j=3, dx=0.3, dy=0.6, elv=terr + 900.0,
                                             fields={"tpw": [(0.0, 0, 0.2)]})]),
        "gpsref": syn.obs_type("gpsref", gpsref),
        "satem": syn.obs_type("satem", [dict(i=4, j=3, dx=0.2, dy=0.8,
                                             p=[0.5 * (pcol[k] + pcol[k + 1]) for k in (4, 12, 25)],
                                             fields={"thickness": [(0.0, 0, 10.0)] * 3})]),
    }
    types["satem"].extra["ref_p"] = np.array([pcol[0] + 300.0])
    return xb, iw.innovations(syn.obs(types), xb, options=iw.Options(check_max_iv=False))


def test_o3_synthetic_nonlinear_types():
    xb, iv = _nonlinear_obs()
    st = oo.ObsStencil(iv, xb)
    assert set(st.types) == {"gpspw", "gpsref", "satem"}
    for n, r in _o3(st, xb["u"].shape, 21).items():
        assert r <= O3_RTOL, (n, r)


def _taylor(h, tl, x0, dx):
    """``|H(x+e dx) - H(x) - e H'dx| / |e H'dx|`` falls like ``e``."""
    lin = tl(dx)
    r = []
    for e in (1e-3, 1e-4, 1e-5):
        x1 = {k: v + e * dx[k] for k, v in x0.items()}
        r.append(np.linalg.norm(h(x1) - h(x0) - e * lin) / np.linalg.norm(e * lin))
    assert r[2] < 2e-4 and r[1] < 2e-3 and r[2] < 0.2 * r[0], r


def _dx(xb, keys, seed, scale):
    rng = np.random.default_rng(seed)
    return {k: scale[k] * rng.standard_normal(xb[k].shape) for k in keys}


def _x_of(st, xb, dx):
    x = {k: np.zeros_like(xb[k]) for k in ("u", "v", "t", "q", "p", "rho")}
    x["psfc"] = np.zeros_like(xb["psfc"])
    x.update(dx)
    return x


def test_taylor_gpspw():
    """H = interp(0.1 sum dz q rho) + 0.1 dpw: the TL's sign of the station-height
    term (da_transform_xtoy_gpspw adds it, the innovation subtracts it)."""
    xb, iv = _nonlinear_obs()
    st = oo.ObsStencil(iv, xb)
    W = list(zip(*iw._w(iv.types["gpspw"].info)))
    elv = iv.types["gpspw"].info["elv"]

    dhf = np.diff(xb["hf"], axis=2)

    def h(x):
        b = dict(xb, q=x["q"], rho=x["rho"])
        pw = np.zeros(xb["psfc"].shape)
        for k in range(dhf.shape[2]):
            pw = pw + dhf[:, :, k] * (b["q"][:, :, k] * b["rho"][:, :, k])
        tpw = 0.1 * pw
        out = []
        for w, e in zip(W, elv):
            branch, _, terms = iw._gpspw_dpw(w, float(e), b)
            dpw = 0.0
            for _, rk, qk, dz in terms:
                dpw = rk * qk * dz if branch == 1 else dpw + rk * qk * dz
            out.append(iw._hcol(tpw, *w) + 0.1 * dpw)
        return np.array(out)

    dx = _dx(xb, ("q", "rho"), 4, {"q": 1e-3, "rho": 1e-2})
    _taylor(h, lambda d: st.tl(_x_of(st, xb, d))["gpspw"]["tpw"], {"q": xb["q"], "rho": xb["rho"]}, dx)


def test_taylor_gpsref():
    """H = interp_lin_3d(N(p_b, t, q)); xa%p does not enter WRFDA's ref TL
    (the innovation interpolates log N, the TL N itself)."""
    xb, iv = _nonlinear_obs()
    st = oo.ObsStencil(iv, xb)
    geo = iv.types["gpsref"].extra["geometry"]

    def h(x):
        t, q = x["t"], x["q"]
        n = 0.776 * xb["p"] / t * (1.0 + oo.COEFF * q / (t * (0.622 + 0.378 * q)))
        return iw.interp_lin_3d(n, geo)

    dx = _dx(xb, ("t", "q"), 6, {"t": 1.0, "q": 1e-3})
    _taylor(h, lambda d: st.tl(_x_of(st, xb, d))["gpsref"]["ref"], {"t": xb["t"], "q": xb["q"]}, dx)


def test_taylor_satem_first_level():
    """H = da_thickness at each report's first level, along t, q, p. Above it
    WRFDA's TL keeps TGL_tv2 at ref_p's value while tv2 moves up a level, so
    H' is the derivative only on the first level (the adjoint is still H'^T)."""
    xb, iv = _nonlinear_obs()
    st = oo.ObsStencil(iv, xb)
    t = iv.types["satem"]
    lm = iw._libm("correctly-rounded")
    W = list(zip(*iw._w(t.info)))
    first = t.offsets[:-1]

    def h(x):
        b = dict(xb, t=x["t"], q=x["q"], p=x["p"])
        out = []
        for n, w in enumerate(W):
            pre_ma, tv_ma = iw.tv_profile(b, w)
            l2, tv2, _ = iw.find_layer(t.extra["ref_p"][n], pre_ma, tv_ma, lm)
            p1 = t.coords["p"][first[n]]
            l1, tv1, _ = iw.find_layer(p1, pre_ma, tv_ma, lm)
            out.append(iw.thickness(pre_ma, tv_ma, tv1, tv2, l1, l2, p1, t.extra["ref_p"][n], lm))
        return np.array(out)

    dx = _dx(xb, ("t", "q", "p"), 8, {"t": 1.0, "q": 1e-3, "p": 50.0})
    _taylor(h, lambda d: st.tl(_x_of(st, xb, d))["satem"]["thickness"][first],
            {k: xb[k] for k in ("t", "q", "p")}, dx)
