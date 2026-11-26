"""diff_opt 2 coefficients TL/AD (P3a item 2b-3,
woof.grad.wrfplus.diffusion.k2_tl/k2_ad, n2_tl/n2_ad, k3_tl/k3_ad):
WRF smag2d_km (km_opt 4, with its 10 mlen cap, def_limit switch and the
terrain-slope limiter alpha), calculate_N2 (dry) and smag_km (km_opt 3,
anisotropic and isotropic, with cal_deform_and_div's defor13/23/33) and
their metrics (zx, zy, rdz, rdzw from ph'), on woof's strict
wrf_smag2d_km, wrf_calc_n2 and wrf_smag3d_km + wrf_smag_km_bc, with
periodic and bounded axes, the cold seam slope and steep terrain.

Breakage this prevents:
- an adjoint that is not the transpose of its TL (dot-product test);
- an adjoint that does not add into its inputs, or leaves the coefficient
  adjoints in place (the coefficients are overwritten, so their adjoints
  are consumed);
- a TL that does not overwrite the coefficients, or is not the derivative
  of the base state (TL vs finite differences, flat and steep);
- a base state, TL or AD that drifts from WRF 4.7.1's smag2d_km,
  calculate_N2 and smag_km (tests/data/wrfplus/kcoef2.npz, made by
  tools/wrfplus_oracle (diffusion2.F90 row 7) from WRFPLUS 4.7.1's
  libwrflib.a; module_diffusion_em.F is unchanged in these routines in
  4.8.1);
- a TL/AD base state that drifts from woof's own strict primal (the float
  build must give the primal's words).

Each probe runs in a child process, as in test_grad_wrfplus_hdiff2.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/wrfplus/kcoef2.npz"


def _has_cuda() -> bool:
    try:
        import cupy  # noqa: F401
        return bool(getattr(cupy, "__woof_facade__", False)) is False and \
            cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _has_metal() -> bool:
    try:
        import mlx.core as mx
        return bool(mx.metal.is_available())
    except Exception:
        return False


HAVE_CXX = any(shutil.which(c) for c in ("clang++", "g++", "c++"))
BACKENDS = [pytest.param("host", marks=pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")),
            pytest.param("cuda", marks=pytest.mark.skipif(not _has_cuda(), reason="no CUDA card")),
            pytest.param("mlx", marks=pytest.mark.skipif(not _has_metal(), reason="no Metal"))]

KINDS = ("k2", "n2", "k3")
# Periodic, bounded (both axes, one), the shortest column and a two-cell
# line (periodic self-reads), the cold seam, isotropic km_opt 3 and a
# bounded channel with the cold seam.
CONFIGS = [dict(nz=5, ny=4, nx=5), dict(nz=5, ny=4, nx=5, boundary_x=1, boundary_y=1),
           dict(nz=4, ny=4, nx=5, boundary_x=1), dict(nz=3, ny=1, nx=2),
           dict(nz=4, ny=4, nx=5, seam_cold=1), dict(nz=4, ny=3, nx=4, isotropic=1),
           dict(nz=6, ny=5, nx=6, boundary_y=1, seam_cold=1)]

STRICT_ENV = {"GPUWM_WRF_EXACT": "1", "WOOF_WRF_EXACT_DIAGNOSTICS": "1",
              "WOOF_WRF_EXACT_ADVECTION": "1", "GPUWM_WRF_EXACT_DIFFUSION": "1",
              "GPUWM_WRF_EXACT_BIGSTEP": "1"}


def _run(backend, body, tmp_path, extra_env=None):
    env = dict(os.environ, WOOF_BACKEND=backend, PYTHONPATH=str(ROOT),
               WOOF_HOST_CACHE=str(tmp_path / "hostcache"), WOOF_HOST_THREADS="4")
    env.update(extra_env or {})
    prelude = (
        "import json, numpy as np, woof\nimport cupy as cp\n"
        "from woof.grad.wrfplus import diffusion as D, cases as C, _checks as K\n"
        "out = {}\n"
        "OPS = {'k2': (D.K2_STATE, D.K2_ACTIVE, D.k2_tl, D.k2_ad),\n"
        "       'n2': (D.N2_STATE, D.N2_ACTIVE, D.n2_tl, D.n2_ad),\n"
        "       'k3': (D.K3_STATE, D.K3_ACTIVE, D.k3_tl, D.k3_ad)}\n"
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n"
        "def run_tl(op, g, tr, pe, nl, fill=0.0):\n"
        "    S, _, tl, _ = OPS[op]\n"
        "    st = {s: cp.full_like(tr['bn2'], fill) for s in S}\n"
        "    sd = {s: cp.full_like(tr['bn2'], fill) for s in S}\n"
        "    tl(st, sd, tr, pe, g, nl=nl)\n"
        "    return st, sd\n"
        "def run_ad(op, g, tr, sa, adj=None):\n"
        "    S, X, _, ad = OPS[op]\n"
        "    sa = {s: sa[s].copy() for s in S}\n"
        "    if adj is None:\n"
        "        adj = {n: cp.zeros_like(tr[n]) for n in X}\n"
        "    ad(sa, tr, adj, g)\n"
        "    return sa, adj\n"
        "def seed(op, tr, s, dt):\n"
        "    rng = np.random.default_rng(s)\n"
        "    return {n: cp.asarray(rng.standard_normal(tr['bn2'].shape).astype(dt))\n"
        "            for n in OPS[op][0]}\n")
    script = prelude + body + "\nprint('RESULT ' + json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=3600)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


@pytest.mark.parametrize("backend", BACKENDS)
def test_dot_product(backend, tmp_path):
    """<TL dx, dy> == <dx, AD dy>: 1e-12 in binary64; 1e-5 in binary32,
    relative to the term magnitudes; flat and steep terrain."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f"""
for dtn in {dtypes!r}:
    dt = np.dtype(dtn)
    for op in {KINDS!r}:
        S, X = OPS[op][:2]
        for i, over in enumerate({CONFIGS!r}):
            for steep in (False, True):
                g = C.kc_geometry(**over)
                tr = dev(C.kc_trajectory(g, 11 + i, dt, hybrid=bool(i % 2), steep=steep))
                p = dev(C.kc_perturbation(g, 100 + i, dt))
                dy = seed(op, tr, 200 + i, dt)
                def tl(dx):
                    sd = run_tl(op, g, tr, dict(zip(X, dx)), False)[1]
                    return [sd[s] for s in S]
                def ad(dys):
                    return [run_ad(op, g, tr, dict(zip(S, dys)))[1][n] for n in X]
                out[f"{{dtn}}/{{op}}/{{i}}/{{steep}}"] = K.dot_product(
                    tl, ad, [p[n] for n in X], [dy[s] for s in S], terms=dtn == "float32")
""", tmp_path)
    assert len(out) == (1 if backend == "mlx" else 2) * len(KINDS) * len(CONFIGS) * 2
    for key, rel in out.items():
        limit = 1e-12 if key.startswith("float64") else 1e-5
        assert rel < limit, (key, rel)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tl_overwrites_and_adjoint_consumes(backend, tmp_path):
    """The TL overwrites the coefficients and their tangents (whatever
    they held); the AD adds into ``adj`` and zeroes the coefficient
    adjoints (the primal overwrites the coefficients)."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f"""
dt = np.dtype({dtn!r})
for op in {KINDS!r}:
    S, X = OPS[op][:2]
    for i, over in enumerate({CONFIGS!r}):
        g = C.kc_geometry(**over)
        tr = dev(C.kc_trajectory(g, 51 + i, dt))
        p = dev(C.kc_perturbation(g, 700 + i, dt))
        a0, d0 = run_tl(op, g, tr, p, True)
        a1, d1 = run_tl(op, g, tr, p, True, fill=7.0)
        for s in S:
            out[f"{{op}}/{{i}}/tl/{{s}}"] = int((cp.asnumpy(a0[s]) != cp.asnumpy(a1[s])).sum()
                                               + (cp.asnumpy(d0[s]) != cp.asnumpy(d1[s])).sum())
        q = seed(op, tr, 500 + i, dt)
        s0 = dev(C.kc_perturbation(g, 600 + i, dt))
        sa, zero = run_ad(op, g, tr, q)
        _, adj = run_ad(op, g, tr, q, {{n: s0[n].copy() for n in X}})
        for s in S:
            out[f"{{op}}/{{i}}/st/{{s}}"] = float(cp.abs(sa[s]).max())
        for n in X:
            want = cp.asnumpy(s0[n]).astype(np.float64) + cp.asnumpy(zero[n])
            got = cp.asnumpy(adj[n]).astype(np.float64)
            scale = max(float(np.abs(want).max()), 1e-300)
            out[f"{{op}}/{{i}}/{{n}}"] = float(np.abs(got - want).max()) / scale
""", tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-13
    bad = {k: v for k, v in out.items()
           if (v != 0 if ("/st/" in k or "/tl/" in k) else not v < limit)}
    assert not bad, bad


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x) - a TL dx|| / ||a TL dx|| -> 0 as O(a), flat
    and steep (the slope limiter alpha active)."""
    out = _run(backend, f"""
for op in {KINDS!r}:
    S, X = OPS[op][:2]
    for i, over in enumerate({CONFIGS!r}):
        for steep in (False, True):
            g = C.kc_geometry(**over)
            tr = dev(C.kc_trajectory(g, 31 + i, np.float64, hybrid=bool(i % 2), steep=steep))
            p = dev(C.kc_perturbation(g, 300 + i, np.float64))
            def model(xs):
                t = dict(tr); t.update(dict(zip(X, xs)))
                st = run_tl(op, g, t, {{n: cp.zeros_like(t[n]) for n in X}}, True)[0]
                return [st[s] for s in S]
            def tl(dx):
                sd = run_tl(op, g, tr, dict(zip(X, dx)), False)[1]
                return [sd[s] for s in S]
            out[f"{{op}}/{{i}}/{{steep}}"] = K.tl_residual(
                model, tl, [tr[n] for n in X], [p[n] for n in X], [1e-2, 1e-4, 1e-6])
""", tmp_path)
    assert len(out) == len(KINDS) * len(CONFIGS) * 2
    for key, rows in out.items():
        misfits = [r for _, r in rows]
        assert misfits[-1] < 1e-5, (key, rows)
        assert misfits[0] < 0.1, (key, rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrf_oracle_agreement(backend, tmp_path):
    """woof's base state, TL and AD reproduce WRF 4.7.1's nonlinear
    smag2d_km, calculate_N2 and smag_km (with compute_diff_metrics and,
    for smag_km, cal_deform_and_div) on the words WRF writes: the base to
    1e-13 in binary64, the TL and AD to 1e-8 of their scale (fourth-order
    central differences of WRF, good to about 1e-10)."""
    if not FIXTURE.exists():
        pytest.skip("tests/data/wrfplus/kcoef2.npz not built")
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f"""
import json as J
dt = np.dtype({dtn!r})
z = np.load({str(FIXTURE)!r})
for case in sorted({{k.split("/")[0] for k in z.files if "/" in k}}):
    pre = case + "/"
    op = str(z[pre + "op"])
    S, X = OPS[op][:2]
    g = D.KCGeometry(**J.loads(str(z[pre + "geometry"])))
    pick = lambda kind: {{k.split("/", 2)[2]: cp.asarray(z[k].astype(dt))
                          for k in z.files if k.startswith(pre + kind + "/")}}
    tr, pe, sd = pick("traj"), pick("pert"), pick("seed")
    m = z[pre + "wrfplus/mask"]
    st, dd = run_tl(op, g, tr, pe, True)
    full = {{s: sd.get(s, cp.zeros_like(tr["bn2"])) for s in S}}
    _, adj = run_ad(op, g, tr, full)
    got = {{}}
    for s in S:
        got["nl_" + s] = cp.asnumpy(st[s]).astype(np.float64)
        got["tl_d_" + s] = cp.asnumpy(dd[s]).astype(np.float64)
    for n in X:
        got["ad_" + n] = cp.asnumpy(adj[n]).astype(np.float64)
    out[pre + "words"] = int(m.sum())
    for name, w in got.items():
        e = z[pre + "wrfplus/" + name]
        if not name.startswith("ad_"):
            e, w = e[m], w[m]
        scale = max(float(np.abs(e).max()), 1e-300)
        out[pre + name] = float(np.abs(w - e).max()) / scale
""", tmp_path)
    for key, v in out.items():
        if key.endswith("/words"):
            assert v > 0, key
            continue
        if backend == "mlx":
            limit = 1e-4
        else:
            limit = 1e-13 if "/nl_" in key else 1e-8
        assert v < limit, (key, v)
    assert {k.split("/")[0].split("_")[1] for k in out} == set(KINDS)


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_float_base_state_is_the_strict_primal(backend, tmp_path):
    """The float build's base state gives woof's strict words: smag2d_km's
    xkmh/xkhh (wrf_smag2d_km + wrf_smag_km_bc as
    dycore.launch_wrf_smag2d_km runs them), calc_n2's BN2 (dry) and
    smag3d_km's four coefficients, flat and steep, so the adjoint is bound
    to the model woof runs.  The cold seam, isotropic mixing and steep
    terrain (smag2d_km's alpha) are checked to move the words.  Host and
    CUDA."""
    out = _run(backend, f"""
from woof.core.kernels import get_kernel
from woof.core.moist_n2_mutation import calc_n2_kernel
from woof.core import dycore
f32, i32 = np.float32, np.int32
def words(a):
    return cp.asnumpy(a).view(np.uint32)
def diff(a, b):
    return int((words(a) != words(b)).sum())
dx, dy = 3000.0, 2500.0
def primal(g, tr):
    nz, ny, nx = g.nz, g.ny, g.nx
    ones = cp.ones((nz, ny, nx), f32)
    one = cp.ones(nz, f32)
    common = (tr["u"], tr["v"], tr["w"], tr["php"], tr["phb"], ones, ones, tr["msft"],
              tr["msfu"], tr["msfv"], one, one, one, one,
              f32(g.rdx), f32(g.rdy), f32(dx), f32(dy), f32(0), f32(0), f32(0), i32(0))
    dims = (i32(nz), i32(ny), i32(nx), i32(1)) + tuple(g.h2().args(f32)[5:7])
    grid = ((nx + 63) // 64, ny, nz)
    def bc(a, b):
        if g.boundary_x or g.boundary_y:
            get_kernel("smag2d", "wrf_smag_km_bc")(grid, (64, 1, 1), (
                a, b, i32(nz), i32(ny), i32(nx), i32(g.boundary_x), i32(g.boundary_y)))
    z = lambda: cp.full((nz, ny, nx), 7, f32)
    km, kh = z(), z()
    get_kernel("smag2d", "wrf_smag2d_km")(grid, (64, 1, 1), common + (
        f32(g.c_s), f32(dycore._PRANDTL), tr["defor11"], tr["defor22"], tr["defor12"],
        km, kh) + dims)
    bc(km, kh)
    bn = z()
    calc_n2_kernel()(grid, (64, 1, 1), common + (
        tr["thp"], tr["thb"], i32(1), ones, ones, ones, i32(0), i32(0), bn) + dims)
    r = [z() for _ in range(4)]
    get_kernel("smag2d", "wrf_smag3d_km")(grid, (64, 1, 1), common + (
        f32(g.c_s), f32(dycore._PRANDTL), f32(g.dt), f32(g.mix_upper_bound),
        i32(g.isotropic), tr["defor11"], tr["defor22"], tr["defor12"], tr["bn2"]) + tuple(r)
        + dims)
    bc(r[0], r[1]); bc(r[2], r[3])
    return {{"k2": dict(zip(D.K2_STATE, (km, kh))), "n2": {{"bn2": bn}},
             "k3": dict(zip(D.K3_STATE, r))}}
def woof(g, tr):
    zero = {{n: cp.zeros_like(tr[n]) for n in D.K3_ACTIVE + ("thp",)}}
    return {{op: run_tl(op, g, tr, zero, True, fill=7.0)[0] for op in OPS}}
import dataclasses
for i, over in enumerate({CONFIGS!r}):
    for steep in (False, True):
        g = C.kc_geometry(rdx=float(f32(1 / dx)), rdy=float(f32(1 / dy)), dx=dx, dy=dy, **over)
        tr = dev(C.kc_trajectory(g, 41 + i, f32, hybrid=bool(i % 2), steep=steep))
        p, w = primal(g, tr), woof(g, tr)
        for op in OPS:
            out[f"{{op}}/{{i}}/{{steep}}"] = sum(diff(w[op][s], p[op][s]) for s in w[op])
        key = None
        if g.seam_cold:
            key, other = "seam", dataclasses.replace(g, seam_cold=0)
        elif g.isotropic:
            key, other = "iso", dataclasses.replace(g, isotropic=0)
        if key:
            t2, w1 = tr, w
            if key == "seam":
                # the seam reaches the coefficients through smag2d_km's
                # alpha only: a top-level slope across the seam face
                t2 = dict(tr)
                t2["php"] = tr["php"].copy()
                t2["php"][-1, :, 0] += f32(3.0e5)
                w1 = woof(g, t2)
            w2 = woof(other, t2)
            out[f"moves/{{key}}/{{i}}/{{steep}}"] = int(
                sum(diff(w1[op][s], w2[op][s]) for op in OPS for s in w1[op]) == 0)
        if steep and min(g.nx, g.ny) >= 4 and not g.boundary_x and not g.boundary_y:
            flat = dict(tr)
            flat["phb"] = cp.asarray(C.kc_trajectory(g, 41 + i, f32, hybrid=bool(i % 2))["phb"])
            w2 = woof(g, flat)
            out[f"moves/alpha/{{i}}"] = int(diff(w["k2"]["xkmh"], w2["k2"]["xkmh"]) == 0)
""", tmp_path, extra_env=STRICT_ENV)
    assert len([k for k in out if not k.startswith("moves")]) == len(CONFIGS) * 2 * len(KINDS)
    assert any(k.startswith("moves/seam") for k in out)
    assert any(k.startswith("moves/iso") for k in out)
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_one_dimensional_base_theta(backend, tmp_path):
    """calculate_N2 takes the base theta as a column profile or a 3-D
    field; a 1-D thb broadcast gives the same words as its 3-D image."""
    out = _run(backend, f"""
for i, over in enumerate({CONFIGS!r}[:3]):
    g = C.kc_geometry(**over)
    tr = C.kc_trajectory(g, 61 + i, np.float64, hybrid=True)
    col = tr["thb"][:, 0, 0].copy()
    tr["thb"] = np.broadcast_to(col[:, None, None], tr["thb"].shape).copy()
    t3 = dev(tr)
    t1 = dict(t3); t1["thb"] = cp.asarray(col)
    p = dev(C.kc_perturbation(g, 71 + i, np.float64))
    a3, d3 = run_tl("n2", g, t3, p, True)
    a1, d1 = run_tl("n2", g, t1, p, True)
    out[str(i)] = int((cp.asnumpy(a3["bn2"]) != cp.asnumpy(a1["bn2"])).sum()
                      + (cp.asnumpy(d3["bn2"]) != cp.asnumpy(d1["bn2"])).sum())
""", tmp_path)
    assert len(out) == 3 and not any(out.values()), out


def test_geometry_rejects_unported_options():
    """Too-short columns (calculate_N2's centred levels), empty axes,
    boundary, seam or isotropic flags other than 0/1 and non-positive or
    non-finite spacings are refused, not run."""
    from woof.grad.wrfplus import cases as C
    for over in (dict(nz=2), dict(nx=0), dict(boundary_y=2), dict(seam_cold=2),
                 dict(isotropic=2), dict(rdx=0.0), dict(rdy=float("inf"))):
        with pytest.raises(ValueError):
            C.kc_geometry(**over).validate()
