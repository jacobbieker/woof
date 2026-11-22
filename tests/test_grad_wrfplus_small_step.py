"""small_step_init's and small_step_finish's tangent-linear and adjoint
(woof's strict small_step_init_uv/_column and small_step_finish_uv/
_column; P3a, woof.grad.wrfplus).

Breakage this prevents:
- an adjoint that is not the transpose of its TL (dot-product test);
- a TL that is not the derivative of the base-state update
  (TL vs finite differences);
- a TL/AD that drifts from WRFPLUS 4.7.1's g_/a_small_step_prep +
  g_/a_calc_p_rho and G_/A_SMALL_STEP_FINISH
  (tests/data/wrfplus/small_step.npz, made by tools/wrfplus_oracle);
- a TL/AD base state that drifts from woof's own strict primal (the
  float build must give the primal's words).

Each probe runs in a child process, as in test_grad_wrfplus_advance_uv.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/wrfplus/small_step.npz"


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


def _run(backend, body, tmp_path, extra_env=None):
    env = dict(os.environ, WOOF_BACKEND=backend, PYTHONPATH=str(ROOT),
               WOOF_HOST_CACHE=str(tmp_path / "hostcache"), WOOF_HOST_THREADS="4")
    env.update(extra_env or {})
    prelude = (
        "import json, numpy as np, woof\nimport cupy as cp\n"
        "from woof.grad.wrfplus import acoustic as A, cases as C, _checks as K\n"
        "out = {}\n"
        "OPS = {'init': (A.SI_STATE, A.SI_ACTIVE), 'finish': (A.SF_STATE, A.SF_ACTIVE)}\n"
        "def run_tl(op, g, tr, pe, nl):\n"
        "    S = OPS[op][0]\n"
        "    st = {n: tr[n].copy() for n in S}\n"
        "    sd = {n: pe[n].copy() for n in S}\n"
        "    if op == 'init':\n"
        "        A.small_step_init_tl(st, sd, tr, pe, g, nl=nl)\n"
        "    else:\n"
        "        A.small_step_finish_tl(st, sd, tr, pe, g, nl=nl)\n"
        "    return st, sd\n"
        "def run_ad(op, g, tr, sa, adj=None):\n"
        "    S, X = OPS[op]\n"
        "    sa = {n: sa[n].copy() for n in S}\n"
        "    if adj is None:\n"
        "        adj = {n: cp.zeros_like(tr[n]) for n in X}\n"
        "    if op == 'init':\n"
        "        A.small_step_init_ad(sa, tr, adj, g)\n"
        "    else:\n"
        "        A.small_step_finish_ad(sa, {n: tr[n] for n in S}, tr, adj, g)\n"
        "    return sa, adj\n"
        "def inputs(op):\n"
        "    S, X = OPS[op]\n"
        "    return X if op == 'init' else S + X\n"
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n")
    script = prelude + body + "\nprint('RESULT ' + json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


# Clamped and periodic face masses, rk_step 1 (mtf = msf) and later
# stages, map factors, the diabatic removal, single cells and one level;
# nz 19, 16 and 17 (rk_step 1) cross the adjoints' level chunks
# (AD_KCHUNK = 8), 16 with
# the top chunk ending at nz.
CONFIGS = [dict(), dict(boundary_x=0, boundary_y=0), dict(rk_step=2, has_msf=1),
           dict(nz=1, ny=1, nx=1, rk_step=3),
           dict(ny=1, nx=2, boundary_x=0, remove_hdiab=1, hdiab_dt=3.0, rk_step=2),
           dict(ny=1, nx=1, boundary_x=0, boundary_y=0, rk_step=2),
           dict(nz=19, ny=5, nx=4, has_msf=1, boundary_y=0, rk_step=3,
                remove_hdiab=1, hdiab_dt=6.0),
           dict(nz=16, ny=3, nx=2, boundary_x=0, rk_step=2),
           dict(nz=17, ny=2, nx=3, has_msf=1, boundary_y=0, rk_step=1)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_dot_product(backend, tmp_path):
    """<TL dx, dy> == <dx, AD dy>: 1e-12 in binary64; 1e-5 in binary32,
    relative to the term magnitudes."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f'''
for dtn in {dtypes!r}:
    dt = np.dtype(dtn)
    for i, over in enumerate({CONFIGS!r}):
        for hybrid in (False, True):
            g = C.ss_geometry(**over)
            tr = dev(C.ss_trajectory(g, 11 + i, dt, hybrid=hybrid))
            p = dev(C.ss_perturbation(g, 100 + i, dt))
            q = dev(C.ss_perturbation(g, 200 + i, dt))
            for op, (S, X) in OPS.items():
                N = inputs(op)
                def tl(dx):
                    pe = {{n: cp.zeros_like(p[n]) for n in S}}
                    pe.update(dict(zip(N, dx)))
                    return [run_tl(op, g, tr, pe, False)[1][n] for n in S]
                def ad(dy):
                    sa, adj = run_ad(op, g, tr, dict(zip(S, dy)))
                    return [sa[n] for n in N if n in S] + [adj[n] for n in X]
                out[f"{{dtn}}/{{op}}/{{i}}/{{hybrid}}"] = K.dot_product(
                    tl, ad, [p[n] for n in N], [q[n] for n in S],
                    terms=dtn == "float32")
''', tmp_path)
    for key, rel in out.items():
        limit = 1e-12 if key.startswith("float64") else 1e-5
        assert rel < limit, (key, rel)


@pytest.mark.parametrize("backend", BACKENDS)
def test_adjoint_accumulates(backend, tmp_path):
    """The AD adds into ``adj``: a random start comes back plus the
    zero-start adjoint (a ``=`` for ``+=`` fails here)."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
dt = np.dtype({dtn!r})
for i, over in enumerate({CONFIGS!r}):
    g = C.ss_geometry(**over)
    tr = dev(C.ss_trajectory(g, 51 + i, dt, hybrid=bool(i % 2)))
    q = dev(C.ss_perturbation(g, 500 + i, dt))
    s = dev(C.ss_perturbation(g, 600 + i, dt))
    for op, (S, X) in OPS.items():
        _, zero = run_ad(op, g, tr, q)
        _, adj = run_ad(op, g, tr, q, {{n: s[n].copy() for n in X}})
        for n in X:
            want = cp.asnumpy(s[n]).astype(np.float64) + cp.asnumpy(zero[n])
            got = cp.asnumpy(adj[n]).astype(np.float64)
            scale = max(float(np.abs(want).max()), 1e-300)
            out[f"{{op}}/{{i}}/{{n}}"] = float(np.abs(got - want).max()) / scale
''', tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-13
    bad = {k: v for k, v in out.items() if not v < limit}
    assert not bad, bad


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x)|| / ||a TL dx|| -> 1, the misfit O(a)."""
    out = _run(backend, f'''
for i, over in enumerate({CONFIGS!r}):
    g = C.ss_geometry(**over)
    tr = dev(C.ss_trajectory(g, 31 + i, np.float64, hybrid=bool(i % 2)))
    p = dev(C.ss_perturbation(g, 300 + i, np.float64))
    for op, (S, X) in OPS.items():
        N = inputs(op)
        def model(xs):
            t = dict(tr); t.update(dict(zip(N, xs)))
            zero = {{n: cp.zeros_like(t[n]) for n in S + X}}
            st, _ = run_tl(op, g, t, zero, True)
            return [st[n] for n in S]
        def tl(dx):
            pe = {{n: cp.zeros_like(p[n]) for n in S}}
            pe.update(dict(zip(N, dx)))
            return [run_tl(op, g, tr, pe, False)[1][n] for n in S]
        out[f"{{op}}/{{i}}"] = K.tl_convergence(model, tl, [tr[n] for n in N],
                                                [p[n] for n in N], [1e-2, 1e-4, 1e-6])
''', tmp_path)
    for key, rows in out.items():
        misfits = [abs(r - 1.0) for _, r in rows]
        assert misfits[-1] < 1e-5, (key, rows)
        assert misfits[0] < 0.1, (key, rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrfplus_oracle_agreement(backend, tmp_path):
    """woof's TL/AD reproduce WRFPLUS 4.7.1 on the fixture: init against
    g_/a_small_step_prep followed by g_/a_calc_p_rho (non-hydrostatic,
    step 0, solve_em's t_save and muts), finish against
    G_/A_SMALL_STEP_FINISH.  muu/muv/mut/muus/muvs map to and from
    mup/muts through the face means; the halo columns carry woof's clamped
    or wrapped neighbours.  WRFPLUS has no hybrid coordinate, so the
    fixture has c1 = 1, c2 = 0; the c1/c2 terms are covered by the
    dot-product, finite-difference and strict-primal tests only."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
import json as J
from woof.grad.wrfplus.acoustic import SSGeometry
z = np.load({str(FIXTURE)!r})
dt = np.dtype({dtn!r})
cases = sorted({{k.split("/")[0] for k in z.files if "/" in k}})
for case in cases:
    g = SSGeometry(**J.loads(str(z[case + "/geometry"])))
    pick = lambda pre: {{k.split("/", 2)[2]: cp.asarray(z[k].astype(dt))
                         for k in z.files if k.startswith(case + "/" + pre + "/")}}
    tr, pe, seed = pick("traj"), pick("pert"), pick("seed")
    exp = {{k.split("/", 2)[2]: z[k] for k in z.files if k.startswith(case + "/wrfplus/")}}
    got = {{}}
    for op, (S, X) in OPS.items():
        st, sd = run_tl(op, g, tr, pe, True)
        sa, adj = run_ad(op, g, tr, seed)
        got.update({{op + "/tl_" + n: st[n] for n in S}})
        got.update({{op + "/tl_d_" + n: sd[n] for n in S}})
        got.update({{op + "/ad_" + n: sa[n] for n in S}})
        got.update({{op + "/ad_" + n: adj[n] for n in X}})
    for name, e in exp.items():
        w = cp.asnumpy(got[name]).astype(np.float64)
        scale = max(float(np.abs(e).max()), 1e-300)
        out[case + "/" + name] = float(np.abs(w - e).max()) / scale
''', tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-12
    bad = {k: v for k, v in out.items() if not v < limit}
    assert not bad, sorted(bad.items())
    # init: 18 TL, 14 input and 9 consumed-state adjoints; finish: 12 TL,
    # 6 state and 8 input adjoints.
    assert len(out) == 3 * (18 + 14 + 9 + 12 + 6 + 8)


@pytest.mark.parametrize("backend", ["host"])
def test_float_base_state_is_the_strict_primal(backend, tmp_path):
    """The float build's base state gives woof's strict primal words
    (small_step_init_uv/_column, small_step_finish_uv/_column under
    GPUWM_WRF_EXACT) in every state array, so the adjoint is bound to the
    model woof runs: a primal edit fails here until the TL/AD is
    re-derived."""
    if not HAVE_CXX:
        pytest.skip("no C++ compiler")
    out = _run(backend, f'''
from woof.core.kernels import get_kernel
f32, i32 = np.float32, np.int32
for i, over in enumerate({CONFIGS!r}):
    g = C.ss_geometry(**over)
    tr = dev(C.ss_trajectory(g, 41 + i, f32, hybrid=bool(i % 2)))
    zero = {{n: cp.zeros_like(tr[n]) for n in tr}}
    dims = (i32(g.nz), i32(g.ny), i32(g.nx))
    uvb = ((g.nz * (g.ny + 1) * (g.nx + 1) + 255) // 256,)
    colb = ((g.ny * g.nx + 255) // 256,)
    flags = (i32(g.has_msf), i32(g.boundary_x), i32(g.boundary_y))
    # init
    st, _ = run_tl("init", g, tr, zero, True)
    T = {{n: cp.zeros_like(tr[n]) for n in A.SI_STATE}}
    get_kernel("dycore", "small_step_init_uv")(uvb, (256,), (
        T["u_pp"], T["v_pp"], tr["u0"], tr["v0"], tr["u"], tr["v"], tr["mup0"],
        tr["mup"], tr["mub2d"], tr["c1h"], tr["c2h"], tr["msfu"], tr["msfv"],
        *flags, *dims, i32(g.rk_step)))
    get_kernel("dycore", "small_step_init_column")(colb, (256,), (
        T["w_pp"], T["th_pp"], T["ph_pp"], T["mu_pp"], T["al_pp"], T["p_pp"],
        T["p_pp_old"], tr["w0"], tr["w"], tr["thp0"], tr["thp"], tr["php0"],
        tr["php"], tr["p"], tr["alt"], tr["mup0"], tr["mup"], tr["mub2d"],
        tr["thp"], tr["c1h"], tr["c2h"], tr["c1f"], tr["c2f"], tr["rdnw"],
        tr["msft"], i32(g.has_msf), i32(1), *dims))
    res = {{"init/" + n: (st[n], T[n]) for n in A.SI_STATE}}
    # finish
    st, _ = run_tl("finish", g, tr, zero, True)
    T = {{n: tr[n].copy() for n in A.SF_STATE}}
    get_kernel("dycore", "small_step_finish_uv")(uvb, (256,), (
        T["u"], T["v"], tr["u_pp"], tr["v_pp"], T["mup"], tr["mu_pp"],
        tr["mub2d"], tr["c1h"], tr["c2h"], tr["msfu"], tr["msfv"], *flags,
        *dims, tr["muts"]))
    get_kernel("dycore", "small_step_finish_column")(colb, (256,), (
        T["w"], T["thp"], T["php"], T["mup"], tr["w_pp"], tr["th_pp"],
        tr["ph_pp"], tr["mu_pp"], tr["mub2d"], tr["thp0"], tr["c1h"],
        tr["c2h"], tr["c1f"], tr["c2f"], tr["msft"], tr["h_diabatic"],
        f32(g.hdiab_dt), i32(g.remove_hdiab), i32(g.has_msf), i32(1), *dims,
        tr["muts"]))
    res.update({{"finish/" + n: (st[n], T[n]) for n in A.SF_STATE}})
    out[str(i)] = {{k: bool((cp.asnumpy(a).view(np.uint32)
                             == cp.asnumpy(b).view(np.uint32)).all())
                    for k, (a, b) in res.items()}}
''', tmp_path, extra_env={"GPUWM_WRF_EXACT": "1"})
    bad = {k: [n for n, ok in v.items() if not ok] for k, v in out.items()
           if not all(v.values())}
    assert not bad, bad


def test_forecast_path_does_not_import_wrfplus():
    """Opt-in: woof's step path imports nothing from woof.grad."""
    code = ("import sys, woof.core.dycore, woof.core.acoustic, woof.runtime\n"
            "bad = sorted(m for m in sys.modules if m.startswith('woof.grad'))\n"
            "assert not bad, bad\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
