"""advance_mu_t's tangent-linear and adjoint (woof's strict advance_mu_th
and the specified-frame advance_exact_frame_mu_t; P3a, woof.grad.wrfplus).

Breakage this prevents:
- an adjoint that is not the transpose of its TL (dot-product test);
- a TL that is not the derivative of the base-state update
  (TL vs finite differences);
- a TL/AD that drifts from WRFPLUS 4.7.1's g_/a_advance_mu_t and
  g_/a_spec_bdyupdate
  (tests/data/wrfplus/advance_mu_t.npz, made by tools/wrfplus_oracle).
  WRF has no counterpart for the history copies (mu_old, p_old, frame
  th_old) or for consuming ww_pp's levels 0 and nz, so the fixture maps
  those through from its inputs rather than comparing them; the
  dot-product and finite-difference tests cover them. The fixture runs
  spec_zone = 1 only;
- a TL/AD base state that drifts from woof's own strict primal
  advance_mu_th[_msf] + advance_exact_frame_mu_t (the float build must
  give the primal's words).

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
FIXTURE = ROOT / "tests/data/wrfplus/advance_mu_t.npz"


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
        "def run_tl(g, tr, pe, nl):\n"
        "    st = {n: tr[n].copy() for n in A.MT_STATE}\n"
        "    sd = {n: pe[n].copy() for n in A.MT_STATE}\n"
        "    A.advance_mu_t_tl(st, sd, tr, pe, g, nl=nl)\n"
        "    return st, sd\n"
        "def run_ad(g, tr, sa, adj=None):\n"
        "    sa = {n: sa[n].copy() for n in A.MT_STATE}\n"
        "    if adj is None:\n"
        "        adj = {n: cp.zeros_like(tr[n]) for n in A.MT_ACTIVE}\n"
        "    A.advance_mu_t_ad(sa, tr, adj, g)\n"
        "    return sa, adj\n"
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n"
        "NAMES = A.MT_STATE + A.MT_ACTIVE\n")
    script = prelude + body + "\nprint('RESULT ' + json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


# Periodic and clamped neighbours, spec_zone 0-2 with and without the
# exact frame, map factors, mudf off, the single column and a deep one.
O = dict(open_x=1, open_y=1)
CONFIGS = [dict(), dict(spec_zone=1, **O), dict(spec_zone=2, write_mudf=0, **O),
           dict(open_x=1), dict(mapped=1, spec_zone=1, **O),
           dict(spec_zone=1, open_y=1), dict(nz=3, ny=1, nx=1),
           dict(nz=3, ny=2, nx=3, open_x=1), dict(nz=3, ny=1, nx=1, **O),
           dict(nz=19, ny=5, nx=4, mapped=1, spec_zone=1, exact_frame=0, **O)]


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
            g = C.mt_geometry(**over)
            tr = dev(C.mt_trajectory(g, 11 + i, dt, hybrid=hybrid))
            def tl(dx):
                return [run_tl(g, tr, dict(zip(NAMES, dx)), False)[1][n] for n in A.MT_STATE]
            def ad(dy):
                sa, adj = run_ad(g, tr, dict(zip(A.MT_STATE, dy)))
                return [sa[n] for n in A.MT_STATE] + [adj[n] for n in A.MT_ACTIVE]
            p = dev(C.mt_perturbation(g, 100 + i, dt))
            q = dev(C.mt_perturbation(g, 200 + i, dt))
            out[f"{{dtn}}/{{i}}/{{hybrid}}"] = K.dot_product(
                tl, ad, [p[n] for n in NAMES], [q[n] for n in A.MT_STATE],
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
    g = C.mt_geometry(**over)
    tr = dev(C.mt_trajectory(g, 51 + i, dt, hybrid=bool(i % 2)))
    q = dev(C.mt_perturbation(g, 500 + i, dt))
    s = dev(C.mt_perturbation(g, 600 + i, dt))
    _, zero = run_ad(g, tr, q)
    _, adj = run_ad(g, tr, q, {{n: s[n].copy() for n in A.MT_ACTIVE}})
    for n in A.MT_ACTIVE:
        want = cp.asnumpy(s[n]).astype(np.float64) + cp.asnumpy(zero[n])
        got = cp.asnumpy(adj[n]).astype(np.float64)
        scale = max(float(np.abs(want).max()), 1e-300)
        out[f"{{i}}/{{n}}"] = float(np.abs(got - want).max()) / scale
''', tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-13
    bad = {k: v for k, v in out.items() if not v < limit}
    assert not bad, bad


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x)|| / ||a TL dx|| -> 1, the misfit O(a)."""
    out = _run(backend, f'''
for i, over in enumerate({CONFIGS!r}):
    g = C.mt_geometry(**over)
    tr = dev(C.mt_trajectory(g, 31 + i, np.float64, hybrid=bool(i % 2)))
    p = dev(C.mt_perturbation(g, 300 + i, np.float64))
    def model(xs):
        t = dict(tr); t.update(dict(zip(NAMES, xs)))
        zero = {{n: cp.zeros_like(t[n]) for n in NAMES}}
        st, _ = run_tl(g, t, zero, True)
        return [st[n] for n in A.MT_STATE]
    def tl(dx):
        return [run_tl(g, tr, dict(zip(NAMES, dx)), False)[1][n] for n in A.MT_STATE]
    out[str(i)] = K.tl_convergence(model, tl, [tr[n] for n in NAMES],
                                   [p[n] for n in NAMES], [1e-2, 1e-4, 1e-6])
''', tmp_path)
    for key, rows in out.items():
        misfits = [abs(r - 1.0) for _, r in rows]
        assert misfits[-1] < 1e-5, (key, rows)
        assert misfits[0] < 0.1, (key, rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrfplus_oracle_agreement(backend, tmp_path):
    """woof's TL/AD reproduce WRFPLUS 4.7.1 g_/a_advance_mu_t followed by
    g_/a_spec_bdyupdate of t_2, mu_2 and muts on the fixture (spec_zone 1,
    as WRFPLUS's specified boundary).  muu/muv/mut map to and from mup;
    the history copies (mu_old, th_old, p_old) are mapped through.
    WRFPLUS has no hybrid coordinate, so the fixture has c1 = 1, c2 = 0;
    the c1h/c2h terms and the periodic/clamped neighbours are covered by
    the dot-product, finite-difference and strict-primal tests only."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
import json as J
from woof.grad.wrfplus.acoustic import MTGeometry
z = np.load({str(FIXTURE)!r})
dt = np.dtype({dtn!r})
cases = sorted({{k.split("/")[0] for k in z.files if "/" in k}})
for case in cases:
    g = MTGeometry(**J.loads(str(z[case + "/geometry"])))
    pick = lambda pre: {{k.split("/", 2)[2]: cp.asarray(z[k].astype(dt))
                         for k in z.files if k.startswith(case + "/" + pre + "/")}}
    tr, pe, seed = pick("traj"), pick("pert"), pick("seed")
    exp = {{k.split("/", 2)[2]: z[k] for k in z.files if k.startswith(case + "/wrfplus/")}}
    st, sd = run_tl(g, tr, pe, True)
    sa, adj = run_ad(g, tr, seed)
    got = {{"tl_" + n: st[n] for n in A.MT_STATE}}
    got.update({{"tl_d_" + n: sd[n] for n in A.MT_STATE}})
    got.update({{"ad_" + n: sa[n] for n in A.MT_STATE}})
    got.update({{"ad_" + n: adj[n] for n in A.MT_ACTIVE}})
    for name, e in exp.items():
        w = cp.asnumpy(got[name]).astype(np.float64)
        m = z[case + "/mask/" + name] if case + "/mask/" + name in z.files else \
            np.ones(e.shape, bool)
        scale = max(float(np.abs(e[m]).max()), 1e-300)
        out[case + "/" + name] = float(np.abs(w[m] - e[m]).max()) / scale
''', tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-12
    bad = {k: v for k, v in out.items() if not v < limit}
    assert not bad, sorted(bad.items())
    assert len(out) == 3 * 34  # three fixture cases, 34 compared arrays each


@pytest.mark.parametrize("backend", ["host"])
def test_float_base_state_is_the_strict_primal(backend, tmp_path):
    """The float build's base state gives woof's strict primal words
    (advance_mu_th or advance_mu_th_msf, then advance_exact_frame_mu_t on
    a specified frame) in every state array, so the adjoint is bound to the
    model woof runs: a primal edit fails here until the TL/AD is
    re-derived."""
    if not HAVE_CXX:
        pytest.skip("no C++ compiler")
    out = _run(backend, f'''
from woof.core.kernels import get_kernel
f32, i32 = np.float32, np.int32
for i, over in enumerate({CONFIGS!r}):
    g = C.mt_geometry(**over)
    tr = dev(C.mt_trajectory(g, 41 + i, f32, hybrid=bool(i % 2)))
    zero = {{n: cp.zeros_like(tr[n]) for n in NAMES}}
    st, _ = run_tl(g, tr, zero, True)
    T = {{n: tr[n].copy() for n in A.MT_STATE}}
    msf = (tr["msft"], tr["msfu"], tr["msfv"]) if g.mapped else ()
    args = (tr["u_pp"], tr["v_pp"], tr["u"], tr["v"], tr["mup"], T["mu_pp"], T["mu_old"],
            tr["rmu_t"], T["mudf"], i32(g.write_mudf), tr["thp"], tr["thp"], T["th_pp"],
            T["th_old"], tr["rth_t"], T["ww_pp"], tr["ww_ref"], tr["p_pp"], T["p_old"],
            tr["dnw"], tr["rdnw"], tr["fnm"], tr["fnp"], tr["c1h"], tr["c2h"], tr["mub2d"],
            *msf, f32(g.rdx), f32(g.rdy), f32(g.dtau), i32(1), i32(g.nz), i32(g.ny),
            i32(g.nx), i32(g.open_x), i32(g.open_y), i32(g.spec_zone))
    blocks = ((g.ny * g.nx + 255) // 256,)
    name = "advance_mu_th_msf" if g.mapped else "advance_mu_th"
    get_kernel("acoustic", name)(blocks, (256,), args)
    if g.frame:
        get_kernel("acoustic", "advance_exact_frame_mu_t")(blocks, (256,), (
            T["mu_pp"], T["th_pp"], tr["rmu_t"], tr["rth_t"], f32(g.dtau),
            i32(g.spec_zone), i32(not g.open_x), i32(g.nz), i32(g.ny), i32(g.nx),
            T["muts"], tr["mub2d"], tr["mup"]))
    out[str(i)] = {{n: bool((cp.asnumpy(st[n]).view(np.uint32)
                             == cp.asnumpy(T[n]).view(np.uint32)).all()) for n in A.MT_STATE}}
''', tmp_path, extra_env={"GPUWM_WRF_EXACT": "1"})
    bad = {k: v for k, v in out.items() if not all(v.values())}
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
