"""advance_w's tangent-linear and adjoint, with calc_p_rho (P3a,
woof.grad.wrfplus).

Breakage this prevents:
- an adjoint that is not the transpose of its TL (dot-product test);
- a TL that is not the derivative of the base-state update
  (TL vs finite differences);
- a TL/AD that drifts from WRFPLUS 4.7.1's g_/a_advance_w and
  g_/a_calc_p_rho (tests/data/wrfplus/advance_w.npz, made by
  tools/wrfplus_oracle);
- a TL/AD base state that drifts from woof's own strict primal
  advance_w_phi (the float build must give the primal's words);
- damp_opt 3's implicit w damper (dampmag > 0) dropped or mis-linearised:
  the configurations below run it, with the heights (php), mut and the
  reference w (WRF w_save) active.

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
FIXTURE = ROOT / "tests/data/wrfplus/advance_w.npz"


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
        "    st = {n: tr[n].copy() for n in A.W_STATE}\n"
        "    sd = {n: pe[n].copy() for n in A.W_STATE}\n"
        "    A.advance_w_tl(st, sd, tr, pe, g, nl=nl)\n"
        "    return st, sd\n"
        "def run_ad(g, tr, sa, adj=None):\n"
        "    sa = {n: sa[n].copy() for n in A.W_STATE}\n"
        "    if adj is None:\n"
        "        adj = {n: cp.zeros_like(tr[n]) for n in A.W_ACTIVE}\n"
        "    A.advance_w_ad(sa, tr, tr, adj, g)\n"
        "    return sa, adj\n"
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n"
        "NAMES = A.W_STATE + A.W_ACTIVE\n")
    script = prelude + body + "\nprint('RESULT ' + json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


# Frame widths, lids, moisture, 1-D base state, clamped/periodic bottom
# neighbours, the 3-level minimum, a deeper column, and map factors
# (advance_w_phi_msf) with each of those.
CONFIGS = [dict(), dict(spec_zone=1), dict(top_lid=1, moist_cq=1),
           dict(base3d=0, boundary_x=1, boundary_y=1),
           dict(spec_zone=1, top_lid=1, moist_cq=1, boundary_x=1, boundary_y=1),
           dict(nz=3, ny=3, nx=3, spec_zone=1, top_lid=1),
           dict(nz=3, ny=2, nx=1), dict(nz=19, ny=7, nx=6, moist_cq=1, spec_zone=2),
           dict(nz=12, ny=4, nx=5, base3d=0, epssm=0.5, boundary_y=1),
           dict(has_msf=1), dict(has_msf=1, spec_zone=1, top_lid=1, moist_cq=1),
           dict(has_msf=1, nz=12, ny=5, nx=6, base3d=0, boundary_x=1, boundary_y=1,
                moist_cq=1, epssm=0.3),
           # damp_opt 3: the top third, a mapped lidded frame, and a layer
           # deeper than the column (every level damped)
           dict(dampmag=0.6, zdamp=5000.0),
           dict(dampmag=0.6, zdamp=8000.0, has_msf=1, spec_zone=1, top_lid=1, moist_cq=1),
           dict(dampmag=0.9, zdamp=16000.0, nz=12, ny=4, nx=5, base3d=0, epssm=0.3,
                boundary_x=1)]


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
            g = C.w_geometry(**over)
            tr = dev(C.w_trajectory(g, 11 + i, dt, hybrid=hybrid))
            def tl(dx):
                return [run_tl(g, tr, dict(zip(NAMES, dx)), False)[1][n] for n in A.W_STATE]
            def ad(dy):
                sa, adj = run_ad(g, tr, dict(zip(A.W_STATE, dy)))
                return [sa[n] for n in A.W_STATE] + [adj[n] for n in A.W_ACTIVE]
            p = dev(C.w_perturbation(g, 100 + i, dt))
            q = dev(C.w_perturbation(g, 200 + i, dt))
            out[f"{{dtn}}/{{i}}/{{hybrid}}"] = K.dot_product(
                tl, ad, [p[n] for n in NAMES], [q[n] for n in A.W_STATE],
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
    g = C.w_geometry(**over)
    tr = dev(C.w_trajectory(g, 51 + i, dt, hybrid=bool(i % 2)))
    q = dev(C.w_perturbation(g, 500 + i, dt))
    s = dev(C.w_perturbation(g, 600 + i, dt))
    _, zero = run_ad(g, tr, q)
    _, adj = run_ad(g, tr, q, {{n: s[n].copy() for n in A.W_ACTIVE}})
    for n in A.W_ACTIVE:
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
    g = C.w_geometry(**over)
    tr = dev(C.w_trajectory(g, 31 + i, np.float64, hybrid=bool(i % 2)))
    p = dev(C.w_perturbation(g, 300 + i, np.float64))
    def model(xs):
        t = dict(tr); t.update(dict(zip(NAMES, xs)))
        zero = {{n: cp.zeros_like(t[n]) for n in NAMES}}
        st, _ = run_tl(g, t, zero, True)
        return [st[n] for n in A.W_STATE]
    def tl(dx):
        return [run_tl(g, tr, dict(zip(NAMES, dx)), False)[1][n] for n in A.W_STATE]
    out[str(i)] = K.tl_convergence(model, tl, [tr[n] for n in NAMES],
                                   [p[n] for n in NAMES], [1e-2, 1e-4, 1e-6])
''', tmp_path)
    for key, rows in out.items():
        misfits = [abs(r - 1.0) for _, r in rows]
        assert misfits[-1] < 1e-5, (key, rows)
        assert misfits[0] < 0.1, (key, rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrfplus_oracle_agreement(backend, tmp_path):
    """woof's TL/AD reproduce WRFPLUS 4.7.1 g_/a_advance_w and
    g_/a_calc_p_rho on the fixture (solved columns only)."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
import json as J
from woof.grad.wrfplus.acoustic import WGeometry
z = np.load({str(FIXTURE)!r})
dt = np.dtype({dtn!r})
cases = sorted({{k.split("/")[0] for k in z.files if "/" in k}})
for case in cases:
    g = WGeometry(**J.loads(str(z[case + "/geometry"])))
    pick = lambda pre: {{k.split("/", 2)[2]: cp.asarray(z[k].astype(dt))
                         for k in z.files if k.startswith(case + "/" + pre + "/")}}
    tr, pe, seed = pick("traj"), pick("pert"), pick("seed")
    exp = {{k.split("/", 2)[2]: z[k] for k in z.files if k.startswith(case + "/wrfplus/")}}
    col = z[case + "/mask/col"]
    st, sd = run_tl(g, tr, pe, True)
    sa, adj = run_ad(g, tr, seed)
    got = {{"tl_" + n: st[n] for n in A.W_STATE}}
    got.update({{"tl_d_" + n: sd[n] for n in A.W_STATE}})
    got.update({{"ad_" + n: sa[n] for n in A.W_STATE}})
    got.update({{"ad_" + n: adj[n] for n in A.W_ACTIVE}})
    for name, e in exp.items():
        w = cp.asnumpy(got[name]).astype(np.float64)
        m = z[case + "/mask/" + name] if case + "/mask/" + name in z.files else \
            np.broadcast_to(col, e.shape)
        scale = max(float(np.abs(e[m]).max()), 1e-300)
        out[case + "/" + name] = float(np.abs(w[m] - e[m]).max()) / scale
''', tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-12
    bad = {k: v for k, v in out.items() if not v < limit}
    assert not bad, bad
    assert len(out) >= 5 * 30   # five fixture cases, two of them mapped


@pytest.mark.parametrize("backend", ["host"])
def test_float_base_state_is_the_strict_primal(backend, tmp_path):
    """The float build's base-state update gives woof's strict primal
    advance_w_phi's words (w'', phi'', p'', alpha''), and
    advance_w_phi_msf's with map factors, so the adjoint is
    bound to the model woof runs: a primal edit fails here until the
    TL/AD is re-derived."""
    if not HAVE_CXX:
        pytest.skip("no C++ compiler")
    out = _run(backend, f'''
from woof.core.acoustic import _w_phi_kernel
f32, i32 = np.float32, np.int32
for i, over in enumerate({CONFIGS!r}):
    g = C.w_geometry(**over)
    tr = dev(C.w_trajectory(g, 41 + i, f32, hybrid=bool(i % 2)))
    zero = {{n: cp.zeros_like(tr[n]) for n in NAMES}}
    st, _ = run_tl(g, tr, zero, True)
    T = {{n: tr[n].copy() for n in A.W_STATE}}
    thb = cp.zeros_like(tr["thp"])
    args = (T["w_pp"], T["ph_pp"], tr["rw_t"], tr["rph_t"], tr["ww_pp"], tr["mu_pp"],
            tr["mu_pp_old"], tr["th_pp"], tr["th_pp_old"], tr["thp"], thb, tr["php"],
            tr["phb"], tr["alt"], tr["c2a"], tr["a"], tr["alpha"], tr["gam"], tr["mup"],
            tr["u_pp"], tr["v_pp"], tr["w"], tr["ht"], tr["rdn"], tr["rdnw"], tr["fnm"],
            tr["fnp"], tr["c1h"], tr["c2h"], tr["c1f"], tr["c2f"], tr["mub2d"], tr["cqw"],
            i32(g.moist_cq)) + ((tr["msft"],) if g.has_msf else ()) + (
            T["p_pp"], T["al_pp"], f32(g.cf1), f32(g.cf2), f32(g.cf3),
            f32(g.rdx), f32(g.rdy), f32(g.dtau), f32(g.epssm), f32(g.dampmag), f32(g.zdamp),
            i32(g.boundary_x), i32(g.boundary_y), i32(g.spec_zone), i32(g.base3d),
            i32(g.top_lid), i32(g.nz), i32(g.ny), i32(g.nx))
    name = "advance_w_phi_msf" if g.has_msf else "advance_w_phi"
    assert (cp.asnumpy(tr["msft"]) != 1).any() == bool(g.has_msf)
    _w_phi_kernel(name, g.nz)(((g.ny * g.nx + 255) // 256,), (256,), args)
    out[str(i)] = {{n: bool((cp.asnumpy(st[n]).view(np.uint32)
                             == cp.asnumpy(T[n]).view(np.uint32)).all()) for n in A.W_STATE}}
''', tmp_path, extra_env={"GPUWM_WRF_EXACT": "1"})
    bad = {k: v for k, v in out.items() if not all(v.values())}
    assert not bad, bad
    assert len(out) == len(CONFIGS)


def test_damper_scalars_checked():
    """A negative dampmag, or a damper with no layer, is refused."""
    from woof.grad.wrfplus import acoustic as A, cases as C
    for g in (C.w_geometry(dampmag=-0.1), C.w_geometry(dampmag=0.3, zdamp=0.0)):
        with pytest.raises(ValueError):
            A.advance_w_tl({}, {}, {}, {}, g)
        with pytest.raises(ValueError):
            A.advance_w_ad({}, {}, {}, {}, g)
