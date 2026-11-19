"""advance_uv's tangent-linear and adjoint (P3a, woof.grad.wrfplus).

Breakage this prevents:
- an adjoint that is not the transpose of its TL (dot-product test);
- a TL that is not the derivative of the base-state update
  (TL vs finite differences);
- a TL/AD that drifts from WRFPLUS 4.7.1's g_/a_advance_uv
  (tests/data/wrfplus/advance_uv.npz, made by tools/wrfplus_oracle);
- a TL/AD base state that drifts from woof's own strict primal
  advance_uv (the float build must give the primal's words).

Each probe runs in a child process because woof picks its backend once
per process: the host backend (binary64 on the CPU, everywhere), CUDA
(binary64 and binary32) where CuPy sees a card, and Metal (binary32 only;
Metal has no binary64) on Apple silicon.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/wrfplus/advance_uv.npz"


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
        "    u, v = tr['u_pp'].copy(), tr['v_pp'].copy()\n"
        "    ud, vd = pe['u_pp'].copy(), pe['v_pp'].copy()\n"
        "    A.advance_uv_tl(u, v, ud, vd, tr, pe, g, nl=nl)\n"
        "    return u, v, ud, vd\n"
        "def run_ad(g, tr, ua, va):\n"
        "    adj = {n: cp.zeros_like(tr[n]) for n in A.UV_ACTIVE}\n"
        "    A.advance_uv_ad(ua, va, tr, adj, g)\n"
        "    return adj\n"
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n")
    script = prelude + body + "\nprint('RESULT ' + json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


CONFIGS = [dict(), dict(spec_zone=1), dict(top_lid=1, moist_cq=1), dict(base3d=0),
           dict(spec_zone=1, top_lid=1, moist_cq=1, use_emdiv=1, has_msf=1,
                xscale=-30.0, yscale=-30.0),
           dict(spec_zone=2, top_lid=1, moist_cq=1, use_emdiv=1,
                xscale=-30.0, yscale=-30.0),
           dict(top_lid=1, moist_cq=1, use_emdiv=1, has_msf=1,
                xscale=-30.0, yscale=-30.0),
           dict(nz=4, ny=3, nx=1), dict(nz=4, ny=2, nx=2, top_lid=1),
           dict(nz=5, ny=5, nx=3, spec_zone=1, base3d=0, has_msf=1)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_dot_product(backend, tmp_path):
    """<TL dx, dy> == <dx, AD dy>: 1e-12 in binary64; 1e-5 in binary32,
    relative to the term magnitudes (small grids cancel to ~1e-3)."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f'''
for dtn in {dtypes!r}:
    dt = np.dtype(dtn)
    for i, over in enumerate({CONFIGS!r}):
        for hybrid in (False, True):
            g = C.uv_geometry(**over)
            tr = dev(C.uv_trajectory(g, 11 + i, dt, hybrid=hybrid))
            names = ("u_pp", "v_pp") + A.UV_ACTIVE
            def tl(dx):
                pe = dict(zip(names, dx))
                return list(run_tl(g, tr, pe, False)[2:])
            def ad(dy):
                adj = run_ad(g, tr, dy[0], dy[1])
                return [dy[0], dy[1]] + [adj[n] for n in A.UV_ACTIVE]
            p = dev(C.uv_perturbation(g, 100 + i, dt))
            q = dev(C.uv_perturbation(g, 200 + i, dt))
            out[f"{{dtn}}/{{i}}/{{hybrid}}"] = K.dot_product(
                tl, ad, [p[n] for n in names], [q["u_pp"], q["v_pp"]],
                terms=dtn == "float32")
''', tmp_path)
    for key, rel in out.items():
        limit = 1e-12 if key.startswith("float64") else 1e-5
        assert rel < limit, (key, rel)


@pytest.mark.parametrize("backend", BACKENDS)
def test_adjoint_accumulates(backend, tmp_path):
    """The AD adds into ``adj``: a random start comes back plus the
    zero-start adjoint, in every output (a ``=`` for ``+=`` fails here)."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
dt = np.dtype({dtn!r})
for i, over in enumerate({CONFIGS!r}):
    g = C.uv_geometry(**over)
    tr = dev(C.uv_trajectory(g, 51 + i, dt, hybrid=bool(i % 2)))
    q = dev(C.uv_perturbation(g, 500 + i, dt))
    s = dev(C.uv_perturbation(g, 600 + i, dt))
    zero = run_ad(g, tr, q["u_pp"], q["v_pp"])
    adj = {{n: s[n].copy() for n in A.UV_ACTIVE}}
    A.advance_uv_ad(q["u_pp"], q["v_pp"], tr, adj, g)
    for n in A.UV_ACTIVE:
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
    """||NL(x + a dx) - NL(x)|| / ||a TL dx|| -> 1, error falling with a.

    The NL is the joint kernel's base-state update; it is bilinear in the
    trajectory, so the misfit is O(a) and has no truncation floor above
    binary64 roundoff over these decades.
    """
    out = _run(backend, f'''
for i, over in enumerate({CONFIGS!r}):
    g = C.uv_geometry(**over)
    tr = dev(C.uv_trajectory(g, 31 + i, np.float64, hybrid=bool(i % 2)))
    names = ("u_pp", "v_pp") + A.UV_ACTIVE
    p = dev(C.uv_perturbation(g, 300 + i, np.float64))
    def model(xs):
        t = dict(tr); t.update(dict(zip(names, xs)))
        zero = {{n: cp.zeros_like(tr[n]) for n in names}}
        u, v, _, _ = run_tl(g, t, zero, True)
        return [u, v]
    def tl(dx):
        return list(run_tl(g, tr, dict(zip(names, dx)), False)[2:])
    out[str(i)] = K.tl_convergence(model, tl, [tr[n] for n in names],
                                   [p[n] for n in names], [1e-2, 1e-4, 1e-6])
''', tmp_path)
    for key, rows in out.items():
        misfits = [abs(r - 1.0) for _, r in rows]
        assert misfits[-1] < 1e-5, (key, rows)
        assert misfits[0] < 0.1, (key, rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrfplus_oracle_agreement(backend, tmp_path):
    """woof's TL/AD reproduce WRFPLUS 4.7.1 g_/a_advance_uv on the fixture."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
import json as J
from woof.grad.wrfplus.acoustic import UVGeometry
z = np.load({str(FIXTURE)!r})
dt = np.dtype({dtn!r})
cases = sorted({{k.split("/")[0] for k in z.files if "/" in k}})
for case in cases:
    g = UVGeometry(**J.loads(str(z[case + "/geometry"])))
    pick = lambda pre: {{k.split("/", 2)[2]: cp.asarray(z[k].astype(dt))
                         for k in z.files if k.startswith(case + "/" + pre + "/")}}
    tr, pe, seed = pick("traj"), pick("pert"), pick("seed")
    exp = {{k.split("/", 2)[2]: z[k] for k in z.files if k.startswith(case + "/wrfplus/")}}
    mu, mv = z[case + "/mask/u"], z[case + "/mask/v"]
    u, v, ud, vd = run_tl(g, tr, pe, True)
    adj = run_ad(g, tr, seed["u_pp"], seed["v_pp"])
    got = {{"tl_u_pp": u, "tl_v_pp": v, "tl_u_d": ud, "tl_v_d": vd}}
    got.update({{"ad_" + n: adj[n] for n in A.UV_ACTIVE}})
    masks = {{"tl_u_pp": mu, "tl_u_d": mu, "ad_ru_t": mu, "ad_cqu": mu,
              "tl_v_pp": mv, "tl_v_d": mv, "ad_rv_t": mv, "ad_cqv": mv}}
    for name, e in exp.items():
        w = cp.asnumpy(got[name]).astype(np.float64)
        m = masks.get(name, np.ones(e.shape, bool))
        scale = max(float(np.abs(e[m]).max()), 1e-300)
        out[case + "/" + name] = float(np.abs(w[m] - e[m]).max()) / scale
''', tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-12
    bad = {k: v for k, v in out.items() if not v < limit}
    assert not bad, bad
    assert len(out) >= 2 * 17


@pytest.mark.parametrize("backend", ["host"])
def test_float_base_state_is_the_strict_primal(backend, tmp_path):
    """The float build's base-state update gives woof's strict primal
    advance_uv's words (woof/core/kernels/acoustic.cu), so the adjoint is
    bound to the model woof runs: a primal edit fails here until the TL/AD
    is re-derived."""
    if not HAVE_CXX:
        pytest.skip("no C++ compiler")
    out = _run(backend, f'''
from woof.core.kernels import get_kernel
f32 = np.float32
for i, over in enumerate({CONFIGS!r}):
    g = C.uv_geometry(**over)
    tr = dev(C.uv_trajectory(g, 41 + i, f32, hybrid=bool(i % 2)))
    zero = {{n: cp.zeros_like(tr[n]) for n in ("u_pp", "v_pp") + A.UV_ACTIVE}}
    u, v, _, _ = run_tl(g, tr, zero, True)
    up, vp = tr["u_pp"].copy(), tr["v_pp"].copy()
    T = tr
    args = (up, vp, T["ru_t"], T["rv_t"], T["p_pp"], T["p_pp_old"], T["ph_pp"], T["php"],
            T["phb"], T["alt"], T["al_pp"], T["pb"], T["mup"], T["mu_pp"], T["mub2d"],
            T["c1h"], T["c2h"], T["fnm"], T["fnp"], T["rdnw"], T["cqu"], T["cqv"],
            np.int32(g.moist_cq), f32(g.cf1), f32(g.cf2), f32(g.cf3), np.int32(g.top_lid),
            f32(g.rdx), f32(g.rdy), f32(g.dtau), f32(g.smdiv), np.int32(g.spec_zone),
            np.int32(g.base3d), np.int32(g.nz), np.int32(g.ny), np.int32(g.nx),
            T["mudf"], T["msfu"], T["msfv"], f32(g.xscale), f32(g.yscale),
            np.int32(g.use_emdiv), np.int32(g.has_msf))
    n = g.nz * (g.ny + 1) * (g.nx + 1)
    get_kernel("acoustic", "advance_uv")(((n + 255) // 256,), (256,), args)
    out[str(i)] = [bool((cp.asnumpy(u).view(np.uint32) == cp.asnumpy(up).view(np.uint32)).all()),
                   bool((cp.asnumpy(v).view(np.uint32) == cp.asnumpy(vp).view(np.uint32)).all())]
''', tmp_path, extra_env={"GPUWM_WRF_EXACT": "1"})
    assert all(all(v) for v in out.values()), out


def test_forecast_path_does_not_import_wrfplus():
    """Opt-in: woof's step path imports nothing from woof.grad."""
    code = ("import sys, woof.core.dycore, woof.core.acoustic, woof.runtime\n"
            "bad = sorted(m for m in sys.modules if m.startswith('woof.grad'))\n"
            "assert not bad, bad\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
