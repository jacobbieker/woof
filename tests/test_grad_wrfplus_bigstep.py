"""Big-step tendency TL/AD (P3a item 5, woof.grad.wrfplus.bigstep): WRF
horizontal_pressure_gradient and pg_buoy_w on woof's strict slow_pgf and
slow_buoyancy (dry), with periodic, open and specified axes, the lid
extrapolation, map factors and hybrid levels.

Breakage this prevents:
- an adjoint that is not the transpose of its TL (dot-product test);
- an adjoint that overwrites instead of adding into its inputs;
- a TL that is not the derivative of the base-state update
  (TL vs finite differences);
- a TL/AD that drifts from WRFPLUS 4.7.1's G_/A_HORIZONTAL_PRESSURE_
  GRADIENT and g_/a_pg_buoy_w (tests/data/wrfplus/bigstep.npz, made by
  tools/wrfplus_oracle);
- a TL/AD base state that drifts from woof's own strict primal (the
  float build must give the primal's words).

Each probe runs in a child process, as in test_grad_wrfplus_small_step.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/wrfplus/bigstep.npz"


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

# Periodic, open/specified (both, one) axes, the lid extrapolation, the
# 3-level minimum, single-cell lines and hybrid levels (odd cases).
CONFIGS = {
    "pgf": [dict(), dict(boundary_x=1, boundary_y=1, top_lid=1), dict(boundary_x=1),
            dict(boundary_y=1, top_lid=1), dict(nz=3, ny=1, nx=2, top_lid=1),
            dict(ny=2, nx=1, boundary_y=1), dict(nz=4, ny=3, nx=3, boundary_x=1, boundary_y=1)],
    "buoy": [dict(), dict(has_msf=1), dict(nz=1, ny=1, nx=1, has_msf=1), dict(nz=2, ny=3, nx=1)],
}


def _run(backend, body, tmp_path, extra_env=None):
    env = dict(os.environ, WOOF_BACKEND=backend, PYTHONPATH=str(ROOT),
               WOOF_HOST_CACHE=str(tmp_path / "hostcache"), WOOF_HOST_THREADS="4")
    env.update(extra_env or {})
    prelude = (
        "import json, numpy as np, woof\nimport cupy as cp\n"
        "from woof.grad.wrfplus import bigstep as B, cases as C, _checks as K\n"
        "out = {}\n"
        "OPS = {\n"
        " 'pgf': (B.PGF_STATE, B.PGF_ACTIVE, C.pgf_geometry, C.pgf_trajectory,\n"
        "         C.pgf_perturbation, B.pgf_tl, B.pgf_ad, B.PGFGeometry),\n"
        " 'buoy': (B.BUOY_STATE, B.BUOY_ACTIVE, C.buoy_geometry, C.buoy_trajectory,\n"
        "          C.buoy_perturbation, B.buoy_tl, B.buoy_ad, B.BuoyGeometry)}\n"
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n"
        "def run_tl(op, g, tr, pe, nl):\n"
        "    St, tlf = OPS[op][0], OPS[op][5]\n"
        "    st = {n: tr[n].copy() for n in St}\n"
        "    sd = {n: cp.zeros_like(tr[n]) for n in St}\n"
        "    tlf(st, sd, tr, pe, g, nl=nl)\n"
        "    return st, sd\n"
        "def run_ad(op, g, tr, sa, adj=None):\n"
        "    St, X, adf = OPS[op][0], OPS[op][1], OPS[op][6]\n"
        "    sa = {n: sa[n].copy() for n in St}\n"
        "    if adj is None:\n"
        "        adj = {n: cp.zeros_like(tr[n]) for n in X}\n"
        "    adf(sa, tr, adj, g)\n"
        "    return sa, adj\n"
        "def seeds(op, tr, s, dt):\n"
        "    rng = np.random.default_rng(s)\n"
        "    return {n: cp.asarray(rng.standard_normal(tr[n].shape).astype(dt)) for n in OPS[op][0]}\n")
    script = prelude + body + "\nprint('RESULT ' + json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


@pytest.mark.parametrize("backend", BACKENDS)
def test_dot_product(backend, tmp_path):
    """<TL dx, dy> == <dx, AD dy>: 1e-12 in binary64; 1e-5 in binary32,
    relative to the term magnitudes."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f'''
for dtn in {dtypes!r}:
    dt = np.dtype(dtn)
    for op, cfgs in {CONFIGS!r}.items():
        St, X, G, T, P = OPS[op][:5]
        for i, over in enumerate(cfgs):
            g = G(**over)
            tr = dev(T(g, 11 + i, dt, hybrid=bool(i % 2)))
            p = dev(P(g, 100 + i, dt))
            dy = seeds(op, tr, 200 + i, dt)
            def tl(dx):
                return [run_tl(op, g, tr, dict(zip(X, dx)), False)[1][n] for n in St]
            def ad(dys):
                return [run_ad(op, g, tr, dict(zip(St, dys)))[1][n] for n in X]
            out[f"{{dtn}}/{{op}}/{{i}}"] = K.dot_product(
                tl, ad, [p[n] for n in X], [dy[n] for n in St], terms=dtn == "float32")
''', tmp_path)
    n = sum(len(c) for c in CONFIGS.values())
    assert len(out) == (1 if backend == "mlx" else 2) * n
    for key, rel in out.items():
        limit = 1e-12 if key.startswith("float64") else 1e-5
        assert rel < limit, (key, rel)


@pytest.mark.parametrize("backend", BACKENDS)
def test_adjoint_accumulates(backend, tmp_path):
    """The AD adds into ``adj`` and leaves the tendencies' adjoints (which
    pass through, the tendencies being accumulated) unchanged."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
dt = np.dtype({dtn!r})
for op, cfgs in {CONFIGS!r}.items():
    St, X, G, T, P = OPS[op][:5]
    for i, over in enumerate(cfgs):
        g = G(**over)
        tr = dev(T(g, 51 + i, dt, hybrid=bool(i % 2)))
        q = seeds(op, tr, 500 + i, dt)
        s = dev(P(g, 600 + i, dt))
        sa, zero = run_ad(op, g, tr, q)
        _, adj = run_ad(op, g, tr, q, {{n: s[n].copy() for n in X}})
        for n in St:
            out[f"{{op}}/{{i}}/{{n}}"] = float(cp.abs(sa[n] - q[n]).max())
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
for op, cfgs in {CONFIGS!r}.items():
    St, X, G, T, P = OPS[op][:5]
    for i, over in enumerate(cfgs):
        g = G(**over)
        tr = dev(T(g, 31 + i, np.float64, hybrid=bool(i % 2)))
        p = dev(P(g, 300 + i, np.float64))
        def model(xs):
            t = dict(tr); t.update(dict(zip(X, xs)))
            st, _ = run_tl(op, g, t, {{n: cp.zeros_like(t[n]) for n in X}}, True)
            return [st[n] for n in St]
        def tl(dx):
            return [run_tl(op, g, tr, dict(zip(X, dx)), False)[1][n] for n in St]
        out[f"{{op}}/{{i}}"] = K.tl_convergence(model, tl, [tr[n] for n in X],
                                                [p[n] for n in X], [1e-3, 1e-5, 1e-7])
''', tmp_path)
    assert len(out) == sum(len(c) for c in CONFIGS.values())
    for key, rows in out.items():
        misfits = [abs(r - 1.0) for _, r in rows]
        assert misfits[-1] < 1e-6, (key, rows)
        assert misfits[0] < 0.1, (key, rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrfplus_oracle_agreement(backend, tmp_path):
    """woof's TL/AD reproduce WRFPLUS 4.7.1 G_/A_HORIZONTAL_PRESSURE_
    GRADIENT and g_/a_pg_buoy_w on the fixture (6 cases: periodic,
    specified and open axes, the lid, map factors; WRFPLUS's versions are
    not hybrid).  With the lid, WRFPLUS's adjoint of p at the top three
    levels is not the transpose of its own TL: the reverse sweep of
    A_HORIZONTAL_PRESSURE_GRADIENT still applies the pre-2016 lid
    extrapolation (cf1-cf3 on kde-1..kde-3) where the forward sweep and the
    TL use cfn/cfn1.  Those levels are left out; woof's AD there is held
    to woof's TL (which matches WRFPLUS's TL) by the dot-product test."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
import json as J
z = np.load({str(FIXTURE)!r})
dt = np.dtype({dtn!r})
for case in sorted({{k.split("/")[0] for k in z.files if "/" in k}}):
    pre = case + "/"
    op = str(z[pre + "op"])
    St, X, Geo = OPS[op][0], OPS[op][1], OPS[op][7]
    g = Geo(**J.loads(str(z[pre + "geometry"])))
    pick = lambda kind: {{k.split("/", 2)[2]: cp.asarray(z[k].astype(dt))
                          for k in z.files if k.startswith(pre + kind + "/")}}
    tr, pe, sd = pick("traj"), pick("pert"), pick("seed")
    st, dd = run_tl(op, g, tr, pe, True)
    _, adj = run_ad(op, g, tr, sd)
    got = {{}}
    for n in St:
        got["tl_" + n], got["tl_d_" + n] = st[n], dd[n]
    for n in X:
        got["ad_" + n] = adj[n]
    for k in z.files:
        if not k.startswith(pre + "wrfplus/"):
            continue
        name = k.split("/", 2)[2]
        e, w = z[k].copy(), cp.asnumpy(got[name]).astype(np.float64)
        if op == "pgf" and g.top_lid and name == "ad_p":
            e, w = e[:g.nz - 3], w[:g.nz - 3]
        scale = max(float(np.abs(e).max()), 1e-300)
        out[pre + name] = float(np.abs(w - e).max()) / scale
''', tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-12
    bad = {k: v for k, v in out.items() if not v < limit}
    assert not bad, sorted(bad.items())
    assert len(out) == 4 * 9 + 2 * 4


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_float_base_state_is_the_strict_primal(backend, tmp_path):
    """The float build's base state gives woof's strict slow_pgf and
    slow_buoyancy words (GPUWM_WRF_EXACT, dry, 3-D base), so the adjoint is
    bound to the model woof runs: a primal edit fails here until the TL/AD
    is re-derived.  Host and CUDA; Metal is not checked here."""
    out = _run(backend, f'''
from woof.core.kernels import get_kernel
f32, i32 = np.float32, np.int32
def diff(a, b):
    return int((cp.asnumpy(a).view(np.uint32) != cp.asnumpy(b).view(np.uint32)).sum())
dx, dy = 3000.0, 2500.0
for i, over in enumerate({CONFIGS["pgf"]!r}):
    g = C.pgf_geometry(rdx=float(f32(1 / dx)), rdy=float(f32(1 / dy)), **over)
    tr = dev(C.pgf_trajectory(g, 41 + i, f32, hybrid=bool(i % 2)))
    st, _ = run_tl("pgf", g, tr, {{n: cp.zeros_like(tr[n]) for n in B.PGF_ACTIVE}}, True)
    ru, rv = tr["ru_t"].copy(), tr["rv_t"].copy()
    nz, ny, nx = g.nz, g.ny, g.nx
    n = nz * (ny + 1) * (nx + 1)
    get_kernel("dycore", "slow_pgf")(((n + 255) // 256,), (256,), (
        ru, rv, tr["p"], tr["pb"], tr["al"], tr["alt"], tr["php"], tr["phb"], tr["mup"],
        tr["mub"], tr["c1h"], tr["c2h"], tr["rdnw"], tr["fnm"], tr["fnp"], f32(g.cf1),
        f32(g.cf2), f32(g.cf3), i32(g.top_lid), f32(g.cfn), f32(g.cfn1), tr["p"], tr["p"],
        i32(0), f32(1 / dx), f32(1 / dy), f32(0.5 / dx), f32(0.5 / dy), i32(g.boundary_x),
        i32(g.boundary_y), i32(1), i32(nz), i32(ny), i32(nx)))
    out[f"pgf/{{i}}"] = diff(st["ru_t"], ru) + diff(st["rv_t"], rv)
for i, over in enumerate({CONFIGS["buoy"]!r}):
    g = C.buoy_geometry(**over)
    tr = dev(C.buoy_trajectory(g, 61 + i, f32, hybrid=bool(i % 2)))
    st, _ = run_tl("buoy", g, tr, {{n: cp.zeros_like(tr[n]) for n in B.BUOY_ACTIVE}}, True)
    rw = tr["rw_t"].copy()
    nz, ny, nx = g.nz, g.ny, g.nx
    q = tr["p"]
    get_kernel("dycore", "slow_buoyancy")((((nz * ny * nx) + 255) // 256,), (256,), (
        rw, q, q, tr["mup"], tr["mup"], q, q, q, q, q, q, q, tr["rdn"], tr["rdnw"],
        tr["c1f"], tr["c1f"], tr["msft"], i32(0), i32(g.has_msf), i32(1), i32(nz), i32(ny),
        i32(nx)))
    out[f"buoy/{{i}}"] = diff(st["rw_t"], rw)
''', tmp_path, extra_env={"GPUWM_WRF_EXACT": "1"})
    assert len(out) == sum(len(c) for c in CONFIGS.values())
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


def test_geometry_rejects_unported_options():
    """A pressure gradient under 3 levels (the bottom extrapolation reads
    levels 0-2) and empty axes are refused, not run."""
    from woof.grad.wrfplus import cases as C
    for make, over in ((C.pgf_geometry, dict(nz=2)), (C.pgf_geometry, dict(nx=0)),
                       (C.buoy_geometry, dict(ny=0))):
        with pytest.raises(ValueError):
            make(**over).validate()
