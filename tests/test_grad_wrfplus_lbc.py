"""Lateral-boundary tendency TL/AD (P3a item 7, woof.grad.wrfplus.lbc):
WRF relax_bdy_dry and spec_bdy_dry on woof's strict
state_relaxation_multi rows, for u, v, theta, phi, w and mu and each row
type the strict stage uses (held capture, specified target, phi's held
compute with the map-factor divide and its held read, mu's direct
relaxation), with map factors, hybrid levels and wider zones.

Breakage this prevents:
- an adjoint that is not the transpose of its TL (dot-product test);
- an adjoint that overwrites instead of adding into its inputs;
- a TL that is not the derivative of the base-state update
  (TL vs finite differences);
- a TL/AD that drifts from WRFPLUS 4.7.1's g_/a_relax_bdytend and
  g_/a_spec_bdytend (tests/data/wrfplus/lbc.npz, made by
  tools/wrfplus_oracle);
- a TL/AD base state that drifts from woof's own strict primal (the
  float build must give state_relaxation_multi's words).

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
FIXTURE = ROOT / "tests/data/wrfplus/lbc.npz"


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

# The strict stage's row types: stage-1 held capture, the specified
# target, phi's stage-1 held compute (map-factor divide), the later
# stages' held read, mu's direct relaxation and a held compute that also
# targets.  mu has only the target and direct rows; u and v never divide.
ROWS = {
    "capture": dict(held_compute=1, held_store=1, target=0),
    "spec": dict(),
    "phi1": dict(held_compute=1, held_store=1, divide_msf=1, has_msf=1),
    "read": dict(held_read=1),
    "direct": dict(apply_relax=1),
    "nostore": dict(held_compute=1, target=1),
}
CONFIGS = [dict(kind=k, **f) for k in ("u", "v", "theta", "phi", "w", "mu")
           for r, f in ROWS.items()
           if not ((k == "mu" and r not in ("spec", "direct")) or (k in "uv" and r == "phi1"))]
CONFIGS += [dict(kind="u", apply_relax=1, has_msf=1), dict(kind="v", apply_relax=1, has_msf=1),
            dict(kind="theta", apply_relax=1, spec_zone=2, relax_zone=5, width=7, ny=14, nx=15),
            dict(kind="u", apply_relax=1, spec_zone=1, relax_zone=1, width=2)]

# Strict mode with each of its four controls pinned on, so an inherited
# =0 cannot leave the reference primal off WRF's order.
STRICT_ENV = {"GPUWM_WRF_EXACT": "1", "WOOF_WRF_EXACT_DIAGNOSTICS": "1",
              "WOOF_WRF_EXACT_ADVECTION": "1", "GPUWM_WRF_EXACT_DIFFUSION": "1",
              "GPUWM_WRF_EXACT_BIGSTEP": "1"}


def _run(backend, body, tmp_path, extra_env=None):
    env = dict(os.environ, WOOF_BACKEND=backend, PYTHONPATH=str(ROOT),
               WOOF_HOST_CACHE=str(tmp_path / "hostcache"), WOOF_HOST_THREADS="4")
    env.update(extra_env or {})
    prelude = (
        "import json, numpy as np, woof\nimport cupy as cp\n"
        "from woof.grad.wrfplus import lbc as L, cases as C, _checks as K\n"
        "out = {}\n"
        "St, X = L.LB_STATE, L.LB_ACTIVE\n"
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n"
        "def run_tl(g, tr, pe, nl):\n"
        "    st = {n: tr[n].copy() for n in St}\n"
        "    sd = {n: cp.zeros_like(tr[n]) for n in St}\n"
        "    L.lbc_tl(st, sd, tr, pe, g, nl=nl)\n"
        "    return st, sd\n"
        "def run_ad(g, tr, sa, adj=None):\n"
        "    sa = {n: sa[n].copy() for n in St}\n"
        "    if adj is None:\n"
        "        adj = {n: cp.zeros_like(tr[n]) for n in X}\n"
        "    L.lbc_ad(sa, tr, adj, g)\n"
        "    return sa, adj\n"
        "def seeds(tr, s, dt):\n"
        "    rng = np.random.default_rng(s)\n"
        "    return {n: cp.asarray(rng.standard_normal(tr[n].shape).astype(dt)) for n in St}\n")
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
    for i, over in enumerate({CONFIGS!r}):
        g = C.lbc_geometry(**over)
        tr = dev(C.lbc_trajectory(g, 11 + i, dt, hybrid=bool(i % 2)))
        p = dev(C.lbc_perturbation(g, 100 + i, dt))
        dy = seeds(tr, 200 + i, dt)
        def tl(dx):
            return [run_tl(g, tr, dict(zip(X, dx)), False)[1][n] for n in St]
        def ad(dys):
            return [run_ad(g, tr, dict(zip(St, dys)))[1][n] for n in X]
        out[f"{{dtn}}/{{i}}"] = K.dot_product(
            tl, ad, [p[n] for n in X], [dy[n] for n in St], terms=dtn == "float32")
''', tmp_path)
    assert len(out) == (1 if backend == "mlx" else 2) * len(CONFIGS)
    for key, rel in out.items():
        limit = 1e-12 if key.startswith("float64") else 1e-5
        assert rel < limit, (key, rel)


@pytest.mark.parametrize("backend", BACKENDS)
def test_adjoint_accumulates(backend, tmp_path):
    """The AD adds into ``adj``: AD with a nonzero start equals the start
    plus AD from zero."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
dt = np.dtype({dtn!r})
for i, over in enumerate({CONFIGS!r}):
    g = C.lbc_geometry(**over)
    tr = dev(C.lbc_trajectory(g, 51 + i, dt, hybrid=bool(i % 2)))
    q = seeds(tr, 500 + i, dt)
    s = dev(C.lbc_perturbation(g, 600 + i, dt))
    sa0, zero = run_ad(g, tr, q)
    sa1, adj = run_ad(g, tr, q, {{n: s[n].copy() for n in X}})
    for n in St:
        out[f"{{i}}/{{n}}"] = float(cp.abs(sa1[n] - sa0[n]).max())
    for n in X:
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
    """||NL(x + a dx) - NL(x) - a TL dx|| / ||a TL dx|| -> 0 as O(a)."""
    out = _run(backend, f'''
for i, over in enumerate({CONFIGS!r}):
    g = C.lbc_geometry(**over)
    tr = dev(C.lbc_trajectory(g, 31 + i, np.float64, hybrid=bool(i % 2)))
    p = dev(C.lbc_perturbation(g, 300 + i, np.float64))
    def model(xs):
        t = dict(tr); t.update(dict(zip(X, xs)))
        st, _ = run_tl(g, t, {{n: cp.zeros_like(t[n]) for n in X}}, True)
        return [st[n] for n in St]
    def tl(dx):
        return [run_tl(g, tr, dict(zip(X, dx)), False)[1][n] for n in St]
    out[f"{{i}}"] = K.tl_residual(model, tl, [tr[n] for n in X],
                                  [p[n] for n in X], [1e-3, 1e-5, 1e-7])
''', tmp_path)
    assert len(out) == len(CONFIGS)
    for key, rows in out.items():
        misfits = [r for _, r in rows]
        assert misfits[-1] < 1e-6, (key, rows)
        assert misfits[0] < 0.1, (key, rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrfplus_oracle_agreement(backend, tmp_path):
    """woof's TL/AD reproduce WRFPLUS 4.7.1 g_relax_bdytend then
    g_spec_bdytend and a_spec_bdytend then a_relax_bdytend on the direct
    row (8 cases: u with and without map factors, v with them, theta,
    phi, w and mu, hybrid levels, a wider spec and relaxation zone).  WRFPLUS's relax_bdy_dry is
    not hybrid and takes the coupled field as given, so the fixture calls
    the core routines on woof's coupled field (formed, with its tangent
    and adjoint, from the row's field and mu')."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
import json as J
dt = np.dtype({dtn!r})
z = np.load({str(FIXTURE)!r})
for case in sorted({{k.split("/")[0] for k in z.files if "/" in k}}):
    pre = case + "/"
    g = L.LBCGeometry(**J.loads(str(z[pre + "geometry"])))
    pick = lambda kind: {{k.split("/", 2)[2]: cp.asarray(z[k].astype(dt))
                          for k in z.files if k.startswith(pre + kind + "/")}}
    tr, pe, sd = pick("traj"), pick("pert"), pick("seed")
    st, dd = run_tl(g, tr, pe, True)
    sa, adj = run_ad(g, tr, sd)
    got = {{}}
    for n in St:
        got["tl_" + n], got["tl_d_" + n], got["adst_" + n] = st[n], dd[n], sa[n]
    for n in X:
        got["ad_" + n] = adj[n]
    for k in z.files:
        if not k.startswith(pre + "wrfplus/"):
            continue
        name = k.split("/", 2)[2]
        e, w = z[k], cp.asnumpy(got[name]).astype(np.float64)
        scale = max(float(np.abs(e).max()), 1e-300)
        out[pre + name] = float(np.abs(w - e).max()) / scale
''', tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-12
    bad = {k: v for k, v in out.items() if not v < limit}
    assert not bad, sorted(bad.items())
    assert len(out) == 8 * 16


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_float_base_state_is_the_strict_primal(backend, tmp_path):
    """The float build's base state gives woof's strict
    state_relaxation_multi words (GPUWM_WRF_EXACT, one row per launch, the
    row table built as woof.ingest.lateral_bc does), so the adjoint is
    bound to the model woof runs: a primal edit fails here until the TL/AD
    is re-derived.  Host and CUDA; Metal is not checked here."""
    out = _run(backend, f'''
from woof.core.kernels import get_kernel
f32, i32 = np.float32, np.int32
def diff(a, b):
    return int((cp.asnumpy(a).view(np.uint32) != cp.asnumpy(b).view(np.uint32)).sum())
for i, over in enumerate({CONFIGS!r}):
    g = C.lbc_geometry(**over)
    tr = dev(C.lbc_trajectory(g, 41 + i, f32, hybrid=bool(i % 2)))
    st, _ = run_tl(g, tr, {{n: cp.zeros_like(tr[n]) for n in X}}, True)
    ref = {{n: tr[n].copy() for n in St}}
    head = [tr[n] for n in ("mub", "mub", "c1h", "c2h", "c1f", "c2f", "msft", "msfu", "msfv",
                            "fcx", "gcx")]
    # 11 shared pointers + pad, then LBC_MULTI_MAX_ROWS (12) rows of 16 words
    table = np.zeros(12 + 12 * 16, dtype=np.uint64)
    table[:11] = [int(a.data.ptr) for a in head]
    flags = (g.held_compute and 1) | (g.held_store and 2) | (g.held_read and 4) \\
        | (g.target and 8) | (g.apply_relax and 16) | (g.divide_msf and 32)
    fz, fy, fx = g.field
    held = ref["held"] if (g.held_read or g.held_store) else None
    words = [int(ref["tend"].data.ptr) if g.target else 0,
             int(held.data.ptr) if held is not None else 0,
             int(tr["src"].data.ptr), int(tr["mup"].data.ptr)]
    for s in ("west", "east", "south", "north"):
        words += [int(tr[s].data.ptr), int(tr[s + "_t"].data.ptr)]
    words += [L.LB_KINDS[g.kind] | (flags << 32), fz | (fy << 32), fx | (g.width << 32), g.aw]
    table[12:12 + 16] = words
    n = fz * fy * fx
    get_kernel("lbc_state", "state_relaxation_multi")(((n + 255) // 256, 1), (256,), (
        table.view(np.dtype(("V", table.nbytes)))[0], i32(1), f32(g.dtbc), i32(g.spec_zone),
        i32(g.relax_zone), i32(g.has_msf), i32(0), i32(15)))
    # the row must change something, or the check is empty
    changed = sum(int((ref[s] != tr[s]).any()) for s in St)
    out[f"{{i}}"] = sum(diff(st[s], ref[s]) for s in St) + int(not changed)
''', tmp_path, extra_env=STRICT_ENV)
    assert len(out) == len(CONFIGS)
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


def test_geometry_rejects_unported_options():
    """spec_zone 0 (the stencil reads outside the row), a table narrower
    than the zones, a domain too small for the frames, conflicting held
    flags, the map-factor divide on u or v, a row that writes nothing and
    an unknown kind are refused, not run."""
    from woof.grad.wrfplus import cases as C
    for over in (dict(spec_zone=0), dict(width=4), dict(ny=10), dict(kind="qv"),
                 dict(held_compute=1, held_read=1), dict(held_store=1),
                 dict(apply_relax=1, held_read=1), dict(divide_msf=1),
                 dict(kind="u", held_compute=1, divide_msf=1), dict(target=0),
                 dict(nz=0)):
        with pytest.raises(ValueError):
            C.lbc_geometry(**over).validate()
