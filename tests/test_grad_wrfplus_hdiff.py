"""diff_opt 1 horizontal-diffusion rows TL/AD (P3a item 5,
woof.grad.wrfplus.diffusion.hd_tl/hd_ad): WRF horizontal_diffusion for
the u, v, w and scalar rows and horizontal_diffusion_3dmp for theta on
woof's strict wrf_diff_opt1_horizontal, with periodic and bounded axes,
single-cell lines, map factors and hybrid levels.

Breakage this prevents:
- an adjoint that is not the transpose of its TL (dot-product test);
- an adjoint that overwrites instead of adding into its inputs, or that
  touches the tendency's adjoint (the rows accumulate);
- a TL that is not the derivative of the base-state update
  (TL vs finite differences);
- a TL/AD that drifts from WRFPLUS 4.7.1's G_/A_HORIZONTAL_DIFFUSION and
  g_/a_horizontal_diffusion_3dmp (tests/data/wrfplus/hdiff.npz, made by
  tools/wrfplus_oracle);
- a TL/AD base state that drifts from woof's own strict primal (the
  float build must give the primal's words).

Each probe runs in a child process, as in test_grad_wrfplus_diffusion.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/wrfplus/hdiff.npz"


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

ROWS = ("u", "v", "w", "t", "s")
# Periodic, bounded (both axes, one), short columns, single-cell and
# three-cell lines (periodic self-reads, one interior point).
CONFIGS = [dict(), dict(boundary_x=1, boundary_y=1), dict(boundary_x=1), dict(boundary_y=1),
           dict(nz=2, ny=1, nx=2), dict(nz=3, ny=3, nx=1, boundary_y=1),
           dict(nz=2, ny=3, nx=3, boundary_x=1, boundary_y=1)]

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
        "X = D.HD_ACTIVE\n"
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n"
        "def run_tl(g, tr, pe, nl, tend=None):\n"
        "    st = {'tend': tr['tend'].copy() if tend is None else tend.copy()}\n"
        "    sd = {'tend': cp.zeros_like(tr['tend'])}\n"
        "    D.hd_tl(st, sd, tr, pe, g, nl=nl)\n"
        "    return st, sd\n"
        "def run_ad(g, tr, sa, adj=None):\n"
        "    sa = {'tend': sa['tend'].copy()}\n"
        "    if adj is None:\n"
        "        adj = {n: cp.zeros_like(tr[n]) for n in X}\n"
        "    D.hd_ad(sa, tr, adj, g)\n"
        "    return sa, adj\n"
        "def seed(tr, s, dt):\n"
        "    rng = np.random.default_rng(s)\n"
        "    return {'tend': cp.asarray(rng.standard_normal(tr['tend'].shape).astype(dt))}\n")
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
    for row in {ROWS!r}:
        for i, over in enumerate({CONFIGS!r}):
            g = C.hd_geometry(row=row, **over)
            tr = dev(C.hd_trajectory(g, 11 + i, dt, hybrid=bool(i % 2)))
            p = dev(C.hd_perturbation(g, 100 + i, dt))
            dy = seed(tr, 200 + i, dt)
            def tl(dx):
                return [run_tl(g, tr, dict(zip(X, dx)), False)[1]["tend"]]
            def ad(dys):
                return [run_ad(g, tr, {{"tend": dys[0]}})[1][n] for n in X]
            out[f"{{dtn}}/{{row}}/{{i}}"] = K.dot_product(
                tl, ad, [p[n] for n in X], [dy["tend"]], terms=dtn == "float32")
''', tmp_path)
    assert len(out) == (1 if backend == "mlx" else 2) * len(ROWS) * len(CONFIGS)
    for key, rel in out.items():
        limit = 1e-12 if key.startswith("float64") else 1e-5
        assert rel < limit, (key, rel)


@pytest.mark.parametrize("backend", BACKENDS)
def test_adjoint_accumulates(backend, tmp_path):
    """The AD adds into ``adj`` and leaves the tendency's adjoint as it is
    (the primal accumulates into the tendency)."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
dt = np.dtype({dtn!r})
for row in {ROWS!r}:
    for i, over in enumerate({CONFIGS!r}):
        g = C.hd_geometry(row=row, **over)
        tr = dev(C.hd_trajectory(g, 51 + i, dt, hybrid=bool(i % 2)))
        q = seed(tr, 500 + i, dt)
        s = dev(C.hd_perturbation(g, 600 + i, dt))
        sa, zero = run_ad(g, tr, q)
        _, adj = run_ad(g, tr, q, {{n: s[n].copy() for n in X}})
        out[f"{{row}}/{{i}}/st"] = float(cp.abs(sa["tend"] - q["tend"]).max())
        for n in X:
            want = cp.asnumpy(s[n]).astype(np.float64) + cp.asnumpy(zero[n])
            got = cp.asnumpy(adj[n]).astype(np.float64)
            scale = max(float(np.abs(want).max()), 1e-300)
            out[f"{{row}}/{{i}}/{{n}}"] = float(np.abs(got - want).max()) / scale
''', tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-13
    bad = {k: v for k, v in out.items()
           if (v != 0.0 if k.endswith("/st") else not v < limit)}
    assert not bad, bad


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x) - a TL dx|| / ||a TL dx|| -> 0 as O(a)."""
    out = _run(backend, f'''
for row in {ROWS!r}:
    for i, over in enumerate({CONFIGS!r}):
        g = C.hd_geometry(row=row, **over)
        tr = dev(C.hd_trajectory(g, 31 + i, np.float64, hybrid=bool(i % 2)))
        p = dev(C.hd_perturbation(g, 300 + i, np.float64))
        def model(xs):
            t = dict(tr); t.update(dict(zip(X, xs)))
            st, _ = run_tl(g, t, {{n: cp.zeros_like(t[n]) for n in X}}, True)
            return [st["tend"]]
        def tl(dx):
            return [run_tl(g, tr, dict(zip(X, dx)), False)[1]["tend"]]
        out[f"{{row}}/{{i}}"] = K.tl_residual(model, tl, [tr[n] for n in X],
                                              [p[n] for n in X], [1e-1, 1e-3, 1e-5])
''', tmp_path)
    assert len(out) == len(ROWS) * len(CONFIGS)
    for key, rows in out.items():
        misfits = [r for _, r in rows]
        assert misfits[-1] < 1e-6, (key, rows)
        assert misfits[0] < 0.1, (key, rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrfplus_oracle_agreement(backend, tmp_path):
    """woof's TL/AD reproduce WRFPLUS 4.7.1 G_/A_HORIZONTAL_DIFFUSION (u,
    v, w and a scalar row) and g_/a_horizontal_diffusion_3dmp (theta),
    on periodic and specified axes with map factors, c1 = 1 and c2 = 0
    (WRFPLUS's rows take mut as the mass).  A periodic axis's end face
    repeats face 0: the field adjoints of faces 0 and nx (ny) are compared
    as a sum (woof reads face 0 for both, WRF its halo)."""
    if not FIXTURE.exists():
        pytest.skip("tests/data/wrfplus/hdiff.npz not built")
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
import json as J
dt = np.dtype({dtn!r})
z = np.load({str(FIXTURE)!r})
for case in sorted({{k.split("/")[0] for k in z.files if "/" in k}}):
    pre = case + "/"
    g = D.HDGeometry(**J.loads(str(z[pre + "geometry"])))
    pick = lambda kind: {{k.split("/", 2)[2]: cp.asarray(z[k].astype(dt))
                          for k in z.files if k.startswith(pre + kind + "/")}}
    tr, pe, sd = pick("traj"), pick("pert"), pick("seed")
    st, dd = run_tl(g, tr, pe, True)
    sa, adj = run_ad(g, tr, sd)
    got = {{"tl_tend": st["tend"], "tl_d_tend": dd["tend"], "adst_tend": sa["tend"]}}
    for n in X:
        got["ad_" + n] = adj[n]
    for k in z.files:
        if not k.startswith(pre + "wrfplus/"):
            continue
        name = k.split("/", 2)[2]
        e, w = z[k].copy(), cp.asnumpy(got[name]).astype(np.float64)
        for x in (e, w):
            if name == "ad_field" and g.row == "u" and not g.boundary_x:
                x[..., 0] += x[..., -1]; x[..., -1] = 0
            if name == "ad_field" and g.row == "v" and not g.boundary_y:
                x[..., 0, :] += x[..., -1, :]; x[..., -1, :] = 0
        scale = max(float(np.abs(e).max()), 1e-300)
        out[pre + name] = float(np.abs(w - e).max()) / scale
''', tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-12
    bad = {k: v for k, v in out.items() if not v < limit}
    assert not bad, sorted(bad.items())
    # per case: the tendency, its tangent and its adjoint, and 3 input adjoints
    assert len(out) == 6 * len({k.split("/")[0] for k in out})
    assert len({k.split("/")[0] for k in out}) >= 10


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_float_base_state_is_the_strict_primal(backend, tmp_path):
    """The float build's base state gives woof's strict
    wrf_diff_opt1_horizontal words (through dycore's
    launch_coordinate_horizontal, as the model launches it), so the
    adjoint is bound to the model woof runs: a primal edit fails here until
    the TL/AD is re-derived.  The theta row's initial field and the hybrid
    mass must also move words, so those terms are checked.  Host and CUDA;
    Metal is not checked here."""
    out = _run(backend, f'''
from woof.core import dycore
f32 = np.float32
def diff(a, b):
    return int((cp.asnumpy(a).view(np.uint32) != cp.asnumpy(b).view(np.uint32)).sum())
dx, dy = 3000.0, 2500.0
STAG = {{"u": "x", "v": "y", "w": "z", "t": "", "s": ""}}
def primal(g, tr):
    tend = tr["tend"].copy()
    dycore.launch_coordinate_horizontal(
        tr["field"], tr["xk"], tr["mut"], tr["c1"], tr["c2"], tr["msft"], tr["msfu"],
        tr["msfv"], dx, dy, tend, stagger=STAG[g.row], boundary_x=bool(g.boundary_x),
        boundary_y=bool(g.boundary_y), theta_initial=tr["base"] if g.perturb else None)
    return tend
for row in {ROWS!r}:
    for i, over in enumerate({CONFIGS!r}):
        g = C.hd_geometry(row=row, rdx=float(f32(1 / dx)), rdy=float(f32(1 / dy)), **over)
        tr = dev(C.hd_trajectory(g, 41 + i, f32, hybrid=True))
        st, _ = run_tl(g, tr, {{n: cp.zeros_like(tr[n]) for n in X}}, True)
        out[f"{{row}}/{{i}}"] = diff(st["tend"], primal(g, tr))
    g = C.hd_geometry(row=row, rdx=float(f32(1 / dx)), rdy=float(f32(1 / dy)))
    tr = dev(C.hd_trajectory(g, 47, f32, hybrid=True))
    st, _ = run_tl(g, tr, {{n: cp.zeros_like(tr[n]) for n in X}}, True)
    t0 = dict(tr); t0["c1"] = cp.ones_like(tr["c1"]); t0["c2"] = cp.zeros_like(tr["c2"])
    s0, _ = run_tl(g, t0, {{n: cp.zeros_like(tr[n]) for n in X}}, True)
    out[f"{{row}}/hybrid_moves"] = int(diff(st["tend"], s0["tend"]) == 0)
    if g.perturb:
        t1 = dict(tr); t1["base"] = cp.zeros_like(tr["base"])
        s1, _ = run_tl(g, t1, {{n: cp.zeros_like(tr[n]) for n in X}}, True)
        out[f"{{row}}/base_moves"] = int(diff(st["tend"], s1["tend"]) == 0)
''', tmp_path, extra_env=STRICT_ENV)
    assert len(out) == len(ROWS) * (len(CONFIGS) + 1) + 1
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


def test_geometry_rejects_unported_options():
    """An unknown row, empty axes, a boundary flag other than 0/1 and
    non-positive or non-finite spacings are refused, not run."""
    from woof.grad.wrfplus import cases as C
    for over in (dict(row="q"), dict(nx=0), dict(boundary_y=2), dict(rdx=0.0),
                 dict(rdy=float("inf"))):
        with pytest.raises(ValueError):
            C.hd_geometry(**over).validate()
