"""diff_opt 2 horizontal-diffusion rows TL/AD (P3a item 2b-1,
woof.grad.wrfplus.diffusion.h2_tl/h2_ad): WRF horizontal_diffusion_2's
horizontal_diffusion_u_2, _v_2, _w_2 and horizontal_diffusion_s (theta and
scalars) with their metrics (zx, zy, rdz, rdzw from ph), the density
(1/alt, times 1 + qv when moist) and the horizontal K as inputs, on woof's
strict wrf_smag_hd_u / _v / _w and wrf_smag_flux_s + wrf_smag_hd_s, with
periodic and bounded axes, single-cell lines and map factors.

Breakage this prevents:
- an adjoint that is not the transpose of its TL (dot-product test);
- an adjoint that overwrites instead of adding into its inputs, or that
  touches the tendency's adjoint (the rows accumulate);
- a TL that is not the derivative of the base-state update
  (TL vs finite differences);
- a base state, TL or AD that drifts from WRF 4.7.1's nonlinear rows
  (tests/data/wrfplus/diffusion2.npz, made by tools/wrfplus_oracle from
  the WRF routines in WRFPLUS 4.7.1's libwrflib.a, with the TL/AD from
  fourth-order differences of them);
- a TL/AD base state that drifts from woof's own strict primal (the
  float build must give the primal's words, through every launch route
  the model takes for these rows).

Each probe runs in a child process, as in test_grad_wrfplus_hdiff.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/wrfplus/diffusion2.npz"


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
# Periodic, bounded (both axes, one), the shortest column, single-cell and
# two-cell lines (periodic self-reads), and moist density.
CONFIGS = [dict(nz=5, ny=4, nx=5), dict(nz=5, ny=4, nx=5, boundary_x=1, boundary_y=1),
           dict(nz=4, ny=4, nx=5, boundary_x=1), dict(nz=4, ny=5, nx=4, boundary_y=1),
           dict(nz=3, ny=1, nx=2), dict(nz=3, ny=3, nx=1, boundary_y=1),
           dict(nz=4, ny=3, nx=4, boundary_x=1, boundary_y=1, moist=1),
           dict(nz=4, ny=4, nx=3, moist=1)]

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
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n"
        "def run_tl(g, tr, pe, nl, tend=None):\n"
        "    st = {'tend': tr['tend'].copy() if tend is None else tend.copy()}\n"
        "    sd = {'tend': cp.zeros_like(tr['tend'])}\n"
        "    D.h2_tl(st, sd, tr, pe, g, nl=nl)\n"
        "    return st, sd\n"
        "def run_ad(g, tr, sa, adj=None):\n"
        "    sa = {'tend': sa['tend'].copy()}\n"
        "    if adj is None:\n"
        "        adj = {n: cp.zeros_like(tr[n]) for n in D.h2_active(g)}\n"
        "    D.h2_ad(sa, tr, adj, g)\n"
        "    return sa, adj\n"
        "def seed(tr, s, dt):\n"
        "    rng = np.random.default_rng(s)\n"
        "    return {'tend': cp.asarray(rng.standard_normal(tr['tend'].shape).astype(dt))}\n")
    script = prelude + body + "\nprint('RESULT ' + json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=3600)
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
            g = C.h2_geometry(row=row, **over)
            X = D.h2_active(g)
            tr = dev(C.h2_trajectory(g, 11 + i, dt, hybrid=bool(i % 2)))
            p = dev(C.h2_perturbation(g, 100 + i, dt))
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
        g = C.h2_geometry(row=row, **over)
        X = D.h2_active(g)
        tr = dev(C.h2_trajectory(g, 51 + i, dt))
        q = seed(tr, 500 + i, dt)
        s = dev(C.h2_perturbation(g, 600 + i, dt))
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
        g = C.h2_geometry(row=row, **over)
        X = D.h2_active(g)
        tr = dev(C.h2_trajectory(g, 31 + i, np.float64))
        p = dev(C.h2_perturbation(g, 300 + i, np.float64))
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
def test_wrf_oracle_agreement(backend, tmp_path):
    """woof's base state, TL and AD reproduce WRF 4.7.1's nonlinear
    horizontal_diffusion_u_2 / _v_2 / _w_2 / _s (with compute_diff_metrics,
    rho and, for w, cal_deform_and_div) on WRF's loop region: the base
    tendency to 1e-13 in binary64 (the same statements), the TL and AD to
    1e-8 of their scale (the fixture differentiates WRF by fourth-order
    central differences, good to about 1e-10).  The tendency adjoint is
    seeded on the loop region only; a periodic staggered end face, which
    woof reads as face 0, is outside it."""
    if not FIXTURE.exists():
        pytest.skip("tests/data/wrfplus/diffusion2.npz not built")
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
import json as J
dt = np.dtype({dtn!r})
z = np.load({str(FIXTURE)!r})
for case in sorted({{k.split("/")[0] for k in z.files if "/" in k}}):
    pre = case + "/"
    g = D.H2Geometry(**J.loads(str(z[pre + "geometry"])))
    X = D.h2_active(g)
    pick = lambda kind: {{k.split("/", 2)[2]: cp.asarray(z[k].astype(dt))
                          for k in z.files if k.startswith(pre + kind + "/")}}
    tr, pe, sd = pick("traj"), pick("pert"), pick("seed")
    m = z[pre + "wrfplus/mask"]
    t0 = z[pre + "traj/tend"]
    st, dd = run_tl(g, tr, pe, True)
    _, adj = run_ad(g, tr, sd)
    got = {{"nl_tend": cp.asnumpy(st["tend"]).astype(np.float64) - t0,
            "tl_d_tend": cp.asnumpy(dd["tend"]).astype(np.float64)}}
    for n in X:
        got["ad_" + n] = cp.asnumpy(adj[n]).astype(np.float64)
    for name, w in got.items():
        e = z[pre + "wrfplus/" + name]
        if name == "nl_tend":
            e = e - t0
        if name.endswith("_tend"):
            e, w = e[m], w[m]
        scale = max(float(np.abs(e).max()), 1e-300)
        out[pre + name] = float(np.abs(w - e).max()) / scale
''', tmp_path)
    for key, v in out.items():
        if backend == "mlx":
            limit = 1e-4
        else:
            limit = 1e-13 if key.endswith("nl_tend") else 1e-8
        assert v < limit, (key, v)
    cases = {k.split("/")[0] for k in out}
    assert len(cases) >= 16
    assert {k.split("/")[0].split("_")[1] for k in out} == {"u", "v", "w", "s", "t"}


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
@pytest.mark.parametrize("route", ["faces", "per_point"])
def test_float_base_state_is_the_strict_primal(backend, route, tmp_path):
    """The float build's base state gives woof's strict diff_opt 2 words
    through dycore.launch_wrf_smag2d_hd as the model launches it (u/v:
    wrf_smag_uv_coef_faces + wrf_smag_hd_u/v_faces, or the per-point
    wrf_smag_hd_u/v with WOOF_SMAG_UV_FACES=0; w: wrf_smag_w_defor_faces +
    wrf_smag_hd_w_faces, or wrf_smag_hd_w with WOOF_SMAG_W_FACES=0; scalars:
    wrf_smag_flux_s + wrf_smag_hd_s, and the multi-row pair), so the
    adjoint is bound to the model woof runs: a primal edit fails here until
    the TL/AD is re-derived.  The rows add every point; the model zeroes
    the open strips afterwards.  Host and CUDA; Metal is not checked here."""
    env = dict(STRICT_ENV)
    if route == "per_point":
        env.update(WOOF_SMAG_UV_FACES="0", WOOF_SMAG_W_FACES="0")
    out = _run(backend, f'''
from types import SimpleNamespace as NS
from woof.core import dycore
f32 = np.float32
def diff(a, b):
    return int((cp.asnumpy(a).view(np.uint32) != cp.asnumpy(b).view(np.uint32)).sum())
dx, dy = 3000.0, 2500.0
STAG = {{"u": "x", "v": "y", "w": "z", "t": "", "s": ""}}
def scratch(shape, name):
    return cp.zeros(shape, dtype=f32)
def setup(g, tr):
    nz, ny, nx = g.nz, g.ny, g.nx
    zero = lambda *s: cp.zeros(s, dtype=f32)
    st = NS(u=tr.get("u", zero(nz, ny, nx + 1)), v=tr.get("v", zero(nz, ny + 1, nx)),
            w=tr.get("w", zero(nz + 1, ny, nx)), php=tr["php"], phb=tr["phb"],
            alt=tr["alt"], qv=tr["qv"] if g.moist else None, msft=tr["msft"],
            msfu=tr["msfu"], msfv=tr["msfv"], fnm=tr["fnm"], fnp=tr["fnp"], dn=tr["dn"],
            dnw=tr["dnw"], cf1=g.cf1, cf2=g.cf2, cf3=g.cf3, thb=zero(nz, ny, nx),
            scratch=scratch)
    cfg = NS(dx=dx, dy=dy, open_x=bool(g.boundary_x), open_y=bool(g.boundary_y),
             specified=False, nested=False)
    return st, cfg
def primal(g, tr, multi=False):
    st, cfg = setup(g, tr)
    tend = tr["tend"].copy()
    # strict storage holds theta - 300, so the theta row's full_theta
    # reconstruction is the identity
    full = g.row == "t"
    if multi:
        dycore.launch_wrf_smag2d_hd_multi(st, cfg, [(tr["field"], tr["xk"], tend, full)],
                                          time_t=False)
        return tend
    f = {{"u": st.u, "v": st.v, "w": st.w}}.get(g.row, tr.get("field"))
    defor = ((tr["defor11"], tr["defor11"], tr["defor12"]) if g.row == "u"
             else (tr["defor22"], tr["defor22"], tr["defor12"]) if g.row == "v" else None)
    dycore.launch_wrf_smag2d_hd(st, cfg, f, tr["xk"], tend, stagger=STAG[g.row],
                                time_t=False, full_theta=full, deformation=defor)
    return tend
for row in {ROWS!r}:
    for i, over in enumerate({CONFIGS!r}):
        g = C.h2_geometry(row=row, rdx=float(f32(1 / dx)), rdy=float(f32(1 / dy)), **over)
        X = D.h2_active(g)
        tr = dev(C.h2_trajectory(g, 41 + i, f32, hybrid=True))
        st, _ = run_tl(g, tr, {{n: cp.zeros_like(tr[n]) for n in X}}, True)
        out[f"{{row}}/{{i}}"] = diff(st["tend"], primal(g, tr))
        if row in ("t", "s"):
            out[f"{{row}}/{{i}}/multi"] = diff(st["tend"], primal(g, tr, multi=True))
        out[f"{{row}}/{{i}}/moves"] = int(diff(st["tend"], tr["tend"]) == 0)
''', tmp_path, extra_env=env)
    assert len(out) == len(CONFIGS) * (2 * len(ROWS) + 2)
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


def test_geometry_rejects_unported_options():
    """An unknown row, empty or too-short axes, a boundary or moist flag
    other than 0/1 and non-positive or non-finite spacings are refused,
    not run."""
    from woof.grad.wrfplus import cases as C
    for over in (dict(row="q"), dict(nx=0), dict(nz=2), dict(boundary_y=2), dict(moist=2),
                 dict(rdx=0.0), dict(rdy=float("inf"))):
        with pytest.raises(ValueError):
            C.h2_geometry(**over).validate()
