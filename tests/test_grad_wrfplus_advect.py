"""Flux-form advection TL/AD (P3a item 6, woof.grad.wrfplus.advect):
woof's strict flux_div_scalar, flux_div_u, flux_div_v and flux_div_w
(WRF advect_scalar/u/v/w) with periodic, open and specified lateral
boundaries, map factors, h_*_adv_order 3-6 and v_*_adv_order 2-6.

Breakage this prevents:
- an adjoint that is not the transpose of its TL (dot-product test);
- an adjoint that overwrites instead of adding into its inputs;
- a TL that is not the derivative of the base-state update
  (TL vs finite differences);
- a TL/AD that drifts from WRFPLUS 4.7.1's G_/A_ADVECT_SCALAR, _U, _V
  and _W (tests/data/wrfplus/advect.npz, made by tools/wrfplus_oracle);
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
FIXTURE = ROOT / "tests/data/wrfplus/advect.npz"


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

# Periodic, open and specified axes (both, one, mixed), every vertical
# order, the four horizontal orders on periodic and open axes, the 2-level
# column, short periodic lines, the 7-cell open minimum (with and without
# spec) and a column too short for the vertical order; stretched levels on
# odd cases.
CONFIGS = [dict(), dict(open_x=1, open_y=1), dict(open_x=1, open_y=1, spec=1),
           dict(open_x=1, vorder=5, nz=9), dict(open_y=1, spec=1, vorder=2),
           dict(vorder=4, nz=9, horder=3), dict(vorder=6, nz=10, open_x=1, horder=4),
           dict(horder=6, open_x=1, open_y=1, spec=1, vorder=5, nz=8),
           dict(nz=2, ny=7, nx=7, open_x=1, open_y=1),
           dict(open_x=1, open_y=1, horder=3), dict(open_x=1, spec=1, horder=3, vorder=4, nz=9),
           dict(horder=6, vorder=6, nz=4), dict(nx=2, ny=3), dict(nx=1, ny=1, horder=4),
           dict(open_x=1, open_y=1, spec=1, nx=7, ny=7, horder=3)]
KINDS = ("scalar", "u", "v", "w")


def _run(backend, body, tmp_path, extra_env=None):
    env = dict(os.environ, WOOF_BACKEND=backend, PYTHONPATH=str(ROOT),
               WOOF_HOST_CACHE=str(tmp_path / "hostcache"), WOOF_HOST_THREADS="4")
    env.update(extra_env or {})
    prelude = (
        "import json, numpy as np, woof\nimport cupy as cp\n"
        "from woof.grad.wrfplus import advect as A, cases as C, _checks as K\n"
        "out = {}\n"
        "N = A.ADV_ACTIVE\n"
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n"
        "def run_tl(g, tr, pe, nl):\n"
        "    st = {'tend': tr['tend'].copy()}\n"
        "    sd = {'tend': pe['tend'].copy() if 'tend' in pe else cp.zeros_like(tr['tend'])}\n"
        "    A.adv_tl(st, sd, tr, pe, g, nl=nl)\n"
        "    return st, sd\n"
        "def run_ad(g, tr, at, adj=None):\n"
        "    sa = {'tend': at.copy()}\n"
        "    if adj is None:\n"
        "        adj = {n: cp.zeros_like(tr[n]) for n in N}\n"
        "    A.adv_ad(sa, tr, adj, g)\n"
        "    return sa, adj\n"
        "def seed(g, s, dt):\n"
        "    sh = A.adv_shapes(g)['tend']\n"
        "    return cp.asarray(np.random.default_rng(s).standard_normal(sh).astype(dt))\n")
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
    for kind in {KINDS!r}:
        for i, over in enumerate({CONFIGS!r}):
            g = C.adv_geometry(kind, **over)
            tr = dev(C.adv_trajectory(g, 11 + i, dt, hybrid=bool(i % 2)))
            p = dev(C.adv_perturbation(g, 100 + i, dt))
            dy = seed(g, 200 + i, dt)
            def tl(dx):
                return [run_tl(g, tr, dict(zip(N, dx)), False)[1]["tend"]]
            def ad(dys):
                _, adj = run_ad(g, tr, dys[0])
                return [adj[n] for n in N]
            out[f"{{dtn}}/{{kind}}/{{i}}"] = K.dot_product(
                tl, ad, [p[n] for n in N], [dy], terms=dtn == "float32")
''', tmp_path)
    assert len(out) == (1 if backend == "mlx" else 2) * len(KINDS) * len(CONFIGS)
    for key, rel in out.items():
        limit = 1e-12 if key.startswith("float64") else 1e-5
        assert rel < limit, (key, rel)


@pytest.mark.parametrize("backend", BACKENDS)
def test_adjoint_accumulates(backend, tmp_path):
    """The AD adds into ``adj`` and leaves the tendency adjoint (which
    passes through, the tendency being accumulated) unchanged."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
dt = np.dtype({dtn!r})
for kind in {KINDS!r}:
    for i, over in enumerate({CONFIGS!r}):
        g = C.adv_geometry(kind, **over)
        tr = dev(C.adv_trajectory(g, 51 + i, dt, hybrid=bool(i % 2)))
        q = seed(g, 500 + i, dt)
        s = dev(C.adv_perturbation(g, 600 + i, dt))
        sa, zero = run_ad(g, tr, q)
        _, adj = run_ad(g, tr, q, {{n: s[n].copy() for n in N}})
        out[f"{{kind}}/{{i}}/tend"] = float(cp.abs(sa["tend"] - q).max())
        for n in N:
            want = cp.asnumpy(s[n]).astype(np.float64) + cp.asnumpy(zero[n])
            got = cp.asnumpy(adj[n]).astype(np.float64)
            scale = max(float(np.abs(want).max()), 1e-300)
            out[f"{{kind}}/{{i}}/{{n}}"] = float(np.abs(got - want).max()) / scale
''', tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-13
    bad = {k: v for k, v in out.items() if not v < limit}
    assert not bad, bad


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x)|| / ||a TL dx|| -> 1, the misfit O(a).  The
    upwind switches (sign of the face flux, the open boundary's min/max)
    are a measure-zero set the random states do not sit on."""
    out = _run(backend, f'''
for kind in {KINDS!r}:
    for i, over in enumerate({CONFIGS!r}):
        g = C.adv_geometry(kind, **over)
        tr = dev(C.adv_trajectory(g, 31 + i, np.float64, hybrid=bool(i % 2)))
        p = dev(C.adv_perturbation(g, 300 + i, np.float64))
        def model(xs):
            t = dict(tr); t.update(dict(zip(N, xs)))
            st, _ = run_tl(g, t, {{n: cp.zeros_like(t[n]) for n in N}}, True)
            return [st["tend"]]
        def tl(dx):
            return [run_tl(g, tr, dict(zip(N, dx)), False)[1]["tend"]]
        out[f"{{kind}}/{{i}}"] = K.tl_convergence(model, tl, [tr[n] for n in N],
                                                  [p[n] for n in N], [1e-3, 1e-5, 1e-7])
''', tmp_path)
    assert len(out) == len(KINDS) * len(CONFIGS)
    for key, rows in out.items():
        misfits = [abs(r - 1.0) for _, r in rows]
        assert misfits[-1] < 1e-6, (key, rows)
        assert misfits[0] < 0.1, (key, rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrfplus_oracle_agreement(backend, tmp_path):
    """woof's TL/AD reproduce WRFPLUS 4.7.1 G_/A_ADVECT_SCALAR, _U, _V
    and _W on the fixture (16 cases over the kinds, boundaries and
    orders), on the points where WRF's advect_* and woof's flux_div_*
    compute the same thing (the fixture's mask): WRF's open non-specified
    u/v normal faces carry its cb terms, which woof applies in a separate
    radiative launch; WRF's advect_w applies its open non-cb terms on the
    lid, which woof's strict flux_div_w does not; and a periodic staggered
    end point repeats point 0.  The seed is zero off the mask.  A periodic
    u (v) reads the end-face mass flux through woof's column n and WRF's
    halo column 0, so their adjoints are compared as a sum."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
import json as J
z = np.load({str(FIXTURE)!r})
dt = np.dtype({dtn!r})
for case in sorted({{k.split("/")[0] for k in z.files if "/" in k}}):
    pre = case + "/"
    g = A.AdvGeometry(**J.loads(str(z[pre + "geometry"])))
    pick = lambda kind: {{k.split("/", 2)[2]: cp.asarray(z[k].astype(dt))
                          for k in z.files if k.startswith(pre + kind + "/")}}
    tr, pe = pick("traj"), pick("pert")
    mask = z[pre + "mask"]
    exp = {{k.split("/", 2)[2]: z[k].copy() for k in z.files if k.startswith(pre + "wrfplus/")}}
    st, sd = run_tl(g, tr, pe, True)
    _, adj = run_ad(g, tr, cp.asarray(z[pre + "seed/tend"].astype(dt)))
    got = {{"tl_tend": st["tend"], "tl_d_tend": sd["tend"]}}
    got.update({{"ad_" + n: adj[n] for n in N}})
    for name, e in exp.items():
        w = cp.asnumpy(got[name]).astype(np.float64)
        if name.startswith("tl_"):
            w, e = w[mask], e[mask]
        for x in (w, e):
            if name == "ad_ru" and g.kind == "u" and not g.open_x:
                x[:, :, 0] += x[:, :, -1]; x[:, :, -1] = 0
            if name == "ad_rv" and g.kind == "v" and not g.open_y:
                x[:, 0, :] += x[:, -1, :]; x[:, -1, :] = 0
        scale = max(float(np.abs(e).max()), 1e-300)
        out[pre + name] = float(np.abs(w - e).max()) / scale
''', tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-12
    bad = {k: v for k, v in out.items() if not v < limit}
    assert not bad, sorted(bad.items())
    assert len(out) == 16 * 6


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_float_base_state_is_the_strict_primal(backend, tmp_path):
    """The float build's base state gives woof's strict flux_div words
    (GPUWM_WRF_EXACT, launched by woof.core.advection with the map factor
    on), so the adjoint is bound to the model woof runs: a primal edit
    fails here until the TL/AD is re-derived.  Host and CUDA; Metal is not
    checked here."""
    out = _run(backend, f'''
from types import SimpleNamespace as NS
from woof.core import advection as P
f32 = np.float32
LAUNCH = {{"scalar": P.launch_flux_div_scalar, "u": P.launch_flux_div_u,
          "v": P.launch_flux_div_v, "w": P.launch_flux_div_w}}
for kind in {KINDS!r}:
    for i, over in enumerate({CONFIGS!r}):
        dx, dy = 3000.0, 2500.0
        g = C.adv_geometry(kind, rdx=float(f32(1.0 / dx)), rdy=float(f32(1.0 / dy)), **over)
        tr = dev(C.adv_trajectory(g, 41 + i, f32, hybrid=bool(i % 2)))
        st, _ = run_tl(g, tr, {{n: cp.zeros_like(tr[n]) for n in N}}, True)
        ref = tr["tend"].copy()
        coord = NS(rdnw=tr["rdz"], rdn=tr["rdz"], fnm=tr["fnm"], fnp=tr["fnp"])
        LAUNCH[kind](tr["q"], tr["ru"], tr["rv"], tr["rw"], ref, coord, dx, dy,
                     open_x=bool(g.open_x), open_y=bool(g.open_y), msf=tr["msf"],
                     has_msf=True, spec=bool(g.spec), vorder=g.vorder, horder=g.horder)
        a, b = cp.asnumpy(st["tend"]), cp.asnumpy(ref)
        out[f"{{kind}}/{{i}}"] = int((a.view(np.uint32) != b.view(np.uint32)).sum())
''', tmp_path, extra_env={"GPUWM_WRF_EXACT": "1"})
    assert len(out) == len(KINDS) * len(CONFIGS)
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


def test_geometry_rejects_unported_options():
    """h_*_adv_order 2, vertical orders outside 2-6, open axes under 7
    cells, specified without an open axis, empty axes and a 1-level w
    column are refused, not run."""
    from woof.grad.wrfplus import cases as C
    for over in (dict(horder=2), dict(vorder=7), dict(open_x=1, nx=6),
                 dict(spec=1), dict(kind="theta"), dict(nx=0),
                 dict(kind="w", nz=1)):
        kind = over.pop("kind", "scalar")
        with pytest.raises(ValueError):
            C.adv_geometry(kind, **over).validate()
