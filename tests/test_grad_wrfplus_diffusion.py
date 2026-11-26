"""diff_opt 1 / km_opt 4 coefficient TL/AD (P3a item 5,
woof.grad.wrfplus.diffusion): WRF compute_diff_metrics +
cal_deform_and_div (defor11, defor22, defor12) and smag2d_km at diff_opt 1
(xkmh, xkhh) on woof's strict wrf_smag_deform and wrf_diff_opt1_km4 +
wrf_smag_km_bc, with periodic and bounded axes, single-cell lines, map
factors and a 3-D base geopotential.

Breakage this prevents:
- an adjoint that is not the transpose of its TL (dot-product test);
- an adjoint that overwrites instead of adding into its inputs, or that
  does not consume the adjoints of the outputs the primal overwrites;
- a TL that is not the derivative of the base-state update
  (TL vs finite differences), including the 10 * mlen cap;
- a TL/AD that drifts from WRFPLUS 4.7.1's g_/a_compute_diff_metrics +
  g_/a_cal_deform_and_div and G_/A_SMAG2D_KM
  (tests/data/wrfplus/diffusion.npz, made by tools/wrfplus_oracle);
- a TL/AD base state that drifts from woof's own strict primal (the
  float build must give the primal's words).

Each probe runs in a child process, as in test_grad_wrfplus_bigstep.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/wrfplus/diffusion.npz"


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

# Periodic, bounded (both axes, one), the 3-level minimum, single-cell and
# two-cell lines (periodic self-reads, a bounded axis's remapped d12 row),
# and WRF's cold seam slope at the top w level of a periodic axis
# (seam_cold, RunConfig.diff_metrics_seam_slope off) on both axes, one and
# a short line.
CONFIGS = {
    "df": [dict(), dict(boundary_x=1, boundary_y=1), dict(boundary_x=1), dict(boundary_y=1),
           dict(nz=3, ny=1, nx=2), dict(nz=3, ny=2, nx=1, boundary_y=1),
           dict(nz=4, ny=3, nx=3, boundary_x=1, boundary_y=1), dict(nz=5, ny=4, nx=2),
           dict(seam_cold=1), dict(boundary_y=1, seam_cold=1), dict(boundary_x=1, seam_cold=1),
           dict(nz=3, ny=2, nx=2, seam_cold=1)],
    "km": [dict(), dict(boundary_x=1, boundary_y=1), dict(boundary_x=1), dict(boundary_y=1),
           dict(nz=1, ny=1, nx=1), dict(nz=2, ny=2, nx=1), dict(ny=3, nx=3, boundary_y=1)],
}

# Strict mode with each of its controls pinned on, so an inherited =0
# cannot leave the reference primal off WRF's order.
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
        "OPS = {\n"
        " 'df': (D.DF_STATE, D.DF_ACTIVE, C.df_geometry, C.df_trajectory,\n"
        "        C.df_perturbation, D.df_tl, D.df_ad, D.DFGeometry),\n"
        " 'km': (D.KM_STATE, D.KM_ACTIVE, C.km_geometry, C.km_trajectory,\n"
        "        C.km_perturbation, D.km_tl, D.km_ad, D.KMGeometry)}\n"
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
    """The AD adds into ``adj`` and consumes (zeroes) the adjoints of the
    outputs, which the primal overwrites."""
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
            out[f"{{op}}/{{i}}/st_{{n}}"] = float(cp.abs(sa[n]).max())
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
    """||NL(x + a dx) - NL(x) - a TL dx|| / ||a TL dx|| -> 0 as O(a)."""
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
        out[f"{{op}}/{{i}}"] = K.tl_residual(model, tl, [tr[n] for n in X],
                                             [p[n] for n in X], [1e-3, 1e-5, 1e-7])
''', tmp_path)
    assert len(out) == sum(len(c) for c in CONFIGS.values())
    for key, rows in out.items():
        misfits = [r for _, r in rows]
        assert misfits[-1] < 1e-6, (key, rows)
        assert misfits[0] < 0.1, (key, rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrfplus_oracle_agreement(backend, tmp_path):
    """woof's TL/AD reproduce WRFPLUS 4.7.1 g_compute_diff_metrics +
    g_cal_deform_and_div then a_cal_deform_and_div + a_compute_diff_metrics
    (defor11, defor22, defor12 of u, v and ph'), and G_/A_SMAG2D_KM at
    diff_opt 1 (xkmh, xkhh of the deformation; WRFPLUS's phy_bc leaves the
    outer row of a specified axis at its cold zero), on periodic and
    specified axes with map factors and a 3-D base geopotential.  A
    periodic axis's end face repeats face 0: the u and v adjoints of faces
    0 and nx (ny) are compared as a sum (woof reads face 0 for both, WRF
    reads its halo).  WRF's compute_diff_metrics leaves a periodic seam's
    top-level slope at its cold zero: the fixture's geometry carries
    seam_cold = 1 (RunConfig.diff_metrics_seam_slope off), so every word is
    compared.  a_cal_deform_and_div overruns its y tapes
    on a periodic y axis, so the fully periodic case is TL only
    (tools/wrfplus_oracle/fixture.py DIFF_CASES)."""
    if not FIXTURE.exists():
        pytest.skip("tests/data/wrfplus/diffusion.npz not built")
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
import json as J
dt = np.dtype({dtn!r})
z = np.load({str(FIXTURE)!r})
for case in sorted({{k.split("/")[0] for k in z.files if "/" in k}}):
    pre = case + "/"
    op = str(z[pre + "op"])
    St, X, Geo = OPS[op][0], OPS[op][1], OPS[op][7]
    g = Geo(**J.loads(str(z[pre + "geometry"])))
    pick = lambda kind: {{k.split("/", 2)[2]: cp.asarray(z[k].astype(dt))
                          for k in z.files if k.startswith(pre + kind + "/")}}
    tr, pe, sd = pick("traj"), pick("pert"), pick("seed")
    st, dd = run_tl(op, g, tr, pe, True)
    sa, adj = run_ad(op, g, tr, sd)
    got = {{}}
    for n in St:
        got["tl_" + n], got["tl_d_" + n], got["adst_" + n] = st[n], dd[n], sa[n]
    for n in X:
        got["ad_" + n] = adj[n]
    for k in z.files:
        if not k.startswith(pre + "wrfplus/"):
            continue
        name = k.split("/", 2)[2]
        e, w = z[k].copy(), cp.asnumpy(got[name]).astype(np.float64)
        for x in (e, w):
            if name == "ad_u" and not g.boundary_x:
                x[..., 0] += x[..., -1]; x[..., -1] = 0
            if name == "ad_v" and not g.boundary_y:
                x[..., 0, :] += x[..., -1, :]; x[..., -1, :] = 0
        scale = max(float(np.abs(e).max()), 1e-300)
        out[pre + name] = float(np.abs(w - e).max()) / scale
''', tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-12
    bad = {k: v for k, v in out.items() if not v < limit}
    assert not bad, sorted(bad.items())
    # 6 TL words per deformation case plus 6 AD words for the 3 with an
    # adjoint; 4 TL and 5 AD words per coefficient case
    assert len(out) == 4 * 6 + 3 * 6 + 3 * 9


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_float_base_state_is_the_strict_primal(backend, tmp_path):
    """The float build's base state gives woof's strict wrf_smag_deform
    (and the stored-hat wrf_smag_deform_hat Metal runs) and
    wrf_diff_opt1_km4 + wrf_smag_km_bc words (GPUWM_WRF_EXACT, 3-D base),
    so the adjoint is bound to the model woof runs: a primal edit fails
    here until the TL/AD is re-derived.  Each check must also move words
    (the coordinate slopes change the deformation; the cap acts), so it is
    not empty.  Host and CUDA; Metal is not checked here."""
    out = _run(backend, f'''
from woof.core.kernels import get_kernel, get_kernel_int_defines
f32, i32 = np.float32, np.int32
from woof.core import dycore
out["prandtl"] = int(float(dycore._PRANDTL) != D.PRANDTL)
def diff(a, b):
    return int((cp.asnumpy(a).view(np.uint32) != cp.asnumpy(b).view(np.uint32)).sum())
dx, dy = 3000.0, 2500.0
for i, over in enumerate({CONFIGS["df"]!r}):
    g = C.df_geometry(rdx=float(f32(1 / dx)), rdy=float(f32(1 / dy)), **over)
    tr = dev(C.df_trajectory(g, 41 + i, f32, hybrid=bool(i % 2)))
    st, _ = run_tl("df", g, tr, {{n: cp.zeros_like(tr[n]) for n in D.DF_ACTIVE}}, True)
    nz, ny, nx = g.nz, g.ny, g.nx
    common = (tr["u"], tr["v"], tr["php"], tr["php"], tr["phb"], tr["phb"], tr["phb"],
              tr["msft"], tr["msfu"], tr["msfv"], tr["fnm"], tr["fnp"], tr["dn"], tr["dnw"],
              f32(g.rdx), f32(g.rdy), f32(dx), f32(dy), f32(g.cf1), f32(g.cf2), f32(g.cf3),
              i32(0))
    # the primal's boundary flags (dycore.smag_boundary_flags): 2 is a
    # periodic axis with WRF's cold seam slope
    dims = (i32(nz), i32(ny), i32(nx), i32(1)) + tuple(g.args(f32)[5:7])
    grid = ((nx + 63) // 64, ny, nz)
    r = [cp.zeros((nz, ny, nx), f32) for _ in range(3)]
    get_kernel("smag2d", "wrf_smag_deform")(grid, (64, 1, 1), common + tuple(r) + dims)
    out[f"df/{{i}}"] = sum(diff(st[n], x) for n, x in zip(D.DF_STATE, r))
    if g.seam_cold:
        # the cold seam must move the words, or it is unchecked
        r0 = [cp.zeros((nz, ny, nx), f32) for _ in range(3)]
        flags0 = (i32(g.boundary_x), i32(g.boundary_y))
        get_kernel("smag2d", "wrf_smag_deform")(grid, (64, 1, 1),
                                                common + tuple(r0) + dims[:4] + flags0)
        out[f"df/{{i}}/seam"] = int(sum(diff(a, b) for a, b in zip(r, r0)) == 0)
    defs = (("GPUWM_SMAG_STORED_HAT", 1),)
    uh = cp.zeros((nz, ny, nx + 1), f32)
    vh = cp.zeros((nz, ny + 1, nx), f32)
    get_kernel_int_defines("smag2d", "wrf_smag_uvhat", defs)(
        ((nx + 64) // 64, ny + 1, nz), (64, 1, 1),
        (tr["u"], tr["v"], tr["msfu"], tr["msfv"], uh, vh, i32(nz), i32(ny), i32(nx)))
    h = [cp.zeros((nz, ny, nx), f32) for _ in range(3)]
    get_kernel_int_defines("smag2d", "wrf_smag_deform_hat", defs)(
        grid, (64, 1, 1), common + tuple(h) + dims + (uh, vh))
    out[f"df/{{i}}/hat"] = sum(diff(st[n], x) for n, x in zip(D.DF_STATE, h))
    # the coordinate slopes must move the words, or the slope terms are
    # unchecked
    t0 = dict(tr); t0["phb"] = cp.asarray(np.broadcast_to(
        cp.asnumpy(tr["phb"]).mean(axis=(1, 2), keepdims=True), tr["phb"].shape).copy())
    t0["php"] = cp.zeros_like(tr["php"])
    s0, _ = run_tl("df", g, t0, {{n: cp.zeros_like(tr[n]) for n in D.DF_ACTIVE}}, True)
    out[f"df/{{i}}/moves"] = int(sum(diff(st[n], s0[n]) for n in D.DF_STATE) == 0)
for i, over in enumerate({CONFIGS["km"]!r}):
    g = C.km_geometry(**over)
    tr = dev(C.km_trajectory(g, 61 + i, f32))
    st, _ = run_tl("km", g, tr, {{n: cp.zeros_like(tr[n]) for n in D.KM_ACTIVE}}, True)
    nz, ny, nx = g.nz, g.ny, g.nx
    km, kh = cp.full((nz, ny, nx), 7, f32), cp.full((nz, ny, nx), 7, f32)
    grid = ((nx + 63) // 64, ny, nz)
    get_kernel("diff_opt1", "wrf_diff_opt1_km4")(grid, (64, 1, 1), (
        tr["defor11"], tr["defor22"], tr["defor12"], tr["msft"], f32(g.dx), f32(g.dy),
        f32(g.c_s), f32(g.prandtl), km, kh, i32(nz), i32(ny), i32(nx), i32(g.boundary_x),
        i32(g.boundary_y)))
    if g.boundary_x or g.boundary_y:
        get_kernel("smag2d", "wrf_smag_km_bc")(grid, (64, 1, 1), (
            km, kh, i32(nz), i32(ny), i32(nx), i32(g.boundary_x), i32(g.boundary_y)))
    out[f"km/{{i}}"] = diff(st["xkmh"], km) + diff(st["xkhh"], kh)
    # the cap must act somewhere on the larger cases, or it is unchecked
    if nz * ny * nx > 4:
        length = np.sqrt(g.dx / cp.asnumpy(tr["msft"]) * g.dy / cp.asnumpy(tr["msft"]))
        capped = np.isclose(cp.asnumpy(km), np.float32(10 * length), rtol=1e-6)
        out[f"km/{{i}}/cap"] = int(not capped.any())
''', tmp_path, extra_env=STRICT_ENV)
    caps = sum(1 for c in CONFIGS["km"] if c.get("nz", 7) * c.get("ny", 8) * c.get("nx", 9) > 4)
    seams = sum(1 for c in CONFIGS["df"] if c.get("seam_cold"))
    assert len(out) == 3 * len(CONFIGS["df"]) + seams + len(CONFIGS["km"]) + caps + 1
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


def test_geometry_rejects_unported_options():
    """The deformation under 3 levels (the bottom extrapolation reads
    levels 0-2), empty axes, a boundary flag other than 0/1 and
    non-positive or non-finite spacings are refused, not run."""
    from woof.grad.wrfplus import cases as C
    for make, over in ((C.df_geometry, dict(nz=2)), (C.df_geometry, dict(nx=0)),
                       (C.df_geometry, dict(boundary_x=2)), (C.df_geometry, dict(rdx=0.0)),
                       (C.df_geometry, dict(rdy=float("nan"))), (C.km_geometry, dict(ny=0)),
                       (C.km_geometry, dict(dx=0.0)), (C.km_geometry, dict(prandtl=-1.0))):
        with pytest.raises(ValueError):
            make(**over).validate()
