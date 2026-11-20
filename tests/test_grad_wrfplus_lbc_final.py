"""Specified-boundary finish TL/AD (P3a item 7b, woof.grad.wrfplus.lbc):
WRF spec_bdy_final on woof's strict finalize_state_field (and
install_mu_boundary for mu), and spec_bdyupdate_ph + zero_grad_bdy(w)
with the frame's calc_p_rho on woof's strict advance_specified_phi_w,
for u, v, theta, phi, w and mu, map factors, hybrid levels and wider
specified zones.

Breakage this prevents:
- an adjoint that is not the transpose of its TL (dot-product test, the
  state inputs included);
- an adjoint that overwrites instead of adding into its inputs;
- a TL that is not the derivative of the base-state update
  (TL vs finite differences);
- a TL/AD that drifts from WRFPLUS 4.7.1's g_/a_spec_bdy_final,
  g_/a_spec_bdyupdate_ph, g_/a_zero_grad_bdy and g_/a_calc_p_rho
  (tests/data/wrfplus/lbc_final.npz, made by tools/wrfplus_oracle);
- a TL/AD base state that drifts from woof's own strict primal.

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
FIXTURE = ROOT / "tests/data/wrfplus/lbc_final.npz"


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

CONFIGS = {
    # every field, map factors (u, v, w), a wider specified zone, the
    # smallest domain
    "lf": [dict(kind=k) for k in ("u", "v", "theta", "phi", "w", "mu")]
    + [dict(kind="u", has_msf=1), dict(kind="v", has_msf=1), dict(kind="w", has_msf=1),
       dict(kind="theta", spec_zone=2), dict(kind="u", spec_zone=3, ny=7, nx=6)],
    # spec_zone 1-3, the smallest domains, one level
    "sp": [dict(), dict(spec_zone=2), dict(nz=1, ny=3, nx=3), dict(spec_zone=3, ny=7, nx=8),
           dict(nz=6, ny=5, nx=12)],
}

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
        "OPS = {\n"
        " 'lf': (L.LF_STATE, L.LF_ACTIVE, C.lf_geometry, C.lf_trajectory, C.lf_perturbation,\n"
        "        L.lbf_tl, L.FinalGeometry),\n"
        " 'sp': (L.SP_STATE, L.SP_ACTIVE, C.sp_geometry, C.sp_trajectory, C.sp_perturbation,\n"
        "        L.sph_tl, L.SpecPhiGeometry)}\n"
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n"
        "def run_tl(op, g, tr, pe, nl, sd=None):\n"
        "    St, tlf = OPS[op][0], OPS[op][5]\n"
        "    st = {n: tr[n].copy() for n in St}\n"
        "    sd = {n: (sd[n].copy() if sd else cp.zeros_like(tr[n])) for n in St}\n"
        "    tlf(st, sd, tr, pe, g, nl=nl)\n"
        "    return st, sd\n"
        "def run_ad(op, g, tr, sa, adj=None):\n"
        "    St, X = OPS[op][0], OPS[op][1]\n"
        "    sa = {n: sa[n].copy() for n in St}\n"
        "    if adj is None:\n"
        "        adj = {n: cp.zeros_like(tr[n]) for n in X}\n"
        "    if op == 'lf':\n"
        "        L.lbf_ad(sa, tr, adj, g)\n"
        "    else:\n"
        "        L.sph_ad(sa, {'ph_pp': tr['ph_pp']}, tr, adj, g)\n"
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
    """<TL dx, dy> == <dx, AD dy> over the state and active inputs: 1e-12
    in binary64; 1e-5 in binary32, relative to the term magnitudes."""
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
            s0 = seeds(op, tr, 150 + i, dt)
            dy = seeds(op, tr, 200 + i, dt)
            n_st = len(St)
            def tl(dx):
                sd = dict(zip(St, dx[:n_st]))
                return [run_tl(op, g, tr, dict(zip(X, dx[n_st:])), False, sd)[1][n] for n in St]
            def ad(dys):
                sa, adj = run_ad(op, g, tr, dict(zip(St, dys)))
                return [sa[n] for n in St] + [adj[n] for n in X]
            out[f"{{dtn}}/{{op}}/{{i}}"] = K.dot_product(
                tl, ad, [s0[n] for n in St] + [p[n] for n in X], [dy[n] for n in St],
                terms=dtn == "float32")
''', tmp_path)
    n = sum(len(c) for c in CONFIGS.values())
    assert len(out) == (1 if backend == "mlx" else 2) * n
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
for op, cfgs in {CONFIGS!r}.items():
    St, X, G, T, P = OPS[op][:5]
    for i, over in enumerate(cfgs):
        g = G(**over)
        tr = dev(T(g, 51 + i, dt, hybrid=bool(i % 2)))
        q = seeds(op, tr, 500 + i, dt)
        s = dev(P(g, 600 + i, dt))
        sa0, zero = run_ad(op, g, tr, q)
        sa1, adj = run_ad(op, g, tr, q, {{n: s[n].copy() for n in X}})
        for n in St:
            out[f"{{op}}/{{i}}/{{n}}"] = float(cp.abs(sa1[n] - sa0[n]).max())
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
    """||NL(x + a dx) - NL(x) - a TL dx|| / ||a TL dx|| -> 0 as O(a), the
    state inputs perturbed too."""
    out = _run(backend, f'''
for op, cfgs in {CONFIGS!r}.items():
    St, X, G, T, P = OPS[op][:5]
    for i, over in enumerate(cfgs):
        g = G(**over)
        tr = dev(T(g, 31 + i, np.float64, hybrid=bool(i % 2)))
        p = dev(P(g, 300 + i, np.float64))
        s0 = seeds(op, tr, 350 + i, np.float64)
        names = St + X
        def model(xs):
            t = dict(tr); t.update(dict(zip(names, xs)))
            st, _ = run_tl(op, g, t, {{n: cp.zeros_like(t[n]) for n in X}}, True)
            return [st[n] for n in St]
        def tl(dx):
            sd = dict(zip(St, dx[:len(St)]))
            return [run_tl(op, g, tr, dict(zip(X, dx[len(St):])), False, sd)[1][n] for n in St]
        out[f"{{op}}/{{i}}"] = K.tl_residual(model, tl, [tr[n] for n in names],
                                             [s0[n] for n in St] + [p[n] for n in X],
                                             [1e-3, 1e-5, 1e-7])
''', tmp_path)
    assert len(out) == sum(len(c) for c in CONFIGS.values())
    for key, rows in out.items():
        misfits = [r for _, r in rows]
        assert misfits[-1] < 1e-6, (key, rows)
        assert misfits[0] < 0.1, (key, rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrfplus_oracle_agreement(backend, tmp_path):
    """woof's TL/AD reproduce WRFPLUS 4.7.1 g_/a_SPEC_BDY_FINAL (u, v with
    map factors, theta, phi, w with map factors, mu, a wider zone) and
    g_spec_bdyupdate_ph + g_zero_grad_bdy(w) + g_calc_p_rho with their
    adjoints (spec_zone 1 and 2).  WRFPLUS's versions are not hybrid and
    take the coupling mass as given (spec_bdy_final: muus/muvs for u/v),
    so the cases are not hybrid and the fixture forms the face masses
    with their tangents and adjoints.  calc_p_rho runs over the whole
    domain in WRFPLUS and on the frame here: p and al are compared on the
    frame columns, with their seeds zero inside."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
import json as J
dt = np.dtype({dtn!r})
z = np.load({str(FIXTURE)!r})
for case in sorted({{k.split("/")[0] for k in z.files if "/" in k}}):
    pre = case + "/"
    op = str(z[pre + "op"])
    St, X, Geo = OPS[op][0], OPS[op][1], OPS[op][6]
    g = Geo(**J.loads(str(z[pre + "geometry"])))
    pick = lambda kind: {{k.split("/", 2)[2]: cp.asarray(z[k].astype(dt))
                          for k in z.files if k.startswith(pre + kind + "/")}}
    tr, pe, sd, s0 = pick("traj"), pick("pert"), pick("seed"), pick("spert")
    st, dd = run_tl(op, g, tr, pe, True, s0)
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
        if pre + "mask" in z.files and name.split("_", 1)[-1] in ("p_pp", "al_pp", "d_p_pp",
                                                                  "d_al_pp"):
            m = np.broadcast_to(z[pre + "mask"], e.shape)
            e, w = e[m], w[m]
        scale = max(float(np.abs(e).max()), 1e-300)
        out[pre + name] = float(np.abs(w - e).max()) / scale
''', tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-12
    # php's adjoint is the ph'' seed times mu_old/muts - 1 = -dtau rmu/muts
    # (about 2e-4 here), formed by that subtraction as in WRF: binary32
    # keeps about eps32 muts/(dtau rmu) of it (2.2e-4 on CUDA float32).
    loose = {"ad_php": 2e-3} if backend == "mlx" else {}
    bad = {k: v for k, v in out.items() if not v < loose.get(k.split("/")[-1], limit)}
    assert not bad, sorted(bad.items())
    assert len(out) == 9 * 12 + 2 * 21


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_float_base_state_is_the_strict_primal(backend, tmp_path):
    """The float build's base state gives woof's strict
    finalize_state_field (GPUWM_WRF_EXACT, muts_wrf), install_mu_boundary
    and advance_specified_phi_w words, so the adjoint is bound to the
    model woof runs.  Host and CUDA; Metal is not checked here."""
    out = _run(backend, f'''
from woof.core.kernels import get_kernel
f32, i32 = np.float32, np.int32
def diff(a, b):
    return int((cp.asnumpy(a).view(np.uint32) != cp.asnumpy(b).view(np.uint32)).sum())
for i, over in enumerate({CONFIGS["lf"]!r}):
    g = C.lf_geometry(**over)
    tr = dev(C.lf_trajectory(g, 41 + i, f32, hybrid=bool(i % 2)))
    st, _ = run_tl("lf", g, tr, {{n: cp.zeros_like(tr[n]) for n in L.LF_ACTIVE}}, True)
    ref = tr["target"].copy()
    sides = [tr[s + t] for s in ("west", "east", "south", "north") for t in ("", "_t")]
    nz, ny, nx = g.field
    if g.kind == "mu":
        from woof.ingest.lateral_bc import _perimeter_count
        mu2 = ref[0]
        fc = _perimeter_count(ny, nx, g.spec_zone)
        old = cp.zeros(fc, dtype=f32)
        get_kernel("lbc_state", "install_mu_boundary")(((fc + 255) // 256,), (256,), (
            mu2, old, *sides, f32(g.dtbc), i32(g.width), i32(g.spec_zone), i32(ny), i32(nx),
            i32(fc)))
    else:
        n = nz * ny * nx
        thb = cp.zeros(g.nz, dtype=f32)
        kind = L.LB_KINDS[g.kind]
        get_kernel("lbc_state", "finalize_state_field")(((n + 255) // 256,), (256,), (
            ref, tr["muts"], tr["muts"], tr["muts"], thb, ref, tr["c1h"], tr["c2h"],
            tr["c1f"], tr["c2f"], tr["msft"], tr["msfu"], tr["msfv"], *sides, f32(g.dtbc),
            i32(g.width), i32(g.spec_zone), i32(g.has_msf), i32(0), i32(kind), i32(nz),
            i32(ny), i32(nx), i32(g.ny), i32(g.nx), tr["muts"]))
    out[f"lf/{{i}}"] = diff(st["target"], ref) + int(not bool((ref != tr["target"]).any()))
for i, over in enumerate({CONFIGS["sp"]!r}):
    g = C.sp_geometry(**over)
    tr = dev(C.sp_trajectory(g, 61 + i, f32, hybrid=bool(i % 2)))
    st, _ = run_tl("sp", g, tr, {{n: cp.zeros_like(tr[n]) for n in L.SP_ACTIVE}}, True)
    r = {{n: tr[n].copy() for n in L.SP_STATE}}
    nz, ny, nx = g.nz, g.ny, g.nx
    z2 = cp.zeros((ny, nx), dtype=f32)
    get_kernel("acoustic", "advance_specified_phi_w")((((ny * nx) + 255) // 256,), (256,), (
        r["ph_pp"], r["w_pp"], r["p_pp"], r["al_pp"], tr["rph_t"], tr["th_pp"], z2,
        tr["mu_pp"], z2, tr["rmu_t"], tr["php"], tr["thp"], tr["thp"], tr["alt"], tr["c2a"],
        tr["rdnw"], tr["c1h"], tr["c2h"], tr["c1f"], tr["c2f"], f32(g.dtau),
        i32(g.spec_zone), i32(1), i32(nz), i32(ny), i32(nx), tr["muts"]))
    out[f"sp/{{i}}"] = sum(diff(st[n], r[n]) for n in L.SP_STATE)
''', tmp_path, extra_env=STRICT_ENV)
    assert len(out) == sum(len(c) for c in CONFIGS.values())
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


def test_geometry_rejects_unported_options():
    """spec_zone 0 or wider than the table, a domain without an interior
    source column, unknown kinds and empty axes are refused, not run."""
    from woof.grad.wrfplus import cases as C
    for make, over in ((C.lf_geometry, dict(spec_zone=0)), (C.lf_geometry, dict(width=1, spec_zone=2)),
                       (C.lf_geometry, dict(kind="qv")), (C.lf_geometry, dict(ny=2)),
                       (C.lf_geometry, dict(nz=0)), (C.sp_geometry, dict(spec_zone=0)),
                       (C.sp_geometry, dict(ny=2)), (C.sp_geometry, dict(nx=4, spec_zone=2)),
                       (C.sp_geometry, dict(nz=0))):
        with pytest.raises(ValueError):
            make(**over).validate()
