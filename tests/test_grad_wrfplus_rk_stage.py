"""The RK-stage glue's tangent-linear and adjoint (P3a item 4,
woof.grad.wrfplus.stage): calc_p_rho_phi (woof's strict calc_p_alpha /
calc_p_alpha_carried_muts), rk_step_prep (stage_fluxes: couple_momentum
on calc_mu_uv's face masses and calc_ww_cp), rk_addtend_dry
(add_fixed_dry_tendencies + add_held_heating) and set_w_surface
(set_surface_w).

Breakage this prevents:
- an adjoint that is not the transpose of its TL (dot-product test);
- an adjoint that overwrites instead of adding into its inputs;
- a TL that is not the derivative of the base-state update
  (TL vs finite differences);
- a TL/AD that drifts from WRFPLUS 4.7.1's G_/A_CALC_P_RHO_PHI,
  G_/A_CALC_MU_UV + G_/A_COUPLE_MOMENTUM + g_/a_calc_ww_cp,
  G_/A_RK_ADDTEND_DRY and g_/a_set_w_surface
  (tests/data/wrfplus/rk_stage.npz, made by tools/wrfplus_oracle);
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
FIXTURE = ROOT / "tests/data/wrfplus/rk_stage.npz"


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

# Clamped and periodic faces, map factors, hybrid levels (odd cases),
# both hypsometric options, carried or formed muts, 1-D or 3-D base,
# single cells and one level; nz 19 and 16 cross the adjoint's level
# chunks (8), 16 with the top chunk ending at nz.
CONFIGS = {
    "prp": [dict(), dict(hypso=2), dict(carried=0), dict(hypso=2, carried=0, base3d=0),
            dict(base3d=0), dict(nz=19, ny=3, nx=2, hypso=2),
            dict(nz=1, ny=1, nx=1, carried=0)],
    "rsp": [dict(), dict(boundary_x=0, boundary_y=0, has_msf=1),
            dict(nz=19, ny=3, nx=1, boundary_x=0), dict(nz=1, ny=1, nx=1),
            dict(nz=16, ny=2, nx=3, boundary_y=0, has_msf=1),
            dict(ny=1, nx=2, boundary_x=0, boundary_y=0)],
    "at": [dict(), dict(has_msf=1), dict(heating=0), dict(nz=1, ny=1, nx=1, has_msf=1),
           dict(nz=19, ny=2, nx=3)],
    "sw": [dict(), dict(mapped=1, boundary_x=0), dict(boundary_y=0, ny=1, nx=2),
           dict(nz=3, ny=2, nx=1, boundary_y=0), dict(mapped=1, boundary_x=0, boundary_y=0, ny=2, nx=3)],
    # the moist diagnosis (DA item 4): qv through WRF's qvf
    "prpm": [dict(), dict(hypso=2), dict(carried=0, base3d=0), dict(nz=19, ny=3, nx=2, hypso=2),
             dict(nz=1, ny=1, nx=1, carried=0)],
}


def _run(backend, body, tmp_path, extra_env=None):
    env = dict(os.environ, WOOF_BACKEND=backend, PYTHONPATH=str(ROOT),
               WOOF_HOST_CACHE=str(tmp_path / "hostcache"), WOOF_HOST_THREADS="4")
    env.update(extra_env or {})
    prelude = (
        "import json, numpy as np, woof\nimport cupy as cp\n"
        "from woof.grad.wrfplus import stage as S, cases as C, _checks as K\n"
        "out = {}\n"
        "OPS = {\n"
        " 'prp': (S.PRP_STATE, S.PRP_ACTIVE, C.prp_geometry, C.prp_trajectory,\n"
        "         C.prp_perturbation, S.calc_p_rho_phi_tl, S.calc_p_rho_phi_ad, False),\n"
        " 'prpm': (S.PRP_STATE, S.PRPM_ACTIVE, C.prpm_geometry, C.prp_trajectory,\n"
        "          C.prp_perturbation, S.calc_p_rho_phi_tl, S.calc_p_rho_phi_ad, False),\n"
        " 'rsp': (S.RSP_STATE, S.RSP_ACTIVE, C.rsp_geometry, C.rsp_trajectory,\n"
        "         C.rsp_perturbation, S.rk_step_prep_tl, S.rk_step_prep_ad, False),\n"
        " 'at': (S.AT_STATE, S.AT_ACTIVE, C.at_geometry, C.at_trajectory,\n"
        "        C.at_perturbation, S.rk_addtend_dry_tl, S.rk_addtend_dry_ad, True),\n"
        " 'sw': (('w',), S.SW_ACTIVE, C.sw_geometry, C.sw_trajectory,\n"
        "        C.sw_perturbation, S.set_w_surface_tl, S.set_w_surface_ad, False)}\n"
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n"
        "def traj(op, g, seed, dt, hybrid):\n"
        "    T = OPS[op][3]\n"
        "    return dev(T(g, seed, dt) if op == 'sw' else T(g, seed, dt, hybrid=hybrid))\n"
        "def inputs(op):\n"
        "    St, X = OPS[op][:2]\n"
        "    return St + X if OPS[op][7] else X\n"
        "def run_tl(op, g, tr, pe, nl):\n"
        "    St, tlf = OPS[op][0], OPS[op][5]\n"
        "    st = {n: tr[n].copy() for n in St}\n"
        "    sd = {n: (pe[n].copy() if n in pe else cp.zeros_like(tr[n])) for n in St}\n"
        "    tlf(st, sd, tr, pe, g, nl=nl)\n"
        "    return st, sd\n"
        "def run_ad(op, g, tr, sa, adj=None):\n"
        "    St, X, adf = OPS[op][0], OPS[op][1], OPS[op][6]\n"
        "    sa = {n: sa[n].copy() for n in St}\n"
        "    if adj is None:\n"
        "        adj = {n: cp.zeros_like(tr[n]) for n in X}\n"
        "    adf(sa, tr, adj, g)\n"
        "    return sa, adj\n")
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
        St, X, G = OPS[op][:3]
        P = OPS[op][4]
        N = inputs(op)
        for i, over in enumerate(cfgs):
            g = G(**over)
            tr = traj(op, g, 11 + i, dt, bool(i % 2))
            p = dev(P(g, 100 + i, dt))
            q = dev(P(g, 200 + i, dt))
            def tl(dx):
                pe = {{n: a.copy() for n, a in zip(N, dx)}}
                return [run_tl(op, g, tr, pe, False)[1][n] for n in St]
            def ad(dy):
                sa, adj = run_ad(op, g, tr, dict(zip(St, dy)))
                return [sa[n] for n in N if n in St] + [adj[n] for n in X]
            out[f"{{dtn}}/{{op}}/{{i}}"] = K.dot_product(
                tl, ad, [p[n] for n in N], [q[n] for n in St],
                terms=dtn == "float32")
''', tmp_path)
    assert len(out) == (1 if backend == "mlx" else 2) * sum(map(len, CONFIGS.values()))
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
for op, cfgs in {CONFIGS!r}.items():
    St, X, G = OPS[op][:3]
    P = OPS[op][4]
    for i, over in enumerate(cfgs):
        g = G(**over)
        tr = traj(op, g, 51 + i, dt, bool(i % 2))
        q = dev(P(g, 500 + i, dt))
        s = dev(P(g, 600 + i, dt))
        sa, zero = run_ad(op, g, tr, q)
        _, adj = run_ad(op, g, tr, q, {{n: s[n].copy() for n in X}})
        # the outputs the forward overwrites come back consumed (zero)
        consumed = {{n: sa[n] for n in St}} if op in ("prp", "prpm", "rsp") else (
            {{"w": sa["w"][0]}} if op == "sw" else {{}})
        for n, v in consumed.items():
            out[f"{{op}}/{{i}}/consumed_{{n}}"] = float(cp.abs(v).max())
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
    St, X, G = OPS[op][:3]
    P = OPS[op][4]
    N = inputs(op)
    for i, over in enumerate(cfgs):
        g = G(**over)
        tr = traj(op, g, 31 + i, np.float64, bool(i % 2))
        p = dev(P(g, 300 + i, np.float64))
        def model(xs):
            t = dict(tr); t.update(dict(zip(N, xs)))
            st, _ = run_tl(op, g, t, {{n: cp.zeros_like(t[n]) for n in X}}, True)
            return [st[n] for n in St]
        def tl(dx):
            pe = {{n: a.copy() for n, a in zip(N, dx)}}
            return [run_tl(op, g, tr, pe, False)[1][n] for n in St]
        out[f"{{op}}/{{i}}"] = K.tl_convergence(model, tl, [tr[n] for n in N],
                                                [p[n] for n in N], [1e-2, 1e-4, 1e-6])
''', tmp_path)
    for key, rows in out.items():
        misfits = [abs(r - 1.0) for _, r in rows]
        assert misfits[-1] < 1e-5, (key, rows)
        assert misfits[0] < 0.1, (key, rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrfplus_oracle_agreement(backend, tmp_path):
    """woof's TL/AD reproduce WRFPLUS 4.7.1 on the fixture:
    calc_p_rho_phi against G_/A_CALC_P_RHO_PHI (WRF's p and al adjoints
    stand for woof's p + p_pp and al + alt; a formed muts is mub + mu),
    rk_step_prep against G_/A_CALC_MU_UV + G_/A_COUPLE_MOMENTUM +
    g_/a_calc_ww_cp, rk_addtend_dry against G_/A_RK_ADDTEND_DRY at
    rk_step 2 (WRF's held tendf is woof's times the map factor), and
    set_w_surface against g_/a_set_w_surface.  The halo columns carry
    woof's clamped or wrapped neighbours.  WRFPLUS has no hybrid
    coordinate, so the fixture has c1 = 1, c2 = 0; the c1/c2 terms are
    covered by the dot-product, finite-difference and strict-primal tests
    only."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
import json as J
z = np.load({str(FIXTURE)!r})
dt = np.dtype({dtn!r})
GEOM = {{"prp": S.PRPGeometry, "rsp": S.RSPGeometry, "at": S.ATGeometry,
         "sw": S.SWGeometry}}
cases = sorted({{k.split("/")[0] for k in z.files if "/" in k}})
for case in cases:
    # no WRFPLUS fixture for the moist diagnosis
    for op, (St, X) in ((op, OPS[op][:2]) for op in OPS if op != "prpm"):
        pre = case + "/" + op + "/"
        g = GEOM[op](**J.loads(str(z[pre + "geometry"])))
        pick = lambda kind: {{k.split("/", 3)[3]: cp.asarray(z[k].astype(dt))
                              for k in z.files if k.startswith(pre + kind + "/")}}
        tr, pe, seed = pick("traj"), pick("pert"), pick("seed")
        exp = {{k.split("/", 3)[3]: z[k] for k in z.files if k.startswith(pre + "wrfplus/")}}
        st, sd = run_tl(op, g, tr, pe, True)
        sa, adj = run_ad(op, g, tr, seed)
        got = {{"tl_" + n: st[n] for n in St}}
        got.update({{"tl_d_" + n: sd[n] for n in St}})
        got.update({{"ad_" + n: sa[n] for n in St}})
        got.update({{"ad_" + n: adj[n] for n in X}})
        for name, e in exp.items():
            w = cp.asnumpy(got[name]).astype(np.float64)
            scale = max(float(np.abs(e).max()), 1e-300)
            out[pre + name] = float(np.abs(w - e).max()) / scale
''', tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-12

    def limit_for(key):
        # binary32 base al (and p' from it) under hypsometric_opt 2 is
        # al = dph/phm/log(pfd/pfu) - alb: the log of a ratio near 1 and
        # the cancellation against alb (al ~ 0.03 alb) cost about 30x
        # epsilon; the tangents keep 1e-5.
        if backend == "mlx" and key.endswith(("prp/tl_al", "prp/tl_p_pp")):
            return 1e-4
        return limit
    bad = {k: v for k, v in out.items() if not v < limit_for(k)}
    assert not bad, sorted(bad.items())
    # per case: rk_step_prep 6 TL + 6 AD, calc_p_rho_phi 6 TL + 8 AD,
    # rk_addtend_dry 12 TL + 14 AD, set_w_surface 2 TL + 3 AD.
    assert len(out) == 4 * (12 + 14 + 26 + 5)


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_float_base_state_is_the_strict_primal(backend, tmp_path):
    """The float build's base state gives woof's strict primal words
    (GPUWM_WRF_EXACT: calc_p_alpha[_carried_muts], dry and moist (qv
    through qvf; the moist cases must move p off the dry words), stage_fluxes' couple
    momentum and calc_ww_cp, the dry tendency add + add_held_heating, and
    set_surface_w), so the adjoint is bound to the model woof runs: a
    primal edit fails here until the TL/AD is re-derived.  Host and CUDA;
    Metal is not checked here."""
    out = _run(backend, f'''
from types import SimpleNamespace as NS
from woof.core.kernels import get_kernel
from woof.core import dycore as D
f32, i32 = np.float32, np.int32
CF = {CONFIGS!r}
def same(a, b):
    return bool((cp.asnumpy(a).view(np.uint32) == cp.asnumpy(b).view(np.uint32)).all())
def zeros(op, g, tr):
    return {{n: cp.zeros_like(tr[n]) for n in OPS[op][1]}}
for i, over in enumerate(CF["prp"]):
    g = C.prp_geometry(**over)
    tr = traj("prp", g, 41 + i, f32, bool(i % 2))
    st, _ = run_tl("prp", g, tr, zeros("prp", g, tr), True)
    T = {{n: cp.zeros_like(tr[n]) for n in S.PRP_STATE}}
    nz, ny, nx = g.nz, g.ny, g.nx
    tail = (tr["thp"], f32(g.p_top), i32(g.hypso), i32(0), i32(g.base3d),
            i32(nz), i32(ny), i32(nx), i32(0), i32(0), i32(ny), i32(nx),
            T["p"], T["al"], T["alt"], T["p_pp"])
    colb = ((ny * nx + 255) // 256,)
    if g.carried:
        get_kernel("diagnostics", "calc_p_alpha_carried_muts")(colb, (256,), (
            tr["thp"], tr["php"], tr["mup"], tr["alb"], tr["phb"], tr["alb"],
            tr["pb"], tr["muts"], tr["rdnw"], tr["c1h"], tr["c2h"], tr["c3h"],
            tr["c4h"], tr["c3f"], tr["c4f"]) + tail)
    else:
        dummy = cp.zeros(nz + 1, dtype=f32)
        get_kernel("diagnostics", "calc_p_alpha", (nz, ny, nx))(colb, (256,), (
            tr["thp"], tr["php"], tr["mup"], tr["alb"], tr["phb"], tr["alb"],
            tr["alb"], tr["pb"], tr["rdnw"], tr["c1h"], tr["c2h"], tr["c3h"],
            tr["c4h"], tr["c3f"], tr["c4f"], dummy, dummy, tr["mub"]) + tail)
    out[f"prp/{{i}}"] = {{n: same(st[n], T[n]) for n in S.PRP_STATE}}
for i, over in enumerate(CF["prpm"]):
    g = C.prpm_geometry(**over)
    tr = traj("prpm", g, 43 + i, f32, bool(i % 2))
    st, _ = run_tl("prpm", g, tr, zeros("prpm", g, tr), True)
    T = {{n: cp.zeros_like(tr[n]) for n in S.PRP_STATE}}
    nz, ny, nx = g.nz, g.ny, g.nx
    tail = (tr["qv"], f32(g.p_top), i32(g.hypso), i32(1), i32(g.base3d),
            i32(nz), i32(ny), i32(nx), i32(0), i32(0), i32(ny), i32(nx),
            T["p"], T["al"], T["alt"], T["p_pp"])
    colb = ((ny * nx + 255) // 256,)
    if g.carried:
        get_kernel("diagnostics", "calc_p_alpha_carried_muts")(colb, (256,), (
            tr["thp"], tr["php"], tr["mup"], tr["alb"], tr["phb"], tr["alb"],
            tr["pb"], tr["muts"], tr["rdnw"], tr["c1h"], tr["c2h"], tr["c3h"],
            tr["c4h"], tr["c3f"], tr["c4f"]) + tail)
    else:
        dummy = cp.zeros(nz + 1, dtype=f32)
        get_kernel("diagnostics", "calc_p_alpha", (nz, ny, nx))(colb, (256,), (
            tr["thp"], tr["php"], tr["mup"], tr["alb"], tr["phb"], tr["alb"],
            tr["alb"], tr["pb"], tr["rdnw"], tr["c1h"], tr["c2h"], tr["c3h"],
            tr["c4h"], tr["c3f"], tr["c4f"], dummy, dummy, tr["mub"]) + tail)
    # qv must move p, or the check is empty: the dry diagnosis differs
    gd = C.prp_geometry(**over)
    sd, _ = run_tl("prp", gd, tr, zeros("prp", gd, tr), True)
    out[f"prpm/{{i}}"] = {{n: same(st[n], T[n]) for n in S.PRP_STATE}}
    out[f"prpm/{{i}}"]["moves"] = not same(sd["p"], T["p"])
for i, over in enumerate(CF["rsp"]):
    dx, dy = 3000.0, 2500.0
    over = dict(over, rdx=float(f32(1) / f32(dx)), rdy=float(f32(1) / f32(dy)))
    g = C.rsp_geometry(**over)
    tr = traj("rsp", g, 41 + i, f32, bool(i % 2))
    st, _ = run_tl("rsp", g, tr, zeros("rsp", g, tr), True)
    scratch = {{}}
    def slot(shape, name):
        return scratch.setdefault(name, cp.empty(shape, dtype=f32))
    state = NS(p=cp.empty((g.nz, g.ny, g.nx), dtype=f32), u=tr["u"], v=tr["v"],
               mup=tr["mup"], mub2d=tr["mub"], c1h=tr["c1h"], c2h=tr["c2h"],
               dnw=tr["dnw"], msfu=tr["msfu"], msfv=1 / tr["msfv"],
               msfv_inv=tr["msfv"], msft=tr["msft"], has_msf=bool(g.has_msf),
               scratch=slot, total_mu=lambda: tr["mub"] + tr["mup"])
    cfg = NS(dx=dx, dy=dy, open_x=bool(g.boundary_x), open_y=bool(g.boundary_y))
    ru, rv, ww = D.stage_fluxes(state, cfg)
    out[f"rsp/{{i}}"] = {{"ru": same(st["ru"], ru), "rv": same(st["rv"], rv),
                         "ww": same(st["ww"], ww)}}
for i, over in enumerate(CF["at"]):
    g = C.at_geometry(**over)
    tr = traj("at", g, 41 + i, f32, bool(i % 2))
    st, _ = run_tl("at", g, tr, zeros("at", g, tr), True)
    T = {{n: tr[n] + tr[n + "f"] for n in S.AT_STATE}}
    if g.heating:
        get_kernel("held_heating", "add_held_heating")(
            ((T["rth_t"].size + 255) // 256,), (256,),
            (T["rth_t"], tr["h_diabatic"], tr["mub"], tr["mup"], tr["c1h"],
             tr["c2h"], tr["msft"], np.uint64(T["rth_t"].size),
             i32(g.ny * g.nx), i32(g.has_msf)))
    out[f"at/{{i}}"] = {{n: same(st[n], T[n]) for n in S.AT_STATE}}
for i, over in enumerate(CF["sw"]):
    dx, dy = 3000.0, 2500.0
    over = dict(over, half_rdx=float(f32(0.5 / dx)), half_rdy=float(f32(0.5 / dy)))
    g = C.sw_geometry(**over)
    tr = traj("sw", g, 41 + i, f32, False)
    st, _ = run_tl("sw", g, tr, zeros("sw", g, tr), True)
    w = tr["w"].copy()
    get_kernel("surface_w", "set_surface_w")(((g.ny * g.nx + 255) // 256,), (256,), (
        tr["u"], tr["v"], tr["ht"], tr["msft"], w, f32(g.cf1), f32(g.cf2),
        f32(g.cf3), f32(g.half_rdx), f32(g.half_rdy), i32(g.mapped),
        i32(g.boundary_x), i32(g.boundary_y), i32(g.ny), i32(g.nx)))
    out[f"sw/{{i}}"] = {{"w": same(st["w"], w)}}
''', tmp_path, extra_env={"GPUWM_WRF_EXACT": "1"})
    assert len(out) == sum(map(len, CONFIGS.values()))
    bad = {k: [n for n, ok in v.items() if not ok] for k, v in out.items()
           if not all(v.values())}
    assert not bad, bad



# The chained stage: rk_step 2 (MUTS from mu_1) and 1 (from mu_2) at the
# substep floor, a lid with hypsometric_opt 2 across the level chunks.
STAGE_CONFIGS = [dict(), dict(rk_step=1, nz=3, ny=2, nx=3),
                 dict(nz=19, ny=3, nx=4, top_lid=1, hypso=2)]
STAGE_HELPERS = """
def copy(d):
    return {k: v.copy() for k, v in d.items()}
def stage_tl(g, tr, pe, tape=None):
    st, sd = copy(tr), copy(pe)
    S.rk_stage_tl(st, sd, g, tape)
    return st, sd
def stage_ad(g, tr, ya):
    tape = []
    st, _ = stage_tl(g, tr, {n: cp.zeros_like(v) for n, v in ya.items()}, tape)
    xa = copy(ya)
    S.rk_stage_ad(xa, st, tape, g)
    assert not tape
    return xa
"""


@pytest.mark.parametrize("backend", BACKENDS)
def test_stage_dot_product(backend, tmp_path):
    """One RK stage with the big-step tendencies held (calc_p_rho_phi,
    rk_step_prep, rk_addtend_dry, small_step_init, small_step_prep's
    MUTS, calc_coef_w, two substeps, the captured MUTS and
    small_step_finish, in woof's strict order): <(T - I) dx, dy> ==
    <dx, (T - I)^T dy> over every perturbed name, 1e-12 in binary64 and
    1e-5 in binary32.  This binds the operators' shared names (p, alt,
    ww_ref, the tendencies, muts) and the stage adjoint's order."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, STAGE_HELPERS + f'''
for dtn in {dtypes!r}:
    dt = np.dtype(dtn)
    for i, over in enumerate({STAGE_CONFIGS!r}):
        for hybrid in (False, True):
            g = C.stage_geometry(**over)
            tr = dev(C.stage_trajectory(g, 11 + i, dt, hybrid=hybrid))
            p = dev(C.stage_perturbation(g, 100 + i, dt))
            q = dev(C.stage_perturbation(g, 200 + i, dt))
            N = list(p)
            def tl(dx):
                sd = stage_tl(g, tr, dict(zip(N, [a.copy() for a in dx])))[1]
                return [sd[n] - d for n, d in zip(N, dx)]
            def ad(dy):
                xa = stage_ad(g, tr, dict(zip(N, [a.copy() for a in dy])))
                return [xa[n] - d for n, d in zip(N, dy)]
            out[f"{{dtn}}/{{i}}/{{hybrid}}"] = K.dot_product(
                tl, ad, [p[n] for n in N], [q[n] for n in N],
                terms=dtn == "float32")
''', tmp_path)
    assert len(out) == len(STAGE_CONFIGS) * 2 * (1 if backend == "mlx" else 2)
    for key, rel in out.items():
        limit = 1e-12 if key.startswith("float64") else 1e-5
        assert rel < limit, (key, rel)


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_stage_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x)|| / ||a TL dx|| -> 1 for the chained stage,
    so each operator is linearised about its own entry state."""
    out = _run(backend, STAGE_HELPERS + f'''
for i, over in enumerate({STAGE_CONFIGS!r}):
    g = C.stage_geometry(**over)
    tr = dev(C.stage_trajectory(g, 31 + i, np.float64, hybrid=bool(i % 2)))
    p = dev(C.stage_perturbation(g, 300 + i, np.float64))
    N = list(p)
    def model(xs):
        t = dict(tr); t.update(dict(zip(N, xs)))
        return [stage_tl(g, t, {{n: cp.zeros_like(t[n]) for n in N}})[0][n] for n in N]
    def tl(dx):
        return [stage_tl(g, tr, dict(zip(N, dx)))[1][n] for n in N]
    out[str(i)] = K.tl_convergence(model, tl, [tr[n] for n in N],
                                   [p[n] for n in N], [1e-2, 1e-4, 1e-6])
''', tmp_path)
    assert len(out) == len(STAGE_CONFIGS)
    for key, rows in out.items():
        misfits = [abs(r - 1.0) for _, r in rows]
        assert misfits[-1] < 1e-5, (key, rows)
        assert misfits[0] < 0.1, (key, rows)


def test_forecast_path_does_not_import_wrfplus():
    """Opt-in: woof's step path imports nothing from woof.grad."""
    code = ("import sys, woof.core.dycore, woof.core.acoustic, woof.runtime\n"
            "bad = sorted(m for m in sys.modules if m.startswith('woof.grad'))\n"
            "assert not bad, bad\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
