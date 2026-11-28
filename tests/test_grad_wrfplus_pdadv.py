"""Positive-definite moist transport TL/AD (P3a item 13,
woof.grad.wrfplus.pdadv): moist_adv_opt = 1's final RK stage on woof's
strict path (the source fold, advect_scalar_pd and the update), and the
advection alone, periodic and specified, with and without map factors,
hybrid levels, h orders 3-6 and v orders 2, 3, 4 and 6.

Breakage this prevents:
- an adjoint that is not the transpose of its TL (dot-product test);
- an adjoint that overwrites instead of adding into its inputs, or that
  leaves the adjoint of a value the primal overwrites (the species, the
  scratch tendency) in place;
- a TL that is not the derivative of the base state (TL vs finite
  differences, on perturbations that keep the limiter's branches: the
  branch words of woof.grad.wrfplus.pdadv.branches are compared at the
  perturbed states and the switches counted);
- a TL/AD base state that drifts from woof's own strict primal
  (moist._advance_species_wrf's PD branch, pd_fluxes and
  pd_renorm_apply): the float build must give the primal's words;
- a TL/AD that departs from WRFPLUS 4.7.1's G_/A_ADVECT_SCALAR_PD (on
  non-hybrid levels, the only ones WRFPLUS linearises) or from WRF 4.7.1's
  nonlinear advect_scalar_pd and its finite differences on hybrid levels
  (tests/data/wrfplus/pdadv.npz, made by tools/wrfplus_oracle);
- a step that runs the non-PD update where the primal limits (the step
  binds PDGeometry on the final stage only).

The step's own words and dot product with moist_adv_opt 1 are in
test_grad_wrfplus_model.py and test_grad_wrfplus_spec_step.py (their
``_pd`` and moist 2 cases).  Each probe runs in a child process, as in
test_grad_wrfplus_scalar.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/wrfplus/pdadv.npz"


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
NO_MLX = [b for b in BACKENDS if b.values[0] != "mlx"]

# The advection alone (pa) and the stage (pd): periodic, map factors,
# specified (one- and two-point rings, the held source), every h order,
# v orders 2, 4 and 6, one level and the 7-point minimum of a bounded
# axis, periodic axes shorter than the stencil, and Courant numbers past
# 0.5 on a share of the faces (``_wind``: flux_upwind's clamp), and the
# specified cases again with RunConfig.pd_adv_west_courant on
# (``west_symmetric``, finding FQ13: the west face in every face's order).
CONFIGS = {
    "pa": [dict(), dict(has_msf=1), dict(spec=1), dict(horder=3, vorder=2),
           dict(horder=6, vorder=6, has_msf=1), dict(horder=4, vorder=4, spec=1),
           dict(nz=1, ny=7, nx=7, spec=1), dict(nz=3, ny=2, nx=3),
           dict(_wind=40.0, has_msf=1), dict(_wind=40.0, spec=1, horder=3),
           dict(spec=1, has_msf=1, west_symmetric=1),
           dict(_wind=40.0, spec=1, horder=3, west_symmetric=1)],
    "pd": [dict(), dict(has_msf=1), dict(spec=1), dict(spec=1, held=1, has_msf=1),
           dict(spec=1, spec_zone=2, horder=3, vorder=2), dict(horder=6, vorder=6, held=1),
           dict(horder=4, vorder=4, held=1), dict(nz=1, ny=7, nx=7, spec=1, held=1),
           dict(_wind=40.0, spec=1, held=1),
           dict(spec=1, held=1, has_msf=1, west_symmetric=1),
           dict(_wind=40.0, spec=1, spec_zone=2, horder=4, vorder=4, west_symmetric=1)],
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
        "from woof.grad.wrfplus import pdadv as P, cases as C, _checks as K\n"
        "from woof.grad.wrfplus._mode import base_only\n"
        "out = {}\n"
        "# state names, outputs, active names, geometry, trajectory, perturbation, tl, ad\n"
        "OPS = {\n"
        " 'pa': (P.PA_STATE, ('tend',), P.pa_active, C.pd_geometry, C.pa_trajectory,\n"
        "        C.pa_perturbation, P.pa_tl, P.pa_ad),\n"
        " 'pd': (P.PD_STATE, ('q',), P.pd_active, C.pd_geometry, C.pd_trajectory,\n"
        "        C.pd_perturbation, P.pd_tl, P.pd_ad)}\n"
        "def case(op, over, seed, dt, hybrid):\n"
        "    over = dict(over)\n"
        "    wind = over.pop('_wind', 1.0)\n"
        "    g = OPS[op][3](**over)\n"
        "    tr = OPS[op][4](g, seed, dt, hybrid=hybrid, wind=wind)\n"
        "    return g, dev(tr), dev(OPS[op][5](g, seed + 1000, dt, traj=tr))\n"
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n"
        "def run_tl(op, g, tr, pe, nl):\n"
        "    St, tlf = OPS[op][0], OPS[op][6]\n"
        "    st = {n: tr[n].copy() for n in St}\n"
        "    sd = {n: cp.zeros_like(tr[n]) for n in St}\n"
        "    tlf(st, sd, tr, pe, g, nl=nl)\n"
        "    return st, sd\n"
        "def run_nl(op, g, tr):\n"
        "    St, tlf = OPS[op][0], OPS[op][6]\n"
        "    st = {n: tr[n].copy() for n in St}\n"
        "    with base_only():\n"
        "        tlf(st, st, tr, tr, g, nl=True)\n"
        "    return st\n"
        "def run_ad(op, g, tr, seed, adj=None):\n"
        "    St, X, adf = OPS[op][0], OPS[op][2](g), OPS[op][7]\n"
        "    sa = {n: (seed[n].copy() if n in seed else cp.zeros_like(tr[n])) for n in St}\n"
        "    if adj is None:\n"
        "        adj = {n: cp.zeros_like(tr[n]) for n in X}\n"
        "    adf(sa, tr, adj, g)\n"
        "    return sa, adj\n"
        "def seeds(op, tr, s, dt):\n"
        "    rng = np.random.default_rng(s)\n"
        "    return {n: cp.asarray(rng.standard_normal(tr[n].shape).astype(dt)) for n in OPS[op][1]}\n"
        "def switches(op, g, a, b):\n"
        "    # branch words that differ between two base states\n"
        "    x, y = P.branches(a, g, stage=op == 'pd'), P.branches(b, g, stage=op == 'pd')\n"
        "    return sum(int((x[k] != y[k]).sum()) for k in x)\n")
    script = prelude + body + "\nprint('RESULT ' + json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


@pytest.mark.parametrize("backend", BACKENDS)
def test_dot_product(backend, tmp_path):
    """<TL dx, dy> == <dx, AD dy>: 1e-12 in binary64; 1e-5 in binary32."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f'''
for dtn in {dtypes!r}:
    dt = np.dtype(dtn)
    for op, cfgs in {CONFIGS!r}.items():
        St, Out, A = OPS[op][:3]
        for i, over in enumerate(cfgs):
            g, tr, p = case(op, over, 11 + i, dt, bool(i % 2))
            X = A(g)
            dy = seeds(op, tr, 200 + i, dt)
            def tl(dx):
                return [run_tl(op, g, tr, dict(zip(X, dx)), False)[1][n] for n in Out]
            def ad(dys):
                return [run_ad(op, g, tr, dict(zip(Out, dys)))[1][n] for n in X]
            out[f"{{dtn}}/{{op}}/{{i}}"] = K.dot_product(
                tl, ad, [p[n] for n in X], [dy[n] for n in Out], terms=dtn == "float32")
''', tmp_path)
    n = sum(len(c) for c in CONFIGS.values())
    assert len(out) == (1 if backend == "mlx" else 2) * n
    for key, rel in out.items():
        limit = 1e-12 if key.startswith("float64") else 1e-5
        assert rel < limit, (key, rel)


@pytest.mark.parametrize("backend", BACKENDS)
def test_adjoint_accumulates(backend, tmp_path):
    """The AD adds into ``adj`` and consumes the adjoints of what the
    primal overwrites: the species and the scratch tendency end at zero."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
dt = np.dtype({dtn!r})
for op, cfgs in {CONFIGS!r}.items():
    St, Out, A = OPS[op][:3]
    for i, over in enumerate(cfgs):
        g, tr, s = case(op, over, 51 + i, dt, bool(i % 2))
        X = A(g)
        q = seeds(op, tr, 500 + i, dt)
        sa, zero = run_ad(op, g, tr, q)
        _, adj = run_ad(op, g, tr, q, {{n: s[n].copy() for n in X}})
        for n in St:
            out[f"{{op}}/{{i}}/state/{{n}}"] = float(cp.abs(sa[n]).max())
        for n in X:
            want = cp.asnumpy(s[n]).astype(np.float64) + cp.asnumpy(zero[n])
            got = cp.asnumpy(adj[n]).astype(np.float64)
            scale = max(float(np.abs(want).max()), 1e-300)
            out[f"{{op}}/{{i}}/{{n}}"] = float(np.abs(got - want).max()) / scale
''', tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-13
    bad = {k: v for k, v in out.items() if not v < limit}
    assert not bad, bad


@pytest.mark.parametrize("backend", BACKENDS)
def test_adjoint_in_one_dict(backend, tmp_path):
    """The step passes one dict as the adjoint state and the input
    adjoints (the species is both): the stage's AD gives the
    separate-dict result there, the species' own seed consumed before
    the advection adds into it."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
dt = np.dtype({dtn!r})
op = "pd"
St, Out, A, _, _, _, _, adf = OPS[op]
for i, over in enumerate({CONFIGS["pd"]!r}):
    g, tr, _ = case(op, over, 71 + i, dt, bool(i % 2))
    X = A(g)
    q = seeds(op, tr, 700 + i, dt)
    sa, adj = run_ad(op, g, tr, q)
    one = {{n: cp.zeros_like(tr[n]) for n in set(St) | set(X)}}
    for n in Out:
        one[n][...] = q[n]
    adf(one, tr, one, g)
    for n in set(St) | set(X):
        want = adj[n] if n in X else sa[n]
        scale = max(float(cp.abs(want).max()), 1e-300)
        out[f"{{op}}/{{i}}/{{n}}"] = float(cp.abs(one[n] - want).max()) / scale
''', tmp_path)
    limit = 1e-6 if backend == "mlx" else 1e-14
    bad = {k: v for k, v in out.items() if not v < limit}
    assert not bad, bad


@pytest.mark.parametrize("backend", NO_MLX)
def test_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x) - a TL dx|| / ||a TL dx|| -> 0 as O(a), in
    binary64.  The limiter is not differentiable where a branch switches:
    the cases' perturbations scale q0 by itself (an empty cell stays
    empty), the branch words of the base and of x +- a dx are compared
    at every a, and at the smallest a none may switch (the switch counts
    at the larger a are reported, and bounded at 1% of the words)."""
    out = _run(backend, f'''
for op, cfgs in {CONFIGS!r}.items():
    St, Out, A = OPS[op][:3]
    for i, over in enumerate(cfgs):
        g, tr, p = case(op, over, 31 + i, np.float64, bool(i % 2))
        X = A(g)
        def model(xs):
            t = dict(tr); t.update(dict(zip(X, xs)))
            return [run_nl(op, g, t)[n] for n in Out]
        def tl(dx):
            return [run_tl(op, g, tr, dict(zip(X, dx)), False)[1][n] for n in Out]
        alphas = [1e-3, 1e-5, 1e-7]
        rows = K.tl_residual(model, tl, [tr[n] for n in X], [p[n] for n in X], alphas)
        sw = []
        for a in alphas:
            n = 0
            for s in (a, -a):
                t = dict(tr); t.update({{k: tr[k] + s * p[k] for k in X}})
                n += switches(op, g, tr, t)
            sw.append(n)
        words = sum(int(v.size) for v in P.branches(tr, g, stage=op == "pd").values())
        cells = P.branches(tr, g, stage=op == "pd")["cell"]
        out[f"{{op}}/{{i}}"] = dict(rows=rows, switches=sw, words=words,
                                    limited=int((cells == 1).sum()), cells=int(cells.size))
''', tmp_path)
    assert len(out) == sum(len(c) for c in CONFIGS.values())
    for key, r in out.items():
        misfits = [m for _, m in r["rows"]]
        # the limiter acts on every case but the 3x2x3 periodic one
        assert r["limited"] > 0 or r["cells"] < 49, (key, r)
        assert r["switches"][-1] == 0, (key, r)
        assert max(r["switches"]) <= 0.01 * r["words"], (key, r)
        assert misfits[-1] < 1e-6, (key, r)
        assert misfits[0] < 0.1, (key, r)


@pytest.mark.parametrize("backend", NO_MLX)
def test_float_base_state_is_the_strict_primal(backend, tmp_path):
    """The float build's base state gives woof's strict words: the stage
    against moist._advance_species_wrf's PD branch (the held source where
    the config has one, the carried muts as the primal reads it), the
    advection against launch_pd_fluxes + launch_pd_renorm_apply.  Each
    case checks the limiter acts and the stage's words differ from the
    non-PD update's (woof.grad.wrfplus.scalar on the same state), so a
    binding that ran the unlimited update cannot pass.  Host and CUDA;
    Metal is not checked here."""
    out = _run(backend, f'''
from types import SimpleNamespace as NS
from woof.core import moist as M
from woof.grad.wrfplus import scalar as S
from woof.state_serialization_contract import WRF_MUTS_SLOT
f32 = np.float32
def words(a, b):
    a, b = cp.asnumpy(a), cp.asnumpy(b)
    return int((a.view(np.uint32) != b.view(np.uint32)).sum())
def coord(g, tr, q0):
    pool = {{}}
    def scratch(shape, name):
        return pool.setdefault(name, cp.zeros(shape, f32))
    return NS(scratch=scratch, msft=tr["msft"], has_msf=bool(g.has_msf), c1h=tr["c1h"],
              c2h=tr["c2h"], rdnw=tr["rdnw"], fnm=tr["fnm"], fnp=tr["fnp"],
              mub2d=tr["mub2d"], mup0=tr["mup0"],
              existing_scratch=lambda name: tr["muts"] if name == WRF_MUTS_SLOT else None)
def bufs(g):
    nz, ny, nx = g.nz, g.ny, g.nx
    return [cp.zeros(s, f32) for s in ((nz, ny, nx + 1),) * 2 + ((nz, ny + 1, nx),) * 2
            + ((nz + 1, ny, nx),) * 2]
for i, over in enumerate({CONFIGS["pd"]!r}):
    g, tr, _ = case("pd", over, 41 + i, f32, bool(i % 2))
    st = run_nl("pd", g, tr)
    state = coord(g, tr, tr["q0"])
    cfg = NS(specified=bool(g.spec), nested=False, spec_zone=g.spec_zone, dx=g.dx, dy=g.dy,
             cu_physics=0, v_sca_adv_order=g.vorder, h_sca_adv_order=g.horder,
             moist_adv_opt=1, pd_adv_west_courant=bool(g.west_symmetric))
    mu0, mu = tr["mub2d"] + tr["mup0"], tr["mub2d"] + tr["mup"]
    chm0 = tr["c1h"][:, None, None] * mu0[None] + tr["c2h"][:, None, None]
    q, tend = tr["q"].copy(), cp.zeros_like(tr["q"])
    M._advance_species_wrf(
        state, cfg, "qc", q, tr["q0"], tend,
        tr["held"] if g.held else cp.zeros_like(q),
        ru=tr["ru_m"], rv=tr["rv_m"], ww=tr["ww_m"], implicit=None, pd=True, bufs=bufs(g),
        mu0=mu0, mu=mu, mu0_row=mu0.reshape(-1), mu_row=mu.reshape(-1), chm0=chm0,
        dt_eff=g.dt, final=True, physics_tendencies=None, fixed_tendencies=None,
        recompute=False, apply_scalar_lbc=None, export_advective_forcing=False,
        boundary_x=bool(g.spec), boundary_y=bool(g.spec))
    out[f"pd/{{i}}/q"] = words(st["q"], q)
    out[f"pd/{{i}}/tend"] = words(st["tend"], tend)
    out[f"pd/{{i}}/moves"] = words(q, tr["q"])
    out[f"pd/{{i}}/limited"] = int((P.branches(tr, g)["cell"] == 1).sum())
    sg = S.ScalarGeometry(g.nz, g.ny, g.nx, float(f32(1.0 / g.dx)), float(f32(1.0 / g.dy)),
                          g.dt, vorder=g.vorder, horder=g.horder, open_x=g.spec,
                          open_y=g.spec, spec=g.spec, spec_zone=g.spec_zone,
                          has_msf=g.has_msf, held=g.held)
    su = {{n: tr[n].copy() for n in S.SU_STATE}}
    S.su_tl(su, {{n: cp.zeros_like(tr[n]) for n in S.SU_STATE}}, tr,
            {{n: cp.zeros_like(tr[n]) for n in S.su_active(sg)}}, sg, nl=True)
    out[f"pd/{{i}}/differs_from_non_pd"] = words(st["q"], su["q"])
for i, over in enumerate({CONFIGS["pa"]!r}):
    g, tr, _ = case("pa", over, 61 + i, f32, bool(i % 2))
    st = run_nl("pa", g, tr)
    state = coord(g, tr, tr["q0e"])
    b = bufs(g)
    ref = cp.zeros_like(tr["q"])
    M.launch_pd_fluxes(tr["q"], tr["q0e"], tr["ru_m"], tr["rv_m"], tr["ww_m"], tr["muts"],
                       state, g.dx, g.dy, g.dt, *b, msft=tr["msft"], has_msf=bool(g.has_msf),
                       open_x=bool(g.spec), open_y=bool(g.spec), vorder=g.vorder,
                       horder=g.horder, west_symmetric=bool(g.west_symmetric))
    M.launch_pd_renorm_apply(tr["q0e"], tr["mub2d"] + tr["mup0"], *b, tend=ref, coord=state,
                             dx=g.dx, dy=g.dy, dt=g.dt, msft=tr["msft"],
                             has_msf=bool(g.has_msf), open_x=bool(g.spec),
                             open_y=bool(g.spec), zero_first=True)
    out[f"pa/{{i}}/tend"] = words(st["tend"], ref)
    out[f"pa/{{i}}/moves"] = words(ref, cp.zeros_like(ref))
''', tmp_path, extra_env=STRICT_ENV)
    assert len(out) == 5 * len(CONFIGS["pd"]) + 2 * len(CONFIGS["pa"])
    must_move = ("/moves", "/limited", "/differs_from_non_pd")
    bad = {k: v for k, v in out.items() if (v == 0 if k.endswith(must_move) else v)}
    assert not bad, bad


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrfplus_oracle_agreement(backend, tmp_path):
    """On non-hybrid levels woof's advection TL/AD reproduce WRFPLUS 4.7.1
    G_/A_ADVECT_SCALAR_PD (the tendency, its TL and every input adjoint;
    with the Courant number under 0.5 the upwind flux does not depend on
    the face mass, and both adjoints of muts are roundoff, so that one is
    scaled by its terms' size).  On hybrid levels, which WRFPLUS does not
    linearise, woof's base is WRF 4.7.1's nonlinear advect_scalar_pd, its
    TL the fourth-order central difference of that routine along the
    perturbation, and its AD's projection on the perturbation the seed's
    on that difference; the difference's points (+-h, +-2h) take the base
    state's branches.  A periodic axis reads its end-face mass flux through
    woof's column n and WRF's halo column 0 (one value in the fixture): the
    adjoints are compared as a sum.  8 cases."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
import json as J
z = np.load({str(FIXTURE)!r})
dt = np.dtype({dtn!r})
for name in sorted({{k.split("/")[0] for k in z.files if "/" in k}}):
    g = P.PDGeometry(**J.loads(str(z[name + "/geometry"])))
    for label in ("plain", "hybrid"):
        pre = f"{{name}}/{{label}}/"
        pick = lambda kind: {{k[len(pre + kind) + 1:]: cp.asarray(z[k].astype(dt))
                              for k in z.files if k.startswith(pre + kind + "/")}}
        tr, pe = pick("traj"), pick("pert")
        at = z[pre + "seed/tend"]
        st, sd = run_tl("pa", g, tr, pe, True)
        _, adj = run_ad("pa", g, tr, {{"tend": cp.asarray(at.astype(dt))}})
        h = lambda a: cp.asnumpy(a).astype(np.float64)
        rel = lambda w, e, s=None: float(np.abs(w - e).max()) / max(
            float(np.abs(e).max()) if s is None else s, 1e-300)
        if label == "plain":
            w = pre + "wrfplus/"
            out[pre + "tl_tend"] = rel(h(st["tend"]), z[w + "tl_tend"])
            out[pre + "tl_d_tend"] = rel(h(sd["tend"]), z[w + "tl_d_tend"])
            for n in P.PA_ACTIVE:
                a, e = h(adj[n]), z[w + "ad_" + n].copy()
                if not g.spec:
                    for x in (a, e):
                        if n == "ru_m":
                            x[:, :, 0] += x[:, :, -1]; x[:, :, -1] = 0
                        if n == "rv_m":
                            x[:, 0, :] += x[:, -1, :]; x[:, -1, :] = 0
                s = None
                if n == "muts":
                    s = max(float(np.abs(e).max()), float(np.abs(z[w + "ad_q0e"]).max())
                            * float(np.abs(z[pre + "traj/q0e"]).max())
                            / float(np.abs(z[pre + "traj/muts"]).max()))
                out[pre + "ad_" + n] = rel(a, e, s)
        else:
            w = pre + "wrf/"
            fd = z[w + "fd_tend"]
            out[pre + "nl_tend"] = rel(h(st["tend"]), z[w + "nl_tend"])
            out[pre + "tl_d_tend"] = rel(h(sd["tend"]), fd)
            lhs = sum(float(np.vdot(h(adj[n]), h(pe[n]))) for n in P.PA_ACTIVE)
            rhs = float(np.vdot(at, fd))
            out[pre + "ad_projection"] = abs(lhs - rhs) / abs(rhs)
            if {dtn!r} == "float64":
                step = float(z[w + "step"])
                n = 0
                for s in (2, -2):
                    t = {{k: (v + s * step * pe[k] if k in pe else v) for k, v in tr.items()}}
                    n += switches("pa", g, tr, t)
                out[pre + "switches"] = n
''', tmp_path)
    assert len(out) == 8 * (9 + (4 if backend != "mlx" else 3))
    limits = {"float64": dict(plain=1e-12, nl=1e-12, fd=1e-8),
              "float32": dict(plain=2e-5, nl=2e-5, fd=1e-3)}[
        "float32" if backend == "mlx" else "float64"]
    bad = {}
    for key, v in out.items():
        if key.endswith("switches"):
            ok = v == 0
        elif "/plain/" in key:
            ok = v < limits["plain"]
        elif key.endswith("nl_tend"):
            ok = v < limits["nl"]
        else:
            ok = v < limits["fd"]
        if not ok:
            bad[key] = v
    assert not bad, sorted(bad.items())


@pytest.mark.parametrize("backend", NO_MLX[:1])
def test_step_binds_the_pd_final_stage(backend, tmp_path):
    """A moist step with moist_adv_opt 1 binds every species to the PD
    stage on rk_step 3 only (rk_steps 1 and 2 keep the non-PD update, as
    moist.advance_scalars_stage's ``final``), with the primal's dx, dy,
    orders and the stage's dt; moist_adv_opt 0 binds none.  The step's
    words and dot product with the stage are test_grad_wrfplus_model's
    ``_pd`` cases; v_sca_adv_order 5 is refused there."""
    out = _run(backend, '''
from tilestream import harness as H
if cp.__name__ != 'cupy' or getattr(cp, '__woof_facade__', False):
    import woof.core.resident_admission as RA
    RA.device_free_bytes = lambda: None
from woof.grad.wrfplus import model as M
for key, over in (("pd", dict(moist_adv_opt=1)), ("v2", dict(moist_adv_opt=1, v_sca_adv_order=2)),
                  ("off", dict(moist_adv_opt=0))):
    cfg = H.make_config(8, 6, 6, h_sca_adv_order=5, terrain_opt=1, moist=True, **over)
    st = H.make_state(cfg, 1, geography=H.neutral_geography(cfg))
    g = M.step_geometry(st, cfg)
    out[key] = [sorted({type(sg).__name__ + "/" + str(sg.vorder) for _, sg in s.su})
                for s in g.stages]
    out[key + "/n"] = [len(s.su) for s in g.stages]
    last = g.stages[-1].su[0][1]
    if isinstance(last, P.PDGeometry):
        sdt = float(g.stages[-1].su[0][1].dt)
        out[key + "/geom"] = [last.dx == float(cfg.dx), last.dy == float(cfg.dy),
                              last.horder == 5, last.held == 0, sdt > 0]
''', tmp_path)
    assert min(out["pd/n"]) > 0, out
    assert out["pd"] == [["ScalarGeometry/3"], ["ScalarGeometry/3"], ["PDGeometry/3"]], out
    assert out["pd/geom"] == [True] * 5, out
    assert out["v2"][-1] == ["PDGeometry/2"], out
    assert all(s == ["ScalarGeometry/3"] for s in out["off"]), out


def test_geometry_rejects_unported_options():
    """v_sca_adv_order 5 (the HRRR fork's vertical arm), h order 2, a
    specified axis under 7 cells, a ring that does not fit or without a
    specified domain, flags outside 0/1 (west_symmetric included), a non-positive or non-finite dt
    or spacing, and empty axes are refused, not run."""
    from woof.grad.wrfplus import cases as C
    for over in (dict(vorder=5), dict(horder=2), dict(spec=1, nx=6), dict(spec=1, spec_zone=5),
                 dict(spec=1, spec_zone=0), dict(spec_zone=1), dict(has_msf=2), dict(held=2),
                 dict(west_symmetric=2),
                 dict(dt=0.0), dict(dt=float("nan")), dict(dx=-1.0), dict(nz=0)):
        with pytest.raises(ValueError):
            C.pd_geometry(**over).validate()


# RunConfig.pd_adv_west_courant (finding FQ13): specified cases, both
# second-order bands, the clamp and map factors.
WEST_CASES = [dict(spec=1), dict(spec=1, has_msf=1, horder=3),
              dict(_wind=40.0, spec=1, horder=6, vorder=6),
              dict(spec=1, spec_zone=2, horder=4, has_msf=1)]

_MIRROR = '''
def host(d):
    return {k: cp.asnumpy(v) for k, v in d.items()}
def mirror(tr):
    # x -> nx-1-x: cells and y/eta faces reversed in i, x faces f -> nx-f
    # with the mass flux negated
    out = {}
    for k, v in host(tr).items():
        m = v[..., ::-1] if v.ndim >= 2 else v
        out[k] = cp.asarray(np.ascontiguousarray(-m if k == "ru_m" else m))
    return out
def mirrored(f, fm, tend, tendm):
    # numeric (0 == -0) mismatches of the mirrored run's fluxes against the
    # mirror of the run's, by array; "tend" the tendency's largest relative
    # mismatch (its divergence sums the four x terms of a cell in one
    # order, (r - l) + r - l, so the mirror reassociates them)
    f, fm = host(f), host(fm)
    bad = {}
    for n in P.FLUXES:
        want = -f[n][..., ::-1] if n in ("fxl", "fxc") else f[n][..., ::-1]
        bad[n] = int((fm[n] != want).sum())
    a, b = cp.asnumpy(tendm).astype(np.float64), cp.asnumpy(tend)[..., ::-1].astype(np.float64)
    bad["tend"] = float(np.abs(a - b).max()) / max(float(np.abs(b).max()), 1e-300)
    return bad
'''


@pytest.mark.parametrize("backend", BACKENDS)
def test_west_courant_correction_moves_only_the_west_face(backend, tmp_path):
    """FQ13 in the TL/AD's base: with west_symmetric the only flux words
    that move are the upwind and correction fluxes of the x face i == 1
    (WRF's ids+1), and some do (the two orders round differently); and
    with it on a mirrored flow (x reversed, ru negated) gives the
    mirrored fluxes word for word, which WRF's west face breaks, and the
    mirrored tendency to the divergence's reassociation.  Binary64 and
    binary32 (binary32 only on Metal)."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, _MIRROR + f'''
import dataclasses
for dtn in {dtypes!r}:
    dt = np.dtype(dtn)
    for i, over in enumerate({WEST_CASES!r}):
        over = dict(over)
        wind = over.pop("_wind", 1.0)
        g0 = C.pd_geometry(**over)
        g1 = dataclasses.replace(g0, west_symmetric=1)
        tr = dev(C.pa_trajectory(g0, 91 + i, dt, hybrid=bool(i % 2), wind=wind))
        t0, f0, _ = P._advect_base(tr, g0, dt)
        t1, f1, _ = P._advect_base(tr, g1, dt)
        key = f"{{dtn}}/{{i}}"
        h0, h1 = host(f0), host(f1)
        for n in P.FLUXES:
            d = h0[n] != h1[n]
            if n in ("fxl", "fxc"):
                out[key + "/west/" + n] = int(d[..., 1].sum())
                d[..., 1] = False
            out[key + "/elsewhere/" + n] = int(d.sum())
        trm = mirror(tr)
        for name, g in (("wrf", g0), ("corrected", g1)):
            tm, fm, _ = P._advect_base(trm, g, dt)
            t, f, _ = (t0, f0, None) if g is g0 else (t1, f1, None)
            out[key + "/mirror/" + name] = mirrored(f, fm, t, tm)
''', tmp_path)
    for dtn in (["float32"] if backend == "mlx" else ["float64", "float32"]):
        moved = sum(out[f"{dtn}/{i}/west/fxl"] for i in range(len(WEST_CASES)))
        assert moved > 0, (dtn, out)
        asym = sum(out[f"{dtn}/{i}/mirror/wrf"]["fxl"] for i in range(len(WEST_CASES)))
        assert asym > 0, (dtn, out)
        for i in range(len(WEST_CASES)):
            key = f"{dtn}/{i}"
            assert all(out[f"{key}/elsewhere/{n}"] == 0 for n in
                       ("fxl", "fxc", "fyl", "fyc", "fzl", "fzc")), (key, out)
            mir = dict(out[f"{key}/mirror/corrected"])
            assert mir.pop("tend") < (1e-12 if dtn == "float64" else 1e-5), (key, out)
            assert all(v == 0 for v in mir.values()), (key, out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_west_courant_correction_strict_primal(backend, tmp_path):
    """FQ13 in woof's strict primal (pd_fluxes, GPUWM_WRF_EXACT=1):
    west_symmetric moves only the x face i == 1, and some of its words;
    with it on a mirrored flow gives the mirrored fluxes word for word;
    and woof.core.moist.pd_west_symmetric reads the key only where
    moist_adv_opt = 1."""
    out = _run(backend, _MIRROR + f'''
from types import SimpleNamespace as NS
from woof.core import moist as M
f32 = np.float32
out["read"] = [M.pd_west_symmetric(NS(pd_adv_west_courant=a, moist_adv_opt=b))
               for a, b in ((True, 1), (False, 1), (True, 0))]
def coord(tr):
    return NS(c1h=tr["c1h"], c2h=tr["c2h"], rdnw=tr["rdnw"], fnm=tr["fnm"], fnp=tr["fnp"])
def fluxes(g, tr, west):
    nz, ny, nx = g.nz, g.ny, g.nx
    b = [cp.zeros(s, f32) for s in ((nz, ny, nx + 1),) * 2 + ((nz, ny + 1, nx),) * 2
         + ((nz + 1, ny, nx),) * 2]
    M.launch_pd_fluxes(tr["q"], tr["q0e"], tr["ru_m"], tr["rv_m"], tr["ww_m"], tr["muts"],
                       coord(tr), g.dx, g.dy, g.dt, *b, msft=tr["msft"],
                       has_msf=bool(g.has_msf), open_x=True, open_y=True, vorder=g.vorder,
                       horder=g.horder, west_symmetric=west)
    return dict(zip(P.FLUXES, b))
for i, over in enumerate({WEST_CASES!r}):
    over = dict(over)
    wind = over.pop("_wind", 1.0)
    g = C.pd_geometry(**over)
    tr = dev(C.pa_trajectory(g, 91 + i, f32, hybrid=bool(i % 2), wind=wind))
    h0, h1 = host(fluxes(g, tr, False)), host(fluxes(g, tr, True))
    for n in P.FLUXES:
        d = h0[n] != h1[n]
        if n in ("fxl", "fxc"):
            out[f"{{i}}/west/{{n}}"] = int(d[..., 1].sum())
            d[..., 1] = False
        out[f"{{i}}/elsewhere/{{n}}"] = int(d.sum())
    trm = mirror(tr)
    z = cp.zeros((1,), f32)
    for name, west in (("wrf", False), ("corrected", True)):
        bad = mirrored(fluxes(g, tr, west), fluxes(g, trm, west), z, z)
        bad.pop("tend")
        out[f"{{i}}/mirror/{{name}}"] = bad
''', tmp_path, extra_env=STRICT_ENV)
    assert out["read"] == [True, False, False], out
    n = len(WEST_CASES)
    assert sum(out[f"{i}/west/fxl"] for i in range(n)) > 0, out
    assert sum(out[f"{i}/mirror/wrf"]["fxl"] for i in range(n)) > 0, out
    for i in range(n):
        assert all(out[f"{i}/elsewhere/{f}"] == 0 for f in
                   ("fxl", "fxc", "fyl", "fyc", "fzl", "fzc")), (i, out)
        assert all(v == 0 for v in out[f"{i}/mirror/corrected"].values()), (i, out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_west_courant_key_leaves_the_default_build_unchanged(backend, tmp_path):
    """FQ13 in woof's default build: pd_fluxes' face never carried WRF's
    (ru/mu)*dt/dx, so west_symmetric moves no word of any flux there."""
    out = _run(backend, _MIRROR + f'''
from types import SimpleNamespace as NS
from woof.core import moist as M
f32 = np.float32
def fluxes(g, tr, west):
    nz, ny, nx = g.nz, g.ny, g.nx
    b = [cp.zeros(s, f32) for s in ((nz, ny, nx + 1),) * 2 + ((nz, ny + 1, nx),) * 2
         + ((nz + 1, ny, nx),) * 2]
    M.launch_pd_fluxes(tr["q"], tr["q0e"], tr["ru_m"], tr["rv_m"], tr["ww_m"], tr["muts"],
                       NS(c1h=tr["c1h"], c2h=tr["c2h"], rdnw=tr["rdnw"], fnm=tr["fnm"],
                          fnp=tr["fnp"]), g.dx, g.dy, g.dt, *b, msft=tr["msft"],
                       has_msf=bool(g.has_msf), open_x=True, open_y=True, vorder=g.vorder,
                       horder=g.horder, west_symmetric=west)
    return host(dict(zip(P.FLUXES, b)))
for i, over in enumerate({WEST_CASES!r}):
    over = dict(over)
    wind = over.pop("_wind", 1.0)
    g = C.pd_geometry(**over)
    tr = dev(C.pa_trajectory(g, 91 + i, f32, hybrid=bool(i % 2), wind=wind))
    h0, h1 = fluxes(g, tr, False), fluxes(g, tr, True)
    out[str(i)] = {{n: int((h0[n].view(np.uint32) != h1[n].view(np.uint32)).sum())
                   for n in P.FLUXES}}
''', tmp_path)
    for key, diff in out.items():
        assert all(v == 0 for v in diff.values()), (key, out)
