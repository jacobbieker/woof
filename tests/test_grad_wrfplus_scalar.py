"""Moist scalar transport TL/AD (P3a 8j, woof.grad.wrfplus.scalar): WRF
sumflux and one species' non-PD rk_scalar_tend + rk_update_scalar on
woof's strict path, periodic and specified (the ring cleared, the held
boundary tendency added), with and without map factors, hybrid levels.

Breakage this prevents:
- an adjoint that is not the transpose of its TL (dot-product test);
- an adjoint that overwrites instead of adding into its inputs, or that
  leaves the adjoint of a value the primal overwrites (the species, the
  sumflux accumulators) in place;
- a TL that is not the derivative of the base-state update
  (TL vs finite differences);
- a TL/AD base state that drifts from woof's own strict primal
  (moist._advance_species_wrf and the sumflux kernels): the float build
  must give the primal's words.

There is no WRFPLUS fixture: WRFPLUS 4.7.1 has no moist scalar TL in the
dry-core oracle build.  Each probe runs in a child process, as in
test_grad_wrfplus_bigstep.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


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

# periodic; map factors; specified with and without the held tendency
# (one-point and two-point rings); every h order; one level and the
# 7-point minimum of a bounded axis; a periodic held source (a mixing row).
CONFIGS = {
    "su": [dict(), dict(has_msf=1), dict(spec=1), dict(spec=1, held=1, has_msf=1),
           dict(spec=1, held=1, spec_zone=2, horder=3, vorder=2), dict(horder=4, vorder=4),
           dict(horder=6, vorder=6, has_msf=1), dict(nz=1, ny=7, nx=7, spec=1, held=1),
           dict(held=1, has_msf=1)],
    "sf": [dict(), dict(nsub=1), dict(nz=1, ny=1, nx=1, nsub=3)],
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
        "from woof.grad.wrfplus import scalar as S, cases as C, _checks as K\n"
        "out = {}\n"
        "def sf_tl(st, sd, tr, pe, g, nl=True):\n"
        "    S.sumflux_zero_tl(st, sd, g, nl=nl)\n"
        "    for _ in range(g.nsub):\n"
        "        S.sumflux_accumulate_tl(st, sd, tr, pe, g, nl=nl)\n"
        "    S.sumflux_finish_tl(st, sd, tr, pe, g, nl=nl)\n"
        "def sf_ad(sa, tr, adj, g):\n"
        "    S.sumflux_finish_ad(sa, tr, adj, g)\n"
        "    for _ in range(g.nsub):\n"
        "        S.sumflux_accumulate_ad(sa, tr, adj, g)\n"
        "    S.sumflux_zero_ad(sa, g)\n"
        "# state names, outputs, active names, geometry, trajectory, perturbation, tl, ad\n"
        "OPS = {\n"
        " 'su': (S.SU_STATE, ('q',), S.su_active, C.su_geometry, C.su_trajectory,\n"
        "        C.su_perturbation, S.su_tl, S.su_ad),\n"
        " 'sf': (S.SUMFLUX, S.SUMFLUX, lambda g: S.SUMFLUX_SRC + S.SUMFLUX_REF,\n"
        "        C.sf_geometry, C.sf_trajectory, C.sf_perturbation, sf_tl, sf_ad)}\n"
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n"
        "def run_tl(op, g, tr, pe, nl):\n"
        "    St, tlf = OPS[op][0], OPS[op][6]\n"
        "    st = {n: tr[n].copy() for n in St}\n"
        "    sd = {n: cp.zeros_like(tr[n]) for n in St}\n"
        "    tlf(st, sd, tr, pe, g, nl=nl)\n"
        "    return st, sd\n"
        "def run_ad(op, g, tr, seed, adj=None):\n"
        "    St, X, adf = OPS[op][0], OPS[op][2](g), OPS[op][7]\n"
        "    sa = {n: (seed[n].copy() if n in seed else cp.zeros_like(tr[n])) for n in St}\n"
        "    if adj is None:\n"
        "        adj = {n: cp.zeros_like(tr[n]) for n in X}\n"
        "    adf(sa, tr, adj, g)\n"
        "    return sa, adj\n"
        "def seeds(op, tr, s, dt):\n"
        "    rng = np.random.default_rng(s)\n"
        "    return {n: cp.asarray(rng.standard_normal(tr[n].shape).astype(dt)) for n in OPS[op][1]}\n")
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
        St, Out, A, G, T, P = OPS[op][:6]
        for i, over in enumerate(cfgs):
            g = G(**over)
            X = A(g)
            tr = dev(T(g, 11 + i, dt, hybrid=bool(i % 2)))
            p = dev(P(g, 100 + i, dt))
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
    primal overwrites: the species, its scratch tendency and the sumflux
    accumulators end at zero."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
dt = np.dtype({dtn!r})
for op, cfgs in {CONFIGS!r}.items():
    St, Out, A, G, T, P = OPS[op][:6]
    for i, over in enumerate(cfgs):
        g = G(**over)
        X = A(g)
        tr = dev(T(g, 51 + i, dt, hybrid=bool(i % 2)))
        q = seeds(op, tr, 500 + i, dt)
        s = dev(P(g, 600 + i, dt))
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
    adjoints (the species is both): the AD gives the separate-dict
    result there, the species' own seed consumed before the advection
    adds into it."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
dt = np.dtype({dtn!r})
for op, cfgs in {CONFIGS!r}.items():
    St, Out, A, G, T, P, _, adf = OPS[op]
    for i, over in enumerate(cfgs):
        g = G(**over)
        X = A(g)
        tr = dev(T(g, 71 + i, dt, hybrid=bool(i % 2)))
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


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x) - a TL dx|| / ||a TL dx|| -> 0 as O(a)."""
    out = _run(backend, f'''
for op, cfgs in {CONFIGS!r}.items():
    St, Out, A, G, T, P = OPS[op][:6]
    for i, over in enumerate(cfgs):
        g = G(**over)
        X = A(g)
        tr = dev(T(g, 31 + i, np.float64, hybrid=bool(i % 2)))
        p = dev(P(g, 300 + i, np.float64))
        def model(xs):
            t = dict(tr); t.update(dict(zip(X, xs)))
            st, _ = run_tl(op, g, t, {{n: cp.zeros_like(t[n]) for n in X}}, True)
            return [st[n] for n in Out]
        def tl(dx):
            return [run_tl(op, g, tr, dict(zip(X, dx)), False)[1][n] for n in Out]
        out[f"{{op}}/{{i}}"] = K.tl_residual(model, tl, [tr[n] for n in X],
                                             [p[n] for n in X], [1e-3, 1e-5, 1e-7])
''', tmp_path)
    assert len(out) == sum(len(c) for c in CONFIGS.values())
    for key, rows in out.items():
        misfits = [r for _, r in rows]
        assert misfits[-1] < 1e-6, (key, rows)
        assert misfits[0] < 0.1, (key, rows)


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_float_base_state_is_the_strict_primal(backend, tmp_path):
    """The float build's base state gives woof's strict words: the update
    against moist._advance_species_wrf's non-PD branch (with the held
    boundary tendency where the config has one), sumflux against the
    dycore's zero/accumulate/finish launches.  Each case also checks the
    update moves the species, so a no-op on both sides cannot pass.
    Host and CUDA; Metal is not checked here."""
    out = _run(backend, f'''
from types import SimpleNamespace as NS
from woof.core import moist as M, dycore as D
f32 = np.float32
def words(a, b):
    a, b = cp.asnumpy(a), cp.asnumpy(b)
    return int((a.view(np.uint32) != b.view(np.uint32)).sum())
for i, over in enumerate({CONFIGS["su"]!r}):
    dx, dy = 3000.0, 2500.0
    g = C.su_geometry(rdx=float(f32(1.0 / dx)), rdy=float(f32(1.0 / dy)), **over)
    tr = dev(C.su_trajectory(g, 41 + i, f32, hybrid=bool(i % 2)))
    st, _ = run_tl("su", g, tr, {{n: cp.zeros_like(tr[n]) for n in S.su_active(g)}}, True)
    pool = {{}}
    def scratch(shape, name):
        return pool.setdefault(name, cp.zeros(shape, f32))
    state = NS(scratch=scratch, msft=tr["msft"], has_msf=bool(g.has_msf), c1h=tr["c1h"],
               c2h=tr["c2h"], rdnw=tr["rdnw"], fnm=tr["fnm"], fnp=tr["fnp"])
    cfg = NS(specified=bool(g.spec), nested=False, spec_zone=g.spec_zone, dx=dx, dy=dy,
             cu_physics=0, v_sca_adv_order=g.vorder, h_sca_adv_order=g.horder)
    mu0, mu = tr["mub2d"] + tr["mup0"], tr["mub2d"] + tr["mup"]
    q = tr["q"].copy()
    M._advance_species_wrf(
        state, cfg, "qc", q, tr["q0"], cp.zeros_like(q), tr["held"] if g.held else None,
        ru=tr["ru_m"], rv=tr["rv_m"], ww=tr["ww_m"], implicit=None, pd=False, bufs=None,
        mu0=mu0, mu=mu, mu0_row=mu0.reshape(-1), mu_row=mu.reshape(-1), chm0=None,
        dt_eff=g.dt, final=False, physics_tendencies=None, fixed_tendencies=None,
        recompute=False, apply_scalar_lbc=None, export_advective_forcing=False,
        boundary_x=bool(g.open_x), boundary_y=bool(g.open_y))
    out[f"su/{{i}}"] = words(st["q"], q)
    out[f"su/{{i}}/moves"] = words(st["q"], tr["q"])
for i, over in enumerate({CONFIGS["sf"]!r}):
    g = C.sf_geometry(**over)
    tr = dev(C.sf_trajectory(g, 61 + i, f32))
    st, _ = run_tl("sf", g, tr, {{n: cp.zeros_like(tr[n]) for n in S.SUMFLUX_SRC + S.SUMFLUX_REF}}, True)
    ref = tuple(tr[n].copy() for n in S.SUMFLUX)
    D._sumflux_launch("zero_sumflux", ref)
    acc = D._prepare_sumflux_launch("accumulate_sumflux", ref, tuple(tr[n] for n in S.SUMFLUX_SRC))
    for _ in range(g.nsub):
        acc()
    D._sumflux_launch("finish_sumflux", ref, tuple(tr[n] for n in S.SUMFLUX_REF), g.nsub)
    out[f"sf/{{i}}"] = sum(words(st[n], r) for n, r in zip(S.SUMFLUX, ref))
''', tmp_path, extra_env=STRICT_ENV)
    assert len(out) == 2 * len(CONFIGS["su"]) + len(CONFIGS["sf"])
    bad = {k: v for k, v in out.items() if (v == 0 if k.endswith("/moves") else v)}
    assert not bad, bad


def test_geometry_rejects_unported_options():
    """A ring that does not fit, a specified domain with a periodic axis, flags outside 0/1, a
    non-positive or non-finite dt, an unported advection order and empty
    axes are refused, not run."""
    from woof.grad.wrfplus import cases as C
    for over in (dict(spec=1, spec_zone=0), dict(spec=1, spec_zone=5),
                 dict(spec_zone=1), dict(has_msf=2), dict(horder=2), dict(nx=0),
                 dict(open_x=2), dict(spec=1, open_y=0), dict(dt=0.0), dict(dt=float("nan"))):
        with pytest.raises(ValueError):
            C.su_geometry(**over).validate()
    for over in (dict(nsub=0), dict(nz=0)):
        with pytest.raises(ValueError):
            C.sf_geometry(**over).validate()
