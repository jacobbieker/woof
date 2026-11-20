"""One dry model step's chained tangent-linear and adjoint (P3a item 8,
woof.grad.wrfplus.model.step_tl/_ad): the three RK stages with their slow
tendencies, acoustic substeps and the step epilogue, in woof's strict order.

Breakage this prevents:
- a step chain that drifts from the strict primal (an operator out of
  order, a scalar rounded otherwise than its launcher, a scratch carrier
  read under the wrong name): the binary32 base must reproduce
  dycore.step's words;
- an adjoint that linearises an operator about the wrong entry state or
  drops the adjoint of a copy (time-t save, muts, the periodic alias);
- a TL that is not the derivative of the step, alone or chained.

Each operator's own dot, finite-difference and WRFPLUS checks are in the
other test_grad_wrfplus_*.py files.  Each probe runs in a child process
on a small periodic case with a ridge (3-D base state).
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
NO_MLX = [b for b in BACKENDS if b.values[0] != "mlx"]

STRICT = dict(GPUWM_WRF_EXACT="1", WOOF_WRF_EXACT_DIAGNOSTICS="1",
              WOOF_WRF_EXACT_ADVECTION="1", GPUWM_WRF_EXACT_DIFFUSION="1",
              GPUWM_WRF_EXACT_BIGSTEP="1")

# The harness case and the step's option branches: hybrid coordinate,
# the w_damp limiter, hypsometric_opt 2 without a lid, other orders, and
# more substeps (nsub 3 and 6 against the default 2 and 4) with another
# divergence damping.
VARIANTS = [dict(), dict(hybrid_opt=2), dict(w_damping=1),
            dict(hypsometric_opt=2, top_lid=False),
            dict(h_mom_adv_order=3, h_sca_adv_order=4, v_mom_adv_order=5),
            dict(time_step_sound=6, smdiv=0.2)]
# Chains of three steps: from a warm state (the previous step's diagnosis
# reused on rk_step 1) and from a cold one (diagnosed on the first step
# only), each followed by warm steps.
CHAINS = [(dict(), 3, 2), (dict(), 3, 0)]


def _run(backend, body, tmp_path):
    env = dict(os.environ, WOOF_BACKEND=backend, PYTHONPATH=str(ROOT),
               WOOF_HOST_CACHE=str(tmp_path / "hostcache"), WOOF_HOST_THREADS="4", **STRICT)
    prelude = (
        "import dataclasses, json, numpy as np, woof\nimport cupy as cp\n"
        "from tilestream import harness as H\n"
        "if cp.__name__ != 'cupy' or getattr(cp, '__woof_facade__', False):\n"
        "    # the host and Metal facades report no card memory; admit the\n"
        "    # 16x12x10 harness state (woof.core.resident_admission)\n"
        "    import woof.core.resident_admission as RA\n"
        "    RA.device_free_bytes = lambda: None\n"
        "from woof.core import dycore as D\n"
        "from woof.grad.wrfplus import model as M, _checks as K\n"
        "out = {}\n"
        "IN = M.STEP_INPUTS\n"
        "def build(over, warm=2, nx=16, ny=12, nz=10, seed=1, hill=400.0):\n"
        "    o = dict(h_sca_adv_order=5, terrain_opt=1); o.update(over)\n"
        "    cfg = H.make_config(nx, ny, nz, **o)\n"
        "    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')\n"
        "    ridge = hill * np.exp(-(((ii - nx / 2) / (nx / 5)) ** 2\n"
        "                           + ((jj - ny / 2) / (ny / 4)) ** 2))\n"
        "    geo = dataclasses.replace(H.neutral_geography(cfg), terrain=ridge)\n"
        "    state = H.make_state(cfg, seed, geography=geo)\n"
        "    for _ in range(warm):\n"
        "        D.step(state, cfg)\n"
        "    return cfg, state\n"
        "def copy(d):\n"
        "    return {k: v.copy() for k, v in d.items()}\n"
        "def zeros(x):\n"
        "    return {k: cp.zeros_like(v) for k, v in x.items()}\n"
        "def pert(x0, seed):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    return [cp.asarray(rng.standard_normal(x0[n].shape)).astype(x0[n].dtype)\n"
        "            * float(cp.abs(x0[n]).max() * 1e-3 + 1e-12) for n in IN]\n"
        "def run_tl(x0, dxs, geoms, tapes=None):\n"
        "    x, d = copy(x0), zeros(x0)\n"
        "    for n, a in zip(IN, dxs):\n"
        "        d[n][...] = a\n"
        "    for k, g in enumerate(geoms):\n"
        "        M.step_tl(x, d, g, None if tapes is None else tapes[k])\n"
        "    return x, [d[n] for n in IN]\n"
        "def run_ad(x0, dys, geoms):\n"
        "    tapes = [[] for _ in geoms]\n"
        "    x, _ = run_tl(x0, [cp.zeros_like(x0[n]) for n in IN], geoms, tapes)\n"
        "    a = zeros(x0)\n"
        "    for n, v in zip(IN, dys):\n"
        "        a[n][...] = v\n"
        "    for k in reversed(range(len(geoms))):\n"
        "        x = M.step_ad(a, x, tapes[k], geoms[k])\n"
        "    return [a[n] for n in IN]\n"
        "# the identity taken out of the prognostics only: the diagnosis\n"
        "# inputs are dead when rk_step 1 reuses them, and an identity there\n"
        "# would only inflate the float32 scale\n"
        "PROG = [n in M.PROGNOSTICS for n in IN]\n")
    script = prelude + body + "\nprint('RESULT ' + json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


@pytest.mark.parametrize("backend", BACKENDS)
def test_step_reproduces_strict_primal_words(backend, tmp_path):
    """The binary32 base of step_tl is dycore.step's strict step word for
    word, on every prognostic, the carried diagnosis and the scratch the
    next step reads: one step per variant and the chains of three."""
    cases = [(v, 1, 2) for v in VARIANTS] + CHAINS
    out = _run(backend, f'''
for i, (over, nstep, warm) in enumerate({cases!r}):
    cfg, state = build(over, warm)
    geoms = M.step_geometries(state, cfg, nstep)
    x = M.run_state(state, cfg, np.float32)
    dx = zeros(x)
    for g in geoms:
        M.step_tl(x, dx, g)
        D.step(state, cfg)
    y = M.run_state(state, cfg, np.float32)
    for n in IN + ("w0", "u_pp", "mu_pp", "ww_ref"):
        a, b = cp.asnumpy(x[n]), cp.asnumpy(y[n])
        out[f"{{i}}/{{n}}"] = int((a.view(np.uint32) != b.view(np.uint32)).sum())
''', tmp_path)
    assert len(out) == len(cases) * (len(("u", "v", "w", "thp", "php", "mup", "p", "al",
                                              "alt", "p_pert", "muts")) + 4)
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


@pytest.mark.parametrize("backend", BACKENDS)
def test_step_dot_product(backend, tmp_path):
    """<(T - I) dx, dy> == <dx, (T - I)^T dy> over the step's inputs: 1e-12
    in binary64, 1e-5 in binary32; one step per variant and the chains of
    three (I on the prognostics only)."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    cases = [(v, 1, 2) for v in VARIANTS] + CHAINS
    out = _run(backend, f'''
for dtn in {dtypes!r}:
    for i, (over, nstep, warm) in enumerate({cases!r}):
        cfg, state = build(over, warm)
        geoms = M.step_geometries(state, cfg, nstep)
        x0 = M.run_state(state, cfg, np.dtype(dtn))
        p, q = pert(x0, 10 + i), pert(x0, 20 + i)
        def tl(dx):
            return [y - d if k else y for y, d, k in zip(run_tl(x0, dx, geoms)[1], dx, PROG)]
        def ad(dy):
            return [y - d if k else y for y, d, k in zip(run_ad(x0, dy, geoms), dy, PROG)]
        out[f"{{dtn}}/{{i}}"] = K.dot_product(tl, ad, p, q, terms=dtn == "float32")
''', tmp_path)
    assert len(out) == len(cases) * (1 if backend == "mlx" else 2)
    for key, rel in out.items():
        limit = 1e-12 if key.startswith("float64") else 1e-5
        assert rel < limit, (key, rel)


@pytest.mark.parametrize("backend", BACKENDS)
def test_base_only_step_is_the_joint_base(backend, tmp_path):
    """step_base (the base-only kernel build) advances the base to the
    joint TL build's words and tapes the same snapshots, so an adjoint
    over its tape is the adjoint over the TL's: binary64 and binary32,
    one step per variant and the chains of three."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    cases = [(v, 1, 2) for v in VARIANTS] + CHAINS
    out = _run(backend, f'''
for dtn in {dtypes!r}:
    for i, (over, nstep, warm) in enumerate({cases!r}):
        cfg, state = build(over, warm)
        geoms = M.step_geometries(state, cfg, nstep)
        x0 = M.run_state(state, cfg, np.dtype(dtn))
        ta, tb = [[] for _ in geoms], [[] for _ in geoms]
        xa, _ = run_tl(x0, pert(x0, 40 + i), geoms, ta)
        xb, scr = copy(x0), None
        for g, tape in zip(geoms, tb):
            scr = M.step_base(xb, g, tape, scr)
        words = lambda a: cp.asnumpy(a).view(np.uint64 if dtn == "float64" else np.uint32)
        bad = sum(int((words(xa[n]) != words(xb[n])).sum()) for n in xa)
        for sa, sb in zip(ta, tb):
            assert len(sa) == len(sb)
            for da, db in zip(sa, sb):
                assert sorted(da) == sorted(db)
                bad += sum(int((words(da[n]) != words(db[n])).sum()) for n in da)
        out[f"{{dtn}}/{{i}}"] = bad
''', tmp_path)
    assert len(out) == len(cases) * (1 if backend == "mlx" else 2)
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


@pytest.mark.parametrize("backend", NO_MLX)
def test_step_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x)|| / ||a TL dx|| -> 1 in binary64, for one step
    per variant and the chains of three."""
    cases = [(v, 1, 2) for v in VARIANTS] + CHAINS
    out = _run(backend, f'''
for i, (over, nstep, warm) in enumerate({cases!r}):
    cfg, state = build(over, warm)
    geoms = M.step_geometries(state, cfg, nstep)
    x0 = M.run_state(state, cfg, np.float64)
    p = pert(x0, 30 + i)
    def model(xs):
        t = copy(x0)
        for n, a in zip(IN, xs):
            t[n][...] = a
        x = run_tl(t, [cp.zeros_like(t[n]) for n in IN], geoms)[0]
        return [x[n] for n in IN]
    out[str(i)] = K.tl_convergence(model, lambda dx: run_tl(x0, dx, geoms)[1],
                                   [x0[n] for n in IN], p, [1e-1, 1e-3, 1e-5])
''', tmp_path)
    assert len(out) == len(cases)
    for key, rows in out.items():
        misfits = [abs(r - 1.0) for _, r in rows]
        assert min(misfits) < 1e-6, (key, rows)
        assert misfits[0] < 1e-2, (key, rows)


@pytest.mark.parametrize("backend", NO_MLX[:1])
def test_step_refuses_what_it_does_not_cover(backend, tmp_path):
    """A flat base (1-D columns), open boundaries, map factors and rotation,
    physics or an odd time_step_sound raise rather than return a step that
    is not the strict primal's."""
    out = _run(backend, '''
cases = {"flat": dict(terrain_opt=0), "open": dict(open_x=True, terrain_opt=1),
         "earth": dict(terrain_opt=1), "pbl": dict(terrain_opt=1, bl_pbl_physics=1),
         "odd": dict(terrain_opt=1, time_step_sound=5)}
for key, over in cases.items():
    cfg = H.make_config(8, 6, 6, h_sca_adv_order=5, **over)
    geo = None
    if over.get("terrain_opt"):
        geo = H.make_geography(cfg) if key == "earth" else H.neutral_geography(cfg)
    try:
        M.step_geometry(H.make_state(cfg, 1, geography=geo), cfg)
        out[key] = "accepted"
    except NotImplementedError as e:
        out[key] = str(e)
''', tmp_path)
    assert "1-D base" in out["flat"], out
    assert "lateral boundaries" in out["open"], out
    assert "map factors" in out["earth"] and "rotation" in out["earth"], out
    assert "physics" in out["pbl"], out
    assert "odd time_step_sound" in out["odd"], out


def test_forecast_path_does_not_import_the_step_adjoint():
    """Opt-in: woof's step path imports nothing from woof.grad."""
    code = ("import sys, woof.core.dycore, woof.core.acoustic, woof.runtime\n"
            "bad = sorted(m for m in sys.modules if m.startswith('woof.grad'))\n"
            "assert not bad, bad\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
