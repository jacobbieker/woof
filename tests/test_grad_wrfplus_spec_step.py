"""The dry step's TL/AD on a specified domain (P3a, DA item 3; woof.grad.
wrfplus.model with lbc.py): WRF's lateral boundary work in the strict step
(fold_lateral_relaxation, the per-stage specified and relaxation rows,
advance_specified_phi_w after each substep, spec_bdy_final in the epilogue
and the carried mu halo on rk_step 1's boundary faces), with the
interval's boundary tables as step parameters.

Breakage this prevents:
- a specified step that drifts from dycore.step's strict words (a row out
  of order, the fold's map factor coupling, the halo ghost operand on the
  boundary faces; the halo is shifted off the ring so a dropped halo
  moves words);
- an adjoint that drops a table's, the halo's or a held row's share: the
  dot test perturbs the tables and the halo with the state;
- a TL that is not the derivative of the step;
- a window that does not carry the tables as parameters, or a refused
  boundary configuration that is not named.
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

# (warm-up primal steps, chained steps, map factors, halo shift): a warm
# step, a cold start chained over three steps, and a warm chain with map
# factors and the carried halo moved off the boundary ring.
CASES = [(2, 1, 0, 0.0), (0, 3, 0, 0.0), (2, 2, 1, 3.0)]


def _run(backend, body, tmp_path):
    env = dict(os.environ, WOOF_BACKEND=backend, PYTHONPATH=str(ROOT),
               WOOF_HOST_CACHE=str(tmp_path / "hostcache"), WOOF_HOST_THREADS="4", **STRICT)
    prelude = (
        "import dataclasses, json, numpy as np, woof\nimport cupy as cp\n"
        "from tilestream import harness as H\n"
        "if cp.__name__ != 'cupy' or getattr(cp, '__woof_facade__', False):\n"
        "    import woof.core.resident_admission as RA\n"
        "    RA.device_free_bytes = lambda: None\n"
        "from woof.core import dycore as D\n"
        "from woof.grad.wrfplus import model as M, window as W, _checks as K\n"
        "import woof.ingest.lateral_bc as LBC\n"
        "out = {}\n"
        "def build(warm=2, msf=0, shift=0.0, nx=20, ny=18, nz=8, relax=4, hill=400.0,\n"
        "          times=(0.0, 3600.0), attach=True, **over):\n"
        "    cfg = H.make_config(nx, ny, nz, h_sca_adv_order=5, terrain_opt=1, specified=True,\n"
        "                        spec_zone=1, relax_zone=relax, spec_bdy_width=relax + 1, **over)\n"
        "    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')\n"
        "    ridge = hill * np.exp(-(((ii - nx / 2) / (nx / 5)) ** 2\n"
        "                           + ((jj - ny / 2) / (ny / 4)) ** 2))\n"
        "    geo = dataclasses.replace(H.neutral_geography(cfg), terrain=ridge)\n"
        "    if msf:\n"
        "        def m(j, i):\n"
        "            return (1.0 + 0.03 * np.cos(2 * np.pi * j / ny)\n"
        "                    + 0.01 * np.sin(2 * np.pi * i / nx))\n"
        "        jf, iu = np.meshgrid(np.arange(ny), np.arange(nx + 1) - 0.5, indexing='ij')\n"
        "        jv, iv = np.meshgrid(np.arange(ny + 1) - 0.5, np.arange(nx), indexing='ij')\n"
        "        geo = dataclasses.replace(geo, msft=m(jj, ii), msfu=m(jf, iu), msfv=m(jv, iv))\n"
        "    state = H.make_state(cfg, 1, geography=geo)\n"
        "    if attach:\n"
        "        snaps = []\n"
        "        for n in range(len(times)):  # boundary states moving away from the start\n"
        "            f = H.make_state(cfg, 1, geography=geo)\n"
        "            f.u += cp.float32(2.0 + n); f.v += cp.float32(1.0 + n)\n"
        "            f.thp += cp.float32(1.0 + n)\n"
        "            snaps.append(dict(LBC.domain_boundary_snapshot(f)))\n"
        "        LBC.attach_lateral_boundaries(state, LBC.build_lateral_boundaries(\n"
        "            snaps, list(times), spec_bdy_width=cfg.spec_bdy_width,\n"
        "            spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone))\n"
        "    for _ in range(warm):\n"
        "        D.step(state, cfg)\n"
        "    if shift:  # the carried halo off the ring: rk_step 1 must read it\n"
        "        state.scratch(state.mup.shape, D.WRF_MU_HALO_SLOT)[...] += cp.float32(shift)\n"
        "    return cfg, state\n"
        "def copy(d):\n"
        "    return {k: v.copy() for k, v in d.items()}\n"
        "def zeros(x):\n"
        "    return {k: cp.zeros_like(v) for k, v in x.items()}\n"
        "def pert(x0, names, seed):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    return [cp.asarray(rng.standard_normal(x0[n].shape)).astype(x0[n].dtype)\n"
        "            * float(cp.abs(x0[n]).max() * 1e-3 + 1e-12) for n in names]\n"
        "def run_tl(x0, names, dxs, geoms, tapes=None):\n"
        "    x, d = copy(x0), zeros(x0)\n"
        "    for n, a in zip(names, dxs):\n"
        "        d[n][...] = a\n"
        "    for k, g in enumerate(geoms):\n"
        "        M.step_tl(x, d, g, None if tapes is None else tapes[k])\n"
        "    return x, d\n"
        "def run_ad(x0, names, out_names, dys, geoms):\n"
        "    tapes = [[] for _ in geoms]\n"
        "    x, _ = run_tl(x0, names, [cp.zeros_like(x0[n]) for n in names], geoms, tapes)\n"
        "    a = zeros(x0)\n"
        "    for n, v in zip(out_names, dys):\n"
        "        a[n][...] = v\n"
        "    for k in reversed(range(len(geoms))):\n"
        "        x = M.step_ad(a, x, tapes[k], geoms[k])\n"
        "    for n in ('msft', 'msfu', 'msfv', 'msfv_inv', 'lbc_fcx', 'lbc_gcx'):\n"
        "        assert not bool(cp.any(a[n] != 0)), n   # passive\n"
        "    return a\n")
    script = prelude + body + "\nprint('RESULT ' + json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


@pytest.mark.parametrize("backend", BACKENDS)
def test_specified_step_reproduces_strict_primal_words(backend, tmp_path):
    """The binary32 base is dycore.step's strict specified step word for
    word on every input (the carried halo included) and the scratch the
    next step reads; and dropping the halo operand moves words when the
    halo is off the ring (the case exercises it)."""
    out = _run(backend, f'''
for i, (warm, nstep, msf, shift) in enumerate({CASES!r}):
    cfg, state = build(warm, msf, shift)
    geoms = M.step_geometries(state, cfg, nstep)
    IN = M.step_inputs(geoms[0])
    x = M.run_state(state, cfg, np.float32)
    dx = zeros(x)
    for g in geoms:
        M.step_tl(x, dx, g)
        D.step(state, cfg)
    y = M.run_state(state, cfg, np.float32)
    for n in IN + ("w0", "u_pp", "mu_pp", "ww_ref"):
        a, b = cp.asnumpy(x[n]), cp.asnumpy(y[n])
        out[f"{{i}}/{{n}}"] = int((a.view(np.uint32) != b.view(np.uint32)).sum())
cfg, state = build(2, 0, 3.0)
geoms = M.step_geometries(state, cfg, 1)
s0 = geoms[0].stages[0]
g = dataclasses.replace(geoms[0], stages=(dataclasses.replace(
    s0, rsp=dataclasses.replace(s0.rsp, halo=0)),) + geoms[0].stages[1:])
x = M.run_state(state, cfg, np.float32)
M.step_tl(x, zeros(x), g)
D.step(state, cfg)
y = M.run_state(state, cfg, np.float32)
out["nohalo"] = sum(int((cp.asnumpy(x[n]).view(np.uint32)
                         != cp.asnumpy(y[n]).view(np.uint32)).sum()) for n in M.STEP_INPUTS)
''', tmp_path)
    nohalo = out.pop("nohalo")
    assert nohalo > 0, "the halo case does not exercise the halo operand"
    # the step inputs (the 11 periodic ones and the halo) and 4 scratch
    assert len(out) == len(CASES) * (11 + 1 + 4)
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


@pytest.mark.parametrize("backend", BACKENDS)
def test_specified_step_dot_product(backend, tmp_path):
    """<(T - I) dx, dy> == <dx, (T - I)^T dy>, dx over the step's inputs
    (the halo included) and the boundary tables, dy over the inputs:
    1e-12 in binary64, 1e-5 (terms-scaled) in binary32."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f'''
for dtn in {dtypes!r}:
    for i, (warm, nstep, msf, shift) in enumerate({CASES!r}):
        cfg, state = build(warm, msf, shift)
        geoms = M.step_geometries(state, cfg, nstep)
        IN, PAR = M.step_inputs(geoms[0]), M.step_parameters(geoms[0])
        assert PAR and M.HALO in IN
        x0 = M.run_state(state, cfg, np.dtype(dtn))
        ALL = IN + PAR
        PROG = [n in M.PROGNOSTICS for n in IN]
        p, q = pert(x0, ALL, 10 + i), pert(x0, IN, 20 + i)
        def tl(dx):
            d = run_tl(x0, ALL, dx, geoms)[1]
            return [d[n] - a if k else d[n] for n, a, k in zip(IN, dx, PROG)]
        def ad(dy):
            a = run_ad(x0, ALL, IN, dy, geoms)
            return ([a[n] - b if k else a[n] for n, b, k in zip(IN, dy, PROG)]
                    + [a[n] for n in PAR])
        out[f"{{dtn}}/{{i}}"] = K.dot_product(tl, ad, p, q, terms=dtn == "float32")
''', tmp_path)
    assert len(out) == len(CASES) * (1 if backend == "mlx" else 2)
    for key, rel in out.items():
        limit = 1e-12 if key.startswith("float64") else 1e-5
        assert rel < limit, (key, rel)


@pytest.mark.parametrize("backend", BACKENDS)
def test_specified_base_only_step_is_the_joint_base(backend, tmp_path):
    """step_base advances the specified base to the joint TL build's words
    and tapes the same snapshots."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f'''
for dtn in {dtypes!r}:
    for i, (warm, nstep, msf, shift) in enumerate({CASES!r}):
        cfg, state = build(warm, msf, shift)
        geoms = M.step_geometries(state, cfg, nstep)
        ALL = M.step_inputs(geoms[0]) + M.step_parameters(geoms[0])
        x0 = M.run_state(state, cfg, np.dtype(dtn))
        ta, tb = [[] for _ in geoms], [[] for _ in geoms]
        xa, _ = run_tl(x0, ALL, pert(x0, ALL, 40 + i), geoms, ta)
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
    assert len(out) == len(CASES) * (1 if backend == "mlx" else 2)
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


@pytest.mark.parametrize("backend", NO_MLX)
def test_specified_step_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x)|| / ||a TL dx|| -> 1 in binary64, dx over
    the inputs and the boundary tables."""
    out = _run(backend, f'''
for i, (warm, nstep, msf, shift) in enumerate({CASES!r}):
    cfg, state = build(warm, msf, shift)
    geoms = M.step_geometries(state, cfg, nstep)
    IN = M.step_inputs(geoms[0])
    ALL = IN + M.step_parameters(geoms[0])
    x0 = M.run_state(state, cfg, np.float64)
    p = pert(x0, ALL, 30 + i)
    def model(xs):
        t = copy(x0)
        for n, a in zip(ALL, xs):
            t[n][...] = a
        x = run_tl(t, ALL, [cp.zeros_like(t[n]) for n in ALL], geoms)[0]
        return [x[n] for n in IN]
    def tl(dx):
        d = run_tl(x0, ALL, dx, geoms)[1]
        return [d[n] for n in IN]
    out[str(i)] = K.tl_convergence(model, tl, [x0[n] for n in ALL], p, [1e-1, 1e-3, 1e-5])
''', tmp_path)
    assert len(out) == len(CASES)
    for key, rows in out.items():
        misfits = [abs(r - 1.0) for _, r in rows]
        assert min(misfits) < 1e-6, (key, rows)
        assert misfits[0] < 1e-2, (key, rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_specified_window(backend, tmp_path):
    """Window on a specified domain: its binary32 forward is dycore.step's
    words (checkpoints every 1 and 3 steps), and sum_k <TL dx_k, f_k> ==
    <dx_0, AD f> with dx_0 over the inputs and the tables and the tables'
    adjoint returned with state 0's (1e-12 binary64, 1e-5 binary32)."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f'''
N = 4
for every in (1, 3):
    cfg, state = build(1)
    w = W.Window(state, cfg, N, dtype=np.float32, checkpoint_every=every)
    xn = w.forward()
    for _ in range(N):
        D.step(state, cfg)
    y = M.run_state(state, cfg, np.float32)
    out[f"fwd/{{every}}"] = sum(int((cp.asnumpy(xn[n]).view(np.uint32)
                                     != cp.asnumpy(y[n]).view(np.uint32)).sum()) for n in w.inputs)
for dtn in {dtypes!r}:
    cfg, state = build(1, 0, 3.0)
    w = W.Window(state, cfg, N, dtype=np.dtype(dtn), checkpoint_every=3)
    w.forward()
    x0 = w._start
    ALL = w.inputs + w.params
    rng = np.random.default_rng(5)
    def rnd(names):
        return {{n: cp.asarray(rng.standard_normal(x0[n].shape)).astype(x0[n].dtype)
                * float(cp.abs(x0[n]).max() * 1e-3 + 1e-12) for n in names}}
    dx0 = rnd(ALL)
    f = {{k: rnd(w.inputs) for k in (1, 3, N)}}
    tls = w.tl(dx0, save_at=(1, 3))
    lhs = [tls[k][n] for k in f for n in w.inputs]
    fs = [f[k][n] for k in f for n in w.inputs]
    a = w.ad(forcing=f)
    assert sorted(a) == sorted(ALL), sorted(a)
    rhs = [a[n] for n in ALL]
    dxs = [dx0[n] for n in ALL]
    l, r = K.inner(lhs, fs), K.inner(dxs, rhs)
    if dtn == "float32":
        scale = max(abs(l), K.inner([cp.abs(t) for t in lhs], [cp.abs(t) for t in fs]),
                    K.inner([cp.abs(t) for t in dxs], [cp.abs(t) for t in rhs]))
    else:
        scale = max(abs(l), abs(r))
    out[f"dot/{{dtn}}"] = abs(l - r) / scale
    out[f"tables/{{dtn}}"] = float(sum(float(cp.abs(a[n]).max()) for n in w.params))
''', tmp_path)
    assert out["fwd/1"] == 0 and out["fwd/3"] == 0, out
    for dtn in dtypes:
        limit = 1e-12 if dtn == "float64" else 1e-5
        assert out[f"dot/{dtn}"] < limit, out
        assert out[f"tables/{dtn}"] > 0, out


@pytest.mark.parametrize("backend", BACKENDS[:1])
def test_specified_step_refuses_what_it_does_not_cover(backend, tmp_path):
    """Specified without tables, relax_w and a window that
    crosses a boundary interval are refused by name; nested and open
    (not specified) boundaries are named apart."""
    out = _run(backend, '''
def refuse(key, fn):
    try:
        fn()
        out[key] = "accepted"
    except NotImplementedError as e:
        out[key] = str(e)
cfg, state = build(0, attach=False)
refuse("bare", lambda: M.step_geometries(state, cfg, 1))
cfg, state = build(0, relax_w=True)
refuse("relax_w", lambda: M.step_geometries(state, cfg, 1))
cfg, state = build(0, times=(0.0, 4.0))
refuse("cross", lambda: M.step_geometries(state, cfg, 3))
refuse("inside", lambda: M.step_geometries(state, cfg, 1))
cfg = H.make_config(8, 6, 6, h_sca_adv_order=5, terrain_opt=1, open_x=True)
refuse("open", lambda: M.step_geometries(H.make_state(cfg, 1, geography=H.neutral_geography(cfg)), cfg, 1))
''', tmp_path)
    assert "without attached tables" in out["bare"], out
    assert "relax_w" in out["relax_w"], out
    assert "crossing a boundary interval" in out["cross"], out
    assert out["inside"] == "accepted", out
    assert "open lateral boundaries" in out["open"], out
