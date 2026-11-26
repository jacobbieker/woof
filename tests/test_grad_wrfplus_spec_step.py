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

# (warm-up primal steps, chained steps, map factors, halo shift, table
# times, moist): a warm step, a cold start chained over three steps, a
# warm chain with map factors and the carried halo moved off the boundary
# ring, a cold chain whose last step ends on its (last) interval's end,
# which the primal serves from that interval, and moist states (qv, qc and
# qr with qv's tables: qv's held capture and spec_bdy_final, flow_dep_bdy
# for qc and qr), warm and as a cold mapped chain.
HOUR = (0.0, 3600.0)
CASES = [(2, 1, 0, 0.0, HOUR, 0), (0, 3, 0, 0.0, HOUR, 0), (2, 2, 1, 3.0, HOUR, 0),
         (0, 2, 0, 0.0, (0.0, 6.0), 0), (2, 1, 0, 0.0, HOUR, 1), (0, 2, 1, 3.0, HOUR, 1)]


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
        "          times=(0.0, 3600.0), attach=True, moist=0, drop=(), **over):\n"
        "    if moist:\n"
        "        over.update(moist=True, moist_adv_opt=0)\n"
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
        "    def make():\n"
        "        s = H.make_state(cfg, 1, geography=geo)\n"
        "        if moist:  # the harness's qv and qc come out zero here: seed them, and qr\n"
        "            rng = np.random.default_rng(8)\n"
        "            k = np.arange(s.qv.shape[0], dtype=np.float64)[:, None, None]\n"
        "            nzq = s.qv.shape[0]\n"
        "            s.qv[...] = cp.asarray(0.014 * np.exp(-3.0 * k / nzq) * (1.0 + 0.2 * rng.standard_normal(\n"
        "                s.qv.shape)), dtype=s.qv.dtype)\n"
        "            s.qc[...] = cp.asarray(np.maximum(4e-4 * np.exp(-((k - nzq / 2) / (nzq / 4)) ** 2)\n"
        "                * (1.0 + 0.3 * rng.standard_normal(s.qc.shape)), 0.0), dtype=s.qc.dtype)\n"
        "            s.qr[...] = cp.asarray(np.maximum(2e-4 * rng.standard_normal(\n"
        "                s.qr.shape), 0.0), dtype=s.qr.dtype)\n"
        "        return s\n"
        "    state = make()\n"
        "    if attach:\n"
        "        snaps = []\n"
        "        for n in range(len(times)):\n"
        "            # boundary states moving away from the start, every field\n"
        "            # (mu and phi too) and varying along the ring, so each table\n"
        "            # and tendency and the relaxation's Laplacian carry signal\n"
        "            f = make()\n"
        "            a = 1.0 + n\n"
        "            def wave(shape, k):\n"
        "                j = np.arange(shape[-2])[:, None]; i = np.arange(shape[-1])[None, :]\n"
        "                return cp.asarray(np.cos(2 * np.pi * (k * i / shape[-1] + j / shape[-2]))\n"
        "                                  .astype(np.float32))\n"
        "            f.u += cp.float32(2.0 * a) + cp.float32(0.5 * a) * wave(f.u.shape, 1)\n"
        "            f.v += cp.float32(1.0 * a) + cp.float32(0.5 * a) * wave(f.v.shape, 2)\n"
        "            f.thp += cp.float32(1.0 * a) + cp.float32(0.3 * a) * wave(f.thp.shape, 1)\n"
        "            f.mup += cp.float32(20.0 * a) * (1 + wave(f.mup.shape, 2))\n"
        "            f.php += cp.float32(5.0 * a) * wave(f.php.shape, 3)\n"
        "            if moist:\n"
        "                f.qv *= cp.float32(1.0 + 0.05 * a) + cp.float32(0.02 * a) * wave(f.qv.shape, 2)\n"
        "            snaps.append({k: a for k, a in LBC.domain_boundary_snapshot(f).items()\n"
        "                          if k not in drop})  # drop: fields without a table\n"
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
        "def scale(x0, n):\n"
        "    # 1e-3 of the field; a table's tendency 1e-3 of its value over an\n"
        "    # hour, so no table starts at roundoff scale\n"
        "    m = float(cp.abs(x0[n]).max())\n"
        "    if n.startswith('lbc_') and n.endswith('_t'):\n"
        "        m = max(m, float(cp.abs(x0[n[:-2]]).max()) / 3600.0)\n"
        "    return 1e-3 * m + 1e-12\n"
        "def pert(x0, names, seed):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    return [cp.asarray(rng.standard_normal(x0[n].shape)).astype(x0[n].dtype)\n"
        "            * scale(x0, n) for n in names]\n"
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
    next step reads; and dropping any part of the binding (the fold,
    spec_bdy_final, advance_specified_phi_w, the stage rows, the halo
    operand with the halo off the ring) moves words, so the cases
    exercise each."""
    out = _run(backend, f'''
for i, (warm, nstep, msf, shift, times, moist) in enumerate({CASES!r}):
    cfg, state = build(warm, msf, shift, times=times, moist=moist)
    geoms = M.step_geometries(state, cfg, nstep)
    IN = M.step_inputs(geoms[0])
    x = M.run_state(state, cfg, np.float32)
    dx = zeros(x)
    for g in geoms:
        M.step_tl(x, dx, g)
        D.step(state, cfg)
    y = M.run_state(state, cfg, np.float32)
    for n in IN + ("w0", "u_pp", "mu_pp", "ww_ref") + (("ru_m", "ww_m") if moist else ()):
        a, b = cp.asnumpy(x[n]), cp.asnumpy(y[n])
        out[f"{{i}}/{{n}}"] = int((a.view(np.uint32) != b.view(np.uint32)).sum())
# each part of the binding dropped in turn must move words
R = dataclasses.replace
def drop(g, part):
    if part == "fold":
        return R(g, fold=())
    if part == "final":
        return R(g, final=())
    if part == "sph":
        return R(g, stages=tuple(R(s, sph=None) for s in g.stages))
    if part == "rows":
        return R(g, stages=tuple(R(s, lbc=()) for s in g.stages))
    if part == "qv_final":
        return R(g, final=tuple(f for f in g.final if f[0] != "qv"))
    if part == "qv_capture":   # the held tendency left at zero
        return R(g, stages=tuple(R(s, qv_capture=None) for s in g.stages))
    if part == "flow":
        return R(g, stages=tuple(R(s, flow=None) for s in g.stages))
    s0 = g.stages[0]
    return R(g, stages=(R(s0, rsp=R(s0.rsp, halo=0)),) + g.stages[1:])
for part in ("fold", "final", "sph", "rows", "halo", "qv_final", "qv_capture", "flow"):
    cfg, state = build(2, 0, 3.0, moist=int(part in ("qv_final", "qv_capture", "flow")))
    g = drop(M.step_geometries(state, cfg, 1)[0], part)
    x = M.run_state(state, cfg, np.float32)
    M.step_tl(x, zeros(x), g)
    D.step(state, cfg)
    y = M.run_state(state, cfg, np.float32)
    out["drop/" + part] = sum(int((cp.asnumpy(x[n]).view(np.uint32)
                                   != cp.asnumpy(y[n]).view(np.uint32)).sum())
                              for n in M.step_inputs(g))
''', tmp_path)
    drops = {k: out.pop(k) for k in list(out) if k.startswith("drop/")}
    assert len(drops) == 8 and all(drops.values()), drops
    # the step inputs (the 11 periodic ones and the halo) and 4 scratch,
    # and on a moist state the species and two sumflux accumulators
    assert len(out) == len(CASES) * (11 + 1 + 4) + sum(c[5] for c in CASES) * (3 + 2)
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


@pytest.mark.parametrize("backend", BACKENDS)
def test_specified_step_dot_product(backend, tmp_path):
    """<(T - I) dx, dy> == <dx, (T - I)^T dy>, dx over the step's inputs
    (the halo included) and the boundary tables, dy over the inputs:
    1e-12 in binary64, 1e-5 (terms-scaled) in binary32; and in binary64
    table by table, each table's share nonzero."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f'''
for dtn in {dtypes!r}:
    for i, (warm, nstep, msf, shift, times, moist) in enumerate({CASES!r}):
        cfg, state = build(warm, msf, shift, times=times, moist=moist)
        geoms = M.step_geometries(state, cfg, nstep)
        IN, PAR = M.step_inputs(geoms[0]), M.step_parameters(geoms[0])
        assert PAR and M.HALO in IN
        x0 = M.run_state(state, cfg, np.dtype(dtn))
        ALL = IN + PAR
        PROG = [n in M.PROGNOSTICS + M.SPECIES for n in IN]
        p, q = pert(x0, ALL, 10 + i), pert(x0, IN, 20 + i)
        def tl(dx):
            d = run_tl(x0, ALL, dx, geoms)[1]
            return [d[n] - a if k else d[n] for n, a, k in zip(IN, dx, PROG)]
        def ad(dy):
            a = run_ad(x0, ALL, IN, dy, geoms)
            return ([a[n] - b if k else a[n] for n, b, k in zip(IN, dy, PROG)]
                    + [a[n] for n in PAR])
        out[f"{{dtn}}/{{i}}"] = K.dot_product(tl, ad, p, q, terms=dtn == "float32")
        if dtn == "float64" and i == 2:
            # table by table (values and tendencies): <TL e_T, q> ==
            # <e_T, (AD q)_T>, each with signal, so no table's share
            # hides under another's
            a = run_ad(x0, ALL, IN, q, geoms)
            for n, e in zip(ALL, p):
                if n not in PAR:
                    continue
                d = run_tl(x0, [n], [e], geoms)[1]
                out[f"table/{{n}}"] = [K.inner([d[m] for m in IN], q), K.inner([e], [a[n]])]
''', tmp_path)
    tables = {k: out.pop(k) for k in list(out) if k.startswith("table/")}
    assert len(out) == len(CASES) * (1 if backend == "mlx" else 2)
    for key, rel in out.items():
        limit = 1e-12 if key.startswith("float64") else 1e-5
        assert rel < limit, (key, rel)
    if backend != "mlx":
        assert len(tables) == 5 * 4 * 2, sorted(tables)
        for key, (lhs, rhs) in tables.items():
            assert lhs != 0.0, (key, lhs)   # the table reaches the outputs
            assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), abs(rhs)), (key, lhs, rhs)


@pytest.mark.parametrize("backend", BACKENDS)
def test_specified_base_only_step_is_the_joint_base(backend, tmp_path):
    """step_base advances the specified base to the joint TL build's words
    and tapes the same snapshots."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f'''
for dtn in {dtypes!r}:
    for i, (warm, nstep, msf, shift, times, moist) in enumerate({CASES!r}):
        cfg, state = build(warm, msf, shift, times=times, moist=moist)
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
for i, (warm, nstep, msf, shift, times, moist) in enumerate({CASES!r}):
    cfg, state = build(warm, msf, shift, times=times, moist=moist)
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
    words (checkpoints every 1 and 3 steps, and a moist state), and sum_k <TL dx_k, f_k> ==
    <dx_0, AD f> with dx_0 over the inputs and the tables and the tables'
    adjoint returned with state 0's (1e-12 binary64, 1e-5 binary32);
    WRFDA's start with ``specified`` seeds the mu halo from g_mu, which
    the first step reads when a halo is carried, and its adjoint pair
    passes the dot test."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f'''
N = 4
for every, moist in ((1, 0), (3, 0), (3, 1)):
    cfg, state = build(1, moist=moist)
    w = W.Window(state, cfg, N, dtype=np.float32, checkpoint_every=every)
    xn = w.forward()
    for _ in range(N):
        D.step(state, cfg)
    y = M.run_state(state, cfg, np.float32)
    out[f"fwd/{{every}}/{{moist}}"] = sum(int((cp.asnumpy(xn[n]).view(np.uint32)
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
                * scale(x0, n) for n in names}}
    def rel(xs, fs, ys, gs):
        l, r = K.inner(xs, fs), K.inner(ys, gs)
        if dtn == "float32":
            m = lambda a, b: K.inner([cp.abs(t) for t in a], [cp.abs(t) for t in b])
            return abs(l - r) / max(abs(l), m(xs, fs), m(ys, gs))
        return abs(l - r) / max(abs(l), abs(r))
    dx0 = rnd(ALL)
    f = {{k: rnd(w.inputs) for k in (1, 3, N)}}
    tls = w.tl(dx0, save_at=(1, 3))
    lhs = [tls[k][n] for k in f for n in w.inputs]
    fs = [f[k][n] for k in f for n in w.inputs]
    a = w.ad(forcing=f)
    assert sorted(a) == sorted(ALL), sorted(a)
    out[f"dot/{{dtn}}"] = rel(lhs, fs, [dx0[n] for n in ALL], [a[n] for n in ALL])
    out[f"tables/{{dtn}}"] = min(float(cp.abs(a[n]).max()) for n in w.params)
    # var4d_lbc = .false.: the tables left out of dx0 start at zero
    st = {{n: dx0[n] for n in w.inputs}}
    t1, t2 = w.tl(st), w.tl({{**st, **{{n: cp.zeros_like(x0[n]) for n in w.params}}}})
    out[f"omitted/{{dtn}}"] = sum(int((cp.asnumpy(t1[N][n]) != cp.asnumpy(t2[N][n])).sum())
                                  for n in w.inputs)
    # var4d_lbc: the tables from the interval's start and end perturbations,
    # through the window: <TL(st, from_ends(s0, s1))_N, f_N> ==
    # <(st, s0, s1), (AD f_N, from_ends_ad(AD f_N))>
    vals = [n for n in w.params if not n.endswith("_t")]
    s0, s1 = rnd(vals), rnd(vals)
    tab = W.lbc_from_ends(s0, s1, w.lbc_seconds)
    tle = w.tl({{**st, **tab}})
    aN = w.ad(forcing={{N: f[N]}})
    sa, ea = W.lbc_from_ends_ad(aN, vals, w.lbc_seconds)
    out[f"ends/{{dtn}}"] = rel(
        [tle[N][n] for n in w.inputs], [f[N][n] for n in w.inputs],
        [st[n] for n in w.inputs] + [s0[n] for n in vals] + [s1[n] for n in vals],
        [aN[n] for n in w.inputs] + [sa[n] for n in vals] + [ea[n] for n in vals])
    # the helper pair alone
    sa, ea = W.lbc_from_ends_ad(a, vals, w.lbc_seconds)
    out[f"pair/{{dtn}}"] = rel([tab[n] for n in w.params], [a[n] for n in w.params],
                               [s0[n] for n in vals] + [s1[n] for n in vals],
                               [sa[n] for n in vals] + [ea[n] for n in vals])
    # WRFDA's start on a specified window seeds the mu halo from g_mu
    # (WRFPLUS start_domain's g_set_physical_bc2d): the pair's dot test,
    # and the halo is read here (the carried halo is off the ring), so a
    # zero halo gives another TL
    g = {{n: dx0[n] for n in W.WRFDA_TL_FIELDS}}
    part = W.wrfda_tl_start(g, specified=True)
    out[f"halo_shared/{{dtn}}"] = int(part[M.HALO] is part["mup"])
    tlw = w.tl(part, partial=True)
    ga = W.wrfda_tl_start_ad(aN, specified=True)
    out[f"wrfda/{{dtn}}"] = rel([tlw[N][n] for n in w.inputs], [f[N][n] for n in w.inputs],
                                [g[n] for n in g], [ga[n] for n in g])
    tl0 = w.tl(W.wrfda_tl_start(g), partial=True)
    out[f"halo_read/{{dtn}}"] = sum(int((cp.asnumpy(tl0[N][n]) != cp.asnumpy(tlw[N][n])).sum())
                                    for n in w.inputs)
''', tmp_path)
    assert out["fwd/1/0"] == 0 and out["fwd/3/0"] == 0 and out["fwd/3/1"] == 0, out
    for dtn in dtypes:
        limit = 1e-12 if dtn == "float64" else 1e-5
        for key in ("dot", "ends", "pair", "wrfda"):
            assert out[f"{key}/{dtn}"] < limit, (key, out)
        assert out[f"halo_shared/{dtn}"] == 1 and out[f"halo_read/{dtn}"] > 0, out
        assert out[f"tables/{dtn}"] > 0, out   # every table's adjoint is nonzero
        assert out[f"omitted/{dtn}"] == 0, out


@pytest.mark.parametrize("backend", BACKENDS)
def test_specified_step_reads_only_its_inputs(backend, tmp_path):
    """The window's checkpoints on a specified domain: every array a step
    writes other than its inputs (the held rows and the fold slots among
    them), and the mu row's unread source slot, poisoned with NaN before
    the second step leave that step's outputs and its adjoint's bit for
    bit the clean run's, and its adjoint leaves nothing on them; the
    tables are never written, and each takes an adjoint."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f'''
for dtn in {dtypes!r}:
    for i, warm in enumerate((2, 0)):
        cfg, state = build(warm, 0, 3.0 if warm else 0.0)
        w = W.Window(state, cfg, 2, dtype=np.dtype(dtn))
        IN = w.inputs
        held = [n for n in w._start if n.startswith(("lbc_held_", "fold_"))]
        out[f"held_written/{{dtn}}/{{i}}"] = int(all(n in w._mutable for n in held))
        out[f"tables_written/{{dtn}}/{{i}}"] = sum(n in w._mutable for n in w.params)
        ad = {{}}
        for poison in (0, 1):
            x = {{n: v.copy() for n, v in w._start.items()}}
            M.step_base(x, w.geoms[0])
            if poison:
                for n in w._mutable:
                    if n not in IN:
                        x[n][...] = np.nan
                # the mu row's source slot: constant, and never read
                x["lbc_mu_src"][...] = np.nan
            tape = []
            M.step_base(x, w.geoms[1], tape)
            ax = {{n: cp.zeros_like(v) for n, v in x.items()}}
            for n in IN:
                ax[n][...] = 1
            M.step_ad(ax, x, tape, w.geoms[1])
            out[f"leak/{{dtn}}/{{i}}/{{poison}}"] = sum(
                int(cp.count_nonzero(ax[n])) for n in w._mutable if n not in IN)
            out[f"tables_ad/{{dtn}}/{{i}}/{{poison}}"] = int(all(
                bool(cp.any(ax[n] != 0)) for n in w.params))
            ad[poison] = ([cp.asnumpy(x[n]) for n in IN] + [cp.asnumpy(ax[n]) for n in IN]
                          + [cp.asnumpy(ax[n]) for n in w.params])
        out[f"{{dtn}}/{{i}}"] = sum(int((a.view(np.uint8) != b.view(np.uint8)).sum())
                                    for a, b in zip(ad[0], ad[1]))
''', tmp_path)
    for key, val in out.items():
        if key.startswith(("held_written/", "tables_ad/")):
            assert val == 1, (key, val)
        else:
            assert val == 0, (key, val)


@pytest.mark.parametrize("backend", BACKENDS[:1])
def test_specified_step_refuses_what_it_does_not_cover(backend, tmp_path):
    """Each configuration the specified binding does not cover is refused
    by name: no tables, nested, spec_zone 0, relax_w, DFI forcing, a bound
    boundary clock, a non-linear time law, streamed seams, and a window
    reaching another interval (one ending on the interval's end is
    accepted when the primal serves that time from the same, last,
    interval, and refused when it serves it from the next); open
    boundaries that are not specified are named apart."""
    out = _run(backend, '''
def refuse(key, fn):
    try:
        fn()
        out[key] = "accepted"
    except NotImplementedError as e:
        out[key] = str(e)
cfg, state = build(0, attach=False)
refuse("bare", lambda: M.step_geometries(state, cfg, 1))
cfg, state = build(0)
for key, kw in (("nested", dict(nested=True)), ("spec_zone", dict(spec_zone=0)),
                ("relax_w", dict(relax_w=True)), ("dfi", dict(dfi_constant_bc=True))):
    refuse(key, lambda: M.step_geometries(state, dataclasses.replace(cfg, **kw), 1))
res = state._lateral_boundary_device
clock, res.clock = res.clock, object()
refuse("clock", lambda: M.step_geometries(state, cfg, 1))
res.clock = clock
side = state.lateral_boundaries.intervals[0].fields["u"].west
law = side.time_law
object.__setattr__(side, "time_law", "cubic")
refuse("law", lambda: M.step_geometries(state, cfg, 1))
object.__setattr__(side, "time_law", law)
bounds = state.lateral_boundaries
seams = getattr(bounds, "seam_sides", ())
object.__setattr__(bounds, "seam_sides", ("west",))
refuse("seams", lambda: M.step_geometries(state, cfg, 1))
object.__setattr__(bounds, "seam_sides", seams)
refuse("restored", lambda: M.step_geometries(state, cfg, 1))
cfg, state = build(0, times=(0.0, 4.0))
refuse("cross", lambda: M.step_geometries(state, cfg, 3))
refuse("inside", lambda: M.step_geometries(state, cfg, 1))
cfg, state = build(0, times=(0.0, 6.0))
refuse("last_end", lambda: M.step_geometries(state, cfg, 2))
cfg, state = build(0, times=(0.0, 6.0, 3600.0))
refuse("next_start", lambda: M.step_geometries(state, cfg, 2))
cfg = H.make_config(8, 6, 6, h_sca_adv_order=5, terrain_opt=1, open_x=True)
refuse("open", lambda: M.step_geometries(H.make_state(cfg, 1, geography=H.neutral_geography(cfg)), cfg, 1))
''', tmp_path)
    assert "without attached tables" in out["bare"], out
    assert "nested lateral boundaries" in out["nested"], out
    assert "spec_zone 0" in out["spec_zone"], out
    assert "relax_w" in out["relax_w"], out
    assert "DFI boundary forcing" in out["dfi"], out
    assert "DomainClock" in out["clock"], out
    assert "time laws other than linear" in out["law"], out
    assert "streamed tile seams" in out["seams"], out
    assert out["restored"] == "accepted", out
    assert "reaching another boundary interval" in out["cross"], out
    assert out["inside"] == "accepted", out
    assert out["last_end"] == "accepted", out
    assert "reaching another boundary interval" in out["next_start"], out
    assert "open lateral boundaries" in out["open"], out
