"""The window driver (P3a item 8, woof.grad.wrfplus.window.Window): the
forward model, TL and adjoint over N strict steps with checkpoints, and
adjoint forcing at intermediate states, the interface 4D-Var uses.

Breakage this prevents:
- a forward that drifts from N primal steps (checkpoints restored wrongly,
  a constant array written through);
- an adjoint whose forcing lands on the wrong state, or whose segment
  recompute from a checkpoint restarts from the wrong state: the sum over
  states of <TL dx_k, f_k> must equal <dx_0, AD(f)>, and the adjoint must
  not depend on the checkpoint interval;
- a refused configuration that is not named.

Each step's own checks are in test_grad_wrfplus_model.py.
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

STRICT = dict(GPUWM_WRF_EXACT="1", WOOF_WRF_EXACT_DIAGNOSTICS="1",
              WOOF_WRF_EXACT_ADVECTION="1", GPUWM_WRF_EXACT_DIFFUSION="1",
              GPUWM_WRF_EXACT_BIGSTEP="1")

# (case, warm-up primal steps): a warm and a cold start, and rotation
# with map factors.
CASES = [(dict(), 1), (dict(), 0), (dict(_rot=1, _msf=1), 1)]
N = 4


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
        "out = {}\n"
        "IN = M.STEP_INPUTS\n"
        "def build(over, warm, nx=16, ny=12, nz=10, seed=1, hill=400.0):\n"
        "    o = dict(h_sca_adv_order=5, terrain_opt=1); o.update(over)\n"
        "    rot = o.pop('_rot', 0)\n"
        "    msf = o.pop('_msf', 0)\n"
        "    cfg = H.make_config(nx, ny, nz, **o)\n"
        "    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')\n"
        "    ridge = hill * np.exp(-(((ii - nx / 2) / (nx / 5)) ** 2\n"
        "                           + ((jj - ny / 2) / (ny / 4)) ** 2))\n"
        "    geo = dataclasses.replace(H.neutral_geography(cfg), terrain=ridge)\n"
        "    if rot:\n"
        "        ang = 0.3 + 0.01 * ii\n"
        "        geo = dataclasses.replace(geo, f=1e-4 + 1e-6 * jj, e=np.full((ny, nx), 5e-5),\n"
        "                                  sina=np.sin(ang), cosa=np.cos(ang))\n"
        "    if msf:  # smooth periodic map factors, a few percent from 1\n"
        "        def m(j, i):\n"
        "            return (1.0 + 0.03 * np.cos(2 * np.pi * j / ny)\n"
        "                    + 0.01 * np.sin(2 * np.pi * i / nx))\n"
        "        jf, iu = np.meshgrid(np.arange(ny), np.arange(nx + 1) - 0.5, indexing='ij')\n"
        "        jv, iv = np.meshgrid(np.arange(ny + 1) - 0.5, np.arange(nx), indexing='ij')\n"
        "        geo = dataclasses.replace(geo, msft=m(jj, ii), msfu=m(jf, iu), msfv=m(jv, iv))\n"
        "    state = H.make_state(cfg, seed, geography=geo)\n"
        "    for _ in range(warm):\n"
        "        D.step(state, cfg)\n"
        "    return cfg, state\n"
        "def rnd(x0, rng):\n"
        "    return {n: cp.asarray(rng.standard_normal(x0[n].shape)).astype(x0[n].dtype)\n"
        "            * float(cp.abs(x0[n]).max() * 1e-3 + 1e-12) for n in IN}\n"
        "def flat(d):\n"
        "    return [d[n] for n in IN]\n")
    script = prelude + body + "\nprint('RESULT ' + json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


@pytest.mark.parametrize("backend", BACKENDS)
def test_window_forward_is_the_primal(backend, tmp_path):
    """The binary32 window's forward is N strict dycore.step calls word for
    word, with checkpoints every 1 or 3 steps."""
    out = _run(backend, f'''
for i, (over, warm) in enumerate({CASES!r}):
    for every in (1, 3):
        cfg, state = build(over, warm)
        w = W.Window(state, cfg, {N}, dtype=np.float32, checkpoint_every=every)
        xn = w.forward()
        for _ in range({N}):
            D.step(state, cfg)
        y = M.run_state(state, cfg, np.float32)
        out[f"{{i}}/{{every}}"] = sum(
            int((cp.asnumpy(xn[n]).view(np.uint32) != cp.asnumpy(y[n]).view(np.uint32)).sum())
            for n in IN)
''', tmp_path)
    assert len(out) == len(CASES) * 2
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


@pytest.mark.parametrize("backend", BACKENDS)
def test_window_adjoint_with_forcing(backend, tmp_path):
    """sum_k <TL(dx_0)_k, f_k> == <dx_0, AD(f)> with forcing at states 0,
    1, 3 and N (1e-12 in binary64, 1e-5 in binary32); the adjoint does
    not depend on the checkpoint interval (1, 3 and 5 > N, bit for bit),
    and the final state's forcing passed as ax_final gives the same."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f"""
for dtn in {dtypes!r}:
    for i, (over, warm) in enumerate({CASES!r}):
        ads = []
        for every in (1, 3, 5):
            cfg, state = build(over, warm)
            w = W.Window(state, cfg, {N}, dtype=np.dtype(dtn), checkpoint_every=every)
            w.forward()
            rng = np.random.default_rng(7 + i)
            x0 = M.run_state(state, cfg, np.dtype(dtn))
            dx0 = rnd(x0, rng)
            f = {{k: rnd(x0, rng) for k in (0, 1, 3, {N})}}
            tl = w.tl(dx0, save_at=f.keys())
            a0 = w.ad(None, f)
            ys = sum((flat(tl[k]) for k in sorted(f)), [])
            fs = sum((flat(f[k]) for k in sorted(f)), [])
            out[f"{{dtn}}/{{i}}/{{every}}"] = K.dot_product(
                lambda _x: ys, lambda _y: flat(a0), flat(dx0), fs,
                terms=dtn == "float32")
            if every == 1:   # the final adjoint passed as ax_final, the same
                a0b = w.ad(f[{N}], {{k: v for k, v in f.items() if k != {N}}})
                out[f"final/{{dtn}}/{{i}}"] = sum(
                    int((cp.asnumpy(a0[n]) != cp.asnumpy(a0b[n])).sum()) for n in IN)
            ads.append(a0)
        out[f"same/{{dtn}}/{{i}}"] = sum(
            int((cp.asnumpy(ads[0][n]) != cp.asnumpy(a[n])).sum()) for a in ads[1:] for n in IN)
""", tmp_path)
    nd = 1 if backend == "mlx" else 2
    assert len(out) == nd * len(CASES) * 5
    for key, val in out.items():
        if key.startswith(("same/", "final/")):
            assert val == 0, (key, val)
        else:
            assert val < (1e-12 if key.startswith("float64") else 1e-5), (key, val)


@pytest.mark.parametrize("backend", BACKENDS)
def test_window_external_trajectory(backend, tmp_path):
    """forward(trajectory=...) linearises step k about the supplied state
    k (DA item 1, WRFPLUS's auxinput6 trajectory):
    - the window's own base states as the trajectory give the same TL and
      adjoint, bit for bit;
    - another run's states (here the primal's binary32 steps from a
      perturbed start, read in the window's dtype) move the TL from the
      state after the first supplied state other than the window's own,
      and not before it;
    - the adjoint with forcing is the TL's (1e-12 binary64, 1e-5
      binary32), and forward returns state N, given or stepped (the own
      states, N of them, end where the own forward does);
    - a plain forward afterwards returns to the own TL and adjoint bit
      for bit, and a refused one, or an external one whose last step
      fails, leaves the external base in place;
    - a trajectory of the wrong length, or with x0, is refused."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f"""
for dtn in {dtypes!r}:
    dt = np.dtype(dtn)
    for i, (over, warm) in enumerate({CASES!r}):
        cfg, state = build(over, warm)
        w = W.Window(state, cfg, {N}, dtype=dt)
        own_end = w.forward()
        own = [w._checkpoints[k] for k in range({N})]
        rng = np.random.default_rng(3 + i)
        x0 = M.run_state(state, cfg, dt)
        dx0 = rnd(x0, rng)
        f = {{k: rnd(x0, rng) for k in (1, 2, {N})}}
        tl_own, ad_own = w.tl(dx0, save_at=(1, 2)), w.ad(None, f)
        end = w.forward(trajectory=own)
        tl_ext, ad_ext = w.tl(dx0, save_at=(1, 2)), w.ad(None, f)
        out[f"same/{{dtn}}/{{i}}"] = (
            sum(int((cp.asnumpy(tl_own[k][n]) != cp.asnumpy(tl_ext[k][n])).sum())
                for k in (1, 2, {N}) for n in IN)
            + sum(int((cp.asnumpy(ad_own[n]) != cp.asnumpy(ad_ext[n])).sum()) for n in IN)
            + sum(int((cp.asnumpy(own_end[n]) != cp.asnumpy(end[n])).sum()) for n in IN))
        # another run: the primal from a perturbed start (its own
        # DomainState); state 0 the window's own, so the TL first moves
        # at state 2
        _, other = build(over, warm)
        other.thp += cp.float32(0.5)
        traj = [own[0]]
        for _ in range({N}):
            D.step(other, cfg)
            traj.append(w.trajectory_state(other, cfg))
        endx = w.forward(trajectory=traj)
        out[f"given_end/{{dtn}}/{{i}}"] = sum(
            int((cp.asnumpy(endx[n]) != cp.asnumpy(traj[{N}][n])).sum()) for n in IN)
        tl = w.tl(dx0, save_at=(1, 2))
        out[f"state1/{{dtn}}/{{i}}"] = sum(
            int((cp.asnumpy(tl[1][n]) != cp.asnumpy(tl_own[1][n])).sum()) for n in IN)
        # moved by more than roundoff: ||dx_2 - own dx_2|| / ||own dx_2||
        out[f"moved/{{dtn}}/{{i}}"] = K.norm([tl[2][n] - tl_own[2][n] for n in IN]) / K.norm(
            [tl_own[2][n] for n in IN])
        a0 = w.ad(None, f)
        ys = sum((flat(tl[k]) for k in sorted(f)), [])
        fs = sum((flat(f[k]) for k in sorted(f)), [])
        out[f"{{dtn}}/{{i}}"] = K.dot_product(lambda _x: ys, lambda _y: flat(a0), flat(dx0), fs,
                                              terms=dtn == "float32")
        # back to the window's own base: the own TL and adjoint, bit for bit
        back_end = w.forward()
        tl_b, ad_b = w.tl(dx0, save_at=(1, 2)), w.ad(None, f)
        out[f"back/{{dtn}}/{{i}}"] = (
            sum(int((cp.asnumpy(tl_own[k][n]) != cp.asnumpy(tl_b[k][n])).sum())
                for k in (1, 2, {N}) for n in IN)
            + sum(int((cp.asnumpy(ad_own[n]) != cp.asnumpy(ad_b[n])).sum()) for n in IN)
            + sum(int((cp.asnumpy(own_end[n]) != cp.asnumpy(back_end[n])).sum()) for n in IN))
        # a refused own forward leaves the external base in place
        w.forward(trajectory=traj)
        try:
            w.forward({{n: v for n, v in x0.items() if n in IN and n != "u"}})
        except ValueError:
            pass
        tl_r = w.tl(dx0, save_at=(1, 2))
        out[f"kept/{{dtn}}/{{i}}"] = sum(
            int((cp.asnumpy(tl_r[k][n]) != cp.asnumpy(tl[k][n])).sum())
            for k in (1, 2, {N}) for n in IN)
        # so does a trajectory whose last step fails (the N-state form
        # steps the last state; here that step raises)
        real = M.step_base
        def boom(*a, **k):
            raise RuntimeError("step failed")
        M.step_base = boom
        try:
            w.forward(trajectory=own)
        except RuntimeError:
            pass
        finally:
            M.step_base = real
        tl_f = w.tl(dx0, save_at=(1, 2))
        out[f"failed/{{dtn}}/{{i}}"] = sum(
            int((cp.asnumpy(tl_f[k][n]) != cp.asnumpy(tl[k][n])).sum())
            for k in (1, 2, {N}) for n in IN)
cfg, state = build({{}}, 1)
w = W.Window(state, cfg, {N})
x = w.trajectory_state(state, cfg)
for key, kw in (("short", dict(trajectory=[x] * ({N} - 1))),
                ("both", dict(x0=x, trajectory=[x] * {N}))):
    try:
        w.forward(**kw)
        out["refuse/" + key] = ""
    except ValueError as e:
        out["refuse/" + key] = str(e)
""", tmp_path)
    assert f"holds {N - 1} states" in out.pop("refuse/short"), out
    assert "not both" in out.pop("refuse/both"), out
    nd = 1 if backend == "mlx" else 2
    assert len(out) == nd * len(CASES) * 8
    for key, val in out.items():
        if key.startswith("moved/"):
            assert val > 1e-4, (key, val)
        elif key.startswith(("same/", "given_end/", "state1/", "back/", "kept/", "failed/")):
            assert val == 0, (key, val)
        else:
            assert val < (1e-12 if key.startswith("float64") else 1e-5), (key, val)


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_window_tl_is_the_derivative_of_forward(backend, tmp_path):
    """Binary64: ||forward(x0 + a dx0) - forward(x0)|| / ||a TL dx0|| -> 1
    at the final state, from a given x0; repeated forward, tl and ad calls
    give the same words and leave the window's start arrays unchanged."""
    out = _run(backend, f"""
for i, (over, warm) in enumerate({CASES!r}):
    cfg, state = build(over, warm)
    w = W.Window(state, cfg, {N}, checkpoint_every=3)
    start = {{n: v.copy() for n, v in w._start.items()}}
    rng = np.random.default_rng(11 + i)
    x0 = {{n: w._start[n] + 1e-4 * v for n, v in rnd(w._start, rng).items()}}
    dx0 = rnd(w._start, rng)
    def model(xs):
        return flat(w.forward(dict(zip(IN, xs))))
    def tl(dxs):
        w.forward(x0)
        return flat(w.tl(dict(zip(IN, dxs)))[{N}])
    out[f"fd/{{i}}"] = K.tl_convergence(model, tl, flat(x0), flat(dx0), [1e-1, 1e-3, 1e-5])
    runs = []
    for _ in range(2):
        xn = w.forward(x0)
        dn = w.tl(dx0, save_at=(2,))
        a0 = w.ad(dx0, {{1: dx0}})
        runs.append([cp.asnumpy(d[n]) for d in (xn, dn[2], dn[{N}], a0) for n in IN])
    out[f"repeat/{{i}}"] = sum(int((a != b).sum()) for a, b in zip(*runs))
    out[f"start/{{i}}"] = sum(int((cp.asnumpy(start[n]) != cp.asnumpy(w._start[n])).sum())
                              for n in start)
""", tmp_path)
    assert len(out) == len(CASES) * 3
    for key, val in out.items():
        if key.startswith("fd/"):
            misfits = [abs(r - 1.0) for _, r in val]
            assert min(misfits) < 1e-6 and misfits[0] < 1e-2, (key, val)
        else:
            assert val == 0, (key, val)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_step_reads_only_its_inputs(backend, tmp_path):
    """What the window's checkpoints rest on: with every array a step
    writes, other than STEP_INPUTS, poisoned with NaN before the second
    step (the warm geometry), that step's outputs and its adjoint's are
    the clean run's, bit for bit, and its adjoint leaves nothing on the
    arrays it overwrites (warm and cold starts, rotation)."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f"""
for dtn in {dtypes!r}:
    for i, (over, warm) in enumerate({CASES!r}):
        cfg, state = build(over, warm)
        w = W.Window(state, cfg, 2, dtype=np.dtype(dtn))
        ad = {{}}
        for poison in (0, 1):
            x = {{n: v.copy() for n, v in w._start.items()}}
            M.step_base(x, w.geoms[0])
            if poison:
                for n in w._mutable:
                    if n not in IN:
                        x[n][...] = np.nan
            tape = []
            M.step_base(x, w.geoms[1], tape)
            ax = {{n: cp.zeros_like(v) for n, v in x.items()}}
            for n in IN:
                ax[n][...] = 1
            M.step_ad(ax, x, tape, w.geoms[1])
            out[f"leak/{{dtn}}/{{i}}/{{poison}}"] = sum(
                int(cp.count_nonzero(ax[n])) for n in w._mutable if n not in IN)
            ad[poison] = [cp.asnumpy(x[n]) for n in IN] + [cp.asnumpy(ax[n]) for n in IN]
        out[f"{{dtn}}/{{i}}"] = sum(int((a.view(np.uint8) != b.view(np.uint8)).sum())
                                    for a, b in zip(ad[0], ad[1]))
""", tmp_path)
    assert len(out) == len(CASES) * (1 if backend == "mlx" else 2) * 3
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


@pytest.mark.parametrize("backend", BACKENDS[:1])
def test_window_names_what_it_refuses(backend, tmp_path):
    """A regional case (specified boundaries, with map factors) is refused
    by name, and the adjoint without a forward run says what is missing."""
    out = _run(backend, '''
cfg = H.make_config(8, 6, 6, h_sca_adv_order=5, terrain_opt=1, specified=True)
try:
    W.Window(H.make_state(cfg, 1, geography=H.make_geography(cfg)), cfg, 2)
    out["spec"] = "accepted"
except NotImplementedError as e:
    out["spec"] = str(e)
cfg, state = build({}, 1)
w = W.Window(state, cfg, 2)
try:
    w.ad()
    out["order"] = "ran"
except RuntimeError as e:
    out["order"] = str(e)
''', tmp_path)
    assert "lateral boundaries other than periodic" in out["spec"], out
    assert "map factors" not in out["spec"], out
    assert "forward first" in out["order"], out


@pytest.mark.parametrize("backend", BACKENDS)
def test_window_partial_tl_start(backend, tmp_path):
    """WRFDA's TL start (W.WRFDA_TL_START, ``partial=True``): the names
    dx0 omits start at zero, word for word the full dict with zeros there;
    the adjoint of that zero fill is the restriction of Window.ad's result
    (dot test); an incomplete dict without ``partial``, or a name outside
    STEP_INPUTS, is refused by name."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f"""
cfg, state = build({{}}, 0)   # WRFDA's window starts cold
w = W.Window(state, cfg, 2, dtype=np.dtype({dtn!r}))
w.forward()
rng = np.random.default_rng(5)
dx = rnd(w._start, rng)
part = W.wrfda_tl_start({{n: dx[n] for n in W.WRFDA_TL_FIELDS}})
def dot(x, y):  # on the host in binary64 (the Metal facade has no vdot)
    return float(np.vdot(cp.asnumpy(x).astype(np.float64), cp.asnumpy(y).astype(np.float64)))
full = {{n: dx[n] if n in part else cp.zeros_like(dx[n]) for n in IN}}
a, b = w.tl(part, save_at=(1,), partial=True), w.tl(full, save_at=(1,))
out["same"] = sum(int((cp.asnumpy(a[k][n]) != cp.asnumpy(b[k][n])).sum())
                  for k in (1, 2) for n in IN)
f = rnd(w._start, rng)
# binary32 (Metal): scale by the inner products of the magnitudes, the
# roundoff bound when the terms cancel, as K.dot_product(terms=True) does
# for the other binary32 dot tests; binary64 scales by the result.
def mag(x, y):
    if {dtn!r} != "float32":
        return 0.0
    h = lambda z: np.abs(cp.asnumpy(z).astype(np.float64))
    return float(np.vdot(h(x), h(y)))
lhs = sum(dot(a[2][n], f[n]) for n in IN)
lmag = sum(mag(a[2][n], f[n]) for n in IN)
adj = w.ad(f)
rhs = sum(dot(part[n], adj[n]) for n in part)
rmag = sum(mag(part[n], adj[n]) for n in part)
out["dot"] = abs(lhs - rhs) / max(abs(lhs), lmag, rmag, 1e-300)
# the helpers' pair: <TL(start(g)), f> == <g, start_ad(AD(f))>
g = {{n: dx[n] for n in W.WRFDA_TL_FIELDS}}
ga = W.wrfda_tl_start_ad(adj)
rhs = sum(dot(g[n], ga[n]) for n in g)
rmag = sum(mag(g[n], ga[n]) for n in g)
out["dot_helper"] = abs(lhs - rhs) / max(abs(lhs), lmag, rmag, 1e-300)
out["shared"] = int(part["p_pert"] is part["p"] and part["muts"] is part["mup"])
for key, bad, kw in (("missing", part, {{}}), ("unknown", {{**full, "qv": full["u"]}}, {{}}),
                     ("empty", {{}}, {{"partial": True}})):
    try:
        w.tl(bad, **kw)
        out[key] = ""
    except ValueError as e:
        out[key] = str(e)
""", tmp_path)
    assert out["same"] == 0
    for key in ("dot", "dot_helper"):
        # binary32 on Metal: 1e-5 of the magnitude-scaled misfit, as
        # test_window_adjoint_with_forcing uses (scaled by the result it
        # measured 2e-5 here, from cancellation across the terms)
        assert out[key] < (1e-5 if backend == "mlx" else 1e-12), (key, out[key])
    assert out["shared"] == 1
    assert "dx0 is empty" in out["empty"], out["empty"]
    assert "lacks ['php', 'al', 'alt']" in out["missing"], out["missing"]
    assert "partial=True" in out["missing"]
    assert "['qv']" in out["unknown"], out["unknown"]
