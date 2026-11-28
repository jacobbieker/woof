"""The external-mode filter's substep chain (emdiv > 0; P3a window request
1, woof.grad.wrfplus.acoustic): advance_uv's MUDF_XY from the mudf the
previous substep's advance_mu_t wrote, over several substeps, against
WRFPLUS 4.7.1's g_/a_advance_uv and g_/a_advance_mu_t (with
g_/a_spec_bdyupdate) chained the same way
(tests/data/wrfplus/emdiv_chain.npz, made by
tools/wrfplus_oracle/emdiv_chain.F90).

Breakage this prevents:
- a filter whose TL or AD is not WRFPLUS's once mudf is carried (the
  read of the previous substep's mudf, advance_mu_t's write of
  dmdt + mu_tend, the adjoint handed from one to the other);
- an adjoint chain that is not the transpose of its TL, or that drops
  what it holds on entry;
- a TL that is not the derivative of the chained base update.

The single-call checks are test_grad_wrfplus_advance_uv.py (the emdiv
branch) and test_grad_wrfplus_advance_mu_t.py (mudf); the full step with
emdiv is test_grad_wrfplus_emdiv_step.py.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/wrfplus/emdiv_chain.npz"


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


def _run(backend, body, tmp_path):
    env = dict(os.environ, WOOF_BACKEND=backend, PYTHONPATH=str(ROOT),
               WOOF_HOST_CACHE=str(tmp_path / "hostcache"), WOOF_HOST_THREADS="4")
    prelude = (
        "import json, sys, numpy as np, woof\nimport cupy as cp\n"
        f"sys.path.insert(0, {str(ROOT / 'tools/wrfplus_oracle')!r})\n"
        "import fixture as F\n"
        "from woof.grad.wrfplus import acoustic as A, _checks as K\n"
        "out = {}\n"
        "STATE = ('u_pp', 'v_pp') + A.MT_STATE\n"
        "def dev(d, dt):\n"
        "    return {k: cp.asarray(np.asarray(v, dtype=dt)) for k, v in d.items()}\n"
        "def copy(d):\n"
        "    return {k: v.copy() for k, v in d.items()}\n"
        "def load(name, dt):\n"
        "    uvg, mtg, emdiv, nsub, tr, pe, seeds = F.chain_inputs(name)\n"
        "    return uvg, mtg, nsub, dev(tr, dt), dev(pe, dt), dev(seeds, dt)\n"
        "def run_tl(uvg, mtg, nsub, tr, pe, tape=None):\n"
        "    # nsub substeps of advance_uv then advance_mu_t, on one dict\n"
        "    st, sd = copy(tr), copy(pe)\n"
        "    for _ in range(nsub):\n"
        "        if tape is not None:\n"
        "            tape.append(copy(st))\n"
        "        A.advance_uv_tl(st['u_pp'], st['v_pp'], sd['u_pp'], sd['v_pp'], st, sd, uvg)\n"
        "        if tape is not None:\n"
        "            tape.append(copy(st))\n"
        "        A.advance_mu_t_tl(st, sd, st, sd, mtg)\n"
        "    return st, sd\n"
        "def run_ad(uvg, mtg, nsub, tr, ya):\n"
        "    tape = []\n"
        "    run_tl(uvg, mtg, nsub, tr, {n: cp.zeros_like(v) for n, v in tr.items()}, tape)\n"
        "    xa = {n: cp.zeros_like(v) for n, v in tr.items()}\n"
        "    for n, v in ya.items():\n"
        "        xa[n] += v\n"
        "    for _ in range(nsub):\n"
        "        A.advance_mu_t_ad(xa, tape.pop(), xa, mtg)\n"
        "        A.advance_uv_ad(xa['u_pp'], xa['v_pp'], tape.pop(), xa, uvg)\n"
        "    return xa\n"
        "ACTIVE = tuple(dict.fromkeys(('u_pp', 'v_pp') + A.UV_ACTIVE + A.MT_STATE\n"
        "                             + A.MT_ACTIVE))\n")
    script = prelude + body + "\nprint('RESULT ' + json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


CASES = ("wrf", "strong_lid_moist")


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrfplus_oracle_agreement(backend, tmp_path):
    """The chained TL and AD reproduce WRFPLUS 4.7.1's chained g_/a_
    advance_uv and advance_mu_t: the winds, mass, theta and mudf after
    the chain, and the adjoints of the chain's start and of its held
    tendencies and mass."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f'''
z = np.load({str(FIXTURE)!r})
dt = np.dtype({dtn!r})
for case in {CASES!r}:
    uvg, mtg, nsub, _, _, _ = load(case, dt)
    assert int(z[case + "/nsub"]) == nsub
    pick = lambda pre: {{k.split("/", 2)[2]: cp.asarray(z[k].astype(dt))
                         for k in z.files if k.startswith(case + "/" + pre + "/")}}
    tr, pe, seeds = pick("traj"), pick("pert"), pick("seed")
    exp = {{k.split("/", 2)[2]: z[k] for k in z.files if k.startswith(case + "/wrfplus/")}}
    masks = {{k.split("/", 2)[2]: z[k] for k in z.files if k.startswith(case + "/mask/")}}
    st, sd = run_tl(uvg, mtg, nsub, tr, pe)
    xa = run_ad(uvg, mtg, nsub, tr, {{n: seeds[n] for n in ("u_pp", "v_pp", "mu_pp",
                                                            "th_pp", "mudf")}})
    got = {{}}
    for n in ("u_pp", "v_pp", "mu_pp", "th_pp", "mudf"):
        got["tl_" + n], got["tl_d_" + n] = st[n], sd[n]
    for n in ("u_pp", "v_pp", "th_pp", "mu_pp", "mudf", "ru_t", "rv_t", "rth_t",
              "rmu_t", "mup"):
        got["ad_" + n] = xa[n]
    assert sorted(got) == sorted(exp), (sorted(got), sorted(exp))
    for name, e in exp.items():
        w = cp.asnumpy(got[name]).astype(np.float64)
        m = masks.get(name, np.ones(e.shape, bool))
        scale = max(float(np.abs(e[m]).max()), 1e-300)
        out[case + "/" + name] = float(np.abs(w[m] - e[m]).max()) / scale
    # the filter must carry signal: without it the chain moves the winds
    off = A.UVGeometry(**{{**uvg.__dict__, "use_emdiv": 0}})
    st0, sd0 = run_tl(off, mtg, nsub, tr, pe)
    m = masks["tl_u_pp"]
    out[case + "/filter"] = float(np.abs(cp.asnumpy(sd0["u_pp"] - sd["u_pp"])[m]).max()
                                  / np.abs(exp["tl_d_u_pp"][m]).max())
''', tmp_path)
    filt = {k: out.pop(k) for k in list(out) if k.endswith("/filter")}
    assert len(filt) == len(CASES) and all(v > 1e-4 for v in filt.values()), filt
    limit = 1e-5 if backend == "mlx" else 1e-12
    bad = {k: v for k, v in out.items() if not v < limit}
    assert not bad, bad
    assert len(out) == len(CASES) * 20


@pytest.mark.parametrize("backend", BACKENDS)
def test_chain_dot_product(backend, tmp_path):
    """<(T - I) dx, dy> == <dx, (T - I)^T dy> over every name the chain
    reads or writes: 1e-12 in binary64, 1e-5 in binary32."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f'''
for dtn in {dtypes!r}:
    dt = np.dtype(dtn)
    for case in {CASES!r}:
        uvg, mtg, nsub, tr, pe, _ = load(case, dt)
        N = [n for n in ACTIVE if n in pe]
        rng = np.random.default_rng(7)
        q = [cp.asarray(rng.standard_normal(pe[n].shape).astype(dt)) * float(cp.abs(pe[n]).max())
             for n in N]
        def tl(dx):
            p = {{n: cp.zeros_like(v) for n, v in tr.items()}}
            p.update(dict(zip(N, dx)))
            sd = run_tl(uvg, mtg, nsub, tr, p)[1]
            return [sd[n] - d for n, d in zip(N, dx)]
        def ad(dy):
            xa = run_ad(uvg, mtg, nsub, tr, dict(zip(N, dy)))
            return [xa[n] - d for n, d in zip(N, dy)]
        out[f"{{dtn}}/{{case}}"] = K.dot_product(tl, ad, [pe[n] for n in N], q,
                                                 terms=dtn == "float32")
''', tmp_path)
    assert len(out) == len(CASES) * (1 if backend == "mlx" else 2)
    for key, rel in out.items():
        limit = 1e-12 if key.startswith("float64") else 1e-5
        assert rel < limit, (key, rel)


@pytest.mark.parametrize("backend", NO_MLX)
def test_chain_adjoint_accumulates(backend, tmp_path):
    """The chain's adjoint adds into the adjoints it is handed: a prior on
    the active names it only reads (the tendencies, the reference winds,
    the base mass) comes out on top of the chain's own, unchanged."""
    out = _run(backend, f'''
for case in {CASES!r}:
    uvg, mtg, nsub, tr, pe, seeds = load(case, np.float64)
    ya = {{n: seeds[n] for n in ("u_pp", "v_pp", "mu_pp", "th_pp", "mudf")}}
    base = run_ad(uvg, mtg, nsub, tr, ya)
    read = ("ru_t", "rv_t", "rth_t", "rmu_t", "mup", "u", "v", "ww_ref", "thp")
    prior = {{n: cp.full_like(tr[n], 3.0) for n in read}}
    got = run_ad(uvg, mtg, nsub, tr, {{**ya, **prior}})
    out[case] = max(float(cp.abs(got[n] - base[n] - prior[n]).max()) for n in read)
''', tmp_path)
    assert len(out) == len(CASES)
    for key, err in out.items():
        assert err < 1e-9, (key, err)


@pytest.mark.parametrize("backend", NO_MLX)
def test_chain_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x)|| / ||a TL dx|| -> 1 over the chain, so
    each call is linearised about its own entry (mudf included)."""
    out = _run(backend, f'''
for case in {CASES!r}:
    uvg, mtg, nsub, tr, pe, _ = load(case, np.float64)
    N = [n for n in ACTIVE if n in pe]
    def model(xs):
        t = dict(tr); t.update(dict(zip(N, xs)))
        zero = {{n: cp.zeros_like(v) for n, v in t.items()}}
        return [run_tl(uvg, mtg, nsub, t, zero)[0][n] for n in STATE]
    def tl(dx):
        p = {{n: cp.zeros_like(v) for n, v in tr.items()}}
        p.update(dict(zip(N, dx)))
        return [run_tl(uvg, mtg, nsub, tr, p)[1][n] for n in STATE]
    out[case] = K.tl_convergence(model, tl, [tr[n] for n in N], [pe[n] for n in N],
                                 [1e-2, 1e-4, 1e-6])
''', tmp_path)
    assert len(out) == len(CASES)
    for key, rows in out.items():
        misfits = [abs(r - 1.0) for _, r in rows]
        assert misfits[-1] < 1e-5, (key, rows)
        assert misfits[0] < 0.1, (key, rows)
