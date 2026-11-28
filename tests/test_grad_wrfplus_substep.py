"""One acoustic substep's chained tangent-linear and adjoint (P3a,
woof.grad.wrfplus.acoustic_substep_tl/_ad): advance_uv, advance_mu_t and
advance_w with calc_p_rho in woof's strict order, two substeps running.

Breakage this prevents:
- operators that pass each other state under the wrong name (mu_old vs
  mu_pp_old, th_old, p_old) or in the wrong order;
- an adjoint chain that linearises an operator about the wrong entry
  state (the tape), or drops an adjoint written by a later operator;
- a TL chain that is not the derivative of the chained base update.

Each operator's own dot, finite-difference and WRFPLUS checks are in
test_grad_wrfplus_advance_{uv,mu_t,w}.py.  Each probe runs in a child
process, as there.
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




def _run(backend, body, tmp_path):
    env = dict(os.environ, WOOF_BACKEND=backend, PYTHONPATH=str(ROOT),
               WOOF_HOST_CACHE=str(tmp_path / "hostcache"), WOOF_HOST_THREADS="4")
    prelude = (
        "import json, numpy as np, woof\nimport cupy as cp\n"
        "from woof.grad.wrfplus import acoustic as A, cases as C, _checks as K\n"
        "out = {}\n"
        "NSUB = 2\n"
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n"
        "def copy(d):\n"
        "    return {k: v.copy() for k, v in d.items()}\n"
        "def run_tl(gs, tr, pe, tape=None):\n"
        "    st, sd = copy(tr), copy(pe)\n"
        "    for _ in range(NSUB):\n"
        "        A.acoustic_substep_tl(st, sd, gs, tape)\n"
        "    return st, sd\n"
        "def run_ad(gs, tr, ya):\n"
        "    tape = []\n"
        "    zero = {n: cp.zeros_like(v) for n, v in ya.items()}\n"
        "    st, _ = run_tl(gs, tr, zero, tape)\n"
        "    xa = copy(ya)\n"
        "    for _ in range(NSUB):\n"
        "        st = A.acoustic_substep_ad(xa, st, tape, gs)\n"
        "    assert not tape\n"
        "    return xa\n")
    script = prelude + body + "\nprint('RESULT ' + json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


# nz 3 (advance_uv's floor) without 3-D base state, a lid, nz 19
# across the adjoints' level chunks (AD_KCHUNK = 8), and the external-mode
# filter (advance_uv reading the mudf the previous substep wrote).
CONFIGS = [dict(), dict(nz=3, ny=2, nx=3, base3d=0),
           dict(nz=19, ny=3, nx=4, top_lid=1), dict(top_lid=1, emdiv=0.05)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_substep_dot_product(backend, tmp_path):
    """<(T - I) dx, dy> == <dx, (T - I)^T dy> over every state and active
    name of two chained substeps: 1e-12 in binary64, 1e-5 in binary32.
    The identity (names a substep only reads) is taken out so it cannot
    swamp the misfit."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f'''
for dtn in {dtypes!r}:
    dt = np.dtype(dtn)
    for i, over in enumerate({CONFIGS!r}):
        for hybrid in (False, True):
            gs = C.substep_geometries(**over)
            tr = dev(C.substep_trajectory(gs, 11 + i, dt, hybrid=hybrid))
            p = dev(C.substep_perturbation(gs, 100 + i, dt))
            q = dev(C.substep_perturbation(gs, 200 + i, dt))
            N = list(p)
            def tl(dx):
                sd = run_tl(gs, tr, dict(zip(N, dx)))[1]
                return [sd[n] - d for n, d in zip(N, dx)]
            def ad(dy):
                xa = run_ad(gs, tr, dict(zip(N, dy)))
                return [xa[n] - d for n, d in zip(N, dy)]
            out[f"{{dtn}}/{{i}}/{{hybrid}}"] = K.dot_product(
                tl, ad, [p[n] for n in N], [q[n] for n in N],
                terms=dtn == "float32")
''', tmp_path)
    assert len(out) == len(CONFIGS) * 2 * (1 if backend == "mlx" else 2)
    for key, rel in out.items():
        limit = 1e-12 if key.startswith("float64") else 1e-5
        assert rel < limit, (key, rel)


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_substep_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x)|| / ||a TL dx|| -> 1 for two chained
    substeps, so each operator is linearised about its own entry."""
    out = _run(backend, f'''
for i, over in enumerate({CONFIGS!r}):
    gs = C.substep_geometries(**over)
    tr = dev(C.substep_trajectory(gs, 31 + i, np.float64, hybrid=bool(i % 2)))
    p = dev(C.substep_perturbation(gs, 300 + i, np.float64))
    N = list(p)
    def model(xs):
        t = dict(tr); t.update(dict(zip(N, xs)))
        zero = {{n: cp.zeros_like(t[n]) for n in N}}
        return [run_tl(gs, t, zero)[0][n] for n in N]
    def tl(dx):
        return [run_tl(gs, tr, dict(zip(N, dx)))[1][n] for n in N]
    out[str(i)] = K.tl_convergence(model, tl, [tr[n] for n in N],
                                   [p[n] for n in N], [1e-2, 1e-4, 1e-6])
''', tmp_path)
    assert len(out) == len(CONFIGS)
    for key, rows in out.items():
        misfits = [abs(r - 1.0) for _, r in rows]
        assert misfits[-1] < 1e-5, (key, rows)
        assert misfits[0] < 0.1, (key, rows)
