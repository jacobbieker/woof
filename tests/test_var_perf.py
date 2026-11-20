"""woof.da.var performance work (plan V1.6): the device H'/H'^T and the timing gate.

- :class:`~woof.da.var.obsop_device.DeviceObsTerm` returns the host
  :class:`~woof.da.var.obsop_wrfda.FlatObsTerm`'s words, ``tl`` and ``ad``,
  on both tiers: the conventional synthetic case (many reports sharing
  columns), the O3 synthetic grids with every implemented type (gpspw,
  gpsref and satem included) and, with ``WOOF_WRFDA471_DIR``, the tutorial's
  ~9k observations.  The host term is NumPy, so this is the same on every
  backend; the child-process test repeats it under fast and strict numerics.
- The section 4.12 target, opt-in (``WOOF_VAR_PERF=1``; a timing on a shared
  card is not a unit test): a 50-iteration production-tier 3D-Var on
  200x200x50 with ~10^5 observations within 5 s on CUDA and 20 s on Metal,
  minimisation only.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

REPO = Path(__file__).resolve().parents[1]
TARGET_S = {"cuda": 5.0, "mlx": 20.0}  # the A6000 and the M3 Max (Metal)


def _synthetic_all():
    """The O3 synthetic grid with the linear and the nonlinear types together."""
    from test_var_obsop_wrfda import _nonlinear_obs, _synthetic_obs

    from woof.da.var.obsop_wrfda import FlatObsTerm

    xb, iv = _synthetic_obs()
    _, iv2 = _nonlinear_obs()
    iv.types.update(iv2.types)
    iv.ntotal.update(iv2.ntotal)
    return FlatObsTerm(iv, xb)


def _conventional():
    from _var_perf_case import case

    return case(40, 30, 12, seed=3, scale=8.0)[2]


def _term(which: str):
    if which == "synthetic":
        return _synthetic_all()
    if which == "conventional":
        return _conventional()
    from test_var_o6 import _dir, _inputs

    return _inputs(str(_dir()))[2]


def words(term, precision: str, seed: int = 4):
    """``(host, device)`` SHA-256 of ``tl`` and ``ad`` words for random x and y."""
    import cupy as cp

    from woof.da.var.obsop_device import DeviceObsTerm

    dt = np.float64 if precision == "oracle" else np.float32
    nx, ny, nz = term.term.stencil.xb["u"].shape
    rng = np.random.default_rng(seed)
    x = {k: rng.standard_normal((nx, ny, nz)).astype(dt) for k in ("u", "v", "t", "q", "p", "rho")}
    x["q"] = (x["q"] * dt(1e-3)).astype(dt)
    x["psfc"] = (100.0 * rng.standard_normal((nx, ny))).astype(dt)
    y = rng.standard_normal(term.size).astype(dt)
    dev = DeviceObsTerm(term, precision)
    keys = ("u", "v", "t", "q", "p", "rho", "psfc")
    h, d = hashlib.sha256(), hashlib.sha256()
    h.update(np.asarray(term.tl(x), dtype="<f8").tobytes())
    d.update(np.asarray(dev.tl({k: cp.asarray(v) for k, v in x.items()}).get(), dtype="<f8").tobytes())
    gh = term.ad(y)
    gd = dev.ad(cp.asarray(y))
    for k in keys:
        h.update(np.ascontiguousarray(np.asarray(gh[k], dtype=np.float64).astype(dt)).tobytes())
        d.update(np.ascontiguousarray(gd[k].get(), dtype=dt).tobytes())
    return h.hexdigest(), d.hexdigest()


@requires_gpu
@pytest.mark.parametrize("precision", ["oracle", "production"])
@pytest.mark.parametrize("which", ["conventional", "synthetic", "tutorial"])
def test_device_obs_term_is_the_host_term_word_for_word(which, precision):
    if which == "tutorial":
        from test_var_o6 import _dir

        _dir()
    host, dev = words(_term(which), precision)
    assert dev == host


_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from test_var_perf import _term, words
out = {}
for which in ("conventional", "synthetic"):
    for p in ("oracle", "production"):
        out[which + "/" + p] = words(_term(which), p)
print("RESULT " + json.dumps(out))
"""


@requires_gpu
@pytest.mark.parametrize("mode", ["fast", "strict"])
def test_device_obs_words_in_fast_and_strict_child_processes(mode):
    env = dict(os.environ, GPUWM_NUMERICS=mode)
    proc = subprocess.run([sys.executable, "-c", _CHILD, str(Path(__file__).parent)], env=env,
                          capture_output=True, text=True, timeout=1800, cwd=REPO)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = [s for s in proc.stdout.splitlines() if s.startswith("RESULT ")][-1]
    for key, (host, dev) in json.loads(line[len("RESULT "):]).items():
        assert dev == host, key


@requires_gpu
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_axpy_with_a_device_dot_is_the_host_axpy(dtype):
    import cupy as cp

    from woof.da.var.control import REDUCTIONS, axpy, axpy_neg_device, dot, dot_device

    rng = np.random.default_rng(9)
    x = cp.asarray(rng.standard_normal(10_000).astype(dtype))
    q = cp.asarray(rng.standard_normal(10_000).astype(dtype))
    for red in REDUCTIONS:
        y1, y2 = x.copy(), x.copy()
        axpy(-dot(x, q, red), q, y1)
        axpy_neg_device(dot_device(x, q, red), q, y2)
        np.testing.assert_array_equal(y1.get(), y2.get())


@requires_gpu
@pytest.mark.skipif(os.environ.get("WOOF_VAR_PERF") != "1", reason="timing gate: set WOOF_VAR_PERF=1")
def test_v16_production_target_200x200x50():
    from _var_perf_case import case

    from woof import backend
    from woof.da.var.cost import VarCost
    from woof.da.var.minimise import minimise_cg

    b, xbu, term = case(200, 200, 50)
    cost = VarCost.device(b, xbu, term, "production")
    minimise_cg(cost, ntmax=2, eps=0.0, orthonorm_gradient=True)  # compile outside the timing
    t0 = time.perf_counter()
    r = minimise_cg(cost, ntmax=50, eps=0.0, orthonorm_gradient=True)
    dt = time.perf_counter() - t0
    name = backend.active()
    print(f"V1.6 200x200x50, {term.size} obs, cv {cost.cv_size}: {r.iterations} iterations in {dt:.2f} s ({name})")
    assert r.iterations == 50
    assert dt <= TARGET_S.get(name, 20.0)
