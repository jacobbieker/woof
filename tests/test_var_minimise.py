"""woof.da.var.minimise: WRFDA's CG (plan V1.4; gates O6 format and O8).

What is asserted, and why it is enough:

- **The log is WRFDA's text.**  ``fortran_d`` and the iteration line
  reproduce every ``minimize_cg`` line of the WRFDA 4.7.1 tutorial run
  recorded in ``tests/data/wrfda471/manifest.json`` from its own numbers,
  so the later O6 gate is a text diff.
- **CG solves the quadratic.**  On the synthetic case the analysis ``xhat``
  equals a dense NumPy solve of ``(I + G^T R^-1 G) xhat = G^T R^-1 d`` (``G
  = H' U`` built column by column through the adjoint) to 1e-8, with and
  without re-orthonormalisation and with the block preconditioner, and the
  quadratic-identity ``J`` equals ``J`` at the solution.
- **Outer loops** relinearise a linear ``H'`` and change nothing more.
- **Device tiers:** the oracle tier's per-iteration J, |g| and step are the
  reference's words; the production tier is repeatable, its words are the
  same in fast and strict child processes, and their SHA-256 is a constant
  of this file that CUDA and Metal both reproduce.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

from woof.da.var.cost import VarCost
from woof.da.var.minimise import _line12, fortran_d, minimise, minimise_cg

from test_var_cost import case

REPO = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).parent / "data" / "wrfda471" / "manifest.json"

#: SHA-256 of the production tier's per-iteration (J, |g|, step) binary64
#: words and final xhat for the synthetic case below; measured on Metal
#: (fast and strict) and CUDA (A6000, fast and strict).
PRODUCTION_CG_DIGEST = "4d174a81c8c4300d4c4783cc4e7f0055b057df9f3159e288612f2898d2ec1eaa"


def _from_d(s: str) -> float:
    return float(s.replace("D", "E"))


# ------------------------------------------------------------ the log format


def test_iteration_lines_reproduce_wrfdas_text():
    o6 = json.loads(MANIFEST.read_text())["o6_minimise"]
    lines = [s for s in o6["log"] if s.startswith(" minimize_cg ")]
    assert len(lines) == o6["iterations"] + 1
    for s in lines:
        f = s.split()
        assert _line12(int(f[1]), int(f[2]), _from_d(f[3]), _from_d(f[4]), _from_d(f[5])) == s
    final = [s for s in o6["log"] if s.startswith("Final:")][0]
    assert final == (f"Final: {o6['iterations']:3d} iter, J={fortran_d(_from_d(o6['j_final']))}, "
                     f"g={fortran_d(_from_d(o6['g_final']))}")
    start = [s for s in o6["log"] if s.startswith("Starting cost")][0]
    assert start == (f"Starting cost function: {fortran_d(_from_d(o6['j0']))}, "
                     f"Gradient= {fortran_d(_from_d(o6['g0']))}")


def test_fortran_d_edges():
    assert fortran_d(0.0) == " 0.000000000000000D+00"
    assert fortran_d(-1.5e-3) == "-1.500000000000000D-03"
    assert fortran_d(1.0e120) == " 1.000000000000000+120"


def test_log_skeleton_is_wrfdas():
    o6 = json.loads(MANIFEST.read_text())["o6_minimise"]["log"]
    b, xb, obs = case()
    r = minimise_cg(VarCost.reference(b, xb, obs), ntmax=3, orthonorm_gradient=True)
    skel = lambda s: s if not any(c.isdigit() for c in s) or s.startswith(("Starting outer", "--")) else None  # noqa: E731
    assert [skel(s) for s in r.lines[:7]] == [skel(s) for s in o6[:7]]
    assert r.lines[-5:-2] == [" ", "Inner iteration stopped after    3 iterations", " "]


# ------------------------------------------------------------ against a dense solve


def _dense(cost):
    """G = H' U column by column (G^T row i is U^T H'^T e_i), and the solution."""
    (obs,) = cost.terms
    m = obs.d.size
    g = np.zeros((m, cost.cv_size))
    for i in range(m):
        e = np.zeros(m)
        e[i] = 1.0
        g[i] = cost.vtoy_adj([e])
    w = np.where(obs.mask, 1.0 / obs.err**2, 0.0)
    a = np.eye(cost.cv_size) + g.T @ (w[:, None] * g)
    rhs = g.T @ (w * np.where(obs.mask, obs.d, 0.0))
    x = np.linalg.solve(a, rhs)
    return x, a, rhs


@pytest.fixture(scope="module")
def small():
    b, xb, obs = case(nobs=60)
    cost = VarCost.reference(b, xb, obs)
    return cost, _dense(cost)


@pytest.mark.parametrize("kw", [dict(orthonorm_gradient=True), dict(orthonorm_gradient=False),
                                dict(orthonorm_gradient=True, precondition_cg=True)],
                         ids=["orthonorm", "plain", "precon"])
def test_cg_matches_dense_solve(small, kw):
    cost, (x, a, rhs) = small
    r = minimise_cg(cost, ntmax=200, eps=1e-10, **kw)
    err = float(np.max(np.abs(r.xhat - x))) / float(np.max(np.abs(x)))
    print(f"{kw}: {r.iterations} iterations, |xhat - x_dense|/|x| = {err:.3e}, J = {r.j_final!r}")
    assert err <= 1e-8
    jmin = cost.calculate_j(x, None, 1).total
    assert r.j_final == pytest.approx(jmin, rel=1e-12)
    assert r.j[-1] == pytest.approx(jmin, rel=1e-10)  # the quadratic identity
    assert r.iterations <= cost.terms[0].d.size + 1


def test_cg_stops_at_eps(small):
    cost, _ = small
    r = minimise_cg(cost, ntmax=200, eps=0.01, orthonorm_gradient=True)
    assert r.g[-1] < 0.01 * r.g[0] <= r.g[-2]
    assert len(r.j) == r.iterations + 1 and r.step[0] == 0.0
    assert all(b <= a for a, b in zip(r.j, r.j[1:]))


def test_outer_loops(small):
    cost, (x, _, _) = small
    results, cvt = minimise(cost, max_ext_its=2, ntmax=[200, 200], eps=1e-10, orthonorm_gradient=True)
    cost.set_innovations([t.d for t in cost.terms])  # the fixture is shared
    assert float(np.max(np.abs(cvt - x))) <= 1e-8 * float(np.max(np.abs(x)))
    r1, r2 = results
    assert r2.j[0] == pytest.approx(r1.j_final, rel=1e-9)
    assert float(np.max(np.abs(r2.xhat))) <= 1e-8 * float(np.max(np.abs(x)))


# ------------------------------------------------------------ device tiers


def _run(precision, ntmax=25):
    b, xb, obs = case()
    cost = VarCost.reference(b, xb, obs) if precision == "reference" else VarCost.device(b, xb, obs, precision)
    r = minimise_cg(cost, ntmax=ntmax, eps=1e-6, orthonorm_gradient=True)
    xhat = r.xhat if precision == "reference" else r.xhat.get()
    return r, np.asarray(xhat)


def _words(r, xhat) -> str:
    h = hashlib.sha256()
    for a in (r.j, r.g, r.step, [r.j_final, r.g_final]):
        h.update(struct.pack(f"<{len(a)}d", *a))
    h.update(np.ascontiguousarray(xhat, dtype="<f4").tobytes())
    return h.hexdigest()


@requires_gpu
def test_oracle_tier_cg_is_the_reference_word_for_word():
    r, x = _run("oracle")
    w, xw = _run("reference")
    assert (r.j, r.g, r.step, r.lines) == (w.j, w.g, w.step, w.lines)
    np.testing.assert_array_equal(x, xw)


@requires_gpu
def test_production_tier_cg_is_repeatable_and_close():
    r, x = _run("production")
    r2, x2 = _run("production")
    assert (r.j, r.g, r.step) == (r2.j, r2.g, r2.step)
    np.testing.assert_array_equal(x, x2)
    w, _ = _run("reference")
    assert r.j[10] == pytest.approx(w.j[10], rel=1e-5)


_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from test_var_minimise import _run, _words
r, x = _run("production")
print("RESULT " + json.dumps({"digest": _words(r, x), "j": [v.hex() for v in r.j]}))
"""


@requires_gpu
def test_production_words_in_fast_and_strict_child_processes():
    got = {}
    for mode in ("fast", "strict"):
        env = dict(os.environ, GPUWM_NUMERICS=mode)
        proc = subprocess.run([sys.executable, "-c", _CHILD, str(Path(__file__).parent)], env=env,
                              capture_output=True, text=True, timeout=1800, cwd=REPO)
        assert proc.returncode == 0, proc.stderr[-4000:]
        line = [s for s in proc.stdout.splitlines() if s.startswith("RESULT ")][-1]
        got[mode] = json.loads(line[len("RESULT "):])
    assert got["fast"]["j"] == got["strict"]["j"]
    assert got["fast"]["digest"] == got["strict"]["digest"] == PRODUCTION_CG_DIGEST, got["strict"]["digest"]
