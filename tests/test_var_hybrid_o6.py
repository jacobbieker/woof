"""Hybrid 3DEnVar against WRFDA 4.7.1 on the tutorial case (plan V3.2; gates O6 and O8).

Inputs: ``WOOF_WRFDA471_DIR`` as for ``test_var_o6`` (``be.dat``, ``ob.ascii``,
the oracle's ``woof_xb.bin``) and ``WOOF_WRFDA471_HYBRID_DIR``, the run
directory of ``tools/wrfda_oracle/run_hybrid.sh hybrid-ne10`` (its ``ep/``
files and ``be.vertloc.dat``; both checked against
``tests/data/wrfda471/manifest.json``).  WRFDA ran the tutorial namelist with
``ensdim_alpha = 10``, ``je_factor = 2`` (so ``jb_factor = 2``),
``alpha_corr_scale = 200`` km and ``alpha_vertloc_opt = 2``.

What is asserted, and why it is enough:

- **O6 on the reference tier**, with WRFDA's ``be.vertloc.dat``: ``J_0`` to
  1e-12, the first ten iterations' ``J`` and ``|g|`` to 1e-8, the final
  ``J`` to 1e-6, the same 50 iterations, and the final Jo, Jb and Je to
  WRFDA's printed digits.  Measured: all 51 ``minimize_cg`` lines within
  3.5e-14 relative.
- **woof's own vertical localisation** (``da_get_alpha_vertloc`` from
  ``znw`` and ``ptop``, no file) meets the same O6 tolerances.
- **Device tiers.** The oracle tier's first iterations are the reference's
  words; the production tier (binary32) meets O6's final ``J`` and
  iteration count (7e-8 measured on CUDA) and its words, the same in fast
  and strict child processes, hash to a constant that CUDA and Metal both
  reproduce (O8).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

REPO = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).parent / "data" / "wrfda471" / "manifest.json"

#: SHA-256 of the production tier's per-iteration (J, |g|, step) words, the
#: final J and |g| and the float32 xhat; measured on CUDA (A6000, fast and
#: strict) and Metal (strict).
PRODUCTION_DIGEST = "71f42528632c45405ebe5ebb1aa3723ce20294186e07c3f88f0509d86a1d1216"


def _d(s: str) -> float:
    return float(s.replace("D", "E"))


def _man():
    return json.loads(MANIFEST.read_text())["hybrid_o6"]


def _dirs():
    d = os.environ.get("WOOF_WRFDA471_DIR")
    h = os.environ.get("WOOF_WRFDA471_HYBRID_DIR")
    if not d or not all((Path(d) / n).is_file() for n in ("be.dat", "ob.ascii", "woof_xb.bin")):
        pytest.skip("WOOF_WRFDA471_DIR with be.dat, ob.ascii and woof_xb.bin is not set")
    if not h or not (Path(h) / "ep").is_dir() or not (Path(h) / "be.vertloc.dat").is_file():
        pytest.skip("WOOF_WRFDA471_HYBRID_DIR (run_hybrid.sh's run directory: ep/, be.vertloc.dat) is not set")
    return d, h


@lru_cache(maxsize=None)
def _cost_inputs(d: str, h: str, own_vertloc: bool = False):
    from test_var_o6 import _inputs

    from woof.da.var import hybrid as hy
    from woof.da.var.oracle import read_dump

    man = _man()
    files = sorted((Path(h) / "ep").iterdir())
    sha = hashlib.sha256()
    for f in files:
        sha.update(f.read_bytes())
    assert sha.hexdigest() == man["ep_sha256"], "ep files differ from the manifest's"
    vl_path = Path(h) / "be.vertloc.dat"
    assert hashlib.sha256(vl_path.read_bytes()).hexdigest() == man["vertloc_sha256"]
    b, xbu, term = _inputs(d)
    xb = read_dump(Path(d) / "woof_xb.bin")
    nx, ny = xb["lat"].shape
    nl = man["namelist"]
    ep = hy.Ensemble.from_perturbations(hy.read_ep(Path(h) / "ep", nl["ensdim_alpha"], shape=(nx, ny)))
    cfg = hy.HybridConfig(je_factor=nl["je_factor"], alpha_corr_scale=nl["alpha_corr_scale"],
                          alpha_vertloc_opt=nl["alpha_vertloc_opt"])
    xbh = dict(xbu, ds=float(xb["ds"]), znw=xb["znw"], ptop=float(xb["ptop"]))
    vertloc = None if own_vertloc else hy.read_vertloc(vl_path)
    return b, xbh, term, ep, cfg, vertloc


def run_tier(tier: str, ntmax: int | None = None, own_vertloc: bool = False):
    from woof.da.var import hybrid as hy
    from woof.da.var.minimise import minimise_cg

    d, h = _dirs()
    b, xbh, term, ep, cfg, vertloc = _cost_inputs(d, h, own_vertloc)
    cost = hy.hybrid_cost(b, xbh, term, ep, cfg, tier=tier, vertloc=vertloc)
    nl = _man()["namelist"]
    r = minimise_cg(cost, ntmax=ntmax or nl["ntmax"], eps=nl["eps"], orthonorm_gradient=nl["orthonorm_gradient"])
    x = r.xhat if tier == "reference" else r.xhat.get()
    return r, np.asarray(x), cost


def words(r, xhat) -> str:
    h = hashlib.sha256()
    for a in (r.j, r.g, r.step, [r.j_final, r.g_final]):
        h.update(struct.pack(f"<{len(a)}d", *a))
    h.update(np.ascontiguousarray(xhat, dtype="<f4").tobytes())
    return h.hexdigest()


def _check_o6(r, final_rtol=1e-6, early_rtol=1e-8):
    man = _man()
    rows = [s.split() for s in man["log"]]
    assert abs(r.j[0] - _d(rows[0][3])) <= 1e-12 * _d(rows[0][3])
    assert abs(r.g[0] - _d(rows[0][4])) <= 1e-12 * _d(rows[0][4])
    for k in range(1, 11):
        jw, gw = _d(rows[k][3]), _d(rows[k][4])
        assert abs(r.j[k] - jw) <= early_rtol * jw, (k, r.j[k], jw)
        assert abs(r.g[k] - gw) <= early_rtol * gw, (k, r.g[k], gw)
    jf = _d(rows[-1][3])
    assert abs(r.j_final - jf) <= final_rtol * jf
    assert r.iterations == man["iterations"]
    return max(max(abs(r.j[k] - _d(w[3])) / _d(w[3]), abs(r.g[k] - _d(w[4])) / _d(w[4]))
               for k, w in enumerate(rows))


@pytest.fixture(scope="module")
def reference():
    return run_tier("reference")


def test_hybrid_o6_reference_tier_against_wrfdas_log(reference):
    r, xhat, cost = reference
    worst = _check_o6(r)
    j = cost.calculate_j(cost.asarray(xhat), None, iter_=1)
    fin = _man()["final"]
    print(f"hybrid O6 reference: worst line {worst:.2e}; J {j.total:.5f} Jo {j.jo:.5f} Jb {j.jb:.5f} Je {j.je:.5f}")
    for key, ours in (("j", j.total), ("jo", j.jo), ("jb", j.jb), ("je", j.je)):
        assert abs(ours - float(fin[key])) <= 1e-5 * max(1.0, abs(ours)), (key, ours, fin[key])
    assert worst <= 1e-12


def test_hybrid_o6_with_woofs_own_vertical_localisation():
    r, _, cost = run_tier("reference", own_vertloc=True)
    worst = _check_o6(r)
    print(f"hybrid O6, woof's log-p vertloc (mz {cost.u.alpha.mz}): worst line {worst:.2e}")


@requires_gpu
def test_hybrid_o6_oracle_tier_is_the_reference_word_for_word(reference):
    # binary64 on the device is slow (soft binary64 on Metal): three iterations
    r, x, _ = run_tier("oracle", ntmax=3)
    rr, xr, _ = run_tier("reference", ntmax=3)
    assert (r.j, r.g, r.step) == (rr.j, rr.g, rr.step)
    np.testing.assert_array_equal(x, xr)


_CHILD = r"""
import sys
sys.path.insert(0, sys.argv[1])
from test_var_hybrid_o6 import run_tier, words
r, x, _ = run_tier("production")
print("RESULT", words(r, x), repr(r.j_final), r.iterations)
"""


@requires_gpu
@pytest.mark.parametrize("mode", ["fast", "strict"])
def test_hybrid_o6_production_tier_fast_strict_and_across_backends(mode):
    _dirs()
    env = dict(os.environ, GPUWM_NUMERICS=mode)
    proc = subprocess.run([sys.executable, "-c", _CHILD, str(Path(__file__).parent)], env=env,
                          capture_output=True, text=True, timeout=3600, cwd=REPO)
    assert proc.returncode == 0, proc.stderr[-4000:]
    _, digest, jf, iters = [s for s in proc.stdout.splitlines() if s.startswith("RESULT ")][-1].split()
    print(f"{mode}: {digest} J_final {jf} iterations {iters}")
    rows = _man()["log"]
    jw = _d(rows[-1].split()[3])
    assert abs(float(jf) - jw) <= 1e-6 * jw and int(iters) == _man()["iterations"]
    from woof import backend

    if mode == "strict" or backend.active() == "cuda":
        assert digest == PRODUCTION_DIGEST
