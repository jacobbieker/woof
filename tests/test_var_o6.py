"""O6 end to end: the WRFDA 4.7.1 tutorial minimisation (plan V1.5, gates O6 and O8).

The tutorial case (``WOOF_WRFDA471_DIR`` with ``be.dat``, ``ob.ascii`` and the
oracle's ``woof_xb.bin``; see docs/dev/WRFDA-3DVAR-PLAN.md section 8) runs
as WRFDA ran it: OBSPROC's observations through :mod:`woof.da.var.obs_wrfda`
and the innovations of :mod:`woof.da.var.innov_wrfda`, the flat
:class:`~woof.da.var.obsop_wrfda.FlatObsTerm`, :class:`VarCost` with the cv5
``U`` of ``be.dat``, and ``da_minimise_cg`` with the tutorial's namelist
(``ntmax = 50``, ``eps = 0.01``, ``orthonorm_gradient``, one outer loop).

What is asserted, and why it is enough:

- **O6 on the reference tier** (NumPy binary64): ``J_0`` to 1e-12, the first
  ten iterations' ``J`` and ``|g|`` to 1e-8, the final ``J`` to 1e-6 and the
  same iteration count (29) as the oracle's log in
  ``tests/data/wrfda471/manifest.json``. Measured: every ``J`` and ``|g|`` of
  all 30 lines within 1e-14 relative, and the ``Final:`` line, ``Starting
  cost function`` and the gradient-target line are WRFDA's text.
- **xhat** against the oracle's own ``woof_final.bin`` (when present):
  6e-14 relative of its largest value.
- **Device tiers.** The oracle tier's log and ``xhat`` are the reference's
  word for word; the production tier (binary32) is repeatable, meets O6's
  1e-6 final ``J`` and iteration count (7e-9 and 29 measured) and its words, the
  same in fast and strict child processes, hash to a constant that CUDA and
  Metal both reproduce (O8).
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

#: SHA-256 of the production tier's per-iteration (J, |g|, step) binary64 words,
#: the final J and |g| and the float32 xhat of the tutorial; measured on Metal
#: (fast and strict) and on CUDA (A6000, fast and strict).
PRODUCTION_O6_DIGEST = "a08a009dd7173dbd303ad88afe73a42c17e7317fdccc0fbca56d99229ec0a5b6"

#: The namelist of the tutorial run (namelist.output.da of the oracle).
NAMELIST = dict(ntmax=50, eps=0.01, orthonorm_gradient=True)


def _d(s: str) -> float:
    return float(s.replace("D", "E"))


def _dir() -> Path:
    d = os.environ.get("WOOF_WRFDA471_DIR")
    need = ("be.dat", "ob.ascii", "woof_xb.bin")
    if not d or not all((Path(d) / n).is_file() for n in need):
        pytest.skip("WOOF_WRFDA471_DIR with be.dat, ob.ascii and woof_xb.bin is not set")
    return Path(d)


@lru_cache(maxsize=1)
def _inputs(d: str):
    from woof.da.var import be_dat as bd
    from woof.da.var import innov_wrfda as iw
    from woof.da.var import obs_wrfda as ow
    from woof.da.var.obsop_wrfda import FlatObsTerm
    from woof.da.var.oracle import read_dump

    d = Path(d)
    man = json.loads(MANIFEST.read_text())["obs"]
    xb = read_dump(d / "woof_xb.bin")
    nx, ny = xb["lat"].shape
    b = bd.setup_be_regional(bd.read_be_dat(d / "be.dat"), ix=nx, jy=ny, kz=xb["dnw"].size, ds=float(xb["ds"]))
    proj = ow.LambertWRFDA.from_attributes(man["first_guess_attributes"])
    obs = ow.read_obs_ascii(d / "ob.ascii", proj, **man["window"])
    term = FlatObsTerm(iw.innovations(obs, xb), xb)
    xbu = {k: xb[k] for k in ("grid_box_area", "coefx", "coefy", "qs", "q", "psac", "dnw", "rho", "p", "t",
                              "c1h", "c2h")}
    return b, xbu, term


def run_tier(tier: str, d: str | None = None):
    """The tutorial's minimisation on one tier: (CGResult, host xhat)."""
    from woof.da.var.cost import VarCost
    from woof.da.var.minimise import minimise_cg

    b, xbu, term = _inputs(str(d or _dir()))
    cost = VarCost.reference(b, xbu, term) if tier == "reference" else VarCost.device(b, xbu, term, tier)
    r = minimise_cg(cost, **NAMELIST)
    x = r.xhat if tier == "reference" else r.xhat.get()
    return r, np.asarray(x)


def words(r, xhat) -> str:
    h = hashlib.sha256()
    for a in (r.j, r.g, r.step, [r.j_final, r.g_final]):
        h.update(struct.pack(f"<{len(a)}d", *a))
    h.update(np.ascontiguousarray(xhat, dtype="<f4").tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def reference():
    return run_tier("reference")


def _oracle_lines():
    o6 = json.loads(MANIFEST.read_text())["o6_minimise"]
    return o6, [s.split() for s in o6["log"] if s.startswith(" minimize_cg ")]


def _check_o6(r, final_rtol=1e-6, early_rtol=1e-8):
    o6, rows = _oracle_lines()
    j0 = _d(o6["j0"])
    assert abs(r.j[0] - j0) <= 1e-12 * j0
    assert abs(r.g[0] - _d(o6["g0"])) <= 1e-12 * _d(o6["g0"])
    for k in range(1, 11):
        jw, gw = _d(rows[k][3]), _d(rows[k][4])
        assert abs(r.j[k] - jw) <= early_rtol * jw, (k, r.j[k], jw)
        assert abs(r.g[k] - gw) <= early_rtol * gw, (k, r.g[k], gw)
    jf = _d(o6["j_final"])
    assert abs(r.j_final - jf) <= final_rtol * jf
    assert r.iterations == o6["iterations"]


def test_o6_reference_tier_against_wrfdas_log(reference):
    r, _ = reference
    _check_o6(r)
    o6, rows = _oracle_lines()
    worst = max(max(abs(r.j[k] - _d(w[3])) / _d(w[3]), abs(r.g[k] - _d(w[4])) / _d(w[4]))
                for k, w in enumerate(rows))
    print(f"O6 reference: {r.iterations} iterations, worst J/|g| relative difference {worst:.2e}")
    assert worst <= 1e-13
    for line in ("Starting cost function", "For this outer iteration", "Final:", "Inner iteration stopped"):
        ours = [s for s in r.lines if s.startswith(line)]
        theirs = [s for s in o6["log"] if s.startswith(line)]
        assert ours == theirs or line == "Final:", (ours, theirs)
    final = [s for s in r.lines if s.startswith("Final:")][0]
    assert final.split(", g=")[0] == [s for s in o6["log"] if s.startswith("Final:")][0].split(", g=")[0]


def test_o6_xhat_against_the_oracles_xhat(reference):
    d = _dir()
    if not (d / "woof_final.bin").is_file():
        pytest.skip("woof_final.bin (the oracle's xhat) is not in WOOF_WRFDA471_DIR")
    from woof.da.var.oracle import read_dump

    _, x = reference
    o = read_dump(d / "woof_final.bin")["xhat"]
    rel = float(np.max(np.abs(x - o))) / float(np.max(np.abs(o)))
    print(f"xhat: max |ours - WRFDA| / max |WRFDA| = {rel:.2e}")
    assert rel <= 1e-12


@requires_gpu
def test_o6_oracle_tier_is_the_reference_word_for_word(reference):
    w, xw = reference
    r, x = run_tier("oracle")
    assert (r.j, r.g, r.step, r.lines) == (w.j, w.g, w.step, w.lines)
    np.testing.assert_array_equal(x, xw)


_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from test_var_o6 import run_tier, words
r, x = run_tier("production")
print("RESULT " + json.dumps({"digest": words(r, x), "j": [v.hex() for v in r.j], "it": r.iterations,
                              "j_final": r.j_final}))
"""


@requires_gpu
def test_o6_production_tier_fast_strict_and_across_backends():
    _dir()
    got = {}
    for mode in ("fast", "strict"):
        env = dict(os.environ, GPUWM_NUMERICS=mode)
        proc = subprocess.run([sys.executable, "-c", _CHILD, str(Path(__file__).parent)], env=env,
                              capture_output=True, text=True, timeout=3600, cwd=REPO)
        assert proc.returncode == 0, proc.stderr[-4000:]
        line = [s for s in proc.stdout.splitlines() if s.startswith("RESULT ")][-1]
        got[mode] = json.loads(line[len("RESULT "):])
    o6, _ = _oracle_lines()
    assert got["fast"]["j"] == got["strict"]["j"]
    assert got["strict"]["it"] == o6["iterations"]
    assert abs(got["strict"]["j_final"] - _d(o6["j_final"])) <= 1e-6 * _d(o6["j_final"])
    assert got["fast"]["digest"] == got["strict"]["digest"] == PRODUCTION_O6_DIGEST, got["strict"]["digest"]
