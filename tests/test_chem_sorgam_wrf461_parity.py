"""MADE/SORGAM against WRF v4.6.1's sorgam_driver (RPMARES branch), word for word.

The oracle (tools/chem_wrf461_oracle/sorgam) copies sorgam_driver's call tree
byte for byte from chem/module_aerosols_sorgam.F, changing only
aeroproc's isorop=1 to isorop=0 (line 1580): ISORROPIA, the isorop = 1
branch, carries a university copyright and no licence grant, so the default
build runs SORGAM's own RPMARES equilibrium.  16 columns x 10 levels cover
clean and polluted air, the eql4 skip (cold and dry cells), sulfate
production from zero to strong, RPMARES's ammonia-rich, sulfate-rich (its
cubic) and sulfate-free branches, nucleation, SOA growth, the number floors
and zero species (88 % of the extracted lines, gcov); three schedules
(dtstep 120 s x 3, 600 s x 2, 300 s x 2 with cloud-borne aerosol) carry every
array across calls.  Two kernels are graded:

* the exact path (WRF's own SOA Newton solver, which is derived from
  Numerical Recipes and so is built only from the user's own tree:
  WOOF_WRF_SOURCE, woof/core/chem_byo.py): every aerosol, gas and
  non-advected row must equal the Fortran's float32 words.  It skips, naming
  the rule, without WOOF_WRF_SOURCE;
* the default build (WOOF's clean-room partitioning solver in place of that
  one): within the tolerance below, since a different solver stops at a
  different point inside the same convergence band.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify.chem_oracle import load

ROOT = Path(__file__).resolve().parents[1] / "tests/data/oracles/chem/sorgam"
# stub_sorgam.F90: p_sulf = 3, p_hno3 = 7, p_nh3 = 16, p_hcl = 41,
# p_so4aj = 44 .. p_corn = 81.
GAS_ROW = {"sulf": 3, "hno3": 7, "nh3": 16, "hcl": 41}
AUX_IN = ("cvaro1", "cvaro2", "cvalk1", "cvole1")
# schedule -> (dtstep, calls, cw_phase); schedule 3 is a package with
# cloud-borne aerosol (cw_phase = 2), which turns on modpar's diameter limits.
# The default build's tolerance against the oracle, as a fraction of each row's
# largest value in the case.  Measured on the cases (pb, 2026-11-08): the
# condensable vapours <= 1.1e-3, the SOA masses <= 7.2e-5, every other row
# <= 1e-6 (it feels the solver only through total mass).  The solver itself
# converges to a relative step of 1e-6; the difference is WRF's solver
# stopping at a different point of an ill-conditioned problem in cells with
# almost no absorbing mass.
SOA = ("orgaro1j", "orgaro1i", "orgaro2j", "orgaro2i", "orgalk1j", "orgalk1i",
       "orgole1j", "orgole1i", "orgba1j", "orgba1i", "orgba2j", "orgba2i",
       "orgba3j", "orgba3i", "orgba4j", "orgba4i")
VAPOURS = ("cvaro1", "cvaro2", "cvalk1", "cvole1", "cvapi1", "cvapi2", "cvlim1", "cvlim2")


class _Tol(dict):
    def __missing__(self, key):
        return 3.e-3 if key in VAPOURS else 3.e-4 if key in SOA else 1.e-5


TOL = _Tol()
SCHEDULES = {1: (120.0, 3, -999888777), 2: (600.0, 2, -999888777), 3: (300.0, 2, 2)}


def _engine(a):
    return np.ascontiguousarray(np.transpose(a, (1, 2, 0)), dtype=np.float32)


def test_fixture_checksums_and_coverage():
    for line in (ROOT / "oracle-sha256sums.txt").read_text().splitlines():
        digest, rel = line.split()
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    cases = load(ROOT)
    init, out = cases["sched1_init"], cases["sched1_step3"]
    assert np.any(init["t_phy"] < 233.15) and np.any(init["t_phy"] > 290)
    assert not np.array_equal(init["chem"][:, :, :, 43], out["chem"][:, :, :, 43])
    assert np.any(out["h2oaj"] > 1.e-3)


def _run(sched, exact):
    import cupy as cp
    from woof.core.chem_sorgam import AEROSOLS, AUX, GASES, LDROG, launch_sorgam
    cases = load(ROOT)
    init = cases[f"sched{sched}_init"]
    dtstep, nstep, cw_phase = SCHEDULES[sched]
    chem = init["chem"]
    aer = [cp.asarray(_engine(chem[:, :, :, 43 + i])) for i in range(len(AEROSOLS))]
    gas = [cp.asarray(_engine(chem[:, :, :, GAS_ROW[g] - 1])) for g in GASES]
    zeros = np.zeros_like(_engine(init["t_phy"]))
    aux = [cp.asarray(_engine(init[n]) if n in AUX_IN else zeros) for n in AUX]
    vdrog = [cp.asarray(_engine(init["vdrog3"][:, :, :, q])) for q in range(LDROG)]
    t_phy, p_phy, qv, alt = (cp.asarray(_engine(init[n])) for n in ("t_phy", "p_phy", "qv", "alt"))
    vcsulf = cp.asarray(_engine(init["vcsulf_old"]))
    for step in range(1, nstep + 1):
        want = cases[f"sched{sched}_step{step}"]
        launch_sorgam(aer, gas, aux, vdrog, t_phy, p_phy, qv, alt, vcsulf, dtstep,
                      cw_phase=cw_phase, exact=exact)
        pairs = [(f"{n}", aer[i], want["chem"][:, :, :, 43 + i]) for i, n in enumerate(AEROSOLS)]
        pairs += [(g, gas[i], want["chem"][:, :, :, GAS_ROW[g] - 1]) for i, g in enumerate(GASES)]
        pairs += [(n, aux[i], want[n]) for i, n in enumerate(AUX)]
        yield step, [(name, got.get(), _engine(exp)) for name, got, exp in pairs]
        vcsulf = cp.asarray(_engine(want["vcsulf_next"]))


EXACT_REASON = None
try:
    from woof.core.chem_byo import unavailable_reason as _byo_reason
    EXACT_REASON = _byo_reason()
except Exception as exc:  # pragma: no cover
    EXACT_REASON = f"woof.core.chem_byo unavailable: {exc}"


@requires_gpu
@pytest.mark.skipif(EXACT_REASON is not None, reason=str(EXACT_REASON))
@pytest.mark.parametrize("sched", sorted(SCHEDULES))
def test_exact_sorgam_matches_wrf_word_for_word(sched):
    for step, pairs in _run(sched, exact=True):
        bad = [(name, int(np.sum(g != e))) for name, g, e in pairs
               if not np.array_equal(g.view(np.uint32), e.view(np.uint32))]
        assert not bad, f"schedule {sched} step {step}: words differ in {bad}"


@requires_gpu
@pytest.mark.parametrize("sched", sorted(SCHEDULES))
def test_default_sorgam_within_tolerance_of_wrf(sched):
    """The clean-room solver against WRF's: every row within TOL of its own
    largest value in the case."""
    for step, pairs in _run(sched, exact=False):
        for name, g, e in pairs:
            scale = max(float(np.abs(e).max()), 1.e-30)
            worst = float(np.abs(g.astype(np.float64) - e).max()) / scale
            assert worst <= TOL[name], (
                f"schedule {sched} step {step}: {name} off by {worst:.3e} of its scale "
                f"(tolerance {TOL[name]:.1e})")


def test_generated_block_is_fresh_against_the_pinned_source():
    tree = Path(os.environ.get("WOOF_WRF461_STOCK_TREE", "wrf-stock-v461-gate-20260721"))
    if not (tree / "chem" / "module_aerosols_sorgam.F").is_file():  # pragma: no cover
        pytest.skip(f"stock WRF v4.6.1 tree not present at {tree}")
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "tools/chem_sorgam/transcribe_sorgam.py"),
                    str(tree), "--check"], check=True)
