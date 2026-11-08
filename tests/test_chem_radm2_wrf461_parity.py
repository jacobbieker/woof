"""RADM2 gas-phase chemistry against WRF v4.6.1's radm_driver, word for word.

The oracle (tools/chem_wrf461_oracle/radm2) compiles chem/module_radm.F and
its data modules whole and unchanged at -O0 and runs radm_driver on 12
columns x 24 levels: night and sunlit columns, clean and polluted air, the
NO/epsilc input floors, the RH clamp and the heterogeneous N2O5 switch, cold
and warm columns.  Two schedules carry species and vdrog3 across calls:
dtstep = 120 s from curr_secs = 0 (the first call runs ~240 dtcmin substeps)
and dtstep = 600 s at curr_secs = 3600 (dt60 > dtcmax, adaptive substeps).

Every species row and the 11 vdrog3 precursor increments must equal the
Fortran's float32 words; the clamp-only rows (hcl, nu0) must equal the
oracle's after the epsilc floor.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify.chem_oracle import load

ROOT = Path(__file__).resolve().parents[1] / "tests/data/oracles/chem/radm2"
# stub_radm2.F90's chem indices (1-based Fortran): p_so2 = 2 .. p_ho2 = 43,
# p_hcl = 41, p_nu0 = 44.
_FORTRAN_ROW = {name: i for i, name in enumerate(
    ("so2", "sulf", "no2", "no", "o3", "hno3", "h2o2", "ald", "hcho", "op1",
     "op2", "paa", "ora1", "ora2", "nh3", "n2o5", "no3", "pan", "hc3", "hc5",
     "hc8", "eth", "co", "ol2", "olt", "oli", "tol", "xyl", "aco3", "tpan",
     "hono", "hno4", "ket", "gly", "mgly", "dcb", "onit", "csl", "iso", "hcl",
     "ho", "ho2", "nu0"), start=2)}
# radm_driver's ph_ argument order, as run_radm2.F90 stacks them.
_PH_ARGS = ("o31d", "o33p", "no2", "no3o2", "no3o", "hno2", "hno3", "hno4",
            "h2o2", "ch2or", "ch2om", "ch3cho", "ch3coch3", "ch3coc2h5",
            "hcocho", "ch3cocho", "hcochest", "ch3o2h", "ch3coo2h", "ch3ono2",
            "hcochob")
SCHEDULES = {1: (120.0, [0.0, 120.0, 240.0]), 2: (600.0, [3600.0, 4200.0])}


def _engine(a):
    """Fortran (i, k, j) -> engine (k, j, i), contiguous float32."""
    return np.ascontiguousarray(np.transpose(a, (1, 2, 0)), dtype=np.float32)


def test_fixture_checksums_and_coverage():
    for line in (ROOT / "oracle-sha256sums.txt").read_text().splitlines():
        digest, rel = line.split()
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    cases = load(ROOT)
    assert set(cases) == {"sched1_init", "sched1_step1", "sched1_step2", "sched1_step3",
                          "sched2_init", "sched2_step1", "sched2_step2"}
    init = cases["sched1_init"]
    ph = init["ph"]
    assert np.any(ph[:, :, :, 2] == 0) and np.any(ph[:, :, :, 2] > 0)
    assert np.any(init["chem"][:, :, :, _FORTRAN_ROW["no"] - 1] == 0)
    # The chemistry moved every sunlit column's radicals.
    out = cases["sched1_step3"]["chem"]
    assert not np.array_equal(out[:, :, :, _FORTRAN_ROW["ho"] - 1],
                              init["chem"][:, :, :, _FORTRAN_ROW["ho"] - 1])
    assert np.any(cases["sched2_step2"]["vdrog3"][:, :, :, :11] > 0)
    assert not np.any(cases["sched2_step2"]["vdrog3"][:, :, :, 11:])


def test_step_limits_follow_the_driver():
    from woof.core.chem_radm2 import step_limits
    dt60, dtcmin, dtcmax = step_limits(120.0, 0.0)
    assert (dt60, dtcmax) == (np.float32(2.0), np.float32(0.0))
    assert dtcmin == np.float32(np.float32(0.5) / np.float32(60.0))
    dt60, dtcmin, dtcmax = step_limits(600.0, 3600.0)
    assert (dt60, dtcmin, dtcmax) == (np.float32(10.0), np.float32(0.05), np.float32(5.0))


@requires_gpu
@pytest.mark.parametrize("sched", sorted(SCHEDULES))
def test_radm2_matches_wrf_word_for_word(sched):
    import cupy as cp
    from woof.core.chem_radm2 import LDROG, SPECIES, launch_radm2
    cases = load(ROOT)
    init = cases[f"sched{sched}_init"]
    dtstep, clock = SCHEDULES[sched]
    assert float(init["dtstep"]) == dtstep
    chem = init["chem"]
    rows = {n: cp.asarray(_engine(chem[:, :, :, r - 1])) for n, r in _FORTRAN_ROW.items()}
    species = [rows[n] for n in SPECIES]
    clamp = [rows["hcl"], rows["nu0"]]
    t_phy = cp.asarray(_engine(init["t_phy"]))
    p_phy = cp.asarray(_engine(init["p_phy"]))
    qv = cp.asarray(_engine(init["qv"]))
    ph = {n: cp.asarray(_engine(init["ph"][:, :, :, i])) for i, n in enumerate(_PH_ARGS)}
    vdrog = [cp.zeros_like(t_phy) for _ in range(LDROG)]
    for step, curr_secs in enumerate(clock, 1):
        want = cases[f"sched{sched}_step{step}"]
        assert float(want["curr_secs_hi"]) == curr_secs
        launch_radm2(species, t_phy, p_phy, qv, ph, dtstep, curr_secs,
                     vdrog=vdrog, clamp=clamp)
        bad = []
        for n, r in _FORTRAN_ROW.items():
            got = rows[n].get()
            exp = _engine(want["chem"][:, :, :, r - 1])
            if not np.array_equal(got.view(np.uint32), exp.view(np.uint32)):
                bad.append((n, int(np.sum(got != exp))))
        for q in range(LDROG):
            got = vdrog[q].get()
            exp = _engine(want["vdrog3"][:, :, :, q])
            if not np.array_equal(got.view(np.uint32), exp.view(np.uint32)):
                bad.append((f"vdrog{q + 1}", int(np.sum(got != exp))))
        assert not bad, f"schedule {sched} step {step}: words differ in {bad}"


def test_generated_block_is_fresh_against_the_pinned_source():
    """chem_radm2.cu's arithmetic is regenerated from the pinned module_radm.F."""
    import os
    import subprocess
    import sys
    tree = Path(os.environ.get("WOOF_WRF461_STOCK_TREE", "wrf-stock-v461-gate-20260721"))
    if not (tree / "chem" / "module_radm.F").is_file():  # pragma: no cover
        pytest.skip(f"stock WRF v4.6.1 tree not present at {tree}")
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "tools/chem_radm2/transcribe_radm2.py"),
                    str(tree), "--check"], check=True)
