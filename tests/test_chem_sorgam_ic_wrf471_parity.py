"""MADE/SORGAM initial conditions against WRF v4.7.1, word for word.

tools/chem_wrf471_oracle/sorgam/run_sorgam_ic.F90 runs the byte-copied
aerosols_sorgam_init past its chem_in_opt = 1 return, for aer_ic_opt = 1
(AER_IC_DEFAULT) and 101 (AER_IC_PNNL, sorgam_init_aer_ic_pnnl, whose four
height bands the 30 mass levels cross), on 8 columns.  Every aerosol row
and the gas sulfate must equal the Fortran's float32 words on all mass
levels.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify.chem_oracle import load

ROOT = Path(__file__).resolve().parents[1] / "tests/data/oracles/chem/sorgam_ic"
NZ = 30  # mass levels; the oracle's arrays run to the staggered level 31


def _engine(a):
    return np.ascontiguousarray(np.transpose(a, (1, 2, 0)), dtype=np.float32)


def test_fixture_checksums_and_coverage():
    for line in (ROOT / "oracle-sha256sums.txt").read_text().splitlines():
        digest, rel = line.split()
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    pnnl = load(ROOT)["aer_ic_opt101"]
    zw = pnnl["z_at_w"]
    mid = 0.5 * (zw[:, :NZ, :] + zw[:, 1:NZ + 1, :])
    for lo, hi in ((0, 500), (500, 1000), (1000, 5000), (5000, 1e9)):
        assert np.any((mid > lo) & (mid <= hi))


@requires_gpu
@pytest.mark.parametrize("opt", [1, 101])
def test_sorgam_ic_matches_wrf_word_for_word(opt):
    import cupy as cp
    from woof.core.chem_sorgam import AEROSOLS, launch_sorgam_ic
    case = load(ROOT)[f"aer_ic_opt{opt}"]
    shape = _engine(case["sulf0"][:, :NZ, :]).shape
    aer = [cp.full(shape, 7., dtype=cp.float32) for _ in AEROSOLS]
    sulf = cp.asarray(_engine(case["sulf0"][:, :NZ, :]))
    launch_sorgam_ic(aer, sulf, cp.asarray(_engine(case["convfac"][:, :NZ, :])),
                     cp.asarray(_engine(case["z_at_w"])), aer_ic_opt=opt)
    want = case["chem"]
    bad = []
    for i, name in enumerate(AEROSOLS):
        exp = _engine(want[:, :NZ, :, 43 + i]); got = aer[i].get()
        if not np.array_equal(got.view(np.uint32), exp.view(np.uint32)):
            bad.append((name, int(np.sum(got != exp))))
    exp = _engine(want[:, :NZ, :, 2]); got = sulf.get()
    if not np.array_equal(got.view(np.uint32), exp.view(np.uint32)):
        bad.append(("sulf", int(np.sum(got != exp))))
    assert not bad, f"aer_ic_opt = {opt}: words differ in {bad}"
