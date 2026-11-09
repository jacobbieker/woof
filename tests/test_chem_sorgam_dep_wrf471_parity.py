"""MADE/SORGAM dry deposition and PM sums against WRF v4.7.1, word for word.

tools/chem_wrf471_oracle/sorgam/run_sorgam_dep.F90 runs the byte-copied
sorgam_depdriver (VDVG_2, aer_drydep_opt = 1, and VDVG, = 11) on 16 columns
covering the friction-velocity floor, stable/neutral/unstable surface
layers, smooth and rough surfaces, shallow and deep boundary layers and a
wide range of aerodynamic resistance, and sum_pm_sorgam on 4 levels.
Every deposition velocity and PM sum must equal the Fortran's float32 word.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify.chem_oracle import load

ROOT = Path(__file__).resolve().parents[1] / "tests/data/oracles/chem/sorgam_dep"
GAS_ROW = {"sulf": 3, "hno3": 7, "nh3": 16, "hcl": 41}
AUX_IN = ("h2oaj", "h2oai", "cvaro1", "cvaro2", "cvalk1", "cvole1")


def _engine(a):
    return np.ascontiguousarray(np.transpose(a, (1, 2, 0)), dtype=np.float32)


def _plane(a):
    return np.ascontiguousarray(a.T, dtype=np.float32)


def _inputs():
    import cupy as cp
    from woof.core.chem_sorgam import AEROSOLS, AUX, GASES
    cases = load(ROOT)
    init = cases["init"]
    chem = init["chem"]
    aer = [cp.asarray(_engine(chem[:, :, :, 43 + i])) for i in range(len(AEROSOLS))]
    gas = [cp.asarray(_engine(chem[:, :, :, GAS_ROW[g] - 1])) for g in GASES]
    zeros = np.zeros_like(_engine(init["t_phy"]))
    aux = [cp.asarray(_engine(init[n]) if n in AUX_IN else zeros) for n in AUX]
    return cases, init, aer, gas, aux


def test_fixture_checksums_and_coverage():
    for line in (ROOT / "oracle-sha256sums.txt").read_text().splitlines():
        digest, rel = line.split()
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    cases = load(ROOT)
    init = cases["init"]
    assert np.any(init["ust"] < 0.1) and np.any(init["rmol"] > 0) and np.any(init["rmol"] < 0)
    assert not np.array_equal(cases["drydep_opt1"]["vgsa"], cases["drydep_opt11"]["vgsa"])
    assert np.array_equal(cases["drydep_opt1"]["aer_res"], init["aer_res"])


@requires_gpu
@pytest.mark.parametrize("opt", [1, 11])
def test_drydep_matches_wrf_word_for_word(opt):
    import cupy as cp
    from woof.core.chem_sorgam import AEROSOLS, launch_sorgam_drydep
    cases, init, aer, gas, aux = _inputs()
    f3 = {n: cp.asarray(_engine(init[n])) for n in ("t_phy", "p_phy", "alt")}
    f2 = {n: cp.asarray(_plane(init[n])) for n in ("ust", "rmol", "znt", "pbl", "aer_res")}
    vgsa = launch_sorgam_drydep(aer, gas, aux, f3["t_phy"], f3["p_phy"], f3["alt"],
                                f2["ust"], f2["rmol"], f2["znt"], f2["pbl"], f2["aer_res"],
                                vdvg_2=(opt == 1))
    from woof.core.chem_sorgam import _AER_V
    want = cases[f"drydep_opt{opt}"]["vgsa"]
    bad = []
    for i, name in enumerate(AEROSOLS):
        exp = _plane(want[:, :, _AER_V[i] - 1])
        got = vgsa[i].get()
        if not np.array_equal(got.view(np.uint32), exp.view(np.uint32)):
            bad.append((name, int(np.sum(got != exp))))
    assert not bad, f"aer_drydep_opt = {opt}: words differ in {bad}"


@requires_gpu
def test_sum_pm_matches_wrf_word_for_word():
    import cupy as cp
    from woof.core.chem_sorgam import launch_sorgam_sum_pm
    cases, init, aer, gas, aux = _inputs()
    out = launch_sorgam_sum_pm(aer, aux[0], aux[1], cp.asarray(_engine(init["alt"])))
    want = cases["sum_pm"]
    for got, name in zip(out, ("pm2_5_dry", "pm2_5_water", "pm2_5_dry_ec", "pm10")):
        exp = _engine(want[name])
        g = got.get()
        assert np.array_equal(g.view(np.uint32), exp.view(np.uint32)), name
