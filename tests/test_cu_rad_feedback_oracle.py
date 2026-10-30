"""The cumulus cloud handed to radiation, graded against WRF v4.7.1's lines.

``module_radiation_driver.F`` modifies the radiation's QC, QI and CLDFRA
for ``cu_rad_feedback`` (ICLOUD_CU 1 for Grell-Freitas/Grell 3D, 2 for
Kain-Fritsch) and for ``shcu_physics = 5`` (Deng).  The oracle
(tools/cu_rad_feedback_oracle) compiles those lines verbatim -- with
cal_cldfra1 and cal_cldfra2 between them, in the driver's order -- and runs
48 cases: ICLOUD 1/2 x ICLOUD_CU 0/1/2 x Deng off/on x four moisture arms.

* ``test_the_radiation_cloud_matches_wrf_word_for_word``: every CLDFRA, QC
  and QI word of WOOF's sequence (the functions both RRTMG adapters call)
  equal to WRF's, on the card.
* ``test_the_merges_are_not_vacuous``: the cases move the cloud -- each
  merge changes words, so the gate above can fail.
* the CPU gates pin the fixture and the doors (config, namelist).
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import ConfigurationRefused, RunConfig, validate_run_config

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "oracles" / "cu_rad_feedback"
CASES = FIXTURE / "rad-cloud-cases.bin.gz"
ORACLE = FIXTURE / "rad-cloud-O0.out.gz"
SHA256 = {
    "rad-cloud-cases.bin.gz":
        "9b613b87809ff2a9ad9da7e19254aec2666551dae5f12418bb500f4d12f1e09d",
    "rad-cloud-O0.out.gz":
        "73ea54a8198e4e8189405e6bcc9ba2aa5f078cb38baa48bd136e648f099c3696",
}
RAW = {"cases": "590a823be6e03e26a4bddb5e583b731bced027411d6d1d989e1ac6888e0f575c",
       "oracle": "0709fc347f0e2f8bcd4f66064100a87860d1cf7933dc49dfe3c8e52eab8aa00f"}
RADIATION_DRIVER_SHA256 = (
    "779875651512709c47d50ef6c1e1216ad94726f1996c5e45a0db8eb800733ffa")


def _tools():
    from tools.cu_rad_feedback_oracle import compare_woof, rad_cloud_cases
    return rad_cloud_cases, compare_woof


def test_the_fixture_is_the_recorded_oracle_output():
    for name, digest in SHA256.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(gzip.decompress(CASES.read_bytes())).hexdigest() \
        == RAW["cases"]
    assert hashlib.sha256(gzip.decompress(ORACLE.read_bytes())).hexdigest() \
        == RAW["oracle"]
    receipt = (FIXTURE / "receipt.txt").read_text(encoding="utf-8")
    assert f"{RADIATION_DRIVER_SHA256}  wrf-4.7.1/phys/module_radiation_driver.F" \
        in receipt
    assert "-ffp-contract=off" in receipt and "expf powf" in receipt
    gen = (ROOT / "tools/cu_rad_feedback_oracle/gen_slices.py").read_text(
        encoding="utf-8")
    assert RADIATION_DRIVER_SHA256 in gen
    R, _ = _tools()
    cases = R.read_input(CASES)
    assert len(cases) == 48
    assert {(c["icloud"], c["icloud_cu"], c["shcu"]) for c in cases} == {
        (a, b, s) for a in (1, 2) for b in (0, 1, 2) for s in (0, 5)}
    assert {c["mp"] for c in cases} == {1, 5, 8, 50}


def test_the_merges_are_not_vacuous():
    """Each merge changes words relative to the same case without it."""
    R, _ = _tools()
    cases = R.read_input(CASES)
    outs = R.read_output(ORACLE, cases)
    by_key = {(c["icloud"], c["icloud_cu"], c["shcu"], c["mp"]): (c, o)
              for c, o in zip(cases, outs)}
    for (icloud, cu, shcu, mp), (case, out) in by_key.items():
        if cu == 2 and icloud == 1 and shcu == 0:
            # the KF merge raises the cloud fraction where CLDFRA_CU > 0
            moved = (case["cldfra_dp"] + case["cldfra_sh"]) > 0
            assert np.any(out["cldfra"][moved] > 0)
        if shcu == 5:
            assert np.all(out["cldfra"] >= case["cldfra_sh"])
        if cu == 1:
            assert np.all(out["qc"] >= case["qc"])


def test_feedback_doors():
    base = dict(nx=8, ny=8, nz=30, dx=3000.0, dy=3000.0, ztop=18000.0,
                dt=20.0, run_seconds=60.0, moist=True, mp_physics=8,
                bl_pbl_physics=1, sf_sfclay_physics=91, ra_physics=4)
    validate_run_config(RunConfig(**base, cu_physics=1, cudt_minutes=5.0,
                                  cu_rad_feedback=True))
    validate_run_config(RunConfig(**base, cu_physics=5, cudt_minutes=0.0,
                                  cu_diag=1, cu_rad_feedback=True))
    validate_run_config(RunConfig(**base, cu_physics=3, cudt_minutes=0.0,
                                  cu_diag=1, cu_rad_feedback=True))
    for bad, fragment in (
            (dict(cu_physics=16, cudt_minutes=0.0), "ICLOUD_CU"),
            (dict(cu_physics=5, cudt_minutes=0.0, cu_diag=0), "cu_diag = 1"),
            (dict(cu_physics=3, cudt_minutes=0.0), "cu_diag = 1")):
        with pytest.raises(ConfigurationRefused) as caught:
            validate_run_config(RunConfig(**base, cu_rad_feedback=True, **bad))
        assert fragment in str(caught.value)
    with pytest.raises(ConfigurationRefused, match="RRTMG"):
        validate_run_config(RunConfig(**{**base, "ra_physics": 1},
                                      cu_physics=1, cudt_minutes=5.0,
                                      cu_rad_feedback=True))


def test_the_namelist_importer_carries_feedback(tmp_path):
    from test_namelist_import import _import_with
    toml, report = _import_with(
        tmp_path, extra_physics=" cu_rad_feedback = .true., .false.,\n")
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert fixed[("physics", "cu_rad_feedback")].fixed_value == [True, False]
    assert "cu_rad_feedback = true" in toml
    with pytest.raises(ValueError, match="cu_rad_feedback"):
        _import_with(tmp_path, extra_physics=" cu_rad_feedback = .false., .true.,\n")


@pytest.mark.gpu
@requires_gpu
def test_the_radiation_cloud_matches_wrf_word_for_word():
    _, compare = _tools()
    rows = compare.run(str(CASES), str(ORACLE))
    bad = [r for r in rows if r["bit_unequal"]]
    assert not bad, bad[:10]
    assert sum(r["words"] for r in rows) == 165_888
    assert max(r["max_ulp"] for r in rows) == 0
