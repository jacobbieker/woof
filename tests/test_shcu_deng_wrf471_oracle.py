"""Deng shallow cumulus (shcu_physics=5) graded word for word against WRF v4.7.1.

The fixture under tests/data/oracles/shcu_deng is the output of the
byte-unmodified ``phys/module_shcu_deng.F`` driven through the
byte-unmodified ``phys/module_shallowcu_driver.F`` (gfortran 16.2 -O0
-ffp-contract=off, float32 libm pinned to WOOF's glibc 2.39 transcription by
tools/shcu_deng_wrf471_oracle/libm_shim.cpp) on 400 columns in eight tiles,
each run for 4-6 consecutive steps with the Deng state carried by the
Fortran: warm/cold/saturated/fog/stable/dry/highland/deep soundings, MYJ and
MYNN TKE, adaptive stepping, dx 1-27 km, dt 10-90 s, the 6 AM PBLMAX reset,
and injected NBC/updraft histories so the dissipation code (edge
evaporation, in-cloud and exposed-base diffusion, CTEI, drizzle, ice
settling) and the CLDDPTHB >= 4 km downdraft arm run from the first step.
tools/shcu_deng_wrf471_oracle builds and runs it; its README has the recipe.

The breakage each gate prevents, named:

* ``test_the_device_kernel_matches_wrf_word_for_word`` -- any drift of
  kernels/shcu_deng.cu, of the launcher's state carry or of the compile
  route from WRF's arithmetic, on every output word of every step.
* ``test_cuda_libm_is_what_the_glibc_words_prevent`` -- the libm pin: the
  same unit on the platform's own expf/logf/powf leaves the oracle, so the
  gate above can fail and the glibc transcription is load-bearing.
* the CPU gates pin the fixture's identity, its coverage, the workspace
  slot table and the oracle receipts, so the device gate cannot pass on a
  shrunken or regenerated set.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "oracles" / "shcu_deng"
COLUMNS = FIXTURE / "columns-structured.bin.gz"
META = FIXTURE / "columns-structured.json"
ORACLE = FIXTURE / "oracle-structured-O0.out.gz"
KERNEL = ROOT / "woof" / "core" / "kernels" / "shcu_deng.cu"

#: The fixture's bytes as tools/shcu_deng_wrf471_oracle wrote them.
FIXTURE_SHA256 = {
    "columns-structured.bin.gz":
        "9a441d158b6502e6d0c6d9eda41643588981f40a8187dbcfe2d947383d925388",
    "columns-structured.json":
        "1cccd8d1a7f62b36fd002f878fb9f697ff525f83a98c3ab11f14c877f1dc0d9f",
    "oracle-structured-O0.out.gz":
        "0538f6541a9a4e69f021889987566621ee3b432b05f30f54c7a234fb434b4156",
}
#: The uncompressed streams (what the Fortran read and wrote).  The SNaN
#: build (every local REAL a signalling NaN at entry) wrote the same bytes.
RAW_SHA256 = {
    "input": "f32249cff2df92309e002977bbabdc7d1b964a979f315f83250f1cc79fd89e21",
    "oracle": "9d40450ce909c5eac45321df733e691a92f04ac7b10022852d644e4322f5b0fc",
}
#: The pinned WRF v4.7.1 sources the oracle refuses to build without.
WRF_SHA256 = {
    "phys/module_shcu_deng.F":
        "d03fd43559e1562bde415eb7a50906b993c6da3f5dc3e501f957b48031629645",
    "phys/module_shallowcu_driver.F":
        "068cf22b8c93cfac74a6223a72d61f4a2f890bc44e212da5d2c2a131c5ae271a",
}


def _harness():
    from tools.shcu_deng_wrf471_oracle import compare_woof, deng_columns
    return deng_columns, compare_woof


def test_the_fixture_is_the_recorded_oracle_output():
    import gzip
    for name, digest in FIXTURE_SHA256.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() \
            == digest, name
    assert hashlib.sha256(gzip.decompress(COLUMNS.read_bytes())).hexdigest() \
        == RAW_SHA256["input"]
    assert hashlib.sha256(gzip.decompress(ORACLE.read_bytes())).hexdigest() \
        == RAW_SHA256["oracle"]
    readme = (FIXTURE / "README.md").read_text(encoding="utf-8")
    for digest in list(FIXTURE_SHA256.values()) + list(RAW_SHA256.values()):
        assert digest in readme
    build = (ROOT / "tools" / "shcu_deng_wrf471_oracle" / "build.sh").read_text(
        encoding="utf-8")
    sums = (FIXTURE / "oracle-sha256sums.txt").read_text(encoding="utf-8")
    for path, digest in WRF_SHA256.items():
        assert digest in build
        assert f"{digest}  wrf-4.7.1/{path}" in sums
    compiler = (FIXTURE / "compiler.txt").read_text(encoding="utf-8")
    assert "GNU Fortran" in compiler and "-O0" in compiler
    assert "-ffp-contract=off" in compiler
    # The oracle's float32 libm is the shim's, exactly these symbols.
    assert re.findall(r"^\s+_?(\w+)$", compiler, flags=re.M) == [
        "expf", "logf", "powf", "sinf"]


def test_the_fixture_covers_the_scheme():
    D, _ = _harness()
    cases = D.read_input(COLUMNS)
    outs = D.read_output(ORACLE)
    meta = json.loads(META.read_text(encoding="utf-8"))
    assert len(cases) == len(outs) == 8
    assert sum(c["ncol"] for c in cases) == len(meta) == 400
    assert {c["bl_pbl"] for c in cases} == {2, 5}
    assert any(c["adapt"] for c in cases)
    assert {c["nz"] for c in cases} >= {32, 35, 40, 45, 50, 60}
    regimes = Counter(m["regime"] for m in meta)
    assert set(regimes) == {"shallow", "deep", "saturated", "stable", "dry",
                            "highland", "cold", "fog", "tiny"}
    steps = [s for c in outs for s in c["steps"]]
    assert len(steps) == 38
    assert not any(s["fatal"] or s["messages"] for s in steps)
    # Every arm is graded somewhere: precipitating updrafts, deep (>= 4 km)
    # clouds, NBC present, a cold NBC layer, the saturated-grid conversion.
    rain = sum(int((s["rainsh"] > 0).sum()) for s in steps)
    deep = sum(int((s["clddpthb"] >= 4000.0).sum()) for s in steps)
    nbc = sum(int((s["cldareab"] > 0).any(axis=0).sum()) for s in steps)
    assert rain >= 150 and deep >= 60 and nbc >= 800, (rain, deep, nbc)
    # No sentinel survives: every INTENT(OUT) word was written.
    for s in steps:
        for k in ("cldfra_sh", "ca_rad", "cw_rad"):
            assert not (s[k] == -999.0).any()
    receipt = (FIXTURE / "coverage-never-executed.txt").read_text(
        encoding="utf-8")
    assert receipt.startswith("executed lines 1868, never-executed")


def test_the_workspace_slot_table_is_consistent():
    from woof.core import shcu_deng
    src = KERNEL.read_text(encoding="utf-8")
    slots = int(re.search(r"#define DENGWS_SLOTS (\d+)", src).group(1))
    assert slots == shcu_deng.DENGWS_SLOTS
    ids = [int(m) for m in re.findall(r"DENG_SLOT\(\w+, (\d+)\)", src)]
    ids += [int(m) for m in re.findall(r"DWS\(wsb, (\d+), kp\)", src)]
    assert len(ids) == len(set(ids)) == slots
    assert sorted(ids) == list(range(slots))


def test_the_ulp_metric_counts_what_it_claims():
    _, compare_woof = _harness()
    one = np.float32(1.0)
    up = np.nextafter(one, np.float32(2.0))
    assert compare_woof.ulp_distance([one], [up])[0] == 1
    assert compare_woof.ulp_distance([np.float32(0.0)],
                                     [np.float32(-0.0)])[0] == 0


@pytest.mark.gpu
@requires_gpu
def test_the_device_kernel_matches_wrf_word_for_word():
    _, compare_woof = _harness()
    rows = compare_woof.run(str(COLUMNS), str(ORACLE))
    summary = compare_woof.summarize(rows)
    bad = [r for r in rows if r["off"] or r["bit_unequal"]]
    assert summary["off"] == 0, json.dumps(bad[:20], indent=1)
    assert summary["bit_unequal"] == 0, json.dumps(bad[:20], indent=1)
    assert summary["max_ulp"] == 0
    assert summary["status_columns"] == 0
    assert summary["words"] == 1_737_808


@pytest.mark.gpu
@requires_gpu
def test_cuda_libm_is_what_the_glibc_words_prevent(monkeypatch):
    import cupy as cp
    from woof.core import shcu_deng
    from woof.core.kernels import module_source

    marker = "/* gfk_exp/gfk_log/gfk_pow: glibc_flt32.cuh, prepended"
    source = module_source("shcu_deng")
    assert marker in source
    platform = source.replace(marker, (
        "#define gfk_exp(x) expf(x)\n#define gfk_log(x) logf(x)\n"
        "#define gfk_pow(x, y) powf(x, y)\n" + marker))
    module = cp.RawModule(code=platform,
                          options=shcu_deng.DENG_MODULE_OPTIONS)
    monkeypatch.setattr(shcu_deng, "_deng_function", module.get_function)
    _, compare_woof = _harness()
    summary = compare_woof.summarize(
        compare_woof.run(str(COLUMNS), str(ORACLE)))
    assert summary["off"] > 0, (
        "the platform libm matched WRF on every word: the glibc pin would "
        "then be carrying nothing, or the control is not substituting")
