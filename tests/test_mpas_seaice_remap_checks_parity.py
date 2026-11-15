"""MPAS-Seaice's incremental-remap conservation and monotonicity checks, word for word against v8.4.2.

The reference is MPAS-Model v8.4.2's ``seaice_run_advection_incremental_remap``
with ``config_conservation_check`` and ``config_monotonicity_check`` on,
compiled verbatim with gfortran -O0 -ffp-contract=off at RKIND 4 and 8 and
driven by ``tools/mpas_seaice_oracle/run_incremental_remap.F90`` (stage 4) on
planar and spherical blocks, two of them variable-resolution
(``woof.verify.mpas_seaice_remap_checks_oracle``). Graded: every tracer, the
sums before and after the update (``<tracer>Cons``, ``Cons@2``), the local
extremes (``<tracer>LocalMin``/``LocalMax``, extended in place), each
MPAS_LOG_ERR message with its arguments, and the error code. When ``GFORTRAN``
and ``WOOF_MPAS_V842_SOURCE`` are set on an x86-64 host, the fixture is
rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. Sums taken over the halo cells too.
2. Local extremes from the wrong masks, or extended in another order (v8.4.2
   extends cell after cell, in place, so an extension carries on).
3. A failed check that logs other words, other indices or another code, or
   that converts the thicknesses back to volume (v8.4.2 returns first).
4. A gate that cannot fail: each of the above, mutated, must be caught.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify import mpas_seaice_remap_checks_oracle as rco
from woof.verify import mpas_seaice_remap_oracle as ro

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in (rco.FIXTURE_DIR / "inputs.sha256").read_text(
        encoding="utf-8").splitlines())
    assert recorded == {f"{c['name']}/{p}": ro.input_digest(c) for p in BOTH for c in rco.cases(p)}


def test_the_cases_cover_planar_spherical_and_variable_resolution_blocks():
    rows = rco.cases("r8")
    assert {(c["mesh_kind"], bool(c.get("sphere"))) for c in rows} == {("hex", False), ("quad", False),
                                                                       ("hex", True), ("quad", True)}
    varres = [c for c in rows if c.get("varres")]
    assert len(varres) == 2
    for case in varres:
        areas = case["pools"]["mesh"]["areaCell"][:case["dims"]["nCells"]]
        assert areas.max() / areas.min() > 2.5
    assert all(c["config"]["config_conservation_check"] or c["config"]["config_monotonicity_check"] for c in rows)


def test_the_fixture_covers_each_outcome():
    logs = rco.load_logs()
    assert set(logs) == {c["name"] for c in rco.cases("r8")}
    codes = {(name, p): logs[name][p]["ierr"] for name in logs for p in BOTH}
    # every outcome at both precisions: a pass, a conservation error and a monotonicity error
    assert {codes[k] for k in codes if k[1] == "r8"} == {0, 22, 23}
    assert {codes[k] for k in codes if k[1] == "r4"} == {22, 23}
    messages = [e["message"] for name in logs for p in BOTH for e in logs[name][p].get("errors", [])]
    assert any(m.startswith("IR: Tracer conservation error: tracer, category =") for m in messages)
    assert "IR advection, monotonicity violation (max, nDims == 2)" in messages
    assert "IR advection, monotonicity violation (min, nDims == 3)" in messages
    assert "IR advection, monotonicity violation (max, nDims == 3)" in messages
    for precision in BOTH:
        fixture = rco.load_fixture(precision)
        for case in rco.cases(precision):
            out, name = fixture[case["name"]], case["name"]
            if case["config"]["config_conservation_check"]:
                assert "tracer_conservation/iceAreaCategoryCons@2" in out, name
            if case["config"]["config_monotonicity_check"]:
                assert "tracer_monotonicity/iceEnthalpyLocalMin" in out, name


def test_the_checks_are_no_longer_refused():
    from woof.hex.mpas_seaice import remap_v842 as rv
    for case in rco.cases("r8"):
        rv.check_config(case["config"])


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = rco.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
def test_a_failed_check_stops_before_the_volumes_return():
    from woof.hex.mpas_seaice import remap_v842 as rv
    case = next(c for c in rco.cases("r8") if c["name"] == "checks_hex_conservation")
    fixture = rco.load_fixture("r8")[case["name"]]
    dev = ro.on_device(case, fixture)
    with pytest.raises(rv.RemapCheckError) as failed:
        rv.run(ro.gpu_tree(case), pools=dev, mesh=dev["mesh"], configs=case["config"])
    assert failed.value.ierr == rv.SEAICE_ERROR_IR_CONS
    assert ro.warning_mismatches(failed.value.log, rco.load_logs()[case["name"]]["r8"]["errors"], "r8") == 0


def _mutated_source(monkeypatch, old: str, new: str) -> None:
    from woof.hex.mpas_seaice import remap_v842 as rv
    source = rv.module_source
    assert source("r8").count(old) == 1, old

    def mutated(precision="r4"):
        return source(precision).replace(old, new)
    monkeypatch.setattr(rv, "module_source", mutated)
    rv._module_on.cache_clear()


def _bad_words(case_name: str, precision: str) -> dict:
    from woof.hex.mpas_seaice import remap_v842 as rv
    rows = tuple(r for r in rco.CASES if r[0] == case_name)
    try:
        report = ro.measure(precision, rows=rows, fixture_dir=rco.FIXTURE_DIR)[case_name]
    finally:
        rv._module_on.cache_clear()
    return {k: v for k, v in report.items() if v}


_MUTATIONS = {
    # the sums over the halo too
    "sum_halo": ("for (int c = 0; c < nCellsSolve; ++c) s = A(s, M(areaCell[c], product[c * nValues + i]));",
                 "for (int c = 0; c <= nCellsSolve; ++c) s = A(s, M(areaCell[c], product[c * nValues + i]));",
                 "checks_hex_closed_both_recover", "r8", "tracer_conservation/iceAreaCategoryCons"),
    # the conservation threshold dropped
    "no_conservation_error": ("if (ABS(r) > IR_EPS11) fail[i] = 1;", "if (0) fail[i] = 1;",
                              "checks_sphere_hex_south_varres_closed_both", "r4", "ierr"),
    # the neighbours' masks ignored in the local extremes
    "unmasked_neighbours": ("            if (mask[j] == 1) {\n                if (field[j] < mn) mn = field[j];",
                            "            if (1) {\n                if (field[j] < mn) mn = field[j];",
                            "checks_hex_closed_both_recover", "r8", "tracer_monotonicity/iceEnthalpyLocalMin"),
    # no extension by a ring
    "no_extension": ("        if (mask[i] != 1) continue;\n        for (int k = 0; k < nEdgesOnCell[c]; ++k) {",
                     "        continue;\n        for (int k = 0; k < nEdgesOnCell[c]; ++k) {",
                     "checks_sphere_hex_north_closed_both", "r8", "tracer_monotonicity/iceEnthalpyLocalMin"),
    # the tolerance without its floor of one
    "tolerance_floor": ("const rk toleranceMax = M(IR_EPS11, ir_max(C(1.0), ABS(localMax[i])));",
                        "const rk toleranceMax = M(IR_EPS11, ABS(localMax[i]));",
                        "checks_quad_closed_ponds_both", "r8", "errors"),
}


@requires_gpu
@pytest.mark.parametrize("mutation", sorted(_MUTATIONS))
def test_a_mutated_check_is_caught(monkeypatch, mutation):
    old, new, case, precision, key = _MUTATIONS[mutation]
    _mutated_source(monkeypatch, old, new)
    assert key in _bad_words(case, precision)


@requires_gpu
def test_an_extension_swept_in_another_order_is_caught(monkeypatch):
    # v8.4.2 extends in place in cell order, so an earlier cell's extension carries on
    _mutated_source(monkeypatch, "    for (int c = 0; c < nCellsSolve; ++c) {\n        const int i = c * nValues + v;\n"
                                 "        if (mask[i] != 1) continue;",
                    "    for (int c = nCellsSolve - 1; c >= 0; --c) {\n        const int i = c * nValues + v;\n"
                    "        if (mask[i] != 1) continue;")
    assert "tracer_monotonicity/iceEnthalpyLocalMin" in _bad_words("checks_sphere_hex_north_closed_both", "r8")


@requires_gpu
def test_volumes_returned_before_a_failed_check_are_caught(monkeypatch):
    from woof.hex.mpas_seaice import remap_v842 as rv
    original = rv.check_conservation

    def late(tree, *, pools):
        rv.thickness_to_volume(tree, tracers=pools["tracers"], remap=pools["incremental_remap"])
        return original(tree, pools=pools)
    monkeypatch.setattr(rv, "check_conservation", late)
    assert "tracers/iceVolumeCategory" in _bad_words("checks_hex_conservation", "r8")


def _oracle_toolchain():
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_seaice").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MPAS-Seaice oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
@pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64"),
                    reason="the fixture is x86-64 gfortran's: arm64 gfortran's MAX/MIN keep another operand "
                           "on signed-zero ties")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    out = tmp_path / "fixture"
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_remap_fixture.py"),
                    str(tmp_path / "build"), "--checks", "--out", str(out)], check=True)
    assert (out / "remap-logs.json").read_text(encoding="utf-8") == \
        (rco.FIXTURE_DIR / "remap-logs.json").read_text(encoding="utf-8")
    for precision in BOTH:
        fresh = np.load(out / f"oracle-{precision}.npz")
        pinned = np.load(rco.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert ro.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
