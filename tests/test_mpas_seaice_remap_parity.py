"""MPAS-Seaice's incremental remap, word for word against v8.4.2 at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's ``mpas_seaice_advection_incremental_remap.F``
and its tracers module, compiled verbatim with gfortran -O0
-ffp-contract=off at RKIND 4 and 8 and driven by
``tools/mpas_seaice_oracle/run_incremental_remap.F90`` on planar hexagon and
quadrilateral blocks (with land and halo layers). The init
(``seaice_init_advection_incremental_remap``) runs verbatim; its geometry is
the GPU remap's input. The outputs are committed
(tests/data/oracles/mpas_seaice_remap, PROVENANCE.md). When ``GFORTRAN`` (or
gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` are set on an x86-64 host, the
fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A tracer list in another order or with other parents than v8.4.2 builds
   from the packages (the remap needs parents before children; a parent the
   packages leave out leaves its child a mass-like field).
2. A volume/thickness conversion that drifts by a word, or treats an
   ice-free category as anything but zero thickness.
3. Departure triangles in the wrong cell, with the wrong sign, or with
   quadrature points summed in another order.
4. A reconstruction without the limiter, or a barycenter whose terms are
   summed in another order.
5. A gate that cannot fail: each of the above, mutated, must be caught.
6. Upwind advection run silently: v8.4.2 cannot run it (refused by name).
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
from woof.hex.errors import ConfigurationRefusal
from woof.verify import mpas_seaice_remap_oracle as ro

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (ro.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": ro.input_digest(case) for p in BOTH for case in ro.cases(p)}


def test_fixture_covers_the_branches():
    logs = ro.load_logs()
    cases = {case["name"]: case for case in ro.cases("r8")}
    assert set(logs) == set(cases)
    trees = [log["r8"]["tree"] for log in logs.values()]
    assert max(n for tree in trees for _, _, n, _ in tree) == 3
    assert any(has_child and n == 2 for tree in trees for _, _, n, has_child in tree)
    # a parent the packages leave out: pondArea under level ponds without level ice
    assert ["pondArea", "levelIceArea", 0, True] in logs["tree_inactive_leaf"]["r8"]["tree"]
    assert any(case["absent"] for case in cases.values())
    assert {case["mesh_kind"] for case in cases.values()} == {"hex", "quad"}
    for precision in BOTH:
        fixture = ro.load_fixture(precision)
        for name in ("convert_hex",):
            assert f"tracers/iceVolumeCategory" in fixture[name]
            assert f"tracers/snowVolumeCategory" in fixture[name]
            assert "tracers/iceAreaCategory" not in fixture[name]
        # the round trip changes some volumes by a rounding, not all
        changed = fixture["convert_quad_round_trip"].get("tracers/iceVolumeCategory")
        case = next(c for c in ro.cases(precision) if c["name"] == "convert_quad_round_trip")
        if changed is not None:
            assert ro.word_mismatches(changed, case["pools"]["tracers"]["iceVolumeCategory"]) < changed.size // 2


def test_the_tracer_tree_is_the_oracles():
    logs = ro.load_logs()
    for case in ro.cases("r8"):
        for precision in BOTH:
            assert ro.tree_rows(ro.gpu_tree(case)) == logs[case["name"]][precision]["tree"], case["name"]


def test_a_mutated_tree_is_caught(monkeypatch):
    from woof.hex.mpas_seaice import remap_v842 as rv
    logs = ro.load_logs()
    case = next(c for c in ro.cases("r8") if c["name"] == "tree_column_tracers")
    # levels in reverse table order within a level: the oracle keeps the table order
    table = list(rv.TRACER_TABLE)
    table[3], table[8] = table[8], table[3]
    monkeypatch.setattr(rv, "TRACER_TABLE", tuple(table))
    assert ro.tree_rows(ro.gpu_tree(case)) != logs[case["name"]]["r8"]["tree"]


def test_an_inactive_parent_with_an_active_child_is_refused():
    from woof.hex.mpas_seaice import remap_v842 as rv
    packages = {"pkgColumnTracerPondsActive": True}
    with pytest.raises(ConfigurationRefusal, match="inactive parent"):
        rv.tracer_tree(packages, {"config_use_level_meltponds": False}, lambda name: name != "pondArea")


@pytest.mark.parametrize("knob, value", [
    ("config_advection_type", "upwind"), ("mesh:on_a_sphere", True), ("config_use_halo_exch", True),
    ("config_conservation_check", True), ("config_monotonicity_check", True),
])
def test_what_the_remap_does_not_carry_is_refused_by_name(knob, value):
    from woof.hex.mpas_seaice import remap_v842 as rv
    config = ro.cases("r8")[-1]["config"]
    rv.check_config(config)
    with pytest.raises(ConfigurationRefusal, match=knob.split(":")[-1]):
        rv.check_config(dict(config, **{knob: value}))


def test_upwind_is_refused_with_its_reason():
    from woof.hex.mpas_seaice import remap_v842 as rv
    with pytest.raises(ConfigurationRefusal, match="undeclared"):
        rv.check_config({"config_advection_type": "upwind"})


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = ro.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("mutation", ["divide_ice_free", "skip_snow"])
def test_a_mutated_conversion_is_caught(monkeypatch, mutation):
    from woof.hex.mpas_seaice import remap_v842 as rv
    case = next(c for c in ro.cases("r8") if c["name"] == "convert_hex")
    fixture = ro.load_fixture("r8")[case["name"]]
    if mutation == "divide_ice_free":
        source = rv.module_source

        def mutated(precision="r4"):
            return source(precision).replace("if (area[i] > C(0.0)) {", "if (area[i] >= C(0.0)) {")
        monkeypatch.setattr(rv, "module_source", mutated)
        rv._module_on.cache_clear()
        field = "tracers/iceVolumeCategory"
    else:
        original = rv._conversion

        def no_snow(tree, tracers, remap, kernel):
            original([t for t in tree if t.name != "snowVolumeCategory"], tracers, remap, kernel)
        monkeypatch.setattr(rv, "_conversion", no_snow)
        field = "tracers/snowVolumeCategory"
    try:
        got = ro.run_gpu(case, fixture)
    finally:
        rv._module_on.cache_clear()
    assert ro.word_mismatches(got[field], fixture[field]) > 0


def _oracle_toolchain() -> tuple[str, Path] | None:
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
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_remap_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    assert (tmp_path / "fixture" / "remap-logs.json").read_text(encoding="utf-8") == \
        (ro.FIXTURE_DIR / "remap-logs.json").read_text(encoding="utf-8")
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(ro.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert ro.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)


def _mutated_source(monkeypatch, old: str, new: str) -> None:
    from woof.hex.mpas_seaice import remap_v842 as rv
    source = rv.module_source
    assert old in source("r8"), old

    def mutated(precision="r4"):
        return source(precision).replace(old, new)
    monkeypatch.setattr(rv, "module_source", mutated)
    rv._module_on.cache_clear()


def _mismatches_after(case_name: str, field: str, precision: str = "r8") -> int:
    from woof.hex.mpas_seaice import remap_v842 as rv
    case = next(c for c in ro.cases(precision) if c["name"] == case_name)
    fixture = ro.load_fixture(precision)[case_name]
    try:
        got = ro.run_gpu(case, fixture)
    finally:
        rv._module_on.cache_clear()
    return ro.word_mismatches(got[field], ro.expected_outputs(case, fixture)[field])


def test_the_departure_cases_cover_the_branches():
    most, missing = 0, False
    for precision in BOTH:
        fixture = ro.load_fixture(precision)
        for case in ro.cases(precision):
            if case["stage"] != 2:
                continue
            out = fixture[case["name"]]
            ne, nc = case["dims"]["nEdges"], case["dims"]["nCells"]
            cells = out["incremental_remap/iCellTriangle"][:ne]
            remap_cells = out["incremental_remap/cellsOnEdgeRemap"][:ne]
            counts = np.count_nonzero(cells, axis=1)
            most = max(most, int(counts.max()))
            assert counts.max() >= 3 and (counts == 2).any() and (counts == 0).any()
            # triangles in side cells, missing ones among them, and both signs of flux
            side = [(cells[e, t] == remap_cells[e, 2:]).any() for e in range(ne) for t in range(6) if cells[e, t]]
            assert sum(side) > 10
            missing = missing or bool((cells == nc + 1).any())
            area = out["incremental_remap/triangleArea"][:ne]
            assert (area > 0).any() and (area < 0).any()
            if case["mesh_kind"] == "quad":
                assert any((cells[e] == remap_cells[e, 4]).any() or (cells[e] == remap_cells[e, 5]).any()
                           for e in range(ne) if cells[e].any())
    assert most >= 4 and missing


@requires_gpu
@pytest.mark.parametrize("mutation", ["side_sign", "quadrature_order"])
def test_a_mutated_departure_geometry_is_caught(monkeypatch, mutation):
    if mutation == "side_sign":
        _mutated_source(monkeypatch, "sign[t] = side == 0 ? 1 : -1;", "sign[t] = 1;")
        field = "incremental_remap/triangleArea"
    else:
        _mutated_source(monkeypatch, "x[0] = A(A(M(IR_Q1, x1), M(IR_Q1, x2)), M(IR_Q2, x3));",
                        "x[0] = A(M(IR_Q1, x1), A(M(IR_Q1, x2), M(IR_Q2, x3)));")
        field = "incremental_remap/xTriangle"
    assert _mismatches_after("departure_quad_fast", field) > 0


@requires_gpu
@pytest.mark.parametrize("mutation", ["no_limiter", "barycenter_order"])
def test_a_mutated_reconstruction_is_caught(monkeypatch, mutation):
    if mutation == "no_limiter":
        _mutated_source(monkeypatch, "gradFactor = ir_max(C(0.0), S(gradFactor, IR_EPS11));", "gradFactor = C(1.0);")
        field = "tracer_reconstruction/iceVolumeCategoryGradx"
    else:
        _mutated_source(monkeypatch, "*xb = M(A(A(M(c0, g[0]), M(cx, g[2])), M(cy, g[3])), reciprocal);",
                        "*xb = M(A(M(c0, g[0]), A(M(cx, g[2]), M(cy, g[3]))), reciprocal);")
        field = "tracer_barycenter/iceAreaCategoryBarycenterx"
    assert _mismatches_after("reconstruct_quad_ponds", field) > 0
