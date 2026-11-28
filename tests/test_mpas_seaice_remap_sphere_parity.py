"""MPAS-Seaice's incremental remap on the sphere, word for word against v8.4.2 at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's ``mpas_seaice_advection_incremental_remap.F``
(and the time step around it), compiled verbatim with gfortran -O0
-ffp-contract=off at RKIND 4 and 8 and driven by
``tools/mpas_seaice_oracle/run_incremental_remap.F90`` and
``run_seaice_step.F90`` on spherical hexagon and quadrilateral blocks
(``woof.verify.mpas_seaice_sphere_mesh``: about either pole with
``config_rotate_cartesian_grid``, at 53 degrees north without it, and
stretched to three times and more the cell area at the rim). The init runs verbatim;
its spherical geometry (local frames, tangent-plane vertices) is the GPU's
input. The outputs are committed (tests/data/oracles/mpas_seaice_remap_sphere
and mpas_seaice_step_remap_sphere, PROVENANCE.md). When ``GFORTRAN`` (or
gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` are set on an x86-64 host, the
fixtures themselves are rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A gradient left in global x/y/z, not turned by the rotated grid, or taken
   into the cell's plane through the transposed frame.
2. Departure triangles shifted into their source cell as on a plane, or
   through the wrong edge-vector basis (the side edge paired with the wrong
   other edge, the basis not reordered clockwise-first).
3. A spherical remap step, an N-step run with the velocity solver, or a
   restart that drifts from v8.4.2 by a word.
4. A gate that cannot fail: each of the above, mutated, must be caught.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
import _oracle_toolchains
from woof.hex.errors import ConfigurationRefusal
from woof.verify import mpas_seaice_remap_oracle as ro
from woof.verify import mpas_seaice_remap_sphere_oracle as rso
from woof.verify import mpas_seaice_step_remap_oracle as sr
from woof.verify import mpas_seaice_step_remap_sphere_oracle as srs

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")


def _recorded(directory: Path) -> dict:
    return dict(reversed(line.split()) for line in (directory / "inputs.sha256").read_text(encoding="utf-8")
                .splitlines())


def test_case_inputs_are_the_recorded_bytes():
    assert _recorded(rso.FIXTURE_DIR) == {f"{c['name']}/{p}": ro.input_digest(c) for p in BOTH
                                          for c in rso.cases(p)}
    assert _recorded(srs.FIXTURE_DIR) == {f"{c['name']}/{p}": sr.input_digest(c) for p in BOTH
                                          for c in srs.cases(p)}


def test_the_meshes_are_spherical_and_some_vary_in_resolution():
    for case in [*rso.cases("r8"), *srs.cases("r8")]:
        mesh = case["pools"]["mesh"]
        nc, nv = case["dims"]["nCells"], case["dims"]["nVertices"]
        radius = np.sqrt(mesh["xVertex"][:nv] ** 2 + mesh["yVertex"][:nv] ** 2 + mesh["zVertex"][:nv] ** 2)
        assert np.allclose(radius, 6371229.0, rtol=1e-12), case["name"]
        assert case["config"]["mesh:on_a_sphere"] is True
        areas = mesh["areaCell"][:nc]
        ratio = areas.max() / areas.min()
        if case.get("varres"):
            assert ratio > 2.5, (case["name"], ratio)
        else:
            assert ratio < 1.6, (case["name"], ratio)
    rows = [*rso.cases("r8"), *srs.cases("r8")]
    assert {(c["mesh_kind"], c["varres"]) for c in rows} == {("hex", False), ("hex", True), ("quad", False),
                                                             ("quad", True)}
    assert {c["config"]["config_rotate_cartesian_grid"] for c in rows} == {True, False}
    assert {c["sphere"] for c in rows} == {"north", "south", "midlat"}


def test_fixture_covers_the_spherical_branches():
    logs = rso.load_logs()
    cases = {c["name"]: c for c in rso.cases("r8")}
    assert set(logs) == set(cases)
    for precision in BOTH:
        fixture = rso.load_fixture(precision)
        assert set(fixture) == set(cases)
        for name, case in cases.items():
            out = fixture[name]
            # the init's spherical frames are the GPU's input
            assert "incremental_remap/transGlobalToCell" in out, name
            if case["stage"] == 4:
                assert logs[name][precision]["ierr"] == 0
                nc = case["dims"]["nCells"]
                before = case["pools"]["tracers"]["iceAreaCategory"][:nc]
                after = out["tracers/iceAreaCategory"][:nc]
                assert ro.word_mismatches(after, before.astype(after.dtype)) > 0 and (after >= 0).all()
            if case["stage"] != 2:
                continue
            ne, nc = case["dims"]["nEdges"], case["dims"]["nCells"]
            cells = out["incremental_remap/iCellTriangle"][:ne]
            remap_cells = out["incremental_remap/cellsOnEdgeRemap"][:ne]
            counts = np.count_nonzero(cells, axis=1)
            assert counts.max() >= 3 and (counts == 2).any() and (counts == 0).any()
            side = [(cells[e, t] == remap_cells[e, 2:]).any() for e in range(ne) for t in range(6) if cells[e, t]]
            assert sum(side) > 10
            area = out["incremental_remap/triangleArea"][:ne]
            assert (area > 0).any() and (area < 0).any()
            if case["mesh_kind"] == "quad":
                assert any((cells[e] == remap_cells[e, 4]).any() or (cells[e] == remap_cells[e, 5]).any()
                           for e in range(ne) if cells[e].any())
    assert len(logs["remap_sphere_hex_south_varres_recover"]["r4"]["warnings"]) > 0


@pytest.mark.parametrize("precision", BOTH)
def test_step_fixture_covers_the_branches(precision):
    fixture = srs.load_fixture(precision)
    cases = {c["name"]: c for c in srs.cases(precision)}
    assert set(fixture) == set(cases)
    for name, out in fixture.items():
        case = cases[name]
        nc = case["dims"]["nCells"]
        area = out["tracers/iceAreaCategory"][:nc]
        assert np.isfinite(area).all() and (area >= 0).all()
        assert ro.word_mismatches(area, case["pools"]["tracers"]["iceAreaCategory"][:nc]) > nc // 4
        assert int(out["mesh/itimestep"][0]) == case["config"]["oracle_n_steps"]
    assert any(c["config"]["config_dynamics_subcycle_number"] > 1 for c in cases.values())


def test_a_spherical_mesh_is_accepted_and_other_cells_refused():
    from woof.hex.mpas_seaice import remap_v842 as rv
    config = rso.cases("r8")[-1]["config"]
    rv.check_config(config)
    assert rv.geometry_flags(config) == (True, True)
    assert rv.geometry_flags(dict(config, config_rotate_cartesian_grid=False)) == (True, False)
    assert rv.geometry_flags(dict(config, **{"mesh:on_a_sphere": False})) == (False, False)
    tree = rv.tracer_tree({}, {}, lambda name: True)
    with pytest.raises(ConfigurationRefusal, match="vertexDegree"):
        rv.run(tree, pools={}, mesh={"vertexDegree": 5}, configs=config)


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = rso.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_time_step_matches_the_v842_oracle_word_for_word_over_the_steps(precision):
    report = srs.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_a_restart_continues_the_spherical_run_bitwise(tmp_path, precision):
    import cupy as cp
    from woof.hex.mpas_seaice import timestep_v842 as ts
    case = next(c for c in srs.cases(precision) if c["config"]["oracle_n_steps"] >= 3)
    fixture = srs.load_fixture(precision)[case["name"]]
    config = case["config"]
    dev = sr.device_pools(case, fixture)
    clock = ts.initialize(pools=dev, configs=config, remap_geometry="input", basis="input")
    next_step = ts.advance(pools=dev, configs=config, clock=clock, n_steps=1)
    dev["diagnostics"]["oracle_next_step"] = next_step
    ts.write_restart(tmp_path / "restart.npz", pools=dev, clock=clock)
    pools, clock = ts.read_restart(tmp_path / "restart.npz", like=dev, configs=case["config"])
    del dev
    ts.advance(pools=pools, configs=config, clock=clock, n_steps=int(config["oracle_n_steps"]) - 1,
               first_step=int(pools["diagnostics"]["oracle_next_step"]))
    want = ro.expected_outputs(case, fixture)
    bad = {key: m for key, words in want.items() if ro.graded(case, key)
           and (m := sr.word_mismatches(cp.asnumpy(pools[key.split("/")[0]][key.split("/", 1)[1]]), words))}
    assert not bad, bad


def _mutated_source(monkeypatch, old: str, new: str) -> None:
    from woof.hex.mpas_seaice import remap_v842 as rv
    source = rv.module_source
    assert source("r8").count(old) == 1, old

    def mutated(precision="r4"):
        return source(precision).replace(old, new)
    monkeypatch.setattr(rv, "module_source", mutated)
    rv._module_on.cache_clear()


def _mismatches_after(case_name: str, field: str, precision: str = "r8") -> int:
    from woof.hex.mpas_seaice import remap_v842 as rv
    case = next(c for c in rso.cases(precision) if c["name"] == case_name)
    fixture = rso.load_fixture(precision)[case_name]
    try:
        got = ro.run_gpu(case, fixture)
    finally:
        rv._module_on.cache_clear()
    return ro.word_mismatches(got[field], ro.expected_outputs(case, fixture)[field])


_GRADX = "tracer_reconstruction/iceVolumeCategoryGradx"
_RECONSTRUCTION = {
    "no_rotation": ("if (rotate) {", "if (0) {", "reconstruct_sphere_hex_north_ponds"),
    "transposed_frame": ("xg = A(A(M(T[0], gx), M(T[3], gy)), M(T[6], gz));",
                         "xg = A(A(M(T[0], gx), M(T[1], gy)), M(T[2], gz));", "reconstruct_sphere_hex_south_varres"),
    "plane_gradient": ("        if (onSphere) {\n            if (rotate) {", "        if (0) {\n            if (rotate) {",
                       "reconstruct_sphere_quad_midlat_snow_layers"),
}


@requires_gpu
@pytest.mark.parametrize("mutation", sorted(_RECONSTRUCTION))
def test_a_mutated_spherical_gradient_is_caught(monkeypatch, mutation):
    old, new, case = _RECONSTRUCTION[mutation]
    _mutated_source(monkeypatch, old, new)
    assert _mismatches_after(case, _GRADX) > 0


_DEPARTURE = {
    "planar_shift": ("            if (onSphere) {\n                // shift_vertices_of_departure_triangle on a sphere",
                     "            if (0) {\n                // shift_vertices_of_departure_triangle on a sphere",
                     "departure_sphere_hex_north"),
    "no_basis_swap": ("if (crossProduct > C(0.0)) {", "if (0) {", "departure_sphere_hex_south_varres_fast"),
    "quad_other_edge": ("int iOtherEdge = vertexDegree == 3 ? iEdgeOnEdgeRemap + 2 : k + 1 + 4;",
                        "int iOtherEdge = vertexDegree == 3 ? iEdgeOnEdgeRemap + 2 : 6 - k;",
                        "departure_sphere_quad_north_varres"),
    "central_side_edge": ("const int iOtherEdge = left ? k + 1 : k + 3;", "const int iOtherEdge = left ? k + 3 : k + 1;",
                          "departure_sphere_quad_midlat_fast"),
}


@requires_gpu
@pytest.mark.parametrize("mutation", sorted(_DEPARTURE))
def test_a_mutated_spherical_shift_is_caught(monkeypatch, mutation):
    old, new, case = _DEPARTURE[mutation]
    _mutated_source(monkeypatch, old, new)
    assert _mismatches_after(case, "incremental_remap/xTriangle") > 0


@requires_gpu
def test_a_step_that_remaps_as_on_a_plane_is_caught(monkeypatch):
    from woof.hex.mpas_seaice import remap_v842 as rv
    monkeypatch.setattr(rv, "geometry_flags", lambda configs: (False, False))
    case = srs.cases("r8")[0]
    fixture = srs.load_fixture("r8")[case["name"]]
    got = sr.run_gpu(case, fixture)
    field = "tracers/iceAreaCategory"
    assert sr.word_mismatches(got[field], fixture[field]) > 0


def _oracle_toolchain():
    return _oracle_toolchains.mpas_v842("core_seaice")


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MPAS-Seaice oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
@pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64"),
                    reason="the fixture is x86-64 gfortran's: arm64 gfortran's MAX/MIN keep another operand "
                           "on signed-zero ties")
def test_the_oracle_rebuilt_here_writes_the_fixtures(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    for maker, pinned_dir in (("make_remap_fixture.py", rso.FIXTURE_DIR),
                              ("make_step_remap_fixture.py", srs.FIXTURE_DIR)):
        out = tmp_path / pinned_dir.name
        subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / maker), str(tmp_path / "build"),
                        "--sphere", "--out", str(out)], check=True)
        if (pinned_dir / "remap-logs.json").is_file():
            assert (out / "remap-logs.json").read_text(encoding="utf-8") == \
                (pinned_dir / "remap-logs.json").read_text(encoding="utf-8")
        for precision in BOTH:
            fresh = np.load(out / f"oracle-{precision}.npz")
            pinned = np.load(pinned_dir / f"oracle-{precision}.npz")
            assert sorted(fresh.files) == sorted(pinned.files)
            for key in pinned.files:
                assert ro.word_mismatches(fresh[key], pinned[key]) == 0, (pinned_dir.name, precision, key)
