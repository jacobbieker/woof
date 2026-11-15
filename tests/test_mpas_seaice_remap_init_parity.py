"""The incremental remap's init geometry, computed by the port, word for word against v8.4.2.

The reference is MPAS-Model v8.4.2's ``seaice_init_advection_incremental_remap``
with MPAS's own RBF set-up (``mpas_initialize_vectors``,
``mpas_init_reconstruct`` and the routines they call, cut verbatim out of
``src/operators``), compiled verbatim with gfortran -O0 -ffp-contract=off at
RKIND 4 and 8 and driven by ``tools/mpas_seaice_oracle/run_incremental_remap.F90``
built as ``run_incremental_remap_init_{r4,r8}``
(``woof.verify.mpas_seaice_remap_init_oracle``): the init alone (stage 5) and
remap steps on the port's own geometry (stage 4), on planar hexagon and
quadrilateral blocks and spherical ones, uniform and variable-resolution.
:mod:`woof.hex.mpas_seaice.remap_init_v842` derives every word of it on the
host: the frames, the vertex coordinates in each plane, the remap neighbours,
the moments, the rotated coordinates, the RBF vectors and
``coeffs_reconstruct``. The earlier remap fixtures (whose geometry the GPU
took as input) are held to it too, and so is the time step's init.

THE BREAKAGES THESE PREVENT
---------------------------
1. Geometry in another operation order than v8.4.2's (a moment's product
   associated otherwise, a unit vector's sum, the RBF scale).
2. The elimination without Tao Pang's partial pivoting, or the RBF fit over
   the halo cells too (v8.4.2 fits the owned cells only).
3. Frames or rotated coordinates turned the other way.
4. A geometry that is only ever the input's: each of the above, mutated, must
   be caught, and the inputs carry none of the remap pool's geometry.
"""
from __future__ import annotations

import copy
import os
import platform
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_seaice import remap_init_v842 as ri
from woof.hex.mpas_seaice.remap_v842 import RemapError
from woof.verify import mpas_seaice_remap_checks_oracle as rco
from woof.verify import mpas_seaice_remap_init_oracle as rio
from woof.verify import mpas_seaice_remap_oracle as ro
from woof.verify import mpas_seaice_remap_sphere_oracle as rso

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")
_OUTPUTS = {"mesh": ri.MESH_OUTPUTS, "incremental_remap": ri.REMAP_OUTPUTS, "rotated_mesh": ro.ROTATED}


def _host_pools(case: dict) -> dict:
    pools = {pool: {name: (np.array(v) if isinstance(v, np.ndarray) else v) for name, v in fields.items()}
             for pool, fields in case["pools"].items()}
    pools.setdefault("rotated_mesh", {})
    return pools


def _host_geometry(case: dict, module=ri) -> dict:
    """``module.initialize`` on the case's host pools (numpy)."""
    pools = _host_pools(case)
    m = pools["mesh"]
    for name, shape in (("localVerticalUnitVectors", (m["nCells"] + 1, 3)),
                        ("edgeNormalVectors", (m["nEdges"] + 1, 3)),
                        ("cellTangentPlane", (m["nCells"] + 1, 2, 3))):
        if m.get(name) is None:
            m[name] = np.zeros(shape, ro.PRECISIONS[case["precision"]])
    module.initialize(mesh=m, remap=pools["incremental_remap"], rotated=pools["rotated_mesh"],
                      configs=case["config"], real=ro.PRECISIONS[case["precision"]])
    return pools


def _host_mismatches(case: dict, fixture: dict, module=ri, graded_only: bool = True) -> dict[str, int]:
    pools = _host_geometry(case, module)
    want = ro.expected_outputs(case, fixture)
    bad = {}
    for pool, names in _OUTPUTS.items():
        for name in names:
            key = f"{pool}/{name}"
            if key not in want or (graded_only and not ro.graded(case, key)):
                continue
            if n := ro.word_mismatches(pools[pool][name], want[key]):
                bad[key] = n
    return bad


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in (rio.FIXTURE_DIR / "inputs.sha256").read_text(
        encoding="utf-8").splitlines())
    assert recorded == {f"{c['name']}/{p}": ro.input_digest(c) for p in BOTH for c in rio.cases(p)}


def test_the_cases_cover_planar_spherical_and_variable_resolution_blocks():
    rows = rio.cases("r8")
    assert all(c.get("init") for c in rows)
    assert {(c["mesh_kind"], bool(c.get("sphere"))) for c in rows} == {("hex", False), ("quad", False),
                                                                       ("hex", True), ("quad", True)}
    assert {c["stage"] for c in rows} == {4, 5}
    assert {c["sphere"] for c in rows if c.get("sphere")} == {"north", "south", "midlat"}
    varres = [c for c in rows if c.get("varres")]
    assert len(varres) >= 3
    for case in varres:
        areas = case["pools"]["mesh"]["areaCell"][:case["dims"]["nCells"]]
        assert areas.max() / areas.min() > 2.5


def test_the_inputs_carry_none_of_the_geometry():
    # the fixture records the fields the oracle changed: the geometry is all among them (the
    # input's are scratch words), as are the RBF vectors and coefficients
    core = {"xVertexOnCell", "yVertexOnCell", "xVertexOnEdge", "yVertexOnEdge", "remapEdge", "cellsOnEdgeRemap",
            "edgesOnEdgeRemap", "minLengthEdgesOnVertex", *(f"{m}AvgCell" for m in ri.MOMENTS)}
    for precision in BOTH:
        fixture = rio.load_fixture(precision)
        for case in rio.cases(precision):
            out = fixture[case["name"]]
            names = core | ({n for pair in ri.FRAMES for n in pair} if case.get("sphere") else set())
            assert {f"incremental_remap/{n}" for n in names} <= set(out), case["name"]
            assert {f"mesh/{n}" for n in ri.MESH_OUTPUTS} <= set(out), case["name"]
            ncs = case["pools"]["mesh"]["nCellsSolve"]
            assert not np.any(out["mesh/coeffs_reconstruct"][ncs:]), case["name"]


@pytest.mark.parametrize("precision", BOTH)
def test_the_host_geometry_matches_the_v842_oracle_word_for_word(precision):
    fixture = rio.load_fixture(precision)
    bad = {c["name"]: m for c in rio.cases(precision) if (m := _host_mismatches(c, fixture[c["name"]]))}
    assert not bad, bad


@pytest.mark.parametrize("family", [ro, rso, rco], ids=["remap", "remap_sphere", "remap_checks"])
@pytest.mark.parametrize("precision", BOTH)
def test_the_host_geometry_is_the_earlier_fixtures_input_geometry(family, precision):
    # those fixtures' geometry came from the oracle's own init (with a stand-in RBF set-up):
    # every word but coeffs_reconstruct's is the port's
    fixture = ro.load_fixture(precision, family.FIXTURE_DIR)
    bad = {}
    for case in ro.cases(precision, family.CASES):
        if case["stage"] == 0:
            continue
        pools = _host_geometry(case)
        want = ro.expected_outputs(case, fixture[case["name"]])
        for pool, names in (("incremental_remap", ri.REMAP_OUTPUTS), ("rotated_mesh", ro.ROTATED)):
            for name in names:
                key = f"{pool}/{name}"
                if key in want and (n := ro.word_mismatches(pools[pool][name], want[key])):
                    bad[(case["name"], key)] = n
    assert not bad, bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = rio.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_the_time_step_init_computes_the_step_fixtures_geometry(precision):
    from woof.hex.mpas_seaice import timestep_v842 as ts
    from woof.verify import mpas_seaice_step_remap_oracle as sr
    from woof.verify import mpas_seaice_step_remap_sphere_oracle as srs
    checked = 0
    for module in (sr, srs):
        fixture = module.load_fixture(precision)
        for case in module.cases(precision):
            dev = ro.on_device(case)       # no fixture: the remap pool's geometry is the input's zeros
            dev["diagnostics"]["xtime"] = case["config"]["diagnostics:xtime"]
            dev["diagnostics"]["simulationStartTime"] = case["config"]["diagnostics:simulationStartTime"]
            ts.initialize(pools=dev, configs=case["config"])
            for name in ro.GEOMETRY:
                key = f"incremental_remap/{name}"
                if key in fixture[case["name"]]:
                    got = np.asarray(dev["incremental_remap"][name].get())
                    assert ro.word_mismatches(got, fixture[case["name"]][key]) == 0, (case["name"], key)
                    checked += 1
    assert checked


def test_the_time_step_refuses_an_unknown_geometry_source():
    from woof.hex.mpas_seaice import timestep_v842 as ts
    case = rio.cases("r8")[0]
    with pytest.raises(ValueError, match="remap_geometry"):
        ts.initialize(pools={}, configs={**case["config"], "config_use_advection": False}, remap_geometry="fixture")


def _mutant(old: str, new: str):
    """remap_init_v842 with ``old`` (once in its source) replaced by ``new``."""
    source = Path(ri.__file__).read_text(encoding="utf-8")
    assert source.count(old) == 1, old
    module = types.ModuleType("woof.hex.mpas_seaice._remap_init_mutant")
    module.__package__ = "woof.hex.mpas_seaice"
    exec(compile(source.replace(old, new), ri.__file__, "exec"), module.__dict__)
    return module


_MUTATIONS = {
    # the local-to-global frame transposed (its global-to-local, which the planes use, kept)
    "frame_transposed": ("acc = acc + eye[k, i] * local[:, k, j]", "acc = acc + eye[k, j] * local[:, k, i]",
                         "init_sphere_hex_north", "r8", "incremental_remap/transCellToGlobal"),
    # the poles turned to the equator the other way
    "rotation_sign": ("rx[:n], ry[:n], rz[:n] = -z[:n], y[:n], x[:n]", "rx[:n], ry[:n], rz[:n] = z[:n], y[:n], -x[:n]",
                      "init_sphere_quad_south_varres", "r8", "rotated_mesh/xCellRotate"),
    # a moment's product associated otherwise
    "moment_order": ('"xy": (w * x) * y', '"xy": w * (x * y)', "init_quad", "r4", "incremental_remap/xyAvgCell"),
    # mpas_unit_vec_in_r3's sum associated otherwise
    "unit_vector_order": ("mag = np.sqrt((x * x + y * y) + z * z)", "mag = np.sqrt(x * x + (y * y + z * z))",
                          "init_sphere_hex_midlat_varres", "r4", "mesh/localVerticalUnitVectors"),
    # the RBF scale as a product with the reciprocal
    "rbf_scale": ("alpha = alpha / real(count)", "alpha = alpha * (c.one / real(count))",
                  "init_sphere_hex_north", "r4", "mesh/coeffs_reconstruct"),
    # no partial pivoting: the first nonzero pivot
    "no_pivoting": ("better = p > best", "better = (p > best) & (k < 0)",
                    "init_hex", "r4", "mesh/coeffs_reconstruct"),
    # the RBF fit over the halo cells too
    "fit_the_halo": ('ncs = int(m["nCellsSolve"])\n    real = c.real\n    zero = c.zero',
                     'ncs = int(m["nCells"])\n    real = c.real\n    zero = c.zero',
                     "init_quad", "r8", "mesh/coeffs_reconstruct"),
}


@pytest.mark.parametrize("mutation", sorted(_MUTATIONS))
def test_a_mutated_init_is_caught(mutation):
    old, new, name, precision, key = _MUTATIONS[mutation]
    case = next(c for c in rio.cases(precision) if c["name"] == name)
    fixture = rio.load_fixture(precision)[name]
    assert not _host_mismatches(case, fixture)
    assert key in _host_mismatches(case, fixture, _mutant(old, new))


def _case(name: str = "init_hex", precision: str = "r8") -> dict:
    return copy.deepcopy(next(c for c in rio.cases(precision) if c["name"] == name))


def test_too_few_halo_layers_are_refused():
    case = _case()
    case["config"]["config_num_halos"] = 1
    with pytest.raises(ConfigurationRefusal, match="too few halo layers"):
        _host_geometry(case)


def test_more_halo_layers_than_needed_are_the_three_warnings_v842_logs():
    assert [w["ints"] for w in ri.check_halo_layer_number({"config_num_halos": 4}, 3)] == [[3], [4], [2]]


def test_a_periodic_mesh_is_refused():
    case = _case()
    case["config"]["mesh:is_periodic"] = True
    with pytest.raises(ConfigurationRefusal, match="periodic"):
        _host_geometry(case)


def test_clockwise_cells_are_refused():
    case = _case()
    m = case["pools"]["mesh"]
    for c in range(m["nCells"]):
        k = m["nEdgesOnCell"][c]
        m["verticesOnCell"][c, :k] = m["verticesOnCell"][c, :k][::-1].copy()
    with pytest.raises(RemapError, match="CCW"):
        _host_geometry(case)


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
                    str(tmp_path / "build"), "--init", "--out", str(out)], check=True)
    assert (out / "remap-logs.json").read_text(encoding="utf-8") == \
        (rio.FIXTURE_DIR / "remap-logs.json").read_text(encoding="utf-8")
    for precision in BOTH:
        fresh = np.load(out / f"oracle-{precision}.npz")
        pinned = np.load(rio.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert ro.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
