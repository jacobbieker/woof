"""One MALI forward-Euler time step, word for word against v8.4.2.

The reference is MALI v8.4.2's own land-ice modules, compiled verbatim with
gfortran -O0 -ffp-contract=off at RKIND 4 and RKIND 8.
``tools/mali_oracle/run_time_integration_fe.F90`` makes a restart's
``li_thermal_init`` and one ``li_time_integrator_forwardeuler``:
prepare_advection (layer velocities, apparent diffusivity, the CFL limits and
the fixed or adaptive time step), the floating basal melt, the thermal solver,
advection, the velocity solve (SIA or none, with the reconstruction, the strain
rates and the fast-ice removal) and the diagnostic variables. The oracle's
outputs are committed (tests/data/oracles/mali_time_integration_fe,
PROVENANCE.md), so the GPU comparison runs without gfortran.

THE BREAKAGES THESE PREVENT
---------------------------
1. A step that agrees with MALI only to a tolerance. Every geometry, thermal
   and velocity word, the masks and the time step included, must be
   bit-identical.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: fixed and adaptive time steps (bound by the
   forcing interval and by the maximum step), SIA and no velocity solve, the
   thickness and tracer advection options, the sphere's reconstruction, the
   divide's zero diffusivity, fast floating ice removed, signed-zero ties in
   the upwind layer thickness.
4. A gate that cannot fail. Mutating one evaluation order in the step's kernels,
   or taking the IEEE max on a signed-zero tie, must produce
   mismatches.
5. Outputs that depend on the model's uninitialised scratch fields: the oracle
   fills them with random words, and the port never reads them.
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
from woof.verify import mali_time_integration_fe_oracle as fo

ROOT = Path(__file__).resolve().parents[1]


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (fo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {**{case["name"]: fo.input_digest(case) for case in fo.cases()},
                        **{case["name"] + "/r8": fo.input_digest(case) for case in fo.cases("r8")}}


@pytest.mark.parametrize("precision", ["r4", "r8"])
def test_fixture_covers_the_branches(precision):
    fixture = fo.load_fixture(precision)
    cases = {case["name"]: case for case in fo.cases(precision)}
    assert set(fixture) == set(cases)
    configs = [c["config"] for c in cases.values()]
    assert {c["config_velocity_solver"] for c in configs} == {"sia", "none"}
    assert {c["config_thermal_solver"] for c in configs} == {"none", "temperature", "enthalpy"}
    assert {c["config_thickness_advection"] for c in configs} == {"fo", "none"}
    assert {c["config_tracer_advection"] for c in configs} == {"fo", "none"}
    assert {c["config_adaptive_timestep"] for c in configs} == {False, True}
    assert {c["on_a_sphere"] for c in configs} == ({False} if precision == "r8" else {False, True})
    zero_ties = removed = 0
    for name, out in fixture.items():
        case = cases[name]
        # the oracle ran clean: no error flag, no logged error
        assert not any(out[k][0] for k in ("errInit", "errStep", "logErrors"))
        assert all(np.isfinite(v).all() for k, v in out.items() if k != "velocity/eTheta")
        dt = out["mesh/deltat"]
        assert 0 < dt <= out["mesh/allowableDtACFL"]
        # the step is done: dHdt is the thickness change over this step's dt
        assert (out["geometry/thicknessOld"] == out["geometry/thickness"]).all()
        diffusivity = out["geometry/apparentDiffusivity"]
        assert (diffusivity == 0).any() and (diffusivity > 0).any()
        edge = out["geometry/layerThicknessEdge"]
        zero_ties += int((np.signbit(edge) & (edge == 0)).any() and ((edge == 0) & ~np.signbit(edge)).any())
        removed += int((out["geometry/calvingThickness"] > case["pools"]["geometry"]["calvingThickness"]).any())
    assert zero_ties == len(fixture)
    # floating ice faster than 100 km/yr calves (remove_fast_ice)
    assert removed > 0
    # the adaptive steps: bound by the next forcing time, and by the floored maximum step
    assert fixture["sia_enthalpy_adaptive"]["mesh/deltat"] == 20 * 86400
    assert fixture["none_temperature_no_tracers"]["mesh/deltat"] == 1234567
    assert fixture["sia_temperature_fo"]["mesh/deltat"] == 30 * 86400


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = fo.measure(precision=precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


_MUTATIONS = [
    # the upwind layer thickness's MAX with its arguments swapped: the tie of signed zeros flips
    ("li_max(M(velSign, layerThickness[cell1 * nVertLevels + k]),\n"
     "                   M(M(velSign, LI_C(-1.0)), layerThickness[cell2 * nVertLevels + k]))",
     "li_max(M(M(velSign, LI_C(-1.0)), layerThickness[cell2 * nVertLevels + k]),\n"
     "                   M(velSign, layerThickness[cell1 * nVertLevels + k]))",
     "geometry/layerThicknessEdge", "sia_no_advection", ("r4", "r8")),
    # the downslope flux's division by the slope taken before the layer thickness
    ("D(M(S(M(M(LI_C(-1.0), s1), v1), M(s2, v2)), layerThickness[iCell * nVertLevels + k]),\n"
     "                             magnitude)",
     "M(S(M(M(LI_C(-1.0), s1), v1), M(s2, v2)), D(layerThickness[iCell * nVertLevels + k],\n"
     "                             magnitude))",
     "geometry/apparentDiffusivity", "sia_temperature_fo", ("r4", "r8")),
    # the diffusive CFL step divided before squaring the cell size in
    ("dt = D(M(LI_C(0.5), M(dCell, dCell)), A(diffusivity, smallNumber));",
     "dt = M(D(M(LI_C(0.5), dCell), A(diffusivity, smallNumber)), dCell);",
     "mesh/allowableDtDCFL", None, ("r4", "r8")),
    # the sphere's meridional velocity with the latitude's sine distributed
    ("me[k] = A(-M(A(M(cx[k], clon), M(cy[k], slon)), slat), M(cz[k], clat));",
     "me[k] = A(-A(M(M(cx[k], clon), slat), M(M(cy[k], slon), slat)), M(cz[k], clat));",
     "velocity/uReconstructMeridional", "sia_no_thermal_sphere", ("r4",)),
    # the largest principal strain rate through a scaled hypotenuse
    ("eMax[i] = A(mean, LI_SQRT(A(M(half, half), M(quarter, quarter))));",
     "eMax[i] = A(mean, M(LI_SQRT(A(M(D(half, quarter), D(half, quarter)), LI_C(1.0))), LI_ABS(quarter)));",
     "velocity/eMax", "sia_temperature_fo", ("r4", "r8")),
    # dH/dt scaled to years before the division by the step
    ("dHdt[i] = M(D(S(thickness[i], thicknessOld[i]), deltat), LI_SCYR);",
     "dHdt[i] = D(M(S(thickness[i], thicknessOld[i]), LI_SCYR), deltat);",
     "geometry/dHdt", None, ("r4", "r8")),
]


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
@pytest.mark.parametrize("original, mutated, field, case, precisions", _MUTATIONS)
def test_a_mutated_evaluation_order_is_caught(monkeypatch, precision, original, mutated, field, case, precisions):
    if precision not in precisions:
        pytest.skip(f"{case} has no {precision} form of this expression")
    from woof.hex.mpas_ice import time_integration_fe_v842 as fe
    source = fe.module_source(precision)
    assert source.count(original) == 1
    monkeypatch.setattr(fe, "module_source", lambda p="r4": source.replace(original, mutated))
    fe._module_on.cache_clear()
    try:
        rows = [c for c in fo.cases(precision) if case is None or c["name"] == case]
        fixture = fo.load_fixture(precision)
        total = sum(fo.word_mismatches(fo.run_gpu(row)[field], fixture[row["name"]][field]) for row in rows)
        assert total > 0
    finally:
        fe._module_on.cache_clear()


@requires_gpu
def test_an_ieee_max_is_caught(monkeypatch):
    # fmaxf (max.f32) orders -0 below +0, as arm64 gfortran's fmaxnm does and as
    # the Metal translator compiles a floating select of MAX's shape; the
    # oracle's maxss keeps the first argument on the tie
    from woof.hex.mpas_ice import time_integration_fe_v842 as fe
    source = fe.module_source("r4")
    original = ("__device__ __forceinline__ rk li_max(rk a, rk b) {\n"
                "    return __int_as_float(b > a ? __float_as_int(b) : __float_as_int(a));\n}")
    assert source.count(original) == 1
    mutated = "__device__ __forceinline__ rk li_max(rk a, rk b) { return fmaxf(a, b); }"
    monkeypatch.setattr(fe, "module_source", lambda p="r4": source.replace(original, mutated))
    fe._module_on.cache_clear()
    try:
        row = next(c for c in fo.cases("r4") if c["name"] == "sia_no_advection")
        got = fo.run_gpu(row)["geometry/layerThicknessEdge"]
        want = fo.load_fixture("r4")["sia_no_advection"]["geometry/layerThicknessEdge"]
        assert fo.word_mismatches(got, want) > 0
    finally:
        fe._module_on.cache_clear()


def _case_on_device(name: str, precision: str = "r4"):
    import cupy as cp
    case = next(c for c in fo.cases(precision) if c["name"] == name)
    dev = {pool: {k: cp.asarray(np.ascontiguousarray(v).reshape(np.shape(v) or (1,)))
                  for k, v in case["pools"][pool].items()} for pool in fo.POOLS}
    mesh = dict(dev["mesh"], nCellsSolve=case["dims"]["nCellsSolve"], nEdgesSolve=case["dims"]["nEdgesSolve"])
    return case, mesh, dev


@requires_gpu
@pytest.mark.parametrize("knob, value", [("config_velocity_solver", "simple"),
                                         ("config_velocity_solver", "FO")])
def test_the_pieces_not_ported_are_refused_by_name(knob, value):
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex.mpas_ice import time_integration_fe_v842 as fe
    case, mesh, dev = _case_on_device("sia_temperature_fo")
    with pytest.raises(ConfigurationRefusal, match=knob):
        fe.forward_euler_step(mesh=mesh, geometry=dev["geometry"], thermal=dev["thermal"],
                              velocity=dev["velocity"], config=dict(case["config"], **{knob: value}))


@requires_gpu
def test_rkind8_refuses_the_sphere_and_the_strain_angle():
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex.mpas_ice import time_integration_fe_v842 as fe
    case, mesh, dev = _case_on_device("sia_temperature_fo", "r8")
    with pytest.raises(ConfigurationRefusal, match="eTheta"):
        fe.forward_euler_step(mesh=mesh, geometry=dev["geometry"], thermal=dev["thermal"],
                              velocity=dev["velocity"], config=case["config"])
    del dev["velocity"]["eTheta"]
    with pytest.raises(ConfigurationRefusal, match="on_a_sphere"):
        fe.forward_euler_step(mesh=mesh, geometry=dev["geometry"], thermal=dev["thermal"],
                              velocity=dev["velocity"], config=dict(case["config"], on_a_sphere=True))


@requires_gpu
def test_halo_cells_and_cfl_violations_raise():
    from woof.hex.mpas_ice import time_integration_fe_v842 as fe
    case, mesh, dev = _case_on_device("sia_temperature_fo")
    with pytest.raises(ValueError, match="nCellsSolve"):
        fe.forward_euler_step(mesh=dict(mesh, nCellsSolve=case["dims"]["nCells"] - 1), geometry=dev["geometry"],
                              thermal=dev["thermal"], velocity=dev["velocity"], config=case["config"])
    mesh["deltat"][...] = np.float32(1.0e12)
    with pytest.raises(ValueError, match="CFL"):
        fe.forward_euler_step(mesh=mesh, geometry=dev["geometry"], thermal=dev["thermal"],
                              velocity=dev["velocity"], config=case["config"])


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_landice").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MALI oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
@pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64"),
                    reason="the fixture is x86-64 gfortran's: its MAX is maxss/maxsd, which keeps an operand on a "
                           "tie of signed zeros, where arm64 gfortran's fmaxnm orders -0 below +0")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mali_oracle" / "build.sh"), str(source), str(tmp_path / "build")],
                   check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mali_oracle" / "make_time_integration_fe_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    for precision in ("r4", "r8"):
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(fo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert fo.word_mismatches(fresh[key], pinned[key]) == 0, key
