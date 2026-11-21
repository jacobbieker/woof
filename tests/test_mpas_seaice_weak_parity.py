"""MPAS-Seaice's weak-form stress divergence, word for word against v8.4.2 (RKIND 4 and 8).

The reference is ``mpas_seaice_velocity_solver_weak.F`` compiled verbatim,
with ``seaice_normal_vectors`` cut verbatim out of ``mpas_seaice_mesh.F``
(``tools/mpas_seaice_oracle/run_velocity_weak.F90``). It runs the init, N
passes of ``seaice_internal_stress_weak`` and
``seaice_final_divergence_shear_weak`` on planar and spherical blocks. The
second reference is the forward model (``run_seaice_step.F90`` with
``config_stress_divergence_scheme = 'weak'``) over N steps. The outputs are
committed (tests/data/oracles/mpas_seaice_weak and
tests/data/oracles/mpas_seaice_step_weak, PROVENANCE.md).

THE BREAKAGES THESE PREVENT
---------------------------
1. Normal vectors, strains, stresses or a stress divergence whose words are
   not v8.4.2's. Causes: an operation out of the Fortran's order (the
   metric-term products, the edge averages), libgfortran's contracted
   ``matmul`` done uncontracted, or a word written that v8.4.2 leaves alone
   (cells and vertices not solved for, rows past nEdgesOnCell).
2. v8.4.2's whole-array ``Delta`` in ridgeShear "fixed" by default, or the
   opt-in correction ``weak_ridge_delta`` not taking effect.
3. The time step running the variational solver, or the input normals, when
   v8.4.2 runs the weak form with its own init.
4. A sphere's C library words guessed instead of refused without a
   ``metric_libm``.
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
from woof.hex.mpas_seaice import weak_v842 as wv
from woof.verify import mpas_seaice_remap_oracle as ro
from woof.verify import mpas_seaice_step_remap_oracle as sr
from woof.verify import mpas_seaice_weak_oracle as wo

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")


def _glibc_244() -> bool:
    """Whether the host C library is the fixtures' (glibc 2.44)."""
    try:
        return os.confstr("CS_GNU_LIBC_VERSION") == "glibc 2.44"
    except (ValueError, OSError, AttributeError):
        return False


def _libm() -> str:
    """The host's own C library where it is the oracle's; else the correctly
    rounded one (the fixtures' sphere cases happen to need no word where
    glibc 2.44 misrounds)."""
    return "host" if _glibc_244() else "correctly-rounded"


def _recorded(fixture_dir: Path) -> dict[str, str]:
    return dict(reversed(line.split()) for line in
                (fixture_dir / "inputs.sha256").read_text(encoding="utf-8").splitlines())


def test_case_inputs_are_the_recorded_bytes():
    assert _recorded(wo.FIXTURE_DIR) == {f"{c['name']}/{p}": wo.input_digest(c) for p in BOTH for c in wo.cases(p)}
    assert _recorded(wo.STEP_FIXTURE_DIR) == {f"{c['name']}/{p}": sr.input_digest(c)
                                              for p in BOTH for c in wo.step_cases(p)}


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_branches(precision):
    fixture = wo.load_fixture(precision)
    cases = {c["name"]: c for c in wo.cases(precision)}
    assert set(fixture) == set(cases)
    configs = [c["config"] for c in cases.values()]
    assert {c["mesh:on_a_sphere"] for c in configs} == {True, False}
    assert {c["config_revised_evp"] for c in configs} == {True, False}
    assert {c["config_use_column_package"] for c in configs} == {True, False}
    assert {c["config_rotate_cartesian_grid"] for c in configs} == {True, False}
    for name, out in fixture.items():
        case = cases[name]
        nc = int(case["pools"]["mesh"]["nCellsSolve"])
        for key in ("velocity_solver/stressDivergenceU", "velocity_solver/divergence", "velocity_weak/stress11",
                    "velocity_weak/normalVectorPolygon"):
            assert np.isfinite(out[key]).all(), (name, key)
        assert (out["velocity_solver/stressDivergenceU"] != 0).any(), name
        # the cells past nCellsSolve keep the input's divergence
        assert wo.word_mismatches(out["velocity_solver/divergence"][nc:],
                                  case["pools"]["velocity_solver"]["divergence"][nc:]) == 0
        assert ("ridging/ridgeShear" in out) == case["config"]["config_use_column_package"], name


def test_the_host_init_matches_the_v842_oracle_word_for_word():
    bad = {}
    for precision in BOTH:
        fixture = wo.load_fixture(precision)
        for case in wo.cases(precision):
            got = wo.run_init(case, metric_libm=_libm())
            for key, value in got.items():
                want = fixture[case["name"]].get(key, case["pools"]["velocity_weak"][key.split("/")[1]])
                if m := wo.word_mismatches(value, want):
                    bad[(precision, case["name"], key)] = m
    assert not bad, bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_the_port_matches_the_v842_oracle_word_for_word(precision):
    report = wo.measure(precision, metric_libm=_libm())
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_the_time_step_runs_the_weak_form_word_for_word_over_the_steps(precision):
    report = wo.measure_step(precision, metric_libm=_libm())
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
def test_a_step_on_the_input_normals_is_caught():
    import cupy as cp
    case = wo.step_cases("r4")[0]
    fixture = wo.load_step_fixture("r4")[case["name"]]
    from woof.hex.mpas_seaice import timestep_v842 as ts
    dev = sr.device_pools(case, fixture)
    clock = ts.initialize(pools=dev, configs=case["config"], remap_geometry="input", basis="input")
    ts.advance(pools=dev, configs=case["config"], clock=clock, n_steps=int(case["config"]["oracle_n_steps"]),
               metric_libm=_libm())      # the noise latitudes are not 0
    assert sr.word_mismatches(cp.asnumpy(dev["velocity_solver"]["uVelocity"]),
                              fixture["velocity_solver/uVelocity"]) > 0


def test_a_sphere_is_refused_without_a_libm():
    sphere = next(c for c in wo.cases("r4") if c["name"] == "weak_sphere_hex_north")
    with pytest.raises(ConfigurationRefusal, match="metric_libm"):
        wo.run_init(sphere)
    with pytest.raises(ConfigurationRefusal, match="metric_libm"):
        wv.metric_tangents(np.array([0.0, 0.3]), np.array([0.0]))
    tan_c, tan_v = wv.metric_tangents(np.zeros(3, np.float32), np.zeros(2, np.float32))
    assert not tan_c.any() and not tan_v.any() and tan_c.dtype == np.float32
    wo.run_init(next(c for c in wo.cases("r8") if c["name"] == "weak_planar_hex"))     # a plane needs none
    with pytest.raises(ValueError, match="metric_libm"):
        wo.run_init(sphere, metric_libm="glibc")


def test_a_mesh_without_coordinates_is_refused_by_name():
    case = next(c for c in wo.cases("r4") if c["name"] == "weak_planar_hex")
    mesh = {k: v for k, v in case["pools"]["mesh"].items() if k not in ("xEdge", "edgesOnVertex")}
    with pytest.raises(ConfigurationRefusal, match="xEdge"):
        wv.initialize(mesh=mesh, configs=case["config"], real=np.float32)


def test_the_weak_form_with_the_column_step_is_refused_by_name():
    from woof.hex.mpas_seaice import column_step_v842 as cs
    from woof.hex.mpas_seaice import timestep_v842 as ts
    for scheme in ("weak", "weak   "):          # a Fortran-padded namelist string too
        configs = {"config_use_column_package": True, "config_stress_divergence_scheme": scheme}
        with pytest.raises(ConfigurationRefusal, match="config_stress_divergence_scheme"):
            cs.check_config(configs)
    with pytest.raises(ConfigurationRefusal, match="config_stress_divergence_scheme"):
        ts.check_config({"config_stress_divergence_scheme": "none"})


@requires_gpu
def test_the_opt_in_ridge_delta_correction():
    case = next(c for c in wo.cases("r8") if c["name"] == "weak_planar_quad_revised_column")
    fixture = wo.load_fixture("r8")[case["name"]]
    with pytest.raises(ValueError, match="corrections"):
        wo.run_gpu(case, metric_libm=_libm(), corrections=("ridge_delta",))
    fixed = wo.run_gpu(case, metric_libm=_libm(), corrections=("weak_ridge_delta",))
    nc = int(case["pools"]["mesh"]["nCellsSolve"])
    quirk, own = fixture["ridging/ridgeShear"][:nc], fixed["ridging/ridgeShear"][:nc]
    assert wo.word_mismatches(fixed["ridging/ridgeConvergence"], fixture["ridging/ridgeConvergence"]) == 0
    assert own[-1] == quirk[-1]                       # the last owned cell's Delta is its own
    assert (own[:-1] != quirk[:-1]).mean() > 0.5
    # each cell's own Delta: 0.5 * (Delta - |divergence|)
    e11, e22, e12 = (fixed[f"velocity_weak/strain{n}"][:nc] for n in ("11", "22", "12"))
    delta = np.sqrt((e11 + e22) ** 2 + ((e11 - e22) ** 2 + (2.0 * e12) ** 2) / 4.0)   # e = 2
    np.testing.assert_allclose(own, 0.5 * (delta - np.abs(e11 + e22)), rtol=1e-9, atol=1e-15)


def test_the_fma_without_math_fma_is_the_same_rounding(monkeypatch):
    rs = np.random.RandomState(7)
    triples = [tuple(rs.standard_normal(3) * 10.0 ** rs.randint(-8, 8, 3)) for _ in range(2000)]
    triples += [(1.0 + 2.0 ** -30, 1.0 - 2.0 ** -30, -1.0), (1e300, 1e300, -1e300), (np.inf, 1.0, 1.0)]
    want = [wv._fma(a, b, c, real) for a, b, c in triples for real in (np.float32, np.float64)]
    if hasattr(wv.math, "fma"):
        monkeypatch.delattr(wv.math, "fma")
    got = [wv._fma(a, b, c, real) for a, b, c in triples for real in (np.float32, np.float64)]
    assert np.array_equal(np.array(got, dtype=object).astype(np.float64), np.array(want, dtype=object).astype(np.float64))


@requires_gpu
def test_the_time_step_refuses_a_sphere_without_its_radius_and_stray_corrections():
    from woof.hex.mpas_seaice import timestep_v842 as ts
    case = wo.step_cases("r4")[2]                         # the sphere about the North Pole
    fixture = wo.load_step_fixture("r4")[case["name"]]
    dev = sr.device_pools(case, fixture)
    clock = ts.initialize(pools=dev, configs=case["config"], remap_geometry="input", metric_libm=_libm())
    no_radius = {k: v for k, v in case["config"].items() if k not in ("mesh:sphere_radius", "sphere_radius")}
    with pytest.raises(ConfigurationRefusal, match="sphere_radius"):
        ts.advance(pools=dev, configs=no_radius, clock=clock, n_steps=1, metric_libm=_libm())
    with pytest.raises(ValueError, match="corrections"):
        ts.advance(pools=dev, configs=case["config"], clock=clock, n_steps=1, metric_libm=_libm(),
                   weak_corrections=("weak_ridge_dleta",))
    variational = dict(case["config"], config_stress_divergence_scheme="variational")
    with pytest.raises(ValueError, match="weak_corrections"):
        ts.advance(pools=dev, configs=variational, clock=clock, n_steps=1, weak_corrections=("weak_ridge_delta",))


def _python_mutant(old: str, new: str):
    source = Path(wv.__file__).read_text(encoding="utf-8")
    assert source.count(old) == 1, old
    namespace = {"__name__": "weak_v842_mutant", "__package__": "woof.hex.mpas_seaice"}
    exec(compile(source.replace(old, new), wv.__file__, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("old, new", [
    ("acc = _fma(m[i][k], v[k], acc, real)", "acc = real(acc + real(m[i][k] * v[k]))"),
    ("norm = np.sqrt((n[0] * n[0] + n[1] * n[1]) + n[2] * n[2])",
     "norm = np.sqrt(n[0] * n[0] + (n[1] * n[1] + n[2] * n[2]))"),
])
def test_a_mutated_init_is_caught(old, new):
    mutant = _python_mutant(old, new)
    caught = 0
    for precision in BOTH:
        fixture = wo.load_fixture(precision)
        for case in wo.cases(precision, [r for r in wo.CASES if r[0] in wo.NEEDS_GLIBC]):
            real = np.float32 if precision == "r4" else np.float64
            mesh = {**case["pools"]["mesh"], "interiorVertex": case["pools"]["boundary"]["interiorVertex"]}
            out = {n: np.array(case["pools"]["velocity_weak"][n], copy=True) for n in wo.INIT_OUTPUTS}
            mutant["initialize"](mesh=mesh, configs=case["config"], real=real, out=out, metric_libm=_libm())
            caught += sum(wo.word_mismatches(out[n], fixture[case["name"]][f"velocity_weak/{n}"])
                          for n in ("normalVectorPolygon", "normalVectorTriangle"))
    assert caught > 0


@requires_gpu
@pytest.mark.parametrize("old, new", [
    ("s11 = A(s11, M(M(uVelocityEdge, n1), dv));", "s11 = A(s11, M(uVelocityEdge, M(n1, dv)));"),
    # the metric terms are ~1e-3 of the divergence on 10 km cells: a reordering
    # inside one rounds below the sum's ulp, so drop a factor instead
    ("s12 = A(s12, D(M(M(uCellCentre, t), RL(0.5)), sphereRadius));",
     "s12 = A(s12, D(M(uCellCentre, t), sphereRadius));"),
    ("divV = A(divV, D(M(t, S(stress11Vertex, stress22Vertex)), sphereRadius));",
     "divV = A(divV, D(M(t, A(stress11Vertex, stress22Vertex)), sphereRadius));"),
    ("const int j = perCellDelta ? iCell : nCellsSolve - 1;", "const int j = iCell;"),
])
def test_a_mutated_kernel_is_caught(monkeypatch, old, new):
    source = wv.module_source
    assert source("r8").count(old) == 1, old
    monkeypatch.setattr(wv, "module_source", lambda precision="r4": source(precision).replace(old, new))
    wv._module_on.cache_clear()
    try:
        report = wo.measure("r8", metric_libm=_libm())
    finally:
        monkeypatch.undo()
        wv._module_on.cache_clear()
    assert sum(sum(fields.values()) for fields in report.values()) > 0


def _oracle_toolchain():
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_seaice").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MPAS-Seaice oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
@pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64") or not _glibc_244(),
                    reason="the fixtures are x86-64 gfortran's on glibc 2.44 (atan2, asin, sin, cos, tan; "
                           "libgfortran's FMA matmul; arm64 gfortran's MAX/MIN keep another operand on "
                           "signed-zero ties)")
def test_the_oracle_rebuilt_here_writes_the_fixtures(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_weak_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "weak")], check=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_step_remap_fixture.py"),
                    str(tmp_path / "build"), "--weak", "--out", str(tmp_path / "step")], check=True)
    for fresh_dir, pinned_dir in ((tmp_path / "weak", wo.FIXTURE_DIR), (tmp_path / "step", wo.STEP_FIXTURE_DIR)):
        for precision in BOTH:
            fresh = np.load(fresh_dir / f"oracle-{precision}.npz")
            pinned = np.load(pinned_dir / f"oracle-{precision}.npz")
            assert sorted(fresh.files) == sorted(pinned.files)
            for key in pinned.files:
                assert ro.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
