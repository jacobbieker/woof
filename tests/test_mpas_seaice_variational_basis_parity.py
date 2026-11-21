"""MPAS-Seaice's variational basis set-up (Wachspress, PWL), word for word against v8.4.2 (RKIND 4 and 8).

The reference is ``seaice_init_velocity_solver_variational`` with the
Wachspress, PWL, variational_shared and numerics modules compiled verbatim
(``tools/mpas_seaice_oracle/run_variational_basis.F90``) on planar and
spherical blocks, for every basis and integration rule; and the forward model
(``run_seaice_step.F90`` under ``oracle_compute_basis``) with the basis
computed in the init, over N steps. The outputs are committed
(tests/data/oracles/mpas_seaice_variational_basis and
tests/data/oracles/mpas_seaice_step_basis, PROVENANCE.md).

THE BREAKAGES THESE PREVENT
---------------------------
1. A basis whose words are not v8.4.2's: an operation out of the Fortran's
   order (the Wachspress kappa recursion, the numerator derivative's product,
   Heron's formula), a quadrature literal or weight, a word written that v8.4.2
   leaves alone (rows past nEdgesOnCell).
2. The time step running on an input basis when v8.4.2 computes its own.
3. An option v8.4.2 stops on, or silently mis-handles, run instead of refused.
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
from woof.hex.mpas_seaice import variational_basis_v842 as vb
from woof.verify import mpas_seaice_remap_oracle as ro
from woof.verify import mpas_seaice_step_basis_oracle as sb
from woof.verify import mpas_seaice_step_remap_oracle as sr
from woof.verify import mpas_seaice_variational_basis_oracle as so

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")
TANLAT = "velocity_variational/tanLatVertexRotatedOverRadius"


def _glibc_244() -> bool:
    """Whether the host C library is the fixtures' (glibc 2.44, whose binary64
    asin and tan the RKIND 8 metric terms are)."""
    try:
        return os.confstr("CS_GNU_LIBC_VERSION") == "glibc 2.44"
    except (ValueError, OSError, AttributeError):
        return False


def _recorded(fixture_dir: Path) -> dict[str, str]:
    return dict(reversed(line.split()) for line in
                (fixture_dir / "inputs.sha256").read_text(encoding="utf-8").splitlines())


def test_case_inputs_are_the_recorded_bytes():
    assert _recorded(so.FIXTURE_DIR) == {f"{c['name']}/{p}": so.input_digest(c) for p in BOTH for c in so.cases(p)}
    assert _recorded(sb.FIXTURE_DIR) == {f"{c['name']}/{p}": sr.input_digest(c) for p in BOTH for c in sb.cases(p)}


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_branches(precision):
    fixture, logs = so.load_fixture(precision)
    cases = {c["name"]: c for c in so.cases(precision)}
    assert set(fixture) == set(cases) == set(logs)
    rules = {(c["config"]["config_wachspress_integration_type"], c["config"]["config_wachspress_integration_order"])
             for c in cases.values() if c["config"]["config_variational_basis"] == "wachspress"}
    assert {("dunavant", n) for n in (*range(1, 11), 12)} <= rules
    assert {("fekete", n) for n in (1, 2, 3, 4, 5, 6, 8, 9)} <= rules
    assert {("trapezoidal", n) for n in (1, 2, 5)} <= rules
    assert {c["config"]["config_variational_basis"] for c in cases.values()} == {"wachspress", "pwl"}
    assert {c["config"]["mesh:on_a_sphere"] for c in cases.values()} == {True, False}
    assert any(not c["config"]["config_rotate_cartesian_grid"] for c in cases.values())
    sides = set()
    for name, out in fixture.items():
        case = cases[name]
        nc, nv = case["dims"]["nCells"], case["dims"]["nVertices"]
        ne = case["pools"]["mesh"]["nEdgesOnCell"][:nc]
        sides |= set(ne.tolist())
        for field in ("basisGradientU", "basisIntegralsMetric"):
            got = out[f"velocity_variational/{field}"]
            assert np.isfinite(got[:nc][:, :1, :1]).all(), (name, field)
            # the words v8.4.2 does not write keep the input's noise
            assert so.word_mismatches(got[nc], case["pools"]["velocity_variational"][field][nc]) == 0
        assert (out["velocity_variational/cellVerticesAtVertex"][:nv] > 0).any()
        # the odd Dunavant orders' WARN, and nothing else
        odd = (case["config"]["config_variational_basis"] == "wachspress"
               and case["config"]["config_wachspress_integration_type"] == "dunavant"
               and case["config"]["config_wachspress_integration_order"] % 2)
        assert ("odd orders of integration not recommended" in logs[name]) == bool(odd), name
    assert {3, 4, 5, 6, 7, 8, 9} <= sides
    planar_nan = fixture["planar_hex_wachspress"][TANLAT]
    assert np.isnan(planar_nan[:-1]).all()      # the planar metric terms' quirk
    assert np.isfinite(fixture["sphere_hex_north_wachspress"][TANLAT][:-1]).all()


@pytest.mark.parametrize("precision", BOTH)
def test_the_port_matches_the_v842_oracle_word_for_word(precision):
    fixture, logs = so.load_fixture(precision)
    libm = "host" if precision == "r8" and _glibc_244() else "correctly-rounded"
    bad, metric_off = {}, {}
    for case in so.cases(precision):
        got, warnings = so.run_port(case, metric_libm=libm)
        want = fixture[case["name"]]
        for key, words in want.items():
            if m := so.word_mismatches(got[key], words):
                if key == TANLAT and libm == "correctly-rounded":
                    metric_off[case["name"]] = (got[key], words)
                else:
                    bad[(case["name"], key)] = m
        assert ["WARN " + w["message"] for w in warnings] == [
            line for line in logs[case["name"]].splitlines() if line.strip()], case["name"]
    assert not bad, bad
    if precision == "r4":
        assert not metric_off, sorted(metric_off)      # glibc's asinf and tanf are correctly rounded
    # off a glibc 2.44 host, its binary64 asin and tan: at most an ulp, at few vertices
    for name, (got, words) in metric_off.items():
        differ = got.view(np.uint64) != words.view(np.uint64)
        assert differ.sum() <= 3, name
        assert (np.abs(got.view(np.int64)[differ] - words.view(np.int64)[differ]) <= 2).all(), name


def test_the_rkind8_metric_terms_are_refused_without_a_libm():
    case = next(c for c in so.cases("r8") if c["name"] == "sphere_hex_north_wachspress")
    with pytest.raises(ConfigurationRefusal, match="config_include_metric_terms"):
        so.run_port(case)
    so.run_port(next(c for c in so.cases("r8") if c["name"] == "sphere_hex_north_wachspress_no_metric"))
    so.run_port(next(c for c in so.cases("r8") if c["name"] == "planar_hex_wachspress"))   # NaN: every libm's


@pytest.mark.parametrize("knob, value, match", [
    ("config_variational_basis", "linear", "config_variational_basis"),
    ("config_wachspress_integration_type", "gauss", "config_wachspress_integration_type"),
    ("config_wachspress_integration_order", 11, "config_wachspress_integration_order"),
    ("config_wachspress_integration_order", 0, "config_wachspress_integration_order"),
])
def test_what_v842_stops_on_or_mishandles_is_refused_by_name(knob, value, match):
    from woof.hex.mpas_seaice import timestep_v842 as ts
    case = next(c for c in so.cases("r4") if c["name"] == "planar_polygons_wachspress")
    configs = dict(case["config"], **{knob: value})
    if knob == "config_wachspress_integration_order" and value == 0:
        configs["config_wachspress_integration_type"] = "trapezoidal"
    with pytest.raises(ConfigurationRefusal, match=match):
        so.run_port(dict(case, config=configs))
    with pytest.raises(ConfigurationRefusal, match=match):
        ts.check_config(configs)
    fekete = dict(case["config"], config_wachspress_integration_type="fekete", config_wachspress_integration_order=7)
    with pytest.raises(ConfigurationRefusal, match="config_wachspress_integration_order"):
        vb.check_config(fekete)


def test_a_mesh_without_coordinates_or_a_bad_libm_is_refused_up_front():
    case = next(c for c in so.cases("r4") if c["name"] == "planar_hex_wachspress")
    mesh = {k: v for k, v in case["pools"]["mesh"].items() if k not in ("xVertex", "edgesOnCell")}
    with pytest.raises(ConfigurationRefusal, match="xVertex"):
        vb.initialize(mesh=mesh, configs=case["config"], real=np.float32)
    # checked on every path, not only where RKIND 8's metric terms read it
    with pytest.raises(ValueError, match="metric_libm"):
        so.run_port(case, metric_libm="Host")
    with pytest.raises(ValueError, match="metric_libm"):
        so.run_port(dict(case, config=dict(case["config"], config_include_metric_terms=False)), metric_libm="glibc")


def test_a_singular_basis_system_stops_as_v842_does():
    matrix = np.array([[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    with pytest.raises(vb.BasisError, match="singular"):
        vb.solve_linear_basis_system(matrix, np.array([[1.0, 0.0, 0.0]]), np.float64)
    # the pivot v8.4.2 bumps to tiny instead of stopping on
    solution = vb.solve_linear_basis_system(np.array([[[0.0, 1.0], [1.0, 0.0]]]), np.array([[2.0, 3.0]]), np.float64)
    assert solution.tolist() == [[3.0, 2.0]]


def test_the_metric_terms_nan_words_are_the_oracles():
    """0/0 gives x86-64's sign-set NaN (asin, tan and the division carry it); asin
    of |x| > 1 a positive one (gfortran on glibc 2.44, pb)."""
    mesh = {"nVertices": 3, "xVertex": np.array([0.0, 5.0, -5.0, 9.0]), "yVertex": np.zeros(4),
            "zVertex": np.zeros(4), "sphere_radius": 0.0}
    for real, sign, plain in ((np.float32, 0xFFC00000, 0x7FC00000), (np.float64, 0xFFF8000000000000,
                                                                       0x7FF8000000000000)):
        got = vb.metric_terms(mesh, rotate=True, include=True, real=real, metric_libm=None)
        words = got.view(np.uint32 if real is np.float32 else np.uint64).tolist()
        assert words == [sign, plain, plain], real
        mesh2 = dict(mesh, sphere_radius=2.0)
        words = vb.metric_terms(mesh2, rotate=True, include=True, real=real, metric_libm=None).view(
            np.uint32 if real is np.float32 else np.uint64).tolist()
        assert words[1:] == [plain, plain] and words[0] == 0


def test_the_opt_in_corrections():
    # planar metric terms: zero rather than v8.4.2's NaN (sphere_radius 0)
    case = next(c for c in so.cases("r8") if c["name"] == "planar_hex_wachspress")
    got, _ = so.run_port(case, corrections=("planar_metric_terms",))
    assert (got[TANLAT][:-1] == 0.0).all()
    with pytest.raises(ValueError, match="corrections"):
        so.run_port(case, corrections=("planar",))
    # periodic local coordinates: a cell whose vertices sit a period away gets the unwrapped cell's basis
    case = next(c for c in so.cases("r8") if c["name"] == "planar_polygons_wachspress")
    base, _ = so.run_port(case)
    period = 4.0e6
    shifted = {k: (dict(v) if isinstance(v, dict) else v) for k, v in case.items()}
    mesh = shifted["pools"]["mesh"] = dict(case["pools"]["mesh"])
    mesh["xVertex"] = mesh["xVertex"].copy()
    first = mesh["verticesOnCell"][0, :mesh["nEdgesOnCell"][0]] - 1
    mesh["xVertex"][first] += period
    shifted["config"] = dict(case["config"], **{"mesh:is_periodic": True, "mesh:x_period": period,
                                                "mesh:y_period": 0.0})
    quirk, _ = so.run_port(shifted)
    fixed, _ = so.run_port(shifted, corrections=("periodic_local_coords",))
    bare = dict(shifted, config=dict(case["config"], is_periodic=True, x_period=period, y_period=0.0))
    fixed_bare, _ = so.run_port(bare, corrections=("periodic_local_coords",))
    key = "velocity_variational/basisIntegralsMetric"
    n = int(mesh["nEdgesOnCell"][0])
    assert not np.allclose(quirk[key][0, :n, :n], base[key][0, :n, :n])
    np.testing.assert_allclose(fixed[key][0, :n, :n], base[key][0, :n, :n], rtol=1e-9)
    assert so.word_mismatches(fixed_bare[key], fixed[key]) == 0      # the bare config keys too


def _mutant(old: str, new: str):
    """variational_basis_v842 with ``old`` (once in its source) replaced by ``new``."""
    source = Path(vb.__file__).read_text(encoding="utf-8")
    assert source.count(old) == 1, old
    namespace = {"__name__": "variational_basis_mutant", "__package__": "woof.hex.mpas_seaice"}
    exec(compile(source.replace(old, new), vb.__file__, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("old, new", [
    ("kappa[:, j] = (kappa[:, j - 1] * (A[:, i2]", "kappa[:, j] = kappa[:, j - 1] * ((A[:, i2]"),
    ("sub[:, s] = np.sqrt(((sp * (sp - a)) * (sp - b)) * (sp - c))",
     "sub[:, s] = np.sqrt((sp * (sp - a)) * ((sp - b) * (sp - c)))"),
    ("jw = (jac * w[p])[:, None, None]", "jw = (w[p] * jac)[:, None, None] * real(1.0000001)"),
    ("out[\"basisGradientU\"][cells, :, :n] = gu", "out[\"basisGradientU\"][cells, :n, :n] = gu[:, :n, :]"),
])
def test_a_mutated_basis_is_caught(monkeypatch, old, new):
    mutant = _mutant(old, new)
    fixture, _ = so.load_fixture("r8")
    caught = 0
    for case in so.cases("r8"):
        if case["config"]["config_include_metric_terms"] and case["config"]["mesh:on_a_sphere"]:
            continue
        out = {name: np.array(case["pools"]["velocity_variational"][name], copy=True) for name in so.OUTPUTS}
        got, _ = mutant["initialize"](mesh=dict(case["pools"]["mesh"]), configs=case["config"], real=np.float64,
                                      out=out, metric_libm="correctly-rounded")
        caught += sum(so.word_mismatches(got[name], fixture[case["name"]][f"velocity_variational/{name}"])
                      for name in so.OUTPUTS[:5])
    assert caught > 0


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_the_time_step_computes_the_basis_word_for_word_over_the_steps(precision):
    rows = sb.CASES if precision == "r4" or _glibc_244() else tuple(
        r for r in sb.CASES if r[0] not in sb.NEEDS_GLIBC)
    report = sb.measure(precision, rows=rows)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@pytest.mark.parametrize("precision", BOTH)
def test_the_step_fixture_holds_the_computed_basis(precision):
    fixture = sb.load_fixture(precision)
    cases = {c["name"]: c for c in sb.cases(precision)}
    assert set(fixture) == set(cases)
    for name, out in fixture.items():
        for key in ("velocity_variational/basisGradientU", "velocity_variational/basisIntegralsMetric",
                    "velocity_variational/cellVerticesAtVertex", "velocity_solver/uVelocity"):
            assert key in out, (name, key)
        assert int(out["mesh/itimestep"][0]) == cases[name]["config"]["oracle_n_steps"]
    assert {c["config"]["config_variational_basis"] for c in cases.values()} == {"wachspress", "pwl"}
    assert {c["config"]["mesh:on_a_sphere"] for c in cases.values()} == {True, False}


@requires_gpu
def test_a_step_on_the_input_basis_is_caught():
    import cupy as cp
    from woof.hex.mpas_seaice import timestep_v842 as ts
    case = sb.cases("r4")[0]
    fixture = sb.load_fixture("r4")[case["name"]]
    dev = sr.device_pools(case, fixture)
    clock = ts.initialize(pools=dev, configs=case["config"], remap_geometry="input", basis="input")
    ts.advance(pools=dev, configs=case["config"], clock=clock, n_steps=int(case["config"]["oracle_n_steps"]))
    key = "velocity_solver/uVelocity"
    assert sr.word_mismatches(cp.asnumpy(dev["velocity_solver"]["uVelocity"]), fixture[key]) > 0


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
                    reason="the fixtures are x86-64 gfortran's on glibc 2.44 (asin, tan; arm64 gfortran's "
                           "MAX/MIN keep another operand on signed-zero ties)")
def test_the_oracle_rebuilt_here_writes_the_fixtures(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_variational_basis_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "basis")], check=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_step_remap_fixture.py"),
                    str(tmp_path / "build"), "--basis", "--out", str(tmp_path / "step")], check=True)
    for fresh_dir, pinned_dir in ((tmp_path / "basis", so.FIXTURE_DIR), (tmp_path / "step", sb.FIXTURE_DIR)):
        for precision in BOTH:
            fresh = np.load(fresh_dir / f"oracle-{precision}.npz")
            pinned = np.load(pinned_dir / f"oracle-{precision}.npz")
            assert sorted(fresh.files) == sorted(pinned.files)
            for key in pinned.files:
                assert ro.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
