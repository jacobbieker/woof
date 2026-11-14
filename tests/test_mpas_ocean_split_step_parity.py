"""One MPAS-Ocean split-explicit step, word for word against v8.4.2, at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's ``ocn_time_integrator_split``, cut verbatim
out of ``mpas_ocn_time_integration_split.F``, with every module it reaches
compiled verbatim: the momentum and thickness tendencies, the diagnostics, the
equation of state, the ALE thickness, tracer advection and mixing, implicit
vertical mixing, time averaging, and ``mpas_reconstruct`` (cut). It is
compiled with gfortran -O0 -ffp-contract=off at MPAS's default single
precision and at its double precision, and driven by
``tools/mpas_ocean_oracle/run_split_step.F90``. The driver first computes the
starting diagnostics, as MPAS's init does. The outputs are committed
(tests/data/oracles/mpas_ocean_split_step, PROVENANCE.md). When ``GFORTRAN``
(or gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` are set, the fixture
itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A step that agrees with MPAS only to a tolerance, in either precision. Its
   pieces are held on their own elsewhere; this holds their composition: the
   time levels each reads, the order, and the step's own loops.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: split and unsplit, two and three outer
   iterations, the budgets per unit thickness, the salinity floor, the
   prescribed thickness, the sphere's reconstruction.
4. A gate that cannot fail: a mutated evaluation order in the step's loops
   must produce mismatches.
5. A configuration this port does not carry, run silently: refused by name.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.verify import mpas_ocean_split_step_oracle as oo

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": oo.input_digest(case) for p in BOTH for case in oo.cases(p)}


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_branches(precision):
    fixture = oo.load_fixture(precision)
    cases = {case["name"]: case for case in oo.cases(precision)}
    assert set(fixture) == set(cases)
    for name, out in fixture.items():
        case, pools = cases[name], cases[name]["pools"]
        nc, ne = case["dims"]["nCells"], case["dims"]["nEdges"]
        assert all(np.isfinite(v).all() for v in out.values() if v.dtype.kind == "f")
        btr = out["state/normalBarotropicVelocity@2"][:ne]
        if case["config"]["config_time_integrator"] == "unsplit_explicit":
            assert (btr == 0).all()
        else:
            assert np.count_nonzero(btr) > ne // 2
        if case["config"]["config_prescribe_thickness"]:
            assert (out["state/layerThickness@2"][:nc] == pools["state"]["layerThickness"][:nc]).all()
        else:
            assert not (out["state/layerThickness@2"][:nc] == pools["state"]["layerThickness"][:nc]).all()
        zonal, x = out["diagnostics/velocityZonal"][:nc], out["diagnostics/velocityX"][:nc]
        assert (zonal == x).all() != case["config"]["mesh:on_a_sphere"]
        if case["config"]["config_compute_active_tracer_budgets"]:
            assert np.count_nonzero(out["diagnostics/activeTracerHorizontalAdvectionTendency"][:nc])
        if name == "split_jm_rich_bulk_budgets":
            salinity = out["tracers/activeTracers@2"][:nc, :, 1]
            assert np.count_nonzero(salinity == salinity.dtype.type(0.001)) > 0
            assert case["config"]["config_n_ts_iter"] == 3


@pytest.mark.parametrize("overrides, knob", [
    ({"config_time_integrator": "RK4"}, "config_time_integrator"),
    ({"config_use_freq_filtered_thickness": True}, "config_use_freq_filtered_thickness"),
    ({"config_use_standardGM": True}, "config_use_standardGM"),
    ({"config_land_ice_flux_mode": "coupled"}, "config_land_ice_flux_mode"),
])
def test_what_the_step_does_not_carry_is_refused_by_name(overrides, knob):
    from woof.hex.mpas_ocean import split_step_v842 as ss
    case = oo.cases()[0]
    mesh = case["pools"]["mesh"]
    ss.check_config(case["config"], "r4", mesh)
    with pytest.raises(ConfigurationRefusal, match=knob):
        ss.check_config(dict(case["config"], **overrides), "r4", mesh)
    # nested halo extents run (tests/test_mpas_ocean_forward_parity.py holds them)
    nested = dict(mesh, nEdgesArray=np.array([mesh["nEdges"] - 3] + [mesh["nEdges"]] * 3, np.int32))
    ss.check_config(case["config"], "r4", nested)


def test_the_sphere_at_rkind_8_needs_the_glibc_opt_in(monkeypatch):
    from woof.hex.mpas_ocean import split_step_v842 as ss
    case = next(c for c in oo.cases("r8") if c["config"]["mesh:on_a_sphere"])
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "0")
    with pytest.raises(ConfigurationRefusal, match="on_a_sphere"):
        ss.check_config(case["config"], "r8", case["pools"]["mesh"])
    ss.check_config(case["config"], "r4", case["pools"]["mesh"])


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    fixture = oo.load_fixture(precision)
    bad = {}
    for case in oo.cases(precision):
        if not oo.runs(case):
            continue
        got = oo.run_gpu(case)
        want = fixture[case["name"]]
        fields = {k: n for k in want if (n := oo.word_mismatches(got[k], want[k]))}
        if fields:
            bad[case["name"]] = fields
    assert not bad, bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
@pytest.mark.parametrize("original, mutated, case_name, field", [
    # the baroclinic prediction's sum regrouped
    ("A(A(tend[b + k], uNew[b + k]),\n                          D(M(M((rk)split, gravity), S(sshNew[cell2], "
     "sshNew[cell1])), dcEdge[e]))",
     "A(tend[b + k], A(uNew[b + k],\n                          D(M(M((rk)split, gravity), S(sshNew[cell2], "
     "sshNew[cell1])), dcEdge[e])))", "split_defaults", "state/normalBaroclinicVelocity@2"),
    # the barotropic forcing divided by the thickness and dt at once
    ("forcing[e] = D(D(M((rk)split, sum), thick), dt);", "forcing[e] = D(M((rk)split, sum), M(thick, dt));",
     "split_defaults", "diagnostics/barotropicForcing"),
    # the baroclinic average regrouped
    ("M(C(0.5), S(A(bclCur[b + k], ut[k]), M(dt, forcing[e])))",
     "M(C(0.5), A(bclCur[b + k], S(ut[k], M(dt, forcing[e]))))", "split_defaults",
     "state/normalBaroclinicVelocity@2"),
    # f taken at the edge, not its neighbour (a regrouping of this small term is
    # absorbed in the baroclinic prediction, and stage 3 overwrites its output)
    ("sum = A(sum, M(M(weightsOnEdge[e * me2 + j], bcl[eoe * nl + k]), fEdge[eoe]));",
     "sum = A(sum, M(M(weightsOnEdge[e * me2 + j], bcl[eoe * nl + k]), fEdge[e]));", "split_defaults",
     "state/normalBaroclinicVelocity@2"),
    # the mid-step tracer average regrouped
    ("trNew[t] = M(C(0.5), A(trCur[t], temp));", "trNew[t] = A(trCur[t], M(C(0.5), S(temp, trCur[t])));",
     "split_jm_rich_bulk_budgets", "tracers/activeTracers@2"),
    # the salinity floor dropped
    ("trNew[t] = ss_max(C(0.001), trNew[t]);", "trNew[t] = trNew[t];", "split_jm_rich_bulk_budgets",
     "tracers/activeTracers@2"),
    # the final velocity regrouped
    ("uNew[i] = S(A(btrNew[e], M(C(2.0), bclNew[i])), bclCur[i]);",
     "uNew[i] = A(btrNew[e], S(M(C(2.0), bclNew[i]), bclCur[i]));", "split_defaults", "state/normalVelocity@2"),
    # the longitude's cosine and sine swapped
    ("const rk clon = SS_COS(lonCell[c]), slon = SS_SIN(lonCell[c]);",
     "const rk clon = SS_SIN(lonCell[c]), slon = SS_COS(lonCell[c]);", "split_sphere",
     "diagnostics/velocityZonal"),
    # the running mean split
    ("avgSSHGradient[c * 2 + iAvgSGZ - 1] = D(A(M(avgSSHGradient[c * 2 + iAvgSGZ - 1], n), gradZonal[c]), n1);",
     "avgSSHGradient[c * 2 + iAvgSGZ - 1] = A(M(avgSSHGradient[c * 2 + iAvgSGZ - 1], D(n, n1)), D(gradZonal[c], n1));",
     "split_defaults", "forcing/avgSSHGradient"),
])
def test_a_mutated_step_is_caught(monkeypatch, precision, original, mutated, case_name, field):
    from woof.hex.mpas_ocean import split_step_v842
    real = split_step_v842.module_source
    assert real(precision).count(original) == 1
    case = next(c for c in oo.cases(precision) if c["name"] == case_name)
    if not oo.runs(case):
        pytest.skip("the sphere at RKIND 8 needs WOOF_GLIBC239_LIBM64=1")
    monkeypatch.setattr(split_step_v842, "module_source", lambda p="r4": real(p).replace(original, mutated))
    split_step_v842._module_on.cache_clear()
    try:
        got = oo.run_gpu(case)
        assert oo.word_mismatches(got[field], oo.load_fixture(precision)[case_name][field]) > 0
    finally:
        split_step_v842._module_on.cache_clear()


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_ocean").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MPAS-Ocean oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_ocean_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_ocean_oracle" / "make_fixture.py"),
                    str(tmp_path / "build"), "--piece", "split_step", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
