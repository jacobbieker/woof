"""MPAS-Ocean's ocn_diagnostic_solve, word for word against v8.4.2, at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's own ``ocn_diagnostic_solve`` (cut
verbatim out of ``mpas_ocn_diagnostics.F`` with ``ocn_diagnostics_init`` and
``ocn_compute_land_ice_flux_input_fields``), with the real equation of state
and ``ocn_relativeVorticity_circulation``, compiled with gfortran -O0
-ffp-contract=off at MPAS's default single precision and at its double
precision, and driven by ``tools/mpas_ocean_oracle/run_diagnostics.F90``.
The outputs are committed (tests/data/oracles/mpas_ocean_diagnostics,
PROVENANCE.md). When ``GFORTRAN`` (or gfortran on PATH) and
``WOOF_MPAS_V842_SOURCE`` are set, the fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance, in either precision,
   or that writes a word the Fortran leaves alone (every field of the block
   is compared, its missing-neighbour element too).
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: each pressure-gradient path, APVM, the
   vertex kinetic energy, KPP's surface layer ending inside and below a
   column, the land-ice pressure package, nested halo extents.
4. A port that "fixes" v8.4.2: the transport and GM-bolus vertical
   velocities start from the words below the column, the surface velocity
   copy runs over the cell count, KPP weighs by FRACTION, the sentinels are
   the single-precision -1e34, and layerThicknessEdge reads a boundary edge's
   missing cell before its sentinel is written.
5. A gate that cannot fail. Mutating one evaluation order must produce
   mismatches.
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
from woof.verify import mpas_ocean_diagnostics_oracle as oo

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
    configs = [c["config"] for c in cases.values()]
    assert {c["config_pressure_gradient_type"] for c in configs} >= {"pressure_and_zmid", "MontgomeryPotential",
                                                                    "Jacobian_from_TS", "Jacobian_from_density"}
    for key in ("config_use_cvmix_kpp", "config_include_KE_vertex", "config_use_frazil_ice_formation"):
        assert {c[key] for c in configs} == {True, False}, key
    assert {c["config_land_ice_flux_mode"] for c in configs} == {"off", "pressure_only"}
    assert any(c["config_apvm_scale_factor"] > 0 for c in configs)
    sentinel = np.float32(-1e34).astype(oo.PRECISIONS[precision])
    kpp_inside = kpp_below = 0
    for name, out in fixture.items():
        case = cases[name]
        d = case["dims"]
        nc, ne = d["nCells"], d["nEdges"]
        pools = case["pools"]
        assert all(np.isfinite(v).all() for k, v in out.items()
                   if k.startswith("diagnostics/") and "normalized" not in k)
        # the sentinels: the single-precision -1e34, widened
        assert (out["state/layerThickness"][nc] == sentinel).all()
        assert (out["state/normalVelocity"][ne] == sentinel).all()
        # a boundary edge's thickness reads its missing cell before the sentinel is written
        boundary = np.flatnonzero(pools["mesh"]["cellsOnEdge"][:ne, 1] == nc + 1)
        boundary = boundary[pools["mesh"]["maxLevelEdgeTop"][boundary] > 0]
        lte = out["diagnostics/layerThicknessEdge"]
        assert all((np.abs(lte[e, :pools["mesh"]["maxLevelEdgeTop"][e]]) < 1.0e30).all() for e in boundary)
        # below the column only vertVelocityTop is zeroed: the transport sum starts from the words there
        mlc = pools["mesh"]["maxLevelCell"]
        cells2 = int(pools["mesh"]["nCellsArray"][1])
        rows = np.arange(cells2)
        assert (out["diagnostics/vertVelocityTop"][rows, mlc[:cells2]] == 0).all()
        prior = pools["diagnostics"]["vertTransportVelocityTop"]
        assert (out["diagnostics/vertTransportVelocityTop"][rows, mlc[:cells2]] == prior[rows, mlc[:cells2]]).all()
        if case["config"]["config_use_cvmix_kpp"]:
            depth = out["diagnostics/indexSurfaceLayerDepth"][: int(pools["mesh"]["nCellsArray"][0])]
            kpp_inside += int((depth != np.float32(-9.e30)).sum())
            kpp_below += int((depth == np.float32(-9.e30)).sum())
        elif case["config"]["config_apvm_scale_factor"] == 0:
            # the surface velocity copy runs over nCellsArray(4) edges, not nEdges
            nvsl = out["diagnostics/normalVelocitySurfaceLayer"]
            assert (nvsl[:nc] == out["state/normalVelocity"][:nc, 0]).all()
            assert (nvsl[nc:ne] == pools["diagnostics"]["normalVelocitySurfaceLayer"][nc:ne]).all()
        if case["config"]["config_land_ice_flux_mode"] == "pressure_only":
            frac = pools["forcing"]["landIceFraction"][:nc]
            assert (out["forcing/landIceMask"][:nc] == (frac >= 0.5)).all()
    assert kpp_inside > 0 and kpp_below > 0


def test_unported_options_are_refused_by_name():
    from woof.hex.mpas_ocean import diagnostics_v842 as dg
    base = dict(oo.BASE_CONFIG)
    dg.check_config(base)
    for knob, value in (("config_land_ice_flux_mode", "standalone"), ("config_land_ice_flux_mode", "coupled"),
                        ("config_land_ice_flux_mode", "data"), ("config_eos_type", "wright"),
                        ("config_pressure_gradient_type", "Jacobian_POM")):
        with pytest.raises(ConfigurationRefusal) as exc:
            dg.check_config(dict(base, **{knob: value}))
        assert knob in str(exc.value)


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
@pytest.mark.parametrize("original, mutated, case_name, field, precisions", [
    # the transport vertical velocity zeroed below the column ("fixing" v8.4.2)
    ("    w[top] = C(0.0);\n", "    w[top] = C(0.0);\n    wT[top] = C(0.0);\n", "zmid_linear",
     "diagnostics/vertTransportVelocityTop", BOTH),
    # KPP's partial layer weighed by its fractional part instead of FRACTION
    ("    const rk frac = diag_fraction(rSurfaceLayer);", "    const rk frac = S(rSurfaceLayer, (rk)whole);",
     "jts_jm_apvm_kpp_land_ice", "diagnostics/tracersSurfaceLayerValue", BOTH),
    ("M(M(diag_fraction(rSurfaceLayer), u[kp - 1]), h[kp - 1])", "M(M(S(rSurfaceLayer, (rk)whole), u[kp - 1]), h[kp - 1])",
     "jts_jm_apvm_kpp_land_ice", "diagnostics/normalVelocitySurfaceLayer", BOTH),
    # the sentinel as an RKIND literal: RKIND 8 widens the binary32 -1e34
    ("#define SENTINEL_DEFAULT ((rk)-1e34f)", "#define SENTINEL_DEFAULT C(-1e34)", "zmid_linear",
     "state/layerThickness", ("r8",)),
    # the vertex kinetic energy's square regrouped
    ("ke = A(ke, M(r_tmp, M(u, u)));", "ke = A(ke, M(M(r_tmp, u), u));", "montgomery_jm_rk4_kpp_deep",
     "diagnostics/kineticEnergyCell", BOTH),
    # the divergence's area division moved
    ("const rk r_tmp = M(M(dv, u), invAreaCell1);", "const rk r_tmp = M(dv, M(u, invAreaCell1));", "zmid_linear",
     "diagnostics/divergence", BOTH),
    # relative vorticity as the circulation over the area
    ("vort = A(vort, M(M(sign, r_tmp), invAreaTri1));", "vort = M(A(circ, M(sign, r_tmp)), invAreaTri1);",
     "zmid_linear", "diagnostics/relativeVorticity", BOTH),
    # the vertex thickness divided by the area instead of multiplied by its inverse
    ("    h = M(h, invAreaTri1);", "    h = D(h, areaTriangle[iVertex]);", "zmid_linear",
     "diagnostics/normalizedRelativeVorticityEdge", BOTH),
    # APVM's normal gradient taken the other way
    ("M(S(nrvCell[cell2 * nl + k], nrvCell[cell1 * nl + k]), D(C(1.0), dcEdge[iEdge]))",
     "M(S(nrvCell[cell1 * nl + k], nrvCell[cell2 * nl + k]), D(C(1.0), dcEdge[iEdge]))", "jdens_linear_apvm",
     "diagnostics/normalizedRelativeVorticityEdge", BOTH),
    # zMid's half layers added one at a time
    ("zm[k] = A(zm[k + 1], M(C(0.5), A(h[k + 1], h[k])));",
     "zm[k] = A(A(zm[k + 1], M(C(0.5), h[k + 1])), M(C(0.5), h[k]));", "zmid_linear", "diagnostics/zMid", BOTH),
    # the Montgomery potential's reciprocal difference over a common denominator
    ("M(pTop, S(D(C(1.0), rho[k]), D(C(1.0), rho[k - 1])))",
     "M(pTop, D(S(rho[k - 1], rho[k]), M(rho[k], rho[k - 1])))", "montgomery_jm_rk4_kpp_deep",
     "diagnostics/montgomeryPotential", BOTH),
    # the Richardson number without its 1e-10
    ("ri[k] = D(bv[k], A(M(shearMean, shearMean), C(1.0e-10)));", "ri[k] = D(bv[k], M(shearMean, shearMean));",
     "zmid_linear", "diagnostics/RiTopOfCell", BOTH),
    # the land-ice fraction kept below 0.5
    ("        landIceFraction[iCell] = C(0.0);\n", "", "jts_jm_apvm_kpp_land_ice", "forcing/landIceFraction", BOTH),
])
def test_a_mutated_port_is_caught(monkeypatch, precision, original, mutated, case_name, field, precisions):
    if precision not in precisions:
        pytest.skip(f"the {precision} words do not depend on this expression")
    from woof.hex.mpas_ocean import diagnostics_v842
    real = diagnostics_v842.module_source
    assert real(precision).count(original) == 1
    monkeypatch.setattr(diagnostics_v842, "module_source", lambda p="r4": real(p).replace(original, mutated))
    diagnostics_v842._module_on.cache_clear()
    try:
        case = next(c for c in oo.cases(precision) if c["name"] == case_name)
        got = oo.run_gpu(case)
        assert oo.word_mismatches(got[field], oo.load_fixture(precision)[case_name][field]) > 0
    finally:
        diagnostics_v842._module_on.cache_clear()


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
                    str(tmp_path / "build"), "--piece", "diagnostics", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
