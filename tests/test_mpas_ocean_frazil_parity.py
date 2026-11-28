"""MPAS-Ocean's frazil ice formation, word for word against v8.4.2, at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's own ``mpas_ocn_frazil_forcing.F``
(compiled verbatim with the real equation of state) with gfortran -O0
-ffp-contract=off at MPAS's default single precision and at its double
precision, driven by ``tools/mpas_ocean_oracle/run_frazil.F90``: the build of
the frazil arrays, then the thickness and active-tracer tendencies they add.
The outputs are committed (tests/data/oracles/mpas_ocean_frazil,
PROVENANCE.md). When ``GFORTRAN`` (or gfortran on PATH) and
``WOOF_MPAS_V842_SOURCE`` are set, the fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance, in either precision, or
   only under strict numerics: the word-exact gate also runs in fast and
   strict child processes.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: frazil formed and melted back, the
   thickness limit, the maximum depth, warm columns skipped, land ice (the
   cavity coefficients, the land-ice frazil mass, either side switched off),
   the surface pressure on and off, halo cells left alone.
4. A gate that cannot fail: a mutated freezing point, melt salinity, depth
   cutoff, surface pressure or land-ice accumulation must produce mismatches.
5. The default changed: with ``config_use_frazil_ice_formation`` off (the
   Registry default) nothing is written.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _numerics_child import MODES, run as run_child
from conftest import requires_gpu
import _oracle_toolchains
from woof.verify import mpas_ocean_frazil_oracle as oo

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")
TH = "forcing/frazilLayerThicknessTendency"


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": oo.input_digest(case) for p in BOTH for case in oo.cases(p)}


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_branches(precision):
    fixture = oo.load_fixture(precision)
    cases = {case["name"]: case for case in oo.cases(precision)}
    assert set(fixture) == set(cases)
    skipped_warm = halo = 0
    for name, out in fixture.items():
        case = cases[name]
        nc, solved = case["dims"]["nCells"], int(case["pools"]["mesh"]["nCellsArray"][1])
        th = out[TH]
        assert (th < 0).any() and (th > 0).any(), name  # frazil formed and melted back
        assert (th[solved:] == 0).all() and np.isfinite(th).all()
        state, given = out, case["pools"]["state"]
        halo += int(solved < nc and (state["state/accumulatedFrazilIceMass@2"][solved:nc]
                                     == given["accumulatedFrazilIceMass@2"][solved:nc]).all())
        temperature = case["pools"]["tracers"]["activeTracers"][:, :, case["pools"]["tracers"]["index_temperature"] - 1]
        warm = temperature[:solved].min(axis=1) > case["config"]["config_frazil_maximum_freezing_temperature"]
        skipped_warm += int(warm.any() and (th[:solved][warm] == 0).all())
        if case["pools"]["forcing"]["landIceMask"] is not None:
            mask = case["pools"]["forcing"]["landIceMask"][:solved] == 1
            li_new = out["state/accumulatedLandIceFrazilMass@2"][:solved]
            li_given = given["accumulatedLandIceFrazilMass@2"][:solved]
            assert (li_new[~mask] == li_given[~mask]).all()
            if case["config"]["config_frazil_under_land_ice"]:
                assert (li_new[mask] != li_given[mask]).any()
            else:
                assert (li_new[mask] == li_given[mask]).all() and (th[:solved][mask] == 0).all()
            if not case["config"]["config_frazil_in_open_ocean"]:
                assert (th[:solved][~mask] == 0).all()
        sp = out["forcing/frazilSurfacePressure"]
        if case["config"]["config_frazil_use_surface_pressure"]:
            # the weight of the accumulated frazil, one rounded product per cell (numpy's is IEEE)
            real = sp.dtype.type
            want = state["state/accumulatedFrazilIceMass@2"][:solved] * real(9.80616)
            assert oo.word_mismatches(sp[:solved], want) == 0
            assert (sp[solved:] == case["pools"]["forcing"]["frazilSurfacePressure"][solved:]).all()
        else:
            assert (sp == case["pools"]["forcing"]["frazilSurfacePressure"]).all()
    assert halo >= 1 and skipped_warm >= 1


def test_the_limit_and_depth_cutoff_bind():
    """The fractional thickness limit decides some formed layers (r8, the
    formula evaluated here), and the maximum depth leaves deep cold layers be."""
    cut = 0
    for case in oo.cases("r8"):
        out = oo.load_fixture("r8")[case["name"]]
        cfg, st = case["config"], case["pools"]["state"]
        solved = int(case["pools"]["mesh"]["nCellsArray"][1])
        th = out[TH][:solved]
        deep = -case["pools"]["diagnostics"]["zMid"][:solved] >= (-st["ssh"][:solved, None]
                                                                   + cfg["config_frazil_maximum_depth"])
        mlc = case["pools"]["mesh"]["maxLevelCell"][:solved]
        below = deep & (np.arange(th.shape[1])[None, :] < mlc[:, None])
        if below.any():
            # levels below the cutoff are only written when no level is above it
            shallow_exists = (~deep[:, 0])
            assert (th[below & shallow_exists[:, None]] == 0).all()
            cut += int(np.count_nonzero(below & shallow_exists[:, None]))
    assert cut > 0
    case = next(c for c in oo.cases("r8") if c["name"] == "configured_limits")
    out = oo.load_fixture("r8")[case["name"]]
    cfg = case["config"]
    h = case["pools"]["state"]["layerThickness"]
    formed = -out[TH] * case["dt"] * case["pools"]["diagnostics"]["density"] / cfg["config_frazil_ice_density"]
    at_limit = np.isclose(formed, h * cfg["config_frazil_fractional_thickness_limit"], rtol=1e-12, atol=0)
    assert at_limit.sum() >= 3


@requires_gpu
def test_frazil_off_writes_nothing():
    """The Registry default (frazil off): the build and the tendencies return at once."""
    import cupy as cp
    from woof.hex.mpas_ocean import frazil_v842 as fz
    from woof.verify import mpas_ocean_stream as stream
    case = oo.cases("r4")[0]
    config = dict(case["config"], config_use_frazil_ice_formation=False)
    assert fz.DEFAULT_CONFIGS["config_use_frazil_ice_formation"] is False and not fz.is_on({})
    dev = stream.on_device(oo.FIELDS, case)
    before = {(p, n): cp.asnumpy(dev[p][n]) for p, n, k, _ in oo.FIELDS if k == oo.REAL and dev[p][n] is not None}
    fz.build_arrays(dt=case["dt"], mesh=dev["mesh"], state=dev["state"], tracers=dev["tracers"],
                    diagnostics=dev["diagnostics"], forcing=dev["forcing"], configs=config)
    fz.layer_thickness_tend(mesh=dev["mesh"], forcing=dev["forcing"], tend=dev["tend"]["layerThickness"],
                            configs=config)
    fz.active_tracers_tend(mesh=dev["mesh"], forcing=dev["forcing"], tend=dev["tracersTend"]["activeTracersTend"],
                           index_temperature=1, index_salinity=2, configs=config)
    for (p, n), words in before.items():
        assert oo.word_mismatches(cp.asnumpy(dev[p][n]), words) == 0, (p, n)


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_v842_oracle_word_for_word"], numerics, exact)


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
@pytest.mark.parametrize("original, mutated, field", [
    # the freezing point's pressure term dropped
    ("A(A(A(c[0], M(c[1], salinity)), M(c[2], pressureOffset)), M(M(c[3], pressureOffset), salinity))",
     "A(A(c[0], M(c[1], salinity)), M(M(c[3], pressureOffset), salinity))", TH),
    # the melted salt by a product before the mean salinity
    ("M(melted, D(sumSalt, sumThickness))", "D(M(melted, sumSalt), sumThickness)",
     "forcing/frazilSalinityTendency"),
    # the depth cutoff measured with the sea surface's sign flipped
    ("const rk depthLimit = A(-ssh[iCell], maximumDepth);",
     "const rk depthLimit = A(ssh[iCell], maximumDepth);", TH),
    # the frazil's surface pressure without gravity
    ("M(accumulatedFrazilIceMassNew[iCell], GRAVITY)", "accumulatedFrazilIceMassNew[iCell]",
     "forcing/frazilSurfacePressure"),
    # the land-ice frazil mass never accumulated
    ("if (underLandIce)\n", "if (false)\n", "state/accumulatedLandIceFrazilMass@2"),
    # the melted thickness not limited by the frozen sum
    ("melted = fz_min(melted, sumThickness);", "", "forcing/frazilTemperatureTendency"),
])
def test_a_mutated_kernel_is_caught(monkeypatch, precision, original, mutated, field):
    from woof.hex.mpas_ocean import frazil_v842
    real = frazil_v842.module_source
    assert real(precision).count(original) == 1
    monkeypatch.setattr(frazil_v842, "module_source", lambda p="r4": real(p).replace(original, mutated))
    frazil_v842._module_on.cache_clear()
    try:
        want = oo.load_fixture(precision)
        mismatches = sum(oo.word_mismatches(oo.run_gpu(case)[field], want[case["name"]][field])
                         for case in oo.cases(precision))
        assert mismatches > 0
    finally:
        frazil_v842._module_on.cache_clear()


@requires_gpu
def test_the_tendencies_add_in_place(monkeypatch):
    """The thickness and tracer tendencies are added into what the tendency
    arrays hold (the last terms of ocn_tend_thick and ocn_tend_tracer): the
    oracle's tendencies move from their inputs only where a frazil tendency
    is nonzero (cells 1..nCellsArray(2), levels 1..maxLevelCell), and without
    the adds the port leaves them as they came."""
    from woof.hex.mpas_ocean import frazil_v842
    monkeypatch.setattr(frazil_v842, "layer_thickness_tend", lambda **kw: None)
    monkeypatch.setattr(frazil_v842, "active_tracers_tend", lambda **kw: None)
    moved = 0
    for case in oo.cases("r8"):
        got = oo.run_gpu(case)
        want = oo.load_fixture("r8")[case["name"]]
        tend, given = want["tend/layerThickness"], case["pools"]["tend"]["layerThickness"]
        solved = int(case["pools"]["mesh"]["nCellsArray"][1])
        mlc = case["pools"]["mesh"]["maxLevelCell"]
        reach = (np.arange(tend.shape[0])[:, None] < solved) & (np.arange(tend.shape[1])[None, :] < mlc[:, None])
        changed = tend != given
        assert not (changed & ~(reach & (want[TH] != 0))).any(), case["name"]
        moved += int(changed.sum())
        assert oo.word_mismatches(got["tend/layerThickness"], given) == 0
        assert oo.word_mismatches(got["tracersTend/activeTracersTend"],
                                  case["pools"]["tracersTend"]["activeTracersTend"]) == 0
    assert moved > 0


def _oracle_toolchain() -> tuple[str, Path] | None:
    return _oracle_toolchains.mpas_v842("core_ocean")


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MPAS-Ocean oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_ocean_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_ocean_oracle" / "make_fixture.py"),
                    str(tmp_path / "build"), "--piece", "frazil", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
