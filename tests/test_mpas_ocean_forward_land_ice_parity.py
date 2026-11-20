"""MPAS-Ocean's forward mode with the land-ice fluxes on ('standalone',
'coupled'), word for word against v8.4.2 over several steps, at RKIND 4 and 8.

The reference is the forward oracle (``tools/mpas_ocean_oracle/run_forward.F90``)
with ``ocn_surface_land_ice_fluxes_build_arrays`` called before each step, as
``ocn_forward_mode_run`` calls it: per step the land-ice fluxes, ``ocn_timestep``
(whose diagnostics compute the land-ice input fields, whose tendencies add the
land-ice forcing, whose split step updates the effective density and the
coupler's means in 'coupled') and the time-level shift, every module compiled
verbatim (tests/data/oracles/mpas_ocean_forward_land_ice, PROVENANCE.md).

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance, in either precision, or
   only under strict numerics: the word-exact gate also runs in fast and
   strict child processes.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: 'standalone' with ISOMIP and with Jenkins,
   'coupled' (the effective density, the coupler's means), no step (the
   startup build), odd and even step counts, nested halos.
4. A driver that cannot fail: the land-ice build skipped (per step or at
   startup), the land-ice state not shifted between steps, or the effective
   density or the coupler's means skipped must produce mismatches.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _numerics_child import MODES, run as run_child
from conftest import requires_gpu
from woof.verify import mpas_ocean_forward_land_ice_oracle as oo

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
    seen = set()
    for name, out in fixture.items():
        case = cases[name]
        cfg, pools = case["config"], case["pools"]
        nc = case["dims"]["nCells"]
        mode = cfg["config_land_ice_flux_mode"]
        seen.add((mode, cfg["config_land_ice_flux_formulation"]))
        assert np.isfinite(out["state/ssh"][:nc]).all()
        assert (out["diagnostics/topDrag"] != pools["diagnostics"]["topDrag"]).any()
        if mode == "standalone":
            # the build accumulates the melt under the ice (into level 2 before any shift)
            given = pools["state"]["accumulatedLandIceMass"][:nc]
            assert any((out[f"state/accumulatedLandIceMass{level}"][:nc] != given).any() for level in ("", "@2")), name
        else:
            # the coupler's fluxes are kept; the means and the effective density move
            assert (out["forcing/landIceFreshwaterFlux"] == pools["forcing"]["landIceFreshwaterFlux"]).all()
            for field in ("forcing/avgEffectiveDensityInLandIce", "forcing/avgLandIceBoundaryLayerTracers",
                          "state/effectiveDensityInLandIce"):
                assert (out[field][:nc] != pools[field.split("/")[0]][field.split("/")[1]][:nc]).any(), field
    assert seen == {("standalone", "Jenkins"), ("standalone", "ISOMIP"), ("coupled", "Jenkins")}
    assert any(c["config"]["oracle_n_steps"] == 0 for c in cases.values())
    assert any(c["config"]["oracle_n_steps"] % 2 for c in cases.values())
    assert any(c["config"]["oracle_n_steps"] % 2 == 0 for c in cases.values())


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word_over_the_steps(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert len(report) == len(oo.CASES) and not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_v842_oracle_word_for_word_over_the_steps"], numerics, exact)


@requires_gpu
@pytest.mark.parametrize("mutation", ["no_build", "land_ice_not_shifted", "no_effective_density", "no_means",
                                      "no_startup_build"])
def test_a_mutated_driver_is_caught(monkeypatch, mutation):
    from woof.hex.mpas_ocean import land_ice_v842
    case = oo.cases("r8")[0]
    if mutation == "no_build":
        monkeypatch.setattr(land_ice_v842, "build_arrays", lambda **kw: None)
        fields = ("state/accumulatedLandIceMass", "forcing/landIceFreshwaterFlux", "state/layerThickness")
    elif mutation == "no_startup_build":
        from woof.hex.mpas_ocean import forward_v842 as fw
        real_build = land_ice_v842.build_arrays
        monkeypatch.setattr(fw, "cold_start", _without(fw.cold_start, land_ice_v842, "build_arrays", real_build))
        case = next(c for c in oo.cases("r8") if c["config"]["oracle_n_steps"] == 0)
        fields = ("state/accumulatedLandIceMass@2", "forcing/landIceFreshwaterFlux")
    elif mutation == "land_ice_not_shifted":
        monkeypatch.setattr(land_ice_v842, "TWO_LEVEL_FLUXES", ())
        fields = ("state/accumulatedLandIceMass", "state/accumulatedLandIceMass@2")
    else:
        case = next(c for c in oo.cases("r8") if c["config"]["config_land_ice_flux_mode"] == "coupled")
        if mutation == "no_effective_density":
            monkeypatch.setattr(land_ice_v842, "effective_density_update", lambda **kw: None)
            fields = ("state/effectiveDensityInLandIce", "forcing/avgEffectiveDensityInLandIce")
        else:
            monkeypatch.setattr(land_ice_v842, "coupled_means", lambda **kw: None)
            fields = ("forcing/avgLandIceBoundaryLayerTracers", "forcing/avgLandIceTracerTransferVelocities")
    got = oo.run_gpu(case)
    want = oo.load_fixture("r8")[case["name"]]
    assert sum(oo.word_mismatches(got[f], want[f]) for f in fields) > 0


def _without(function, module, name, real):
    """``function`` with ``module.name`` a no-op while it runs."""
    def wrapped(**kw):
        setattr(module, name, lambda **_: None)
        try:
            return function(**kw)
        finally:
            setattr(module, name, real)
    return wrapped


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
                    str(tmp_path / "build"), "--piece", "forward_land_ice", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
