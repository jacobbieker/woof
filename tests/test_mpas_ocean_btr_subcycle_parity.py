"""MPAS-Ocean's split-explicit barotropic stage, word for word against v8.4.2, at RKIND 4 and 8.

The reference is stage 2 of MPAS-Model v8.4.2's ``ocn_time_integrator_split``
(the barotropic velocity and SSH, explicitly subcycled), cut verbatim out of
``mpas_ocn_time_integration_split.F`` with the routine's own declarations and
config lookups, compiled with gfortran -O0 -ffp-contract=off at MPAS's
default single precision and at its double precision, driven by
``tools/mpas_ocean_oracle/run_btr_subcycle.F90``. The outputs are committed
(tests/data/oracles/mpas_ocean_btr_subcycle, PROVENANCE.md). When
``GFORTRAN`` (or gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` are set, the
fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance, in either precision.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: the predictor and correctors, a skipped
   predictor, the filtered forcing, the SSH corrector, nested halo extents,
   the velocity correction on and off, the unsplit integrator.
4. A gate that cannot fail: a mutated evaluation order, the SSH corrector's
   cell order, or a wrong normalisation must produce mismatches.
5. A configuration on which v8.4.2 reads outside its halo arrays, or an
   uninitialised scratch word, run silently: refused by name.
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
from woof.verify import mpas_ocean_btr_subcycle_oracle as oo

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
        pools, ne = cases[name]["pools"], cases[name]["dims"]["nEdges"]
        assert all(np.isfinite(v).all() for v in out.values())
        moved = out["state/normalBarotropicVelocity@2"][:ne]
        if name == "unsplit":
            assert (moved == 0).all()
            assert (out["state/normalVelocity@2"][:ne] == pools["state"]["normalBaroclinicVelocity@2"][:ne]).all()
            continue
        assert not (moved == pools["state"]["normalBarotropicVelocity@2"][:ne]).any()
        transport = out["diagnostics/normalTransportVelocity"]
        n_h = int(pools["mesh"]["nEdgesArray"][2])
        assert (transport[n_h:] == pools["diagnostics"]["normalTransportVelocity"][n_h:]).all()
        if name == "nested_halos":
            assert n_h < ne
        if name == "no_predictor_filtered":
            assert (out["diagnostics/barotropicForcing"][:ne] == 0).all()
            # without the correction the transport is the masked velocity sum
            bcl = pools["state"]["normalBaroclinicVelocity@2"]
            bolus = pools["diagnostics"]["normalGMBolusVelocity"]
            mask = pools["mesh"]["edgeMask"].astype(bcl.dtype)
            want = mask[:ne] * ((moved[:, None] + bcl[:ne]) + bolus[:ne])
            assert (transport[:ne] == want).all()


@pytest.mark.parametrize("overrides, knob", [
    ({"config_btr_solve_SSH2": True}, "config_btr_solve_SSH2"),
    ({"config_n_btr_cor_iter": 3}, "config_n_btr_cor_iter"),
    ({"config_btr_solve_SSH2": True, "config_n_btr_cor_iter": 0, "nBtrSubcycles": 2}, "config_btr_solve_SSH2"),
])
def test_out_of_range_halo_counters_are_refused_by_name(overrides, knob):
    from woof.hex.mpas_ocean import btr_subcycle_v842 as bt
    config = dict(oo.BASE_CONFIG, **overrides)
    with pytest.raises(ConfigurationRefusal, match=knob):
        bt.check_config(config, config["nBtrSubcycles"])
    for case in oo.cases():
        bt.check_config(case["config"], case["config"]["nBtrSubcycles"])


@requires_gpu
def test_an_uncopied_corrector_neighbour_is_refused():
    from woof.verify import mpas_ocean_stream as stream
    from woof.hex.mpas_ocean import btr_subcycle_v842 as bt
    case = next(c for c in oo.cases() if c["name"] == "nested_halos")
    eoe = case["pools"]["mesh"]["edgesOnEdge"]
    eoe[0, 0] = case["dims"]["nEdges"]          # an inner edge reading the outermost layer
    dev = stream.on_device(oo.FIELDS, case)
    with pytest.raises(ConfigurationRefusal, match="nEdgesArray"):
        bt._check_copies(dev["mesh"], bt.halo_schedule(case["config"], case["config"]["nBtrSubcycles"]))


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
@pytest.mark.parametrize("original, mutated, case_name, field", [
    # the Coriolis term's product regrouped
    ("A(coriolis, M(M(weightsOnEdge[e * me2 + i], uPerp[eoe]), fEdge[eoe]))",
     "A(coriolis, M(weightsOnEdge[e * me2 + i], M(uPerp[eoe], fEdge[eoe])))", "defaults",
     "state/normalBarotropicVelocity@2"),
    # the SSH corrector's cells walked backwards
    ("for (int c = 0; c < nCells; ++c)", "for (int c = nCells - 1; c >= 0; --c)", "ssh_corrector",
     "state/sshSubcycle"),
    # the edge depth the deeper cell's
    ("bt_min(bottomDepth[cell1], bottomDepth[cell2])", "-bt_min(-bottomDepth[cell1], -bottomDepth[cell2])",
     "defaults", "diagnostics/barotropicThicknessFlux"),
    # the velocity sum normalised by the subcycle count, not count + 1
    ("btrNew[e] = D(btrNew[e], (rk)(nSum + 1));", "btrNew[e] = D(btrNew[e], (rk)nSum);", "defaults",
     "state/normalBarotropicVelocity@2"),
    # the correction split over the thickness
    ("D(S(thicknessFlux[e], fluxSum), thicknessSum)",
     "S(D(thicknessFlux[e], thicknessSum), D(fluxSum, thicknessSum))", "nested_halos",
     "diagnostics/normalTransportVelocity"),
    # the SSH step divided by the area first (the sign times the flux is
    # exact, so regrouping the tendency's term would not show)
    ("D(M(dtn, sshTend[c]), areaCell[c])", "M(dtn, D(sshTend[c], areaCell[c]))", "no_predictor_filtered",
     "state/sshSubcycle"),
])
def test_a_mutated_port_is_caught(monkeypatch, precision, original, mutated, case_name, field):
    from woof.hex.mpas_ocean import btr_subcycle_v842
    real = btr_subcycle_v842.module_source
    assert real(precision).count(original) == 1
    monkeypatch.setattr(btr_subcycle_v842, "module_source", lambda p="r4": real(p).replace(original, mutated))
    btr_subcycle_v842._module_on.cache_clear()
    try:
        case = next(c for c in oo.cases(precision) if c["name"] == case_name)
        got = oo.run_gpu(case)
        assert oo.word_mismatches(got[field], oo.load_fixture(precision)[case_name][field]) > 0
    finally:
        btr_subcycle_v842._module_on.cache_clear()


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
                    str(tmp_path / "build"), "--piece", "btr_subcycle", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
