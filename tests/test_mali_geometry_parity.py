"""MALI's masks, geometry and layer diagnostics, word for word against v8.4.2.

The reference is the calls the model makes before thickness advection:
``li_calculate_mask``, ``li_update_geometry``, ``li_calculate_layerThickness``
and ``li_layer_normal_velocity``. They come from MPAS-Model v8.4.2 compiled
verbatim by ``tools/mali_oracle`` (``run_geometry.F90``) with gfortran -O0
-ffp-contract=off at RKIND 4. The outputs are committed
(tests/data/oracles/mali_geometry, PROVENANCE.md).

THE BREAKAGES THESE PREVENT
---------------------------
1. A mask bit set in the wrong pass order, or a geometry word off by a rounding.
2. A case generator whose bytes drift. The input digests are pinned.
3. A gate that cannot fail. A mutated pass must produce mismatches.
4. The higher-order solvers' Albany mask bits being skipped silently. They are
   refused by name.
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
from woof.verify import mali_geometry_oracle as go

ROOT = Path(__file__).resolve().parents[1]


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (go.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {**{case["name"]: go.input_digest(case) for case in go.cases()},
                        **{case["name"] + "/r8": go.input_digest(case) for case in go.cases("r8")}}


def test_fixture_covers_the_branches():
    fixture = go.load_fixture()
    for case in go.cases():
        out = fixture[case["name"]]
        assert out["err"].tolist() == [0, 0, 0]
        cm, t = out["cellMask"], case["thickness"]
        for bit in (1, 2, 4, 8, 16, 32, 256):
            assert ((cm & bit) > 0).any(), bit
        # the calving-front rule cleared the dynamic bit of thick floating margin cells
        assert (((cm & 8) > 0) & ((cm & 4) > 0) & (t > case["dynamicThickness"]) & ((cm & 2) == 0)).any()
        assert (case["normalVelocity"][:case["nEdgesSolve"], :2] == 0).all(axis=1).any()
    assert {c["velocitySolver"] for c in go.cases()} == {"sia", "none"}


def test_higher_order_mask_paths_are_refused_by_name():
    from woof.hex.mpas_ice import geometry_v842
    with pytest.raises(ConfigurationRefusal) as exc:
        geometry_v842.calculate_mask(mesh={}, geometry={}, config_ice_density=910.0, config_ocean_density=1028.0,
                                     config_sea_level=0.0, config_dynamic_thickness=10.0,
                                     config_velocity_solver="FO")
    assert "config_velocity_solver" in str(exc.value)


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = go.measure(precision=precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
@pytest.mark.parametrize("original, mutated, field", [
    # the floating lower surface without its parenthesised density ratio
    ("S(seaLevel, M(h, D(iceDensity, oceanDensity)))", "S(seaLevel, D(M(h, iceDensity), oceanDensity))",
     "lowerSurface"),
    # the calving-front rule not applied
    ("mask &= ~LI_DYNAMIC_ICE;", "mask &= ~0;", "cellMask"),
])
def test_a_mutated_operation_is_caught(monkeypatch, precision, original, mutated, field):
    from woof.hex.mpas_ice import geometry_v842
    sources = {p: geometry_v842.module_source(p) for p in ("r4", "r8")}
    assert sources[precision].count(original) == 1
    monkeypatch.setattr(geometry_v842, "module_source", lambda p="r4": sources[p].replace(original, mutated))
    geometry_v842._module_on.cache_clear()
    try:
        fixture = go.load_fixture(precision)
        total = sum(go.word_mismatches(go.run_gpu(c)[field], fixture[c["name"]][field])
                    for c in go.cases(precision))
        assert total > 0
    finally:
        geometry_v842._module_on.cache_clear()


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_landice").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MALI oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mali_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mali_oracle" / "make_geometry_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    for precision in ("r4", "r8"):
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(go.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert go.word_mismatches(fresh[key], pinned[key]) == 0, key
