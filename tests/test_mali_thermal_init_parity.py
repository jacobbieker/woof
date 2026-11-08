"""MALI's thermal initialisation and enthalpy conversions, word for word against v8.4.2.

The reference is ``li_thermal_init``, then
``li_compute_pressure_melting_point_fields``, ``li_temperature_to_enthalpy``
and ``li_enthalpy_to_temperature``. They come from MPAS-Model v8.4.2 compiled
verbatim by ``tools/mali_oracle`` (``run_thermal.F90``) with gfortran -O0
-ffp-contract=off at RKIND 4. The outputs are committed
(tests/data/oracles/mali_thermal_init, PROVENANCE.md).

THE BREAKAGES THESE PREVENT
---------------------------
1. A conversion or initial temperature off by a rounding, or a branch taken on
   the wrong side of a pressure-melting enthalpy.
2. A case generator whose bytes drift. The input digests are pinned.
3. A gate that cannot fail. A mutated operation must produce mismatches.
4. An unknown init option silently doing nothing. It is refused by name.
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
from woof.verify import mali_thermal_oracle as to

ROOT = Path(__file__).resolve().parents[1]


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (to.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {case["name"]: to.input_digest(case) for case in to.cases()}


def test_fixture_covers_the_branches():
    fixture = to.load_fixture()
    assert {c["temperatureInit"] for c in to.cases()} == set(to.INIT)
    assert {c["airSource"] for c in to.cases()} == set(to.AIR)
    for case in to.cases():
        out = fixture[case["name"]]
        assert out["err"].tolist() == [0, 0]
        assert ((out["convWaterfrac"] > 0).any() and (out["convWaterfrac"] == 0).any())
        assert (out["convSurfaceEnthalpy"] == 0).any() and (out["convSurfaceEnthalpy"] < 0).any()
        assert all(np.isfinite(v).all() for v in out.values())


def test_unknown_init_options_are_refused_by_name():
    from woof.hex.mpas_ice import thermal_init_v842 as th
    with pytest.raises(ConfigurationRefusal) as exc:
        th._option(th.TEMPERATURE_INIT, "config_temperature_init", "isothermal")
    assert "config_temperature_init" in str(exc.value)


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word():
    report = to.measure()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("original, mutated, field", [
    # the pressure-melting column product associated differently
    ("return -M(M(M(M(LI_ICE_MELTING_POINT_PRESSURE_DEPENDENCE, rhoi), LI_GRAVITY), thickness), sigma);",
     "return -M(M(M(LI_ICE_MELTING_POINT_PRESSURE_DEPENDENCE, rhoi), LI_GRAVITY), M(thickness, sigma));",
     "pmpTemperature"),
    # the cold-ice temperature divided in two steps
    ("temperature[p] = D(enthalpy[p], M(rhoi, LI_CP_ICE));",
     "temperature[p] = D(D(enthalpy[p], rhoi), LI_CP_ICE);", "convTemperature"),
])
def test_a_mutated_operation_is_caught(monkeypatch, original, mutated, field):
    from woof.hex.mpas_ice import thermal_init_v842 as th
    source = th.module_source()
    assert source.count(original) == 1
    monkeypatch.setattr(th, "module_source", lambda: source.replace(original, mutated))
    th._module_on.cache_clear()
    try:
        fixture = to.load_fixture()
        total = sum(to.bit_mismatches(to.run_gpu(c)[field], fixture[c["name"]][field]) for c in to.cases())
        assert total > 0
    finally:
        th._module_on.cache_clear()


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
    subprocess.run([sys.executable, str(ROOT / "tools" / "mali_oracle" / "make_thermal_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    fresh = np.load(tmp_path / "fixture" / "oracle-r4.npz")
    pinned = np.load(to.FIXTURE_DIR / "oracle-r4.npz")
    assert sorted(fresh.files) == sorted(pinned.files)
    for key in pinned.files:
        assert np.array_equal(fresh[key].view(np.uint32), pinned[key].view(np.uint32)), key
