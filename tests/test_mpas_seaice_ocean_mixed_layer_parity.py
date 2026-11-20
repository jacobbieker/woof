"""MPAS-Seaice's stand-alone ocean mixed layer, word for word against v8.4.2.

The reference is MPAS-Model v8.4.2's ``seaice_column_ocean_mixed_layer``
(``mpas_seaice_column.F``) with ``colpkg_atm_boundary`` for the open ocean
(``ice_atmo.F90``'s ``atmo_boundary_layer``, sfctype ``'ocn'``) and
``colpkg_ocn_mixed_layer``, compiled with gfortran -O0 -ffp-contract=off at
RKIND 8 on an x86-64 glibc 2.44 host. ``tools/mpas_seaice_oracle/run_column.F90``
runs it alone (stage ``ocean_mixed_layer``) and inside the column step
(stage ``column_step``, where ``seaice_column_coupling_prep`` runs it first).
The committed outputs (tests/data/oracles/mpas_seaice_ocean_mixed_layer,
PROVENANCE.md) let the GPU comparison run without gfortran. The stability
functions call glibc's binary64 atan, so the GPU comparison runs under the
opt-in transcription (``WOOF_GLIBC239_LIBM64=1``, source checkouts only).

THE BREAKAGES THESE PREVENT
---------------------------
1. A mixed layer that agrees with MPAS only to a tolerance: every word of
   every field it may write must be bit-identical, alone and over the column
   step's steps.
2. A branch lost: the clamp on the freezing/melting potential (both signs),
   the deep heat flux zeroed at freezing, the SST reset to freezing, the test
   ice shelf's zeroing, the stresses only under ``config_calc_surface_stresses``.
3. The mixed layer skipped in the coupling preparation, or a config not read.
4. A configuration the port does not carry running silently: each is refused
   by name.
5. A case generator whose bytes drift: the input digests are pinned.
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
from woof.hex.mpas_seaice import column_radiation_v842 as crad
from woof.hex.mpas_seaice import column_therm_v842 as ct
from woof.hex.mpas_seaice import coupling_prep_v842 as cprep
from woof.hex.mpas_seaice import ocean_mixed_layer_v842 as oml
from woof.verify import mpas_seaice_column_oracle as co

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = co.OML_FIXTURE_DIR
ROWS = co.OML_CASES + co.OML_STEP_CASES


def _case(name):
    return co.make_case(*next(row for row in ROWS if row[0] == name))


@pytest.fixture
def glibc239(monkeypatch):
    """The opt-in glibc 2.39 binary64 transcriptions (LGPL, source checkout only)."""
    from woof.core import glibc239_gpu
    if not glibc239_gpu.header_path().is_file():
        pytest.skip("the opt-in glibc 2.39 transcription is only in a source checkout")
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")
    ct._module_on.cache_clear()
    crad._module_on.cache_clear()
    yield
    ct._module_on.cache_clear()
    crad._module_on.cache_clear()


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (FIXTURE / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", ROWS)}


def test_fixture_covers_the_branches():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", ROWS)}
    assert set(logs) == set(cases)
    n = co.MESH["nCellsSolve"]
    frz = np.concatenate([fixture[name]["ocean_coupling/freezingMeltingPotential"][:n]
                          for name in (r[0] for r in co.OML_CASES)])
    # the clamp, both signs, and potentials inside it
    assert (frz == 1000.0).any() and (frz == -1000.0).any() and ((abs(frz) < 1000.0) & (frz != 0)).any()
    for name in (r[0] for r in co.OML_CASES):
        out, start = fixture[name], cases[name]["pools"]["ocean_coupling"]
        tf = start["seaFreezingTemperature"][:n]
        sst = out["ocean_coupling/seaSurfaceTemperature"][:n]
        # the SST reset to freezing where the mixed layer is at or below it, and warm cells kept
        assert (sst == tf).any() and (sst > tf).any(), name
        assert (sst >= tf).all(), name
        # the deep heat flux zeroed where it would cool a mixed layer at freezing
        qdp = out.get("ocean_coupling/oceanHeatFluxConvergence", start["oceanHeatFluxConvergence"])[:n]
        assert ((qdp == 0) & (start["oceanHeatFluxConvergence"][:n] > 0)).any(), name
        assert (out["ocean_atmosphere/albedoVisibleDirectOcean"][:n] == 0.06).all(), name
    # the test ice shelf zeroes the potential under land ice only
    shelf = fixture["oml_test_ice_shelf_ten_iterations"]["ocean_coupling/freezingMeltingPotential"][:n]
    land = cases["oml_test_ice_shelf_ten_iterations"]["pools"]["ocean_coupling"]["landIceMask"][:n]
    assert (shelf[land == 1] == 0).all() and (shelf[land == 0] != 0).all()
    # without the surface stresses the ocean stresses and the drag ratio keep their words
    assert "ocean_atmosphere/airStressOceanU" not in fixture["oml_no_surface_stresses"]
    assert "drag/airOceanDragCoefficientRatio" not in fixture["oml_no_surface_stresses"]
    assert "ocean_atmosphere/airStressOceanU" in fixture["oml_registry_defaults"]
    # in the column step the coupling preparation runs it each step
    for name in (r[0] for r in co.OML_STEP_CASES):
        assert co.oracle_step_abort(logs[name]) == (0, None, None)
        assert "ocean_coupling/seaSurfaceTemperature" in fixture[name], name


def test_unported_configurations_are_refused_by_name(monkeypatch):
    config = _case("oml_registry_defaults")["config"]
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")
    with pytest.raises(ConfigurationRefusal, match="config_atmos_boundary_method"):
        cprep.check_config({**config, "config_atmos_boundary_method": "constant"})
    monkeypatch.delenv("WOOF_GLIBC239_LIBM64")
    with pytest.raises(ConfigurationRefusal, match="config_use_ocean_mixed_layer"):
        cprep.check_config(config)
    cprep.check_config({**config, "config_use_ocean_mixed_layer": False})


def _host_pools(name="oml_test_ice_shelf_ten_iterations"):
    case = _case(name)
    return case["pools"], {**case["config"], "config_dt": case["dt"]}


@pytest.mark.parametrize("pool, field", [(pool, names[i]) for pool, names in oml._FIELDS
                                         for i in (0, len(names) - 1)])
def test_the_mixed_layer_at_rkind_4_is_refused(pool, field):
    # every field the kernel reads or writes is checked, before the glibc opt-in's refusal
    pools, configs = _host_pools()
    pools[pool][field] = pools[pool][field].astype(np.float32)
    with pytest.raises(ConfigurationRefusal, match=f"{pool}/{field} is float32.*RKIND 8"):
        oml.ocean_mixed_layer(pools, configs, 3)


def test_a_non_integer_land_ice_mask_is_refused():
    pools, configs = _host_pools()
    pools["ocean_coupling"]["landIceMask"] = pools["ocean_coupling"]["landIceMask"].astype(np.float64)
    with pytest.raises(ConfigurationRefusal, match="landIceMask is not int32"):
        oml.ocean_mixed_layer(pools, configs, 3)


@pytest.mark.parametrize("knob", ["config_dt", "config_nIceLayers"])
def test_a_direct_call_without_the_step_configs_is_refused(knob):
    pools, configs = _host_pools()
    with pytest.raises(ConfigurationRefusal, match=knob):
        oml.ocean_mixed_layer(pools, {k: v for k, v in configs.items() if k != knob}, 3)


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word(glibc239):
    report = co.measure(rows=co.OML_CASES, fixture_dir=FIXTURE)
    report.update(co.measure_step(rows=co.OML_STEP_CASES, fixture_dir=FIXTURE))
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


def _config_flipped(monkeypatch, knob):
    real = oml._config
    monkeypatch.setattr(oml, "_config", lambda configs, name: (not real(configs, name)) if name == knob
                        else real(configs, name))


def _no_test_ice_shelf(monkeypatch):
    _config_flipped(monkeypatch, "config_use_test_ice_shelf")


def _stresses_always(monkeypatch):
    _config_flipped(monkeypatch, "config_calc_surface_stresses")


def _one_iteration(monkeypatch):
    real = oml._config
    monkeypatch.setattr(oml, "_config", lambda configs, name: 1 if name == "config_boundary_layer_iteration_number"
                        else real(configs, name))


def _no_mixed_layer_in_the_step(monkeypatch):
    monkeypatch.setattr(oml, "ocean_mixed_layer", lambda pools, configs, n: None)


def _wrong_reference_height(monkeypatch):
    real = ct.folded_constants()
    monkeypatch.setattr(ct, "folded_constants", lambda: {**real, "al2": float(np.log(10.0 / 1.5))})


@requires_gpu
@pytest.mark.parametrize("mutate, field, case", [
    # the land-ice mask ignored under config_use_test_ice_shelf
    (_no_test_ice_shelf, "ocean_coupling/freezingMeltingPotential", "oml_test_ice_shelf_ten_iterations"),
    # the stresses computed without config_calc_surface_stresses
    (_stresses_always, "ocean_atmosphere/airStressOceanU", "oml_no_surface_stresses"),
    # config_boundary_layer_iteration_number not read
    (_one_iteration, "ocean_atmosphere/atmosReferenceTemperature2mOcean", "oml_registry_defaults"),
    # the reference temperature's height not 2 m
    (_wrong_reference_height, "ocean_atmosphere/atmosReferenceTemperature2mOcean", "oml_registry_defaults"),
    # seaice_column_coupling_prep not running the mixed layer
    (_no_mixed_layer_in_the_step, "ocean_coupling/seaSurfaceTemperature", "oml_step_bl99_dedd"),
])
def test_a_mutated_mixed_layer_is_caught(monkeypatch, glibc239, mutate, field, case):
    mutate(monkeypatch)
    run = _case(case)
    got, _ = co.run_gpu(run)
    want = co.expected_outputs(run, co.load_fixture(FIXTURE)[case])[field]
    assert co.word_mismatches(got[field], want) > 0


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_seaice").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None or platform.machine() != "x86_64"
                    or platform.libc_ver() != ("glibc", "2.44"),
                    reason="the fixture is the x86-64 glibc 2.44 host's (its atan, exp, log, pow, sin, cos and "
                           "asin, PROVENANCE.md); it needs gfortran (pixi -e mpas-oracle) and WOOF_MPAS_V842_SOURCE")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_column_fixture.py"),
                    str(tmp_path / "build"), "--cases", "oml", "--out", str(tmp_path / "fixture")], check=True)
    fresh = np.load(tmp_path / "fixture" / "oracle-r8.npz")
    pinned = np.load(FIXTURE / "oracle-r8.npz")
    assert sorted(fresh.files) == sorted(pinned.files)
    for key in pinned.files:
        assert co.word_mismatches(fresh[key], pinned[key]) == 0, key
    for name in ("column-logs.json", "inputs.sha256"):
        assert (tmp_path / "fixture" / name).read_text() == (FIXTURE / name).read_text()
