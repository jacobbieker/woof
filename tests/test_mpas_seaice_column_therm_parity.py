"""MPAS-Seaice's column vertical thermodynamics (colpkg stage 2), word for word against v8.4.2.

The reference is MPAS-Model v8.4.2's ``column_vertical_thermodynamics``
(``mpas_seaice_column.F``) and the column package under it
(``colpkg_step_therm1``: ``frzmlt_bottom_lateral``, ``atmo_boundary_layer``,
the BL99 ``temperature_changes``, ``thickness_changes``, ``merge_fluxes``),
compiled with gfortran -O0 -ffp-contract=off at RKIND 8 on an x86-64 glibc
2.44 host. ``tools/mpas_seaice_oracle/run_column.F90`` (stage
``vertical_thermodynamics``) runs ``colpkg_init_thermo`` (the column
package's ``heat_capacity`` and ``l_brine``) and then the routine at the
clock's start; its ierr and abort message are logged. The oracle's outputs
are committed (tests/data/oracles/mpas_seaice_column_therm, PROVENANCE.md),
so the GPU comparison runs without gfortran.

``atmo_boundary_layer``'s ``psimhu`` calls glibc's binary64 ``atan`` (LGPL,
not correctly rounded), so the GPU comparison runs under the opt-in
transcription (``WOOF_GLIBC239_LIBM64=1``, source checkouts only); the
default build refuses by name.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the routine may write must be bit-identical.
2. v8.4.2's own quirks "fixed": colpkg_step_therm1 hands set_sfcflux the
   coupled surface and conductive fluxes swapped, colpkg_atm_boundary
   assigns ``worku = uvel`` twice and never sets ``workv``, and the ice and
   snow layers' shortwave absorption uses frac = 0.9 (no CCSMCOUPLED).
3. An abort that the port reports differently: the first aborting owned
   cell, its ``indexToCellID`` and its message must be MPAS's.
4. A gate that cannot fail: a mutated evaluation order must produce
   mismatches.
5. A configuration the port does not carry running silently: each is
   refused by name, glibc's atan included.
6. A case generator whose bytes drift: the input digests are pinned.
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
from woof.hex.mpas_seaice import column_therm_v842 as ct
from woof.hex.mpas_seaice import column_v842 as cv
from woof.verify import mpas_seaice_column_oracle as co

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = co.VT_FIXTURE_DIR


def _case(name):
    return co.make_case(*next(row for row in co.VT_CASES if row[0] == name))


@pytest.fixture
def glibc239(monkeypatch):
    """The opt-in glibc 2.39 binary64 atan transcription (LGPL, source checkout only)."""
    from woof.core import glibc239_gpu
    if not glibc239_gpu.header_path().is_file():
        pytest.skip("the opt-in glibc 2.39 transcription is only in a source checkout")
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")
    ct._module_on.cache_clear()
    yield
    ct._module_on.cache_clear()


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (FIXTURE / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", co.VT_CASES)}


def test_fixture_covers_the_branches():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", co.VT_CASES)}
    assert set(logs) == set(cases)
    outcomes = {name: co.oracle_abort(log) for name, log in logs.items()}
    # one case aborts in its sixth cell (indexToCellID 6), before a later aborting cell; the rest run through
    assert outcomes.pop("vt_bl99_abort_salinity") == (
        ct.SEAICE_ERROR_COL_VERT_THERM, 6,
        "column_vertical_thermodynamics: ice: Vertical thermo error: " + ct.STOP_LABELS[3])
    assert set(outcomes.values()) == {(0, None, None)}
    for name, out in fixture.items():
        for key, value in out.items():
            assert np.isfinite(value).all(), (name, key)
    n_solve = co.MESH["nCellsSolve"]

    def got(name, key):
        return fixture[name][key][:n_solve]

    def changed(name, key):
        pool, field = key.split("/")
        return got(name, key) != cases[name]["pools"][pool][field][:n_solve]

    # surface melt, a surface at the melting point, basal growth and snow-ice formation
    assert (got("vt_bl99_melting", "melt_growth_rates/surfaceIceMelt") > 0).any()
    assert (got("vt_bl99_melting", "tracers/surfaceTemperature") == 0.0).any()
    assert (got("vt_bl99_bubbly", "melt_growth_rates/congelationCategory") > 0).any()
    assert (got("vt_bl99_bubbly", "melt_growth_rates/snowiceFormationCategory") > 0).any()
    assert changed("vt_bl99_bubbly", "melt_growth_rates/snowiceFormation").any()
    assert (got("vt_bl99_bubbly", "ocean_fluxes/oceanHeatFluxIceBottom") < 0).any()
    # the first-year ice area resets on day 259 in the north and day 75 in the south only
    for name, north in (("vt_bl99_first_year_north", True), ("vt_bl99_first_year_south", False)):
        lat = cases[name]["pools"]["mesh"]["latCell"][:n_solve]
        fy = got(name, "tracers/firstYearIceArea")[:, :, 0]
        area = cases[name]["pools"]["tracers"]["iceAreaCategory"][:n_solve, :, 0]
        hemisphere = np.broadcast_to((lat > 0 if north else lat <= 0)[:, None], fy.shape)
        assert (fy[hemisphere & (area > 1e-11)] == 0).all() and (fy[~hemisphere] != 0).any()
    assert "tracers/firstYearIceArea" not in fixture["vt_bl99_bubbly"]
    # the coupled fluxes reach the category fluxes swapped (v8.4.2's set_sfcflux call)
    alt = cases["vt_bl99_coupled_surface_temperature"]["pools"]["alternative_atmos_forcing"]
    surface = got("vt_bl99_coupled_surface_temperature", "atmos_fluxes/surfaceHeatFluxCategory")
    iced = cases["vt_bl99_coupled_surface_temperature"]["pools"]["tracers"]["iceAreaCategory"][:n_solve, :, 0]
    assert (surface[iced > 1e-11] == alt["surfaceConductiveFluxCouple"][:n_solve][iced > 1e-11]).all()


def test_the_day_of_year_is_the_clock_s():
    assert ct.day_of_year("2000-01-01_00:00:00") == 1.0
    assert ct.day_of_year("2000-09-15_00:00:00") == 259.0
    assert ct.day_of_year("2001-03-16_00:00:00") == 75.0


@pytest.mark.parametrize("overrides, match", [
    ({"config_thermodynamics_type": "mushy"}, "mushy"),
    ({"config_thermodynamics_type": "zero layer"}, "zero-layer"),
    ({"config_use_level_meltponds": True}, "melt ponds"),
    ({"config_use_aerosols": True}, "aerosols"),
    ({"config_atmos_boundary_method": "constant"}, "Cdn_atm_ratio_n"),
    ({"config_calc_surface_stresses": False}, "Cdn_atm_ratio_n"),
])
def test_unported_configurations_are_refused_by_name(monkeypatch, overrides, match):
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")
    configs = {**_case("vt_bl99_bubbly")["config"], **overrides}
    dims = _case("vt_bl99_bubbly")["dims"]
    with pytest.raises(ConfigurationRefusal, match=match):
        ct.refuse_unported(configs, {**dims, "nSnowLayers": 1})


def test_the_default_build_refuses_glibc_s_atan(monkeypatch):
    monkeypatch.delenv("WOOF_GLIBC239_LIBM64", raising=False)
    case = _case("vt_bl99_bubbly")
    with pytest.raises(ConfigurationRefusal, match="atan.*WOOF_GLIBC239_LIBM64=1"):
        ct.refuse_unported(case["config"], case["dims"])


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word(glibc239):
    report = co.measure_vt()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("original, mutated, field, case", [
    # freeboard's snow-ice thickness through the density ratio (dhsn*rhos/rhoi)
    ("dhin = D(M(dhsn, RHOS), RHOI);", "dhin = M(dhsn, D(RHOS, RHOI));",
     "melt_growth_rates/snowiceFormationCategory", "vt_bl99_bubbly"),
    # set_sfcflux's swapped arguments "fixed"
    ("surfaceHeatFluxCategory[cn] = M(surfaceConductiveFluxCouple[cn], 1.0);",
     "surfaceHeatFluxCategory[cn] = M(surfaceHeatFluxCouple[cn], 1.0);",
     "atmos_fluxes/surfaceHeatFlux", "vt_bl99_coupled_surface_temperature"),
    # colpkg_atm_boundary's vvel = 0 "fixed" (high-frequency coupling)
    ("&Cdn_atm_ratio_n, uVelocity[c], 0.0,", "&Cdn_atm_ratio_n, uVelocity[c], uVelocity[c],",
     "velocity_solver/airStressCellV", "vt_bl99_mu71_highfreq"),
    # TsfK**4 as three multiplications instead of __powidf2's two squarings
    ("__device__ __forceinline__ double powi4(double x) { double t = M(x, x); return M(t, t); }",
     "__device__ __forceinline__ double powi4(double x) { return M(M(M(x, x), x), x); }",
     "tracers/surfaceTemperature", "vt_bl99_bubbly"),
    # the shortwave absorption fraction of a CCSMCOUPLED build
    ("double frac = 0.9, dTemp = 0.02;", "double frac = 1.0, dTemp = 0.01;",
     "shortwave/surfaceShortwaveFlux", "vt_bl99_melting"),
    # the tridiagonal elimination's division distributed over the difference
    ("xout[k - 1] = D(S(rhs[k - 1], M(sbdiag[k - 1], xout[k - 2])), wbeta);",
     "xout[k - 1] = S(D(rhs[k - 1], wbeta), D(M(sbdiag[k - 1], xout[k - 2]), wbeta));",
     "atmos_fluxes/surfaceConductiveFluxCategory", "vt_bl99_bubbly"),
])
def test_a_mutated_evaluation_order_is_caught(monkeypatch, glibc239, original, mutated, field, case):
    source = ct.module_source
    probe = source(7)
    assert probe.count(original) == 1
    monkeypatch.setattr(ct, "module_source", lambda n: source(n).replace(original, mutated))
    ct._module_on.cache_clear()
    try:
        got, _ = co.run_gpu(_case(case))
        want = co.load_fixture(FIXTURE)[case][field]
        assert co.word_mismatches(got[field], want) > 0
    finally:
        ct._module_on.cache_clear()


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_seaice").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None or platform.machine() != "x86_64"
                    or platform.libc_ver()[0] != "glibc",
                    reason="the column oracle's words are the x86-64 glibc host's libm (atan, exp, log, pow); "
                           "it needs gfortran (pixi -e mpas-oracle) and WOOF_MPAS_V842_SOURCE")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_column_fixture.py"),
                    str(tmp_path / "build"), "--cases", "therm", "--out", str(tmp_path / "fixture")], check=True)
    fresh = np.load(tmp_path / "fixture" / "oracle-r8.npz")
    pinned = np.load(FIXTURE / "oracle-r8.npz")
    assert sorted(fresh.files) == sorted(pinned.files)
    for key in pinned.files:
        assert co.word_mismatches(fresh[key], pinned[key]) == 0, key
    for name in ("column-logs.json", "inputs.sha256"):
        assert (tmp_path / "fixture" / name).read_text() == (FIXTURE / name).read_text()
