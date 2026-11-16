"""WOOF's RRTMG-K (ra_lw_physics = ra_sw_physics = 14) against WRF 4.7.1.

tools/rrtmg_k_column_oracle runs the UNMODIFIED WRF 4.7.1 rad_rrtmg_driver
(phys/module_ra_rrtmg_lwk.F, module_ra_rrtmg_swk.F,
module_ra_effective_radius.F, built with -DBUILD_RRTMK=1) the way
module_radiation_driver.F drives it on one radiation step -- phy_prep's
t8w, radconst + calc_coszen, cal_cldfra1, the CAM ozone chain or the
wrapper's O3DATA, then RTHRATEN, RTHRATENSW = RTHRATEN - RTHRATENLW, GLW,
OLR, GSW, SWDOWN and the sixteen boundary fluxes -- compiled with gfortran
-O0 and no contraction against glibc, on the legacy oracle's 128 columns
plus a cloud droplet number, under six regimes: Thompson (CAM ozone,
wrapper ozone, p_top 2 kPa), aerosol-aware Thompson with QNC (F_QNC),
WSM6 and Morrison.  The fixture in tests/data/oracles/rrtmg_k/column holds
the inputs and WRF's outputs.

Two arms, both graded bit for bit under GPUWM_WRF_EXACT=1:

``engine``  woof.core.rrtmgk.rrtmgk_columns fed WRF's own driver inputs
            (t8w, cldfra, o3rad, coszen, solcon): the wrapper (effectRad,
            cloud paths, the extra top layer, ozone), McICA and both K
            engines, plus the radii effectRad leaves;
``adapter`` RRTMGKRadiation.__call__ as the forecast calls it, with its own
            t8w, cal_cldfra1, sun and CAM ozone.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import HAS_GPU

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "data" / "oracles" / "rrtmg_k" / "column"
TOOLS = REPO / "tools" / "rrtmg_k_column_oracle"
F = np.float32


def _tool(name):
    key = f"_rrtmg_k_column_oracle_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, TOOLS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def _meta():
    return json.loads((FIXTURE / "regimes.json").read_text())


REGIMES = sorted(_meta()["regimes"])
STRICT = __import__("woof.wrf_exact", fromlist=["ENABLED"]).ENABLED


def gpu_gate(fn):
    return pytest.mark.gpu(
        pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU / cupy")(fn))


def _ordered(a):
    i = np.asarray(a, F).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def bit_report(wrf, woof):
    """(differing values, max ULP) on the float32 bit patterns."""
    wrf = np.ascontiguousarray(wrf, F)
    woof = np.ascontiguousarray(woof, F)
    assert wrf.shape == woof.shape, (wrf.shape, woof.shape)
    differ = int((wrf.view(np.uint32) != woof.view(np.uint32)).sum())
    return differ, int(np.abs(_ordered(wrf) - _ordered(woof)).max()) if differ else 0


@pytest.fixture(scope="module")
def wrf():
    return dict(np.load(FIXTURE / "wrf471.npz"))


def _record(wrf, regime):
    prefix = regime + "/"
    return {k[len(prefix):]: v for k, v in wrf.items() if k.startswith(prefix)}


def test_fixture_covers_every_regime_and_field(wrf):
    meta = _meta()
    assert len(meta["regimes"]) == 6 and meta["ncol"] == 128
    pairs = _tool("compare").PAIRS
    for name, regime in meta["regimes"].items():
        for field in set(pairs["engine"]) | set(pairs["adapter"]):
            if field in ("o3rad",) and regime["o3input"] != 2:
                continue
            assert f"{name}/{field}" in wrf, (name, field)
        coszen = wrf[f"{name}/coszen"]
        assert (coszen > 0).sum() >= 60 and (coszen <= 0).sum() >= 30
    assert any(r["f_qnc"] for r in meta["regimes"].values())


def test_the_bit_comparison_can_fail(wrf):
    a = wrf["thompson8/rthratenlw"]
    b = a.copy()
    b.view(np.uint32)[3, 7] += 1
    assert bit_report(a, b) == (1, 1)
    assert bit_report(a, a) == (0, 0)


def test_host_constants_are_wrfs(wrf):
    """rgmma(5/3) and K's liquid cloud tables, against the compiled K."""
    from woof.core import rrtmgk
    assert bit_report(np.ravel(wrf["thompson8/cdm2"])[:1],
                      np.array([rrtmgk.cdm2()], F)) == (0, 0)
    for name in ("extliq1", "ssaliq1", "asyliq1"):
        want = np.ravel(wrf[f"thompson8/{name}"])
        got = rrtmgk._k_liquid(name).ravel(order="F")
        assert bit_report(want, got) == (0, 0), name


@gpu_gate
@pytest.mark.parametrize("arm", ["engine", "adapter"])
@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.skipif(not STRICT, reason="WRF word-for-word oracle of the strict "
                    "(GPUWM_WRF_EXACT=1) path")
def test_matches_wrf471_bit_for_bit(arm, regime, wrf):
    import cupy  # noqa: F401
    meta = _meta()
    run = _tool("run_woof")
    fn = run.run_engine if arm == "engine" else run.run_adapter
    woof = fn(FIXTURE, _record(wrf, regime), meta, regime)
    misses = {}
    for wk, ok in _tool("compare").PAIRS[arm].items():
        key = f"{regime}/{wk}"
        if wk == "o3rad" and meta["regimes"][regime]["o3input"] != 2:
            continue
        assert key in wrf, key
        assert ok in woof, (regime, ok)
        differ, max_ulp = bit_report(wrf[key], woof[ok])
        if differ:
            misses[wk] = (differ, max_ulp)
    assert not misses, f"{arm} {regime}: (differing values, max ULP) {misses}"


@gpu_gate
@pytest.mark.parametrize("regime", ["thompson8", "thompson28"])
def test_default_numerics_run_the_same_scheme(regime, wrf):
    """The default (fast) process runs the same K code; its heating, GLW,
    OLR and GSW stay close to WRF's.  The engine is the same words in both
    modes (CUDA, 2026-11-16: 0 of 284160 differ); what moves is the
    default-mode sun (coszen up to 64 ULP near the terminator), so the SW
    heating of a few grazing columns moves by up to ~1.1e-8 K s-1."""
    import cupy  # noqa: F401
    meta = _meta()
    woof = _tool("run_woof").run_adapter(FIXTURE, _record(wrf, regime), meta, regime)
    for field in ("glw", "olr", "gsw", "swdown"):
        np.testing.assert_allclose(woof[field], wrf[f"{regime}/{field}"],
                                   rtol=1e-4, atol=1e-3, err_msg=field)
    for field in ("rthratenlw", "rthratensw"):
        np.testing.assert_allclose(woof[field], wrf[f"{regime}/{field}"],
                                   rtol=1e-3, atol=5e-8, err_msg=field)
