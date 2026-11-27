"""Noah's snow-soil series conductivity (RunConfig.noah_snow_series_conductivity, FQ9).

WRF 4.7.1 phys/module_sf_noahlsm.F, SFLX, the "plane parallel" snowpack effect
on the first-guess soil heat flux (:644-664).  Under a snowpack of depth SNOWH
the heat leaves the skin through the snow and then the half first soil layer
(DSOIL = -0.5 ZSOIL(1)): two slabs in series, whose effective conductivity is
the thickness-weighted harmonic mean

    DF1H = SNCOND*DF1 / (FRCSOI*SNCOND + FRCSNO*DF1)         (:649)

WRF computes DF1H and never reads it.  The flux uses the thickness-weighted
ARITHMETIC mean DF1A = FRCSNO*SNCOND + FRCSOI*DF1 (:656), the conductivity of
slabs side by side (parallel flow), blended with the snow-free DF1 by SNCOVR
(:663).  The commented lines at :654-655 show the author meant the harmonic
DF1H in that blend.  Harmonic <= arithmetic, so WRF's mean overstates the
conductivity of a thin pack (snow conducts far less than soil) and the soil
heat flux out of the skin.

The correction (noah.cu and noah_mosaic.cu ``NOAH_SNOW_SERIES_CONDUCTIVITY``,
npref ``snow_series_conductivity``) blends DF1H.  Nothing else moves except
what the flux feeds (skin and soil temperatures, ground heat flux).

Graded here:

* exact mode is word for word: the kernels with the block removed are byte for
  byte the sources before the option; the float64 mirror and the device kernel
  are bitwise unchanged with the option off;
* on a column with a partial snow cover the exact word and the corrected one
  differ in the ground heat flux and the soil temperatures, the corrected flux
  being the smaller (the series conductivity is below the parallel one);
* a snow-free column and a fully covered one (SNCOVR > 0.97, DF1 = SNCOND, so
  DF1H = DF1A = SNCOND) are unchanged.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

from woof.config import ConfigurationRefused, RunConfig, validate_run_config

ROOT = Path(__file__).resolve().parents[1]
KDIR = ROOT / "woof" / "core" / "kernels"
#: sha256 of each kernel before the option (golden: FQ8's program)
BEFORE = {
    "noah.cu": "c812c93845340096214f805aa69fe999e2c8483d2b798671260c6151d59361d4",
    "noah_mosaic.cu": "6c208d40e19e5ceb1804eedb2d3c4deebee0cdcfb7b4a3e10a18991e6e9d60c9",
}
BLOCK = re.compile(r"#ifdef NOAH_SNOW_SERIES_CONDUCTIVITY\n.*?#endif\n", re.S)


def _run(**kw):
    return RunConfig(nx=8, ny=8, nz=12, dx=3000.0, dy=3000.0, ztop=12000.0, dt=10.0, run_seconds=60.0,
                     sf_sfclay_physics=1, **kw)


# --- the option -----------------------------------------------------------

def test_the_option_follows_the_mode_and_is_read_by_noah_only():
    from woof import correct_mode
    from woof.quirk_corrections import run_reads
    field = RunConfig.__dataclass_fields__["noah_snow_series_conductivity"]
    # correct-mode finding FQ9: on in a default process, off in exact mode
    assert field.default is correct_mode.CORRECT_KEYS_DEFAULT and field.kw_only
    assert correct_mode.resolve("exact").settings["noah_snow_series_conductivity"] is False
    assert correct_mode.KEY_SCOPE["noah_snow_series_conductivity"] == ("sf_surface_physics", (2,))
    assert run_reads(validate_run_config(_run(sf_surface_physics=2, noah_snow_series_conductivity=True)),
                     "noah_snow_series_conductivity")
    assert not run_reads(validate_run_config(_run(sf_surface_physics=1, num_soil_layers=5,
                                                  noah_snow_series_conductivity=True)),
                         "noah_snow_series_conductivity")
    with pytest.raises(ConfigurationRefused, match="noah_snow_series_conductivity"):
        validate_run_config(_run(sf_surface_physics=2, noah_snow_series_conductivity=1))


def test_the_umbrella_sets_it():
    assert _run(sf_surface_physics=2, woof_quirk_corrections="all").noah_snow_series_conductivity is True
    assert _run(sf_surface_physics=2, woof_quirk_corrections="none").noah_snow_series_conductivity is False
    from woof import quirk_corrections as qc
    with pytest.raises(ConfigurationRefused, match="noah_snow_series_conductivity"):
        qc.apply_run_corrections({qc.UMBRELLA: "all", "sf_surface_physics": 2,
                                  "noah_snow_series_conductivity": False})


def test_the_define_tier_composes_in_one_fixed_order():
    from woof.core.noah import noah_define_prefix, noah_define_tier
    assert noah_define_tier(False, False, True) == (("NOAH_SNOW_SERIES_CONDUCTIVITY", 1),)
    assert noah_define_tier(True, True, True) == (
        ("NOAH_SMC_FLOOR_BOOKING", 1), ("NOAH_LAND_ICE_RAIN_RUNOFF", 1),
        ("NOAH_SNOW_SERIES_CONDUCTIVITY", 1))
    assert noah_define_tier(True, True) == (("NOAH_SMC_FLOOR_BOOKING", 1), ("NOAH_LAND_ICE_RAIN_RUNOFF", 1))
    assert noah_define_tier() == () and noah_define_prefix() == ""
    assert noah_define_prefix(False, False, True) == "#define NOAH_SNOW_SERIES_CONDUCTIVITY 1\n"


# --- the default ----------------------------------------------------------

@pytest.mark.parametrize("name", sorted(BEFORE))
def test_the_default_kernel_source_is_the_source_before_the_option(name):
    """Exact mode is WRF's word: the block is a pure addition."""
    text = (KDIR / name).read_text(encoding="utf-8")
    blocks = BLOCK.findall(text)
    assert len(blocks) == 1, name
    assert "df1a" in blocks[0] and "sncond" in blocks[0]
    # the later correction blocks (none yet) would be stripped here too
    assert hashlib.sha256(BLOCK.sub("", text).encode()).hexdigest() == BEFORE[name]


# --- the mirror -------------------------------------------------------------

@pytest.fixture(scope="module")
def params():
    from woof.core.noah import load_tables, pack_params
    return pack_params(load_tables())


@pytest.fixture(autouse=True)
def _no_cpu_bridge(monkeypatch):
    """Frozen-soil SH2O = SMC (WRF's init without its CPU bridge): the columns
    are near freezing, where the init changes nothing the test grades."""
    import test_noah
    monkeypatch.setattr(test_noah, "sh2o_init", lambda smc, tslb, soil, p: np.array(smc, np.float64))


def _snow_col(params, *, swe, snowh, seed=3):
    """A cold bare-ish land column under a pack of ``swe`` mm and ``snowh`` m."""
    import test_noah
    col = test_noah._base_col(params, veg=10, soil=6, rng=np.random.default_rng(seed), snow=True,
                              sfctmp=268.0, tsk=268.0, swdown=300.0)
    col["snow"] = swe
    col["snowh"] = snowh
    col["snowc"] = min(1.0, swe / 40.0)
    col["tslb"] = np.array([271.0, 272.0, 273.0, 274.0])
    return col


def _run_mirror(params, col, **kw):
    from test_noah import DT, DZS
    from woof.verify.npref import np_noah_column
    return np_noah_column(dict(col), params, DT, DZS, **kw)


@pytest.mark.parametrize("swe,snowh", [(10.0, 0.05), (30.0, 0.12)])
def test_the_word_difference_on_a_snow_covered_column(params, swe, snowh):
    col = _snow_col(params, swe=swe, snowh=snowh)
    exact = _run_mirror(params, col)
    fixed = _run_mirror(params, col, snow_series_conductivity=True)
    assert 0.0 < exact["sncovr"] < 0.97              # the blend is exercised
    # the flux out of the skin and the soil temperatures move
    assert fixed["grdflx"] != exact["grdflx"]
    assert not np.array_equal(fixed["tslb"], exact["tslb"])
    assert fixed["tsk"] != exact["tsk"]
    # the physical claim: the series (harmonic) conductivity is below the
    # parallel (arithmetic) one, so the corrected soil heat flux is smaller in
    # magnitude and the skin, which loses less heat to the soil, is warmer
    assert abs(fixed["grdflx"]) < abs(exact["grdflx"])
    assert fixed["tsk"] > exact["tsk"]
    # the shipped DF1H: harmonic <= arithmetic for any positive weights
    sncond, df1, frcsno = 0.15, 1.2, 0.35
    frcsoi = 1.0 - frcsno
    assert (sncond * df1 / (frcsoi * sncond + frcsno * df1)
            < frcsno * sncond + frcsoi * df1)
    # the pack's water is not touched by the conductivity at this step's phase
    assert fixed["sncovr"] == exact["sncovr"]


def test_a_snow_free_column_is_unchanged(params):
    col = _snow_col(params, swe=0.0, snowh=0.0)
    exact = _run_mirror(params, col)
    fixed = _run_mirror(params, col, snow_series_conductivity=True)
    for key, value in exact.items():
        np.testing.assert_array_equal(np.asarray(fixed[key]), np.asarray(value), err_msg=key)


def test_a_fully_covered_column_is_unchanged_to_rounding(params):
    """SNCOVR > 0.97 sets DF1 = SNCOND: DF1H = DF1A = SNCOND in exact
    arithmetic (the two expressions round differently in float64)."""
    col = _snow_col(params, swe=200.0, snowh=0.6)
    exact = _run_mirror(params, col)
    fixed = _run_mirror(params, col, snow_series_conductivity=True)
    assert exact["sncovr"] > 0.97
    for key in ("grdflx", "tsk", "tslb", "snow", "snowh"):
        np.testing.assert_allclose(np.asarray(fixed[key]), np.asarray(exact[key]), rtol=1e-12, atol=1e-12,
                                   err_msg=key)


# --- the device -------------------------------------------------------------

def _launch(cols, params, **options):
    import cupy as cp
    from test_noah import DT, DZS, _grid_from_cols
    from woof.core.noah import launch_noah
    grids, n, ny, nx = _grid_from_cols(cols)
    dev = {k: cp.asarray(v.astype(np.int32 if k in ("ivgtyp", "isltyp") else np.float32))
           for k, v in grids.items()}
    for k in ("smstav", "smstot", "noahres", "reslin", "chklowq"):
        dev[k] = cp.zeros((ny, nx), dtype=cp.float32)
    dev["smcrel"] = cp.zeros((4, ny, nx), dtype=cp.float32)
    dev["ebal"] = cp.zeros((ny, nx), dtype=cp.int32)
    launch_noah(dev, params, DT, DZS, **options)
    return {k: cp.asnumpy(v) for k, v in dev.items()}, n, nx


@requires_gpu
@pytest.mark.gpu
def test_the_kernel_matches_the_mirror_and_leaves_snow_free_columns_alone(params):
    from test_noah import DT, DZS
    from woof.verify.npref import np_noah_column
    cols = [_snow_col(params, swe=10.0, snowh=0.05), _snow_col(params, swe=30.0, snowh=0.12),
            _snow_col(params, swe=0.0, snowh=0.0)]
    stock, n, nx = _launch(cols, params)
    fixed, _, _ = _launch(cols, params, snow_series_conductivity=True)
    for m, col in enumerate(cols):
        j, i = m // nx, m % nx
        c32 = {k: (np.asarray(v, np.float32).astype(np.float64) if k in ("smois", "tslb", "sh2o")
                   else int(v) if k in ("ivgtyp", "isltyp") else float(np.float32(v))) for k, v in col.items()}
        ref_stock = np_noah_column(dict(c32), params, DT, DZS)
        ref_fixed = np_noah_column(dict(c32), params, DT, DZS, snow_series_conductivity=True)
        if m < 2:
            assert fixed["grdflx"][j, i] != stock["grdflx"][j, i]
            assert abs(fixed["grdflx"][j, i]) < abs(stock["grdflx"][j, i])
            assert fixed["grdflx"][j, i] == pytest.approx(ref_fixed["grdflx"], rel=2e-3, abs=1e-3)
            assert stock["grdflx"][j, i] == pytest.approx(ref_stock["grdflx"], rel=2e-3, abs=1e-3)
            assert not np.array_equal(fixed["tslb"][:, j, i], stock["tslb"][:, j, i])
        else:
            for key in ("grdflx", "tsk", "tslb", "snow", "hfx", "lh"):
                np.testing.assert_array_equal(np.atleast_1d(fixed[key][..., j, i]).view(np.uint32),
                                              np.atleast_1d(stock[key][..., j, i]).view(np.uint32),
                                              err_msg=key)


@requires_gpu
@pytest.mark.gpu
def test_the_composed_units_compile():
    from woof.core import noah_mosaic
    from woof.core.kernels import get_kernel_int_defines
    from woof.core.noah import noah_define_module, noah_define_tier
    assert noah_mosaic._mosaic_kernel(False, False, False, True) is not None
    assert noah_mosaic._mosaic_kernel(False, True, True, True) is not None
    assert noah_mosaic._mosaic_kernel(True, True, True, True) is not None
    assert noah_mosaic._mosaic_ucm_module(False, True, True, True).get_function("noah_mosaic_ucm_column") is not None
    for module in ("noah", "noah_ua"):
        for tier in (noah_define_tier(False, False, True), noah_define_tier(True, True, True)):
            assert get_kernel_int_defines(module, "noah_column", tier) is not None
    assert noah_define_module("noah_ua", noah_define_tier(True, True, True)) is not None
