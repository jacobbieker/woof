"""Noah's land-ice rain runoff (RunConfig.noah_land_ice_rain_runoff, FQ8).

WRF 4.7.1 phys/module_sf_noahlsm_glacial_only.F, SFLX_GLACIAL.  Rain that
falls on a land-ice point (no snow, no freezing rain) is kept in ``PRCPF`` as
"precip that later can wholly or partially infiltrate the soil" (:286-288);
SNOPAC converts it to ``PRCP1 = PRCPF*0.001`` (:912), and nothing reads PRCP1
again.  There is no soil to infiltrate, and the rain is not added to SNEQV
(only snow and freezing rain are, :259-260).  The driver then books only
``RUNOFF1 = SNOMLT/DT`` (:404-406), "since there is no soil medium", so the
rain leaves the water budget without being booked.

The correction (noah.cu and noah_mosaic.cu ``NOAH_LAND_ICE_RAIN_RUNOFF``, npref
``land_ice_rain_runoff``) adds ``PRCPF*0.001`` (m s-1) to RUNOFF1: all of it,
because WRF's own logic refreezes none of it into the pack.

Graded here:

* the default is word for word: the kernels with the block removed are byte
  for byte the sources before the option; the float64 mirror and the device
  kernel are bitwise unchanged with the option off, and with it on wherever
  no liquid rain reaches the ice (snow, freezing rain, dry);
* on a rainy land-ice column the exact word loses the rain (the water budget
  d(snow) = dt (P - E) - R is out by the rain) and the corrected mirror
  and kernel close it;
* the option composes with FQ4 (noah_smc_floor_booking).
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
#: sha256 of each kernel before the option (golden: the default program)
BEFORE = {
    "noah.cu": "3736ba8f07a6ff915255159e6fec8b29a617e76465364d99d276c0795470e17d",
    "noah_mosaic.cu": "5c09214c29f2c9f6b01bec651281778b8d759facfc6c755cbb9cd02bd7237b00",
}
BLOCK = re.compile(r"#ifdef NOAH_LAND_ICE_RAIN_RUNOFF\n.*?#endif\n", re.S)
ICE = 15          # the USGS land-ice category the Noah tables name ISICE


def _run(**kw):
    return RunConfig(nx=8, ny=8, nz=12, dx=3000.0, dy=3000.0, ztop=12000.0, dt=10.0, run_seconds=60.0,
                     sf_sfclay_physics=1, **kw)


# --- the option -----------------------------------------------------------

def test_the_option_follows_the_mode_and_is_read_by_noah_only():
    from woof import correct_mode
    from woof.quirk_corrections import run_reads
    field = RunConfig.__dataclass_fields__["noah_land_ice_rain_runoff"]
    # correct-mode finding FQ8: on in a default process, off in exact mode
    assert field.default is correct_mode.CORRECT_KEYS_DEFAULT and field.kw_only
    assert correct_mode.resolve("exact").settings["noah_land_ice_rain_runoff"] is False
    assert run_reads(validate_run_config(_run(sf_surface_physics=2, noah_land_ice_rain_runoff=True)),
                     "noah_land_ice_rain_runoff")
    assert not run_reads(validate_run_config(_run(sf_surface_physics=1, num_soil_layers=5,
                                                  noah_land_ice_rain_runoff=True)),
                         "noah_land_ice_rain_runoff")
    with pytest.raises(ConfigurationRefused, match="noah_land_ice_rain_runoff"):
        validate_run_config(_run(sf_surface_physics=2, noah_land_ice_rain_runoff=1))


def test_the_umbrella_sets_it():
    assert _run(sf_surface_physics=2, woof_quirk_corrections="all").noah_land_ice_rain_runoff is True
    assert _run(sf_surface_physics=2, woof_quirk_corrections="none").noah_land_ice_rain_runoff is False
    from woof import quirk_corrections as qc
    with pytest.raises(ConfigurationRefused, match="noah_land_ice_rain_runoff"):
        qc.apply_run_corrections({qc.UMBRELLA: "all", "sf_surface_physics": 2,
                                  "noah_land_ice_rain_runoff": False})


def test_a_wrf_hydro_coupling_reads_it():
    """The glacial arm writes SFCRUNOFF only, never the routing planes, so
    unlike FQ4 the coupling carries FQ8 (nothing negative reaches SOLDRAIN)."""
    from woof.core.noah import noah_define_tier
    assert noah_define_tier(False, True) == (("NOAH_LAND_ICE_RAIN_RUNOFF", 1),)
    assert noah_define_tier(True, True) == (("NOAH_SMC_FLOOR_BOOKING", 1), ("NOAH_LAND_ICE_RAIN_RUNOFF", 1))
    assert noah_define_tier(True, False) == (("NOAH_SMC_FLOOR_BOOKING", 1),)
    assert noah_define_tier() == ()


# --- the default ----------------------------------------------------------

@pytest.mark.parametrize("name", sorted(BEFORE))
def test_the_default_kernel_source_is_the_source_before_the_option(name):
    text = (KDIR / name).read_text(encoding="utf-8")
    blocks = BLOCK.findall(text)
    assert len(blocks) == 1, name
    assert "runoff1" in blocks[0] and "prcpf" in blocks[0]
    assert hashlib.sha256(BLOCK.sub("", text).encode()).hexdigest() == BEFORE[name]


# --- the mirror -------------------------------------------------------------

@pytest.fixture(scope="module")
def params():
    from woof.core.noah import load_tables, pack_params
    return pack_params(load_tables())


@pytest.fixture(autouse=True)
def _no_cpu_bridge(monkeypatch):
    """Frozen-soil SH2O = SMC (WRF's init without its CPU bridge): the columns
    are warm or glacial, where the init changes nothing the test grades."""
    import test_noah
    monkeypatch.setattr(test_noah, "sh2o_init", lambda smc, tslb, soil, p: np.array(smc, np.float64))


def _ice_col(params, *, rain, sfctmp=277.0, tsk=277.0, swdown=300.0, seed=1):
    """A land-ice column under a deep pack (no 0.10 m reset), warm air."""
    import test_noah
    col = test_noah._base_col(params, veg=ICE, soil=14, rng=np.random.default_rng(seed),
                              sfctmp=sfctmp, tsk=tsk, swdown=swdown, rain=rain, snow=True)
    col["snow"] = 400.0       # mm of water equivalent
    col["snowh"] = 1.5
    col["snowc"] = 1.0
    col["tslb"] = np.full(4, 270.0)
    col["sh2o"] = np.ones(4)
    col["smois"] = np.ones(4)
    return col


def _imbalance(col, out, dt):
    """Pack water created (m): d(snow) - dt (P - E) + R dt, with R the booked runoff."""
    rhs = dt * (1e-3 * (col["rainbl"] / dt - out["qfx"])) - out["runoff1"] * dt
    return (out["snow"] - col["snow"]) * 1e-3 - rhs


def test_the_mirror_books_the_rain_and_closes_the_water(params):
    from test_noah import DT, DZS
    from woof.verify.npref import np_noah_column
    rain = 0.8      # mm in the step
    for swdown in (0.0, 300.0, 900.0):          # frozen skin, and a melting pack
        col = _ice_col(params, rain=rain, swdown=swdown)
        exact = np_noah_column(dict(col), params, DT, DZS)
        fixed = np_noah_column(dict(col), params, DT, DZS, land_ice_rain_runoff=True)
        assert exact["skip"] == 3 and fixed["skip"] == 3
        # WRF's word: the rain is in nobody's budget, whatever melts
        assert _imbalance(col, exact, DT) == pytest.approx(-rain * 1e-3, rel=1e-9, abs=1e-12)
        # booked: all of it leaves as surface runoff and the pack budget closes
        assert fixed["runoff1"] - exact["runoff1"] == pytest.approx(rain * 1e-3 / DT, rel=1e-12)
        assert abs(_imbalance(col, fixed, DT)) < 1e-12
        assert fixed["sfcrunoff"] - exact["sfcrunoff"] == pytest.approx(rain, rel=1e-9)
        # only the runoff moves: the pack and the energy balance are the exact ones
        for key in ("snow", "snowh", "tsk", "hfx", "lh", "grdflx", "noahres", "acsnom", "udrunoff"):
            np.testing.assert_array_equal(np.asarray(fixed[key]), np.asarray(exact[key]), err_msg=key)
    melted = np_noah_column(dict(_ice_col(params, rain=rain, swdown=900.0)), params, DT, DZS)
    assert melted["runoff1"] > 0.0          # the melting arm was exercised


@pytest.mark.parametrize("case", ["snowing", "freezing_rain", "dry"])
def test_the_mirror_is_unchanged_where_no_liquid_rain_reaches_the_ice(params, case):
    """Snow and freezing rain are added to the pack (:259-260, PRCPF = 0);
    a dry step has nothing to book."""
    from test_noah import DT, DZS
    from woof.verify.npref import np_noah_column
    col = {"snowing": dict(rain=0.8, sfctmp=268.0, tsk=268.0),
           "freezing_rain": dict(rain=0.8, sfctmp=275.0, tsk=271.0),
           "dry": dict(rain=0.0)}[case]
    col = _ice_col(params, **col)
    exact = np_noah_column(dict(col), params, DT, DZS)
    fixed = np_noah_column(dict(col), params, DT, DZS, land_ice_rain_runoff=True)
    for key, value in exact.items():
        np.testing.assert_array_equal(np.asarray(fixed[key]), np.asarray(value), err_msg=key)
    if case != "dry":
        assert exact["snow"] > col["snow"] - 1.0       # the precipitation went into the pack


def test_ordinary_land_is_unchanged(params):
    """Only the land-ice arm reads the option."""
    import test_noah
    from test_noah import DT, DZS
    from woof.verify.npref import np_noah_column
    col = test_noah._base_col(params, veg=7, soil=6, rain=1.0, sfctmp=285.0)
    exact = np_noah_column(dict(col), params, DT, DZS)
    fixed = np_noah_column(dict(col), params, DT, DZS, land_ice_rain_runoff=True)
    for key, value in exact.items():
        np.testing.assert_array_equal(np.asarray(fixed[key]), np.asarray(value), err_msg=key)


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


def _device_cols(params):
    rainy = [_ice_col(params, rain=0.8, swdown=s, seed=i) for i, s in enumerate((0.0, 300.0, 900.0))]
    others = [_ice_col(params, rain=0.8, sfctmp=268.0, tsk=268.0),              # snowing
              _ice_col(params, rain=0.8, sfctmp=275.0, tsk=271.0),              # freezing rain
              _ice_col(params, rain=0.0)]                                       # dry
    import test_noah
    land = [test_noah._base_col(params, veg=7, soil=6, rain=1.0, sfctmp=285.0)]
    return rainy, others, land


@requires_gpu
@pytest.mark.gpu
def test_the_kernel_books_the_rain_and_closes_the_water(params):
    from test_noah import DT, DZS
    from woof.verify.npref import np_noah_column
    rainy, others, land = _device_cols(params)
    cols = rainy + others + land
    stock, n, nx = _launch(cols, params)
    fixed, _, _ = _launch(cols, params, land_ice_rain_runoff=True)
    both, _, _ = _launch(cols, params, land_ice_rain_runoff=True, smc_floor_booking=True)
    for m, col in enumerate(cols):
        j, i = m // nx, m % nx
        c32 = {k: (np.asarray(v, np.float32).astype(np.float64) if k in ("smois", "tslb", "sh2o")
                   else int(v) if k in ("ivgtyp", "isltyp") else float(np.float32(v))) for k, v in col.items()}
        ref_stock = np_noah_column(dict(c32), params, DT, DZS)
        ref_fixed = np_noah_column(dict(c32), params, DT, DZS, land_ice_rain_runoff=True)
        booked = (fixed["sfcrunoff"][j, i] - stock["sfcrunoff"][j, i])       # mm
        if m < len(rainy):
            # the kernel books the rain the mirror books, and the exact word loses it
            want = (ref_fixed["sfcrunoff"] - ref_stock["sfcrunoff"])
            assert want == pytest.approx(0.8, rel=1e-6)
            assert booked == pytest.approx(want, rel=2e-3, abs=1e-5)
            assert fixed["sfcrunoff"][j, i] == pytest.approx(ref_fixed["sfcrunoff"], rel=2e-3, abs=1e-4)
            snow_in = np.float32(c32["snow"])
            created = ((fixed["snow"][j, i] - snow_in) * 1e-3
                       - (0.8e-3 - fixed["qfx"][j, i] * DT * 1e-3) + (fixed["sfcrunoff"][j, i] - c32["sfcrunoff"]) * 1e-3)
            assert abs(created) < 2e-6, created                # m: the f32 pack, closed to its rounding
            lost = ((stock["snow"][j, i] - snow_in) * 1e-3
                    - (0.8e-3 - stock["qfx"][j, i] * DT * 1e-3) + (stock["sfcrunoff"][j, i] - c32["sfcrunoff"]) * 1e-3)
            assert lost == pytest.approx(-0.8e-3, abs=2e-6)    # WRF's word drops the rain
        else:
            assert booked == 0.0, (m, booked)
        # FQ4 does not touch the glacial arm, and FQ8 nothing but SFCRUNOFF
        if m < len(rainy + others):
            for key in ("snow", "tsk", "sfcrunoff", "udrunoff"):
                np.testing.assert_array_equal(both[key][j, i].view(np.uint32),
                                              fixed[key][j, i].view(np.uint32), err_msg=key)
        for key in ("snow", "snowh", "tsk", "hfx", "lh", "grdflx", "noahres"):
            np.testing.assert_array_equal(fixed[key][j, i].view(np.uint32), stock[key][j, i].view(np.uint32),
                                          err_msg=key)


@requires_gpu
@pytest.mark.gpu
def test_the_composed_units_compile():
    from woof.core import noah_mosaic
    from woof.core.kernels import get_kernel_int_defines
    from woof.core.noah import noah_define_module, noah_define_tier
    assert noah_mosaic._mosaic_kernel(False, False, True) is not None
    assert noah_mosaic._mosaic_kernel(False, True, True) is not None
    assert noah_mosaic._mosaic_kernel(True, True, True) is not None
    assert noah_mosaic._mosaic_ucm_module(False, True, True).get_function("noah_mosaic_ucm_column") is not None
    for module in ("noah", "noah_ua"):
        for tier in (noah_define_tier(False, True), noah_define_tier(True, True)):
            assert get_kernel_int_defines(module, "noah_column", tier) is not None
    assert noah_define_module("noah_ua", noah_define_tier(True, True)) is not None
