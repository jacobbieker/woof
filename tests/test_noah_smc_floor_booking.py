"""The opt-in Noah soil-moisture floor booking (RunConfig.noah_smc_floor_booking).

WRF 4.7.1 module_sf_noahlsm.F SSTEP sets ``SMC(K) = MAX(MIN(STOT, SMCMAX),
0.02)`` (:4030).  The upper clip books its excess: WPLUS carries it to the
layer below and, past the bottom layer, into RUNOFF3 (:4026, :4075).  The
lower floor books nothing, so a sandy or desert column whose layer dries
below 0.02 gains the difference out of nowhere (up to 30 mm in one step on
a column that starts below the floor).  The booking (noah.cu and
noah_mosaic.cu ``NOAH_SMC_FLOOR_BOOKING``, npref ``smc_floor_booking``)
takes that water from the layer below as a negative WPLUS, and past the
bottom layer from RUNOFF3, which may then go negative.

Graded here:

* the default is word for word: the kernels with their opt-in blocks
  removed are byte for byte the sources before the option, and a column the
  floor never reaches is bitwise unchanged with the booking on;
* booked, the float64 mirror closes the column water budget
  d(canopy + soil + snow) = dt (P - E) - R on columns the default does not;
* on the device, the booked kernel matches the booked mirror and closes the
  same budget; the mosaic and UA units compile with the define.
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
    "noah.cu": "96e0b62e529d90ecdc25f06b91bf5aaa918afd508513d66073bec7553b8cef03",
    "noah_mosaic.cu": "d8739ffa18b0f3f594871cd88a7be274697ba3fb583bf788b4ed49209bdc7e19",
}
BLOCK = re.compile(r"#ifdef NOAH_SMC_FLOOR_BOOKING\n.*?#endif\n", re.S)
#: noah_land_ice_rain_runoff's block (FQ8), stripped too: it came after this option
FQ8_BLOCK = re.compile(r"#ifdef NOAH_LAND_ICE_RAIN_RUNOFF\n.*?#endif\n", re.S)


# --- the option -----------------------------------------------------------

def _run(**kw):
    return RunConfig(nx=8, ny=8, nz=12, dx=3000.0, dy=3000.0, ztop=12000.0, dt=10.0, run_seconds=60.0,
                     sf_sfclay_physics=1, **kw)


def test_the_option_follows_the_mode_and_is_read_by_noah_only():
    from woof import correct_mode
    from woof.quirk_corrections import run_reads
    field = RunConfig.__dataclass_fields__["noah_smc_floor_booking"]
    # correct-mode finding FQ4: on in a default process, off in exact mode
    assert field.default is correct_mode.CORRECT_KEYS_DEFAULT and field.kw_only
    assert correct_mode.resolve("exact").settings["noah_smc_floor_booking"] is False
    assert run_reads(validate_run_config(_run(sf_surface_physics=2, noah_smc_floor_booking=True)),
                     "noah_smc_floor_booking")
    assert not run_reads(validate_run_config(_run(sf_surface_physics=1, num_soil_layers=5,
                                                  noah_smc_floor_booking=True)), "noah_smc_floor_booking")
    with pytest.raises(ConfigurationRefused, match="noah_smc_floor_booking"):
        validate_run_config(_run(sf_surface_physics=2, noah_smc_floor_booking=1))


def test_the_umbrella_sets_it():
    assert _run(sf_surface_physics=2, woof_quirk_corrections="all").noah_smc_floor_booking is True
    assert _run(sf_surface_physics=2, woof_quirk_corrections="none").noah_smc_floor_booking is False
    from woof import quirk_corrections as qc
    with pytest.raises(ConfigurationRefused, match="noah_smc_floor_booking"):
        qc.apply_run_corrections({qc.UMBRELLA: "all", "sf_surface_physics": 2, "noah_smc_floor_booking": False})


def test_a_wrf_hydro_coupling_does_not_read_the_booking():
    from types import SimpleNamespace
    from woof.core.noah import launch_noah
    from woof.core.physics import _noah_smc_floor_booking
    with pytest.raises(ValueError, match="noah_smc_floor_booking"):
        launch_noah({}, None, 10.0, None, wrf_hydro={}, smc_floor_booking=True)
    hydro = {"wrf_hydro": {}}
    on = SimpleNamespace(noah_smc_floor_booking=True, sf_surface_physics=2)
    assert _noah_smc_floor_booking(on, hydro) is False
    assert _noah_smc_floor_booking(on, {}) is True
    assert _noah_smc_floor_booking(SimpleNamespace(noah_smc_floor_booking=True, sf_surface_physics=1), {}) is False


# --- the default ----------------------------------------------------------

@pytest.mark.parametrize("name", sorted(BEFORE))
def test_the_default_kernel_source_is_the_source_before_the_option(name):
    text = (KDIR / name).read_text(encoding="utf-8")
    blocks = BLOCK.findall(text)
    assert len(blocks) == 1, name
    assert "wplus" in blocks[0] and "0.02f" in blocks[0]
    assert hashlib.sha256(FQ8_BLOCK.sub("", BLOCK.sub("", text)).encode()).hexdigest() == BEFORE[name]


# --- the mirror -------------------------------------------------------------

@pytest.fixture(scope="module")
def params():
    from woof.core.noah import load_tables, pack_params
    return pack_params(load_tables())


def _columns(params, monkeypatch):
    """Warm, sunny, dry columns over every soil, two of them below the floor."""
    import test_noah
    # warm soils hold no ice, so SH2O = SMC (WRF's init, without its CPU bridge)
    monkeypatch.setattr(test_noah, "sh2o_init", lambda smc, tslb, soil, p: np.array(smc, np.float64))
    rng = np.random.default_rng(3)
    cols = []
    for soil in range(1, params.slcats + 1):
        for frac in (0.0, 0.05, 0.6):
            col = test_noah._base_col(params, veg=7, soil=soil, rng=rng, swdown=800.0, rh=0.1,
                                      chs=0.03, smc_frac=frac)
            col["smois"] = np.maximum(col["smois"], 0.005)
            col["sh2o"] = col["smois"].copy()
            assert min(col["tslb"]) > 273.15
            cols.append(col)
    return cols


def _imbalance(col, out, dt):
    """Column water created (m): storage change minus dt (P - E) - R."""
    from test_noah import _storage_m
    rhs = dt * (1e-3 * (col["rainbl"] / dt - out["qfx"]) - out["runoff1"] - out["runoff2t"])
    return (_storage_m(out) - _storage_m(col)) - rhs


def test_the_mirror_books_the_floor_water(params, monkeypatch):
    from test_noah import DT, DZS
    from woof.verify.npref import np_noah_column
    created = closed = untouched = unflagged = 0
    for col in _columns(params, monkeypatch):
        stock = np_noah_column(dict(col), params, DT, DZS)
        default = np_noah_column(dict(col), params, DT, DZS, smc_floor_booking=False)
        for key, value in stock.items():
            np.testing.assert_array_equal(np.asarray(default[key]), np.asarray(value), err_msg=key)
        booked = np_noah_column(dict(col), params, DT, DZS, smc_floor_booking=True)
        if stock["skip"]:
            continue
        # a layer the floor lifted may refill from above within the step: the stock flag tells
        floored = stock["clamped"] or any(stock["smois"] <= 0.02 + 1e-12)
        if not floored:
            for key, value in stock.items():
                np.testing.assert_array_equal(np.asarray(booked[key]), np.asarray(value), err_msg=key)
            untouched += 1
            continue
        if _imbalance(col, stock, DT) > 1e-6:
            created += 1
        assert abs(_imbalance(col, booked, DT)) < 1e-12, (col["isltyp"], _imbalance(col, booked, DT))
        assert stock["clamped"]
        unflagged += not booked["clamped"]
        closed += 1
    assert created >= 5 and closed >= created and untouched >= 5, (created, closed, untouched)
    # booked, the 0.02 floor no longer marks a column as creating water
    assert unflagged >= 5, unflagged


# --- the device -------------------------------------------------------------

def _launch(cols, params, booked):
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
    launch_noah(dev, params, DT, DZS, smc_floor_booking=booked)
    return {k: cp.asnumpy(v) for k, v in dev.items()}, n, nx


@requires_gpu
@pytest.mark.gpu
def test_the_kernel_books_the_floor_water(params, monkeypatch):
    from test_noah import DT, DZS
    from woof.verify.npref import np_noah_column
    cols = _columns(params, monkeypatch)
    stock, n, nx = _launch(cols, params, False)
    booked, _, _ = _launch(cols, params, True)
    graded = 0
    for m, col in enumerate(cols):
        j, i = m // nx, m % nx
        c32 = {k: (np.asarray(v, np.float32).astype(np.float64) if k in ("smois", "tslb", "sh2o")
                   else int(v) if k in ("ivgtyp", "isltyp") else float(np.float32(v))) for k, v in col.items()}
        ref = np_noah_column(dict(c32), params, DT, DZS, smc_floor_booking=True)
        if ref["skip"]:
            continue
        stock_ref = np_noah_column(dict(c32), params, DT, DZS)
        if not stock_ref["clamped"] and all(np.asarray(ref["smois"]) > 0.02 + 1e-6):
            # the floor never acts: the booked unit is the stock unit's arithmetic
            for key in ("smois", "sh2o", "tslb"):
                np.testing.assert_array_equal(booked[key][:, j, i].view(np.uint32),
                                              stock[key][:, j, i].view(np.uint32), err_msg=key)
            continue
        np.testing.assert_allclose(booked["smois"][:, j, i], ref["smois"], rtol=2e-4, atol=2e-6)
        # mm accumulators: the booked bottom drainage carries the floor's water
        drained = (booked["udrunoff"][j, i] - c32["udrunoff"]) * 1e-3
        np.testing.assert_allclose(drained, ref["runoff2t"] * DT, rtol=2e-3, atol=2e-6)
        graded += 1
    assert graded >= 5


@requires_gpu
@pytest.mark.gpu
def test_the_mosaic_and_ua_units_compile_with_the_define():
    from woof.core import noah_mosaic
    from woof.core.kernels import get_kernel_int_defines
    from woof.core.noah import NOAH_SMC_FLOOR_BOOKING_DEFINES
    assert noah_mosaic._mosaic_kernel(False, True) is not None
    assert noah_mosaic._mosaic_kernel(True, True) is not None
    assert noah_mosaic._mosaic_ucm_module(False, True).get_function("noah_mosaic_ucm_column") is not None
    for module in ("noah", "noah_ua"):
        assert get_kernel_int_defines(module, "noah_column", NOAH_SMC_FLOOR_BOOKING_DEFINES) is not None
