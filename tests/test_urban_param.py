"""urban_param_file: WPS's URB_PARAM through real.exe's split, its FRC_URB2D
rules and the refusals (woof.core.urban_param).  Host-only, synthetic files.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from woof.core.urban_param import (
    URB_PARAM_INDEX,
    URB_PARAM_SIZE,
    UrbanParamError,
    check_nudapt_columns,
    load_urban_morphology,
    read_urban_param,
    real_frc_fill,
    real_ivgtyp_promotion,
    split_urb_param,
)

NY, NX = 3, 4


def _block(seed=0):
    """A (132, ny, nx) block whose every slot is identifiable: slot s
    (1-based) holds s + cell / 100."""
    cell = np.arange(NY * NX, dtype=np.float32).reshape(NY, NX) / 100
    return np.stack([np.float32(s) + cell for s in range(1, 133)]) + seed


def _morph_block():
    """A physically plausible block on every column."""
    block = np.zeros((URB_PARAM_SIZE, NY, NX), np.float32)
    put = lambda index, value: block.__setitem__(index - 1, value)
    put(91, 0.4)        # LP
    put(92, 12.0)       # MH
    put(93, 4.0)        # STDH
    put(94, 12.0)       # HGT
    put(95, 1.6)        # LB
    for k in range(4):
        put(96 + k, 0.3 + 0.05 * k)
    for k in range(15):
        put(118 + k, 1.0 / 15)
    return block


def _write(path, block, *, time_axis=True, frc=None, lat=None, lon=None,
           dx=None, dy=None, schema=None):
    import netCDF4

    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("south_north", block.shape[1])
        ds.createDimension("west_east", block.shape[2])
        ds.createDimension("num_urb_params", block.shape[0])
        if time_axis:
            ds.createDimension("Time", None)
            dims = ("Time", "num_urb_params", "south_north", "west_east")
            ds.createVariable("URB_PARAM", "f4", dims)[:] = block[None]
        else:
            dims = ("num_urb_params", "south_north", "west_east")
            ds.createVariable("URB_PARAM", "f4", dims)[:] = block
        two = (("Time",) if time_axis else ()) + ("south_north", "west_east")
        for name, value in (("FRC_URB2D", frc), ("XLAT_M", lat),
                            ("XLONG_M", lon)):
            if value is not None:
                var = ds.createVariable(name, "f4", two)
                var[:] = value[None] if time_axis else value
        if dx is not None:
            ds.DX = np.float32(dx)
        if dy is not None:
            ds.DY = np.float32(dy)
        if schema is not None:
            ds.schema = schema
    return path


# ---------------------------------------------------------------------------
# The index map and the split
# ---------------------------------------------------------------------------

def test_index_map_is_real_exes_and_the_products():
    from woof.static import buildings

    assert URB_PARAM_INDEX == {
        "LP_URB2D": 91, "MH_URB2D": 92, "STDH_URB2D": 93, "HGT_URB2D": 94,
        "LB_URB2D": 95, "LF_URB2D": 96, "HI_URB2D": 118}
    assert URB_PARAM_INDEX == buildings.URB_PARAM_INDEX
    assert URB_PARAM_SIZE == buildings.URB_PARAM_SIZE == 132
    assert URB_PARAM_INDEX["HI_URB2D"] + 14 == URB_PARAM_SIZE


@pytest.mark.parametrize("option", (1, 2, 3))
def test_split_reads_the_slots_each_option_reads(option):
    block = _block()
    out = split_urb_param(block, option)
    slot = lambda s: block[s - 1]
    np.testing.assert_array_equal(out["lp"], slot(91))
    np.testing.assert_array_equal(out["hgt"], slot(94))
    np.testing.assert_array_equal(out["lb"], slot(95))
    if option == 1:
        np.testing.assert_array_equal(out["mh"], slot(92))
        np.testing.assert_array_equal(out["stdh"], slot(93))
        for k in range(4):
            np.testing.assert_array_equal(out["lf"][k], slot(96 + k))
        assert not out["hi"].any()
    else:
        assert not (out["mh"].any() or out["stdh"].any() or out["lf"].any())
        for k in range(15):
            np.testing.assert_array_equal(out["hi"][k], slot(118 + k))
    assert all(v.dtype == np.float32 for v in out.values())


def test_split_refuses_option_zero():
    with pytest.raises(UrbanParamError):
        split_urb_param(_block(), 0)


def test_the_product_block_splits_back_to_its_morphology():
    """woof.static.buildings.urb_param -> split recovers each plane."""
    from woof.static.buildings import BuildingMorphology, urb_param

    rng = np.random.default_rng(3)
    plane = lambda: rng.uniform(0.1, 0.9, (NY, NX)).astype(np.float32)
    morph = BuildingMorphology(
        sampled_area=plane(), lambda_p=plane(), mean_height=plane() * 30,
        std_height=plane(), lambda_b=plane() + 1, frontal_x=plane(),
        frontal_y=plane(), frontal_d45=plane(), frontal_d135=plane(),
        hi=rng.uniform(0, 1, (15, NY, NX)).astype(np.float32),
        bldg_hist=np.zeros((1, NY, NX), np.float32),
        bldg_hbin=np.zeros((1, NY, NX), np.float32))
    one = split_urb_param(urb_param(morph), 1)
    np.testing.assert_array_equal(one["lp"], morph.lambda_p)
    np.testing.assert_array_equal(one["mh"], morph.mean_height)
    np.testing.assert_array_equal(one["hgt"], morph.mean_height)
    np.testing.assert_array_equal(one["lf"], morph.lambda_f_sectors())
    np.testing.assert_array_equal(split_urb_param(urb_param(morph), 2)["hi"],
                                  morph.hi)


# ---------------------------------------------------------------------------
# real.exe's FRC_URB2D rules
# ---------------------------------------------------------------------------

def test_frc_fill_is_point_nine_on_urban_categories_only():
    ivgtyp = np.array([[13, 24, 25, 26, 5, 1]])
    frc = np.zeros((1, 6), np.float32)
    nlcd = real_frc_fill(frc, ivgtyp, "NLCD40")
    np.testing.assert_array_equal(
        nlcd, np.float32([[0.9, 0.9, 0.9, 0.9, 0, 0]]))
    usgs = real_frc_fill(frc, ivgtyp, "USGS")
    np.testing.assert_array_equal(usgs, np.float32([[0, 0, 0, 0, 0, 0.9]]))
    assert not real_frc_fill(frc, ivgtyp, "SOMETHING_ELSE").any()
    kept = real_frc_fill(np.float32([[0.3]]), np.array([[13]]), "NLCD40")
    assert kept[0, 0] == np.float32(0.3)


def test_promotion_is_detected_as_real_exe_would_apply_it():
    ivgtyp = np.array([[5, 13, 31, 5, 24]])
    frc = np.float32([[0.5, 0.9, 0.9, 0.49, 0.7]])
    promoted = real_ivgtyp_promotion(frc, ivgtyp, "MODIFIED_IGBP_MODIS_NOAH")
    assert promoted.tolist() == [[True, False, False, False, False]]
    assert real_ivgtyp_promotion(frc, np.array([[2, 1, 2, 2, 2]]),
                                 "USGS").tolist() == \
        [[True, False, True, False, True]]


# ---------------------------------------------------------------------------
# The reader and loader
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("time_axis", (True, False))
def test_reader_takes_wps_and_product_layouts(tmp_path, time_axis):
    block = _block()
    path = _write(tmp_path / "u.nc", block, time_axis=time_axis)
    got, frc, grid = read_urban_param(path, shape=(NY, NX))
    np.testing.assert_array_equal(got, block)
    assert frc is None and "lat" not in grid


def test_reader_refuses_another_grid_and_a_missing_file(tmp_path):
    path = _write(tmp_path / "u.nc", _block())
    with pytest.raises(UrbanParamError, match="mass grid"):
        read_urban_param(path, shape=(NY + 1, NX))
    with pytest.raises(UrbanParamError, match="does not exist"):
        read_urban_param(tmp_path / "nope.nc", shape=(NY, NX))
    import netCDF4
    with netCDF4.Dataset(tmp_path / "empty.nc", "w") as ds:
        ds.createDimension("x", 1)
    with pytest.raises(UrbanParamError, match="no URB_PARAM"):
        read_urban_param(tmp_path / "empty.nc", shape=(NY, NX))


def _load(path, *, ivgtyp=None, urban=None, option=1, **kw):
    ivgtyp = np.full((NY, NX), 13) if ivgtyp is None else ivgtyp
    urban = (ivgtyp == 13) if urban is None else urban
    return load_urban_morphology(path, option=option, ivgtyp=ivgtyp,
                                 urban=urban, mminlu="NLCD40", **kw)


def test_loader_checks_lat_lon_and_dx(tmp_path):
    lat = np.linspace(40, 41, NY * NX, dtype=np.float32).reshape(NY, NX)
    lon = np.linspace(-90, -89, NY * NX, dtype=np.float32).reshape(NY, NX)
    path = _write(tmp_path / "u.nc", _morph_block(), lat=lat, lon=lon,
                  dx=1000.0, dy=1000.0)
    _load(path, latitude=lat, longitude=lon, dx=1000.0, dy=1000.0)
    with pytest.raises(UrbanParamError, match="another grid"):
        _load(path, latitude=lat + 0.01, longitude=lon)
    with pytest.raises(UrbanParamError, match="DX"):
        _load(path, dx=3000.0)
    # a longitude that wraps the dateline is the same grid
    wrapped = _write(tmp_path / "w.nc", _morph_block(), lat=lat,
                     lon=np.full((NY, NX), 179.9999, np.float32))
    _load(wrapped, latitude=lat, longitude=np.full((NY, NX), -180.0))


def test_loader_refuses_non_finite_on_urban_columns_only(tmp_path):
    block = _morph_block()
    block[94 - 1, 0, 0] = np.nan                       # HGT on column (0, 0)
    path = _write(tmp_path / "u.nc", block)
    with pytest.raises(UrbanParamError, match="HGT_URB2D is missing"):
        _load(path)
    rural = np.ones((NY, NX), bool)
    rural[0, 0] = False
    _load(path, urban=rural)                           # (0, 0) not urban
    block = _morph_block()
    block[120 - 1, 1, 1] = np.inf                      # an HI bin
    path = _write(tmp_path / "hi.nc", block)
    _load(path, option=1)                              # option 1 reads no HI
    with pytest.raises(UrbanParamError, match="HI_URB2D"):
        _load(path, option=2)


def test_loader_applies_the_fill_and_refuses_promotion(tmp_path):
    ivgtyp = np.full((NY, NX), 13)
    ivgtyp[0, :] = 5
    frc = np.full((NY, NX), 0.0, np.float32)
    frc[0, :] = 0.2                                     # non-urban, < 0.5
    path = _write(tmp_path / "u.nc", _morph_block(), frc=frc)
    out = _load(path, ivgtyp=ivgtyp)
    np.testing.assert_array_equal(out.frc[1:], np.float32(0.9))
    np.testing.assert_array_equal(out.frc[0], np.float32(0.2))
    frc[0, 1] = 0.5
    path = _write(tmp_path / "p.nc", _morph_block(), frc=frc)
    with pytest.raises(UrbanParamError, match="reclassify 1 columns"):
        _load(path, ivgtyp=ivgtyp)
    frc[0, 1] = np.nan
    path = _write(tmp_path / "n.nc", _morph_block(), frc=frc)
    with pytest.raises(UrbanParamError, match="FRC_URB2D is missing"):
        _load(path, ivgtyp=ivgtyp)


def test_a_file_without_frc_keeps_the_routes(tmp_path):
    out = _load(_write(tmp_path / "u.nc", _morph_block(), time_axis=False))
    assert out.frc is None
    assert out.mh.dtype == np.float32 and out.lf.shape == (4, NY, NX)


def test_nudapt_refusals_name_what_wrf_would_do():
    base = split_urb_param(_morph_block(), 1)
    urban = np.ones((NY, NX), bool)
    check_nudapt_columns(base, urban)
    for edit, match in (
            (lambda m: m["lf"].__setitem__((2, 0, 0), 0.0), "LF_URB2D sector"),
            (lambda m: m["lp"].__setitem__((0, 0), 0.0), "LP_URB2D outside"),
            (lambda m: m["lp"].__setitem__((0, 0), 1.0), "LP_URB2D outside"),
            (lambda m: m["stdh"].__setitem__((0, 0), -1.0), "negative")):
        morph = {k: v.copy() for k, v in base.items()}
        edit(morph)
        with pytest.raises(UrbanParamError, match=match):
            check_nudapt_columns(morph, urban)
        # the same column, not urban or without MH, is not refused
        off = urban.copy()
        off[0, 0] = False
        check_nudapt_columns(morph, off)
        morph["mh"][0, 0] = 0.0
        check_nudapt_columns(morph, urban)


# ---------------------------------------------------------------------------
# The configuration door
# ---------------------------------------------------------------------------

def _cfg(**kw):
    from woof.config import RunConfig

    base = dict(nx=8, ny=8, nz=10, dx=3000.0, dy=3000.0, ztop=20000.0,
                dt=10.0, run_seconds=60.0, sf_sfclay_physics=1,
                sf_surface_physics=2, bl_pbl_physics=1)
    base.update(kw)
    return RunConfig(**base)


def test_config_default_is_off():
    assert _cfg().urban_param_file == ""
    field = {f.name: f for f in dataclasses.fields(_cfg())}["urban_param_file"]
    assert field.default == ""


@pytest.mark.parametrize("kw, match", (
    (dict(sf_urban_physics=0, urban_param_file="g.nc"), "sf_urban_physics = 0"),
    (dict(sf_urban_physics=1, urban_ucm_generation="wrf_480",
          urban_param_file="g.nc"), "wrf_480"),
    (dict(sf_urban_physics=1, sf_surface_mosaic=1, urban_param_file="g.nc"),
     "sf_surface_mosaic"),
))
def test_config_refusals(kw, match):
    from woof.config import ConfigurationRefused, validate_urban_config

    with pytest.raises((ConfigurationRefused, ValueError), match=match):
        validate_urban_config(_cfg(**kw))


def test_config_refuses_a_non_string():
    from woof.config import ConfigurationRefused, validate_urban_config

    cfg = _cfg(sf_urban_physics=1)
    object.__setattr__(cfg, "urban_param_file", 3)
    with pytest.raises(ConfigurationRefused, match="path string"):
        validate_urban_config(cfg)
