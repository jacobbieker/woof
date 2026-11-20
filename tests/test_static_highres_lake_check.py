"""``[static.highres] fields = "lakes"``: inland lake extent from the land cover.

The WPS ``modis_landuse_20class_30s_with_lakes`` tiles mark glacier, rock and
shaded alpine pixels as lakes (MODIS IGBP water), so an Alpine domain gets
hundreds of LAKEMASK cells at 2000-4000 m.  CGLC-MODIS-LCZ (100 m) calls them
ice, barren land and grassland.  Their real lakes also sit about one
30-arc-second pixel north of the 100 m shorelines.  The mode takes the lake
status of each cell from the land cover, and leaves terrain, the sea and
every other cell on the baseline, bit for bit.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from woof.static import highres as highres_module
from woof.static.build import HALO
from woof.static.highres_production import (HighresRefusal,
                                             HighresStaticConfig,
                                             apply_highres_statics,
                                             parse_static_table)
from woof.static.lambert import LambertGrid

MODIS21_ATTRS = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
                 "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}
N = 12
#: Baseline lakes: a glacier "lake" the source calls ice (15), a rock one
#: it calls barren (16), a real lake it confirms (21).
GLACIER, ROCK, REAL = (2, 2), (2, 3), (8, 8)
#: A baseline grassland cell the source calls a lake: becomes a lake.
NEW_WATER = (5, 5)
#: A baseline sea cell the source calls a lake, and a baseline lake it calls
#: the sea: neither changes (the sea is not this mode's).
SEA_TO_LAKE, LAKE_TO_SEA = (10, 1), (10, 3)


class _Stub:
    def __init__(self, source_id: str):
        self._source_id = source_id

    def receipt(self):
        return {"source_id": self._source_id}


def _grid() -> LambertGrid:
    return LambertGrid(ref_lat=46.5, ref_lon=8.0, truelat1=30.0,
                       truelat2=60.0, stand_lon=8.0, dx=1000.0, dy=1000.0,
                       e_we=N + 1, e_sn=N + 1)


def _baseline(lakes=(GLACIER, ROCK, REAL, LAKE_TO_SEA)):
    luf = np.zeros((21, N, N))
    luf[9] = 1.0  # grassland (10)
    landmask = np.ones((N, N))
    lu_index = np.full((N, N), 10.0)
    soil = np.zeros((16, N, N))
    soil[2] = 1.0
    hgt = np.full((N, N), 600.0)
    for cell in lakes:
        luf[:, cell[0], cell[1]] = 0.0
        luf[20, cell[0], cell[1]] = 1.0
        landmask[cell] = 0.0
        lu_index[cell] = 21.0
        soil[:, cell[0], cell[1]] = 0.0
        soil[13, cell[0], cell[1]] = 1.0  # water soil
    luf[:, SEA_TO_LAKE[0], SEA_TO_LAKE[1]] = 0.0
    luf[16, SEA_TO_LAKE[0], SEA_TO_LAKE[1]] = 1.0
    landmask[SEA_TO_LAKE] = 0.0
    lu_index[SEA_TO_LAKE] = 17.0
    soil[:, SEA_TO_LAKE[0], SEA_TO_LAKE[1]] = 0.0
    soil[13, SEA_TO_LAKE[0], SEA_TO_LAKE[1]] = 1.0
    hgt[GLACIER], hgt[ROCK] = 640.0, 620.0
    rng = np.random.default_rng(4)
    water = landmask < 0.5
    climatology = {
        "GREENFRAC": rng.uniform(0.1, 0.9, (12, N, N)),
        "LAI12M": rng.uniform(0.5, 3.0, (12, N, N)),
        "ALBEDO12M": rng.uniform(12.0, 20.0, (12, N, N)),
        "SNOALB": rng.uniform(40.0, 70.0, (N, N)),
        "SOILTEMP": rng.uniform(280.0, 286.0, (N, N)),
    }
    # geo_em's convention over water, as the merge writes it.
    for name, fill in (("GREENFRAC", 0.0), ("LAI12M", 0.0), ("ALBEDO12M", 8.0),
                       ("SNOALB", 0.0), ("SOILTEMP", 0.0)):
        climatology[name][..., water] = fill
    return {
        "HGT_M": hgt, "LANDUSEF": luf, "LANDMASK": landmask,
        "LU_INDEX": lu_index, "SOILCTOP": soil.copy(),
        "SOILCBOT": soil.copy(), "SCT_DOM": np.where(landmask > 0, 3.0, 14.0),
        "SCB_DOM": np.where(landmask > 0, 3.0, 14.0),
        **climatology,
    }


_PLACEMENTS = ((GLACIER, 15), (ROCK, 16), (REAL, 21), (NEW_WATER, 21),
               (SEA_TO_LAKE, 21), (LAKE_TO_SEA, 17))


def _stub(monkeypatch, seen, placements=_PLACEMENTS):
    def categories(source, grid, mapping, *, category_count):
        ny, nx = grid.e_sn - 1, grid.e_we - 1
        halo = (ny - N) // 2
        fractions = np.zeros((category_count, ny, nx))
        fractions[9] = 1.0
        for (row, col), category in placements:
            fractions[:, row + halo, col + halo] = 0.0
            fractions[category - 1, row + halo, col + halo] = 1.0
        return fractions

    def fetch(bbox, cache_root, case_date, *, landcover_source, urlopen=None):
        seen["landcover_source"] = landcover_source.source_id
        year, anachronism = landcover_source.year_for(case_date)
        return _Stub(landcover_source.bound_id(year)), {
            "landcover_source": landcover_source.source_id,
            "landcover_year": year,
            "landcover_anachronism_years": anachronism,
            "landcover_window_audit": None, "bytes_fetched": 0}

    def no_terrain_or_soil(*args, **kwargs):
        raise AssertionError("fields = 'lakes' fetched terrain or soil")

    monkeypatch.setattr(highres_module, "resample_mapped_categories",
                        categories)
    monkeypatch.setattr("woof.static.highres_production._fetch_landcover_only",
                        fetch)
    monkeypatch.setattr("woof.static.highres_production._fetch_and_bind",
                        no_terrain_or_soil)


def test_the_mode_is_a_declared_choice(tmp_path):
    config = parse_static_table(
        {"highres": {"enabled": True, "cache_root": "c", "fields": "lakes"}},
        source="case.toml", base_dir=tmp_path)
    assert config.fields == "lakes"
    assert config.echo()["fields"] == "lakes"
    with pytest.raises(ValueError, match="'lakes' takes the inland lake extent"):
        parse_static_table(
            {"highres": {"enabled": True, "cache_root": "c", "fields": "lake"}},
            source="case.toml", base_dir=tmp_path)


def test_only_the_lakes_the_land_cover_denies_change(tmp_path, monkeypatch,
                                                      capsys):
    seen = {}
    _stub(monkeypatch, seen)
    baseline = _baseline()
    fields, receipt = apply_highres_statics(
        baseline, _grid(),
        config=HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                   fields="lakes"),
        domain_id=2, case_date=date(2026, 11, 9), landuse_attrs=MODIS21_ATTRS)
    assert seen["landcover_source"] == "cglc-modis-lcz"
    assert receipt["status"] == "APPLIED" and receipt["mode"] == "lakes"

    for cell, category in ((GLACIER, 15.0), (ROCK, 16.0)):
        assert fields["LU_INDEX"][cell] == category
        assert fields["LANDMASK"][cell] == 1.0
        assert fields["LANDUSEF"][int(category) - 1][cell] == 1.0
        # Land soil (the nearest land cell's), not water soil.
        assert fields["SCT_DOM"][cell] == 3.0
        assert fields["SOILCTOP"][13][cell] == 0.0
        assert fields["SOILTEMP"][cell] > 0.0
    assert fields["LU_INDEX"][REAL] == 21.0 and fields["LANDMASK"][REAL] == 0.0
    assert fields["LU_INDEX"][NEW_WATER] == 21.0
    assert fields["LANDMASK"][NEW_WATER] == 0.0
    assert fields["SOILCTOP"][13][NEW_WATER] == 1.0
    assert fields["GREENFRAC"][:, NEW_WATER[0], NEW_WATER[1]].max() == 0.0
    assert fields["LU_INDEX"][SEA_TO_LAKE] == 17.0
    assert fields["LU_INDEX"][LAKE_TO_SEA] == 21.0
    np.testing.assert_array_equal(fields["HGT_M"], baseline["HGT_M"])

    changed = np.zeros((N, N), dtype=bool)
    changed[GLACIER] = changed[ROCK] = changed[NEW_WATER] = True
    for name in ("LANDUSEF", "LANDMASK", "LU_INDEX", "SOILCTOP", "SOILCBOT",
                 "SCT_DOM", "SCB_DOM", "GREENFRAC", "LAI12M", "ALBEDO12M",
                 "SNOALB"):
        after, before = np.asarray(fields[name]), np.asarray(baseline[name])
        np.testing.assert_array_equal(after[..., ~changed],
                                      before[..., ~changed], err_msg=name)

    check = receipt["lake_check"]
    assert check["baseline_lake_cells"] == 4 and check["lake_cells"] == 3
    assert check["demoted_cells"] == 2 and check["kept_lake_cells"] == 2
    assert check["promoted_cells"] == 1
    assert check["promoted_from_category"] == {"10": 1}
    assert check["demoted_to_category"] == {"15": 1, "16": 1}
    assert check["demoted_height_m"]["min"] == 620.0
    assert receipt["cells_replaced"]["total"] == 3
    assert ("lakes only, cglc-modis-lcz-2018: 2 of 4 lake cells are land "
            "there, 1 land cells are lake") in capsys.readouterr().out


def test_a_domain_whose_lakes_agree_is_applied_unchanged(tmp_path,
                                                         monkeypatch):
    seen = {}
    _stub(monkeypatch, seen)
    baseline = _baseline(lakes=(REAL, NEW_WATER, LAKE_TO_SEA))
    fields, receipt = apply_highres_statics(
        baseline, _grid(),
        config=HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                   fields="lakes"),
        domain_id=1, case_date=date(2026, 11, 9), landuse_attrs=MODIS21_ATTRS)
    assert receipt["lake_check"]["demoted_cells"] == 0
    assert receipt["lake_check"]["promoted_cells"] == 0
    for name in ("LU_INDEX", "LANDMASK", "LANDUSEF", "HGT_M"):
        np.testing.assert_array_equal(fields[name], baseline[name])


def test_a_domain_the_land_cover_does_not_reach_is_refused(tmp_path,
                                                          monkeypatch):
    seen = {}
    _stub(monkeypatch, seen)
    grid = LambertGrid(ref_lat=80.0, ref_lon=20.0, truelat1=60.0,
                       truelat2=80.0, stand_lon=20.0, dx=1000.0, dy=1000.0,
                       e_we=N + 1, e_sn=N + 1)
    with pytest.raises(HighresRefusal, match='fields = "lakes" needs'):
        apply_highres_statics(
            _baseline(), grid,
            config=HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                       fields="lakes"),
            domain_id=1, case_date=date(2026, 11, 9),
            landuse_attrs=MODIS21_ATTRS)


def _lakes(**kwargs):
    return HighresStaticConfig(enabled=True, cache_root=kwargs.pop("root"),
                               fields="lakes", **kwargs)


def test_lake_depth_follows_the_changed_cells(tmp_path, monkeypatch):
    """``woof go --lakes clm`` reads LAKE_DEPTH: a demoted cell takes the
    WPS land fill, a promoted one the nearest kept lake's depth."""
    _stub(monkeypatch, {})
    baseline = _baseline()
    depth = np.full((N, N), 10.0)
    depth[REAL], depth[LAKE_TO_SEA] = 55.0, 7.0
    depth[GLACIER], depth[ROCK] = 30.0, 31.0
    baseline["LAKE_DEPTH"] = depth
    fields, receipt = apply_highres_statics(
        baseline, _grid(), config=_lakes(root=tmp_path), domain_id=2,
        case_date=date(2026, 11, 9), landuse_attrs=MODIS21_ATTRS)
    after = fields["LAKE_DEPTH"]
    assert after[GLACIER] == 10.0 and after[ROCK] == 10.0
    # (5,5) -> (8,8) is 4.2 km, inside WPS's five-pixel search reach.
    assert after[NEW_WATER] == 55.0
    unchanged = np.ones((N, N), dtype=bool)
    unchanged[GLACIER] = unchanged[ROCK] = unchanged[NEW_WATER] = False
    np.testing.assert_array_equal(after[unchanged], depth[unchanged])
    record = receipt["lake_check"]["lake_depth"]
    assert record["demoted_to_fill_cells"] == 2
    assert record["promoted_from_nearest_lake_cells"] == 1
    assert "LAKE_DEPTH" in receipt["fields_replaced"]


def test_a_promoted_lake_beyond_the_search_reach_takes_the_fill(
        tmp_path, monkeypatch):
    _stub(monkeypatch, {})
    baseline = _baseline()
    baseline["LAKE_DEPTH"] = np.where(baseline["LU_INDEX"] == 21, 55.0, 10.0)
    grid = LambertGrid(ref_lat=46.5, ref_lon=8.0, truelat1=30.0,
                       truelat2=60.0, stand_lon=8.0, dx=3000.0, dy=3000.0,
                       e_we=N + 1, e_sn=N + 1)
    fields, receipt = apply_highres_statics(
        baseline, grid, config=_lakes(root=tmp_path), domain_id=1,
        case_date=date(2026, 11, 9), landuse_attrs=MODIS21_ATTRS)
    assert fields["LAKE_DEPTH"][NEW_WATER] == 10.0
    assert receipt["lake_check"]["lake_depth"]["promoted_to_fill_cells"] == 1


def test_unchanged_cells_are_the_baseline_bit_for_bit(tmp_path, monkeypatch):
    _stub(monkeypatch, {})
    baseline = _baseline()
    # A TMN no recompute reproduces, and a float32 field.
    baseline["TMN"] = np.where(baseline["LANDMASK"] > 0, 281.25, 0.0)
    baseline["SNOALB"] = baseline["SNOALB"].astype(np.float32)
    fields, receipt = apply_highres_statics(
        baseline, _grid(), config=_lakes(root=tmp_path), domain_id=2,
        case_date=date(2026, 11, 9), landuse_attrs=MODIS21_ATTRS)
    changed = np.zeros((N, N), dtype=bool)
    changed[GLACIER] = changed[ROCK] = changed[NEW_WATER] = True
    for name, before in baseline.items():
        after = np.asarray(fields[name])
        assert after.dtype == np.asarray(before).dtype, name
        np.testing.assert_array_equal(after[..., ~changed],
                                      np.asarray(before)[..., ~changed],
                                      err_msg=name)
    assert fields["TMN"][GLACIER] != 281.25
    assert "TMN" in receipt["fields_replaced"]
    assert "smooth-desmooth" not in receipt["override_audit"]["method"]
    assert "terrain" not in receipt["override_audit"]
    assert receipt["landcover"]["crosswalk_target"] == (
        "MODIFIED_IGBP_MODIS_NOAH, 21 categories")


def test_nlcd_open_water_joined_to_the_sea_is_not_a_lake(tmp_path,
                                                         monkeypatch):
    """Annual NLCD has one open-water class: the strip the 30-arc-second
    coast calls land and NLCD calls water is the sea, not a lake."""
    coast, inland = (6, 2), (6, 8)
    _stub(monkeypatch, {}, placements=((coast, 21), (inland, 21)))
    baseline = _baseline(lakes=())
    for row in range(N):
        for col in (0, 1):
            baseline["LANDUSEF"][:, row, col] = 0.0
            baseline["LANDUSEF"][16, row, col] = 1.0
            baseline["LANDMASK"][row, col] = 0.0
            baseline["LU_INDEX"][row, col] = 17.0
    grid = LambertGrid(ref_lat=40.0, ref_lon=-100.0, truelat1=30.0,
                       truelat2=60.0, stand_lon=-100.0, dx=1000.0, dy=1000.0,
                       e_we=N + 1, e_sn=N + 1)
    fields, receipt = apply_highres_statics(
        baseline, grid, config=_lakes(root=tmp_path,
                                      landcover_source="annual-nlcd"),
        domain_id=1, case_date=date(2020, 6, 1), landuse_attrs=MODIS21_ATTRS)
    assert fields["LU_INDEX"][coast] == 10.0
    assert fields["LU_INDEX"][inland] == 21.0
    check = receipt["lake_check"]
    assert check["promoted_cells"] == 1
    assert check["sea_connected_water_not_promoted_cells"] == 1


def test_lakes_reads_no_terrain_source(tmp_path, monkeypatch):
    """A pinned terrain source the domain leaves (3DEP over the Alps) is
    not resolved, refused or routed: terrain is the baseline."""
    from woof.static.highres_production import _resolve_plan
    from woof.static.highres_fetch import domain_footprint
    _stub(monkeypatch, {})
    config = _lakes(root=tmp_path, terrain_source="usgs-3dep-13as")
    assert _resolve_plan(config, domain_footprint(_grid(), HALO)) == (
        "lakes", None)
    fields, receipt = apply_highres_statics(
        _baseline(), _grid(), config=config, domain_id=2,
        case_date=date(2026, 11, 9), landuse_attrs=MODIS21_ATTRS)
    assert receipt["status"] == "APPLIED"
    assert "terrain_source_selection" not in receipt


def test_lakes_admits_no_land_cover_token_and_carries_no_selection(tmp_path):
    from woof.static.build import _highres_landcover_on_slot
    from woof.static.terrain_smoothing import selection_carrier
    config = _lakes(root=tmp_path)
    assert _highres_landcover_on_slot(config, None, 0,
                                      "cglc_modis_lcz+default") is None
    assert selection_carrier(config) is None
    full = HighresStaticConfig(enabled=True, cache_root=tmp_path,
                               fields="all")
    assert selection_carrier(full) is full


def test_lcz_pervious_split_reaches_the_lakes_build(tmp_path, monkeypatch):
    seen = {}
    _stub(monkeypatch, seen)
    plain = highres_module.resample_mapped_categories

    def split(source, grid, mapping, *, category_count):
        seen["split"] = True
        return plain(source, grid, mapping, category_count=category_count)

    monkeypatch.setattr(highres_module, "resample_lcz_pervious_split", split)
    _, receipt = apply_highres_statics(
        _baseline(), _grid(),
        config=_lakes(root=tmp_path, lcz_pervious="split"), domain_id=2,
        case_date=date(2026, 11, 9), landuse_attrs=MODIS21_ATTRS)
    assert seen.get("split") is True
    assert receipt["override_audit"]["lcz_pervious"]["rule"] == "split"


def test_a_footprint_outside_the_raster_keeps_its_lakes(tmp_path,
                                                        monkeypatch):
    """The real land-cover leg (not stubbed): a footprint wholly outside
    the published raster keeps the baseline, as fields = "all" does."""
    from woof.static import highres_production as production
    from woof.static.highres_fetch import CoverageError

    class _Raster:
        def receipt(self):
            return {"path": "stub"}

    monkeypatch.setattr(production, "fetch_landcover",
                        lambda source, year, root, urlopen=None:
                        ([], _Raster()))

    def outside(raster, bbox, root):
        raise CoverageError("footprint outside the stub raster")

    monkeypatch.setattr(production, "derive_landcover_window", outside)
    baseline = _baseline()
    fields, receipt = apply_highres_statics(
        baseline, _grid(), config=_lakes(root=tmp_path), domain_id=2,
        case_date=date(2026, 11, 9), landuse_attrs=MODIS21_ATTRS)
    assert receipt["status"] == "APPLIED"
    assert receipt["cells_replaced"]["total"] == 0
    assert "outside the published land-cover raster" in (
        receipt["unchanged_reason"])
    assert receipt["fetch"]["landcover_window_outside_raster"] == (
        "footprint outside the stub raster")
    for name in ("LU_INDEX", "LANDMASK", "LANDUSEF"):
        np.testing.assert_array_equal(fields[name], baseline[name])


def test_lakes_messages_name_the_land_cover_not_terrain(tmp_path):
    from woof.static.highres_production import (_refusal_remedy,
                                                 _scoped_plan_line)
    config = _lakes(root=tmp_path, terrain_source="copernicus-dem-glo30",
                    max_dx_m=1500.0)
    line = _scoped_plan_line(config, _grid(), 2)
    assert "lake extent comes from the cglc-modis-lcz" in line
    assert "terrain comes from" not in line and "tile(s)" not in line
    remedy = _refusal_remedy(config)
    assert "land-cover source" in remedy
    assert "declared terrain source" not in remedy


def test_a_lake_outflow_to_the_sea_does_not_hold_its_shore():
    """Review (#587): new shore water reaching the sea only through a
    baseline lake (and its outflow) is the lake's, not the sea's."""
    from woof.static.highres_production import _water_reaching_sea
    ocean = np.zeros((3, 6), dtype=bool)
    ocean[:, 0] = True
    lake = np.zeros_like(ocean)
    lake[:, 2:4] = True
    water = lake.copy()
    water[1, 1] = True   # the outflow channel, new water joining the sea
    water[1, 4] = True   # new shore water of the lake
    reach = _water_reaching_sea(water, ocean, lake)
    assert reach[1, 1] and not reach[1, 4]


def test_lake_depth_donor_is_the_euclidean_nearest_within_reach():
    """Review (#587): a promoted cell with kept lakes at (0, 6) and (3, 3)
    cells takes the (3, 3) one, 4.2 cells off, inside a 5-cell reach."""
    from types import SimpleNamespace
    from woof.static import highres_production as hp
    shape = (12, 12)
    lake = np.zeros(shape, dtype=bool)
    lake[2, 8] = lake[5, 5] = True
    depth = np.full(shape, 10.0)
    depth[2, 8], depth[5, 5] = 30.0, 70.0
    promote = np.zeros(shape, dtype=bool)
    promote[2, 2] = True
    dx = hp._LAKE_DEPTH_REACH_M / 5.0
    out, record = hp._lake_depth_for(
        {"LAKE_DEPTH": depth}, SimpleNamespace(dx=dx, dy=dx), lake=lake,
        demote=np.zeros(shape, dtype=bool), promote=promote)
    assert out[2, 2] == 70.0 and record["promoted_from_nearest_lake_cells"] == 1
