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


def _stub(monkeypatch, seen):
    def categories(source, grid, mapping, *, category_count):
        ny, nx = grid.e_sn - 1, grid.e_we - 1
        halo = (ny - N) // 2
        fractions = np.zeros((category_count, ny, nx))
        fractions[9] = 1.0
        for (row, col), category in ((GLACIER, 15), (ROCK, 16), (REAL, 21),
                                     (NEW_WATER, 21), (SEA_TO_LAKE, 21),
                                     (LAKE_TO_SEA, 17)):
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
