"""GEM-15 through WRF's ``[static.highres]``: selection, experimental
warning, GLO-30 fallback, receipts, and an end-to-end terrain build.

Offline: the end-to-end build reads a synthetic BigTIFF COG served by the
local bucket of ``test_gem15_range_reader`` (no real GEM-15 bytes); the
fallback is exercised at plan level so no GLO-30 tile is fetched.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from test_gem15_range_reader import bucket  # noqa: F401  (fixture)
from woof.static import gem15
from woof.static import highres_fetch as hf
from woof.static import highres_production as hp
from woof.static.highres_fetch import FootprintBBox
from woof.static.lambert import LambertGrid

MODIS21_ATTRS = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
                 "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}


def _grid(lat, lon, dx=1000.0, n=11):
    return LambertGrid(ref_lat=lat, ref_lon=lon, truelat1=30.0, truelat2=60.0,
                       stand_lon=lon, dx=dx, dy=dx, e_we=n, e_sn=n)


def _baseline(n, height=500.0):
    return {"HGT_M": np.full((n, n), height),
            "LU_INDEX": np.full((n, n), 10.0),
            "LANDMASK": np.full((n, n), 1.0),
            "SOILTEMP": np.full((n, n), 285.0)}


@pytest.mark.parametrize("source_id", ["gem15-dtm", "gem15-dsm"])
def test_named_choice_parses_and_auto_never_picks_it(tmp_path, source_id):
    config = hp.parse_static_table(
        {"highres": {"enabled": True, "cache_root": str(tmp_path),
                     "terrain_source": source_id, "fields": "terrain"}},
        source="case.toml", base_dir=tmp_path)
    assert config.terrain_source == source_id
    assert hf.terrain_source_coverage(source_id).experimental
    # auto, everywhere: never an experimental row.
    for lat, lon in ((40.5, -99.7), (46.5, 8.0), (-33.9, 18.4)):
        auto = hp.HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                      fields="terrain")
        _, coverage = hp._resolve_plan(
            auto, FootprintBBox(lat - .2, lat + .2, lon - .2, lon + .2))
        assert not coverage.experimental


def test_established_rows_route_unchanged():
    for row in hf.TERRAIN_SOURCES.values():
        coverage, selection = hp._experimental_terrain(
            row, FootprintBBox(40.0, 40.4, -100.0, -99.6))
        assert coverage is row and selection is None
    assert hp._with_selection({"a": 1}, None) == {"a": 1}


def test_gem15_where_it_serves_warns_and_records(capsys):
    row = hf.terrain_source_coverage("gem15-dtm")
    coverage, selection = hp._experimental_terrain(
        row, FootprintBBox(39.8, 40.2, -105.6, -105.2))
    assert coverage is row
    assert selection == {"requested": "gem15-dtm", "used": "gem15-dtm",
                         "experimental": True,
                         "warning": hf.GEM15_EXPERIMENTAL_WARNING}
    assert "EXPERIMENTAL" in capsys.readouterr().err


@pytest.mark.parametrize("place,lat,lon", [
    ("Moscow", 55.75, 37.6), ("New Delhi", 28.6, 77.2),
    ("Nuuk", 64.2, -51.7)])
def test_excluded_region_falls_back_to_glo30_with_warning(capsys, place, lat,
                                                          lon):
    row = hf.terrain_source_coverage("gem15-dsm")
    coverage, selection = hp._experimental_terrain(
        row, FootprintBBox(lat - .2, lat + .2, lon - .2, lon + .2),
        domain_id=2)
    assert coverage.source_id == "copernicus-dem-glo30"
    assert selection["requested"] == "gem15-dsm"
    assert selection["used"] == "copernicus-dem-glo30"
    assert selection["fallback_reason"]
    err = capsys.readouterr().err
    assert "d02: WARNING" in err and "copernicus-dem-glo30 instead" in err


def test_end_to_end_terrain_build_from_gem15(bucket, tmp_path, monkeypatch,  # noqa: F811
                                             capsys):
    """A 10 km domain inside the synthetic cell: HGT_M is the area mean of
    the 15 m source, the receipt names GEM-15, its attribution, datum and
    experimental status, and the default receipt keys are untouched."""
    monkeypatch.setattr(gem15, "load_index", lambda: bucket["index"]())
    grid = _grid(40.51, -99.74)
    config = hp.HighresStaticConfig(enabled=True, cache_root=tmp_path,
                                    fields="terrain",
                                    terrain_source="gem15-dtm")
    fields, receipt = hp.apply_highres_statics(
        _baseline(10), grid, config=config, domain_id=1,
        case_date=date(2025, 6, 1), landuse_attrs=MODIS21_ATTRS)
    hgt = np.asarray(fields["HGT_M"])
    # Synthetic heights are 1000 + 0.5 * row + 0.25 * col (m); the model
    # cells sit inside rows/cols 0..2047, so their means lie in that span.
    assert hgt.shape == (10, 10)
    assert np.all(hgt > 1000.0) and np.all(hgt < 1000.0 + 0.75 * 2048)
    assert np.ptp(hgt) > 10.0                     # real relief, not a fill
    assert receipt["status"] == "APPLIED"
    assert receipt["terrain_source"]["source_id"] == "gem15-dtm"
    assert receipt["terrain_source"]["experimental"] is True
    assert receipt["terrain_source"]["attribution"].startswith(
        "This dataset is produced by Google")
    assert receipt["terrain_source_selection"]["used"] == "gem15-dtm"
    fetch = receipt["fetch"]
    assert fetch["terrain_vertical_datum"].startswith("EGM96")
    assert fetch["gem15"]["gem15_cells"][0]["cell"] == "wm100s40"
    assert fetch["terrain_window_audit"]["vertical_datum"].startswith("EGM96")
    on_disk = json.loads(next((tmp_path / "receipts").glob(
        "*gem15-dtm*.json")).read_text())
    assert on_disk["terrain_source_selection"]["requested"] == "gem15-dtm"
    out = capsys.readouterr()
    assert "EXPERIMENTAL" in out.err and "APPLIED" in out.out
    # A second build is served from the cache: no new range request.
    requests = len(bucket["log"])
    hp.apply_highres_statics(
        _baseline(10), grid, config=config, domain_id=1,
        case_date=date(2025, 6, 1), landuse_attrs=MODIS21_ATTRS)
    assert len(bucket["log"]) == requests
