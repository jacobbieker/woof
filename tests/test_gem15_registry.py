"""GEM-15 registry rows, pinned cell index, licence and fallback policy.

Offline and deterministic: everything here reads the pinned snapshot and
region outlines shipped in ``woof/data``; nothing touches the network.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from woof.static import gem15
from woof.static import highres_fetch as hf
from woof.static.highres_fetch import FootprintBBox

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_ATTRIBUTION = "This dataset is produced by Google"


def _box(lat, lon, half=0.2):
    return FootprintBBox(lat - half, lat + half, lon - half, lon + half)


@pytest.mark.parametrize("source_id", ["gem15-dtm", "gem15-dsm"])
def test_rows_are_experimental_cc_by_and_attributed(source_id):
    row = hf.EXPERIMENTAL_TERRAIN_SOURCES[source_id]
    assert row.source_id == source_id and row.role == "terrain"
    assert row.experimental is True
    assert (row.license_id, row.license_url) == (
        "CC-BY-4.0", "https://creativecommons.org/licenses/by/4.0/")
    assert row.attribution.startswith(REQUIRED_ATTRIBUTION + ".")
    assert "10.31223/X55N75" in row.attribution
    assert (row.envelope.lat_min, row.envelope.lat_max) == (-61.0, 84.0)
    echo = row.echo()
    assert echo["experimental"] is True
    assert echo["attribution"] == row.attribution
    assert gem15.source_row(source_id) is row


def test_rows_are_named_only_and_established_echoes_unchanged():
    # Never in the established registry, so ``auto`` and the default rows
    # cannot reach them; a named lookup finds them.
    assert not set(hf.EXPERIMENTAL_TERRAIN_SOURCES) & set(hf.TERRAIN_SOURCES)
    for row in hf.TERRAIN_SOURCES.values():
        assert row.experimental is False
        assert "experimental" not in row.echo()
    assert hf.terrain_source_coverage("gem15-dtm").experimental
    with pytest.raises(hf.CoverageError, match="gem15-dsm"):
        hf.terrain_source_coverage("gem15-xyz")


def test_datum_and_lattice_constants():
    assert hf.GEM15_VERTICAL_DATUM == hf.SRTM_GL1_VERTICAL_DATUM
    assert hf.GEM15_VERTICAL_DATUM.startswith("EGM96")
    assert hf.GEM15_PIXEL_M == 15.0
    # The lattice is no coarser than the 15 m pixel in latitude.
    assert hf.GEM15_LAT_STEP_DEG * 111_320.0 < 15.5
    assert "EXPERIMENTAL" in hf.GEM15_EXPERIMENTAL_WARNING


def test_pinned_index_is_complete_and_self_consistent():
    index = gem15.load_index()
    assert len(index.cells) == 13485
    for (lat, lon), cell in index.cells.items():
        assert (cell.lat_sw, cell.lon_sw) == (lat, lon)
        assert -61 <= lat < 84 and -180 <= lon < 180
        # UTM: north 326zz / south 327zz, zone near the cell's longitude.
        assert cell.epsg // 100 in (326, 327)
        assert (cell.epsg // 100 == 327) == (lat < 0)
        zone = cell.epsg % 100
        assert 1 <= zone <= 60
        centre_zone = int((lon + 0.5 + 180.0) // 6.0) + 1
        assert abs(zone - centre_zone) <= 1 or abs(zone - centre_zone) == 59
        for source_id in gem15.ASSETS:
            obj = cell.object(source_id)
            asset = gem15.ASSETS[source_id][0]
            assert obj.url == (f"{hf.GEM15_BUCKET_URL}/{cell.group}/"
                               f"{cell.cell_id}/{asset}/zall/mosaic.tif")
            assert obj.bytes > 0 and len(obj.crc32c) == 8
            assert obj.md5 is None or len(obj.md5) == 32
            assert obj.transform[0] == 15.0 and obj.transform[4] == -15.0
            assert obj.shape[0] % 512 == 0 and obj.shape[1] % 512 == 0


def test_index_names_the_cell_it_is_asked_for():
    index = gem15.load_index()
    cell = index.cell(40, -100)
    assert cell.cell_id == "wm100s40" and cell.group == "p40_m100"
    assert cell.epsg == 32614
    published, unpublished = index.cells_over(
        FootprintBBox(40.2, 41.6, -99.8, -98.5))
    assert [c.cell_id for c in published] == [
        "wm100s40", "wm99s40", "wm100s41", "wm99s41"]
    assert unpublished == ()


def test_index_is_regenerable_bytes():
    raw = (ROOT / "woof" / "data" / "gem15_index.v1.json.gz").read_bytes()
    payload = json.loads(gzip.decompress(raw))
    assert payload["schema"] == gem15.INDEX_SCHEMA
    # Written by tools/gem15_index.py as sorted, compact JSON, so a
    # regeneration of the same snapshot is the same text.  (The gzip bytes
    # depend on the zlib build, so the text is what is compared.)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert text.encode("utf-8") == gzip.decompress(raw)


@pytest.mark.parametrize("place,lat,lon", [
    ("Denver", 39.7, -105.0), ("Alps", 46.5, 8.0), ("Cape Town", -33.9, 18.4),
    ("Tokyo", 35.7, 139.7), ("Almaty", 43.2, 76.9), ("Hanoi", 21.0, 105.8),
])
def test_gem15_serves(place, lat, lon):
    for source_id in gem15.ASSETS:
        assert gem15.fallback_reason(_box(lat, lon), source_id) is None, place


@pytest.mark.parametrize("place,lat,lon,named", [
    ("Moscow", 55.75, 37.6, "Russia"), ("Beijing", 39.9, 116.4, "China"),
    ("Ulaanbaatar", 47.9, 106.9, "Mongolia"), ("Nuuk", 64.2, -51.7,
                                               "Greenland"),
    ("New Delhi", 28.6, 77.2, "India"), ("Kathmandu", 27.7, 85.3, "India"),
    ("Chukotka over the dateline", 65.0, 179.8, "Russia"),
])
def test_excluded_and_offset_regions_fall_back(place, lat, lon, named):
    reason = gem15.fallback_reason(_box(lat, lon), "gem15-dtm")
    assert reason is not None and named in reason, place
    assert gem15.FALLBACK_SOURCE == "copernicus-dem-glo30"


def test_outside_envelope_and_unpublished_fall_back():
    antarctic = gem15.fallback_reason(_box(-77.8, 166.7), "gem15-dsm")
    assert "-61..84" in antarctic
    straddle = gem15.fallback_reason(
        FootprintBBox(-61.4, -60.6, -46.0, -45.0), "gem15-dtm")
    assert "-61..84" in straddle
    ocean = gem15.fallback_reason(_box(0.5, -140.5), "gem15-dtm")
    assert "lists no cell" in ocean


def test_unknown_source_refused():
    with pytest.raises(KeyError):
        gem15.fallback_reason(_box(40, -100), "copernicus-dem-glo30")


def test_region_geometry_primitives():
    square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    assert gem15._box_meets_ring((2, 2, 3, 3), square)          # inside
    assert gem15._box_meets_ring((-5, -5, 20, 20), square)      # encloses
    assert gem15._box_meets_ring((9, 4, 12, 5), square)         # crosses edge
    assert gem15._box_meets_ring((-1, 4, 11, 5), square)        # cuts through
    assert not gem15._box_meets_ring((11, 11, 12, 12), square)  # apart


def test_regions_tool_is_deterministic(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gem15_index_tool", ROOT / "tools" / "gem15_index.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    ring = [[0.12345, 1.0], [2.0, 1.0], [2.0, 3.0], [0.12345, 1.0]]
    features = [{"properties": {"ADM0_A3": code},
                 "geometry": {"type": "Polygon", "coordinates": [ring]}}
                for code in tool.REGIONS]
    source = tmp_path / "countries.geojson"
    source.write_text(json.dumps({"features": features}))
    regions = tool.build_regions(source)
    assert regions["schema"] == gem15.REGIONS_SCHEMA
    assert regions["regions"]["IND"]["margin_deg"] == 1.0
    assert regions["regions"]["RUS"]["rings"][0][0] == [0.123, 1.0]
    shipped = gem15.load_regions()
    assert sorted(shipped) == sorted(tool.REGIONS)


def test_attribution_carried_in_notice_and_docs():
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert REQUIRED_ATTRIBUTION in notice and "GEM-15" in notice
    terrain_doc = (ROOT / "docs" / "public" / "HIGHRES-TERRAIN.md").read_text(
        encoding="utf-8")
    assert REQUIRED_ATTRIBUTION in terrain_doc
