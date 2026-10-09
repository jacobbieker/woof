"""``woof energy fetch``: Overpass queries, tiling, cache and normalization.

Every test but the last runs offline against hand-written Overpass JSON in
``tests/fixtures/energy/overpass_*.json`` (synthetic, not OSM data) with the
module's one urlopen (``osm._open_url``) replaced.
"""

from __future__ import annotations

import email.message
import io
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs

import pytest

from woof.cli import build_parser
from woof.energy import osm
from woof.energy.contracts import load_assets

FIXTURES = Path(__file__).parent / "fixtures" / "energy"
BBOX = (-3.4, 51.6, -3.0, 51.8)


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _raw(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _http_error(url: str, code: int, body: bytes = b"") -> HTTPError:
    return HTTPError(url, code, f"HTTP {code}", email.message.Message(),
                     io.BytesIO(body))


class _Server:
    """Stands in for ``osm._open_url``: answers from a script of replies.

    Each reply is bytes (served with HTTP 200), an exception (raised), or
    a callable taking the query text.  The last reply repeats.
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, request, timeout):
        query = parse_qs(request.data.decode())["data"][0]
        self.calls.append((request.full_url, query))
        assert request.get_method() == "POST"
        assert request.get_header("User-agent").startswith("woof-energy-osm/")
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if callable(reply) and not isinstance(reply, BaseException):
            reply = reply(query)
        if isinstance(reply, BaseException):
            raise reply
        return _Response(reply)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv(osm.CACHE_ENV, str(root))
    monkeypatch.delenv(osm.BUDGET_ENV, raising=False)
    monkeypatch.setattr(osm, "MIN_REQUEST_GAP_S", 0.0)
    monkeypatch.setattr(osm._Budget, "pause", lambda self, seconds: None)
    return root


def _serve(monkeypatch, *replies) -> _Server:
    server = _Server(*replies)
    monkeypatch.setattr(osm, "_open_url", server)
    return server


def _by_id(collection):
    return {asset.asset_id: asset for asset in collection.assets}


# --------------------------------------------------------------------------
# tag parsing


@pytest.mark.parametrize("text,expected", [
    ("400000;132000", ((400.0, 132.0), True)),
    ("132000;400000", ((400.0, 132.0), True)),
    ("400000;400000", ((400.0,), True)),
    ("275 kV", ((275.0,), True)),
    ("230V", ((0.23,), True)),
    ("11000;medium", ((11.0,), False)),
    ("medium", ((), False)),
    ("0", ((), False)),
    ("", ((), False)),
    ("±500000", ((), False)),
    (None, ((), True)),
])
def test_parse_voltage(text, expected):
    assert osm.parse_voltage_kv(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("3 MW", 3.0), ("500 kW", 0.5), ("1.5GW", 1500.0), ("5 MWp", 5.0),
    ("2000 W", 0.002), ("yes", None), ("3", None), ("3,5 MW", None),
    ("0 MW", None), ("3 MWh", None), (None, None),
])
def test_parse_capacity(text, expected):
    got = osm.parse_capacity_mw(text)
    assert got == pytest.approx(expected) if expected else got is None


def test_parse_scalars():
    assert osm.parse_metres("80") == 80.0
    assert osm.parse_metres("80 m") == 80.0
    assert osm.parse_metres("262'") is None
    assert osm.parse_metres("80;90") is None
    assert osm.parse_positive_int("2") == 2
    assert osm.parse_positive_int("0") is None
    assert osm.parse_positive_int("1;2") is None
    assert osm.parse_frequency_hz("50") == 50.0
    assert osm.parse_frequency_hz("0") is None
    assert osm.parse_frequency_hz("50;16.7") is None
    assert osm.map_generator_source("Wind") == ("wind", True)
    assert osm.map_generator_source("diesel") == ("oil", False)
    assert osm.map_generator_source("osmotic") == ("other", False)
    assert osm.map_generator_source("gas;oil") == ("other", False)


# --------------------------------------------------------------------------
# normalization from fixtures


def test_normalize_fixture_assets(tmp_path):
    assets, counts = osm.normalize(
        [_fixture("overpass_wales_a.json")],
        ("line", "cable", "substation", "plant", "generator", "tower"))
    by_id = {a.asset_id: a for a in assets}
    assert len(by_id) == 13
    assert counts["centroid_fallbacks"] == 1
    assert counts["skipped"] == {}

    line = by_id["osm:way/2001"]
    assert line.kind == "line" and line.geometry["type"] == "LineString"
    assert line.voltage_kv == (400.0, 132.0)
    assert (line.circuits, line.cables, line.frequency_hz) == (2, 6, 50.0)
    assert (line.name, line.operator) == ("Synthetic 400 kV", "Synthetic Grid")
    assert line.source == "osm" and line.license == "ODbL-1.0"
    assert line.source_ref == "way/2001"
    assert line.tags == {"line": "busbar"}

    odd = by_id["osm:way/2005"]
    assert odd.voltage_kv == () and odd.circuits is None
    assert odd.frequency_hz is None
    assert odd.tags == {"voltage": "medium", "circuits": "1;2",
                        "frequency": "0"}

    turbine = by_id["osm:node/1003"]
    assert turbine.generator_source == "wind"
    assert turbine.capacity_mw == 3.0
    assert (turbine.hub_height_m, turbine.rotor_diameter_m) == (80.0, 90.0)
    assert "generator:source" not in turbine.tags

    diesel = by_id["osm:node/1004"]
    assert diesel.generator_source == "oil" and diesel.capacity_mw == 0.5
    assert diesel.tags["generator:source"] == "diesel"

    solar = by_id["osm:way/2007"]
    assert solar.geometry["type"] == "Polygon"
    assert solar.generator_source == "solar" and solar.capacity_mw is None
    assert solar.tags["generator:output:electricity"] == "yes"

    assert by_id["osm:way/2006"].geometry["type"] == "Polygon"
    assert by_id["osm:way/2006"].voltage_kv == (400.0, 132.0)
    assert by_id["osm:node/1001"].geometry["type"] == "Point"
    assert by_id["osm:node/1002"].voltage_kv == (33.0,)

    plant = by_id["osm:relation/3001"]
    assert plant.geometry["type"] == "MultiPolygon"
    (polygon,) = plant.geometry["coordinates"]
    assert len(polygon) == 2                      # outer + one hole
    assert plant.generator_source == "other" and plant.capacity_mw == 20.0
    assert plant.tags["plant:source"] == "wind;solar"
    assert "type" not in plant.tags

    fallback = by_id["osm:relation/3002"]
    assert fallback.geometry["type"] == "Point"
    assert fallback.tags[osm.GEOMETRY_TAG] == osm.CENTROID_FALLBACK

    # Everything survives the contract round trip.
    from woof.energy.contracts import AssetCollection, dump_assets
    path = dump_assets(AssetCollection(assets=assets,
                                       sources=[{"source": "osm"}]),
                       tmp_path / "a.geojson")
    assert len(load_assets(path)) == 13


def test_normalize_keeps_only_requested_kinds():
    assets, _ = osm.normalize([_fixture("overpass_wales_a.json")], ("tower",))
    assert [a.asset_id for a in assets] == ["osm:node/1001"]


def test_normalize_dedupes_across_tiles():
    assets, _ = osm.normalize(
        [_fixture("overpass_wales_a.json"), _fixture("overpass_wales_b.json")],
        ("line",))
    ids = [a.asset_id for a in assets]
    assert ids.count("osm:way/2001") == 1
    assert "osm:way/2101" in ids
    assert _by_id_list(assets)["osm:way/2101"].voltage_kv == (275.0,)


def _by_id_list(assets):
    return {a.asset_id: a for a in assets}


def test_min_voltage_filter_drops_and_counts():
    assets, counts = osm.normalize(
        [_fixture("overpass_wales_a.json")],
        ("line", "cable", "substation", "generator"), min_voltage_kv=100.0)
    ids = {a.asset_id for a in assets}
    assert "osm:way/2003" not in ids              # 11 kV line
    assert "osm:node/1002" not in ids             # 33 kV substation
    assert {"osm:way/2004", "osm:way/2005"} <= ids   # no parsable voltage
    assert {"osm:way/2001", "osm:way/2002", "osm:relation/3002"} <= ids
    assert "osm:node/1003" in ids                 # generators never filtered
    assert counts["dropped_below_min_voltage"] == 2
    assert counts["kept_without_voltage"] == 2


# --------------------------------------------------------------------------
# geometry assembly


def test_assemble_rings_joins_reversed_segments():
    a = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    b = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]       # runs the other way
    (ring,) = osm.assemble_rings([a, b])
    assert ring[0] == ring[-1] and len(ring) == 5


def test_assemble_rings_refuses_dangling_segment():
    assert osm.assemble_rings([[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]]) is None


def _member(role, coords):
    return {"type": "way", "role": role,
            "geometry": [{"lon": x, "lat": y} for x, y in coords]}


def test_multipolygon_two_outers_and_orientation():
    square = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    other = [(5, 5), (5, 6), (6, 6), (6, 5), (5, 5)]
    hole = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8), (0.2, 0.2)]
    geometry = osm.multipolygon_geometry([
        _member("outer", square), _member("outer", other),
        _member("inner", hole)])
    assert geometry["type"] == "MultiPolygon"
    first, second = geometry["coordinates"]
    assert len(first) == 2 and len(second) == 1
    assert osm._signed_area([tuple(p) for p in first[0]]) > 0     # CCW outer
    assert osm._signed_area([tuple(p) for p in first[1]]) < 0     # CW hole


def test_multipolygon_inner_outside_outer_falls_back():
    square = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    stray = [(3, 3), (4, 3), (4, 4), (3, 3)]
    assert osm.multipolygon_geometry([_member("outer", square),
                                      _member("inner", stray)]) is None
    element = {"type": "relation", "id": 9,
               "members": [_member("outer", square), _member("inner", stray)],
               "tags": {"type": "multipolygon", "power": "plant"}}
    asset, why = osm.element_to_asset(element)
    assert why == osm.CENTROID_FALLBACK
    assert asset.geometry["type"] == "Point"
    assert asset.tags[osm.GEOMETRY_TAG] == osm.CENTROID_FALLBACK


def test_multipolygon_member_without_geometry_falls_back_to_bounds():
    element = {"type": "relation", "id": 10,
               "bounds": {"minlat": 1.0, "minlon": 2.0, "maxlat": 3.0,
                          "maxlon": 4.0},
               "members": [{"type": "way", "role": "outer", "ref": 1}],
               "tags": {"type": "multipolygon", "power": "substation"}}
    asset, why = osm.element_to_asset(element)
    assert why == osm.CENTROID_FALLBACK
    assert asset.geometry["coordinates"] == [3.0, 2.0]


def test_inner_ring_touching_outer_is_assembled():
    square = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
    # The courtyard shares the outer vertex (0, 0).
    yard = [(0, 0), (1, 0.5), (1, 1), (0.5, 1), (0, 0)]
    geometry = osm.multipolygon_geometry([_member("outer", square),
                                          _member("inner", yard)])
    assert geometry is not None
    assert len(geometry["coordinates"][0]) == 2


def test_capacity_and_source_fall_back_to_the_other_key():
    element = {"type": "node", "id": 13, "lat": 51.0, "lon": -3.0,
               "tags": {"power": "plant", "plant:output:electricity": "yes",
                        "generator:output:electricity": "50 MW",
                        "plant:source": "yes", "generator:source": "wind"}}
    asset, _why = osm.element_to_asset(element)
    assert asset.capacity_mw == 50.0 and asset.generator_source == "wind"
    assert asset.tags["plant:output:electricity"] == "yes"
    assert asset.tags["plant:source"] == "yes"


def test_unclosed_substation_way_is_a_centroid_point():
    element = {"type": "way", "id": 11, "nodes": [1, 2, 3],
               "geometry": [{"lon": 0, "lat": 0}, {"lon": 1, "lat": 0},
                            {"lon": 1, "lat": 1}],
               "tags": {"power": "substation"}}
    asset, why = osm.element_to_asset(element)
    assert why == osm.CENTROID_FALLBACK and asset.geometry["type"] == "Point"


def test_tower_way_is_skipped_with_reason():
    element = {"type": "way", "id": 12, "nodes": [1, 2],
               "geometry": [{"lon": 0, "lat": 0}, {"lon": 1, "lat": 0}],
               "tags": {"power": "tower"}}
    asset, why = osm.element_to_asset(element)
    assert asset is None and "not a node" in why


# --------------------------------------------------------------------------
# tiling and query text


def test_tiles_are_deterministic_and_grid_aligned():
    area = osm.Area(bbox=(-4.5, 50.2, -2.0, 52.0))
    tiles = osm.tiles_for(area)
    assert tiles == osm.tiles_for(osm.Area(bbox=(-4.5, 50.2, -2.0, 52.0)))
    assert len(tiles) == 6
    assert tiles[0].bbox == (-4.5, 50.2, -4.0, 51.0)
    assert tiles[-1].bbox == (-3.0, 51.0, -2.0, 52.0)
    for tile in tiles:
        assert tile.east - tile.west <= 1.0 and tile.north - tile.south <= 1.0


def test_interior_tile_query_is_shared_between_areas():
    kinds = ("line", "substation")
    small = {osm.query_key(osm.query_body(t, kinds))
             for t in osm.tiles_for(osm.Area(bbox=(-4.5, 50.5, -2.5, 52.5)))}
    large = {osm.query_key(osm.query_body(t, kinds))
             for t in osm.tiles_for(osm.Area(bbox=(-5.0, 50.0, -2.2, 52.2)))}
    interior = osm.query_key(osm.query_body(osm.Tile(-4.0, 51.0, -3.0, 52.0),
                                            kinds))
    assert interior in small and interior in large


def test_query_text_order_and_key_independence():
    tile = osm.Tile(-3.4, 51.6, -3.0, 51.8)
    body = osm.query_body(tile, ("line", "substation", "tower"))
    assert "(51.6,-3.4,51.8,-3)" in body           # south,west,north,east
    assert 'relation["type"="multipolygon"]' in body
    assert body.rstrip().endswith("out body geom;")
    assert osm.query_text(body, 180).startswith("[out:json][timeout:180];")
    assert osm.query_key(body) == osm.query_key(body)
    assert osm.query_key(body) != osm.query_key(
        osm.query_body(tile, ("line",)))


def test_tile_quadrants():
    quads = osm.Tile(-4.0, 51.0, -3.0, 52.0).quadrants()
    assert [q.bbox for q in quads] == [(-4.0, 51.0, -3.5, 51.5),
                                       (-3.5, 51.0, -3.0, 51.5),
                                       (-4.0, 51.5, -3.5, 52.0),
                                       (-3.5, 51.5, -3.0, 52.0)]
    assert all(q.depth == 1 for q in quads)


def _write_polygon(path: Path, ring, holes=()):
    path.write_text(json.dumps({"type": "Feature", "properties": {},
                                "geometry": {"type": "Polygon",
                                             "coordinates": [ring, *holes]}}))
    return path


def test_polygon_area_tiles_skip_outside_cells(tmp_path):
    # An L-shape: the north-east cell of its 2x2 bbox is outside.
    ring = [[-4, 51], [-2, 51], [-2, 51.5], [-3.5, 51.5], [-3.5, 53],
            [-4, 53], [-4, 51]]
    area = osm.load_polygon_area(_write_polygon(tmp_path / "l.geojson", ring))
    boxes = [t.bbox for t in osm.tiles_for(area)]
    assert (-3.0, 52.0, -2.0, 53.0) not in boxes
    assert len(boxes) == 3
    body = osm.query_body(osm.tiles_for(area)[0], ("line",), area.polygons)
    assert '(poly:"51 -4 51 -2 51.5 -2 51.5 -3.5 53 -3.5 53 -4")(51,-4,52,-3)' in body


def test_tile_inside_polygon_is_the_bbox_query(tmp_path):
    ring = [[-5.5, 50.5], [-1.5, 50.5], [-1.5, 53.5], [-5.5, 53.5],
            [-5.5, 50.5]]
    area = osm.load_polygon_area(_write_polygon(tmp_path / "big.geojson",
                                                ring))
    inner = osm.Tile(-4.0, 51.0, -3.0, 52.0)
    assert inner in osm.tiles_for(area)
    kinds = ("line",)
    assert osm.query_body(inner, kinds, area.polygons) == \
        osm.query_body(inner, kinds)
    edge = osm.tiles_for(area)[0]
    assert "poly:" in osm.query_body(edge, kinds, area.polygons)


def test_polygon_simplified_to_vertex_limit(tmp_path):
    import math
    ring = [[-3.0 + math.cos(2 * math.pi * i / 5000) * (1 + 0.01 * (i % 7)),
             52.0 + math.sin(2 * math.pi * i / 5000)] for i in range(5000)]
    ring.append(ring[0])
    hole = [[-3.1, 51.9], [-2.9, 51.9], [-2.9, 52.1], [-3.1, 51.9]]
    area = osm.load_polygon_area(_write_polygon(tmp_path / "c.geojson", ring,
                                                [hole]))
    assert sum(len(r) for r in area.polygons) <= osm.MAX_POLYGON_VERTICES
    assert any("simplified from 5000" in n for n in area.notes)
    assert any("inner ring" in n for n in area.notes)


def test_bad_polygon_file_refused(tmp_path):
    bad = tmp_path / "bad.geojson"
    bad.write_text(json.dumps({"type": "LineString",
                               "coordinates": [[0, 0], [1, 1]]}))
    with pytest.raises(osm.OsmAreaError, match="Polygon"):
        osm.load_polygon_area(bad)
    with pytest.raises(osm.OsmAreaError, match="does not exist"):
        osm.load_polygon_area(tmp_path / "missing.geojson")


def test_area_arguments_refused(cache):
    with pytest.raises(osm.OsmAreaError, match="exactly one"):
        osm.fetch_assets()
    with pytest.raises(osm.OsmAreaError, match="contradict"):
        osm.fetch_assets(bbox=BBOX, refresh=True, offline=True)
    with pytest.raises(osm.OsmAreaError, match="unknown kinds"):
        osm.fetch_assets(bbox=BBOX, kinds=("pylon",), offline=True)
    with pytest.raises(osm.OsmAreaError, match="http"):
        osm.fetch_assets(bbox=BBOX, endpoint="ftp://example.org/x")


def test_endpoint_ladder_puts_override_first():
    ladder = osm.endpoint_ladder("https://example.org/api/interpreter")
    assert ladder[0] == "https://example.org/api/interpreter"
    assert ladder[1:] == osm.ENDPOINTS
    assert osm.endpoint_ladder(osm.ENDPOINTS[1])[:2] == (osm.ENDPOINTS[1],
                                                        osm.ENDPOINTS[0])


# --------------------------------------------------------------------------
# cache and network layer


def test_fetch_then_cache_hit_then_refresh(cache, monkeypatch):
    server = _serve(monkeypatch, _raw("overpass_wales_a.json"))
    first = osm.fetch_assets(bbox=BBOX)
    assert len(server.calls) == 1
    url, query = server.calls[0]
    assert url == osm.ENDPOINTS[0]
    assert query.startswith("[out:json][timeout:180];")

    (meta_path,) = cache.glob("tiles/*/*.meta.json")
    meta = json.loads(meta_path.read_text())
    assert meta["status"] == "complete"
    assert meta["endpoint"] == osm.ENDPOINTS[0]
    assert meta["osm_base"] == "2026-10-01T12:00:00Z"
    assert meta["retrieved_utc"].endswith("Z")
    assert meta_path.name == f"{meta['key']}.meta.json"

    record = first.sources[0]
    assert record["source"] == "osm" and record["license"] == "ODbL-1.0"
    assert record["attribution"] == osm.ATTRIBUTION
    assert record["endpoints"] == [osm.ENDPOINTS[0]]
    assert record["osm_base"] == "2026-10-01T12:00:00Z"
    assert record["query"]["tiles"] == [meta["key"]]

    second = osm.fetch_assets(bbox=BBOX, timeout_s=60.0)   # timeout not keyed
    assert len(server.calls) == 1
    assert [a.asset_id for a in second.assets] == \
        [a.asset_id for a in first.assets]

    offline = osm.fetch_assets(bbox=BBOX, offline=True)
    assert len(offline) == len(first) and len(server.calls) == 1

    osm.fetch_assets(bbox=BBOX, refresh=True)
    assert len(server.calls) == 2


def test_offline_refuses_and_lists_missing_tiles(cache, monkeypatch):
    server = _serve(monkeypatch, AssertionError("offline must not ask"))
    with pytest.raises(osm.OverpassCacheMiss) as caught:
        osm.fetch_assets(bbox=(-4.5, 51.2, -3.5, 51.8), offline=True)
    assert len(caught.value.missing) == 2
    assert "-4.5,51.2,-4,51.8" in str(caught.value)
    assert "-4,51.2,-3.5,51.8" in str(caught.value)
    assert server.calls == []


def test_offline_refuses_partly_cached_area(cache, monkeypatch):
    _serve(monkeypatch, _raw("overpass_wales_a.json"))
    osm.fetch_assets(bbox=(-3.4, 51.6, -3.0, 51.8))
    with pytest.raises(osm.OverpassCacheMiss) as caught:
        osm.fetch_assets(bbox=(-3.4, 51.6, -2.5, 51.8), offline=True)
    # The western cell is the same query and is reused; only the new
    # eastern cell is missing.
    assert [bbox for bbox, _key in caught.value.missing] == \
        [(-3.0, 51.6, -2.5, 51.8)]


def test_corrupt_cache_body_is_not_a_hit(cache, monkeypatch):
    server = _serve(monkeypatch, _raw("overpass_wales_a.json"))
    osm.fetch_assets(bbox=BBOX)
    (body,) = [p for p in cache.glob("tiles/*/*.json")
               if not p.name.endswith(".meta.json")]
    body.write_text("{truncated")
    with pytest.raises(osm.OverpassCacheMiss):
        osm.fetch_assets(bbox=BBOX, offline=True)
    osm.fetch_assets(bbox=BBOX)
    assert len(server.calls) == 2


def test_failing_endpoint_falls_through_to_next(cache, monkeypatch):
    server = _serve(monkeypatch,
                    _http_error(osm.ENDPOINTS[0], 504),
                    _raw("overpass_wales_a.json"))
    collection = osm.fetch_assets(bbox=BBOX)
    assert [url for url, _ in server.calls] == list(osm.ENDPOINTS)
    assert collection.sources[0]["endpoints"] == [osm.ENDPOINTS[1]]


def test_unexpected_status_asks_next_endpoint(cache, monkeypatch):
    server = _serve(monkeypatch, _http_error(osm.ENDPOINTS[0], 405),
                    _raw("overpass_wales_a.json"))
    collection = osm.fetch_assets(bbox=BBOX)
    assert len(server.calls) == 2
    assert collection.sources[0]["endpoints"] == [osm.ENDPOINTS[1]]


def test_slow_answer_is_cut_off():
    class Trickle:
        def __init__(self):
            self.parts = [b"a", b"b", b""]

        def read(self, size):
            return self.parts.pop(0)

    with pytest.raises(TimeoutError, match="longer than"):
        osm._read_within(Trickle(), -1.0, osm._Budget(60.0))


def test_busy_page_is_retried(cache, monkeypatch):
    busy = (b"<html><body><p><strong>Error</strong>: runtime error: open64: "
            b"0 Success /osm3s_osm_base Dispatcher_Client::request_read_and_"
            b"idx::rate_limited. Please check /api/status</p></body></html>")
    server = _serve(monkeypatch, busy, busy, _raw("overpass_wales_a.json"))
    collection = osm.fetch_assets(bbox=BBOX)
    assert len(server.calls) == 3 and len(collection) > 0


def test_every_endpoint_failing_raises_unavailable(cache, monkeypatch):
    server = _serve(monkeypatch, _http_error(osm.ENDPOINTS[0], 429))
    with pytest.raises(osm.OverpassUnavailable, match="every endpoint"):
        osm.fetch_assets(bbox=BBOX)
    assert len(server.calls) == 2 * 5            # two hosts, five rounds
    assert not list(cache.glob("tiles/*/*.meta.json"))


def test_html_error_page_is_not_cached(cache, monkeypatch):
    _serve(monkeypatch, b"<html>502 Bad Gateway</html>")
    with pytest.raises(osm.OverpassUnavailable, match="not JSON"):
        osm.fetch_assets(bbox=BBOX)
    assert not list(cache.glob("tiles/*/*"))


def test_wall_clock_budget_refusal(cache, monkeypatch):
    monkeypatch.setenv(osm.BUDGET_ENV, "0.000001")
    _serve(monkeypatch, _raw("overpass_wales_a.json"))
    with pytest.raises(osm.OverpassUnavailable, match="wall-clock budget"):
        osm.fetch_assets(bbox=BBOX)


def test_bad_request_is_a_query_error(cache, monkeypatch):
    _serve(monkeypatch, _http_error(
        osm.ENDPOINTS[0], 400,
        b"<p><strong>Error</strong>: line 3: parse error</p>"))
    with pytest.raises(osm.OverpassQueryError, match="parse error"):
        osm.fetch_assets(bbox=BBOX)


def test_timed_out_tile_is_split_and_split_is_remembered(cache, monkeypatch):
    tile = (-4.0, 51.0, -3.0, 52.0)
    server = _serve(monkeypatch, _raw("overpass_timeout.json"),
                    _raw("overpass_wales_b.json"))
    collection, report = osm.fetch_with_report(bbox=tile)
    assert len(server.calls) == 5                 # parent + four quadrants
    assert report.splits == 1 and len(report.tiles) == 4
    assert {t.tile.depth for t in report.tiles} == {1}
    assert any("split" in note for note in collection.sources[0]["notes"])
    # Offline walks the remembered split.
    again, report = osm.fetch_with_report(bbox=tile, offline=True)
    assert len(server.calls) == 5 and report.tiles_cached == 4
    assert [a.asset_id for a in again.assets] == \
        [a.asset_id for a in collection.assets]
    # A plain rerun follows the split; --refresh asks for the whole tile.
    osm.fetch_with_report(bbox=tile)
    assert len(server.calls) == 5
    _collection, report = osm.fetch_with_report(bbox=tile, refresh=True)
    assert len(server.calls) == 6 and report.splits == 0


def test_longer_timeout_retries_a_split_tile(cache, monkeypatch):
    tile = (-4.0, 51.0, -3.0, 52.0)
    server = _serve(monkeypatch, _raw("overpass_timeout.json"),
                    _raw("overpass_wales_b.json"))
    osm.fetch_with_report(bbox=tile, timeout_s=60.0)
    assert len(server.calls) == 5
    _collection, report = osm.fetch_with_report(bbox=tile, timeout_s=600.0)
    assert len(server.calls) == 6 and report.splits == 0


def test_narrow_tall_tile_is_split_not_refused(cache, monkeypatch):
    server = _serve(monkeypatch, _raw("overpass_timeout.json"),
                    _raw("overpass_wales_b.json"))
    _collection, report = osm.fetch_with_report(bbox=(-3.4, 51.0, -3.3, 52.0))
    assert report.splits == 1 and len(server.calls) == 5


def test_split_polygon_tile_drops_quadrants_outside(cache, monkeypatch,
                                                     tmp_path):
    # An L along the south and west edges: the NE quadrant misses it.
    ring = [[-4.0, 51.0], [-3.5, 51.0], [-3.5, 51.1], [-3.9, 51.1],
            [-3.9, 51.5], [-4.0, 51.5], [-4.0, 51.0]]
    polygon = _write_polygon(tmp_path / "tri.geojson", ring)
    server = _serve(monkeypatch, _raw("overpass_timeout.json"),
                    _raw("overpass_wales_b.json"))
    _collection, report = osm.fetch_with_report(polygon=polygon)
    assert report.splits == 1
    assert len(report.tiles) == 3 and len(server.calls) == 4
    assert (-3.75, 51.25, -3.5, 51.5) not in \
        [r.tile.bbox for r in report.tiles]


def test_tile_too_large_at_smallest_split(cache, monkeypatch):
    monkeypatch.setattr(osm, "MIN_TILE_DEG", 0.5)
    _serve(monkeypatch, _raw("overpass_timeout.json"))
    with pytest.raises(osm.OverpassTileTooLarge, match="smallest split"):
        osm.fetch_assets(bbox=(-4.0, 51.0, -3.0, 52.0))


def test_fetch_min_voltage_records_note(cache, monkeypatch):
    _serve(monkeypatch, _raw("overpass_wales_a.json"))
    collection, report = osm.fetch_with_report(bbox=BBOX, min_voltage_kv=100,
                                               kinds=("line", "substation"))
    assert report.dropped_below_min_voltage == 2
    assert any("min_voltage_kv=100" in n for n in collection.sources[0]["notes"])


# --------------------------------------------------------------------------
# the command


def _args(*extra):
    return build_parser().parse_args(["energy", "fetch", *extra])


def test_main_writes_assets_and_summary(cache, monkeypatch, tmp_path, capsys):
    _serve(monkeypatch, _raw("overpass_wales_a.json"))
    output = tmp_path / "assets.geojson"
    args = _args("--bbox=-3.4,51.6,-3.0,51.8", "--kinds",
                 "line,cable,substation,plant,generator,tower",
                 "-o", str(output))
    assert args.func(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema"] == osm.SUMMARY_SCHEMA
    assert summary["by_kind"] == {"cable": 1, "generator": 3, "line": 4,
                                  "plant": 1, "substation": 3, "tower": 1}
    assert summary["tiles"] == {"total": 1, "fetched": 1, "cached": 0,
                                "split": 0}
    assert summary["centroid_fallbacks"] == 1
    collection = load_assets(output)
    assert len(collection) == summary["assets"] == 13
    assert collection.sources[0]["attribution"] == osm.ATTRIBUTION

    args = _args("--bbox=-3.4,51.6,-3.0,51.8", "--kinds",
                 "line,cable,substation,plant,generator,tower", "--offline",
                 "-o", str(output))
    assert args.func(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["tiles"]["cached"] == 1 and summary["tiles"]["fetched"] == 0


def test_main_offline_miss_returns_nonzero(cache, monkeypatch, tmp_path,
                                           capsys):
    _serve(monkeypatch, AssertionError("offline must not ask"))
    output = tmp_path / "assets.geojson"
    args = _args("--bbox=-3.4,51.6,-3.0,51.8", "--offline", "-o", str(output))
    assert args.func(args) == 2
    err = capsys.readouterr().err
    assert "woof energy fetch: --offline" in err and "-3.4,51.6,-3,51.8" in err
    assert not output.exists()


def test_main_polygon_area(cache, monkeypatch, tmp_path, capsys):
    server = _serve(monkeypatch, _raw("overpass_wales_a.json"))
    ring = [[-3.4, 51.6], [-3.0, 51.6], [-3.0, 51.8], [-3.4, 51.8],
            [-3.4, 51.6]]
    polygon = _write_polygon(tmp_path / "area.geojson", ring)
    args = _args("--polygon", str(polygon), "-o", str(tmp_path / "a.geojson"))
    assert args.func(args) == 0
    assert '(poly:"' in server.calls[0][1]
    record = load_assets(tmp_path / "a.geojson").sources[0]
    assert record["query"]["area"]["polygon"]["rings"] == 1


# --------------------------------------------------------------------------
# live


@pytest.mark.network
def test_live_overpass_tiny_bbox(tmp_path, monkeypatch):
    if os.environ.get("WOOF_NETWORK_TESTS") != "1":
        pytest.skip("live Overpass smoke needs WOOF_NETWORK_TESTS=1")
    monkeypatch.setenv(osm.CACHE_ENV, str(tmp_path / "cache"))
    collection = osm.fetch_assets(bbox=(-3.30, 51.62, -3.25, 51.66),
                                  kinds=("line", "substation"),
                                  timeout_s=60.0)
    assert collection.sources[0]["osm_base"]
    for asset in collection.assets:
        assert asset.asset_id.startswith("osm:")
    again = osm.fetch_assets(bbox=(-3.30, 51.62, -3.25, 51.66),
                             kinds=("line", "substation"), offline=True)
    assert len(again) == len(collection)
