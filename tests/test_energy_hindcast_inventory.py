"""Global wind-farm inventory: USWTDB, GEM, EMODnet and OSM-pbf importers,
the stdlib xlsx reader, status filtering and the cross-source merge.

Every fixture under ``tests/fixtures/energy/inventory`` is synthetic (see
its README); nothing here touches the network.
"""

from __future__ import annotations

import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import types
import zipfile

import pytest

from woof.cli import build_parser
from woof.energy import cli as energy_cli
from woof.energy import inventory_global as inv
from woof.energy.contracts import (
    ASSETS_SCHEMA,
    Asset,
    AssetCollection,
    dump_assets,
    load_assets,
    load_sites,
)
from woof.energy.importers import ImportRefused, import_assets

FIXTURES = Path(__file__).parent / "fixtures" / "energy"
INVENTORY = FIXTURES / "inventory"
REPD = FIXTURES / "import" / "repd_sample.csv"


# --------------------------------------------------------------------------
# helpers


def _xml_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _column(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def write_xlsx(path: Path, sheets: list[tuple[str, list[list[str]]]], *,
               inline_columns: tuple[int, ...] = ()) -> Path:
    """A minimal SpreadsheetML workbook: numbers as numbers, text in the
    shared-string table (or inline for ``inline_columns``)."""

    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = ("http://schemas.openxmlformats.org/officeDocument/2006/"
           "relationships")
    shared: list[str] = []
    index: dict[str, int] = {}
    parts = {}
    for number, (_, rows) in enumerate(sheets, start=1):
        xml_rows = []
        for r, row in enumerate(rows, start=1):
            cells = []
            for c, value in enumerate(row):
                if value == "":
                    continue
                ref = f"{_column(c)}{r}"
                try:
                    float(value)
                    numeric = r > 1
                except ValueError:
                    numeric = False
                if numeric:
                    cells.append(f'<c r="{ref}"><v>{value}</v></c>')
                elif c in inline_columns:
                    cells.append(f'<c r="{ref}" t="inlineStr"><is><t>'
                                 f'{_xml_escape(value)}</t></is></c>')
                else:
                    if value not in index:
                        index[value] = len(shared)
                        shared.append(value)
                    cells.append(f'<c r="{ref}" t="s"><v>{index[value]}'
                                 '</v></c>')
            xml_rows.append(f'<row r="{r}">{"".join(cells)}</row>')
        parts[f"xl/worksheets/sheet{number}.xml"] = (
            f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="{main}">'
            f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>')
    sheet_list = "".join(
        f'<sheet name="{_xml_escape(name)}" sheetId="{n}" r:id="rId{n}"/>'
        for n, (name, _) in enumerate(sheets, start=1))
    parts["xl/workbook.xml"] = (
        f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{main}" '
        f'xmlns:r="{rel}"><sheets>{sheet_list}</sheets></workbook>')
    rels = "".join(
        f'<Relationship Id="rId{n}" Type="{rel}/worksheet" '
        f'Target="worksheets/sheet{n}.xml"/>'
        for n in range(1, len(sheets) + 1))
    parts["xl/_rels/workbook.xml.rels"] = (
        '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="'
        'http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{rels}</Relationships>')
    strings = "".join(f"<si><t>{_xml_escape(s)}</t></si>" for s in shared)
    parts["xl/sharedStrings.xml"] = (
        f'<?xml version="1.0" encoding="UTF-8"?><sst xmlns="{main}" '
        f'count="{len(shared)}" uniqueCount="{len(shared)}">{strings}</sst>')
    parts["[Content_Types].xml"] = (
        '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.'
        'openxmlformats.org/package/2006/content-types"/>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in sorted(parts.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            archive.writestr(info, text)
    return path


def gem_workbook_sheets() -> list[tuple[str, list[list[str]]]]:
    """The committed ``gem_sample.xlsx``: an About sheet, the CSV rows as
    the data sheet with a title row above the header."""

    rows = list(csv.reader(io.StringIO(
        (INVENTORY / "gem_sample.csv").read_text(encoding="utf-8"))))
    return [("About", [["Synthetic GEM-shaped workbook for woof tests"]]),
            ("Data", [["Global Wind Power Tracker (synthetic)"]] + rows)]


def _by_id(collection: AssetCollection) -> dict[str, Asset]:
    return {a.asset_id: a for a in collection.assets}


def _osm_elements() -> list[dict]:
    return json.loads((INVENTORY / "osm_wind_elements.json").read_text())


def _all_sources() -> list[AssetCollection]:
    return [inv.import_uswtdb(INVENTORY / "uswtdb_sample.csv"),
            inv.import_gem(INVENTORY / "gem_sample.csv"),
            inv.import_emodnet(INVENTORY / "emodnet_sample.geojson"),
            inv.osm_elements_to_collection(_osm_elements()),
            import_assets([REPD], fmt="repd")]


def _provenance(asset: Asset) -> list[str]:
    return [e["asset_id"] for e in json.loads(asset.tags[inv.TAG_PROVENANCE])]


def _plant(name: str, lon: float, lat: float, *, source: str = "a",
           capacity: float | None = None, tags: dict | None = None,
           geometry: dict | None = None) -> Asset:
    return Asset(asset_id=f"{source}:{name}", kind="plant",
                 geometry=geometry or {"type": "Point",
                                       "coordinates": [lon, lat]},
                 source=source, source_ref=name, name=name,
                 generator_source="wind", capacity_mw=capacity,
                 tags={inv.TAG_STATUS: "operating", **(tags or {})})


def _turbine(ref: str, lon: float, lat: float, *, source: str = "t",
             capacity: float | None = None, project: str | None = None
             ) -> Asset:
    tags = {inv.TAG_STATUS: "operating"}
    if project:
        tags[inv.TAG_PROJECT] = project
    return Asset(asset_id=f"{source}:{ref}", kind="generator",
                 geometry={"type": "Point", "coordinates": [lon, lat]},
                 source=source, source_ref=ref, generator_source="wind",
                 capacity_mw=capacity, tags=tags)


# --------------------------------------------------------------------------
# USWTDB


def test_uswtdb_csv_rows_become_turbines():
    collection = inv.import_uswtdb(INVENTORY / "uswtdb_sample.csv")
    assets = _by_id(collection)
    assert sorted(assets) == [f"uswtdb:900000{i}" for i in range(1, 7)]
    first = assets["uswtdb:9000001"]
    assert first.kind == "generator" and first.generator_source == "wind"
    assert first.geometry == {"type": "Point", "coordinates": [-101.5, 32.4]}
    assert first.capacity_mw == pytest.approx(2.5)        # t_cap kW -> MW
    assert first.hub_height_m == 80.0 and first.rotor_diameter_m == 100.0
    assert first.name == "Synthetic Prairie"
    assert first.license == "public-domain"
    assert first.tags[inv.TAG_STATUS] == "operating"
    assert first.tags[inv.TAG_PROJECT] == "Synthetic Prairie"
    assert first.tags[inv.TAG_START_YEAR] == "2015"
    assert first.tags["uswtdb:t_ttlh"] == "130"
    # -9999 is the database's missing value, not a hub height
    missing = assets["uswtdb:9000003"]
    assert missing.hub_height_m is None
    assert "uswtdb:t_hh" not in missing.tags
    record = collection.sources[0]
    assert record["license"] == "public-domain"
    assert record["refusals"] == {"no xlong/ylat": 1, "duplicate case_id": 1}


def test_uswtdb_json_api_list_is_read():
    collection = inv.import_uswtdb(INVENTORY / "uswtdb_sample.json")
    assets = _by_id(collection)
    assert set(assets) == {"uswtdb:9000101", "uswtdb:9000102"}
    assert assets["uswtdb:9000102"].hub_height_m is None
    assert assets["uswtdb:9000101"].capacity_mw == pytest.approx(1.5)


def test_uswtdb_refuses_a_file_without_its_columns(tmp_path):
    path = tmp_path / "not_uswtdb.csv"
    path.write_text("id,lat,lon\n1,2,3\n")
    with pytest.raises(inv.InventoryRefused, match="not a USWTDB"):
        inv.import_uswtdb(path)


def test_uswtdb_status_filter_without_operating_keeps_nothing():
    collection = inv.import_uswtdb(INVENTORY / "uswtdb_sample.csv",
                                   statuses=("construction",))
    assert collection.assets == []
    assert collection.sources[0]["rows_skipped"] == {"status operating": 7}
    assert collection.sources[0]["refusals"] == {"duplicate case_id": 1}


# --------------------------------------------------------------------------
# GEM


def test_gem_csv_phases_filter_and_aggregate():
    collection = inv.import_gem(INVENTORY / "gem_sample.csv")
    assets = _by_id(collection)
    assert sorted(assets) == ["gem:L900001", "gem:L900002", "gem:L900003",
                              "gem:L900004", "gem:L900007"]
    array = assets["gem:L900003"]
    assert array.capacity_mw == 300.0                    # two phases summed
    assert array.tags[inv.TAG_INSTALLATION] == "offshore"
    assert array.tags[inv.TAG_START_YEAR] == "2018"
    phases = json.loads(array.tags["gem:phases"])
    assert [p["phase_id"] for p in phases] == ["G900003", "G900004"]
    assert array.license == "CC-BY-4.0"
    assert assets["gem:L900001"].tags[inv.TAG_INSTALLATION] == "onshore"
    record = collection.sources[0]
    assert record["rows_skipped"] == {"status construction": 1,
                                      "status announced": 1,
                                      "status shelved": 1}
    assert record["refusals"] == {"no Latitude/Longitude": 1}
    assert record["status_counts"]["shelved"] == 1       # "- inferred" cut


def test_gem_status_widening_keeps_construction():
    collection = inv.import_gem(INVENTORY / "gem_sample.csv",
                                statuses=("operating", "construction"))
    offshore = _by_id(collection)["gem:L900005"]
    assert offshore.tags[inv.TAG_STATUS] == "construction"


def test_gem_xlsx_fixture_matches_the_csv(tmp_path):
    fresh = write_xlsx(tmp_path / "gem.xlsx", gem_workbook_sheets())
    from_csv = inv.import_gem(INVENTORY / "gem_sample.csv")
    for workbook in (INVENTORY / "gem_sample.xlsx", fresh):
        from_xlsx = inv.import_gem(workbook)
        assert ([a.to_feature() for a in from_xlsx.assets]
                == [a.to_feature() for a in from_csv.assets])
        assert from_xlsx.sources[0]["sheets"] == ["Data"]


def test_read_xlsx_reads_shared_inline_and_numbers(tmp_path):
    path = write_xlsx(tmp_path / "w.xlsx",
                      [("One", [["a", "b", "c"], ["x & y", "", "2.5"]])],
                      inline_columns=(0,))
    sheets = inv.read_xlsx(path)
    assert sheets == [("One", [["a", "b", "c"], ["x & y", "", "2.5"]])]


def test_read_xlsx_refuses_a_non_workbook(tmp_path):
    path = tmp_path / "fake.xlsx"
    path.write_text("not a zip")
    with pytest.raises(inv.InventoryRefused, match="not an .xlsx"):
        inv.read_xlsx(path)


def test_gem_refuses_a_sheet_without_its_columns(tmp_path):
    path = write_xlsx(tmp_path / "w.xlsx", [("Data", [["Name", "Lat"]])])
    with pytest.raises(inv.InventoryRefused, match="Global Wind Power"):
        inv.import_gem(path)


# --------------------------------------------------------------------------
# EMODnet


def test_emodnet_features_become_offshore_plants():
    collection = inv.import_emodnet(INVENTORY / "emodnet_sample.geojson")
    assets = _by_id(collection)
    assert sorted(assets) == ["emodnet:windfarms.5", "emodnet:windfarmspoly.1",
                              "emodnet:windfarmspoly.2"]
    array = assets["emodnet:windfarmspoly.1"]
    assert array.kind == "plant" and array.geometry["type"] == "Polygon"
    assert array.capacity_mw == 300.0
    assert array.tags[inv.TAG_TURBINES] == "50"
    assert array.tags[inv.TAG_INSTALLATION] == "offshore"
    assert array.tags["emodnet:status"] == "Production"
    record = collection.sources[0]
    assert record["rows_skipped"] == {"status retired": 1,
                                      "status construction": 1}
    assert sum(record["refusals"].values()) == 1         # the LineString


@pytest.mark.parametrize("crs", ["urn:ogc:def:crs:EPSG::4326",
                                 "EPSG:3035"])
def test_emodnet_refuses_latitude_first_or_projected_crs(tmp_path, crs):
    document = json.loads((INVENTORY / "emodnet_sample.geojson").read_text())
    document["crs"]["properties"]["name"] = crs
    path = tmp_path / "e.geojson"
    path.write_text(json.dumps(document))
    with pytest.raises(inv.InventoryRefused, match="crs"):
        inv.import_emodnet(path)


# --------------------------------------------------------------------------
# OSM


def test_osm_elements_keep_wind_turbines_and_plants():
    collection = inv.osm_elements_to_collection(_osm_elements())
    assets = _by_id(collection)
    assert set(assets) == {
        "osm:node/9100001", "osm:node/9100002", "osm:node/9100003",
        "osm:node/9100005", "osm:node/9100006", "osm:way/9200001",
        "osm:relation/9300001"}
    assert assets["osm:node/9100001"].hub_height_m == 100.0
    assert assets["osm:way/9200001"].geometry["type"] == "Polygon"
    assert assets["osm:relation/9300001"].capacity_mw == 40.0
    assert all(a.tags[inv.TAG_STATUS] == "operating"
               for a in collection.assets)
    record = collection.sources[0]
    assert record["license"] == "ODbL-1.0"
    assert record["rows_skipped"] == {"not a wind generator or plant": 1}
    assert record["refusals"] == {"no coordinates": 1}


def test_osm_pbf_refuses_without_pyosmium(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "osmium", None)
    path = tmp_path / "x.osm.pbf"
    path.write_bytes(b"")
    with pytest.raises(ImportRefused, match="pyosmium"):
        import_assets([path], fmt="osm-pbf")


class _Location:
    def __init__(self, lon, lat):
        self.lon, self.lat = lon, lat

    def valid(self):
        return self.lon is not None


class _Ref:
    def __init__(self, ref, lon, lat):
        self.ref = ref
        self.location = _Location(lon, lat)


class _Tag:
    def __init__(self, k, v):
        self.k, self.v = k, v


def _tags(mapping):
    return [_Tag(k, v) for k, v in mapping.items()]


def test_osm_pbf_handler_feeds_the_shared_conversion(monkeypatch, tmp_path):
    """A stand-in for pyosmium's SimpleHandler drives the handler code."""

    elements = {e["id"]: e for e in _osm_elements()}
    seen_args = {}

    class SimpleHandler:
        def apply_file(self, filename, locations=False, idx="flex_mem"):
            seen_args.update(filename=filename, locations=locations, idx=idx)
            for element in elements.values():
                if element["type"] != "node":
                    continue
                self.node(types.SimpleNamespace(
                    id=element["id"], tags=_tags(element["tags"]),
                    location=_Location(element.get("lon"),
                                       element.get("lat"))))
            way = elements[9200001]
            self.way(types.SimpleNamespace(
                id=way["id"], tags=_tags(way["tags"]),
                nodes=[_Ref(n, p["lon"], p["lat"])
                       for n, p in zip(way["nodes"], way["geometry"])]))
            # a closed way also arrives as an area; it must be ignored
            self.area(types.SimpleNamespace(
                from_way=lambda: True, tags=_tags(way["tags"])))
            relation = elements[9300001]
            ring = [_Ref(0, p["lon"], p["lat"])
                    for p in relation["members"][0]["geometry"]]
            self.area(types.SimpleNamespace(
                from_way=lambda: False, orig_id=lambda: relation["id"],
                tags=_tags(relation["tags"]),
                outer_rings=lambda: [ring], inner_rings=lambda outer: []))

    monkeypatch.setitem(sys.modules, "osmium",
                        types.SimpleNamespace(SimpleHandler=SimpleHandler))
    path = tmp_path / "region.osm.pbf"
    path.write_bytes(b"fake")
    collection = import_assets([path], fmt="osm-pbf",
                               osm_index="sparse_file_array,/tmp/x")
    expected = inv.osm_elements_to_collection(_osm_elements())
    assert ([a.to_feature() for a in collection.assets]
            == [a.to_feature() for a in expected.assets])
    assert seen_args["locations"] is True
    assert seen_args["idx"] == "sparse_file_array,/tmp/x"


# --------------------------------------------------------------------------
# status, names


def test_asset_status_maps_repd_and_osm():
    repd = _by_id(import_assets([REPD], fmt="repd"))
    assert inv.asset_status(repd["repd:90001"]) == "operating"
    assert inv.asset_status(repd["repd:90004"]) == "construction"
    assert inv.asset_status(repd["repd:90002"]) == "pre-construction"
    assert inv.asset_status(repd["repd:90008"]) == "cancelled"
    assert inv.asset_installation(repd["repd:90004"]) == "offshore"
    assert inv.asset_installation(repd["repd:90001"]) == "onshore"


def test_repd_import_status_filter_is_opt_in():
    everything = import_assets([REPD], fmt="repd")
    operating = import_assets([REPD], fmt="repd", statuses=("operating",))
    assert len(everything.assets) == 7
    assert sorted(a.asset_id for a in operating.assets) == ["repd:90001",
                                                            "repd:90006"]
    skipped = operating.sources[0]["rows_skipped"]
    assert skipped["status construction"] == 1


def test_status_flag_is_refused_for_formats_without_status():
    with pytest.raises(ImportRefused, match="--status applies"):
        import_assets([FIXTURES / "assets_wales.geojson"], fmt="geojson",
                      statuses=("operating",))


@pytest.mark.parametrize("a, b, verdict", [
    ("Whitelee Wind Farm", "Whitelee", "agree"),
    ("Hornsea 1", "Hornsea Project One", "agree"),
    ("Hornsea 1", "Hornsea 2", "disagree"),
    ("Walney", "Walney Extension", "disagree"),
    ("North Hoyle", "South Hoyle", "disagree"),
    ("Parque Eólico Los Vientos", "Los Vientos Wind Farm", "agree"),
    ("Synthetic Moor", "Completely Different", "disagree"),
    (None, "Anything", None),
])
def test_name_similarity(a, b, verdict):
    similarity = inv.name_similarity(inv.name_key(a), inv.name_key(b))
    if verdict is None:
        assert similarity is None
    elif verdict == "agree":
        assert similarity >= 0.7
    else:
        assert similarity < 0.3


# --------------------------------------------------------------------------
# merge


def test_merge_deduplicates_across_sources():
    merged, report = inv.merge_inventory(_all_sources())
    assets = _by_id(merged)
    # offshore arrays: EMODnet polygon wins, GEM phases folded in; Array 1
    # and Array 2 stay apart (distinguishing numbers differ)
    one = assets["emodnet:windfarmspoly.1"]
    assert _provenance(one) == ["emodnet:windfarmspoly.1", "gem:L900003"]
    assert one.license == "CC-BY-4.0"
    assert _provenance(assets["emodnet:windfarmspoly.2"]) == [
        "emodnet:windfarmspoly.2", "gem:L900004"]
    # REPD and GEM agree on the Welsh farm by name; REPD wins over GEM
    brecon = assets["repd:90001"]
    assert _provenance(brecon) == ["repd:90001", "gem:L900002"]
    assert brecon.license == "CC-BY-4.0 AND OGL-UK-3.0"
    assert brecon.tags[inv.TAG_LICENSES] == "CC-BY-4.0;OGL-UK-3.0"
    # OSM polygon wins over the GEM point inside it and takes its capacity
    hill = assets["osm:way/9200001"]
    assert hill.capacity_mw == 30.0
    assert _provenance(hill) == ["osm:way/9200001", "gem:L900007"]
    # OSM and USWTDB map the same turbine 17 m apart
    turbine = assets["osm:node/9100005"]
    assert _provenance(turbine) == ["osm:node/9100005", "uswtdb:9000001"]
    assert turbine.hub_height_m == 80.0                  # filled from USWTDB
    assert "uswtdb:9000001" not in assets
    # REPD's under-construction offshore farm is filtered; non-wind dropped
    assert "repd:90004" not in assets and "repd:90002" not in assets
    assert report["dropped_by_status"] == {"construction": 1}
    assert report["counts"]["dropped: not a wind plant or generator"] == 4
    assert all(inv.TAG_PROVENANCE in a.tags for a in merged.assets)
    assert merged.sources[-1]["source"] == inv.MERGE_RECORD_SOURCE


def test_merge_prefers_turbine_level_when_turbines_cover_the_plant():
    merged, report = inv.merge_inventory(_all_sources())
    assets = _by_id(merged)
    prairie = assets["gem:L900001"]
    assert prairie.tags[inv.TAG_TURBINE_LEVEL] == "yes"
    linked = sorted(a.asset_id for a in merged.assets
                    if a.tags.get("plant") == "gem:L900001")
    assert linked == ["osm:node/9100005", "uswtdb:9000002", "uswtdb:9000003",
                      "uswtdb:9000004"]
    # three hub-height-only OSM turbines cannot cover a 30 MW plant
    hill = assets["osm:way/9200001"]
    assert hill.tags[inv.TAG_TURBINES_PARTIAL] == "3"
    assert inv.TAG_TURBINE_LEVEL not in hill.tags
    assert not any(a.tags.get("plant") == hill.asset_id
                   for a in merged.assets)
    # Ridge turbines have no plant anywhere near
    assert "plant" not in assets["uswtdb:9000005"].tags
    assert report["counts"]["plants turbine-level"] == 1


def test_merge_never_joins_two_assets_of_one_source():
    a = _plant("Twin", 0.0, 50.0, source="s", capacity=10.0)
    b = replace(_plant("Twin", 0.001, 50.0, source="s", capacity=10.0),
                asset_id="s:Twin-b", source_ref="Twin-b")
    c = _plant("Twin", 0.0005, 50.0, source="other", capacity=10.0)
    merged, _ = inv.merge_inventory([AssetCollection([a, b]),
                                     AssetCollection([c])])
    assert len(merged.assets) == 2
    assert sorted(len(_provenance(x)) for x in merged.assets) == [1, 2]


def test_merge_keeps_nearby_farms_whose_capacity_disagrees():
    a = _plant("Alpha", 0.0, 50.0, source="s", capacity=10.0)
    b = _plant("Alpha", 0.003, 50.0, source="t", capacity=100.0)
    merged, _ = inv.merge_inventory([AssetCollection([a]),
                                     AssetCollection([b])])
    assert len(merged.assets) == 2


def test_merge_matches_unnamed_plants_only_when_near():
    a = _plant("Alpha", 0.0, 50.0, source="s", capacity=10.0)
    near = Asset(asset_id="t:1", kind="plant",
                 geometry={"type": "Point", "coordinates": [0.01, 50.0]},
                 source="t", source_ref="1", generator_source="wind",
                 tags={inv.TAG_STATUS: "operating"})
    far = replace_geometry(near, 0.03, "u")
    merged, _ = inv.merge_inventory([AssetCollection([a]),
                                     AssetCollection([near]),
                                     AssetCollection([far])])
    by_id = _by_id(merged)
    assert _provenance(by_id["s:Alpha"]) == ["s:Alpha", "t:1"]
    assert "u:1" in by_id


def replace_geometry(asset: Asset, lon: float, source: str) -> Asset:
    return replace(asset, asset_id=f"{source}:1", source=source,
                   geometry={"type": "Point", "coordinates": [lon, 50.0]})


def test_merge_never_matches_onshore_with_offshore():
    a = _plant("Coast", 0.0, 50.0, source="s",
               tags={inv.TAG_INSTALLATION: "onshore"})
    b = _plant("Coast", 0.0, 50.0, source="t",
               tags={inv.TAG_INSTALLATION: "offshore"})
    merged, _ = inv.merge_inventory([AssetCollection([a]),
                                     AssetCollection([b])])
    assert len(merged.assets) == 2


def test_merge_status_filter_and_unknown():
    unknown = Asset(asset_id="x:1", kind="plant",
                    geometry={"type": "Point", "coordinates": [5.0, 5.0]},
                    source="x", source_ref="1", generator_source="wind")
    merged, report = inv.merge_inventory([AssetCollection([unknown])])
    assert merged.assets == [] and report["dropped_by_status"] == {
        "unknown": 1}
    merged, _ = inv.merge_inventory([AssetCollection([unknown])],
                                    statuses=("operating", "unknown"))
    assert [a.asset_id for a in merged.assets] == ["x:1"]


def test_merge_keep_non_wind_passes_assets_through():
    wales = load_assets(FIXTURES / "assets_wales.geojson")
    merged, report = inv.merge_inventory(
        [wales], statuses=("unknown",), keep_non_wind=True)
    assert len(merged.assets) == len(wales.assets)
    lines = [a for a in merged.assets if a.kind == "line"]
    assert lines and all(inv.TAG_PROVENANCE not in a.tags for a in lines)


def test_remerge_extends_provenance_and_is_idempotent():
    merged, _ = inv.merge_inventory(_all_sources())
    again, report = inv.merge_inventory([merged, merged])
    assert ([a.to_feature() for a in again.assets]
            == [a.to_feature() for a in merged.assets])
    assert report["counts"]["dropped: same asset_id"] == len(merged.assets)
    gem_csv = inv.import_gem(INVENTORY / "gem_sample.csv")
    extended, _ = inv.merge_inventory([merged, gem_csv])
    # the same GEM rows fold into the same clusters: nothing new appears
    assert len(extended.assets) == len(merged.assets)


def test_merge_rules_refuse_nonsense():
    with pytest.raises(inv.InventoryRefused):
        inv.MergeRules(plant_near_m=9000.0, plant_match_m=5000.0)
    with pytest.raises(inv.InventoryRefused):
        inv.MergeRules(name_threshold=1.5)
    with pytest.raises(inv.InventoryRefused):
        inv.merge_inventory([AssetCollection([])], statuses=("built",))


def test_link_needs_a_project_name_beyond_the_near_radius():
    plant = _plant("Big Sky", 0.0, 45.0, source="g", capacity=6.0)
    named = [_turbine(str(i), 0.05 + 0.001 * i, 45.0, capacity=3.0,
                      project="Big Sky") for i in range(2)]
    other = [_turbine(f"o{i}", 0.05 + 0.001 * i, 45.01, capacity=3.0,
                      project="Little Sky Ridge", source="u")
             for i in range(2)]
    merged, _ = inv.merge_inventory([AssetCollection([plant]),
                                     AssetCollection(named),
                                     AssetCollection(other)])
    by_id = _by_id(merged)
    assert by_id["g:Big Sky"].tags[inv.TAG_TURBINE_LEVEL] == "yes"
    assert by_id["t:0"].tags["plant"] == "g:Big Sky"
    assert "plant" not in by_id["u:o0"].tags


# --------------------------------------------------------------------------
# CLI


def test_cli_statuses_match_the_module():
    assert energy_cli._STATUSES == inv.STATUSES
    assert set(inv.INVENTORY_FORMATS) <= set(energy_cli._IMPORT_FORMATS)


def _run(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


def test_cli_import_merge_sites_end_to_end(tmp_path, capsys):
    out = {}
    for name, fmt, files, extra in (
            ("uswtdb", "uswtdb", ["uswtdb_sample.csv", "uswtdb_sample.json"],
             []),
            ("gem", "gem", ["gem_sample.xlsx"], []),
            ("emodnet", "emodnet", ["emodnet_sample.geojson"],
             ["--status", "operating,construction"])):
        out[name] = tmp_path / f"{name}.geojson"
        assert _run(["energy", "import", *[str(INVENTORY / f) for f in files],
                     "--format", fmt, *extra, "-o", str(out[name])]) == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["assets"] > 0
    out["repd"] = tmp_path / "repd.geojson"
    assert _run(["energy", "import", str(REPD), "--format", "repd",
                 "-o", str(out["repd"])]) == 0
    capsys.readouterr()
    out["osm"] = tmp_path / "osm.geojson"
    dump_assets(inv.osm_elements_to_collection(_osm_elements()), out["osm"])
    merged = tmp_path / "merged.geojson"
    assert _run(["energy", "inventory", "merge", *map(str, out.values()),
                 "-o", str(merged)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema"] == inv.SUMMARY_SCHEMA
    # --status at import kept EMODnet's construction farm; merge drops it
    assert summary["dropped_by_status"] == {"construction": 2}
    document = json.loads(merged.read_text())
    assert document["woof"]["schema"] == ASSETS_SCHEMA
    collection = load_assets(merged)
    sites_path = tmp_path / "sites.json"
    assert _run(["energy", "sites", str(merged), "-o", str(sites_path)]) == 0
    sites_summary = json.loads(capsys.readouterr().out)
    sites = load_sites(sites_path)
    kinds = {s.kind for s in sites.sites}
    assert kinds == {"turbine", "plant"}
    site_assets = {s.asset_id for s in sites.sites}
    # the turbine-level plant has no site of its own; its turbines do
    assert "gem:L900001" not in site_assets
    assert {"uswtdb:9000002", "osm:node/9100005"} <= site_assets
    assert sites_summary["drops"]["wind_plant_with_turbines"] >= 1
    assert 80.0 in sites.heights_m and 90.0 in sites.heights_m
    assert len(collection.assets) == summary["assets"]


def test_cli_merge_refusal_prints_json(tmp_path, capsys):
    rc = _run(["energy", "inventory", "merge", str(tmp_path / "nope.geojson"),
               "-o", str(tmp_path / "m.geojson")])
    assert rc == 2
    assert "refusal" in json.loads(capsys.readouterr().out)


def test_cli_download_refuses_gem_with_guidance(tmp_path, capsys):
    rc = _run(["energy", "inventory", "download", "--dataset", "gem",
               "-o", str(tmp_path / "gem.xlsx")])
    assert rc == 2
    assert "globalenergymonitor" in json.loads(
        capsys.readouterr().out)["refusal"]


def test_cli_import_refuses_osm_index_for_other_formats(tmp_path, capsys):
    rc = _run(["energy", "import", str(INVENTORY / "gem_sample.csv"),
               "--format", "gem", "--osm-index", "flex_mem",
               "-o", str(tmp_path / "x.geojson")])
    assert rc == 2
    capsys.readouterr()


# --------------------------------------------------------------------------
# review regressions


def test_partial_polygon_plant_keeps_its_site(tmp_path, capsys):
    merged, _ = inv.merge_inventory(_all_sources())
    path = tmp_path / "merged.geojson"
    dump_assets(merged, path)
    sites_path = tmp_path / "sites.json"
    assert _run(["energy", "sites", str(path), "-o", str(sites_path)]) == 0
    capsys.readouterr()
    sites = load_sites(sites_path)
    # the OSM Hill polygon holds three turbines covering too little of
    # its 30 MW: the plant keeps its own site next to them
    assert "osm:way/9200001" in {s.asset_id for s in sites.sites}


def test_plant_without_capacity_or_count_is_not_turbine_level():
    plant = _plant("Bare", 0.0, 45.0, source="g")
    turbine = _turbine("1", 0.001, 45.0, capacity=3.0)
    merged, _ = inv.merge_inventory([AssetCollection([plant]),
                                     AssetCollection([turbine])])
    by_id = _by_id(merged)
    assert by_id["g:Bare"].tags[inv.TAG_TURBINES_PARTIAL] == "1"
    assert "plant" not in by_id["t:1"].tags
    counted = _plant("Counted", 0.0, 45.0, source="g",
                     tags={inv.TAG_TURBINES: "1"})
    merged, _ = inv.merge_inventory([AssetCollection([counted]),
                                     AssetCollection([turbine])])
    assert _by_id(merged)["g:Counted"].tags[inv.TAG_TURBINE_LEVEL] == "yes"


def test_remerge_of_two_merges_sharing_a_record_joins_them():
    gem = _plant("Shared", 0.0, 50.0, source="gem", capacity=100.0)
    osm = _plant("Shared", 0.0005, 50.0, source="osm", capacity=100.0)
    emod = _plant("Shared", -0.0005, 50.0, source="emodnet",
                  capacity=100.0)
    first, _ = inv.merge_inventory([AssetCollection([osm]),
                                    AssetCollection([gem])])
    second, _ = inv.merge_inventory([AssetCollection([emod]),
                                     AssetCollection([gem])])
    both, report = inv.merge_inventory([first, second])
    assert len(both.assets) == 1
    assert sorted(_provenance(both.assets[0])) == [
        "emodnet:Shared", "gem:Shared", "osm:Shared"]
    assert report["counts"]["plants joined by a shared provenance record"] \
        == 1


def test_turbine_plant_tag_follows_its_plant_into_a_cluster():
    point = _plant("Old", 0.0, 50.0, source="osm", capacity=6.0)
    polygon = _plant("Old", 0.0, 50.0, source="emodnet", capacity=6.0,
                     geometry={"type": "Polygon", "coordinates": [[
                         [-0.01, 49.99], [0.01, 49.99], [0.01, 50.01],
                         [-0.01, 50.01], [-0.01, 49.99]]]})
    turbines = [replace(_turbine(str(i), 0.002 * i, 50.0, source="osm",
                                 capacity=3.0),
                        tags={inv.TAG_STATUS: "operating",
                              "plant": "osm:Old"}) for i in range(2)]
    merged, _ = inv.merge_inventory([
        AssetCollection([point] + turbines), AssetCollection([polygon])])
    by_id = _by_id(merged)
    assert "osm:Old" not in by_id                       # polygon won
    assert {by_id[f"osm:{i}"].tags["plant"] for i in range(2)} == {
        "emodnet:Old"}
    assert by_id["emodnet:Old"].tags[inv.TAG_TURBINE_LEVEL] == "yes"


def test_french_sur_is_not_a_compass_word():
    similarity = inv.name_similarity(
        inv.name_key("Fresnoy"), inv.name_key("Parc éolien de Fresnoy-sur-Mer"))
    assert similarity is not None and similarity > 0.3


def test_unreadable_uswtdb_offshore_flag_leaves_installation_unset(tmp_path):
    path = tmp_path / "u.csv"
    path.write_text("case_id,t_offshore,xlong,ylat\n1,maybe,-90,40\n"
                    "2,1,-90.1,40\n")
    assets = _by_id(inv.import_uswtdb(path))
    assert inv.TAG_INSTALLATION not in assets["uswtdb:1"].tags
    assert assets["uswtdb:2"].tags[inv.TAG_INSTALLATION] == "offshore"


def test_osm_pbf_without_operating_status_is_refused_before_reading(
        monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "osmium", None)
    path = tmp_path / "x.osm.pbf"
    path.write_bytes(b"")
    with pytest.raises(ImportRefused, match="keeps nothing"):
        import_assets([path], fmt="osm-pbf", statuses=("construction",))


def test_gem_refusals_name_the_spreadsheet_row(tmp_path):
    rows = list(csv.reader(io.StringIO(
        (INVENTORY / "gem_sample.csv").read_text(encoding="utf-8"))))
    path = write_xlsx(tmp_path / "g.xlsx",
                      [("Data", [["title"], ["subtitle"]] + rows)])
    record = inv.import_gem(path).sources[0]
    # the row without coordinates is CSV line 10, sheet row 12 here
    assert record["refusal_examples"] == [
        "Data row 12 (G900010): no Latitude/Longitude"]


def test_cli_import_formats_match_the_importers():
    from woof.energy import importers

    assert energy_cli._IMPORT_FORMATS == importers.FORMATS
