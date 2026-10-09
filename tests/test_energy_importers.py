"""``woof energy import``: PyPSA-Eur, REPD, GeoJSON and CSV importers, the
WKT parser, the National Grid transform and the merge rules."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest

from woof.cli import build_parser
from woof.energy import importers
from woof.energy.contracts import (
    ASSETS_SCHEMA,
    Asset,
    AssetCollection,
    dump_assets,
    load_assets,
)
from woof.energy.importers import (
    ImportRefused,
    WKTError,
    bng_to_osgb36,
    bng_to_wgs84,
    import_assets,
    merge_collections,
    merge_with_report,
    parse_wkt,
)

FIXTURES = Path(__file__).parent / "fixtures" / "energy"
IMPORT = FIXTURES / "import"


def _dms(degrees: int, minutes: int, seconds: float) -> float:
    return degrees + minutes / 60.0 + seconds / 3600.0


def _distance_m(lon1, lat1, lon2, lat2) -> float:
    return float(importers._haversine_m(lon1, lat1, lon2, lat2))


def _by_id(collection: AssetCollection) -> dict[str, Asset]:
    return {asset.asset_id: asset for asset in collection.assets}


def _record(collection: AssetCollection, source: str) -> dict:
    return next(r for r in collection.sources if r["source"] == source)


# --------------------------------------------------------------------------
# WKT


@pytest.mark.parametrize("text, expected", [
    ("POINT (-3.1 51.5)", {"type": "Point", "coordinates": [-3.1, 51.5]}),
    ("point(-3.1 51.5)", {"type": "Point", "coordinates": [-3.1, 51.5]}),
    ("POINT Z (-3.1 51.5 120)", {"type": "Point",
                                 "coordinates": [-3.1, 51.5]}),
    ("'LINESTRING (1 2, 3 4)'", {"type": "LineString",
                                 "coordinates": [[1.0, 2.0], [3.0, 4.0]]}),
    ("LINESTRING(1 2,3 4,5 6)", {"type": "LineString",
                                 "coordinates": [[1.0, 2.0], [3.0, 4.0],
                                                 [5.0, 6.0]]}),
    ("LINESTRING ZM (1 2 3 4, 5 6 7 8)",
     {"type": "LineString", "coordinates": [[1.0, 2.0], [5.0, 6.0]]}),
    ("MULTILINESTRING ((1 2, 3 4), (5 6, 7 8))",
     {"type": "MultiLineString",
      "coordinates": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]}),
    ("  LINESTRING (1e-1 -2.5E1, 3 4)  ",
     {"type": "LineString", "coordinates": [[0.1, -25.0], [3.0, 4.0]]}),
])
def test_parse_wkt(text, expected):
    assert parse_wkt(text) == expected


@pytest.mark.parametrize("text, match", [
    ("", "empty"),
    ("''", "empty"),
    ("POLYGON ((0 0, 1 0, 1 1, 0 0))", "unsupported"),
    ("LINESTRING EMPTY", "EMPTY"),
    ("LINESTRING (1 2)", "at least two"),
    ("LINESTRING (1 2, 3)", "2 to 4 numbers"),
    ("LINESTRING (1 2 3 4 5, 6 7)", "2 to 4 numbers"),
    ("LINESTRING (1 2, 3 4", "expected '\\)'"),
    ("LINESTRING 1 2, 3 4", "expected '\\('"),
    ("LINESTRING (1 2, a 4)", "not a number"),
    ("POINT (nan 2)", "not finite"),
    ("POINT (1 2) extra", "trailing"),
    ("MULTILINESTRING ((1 2, 3 4), (5 6))", "at least two"),
    ("MULTILINESTRING ((1 2, 3 4)", "expected '\\)'"),
])
def test_parse_wkt_refusals(text, match):
    with pytest.raises(WKTError, match=match):
        parse_wkt(text)


# --------------------------------------------------------------------------
# British National Grid


def test_bng_inverse_transverse_mercator_matches_os_annexe_c():
    """OS "A guide to coordinate systems in Great Britain" v3.6, Annexe
    C.2 worked example: E 651409.903, N 313177.270 on Airy 1830 is
    52 39 27.2531 N, 001 43 04.5177 E (OSGB36)."""

    lat, lon = bng_to_osgb36(651409.903, 313177.270)
    assert float(lat) == pytest.approx(_dms(52, 39, 27.2531), abs=2e-8)
    assert float(lon) == pytest.approx(_dms(1, 43, 4.5177), abs=2e-8)


def test_bng_to_wgs84_matches_os_annexe_d():
    """OS guide v3.6, Annexe D: WGS84 53 36 43.1653 N, 001 39 51.9920 W
    transforms by the OS Helmert to National Grid 422297.792 mE,
    412878.741 mN.  The inverse must land back on the WGS84 point; the
    Helmert itself is quoted as good to 3.5 m (95 %) against OSTN15, so
    the import tolerance is about 5 m and this round trip is far inside
    it."""

    lon, lat = bng_to_wgs84(422297.792, 412878.741)
    expected_lat = _dms(53, 36, 43.1653)
    expected_lon = -_dms(1, 39, 51.9920)
    error = _distance_m(lon[0], lat[0], expected_lon, expected_lat)
    assert error < 0.05
    assert error < 5.0


def test_bng_to_wgs84_is_vectorised_and_orders_lon_lat():
    lon, lat = bng_to_wgs84([302219.0, 422297.792], [166320.0, 412878.741])
    assert lon.shape == lat.shape == (2,)
    # Aberthaw, Vale of Glamorgan: about 51.387 N, 3.407 W.
    assert lat[0] == pytest.approx(51.387, abs=0.002)
    assert lon[0] == pytest.approx(-3.407, abs=0.002)


# --------------------------------------------------------------------------
# PyPSA-Eur


def test_pypsa_osm_prebuilt_layout():
    collection = import_assets([IMPORT / "pypsa_osm"], fmt="pypsa-eur")
    assets = _by_id(collection)
    kinds = sorted(a.kind for a in collection.assets)
    assert kinds.count("substation") == 5
    assert kinds.count("line") == 3 and kinds.count("cable") == 2

    # Two buses at one location are one substation with both voltages.
    west = assets["pypsa-eur:substation/way/1001-400"]
    assert west.voltage_kv == (400.0, 132.0)
    assert west.tags["pypsa:buses"] == "way/1001-132;way/1001-400"
    # Buses ~120 m apart joined by a transformer are one substation.
    joined = assets["pypsa-eur:substation/way/1003-400"]
    assert joined.voltage_kv == (400.0, 132.0)
    assert "pypsa-eur:substation/way/1003-132" not in assets
    assert assets["pypsa-eur:substation/relation/1004"].tags["dc"] == "yes"

    line = assets["pypsa-eur:line/merged_way/2001-400+1"]
    assert line.kind == "line" and line.circuits == 2
    assert line.voltage_kv == (400.0,)
    assert line.geometry["coordinates"][0] == [-4.99, 51.684]
    assert line.tags["osm:refs"] == "way/2001;way/2011"
    cable = assets["pypsa-eur:line/way/2003-132"]
    assert cable.kind == "cable"

    straight = assets["pypsa-eur:line/way/2004-400"]
    assert straight.tags["woof:geometry"] == "bus-to-bus"
    assert straight.tags["under_construction"] == "yes"
    assert straight.geometry["coordinates"] == [[-4.2, 51.8], [-4.99, 51.684]]

    link = assets["pypsa-eur:link/relation/3001-320-DC"]
    assert link.kind == "cable" and link.tags["carrier"] == "DC"
    assert link.tags["p_nom_mw"] == "1000"

    record = _record(collection, "pypsa-eur")
    assert record["variant"] == "osm-prebuilt"
    assert record["license"] == "ODbL-1.0"
    assert "OpenStreetMap" in record["attribution"]
    assert {a.license for a in collection.assets} == {"ODbL-1.0"}
    assert record["geometry_fallbacks"] == {"bus-to-bus": 1}
    assert record["refusals"] == {"bad WKT geometry": 1}
    assert "way/2005" not in json.dumps([a.asset_id for a in
                                         collection.assets])
    assert {f["name"] for f in record["files"]} == {
        "buses.csv", "lines.csv", "links.csv", "transformers.csv"}


def test_pypsa_gridkit_layout():
    collection = import_assets([IMPORT / "pypsa_gridkit"], fmt="pypsa-eur")
    assets = _by_id(collection)
    record = _record(collection, "pypsa-eur")
    assert record["variant"] == "gridkit"
    assert record["license"] == "CC-BY-4.0"
    # The joint and the wind-farm bus are not substations.
    substations = sorted(a.asset_id for a in collection.assets
                         if a.kind == "substation")
    assert substations == ["pypsa-eur:substation/1",
                           "pypsa-eur:substation/3"]
    assert assets["pypsa-eur:substation/1"].operator == "Synthetic TSO"
    assert assets["pypsa-eur:substation/1"].tags["entsoe:oid"] == "9001"
    assert record["rows_skipped"] == {
        "bus symbol joint (not a substation)": 1,
        "bus symbol Wind farm (not a substation)": 1}

    assert assets["pypsa-eur:line/5003"].kind == "cable"
    assert assets["pypsa-eur:line/5003"].tags["woof:geometry"] == "bus-to-bus"
    assert assets["pypsa-eur:line/5002"].tags["under_construction"] == "yes"
    link = assets["pypsa-eur:link/6001"]
    assert link.kind == "line" and link.tags["carrier"] == "DC"
    assert link.voltage_kv == (400.0,)
    assert link.tags["woof:voltage"] == "from-bus0"

    wind = assets["pypsa-eur:generator/8001"]
    assert wind.kind == "plant" and wind.generator_source == "wind"
    assert wind.capacity_mw == 120.0 and wind.name == "Synthetic Moor"
    gas = assets["pypsa-eur:generator/8002"]
    assert gas.generator_source == "gas" and gas.capacity_mw is None
    hydro = assets["pypsa-eur:generator/8003"]
    assert hydro.generator_source == "hydro"
    assert hydro.tags["woof:geometry"] == "bus-location"
    assert hydro.geometry["coordinates"] == [-3.2, 51.5]
    assert record["geometry_fallbacks"] == {"bus-to-bus": 1,
                                            "bus-location": 1}
    assert record["attributes_derived"] == {"voltage_kv from bus0": 1}


def test_pypsa_network_export_layout_reads_double_quoted_csv():
    collection = import_assets(
        [IMPORT / "pypsa_network" / name for name in
         ("buses.csv", "lines.csv", "links.csv", "generators.csv")],
        fmt="pypsa-eur")
    assets = _by_id(collection)
    record = _record(collection, "pypsa-eur")
    assert record["variant"] == "pypsa-network"
    line = assets["pypsa-eur:line/1"]
    assert len(line.geometry["coordinates"]) == 3
    assert line.circuits is None and line.tags["pypsa:num_parallel"] == "1.5"
    assert assets["pypsa-eur:line/2"].circuits == 2
    assert assets["pypsa-eur:line/2"].voltage_kv == (400.0,)
    assert assets["pypsa-eur:link/DC1"].kind == "cable"
    assert "pypsa-eur:link/GB0 0 H2 Electrolysis" not in assets
    assert record["rows_skipped"]["link carrier H2 Electrolysis "
                                  "(not a DC link)"] == 1
    assert assets["pypsa-eur:generator/GB0 0 onwind"].generator_source == "wind"
    assert assets["pypsa-eur:generator/GB0 1 solar"].capacity_mw is None
    station = assets["pypsa-eur:substation/GB0 0"]
    # The H2 bus shares GB0 0's coordinates but is not part of the
    # substation, and its v_nom of 1 does not become a substation voltage.
    assert station.tags["pypsa:buses"] == "GB0 0"
    assert station.voltage_kv == (400.0,)
    # pypsa exports store length in km.
    assert line.tags["length_m"] == "17000.0"


@pytest.mark.parametrize("generator, expected", [
    ("Fossil gas", "gas"), ("biogas", "biogas"), ("Biomass", "biomass"),
    ("offwind-dc", "wind"), ("solar-hsat", "solar"), ("ror", "hydro"),
    ("Brown coal/Lignite", "coal"), ("Other or not listed", "other"),
    ("H2", None), ("hydrogen CCGT", None), ("", None),
])
def test_pypsa_generator_vocabulary(generator, expected):
    assert importers._pypsa_generator_source(generator) == expected


def test_pypsa_refuses_missing_table(tmp_path):
    for name in ("buses.csv", "lines.csv"):
        shutil.copy(IMPORT / "pypsa_osm" / name, tmp_path / name)
    with pytest.raises(ImportRefused, match="missing links.csv"):
        import_assets([tmp_path], fmt="pypsa-eur")


def test_pypsa_refuses_unknown_layout_listing_expected_columns(tmp_path):
    shutil.copytree(IMPORT / "pypsa_osm", tmp_path / "net")
    lines = tmp_path / "net" / "lines.csv"
    lines.write_text("line_id,from_bus,to_bus\n1,a,b\n")
    with pytest.raises(ImportRefused) as caught:
        import_assets([tmp_path / "net"], fmt="pypsa-eur")
    message = str(caught.value)
    assert "lines.csv" in message and "'bus0'" in message
    assert "bus0, bus1" in message and "geometry (WKT)" in message


def test_pypsa_refuses_buses_without_coordinates(tmp_path):
    shutil.copytree(IMPORT / "pypsa_osm", tmp_path / "net")
    (tmp_path / "net" / "buses.csv").write_text("bus_id,voltage\n1,400\n")
    with pytest.raises(ImportRefused, match="lacks 'x'"):
        import_assets([tmp_path / "net"], fmt="pypsa-eur")


def test_pypsa_refuses_two_networks_and_foreign_tables(tmp_path):
    with pytest.raises(ImportRefused, match="one network at a time"):
        import_assets([IMPORT / "pypsa_osm", IMPORT / "pypsa_gridkit"],
                      fmt="pypsa-eur")
    stray = tmp_path / "storage_units.csv"
    stray.write_text("name\n")
    with pytest.raises(ImportRefused, match="not a PyPSA-Eur network table"):
        import_assets([IMPORT / "pypsa_osm", stray], fmt="pypsa-eur")


def test_pypsa_refuses_ragged_csv(tmp_path):
    shutil.copytree(IMPORT / "pypsa_osm", tmp_path / "net")
    with open(tmp_path / "net" / "links.csv", "a") as handle:
        handle.write("a,b\n")
    with pytest.raises(ImportRefused, match="header width"):
        import_assets([tmp_path / "net"], fmt="pypsa-eur")


def test_pypsa_refuses_when_every_row_is_refused(tmp_path):
    net = tmp_path / "net"
    net.mkdir()
    (net / "buses.csv").write_text("bus_id,x,y\n1,999,0\n")
    (net / "lines.csv").write_text("line_id,bus0,bus1,geometry\n"
                                   "1,1,2,'LINESTRING (1 2)'\n")
    (net / "links.csv").write_text("link_id,bus0,bus1\n")
    with pytest.raises(ImportRefused, match="no assets imported") as caught:
        import_assets([net], fmt="pypsa-eur")
    assert "bad WKT geometry" in str(caught.value)


# --------------------------------------------------------------------------
# REPD


def test_repd_import():
    collection = import_assets([IMPORT / "repd_sample.csv"], fmt="repd")
    assets = _by_id(collection)
    assert sorted(assets) == ["repd:90001", "repd:90002", "repd:90003",
                              "repd:90004", "repd:90006", "repd:90008",
                              "repd:90009"]
    wind = assets["repd:90001"]
    assert wind.kind == "plant" and wind.generator_source == "wind"
    assert wind.capacity_mw == 24.0
    assert wind.name == "Synthetic Fferm Wynt Café – Brecon"
    assert wind.operator == "Synthetic Wind Ltd"
    assert wind.license == "OGL-UK-3.0"
    assert wind.hub_height_m is None
    assert wind.tags["repd:turbine_height_m"] == "125"
    assert wind.tags["repd:status"] == "Operational"
    lon, lat = wind.geometry["coordinates"]
    expected_lon, expected_lat = bng_to_wgs84(300000.0, 230000.0)
    assert _distance_m(lon, lat, expected_lon[0], expected_lat[0]) < 0.05
    assert assets["repd:90002"].generator_source == "solar"
    assert assets["repd:90003"].generator_source == "battery"
    assert assets["repd:90004"].tags["offshore"] == "yes"
    assert assets["repd:90008"].generator_source is None
    assert assets["repd:90009"].capacity_mw is None
    ni_lon, ni_lat = assets["repd:90006"].geometry["coordinates"]
    assert -8.3 < ni_lon < -5.3 and 53.9 < ni_lat < 55.5

    record = _record(collection, "repd")
    assert record["encoding"] == "cp1252"
    assert record["license"] == "OGL-UK-3.0"
    assert "Open Government Licence" in record["attribution"]
    assert record["rows_read"] == 10 and record["rows_refused"] == 3
    assert set(record["refusals"]) == {
        "no X/Y coordinates", "no Ref ID",
        "X/Y not on the National Grid for a Northern Ireland row "
        "(likely Irish Grid)"}
    assert record["status_counts"]["Operational"] == 2


def test_repd_refuses_missing_columns(tmp_path):
    path = tmp_path / "repd.csv"
    path.write_text("Ref ID,Site Name,Technology Type\n1,A,Battery\n")
    with pytest.raises(ImportRefused, match="not a REPD extract") as caught:
        import_assets([path], fmt="repd")
    assert "X-coordinate" in str(caught.value)


def test_repd_refuses_kind_and_id_flags():
    with pytest.raises(ImportRefused, match="do not apply"):
        import_assets([IMPORT / "repd_sample.csv"], fmt="repd",
                      kind="plant")


# --------------------------------------------------------------------------
# GeoJSON / CSV


def _unlabelled(tmp_path) -> Path:
    path = tmp_path / "unlabelled.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point",
                                         "coordinates": [-3.0, 51.6]},
         "properties": {"name": "no kind"}}]}))
    return path


def test_geojson_feature_without_kind_is_refused_without_flag():
    collection = import_assets([IMPORT / "geojson_sample.geojson"],
                               fmt="geojson")
    assert len(collection.assets) == 4
    refusals = _record(collection, "geojson:geojson_sample")["refusals"]
    assert refusals["no 'kind' or 'power' property and no --kind"] == 1


def test_geojson_with_no_kind_anywhere_is_refused(tmp_path):
    with pytest.raises(ImportRefused, match="--kind"):
        import_assets([_unlabelled(tmp_path)], fmt="geojson")


def test_geojson_import():
    collection = import_assets([IMPORT / "geojson_sample.geojson"],
                               fmt="geojson", kind="generator")
    assets = _by_id(collection)
    source = "geojson:geojson_sample"
    line = assets[f"{source}/way/1"]
    assert line.kind == "line" and line.voltage_kv == (132.0, 33.0)
    assert line.circuits == 1 and line.cables == 3
    assert line.frequency_hz == 50.0 and line.operator == "Synthetic DNO"
    assert line.tags == {"source": "survey"}
    assert line.source == source and line.source_ref == "way/1"
    assert assets[f"{source}/way/2"].kind == "substation"
    plant = assets[f"{source}/3"]
    assert plant.capacity_mw == 20.0 and plant.generator_source == "wind"
    unlabelled = assets[f"{source}/node/5"]
    assert unlabelled.kind == "generator" and unlabelled.capacity_mw == 5.5
    solar = assets[f"{source}/way/6"]
    assert solar.generator_source == "solar" and solar.capacity_mw == 4.5
    assert solar.voltage_kv == () and solar.tags["voltage"] == "11"
    record = _record(collection, source)
    assert record["refusals"] == {
        "power=pole is not an asset kind (line, minor_line, cable, "
        "substation, plant, generator, tower)": 1}
    assert record["license"] is None
    assert any("ambiguous" in note for note in record["notes"])


def test_geojson_reads_an_assets_document_as_is():
    collection = import_assets([FIXTURES / "assets_wales.geojson"],
                               fmt="geojson")
    assert collection.to_geojson() == load_assets(
        FIXTURES / "assets_wales.geojson").to_geojson()


def test_geojson_contract_mismatch_is_a_row_refusal(tmp_path):
    path = tmp_path / "points.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point",
                                         "coordinates": [0.0, 51.0]},
         "properties": {"power": "line"}},
        {"type": "Feature", "geometry": {"type": "Point",
                                         "coordinates": [0.1, 51.0]},
         "properties": {"power": "tower"}}]}))
    collection = import_assets([path], fmt="geojson")
    assert [a.kind for a in collection.assets] == ["tower"]
    assert _record(collection, "geojson:points")["refusals"] == {
        "fails the assets contract": 1}


def test_csv_import():
    collection = import_assets(
        [IMPORT / "csv_turbines.csv"], fmt="csv", kind="generator",
        column_map={"lat": "latitude", "lon": "longitude",
                    "id": "turbine_ref"})
    assets = _by_id(collection)
    assert sorted(assets) == ["csv:csv_turbines/T1", "csv:csv_turbines/T2",
                              "csv:csv_turbines/T3"]
    t1 = assets["csv:csv_turbines/T1"]
    assert t1.geometry == {"type": "Point", "coordinates": [-3.6402, 51.7001]}
    assert t1.hub_height_m == 80.0 and t1.rotor_diameter_m == 90.0
    assert t1.capacity_mw == 2.5 and t1.name == "Synthetic T1"
    assert "turbine_ref" not in t1.tags and "latitude" not in t1.tags
    record = _record(collection, "csv:csv_turbines")
    assert record["refusal_examples"] == [
        "row 4 (turbine_ref=T4): no latitude/longitude: 'n/a', '-3.48'"]


def test_csv_ids_default_to_row_numbers():
    collection = import_assets(
        [IMPORT / "csv_turbines.csv"], fmt="csv", kind="generator",
        column_map={"lat": "latitude", "lon": "longitude"})
    assert sorted(a.source_ref for a in collection.assets) == ["1", "2", "3"]


@pytest.mark.parametrize("kwargs, match", [
    (dict(kind="generator", column_map={"lat": "latitude"}),
     "--lat-col and --lon-col"),
    (dict(column_map={"lat": "latitude", "lon": "longitude"}), "--kind"),
    (dict(kind="line", column_map={"lat": "latitude", "lon": "longitude"}),
     "import lines as GeoJSON"),
    (dict(kind="generator", column_map={"lat": "lat", "lon": "longitude"}),
     "lacks columns \\['lat'\\]"),
    (dict(kind="generator", column_map={"lat": "latitude", "lon": "longitude",
                                        "id": "ref"}),
     "lacks columns \\['ref'\\]"),
])
def test_csv_refusals(kwargs, match):
    with pytest.raises(ImportRefused, match=match):
        import_assets([IMPORT / "csv_turbines.csv"], fmt="csv", **kwargs)


def test_lat_lon_columns_refused_outside_csv():
    with pytest.raises(ImportRefused, match="--format csv only"):
        import_assets([IMPORT / "geojson_sample.geojson"], fmt="geojson",
                      column_map={"lat": "y", "lon": "x"})


def test_several_csv_files_fold_with_merge(tmp_path):
    other = tmp_path / "more_turbines.csv"
    other.write_text("ref,latitude,longitude\n"
                     "A,51.70012,-3.64021\n"    # ~2 m from T1: duplicate
                     "B,52.5,-3.0\n")
    collection = import_assets(
        [IMPORT / "csv_turbines.csv", other], fmt="csv", kind="generator",
        column_map={"lat": "latitude", "lon": "longitude"})
    assert len(collection.assets) == 4
    folded = _record(collection, "csv:more_turbines")
    assert folded["merges"][0]["duplicates_dropped"] == 1


def test_same_stem_twice_is_refused(tmp_path):
    (tmp_path / "a").mkdir()
    shutil.copy(IMPORT / "csv_turbines.csv", tmp_path / "a" / "csv_turbines.csv")
    with pytest.raises(ImportRefused, match="rename one"):
        import_assets([IMPORT / "csv_turbines.csv",
                       tmp_path / "a" / "csv_turbines.csv"], fmt="csv",
                      kind="generator",
                      column_map={"lat": "latitude", "lon": "longitude"})


# --------------------------------------------------------------------------
# merge


def _point(asset_id, lon, lat, kind="substation", source="a", ref=None):
    return Asset(asset_id=asset_id, kind=kind, source=source, source_ref=ref,
                 geometry={"type": "Point", "coordinates": [lon, lat]})


def _line(asset_id, coords, kind="line", source="a", ref=None):
    return Asset(asset_id=asset_id, kind=kind, source=source, source_ref=ref,
                 geometry={"type": "LineString", "coordinates": coords})


# 0.0001 degree of latitude is about 11.1 m.
_DLAT = 1.0 / 111195.0


def test_merge_drops_same_source_reference_and_asset_id():
    base = AssetCollection([_point("x:1", 0.0, 51.0, source="osm", ref="n1")],
                           sources=[{"source": "osm", "license": "ODbL-1.0"}])
    extra = AssetCollection(
        [_point("x:2", 5.0, 51.0, source="osm", ref="n1"),
         _point("x:1", 6.0, 51.0, source="other"),
         _point("x:3", 7.0, 51.0, source="osm", ref="n3")],
        sources=[{"source": "osm", "license": "ODbL-1.0"}])
    merged, report = merge_with_report(base, extra)
    assert [a.asset_id for a in merged.assets] == ["x:1", "x:3"]
    assert report["duplicates_by_rule"] == {
        "same source and source_ref": 1, "same asset_id": 1}
    assert merged.assets[0].source == "osm"  # base's copy kept
    assert len(merged.sources) == 2
    assert merged.sources[1]["merges"][0]["kept"] == 1


def test_merge_points_within_50_m_of_the_same_kind():
    base = AssetCollection([_point("b:1", -3.0, 51.0)])
    extra = AssetCollection([
        _point("e:near", -3.0, 51.0 + 45 * _DLAT),
        _point("e:far", -3.0, 51.0 + 60 * _DLAT),
        _point("e:other-kind", -3.0, 51.0 + 10 * _DLAT, kind="plant"),
    ])
    merged = merge_collections(base, extra)
    assert [a.asset_id for a in merged.assets] == ["b:1", "e:far",
                                                   "e:other-kind"]


def test_merge_lines_by_endpoints_and_length():
    coords = [[-3.0, 51.0], [-2.99, 51.01], [-2.98, 51.0]]
    base = AssetCollection([_line("b:1", coords)])
    shift = 80 * _DLAT
    reversed_near = [[c[0], c[1] + shift] for c in reversed(coords)]
    longer = [[-3.0, 51.0], [-2.99, 51.02], [-2.98, 51.0]]
    ends_off = [[-3.0, 51.0 + 150 * _DLAT], [-2.99, 51.01], [-2.98, 51.0]]
    extra = AssetCollection([
        _line("e:reversed", reversed_near),
        _line("e:longer", longer),
        _line("e:ends-off", ends_off),
        _line("e:cable", coords, kind="cable"),
    ])
    merged, report = merge_with_report(base, extra)
    assert [a.asset_id for a in merged.assets] == [
        "b:1", "e:longer", "e:ends-off", "e:cable"]
    assert report["duplicates_by_rule"] == {
        "line with matching ends and length": 1}


def test_merge_line_length_tolerance_is_five_percent():
    base = AssetCollection([_line("b:1", [[0.0, 51.0], [0.01, 51.0]])])
    length = _distance_m(0.0, 51.0, 0.01, 51.0)
    # Same ends, a midpoint kink making the line ~4 % and ~6 % longer.
    def kinked(fraction):
        half = length / 2.0
        rise = math.sqrt((half * (1 + fraction)) ** 2 - half ** 2)
        return [[0.0, 51.0], [0.005, 51.0 + rise * _DLAT], [0.01, 51.0]]
    extra = AssetCollection([_line("e:4pc", kinked(0.04)),
                             _line("e:6pc", kinked(0.06))])
    merged = merge_collections(base, extra)
    assert [a.asset_id for a in merged.assets] == ["b:1", "e:6pc"]


def test_merge_into_wales_fixture():
    base = load_assets(FIXTURES / "assets_wales.geojson")
    extra = import_assets([IMPORT / "pypsa_osm"], fmt="pypsa-eur")
    merged, report = merge_with_report(base, extra)
    assert report["duplicates_by_rule"] == {
        "substation point within 50 m": 1,
        "line with matching ends and length": 1}
    assert len(merged.assets) == len(base.assets) + len(extra.assets) - 2
    assert [r["source"] for r in merged.sources] == ["synthetic", "pypsa-eur"]


# --------------------------------------------------------------------------
# CLI


def _run(argv, capsys):
    args = build_parser().parse_args(["energy", "import", *argv])
    code = args.func(args)
    return code, capsys.readouterr()


def test_cli_import_and_merge(tmp_path, capsys):
    out = tmp_path / "assets.geojson"
    code, captured = _run([str(IMPORT / "pypsa_osm"), "--format", "pypsa-eur",
                           "--merge", str(FIXTURES / "assets_wales.geojson"),
                           "-o", str(out)], capsys)
    assert code == 0
    summary = json.loads(captured.out)
    assert summary["assets"] == 23
    assert summary["by_source"] == {"synthetic": 15, "pypsa-eur": 8}
    assert summary["duplicates_dropped"] == 2
    assert summary["geometry_fallbacks"] == {"bus-to-bus": 1}
    assert summary["rows_refused"] == 1
    assert summary["merge"]["kept"] == 8
    written = load_assets(out)
    assert len(written) == 23
    document = json.loads(out.read_text())
    assert document["woof"]["schema"] == ASSETS_SCHEMA


def test_cli_csv_flags(tmp_path, capsys):
    out = tmp_path / "t.geojson"
    code, captured = _run([str(IMPORT / "csv_turbines.csv"), "--format", "csv",
                           "--lat-col", "latitude", "--lon-col", "longitude",
                           "--id-col", "turbine_ref", "--kind", "generator",
                           "-o", str(out)], capsys)
    assert code == 0
    summary = json.loads(captured.out)
    assert summary["by_kind"] == {"generator": 3}
    assert summary["by_source"] == {"csv:csv_turbines": 3}


def test_cli_refusal_exit_code(tmp_path, capsys):
    out = tmp_path / "x.geojson"
    code, captured = _run([str(_unlabelled(tmp_path)),
                           "--format", "geojson", "-o", str(out)], capsys)
    assert code == 2
    assert "--kind" in json.loads(captured.out)["refusal"]
    assert "refused" in captured.err
    assert not out.exists()


def test_cli_refuses_missing_merge_target(tmp_path, capsys):
    code, captured = _run([str(IMPORT / "repd_sample.csv"), "--format", "repd",
                           "--merge", str(tmp_path / "absent.geojson"),
                           "-o", str(tmp_path / "x.geojson")], capsys)
    assert code == 2
    assert "does not exist" in json.loads(captured.out)["refusal"]


def test_written_document_round_trips(tmp_path):
    collection = import_assets([IMPORT / "repd_sample.csv"], fmt="repd")
    path = dump_assets(collection, tmp_path / "r.geojson")
    assert load_assets(path).to_geojson() == collection.to_geojson()
    assert np.isfinite([a.geometry["coordinates"] for a in
                        collection.assets]).all()


def test_pypsa_duplicate_branch_ids_are_row_refusals(tmp_path):
    shutil.copytree(IMPORT / "pypsa_gridkit", tmp_path / "net")
    generators = tmp_path / "net" / "generators.csv"
    lines = generators.read_text().splitlines()
    generators.write_text("\n".join(lines + [lines[1]]) + "\n")
    collection = import_assets([tmp_path / "net"], fmt="pypsa-eur")
    record = _record(collection, "pypsa-eur")
    assert record["refusals"] == {"duplicate id": 1}
    assert len(collection.assets) == 9


def test_csv_refuses_repeated_header_names(tmp_path):
    path = tmp_path / "dupe.csv"
    path.write_text("lat,lon,lat\n51,-3,52\n")
    with pytest.raises(ImportRefused, match="repeats a column"):
        import_assets([path], fmt="csv", kind="tower",
                      column_map={"lat": "lat", "lon": "lon"})


def test_csv_duplicate_ids_share_one_refusal_key(tmp_path):
    path = tmp_path / "dupes.csv"
    path.write_text("ref,lat,lon\nA,51.0,-3.0\nA,51.1,-3.0\nB,51.2,-3.0\n"
                    "B,51.3,-3.0\n")
    collection = import_assets([path], fmt="csv", kind="tower",
                               column_map={"lat": "lat", "lon": "lon",
                                           "id": "ref"})
    assert _record(collection, "csv:dupes")["refusals"] == {"duplicate id": 2}


def test_generic_properties_tags_capacity_and_licence(tmp_path):
    path = tmp_path / "plants.csv"
    path.write_text(
        "lat,lon,capacity_mw,generator:output:electricity,tags,license,"
        "northing\n"
        "51.0,-3.0,,2.5 MW,osm:way/123,ODbL-1.0,1153456.5\n")
    collection = import_assets([path], fmt="csv", kind="plant",
                               column_map={"lat": "lat", "lon": "lon"})
    plant = collection.assets[0]
    assert plant.capacity_mw == 2.5
    assert plant.tags["tags"] == "osm:way/123"
    assert plant.tags["northing"] == "1153456.5"
    assert plant.license == "ODbL-1.0"
    assert "license" not in plant.tags


def test_assets_document_refuses_kind_and_id_flags():
    with pytest.raises(ImportRefused, match="do not apply"):
        import_assets([FIXTURES / "assets_wales.geojson"], fmt="geojson",
                      kind="tower")


def test_repd_grid_references_keep_full_precision(tmp_path):
    source = (IMPORT / "repd_sample.csv").read_bytes().decode("cp1252")
    header, first = source.splitlines()[:2]
    columns = header.split(",")
    # Rewrite the first row's coordinates to a Shetland position.

    row = next(csv.reader([first]))
    row[columns.index("X-coordinate")] = "446123.5"
    row[columns.index("Y-coordinate")] = "1153456"
    row[columns.index("Country")] = "Scotland"
    path = tmp_path / "repd.csv"
    with open(path, "w", newline="", encoding="cp1252") as handle:
        writer = csv.writer(handle)
        writer.writerow(next(csv.reader([header])))
        writer.writerow(row)
    asset = import_assets([path], fmt="repd").assets[0]
    assert asset.tags["repd:easting"] == "446123.5"
    assert asset.tags["repd:northing"] == "1153456"
    lon, lat = asset.geometry["coordinates"]
    assert 60.0 < lat < 60.5 and -1.5 < lon < -1.0
