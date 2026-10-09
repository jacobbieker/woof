"""The ``woof energy`` file contracts read what they write and refuse the rest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from woof.cli import build_parser
from woof.energy.contracts import (
    ASSETS_SCHEMA,
    Asset,
    AssetCollection,
    ContractError,
    EnergyNotImplemented,
    FORECAST_VARIABLES,
    Plan,
    PlanDomain,
    Site,
    SiteSet,
    dump_assets,
    dump_plan,
    dump_sites,
    load_assets,
    load_plan,
    load_sites,
)

FIXTURES = Path(__file__).parent / "fixtures" / "energy"
RING = ((-4.0, 51.6), (-3.9, 51.6), (-3.9, 51.7), (-4.0, 51.7), (-4.0, 51.6))


def _domain(domain_id="d1", **overrides):
    record = dict(domain_id=domain_id, topology="wrf-tiles", role="child",
                  dx_m=100.0, run_dir=f"runs/{domain_id}",
                  output_glob="wrfout_d01_*", footprint=RING,
                  config=f"{domain_id}.toml", grid_id=1)
    record.update(overrides)
    return PlanDomain(**record)


def test_fixture_assets_round_trip(tmp_path):
    collection = load_assets(FIXTURES / "assets_wales.geojson")
    assert len(collection.by_kind("line")) == 3
    assert collection.by_kind("substation")[1].max_voltage_kv == 400.0
    written = dump_assets(collection, tmp_path / "a.geojson")
    assert load_assets(written).to_geojson() == collection.to_geojson()


def test_fixture_sites_round_trip(tmp_path):
    site_set = load_sites(FIXTURES / "sites_wales.json")
    assert {s.kind for s in site_set.sites} >= {"line_sample", "substation",
                                                "turbine", "pv"}
    arrays = site_set.as_arrays()
    assert arrays["lat"].shape == (len(site_set),)
    again = load_sites(dump_sites(site_set, tmp_path / "s.json"))
    assert again.to_json() == site_set.to_json()


def test_assets_refuse_wrong_schema(tmp_path):
    document = json.loads((FIXTURES / "assets_wales.geojson").read_text())
    document["woof"]["schema"] = "woof-energy.assets.v0"
    path = tmp_path / "bad.geojson"
    path.write_text(json.dumps(document))
    with pytest.raises(ContractError, match="assets.v1"):
        load_assets(path)


@pytest.mark.parametrize("change, match", [
    (dict(kind="pylon"), "kind"),
    (dict(geometry={"type": "Point", "coordinates": [0.0, 0.0]}), "must be"),
    (dict(voltage_kv=(-1.0,)), "positive"),
    (dict(geometry={"type": "LineString",
                    "coordinates": [[0.0, 0.0], [200.0, 0.0]]}), "longitude"),
    (dict(generator_source="fusion"), "generator_source"),
    (dict(circuits=0), "circuits"),
])
def test_asset_refusals(change, match):
    record = dict(asset_id="x:1", kind="line", source="x",
                  geometry={"type": "LineString",
                            "coordinates": [[0.0, 0.0], [0.1, 0.1]]})
    record.update(change)
    with pytest.raises(ContractError, match=match):
        Asset(**record)


def test_duplicate_asset_ids_refused():
    asset = Asset(asset_id="x:1", kind="tower", source="x",
                  geometry={"type": "Point", "coordinates": [0.0, 0.0]})
    with pytest.raises(ContractError, match="duplicate"):
        AssetCollection(assets=[asset, asset])


@pytest.mark.parametrize("heights", [(), (0.0,), (30.0, 10.0), (10.0, 10.0)])
def test_site_heights_refused(heights):
    with pytest.raises(ContractError):
        SiteSet(sites=[], heights_m=heights)


def test_site_bearing_range():
    with pytest.raises(ContractError, match="bearing"):
        Site(site_id="s", asset_id="a", kind="line_sample", lat=0.0, lon=0.0,
             bearing_deg=360.0)


def test_plan_round_trip_and_order(tmp_path):
    plan = Plan(topology="wrf-tiles", dx_m=100.0, start="2026-10-01T00",
                hours=24.0, domains=[
                    _domain("tile1", parent="parent", site_ids=("s1",)),
                    _domain("parent", role="parent", dx_m=1000.0),
                ])
    path = dump_plan(plan, tmp_path / "plan.json")
    again = load_plan(path)
    assert again.to_json() == plan.to_json()
    assert [d.domain_id for d in again.run_order()] == ["parent", "tile1"]
    assert again.resolve("runs/tile1") == tmp_path / "runs" / "tile1"


def test_plan_refuses_double_ownership_and_dangling_parent():
    with pytest.raises(ContractError, match="owned by both"):
        Plan(topology="wrf-tiles", dx_m=100.0, start="s", hours=1.0,
             domains=[_domain("a", site_ids=("s1",)),
                      _domain("b", site_ids=("s1",))])
    with pytest.raises(ContractError, match="not in the plan"):
        Plan(topology="wrf-tiles", dx_m=100.0, start="s", hours=1.0,
             domains=[_domain("a", parent="ghost")])


def test_plan_domain_refusals():
    with pytest.raises(ContractError, match="relative"):
        _domain(run_dir="/abs/run")
    with pytest.raises(ContractError, match="config and grid_id"):
        _domain(config=None)
    with pytest.raises(ContractError, match="mesh"):
        _domain(topology="hex-swath", role="mesh", config=None, grid_id=None)
    with pytest.raises(ContractError, match="closed ring"):
        _domain(footprint=RING[:-1])


def test_forecast_table_dims():
    for name, (dims, units, _) in FORECAST_VARIABLES.items():
        assert dims in (("time", "site", "height"), ("time", "site")), name
        assert units


def test_cli_surface_parses():
    parser = build_parser()
    args = parser.parse_args(["energy", "fetch", "--bbox=-3.4,51.6,-3.0,51.8",
                              "-o", "a.geojson"])
    assert args.bbox == (-3.4, 51.6, -3.0, 51.8)
    args = parser.parse_args(["energy", "plan", "sites.json", "--topology",
                              "wrf-tiles", "--dx-m", "50", "-o", "plan"])
    assert args.dx_m == 50.0 and args.topology == "wrf-tiles"
    args = parser.parse_args(["energy", "sites", "a.geojson", "--heights-m",
                              "100,10", "-o", "s.json"])
    assert args.heights_m == (10.0, 100.0)
    with pytest.raises(SystemExit):
        parser.parse_args(["energy", "fetch", "--bbox", "1,2,3", "-o", "x"])
    with pytest.raises(SystemExit):
        parser.parse_args(["energy", "rating", "f.nc", "--products", "magic",
                           "-o", "x"])


def test_schema_constant_matches_fixture():
    document = json.loads((FIXTURES / "assets_wales.geojson").read_text())
    assert document["woof"]["schema"] == ASSETS_SCHEMA


def test_not_implemented_names_the_stage():
    error = EnergyNotImplemented("woof energy fetch (OSM/Overpass)")
    assert "woof energy fetch" in str(error)
