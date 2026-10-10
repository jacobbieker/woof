"""``woof energy sites`` turns assets into forecast sites and counts its drops.

The three :mod:`woof.energy.geometry` functions the builder calls are
replaced here by small local implementations, so these tests exercise the
builder alone and run before (or without) the geometry unit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pytest

from woof.cli import build_parser
from woof.energy import geometry, sites
from woof.energy.contracts import (
    Asset,
    AssetCollection,
    load_assets,
    load_sites,
)
from woof.energy.sites import SitesRefused, build_sites, load_region

FIXTURES = Path(__file__).parent / "fixtures" / "energy"
ASSETS = FIXTURES / "assets_wales.geojson"
R = 6370000.0


# --------------------------------------------------------------------------
# local stand-ins for woof.energy.geometry


def _haversine(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _fake_densify(coords, spacing_m):
    pts = np.asarray(coords, dtype=float)
    seg = _haversine(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1])
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    chain = np.arange(0.0, total, spacing_m)
    if total - chain[-1] > 1e-6:
        chain = np.append(chain, total)
    lon = np.interp(chain, cum, pts[:, 0])
    lat = np.interp(chain, cum, pts[:, 1])
    return lon, lat, chain


def _azimuth(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360.0


def _fake_bearings(lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    a = np.concatenate(([0], np.arange(0, lon.size - 2), [lon.size - 2]))
    b = np.concatenate(([1], np.arange(2, lon.size), [lon.size - 1]))
    return _azimuth(lon[a], lat[a], lon[b], lat[b])


def _in_ring(lon, lat, ring):
    ring = np.asarray(ring, dtype=float)
    inside = np.zeros(lon.shape, dtype=bool)
    for (x1, y1), (x2, y2) in zip(ring[:-1], ring[1:]):
        crosses = (y1 > lat) != (y2 > lat)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (lon < xint)
    return inside


def _fake_pip(lon, lat, polygons):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    out = np.zeros(lon.shape, dtype=bool)
    for polygon in polygons:
        hit = _in_ring(lon, lat, polygon[0])
        for hole in polygon[1:]:
            hit &= ~_in_ring(lon, lat, hole)
        out |= hit
    return out


class _FakePlane:
    """Equirectangular stand-in for ``geometry.LocalProjection``."""

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = float(lat0), float(lon0)
        self.kx = R * math.cos(math.radians(self.lat0)) * math.pi / 180.0
        self.ky = R * math.pi / 180.0

    def forward(self, lon, lat):
        return ((np.asarray(lon) - self.lon0) * self.kx,
                (np.asarray(lat) - self.lat0) * self.ky)

    def inverse(self, x, y):
        return (self.lon0 + np.asarray(x) / self.kx,
                self.lat0 + np.asarray(y) / self.ky)


@pytest.fixture(autouse=True)
def fake_geometry(monkeypatch):
    monkeypatch.setattr(geometry, "local_projection", _FakePlane)
    monkeypatch.setattr(geometry, "densify_polyline", _fake_densify)
    monkeypatch.setattr(geometry, "segment_bearings", _fake_bearings)
    monkeypatch.setattr(geometry, "point_in_polygon", _fake_pip)


@pytest.fixture
def wales() -> AssetCollection:
    return load_assets(ASSETS)


def _line(asset_id, coords, voltage=(132.0,), kind="line"):
    geom = {"type": "LineString", "coordinates": [list(c) for c in coords]}
    return Asset(asset_id=asset_id, kind=kind, geometry=geom, source="test",
                 voltage_kv=voltage)


def _square(lon0, lat0, dlon, dlat):
    return [[lon0, lat0], [lon0 + dlon, lat0], [lon0 + dlon, lat0 + dlat],
            [lon0, lat0 + dlat], [lon0, lat0]]


def _by_kind(site_set, kind):
    return [s for s in site_set.sites if s.kind == kind]


def _length(coords):
    pts = np.asarray(coords, dtype=float)
    return float(_haversine(pts[:-1, 0], pts[:-1, 1],
                            pts[1:, 0], pts[1:, 1]).sum())


# --------------------------------------------------------------------------
# counts and per-kind rules


def test_fixture_counts_per_kind(wales):
    spacing = 1000.0
    site_set = build_sites(wales, spacing_m=spacing)
    counts = site_set.provenance["counts_by_kind"]
    expected_samples = 0
    for asset in wales.by_kind("line", "cable"):
        length = _length(asset.geometry["coordinates"])
        expected_samples += math.ceil(length / spacing) + 1
    # Four linear assets meet at the 400 kV junction; the coincident end
    # samples of lines running near-parallel through it are near-duplicates,
    # the ones that leave at an angle are kept.
    junction_drops = site_set.provenance["drops"]["near_duplicate"]
    assert 1 <= junction_drops <= 3
    assert counts == {"line_sample": expected_samples - junction_drops,
                      "substation": 2, "turbine": 4, "pv": 1}
    # The wind plant is represented by its turbines, towers by request only.
    assert site_set.provenance["drops"]["wind_plant_with_turbines"] == 1
    assert site_set.provenance["drops"]["towers_not_requested"] == 3
    assert site_set.spacing_m == spacing
    assert len({s.site_id for s in site_set.sites}) == len(site_set)


def test_line_sample_spacing_ids_and_voltage(wales):
    site_set = build_sites(wales, spacing_m=500.0)
    samples = [s for s in site_set.sites if s.asset_id == "synthetic:line/2"]
    lon = np.array([s.lon for s in samples])
    lat = np.array([s.lat for s in samples])
    steps = _haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    assert np.allclose(steps[:-1], 500.0, rtol=2e-3)
    assert steps[-1] <= 500.0 + 1.0
    chain = [s.chainage_m for s in samples]
    assert chain == sorted(chain)
    assert all(s.voltage_kv == 400.0 for s in samples)
    assert all(s.site_id == f"synthetic:line/2@0:{s.chainage_m:.0f}"
               for s in samples)
    assert all(0.0 <= s.bearing_deg < 360.0 for s in samples)


def test_bearing_on_east_west_segment():
    east = _line("t:east", [(-4.0, 51.6), (-3.9, 51.6)])
    west = _line("t:west", [(-3.9, 51.7), (-4.0, 51.7)])
    site_set = build_sites(AssetCollection([east, west]), spacing_m=200.0)
    for site in site_set.sites:
        target = 90.0 if site.asset_id == "t:east" else 270.0
        assert site.bearing_deg == pytest.approx(target, abs=0.1)
        assert site.kind == "line_sample"


def test_bearing_normalised_into_range(monkeypatch):
    monkeypatch.setattr(geometry, "segment_bearings",
                        lambda lon, lat: np.full(np.size(lon), 360.0))
    site_set = build_sites(AssetCollection(
        [_line("t:n", [(-4.0, 51.6), (-4.0, 51.61)])]), spacing_m=500.0)
    assert {s.bearing_deg for s in site_set.sites} == {0.0}


def test_multilinestring_parts_are_numbered():
    geom = {"type": "MultiLineString",
            "coordinates": [[[-4.0, 51.6], [-3.99, 51.6]],
                            [[-4.0, 51.7], [-3.99, 51.7]]]}
    cable = Asset(asset_id="t:c", kind="cable", geometry=geom, source="test")
    site_set = build_sites(AssetCollection([cable]), spacing_m=300.0)
    parts = {s.site_id.split("@")[1].split(":")[0] for s in site_set.sites}
    assert parts == {"0", "1"}
    assert site_set.provenance["kept_without_voltage"] == {"cable": 1}


def test_site_id_rounding_collision_is_disambiguated():
    # Chainages 0, 0.4 and 0.8 m all round to ":0" / ":1".
    line = _line("t:short", [(-4.0, 51.6), (-4.0 + 1.2e-5, 51.6)])
    site_set = build_sites(AssetCollection([line]), spacing_m=0.4)
    ids = [s.site_id for s in site_set.sites]
    assert len(ids) == len(set(ids))
    assert any("~" in i for i in ids)


def test_substation_polygon_centroid(wales):
    site_set = build_sites(wales, spacing_m=1000.0)
    poly = next(s for s in site_set.sites
                if s.asset_id == "synthetic:substation/2")
    assert poly.kind == "substation"
    assert poly.lon == pytest.approx(-3.950, abs=1e-6)
    assert poly.lat == pytest.approx(51.668, abs=1e-6)
    assert poly.voltage_kv == 400.0


def test_hub_height_union_and_turbine_attributes(wales):
    site_set = build_sites(wales, spacing_m=1000.0, heights_m=(10.0, 150.0))
    assert site_set.heights_m == (10.0, 90.0, 150.0)
    turbines = _by_kind(site_set, "turbine")
    assert {t.hub_height_m for t in turbines} == {90.0}
    assert {t.capacity_mw for t in turbines} == {3.0}
    assert site_set.provenance["hub_heights_m"] == [90.0]


def test_missing_hub_height_is_counted_not_invented():
    turbine = Asset(asset_id="t:wt", kind="generator", source="test",
                    geometry={"type": "Point", "coordinates": [-4.0, 51.6]},
                    generator_source="wind", capacity_mw=2.0)
    site_set = build_sites(AssetCollection([turbine]), heights_m=(10.0,))
    assert site_set.sites[0].hub_height_m is None
    assert site_set.provenance["turbines_missing_hub_height"] == 1
    assert site_set.heights_m == (10.0,)
    assert any("no hub height" in n for n in site_set.provenance["notes"])


def test_hub_heights_rounded_to_decimetre():
    turbines = [Asset(asset_id=f"t:wt{i}", kind="generator", source="test",
                      geometry={"type": "Point",
                                "coordinates": [-4.0 + 0.01 * i, 51.6]},
                      generator_source="wind", hub_height_m=h)
                for i, h in enumerate((80.04, 79.96, 101.27))]
    site_set = build_sites(AssetCollection(turbines), heights_m=(10.0,))
    assert site_set.heights_m == (10.0, 80.0, 101.3)


def test_wind_plant_without_turbines_gets_plant_site(wales):
    site_set = build_sites(wales, kinds=("plant",))
    assert [s.kind for s in site_set.sites] == ["plant"]
    assert site_set.sites[0].capacity_mw == 12.0


def test_wind_plant_covered_by_turbine_inside_polygon():
    plant = Asset(asset_id="t:wp", kind="plant", source="test",
                  geometry={"type": "Polygon",
                            "coordinates": [_square(-4.0, 51.6, 0.1, 0.1)]},
                  generator_source="wind")
    turbine = Asset(asset_id="t:wt", kind="generator", source="test",
                    geometry={"type": "Point", "coordinates": [-3.95, 51.65]},
                    generator_source="wind", hub_height_m=80.0)
    site_set = build_sites(AssetCollection([plant, turbine]))
    assert [s.kind for s in site_set.sites] == ["turbine"]


def test_solar_plant_with_member_panels_is_not_double_counted():
    plant = Asset(asset_id="t:sp", kind="plant", source="test",
                  geometry={"type": "Polygon",
                            "coordinates": [_square(-4.0, 51.6, 0.02, 0.01)]},
                  generator_source="solar", capacity_mw=50.0)
    panels = [Asset(asset_id=f"t:pn{i}", kind="generator", source="test",
                    geometry={"type": "Polygon",
                              "coordinates": [_square(-3.999 + 0.01 * i,
                                                      51.601, 0.008, 0.008)]},
                    generator_source="solar", capacity_mw=25.0)
              for i in range(2)]
    site_set = build_sites(AssetCollection([plant, *panels]))
    pv = _by_kind(site_set, "pv")
    assert {s.asset_id for s in pv} == {"t:pn0", "t:pn1"}
    assert sum(s.capacity_mw for s in pv) == pytest.approx(50.0)
    assert site_set.provenance["drops"]["plant_with_member_generators"] == 1


def test_region_applies_before_wind_plant_coverage(tmp_path):
    # The region holds the plant centroid (-3.95, 51.65) but not its turbine.
    plant = Asset(asset_id="t:wp", kind="plant", source="test",
                  geometry={"type": "Polygon",
                            "coordinates": [_square(-4.0, 51.6, 0.1, 0.1)]},
                  generator_source="wind", capacity_mw=6.0)
    turbine = Asset(asset_id="t:wt", kind="generator", source="test",
                    geometry={"type": "Point", "coordinates": [-3.91, 51.69]},
                    generator_source="wind", hub_height_m=80.0,
                    tags={"plant": "t:wp"})
    path = tmp_path / "region.geojson"
    path.write_text(json.dumps({"type": "Polygon", "coordinates": [
        _square(-3.96, 51.64, 0.02, 0.02)]}))
    site_set = build_sites(AssetCollection([plant, turbine]),
                           region=load_region(path))
    assert [s.kind for s in site_set.sites] == ["plant"]
    assert site_set.sites[0].asset_id == "t:wp"
    assert site_set.heights_m == (10.0, 30.0, 100.0)


def test_multipolygon_open_ring_is_closed_and_noted():
    ring = _square(-4.0, 51.6, 0.02, 0.01)[:-1]
    gas = Asset(asset_id="t:gas", kind="plant", source="test",
                geometry={"type": "MultiPolygon", "coordinates": [[ring]]},
                generator_source="gas")
    site_set = build_sites(AssetCollection([gas]))
    (site,) = site_set.sites
    assert site.lon == pytest.approx(-3.99, abs=1e-6)
    assert site.lat == pytest.approx(51.605, abs=1e-6)
    assert any("open polygon ring" in n for n in site_set.provenance["notes"])


def test_other_plant_at_centroid():
    gas = Asset(asset_id="t:gas", kind="plant", source="test",
                geometry={"type": "Polygon",
                          "coordinates": [_square(-4.0, 51.6, 0.02, 0.01)]},
                generator_source="gas", capacity_mw=500.0)
    site_set = build_sites(AssetCollection([gas]))
    (site,) = site_set.sites
    assert site.kind == "plant"
    assert site.lon == pytest.approx(-3.99, abs=1e-6)
    assert site.lat == pytest.approx(51.605, abs=1e-6)


def test_pv_centroid_and_grid(wales):
    centroid = _by_kind(build_sites(wales), "pv")
    assert len(centroid) == 1
    assert centroid[0].lon == pytest.approx(-4.71, abs=1e-6)
    assert centroid[0].lat == pytest.approx(51.72, abs=1e-6)
    assert centroid[0].capacity_mw == 8.0

    gridded = build_sites(wales, pv_grid_m=200.0)
    pv = _by_kind(gridded, "pv")
    # The polygon is 0.02 deg x 0.01 deg, about 1380 m x 1110 m.
    assert 25 <= len(pv) <= 50
    assert sum(s.capacity_mw for s in pv) == pytest.approx(8.0)
    assert len({s.capacity_mw for s in pv}) == 1
    lon = np.array([s.lon for s in pv])
    lat = np.array([s.lat for s in pv])
    assert ((lon > -4.72) & (lon < -4.70)).all()
    assert ((lat > 51.715) & (lat < 51.725)).all()
    assert all(s.site_id.startswith("synthetic:generator/5#pv") for s in pv)


def test_pv_grid_coarser_than_polygon_falls_back_to_centroid(monkeypatch,
                                                             wales):
    monkeypatch.setattr(geometry, "point_in_polygon",
                        lambda lon, lat, polys: np.zeros(np.size(lon), bool))
    site_set = build_sites(AssetCollection(wales.by_kind("generator")),
                           pv_grid_m=5000.0)
    pv = _by_kind(site_set, "pv")
    assert len(pv) == 1 and pv[0].site_id == "synthetic:generator/5"
    assert any("smaller than --pv-grid-m" in n
               for n in site_set.provenance["notes"])


def test_towers_take_bearing_from_nearest_line(wales):
    site_set = build_sites(wales, spacing_m=500.0, include_towers=True)
    towers = _by_kind(site_set, "tower")
    assert len(towers) == 3
    assert all(t.bearing_deg is not None for t in towers)
    # Line 3 runs north-east from the junction.
    assert all(0.0 < t.bearing_deg < 90.0 for t in towers)
    assert all(t.voltage_kv == 132.0 for t in towers)


def test_tower_far_from_lines_has_no_bearing():
    tower = Asset(asset_id="t:tw", kind="tower", source="test",
                  geometry={"type": "Point", "coordinates": [-4.0, 51.601]})
    line = _line("t:l", [(-4.1, 51.6), (-3.9, 51.6)])
    site_set = build_sites(AssetCollection([line, tower]),
                           include_towers=True)
    (site,) = _by_kind(site_set, "tower")
    assert site.bearing_deg is None
    assert any("no line within" in n for n in site_set.provenance["notes"])


def test_tower_bearing_and_voltage_come_from_the_same_nearest_line():
    # 11 m east of a N-S 132 kV line, 5 m north of an E-W 400 kV line.
    lat0 = 51.6
    tower = Asset(asset_id="t:tw", kind="tower", source="test",
                  geometry={"type": "Point",
                            "coordinates": [-4.0 + 11.0 / 69_000.0,
                                            lat0 + 5.0 / 111_000.0]})
    ns = _line("t:ns", [(-4.0, 51.59), (-4.0, 51.61)], voltage=(132.0,))
    ew = _line("t:ew", [(-4.01, lat0), (-3.99, lat0)], voltage=(400.0,))
    for order in ([ns, ew, tower], [ew, ns, tower]):
        site_set = build_sites(AssetCollection(order), spacing_m=100.0,
                               include_towers=True)
        (site,) = _by_kind(site_set, "tower")
        assert site.bearing_deg == pytest.approx(90.0, abs=0.5)
        assert site.voltage_kv == 400.0


def test_towers_of_a_dropped_low_voltage_line_are_dropped():
    low = _line("t:11kv", [(-4.0, 51.6), (-3.99, 51.6)], voltage=(11.0,))
    high = _line("t:400kv", [(-4.0, 51.7), (-3.99, 51.7)], voltage=(400.0,))
    on_low = Asset(asset_id="t:tw1", kind="tower", source="test",
                   geometry={"type": "Point", "coordinates": [-3.995, 51.6]})
    on_high = Asset(asset_id="t:tw2", kind="tower", source="test",
                    geometry={"type": "Point",
                              "coordinates": [-3.995, 51.7]})
    site_set = build_sites(AssetCollection([low, high, on_low, on_high]),
                           min_voltage_kv=132.0, include_towers=True)
    towers = _by_kind(site_set, "tower")
    assert [t.asset_id for t in towers] == ["t:tw2"]
    assert towers[0].voltage_kv == 400.0
    assert site_set.provenance["drops"]["below_min_voltage"] == 2


# --------------------------------------------------------------------------
# filters and dedupe


def test_voltage_filter(wales):
    site_set = build_sites(wales, spacing_m=1000.0, min_voltage_kv=200.0)
    kept = {s.asset_id for s in site_set.sites if s.kind == "line_sample"}
    assert kept == {"synthetic:line/1", "synthetic:line/2"}
    assert site_set.provenance["drops"]["below_min_voltage"] == 2
    # Turbines and PV are not voltage filtered.
    assert len(_by_kind(site_set, "turbine")) == 4
    assert len(_by_kind(site_set, "substation")) == 2


def test_voltage_filter_keeps_and_counts_untagged():
    tagged = _line("t:low", [(-4.0, 51.6), (-3.99, 51.6)], voltage=(11.0,))
    untagged = _line("t:none", [(-4.0, 51.7), (-3.99, 51.7)], voltage=())
    site_set = build_sites(AssetCollection([tagged, untagged]),
                           min_voltage_kv=33.0)
    assert {s.asset_id for s in site_set.sites} == {"t:none"}
    assert site_set.provenance["kept_without_voltage"] == {"line": 1}


def test_kinds_filter(wales):
    site_set = build_sites(wales, kinds=("substation", "generator"))
    assert set(site_set.provenance["counts_by_kind"]) == {
        "substation", "turbine", "pv"}


def test_region_filter(tmp_path, wales):
    region = {"type": "Feature", "properties": {},
              "geometry": {"type": "Polygon",
                           "coordinates": [_square(-3.7, 51.65, 0.2, 0.1)]}}
    path = tmp_path / "region.geojson"
    path.write_text(json.dumps(region))
    site_set = build_sites(wales, spacing_m=1000.0, region=load_region(path))
    assert all(-3.7 < s.lon < -3.5 and 51.65 < s.lat < 51.75
               for s in site_set.sites)
    assert len(_by_kind(site_set, "turbine")) == 4
    assert site_set.provenance["drops"]["outside_region"] > 0


def test_region_hole_excludes_sites(tmp_path, wales):
    outer = _square(-3.8, 51.55, 0.3, 0.2)
    hole = _square(-3.65, 51.695, 0.06, 0.035)
    path = tmp_path / "region.geojson"
    path.write_text(json.dumps({"type": "MultiPolygon",
                                "coordinates": [[outer, hole]]}))
    site_set = build_sites(wales, spacing_m=1000.0, region=load_region(path))
    assert not _by_kind(site_set, "turbine")
    assert _by_kind(site_set, "line_sample")


def test_dedupe_parallel_duplicate_ways():
    coords = [(-4.0, 51.6), (-3.98, 51.6)]
    shifted = [(lon, lat + 5.0 / 111_000.0) for lon, lat in coords]
    a = _line("t:a", coords)
    b = _line("t:b", shifted)
    c = _line("t:c", [(lon, lat + 0.01) for lon, lat in coords])
    site_set = build_sites(AssetCollection([a, b, c]), spacing_m=100.0)
    owners = {s.asset_id for s in site_set.sites}
    assert owners == {"t:a", "t:c"}
    assert site_set.provenance["drops"]["near_duplicate"] == \
        len([s for s in site_set.sites if s.asset_id == "t:a"])


def test_dedupe_keeps_crossing_lines():
    ew = _line("t:ew", [(-4.01, 51.6), (-3.99, 51.6)], voltage=(400.0,))
    ns = _line("t:ns", [(-4.0, 51.59), (-4.0, 51.61)], voltage=(132.0,))
    site_set = build_sites(AssetCollection([ew, ns]), spacing_m=100.0)
    assert "near_duplicate" not in site_set.provenance["drops"]
    ns_samples = [s for s in site_set.sites if s.asset_id == "t:ns"]
    lat = sorted(s.lat for s in ns_samples)
    assert max(np.diff(lat)) * 111_000.0 < 101.0


def test_dedupe_never_drops_samples_of_the_same_asset():
    # A tight switchback: the second leg passes within 5 m of the first.
    line = _line("t:zz", [(-4.0, 51.6), (-3.99, 51.6),
                          (-3.99, 51.60005), (-4.0, 51.60005)])
    site_set = build_sites(AssetCollection([line]), spacing_m=100.0)
    assert "near_duplicate" not in site_set.provenance["drops"]


# --------------------------------------------------------------------------
# refusals


def test_empty_after_filtering_is_refused_with_breakdown(wales):
    with pytest.raises(SitesRefused) as raised:
        build_sites(wales, kinds=("tower",))
    message = str(raised.value)
    assert "no sites remain" in message
    assert "towers_not_requested" in message
    assert "kind_filtered" in message


def test_empty_after_region_is_refused(tmp_path, wales):
    path = tmp_path / "far.geojson"
    path.write_text(json.dumps({"type": "Polygon",
                                "coordinates": [_square(10.0, 10.0, 1, 1)]}))
    with pytest.raises(SitesRefused, match="outside_region"):
        build_sites(wales, region=load_region(path))


@pytest.mark.parametrize("content, match", [
    ("not json", "not readable GeoJSON"),
    (json.dumps([1, 2]), "not a GeoJSON object"),
    (json.dumps({"type": "Point", "coordinates": [0, 0]}), "type 'Point'"),
    (json.dumps({"type": "Feature",
                 "geometry": {"type": "LineString",
                              "coordinates": [[0, 0], [1, 1]]}}),
     "not a Polygon"),
    (json.dumps({"type": "Polygon",
                 "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}),
     "closed"),
    (json.dumps({"type": "FeatureCollection", "features": []}),
     "no features"),
])
def test_bad_region_file_refused(tmp_path, content, match):
    path = tmp_path / "region.geojson"
    path.write_text(content)
    with pytest.raises(SitesRefused, match=match):
        load_region(path)


def test_missing_region_file_refused(tmp_path):
    with pytest.raises(SitesRefused, match="does not exist"):
        load_region(tmp_path / "nope.geojson")


def test_pv_grid_too_fine_is_refused(wales):
    with pytest.raises(SitesRefused, match="coarser spacing"):
        build_sites(wales, pv_grid_m=1.0)


@pytest.mark.parametrize("kwargs, match", [
    (dict(spacing_m=0.0), "spacing_m"),
    (dict(pv_grid_m=-1.0), "pv_grid_m"),
    (dict(heights_m=()), "heights_m"),
])
def test_bad_numeric_controls_refused(wales, kwargs, match):
    with pytest.raises(SitesRefused, match=match):
        build_sites(wales, **kwargs)


# --------------------------------------------------------------------------
# CLI


def _args(*argv):
    return build_parser().parse_args(["energy", "sites", *argv])


def test_main_writes_sites_and_summary(tmp_path, capsys):
    out = tmp_path / "out" / "sites.json"
    args = _args(str(ASSETS), "--spacing-m", "1000", "--include-towers",
                 "--pv-grid-m", "400", "-o", str(out))
    assert sites.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["output"] == str(out)
    loaded = load_sites(out)
    assert summary["count"] == len(loaded)
    assert summary["counts_by_kind"]["tower"] == 3
    assert loaded.assets_ref["sha256"]
    assert loaded.provenance["counts_by_kind"] == summary["counts_by_kind"]
    assert 90.0 in loaded.heights_m


def test_main_assets_ref_relative_to_output(tmp_path, capsys):
    assets = tmp_path / "assets.geojson"
    assets.write_text(ASSETS.read_text())
    out = tmp_path / "sites.json"
    assert sites.main(_args(str(assets), "-o", str(out))) == 0
    capsys.readouterr()
    assert load_sites(out).assets_ref["path"] == "assets.geojson"


def test_main_refuses_bad_region(tmp_path):
    bad = tmp_path / "bad.geojson"
    bad.write_text("{")
    args = _args(str(ASSETS), "--region", str(bad), "-o",
                 str(tmp_path / "s.json"))
    with pytest.raises(SitesRefused):
        sites.main(args)
    assert not (tmp_path / "s.json").exists()


def test_main_refusal_exits_two_through_woof_cli(tmp_path, capsys):
    from woof.cli import main as woof_main

    code = woof_main(["energy", "sites", str(ASSETS), "--kinds", "tower",
                      "-o", str(tmp_path / "s.json")])
    assert code == 2
    assert "no sites remain" in capsys.readouterr().err
