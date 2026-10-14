"""``woof energy partition`` and ``plan-cluster`` (woof.energy.partition).

The linkage is held against a brute-force connected-components oracle;
the document against its loader; ``plan-cluster`` against the real
``woof domain`` wizard and the real tile and nest planners, sized with an
explicit card so no GPU is probed.  CPU only, no network.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import random
import shlex
from pathlib import Path

import numpy as np
import pytest

from woof.energy import partition as part
from woof.energy import plan_wrf_tiles as tiles
from woof.energy.contracts import (
    ContractError,
    Site,
    SiteSet,
    dump_sites,
    load_plan,
    load_sites,
    sha256_file,
)
from woof.static.projection import EARTH_RADIUS_M

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "energy"
SITES_WORLD = FIXTURES / "sites_world.json"
GUIDE = REPO / "docs" / "energy-hindcast" / "partition.md"
START = "2025-01-01T00"


def _cli(*argv: str) -> tuple[int, str]:
    from woof.cli import build_parser

    args = build_parser().parse_args(["energy", *argv])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = args.func(args)
    return code, out.getvalue()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The fixture partitioned for an 80 GiB card, through the CLI."""

    root = tmp_path_factory.mktemp("world")
    output = root / "partition.json"
    code, out = _cli("partition", str(SITES_WORLD), "--vram-gib", "80",
                     "-o", str(output))
    assert code == 0
    return part.load_partition(output), json.loads(out), root


def _by_member(partition, prefix: str):
    found = [c for c in partition.clusters
             if any(s.startswith(prefix) for s in c.site_ids)]
    assert found, prefix
    return found


def _sites(points, prefix="s") -> SiteSet:
    return SiteSet(sites=[
        Site(site_id=f"{prefix}{k:05d}", asset_id="a", kind="turbine",
             lat=float(lat), lon=float(lon))
        for k, (lon, lat) in enumerate(points)], heights_m=(100.0,))


# --------------------------------------------------------------------------
# single linkage


def _oracle(lon, lat, linkage_km) -> np.ndarray:
    """Connected components by brute force on great-circle distance."""

    lam, phi = np.radians(lon), np.radians(lat)
    v = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam),
                  np.sin(phi)], axis=1)
    angle = np.arccos(np.clip(v @ v.T, -1.0, 1.0))
    linked = angle * EARTH_RADIUS_M <= linkage_km * 1000.0
    labels = -np.ones(lon.size, dtype=int)
    current = 0
    for seed in range(lon.size):
        if labels[seed] >= 0:
            continue
        frontier = [seed]
        labels[seed] = current
        while frontier:
            node = frontier.pop()
            for other in np.flatnonzero(linked[node] & (labels < 0)):
                labels[other] = current
                frontier.append(int(other))
        current += 1
    return labels


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    pairs = set(zip(a.tolist(), b.tolist()))
    return (len(pairs) == len(set(a.tolist())) == len(set(b.tolist())))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("linkage_km", [5.0, 30.0, 120.0])
def test_linkage_matches_brute_force(seed, linkage_km):
    rng = np.random.default_rng(seed)
    centres = [(179.9, -17.0), (-179.9, -17.1), (0.0, 89.8), (120.0, 89.9),
               (10.0, 50.0), (10.4, 50.1), (-70.0, -45.0), (150.0, 0.0)]
    lon, lat = [], []
    for clon, clat in centres:
        lon.append(clon + rng.normal(0.0, 0.3, 40))
        lat.append(clat + rng.normal(0.0, 0.2, 40))
    lon = (np.concatenate(lon) + 180.0) % 360.0 - 180.0
    lat = np.clip(np.concatenate(lat), -90.0, 90.0)
    labels = part.linkage_labels(lon, lat, linkage_km)
    assert _same_partition(labels, _oracle(lon, lat, linkage_km))
    # numbered by first appearance
    firsts = [int(np.flatnonzero(labels == k)[0]) for k in range(labels.max() + 1)]
    assert firsts == sorted(firsts)


def test_linkage_joins_across_the_antimeridian_and_respects_the_distance():
    one_km_deg = math.degrees(1000.0 / EARTH_RADIUS_M)
    lon = np.array([180.0 - 14.5 * one_km_deg, -180.0 + 14.5 * one_km_deg,
                    -180.0 + 46.0 * one_km_deg])
    lat = np.zeros(3)
    assert part.linkage_labels(lon, lat, 30.0).tolist() == [0, 0, 1]
    assert part.linkage_labels(lon, lat, 31.5).tolist() == [0, 0, 0]


def test_linkage_refuses_a_distance_below_the_hash_floor():
    with pytest.raises(part.PartitionRefused, match="linkage"):
        part.linkage_labels(np.zeros(2), np.zeros(2), 0.01)


def test_linkage_of_nothing_and_of_one():
    assert part.linkage_labels([], [], 30.0).size == 0
    assert part.linkage_labels([5.0], [5.0], 30.0).tolist() == [0]


# --------------------------------------------------------------------------
# the fixture partition


def test_world_partition_summary(world):
    partition, summary, _ = world
    assert summary["schema"] == part.PARTITION_SCHEMA
    assert summary["clusters"] == len(partition.clusters) == 14
    assert summary["sites"] == len(load_sites(SITES_WORLD)) == 415
    assert summary["crossing_antimeridian"] == 1
    assert summary["capacity"] == {"vram_gib": 80.0, "basis": "--vram-gib"}
    assert summary["cluster_ids"] == sorted(summary["cluster_ids"])


def test_every_site_in_exactly_one_cluster(world):
    partition, _, _ = world
    ids = [s for c in partition.clusters for s in c.site_ids]
    assert sorted(ids) == sorted(load_sites(SITES_WORLD).column("site_id"))


def test_dense_cluster_is_linked_as_one(world):
    partition, _, _ = world
    (north_sea,) = _by_member(partition, "synthetic:turbine/northsea")
    assert len(north_sea.site_ids) == 144 + 64
    assert all(s.startswith("synthetic:turbine/northsea")
               for s in north_sea.site_ids)
    assert north_sea.topology == "wrf-tiles"
    assert north_sea.projection == "lambert"
    assert north_sea.parent_chain["ratios"] == [5]
    assert north_sea.parent_chain["dx_m"] == [1000.0, 200.0]
    assert north_sea.estimated_tiles["leaf"] >= 1


def test_antimeridian_cluster_uses_a_local_frame(world):
    partition, _, _ = world
    (fiji,) = _by_member(partition, "synthetic:turbine/fiji")
    assert len(fiji.site_ids) == 12
    assert fiji.crosses_antimeridian
    assert fiji.bbox["west"] < 180.0 < fiji.bbox["east"]
    assert fiji.bbox["east"] - fiji.bbox["west"] < 0.1
    assert fiji.region["type"] == "MultiPolygon"
    lons = [p[0] for poly in fiji.region["coordinates"] for p in poly[0]]
    assert min(lons) == -180.0 and max(lons) == 180.0
    assert fiji.projection == "mercator"
    assert fiji.topology == "wrf-nests"
    assert fiji.extent_km[0] < 10.0


def test_wide_component_is_bisected_between_farms(world):
    partition, _, _ = world
    chain = _by_member(partition, "synthetic:turbine/chain-")
    assert len(chain) == 2
    for cluster in chain:
        assert cluster.linkage_component_sites == 80
        assert max(cluster.extent_km) <= 250.0
        farms = {s.split("#")[0] for s in cluster.site_ids}
        # no farm is cut in two
        assert all(sum(s.startswith(f + "#") for s in cluster.site_ids) == 2
                   for f in farms)


def test_projection_bands(world):
    partition, _, _ = world
    (finnmark,) = _by_member(partition, "synthetic:turbine/finnmark")
    (turkana,) = _by_member(partition, "synthetic:turbine/turkana")
    assert finnmark.projection == "polar"
    assert turkana.projection == "mercator"


def test_cluster_ids_are_member_hashes(world):
    partition, _, _ = world
    for cluster in partition.clusters:
        assert cluster.cluster_id == part.cluster_id_for(
            list(reversed(cluster.site_ids)))
        assert len(cluster.cluster_id) == 1 + part.CLUSTER_ID_HEX


def test_partition_is_independent_of_document_order(world, tmp_path):
    partition, _, _ = world
    sites = load_sites(SITES_WORLD)
    shuffled = list(sites.sites)
    random.Random(7).shuffle(shuffled)
    again = part.build_partition(
        SiteSet(sites=shuffled, heights_m=sites.heights_m),
        sites_ref=partition.sites_ref, vram_gib=80.0)
    assert again.to_json()["clusters"] == partition.to_json()["clusters"]


def test_max_sites_splits_the_dense_cluster():
    sites = load_sites(SITES_WORLD)
    result = part.build_partition(sites, sites_ref={"path": "s", "sha256": "0"},
                                  vram_gib=80.0, max_sites=50)
    north_sea = [c for c in result.clusters
                 if c.site_ids[0].startswith("synthetic:turbine/northsea")]
    assert len(north_sea) >= 5
    assert all(len(c.site_ids) <= 50 for c in result.clusters)
    assert sum(len(c.site_ids) for c in north_sea) == 208
    assert all(c.linkage_component_sites == 208 for c in north_sea)


def test_small_card_needs_a_longer_chain_or_more_tiles():
    sites = load_sites(SITES_WORLD)
    small = part.build_partition(sites, sites_ref={"path": "s", "sha256": "0"},
                                 card="12gb")
    big = part.build_partition(sites, sites_ref={"path": "s", "sha256": "0"},
                               vram_gib=80.0)
    assert small.capacity["basis"] == "--card"
    leaf = {c.cluster_id: c.estimated_tiles["leaf_tile_cells"]
            for c in small.clusters}
    assert all(leaf[c.cluster_id] < c.estimated_tiles["leaf_tile_cells"]
               for c in big.clusters)


def test_topology_can_be_pinned():
    sites = load_sites(SITES_WORLD)
    result = part.build_partition(sites, sites_ref={"path": "s", "sha256": "0"},
                                  vram_gib=80.0, topology="wrf-tiles")
    assert {c.topology for c in result.clusters} == {"wrf-tiles"}
    assert result.parameters["topology"] == "wrf-tiles"


def test_default_card_is_noted():
    result = part.build_partition(_sites([(0.0, 0.0)]),
                                  sites_ref={"path": "s", "sha256": "0"})
    assert result.capacity == {"vram_gib": 24.0, "basis": "--card",
                               "card": tiles.DEFAULT_CARD}
    assert any("nothing probed the local GPU" in n for n in result.notes)


# --------------------------------------------------------------------------
# refusals


@pytest.mark.parametrize("kwargs, match", [
    ({"leaf_dx_m": 50.0}, "leaf-dx-m"),
    ({"leaf_dx_m": 400.0}, "leaf-dx-m"),
    ({"linkage_km": 0.01}, "linkage"),
    ({"max_sites": 0}, "max-sites"),
    ({"max_extent_km": 0.0}, "max-extent-km"),
    ({"topology": "hex-swath"}, "topology"),
    ({"nz": 2}, "nz"),
    ({"card": "24gb", "vram_gib": 24.0}, "choose one"),
    ({"card": "no such card"}, "card"),
    ({"vram_gib": 0.5}, "card"),
])
def test_refusals(kwargs, match):
    with pytest.raises(part.PartitionRefused, match=match):
        part.build_partition(_sites([(0.0, 0.0), (1.0, 1.0)]),
                             sites_ref={"path": "s", "sha256": "0"}, **kwargs)


def test_empty_sites_refused():
    with pytest.raises(part.PartitionRefused, match="no sites"):
        part.build_partition(SiteSet(sites=[], heights_m=(10.0,)),
                             sites_ref={"path": "s", "sha256": "0"})


def test_cli_refusal_exits_2(tmp_path, capsys):
    from woof.cli import build_parser

    args = build_parser().parse_args(
        ["energy", "partition", str(SITES_WORLD), "--leaf-dx-m", "50",
         "-o", str(tmp_path / "p.json")])
    assert args.func(args) == 2
    assert "leaf-dx-m" in capsys.readouterr().err
    assert not (tmp_path / "p.json").exists()


# --------------------------------------------------------------------------
# capacity


def _snapshot(tmp_path, total_gib: int) -> Path:
    path = tmp_path / "hardware.json"
    path.write_text(json.dumps({"sizing": {
        "schema": "arwen.target-sizing.v1", "measured_unix_ms": 1788928327918,
        "total_bytes": total_gib << 30, "free_bytes": (total_gib - 1) << 30,
        "profile": {"name": "NVIDIA H100 80GB HBM3",
                    "multiprocessor_count": 132,
                    "max_threads_per_multiprocessor": 2048,
                    "default_stack_limit_bytes": 1024,
                    "bare_context_bytes": None}}}))
    return path


def test_hardware_json_sizes_without_a_probe(tmp_path):
    path = _snapshot(tmp_path, 80)
    gib, record, note = part.resolve_capacity(hardware_json=path)
    assert gib == 80.0 and note is None
    assert record["basis"] == "--hardware-json"
    assert record["device"] == "NVIDIA H100 80GB HBM3"
    assert record["sha256"] == sha256_file(path)


def test_bad_hardware_json_refused(tmp_path):
    path = tmp_path / "hardware.json"
    path.write_text(json.dumps({"sizing": {"schema": "other"}}))
    with pytest.raises(part.PartitionRefused, match="hardware-json"):
        part.resolve_capacity(hardware_json=path)


def test_planner_capacity_defaults_to_the_partition(world):
    partition, _, _ = world
    assert part.planner_capacity(partition) == {"vram_gib": 80.0}
    assert part.planner_capacity(partition, card="24gb") == {"card": "24gb"}
    assert part.planner_capacity(partition, vram_gib=40.0) == {"vram_gib": 40.0}


# --------------------------------------------------------------------------
# the document


def test_document_round_trips(world, tmp_path):
    partition, _, root = world
    text = (root / "partition.json").read_text()
    again = part.dump_partition(part.load_partition(root / "partition.json"),
                                tmp_path / "again.json")
    assert again.read_text() == text
    assert not (tmp_path / "again.json.partial").exists()
    assert partition.sites_ref["sha256"] == sha256_file(SITES_WORLD)
    assert (root / partition.sites_ref["path"]).resolve() == SITES_WORLD.resolve()


def _tampered(world, tmp_path, edit) -> Path:
    _, _, root = world
    document = json.loads((root / "partition.json").read_text())
    edit(document)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(document))
    return path


@pytest.mark.parametrize("edit, match", [
    (lambda d: d.update(schema="woof-energy.partition.v0"), "schema"),
    (lambda d: d["clusters"][0].update(cluster_id="c0000000000000000"),
     "hash"),
    (lambda d: d["clusters"][1]["site_ids"].append(
        d["clusters"][0]["site_ids"][0]), "site_count"),
    (lambda d: d["clusters"].reverse(), "sorted"),
    (lambda d: d["clusters"][0].update(leaf_dx_m=50.0), "leaf_dx_m"),
    (lambda d: d["clusters"][0].update(topology="hex-swath"), "topology"),
    (lambda d: d["clusters"][0].update(crosses_antimeridian=True),
     "crosses_antimeridian"),
    (lambda d: d["clusters"][0].update(site_count=1), "site_count"),
    (lambda d: d["clusters"][0].pop("region"), "region"),
    (lambda d: d["clusters"][0]["parent_chain"].update(parent_dx_m=999.0),
     "parent_dx_m"),
    (lambda d: d.update(count=99), "count"),
    (lambda d: d["sites"].update(path="/abs/sites.json"), "relative"),
    (lambda d: d["parameters"].pop("linkage_km"), "parameters"),
])
def test_loader_refuses_a_bad_document(world, tmp_path, edit, match):
    path = _tampered(world, tmp_path, edit)
    with pytest.raises(ContractError, match=match):
        part.load_partition(path)


def test_loader_refuses_a_site_in_two_clusters(world, tmp_path):
    def edit(document):
        first, second = document["clusters"][0], document["clusters"][1]
        ids = sorted(set(second["site_ids"]) | {first["site_ids"][0]})
        second.update(site_ids=ids, site_count=len(ids),
                      linkage_component_sites=len(ids),
                      cluster_id=part.cluster_id_for(ids))
        document["clusters"].sort(key=lambda c: c["cluster_id"])

    with pytest.raises(ContractError, match="more than one cluster"):
        part.load_partition(_tampered(world, tmp_path, edit))


# --------------------------------------------------------------------------
# a cluster's sites


def test_cluster_sites_subset(world, tmp_path):
    partition, _, _ = world
    (fiji,) = _by_member(partition, "synthetic:turbine/fiji")
    path = part.write_cluster_sites(partition, fiji.cluster_id,
                                    tmp_path / "c" / "sites.json")
    subset = load_sites(path)
    whole = load_sites(SITES_WORLD)
    order = [s for s in whole.column("site_id") if s in set(fiji.site_ids)]
    assert subset.column("site_id") == order
    assert subset.heights_m == whole.heights_m
    record = subset.provenance["partition"]
    assert record["cluster_id"] == fiji.cluster_id
    assert record["sha256"] == sha256_file(partition.path)
    assert (path.parent / record["path"]).resolve() == partition.path.resolve()
    assert subset.provenance["note"] == whole.provenance["note"]


def test_assets_path_is_rebased(tmp_path):
    (tmp_path / "data").mkdir()
    sites = _sites([(0.0, 0.0), (0.01, 0.0)])
    sites.assets_ref = {"path": "assets.geojson", "sha256": "0"}
    source = dump_sites(sites, tmp_path / "data" / "sites.json")
    partition = part.build_partition(
        sites, sites_ref={"path": "data/sites.json",
                          "sha256": sha256_file(source)}, vram_gib=80.0)
    part.dump_partition(partition, tmp_path / "partition.json")
    out = part.write_cluster_sites(tmp_path / "partition.json",
                                   partition.clusters[0].cluster_id,
                                   tmp_path / "plans" / "x" / "sites.json")
    assert load_sites(out).assets_ref["path"] == "../../data/assets.geojson"


def test_changed_sites_document_refused(world, tmp_path):
    partition, _, _ = world
    copy = tmp_path / "sites.json"
    copy.write_text(SITES_WORLD.read_text() + "\n")
    partition = part.Partition.from_json(
        {**partition.to_json(), "sites": {"path": "sites.json",
                                          "sha256": sha256_file(SITES_WORLD)}},
        path=tmp_path / "partition.json")
    with pytest.raises(ContractError, match="sha256"):
        part.write_cluster_sites(partition, partition.clusters[0].cluster_id,
                                 tmp_path / "out" / "sites.json")


def test_unknown_cluster_refused(world, tmp_path):
    partition, _, _ = world
    with pytest.raises(ContractError, match="no cluster"):
        part.write_cluster_sites(partition, "cnope", tmp_path / "s.json")


# --------------------------------------------------------------------------
# planning a cluster (real woof domain wizard and planners)


def test_plan_cluster_tiles_north_sea(world, tmp_path):
    partition, _, root = world
    (north_sea,) = _by_member(partition, "synthetic:turbine/northsea")
    outdir = tmp_path / "north-sea"
    code, out = _cli("plan-cluster", str(root / "partition.json"),
                     "--cluster", north_sea.cluster_id, "--start", START,
                     "--hours", "6", "-o", str(outdir))
    assert code == 0, out
    record = json.loads(out)
    assert record["topology"] == "wrf-tiles"
    plan = load_plan(outdir / "plan.json")
    assert plan.topology == "wrf-tiles" and plan.dx_m == 200.0
    owned = sorted(s for d in plan.domains for s in d.site_ids)
    assert owned == list(north_sea.site_ids)
    assert plan.sites_ref["path"] == "sites.json"
    assert plan.sites_ref["sha256"] == sha256_file(outdir / "sites.json")
    assert any(north_sea.cluster_id in note for note in plan.notes)
    # the partition's 80 GiB card reached the planner
    tile = next(d for d in plan.domains if d.role == "child")
    args = tile.extra["downscale_args"]
    assert args[args.index("--vram-gib") + 1] == "80"


def test_plan_cluster_across_the_antimeridian(world, tmp_path):
    partition, _, _ = world
    (fiji,) = _by_member(partition, "synthetic:turbine/fiji")
    plan = part.plan_cluster(partition, fiji.cluster_id,
                             outdir=tmp_path / "fiji", topology="wrf-tiles",
                             start=START, hours=6)
    assert load_plan(tmp_path / "fiji" / "plan.json").topology == "wrf-tiles"
    region = json.loads((tmp_path / "fiji" / "parent" / "region.geojson")
                        .read_text())
    assert region["type"] == "MultiPolygon"
    owned = sorted(s for d in plan.domains for s in d.site_ids)
    assert owned == list(fiji.site_ids)


def test_plan_cluster_nests_for_a_small_farm(world, tmp_path):
    partition, _, _ = world
    (texas,) = _by_member(partition, "synthetic:turbine/texas")
    assert texas.topology == "wrf-nests"
    plan = part.plan_cluster(partition, texas.cluster_id,
                             outdir=tmp_path / "texas", start=START, hours=6)
    again = load_plan(tmp_path / "texas" / "plan.json")
    assert again.topology == plan.topology == "wrf-nests"
    assert again.sites_ref["path"] == "sites.json"
    leaves = [d for d in again.domains if d.site_ids]
    assert sorted(s for d in leaves for s in d.site_ids) == list(texas.site_ids)
    assert min(d.dx_m for d in again.domains) == 200.0


def test_plan_cluster_dry_run(world, tmp_path):
    partition, _, root = world
    (chain,) = [c for c in _by_member(partition, "synthetic:turbine/chain-")
                if "synthetic:turbine/chain-00#000" in c.site_ids]
    code, out = _cli("plan-cluster", str(root / "partition.json"),
                     "--cluster", chain.cluster_id, "--dry-run",
                     "--card", "48gb", "-o", str(tmp_path / "dry"))
    assert code == 0
    record = json.loads(out)
    assert record["dry_run"] is True
    assert record["planner_kwargs"]["card"] == "48gb"
    assert record["planner_kwargs"]["dx_m"] == 200.0
    assert len(load_sites(tmp_path / "dry" / "sites.json")) == 40
    assert not (tmp_path / "dry" / "plan.json").exists()


def test_plan_cluster_refusal_exits_2(world, tmp_path):
    _, _, root = world
    code, out = _cli("plan-cluster", str(root / "partition.json"),
                     "--cluster", "cnope", "-o", str(tmp_path / "x"))
    assert code == 2
    assert json.loads(out)["refused"] == "ContractError"


# --------------------------------------------------------------------------
# the tile planner's region


def test_tile_region_spans_the_shortest_arc():
    region = tiles._sites_region(np.array([179.9, -179.9]),
                                 np.array([-17.0, -17.1]))
    assert region["type"] == "MultiPolygon"
    west = region["coordinates"][0][0]
    east = region["coordinates"][1][0]
    assert {p[0] for p in west} == {179.9, 180.0}
    assert sorted({p[0] for p in east}) == [-180.0, pytest.approx(-179.9)]
    plain = tiles._sites_region(np.array([-4.0, -3.0]), np.array([51.0, 52.0]))
    assert plain == {"type": "Polygon", "coordinates": [[
        [-4.0, 51.0], [-3.0, 51.0], [-3.0, 52.0], [-4.0, 52.0], [-4.0, 51.0]]]}


def test_tile_region_refuses_more_than_half_the_globe():
    with pytest.raises(tiles.TilePlanRefusal, match="woof energy partition"):
        tiles._sites_region(np.array([-120.0, 0.0, 120.0]), np.zeros(3))


# --------------------------------------------------------------------------
# CLI and docs parity


def test_cli_defaults_are_the_module_constants():
    from woof.energy import cli

    assert cli._PARTITION_DEFAULTS == {
        "linkage_km": part.DEFAULT_LINKAGE_KM,
        "max_extent_km": part.DEFAULT_MAX_EXTENT_KM,
        "max_sites": part.DEFAULT_MAX_SITES,
        "leaf_dx_m": part.DEFAULT_LEAF_DX_M,
        "corridor_km": part.DEFAULT_CORRIDOR_KM}
    assert cli._PARTITION_TOPOLOGIES == part.TOPOLOGY_CHOICES


def _guide_commands() -> list[str]:
    found, inside = [], False
    for line in GUIDE.read_text("utf-8").splitlines():
        fence = line.strip()
        if fence.startswith("```"):
            inside = (not inside) and fence == "```bash"
            continue
        if inside and fence.startswith("woof energy "):
            found.append(fence)
    return found


def test_the_guide_prints_both_commands():
    stages = {body.split()[2] for body in _guide_commands()}
    assert {"partition", "plan-cluster"} <= stages


@pytest.mark.parametrize("body", _guide_commands())
def test_every_guide_command_parses(body):
    from woof.cli import build_parser

    try:
        build_parser().parse_args(shlex.split(body)[1:])
    except SystemExit as error:
        pytest.fail(f"`{body}` exits {error.code}")


def test_the_guide_names_every_cluster_field(world):
    partition, _, _ = world
    text = GUIDE.read_text("utf-8")
    for key in partition.clusters[0].to_json():
        assert f"`{key}`" in text, key
    for key in partition.to_json():
        assert f"`{key}`" in text, key
    for key in partition.parameters:
        assert f"`{key}`" in text, key


def test_the_guide_cluster_id_is_the_fixture_north_sea(world):
    partition, _, _ = world
    (north_sea,) = _by_member(partition, "synthetic:turbine/northsea")
    assert north_sea.cluster_id in GUIDE.read_text("utf-8")


def test_tile_region_of_one_site_on_the_antimeridian_stays_in_range():
    for lon in (-180.0, 180.0):
        region = tiles._sites_region(np.array([lon]), np.array([10.0]))
        lons = [p[0] for poly in (region["coordinates"]
                                  if region["type"] == "MultiPolygon"
                                  else [region["coordinates"]])
                for p in poly[0]]
        assert all(-180.0 <= v <= 180.0 for v in lons), region


def test_sites_round_a_pole_are_split_until_the_planner_accepts_them():
    points = [(lon, 89.8) for lon in (0.0, 60.0, 120.0, 180.0, -120.0, -60.0)]
    result = part.build_partition(_sites(points),
                                  sites_ref={"path": "s", "sha256": "0"},
                                  vram_gib=80.0, topology="wrf-tiles")
    assert len(result.clusters) > 1
    lon_of = {f"s{k:05d}": lon for k, (lon, _) in enumerate(points)}
    for cluster in result.clusters:
        lons = np.array([lon_of[s] for s in cluster.site_ids])
        tiles._sites_region(lons, np.full(lons.size, 89.8))  # not refused
        assert cluster.bbox["east"] - cluster.bbox["west"] <= 180.0


def test_cluster_sites_reuse_a_loaded_document(world, tmp_path, monkeypatch):
    partition, _, _ = world
    (texas,) = _by_member(partition, "synthetic:turbine/texas")
    sites = part.load_partitioned_sites(partition)

    def no_reload(*_, **__):
        raise AssertionError("the partitioned document was read again")

    monkeypatch.setattr(part, "load_partitioned_sites", no_reload)
    path = part.write_cluster_sites(partition, texas.cluster_id,
                                    tmp_path / "sites.json", sites=sites)
    assert len(load_sites(path)) == len(texas.site_ids)


def test_partition_cli_reports_an_unwritable_output(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("")
    code, _ = _cli("partition", str(SITES_WORLD), "--vram-gib", "80",
                   "-o", str(blocker / "partition.json"))
    assert code == 2
    assert "refused" in capsys.readouterr().err
