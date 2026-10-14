# Partitioning a worldwide site set into regional clusters

`woof energy plan` emits one parent over all the sites it is given. That
works for one country's grid. It does not work for every public wind farm
in the world: a single parent would have to cover the globe, and the
wrf-tiles planner refuses sites whose shortest longitude arc is wider than
180 degrees. `woof energy partition` cuts a worldwide
`woof-energy.sites.v1` document into regional clusters. Each cluster is then
planned, run and extracted on its own, with its own parent, so clusters can
run on separate GPUs or nodes.

```text
woof energy partition       sites.json      ->  partition.json   woof-energy.partition.v1
woof energy plan-cluster    partition.json  ->  DIR/sites.json   woof-energy.sites.v1 (the cluster's subset)
                                                DIR/plan.json    woof-energy.plan.v1  (+ emitted configs)
```

The code is `woof/energy/partition.py`.

## Commands

```bash
woof energy partition sites_world.json --vram-gib 80 -o partition.json
woof energy partition sites_world.json --linkage-km 30 --max-extent-km 250 --max-sites 20000 --leaf-dx-m 200 --card 80gb -o partition.json
woof energy plan-cluster partition.json --cluster c03a93669ab3aa52a --start 2025-01-01T00 --hours 24 -o clusters/c03a93669ab3aa52a
woof energy plan-cluster partition.json --cluster c03a93669ab3aa52a --dry-run -o clusters/c03a93669ab3aa52a
woof energy run clusters/c03a93669ab3aa52a/plan.json --dry-run
```

`woof energy partition` prints a JSON summary on stdout: the cluster count,
the site count, the largest cluster, how many clusters cross the
antimeridian, the recommended topologies, the estimated tile totals and
every `cluster_id`. Notes go to stderr. The document is written atomically:
a `.partial` file is renamed into place.

`woof energy plan-cluster` writes the cluster's sites to `DIR/sites.json`,
then runs the wrf-tiles or wrf-nests planner on them into `DIR`. With
`--dry-run` it writes only `sites.json` and prints the planner call it would
make. A reanalysis orchestrator can fan the cluster ids out across GPUs and
run one `plan-cluster` per cluster.

## How clusters are found

1. **Single linkage on the sphere.** Two sites are linked when their
   great-circle distance on the WRF sphere (radius 6 370 km) is at most
   `--linkage-km` (default 30 km). A cluster is a connected component of
   that graph. Distances are compared as 3-D chords between unit vectors,
   so the antimeridian and the poles need no special handling.
2. **Grid hashing, no O(n²) step.** Sites are binned into cubes whose edge
   is the linkage chord divided by √3, so every pair inside one cube is
   within the chord and is linked without a test. A pair of neighbouring
   cubes is skipped when its point boxes are farther apart than the chord,
   and linked without a test when the boxes are entirely within the chord.
   The remaining cube pairs are tested point to point, cheapest first, on
   the points near the other cube only, and a pair whose cubes are already
   joined is skipped. On a workstation 400 000 synthetic sites (3 000 farms
   worldwide plus 100 000 turbines spread over a Germany-sized box) are
   linked in about 0.5 s and partitioned in about 3 s.
3. **Size caps.** A component is kept as a cluster when it has at most
   `--max-sites` sites (default 20 000), its extent in a local
   azimuthal-equidistant frame about its centre is at most
   `--max-extent-km` (default 250 km) on both axes, its shortest longitude
   arc is at most 180 degrees (the rule the tile planner and `woof domain`
   apply; only sites spread round a pole break it), and a parent chain
   for it fits the card (see below). Otherwise it is cut in two across its
   longer axis. The cut goes at the widest gap between consecutive sites in
   the middle half of the sites, so it falls between wind farms when there
   is a gap between farms. Each half is linked again and the pieces are
   checked in turn.
4. **Determinism.** Sites are put in `site_id` order first, so the
   partition depends on the set of sites, not on their order in the
   document. A `cluster_id` is `c` followed by the first 16 hex digits of
   the sha256 of the cluster's sorted site ids joined by newlines. Clusters
   are written sorted by `cluster_id`. The same sites always give the same
   ids, so ids can name runs, buckets and Icechunk groups.

## The card

`--card`, `--vram-gib` or `--hardware-json` sets the card the clusters are
sized for. If none is given the default is `24gb`, the tile planner's
default. `--hardware-json` reads a target hardware snapshot (the
`arwen.target-sizing.v1` measurement `woof domain --hardware-json` reads)
and uses its total capacity. None of the three probes the GPU of the
machine running the command, so an 80 GB cloud card can be sized from a
laptop. The partition records the card. `plan-cluster` passes the same
card to the planner unless you give `--card`, `--vram-gib` or
`--hardware-json` again.

Grid sizes are estimates. The largest square grid a card can hold is
priced by `woof.core.preflight.estimate_experiment` on a bare `RunConfig`
(leaves at or finer than 250 m get the LES recipe of the tile planner),
against the ceiling `woof downscale` applies. The planner later prices each
tile on the parent's emitted physics suite, which can be heavier.
`estimate_basis` in the document says this.

## `woof-energy.partition.v1` (`partition.json`)

| Field | Meaning |
|---|---|
| `schema` | `woof-energy.partition.v1` |
| `sites` | `{path, sha256}` of the partitioned sites document; `path` is relative to the partition file's directory |
| `parameters` | `linkage_km`, `max_extent_km`, `max_sites`, `leaf_dx_m`, `corridor_km`, `leaf_nz`, `parent_nz`, `step_ratio`, `topology` |
| `capacity` | `vram_gib` and how it was given (`basis`: `--card`, `--vram-gib` or `--hardware-json`, with the card name or the snapshot's device and sha256) |
| `estimate_basis` | How the grid sizes were estimated |
| `count`, `site_count` | Number of clusters and of sites across them |
| `notes` | What was done and what was assumed |
| `clusters` | One record per cluster, sorted by `cluster_id` |

Each cluster record:

| Field | Meaning |
|---|---|
| `cluster_id` | `c` + 16 hex digits of the sha256 of the sorted member ids |
| `site_count`, `site_ids` | The members, sorted |
| `centre` | `{lat, lon}`: the normalised mean of the members' unit vectors |
| `bbox` | `{west, south, east, north, lon_frame}`: the members' box on their shortest longitude arc, whose centre is `lon_frame`, so `west` may be below -180 or `east` above 180 |
| `crosses_antimeridian` | `bbox` reaches past ±180 |
| `region` | The box buffered by `corridor_km`, as GeoJSON in [-180, 180]: a Polygon, or a MultiPolygon split at 180 degrees when the buffered box crosses it (RFC 7946 §3.1.9) |
| `extent_km` | `[x, y]` extent of the members in the local azimuthal-equidistant frame |
| `leaf_dx_m` | Leaf spacing, `--leaf-dx-m` (100 to 300 m, default 300, the cheapest; see [cost.md](cost.md#cheapest-configuration)) |
| `parent_chain` | `{ratios, parent_dx_m, dx_m, parent_cells}`: the shortest chain of ×5 steps (at most 3) whose parent fits the card. The parent is the cluster extent plus the buffer the tile planner adds. `dx_m` runs from the parent to the leaf |
| `projection` | `lambert`, `mercator` or `polar`: what `woof domain` selects for the box's centre latitude |
| `topology` | `wrf-tiles` or `wrf-nests`: the planner recommended for the cluster |
| `estimated_tiles` | `{leaf, intermediate, per_level, leaf_tile_cells}`: how many tiles cover the sites at each level, and the leaf tile's largest side in cells |
| `linkage_component_sites` | Size of the linkage component the cluster was cut from (equal to `site_count` when it was not cut) |

`woof.energy.partition.load_partition` refuses a document whose schema is
wrong, whose `cluster_id` is not the hash of its `site_ids`, in which a site
belongs to two clusters, whose clusters are not sorted, or whose fields are
out of range. Nothing is coerced.

## Topology

With `--topology auto`, the default, a cluster is recommended for
`wrf-nests` when it is at most 25 km across, has at most 500 sites and fits
one estimated leaf tile. That is one wind farm, run in one job on a 3/5
nest ladder from the source (5 km, 1 km, 200 m for a 200 m leaf). Every
other cluster is recommended for `wrf-tiles`: one parent run plus offline
tiles, with no limit on the tile count. `--topology wrf-tiles` or
`--topology wrf-nests` on `partition` recommends that planner for every
cluster. On `plan-cluster` it overrides the recommendation for that one
cluster.

The parent chain recorded in the partition is a suggestion.
`plan-cluster` does not pass it to the planner. The planner chooses its own
chain, trying one ×5 step before 5×5, and the `woof domain` wizard's fit
decides whether a parent fits. Use `--parent-dx-m` to pin the chain.

## Planning a cluster from Python

```python
from woof.energy.partition import load_partition, plan_cluster

partition = load_partition("partition.json")
plan = plan_cluster(partition, "c03a93669ab3aa52a",
                    outdir="clusters/c03a93669ab3aa52a",
                    start="2025-01-01T00", hours=24, vram_gib=80)
```

The keyword arguments go to the planner's `build_plan`. When planning many
clusters, load the partitioned sites once with
`load_partitioned_sites(partition)` (it checks the sha256) and pass them as
`sites=`, so the worldwide document is read and hashed once, not per
cluster. `dx_m` defaults to
the cluster's leaf spacing, `corridor_km` to the partition's, and the card
to the partition's. The cluster's sites keep their order from the
partitioned document. Their provenance gains a `partition` record (path,
sha256 and `cluster_id`), and a relative assets path is rebased onto the
new location. If the partitioned sites document's sha256 no longer matches
the one in the partition, the plan is refused.

## The antimeridian

The tile planner's parent region used to be the sites' plain
longitude/latitude box, and it refused any set spanning more than
180 degrees of longitude, which includes any set across the antimeridian.
The region now spans the sites' shortest longitude arc, the rule
`woof domain` itself applies to `--polygon`. A set across the antimeridian
gets one box, written as a MultiPolygon split at 180 degrees. Only a set
whose shortest arc is wider than 180 degrees is refused, and the refusal
names `woof energy partition`.

## Limits

- A cluster's `region` is a box. For a long diagonal cluster, the parent
  covers more empty ground than a hull would.
- Sites within a few hundred kilometres of a pole are not handled
  specially. `woof domain` refuses a parent that contains a pole.
- Tile counts and parent chains are estimates (see [The card](#the-card)).
  The planner and the wizard decide the real ones.
