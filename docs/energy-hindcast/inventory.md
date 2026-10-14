# Global wind-farm inventory

The wind-farm reanalysis starts from a list of where the wind farms are.
No single public dataset covers the whole world, so `woof energy` reads
several of them and merges them into one `woof-energy.assets.v1`
document. Each asset carries its status, whether it is onshore or
offshore, and the sources and licences it came from. The document then
goes to `woof energy sites` and the planners unchanged.

```text
USWTDB CSV/JSON ─┐
GEM xlsx/CSV ────┤  woof energy import --format …   woof energy inventory merge   woof energy sites
EMODnet GeoJSON ─┼──────────────────────────────▶ assets.v1 ×N ─────────────────▶ inventory.geojson ──▶ sites.json
OSM .osm.pbf ────┤
REPD CSV ────────┘
```

Code: `woof/energy/inventory_global.py` (importers, merge, downloads),
`woof/energy/importers.py` (the `import` front door), `woof/energy/cli.py`.
Tests: `tests/test_energy_hindcast_inventory.py` with synthetic fixtures
under `tests/fixtures/energy/inventory/`.

## Sources

| `--format` | Dataset | Level | Coverage | Licence (`license`) |
|---|---|---|---|---|
| `uswtdb` | U.S. Wind Turbine Database (USGS, LBNL, ACP), CSV or API JSON | turbine | United States | public domain (`public-domain`) |
| `gem` | Global Energy Monitor Global Wind Power Tracker, `.xlsx` or CSV | project phase, merged per location | world, about 10 MW and up (plus the "below threshold" sheet) | CC BY 4.0 (`CC-BY-4.0`) |
| `emodnet` | EMODnet Human Activities offshore wind farms, WFS GeoJSON | farm polygon or point | European seas | CC BY 4.0 (`CC-BY-4.0`) |
| `osm-pbf` | OpenStreetMap planet or regional extract, read with pyosmium | turbine and plant | world, as mapped | ODbL 1.0 (`ODbL-1.0`) |
| `repd` | UK Renewable Energy Planning Database (already supported) | project | UK | OGL v3.0 (`OGL-UK-3.0`) |

An Overpass fetch (`woof energy fetch`) gives the same OSM assets for a
bounded area, with the same `osm:node/…` identifiers. Its documents merge
with `osm-pbf` imports without any special handling.

What each importer reads:

- **USWTDB.** One `generator` Point per row (`case_id`). `t_cap` is in kW
  and becomes `capacity_mw`. `t_hh` becomes `hub_height_m` and `t_rd`
  becomes `rotor_diameter_m`. `p_name` is both the asset name and the
  `woof:project` tag, which the merge uses to link turbines to plants.
  `-9999` means missing. Every row is an installed turbine.
- **GEM.** Columns are matched by name, ignoring case. Required: `Project
  Name`, `Capacity (MW)`, `Latitude`, `Longitude`, `Status`. The
  importer reads every sheet whose header row (one of the first five rows)
  has those columns. Phases that share a `GEM location ID` and the same
  coordinates become one `plant` Point. Its capacity is the sum of the
  phases' capacities, and its status is the most advanced phase status.
  The phases are listed in the `gem:phases` tag. The importer reads
  `.xlsx` files itself, using only the standard library, so it does not
  need openpyxl.
- **EMODnet.** One `plant` per feature, keeping its Polygon, MultiPolygon
  or Point geometry and the `woof:installation=offshore` tag.
  `N_TURBINES` is kept as `woof:turbines`. A `crs` member that is not
  CRS84 or `EPSG:4326` is refused, and so is the latitude-first
  `urn:ogc:def:crs:EPSG::4326`. The importer never swaps axes.
- **OSM `.pbf`.** Reads `power=generator` with `generator:source=wind`
  (turbines) and `power=plant` with `plant:source=wind` (plants). Nodes
  become Points. Closed ways and multipolygon relations become polygons.
  The conversion is the same `element_to_asset` that the Overpass fetch
  uses. If pyosmium is missing, the import is refused:

  ```text
  woof energy import: refused: --format osm-pbf reads .osm.pbf extracts with
  pyosmium, which is not installed in this environment: pip install osmium …
  ```

  Filter the planet down to power features with osmium-tool before
  importing it. If the node-location index does not fit in RAM, put it on
  disk:

  ```bash
  osmium tags-filter planet-latest.osm.pbf nwr/power=generator,plant -o power.osm.pbf
  woof energy import power.osm.pbf --format osm-pbf --osm-index sparse_file_array,/scratch/osm.idx -o osm.geojson
  ```

The GEM tracker is only available through a form on GEM's site.
`woof energy inventory download` fetches the two datasets that have a
plain URL. For GEM and OSM it prints where to get the data:

```bash
woof energy inventory download --dataset uswtdb -o raw/uswtdb.json
woof energy inventory download --dataset emodnet -o raw/emodnet_windfarms.geojson
```

The download URLs are pinned in `DOWNLOADS`. They are not tested, because
tests make no network calls.

## Status

Every wind asset in a merged inventory has a `woof:status` tag:

| `woof:status` | GEM `Status` | EMODnet `STATUS` | REPD `Development Status (short)` |
|---|---|---|---|
| `operating` | operating | Production, Test site | Operational |
| `construction` | construction | Construction | Under Construction |
| `pre-construction` | pre-construction | Approved | Awaiting Construction, Appeal Granted, No Application Required |
| `announced` | announced | Planned | Application Submitted, Revised, Appeal Lodged |
| `shelved` | shelved (an "- inferred …" suffix is dropped) | | |
| `mothballed` | mothballed | | |
| `cancelled` | cancelled | | refused, withdrawn, abandoned, expired |
| `retired` | retired | Dismantled | Decommissioned |
| `unknown` | anything else | anything else | anything else |

USWTDB rows count as `operating`. So do OSM features, whether from a pbf
or an Overpass fetch, because they carry a primary `power=*` tag, which
means the infrastructure exists. An asset that no rule covers is
`unknown`.

By default, the four inventory importers and `inventory merge` keep
`operating` assets only. Use `--status` to widen this:

```bash
woof energy import gem.xlsx --format gem --status operating,construction -o gem.geojson
```

`--status` also works with `--format repd`, but only when given (REPD
imports are otherwise unfiltered, as before). For formats that record no
status (`pypsa-eur`, `geojson`, `csv`), `--status` is refused.

## Merge

```bash
woof energy import raw/uswtdb.json --format uswtdb -o work/uswtdb.geojson
woof energy import raw/gem.xlsx --format gem -o work/gem.geojson
woof energy import raw/emodnet_windfarms.geojson --format emodnet -o work/emodnet.geojson
woof energy import raw/power.osm.pbf --format osm-pbf -o work/osm.geojson
woof energy import raw/repd.csv --format repd -o work/repd.geojson
woof energy inventory merge work/osm.geojson work/uswtdb.geojson work/emodnet.geojson work/repd.geojson work/gem.geojson -o inventory.geojson
woof energy sites inventory.geojson --kinds plant,generator -o sites.json
```

`inventory merge` prints a JSON summary on stdout
(`woof-energy.inventory-summary.v1`). It lists counts per rule, attributes
filled in from other sources, and the result broken down by kind, source,
status and installation. The same summary goes into a final
`woof-energy.inventory-merge` record in the document's `sources`.

The merge is designed to err toward keeping duplicates. A missed duplicate
costs one extra site. A false merge loses a wind farm. The rules, with
their flags and defaults:

| Rule | Flag (default) |
|---|---|
| Only wind `plant`/`generator` assets take part. Other assets are dropped, or passed through unmatched with `--keep-non-wind`. | `--keep-non-wind` |
| Status filter, applied before any matching. | `--status` (operating) |
| An asset is dropped if its `asset_id`, or its `(source, source_ref)`, has already been seen. | |
| Two assets from the same source are never merged. A cluster holds at most one record per source. | |
| Turbines from different sources within this distance are one turbine. Pairs are taken nearest first. | `--turbine-match-m` (50) |
| Plants match when names agree and they lie within this distance. | `--plant-match-m` (5000) |
| Plants match when they lie within this distance and their names do not disagree. | `--plant-near-m` (1000) |
| Names agree at this similarity or above. | `--name-threshold` (0.7) |
| Names disagree below this similarity. | `--name-floor` (0.3) |
| Capacities disagree beyond this relative difference. A capacity disagreement vetoes any plant match. | `--capacity-tolerance` (0.25) |
| Onshore and offshore assets never match. | |
| Order of preference for each cluster's winner, after polygons (a polygon outline beats a point). | `--prefer` (osm,uswtdb,emodnet,repd,gem) |

Plant distance is measured between centroids. It is zero when one plant's
centroid lies inside the other's polygon. Rule-1 pairs (names agree) are
merged before rule-2 pairs (close together).

Names are compared after normalising: lower case, accents stripped, number
words and Roman numerals turned into digits, compass words turned into
direction codes. Digits, directions and "extension" are distinguishing
tokens. If two names have different distinguishing tokens, they disagree
outright: "Hornsea 1" and "Hornsea 2", "Walney" and "Walney Extension",
"North Hoyle" and "South Hoyle". Generic words ("wind", "farm", "park",
"parque", "eolico", "windpark", "offshore", …; see `NAME_STOPWORDS`) are
dropped. The remaining words are scored as the larger of two measures:
the token Jaccard index, and the `difflib` ratio of the sorted tokens.

The winner of each cluster keeps its `asset_id`, kind and geometry. Its
empty `name`, `operator`, `capacity_mw`, `hub_height_m` and
`rotor_diameter_m` are filled from the other assets in preference order.
Their tags are added wherever the winner has no tag of that name. This is
how an OSM turbine with no attributes picks up the USWTDB hub height, and
how an OSM plant outline picks up GEM's capacity.

### Turbine-level over plant-level

Next, each merged turbine is offered to nearby wind plants and assigned to
the best match:

1. a plant whose polygon contains the turbine;
2. a plant within `--plant-link-m` (10 km) whose name agrees with the
   turbine's `woof:project` name;
3. a plant within `--plant-link-near-m` (2 km), provided the project name
   does not disagree.

A plant is **turbine-level** when its linked turbines cover at least
`--turbine-coverage` (0.8) of its capacity. A turbine with no capacity
counts as `COVERAGE_TURBINE_MW` (2 MW). A plant with no capacity is
covered only when it states a `woof:turbines` count and its linked
turbines reach that fraction of it. With neither a capacity nor a count,
coverage cannot be shown, so the plant keeps its site. For a
turbine-level plant:

- each linked turbine gets `tags["plant"] = <plant asset_id>`;
- the plant gets `woof:turbine_level=yes` and `woof:turbines_linked`.

`woof energy sites` already reads `tags["plant"]`. It drops the plant's
own site and does not count its capacity a second time: the plant appears
under `drops.wind_plant_with_turbines`, and its turbines become `turbine`
sites at hub height.

Any other plant with linked turbines records `woof:turbines_partial=<n>`
and keeps its own `plant` site. Its turbines keep their own sites too.
`woof energy sites` treats this tag as final, even for a polygon plant
with turbines inside it, which it would otherwise drop. If a turbine's
`plant` tag names a record that now belongs to another cluster, the tag
is re-pointed at that cluster's winner.

### Provenance and licences

Every wind asset in the output carries:

| Tag | Content |
|---|---|
| `woof:provenance` | compact JSON list of `{"source", "ref", "asset_id", "license"}`, winner first |
| `woof:licenses` | `;`-joined sorted licences of every contributing record |
| `woof:status` | see above |
| `woof:installation` | `onshore` / `offshore` (only when a source says) |
| `woof:project` | turbine project name (USWTDB `p_name`) |
| `woof:start_year` | commissioning year (USWTDB `p_year`, GEM `Start Year`, EMODnet `YEAR`) |
| `woof:turbines` | turbine count a plant source states (EMODnet `N_TURBINES`) |
| `woof:country` | country as the source spells it |
| `woof:turbine_level`, `woof:turbines_linked`, `woof:turbines_partial` | the turbine-level decision |

The asset's `license` field becomes the SPDX `AND` of every contributing
licence, for example `CC-BY-4.0 AND ODbL-1.0`. When you redistribute a
derived product, honour all of them: ODbL share-alike for anything
derived from OSM, and CC BY attribution for GEM and EMODnet. The
attribution strings are in the per-source records in the document's
`sources`.

Merging a merged document again extends `woof:provenance` instead of
starting it over. Two rules make this work:

- a single-record input asset that already appears in another asset's
  provenance is dropped as already merged;
- assets whose provenance lists the same record are joined first, for
  example two earlier merges that both absorbed GEM record `gem:L1`;
- a cluster tracks every record in its members' provenance, and two
  clusters join only when every source they share names the same
  records.

So adding a fresh GEM import to last month's inventory updates the
existing clusters and does not create duplicates. Merging a document with
itself is a no-op: every asset is dropped as a duplicate `asset_id`. All of this lives in string tags,
so the document is plain `woof-energy.assets.v1`, and older readers load
it unchanged.

## Limits

- GEM points are often approximate (`gem:location_accuracy`). A GEM plant
  more than 1 km from its counterpart in another source, with a name the
  normaliser cannot reconcile, stays a separate plant. The result is a
  duplicate site, not a lost farm.
- Phases of one project that GEM stores under different location IDs stay
  separate plants.
- `woof energy sites` adds the turbines' hub heights to the set-wide
  `heights_m` rounded to `--hub-height-step-m` (default 10 m), so a world
  inventory's hundred-odd distinct heights become about a dozen output
  heights. Each site keeps its exact `hub_height_m`, which the products
  interpolate to; `--hub-height-step-m 0` restores the 0.1 m rounding.
- `woof energy inventory download --dataset uswtdb` saves whatever the
  USGS API serves (CSV as of 2026-10); the importer reads the content, so the
  file name does not matter. `--dataset emodnet` asks EMODnet's WFS for
  CRS84; its GeoServer writes longitude-first GeoJSON but labels it
  `urn:ogc:def:crs:EPSG::4326`, so the download relabels it CRS84 and says so.
- Scale: the merge does per-asset bookkeeping with one KD-tree per pass.
  A synthetic world-sized input (300k OSM-like turbines, 70k USWTDB-like
  turbines, 30k GEM-like plants) merged in about 20 s on one core, peaking
  at about 1 GB of RAM. Real data has denser clusters and longer names, so
  expect somewhat more. No gridded work is done here.
