# DWD ICON global 13 km forcing

`icon-global` is DWD's deterministic ICON R03B07 pressure-level product, a
nominal 13 km icosahedral mesh of 2,949,120 cells. Aliases are `icon`,
`icon-13km`, `dwd-icon` and `dwd-icon-global`. It is not ICON-EU, not an
ensemble member or mean, and not a claim that ICON itself runs on a
latitude/longitude grid.

Data source: Deutscher Wetterdienst, https://opendata.dwd.de. DWD open data
is published under CC BY 4.0; anything you publish from it carries the
attribution "Source: Deutscher Wetterdienst (DWD), CC BY 4.0". The fetch
manifest, `prep-command.txt` and the profile's provenance document carry
that line too.

## Run one

```sh
woof domain --point 35.18,-97.44 --card "RTX 5070 Ti" --root-dx 3 \
  --hours 6 --source icon-global --cycle latest --out icon-global-case.toml
woof go icon-global-case.toml
```

`woof go` fetches the cycle, normalizes it, prepares the domain, runs the
forecast and renders the products. The two halves also run on their own:

```sh
woof fetch --source icon-global --cycle 2026-10-11T00 --hours 6 --out data/icon
woof prep --source icon-global --input-list data/icon/inputs.txt ...
```

The fetch writes `inputs.txt`, `prep-command.txt` and `prep-arguments.json`,
including the native HSURF supplement binding. Use that handoff rather than
enumerating hundreds of objects by hand. `--area` is refused on this route:
there is no provider-side subset, complete global objects transfer, and the
regional reduction happens on this machine before preparation.

## Where the bytes come from

Two hosts, asked in order (measured 2026-10-11):

| Rung | Host | Holds |
| --- | --- | --- |
| `dwd` | `https://opendata.dwd.de/weather/nwp/v1/m/icon` | the last four runs, about 27 h |
| `source-coop` | `https://data.source.coop/bkr/icon/icon` | runs from 2026-08-25 12 UTC, about a day and a half behind DWD |

Both serve DWD's open-data v1 layout, one plain GRIB2 message (CCSDS
packing) per field, level and step:

```
p/<FIELD>/r/<YYYY-MM-DDTHH:MM>/s/PT<HHH>H00M.grib2
p/<FIELD>/lvt1/<type>/lv1/<level>/r/<YYYY-MM-DDTHH:MM>/s/PT<HHH>H00M.grib2
```

The colon in the run is literal. The level is DWD's own spelling of the
first fixed surface: `lvt1/100` is isobaric in Pa (`85000`), `lvt1/106` is
soil depth in metres (`0.005`; soil water by its layer top, `0.01`). HSURF,
FR_LAND, CLAT and CLON are posted at `PT000H00M` only. The fetch writes
each object under `--out` at its key below `p/`, colon kept, for example
`data/icon/T/lvt1/100/lv1/85000/r/2026-10-11T00:00/s/PT000H00M.grib2`; the
normalization reads that layout, and no two objects share a path.

A run DWD still holds is fetched from DWD alone; an older run goes
straight to the mirror (`--transport source-coop` asks the mirror for any
run). The mirror is never a per-object fall-through behind DWD, because it
holds many runs only in part (on 2026-10-11, two of nine runs listed for an
f000/f003 window were whole) and a part-run from two hosts is not a
forcing series. Before anything is fetched from it the run is listed, and
a run missing any requested object is refused with the count per field and
the missing keys named:

```
fetch icon-global: source-coop (data.source.coop) holds run 2026-08-25T12:00 only in part: 178 of 236 requested objects are missing, so nothing is fetched from it.
  missing by field: RELHUM 36, T 36, U 36, V 36, T_SO 18, W_SO 16
```

Pick another `--cycle` then. A run the mirror has not copied at all (it
runs about a day and a half behind, so a run DWD dropped in the last few
hours is usually not there yet) is refused as not held, and a run before
2026-08-25 12 UTC that DWD no longer holds is refused before anything
moves. Objects fetched from the legacy tree before this change still
prepare: the normalization accepts both layouts. Only the mirror's `p/` tree
is read; its yearly archives are not.

DWD's legacy `/weather/nwp/icon/grib/` tree of bz2-wrapped objects, which
this route read until 2026-10-11, stops after 30 November 2026 (DWD
open-data newsletter, September 2026). It held the same runs as the v1 tree
for the same day, so the route reads v1 alone and keeps no dated era for it.

## What the source gives you

Eighteen pressure levels from 30 to 1000 hPa, surface and 2 m / 10 m state,
the nine-node TERRA soil column with its eight column-mass water layers, sea
ice, snow, land fraction and terrain. Forcing is a uniform three-hour series:
`interval_seconds` is 10800 in the companion `namelist.wps`. The forecast
ceiling is 180 hours from 00 and 12 UTC and 120 hours from 06 and 18 UTC.

Output exists hourly early in the run, but this route publishes the uniform
three-hour series deliberately: the provider's own cadence changes at hour 78,
and an hourly adapter would work until exactly that hour and then stop.

A six-hour request is 352 objects: 116 for each of f000, f003 and f006
plus four invariants, about 1.1 GB.

## How the native grid is handled

ICON's cells are an unstructured mesh. WMO grid-definition template 101 says
so and carries no coordinates: the cell latitudes and longitudes travel in
separate CLAT and CLON records. No structured-grid decoder can read that, so
the source declares a normalization stage that runs before the ordinary
mapped preparation.

The stage is table-driven. Every ICON-specific fact -- the v1 key-path
grammar (`objects.match = "path"`), the cycle grid and cadence, each field's GRIB selector octets and
remap method, the pressure ladder, the soil-depth semantics, the intermediate
window's spacing and envelope -- lives in
`woof/authorities/rw-wps-icon-global-grib2.normalization.json`, a fourth
packaged authority pinned by SHA-256 beside the profile's mapping,
composition and provenance documents. The Python that runs it is generic and
names no model.

The numerical work is the `gdt101_remap` binary in the `grib1_bridge` crate,
which reads any GDT-101 source: the caller passes the mesh identity, the
target window, and the seven GRIB selector octets of every record it wants
read -- the three the plan is built from (latitude, longitude, land
fraction) exactly as much as the fields the plan is applied to. Coordinate
records are numbered in producer-local code tables, so nothing about one
producer's numbering is compiled into the binary. It
verifies the complete grid identity, cell count, originating centre, product
template, reference time and lead of every object against what was asked for,
builds a spherical four-neighbour plan from the same-cycle CLAT, CLON and
FR_LAND records, and writes a regional 0.125-degree GRIB2 intermediate that
covers every projected target-domain corner plus a one-degree halo.

Interpolation is by inverse squared chord distance over four native cell
centres. Surface fields blend only donors of the same land/water class. Land
fraction is nearest-neighbour. Soil takes one nearest native column for every
depth and is explicitly missing on water; the existing TERRA-to-target-soil
contract does the depth and unit conversion. Sea ice is nearest-neighbour on
water and exactly zero on land. A missing donor with positive weight is an
error, never a silently dropped contribution. This is point interpolation,
not conservative cell-overlap remapping.

The intermediate keeps northward rows and Earth-relative winds, and retains
the identification, local metadata and complete product section of the object
it came from; only the horizontal grid and the packing change. Everything
after that is the ordinary mapped route: no new dynamics and no new physics.

The longitude branch may cross the dateline. The window is refused, by name
and with the limit stated, if it would span 180 degrees or more, reach beyond
88 degrees latitude, or exceed two million points. A domain is never quietly
cropped to fit.

## What the stage seals

The normalization cache is keyed on the request: the converter's own hash,
every raw input hash, the target window, and the digests of all four packaged
authorities. Publication is atomic, and a warm cache is re-verified on every
hit -- a changed or missing artifact fails rather than being accepted. The
raw fetch list and the raw objects are never modified.

Two provenance rows reach the prepared tree: the profile's provenance
authority, byte for byte, and the stage's own receipt -- raw inputs,
converter identity, interpolation plan, target geometry and the normalized
output inventory -- under the `icon_global_native_normalization` role.

## Building the native binaries

Two binaries run this route: `gdt101_remap`, which writes the regional
intermediates, and `gpuwm_mapped_engine`, which decodes them. A wheel
install stages both with the rest of the bridges:

```sh
woof fetch-bridges
```

The standalone rw-wps bundle carries `gdt101_remap` too: it is a row in
`BUNDLED_BRIDGES`, so the bundle builds, stages, identity-probes and binds
it the way it does every other bridge, and its launcher exports
`WOOF_GDT101_REMAP` at the staged copy. `rw-wps --source icon-global` needs
nothing built by hand.

From a source checkout, build both:

```sh
(cd tools/grib1_bridge && cargo test --locked --offline --bin gdt101_remap)
(cd tools/grib1_bridge && cargo build --release --locked --offline --bin gdt101_remap)
(cd tools/rw_wps && cargo build --release --locked --offline --bin gpuwm_mapped_engine)
```

Run each build inside its workspace. Cargo reads the vendored-crate
configuration from the directory it runs in, so the same build driven from
the checkout root with `--manifest-path` looks for its crates online and
fails offline.

The resolver finds `tools/grib1_bridge/target/release/` and
`tools/rw_wps/target/release/`. For a build elsewhere, set
`WOOF_GDT101_REMAP` or `WOOF_MAPPED_ENGINE_BIN` to the executable.

Each must carry its contract marker or it is named as a rebuild rather
than used: `arwen.gdt101-regional-remap.v1` for the remapper, and for the
engine a line naming both the frameset it writes and the Section-5
template set it reads. The engine's marker names the template set because
the intermediates are IEEE-packed (template 5.4): an engine built before
this release's decoder passed a frameset-only handshake and then refused a
correct intermediate as "Section 5 simple packing too short", pointing the
user at the raw download. A bridge bundle built before 2.7.5 does not
contain `gdt101_remap` at all, and preparation refuses with the rebuild
remedy rather than falling back.

## Readiness

The registry records this source as runnable: a strict implementation route
exists and a real cycle prepares and runs end to end. WOOF's separate
stock-WRF acceptance status is not claimed for it.

What this route does not do in this release: model-level SLEVE/HHL ingestion,
direct native-cell-to-target interpolation, ICON ensemble products, polar
domains, or runs before the source.coop mirror's first (2026-08-25 12 UTC).
