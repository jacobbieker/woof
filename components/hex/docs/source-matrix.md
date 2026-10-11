# The arbitrary-source proof matrix

Every source registered in the RW-WPS registry (`woof.source_adapters`),
driven with **real bytes** through the whole hex chain:

```
source bytes -> RW-WPS met_intermediate -> woof hex init (native-free) -> 5-composite-step CUDA v8.4.1 forecast, x1.40962
```

one verdict per source: **PASS** with receipts, or the chain's named
refusal, **verbatim, from the first tool that really refused**.  A refusal
naming a capability the *source* does not publish (an AI model with no land
mask) is a correct row, not a failure; a refusal naming a missing table row
names its own remedy.  No verdict below is inferred: every row was executed
on 2026-08-24 and its receipt exists.

**Score: six sources drive the whole chain to a green forecast smoke**
(ERA5, GFS, GDAS, GEFS member, ECMWF IFS open data, ECMWF AIFS, the last
an AI emulator); two AI products refuse at the door for the land state they
genuinely do not publish; four Lambert regional products and the remaining
registry rows refuse by name at their earliest defensible layer.  Each green
source additionally rendered a global T2m/PWAT pair through the render door,
kept with the receipts.

## Instruments of record

| instrument | identity |
|---|---|
| intermediate writer | `met_intermediate`, exe sha256 `b8546fa940004759...`, from the `woof` grid-uniformity work that reached its 2.5.0 release line. This build refuses projected/Gaussian grids by name and carries type-151 ordinal soil; the pre-existing build silently minted Lambert grids as uniform lat-lon and is retired. |
| init door + engine | `woof hex init`, native-free road, at this tree; engine `rw_mpas_init` sha256 `00321d208371147b...`. The exe digests are the instrument identity: a rebuild that moves them is a different instrument. |
| mesh/static pair | `x1.40962`: grid sha256 `9a9e1909a755...` (56,039,332 B), static sha256 `cf1a47d41683...` (94,766,584 B), byte-verified against the registry pins in `src/hexcore/drivers/mpas_mesh_binding.py`. The unified static writer unified-writer re-registration landed for `v15.150.38857` only; x1.40962's registration is unchanged, so these rows ride the current registration (which carries the known antipodal-GWD-band statics defect: a constant of every row, not a source variable). `v15` is excluded: it has a pre-existing step-0 FloatingPointError with any static (bisected upstream, not statics- or source-related). |
| forecast | `tools/run_cuda_v841_forecast_mesh.py --mesh x1.40962`, dt 120 s, 5 composite steps, 2 history frames, `--stop-on-refusal`, WOOF checkout pinned 0d04db71, the proving node (RTX 5070 Ti) |
| vertical | `verification/vertical-specs/tc55-v1.json` (native 55-level tc contract) for every row: the vertical is held constant so the rows vary the SOURCE only.  Capsule mode remains the proven lineage for the x4 native pin (acid test of record, 2026-08-20); at x1 the only green road is native-free, because the sole existing x1 capsule predates the schema-completion work and lacks the init carriers the forecast driver demands. |

Each leg wrote an init provenance, a vertical receipt, a mesh-bind receipt
and a forecast receipt, with one met_intermediate receipt or verbatim refusal
per source. Every verdict quoted in the tables below is reproduced verbatim
here, so the table is readable on its own.

The one authored table this matrix required, `Vtable.ECMWF-OD.rw`, was a
repository asset that this checkout no longer carried.  It is now packaged
at `woof/data/vtables/Vtable.ECMWF-OD.rw`, beside `Vtable.GFS.rw` and
`Vtable.ERA5.rw`, and `woof hex intermediate --source
gfs|gdas|ecmwf-open-data|era5|aifs` (`woof/hex/met_intermediate_door.py`)
now drives `met_intermediate` with them; this matrix's rows were minted by
hand.  The recreated table is read off a 2026-09-24 IFS oper file and is
not byte-identical to the one these rows used.  The writer changed too:
time-processed messages (2 m maximum and minimum temperature) no longer
compete with instantaneous ones for a key, and each matched valid time is
reported.  Its exe digest therefore differs from the instrument recorded
below.

```
woof hex intermediate --source ecmwf-open-data --grib-dir DIR \
  --cycle 2026-09-24T12 --hours 0-12 --out-dir MET
```

DWD ICON global joins the door as two rows, `icon-global` (18 pressure
levels, 1000-30 hPa) and `icon-global-ml` (120 model levels plus the 121
hour-zero HHL half levels; specific humidity aloft, so `--use-spechumd
yes`).  ICON is not lat-lon: it publishes one file per field and level on
an icosahedral GDT-101 mesh, in DWD's v1 tree
`<PARAM>[/lvt1/<type>/lv1/<value>]/r/<YYYY-MM-DDTHH:MM>/s/PT<HHH>H00M.grib2`.
`woof fetch` writes that tree with the run directory spelt
`<YYYY-MM-DDTHHMM>` (`r/2026-10-11T0000/`), because Windows file systems
refuse the colon; the door reads both spellings, so a tree fetched with the
colon still works.
These rows use the `field-files` layout (`woof/hex/gdt101_door.py`): the door
collects every file each lead needs and names the missing ones, takes the
window from the target mesh (`--grid`, its cells and vertices plus a
1-degree halo, snapped to 0.125 degrees), builds one `gdt101_remap` plan from
the hour-zero CLAT, CLON and FR_LAND, remaps each lead into one regular
GRIB2 file, and runs `met_intermediate --map-source dwd` once per lead with
the remapped HSURF and FR_LAND (and, on the model-level row, HHL) as its
`--invariant` file.  The window must be regional (under 180 degrees of
longitude, inside +/-88 degrees, at most two million points); a global mesh
is refused by name and pointed at `gdt101_remap --global --dx-deg 0.125`.
The receipt carries DWD's attribution, "Source: Deutscher Wetterdienst
(DWD), CC BY 4.0".

```
woof hex intermediate --source icon-global --grib-dir DIR \
  --cycle 2026-10-11T00 --hours 0-3 --grid cull.grid.nc --out-dir MET
```

## Met Office global 10 km (`met-office-global`): its own door

The Met Office global deterministic model on AWS open data
(`registry.opendata.aws/met-office-global-deterministic`) is NetCDF, one
file per variable per lead, so it is not a `met_intermediate` row.
`woof/hex/metoffice_intermediate.py` reads it with `netCDF4`, crops it to
the cull and writes the WPS records itself:

```
woof hex intermediate --source met-office-global --input-dir MO \
  --cycle 2026-10-11T00 --hours 0-3 --supplement-dir GDAS \
  --mesh-grid cull.grid.nc --out-dir MET
```

* **The box** comes from `--mesh-grid` (the culled grid, static or init:
  the extent of its cells and vertices), `--from-plan`, `--cull-region`
  with `--halo-km`, or `--point` with `--radius-km`; `--margin-km` widens
  it and three source cells are kept beyond that for the init's stencil.
  Only that window is read, one pressure level at a time, on the source's
  own 0.140625 x 0.09375 degree grid: nothing is regridded.
* **Levels.** Relative humidity is published on 33 levels (to 10 hPa), the
  rest on 37 (to 0.4 hPa).  Only levels carrying all of TT, GHT, RH, UU and
  VV are written, so the column tops at 10 hPa, above the 30 km model top,
  and the init and boundary switches are `--extrap-airtemp lapse-rate`.
  The levels left out are named in the receipt.
* **Conversions.** Wind speed and the direction it blows from become
  `u = -s sin d`, `v = -s cos d`; relative humidity goes from a fraction
  to percent; snow water equivalent (published as a liquid-water thickness
  in metres) becomes kg m-2; the 1.5 m screen temperature and humidity are
  written at the 2 m surface level 200100, the height recorded in the
  receipt.  No PSFC is written: `rw_mpas_init` interpolates surface
  pressure from the isobaric column and `rw_mpas_lbc` reads none.
* **The land state** the product does not publish (soil temperature and
  moisture, LANDSEA, SEAICE) comes from the same-cycle GDAS (`--supplement-source
  gfs` past f009) through `met_intermediate --map-source ncep-gfs` with a
  subset of `Vtable.GFS.rw`.  Those records are cropped on their own
  0.25-degree grid and appended as **declared supplement records**: the
  shared validation relaxes its single-grid rule for them alone, and
  requires them to share one grid among themselves because `rw_mpas_init`
  masks soil and sea ice with LANDSEA index by index.
* **Licence.** British Crown copyright, the Met Office, CC BY-SA 4.0; the
  share-alike term applies to derived outputs.  The receipt carries it
  beside the GDAS attribution.

Measured 2026-10-11 on the 00Z run, leads 0 and 3, over the Greenland
100 m cull (72 N 42 W, 24,133 cells): each intermediate holds 183 records
(173 Met Office on a 16 x 26 window, 10 GDAS on 11 x 19), 34 profile
levels; `woof hex init` (`--nfglevels 34 --nfgsoillevels 4 --use-spechumd
no --extrap-airtemp lapse-rate`) exits 0 using 182 records, and `woof hex
lbc` writes both boundary files.  At 250 hPa the Met Office wind at the
centre is u, v = +15.0, +15.0 m/s against GDAS's +15.6, +15.7.

## Sources with a runnable registry route

| source | bytes driven | verdict |
|---|---|---|
| `era5` | ERA5 GRIB1 pl+sfc, 2025-03-14 12Z (the factory case) | **PASS**: native-free init rc 0 (203 met records, door 248 s), 5-step smoke rc 0 / status `passed` (29.1 s).  Intermediate: the 2026-08-12 mint of record, receipted in `woof`. |
| `gfs` | GFS pgrb2.0p25 f000, 2026-08-12 06Z | **PASS**: init rc 0 (183 met records, door 243 s), smoke rc 0 / `passed` (29.3 s).  Intermediate: the 2026-08-12 mint of record. |
| `gdas` | GDAS pgrb2.0p25 f000, 2026-08-17 06Z | **PASS**: init rc 0 (183 met records, door 244 s), smoke rc 0 / `passed` (29.1 s).  Minted with stock `Vtable.GFS` + label `ncep-gfs`: GDAS is byte-for-byte the GFS catalogue, and the chain measured exactly that. |
| `gefs` | GEFS c00 pgrb2a+pgrb2b f000, 2026-08-17 00Z | **PASS**: init rc 0 (168 met records, door 246 s), smoke rc 0 / `passed` (29.2 s).  Control member's pgrb2a+pgrb2b pair through stock `Vtable.GFSENS` + label `ncep-gefs`; the 32-level ladder only closes with both files of one member. |
| `ecmwf-open-data` | IFS oper 0.25° f000, 2026-08-16 00Z | **PASS**: init rc 0 (98 met records, door 243 s), smoke rc 0 / `passed` (127.7 s).  Authored `Vtable.ECMWF-OD.rw` (type-151 ordinal soil, dewpoint-derived surface RH, SOILGEO terrain), CCSDS packing decodes in-bridge.  Declares `--extrap-airtemp constant`: the product tops at 50 hPa and lapse-rate extrapolation above the first-guess column top is the Fortran's own fatal (`rw_mpas_init: temperature extrapolation above the first-guess column top is not implemented for the lapse-rate mode`), measured, receipt kept. |
| `aifs` | AIFS single 0.25° f000, 2026-08-17 00Z | **PASS**: init rc 0 (79 met records, door 242 s), smoke rc 0 / `passed` (127.4 s).  An AI emulator's output driving the hex dycore: same authored Vtable, two-layer ordinal soil (`--nfgsoillevels 2`), `--use-spechumd yes` (no pressure-level RH published), `--extrap-airtemp constant` (50 hPa top).  Bare-ground/no-snow start is a property of the product, declared in its registry row. |
| `aigfs` | AIGFS NOMADS pres+sfc f000, 2026-08-17 00Z | **refused by design** at the init door (exit 2): *"the intermediate file ... carries no LANDSEA field; coastal soil and sea-ice interpolation would use the wrong surface type. Use a Vtable that emits LANDSEA"*.  The product publishes no land mask, no soil, no skin temperature; its runnable WRF route is the same-cycle GDAS-donor hybrid: a cross-source composition the WPS-intermediate format cannot express.  The atmosphere itself minted fine (55 records). |
| `aigefs` | AIGEFS mem000 pres+sfc f000, 2026-08-17 00Z | **refused by design** at the init door (exit 2): same LANDSEA refusal, verbatim; same donor-composition reasoning (member atmosphere minted, 56 records). |
| `hrrr` | HRRR wrfsfc f02, 2020-03-03 00Z | **refused by design** at the intermediate writer: *"grid is not uniform lat-lon: latitude varies by 0.006987 deg along row 0 (column 1); a cylindrical-equidistant row is a parallel.  Writing this projected grid as iproj 0 would silently mis-georeference every point, so this tool refuses.  Projected products (e.g. Lambert HRRR/RAP/RRFS) are outside the WPS-intermediate chain; use the mapped native route"*.  The port's met reader independently inverts only projection code 0, so the refusal is doubled at the engine boundary. |
| `hrrr-prs` | HRRR wrfprs f00, 2026-08-16 00Z | **refused by design**: same uniformity refusal, measured deviation 0.006987 deg (the 3 km Lambert CONUS grid). |
| `rap` | RAP awip32 f00, 2026-08-17 00Z | **refused by design**: same uniformity refusal, measured deviation 0.104305 deg (AWIPS grid 221, 32 km Lambert).  The JPEG2000 packing decodes fine (openjp2 is in the build); the grid is the boundary. |
| `rrfs` | RRFS prslev f000 (ops), 2026-08-17 00Z | **refused by design**: same uniformity refusal; RRFS rides HRRR's byte-identical Lambert. |
| `icon-eu` | DWD ICON-EU regular lat-lon bz2 set, 2026-08-17 00Z | **refused, table row named**: *"unknown --map-source \"dwd\". known --map-source labels: ncep-gfs ... ncep-gefs ... ncep-cdas-cfsv2 ... ecmwf ... The label is not decoration: it selects the RH over-ice conversion, the masked-surface-field repair, the snow water equivalent and snow depth rules, the land-sea flag conversion and the GRIB1 earth radius"*, the regular-lat-lon grid family is inside the writer's admission.  **This row previously said the remediation was one `KNOWN_MAP_SOURCES` row, a Vtable and bz2 staging.  That was measured on 2026-08-24 and it is not true.**  Those three are real and each is cheap, every semantic the refusal names is an existing `bool` or a two-arm enum (`rh_over_ice: false`, `masked_surface_repair: false`, `snow_water_equivalent_doubled: false`, `landsea_to_flag: true`, and the GRIB1 earth radius is unused because ICON is GRIB2 and the shape-of-earth code is read from the message); the Vtable is plain data like `Vtable.ECMWF-OD.rw`; and bz2 is already a magic-byte dispatch table in two places, so it is operator staging, not a capability.  What blocks the row is a FOURTH piece nobody had named: ICON publishes soil moisture (`W_SO`) as column-integrated layer mass in kg m-2, and the WPS-intermediate writer has **no unit-conversion slot at all**, `FieldMeta.units` is copied into the record header, the Vtable format has no column that transforms a value, and none of the seven `MapSource` fields can select one.  Authoring `SM000001` from `2-3-20` without it would write kg m-2 into a field the consumer reads as a fraction: a structurally perfect intermediate with wrong numbers and exit status 0, which is the exact failure this writer's refusals exist to prevent.  Closing it needs an eleventh rule in `apply_ungrib_rules` plus per-layer bounds carriage: a new mechanism, so it is NOT table work and is not being bandaided in.  Two further facts for whoever takes it: the registry row for `icon-eu` needs no change, because it already declares a runnable mapped route that converts this field correctly (`volumetric_soil_moisture_from_layer_mass`); and `met_intermediate` is not a bundled binary, so any row at all costs a rebuild and a new exe digest in the instruments table above. |
| `icon-global`, `icon-global-ml` | DWD ICON R03B07 v1 open data, 2026-10-11 00Z f000+f003, over the 24,133-cell Greenland 100 m cull (72N 42W) | **door wired, no PASS claimed yet**.  Measured: the remap stage writes a 25 x 19 window at 0.125 degrees (-43.5..-40.5 E, 70.875..73.125 N) from 232 lead records plus 2 invariants on the pressure row in about 25 s, and decodes to plausible ice-sheet values (HSURF 2700-3106 m, surface pressure 670-708 hPa).  The model-level row was remapped over its bottom 20 levels only, for the data budget.  The next tool refuses: *"met_intermediate: unknown --map-source \"dwd\"."*  The `dwd` label and `Vtable.ICON-GLOBAL.rw` / `Vtable.ICON-GLOBAL-ML.rw` arrive with the met_intermediate DWD work (W_SO kg m-2 to volumetric, the eight TERRA SOILT layers); init and LBC follow once they land. |
| `gem-gdps` | GDPS 15 km lat-lon f000, 2026-08-16 00Z | **refused, table row named**: same closed-vocabulary refusal for `eccc-gdps`.  Grid is uniform lat-lon; JPEG2000 decodes; one label row + one Vtable away. |
| `20crv3` | 20CRv3 member 072 pl+sfc, 1932-03-21 00Z | **refused, table row named**: same closed-vocabulary refusal for `ncep-20crv3`.  (If a label existed, the member files' grid would still face the Gaussian-spacing gate (20CRv3's native grid is Gaussian) so this source likely needs iproj-4 writer support as well.) |
| `20crv3-cf` | NOAA PSL NetCDF SI series | **refused by design** at the intermediate writer: *"skt.nc does not begin with a GRIB envelope"*, the WPS-intermediate chain is GRIB-only; 20CRv3's NetCDF form is a mapped-route (rw_netcdf) source. |
| `era5-l137` | ERA5 native 137-level GRIB2 + lnsp, 2026-05-30 | **refused by design**: *"no field in era5-ml-l137-20260530-00z10z.grib, era5-ml-zlnsp-20260530-00z21z.grib matched any of the 29 Vtable rows across 6072 GRIB messages; refusing to write an empty intermediate file"*.  Hybrid model levels are not a Vtable operation: materializing p = A + B·ps and hydrostatic height belongs to the mapped route, which already proves this source to rc 0 for WRF.  Even a perfect ML intermediate would then refuse at the door: the product publishes no soil, no land mask, no surface state. |
| `mapped` | (meta-row) | The generic `rw-wps.mapping.v1` route decodes to canonical frames and exports native WRF inputs directly: it does not emit WPS intermediates, so it does not participate in this chain.  It is the designated road for everything the intermediate writer refuses above. |

## Sources the registry itself refuses (no runnable route; the registry row is the named refusal)

| source | registry status | registry's own words |
|---|---|---|
| `hrrr-ak` | adapter_mapping_required | "Alaska grid/projection and field contract are not yet mapped." |
| `nam` | adapter_mapping_required | (mapping pending) |
| `hgefs` | member_selection_and_mapping_required | "Select a member and combine pressure/surface products; averages are not member states." |
| `hiresw` | member_selection_and_mapping_required | "Select the concrete ARW/FV3 member before initialization." |
| `href` | explicit_composition_required | "Use a constituent deterministic member; spread/probability products cannot initialize WRF." |
| `sref` | explicit_composition_required | "Use a constituent member; the ensemble mean is not a dynamically balanced member state." |
| `rtma` | explicit_composition_required | "Provide a complete 3-D atmosphere source; RTMA may replace declared surface fields only." |
| `urma` | explicit_composition_required | "Provide a complete 3-D atmosphere source; URMA may replace declared surface fields only." |
| `nbm` | explicit_composition_required | "Provide a complete 3-D analysis/forecast state; NBM is postprocessed guidance." |
| `rrfs-a` | adapter_mapping_required | "Frozen prototype feed (noaa-rrfs-pds, halted 2026-08-12 by design); the live operational feed is the `rrfs` row." |
| `rrfs-public` | adapter_mapping_required | (mapping pending) |
| `refs` | explicit_composition_required | "Use a constituent RRFS member; mean/PMMN/spread products cannot initialize a member." |
| `rrfs-firewx` | explicit_composition_required | "Combine the 2-D fire-weather product with a complete pressure/native atmosphere and soil state." |
| `wrf` | wrf_archive_mapping_required | "Map a compatible WRF archive state, vertical coordinate, physics state, and boundary source." |

## What the matrix says about the chain

1. **Uniform lat-lon GRIB sources are table work end to end.**  GDAS, GEFS,
   IFS open data and AIFS had never touched the hex chain before this
   matrix; each cost at most one authored Vtable
   (`Vtable.ECMWF-OD.rw`) and
   zero code on the source axis.
2. **The AI-model refusals are the door working as designed.**  AIGFS and
   AIGEFS publish no land mask and no soil; the door names exactly that.
   Their runnable WRF routes are cross-source hybrids (GDAS donor): a
   composition concept the WPS-intermediate format cannot carry, by design.
3. **Projected (Lambert) regional products are outside this chain at two
   independent layers** (writer and engine reader) and both now refuse by
   name.  Before 2026-08-24 the writer silently mis-georeferenced them; the
   uniformity gate closed that.
4. **The map-source closed vocabulary is the current boundary of
   "arbitrary".**  ICON-EU, GDPS and 20CRv3 refuse at one named table
   (`KNOWN_MAP_SOURCES`) whose refusal text names the file and the facts a
   new row needs.
5. **The native-free init road is real.**  Every PASS row below minted its
   init from grid + static + a versioned JSON vertical declaration and real
   meteorology: no native MPAS file anywhere in the chain.
