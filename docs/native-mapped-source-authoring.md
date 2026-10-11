# General mapped-source authoring

RW-WPS can now compile an explicit source descriptor and author the exact
input manifest used by the native GRIB1, GRIB2, and NetCDF path. This removes
the need to edit a hash manifest by hand and avoids adding a Python adapter for
every named forecast model.

This is a general *configuration* path, not automatic scientific inference.
The emitted status is `VALIDATED_NOT_STOCK_WRF_CERTIFIED`. A new descriptor
does not inherit the retained ERA5/GFS stock-WRF gates merely because it uses
the same decoder.

## Native preparation and WRF files

Mapped preparation uses the experiment's configured physics. The companion
WRF file export is `optional` by default, for both a single domain and a tree.
A configuration the native runtime accepts can therefore produce its prepared
forecast even when the WRF exporter cannot represent it. The proof records
`REFUSED` with the exporter's reason, and the command prints that outcome.

Use `--stock-wrf-export required` when `wrfinput`/`wrfbdy` are required products.
The same exporter-owned configuration checks run before source decoding or
initialization; prepared arrays still receive their full validation at export.
The hierarchy export retains its existing supported physics slice. Use
`--stock-wrf-export off` (or `--no-stock-wrf-export`) to skip WRF files entirely.
These controls apply to the mapped runner, including packaged source profiles.

Only an explicit unsupported-export result is optional. Failed writes,
corrupt caches, and mismatched physics remain preparation failures. Export
mode does not change the native meteorological arrays or requested physics.
The low-level WRF export API keeps its existing required behavior.

## Authority boundary

`rw-wps.descriptor.v1` has the same coordinate, field, derivation, and target
semantics as executable `rw-wps.mapping.v1`. For GRIB input, each direct
field replaces `selectors` with `vtable_selectors`. Each reference must resolve
exactly one row in the supplied 11-column WPS Vtable. RW-WPS imports only the
numeric GRIB selector from that row. The descriptor remains the authority for:

- canonical field meaning and source/target units;
- axes, rank, staggering, and grid location;
- missing-data and mask policy;
- vertical inventory, ordering, derivations, and cadence;
- soil meaning and target WRF/domain requirements.

A Vtable's mnemonic, unit label, and description are retained in the
authoring receipt but are not treated as those scientific decisions. This is
important because real Vtables reuse names such as `TT` for both 3-D model
levels and 2-m temperature.

## Declarative soil geometry and remapping

The composition contract owns soil semantics independently of a source name.
`soil_layers` now declares:

- canonical temperature and volumetric-moisture fields;
- metre-valued, contiguous, shallow-to-deep source and target layer bounds,
  with the exact temperature and moisture selector bound to every source
  depth;
- either layer-bottom point interpolation to Noah midpoints with explicit
  surface/deep anchors, or conservative layer-mean overlap remapping with
  complete source coverage; and
- a fail-closed missing-data rule: source-land gaps are rejected before
  horizontal interpolation, while ocean soil is repaired after interpolation
  from skin temperature and unit moisture.

The checked ERA5 contract declares its historical WRF `module_soil_pre`
layer-bottom interpolation. The GFS contract declares conservative layer
means; because its four bounds exactly equal Noah's, that route remains a
bit-identical copy. No runtime branch refers to ERA5, GFS, or legacy soil slab
field names. Every GRIB1, GRIB2, and NetCDF selector is checked
position-by-position against the selector stored beside its declared source
depth; GRIB2 type-106 numeric bounds receive an additional semantic check.
The Rust frontend independently repeats the engine's selector, depth, order,
remap, unit, and ocean-policy checks. This incompatible replacement of the
old named-packing document is explicitly versioned as
`gpuwm-mapped-composition-v2`; v1 evidence remains replayable at its sealed
implementation commit but cannot be mistaken for this contract.
Changing the composition bytes intentionally invalidates the retained exact
mapping/composition stock-WRF certificate. The new contracts remain runnable
but are not rebound to that certificate until a fresh unchanged-stock-WRF gate
is recorded, even though focused tests prove byte-identical ERA5/GFS soil
state relative to the retired packing branches.

One GRIB direct field looks like this:

```json
{
  "air_temperature_2m": {
    "vtable_selectors": [{
      "metgrid_name": "TT",
      "grib2_level_type": 103,
      "level1": 2,
      "selector": {"level_value": 2.0}
    }],
    "units": {"source": "K", "target": "K"},
    "source_axes": ["y", "x"],
    "target_axes": ["y", "x"],
    "location": "surface",
    "staggering": "none",
    "missing": {"kind": "reject"}
  }
}
```

The extra explicit `selector.level_value` is required for numeric GRIB2
`Level1` rows because the Vtable does not provide a general physical-unit
contract for GRIB2 scaled fixed surfaces. Bounded layers require the atomic
`second_level_type` and `second_level_value` pair. Zero matches, multiple
matches, reused rows, overlapping broad/narrow direct selectors, hand-authored
GRIB selectors, incomplete surfaces, and unknown keys all fail before a
mapping is written. NetCDF descriptors keep
the existing explicit `name`/`standard_name` selectors and reject Vtables.

GRIB1 references must additionally declare both `selector.center` and
`selector.table_version`; a Vtable does not carry enough authority to infer
them. A numeric GRIB1 `Level1` cannot be changed by an override (a `*` row may
be narrowed explicitly). GRIB2 local-use codes 192--254 require an explicit
four-part Section 1 authority binding: `center`, `subcenter`,
`master_table_version`, and `local_table_version`. Both Rust inventory and
dump tools emit those values, and the Python runtime rejects any disagreement
between them. `local_table_version=255` cannot authorize a local-use code.
Missing/undefined identifier code 255 is also refused for required GRIB1
identifiers and for GRIB2 parameters, categories, disciplines, and second
surfaces.
GRIB2 `level_type=255` is retained only as the explicit no-fixed-surface case
and cannot carry fixed-surface values. Duplicate JSON object keys are rejected
at every nesting level rather than silently taking the last value.

## Create-only authoring

The public command can compile the mapping and create the input manifest in
one author-only call:

```bash
rw-wps --source mapped --source-format grib2 \
  --descriptor /case/source.descriptor.json \
  --vtable /case/Vtable.PRODUCT \
  --author-mapping /case/source.mapping.json \
  --composition /case/source.composition.json \
  --input /data/source-f000.grib2 \
  --input /data/source-f003.grib2 \
  --supplement terrain=/data/terrain-f000.grib2 \
  --supplement terrain=/data/terrain-f003.grib2 \
  --provenance terrain_provenance=/case/terrain-source.md \
  --author-input-manifest /case/input-manifest.json \
  --author-only
```

`--input-list FILE` says the same thing as the repeated `--input` flag as a
file -- one path per line, UTF-8, in the same deterministic time/file order,
blank lines skipped, nothing interpreted. It exists for command-line length:
Windows caps a whole command line at 32 KB, and a field-per-file source is
hundreds of `--input` flags per prepared state (a real ICON-EU cycle is 251).
Exactly one of the two spellings per invocation; both `rw-wps`/`woof prep`
and the internal mapped runner accept it, so a `--dry-run` preview of a large
run stays pasteable. When the per-file spelling is used and the platform
refuses the internal relaunch's length, the front door retries once through a
temporary list file by itself -- a command line the platform accepts is never
rewritten.

Use the exact role names declared by the composition. For an installed
distribution the launcher discovers its bundled GRIB bridge paths relative to
the installation, not through `PATH`; explicit decoder flags that resolve to
any other file are rejected. Manifest authoring verifies the ELF identity,
usage probe, and the generic GRIB2 tabular ABI before hashing a decoder.
Author-only and dry-run modes do not require a CUDA device; end-to-end native
WRF file production still does.

The same flags may be supplied with the normal WPS geometry, geography,
experiment, and output arguments. In that form RW-WPS authors the create-only
contract, inserts the resulting manifest path and digest into the internal
argv, and immediately runs the native initializer. `--dry-run` is strictly
side-effect free and refuses authoring flags. Use `--author-only` to write the
contract without starting initialization, then use those emitted paths in a
separate `--dry-run` if an argv preview is wanted.

Authoring never overwrites a mapping, receipt, or manifest. Source order and
role partitions are preserved; every resolved file is bound by byte count and
lowercase SHA-256. Files are hashed through stable open handles, the candidate
manifest is checked by the runtime verifier, all authorities are rechecked,
and only then is it published with an atomic no-clobber link. A failed rewrite
therefore cannot replace a prior valid manifest.

The stable-handle checks detect persistent path, identity, metadata, or byte
drift visible during the before/after snapshots. They are not a filesystem
lock and do not claim to defeat a hostile same-user process that can swap an
authority to different bytes only while an external decoder reopens it and
restore the original perfectly before the final snapshot. Production inputs
and bundled decoders should therefore be staged read-only or protected from
concurrent writers while authoring and decoding run.

When mapping and manifest authoring are requested in one call, a manifest
failure removes only the exact mapping and receipt proven to have been created
by that invocation. A corrected retry therefore does not hit a stale partial
transaction.

The generic mapped runner is reported as runnable but not globally
stock-WRF-certified. Machine-readable stock-WRF evidence is keyed to exact
retained mapping/composition SHA-256 pairs; a newly authored pair never
inherits those gates by sharing a decoder or format.

## Which engine reads the bytes

Nothing above depends on it -- a mapping document is the same document
either way -- but it changes what a mapped command needs on the machine.

By default the mapped routes decode through `gpuwm_mapped_engine`, one
executable resolved through the usual bridge ladder
(`WOOF_MAPPED_ENGINE_BIN`, this checkout's `tools/rw_wps/target`,
`libexec/bridges`, the packaged copy, then `~/.woof/bridges`) and held
to the contract marker `gpuwm-mapped-frameset-v1`. It reads GRIB2 in
process, so a GRIB2 mapped command needs no `--grib2-inventory` /
`--grib2-dump` paths at all, and an input manifest seals the ENGINE as
its decoder rather than the subprocess pair.

The engine reads GRIB1, GRIB2 and NetCDF for `decode`, `inspect` and
`compose` alike, cross-source compositions included; NetCDF goes through
the vendored pure-Rust `netcrust` reader in process (chunked,
zlib-compressed NetCDF-4 included; the Met Office global files are read
this way). The Python engine reaches the same bytes through `rw_netcdf`. `gpuwm_mapped_engine capabilities` is the binary's own statement
of what it can do, `woof doctor` prints it, and the parity battery
(`tests/test_mapped_engine_parity.py`) holds the Rust engine to the
Python engine's bytes on every format.

`--mapped-engine python` (or `GPUWM_MAPPED_ENGINE=python`) runs the
Python decode path instead. It is a WORKAROUND for a decode the Rust
engine gets wrong, not a supported mode; manifests sealed against the
subprocess tools replay on it. Naming a decoder tool explicitly selects
it for that run, because those flags pin which binary reads the bytes.

`woof doctor` reports the engine and which one a bare run uses. The
contract, the refusal-class table and the parity gate are in
`docs/dev/decode-vendor-design.md`.

## Current boundary

This slice removes source-family dispatch, manual manifest JSON, and the two
named four-layer soil packings from the mapped path. It does not yet generalize
the terrain-only supplement engine. Curvilinear GRIB grids, vertical nest
refinement, moving nests, and two-way feedback also remain unsupported. A
descriptor must materialize the existing complete canonical source frame and
pass the existing donor-grid, hierarchy, initialization, and export gates.


## Height-based model levels

These keys are read by the mapped engine from a mapping JSON document.
The descriptor authoring path (`rw-wps --author-mapping`) does not yet
accept `interface_levels`, `half_level`, `height_from_interfaces`,
`mass_fraction_rebase`, or `dependency_only`. Author these mappings directly
as JSON and validate them with the mapped input-manifest authoring path.

A GRIB source that publishes pressure on mass levels and heights on their
interfaces uses `vertical.kind = "model_level"`, dimensionless `levels`, and
an ordered `interface_levels` list with one more entry than `levels`.
Direct interface heights use `source_axes` and `target_axes` of
`["half_level", "y", "x"]` and target units `m`. The
`height_from_interfaces` derivation takes that field as `source` and averages
adjacent interfaces into `geopotential_height` on `["vertical", "y", "x"]`.
Every column retains its own heights. Missing interfaces, nonfinite heights,
and crossing layers refuse because they would misplace the mass levels.

Interface heights may declare `time_binding = "cycle_invariant"`; assembly
broadcasts the complete coordinate to each forecast time. Prognostic mass
and soil fields still require valid-time data. Pressure remains a direct
field. No pressure-from-level-number inference is made.

The packaged ICON-D2 route uses these declarations by default: 65 mass levels,
66 invariant height interfaces, pressure, specific humidity, and five
condensate species. The emitted model top is 6000 Pa. These are source-table
entries; the mapped engine does not dispatch on the source name.

`mass_fraction_rebase` converts a declared `source` fraction to a reference
mass with the fractions listed in `exclude` removed: source divided by
(1 minus those fractions). Dependencies must share axes and units `kg kg-1`.
Nonfinite or negative fractions, or a nonpositive reference mass, refuse.
The source table excludes all water species for dry-air condensate mixing
ratios, and only condensate for gas-specific humidity. This preserves the
same dry-air basis when initialization converts humidity to vapor mixing ratio.
DWD notes that the condensate GRIB names are mixing ratios but the values are
specific quantities in its [field documentation](https://www.dwd.de/SharedDocs/downloads/DE/modelldokumentationen/nwv/icon/icon_dbbeschr_aktuell.pdf).

A field that is decoded only so a derivation can read it declares
`"dependency_only": true`. Both engines decode and validate it, pass it to
every derivation that reads it, and leave it out of the frame, so it takes
no space in the compose stream. The packaged ICON-D2 mapping marks its six
raw mass fractions this way, and each frame carries only the humidity and
mixing ratios derived from them. A field the target requires cannot be
`dependency_only`; that mapping is refused because its frames would lack
the field. A field a composition borrows from a donor is published even
when the donor's own mapping marks it `dependency_only`.


## Scalar-time NetCDF, speed/direction winds and screen height

These keys exist for producers that write CF NetCDF the way Iris does,
the Met Office global deterministic model being the first: one variable
per file, the valid time as a 0-d coordinate, winds as speed and
direction, and screen fields at 1.5 m. Both engines read them, and their
output is byte-identical (`tests/test_mapped_netcdf_scalar_time.py`).
An existing mapping that declares none of them decodes exactly as before.

### Scalar time

```json
"time": {
  "kind": "dimension",
  "time_coordinate": "scalar",
  "selector": {"name": ["time"], "standard_name": "time"},
  "units": "seconds since 1970-01-01 00:00:00",
  "calendar": "standard",
  "reference_time_selector": {
    "name": ["forecast_reference_time"],
    "standard_name": "forecast_reference_time"
  }
}
```

- `time_coordinate: "scalar"` is the opt-in, and `"scalar"` is its only
  value. `kind: "scalar"` turns it on as well. A mapping without the
  opt-in still refuses a file whose time has no dimension.
- `selector` must resolve one 0-d variable. Its `units` must equal the
  declared units, as for a dimension time, and each file supplies one
  valid time.
- `reference_time_selector` is optional and is read only beside the
  opt-in. It must resolve one 0-d variable with CF reference-time units.
  - When it is declared, that time is the frame's source cycle, so every
    lead of one run shares one cycle. Same-cycle borrows such as
    `source_cycle_analysis_broadcast` and `cycle_invariant_broadcast` rely
    on that.
  - A valid time earlier than the reference time is refused, and so is a
    reference `calendar` other than standard, gregorian or
    proleptic_gregorian.
  - Files of two runs for one valid time are refused ("NetCDF files mix
    source cycles"). Each file passes its own check, but one frame is
    one run.
  - If the file also carries a 0-d variable with `standard_name`
    `forecast_period`, the valid time must equal reference plus period,
    or the decode refuses. The period must be a whole number of seconds,
    in units of seconds, minutes, hours or days.
  - Without the selector, the valid time is the source cycle, as for
    every dimension-time NetCDF source.
- No field may list `time` in `source_axes` under scalar time.

A composition supplement decoded from a scalar-time file uses the same
contract, because its partition is a copy of the mapping. The Met Office
orography at step 0 is one: its scalar time is the cycle time.

### Wind components from speed and direction

```json
{"name": "u-from-speed", "operation": "eastward_wind_from_speed_direction",
 "speed": "wind_speed", "direction": "wind_from_direction"},
{"name": "v-from-speed", "operation": "northward_wind_from_speed_direction",
 "speed": "wind_speed", "direction": "wind_from_direction"}
```

The direction is meteorological: the degrees the wind blows FROM,
clockwise from north (CF `wind_from_direction`). The two operations
compute `u = -speed * sin(dir)` and `v = -speed * cos(dir)`, with
`dir = degrees * (pi / 180)` in float64. The result is earth-relative on
the source's regular latitude/longitude grid, so no rotation follows.
`load_mapping` refuses any other convention:

- the mapping must be NetCDF, whose decoder admits only regular
  latitude/longitude grids. A GRIB direction on a projected grid can be
  grid-relative, and the derived components would skip the rotation;
- the `direction` field must be directly decoded, in `degree` or
  `degrees` on both sides, with no unit scale or offset;
- every selector of the direction field must pin
  `"attributes": {"standard_name": "wind_from_direction"}`, so the
  decoder checks the variable's own CF identity and not only its name;
- the `speed` field's target units must be `m s-1`, and the two fields
  must share `target_axes`.

At evaluation, a finite direction outside [0, 360] or a negative speed
refuses by count. A missing value stays missing. Mark the speed and
direction fields `"dependency_only": true` when the frame should carry
only the components.

### Screen height

```json
"air_temperature_2m": {"selectors": [...], "screen_height_m": 1.5, ...}
```

`screen_height_m` declares the height a screen-level field was measured
at. It is allowed on directly decoded NetCDF surface (`["y", "x"]`)
fields, in (0, 10] m.

- The value is used as published. A 1.5 m temperature, dew point or
  humidity fills the 2 m slot without adjustment: the 0.5 m difference is
  within the screen layer's own spread, and any adjustment would need a
  stability profile the source does not publish.
- The height is recorded as the source reference `@screen_height_m=1.5`
  on the field. A field derived from it, such as `specific_humidity_2m`
  from a dew point, inherits that reference, so the frame header says
  which height the bytes come from.
- When the variable's CF `coordinates` attribute names a 0-d
  `standard_name = height` coordinate, it must be in `m` and equal the
  declaration, or the decode refuses.

### Pressure levels and the Met Office grid

- The Met Office files give pressure in Pa, highest first, with 37 levels
  for temperature, height and wind and 33 for relative humidity. Declare
  the common levels in `coordinates.vertical.levels` (units `"Pa"`), and
  each file is cut to that ladder by value.
- Latitude ascends from the south and longitude runs from -180 to 180.
- `grid_mapping` names a `latitude_longitude` container with a spherical
  `earth_radius`. That is the supported regular grid, and the radius is
  not read.
- The data are float32, chunked and zlib-compressed. The files declare no
  `_FillValue`, and the step-0 files hold no default-fill values.
- Several files reuse a variable name: `air_temperature` on pressure
  levels and at screen level, and `wind_speed`, `wind_from_direction` and
  `relative_humidity` on pressure levels and at screen or 10 m height.
  Each file is read for every field, so pin the level family through the
  CF `coordinates` attribute, which the selector's `attributes` match
  exactly:
  - pressure-level files: `"forecast_period forecast_reference_time time"`;
  - screen and 10 m files: `"forecast_period forecast_reference_time height time"`.

  Without the pin, the pressure-level selector also resolves the screen
  variable, and the decode refuses on rank or on two fields claiming one
  variable.
