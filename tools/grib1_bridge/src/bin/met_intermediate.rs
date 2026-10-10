//! `met_intermediate` -- decode GRIB to the WPS version-5 intermediate
//! format, in Rust, with no Fortran `ungrib` in the path.
//!
//! The intermediate format is the seam every downstream preprocessing
//! stage already speaks: WPS `metgrid` reads it, and MPAS's
//! `init_atmosphere` reads it directly (`mpas_init_atm_read_met.F`).
//! Reproducing it exactly is therefore the smallest change that removes
//! `ungrib.exe` from a real-data initialization without asking anything
//! downstream to move.
//!
//! What this tool is NOT: it is not a horizontal or vertical
//! interpolator, and it does not know what a mesh is.  It decodes,
//! selects by Vtable, converts units, applies the repairs `ungrib`'s
//! `rrpr` applies, derives the fields `rrpr` derives, and writes
//! records.  That is precisely `ungrib`'s job description.
//!
//! Field selection is driven by the same Vtable files `ungrib` reads, so
//! a user's existing table is the interface -- no new vocabulary, and no
//! table baked into this binary.
//!
//! The producing centre is NOT free text and has no default.  Which
//! repairs run is a function of it, and every one of them is the
//! difference between a right number and a wrong one, so an unknown
//! label is a refusal.  See `MapSource`.
//!
//! usage: met_intermediate --vtable VTABLE --date YYYY-MM-DD_HH:MM:SS
//!            --map-source LABEL --out OUTFILE [--pmin PA]
//!            [--select-valid-time] [--invariant GRIB --invariant-field NAME ...]
//!            GRIB [GRIB ...]
//!
//! Time identity.  Every matched message's valid time is computed from
//! its own reference time and lead and reported in the receipt
//! (`matched_valid_times`).  `--select-valid-time` additionally drops
//! every message that is not valid at `--date`, which is what a file
//! carrying a whole time series (an ERA5 combined retrieval) needs:
//! without it the last time in the file would silently win every key.
//! Messages produced by statistical processing over a time range (GRIB2
//! product templates other than 4.0/4.1, GRIB1 time-range indicators
//! other than 0/1/10) never match a row: a maximum or minimum 2 m
//! temperature shares the instantaneous field's identity, and whichever
//! the file stored last used to win.
//!
//! Invariants.  `--invariant GRIB` names a file whose messages may supply
//! ONLY the fields named by `--invariant-field`, and only when the main
//! inputs do not carry them -- the AIFS land mask and surface
//! geopotential ride the 0-hour file alone.  An invariant field the
//! invariant files do not carry is a refusal, not a gap.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use chrono::{NaiveDate, NaiveDateTime, TimeDelta};
use grib_core::grib1::Grib1File;
use grib_core::grib2::{grid_latlon, unpack_message, Grib2File, Grib2Message};

/// The receipt schema this tool prints on stdout.  v3 added the time
/// identity (`matched_valid_times`, `--select-valid-time`, the
/// time-processed skip) and the invariant-file provenance; the literal
/// doubles as the contract marker `woof.bridges` looks for.
const RECEIPT_SCHEMA: &str = "gpuwm.rw-wps.met-intermediate/v3";

/// WPS intermediate format version this tool writes.
const WPS_FORMAT_VERSION: i32 = 5;

/// `xlvl` for a field that lives on a single surface rather than on a
/// pressure level.  WPS uses one sentinel for every such surface.
const XLVL_SURFACE: f32 = 200_100.0;

/// `xlvl` for mean-sea-level fields.
const XLVL_MSL: f32 = 201_300.0;

/// The value a bitmap-masked point carries into the intermediate file.
///
/// A GRIB bitmap says "this point has no data" -- over ocean, for a
/// soil field, that is the normal case, not an error.  The decoder
/// hands those points back as non-finite; writing a NaN into the
/// intermediate file would be the worst outcome, because a NaN survives
/// every finiteness check downstream and then poisons an arithmetic
/// mean.  This is the value `ungrib` itself writes at a masked point --
/// `gribcode.F` `SGUP_BITMAP` opens with `datarray = -1.E30` -- so one
/// repair path handles both, and any field the repair does *not* know
/// about keeps a value that is obviously not weather.
const MASKED_SENTINEL: f32 = -1.0e30;

/// Above this magnitude a surface value is a missing code rather than
/// weather (`rrpr.F` `fix_gfs_miss`: `if (abs(f) .gt. 1.e18)`).
const MISSING_MAGNITUDE: f32 = 1.0e18;

/// The WPS soil fill, as `fix_gfs_miss` writes it.
const SOIL_FILL: f32 = -1.0e30;

/// NCEP attributes soil moisture above this to permanent land ice
/// (`fix_gfs_miss`).
const SOIL_MOISTURE_CEILING: f32 = 0.468;

/// Heights above ground, in metres, that the intermediate format
/// admits.  WPS collapses every surface onto one `xlvl`, so a file can
/// hold only one field per name per surface; these are the heights the
/// downstream models actually want.
const ADMISSIBLE_HEIGHTS_M: [f64; 3] = [2.0, 10.0, 1000.0];

/// The gravity `rrpr.F` divides geopotential by.  It is 9.81 there, not
/// a more precise constant, and the whole point of this tool is to
/// write the file `ungrib` would have written.
const GRAVITY: f32 = 9.81;

/// Lowest isobaric pressure, in pascals, that reaches the intermediate
/// file.  `ungrib`'s namelist calls this `pmin` and defaults it to
/// 1 hPa; levels above that are dropped rather than carried, because
/// the models these files feed have no top up there and the vertical
/// interpolation that consumes them would be extrapolating through a
/// near-vacuum.  Dropping them is documented behaviour, not an
/// accident, so this tool reproduces it -- and counts what it dropped.
const DEFAULT_PMIN_PA: f64 = 100.0;

// ---------------------------------------------------------------------
// The producing centre
// ---------------------------------------------------------------------

/// How physical snow depth comes out of water-equivalent snow.
///
/// There is no universal conversion -- it depends on the snow density
/// the producing model assumed -- which is why this is a property of
/// the centre and not a constant (`rrpr.F`, "compute physical snow
/// depth (SNOWH) for various models").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SnowDepthRule {
    /// `scr2d * 0.005` -- the 200:1 ratio NCEP and NOAH assume.
    TwoHundredToOne,
    /// `scr2d / SNOW_DEN` when the centre reports a density, else
    /// `scr2d * 0.004` (250:1).
    EcmwfReportedDensity,
}

/// A producing centre, and every decision that depends on knowing it.
///
/// `ungrib` gets this from the GRIB itself and then makes eight
/// separate decisions by `index(map%source, ...)` substring test.  This
/// tool takes it from the command line, which means a typo would
/// otherwise silently select "none of the repairs" -- exit 0, a
/// structurally perfect file, relative humidity wrong by up to fifty
/// points aloft and missing codes left in eight soil fields.  So the
/// vocabulary is closed: a label either resolves to one of these rows
/// or the run refuses.  There is deliberately no default and no
/// fallthrough arm; adding a centre means reading its rules out of
/// `rrpr.F` and writing a row here.
#[derive(Debug, Clone, Copy, PartialEq)]
struct MapSource {
    /// The label a user types.
    label: &'static str,
    /// The exact text written into every record header, which is the
    /// string `ungrib` puts in `map%source` for this centre.
    header: &'static str,
    /// `rrpr.F`: RH on pressure levels is reported over ice below
    /// freezing and must be converted to over-liquid (`fix_gfs_rh`).
    rh_over_ice: bool,
    /// `rrpr.F`: the July-2017 masked-surface-field repair
    /// (`fix_gfs_miss`).
    masked_surface_repair: bool,
    /// `rrpr.F`: "NCEP GFS weasd is one-half of the NAM value.
    /// Increase it for use in WRF."
    snow_water_equivalent_doubled: bool,
    /// `rrpr.F`: "Convert the ECMWF LANDSEA mask from a fraction to a
    /// flag" (`make_zero_or_one`).
    landsea_to_flag: bool,
    /// `rrpr.F`: the max-wind and tropopause pressures are duplicated
    /// under a second name so metgrid can interpolate them
    /// nearest-neighbour (`gfs_trop_maxw_pressures`).
    nearest_neighbour_pressure_twins: bool,
    snow_depth: SnowDepthRule,
    /// `rd_grib1.F`: for GRIB1 the earth radius is not read from the
    /// message.  NCEP's messages all flag 6367.47 and mean 6371.229, so
    /// WPS hardcodes the radius by centre.  This value lands in every
    /// record header and metgrid uses it.
    grib1_earth_radius_km: f32,
}

/// Every centre whose rules have been read out of WPS source.
const KNOWN_MAP_SOURCES: [MapSource; 4] = [
    MapSource {
        label: "ncep-gfs",
        header: "NCEP GFS Analysis",
        rh_over_ice: true,
        masked_surface_repair: true,
        snow_water_equivalent_doubled: true,
        landsea_to_flag: false,
        nearest_neighbour_pressure_twins: true,
        snow_depth: SnowDepthRule::TwoHundredToOne,
        grib1_earth_radius_km: 6371.229,
    },
    MapSource {
        label: "ncep-gefs",
        header: "NCEP GEFS",
        rh_over_ice: true,
        masked_surface_repair: true,
        snow_water_equivalent_doubled: true,
        landsea_to_flag: false,
        nearest_neighbour_pressure_twins: false,
        snow_depth: SnowDepthRule::TwoHundredToOne,
        grib1_earth_radius_km: 6371.229,
    },
    MapSource {
        label: "ncep-cdas-cfsv2",
        header: "NCEP CDAS CFSV2",
        rh_over_ice: true,
        masked_surface_repair: false,
        snow_water_equivalent_doubled: false,
        landsea_to_flag: false,
        nearest_neighbour_pressure_twins: false,
        snow_depth: SnowDepthRule::TwoHundredToOne,
        grib1_earth_radius_km: 6371.229,
    },
    MapSource {
        label: "ecmwf",
        header: "ECMWF",
        rh_over_ice: true,
        masked_surface_repair: false,
        snow_water_equivalent_doubled: false,
        landsea_to_flag: true,
        nearest_neighbour_pressure_twins: false,
        snow_depth: SnowDepthRule::EcmwfReportedDensity,
        grib1_earth_radius_km: 6367.47,
    },
];

fn known_map_source_list() -> String {
    KNOWN_MAP_SOURCES
        .iter()
        .map(|s| format!("{} (writes \"{}\")", s.label, s.header))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Why an unknown or absent centre is a refusal rather than a default.
fn map_source_refusal(problem: &str) -> String {
    format!(
        "{problem}\n\
         known --map-source labels: {}\n\
         The label is not decoration: it selects the RH over-ice conversion, the \
         masked-surface-field repair, the snow water equivalent and snow depth rules, \
         the land-sea flag conversion and the GRIB1 earth radius.  Guessing it wrong \
         produces a structurally perfect file with wrong numbers and an exit status of \
         zero, so this tool has no default and no fallback.  A centre not on this list \
         needs its rules read out of WPS's rrpr.F and added to KNOWN_MAP_SOURCES.",
        known_map_source_list()
    )
}

fn resolve_map_source(label: &str) -> Result<&'static MapSource, String> {
    let wanted = label.trim();
    KNOWN_MAP_SOURCES
        .iter()
        .find(|s| s.label.eq_ignore_ascii_case(wanted))
        .ok_or_else(|| map_source_refusal(&format!("unknown --map-source {label:?}.")))
}

// ---------------------------------------------------------------------
// Vtable
// ---------------------------------------------------------------------

/// One Vtable row: a selection rule plus the output naming it implies.
#[derive(Debug, Clone)]
struct VtableRow {
    /// GRIB1 parameter number, absent when the row is derived-only.
    g1_param: Option<u16>,
    g1_level_type: Option<u16>,
    /// `None` means the Vtable wildcard `*` -- match every level.
    g1_level1: Option<i64>,
    g1_level2: Option<i64>,

    g2_discipline: Option<u16>,
    g2_category: Option<u16>,
    g2_param: Option<u16>,
    g2_level_type: Option<u16>,

    name: String,
    units: String,
    desc: String,
}

/// The output identity of a field name, as `output.F` computes it: the
/// FIRST Vtable row carrying that name supplies the units and the
/// description for every record of it, and rows are emitted in Vtable
/// order.
#[derive(Debug, Clone)]
struct FieldMeta {
    name: String,
    units: String,
    desc: String,
}

impl FieldMeta {
    /// `output.F`, final pass: `if (desc.eq.' ') cycle OUTLOOP`.  A
    /// Vtable row with an empty Description column names a field
    /// `ungrib` reads and consumes but never publishes -- ECMWF's
    /// GEOPT, SOILGEO, SNOW_EC, SNOW_DEN and DEWPT are exactly that.
    /// The Vtable itself is where that decision lives; there is no
    /// list of scratch names in this binary.
    fn is_published(&self) -> bool {
        !self.desc.trim().is_empty()
    }
}

/// Parse a WPS Vtable.  Columns are pipe-separated; the GRIB2 columns
/// are optional (ECMWF tables omit them entirely).
fn parse_vtable(path: &Path) -> Result<Vec<VtableRow>, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("cannot read Vtable {}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        // Comments, rules and the two header lines all fail the
        // "first column parses as a code or is blank" test below, but
        // skip the obvious ones first so a malformed real row is not
        // hidden among them.
        if trimmed.starts_with('#') || trimmed.starts_with('-') || trimmed.is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split('|').collect();
        if cells.len() < 7 {
            continue;
        }
        let name = cells[4].trim();
        if name.is_empty() || name.eq_ignore_ascii_case("metgrid") || name.eq_ignore_ascii_case("Name") {
            continue;
        }
        // A row whose level-type column is not an integer is a header.
        let g1_level_type = parse_opt_u16(cells[1]);
        if g1_level_type.is_none() && parse_opt_u16(cells[0]).is_none() {
            continue;
        }
        let (g2_discipline, g2_category, g2_param, g2_level_type) = if cells.len() >= 11 {
            (
                parse_opt_u16(cells[7]),
                parse_opt_u16(cells[8]),
                parse_opt_u16(cells[9]),
                parse_opt_u16(cells[10]),
            )
        } else {
            (None, None, None, None)
        };
        rows.push(VtableRow {
            g1_param: parse_opt_u16(cells[0]),
            g1_level_type,
            g1_level1: parse_opt_level(cells[2]),
            g1_level2: parse_opt_level(cells[3]),
            g2_discipline,
            g2_category,
            g2_param,
            g2_level_type,
            name: name.to_string(),
            units: cells[5].trim().to_string(),
            desc: cells[6].trim().to_string(),
        });
    }
    if rows.is_empty() {
        return Err(format!("Vtable {} declares no fields", path.display()));
    }
    Ok(rows)
}

/// Collapse the Vtable to its output field list, first occurrence wins,
/// order preserved -- `output.F`'s `OUTLOOP` with its dedup guard.
fn field_metas(vtable: &[VtableRow]) -> Vec<FieldMeta> {
    let mut out: Vec<FieldMeta> = Vec::new();
    for row in vtable {
        if out.iter().any(|m| m.name == row.name) {
            continue;
        }
        out.push(FieldMeta {
            name: row.name.clone(),
            units: row.units.clone(),
            desc: row.desc.clone(),
        });
    }
    out
}

fn parse_opt_u16(cell: &str) -> Option<u16> {
    let t = cell.trim();
    if t.is_empty() || t == "*" {
        None
    } else {
        t.parse::<u16>().ok()
    }
}

/// Level columns accept `*` (wildcard) as well as blanks and integers.
/// A wildcard and a blank mean different things -- wildcard matches any
/// level, blank means the level is not part of the key -- but both are
/// represented as `None` here because in both cases the row places no
/// constraint on that column.
fn parse_opt_level(cell: &str) -> Option<i64> {
    let t = cell.trim();
    if t.is_empty() || t == "*" {
        None
    } else {
        t.parse::<i64>().ok()
    }
}

// ---------------------------------------------------------------------
// Decoded slabs
// ---------------------------------------------------------------------

/// The grid a slab lives on, in the terms the intermediate format
/// records.  Only the projections WPS writes for global analyses are
/// covered; anything else refuses rather than guessing a header.
#[derive(Debug, Clone, PartialEq)]
struct SlabGrid {
    iproj: i32,
    nx: usize,
    ny: usize,
    /// Latitude of the (1,1) point after normalization to SW-corner
    /// storage.
    startlat: f32,
    startlon: f32,
    deltalat: f32,
    deltalon: f32,
    /// The spherical earth radius this file declares, in kilometres.
    /// `read_met` overrides it with MPAS's own constant, but metgrid
    /// does not, so the value still has to be the one `ungrib` writes.
    earth_radius_km: f32,
}

/// One field at one level, ready to write.
struct Slab {
    name: String,
    xlvl: f32,
    grid: SlabGrid,
    values: Vec<f32>,
}

/// Key that orders the store: field name, then level.  The order
/// records are *written* in is computed separately, in `emission_order`.
type SlabKey = (String, ordered_f32::OrderedF32);

mod ordered_f32 {
    /// A total order over the `xlvl` values actually used, which are
    /// finite by construction.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct OrderedF32(pub f32);
    impl Eq for OrderedF32 {}
    impl PartialOrd for OrderedF32 {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for OrderedF32 {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
        }
    }
}

use ordered_f32::OrderedF32;

// ---------------------------------------------------------------------
// Level mapping
// ---------------------------------------------------------------------

/// The `xlvl` WPS records for a GRIB2 fixed-surface type.
///
/// Every non-isobaric surface collapses onto one sentinel: the
/// intermediate format carries no surface vocabulary, so the *field
/// name* (`ST000010`, `UMAXW`, ...) is what distinguishes them, exactly
/// as the Vtable's name column implies.
fn xlvl_for_grib2_level(level_type: u16, level_value: f64) -> Option<f32> {
    match level_type {
        // Isobaric surface: GRIB2 records pascals, and so does WPS.
        100 => Some(level_value as f32),
        // Mean sea level.
        101 => Some(XLVL_MSL),
        // Ground/water surface, specific height above ground, depth
        // below land surface, max-wind level, tropopause.
        1 | 103 | 106 | 6 | 7 => Some(XLVL_SURFACE),
        // Generalized vertical coordinate (type 151): ECMWF's open-data
        // products address their soil column by ordinal on this surface
        // (sot/vsw layers 1..4).  Each ordinal is its own Vtable row and
        // its own named field, all at the surface sentinel, exactly like
        // the depth-bounded type-106 layers.
        151 => Some(XLVL_SURFACE),
        _ => None,
    }
}

/// The `xlvl` WPS records for a GRIB1 level-type indicator.
fn xlvl_for_grib1_level(level_type: u16, level_value: f64) -> Option<f32> {
    match level_type {
        // Isobaric: GRIB1 records hectopascals, WPS records pascals.
        100 => Some((level_value * 100.0) as f32),
        102 => Some(XLVL_MSL),
        1 | 105 | 112 => Some(XLVL_SURFACE),
        _ => None,
    }
}

// ---------------------------------------------------------------------
// Earth radius
// ---------------------------------------------------------------------

/// GRIB2 Code Table 3.2, as `rd_grib2.F`'s `earth_radius` reads it.
///
/// WPS calls `mprintf(ERROR)` -- a hard stop -- on a shape it does not
/// know, and so does this, for the same reason: the radius goes into
/// every record header and metgrid trusts it.  Shape 1 carries a scaled
/// radius in the grid template that this decoder does not surface, so
/// it refuses by name rather than substituting a plausible sphere.
fn grib2_earth_radius_km(shape: u8) -> Result<f32, String> {
    match shape {
        0 => Ok(6367470.0 * 0.001),
        6 => Ok(6371229.0 * 0.001),
        8 => Ok(6371200.0 * 0.001),
        1 => Err(
            "GRIB2 shape-of-earth 1 carries a scaled radius in the grid definition \
             template, which this decoder does not surface; refusing rather than \
             writing a guessed earth radius into every record header"
                .to_string(),
        ),
        other => Err(format!(
            "unknown GRIB2 shape-of-earth code {other}; WPS's rd_grib2.F stops here too"
        )),
    }
}

// ---------------------------------------------------------------------
// Grid normalization
// ---------------------------------------------------------------------

/// Describe a decoded slab in the terms the intermediate format
/// records, leaving the values in the order the GRIB message stored
/// them.
///
/// The format's `startloc` is always `SWCORNER`, but the "start" it
/// names is the *first stored point*, and `deltalat` carries the sign
/// of the scan.  A north-to-south GRIB therefore becomes
/// `startlat = +90, deltalat = -0.25`, which is what `ungrib` writes
/// and what every reader of these files already expects.  Flipping the
/// rows to a genuinely south-first array and reporting a positive
/// increment would also be self-consistent, and would also be read
/// correctly -- but it would not be the same file, so it is not what
/// this tool does.
fn describe_latlon(
    nx: usize,
    ny: usize,
    lats: &[f64],
    lons: &[f64],
    values: &[f64],
    earth_radius_km: f32,
) -> Result<(SlabGrid, Vec<f32>), String> {
    if lats.len() != nx * ny || lons.len() != nx * ny {
        return Err("coordinate arrays do not match the grid shape".into());
    }
    if values.len() != nx * ny {
        return Err(format!(
            "value count {} does not match grid {nx}x{ny}",
            values.len()
        ));
    }
    if nx < 2 || ny < 2 {
        return Err(format!("grid {nx}x{ny} is too small to carry an increment"));
    }

    // The intermediate format this tool writes declares iproj 0 -- a
    // uniform cylindrical-equidistant grid -- and nothing else.  Before
    // this gate, a projected product (Lambert HRRR/RAP/RRFS) or a full
    // Gaussian grid sailed through and was stamped as uniform lat-lon
    // with increments taken from its first two points: every point except
    // the corner silently mis-georeferenced, in a file that opens and
    // reads perfectly.  Measured on real HRRR wrfprs bytes, 2026-08-24:
    // the 3 km Lambert CONUS grid was written as latlon
    // deltalat=0.0249/deltalon=0.0267.  A grid is admitted only when
    // every row is a parallel, every column is a meridian, and both
    // increments are constant.
    const UNIFORM_TOL_DEG: f64 = 1e-4;
    let wrap = |d: f64| -> f64 {
        let mut w = d;
        while w > 180.0 {
            w -= 360.0;
        }
        while w < -180.0 {
            w += 360.0;
        }
        w
    };
    let row_delta = lats[nx] - lats[0];
    let col_delta = wrap(lons[1] - lons[0]);
    for r in 0..ny {
        let row = &lats[r * nx..(r + 1) * nx];
        for (i, lat) in row.iter().enumerate() {
            if (lat - row[0]).abs() > UNIFORM_TOL_DEG {
                return Err(format!(
                    "grid is not uniform lat-lon: latitude varies by {:.6} deg \
                     along row {r} (column {i}); a cylindrical-equidistant row \
                     is a parallel.  Writing this projected grid as iproj 0 \
                     would silently mis-georeference every point, so this tool \
                     refuses.  Projected products (e.g. Lambert HRRR/RAP/RRFS) \
                     are outside the WPS-intermediate chain; use the mapped \
                     native route",
                    (lat - row[0]).abs()
                ));
            }
        }
        if r + 1 < ny {
            let step = lats[(r + 1) * nx] - lats[r * nx];
            if (step - row_delta).abs() > UNIFORM_TOL_DEG {
                return Err(format!(
                    "grid is not uniform lat-lon: row spacing moves from \
                     {row_delta:.6} to {step:.6} deg at row {r} (a full \
                     Gaussian grid does this).  iproj 0 declares one constant \
                     latitude increment, so writing this file would silently \
                     mis-georeference rows away from the corner; this tool \
                     refuses rather than guess"
                ));
            }
        }
    }
    for r in 0..ny {
        let row = &lons[r * nx..(r + 1) * nx];
        for i in 0..nx {
            if wrap(row[i] - lons[i]).abs() > UNIFORM_TOL_DEG {
                return Err(format!(
                    "grid is not uniform lat-lon: longitude of column {i} \
                     moves by {:.6} deg between row 0 and row {r}; a \
                     cylindrical-equidistant column is a meridian.  Writing \
                     this projected grid as iproj 0 would silently \
                     mis-georeference every point, so this tool refuses",
                    wrap(row[i] - lons[i]).abs()
                ));
            }
            if i + 1 < nx {
                let step = wrap(row[i + 1] - row[i]);
                if (step - col_delta).abs() > UNIFORM_TOL_DEG {
                    return Err(format!(
                        "grid is not uniform lat-lon: longitude spacing moves \
                         from {col_delta:.6} to {step:.6} deg at row {r}, \
                         column {i}; iproj 0 declares one constant longitude \
                         increment, so this tool refuses rather than \
                         mis-georeference"
                    ));
                }
            }
        }
    }

    let startlat = lats[0] as f32;
    let mut startlon = lons[0] as f32;
    if startlon > 180.0 {
        startlon -= 360.0;
    }
    // Signed increments between the first two rows and the first two
    // columns of the stored array.
    let deltalat = (lats[nx] - lats[0]) as f32;
    let mut dlon = lons[1] - lons[0];
    if dlon < -180.0 {
        dlon += 360.0;
    } else if dlon > 180.0 {
        dlon -= 360.0;
    }
    let deltalon = dlon as f32;

    let out: Vec<f32> = values.iter().map(|v| *v as f32).collect();

    Ok((
        SlabGrid {
            iproj: 0,
            nx,
            ny,
            startlat,
            startlon,
            deltalat,
            deltalon,
            earth_radius_km,
        },
        out,
    ))
}

// ---------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------

struct Collector {
    pmin_pa: f64,
    /// When set, a matched message valid at any other time is dropped
    /// (and counted) instead of competing for the key.
    select_valid_time: Option<NaiveDateTime>,
    /// Matched messages dropped by `select_valid_time`.
    messages_other_valid_time: usize,
    /// Matched messages dropped for being statistically processed over a
    /// time range (max/min/accumulation/average), not instantaneous.
    messages_time_processed: usize,
    /// Distinct valid times of every message that reached the store.
    matched_valid_times: BTreeSet<NaiveDateTime>,
    /// Isobaric levels dropped for sitting above `pmin_pa`, counted so
    /// the omission is reported rather than silent.
    levels_below_pmin: usize,
    /// Points a GRIB bitmap declared absent, across every field.
    masked_points: usize,
    slabs: BTreeMap<SlabKey, Slab>,
    messages_seen: usize,
    messages_matched: usize,
}

impl Collector {
    fn new(pmin_pa: f64) -> Self {
        Collector {
            pmin_pa,
            select_valid_time: None,
            messages_other_valid_time: 0,
            messages_time_processed: 0,
            matched_valid_times: BTreeSet::new(),
            levels_below_pmin: 0,
            masked_points: 0,
            slabs: BTreeMap::new(),
            messages_seen: 0,
            messages_matched: 0,
        }
    }

    fn insert(&mut self, slab: Slab) {
        let key = (slab.name.clone(), OrderedF32(slab.xlvl));
        // Later files win, which is how ungrib treats a field that
        // appears in more than one input: the last GRIBFILE read
        // supplies the value.
        self.slabs.insert(key, slab);
    }

    /// Decide whether a matched message valid at `valid` may enter the
    /// store, recording the decision.
    fn admit_valid_time(&mut self, valid: NaiveDateTime) -> bool {
        if let Some(want) = self.select_valid_time {
            if valid != want {
                self.messages_other_valid_time += 1;
                return false;
            }
        }
        self.matched_valid_times.insert(valid);
        true
    }

    fn has_field(&self, name: &str) -> bool {
        self.slabs.keys().any(|(n, _)| n == name)
    }

    fn get(&self, name: &str, xlvl: f32) -> Option<&Slab> {
        self.slabs.get(&(name.to_string(), OrderedF32(xlvl)))
    }

    fn surface(&self, name: &str) -> Option<&Slab> {
        self.get(name, XLVL_SURFACE)
    }

    fn has_surface(&self, name: &str) -> bool {
        self.surface(name).is_some()
    }

    /// Levels present, in the order `ungrib` emits them: `get_plvls` in
    /// `new_storage.F` is an insertion sort that puts the largest first,
    /// so the surface and mean-sea-level sentinels lead and pressure
    /// descends behind them.
    fn levels_descending(&self) -> Vec<f32> {
        let mut levels: Vec<f32> = Vec::new();
        for (_, lvl) in self.slabs.keys() {
            if !levels.contains(&lvl.0) {
                levels.push(lvl.0);
            }
        }
        levels.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        levels
    }
}

fn collect_grib2(path: &Path, vtable: &[VtableRow], out: &mut Collector) -> Result<(), String> {
    let path_text = path
        .to_str()
        .ok_or_else(|| format!("{}: path is not valid UTF-8", path.display()))?;
    let file = Grib2File::open(path_text).map_err(|e| format!("{}: {e:?}", path.display()))?;
    for message in &file.messages {
        out.messages_seen += 1;
        let product = &message.product;
        let row = vtable.iter().find(|r| {
            r.g2_param.is_some()
                && r.g2_discipline == Some(message.discipline as u16)
                && r.g2_category == Some(product.parameter_category as u16)
                && r.g2_param == Some(product.parameter_number as u16)
                && r.g2_level_type == Some(product.level_type as u16)
                && grib2_level_bounds_match(r, product.level_value, product.second_level_value, product.level_type)
        });
        let Some(row) = row else { continue };
        let Some(xlvl) = xlvl_for_grib2_level(product.level_type as u16, product.level_value) else {
            continue;
        };
        if !grib2_is_instantaneous(message) {
            out.messages_time_processed += 1;
            continue;
        }
        if product.level_type == 100 && (xlvl as f64) < out.pmin_pa {
            out.levels_below_pmin += 1;
            continue;
        }
        let valid = grib2_valid_time(message)
            .map_err(|e| format!("{}: field {}: {e}", path.display(), row.name))?;
        if !out.admit_valid_time(valid) {
            continue;
        }
        // Raw, not scan-normalized: `grid_latlon` reports coordinates in
        // the grid definition's own order, so taking values in that same
        // order is what keeps a point and its coordinate together.
        let values = unpack_message(message)
            .map_err(|e| format!("{}: unpacking {} failed: {e:?}", path.display(), row.name))?;
        let grid = &message.grid;
        if grid.is_reduced {
            return Err(format!(
                "{}: field {} is on a reduced Gaussian grid, which this tool does not \
                 write to the intermediate format",
                path.display(),
                row.name
            ));
        }
        let radius = grib2_earth_radius_km(grid.shape_of_earth)
            .map_err(|e| format!("{}: field {}: {e}", path.display(), row.name))?;
        let (lats, lons) = grid_latlon(grid)
            .map_err(|e| format!("{}: field {}: {e}", path.display(), row.name))?;
        let (slab_grid, mut data) =
            describe_latlon(grid.nx as usize, grid.ny as usize, &lats, &lons, &values, radius)
                .map_err(|e| format!("{}: field {}: {e}", path.display(), row.name))?;
        out.masked_points += mark_masked(&mut data);
        out.messages_matched += 1;
        out.insert(Slab {
            name: row.name.clone(),
            xlvl,
            grid: slab_grid,
            values: data,
        });
    }
    Ok(())
}

/// Replace the decoder's non-finite bitmap holes with the value
/// `ungrib` writes there, and report how many there were.
fn mark_masked(values: &mut [f32]) -> usize {
    let mut n = 0;
    for value in values.iter_mut() {
        if !value.is_finite() {
            *value = MASKED_SENTINEL;
            n += 1;
        }
    }
    n
}

// ---------------------------------------------------------------------
// Time identity
// ---------------------------------------------------------------------

/// The `--date` text WPS writes into every header, as a time.
fn parse_hdate(text: &str) -> Result<NaiveDateTime, String> {
    let bad = || format!("--date {text:?} is not YYYY-MM-DD_HH:MM:SS");
    let t = text.trim();
    let b = t.as_bytes();
    if b.len() != 19 || b[4] != b'-' || b[7] != b'-' || b[10] != b'_' || b[13] != b':' || b[16] != b':' {
        return Err(bad());
    }
    let num = |range: std::ops::Range<usize>| -> Result<u32, String> {
        t.get(range).and_then(|s| s.parse::<u32>().ok()).ok_or_else(bad)
    };
    let (year, month, day) = (num(0..4)? as i32, num(5..7)?, num(8..10)?);
    let (hour, minute, second) = (num(11..13)?, num(14..16)?, num(17..19)?);
    NaiveDate::from_ymd_opt(year, month, day)
        .and_then(|d| d.and_hms_opt(hour, minute, second))
        .ok_or_else(bad)
}

fn format_hdate(time: &NaiveDateTime) -> String {
    use chrono::{Datelike, Timelike};
    format!(
        "{:04}-{:02}-{:02}_{:02}:{:02}:{:02}",
        time.year(),
        time.month(),
        time.day(),
        time.hour(),
        time.minute(),
        time.second()
    )
}

/// Seconds in one unit of GRIB2 Code Table 4.4 / GRIB1 Table 4 (the
/// codes the two editions share; GRIB1 spells seconds 254, GRIB2 13).
fn time_unit_seconds(unit: u8, edition: u8) -> Result<i64, String> {
    Ok(match (unit, edition) {
        (0, _) => 60,
        (1, _) => 3_600,
        (2, _) => 86_400,
        (10, _) => 3 * 3_600,
        (11, _) => 6 * 3_600,
        (12, _) => 12 * 3_600,
        (13, 2) | (254, 1) => 1,
        (other, _) => {
            return Err(format!(
                "GRIB{edition} time unit code {other} is not one this tool converts; \
                 refusing rather than guessing the message's valid time"
            ))
        }
    })
}

/// GRIB2 product templates whose field is a value AT the valid time:
/// 4.0 (deterministic) and 4.1 (one ensemble member).  Every other
/// template either processes a time range (4.8, 4.11, 4.12 ...) or is a
/// derived statistic (mean, spread, percentile, probability), none of
/// which a Vtable row means.
fn grib2_is_instantaneous(message: &Grib2Message) -> bool {
    matches!(message.product.template, 0 | 1) && message.product.statistical_process_type.is_none()
}

fn grib2_valid_time(message: &Grib2Message) -> Result<NaiveDateTime, String> {
    let product = &message.product;
    let unit = time_unit_seconds(product.time_range_unit, 2)?;
    let lead = TimeDelta::try_seconds(unit * product.forecast_time as i64)
        .ok_or_else(|| format!("forecast time {} overflows", product.forecast_time))?;
    Ok(message.reference_time + lead)
}

fn grib1_valid_time(pds: &grib_core::grib1::ProductDefinitionSection) -> Result<NaiveDateTime, String> {
    let reference = NaiveDate::from_ymd_opt(pds.year(), pds.month as u32, pds.day as u32)
        .and_then(|d| d.and_hms_opt(pds.hour as u32, pds.minute as u32, 0))
        .ok_or_else(|| {
            format!(
                "GRIB1 reference time {}-{}-{} {}:{} is not a date",
                pds.year(),
                pds.month,
                pds.day,
                pds.hour,
                pds.minute
            )
        })?;
    let steps: i64 = match pds.time_range_indicator {
        0 => pds.p1 as i64,
        1 => 0,
        10 => (pds.p1 as i64) * 256 + pds.p2 as i64,
        other => {
            return Err(format!(
                "GRIB1 time-range indicator {other} is not an instantaneous value"
            ))
        }
    };
    let unit = time_unit_seconds(pds.time_unit, 1)?;
    let lead = TimeDelta::try_seconds(unit * steps)
        .ok_or_else(|| format!("GRIB1 lead {steps} overflows"))?;
    Ok(reference + lead)
}

/// Soil-layer rows key on the layer bounds as well as the surface type.
/// GRIB2 records those bounds as scaled values in metres; the Vtable
/// writes them in centimetres, as WPS has since PREGRID.
fn grib2_level_bounds_match(row: &VtableRow, level1: f64, level2: f64, level_type: u8) -> bool {
    // Height above ground.  A global GRIB carries temperature at 2 m and
    // at 80 m, and wind at 10 m, 80 m and 100 m, all under the same
    // discipline/category/parameter and the same surface type -- so a
    // rule that ignores the height silently picks whichever message the
    // file happens to store first.  The Vtable writes the height it
    // means in its Level1 column; honour it.
    if level_type == 103 {
        return match row.g1_level1 {
            Some(want) => (level1 - want as f64).abs() < 0.5,
            // No height in the table: fall back to the set WPS admits,
            // rather than accepting an arbitrary height.
            None => ADMISSIBLE_HEIGHTS_M.iter().any(|h| (level1 - *h).abs() < 0.5),
        };
    }
    // Generalized vertical coordinate: the level "value" is an ordinal
    // pair, not a depth, so the Vtable's Level1/Level2 columns carry the
    // ordinals verbatim.  A row without them would match all four soil
    // layers and last-write-wins would keep exactly one, silently, so
    // the ordinals are required for this surface type.
    if level_type == 151 {
        return match (row.g1_level1, row.g1_level2) {
            (Some(want1), Some(want2)) => {
                level1.round() as i64 == want1 && level2.round() as i64 == want2
            }
            _ => false,
        };
    }
    if level_type != 106 {
        return true;
    }
    let (Some(want1), Some(want2)) = (row.g1_level1, row.g1_level2) else {
        return true;
    };
    let got1 = (level1 * 100.0).round() as i64;
    let got2 = (level2 * 100.0).round() as i64;
    got1 == want1 && got2 == want2
}

fn collect_grib1(
    path: &Path,
    vtable: &[VtableRow],
    source: &MapSource,
    out: &mut Collector,
) -> Result<(), String> {
    let file = Grib1File::open(path).map_err(|e| format!("{}: {e:?}", path.display()))?;
    for message in &file.messages {
        out.messages_seen += 1;
        let pds = &message.pds;
        let row = vtable.iter().find(|r| {
            r.g1_param == Some(pds.parameter as u16)
                && r.g1_level_type == Some(pds.level_type as u16)
                && grib1_level_bounds_match(r, pds.level_type, pds.level_top, pds.level_bottom)
                && grib1_height_matches(r, pds.level_type, pds.level_value as f64)
        });
        let Some(row) = row else { continue };
        let Some(xlvl) = xlvl_for_grib1_level(pds.level_type as u16, pds.level_value as f64) else {
            continue;
        };
        if !matches!(pds.time_range_indicator, 0 | 1 | 10) {
            out.messages_time_processed += 1;
            continue;
        }
        if pds.level_type == 100 && (xlvl as f64) < out.pmin_pa {
            out.levels_below_pmin += 1;
            continue;
        }
        let valid = grib1_valid_time(pds)
            .map_err(|e| format!("{}: field {}: {e}", path.display(), row.name))?;
        if !out.admit_valid_time(valid) {
            continue;
        }
        let Some(gds) = message.gds.as_ref() else {
            return Err(format!("{}: field {} has no GDS", path.display(), row.name));
        };
        let values = message
            .values()
            .map_err(|e| format!("{}: unpacking {} failed: {e:?}", path.display(), row.name))?;
        let coords = message
            .latlons()
            .map_err(|e| format!("{}: coordinates for {} failed: {e:?}", path.display(), row.name))?;
        let (nx, ny) = grib1_shape(gds)?;
        let lats: Vec<f64> = coords.iter().map(|c| c.lat).collect();
        let lons: Vec<f64> = coords.iter().map(|c| c.lon).collect();
        let (slab_grid, mut data) = describe_latlon(
            nx,
            ny,
            &lats,
            &lons,
            &values,
            source.grib1_earth_radius_km,
        )
        .map_err(|e| format!("{}: field {}: {e}", path.display(), row.name))?;
        out.masked_points += mark_masked(&mut data);
        out.messages_matched += 1;
        out.insert(Slab {
            name: row.name.clone(),
            xlvl,
            grid: slab_grid,
            values: data,
        });
    }
    Ok(())
}

fn grib1_level_bounds_match(row: &VtableRow, level_type: u8, top: u8, bottom: u8) -> bool {
    if level_type != 112 {
        return true;
    }
    let (Some(want1), Some(want2)) = (row.g1_level1, row.g1_level2) else {
        return true;
    };
    top as i64 == want1 && bottom as i64 == want2
}

/// The GRIB1 twin of the height rule: level type 105 is "specified
/// height above ground", and the Vtable's Level1 column names which
/// height the row means.
fn grib1_height_matches(row: &VtableRow, level_type: u8, level_value: f64) -> bool {
    if level_type != 105 {
        return true;
    }
    match row.g1_level1 {
        Some(want) => (level_value - want as f64).abs() < 0.5,
        None => ADMISSIBLE_HEIGHTS_M.iter().any(|h| (level_value - *h).abs() < 0.5),
    }
}

fn grib1_shape(gds: &grib_core::grib1::GridDescriptionSection) -> Result<(usize, usize), String> {
    use grib_core::grib1::GridType;
    match &gds.grid_type {
        GridType::LatLon { ni, nj, .. } => Ok((*ni as usize, *nj as usize)),
        GridType::Gaussian { ni, nj, .. } => Ok((*ni as usize, *nj as usize)),
        other => Err(format!(
            "unsupported GRIB1 grid type for the intermediate format: {other:?}"
        )),
    }
}

// ---------------------------------------------------------------------
// The rules `rrpr` applies
// ---------------------------------------------------------------------

/// What one rule did, and -- when it did nothing -- why.
///
/// Every rule reports, including the ones that stayed off.  A receipt
/// that only lists what ran cannot distinguish "this centre does not
/// need the repair" from "the repair never fired because the label was
/// wrong", which is the failure this whole vocabulary exists to stop.
struct RuleDecision {
    rule: &'static str,
    applied: bool,
    basis: String,
    points: usize,
    detail: String,
}

#[derive(Default)]
struct RuleLedger {
    entries: Vec<RuleDecision>,
}

impl RuleLedger {
    fn on(&mut self, rule: &'static str, basis: String, points: usize, detail: String) {
        self.entries.push(RuleDecision { rule, applied: true, basis, points, detail });
    }
    fn off(&mut self, rule: &'static str, basis: String) {
        self.entries.push(RuleDecision {
            rule,
            applied: false,
            basis,
            points: 0,
            detail: String::new(),
        });
    }
}

/// Apply, in `rrpr.F`'s own order, every rule this tool reproduces.
///
/// The order is not cosmetic.  `fix_gfs_miss` zeroes masked snow before
/// the GFS doubling reaches it, and the doubling lands before snow
/// depth is derived from snow; running them in any other sequence
/// produces different numbers at every snow-covered point.
fn apply_ungrib_rules(collector: &mut Collector, source: &MapSource, ledger: &mut RuleLedger) {
    height_from_geopotential(collector, ledger);
    relative_humidity_over_ice(collector, source, ledger);
    surface_rh_from_dewpoint(collector, ledger);
    masked_surface_fields(collector, source, ledger);
    soil_height_from_geopotential(collector, ledger);
    snow_from_ecmwf_water_equivalent(collector, ledger);
    landsea_fraction_to_flag(collector, source, ledger);
    snow_water_equivalent_doubling(collector, source, ledger);
    snow_depth_from_snow(collector, source, ledger);
    nearest_neighbour_pressure_twins(collector, source, ledger);
}

/// `rrpr.F`: `scr2d = scr2d / 9.81`, at every pressure level that has a
/// geopotential and no height.  Single precision, as there.
fn height_from_geopotential(collector: &mut Collector, ledger: &mut RuleLedger) {
    let sources: Vec<(f32, Vec<f32>, SlabGrid)> = collector
        .slabs
        .iter()
        .filter(|((n, _), _)| n == "GEOPT")
        .filter(|((_, lvl), _)| collector.get("HGT", lvl.0).is_none())
        .map(|((_, lvl), s)| (lvl.0, s.values.clone(), s.grid.clone()))
        .collect();
    if sources.is_empty() {
        ledger.off("hgt_from_geopt", "no GEOPT level lacks an HGT".into());
        return;
    }
    let mut points = 0usize;
    let levels = sources.len();
    for (xlvl, values, grid) in sources {
        let derived: Vec<f32> = values.iter().map(|v| v / GRAVITY).collect();
        points += derived.len();
        collector.insert(Slab { name: "HGT".into(), xlvl, grid, values: derived });
    }
    ledger.on(
        "hgt_from_geopt",
        "GEOPT present without HGT (not source-gated)".into(),
        points,
        format!("HGT = GEOPT / 9.81 on {levels} levels"),
    );
}

/// `rrpr.F`: "Check to see if we need to fill SOILHGT from SOILGEO" --
/// the surface twin of the height derivation, at the same precision.
fn soil_height_from_geopotential(collector: &mut Collector, ledger: &mut RuleLedger) {
    if collector.has_surface("SOILHGT") || !collector.has_surface("SOILGEO") {
        ledger.off(
            "soilhgt_from_soilgeo",
            "SOILHGT already present, or no SOILGEO".into(),
        );
        return;
    }
    let src = collector.surface("SOILGEO").expect("checked");
    let grid = src.grid.clone();
    let values: Vec<f32> = src.values.iter().map(|v| v / GRAVITY).collect();
    let points = values.len();
    collector.insert(Slab { name: "SOILHGT".into(), xlvl: XLVL_SURFACE, grid, values });
    ledger.on(
        "soilhgt_from_soilgeo",
        "SOILGEO present without SOILHGT (not source-gated)".into(),
        points,
        "SOILHGT = SOILGEO / 9.81".into(),
    );
}

/// Convert relative humidity reported over ice to relative humidity over
/// liquid water -- `rrpr.F` `fix_gfs_rh`, in single precision as there.
///
/// This matters more than it looks.  GFS and ECMWF report RH with
/// respect to ice below freezing; everything downstream -- WRF's `real`,
/// MPAS's `init_atmosphere` -- assumes RH over liquid water when it
/// turns RH into a mixing ratio.  Skipping the conversion does not fail,
/// it just makes the initial atmosphere too dry aloft, by up to tens of
/// percentage points in the coldest air.
fn relative_humidity_over_ice(
    collector: &mut Collector,
    source: &MapSource,
    ledger: &mut RuleLedger,
) {
    if !source.rh_over_ice {
        ledger.off(
            "rh_over_ice_to_liquid",
            format!("{} does not report RH over ice", source.header),
        );
        return;
    }
    let levels: Vec<OrderedF32> = collector
        .slabs
        .keys()
        .filter(|(name, _)| name == "RH")
        .map(|(_, lvl)| *lvl)
        .collect();

    let mut adjusted_points = 0usize;
    let mut adjusted_levels = 0usize;
    for level in levels {
        let Some(temperature) = collector.get("TT", level.0).map(|s| s.values.clone()) else {
            continue;
        };
        let Some(humidity) = collector.slabs.get_mut(&("RH".to_string(), level)) else {
            continue;
        };
        if humidity.values.len() != temperature.len() {
            continue;
        }
        let mut touched = 0usize;
        for (rh, t) in humidity.values.iter_mut().zip(temperature.iter()) {
            let t = *t;
            if t > 273.15 {
                continue;
            }
            // Murphy and Koop (2005) over ice, Bolton (1980) over
            // liquid, blended linearly through the mixed-phase band --
            // the arithmetic of fix_gfs_rh, at its precision.
            let over_ice =
                0.01f32 * (9.550426 - (5723.265 / t) + (3.53068 * t.ln()) - (0.00728332 * t)).exp();
            let over_liquid = 6.112f32 * ((17.67 * (t - 273.15)) / ((t - 273.15) + 243.5)).exp();
            let reference = if t > 253.15 {
                let blend = (273.15 - t) / 20.0;
                (blend * over_ice) + ((1.0 - blend) * over_liquid)
            } else {
                over_ice
            };
            *rh *= reference / over_liquid;
            touched += 1;
        }
        if touched > 0 {
            adjusted_levels += 1;
            adjusted_points += touched;
        }
    }
    ledger.on(
        "rh_over_ice_to_liquid",
        format!("{} reports pressure-level RH over ice", source.header),
        adjusted_points,
        format!(
            "Murphy-Koop over ice / Bolton over liquid, blended -20..0 C, on \
             {adjusted_levels} levels"
        ),
    );
}

/// `rrpr.F` `compute_rh_dewpt`, reached only when the file carries no
/// surface RH of its own.
///
/// The formula is WPS's, not a better one: a Clausius-Clapeyron
/// integration with a constant latent heat, unclamped.  A more accurate
/// saturation-vapour-pressure form disagrees with it by up to two
/// percentage points, and this tool exists to write the file `ungrib`
/// would have written.
fn surface_rh_from_dewpoint(collector: &mut Collector, ledger: &mut RuleLedger) {
    if collector.has_surface("RH") {
        ledger.off("surface_rh_from_dewpoint", "the file carries surface RH".into());
        return;
    }
    let (Some(temp), Some(dewpt)) = (collector.surface("TT"), collector.surface("DEWPT")) else {
        ledger.off(
            "surface_rh_from_dewpoint",
            "no surface RH, and no TT/DEWPT pair to build one from".into(),
        );
        return;
    };
    if temp.values.len() != dewpt.values.len() {
        ledger.off(
            "surface_rh_from_dewpoint",
            "surface TT and DEWPT are on different grids".into(),
        );
        return;
    }
    const XLV: f32 = 2.5e6;
    const RV: f32 = 461.5;
    let grid = dewpt.grid.clone();
    let values: Vec<f32> = temp
        .values
        .iter()
        .zip(dewpt.values.iter())
        .map(|(t, dp)| (XLV / RV * (1.0 / t - 1.0 / dp)).exp() * 1.0e2)
        .collect();
    let points = values.len();
    collector.insert(Slab { name: "RH".into(), xlvl: XLVL_SURFACE, grid, values });
    ledger.on(
        "surface_rh_from_dewpoint",
        "no surface RH in the file; surface TT and DEWPT present".into(),
        points,
        "RH = exp(Lv/Rv * (1/T - 1/Td)) * 100, unclamped, as compute_rh_dewpt writes it".into(),
    );
}

/// `rrpr.F` `fix_gfs_miss`: repair the fill values NCEP writes over
/// ocean for masked surface fields.
///
/// Since July 2017 GFS writes something enormous rather than a missing
/// code over water.  Carried through, those values are not obviously
/// wrong -- they are finite, and they survive every finiteness check --
/// but a soil temperature of 10^20 poisons whatever consumes it.
fn masked_surface_fields(collector: &mut Collector, source: &MapSource, ledger: &mut RuleLedger) {
    if !source.masked_surface_repair {
        ledger.off(
            "masked_surface_fields",
            format!("{} does not write the 2017 GFS masked-field convention", source.header),
        );
        return;
    }
    // ungrib gates the repair on the field whose presence tells it this
    // is a file with the new convention.
    if !collector.has_surface("ST000010") {
        ledger.off(
            "masked_surface_fields",
            format!("{} selects the repair, but the file has no ST000010", source.header),
        );
        return;
    }

    let soil_fields = [
        "ST000010", "ST010040", "ST040100", "ST100200", "ST010200",
        "SM000010", "SM010040", "SM040100", "SM100200", "SM010200",
    ];
    let mut filled = 0usize;
    let mut clipped = 0usize;
    for name in soil_fields {
        let Some(slab) = collector
            .slabs
            .get_mut(&(name.to_string(), OrderedF32(XLVL_SURFACE)))
        else {
            continue;
        };
        let is_moisture = name.starts_with("SM");
        for value in slab.values.iter_mut() {
            if value.abs() > MISSING_MAGNITUDE {
                *value = SOIL_FILL;
                filled += 1;
            } else if is_moisture && *value > SOIL_MOISTURE_CEILING {
                *value = SOIL_MOISTURE_CEILING;
                clipped += 1;
            }
        }
    }

    // Snow carries a different convention: its masked points become
    // zero, not the soil fill, because zero snow is the physical truth
    // over open water.
    let mut snow_zeroed = 0usize;
    for name in ["SNOW", "SNOWH"] {
        let Some(slab) = collector
            .slabs
            .get_mut(&(name.to_string(), OrderedF32(XLVL_SURFACE)))
        else {
            continue;
        };
        for value in slab.values.iter_mut() {
            if value.abs() > MISSING_MAGNITUDE {
                *value = 0.0;
                snow_zeroed += 1;
            }
        }
    }

    ledger.on(
        "masked_surface_fields",
        format!("{} writes the 2017 GFS masked-field convention", source.header),
        filled + clipped + snow_zeroed,
        format!(
            "{filled} soil points to the WPS fill, {clipped} soil-moisture points clipped \
             to 0.468, {snow_zeroed} snow points to zero"
        ),
    );
}

/// `rrpr.F`: "ECMWF snow depth in meters of water equivalent (Table
/// 128). Convert to kg/m2" -- `SNOW = SNOW_EC * 1000`.
///
/// The density of water, not the reported snow density: SNOW_EC is a
/// depth of *water*, and SNOW_DEN belongs to the snow-depth rule
/// further down.  Multiplying by the reported density instead is wrong
/// by a factor of three or four at every snow-covered point.
fn snow_from_ecmwf_water_equivalent(collector: &mut Collector, ledger: &mut RuleLedger) {
    let Some(src) = collector.surface("SNOW_EC") else {
        ledger.off("snow_from_snow_ec", "no SNOW_EC in the file".into());
        return;
    };
    let grid = src.grid.clone();
    let values: Vec<f32> = src.values.iter().map(|v| v * 1000.0).collect();
    let points = values.len();
    let overwrote = collector.has_surface("SNOW");
    collector.insert(Slab { name: "SNOW".into(), xlvl: XLVL_SURFACE, grid, values });
    ledger.on(
        "snow_from_snow_ec",
        "SNOW_EC present (not source-gated)".into(),
        points,
        format!(
            "SNOW = SNOW_EC * 1000 (metres of water equivalent to kg m-2){}",
            if overwrote { ", overwriting the decoded SNOW" } else { "" }
        ),
    );
}

/// `rrpr.F`: "Convert the ECMWF LANDSEA mask from a fraction to a flag"
/// (`make_zero_or_one`).
fn landsea_fraction_to_flag(
    collector: &mut Collector,
    source: &MapSource,
    ledger: &mut RuleLedger,
) {
    if !source.landsea_to_flag {
        ledger.off(
            "landsea_fraction_to_flag",
            format!("{} already publishes LANDSEA as a flag", source.header),
        );
        return;
    }
    let Some(slab) = collector
        .slabs
        .get_mut(&("LANDSEA".to_string(), OrderedF32(XLVL_SURFACE)))
    else {
        ledger.off(
            "landsea_fraction_to_flag",
            format!("{} selects the conversion, but the file has no LANDSEA", source.header),
        );
        return;
    };
    let mut moved = 0usize;
    for value in slab.values.iter_mut() {
        let flag = if *value > 0.5 { 1.0 } else { 0.0 };
        if flag != *value {
            moved += 1;
        }
        *value = flag;
    }
    ledger.on(
        "landsea_fraction_to_flag",
        format!("{} publishes LANDSEA as a fraction", source.header),
        moved,
        "LANDSEA > 0.5 becomes 1, everything else 0".into(),
    );
}

/// `rrpr.F`: "NCEP GFS weasd is one-half of the NAM value. Increase it
/// for use in WRF."
///
/// This is the whole of the factor-of-two disagreement an independent
/// ecCodes decode found between this tool and `ungrib` on the GFS arm:
/// `ungrib` is not misreading the message, it is deliberately doubling
/// it, and every native init built from a GFS analysis carries the
/// doubled value.  Whether the doubling is still right for the modern
/// GFS is a question for the centre; reproducing it is what keeps this
/// tool a drop-in for the file `ungrib` writes.
fn snow_water_equivalent_doubling(
    collector: &mut Collector,
    source: &MapSource,
    ledger: &mut RuleLedger,
) {
    if !source.snow_water_equivalent_doubled {
        ledger.off(
            "snow_water_equivalent_doubling",
            format!("{} publishes snow water equivalent unhalved", source.header),
        );
        return;
    }
    let Some(slab) = collector
        .slabs
        .get_mut(&("SNOW".to_string(), OrderedF32(XLVL_SURFACE)))
    else {
        ledger.off(
            "snow_water_equivalent_doubling",
            format!("{} selects the doubling, but the file has no SNOW", source.header),
        );
        return;
    };
    let mut touched = 0usize;
    for value in slab.values.iter_mut() {
        if *value != 0.0 {
            touched += 1;
        }
        *value *= 2.0;
    }
    ledger.on(
        "snow_water_equivalent_doubling",
        format!("{} reports weasd at half the NAM value", source.header),
        touched,
        "SNOW = SNOW * 2, as rrpr.F does for NCEP GFS and GEFS".into(),
    );
}

/// `rrpr.F`: physical snow depth from water-equivalent snow, by the
/// producing centre's assumed density.
///
/// A wrong branch here is a silently wrong snow depth over every
/// snow-covered point, which is why the branch that fired is named in
/// the receipt rather than left for a reader to infer.
fn snow_depth_from_snow(collector: &mut Collector, source: &MapSource, ledger: &mut RuleLedger) {
    if collector.has_surface("SNOWH") {
        ledger.off("snowh_from_snow", "the file already carries SNOWH".into());
        return;
    }
    let Some(snow) = collector.surface("SNOW") else {
        ledger.off("snowh_from_snow", "no SNOW to derive a depth from".into());
        return;
    };
    let grid = snow.grid.clone();
    let water_equivalent = snow.values.clone();
    let density = collector.surface("SNOW_DEN").map(|s| s.values.clone());

    let (values, detail) = match source.snow_depth {
        SnowDepthRule::TwoHundredToOne => (
            water_equivalent.iter().map(|m| m * 0.005).collect::<Vec<f32>>(),
            "SNOWH = SNOW * 0.005 (200:1, as NCEP and NOAH assume)".to_string(),
        ),
        SnowDepthRule::EcmwfReportedDensity => match density {
            Some(den) if den.len() == water_equivalent.len() => (
                water_equivalent
                    .iter()
                    .zip(den.iter())
                    .map(|(m, d)| m / d)
                    .collect::<Vec<f32>>(),
                "SNOWH = SNOW / SNOW_DEN (the reported density)".to_string(),
            ),
            _ => (
                water_equivalent.iter().map(|m| m * 0.004).collect(),
                "SNOWH = SNOW * 0.004 (250:1; no SNOW_DEN in the file)".to_string(),
            ),
        },
    };

    let points = values.len();
    collector.insert(Slab { name: "SNOWH".into(), xlvl: XLVL_SURFACE, grid, values });
    ledger.on(
        "snowh_from_snow",
        format!("SNOW present without SNOWH; {} density rule", source.header),
        points,
        detail,
    );
}

/// `rrpr.F` `gfs_trop_maxw_pressures`: the max-wind and tropopause
/// pressures are duplicated under a second name, because `metgrid`
/// picks an interpolation method from a field's *name* and these two
/// must not be smoothed across the discontinuity.  The values are a
/// copy on purpose: the point of the twin is the name.
fn nearest_neighbour_pressure_twins(
    collector: &mut Collector,
    source: &MapSource,
    ledger: &mut RuleLedger,
) {
    if !source.nearest_neighbour_pressure_twins {
        ledger.off(
            "nearest_neighbour_pressure_twins",
            format!("rrpr.F duplicates these only for NCEP GFS, not {}", source.header),
        );
        return;
    }
    let mut made = Vec::new();
    let mut points = 0usize;
    for (from, to) in [("PMAXW", "PMAXWNN"), ("PTROP", "PTROPNN")] {
        let Some(original) = collector.surface(from) else { continue };
        let copy = Slab {
            name: to.to_string(),
            xlvl: XLVL_SURFACE,
            grid: original.grid.clone(),
            values: original.values.clone(),
        };
        points += copy.values.len();
        collector.insert(copy);
        made.push(format!("{to} from {from}"));
    }
    if made.is_empty() {
        ledger.off(
            "nearest_neighbour_pressure_twins",
            format!("{} selects the twins, but the file has neither PMAXW nor PTROP", source.header),
        );
        return;
    }
    ledger.on(
        "nearest_neighbour_pressure_twins",
        format!("{} carries max-wind and tropopause pressures", source.header),
        points,
        made.join(", "),
    );
}

// ---------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------

/// One record the file will carry.
struct Emission<'a> {
    meta: &'a FieldMeta,
    slab: &'a Slab,
}

/// The records `output.F` would write, in the order it would write
/// them: levels descending (`get_plvls` sorts largest first), and
/// within a level the Vtable's field order, first occurrence only,
/// skipping the rows whose Description column is blank.
fn emission_order<'a>(
    collector: &'a Collector,
    metas: &'a [FieldMeta],
) -> (Vec<Emission<'a>>, usize) {
    let mut out = Vec::new();
    let mut held = 0usize;
    for level in collector.levels_descending() {
        for meta in metas {
            let Some(slab) = collector.get(&meta.name, level) else { continue };
            if meta.is_published() {
                out.push(Emission { meta, slab });
            } else {
                held += 1;
            }
        }
    }
    (out, held)
}

/// Write a Fortran sequential unformatted record: a big-endian byte
/// count, the payload, then the count again.
fn write_record<W: Write>(w: &mut W, payload: &[u8]) -> std::io::Result<()> {
    let n = payload.len() as u32;
    w.write_all(&n.to_be_bytes())?;
    w.write_all(payload)?;
    w.write_all(&n.to_be_bytes())
}

fn fixed(text: &str, width: usize) -> Vec<u8> {
    let mut out = text.as_bytes().to_vec();
    out.truncate(width);
    out.resize(width, b' ');
    out
}

fn write_slab<W: Write>(
    w: &mut W,
    emission: &Emission<'_>,
    hdate: &str,
    map_source: &str,
) -> std::io::Result<()> {
    let slab = emission.slab;
    write_record(w, &WPS_FORMAT_VERSION.to_be_bytes())?;

    let mut header = Vec::new();
    header.extend_from_slice(&fixed(hdate, 24));
    header.extend_from_slice(&0f32.to_be_bytes()); // xfcst: analysis
    header.extend_from_slice(&fixed(map_source, 32));
    header.extend_from_slice(&fixed(&emission.meta.name, 9));
    header.extend_from_slice(&fixed(&emission.meta.units, 25));
    header.extend_from_slice(&fixed(&emission.meta.desc, 46));
    header.extend_from_slice(&slab.xlvl.to_be_bytes());
    header.extend_from_slice(&(slab.grid.nx as i32).to_be_bytes());
    header.extend_from_slice(&(slab.grid.ny as i32).to_be_bytes());
    header.extend_from_slice(&slab.grid.iproj.to_be_bytes());
    write_record(w, &header)?;

    let mut proj = Vec::new();
    proj.extend_from_slice(&fixed("SWCORNER", 8));
    proj.extend_from_slice(&slab.grid.startlat.to_be_bytes());
    proj.extend_from_slice(&slab.grid.startlon.to_be_bytes());
    proj.extend_from_slice(&slab.grid.deltalat.to_be_bytes());
    proj.extend_from_slice(&slab.grid.deltalon.to_be_bytes());
    proj.extend_from_slice(&slab.grid.earth_radius_km.to_be_bytes());
    write_record(w, &proj)?;

    // is_wind_grid_rel: these are earth-relative global analyses.
    write_record(w, &0i32.to_be_bytes())?;

    let mut data = Vec::with_capacity(slab.values.len() * 4);
    for v in &slab.values {
        data.extend_from_slice(&v.to_be_bytes());
    }
    write_record(w, &data)
}

// ---------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------

fn usage() -> String {
    format!(
        "usage: met_intermediate --vtable VTABLE --date YYYY-MM-DD_HH:MM:SS \
         --map-source LABEL --out OUTFILE [--pmin PA] [--select-valid-time] \
         [--invariant GRIB ... --invariant-field NAME ...] GRIB [GRIB ...]\n\
         --map-source is required; known labels: {}",
        known_map_source_list()
    )
}

/// The command line, once every argument has been resolved to
/// something that cannot be misread later.
#[derive(Debug)]
struct Config {
    vtable_path: PathBuf,
    date: String,
    /// `date` as a time, parsed before any decode.
    valid_time: NaiveDateTime,
    out_path: PathBuf,
    source: &'static MapSource,
    pmin_pa: f64,
    /// Drop matched messages not valid at `date`.
    select_valid_time: bool,
    inputs: Vec<PathBuf>,
    /// Files that may supply only `invariant_fields`.
    invariants: Vec<PathBuf>,
    invariant_fields: Vec<String>,
}

/// Returned when `--help` asked for the usage text rather than a run.
#[derive(Debug)]
enum Invocation {
    Run(Box<Config>),
    Help,
}

fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<Invocation, String> {
    let mut vtable_path: Option<PathBuf> = None;
    let mut date: Option<String> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut map_source: Option<String> = None;
    let mut pmin_pa = DEFAULT_PMIN_PA;
    let mut select_valid_time = false;
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut invariants: Vec<PathBuf> = Vec::new();
    let mut invariant_fields: Vec<String> = Vec::new();

    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--vtable" => vtable_path = Some(PathBuf::from(args.next().ok_or_else(usage)?)),
            "--date" => date = Some(args.next().ok_or_else(usage)?),
            "--out" => out_path = Some(PathBuf::from(args.next().ok_or_else(usage)?)),
            "--map-source" => map_source = Some(args.next().ok_or_else(usage)?),
            "--pmin" => {
                let text = args.next().ok_or_else(usage)?;
                pmin_pa = text
                    .parse::<f64>()
                    .map_err(|_| format!("--pmin wants pascals, not {text:?}"))?;
            }
            "--select-valid-time" => select_valid_time = true,
            "--invariant" => invariants.push(PathBuf::from(args.next().ok_or_else(usage)?)),
            "--invariant-field" => {
                let name = args.next().ok_or_else(usage)?;
                if name.trim().is_empty() {
                    return Err("--invariant-field wants a field name".to_string());
                }
                invariant_fields.push(name.trim().to_string());
            }
            "--help" | "-h" => return Ok(Invocation::Help),
            other if other.starts_with("--") => {
                return Err(format!("unknown option {other}\n{}", usage()));
            }
            other => inputs.push(PathBuf::from(other)),
        }
    }

    // The producing centre is resolved here, before a single message is
    // decoded, so a run either knows which repairs it is applying or
    // does not start.
    let source = match map_source {
        Some(label) => resolve_map_source(&label)?,
        None => {
            return Err(map_source_refusal(
                "--map-source was not given, and this tool has no default source.",
            ))
        }
    };

    let vtable_path = vtable_path.ok_or_else(usage)?;
    let date = date.ok_or_else(usage)?;
    let out_path = out_path.ok_or_else(usage)?;
    if inputs.is_empty() {
        return Err(format!("no GRIB inputs given\n{}", usage()));
    }
    let valid_time = parse_hdate(&date)?;
    if invariants.is_empty() != invariant_fields.is_empty() {
        return Err(
            "--invariant and --invariant-field come together: an invariant file supplies \
             only the fields named for it, and a named field needs a file to come from"
                .to_string(),
        );
    }
    Ok(Invocation::Run(Box::new(Config {
        vtable_path,
        date,
        valid_time,
        out_path,
        source,
        pmin_pa,
        select_valid_time,
        inputs,
        invariants,
        invariant_fields,
    })))
}

fn main() {
    // Keep the release-provenance stamp in the binary: the bundle cut
    // proves a staged bridge by these bytes (see lib.rs).
    let _ = std::hint::black_box(gpuwm_preprocess_cpu::SOURCE_REV_STAMP);
    if let Err(message) = run() {
        eprintln!("met_intermediate: {message}");
        std::process::exit(1);
    }
}

fn json_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

fn json_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|p| format!("\"{}\"", json_escape(&p.display().to_string())))
        .collect::<Vec<_>>()
        .join(", ")
}

fn run() -> Result<(), String> {
    let config = match parse_args(env::args().skip(1))? {
        Invocation::Help => {
            println!("{}", usage());
            return Ok(());
        }
        Invocation::Run(config) => *config,
    };

    let vtable = parse_vtable(&config.vtable_path)?;
    let metas = field_metas(&vtable);
    let mut collector = Collector::new(config.pmin_pa);
    if config.select_valid_time {
        collector.select_valid_time = Some(config.valid_time);
    }

    for input in &config.inputs {
        collect_file(input, &vtable, config.source, &mut collector)?;
    }
    if collector.select_valid_time.is_some()
        && collector.slabs.is_empty()
        && collector.messages_other_valid_time > 0
    {
        return Err(format!(
            "--select-valid-time {}: {} matched messages are valid at other times and none \
             at this one; the inputs do not hold this time",
            config.date, collector.messages_other_valid_time
        ));
    }
    let invariant_sources = merge_invariants(&config, &vtable, &mut collector)?;

    let mut ledger = RuleLedger::default();
    apply_ungrib_rules(&mut collector, config.source, &mut ledger);

    if collector.slabs.is_empty() {
        return Err(format!(
            "no field in {} matched any of the {} Vtable rows across {} GRIB messages; \
             refusing to write an empty intermediate file",
            config
                .inputs
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", "),
            vtable.len(),
            collector.messages_seen
        ));
    }

    let (emissions, held) = emission_order(&collector, &metas);
    if emissions.is_empty() {
        return Err(format!(
            "every one of the {} decoded fields is held back by its Vtable row's blank \
             Description column; refusing to write an empty intermediate file",
            collector.slabs.len()
        ));
    }

    let file = File::create(&config.out_path)
        .map_err(|e| format!("cannot write {}: {e}", config.out_path.display()))?;
    let mut writer = BufWriter::new(file);
    for emission in &emissions {
        write_slab(&mut writer, emission, &config.date, config.source.header)
            .map_err(|e| format!("writing {} at level {}: {e}", emission.meta.name, emission.slab.xlvl))?;
    }
    writer
        .flush()
        .map_err(|e| format!("flushing {}: {e}", config.out_path.display()))?;

    // The report is on stdout so a caller can capture it as a receipt.
    println!("{{");
    println!("  \"schema\": \"{RECEIPT_SCHEMA}\",");
    println!("  \"out\": \"{}\",", json_escape(&config.out_path.display().to_string()));
    println!("  \"format_version\": {WPS_FORMAT_VERSION},");
    println!("  \"hdate\": \"{}\",", json_escape(&config.date));
    println!("  \"map_source\": {{");
    println!("    \"label\": \"{}\",", config.source.label);
    println!("    \"header_text\": \"{}\",", json_escape(config.source.header));
    println!(
        "    \"grib1_earth_radius_km\": {}",
        config.source.grib1_earth_radius_km
    );
    println!("  }},");
    println!("  \"vtable\": \"{}\",", json_escape(&config.vtable_path.display().to_string()));
    println!("  \"vtable_rows\": {},", vtable.len());
    println!("  \"vtable_field_names\": {},", metas.len());
    println!("  \"grib_messages_read\": {},", collector.messages_seen);
    println!("  \"grib_messages_matched\": {},", collector.messages_matched);
    println!("  \"records_written\": {},", emissions.len());
    println!(
        "  \"records_held_blank_description\": {held},"
    );
    println!("  \"pmin_pa\": {},", config.pmin_pa);
    println!(
        "  \"isobaric_levels_dropped_above_pmin\": {},",
        collector.levels_below_pmin
    );
    println!("  \"bitmap_masked_points\": {},", collector.masked_points);
    println!(
        "  \"valid_time_selection\": \"{}\",",
        if config.select_valid_time { "select" } else { "all" }
    );
    println!(
        "  \"messages_other_valid_time\": {},",
        collector.messages_other_valid_time
    );
    println!(
        "  \"messages_time_processed_skipped\": {},",
        collector.messages_time_processed
    );
    println!(
        "  \"matched_valid_times\": [{}],",
        collector
            .matched_valid_times
            .iter()
            .map(|t| format!("\"{}\"", format_hdate(t)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  \"inputs\": [{}],", json_paths(&config.inputs));
    println!(
        "  \"invariants\": {{\"files\": [{}], \"fields\": [{}]}},",
        json_paths(&config.invariants),
        invariant_sources
            .iter()
            .map(|(name, from)| format!(
                "{{\"field\": \"{}\", \"from\": \"{from}\"}}",
                json_escape(name)
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  \"masked_point_value\": {MASKED_SENTINEL:e},");
    println!("  \"rules\": [");
    for (i, decision) in ledger.entries.iter().enumerate() {
        let comma = if i + 1 == ledger.entries.len() { "" } else { "," };
        println!(
            "    {{\"rule\": \"{}\", \"applied\": {}, \"basis\": \"{}\", \"points\": {}, \
             \"detail\": \"{}\"}}{comma}",
            decision.rule,
            decision.applied,
            json_escape(&decision.basis),
            decision.points,
            json_escape(&decision.detail)
        );
    }
    println!("  ],");
    println!("  \"fields\": [");
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for emission in &emissions {
        *counts.entry(emission.meta.name.as_str()).or_insert(0) += 1;
    }
    for (i, (name, count)) in counts.iter().enumerate() {
        let comma = if i + 1 == counts.len() { "" } else { "," };
        println!("    {{\"field\": \"{name}\", \"levels\": {count}}}{comma}");
    }
    println!("  ],");
    println!("  \"fields_held\": [");
    let held_names: Vec<&FieldMeta> = metas
        .iter()
        .filter(|m| !m.is_published())
        .filter(|m| collector.slabs.keys().any(|(n, _)| n == &m.name))
        .collect();
    for (i, meta) in held_names.iter().enumerate() {
        let comma = if i + 1 == held_names.len() { "" } else { "," };
        println!(
            "    {{\"field\": \"{}\", \"reason\": \"the Vtable row has a blank Description \
             column, so ungrib consumes it and does not publish it\"}}{comma}",
            meta.name
        );
    }
    println!("  ]");
    println!("}}");

    Ok(())
}

/// Decode one GRIB file of either edition into `out`.
fn collect_file(
    input: &Path,
    vtable: &[VtableRow],
    source: &MapSource,
    out: &mut Collector,
) -> Result<(), String> {
    match grib_edition(input)? {
        1 => collect_grib1(input, vtable, source, out),
        2 => collect_grib2(input, vtable, out),
        other => Err(format!(
            "{}: GRIB edition {other} is not supported",
            input.display()
        )),
    }
}

/// Fill the named invariant fields the main inputs lack from the
/// `--invariant` files, and say where each one came from.
///
/// An invariant file is decoded into its own store with no valid-time
/// selection (a land mask published once at hour zero is the point),
/// and only the named fields cross over.  A named field the main inputs
/// already carry keeps the main inputs' value; one that neither carries
/// is a refusal; one on a different grid from the rest of the file is a
/// refusal, because a record header cannot say "this one is elsewhere".
fn merge_invariants(
    config: &Config,
    vtable: &[VtableRow],
    collector: &mut Collector,
) -> Result<Vec<(String, &'static str)>, String> {
    let mut decided: Vec<(String, &'static str)> = Vec::new();
    if config.invariants.is_empty() {
        return Ok(decided);
    }
    // Decode only what is still needed, and only its rows: the invariant
    // file is a whole hour-zero product, and unpacking every matched
    // message in it to keep two fields would double each lead's decode.
    let needed: Vec<&String> = config
        .invariant_fields
        .iter()
        .filter(|name| !collector.has_field(name))
        .collect();
    let rows: Vec<VtableRow> = vtable
        .iter()
        .filter(|row| needed.iter().any(|name| **name == row.name))
        .cloned()
        .collect();
    let mut store = Collector::new(config.pmin_pa);
    if !rows.is_empty() {
        for input in &config.invariants {
            collect_file(input, &rows, config.source, &mut store)?;
        }
    }
    let reference_grid = collector.slabs.values().next().map(|s| s.grid.clone());
    for name in &config.invariant_fields {
        if collector.has_field(name) {
            decided.push((name.clone(), "inputs"));
            continue;
        }
        let keys: Vec<SlabKey> = store
            .slabs
            .keys()
            .filter(|(n, _)| n == name)
            .cloned()
            .collect();
        if keys.is_empty() {
            return Err(format!(
                "--invariant-field {name}: neither the inputs nor the invariant files ({}) \
                 carry it under any Vtable row",
                json_paths(&config.invariants)
            ));
        }
        for key in keys {
            let slab = store.slabs.remove(&key).expect("key listed above");
            if let Some(grid) = &reference_grid {
                if &slab.grid != grid {
                    return Err(format!(
                        "--invariant-field {name}: the invariant file's grid ({}x{}, start \
                         {},{} step {},{}) is not the inputs' grid ({}x{}, start {},{} step \
                         {},{}); one intermediate file has one grid",
                        slab.grid.nx, slab.grid.ny, slab.grid.startlat, slab.grid.startlon,
                        slab.grid.deltalat, slab.grid.deltalon, grid.nx, grid.ny,
                        grid.startlat, grid.startlon, grid.deltalat, grid.deltalon
                    ));
                }
            }
            collector.insert(slab);
        }
        decided.push((name.clone(), "invariant"));
    }
    Ok(decided)
}

/// Read the edition byte from a GRIB envelope.
fn grib_edition(path: &Path) -> Result<u8, String> {
    use std::io::Read;
    let mut file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut head = [0u8; 8];
    file.read_exact(&mut head)
        .map_err(|e| format!("cannot read the GRIB envelope of {}: {e}", path.display()))?;
    if &head[0..4] != b"GRIB" {
        return Err(format!("{} does not begin with a GRIB envelope", path.display()));
    }
    Ok(head[7])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn grid(nx: usize, ny: usize) -> SlabGrid {
        SlabGrid {
            iproj: 0,
            nx,
            ny,
            startlat: 0.0,
            startlon: 0.0,
            deltalat: 1.0,
            deltalon: 1.0,
            earth_radius_km: 6367.47,
        }
    }

    fn slab(name: &str, xlvl: f32, values: Vec<f32>) -> Slab {
        let n = values.len();
        Slab { name: name.into(), xlvl, grid: grid(n, 1), values }
    }

    fn ecmwf() -> &'static MapSource {
        resolve_map_source("ecmwf").unwrap()
    }

    fn gfs() -> &'static MapSource {
        resolve_map_source("ncep-gfs").unwrap()
    }

    fn decision<'a>(ledger: &'a RuleLedger, rule: &str) -> &'a RuleDecision {
        ledger
            .entries
            .iter()
            .find(|d| d.rule == rule)
            .unwrap_or_else(|| panic!("no decision recorded for {rule}"))
    }

    // -----------------------------------------------------------------
    // The map source is a closed vocabulary with no default
    // -----------------------------------------------------------------

    #[test]
    fn a_run_without_a_map_source_refuses_and_names_the_labels() {
        // The silent path this replaces: --map-source used to default to
        // a label no rule matched, so every repair quietly stayed off and
        // the tool still exited zero.
        let err = parse_args(args(&[
            "--vtable", "V", "--date", "2025-03-14_12:00:00", "--out", "O", "a.grib",
        ]))
        .unwrap_err();
        assert!(err.contains("no default source"), "{err}");
        for source in KNOWN_MAP_SOURCES {
            assert!(err.contains(source.label), "refusal omits {}: {err}", source.label);
        }
    }

    #[test]
    fn the_old_silent_default_label_is_now_a_refusal() {
        // "rw-wps" was the default.  It matched none of ungrib's
        // substring tests, so it disabled the RH over-ice conversion,
        // the soil bitmap repair and the snow rules at once -- and said
        // nothing.  It must not resolve.
        let err = resolve_map_source("rw-wps").unwrap_err();
        assert!(err.contains("unknown --map-source \"rw-wps\""), "{err}");
        assert!(err.contains("no default and no fallback"), "{err}");
        let err = parse_args(args(&[
            "--vtable", "V", "--date", "D", "--out", "O", "--map-source", "rw-wps", "a.grib",
        ]))
        .unwrap_err();
        assert!(err.contains("ncep-gfs"), "{err}");
    }

    // -----------------------------------------------------------------
    // Time identity and invariant files
    // -----------------------------------------------------------------

    fn base_args() -> Vec<&'static str> {
        vec![
            "--vtable", "V", "--date", "2026-09-24_15:00:00", "--out", "O",
            "--map-source", "ecmwf",
        ]
    }

    #[test]
    fn the_date_is_parsed_before_any_decode() {
        let mut list = base_args();
        list[3] = "2026-09-24T15";
        list.push("a.grib");
        let err = parse_args(args(&list)).unwrap_err();
        assert!(err.contains("is not YYYY-MM-DD_HH:MM:SS"), "{err}");
        let mut list = base_args();
        list[3] = "2026-02-30_00:00:00";
        list.push("a.grib");
        assert!(parse_args(args(&list)).is_err());
    }

    #[test]
    fn select_valid_time_and_invariants_parse() {
        let mut list = base_args();
        list.extend([
            "--select-valid-time", "--invariant", "f0.grib2", "--invariant-field", "LANDSEA",
            "--invariant-field", "SOILGEO", "a.grib2",
        ]);
        let Invocation::Run(config) = parse_args(args(&list)).unwrap() else {
            panic!("help")
        };
        assert!(config.select_valid_time);
        assert_eq!(config.invariants, vec![PathBuf::from("f0.grib2")]);
        assert_eq!(config.invariant_fields, vec!["LANDSEA", "SOILGEO"]);
        assert_eq!(config.inputs, vec![PathBuf::from("a.grib2")]);
        assert_eq!(format_hdate(&config.valid_time), "2026-09-24_15:00:00");
    }

    #[test]
    fn an_invariant_file_without_named_fields_refuses() {
        let mut list = base_args();
        list.extend(["--invariant", "f0.grib2", "a.grib2"]);
        let err = parse_args(args(&list)).unwrap_err();
        assert!(err.contains("come together"), "{err}");
        let mut list = base_args();
        list.extend(["--invariant-field", "LANDSEA", "a.grib2"]);
        assert!(parse_args(args(&list)).unwrap_err().contains("come together"));
    }

    #[test]
    fn time_units_convert_or_refuse() {
        assert_eq!(time_unit_seconds(1, 2).unwrap(), 3600);
        assert_eq!(time_unit_seconds(11, 1).unwrap(), 21600);
        assert_eq!(time_unit_seconds(13, 2).unwrap(), 1);
        assert_eq!(time_unit_seconds(254, 1).unwrap(), 1);
        assert!(time_unit_seconds(13, 1).is_err());
        assert!(time_unit_seconds(7, 2).unwrap_err().contains("refusing"));
    }

    #[test]
    fn selection_drops_other_times_and_records_the_admitted_one() {
        let mut collector = Collector::new(DEFAULT_PMIN_PA);
        let want = parse_hdate("2025-03-14_12:00:00").unwrap();
        collector.select_valid_time = Some(want);
        assert!(!collector.admit_valid_time(parse_hdate("2025-03-14_06:00:00").unwrap()));
        assert!(collector.admit_valid_time(want));
        assert_eq!(collector.messages_other_valid_time, 1);
        assert_eq!(collector.matched_valid_times.len(), 1);
        // Without selection every time is admitted and every one is
        // reported, so a caller can see a mixed file.
        let mut open = Collector::new(DEFAULT_PMIN_PA);
        assert!(open.admit_valid_time(parse_hdate("2025-03-14_06:00:00").unwrap()));
        assert!(open.admit_valid_time(want));
        assert_eq!(open.matched_valid_times.len(), 2);
    }

    #[test]
    fn a_known_label_resolves_to_the_header_text_ungrib_writes() {
        assert_eq!(resolve_map_source("ecmwf").unwrap().header, "ECMWF");
        assert_eq!(resolve_map_source("NCEP-GFS").unwrap().header, "NCEP GFS Analysis");
        assert_eq!(resolve_map_source(" ecmwf ").unwrap().header, "ECMWF");
    }

    #[test]
    fn every_repair_is_decided_by_the_resolved_source_not_by_a_substring() {
        // Each centre answers every question; there is no arm that can
        // fall through to "apply nothing".
        for source in KNOWN_MAP_SOURCES {
            let _ = source.rh_over_ice;
            let _ = source.masked_surface_repair;
            let _ = source.snow_water_equivalent_doubled;
            let _ = source.landsea_to_flag;
            let _ = source.nearest_neighbour_pressure_twins;
            assert!(source.grib1_earth_radius_km > 6000.0);
        }
        assert!(gfs().rh_over_ice && gfs().masked_surface_repair && gfs().snow_water_equivalent_doubled);
        assert!(ecmwf().landsea_to_flag && !ecmwf().masked_surface_repair);
    }

    #[test]
    fn every_rule_reports_whether_it_ran_including_the_ones_that_did_not() {
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("TT", XLVL_SURFACE, vec![280.0, 290.0]));
        let mut ledger = RuleLedger::default();
        apply_ungrib_rules(&mut c, ecmwf(), &mut ledger);
        for rule in [
            "hgt_from_geopt",
            "rh_over_ice_to_liquid",
            "surface_rh_from_dewpoint",
            "masked_surface_fields",
            "soilhgt_from_soilgeo",
            "snow_from_snow_ec",
            "landsea_fraction_to_flag",
            "snow_water_equivalent_doubling",
            "snowh_from_snow",
            "nearest_neighbour_pressure_twins",
        ] {
            let d = decision(&ledger, rule);
            assert!(!d.basis.is_empty(), "{rule} reports no basis");
        }
        // And the ones that stayed off say which centre turned them off.
        assert!(decision(&ledger, "masked_surface_fields").basis.contains("ECMWF"));
        assert!(!decision(&ledger, "masked_surface_fields").applied);
    }

    // -----------------------------------------------------------------
    // The rules themselves
    // -----------------------------------------------------------------

    #[test]
    fn ecmwf_snow_water_equivalent_is_a_depth_of_water_not_of_snow() {
        // SNOW_EC is metres of water equivalent; rrpr.F multiplies by
        // 1000, the density of water.  Multiplying by the reported snow
        // density instead is wrong by three to four at every snowy point.
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("SNOW_EC", XLVL_SURFACE, vec![10.0, 0.0]));
        c.insert(slab("SNOW_DEN", XLVL_SURFACE, vec![300.0, 300.0]));
        let mut ledger = RuleLedger::default();
        apply_ungrib_rules(&mut c, ecmwf(), &mut ledger);
        assert_eq!(c.surface("SNOW").unwrap().values, vec![10_000.0, 0.0]);
        // and the depth then divides by the reported density
        assert_eq!(c.surface("SNOWH").unwrap().values[0], 10_000.0f32 / 300.0);
    }

    #[test]
    fn the_gfs_snow_doubling_is_reproduced_and_named() {
        // rrpr.F: "NCEP GFS weasd is one-half of the NAM value."  This
        // is the factor of two an independent ecCodes decode found
        // between this tool and ungrib; it is ungrib's rule, not a
        // decode error, and every GFS-based init carries it.
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("SNOW", XLVL_SURFACE, vec![0.656, 0.0]));
        let mut ledger = RuleLedger::default();
        apply_ungrib_rules(&mut c, gfs(), &mut ledger);
        assert_eq!(c.surface("SNOW").unwrap().values, vec![1.312, 0.0]);
        assert_eq!(c.surface("SNOWH").unwrap().values[0], 1.312f32 * 0.005);
        assert!(decision(&ledger, "snow_water_equivalent_doubling").applied);
        // ECMWF must not inherit it.
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("SNOW", XLVL_SURFACE, vec![0.656]));
        let mut ledger = RuleLedger::default();
        apply_ungrib_rules(&mut c, ecmwf(), &mut ledger);
        assert_eq!(c.surface("SNOW").unwrap().values, vec![0.656]);
        assert!(!decision(&ledger, "snow_water_equivalent_doubling").applied);
    }

    #[test]
    fn the_ecmwf_land_sea_fraction_becomes_a_flag_and_gfs_is_left_alone() {
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("LANDSEA", XLVL_SURFACE, vec![0.0, 0.4999, 0.5, 0.5001, 1.0]));
        let mut ledger = RuleLedger::default();
        apply_ungrib_rules(&mut c, ecmwf(), &mut ledger);
        assert_eq!(
            c.surface("LANDSEA").unwrap().values,
            vec![0.0, 0.0, 0.0, 1.0, 1.0]
        );
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("LANDSEA", XLVL_SURFACE, vec![0.4999]));
        let mut ledger = RuleLedger::default();
        apply_ungrib_rules(&mut c, gfs(), &mut ledger);
        assert_eq!(c.surface("LANDSEA").unwrap().values, vec![0.4999]);
    }

    #[test]
    fn surface_humidity_uses_the_formula_wps_uses_and_does_not_clamp() {
        // compute_rh_dewpt: exp(Lv/Rv*(1/T - 1/Td)) * 100, no clamp.  A
        // saturated-ish point with Td above T therefore reports above
        // 100, exactly as ungrib's own file does.
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("TT", XLVL_SURFACE, vec![300.0, 300.0]));
        c.insert(slab("DEWPT", XLVL_SURFACE, vec![290.0, 301.0]));
        let mut ledger = RuleLedger::default();
        apply_ungrib_rules(&mut c, ecmwf(), &mut ledger);
        let rh = &c.surface("RH").unwrap().values;
        let want = (2.5e6f32 / 461.5 * (1.0 / 300.0 - 1.0 / 290.0)).exp() * 1.0e2;
        assert!((rh[0] - want).abs() < 1e-4, "{} vs {want}", rh[0]);
        assert!(rh[1] > 100.0, "unclamped: {}", rh[1]);
    }

    #[test]
    fn a_file_that_already_carries_surface_humidity_keeps_it() {
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("RH", XLVL_SURFACE, vec![42.0]));
        c.insert(slab("TT", XLVL_SURFACE, vec![300.0]));
        c.insert(slab("DEWPT", XLVL_SURFACE, vec![290.0]));
        let mut ledger = RuleLedger::default();
        apply_ungrib_rules(&mut c, ecmwf(), &mut ledger);
        assert_eq!(c.surface("RH").unwrap().values, vec![42.0]);
        assert!(!decision(&ledger, "surface_rh_from_dewpoint").applied);
    }

    #[test]
    fn height_comes_out_of_geopotential_at_single_precision() {
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("GEOPT", 50_000.0, vec![50_000.0, 0.0]));
        let mut ledger = RuleLedger::default();
        apply_ungrib_rules(&mut c, ecmwf(), &mut ledger);
        // Not 50000.0/9.81 in double and rounded: the divide itself is
        // single precision, which is what rrpr.F does and what makes the
        // last bit match.
        assert_eq!(c.get("HGT", 50_000.0).unwrap().values[0], 50_000.0f32 / 9.81f32);
    }

    #[test]
    fn the_snow_rules_run_in_ungribs_order() {
        // fix_gfs_miss zeroes masked snow BEFORE the doubling, so a
        // masked ocean point stays zero rather than becoming a doubled
        // fill; and the doubling lands before the depth is derived.
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("ST000010", XLVL_SURFACE, vec![280.0, 9.999e20]));
        c.insert(slab("SNOW", XLVL_SURFACE, vec![10.0, 9.999e20]));
        let mut ledger = RuleLedger::default();
        apply_ungrib_rules(&mut c, gfs(), &mut ledger);
        assert_eq!(c.surface("SNOW").unwrap().values, vec![20.0, 0.0]);
        assert_eq!(c.surface("SNOWH").unwrap().values, vec![20.0f32 * 0.005, 0.0]);
        assert_eq!(c.surface("ST000010").unwrap().values[1], SOIL_FILL);
    }

    // -----------------------------------------------------------------
    // What reaches the file
    // -----------------------------------------------------------------

    #[test]
    fn a_vtable_row_with_a_blank_description_is_consumed_and_never_published() {
        // output.F, final pass: `if (desc.eq.' ') cycle OUTLOOP`.  This
        // is how ungrib emits 206 records where this tool used to emit
        // 247: GEOPT, DEWPT, SOILGEO, SNOW_EC and SNOW_DEN are scratch.
        let metas = vec![
            FieldMeta { name: "GEOPT".into(), units: "m2 s-2".into(), desc: "".into() },
            FieldMeta { name: "HGT".into(), units: "m".into(), desc: "Height".into() },
        ];
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("GEOPT", 50_000.0, vec![1.0]));
        c.insert(slab("HGT", 50_000.0, vec![1.0]));
        let (emissions, held) = emission_order(&c, &metas);
        assert_eq!(held, 1);
        assert_eq!(emissions.len(), 1);
        assert_eq!(emissions[0].meta.name, "HGT");
    }

    #[test]
    fn records_are_written_level_descending_then_in_vtable_order() {
        // get_plvls sorts levels largest first, so the surface sentinel
        // leads; output.F then walks the Vtable in order within a level.
        let metas = vec![
            FieldMeta { name: "TT".into(), units: "K".into(), desc: "Temperature".into() },
            FieldMeta { name: "UU".into(), units: "m s-1".into(), desc: "U".into() },
        ];
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("UU", 50_000.0, vec![1.0]));
        c.insert(slab("TT", 50_000.0, vec![1.0]));
        c.insert(slab("UU", XLVL_SURFACE, vec![1.0]));
        c.insert(slab("TT", 100_000.0, vec![1.0]));
        let (emissions, _) = emission_order(&c, &metas);
        let seen: Vec<(String, f32)> = emissions
            .iter()
            .map(|e| (e.meta.name.clone(), e.slab.xlvl))
            .collect();
        assert_eq!(
            seen,
            vec![
                ("UU".to_string(), XLVL_SURFACE),
                ("TT".to_string(), 100_000.0),
                ("TT".to_string(), 50_000.0),
                ("UU".to_string(), 50_000.0),
            ]
        );
    }

    #[test]
    fn units_and_description_come_from_the_first_vtable_row_of_that_name() {
        // output.F dedups namvar and keeps the first row's ddesc/dunits,
        // so a surface row that spells the description differently does
        // not change the header of its own records.
        let rows = "\
 129 | 100  |   *  |      | TT       | K        | Temperature                              |\n\
 167 |  1   |   0  |      | TT       | KELVIN   | Temperature at 2 m                       |\n";
        let path = std::env::temp_dir().join("met_intermediate_first_row_wins.vtable");
        std::fs::write(&path, rows).unwrap();
        let metas = field_metas(&parse_vtable(&path).unwrap());
        std::fs::remove_file(&path).ok();
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].units, "K");
        assert_eq!(metas[0].desc, "Temperature");
    }

    // -----------------------------------------------------------------
    // Headers and framing
    // -----------------------------------------------------------------

    #[test]
    fn the_grib1_earth_radius_is_a_property_of_the_centre() {
        // rd_grib1.F hardcodes it: NCEP's messages flag 6367.47 and mean
        // 6371.229.  It lands in every record header and metgrid trusts
        // it, so getting it from the wrong centre shifts every grid.
        assert_eq!(gfs().grib1_earth_radius_km, 6371.229);
        assert_eq!(ecmwf().grib1_earth_radius_km, 6367.47);
    }

    #[test]
    fn an_unknown_grib2_earth_shape_refuses_rather_than_guessing() {
        // rd_grib2.F writes `6371229. * .001`, not the literal
        // 6371.229 that rd_grib1.F writes -- and in single precision
        // those are different numbers, 6371.2295 against 6371.229.  The
        // header carries whichever the edition produced, so each path
        // reproduces its own arithmetic rather than sharing a constant.
        assert_eq!(grib2_earth_radius_km(6).unwrap(), 6371229.0f32 * 0.001);
        assert_ne!(grib2_earth_radius_km(6).unwrap(), 6371.229f32);
        assert_eq!(grib2_earth_radius_km(0).unwrap(), 6367470.0f32 * 0.001);
        assert_eq!(grib2_earth_radius_km(8).unwrap(), 6371200.0f32 * 0.001);
        assert!(grib2_earth_radius_km(1).is_err());
        assert!(grib2_earth_radius_km(200).is_err());
    }

    #[test]
    fn isobaric_levels_are_recorded_in_pascals() {
        assert_eq!(xlvl_for_grib2_level(100, 85_000.0), Some(85_000.0));
        assert_eq!(xlvl_for_grib1_level(100, 850.0), Some(85_000.0));
    }

    #[test]
    fn every_non_isobaric_surface_collapses_to_one_sentinel() {
        for lt in [1u16, 103, 106, 6, 7] {
            assert_eq!(xlvl_for_grib2_level(lt, 0.0), Some(XLVL_SURFACE));
        }
        assert_eq!(xlvl_for_grib2_level(101, 0.0), Some(XLVL_MSL));
    }

    #[test]
    fn a_north_first_scan_keeps_its_order_and_signs_the_increment() {
        let lats = vec![10.0, 10.0, 0.0, 0.0];
        let lons = vec![0.0, 1.0, 0.0, 1.0];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let (grid, out) = describe_latlon(2, 2, &lats, &lons, &values, 6371.229).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(grid.startlat, 10.0);
        assert_eq!(grid.deltalat, -10.0);
        assert_eq!(grid.deltalon, 1.0);
        assert_eq!(grid.earth_radius_km, 6371.229);
    }

    #[test]
    fn a_south_first_scan_reports_a_positive_increment() {
        let lats = vec![0.0, 0.0, 10.0, 10.0];
        let lons = vec![0.0, 1.0, 0.0, 1.0];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let (grid, out) = describe_latlon(2, 2, &lats, &lons, &values, 6367.47).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(grid.startlat, 0.0);
        assert_eq!(grid.deltalat, 10.0);
    }

    #[test]
    fn a_longitude_wrap_does_not_become_a_minus_360_increment() {
        let lats = vec![0.0, 0.0, 0.0, 0.0];
        let lons = vec![359.5, 359.75, 359.5, 359.75];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let (grid, _) = describe_latlon(2, 2, &lats, &lons, &values, 6367.47).unwrap();
        assert!((grid.deltalon - 0.25).abs() < 1e-5);
        assert!((grid.startlon - (-0.5)).abs() < 1e-4);
    }

    #[test]
    fn a_record_carries_its_length_on_both_sides() {
        let mut buffer = Vec::new();
        write_record(&mut buffer, &[1, 2, 3, 4]).unwrap();
        assert_eq!(buffer, vec![0, 0, 0, 4, 1, 2, 3, 4, 0, 0, 0, 4]);
    }

    #[test]
    fn header_fields_are_padded_to_the_widths_fortran_reads() {
        assert_eq!(fixed("TT", 9), b"TT       ".to_vec());
        assert_eq!(fixed("0123456789", 4), b"0123".to_vec());
    }

    #[test]
    fn soil_layer_bounds_are_matched_in_centimetres() {
        let mut r = VtableRow {
            g1_param: None,
            g1_level_type: None,
            g1_level1: Some(0),
            g1_level2: Some(10),
            g2_discipline: None,
            g2_category: None,
            g2_param: None,
            g2_level_type: None,
            name: "ST000010".into(),
            units: String::new(),
            desc: String::new(),
        };
        assert!(grib2_level_bounds_match(&r, 0.0, 0.1, 106));
        assert!(!grib2_level_bounds_match(&r, 0.1, 0.4, 106));
        assert!(grib2_level_bounds_match(&r, 0.0, 0.0, 1));
        r.g1_level1 = Some(2);
        assert!(grib2_level_bounds_match(&r, 2.0, 0.0, 103));
        assert!(!grib2_level_bounds_match(&r, 80.0, 0.0, 103));
    }

    #[test]
    fn a_bitmap_hole_reaches_the_file_as_the_value_ungrib_writes_there() {
        // A NaN passes every finiteness check downstream and only shows
        // up later as a poisoned average, so it must not survive here --
        // and the value that replaces it is ungrib's own -1e30, so the
        // two files agree at masked points instead of differing by 2e30.
        let mut values = vec![1.0f32, f32::NAN, 3.0, f32::INFINITY];
        let n = mark_masked(&mut values);
        assert_eq!(n, 2);
        assert!(values.iter().all(|v| v.is_finite()));
        assert_eq!(values[1], -1.0e30);
        assert!(values[1].abs() > MISSING_MAGNITUDE);
    }

    #[test]
    fn masked_snow_becomes_zero_and_masked_soil_becomes_the_soil_fill() {
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        for name in ["ST000010", "SNOW"] {
            c.insert(slab(name, XLVL_SURFACE, vec![280.0, MASKED_SENTINEL]));
        }
        let mut ledger = RuleLedger::default();
        masked_surface_fields(&mut c, gfs(), &mut ledger);
        assert_eq!(c.surface("ST000010").unwrap().values[1], SOIL_FILL);
        assert_eq!(c.surface("SNOW").unwrap().values[1], 0.0);
        assert!(decision(&ledger, "masked_surface_fields").applied);
    }
}
