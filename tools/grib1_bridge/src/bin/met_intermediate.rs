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
    /// The centre publishes physical snow depth itself (DWD's H_SNOW), so
    /// SNOWH comes from the Vtable or not at all: there is no density of
    /// the producer's own to derive one with, so none is guessed.
    PublishedOnly,
}

/// Rules that belong to one producer's encoding rather than to `rrpr.F`.
///
/// `rrpr.F` has no arm for these centres, so their conversions are this
/// tool's, written out in `apply_centre_rules` and reported in the same
/// ledger as the `rrpr` repairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CentreRules {
    /// The WPS centres: every rule they need is one of `rrpr.F`'s.
    Wps,
    /// DWD ICON: geopotential to height, half-level heights to full-level
    /// heights, column soil water to volumetric moisture and the TERRA
    /// soil-temperature nodes onto the soil layers.
    Dwd,
    /// The Met Office: its open data is NetCDF and the Met Office door
    /// converts it.  The label exists so a receipt can name the centre,
    /// and it refuses the one record whose meaning it cannot settle.
    Ukmo,
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
    /// The producer-specific conversions `apply_centre_rules` runs.
    centre_rules: CentreRules,
}

/// Every centre whose rules have been read out of WPS source, plus the
/// two (DWD and the Met Office) that `rrpr.F` has no arm for, whose rules
/// are this tool's own (`CentreRules`).
const KNOWN_MAP_SOURCES: [MapSource; 6] = [
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
        centre_rules: CentreRules::Wps,
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
        centre_rules: CentreRules::Wps,
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
        centre_rules: CentreRules::Wps,
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
        centre_rules: CentreRules::Wps,
    },
    // DWD ICON (centre 78).  rrpr.F has no DWD arm; the header is the text
    // ungrib's rd_grib2.F writes for centre 78, and it matches none of
    // rrpr.F's substring tests ("NCEP", "ECMWF"), so none of those repairs
    // runs.  ICON reports RELHUM over liquid water at every temperature,
    // publishes the land fraction (FR_LAND) rather than a flag (the init's
    // masked soil interpolation tests LANDSEA against 0, so a coastal cell at
    // 0.3 would be read as land), publishes snow water equivalent unhalved
    // (W_SNOW, kg m-2) and physical snow depth (H_SNOW, m), and integrates on
    // a sphere of 6371.229 km.  One row serves both ICON tables: the
    // regional Vtable.ICON.rw (GEOPT and SOILW, converted by the WPS rules)
    // and the global ICON tables (FI, HHL, W_SO and T_SO, converted by
    // `apply_centre_rules`); each rule engages only on the fields its table
    // names.
    MapSource {
        label: "dwd",
        header: "DWD Offenbach",
        rh_over_ice: false,
        masked_surface_repair: false,
        snow_water_equivalent_doubled: false,
        landsea_to_flag: true,
        nearest_neighbour_pressure_twins: false,
        snow_depth: SnowDepthRule::PublishedOnly,
        grib1_earth_radius_km: 6371.229,
        centre_rules: CentreRules::Dwd,
    },
    // The Met Office global deterministic model.  Its open data is NetCDF
    // and the Met Office door converts it; the label is here so receipts
    // can name the centre.  "UKMO" matches none of rrpr.F's substring
    // tests, so no rrpr repair runs.  A pressure-level RH record refuses
    // under this label (see `apply_centre_rules`).
    MapSource {
        label: "ukmo",
        header: "UKMO",
        rh_over_ice: false,
        masked_surface_repair: false,
        snow_water_equivalent_doubled: false,
        landsea_to_flag: true,
        nearest_neighbour_pressure_twins: false,
        snow_depth: SnowDepthRule::PublishedOnly,
        grib1_earth_radius_km: 6371.229,
        centre_rules: CentreRules::Ukmo,
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
    /// Level1 in tenths of the column's unit.  Integer cells carry it too
    /// (`10 * g1_level1`); a one-decimal cell carries ONLY this, and is
    /// admitted only on a single-depth type-106 row, where the column is
    /// centimetres and DWD's TERRA node at 0.5 cm needs the decimal.
    g1_level1_tenths: Option<i64>,

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

/// The Vtable "Level Type" (GRIB1 Table 3) that declares a row a SINGLE
/// depth below the land surface, as opposed to 112, a layer between two.
/// A type-106 GRIB2 row carrying it and a Level1 alone matches that one
/// depth and nothing else (`grib2_level_bounds_match`).  Rows that do
/// not carry it keep the open match they always had, so the packaged
/// tables that rely on it (Vtable.ERA5_CDO) decode as before.
const SINGLE_DEPTH_LEVEL_TYPE: u16 = 111;

/// Parse a WPS Vtable.  Columns are pipe-separated; the GRIB2 columns
/// are optional (ECMWF tables omit them entirely).
fn parse_vtable(path: &Path) -> Result<Vec<VtableRow>, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("cannot read Vtable {}: {e}", path.display()))?;
    parse_vtable_text(&text, &path.display().to_string())
}

/// The Vtable grammar itself; `label` names the table in refusals.
fn parse_vtable_text(text: &str, label: &str) -> Result<Vec<VtableRow>, String> {
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
        let bad_level = |cell: &str, which: &str| {
            format!(
                "Vtable {}: row {name}: {which} {:?} is neither blank, `*`, an integer nor \
                 a one-decimal depth; refusing rather than reading it as a wildcard",
                label,
                cell.trim()
            )
        };
        let (g1_level1, g1_level1_tenths) =
            parse_level(cells[2]).ok_or_else(|| bad_level(cells[2], "Level1"))?;
        let (g1_level2, level2_tenths) =
            parse_level(cells[3]).ok_or_else(|| bad_level(cells[3], "Level2"))?;
        // A decimal is a single-depth soil node and nothing else: anywhere
        // else every comparison is integral and a decimal would never
        // match, silently.
        if g1_level2.is_none() && level2_tenths.is_some() {
            return Err(format!(
                "Vtable {}: row {name}: Level2 {:?} is a decimal; only a single-depth \
                 type-106 row's Level1 may be one",
                label,
                cells[3].trim()
            ));
        }
        if g1_level1.is_none()
            && g1_level1_tenths.is_some()
            && (g2_level_type != Some(106)
                || g1_level_type != Some(SINGLE_DEPTH_LEVEL_TYPE)
                || level2_tenths.is_some()
                || parse_opt_u16(cells[0]).is_some())
        {
            return Err(format!(
                "Vtable {}: row {name}: Level1 {:?} is a decimal, which only a single-depth \
                 type-106 row (Level Type 111, blank Level2, no GRIB1 parameter) may carry",
                label,
                cells[2].trim()
            ));
        }
        if g1_level_type == Some(SINGLE_DEPTH_LEVEL_TYPE)
            && g2_level_type == Some(106)
            && g1_level1_tenths.is_none()
        {
            return Err(format!(
                "Vtable {label}: row {name}: a single-depth row (Level Type 111) on GRIB2 \
                 surface 106 names no depth in Level1, so it would match no message"
            ));
        }
        rows.push(VtableRow {
            g1_param: parse_opt_u16(cells[0]),
            g1_level_type,
            g1_level1,
            g1_level2,
            g1_level1_tenths,
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
        return Err(format!("Vtable {} declares no fields", label));
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
///
/// Returns `(integer, tenths)`.  An integer cell gives both; a decimal
/// with one fractional digit gives only the tenths; blank and `*` give
/// neither.  Anything else is `None`, which the caller refuses: before
/// this, an unparseable cell became a silent wildcard.
fn parse_level(cell: &str) -> Option<(Option<i64>, Option<i64>)> {
    let t = cell.trim();
    if t.is_empty() || t == "*" {
        return Some((None, None));
    }
    if let Ok(v) = t.parse::<i64>() {
        return Some((Some(v), v.checked_mul(10)));
    }
    let (whole, frac) = t.split_once('.')?;
    if frac.len() != 1
        || !frac.bytes().all(|b| b.is_ascii_digit())
        || whole.is_empty()
        || !whole.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let whole: i64 = whole.parse().ok()?;
    let tenths = whole.checked_mul(10)?.checked_add(frac.parse::<i64>().ok()?)?;
    Some((None, Some(tenths)))
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
        // Generalized vertical height coordinate (type 150): DWD ICON's
        // model levels.  WPS records a model level by its number, as
        // ungrib does for hybrid levels, so each ordinal is its own
        // `xlvl` -- far below every pressure in pascals and every
        // sentinel, so the level sort keeps them apart.  A file that
        // mixes these with isobaric levels is refused (`run`): the init
        // keys its level table on `xlvl` alone.
        150 if (1.0..1000.0).contains(&level_value) && level_value.fract() == 0.0 => {
            Some(level_value as f32)
        }
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
    /// GRIB2 vertical coordinates (100 isobaric, 150 model level) that
    /// reached the store.  Both at once is refused: their `xlvl`s are
    /// different quantities in one level table.
    vertical_coordinates: BTreeSet<u8>,
    /// Depth bounds in metres of every field selected on a below-ground
    /// layer (GRIB2 type 106 with both surfaces), by field name: the
    /// soil-water-mass rule needs the layer thickness the record carried.
    layer_bounds_m: BTreeMap<String, (f64, f64)>,
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
            vertical_coordinates: BTreeSet::new(),
            layer_bounds_m: BTreeMap::new(),
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
                && grib2_level_bounds_match(r, product.level_value, product.second_level_value, product.level_type, product.second_level_type)
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
        if matches!(product.level_type, 100 | 150) {
            out.vertical_coordinates.insert(product.level_type);
        }
        if product.level_type == 106
            && product.second_level_value.is_finite()
            && product.second_level_value > product.level_value
        {
            out.layer_bounds_m
                .insert(row.name.clone(), (product.level_value, product.second_level_value));
        }
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
fn grib2_level_bounds_match(row: &VtableRow, level1: f64, level2: f64, level_type: u8, second_level_type: u8) -> bool {
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
    // Generalized vertical height coordinate (type 150): DWD's ICON
    // model levels, numbered 1..120 from the top (full levels) and
    // 1..121 (half levels, HHL).  A Level1 in the table names one
    // level; a wildcard takes every one, each at its own `xlvl`.
    if level_type == 150 {
        return match row.g1_level1 {
            Some(want) => level1.round() as i64 == want,
            None => true,
        };
    }
    if level_type != 106 {
        return true;
    }
    // A single depth (Level Type 111, Level1 only): DWD's TERRA soil
    // temperature lives at nine nodes, each its own message with no second
    // surface.  A blank Level2 otherwise matches every depth and whichever
    // node the files stored last wins the row, silently.  A 111 row names
    // its node, compared in millimetres (0.5 cm is one of them), and a
    // depth-BOUNDED message never satisfies it.
    if row.g1_level_type == Some(SINGLE_DEPTH_LEVEL_TYPE) && row.g1_level2.is_none() {
        return match row.g1_level1_tenths {
            Some(want_tenths_cm) => {
                second_level_type == 255 && (level1 * 1000.0).round() as i64 == want_tenths_cm
            }
            // A 111 row with no depth names nothing: refusing to match is
            // the only reading that cannot pick a node by read order.
            None => false,
        };
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
        // GRIB1 isobaric levels share the GRIB2 type-100 code, and are as
        // much pascals in the level table.
        if pds.level_type == 100 {
            out.vertical_coordinates.insert(100);
        }
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

// ---------------------------------------------------------------------
// Producer rules rrpr.F has no arm for
// ---------------------------------------------------------------------

/// The 3-D fields whose levels make up the init's first-guess column.
const PROFILE_FIELD_NAMES: [&str; 7] = ["TT", "RH", "SPECHUMD", "GHT", "PRES", "UU", "VV"];

/// WMO GRIB2 identities (discipline, category, parameter) of the two
/// TERRA soil fields, on fixed-surface type 106: soil temperature
/// (2-3-18) at single depths and column-integrated soil moisture
/// (2-3-20, kg m-2) between two depths.  WMO Code Table 4.2 entries, not
/// a producer's local numbering.
const SOIL_TEMPERATURE_ID: (u16, u16, u16) = (2, 3, 18);
const SOIL_COLUMN_WATER_ID: (u16, u16, u16) = (2, 3, 20);

/// The density of liquid water, kg m-3, that turns a layer's water mass
/// per square metre into a volume fraction.
const WATER_DENSITY: f32 = 1000.0;

/// Refuse a vertical coordinate this run cannot write meaningfully.
///
/// Isobaric and model levels together: the init keys its level table on
/// `xlvl` alone, and a pressure in pascals and a model-level number are
/// different quantities in that one table.  Model levels (type 150) under
/// any centre but DWD: the ordinal convention and the HHL height rule are
/// DWD's, and no other producer's type-150 encoding has been read.
fn refuse_unwritable_vertical_coordinates(
    collector: &Collector,
    source: &MapSource,
) -> Result<(), String> {
    let coordinates = &collector.vertical_coordinates;
    if coordinates.contains(&100) && coordinates.contains(&150) {
        return Err(
            "the matched messages mix isobaric levels (type 100) with model levels (type \
             150); one intermediate file carries one vertical coordinate, so write the \
             pressure-level and model-level files separately (Vtable.ICON-GLOBAL.rw and \
             Vtable.ICON-GLOBAL-ML.rw)"
                .to_string(),
        );
    }
    if coordinates.contains(&150) && source.centre_rules != CentreRules::Dwd {
        return Err(format!(
            "messages on fixed-surface type 150 (generalized vertical height) matched under \
             --map-source {}; only DWD ICON's model levels on that surface have been read, \
             with their ordinal numbering and their HHL heights, so this tool refuses \
             rather than write another producer's type-150 levels on DWD's terms",
            source.label
        ));
    }
    Ok(())
}

/// A value the bitmap or a missing code made absent.
fn is_absent(value: f32) -> bool {
    !value.is_finite() || value.abs() > MISSING_MAGNITUDE
}

/// The DWD rules, in the order they run.
const DWD_RULE_NAMES: [&str; 5] =
    ["ght_from_fi", "ght_from_hhl", "dwd_heights_complete", "soilm_from_w_so", "soilt_from_t_so"];

/// The Met Office rules.
const UKMO_RULE_NAMES: [&str; 1] = ["ukmo_isobaric_rh_refused"];

/// Apply the conversions that belong to one producer's encoding, before
/// the `rrpr.F` rules.  These are the ones that can refuse: a soil column
/// or a height column delivered in part is an error, not a gap.  Every
/// centre reports every one of them, the WPS centres as off.
fn apply_centre_rules(
    collector: &mut Collector,
    source: &MapSource,
    vtable: &[VtableRow],
    ledger: &mut RuleLedger,
) -> Result<(), String> {
    if source.centre_rules == CentreRules::Dwd {
        ght_from_fi(collector, ledger);
        ght_from_hhl(collector, ledger)?;
        // The heights check belongs to runs on ICON's own height sources
        // (FI, HHL, model levels): a table that reads GEOPT instead
        // (Vtable.ICON.rw) gets its heights from the WPS rule
        // `hgt_from_geopt`, which runs after these.
        let icon_heights = vtable.iter().any(|row| row.name == "FI" || row.name == "HHL")
            || collector.has_field("FI")
            || collector.has_field("HHL")
            || collector.vertical_coordinates.contains(&150);
        if icon_heights {
            dwd_heights_complete(collector, ledger)?;
        } else {
            ledger.off(
                "dwd_heights_complete",
                "DWD: the Vtable names no FI or HHL; heights come from hgt_from_geopt".into(),
            );
        }
        soilm_from_w_so(collector, vtable, ledger)?;
        soilt_from_t_so(collector, vtable, ledger)?;
    } else {
        for rule in DWD_RULE_NAMES {
            ledger.off(rule, format!("{} is not DWD", source.header));
        }
    }
    if source.centre_rules == CentreRules::Ukmo {
        let aloft = collector
            .slabs
            .keys()
            .filter(|(name, lvl)| name == "RH" && lvl.0 != XLVL_SURFACE)
            .count();
        if aloft > 0 {
            return Err(format!(
                "--map-source ukmo: {aloft} RH records off the surface.  Whether the Met \
                 Office's RH aloft is over liquid water or over ice below freezing is not \
                 settled for this label, and the over-ice conversion is a difference of tens \
                 of points in cold air, so it is refused here.  Convert it in the Met Office \
                 door, where the product says which"
            ));
        }
        ledger.off(
            "ukmo_isobaric_rh_refused",
            "UKMO: no RH record off the surface reached the store".into(),
        );
    } else {
        for rule in UKMO_RULE_NAMES {
            ledger.off(rule, format!("{} is not the Met Office", source.header));
        }
    }
    Ok(())
}

/// Divide a geopotential field by `rrpr.F`'s gravity into a height field,
/// at every level that carries `from` and no `to`.  Returns (levels,
/// points).  `keep_masks` leaves a masked point masked; the ECMWF rule
/// does not, because `rrpr.F` divides whatever is there and this tool
/// writes the file `ungrib` would have written.
fn geopotential_to_height(
    collector: &mut Collector,
    from: &str,
    to: &str,
    keep_masks: bool,
) -> (usize, usize) {
    let derived: Vec<Slab> = collector
        .slabs
        .iter()
        .filter(|((n, lvl), _)| n == from && collector.get(to, lvl.0).is_none())
        .map(|((_, lvl), s)| Slab {
            name: to.to_string(),
            xlvl: lvl.0,
            grid: s.grid.clone(),
            values: s
                .values
                .iter()
                .map(|v| if keep_masks && is_absent(*v) { MASKED_SENTINEL } else { v / GRAVITY })
                .collect(),
        })
        .collect();
    let levels = derived.len();
    let points = derived.iter().map(|s| s.values.len()).sum();
    for slab in derived {
        collector.insert(slab);
    }
    (levels, points)
}

/// DWD publishes geopotential (FI, m2 s-2) on pressure levels and no
/// height.  `GHT = FI / 9.81` at every level that carries an FI and no
/// GHT, in single precision with `rrpr.F`'s gravity: the same arithmetic
/// this tool applies to ECMWF's GEOPT, so a height means the same thing
/// whichever centre it came from.  A masked point stays masked.
fn ght_from_fi(collector: &mut Collector, ledger: &mut RuleLedger) {
    let (levels, points) = geopotential_to_height(collector, "FI", "GHT", true);
    if levels == 0 {
        ledger.off("ght_from_fi", "DWD: no FI level lacks a GHT".into());
        return;
    }
    ledger.on(
        "ght_from_fi",
        "DWD publishes geopotential FI, not height".into(),
        points,
        format!("GHT = FI / 9.81 on {levels} levels"),
    );
}

/// DWD's model levels carry their heights as HHL: the geometric height
/// of the 121 half levels (interfaces), numbered from the top, so full
/// level k lies between half levels k and k+1.  `GHT(k) = (HHL(k) +
/// HHL(k+1)) / 2`, the arithmetic mean of the bounding interfaces --
/// ICON's own definition of a full level's height.
fn ght_from_hhl(collector: &mut Collector, ledger: &mut RuleLedger) -> Result<(), String> {
    let half: BTreeMap<i64, &Slab> = collector
        .slabs
        .iter()
        .filter(|((n, lvl), _)| n == "HHL" && lvl.0 < XLVL_SURFACE)
        .map(|((_, lvl), s)| (lvl.0 as i64, s))
        .collect();
    if half.is_empty() {
        ledger.off("ght_from_hhl", "DWD: no HHL half-level heights in the store".into());
        return Ok(());
    }
    let mut derived: Vec<Slab> = Vec::new();
    for (k, upper) in &half {
        let Some(lower) = half.get(&(k + 1)) else { continue };
        if collector.get("GHT", *k as f32).is_some() {
            continue;
        }
        if lower.grid != upper.grid || lower.values.len() != upper.values.len() {
            return Err(format!(
                "HHL half levels {k} and {} are on different grids; one intermediate file \
                 has one grid",
                k + 1
            ));
        }
        let values: Vec<f32> = upper
            .values
            .iter()
            .zip(lower.values.iter())
            .map(|(a, b)| {
                if is_absent(*a) || is_absent(*b) {
                    MASKED_SENTINEL
                } else {
                    0.5 * (a + b)
                }
            })
            .collect();
        derived.push(Slab { name: "GHT".into(), xlvl: *k as f32, grid: upper.grid.clone(), values });
    }
    let made = derived.len();
    let points: usize = derived.iter().map(|s| s.values.len()).sum();
    for slab in derived {
        collector.insert(slab);
    }
    if made == 0 {
        ledger.off(
            "ght_from_hhl",
            "DWD: no full level is bounded by two HHL half levels lacking a GHT".into(),
        );
        return Ok(());
    }
    ledger.on(
        "ght_from_hhl",
        "DWD publishes model-level heights as half-level HHL".into(),
        points,
        format!("GHT(k) = (HHL(k) + HHL(k+1)) / 2 on {made} full levels"),
    );
    Ok(())
}

/// Every level a DWD profile field arrived on must carry a height once
/// the two height rules have run.  A level with no height is refused,
/// naming what is missing, rather than written for the init to drop or
/// misplace -- most often a model-level lead after step 0 run without
/// the step-0 HHL as an invariant, or a pressure level whose FI object
/// did not arrive.
fn dwd_heights_complete(collector: &Collector, ledger: &mut RuleLedger) -> Result<(), String> {
    let mut missing: Vec<f32> = collector
        .slabs
        .keys()
        .filter(|(n, lvl)| {
            PROFILE_FIELD_NAMES.contains(&n.as_str())
                && n != "GHT"
                && lvl.0 != XLVL_SURFACE
                && lvl.0 != XLVL_MSL
        })
        .map(|(_, lvl)| lvl.0)
        .filter(|xlvl| collector.get("GHT", *xlvl).is_none())
        .collect();
    missing.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    missing.dedup();
    let model_levels =
        collector.vertical_coordinates.contains(&150) || collector.has_field("HHL");
    if !missing.is_empty() {
        let shown: Vec<String> = missing.iter().map(|v| format!("{v}")).collect();
        return Err(if model_levels {
            format!(
                "model levels [{}] carry profile fields but no height: full level k needs HHL \
                 half levels k and k+1, and they are not both in the inputs or the invariant \
                 files (HHL is published at step 0 only; pass the step-0 file with \
                 --invariant and --invariant-field GHT, or --invariant-field HHL)",
                shown.join(", ")
            )
        } else {
            format!(
                "levels [{}] (Pa) carry profile fields but no height: GHT comes from FI, and \
                 no FI reached the store at those levels",
                shown.join(", ")
            )
        });
    }
    let checked = collector
        .slabs
        .keys()
        .filter(|(n, lvl)| n == "GHT" && lvl.0 != XLVL_SURFACE)
        .count();
    if checked == 0 {
        ledger.off("dwd_heights_complete", "DWD: no profile level in the store".into());
    } else {
        ledger.on(
            "dwd_heights_complete",
            "DWD: every profile level must carry a height".into(),
            0,
            format!("{checked} profile levels, each with a GHT"),
        );
    }
    Ok(())
}

/// The layer bounds, in centimetres, of a derived soil row: blank GRIB2
/// parameter (nothing decodes into it), its name under `prefix`, and both
/// Level columns set.  The Vtable is where the target soil column lives.
fn derived_soil_layers<'a>(vtable: &'a [VtableRow], prefix: &str) -> Vec<(&'a VtableRow, i64, i64)> {
    let mut seen = BTreeSet::new();
    vtable
        .iter()
        .filter(|r| r.g2_param.is_none() && r.g1_param.is_none() && r.name.starts_with(prefix))
        .filter_map(|r| Some((r, r.g1_level1?, r.g1_level2?)))
        .filter(|(r, _, _)| seen.insert(r.name.clone()))
        .collect()
}

fn row_has_id(row: &VtableRow, id: (u16, u16, u16)) -> bool {
    row.g2_discipline == Some(id.0)
        && row.g2_category == Some(id.1)
        && row.g2_param == Some(id.2)
        && row.g2_level_type == Some(106)
}

/// DWD's W_SO is the water (liquid plus ice) in one TERRA layer, as a
/// mass per unit area (kg m-2).  The init reads a volume fraction, so
/// `SOILM = W_SO / (1000 kg m-3 * thickness)`, the thickness in metres
/// taken from the layer's own bounds: layer 0-1 cm divides by 10 kg m-2,
/// layer 729-2187 cm by 14580.
///
/// The target layers are the Vtable's derived SOILM rows (blank GRIB2
/// parameter, Level1/Level2 the bounds in cm), each filled from the W_SO
/// row with exactly those bounds.  A target with no W_SO layer is refused
/// when any W_SO reached the store; with none at all the rule is off (an
/// input set without soil, such as a boundary-only lead).
fn soilm_from_w_so(
    collector: &mut Collector,
    vtable: &[VtableRow],
    ledger: &mut RuleLedger,
) -> Result<(), String> {
    let targets = derived_soil_layers(vtable, "SOILM");
    if targets.is_empty() {
        ledger.off("soilm_from_w_so", "DWD: the Vtable declares no derived SOILM layer".into());
        return Ok(());
    }
    let sources: Vec<(i64, i64, &VtableRow)> = vtable
        .iter()
        .filter(|r| row_has_id(r, SOIL_COLUMN_WATER_ID))
        .filter_map(|r| Some((r.g1_level1?, r.g1_level2?, r)))
        .filter(|(_, _, r)| collector.has_surface(&r.name))
        .collect();
    if sources.is_empty() {
        ledger.off("soilm_from_w_so", "DWD: no W_SO layer reached the store".into());
        return Ok(());
    }
    let mut made = Vec::new();
    let mut points = 0usize;
    for (target, top, bottom) in &targets {
        if bottom <= top {
            return Err(format!(
                "Vtable row {}: soil layer {top}-{bottom} cm has no thickness",
                target.name
            ));
        }
        let Some((_, _, row)) = sources.iter().find(|(a, b, _)| a == top && b == bottom) else {
            return Err(format!(
                "{}: no W_SO layer {top}-{bottom} cm reached the store, but {} other W_SO \
                 layers did; a soil column delivered in part is refused rather than written \
                 with a hole",
                target.name,
                sources.len()
            ));
        };
        let src = collector.surface(&row.name).expect("filtered on presence");
        let divisor = WATER_DENSITY * ((bottom - top) as f32 / 100.0);
        let values: Vec<f32> = src
            .values
            .iter()
            .map(|w| if is_absent(*w) { MASKED_SENTINEL } else { w / divisor })
            .collect();
        let grid = src.grid.clone();
        points += values.len();
        made.push(format!("{} = {} / {divisor}", target.name, row.name));
        collector.insert(Slab { name: target.name.clone(), xlvl: XLVL_SURFACE, grid, values });
    }
    ledger.on(
        "soilm_from_w_so",
        "DWD publishes soil water as kg m-2 per TERRA layer".into(),
        points,
        format!("SOILM = W_SO / (1000 kg m-3 * layer thickness m): {}", made.join(", ")),
    );
    Ok(())
}

/// DWD's T_SO is soil temperature at single depths (nodes); the init
/// reads one temperature per layer.  Each derived SOILT layer takes the
/// temperature at its MIDPOINT depth, interpolated linearly in depth
/// between the two nodes that bracket it.
///
/// For TERRA this is exact, not an approximation: the eight layers are
/// 0-1, 1-3, 3-9, 9-27, 27-81, 81-243, 243-729 and 729-2187 cm, and the
/// nodes below the surface sit at 0.5, 2, 6, 18, 54, 162, 486 and 1458
/// cm -- each layer's midpoint.  So every layer copies one node, the 0 cm
/// (surface) node feeds none, and nine depths become eight layers whose
/// count equals SOILM's.  A target whose midpoint no two nodes bracket
/// is refused rather than extrapolated.
fn soilt_from_t_so(
    collector: &mut Collector,
    vtable: &[VtableRow],
    ledger: &mut RuleLedger,
) -> Result<(), String> {
    let targets = derived_soil_layers(vtable, "SOILT");
    if targets.is_empty() {
        ledger.off("soilt_from_t_so", "DWD: the Vtable declares no derived SOILT layer".into());
        return Ok(());
    }
    // (depth in tenths of a cm, row)
    let mut nodes: Vec<(i64, &VtableRow)> = vtable
        .iter()
        .filter(|r| row_has_id(r, SOIL_TEMPERATURE_ID) && r.g1_level2.is_none())
        .filter_map(|r| Some((r.g1_level1_tenths?, r)))
        .filter(|(_, r)| collector.has_surface(&r.name))
        .collect();
    if nodes.is_empty() {
        ledger.off("soilt_from_t_so", "DWD: no T_SO node reached the store".into());
        return Ok(());
    }
    nodes.sort_by_key(|(d, _)| *d);
    nodes.dedup_by_key(|(d, _)| *d);
    let mut made = Vec::new();
    let mut points = 0usize;
    for (target, top, bottom) in &targets {
        if bottom <= top {
            return Err(format!(
                "Vtable row {}: soil layer {top}-{bottom} cm has no thickness",
                target.name
            ));
        }
        // Midpoint in twentieths of a cm keeps every comparison integral.
        let mid = 10 * (top + bottom);
        let exact = nodes.iter().find(|(d, _)| 2 * d == mid);
        let (values, grid, how) = if let Some((_, row)) = exact {
            let src = collector.surface(&row.name).expect("filtered on presence");
            (src.values.clone(), src.grid.clone(), format!("{} = {}", target.name, row.name))
        } else {
            let above = nodes.iter().rev().find(|(d, _)| 2 * d < mid);
            let below = nodes.iter().find(|(d, _)| 2 * d > mid);
            let (Some((d0, r0)), Some((d1, r1))) = (above, below) else {
                return Err(format!(
                    "{}: layer {top}-{bottom} cm has its midpoint outside the T_SO nodes \
                     that reached the store ({} nodes, {:.1}-{:.1} cm); refusing to \
                     extrapolate a soil temperature",
                    target.name,
                    nodes.len(),
                    nodes[0].0 as f32 / 10.0,
                    nodes[nodes.len() - 1].0 as f32 / 10.0
                ));
            };
            let a = collector.surface(&r0.name).expect("filtered on presence");
            let b = collector.surface(&r1.name).expect("filtered on presence");
            if a.grid != b.grid || a.values.len() != b.values.len() {
                return Err(format!(
                    "T_SO nodes {} and {} are on different grids",
                    r0.name, r1.name
                ));
            }
            let w = (mid - 2 * d0) as f32 / (2 * (d1 - d0)) as f32;
            let values: Vec<f32> = a
                .values
                .iter()
                .zip(b.values.iter())
                .map(|(x, y)| {
                    if is_absent(*x) || is_absent(*y) {
                        MASKED_SENTINEL
                    } else {
                        x + w * (y - x)
                    }
                })
                .collect();
            (
                values,
                a.grid.clone(),
                format!("{} = {} + {w} * ({} - {})", target.name, r0.name, r1.name, r0.name),
            )
        };
        points += values.len();
        made.push(how);
        collector.insert(Slab { name: target.name.clone(), xlvl: XLVL_SURFACE, grid, values });
    }
    ledger.on(
        "soilt_from_t_so",
        "DWD publishes soil temperature at single-depth TERRA nodes".into(),
        points,
        format!("SOILT = T_SO at the layer midpoint, linear in depth: {}", made.join(", ")),
    );
    Ok(())
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
    soil_moisture_from_water_mass(collector, ledger);
    snow_from_ecmwf_water_equivalent(collector, ledger);
    landsea_fraction_to_flag(collector, source, ledger);
    snow_water_equivalent_doubling(collector, source, ledger);
    snow_depth_from_snow(collector, source, ledger);
    nearest_neighbour_pressure_twins(collector, source, ledger);
}

/// `rrpr.F`: `scr2d = scr2d / 9.81`, at every pressure level that has a
/// geopotential and no height.  Single precision, as there.
fn height_from_geopotential(collector: &mut Collector, ledger: &mut RuleLedger) {
    let (levels, points) = geopotential_to_height(collector, "GEOPT", "HGT", false);
    if levels == 0 {
        ledger.off("hgt_from_geopt", "no GEOPT level lacks an HGT".into());
        return;
    }
    ledger.on(
        "hgt_from_geopt",
        "GEOPT present without HGT (not source-gated)".into(),
        points,
        format!("HGT = GEOPT / 9.81 on {levels} levels"),
    );
}

/// Soil water published as a MASS per layer (kg m-2) becomes the
/// volumetric fraction the init reads: `SOILMxxx = SOILWxxx / (1000 kg m-3
/// x layer thickness)`, the thickness taken from the record's own type-106
/// depth pair.  A Vtable selects the mass under a `SOILW` scratch name (blank
/// description) and declares the `SOILM` output with a blank selector, the
/// same pattern as GEOPT -> HGT and SOILGEO -> SOILHGT.  Not source-gated:
/// the units are the table's declaration, not a centre's.  Masked points
/// (water, where a land-only layer has no value) stay masked.
fn soil_moisture_from_water_mass(collector: &mut Collector, ledger: &mut RuleLedger) {
    let sources: Vec<(String, f64, Vec<f32>, SlabGrid)> = collector
        .slabs
        .iter()
        .filter(|((n, lvl), _)| n.starts_with("SOILW") && lvl.0 == XLVL_SURFACE)
        .filter_map(|((n, _), s)| {
            collector
                .layer_bounds_m
                .get(n)
                .map(|(top, bottom)| (n.clone(), bottom - top, s.values.clone(), s.grid.clone()))
        })
        .collect();
    if sources.is_empty() {
        ledger.off(
            "soilm_from_soil_water_mass",
            "no SOILW layer with type-106 depth bounds was selected".into(),
        );
        return;
    }
    let mut points = 0usize;
    let mut detail = Vec::new();
    for (name, thickness, values, grid) in sources {
        let target = format!("SOILM{}", &name[5..]);
        if collector.has_surface(&target) {
            continue;
        }
        let divisor = (1000.0 * thickness) as f32;
        let derived: Vec<f32> = values
            .iter()
            .map(|v| if *v == MASKED_SENTINEL { *v } else { v / divisor })
            .collect();
        points += derived.iter().filter(|v| **v != MASKED_SENTINEL).count();
        detail.push(format!("{target} = {name} / (1000 * {thickness:.4} m)"));
        collector.insert(Slab { name: target, xlvl: XLVL_SURFACE, grid, values: derived });
    }
    ledger.on(
        "soilm_from_soil_water_mass",
        "SOILW mass-per-layer fields with depth bounds (not source-gated)".into(),
        points,
        detail.join("; "),
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
    if source.snow_depth == SnowDepthRule::PublishedOnly {
        ledger.off(
            "snowh_from_snow",
            format!(
                "{} publishes physical snow depth itself and none was decoded; no \
                 density of the producer's own exists to derive one, so SNOWH is absent",
                source.header
            ),
        );
        return;
    }
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
        SnowDepthRule::PublishedOnly => unreachable!("returned above"),
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
    // Keep the release owner's build.rs stamp in the executable (2.9.0: this
    // binary joined the bridge bundle, and a bridge the cut cannot prove from
    // its bytes must not ship), as gdt101_remap does.
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
    refuse_unwritable_vertical_coordinates(&collector, config.source)?;

    let mut ledger = RuleLedger::default();
    apply_centre_rules(&mut collector, config.source, &vtable, &mut ledger)?;
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

/// The decoded fields that carry a named invariant field.
///
/// A field with a decodable Vtable row carries itself.  A DERIVED field
/// (no GRIB parameter on any of its rows) is carried by what its centre
/// rule derives it from: for DWD, GHT comes from HHL on model levels and
/// from FI on pressure levels, whichever of the two the Vtable decodes.
/// So `--invariant-field GHT` with the step-0 file brings ICON's HHL --
/// published at step 0 only -- into a later lead, and `ght_from_hhl`
/// turns it into the full-level heights there.
fn invariant_carriers(source: &MapSource, vtable: &[VtableRow], name: &str) -> Vec<String> {
    let decodable = |field: &str| {
        vtable
            .iter()
            .any(|r| r.name == field && (r.g1_param.is_some() || r.g2_param.is_some()))
    };
    if decodable(name) {
        return vec![name.to_string()];
    }
    if source.centre_rules == CentreRules::Dwd && name == "GHT" {
        let carriers: Vec<String> =
            ["HHL", "FI"].iter().filter(|f| decodable(f)).map(|f| f.to_string()).collect();
        if !carriers.is_empty() {
            return carriers;
        }
    }
    vec![name.to_string()]
}

/// Fill the named invariant fields the main inputs lack from the
/// `--invariant` files, and say where each one came from.
///
/// An invariant file is decoded into its own store with no valid-time
/// selection (a land mask published once at hour zero is the point),
/// and only the named fields -- or, for a derived field, the fields that
/// carry it (`invariant_carriers`) -- cross over.  A named field the main
/// inputs already carry keeps the main inputs' value; one that neither
/// carries is a refusal; one on a different grid from the rest of the
/// file is a refusal, because a record header cannot say "this one is
/// elsewhere".
fn merge_invariants(
    config: &Config,
    vtable: &[VtableRow],
    collector: &mut Collector,
) -> Result<Vec<(String, &'static str)>, String> {
    let mut decided: Vec<(String, &'static str)> = Vec::new();
    if config.invariants.is_empty() {
        return Ok(decided);
    }
    let carried_by: Vec<(&String, Vec<String>)> = config
        .invariant_fields
        .iter()
        .map(|name| (name, invariant_carriers(config.source, vtable, name)))
        .collect();
    let in_inputs = |collector: &Collector, name: &str, carriers: &[String]| {
        collector.has_field(name) || carriers.iter().any(|c| collector.has_field(c))
    };
    // Decode only what is still needed, and only its rows: the invariant
    // file is a whole hour-zero product, and unpacking every matched
    // message in it to keep two fields would double each lead's decode.
    let needed: Vec<&String> = carried_by
        .iter()
        .filter(|(name, carriers)| !in_inputs(collector, name, carriers))
        .flat_map(|(_, carriers)| carriers.iter())
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
    for (name, carriers) in &carried_by {
        if in_inputs(collector, name, carriers) {
            decided.push(((*name).clone(), "inputs"));
            continue;
        }
        let keys: Vec<SlabKey> = store
            .slabs
            .keys()
            .filter(|(n, _)| carriers.contains(n))
            .cloned()
            .collect();
        if keys.is_empty() {
            return Err(format!(
                "--invariant-field {name}: neither the inputs nor the invariant files ({}) \
                 carry it under any Vtable row{}",
                json_paths(&config.invariants),
                if carriers.len() == 1 && &carriers[0] == *name {
                    String::new()
                } else {
                    format!(" (it is derived, and carried by {})", carriers.join(" or "))
                }
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
        decided.push(((*name).clone(), "invariant"));
    }
    // Only needed fields were decoded and every one of them crossed over
    // (or this returned above), so the store's coordinates are theirs.
    collector
        .vertical_coordinates
        .extend(store.vertical_coordinates.iter().copied());
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

/// The remapper's regular lat-lon encoder, for tests that need real GRIB2
/// bytes shaped exactly as `gdt101_remap` hands them to this tool.
#[cfg(test)]
#[allow(dead_code, clippy::all)]
#[path = "../gdt101_remap_core.rs"]
mod core101;

#[cfg(test)]
mod tests {
    #[test]
    fn soil_water_mass_becomes_volumetric_by_the_records_own_layer_thickness() {
        // The breakage this prevents: ICON publishes W_SO in kg m-2 per
        // TERRA layer; handed to the init as a fraction it would read 3.9
        // (a 0-1 cm layer holding 39 kg m-2 would be 3.9 m3 m-3).
        let mut c = Collector::new(0.0);
        c.insert(slab("SOILW002", XLVL_SURFACE, vec![20.0, MASKED_SENTINEL]));
        c.layer_bounds_m.insert("SOILW002".into(), (0.01, 0.03));
        let mut ledger = RuleLedger::default();
        soil_moisture_from_water_mass(&mut c, &mut ledger);
        let out = &c.surface("SOILM002").unwrap().values;
        assert!((out[0] - 1.0).abs() < 1e-6, "{out:?}");
        assert_eq!(out[1], MASKED_SENTINEL);
        assert!(ledger.entries.iter().any(|e| e.rule == "soilm_from_soil_water_mass" && e.applied));
    }

    #[test]
    fn soil_water_mass_without_depth_bounds_is_not_converted() {
        let mut c = Collector::new(0.0);
        c.insert(slab("SOILW002", XLVL_SURFACE, vec![20.0]));
        let mut ledger = RuleLedger::default();
        soil_moisture_from_water_mass(&mut c, &mut ledger);
        assert!(c.surface("SOILM002").is_none());
    }

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
        assert_eq!(resolve_map_source("DWD").unwrap().header, "DWD Offenbach");
        assert!(resolve_map_source("dwd").unwrap().landsea_to_flag);
        assert!(!resolve_map_source("dwd").unwrap().rh_over_ice);
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
            g1_level1_tenths: Some(0),
            g2_discipline: None,
            g2_category: None,
            g2_param: None,
            g2_level_type: None,
            name: "ST000010".into(),
            units: String::new(),
            desc: String::new(),
        };
        assert!(grib2_level_bounds_match(&r, 0.0, 0.1, 106, 106));
        assert!(!grib2_level_bounds_match(&r, 0.1, 0.4, 106, 106));
        assert!(grib2_level_bounds_match(&r, 0.0, 0.0, 1, 255));
        r.g1_level1 = Some(2);
        assert!(grib2_level_bounds_match(&r, 2.0, 0.0, 103, 255));
        assert!(!grib2_level_bounds_match(&r, 80.0, 0.0, 103, 255));
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

    // -----------------------------------------------------------------
    // The WPS centres write the same bytes they wrote before DWD and the
    // Met Office joined the vocabulary
    // -----------------------------------------------------------------

    const GOLDEN_GFS_VTABLE: &str = "\
     | 100  |   *  |      | HGT      | m        | Height                                   |  0  |  3  |  5  | 100 |\n\
     | 100  |   *  |      | TT       | K        | Temperature                              |  0  |  0  |  0  | 100 |\n\
     | 100  |   *  |      | RH       | %        | Relative Humidity                        |  0  |  1  |  1  | 100 |\n\
     |   1  |   0  |      | LANDSEA  | proprtn  | Land/Sea flag                            |  2  |  0  |  0  |  1  |\n\
     |   1  |   0  |      | SNOW     | kg m-2   |Water Equivalent of Accumulated Snow Depth|  0  |  1  | 13  |  1  |\n\
     |   6  |   0  |      | PMAXW    | Pa       | Pressure of max wind level               |  0  |  3  |  0  |  6  |\n\
     |   6  |   0  |      | PMAXWNN  | Pa       | PMAXW, used for nearest neighbor interp  |     |     |     |  6  |\n\
     |   1  |   0  |      | SNOWH    | m        | Physical Snow Depth                      |     |     |     |  1  |\n\
     | 112  |   0  |  10  | SM000010 | fraction | Soil Moist 0-10 cm below grn layer (Up)  |  2  |  0  | 192 | 106 |\n\
     | 112  |   0  |  10  | ST000010 | K        | T 0-10 cm below ground layer (Upper)     |  2  |  0  |  2  | 106 |\n";

    const GOLDEN_ECMWF_VTABLE: &str = "\
     | 100  |   *  |      | GEOPT    | m2 s-2   |                                          |  0  |  3  |  4  | 100 |\n\
     | 100  |   *  |      | HGT      | m        | Height                                   |  0  |  3  |  5  | 100 |\n\
     | 100  |   *  |      | TT       | K        | Temperature                              |  0  |  0  |  0  | 100 |\n\
     | 100  |   *  |      | RH       | %        | Relative Humidity                        |  0  |  1  |  1  | 100 |\n\
     | 105  |   2  |      | DEWPT    | K        |                                          |  0  |  0  |  6  | 103 |\n\
     |   1  |   0  |      | LANDSEA  | proprtn  | Land/Sea flag                            |  2  |  0  |  0  |  1  |\n\
     |   1  |   0  |      | SOILGEO  | m2 s-2   |                                          |  0  |  3  |  4  |  1  |\n\
     |   1  |   0  |      | SOILHGT  | m        | Terrain field of source analysis         |     |     |     |  1  |\n\
     |   1  |   0  |      | SNOW_EC  | m        |                                          |  0  |  1  | 254 |  1  |\n\
     |   1  |   0  |      | SNOW_DEN | kg m-3   |                                          |  0  |  1  | 61  |  1  |\n\
     |   1  |   0  |      | SNOW     | kg m-2   |Water Equivalent of Accumulated Snow Depth|     |     |     |  1  |\n\
     |   1  |   0  |      | SNOWH    | m        | Physical Snow Depth                      |     |     |     |  1  |\n";

    fn golden_metas(text: &str) -> Vec<FieldMeta> {
        field_metas(&vtable_rows(text))
    }

    /// Slabs that drive every rrpr rule each centre runs: RH over ice in
    /// the cold, the masked soil and snow repair, the soil-moisture clip,
    /// the snow doubling and depth, the max-wind twin, the land-sea flag,
    /// HGT and SOILHGT from geopotential and 2 m RH from dewpoint.
    fn golden_store(label: &str) -> Collector {
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        let cold = vec![230.0, 250.0, 265.0, 280.0];
        if label == "ncep-gfs" {
            c.insert(slab("TT", 50_000.0, cold.clone()));
            c.insert(slab("RH", 50_000.0, vec![40.0, 60.0, 80.0, 95.0]));
            c.insert(slab("HGT", 50_000.0, vec![5500.0, 5510.0, 5520.0, 5530.0]));
            c.insert(slab("TT", XLVL_SURFACE, vec![270.0, 280.0, 290.0, 300.0]));
            c.insert(slab("LANDSEA", XLVL_SURFACE, vec![0.0, 1.0, 1.0, 0.0]));
            c.insert(slab("SNOW", XLVL_SURFACE, vec![9.999e20, 3.5, 0.0, 9.999e20]));
            c.insert(slab("PMAXW", XLVL_SURFACE, vec![25_000.0, 24_000.0, 23_000.0, 22_000.0]));
            c.insert(slab("SM000010", XLVL_SURFACE, vec![9.999e20, 0.31, 0.52, 9.999e20]));
            c.insert(slab("ST000010", XLVL_SURFACE, vec![9.999e20, 281.0, 283.0, 9.999e20]));
        } else {
            c.insert(slab("TT", 50_000.0, cold));
            c.insert(slab("RH", 50_000.0, vec![40.0, 60.0, 80.0, 95.0]));
            c.insert(slab("GEOPT", 50_000.0, vec![53_955.0, 54_000.0, 54_100.0, 54_200.0]));
            c.insert(slab("TT", XLVL_SURFACE, vec![270.0, 280.0, 290.0, 300.0]));
            c.insert(slab("DEWPT", XLVL_SURFACE, vec![268.0, 275.0, 285.0, 290.0]));
            c.insert(slab("LANDSEA", XLVL_SURFACE, vec![0.0, 0.3, 0.7, 1.0]));
            c.insert(slab("SOILGEO", XLVL_SURFACE, vec![0.0, 981.0, 1962.0, 9810.0]));
            c.insert(slab("SNOW_EC", XLVL_SURFACE, vec![0.0, 0.01, 0.2, 1.5]));
            c.insert(slab("SNOW_DEN", XLVL_SURFACE, vec![100.0, 250.0, 300.0, 400.0]));
        }
        c
    }

    fn golden_bytes(label: &str, vtable_text: &str) -> Vec<u8> {
        let source = resolve_map_source(label).unwrap();
        let metas = golden_metas(vtable_text);
        let mut c = golden_store(label);
        let mut ledger = RuleLedger::default();
        centre_rules_hook(&mut c, source, vtable_text, &mut ledger);
        apply_ungrib_rules(&mut c, source, &mut ledger);
        let (emissions, _) = emission_order(&c, &metas);
        let mut out = Vec::new();
        for e in &emissions {
            write_slab(&mut out, e, "2026-10-11_00:00:00", source.header).unwrap();
        }
        out
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn the_wps_centres_write_the_bytes_they_wrote_before_dwd_and_ukmo_joined() {
        // Digests captured by running this same fixture through the
        // jacob/main build of this file (695a0a9e), before the DWD and
        // UKMO rows, the centre rules, the single-depth soil match and the
        // model-level coordinate existed.  A change here is a change to
        // what GFS, GEFS, CFSv2 or ECMWF users get.
        let gfs = golden_bytes("ncep-gfs", GOLDEN_GFS_VTABLE);
        let ecmwf = golden_bytes("ecmwf", GOLDEN_ECMWF_VTABLE);
        assert_eq!(sha256_hex(&gfs), GOLDEN_GFS_SHA256);
        assert_eq!(sha256_hex(&ecmwf), GOLDEN_ECMWF_SHA256);
    }

    const GOLDEN_GFS_SHA256: &str = "b9b3d4b7de7278d1393cb71f2af39ce06a8cfc51ba0a0bfe51dcd129f6d219e3";
    const GOLDEN_ECMWF_SHA256: &str = "7b526345221296819cccb31ebdd727e2784eacda9eb79a027b93745a7585d073";

    /// The run's own order: producer rules, then rrpr's.
    fn centre_rules_hook(c: &mut Collector, s: &MapSource, vtable_text: &str, l: &mut RuleLedger) {
        let rows = vtable_rows(vtable_text);
        apply_centre_rules(c, s, &rows, l).unwrap();
    }

    fn vtable_rows(text: &str) -> Vec<VtableRow> {
        parse_vtable_text(text, "test").unwrap()
    }

    fn vtable_error(text: &str) -> String {
        parse_vtable_text(text, "test").unwrap_err()
    }

    fn dwd() -> &'static MapSource {
        resolve_map_source("dwd").unwrap()
    }

    fn packaged_vtable(name: &str) -> Vec<VtableRow> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../woof/data/vtables").join(name);
        parse_vtable(&path).unwrap()
    }

    fn centre_rules(c: &mut Collector, source: &MapSource, rows: &[VtableRow]) -> Result<RuleLedger, String> {
        let mut ledger = RuleLedger::default();
        apply_centre_rules(c, source, rows, &mut ledger)?;
        Ok(ledger)
    }

    // -----------------------------------------------------------------
    // DWD and the Met Office in the vocabulary
    // -----------------------------------------------------------------

    #[test]
    fn dwd_and_ukmo_resolve_and_say_which_repairs_they_take() {
        assert_eq!(dwd().header, "DWD Offenbach");
        assert_eq!(resolve_map_source("UKMO").unwrap().header, "UKMO");
        // ICON reports RELHUM over water, publishes FR_LAND as a fraction
        // and W_SNOW unhalved; none of the NCEP repairs is DWD's.
        assert!(!dwd().rh_over_ice && dwd().landsea_to_flag);
        assert!(!dwd().masked_surface_repair && !dwd().snow_water_equivalent_doubled);
        assert_eq!(dwd().centre_rules, CentreRules::Dwd);
        assert_eq!(resolve_map_source("ukmo").unwrap().centre_rules, CentreRules::Ukmo);
        for source in KNOWN_MAP_SOURCES {
            // Neither header may trip an rrpr.F substring test.
            if matches!(source.centre_rules, CentreRules::Dwd | CentreRules::Ukmo) {
                assert!(!source.header.contains("NCEP") && !source.header.contains("ECMWF"));
            } else {
                assert_eq!(source.centre_rules, CentreRules::Wps);
            }
        }
        let list = known_map_source_list();
        assert!(list.contains("dwd") && list.contains("ukmo"), "{list}");
    }

    #[test]
    fn the_dwd_land_fraction_becomes_a_flag_and_its_rh_is_left_alone() {
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("LANDSEA", XLVL_SURFACE, vec![0.0, 0.5, 0.51, 1.0]));
        c.insert(slab("TT", 50_000.0, vec![230.0]));
        c.insert(slab("RH", 50_000.0, vec![80.0]));
        let mut ledger = RuleLedger::default();
        apply_ungrib_rules(&mut c, dwd(), &mut ledger);
        assert_eq!(c.surface("LANDSEA").unwrap().values, vec![0.0, 0.0, 1.0, 1.0]);
        assert_eq!(c.get("RH", 50_000.0).unwrap().values, vec![80.0]);
        assert!(!decision(&ledger, "rh_over_ice_to_liquid").applied);
    }

    #[test]
    fn dwd_snow_depth_is_published_not_derived() {
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("SNOW", XLVL_SURFACE, vec![10.0]));
        let mut ledger = RuleLedger::default();
        apply_ungrib_rules(&mut c, dwd(), &mut ledger);
        assert!(c.surface("SNOWH").is_none());
        assert!(decision(&ledger, "snowh_from_snow").basis.contains("publishes physical snow depth"));
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("SNOW", XLVL_SURFACE, vec![10.0]));
        c.insert(slab("SNOWH", XLVL_SURFACE, vec![0.04]));
        let mut ledger = RuleLedger::default();
        apply_ungrib_rules(&mut c, dwd(), &mut ledger);
        assert_eq!(c.surface("SNOWH").unwrap().values, vec![0.04]);
    }

    #[test]
    fn ukmo_refuses_relative_humidity_aloft_and_accepts_the_rest() {
        let ukmo = resolve_map_source("ukmo").unwrap();
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("RH", XLVL_SURFACE, vec![70.0]));
        c.insert(slab("PSFC", XLVL_SURFACE, vec![100_000.0]));
        let ledger = centre_rules(&mut c, ukmo, &[]).unwrap();
        assert!(!decision(&ledger, "ukmo_isobaric_rh_refused").applied);
        c.insert(slab("RH", 50_000.0, vec![70.0]));
        let err = centre_rules(&mut c, ukmo, &[]).err().unwrap();
        assert!(err.contains("--map-source ukmo") && err.contains("over ice"), "{err}");
    }

    #[test]
    fn every_centre_rule_reports_for_every_centre() {
        for source in KNOWN_MAP_SOURCES {
            let mut c = Collector::new(DEFAULT_PMIN_PA);
            let ledger = centre_rules(&mut c, &source, &[]).unwrap();
            for rule in DWD_RULE_NAMES.iter().chain(UKMO_RULE_NAMES.iter()) {
                let d = decision(&ledger, rule);
                assert!(!d.applied, "{} {rule}", source.label);
                assert!(!d.basis.is_empty());
            }
        }
    }

    // -----------------------------------------------------------------
    // Vtable levels
    // -----------------------------------------------------------------

    #[test]
    fn a_level_cell_is_an_integer_a_one_decimal_depth_a_wildcard_or_a_refusal() {
        assert_eq!(parse_level("  18 "), Some((Some(18), Some(180))));
        assert_eq!(parse_level(" 0.5 "), Some((None, Some(5))));
        assert_eq!(parse_level("1458.0"), Some((None, Some(14580))));
        assert_eq!(parse_level(" * "), Some((None, None)));
        assert_eq!(parse_level("   "), Some((None, None)));
        for bad in ["0.25", "x", ".5", "-0.5", "1e3", "5."] {
            assert_eq!(parse_level(bad), None, "{bad}");
        }
    }

    #[test]
    fn a_malformed_or_misplaced_level_is_refused_not_read_as_a_wildcard() {
        let err = vtable_error(
            "     | 100  |  abc |      | TT       | K        | Temperature   |  0  |  0  |  0  | 100 |\n",
        );
        assert!(err.contains("Level1") && err.contains("wildcard"), "{err}");
        // A decimal on an isobaric row would never match, silently.
        let err = vtable_error(
            "     | 100  |  0.5 |      | TT       | K        | Temperature   |  0  |  0  |  0  | 100 |\n",
        );
        assert!(err.contains("single-depth type-106"), "{err}");
        // ...and on a bounded soil row too.
        let err = vtable_error(
            "     | 112  |  0.5 |   1  | W        | kg m-2   |               |  2  |  3  | 20  | 106 |\n",
        );
        assert!(err.contains("single-depth"), "{err}");
        let err = vtable_error(
            "     | 112  |   0  |  1.5 | W        | kg m-2   |               |  2  |  3  | 20  | 106 |\n",
        );
        assert!(err.contains("Level2"), "{err}");
    }

    fn row_at(level1: &str, level2: &str, level_type: u16) -> VtableRow {
        let (g1_level1, g1_level1_tenths) = parse_level(level1).unwrap();
        let (g1_level2, _) = parse_level(level2).unwrap();
        VtableRow {
            g1_param: None,
            g1_level_type: Some(111),
            g1_level1,
            g1_level2,
            g1_level1_tenths,
            g2_discipline: Some(2),
            g2_category: Some(3),
            g2_param: Some(18),
            g2_level_type: Some(level_type),
            name: "T".into(),
            units: "K".into(),
            desc: String::new(),
        }
    }

    #[test]
    fn a_single_depth_row_matches_its_own_node_and_no_other() {
        // The nine TERRA nodes, in metres as the GRIB2 carries them, and in
        // centimetres as the Vtable writes them.  Before this, a blank
        // Level2 matched every depth and the node read last won the row.
        let metres = [0.0, 0.005, 0.02, 0.06, 0.18, 0.54, 1.62, 4.86, 14.58];
        let cm = ["0", "0.5", "2", "6", "18", "54", "162", "486", "1458"];
        for (i, want) in cm.iter().enumerate() {
            let row = row_at(want, "", 106);
            for (j, depth) in metres.iter().enumerate() {
                assert_eq!(
                    grib2_level_bounds_match(&row, *depth, 0.0, 106, 255),
                    i == j,
                    "row {want} cm against node {depth} m"
                );
            }
            // A depth-bounded message never satisfies a single-depth row.
            assert!(!grib2_level_bounds_match(&row, metres[i], 0.03, 106, 106));
        }
        // The bounded rule keeps its centimetre arithmetic.
        let bounded = row_at("1", "3", 106);
        assert!(grib2_level_bounds_match(&bounded, 0.01, 0.03, 106, 106));
        assert!(!grib2_level_bounds_match(&bounded, 0.0, 0.01, 106, 106));
        // Both blank keeps the old open match, for the tables that rely on it.
        let mut open = row_at("", "", 106);
        open.g1_level_type = Some(112);
        assert!(grib2_level_bounds_match(&open, 0.18, 0.0, 106, 255));
    }

    #[test]
    fn model_levels_are_written_at_their_number() {
        assert_eq!(xlvl_for_grib2_level(150, 120.0), Some(120.0));
        assert_eq!(xlvl_for_grib2_level(150, 1.0), Some(1.0));
        assert_eq!(xlvl_for_grib2_level(150, 0.0), None);
        assert_eq!(xlvl_for_grib2_level(150, 12.5), None);
        let any = row_at("*", "", 150);
        assert!(grib2_level_bounds_match(&any, 37.0, 38.0, 150, 150));
        let one = row_at("120", "", 150);
        assert!(grib2_level_bounds_match(&one, 120.0, 121.0, 150, 150));
        assert!(!grib2_level_bounds_match(&one, 119.0, 120.0, 150, 150));
    }

    #[test]
    fn a_store_mixing_isobaric_and_model_levels_is_refused() {
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.vertical_coordinates.insert(100);
        assert!(refuse_unwritable_vertical_coordinates(&c, dwd()).is_ok());
        assert!(refuse_unwritable_vertical_coordinates(&c, gfs()).is_ok());
        c.vertical_coordinates.insert(150);
        let err = refuse_unwritable_vertical_coordinates(&c, dwd()).unwrap_err();
        assert!(err.contains("one vertical coordinate"), "{err}");
    }

    #[test]
    fn model_levels_are_dwd_s_alone() {
        // Type 150 used to be skipped for every centre.  It is admitted
        // now for DWD's ICON levels, and refused, not reinterpreted, for a
        // centre whose type-150 encoding has not been read.
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.vertical_coordinates.insert(150);
        assert!(refuse_unwritable_vertical_coordinates(&c, dwd()).is_ok());
        for label in ["ecmwf", "ncep-gfs", "ukmo"] {
            let err = refuse_unwritable_vertical_coordinates(&c, resolve_map_source(label).unwrap())
                .unwrap_err();
            assert!(err.contains("type 150") && err.contains(label), "{err}");
        }
    }

    #[test]
    fn a_single_depth_row_is_declared_by_level_type_111_and_others_keep_the_open_match() {
        // Vtable.ERA5_CDO's soil rows are Level Type 1 with Level1 0 and a
        // blank Level2 on GRIB2 surface 106: an open match it relies on.
        let mut era5 = row_at("0", "", 106);
        era5.g1_level_type = Some(1);
        assert!(grib2_level_bounds_match(&era5, 0.0, 0.07, 106, 106));
        assert!(grib2_level_bounds_match(&era5, 0.28, 1.0, 106, 106));
        // The packaged table itself still parses and keeps those rows.
        let rows = packaged_vtable("Vtable.ERA5_CDO");
        assert!(rows.iter().any(|r| r.name == "ST000007" && r.g1_level_type == Some(1)));
        // A 111 row with no depth on surface 106 would match nothing.
        let err = vtable_error(
            "     | 111  |   *  |      | T        | K        |               |  2  |  3  | 18  | 106 |\n",
        );
        assert!(err.contains("names no depth"), "{err}");
        // A decimal is a 111 row's privilege.
        let err = vtable_error(
            "     |   1  |  0.5 |      | T        | K        |               |  2  |  3  | 18  | 106 |\n",
        );
        assert!(err.contains("Level Type 111"), "{err}");
    }

    #[test]
    fn a_model_level_run_without_any_hhl_is_refused() {
        // The common slip: a lead after step 0 run without the step-0 HHL.
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.vertical_coordinates.insert(150);
        c.insert(slab("TT", 120.0, vec![280.0]));
        c.insert(slab("PRES", 120.0, vec![99_000.0]));
        c.insert(slab("TT", XLVL_SURFACE, vec![282.0]));
        let err = centre_rules(&mut c, dwd(), &[]).err().unwrap();
        assert!(err.contains("model levels [120]") && err.contains("--invariant-field HHL"), "{err}");
    }

    #[test]
    fn a_pressure_level_without_fi_is_refused() {
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.vertical_coordinates.insert(100);
        c.insert(slab("FI", 50_000.0, vec![54_936.0]));
        c.insert(slab("TT", 50_000.0, vec![250.0]));
        c.insert(slab("TT", 85_000.0, vec![270.0]));
        let err = centre_rules(&mut c, dwd(), &[]).err().unwrap();
        assert!(err.contains("[85000]") && err.contains("FI"), "{err}");
        // A surface-only store has no profile level to check.
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("TT", XLVL_SURFACE, vec![282.0]));
        c.insert(slab("PMSL", XLVL_MSL, vec![101_000.0]));
        let ledger = centre_rules(&mut c, dwd(), &[]).unwrap();
        assert!(!decision(&ledger, "dwd_heights_complete").applied);
    }

    // -----------------------------------------------------------------
    // GHT from the step-0 HHL, through --invariant-field GHT, in real
    // GRIB2 bytes shaped as gdt101_remap writes them
    // -----------------------------------------------------------------

    /// One ICON-like message on a 2x2 regular lat-lon window, encoded by
    /// the remapper's own writer from a GDT-101 template that carries the
    /// product section (centre 78, 2026-09-15 00 UTC, `lead_hours`).
    #[allow(clippy::too_many_arguments)]
    fn icon_record(
        category: u8,
        parameter: u8,
        level_type: u8,
        level: u32,
        second_type: u8,
        second: u32,
        lead_hours: u32,
        values: [f64; 4],
    ) -> Vec<u8> {
        fn section(number: u8, length: usize) -> Vec<u8> {
            let mut bytes = vec![0; length];
            bytes[..4].copy_from_slice(&(length as u32).to_be_bytes());
            bytes[4] = number;
            bytes
        }
        let mut s1 = section(1, 21);
        s1[5..7].copy_from_slice(&78u16.to_be_bytes());
        s1[9] = 30;
        s1[12..14].copy_from_slice(&2026u16.to_be_bytes());
        s1[14] = 9;
        s1[15] = 15;
        s1[20] = 1;
        let mut s3 = section(3, 35);
        s3[6..10].copy_from_slice(&4u32.to_be_bytes());
        s3[12..14].copy_from_slice(&101u16.to_be_bytes());
        let mut s4 = section(4, 34);
        s4[9] = category;
        s4[10] = parameter;
        s4[11] = 2;
        s4[17] = 1;
        s4[18..22].copy_from_slice(&lead_hours.to_be_bytes());
        s4[22] = level_type;
        s4[24..28].copy_from_slice(&level.to_be_bytes());
        s4[28] = second_type;
        s4[30..34].copy_from_slice(&second.to_be_bytes());
        let mut s5 = section(5, 21);
        s5[5..9].copy_from_slice(&4u32.to_be_bytes());
        let mut s6 = section(6, 6);
        s6[5] = 255;
        let mut raw = vec![0; 16];
        raw[..4].copy_from_slice(b"GRIB");
        raw[7] = 2;
        for s in [s1, s3, s4, s5, s6, section(7, 5)] {
            raw.extend(s);
        }
        raw.extend(b"7777");
        let n = raw.len() as u64;
        raw[8..16].copy_from_slice(&n.to_be_bytes());
        let target = core101::Target { west: 10.0, south: 50.0, dx: 0.125, dy: 0.125, nx: 2, ny: 2 };
        core101::encode_regular(&raw, target, &values).unwrap()
    }

    #[test]
    fn ght_comes_from_the_step_0_hhl_when_the_door_names_ght_as_an_invariant() {
        let dir = std::env::temp_dir().join(format!(
            "met_intermediate_hhl_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // The lead (f003): T and P on full levels 119 and 120 (150/150),
        // the surface fields, and no HHL -- DWD publishes HHL at step 0 only.
        let mut lead = Vec::new();
        for k in [119u32, 120] {
            lead.extend(icon_record(0, 0, 150, k, 150, k + 1, 3, [280.0; 4]));
            lead.extend(icon_record(3, 0, 150, k, 150, k + 1, 3, [95_000.0; 4]));
        }
        lead.extend(icon_record(0, 0, 103, 2, 255, 0, 3, [281.0; 4]));
        // The step-0 file the door names with --invariant: HHL 119..121
        // (150 with a second surface of 101, as the real objects carry it),
        // plus FR_LAND and HSURF.
        let mut step0 = Vec::new();
        for (k, h) in [(119u32, 300.0), (120, 100.0), (121, 20.0)] {
            step0.extend(icon_record(3, 6, 150, k, 101, 0, 0, [h, h + 10.0, h + 20.0, h + 30.0]));
        }
        let lead_path = dir.join("f003.grib2");
        let step0_path = dir.join("f000.grib2");
        std::fs::write(&lead_path, &lead).unwrap();
        std::fs::write(&step0_path, &step0).unwrap();
        let vtable = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../woof/data/vtables/Vtable.ICON-GLOBAL-ML.rw");
        let Invocation::Run(config) = parse_args(args(&[
            "--vtable", vtable.to_str().unwrap(), "--date", "2026-09-15_03:00:00", "--out",
            dir.join("out").to_str().unwrap(), "--map-source", "dwd", "--select-valid-time",
            "--invariant", step0_path.to_str().unwrap(), "--invariant-field", "GHT",
            lead_path.to_str().unwrap(),
        ]))
        .unwrap() else {
            panic!("help")
        };
        let rows = parse_vtable(&config.vtable_path).unwrap();
        let mut c = Collector::new(config.pmin_pa);
        c.select_valid_time = Some(config.valid_time);
        collect_file(&lead_path, &rows, config.source, &mut c).unwrap();
        assert!(!c.has_field("HHL"));
        let decided = merge_invariants(&config, &rows, &mut c).unwrap();
        assert_eq!(decided, vec![("GHT".to_string(), "invariant")]);
        refuse_unwritable_vertical_coordinates(&c, config.source).unwrap();
        let ledger = centre_rules(&mut c, config.source, &rows).unwrap();
        assert!(decision(&ledger, "ght_from_hhl").applied);
        let ght119 = &c.get("GHT", 119.0).unwrap().values;
        let ght120 = &c.get("GHT", 120.0).unwrap().values;
        // Means of the bounding half levels, point by point.
        assert_eq!(ght119.len(), 4);
        for i in 0..4 {
            let off = 10.0 * i as f32;
            assert_eq!(ght119[i], 0.5 * ((300.0 + off) + (100.0 + off)));
            assert_eq!(ght120[i], 0.5 * ((100.0 + off) + (20.0 + off)));
        }
        // Without the invariant, the same lead is refused by name.
        let mut bare = Collector::new(config.pmin_pa);
        bare.select_valid_time = Some(config.valid_time);
        collect_file(&lead_path, &rows, config.source, &mut bare).unwrap();
        let err = centre_rules(&mut bare, config.source, &rows).err().unwrap();
        assert!(err.contains("--invariant-field HHL") || err.contains("HHL"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_decodable_invariant_carries_itself_and_dwd_ght_is_carried_by_its_sources() {
        let ml = packaged_vtable("Vtable.ICON-GLOBAL-ML.rw");
        let pl = packaged_vtable("Vtable.ICON-GLOBAL.rw");
        assert_eq!(invariant_carriers(dwd(), &ml, "LANDSEA"), vec!["LANDSEA"]);
        assert_eq!(invariant_carriers(dwd(), &ml, "SOILHGT"), vec!["SOILHGT"]);
        assert_eq!(invariant_carriers(dwd(), &ml, "GHT"), vec!["HHL"]);
        assert_eq!(invariant_carriers(dwd(), &pl, "GHT"), vec!["FI"]);
        // Only DWD derives GHT so; any other centre names the field itself.
        assert_eq!(invariant_carriers(ecmwf(), &ml, "GHT"), vec!["GHT"]);
    }

    // -----------------------------------------------------------------
    // The DWD rules
    // -----------------------------------------------------------------

    #[test]
    fn dwd_geopotential_becomes_ght_with_rrprs_gravity_and_masks_survive() {
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("FI", 50_000.0, vec![54_936.0, MASKED_SENTINEL]));
        c.insert(slab("FI", 85_000.0, vec![14_715.0, 14_700.0]));
        let ledger = centre_rules(&mut c, dwd(), &[]).unwrap();
        assert_eq!(c.get("GHT", 50_000.0).unwrap().values, vec![54_936.0f32 / 9.81, MASKED_SENTINEL]);
        assert_eq!(c.get("GHT", 85_000.0).unwrap().values[0], 14_715.0f32 / 9.81);
        assert!(decision(&ledger, "ght_from_fi").applied);
        // No HGT: the init reads GHT, and the ECMWF rule stays off.
        let mut ledger = RuleLedger::default();
        apply_ungrib_rules(&mut c, dwd(), &mut ledger);
        assert!(c.get("HGT", 50_000.0).is_none());
    }

    #[test]
    fn model_level_height_is_the_mean_of_its_two_half_levels() {
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("HHL", 119.0, vec![120.0, 300.0]));
        c.insert(slab("HHL", 120.0, vec![40.0, 220.0]));
        c.insert(slab("HHL", 121.0, vec![0.0, MASKED_SENTINEL]));
        c.insert(slab("TT", 119.0, vec![280.0, 280.0]));
        c.insert(slab("TT", 120.0, vec![281.0, 281.0]));
        c.insert(slab("TT", XLVL_SURFACE, vec![282.0, 282.0]));
        let ledger = centre_rules(&mut c, dwd(), &[]).unwrap();
        assert_eq!(c.get("GHT", 119.0).unwrap().values, vec![80.0, 260.0]);
        assert_eq!(c.get("GHT", 120.0).unwrap().values, vec![20.0, MASKED_SENTINEL]);
        // Half level 121 is the bottom interface, not a full level.
        assert!(c.get("GHT", 121.0).is_none());
        assert!(decision(&ledger, "ght_from_hhl").applied);
    }

    #[test]
    fn a_model_level_without_both_half_levels_is_refused() {
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("HHL", 119.0, vec![120.0]));
        c.insert(slab("HHL", 120.0, vec![40.0]));
        c.insert(slab("TT", 119.0, vec![280.0]));
        c.insert(slab("PRES", 120.0, vec![99_000.0]));
        let err = centre_rules(&mut c, dwd(), &[]).err().unwrap();
        assert!(err.contains("[120]") && err.contains("HHL half levels k and k+1"), "{err}");
    }

    /// A store holding one value per TERRA node and layer, decoded the way
    /// the packaged pressure-level Vtable names them.
    fn terra_store(rows: &[VtableRow]) -> Collector {
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        for row in rows {
            if row_has_id(row, SOIL_TEMPERATURE_ID) {
                // 280 K at the surface warming 1 K per node, masked at the
                // second point (water).
                let node = row.name.trim_start_matches("T_SO_N").parse::<f32>().unwrap();
                c.insert(slab(&row.name, XLVL_SURFACE, vec![280.0 + node, MASKED_SENTINEL]));
            } else if row_has_id(row, SOIL_COLUMN_WATER_ID) {
                // 0.25 m3 m-3 in every layer, as kg m-2 per that layer.
                let thickness_m = (row.g1_level2.unwrap() - row.g1_level1.unwrap()) as f32 / 100.0;
                c.insert(slab(&row.name, XLVL_SURFACE, vec![250.0 * thickness_m, MASKED_SENTINEL]));
            }
        }
        c
    }

    #[test]
    fn the_packaged_icon_tables_carry_eight_matching_soil_layers() {
        for name in ["Vtable.ICON-GLOBAL.rw", "Vtable.ICON-GLOBAL-ML.rw"] {
            let rows = packaged_vtable(name);
            let nodes: Vec<&VtableRow> =
                rows.iter().filter(|r| row_has_id(r, SOIL_TEMPERATURE_ID)).collect();
            let layers: Vec<&VtableRow> =
                rows.iter().filter(|r| row_has_id(r, SOIL_COLUMN_WATER_ID)).collect();
            assert_eq!(nodes.len(), 9, "{name}");
            assert_eq!(layers.len(), 8, "{name}");
            // Every soil source row names its depth(s): none is the open
            // match that lets read order pick a node.
            assert!(nodes.iter().all(|r| r.g1_level1_tenths.is_some() && r.g1_level2.is_none()));
            assert!(layers.iter().all(|r| r.g1_level1.is_some() && r.g1_level2.is_some()));
            assert!(rows
                .iter()
                .filter(|r| r.g2_level_type == Some(106) && r.g2_param.is_some())
                .all(|r| r.g1_level1_tenths.is_some()));
            // The derived layers carry the names and thicknesses
            // rw_mpas_init dispatches.
            for prefix in ["SOILT", "SOILM"] {
                let targets = derived_soil_layers(&rows, prefix);
                let got: Vec<(String, i64)> =
                    targets.iter().map(|(r, a, b)| (r.name.clone(), b - a)).collect();
                let want: Vec<(String, i64)> = [
                    ("001", 1), ("002", 2), ("006", 6), ("018", 18), ("054", 54), ("162", 162),
                    ("486", 486), ("999", 1458),
                ]
                .iter()
                .map(|(s, t)| (format!("{prefix}{s}"), *t))
                .collect();
                assert_eq!(got, want, "{name} {prefix}");
                assert!(targets.iter().all(|(r, _, _)| !r.desc.is_empty()));
            }
            // Raw soil and the scratch fields are consumed, never published.
            for row in &rows {
                let scratch = row.name.starts_with("T_SO_")
                    || row.name.starts_with("W_SO_")
                    || ["FI", "HHL", "DEWPT"].contains(&row.name.as_str());
                assert_eq!(row.desc.is_empty(), scratch, "{name} {}", row.name);
            }
        }
    }

    #[test]
    fn terra_soil_becomes_eight_volumetric_moisture_and_temperature_layers() {
        let rows = packaged_vtable("Vtable.ICON-GLOBAL.rw");
        let mut c = terra_store(&rows);
        let ledger = centre_rules(&mut c, dwd(), &rows).unwrap();
        let suffixes = ["001", "002", "006", "018", "054", "162", "486", "999"];
        for (k, sfx) in suffixes.iter().enumerate() {
            let sm = &c.surface(&format!("SOILM{sfx}")).unwrap().values;
            assert!((sm[0] - 0.25).abs() < 1e-6, "SOILM{sfx} {}", sm[0]);
            assert_eq!(sm[1], MASKED_SENTINEL);
            // Layer k's midpoint is node k+1 exactly: a copy, not a blend.
            let st = &c.surface(&format!("SOILT{sfx}")).unwrap().values;
            assert_eq!(st[0], 280.0 + (k + 1) as f32, "SOILT{sfx}");
            assert_eq!(st[1], MASKED_SENTINEL);
        }
        assert!(decision(&ledger, "soilm_from_w_so").detail.contains("SOILM999 = W_SO_L8 / 14580"));
        assert!(decision(&ledger, "soilt_from_t_so").detail.contains("SOILT001 = T_SO_N1,"));
        // And what reaches the file: sixteen soil records, none raw.
        let metas = field_metas(&rows);
        let (emissions, _) = emission_order(&c, &metas);
        let names: Vec<&str> = emissions.iter().map(|e| e.meta.name.as_str()).collect();
        assert_eq!(names.iter().filter(|n| n.starts_with("SOILT")).count(), 8);
        assert_eq!(names.iter().filter(|n| n.starts_with("SOILM")).count(), 8);
        assert!(!names.iter().any(|n| n.starts_with("T_SO") || n.starts_with("W_SO")));
    }

    #[test]
    fn the_model_level_table_runs_the_same_soil_rules() {
        let rows = packaged_vtable("Vtable.ICON-GLOBAL-ML.rw");
        let mut c = terra_store(&rows);
        centre_rules(&mut c, dwd(), &rows).unwrap();
        assert!(c.surface("SOILM054").is_some() && c.surface("SOILT999").is_some());
    }

    #[test]
    fn a_soil_temperature_layer_between_nodes_is_interpolated_at_its_midpoint() {
        // A target layer 0-4 cm has its midpoint at 2 cm (node), 4-8 cm at
        // 6 cm (node), and 2-12 cm at 7 cm: between the 6 and 18 cm nodes.
        let text = "\
     | 111  |   6  |      | N6       | K        |                  |  2  |  3  | 18  | 106 |\n\
     | 111  |  18  |      | N18      | K        |                  |  2  |  3  | 18  | 106 |\n\
     | 112  |   2  |  12  | SOILT010 | K        | T of 2-12 cm     |     |     |     | 106 |\n";
        let rows = vtable_rows(text);
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("N6", XLVL_SURFACE, vec![280.0, 290.0]));
        c.insert(slab("N18", XLVL_SURFACE, vec![292.0, MASKED_SENTINEL]));
        centre_rules(&mut c, dwd(), &rows).unwrap();
        let v = &c.surface("SOILT010").unwrap().values;
        assert!((v[0] - 281.0).abs() < 1e-4, "{}", v[0]);
        assert_eq!(v[1], MASKED_SENTINEL);
    }

    #[test]
    fn a_soil_column_delivered_in_part_or_beyond_the_nodes_is_refused() {
        let rows = packaged_vtable("Vtable.ICON-GLOBAL.rw");
        // One W_SO layer missing.
        let mut c = terra_store(&rows);
        c.slabs.remove(&("W_SO_L5".to_string(), OrderedF32(XLVL_SURFACE)));
        let err = centre_rules(&mut c, dwd(), &rows).err().unwrap();
        assert!(err.contains("SOILM054") && err.contains("27-81 cm"), "{err}");
        // The deepest node missing: layer 729-2187 cm would extrapolate.
        let mut c = terra_store(&rows);
        c.slabs.remove(&("T_SO_N8".to_string(), OrderedF32(XLVL_SURFACE)));
        let err = centre_rules(&mut c, dwd(), &rows).err().unwrap();
        assert!(err.contains("SOILT999") && err.contains("extrapolate"), "{err}");
        // No soil at all is a gap the rules report, not an error.
        let mut c = Collector::new(DEFAULT_PMIN_PA);
        c.insert(slab("TT", XLVL_SURFACE, vec![250.0]));
        let ledger = centre_rules(&mut c, dwd(), &rows).unwrap();
        assert!(!decision(&ledger, "soilm_from_w_so").applied);
        assert!(!decision(&ledger, "soilt_from_t_so").applied);
    }
}
