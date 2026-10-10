//! FIELD-DRIVEN RESOLUTION: a spacing raster as a refinement region.
//!
//! The analytic regions in [`crate::mesh::density`] describe a refinement by
//! a SHAPE and one spacing. That covers a cap, a box and a corridor polygon,
//! and it cannot describe "0.1 km within 2 km of a power line, graded out
//! over the terrain": the spacing there is a FIELD, computed by somebody who
//! knows where the line, the assets and the steep ground are. This module is
//! the door for that field.
//!
//! # The format, `woof-hex.density.v1`
//!
//! CF netCDF on a regular latitude/longitude grid:
//!
//! * 1-D `lat` and `lon` coordinates in degrees, strictly ascending. They need
//!   not be evenly spaced, so a raster can be fine near a line and coarse far
//!   from it without paying for the fine pitch everywhere;
//! * `spacing_km`, float64, dimensions `(lat, lon)`: the requested hexagon
//!   across-flats spacing at each node;
//! * global attributes `schema = "woof-hex.density.v1"` and `min_spacing_km`,
//!   which must agree with the finest value actually stored.
//!
//! A wrong schema, a NaN, a non-positive spacing, a declared minimum the data
//! does not carry, a descending axis or a raster that reaches past 85 degrees
//! of latitude is REFUSED by name. Nothing is guessed.
//!
//! # How the raster joins the field
//!
//! Inside its extent the raster's spacing `h_r(x)` is BILINEAR in latitude
//! and longitude between the four surrounding nodes. Outside it the raster
//! says nothing and the rest of the spec (analytic regions, other rasters,
//! the background) is the field. Where the raster and anything else overlap
//! the FINER one wins -- the rule every region already obeys -- so the
//! combined field is `min(h_background, h_regions, h_rasters)`.
//!
//! # What is done to the raster before it is used, in order
//!
//! 1. **Clamp at the background.** A node coarser than the background cannot
//!    win the `min` anywhere, so it is clamped to the background; this only
//!    makes the raster's own grading gentler where it was masked anyway.
//! 2. **Ladder snap.** The hierarchical generator refines by midpoint
//!    insertion, so a refined plateau can only sit at `background / 2^k`
//!    ([`crate::mesh::ladder_snap`] measured what an off-rung request silently
//!    delivered). The raster's finest value is snapped the same way a region
//!    spacing is -- finer, never coarser, to
//!    `rung = background / 2^ceil(log2(background / finest))` -- by the
//!    AFFINE map `h -> rung + (h - finest) (background - rung) /
//!    (background - finest)`. That map sends the finest node exactly to the
//!    rung, leaves the background exactly where it was (so the raster still
//!    meets the field outside it), never coarsens a node, and steepens the
//!    raster's slope by the factor `(background - rung) / (background -
//!    finest)`, which is one part in a few thousand for any real request.
//!    The limiter below runs AFTER the snap, so that steepening cannot carry
//!    the field over the gradient ceiling.
//! 3. **Gradient limiting** (default on). The largest field `h' <= h` whose
//!    slope is at most `g` per cell, found as a multi-source shortest-path
//!    problem: `h'(u) = min_v ( h(v) + d(u, v) )` over the node graph, with
//!    each edge weighted by `g / sqrt(2)` times its length on the sphere.
//!    Solved exactly by Dijkstra's algorithm with every node a source. It
//!    only ever LOWERS a node (finer, never coarser) and never lowers the
//!    finest node, so the snapped plateau survives it.
//!
//!    WHY `g / sqrt(2)`, and why that is a GUARANTEE rather than a hope.
//!    Inside one raster cell the bilinear field's east derivative is the
//!    latitude-interpolated east difference over the cell's east width, and
//!    its north derivative the longitude-interpolated north difference over
//!    its north height. Holding every east edge to `a` times the cell's
//!    narrowest east width (its POLEWARD side, so the cosine cannot loosen
//!    it anywhere inside) and every north edge to `a` times its height bounds
//!    both components by `a` everywhere in the cell, so the magnitude is at
//!    most `a sqrt(2)`. With `a = g / sqrt(2)` the slope is at most `g`. The
//!    price is conservatism on a slope aligned with an axis, which is
//!    graded `sqrt(2)` more gently than it would need to be.
//!
//!    And why a slope bound IS the generator's gate: the gradient meter
//!    ([`crate::mesh::density::steepest_gradient_reading_of`]) reads
//!    `|h(q) / h(p) - 1|` with `q` one local spacing `h(p)` from `p`. For a
//!    field of slope at most `g`, `|h(q) - h(p)| <= g * arc(p, q) <= g h(p)`,
//!    so the meter can never read more than `g`. The transition-band gate's
//!    ceiling is `2^(1/6) - 1`; the default target sits five percent below
//!    it, and `woof mesh` passes its own (stricter) smoothness bound instead.
//!
//!    With `"limit": false` nothing is lowered; a raster steeper than its
//!    ceiling is then REFUSED, with the cell where it is steepest named.
//!
//! # The certificate
//!
//! After limiting, every cell's slope is bounded in closed form (the
//! componentwise bound above, evaluated on the stored nodes) and the largest
//! is the raster's `certified_peak_per_cell`. It is folded into the
//! gradient reading by `max`, so the reading a gate sees can only rise and
//! a raster is judged by an upper bound on its slope rather than by wherever
//! a probe happened to land. The raster's EDGE is part of it too: where the
//! raster's outermost node is finer than the field just outside it, the
//! field steps there, and that step is folded in the same way (a step larger
//! than the ceiling is refused, with the remedy of extending the raster).

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{MpasError, MpasResult};
use crate::mesh::geom::{EARTH_RADIUS_M, V3, arc, dot, from_lat_lon, lat_lon};

/// The schema string a density raster carries as its `schema` attribute.
pub const RASTER_SCHEMA: &str = "woof-hex.density.v1";

/// The transition-band gate's ceiling on the per-cell spacing change:
/// `generate_graded` refuses a band narrower than twice the 3-cell surgery
/// locality, `ln 2 / ln(1 + g) >= 6`, so `g <= 2^(1/6) - 1`.
pub const BAND_CEILING_PER_CELL: f64 = 0.122_462_048_309_373;

/// The default limiter target when a raster row names none: five percent
/// inside the transition-band ceiling, so floating-point rounding in the
/// meter cannot carry a limited raster over the gate.
pub const DEFAULT_MAX_GRADIENT_PER_CELL: f64 = 0.95 * BAND_CEILING_PER_CELL;

/// Rasters may not reach past this latitude. A latitude/longitude grid's
/// east metric vanishes at the pole, where a bilinear cell is not a cell.
pub const MAX_ABS_LAT_DEG: f64 = 85.0;

/// Relative disagreement allowed between `min_spacing_km` and the data.
const DECLARED_MIN_TOLERANCE: f64 = 1e-6;

/// The limiter's allowance is shaved by this fraction so the float sum
/// `h(v) + w` cannot land a last-bit above the ceiling it is meant to hold.
const LIMITER_MARGIN: f64 = 1e-9;

/// The most probe nodes one raster contributes to the gradient meter. The
/// raster's slope is CERTIFIED in closed form, so these probes are not what
/// makes the reading complete; they are what lets the meter see the
/// refinement at all, and the decimation keeps a long thin raster far inside
/// [`crate::mesh::density::PROBE_BUDGET`].
const RASTER_PROBE_CAP: usize = 200_000;

/// Sub-samples per raster cell side, at most, in the sizing quadrature.
const QUADRATURE_MAX_SPLIT: usize = 16;

/// A density raster as read from its file, before anything is done to it.
#[derive(Debug)]
pub struct RasterSource {
    pub path: PathBuf,
    pub sha256: String,
    pub lat_deg: Vec<f64>,
    pub lon_deg: Vec<f64>,
    /// `spacing_km[i * nlon + j]` at `(lat_deg[i], lon_deg[j])`.
    pub spacing_km: Vec<f64>,
    pub declared_min_spacing_km: f64,
    pub finest_km: f64,
    pub finest_at_deg: [f64; 2],
}

fn refusal(path: &Path, what: String) -> MpasError {
    MpasError::Refusal(format!("density raster {}: {what}", path.display()))
}

impl RasterSource {
    /// Read and check a `woof-hex.density.v1` raster.
    pub fn read(path: &Path) -> MpasResult<RasterSource> {
        let bytes = std::fs::read(path)
            .map_err(|e| refusal(path, format!("cannot be read: {e}")))?;
        let sha256 = {
            use sha2::Digest;
            format!("{:x}", sha2::Sha256::digest(&bytes))
        };
        let file = netcrust::File::from_bytes(&bytes).map_err(|e| {
            refusal(path, format!("is not a netCDF file this reader can open: {e}"))
        })?;
        match file.attribute("schema").and_then(|a| a.as_string().map(str::to_string)) {
            Some(s) if s.trim_end_matches('\0') == RASTER_SCHEMA => {}
            Some(s) => {
                return Err(refusal(
                    path,
                    format!(
                        "declares schema {s:?}, not {RASTER_SCHEMA:?}. A raster of some other meaning read as spacing in km would refine the wrong places at the wrong resolution"
                    ),
                ));
            }
            None => {
                return Err(refusal(
                    path,
                    format!(
                        "carries no `schema` global attribute; a density raster declares schema = {RASTER_SCHEMA:?}, so nothing says this grid holds a hexagon spacing in km"
                    ),
                ));
            }
        }
        let declared = match file.attribute("min_spacing_km").and_then(|a| a.as_f64()) {
            Some(v) if v.is_finite() && v > 0.0 => v,
            Some(v) => {
                return Err(refusal(
                    path,
                    format!("declares min_spacing_km = {v}; a finest spacing has to be finite and positive"),
                ));
            }
            None => {
                return Err(refusal(
                    path,
                    "carries no numeric `min_spacing_km` global attribute, which the schema requires".to_string(),
                ));
            }
        };

        let axis = |name: &str| -> MpasResult<Vec<f64>> {
            // A netCDF-4 coordinate variable is an HDF5 dimension scale, which
            // the reader does not list among the variables; its DIMENSION is
            // what is checked then, and the values must fill it exactly.
            let dim = file.dimension(name).ok_or_else(|| {
                refusal(path, format!("has no `{name}` dimension; the schema's coordinates are 1-D variables on dimensions of the same name"))
            })?;
            if let Some(var) = file.variable(name) {
                let dims: Vec<String> =
                    var.dimensions().iter().map(|d| d.name().to_string()).collect();
                if dims != [name] {
                    return Err(refusal(
                        path,
                        format!("`{name}` has dimensions {dims:?}; the schema's coordinates are 1-D on `{name}`"),
                    ));
                }
            }
            let v = file
                .read_f64(name)
                .map_err(|e| refusal(path, format!("has no readable `{name}` coordinate: {e}")))?;
            if v.len() != dim.len() {
                return Err(refusal(
                    path,
                    format!("`{name}` holds {} values on a dimension of length {}", v.len(), dim.len()),
                ));
            }
            if v.len() < 2 {
                return Err(refusal(
                    path,
                    format!("`{name}` has {} value(s); bilinear sampling needs at least two along each axis", v.len()),
                ));
            }
            for (k, w) in v.windows(2).enumerate() {
                if !(w[0].is_finite() && w[1].is_finite()) {
                    return Err(refusal(path, format!("`{name}`[{k}..{}] is not finite", k + 1)));
                }
                if !(w[1] > w[0]) {
                    return Err(refusal(
                        path,
                        format!(
                            "`{name}` is not strictly ascending at index {k} ({} then {}); the schema requires ascending degrees, and a flipped axis would mirror the refinement",
                            w[0], w[1]
                        ),
                    ));
                }
            }
            Ok(v)
        };
        let lat_deg = axis("lat")?;
        let lon_deg = axis("lon")?;
        let (lat_lo, lat_hi) = (lat_deg[0], lat_deg[lat_deg.len() - 1]);
        if lat_lo < -MAX_ABS_LAT_DEG || lat_hi > MAX_ABS_LAT_DEG {
            return Err(refusal(
                path,
                format!(
                    "spans latitude {lat_lo}..{lat_hi}; rasters are admitted within +/-{MAX_ABS_LAT_DEG} degrees, because a latitude/longitude cell degenerates at the pole and the slope bound this raster is gated on uses the cell's poleward width"
                ),
            ));
        }
        let (lon_lo, lon_hi) = (lon_deg[0], lon_deg[lon_deg.len() - 1]);
        if !(lon_hi - lon_lo < 360.0) || lon_lo < -360.0 || lon_hi > 720.0 {
            return Err(refusal(
                path,
                format!("spans longitude {lon_lo}..{lon_hi}; a raster covers less than one full turn of longitude"),
            ));
        }

        let var = file
            .variable("spacing_km")
            .ok_or_else(|| refusal(path, "has no `spacing_km` variable".to_string()))?;
        let dims: Vec<String> = var.dimensions().iter().map(|d| d.name().to_string()).collect();
        if dims != ["lat", "lon"] {
            return Err(refusal(
                path,
                format!("`spacing_km` has dimensions {dims:?}; the schema requires (lat, lon) in that order"),
            ));
        }
        if !matches!(var.dtype(), netcrust::DataType::F64) {
            return Err(refusal(
                path,
                format!("`spacing_km` is stored as {:?}; the schema requires float64", var.dtype()),
            ));
        }
        let spacing_km = file
            .read_f64("spacing_km")
            .map_err(|e| refusal(path, format!("`spacing_km` cannot be read: {e}")))?;
        let (nlat, nlon) = (lat_deg.len(), lon_deg.len());
        if spacing_km.len() != nlat * nlon {
            return Err(refusal(
                path,
                format!(
                    "`spacing_km` holds {} values for a {nlat} x {nlon} grid",
                    spacing_km.len()
                ),
            ));
        }
        let mut finest = f64::INFINITY;
        let mut finest_at = [0.0, 0.0];
        for (k, &v) in spacing_km.iter().enumerate() {
            let (i, j) = (k / nlon, k % nlon);
            if !(v.is_finite() && v > 0.0) {
                return Err(refusal(
                    path,
                    format!(
                        "`spacing_km` is {v} at lat {} lon {}; every node has to be a finite positive spacing (a NaN or a hole is not \"no refinement\", it is an unknown request)",
                        lat_deg[i], lon_deg[j]
                    ),
                ));
            }
            if v < finest {
                finest = v;
                finest_at = [lat_deg[i], lon_deg[j]];
            }
        }
        if (declared / finest - 1.0).abs() > DECLARED_MIN_TOLERANCE {
            return Err(refusal(
                path,
                format!(
                    "declares min_spacing_km = {declared} but the finest stored node is {finest} km (at lat {}, lon {}); the attribute and the data describe different rasters",
                    finest_at[0], finest_at[1]
                ),
            ));
        }
        Ok(RasterSource {
            path: path.to_path_buf(),
            sha256,
            lat_deg,
            lon_deg,
            spacing_km,
            declared_min_spacing_km: declared,
            finest_km: finest,
            finest_at_deg: finest_at,
        })
    }
}

/// One `{"shape": "raster", ...}` row of a resolution spec.
///
/// Holds the loaded source, so evaluating the field never touches the disk,
/// and a cache of the prepared (clamped, snapped, limited) field keyed by
/// everything the preparation reads -- so the many places that call
/// [`crate::mesh::density::MeshSpec::prepared`] pay for the limiter once.
#[derive(Debug, Clone)]
pub struct RasterRegion {
    /// The path as the spec gave it (resolved against the spec's directory
    /// when it was relative and the spec named one).
    pub path: String,
    /// Gradient-limit the raster (`true`, the default) or refuse it if it is
    /// steeper than `max_gradient_per_cell` (`false`).
    pub limit: bool,
    /// The ceiling on the raster's slope, as a per-cell fraction.
    pub max_gradient_per_cell: f64,
    /// The ladder rung the raster's finest value was snapped to, set by
    /// [`crate::mesh::ladder_snap::snap_to_ladder`]. `None` until then (and
    /// when the finest value was already on the ladder).
    pub ladder_rung_km: Option<f64>,
    /// Index of this row in the spec's `regions` array, so a spec serialises
    /// back in the order it was written.
    pub position: usize,
    source: Arc<RasterSource>,
    cache: Arc<Mutex<Option<(PrepKey, Arc<PreparedRaster>)>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrepKey {
    background_bits: u64,
    rung_bits: Option<u64>,
    gradient_bits: u64,
    limit: bool,
}

/// The wire form of a raster row. Unknown keys are refused: a raster row
/// carrying `spacing_km` or `transition_km` is a confusion between the two
/// kinds of region, and taking either half silently would build one of them.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RasterRow {
    shape: String,
    path: String,
    #[serde(default = "default_true")]
    limit: bool,
    #[serde(default)]
    max_gradient_per_cell: Option<f64>,
    #[serde(default)]
    sha256: Option<String>,
    #[serde(default)]
    ladder_rung_km: Option<f64>,
    /// Written by this crate when it serialises a spec (the effective finest
    /// spacing, for readers that cannot open the raster). Read back and
    /// ignored: the raster itself is the authority.
    #[serde(default)]
    #[allow(dead_code)]
    finest_km: Option<f64>,
}

fn default_true() -> bool {
    true
}

/// Is this JSON region row a raster row?
pub fn is_raster_row(value: &serde_json::Value) -> bool {
    value.get("shape").and_then(|s| s.as_str()) == Some("raster")
}

impl RasterRegion {
    /// Parse one raster row, reading its file. `base` resolves a relative
    /// path; `None` leaves it relative to the working directory.
    pub fn from_row(
        value: serde_json::Value,
        position: usize,
        base: Option<&Path>,
    ) -> MpasResult<RasterRegion> {
        let row: RasterRow = serde_json::from_value(value).map_err(|e| {
            MpasError::Refusal(format!(
                "regions[{position}] is a raster row that does not parse: {e}. A raster row is {{\"shape\": \"raster\", \"path\": \"density.nc\"}} with optional \"limit\" (true|false) and \"max_gradient_per_cell\"; it carries no spacing_km or transition, because the raster is the spacing"
            ))
        })?;
        if row.shape != "raster" {
            return Err(MpasError::Refusal(format!(
                "regions[{position}] has shape {:?} where a raster row has \"raster\"",
                row.shape
            )));
        }
        let mut path = PathBuf::from(&row.path);
        if path.is_relative() {
            if let Some(b) = base {
                path = b.join(path);
            }
        }
        let g = row.max_gradient_per_cell.unwrap_or(DEFAULT_MAX_GRADIENT_PER_CELL);
        if !(g.is_finite() && g > 0.0 && g <= BAND_CEILING_PER_CELL * (1.0 + 1e-12)) {
            return Err(MpasError::Refusal(format!(
                "regions[{position}] asks for max_gradient_per_cell = {g}; the raster's slope ceiling has to be positive and no steeper than the generator's transition-band ceiling {BAND_CEILING_PER_CELL:.6} (2^(1/6) - 1), which refuses anything steeper at build time"
            )));
        }
        if let Some(r) = row.ladder_rung_km {
            if !(r.is_finite() && r > 0.0) {
                return Err(MpasError::Refusal(format!(
                    "regions[{position}] carries ladder_rung_km = {r}, which is not a spacing"
                )));
            }
        }
        let source = RasterSource::read(&path)?;
        if let Some(pin) = &row.sha256 {
            if !pin.eq_ignore_ascii_case(&source.sha256) {
                return Err(MpasError::Refusal(format!(
                    "regions[{position}] pins the raster at sha256 {pin}, but {} reads {}; the spec was written against a different file",
                    path.display(),
                    source.sha256
                )));
            }
        }
        Ok(RasterRegion {
            path: path.to_string_lossy().into_owned(),
            limit: row.limit,
            max_gradient_per_cell: g,
            ladder_rung_km: row.ladder_rung_km,
            position,
            source: Arc::new(source),
            cache: Arc::new(Mutex::new(None)),
        })
    }

    /// The raster as read.
    pub fn source(&self) -> &RasterSource {
        &self.source
    }

    /// The finest spacing the raster asks for once clamped at the
    /// background, before any ladder snap, in km.
    pub fn requested_finest_km(&self, background_km: f64) -> f64 {
        self.source.finest_km.min(background_km)
    }

    /// The finest spacing the prepared raster delivers, in km: the ladder
    /// rung when one was set, otherwise the clamped finest. Exact without
    /// running the limiter, because the limiter never lowers the minimum.
    pub fn effective_finest_km(&self, background_km: f64) -> f64 {
        match self.ladder_rung_km {
            Some(r) => r.min(self.requested_finest_km(background_km)),
            None => self.requested_finest_km(background_km),
        }
    }

    /// The clamped, snapped and (by default) limited field, cached.
    pub fn prepared(&self, background_km: f64) -> Arc<PreparedRaster> {
        let key = PrepKey {
            background_bits: background_km.to_bits(),
            rung_bits: self.ladder_rung_km.map(f64::to_bits),
            gradient_bits: self.max_gradient_per_cell.to_bits(),
            limit: self.limit,
        };
        {
            let slot = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((k, p)) = slot.as_ref() {
                if *k == key {
                    return Arc::clone(p);
                }
            }
        }
        // Built OUTSIDE the lock: the build runs rayon, and a worker that
        // steals a task asking this same region for its field must not find
        // the lock held by its own thread. Two racing builders compute the
        // same deterministic field and the second simply replaces the first.
        let p = Arc::new(PreparedRaster::build(
            &self.source,
            background_km,
            self.ladder_rung_km,
            self.max_gradient_per_cell,
            self.limit,
        ));
        let mut slot = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        *slot = Some((key, Arc::clone(&p)));
        p
    }

    /// The row as a spec serialises it.
    pub fn to_json(&self, background_km: f64) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("shape".into(), "raster".into());
        m.insert("path".into(), self.path.clone().into());
        m.insert("sha256".into(), self.source.sha256.clone().into());
        m.insert("limit".into(), self.limit.into());
        m.insert("max_gradient_per_cell".into(), self.max_gradient_per_cell.into());
        if let Some(r) = self.ladder_rung_km {
            m.insert("ladder_rung_km".into(), r.into());
        }
        m.insert("finest_km".into(), self.effective_finest_km(background_km).into());
        serde_json::Value::Object(m)
    }
}

/// What the limiter did, for the receipt.
#[derive(Debug, Clone, Serialize)]
pub struct LimiterRecord {
    pub applied: bool,
    pub method: &'static str,
    pub max_gradient_per_cell: f64,
    /// The per-edge allowance actually enforced, `g / sqrt(2)`.
    pub axis_allowance_per_cell: f64,
    pub nodes: usize,
    pub nodes_clamped_at_background: usize,
    pub nodes_lowered: usize,
    /// The largest `1 - h_limited / h_snapped` over the nodes.
    pub max_relative_lowering: f64,
    pub max_lowering_at_deg: [f64; 2],
    /// The steepest cell before limiting (after clamp and snap), as a
    /// per-cell fraction, and where it is.
    pub peak_per_cell_before: f64,
    pub peak_before_at_deg: [f64; 2],
}

/// A raster ready to be evaluated.
#[derive(Debug)]
pub struct PreparedRaster {
    lat_deg: Vec<f64>,
    lon_deg: Vec<f64>,
    /// Effective node spacing in metres, `[i * nlon + j]`.
    h_m: Vec<f64>,
    lon0: f64,
    lon_span: f64,
    cap_centre: V3,
    cos_cap: f64,
    pub finest_m: f64,
    pub certified_peak_per_cell: f64,
    pub certified_peak_at_deg: [f64; 2],
    pub limiter: LimiterRecord,
}

/// The cosine of the poleward latitude of cell row `i` (rows `i`, `i+1`).
fn cos_poleward(lat_deg: &[f64], i: usize) -> f64 {
    lat_deg[i].abs().max(lat_deg[i + 1].abs()).to_radians().cos()
}

/// The closed-form slope bound of every cell, and the largest, with where.
/// `componentwise`: east and north derivative bounds per cell, as in the
/// module docs.
fn peak_slope(lat: &[f64], lon: &[f64], h: &[f64]) -> (f64, [f64; 2]) {
    let (nlat, nlon) = (lat.len(), lon.len());
    let rows: Vec<(f64, [f64; 2])> = (0..nlat - 1)
        .into_par_iter()
        .map(|i| {
            let dphi = (lat[i + 1] - lat[i]).to_radians() * EARTH_RADIUS_M;
            let cpw = cos_poleward(lat, i);
            let mut best = (0.0f64, [lat[i], lon[0]]);
            for j in 0..nlon - 1 {
                let dlam = (lon[j + 1] - lon[j]).to_radians() * EARTH_RADIUS_M * cpw;
                let (a, b) = (h[i * nlon + j], h[i * nlon + j + 1]);
                let (c, d) = (h[(i + 1) * nlon + j], h[(i + 1) * nlon + j + 1]);
                let x = (b - a).abs().max((d - c).abs()) / dlam;
                let y = (c - a).abs().max((d - b).abs()) / dphi;
                let s = (x * x + y * y).sqrt();
                if s > best.0 {
                    best = (
                        s,
                        [0.5 * (lat[i] + lat[i + 1]), 0.5 * (lon[j] + lon[j + 1])],
                    );
                }
            }
            best
        })
        .collect();
    rows.into_iter()
        .fold((0.0, [lat[0], lon[0]]), |acc, r| if r.0 > acc.0 { r } else { acc })
}

/// Total order on f64 for the limiter's heap; values are finite here.
#[derive(Clone, Copy, PartialEq)]
struct Key(f64);
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// The largest field `<= h` with every east edge at most `a` times its
/// poleward-safe length and every north edge at most `a` times its length:
/// multi-source Dijkstra, every node a source at its own value.
fn limit_slope(lat: &[f64], lon: &[f64], h: &[f64], a: f64) -> Vec<f64> {
    let (nlat, nlon) = (lat.len(), lon.len());
    // East allowance on row i: the narrower of the two cells it borders.
    let east_cos: Vec<f64> = (0..nlat)
        .map(|i| {
            let mut c = f64::INFINITY;
            if i > 0 {
                c = c.min(cos_poleward(lat, i - 1));
            }
            if i + 1 < nlat {
                c = c.min(cos_poleward(lat, i));
            }
            c
        })
        .collect();
    let east_w = |i: usize, j: usize| -> f64 {
        a * (lon[j + 1] - lon[j]).to_radians() * EARTH_RADIUS_M * east_cos[i]
    };
    let north_w = |i: usize| -> f64 { a * (lat[i + 1] - lat[i]).to_radians() * EARTH_RADIUS_M };

    let mut dist = h.to_vec();
    let mut heap: BinaryHeap<Reverse<(Key, usize)>> =
        dist.iter().enumerate().map(|(k, &v)| Reverse((Key(v), k))).collect();
    let mut done = vec![false; dist.len()];
    while let Some(Reverse((Key(d), u))) = heap.pop() {
        if done[u] || d > dist[u] {
            continue;
        }
        done[u] = true;
        let (i, j) = (u / nlon, u % nlon);
        let mut relax = |v: usize, w: f64, heap: &mut BinaryHeap<Reverse<(Key, usize)>>| {
            let cand = d + w;
            if cand < dist[v] {
                dist[v] = cand;
                heap.push(Reverse((Key(cand), v)));
            }
        };
        if j + 1 < nlon {
            relax(u + 1, east_w(i, j), &mut heap);
        }
        if j > 0 {
            relax(u - 1, east_w(i, j - 1), &mut heap);
        }
        if i + 1 < nlat {
            relax(u + nlon, north_w(i), &mut heap);
        }
        if i > 0 {
            relax(u - nlon, north_w(i - 1), &mut heap);
        }
    }
    dist
}

impl PreparedRaster {
    fn build(
        src: &RasterSource,
        background_km: f64,
        rung_km: Option<f64>,
        g: f64,
        limit: bool,
    ) -> PreparedRaster {
        let bg_m = background_km * 1000.0;
        let (nlat, nlon) = (src.lat_deg.len(), src.lon_deg.len());
        // 1. clamp at the background
        let mut clamped = 0usize;
        let mut h: Vec<f64> = src
            .spacing_km
            .iter()
            .map(|&v| {
                let m = v * 1000.0;
                if m > bg_m {
                    clamped += 1;
                    bg_m
                } else {
                    m
                }
            })
            .collect();
        let finest_m = h.iter().copied().fold(f64::INFINITY, f64::min);
        // 2. the affine ladder snap: finest -> rung, background -> background
        if let Some(r) = rung_km {
            let rung_m = r * 1000.0;
            if rung_m < finest_m && finest_m < bg_m {
                let slope = (bg_m - rung_m) / (bg_m - finest_m);
                for v in h.iter_mut() {
                    if *v >= bg_m {
                        *v = bg_m;
                    } else if *v == finest_m {
                        *v = rung_m;
                    } else {
                        *v = (rung_m + (*v - finest_m) * slope).min(bg_m);
                    }
                }
            }
        }
        let snapped = h.clone();
        let (before, before_at) = peak_slope(&src.lat_deg, &src.lon_deg, &snapped);
        // 3. the limiter
        let a = g / std::f64::consts::SQRT_2 * (1.0 - LIMITER_MARGIN);
        if limit && before > a {
            h = limit_slope(&src.lat_deg, &src.lon_deg, &snapped, a);
        }
        let mut lowered = 0usize;
        let mut worst = (0.0f64, [src.lat_deg[0], src.lon_deg[0]]);
        for (k, (&s, &l)) in snapped.iter().zip(&h).enumerate() {
            if l < s {
                lowered += 1;
                let rel = 1.0 - l / s;
                if rel > worst.0 {
                    worst = (rel, [src.lat_deg[k / nlon], src.lon_deg[k % nlon]]);
                }
            }
        }
        let (cert, cert_at) = peak_slope(&src.lat_deg, &src.lon_deg, &h);
        let finest_after = h.iter().copied().fold(f64::INFINITY, f64::min);

        // A cap that contains the whole extent: the farthest point of a
        // latitude/longitude box from its centre is one of the corner
        // candidates (see `PreparedShape::probe_bound`'s proof), padded by
        // a hair so the reject test can never drop an inside point. That
        // proof holds only under a quarter turn -- past it the farthest point
        // can sit partway along the far meridian -- so a wider extent gets no
        // cap at all and every point takes the exact latitude/longitude test.
        let (lat0, lat1) = (src.lat_deg[0].to_radians(), src.lat_deg[nlat - 1].to_radians());
        let lon0 = src.lon_deg[0];
        let lon_span = src.lon_deg[nlon - 1] - lon0;
        let lon_c = (lon0 + 0.5 * lon_span).to_radians();
        let cap_centre = from_lat_lon(0.5 * (lat0 + lat1), lon_c);
        let half = (0.5 * lon_span).to_radians();
        let mut radius = 0.0f64;
        for lat in [lat0, lat1] {
            for s in [1.0, -1.0] {
                radius = radius.max(arc(cap_centre, from_lat_lon(lat, lon_c + s * half)));
            }
        }
        let radius = radius + 1e-9;
        let cos_cap = if radius >= std::f64::consts::FRAC_PI_2 || half >= std::f64::consts::FRAC_PI_2 {
            -2.0
        } else {
            radius.cos()
        };

        PreparedRaster {
            lat_deg: src.lat_deg.clone(),
            lon_deg: src.lon_deg.clone(),
            h_m: h,
            lon0,
            lon_span,
            cap_centre,
            cos_cap,
            finest_m: finest_after,
            certified_peak_per_cell: cert,
            certified_peak_at_deg: cert_at,
            limiter: LimiterRecord {
                applied: limit,
                method: "multi-source Dijkstra slope limiting on the node graph (h' = min_v h(v) + a d(u,v)), east edges at the poleward cell width, axis allowance a = g / sqrt(2)",
                max_gradient_per_cell: g,
                axis_allowance_per_cell: a,
                nodes: nlat * nlon,
                nodes_clamped_at_background: clamped,
                nodes_lowered: lowered,
                max_relative_lowering: worst.0,
                max_lowering_at_deg: worst.1,
                peak_per_cell_before: before,
                peak_before_at_deg: before_at,
            },
        }
    }

    pub fn nlat(&self) -> usize {
        self.lat_deg.len()
    }
    pub fn nlon(&self) -> usize {
        self.lon_deg.len()
    }

    /// The node value in metres.
    pub fn node_m(&self, i: usize, j: usize) -> f64 {
        self.h_m[i * self.lon_deg.len() + j]
    }

    /// The node position.
    pub fn node_xyz(&self, i: usize, j: usize) -> V3 {
        from_lat_lon(self.lat_deg[i].to_radians(), self.lon_deg[j].to_radians())
    }

    /// The node position in degrees.
    pub fn node_deg(&self, i: usize, j: usize) -> [f64; 2] {
        [self.lat_deg[i], self.lon_deg[j]]
    }

    /// `(i, j, ty, tx)` for a point inside the extent, `None` outside.
    #[inline]
    fn locate(&self, p: V3) -> Option<(usize, usize, f64, f64)> {
        if dot(p, self.cap_centre) < self.cos_cap {
            return None;
        }
        let (lat, lon) = lat_lon(p);
        let (lat, lon) = (lat.to_degrees(), lon.to_degrees());
        let nlat = self.lat_deg.len();
        // EDGE_SLACK: a point built from an edge node's own coordinates comes
        // back through `lat_lon` a few ulps either side of it, and an edge
        // node that read as "outside" would hide the raster's own boundary
        // from every check made there. 1e-9 degrees is 0.1 mm.
        const EDGE_SLACK: f64 = 1e-9;
        if !(lat >= self.lat_deg[0] - EDGE_SLACK && lat <= self.lat_deg[nlat - 1] + EDGE_SLACK) {
            return None;
        }
        let lat = lat.clamp(self.lat_deg[0], self.lat_deg[nlat - 1]);
        let mut x = (lon - self.lon0 + EDGE_SLACK).rem_euclid(360.0) - EDGE_SLACK;
        if !(x <= self.lon_span + EDGE_SLACK) {
            return None;
        }
        x = x.clamp(0.0, self.lon_span);
        let lon = self.lon0 + x;
        let nlon = self.lon_deg.len();
        let i = self.lat_deg.partition_point(|&v| v <= lat).clamp(1, nlat - 1) - 1;
        let j = self.lon_deg.partition_point(|&v| v <= lon).clamp(1, nlon - 1) - 1;
        let ty = ((lat - self.lat_deg[i]) / (self.lat_deg[i + 1] - self.lat_deg[i])).clamp(0.0, 1.0);
        let tx = ((lon - self.lon_deg[j]) / (self.lon_deg[j + 1] - self.lon_deg[j])).clamp(0.0, 1.0);
        Some((i, j, ty, tx))
    }

    /// Is `p` inside the raster's extent?
    pub fn contains(&self, p: V3) -> bool {
        self.locate(p).is_some()
    }

    /// The raster's spacing at `p` in metres, bilinear in latitude and
    /// longitude, or `None` outside the extent.
    #[inline]
    pub fn sample_m(&self, p: V3) -> Option<f64> {
        let (i, j, ty, tx) = self.locate(p)?;
        let nlon = self.lon_deg.len();
        let a = self.h_m[i * nlon + j];
        let b = self.h_m[i * nlon + j + 1];
        let c = self.h_m[(i + 1) * nlon + j];
        let d = self.h_m[(i + 1) * nlon + j + 1];
        let bottom = a + (b - a) * tx;
        let top = c + (d - c) * tx;
        Some(bottom + (top - bottom) * ty)
    }

    /// The outermost ring of nodes, each once.
    pub fn edge_nodes(&self) -> Vec<(usize, usize)> {
        let (nlat, nlon) = (self.nlat(), self.nlon());
        let mut out = Vec::with_capacity(2 * (nlat + nlon));
        for j in 0..nlon {
            out.push((0, j));
            out.push((nlat - 1, j));
        }
        for i in 1..nlat - 1 {
            out.push((i, 0));
            out.push((i, nlon - 1));
        }
        out
    }

    /// Points on the raster's boundary, in degrees: every edge node and
    /// `between` evenly spaced points inside each edge segment. Along its
    /// boundary the raster is linear between nodes while the field outside
    /// need not be, so the edge step is read between the nodes as well.
    pub fn boundary_points_deg(&self, between: usize) -> Vec<[f64; 2]> {
        let (lat, lon) = (&self.lat_deg, &self.lon_deg);
        let (nlat, nlon) = (lat.len(), lon.len());
        let mut out = Vec::with_capacity(2 * (nlat + nlon) * (between + 1));
        let mut side = |fixed_is_lat: bool, fixed: f64, axis: &[f64]| {
            for k in 0..axis.len() {
                let mut push = |x: f64| {
                    out.push(if fixed_is_lat { [fixed, x] } else { [x, fixed] });
                };
                push(axis[k]);
                if k + 1 < axis.len() {
                    for b in 1..=between {
                        let t = b as f64 / (between + 1) as f64;
                        push(axis[k] + (axis[k + 1] - axis[k]) * t);
                    }
                }
            }
        };
        side(true, lat[0], lon);
        side(true, lat[nlat - 1], lon);
        side(false, lon[0], lat);
        side(false, lon[nlon - 1], lat);
        out
    }

    /// Probe points for the gradient meter: the finest node, every edge node
    /// (where the raster meets the field outside it), and the interior nodes
    /// of every non-constant cell, decimated by a common stride so the count
    /// stays under `cap`.
    pub fn probes(&self, cap: usize, out: &mut Vec<V3>) {
        let (nlat, nlon) = (self.nlat(), self.nlon());
        let cap = cap.min(RASTER_PROBE_CAP).max(1);
        let mut k_min = 0usize;
        for (k, &v) in self.h_m.iter().enumerate() {
            if v < self.h_m[k_min] {
                k_min = k;
            }
        }
        out.push(self.node_xyz(k_min / nlon, k_min % nlon));
        let edges = self.edge_nodes();
        let edge_stride = edges.len().div_ceil(cap / 2 + 1).max(1);
        for (n, &(i, j)) in edges.iter().enumerate() {
            if n % edge_stride == 0 {
                out.push(self.node_xyz(i, j));
            }
        }
        // One stride per axis, so a corridor two nodes wide and a million
        // long is decimated along its length rather than only across it:
        // the latitude stride never exceeds the axis, and the longitude
        // stride is then solved for what is left of the budget.
        let target = (cap / 2 + 1) as f64;
        let s = ((nlat * nlon) as f64 / target).sqrt().ceil().max(1.0);
        let si = (s as usize).clamp(1, nlat);
        let rows = nlat.div_ceil(si) as f64;
        let sj = ((rows * nlon as f64 / target).ceil() as usize).max(1);
        for i in (0..nlat).step_by(si) {
            for j in (0..nlon).step_by(sj) {
                // A node every one of whose cells is flat carries nothing.
                let v = self.node_m(i, j);
                let mut varies = false;
                for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let (ii, jj) = (i as i64 + di, j as i64 + dj);
                    if ii >= 0 && jj >= 0 && (ii as usize) < nlat && (jj as usize) < nlon
                        && self.node_m(ii as usize, jj as usize) != v
                    {
                        varies = true;
                        break;
                    }
                }
                if varies {
                    out.push(self.node_xyz(i, j));
                }
            }
        }
    }

    /// `integral over the extent of dA / h^2` for the COMBINED field
    /// `spacing` (metres), skipping any sub-point `skip` claims (an earlier
    /// raster's extent, so overlaps are counted once). Each cell is split
    /// into `k x k` equal-area pieces with `k` set by the finest combined
    /// spacing at its corners, so a cell finer than its own pitch is not
    /// read from one midpoint. Deterministic: rows are summed in order.
    pub fn inverse_area_integral(
        &self,
        spacing: &(impl Fn(V3) -> f64 + Sync),
        skip: &(impl Fn(V3) -> bool + Sync),
    ) -> f64 {
        let (nlat, nlon) = (self.nlat(), self.nlon());
        let node_h: Vec<f64> = (0..nlat * nlon)
            .into_par_iter()
            .map(|k| spacing(self.node_xyz(k / nlon, k % nlon)))
            .collect();
        let r2 = EARTH_RADIUS_M * EARTH_RADIUS_M;
        let rows: Vec<f64> = (0..nlat - 1)
            .into_par_iter()
            .map(|i| {
                let (p0, p1) = (self.lat_deg[i].to_radians(), self.lat_deg[i + 1].to_radians());
                let (s0, s1) = (p0.sin(), p1.sin());
                let wide_cos = if p0 <= 0.0 && p1 >= 0.0 {
                    1.0
                } else {
                    p0.abs().min(p1.abs()).cos()
                };
                let dy = (p1 - p0) * EARTH_RADIUS_M;
                let mut row = 0.0f64;
                for j in 0..nlon - 1 {
                    let (l0, l1) = (self.lon_deg[j].to_radians(), self.lon_deg[j + 1].to_radians());
                    let dx = (l1 - l0) * EARTH_RADIUS_M * wide_cos;
                    let hmin = node_h[i * nlon + j]
                        .min(node_h[i * nlon + j + 1])
                        .min(node_h[(i + 1) * nlon + j])
                        .min(node_h[(i + 1) * nlon + j + 1]);
                    let k = ((dx.max(dy) / hmin).ceil() as usize).clamp(1, QUADRATURE_MAX_SPLIT);
                    let piece = r2 * (l1 - l0) / k as f64 * (s1 - s0) / k as f64;
                    for a in 0..k {
                        let s = s0 + (s1 - s0) * (a as f64 + 0.5) / k as f64;
                        let lat = s.clamp(-1.0, 1.0).asin();
                        for b in 0..k {
                            let lon = l0 + (l1 - l0) * (b as f64 + 0.5) / k as f64;
                            let p = from_lat_lon(lat, lon);
                            if skip(p) {
                                continue;
                            }
                            let h = spacing(p);
                            row += piece / (h * h);
                        }
                    }
                }
                row
            })
            .collect();
        rows.into_iter().sum()
    }
}

/// What one raster region delivered, for the dry run and the receipt.
#[derive(Debug, Clone, Serialize)]
pub struct RasterReport {
    /// Index of the row in the spec's `regions` array.
    pub region: usize,
    pub schema: &'static str,
    pub sha256: String,
    pub lat_nodes: usize,
    pub lon_nodes: usize,
    pub lat_extent_deg: [f64; 2],
    pub lon_extent_deg: [f64; 2],
    /// The finest value stored in the file, in km.
    pub raster_finest_km: f64,
    pub raster_finest_at_deg: [f64; 2],
    /// The finest value delivered after the ladder snap, in km.
    pub delivered_finest_km: f64,
    pub ladder_rung_km: Option<f64>,
    pub limiter: LimiterRecord,
    /// The closed-form upper bound on the prepared raster's slope.
    pub certified_peak_per_cell: f64,
    pub certified_peak_at_deg: [f64; 2],
    /// The largest step where the raster's edge meets the field outside it,
    /// `h_outside / h_edge - 1`, zero when the edge is no finer than outside.
    pub edge_step_per_cell: f64,
    pub edge_step_at_deg: [f64; 2],
}

/// Write a `woof-hex.density.v1` raster as classic netCDF (CDF-2).
///
/// The Rust twin of the Python builder, for tests and probes that need a
/// raster without a Python environment. `spacing_km` is `[i * nlon + j]`;
/// `min_spacing_km` and `schema` are written as given, so a test can write a
/// raster that breaks the contract on purpose.
pub fn write_density_raster(
    path: &Path,
    lat_deg: &[f64],
    lon_deg: &[f64],
    spacing_km: &[f64],
    min_spacing_km: f64,
    schema: &str,
) -> MpasResult<()> {
    use rw_store::netcdf_classic::{
        NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
    };
    let io = |e: rw_store::RwStoreError| {
        MpasError::Refusal(format!("cannot write {}: {e}", path.display()))
    };
    let mut w = NcClassicWriter::create(
        path,
        NcFormat::Offset64,
        vec![NcDim::fixed("lat", lat_deg.len()), NcDim::fixed("lon", lon_deg.len())],
        vec![
            NcAttr::text("schema", schema),
            NcAttr::doubles("min_spacing_km", vec![min_spacing_km]),
            NcAttr::text("Conventions", "CF-1.8"),
        ],
        vec![
            NcVarDef::new("lat", NcType::Double, vec![0])
                .with_attrs(vec![NcAttr::text("units", "degrees_north")]),
            NcVarDef::new("lon", NcType::Double, vec![1])
                .with_attrs(vec![NcAttr::text("units", "degrees_east")]),
            NcVarDef::new("spacing_km", NcType::Double, vec![0, 1])
                .with_attrs(vec![NcAttr::text("units", "km")]),
        ],
        0,
    )
    .map_err(io)?;
    w.put("lat", NcData::Doubles(lat_deg)).map_err(io)?;
    w.put("lon", NcData::Doubles(lon_deg)).map_err(io)?;
    w.put("spacing_km", NcData::Doubles(spacing_km)).map_err(io)?;
    w.finish().map_err(io)?;
    Ok(())
}

#[cfg(test)]
mod tests;
