//! Regional-window generation: refine and relax only where a window asks.
//!
//! EXPERIMENTAL, and every surface that carries its output says so.
//!
//! THE MECHANISM. The hierarchical ladder ([`crate::mesh::hierarchy`]) builds
//! level 0 -- the uniform Goldberg arm at the background spacing -- over the
//! whole sphere exactly as it always has. With a window, every later level
//! then works inside one ACTIVE ZONE only: the window plus a halo. Edges are
//! split only where their midpoint is in the zone, Lloyd moves only zone
//! generators, and surgery neither moves, deletes nor inserts a generator
//! outside it. Generators outside the zone are the level-0 background, FROZEN
//! bit for bit, and they still take part in every triangulation, so the
//! result is a valid whole-sphere MPAS mesh that culls and validates exactly
//! like a global one.
//!
//! WHAT THE HALO IS FOR. A frozen generator was relaxed under the uniform
//! background field. That is only the right answer where the request IS the
//! background, so the zone has to contain every place the spec asks for a
//! finer spacing. Two distances therefore make the halo:
//!
//! * the SPILL: how far beyond the window the spec still asks for more than
//!   [`SPILL_TOLERANCE`] finer than the background. A `tanh` ramp reaches
//!   past its region by a few ramp widths, so this is rarely zero. It is
//!   measured on the field itself (adaptive quadrature below), never assumed.
//! * the RINGS: a margin of background cells beyond the spill, sized from the
//!   transition band -- `max(2 x surgery locality, ceil(band_cells))`, where
//!   `band_cells = ln 2 / ln(1 + g)` is how many cells one 2x band spans at
//!   the steepest requested gradient `g`. It is the room the relaxation has to
//!   settle the active cells against the frozen ones.
//!
//! The default halo is spill + rings. An explicit `--regional-halo-rings N`
//! means N background rings measured from the WINDOW, and a halo that does
//! not cover the spill is REFUSED with the minimum that would, rather than
//! silently building a mesh whose frozen cells are coarser than requested.
//!
//! THE TRADE-OFF, MEASURED rather than asserted: every regional receipt
//! carries [`HaloQuality`] -- the centroidal error (`delta/h` against the
//! spec's own field) of the frozen cells on the zone boundary, of the active
//! cells next to them, of the halo and of the window, and the largest
//! adjacent spacing ratio in the halo and in the window.
//!
//! COUNT TARGETING. The ladder's per-level cell target is a sizing integral.
//! The global path takes it on a 200,000-point golden-ratio lattice, about
//! 50 km between points, which cannot see a 100 m corridor at all. The zone
//! count here comes from an ADAPTIVE quadrature whose leaves are no wider
//! than the local requested spacing ([`LEAF_EDGE_OVER_SPACING`]). It cannot
//! skip a feature: the build already refuses any spec steeper than the
//! transition-band ceiling, so the field is Lipschitz with a constant under
//! 12.25 % per cell, and a leaf of edge `e <= h(centre)` therefore sees the
//! field vary by at most an eighth of its own spacing.

use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use serde::Serialize;

use crate::error::{MpasError, MpasResult};
use crate::mesh::density::{DensityField, PreparedShape, Shape};
use crate::mesh::derive::Rings;
use crate::mesh::geom::{EARTH_RADIUS_M, V3, add, arc, tri_area, unit};

/// How much finer than the background the spec may still ask for at a frozen
/// generator. Two percent is the ladder's own level-delivery median bound: a
/// frozen cell inside it is delivering its request as well as the gate holds
/// any active cell to.
pub const SPILL_TOLERANCE: f64 = 0.02;

/// A quadrature leaf stops splitting once its longest edge is at most this
/// many times the requested spacing at its centre.
pub const LEAF_EDGE_OVER_SPACING: f64 = 1.0;

/// Hard ceiling on quadrature leaves. About two leaves per delivered cell, so
/// this is a request in the tens of millions of cells -- refused here before
/// it is allowed to exhaust memory.
pub const QUADRATURE_MAX_LEAVES: usize = 60_000_000;

/// Base subdivision of the icosahedron the quadrature starts from:
/// `20 * 4^3 = 1,280` triangles of about 700 km.
const QUADRATURE_BASE_LEVEL: usize = 3;

/// Depth cap per leaf, below the base level. Thirty halvings of a 700 km edge
/// is under a millimetre; a field asking for less is refused by the leaf cap
/// long before this binds.
const QUADRATURE_MAX_DEPTH: usize = 30;

/// The halo margin's floor in background rings: twice the surgery locality
/// radius, the same floor the ladder puts under a transition band.
pub const MIN_HALO_RINGS: f64 = 6.0;

/// The receipt's status word. Not a quality grade: it says the generation
/// route itself has not been anchored by a forecast campaign.
pub const STATUS: &str = "experimental";

/// The window as the caller handed it, before anything was measured.
#[derive(Debug, Clone, Serialize)]
pub struct RegionalWindow {
    /// The window as a Shape row: the same `polygon` / `cap` / `lat_lon_box`
    /// grammar a spec's regions and a cull region use.
    pub shape: Shape,
    /// `geojson` (a GeoJSON Polygon, `[lon, lat]` order) or `shape_row`.
    pub source_format: String,
    /// An explicit halo in background rings measured from the window, or
    /// `None` for the default (spill plus the band-sized margin).
    pub halo_rings: Option<f64>,
}

/// Parse `--regional-window`: a GeoJSON Polygon (bare, as a Feature, or as a
/// one-feature FeatureCollection) or a Shape row.
///
/// GeoJSON writes positions `[longitude, latitude]` and a Shape row writes
/// `[latitude, longitude]`; the swap happens here, once. Holes and
/// multi-part geometries are refused rather than reduced to their outer ring
/// of the first part: a window that silently lost a part would leave that
/// part frozen at the background.
pub fn parse_window(text: &str) -> MpasResult<(Shape, &'static str)> {
    let value: serde_json::Value = serde_json::from_str(text).map_err(|e| {
        MpasError::Refusal(format!("the regional window is not valid JSON: {e}"))
    })?;
    if value.get("kind").is_some() {
        let shape: Shape = serde_json::from_value(value).map_err(|e| {
            MpasError::Refusal(format!(
                "the regional window is not a Shape row: {e}. A row is {{\"kind\": \"polygon\", \"vertices_deg\": [[lat, lon], ...]}} (or cap / lat_lon_box)"
            ))
        })?;
        check_shape(&shape)?;
        return Ok((shape, "shape_row"));
    }
    let shape = geojson_polygon(&value)?;
    check_shape(&shape)?;
    Ok((shape, "geojson"))
}

fn geojson_polygon(value: &serde_json::Value) -> MpasResult<Shape> {
    let kind = value.get("type").and_then(|t| t.as_str()).ok_or_else(|| {
        MpasError::Refusal(
            "the regional window is neither a GeoJSON geometry (no \"type\") nor a Shape row (no \"kind\")".to_string(),
        )
    })?;
    match kind {
        "Feature" => {
            let geometry = value.get("geometry").ok_or_else(|| {
                MpasError::Refusal("the regional window Feature has no geometry".to_string())
            })?;
            geojson_polygon(geometry)
        }
        "FeatureCollection" => {
            let features = value
                .get("features")
                .and_then(|f| f.as_array())
                .ok_or_else(|| {
                    MpasError::Refusal(
                        "the regional window FeatureCollection has no features array".to_string(),
                    )
                })?;
            if features.len() != 1 {
                return Err(MpasError::Refusal(format!(
                    "the regional window FeatureCollection holds {} features; it must hold exactly one Polygon. Two windows are two meshes, and picking one would freeze the other at the background",
                    features.len()
                )));
            }
            geojson_polygon(&features[0])
        }
        "MultiPolygon" => {
            let parts = value
                .get("coordinates")
                .and_then(|c| c.as_array())
                .ok_or_else(|| {
                    MpasError::Refusal("the regional window MultiPolygon has no coordinates".to_string())
                })?;
            if parts.len() != 1 {
                return Err(MpasError::Refusal(format!(
                    "the regional window is a MultiPolygon of {} parts; give one Polygon (for example the convex hull of the parts). Keeping the first part would freeze the others at the background",
                    parts.len()
                )));
            }
            polygon_rings(&parts[0])
        }
        "Polygon" => {
            let rings = value.get("coordinates").ok_or_else(|| {
                MpasError::Refusal("the regional window Polygon has no coordinates".to_string())
            })?;
            polygon_rings(rings)
        }
        other => Err(MpasError::Refusal(format!(
            "the regional window is a GeoJSON {other}; only a Polygon (bare, a Feature, or a one-feature FeatureCollection) bounds an area"
        ))),
    }
}

fn polygon_rings(rings: &serde_json::Value) -> MpasResult<Shape> {
    let rings = rings.as_array().ok_or_else(|| {
        MpasError::Refusal("the regional window Polygon coordinates are not an array of rings".to_string())
    })?;
    if rings.len() != 1 {
        return Err(MpasError::Refusal(format!(
            "the regional window Polygon has {} rings; holes are not supported. A hole would be a frozen island inside the refined area, which the halo cannot grade into",
            rings.len()
        )));
    }
    let ring = rings[0].as_array().ok_or_else(|| {
        MpasError::Refusal("the regional window Polygon ring is not an array of positions".to_string())
    })?;
    let mut vertices: Vec<[f64; 2]> = Vec::with_capacity(ring.len());
    for (k, position) in ring.iter().enumerate() {
        let pair = position.as_array().filter(|p| p.len() >= 2).ok_or_else(|| {
            MpasError::Refusal(format!(
                "regional window position {k} is not [longitude, latitude]"
            ))
        })?;
        let lon = pair[0].as_f64();
        let lat = pair[1].as_f64();
        match (lat, lon) {
            (Some(lat), Some(lon)) => vertices.push([lat, lon]),
            _ => {
                return Err(MpasError::Refusal(format!(
                    "regional window position {k} is not two numbers"
                )));
            }
        }
    }
    // GeoJSON closes a ring by repeating the first position; a Shape row
    // does not.
    if vertices.len() >= 2 && vertices.first() == vertices.last() {
        vertices.pop();
    }
    Ok(Shape::Polygon {
        vertices_deg: vertices,
    })
}

fn check_shape(shape: &Shape) -> MpasResult<()> {
    match shape {
        Shape::Polygon { vertices_deg } => {
            if vertices_deg.len() < 3 {
                return Err(MpasError::Refusal(format!(
                    "the regional window polygon has {} distinct vertices; three is the fewest that bound an area",
                    vertices_deg.len()
                )));
            }
            for (k, v) in vertices_deg.iter().enumerate() {
                if !(v[0].is_finite() && v[1].is_finite() && v[0].abs() <= 90.0) {
                    return Err(MpasError::Refusal(format!(
                        "regional window vertex {k} reads latitude {} longitude {}; a latitude outside -90..90 (or a non-number) is not on the sphere. GeoJSON positions are [longitude, latitude]",
                        v[0], v[1]
                    )));
                }
            }
        }
        Shape::Cap { radius_km, .. } => {
            if !(radius_km.is_finite() && *radius_km > 0.0) {
                return Err(MpasError::Refusal(format!(
                    "the regional window cap has radius {radius_km} km; a window needs a positive radius"
                )));
            }
        }
        Shape::LatLonBox { lat_deg, .. } => {
            if !(lat_deg[0] - lat_deg[1]).is_normal() {
                return Err(MpasError::Refusal(
                    "the regional window box has no latitude extent".to_string(),
                ));
            }
        }
    }
    Ok(())
}

/// The window plus its halo: the only place a regional build moves anything.
#[derive(Debug, Clone)]
pub struct Zone {
    window: PreparedShape,
    /// The halo as a great-circle distance from the window boundary, radians.
    pub halo_rad: f64,
}

impl Zone {
    pub fn new(window: &Shape, halo_rad: f64) -> Zone {
        Zone {
            window: window.prepare(),
            halo_rad,
        }
    }

    /// Whether `p` is inside the window or within the halo of it.
    #[inline]
    pub fn active(&self, p: V3) -> bool {
        self.window.signed_distance(p) <= self.halo_rad
    }

    /// Signed distance to the WINDOW (not the zone), radians.
    #[inline]
    pub fn window_distance(&self, p: V3) -> f64 {
        self.window.signed_distance(p)
    }

    /// One flag per generator: `true` where it is FROZEN (outside the zone).
    pub fn frozen_mask(&self, points: &[V3]) -> Vec<bool> {
        points.par_iter().map(|&p| !self.active(p)).collect()
    }
}

/// One adaptive-quadrature leaf: a spherical triangle small against the
/// spacing the field asks for at its centre.
#[derive(Debug, Clone, Copy)]
pub struct Leaf {
    pub centre: V3,
    /// Spherical area, unit sphere.
    pub area: f64,
    /// Longest edge, radians.
    pub edge: f64,
    /// Requested spacing at the centre, radians.
    pub h: f64,
}

fn icosahedron() -> (Vec<V3>, Vec<[usize; 3]>) {
    let phi = (1.0 + 5f64.sqrt()) / 2.0;
    let raw: [V3; 12] = [
        [-1.0, phi, 0.0],
        [1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [1.0, -phi, 0.0],
        [0.0, -1.0, phi],
        [0.0, 1.0, phi],
        [0.0, -1.0, -phi],
        [0.0, 1.0, -phi],
        [phi, 0.0, -1.0],
        [phi, 0.0, 1.0],
        [-phi, 0.0, -1.0],
        [-phi, 0.0, 1.0],
    ];
    let verts: Vec<V3> = raw.iter().map(|&v| unit(v).expect("icosahedron vertex")).collect();
    let faces = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    (verts, faces)
}

fn midpoint(a: V3, b: V3) -> V3 {
    unit(add(a, b)).unwrap_or(a)
}

/// Adaptive quadrature leaves of `field` over the whole sphere, in a
/// deterministic order (base triangle, then depth-first).
pub fn quadrature_leaves<F: DensityField + Sync>(field: &F) -> MpasResult<Vec<Leaf>> {
    let (verts, faces) = icosahedron();
    let mut base: Vec<[V3; 3]> = faces
        .iter()
        .map(|f| [verts[f[0]], verts[f[1]], verts[f[2]]])
        .collect();
    for _ in 0..QUADRATURE_BASE_LEVEL {
        let mut next = Vec::with_capacity(base.len() * 4);
        for [a, b, c] in base {
            let (ab, bc, ca) = (midpoint(a, b), midpoint(b, c), midpoint(c, a));
            next.push([a, ab, ca]);
            next.push([ab, b, bc]);
            next.push([ca, bc, c]);
            next.push([ab, bc, ca]);
        }
        base = next;
    }
    let count = AtomicUsize::new(0);
    let per_base: Vec<Option<Vec<Leaf>>> = base
        .par_iter()
        .map(|tri| {
            let mut out: Vec<Leaf> = Vec::new();
            let mut stack: Vec<([V3; 3], usize)> = vec![(*tri, 0)];
            while let Some(([a, b, c], depth)) = stack.pop() {
                let centre = unit(add(add(a, b), c)).unwrap_or(a);
                let h = field.spacing_m(centre) / EARTH_RADIUS_M;
                let edge = arc(a, b).max(arc(b, c)).max(arc(c, a));
                if edge <= LEAF_EDGE_OVER_SPACING * h || depth >= QUADRATURE_MAX_DEPTH {
                    out.push(Leaf {
                        centre,
                        area: tri_area(a, b, c).abs(),
                        edge,
                        h,
                    });
                    if count.fetch_add(1, Ordering::Relaxed) >= QUADRATURE_MAX_LEAVES {
                        return None;
                    }
                    continue;
                }
                let (ab, bc, ca) = (midpoint(a, b), midpoint(b, c), midpoint(c, a));
                // Pushed in reverse so the pop order is the canonical
                // a-corner, b-corner, c-corner, centre.
                stack.push(([ab, bc, ca], depth + 1));
                stack.push(([ca, bc, c], depth + 1));
                stack.push(([ab, b, bc], depth + 1));
                stack.push(([a, ab, ca], depth + 1));
            }
            Some(out)
        })
        .collect();
    let mut leaves = Vec::new();
    for part in per_base {
        match part {
            Some(p) => leaves.extend(p),
            None => {
                return Err(MpasError::Refusal(format!(
                    "the regional quadrature needed more than {QUADRATURE_MAX_LEAVES} leaves -- about two per delivered cell, so this request is tens of millions of cells. Coarsen the finest spacing or shrink the refined area"
                )));
            }
        }
    }
    Ok(leaves)
}

/// Cells a hexagonal tessellation puts on `area` at spacing `h` (radians).
#[inline]
fn hex_cells(area: f64, h: f64) -> f64 {
    area / (3f64.sqrt() / 2.0 * h * h)
}

/// What the halo has to cover, measured on the field.
#[derive(Debug, Clone)]
pub struct RegionalPlan {
    pub zone: Zone,
    pub window: RegionalWindow,
    pub halo_rule: String,
    pub halo_rings: f64,
    pub spill_rad: f64,
    pub background_rad: f64,
    pub leaves: usize,
    /// `(area, h)` of every leaf inside the zone; the per-level count target
    /// is a sum over these under the level clamp.
    zone_leaves: Vec<(f64, f64)>,
    pub predicted_active_cells: f64,
    pub predicted_frozen_cells: f64,
}

impl RegionalPlan {
    /// Build the plan: measure the spill, size the halo, and refuse what the
    /// halo cannot honour.
    ///
    /// `band_cells` is the ladder's own `ln 2 / ln(1 + g)` reading for this
    /// spec. `background_m` is the spec's background spacing.
    pub fn new<F: DensityField + Sync>(
        window: &RegionalWindow,
        field: &F,
        background_m: f64,
        band_cells: f64,
    ) -> MpasResult<RegionalPlan> {
        let leaves = quadrature_leaves(field)?;
        let background_rad = background_m / EARTH_RADIUS_M;
        let probe = Zone::new(&window.shape, 0.0);
        // THE SPILL: the farthest a leaf that still asks for more than the
        // tolerance below the background sits from the window, plus its own
        // extent so a leaf straddling the line is not undercounted.
        let threshold = (1.0 - SPILL_TOLERANCE) * background_rad;
        let spill_rad = leaves
            .par_iter()
            .filter(|l| l.h < threshold)
            .map(|l| (probe.window_distance(l.centre) + l.edge).max(0.0))
            .reduce(|| 0.0f64, f64::max);
        let band_rings = if band_cells.is_finite() {
            band_cells.ceil().max(MIN_HALO_RINGS)
        } else {
            MIN_HALO_RINGS
        };
        let (halo_rad, halo_rings, rule) = match window.halo_rings {
            None => (
                spill_rad + band_rings * background_rad,
                band_rings,
                format!(
                    "auto: the spill ({:.1} km, where the spec still asks for more than {:.0}% finer than the background) plus max({MIN_HALO_RINGS:.0}, ceil(band_cells = {band_cells:.2})) = {band_rings:.0} background rings",
                    spill_rad * EARTH_RADIUS_M / 1000.0,
                    SPILL_TOLERANCE * 100.0
                ),
            ),
            Some(n) => {
                if !(n.is_finite() && n >= 0.0) {
                    return Err(MpasError::Refusal(format!(
                        "--regional-halo-rings {n} is not a ring count"
                    )));
                }
                let halo = n * background_rad;
                if halo < spill_rad {
                    let need = (spill_rad / background_rad).ceil();
                    return Err(MpasError::Refusal(format!(
                        "--regional-halo-rings {n} puts the frozen boundary {:.1} km from the window, but the spec still asks for more than {:.0}% finer than the {:.3} km background out to {:.1} km from it. Generators past the halo are frozen at the background, so that part of the request would be delivered {:.0}%+ too coarse with nothing in the file saying so. Pass at least --regional-halo-rings {need:.0}, omit the flag for the measured default (spill plus {band_rings:.0} rings), widen the window, or narrow the spec's transitions",
                        halo * EARTH_RADIUS_M / 1000.0,
                        SPILL_TOLERANCE * 100.0,
                        background_m / 1000.0,
                        spill_rad * EARTH_RADIUS_M / 1000.0,
                        SPILL_TOLERANCE * 100.0
                    )));
                }
                (halo, n, format!("explicit: --regional-halo-rings {n} background rings from the window (covers the {:.1} km spill)", spill_rad * EARTH_RADIUS_M / 1000.0))
            }
        };
        let zone = Zone::new(&window.shape, halo_rad);
        let flagged: Vec<bool> = leaves.par_iter().map(|l| zone.active(l.centre)).collect();
        let mut zone_leaves = Vec::new();
        let mut active = 0.0f64;
        let mut frozen = 0.0f64;
        for (l, &inside) in leaves.iter().zip(flagged.iter()) {
            if inside {
                zone_leaves.push((l.area, l.h));
                active += hex_cells(l.area, l.h);
            } else {
                frozen += hex_cells(l.area, background_rad);
            }
        }
        if frozen < 1.0 {
            return Err(MpasError::Refusal(format!(
                "the regional zone (window plus a {:.1} km halo) covers the whole sphere, so nothing would be frozen and the window would only add bookkeeping to a global build. The spec refines out to {:.1} km beyond the window; drop --regional-window, or use a spec whose refinement is contained in the window",
                halo_rad * EARTH_RADIUS_M / 1000.0,
                spill_rad * EARTH_RADIUS_M / 1000.0
            )));
        }
        Ok(RegionalPlan {
            zone,
            window: window.clone(),
            halo_rule: rule,
            halo_rings,
            spill_rad,
            background_rad,
            leaves: leaves.len(),
            zone_leaves,
            predicted_active_cells: active,
            predicted_frozen_cells: frozen,
        })
    }

    /// The zone's cell count under the level clamp `h >= level_spacing_m`.
    pub fn zone_cells_at(&self, level_spacing_m: f64) -> f64 {
        // SERIAL, in leaf order: this rounds into the level's cell target,
        // and a parallel float sum's order (so its last bit) depends on the
        // thread count -- one point more or fewer, and a different mesh.
        let floor = level_spacing_m / EARTH_RADIUS_M;
        self.zone_leaves
            .iter()
            .map(|&(area, h)| hex_cells(area, h.max(floor)))
            .sum()
    }

    /// The dry-run's account of the plan, before any generator exists.
    pub fn preview(&self) -> serde_json::Value {
        serde_json::json!({
            "status": STATUS,
            "window": self.window.shape,
            "window_format": self.window.source_format,
            "halo_rule": self.halo_rule,
            "halo_rings": self.halo_rings,
            "halo_km": self.zone.halo_rad * EARTH_RADIUS_M / 1000.0,
            "spill_km": self.spill_rad * EARTH_RADIUS_M / 1000.0,
            "spill_tolerance": SPILL_TOLERANCE,
            "quadrature_leaves": self.leaves,
            "predicted_active_cells": self.predicted_active_cells,
            "predicted_frozen_cells": self.predicted_frozen_cells,
            "predicted_cells": self.predicted_active_cells + self.predicted_frozen_cells,
        })
    }
}

/// Centroidal error and gradient in the places a regional build trades
/// quality for speed. Every `delta/h` here is measured against the SPEC's own
/// field, so the frozen cells are judged by what was asked of them, not by
/// the uniform field they were relaxed under.
#[derive(Debug, Clone, Serialize)]
pub struct HaloQuality {
    /// Frozen generators with at least one active neighbour.
    pub interface_frozen_cells: usize,
    pub interface_frozen_mean_delta_over_h: f64,
    pub interface_frozen_max_delta_over_h: f64,
    /// Active generators with at least one frozen neighbour.
    pub interface_active_cells: usize,
    pub interface_active_mean_delta_over_h: f64,
    pub interface_active_max_delta_over_h: f64,
    /// Active generators outside the window.
    pub halo_cells: usize,
    pub halo_mean_delta_over_h: f64,
    pub halo_max_delta_over_h: f64,
    /// Generators inside the window.
    pub window_cells: usize,
    pub window_mean_delta_over_h: f64,
    pub window_max_delta_over_h: f64,
    /// Largest neighbour-to-neighbour delivered spacing ratio over edges with
    /// an end in the halo (including the frozen interface), and in the window.
    pub halo_max_adjacent_spacing_ratio: f64,
    pub window_max_adjacent_spacing_ratio: f64,
}

/// What a regional build stamps into its receipt.
#[derive(Debug, Clone, Serialize)]
pub struct RegionalReceipt {
    pub status: &'static str,
    pub window: Shape,
    pub window_format: String,
    pub halo_rule: String,
    pub halo_rings: f64,
    pub halo_km: f64,
    pub spill_km: f64,
    pub spill_tolerance: f64,
    pub quadrature_leaves: usize,
    pub predicted_active_cells: f64,
    pub predicted_frozen_cells: f64,
    /// Level-0 generators held fixed through every later level.
    pub frozen_cells: usize,
    /// Every frozen generator was found, bit for bit, in the finished mesh.
    pub frozen_bitwise_unchanged: bool,
    pub active_cells: usize,
    pub halo_quality: HaloQuality,
    pub deliverable_note: &'static str,
}

/// The note every regional receipt carries.
pub const DELIVERABLE_NOTE: &str = "experimental regional window: generators outside the window plus halo are the level-0 background, frozen bit for bit; the spec is honoured inside the zone only, and the halo_quality block measures what that costs";

fn mean_max(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let max = values.iter().cloned().fold(0.0f64, f64::max);
    (values.iter().sum::<f64>() / values.len() as f64, max)
}

/// Measure [`HaloQuality`] on a finished point set and its Delaunay rings.
pub fn measure_halo<F: DensityField + Sync>(
    points: &[V3],
    rings: &Rings,
    zone: &Zone,
    field: &F,
) -> HaloQuality {
    let n = points.len();
    let frozen = zone.frozen_mask(points);
    let in_window: Vec<bool> = points.par_iter().map(|&p| zone.window_distance(p) <= 0.0).collect();
    let touches = |i: usize, want_frozen: bool| rings.ring(i).iter().any(|&j| frozen[j as usize] == want_frozen);
    // Only the cells a set below reads: every active one, and the frozen
    // ones on the zone boundary. A mostly frozen sphere is not integrated.
    let frozen_interface: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|i| frozen[i] && touches(i, false))
        .collect();
    let residual: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            if frozen[i] && !frozen_interface[i] {
                0.0
            } else {
                crate::mesh::lloyd::centroid_residual(points, rings, field, i)
            }
        })
        .collect();
    let spacing = crate::mesh::hierarchy::delivered_spacing_m(points, rings);
    let mut iface_frozen = Vec::new();
    let mut iface_active = Vec::new();
    let mut halo = Vec::new();
    let mut window = Vec::new();
    for i in 0..n {
        if frozen[i] {
            if frozen_interface[i] {
                iface_frozen.push(residual[i]);
            }
            continue;
        }
        if touches(i, true) {
            iface_active.push(residual[i]);
        }
        if in_window[i] {
            window.push(residual[i]);
        } else {
            halo.push(residual[i]);
        }
    }
    let halo_end: Vec<bool> = (0..n)
        .map(|k| !in_window[k] && (!frozen[k] || frozen_interface[k]))
        .collect();
    let mut halo_ratio = 1.0f64;
    let mut window_ratio = 1.0f64;
    for i in 0..n {
        for &j in rings.ring(i) {
            let j = j as usize;
            if j <= i || spacing[i] <= 0.0 || spacing[j] <= 0.0 {
                continue;
            }
            let r = (spacing[i] / spacing[j]).max(spacing[j] / spacing[i]);
            if halo_end[i] || halo_end[j] {
                halo_ratio = halo_ratio.max(r);
            }
            if in_window[i] && in_window[j] {
                window_ratio = window_ratio.max(r);
            }
        }
    }
    let (fm, fx) = mean_max(&iface_frozen);
    let (am, ax) = mean_max(&iface_active);
    let (hm, hx) = mean_max(&halo);
    let (wm, wx) = mean_max(&window);
    HaloQuality {
        interface_frozen_cells: iface_frozen.len(),
        interface_frozen_mean_delta_over_h: fm,
        interface_frozen_max_delta_over_h: fx,
        interface_active_cells: iface_active.len(),
        interface_active_mean_delta_over_h: am,
        interface_active_max_delta_over_h: ax,
        halo_cells: halo.len(),
        halo_mean_delta_over_h: hm,
        halo_max_delta_over_h: hx,
        window_cells: window.len(),
        window_mean_delta_over_h: wm,
        window_max_delta_over_h: wx,
        halo_max_adjacent_spacing_ratio: halo_ratio,
        window_max_adjacent_spacing_ratio: window_ratio,
    }
}

/// Bit patterns of a point set, for the frozen-generator invariant.
pub fn bit_set(points: &[V3]) -> std::collections::HashSet<[u64; 3]> {
    points
        .iter()
        .map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::density::{MeshSpec, Region, TransitionField};
    use crate::mesh::geom::from_lat_lon;

    #[test]
    fn a_geojson_polygon_is_read_longitude_first_and_its_closing_vertex_dropped() {
        let text = r#"{"type": "Polygon", "coordinates": [[[-4.0, 51.0], [-3.0, 51.0], [-3.0, 52.0], [-4.0, 52.0], [-4.0, 51.0]]]}"#;
        let (shape, format) = parse_window(text).unwrap();
        assert_eq!(format, "geojson");
        match shape {
            Shape::Polygon { vertices_deg } => {
                assert_eq!(vertices_deg.len(), 4);
                assert_eq!(vertices_deg[0], [51.0, -4.0], "GeoJSON [lon, lat] must become [lat, lon]");
            }
            other => panic!("not a polygon: {other:?}"),
        }
        // The same window as a Feature and as a one-feature collection.
        let feature = format!(r#"{{"type": "Feature", "properties": {{}}, "geometry": {text}}}"#);
        assert!(parse_window(&feature).is_ok());
        let collection = format!(r#"{{"type": "FeatureCollection", "features": [{feature}]}}"#);
        assert!(parse_window(&collection).is_ok());
    }

    #[test]
    fn a_shape_row_window_is_the_cull_grammar_unchanged() {
        let text = r#"{"kind": "polygon", "vertices_deg": [[51.0, -4.0], [51.0, -3.0], [52.0, -3.0]]}"#;
        let (shape, format) = parse_window(text).unwrap();
        assert_eq!(format, "shape_row");
        assert!(matches!(shape, Shape::Polygon { .. }));
        let cap = r#"{"kind": "cap", "center_deg": [51.5, -3.5], "radius_km": 100}"#;
        assert!(parse_window(cap).is_ok());
    }

    #[test]
    fn windows_that_would_lose_area_are_refused_by_name() {
        let hole = r#"{"type": "Polygon", "coordinates": [[[0,0],[2,0],[2,2],[0,2],[0,0]], [[0.5,0.5],[1,0.5],[1,1],[0.5,0.5]]]}"#;
        assert!(parse_window(hole).unwrap_err().to_string().contains("holes"));
        let multi = r#"{"type": "MultiPolygon", "coordinates": [[[[0,0],[1,0],[1,1],[0,0]]], [[[5,5],[6,5],[6,6],[5,5]]]]}"#;
        assert!(parse_window(multi).unwrap_err().to_string().contains("MultiPolygon of 2"));
        let two = r#"{"type": "FeatureCollection", "features": []}"#;
        assert!(parse_window(two).unwrap_err().to_string().contains("exactly one"));
        let line = r#"{"type": "LineString", "coordinates": [[0,0],[1,1]]}"#;
        assert!(parse_window(line).unwrap_err().to_string().contains("LineString"));
        let swapped = r#"{"kind": "polygon", "vertices_deg": [[-120.0, 50.0], [51.0, -3.0], [52.0, -3.0]]}"#;
        assert!(parse_window(swapped).unwrap_err().to_string().contains("latitude"));
        assert!(parse_window("{").unwrap_err().to_string().contains("not valid JSON"));
    }

    /// The adaptive count agrees with the global lattice integral where the
    /// lattice CAN see the field (a broad request), and it is the one that
    /// still sees a narrow one.
    #[test]
    fn the_adaptive_quadrature_counts_what_the_lattice_counts_on_a_broad_field() {
        let uniform = MeshSpec::uniform(240.0).prepared();
        let leaves = quadrature_leaves(&uniform).unwrap();
        let total: f64 = leaves.iter().map(|l| hex_cells(l.area, l.h)).sum();
        let lattice = crate::mesh::density::predicted_cells_of(&uniform, 200_000);
        assert!((total / lattice - 1.0).abs() < 1e-3, "uniform: adaptive {total:.1} vs lattice {lattice:.1}");
        let area: f64 = leaves.iter().map(|l| l.area).sum();
        assert!((area / (4.0 * std::f64::consts::PI) - 1.0).abs() < 1e-9, "leaves tile the sphere");

        let graded = MeshSpec {
            background_km: 480.0,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [39.0, -98.0],
                    radius_km: 2000.0,
                },
                spacing_km: 120.0,
                transition: TransitionField::Km(1500.0),
            }],
            name: None,
        }
        .prepared();
        let leaves = quadrature_leaves(&graded).unwrap();
        let total: f64 = leaves.iter().map(|l| hex_cells(l.area, l.h)).sum();
        let lattice = crate::mesh::density::predicted_cells_of(&graded, 400_000);
        assert!((total / lattice - 1.0).abs() < 5e-3, "graded: adaptive {total:.1} vs lattice {lattice:.1}");
    }

    fn corridor_spec() -> MeshSpec {
        MeshSpec {
            background_km: 480.0,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [45.0, 10.0],
                    radius_km: 800.0,
                },
                spacing_km: 240.0,
                transition: TransitionField::Km(1200.0),
            }],
            name: None,
        }
    }

    fn window_cap(radius_km: f64, halo_rings: Option<f64>) -> RegionalWindow {
        RegionalWindow {
            shape: Shape::Cap {
                center_deg: [45.0, 10.0],
                radius_km,
            },
            source_format: "shape_row".to_string(),
            halo_rings,
        }
    }

    #[test]
    fn the_default_halo_covers_the_measured_spill_and_an_explicit_short_one_is_refused() {
        let spec = corridor_spec().prepared();
        let plan = RegionalPlan::new(&window_cap(800.0, None), &spec, 480_000.0, 6.6).unwrap();
        let spill_km = plan.spill_rad * EARTH_RADIUS_M / 1000.0;
        assert!(spill_km > 500.0, "a 1200 km tanh ramp reaches well past its cap: spill {spill_km:.1} km");
        // Nothing frozen asks for more than the tolerance below background.
        let leaves = quadrature_leaves(&spec).unwrap();
        for l in &leaves {
            if !plan.zone.active(l.centre) {
                assert!(l.h * EARTH_RADIUS_M >= (1.0 - SPILL_TOLERANCE) * 480_000.0 - 1e-6);
            }
        }
        assert!(plan.predicted_frozen_cells > 100.0);
        let err = RegionalPlan::new(&window_cap(800.0, Some(1.0)), &spec, 480_000.0, 6.6)
            .unwrap_err()
            .to_string();
        assert!(err.contains("--regional-halo-rings"), "{err}");
        assert!(err.contains("too coarse"), "{err}");
    }

    #[test]
    fn a_zone_that_swallows_the_sphere_is_refused() {
        let spec = corridor_spec().prepared();
        let err = RegionalPlan::new(&window_cap(18_000.0, None), &spec, 480_000.0, 6.6)
            .unwrap_err()
            .to_string();
        assert!(err.contains("covers the whole sphere"), "{err}");
    }

    #[test]
    fn the_zone_is_the_window_grown_by_the_halo() {
        let zone = Zone::new(
            &Shape::Cap {
                center_deg: [0.0, 0.0],
                radius_km: 100.0,
            },
            50_000.0 / EARTH_RADIUS_M,
        );
        let at = |km: f64| from_lat_lon(0.0, (km * 1000.0 / EARTH_RADIUS_M) as f64);
        assert!(zone.active(at(0.0)));
        assert!(zone.active(at(149.0)));
        assert!(!zone.active(at(151.0)));
    }
}
