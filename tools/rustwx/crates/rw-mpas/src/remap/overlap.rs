//! First-order conservative weights: Voronoi-polygon intersections on the
//! sphere.
//!
//! For every target cell `j` and every source cell `i` the operator holds the
//! area `a_ij` of the spherical polygon `S_i ∩ T_j`.  A conservative remap is
//! then `f_j = Σ_i a_ij f_i / Σ_i a_ij`, and when the target is covered
//! (`Σ_i a_ij = |T_j|`) its integral is
//! `Σ_j |T_j| f_j = Σ_i f_i Σ_j a_ij`: the source integral over the part of
//! the source the target covers, exactly, whatever the two meshes are.
//!
//! ## The clip
//! Both polygons are convex on the sphere (a Voronoi cell is the
//! intersection of hemispheres), so the intersection is Sutherland-Hodgman
//! against each great-circle edge of the target.  A great-circle arc is the
//! sphere's straight line, so the clip is exact in three dimensions: the arc
//! from `p` to `q` lies in the plane through `p`, `q` and the origin, and the
//! intersection with the clip plane is that chord's crossing, renormalised.
//!
//! Every signed distance and area is computed in DIFFERENCED form,
//! `n · (p - a)` with `n = a × (b - a)`, and the area through
//! [`crate::mesh::geom::tri_area`], which is the same expression the
//! generator's kite partition closes to machine precision with.  At 50 m
//! cells the undifferenced triple product cancels ten of its sixteen digits.
//!
//! ## Which source cells are tried
//! The target cell's own centre lies in the Voronoi cell of its nearest
//! source generator, so the walk starts there and spreads over
//! `cellsOnCell` from every source cell that actually overlaps.  The overlap
//! of a convex target with a tiling is connected, so the walk visits every
//! overlapping cell and few others.  The nearest cell's neighbours are
//! always tried even when it does not overlap, which is what lets a target
//! hanging past a regional source's edge still find the cells it does touch,
//! and report the rest as uncovered rather than as zero.

#![allow(clippy::needless_range_loop)]

use rayon::prelude::*;

use crate::error::{MpasError, MpasResult};
use crate::mesh::geom::{self, V3};
use crate::remap::mesh::RemapMesh;
use crate::weights::KdTree;

/// Intersections smaller than this fraction of the target cell are slivers
/// produced by shared or collinear edges and are dropped.
const SLIVER: f64 = 1.0e-12;

/// A walk that visits more source cells than this for one target cell is a
/// mesh pair the walk was not designed for (or a broken polygon), not a
/// coarse target: a 50 m source under a 100 km target is ~10^6, and that
/// remap is refused by name rather than run for an hour.
const MAX_VISITS: usize = 200_000;

/// The conservative operator, one row per target cell.
#[derive(Debug, Clone)]
pub struct Overlap {
    /// `rows[j]` = `(source cell, overlap area on the unit sphere)`.
    pub rows: Vec<Vec<(u32, f64)>>,
    /// Target polygon area on the unit sphere.
    pub target_area: Vec<f64>,
    /// `Σ_i a_ij / |T_j|`.  `NaN` for a target cell with no polygon.
    pub coverage: Vec<f64>,
    /// Per source cell, `Σ_j a_ij`: how much of it the target covers.
    pub source_covered: Vec<f64>,
}

impl Overlap {
    /// Target cells whose row is usable (non-empty).
    pub fn has_row(&self, j: usize) -> bool {
        !self.rows[j].is_empty()
    }
}

/// Signed side of `p` against the great circle through `a` then `b`:
/// positive on the left (the interior of a counter-clockwise ring).
#[inline]
fn side(n: V3, a: V3, p: V3) -> f64 {
    geom::dot(n, geom::sub(p, a))
}

/// Clip a convex spherical polygon by a convex spherical polygon.  Both are
/// counter-clockwise seen from outside the sphere.
pub fn clip(subject: &[V3], clip_ring: &[V3]) -> Vec<V3> {
    let mut output: Vec<V3> = subject.to_vec();
    let m = clip_ring.len();
    let mut input: Vec<V3> = Vec::with_capacity(subject.len() + m);
    for k in 0..m {
        if output.is_empty() {
            break;
        }
        let a = clip_ring[k];
        let b = clip_ring[(k + 1) % m];
        let n = geom::cross(a, geom::sub(b, a));
        std::mem::swap(&mut input, &mut output);
        output.clear();
        let len = input.len();
        for i in 0..len {
            let p = input[i];
            let q = input[(i + 1) % len];
            let dp = side(n, a, p);
            let dq = side(n, a, q);
            let p_in = dp >= 0.0;
            let q_in = dq >= 0.0;
            if p_in {
                output.push(p);
            }
            if p_in != q_in {
                let t = dp / (dp - dq);
                let x = geom::add(p, geom::scale(geom::sub(q, p), t));
                if let Some(u) = geom::unit(x) {
                    output.push(u);
                }
            }
        }
    }
    output
}

/// Area of the intersection of two convex spherical polygons.
pub fn intersection_area(subject: &[V3], clip_ring: &[V3]) -> f64 {
    let poly = clip(subject, clip_ring);
    if poly.len() < 3 {
        return 0.0;
    }
    geom::polygon_area(&poly).max(0.0)
}

/// Build the conservative operator from `source` onto `target`.
pub fn build(source: &RemapMesh, target: &RemapMesh) -> MpasResult<Overlap> {
    let tree = KdTree::build(source.cell_xyz.clone());
    let n_src = source.n_cells;

    let rows: Vec<MpasResult<Vec<(u32, f64)>>> = (0..target.n_cells)
        .into_par_iter()
        .map(|j| {
            let Some(tpoly) = &target.polygons[j] else {
                return Ok(Vec::new());
            };
            let tarea = target.polygon_area[j];
            let start = match tree.nearest_k(target.cell_xyz[j], 1).first() {
                Some(&(i, _)) => i as usize,
                None => return Ok(Vec::new()),
            };
            let mut visited: Vec<u32> = Vec::with_capacity(32);
            let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
            let mut row: Vec<(u32, f64)> = Vec::new();
            let push = |c: usize, visited: &mut Vec<u32>, queue: &mut std::collections::VecDeque<usize>| {
                if !visited.contains(&(c as u32)) {
                    visited.push(c as u32);
                    queue.push_back(c);
                }
            };
            push(start, &mut visited, &mut queue);
            // The nearest cell's ring is always tried.
            for &nb in &source.cells_on_cell[start][..source.n_edges_on_cell[start]] {
                if nb >= 1 && nb <= n_src {
                    push(nb - 1, &mut visited, &mut queue);
                }
            }
            while let Some(i) = queue.pop_front() {
                if visited.len() > MAX_VISITS {
                    return Err(MpasError::Refusal(format!(
                        "target cell {} overlaps more than {MAX_VISITS} source cells; a target this \
                         much coarser than its source is a coarsening the first-order operator \
                         would spend hours on.  Remap through an intermediate mesh",
                        j + 1
                    )));
                }
                let Some(spoly) = &source.polygons[i] else {
                    continue;
                };
                let a = intersection_area(spoly, tpoly);
                if a <= SLIVER * tarea {
                    continue;
                }
                row.push((i as u32, a));
                for &nb in &source.cells_on_cell[i][..source.n_edges_on_cell[i]] {
                    if nb >= 1 && nb <= n_src {
                        push(nb - 1, &mut visited, &mut queue);
                    }
                }
            }
            row.sort_unstable_by_key(|r| r.0);
            Ok(row)
        })
        .collect();

    let mut out_rows = Vec::with_capacity(target.n_cells);
    for r in rows {
        out_rows.push(r?);
    }
    let mut coverage = vec![f64::NAN; target.n_cells];
    let mut source_covered = vec![0.0f64; n_src];
    for j in 0..target.n_cells {
        if target.polygons[j].is_none() {
            continue;
        }
        let s: f64 = out_rows[j].iter().map(|r| r.1).sum();
        coverage[j] = s / target.polygon_area[j];
        for &(i, a) in &out_rows[j] {
            source_covered[i as usize] += a;
        }
    }
    Ok(Overlap {
        rows: out_rows,
        target_area: target.polygon_area.clone(),
        coverage,
        source_covered,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ll(lat_deg: f64, lon_deg: f64) -> V3 {
        geom::from_lat_lon(lat_deg.to_radians(), lon_deg.to_radians())
    }

    fn square(lat0: f64, lon0: f64, d: f64) -> Vec<V3> {
        vec![
            ll(lat0, lon0),
            ll(lat0, lon0 + d),
            ll(lat0 + d, lon0 + d),
            ll(lat0 + d, lon0),
        ]
    }

    #[test]
    fn a_polygon_clipped_by_itself_is_itself() {
        let s = square(10.0, 20.0, 1.0);
        let a = geom::polygon_area(&s);
        assert!((intersection_area(&s, &s) - a).abs() < 1e-15);
    }

    #[test]
    fn disjoint_polygons_do_not_intersect() {
        let s = square(10.0, 20.0, 1.0);
        let t = square(10.0, 22.0, 1.0);
        assert_eq!(intersection_area(&s, &t), 0.0);
    }

    #[test]
    fn half_overlap_is_about_half_the_area() {
        let s = square(0.0, 0.0, 0.01);
        let t = square(0.0, 0.005, 0.01);
        let a = geom::polygon_area(&s);
        let x = intersection_area(&s, &t);
        assert!((x / a - 0.5).abs() < 1e-6, "{}", x / a);
    }

    #[test]
    fn a_fifty_metre_cell_keeps_its_digits() {
        // 50 m on Earth is 7.85e-6 rad.
        let d = (50.0 / 6_371_229.0f64).to_degrees();
        let s = square(45.0, 7.0, d);
        let t = square(45.0, 7.0 + 0.25 * d, d);
        let a = geom::polygon_area(&s);
        let x = intersection_area(&s, &t);
        assert!((x / a - 0.75).abs() < 1e-6, "{}", x / a);
    }
}
