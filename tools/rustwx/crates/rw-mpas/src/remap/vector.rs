//! The edge-normal wind, through the cell-centre vector.
//!
//! MPAS carries one component of the wind per edge: `u`, normal to the edge,
//! positive from `cellsOnEdge[0]` to `cellsOnEdge[1]`.  A different mesh has
//! different edges pointing in different directions, so `u` cannot be
//! remapped as a scalar.  The route is MPAS's own diagnostic one:
//!
//! 1. reconstruct the full 3-D wind at every source cell centre from the
//!    normal components on its edges (`uReconstructX/Y/Z`);
//! 2. remap the three Cartesian components like any mass-specific scalar
//!    (momentum is what is conserved);
//! 3. project onto each target edge's normal, from the mean of the two
//!    target cells the edge separates.
//!
//! ## The reconstruction
//! When the source file carries a non-zero `coeffs_reconstruct` (the RBF
//! coefficients `mpas_rbf_interpolation.F` fills at model start-up), they are
//! used as is: `V_c = Σ_i coeffs[c][i] u_{e_i}`.  A static or freshly-cut
//! grid declares them as zeros, and then a least-squares fit of a tangent
//! vector to the cell's own normal components is used instead, which is the
//! same linear functional shape, exact for a uniform flow and second order
//! on the near-regular hexagons of a centroidal Voronoi mesh.  The receipt
//! says which ran.

#![allow(clippy::needless_range_loop)]

use crate::error::{MpasError, MpasResult};
use crate::mesh::geom::{self, V3};
use crate::remap::mesh::{tangent_basis, RemapMesh};

/// How the cell vectors were obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Reconstruction {
    CoeffsReconstruct,
    LeastSquares,
}

/// One row per cell: the edges it reads and the 3-D coefficient for each.
#[derive(Debug, Clone)]
pub struct CellVectorOperator {
    pub rows: Vec<Vec<(u32, V3)>>,
    pub method: Reconstruction,
}

impl CellVectorOperator {
    pub fn build(mesh: &RemapMesh) -> MpasResult<CellVectorOperator> {
        if let Some(coeffs) = &mesh.coeffs_reconstruct {
            let rows = (0..mesh.n_cells)
                .map(|c| {
                    (0..mesh.n_edges_on_cell[c])
                        .filter_map(|i| {
                            let e = mesh.edges_on_cell[c][i];
                            (e >= 1 && e <= mesh.n_edges).then(|| ((e - 1) as u32, coeffs[c][i]))
                        })
                        .collect()
                })
                .collect();
            return Ok(CellVectorOperator {
                rows,
                method: Reconstruction::CoeffsReconstruct,
            });
        }
        let mut rows = Vec::with_capacity(mesh.n_cells);
        for c in 0..mesh.n_cells {
            let own: Vec<usize> = (0..mesh.n_edges_on_cell[c])
                .filter_map(|i| {
                    let e = mesh.edges_on_cell[c][i];
                    (e >= 1 && e <= mesh.n_edges).then(|| e - 1)
                })
                .collect();
            let row = match fit(mesh, c, &own) {
                Some(r) => r,
                None => {
                    // Widen to the neighbours' edges before giving up.
                    let mut patch = own.clone();
                    for i in 0..mesh.n_edges_on_cell[c] {
                        let nb = mesh.cells_on_cell[c][i];
                        if nb >= 1 && nb <= mesh.n_cells {
                            for j in 0..mesh.n_edges_on_cell[nb - 1] {
                                let e = mesh.edges_on_cell[nb - 1][j];
                                if e >= 1 && e <= mesh.n_edges && !patch.contains(&(e - 1)) {
                                    patch.push(e - 1);
                                }
                            }
                        }
                    }
                    fit(mesh, c, &patch).ok_or_else(|| {
                        MpasError::Refusal(format!(
                            "source cell {} has no two edges in different directions, even with \
                             its neighbours' edges; its wind cannot be recovered from normal \
                             components and any value written for it would be one direction of \
                             the flow asserted as both",
                            c + 1
                        ))
                    })?
                }
            };
            rows.push(row);
        }
        Ok(CellVectorOperator {
            rows,
            method: Reconstruction::LeastSquares,
        })
    }

    /// The 3-D vector at cell `c` from one level of edge values laid out
    /// `[edge * levels + k]`.
    #[inline]
    pub fn apply(&self, c: usize, u: &[f32], levels: usize, k: usize) -> V3 {
        let mut v = [0.0f64; 3];
        for &(e, w) in &self.rows[c] {
            let ue = u[e as usize * levels + k] as f64;
            v[0] += w[0] * ue;
            v[1] += w[1] * ue;
            v[2] += w[2] * ue;
        }
        v
    }
}

/// Least-squares tangent vector at cell `c` from the normal components on
/// `edges`, as one coefficient vector per edge.
fn fit(mesh: &RemapMesh, c: usize, edges: &[usize]) -> Option<Vec<(u32, V3)>> {
    if edges.len() < 2 {
        return None;
    }
    let (east, north) = tangent_basis(mesh.cell_xyz[c]);
    let mut m = [[0.0f64; 2]; 2];
    let rows: Vec<[f64; 2]> = edges
        .iter()
        .map(|&e| {
            let n = mesh.edge_normal[e];
            [geom::dot(east, n), geom::dot(north, n)]
        })
        .collect();
    for r in &rows {
        m[0][0] += r[0] * r[0];
        m[0][1] += r[0] * r[1];
        m[1][0] += r[1] * r[0];
        m[1][1] += r[1] * r[1];
    }
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let trace = m[0][0] + m[1][1];
    if !(det > 1.0e-6 * trace * trace) {
        return None;
    }
    let inv = [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]];
    Some(
        edges
            .iter()
            .zip(rows.iter())
            .map(|(&e, r)| {
                let a = inv[0][0] * r[0] + inv[0][1] * r[1];
                let b = inv[1][0] * r[0] + inv[1][1] * r[1];
                (e as u32, geom::add(geom::scale(east, a), geom::scale(north, b)))
            })
            .collect(),
    )
}

/// The normal component on target edge `e` from the cell vectors either
/// side, `[cell * levels + k]`.
#[inline]
pub fn project(mesh: &RemapMesh, e: usize, cells: &[usize], vectors: &[[f32; 3]], levels: usize, k: usize) -> f64 {
    let n = mesh.edge_normal[e];
    let mut v = [0.0f64; 3];
    for &c in cells {
        let w = vectors[c * levels + k];
        v[0] += w[0] as f64;
        v[1] += w[1] as f64;
        v[2] += w[2] as f64;
    }
    let inv = 1.0 / cells.len().max(1) as f64;
    geom::dot(n, v) * inv
}
