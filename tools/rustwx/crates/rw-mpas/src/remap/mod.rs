//! Mesh-to-mesh state remap: an MPAS init or restart state on one mesh onto
//! another mesh with a different cell set.
//!
//! The case it exists for is the adaptive cycle: the next cycle's mesh is
//! regenerated around a feature the last one found, and the state the last
//! one ended on has to start the next one.  Nothing else in the tree did it:
//! the delayed-start path handles a SUBSET of the same mesh, the regridders
//! go to and from regular latitude/longitude grids, and the parent-native
//! boundary route samples point values onto a child (accurate, but not
//! conservative, and not a full state).
//!
//! ## The pipeline, per target column
//! 1. **Horizontal, conservative.**  The target cell's Voronoi polygon is
//!    intersected with every source polygon it overlaps ([`overlap`]).
//!    Density is averaged as layer MASS `rho dz`, mass-specific scalars as
//!    `rho dz s`, the source interface heights linearly; so the target
//!    column, still on the source's layers, holds exactly the mass and water
//!    the source holds over the same area.
//! 2. **Vertical, conservative in height.**  That column is remapped onto
//!    the target's own interfaces with a limited piecewise-linear profile
//!    ([`column`]): conservative where the two columns overlap, exact for a
//!    linear profile, extended past the source column where the target's
//!    terrain is lower or its top higher, and every extension counted.
//! 3. **Rebalance (default).**  Temperature and pressure are recovered from
//!    `(rho, theta, qv)` by the dycore's equation of state and handed to the
//!    SAME [`crate::init::dynamics::build_column`] `rw_mpas_init` builds
//!    every init column with: same base state, same hydrostatic fixed point,
//!    same metric coupling.  `theta` and `qv` come back unchanged; `rho`
//!    comes back hydrostatic on the target's own terrain.  `--balance carry`
//!    keeps the conservatively remapped `rho` instead, and the receipt then
//!    shows the mass budget closing.
//!
//! The edge wind goes through the cell-centre vector ([`vector`]); `w` is
//! point-interpolated on interfaces; diagnostics take the barycentric
//! fallback on the source's dual triangles; soil fields are masked by
//! surface type.  [`table`] is the per-variable list.
//!
//! ## What is refused
//! A target cell the source does not cover (a regional source, a target
//! reaching past it) is refused with the coverage report: how many cells,
//! the worst coverage and where.  A target model top above the source's is
//! refused (that atmosphere was never computed).  A restart-class template
//! is refused by name ([`table::RESTART_ONLY`]).  A source file with more
//! than one time record is refused rather than silently read at record 0.

#![allow(clippy::needless_range_loop)]

pub mod column;
pub mod mesh;
pub mod overlap;
pub mod table;
pub mod vector;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde::Serialize;

use crate::error::{MpasError, MpasResult};
use crate::init::dynamics::VirtualFactor;
use crate::init::emit;
use crate::lbc::sphere::CellOperator;
use column::{Extend, Extended};
use mesh::{PairReader, RemapMesh};
use overlap::Overlap;
use table::{method_for, Method};
use vector::CellVectorOperator;

/// What happens to density after the conservative remap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Balance {
    /// Rebuilt in hydrostatic balance on the target, as `rw_mpas_init` does.
    Hydrostatic,
    /// Kept as remapped: the mass budget closes, the column is not balanced.
    Carry,
}

impl Balance {
    pub fn parse(text: &str) -> Result<Balance, String> {
        match text {
            "hydrostatic" => Ok(Balance::Hydrostatic),
            "carry" => Ok(Balance::Carry),
            other => Err(format!("--balance takes hydrostatic or carry, not \"{other}\"")),
        }
    }
}

/// A target top this far above the source top is still "the same top": two
/// vertical grids built from one spec agree to rounding, not to the bit.
pub const TOP_TOLERANCE_M: f64 = 1.0;

/// Default minimum coverage a target cell must have.
pub const DEFAULT_MIN_COVERAGE: f64 = 1.0 - 1.0e-6;

// ---------------------------------------------------------------------------
// In-memory state
// ---------------------------------------------------------------------------

/// A column state on one mesh, all fields cell-major with the level fastest.
#[derive(Debug, Clone, Default)]
pub struct State {
    pub levels: usize,
    /// `[cell * (levels + 1) + k]`, metres.
    pub zgrid: Vec<f32>,
    /// `(nCells, nVertLevels)` fields by name.
    pub cell3: BTreeMap<String, Vec<f32>>,
    /// `(nCells, nVertLevelsP1)` vertical velocity.
    pub w: Option<Vec<f32>>,
    /// `(nEdges, nVertLevels)` edge-normal wind.
    pub u: Option<Vec<f32>>,
}

/// The target's vertical metrics the rebalance needs.
#[derive(Debug, Clone)]
pub struct Metrics {
    /// `[cell * levels + k]`.
    pub zz: Vec<f32>,
    pub fzm: Vec<f32>,
    pub fzp: Vec<f32>,
    pub dzu: Vec<f32>,
    pub rdzw: Vec<f32>,
}

/// The precomputed horizontal operators of one source/target pair.
pub struct Operators {
    pub overlap: Overlap,
    /// Barycentric operator per target cell (`None` where the target cell
    /// lies outside the source's triangulation and skirt).
    pub cell_bary: Vec<Option<CellOperator>>,
    /// Barycentric operator per target edge midpoint.
    pub edge_bary: Vec<Option<CellOperator>>,
    pub cell_vectors: CellVectorOperator,
}

impl Operators {
    pub fn build(source: &RemapMesh, target: &RemapMesh) -> MpasResult<Operators> {
        let overlap = overlap::build(source, target)?;
        let sampler = source.sampler()?;
        let cell_bary: Vec<Option<CellOperator>> = (0..target.n_cells)
            .into_par_iter()
            .map(|j| sampler.cell_weights(target.cell_lat[j], target.cell_lon[j]).ok())
            .collect();
        let edge_bary: Vec<Option<CellOperator>> = (0..target.n_edges)
            .into_par_iter()
            .map(|e| sampler.cell_weights(target.edge_lat[e], target.edge_lon[e]).ok())
            .collect();
        let cell_vectors = CellVectorOperator::build(source)?;
        Ok(Operators {
            overlap,
            cell_bary,
            edge_bary,
            cell_vectors,
        })
    }

    /// Normalised horizontal weights for target cell `j`: the overlap row
    /// when there is one, else the barycentric operator.
    pub fn weights(&self, j: usize) -> Option<Vec<(usize, f64)>> {
        let row = &self.overlap.rows[j];
        if !row.is_empty() {
            let s: f64 = row.iter().map(|r| r.1).sum();
            return Some(row.iter().map(|&(i, a)| (i as usize, a / s)).collect());
        }
        self.cell_bary[j].as_ref().map(bary_weights)
    }
}

fn bary_weights(op: &CellOperator) -> Vec<(usize, f64)> {
    let mut out: Vec<(usize, f64)> = Vec::with_capacity(3);
    for t in 0..3 {
        if op.weights[t] == 0.0 {
            continue;
        }
        match out.iter_mut().find(|(c, _)| *c == op.cells[t]) {
            Some(slot) => slot.1 += op.weights[t],
            None => out.push((op.cells[t], op.weights[t])),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Receipt pieces
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize)]
pub struct CoverageCell {
    pub cell: usize,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub coverage: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CoverageReport {
    pub min_required: f64,
    pub target_cells: usize,
    pub target_cells_with_polygon: usize,
    /// Target cells with no closed polygon, placed by the barycentric
    /// operator instead (first order there, and not conservative).
    pub target_cells_barycentric_fallback: usize,
    pub min_coverage: f64,
    pub mean_coverage: f64,
    pub cells_below_required: usize,
    /// Up to twenty of the least-covered cells, worst first.
    pub worst: Vec<CoverageCell>,
    pub source_cells: usize,
    pub source_cells_overlapped: usize,
    pub source_open_polygons: usize,
    pub source_area_covered_fraction: f64,
    pub overlap_pairs: usize,
}

/// Integrals over a domain.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Budget {
    pub dry_air_kg: f64,
    pub water_vapour_kg: f64,
    pub total_water_kg: f64,
    pub total_mass_kg: f64,
    pub kinetic_energy_j: f64,
    pub area_m2: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Budgets {
    /// The whole source domain.
    pub source: Budget,
    /// The source weighted by how much of each cell the target covers: what
    /// a conservative remap must reproduce.
    pub source_over_target_footprint: Budget,
    pub target: Budget,
    /// After the horizontal pass, before the vertical one: the operator's
    /// own conservation, independent of terrain.
    pub horizontal_stage_dry_air_kg: f64,
    pub horizontal_stage_water_vapour_kg: f64,
    pub horizontal_stage_dry_air_relative_change: f64,
    pub horizontal_stage_water_vapour_relative_change: f64,
    /// After the vertical pass, before any rebalance.
    pub carried_dry_air_kg: f64,
    pub carried_dry_air_relative_change: f64,
    pub dry_air_relative_change: f64,
    pub water_vapour_relative_change: f64,
    pub total_water_relative_change: f64,
    pub total_mass_relative_change: f64,
    pub kinetic_energy_relative_change: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct VariableReport {
    pub name: String,
    pub method: String,
    pub source_min: f64,
    pub source_max: f64,
    pub target_min: f64,
    pub target_max: f64,
    /// Target values outside the source's range by more than 1e-6 of it.
    pub overshoot_points: usize,
    /// Largest |conservative - barycentric| at a target point: two
    /// independent operators' disagreement, the remap's own error bar.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_abs_difference_vs_barycentric: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rms_difference_vs_barycentric: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct VerticalReport {
    pub source_levels: usize,
    pub target_levels: usize,
    pub columns_extended_below: usize,
    pub max_metres_below_source_column: f64,
    pub columns_extended_above: usize,
    pub max_metres_above_source_top: f64,
    pub balance: String,
    pub virtual_factor: String,
    /// The init writer's fixed point: at most 30 passes per level, converged
    /// at |dp| <= 1e-4 Pa.  In f32 a perturbation pressure of a few kPa has
    /// an ulp above that criterion, so columns reaching the cap are normal
    /// for `rw_mpas_init` too; zero in carry mode, where no rebalance runs.
    pub hydrostatic_iterations_max: u32,
    pub hydrostatic_cells_hitting_the_cap: usize,
    /// max |rho_rebalanced - rho_carried| / rho_carried.
    pub max_relative_rebalance_of_rho: f64,
    pub w_boundary: String,
}

#[derive(Debug, Default, Serialize)]
pub struct RemapReceipt {
    pub schema: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    pub from_grid: String,
    pub from_state: String,
    pub to_grid: String,
    pub to_static: Option<String>,
    pub to_vertical: Option<String>,
    pub template: String,
    pub out: String,
    pub source_cells: usize,
    pub source_edges: usize,
    pub target_cells: usize,
    pub target_edges: usize,
    pub coverage: CoverageReport,
    pub wind_reconstruction: Option<vector::Reconstruction>,
    pub vertical: VerticalReport,
    pub budgets: Budgets,
    pub variables: Vec<VariableReport>,
    pub absent_in_source_written_as_zero: Vec<String>,
    pub source_variables_not_remapped: Vec<String>,
    pub soil_mask: String,
    pub soil_cells_mask_fallback: usize,
    pub emit: Option<emit::EmitLedger>,
    pub seconds_operators: f64,
    pub seconds: f64,
}

// ---------------------------------------------------------------------------
// The column pass
// ---------------------------------------------------------------------------

/// What the column pass produced on the target.
pub struct ColumnResult {
    pub state: State,
    /// Derived on the target (`rho_base`, `theta_base`, `surface_pressure`,
    /// `precipw`) when metrics were given.
    pub derived3: BTreeMap<String, Vec<f32>>,
    pub derived2: BTreeMap<String, Vec<f32>>,
    pub vertical: VerticalReport,
    pub horizontal_dry_air_kg: f64,
    pub horizontal_water_vapour_kg: f64,
    pub carried_dry_air_kg: f64,
    pub spread: BTreeMap<String, (f64, f64)>,
    /// The remapped 3-D wind at every target cell, `[cell * levels + k]`.
    pub cell_vectors: Option<Vec<[f32; 3]>>,
}

/// Remap every column field of `src` onto the target.
#[allow(clippy::too_many_arguments)]
pub fn remap_columns(
    source: &RemapMesh,
    target: &RemapMesh,
    ops: &Operators,
    src: &State,
    tgt_zgrid: &[f32],
    tgt_levels: usize,
    metrics: Option<&Metrics>,
    balance: Balance,
    factor: VirtualFactor,
) -> MpasResult<ColumnResult> {
    let ns = src.levels;
    let nt = tgt_levels;
    let n_tgt = target.n_cells;
    let rho = src
        .cell3
        .get("rho")
        .ok_or_else(|| MpasError::Refusal("the source state carries no rho".to_string()))?;
    if balance == Balance::Hydrostatic && metrics.is_none() {
        return Err(MpasError::Refusal(
            "--balance hydrostatic needs the target's zz, fzm, fzp, dzu and rdzw; the target \
             template carries no complete vertical metric.  Pass a vertical artifact \
             (--to-vertical) or use --balance carry"
                .to_string(),
        ));
    }
    // Mass-specific scalars, in a fixed order.
    let scalars: Vec<(String, Extend)> = src
        .cell3
        .keys()
        .filter_map(|k| match method_for(k) {
            Some(Method::MassWeighted(how)) => Some((k.clone(), how)),
            _ => None,
        })
        .collect();
    let theta_index = scalars.iter().position(|(n, _)| n == "theta");
    let qv_index = scalars.iter().position(|(n, _)| n == "qv");
    if balance == Balance::Hydrostatic && (theta_index.is_none() || qv_index.is_none()) {
        return Err(MpasError::Refusal(
            "the hydrostatic rebalance needs theta and qv in the source state".to_string(),
        ));
    }
    let relhum = src.cell3.get("relhum");

    // Source cell vectors, [cell * ns + k].
    let src_vec: Option<Vec<[f32; 3]>> = src.u.as_ref().map(|u| {
        (0..source.n_cells)
            .into_par_iter()
            .flat_map_iter(|c| {
                (0..ns).map(move |k| {
                    let v = ops.cell_vectors.apply(c, u, ns, k);
                    [v[0] as f32, v[1] as f32, v[2] as f32]
                })
            })
            .collect()
    });

    struct Out {
        rho: Vec<f32>,
        scalars: Vec<Vec<f32>>,
        w: Option<Vec<f32>>,
        vec: Option<Vec<[f32; 3]>>,
        relhum: Option<Vec<f32>>,
        rho_base: Vec<f32>,
        theta_base: Vec<f32>,
        sp: f32,
        pw: f32,
        h_mass: f64,
        h_vap: f64,
        c_mass: f64,
        below: f64,
        above: f64,
        iters: u32,
        cap: bool,
        rebalance: f64,
        spread: Vec<(f64, f64, usize)>,
    }

    let zs = |c: usize, k: usize| src.zgrid[c * (ns + 1) + k] as f64;
    let results: Vec<MpasResult<Out>> = (0..n_tgt)
        .into_par_iter()
        .map(|j| {
            let wts = ops.weights(j).ok_or_else(|| {
                MpasError::Refusal(format!(
                    "target cell {} has neither a polygon overlap nor a barycentric placement in \
                     the source",
                    j + 1
                ))
            })?;
            // Horizontal pass, on the source's layers.
            let mut z = vec![0.0f64; ns + 1];
            let mut m = vec![0.0f64; ns];
            let mut acc = vec![vec![0.0f64; ns]; scalars.len()];
            let mut wint = src.w.as_ref().map(|_| vec![0.0f64; ns + 1]);
            let mut mom = src_vec.as_ref().map(|_| vec![[0.0f64; 3]; ns]);
            for &(i, wi) in &wts {
                for k in 0..=ns {
                    z[k] += wi * zs(i, k);
                }
                if let (Some(wint), Some(w)) = (wint.as_mut(), src.w.as_ref()) {
                    for k in 0..=ns {
                        wint[k] += wi * w[i * (ns + 1) + k] as f64;
                    }
                }
                for k in 0..ns {
                    let dz = zs(i, k + 1) - zs(i, k);
                    let mi = rho[i * ns + k] as f64 * dz;
                    m[k] += wi * mi;
                    for (s, (name, _)) in scalars.iter().enumerate() {
                        acc[s][k] += wi * mi * src.cell3[name][i * ns + k] as f64;
                    }
                    if let (Some(mom), Some(v)) = (mom.as_mut(), src_vec.as_ref()) {
                        let vv = v[i * ns + k];
                        for d in 0..3 {
                            mom[k][d] += wi * mi * vv[d] as f64;
                        }
                    }
                }
            }
            let area = target.area_m2(j);
            let h_mass = m.iter().sum::<f64>() * area;
            let mut h_vap = 0.0;
            let rho_bar: Vec<f64> = (0..ns).map(|k| m[k] / (z[k + 1] - z[k])).collect();
            let svals: Vec<Vec<f64>> = (0..scalars.len())
                .map(|s| (0..ns).map(|k| if m[k] > 0.0 { acc[s][k] / m[k] } else { 0.0 }).collect())
                .collect();
            if let Some(q) = qv_index {
                h_vap = (0..ns).map(|k| acc[q][k]).sum::<f64>() * area;
            }

            // Vertical pass.
            let tz: Vec<f64> = (0..=nt).map(|k| tgt_zgrid[j * (nt + 1) + k] as f64).collect();
            let ext = Extended::new(&z, &tz);
            let (mass_t, prof) = column::remap_density(&ext, &rho_bar, &tz);
            let dzt: Vec<f64> = (0..nt).map(|k| tz[k + 1] - tz[k]).collect();
            let rho_t: Vec<f64> = (0..nt).map(|k| mass_t[k] / dzt[k]).collect();
            let c_mass = mass_t.iter().sum::<f64>() * area;
            let svals_t: Vec<Vec<f64>> = scalars
                .iter()
                .enumerate()
                .map(|(s, (_, how))| column::remap_scalar(&ext, &prof, &svals[s], &tz, &mass_t, *how))
                .collect();
            let vec_t: Option<Vec<[f32; 3]>> = mom.as_ref().map(|mom| {
                let comps: Vec<Vec<f64>> = (0..3)
                    .map(|d| {
                        let v: Vec<f64> = (0..ns)
                            .map(|k| if m[k] > 0.0 { mom[k][d] / m[k] } else { 0.0 })
                            .collect();
                        column::remap_scalar(&ext, &prof, &v, &tz, &mass_t, Extend::Constant)
                    })
                    .collect();
                (0..nt)
                    .map(|k| [comps[0][k] as f32, comps[1][k] as f32, comps[2][k] as f32])
                    .collect()
            });
            let w_t = wint.as_ref().map(|wint| {
                let mut w = column::interp_points(&z, wint, &tz);
                w[0] = 0.0;
                w[nt] = 0.0;
                w.into_iter().map(|v| v as f32).collect::<Vec<f32>>()
            });

            // Barycentric companions: relhum, and the error bar on theta/qv.
            let tmid: Vec<f64> = (0..nt).map(|k| 0.5 * (tz[k] + tz[k + 1])).collect();
            let bary_column = |field: &[f32], op: &CellOperator| -> Vec<f64> {
                let zmid: Vec<f64> = (0..ns)
                    .map(|k| {
                        (0..3)
                            .map(|t| op.weights[t] * 0.5 * (zs(op.cells[t], k) + zs(op.cells[t], k + 1)))
                            .sum()
                    })
                    .collect();
                let v: Vec<f64> = (0..ns)
                    .map(|k| (0..3).map(|t| op.weights[t] * field[op.cells[t] * ns + k] as f64).sum())
                    .collect();
                column::interp_points(&zmid, &v, &tmid)
            };
            let bary = ops.cell_bary[j].as_ref();
            let relhum_t = match (relhum, bary) {
                (Some(rh), Some(op)) => Some(bary_column(rh, op).into_iter().map(|v| v as f32).collect()),
                (Some(_), None) => {
                    return Err(MpasError::Refusal(format!(
                        "target cell {} lies outside the source's dual triangulation, so the \
                         barycentric diagnostics (relhum) cannot be placed there",
                        j + 1
                    )))
                }
                _ => None,
            };
            let mut spread: Vec<(f64, f64, usize)> = Vec::new();
            if let Some(op) = bary {
                for (s, (name, _)) in scalars.iter().enumerate() {
                    if name != "theta" && name != "qv" {
                        continue;
                    }
                    let b = bary_column(&src.cell3[name], op);
                    let mut mx = 0.0f64;
                    let mut sq = 0.0f64;
                    for k in 0..nt {
                        let d = (svals_t[s][k] - b[k]).abs();
                        mx = mx.max(d);
                        sq += d * d;
                    }
                    spread.push((mx, sq, nt));
                }
            }

            // Rebalance and the base state.
            let mut out_rho: Vec<f32> = rho_t.iter().map(|&v| v as f32).collect();
            let mut out_scalars: Vec<Vec<f32>> = svals_t
                .iter()
                .map(|col| col.iter().map(|&v| v as f32).collect())
                .collect();
            let (mut rho_base, mut theta_base) = (Vec::new(), Vec::new());
            let (mut sp, mut pw, mut iters, mut cap, mut reb) = (0.0f32, 0.0f32, 0u32, false, 0.0f64);
            if let (Some(mt), Some(ti), Some(qi)) = (metrics, theta_index, qv_index) {
                let zg = &tgt_zgrid[j * (nt + 1)..(j + 1) * (nt + 1)];
                let zz = &mt.zz[j * nt..(j + 1) * nt];
                let st = column::rebalance(
                    &rho_t,
                    &svals_t[ti],
                    &svals_t[qi],
                    zg,
                    zz,
                    &mt.fzm,
                    &mt.fzp,
                    &mt.dzu,
                    mt.rdzw[0],
                    factor,
                );
                rho_base = st.rho_base.clone();
                theta_base = st.theta_base.clone();
                if balance == Balance::Hydrostatic {
                    iters = st.hydrostatic_iterations.iter().copied().max().unwrap_or(0);
                    cap = st.hydrostatic_iterations.iter().any(|&i| i >= 30);
                    for k in 0..nt {
                        reb = reb.max(((st.rho[k] as f64 - rho_t[k]) / rho_t[k]).abs());
                    }
                    out_rho = st.rho.clone();
                    out_scalars[ti] = st.theta.clone();
                    sp = st.surface_pressure;
                    pw = st.precipw;
                } else {
                    pw = (0..nt).map(|k| rho_t[k] * svals_t[qi][k] * dzt[k]).sum::<f64>() as f32;
                }
            }

            Ok(Out {
                rho: out_rho,
                scalars: out_scalars,
                w: w_t,
                vec: vec_t,
                relhum: relhum_t,
                rho_base,
                theta_base,
                sp,
                pw,
                h_mass,
                h_vap,
                c_mass,
                below: ext.below_m,
                above: ext.above_m,
                iters,
                cap,
                rebalance: reb,
                spread,
            })
        })
        .collect();

    let mut outs = Vec::with_capacity(n_tgt);
    for r in results {
        outs.push(r?);
    }

    // Above-top refusal, with the worst column named.
    let (worst_above, worst_cell) = outs
        .iter()
        .enumerate()
        .map(|(j, o)| (o.above, j))
        .fold((0.0f64, 0usize), |a, b| if b.0 > a.0 { b } else { a });
    if worst_above > TOP_TOLERANCE_M {
        let n_above = outs.iter().filter(|o| o.above > TOP_TOLERANCE_M).count();
        return Err(MpasError::Refusal(format!(
            "{n_above} target column(s) reach above the source's model top; the worst is target \
             cell {} by {worst_above:.1} m.  Above the source's top there is no source \
             atmosphere, so those layers would be invented by extension.  Give the target a \
             model top at or below the source's",
            worst_cell + 1
        )));
    }

    let mut state = State {
        levels: nt,
        zgrid: tgt_zgrid.to_vec(),
        ..Default::default()
    };
    state.cell3.insert("rho".to_string(), outs.iter().flat_map(|o| o.rho.iter().copied()).collect());
    for (s, (name, _)) in scalars.iter().enumerate() {
        state
            .cell3
            .insert(name.clone(), outs.iter().flat_map(|o| o.scalars[s].iter().copied()).collect());
    }
    if relhum.is_some() {
        state.cell3.insert(
            "relhum".to_string(),
            outs.iter().flat_map(|o| o.relhum.clone().unwrap_or_default()).collect(),
        );
    }
    if src.w.is_some() {
        state.w = Some(outs.iter().flat_map(|o| o.w.clone().unwrap_or_default()).collect());
    }
    let mut cell_vectors = None;
    if src_vec.is_some() {
        let vectors: Vec<[f32; 3]> = outs.iter().flat_map(|o| o.vec.clone().unwrap_or_default()).collect();
        let u: Vec<f32> = (0..target.n_edges)
            .into_par_iter()
            .flat_map_iter(|e| {
                let cells = target.edge_cells(e);
                let vectors = &vectors;
                (0..nt).map(move |k| vector::project(target, e, &cells, vectors, nt, k) as f32)
            })
            .collect();
        state.u = Some(u);
        cell_vectors = Some(vectors);
    }

    let mut derived3 = BTreeMap::new();
    let mut derived2 = BTreeMap::new();
    if metrics.is_some() && theta_index.is_some() && qv_index.is_some() {
        derived3.insert("rho_base".to_string(), outs.iter().flat_map(|o| o.rho_base.iter().copied()).collect());
        derived3.insert("theta_base".to_string(), outs.iter().flat_map(|o| o.theta_base.iter().copied()).collect());
        derived2.insert("precipw".to_string(), outs.iter().map(|o| o.pw).collect());
        if balance == Balance::Hydrostatic {
            derived2.insert("surface_pressure".to_string(), outs.iter().map(|o| o.sp).collect());
        }
    }

    let mut spread: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    let names: Vec<&str> = scalars
        .iter()
        .map(|(n, _)| n.as_str())
        .filter(|n| *n == "theta" || *n == "qv")
        .collect();
    for (idx, name) in names.iter().enumerate() {
        let mut mx = 0.0f64;
        let mut sq = 0.0f64;
        let mut n = 0usize;
        for o in &outs {
            if let Some(&(m, s, c)) = o.spread.get(idx) {
                mx = mx.max(m);
                sq += s;
                n += c;
            }
        }
        if n > 0 {
            spread.insert(name.to_string(), (mx, (sq / n as f64).sqrt()));
        }
    }

    let vertical = VerticalReport {
        source_levels: ns,
        target_levels: nt,
        columns_extended_below: outs.iter().filter(|o| o.below > 0.0).count(),
        max_metres_below_source_column: outs.iter().map(|o| o.below).fold(0.0, f64::max),
        columns_extended_above: outs.iter().filter(|o| o.above > 0.0).count(),
        max_metres_above_source_top: worst_above,
        balance: match balance {
            Balance::Hydrostatic => "hydrostatic".to_string(),
            Balance::Carry => "carry".to_string(),
        },
        virtual_factor: match factor {
            VirtualFactor::ReproduceFortran => "reproduce-fortran".to_string(),
            VirtualFactor::Consistent => "consistent".to_string(),
        },
        hydrostatic_iterations_max: outs.iter().map(|o| o.iters).max().unwrap_or(0),
        hydrostatic_cells_hitting_the_cap: outs.iter().filter(|o| o.cap).count(),
        max_relative_rebalance_of_rho: outs.iter().map(|o| o.rebalance).fold(0.0, f64::max),
        w_boundary: "zero at the surface and the lid, as rw_mpas_init writes it".to_string(),
    };

    Ok(ColumnResult {
        state,
        derived3,
        derived2,
        vertical,
        horizontal_dry_air_kg: outs.iter().map(|o| o.h_mass).sum(),
        horizontal_water_vapour_kg: outs.iter().map(|o| o.h_vap).sum(),
        carried_dry_air_kg: outs.iter().map(|o| o.c_mass).sum(),
        spread,
        cell_vectors,
    })
}

/// The edge-wind error bar: the projected `u` against a barycentric sample
/// of the source cell vectors at each target edge midpoint, in height.
pub fn wind_spread(
    source: &RemapMesh,
    target: &RemapMesh,
    ops: &Operators,
    src: &State,
    tgt: &State,
) -> Option<(f64, f64)> {
    let u = src.u.as_ref()?;
    let tu = tgt.u.as_ref()?;
    let ns = src.levels;
    let nt = tgt.levels;
    let zs = |c: usize, k: usize| src.zgrid[c * (ns + 1) + k] as f64;
    let zt = |c: usize, k: usize| tgt.zgrid[c * (nt + 1) + k] as f64;
    let per_edge: Vec<(f64, f64, usize)> = (0..target.n_edges)
        .into_par_iter()
        .filter_map(|e| {
            let op = ops.edge_bary[e].as_ref()?;
            let cells = target.edge_cells(e);
            if cells.is_empty() {
                return None;
            }
            let zmid: Vec<f64> = (0..ns)
                .map(|k| (0..3).map(|t| op.weights[t] * 0.5 * (zs(op.cells[t], k) + zs(op.cells[t], k + 1))).sum())
                .collect();
            let n = target.edge_normal[e];
            let un: Vec<f64> = (0..ns)
                .map(|k| {
                    let mut v = [0.0f64; 3];
                    for t in 0..3 {
                        let vv = ops.cell_vectors.apply(op.cells[t], u, ns, k);
                        for d in 0..3 {
                            v[d] += op.weights[t] * vv[d];
                        }
                    }
                    crate::mesh::geom::dot(n, v)
                })
                .collect();
            let tmid: Vec<f64> = (0..nt)
                .map(|k| {
                    cells.iter().map(|&c| 0.5 * (zt(c, k) + zt(c, k + 1))).sum::<f64>() / cells.len() as f64
                })
                .collect();
            let b = column::interp_points(&zmid, &un, &tmid);
            let mut mx = 0.0f64;
            let mut sq = 0.0f64;
            for k in 0..nt {
                let d = (tu[e * nt + k] as f64 - b[k]).abs();
                mx = mx.max(d);
                sq += d * d;
            }
            Some((mx, sq, nt))
        })
        .collect();
    let n: usize = per_edge.iter().map(|p| p.2).sum();
    if n == 0 {
        return None;
    }
    let mx = per_edge.iter().map(|p| p.0).fold(0.0, f64::max);
    let sq: f64 = per_edge.iter().map(|p| p.1).sum();
    let _ = source;
    Some((mx, (sq / n as f64).sqrt()))
}

/// Integrate the budgets of a state, each cell weighted by `fraction(c)`.
pub fn budget(mesh: &RemapMesh, st: &State, fraction: &(dyn Fn(usize) -> f64 + Sync)) -> Budget {
    let n = st.levels;
    let rho = match st.cell3.get("rho") {
        Some(r) => r,
        None => return Budget::default(),
    };
    let species: Vec<(&str, &Vec<f32>)> = table::WATER_SPECIES
        .iter()
        .filter_map(|s| st.cell3.get(*s).map(|v| (*s, v)))
        .collect();
    let parts: Vec<(f64, f64, f64, f64, f64)> = (0..mesh.n_cells)
        .into_par_iter()
        .map(|c| {
            let f = fraction(c);
            if f == 0.0 {
                return (0.0, 0.0, 0.0, 0.0, 0.0);
            }
            let area = mesh.area_m2(c);
            let (mut dry, mut vap, mut tot, mut ke) = (0.0, 0.0, 0.0, 0.0);
            for k in 0..n {
                let dz = (st.zgrid[c * (n + 1) + k + 1] - st.zgrid[c * (n + 1) + k]) as f64;
                let m = rho[c * n + k] as f64 * dz;
                dry += m * area;
                for (name, v) in &species {
                    let q = v[c * n + k] as f64 * m * area;
                    tot += q;
                    if *name == "qv" {
                        vap += q;
                    }
                }
                if let Some(u) = &st.u {
                    let mut s = 0.0;
                    for i in 0..mesh.n_edges_on_cell[c] {
                        let e = mesh.edges_on_cell[c][i];
                        if e >= 1 && e <= mesh.n_edges {
                            let e = e - 1;
                            let ue = u[e * n + k] as f64;
                            s += 0.25 * mesh.dc_edge_m[e] * mesh.dv_edge_m[e] * ue * ue;
                        }
                    }
                    ke += m * s;
                }
            }
            (dry * f, vap * f, tot * f, ke * f, area * f)
        })
        .collect();
    let mut b = Budget::default();
    for p in parts {
        b.dry_air_kg += p.0;
        b.water_vapour_kg += p.1;
        b.total_water_kg += p.2;
        b.kinetic_energy_j += p.3;
        b.area_m2 += p.4;
    }
    b.total_mass_kg = b.dry_air_kg + b.total_water_kg;
    b
}

fn rel(after: f64, before: f64) -> f64 {
    if before == 0.0 {
        if after == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (after - before) / before
    }
}

/// The coverage report, and whether it passes.
pub fn coverage_report(source: &RemapMesh, target: &RemapMesh, ops: &Operators, min_required: f64) -> CoverageReport {
    let ov = &ops.overlap;
    let with_poly: Vec<usize> = (0..target.n_cells).filter(|&j| target.polygons[j].is_some()).collect();
    let mut worst: Vec<(f64, usize)> = with_poly.iter().map(|&j| (ov.coverage[j], j)).collect();
    worst.sort_by(|a, b| a.0.total_cmp(&b.0));
    let min_coverage = worst.first().map(|w| w.0).unwrap_or(f64::NAN);
    let mean_coverage = if with_poly.is_empty() {
        f64::NAN
    } else {
        worst.iter().map(|w| w.0).sum::<f64>() / with_poly.len() as f64
    };
    let fallback_unplaced = (0..target.n_cells)
        .filter(|&j| target.polygons[j].is_none() && ops.cell_bary[j].is_none())
        .count();
    let below = worst.iter().filter(|w| !(w.0 >= min_required)).count() + fallback_unplaced;
    let covered_area: f64 = ov.source_covered.iter().sum();
    let src_area: f64 = source.polygon_area.iter().sum();
    CoverageReport {
        min_required,
        target_cells: target.n_cells,
        target_cells_with_polygon: with_poly.len(),
        target_cells_barycentric_fallback: target.n_cells - with_poly.len(),
        min_coverage,
        mean_coverage,
        cells_below_required: below,
        worst: worst
            .iter()
            .take(20)
            .filter(|w| !(w.0 >= min_required))
            .map(|&(c, j)| CoverageCell {
                cell: j + 1,
                lat_deg: target.cell_lat[j].to_degrees(),
                lon_deg: target.cell_lon[j].to_degrees(),
                coverage: c,
            })
            .collect(),
        source_cells: source.n_cells,
        source_cells_overlapped: ov.source_covered.iter().filter(|&&a| a > 0.0).count(),
        source_open_polygons: source.open_polygons(),
        source_area_covered_fraction: covered_area / src_area,
        overlap_pairs: ov.rows.iter().map(|r| r.len()).sum(),
    }
}

// ---------------------------------------------------------------------------
// Surface and soil fields
// ---------------------------------------------------------------------------

/// Remap a `(nCells, width)` surface field by `method`.
pub fn remap_surface(
    ops: &Operators,
    target_cells: usize,
    values: &[f32],
    width: usize,
    method: Method,
    src_class: Option<&[i32]>,
    tgt_class: Option<&[i32]>,
    mask_fallbacks: &mut usize,
) -> MpasResult<Vec<f32>> {
    let mut out = vec![0.0f32; target_cells * width];
    let mut fallbacks = 0usize;
    for j in 0..target_cells {
        let place = |j: usize| {
            ops.weights(j).ok_or_else(|| {
                MpasError::Refusal(format!("target cell {} cannot be placed in the source", j + 1))
            })
        };
        let row: Vec<(usize, f64)> = match method {
            Method::Barycentric => match ops.cell_bary[j].as_ref() {
                Some(op) => bary_weights(op),
                None => {
                    return Err(MpasError::Refusal(format!(
                        "target cell {} lies outside the source's dual triangulation; a \
                         barycentric diagnostic cannot be placed there",
                        j + 1
                    )))
                }
            },
            Method::Dominant => {
                let w = place(j)?;
                let best = w
                    .iter()
                    .copied()
                    .fold((usize::MAX, -1.0f64), |a, b| if b.1 > a.1 { b } else { a });
                vec![(best.0, 1.0)]
            }
            Method::MaskedConservative => {
                let w = place(j)?;
                match (src_class, tgt_class) {
                    (Some(sc), Some(tc)) => {
                        let want = tc[j];
                        let same: Vec<(usize, f64)> =
                            w.iter().copied().filter(|&(i, _)| sc[i] == want).collect();
                        let s: f64 = same.iter().map(|r| r.1).sum();
                        if s > 0.0 {
                            same.into_iter().map(|(i, a)| (i, a / s)).collect()
                        } else {
                            fallbacks += 1;
                            w
                        }
                    }
                    _ => w,
                }
            }
            _ => place(j)?,
        };
        for l in 0..width {
            let mut v = 0.0f64;
            for &(i, wi) in &row {
                v += wi * values[i * width + l] as f64;
            }
            out[j * width + l] = v as f32;
        }
    }
    *mask_fallbacks += fallbacks;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

/// Everything the caller states.
#[derive(Debug, Clone)]
pub struct RemapConfig {
    pub from_grid: PathBuf,
    pub from_state: PathBuf,
    pub to_grid: PathBuf,
    pub to_static: Option<PathBuf>,
    pub to_vertical: Option<PathBuf>,
    pub out: PathBuf,
    pub balance: Balance,
    pub virtual_factor: VirtualFactor,
    pub min_coverage: f64,
    pub provenance: String,
}

/// One record of a single-time file.
struct StateFile {
    file: netcrust::File,
    path: PathBuf,
}

impl StateFile {
    fn open(path: &Path) -> MpasResult<StateFile> {
        if !path.exists() {
            return Err(MpasError::Refusal(format!("{} does not exist", path.display())));
        }
        let file = netcrust::File::open(path)?;
        if let Some(t) = file.dimension("Time") {
            if t.len() > 1 {
                return Err(MpasError::Refusal(format!(
                    "{} holds {} time records; the remap reads one state, and reading record 0 \
                     of a series silently would remap the wrong hour.  Split the frame out first",
                    path.display(),
                    t.len()
                )));
            }
        }
        Ok(StateFile {
            file,
            path: path.to_path_buf(),
        })
    }

    fn has(&self, name: &str) -> bool {
        self.file.variable(name).is_some()
    }

    fn dims(&self, name: &str) -> Vec<String> {
        self.file
            .variable(name)
            .map(|v| v.dimensions().iter().map(|d| d.name().to_string()).collect())
            .unwrap_or_default()
    }

    fn f32s(&self, name: &str, expect: usize) -> MpasResult<Vec<f32>> {
        let v = self.file.read_array_f64_first_record_or_all(name)?.into_values();
        if v.len() != expect {
            return Err(MpasError::Refusal(format!(
                "{name} in {} holds {} value(s); expected {expect}",
                self.path.display(),
                v.len()
            )));
        }
        Ok(v.into_iter().map(|x| x as f32).collect())
    }
}

fn read_text(path: &Path, name: &str) -> Option<String> {
    if let Ok(f) = netcrust::File::open(path) {
        if let Ok(v) = f.read_strings(name) {
            let joined: String = v.join("");
            let t = joined.trim_end_matches(['\0', ' ']).to_string();
            if !t.is_empty() {
                return Some(t);
            }
        }
    }
    crate::lbc::compare::read_char_variable(path, name)
        .ok()
        .flatten()
        .map(|t| t.trim_end_matches(['\0', ' ']).to_string())
        .filter(|t| !t.is_empty())
}

fn read_class(file: &PairReader, n: usize) -> Option<Vec<i32>> {
    if let Some(v) = file.f64s_opt("landmask") {
        if v.len() == n {
            return Some(v.into_iter().map(|x| x.round() as i32).collect());
        }
    }
    if let Some(v) = file.f64s_opt("xland") {
        if v.len() == n && v.iter().any(|x| *x != 0.0) {
            return Some(v.into_iter().map(|x| if x.round() as i32 == 1 { 1 } else { 0 }).collect());
        }
    }
    None
}

/// Run a whole remap from files.
pub fn run(cfg: &RemapConfig) -> MpasResult<RemapReceipt> {
    let started = std::time::Instant::now();
    let template = cfg.to_vertical.clone().unwrap_or_else(|| cfg.to_grid.clone());
    let mut receipt = RemapReceipt {
        schema: "rw-mpas.remap-receipt/v1".to_string(),
        status: "running".to_string(),
        from_grid: cfg.from_grid.display().to_string(),
        from_state: cfg.from_state.display().to_string(),
        to_grid: cfg.to_grid.display().to_string(),
        to_static: cfg.to_static.as_ref().map(|p| p.display().to_string()),
        to_vertical: cfg.to_vertical.as_ref().map(|p| p.display().to_string()),
        template: template.display().to_string(),
        out: cfg.out.display().to_string(),
        ..Default::default()
    };
    if !(cfg.min_coverage > 0.0 && cfg.min_coverage <= 1.0) {
        return Err(MpasError::Refusal(format!(
            "--min-coverage {} is outside (0, 1]",
            cfg.min_coverage
        )));
    }
    if cfg.out == cfg.from_state || cfg.out == template || Some(&cfg.out) == cfg.to_static.as_ref() {
        return Err(MpasError::Refusal(
            "the output path names one of the inputs; the remap never overwrites its own source \
             or template"
                .to_string(),
        ));
    }

    // Meshes.
    let source = RemapMesh::read(&cfg.from_grid, Some(&cfg.from_state))?;
    let target = RemapMesh::read(&cfg.to_grid, None)?;
    receipt.source_cells = source.n_cells;
    receipt.source_edges = source.n_edges;
    receipt.target_cells = target.n_cells;
    receipt.target_edges = target.n_edges;

    // The template and the static must describe the target mesh.
    let tfile = StateFile::open(&template)?;
    for (what, path) in std::iter::once(("template", Some(&template))).chain(std::iter::once(("--to-static", cfg.to_static.as_ref()))) {
        let Some(path) = path else { continue };
        let f = netcrust::File::open(path)?;
        let nc = f.dimension("nCells").map(|d| d.len()).unwrap_or(0);
        let ne = f.dimension("nEdges").map(|d| d.len()).unwrap_or(0);
        if nc != target.n_cells || ne != target.n_edges {
            return Err(MpasError::Refusal(format!(
                "{what} {} declares {nc} cells and {ne} edges; --to-grid declares {} and {}.  \
                 They are different meshes, and every remapped value would land on the wrong cell",
                path.display(),
                target.n_cells,
                target.n_edges
            )));
        }
        if let Ok(lat) = f.read_array_f64_first_record_or_all("latCell") {
            let worst = lat
                .into_values()
                .iter()
                .zip(target.cell_lat.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            if worst > 1.0e-6 {
                return Err(MpasError::Refusal(format!(
                    "{what} {}'s latCell differs from --to-grid's by up to {worst:.3e} rad; the \
                     same cell count on a different mesh",
                    path.display()
                )));
            }
        }
    }
    let restart_slots: Vec<&str> = table::RESTART_ONLY.iter().copied().filter(|n| tfile.has(n)).collect();
    if !restart_slots.is_empty() {
        return Err(MpasError::Refusal(format!(
            "the target template {} declares the dycore working state {:?}; it is a restart-class \
             file, and the remap writes init-class state only (the forecast rebuilds those from \
             theta, rho, qv, u and w at start-up).  Pass an init or a vertical artifact as the \
             template",
            template.display(),
            restart_slots
        )));
    }

    // Vertical columns.
    let sfile = StateFile::open(&cfg.from_state)?;
    let src_reader = PairReader::open(&cfg.from_grid, Some(&cfg.from_state))?;
    let ns = src_reader.dim("nVertLevels")?;
    let src_zgrid: Vec<f32> = src_reader
        .f64s("zgrid", source.n_cells * (ns + 1))?
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let nt = tfile.file.dimension("nVertLevels").map(|d| d.len()).ok_or_else(|| {
        MpasError::Refusal(format!(
            "the target template {} declares no nVertLevels: it carries no vertical grid.  Build \
             one with `woof hex vertical --grid {} --static STATIC --vertical-spec SPEC -o \
             B.vertical.nc` and pass it as --to-vertical",
            template.display(),
            cfg.to_grid.display()
        ))
    })?;
    if !tfile.has("zgrid") {
        return Err(MpasError::Refusal(format!(
            "the target template {} carries no zgrid; the target has no vertical grid to remap \
             onto.  Build one with `woof hex vertical` and pass it as --to-vertical",
            template.display()
        )));
    }
    let tgt_zgrid = tfile.f32s("zgrid", target.n_cells * (nt + 1))?;
    if tgt_zgrid.iter().all(|&z| z == 0.0) {
        return Err(MpasError::Refusal(format!(
            "the target template {}'s zgrid is all zero: a declared slot nobody filled",
            template.display()
        )));
    }
    for (name, z, n, cells) in [("source", &src_zgrid, ns, source.n_cells), ("target", &tgt_zgrid, nt, target.n_cells)] {
        for c in 0..cells {
            for k in 0..n {
                if !(z[c * (n + 1) + k + 1] > z[c * (n + 1) + k]) {
                    return Err(MpasError::Refusal(format!(
                        "the {name} zgrid does not ascend at cell {} between interfaces {k} and {}; \
                         a column that folds back on itself cannot be remapped",
                        c + 1,
                        k + 1
                    )));
                }
            }
        }
    }
    let metrics = (|| -> Option<Metrics> {
        let zz = tfile.f32s("zz", target.n_cells * nt).ok()?;
        let fzm = tfile.f32s("fzm", nt).ok()?;
        let fzp = tfile.f32s("fzp", nt).ok()?;
        let dzu = tfile.f32s("dzu", nt).ok()?;
        let rdzw = tfile.f32s("rdzw", nt).ok()?;
        if zz.iter().all(|&v| v == 0.0) || rdzw[0] == 0.0 {
            return None;
        }
        Some(Metrics { zz, fzm, fzp, dzu, rdzw })
    })();

    // The source state.
    let mut src = State {
        levels: ns,
        zgrid: src_zgrid,
        ..Default::default()
    };
    let mut not_remapped: Vec<String> = Vec::new();
    for v in sfile.file.variables()? {
        let name = v.name().to_string();
        let dims: Vec<String> = v.dimensions().iter().map(|d| d.name().to_string()).collect();
        let has_time = dims.first().map(|d| d == "Time").unwrap_or(false);
        let tail: Vec<&str> = dims.iter().skip(usize::from(has_time)).map(String::as_str).collect();
        match (method_for(&name), tail.as_slice()) {
            (Some(Method::Density | Method::MassWeighted(_) | Method::BarycentricColumn), ["nCells", "nVertLevels"]) => {
                src.cell3.insert(name.clone(), sfile.f32s(&name, source.n_cells * ns)?);
            }
            (Some(Method::Interface), ["nCells", "nVertLevelsP1"]) => {
                src.w = Some(sfile.f32s(&name, source.n_cells * (ns + 1))?);
            }
            (Some(Method::EdgeNormalWind), ["nEdges", "nVertLevels"]) => {
                src.u = Some(sfile.f32s(&name, source.n_edges * ns)?);
            }
            (None, ["nCells", "nVertLevels"] | ["nEdges", "nVertLevels"] | ["nCells", "nVertLevelsP1"])
                if has_time && name != "normal_u" =>
            {
                not_remapped.push(name);
            }
            _ => {}
        }
    }
    // The port's own history spells the edge wind `normal_u`.
    if src.u.is_none() && sfile.has("normal_u") {
        src.u = Some(sfile.f32s("normal_u", source.n_edges * ns)?);
    }
    for need in ["rho", "theta", "qv"] {
        if !src.cell3.contains_key(need) {
            return Err(MpasError::Refusal(format!(
                "the source state {} carries no {need}; the remap needs rho, theta and qv at the \
                 least, and inventing one would write a state nothing computed",
                cfg.from_state.display()
            )));
        }
    }
    receipt.source_variables_not_remapped = not_remapped;

    // Operators and coverage.
    let t_ops = std::time::Instant::now();
    let ops = Operators::build(&source, &target)?;
    receipt.seconds_operators = t_ops.elapsed().as_secs_f64();
    receipt.wind_reconstruction = src.u.as_ref().map(|_| ops.cell_vectors.method);
    receipt.coverage = coverage_report(&source, &target, &ops, cfg.min_coverage);
    if receipt.coverage.cells_below_required > 0 {
        let c = &receipt.coverage;
        let worst = c.worst.first().cloned().unwrap_or_default();
        receipt.status = "refused".to_string();
        receipt.refusal = Some(format!(
            "{} of {} target cell(s) are not covered by the source to the required {:.6}; the \
             least covered is target cell {} at {:.4}N {:.4}E with coverage {:.6}.  A regional \
             source must contain the whole target, or the uncovered cells would be filled by \
             extrapolating a state the source never held",
            c.cells_below_required, c.target_cells, c.min_required, worst.cell, worst.lat_deg,
            worst.lon_deg, worst.coverage
        ));
        receipt.seconds = started.elapsed().as_secs_f64();
        return Ok(receipt);
    }

    // Columns.
    let cols = remap_columns(
        &source,
        &target,
        &ops,
        &src,
        &tgt_zgrid,
        nt,
        metrics.as_ref(),
        cfg.balance,
        cfg.virtual_factor,
    )?;
    receipt.vertical = cols.vertical.clone();

    // Budgets.
    let src_budget = budget(&source, &src, &|_| 1.0);
    let foot = budget(&source, &src, &|c| {
        if source.polygon_area[c] > 0.0 {
            (ops.overlap.source_covered[c] / source.polygon_area[c]).min(1.0)
        } else {
            0.0
        }
    });
    let tgt_budget = budget(&target, &cols.state, &|_| 1.0);
    receipt.budgets = Budgets {
        horizontal_stage_dry_air_kg: cols.horizontal_dry_air_kg,
        horizontal_stage_water_vapour_kg: cols.horizontal_water_vapour_kg,
        horizontal_stage_dry_air_relative_change: rel(cols.horizontal_dry_air_kg, foot.dry_air_kg),
        horizontal_stage_water_vapour_relative_change: rel(cols.horizontal_water_vapour_kg, foot.water_vapour_kg),
        carried_dry_air_kg: cols.carried_dry_air_kg,
        carried_dry_air_relative_change: rel(cols.carried_dry_air_kg, foot.dry_air_kg),
        dry_air_relative_change: rel(tgt_budget.dry_air_kg, foot.dry_air_kg),
        water_vapour_relative_change: rel(tgt_budget.water_vapour_kg, foot.water_vapour_kg),
        total_water_relative_change: rel(tgt_budget.total_water_kg, foot.total_water_kg),
        total_mass_relative_change: rel(tgt_budget.total_mass_kg, foot.total_mass_kg),
        kinetic_energy_relative_change: rel(tgt_budget.kinetic_energy_j, foot.kinetic_energy_j),
        source: src_budget,
        source_over_target_footprint: foot,
        target: tgt_budget,
    };

    // Variable reports for the column fields.
    let range = |v: &[f32]| v.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, &x| (a.0.min(x as f64), a.1.max(x as f64)));
    let mut variables: Vec<VariableReport> = Vec::new();
    let mut report = |name: &str, method: &str, s: &[f32], t: &[f32], spread: Option<(f64, f64)>| {
        let (smin, smax) = range(s);
        let (tmin, tmax) = range(t);
        let tol = 1.0e-6 * (smax - smin).abs().max(smax.abs()).max(1e-30);
        variables.push(VariableReport {
            name: name.to_string(),
            method: method.to_string(),
            source_min: smin,
            source_max: smax,
            target_min: tmin,
            target_max: tmax,
            overshoot_points: t.iter().filter(|&&x| (x as f64) < smin - tol || (x as f64) > smax + tol).count(),
            max_abs_difference_vs_barycentric: spread.map(|s| s.0),
            rms_difference_vs_barycentric: spread.map(|s| s.1),
        });
    };
    for (name, s) in &src.cell3 {
        if let (Some(m), Some(t)) = (method_for(name), cols.state.cell3.get(name)) {
            report(name, m.label(), s, t, cols.spread.get(name).copied());
        }
    }
    if let (Some(s), Some(t)) = (&src.w, &cols.state.w) {
        report("w", Method::Interface.label(), s, t, None);
    }
    let wspread = wind_spread(&source, &target, &ops, &src, &cols.state);
    if let (Some(s), Some(t)) = (&src.u, &cols.state.u) {
        report("u", Method::EdgeNormalWind.label(), s, t, wspread);
    }

    // Surface and soil fields.
    let tgt_reader = PairReader::open(&template, cfg.to_static.as_deref())?;
    let src_class = read_class(&src_reader, source.n_cells);
    let tgt_class = read_class(&tgt_reader, target.n_cells);
    receipt.soil_mask = match (&src_class, &tgt_class) {
        (Some(_), Some(_)) => "landmask on both sides".to_string(),
        _ => "unmasked: no landmask on one side".to_string(),
    };
    let mut computed: BTreeMap<String, emit::Computed> = BTreeMap::new();
    let mut absent: Vec<String> = Vec::new();
    let mut fallbacks = 0usize;
    for v in tfile.file.variables()? {
        let name = v.name().to_string();
        let Some(method) = method_for(&name) else { continue };
        let dims = tfile.dims(&name);
        let has_time = dims.first().map(|d| d == "Time").unwrap_or(false);
        let tail: Vec<&str> = dims.iter().skip(usize::from(has_time)).map(String::as_str).collect();
        let width = match tail.as_slice() {
            ["nCells"] => 1,
            ["nCells", w] => tfile.file.dimension(w).map(|d| d.len()).unwrap_or(0),
            ["nEdges", _] => 0,
            _ => 0,
        };
        let floats = |v: Vec<f32>| emit::Computed::Floats(v);
        match method {
            Method::Label => {}
            Method::Density | Method::MassWeighted(_) | Method::BarycentricColumn => {
                match cols.state.cell3.get(&name) {
                    Some(t) => {
                        computed.insert(name, floats(t.clone()));
                    }
                    None => {
                        computed.insert(name.clone(), floats(vec![0.0; target.n_cells * nt]));
                        absent.push(name);
                    }
                }
            }
            Method::Interface => match &cols.state.w {
                Some(w) => {
                    computed.insert(name, floats(w.clone()));
                }
                None => {
                    computed.insert(name.clone(), floats(vec![0.0; target.n_cells * (nt + 1)]));
                    absent.push(name);
                }
            },
            Method::EdgeNormalWind => match &cols.state.u {
                Some(u) => {
                    computed.insert(name, floats(u.clone()));
                }
                None => {
                    computed.insert(name.clone(), floats(vec![0.0; target.n_edges * nt]));
                    absent.push(name);
                }
            },
            Method::Derived => {
                if let Some(t) = cols.derived3.get(&name).or_else(|| cols.derived2.get(&name)) {
                    computed.insert(name, floats(t.clone()));
                } else if name == "xland" {
                    if let Some(tc) = &tgt_class {
                        computed.insert(name, floats(tc.iter().map(|&c| if c == 1 { 1.0 } else { 2.0 }).collect()));
                    }
                } else if name == "surface_pressure" && sfile.has(&name) {
                    let vals = sfile.f32s(&name, source.n_cells)?;
                    let t = remap_surface(&ops, target.n_cells, &vals, 1, Method::Barycentric, None, None, &mut fallbacks)?;
                    report_surface(&mut variables, &name, Method::Barycentric, &vals, &t);
                    computed.insert(name, floats(t));
                }
            }
            Method::AreaConservative | Method::MaskedConservative | Method::Barycentric | Method::Dominant => {
                if width == 0 {
                    continue;
                }
                if !sfile.has(&name) {
                    computed.insert(name.clone(), floats(vec![0.0; target.n_cells * width]));
                    absent.push(name);
                    continue;
                }
                let vals = sfile.f32s(&name, source.n_cells * width)?;
                let t = remap_surface(
                    &ops,
                    target.n_cells,
                    &vals,
                    width,
                    method,
                    src_class.as_deref(),
                    tgt_class.as_deref(),
                    &mut fallbacks,
                )?;
                report_surface(&mut variables, &name, method, &vals, &t);
                computed.insert(name, floats(t));
            }
        }
    }
    // Sea ice makes a water cell land in xland, as physics_init_seaice does.
    if let (Some(emit::Computed::Floats(seaice)), true) = (computed.get("seaice").map(clone_floats), computed.contains_key("xland")) {
        if let Some(emit::Computed::Floats(x)) = computed.get_mut("xland") {
            for (xv, s) in x.iter_mut().zip(seaice.iter()) {
                if *s >= 0.5 {
                    *xv = 1.0;
                }
            }
        }
    }
    receipt.soil_cells_mask_fallback = fallbacks;
    receipt.absent_in_source_written_as_zero = absent;
    receipt.variables = variables;

    // Identity labels.
    for v in tfile.file.variables()? {
        if !matches!(v.dtype(), netcrust::DataType::Char) {
            continue;
        }
        let name = v.name().to_string();
        let text = if matches!(method_for(&name), Some(Method::Label)) {
            // The valid time is the SOURCE's: the state is the source's.
            let Some(text) = read_text(&cfg.from_state, &name).or_else(|| read_text(&cfg.from_state, "xtime"))
            else {
                return Err(MpasError::Refusal(format!(
                    "the target template declares the label {name} and the source state carries                      no valid time to fill it with; an identity label is never invented"
                )));
            };
            text
        } else {
            // Any other label (mminlu, densityFunctionCode) describes the
            // target's statics: the template's own, else the source's, else
            // left blank exactly as the template holds it.
            read_text(&template, &name)
                .or_else(|| read_text(&cfg.from_state, &name))
                .unwrap_or_default()
        };
        computed.insert(name, emit::Computed::Text(text));
    }

    // Lineage: the source's own when it has one.
    let src_attr = |name: &str| -> Option<String> {
        sfile.file.attribute(name).and_then(|a| a.as_string().map(str::to_string))
    };
    let default = crate::init::Lineage::default();
    let lineage = crate::init::Lineage {
        model_name: src_attr("model_name").unwrap_or(default.model_name),
        core_name: src_attr("core_name").unwrap_or(default.core_name),
        version: src_attr("version").unwrap_or(default.version),
        git_version: src_attr("git_version").unwrap_or(default.git_version),
    };
    let seed = format!(
        "remap|{}|{}|{}",
        cfg.from_state.display(),
        cfg.to_grid.display(),
        template.display()
    );
    let file_id = emit::mint_file_id(&seed);
    let parent = src_attr("file_id").unwrap_or_default();
    let provenance = format!(
        "{} | remap from-state={} to-grid={} template={} balance={} virtual-factor={} \
         min-coverage={}",
        cfg.provenance,
        cfg.from_state.display(),
        cfg.to_grid.display(),
        template.display(),
        receipt.vertical.balance,
        receipt.vertical.virtual_factor,
        cfg.min_coverage
    );
    if let Some(dir) = cfg.out.parent() {
        if !dir.as_os_str().is_empty() && !dir.is_dir() {
            return Err(MpasError::Refusal(format!(
                "the output directory {} does not exist",
                dir.display()
            )));
        }
    }
    receipt.emit = Some(emit::write_init(
        &cfg.out,
        &template,
        computed,
        &file_id,
        &[parent],
        &provenance,
        &lineage,
    )?);
    receipt.status = "remapped".to_string();
    receipt.seconds = started.elapsed().as_secs_f64();
    Ok(receipt)
}

fn clone_floats(c: &emit::Computed) -> emit::Computed {
    match c {
        emit::Computed::Floats(v) => emit::Computed::Floats(v.clone()),
        emit::Computed::Ints(v) => emit::Computed::Ints(v.clone()),
        emit::Computed::Text(t) => emit::Computed::Text(t.clone()),
    }
}

fn report_surface(variables: &mut Vec<VariableReport>, name: &str, method: Method, s: &[f32], t: &[f32]) {
    let range = |v: &[f32]| v.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, &x| (a.0.min(x as f64), a.1.max(x as f64)));
    let (smin, smax) = range(s);
    let (tmin, tmax) = range(t);
    let tol = 1.0e-6 * (smax - smin).abs().max(smax.abs()).max(1e-30);
    variables.push(VariableReport {
        name: name.to_string(),
        method: method.label().to_string(),
        source_min: smin,
        source_max: smax,
        target_min: tmin,
        target_max: tmax,
        overshoot_points: t.iter().filter(|&&x| (x as f64) < smin - tol || (x as f64) > smax + tol).count(),
        ..Default::default()
    });
}

#[cfg(test)]
mod tests;
