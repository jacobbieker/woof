//! Whole-pipeline tests on meshes built in memory: two different
//! icosahedral Voronoi meshes (one jittered and re-triangulated, so no cell
//! of one coincides with a cell of the other).

use super::*;
use crate::mesh::derive::MpasMesh;
use crate::mesh::geom::{self, V3};
use crate::mesh::{hull, icosa};

fn build_mesh(m: u32, n: u32, jitter: f64) -> RemapMesh {
    let mut pts = icosa::seed(m, n).unwrap();
    if jitter > 0.0 {
        // Deterministic jitter, a fraction of the spacing.
        let h = (4.0 * std::f64::consts::PI / pts.len() as f64).sqrt();
        for (i, p) in pts.iter_mut().enumerate() {
            let a = (i as f64 * 12.9898).sin() * 43758.5453;
            let b = (i as f64 * 78.233).sin() * 12345.678;
            let d = [a.fract() - 0.5, b.fract() - 0.5, (a * b).fract() - 0.5];
            *p = geom::unit(geom::add(*p, geom::scale(d, jitter * h))).unwrap();
        }
    }
    let pts = hull::to_unit_sphere(&pts).unwrap();
    let rings = hull::delaunay_rings(&pts).unwrap();
    let density = vec![1.0; pts.len()];
    let mm = MpasMesh::derive(pts, density, &rings, 0.1).unwrap();
    RemapMesh::from_mpas_mesh(&mm).unwrap()
}

fn mesh_a() -> RemapMesh {
    build_mesh(6, 0, 0.0)
}

fn mesh_b() -> RemapMesh {
    build_mesh(5, 2, 0.15)
}

const LEVELS: usize = 12;
const TOP: f64 = 24000.0;

fn terrain(x: V3) -> f64 {
    600.0 * (1.0 + x[0]) * (0.5 + 0.5 * x[2] * x[2])
}

fn zgrid(mesh: &RemapMesh, with_terrain: bool) -> Vec<f32> {
    let mut z = Vec::with_capacity(mesh.n_cells * (LEVELS + 1));
    for c in 0..mesh.n_cells {
        let h = if with_terrain { terrain(mesh.cell_xyz[c]) } else { 0.0 };
        for k in 0..=LEVELS {
            let s = k as f64 / LEVELS as f64;
            z.push((h + (TOP - h) * s) as f32);
        }
    }
    z
}

fn state(mesh: &RemapMesh, zg: Vec<f32>, f: &dyn Fn(V3, f64) -> (f64, f64, f64)) -> State {
    let mut rho = Vec::new();
    let mut theta = Vec::new();
    let mut qv = Vec::new();
    for c in 0..mesh.n_cells {
        for k in 0..LEVELS {
            let zm = 0.5 * (zg[c * (LEVELS + 1) + k] + zg[c * (LEVELS + 1) + k + 1]) as f64;
            let (r, t, q) = f(mesh.cell_xyz[c], zm);
            rho.push(r as f32);
            theta.push(t as f32);
            qv.push(q as f32);
        }
    }
    let mut s = State {
        levels: LEVELS,
        zgrid: zg,
        ..Default::default()
    };
    s.cell3.insert("rho".into(), rho);
    s.cell3.insert("theta".into(), theta);
    s.cell3.insert("qv".into(), qv);
    s
}

/// Solid-body rotation about an axis tilted 30 degrees off the pole,
/// 20 m/s at the rotation equator.
fn solid_body(x: V3) -> V3 {
    let t = 30f64.to_radians();
    let axis = [t.sin(), 0.0, t.cos()];
    geom::scale(geom::cross(axis, x), 20.0)
}

fn edge_wind(mesh: &RemapMesh) -> Vec<f32> {
    let mut u = Vec::with_capacity(mesh.n_edges * LEVELS);
    for e in 0..mesh.n_edges {
        let v = geom::dot(solid_body(mesh.edge_xyz[e]), mesh.edge_normal[e]);
        for _ in 0..LEVELS {
            u.push(v as f32);
        }
    }
    u
}

#[test]
fn the_intersection_areas_of_every_cell_sum_to_its_area() {
    let a = mesh_a();
    let b = mesh_b();
    let ov = overlap::build(&a, &b).unwrap();
    let mut worst_target = 0.0f64;
    for j in 0..b.n_cells {
        worst_target = worst_target.max((ov.coverage[j] - 1.0).abs());
    }
    let mut worst_source = 0.0f64;
    for i in 0..a.n_cells {
        worst_source = worst_source.max((ov.source_covered[i] / a.polygon_area[i] - 1.0).abs());
    }
    assert!(worst_target < 1e-10, "target coverage off by {worst_target:e}");
    assert!(worst_source < 1e-10, "source coverage off by {worst_source:e}");
    let total: f64 = ov.rows.iter().flat_map(|r| r.iter().map(|x| x.1)).sum();
    assert!((total / (4.0 * std::f64::consts::PI) - 1.0).abs() < 1e-11);
}

#[test]
fn the_identity_remap_reproduces_every_field() {
    let a = mesh_a();
    let zg = zgrid(&a, true);
    let mut src = state(&a, zg.clone(), &|x, z| {
        (1.2 * (-z / 8000.0).exp() * (1.0 + 0.05 * x[1]), 290.0 + 0.004 * z + 5.0 * x[0], 0.01 * (-z / 3000.0).exp())
    });
    src.u = Some(edge_wind(&a));
    let ops = Operators::build(&a, &a).unwrap();
    // Each target overlaps itself and only slivers of anything else.
    for j in 0..a.n_cells {
        assert_eq!(ops.overlap.rows[j].len(), 1, "cell {j}: {:?}", ops.overlap.rows[j]);
    }
    let out = remap_columns(&a, &a, &ops, &src, &zg, LEVELS, None, Balance::Carry, VirtualFactor::ReproduceFortran).unwrap();
    for name in ["rho", "theta", "qv"] {
        let s = &src.cell3[name];
        let t = &out.state.cell3[name];
        let worst = s.iter().zip(t).map(|(a, b)| ((a - b) / a).abs()).fold(0.0f32, f32::max);
        assert!(worst < 2e-6, "{name}: {worst:e}");
    }
    // The wind passes through a reconstruction, so it is not bitwise; it is
    // the reconstruction's own error, a few percent at this resolution.
    let su = src.u.as_ref().unwrap();
    let tu = out.state.u.as_ref().unwrap();
    let rms = (su.iter().zip(tu).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / su.len() as f64).sqrt();
    assert!(rms < 0.5, "identity wind rms {rms}");
}

#[test]
fn a_constant_state_is_preserved_exactly() {
    let a = mesh_a();
    let b = mesh_b();
    let src = state(&a, zgrid(&a, false), &|_, _| (1.0, 300.0, 0.005));
    let ops = Operators::build(&a, &b).unwrap();
    let out = remap_columns(&a, &b, &ops, &src, &zgrid(&b, false), LEVELS, None, Balance::Carry, VirtualFactor::ReproduceFortran)
        .unwrap();
    for (name, want) in [("rho", 1.0f32), ("theta", 300.0), ("qv", 0.005)] {
        for &v in &out.state.cell3[name] {
            assert!(((v - want) / want).abs() < 1e-6, "{name}: {v} vs {want}");
        }
    }
}

#[test]
fn a_constant_state_survives_different_terrain() {
    let a = mesh_a();
    let b = mesh_b();
    let src = state(&a, zgrid(&a, true), &|_, _| (1.0, 300.0, 0.005));
    let ops = Operators::build(&a, &b).unwrap();
    let out = remap_columns(&a, &b, &ops, &src, &zgrid(&b, true), LEVELS, None, Balance::Carry, VirtualFactor::ReproduceFortran)
        .unwrap();
    for (name, want) in [("rho", 1.0f32), ("theta", 300.0), ("qv", 0.005)] {
        for &v in &out.state.cell3[name] {
            assert!(((v - want) / want).abs() < 1e-5, "{name}: {v} vs {want}");
        }
    }
}

#[test]
fn a_linear_field_is_remapped_within_its_first_order_tolerance() {
    let a = mesh_a();
    let b = mesh_b();
    let theta_of = |x: V3, z: f64| 290.0 + 0.004 * z + 10.0 * x[0] - 6.0 * x[2];
    let src = state(&a, zgrid(&a, true), &|x, z| (1.0, theta_of(x, z), 0.004));
    let ops = Operators::build(&a, &b).unwrap();
    let tz = zgrid(&b, true);
    let out = remap_columns(&a, &b, &ops, &src, &tz, LEVELS, None, Balance::Carry, VirtualFactor::ReproduceFortran).unwrap();
    let t = &out.state.cell3["theta"];
    // A first-order conservative cell mean of a linear field is the value
    // at the polygon's centroid; the generator is near but not at it.  The
    // gradient here is ~12 K per radian and the spacing ~0.19 rad, so a
    // tenth of a spacing of offset is ~0.2 K.
    let mut worst = 0.0f64;
    for c in 0..b.n_cells {
        for k in 0..LEVELS {
            let zm = 0.5 * (tz[c * (LEVELS + 1) + k] + tz[c * (LEVELS + 1) + k + 1]) as f64;
            let want = theta_of(b.cell_xyz[c], zm);
            worst = worst.max((t[c * LEVELS + k] as f64 - want).abs());
        }
    }
    assert!(worst < 0.5, "worst linear error {worst} K");
}

#[test]
fn mass_is_conserved_when_both_columns_span_the_same_heights() {
    let a = mesh_a();
    let b = mesh_b();
    let src = state(&a, zgrid(&a, false), &|x, z| {
        (1.2 * (-z / 8000.0).exp() * (1.0 + 0.1 * x[0]), 290.0 + 0.004 * z, 0.01 * (-z / 2500.0).exp() * (1.0 + 0.5 * x[2]))
    });
    let ops = Operators::build(&a, &b).unwrap();
    let tz = zgrid(&b, false);
    let out = remap_columns(&a, &b, &ops, &src, &tz, LEVELS, None, Balance::Carry, VirtualFactor::ReproduceFortran).unwrap();
    let before = budget(&a, &src, &|_| 1.0);
    let after = budget(&b, &out.state, &|_| 1.0);
    assert!((out.horizontal_dry_air_kg / before.dry_air_kg - 1.0).abs() < 1e-12);
    assert!((out.horizontal_water_vapour_kg / before.water_vapour_kg - 1.0).abs() < 1e-12);
    assert!((out.carried_dry_air_kg / before.dry_air_kg - 1.0).abs() < 1e-12);
    // The written state is f32: conserved to its rounding.
    assert!((after.dry_air_kg / before.dry_air_kg - 1.0).abs() < 1e-6);
    assert!((after.water_vapour_kg / before.water_vapour_kg - 1.0).abs() < 1e-6);
}

#[test]
fn solid_body_rotation_survives_the_remap_with_small_error() {
    let a = mesh_a();
    let b = mesh_b();
    let mut src = state(&a, zgrid(&a, false), &|_, _| (1.0, 300.0, 0.0));
    src.u = Some(edge_wind(&a));
    let ops = Operators::build(&a, &b).unwrap();
    let out = remap_columns(&a, &b, &ops, &src, &zgrid(&b, false), LEVELS, None, Balance::Carry, VirtualFactor::ReproduceFortran)
        .unwrap();
    let want = edge_wind(&b);
    let got = out.state.u.as_ref().unwrap();
    let rms_want = (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    let rms_err = (want.iter().zip(got).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    let max_err = want.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(rms_err / rms_want < 0.05, "relative rms error {}", rms_err / rms_want);
    assert!(max_err < 3.0, "max error {max_err} m/s of 20");
    // The kinetic energy is kept to the same order.
    let ke_a = budget(&a, &src, &|_| 1.0).kinetic_energy_j;
    let ke_b = budget(&b, &out.state, &|_| 1.0).kinetic_energy_j;
    assert!((ke_b / ke_a - 1.0).abs() < 0.05, "KE ratio {}", ke_b / ke_a);
}

#[test]
fn the_hydrostatic_rebalance_keeps_theta_and_moves_rho_little_on_a_balanced_state() {
    let a = mesh_a();
    let b = mesh_b();
    // An isothermal 250 K atmosphere is hydrostatic with this scale height.
    let h = 287.0 * 250.0 / 9.80616;
    let src = state(&a, zgrid(&a, true), &|_, z| {
        let p = 1.0e5 * (-z / h).exp();
        let rho = p / (287.0 * 250.0);
        let theta = 250.0 * (1.0e5 / p).powf(287.0 / 1004.5);
        (rho, theta, 0.0)
    });
    let tz = zgrid(&b, true);
    let nt = LEVELS;
    let mut zz = Vec::new();
    for _ in 0..b.n_cells {
        for _ in 0..nt {
            zz.push(1.0f32);
        }
    }
    let dz = (TOP / LEVELS as f64) as f32;
    let metrics = Metrics {
        zz,
        fzm: (0..nt).map(|k| if k == 0 { 0.0 } else { 0.5 }).collect(),
        fzp: (0..nt).map(|k| if k == 0 { 0.0 } else { 0.5 }).collect(),
        dzu: vec![dz; nt],
        rdzw: vec![1.0 / dz; nt],
    };
    let ops = Operators::build(&a, &b).unwrap();
    let out = remap_columns(&a, &b, &ops, &src, &tz, nt, Some(&metrics), Balance::Hydrostatic, VirtualFactor::ReproduceFortran)
        .unwrap();
    assert!(out.vertical.max_relative_rebalance_of_rho < 0.03, "{}", out.vertical.max_relative_rebalance_of_rho);
    // The fixed point's own pass cap (30, as the init writer has it) is
    // reported, not asserted: in f32 near an exactly balanced column the
    // 1e-4 Pa criterion can sit inside the rounding of the pressure.
    assert!(out.vertical.hydrostatic_iterations_max <= 30);
    assert!(out.derived2.contains_key("surface_pressure"));
}

#[test]
fn an_uncovered_target_is_reported_and_counted() {
    let mut a = mesh_a();
    let b = mesh_b();
    // Knock a cap of source cells out, as a regional source would lack it.
    for c in 0..a.n_cells {
        if a.cell_xyz[c][2] > 0.9 {
            a.polygons[c] = None;
        }
    }
    let ops = Operators::build(&a, &b).unwrap();
    let report = coverage_report(&a, &b, &ops, DEFAULT_MIN_COVERAGE);
    assert!(report.cells_below_required > 0);
    assert!(report.min_coverage < 0.5);
    let w = &report.worst[0];
    assert!(w.lat_deg > 50.0, "the worst cell is in the cap: {}", w.lat_deg);
}

#[test]
fn a_target_above_the_source_top_is_refused() {
    let a = mesh_a();
    let b = mesh_b();
    let src = state(&a, zgrid(&a, false), &|_, _| (1.0, 300.0, 0.005));
    let ops = Operators::build(&a, &b).unwrap();
    let mut tz = zgrid(&b, false);
    for c in 0..b.n_cells {
        tz[c * (LEVELS + 1) + LEVELS] += 500.0;
    }
    let err = remap_columns(&a, &b, &ops, &src, &tz, LEVELS, None, Balance::Carry, VirtualFactor::ReproduceFortran)
        .err()
        .expect("refused");
    assert!(err.to_string().contains("above the source's model top"), "{err}");
}

#[test]
fn hydrostatic_balance_without_metrics_is_refused() {
    let a = mesh_a();
    let src = state(&a, zgrid(&a, false), &|_, _| (1.0, 300.0, 0.005));
    let ops = Operators::build(&a, &a).unwrap();
    let err = remap_columns(&a, &a, &ops, &src, &zgrid(&a, false), LEVELS, None, Balance::Hydrostatic, VirtualFactor::ReproduceFortran)
        .err()
        .expect("refused");
    assert!(err.to_string().contains("--balance carry"), "{err}");
}
