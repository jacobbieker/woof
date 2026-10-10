//! The regional window (`--regional-window`), end to end: what it freezes,
//! what it delivers, and that its output is an ordinary whole-sphere mesh.

use std::path::PathBuf;
use std::process::Command;

use rw_mpas::mesh::density::{MeshSpec, Region, Shape, TransitionField};
use rw_mpas::mesh::emit::{Provenance, write_grid};
use rw_mpas::mesh::geom::{EARTH_RADIUS_M, V3, add, scale, unit};
use rw_mpas::mesh::hull::{TriangulationMode, delaunay_rings, delaunay_triangulation, lawson_repair};
use rw_mpas::mesh::regional::{RegionalWindow, Zone};
use rw_mpas::mesh::{GenerateRequest, Generated, LloydOptions, generate};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("rw_mpas_mesh_regional_tests");
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir.join(name)
}

const CENTRE: [f64; 2] = [45.0, 10.0];

fn small_spec() -> MeshSpec {
    MeshSpec {
        background_km: 480.0,
        regions: vec![Region {
            shape: Shape::Cap {
                center_deg: CENTRE,
                radius_km: 800.0,
            },
            spacing_km: 120.0,
            transition: TransitionField::Km(1500.0),
        }],
        name: Some("regional window test".into()),
    }
}

fn window() -> RegionalWindow {
    RegionalWindow {
        shape: Shape::Cap {
            center_deg: CENTRE,
            radius_km: 800.0,
        },
        source_format: "shape_row".into(),
        halo_rings: None,
    }
}

fn request(mode: TriangulationMode) -> GenerateRequest {
    GenerateRequest {
        spec: small_spec(),
        sizing_samples: 50_000,
        lloyd: LloydOptions {
            triangulation: mode,
            ..Default::default()
        },
        regional: Some(window()),
        ..Default::default()
    }
}

fn generated() -> Generated {
    generate(&request(TriangulationMode::Maintained), |_| {}).unwrap_or_else(|e| panic!("{e}"))
}

fn bits(p: V3) -> [u64; 3] {
    [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]
}

/// The level-0 background, rebuilt independently from the public pieces the
/// ladder uses: the same Goldberg snap and seed, the same relaxation under the
/// same uniform clamp. Every generator of it outside the zone must survive in
/// the finished mesh to the last bit.
#[test]
fn every_generator_outside_the_zone_is_the_level_0_background_bit_for_bit() {
    let out = generated();
    let regional = out.receipt.regional_window.as_ref().expect("regional receipt");
    assert!(regional.frozen_bitwise_unchanged);
    assert_eq!(regional.status, "experimental");

    let spec = &out.spec;
    let h_bg = spec.background_km * 1000.0;
    let n0 = MeshSpec::uniform(spec.background_km).predicted_cells(50_000).round() as usize;
    let choice = rw_mpas::mesh::icosa::snap_cells(n0.max(12), false).unwrap();
    let mut level0 = rw_mpas::mesh::icosa::seed(choice.m, choice.n).unwrap();
    let prepared = spec.prepared();
    let clamp0 = prepared.clamped(h_bg);
    rw_mpas::mesh::lloyd::relax(
        &mut level0,
        &clamp0,
        &LloydOptions {
            triangulation: TriangulationMode::Maintained,
            ..Default::default()
        },
    )
    .unwrap();

    let zone = Zone::new(&window().shape, regional.halo_km * 1000.0 / EARTH_RADIUS_M);
    let frozen: Vec<V3> = level0.iter().copied().filter(|&p| !zone.active(p)).collect();
    assert_eq!(frozen.len(), regional.frozen_cells, "the frozen count moved");
    assert!(frozen.len() > level0.len() / 4, "the case freezes too little to test anything");
    let present: std::collections::HashSet<[u64; 3]> = out.mesh.cell_xyz.iter().map(|&p| bits(p)).collect();
    let moved = frozen.iter().filter(|&&p| !present.contains(&bits(p))).count();
    assert_eq!(moved, 0, "{moved} frozen background generators moved");
    // And nothing new appeared outside the zone.
    let outside = out.mesh.cell_xyz.iter().filter(|&&p| !zone.active(p)).count();
    assert_eq!(outside, frozen.len(), "a generator was inserted outside the zone");
}

/// Inside the window the delivered spacing follows the spec, and the mesh
/// passed the full emit gate (validation, dual edges, coordination).
#[test]
fn inside_the_window_the_spacing_tracks_the_spec_and_the_mesh_validates() {
    let out = generated();
    let prepared = out.spec.prepared();
    let zone = Zone::new(&window().shape, 0.0);
    let delivered = out.mesh.spacing_m();
    let mut ratios: Vec<f64> = (0..out.mesh.n_cells)
        .filter(|&i| zone.active(out.mesh.cell_xyz[i]))
        .map(|i| delivered[i] / prepared.spacing_m(out.mesh.cell_xyz[i]))
        .collect();
    assert!(ratios.len() > 30, "only {} cells in the window", ratios.len());
    ratios.sort_by(|a, b| a.total_cmp(b));
    let at = |q: f64| ratios[((ratios.len() - 1) as f64 * q).round() as usize];
    eprintln!("window delivered/requested p05 {:.4} median {:.4} p95 {:.4}", at(0.05), at(0.5), at(0.95));
    assert!((at(0.5) - 1.0).abs() < 0.05, "window median {:.4}", at(0.5));
    assert!(at(0.05) > 0.8 && at(0.95) < 1.25);

    let report = &out.receipt.mesh;
    assert_eq!(report.coordination_defect, 12);
    assert!(report.min_dv_over_dc >= 0.02, "dv/dc {}", report.min_dv_over_dc);
    let hist = &report.coordination_histogram;
    assert!(hist.iter().all(|(k, _)| *k >= 5), "a cell below five edges: {hist:?}");
    let q = &out.receipt.regional_window.as_ref().unwrap().halo_quality;
    assert!(q.interface_frozen_cells > 0 && q.halo_cells > 0 && q.window_cells > 0);
    assert!(q.halo_max_adjacent_spacing_ratio.is_finite());
}

/// The output is an ordinary whole-sphere grid: written, and cut by the same
/// culler the published meshes go through.
#[test]
fn the_regional_mesh_writes_and_culls_like_a_global_one() {
    let out = generated();
    let grid = scratch("regional.grid.nc");
    let cut = scratch("regional.cull.nc");
    let _ = std::fs::remove_file(&grid);
    let _ = std::fs::remove_file(&cut);
    write_grid(
        &out.mesh,
        &grid,
        &Provenance {
            spec_json: serde_json::to_string(&out.spec).unwrap(),
            request: "regional window test".into(),
            receipt_json: rw_mpas::mesh::provenance_json(&out.receipt).unwrap(),
            static_coordinates: None,
        },
        false,
    )
    .expect("write");
    let receipt = rw_mpas::mesh::cull::cull_file(&grid, &window().shape, &cut, None).expect("cull");
    assert!(receipt.region_cells > 30 && receipt.region_cells < out.mesh.n_cells);
    let _ = std::fs::remove_file(&grid);
    let _ = std::fs::remove_file(&cut);
}

/// The two arms are the same triangulation: on this graded point set, moved
/// the way a relaxation sweep moves it, a Lawson-repaired triangulation and a
/// rebuilt one give every cell the same neighbour SET.
#[test]
fn incremental_and_rebuild_triangulations_agree_topologically() {
    let out = generated();
    let points = out.mesh.cell_xyz.clone();
    let rings = delaunay_rings(&points).unwrap();
    // Each generator 5% of the way to its neighbours' mean: a sweep-sized,
    // deterministic motion that flips edges in the graded band.
    let moved: Vec<V3> = (0..points.len())
        .map(|i| {
            let ring = rings.ring(i);
            let mut acc = [0.0; 3];
            for &j in ring {
                acc = add(acc, points[j as usize]);
            }
            let mean = unit(acc).unwrap();
            unit(add(scale(points[i], 0.95), scale(mean, 0.05))).unwrap()
        })
        .collect();
    let mut tri = delaunay_triangulation(&points).unwrap();
    lawson_repair(&moved, &mut tri).unwrap();
    let repaired = tri.rings().unwrap();
    let rebuilt = delaunay_rings(&moved).unwrap();
    for i in 0..moved.len() {
        let mut a = repaired.ring(i).to_vec();
        let mut b = rebuilt.ring(i).to_vec();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "cell {i} has different neighbours on the two arms");
    }
}

/// The rebuild arm still runs a window, and the class is stamped: absent on
/// rebuild, `incremental` on the maintained arm.
#[test]
fn the_window_runs_on_either_arm_and_the_arm_is_stamped() {
    let incremental = generated();
    assert_eq!(incremental.receipt.triangulation, Some("incremental"));
    let rebuild = generate(&request(TriangulationMode::Rebuild), |_| {}).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(rebuild.receipt.triangulation, None);
    assert!(rebuild.receipt.regional_window.as_ref().unwrap().frozen_bitwise_unchanged);
}

#[test]
fn a_window_on_a_uniform_request_is_refused() {
    let req = GenerateRequest {
        spec: MeshSpec::uniform(480.0),
        regional: Some(window()),
        ..Default::default()
    };
    let err = generate(&req, |_| {}).unwrap_err().to_string();
    assert!(err.contains("needs a graded spec"), "{err}");
}

// ---- the binary's flags and the triangulation rule ----------------------

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rw_mpas_mesh"))
}

fn write(name: &str, text: &str) -> PathBuf {
    let p = scratch(name);
    std::fs::write(&p, text).unwrap();
    p
}

fn dry_run(args: &[&str]) -> serde_json::Value {
    let out = bin().args(args).arg("--dry-run").output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    serde_json::from_slice(&out.stdout).expect("dry-run prints JSON alone")
}

#[test]
fn the_triangulation_rule_is_rebuild_globally_and_incremental_with_a_window() {
    let spec = write("rule.spec.json", &serde_json::to_string(&small_spec()).unwrap());
    let win = write(
        "rule.window.geojson",
        r#"{"type": "Polygon", "coordinates": [[[3.0, 40.0], [17.0, 40.0], [17.0, 50.0], [3.0, 50.0], [3.0, 40.0]]]}"#,
    );
    let spec = spec.to_str().unwrap();
    let win = win.to_str().unwrap();
    let global = dry_run(&["--spec", spec]);
    assert_eq!(global["triangulation"], "rebuild");
    assert!(global["regional_window"].is_null());
    let regional = dry_run(&["--spec", spec, "--regional-window", win]);
    assert_eq!(regional["triangulation"], "incremental");
    assert_eq!(regional["regional_window"]["window_format"], "geojson");
    assert_eq!(regional["regional_window"]["status"], "experimental");
    let named = dry_run(&["--spec", spec, "--regional-window", win, "--triangulation", "rebuild"]);
    assert_eq!(named["triangulation"], "rebuild");
}

#[test]
fn regional_flags_are_refused_where_they_cannot_apply() {
    let spec = write("refuse.spec.json", &serde_json::to_string(&small_spec()).unwrap());
    let out = bin()
        .args(["--spec", spec.to_str().unwrap(), "--regional-halo-rings", "8", "--dry-run"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("no window"));
    for flag in [["--triangulation", "incremental"], ["--regional-window", "w.json"]] {
        let out = bin()
            .args(["--from-centres", "x.nc", flag[0], flag[1], "--dry-run"])
            .output()
            .unwrap();
        assert!(!out.status.success());
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("never generates"), "{err}");
    }
    let bad = write("bad.window.json", r#"{"type": "LineString", "coordinates": [[0, 0], [1, 1]]}"#);
    let out = bin()
        .args(["--spec", spec.to_str().unwrap(), "--regional-window", bad.to_str().unwrap(), "--dry-run"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("LineString"));
}
