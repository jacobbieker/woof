//! A mesh generated from a density raster, end to end, and the delivered
//! spacing held against the raster it was asked for.
//!
//! Coarse on purpose (480 km background, 240 km core: one ladder level, a
//! few thousand cells) so it runs at test speed; the arithmetic it exercises
//! -- raster sampling inside the relaxation, the exact sizing integral, the
//! limiter, the stamped provenance -- is the same at any resolution.

use rw_mpas::mesh::geom::{EARTH_RADIUS_M, V3, arc, from_lat_lon};
use rw_mpas::mesh::raster::{RASTER_SCHEMA, write_density_raster};
use rw_mpas::mesh::{GenerateRequest, LloydOptions, MeshSpec, generate};

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("rw-mpas-mesh-raster-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn at(lat: f64, lon: f64) -> V3 {
    from_lat_lon(lat.to_radians(), lon.to_radians())
}

#[test]
fn a_mesh_from_a_raster_delivers_the_raster_spacing() {
    let centre = at(20.0, -90.0);
    // 240 km within 1,500 km of the centre, graded at 7 % out to the 480 km
    // background, which it reaches about 3,400 km further out -- well inside
    // the raster's edge, so the raster meets the background with no step.
    let raster_km = |p: V3| -> f64 {
        let d_km = arc(centre, p) * EARTH_RADIUS_M / 1000.0;
        (240.0 + 0.07 * (d_km - 1500.0).max(0.0)).min(480.0)
    };
    let lat: Vec<f64> = (0..=110).map(|k| -35.0 + k as f64).collect();
    let lon: Vec<f64> = (0..=140).map(|k| -160.0 + k as f64).collect();
    let mut values = Vec::with_capacity(lat.len() * lon.len());
    for &a in &lat {
        for &b in &lon {
            values.push(raster_km(at(a, b)));
        }
    }
    let path = scratch("graded.density.nc");
    write_density_raster(&path, &lat, &lon, &values, 240.0, RASTER_SCHEMA).unwrap();
    let text = format!(
        r#"{{"background_km": 480.0, "regions": [{{"shape": "raster", "path": {:?}}}], "name": "raster e2e"}}"#,
        path.to_string_lossy()
    );
    let spec = MeshSpec::from_json(&text).unwrap();
    let predicted = spec.predicted_cells(200_000);

    let out = generate(
        &GenerateRequest {
            spec,
            lloyd: LloydOptions {
                tolerance: 3e-3,
                max_sweeps: 200,
                ..Default::default()
            },
            ..Default::default()
        },
        |line| eprintln!("  {line}"),
    )
    .unwrap_or_else(|e| panic!("generate refused: {e}"));
    let r = &out.receipt;
    eprintln!(
        "raster mesh: {} cells (predicted {predicted:.0}); delivered/requested p05 {:.3} median {:.3} p95 {:.3}",
        r.delivered_cells,
        r.delivered_over_requested_p05,
        r.delivered_over_requested_median,
        r.delivered_over_requested_p95
    );
    assert_eq!(r.raster_regions.len(), 1);
    assert!(r.raster_regions[0].certified_peak_per_cell <= r.raster_regions[0].limiter.max_gradient_per_cell);
    assert_eq!(r.gradient_probe_coverage, "complete");
    assert!(
        (r.delivered_cells as f64 / predicted - 1.0).abs() < 0.05,
        "delivered {} against a predicted {predicted:.0}",
        r.delivered_cells
    );
    assert!(
        (0.93..=1.07).contains(&r.delivered_over_requested_median),
        "median delivered/requested {}",
        r.delivered_over_requested_median
    );

    // The core and the far field, measured on the mesh itself.
    let spacing = out.mesh.spacing_m();
    let mut core = Vec::new();
    let mut far = Vec::new();
    for (i, &p) in out.mesh.cell_xyz.iter().enumerate() {
        let d_km = arc(centre, p) * EARTH_RADIUS_M / 1000.0;
        if d_km < 1000.0 {
            core.push(spacing[i] / 1000.0);
        } else if d_km > 7000.0 {
            far.push(spacing[i] / 1000.0);
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (core_km, far_km) = (median(&mut core), median(&mut far));
    eprintln!("core median {core_km:.1} km over {} cells; far median {far_km:.1} km", core.len());
    assert!(core.len() > 50, "only {} core cells", core.len());
    assert!((core_km / 240.0 - 1.0).abs() < 0.08, "core delivered {core_km:.1} km for 240");
    assert!((far_km / 480.0 - 1.0).abs() < 0.08, "far field delivered {far_km:.1} km for 480");

    // The stamped spec names the raster by content, not by where it sat.
    let stamped = rw_mpas::mesh::density::provenance_spec_json(&out.spec).unwrap();
    assert!(!stamped.contains("graded.density.nc"), "{stamped}");
    let receipt = rw_mpas::mesh::provenance_json(&out.receipt).unwrap();
    assert!(!receipt.contains("graded.density.nc"));
    assert!(receipt.contains("raster_regions"));
}
