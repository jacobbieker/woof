//! The density raster's contract, each clause measured in both directions.

use super::*;
use crate::mesh::density::{DensityField, MeshSpec, steepest_gradient_reading_of};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rw-mpas-raster-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn axis(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n).map(|k| lo + (hi - lo) * k as f64 / (n - 1) as f64).collect()
}

/// A raster `f(lat, lon)` on the given axes, written to a scratch file.
fn raster_file(name: &str, lat: &[f64], lon: &[f64], f: impl Fn(f64, f64) -> f64) -> PathBuf {
    let mut v = Vec::with_capacity(lat.len() * lon.len());
    for &a in lat {
        for &b in lon {
            v.push(f(a, b));
        }
    }
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    let path = scratch(name);
    write_density_raster(&path, lat, lon, &v, min, RASTER_SCHEMA).unwrap();
    path
}

fn spec_with(
    background_km: f64,
    extra_regions: &str,
    path: &Path,
    row_extra: &str,
) -> MpasResult<MeshSpec> {
    let text = format!(
        r#"{{"background_km": {background_km}, "regions": [{extra_regions}{{"shape": "raster", "path": {:?}{row_extra}}}]}}"#,
        path.to_string_lossy()
    );
    MeshSpec::from_json(&text)
}

fn at(lat: f64, lon: f64) -> V3 {
    from_lat_lon(lat.to_radians(), lon.to_radians())
}

/// Bilinear in latitude and longitude: a field linear in both is reproduced
/// EXACTLY between the nodes, every node is read back, and outside the
/// extent the raster says nothing.
#[test]
fn sampling_is_bilinear_and_reads_the_nodes() {
    let lat = axis(50.0, 52.0, 5);
    let lon = vec![-4.0, -3.7, -3.1, -3.0];
    let f = |a: f64, b: f64| 10.0 + 2.0 * (a - 50.0) + 3.0 * (b + 4.0);
    let path = raster_file("bilinear.nc", &lat, &lon, f);
    let spec = spec_with(1000.0, "", &path, r#", "limit": false"#).unwrap();
    let prepared = spec.prepared();
    for &a in &lat {
        for &b in &lon {
            let h = prepared.spacing_m(at(a, b));
            assert!((h / (f(a, b) * 1000.0) - 1.0).abs() < 1e-9, "node {a},{b}: {h}");
        }
    }
    for (a, b) in [(50.25, -3.85), (51.9, -3.05), (50.5, -3.4)] {
        let h = prepared.spacing_m(at(a, b));
        assert!(
            (h / (f(a, b) * 1000.0) - 1.0).abs() < 1e-9,
            "{a},{b}: {h} vs {}",
            f(a, b) * 1000.0
        );
    }
    assert_eq!(prepared.spacing_m(at(49.9, -3.5)), 1_000_000.0);
    assert_eq!(prepared.spacing_m(at(51.0, -2.9)), 1_000_000.0);
    assert_eq!(prepared.spacing_m(at(51.0, 176.5)), 1_000_000.0);
}

/// Longitudes written 0..360 or -180..180 address the same place.
#[test]
fn longitude_convention_does_not_move_the_raster() {
    let lat = axis(10.0, 12.0, 3);
    let lon = axis(350.0, 356.0, 4);
    let path = raster_file("wrap.nc", &lat, &lon, |_, b| 10.0 + (b - 350.0));
    let spec = spec_with(1000.0, "", &path, r#", "limit": false"#).unwrap();
    let h = spec.prepared().spacing_m(at(11.0, -8.0));
    assert!((h / 12_000.0 - 1.0).abs() < 1e-9, "{h}");
}

/// A raster wider than a quarter turn is not cut down by its own quick
/// reject: the corner-based cap only bounds a box under a quarter turn, and
/// past it the far meridian edge is farther than any corner.
#[test]
fn a_near_global_raster_is_read_on_its_far_side() {
    let lat = axis(-80.0, 80.0, 17);
    let lon = axis(-180.0, 179.0, 360);
    let path = raster_file("global.nc", &lat, &lon, |_, _| 50.0);
    let spec = spec_with(100.0, "", &path, r#", "limit": false"#).unwrap();
    let p = spec.prepared();
    for (a, b) in [(0.0, 170.0), (0.0, -170.0), (75.0, 0.0), (-75.0, 90.0), (0.0, 0.0)] {
        let h = p.spacing_m(at(a, b));
        assert!((h / 50_000.0 - 1.0).abs() < 1e-12, "{a},{b}: {h}");
    }
}

/// Finer wins, both ways round.
#[test]
fn the_finer_of_raster_and_region_wins() {
    let lat = axis(0.0, 10.0, 11);
    let lon = axis(0.0, 10.0, 11);
    let path = raster_file("combine.nc", &lat, &lon, |_, _| 50.0);
    let cap = r#"{"shape": {"kind": "cap", "center_deg": [5.0, 5.0], "radius_km": 100}, "spacing_km": 20.0, "transition_km": 10.0},"#;
    let spec = spec_with(200.0, cap, &path, r#", "limit": false"#).unwrap();
    let p = spec.prepared();
    assert!((p.spacing_m(at(5.0, 5.0)) / 20_000.0 - 1.0).abs() < 1e-6);
    assert!((p.spacing_m(at(2.0, 2.0)) / 50_000.0 - 1.0).abs() < 1e-12);
    assert!((p.spacing_m(at(-20.0, -20.0)) / 200_000.0 - 1.0).abs() < 1e-12);
    assert!((spec.finest_km() - 20.0).abs() < 1e-12);
}

/// THE LIMITER'S GUARANTEES on a deliberately rough raster: the certified
/// slope is under the ceiling, no node is coarsened, the finest node is
/// untouched, a second pass moves nothing, and the gradient meter -- the
/// instrument the gates read -- never reads more than the ceiling.
#[test]
fn the_limiter_holds_the_ceiling_and_never_coarsens() {
    let lat = axis(40.0, 44.0, 41);
    let lon = axis(-6.0, 0.0, 61);
    let mut state = 12345u64;
    let mut rnd = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut v = Vec::new();
    for _ in &lat {
        for _ in &lon {
            v.push(2.0 + 58.0 * rnd().powi(3));
        }
    }
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    let path = scratch("rough.nc");
    write_density_raster(&path, &lat, &lon, &v, min, RASTER_SCHEMA).unwrap();
    let g = 0.05;
    let spec = spec_with(64.0, "", &path, &format!(r#", "max_gradient_per_cell": {g}"#)).unwrap();
    let raster = spec.rasters[0].prepared(64.0);
    assert!(raster.limiter.peak_per_cell_before > 0.5, "the fixture is not rough");
    assert!(
        raster.certified_peak_per_cell <= g,
        "certified {}",
        raster.certified_peak_per_cell
    );
    assert!(raster.limiter.nodes_lowered > 0);
    for (k, &orig) in v.iter().enumerate() {
        assert!(raster.h_m[k] <= (orig * 1000.0).min(64_000.0), "node {k} coarsened");
    }
    assert_eq!(raster.finest_m, min * 1000.0, "the finest node moved");
    let again = limit_slope(&lat, &lon, &raster.h_m, raster.limiter.axis_allowance_per_cell);
    assert_eq!(again, raster.h_m, "a second pass moved a node");

    // The meter alone, on a dense cloud over the extent. Steps that leave
    // the extent read the EDGE step, which is a separate gate (this fixture
    // is finer than its background at the edge on purpose); the limiter's
    // claim is about the raster's interior.
    let prepared = spec.prepared();
    let mut worst = 0.0f64;
    for a in axis(40.0, 44.0, 161) {
        for b in axis(-6.0, 0.0, 241) {
            let p = at(a, b);
            let h = prepared.spacing_m(p);
            let (e, n) = crate::mesh::geom::east_north(p).unwrap();
            for d in [e, n] {
                for s in [1.0, -1.0] {
                    let q = crate::mesh::geom::unit(crate::mesh::geom::add(
                        p,
                        crate::mesh::geom::scale(d, s * h / EARTH_RADIUS_M),
                    ))
                    .unwrap();
                    if !raster.contains(q) {
                        continue;
                    }
                    worst = worst.max((prepared.spacing_m(q) / h - 1.0).abs());
                }
            }
        }
    }
    assert!(worst <= g, "the meter reads {worst} against a {g} ceiling");
    assert!(worst > 0.5 * g / std::f64::consts::SQRT_2, "the limiter flattened far more than needed: {worst}");
    // And the gate reading carries the certificate (here the edge step,
    // which is larger, so the whole raster is refused by check_rasters).
    let reading = steepest_gradient_reading_of(&prepared, 10_000);
    assert!(reading.per_cell >= raster.certified_peak_per_cell);
    assert!(reading.coverage.is_complete());
    assert!(reading.saw_the_refinement());
    assert!(spec.check_rasters().is_err());
}

#[test]
fn limit_false_refuses_a_steep_raster_by_location() {
    let lat = axis(10.0, 12.0, 21);
    let lon = axis(10.0, 12.0, 21);
    let path = raster_file("steep.nc", &lat, &lon, |a, b| {
        if (a - 11.0).abs() < 0.05 && (b - 11.0).abs() < 0.05 { 1.0 } else { 40.0 }
    });
    let spec = spec_with(4.0, "", &path, r#", "limit": false"#).unwrap();
    let err = spec.check_rasters().unwrap_err().to_string();
    assert!(err.contains("\"limit\": false") && err.contains("lat 1"), "{err}");
    // The same raster limited is admitted, and records what moved.
    let spec = spec_with(4.0, "", &path, "").unwrap();
    let reports = spec.check_rasters().unwrap();
    assert!(reports[0].limiter.nodes_lowered > 0);
    assert!(reports[0].certified_peak_per_cell <= DEFAULT_MAX_GRADIENT_PER_CELL);
    assert!(reports[0].edge_step_per_cell <= DEFAULT_MAX_GRADIENT_PER_CELL);
}

#[test]
fn an_edge_finer_than_the_field_outside_is_refused() {
    let lat = axis(10.0, 11.0, 11);
    let lon = axis(10.0, 11.0, 11);
    // Uniformly 5 km against a 40 km background: the field steps 700 %.
    let path = raster_file("edge.nc", &lat, &lon, |_, _| 5.0);
    let spec = spec_with(40.0, "", &path, "").unwrap();
    let err = spec.check_rasters().unwrap_err().to_string();
    assert!(err.contains("ends finer than the field outside it"), "{err}");
    // Nested in a wide cap at the raster's own spacing, the edge meets it.
    let cap = r#"{"shape": {"kind": "cap", "center_deg": [10.5, 10.5], "radius_km": 3000}, "spacing_km": 5.0, "transition_km": 300.0},"#;
    let spec = spec_with(40.0, cap, &path, "").unwrap();
    let reports = spec.check_rasters().unwrap();
    assert!(reports[0].edge_step_per_cell < 1e-6, "{}", reports[0].edge_step_per_cell);
}

#[test]
fn schema_violations_are_refused_by_name() {
    let lat = axis(0.0, 1.0, 3);
    let lon = axis(0.0, 1.0, 3);
    let good = vec![5.0; 9];
    let with = |k: usize, x: f64| {
        let mut v = good.clone();
        v[k] = x;
        v
    };
    let cases: Vec<(&str, Vec<f64>, Vec<f64>, f64, &str, &str)> = vec![
        ("schema.nc", lat.clone(), good.clone(), 5.0, "woof-hex.density.v0", "declares schema"),
        ("nan.nc", lat.clone(), with(4, f64::NAN), 5.0, RASTER_SCHEMA, "NaN"),
        ("zero.nc", lat.clone(), with(2, 0.0), 5.0, RASTER_SCHEMA, "finite positive"),
        ("neg.nc", lat.clone(), with(0, -1.0), 5.0, RASTER_SCHEMA, "finite positive"),
        ("min.nc", lat.clone(), good.clone(), 4.0, RASTER_SCHEMA, "min_spacing_km"),
        ("desc.nc", vec![1.0, 0.5, 0.0], good.clone(), 5.0, RASTER_SCHEMA, "strictly ascending"),
        ("polar.nc", vec![80.0, 85.0, 89.0], good.clone(), 5.0, RASTER_SCHEMA, "85"),
    ];
    for (name, la, v, min, schema, want) in cases {
        let path = scratch(name);
        write_density_raster(&path, &la, &lon, &v, min, schema).unwrap();
        let err = spec_with(100.0, "", &path, "").unwrap_err().to_string();
        assert!(err.contains(want), "{name}: {err}");
    }
    let path = scratch("ok.nc");
    write_density_raster(&path, &lat, &lon, &good, 5.0, RASTER_SCHEMA).unwrap();
    spec_with(100.0, "", &path, "").unwrap();
    // A raster row that also carries spacing_km is a confusion, refused.
    let err = spec_with(100.0, "", &path, r#", "spacing_km": 3.0"#).unwrap_err().to_string();
    assert!(err.contains("spacing_km"), "{err}");
    // A ceiling steeper than the build gate is refused.
    let err = spec_with(100.0, "", &path, r#", "max_gradient_per_cell": 0.2"#)
        .unwrap_err()
        .to_string();
    assert!(err.contains("transition-band ceiling"), "{err}");
    // A raster nowhere finer than the background refines nothing.
    let err = spec_with(5.0, "", &path, "").unwrap_err().to_string();
    assert!(err.contains("refines nothing"), "{err}");
    // A sha256 pin that does not match is refused.
    let err = spec_with(100.0, "", &path, r#", "sha256": "00""#).unwrap_err().to_string();
    assert!(err.contains("pins the raster"), "{err}");
    // A missing file is refused, not skipped.
    let err = spec_with(100.0, "", &scratch("absent.nc"), "").unwrap_err().to_string();
    assert!(err.contains("cannot be read"), "{err}");
}

/// The ladder snap: the finest node lands EXACTLY on the rung, the
/// background stays the background, nothing is coarsened, and snapping a
/// snapped spec lands on the same rung.
#[test]
fn the_finest_value_snaps_to_the_ladder_affinely() {
    // 100 km in a 2-degree core, graded at 8 km per degree out to a 160 km
    // background well inside the 30-degree box.
    let lat = axis(0.0, 30.0, 61);
    let lon = axis(0.0, 30.0, 61);
    let path = raster_file("snap.nc", &lat, &lon, |a, b| {
        let r = ((a - 15.0).powi(2) + (b - 15.0).powi(2)).sqrt();
        (100.0 + 8.0 * (r - 2.0).max(0.0)).min(160.0)
    });
    let spec = spec_with(160.0, "", &path, "").unwrap();
    assert!((spec.finest_km() - 100.0).abs() < 1e-12);
    let (snapped, record) = crate::mesh::ladder_snap::snap_to_ladder(&spec);
    assert!(record.moved);
    assert_eq!(record.rasters[0].delivered_finest_km, 80.0);
    assert_eq!(snapped.rasters[0].ladder_rung_km, Some(80.0));
    assert_eq!(snapped.finest_km(), 80.0);
    assert!(crate::mesh::ladder_snap::is_on_ladder(&snapped));
    assert!(!crate::mesh::ladder_snap::is_on_ladder(&spec));
    let before = spec.rasters[0].prepared(160.0);
    let after = snapped.rasters[0].prepared(160.0);
    assert_eq!(after.finest_m, 80_000.0);
    for (k, (&b, &a)) in before.h_m.iter().zip(&after.h_m).enumerate() {
        assert!(a <= b, "node {k} coarsened by the snap: {b} -> {a}");
    }
    // The corners are far enough out that the snapped, limited grading still
    // reaches the background there.
    assert_eq!(after.node_m(0, 0), 160_000.0, "the background corner moved");
    let (twice, _) = crate::mesh::ladder_snap::snap_to_ladder(&snapped);
    assert_eq!(twice.rasters[0].ladder_rung_km, Some(80.0));
    let reports = snapped.check_rasters().unwrap();
    assert_eq!(reports[0].delivered_finest_km, 80.0);
    assert_eq!(reports[0].edge_step_per_cell, 0.0);
}

/// The spec round-trips with the raster at the position it was written, and
/// the stamped form carries the raster's digest but not its path.
#[test]
fn a_raster_spec_round_trips_and_stamps_no_path() {
    let lat = axis(0.0, 4.0, 5);
    let lon = axis(0.0, 4.0, 5);
    let path = raster_file("trip.nc", &lat, &lon, |_, _| 50.0);
    let text = format!(
        r#"{{"background_km": 200, "regions": [{{"shape": "raster", "path": {:?}}}, {{"shape": {{"kind": "cap", "center_deg": [0, 0], "radius_km": 100}}, "spacing_km": 100, "transition_km": 50}}]}}"#,
        path.to_string_lossy()
    );
    let spec = MeshSpec::from_json(&text).unwrap();
    assert_eq!(spec.rasters[0].position, 0);
    let json = serde_json::to_value(&spec).unwrap();
    let rows = json["regions"].as_array().unwrap();
    assert_eq!(rows[0]["shape"], "raster");
    assert_eq!(rows[1]["shape"]["kind"], "cap");
    assert_eq!(rows[0]["finest_km"], 50.0);
    let again = MeshSpec::from_json(&serde_json::to_string(&spec).unwrap()).unwrap();
    assert_eq!(again.rasters[0].position, 0);
    let stamped = crate::mesh::density::provenance_spec_json(&spec).unwrap();
    assert!(!stamped.contains("trip.nc"), "{stamped}");
    assert!(stamped.contains(&spec.rasters[0].source().sha256));
    // A relative path is read against the base the caller names.
    let rel = r#"{"background_km": 200, "regions": [{"shape": "raster", "path": "trip.nc"}]}"#;
    let spec = MeshSpec::from_json_at(rel, path.parent()).unwrap();
    assert_eq!(spec.rasters[0].source().sha256, again.rasters[0].source().sha256);
    // fit-spacing cannot rescale a raster.
    let err = spec.fitted_to(1000, 1000).unwrap_err().to_string();
    assert!(err.contains("cannot be rescaled"), "{err}");
}

/// The sizing integral counts a raster's cells exactly: a constant raster
/// over a box predicts the box at the raster spacing plus the rest of the
/// sphere at the background -- where the lattice alone misses it.
#[test]
fn the_sizing_integral_counts_the_raster_cells() {
    let lat = axis(50.0, 52.0, 9);
    let lon = axis(-5.0, -2.0, 13);
    let r_km = 2.0;
    let path = raster_file("count.nc", &lat, &lon, |_, _| r_km);
    let bg = 200.0;
    let spec = spec_with(bg, "", &path, r#", "limit": false"#).unwrap();
    let rr = EARTH_RADIUS_M / 1000.0;
    let box_area =
        rr * rr * 3f64.to_radians() * (52f64.to_radians().sin() - 50f64.to_radians().sin());
    let sphere = 4.0 * std::f64::consts::PI * rr * rr;
    let hex = |h: f64| 3f64.sqrt() / 2.0 * h * h;
    let want = box_area / hex(r_km) + (sphere - box_area) / hex(bg);
    let got = spec.predicted_cells(200_000);
    assert!((got / want - 1.0).abs() < 2e-3, "predicted {got:.0}, exact {want:.0}");
    // At a fine raster spacing the box dominates the count, and the lattice
    // alone -- a couple of points inside the box -- is off by whatever those
    // points happen to be. The exact arm is not.
    let (small_lat, small_lon) = (axis(51.0, 51.2, 5), axis(-3.4, -3.1, 7));
    let fine = raster_file("count-fine.nc", &small_lat, &small_lon, |_, _| 0.25);
    let spec = spec_with(bg, "", &fine, r#", "limit": false"#).unwrap();
    let box_area =
        rr * rr * 0.3f64.to_radians() * (51.2f64.to_radians().sin() - 51f64.to_radians().sin());
    let want = box_area / hex(0.25) + (sphere - box_area) / hex(bg);
    let got = spec.predicted_cells(200_000);
    assert!((got / want - 1.0).abs() < 2e-3, "predicted {got:.0}, exact {want:.0}");
    let p = spec.prepared();
    let mut acc = 0.0;
    for q in crate::mesh::density::fibonacci_lattice(200_000) {
        let h = p.spacing_m(q) / 1000.0;
        acc += 1.0 / (h * h);
    }
    let lattice_only = sphere * acc / 200_000.0 / (3f64.sqrt() / 2.0);
    assert!(
        (lattice_only / want - 1.0).abs() > 0.05,
        "the lattice happened to agree ({lattice_only:.0} vs {want:.0}); the test proves nothing"
    );
}

/// A long thin raster (a corridor) keeps complete coverage: the raster
/// probes are decimated rather than reported unmeasured.
#[test]
fn a_long_thin_raster_is_not_falsely_refused() {
    let lat = axis(51.0, 51.2, 41);
    let lon = axis(-10.0, 10.0, 8001);
    let path = raster_file("thin.nc", &lat, &lon, |a, _| 1.0 + 200.0 * (a - 51.1).abs());
    let spec = spec_with(64.0, "", &path, "").unwrap();
    let prepared = spec.prepared();
    let mut probes = Vec::new();
    let coverage = prepared.variation_probes(crate::mesh::density::PROBE_BUDGET, &mut probes);
    assert!(coverage.is_complete(), "{coverage:?}");
    assert!(probes.len() <= RASTER_PROBE_CAP + 2, "{} probes", probes.len());
    assert!(DensityField::certified_peak_per_cell(&prepared) > 0.0);
}
