//! Soil columns of the WRF land surfaces whose geometry MPAS init has no row
//! for: the 5-layer thermal-diffusion slab (`sf_surface_physics = 1`) and the
//! 2-layer Pleim-Xiu LSM (`sf_surface_physics = 7`), as WRF v4.7.1 real.exe
//! builds them (share/module_soil_pre.F: `process_soil_real`,
//! `init_soil_depth_1` / `init_soil_1_real`, `init_soil_depth_7` /
//! `init_soil_7_real`).  Layer sources only (WPS `FLAG_SOIL_LAYERS`: GFS,
//! ERA5); the same integer centimetre mid-depths as the RUC branch.
use crate::error::{MpasError, MpasResult};

/// `init_soil_depth_1`: top layer 1 cm, each layer twice the one above.
pub fn slab_geometry() -> (Vec<f32>, Vec<f32>) {
    let mut dzs = vec![0.0f32; 5];
    let mut zs = vec![0.0f32; 5];
    dzs[0] = 0.01;
    zs[0] = 0.5 * dzs[0];
    for l in 1..5 {
        dzs[l] = 2.0 * dzs[l - 1];
        zs[l] = zs[l - 1] + 0.5 * dzs[l - 1] + 0.5 * dzs[l];
    }
    (zs, dzs)
}

/// `init_soil_depth_7`: 1 cm over 99 cm.
pub fn px_geometry() -> (Vec<f32>, Vec<f32>) {
    let dzs = vec![0.01f32, 0.99f32];
    let zs = vec![0.5 * dzs[0], dzs[0] + 0.5 * dzs[1]];
    (zs, dzs)
}

/// `process_soil_real`'s `fix_bottom_level_for_temp` for SLABSCHEME and
/// PXLSMSCHEME (no average-surface-temperature input): the deep temperature is
/// the source soil temperature whose level is closest to 30 cm (the first on a
/// tie: a strict `<`).  Returns the source layer index.
pub const TMN_CLOSEST_TO_CM: f32 = 30.0;
pub fn closest_to_30cm(mid_cm: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (k, &z) in mid_cm.iter().enumerate() {
        let diff = (z - TMN_CLOSEST_TO_CM).abs();
        if best.map_or(true, |(_, d)| diff < d) {
            best = Some((k, diff));
        }
    }
    best.map(|(k, _)| k)
}

/// `init_soil_1_real`, one land column: linear in depth between the skin
/// temperature at the top layer's mid-depth and the deep temperature at the
/// bottom layer's, in WRF's expression order.
pub fn slab_column(zs: &[f32], tsk: f32, tmn: f32) -> Vec<f32> {
    let n = zs.len();
    zs.iter()
        .map(|&z| (tsk * (zs[n - 1] - z) + tmn * (z - zs[0])) / (zs[n - 1] - zs[0]))
        .collect()
}

fn interpolate(zhave: &[f32], values: &[f32], zs: &[f32]) -> MpasResult<Vec<f32>> {
    zs.iter()
        .map(|&z| {
            let k = (0..zhave.len() - 1)
                .find(|&k| z >= zhave[k] && z <= zhave[k + 1])
                .ok_or_else(|| MpasError::Refusal(format!("soil depth {z} m has no source bracket")))?;
            Ok((values[k] * (zhave[k + 1] - z) + values[k + 1] * (z - zhave[k]))
                / (zhave[k + 1] - zhave[k]))
        })
        .collect()
}

/// `init_soil_7_real`, one land column from a layer source.  The moisture
/// samples are the source layers with the top one repeated at 0 m and the
/// deepest at 3 m, interpolated to the PX mid-depths.  The temperatures are
/// interpolated the same way (skin at 0 m, deep temperature at 3 m) and then
/// overwritten exactly as the routine ends: layer 1 is the skin temperature
/// and layer 2 the deep temperature.  Returns `(tslb, smois)`.
pub fn px_column(zs: &[f32], mid_cm: &[f32], sm: &[f32], tsk: f32, tmn: f32)
    -> MpasResult<(Vec<f32>, Vec<f32>)> {
    if mid_cm.len() < 2 || sm.len() != mid_cm.len() {
        return Err(MpasError::Refusal(
            "Not enough soil temperature data for P-X LSM scheme (two source layers needed)".into()));
    }
    let mut zhave = Vec::with_capacity(mid_cm.len() + 2);
    zhave.push(0.0f32);
    zhave.extend(mid_cm.iter().map(|z| z / 100.0f32));
    zhave.push(300.0f32 / 100.0f32);
    let mut m = Vec::with_capacity(zhave.len());
    m.push(sm[0]);
    m.extend_from_slice(sm);
    m.push(sm[sm.len() - 1]);
    let smois = interpolate(&zhave, &m, zs)?;
    let mut tslb = vec![0.0f32; zs.len()];
    tslb[0] = tsk;
    tslb[1] = tmn;
    Ok((tslb, smois))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slab_layers_are_wrfs_doubling_column() {
        let (zs, dzs) = slab_geometry();
        assert_eq!(dzs, vec![0.01, 0.02, 0.04, 0.08, 0.16]);
        assert!((zs[4] - 0.23).abs() < 1e-6 && (zs[0] - 0.005).abs() < 1e-9);
    }

    #[test]
    fn px_layers_are_one_and_ninety_nine_centimetres() {
        let (zs, dzs) = px_geometry();
        assert_eq!(dzs, vec![0.01, 0.99]);
        assert_eq!(zs, vec![0.005, 0.505]);
    }

    #[test]
    fn the_deep_temperature_is_the_source_layer_nearest_30_cm() {
        assert_eq!(closest_to_30cm(&[5.0, 25.0, 70.0, 150.0]), Some(1));
        // a tie keeps the shallower layer (strict comparison)
        assert_eq!(closest_to_30cm(&[20.0, 40.0]), Some(0));
    }

    #[test]
    fn slab_ends_at_skin_and_deep_temperature() {
        let (zs, _) = slab_geometry();
        let t = slab_column(&zs, 300.0, 290.0);
        assert_eq!(t[0], 300.0);
        assert_eq!(t[4], 290.0);
    }

    #[test]
    fn px_temperatures_are_skin_and_deep_and_moisture_is_interpolated() {
        let (zs, _) = px_geometry();
        let (t, m) = px_column(&zs, &[5.0, 25.0, 70.0, 150.0], &[0.2, 0.25, 0.3, 0.35], 301.0, 289.0).unwrap();
        assert_eq!(t, vec![301.0, 289.0]);
        assert!((m[0] - 0.2).abs() < 1e-6);   // 0.5 cm lies between the repeated top and 5 cm
        assert!((m[1] - (0.25 * (0.70 - 0.505) + 0.3 * (0.505 - 0.25)) / (0.70 - 0.25)).abs() < 1e-6);
    }
}
