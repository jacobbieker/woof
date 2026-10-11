//! Two init-time ceilings that native MPAS `init_atmosphere` does not apply,
//! each reported with a count, the cap value and the largest value it met.
//!
//! ## Snow water equivalent: 5000 kg m-2
//!
//! Some analyses publish the snow on an ice sheet as a store of tens of metres
//! of water equivalent: Met Office Global carries about 51,800 kg m-2 over
//! Greenland and ICON up to about 15,200 kg m-2.  Case 7 takes the snow depth
//! as `snow * 5 / 1000`, so 51,800 kg m-2 starts the land model under 259 m of
//! snow.  The ceiling is the one WRF's Noah-MP initialisation applies
//! (`module_sf_noahmpdrv.F`, `NOAHMP_INIT`: "cap SWE at 5000mm ... cap SNOW
//! at 5000, maintain density"), and the same 5000 mm is Noah-MP's own glacier
//! equilibrium in `SNOWWATER`.  As there, the snow depth is scaled by the same
//! factor so the snow density is unchanged.  It applies at every cell, after
//! the sea-ice step has cleared the snow off open water.
//!
//! ## Soil moisture: the cell's own SMCMAX
//!
//! Volumetric soil moisture above the porosity of the cell's soil is water
//! the column cannot hold.  ICON's peat (its soil type 8, porosity 0.863)
//! reaches 0.86, and an interpolated value from a source cell of one texture
//! can sit above the porosity of the target cell's texture.  The ceiling is
//! `MAXSMC` for the cell's soil category in `SOILPARM.TBL`'s `STAS` block, the
//! table the Noah land model reads, and the rule is `NOAHMP_INIT`'s:
//! `IF ( SMOIS > SMCMAX ) SMOIS = SMCMAX` on every land cell that is not land
//! ice, whose column the land model treats as glacier.  Land ice is
//! `NOAHMP_INIT`'s `ivgtyp == isice` or soil category 16 (`STAS`'s land-ice
//! row), the test the WRF path makes from the soil category alone
//! (`woof/ingest/soil.py`), so the two paths skip the same cells.

use std::collections::BTreeMap;

/// `SOILPARM.TBL`'s land-ice soil category.
pub const LAND_ICE_SOIL_CATEGORY: i32 = 16;

/// The snow water equivalent ceiling, kg m-2.
pub const SNOW_WATER_CAP_KG_M2: f32 = 5000.0;

/// `MAXSMC` for soil categories 1..=19 of `SOILPARM.TBL`'s `STAS` block, in
/// table order.  Category 14 is water (1.0) and 16 is land ice.  A test reads
/// the packaged table and requires these to match it.
pub const STAS_MAXSMC: [f32; 19] = [
    0.339, 0.421, 0.434, 0.476, 0.484, 0.439, 0.404, 0.464, 0.465, 0.406, 0.468, 0.468, 0.439, 1.0,
    0.20, 0.421, 0.468, 0.200, 0.339,
];

/// What the snow ceiling did.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct SnowCap {
    pub cap_kg_m2: f32,
    /// Cells whose snow water equivalent was above the cap and was set to it.
    pub cells_capped: usize,
    /// The largest snow water equivalent before the cap, kg m-2.
    pub source_max_kg_m2: f32,
    /// The largest snow depth before the cap, m.
    pub source_max_snowh_m: f32,
    pub rule: String,
    /// The one-line message the run logs; empty when no cell was capped.
    pub note: String,
}

/// Capped cells of one soil category.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct CategoryCap {
    pub smcmax: f32,
    pub cells: usize,
    pub layers: usize,
    /// The largest soil moisture capped in this category, m3 m-3.
    pub source_max: f32,
}

/// What the soil moisture ceiling did.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct SoilMoistureCap {
    pub table: String,
    /// Land cells with at least one layer above their category's SMCMAX.
    pub cells_capped: usize,
    /// Soil layers set to SMCMAX, over all capped cells.
    pub layers_capped: usize,
    /// The largest soil moisture capped, m3 m-3.
    pub source_max: f32,
    /// Land-ice cells (`ivgtyp == isice` or soil category 16) the rule
    /// leaves alone.
    pub land_ice_cells_skipped: usize,
    /// Land cells whose soil category is outside 1..=19, left alone.
    pub cells_without_category: usize,
    /// Keyed by soil category.
    pub by_category: BTreeMap<String, CategoryCap>,
    pub rule: String,
    /// The one-line message the run logs; empty when no cell was capped.
    pub note: String,
}

/// Both ceilings, as the receipt carries them.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct InitCaps {
    pub snow: SnowCap,
    pub soil_moisture: SoilMoistureCap,
}

impl InitCaps {
    /// The one-line messages for the ceilings that capped a cell.
    pub fn notes(&self) -> Vec<&str> {
        [self.snow.note.as_str(), self.soil_moisture.note.as_str()]
            .into_iter()
            .filter(|note| !note.is_empty())
            .collect()
    }
}

/// Cap the snow water equivalent at [`SNOW_WATER_CAP_KG_M2`], scaling the snow
/// depth by the same factor.  Values at or below the cap are untouched.
pub fn cap_snow(snow: &mut [f32], snowh: &mut [f32]) -> SnowCap {
    let cap = SNOW_WATER_CAP_KG_M2;
    let mut out = SnowCap {
        cap_kg_m2: cap,
        rule: "SNOW > 5000 kg m-2: SNOWH = SNOWH * 5000 / SNOW, SNOW = 5000 (WRF \
               module_sf_noahmpdrv.F NOAHMP_INIT, density kept)"
            .to_string(),
        ..SnowCap::default()
    };
    for c in 0..snow.len() {
        out.source_max_kg_m2 = out.source_max_kg_m2.max(snow[c]);
        out.source_max_snowh_m = out.source_max_snowh_m.max(snowh[c]);
        if snow[c] > cap {
            snowh[c] *= cap / snow[c];
            snow[c] = cap;
            out.cells_capped += 1;
        }
    }
    if out.cells_capped > 0 {
        out.note = format!(
            "rw_mpas_init: capped SNOW at {cap} kg m-2 on {} cell(s) (source value up to {} \
             kg m-2, snow depth up to {} m), snow depth scaled to keep the density; WRF \
             NOAHMP_INIT's ceiling",
            out.cells_capped, out.source_max_kg_m2, out.source_max_snowh_m
        );
    }
    out
}

/// Cap each land, non-land-ice cell's soil moisture at its soil category's
/// `STAS` `MAXSMC`.  The liquid part is held at or below the same ceiling.
pub fn cap_soil_moisture(
    landmask: &[i32],
    ivgtyp: &[i32],
    isice_lu: i32,
    isltyp: &[i32],
    smois: &mut [[f32; 4]],
    sh2o: &mut [[f32; 4]],
) -> SoilMoistureCap {
    let mut out = SoilMoistureCap {
        table: "SOILPARM.TBL STAS MAXSMC".to_string(),
        rule: "SMOIS > SMCMAX(isltyp): SMOIS = SMCMAX on land cells that are not land ice \
               (WRF module_sf_noahmpdrv.F NOAHMP_INIT)"
            .to_string(),
        ..SoilMoistureCap::default()
    };
    for c in 0..landmask.len() {
        if landmask[c] != 1 {
            continue;
        }
        if ivgtyp[c] == isice_lu || isltyp[c] == LAND_ICE_SOIL_CATEGORY {
            out.land_ice_cells_skipped += 1;
            continue;
        }
        let category = isltyp[c];
        if !(1..=STAS_MAXSMC.len() as i32).contains(&category) {
            out.cells_without_category += 1;
            continue;
        }
        let smcmax = STAS_MAXSMC[(category - 1) as usize];
        let mut layers = 0usize;
        let mut largest = 0.0f32;
        for l in 0..4 {
            if smois[c][l] > smcmax {
                largest = largest.max(smois[c][l]);
                smois[c][l] = smcmax;
                layers += 1;
            }
            if sh2o[c][l] > smcmax {
                sh2o[c][l] = smcmax;
            }
        }
        if layers > 0 {
            out.cells_capped += 1;
            out.layers_capped += layers;
            out.source_max = out.source_max.max(largest);
            let entry = out
                .by_category
                .entry(category.to_string())
                .or_insert(CategoryCap {
                    smcmax,
                    ..CategoryCap::default()
                });
            entry.cells += 1;
            entry.layers += layers;
            entry.source_max = entry.source_max.max(largest);
        }
    }
    if out.cells_capped > 0 {
        let ceilings: Vec<String> = out
            .by_category
            .iter()
            .map(|(category, cap)| {
                format!("category {} at {} on {}", category, cap.smcmax, cap.cells)
            })
            .collect();
        out.note = format!(
            "rw_mpas_init: capped soil moisture at the cell's SOILPARM.TBL STAS MAXSMC on {} \
             land cell(s), {} layer(s) (source value up to {} m3 m-3; {}); WRF NOAHMP_INIT's \
             ceiling",
            out.cells_capped,
            out.layers_capped,
            out.source_max,
            ceilings.join(", ")
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snow_below_at_and_above_the_cap() {
        let mut snow = vec![0.0, 4999.0, 5000.0, 5001.0, 51_800.0];
        let mut snowh: Vec<f32> = snow.iter().map(|s| s * 5.0 / 1000.0).collect();
        let before_h = snowh.clone();
        let cap = cap_snow(&mut snow, &mut snowh);
        assert_eq!(snow, vec![0.0, 4999.0, 5000.0, 5000.0, 5000.0]);
        // Below and at the cap, untouched to the bit.
        assert_eq!(&snowh[..3], &before_h[..3]);
        // Above it the density (snow / snowh) is kept.
        for c in 3..5 {
            assert!((snowh[c] - 25.0).abs() < 1e-4, "{}", snowh[c]);
        }
        assert_eq!(cap.cells_capped, 2);
        assert_eq!(cap.cap_kg_m2, 5000.0);
        assert_eq!(cap.source_max_kg_m2, 51_800.0);
        assert!((cap.source_max_snowh_m - 259.0).abs() < 1e-3);
        assert!(cap
            .note
            .starts_with("rw_mpas_init: capped SNOW at 5000 kg m-2 on 2 cell(s)"));
        assert!(cap.note.contains("source value up to 51800 kg m-2"));
    }

    #[test]
    fn snow_cap_is_silent_when_nothing_is_above_it() {
        let mut snow = vec![0.0, 415.0, 5000.0];
        let mut snowh = vec![0.0, 2.075, 25.0];
        let cap = cap_snow(&mut snow, &mut snowh);
        assert_eq!(cap.cells_capped, 0);
        assert!(cap.note.is_empty());
        assert_eq!(snow, vec![0.0, 415.0, 5000.0]);
        assert_eq!(snowh, vec![0.0, 2.075, 25.0]);
    }

    #[test]
    fn soil_moisture_below_at_and_above_smcmax() {
        // Category 13 (organic material) has SMCMAX 0.439.
        let landmask = vec![1, 1, 1, 1, 0, 1, 1, 1];
        let ivgtyp = vec![10, 10, 10, 15, 17, 10, 10, 15];
        let isltyp = vec![13, 13, 13, 16, 14, 0, 16, 13];
        let mut smois = vec![
            [0.30, 0.43, 0.438, 0.20],
            [0.439, 0.439, 0.439, 0.439],
            [0.86, 0.70, 0.44, 0.30],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.9, 0.9, 0.9, 0.9],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ];
        let mut sh2o = vec![[0.0f32; 4]; 8];
        sh2o[2] = [0.86, 0.70, 0.44, 0.30];
        let before = smois.clone();
        let cap = cap_soil_moisture(&landmask, &ivgtyp, 15, &isltyp, &mut smois, &mut sh2o);
        assert_eq!(smois[0], before[0]);
        assert_eq!(smois[1], before[1]);
        assert_eq!(smois[2], [0.439, 0.439, 0.439, 0.30]);
        assert_eq!(sh2o[2], [0.439, 0.439, 0.439, 0.30]);
        // Land ice, water and an out-of-table category are left alone.
        assert_eq!(smois[3], before[3]);
        assert_eq!(smois[4], before[4]);
        assert_eq!(smois[5], before[5]);
        // Land ice by soil category alone, and by land use alone.
        assert_eq!(smois[6], before[6]);
        assert_eq!(smois[7], before[7]);
        assert_eq!(cap.cells_capped, 1);
        assert_eq!(cap.layers_capped, 3);
        assert_eq!(cap.source_max, 0.86);
        assert_eq!(cap.land_ice_cells_skipped, 3);
        assert_eq!(cap.cells_without_category, 1);
        assert_eq!(cap.by_category["13"].smcmax, 0.439);
        assert_eq!(cap.by_category["13"].cells, 1);
        assert!(cap.note.contains("on 1 land cell(s), 3 layer(s)"));
        assert!(cap.note.contains("source value up to 0.86"));
        assert!(cap.note.contains("category 13 at 0.439 on 1"));
    }

    #[test]
    fn stas_maxsmc_matches_the_packaged_soilparm_table() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../woof/data/noah_tables/SOILPARM.TBL");
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("skipped: {} is not beside this crate", path.display());
            return;
        };
        let mut lines = text
            .lines()
            .skip_while(|line| line.trim() != "STAS")
            .skip(2);
        for (index, expected) in STAS_MAXSMC.iter().enumerate() {
            let row = lines.next().expect("STAS row");
            let cells: Vec<&str> = row.split(',').map(str::trim).collect();
            assert_eq!(cells[0].parse::<usize>().unwrap(), index + 1);
            let maxsmc: f32 = cells[4].parse().unwrap();
            assert_eq!(maxsmc, *expected, "category {}", index + 1);
        }
    }
}
