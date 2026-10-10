//! The per-variable method table.
//!
//! One row per init-stream variable the remap computes.  A variable not in
//! this table is never remapped: when the target template declares it, it is
//! carried from the template (statics, mesh, vertical metrics) and the emit
//! ledger lists it as carried.  A source variable with a state shape that is
//! not in the table is named in the receipt as not remapped, so nothing is
//! dropped without a word.

use crate::remap::column::Extend;

/// How one variable crosses from the source mesh to the target mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// `(nCells, nVertLevels)` dry density: conservative in the horizontal
    /// on layer mass `rho dz`, then conservative (PLM) in height.
    Density,
    /// `(nCells, nVertLevels)` mass-specific scalar: conservative on
    /// `rho dz s` in the horizontal, `∫rho s dz / ∫rho dz` in height.
    MassWeighted(Extend),
    /// `(nCells, nVertLevelsP1)` vertical velocity: area-conservative on the
    /// source interfaces, linear in height onto the target interfaces, zero
    /// at the surface and the lid as the init writes it.
    Interface,
    /// `(nCells, nVertLevels)` diagnostic: barycentric on the source dual
    /// triangles, linear in height.
    BarycentricColumn,
    /// `(nCells)` or `(nCells, nSoilLevels)`: area-conservative.
    AreaConservative,
    /// `(nCells, nSoilLevels)` soil state: area-conservative over source
    /// cells of the target cell's own surface type (land or water), falling
    /// back to all overlapping cells where none of that type overlap.
    MaskedConservative,
    /// `(nCells)` diagnostic: barycentric.
    Barycentric,
    /// `(nCells)` or `(nCells, nSoilLevels)` category or flag: the value of
    /// the source cell with the largest overlap.
    Dominant,
    /// `(nEdges, nVertLevels)` edge-normal wind: cell-vector reconstruction,
    /// mass-weighted remap of the components, projection on target normals.
    EdgeNormalWind,
    /// Computed on the target from the remapped state: the base state,
    /// surface pressure, precipitable water, the land-sea index.
    Derived,
    /// `NC_CHAR` identity label.
    Label,
}

impl Method {
    pub fn label(self) -> &'static str {
        match self {
            Method::Density => "conservative-layer-mass+plm-height",
            Method::MassWeighted(Extend::LinearBelow) => {
                "conservative-mass-weighted+plm-height(linear-below)"
            }
            Method::MassWeighted(Extend::Constant) => "conservative-mass-weighted+plm-height",
            Method::MassWeighted(Extend::PositiveConstant) => {
                "conservative-mass-weighted+plm-height(positive)"
            }
            Method::Interface => "conservative-area+linear-height(zero-at-surface-and-lid)",
            Method::BarycentricColumn => "barycentric+linear-height",
            Method::AreaConservative => "conservative-area",
            Method::MaskedConservative => "conservative-area-masked-by-surface-type",
            Method::Barycentric => "barycentric",
            Method::Dominant => "dominant-overlap",
            Method::EdgeNormalWind => "cell-vector-reconstruct+conservative-momentum+edge-projection",
            Method::Derived => "derived-on-target",
            Method::Label => "identity-label",
        }
    }
}

/// Hydrometeor and aerosol mixing ratios and number concentrations every
/// microphysics row in the v8.4.1 registry can carry.
pub const POSITIVE_SCALARS: &[&str] = &[
    "qv", "qc", "qr", "qi", "qs", "qg", "qh", "nc", "ni", "nr", "ns", "ng", "nh", "nifa",
    "nwfa", "volg", "tke",
];

/// The water species summed into the total-water budget.
pub const WATER_SPECIES: &[&str] = &["qv", "qc", "qr", "qi", "qs", "qg", "qh"];

/// The table.
pub fn method_for(name: &str) -> Option<Method> {
    if POSITIVE_SCALARS.contains(&name) {
        return Some(Method::MassWeighted(Extend::PositiveConstant));
    }
    Some(match name {
        "rho" => Method::Density,
        "theta" => Method::MassWeighted(Extend::LinearBelow),
        "w" => Method::Interface,
        "u" => Method::EdgeNormalWind,
        "relhum" => Method::BarycentricColumn,
        "skintemp" | "sst" | "tmn" | "snow" | "snowh" | "xice" | "vegfra" | "sfc_albbck"
        | "h_oml_initial" => Method::AreaConservative,
        "tslb" | "smois" | "sh2o" => Method::MaskedConservative,
        "t2m" | "q2" | "rh2" | "u10" | "v10" => Method::Barycentric,
        "seaice" | "snowc" | "dzs" | "zs" | "dz" => Method::Dominant,
        "rho_base" | "theta_base" | "surface_pressure" | "precipw" | "xland" => Method::Derived,
        "xtime" | "initial_time" => Method::Label,
        _ => return None,
    })
}

/// The dycore's working state a restart carries and an init does not.  A
/// target template declaring any of these is a restart-class template, and
/// the remap refuses it by name rather than write numbers whose coupling
/// conventions it does not rebuild: the forecast reconstructs every one of
/// them from `theta`, `rho`, `qv`, `u` and `w` at start-up.
pub const RESTART_ONLY: &[&str] = &[
    "rho_zz", "theta_m", "pressure_p", "pressure_base", "exner", "exner_base", "rho_p",
    "rtheta_p", "rtheta_base", "ru", "rw", "ru_p", "rw_p",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prognostic_state_is_conservative_and_diagnostics_are_barycentric() {
        assert_eq!(method_for("rho"), Some(Method::Density));
        assert!(matches!(method_for("theta"), Some(Method::MassWeighted(_))));
        assert!(matches!(method_for("qv"), Some(Method::MassWeighted(Extend::PositiveConstant))));
        assert_eq!(method_for("relhum"), Some(Method::BarycentricColumn));
        assert_eq!(method_for("t2m"), Some(Method::Barycentric));
        assert_eq!(method_for("tslb"), Some(Method::MaskedConservative));
        assert_eq!(method_for("u"), Some(Method::EdgeNormalWind));
    }

    #[test]
    fn statics_are_not_in_the_table() {
        for name in ["ter", "landmask", "ivgtyp", "zgrid", "latCell", "areaCell"] {
            assert_eq!(method_for(name), None, "{name}");
        }
    }
}
