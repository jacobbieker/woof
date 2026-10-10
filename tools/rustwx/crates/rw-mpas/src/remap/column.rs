//! The vertical half of a remap: one column, source layers onto target
//! layers.
//!
//! After the horizontal pass every target cell holds a column on the
//! SOURCE's layers: the source's interface heights, layer masses and
//! mass-weighted scalars, averaged over the source cells it overlaps.  The
//! target's own layers differ (different terrain at a different resolution,
//! possibly a different level set), so the column is remapped in height.
//!
//! ## Conservative, and exact on linear profiles
//! Each source layer carries a piecewise-linear profile (PLM) whose layer
//! mean is the layer's own value, with the slope limited by the
//! monotonised-central rule.  Integrating that profile over each target
//! layer conserves the column integral wherever the two columns overlap, and
//! reproduces a linear profile exactly, because the MC limiter leaves a
//! linear profile's slopes alone.
//!
//! Density is remapped as `∫ρ dz`.  A mass-specific scalar `s` (`theta`,
//! the mixing ratios, a momentum component) is remapped as `∫ρ s dz / ∫ρ dz`
//! with both profiles linear in the layer, the scalar's intercept corrected
//! so the layer's MASS-weighted mean stays its value; that is what makes the
//! water and momentum budgets close with the dry-air one.
//!
//! ## Outside the source column
//! A target whose terrain is lower than the source's reaches below the
//! source column, and one with a higher model top reaches above it.  Below,
//! density and `theta` extend the lowest layer's gradient (a column is not
//! isothermal near the ground) and positive-definite scalars extend
//! constant; above, every field extends constant.  How far each column
//! reached is counted so the receipt can say how much of the result was
//! extrapolated rather than remapped.

#![allow(clippy::needless_range_loop)]

use crate::init::dynamics::{self, constants::*, ColumnState, VirtualFactor};

/// How a field is continued past the ends of the source column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extend {
    /// Linear below (the lowest layer's gradient), constant above.
    LinearBelow,
    /// Constant both ways (a momentum component).
    Constant,
    /// Constant both ways, and the profile is kept non-negative.
    PositiveConstant,
}

/// The source column, extended to span the target column.
#[derive(Debug, Clone)]
pub struct Extended {
    /// Layer interfaces, ascending, `n + 1` of them.
    pub z: Vec<f64>,
    /// Index of the first real source layer in `z` (1 when a layer was added
    /// below, else 0).
    pub offset: usize,
    pub n_source: usize,
    pub below_m: f64,
    pub above_m: f64,
}

impl Extended {
    pub fn new(src_z: &[f64], tgt_z: &[f64]) -> Extended {
        let n = src_z.len() - 1;
        let mut z = Vec::with_capacity(n + 3);
        let mut offset = 0;
        let below_m = (src_z[0] - tgt_z[0]).max(0.0);
        let above_m = (tgt_z[tgt_z.len() - 1] - src_z[n]).max(0.0);
        if below_m > 0.0 {
            z.push(tgt_z[0]);
            offset = 1;
        }
        z.extend_from_slice(src_z);
        if above_m > 0.0 {
            z.push(tgt_z[tgt_z.len() - 1]);
        }
        Extended {
            z,
            offset,
            n_source: n,
            below_m,
            above_m,
        }
    }

    pub fn n_layers(&self) -> usize {
        self.z.len() - 1
    }

    fn mid(&self, k: usize) -> f64 {
        0.5 * (self.z[k] + self.z[k + 1])
    }

    /// Extend a source-layer field onto this column's layers.
    pub fn extend(&self, src: &[f64], how: Extend) -> Vec<f64> {
        let n = self.n_source;
        let mut out = Vec::with_capacity(self.n_layers());
        if self.offset == 1 {
            let v = match how {
                Extend::LinearBelow if n >= 2 => {
                    let z0 = self.mid(1);
                    let z1 = self.mid(2);
                    let g = (src[1] - src[0]) / (z1 - z0);
                    let v = src[0] + g * (self.mid(0) - z0);
                    // A linear extension of a positive field never goes
                    // below a tenth of the value it extends.
                    if src[0] > 0.0 {
                        v.max(0.1 * src[0])
                    } else {
                        v
                    }
                }
                _ => src[0],
            };
            out.push(v);
        }
        out.extend_from_slice(src);
        if out.len() < self.n_layers() {
            out.push(src[n - 1]);
        }
        out
    }

    /// MC-limited slopes of a layer-mean field on this column.
    pub fn slopes(&self, v: &[f64], positive: bool) -> Vec<f64> {
        let n = self.n_layers();
        let mut s = vec![0.0f64; n];
        if n < 2 {
            return s;
        }
        let zc: Vec<f64> = (0..n).map(|k| self.mid(k)).collect();
        for k in 0..n {
            let h = self.z[k + 1] - self.z[k];
            let slope = if k == 0 {
                (v[1] - v[0]) / (zc[1] - zc[0])
            } else if k == n - 1 {
                (v[k] - v[k - 1]) / (zc[k] - zc[k - 1])
            } else {
                let left = (v[k] - v[k - 1]) / (zc[k] - zc[k - 1]);
                let right = (v[k + 1] - v[k]) / (zc[k + 1] - zc[k]);
                if left * right <= 0.0 {
                    0.0
                } else {
                    let centred = (v[k + 1] - v[k - 1]) / (zc[k + 1] - zc[k - 1]);
                    centred.signum() * centred.abs().min(2.0 * left.abs()).min(2.0 * right.abs())
                }
            };
            s[k] = if positive && h > 0.0 {
                // Keep the profile non-negative across the layer.
                let cap = 2.0 * v[k].max(0.0) / h;
                slope.clamp(-cap, cap)
            } else {
                slope
            };
        }
        s
    }
}

/// `∫ρ dz` over every target layer, and the extension's density profile.
pub struct DensityProfile {
    pub mean: Vec<f64>,
    pub slope: Vec<f64>,
}

/// Remap density.  `src_rho` are source layer means (`mass / dz`).  Returns
/// target layer masses (`∫ρ dz`, kg m^-2) and the profile used, which the
/// mass-weighted scalars need.
pub fn remap_density(ext: &Extended, src_rho: &[f64], tgt_z: &[f64]) -> (Vec<f64>, DensityProfile) {
    let mean = ext.extend(src_rho, Extend::LinearBelow);
    let slope = ext.slopes(&mean, true);
    let nt = tgt_z.len() - 1;
    let mut mass = vec![0.0f64; nt];
    sweep(ext, tgt_z, |t, k, a, b| {
        let zc = ext.mid(k);
        let (a, b) = (a - zc, b - zc);
        mass[t] += mean[k] * (b - a) + slope[k] * (b * b - a * a) * 0.5;
    });
    (mass, DensityProfile { mean, slope })
}

/// Remap a mass-specific scalar given the density profile.  Returns the
/// target layer values (`∫ρ s dz / ∫ρ dz`).
pub fn remap_scalar(
    ext: &Extended,
    rho: &DensityProfile,
    src_s: &[f64],
    tgt_z: &[f64],
    tgt_mass: &[f64],
    how: Extend,
) -> Vec<f64> {
    let positive = how == Extend::PositiveConstant;
    let mean = ext.extend(src_s, how);
    let n = ext.n_layers();
    // Intercept such that the layer's mass-weighted mean stays `mean`.  The
    // slopes are read off the intercepts (the layer-centre values), not off
    // the mass-weighted means, which differ from them under a sloping
    // density; two passes settle both to rounding on a linear profile.
    let intercept = |slope: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|k| {
                let h = ext.z[k + 1] - ext.z[k];
                if rho.mean[k] > 0.0 {
                    mean[k] - rho.slope[k] * slope[k] * h * h / (12.0 * rho.mean[k])
                } else {
                    mean[k]
                }
            })
            .collect()
    };
    let mut slope = ext.slopes(&mean, positive);
    let mut s0 = intercept(&slope);
    for _ in 0..2 {
        slope = ext.slopes(&s0, positive);
        s0 = intercept(&slope);
    }
    let nt = tgt_z.len() - 1;
    let mut acc = vec![0.0f64; nt];
    sweep(ext, tgt_z, |t, k, a, b| {
        let zc = ext.mid(k);
        let (a, b) = (a - zc, b - zc);
        let (r0, rs, q0, qs) = (rho.mean[k], rho.slope[k], s0[k], slope[k]);
        acc[t] += r0 * q0 * (b - a)
            + (r0 * qs + rs * q0) * (b * b - a * a) * 0.5
            + rs * qs * (b * b * b - a * a * a) / 3.0;
    });
    let mut out: Vec<f64> = (0..nt)
        .map(|t| if tgt_mass[t] > 0.0 { acc[t] / tgt_mass[t] } else { 0.0 })
        .collect();
    if positive {
        for v in &mut out {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
    }
    out
}

/// Visit every (target layer, extended source layer) overlap `[a, b]`.
fn sweep(ext: &Extended, tgt_z: &[f64], mut f: impl FnMut(usize, usize, f64, f64)) {
    let ns = ext.n_layers();
    let nt = tgt_z.len() - 1;
    let mut k = 0usize;
    for t in 0..nt {
        let (lo, hi) = (tgt_z[t], tgt_z[t + 1]);
        while k < ns && ext.z[k + 1] <= lo {
            k += 1;
        }
        let mut kk = k;
        while kk < ns && ext.z[kk] < hi {
            let a = lo.max(ext.z[kk]);
            let b = hi.min(ext.z[kk + 1]);
            if b > a {
                f(t, kk, a, b);
            }
            kk += 1;
        }
    }
}

/// Linear interpolation of point values in height, constant past the ends.
pub fn interp_points(src_z: &[f64], src_v: &[f64], tgt_z: &[f64]) -> Vec<f64> {
    let n = src_z.len();
    tgt_z
        .iter()
        .map(|&z| {
            if z <= src_z[0] {
                return src_v[0];
            }
            if z >= src_z[n - 1] {
                return src_v[n - 1];
            }
            let i = src_z.partition_point(|&s| s <= z).clamp(1, n - 1);
            let (z0, z1) = (src_z[i - 1], src_z[i]);
            let w = if z1 > z0 { (z - z0) / (z1 - z0) } else { 0.0 };
            src_v[i - 1] + w * (src_v[i] - src_v[i - 1])
        })
        .collect()
}

/// The theta_m factor for a virtual-factor arm: what `init_atm_case_gfs`
/// multiplies `qv` by in `theta_m`, and so the factor its hydrostatic
/// rebalance and the dycore's equation of state agree on.
pub fn theta_m_factor(factor: VirtualFactor) -> f64 {
    match factor {
        VirtualFactor::ReproduceFortran => 1.61,
        VirtualFactor::Consistent => (RVORD - 1.0) as f64,
    }
}

/// Pressure from dry density, theta and qv: the dycore's equation of state,
/// `p = p0 (R rho theta_m / p0)^(cp/cv)`.
pub fn pressure_from_state(rho: f64, theta: f64, qv: f64, factor: VirtualFactor) -> f64 {
    let theta_m = theta * (1.0 + theta_m_factor(factor) * qv);
    let p0 = P0 as f64;
    p0 * ((RGAS as f64) * rho * theta_m / p0).powf((CP / CV) as f64)
}

/// Rebalance one target column the way `rw_mpas_init` builds one.
///
/// Temperature and pressure are recovered from the remapped
/// `(rho, theta, qv)` by the equation of state, and handed to
/// [`dynamics::build_column`] with `qv` as specific humidity: the same
/// routine, the same hydrostatic fixed point, the same base state the init
/// writer runs.  `theta` and `qv` come back unchanged (to rounding); `rho`
/// comes back in hydrostatic balance on the target's own metric.
#[allow(clippy::too_many_arguments)]
pub fn rebalance(
    rho: &[f64],
    theta: &[f64],
    qv: &[f64],
    zgrid: &[f32],
    zz: &[f32],
    fzm: &[f32],
    fzp: &[f32],
    dzu: &[f32],
    rdzw0: f32,
    factor: VirtualFactor,
) -> ColumnState {
    let nz = theta.len();
    let mut t = vec![0.0f32; nz];
    let mut p = vec![0.0f32; nz];
    let mut sh = vec![0.0f32; nz];
    for k in 0..nz {
        let pk = pressure_from_state(rho[k], theta[k], qv[k], factor);
        p[k] = pk as f32;
        t[k] = (theta[k] * (pk / P0 as f64).powf((RGAS / CP) as f64)) as f32;
        sh[k] = (qv[k] / (1.0 + qv[k])) as f32;
    }
    let rh = vec![0.0f32; nz];
    dynamics::build_column(&t, &p, &rh, &sh, true, zgrid, zz, fzm, fzp, dzu, rdzw0, factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(n: usize, bottom: f64, top: f64) -> Vec<f64> {
        (0..=n).map(|k| bottom + (top - bottom) * k as f64 / n as f64).collect()
    }

    #[test]
    fn a_linear_density_profile_is_reproduced_exactly() {
        let src = column(20, 0.0, 20000.0);
        let tgt = column(17, 0.0, 20000.0);
        let rho_of = |z: f64| 1.2 - 4.0e-5 * z;
        let src_rho: Vec<f64> = (0..20).map(|k| rho_of(0.5 * (src[k] + src[k + 1]))).collect();
        let ext = Extended::new(&src, &tgt);
        let (mass, _) = remap_density(&ext, &src_rho, &tgt);
        for t in 0..17 {
            let h = tgt[t + 1] - tgt[t];
            let want = rho_of(0.5 * (tgt[t] + tgt[t + 1]));
            assert!((mass[t] / h - want).abs() < 1e-12, "{t}: {} vs {want}", mass[t] / h);
        }
    }

    #[test]
    fn column_mass_is_conserved_when_the_columns_span_the_same_heights() {
        let src = column(30, 0.0, 22000.0);
        let mut tgt = column(25, 0.0, 22000.0);
        // Stretch the target's interior.
        for k in 1..25 {
            let x = tgt[k] / 22000.0;
            tgt[k] = 22000.0 * x.powf(1.4);
        }
        let src_rho: Vec<f64> = (0..30)
            .map(|k| 1.2 * (-(0.5 * (src[k] + src[k + 1])) / 8000.0).exp())
            .collect();
        let src_mass: f64 = (0..30).map(|k| src_rho[k] * (src[k + 1] - src[k])).sum();
        let ext = Extended::new(&src, &tgt);
        let (mass, _) = remap_density(&ext, &src_rho, &tgt);
        let got: f64 = mass.iter().sum();
        assert!(((got - src_mass) / src_mass).abs() < 1e-13);
    }

    #[test]
    fn a_linear_scalar_under_varying_density_is_reproduced_closely() {
        let src = column(20, 100.0, 20000.0);
        let tgt = column(23, 100.0, 20000.0);
        let rho_of = |z: f64| 1.2 - 4.0e-5 * z;
        let th_of = |z: f64| 290.0 + 0.004 * z;
        let mid = |z: &[f64], k: usize| 0.5 * (z[k] + z[k + 1]);
        let src_rho: Vec<f64> = (0..20).map(|k| rho_of(mid(&src, k))).collect();
        // The mass-weighted layer mean of a linear theta under linear rho.
        let src_th: Vec<f64> = (0..20)
            .map(|k| {
                let (a, b) = (src[k], src[k + 1]);
                let n = 2000;
                let (mut num, mut den) = (0.0, 0.0);
                for i in 0..n {
                    let z = a + (b - a) * (i as f64 + 0.5) / n as f64;
                    num += rho_of(z) * th_of(z);
                    den += rho_of(z);
                }
                num / den
            })
            .collect();
        let ext = Extended::new(&src, &tgt);
        let (mass, prof) = remap_density(&ext, &src_rho, &tgt);
        let th = remap_scalar(&ext, &prof, &src_th, &tgt, &mass, Extend::LinearBelow);
        for t in 0..23 {
            let (a, b) = (tgt[t], tgt[t + 1]);
            let n = 2000;
            let (mut num, mut den) = (0.0, 0.0);
            for i in 0..n {
                let z = a + (b - a) * (i as f64 + 0.5) / n as f64;
                num += rho_of(z) * th_of(z);
                den += rho_of(z);
            }
            assert!((th[t] - num / den).abs() < 1e-6, "{t}: {} vs {}", th[t], num / den);
        }
    }

    #[test]
    fn a_lower_target_terrain_extends_below_and_counts_it() {
        let src = column(10, 500.0, 20000.0);
        let tgt = column(10, 200.0, 20000.0);
        let ext = Extended::new(&src, &tgt);
        assert_eq!(ext.offset, 1);
        assert!((ext.below_m - 300.0).abs() < 1e-9);
        let q: Vec<f64> = (0..10).map(|k| 0.01 * (1.0 - k as f64 / 10.0)).collect();
        let e = ext.extend(&q, Extend::PositiveConstant);
        assert_eq!(e[0], q[0]);
    }

    #[test]
    fn point_interpolation_is_linear_inside_and_constant_outside() {
        let z = [0.0, 100.0, 300.0];
        let v = [1.0, 2.0, 4.0];
        let got = interp_points(&z, &v, &[-10.0, 50.0, 200.0, 400.0]);
        assert_eq!(got, vec![1.0, 1.5, 3.0, 4.0]);
    }

    #[test]
    fn the_rebalance_keeps_theta_and_qv_and_returns_a_hydrostatic_column() {
        let nz = 20;
        let zgrid: Vec<f32> = (0..=nz).map(|k| 1000.0 * k as f32).collect();
        let zz = vec![1.0f32; nz];
        let mut fzm = vec![0.5f32; nz];
        let mut fzp = vec![0.5f32; nz];
        fzm[0] = 0.0;
        fzp[0] = 0.0;
        let dzu = vec![1000.0f32; nz];
        let theta: Vec<f64> = (0..nz).map(|k| 290.0 + 3.5 * k as f64).collect();
        let qv: Vec<f64> = (0..nz).map(|k| 0.01 * (-(k as f64) / 4.0).exp()).collect();
        let rho: Vec<f64> = (0..nz).map(|k| 1.2 * (-(500.0 + 1000.0 * k as f64) / 8500.0).exp()).collect();
        let s = rebalance(&rho, &theta, &qv, &zgrid, &zz, &fzm, &fzp, &dzu, 1.0e-3, VirtualFactor::ReproduceFortran);
        for k in 0..nz {
            assert!((s.theta[k] as f64 - theta[k]).abs() < 1e-3 * theta[k], "theta {k}");
            assert!((s.qv[k] as f64 - qv[k]).abs() < 1e-5 * qv[k].max(1e-6) + 1e-9, "qv {k}");
        }
        // Hydrostatic: dp/dz = -g rho (1 + qv) to the discretisation's order.
        for k in 1..nz {
            let dp = s.pressure[k] - s.pressure[k - 1];
            let rho_m = 0.5 * (s.rho[k] * (1.0 + s.qv[k]) + s.rho[k - 1] * (1.0 + s.qv[k - 1]));
            let want = -GRAVITY * rho_m * 1000.0;
            assert!(((dp - want) / want).abs() < 0.02, "level {k}: {dp} vs {want}");
        }
    }
}
