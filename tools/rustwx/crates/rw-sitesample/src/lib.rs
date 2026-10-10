//! Sampling WRF history at forecast sites and heights.
//!
//! `woof energy` asks a WRF run for wind, temperature and moisture at
//! irregular points (towers, line spans, turbines) and at fixed heights
//! above the model terrain.  Python works out where each site falls on the
//! grid (a fractional mass-point index, from the file's projection) and
//! reads a window of each field around the sites; this crate does every
//! per-cell operation on those windows:
//!
//! * destaggering onto mass points: U along `west_east_stag`, V along
//!   `south_north_stag`, W and PH/PHB along `bottom_top_stag`, each as the
//!   mean of the two bracketing staggered values (WRF's own `destagger`);
//! * bilinear interpolation on mass points at the fractional index;
//! * height above model terrain at mass levels,
//!   `0.5 * (z[k] + z[k+1]) - HGT` with `z = (PH + PHB) / g`;
//! * linear interpolation in height to the requested heights, NaN outside
//!   the column's mass-level range (nothing is extrapolated);
//! * earth-relative wind, `u_e = u cos a - v sin a`, `v_e = v cos a + u sin
//!   a` (wrf-python `uvmet`, `rw-mlexport` `ops::rotate`), applied at each
//!   mass point of the stencil before the horizontal interpolation, so a
//!   sampled wind is the bilinear interpolant of the earth-relative field.
//!
//! A site whose bilinear stencil leaves the mass grid (or whose index is
//! not finite) is outside: every value sampled there is NaN.
//!
//! Index conventions: fields arrive as C-contiguous `f32` exactly as the
//! history stores them, level outermost (`[k][j][i]`); arithmetic is `f64`.
//! `nz`, `ny`, `nx` always name the MASS dimensions; a staggered field is one
//! longer along its staggered axis.  Fractional indices are zero-based on
//! mass points: `fi == 0` is the first mass column, `fi == nx - 1` the last.

pub mod capi;

/// WRF's gravitational acceleration (`module_model_constants`, `g = 9.81`),
/// which is what the model's geopotential is divided by to read a height.
pub const WRF_GRAVITY: f64 = 9.81;

/// Which axis, if any, a field is staggered along.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stagger {
    Mass,
    X,
    Y,
    Z,
}

impl Stagger {
    /// The ABI's integer code: 0 mass, 1 x, 2 y, 3 z.
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::Mass),
            1 => Some(Self::X),
            2 => Some(Self::Y),
            3 => Some(Self::Z),
            _ => None,
        }
    }

    /// Stored shape `(levels, rows, columns)` for mass dims `(nz, ny, nx)`.
    pub fn shape(self, nz: usize, ny: usize, nx: usize) -> (usize, usize, usize) {
        match self {
            Self::Mass => (nz, ny, nx),
            Self::X => (nz, ny, nx + 1),
            Self::Y => (nz, ny + 1, nx),
            Self::Z => (nz + 1, ny, nx),
        }
    }
}

/// Mass dimensions of a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub nz: usize,
    pub ny: usize,
    pub nx: usize,
}

impl Dims {
    pub fn len(self, stagger: Stagger) -> usize {
        let (a, b, c) = stagger.shape(self.nz, self.ny, self.nx);
        a * b * c
    }

    pub fn is_empty(self) -> bool {
        self.nz == 0 || self.ny == 0 || self.nx == 0
    }
}

/// The four mass points around a site and its bilinear weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stencil {
    pub i0: usize,
    pub j0: usize,
    pub wx: f64,
    pub wy: f64,
}

/// The stencil at fractional mass-point index `(fi, fj)` on an `ny` by `nx`
/// mass grid, or `None` when the site is outside it.  A site exactly on the
/// last row or column uses the last cell with weight one on its far side.
pub fn stencil(fi: f64, fj: f64, ny: usize, nx: usize) -> Option<Stencil> {
    if nx < 2 || ny < 2 || !fi.is_finite() || !fj.is_finite() {
        return None;
    }
    if fi < 0.0 || fj < 0.0 || fi > (nx - 1) as f64 || fj > (ny - 1) as f64 {
        return None;
    }
    let i0 = (fi.floor() as usize).min(nx - 2);
    let j0 = (fj.floor() as usize).min(ny - 2);
    Some(Stencil {
        i0,
        j0,
        wx: fi - i0 as f64,
        wy: fj - j0 as f64,
    })
}

/// One field window: `scale * (data + plus) + offset`, read on mass points.
#[derive(Clone, Copy)]
pub struct Field<'a> {
    pub data: &'a [f32],
    pub plus: Option<&'a [f32]>,
    pub scale: f64,
    pub offset: f64,
    pub stagger: Stagger,
    pub dims: Dims,
}

impl<'a> Field<'a> {
    /// A field read as stored.
    pub fn plain(data: &'a [f32], stagger: Stagger, dims: Dims) -> Self {
        Self {
            data,
            plus: None,
            scale: 1.0,
            offset: 0.0,
            stagger,
            dims,
        }
    }

    #[inline]
    fn stored(&self, k: usize, j: usize, i: usize) -> f64 {
        let (_, rows, cols) = self.stagger.shape(self.dims.nz, self.dims.ny, self.dims.nx);
        let index = (k * rows + j) * cols + i;
        let mut value = f64::from(self.data[index]);
        if let Some(plus) = self.plus {
            value += f64::from(plus[index]);
        }
        value
    }

    /// The field at mass point `(k, j, i)`, destaggered when staggered.
    #[inline]
    pub fn mass(&self, k: usize, j: usize, i: usize) -> f64 {
        let raw = match self.stagger {
            Stagger::Mass => self.stored(k, j, i),
            Stagger::X => 0.5 * (self.stored(k, j, i) + self.stored(k, j, i + 1)),
            Stagger::Y => 0.5 * (self.stored(k, j, i) + self.stored(k, j + 1, i)),
            Stagger::Z => 0.5 * (self.stored(k, j, i) + self.stored(k + 1, j, i)),
        };
        self.scale * raw + self.offset
    }

    /// Bilinear value at level `k` of the stencil.
    #[inline]
    pub fn bilinear(&self, k: usize, s: &Stencil) -> f64 {
        bilinear(s, |j, i| self.mass(k, j, i))
    }

    /// The whole mass-level column at the stencil.
    pub fn column(&self, s: &Stencil) -> Vec<f64> {
        (0..self.dims.nz).map(|k| self.bilinear(k, s)).collect()
    }
}

/// Bilinear combination of the stencil's south-west, south-east, north-west
/// and north-east values.
#[inline]
pub fn blend(s: &Stencil, sw: f64, se: f64, nw: f64, ne: f64) -> f64 {
    let south = (1.0 - s.wx) * sw + s.wx * se;
    let north = (1.0 - s.wx) * nw + s.wx * ne;
    (1.0 - s.wy) * south + s.wy * north
}

/// Bilinear combination of `value(j, i)` over the stencil's four points.
#[inline]
pub fn bilinear(s: &Stencil, value: impl Fn(usize, usize) -> f64) -> f64 {
    let (i0, j0) = (s.i0, s.j0);
    blend(
        s,
        value(j0, i0),
        value(j0, i0 + 1),
        value(j0 + 1, i0),
        value(j0 + 1, i0 + 1),
    )
}

/// Earth-relative `(u_e, v_e)` from grid-relative `(u, v)` and the local
/// rotation `(sin a, cos a)`: wrf-python `uvmet`, `rw-mlexport` `rotate`.
#[inline]
pub fn rotate(u: f64, v: f64, sin: f64, cos: f64) -> (f64, f64) {
    (u * cos - v * sin, v * cos + u * sin)
}

/// Height above model terrain of each mass level at the stencil:
/// `0.5 * (z[k] + z[k+1]) - HGT`, `z = (PH + PHB) / gravity`.
pub fn heights_agl(
    ph: &[f32],
    phb: &[f32],
    hgt: &[f32],
    dims: Dims,
    gravity: f64,
    s: &Stencil,
) -> Vec<f64> {
    let z = Field {
        data: ph,
        plus: Some(phb),
        scale: 1.0 / gravity,
        offset: 0.0,
        stagger: Stagger::Z,
        dims,
    };
    let terrain = Field::plain(hgt, Stagger::Mass, Dims { nz: 1, ..dims }).bilinear(0, s);
    (0..dims.nz).map(|k| z.bilinear(k, s) - terrain).collect()
}

/// Linear interpolation in height of `column` (on levels at heights `zagl`)
/// to `target`; NaN outside `[zagl[0], zagl[last]]` or when the bracketing
/// values are not finite.
pub fn interpolate_height(column: &[f64], zagl: &[f64], target: f64) -> f64 {
    let n = column.len().min(zagl.len());
    if n == 0 || !target.is_finite() {
        return f64::NAN;
    }
    if n == 1 {
        return if zagl[0] == target {
            column[0]
        } else {
            f64::NAN
        };
    }
    for k in 0..n - 1 {
        let (z0, z1) = (zagl[k], zagl[k + 1]);
        if !(z0.is_finite() && z1.is_finite()) {
            return f64::NAN;
        }
        if z0 <= target && target <= z1 {
            if z1 == z0 {
                return column[k];
            }
            let w = (target - z0) / (z1 - z0);
            return column[k] + w * (column[k + 1] - column[k]);
        }
    }
    f64::NAN
}

/// Every requested height of one column (`out.len() == heights.len()`).
pub fn interpolate_heights(column: &[f64], zagl: &[f64], heights: &[f64], out: &mut [f64]) {
    for (value, &target) in out.iter_mut().zip(heights) {
        *value = interpolate_height(column, zagl, target);
    }
}

/// Earth-relative wind at the stencil on every mass level: U (x-staggered)
/// and V (y-staggered) are destaggered and rotated at each of the four mass
/// points, then combined bilinearly.  `sin`/`cos` are mass-point planes.
pub fn wind_columns(
    u: &Field,
    v: &Field,
    sin: &[f32],
    cos: &[f32],
    s: &Stencil,
) -> (Vec<f64>, Vec<f64>) {
    let nx = u.dims.nx;
    let (i0, j0) = (s.i0, s.j0);
    let mut ue = Vec::with_capacity(u.dims.nz);
    let mut ve = Vec::with_capacity(u.dims.nz);
    for k in 0..u.dims.nz {
        // each stencil point destaggered and rotated once
        let at = |j: usize, i: usize| {
            let c = j * nx + i;
            rotate(
                u.mass(k, j, i),
                v.mass(k, j, i),
                f64::from(sin[c]),
                f64::from(cos[c]),
            )
        };
        let (sw, se) = (at(j0, i0), at(j0, i0 + 1));
        let (nw, ne) = (at(j0 + 1, i0), at(j0 + 1, i0 + 1));
        ue.push(blend(s, sw.0, se.0, nw.0, ne.0));
        ve.push(blend(s, sw.1, se.1, nw.1, ne.1));
    }
    (ue, ve)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-12 * (1.0 + b.abs()), "{a} != {b}");
    }

    #[test]
    fn stencil_covers_the_mass_grid_and_nothing_else() {
        let s = stencil(1.25, 2.5, 4, 3).unwrap();
        assert_eq!((s.i0, s.j0), (1, 2));
        close(s.wx, 0.25);
        close(s.wy, 0.5);
        // the far edge is inside, using the last cell with full weight
        let edge = stencil(2.0, 3.0, 4, 3).unwrap();
        assert_eq!((edge.i0, edge.j0), (1, 2));
        close(edge.wx, 1.0);
        close(edge.wy, 1.0);
        assert!(stencil(-1e-9, 1.0, 4, 3).is_none());
        assert!(stencil(2.0 + 1e-9, 1.0, 4, 3).is_none());
        assert!(stencil(1.0, 3.0 + 1e-9, 4, 3).is_none());
        assert!(stencil(f64::NAN, 1.0, 4, 3).is_none());
        assert!(stencil(0.0, 0.0, 4, 1).is_none());
    }

    #[test]
    fn bilinear_reproduces_a_plane_exactly() {
        let dims = Dims {
            nz: 1,
            ny: 3,
            nx: 4,
        };
        let data: Vec<f32> = (0..3)
            .flat_map(|j| (0..4).map(move |i| (2 * i + 3 * j) as f32))
            .collect();
        let field = Field::plain(&data, Stagger::Mass, dims);
        let s = stencil(2.3, 0.7, 3, 4).unwrap();
        close(field.bilinear(0, &s), 2.0 * 2.3 + 3.0 * 0.7);
    }

    #[test]
    fn destaggering_averages_the_bracketing_faces() {
        let dims = Dims {
            nz: 2,
            ny: 2,
            nx: 2,
        };
        // x-staggered: (2, 2, 3); value = i (face index)
        let ux: Vec<f32> = (0..12).map(|n| (n % 3) as f32).collect();
        let u = Field::plain(&ux, Stagger::X, dims);
        close(u.mass(0, 0, 0), 0.5);
        close(u.mass(1, 1, 1), 1.5);
        // y-staggered: (2, 3, 2); value = j
        let vy: Vec<f32> = (0..12).map(|n| ((n / 2) % 3) as f32).collect();
        let v = Field::plain(&vy, Stagger::Y, dims);
        close(v.mass(0, 0, 1), 0.5);
        close(v.mass(1, 1, 0), 1.5);
        // z-staggered: (3, 2, 2); value = 10 k
        let wz: Vec<f32> = (0..12).map(|n| (10 * (n / 4)) as f32).collect();
        let w = Field::plain(&wz, Stagger::Z, dims);
        close(w.mass(0, 1, 1), 5.0);
        close(w.mass(1, 0, 0), 15.0);
    }

    #[test]
    fn scale_plus_and_offset_apply_after_destaggering() {
        let dims = Dims {
            nz: 1,
            ny: 2,
            nx: 2,
        };
        let a: Vec<f32> = vec![9.81, 9.81, 9.81, 9.81, 19.62, 19.62, 19.62, 19.62];
        let b: Vec<f32> = vec![0.0; 8];
        let z = Field {
            data: &a,
            plus: Some(&b),
            scale: 1.0 / WRF_GRAVITY,
            offset: 300.0,
            stagger: Stagger::Z,
            dims,
        };
        let expected = 0.5 * (f64::from(9.81f32) + f64::from(19.62f32)) / WRF_GRAVITY + 300.0;
        close(z.mass(0, 0, 0), expected);
    }

    #[test]
    fn heights_are_layer_midpoints_above_terrain() {
        let dims = Dims {
            nz: 2,
            ny: 2,
            nx: 2,
        };
        // z at w-levels: 100, 120, 160 m everywhere; terrain 100 m
        let ph: Vec<f32> = [100.0f32, 120.0, 160.0]
            .iter()
            .flat_map(|z| std::iter::repeat_n(z * 9.81, 4))
            .collect();
        let phb = vec![0.0f32; 12];
        let hgt = vec![100.0f32; 4];
        let s = stencil(0.5, 0.5, 2, 2).unwrap();
        let z = heights_agl(&ph, &phb, &hgt, dims, WRF_GRAVITY, &s);
        assert!((z[0] - 10.0).abs() < 1e-4, "{z:?}");
        assert!((z[1] - 40.0).abs() < 1e-4, "{z:?}");
    }

    #[test]
    fn vertical_interpolation_is_linear_and_never_extrapolates() {
        let zagl = [10.0, 30.0, 70.0];
        let col = [1.0, 3.0, 5.0];
        close(interpolate_height(&col, &zagl, 10.0), 1.0);
        close(interpolate_height(&col, &zagl, 20.0), 2.0);
        close(interpolate_height(&col, &zagl, 50.0), 4.0);
        close(interpolate_height(&col, &zagl, 70.0), 5.0);
        assert!(interpolate_height(&col, &zagl, 9.99).is_nan());
        assert!(interpolate_height(&col, &zagl, 70.01).is_nan());
        assert!(interpolate_height(&col, &[10.0, f64::NAN, 70.0], 50.0).is_nan());
        let mut out = [0.0; 2];
        interpolate_heights(&col, &zagl, &[30.0, 100.0], &mut out);
        close(out[0], 3.0);
        assert!(out[1].is_nan());
    }

    #[test]
    fn rotation_matches_uvmet() {
        // a = 90 degrees: grid-x points earth-north
        let (ue, ve) = rotate(1.0, 0.0, 1.0, 0.0);
        close(ue, 0.0);
        close(ve, 1.0);
        let (ue, ve) = rotate(0.0, 1.0, 1.0, 0.0);
        close(ue, -1.0);
        close(ve, 0.0);
        let (s, c) = (0.3f64.sin(), 0.3f64.cos());
        let (ue, ve) = rotate(3.0, -2.0, s, c);
        close(ue, 3.0 * c + 2.0 * s);
        close(ve, -2.0 * c + 3.0 * s);
        // speed is preserved
        close(ue.hypot(ve), 3.0f64.hypot(2.0));
    }

    #[test]
    fn wind_is_rotated_at_mass_points_before_interpolating() {
        let dims = Dims {
            nz: 1,
            ny: 2,
            nx: 2,
        };
        let u = vec![1.0f32; 6]; // x-staggered (1, 2, 3)
        let v = vec![0.0f32; 6]; // y-staggered (1, 3, 2)
        // rotation 0 on the west column, 90 degrees on the east column
        let sin = vec![0.0f32, 1.0, 0.0, 1.0];
        let cos = vec![1.0f32, 0.0, 1.0, 0.0];
        let uf = Field::plain(&u, Stagger::X, dims);
        let vf = Field::plain(&v, Stagger::Y, dims);
        let s = stencil(0.5, 0.0, 2, 2).unwrap();
        let (ue, ve) = wind_columns(&uf, &vf, &sin, &cos, &s);
        // mean of earth-relative (1, 0) and (0, 1)
        close(ue[0], 0.5);
        close(ve[0], 0.5);
    }
}
