// IBM Accurate Mathematical Library
// written by International Business Machines Corp.
// Copyright (C) 2001-2024 Free Software Foundation, Inc.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Modified: transcribed to Rust from glibc 2.39 sysdeps/ieee754/dbl-64/s_atan.c and atnat.h
// by Recast WOOF contributors, 2026.

//! glibc 2.39 `__atan`, as `__atan_fma` (the x86-64 ifunc variant built
//! with -mfma -mavx2) computes it.  Each `mul_add` is a site that build
//! fuses; nothing else is fused.

use super::dla::{emulv, esub};

#[path = "uatan_tbl.rs"]
pub(super) mod uatan_tbl;
use self::uatan_tbl::CIJ;

const TWO52: f64 = 4503599627370496.0;

// atnat.h, polynomial I and constants.
pub(super) const D3: f64 = f64::from_bits(0xbfd5555555555555);
pub(super) const D5: f64 = f64::from_bits(0x3fc99999999997fd);
pub(super) const D7: f64 = f64::from_bits(0xbfc24924923f7603);
pub(super) const D9: f64 = f64::from_bits(0x3fbc71c6e5129a3b);
pub(super) const D11: f64 = f64::from_bits(0xbfb7458022b13c25);
pub(super) const D13: f64 = f64::from_bits(0x3fb375f08b31cbce);
const A: f64 = f64::from_bits(0x3e4bb67a00000000);
const B: f64 = f64::from_bits(0x3fb0000000000000);
const C: f64 = f64::from_bits(0x3ff0000000000000);
const D: f64 = f64::from_bits(0x4030000000000000);
const E: f64 = f64::from_bits(0x43349ff200000000);
const HPI: f64 = f64::from_bits(0x3ff921fb54442d18);
const MHPI: f64 = f64::from_bits(0xbff921fb54442d18);
const HPI1: f64 = f64::from_bits(0x3c91a62633145c07);

#[inline(always)]
pub(super) fn cij(i: usize, j: usize) -> f64 {
    f64::from_bits(CIJ[i][j])
}

/// The table row for u in [1/16, 1): (TWO52 + 256 * u) - TWO52 - 16, the
/// sum fused (exact here either way).
#[inline(always)]
pub(super) fn row(u: f64) -> usize {
    (u.mul_add(256.0, TWO52) - TWO52) as i32 as usize - 16
}

/// d3..d13 in Horner form, each step fused.
#[inline(always)]
pub(super) fn poly_d(v: f64) -> f64 {
    let mut yy = v.mul_add(D13, D11);
    yy = v.mul_add(yy, D9);
    yy = v.mul_add(yy, D7);
    yy = v.mul_add(yy, D5);
    v.mul_add(yy, D3)
}

pub fn atan(x: f64) -> f64 {
    let bits = x.to_bits();
    let ux = (bits >> 32) as u32;
    let dx = bits as u32;
    if (ux & 0x7ff00000) == 0x7ff00000 && ((ux & 0x000fffff) | dx) != 0 {
        return super::atan2::sse_nan_sum(x, x);
    }
    let u = if x < 0.0 { -x } else { x };
    if u < C {
        if u < B {
            if u < A {
                return x;
            }
            // A <= u < B: x + (x * v) * poly, fused.
            let v = x * x;
            let yy = poly_d(v);
            return (x * v).mul_add(yy, x);
        }
        // B <= u < C
        let i = row(u);
        let z = u - cij(i, 0);
        let mut yy = z.mul_add(cij(i, 6), cij(i, 5));
        yy = z.mul_add(yy, cij(i, 4));
        yy = z.mul_add(yy, cij(i, 3));
        yy = z.mul_add(yy, cij(i, 2));
        let y = yy.mul_add(z, cij(i, 1));
        return y.copysign(x);
    }
    if u < D {
        // C <= u < D
        let w = 1.0 / u;
        let (t1, t2) = emulv(w, u);
        let i = row(w);
        let z = w.mul_add((1.0 - t1) - t2, w - cij(i, 0));
        let mut yy = z.mul_add(cij(i, 6), cij(i, 5));
        yy = z.mul_add(yy, cij(i, 4));
        yy = z.mul_add(yy, cij(i, 3));
        yy = z.mul_add(yy, cij(i, 2));
        yy = (-z).mul_add(yy, HPI1);
        let t1 = HPI - cij(i, 1);
        return (t1 + yy).copysign(x);
    }
    if u < E {
        // D <= u < E
        let w = 1.0 / u;
        let v = w * w;
        let (t1, t2) = emulv(w, u);
        let yy = poly_d(v);
        let (t3, cor) = esub(HPI, w);
        let yy = (-(w * v)).mul_add(yy, (-w).mul_add((1.0 - t1) - t2, HPI1 + cor));
        return (t3 + yy).copysign(x);
    }
    if x > 0.0 { HPI } else { MHPI }
}

#[cfg(test)]
mod tests {
    use super::atan;

    #[test]
    fn atan_is_glibc_2_39s_words() {
        // (argument, glibc 2.39 x86-64 result), recorded from Ubuntu 24.04.
        let pins: [(u64, u64); 14] = [
            (0x3e112e0be826d695, 0x3e112e0be826d695), (0x3fa47ae147ae147b, 0x3fa4781623768d56),
            (0xbfb0000000000000, 0xbfaff55bb72cfdea), (0x3fd3333333333333, 0x3fd2a73a661eaf06),
            (0xbfe6a09e667f3bcd, 0xbfe3b2028082e8d4), (0x3fefff2e48e8a71e, 0x3fe921927760e4cb),
            (0x3ff0000000000000, 0x3fe921fb54442d18), (0x3ff8000000000000, 0x3fef730bd281f69b),
            (0xc01d000000000000, 0xbff6f08f07435fec), (0x402ffae147ae147b, 0x3ff82227a313716c),
            (0x4030000000000000, 0x3ff82250768ac529), (0xc0934a0000000000, 0xbff91ea9efa2930a),
            (0x432550f7dca70000, 0x3ff921fb54442d17), (0xfe37e43c8800759c, 0xbff921fb54442d18),
        ];
        for (x, want) in pins {
            assert_eq!(atan(f64::from_bits(x)).to_bits(), want, "atan({x:#018x})");
        }
        assert_eq!(atan(-0.0).to_bits(), 0x8000_0000_0000_0000);
        assert!(atan(f64::NAN).is_nan());
    }
}
