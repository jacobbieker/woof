// IBM Accurate Mathematical Library
// written by International Business Machines Corp.
// Copyright (C) 2001-2024 Free Software Foundation, Inc.
// SPDX-License-Identifier: LGPL-2.1-or-later
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
// Modified: transcribed to Rust from glibc 2.39 sysdeps/ieee754/dbl-64/
// s_tan.c, utan.h and the dla.h macros it uses, by Recast WOOF
// contributors in 2026.

//! glibc 2.39 `tan` as its x86-64 FMA variant (`__tan_fma`) computes it.
//!
//! Every `mul_add` below is a site where that build fuses (GCC 13,
//! `-mfma -mavx2`, read from the installed Ubuntu glibc 2.39 libm's
//! disassembly); dla.h takes its `__FP_FAST_FMA` forms there.  Arguments
//! above 1e8 reduce through `branred`, which glibc compiles without FMA.

#[path = "utan_tbl.rs"]
mod utan_tbl;

use self::utan_tbl::XFG;
use super::dla::eadd;

const fn d(bits: u64) -> f64 { f64::from_bits(bits) }

// utan.h
const D3: f64 = d(0x3FD5_5555_5555_5555);
const D5: f64 = d(0x3FC1_1111_1111_07C6);
const D7: f64 = d(0x3FAB_A1BA_1CDB_8745);
const D9: f64 = d(0x3F96_64ED_49CF_C666);
const D11: f64 = d(0x3F82_385A_3CF2_E4EA);
const E0: f64 = d(0x3FD5_5555_5555_4DBD);
const E1: f64 = d(0x3FC1_1112_E0A6_B45F);
const MFFTNHF: f64 = d(0xc02f_0000_0000_0000);
const G1: f64 = d(0x3e4b_096c_0000_0000);
const G2: f64 = d(0x3faf_212d_0000_0000);
const G3: f64 = d(0x3fe9_2f1a_0000_0000);
const G4: f64 = d(0x4039_0000_0000_0000);
const G5: f64 = d(0x4197_d784_0000_0000);
const GY2: f64 = d(0x3faf_212d_0000_0000);
const MP1: f64 = d(0x3FF9_21FB_5800_0000);
const MP2: f64 = d(0xBE4D_DE97_3C00_0000);
const MP3: f64 = d(0xBC8C_B3B3_99D7_47F2);
const PP3: f64 = d(0xBC8C_B3B3_9800_0000);
const PP4: f64 = d(0xbacd_747f_23e3_2ed7);
const HPINV: f64 = d(0x3FE4_5F30_6DC9_C883);
const TOINT: f64 = d(0x4338_0000_0000_0000);

/// d3 + x2*(d5 + x2*(d7 + x2*(d9 + x2*d11))), every step fused.
fn poly_d(x2: f64) -> f64 {
    let t2 = x2.mul_add(D11, D9);
    let t2 = x2.mul_add(t2, D7);
    let t2 = x2.mul_add(t2, D5);
    x2.mul_add(t2, D3)
}

/// The table step: xfg row i and pz = z + z*z2*(e0 + z2*e1).
fn table(y: f64, yy: f64) -> (f64, f64, f64) {
    let i = 256.0f64.mul_add(y, MFFTNHF) as i32 as usize;
    let row = &XFG[i];
    let z = (y - d(row[0])) + yy;
    let z2 = z * z;
    let pz = (z * z2).mul_add(z2.mul_add(E1, E0), z);
    (pz, d(row[1]), d(row[2]))
}

/// Cases (VI)-(XI): tan or -cot of the reduced a + da.
fn reduced(n: i32, a: f64, da: f64) -> f64 {
    let (ya, yya, sy) = if a < 0.0 { (-a, -da, -1.0) } else { (a, da, 1.0) };
    if ya <= GY2 {
        let a2 = a * a;
        let t2 = (a * a2).mul_add(poly_d(a2), da);
        if n != 0 {
            // EADD (a, t2, b, db); DIV2 (1.0, 0.0, b, db, c, dc, ...)
            let b = a + t2;
            let db = if a.abs() > t2.abs() { (a - b) + t2 } else { (t2 - b) + a };
            let c = 1.0 / b;
            let u = c * b;
            let uu = c.mul_add(b, -u);
            let cc = (-c).mul_add(db, ((1.0 - u) - uu) + 0.0) / b;
            let z = c + cc;
            let zz = (c - z) + cc;
            return -(z + zz);
        }
        return a + t2;
    }
    let (pz, fi, gi) = table(ya, yya);
    if n != 0 {
        let t2 = pz * (fi + gi) / (fi + pz);
        let y = gi - t2;
        -sy * y
    } else {
        let t2 = pz * (gi + fi) / (gi - pz);
        let y = fi + t2;
        sy * y
    }
}

/// tan(x) with glibc 2.39's words.
pub fn tan(x: f64) -> f64 {
    let ux = (x.to_bits() >> 32) as u32;
    if ux & 0x7ff0_0000 == 0x7ff0_0000 {
        return x - x;
    }
    let w = x.abs();
    // (I) |x| <= 1.259e-8
    if w <= G1 {
        return x;
    }
    // (II) |x| <= 0.0608
    if w <= G2 {
        let x2 = x * x;
        return (x * x2).mul_add(poly_d(x2), x);
    }
    // (III) |x| <= 0.787
    if w <= G3 {
        let (pz, fi, gi) = table(w, 0.0);
        let s = if x < 0.0 { -1.0 } else { 1.0 };
        let t2 = pz * (gi + fi) / (gi - pz);
        let y = fi + t2;
        return s * y;
    }
    // (IV) |x| <= 25: reduction by algorithm i
    if w <= G4 {
        let t = x.mul_add(HPINV, TOINT);
        let xn = t - TOINT;
        let n = (t.to_bits() & 1) as i32;
        let t1 = (-xn).mul_add(MP2, (-xn).mul_add(MP1, x));
        let a = (-xn).mul_add(MP3, t1);
        let da = (-xn).mul_add(MP3, t1 - a);
        return reduced(n, a, da);
    }
    // (V) |x| <= 1e8: reduction by algorithm ii
    if w <= G5 {
        let t = x.mul_add(HPINV, TOINT);
        let xn = t - TOINT;
        let n = (t.to_bits() & 1) as i32;
        let t1 = (-xn).mul_add(MP2, (-xn).mul_add(MP1, x));
        let t = (-xn).mul_add(PP3, t1);
        let da = (-xn).mul_add(PP3, t1 - t);
        let a = (-xn).mul_add(PP4, t);
        let da = (-xn).mul_add(PP4, t - a) + da;
        let (a, da) = eadd(a, da);
        return reduced(n, a, da);
    }
    // (X)-(XI) |x| > 1e8: reduction by algorithm iii
    let (n, a, da) = super::branred::branred(x);
    let (a, da) = eadd(a, da);
    reduced(n & 1, a, da)
}

#[cfg(test)]
mod tests {
    use super::tan;

    #[test]
    fn tan_gives_recorded_glibc_2_39_words() {
        // (argument, glibc 2.39 tan) bit patterns, one or more per branch;
        // the last three are huge arguments where glibc 2.44 differs.
        let cases: [(u64, u64); 15] = [
            (0x3e112e0be826d695, 0x3e112e0be826d695),
            (0x3fa999999999999a, 0x3fa99f1119d15c71),
            (0xbfd3333333333333, 0xbfd3cc2a44e29998),
            (0x3fe6666666666666, 0x3feaf406c2fc78ae),
            (0x3ff0000000000000, 0x3ff8eb245cbee3a6),
            (0xc004000000000000, 0x3fe7e79b4e00bb15),
            (0x4024000000000000, 0x3fe4bf5f34be3782),
            (0x4038e66666666666, 0xbfce5738470a46cf),
            (0x4059000000000000, 0xbfe2ca74d62b5d38),
            (0xc0c81cd6c8b43958, 0x3fefba5836323a4e),
            (0x416312d000000000, 0xbfddaa7d34937ac4),
            (0x4480f0cf064dd592, 0xbffa0f79c1b6b257),
            (0x4a34ecefa35e3e79, 0xbff0e1a473ad3a37),
            (0xc41453a2a9527be9, 0xbfc1382a7cc80276),
            (0x42c3f74ff5004d99, 0xbfe8b3d02c116548),
        ];
        for (x, want) in cases {
            assert_eq!(tan(f64::from_bits(x)).to_bits(), want, "tan({x:#018x})");
        }
        assert!(tan(f64::INFINITY).is_nan() && tan(f64::NAN).is_nan());
        assert_eq!(tan(-0.0).to_bits(), (-0.0f64).to_bits());
    }
}
