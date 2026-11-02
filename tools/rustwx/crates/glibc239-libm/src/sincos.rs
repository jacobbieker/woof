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
// You should have received a copy of the GNU  Lesser General Public License
// along with this program; if not, see <https://www.gnu.org/licenses/>.
//
// Transcribed to Rust from glibc 2.39 sysdeps/ieee754/dbl-64/s_sin.c and
// usncs.h by Recast WOOF contributors, 2026, in the form of the x86-64
// __sin_fma/__cos_fma variants: every mul_add below is a vfmadd/vfnmadd/
// vfmsub of Ubuntu glibc 2.39's libm.so.6 (2.39-0ubuntu8.9), and nothing
// else is fused.

use super::branred::branred;

#[path = "sincostab.rs"]
mod sincostab;
use self::sincostab::SINCOSTAB;

const S1: f64 = f64::from_bits(0xbfc5_5555_5555_5555); // -0.16666666666666666
const S2: f64 = f64::from_bits(0x3f81_1111_1111_0ece); // 0.0083333333333323288
const S3: f64 = f64::from_bits(0xbf2a_01a0_19db_08b8); // -0.00019841269834414642
const S4: f64 = f64::from_bits(0x3ec7_1de2_7b9a_7ed9); // 2.755729806860771e-06
const S5: f64 = f64::from_bits(0xbe5a_ddff_c2fc_df59); // -2.5022014848318398e-08
const BIG: f64 = f64::from_bits(0x42c8_0000_0000_0000); // 52776558133248
const HP0: f64 = f64::from_bits(0x3ff9_21fb_5444_2d18); // 1.5707963267948966
const HP1: f64 = f64::from_bits(0x3c91_a626_3314_5c07); // 6.123233995736766e-17
const MP1: f64 = f64::from_bits(0x3ff9_21fb_5800_0000); // 1.5707963407039642
const MP2: f64 = f64::from_bits(0xbe4d_de97_3c00_0000); // -1.3909067564377153e-08
const PP3: f64 = f64::from_bits(0xbc8c_b3b3_9800_0000); // -4.9789962314799099e-17
const PP4: f64 = f64::from_bits(0xbacd_747f_23e3_2ed7); // -1.9034889620193266e-25
const HPINV: f64 = f64::from_bits(0x3fe4_5f30_6dc9_c883); // 0.63661977236758138
const TOINT: f64 = f64::from_bits(0x4338_0000_0000_0000); // 6755399441055744

const SN3: f64 = -1.66666666666664880952546298448555E-01;
const SN5: f64 = 8.33333214285722277379541354343671E-03;
const CS2: f64 = 4.99999999999999999999950396842453E-01;
const CS4: f64 = -4.16666666666664434524222570944589E-02;
const CS6: f64 = 1.38888874007937613028114285595617E-03;

/// SINCOS_TABLE_LOOKUP: (sn, ssn, cs, ccs) at the table point of u.
#[inline(always)]
fn lookup(u: f64) -> (f64, f64, f64, f64) {
    let k = ((u.to_bits() as u32 as i32) << 2) as usize;
    (f64::from_bits(SINCOSTAB[k]), f64::from_bits(SINCOSTAB[k + 1]),
     f64::from_bits(SINCOSTAB[k + 2]), f64::from_bits(SINCOSTAB[k + 3]))
}

/// cos(x + dx) from the table point and short Taylor series.
#[inline(always)]
fn do_cos(x: f64, mut dx: f64) -> f64 {
    if x < 0.0 {
        dx = -dx;
    }
    let u = BIG + x.abs();
    let x = x.abs() - (u - BIG) + dx;
    let xx = x * x;
    let s = (x * xx).mul_add(xx.mul_add(SN5, SN3), x);
    let c = xx * xx.mul_add(xx.mul_add(CS6, CS4), CS2);
    let (sn, ssn, cs, ccs) = lookup(u);
    let cor = (-sn).mul_add(s, (-cs).mul_add(c, (-s).mul_add(ssn, ccs)));
    cs + cor
}

/// sin(x + dx): TAYLOR_SIN below 0.126, else the table.
#[inline(always)]
fn do_sin(x: f64, mut dx: f64) -> f64 {
    let xold = x;
    if x.abs() < 0.126 {
        let xx = x * x;
        let p = S5.mul_add(xx, S4).mul_add(xx, S3).mul_add(xx, S2).mul_add(xx, S1);
        let t = xx.mul_add(p.mul_add(x, -(0.5 * dx)), dx);
        return x + t;
    }
    if x <= 0.0 {
        dx = -dx;
    }
    let u = BIG + x.abs();
    let x = x.abs() - (u - BIG);
    let xx = x * x;
    let s = x + (x * xx).mul_add(xx.mul_add(SN5, SN3), dx);
    let c = x.mul_add(dx, xx * xx.mul_add(xx.mul_add(CS6, CS4), CS2));
    let (sn, ssn, cs, ccs) = lookup(u);
    let cor = s.mul_add(cs, (-c).mul_add(sn, s.mul_add(ccs, ssn)));
    (sn + cor).copysign(xold)
}

/// Reduce |x| < 105414350 to n*pi/2 + (a + da), returning (n mod 4, a, da).
#[inline(always)]
fn reduce_sincos(x: f64) -> (i32, f64, f64) {
    let t = x.mul_add(HPINV, TOINT);
    let xn = t - TOINT;
    let y = (-xn).mul_add(MP2, (-xn).mul_add(MP1, x));
    let n = (t.to_bits() as u32 & 3) as i32;
    let t2 = (-xn).mul_add(PP3, y);
    let mut db = (-xn).mul_add(PP3, y - t2);
    let b = (-xn).mul_add(PP4, t2);
    db += (-xn).mul_add(PP4, t2 - b);
    (n, b, db)
}

#[inline(always)]
fn do_sincos(a: f64, da: f64, n: i32) -> f64 {
    let retval = if n & 1 != 0 { do_cos(a, da) } else { do_sin(a, da) };
    if n & 2 != 0 { -retval } else { retval }
}

/// glibc 2.39's binary64 sin.
pub fn sin(x: f64) -> f64 {
    let k = ((x.to_bits() >> 32) as u32) & 0x7fff_ffff;
    if k < 0x3e50_0000 {
        // |x| < 2^-26: sin(x) = x
        x
    } else if k < 0x3feb_6000 {
        // 2^-26 < |x| < 0.855469
        do_sin(x, 0.0)
    } else if k < 0x4003_68fd {
        // 0.855469 < |x| < 2.426265
        let t = HP0 - x.abs();
        do_cos(t, HP1).copysign(x)
    } else if k < 0x4199_21FB {
        // 2.426265 < |x| < 105414350
        let (n, a, da) = reduce_sincos(x);
        do_sincos(a, da, n)
    } else if k < 0x7ff0_0000 {
        // 105414350 < |x| < 2^1024
        let (n, a, da) = branred(x);
        do_sincos(a, da, n)
    } else {
        x / x
    }
}

/// glibc 2.39's binary64 cos.
pub fn cos(x: f64) -> f64 {
    let k = ((x.to_bits() >> 32) as u32) & 0x7fff_ffff;
    if k < 0x3e40_0000 {
        // |x| < 2^-27: cos(x) = 1
        1.0
    } else if k < 0x3feb_6000 {
        // 2^-27 < |x| < 0.855469
        do_cos(x, 0.0)
    } else if k < 0x4003_68fd {
        // 0.855469 < |x| < 2.426265
        let y = HP0 - x.abs();
        let a = y + HP1;
        let da = (y - a) + HP1;
        do_sin(a, da)
    } else if k < 0x4199_21FB {
        // 2.426265 < |x| < 105414350
        let (n, a, da) = reduce_sincos(x);
        do_sincos(a, da, n + 1)
    } else if k < 0x7ff0_0000 {
        // 105414350 < |x| < 2^1024
        let (n, a, da) = branred(x);
        do_sincos(a, da, n + 1)
    } else {
        x / x
    }
}

#[cfg(test)]
mod tests {
    use super::{cos, sin};

    // (x, sin x, cos x): Ubuntu glibc 2.39's words, one per branch, with the
    // branred range and three huge arguments where glibc 2.44 differs.
    const WORDS: [(u64, u64, u64); 14] = [
        (0x3fe0000000000000, 0x3fdeaee8744b05f0, 0x3fec1528065b7d50),
        (0x3ff0000000000000, 0x3feaed548f090cee, 0x3fe14a280fb5068c),
        (0x4000000000000000, 0x3fed18f6ead1b446, 0xbfdaa22657537205),
        (0x4008000000000000, 0x3fc210386db6d55b, 0xbfefae04be85e5d2),
        (0x41cdcd6500000000, 0x3fe1778cae83c69b, 0x3feacff8c7364234),
        (0x7e37e43c8800759c, 0xbfea2c16b010e385, 0xbfe2699022adc4c1),
        (0x3fb999999999999a, 0x3fb98eaecb8bcb2c, 0x3fefd712f9a817c1),
        (0x3e112e0be826d695, 0x3e112e0be826d695, 0x3ff0000000000000),
        (0x3feb5fffffffffff, 0x3fe827f6d2325577, 0x3fe4fcd8091b8f24),
        (0x400368fd00000000, 0x3fe4fcda0ad3a77a, 0xbfe827f513dbe3a2),
        (0x419921fb00000000, 0xbfe98c2ea3441c6b, 0xbfe3450a2c59b604),
        (0xeb1e50585fd26e73, 0x3fbe7a914ead0d9a, 0xbfefc5bba8036a59),
        (0x7b9b9d1e3a18bed9, 0xbfefd9f8163b393a, 0xbfb8a397e5e1946d),
        (0x43fb5fd2b98ab782, 0x3feff6e7ebd15d1e, 0x3fa81e4eb2538ae8),
    ];

    #[test]
    fn sin_and_cos_are_glibc_2_39_words() {
        for &(x, s, c) in WORDS.iter() {
            let x = f64::from_bits(x);
            assert_eq!(sin(x).to_bits(), s, "sin({x:e})");
            assert_eq!(cos(x).to_bits(), c, "cos({x:e})");
        }
    }

    #[test]
    fn specials() {
        assert_eq!(sin(-0.0).to_bits(), 0x8000_0000_0000_0000);
        assert_eq!(cos(-0.0), 1.0);
        assert!(sin(f64::INFINITY).is_nan() && cos(f64::NEG_INFINITY).is_nan());
        assert!(sin(f64::NAN).is_nan() && cos(f64::NAN).is_nan());
    }
}
