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
// e_asin.c and uasncs.h by Recast WOOF contributors, 2026.

//! `__ieee754_asin` and `__ieee754_acos` as glibc 2.39 builds them in
//! e_asin-fma.c (x86-64, `-mfma -mavx2`).  Every `mul_add` below is a
//! `vfmadd`/`vfnmadd` in that variant's disassembly; every other operation
//! is rounded on its own, as there.

#[path = "asincos_tbl.rs"]
mod tbl;

use self::tbl::{ASNCS, INROOT, POWTWO};

// uasncs.h
const HP0: f64 = 1.5707963267948966; // 0x3FF921FB54442D18
const HP1: f64 = 6.123233995736766e-17; // 0x3C91A62633145C07
const F1: f64 = 1.66666666666664110590506577996662E-01;
const F2: f64 = 7.50000000026122686814431784722623E-02;
const F3: f64 = 4.46428561421059750978517350006940E-02;
const F4: f64 = 3.03821268582119319911193410625235E-02;
const F5: f64 = 2.23551211026525610742786300334557E-02;
const F6: f64 = 1.81382903404565056280372531963613E-02;
const T24: f64 = 16777216.0;
const T27: f64 = 134217728.0;
const RT0: f64 = 9.99999999859990725855365213134618E-01;
const RT1: f64 = 4.99999999495955425917856814202739E-01;
const RT2: f64 = 3.75017500867345182581453026130850E-01;
const RT3: f64 = 3.12523626554518656309172508769531E-01;

#[inline(always)]
fn a(n: usize) -> f64 {
    f64::from_bits(ASNCS[n])
}

/// f1 + x2*(f2 + x2*(f3 + x2*(f4 + x2*(f5 + x2*f6)))), all five fused.
#[inline(always)]
fn fpoly(x2: f64) -> f64 {
    F6.mul_add(x2, F5).mul_add(x2, F4).mul_add(x2, F3).mul_add(x2, F2).mul_add(x2, F1)
}

/// The table-driven ranges: t = a[n+1]*xx + (xx*xx*(a[n+2] + xx*(... +
/// xx*a[n+deg])) + a[n+deg+1]), every Horner step and both outer products
/// fused.  The caller adds the base value a[n+deg+2].
#[inline(always)]
fn tpoly(xx: f64, n: usize, deg: usize) -> f64 {
    let mut q = a(n + deg);
    let mut i = deg;
    while i > 2 {
        i -= 1;
        q = q.mul_add(xx, a(n + i));
    }
    let p = (xx * xx).mul_add(q, a(n + deg + 1));
    xx.mul_add(a(n + 1), p)
}

/// Index and Horner degree of the table range holding k = |x|'s high word,
/// 0.125 <= |x| < 0.96875.
#[inline(always)]
fn range(k: u32) -> (usize, usize) {
    if k < 0x3fe0_0000 {
        let n = if k < 0x3fd0_0000 { 11 * ((k & 0x000f_ffff) >> 15) }
                else { 11 * ((k & 0x000f_ffff) >> 14) + 352 };
        (n as usize, 6)
    } else if k < 0x3fe8_0000 {
        (1056 + (((k & 0x000f_e000) >> 11) * 3) as usize, 7)
    } else if k < 0x3fed_8000 {
        (992 + (((k & 0x000f_e000) >> 13) * 13) as usize, 8)
    } else if k < 0x3fee_8000 {
        (884 + (((k & 0x000f_e000) >> 13) * 14) as usize, 9)
    } else {
        (768 + (((k & 0x000f_e000) >> 13) * 15) as usize, 10)
    }
}

/// z = (1 - |x|)/2 and its square-root pieces for 0.96875 <= |x| < 1:
/// returns (c, t) after the Newton step, before y is split off.
#[inline(always)]
fn root(z: f64) -> (f64, f64) {
    let k = (z.to_bits() >> 32) as u32;
    let mut t = INROOT[((k & 0x001f_ffff) >> 14) as usize] * POWTWO[(511 - (k >> 21)) as usize];
    let r = (-(t * t)).mul_add(z, 1.0);
    t *= RT3.mul_add(r, RT2).mul_add(r, RT1).mul_add(r, RT0);
    let c = t * z;
    (c, (-(0.5 * t)).mul_add(c, 1.5))
}

fn ieee754_asin(x: f64) -> f64 {
    let u = x.to_bits();
    let m = (u >> 32) as u32 as i32;
    let k = (m & 0x7fff_ffff) as u32;
    if k < 0x3e50_0000 {
        x
    } else if k < 0x3fc0_0000 {
        let x2 = x * x;
        fpoly(x2).mul_add(x * x2, x)
    } else if k < 0x3fef_0000 {
        let (n, deg) = range(k);
        let xx = if m > 0 { x - a(n) } else { -x - a(n) };
        let res = a(n + deg + 2) + tpoly(xx, n, deg);
        if m > 0 { res } else { -res }
    } else if k < 0x3ff0_0000 {
        let z = 0.5 * if m > 0 { 1.0 - x } else { 1.0 + x };
        let (c, h) = root(z);
        let y = (c + T24) - T24;
        let cc = (-y).mul_add(y, z) / h.mul_add(c, y);
        let p = fpoly(z) * z;
        let cor = (-(2.0 * (y + cc))).mul_add(p, (-2.0f64).mul_add(cc, HP1));
        let res1 = (-2.0f64).mul_add(y, HP0);
        let res = res1 + cor;
        if m > 0 { res } else { -res }
    } else if k == 0x3ff0_0000 && u as u32 == 0 {
        if m > 0 { HP0 } else { -HP0 }
    } else {
        (x - x) / (x - x)
    }
}

fn ieee754_acos(x: f64) -> f64 {
    let u = x.to_bits();
    let m = (u >> 32) as u32 as i32;
    let k = (m & 0x7fff_ffff) as u32;
    if k < 0x3c88_0000 {
        HP0
    } else if k < 0x3fc0_0000 {
        let x2 = x * x;
        let r = HP0 - x;
        let cor = (-fpoly(x2)).mul_add(x2 * x, ((HP0 - r) - x) + HP1);
        r + cor
    } else if k < 0x3fef_0000 {
        let (n, deg) = range(k);
        let xx = if m > 0 { x - a(n) } else { -x - a(n) };
        let t = tpoly(xx, n, deg);
        let b = a(n + deg + 2);
        let y = if m > 0 { HP0 - b } else { HP0 + b };
        let t = if m > 0 { HP1 - t } else { HP1 + t };
        y + t
    } else if k < 0x3ff0_0000 {
        let z = 0.5 * if m > 0 { 1.0 - x } else { 1.0 + x };
        let (c, h) = root(z);
        let y = (-T27).mul_add(c, T27.mul_add(c, c));
        let cc = (-y).mul_add(y, z) / h.mul_add(c, y);
        let p = fpoly(z) * z;
        let py = p * (y + cc);
        let res = if m < 0 { ((HP1 - cc) - py) + (HP0 - y) } else { (cc + py) + y };
        res + res
    } else if k == 0x3ff0_0000 && u as u32 == 0 {
        if m > 0 { 0.0 } else { 2.0 * HP0 }
    } else {
        (x - x) / (x - x)
    }
}

/// glibc 2.39 `asin`: the SVID-compat wrapper returns NaN for |x| > 1.
pub fn asin(x: f64) -> f64 {
    if x.abs() > 1.0 { f64::NAN } else { ieee754_asin(x) }
}

/// glibc 2.39 `acos`: the SVID-compat wrapper returns NaN for |x| > 1.
pub fn acos(x: f64) -> f64 {
    if x.abs() > 1.0 { f64::NAN } else { ieee754_acos(x) }
}

#[cfg(test)]
mod tests {
    use super::{acos, asin};

    /// (argument, result) bit patterns recorded from Ubuntu 24.04's glibc
    /// 2.39 on an FMA x86-64 host: each branch of e_asin.c and its edges.
    const ASIN: [(u64, u64); 14] = [
        (0x3e4fffffffffffff, 0x3e4fffffffffffff), (0x3fbfffffffffffff, 0x3fc00abe0c129e1e),
        (0x3fc0000000000000, 0x3fc00abe0c129e1e), (0x3fd0000000000000, 0x3fd02be9ce0b87cd),
        (0x3fdfffffffffffff, 0x3fe0c152382d7365), (0x3fe0000000000000, 0x3fe0c152382d7366),
        (0x3fe8000000000000, 0x3feb235315c680dc), (0x3fed800000000000, 0x3ff2c42a3a3c7a87),
        (0x3fee800000000000, 0x3ff436e4418e69c0), (0x3fef000000000000, 0x3ff51f4bd13f8591),
        (0x3fefffffffffffff, 0x3ff921fb50442d18), (0xbfe6a09e667f3bcd, 0xbfe921fb54442d19),
        (0xbfefe00000000001, 0xbff7b7d33b928c60), (0x3c87ffffffffffff, 0x3c87ffffffffffff),
    ];
    const ACOS: [(u64, u64); 14] = [
        (0x3e4fffffffffffff, 0x3ff921fb50442d18), (0x3fbfffffffffffff, 0x3ff720a392c1d955),
        (0x3fc0000000000000, 0x3ff720a392c1d955), (0x3fd0000000000000, 0x3ff51700e0c14b25),
        (0x3fdfffffffffffff, 0x3ff0c152382d7366), (0x3fe0000000000000, 0x3ff0c152382d7366),
        (0x3fe8000000000000, 0x3fe720a392c1d955), (0x3fed800000000000, 0x3fd97744681eca44),
        (0x3fee800000000000, 0x3fd3ac5c4ad70d60), (0x3fef000000000000, 0x3fd00abe0c129e1e),
        (0x3fefffffffffffff, 0x3e50000000000000), (0xbfe6a09e667f3bcd, 0x4002d97c7f3321d2),
        (0xbfefe00000000001, 0x40086ce747eb5cbc), (0x3c87ffffffffffff, 0x3ff921fb54442d18),
    ];

    #[test]
    fn asin_and_acos_are_glibc_2_39_words() {
        for (x, want) in ASIN {
            assert_eq!(asin(f64::from_bits(x)).to_bits(), want, "asin {x:#018x}");
        }
        for (x, want) in ACOS {
            assert_eq!(acos(f64::from_bits(x)).to_bits(), want, "acos {x:#018x}");
        }
    }

    #[test]
    fn the_ends_and_outside_the_domain() {
        assert_eq!(asin(1.0).to_bits(), 0x3ff921fb54442d18);
        assert_eq!(asin(-1.0).to_bits(), 0xbff921fb54442d18);
        assert_eq!(acos(1.0).to_bits(), 0);
        assert_eq!(acos(-1.0).to_bits(), 0x400921fb54442d18);
        assert_eq!(asin(-0.0).to_bits(), 0x8000000000000000);
        for x in [1.0000000000000002, -2.0, f64::INFINITY, f64::NAN] {
            assert!(asin(x).is_nan() && acos(x).is_nan());
        }
    }
}
