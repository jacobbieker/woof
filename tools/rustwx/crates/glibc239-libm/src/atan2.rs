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
// Modified: transcribed to Rust from glibc 2.39 sysdeps/ieee754/dbl-64/e_atan2.c and atnat2.h
// by Recast WOOF contributors, 2026.

//! glibc 2.39 `__ieee754_atan2`, as `__ieee754_atan2_fma` (the x86-64
//! ifunc variant built with -mfma -mavx2) computes it.  GCC fuses a
//! product only into an addition in its own basic block, so the small-u
//! branches that finish through ESUB or EADD keep `zz` unfused.

use super::atan::{cij, poly_d, row};
use super::dla::{eadd, emulv, esub};

// atnat2.h constants.
const INV16: f64 = f64::from_bits(0x3fb0000000000000);
const OPI: f64 = f64::from_bits(0x400921fb54442d18);
const OPI1: f64 = f64::from_bits(0x3ca1a62633145c07);
const MOPI: f64 = f64::from_bits(0xc00921fb54442d18);
const HPI: f64 = f64::from_bits(0x3ff921fb54442d18);
const HPI1: f64 = f64::from_bits(0x3c91a62633145c07);
const MHPI: f64 = f64::from_bits(0xbff921fb54442d18);
const QPI: f64 = f64::from_bits(0x3fe921fb54442d18);
const MQPI: f64 = f64::from_bits(0xbfe921fb54442d18);
const TQPI: f64 = f64::from_bits(0x4002d97c7f3321d2);
const MTQPI: f64 = f64::from_bits(0xc002d97c7f3321d2);
const TWO500: f64 = f64::from_bits(0x5f30000000000000);
const TWOM500: f64 = f64::from_bits(0x20b0000000000000);

/// `a + b` with a NaN operand as SSE2 computes it (the first NaN operand,
/// quieted), whichever operand order or NaN rule the target has.
#[inline(always)]
pub(super) fn sse_nan_sum(a: f64, b: f64) -> f64 {
    let n = if a.is_nan() { a } else { b };
    f64::from_bits(n.to_bits() | 0x0008_0000_0000_0000)
}

pub fn atan2(y: f64, x: f64) -> f64 {
    const EP: i32 = 59768832; // 57*16**5
    const EM: i32 = -59768832;
    let xb = x.to_bits();
    let ux = (xb >> 32) as u32;
    let dx = xb as u32;
    if (ux & 0x7ff00000) == 0x7ff00000 && ((ux & 0x000fffff) | dx) != 0 {
        return sse_nan_sum(x, y);
    }
    let yb = y.to_bits();
    let uy = (yb >> 32) as u32;
    let dy = yb as u32;
    if (uy & 0x7ff00000) == 0x7ff00000 && ((uy & 0x000fffff) | dy) != 0 {
        return sse_nan_sum(y, y);
    }

    // y = +-0
    if uy == 0 && dy == 0 {
        return if ux & 0x80000000 == 0 { 0.0 } else { OPI };
    } else if uy == 0x80000000 && dy == 0 {
        return if ux & 0x80000000 == 0 { -0.0 } else { MOPI };
    }
    // x = +-0
    if x == 0.0 {
        return if uy & 0x80000000 == 0 { HPI } else { MHPI };
    }
    // x = +-INF
    if ux == 0x7ff00000 {
        if dx == 0 {
            if uy == 0x7ff00000 {
                if dy == 0 {
                    return QPI;
                }
            } else if uy == 0xfff00000 {
                if dy == 0 {
                    return MQPI;
                }
            } else {
                return if uy & 0x80000000 == 0 { 0.0 } else { -0.0 };
            }
        }
    } else if ux == 0xfff00000 && dx == 0 {
        if uy == 0x7ff00000 {
            if dy == 0 {
                return TQPI;
            }
        } else if uy == 0xfff00000 {
            if dy == 0 {
                return MTQPI;
            }
        } else {
            return if uy & 0x80000000 == 0 { OPI } else { MOPI };
        }
    }
    // y = +-INF
    if uy == 0x7ff00000 {
        if dy == 0 {
            return HPI;
        }
    } else if uy == 0xfff00000 && dy == 0 {
        return MHPI;
    }

    let mut ax = if x < 0.0 { -x } else { x };
    let mut ay = if y < 0.0 { -y } else { y };
    let de = (uy & 0x7ff00000) as i32 - (ux & 0x7ff00000) as i32;
    if de >= EP {
        return if y > 0.0 { HPI } else { MHPI };
    } else if de <= EM {
        if x > 0.0 {
            return (ay / ax).copysign(y);
        }
        return if y > 0.0 { OPI } else { MOPI };
    }

    // Scale abs(x), abs(y) away from the extremes.
    if ax < TWOM500 || ay < TWOM500 {
        ax *= TWO500;
        ay *= TWO500;
    }
    if ax > TWO500 || ay > TWO500 {
        ax *= TWOM500;
        ay *= TWOM500;
    }

    let (u, du);
    if ay < ax {
        u = ay / ax;
        let (v, vv) = emulv(ax, u);
        du = ((ay - v) - vv) / ax;
    } else {
        u = ax / ay;
        let (v, vv) = emulv(ay, u);
        du = ((ax - v) - vv) / ay;
    }

    // cij[3] + v * (cij[4] + v * (cij[5] + v * cij[6])), and the same
    // continued through cij[2], each step fused.
    let poly36 = |i: usize, v: f64| {
        let p = v.mul_add(cij(i, 6), cij(i, 5));
        let p = v.mul_add(p, cij(i, 4));
        v.mul_add(p, cij(i, 3))
    };
    let poly26 = |i: usize, v: f64| v.mul_add(poly36(i, v), cij(i, 2));

    if x > 0.0 {
        // (i) x > 0, abs(y) < abs(x): atan(ay/ax)
        if ay < ax {
            if u < INV16 {
                let v = u * u;
                let zz = (u * v).mul_add(poly_d(v), du);
                return (u + zz).copysign(y);
            }
            let i = row(u);
            let t3 = u - cij(i, 0);
            let (v, dv) = eadd(t3, du);
            let t1 = cij(i, 1);
            let t2 = cij(i, 2);
            let zz = v.mul_add(t2, dv.mul_add(t2, (v * v) * poly36(i, v)));
            return (t1 + zz).copysign(y);
        }
        // (ii) x > 0, abs(x) <= abs(y): pi/2 - atan(ax/ay)
        if u < INV16 {
            let v = u * u;
            let zz = (u * v) * poly_d(v);
            let (t2, cor) = esub(HPI, u);
            let t3 = ((HPI1 + cor) - du) - zz;
            return (t2 + t3).copysign(y);
        }
        let i = row(u);
        let v = (u - cij(i, 0)) + du;
        let zz = (-v).mul_add(poly26(i, v), HPI1);
        let t1 = HPI - cij(i, 1);
        return (t1 + zz).copysign(y);
    }

    // (iii) x < 0, abs(x) < abs(y): pi/2 + atan(ax/ay)
    if ax < ay {
        if u < INV16 {
            let v = u * u;
            let zz = (u * v) * poly_d(v);
            let (t2, cor) = eadd(HPI, u);
            let t3 = ((HPI1 + cor) + du) + zz;
            return (t2 + t3).copysign(y);
        }
        let i = row(u);
        let v = (u - cij(i, 0)) + du;
        let zz = v.mul_add(poly26(i, v), HPI1);
        let t1 = HPI + cij(i, 1);
        return (t1 + zz).copysign(y);
    }

    // (iv) x < 0, abs(y) <= abs(x): pi - atan(ax/ay)
    if u < INV16 {
        let v = u * u;
        let zz = (u * v) * poly_d(v);
        let (t2, cor) = esub(OPI, u);
        let t3 = ((OPI1 + cor) - du) - zz;
        return (t2 + t3).copysign(y);
    }
    let i = row(u);
    let v = (u - cij(i, 0)) + du;
    let zz = (-v).mul_add(poly26(i, v), OPI1);
    let t1 = OPI - cij(i, 1);
    (t1 + zz).copysign(y)
}

#[cfg(test)]
mod tests {
    use super::atan2;

    #[test]
    fn atan2_is_glibc_2_39s_words() {
        // (y, x, glibc 2.39 x86-64 result), recorded from Ubuntu 24.04.
        let pins: [(u64, u64, u64); 13] = [
            (0x3ff0000000000000, 0x3ff0000000000000, 0x3fe921fb54442d18),
            (0x8000000000000000, 0xbff0000000000000, 0xc00921fb54442d18),
            (0x3f50624dd2f1a9fc, 0x3ff0000000000000, 0x3f50624d77516e16),
            (0x3fd3333333333333, 0x3feccccccccccccd, 0x3fd4978fa3269ee1),
            (0x4000000000000000, 0x3fb999999999999a, 0x3ff8555a2787981f),
            (0xc008000000000000, 0x3fc999999999999a, 0xbff811518cde39a6),
            (0x3fe0000000000000, 0xc010000000000000, 0x4008234d7f6ecb9d),
            (0xbf847ae147ae147b, 0xc000000000000000, 0xc00917bde937f86f),
            (0x4014000000000000, 0xbfd0000000000000, 0x3ff9ee9c8100c211),
            (0xc156e36000000000, 0x410e848000000000, 0xbff87769eb8e956b),
            (0x40fd4c0000000000, 0x415d5f8800000000, 0x3f8fea0f894fbe16),
            (0x01a56e1fc2f8f359, 0x7e37e43c8800759c, 0x0000000000000000),
            (0x7ff0000000000000, 0xfff0000000000000, 0x4002d97c7f3321d2),
        ];
        for (y, x, want) in pins {
            let got = atan2(f64::from_bits(y), f64::from_bits(x)).to_bits();
            assert_eq!(got, want, "atan2({y:#018x}, {x:#018x})");
        }
        // A NaN operand returns the first NaN, quieted, as SSE2 adds.
        let (q, s) = (f64::from_bits(0x7ffa_3ba2_dae2_1134), f64::from_bits(0xfff2_f1d1_e02b_5f98));
        assert_eq!(atan2(q, s).to_bits(), 0xfffa_f1d1_e02b_5f98);
    }
}
