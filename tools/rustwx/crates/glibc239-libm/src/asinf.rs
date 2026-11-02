// glibc 2.39 binary32 asinf (sysdeps/ieee754/flt-32/e_asinf.c), the
// Sun/Cephes form every glibc before 2.41 carries, statement for statement
// in binary32.  glibc builds it once, for baseline x86-64, so nothing is
// fused; Rust never contracts.  From glibc 2.41 on the host libm's asinf is
// CORE-MATH's correctly rounded one, which differs by one word on arguments
// such as static-fields' strict terrain-rotation probes.
//
// e_asinf.c -- float version of e_asin.c.
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
//
//   Modifications for single precision expansion are
//   Copyright (C) 2001 Stephen L. Moshier <moshier@na-net.ornl.gov>
//   and are incorporated herein by permission of the author.  The author
//   reserves the right to distribute this material elsewhere under different
//   copying permissions.  These modifications are distributed here under
//   the following terms:
//
//     This library is free software; you can redistribute it and/or
//     modify it under the terms of the GNU Lesser General Public
//     License as published by the Free Software Foundation; either
//     version 2.1 of the License, or (at your option) any later version.
//
//     This library is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//     Lesser General Public License for more details.
//
//     You should have received a copy of the GNU Lesser General Public
//     License along with this library; if not, see
//     <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Modified: transcribed to Rust from glibc 2.39 sysdeps/ieee754/flt-32/
// e_asinf.c by Recast WOOF contributors, 2026.

/// glibc 2.39 `asinf`.
pub fn asinf(x: f32) -> f32 {
    const ONE: f32 = 1.0;
    const HUGE: f32 = 1.000e+30;
    const PIO2_HI: f32 = 1.57079637050628662109375;
    const PIO2_LO: f32 = -4.37113900018624283e-8;
    const PIO4_HI: f32 = 0.785398185253143310546875;
    const P0: f32 = 1.666675248e-1;
    const P1: f32 = 7.495297643e-2;
    const P2: f32 = 4.547037598e-2;
    const P3: f32 = 2.417951451e-2;
    const P4: f32 = 4.216630880e-2;
    let hx = x.to_bits() as i32;
    let ix = hx & 0x7fff_ffff;
    if ix == 0x3f80_0000 {
        return x * PIO2_HI + x * PIO2_LO;
    } else if ix > 0x3f80_0000 {
        // The public symbol is the SVID compat wrapper: |x| > 1 gives the
        // default quiet NaN, and a NaN argument the core's (x-x)/(x-x).
        return if ix > 0x7f80_0000 { (x - x) / (x - x) } else { f32::NAN };
    } else if ix < 0x3f00_0000 {
        if ix < 0x3200_0000 {
            if HUGE + x > ONE {
                return x;
            }
        } else {
            let t = x * x;
            let w = t * (P0 + t * (P1 + t * (P2 + t * (P3 + t * P4))));
            return x + x * w;
        }
    }
    let w = ONE - x.abs();
    let mut t = w * 0.5;
    let mut p = t * (P0 + t * (P1 + t * (P2 + t * (P3 + t * P4))));
    let s = t.sqrt();
    if ix >= 0x3F79_999A {
        t = PIO2_HI - (2.0 * (s + s * p) - PIO2_LO);
    } else {
        let w = f32::from_bits(s.to_bits() & 0xffff_f000);
        let c = (t - w * w) / (s + w);
        let r = p;
        p = 2.0 * s * r - (PIO2_LO - 2.0 * c);
        let q = PIO4_HI - 2.0 * w;
        t = PIO4_HI - (p - q);
    }
    if hx > 0 { t } else { -t }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_strict_terrain_probes_are_glibc_2_39_words() {
        // One word above the correctly rounded asinf of glibc 2.41 on.
        let input = [0xbd13_7442_u32, 0xbd22_2103, 0xbd28_9002].map(f32::from_bits);
        assert_eq!(input.map(|x| super::asinf(x).to_bits()), [0xbd13_7c6b, 0xbd22_2bdc, 0xbd28_9c33]);
    }
}
