// glibc 2.39 binary64 log10 (sysdeps/ieee754/dbl-64/e_log10.c), FDLIBM's
// method around glibc's log.  glibc builds this file once, for baseline
// x86-64 without FMA, so nothing here is fused; the log it calls is the
// ifunc-selected FMA variant, super::log.
//
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.

const TWO54: f64 = f64::from_bits(0x4350000000000000);
const IVLN10: f64 = f64::from_bits(0x3fdbcb7b1526e50e);
const LOG10_2HI: f64 = f64::from_bits(0x3fd34413509f6000);
const LOG10_2LO: f64 = f64::from_bits(0x3d59fef311f12b36);

/// glibc 2.39 `log10`: the public symbol is the SVID compat wrapper, whose
/// answer for a negative argument is the default quiet NaN.
pub fn log10(x: f64) -> f64 {
    if x < 0.0 { f64::NAN } else { ieee754_log10(x) }
}

fn ieee754_log10(mut x: f64) -> f64 {
    let mut hx = x.to_bits() as i64;
    let mut k: i32 = 0;
    if hx < 0x0010000000000000 {
        // x < 2**-1022
        if hx & 0x7fffffffffffffff == 0 {
            return -TWO54 / x.abs(); // log(+-0)=-inf
        }
        if hx < 0 {
            return (x - x) / (x - x); // log(-#) = NaN
        }
        k -= 54;
        x *= TWO54; // subnormal number, scale up x
        hx = x.to_bits() as i64;
    }
    if hx >= 0x7ff0000000000000 {
        return x + x;
    }
    k += ((hx >> 52) - 1023) as i32;
    let i = ((k as i64 as u64) & 0x8000000000000000) >> 63;
    hx = (hx & 0x000fffffffffffff) | ((0x3ff - i as i64) << 52);
    let y = (k as i64 + i as i64) as f64;
    x = f64::from_bits(hx as u64);
    let z = y * LOG10_2LO + IVLN10 * super::log(x);
    z + y * LOG10_2HI
}
