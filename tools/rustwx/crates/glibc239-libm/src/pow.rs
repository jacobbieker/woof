// glibc 2.39 binary64 pow (e_pow.c), FMA variant.
//
// Arm optimized-routines (math/pow.c, math/pow_log_data.c, math/exp_data.c), the code glibc 2.39 carries in
// sysdeps/ieee754/dbl-64/, taken from the upstream MIT sources kept at
// tools/uwpbl_wrf471_oracle/libm64/upstream/.  The tables are generated
// from those sources and equal glibc 2.39's word for word.  Fused sites are
// the installed x86-64 FMA variant's (see
// tools/uwpbl_wrf471_oracle/libm64/FMA-SITES.md).
//
// Copyright (c) 2018-2026, Arm Limited.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the Software),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

use super::exp::{SHIFT, POLY as EXP_POLY, TAB as EXP_TAB};

const LN2_HI: f64 = f64::from_bits(0x3fe62e42fefa3800);
const LN2_LO: f64 = f64::from_bits(0x3d2ef35793c76730);
const INV_LN2_N: f64 = f64::from_bits(0x40671547652b82fe);
const NEG_LN2_HI_N: f64 = f64::from_bits(0xbf762e42fefa0000);
const NEG_LN2_LO_N: f64 = f64::from_bits(0xbd0cf79abc9e3b3a);
const LOG_TABLE_BITS: u32 = 7;
const EXP_TABLE_BITS: u32 = 7;
const OFF: u64 = 0x3fe6955500000000;
const SIGN_BIAS: u32 = 0x800 << EXP_TABLE_BITS;

#[inline]
fn top12(x: f64) -> u32 { (x.to_bits() >> 52) as u32 }
#[inline]
fn a(i: usize) -> f64 { f64::from_bits(POLY[i]) }
#[inline]
fn c(i: usize) -> f64 { f64::from_bits(EXP_POLY[i]) }

/// log(x) as hi + tail, for x's bits ix (normalized, positive).
fn log_inline(ix: u64) -> (f64, f64) {
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> (52 - LOG_TABLE_BITS)) % (1 << LOG_TABLE_BITS)) as usize;
    let k = (tmp as i64) >> 52;
    let iz = ix.wrapping_sub(tmp & (0xfffu64 << 52));
    let z = f64::from_bits(iz);
    let kd = k as f64;
    let invc = f64::from_bits(TAB[3 * i]);
    let logc = f64::from_bits(TAB[3 * i + 1]);
    let logctail = f64::from_bits(TAB[3 * i + 2]);
    let r = z.mul_add(invc, -1.0);
    let t1 = kd.mul_add(LN2_HI, logc);
    let t2 = t1 + r;
    let lo1 = kd.mul_add(LN2_LO, logctail);
    let lo2 = t1 - t2 + r;
    let ar = a(0) * r;
    let ar2 = r * ar;
    let ar3 = r * ar2;
    let hi = t2 + ar2;
    let lo3 = ar.mul_add(r, -ar2);
    let lo4 = t2 - hi + ar2;
    let p = ar2.mul_add(ar2.mul_add(r.mul_add(a(6), a(5)), r.mul_add(a(4), a(3))),
                        r.mul_add(a(2), a(1)));
    let lo = ar3.mul_add(p, lo1 + lo2 + lo3 + lo4);
    let y = hi + lo;
    (y, hi - y + lo)
}

fn specialcase(tmp: f64, mut sbits: u64, ki: u64) -> f64 {
    if ki & 0x8000_0000 == 0 {
        sbits = sbits.wrapping_sub(1009u64 << 52);
        let scale = f64::from_bits(sbits);
        return scale.mul_add(tmp, scale) * f64::from_bits(0x7f00000000000000);
    }
    sbits = sbits.wrapping_add(1022u64 << 52);
    let scale = f64::from_bits(sbits);
    let mut y = scale + scale * tmp;
    if y.abs() < 1.0 {
        let one = if y < 0.0 { -1.0 } else { 1.0 };
        let mut lo = scale - y + scale * tmp;
        let hi = one + y;
        lo = one - hi + y + lo;
        y = (hi + lo) - one;
        if y == 0.0 {
            y = f64::from_bits(sbits & 0x8000000000000000);
        }
    }
    f64::from_bits(0x0010000000000000) * y
}

fn exp_inline(x: f64, xtail: f64, sign_bias: u32) -> f64 {
    let mut abstop = top12(x) & 0x7ff;
    let tiny = top12(f64::from_bits(0x3c90000000000000));
    if abstop.wrapping_sub(tiny) >= top12(512.0) - tiny {
        if abstop.wrapping_sub(tiny) >= 0x8000_0000 {
            let one = 1.0 + x;
            return if sign_bias != 0 { -one } else { one };
        }
        if abstop >= top12(1024.0) {
            let negative = sign_bias != 0;
            return if x.to_bits() >> 63 != 0 {
                if negative { -0.0 } else { 0.0 }
            } else if negative { f64::NEG_INFINITY } else { f64::INFINITY };
        }
        abstop = 0;
    }
    let kd = INV_LN2_N.mul_add(x, SHIFT);
    let ki = kd.to_bits();
    let kd = kd - SHIFT;
    let mut r = kd.mul_add(NEG_LN2_LO_N, kd.mul_add(NEG_LN2_HI_N, x));
    r += xtail;
    let idx = (2 * (ki % (1 << EXP_TABLE_BITS))) as usize;
    let top = ki.wrapping_add(sign_bias as u64) << (52 - EXP_TABLE_BITS);
    let tail = f64::from_bits(EXP_TAB[idx]);
    let sbits = EXP_TAB[idx + 1].wrapping_add(top);
    let r2 = r * r;
    let tmp = (r2 * r2).mul_add(r.mul_add(c(3), c(2)), r2.mul_add(r.mul_add(c(1), c(0)), tail + r));
    if abstop == 0 {
        return specialcase(tmp, sbits, ki);
    }
    let scale = f64::from_bits(sbits);
    scale.mul_add(tmp, scale)
}

/// 0: y is not an integer, 1: odd integer, 2: even integer.
fn checkint(iy: u64) -> i32 {
    let e = (iy >> 52 & 0x7ff) as i32;
    if e < 0x3ff {
        return 0;
    }
    if e > 0x3ff + 52 {
        return 2;
    }
    if iy & ((1u64 << (0x3ff + 52 - e)) - 1) != 0 {
        return 0;
    }
    if iy & (1u64 << (0x3ff + 52 - e)) != 0 {
        return 1;
    }
    2
}

#[inline]
fn zeroinfnan(i: u64) -> bool {
    i.wrapping_mul(2).wrapping_sub(1) >= f64::INFINITY.to_bits().wrapping_mul(2) - 1
}

#[inline]
fn issignaling(x: f64) -> bool {
    let u = x.to_bits();
    u & 0x7ff8000000000000 == 0x7ff0000000000000 && u & 0x000fffffffffffff != 0
}

/// glibc 2.39 `pow`.
pub fn pow(x: f64, y: f64) -> f64 {
    let mut sign_bias = 0u32;
    let mut ix = x.to_bits();
    let iy = y.to_bits();
    let mut topx = top12(x);
    let topy = top12(y);
    if topx.wrapping_sub(0x001) >= 0x7ff - 0x001
        || (topy & 0x7ff).wrapping_sub(0x3be) >= 0x43e - 0x3be {
        // x is subnormal, zero, inf, nan or negative, or |y| is tiny or huge.
        if zeroinfnan(iy) {
            if iy.wrapping_mul(2) == 0 {
                return if issignaling(x) { x + y } else { 1.0 };
            }
            if ix == 1.0f64.to_bits() {
                return if issignaling(y) { x + y } else { 1.0 };
            }
            let inf2 = f64::INFINITY.to_bits().wrapping_mul(2);
            if ix.wrapping_mul(2) > inf2 || iy.wrapping_mul(2) > inf2 {
                return x + y;
            }
            if ix.wrapping_mul(2) == 1.0f64.to_bits().wrapping_mul(2) {
                return 1.0;
            }
            if (ix.wrapping_mul(2) < 1.0f64.to_bits().wrapping_mul(2)) == (iy >> 63 == 0) {
                return 0.0; // |x| < 1 && y == inf, or |x| > 1 && y == -inf.
            }
            return y * y;
        }
        if zeroinfnan(ix) {
            let mut x2 = x * x;
            if ix >> 63 != 0 && checkint(iy) == 1 {
                x2 = -x2;
            }
            return if iy >> 63 != 0 { 1.0 / x2 } else { x2 };
        }
        // Here x and y are non-zero finite.
        if ix >> 63 != 0 {
            // Finite x < 0.
            let yint = checkint(iy);
            if yint == 0 {
                return (x - x) / (x - x);
            }
            if yint == 1 {
                sign_bias = SIGN_BIAS;
            }
            ix &= 0x7fffffffffffffff;
            topx &= 0x7ff;
        }
        if (topy & 0x7ff).wrapping_sub(0x3be) >= 0x43e - 0x3be {
            // Note: sign_bias == 0 here because y is not odd.
            if ix == 1.0f64.to_bits() {
                return 1.0;
            }
            if (topy & 0x7ff) < 0x3be {
                // |y| < 2^-65, x^y ~= 1 + y*log(x).
                return if ix > 1.0f64.to_bits() { 1.0 + y } else { 1.0 - y };
            }
            return if (ix > 1.0f64.to_bits()) == (topy < 0x800) { f64::INFINITY } else { 0.0 };
        }
        if topx == 0 {
            // Normalize subnormal x so exponent becomes negative.
            ix = (x * f64::from_bits(0x4330000000000000)).to_bits();
            ix &= 0x7fffffffffffffff;
            ix = ix.wrapping_sub(52u64 << 52);
        }
    }
    let (hi, lo) = log_inline(ix);
    let ehi = y * hi;
    let elo = y.mul_add(lo, y.mul_add(hi, -ehi));
    exp_inline(ehi, elo, sign_bias)
}

// The log table (invc, logc, logctail) and polynomial, as words.
static POLY: [u64; 7] = [
    0xbfe0000000000000,
    0xbfe5555555555560,
    0x3fe0000000000006,
    0x3fe999999959554e,
    0xbfe555555529a47a,
    0xbff2495b9b4845e9,
    0x3ff0002b8b263fc3,
];
static TAB: [u64; 384] = [
    0x3ff6a00000000000, 0xbfd62c82f2b9c800, 0x3cfab42428375680,
    0x3ff6800000000000, 0xbfd5d1bdbf580800, 0xbd1ca508d8e0f720,
    0x3ff6600000000000, 0xbfd5767717455800, 0xbd2362a4d5b6506d,
    0x3ff6400000000000, 0xbfd51aad872df800, 0xbce684e49eb067d5,
    0x3ff6200000000000, 0xbfd4be5f95777800, 0xbd041b6993293ee0,
    0x3ff6000000000000, 0xbfd4618bc21c6000, 0x3d13d82f484c84cc,
    0x3ff5e00000000000, 0xbfd404308686a800, 0x3cdc42f3ed820b3a,
    0x3ff5c00000000000, 0xbfd3a64c55694800, 0x3d20b1c686519460,
    0x3ff5a00000000000, 0xbfd347dd9a988000, 0x3d25594dd4c58092,
    0x3ff5800000000000, 0xbfd2e8e2bae12000, 0x3d267b1e99b72bd8,
    0x3ff5600000000000, 0xbfd2895a13de8800, 0x3d15ca14b6cfb03f,
    0x3ff5600000000000, 0xbfd2895a13de8800, 0x3d15ca14b6cfb03f,
    0x3ff5400000000000, 0xbfd22941fbcf7800, 0xbd165a242853da76,
    0x3ff5200000000000, 0xbfd1c898c1699800, 0xbd1fafbc68e75404,
    0x3ff5000000000000, 0xbfd1675cababa800, 0x3d1f1fc63382a8f0,
    0x3ff4e00000000000, 0xbfd1058bf9ae4800, 0xbd26a8c4fd055a66,
    0x3ff4c00000000000, 0xbfd0a324e2739000, 0xbd0c6bee7ef4030e,
    0x3ff4a00000000000, 0xbfd0402594b4d000, 0xbcf036b89ef42d7f,
    0x3ff4a00000000000, 0xbfd0402594b4d000, 0xbcf036b89ef42d7f,
    0x3ff4800000000000, 0xbfcfb9186d5e4000, 0x3d0d572aab993c87,
    0x3ff4600000000000, 0xbfcef0adcbdc6000, 0x3d2b26b79c86af24,
    0x3ff4400000000000, 0xbfce27076e2af000, 0xbd172f4f543fff10,
    0x3ff4200000000000, 0xbfcd5c216b4fc000, 0x3d21ba91bbca681b,
    0x3ff4000000000000, 0xbfcc8ff7c79aa000, 0x3d27794f689f8434,
    0x3ff4000000000000, 0xbfcc8ff7c79aa000, 0x3d27794f689f8434,
    0x3ff3e00000000000, 0xbfcbc286742d9000, 0x3d194eb0318bb78f,
    0x3ff3c00000000000, 0xbfcaf3c94e80c000, 0x3cba4e633fcd9066,
    0x3ff3a00000000000, 0xbfca23bc1fe2b000, 0xbd258c64dc46c1ea,
    0x3ff3a00000000000, 0xbfca23bc1fe2b000, 0xbd258c64dc46c1ea,
    0x3ff3800000000000, 0xbfc9525a9cf45000, 0xbd2ad1d904c1d4e3,
    0x3ff3600000000000, 0xbfc87fa06520d000, 0x3d2bbdbf7fdbfa09,
    0x3ff3400000000000, 0xbfc7ab890210e000, 0x3d2bdb9072534a58,
    0x3ff3400000000000, 0xbfc7ab890210e000, 0x3d2bdb9072534a58,
    0x3ff3200000000000, 0xbfc6d60fe719d000, 0xbd10e46aa3b2e266,
    0x3ff3000000000000, 0xbfc5ff3070a79000, 0xbd1e9e439f105039,
    0x3ff3000000000000, 0xbfc5ff3070a79000, 0xbd1e9e439f105039,
    0x3ff2e00000000000, 0xbfc526e5e3a1b000, 0xbd20de8b90075b8f,
    0x3ff2c00000000000, 0xbfc44d2b6ccb8000, 0x3d170cc16135783c,
    0x3ff2c00000000000, 0xbfc44d2b6ccb8000, 0x3d170cc16135783c,
    0x3ff2a00000000000, 0xbfc371fc201e9000, 0x3cf178864d27543a,
    0x3ff2800000000000, 0xbfc29552f81ff000, 0xbd248d301771c408,
    0x3ff2600000000000, 0xbfc1b72ad52f6000, 0xbd2e80a41811a396,
    0x3ff2600000000000, 0xbfc1b72ad52f6000, 0xbd2e80a41811a396,
    0x3ff2400000000000, 0xbfc0d77e7cd09000, 0x3d0a699688e85bf4,
    0x3ff2400000000000, 0xbfc0d77e7cd09000, 0x3d0a699688e85bf4,
    0x3ff2200000000000, 0xbfbfec9131dbe000, 0xbd2575545ca333f2,
    0x3ff2000000000000, 0xbfbe27076e2b0000, 0x3d2a342c2af0003c,
    0x3ff2000000000000, 0xbfbe27076e2b0000, 0x3d2a342c2af0003c,
    0x3ff1e00000000000, 0xbfbc5e548f5bc000, 0xbd1d0c57585fbe06,
    0x3ff1c00000000000, 0xbfba926d3a4ae000, 0x3d253935e85baac8,
    0x3ff1c00000000000, 0xbfba926d3a4ae000, 0x3d253935e85baac8,
    0x3ff1a00000000000, 0xbfb8c345d631a000, 0x3d137c294d2f5668,
    0x3ff1a00000000000, 0xbfb8c345d631a000, 0x3d137c294d2f5668,
    0x3ff1800000000000, 0xbfb6f0d28ae56000, 0xbd269737c93373da,
    0x3ff1600000000000, 0xbfb51b073f062000, 0x3d1f025b61c65e57,
    0x3ff1600000000000, 0xbfb51b073f062000, 0x3d1f025b61c65e57,
    0x3ff1400000000000, 0xbfb341d7961be000, 0x3d2c5edaccf913df,
    0x3ff1400000000000, 0xbfb341d7961be000, 0x3d2c5edaccf913df,
    0x3ff1200000000000, 0xbfb16536eea38000, 0x3d147c5e768fa309,
    0x3ff1000000000000, 0xbfaf0a30c0118000, 0x3d2d599e83368e91,
    0x3ff1000000000000, 0xbfaf0a30c0118000, 0x3d2d599e83368e91,
    0x3ff0e00000000000, 0xbfab42dd71198000, 0x3d1c827ae5d6704c,
    0x3ff0e00000000000, 0xbfab42dd71198000, 0x3d1c827ae5d6704c,
    0x3ff0c00000000000, 0xbfa77458f632c000, 0xbd2cfc4634f2a1ee,
    0x3ff0c00000000000, 0xbfa77458f632c000, 0xbd2cfc4634f2a1ee,
    0x3ff0a00000000000, 0xbfa39e87b9fec000, 0x3cf502b7f526feaa,
    0x3ff0a00000000000, 0xbfa39e87b9fec000, 0x3cf502b7f526feaa,
    0x3ff0800000000000, 0xbf9f829b0e780000, 0xbd2980267c7e09e4,
    0x3ff0800000000000, 0xbf9f829b0e780000, 0xbd2980267c7e09e4,
    0x3ff0600000000000, 0xbf97b91b07d58000, 0xbd288d5493faa639,
    0x3ff0400000000000, 0xbf8fc0a8b0fc0000, 0xbcdf1e7cf6d3a69c,
    0x3ff0400000000000, 0xbf8fc0a8b0fc0000, 0xbcdf1e7cf6d3a69c,
    0x3ff0200000000000, 0xbf7fe02a6b100000, 0xbd19e23f0dda40e4,
    0x3ff0200000000000, 0xbf7fe02a6b100000, 0xbd19e23f0dda40e4,
    0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000,
    0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000,
    0x3fefc00000000000, 0x3f80101575890000, 0xbd10c76b999d2be8,
    0x3fef800000000000, 0x3f90205658938000, 0xbd23dc5b06e2f7d2,
    0x3fef400000000000, 0x3f98492528c90000, 0xbd2aa0ba325a0c34,
    0x3fef000000000000, 0x3fa0415d89e74000, 0x3d0111c05cf1d753,
    0x3feec00000000000, 0x3fa466aed42e0000, 0xbd2c167375bdfd28,
    0x3fee800000000000, 0x3fa894aa149fc000, 0xbd197995d05a267d,
    0x3fee400000000000, 0x3faccb73cdddc000, 0xbd1a68f247d82807,
    0x3fee200000000000, 0x3faeea31c006c000, 0xbd0e113e4fc93b7b,
    0x3fede00000000000, 0x3fb1973bd1466000, 0xbd25325d560d9e9b,
    0x3feda00000000000, 0x3fb3bdf5a7d1e000, 0x3d2cc85ea5db4ed7,
    0x3fed600000000000, 0x3fb5e95a4d97a000, 0xbd2c69063c5d1d1e,
    0x3fed400000000000, 0x3fb700d30aeac000, 0x3cec1e8da99ded32,
    0x3fed000000000000, 0x3fb9335e5d594000, 0x3d23115c3abd47da,
    0x3fecc00000000000, 0x3fbb6ac88dad6000, 0xbd1390802bf768e5,
    0x3feca00000000000, 0x3fbc885801bc4000, 0x3d2646d1c65aacd3,
    0x3fec600000000000, 0x3fbec739830a2000, 0xbd2dc068afe645e0,
    0x3fec400000000000, 0x3fbfe89139dbe000, 0xbd2534d64fa10afd,
    0x3fec000000000000, 0x3fc1178e8227e000, 0x3d21ef78ce2d07f2,
    0x3febe00000000000, 0x3fc1aa2b7e23f000, 0x3d2ca78e44389934,
    0x3feba00000000000, 0x3fc2d1610c868000, 0x3d039d6ccb81b4a1,
    0x3feb800000000000, 0x3fc365fcb0159000, 0x3cc62fa8234b7289,
    0x3feb400000000000, 0x3fc4913d8333b000, 0x3d25837954fdb678,
    0x3feb200000000000, 0x3fc527e5e4a1b000, 0x3d2633e8e5697dc7,
    0x3feae00000000000, 0x3fc6574ebe8c1000, 0x3d19cf8b2c3c2e78,
    0x3feac00000000000, 0x3fc6f0128b757000, 0xbd25118de59c21e1,
    0x3feaa00000000000, 0x3fc7898d85445000, 0xbd1c661070914305,
    0x3fea600000000000, 0x3fc8beafeb390000, 0xbd073d54aae92cd1,
    0x3fea400000000000, 0x3fc95a5adcf70000, 0x3d07f22858a0ff6f,
    0x3fea000000000000, 0x3fca93ed3c8ae000, 0xbd28724350562169,
    0x3fe9e00000000000, 0x3fcb31d8575bd000, 0xbd0c358d4eace1aa,
    0x3fe9c00000000000, 0x3fcbd087383be000, 0xbd2d4bc4595412b6,
    0x3fe9a00000000000, 0x3fcc6ffbc6f01000, 0xbcf1ec72c5962bd2,
    0x3fe9600000000000, 0x3fcdb13db0d49000, 0xbd2aff2af715b035,
    0x3fe9400000000000, 0x3fce530effe71000, 0x3cc212276041f430,
    0x3fe9200000000000, 0x3fcef5ade4dd0000, 0xbcca211565bb8e11,
    0x3fe9000000000000, 0x3fcf991c6cb3b000, 0x3d1bcbecca0cdf30,
    0x3fe8c00000000000, 0x3fd07138604d5800, 0x3cf89cdb16ed4e91,
    0x3fe8a00000000000, 0x3fd0c42d67616000, 0x3d27188b163ceae9,
    0x3fe8800000000000, 0x3fd1178e8227e800, 0xbd2c210e63a5f01c,
    0x3fe8600000000000, 0x3fd16b5ccbacf800, 0x3d2b9acdf7a51681,
    0x3fe8400000000000, 0x3fd1bf99635a6800, 0x3d2ca6ed5147bdb7,
    0x3fe8200000000000, 0x3fd214456d0eb800, 0x3d0a87deba46baea,
    0x3fe7e00000000000, 0x3fd2bef07cdc9000, 0x3d2a9cfa4a5004f4,
    0x3fe7c00000000000, 0x3fd314f1e1d36000, 0xbd28e27ad3213cb8,
    0x3fe7a00000000000, 0x3fd36b6776be1000, 0x3d116ecdb0f177c8,
    0x3fe7800000000000, 0x3fd3c25277333000, 0x3d183b54b606bd5c,
    0x3fe7600000000000, 0x3fd419b423d5e800, 0x3d08e436ec90e09d,
    0x3fe7400000000000, 0x3fd4718dc271c800, 0xbd2f27ce0967d675,
    0x3fe7200000000000, 0x3fd4c9e09e173000, 0xbd2e20891b0ad8a4,
    0x3fe7000000000000, 0x3fd522ae0738a000, 0x3d2ebe708164c759,
    0x3fe6e00000000000, 0x3fd57bf753c8d000, 0x3d1fadedee5d40ef,
    0x3fe6c00000000000, 0x3fd5d5bddf596000, 0xbd0a0b2a08a465dc,
];
