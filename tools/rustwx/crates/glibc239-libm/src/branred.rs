// IBM Accurate Mathematical Library
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
// Transcribed to Rust from glibc 2.39 sysdeps/ieee754/dbl-64/branred.c and
// branred.h by Recast WOOF contributors, 2026.  glibc builds branred.o once,
// without -mfma, and every FMA variant calls it, so nothing here is fused.

const T576: u64 = 0x63f0_0000_0000_0000; // 2^576
const TM600: f64 = f64::from_bits(0x1a70_0000_0000_0000); // 2^-600
const TM24: f64 = f64::from_bits(0x3e70_0000_0000_0000); // 2^-24
const BIG: f64 = f64::from_bits(0x4338_0000_0000_0000); // 6755399441055744
const BIG1: f64 = f64::from_bits(0x4358_0000_0000_0000); // 27021597764222976
const HP0: f64 = f64::from_bits(0x3FF9_21FB_5444_2D18); // 1.5707963267948966
const HP1: f64 = f64::from_bits(0x3C91_A626_3314_5C07); // 6.123233995736766e-17
const MP1: f64 = f64::from_bits(0x3FF9_21FB_5800_0000); // 1.5707963407039642
const MP2: f64 = f64::from_bits(0xBE4D_DE97_4000_0000); // -1.3909067675399456e-08
const SPLIT: f64 = 134217729.0; // 2^27 + 1

// 2/pi in base 2^24.
const TOVERP: [f64; 75] = [
    10680707.0, 7228996.0, 1387004.0, 2578385.0, 16069853.0,
    12639074.0, 9804092.0, 4427841.0, 16666979.0, 11263675.0,
    12935607.0, 2387514.0, 4345298.0, 14681673.0, 3074569.0,
    13734428.0, 16653803.0, 1880361.0, 10960616.0, 8533493.0,
    3062596.0, 8710556.0, 7349940.0, 6258241.0, 3772886.0,
    3769171.0, 3798172.0, 8675211.0, 12450088.0, 3874808.0,
    9961438.0, 366607.0, 15675153.0, 9132554.0, 7151469.0,
    3571407.0, 2607881.0, 12013382.0, 4155038.0, 6285869.0,
    7677882.0, 13102053.0, 15825725.0, 473591.0, 9065106.0,
    15363067.0, 6271263.0, 9264392.0, 5636912.0, 4652155.0,
    7056368.0, 13614112.0, 10155062.0, 1944035.0, 9527646.0,
    15080200.0, 6658437.0, 6231200.0, 6832269.0, 16767104.0,
    5075751.0, 3212806.0, 1398474.0, 7579849.0, 6349435.0,
    12618859.0, 4703257.0, 12806093.0, 14477321.0, 2786137.0,
    12875403.0, 9837734.0, 14528324.0, 13719321.0, 343717.0,
];

/// One half of the split argument times 2/pi: (sum, b, bb).
fn reduce_part(xp: f64) -> (f64, f64, f64) {
    let mut r = [0.0f64; 6];
    let mut sum = 0.0;
    let mut k = ((xp.to_bits() >> 52) & 2047) as i32;
    k = (k - 450) / 24;
    if k < 0 {
        k = 0;
    }
    let k = k as usize;
    let mut gor = f64::from_bits(T576 - (((k * 24) as u64) << 52));
    for i in 0..6 {
        r[i] = xp * TOVERP[k + i] * gor;
        gor *= TM24;
    }
    for i in 0..3 {
        let s = (r[i] + BIG) - BIG;
        sum += s;
        r[i] -= s;
    }
    let mut t = 0.0;
    for i in 0..6 {
        t += r[5 - i];
    }
    let mut bb = (((((r[0] - t) + r[1]) + r[2]) + r[3]) + r[4]) + r[5];
    let s = (t + BIG) - BIG;
    sum += s;
    t -= s;
    let b = t + bb;
    bb = (t - b) + bb;
    let s = (sum + BIG1) - BIG1;
    sum -= s;
    (sum, b, bb)
}

/// glibc's `__branred`: x = n*pi/2 + (a + aa), |a + aa| < pi/4.  Returns
/// (n mod 4, a, aa).
pub(super) fn branred(x: f64) -> (i32, f64, f64) {
    let x = x * TM600;
    let t = x * SPLIT; // split x to two numbers
    let x1 = t - (t - x);
    let x2 = x - x1;
    let (sum1, b1, bb1) = reduce_part(x1);
    let (sum2, b2, bb2) = reduce_part(x2);

    let mut sum = sum1 + sum2;
    let mut b = b1 + b2;
    let bb = if b1.abs() > b2.abs() { (b1 - b) + b2 } else { (b2 - b) + b1 };
    if b > 0.5 {
        b -= 1.0;
        sum += 1.0;
    } else if b < -0.5 {
        b += 1.0;
        sum -= 1.0;
    }
    let s = b + (bb + bb1 + bb2);
    let t = ((b - s) + bb) + (bb1 + bb2);
    let b = s * SPLIT;
    let t1 = b - (b - s);
    let t2 = s - t1;
    let b = s * HP0;
    let bb = (((t1 * MP1 - b) + t1 * MP2) + t2 * MP1) + (t2 * MP2 + s * HP1 + t * HP0);
    let s = b + bb;
    let t = (b - s) + bb;
    ((sum as i32) & 3, s, t)
}
