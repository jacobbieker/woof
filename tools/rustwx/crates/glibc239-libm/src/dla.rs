// IBM Accurate Mathematical Library
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
// Modified: transcribed to Rust from glibc 2.39 sysdeps/ieee754/dbl-64/dla.h
// by Recast WOOF contributors, 2026.

//! dla.h's double-length macros in the form glibc's x86-64 FMA variants
//! compile them (`__FP_FAST_FMA` defined, so EMULV is a product and its
//! exact FMA residual), as far as atan, atan2 and tan use them.  Macros
//! with a contractible product (MUL2, DIV2) are expanded at their call
//! sites instead, because GCC fuses each expansion differently.

/// EMULV(x, y, z, zz): z + zz = x * y exactly.
#[inline(always)]
pub(super) fn emulv(x: f64, y: f64) -> (f64, f64) {
    let z = x * y;
    (z, x.mul_add(y, -z))
}

/// EADD(x, y, z, zz): z + zz = x + y exactly.
#[inline(always)]
pub(super) fn eadd(x: f64, y: f64) -> (f64, f64) {
    let z = x + y;
    let zz = if x.abs() > y.abs() { (x - z) + y } else { (y - z) + x };
    (z, zz)
}

/// ESUB(x, y, z, zz): z + zz = x - y exactly.
#[inline(always)]
pub(super) fn esub(x: f64, y: f64) -> (f64, f64) {
    let z = x - y;
    let zz = if x.abs() > y.abs() { (x - z) - y } else { x - (y + z) };
    (z, zz)
}
