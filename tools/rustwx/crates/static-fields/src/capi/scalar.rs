//! Scalar rounding at the strict terrain initialization boundary.
//!
//! This is initialization of a static rotation plane, once per domain.
//! It does not replace the device math routines or any time-step producer.

use super::{clear_error, guard, set_error, ERR, OK};

/// glibc 2.39 `sysdeps/ieee754/flt-32/e_asinf.c` (the Sun/Cephes form every
/// glibc before 2.41 carries), statement for statement in binary32.  The
/// reference WRF builds link glibc 2.39; a host libm from glibc 2.41 on is
/// CORE-MATH's correctly rounded `asinf`, which differs by one word on
/// arguments such as the contract probes, so the bridge does not call the
/// process libm.  Rust never contracts these into FMA.
fn glibc239_asinf(x: f32) -> f32 {
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
        return (x - x) / (x - x);
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

/// Evaluate the reference scalar inverse sine on input binary32 words.
/// The caller validates the scalar-rounding contract before using this
/// optional entry point in a strict forecast.
///
/// # Safety
/// Input and output describe `count` aligned, nonoverlapping binary32 words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_scalar_asin_f32(
    input: *const f32,
    output: *mut f32,
    count: usize,
) -> i32 {
    guard(ERR, || {
        clear_error();
        if count == 0 {
            return OK;
        }
        if input.is_null() || output.is_null()
            || count > isize::MAX as usize / std::mem::size_of::<f32>()
        {
            return set_error("invalid scalar rotation input/output buffer");
        }
        let input = unsafe { std::slice::from_raw_parts(input, count) };
        let output = unsafe { std::slice::from_raw_parts_mut(output, count) };
        for (&value, result) in input.iter().zip(output.iter_mut()) {
            *result = glibc239_asinf(value);
        }
        OK
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_rotation_preserves_zero_sign_and_endpoints() {
        let input = std::hint::black_box([0.0_f32, -0.0, 1.0, -1.0]);
        let mut output = [0.0; 4];
        assert_eq!(unsafe {
            gpuwm_static_scalar_asin_f32(input.as_ptr(), output.as_mut_ptr(), input.len())
        }, OK);
        assert_eq!(output.map(f32::to_bits), [0, 0x8000_0000, 0x3fc9_0fdb, 0xbfc9_0fdb]);
    }

    #[test]
    fn scalar_rotation_is_glibc_2_39_asinf_not_the_process_libm() {
        // The bridge's qualification probes: glibc 2.39 words, one above
        // the correctly rounded asinf of glibc 2.41 and later.
        let input = [0xbd13_7442_u32, 0xbd22_2103, 0xbd28_9002].map(f32::from_bits);
        let mut output = [0.0; 3];
        assert_eq!(unsafe {
            gpuwm_static_scalar_asin_f32(input.as_ptr(), output.as_mut_ptr(), input.len())
        }, OK);
        assert_eq!(output.map(f32::to_bits), [0xbd13_7c6b, 0xbd22_2bdc, 0xbd28_9c33]);
    }

    #[test]
    fn scalar_rotation_refuses_nonempty_null_buffers() {
        assert_eq!(unsafe {
            gpuwm_static_scalar_asin_f32(std::ptr::null(), std::ptr::null_mut(), 1)
        }, ERR);
    }
}
