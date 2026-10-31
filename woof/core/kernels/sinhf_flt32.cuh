// ======================================================================
// WOOF's own float32 sinhf (gfk_sinh): the correctly rounded binary32
// hyperbolic sine, round to nearest even.
//
// WHY.  WRF-Solar's FARMS (phys/module_ra_farms.F, WATERMODEL/ICEMODEL)
// evaluates SINH on REAL(4), which gfortran lowers to the C library's
// sinhf.  glibc returns the correctly rounded word from release 2.41 on
// (its sinhf is CORE-MATH's); CUDA's sinhf and Apple's are not that
// function.  This is not a transcription of anything: the value is formed
// in binary64 from IEEE basic operations only (no libm call, no fused
// multiply-add), with the few arguments where that binary64 value sits
// too close to a binary32 rounding boundary listed explicitly, so the
// device, the host build behind cuda_host_shim.h and the Metal translation
// return one word.  tools/farms_wrf471_oracle/sinhf_proof.cpp grades it
// over every binary32 argument against the oracle host's glibc sinhf.
//
// Include AFTER glibc_flt32.cuh (it uses no names from it, but units that
// name this header compile it in that order).
// ======================================================================
#pragma once

// exp(x) for 0.5 <= x < 90 in binary64, about 1 ulp: x = k ln2 + r with a
// Cody-Waite split of ln2 (ln2_hi carries 11 trailing zero bits, so k ln2_hi
// is exact for every k this range reaches), |r| <= ln2/2, exp(r) - 1 by its
// Taylor series to r^13, then the power of two by exponent arithmetic.
__device__ __forceinline__ double gfk_sinh_exp_pos(double x)
{
    const double inv_ln2 = 0x1.71547652b82fep+0;
    const double ln2_hi = 0x1.62e42fefa3800p-1;
    const double ln2_lo = 0x1.ef35793c76730p-45;
    const double shift = 0x1.8p52;
    double kd = __dadd_rn(__dmul_rn(x, inv_ln2), shift);
    long long ki = __double_as_longlong(kd) - __double_as_longlong(shift);
    kd = __dsub_rn(kd, shift);
    double r = __dsub_rn(__dsub_rn(x, __dmul_rn(kd, ln2_hi)),
                         __dmul_rn(kd, ln2_lo));
    double p = 0x1.6124613a86d09p-33;                 // 1/13!
    p = __dadd_rn(0x1.1eed8eff8d898p-29, __dmul_rn(r, p));  // 1/12!
    p = __dadd_rn(0x1.ae64567f544e4p-26, __dmul_rn(r, p));  // 1/11!
    p = __dadd_rn(0x1.27e4fb7789f5cp-22, __dmul_rn(r, p));  // 1/10!
    p = __dadd_rn(0x1.71de3a556c734p-19, __dmul_rn(r, p));  // 1/9!
    p = __dadd_rn(0x1.a01a01a01a01ap-16, __dmul_rn(r, p));  // 1/8!
    p = __dadd_rn(0x1.a01a01a01a01ap-13, __dmul_rn(r, p));  // 1/7!
    p = __dadd_rn(0x1.6c16c16c16c17p-10, __dmul_rn(r, p));  // 1/6!
    p = __dadd_rn(0x1.1111111111111p-7, __dmul_rn(r, p));   // 1/5!
    p = __dadd_rn(0x1.5555555555555p-5, __dmul_rn(r, p));   // 1/4!
    p = __dadd_rn(0x1.5555555555555p-3, __dmul_rn(r, p));   // 1/3!
    p = __dadd_rn(0x1.0p-1, __dmul_rn(r, p));               // 1/2!
    p = __dadd_rn(1.0, __dmul_rn(r, p));
    double em1 = __dmul_rn(r, p);                           // exp(r) - 1
    double e = __dadd_rn(1.0, em1);
    // e * 2^ki: ki is at most 130 and e in [0.7, 1.42], so the product is a
    // normal binary64 and the exponent field add is exact.
    return __longlong_as_double(__double_as_longlong(e) + (ki << 52));
}

__device__ float gfk_sinh(float xf)
{
    unsigned int ux = __float_as_uint(xf);
    unsigned int ax_bits = ux & 0x7fffffffu;
    unsigned int sign = ux & 0x80000000u;
    if (ax_bits >= 0x7f800000u) {
        return xf;                       // NaN (quiet as given) or +-Inf
    }
    if (ax_bits >= 0x42b40000u) {        // |x| >= 90: overflows binary32
        return __uint_as_float(sign | 0x7f800000u);
    }
    if (ax_bits == 0u) {
        return xf;                       // +-0
    }
    double ax = (double)__uint_as_float(ax_bits);
    double s;
    if (ax_bits < 0x3f000000u) {         // |x| < 0.5: odd Taylor series
        double x2 = __dmul_rn(ax, ax);
        double p = 0x1.2f49b46814157p-57;                         // 1/19!
        p = __dadd_rn(0x1.952c77030ad4ap-49, __dmul_rn(x2, p));   // 1/17!
        p = __dadd_rn(0x1.ae7f3e733b81fp-41, __dmul_rn(x2, p));   // 1/15!
        p = __dadd_rn(0x1.6124613a86d09p-33, __dmul_rn(x2, p));   // 1/13!
        p = __dadd_rn(0x1.ae64567f544e4p-26, __dmul_rn(x2, p));   // 1/11!
        p = __dadd_rn(0x1.71de3a556c734p-19, __dmul_rn(x2, p));   // 1/9!
        p = __dadd_rn(0x1.a01a01a01a01ap-13, __dmul_rn(x2, p));   // 1/7!
        p = __dadd_rn(0x1.1111111111111p-7, __dmul_rn(x2, p));    // 1/5!
        p = __dadd_rn(0x1.5555555555555p-3, __dmul_rn(x2, p));    // 1/3!
        s = __dadd_rn(ax, __dmul_rn(ax, __dmul_rn(x2, p)));
    } else {                             // 0.5 <= |x| < 90
        double e = gfk_sinh_exp_pos(ax);
        s = __dmul_rn(0.5, __dsub_rn(e, __ddiv_rn(1.0, e)));
    }
    float r = __double2float_rn(s);
    // The one argument magnitude where the binary64 value above rounds to
    // the neighbour of the correctly rounded binary32 word: sinh(x) for
    // x = 0x1.250bfep-11 is 0x1.250bfeffffffffc3...p-11, 2^-62 relative
    // below the rounding midpoint (sinhf_proof.cpp, every 2^32 argument:
    // tools/farms_wrf471_oracle/sinhf-proof.log).
    if (ax_bits == 0x3a1285ffu) r = __uint_as_float(0x3a1285ffu);
    return __uint_as_float(__float_as_uint(r) | sign);
}
