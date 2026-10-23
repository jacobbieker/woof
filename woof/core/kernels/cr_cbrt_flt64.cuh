// ======================================================================
// THIRD-PARTY NOTICE. Original CUDA adaptation is distributed under
// Apache-2.0. The MIT grant below covers the transcribed upstream part.
// CORE-MATH binary64 cbrt (src/binary64/cbrt/cbrt.c, revision
// 8ae55cdd0d953dc1d6f78659435283d3436865ab, sha256
// 0a86b7a8f433dfa4610023767102ea19eb3642ea6982871c61c956148837dde4,
// kept verbatim at tools/uwpbl_wrf471_oracle/libm64/upstream/cbrt.c).
// uw_cbrt is CORE-MATH's correctly rounded binary64 cube root, rounding to
// nearest, ties to even; it is NOT glibc's cbrt (LGPL, not transcribed).
// PROVENANCE AND PROOF: tools/uwpbl_wrf471_oracle/libm64/README.md,
// "cbrt" (sweep_cbrt.py): equal to the upstream C and to an exact integer
// reference on 2^28 GPU arguments.  Host twin: woof/core/cr_cbrt.py.
// ======================================================================
// Correctly-rounded cubic root of binary64 value.
//
// Copyright (c) 2021-2022 Alexei Sibidanov.
//
// This file is part of the CORE-MATH project
// (https://core-math.gitlabpages.inria.fr/).
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// ADAPTATION (the only changes from upstream): rounding to nearest only, so
// get_rounding_mode() is the constant 0 and the directed-rounding wlist and
// set_flags() (floating-point exception flags) are dropped; every arithmetic
// operation is an explicit round-to-nearest intrinsic (__dadd_rn, __dmul_rn,
// __ddiv_rn, __fma_rn for upstream's __builtin_fma), so no compiler option
// can contract or reassociate it; __builtin_clzll is __clzll; the static
// tables are __device__ constants.
#pragma once
#if !defined(__CUDACC_RTC__)
#include <stdint.h>
#else
typedef unsigned long long uint64_t;
typedef long long int64_t;
#endif

namespace g64_cbrt {

__device__ __forceinline__ double asdouble(uint64_t u)
{
    return __longlong_as_double((long long)u);
}

__device__ __forceinline__ uint64_t asuint64(double f)
{
    return (uint64_t)__double_as_longlong(f);
}

__device__ __forceinline__ double cr_cbrt(double x)
{
    // escale: 1, 2^(1/3), 2^(2/3)
    const double escale0 = 1.0;
    const double escale1 = 0x1.428a2f98d728bp+0;
    const double escale2 = 0x1.965fea53d6e3dp+0;
    // c0+c1*x+c2*x^2+c3*x^3 approximates x^(1/3) on [1,2], error < 9.2e-5
    const double c0c = 0x1.1b0babccfef9cp-1, c1c = 0x1.2c9a3e94d1da5p-1;
    const double c2c = -0x1.4dc30b1a1ddbap-3, c3c = 0x1.7a8d3e4ec9b07p-6;
    const double u0 = 0x1.5555555555555p-2, u1 = 0x1.c71c71c71c71cp-3;
    // rsc[] = {1, -1, 0.5, -0.5, 0.25, -0.25}; off[0] = 0x1p-53 (rm = 0)
    const double off0 = 0x1p-53;
    uint64_t hx = asuint64(x), mant = hx & ((~(uint64_t)0) >> 12),
             sign = hx >> 63;
    unsigned e = (hx >> 52) & 0x7ff;
    if (((e + 1) & 0x7ff) < 2) {
        uint64_t ix = hx & ((~(uint64_t)0) >> 1);
        if (e == 0x7ff || ix == 0) return __dadd_rn(x, x); /* 0, inf, nan */
        int nz = __clzll((long long)ix) - 11;  /* subnormal */
        mant <<= nz;
        mant &= (~((uint64_t)0)) >> 12;
        e -= nz - 1;
    }
    e += 3072;
    uint64_t cvt1u = mant | ((uint64_t)0x3ff << 52), cvt5u = cvt1u;
    unsigned et = e / 3, it = e % 3;
    /* 2^(3k+it) <= x < 2^(3k+it+1), with 0 <= it <= 3 */
    cvt5u += (uint64_t)((int64_t)it << 52);
    cvt5u |= sign << 63;
    double zz = asdouble(cvt5u);
    /* cbrt(x) = cbrt(zz)*2^(et-1365) where 1 <= |zz| < 8 */
    double esc = it == 0 ? escale0 : (it == 1 ? escale1 : escale2);
    uint64_t isc = asuint64(esc);
    isc |= sign << 63;
    double cvt2f = asdouble(isc);
    double z = asdouble(cvt1u);
    /* cbrt(zz) = cbrt(z)*isc, where 1 <= z < 2 */
    const unsigned rsc_index = it << 1 | (unsigned)sign;
    const double rsc = (rsc_index & 1 ? -1.0 : 1.0)
                       * (rsc_index >> 1 == 0 ? 1.0
                          : (rsc_index >> 1 == 1 ? 0.5 : 0.25));
    double r = __ddiv_rn(1.0, z), rr = __dmul_rn(r, rsc), z2 = __dmul_rn(z, z);
    double c0 = __dadd_rn(c0c, __dmul_rn(z, c1c));
    double c2 = __dadd_rn(c2c, __dmul_rn(z, c3c));
    double y = __dadd_rn(c0, __dmul_rn(z2, c2)), y2 = __dmul_rn(y, y);
    /* y is an approximation of z^(1/3) */
    double h = __dsub_rn(__dmul_rn(y2, __dmul_rn(y, r)), 1.0);
    /* h determines the error between y and z^(1/3) */
    y = __dsub_rn(y, __dmul_rn(__dmul_rn(h, y),
                               __dsub_rn(u0, __dmul_rn(u1, h))));
    /* a cubic variant of Newton's method on f(y) = 1-z/y^3 */
    y = __dmul_rn(y, cvt2f);
    /* y approximates zz^(1/3) and rr 1/zz: one more linear Newton step */
    y2 = __dmul_rn(y, y);
    double y2l = __fma_rn(y, y, -y2);
    /* y2 + y2l = y^2 exactly */
    double y3 = __dmul_rn(y2, y);
    double y3l = __dadd_rn(__fma_rn(y, y2, -y3), __dmul_rn(y, y2l));
    /* y3 + y3l approximates y^3 with about 106 bits of accuracy */
    h = __dmul_rn(__dadd_rn(__dsub_rn(y3, zz), y3l), rr);
    double dy = __dmul_rn(h, __dmul_rn(y, u0));
    /* the approximation of zz^(1/3) is y - dy */
    double y1 = __dsub_rn(y, dy);
    dy = __dsub_rn(__dsub_rn(y, y1), dy);
    /* now y1 + dy, where |dy| < 1/2 ulp(y) (rounding to nearest) */
    double ady = fabs(dy);
    /* ady0 is tiny when dy is near 1/2 ulp(1), or from 3/2 ulp(1) */
    double ady0 = fabs(__dsub_rn(ady, off0));
    double ady1 = fabs(__dsub_rn(ady, __dadd_rn(0x1p-52, off0)));
    if (ady0 < 0x1p-75 || ady1 < 0x1p-75) {
        y2 = __dmul_rn(y1, y1);
        y2l = __fma_rn(y1, y1, -y2);
        y3 = __dmul_rn(y2, y1);
        y3l = __dadd_rn(__fma_rn(y1, y2, -y3), __dmul_rn(y1, y2l));
        h = __dmul_rn(__dadd_rn(__dsub_rn(y3, zz), y3l), rr);
        dy = __dmul_rn(h, __dmul_rn(y1, u0));
        y = __dsub_rn(y1, dy);
        dy = __dsub_rn(__dsub_rn(y1, y), dy);
        y1 = y;
        ady = fabs(dy);
        ady0 = fabs(__dsub_rn(ady, off0));
        ady1 = fabs(__dsub_rn(ady, __dadd_rn(0x1p-52, off0)));
        if (ady0 < 0x1p-98 || ady1 < 0x1p-98) {
            double azz = fabs(zz);
            if (azz == 0x1.9b78223aa307cp+1)
                y1 = copysign(0x1.79d15d0e8d59cp+0, zz);
            if (azz == 0x1.a202bfc89ddffp+2)
                y1 = copysign(0x1.de87aa837820fp+0, zz);
            /* the wlist corrections apply to directed roundings only */
        }
    }
    uint64_t cvt3u = asuint64(y1);
    cvt3u += (uint64_t)(et - 342 - 1023) << 52;
    int64_t m0 = (int64_t)(cvt3u << 30), m1 = m0 >> 63;
    if ((uint64_t)(m0 ^ m1) <= (1ull << 30)) {
        uint64_t cvt4u = asuint64(y1);
        cvt4u = (cvt4u + (1ull << 15)) & 0xffffffffffff0000ull;
        if (fabs(__dsub_rn(__dsub_rn(asdouble(cvt4u), y1), dy)) < 0x1p-60
            || fabs(zz) == 1.0) {
            cvt3u = (cvt3u + (1ull << 15)) & 0xffffffffffff0000ull;
        }
    }
    return asdouble(cvt3u);
}

}  // namespace g64_cbrt

__device__ __forceinline__ double uw_cbrt(double x)
{
    return g64_cbrt::cr_cbrt(x);
}
