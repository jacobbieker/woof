// The sin/cos/atan/asin grading argument stream, shared verbatim by the host
// (grade_trig.cpp) and device (grade_trig.py) legs: argument i of function
// f (0 sin, 1 cos, 2 atan, 3 asin) is a pure function of (f, i).
#pragma once
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
typedef long long int64_t;
#endif
__device__ __forceinline__ uint64_t trig_mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
__device__ __forceinline__ uint64_t trig_span(uint64_t r, uint64_t lo, uint64_t hi) { return lo + r % (hi - lo); }
// The branch points of glibc's sin/cos (s_sin.c) and atan (s_atan.c, atnat.h).
__device__ const uint64_t trig_edges[16] = {
    0x3e40000000000000ULL, 0x3e50000000000000ULL, 0x3fc020c49ba5e354ULL, 0x3feb600000000000ULL,
    0x400368fd00000000ULL, 0x419921fb00000000ULL, 0x3ff921fb54442d18ULL, 0x400921fb54442d18ULL,
    0x3e4bb67a00000000ULL, 0x3fb0000000000000ULL, 0x3ff0000000000000ULL, 0x4030000000000000ULL,
    0x43349ff200000000ULL, 0x4012d97c7f3321d2ULL, 0x401921fb54442d18ULL, 0x7fefffffffffffffULL,
};
// The branch points of glibc's asin (e_asin.c: the polynomial, table and
// square-root ranges and the table's sub-ranges) and two arguments of note.
__device__ const uint64_t asin_edges[16] = {
    0x3e50000000000000ULL, 0x3fc0000000000000ULL, 0x3fd0000000000000ULL, 0x3fe0000000000000ULL,
    0x3fe8000000000000ULL, 0x3fed800000000000ULL, 0x3fee800000000000ULL, 0x3fef000000000000ULL,
    0x3ff0000000000000ULL, 0x3fc8000000000000ULL, 0x3fd8000000000000ULL, 0x3fe4000000000000ULL,
    0x3fea000000000000ULL, 0x3fee000000000000ULL, 0x3fe6a09e667f3bcdULL, 0x3fd9800000000000ULL,
};
__device__ __forceinline__ double asin_arg(uint64_t r, uint64_t s, uint64_t i) {
    uint64_t u;
    switch (i & 7) {
    case 0: u = r; break;                                                            // every binary64 word
    case 1: u = (r & 0x7fffffffffffffffULL) % 0x3ff0000000000001ULL | s; break;     // |x| <= 1, all binades
    case 2: u = trig_span(r, 0x3e50000000000000ULL, 0x3fc0000000000000ULL) | s; break;  // 2^-26 .. 1/8
    case 3: u = trig_span(r, 0x3fc0000000000000ULL, 0x3fef000000000000ULL) | s; break;  // the table: 1/8 .. 0.96875
    case 4: u = trig_span(r, 0x3fef000000000000ULL, 0x3ff0000000000001ULL) | s; break;  // the root: 0.96875 .. 1
    case 5: u = (asin_edges[(r >> 40) & 15] + (int64_t)(r & 0xffffff) - 0x800000) | s; break;  // edges +- 2^23 ulp
    case 6: u = trig_span(r, 0x0000000000000000ULL, 0x3fda000000000000ULL) | s; break;  // |x| < 0.40625 (declinations)
    default: u = (0x3ff0000000000000ULL - (r & 0xffffffffffULL)) | s; break;          // within 2^40 ulp below 1
    }
    return __longlong_as_double((long long)u);
}
__device__ __forceinline__ double trig_arg(int f, uint64_t i) {
    uint64_t r = trig_mix(i ^ ((uint64_t)f << 60)), s = trig_mix(r) & 0x8000000000000000ULL, u;
    if (f == 3) return asin_arg(r, s, i);
    switch (i & 7) {
    case 0: u = r; break;                                                            // every binary64 word
    case 1: u = (r & 0x7fffffffffffffffULL) % 0x419921fb00000000ULL | s; break;     // |x| < 105414350, all binades
    case 2: u = trig_span(r, 0x3e40000000000000ULL, 0x401921fb54442d19ULL) | s; break;  // 2^-27 .. 2 pi
    case 3: u = trig_span(r, 0x3fb0000000000000ULL, 0x4030000000000000ULL) | s; break;  // 1/16 .. 16
    case 4: u = trig_span(r, 0x4030000000000000ULL, 0x43349ff200000000ULL) | s; break;  // 16 .. atan's E
    case 5: u = (trig_edges[(r >> 40) & 15] + (int64_t)(r & 0xffffff) - 0x800000) | s; break;  // edges +- 2^23 ulp
    case 6: {                                                                        // near k pi/2, |k| < 2^20
        const double k = (double)(int64_t)((r >> 12) & 0xfffff);
        const double x = __dadd_rn(__dmul_rn(k, 1.5707963267948966),
                                   __dmul_rn((double)((int64_t)(r & 0xfff) - 0x800), 1.0e-15));
        u = (uint64_t)__double_as_longlong(x) | s;
        break;
    }
    default: u = trig_span(r, 0x419921fb00000000ULL, 0x7ff0000000000000ULL) | s; break;  // branred: huge |x|
    }
    return __longlong_as_double((long long)u);
}
