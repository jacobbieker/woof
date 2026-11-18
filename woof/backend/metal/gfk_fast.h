// gfk_fast.h -- binary32 shortcuts to glibc_flt32.cuh's gfk_log, gfk_exp
// and gfk_pow on Metal, giving the binary64-evaluated words of those routines.
//
// glibc's logf/expf/powf (woof/core/kernels/glibc_flt32.cuh) evaluate in
// binary64, and Apple GPUs have none: on Metal every DADD/DMUL there runs
// f64_soft.h's integer soft-float, so one call costs hundreds of integer
// operations (sfclay's option-1 secant loop measured 12 ms for 4,565 points
// on an M3 Max).  These functions run the SAME formula -- the same table
// entries, the same polynomial in the same order -- in double-float
// (unevaluated sums of two binary32 values, about 2^-46), and accept the
// result only where its binary32 rounding cannot depend on the difference:
//
//   the binary64 routine computes some y_d; the double-float evaluation
//   computes Y with |Y - y_d| <= E (E is a bound stated per routine below,
//   with a factor of at least 16 to spare over the measured error); if no
//   binary32 rounding boundary (the midpoint of two adjacent binary32
//   values) lies within E of Y, RN32(y_d) == RN32(Y), and that is returned.
//
// Otherwise -- a boundary too close, a special operand (zero, subnormal,
// negative, infinite, NaN), a result that would leave the normal range, or
// an expf/powf reduction whose integer part sits on a tie -- the function
// returns WOOF_GFK_DEFER and the caller runs the binary64 routine itself
// (woof.backend.translate renames it gfk_*_soft and calls it).  So every
// returned word is the binary64 routine's word, in fast and strict numerics
// alike.  tools/metal_gfk_fast_grade.py checks the claim against the
// soft-float routine on the GPU: all 2^32 inputs of logf and expf, and
// powf over every base for a set of exponents plus random pairs.
//
// The prelude compiles without contraction, and LLVM records contraction
// per instruction, so the error-free transformations below (two_sum,
// fma-based two_prod) keep their single roundings even when inlined into a
// unit built with --fmad=true.  Subnormal flushing does not reach them: the
// operands are normal by the guards, and no intermediate that matters to E
// falls below 2^-126.
#pragma once

#define WOOF_GFK_DEFER 0xFFFFFFFFu

// BEGIN tools/metal_gfk_fast_constants.py
// generated: do not edit by hand
constant float WGFK_LOGF_INVC_HI[16] = {
    0x1.661ec80000000p+0f, 0x1.571ed40000000p+0f, 0x1.4953a00000000p+0f, 0x1.3c995c0000000p+0f,
    0x1.30d1900000000p+0f, 0x1.25e2280000000p+0f, 0x1.1bb4a40000000p+0f, 0x1.1235900000000p+0f,
    0x1.0953f40000000p+0f, 0x1.0000000000000p+0f, 0x1.e608d00000000p-1f, 0x1.ca4b320000000p-1f,
    0x1.b203660000000p-1f, 0x1.9c2d160000000p-1f, 0x1.886e600000000p-1f, 0x1.767dd00000000p-1f,
};
constant float WGFK_LOGF_INVC_LO[16] = {
    -0x1.81c3100000000p-26f, 0x1.55f1080000000p-25f, -0x1.e1fdea0000000p-25f, -0x1.e8ff900000000p-25f,
    0x1.910c940000000p-25f, -0x1.3d1c580000000p-26f, 0x1.4346880000000p-25f, -0x1.eea3480000000p-25f,
    0x1.9900a80000000p-28f, 0.0f, -0x1.32dc2a0000000p-28f, -0x1.fb2ac00000000p-30f,
    -0x1.12a0640000000p-26f, 0x1.d0d5160000000p-28f, 0x1.bc20f60000000p-28f, -0x1.5596f40000000p-26f,
};
constant float WGFK_LOGF_LOGC_HI[16] = {
    -0x1.57bf780000000p-2f, -0x1.2bef0a0000000p-2f, -0x1.01eae80000000p-2f, -0x1.b31d8a0000000p-3f,
    -0x1.6574f00000000p-3f, -0x1.1aa2bc0000000p-3f, -0x1.a4e76c0000000p-4f, -0x1.1973c60000000p-4f,
    -0x1.252f440000000p-5f, 0.0f, 0x1.aa5aa60000000p-5f, 0x1.c5e53a0000000p-4f,
    0x1.526e580000000p-3f, 0x1.bc28600000000p-3f, 0x1.1058bc0000000p-2f, 0x1.4043060000000p-2f,
};
constant float WGFK_LOGF_LOGC_LO[16] = {
    -0x1.1955bc0000000p-31f, -0x1.f01b760000000p-28f, 0x1.5d8b320000000p-31f, -0x1.a0893a0000000p-29f,
    -0x1.580eec0000000p-28f, -0x1.e720400000000p-29f, -0x1.d181cc0000000p-29f, 0x1.67b8cc0000000p-30f,
    0x1.c7bcf80000000p-31f, 0.0f, -0x1.06d33e0000000p-32f, 0x1.46c5d60000000p-29f,
    -0x1.1be4a00000000p-28f, 0x1.a448ee0000000p-28f, 0x1.140fdc0000000p-27f, -0x1.09223e0000000p-27f,
};
constant float WGFK_POWF_INVC_HI[16] = {
    0x1.661ec80000000p+0f, 0x1.571ed40000000p+0f, 0x1.4953a00000000p+0f, 0x1.3c995c0000000p+0f,
    0x1.30d1900000000p+0f, 0x1.25e2280000000p+0f, 0x1.1bb4a40000000p+0f, 0x1.1235900000000p+0f,
    0x1.0953f40000000p+0f, 0x1.0000000000000p+0f, 0x1.e608d00000000p-1f, 0x1.ca4b320000000p-1f,
    0x1.b203660000000p-1f, 0x1.9c2d160000000p-1f, 0x1.886e600000000p-1f, 0x1.767dd00000000p-1f,
};
constant float WGFK_POWF_INVC_LO[16] = {
    -0x1.81c3100000000p-26f, 0x1.55f1080000000p-25f, -0x1.e1fdea0000000p-25f, -0x1.e8ff900000000p-25f,
    0x1.910c940000000p-25f, -0x1.3d1c580000000p-26f, 0x1.4346880000000p-25f, -0x1.eea3480000000p-25f,
    0x1.9900a80000000p-28f, 0.0f, -0x1.32dc2a0000000p-28f, -0x1.fb2ac00000000p-30f,
    -0x1.12a0640000000p-26f, 0x1.d0d5160000000p-28f, 0x1.bc20f60000000p-28f, -0x1.5596f40000000p-26f,
};
constant float WGFK_POWF_LOGC_HI[16] = {
    -0x1.efec660000000p-2f, -0x1.b0b6840000000p-2f, -0x1.7418b00000000p-2f, -0x1.39de920000000p-2f,
    -0x1.01d9c00000000p-2f, -0x1.97c1d20000000p-3f, -0x1.2f9e3a0000000p-3f, -0x1.960cbc0000000p-4f,
    -0x1.a6f9dc0000000p-5f, 0.0f, 0x1.338caa0000000p-4f, 0x1.476a960000000p-3f,
    0x1.e840b40000000p-3f, 0x1.4064600000000p-2f, 0x1.88e9c20000000p-2f, 0x1.ce0a440000000p-2f,
};
constant float WGFK_POWF_LOGC_LO[16] = {
    0x1.1a73fa0000000p-28f, 0x1.a5606c0000000p-27f, -0x1.43f6f00000000p-27f, 0x1.648c220000000p-28f,
    0x1.81a93a0000000p-27f, 0x1.3121440000000p-29f, 0x1.8a186c0000000p-28f, 0x1.0ee5480000000p-33f,
    0x1.3714060000000p-30f, 0.0f, -0x1.b615860000000p-33f, -0x1.78edc80000000p-28f,
    0x1.589c9a0000000p-28f, -0x1.e7335c0000000p-27f, 0x1.8374000000000p-27f, 0x1.d62f7a0000000p-27f,
};
constant float WGFK_EXP2_T_HI[32] = {
    0x1.0000000000000p+0f, 0x1.059b0e0000000p+0f, 0x1.0b55860000000p+0f, 0x1.11301e0000000p+0f,
    0x1.172b840000000p+0f, 0x1.1d48740000000p+0f, 0x1.2387a60000000p+0f, 0x1.29e9e00000000p+0f,
    0x1.306fe00000000p+0f, 0x1.371a740000000p+0f, 0x1.3dea640000000p+0f, 0x1.44e0860000000p+0f,
    0x1.4bfdae0000000p+0f, 0x1.5342b60000000p+0f, 0x1.5ab07e0000000p+0f, 0x1.6247ec0000000p+0f,
    0x1.6a09e60000000p+0f, 0x1.71f75e0000000p+0f, 0x1.7a11480000000p+0f, 0x1.82589a0000000p+0f,
    0x1.8ace540000000p+0f, 0x1.93737c0000000p+0f, 0x1.9c49180000000p+0f, 0x1.a5503c0000000p+0f,
    0x1.ae89fa0000000p+0f, 0x1.b7f7700000000p+0f, 0x1.c199be0000000p+0f, 0x1.cb720e0000000p+0f,
    0x1.d5818e0000000p+0f, 0x1.dfc9740000000p+0f, 0x1.ea4afa0000000p+0f, 0x1.f507660000000p+0f,
};
constant float WGFK_EXP2_T_LO[32] = {
    0.0f, -0x1.9d4f520000000p-25f, 0x1.9f31220000000p-25f, -0x1.fdb4960000000p-25f,
    -0x1.c157420000000p-27f, -0x1.d2e8ca0000000p-25f, 0x1.ceac480000000p-25f, -0x1.5c04240000000p-25f,
    0x1.4636e20000000p-25f, -0x1.18aac60000000p-25f, 0x1.8246840000000p-25f, 0x1.8624b40000000p-30f,
    -0x1.593abc0000000p-25f, -0x1.2c56100000000p-25f, -0x1.5bd5ec0000000p-27f, -0x1.f8b5500000000p-25f,
    0x1.9fcef40000000p-26f, 0x1.1d8bee0000000p-25f, -0x1.829fd00000000p-25f, -0x1.accc7c0000000p-26f,
    0x1.15506e0000000p-27f, -0x1.e647440000000p-25f, 0x1.51f8480000000p-27f, -0x1.b83b540000000p-25f,
    -0x1.a94b140000000p-26f, -0x1.a094380000000p-25f, -0x1.3d56b20000000p-27f, -0x1.8837cc0000000p-27f,
    -0x1.822dbc0000000p-27f, -0x1.908c940000000p-25f, 0x1.52486c0000000p-27f, -0x1.246eb00000000p-26f,
};
#define WGFK_LOGF_LN2_HI 0x1.62e4300000000p-1f
#define WGFK_LOGF_LN2_LO -0x1.05c6100000000p-29f
#define WGFK_LOGF_A0_HI -0x1.00ea340000000p-2f
#define WGFK_LOGF_A0_LO -0x1.1710660000000p-27f
#define WGFK_LOGF_A1_HI 0x1.5575b00000000p-2f
#define WGFK_LOGF_A1_LO 0x1.7c016e0000000p-27f
#define WGFK_LOGF_A2_HI -0x1.fffff00000000p-2f
#define WGFK_LOGF_A2_LO 0x1.beb7dc0000000p-27f
#define WGFK_EXPF_P0S_HI 0x1.c6af840000000p-20f
#define WGFK_EXPF_P0S_LO 0x1.7224720000000p-45f
#define WGFK_EXPF_P1S_HI 0x1.ebfce60000000p-13f
#define WGFK_EXPF_P1S_LO -0x1.e0a7620000000p-38f
#define WGFK_EXPF_P2S_HI 0x1.62e4300000000p-6f
#define WGFK_EXPF_P2S_LO -0x1.e75a540000000p-35f
#define WGFK_EXP2F_P0_HI 0x1.c6af840000000p-5f
#define WGFK_EXP2F_P0_LO 0x1.7224720000000p-30f
#define WGFK_EXP2F_P1_HI 0x1.ebfce60000000p-3f
#define WGFK_EXP2F_P1_LO -0x1.e0a7620000000p-28f
#define WGFK_EXP2F_P2_HI 0x1.62e4300000000p-1f
#define WGFK_EXP2F_P2_LO -0x1.e75a540000000p-30f
#define WGFK_POWF_A0_HI 0x1.27616c0000000p-2f
#define WGFK_POWF_A0_LO 0x1.292dc20000000p-27f
#define WGFK_POWF_A1_HI -0x1.71969a0000000p-2f
#define WGFK_POWF_A1_LO -0x1.d719e80000000p-32f
#define WGFK_POWF_A2_HI 0x1.ec70a60000000p-2f
#define WGFK_POWF_A2_LO 0x1.94f75c0000000p-27f
#define WGFK_POWF_A3_HI -0x1.7154740000000p-1f
#define WGFK_POWF_A3_LO -0x1.17ded80000000p-26f
#define WGFK_POWF_A4_HI 0x1.7154760000000p+0f
#define WGFK_POWF_A4_LO 0x1.4aae0a0000000p-26f
#define WGFK_EXPF_INVLN2N_HI 0x1.7154760000000p+5f
#define WGFK_EXPF_INVLN2N_LO 0x1.4ae0c00000000p-21f
#define WGFK_EXPF_INVLN2N_LO2 -0x1.0000000000000p-46f
// END tools/metal_gfk_fast_constants.py

struct __woof_ff { float hi; float lo; };

static inline __woof_ff __woof_ff_two_sum(float a, float b) {
  float s = a + b; float bb = s - a;
  return {s, (a - (s - bb)) + (b - bb)};
}
static inline __woof_ff __woof_ff_fast(float a, float b) {   // |a| >= |b| or a == 0
  float s = a + b;
  return {s, b - (s - a)};
}
static inline __woof_ff __woof_ff_add(__woof_ff x, __woof_ff y) {
  __woof_ff s = __woof_ff_two_sum(x.hi, y.hi);
  return __woof_ff_fast(s.hi, s.lo + (x.lo + y.lo));
}
static inline __woof_ff __woof_ff_mul(__woof_ff x, __woof_ff y) {
  float p = x.hi * y.hi;
  float e = metal::fma(x.hi, y.hi, -p);
  e = metal::fma(x.hi, y.lo, e);
  e = metal::fma(x.lo, y.hi, e);
  return __woof_ff_fast(p, e);
}
static inline __woof_ff __woof_ff_mulf(__woof_ff x, float f) {
  float p = x.hi * f;
  float e = metal::fma(x.hi, f, -p);
  e = metal::fma(x.lo, f, e);
  return __woof_ff_fast(p, e);
}
#define WOOF_FF(name) (__woof_ff{name##_HI, name##_LO})
#define WOOF_FFT(name, i) (__woof_ff{name##_HI[i], name##_LO[i]})

// RN32 of any value within E of v.hi + v.lo, or WOOF_GFK_DEFER: the bits of
// res = RN32(hi + lo) when |(hi + lo) - res| + E stays below half an ulp of
// res (a quarter at a power of two, where the ulp below is half), and res
// is normal with room for the ulp arithmetic.
static inline uint __woof_ff_round(__woof_ff v, float E) {
  float res = v.hi + v.lo;
  float d = (v.hi - res) + v.lo;
  uint ur = as_type<uint>(res);
  uint ex = ur & 0x7f800000u;
  if (ex <= (25u << 23) || ex == 0x7f800000u) return WOOF_GFK_DEFER;
  float hu = as_type<float>(ex - (24u << 23));   // half an ulp
  if ((ur & 0x007fffffu) == 0u) hu *= 0.5f;
  // d carries one rounding (2^-24 relative); the 2^-20 term covers it.
  if (metal::fabs(d) + E + hu * 0x1p-20f >= hu) return WOOF_GFK_DEFER;
  return ur;
}

// gfk_log: glibc 2.39 e_logf.c.  Each term of y0 + r + poly is evaluated to
// about 2^-46 of its size and the binary64 routine to 2^-52, so
// |Y - y_d| < 2^-42 (|y0| + |r|); E = 2^-36 (|y0| + |r|).
static inline uint __woof_gfk_log(float x) {
  uint ix = as_type<uint>(x);
  if (ix == 0x3f800000u) return 0u;
  if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u) return WOOF_GFK_DEFER;
  uint tmp = ix - 0x3f330000u;
  int i = int((tmp >> 19) & 15u);
  int k = int(tmp) >> 23;
  float z = as_type<float>(ix - (tmp & 0xff800000u));
  // r = z * invc - 1: z * invc_hi exactly, then z * invc_lo; p is within
  // 2^-5 of 1, so p - 1 is exact
  float p = z * WGFK_LOGF_INVC_HI[i];
  float e = metal::fma(z, WGFK_LOGF_INVC_HI[i], -p);
  e = metal::fma(z, WGFK_LOGF_INVC_LO[i], e);
  __woof_ff r = __woof_ff_two_sum(p - 1.0f, e);
  float kf = float(k);
  __woof_ff y0 = __woof_ff_add(WOOF_FFT(WGFK_LOGF_LOGC, i), __woof_ff_mulf(WOOF_FF(WGFK_LOGF_LN2), kf));
  __woof_ff r2 = __woof_ff_mul(r, r);
  __woof_ff y = __woof_ff_add(__woof_ff_mul(WOOF_FF(WGFK_LOGF_A1), r), WOOF_FF(WGFK_LOGF_A2));
  y = __woof_ff_add(__woof_ff_mul(WOOF_FF(WGFK_LOGF_A0), r2), y);
  y = __woof_ff_add(__woof_ff_mul(y, r2), __woof_ff_add(y0, r));
  return __woof_ff_round(y, (metal::fabs(y0.hi) + metal::fabs(r.hi)) * 0x1p-36f);
}

// The 2^(N/32) * (1 + poly(r)) tail shared by gfk_exp and gfk_pow: V = y *
// T[N & 31] lies in [0.98, 2.02] and is rounded there; the binary32 result
// is that word scaled by 2^(N >> 5), exact while it stays normal.
static inline uint __woof_gfk_exp2_tail(__woof_ff y, int n, float E) {
  int j = n & 31, q = n >> 5;
  if (q < -125 || q > 126) return WOOF_GFK_DEFER;
  __woof_ff v = __woof_ff_mul(y, WOOF_FFT(WGFK_EXP2_T, j));
  uint b = __woof_ff_round(v, E * 2.03f);
  if (b == WOOF_GFK_DEFER) return b;
  return b + (uint(q) << 23);
}

// gfk_exp: glibc 2.39 e_expf.c, FMA build (the reduction's product fused).
// P = x * 32/ln2 to 2^-46 |P| (the constant in three binary32 parts, so the
// products carry it exactly); N = RNE(P) unless P is within 2^-30 of a tie;
// r = P - N to 2^-46 |P| + 2^-48 against the binary64 r's 2^-53.  The
// result's relative error is then below 2^-41 + 2^-51 |P|;
// E = 2^-37 + 2^-46 |P| (relative).
static inline uint __woof_gfk_exp(float x) {
  uint ux = as_type<uint>(x);
  if (((ux >> 20) & 0x7ffu) >= (as_type<uint>(88.0f) >> 20)) return WOOF_GFK_DEFER;
  if ((ux & 0x7f800000u) == 0u) return WOOF_GFK_DEFER;
  float p = x * WGFK_EXPF_INVLN2N_HI;
  float pe = metal::fma(x, WGFK_EXPF_INVLN2N_HI, -p);
  pe = pe + metal::fma(x, WGFK_EXPF_INVLN2N_LO, x * WGFK_EXPF_INVLN2N_LO2);
  float nf = metal::rint(p);
  __woof_ff r = __woof_ff_two_sum(p - nf, pe);       // p - nf is exact
  if (r.hi > 0.5f) { nf += 1.0f; r = __woof_ff_two_sum(r.hi - 1.0f, r.lo); }
  else if (r.hi < -0.5f) { nf -= 1.0f; r = __woof_ff_two_sum(r.hi + 1.0f, r.lo); }
  if (metal::fabs(metal::fabs(r.hi) - 0.5f) < 0x1p-30f) return WOOF_GFK_DEFER;
  __woof_ff z = __woof_ff_add(__woof_ff_mul(WOOF_FF(WGFK_EXPF_P0S), r), WOOF_FF(WGFK_EXPF_P1S));
  __woof_ff r2 = __woof_ff_mul(r, r);
  __woof_ff y = __woof_ff_add(__woof_ff_mul(WOOF_FF(WGFK_EXPF_P2S), r), __woof_ff{1.0f, 0.0f});
  y = __woof_ff_add(__woof_ff_mul(z, r2), y);
  return __woof_gfk_exp2_tail(y, int(nf), 0x1p-37f + metal::fabs(p) * 0x1p-46f);
}

// gfk_pow: glibc 2.39 e_powf.c for a positive normal base and a normal
// exponent with |y log2 x| < 120 (everything else defers).  log2 x as
// y0 + r + poly, each term to about 2^-46; y * log2 x to
// delta = 2^-40 |y| (|y0| + |r| + 2^-20) (a factor of 16 over the
// error); N = RNE(32 y log2 x) unless within 32 delta + 2^-30 of a tie;
// the result's relative error is below ln2 delta + 2^-41; E = delta + 2^-37.
static inline uint __woof_gfk_pow(float x, float y) {
  uint ix = as_type<uint>(x), iy = as_type<uint>(y);
  if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u) return WOOF_GFK_DEFER;
  uint ey = iy & 0x7f800000u;
  if (ey == 0u || ey == 0x7f800000u) return WOOF_GFK_DEFER;
  uint tmp = ix - 0x3f330000u;
  int i = int((tmp >> 19) & 15u);
  uint top = tmp & 0xff800000u;
  float z = as_type<float>(ix - top);
  int k = int(top) >> 23;
  float p = z * WGFK_POWF_INVC_HI[i];
  float e = metal::fma(z, WGFK_POWF_INVC_HI[i], -p);
  e = metal::fma(z, WGFK_POWF_INVC_LO[i], e);
  __woof_ff r = __woof_ff_two_sum(p - 1.0f, e);
  __woof_ff y0 = __woof_ff_add(WOOF_FFT(WGFK_POWF_LOGC, i), __woof_ff{float(k), 0.0f});
  __woof_ff r2 = __woof_ff_mul(r, r);
  __woof_ff yy = __woof_ff_add(__woof_ff_mul(WOOF_FF(WGFK_POWF_A0), r), WOOF_FF(WGFK_POWF_A1));
  __woof_ff pp = __woof_ff_add(__woof_ff_mul(WOOF_FF(WGFK_POWF_A2), r), WOOF_FF(WGFK_POWF_A3));
  __woof_ff r4 = __woof_ff_mul(r2, r2);
  __woof_ff qq = __woof_ff_add(__woof_ff_mul(WOOF_FF(WGFK_POWF_A4), r), y0);
  qq = __woof_ff_add(__woof_ff_mul(pp, r2), qq);
  __woof_ff logx = __woof_ff_add(__woof_ff_mul(yy, r4), qq);
  __woof_ff t = __woof_ff_mulf(logx, y);
  if (!(metal::fabs(t.hi) < 120.0f)) return WOOF_GFK_DEFER;
  float delta = metal::fabs(y) * (metal::fabs(y0.hi) + metal::fabs(r.hi) + 0x1p-20f) * 0x1p-40f;
  // N = RNE(32 t): the binary64 kd = t + 1.5 * 2^47 rounds t to 1/32
  float s32 = t.hi * 32.0f, l32 = t.lo * 32.0f;      // exact scalings
  float nf = metal::rint(s32);
  __woof_ff f = __woof_ff_two_sum(s32 - nf, l32);    // s32 - nf is exact
  if (f.hi > 0.5f) { nf += 1.0f; f = __woof_ff_two_sum(f.hi - 1.0f, f.lo); }
  else if (f.hi < -0.5f) { nf -= 1.0f; f = __woof_ff_two_sum(f.hi + 1.0f, f.lo); }
  if (metal::fabs(metal::fabs(f.hi) - 0.5f) < 32.0f * delta + 0x1p-30f) return WOOF_GFK_DEFER;
  __woof_ff rr = __woof_ff{f.hi * 0x1p-5f, f.lo * 0x1p-5f};   // t - N/32, exact scaling
  __woof_ff zz = __woof_ff_add(__woof_ff_mul(WOOF_FF(WGFK_EXP2F_P0), rr), WOOF_FF(WGFK_EXP2F_P1));
  __woof_ff rr2 = __woof_ff_mul(rr, rr);
  __woof_ff w = __woof_ff_add(__woof_ff_mul(WOOF_FF(WGFK_EXP2F_P2), rr), __woof_ff{1.0f, 0.0f});
  w = __woof_ff_add(__woof_ff_mul(zz, rr2), w);
  return __woof_gfk_exp2_tail(w, int(nf), delta + 0x1p-37f);
}

#if defined(WOOF_RETAIN_SUBNORMALS)
// --ftz=false units: float is f32s; the shortcuts read its binary32 word
// (a subnormal operand defers, so nothing here depends on gradual underflow)
static inline uint __woof_gfk_log(f32s x) { return __woof_gfk_log(x.v); }
static inline uint __woof_gfk_exp(f32s x) { return __woof_gfk_exp(x.v); }
static inline uint __woof_gfk_pow(f32s x, f32s y) { return __woof_gfk_pow(x.v, y.v); }
#endif
