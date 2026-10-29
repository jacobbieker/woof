// Metal prelude for kernels translated from woof's CUDA sources.
//
// woof.backend.translate emits this text ahead of every translated unit.
// It gives CUDA's names the arithmetic woof.backend.metal.arith_receipt.json
// measured to match CUDA word for word on Apple GPUs:
//
// * binary32 + - * / and fma are IEEE round-to-nearest in math_mode safe;
//   sqrt is only correctly rounded as metal::precise::sqrt, so every CUDA
//   sqrt spelling goes there;
// * contraction is off unless the unit asked for --fmad=true (the
//   translator writes the pragma ahead of this prelude);
// * binary32 subnormals flush, as CUDA's -ftz=true route does.
//
// `double` is `f64`: binary64 words computed by f64_soft.h, one value type
// with free-function operators (member operators would need one overload
// per address space of `this`).  Conversions both ways are implicit, as
// C++'s are between double and the other arithmetic types; the ambiguity
// that would leave in `f64 op float` is settled by exact-match templates
// for every mixed operator, each evaluating in binary64 exactly as C++
// promotes.  Template bodies (whose conversions libclang cannot see) thus
// resolve as CUDA's do; outside templates the translator also spells every
// narrowing conversion as a cast.
#pragma once
#include <metal_stdlib>

// F64_SOFT_H_TEXT

// binary64 -> binary32 as CUDA converts: under -ftz=true (every unit that
// does not keep subnormals) a subnormal result flushes to a signed zero
// (cvt.rn.ftz.f32.f64).  Tested on the bits: a Metal compare would itself
// flush the subnormal it is testing.
static inline float __woof_cvt_f32(ulong b) {
  float r = f64_to_f32(b);
#ifndef WOOF_RETAIN_SUBNORMALS
  uint u = as_type<uint>(r);
  if ((u & 0x7F800000u) == 0u) u &= 0x80000000u;
  r = as_type<float>(u);
#endif
  return r;
}

struct f64 {
  ulong b;
  f64() = default;
  f64() threadgroup = default;  // threadgroup arrays of f64
  f64(float x) : b(f64_from_f32(x)) {}
  f64(int x) : b(f64_from_i32(x)) {}
  f64(uint x) : b(f64_from_u64((ulong)x)) {}
  f64(long x) : b(f64_from_i64(x)) {}
  f64(ulong x) : b(f64_from_u64(x)) {}
  f64(short x) : b(f64_from_i32(x)) {}
  f64(ushort x) : b(f64_from_i32(x)) {}
  f64(char x) : b(f64_from_i32(x)) {}
  f64(uchar x) : b(f64_from_i32(x)) {}
  f64(bool x) : b(x ? 0x3FF0000000000000UL : 0UL) {}
  static constexpr f64 bits(ulong u) { f64 r = f64(); r.b = u; return r; }
#define WOOF_F64_CONV(AS)                                                        \
  operator float() const AS { return __woof_cvt_f32(b); }                        \
  operator int() const AS { return f64_to_i32(b); }                              \
  operator uint() const AS { return (uint)f64_to_i64(b); }                       \
  operator long() const AS { return f64_to_i64(b); }                             \
  operator ulong() const AS { return (ulong)f64_to_i64(b); }                     \
  operator short() const AS { return (short)f64_to_i32(b); }                     \
  operator ushort() const AS { return (ushort)f64_to_i32(b); }                   \
  operator char() const AS { return (char)f64_to_i32(b); }                       \
  operator uchar() const AS { return (uchar)f64_to_i32(b); }                     \
  explicit operator bool() const AS { return (b << 1) != 0; }
  WOOF_F64_CONV(thread)
  WOOF_F64_CONV(device)
  WOOF_F64_CONV(constant)
  WOOF_F64_CONV(threadgroup)
#undef WOOF_F64_CONV
};

static constexpr f64 __woof_f64_lit(ulong u) { return f64::bits(u); }

static inline f64 operator+(f64 a, f64 b) { return f64::bits(f64_add(a.b, b.b)); }
static inline f64 operator-(f64 a, f64 b) { return f64::bits(f64_sub(a.b, b.b)); }
static inline f64 operator*(f64 a, f64 b) { return f64::bits(f64_mul(a.b, b.b)); }
static inline f64 operator/(f64 a, f64 b) { return f64::bits(f64_div(a.b, b.b)); }
static constexpr f64 operator-(f64 a) { return f64::bits(a.b ^ 0x8000000000000000UL); }
static constexpr f64 operator+(f64 a) { return a; }
static inline bool operator==(f64 a, f64 b) { return f64_eq(a.b, b.b); }
static inline bool operator!=(f64 a, f64 b) { return !f64_eq(a.b, b.b); }
static inline bool operator<(f64 a, f64 b) { return f64_lt(a.b, b.b); }
static inline bool operator<=(f64 a, f64 b) { return f64_le(a.b, b.b); }
static inline bool operator>(f64 a, f64 b) { return f64_lt(b.b, a.b); }
static inline bool operator>=(f64 a, f64 b) { return f64_le(b.b, a.b); }
static inline bool operator!(f64 a) { return (a.b << 1) == 0; }

// Mixed operands: an exact match for every arithmetic type, evaluated in
// binary64 as C++ promotes (`f64 op T` and `T op f64`).
#define WOOF_ARITH(T) typename T, typename = metal::enable_if_t<metal::is_arithmetic_v<T>>
#define WOOF_F64_MIXED(OP, R)                                                         \
  template <WOOF_ARITH(T)> static inline R operator OP(f64 a, T b) { return a OP f64(b); } \
  template <WOOF_ARITH(T)> static inline R operator OP(T a, f64 b) { return f64(a) OP b; }
WOOF_F64_MIXED(+, f64)
WOOF_F64_MIXED(-, f64)
WOOF_F64_MIXED(*, f64)
WOOF_F64_MIXED(/, f64)
WOOF_F64_MIXED(==, bool)
WOOF_F64_MIXED(!=, bool)
WOOF_F64_MIXED(<, bool)
WOOF_F64_MIXED(<=, bool)
WOOF_F64_MIXED(>, bool)
WOOF_F64_MIXED(>=, bool)
#undef WOOF_F64_MIXED

#define WOOF_F64_ASSIGN(AS)                                                              \
  static inline AS f64& operator+=(AS f64& a, f64 b) { a.b = f64_add(a.b, b.b); return a; } \
  static inline AS f64& operator-=(AS f64& a, f64 b) { a.b = f64_sub(a.b, b.b); return a; } \
  static inline AS f64& operator*=(AS f64& a, f64 b) { a.b = f64_mul(a.b, b.b); return a; } \
  static inline AS f64& operator/=(AS f64& a, f64 b) { a.b = f64_div(a.b, b.b); return a; } \
  static inline AS f64& operator++(AS f64& a) { a.b = f64_add(a.b, 0x3FF0000000000000UL); return a; } \
  static inline AS f64& operator--(AS f64& a) { a.b = f64_sub(a.b, 0x3FF0000000000000UL); return a; } \
  static inline f64 operator++(AS f64& a, int) { f64 o = a; a.b = f64_add(a.b, 0x3FF0000000000000UL); return o; } \
  static inline f64 operator--(AS f64& a, int) { f64 o = a; a.b = f64_sub(a.b, 0x3FF0000000000000UL); return o; }
WOOF_F64_ASSIGN(thread)
WOOF_F64_ASSIGN(device)
WOOF_F64_ASSIGN(threadgroup)
#undef WOOF_F64_ASSIGN

// `f64 op= T` (T converted up) and `T op= f64` (computed in binary64, then
// rounded to T once): C++'s compound-assignment semantics.
#define WOOF_MIXED_ASSIGN(AS, OP)                                                     \
  template <WOOF_ARITH(T)> static inline AS f64& operator OP##=(AS f64& a, T b) {    \
    a = a OP f64(b); return a; }                                                      \
  template <WOOF_ARITH(T)> static inline AS T& operator OP##=(AS T& a, f64 b) {      \
    a = T(f64(a) OP b); return a; }
#define WOOF_MIXED_ASSIGN_ALL(AS) \
  WOOF_MIXED_ASSIGN(AS, +) WOOF_MIXED_ASSIGN(AS, -) WOOF_MIXED_ASSIGN(AS, *) WOOF_MIXED_ASSIGN(AS, /)
WOOF_MIXED_ASSIGN_ALL(thread)
WOOF_MIXED_ASSIGN_ALL(device)
WOOF_MIXED_ASSIGN_ALL(threadgroup)
#undef WOOF_MIXED_ASSIGN_ALL
#undef WOOF_MIXED_ASSIGN

// ---------------------------------------------------------------- binary64 math
static inline f64 sqrt(f64 a) { return f64::bits(f64_sqrt(a.b)); }
static inline f64 fabs(f64 a) { return f64::bits(a.b & 0x7FFFFFFFFFFFFFFFUL); }
static inline f64 fma(f64 a, f64 b, f64 c) { return f64::bits(f64_fma(a.b, b.b, c.b)); }
static inline f64 copysign(f64 a, f64 b) {
  return f64::bits((a.b & 0x7FFFFFFFFFFFFFFFUL) | (b.b & 0x8000000000000000UL));
}
static inline bool isnan(f64 a) { return F64_IS_NAN(a.b); }
static inline bool isinf(f64 a) { return (a.b << 1) == 0xFFE0000000000000UL; }
static inline bool isfinite(f64 a) { return ((a.b >> 52) & 0x7FF) != 0x7FF; }
static inline bool signbit(f64 a) { return (a.b >> 63) != 0; }
// fmin/fmax: the non-NaN operand; -0 orders below +0 (PTX min/max.f64).
static inline f64 fmin(f64 a, f64 b) {
  if (isnan(a)) return b;
  if (isnan(b)) return a;
  if (f64_lt(a.b, b.b)) return a;
  if (f64_lt(b.b, a.b)) return b;
  return f64::bits(a.b | b.b);
}
static inline f64 fmax(f64 a, f64 b) {
  if (isnan(a)) return b;
  if (isnan(b)) return a;
  if (f64_lt(a.b, b.b)) return b;
  if (f64_lt(b.b, a.b)) return a;
  return f64::bits(a.b & b.b);
}
static inline f64 min(f64 a, f64 b) { return fmin(a, b); }
static inline f64 max(f64 a, f64 b) { return fmax(a, b); }
#define WOOF_F64_MIXED_FN(F)                                                          \
  template <WOOF_ARITH(T)> static inline f64 F(f64 a, T b) { return F(a, f64(b)); }   \
  template <WOOF_ARITH(T)> static inline f64 F(T a, f64 b) { return F(f64(a), b); }
WOOF_F64_MIXED_FN(fmin)
WOOF_F64_MIXED_FN(fmax)
WOOF_F64_MIXED_FN(min)
WOOF_F64_MIXED_FN(max)
WOOF_F64_MIXED_FN(copysign)
// Integral rounding by exponent masking: exact.
static inline f64 trunc(f64 a) {
  int e = F64_EXP(a.b) - 0x3FF;
  if (e >= 52) return a;
  if (e < 0) return f64::bits(a.b & 0x8000000000000000UL);
  return f64::bits(a.b & ~(0x000FFFFFFFFFFFFFUL >> e));
}
static inline f64 floor(f64 a) {
  f64 t = trunc(a);
  return (signbit(a) && f64_lt(a.b, t.b)) ? t - f64(1) : t;
}
static inline f64 ceil(f64 a) {
  f64 t = trunc(a);
  return (!signbit(a) && f64_lt(t.b, a.b)) ? t + f64(1) : t;
}
static inline f64 rint(f64 a) {  // round half to even, via the 2^52 shift
  if (!isfinite(a) || F64_EXP(a.b) >= 0x3FF + 52) return a;
  f64 big = f64::bits(0x4330000000000000UL | (a.b & 0x8000000000000000UL));
  f64 r = (a + big) - big;
  return copysign(r, a);
}
static inline f64 nearbyint(f64 a) { return rint(a); }
static inline f64 round(f64 a) {  // half away from zero
  f64 t = trunc(a);
  f64 d = fabs(a - t);
  return (f64_le(0x3FE0000000000000UL, d.b)) ? t + copysign(f64(1), a) : t;
}
// fmod: exact (the remainder is representable), by the integer long
// division of musl's fmod.c on the significands.
static inline f64 fmod(f64 x, f64 y) {
  ulong ux = x.b, uy = y.b;
  int ex = int((ux >> 52) & 0x7FF), ey = int((uy >> 52) & 0x7FF);
  ulong sx = ux >> 63;
  if ((uy << 1) == 0 || isnan(y) || ex == 0x7FF) return (x * y) / (x * y);
  if ((ux << 1) <= (uy << 1)) return (ux << 1) == (uy << 1) ? x * f64(0) : x;
  if (!ex) { for (ulong i = ux << 12; (long)i >= 0; ex--, i <<= 1) {} ux <<= -ex + 1; }
  else { ux &= ~0UL >> 12; ux |= 1UL << 52; }
  if (!ey) { for (ulong i = uy << 12; (long)i >= 0; ey--, i <<= 1) {} uy <<= -ey + 1; }
  else { uy &= ~0UL >> 12; uy |= 1UL << 52; }
  for (; ex > ey; ex--) {
    ulong i = ux - uy;
    if ((i >> 63) == 0) { if (i == 0) return x * f64(0); ux = i; }
    ux <<= 1;
  }
  ulong i = ux - uy;
  if ((i >> 63) == 0) { if (i == 0) return x * f64(0); ux = i; }
  for (; (ux >> 52) == 0; ux <<= 1, ex--) {}
  if (ex > 0) { ux -= 1UL << 52; ux |= (ulong)ex << 52; }
  else { ux >>= -ex + 1; }
  ux |= sx << 63;
  return f64::bits(ux);
}

static inline f64 ldexp(f64 a, int n) {
  // Exact scaling by repeated power-of-two multiplies (each exact unless the
  // result under/overflows, where the final multiply rounds once).
  while (n > 1023) { a = a * f64::bits(0x7FE0000000000000UL); n -= 1023; }
  while (n < -1022) { a = a * f64::bits(0x0010000000000000UL); n += 1022; }
  return a * f64::bits((ulong)(n + 0x3FF) << 52);
}

// CUDA's binary64 intrinsics.
static inline f64 __dadd_rn(f64 a, f64 b) { return a + b; }
static inline f64 __dsub_rn(f64 a, f64 b) { return a - b; }
static inline f64 __dmul_rn(f64 a, f64 b) { return a * b; }
static inline f64 __ddiv_rn(f64 a, f64 b) { return a / b; }
static inline f64 __dsqrt_rn(f64 a) { return sqrt(a); }
static inline f64 __drcp_rn(f64 a) { return f64(1) / a; }
static inline f64 __fma_rn(f64 a, f64 b, f64 c) { return fma(a, b, c); }
static inline long __double_as_longlong(f64 a) { return (long)a.b; }
static inline f64 __longlong_as_double(long u) { return f64::bits((ulong)u); }
static inline int __double2hiint(f64 a) { return (int)(a.b >> 32); }
static inline int __double2loint(f64 a) { return (int)(uint)a.b; }
static inline f64 __hiloint2double(int hi, int lo) {
  return f64::bits(((ulong)(uint)hi << 32) | (ulong)(uint)lo);
}
static inline float __double2float_rn(f64 a) { return __woof_cvt_f32(a.b); }
static inline int __double2int_rz(f64 a) { return f64_to_i32(a.b); }
static inline int __double2int_rn(f64 a) { return f64_to_i32(rint(a).b); }
static inline int __double2int_rd(f64 a) { return f64_to_i32(floor(a).b); }
static inline int __double2int_ru(f64 a) { return f64_to_i32(ceil(a).b); }
static inline long __double2ll_rz(f64 a) { return f64_to_i64(a.b); }
static inline long __double2ll_rn(f64 a) { return f64_to_i64(rint(a).b); }
static inline f64 __int2double_rn(int a) { return f64(a); }
static inline f64 __ll2double_rn(long a) { return f64(a); }

// ---------------------------------------------------------------- binary32
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __frcp_rn(float a) { return 1.0f / a; }
static inline float __fsqrt_rn(float a) { return metal::precise::sqrt(a); }
static inline float __fmaf_rn(float a, float b, float c) { return metal::fma(a, b, c); }
static inline float __int_as_float(int a) { return as_type<float>(a); }
static inline int __float_as_int(float a) { return as_type<int>(a); }
static inline float __uint_as_float(uint a) { return as_type<float>(a); }
static inline uint __float_as_uint(float a) { return as_type<uint>(a); }
static inline int __float2int_rz(float a) { return (int)a; }
static inline int __float2int_rn(float a) { return (int)metal::rint(a); }
static inline int __float2int_rd(float a) { return (int)metal::floor(a); }
static inline int __float2int_ru(float a) { return (int)metal::ceil(a); }
static inline float __int2float_rn(int a) { return (float)a; }
static inline float __uint2float_rn(uint a) { return (float)a; }
static inline float __saturatef(float a) { return metal::clamp(a, 0.0f, 1.0f); }

static inline float sqrtf(float a) { return metal::precise::sqrt(a); }
static inline float sqrt(float a) { return metal::precise::sqrt(a); }
// -ftz=true: CUDA flushes a subnormal operand of abs, neg, copysign, min
// and max (abs.ftz / neg.ftz / min.ftz ...), which Metal implements as bit
// operations or compares that keep it.  __woof_ftz flushes on the bits;
// a unit that keeps subnormals (WOOF_RETAIN_SUBNORMALS) never flushes.
static inline float __woof_ftz(float a) {
#ifndef WOOF_RETAIN_SUBNORMALS
  uint u = as_type<uint>(a);
  if ((u & 0x7F800000u) == 0u) u &= 0x80000000u;
  return as_type<float>(u);
#else
  return a;
#endif
}
// unary minus on a float (the translator rewrites `-x` in flushing units)
static inline float __woof_neg(float a) { return -__woof_ftz(a); }
static inline float fabsf(float a) { return metal::fabs(__woof_ftz(a)); }
static inline float fabs(float a) { return fabsf(a); }
static inline float floorf(float a) { return metal::floor(a); }
static inline float floor(float a) { return metal::floor(a); }
static inline float ceilf(float a) { return metal::ceil(a); }
static inline float ceil(float a) { return metal::ceil(a); }
static inline float truncf(float a) { return metal::trunc(a); }
static inline float trunc(float a) { return metal::trunc(a); }
static inline float rintf(float a) { return metal::rint(a); }
static inline float rint(float a) { return metal::rint(a); }
static inline float nearbyintf(float a) { return metal::rint(a); }
static inline float roundf(float a) { return metal::round(a); }
static inline float round(float a) { return metal::round(a); }
static inline float copysignf(float a, float b) { return metal::copysign(__woof_ftz(a), b); }
static inline float copysign(float a, float b) { return copysignf(a, b); }
static inline float fmaf(float a, float b, float c) { return metal::fma(a, b, c); }
static inline float fma(float a, float b, float c) { return metal::fma(a, b, c); }
// CUDA's min/max order -0 below +0 (max(-0, +0) = +0, min = -0, in either
// operand order); Metal's return the second operand for equal zeros.  Equal
// operands combine their bits: OR for min (sign set if either), AND for max.
static inline float fminf(float a, float b) {
  a = __woof_ftz(a); b = __woof_ftz(b);
  return a == b ? as_type<float>(as_type<uint>(a) | as_type<uint>(b)) : metal::fmin(a, b);
}
static inline float fmaxf(float a, float b) {
  a = __woof_ftz(a); b = __woof_ftz(b);
  return a == b ? as_type<float>(as_type<uint>(a) & as_type<uint>(b)) : metal::fmax(a, b);
}
// `a > b ? a : b` / `a < b ? a : b` as NVCC compiles them (max.f32 /
// min.f32): the conditional's result, except that equal operands combine
// their bits, so -0 orders below +0 (translate select_minmax).
static inline float __woof_selmax(float a, float b) {
  return a > b ? a : (a == b ? as_type<float>(as_type<uint>(a) & as_type<uint>(b)) : b);
}
static inline float __woof_selmin(float a, float b) {
  return a < b ? a : (a == b ? as_type<float>(as_type<uint>(a) | as_type<uint>(b)) : b);
}
static inline float fmin(float a, float b) { return fminf(a, b); }
static inline float fmax(float a, float b) { return fmaxf(a, b); }
static inline float ldexpf(float a, int n) { return metal::ldexp(a, n); }
static inline bool isnan(float a) { return metal::isnan(a); }
static inline bool isinf(float a) { return metal::isinf(a); }
static inline bool isfinite(float a) { return metal::isfinite(a); }
static inline bool signbit(float a) { return metal::signbit(a); }
// Vendor libm: not CUDA's words.  Units graded word for word call woof's
// own ports (glibc_flt32.cuh and kin) instead; these are known deltas.
static inline float expf(float a) { return metal::precise::exp(a); }
static inline float exp2f(float a) { return metal::precise::exp2(a); }
static inline float logf(float a) { return metal::precise::log(a); }
static inline float log2f(float a) { return metal::precise::log2(a); }
static inline float log10f(float a) { return metal::precise::log10(a); }
static inline float powf(float a, float b) { return metal::precise::pow(a, b); }
static inline float sinf(float a) { return metal::precise::sin(a); }
static inline float cosf(float a) { return metal::precise::cos(a); }
static inline float tanf(float a) { return metal::precise::tan(a); }
static inline float asinf(float a) { return metal::precise::asin(a); }
static inline float acosf(float a) { return metal::precise::acos(a); }
static inline float atanf(float a) { return metal::precise::atan(a); }
static inline float atan2f(float a, float b) { return metal::precise::atan2(a, b); }
static inline float sinhf(float a) { return metal::precise::sinh(a); }
static inline float coshf(float a) { return metal::precise::cosh(a); }
static inline float tanhf(float a) { return metal::precise::tanh(a); }
static inline float fmodf(float a, float b) { return metal::precise::fmod(a, b); }

// ---------------------------------------------------------------- integers
static inline int min(int a, int b) { return metal::min(a, b); }
static inline int max(int a, int b) { return metal::max(a, b); }
static inline uint min(uint a, uint b) { return metal::min(a, b); }
static inline uint max(uint a, uint b) { return metal::max(a, b); }
static inline long min(long a, long b) { return metal::min(a, b); }
static inline long max(long a, long b) { return metal::max(a, b); }
static inline ulong min(ulong a, ulong b) { return metal::min(a, b); }
static inline ulong max(ulong a, ulong b) { return metal::max(a, b); }
static inline float min(float a, float b) { return fminf(a, b); }
static inline float max(float a, float b) { return fmaxf(a, b); }
static inline int abs(int a) { return metal::abs(a); }
static inline long llabs(long a) { return metal::abs(a); }
static inline int __clz(int a) { return metal::clz(a); }
static inline int __clzll(long a) { return (int)metal::clz(a); }
static inline int __popc(uint a) { return metal::popcount(a); }
static inline int __ffs(int a) { return a == 0 ? 0 : (int)metal::ctz((uint)a) + 1; }
static inline int __ffsll(long a) { return a == 0 ? 0 : (int)metal::ctz((ulong)a) + 1; }
static inline int __popcll(ulong a) { return (int)metal::popcount(a); }
static inline uint __umulhi(uint a, uint b) { return metal::mulhi(a, b); }
static inline ulong __umul64hi(ulong a, ulong b) { return metal::mulhi(a, b); }
static inline int __mulhi(int a, int b) { return metal::mulhi(a, b); }
static inline uint __brev(uint a) { return metal::reverse_bits(a); }

static inline void __syncthreads() { metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup | metal::mem_flags::mem_device); }
static inline void __threadfence() { metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device); }
static inline void __threadfence_block() { metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup); }

constant f64 DBL_MAX = f64::bits(0x7FEFFFFFFFFFFFFFUL);
constant f64 DBL_MIN = f64::bits(0x0010000000000000UL);
constant f64 DBL_EPSILON = f64::bits(0x3CB0000000000000UL);

// ---------------------------------------------------------------- atomics
// CUDA's atomics on device and threadgroup memory, relaxed ordering (CUDA's
// atomicX are relaxed).  CAS emulates CUDA's strong compare-and-swap with a
// weak-exchange loop and returns the old value as CUDA does.  64-bit
// atomics are not declared: Apple GPUs support 64-bit min/max only, so a
// unit using atomicAdd on double or unsigned long long fails by name.
#define WOOF_ATOMIC_RMW(AS, T, A, NAME, OP)                                              \
  static inline T NAME(AS T* p, T v) {                                                   \
    return metal::atomic_fetch_##OP##_explicit((AS A*)p, v, metal::memory_order_relaxed); }
#define WOOF_ATOMIC_INT(AS, T, A)                                                        \
  WOOF_ATOMIC_RMW(AS, T, A, atomicAdd, add) WOOF_ATOMIC_RMW(AS, T, A, atomicSub, sub)    \
  WOOF_ATOMIC_RMW(AS, T, A, atomicMax, max) WOOF_ATOMIC_RMW(AS, T, A, atomicMin, min)    \
  WOOF_ATOMIC_RMW(AS, T, A, atomicOr, or) WOOF_ATOMIC_RMW(AS, T, A, atomicAnd, and)      \
  WOOF_ATOMIC_RMW(AS, T, A, atomicXor, xor)                                              \
  static inline T atomicExch(AS T* p, T v) {                                             \
    return metal::atomic_exchange_explicit((AS A*)p, v, metal::memory_order_relaxed); }  \
  static inline T atomicCAS(AS T* p, T compare, T value) {                               \
    T expected = compare;                                                                \
    while (!metal::atomic_compare_exchange_weak_explicit((AS A*)p, &expected, value,     \
              metal::memory_order_relaxed, metal::memory_order_relaxed)) {               \
      if (expected != compare) break;                                                    \
    }                                                                                    \
    return expected;                                                                     \
  }
WOOF_ATOMIC_INT(device, int, metal::atomic_int)
WOOF_ATOMIC_INT(device, uint, metal::atomic_uint)
WOOF_ATOMIC_INT(threadgroup, int, metal::atomic_int)
WOOF_ATOMIC_INT(threadgroup, uint, metal::atomic_uint)
#define WOOF_ATOMIC_FLOAT(AS)                                                            \
  static inline float atomicAdd(AS float* p, float v) {                                  \
    return metal::atomic_fetch_add_explicit((AS metal::atomic_float*)p, v, metal::memory_order_relaxed); } \
  static inline float atomicExch(AS float* p, float v) {                                 \
    return metal::atomic_exchange_explicit((AS metal::atomic_float*)p, v, metal::memory_order_relaxed); }
WOOF_ATOMIC_FLOAT(device)
#undef WOOF_ATOMIC_FLOAT
// threadgroup memory has no atomic_float: a compare-exchange on the bits.
static inline float atomicAdd(threadgroup float* p, float v) {
  threadgroup metal::atomic_uint* a = (threadgroup metal::atomic_uint*)p;
  uint old = metal::atomic_load_explicit(a, metal::memory_order_relaxed);
  while (!metal::atomic_compare_exchange_weak_explicit(a, &old, as_type<uint>(as_type<float>(old) + v),
                                                       metal::memory_order_relaxed, metal::memory_order_relaxed)) {}
  return as_type<float>(old);
}
static inline float atomicExch(threadgroup float* p, float v) {
  return as_type<float>(metal::atomic_exchange_explicit((threadgroup metal::atomic_uint*)p, as_type<uint>(v),
                                                        metal::memory_order_relaxed));
}
#undef WOOF_ATOMIC_INT
#undef WOOF_ATOMIC_RMW

// ---------------------------------------------------------------- SIMD-groups
// A CUDA warp is an Apple SIMD-group: 32 lanes, consecutive threads of a
// threadgroup.  Masks are taken as "every lane" (woof's kernels pass
// 0xffffffff); a sub-warp width takes the segmented form CUDA defines.
static inline uint __woof_lane() { return metal::simd_prefix_exclusive_sum(1u); }
static inline void __syncwarp(uint mask = 0xffffffffu) {
  metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup | metal::mem_flags::mem_device);
}
static inline uint __ballot_sync(uint mask, int pred) {
  return (uint)(ulong)(metal::simd_vote::vote_t)metal::simd_ballot(pred != 0);
}
static inline int __any_sync(uint mask, int pred) { return metal::simd_any(pred != 0) ? 1 : 0; }
static inline int __all_sync(uint mask, int pred) { return metal::simd_all(pred != 0) ? 1 : 0; }
template <typename T> static inline T __woof_shfl(T v, uint src) { return metal::simd_shuffle(v, ushort(src)); }
static inline f64 __woof_shfl(f64 v, uint src) {
  uint lo = metal::simd_shuffle((uint)v.b, ushort(src));
  uint hi = metal::simd_shuffle((uint)(v.b >> 32), ushort(src));
  return f64::bits(((ulong)hi << 32) | lo);
}
template <typename T> static inline T __shfl_sync(uint mask, T v, int src, int width = 32) {
  uint lane = __woof_lane();
  uint base = lane & ~uint(width - 1);
  return __woof_shfl(v, base + (uint(src) & uint(width - 1)));
}
template <typename T> static inline T __shfl_down_sync(uint mask, T v, uint delta, int width = 32) {
  uint lane = __woof_lane();
  uint seg = lane & uint(width - 1);
  T got = __woof_shfl(v, (seg + delta < uint(width)) ? lane + delta : lane);
  return got;
}
template <typename T> static inline T __shfl_up_sync(uint mask, T v, uint delta, int width = 32) {
  uint lane = __woof_lane();
  uint seg = lane & uint(width - 1);
  return __woof_shfl(v, (seg >= delta) ? lane - delta : lane);
}
template <typename T> static inline T __shfl_xor_sync(uint mask, T v, int lane_mask, int width = 32) {
  uint lane = __woof_lane();
  uint other = lane ^ uint(lane_mask);
  return __woof_shfl(v, ((other & ~uint(width - 1)) == (lane & ~uint(width - 1))) ? other : lane);
}

// A CUDA T** kernel parameter: a buffer of device addresses (the facade
// passes real GPU addresses), read one entry at a time.
template <typename T> struct __woof_addr_table {
  const device ulong* w;
  __woof_addr_table(const device ulong* p) : w(p) {}
  // a table reached through a device pointer to its entries (`(T**)address`):
  // in device memory, an array of pointers is an array of 64-bit words
  template <typename P> __woof_addr_table(const device P* p) : w(reinterpret_cast<const device ulong*>(p)) {}
  T operator[](long i) const { return reinterpret_cast<T>(w[i]); }
  T operator*() const { return reinterpret_cast<T>(w[0]); }
};

// A read-only pointer that, as CUDA's generic pointers may, points into one
// address space now and another later (a const parameter reassigned to a
// threadgroup tile, then indexed): one pointer per space and a tag.  Reads
// dispatch on the tag; a use that needs a device pointer gets the device one.
template <typename T> struct __woof_vptr {
  const device T* d = nullptr; const threadgroup T* g = nullptr;
  const thread T* t = nullptr; const constant T* c = nullptr; int which = 0;
  __woof_vptr(const device T* p) : d(p), which(0) {}
  __woof_vptr(const threadgroup T* p) : g(p), which(1) {}
  __woof_vptr(const thread T* p) : t(p), which(2) {}
  __woof_vptr(const constant T* p) : c(p), which(3) {}
  thread __woof_vptr& operator=(const device T* p) { d = p; which = 0; return *this; }
  thread __woof_vptr& operator=(const threadgroup T* p) { g = p; which = 1; return *this; }
  thread __woof_vptr& operator=(const thread T* p) { t = p; which = 2; return *this; }
  thread __woof_vptr& operator=(const constant T* p) { c = p; which = 3; return *this; }
  T operator[](long i) const {
    return which == 0 ? d[i] : which == 1 ? g[i] : which == 2 ? t[i] : c[i];
  }
  T operator*() const { return (*this)[0]; }
  __woof_vptr operator+(long i) const {
    __woof_vptr r = *this;
    if (which == 0) r.d += i; else if (which == 1) r.g += i; else if (which == 2) r.t += i; else r.c += i;
    return r;
  }
  operator const device T*() const { return d; }
};

// ---------------------------------------------------------------- misc
static inline float nanf(constant char*) { return as_type<float>(0x7FC00000u); }
static inline f64 nan(constant char*) { return f64::bits(0x7FF8000000000000UL); }
static inline long lroundf(float a) { return (long)metal::round(a); }
static inline long llroundf(float a) { return (long)metal::round(a); }
static inline long lround(f64 a) { return f64_to_i64(round(a).b); }
static inline long llround(f64 a) { return f64_to_i64(round(a).b); }
static inline long lround(float a) { return (long)metal::round(a); }
static inline long __float2ll_rz(float a) { return (long)a; }
static inline long __float2ll_rn(float a) { return (long)metal::rint(a); }
// No trap on Metal: the thread carries on (woof's kernels trap only on
// states their launchers already refuse).
static inline void __trap() {}
// binary64 -> binary32 toward zero / up / down: the nearest result, stepped
// back across x when it rounded the wrong way.
static inline float __woof_f2f_dir(f64 a, int dir) {
  float r = f64_to_f32(a.b);
  if (isnan(a) || isinf(a)) return r;
  f64 back = f64(r);
  if (back == a) return r;
  bool above = a < back;  // r rounded up
  uint u = as_type<uint>(r);
  bool step = (dir == 0) ? ((r > 0.0f || (r == 0.0f && !signbit(a))) == above) :
              (dir > 0) ? !above : above;
  if (!step) return r;
  // one binary32 step toward the side x lies on
  bool toward_negative = above;
  if (r == 0.0f) return toward_negative ? as_type<float>(0x80000001u) : as_type<float>(0x00000001u);
  bool positive = (u >> 31) == 0u;
  return as_type<float>((positive == toward_negative) ? u - 1u : u + 1u);
}
static inline float __double2float_rz(f64 a) { return __woof_f2f_dir(a, 0); }
static inline float __double2float_ru(f64 a) { return __woof_f2f_dir(a, 1); }
static inline float __double2float_rd(f64 a) { return __woof_f2f_dir(a, -1); }

// ---------------------------------------------------------------- vectors
static inline float2 make_float2(float x, float y) { return float2(x, y); }
static inline float4 make_float4(float x, float y, float z, float w) { return float4(x, y, z, w); }
static inline packed_float3 make_packed_float3(float x, float y, float z) { return packed_float3(x, y, z); }
static inline int2 make_int2(int x, int y) { return int2(x, y); }
static inline int3 make_int3(int x, int y, int z) { return int3(x, y, z); }
static inline int4 make_int4(int x, int y, int z, int w) { return int4(x, y, z, w); }
static inline uint2 make_uint2(uint x, uint y) { return uint2(x, y); }
static inline uint3 make_uint3(uint x, uint y, uint z) { return uint3(x, y, z); }
static inline uint4 make_uint4(uint x, uint y, uint z, uint w) { return uint4(x, y, z, w); }
struct __woof_double2 { f64 x, y; };
struct __woof_double3 { f64 x, y, z; };
struct alignas(16) __woof_double4 { f64 x, y, z, w; };
static inline __woof_double2 __woof_make_double2(f64 x, f64 y) { __woof_double2 r; r.x = x; r.y = y; return r; }
static inline __woof_double3 __woof_make_double3(f64 x, f64 y, f64 z) { __woof_double3 r; r.x = x; r.y = y; r.z = z; return r; }
static inline __woof_double4 __woof_make_double4(f64 x, f64 y, f64 z, f64 w) {
  __woof_double4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }

// __syncthreads_or/_and/_count: a barrier-bracketed reduction in one
// threadgroup word (declared by the kernel, passed down by the translator).
#define WOOF_SYNC_REDUCE(NAME, INIT, OP)                                                      \
  static inline int NAME(threadgroup metal::atomic_int* s, int p) {                           \
    metal::atomic_store_explicit(s, INIT, metal::memory_order_relaxed);                       \
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);                            \
    OP;                                                                                       \
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);                            \
    int r = metal::atomic_load_explicit(s, metal::memory_order_relaxed);                      \
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);                            \
    return r;                                                                                 \
  }
WOOF_SYNC_REDUCE(__woof_syncthreads_or, 0, if (p) metal::atomic_fetch_or_explicit(s, 1, metal::memory_order_relaxed))
WOOF_SYNC_REDUCE(__woof_syncthreads_and, 1, if (!p) metal::atomic_fetch_and_explicit(s, 0, metal::memory_order_relaxed))
WOOF_SYNC_REDUCE(__woof_syncthreads_count, 0, if (p) metal::atomic_fetch_add_explicit(s, 1, metal::memory_order_relaxed))
#undef WOOF_SYNC_REDUCE

// 64-bit atomics.  Apple GPUs have 64-bit atomic min/max only, returning
// nothing; bitwise OR/AND/XOR split exactly into two 32-bit atomics on the
// halves.  These return void: code that reads CUDA's returned old value
// fails to compile, by name, instead of reading garbage.  (No 64-bit add.)
static inline void atomicMin(device ulong* p, ulong v) {
  metal::atomic_min_explicit((device metal::atomic_ulong*)p, v, metal::memory_order_relaxed); }
static inline void atomicMax(device ulong* p, ulong v) {
  metal::atomic_max_explicit((device metal::atomic_ulong*)p, v, metal::memory_order_relaxed); }
#define WOOF_ATOMIC64_SPLIT(NAME, OP)                                                          \
  static inline void NAME(device ulong* p, ulong v) {                                          \
    device metal::atomic_uint* w = (device metal::atomic_uint*)p;                              \
    metal::atomic_fetch_##OP##_explicit(w, (uint)v, metal::memory_order_relaxed);              \
    metal::atomic_fetch_##OP##_explicit(w + 1, (uint)(v >> 32), metal::memory_order_relaxed);  \
  }
WOOF_ATOMIC64_SPLIT(atomicOr, or)
WOOF_ATOMIC64_SPLIT(atomicAnd, and)
WOOF_ATOMIC64_SPLIT(atomicXor, xor)
#undef WOOF_ATOMIC64_SPLIT

// CUDA's max/min over mixed integer types: the usual arithmetic
// conversions, as C++ would choose the common type.
template <WOOF_ARITH(A), WOOF_ARITH(B)>
static inline auto max(A a, B b) -> decltype(a + b) { return max(decltype(a + b)(a), decltype(a + b)(b)); }  // the same-type max: CUDA's zero order
template <WOOF_ARITH(A), WOOF_ARITH(B)>
static inline auto min(A a, B b) -> decltype(a + b) { return min(decltype(a + b)(a), decltype(a + b)(b)); }
