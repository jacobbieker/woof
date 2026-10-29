// binary32 with gradual underflow, on GPUs that flush subnormals.
//
// Apple GPUs flush binary32 subnormal operands and results to zero on every
// arithmetic operation, in every math mode (woof/backend/metal/
// arith_receipt.json); copies keep the bits.  CUDA keeps subnormals where a
// unit compiles with --ftz=false -- every unit under GPUWM_WRF_EXACT=1, and
// the land, urban and solar units whose physics needs them.  In those units
// the translator spells `float` as `f32s`:
//
// * fast path: the hardware operation, whenever no operand is subnormal and
//   the hardware result is neither zero nor below the normal range -- then
//   no flush can have happened and the receipt's correctly rounded result
//   stands;
// * slow path: the operands widened exactly to binary64 (f64_soft.h), the
//   operation there, one rounding to binary32 with subnormals.  For + - * /
//   and sqrt that is the correctly rounded binary32 result (binary64 has
//   53 >= 2 * 24 + 2 bits, so the double rounding is innocuous); fma rounds
//   the exact binary64 sum with its TwoSum error as a sticky bit.
//
// Same layout as float, so `device float*` buffers stay float memory.
// Conversions: from the narrower types and from float, implicit; to f64
// (exact widening), implicit; from f64 and to float or an integer type,
// explicit -- the translator spells every such conversion as a cast, so a
// function with only a float overload cannot silently drop a subnormal
// through the hardware, and no call can reach two overloads equally.
#pragma once

static inline bool __woof_f32_sub(float x) {
  uint u = as_type<uint>(x) & 0x7FFFFFFFu;
  return u != 0u && u < 0x00800000u;
}
static inline bool __woof_f32_tiny(float x) {  // zero or below the normal range
  return (as_type<uint>(x) & 0x7FFFFFFFu) < 0x00800000u;
}
static inline bool __woof_f32_zero(float x) { return (as_type<uint>(x) & 0x7FFFFFFFu) == 0u; }
static inline ulong __woof_w(float x) { return f64_from_f32(x); }
static inline float __woof_n(ulong d) { return f64_to_f32(d); }
// The slow paths, out of line: inlining soft binary64 into every float
// operation of a large unit makes Metal's compile time explode, and these
// run only when an operand or result is tiny.
#define WOOF_SLOW __attribute__((noinline))
WOOF_SLOW static float __woof_slow_add(float a, float b) { return __woof_n(f64_add(__woof_w(a), __woof_w(b))); }
WOOF_SLOW static float __woof_slow_sub(float a, float b) { return __woof_n(f64_sub(__woof_w(a), __woof_w(b))); }
WOOF_SLOW static float __woof_slow_mul(float a, float b) { return __woof_n(f64_mul(__woof_w(a), __woof_w(b))); }
WOOF_SLOW static float __woof_slow_div(float a, float b) { return __woof_n(f64_div(__woof_w(a), __woof_w(b))); }
WOOF_SLOW static float __woof_slow_sqrt(float a) { return __woof_n(f64_sqrt(__woof_w(a))); }
WOOF_SLOW static int __woof_slow_cmp(float a, float b) {  // -1, 0, 1, or 2 unordered
  ulong x = __woof_w(a), y = __woof_w(b);
  if (F64_IS_NAN(x) || F64_IS_NAN(y)) return 2;
  return f64_lt(x, y) ? -1 : (f64_eq(x, y) ? 0 : 1);
}

struct f32s {
  float v;
  f32s() = default;
  f32s() threadgroup = default;  // threadgroup arrays of f32s
  constexpr f32s(float x) : v(x) {}
  // a constant table's elements are constructed in constant memory
  constexpr f32s(float x) constant : v(x) {}
  f32s(int x) : v((float)x) {}
  f32s(uint x) : v((float)x) {}
  f32s(long x) : v((float)x) {}
  f32s(ulong x) : v((float)x) {}
  f32s(short x) : v((float)x) {}
  f32s(ushort x) : v((float)x) {}
  f32s(char x) : v((float)x) {}
  f32s(uchar x) : v((float)x) {}
  f32s(bool x) : v(x ? 1.0f : 0.0f) {}
  explicit f32s(f64 x) : v(f64_to_f32(x.b)) {}
  static f32s bits(uint u) { f32s r; r.v = as_type<float>(u); return r; }
#define WOOF_F32S_CONV(AS)                                                    \
  explicit operator float() const AS { return v; }                            \
  operator f64() const AS { return f64::bits(f64_from_f32(v)); }              \
  explicit operator int() const AS { return (int)v; }                                  \
  explicit operator uint() const AS { return (uint)v; }                                \
  explicit operator long() const AS { return (long)v; }                                \
  explicit operator ulong() const AS { return (ulong)v; }                              \
  explicit operator short() const AS { return (short)v; }                              \
  explicit operator ushort() const AS { return (ushort)v; }                            \
  explicit operator char() const AS { return (char)v; }                                \
  explicit operator uchar() const AS { return (uchar)v; }                              \
  explicit operator bool() const AS { return !__woof_f32_zero(v); }
  WOOF_F32S_CONV(thread)
  WOOF_F32S_CONV(device)
  WOOF_F32S_CONV(constant)
  WOOF_F32S_CONV(threadgroup)
#undef WOOF_F32S_CONV
};

static inline f32s operator+(f32s a, f32s b) {
  float r = a.v + b.v;
  if (__woof_f32_tiny(r) || __woof_f32_sub(a.v) || __woof_f32_sub(b.v))
    return f32s(__woof_slow_add(a.v, b.v));
  return f32s(r);
}
static inline f32s operator-(f32s a, f32s b) {
  float r = a.v - b.v;
  if (__woof_f32_tiny(r) || __woof_f32_sub(a.v) || __woof_f32_sub(b.v))
    return f32s(__woof_slow_sub(a.v, b.v));
  return f32s(r);
}
static inline f32s operator*(f32s a, f32s b) {
  float r = a.v * b.v;
  if ((__woof_f32_tiny(r) && !__woof_f32_zero(a.v) && !__woof_f32_zero(b.v))
      || __woof_f32_sub(a.v) || __woof_f32_sub(b.v))
    return f32s(__woof_slow_mul(a.v, b.v));
  return f32s(r);
}
static inline f32s operator/(f32s a, f32s b) {
  float r = a.v / b.v;
  if ((__woof_f32_tiny(r) && !__woof_f32_zero(a.v)) || __woof_f32_sub(a.v) || __woof_f32_sub(b.v))
    return f32s(__woof_slow_div(a.v, b.v));
  return f32s(r);
}
static inline f32s operator-(f32s a) { return f32s::bits(as_type<uint>(a.v) ^ 0x80000000u); }
static inline f32s operator+(f32s a) { return a; }
static inline bool operator!(f32s a) { return __woof_f32_zero(a.v); }

// Comparisons: the hardware reads a subnormal operand as zero.
#define WOOF_F32S_CMP(OP, TEST)                                                    \
  static inline bool operator OP(f32s a, f32s b) {                                 \
    if (__woof_f32_sub(a.v) || __woof_f32_sub(b.v)) {                              \
      int c = __woof_slow_cmp(a.v, b.v);                                           \
      return TEST;                                                                 \
    }                                                                              \
    return a.v OP b.v;                                                             \
  }
WOOF_F32S_CMP(==, c == 0)
WOOF_F32S_CMP(!=, c != 0)
WOOF_F32S_CMP(<, c == -1)
WOOF_F32S_CMP(<=, c == -1 || c == 0)
WOOF_F32S_CMP(>, c == 1)
WOOF_F32S_CMP(>=, c == 1 || c == 0)
#undef WOOF_F32S_CMP

// Mixed operands, as C++ evaluates float with the other types: integers
// convert to float; f64 makes the operation binary64.
#define WOOF_F32S_MIXED(OP, R)                                                                   \
  template <WOOF_ARITH(T)> static inline R operator OP(f32s a, T b) { return a OP f32s(b); }     \
  template <WOOF_ARITH(T)> static inline R operator OP(T a, f32s b) { return f32s(a) OP b; }     \
  static inline auto operator OP(f32s a, f64 b) { return f64(a) OP b; }                         \
  static inline auto operator OP(f64 a, f32s b) { return a OP f64(b); }
WOOF_F32S_MIXED(+, f32s)
WOOF_F32S_MIXED(-, f32s)
WOOF_F32S_MIXED(*, f32s)
WOOF_F32S_MIXED(/, f32s)
WOOF_F32S_MIXED(==, bool)
WOOF_F32S_MIXED(!=, bool)
WOOF_F32S_MIXED(<, bool)
WOOF_F32S_MIXED(<=, bool)
WOOF_F32S_MIXED(>, bool)
WOOF_F32S_MIXED(>=, bool)
#undef WOOF_F32S_MIXED

#define WOOF_F32S_ASSIGN(AS)                                                                       \
  static inline AS f32s& operator+=(AS f32s& a, f32s b) { a = a + b; return a; }                    \
  static inline AS f32s& operator-=(AS f32s& a, f32s b) { a = a - b; return a; }                    \
  static inline AS f32s& operator*=(AS f32s& a, f32s b) { a = a * b; return a; }                    \
  static inline AS f32s& operator/=(AS f32s& a, f32s b) { a = a / b; return a; }                    \
  template <WOOF_ARITH(T)> static inline AS f32s& operator+=(AS f32s& a, T b) { a = a + f32s(b); return a; } \
  template <WOOF_ARITH(T)> static inline AS f32s& operator-=(AS f32s& a, T b) { a = a - f32s(b); return a; } \
  template <WOOF_ARITH(T)> static inline AS f32s& operator*=(AS f32s& a, T b) { a = a * f32s(b); return a; } \
  template <WOOF_ARITH(T)> static inline AS f32s& operator/=(AS f32s& a, T b) { a = a / f32s(b); return a; } \
  static inline AS f32s& operator+=(AS f32s& a, f64 b) { a = f32s(f64(a) + b); return a; }          \
  static inline AS f32s& operator-=(AS f32s& a, f64 b) { a = f32s(f64(a) - b); return a; }          \
  static inline AS f32s& operator*=(AS f32s& a, f64 b) { a = f32s(f64(a) * b); return a; }          \
  static inline AS f32s& operator/=(AS f32s& a, f64 b) { a = f32s(f64(a) / b); return a; }          \
  static inline AS f64& operator+=(AS f64& a, f32s b) { a = a + f64(b); return a; }                 \
  static inline AS f64& operator-=(AS f64& a, f32s b) { a = a - f64(b); return a; }                 \
  static inline AS f64& operator*=(AS f64& a, f32s b) { a = a * f64(b); return a; }                 \
  static inline AS f64& operator/=(AS f64& a, f32s b) { a = a / f64(b); return a; }                 \
  static inline AS f32s& operator++(AS f32s& a) { a = a + f32s(1.0f); return a; }                   \
  static inline AS f32s& operator--(AS f32s& a) { a = a - f32s(1.0f); return a; }                   \
  static inline f32s operator++(AS f32s& a, int) { f32s o = a; a = a + f32s(1.0f); return o; }      \
  static inline f32s operator--(AS f32s& a, int) { f32s o = a; a = a - f32s(1.0f); return o; }
WOOF_F32S_ASSIGN(thread)
WOOF_F32S_ASSIGN(device)
WOOF_F32S_ASSIGN(threadgroup)
#undef WOOF_F32S_ASSIGN

// ---------------------------------------------------------------- math
static inline f32s fabsf(f32s a) { return f32s::bits(as_type<uint>(a.v) & 0x7FFFFFFFu); }
static inline f32s fabs(f32s a) { return fabsf(a); }
static inline f32s copysignf(f32s a, f32s b) {
  return f32s::bits((as_type<uint>(a.v) & 0x7FFFFFFFu) | (as_type<uint>(b.v) & 0x80000000u));
}
static inline f32s copysign(f32s a, f32s b) { return copysignf(a, b); }
static inline bool isnan(f32s a) { return (as_type<uint>(a.v) & 0x7FFFFFFFu) > 0x7F800000u; }
static inline bool isinf(f32s a) { return (as_type<uint>(a.v) & 0x7FFFFFFFu) == 0x7F800000u; }
static inline bool isfinite(f32s a) { return (as_type<uint>(a.v) & 0x7F800000u) != 0x7F800000u; }
static inline bool signbit(f32s a) { return (as_type<uint>(a.v) >> 31) != 0u; }
static inline f32s sqrtf(f32s a) {
  if (__woof_f32_sub(a.v)) return f32s(__woof_slow_sqrt(a.v));
  return f32s(metal::precise::sqrt(a.v));
}
static inline f32s sqrt(f32s a) { return sqrtf(a); }
static inline f32s fmaf(f32s a, f32s b, f32s c) {
  float r = metal::fma(a.v, b.v, c.v);
  if (!(__woof_f32_tiny(r) || __woof_f32_sub(a.v) || __woof_f32_sub(b.v) || __woof_f32_sub(c.v)))
    return f32s(r);
  // The product is exact in binary64; TwoSum gives the rounded sum s and
  // its exact error e.  Rounding s to odd (truncate toward zero when
  // inexact, then set the last bit) and then to binary32 is the correctly
  // rounded binary32 fma, since 53 >= 24 + 2 (Boldo & Melquiond).
  ulong p = f64_mul(__woof_w(a.v), __woof_w(b.v)), cc = __woof_w(c.v);
  ulong s = f64_add(p, cc);
  ulong bp = f64_sub(s, cc);
  ulong e = f64_add(f64_sub(p, bp), f64_sub(cc, f64_sub(s, bp)));
  if ((e << 1) != 0 && ((s >> 52) & 0x7FF) != 0x7FF) {
    bool beyond = (e >> 63) != (s >> 63);  // |exact| < |s|: s rounded away from zero
    if (beyond) s -= 1;                    // the next binary64 toward zero
    s |= 1;
  }
  return f32s(__woof_n(s));
}
static inline f32s fma(f32s a, f32s b, f32s c) { return fmaf(a, b, c); }
static inline f32s fminf(f32s a, f32s b) {
  if (isnan(a)) return b;
  if (isnan(b)) return a;
  if (a == b) return f32s(as_type<float>(as_type<uint>(a.v) | as_type<uint>(b.v)));  // -0 < +0, as CUDA
  return (b < a) ? b : a;
}
static inline f32s fmaxf(f32s a, f32s b) {
  if (isnan(a)) return b;
  if (isnan(b)) return a;
  if (a == b) return f32s(as_type<float>(as_type<uint>(a.v) & as_type<uint>(b.v)));
  return (a < b) ? b : a;
}
// float beside f32s (a literal beside a value): the f32s overloads, exactly,
// or the call is ambiguous with the binary64 mixed templates
#define WOOF_F32S_MIXED_MINMAX(N, F)                                              \
  static inline f32s N(float a, f32s b) { return F(f32s(a), b); }                 \
  static inline f32s N(f32s a, float b) { return F(a, f32s(b)); }
WOOF_F32S_MIXED_MINMAX(fminf, fminf) WOOF_F32S_MIXED_MINMAX(fmaxf, fmaxf)
WOOF_F32S_MIXED_MINMAX(fmin, fminf) WOOF_F32S_MIXED_MINMAX(fmax, fmaxf)
WOOF_F32S_MIXED_MINMAX(min, fminf) WOOF_F32S_MIXED_MINMAX(max, fmaxf)
#undef WOOF_F32S_MIXED_MINMAX
static inline f32s fmin(f32s a, f32s b) { return fminf(a, b); }
static inline f32s fmax(f32s a, f32s b) { return fmaxf(a, b); }
static inline f32s min(f32s a, f32s b) { return fminf(a, b); }
static inline f32s max(f32s a, f32s b) { return fmaxf(a, b); }
// Integral rounding: a subnormal rounds to a signed zero or to +-1.
static inline f32s floorf(f32s a) {
  if (__woof_f32_sub(a.v)) return signbit(a) ? f32s(-1.0f) : f32s::bits(0u);
  return f32s(metal::floor(a.v));
}
static inline f32s ceilf(f32s a) {
  if (__woof_f32_sub(a.v)) return signbit(a) ? f32s::bits(0x80000000u) : f32s(1.0f);
  return f32s(metal::ceil(a.v));
}
static inline f32s truncf(f32s a) {
  if (__woof_f32_sub(a.v)) return f32s::bits(as_type<uint>(a.v) & 0x80000000u);
  return f32s(metal::trunc(a.v));
}
static inline f32s roundf(f32s a) { return __woof_f32_sub(a.v) ? truncf(a) : f32s(metal::round(a.v)); }
static inline f32s rintf(f32s a) { return __woof_f32_sub(a.v) ? truncf(a) : f32s(metal::rint(a.v)); }
static inline f32s nearbyintf(f32s a) { return rintf(a); }
static inline f32s floor(f32s a) { return floorf(a); }
static inline f32s ceil(f32s a) { return ceilf(a); }
static inline f32s trunc(f32s a) { return truncf(a); }
static inline f32s round(f32s a) { return roundf(a); }
static inline f32s rint(f32s a) { return rintf(a); }
static inline f32s ldexpf(f32s a, int n) { return f32s(__woof_n(ldexp(f64::bits(__woof_w(a.v)), n).b)); }

// CUDA's binary32 intrinsics.
static inline f32s __fadd_rn(f32s a, f32s b) { return a + b; }
static inline f32s __fsub_rn(f32s a, f32s b) { return a - b; }
static inline f32s __fmul_rn(f32s a, f32s b) { return a * b; }
static inline f32s __fdiv_rn(f32s a, f32s b) { return a / b; }
static inline f32s __frcp_rn(f32s a) { return f32s(1.0f) / a; }
static inline f32s __fsqrt_rn(f32s a) { return sqrtf(a); }
static inline f32s __fmaf_rn(f32s a, f32s b, f32s c) { return fmaf(a, b, c); }
static inline uint __float_as_uint(f32s a) { return as_type<uint>(a.v); }
static inline int __float_as_int(f32s a) { return as_type<int>(a.v); }
static inline int __float2int_rz(f32s a) { return (int)a.v; }
static inline int __float2int_rn(f32s a) { return (int)metal::rint(a.v); }
static inline int __float2int_rd(f32s a) { return (int)floorf(a).v; }
static inline int __float2int_ru(f32s a) { return (int)ceilf(a).v; }
// Vendor libm (known deltas, as in the flushing build).
#define WOOF_F32S_VENDOR1(N) static inline f32s N##f(f32s a) { return f32s(N##f(a.v)); }
WOOF_F32S_VENDOR1(exp) WOOF_F32S_VENDOR1(exp2) WOOF_F32S_VENDOR1(log) WOOF_F32S_VENDOR1(log2)
WOOF_F32S_VENDOR1(log10) WOOF_F32S_VENDOR1(sin) WOOF_F32S_VENDOR1(cos) WOOF_F32S_VENDOR1(tan)
WOOF_F32S_VENDOR1(asin) WOOF_F32S_VENDOR1(acos) WOOF_F32S_VENDOR1(atan) WOOF_F32S_VENDOR1(sinh)
WOOF_F32S_VENDOR1(cosh) WOOF_F32S_VENDOR1(tanh)
#undef WOOF_F32S_VENDOR1
static inline f32s powf(f32s a, f32s b) { return f32s(powf(a.v, b.v)); }
static inline f32s atan2f(f32s a, f32s b) { return f32s(atan2f(a.v, b.v)); }
static inline f32s fmodf(f32s a, f32s b) { return f32s(__woof_n(fmod(f64(a), f64(b)).b)); }
static inline long lroundf(f32s a) { return (long)roundf(a).v; }
static inline long lrintf(f32s a) { return (long)rintf(a).v; }
static inline long llroundf(f32s a) { return (long)roundf(a).v; }
// Atomics: CUDA's atom.add.f32 flushes subnormal inputs and results even in
// a --ftz=false unit (PTX ISA), as Metal's hardware float atomics do, so
// f32s atomics are the hardware ones on the same words.
#define WOOF_F32S_ATOMIC(AS)                                                              \
  static inline f32s atomicAdd(AS f32s* p, f32s v) { return f32s(atomicAdd((AS float*)p, v.v)); } \
  static inline f32s atomicExch(AS f32s* p, f32s v) { return f32s(atomicExch((AS float*)p, v.v)); }
WOOF_F32S_ATOMIC(device)
WOOF_F32S_ATOMIC(threadgroup)
#undef WOOF_F32S_ATOMIC
// PTX arithmetic without .ftz (add/sub/mul/div.rn.f32, min/max.f32): IEEE
// binary32 with its subnormals in any unit, flushing or not -- CUDA executes
// these unflushed whatever -ftz says, and woof's kernels use them to pin a
// rounding (woof/backend/translate rewrite_ptx_asm).
static inline float __woof_ieee_add(float a, float b) { return (f32s(a) + f32s(b)).v; }
static inline float __woof_ieee_sub(float a, float b) { return (f32s(a) - f32s(b)).v; }
static inline float __woof_ieee_mul(float a, float b) { return (f32s(a) * f32s(b)).v; }
static inline float __woof_ieee_div(float a, float b) { return (f32s(a) / f32s(b)).v; }
static inline float __woof_ieee_min(float a, float b) { return fminf(f32s(a), f32s(b)).v; }
static inline float __woof_ieee_max(float a, float b) { return fmaxf(f32s(a), f32s(b)).v; }
static inline f32s __woof_ieee_add(f32s a, f32s b) { return a + b; }
static inline f32s __woof_ieee_sub(f32s a, f32s b) { return a - b; }
static inline f32s __woof_ieee_mul(f32s a, f32s b) { return a * b; }
static inline f32s __woof_ieee_div(f32s a, f32s b) { return a / b; }
static inline f32s __woof_ieee_min(f32s a, f32s b) { return fminf(a, b); }
static inline f32s __woof_ieee_max(f32s a, f32s b) { return fmaxf(a, b); }
static inline f32s __woof_selmax(f32s a, f32s b) {
  return a > b ? a : (a == b ? f32s(as_type<float>(as_type<uint>(a.v) & as_type<uint>(b.v))) : b);
}
static inline f32s __woof_selmin(f32s a, f32s b) {
  return a < b ? a : (a == b ? f32s(as_type<float>(as_type<uint>(a.v) | as_type<uint>(b.v))) : b);
}
