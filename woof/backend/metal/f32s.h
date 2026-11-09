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
WOOF_SLOW static float __woof_soft_add(float a, float b) { return __woof_n(f64_add(__woof_w(a), __woof_w(b))); }
WOOF_SLOW static float __woof_soft_sub(float a, float b) { return __woof_n(f64_sub(__woof_w(a), __woof_w(b))); }
WOOF_SLOW static float __woof_soft_mul(float a, float b) { return __woof_n(f64_mul(__woof_w(a), __woof_w(b))); }
WOOF_SLOW static float __woof_soft_div(float a, float b) { return __woof_n(f64_div(__woof_w(a), __woof_w(b))); }

// The slow paths without soft binary64 where the operands allow; each result
// is the correctly rounded binary32 with gradual underflow, the soft path's
// word.  Infinities, NaNs and zeros keep the soft path (and its NaN words).
//
// + and -: when the exact sum is below the normal range, or an operand is
// subnormal, both operands are below 2^-100 (a nonzero sum smaller than 2^-126
// needs both operands' ulps <= 2^-126).  Below 2^-100 an operand is an exact
// integer count of 2^-149 under 2^50, so the sum is exact in 64-bit integers;
// under 2^23 counts it is the subnormal itself, otherwise its conversion to
// float rounds to nearest even once and the scaling by 2^-149 is exact.
static inline bool __woof_f32_small(float x) {   // finite and |x| < 2^-100
  return (as_type<uint>(x) & 0x7FFFFFFFu) < 0x0D800000u;
}
static inline long __woof_f32_units(float x) {   // x / 2^-149, |x| < 2^-100
  uint u = as_type<uint>(x), e = (u >> 23) & 0xFFu, m = u & 0x7FFFFFu;
  long v = e == 0u ? long(m) : (long(m | 0x800000u) << (e - 1u));
  return (u >> 31) != 0u ? -v : v;
}
static inline float __woof_f32_from_units(long v) {
  ulong a = v < 0 ? ulong(-v) : ulong(v);
  uint sign = v < 0 ? 0x80000000u : 0u;
  if (a < 0x800000ul) return as_type<float>(sign | uint(a));
  float f = float(a) * 0x1p-100f * 0x1p-49f;
  return as_type<float>(sign | as_type<uint>(f));
}
WOOF_SLOW static float __woof_slow_add(float a, float b) {
  if (!__woof_f32_small(a) || !__woof_f32_small(b)) return __woof_soft_add(a, b);
  long v = __woof_f32_units(a) + __woof_f32_units(b);
  if (v == 0) return __woof_soft_add(a, b);   // a signed zero: the soft path's sign rules
  return __woof_f32_from_units(v);
}
WOOF_SLOW static float __woof_slow_sub(float a, float b) {
  if (!__woof_f32_small(a) || !__woof_f32_small(b)) return __woof_soft_sub(a, b);
  long v = __woof_f32_units(a) - __woof_f32_units(b);
  if (v == 0) return __woof_soft_sub(a, b);   // a signed zero: the soft path's sign rules
  return __woof_f32_from_units(v);
}

// * and /: on the significands in [1, 2) the hardware rounds once with no
// underflow, and fma gives the exact error (product) or remainder (quotient),
// whose sign says where the exact value lies.  A normal result is the rounded
// significand scaled by a power of two; a subnormal one shifts the significand
// right, rounding to nearest even with the error as the sticky bit.
static inline uint __woof_f32_significand(uint u, thread int& e) {   // u = |x| bits, x != 0
  uint ex = u >> 23, m = u & 0x7FFFFFu;
  if (ex != 0u) { e = int(ex) - 127; return m | 0x800000u; }
  int lz = int(metal::clz(m)) - 8;
  e = -126 - lz;
  return m << uint(lz);
}
static inline float __woof_f32_place(uint sign, float p, float err, int e) {
  uint up = as_type<uint>(p);
  int E = e + int(up >> 23) - 127;
  uint M = (up & 0x7FFFFFu) | 0x800000u;
  if (E > 127) return as_type<float>(sign | 0x7F800000u);
  if (E >= -126) return as_type<float>(sign | (uint(E + 127) << 23) | (M & 0x7FFFFFu));
  int s = -(E + 126);
  if (s > 25) return as_type<float>(sign);
  uint R = M >> uint(s), rem = M & ((1u << uint(s)) - 1u), halfway = 1u << uint(s - 1);
  if (rem > halfway || (rem == halfway && (err > 0.0f || (err == 0.0f && (R & 1u) != 0u)))) R += 1u;
  return as_type<float>(sign | R);
}
static inline bool __woof_f32_ordinary(float x) {   // finite and nonzero
  uint u = as_type<uint>(x) & 0x7FFFFFFFu;
  return u != 0u && u < 0x7F800000u;
}
WOOF_SLOW static float __woof_slow_mul(float a, float b) {
  if (!__woof_f32_ordinary(a) || !__woof_f32_ordinary(b)) return __woof_soft_mul(a, b);
  uint ua = as_type<uint>(a), ub = as_type<uint>(b);
  int ea, eb;
  uint ma = __woof_f32_significand(ua & 0x7FFFFFFFu, ea), mb = __woof_f32_significand(ub & 0x7FFFFFFFu, eb);
  float fa = as_type<float>(0x3F800000u | (ma & 0x7FFFFFu)), fb = as_type<float>(0x3F800000u | (mb & 0x7FFFFFu));
  float p = fa * fb;
  float err = metal::fma(fa, fb, -p);
  return __woof_f32_place((ua ^ ub) & 0x80000000u, p, err, ea + eb);
}
WOOF_SLOW static float __woof_slow_div(float a, float b) {
  if (!__woof_f32_ordinary(a) || !__woof_f32_ordinary(b)) return __woof_soft_div(a, b);
  uint ua = as_type<uint>(a), ub = as_type<uint>(b);
  int ea, eb;
  uint ma = __woof_f32_significand(ua & 0x7FFFFFFFu, ea), mb = __woof_f32_significand(ub & 0x7FFFFFFFu, eb);
  float fa = as_type<float>(0x3F800000u | (ma & 0x7FFFFFu)), fb = as_type<float>(0x3F800000u | (mb & 0x7FFFFFu));
  float q = fa / fb;
  float rem = metal::fma(-q, fb, fa);
  return __woof_f32_place((ua ^ ub) & 0x80000000u, q, rem, ea - eb);
}
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
  // `float_lvalue = double`: C++ converts implicitly, and in a dependent
  // context (a template the translator made, CLM lake's generic array
  // bases) the translator sees no conversion to spell.  Assignment only,
  // and only from f64 (a float argument keeps the copy assignment).
#define WOOF_F32S_ASSIGN_F64(AS)                                                        \
  template <typename __T, typename = metal::enable_if_t<metal::is_same_v<__T, f64>>>    \
  AS f32s& operator=(__T x) AS { v = f64_to_f32(x.b); return *this; }
  WOOF_F32S_ASSIGN_F64(thread)
  WOOF_F32S_ASSIGN_F64(device)
  WOOF_F32S_ASSIGN_F64(threadgroup)
#undef WOOF_F32S_ASSIGN_F64
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

// A zero result from operands that are not subnormal is exact when the
// operands cancel (a == -b for +, a == b for -; both zeros included): no
// flush happened, and the hardware's IEEE zero, signed as round-to-nearest
// signs it, is the result.  Only a nonzero exact sum below the normal range
// needs the slow path.  Fields of zeros (an absent hydrometeor) otherwise
// took soft binary64 for every 0 + 0: a stencil over zeros ran 12x slower
// than plain float, 4x slower than over nonzero data.
// One test on the hardware result first, where it settles the operation.
// + and -: |r| >= 2^-101 and finite (exponent field 26..254).  A flushed
// operand (|x| < 2^-126) is then below half the gap from r to either
// neighbour (that gap is at least ulp(r) / 2 >= 2^-125), so r is the correctly
// rounded result whether or not one was flushed -- the slow path's word too.
// * and /: r finite and normal (exponent field 1..254).  A flushed operand
// makes the hardware product or quotient zero, infinite or NaN, so no flush
// happened and the full test below would take the hardware result anyway.
// Inf, NaN and small results take the full test, as before.
static inline bool __woof_f32_sum_settled(float r) {
  return (as_type<uint>(r) & 0x7F800000u) - 0x0D000000u <= 0x72000000u;
}
static inline bool __woof_f32_normal(float r) {
  return (as_type<uint>(r) & 0x7F800000u) - 0x00800000u <= 0x7E800000u;
}
static inline f32s operator+(f32s a, f32s b) {
  float r = a.v + b.v;
  if (__woof_f32_sum_settled(r)) return f32s(r);
  if ((__woof_f32_tiny(r) && a.v != -b.v) || __woof_f32_sub(a.v) || __woof_f32_sub(b.v))
    return f32s(__woof_slow_add(a.v, b.v));
  return f32s(r);
}
static inline f32s operator-(f32s a, f32s b) {
  float r = a.v - b.v;
  if (__woof_f32_sum_settled(r)) return f32s(r);
  if ((__woof_f32_tiny(r) && a.v != b.v) || __woof_f32_sub(a.v) || __woof_f32_sub(b.v))
    return f32s(__woof_slow_sub(a.v, b.v));
  return f32s(r);
}
static inline f32s operator*(f32s a, f32s b) {
  float r = a.v * b.v;
  if (__woof_f32_normal(r)) return f32s(r);
  if ((__woof_f32_tiny(r) && !__woof_f32_zero(a.v) && !__woof_f32_zero(b.v))
      || __woof_f32_sub(a.v) || __woof_f32_sub(b.v))
    return f32s(__woof_slow_mul(a.v, b.v));
  return f32s(r);
}
static inline f32s operator/(f32s a, f32s b) {
  float r = a.v / b.v;
  if (__woof_f32_normal(r)) return f32s(r);
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
static inline f32s abs(f32s a) { return fabsf(a); }
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
  // a zero product plus a zero is an exact (signed) zero, as for + above
  if (!((__woof_f32_tiny(r) && !((a.v == 0.0f || b.v == 0.0f) && c.v == 0.0f))
        || __woof_f32_sub(a.v) || __woof_f32_sub(b.v) || __woof_f32_sub(c.v)))
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
static inline f32s nextafterf(f32s x, f32s y) {
  // on the bits, ordered as sign-magnitude integers: exact for subnormals
  uint ux = as_type<uint>(x.v), uy = as_type<uint>(y.v);
  if ((ux & 0x7FFFFFFFu) > 0x7F800000u || (uy & 0x7FFFFFFFu) > 0x7F800000u)
    return f32s(as_type<float>(0x7FC00000u));
  int ix = (ux >> 31) ? -int(ux & 0x7FFFFFFFu) : int(ux);
  int iy = (uy >> 31) ? -int(uy & 0x7FFFFFFFu) : int(uy);
  if (ix == iy) return y;
  ix += (ix < iy) ? 1 : -1;
  return f32s(as_type<float>(ix < 0 ? (uint(-ix) | 0x80000000u) : uint(ix)));
}
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
static inline f32s scalbnf(f32s a, int n) { return ldexpf(a, n); }
static inline f32s frexpf(f32s a, thread int* e) { return f32s(frexpf(a.v, e)); }

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
WOOF_F32S_VENDOR1(cosh) WOOF_F32S_VENDOR1(tanh) WOOF_F32S_VENDOR1(expm1) WOOF_F32S_VENDOR1(erf)
WOOF_F32S_VENDOR1(erfc)
#undef WOOF_F32S_VENDOR1
static inline f32s rsqrtf(f32s a) { return f32s(rsqrtf(a.v)); }
// a warp shuffle moves the float's bits (subnormal words included); the
// generic __woof_shfl has no simd_shuffle of a struct
static inline f32s __woof_shfl(f32s v, uint src) {
  return f32s(as_type<float>(metal::simd_shuffle(as_type<uint>(v.v), ushort(src))));
}
static inline f32s __powf(f32s a, f32s b) { return f32s(__powf(a.v, b.v)); }
// a volatile load of an f32s reads its float's bits (prelude __woof_vload)
#define WOOF_F32S_VLOAD(AS) \
  static inline f32s __woof_vload(const AS f32s& x) { return f32s(*(const volatile AS float*)&x.v); }
WOOF_F32S_VLOAD(thread) WOOF_F32S_VLOAD(device) WOOF_F32S_VLOAD(threadgroup)
#undef WOOF_F32S_VLOAD
static inline f32s __expf(f32s a) { return f32s(__expf(a.v)); }
static inline f32s __logf(f32s a) { return f32s(__logf(a.v)); }
static inline f32s powf(f32s a, f32s b) { return f32s(powf(a.v, b.v)); }
static inline f32s atan2f(f32s a, f32s b) { return f32s(atan2f(a.v, b.v)); }
static inline f32s fmodf(f32s a, f32s b) { return f32s(__woof_n(fmod(f64(a), f64(b)).b)); }
// C++'s float overloads (CUDA's <cmath>: pow(float, float) is powf, not
// the double function), so `pow(a, b)` of two floats stays binary32.
#define WOOF_F32S_CXX1(N) static inline f32s N(f32s a) { return N##f(a); }
WOOF_F32S_CXX1(exp) WOOF_F32S_CXX1(exp2) WOOF_F32S_CXX1(log) WOOF_F32S_CXX1(log2)
WOOF_F32S_CXX1(log10) WOOF_F32S_CXX1(sin) WOOF_F32S_CXX1(cos) WOOF_F32S_CXX1(tan)
WOOF_F32S_CXX1(asin) WOOF_F32S_CXX1(acos) WOOF_F32S_CXX1(atan) WOOF_F32S_CXX1(sinh)
WOOF_F32S_CXX1(cosh) WOOF_F32S_CXX1(tanh)
#undef WOOF_F32S_CXX1
static inline f32s pow(f32s a, f32s b) { return powf(a, b); }
static inline f32s atan2(f32s a, f32s b) { return atan2f(a, b); }
static inline f32s fmod(f32s a, f32s b) { return fmodf(a, b); }
// hypotf / remainderf (prelude.h's float versions; remainder is exact).
static inline f32s hypotf(f32s a, f32s b) { return f32s(hypotf(a.v, b.v)); }
static inline f32s hypotf(f32s a, float b) { return hypotf(a, f32s(b)); }
static inline f32s hypotf(float a, f32s b) { return hypotf(f32s(a), b); }
static inline f32s remainderf(f32s a, f32s b) { return f32s(remainderf(a.v, b.v)); }
static inline f32s remainderf(f32s a, float b) { return remainderf(a, f32s(b)); }
static inline f32s remainderf(float a, f32s b) { return remainderf(f32s(a), b); }
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
// setp.<cmp>.f32 without .ftz: an ordered comparison that reads a subnormal
// as itself (Metal's compare reads it as zero); ne is false for a NaN.
static inline bool __woof_ieee_eq(f32s a, f32s b) { return a == b; }
static inline bool __woof_ieee_lt(f32s a, f32s b) { return a < b; }
static inline bool __woof_ieee_le(f32s a, f32s b) { return a <= b; }
static inline bool __woof_ieee_gt(f32s a, f32s b) { return a > b; }
static inline bool __woof_ieee_ge(f32s a, f32s b) { return a >= b; }
static inline bool __woof_ieee_ne(f32s a, f32s b) { return a < b || a > b; }
static inline bool __woof_ieee_eq(float a, float b) { return f32s(a) == f32s(b); }
static inline bool __woof_ieee_lt(float a, float b) { return f32s(a) < f32s(b); }
static inline bool __woof_ieee_le(float a, float b) { return f32s(a) <= f32s(b); }
static inline bool __woof_ieee_gt(float a, float b) { return f32s(a) > f32s(b); }
static inline bool __woof_ieee_ge(float a, float b) { return f32s(a) >= f32s(b); }
static inline bool __woof_ieee_ne(float a, float b) { return __woof_ieee_lt(a, b) || __woof_ieee_gt(a, b); }
static inline f32s __woof_selmax(f32s a, f32s b) {
  return a > b ? a : (a == b ? f32s(as_type<float>(as_type<uint>(a.v) & as_type<uint>(b.v))) : b);
}
static inline f32s __woof_selmin(f32s a, f32s b) {
  return a < b ? a : (a == b ? f32s(as_type<float>(as_type<uint>(a.v) | as_type<uint>(b.v))) : b);
}

#ifdef WOOF_RETAIN_SUBNORMALS
// In an --ftz=false unit float is f32s, so the bit casts return f32s too.
// Returning float, `f(__uint_as_float(u))` with f(int) and f(f32s)
// overloads took f(int): float -> int is a standard conversion and float ->
// f32s a user-defined one (lake's lake_abs probe got the integer value of
// the word: abs(+Inf) read 0x4f000000).  Defined last, for the unit's code.
#define __uint_as_float(a) (f32s::bits((uint)(a)))
#define __int_as_float(a) (f32s::bits((uint)(int)(a)))
#endif
