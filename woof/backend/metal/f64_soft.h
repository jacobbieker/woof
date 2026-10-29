// IEEE-754 binary64 in software, on 64-bit integers, bit-exact.
//
// Apple GPUs have no binary64 hardware.  woof's strict numerics need the
// words CUDA's double units produce, so translated kernels compute every
// `double` through this header: round-to-nearest-even, gradual underflow
// (binary64 subnormals are kept, as CUDA keeps them: -ftz applies to binary32
// only), and the same storage layout as a C double, so `double*` buffers stay
// ordinary binary64 memory.
//
// The algorithms follow Berkeley SoftFloat 3e (J. Hauser, BSD-3-Clause;
// licenses/softfloat-NOTICE.txt): roundPackToF64, addMags/subMags, mul,
// with division and square root done by exact digit recurrences.  The header
// compiles as Metal Shading Language and as host C++17 (the CPU grader in
// tools/f64_soft_grade.py builds it with clang++), so one text is graded on
// both.
//
// NaN results: an operation with a NaN operand returns the first NaN operand,
// quieted; an invalid operation returns the default NaN 0x7FF8000000000000.
#pragma once

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
typedef ulong f64_u64;
typedef long f64_i64;
typedef uint f64_u32;
typedef int f64_i32;
#define F64_FN static inline
// The operations themselves are called, not inlined: inlined at every use
// they make large fp64 units (CLM lake) take tens of minutes to compile.
#define F64_OP static __attribute__((noinline))
static inline int f64_clz64(f64_u64 x) { return int(metal::clz(x)); }
static inline f64_u64 f64_mulhi(f64_u64 a, f64_u64 b) { return metal::mulhi(a, b); }
static inline float f64_bits_to_f32(f64_u32 u) { return as_type<float>(u); }
static inline f64_u32 f64_f32_to_bits(float f) { return as_type<f64_u32>(f); }
#else
#include <cstdint>
#include <cstring>
typedef uint64_t f64_u64;
typedef int64_t f64_i64;
typedef uint32_t f64_u32;
typedef int32_t f64_i32;
#define F64_FN static inline
#define F64_OP static inline
static inline int f64_clz64(f64_u64 x) { return x ? __builtin_clzll(x) : 64; }
static inline f64_u64 f64_mulhi(f64_u64 a, f64_u64 b) {
  return (f64_u64)(((unsigned __int128)a * b) >> 64);
}
static inline float f64_bits_to_f32(f64_u32 u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline f64_u32 f64_f32_to_bits(float f) { f64_u32 u; std::memcpy(&u, &f, 4); return u; }
#endif

#define F64_DEFAULT_NAN 0x7FF8000000000000UL
#define F64_SIGN(u) ((u) >> 63)
#define F64_EXP(u) (f64_i32)(((u) >> 52) & 0x7FF)
#define F64_FRAC(u) ((u) & 0x000FFFFFFFFFFFFFUL)
#define F64_PACK(s, e, m) (((f64_u64)(s) << 63) + ((f64_u64)(e) << 52) + (m))
#define F64_IS_NAN(u) ((((u) >> 52) & 0x7FF) == 0x7FF && F64_FRAC(u))

// ---------------------------------------------------------------- helpers

F64_FN f64_u64 f64_shift_right_jam(f64_u64 a, f64_i32 dist) {
  return dist < 63 ? (a >> dist) | (f64_u64)((a << ((-dist) & 63)) != 0) : (f64_u64)(a != 0);
}

struct f64_u128 { f64_u64 hi, lo; };

F64_FN f64_u128 f64_mul64x64(f64_u64 a, f64_u64 b) {
  f64_u128 r; r.lo = a * b; r.hi = f64_mulhi(a, b); return r;
}
F64_FN f64_u128 f64_u128_shl(f64_u128 x, f64_i32 n) {
  f64_u128 r;
  if (n == 0) return x;
  if (n >= 128) { r.hi = 0; r.lo = 0; return r; }
  if (n >= 64) { r.hi = x.lo << (n - 64); r.lo = 0; return r; }
  r.hi = (x.hi << n) | (x.lo >> (64 - n)); r.lo = x.lo << n; return r;
}
// Shift right, OR-ing every bit shifted out into bit 0.
F64_FN f64_u128 f64_u128_shr_jam(f64_u128 x, f64_i32 n) {
  f64_u128 r;
  if (n == 0) return x;
  if (n >= 128) { r.hi = 0; r.lo = (f64_u64)((x.hi | x.lo) != 0); return r; }
  if (n >= 64) {
    f64_u64 lost = (n == 64) ? x.lo : (x.lo | (x.hi << (128 - n)));
    r.hi = 0; r.lo = (n == 64 ? x.hi : x.hi >> (n - 64)) | (f64_u64)(lost != 0); return r;
  }
  f64_u64 lost = x.lo << (64 - n);
  r.hi = x.hi >> n; r.lo = (x.lo >> n) | (x.hi << (64 - n)) | (f64_u64)(lost != 0); return r;
}
F64_FN f64_u128 f64_u128_add(f64_u128 a, f64_u128 b) {
  f64_u128 r; r.lo = a.lo + b.lo; r.hi = a.hi + b.hi + (f64_u64)(r.lo < a.lo); return r;
}
F64_FN f64_u128 f64_u128_sub(f64_u128 a, f64_u128 b) {
  f64_u128 r; r.lo = a.lo - b.lo; r.hi = a.hi - b.hi - (f64_u64)(a.lo < b.lo); return r;
}
F64_FN bool f64_u128_lt(f64_u128 a, f64_u128 b) {
  return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
F64_FN f64_i32 f64_u128_msb(f64_u128 x) {  // -1 for zero
  return x.hi ? 127 - f64_clz64(x.hi) : (x.lo ? 63 - f64_clz64(x.lo) : -1);
}

F64_FN f64_u64 f64_propagate_nan(f64_u64 a, f64_u64 b) {
  return (F64_IS_NAN(a) ? a : b) | 0x0008000000000000UL;
}

// sig: hidden bit at bit 62, ten rounding bits below bit 10.  exp is the
// biased exponent minus one (the hidden bit carries into the field).
F64_FN f64_u64 f64_round_pack(f64_u64 sign, f64_i32 exp, f64_u64 sig) {
  f64_u64 round_bits = sig & 0x3FF;
  if (0x7FD <= (f64_u32)exp) {
    if (exp < 0) {
      sig = f64_shift_right_jam(sig, -exp);
      exp = 0;
      round_bits = sig & 0x3FF;
    } else if (0x7FD < exp || 0x8000000000000000UL <= sig + 0x200) {
      return F64_PACK(sign, 0x7FF, 0);
    }
  }
  sig = (sig + 0x200) >> 10;
  sig &= ~(f64_u64)((round_bits ^ 0x200) == 0);
  if (!sig) exp = 0;
  return F64_PACK(sign, exp, sig);
}

F64_FN f64_u64 f64_norm_round_pack(f64_u64 sign, f64_i32 exp, f64_u64 sig) {
  f64_i32 shift = f64_clz64(sig) - 1;
  exp -= shift;
  if (10 <= shift && (f64_u32)exp < 0x7FD) {
    return F64_PACK(sign, sig ? exp : 0, sig << (shift - 10));
  }
  return f64_round_pack(sign, exp, sig << shift);
}

// Normalise a subnormal significand: returns the shift; exp = 1 - shift.
F64_FN f64_i32 f64_norm_subnormal_shift(f64_u64 frac) { return f64_clz64(frac) - 11; }

// ------------------------------------------------------------ arithmetic

F64_OP f64_u64 f64_add_mags(f64_u64 a, f64_u64 b, f64_u64 sign) {
  f64_i32 ea = F64_EXP(a), eb = F64_EXP(b);
  f64_u64 ma = F64_FRAC(a), mb = F64_FRAC(b);
  f64_i32 diff = ea - eb, ez;
  f64_u64 mz;
  if (!diff) {
    if (!ea) return a + mb;
    if (ea == 0x7FF) return (ma | mb) ? f64_propagate_nan(a, b) : a;
    ez = ea;
    mz = (0x0020000000000000UL + ma + mb) << 9;
  } else {
    ma <<= 9; mb <<= 9;
    if (diff < 0) {
      if (eb == 0x7FF) return mb ? f64_propagate_nan(a, b) : F64_PACK(sign, 0x7FF, 0);
      ez = eb;
      ma = ea ? ma + 0x2000000000000000UL : ma << 1;
      ma = f64_shift_right_jam(ma, -diff);
    } else {
      if (ea == 0x7FF) return ma ? f64_propagate_nan(a, b) : a;
      ez = ea;
      mb = eb ? mb + 0x2000000000000000UL : mb << 1;
      mb = f64_shift_right_jam(mb, diff);
    }
    mz = 0x2000000000000000UL + ma + mb;
    if (mz < 0x4000000000000000UL) { --ez; mz <<= 1; }
  }
  return f64_round_pack(sign, ez, mz);
}

F64_OP f64_u64 f64_sub_mags(f64_u64 a, f64_u64 b, f64_u64 sign) {
  f64_i32 ea = F64_EXP(a), eb = F64_EXP(b);
  f64_u64 ma = F64_FRAC(a), mb = F64_FRAC(b);
  f64_i32 diff = ea - eb;
  if (!diff) {
    if (ea == 0x7FF) return (ma | mb) ? f64_propagate_nan(a, b) : F64_DEFAULT_NAN;
    f64_i64 d = (f64_i64)ma - (f64_i64)mb;
    if (!d) return F64_PACK(0, 0, 0);
    if (ea) --ea;
    if (d < 0) { sign ^= 1; d = -d; }
    f64_i32 shift = f64_clz64((f64_u64)d) - 11;
    f64_i32 ez = ea - shift;
    if (ez < 0) { shift = ea; ez = 0; }
    return F64_PACK(sign, ez, (f64_u64)d << shift);
  }
  ma <<= 10; mb <<= 10;
  f64_i32 ez;
  f64_u64 mz;
  if (diff < 0) {
    sign ^= 1;
    if (eb == 0x7FF) return mb ? f64_propagate_nan(a, b) : F64_PACK(sign, 0x7FF, 0);
    ma += ea ? 0x4000000000000000UL : ma;
    ma = f64_shift_right_jam(ma, -diff);
    mb |= 0x4000000000000000UL;
    ez = eb; mz = mb - ma;
  } else {
    if (ea == 0x7FF) return ma ? f64_propagate_nan(a, b) : a;
    mb += eb ? 0x4000000000000000UL : mb;
    mb = f64_shift_right_jam(mb, diff);
    ma |= 0x4000000000000000UL;
    ez = ea; mz = ma - mb;
  }
  return f64_norm_round_pack(sign, ez - 1, mz);
}

F64_FN f64_u64 f64_add(f64_u64 a, f64_u64 b) {
  return F64_SIGN(a) == F64_SIGN(b) ? f64_add_mags(a, b, F64_SIGN(a)) : f64_sub_mags(a, b, F64_SIGN(a));
}
F64_FN f64_u64 f64_sub(f64_u64 a, f64_u64 b) {
  return F64_SIGN(a) == F64_SIGN(b) ? f64_sub_mags(a, b, F64_SIGN(a)) : f64_add_mags(a, b, F64_SIGN(a));
}

F64_OP f64_u64 f64_mul(f64_u64 a, f64_u64 b) {
  f64_u64 sign = F64_SIGN(a) ^ F64_SIGN(b);
  f64_i32 ea = F64_EXP(a), eb = F64_EXP(b);
  f64_u64 ma = F64_FRAC(a), mb = F64_FRAC(b);
  if (ea == 0x7FF) {
    if (ma || (eb == 0x7FF && mb)) return f64_propagate_nan(a, b);
    return ((f64_u64)eb | mb) ? F64_PACK(sign, 0x7FF, 0) : F64_DEFAULT_NAN;
  }
  if (eb == 0x7FF) {
    if (mb) return f64_propagate_nan(a, b);
    return ((f64_u64)ea | ma) ? F64_PACK(sign, 0x7FF, 0) : F64_DEFAULT_NAN;
  }
  if (!ea) {
    if (!ma) return F64_PACK(sign, 0, 0);
    f64_i32 s = f64_norm_subnormal_shift(ma); ea = 1 - s; ma <<= s;
  }
  if (!eb) {
    if (!mb) return F64_PACK(sign, 0, 0);
    f64_i32 s = f64_norm_subnormal_shift(mb); eb = 1 - s; mb <<= s;
  }
  f64_i32 ez = ea + eb - 0x3FF;
  ma = (ma | 0x0010000000000000UL) << 10;
  mb = (mb | 0x0010000000000000UL) << 11;
  f64_u128 p = f64_mul64x64(ma, mb);
  f64_u64 mz = p.hi | (f64_u64)(p.lo != 0);
  if (mz < 0x4000000000000000UL) { --ez; mz <<= 1; }
  return f64_round_pack(sign, ez, mz);
}

F64_OP f64_u64 f64_div(f64_u64 a, f64_u64 b) {
  f64_u64 sign = F64_SIGN(a) ^ F64_SIGN(b);
  f64_i32 ea = F64_EXP(a), eb = F64_EXP(b);
  f64_u64 ma = F64_FRAC(a), mb = F64_FRAC(b);
  if (ea == 0x7FF) {
    if (ma) return f64_propagate_nan(a, b);
    if (eb == 0x7FF) return mb ? f64_propagate_nan(a, b) : F64_DEFAULT_NAN;
    return F64_PACK(sign, 0x7FF, 0);
  }
  if (eb == 0x7FF) return mb ? f64_propagate_nan(a, b) : F64_PACK(sign, 0, 0);
  if (!eb) {
    if (!mb) return ((f64_u64)ea | ma) ? F64_PACK(sign, 0x7FF, 0) : F64_DEFAULT_NAN;
    f64_i32 s = f64_norm_subnormal_shift(mb); eb = 1 - s; mb <<= s;
  }
  if (!ea) {
    if (!ma) return F64_PACK(sign, 0, 0);
    f64_i32 s = f64_norm_subnormal_shift(ma); ea = 1 - s; ma <<= s;
  }
  f64_i32 ez = ea - eb + 0x3FE;
  ma |= 0x0010000000000000UL;
  mb |= 0x0010000000000000UL;
  if (ma < mb) { --ez; ma <<= 1; }
  // Restoring division: 63 quotient bits, bit 62 the integer bit.
  f64_u64 rem = ma, q = 0;
  for (int i = 0; i < 63; ++i) {
    q <<= 1;
    if (rem >= mb) { rem -= mb; q |= 1; }
    rem <<= 1;
  }
  return f64_round_pack(sign, ez, q | (f64_u64)(rem != 0));
}

F64_OP f64_u64 f64_sqrt(f64_u64 a) {
  f64_u64 sign = F64_SIGN(a);
  f64_i32 ea = F64_EXP(a);
  f64_u64 ma = F64_FRAC(a);
  if (ea == 0x7FF) {
    if (ma) return f64_propagate_nan(a, 0);
    return sign ? F64_DEFAULT_NAN : a;
  }
  if (sign) return ((f64_u64)ea | ma) ? F64_DEFAULT_NAN : a;  // sqrt(-0) = -0
  if (!ea) {
    if (!ma) return a;
    f64_i32 s = f64_norm_subnormal_shift(ma); ea = 1 - s; ma <<= s;
  }
  f64_i32 e = ea - 0x3FF;
  f64_u64 m = ma | 0x0010000000000000UL;  // value m * 2^(e - 52)
  if (e & 1) { m <<= 1; e -= 1; }
  // Root of N = m * 2^72 = (m << 8) * 2^64, in [2^124, 2^126): 63 bits,
  // bit 62 the integer bit, by the digit-by-digit recurrence.
  f64_u128 n; n.hi = m << 8; n.lo = 0;
  f64_u128 rem; rem.hi = 0; rem.lo = 0;
  f64_u64 r = 0;
  for (int i = 63; i >= 0; --i) {
    f64_u64 pair = (i >= 32) ? (n.hi >> (2 * (i - 32))) & 3 : (n.lo >> (2 * i)) & 3;
    rem = f64_u128_shl(rem, 2);
    rem.lo |= pair;
    f64_u128 t; t.hi = r >> 62; t.lo = (r << 2) | 1;
    r <<= 1;
    if (!f64_u128_lt(rem, t)) { rem = f64_u128_sub(rem, t); r |= 1; }
  }
  f64_i32 ez = (e >> 1) + 0x3FE;
  return f64_round_pack(0, ez, r | (f64_u64)((rem.hi | rem.lo) != 0));
}

// Fused multiply-add, one rounding.
F64_OP f64_u64 f64_fma(f64_u64 a, f64_u64 b, f64_u64 c) {
  f64_u64 sp = F64_SIGN(a) ^ F64_SIGN(b), sc = F64_SIGN(c);
  f64_i32 ea = F64_EXP(a), eb = F64_EXP(b), ec = F64_EXP(c);
  f64_u64 ma = F64_FRAC(a), mb = F64_FRAC(b), mc = F64_FRAC(c);
  if (F64_IS_NAN(a) || F64_IS_NAN(b)) return f64_propagate_nan(a, b);
  bool a_zero = !ea && !ma, b_zero = !eb && !mb;
  if (ea == 0x7FF || eb == 0x7FF) {
    if (a_zero || b_zero) return F64_DEFAULT_NAN;  // inf * 0
    if (F64_IS_NAN(c)) return f64_propagate_nan(c, c);
    if (ec == 0x7FF && sc != sp) return F64_DEFAULT_NAN;  // inf - inf
    return F64_PACK(sp, 0x7FF, 0);
  }
  if (ec == 0x7FF) return mc ? f64_propagate_nan(c, c) : c;
  bool c_zero = !ec && !mc;
  if (a_zero || b_zero) {
    if (c_zero) return F64_PACK(sp & sc, 0, 0);
    return c;
  }
  if (!ea) { f64_i32 s = f64_norm_subnormal_shift(ma); ea = 1 - s; ma <<= s; }
  if (!eb) { f64_i32 s = f64_norm_subnormal_shift(mb); eb = 1 - s; mb <<= s; }
  ma |= 0x0010000000000000UL;
  mb |= 0x0010000000000000UL;
  // Product: value P * 2^(ea + eb - 2*0x3FF - 104).
  f64_u128 p = f64_mul64x64(ma, mb);
  f64_i32 msb_p = f64_u128_msb(p);
  f64_i32 sh_p = 125 - msb_p;
  p = f64_u128_shl(p, sh_p);
  f64_i32 Ep = ea + eb - 2 * 0x3FF - 104 - sh_p;  // scale of p's bit 0
  if (c_zero) {
    // Exact product, rounded once.
    f64_i32 msb = 125;
    f64_u128 s = f64_u128_shr_jam(p, msb - 62);
    return f64_round_pack(sp, Ep + msb + 0x3FF - 1, s.lo);
  }
  if (!ec) { f64_i32 s = f64_norm_subnormal_shift(mc); ec = 1 - s; mc <<= s; }
  mc |= 0x0010000000000000UL;
  f64_u128 q; q.hi = 0; q.lo = mc;
  q = f64_u128_shl(q, 125 - 52);
  f64_i32 Ec = ec - 0x3FF - 52 - (125 - 52);
  f64_i32 E;
  if (Ep >= Ec) { q = f64_u128_shr_jam(q, Ep - Ec); E = Ep; }
  else { p = f64_u128_shr_jam(p, Ec - Ep); E = Ec; }
  f64_u128 r;
  f64_u64 sign;
  if (sp == sc) {
    r = f64_u128_add(p, q); sign = sp;
  } else if (f64_u128_lt(p, q)) {
    r = f64_u128_sub(q, p); sign = sc;
  } else {
    r = f64_u128_sub(p, q); sign = sp;
  }
  f64_i32 msb = f64_u128_msb(r);
  if (msb < 0) return F64_PACK(0, 0, 0);  // exact cancellation: +0 under RNE
  f64_u64 sig;
  if (msb > 62) sig = f64_u128_shr_jam(r, msb - 62).lo;
  else sig = r.lo << (62 - msb);
  return f64_round_pack(sign, E + msb + 0x3FF - 1, sig);
}

// ------------------------------------------------------------ conversions

F64_FN f64_u32 f32_round_pack(f64_u32 sign, f64_i32 exp, f64_u32 sig) {
  // sig: hidden bit at bit 30, seven rounding bits.
  f64_u32 round_bits = sig & 0x7F;
  if (0xFD <= (f64_u32)exp) {
    if (exp < 0) {
      sig = (-exp < 31) ? (sig >> (-exp)) | (f64_u32)((sig << ((exp) & 31)) != 0) : (f64_u32)(sig != 0);
      exp = 0;
      round_bits = sig & 0x7F;
    } else if (0xFD < exp || 0x80000000u <= sig + 0x40) {
      return (sign << 31) | 0x7F800000u;
    }
  }
  sig = (sig + 0x40) >> 7;
  sig &= ~(f64_u32)((round_bits ^ 0x40) == 0);
  if (!sig) exp = 0;
  return (sign << 31) + ((f64_u32)exp << 23) + sig;
}

F64_FN float f64_to_f32(f64_u64 a) {
  f64_u32 sign = (f64_u32)F64_SIGN(a);
  f64_i32 e = F64_EXP(a);
  f64_u64 m = F64_FRAC(a);
  if (e == 0x7FF) {
    if (m) return f64_bits_to_f32((sign << 31) | 0x7FC00000u | (f64_u32)(m >> 29));
    return f64_bits_to_f32((sign << 31) | 0x7F800000u);
  }
  f64_u32 m32 = (f64_u32)f64_shift_right_jam(m, 22);
  if (!(e | (f64_i32)m32)) return f64_bits_to_f32(sign << 31);
  return f64_bits_to_f32(f32_round_pack(sign, e - 0x381, m32 | 0x40000000u));
}

F64_FN f64_u64 f64_from_f32(float f) {
  f64_u32 u = f64_f32_to_bits(f);
  f64_u64 sign = u >> 31;
  f64_i32 e = (f64_i32)((u >> 23) & 0xFF);
  f64_u64 m = u & 0x7FFFFF;
  if (e == 0xFF) {
    if (m) return F64_PACK(sign, 0x7FF, (m << 29) | 0x0008000000000000UL);
    return F64_PACK(sign, 0x7FF, 0);
  }
  if (!e) {
    if (!m) return F64_PACK(sign, 0, 0);
    f64_i32 s = f64_clz64(m) - 40;  // move the leading one to bit 23
    e = 1 - s;
    m = (m << s) & 0x7FFFFF;
  }
  return F64_PACK(sign, e + 0x380, m << 29);
}

// mag >= 2^53: round to 53 bits.  normRoundPack needs bit 63 clear; only
// then is it safe to halve with a sticky bit (eleven bits fall below the
// rounding point, so the jammed bit 0 is never the half bit).
F64_FN f64_u64 f64_from_u64_rounded(f64_u64 sign, f64_u64 mag) {
  if (mag >> 63) return f64_norm_round_pack(sign, 0x43D, (mag >> 1) | (mag & 1));
  return f64_norm_round_pack(sign, 0x43C, mag);
}

F64_FN f64_u64 f64_from_i64(f64_i64 v) {
  f64_u64 sign = v < 0;
  f64_u64 mag = sign ? (f64_u64)0 - (f64_u64)v : (f64_u64)v;
  if (!mag) return 0;
  if ((mag >> 53) == 0) {
    f64_i32 shift = f64_clz64(mag) - 11;
    return F64_PACK(sign, 0x432 - shift, mag << shift);  // hidden bit carries into the field
  }
  return f64_from_u64_rounded(sign, mag);
}
F64_FN f64_u64 f64_from_u64(f64_u64 mag) {
  if (!mag) return 0;
  if ((mag >> 53) == 0) {
    f64_i32 shift = f64_clz64(mag) - 11;
    return F64_PACK(0, 0x432 - shift, mag << shift);  // hidden bit carries into the field
  }
  return f64_from_u64_rounded(0, mag);
}
F64_FN f64_u64 f64_from_i32(f64_i32 v) { return f64_from_i64((f64_i64)v); }

// Truncation toward zero, saturating, NaN -> 0 (PTX cvt.rzi semantics).
F64_FN f64_i64 f64_to_i64(f64_u64 a) {
  f64_i32 e = F64_EXP(a);
  f64_u64 m = F64_FRAC(a);
  if (e == 0x7FF && m) return 0;
  f64_u64 sign = F64_SIGN(a);
  if (e < 0x3FF) return 0;
  f64_i32 shift = e - 0x433;
  if (shift >= 11) return sign ? (f64_i64)0x8000000000000000UL : (f64_i64)0x7FFFFFFFFFFFFFFFUL;
  f64_u64 sig = m | 0x0010000000000000UL;
  f64_u64 mag = shift >= 0 ? sig << shift : sig >> (-shift);
  if (!sign && mag > 0x7FFFFFFFFFFFFFFFUL) return (f64_i64)0x7FFFFFFFFFFFFFFFUL;
  return sign ? (f64_i64)((f64_u64)0 - mag) : (f64_i64)mag;
}
F64_FN f64_i32 f64_to_i32(f64_u64 a) {
  f64_i32 e = F64_EXP(a);
  if (e == 0x7FF && F64_FRAC(a)) return 0;
  f64_i64 v = f64_to_i64(a);
  if (v > 2147483647L) return 2147483647;
  if (v < -2147483647L - 1) return -2147483647 - 1;
  return (f64_i32)v;
}

// ------------------------------------------------------------ comparisons

F64_FN bool f64_eq(f64_u64 a, f64_u64 b) {
  if (F64_IS_NAN(a) || F64_IS_NAN(b)) return false;
  return a == b || ((a | b) << 1) == 0;
}
F64_FN bool f64_lt(f64_u64 a, f64_u64 b) {
  if (F64_IS_NAN(a) || F64_IS_NAN(b)) return false;
  f64_u64 sa = F64_SIGN(a), sb = F64_SIGN(b);
  if (sa != sb) return sa && ((a | b) << 1) != 0;
  return a != b && (sa ^ (a < b));
}
F64_FN bool f64_le(f64_u64 a, f64_u64 b) {
  if (F64_IS_NAN(a) || F64_IS_NAN(b)) return false;
  f64_u64 sa = F64_SIGN(a), sb = F64_SIGN(b);
  if (sa != sb) return sa || ((a | b) << 1) == 0;
  return a == b || (sa ^ (a < b));
}
