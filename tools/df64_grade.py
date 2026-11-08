"""Grade woof/backend/metal/f64_df64.h: the fast numerics' df64 binary64.

Three legs, one header text:

``host``  builds f64_soft.h + f64_df64.h as C++17 and grades every operation
          against an extended-precision reference (x87 ``long double``, 64
          significant bits, so a host with it: x86-64), over random words,
          per-function domains and edge cases.  It reports, per operation,
          the largest error in binary64 ulps of the reference (an ulp of the
          correctly rounded result) and the word that gave it.  Division and
          square root must be the correctly rounded words (0 mismatches);
          every other operation must stay within its stated bound
          (``BOUNDS``).  Binary32 runs flush-to-zero, as on the GPU.
``gpu``   runs the same operations as Metal through woof.backend.metal.ext
          and compares the GPU's words with the host build's words for the
          same operands (any host: the comparison is word for word, so an
          Apple-silicon host without long double serves).
``bench`` times soft-float against df64 per operation on the GPU.

Usage::

    pixi run -e default python tools/df64_grade.py host --samples 2000000
    pixi run -e mlx python tools/df64_grade.py gpu --samples 1048576
    pixi run -e mlx python tools/df64_grade.py bench
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METAL = ROOT / "woof" / "backend" / "metal"

# (name, arity, word expression, kind): kind "exact" = the correctly rounded
# word; "ulp" = graded in ulps against the reference
OPS = [
    ("div", 2, "f64_df_div(a, b)", "exact"),
    ("sqrt", 1, "f64_df_sqrt(a)", "exact"),
    ("add_df", 2, "f64_df_add(a, b)", "ulp"),
    ("sub_df", 2, "f64_df_sub(a, b)", "ulp"),
    ("mul_df", 2, "f64_df_mul(a, b)", "ulp"),
    ("fma_df", 3, "f64_df_fma(a, b, c)", "ulp"),
    ("exp", 1, "df_exp_word(a)", "ulp"),
    ("expm1", 1, "df_expm1_word(a)", "ulp"),
    ("log", 1, "df_log_word(a)", "ulp"),
    ("log2", 1, "df_log2_word(a)", "ulp"),
    ("log10", 1, "df_log10_word(a)", "ulp"),
    ("log1p", 1, "df_log1p_word(a)", "ulp"),
    ("pow", 2, "df_pow_word(a, b)", "ulp"),
    ("sin", 1, "df_sin_word(a)", "ulp"),
    ("cos", 1, "df_cos_word(a)", "ulp"),
    ("tan", 1, "df_tan_word(a)", "ulp"),
    ("atan", 1, "df_atan_word(a)", "ulp"),
    ("atan2", 2, "df_atan2_word(a, b)", "ulp"),
    ("tanh", 1, "df_tanh_word(a)", "ulp"),
    ("hypot", 2, "df_hypot_word(a, b)", "ulp"),
    ("cbrt", 1, "df_cbrt_word(a)", "ulp"),
]

# Max error, binary64 ulps of the reference, that the host leg accepts.
# The df64 core carries ~48 bits (2^-48 relative = 16 binary64 ulps); the
# bounds leave a factor of about two over the measured maxima.  pow's error
# grows with |y log x| (its argument to exp carries the df64 error times
# that magnitude): its row is the error divided by 1 + |y log x|.
BOUNDS = {
    "add_df": 64, "sub_df": 64, "mul_df": 256, "fma_df": 256,
    "exp": 256, "expm1": 256, "log": 256, "log2": 256, "log10": 256, "log1p": 256,
    "pow": 256, "sin": 256, "cos": 256, "tan": 256, "atan": 256, "atan2": 256,
    "tanh": 256, "hypot": 256, "cbrt": 256,
    # pure df64 core (df64 operands, the df64 result against the exact one)
    "df_add": 64, "df_sub": 64, "df_mul": 128, "df_div": 128, "df_sqrt": 128, "df_fma": 128,
    "df_from_i64": 16, "df_to_f32": 0, "df_from_f32": 0, "df_to_i64": 0, "df_to_i32": 0,
    "df_compare": 0,
}

HARNESS = r'''
#include "f64_soft.h"
#include "f64_df64.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(__x86_64__)
#include <xmmintrin.h>
static void ftz(bool on) { _mm_setcsr(on ? (_mm_getcsr() | 0x8040) : (_mm_getcsr() & ~0x8040u)); }
#else
static void ftz(bool) {}
#endif

static inline uint64_t bits(double d) { uint64_t u; std::memcpy(&u, &d, 8); return u; }
static inline double dbl(uint64_t u) { double d; std::memcpy(&d, &u, 8); return d; }

static uint64_t apply(int op, uint64_t a, uint64_t b, uint64_t c) {
  switch (op) {
OPS_CASES
  }
  return 0;
}
'''

HOST_MAIN = r'''
typedef long double LD;
static uint64_t s[2];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t next() {
  uint64_t s0 = s[0], s1 = s[1], r = s0 + s1;
  s1 ^= s0; s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); s[1] = rotl(s1, 36);
  return r;
}
static double unif() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
static double range(double lo, double hi) { return lo + (hi - lo) * unif(); }
static uint64_t anyword() { return next(); }
static const uint64_t SPECIAL[] = {
  0x0000000000000000ULL, 0x8000000000000000ULL, 0x7FF0000000000000ULL, 0xFFF0000000000000ULL,
  0x7FF8000000000000ULL, 0x3FF0000000000000ULL, 0xBFF0000000000000ULL, 0x0000000000000001ULL,
  0x000FFFFFFFFFFFFFULL, 0x0010000000000000ULL, 0x7FEFFFFFFFFFFFFFULL, 0x3FEFFFFFFFFFFFFFULL,
  0x3FF0000000000001ULL, 0x4000000000000000ULL, 0x3FE0000000000000ULL, 0x4340000000000000ULL};
static const int NSPECIAL = sizeof(SPECIAL) / sizeof(SPECIAL[0]);
static uint64_t modest() {     // exponents within 2^+-40, either sign
  uint64_t e = 0x3FF - 40 + next() % 80;
  return (e << 52) | (next() & 0x800FFFFFFFFFFFFFULL);
}
static uint64_t pick(int op, int which) {
  uint64_t r = next();
  if ((r & 15) == 0) return SPECIAL[(r >> 4) % NSPECIAL];
  int k = (int)((r >> 8) & 3);
  switch (op) {
OPS_DOMAINS
  }
  return anyword();
}

static LD ref(int op, LD x, LD y, LD z) {
  switch (op) {
OPS_REFS
  }
  return 0;
}

static double ulp_err(uint64_t got, LD want) {
  double g = dbl(got);
  if (std::isnan((double)want)) return std::isnan(g) ? 0 : 1e300;
  if (std::isnan(g)) return 1e300;
  LD aw = fabsl(want);
  const LD overflow = ldexpl(1.0L, 1024) - ldexpl(1.0L, 970);   // DBL_MAX + half an ulp
  if (aw >= overflow) {
    if (std::isinf(g) && (g > 0) == (want > 0)) return 0;
    // DBL_MAX for an exact result that overflows by less than df64's error:
    // the saturation f64_df64.h chooses over an infinity it cannot prove
    if (fabs(g) == 1.7976931348623157e308 && aw < overflow * (1 + 0x1p-30L)) return 0;
    return 1e300;
  }
  if (std::isinf(g)) return 1e300;
  int e; frexpl(aw, &e);            // aw = f 2^e, f in [0.5, 1)
  int ue = e - 53; if (ue < -1074) ue = -1074;
  return (double)(fabsl((LD)g - want) / ldexpl(1.0L, ue));
}

int main(int argc, char** argv) {
  long long n = atoll(argv[1]);
  s[0] = strtoull(argv[2], 0, 10) | 1; s[1] = 0x9E3779B97F4A7C15ULL;
  const int NOPS = OPS_COUNT;
  static const char* names[] = {OPS_NAMES};
  static const int exact[] = {OPS_EXACT};
  printf("{");
  for (int op = 0; op < NOPS; ++op) {
    double worst = 0, sum = 0; long long bad = 0, cnt = 0;
    uint64_t wa = 0, wb = 0, wc = 0, wg = 0;
    for (long long i = 0; i < n; ++i) {
      uint64_t a = pick(op, 0), b = pick(op, 1), c = pick(op, 2);
      if (i < NSPECIAL * NSPECIAL) { a = SPECIAL[i % NSPECIAL]; b = SPECIAL[i / NSPECIAL]; }
      ftz(true);
      uint64_t g = apply(op, a, b, c);
      ftz(false);
      double err;
      if (exact[op]) {
        uint64_t w = op == 0 ? bits(dbl(a) / dbl(b)) : bits(std::sqrt(dbl(a)));
        err = (g == w || (std::isnan(dbl(g)) && std::isnan(dbl(w)))) ? 0 : 1e300;
      } else {
        LD r = ref(op, (LD)dbl(a), (LD)dbl(b), (LD)dbl(c));
        err = ulp_err(g, r);
        // + - fma through df64 lose the operands' low bits (2^-48 of each), so
        // they are graded in ulps of the largest operand (or product)
        if (op == ADD_OP || op == SUB_OP || op == FMA_OP) {
          LD big = op == FMA_OP ? fmaxl(fabsl((LD)dbl(a) * (LD)dbl(b)), fabsl((LD)dbl(c)))
                                : fmaxl(fabsl((LD)dbl(a)), fabsl((LD)dbl(b)));
          if (std::isfinite((double)big) && big > 0 && std::isfinite(dbl(g)) && std::isfinite((double)r))
            { int be; frexpl(big, &be); be = be - 53 < -1074 ? -1074 : be - 53;
              err = (double)(fabsl((LD)dbl(g) - r) / ldexpl(1.0L, be)); }
        }
        // pow: the error scales with |y log|x||, the argument handed to exp
        if (op == POW_OP && err < 1e300 && std::isfinite(dbl(a)) && std::isfinite(dbl(b)) && dbl(a) != 0)
          err /= (double)(1 + fabsl((LD)dbl(b) * logl(fabsl((LD)dbl(a)))));
      }
      cnt++;
      if (err > 0) { sum += err < 1e300 ? err : 0; }
      if (exact[op] && err > 0) bad++;
      if (err > worst) { worst = err; wa = a; wb = b; wc = c; wg = g; }
    }
    printf("%s\"%s\":{\"n\":%lld,\"max_ulp\":%.3f,\"mean_ulp\":%.4f,\"mismatch\":%lld,"
           "\"worst\":[\"%016llx\",\"%016llx\",\"%016llx\",\"%016llx\"]}", op ? "," : "", names[op], cnt,
           worst, sum / cnt, bad, (unsigned long long)wa, (unsigned long long)wb, (unsigned long long)wc,
           (unsigned long long)wg);
    fflush(stdout);
  }
  // the pure df64 core: df64 operands (hi, lo) from random binary64 values
  struct G { const char* name; double worst; long long n; };
  G g[] = {{"df_add", 0, 0}, {"df_sub", 0, 0}, {"df_mul", 0, 0}, {"df_div", 0, 0}, {"df_sqrt", 0, 0},
           {"df_fma", 0, 0}, {"df_from_i64", 0, 0}, {"df_to_f32", 0, 0}, {"df_from_f32", 0, 0},
           {"df_to_i64", 0, 0}, {"df_to_i32", 0, 0}, {"df_compare", 0, 0}};
  auto mk = [](double v) { float h = (float)v; return df_make(h, (float)(v - (double)h)); };
  auto val = [](df64 d) { return (LD)d.hi + (LD)d.lo; };
  auto rel = [](LD got, LD want) -> double {
    if (want == 0) return got == 0 ? 0 : 1e300;
    int e; frexpl(fabsl(want), &e);
    return (double)(fabsl(got - want) / ldexpl(1.0L, e - 53));
  };
  for (long long i = 0; i < n; ++i) {
    // operands within 2^+-30, so every partial product's low part stays a
    // normal binary32 (the libm keeps its df64 values in [2^-60, 2^60])
    double x = range(-1, 1) * ldexp(1.0, (int)(next() % 60) - 30);
    double y = range(-1, 1) * ldexp(1.0, (int)(next() % 60) - 30);
    double z = range(-1, 1) * ldexp(1.0, (int)(next() % 60) - 30);
    if (i & 1) y = x * (1 + range(-1e-6, 1e-6));          // near cancellation
    df64 a = mk(x), b = mk(y), c = mk(z);
    LD A = val(a), B = val(b), C = val(c);
    ftz(true);
    df64 ra = df_add(a, b), rs = df_sub(a, b), rm = df_mul(a, b), rd = df_div(a, b);
    df64 rq = df_sqrt(df_make(fabsf(a.hi), a.hi < 0 ? -a.lo : a.lo)), rf = df_fma(a, b, c);
    ftz(false);
    double e;
    e = rel(val(ra), A + B); if (e > g[0].worst) g[0].worst = e;
    e = rel(val(rs), A - B); if (e > g[1].worst) g[1].worst = e;
    e = rel(val(rm), A * B); if (e > g[2].worst) g[2].worst = e;
    e = rel(val(rd), A / B); if (e > g[3].worst) g[3].worst = e;
    e = rel(val(rq), sqrtl(fabsl(A))); if (e > g[4].worst) g[4].worst = e;
    LD F = A * B + C, FB = fmaxl(fabsl(A * B), fabsl(C));   // graded against the larger term
    e = rel(FB + (val(rf) - F), FB); if (e > g[5].worst) g[5].worst = e;
    int64_t v = (int64_t)next(); if (i & 1) v >>= (next() % 64);
    ftz(true); df64 dv = df_from_i64(v); ftz(false);
    e = rel(val(dv), (LD)v); if (e > g[6].worst) g[6].worst = e;
    ftz(true); float f = df_to_f32(a); ftz(false);
    if (f != (float)A && fabsl(A) > 1.2e-38L) g[7].worst = 1e300;
    float ff = (float)x;
    ftz(true); df64 dff = df_from_f32(ff); ftz(false);
    if (val(dff) != (LD)ff) g[8].worst = 1e300;
    double big = range(-1, 1) * ldexp(1.0, (int)(next() % 62));
    df64 db = mk(big);
    LD BV = val(db);
    ftz(true); int64_t ti = df_to_i64(db); int32_t t32 = df_to_i32(db); ftz(false);
    if (ti != (int64_t)BV) g[9].worst = 1e300;
    LD c32 = BV > 2147483647.0L ? 2147483647.0L : BV < -2147483648.0L ? -2147483648.0L : BV;
    if (t32 != (int32_t)c32) g[10].worst = 1e300;
    ftz(true); bool lt = df_lt(a, b), le = df_le(a, b), eq = df_eq(a, b); ftz(false);
    if (lt != (A < B) || le != (A <= B) || eq != (A == B)) g[11].worst = 1e300;
  }
  for (int k = 0; k < 12; ++k)
    printf(",\"%s\":{\"n\":%lld,\"max_ulp\":%.3f}", g[k].name, n, g[k].worst);
  printf("}\n");
  return 0;
}
'''

# operand domains per op: C statements returning a word, by `which` operand
DOMAINS = {
    "div": "return k == 0 ? anyword() : modest();",
    "sqrt": "return k == 0 ? (anyword() & 0x7FFFFFFFFFFFFFFFULL) : (modest() & 0x7FFFFFFFFFFFFFFFULL);",
    "add_df": "return k == 0 ? anyword() : modest();",
    "sub_df": "return k == 0 ? anyword() : modest();",
    "mul_df": "return k == 0 ? anyword() : modest();",
    "fma_df": "return modest();",
    "exp": "return bits(k == 0 ? range(-746, 710) : k == 1 ? range(-2, 2) : range(-1e-10, 1e-10));",
    "expm1": "return bits(k == 0 ? range(-50, 710) : k == 1 ? range(-0.4, 0.4) : range(-1e-12, 1e-12));",
    "log": "return k == 0 ? (anyword() & 0x7FFFFFFFFFFFFFFFULL) : bits(range(0.5, 2.0));",
    "log2": "return k == 0 ? (anyword() & 0x7FFFFFFFFFFFFFFFULL) : bits(range(0.5, 2.0));",
    "log10": "return k == 0 ? (anyword() & 0x7FFFFFFFFFFFFFFFULL) : bits(range(0.5, 2.0));",
    "log1p": "return bits(k == 0 ? range(-1, 1) : k == 1 ? range(-1e-10, 1e-10) : ldexp(unif(), (int)(next() % 100)));",
    "pow": ("if (which == 0) return bits(k == 0 ? range(0, 4) : k == 1 ? range(0.9, 1.1) : range(0, 1e6));\n"
            "      return bits(k == 0 ? range(-40, 40) : k == 1 ? (double)(int)range(-20, 20) : range(-3, 3));"),
    "sin": "return bits(k == 0 ? range(-10, 10) : k == 1 ? range(-1e6, 1e6) : k == 2 ? (double)((int64_t)range(-1e5, 1e5)) * 1.5707963267948966 : dbl(anyword()));",
    "cos": "return bits(k == 0 ? range(-10, 10) : k == 1 ? range(-1e6, 1e6) : k == 2 ? (double)((int64_t)range(-1e5, 1e5)) * 1.5707963267948966 : dbl(anyword()));",
    "tan": "return bits(k == 0 ? range(-10, 10) : k == 1 ? range(-1e6, 1e6) : k == 2 ? range(-1.6, 1.6) : dbl(anyword()));",
    "atan": "return k == 0 ? anyword() : bits(range(-4, 4));",
    "atan2": "return k == 0 ? anyword() : bits(range(-4, 4));",
    "tanh": "return bits(k == 0 ? range(-25, 25) : range(-1, 1));",
    "hypot": "return k == 0 ? anyword() : modest();",
    "cbrt": "return k == 0 ? anyword() : bits(range(-10, 10));",
}

REFS = {
    "add_df": "return x + y;", "sub_df": "return x - y;", "mul_df": "return x * y;",
    "fma_df": "return fmal(x, y, z);",
    "exp": "return expl(x);", "expm1": "return expm1l(x);", "log": "return logl(x);",
    "log2": "return log2l(x);", "log10": "return log10l(x);", "log1p": "return log1pl(x);",
    "pow": "return powl(x, y);", "sin": "return sinl(x);", "cos": "return cosl(x);",
    "tan": "return tanl(x);", "atan": "return atanl(x);", "atan2": "return atan2l(x, y);",
    "tanh": "return tanhl(x);", "hypot": "return hypotl(x, y);", "cbrt": "return cbrtl(x);",
}


def _source(main: str) -> str:
    cases = "\n".join(f"    case {i}: return {expr};" for i, (_, _, expr, _) in enumerate(OPS))
    domains = "\n".join(f"    case {i}: {DOMAINS[name]}" for i, (name, *_ ) in enumerate(OPS))
    refs = "\n".join(f"    case {i}: {REFS[name]}" for i, (name, *_ ) in enumerate(OPS) if name in REFS)
    text = HARNESS.replace("OPS_CASES", cases) + main
    return (text.replace("OPS_DOMAINS", domains).replace("OPS_REFS", refs)
            .replace("OPS_COUNT", str(len(OPS)))
            .replace("ADD_OP", str([n for n, *_ in OPS].index("add_df")))
            .replace("SUB_OP", str([n for n, *_ in OPS].index("sub_df")))
            .replace("FMA_OP", str([n for n, *_ in OPS].index("fma_df")))
            .replace("POW_OP", str([n for n, *_ in OPS].index("pow")))
            .replace("OPS_NAMES", ", ".join(f'"{n}"' for n, *_ in OPS))
            .replace("OPS_EXACT", ", ".join("1" if k == "exact" else "0" for *_, k in OPS)))


def _build(source: str, tmp: Path) -> Path:
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
    src, exe = tmp / "df64.cpp", tmp / "df64"
    src.write_text(source)
    flags = ["-O2", "-std=c++17", "-ffp-contract=off", "-fno-unsafe-math-optimizations"]
    if platform.machine() in ("x86_64", "AMD64"):
        flags.append("-mfma")
    subprocess.run([cxx, *flags, f"-I{METAL}", str(src), "-o", str(exe)], check=True)
    return exe


def host(samples: int, seed: int) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(_source(HOST_MAIN), Path(tmp))
        out = subprocess.run([str(exe), str(samples), str(seed)], capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)
    status = 0
    for name, row in result.items():
        bound = 0 if row.get("mismatch") is not None and name in ("div", "sqrt") else BOUNDS.get(name)
        ok = (row.get("mismatch", 0) == 0) if name in ("div", "sqrt") else row["max_ulp"] <= bound
        row["bound"] = bound
        row["ok"] = bool(ok)
        status |= int(not ok)
        print(json.dumps({name: row}))
    return status


DUMP_MAIN = r'''
int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "rb");
  long long n = atoll(argv[2]);
  uint64_t* in = (uint64_t*)malloc(n * 3 * 8);
  fread(in, 8, n * 3, f); fclose(f);
  FILE* o = fopen(argv[3], "wb");
  ftz(true);
  for (int op = 0; op < OPS_COUNT; ++op)
    for (long long i = 0; i < n; ++i) {
      uint64_t g = apply(op, in[3 * i], in[3 * i + 1], in[3 * i + 2]);
      fwrite(&g, 8, 1, o);
    }
  fclose(o);
  return 0;
}
'''

GPU_KERNEL = r'''
using namespace metal;
[[kernel]] void grade(const device ulong* in [[buffer(0)]], device ulong* out [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]], uint3 bdim [[threads_per_threadgroup]]) {
  uint i = bid.x * bdim.x + tid.x;
  if (i >= n) return;
  ulong a = in[3 * i], b = in[3 * i + 1], c = in[3 * i + 2];
OPS_STORES
}
'''


def _metal_text(kernel: str, defines: str = "") -> str:
    return ("#pragma METAL fp contract(off)\n#include <metal_stdlib>\n" + defines
            + (METAL / "f64_soft.h").read_text() + "\n" + (METAL / "f64_df64.h").read_text() + "\n" + kernel)


def _operands(rng, n):
    """Per op a third of random words, the rest each op's typical domain
    (mixed: the GPU leg checks words, not accuracy)."""
    w = rng.integers(0, 2**63, size=(n, 3), dtype=np.uint64) * np.uint64(2)
    k = n // 3
    w[k:2 * k] = rng.uniform(-50, 50, size=(k, 3)).view(np.uint64)
    w[2 * k:] = rng.uniform(0, 4, size=(n - 2 * k, 3)).view(np.uint64)
    return w


def gpu(samples: int, seed: int) -> int:
    import mlx.core as mx
    from woof.backend.metal import ext
    launch = ext.load().launch
    rng = np.random.default_rng(seed)
    operands = _operands(rng, samples)
    stores = "\n".join(f"  out[{i}u * n + i] = {expr};" for i, (_, _, expr, _) in enumerate(OPS))
    source = _metal_text(GPU_KERNEL.replace("OPS_STORES", stores))
    out = mx.zeros((len(OPS) * samples,), dtype=mx.uint64)
    (res,) = launch("woof-df64-grade", source, "grade", ((samples + 255) // 256, 1, 1), (256, 1, 1),
                    [mx.array(operands.reshape(-1)), out], [0, 0], [False, True], [0, 1], -1,
                    [struct.pack("I", samples)], [2], 0, 0)
    got = np.array(res).reshape(len(OPS), samples)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        exe = _build(_source(DUMP_MAIN), tmp)
        (tmp / "in.bin").write_bytes(operands.tobytes())
        subprocess.run([str(exe), str(tmp / "in.bin"), str(samples), str(tmp / "out.bin")], check=True)
        want = np.fromfile(tmp / "out.bin", dtype=np.uint64).reshape(len(OPS), samples)
    summary, status = {}, 0
    for i, (name, *_ ) in enumerate(OPS):
        g, w = got[i], want[i]
        nan = np.isnan(g.view(np.float64)) & np.isnan(w.view(np.float64))
        bad = int(((g != w) & ~nan).sum())
        summary[name] = [samples, bad]
        if bad:
            j = int(np.flatnonzero((g != w) & ~nan)[0])
            summary[name].append([f"{x:016x}" for x in operands[j]] + [f"{g[j]:016x}", f"{w[j]:016x}"])
        status |= int(bad > 0)
    print(json.dumps({"summary": summary}))
    return status


BENCH_KERNEL = r'''
using namespace metal;
[[kernel]] void bench(const device ulong* in [[buffer(0)]], device ulong* out [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]], uint3 bdim [[threads_per_threadgroup]]) {
  uint i = bid.x * bdim.x + tid.x;
  if (i >= n) return;
  ulong a = in[2 * i], b = in[2 * i + 1], acc = 0;
  for (int k = 0; k < 64; ++k) {
    ulong r = BENCH_EXPR;
    acc ^= r;
    a = (a & 0xFFF0000000000000UL) | ((a + r) & 0x000FFFFFFFFFFFFFUL);
  }
  out[i] = acc;
}
'''

# soft-float expressions to time against their df64 counterparts
BENCH = [
    ("add", "f64_add(a, b)", "f64_df_add(a, b)"),
    ("mul", "f64_mul(a, b)", "f64_df_mul(a, b)"),
    ("fma", "f64_fma(a, b, a)", "f64_df_fma(a, b, a)"),
    ("div", "f64_div(a, b)", "f64_df_div(a, b)"),
    ("sqrt", "f64_sqrt(a)", "f64_df_sqrt(a)"),
]


def bench(n: int, reps: int) -> int:
    import mlx.core as mx
    from woof.backend.metal import ext
    launch = ext.load().launch
    rng = np.random.default_rng(1)
    operands = rng.uniform(0.5, 2.0, size=(n, 2)).view(np.uint64).reshape(-1)
    inp = mx.array(operands)
    rows = {}
    for name, soft, fast in BENCH:
        row = {}
        for label, expr in (("soft", soft), ("df64", fast)):
            src = _metal_text(BENCH_KERNEL.replace("BENCH_EXPR", expr))
            key = f"woof-df64-bench-{name}-{label}"
            times = []
            for r in range(reps + 1):
                out = mx.zeros((n,), dtype=mx.uint64)
                t0 = time.perf_counter()
                (res,) = launch(key, src, "bench", ((n + 255) // 256, 1, 1), (256, 1, 1), [inp, out], [0, 0],
                                [False, True], [0, 1], -1, [struct.pack("I", n)], [2], 0, 0)
                mx.eval(res)
                times.append(time.perf_counter() - t0)
            row[label] = min(times[1:]) * 1e3
        row["speedup"] = row["soft"] / row["df64"]
        rows[name] = {k: round(v, 3) for k, v in row.items()}
        print(json.dumps({name: rows[name]}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("leg", choices=("host", "gpu", "bench"))
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args(argv)
    if not __package__:
        sys.path.insert(0, str(ROOT))
    if args.leg == "host":
        return host(args.samples, args.seed)
    if args.leg == "gpu":
        return gpu(args.samples, args.seed)
    return bench(args.samples, args.reps)


if __name__ == "__main__":
    sys.exit(main())
