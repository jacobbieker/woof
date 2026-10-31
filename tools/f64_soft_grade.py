"""Grade woof/backend/metal/f64_soft.h word for word.

Two legs, one header text:

``host``  builds the header as C++17 with clang++ and compares every
          operation against the host's IEEE binary64 hardware (AArch64 or
          x86-64 with flush-to-zero off) over random words, a subnormal-heavy
          band, near-tie operands and every special-value pair.
``gpu``   runs the same header as Metal through woof.backend.metal.ext on
          operands generated here, and compares against NumPy binary64.

Every NaN result matches every NaN (payloads are recorded, not graded: the
CUDA receipts decide the payload policy).  Exit status is non-zero on any
mismatch.

Usage::

    pixi run -e mlx python tools/f64_soft_grade.py host --samples 100000000
    pixi run -e mlx python tools/f64_soft_grade.py gpu --samples 16777216
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "woof" / "backend" / "metal" / "f64_soft.h"

HOST_TEST = r'''
#include "f64_soft.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static inline uint64_t bits(double d) { uint64_t u; std::memcpy(&u, &d, 8); return u; }
static inline double dbl(uint64_t u) { double d; std::memcpy(&d, &u, 8); return d; }
static inline uint32_t bits32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static uint64_t s[2];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t next() {  // xoroshiro128+
  uint64_t s0 = s[0], s1 = s[1], r = s0 + s1;
  s1 ^= s0; s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); s[1] = rotl(s1, 36);
  return r;
}

static const uint64_t SPECIAL[] = {
  0x0000000000000000ULL, 0x8000000000000000ULL, 0x7FF0000000000000ULL, 0xFFF0000000000000ULL,
  0x7FF8000000000000ULL, 0x7FF0000000000001ULL, 0xFFF8000000000123ULL, 0x3FF0000000000000ULL,
  0xBFF0000000000000ULL, 0x0000000000000001ULL, 0x8000000000000001ULL, 0x000FFFFFFFFFFFFFULL,
  0x0010000000000000ULL, 0x7FEFFFFFFFFFFFFFULL, 0x3FEFFFFFFFFFFFFFULL, 0x3FF0000000000001ULL,
  0x4340000000000000ULL, 0x433FFFFFFFFFFFFFULL, 0xC1E0000000000000ULL, 0x41DFFFFFFFC00000ULL,
  0x3CA0000000000000ULL, 0x0008000000000000ULL};
static const int NSPECIAL = sizeof(SPECIAL) / sizeof(SPECIAL[0]);

static uint64_t operand() {
  uint64_t r = next();
  switch (r & 7) {
    case 0: return next();                                   // any word
    case 1: return (next() & 0x800FFFFFFFFFFFFFULL);         // subnormal band
    case 2: return SPECIAL[(r >> 3) % NSPECIAL];
    case 3: {                                                // near 1, ties
      uint64_t m = next() & 0x000FFFFFFFFFFFFFULL;
      return (0x3FF0000000000000ULL | m) ^ ((r >> 8) & 1 ? 0x8000000000000000ULL : 0);
    }
    case 4: {                                                // tiny exponents
      uint64_t e = 1 + ((r >> 3) % 60);
      return (e << 52) | (next() & 0x800FFFFFFFFFFFFFULL);
    }
    case 5: {                                                // huge exponents
      uint64_t e = 0x7FE - ((r >> 3) % 60);
      return (e << 52) | (next() & 0x800FFFFFFFFFFFFFULL);
    }
    default: {                                               // modest range
      uint64_t e = 0x3FF - 40 + ((r >> 3) % 80);
      return (e << 52) | (next() & 0x800FFFFFFFFFFFFFULL);
    }
  }
}

static bool same(uint64_t got, uint64_t want) {
  if (got == want) return true;
  return std::isnan(dbl(got)) && std::isnan(dbl(want));
}

enum { ADD, SUB, MUL, DIV, SQRT, FMA, TOF32, FROMF32, FROMI64, TOI64, TOI32, CMP, NOPS };
static const char* NAMES[NOPS] = {"add", "sub", "mul", "div", "sqrt", "fma", "to_f32",
                                  "from_f32", "from_i64", "to_i64", "to_i32", "compare"};

int main(int argc, char** argv) {
  long long n = atoll(argv[1]);
  s[0] = strtoull(argv[2], 0, 10) | 1; s[1] = 0x9E3779B97F4A7C15ULL;
  long long bad[NOPS] = {0}, count[NOPS] = {0};
  int shown[NOPS] = {0};
  #define CHECK(op, got, want, a, b, c) do { count[op]++; \
    if (!same((got), (want))) { bad[op]++; if (shown[op]++ < 3) \
      printf("{\"op\":\"%s\",\"a\":\"%016llx\",\"b\":\"%016llx\",\"c\":\"%016llx\",\"got\":\"%016llx\",\"want\":\"%016llx\"}\n", \
        NAMES[op], (unsigned long long)(a), (unsigned long long)(b), (unsigned long long)(c), \
        (unsigned long long)(got), (unsigned long long)(want)); } } while (0)
  for (long long i = 0; i < n; ++i) {
    uint64_t a = operand(), b = operand(), c = operand();
    if (i < (long long)NSPECIAL * NSPECIAL) { a = SPECIAL[i % NSPECIAL]; b = SPECIAL[i / NSPECIAL]; }
    double x = dbl(a), y = dbl(b), z = dbl(c);
    CHECK(ADD, f64_add(a, b), bits(x + y), a, b, 0);
    CHECK(SUB, f64_sub(a, b), bits(x - y), a, b, 0);
    CHECK(MUL, f64_mul(a, b), bits(x * y), a, b, 0);
    CHECK(DIV, f64_div(a, b), bits(x / y), a, b, 0);
    CHECK(SQRT, f64_sqrt(a), bits(std::sqrt(x)), a, 0, 0);
    CHECK(FMA, f64_fma(a, b, c), bits(std::fma(x, y, z)), a, b, c);
    {
      float g = f64_to_f32(a), w = (float)x;
      count[TOF32]++;
      bool ok = bits32(g) == bits32(w) || (std::isnan(g) && std::isnan(w));
      if (!ok) { bad[TOF32]++; if (shown[TOF32]++ < 3)
        printf("{\"op\":\"to_f32\",\"a\":\"%016llx\",\"got\":\"%08x\",\"want\":\"%08x\"}\n",
               (unsigned long long)a, bits32(g), bits32(w)); }
    }
    {
      uint32_t fw = (uint32_t)next();
      float f; std::memcpy(&f, &fw, 4);
      CHECK(FROMF32, f64_from_f32(f), bits((double)f), fw, 0, 0);
    }
    {
      int64_t v = (int64_t)next();
      if (i & 1) v >>= (next() % 64);
      CHECK(FROMI64, f64_from_i64(v), bits((double)v), (uint64_t)v, 0, 0);
    }
    if (!std::isnan(x) && std::fabs(x) < 9.2e18) {
      CHECK(TOI64, (uint64_t)f64_to_i64(a), (uint64_t)(int64_t)x, a, 0, 0);
    }
    if (!std::isnan(x) && std::fabs(x) < 2.1e9) {
      CHECK(TOI32, (uint64_t)(uint32_t)f64_to_i32(a), (uint64_t)(uint32_t)(int32_t)x, a, 0, 0);
    }
    {
      uint64_t got = (uint64_t)f64_eq(a, b) | ((uint64_t)f64_lt(a, b) << 1) | ((uint64_t)f64_le(a, b) << 2);
      uint64_t want = (uint64_t)(x == y) | ((uint64_t)(x < y) << 1) | ((uint64_t)(x <= y) << 2);
      CHECK(CMP, got, want, a, b, 0);
    }
  }
  printf("{\"summary\":{");
  for (int k = 0; k < NOPS; ++k)
    printf("%s\"%s\":[%lld,%lld]", k ? "," : "", NAMES[k], count[k], bad[k]);
  printf("}}\n");
  for (int k = 0; k < NOPS; ++k) if (bad[k]) return 1;
  return 0;
}
'''


def host(samples: int, seed: int) -> int:
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("c++")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "grade.cpp"
        exe = Path(tmp) / "grade"
        src.write_text(HOST_TEST)
        subprocess.run([cxx, "-O2", "-std=c++17", "-ffp-contract=off",
                        "-fno-unsafe-math-optimizations", f"-I{HEADER.parent}",
                        str(src), "-o", str(exe)], check=True)
        result = subprocess.run([str(exe), str(samples), str(seed)], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        print(line)
    return result.returncode


GPU_KERNEL = r'''
#include "F64_SOFT"
using namespace metal;
[[kernel]] void grade(const device ulong* a [[buffer(0)]], const device ulong* b [[buffer(1)]],
    const device ulong* c [[buffer(2)]], device ulong* out [[buffer(3)]], constant uint& n [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]], uint3 bid [[threadgroup_position_in_grid]],
    uint3 bdim [[threads_per_threadgroup]]) {
  uint i = bid.x * bdim.x + tid.x;
  if (i >= n) return;
  ulong x = a[i], y = b[i], z = c[i];
  out[7 * i + 0] = f64_add(x, y);
  out[7 * i + 1] = f64_sub(x, y);
  out[7 * i + 2] = f64_mul(x, y);
  out[7 * i + 3] = f64_div(x, y);
  out[7 * i + 4] = f64_sqrt(x);
  out[7 * i + 5] = f64_fma(x, y, z);
  out[7 * i + 6] = as_type<uint>(f64_to_f32(x));
}
'''


def _gpu_operands(rng, n):
    words = rng.integers(0, 2**63, size=n, dtype=np.uint64) * np.uint64(2) + rng.integers(0, 2, size=n, dtype=np.uint64)
    k = n // 6
    words[:k] &= np.uint64(0x800FFFFFFFFFFFFF)                       # subnormals
    e = rng.integers(0x3FF - 40, 0x3FF + 40, size=k, dtype=np.uint64)
    words[k:2 * k] = (e << np.uint64(52)) | (words[k:2 * k] & np.uint64(0x800FFFFFFFFFFFFF))
    return words


def gpu(samples: int, seed: int) -> int:
    import mlx.core as mx
    from woof.backend.metal import ext
    launch = ext.load().launch
    source = GPU_KERNEL.replace('#include "F64_SOFT"', HEADER.read_text())
    rng = np.random.default_rng(seed)
    a, b, c = (_gpu_operands(rng, samples) for _ in range(3))
    out = mx.zeros((7 * samples,), dtype=mx.uint64)
    (res,) = launch("woof-f64-soft-grade", source, "grade", ((samples + 255) // 256, 1, 1), (256, 1, 1),
                    [mx.array(a), mx.array(b), mx.array(c), out], [0, 0, 0, 0],
                    [False, False, False, True], [0, 1, 2, 3], -1, [struct.pack("I", samples)], [4], 0, 0)
    got = np.array(res).reshape(samples, 7)
    x, y, z = a.view(np.float64), b.view(np.float64), c.view(np.float64)
    with np.errstate(all="ignore"):
        import math
        fma = np.array([math.fma(p, q, r) for p, q, r in zip(x[:1 << 20], y[:1 << 20], z[:1 << 20])])
        want = {"add": x + y, "sub": x - y, "mul": x * y, "div": x / y, "sqrt": np.sqrt(x)}
    summary, status = {}, 0
    for k, name in enumerate(("add", "sub", "mul", "div", "sqrt")):
        g = got[:, k].view(np.float64)
        w = want[name]
        ok = (g.view(np.uint64) == w.view(np.uint64)) | (np.isnan(g) & np.isnan(w))
        summary[name] = [samples, int((~ok).sum())]
        status |= int((~ok).any())
    g = got[:1 << 20, 5].view(np.float64)
    ok = (g.view(np.uint64) == fma.view(np.uint64)) | (np.isnan(g) & np.isnan(fma))
    summary["fma"] = [int(ok.size), int((~ok).sum())]
    status |= int((~ok).any())
    g32 = got[:, 6].astype(np.uint32).view(np.float32)
    with np.errstate(all="ignore"):
        w32 = x.astype(np.float32)
    ok = (g32.view(np.uint32) == w32.view(np.uint32)) | (np.isnan(g32) & np.isnan(w32))
    summary["to_f32"] = [samples, int((~ok).sum())]
    status |= int((~ok).any())
    print(json.dumps({"summary": summary}))
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("leg", choices=("host", "gpu"))
    parser.add_argument("--samples", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args(argv)
    if not __package__:
        sys.path.insert(0, str(ROOT))
    return host(args.samples, args.seed) if args.leg == "host" else gpu(args.samples, args.seed)


if __name__ == "__main__":
    sys.exit(main())
