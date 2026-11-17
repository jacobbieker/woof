"""Grade woof/backend/metal/gfk_fast.h on the GPU: every word the shortcut
returns must be the binary64 routine's word.

The probe unit is glibc_flt32.cuh plus three kernels, translated the way
every kernel unit is (woof.backend.translate.gfk_fast renames each
routine's binary64 body ``*_soft``).  For each input the kernel compares
the routed call (shortcut, or the binary64 body where the shortcut
defers) with the binary64 body called directly, and counts mismatches and
deferrals.  Run under ``pixi run -e mlx`` on a Mac.

  logf, expf: all 2^32 inputs (``--unary-stride 1``, the default), in a
              flushing unit and a --ftz=false (f32s) unit;
  powf:       every positive normal base (``--pow-stride``) for each
              exponent in POW_EXPONENTS, and ``--pow-random`` random pairs
              (positive normal bases, exponents of either sign over
              2^-12..2^7).

Exit status 1 on any mismatch.  tests/test_metal_gfk_fast.py runs a
strided subset of the same probes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "woof" / "core" / "kernels" / "glibc_flt32.cuh"

PROBES = r'''
extern "C" __global__ void gfk_probe_unary(unsigned int base, unsigned int stride, unsigned int n,
                                           int which, unsigned int* stats, unsigned int* first)
{
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    unsigned int u = base + t * stride;
    float x = __uint_as_float(u);
    float a, b;
    unsigned int f;
    if (which == 0) { a = gfk_log(x); b = gfk_log_soft(x); f = __woof_gfk_log(x); }
    else            { a = gfk_exp(x); b = gfk_exp_soft(x); f = __woof_gfk_exp(x); }
    if (f == 0xFFFFFFFFu) atomicAdd(&stats[1], 1u);
    if (__float_as_uint(a) != __float_as_uint(b)) { atomicAdd(&stats[0], 1u); atomicMin(first, u); }
}

extern "C" __global__ void gfk_probe_pow(const float* xs, const float* ys, unsigned int n,
                                         unsigned int* stats, unsigned int* first)
{
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    float x = xs[t], y = ys[t];
    float a = gfk_pow(x, y), b = gfk_pow_soft(x, y);
    if (__woof_gfk_pow(x, y) == 0xFFFFFFFFu) atomicAdd(&stats[1], 1u);
    if (__float_as_uint(a) != __float_as_uint(b)) { atomicAdd(&stats[0], 1u); atomicMin(first, t); }
}

extern "C" __global__ void gfk_probe_pow_sweep(unsigned int base, unsigned int stride, unsigned int n,
                                               float y, unsigned int* stats, unsigned int* first)
{
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    unsigned int u = base + t * stride;
    float x = __uint_as_float(u);
    float a = gfk_pow(x, y), b = gfk_pow_soft(x, y);
    if (__woof_gfk_pow(x, y) == 0xFFFFFFFFu) atomicAdd(&stats[1], 1u);
    if (__float_as_uint(a) != __float_as_uint(b)) { atomicAdd(&stats[0], 1u); atomicMin(first, u); }
}
'''

#: exponents the physics raises to (WRF's R/cp and 1/R/cp-style powers,
#: square and cube roots, the Marshall-Palmer and fall-speed exponents),
#: and a few of each sign
POW_EXPONENTS = (0.2857143, 3.5, 0.25, 0.5, 1.0 / 3.0, 0.4, 1.5, 2.5, -0.5, 0.19, 0.16,
                 0.8, 1.4, -1.1, 4.0, 0.0625, 2.0, -2.0, 0.75, 1.1)

#: the two arithmetic contracts a unit can carry: flushing (CUDA's default
#: -ftz=true) and --ftz=false (float is f32s on Metal)
OPTION_SETS = {"ftz": ("-std=c++17", "--fmad=false"),
               "noftz": ("-std=c++17", "--fmad=false", "--ftz=false")}


def probe_module(options):
    from woof.backend.metal.loader import load_unit
    source = HEADER.read_text(encoding="utf-8") + PROBES
    return load_unit("gfk_probe_" + ("noftz" if "--ftz=false" in options else "ftz"), source,
                     options=options)


def _sync():
    import mlx.core as mx
    mx.synchronize()


def run_unary(module, which: int, *, stride: int = 1, chunk: int = 1 << 26, start: int = 0,
              stop: int = 1 << 32, log=print):
    import cupy as cp
    fn = module.get_function("gfk_probe_unary")
    stats = cp.zeros(2, dtype=cp.uint32)
    first = cp.full(1, 0xFFFFFFFF, dtype=cp.uint32)
    total = 0
    t0 = time.perf_counter()
    for base in range(start, stop, chunk * stride):
        n = min(chunk, (stop - base + stride - 1) // stride)
        fn(((n + 255) // 256,), (256,), (np.uint32(base), np.uint32(stride), np.uint32(n),
                                         np.int32(which), stats, first))
        total += n
    _sync()
    s = cp.asnumpy(stats)
    return int(s[0]), int(s[1]), total, int(cp.asnumpy(first)[0]), time.perf_counter() - t0


def run_pow_sweep(module, y: float, *, stride: int = 1, chunk: int = 1 << 26):
    import cupy as cp
    fn = module.get_function("gfk_probe_pow_sweep")
    stats = cp.zeros(2, dtype=cp.uint32)
    first = cp.full(1, 0xFFFFFFFF, dtype=cp.uint32)
    lo, hi = 0x00800000, 0x7f800000           # every positive normal base
    total = 0
    for base in range(lo, hi, chunk * stride):
        n = min(chunk, (hi - base + stride - 1) // stride)
        fn(((n + 255) // 256,), (256,), (np.uint32(base), np.uint32(stride), np.uint32(n),
                                         np.float32(y), stats, first))
        total += n
    _sync()
    s = cp.asnumpy(stats)
    return int(s[0]), int(s[1]), total, int(cp.asnumpy(first)[0])


def run_pow_random(module, count: int, *, seed: int = 0, chunk: int = 1 << 24):
    import cupy as cp
    fn = module.get_function("gfk_probe_pow")
    rng = np.random.default_rng(seed)
    stats = cp.zeros(2, dtype=cp.uint32)
    first = cp.full(1, 0xFFFFFFFF, dtype=cp.uint32)
    bad = []
    done = 0
    while done < count:
        n = min(chunk, count - done)
        xs = rng.integers(0x00800000, 0x7f800000, n, dtype=np.uint32).view(np.float32)
        mag = np.exp2(rng.uniform(-12.0, 7.0, n)).astype(np.float32)
        ys = np.where(rng.random(n) < 0.5, mag, -mag).astype(np.float32)
        before = int(cp.asnumpy(stats)[0])
        fn(((n + 255) // 256,), (256,), (cp.asarray(xs), cp.asarray(ys), np.uint32(n), stats, first))
        _sync()
        if int(cp.asnumpy(stats)[0]) != before:
            i = int(cp.asnumpy(first)[0])
            bad.append((float(xs[i]), float(ys[i])))
            first.fill(0xFFFFFFFF)
        done += n
    s = cp.asnumpy(stats)
    return int(s[0]), int(s[1]), done, bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--unary-stride", type=int, default=1)
    ap.add_argument("--pow-stride", type=int, default=1)
    ap.add_argument("--pow-random", type=int, default=1 << 30)
    ap.add_argument("--options", choices=sorted(OPTION_SETS), nargs="*", default=sorted(OPTION_SETS))
    args = ap.parse_args(argv)
    failed = False
    for tag in args.options:
        module = probe_module(OPTION_SETS[tag])
        for which, name in ((0, "logf"), (1, "expf")):
            bad, deferred, total, first, secs = run_unary(module, which, stride=args.unary_stride)
            failed |= bad > 0
            print(f"{tag} {name}: {total} inputs, {bad} differ"
                  + (f" (first 0x{first:08x})" if bad else "")
                  + f", {deferred} deferred ({deferred / total:.2e}), {secs:.0f} s", flush=True)
        for y in POW_EXPONENTS:
            bad, deferred, total, first = run_pow_sweep(module, y, stride=args.pow_stride)
            failed |= bad > 0
            print(f"{tag} powf(x, {y!r}): {total} bases, {bad} differ"
                  + (f" (first x 0x{first:08x})" if bad else "")
                  + f", {deferred} deferred ({deferred / total:.2e})", flush=True)
        if args.pow_random:
            bad, deferred, total, pairs = run_pow_random(module, args.pow_random)
            failed |= bad > 0
            print(f"{tag} powf random: {total} pairs, {bad} differ {pairs[:5]}, {deferred} deferred "
                  f"({deferred / total:.2e})", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
