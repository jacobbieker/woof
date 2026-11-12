"""Grade tools/glibc239_libm_gpu/glibc239_trig_flt64.cuh word for word.

The transcription (``g239::g239_sin``, ``g239_cos``, ``g239_atan``) is glibc
2.39's binary64 sin, cos and atan as its x86-64 FMA variants compute them.
It is held to the oracle host's libm (glibc 2.44, the same sources) and to
the repository's Rust transcription's pinned words.

``cpu``  compiles the header for the host (woof/backend/host/cuda_host_shim.h)
         and sweeps arguments 0..count-1 of each function's stream in
         grade_trig_args.h through the transcription and libm.so.6 (OpenMP).
         Needs a glibc host.  Mismatches with |x| >= 105414350 (sin, cos:
         glibc's branred range) are counted apart: glibc 2.44's build of
         branred.c rounds some of those reductions differently from 2.39's.
``gpu``  compiles the header with NVRTC option --fmad=true (so only the
         explicit rounding holds the words) on the active backend (CUDA, or
         Metal through the facade), evaluates the same stream on the device
         and compares with the host transcription (every argument) and, on a
         glibc host, libm.so.6 (outside the branred range).

Both legs check the pinned words.  Exit status is non-zero on any mismatch
outside the branred range, or any device/host transcription mismatch.

    pixi run -e default python tools/glibc239_libm_gpu/grade_trig.py cpu --count 4294967296
    pixi run -e default python tools/glibc239_libm_gpu/grade_trig.py gpu --count 268435456
"""
from __future__ import annotations

import argparse
import ctypes
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HEADER = HERE / "glibc239_trig_flt64.cuh"
ARGS = HERE / "grade_trig_args.h"
CHUNK = 1 << 26
FUNCTIONS = ("sin", "cos", "atan")
BRANRED = 105414350.0

KERNEL = r'''
extern "C" __global__ void trig_grade(int f, unsigned long long start, unsigned long long n, double *x, double *y)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double a = trig_arg(f, start + i);
    x[i] = a;
    y[i] = f == 0 ? g239::g239_sin(a) : f == 1 ? g239::g239_cos(a) : g239::g239_atan(a);
}
'''

PIN_KERNEL = r'''
extern "C" __global__ void trig_pin(int f, unsigned long long n, const double *x, double *y)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = f == 0 ? g239::g239_sin(x[i]) : f == 1 ? g239::g239_cos(x[i]) : g239::g239_atan(x[i]);
}
'''

#: (function, argument word, glibc 2.39 x86-64 result word): the Rust
#: crate's pins (tools/rustwx/crates/glibc239-libm/src/sincos.rs, atan.rs).
#: The last three sin/cos arguments are huge ones where glibc 2.44 differs.
PINNED = [
    *((f, x, w) for x, s, c in (
        (0x3fe0000000000000, 0x3fdeaee8744b05f0, 0x3fec1528065b7d50),
        (0x3ff0000000000000, 0x3feaed548f090cee, 0x3fe14a280fb5068c),
        (0x4000000000000000, 0x3fed18f6ead1b446, 0xbfdaa22657537205),
        (0x4008000000000000, 0x3fc210386db6d55b, 0xbfefae04be85e5d2),
        (0x41cdcd6500000000, 0x3fe1778cae83c69b, 0x3feacff8c7364234),
        (0x7e37e43c8800759c, 0xbfea2c16b010e385, 0xbfe2699022adc4c1),
        (0x3fb999999999999a, 0x3fb98eaecb8bcb2c, 0x3fefd712f9a817c1),
        (0x3e112e0be826d695, 0x3e112e0be826d695, 0x3ff0000000000000),
        (0x3feb5fffffffffff, 0x3fe827f6d2325577, 0x3fe4fcd8091b8f24),
        (0x400368fd00000000, 0x3fe4fcda0ad3a77a, 0xbfe827f513dbe3a2),
        (0x419921fb00000000, 0xbfe98c2ea3441c6b, 0xbfe3450a2c59b604),
        (0xeb1e50585fd26e73, 0x3fbe7a914ead0d9a, 0xbfefc5bba8036a59),
        (0x7b9b9d1e3a18bed9, 0xbfefd9f8163b393a, 0xbfb8a397e5e1946d),
        (0x43fb5fd2b98ab782, 0x3feff6e7ebd15d1e, 0x3fa81e4eb2538ae8),
    ) for f, w in ((0, s), (1, c))),
    *((2, x, w) for x, w in (
        (0x3e112e0be826d695, 0x3e112e0be826d695), (0x3fa47ae147ae147b, 0x3fa4781623768d56),
        (0xbfb0000000000000, 0xbfaff55bb72cfdea), (0x3fd3333333333333, 0x3fd2a73a661eaf06),
        (0xbfe6a09e667f3bcd, 0xbfe3b2028082e8d4), (0x3fefff2e48e8a71e, 0x3fe921927760e4cb),
        (0x3ff0000000000000, 0x3fe921fb54442d18), (0x3ff8000000000000, 0x3fef730bd281f69b),
        (0xc01d000000000000, 0xbff6f08f07435fec), (0x402ffae147ae147b, 0x3ff82227a313716c),
        (0x4030000000000000, 0x3ff82250768ac529), (0xc0934a0000000000, 0xbff91ea9efa2930a),
        (0x432550f7dca70000, 0x3ff921fb54442d17), (0xfe37e43c8800759c, 0xbff921fb54442d18),
    )),
]


def glibc_version() -> tuple[int, int] | None:
    name, version = platform.libc_ver()
    if name != "glibc" or not version:
        return None
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


def build(work: Path) -> ctypes.CDLL:
    cxx = os.environ.get("CXX", "c++")
    omp = [] if sys.platform == "darwin" else ["-fopenmp"]
    subprocess.run([cxx, "-std=c++17", "-O2", "-ffp-contract=off", *omp, "-shared", "-fPIC",
                    f"-I{ROOT / 'woof' / 'backend' / 'host'}", f"-I{HERE}",
                    str(HERE / "grade_trig.cpp"), "-ldl", "-o", str(work / "grade_trig.so")], check=True)
    lib = ctypes.CDLL(str(work / "grade_trig.so"))
    i32, u64, ptr = ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p
    lib.trig_args.argtypes = (i32, u64, u64, ptr)
    for name in ("trig_host_libm", "trig_transcription"):
        getattr(lib, name).argtypes = (i32, ptr, u64, ptr)
    lib.trig_sweep.argtypes = (i32, u64, u64, ptr, ctypes.POINTER(u64))
    return lib


def _p(a: np.ndarray) -> int:
    return a.ctypes.data


def mismatches(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    both_nan = np.isnan(a) & np.isnan(b)
    return np.flatnonzero((a.view(np.uint64) != b.view(np.uint64)) & ~both_nan)


def pinned_check(evaluate) -> int:
    """evaluate(f, x array) -> result array; every pinned word must match."""
    bad = 0
    for f in range(3):
        rows = [(x, w) for g, x, w in PINNED if g == f]
        x = np.array([r[0] for r in rows], dtype=np.uint64).view(np.float64)
        want = np.array([r[1] for r in rows], dtype=np.uint64)
        got = evaluate(f, x).view(np.uint64)
        for k in np.flatnonzero(got != want):
            print(f"  pinned mismatch: {FUNCTIONS[f]}({int(x.view(np.uint64)[k]):#018x}) "
                  f"got {int(got[k]):#018x} want {int(want[k]):#018x}")
        bad += int(np.count_nonzero(got != want))
    print(f"pinned: {len(PINNED)} glibc 2.39 words: {bad} mismatches")
    return bad


def _host_evaluate(lib):
    def evaluate(f, x):
        out = np.empty_like(x)
        lib.trig_transcription(f, _p(x), x.size, _p(out))
        return out
    return evaluate


def cpu(lib, count: int) -> int:
    if not lib.trig_bind_host_libm():
        raise SystemExit("the cpu leg needs a glibc host (libm.so.6)")
    print(f"host libm: glibc {'.'.join(map(str, glibc_version() or ()))}; {platform.processor() or platform.machine()}")
    total = 0
    for f, name in enumerate(FUNCTIONS):
        sums = np.zeros(3, dtype=np.uint64)
        for start in range(0, count, CHUNK):
            n = min(CHUNK, count - start)
            counts, first = np.zeros(3, dtype=np.uint64), ctypes.c_uint64()
            lib.trig_sweep(f, start, n, _p(counts), ctypes.byref(first))
            sums += counts
            if counts[0]:
                x = np.empty(1)
                lib.trig_args(f, first.value, 1, _p(x))
                print(f"  first {name} mismatch: argument {first.value} = {float(x[0]).hex()}")
        total += int(sums[0])
        print(f"cpu {name}: {count} arguments; transcription vs glibc: {int(sums[0])} mismatches below "
              f"{BRANRED:.0f}" + (f"; branred range: {int(sums[1])} of {int(sums[2])} differ" if f < 2 else ""))
    return total + pinned_check(_host_evaluate(lib))


def gpu(lib, count: int) -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from woof.backend import install

    install()  # WOOF_BACKEND: CUDA, or Metal through woof's cupy facade
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    have_libm = glibc_version() is not None and lib.trig_bind_host_libm()
    source = HEADER.read_text(encoding="utf-8") + "\n" + ARGS.read_text(encoding="utf-8") + KERNEL
    image, _ = compile_using_nvrtc(source, ("--std=c++17", "--fmad=true"), None, "glibc239_trig_grade.cu")
    module = cp.cuda.function.Module()
    module.load(image.encode() if isinstance(image, str) else image)
    kernel = module.get_function("trig_grade")
    total = 0
    for f, name in enumerate(FUNCTIONS):
        bad_host = bad_libm = huge_libm = 0
        for start in range(0, count, CHUNK):
            n = min(CHUNK, count - start)
            dx, dy = cp.empty(n, cp.float64), cp.empty(n, cp.float64)
            kernel(((n + 255) // 256,), (256,), (np.int32(f), np.uint64(start), np.uint64(n), dx, dy))
            x, y = cp.asnumpy(dx), cp.asnumpy(dy)
            want = np.empty(n)
            lib.trig_args(f, start, n, _p(want))
            if mismatches(x, want).size:
                raise SystemExit("device and host argument streams differ")
            lib.trig_transcription(f, _p(x), n, _p(want))
            bad = mismatches(y, want)
            bad_host += bad.size
            if bad.size:
                print(f"  first host mismatch: {name}({float(x[bad[0]]).hex()}) device {float(y[bad[0]]).hex()} "
                      f"host {float(want[bad[0]]).hex()}")
            if have_libm:
                lib.trig_host_libm(f, _p(x), n, _p(want))
                bad = mismatches(y, want)
                huge = (np.abs(x[bad]) >= BRANRED) if f < 2 else np.zeros(bad.size, bool)
                bad_libm += int(np.count_nonzero(~huge))
                huge_libm += int(np.count_nonzero(huge))
        total += bad_host + bad_libm
        print(f"gpu {name}: {count} arguments; device vs host transcription: {bad_host} mismatches; "
              + (f"vs glibc {'.'.join(map(str, glibc_version()))}: {bad_libm} mismatches below {BRANRED:.0f}"
                 + (f", {huge_libm} in the branred range" if f < 2 else "") if have_libm else "no glibc on this host"))

    pin_source = HEADER.read_text(encoding="utf-8") + PIN_KERNEL
    pin_image, _ = compile_using_nvrtc(pin_source, ("--std=c++17", "--fmad=true"), None, "glibc239_trig_pin.cu")
    pin_module = cp.cuda.function.Module()
    pin_module.load(pin_image.encode() if isinstance(pin_image, str) else pin_image)
    pin = pin_module.get_function("trig_pin")

    def evaluate(f, x):
        dx, dy = cp.asarray(x), cp.empty(x.size, cp.float64)
        pin((1,), (256,), (np.int32(f), np.uint64(x.size), dx, dy))
        return cp.asnumpy(dy)

    return total + pinned_check(evaluate)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("leg", choices=("cpu", "gpu"))
    parser.add_argument("--count", type=int, default=1 << 26)
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        lib = build(Path(tmp))
        bad = cpu(lib, args.count) if args.leg == "cpu" else gpu(lib, args.count)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
