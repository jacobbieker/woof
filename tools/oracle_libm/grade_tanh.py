"""Grade woof/core/kernels/cr_tanh_flt64.cuh word for word.

The transcription (``g64_tanh::cr_tanh``) is held to the oracle host's glibc
``tanh``: glibc 2.44 carries CORE-MATH's correctly rounded binary64 tanh, so
those are the words gfortran's ``tanh`` gives the MALI oracle at RKIND 8.

``cpu``  compiles the header for the host (woof/backend/host/cuda_host_shim.h)
         and sweeps arguments 0..count-1 of the stream in grade_tanh_args.h
         through the transcription and libm.so.6's ``tanh`` (OpenMP).  Needs
         a glibc host.
``gpu``  compiles the header with NVRTC option --fmad=true (the project
         default, so only the explicit rounding holds the words) on the active
         backend (CUDA, or Metal through the facade), evaluates the same stream
         on the device and compares with the host transcription and, on a
         glibc >= 2.44 host, libm.so.6.

Both legs also check every 2^16th argument against an exact decimal tanh
(``exact_tanh``) rounded once to binary64, which needs no libm at all.
Exit status is non-zero on any mismatch.

    pixi run -e default python tools/oracle_libm/grade_tanh.py cpu --count 17179869184
    pixi run -e default python tools/oracle_libm/grade_tanh.py gpu --count 268435456
"""
from __future__ import annotations

import argparse
import ctypes
import os
import platform
import subprocess
import sys
import tempfile
from decimal import Context, Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HEADER = ROOT / "woof" / "core" / "kernels" / "cr_tanh_flt64.cuh"
ARGS = HERE / "grade_tanh_args.h"
CHUNK = 1 << 26
EXACT_STRIDE = 1 << 16

KERNEL = r'''
extern "C" __global__ void tanh_grade(unsigned long long start, unsigned long long n, double *x, double *y)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double a = tanh_arg(start + i);
    x[i] = a;
    y[i] = g64_tanh::cr_tanh(a);
}
'''


def exact_tanh(x: float) -> float:
    """tanh(x) to 80 significant digits in decimal, rounded once to binary64."""
    if x != x or x == 0.0:
        return x + x if x != x else x
    if abs(x) > 40.0:
        return 1.0 if x > 0 else -1.0
    ctx = Context(prec=80)
    d = Decimal(x)
    if abs(x) < 1e-6:
        # odd Taylor series, terms to x^9 are far below 2^-60 relative
        d2 = ctx.multiply(d, d)
        value = d * (1 - d2 / 3 + 2 * d2 * d2 / 15 - 17 * d2 ** 3 / 315 + 62 * d2 ** 4 / 2835)
    else:
        e = ctx.exp(ctx.multiply(2, d))
        value = ctx.divide(ctx.subtract(e, 1), ctx.add(e, 1))
    return float(value)


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
                    f"-I{ROOT / 'woof' / 'backend' / 'host'}", f"-I{HEADER.parent}", f"-I{HERE}",
                    str(HERE / "grade_tanh.cpp"), "-ldl", "-o", str(work / "grade_tanh.so")], check=True)
    lib = ctypes.CDLL(str(work / "grade_tanh.so"))
    u64, ptr = ctypes.c_uint64, ctypes.c_void_p
    lib.tanh_args.argtypes = (u64, u64, ptr)
    for name in ("tanh_host_libm", "tanh_transcription"):
        getattr(lib, name).argtypes = (ptr, u64, ptr)
    lib.tanh_sweep.argtypes = (u64, u64, ctypes.POINTER(u64), ctypes.POINTER(u64))
    return lib


def _p(a: np.ndarray) -> int:
    return a.ctypes.data


def mismatches(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    both_nan = np.isnan(a) & np.isnan(b)
    return np.flatnonzero((a.view(np.uint64) != b.view(np.uint64)) & ~both_nan)


def exact_check(lib, count: int) -> int:
    idx = np.arange(0, count, EXACT_STRIDE, dtype=np.uint64)
    x = np.empty(idx.size)
    for k, i in enumerate(idx):
        lib.tanh_args(int(i), 1, _p(x[k:k + 1]))
    got = np.empty_like(x)
    lib.tanh_transcription(_p(x), x.size, _p(got))
    want = np.array([exact_tanh(float(v)) for v in x])
    bad = mismatches(got, want)
    for k in bad[:5]:
        print(f"  exact mismatch: {float(x[k]).hex()} got {float(got[k]).hex()} want {float(want[k]).hex()}")
    print(f"exact: {x.size} arguments (every {EXACT_STRIDE}th); transcription vs exact decimal tanh: "
          f"{bad.size} mismatches")
    return bad.size


def cpu(lib, count: int) -> int:
    if not lib.tanh_bind_host_libm():
        raise SystemExit("the cpu leg needs a glibc host (libm.so.6)")
    print(f"host libm: glibc {'.'.join(map(str, glibc_version() or ()))}")
    total = 0
    for start in range(0, count, CHUNK):
        n = min(CHUNK, count - start)
        ml, first = ctypes.c_uint64(), ctypes.c_uint64()
        lib.tanh_sweep(start, n, ctypes.byref(ml), ctypes.byref(first))
        total += ml.value
        if ml.value:
            x = np.empty(1)
            lib.tanh_args(first.value, 1, _p(x))
            print(f"  first glibc mismatch: argument {first.value} = {float(x[0]).hex()}")
    print(f"cpu: {count} arguments; transcription vs glibc tanh: {total} mismatches")
    return total + exact_check(lib, count)


def gpu(lib, count: int) -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from woof.backend import install

    install()  # WOOF_BACKEND: CUDA, or Metal through woof's cupy facade
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    have_libm = (glibc_version() or (0, 0)) >= (2, 44) and lib.tanh_bind_host_libm()
    source = HEADER.read_text(encoding="utf-8") + "\n" + ARGS.read_text(encoding="utf-8") + KERNEL
    image, _ = compile_using_nvrtc(source, ("--std=c++17", "--fmad=true"), None, "cr_tanh_grade.cu")
    module = cp.cuda.function.Module()
    module.load(image.encode() if isinstance(image, str) else image)
    kernel = module.get_function("tanh_grade")
    bad_host = bad_libm = 0
    for start in range(0, count, CHUNK):
        n = min(CHUNK, count - start)
        dx, dy = cp.empty(n, cp.float64), cp.empty(n, cp.float64)
        kernel(((n + 255) // 256,), (256,), (np.uint64(start), np.uint64(n), dx, dy))
        x, y = cp.asnumpy(dx), cp.asnumpy(dy)
        want = np.empty(n)
        lib.tanh_args(start, n, _p(want))
        if mismatches(x, want).size:
            raise SystemExit("device and host argument streams differ")
        lib.tanh_transcription(_p(x), n, _p(want))
        bad = mismatches(y, want)
        bad_host += bad.size
        if bad.size:
            print(f"  first host mismatch: {float(x[bad[0]]).hex()} device {float(y[bad[0]]).hex()} "
                  f"host {float(want[bad[0]]).hex()}")
        if have_libm:
            lib.tanh_host_libm(_p(x), n, _p(want))
            bad_libm += mismatches(y, want).size
    print(f"gpu: {count} arguments; device vs host transcription: {bad_host} mismatches; "
          + (f"vs glibc tanh: {bad_libm} mismatches" if have_libm else "no glibc >= 2.44 on this host"))
    return bad_host + bad_libm + exact_check(lib, count)


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
