"""corem_log10f.cuh on the device against the host C library, every float32.

The header is compiled as the cldfra3 unit compiles it (glibc_flt32.cuh
first; -std=c++17 --fmad=false --ftz=false) and, as a control, with FMA
contraction on.  The 2^32 inputs run in 16 chunks against the host's
log10f (gpu_oracle.so).  A correctly rounded log10f is one function, so any
glibc >= 2.42 host is the same reference as glibc 2.43 (exhaustive.cpp
proves the transcription against 2.43 itself).  NaN results compare as NaN.

    g++ -O2 -std=c++17 -fno-builtin -fopenmp -shared -fPIC gpu_oracle.cpp -lm -o gpu_oracle.so
    python gpu_check.py > gpu-final.log
"""
from pathlib import Path
import ctypes
import hashlib

import cupy as cp
import numpy as np

root = Path(__file__).resolve().parent
kdir = root.parents[1] / "woof" / "core" / "kernels"
lib = ctypes.CDLL(str(root / "gpu_oracle.so"))
lib.reference.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
header = (kdir / "corem_log10f.cuh").read_text(encoding="utf-8")
pre = (kdir / "glibc_flt32.cuh").read_text(encoding="utf-8")
src = pre + "\n" + header + "\n" + (
    'extern "C" __global__ void check(unsigned long long start, unsigned int* y,'
    ' unsigned long long count) { unsigned long long k = (unsigned long long)'
    'blockIdx.x * blockDim.x + threadIdx.x; if (k < count) y[k] ='
    ' __float_as_uint(cm_log10f(__uint_as_float((unsigned int)(start + k)))); }\n')
print("CuPy", cp.__version__, "NVRTC", cp.cuda.nvrtc.getVersion(), "GPU",
      cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), flush=True)
print("header_sha256", hashlib.sha256(header.encode()).hexdigest(), flush=True)
chunk = 1 << 28
for options in (("-std=c++17", "--fmad=false", "--ftz=false"),
                ("-std=c++17", "--fmad=true", "--ftz=false")):
    module = cp.RawModule(code=src, options=options)
    bad = 0
    examples = []
    dy = cp.empty(chunk, dtype=cp.uint32)
    expected = np.empty(chunk, dtype=np.uint32)
    for start in range(0, 1 << 32, chunk):
        lib.reference(start, expected.ctypes.data, chunk)
        module.get_function("check")(((chunk + 255) // 256,), (256,),
                                     (np.uint64(start), dy, np.uint64(chunk)))
        got = dy.get()
        diff = (got != expected) & ~(np.isnan(expected.view(np.float32))
                                     & np.isnan(got.view(np.float32)))
        if diff.any():
            idx = np.flatnonzero(diff)
            bad += idx.size
            for j in idx[: max(0, 8 - len(examples))]:
                examples.append((hex(start + int(j)), hex(int(expected[j])),
                                 hex(int(got[j]))))
    print(options, "log10f inputs 4294967296 mismatches", bad, "examples",
          examples, flush=True)
