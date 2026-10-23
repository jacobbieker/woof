"""Prove the correctly rounded binary64 cbrt on the device and on the host.

Three implementations of one function, graded word for word over a large
deterministic argument sweep:

* ``uw_cbrt`` -- the device transcription,
  ``woof/core/kernels/cr_cbrt_flt64.cuh`` (NVRTC, -fmad=false --ftz=false);
* ``woof.core.cr_cbrt.cbrt`` -- the NumPy host transcription;
* upstream CORE-MATH ``cr_cbrt`` (``upstream/cbrt.c``, verbatim), compiled
  with ``gcc -O2 -ffp-contract=off`` through ``cbrt_batch.c``;

and every one against ``woof.core.correctly_rounded_libm.cbrt_cr``, an
exact integer reference, on a stride of each chunk (the reference is ~4 us
an argument in Python, so it grades 2^20 per 2^24 chunk; the three fast
implementations grade every argument against each other).  The host C
library's ``cbrt`` (``np.cbrt``) is counted too, to show what is replaced.

Run from the repository root, with CuPy and a device:

    gcc -O2 -ffp-contract=off -fopenmp -shared -fPIC \\
        tools/uwpbl_wrf471_oracle/libm64/cbrt_batch.c \\
        tools/uwpbl_wrf471_oracle/libm64/upstream/cbrt.c -o /tmp/cbrt_batch.so
    python tools/uwpbl_wrf471_oracle/libm64/sweep_cbrt.py \\
        --library /tmp/cbrt_batch.so --chunks 16 > \\
        tools/uwpbl_wrf471_oracle/libm64/gpu-cbrt.log

``--chunks 16`` is 16 x 2^24 = 2^28 arguments.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing
import platform
import sys
import time

import numpy as np

CHUNK = 1 << 24
#: Arguments per chunk graded against the exact integer reference.
EXACT_PER_CHUNK = 1 << 20

#: Upstream's own hard cases (the rounding-to-nearest exceptions and the
#: directed-rounding worst cases), exact cubes and the edges of the format.
_HARD = ["0x1.9b78223aa307cp+1", "0x1.a202bfc89ddffp+2",
         "0x1.3a9ccd7f022dbp+0", "0x1.7845d2faac6fep+0",
         "0x1.d1ef81cbbbe71p+0", "0x1.0a2014f62987cp+1",
         "0x1.fe18a044a5501p+1", "0x1.a6bb8c803147bp+2",
         "0x1.ac8538a031cbdp+2"]


def arguments(chunk: int) -> np.ndarray:
    """2^24 deterministic arguments: eight interleaved streams."""
    rng = np.random.default_rng(0x63627274 + chunk)
    part = CHUNK // 8
    streams = [
        # every binary64 bit pattern (NaNs, infinities, subnormals included)
        rng.integers(0, 2**64 - 1, part, dtype=np.uint64,
                     endpoint=True).view(np.float64),
        # the density ratio rho / 1.225 and the REWS cube of the diagnostic
        rng.uniform(0.5, 1.5, part),
        rng.uniform(0.0, 1.0e5, part),
        # log-uniform over the whole normal range, both signs
        np.ldexp(rng.uniform(0.5, 1.0, part),
                 rng.integers(-1021, 1025, part))
        * np.where(rng.integers(0, 2, part) == 1, -1.0, 1.0),
        # subnormals
        rng.integers(1, 2**52, part, dtype=np.uint64).view(np.float64),
        # exact cubes of integers, scaled by powers of eight, and their
        # immediate neighbours (where an exact case must be recognised)
        _near(np.ldexp(rng.integers(1, 2**17, part).astype(np.float64) ** 3,
                       3 * rng.integers(-320, 320, part)), rng),
        # every binade of [1, 8) densely: consecutive words from random starts
        _runs(rng, part),
        # upstream's hard cases and their neighbourhoods, tiled
        _hard(part),
    ]
    out = np.empty(CHUNK, dtype=np.float64)
    for index, stream in enumerate(streams):
        out[index::8] = stream
    return out


def _near(values, rng):
    offsets = rng.integers(-4, 5, values.size).astype(np.int64)
    return (values.view(np.int64) + offsets).view(np.float64)


def _runs(rng, size):
    starts = rng.uniform(1.0, 8.0, size // 1024).view(np.int64)
    return (starts[:, None] + np.arange(1024)[None, :]).ravel().view(np.float64)


def _hard(size):
    words = np.array([float.fromhex(h) for h in _HARD]).view(np.int64)
    base = (words[:, None] + np.arange(-64, 65)[None, :]).ravel()
    values = base.view(np.float64)
    values = np.concatenate([values, -values, values * 8.0, values / 8.0])
    return np.resize(values, size)


def _exact(chunk_values):
    from woof.core.correctly_rounded_libm import cbrt_cr
    return np.array([cbrt_cr(float(v)) for v in chunk_values])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", required=True,
                        help="cbrt_batch.so built from upstream/cbrt.c")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    import cupy as cp
    from woof.core.cr_cbrt import cbrt, device_cbrt

    lib = ctypes.CDLL(args.library)
    lib.cbrt_batch.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_longlong]
    totals = {"arguments": 0, "exact_graded": 0, "gpu_vs_upstream": 0,
              "host_vs_upstream": 0, "gpu_vs_host": 0, "upstream_vs_exact": 0,
              "gpu_vs_exact": 0, "host_vs_exact": 0, "libc_cbrt_vs_exact": 0}
    first = {}
    started = time.time()
    with multiprocessing.Pool(args.workers) as pool:
        for chunk in range(args.chunks):
            x = arguments(chunk)
            upstream = np.empty_like(x)
            lib.cbrt_batch(x.ctypes.data, upstream.ctypes.data, x.size)
            gpu = cp.asnumpy(device_cbrt(cp.asarray(x)))
            host = cbrt(x)
            words = {"gpu": gpu.view(np.uint64), "host": host.view(np.uint64),
                     "upstream": upstream.view(np.uint64)}
            for a, b in (("gpu", "upstream"), ("host", "upstream"),
                         ("gpu", "host")):
                bad = np.flatnonzero(words[a] != words[b])
                totals[f"{a}_vs_{b}"] += int(bad.size)
                if bad.size and f"{a}_vs_{b}" not in first:
                    first[f"{a}_vs_{b}"] = [hex(int(x.view(np.uint64)[i]))
                                            for i in bad[:20]]
            stride = x.size // EXACT_PER_CHUNK
            sample = x[::stride][:EXACT_PER_CHUNK]
            pieces = np.array_split(sample, args.workers * 4)
            exact = np.concatenate(pool.map(_exact, pieces)).view(np.uint64)
            for name, values in (("upstream", upstream), ("gpu", gpu),
                                 ("host", host)):
                bad = values[::stride][:EXACT_PER_CHUNK].view(np.uint64) != exact
                totals[f"{name}_vs_exact"] += int(np.count_nonzero(bad))
            with np.errstate(all="ignore"):
                libc = np.cbrt(sample).view(np.uint64)
            totals["libc_cbrt_vs_exact"] += int(np.count_nonzero(libc != exact))
            totals["arguments"] += int(x.size)
            totals["exact_graded"] += int(sample.size)
            print(f"chunk {chunk}: {json.dumps(totals)}", flush=True)
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    record = {
        "totals": totals, "first_mismatches": first,
        "device": device["name"].decode(),
        "nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion())),
        "cupy": cp.__version__, "numpy": np.__version__,
        "python": sys.version.split()[0], "libc": " ".join(platform.libc_ver()),
        "machine": platform.machine(), "seconds": round(time.time() - started, 1),
    }
    print(json.dumps(record, indent=2))
    failed = any(totals[k] for k in totals
                 if k not in ("arguments", "exact_graded", "libc_cbrt_vs_exact"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
