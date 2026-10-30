"""The radtran kernels compiled as host C++ (woof's CPU tier for this unit).

The exact source string the GPU loaders compile
(``woof.core.kernels.module_source("radtran_disort")``) is built with
woof/backend/host/cuda_host_shim.h and -ffp-contract=off, plus one serial
launcher per kernel; the binary64 libm is woof's own port in the source, so
the host words equal the device words wherever the compiler honours IEEE
binary64 (no contraction, correctly rounded sqrt and division).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

HOST = Path(__file__).resolve().parents[1] / "backend" / "host"
CXX_FLAGS = ("-O2", "-std=c++17", "-ffp-contract=off", "-fno-unsafe-math-optimizations",
             "-fno-fast-math", "-shared", "-fPIC", "-w")

_LAUNCHER = r"""
extern "C" void radtran_disort_host(
    int nprob, int nlyr, int nmom, int nstr, int flags,
    const double *dtau, const double *ssa, const double *pmom,
    const double *umu0, const double *fbeam, const double *albedo,
    const double *zlev, double radius, int nout, const int *outlev,
    int numu, const double *umu, int nphi, const double *phi,
    const double *ptrue, double *work, long long wstride,
    double *flux, double *rad)
{
    blockDim.x = 1; threadIdx.x = 0;
    for (int p = 0; p < nprob; ++p) {
        blockIdx.x = (unsigned)p;
        radtran_disort(0, nprob, nlyr, nmom, nstr, flags, dtau, ssa, pmom, umu0, fbeam,
                       albedo, zlev, radius, nout, outlev, numu, umu, nphi, phi, ptrue,
                       work, wstride, flux, rad);
    }
}
extern "C" long long radtran_work_size_host(int nlyr, int N, int nstr)
{
    return (long long)rt_work_size(nlyr, N, nstr);
}
"""


def _cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "woof" / "host"


@lru_cache(maxsize=None)
def library(name: str = "radtran_disort"):
    from woof.core.kernels import module_source
    src = ("#include \"cuda_host_shim.h\"\n" + module_source(name) + _LAUNCHER)
    digest = hashlib.sha256((src + " ".join(CXX_FLAGS)).encode()).hexdigest()[:16]
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    out = _cache_dir() / f"{name}-{digest}{suffix}"
    if not out.exists():
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
        if cxx is None:
            raise RuntimeError("the radtran host build needs a C++ compiler")
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            cpp = Path(td) / f"{name}.cpp"
            cpp.write_text(src)
            tmp = Path(td) / ("lib" + suffix)
            subprocess.run([cxx, *CXX_FLAGS, "-I", str(HOST), str(cpp), "-o", str(tmp)], check=True)
            shutil.copyfile(tmp, out.with_suffix(".tmp"))
            os.replace(out.with_suffix(".tmp"), out)
    lib = ctypes.CDLL(str(out))
    P = ctypes.c_void_p
    lib.radtran_disort_host.argtypes = [ctypes.c_int] * 5 + [P] * 7 + [ctypes.c_double, ctypes.c_int, P,
                                        ctypes.c_int, P, ctypes.c_int, P, P, P, ctypes.c_longlong, P, P]
    lib.radtran_disort_host.restype = None
    lib.radtran_work_size_host.argtypes = [ctypes.c_int] * 3
    lib.radtran_work_size_host.restype = ctypes.c_longlong
    return lib


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def solve(*, nstr, flags, dtau, ssa, pmom, umu0, fbeam, albedo, zlev=None, radius=6370.0,
          outlev, umu=(), phi=(), ptrue=None):
    """Run the host build over (nprob, nlyr) inputs; returns (flux, rad)."""
    lib = library()
    dtau = np.ascontiguousarray(dtau, dtype=np.float64)
    nprob, nlyr = dtau.shape
    ssa = np.ascontiguousarray(ssa, dtype=np.float64)
    pmom = np.ascontiguousarray(pmom, dtype=np.float64)
    nmom = pmom.shape[-1] - 1
    umu0 = np.ascontiguousarray(np.broadcast_to(umu0, (nprob,)), dtype=np.float64)
    fbeam = np.ascontiguousarray(np.broadcast_to(fbeam, (nprob,)), dtype=np.float64)
    albedo = np.ascontiguousarray(np.broadcast_to(albedo, (nprob,)), dtype=np.float64)
    zl = (np.zeros((nprob, nlyr + 1)) if zlev is None
          else np.ascontiguousarray(np.broadcast_to(zlev, (nprob, nlyr + 1)), dtype=np.float64))
    outlev = np.ascontiguousarray(outlev, dtype=np.int32)
    umu = np.ascontiguousarray(umu, dtype=np.float64)
    phi = np.ascontiguousarray(phi, dtype=np.float64)
    numu, nphi, nout = umu.size, phi.size, outlev.size
    pt = (np.zeros(1) if ptrue is None else np.ascontiguousarray(ptrue, dtype=np.float64))
    n_str = 2 if flags & 4 else nstr
    wstride = int(lib.radtran_work_size_host(nlyr, n_str // 2, n_str))
    work = np.zeros(wstride * nprob)
    flux = np.zeros((nprob, nout, 6))
    rad = np.zeros((nprob, nout, max(numu, 1), max(nphi, 1)))
    lib.radtran_disort_host(nprob, nlyr, nmom, nstr, flags, _ptr(dtau), _ptr(ssa), _ptr(pmom),
                            _ptr(umu0), _ptr(fbeam), _ptr(albedo), _ptr(zl), float(radius),
                            nout, _ptr(outlev), numu, _ptr(umu), nphi, _ptr(phi), _ptr(pt),
                            _ptr(work), wstride, _ptr(flux), _ptr(rad))
    return flux, rad[:, :, :numu, :nphi]
