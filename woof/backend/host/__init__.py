"""woof's host tier: CUDA sources compiled as host C++ (cuda_host_shim.h).

:func:`libm32` is glibc's binary32 libm built from woof's own word-exact
ports, for host code that must reproduce gfortran/glibc calls on a host
whose C library is not glibc (macOS).
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

HERE = Path(__file__).resolve().parent
KERNELS = HERE.parents[1] / "core" / "kernels"
CXX_FLAGS = ("-O2", "-std=c++17", "-ffp-contract=off", "-fno-unsafe-math-optimizations",
             "-shared", "-fPIC")
_SOURCES = ("libm32.cpp", "cuda_host_shim.h")
_PORTS = ("glibc_flt32.cuh", "glibc_trig_flt32.cuh", "log10_flt32.cuh")
_UNARY = ("expf", "logf", "log10f", "sinf", "cosf", "tanf", "asinf", "acosf", "atanf")


def _cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "woof" / "host"


@lru_cache(maxsize=1)
def libm32():
    """The built library (ctypes.CDLL) with woof_<name> float functions."""
    digest = hashlib.sha256()
    for name in _SOURCES:
        digest.update((HERE / name).read_bytes())
    for name in _PORTS:
        digest.update((KERNELS / name).read_bytes())
    digest.update(" ".join(CXX_FLAGS).encode())
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    out = _cache_dir() / f"libm32-{digest.hexdigest()[:16]}{suffix}"
    if not out.exists():
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("c++") or shutil.which("g++")
        if cxx is None:
            raise RuntimeError("woof's host libm needs a C++ compiler (clang++ or g++)")
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=out.parent, suffix=suffix, delete=False) as tmp:
            pass
        subprocess.run([cxx, *CXX_FLAGS, "-I", str(HERE), "-I", str(KERNELS),
                        str(HERE / "libm32.cpp"), "-o", tmp.name], check=True)
        os.replace(tmp.name, out)
    lib = ctypes.CDLL(str(out))
    for name in _UNARY:
        fn = getattr(lib, "woof_" + name)
        fn.argtypes, fn.restype = [ctypes.c_float], ctypes.c_float
    lib.woof_powf.argtypes, lib.woof_powf.restype = [ctypes.c_float, ctypes.c_float], ctypes.c_float
    return lib


def host_libc_is_glibc() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    import platform
    return platform.libc_ver()[0] == "glibc"
