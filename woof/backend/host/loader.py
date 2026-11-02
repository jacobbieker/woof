"""Build CUDA units as host shared libraries and launch their kernels
(``WOOF_BACKEND=host``).

:func:`load_unit` returns a :class:`HostModule` with CuPy's ``RawModule``
interface; its functions take CuPy's launch signature, ``kernel(grid,
block, args, shared_mem=0, stream=None)``, so woof's launchers run
unchanged.

The build of a unit:

1. ``extern __shared__ T name[];`` becomes a pointer to the launch's
   dynamic shared buffer.
2. The text is preprocessed by the host compiler after ``rt.h`` (CUDA's
   language on the host), with the unit's ``-D``/``-I`` options, as NVRTC
   preprocesses it.  Inline PTX is rewritten to C++ by the same rules as
   the Metal translation (:func:`woof.backend.translate.rewrite_ptx_asm`).
3. Kernels are read off the preprocessed text (``__global__`` is a marker
   there), and one launcher per kernel is appended: it unpacks the
   arguments by the kernel's own parameter types and runs the grid on the
   runtime's worker pool (``rt_pool.cpp``).
4. The result is compiled once (``-O2 -ffp-contract=off``, no fast math)
   and cached under ``~/.cache/woof/host-kernels`` (``WOOF_HOST_CACHE``),
   keyed on the source, options, headers, compiler and flags.

A unit whose kernels synchronise (``__syncthreads``, ``__syncwarp``, warp
shuffles and votes) runs each block's threads as fibers; any other unit
runs a block's threads one after another on its worker.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RT_HEADER = HERE / "rt.h"
RT_SOURCE = HERE / "rt_pool.cpp"
INCLUDE = HERE / "include"

#: The arithmetic contract: IEEE per operation, nothing contracted or
#: reassociated (CUDA's --fmad=false words).  -fwrapv and
#: -fno-strict-aliasing keep the GPU's integer wrap and type punning.
OPT_FLAGS = ("-O2",)
CXX_FLAGS = ("-std=c++17", "-fPIC", "-ffp-contract=off", "-fno-fast-math", "-fno-math-errno",
             "-fwrapv", "-fno-strict-aliasing", "-w")
ENV_CACHE = "WOOF_HOST_CACHE"
ENV_THREADS = "WOOF_HOST_THREADS"

_BARRIER = re.compile(r"\b(?:__syncthreads\w*|__syncwarp|__shfl\w*_sync|__ballot_sync|__any_sync|"
                      r"__all_sync|__activemask)\b")
_EXTERN_SHARED = re.compile(
    r"\bextern\s+__shared__\s+((?:__align__\s*\(\s*\d+\s*\)\s*|alignas\s*\(\s*\d+\s*\)\s*)?)"
    r"((?:const\s+|volatile\s+|unsigned\s+|signed\s+)*[A-Za-z_][\w:]*(?:\s*<[^<>;]*>)?)\s+"
    r"(\w+)\s*\[\s*\]\s*;")
_KERNEL = re.compile(r"\b__woof_kernel__\b")
#: A unit's own fixed-width typedefs (NVRTC has no <stdint.h>): the host's
#: C library already declares them, with a 64-bit ``long`` where the unit
#: says ``long long``, so the unit's are dropped.
_STDINT_TYPEDEF = re.compile(
    r"\btypedef\s+(?:(?:un)?signed\s+)?(?:long\s+long|long|int|short|char)(?:\s+int)?\s+"
    r"u?int(?:8|16|32|64)_t\s*;")
_UNIT_BEGIN = "#pragma woof_host_unit_begin"


class CompileError(RuntimeError):
    """The host compiler refused a unit."""


class LaunchError(RuntimeError):
    """A launch failed (bad arguments, a device assertion, a deadlock)."""


def _cache_dir() -> Path:
    env = os.environ.get(ENV_CACHE)
    if env:
        return Path(env)
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "woof" / "host-kernels"


@lru_cache(maxsize=1)
def compiler() -> str:
    """``$WOOF_HOST_CXX``, ``$CXX``, else clang++, else g++."""
    for cand in (os.environ.get("WOOF_HOST_CXX"), os.environ.get("CXX"), "clang++", "g++", "c++"):
        if cand and shutil.which(cand):
            return shutil.which(cand)
    raise CompileError("the host backend needs a C++17 compiler (clang++ or g++) on PATH")


@lru_cache(maxsize=1)
def _compiler_identity() -> str:
    cxx = compiler()
    done = subprocess.run([cxx, "--version"], capture_output=True, text=True)
    return cxx + "\n" + done.stdout.splitlines()[0] if done.stdout else cxx


def _opt_flags() -> tuple[str, ...]:
    """``WOOF_HOST_OPT`` (e.g. ``-O1``) trades run time for compile time."""
    opt = os.environ.get("WOOF_HOST_OPT", "").split()
    return tuple(opt) or OPT_FLAGS


def workers() -> int:
    env = os.environ.get(ENV_THREADS, "").strip()
    if env:
        return max(1, int(env))
    return max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


_RT_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def runtime() -> Path:
    """The worker pool library, built once per compiler; loaded globally so
    every unit library shares it."""
    with _RT_LOCK:
        key = hashlib.sha256("\0".join((RT_SOURCE.read_text(), _compiler_identity(),
                                        " ".join(CXX_FLAGS))).encode()).hexdigest()[:16]
        out = _cache_dir() / f"libwoofhostrt-{key}.so"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp.so")
            done = subprocess.run([compiler(), "-O2", *CXX_FLAGS, "-shared", str(RT_SOURCE), "-o", str(tmp),
                                   "-lpthread"], capture_output=True, text=True)
            if done.returncode != 0:
                raise CompileError(f"the host runtime did not build:\n{done.stderr[-4000:]}")
            os.replace(tmp, out)
        lib = ctypes.CDLL(str(out), mode=ctypes.RTLD_GLOBAL)
        lib.woof_rt_take_error.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.woof_rt_take_error.restype = ctypes.c_int
        _RT["lib"] = lib
        return out


_RT: dict = {}


def _take_error() -> str:
    buf = ctypes.create_string_buffer(1024)
    if _RT["lib"].woof_rt_take_error(buf, len(buf)):
        return buf.value.decode("utf-8", "replace")
    return ""


@lru_cache(maxsize=None)
def _tree_digest(directory: str) -> str:
    """Every header a unit may include from ``directory`` (once per process)."""
    h = hashlib.sha256()
    root = Path(directory)
    if root.is_dir():
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in (".h", ".cuh", ".hpp", ".cu", ".inc"):
                h.update(str(p.relative_to(root)).encode())
                h.update(p.read_bytes())
    return h.hexdigest()


def rewrite_extern_shared(text: str) -> str:
    """``extern __shared__ T name[];`` -> ``T *const name = (T *)__woof_dynamic_shared();``.
    CUDA sizes this array at launch (``shared_mem``); here it is the
    worker's dynamic shared buffer, 64-byte aligned."""
    return _EXTERN_SHARED.sub(
        lambda m: f"{m.group(2)} *const {m.group(3)} = ({m.group(2)} *)__woof_dynamic_shared();", text)


def _kernel_names(unit: str) -> list[str]:
    """Non-template ``__global__`` functions of the preprocessed unit."""
    names = []
    for m in _KERNEL.finditer(unit):
        head = unit[max(0, m.start() - 4000):m.start()]
        cut = max(head.rfind(";"), head.rfind("}"), head.rfind("{"))
        if re.search(r"\btemplate\b", head[cut + 1:]):
            continue
        tail = unit[m.end():m.end() + 2000]
        call = re.match(r"(?:\s*(?:static|inline|extern\s*\"C\"|__attribute__\s*\(\([^)]*\)\)))*"
                        r"\s*void\s+(\w+)\s*\(", tail)
        if call and call.group(1) not in names:
            names.append(call.group(1))
    return names


def _sanitize(text: str) -> str:
    return re.sub(r"\W+", "_", text)


class HostFunction:
    """One kernel: ``f(grid, block, args, shared_mem=0, stream=None)``."""

    local_size_bytes = shared_size_bytes = const_size_bytes = num_regs = 0
    ptx_version = binary_version = 0
    max_threads_per_block = 1024

    def __init__(self, module: "HostModule", name: str, ident: str):
        self.module = module
        self.name = name
        self.ptr = 0
        self._launch = getattr(module.lib, f"__woof_launch_{ident}")
        self._launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._launch.restype = ctypes.c_int
        sig = getattr(module.lib, f"__woof_sig_{ident}")
        sig.restype = ctypes.c_char_p
        self.params = [(p[0], int(p[1:])) for p in sig().decode().split(",") if p]

    @property
    def attributes(self):
        return {"max_threads_per_block": 1024, "shared_size_bytes": 0, "const_size_bytes": 0,
                "local_size_bytes": 0, "num_regs": 0, "ptx_version": 0, "binary_version": 0}

    def _pack(self, i, kind, size, value, keep):
        if kind == "p":
            if value is None:
                return ctypes.c_void_p(0)
            if isinstance(value, (bool, np.bool_)):
                raise LaunchError(f"{self.name}: argument {i} is a pointer; got a bool")
            if isinstance(value, (int, np.integer)):
                return ctypes.c_void_p(int(value))
            iface = getattr(value, "__array_interface__", None)   # a facade or NumPy array
            if iface is not None:
                keep.append(value)
                return ctypes.c_void_p(iface["data"][0])
            ptr = getattr(value, "ptr", None)                      # a MemoryPointer
            if ptr is not None:
                return ctypes.c_void_p(int(ptr))
            raise LaunchError(f"{self.name}: argument {i} is a pointer; got {type(value).__name__}")
        if kind == "s":
            if isinstance(value, (bytes, bytearray)):
                raw = bytes(value)
            elif isinstance(value, (np.ndarray, np.generic)) or hasattr(value, "__array_interface__"):
                raw = np.ascontiguousarray(np.asarray(value)).tobytes()
            else:
                try:
                    raw = bytes(memoryview(value))
                except TypeError:
                    raise LaunchError(f"{self.name}: argument {i} is a {size}-byte struct; "
                                      f"got {type(value).__name__}") from None
            if len(raw) != size:
                raise LaunchError(f"{self.name}: argument {i} is a {size}-byte struct; got {len(raw)} bytes")
            return ctypes.create_string_buffer(raw, size)
        if not isinstance(value, np.ndarray) and hasattr(value, "__array_interface__"):
            value = np.asarray(value)
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise LaunchError(f"{self.name}: argument {i} is a scalar; got an array of {value.size}")
            value = value.reshape(()).item()
        elif isinstance(value, np.generic):
            value = value.item()
        if kind == "f":
            fmt = {4: "<f", 8: "<d", 2: "<e"}[size]
            raw = struct.pack(fmt, float(value))
        elif kind == "b":
            raw = struct.pack("<?", bool(value))
        else:
            v = int(value)
            bits = 8 * size
            v &= (1 << bits) - 1                        # two's complement, as C converts
            raw = v.to_bytes(size, "little")
        return ctypes.create_string_buffer(raw, size)

    def __call__(self, grid, block, args, shared_mem=0, stream=None):
        grid = tuple(int(g) for g in (grid if isinstance(grid, (tuple, list)) else (grid,)))
        block = tuple(int(b) for b in (block if isinstance(block, (tuple, list)) else (block,)))
        grid = (grid + (1, 1, 1))[:3]
        block = (block + (1, 1, 1))[:3]
        args = tuple(args)
        if len(args) != len(self.params):
            raise LaunchError(f"{self.name}: {len(args)} arguments for {len(self.params)} parameters")
        threads = block[0] * block[1] * block[2]
        if threads > 1024 or min(block) < 1:
            raise LaunchError(f"{self.name}: block {block} is not a CUDA block (1..1024 threads)")
        if min(grid) < 1:
            raise LaunchError(f"{self.name}: grid {grid} is empty")
        keep: list = []
        slots = [self._pack(i, k, s, a, keep) for i, ((k, s), a) in enumerate(zip(self.params, args))]
        argv = (ctypes.c_void_p * max(len(slots), 1))(*[ctypes.cast(ctypes.pointer(s), ctypes.c_void_p)
                                                        for s in slots])
        record = struct.pack("<6IQiiii", *grid, *block, int(shared_mem or 0), workers(),
                             1 if self.module.ftz else 0, 1 if self.module.fibers else 0, 0)
        buf = ctypes.create_string_buffer(record, len(record))
        rc = self._launch(ctypes.addressof(buf), ctypes.addressof(argv))
        if rc:
            raise LaunchError(f"{self.name}: {_take_error() or 'launch failed'}")


class HostModule:
    """CuPy ``RawModule``'s interface over one host-built unit."""

    def __init__(self, name: str, source: str, *, options=(), name_expressions=None, ftz=None):
        from woof import numerics
        from woof.backend.translate import KERNEL_INCLUDE_DIRS, option_defines, option_include_dirs
        runtime()
        self.name = name
        effective = numerics.effective_options(tuple(options or ()))
        defines = option_defines(effective)
        include_dirs = (*option_include_dirs(effective), *map(str, KERNEL_INCLUDE_DIRS))
        if ftz is None:
            from woof.backend.metal.loader import retains_subnormals
            ftz = not retains_subnormals(name, source, options)
        self.ftz = bool(ftz)
        expressions = tuple(str(e) for e in (name_expressions or ()))
        key = hashlib.sha256("\0".join((
            source, repr(defines), repr(include_dirs), *(_tree_digest(d) for d in include_dirs),
            RT_HEADER.read_text(), _compiler_identity(), " ".join(CXX_FLAGS + _opt_flags()),
            repr(expressions))).encode()).hexdigest()[:24]
        cache = _cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        stem = f"{_sanitize(name)[:48]}-{key}"
        so, meta_path = cache / f"{stem}.so", cache / f"{stem}.json"
        if not (so.exists() and meta_path.exists()):
            self._build(source, defines, include_dirs, expressions, so, meta_path)
        meta = json.loads(meta_path.read_text())
        self.kernels: dict[str, str] = meta["kernels"]
        self.fibers = bool(meta["fibers"])
        self.path = so
        self.lib = ctypes.CDLL(str(so))
        self._functions: dict[str, HostFunction] = {}
        self.code = source
        self.options = tuple(options or ())

    def _build(self, source, defines, include_dirs, expressions, so, meta_path):
        cxx = compiler()
        text = rewrite_extern_shared(source)
        cmd = [cxx, "-E", "-x", "c++", "-std=c++17", "-DWOOF_HOST_SCAN", "-include", str(RT_HEADER),
               "-I", str(INCLUDE), *[f"-I{d}" for d in include_dirs], *[f"-D{d}" for d in defines], "-"]
        done = subprocess.run(cmd, input=(_UNIT_BEGIN + "\n" + text).encode(), capture_output=True)
        if done.returncode != 0:
            raise CompileError(f"{self.name}: preprocessing failed:\n"
                               + done.stderr.decode("utf-8", "replace")[-4000:])
        pre = done.stdout.decode("utf-8")
        at = pre.find(_UNIT_BEGIN)
        if at < 0:
            raise CompileError(f"{self.name}: the unit marker did not survive preprocessing")
        head, unit = pre[:at], pre[at + len(_UNIT_BEGIN):]
        from woof.backend.translate import TranslationError, rewrite_ptx_asm
        try:
            unit = rewrite_ptx_asm(unit, self.name)
        except TranslationError as exc:
            raise CompileError(str(exc)) from None
        unit = rewrite_extern_shared(unit)
        unit = _STDINT_TYPEDEF.sub("", unit)
        names = _kernel_names(unit)
        fibers = bool(_BARRIER.search(unit))
        kernels = {}
        tail = ["\n// ---- woof host launchers ----\n"]
        for i, expr in enumerate([*names, *expressions]):
            ident = f"{i}_{_sanitize(expr)[:40]}"
            kernels[expr] = ident
            tail.append(f'extern "C" int __woof_launch_{ident}(const woof_host::Launch *L, void **argv)'
                        f" {{ return woof_host::launch(&{expr}, L, argv); }}\n"
                        f'extern "C" const char *__woof_sig_{ident}()'
                        f" {{ return woof_host::signature(&{expr}); }}\n")
        full = "#define __woof_kernel__\n" + head + unit + "".join(tail)
        cpp = so.with_name(f"{so.stem}.{os.getpid()}.cpp")
        cpp.write_text(full, encoding="utf-8")
        tmp = so.with_name(f"{so.stem}.{os.getpid()}.tmp.so")
        rt = runtime()
        cmd = [cxx, *_opt_flags(), *CXX_FLAGS, "-shared", "-Wno-unknown-pragmas", str(cpp), "-o", str(tmp),
               str(rt), f"-Wl,-rpath,{rt.parent}", "-lm"]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            errors = [ln for ln in done.stderr.splitlines() if "error" in ln][:20] or \
                done.stderr.splitlines()[:20]
            raise CompileError(f"{self.name}: the host compiler refused the unit (source kept at {cpp}):\n"
                               + "\n".join(errors))
        cpp.unlink(missing_ok=True)
        os.replace(tmp, so)
        _atomic_write(meta_path, json.dumps({"kernels": kernels, "fibers": fibers}).encode())

    def get_function(self, name: str) -> HostFunction:
        fn = self._functions.get(name)
        if fn is None:
            ident = self.kernels.get(name)
            if ident is None:
                raise LaunchError(f"no kernel {name!r} in {self.name} "
                                  f"(kernels: {', '.join(sorted(self.kernels))})")
            fn = self._functions[name] = HostFunction(self, name, ident)
        return fn

    def get_global(self, name):
        """The address of a module global (``__device__``/``__constant__``
        variables with external linkage) as a MemoryPointer."""
        from woof.backend.hostfacade.cuda import MemoryPointer, UnownedMemory
        try:
            var = ctypes.c_char.in_dll(self.lib, name)
        except ValueError:
            raise KeyError(f"module {self.name!r} has no exported global {name!r}") from None
        return MemoryPointer(UnownedMemory(ctypes.addressof(var), 0, self), 0)

    get_global_var = get_global

    def compile(self, *a, **k):
        return None


_LOCK = threading.Lock()


@lru_cache(maxsize=None)
def _load(name: str, source: str, options: tuple, name_expressions: tuple) -> HostModule:
    return HostModule(name, source, options=options, name_expressions=name_expressions)


def load_unit(name: str, source: str, *, options=(), name_expressions=None) -> HostModule:
    with _LOCK:
        return _load(name, source, tuple(options or ()), tuple(name_expressions or ()))
