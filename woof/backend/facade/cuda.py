"""``cupy.cuda`` for the facade: one device, streams, events, memory.

An Apple-silicon Mac has one GPU and unified memory.  ``Device(0)`` is it;
any other ordinal is refused by name.  Streams map to MLX streams on the GPU
device; ``synchronize`` evaluates every pending value (MLX is lazy, so
"finished" means "evaluated").  Events record wall time at the point the
stream has drained, which is what woof's timing code reads them for.
"""

from __future__ import annotations

import time
import types

import numpy as np


def _mx():
    import mlx.core as mx
    return mx


def _drain():
    from woof.backend.facade._array import PENDING
    mx = _mx()
    arrays = [s.arr for s in PENDING]
    PENDING.clear()
    if arrays:
        mx.eval(*arrays)
    mx.synchronize()


class Device:
    def __init__(self, device=None):
        ordinal = 0 if device is None else int(device)
        if ordinal != 0:
            raise RuntimeError(f"the mlx backend has one device; ordinal {ordinal} does not exist")
        self.id = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def use(self):
        return self

    def synchronize(self):
        _drain()

    @property
    def compute_capability(self):
        return "metal"

    @property
    def mem_info(self):
        return runtime.memGetInfo()

    @property
    def attributes(self):
        return _attributes()

    def __int__(self):
        return 0

    def __eq__(self, other):
        return isinstance(other, Device) and other.id == self.id

    def __hash__(self):
        return 0


def _gpu_core_count() -> int:
    """Apple GPU cores (the IORegistry's gpu-core-count): the unit woof's
    launchers size per-SM tiles against."""
    import re
    import subprocess
    try:
        out = subprocess.run(["ioreg", "-l", "-k", "gpu-core-count"], capture_output=True,
                             timeout=10).stdout.decode("utf-8", "replace")
        m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', out)
        if m:
            return int(m.group(1))
    except (OSError, subprocess.SubprocessError):
        pass
    return 8  # the smallest Apple-silicon GPU


_ATTRIBUTES = None


def _attributes() -> dict:
    """CUDA's device attributes, as an Apple GPU answers them: one GPU core
    per "multiprocessor", SIMD-groups of 32 as warps, the 1024-thread
    threadgroup limit as both the block and the per-core resident bound."""
    global _ATTRIBUTES
    if _ATTRIBUTES is None:
        _ATTRIBUTES = {
            "MultiProcessorCount": _gpu_core_count(),
            "MaxThreadsPerMultiProcessor": 1024,
            "MaxThreadsPerBlock": 1024,
            "WarpSize": 32,
            "MaxRegistersPerMultiprocessor": 65536,
            "MaxRegistersPerBlock": 65536,
            "MaxSharedMemoryPerBlock": 32768,
            "MaxSharedMemoryPerMultiprocessor": 32768,
            "MaxBlockDimX": 1024, "MaxBlockDimY": 1024, "MaxBlockDimZ": 64,
            "MaxGridDimX": 2**31 - 1, "MaxGridDimY": 65535, "MaxGridDimZ": 65535,
            "ComputeCapabilityMajor": 0, "ComputeCapabilityMinor": 0,
            "UnifiedAddressing": 1, "ManagedMemory": 1, "ConcurrentKernels": 1,
        }
    return _ATTRIBUTES


class Stream:
    _current: "Stream | None" = None

    def __init__(self, null=False, non_blocking=False, ptr=0, priority=0):
        self._null = null
        self.ptr = 0 if null else id(self)
        # one Metal queue serves every stream: the priority is recorded only
        self.priority = int(priority)

    def __enter__(self):
        self._prev = Stream._current
        Stream._current = self
        return self

    def __exit__(self, *exc):
        Stream._current = self._prev
        return False

    def use(self):
        Stream._current = self
        return self

    def synchronize(self):
        _drain()

    def wait_event(self, event):
        return None

    def record(self, event=None):
        event = event or Event()
        event.record(self)
        return event

    @property
    def done(self):
        return True

    def is_capturing(self):
        # CUDA-graph capture: never on Metal (MLX records its own lazy graph)
        return False

    def begin_capture(self, mode=None):
        self._capturing = True

    def end_capture(self):
        self._capturing = False
        return _Graph()

    def launch_host_func(self, callback, arg):
        _drain()
        callback(arg)


Stream.null = Stream(null=True)


def get_current_stream(device_id=None):
    return Stream._current or Stream.null


class Event:
    def __init__(self, block=False, disable_timing=False, interprocess=False):
        self.t = None
        self.ptr = id(self)

    def record(self, stream=None):
        _drain()
        self.t = time.perf_counter()

    def synchronize(self):
        _drain()

    @property
    def done(self):
        return True


def get_elapsed_time(start: Event, end: Event) -> float:
    if start.t is None or end.t is None:
        raise RuntimeError("both events must be recorded")
    return (end.t - start.t) * 1e3


def _device_info():
    mx = _mx()
    getter = getattr(mx, "device_info", None) or mx.metal.device_info
    return getter()


def _mem_get_info():
    mx = _mx()
    info = _device_info()
    total = int(info.get("max_recommended_working_set_size") or info.get("memory_size"))
    used = int(mx.get_active_memory()) if hasattr(mx, "get_active_memory") else 0
    return max(total - used, 0), total


def _properties(device=0):
    info = _device_info()
    return {
        "name": str(info.get("device_name", "Apple GPU")).encode(),
        "totalGlobalMem": int(info.get("memory_size", 0)),
        "major": 0, "minor": 0,
        "multiProcessorCount": _attributes()["MultiProcessorCount"],
        "maxThreadsPerBlock": 1024,
        # resident threads per GPU core: only local-memory frames are priced
        # with it (woof.core.preflight), and Metal reports no CUDA frame
        "maxThreadsPerMultiProcessor": 1024,
        "warpSize": 32,
        "sharedMemPerBlock": 32768,
        "architecture": str(info.get("architecture", "")),
    }


#: hostAlloc's blocks by host address: (host view, storage).  Memory is
#: unified, so a block is one Metal buffer seen two ways -- a NumPy view of
#: its bytes, and the facade storage kernels bind -- which is CUDA's mapped
#: page-locked memory (cudaHostAllocMapped) exactly.  It lives until freeHost.
_HOST_BLOCKS: dict = {}

#: cudaMemoryType
_MEMORY_UNREGISTERED, _MEMORY_HOST, _MEMORY_DEVICE = 0, 1, 2


def _host_alloc(nbytes, flags=0):
    import mlx.core as mx
    from woof.backend.facade._array import ndarray
    block = ndarray((max(int(nbytes), 1),), np.uint8)
    st = block._st
    mx.eval(st.arr)
    host = np.asarray(memoryview(st.arr))
    ptr = int(host.ctypes.data)
    st.address()                  # registered, so its GPU address resolves (UnownedMemory)
    _HOST_BLOCKS[ptr] = (host, st)
    return ptr


def _free_host(ptr):
    _HOST_BLOCKS.pop(int(ptr), None)


def _host_block_at(ptr):
    for base, (host, st) in _HOST_BLOCKS.items():
        if base <= ptr < base + host.nbytes:
            return base, st
    return None


def _pointer_attributes(ptr):
    """cudaPointerGetAttributes: a hostAlloc block is page-locked host memory
    with a device address; a storage's GPU address is device memory."""
    from woof.backend.facade._array import storage_at
    ptr = int(ptr)
    found = _host_block_at(ptr)
    if found is not None:
        base, st = found
        return types.SimpleNamespace(type=_MEMORY_HOST, device=0, hostPointer=ptr,
                                     devicePointer=st.address() + (ptr - base))
    if storage_at(ptr) is not None:
        return types.SimpleNamespace(type=_MEMORY_DEVICE, device=0, hostPointer=0, devicePointer=ptr)
    return types.SimpleNamespace(type=_MEMORY_UNREGISTERED, device=-2, hostPointer=0, devicePointer=0)


#: cudaMemcpyKind
_MEMCPY_KINDS = dict(memcpyHostToHost=0, memcpyHostToDevice=1, memcpyDeviceToHost=2,
                     memcpyDeviceToDevice=3, memcpyDefault=4)


def _bytes_view(ptr, nbytes):
    """``nbytes`` at address ``ptr``: a uint8 facade view of the storage that
    handed the address out, or None for host memory."""
    from woof.backend.facade._array import ndarray, storage_at
    found = storage_at(int(ptr))
    if found is None:
        return None
    st, offset = found
    if offset + nbytes > st.nbytes:
        raise ValueError("cudaMemcpy range runs past the end of its device allocation")
    return ndarray((int(nbytes),), np.uint8, _storage=st, _offset=offset)


def _memcpy(dst, src, size, kind=4):
    """cudaMemcpy over raw addresses: a device side is a facade storage's GPU
    address (``a.data.ptr``), a host side any process address.  Memory is
    unified and the facade orders work itself, so the copy is complete when
    this returns; the kind is implied by the addresses (cudaMemcpyDefault)."""
    import ctypes
    size = int(size)
    if size <= 0:
        return
    source = _bytes_view(src, size)
    data = source.get() if source is not None else \
        np.frombuffer((ctypes.c_char * size).from_address(int(src)), np.uint8).copy()
    target = _bytes_view(dst, size)
    if target is not None:
        target.set(data)
    else:
        ctypes.memmove(int(dst), data.ctypes.data, size)


def _memcpy_async(dst, src, size, kind=4, stream=0):
    _memcpy(dst, src, size, kind)


def _memset(ptr, value, size):
    view = _bytes_view(ptr, int(size))
    if view is None:
        import ctypes
        ctypes.memset(int(ptr), int(value), int(size))
    else:
        view.fill(np.uint8(int(value) & 0xFF))


runtime = types.SimpleNamespace(
    memcpy=_memcpy, memcpyAsync=_memcpy_async, memset=_memset,
    memsetAsync=lambda ptr, value, size, stream=0: _memset(ptr, value, size),
    **_MEMCPY_KINDS,
    getDevice=lambda: 0,
    setDevice=lambda d: Device(d) and None,
    getDeviceCount=lambda: 1,
    memGetInfo=_mem_get_info,
    deviceSynchronize=_drain,
    hostAlloc=_host_alloc,
    pointerGetAttributes=_pointer_attributes,
    memoryTypeHost=_MEMORY_HOST,
    memoryTypeDevice=_MEMORY_DEVICE,
    # cudaDeviceGetLimit: Metal has no CUDA stack/heap limit to report (0)
    deviceGetLimit=lambda limit: 0,
    freeHost=_free_host,
    getDeviceProperties=_properties,
    runtimeGetVersion=lambda: 0,
    driverGetVersion=lambda: 0,
    is_hip=False,
    CUDARuntimeError=RuntimeError,
)


class _MemoryPool:
    def __init__(self, allocator=None):
        # CuPy's pool draws on ``allocator``; MLX's allocator is the facade's only one
        self.allocator = allocator

    def used_bytes(self):
        mx = _mx()
        return int(mx.get_active_memory()) if hasattr(mx, "get_active_memory") else 0

    def total_bytes(self):
        mx = _mx()
        return self.used_bytes() + (int(mx.get_cache_memory()) if hasattr(mx, "get_cache_memory") else 0)

    def free_all_blocks(self, stream=None):
        mx = _mx()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    def set_limit(self, size=None, fraction=None):
        mx = _mx()
        if size is not None and hasattr(mx, "set_memory_limit"):
            mx.set_memory_limit(int(size))

    def get_limit(self):
        return 0

    def n_free_blocks(self):
        return 0

    def free_bytes(self):
        """Bytes the pool holds but does not lend: MLX's buffer cache."""
        mx = _mx()
        return int(mx.get_cache_memory()) if hasattr(mx, "get_cache_memory") else 0

    def malloc(self, size):
        """A fresh allocation of ``size`` bytes, as a MemoryPointer."""
        from woof.backend.facade._array import MemoryPointer, ndarray
        a = ndarray((max(int(size), 1),), np.uint8)
        return MemoryPointer(a._st, 0, int(size))


_POOL = _MemoryPool()
_PINNED = _MemoryPool()


def get_default_memory_pool():
    return _POOL


def get_default_pinned_memory_pool():
    return _PINNED


class PinnedMemoryPointer:
    """``cupy.cuda.alloc_pinned_memory``'s result: a hostAlloc block (so
    ``pointerGetAttributes`` reports page-locked host memory, as CuPy's
    pinned pool's blocks are), exported through the buffer protocol."""

    def __init__(self, nbytes):
        import weakref
        self.ptr = _host_alloc(max(int(nbytes), 1))
        self.size = int(nbytes)
        self._host = _HOST_BLOCKS[self.ptr][0][:self.size]
        weakref.finalize(self, _free_host, self.ptr)

    def __buffer__(self, flags):
        return memoryview(self._host)

    def __len__(self):
        return self.size

    def __int__(self):
        return self.ptr


def alloc_pinned_memory(nbytes):
    return PinnedMemoryPointer(nbytes)


def set_allocator(allocator=None):
    return None


def set_pinned_memory_allocator(allocator=None):
    return None


class _Graph:
    """CUDA-graph capture is a no-op: MLX already records a lazy graph and
    encodes it in batches."""

    def launch(self, stream=None):
        return None

    def upload(self, stream=None):
        return None


def is_available():
    return True


profiler = types.SimpleNamespace(start=lambda: None, stop=lambda: None)
nvtx = types.SimpleNamespace(RangePush=lambda *a, **k: None, RangePop=lambda: None,
                             Mark=lambda *a, **k: None)
memory = types.SimpleNamespace(MemoryPool=_MemoryPool, OutOfMemoryError=MemoryError,
                               set_allocator=set_allocator)
MemoryPool = _MemoryPool
PinnedMemoryPool = _MemoryPool


def get_allocator():
    return None


class using_allocator:
    """``with cp.cuda.using_allocator(alloc):`` -- MLX owns allocation; the
    allocator is not consulted."""

    def __init__(self, allocator=None):
        self.allocator = allocator

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
def _compile_using_nvrtc(source, options=(), arch=None, filename="kern.cu", name_expressions=None,
                         log_stream=None, cache_in_memory=False, jitify=False, method=None):
    """CuPy's direct NVRTC call: a translated image and no name mapping.
    The unit is translated and its Metal library built here, so a unit that
    does not compile fails at this call, as it does under NVRTC; loading the
    image (``function.Module().load``) then finds the library built."""
    if name_expressions:
        raise AttributeError("NVRTC name expressions are not supported by woof's mlx backend")
    image = translated_image(source, tuple(options or ()), filename)
    _FunctionModule().load(image)
    return image, None


def _compile_with_cache_cuda(source, options, arch=None, cache_dir=None, extra_source=None,
                             backend="nvrtc", enable_cooperative_groups=False, name_expressions=None,
                             log_stream=None, jitify=False, *args, **kwargs):
    """CuPy's cached compile behind every RawModule: the translated module."""
    module = _FunctionModule()
    module.load(translated_image(source, tuple(options or ()), "kern.cu"))
    return module


class _NVRTCProgram:
    """Present so woof's NVRTC option hooks (woof.numerics, woof.wrf_exact,
    woof.cuda_subnormals) install; the facade never runs NVRTC."""

    def __init__(self, src, name="kern.cu", *args, **kwargs):
        self.src = src

    def compile(self, options=(), log_stream=None):
        raise AttributeError("NVRTC is not available on woof's mlx backend")


compiler = types.SimpleNamespace(
    CompileException=RuntimeError, compile_using_nvrtc=_compile_using_nvrtc,
    # CuPy's raw route under compile_using_nvrtc (woof.kernel_aot wraps it)
    _compile_using_nvrtc_no_warning=_compile_using_nvrtc,
    _compile_with_cache_cuda=_compile_with_cache_cuda, _NVRTCProgram=_NVRTCProgram,
    _get_arch=lambda: "metal",
    # the architecture translated units are preprocessed for (__CUDA_ARCH__)
    _get_arch_for_options_for_nvrtc=lambda arch=None: _nvrtc_arch())


def _nvrtc_arch():
    from woof.backend.translate import CUDA_ARCH
    return f"-arch=compute_{CUDA_ARCH // 10}", "ptx"


def _func_get_attribute(attribute, function_ptr):
    """CUfunction attributes: Metal has no CUDA local frame, registers or
    PTX version to report; 0 for each (max threads: the pipeline's own
    limit is checked at launch)."""
    return 0


driver = types.SimpleNamespace(
    CUDADriverError=RuntimeError, funcGetAttribute=_func_get_attribute,
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK=0, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES=1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES=2, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES=3,
    CU_FUNC_ATTRIBUTE_NUM_REGS=4, CU_FUNC_ATTRIBUTE_PTX_VERSION=5,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION=6)

class _NvrtcProgram:
    def __init__(self, source, name, headers=(), include_names=()):
        self.source, self.name = source, name or "kern.cu"
        self.headers = dict(zip(include_names or (), headers or ()))
        self.image, self.log = None, ""


def _compile_program(program, options=()):
    """nvrtcCompileProgram: translate and build the unit (the image is what
    getPTX/getCUBIN hand back; ``function.Module().load`` takes it).  The
    in-memory headers of nvrtcCreateProgram are written to a directory the
    preprocessor searches."""
    options = tuple(str(o) for o in options or ())
    if program.headers:
        import hashlib
        import tempfile
        from pathlib import Path
        digest = hashlib.sha256(repr(sorted(program.headers.items())).encode()).hexdigest()[:16]
        root = Path(tempfile.gettempdir()) / f"woof-nvrtc-headers-{digest}"
        for name, text in program.headers.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text if isinstance(text, str) else text.decode())
        options += (f"-I{root}",)
    try:
        program.image = _compile_using_nvrtc(program.source, options, filename=program.name)[0]
    except Exception as exc:
        program.log = str(exc)
        raise RuntimeError(f"NVRTC_ERROR_COMPILATION (woof mlx backend): {exc}") from exc


#: No NVRTC: version (0, 0) so a receipt header cannot be mistaken for a
#: CUDA build (the facade's cupy.__version__ names the backend).  The
#: program API compiles through the translator.
nvrtc = types.SimpleNamespace(
    getVersion=lambda: (0, 0), NVRTCError=RuntimeError,
    createProgram=_NvrtcProgram, compileProgram=_compile_program,
    getPTX=lambda program: program.image, getCUBIN=lambda program: program.image,
    getProgramLog=lambda program: program.log, destroyProgram=lambda program: None)


# ---------------------------------------------------------------- direct NVRTC route

_MARK = b"// woof-metal-image "


def translated_image(source: str, options: tuple, filename: str = "kern.cu") -> bytes:
    """What woof.nvrtc_ptx_cache.compile_using_nvrtc returns on a facade
    backend: PTX-looking (callers test ``startswith(b"//")``) and
    self-contained -- the source and options travel compressed inside it,
    so an image read back from any cache still loads."""
    import hashlib
    import json
    import zlib
    payload = zlib.compress(json.dumps({"source": source, "options": list(options),
                                        "filename": filename}).encode())
    digest = hashlib.sha256(payload).hexdigest()
    return _MARK + digest.encode() + b"\n" + payload


def _read_image(image: bytes):
    import json
    import zlib
    head, _, payload = image.partition(b"\n")
    if not head.startswith(_MARK):
        raise RuntimeError("the mlx backend loads only images from woof's compile route "
                           "(compiled PTX/cubin cannot run on Metal)")
    data = json.loads(zlib.decompress(payload))
    return data["source"], tuple(data["options"]), data["filename"], head[len(_MARK):].decode()


class _FunctionModule:
    """``cupy.cuda.function.Module``: load(image) -> a translated module."""

    def __init__(self):
        self._module = None

    def load(self, image):
        if isinstance(image, str):
            image = image.encode()
        source, options, filename, digest = _read_image(bytes(image))
        from woof.backend.metal.loader import load_unit
        self._module = load_unit(filename.rsplit(".", 1)[0] + ":" + digest[:12], source, options=options)

    def load_file(self, path):
        with open(path, "rb") as f:
            self.load(f.read())

    def get_function(self, name):
        return self._module.get_function(name)

    def get_global_var(self, name):
        return self._module.get_global(name)

    get_global = get_global_var


function = types.SimpleNamespace(Module=_FunctionModule, Function=object)


def UnownedMemory(ptr, size, owner, device_id=-1):
    """CuPy wraps a raw address.  The facade's module globals are already
    MemoryPointers (loader.get_global_var), so that pointer is the memory;
    an integer must be a GPU address a storage handed out (a hostAlloc
    block's device pointer, or ``a.data.ptr``).  A hostAlloc block is
    untyped bytes, so an ndarray over it takes the dtype it asks for."""
    from woof.backend.facade._array import MemoryPointer, storage_at
    if isinstance(ptr, MemoryPointer):
        return ptr
    found = storage_at(int(ptr))
    if found is None:
        raise AttributeError("cupy.cuda.UnownedMemory over an address no facade storage handed out")
    st, offset = found
    return MemoryPointer(st, offset, int(size),
                         untyped=any(block is st for _host, block in _HOST_BLOCKS.values()))


def _memory_pointer(mem, offset=0):
    """``cupy.cuda.MemoryPointer(mem, offset)``: over facade storage, or over
    a MemoryPointer standing for its memory (UnownedMemory above)."""
    from woof.backend.facade._array import MemoryPointer
    if isinstance(mem, MemoryPointer):
        return MemoryPointer(mem.storage, mem.offset + int(offset), untyped=mem.untyped)
    return MemoryPointer(mem, offset)


def __getattr__(name):
    if name == "MemoryPointer":
        return _memory_pointer
    raise AttributeError(f"cupy.cuda.{name} is not implemented by woof's mlx backend")


# `import cupy.cuda.compiler` / `from cupy.cuda.runtime import ...`: Python
# finds a submodule already in sys.modules without asking the parent for a
# package path.
import sys as _sys  # noqa: E402

for _name in ("compiler", "runtime", "driver", "nvrtc"):
    _sys.modules.setdefault(f"cupy.cuda.{_name}", globals()[_name])
# CuPy's low-level bindings package, by the names woof imports them
for _name, _value in (("cupy_backends.cuda.api.runtime", runtime), ("cupy_backends.cuda.api.driver", driver),
                      ("cupy_backends.cuda.libs.nvrtc", nvrtc)):
    _sys.modules.setdefault(_name, _value)
