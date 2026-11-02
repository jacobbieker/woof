"""``cupy.cuda`` for the host facade: one "device" (the host), synchronous
streams, events, and raw memory over host addresses.

Every launch has finished when its call returns (woof.backend.host.loader),
so streams and events order nothing: ``synchronize`` returns at once and an
event records the wall time it was recorded at.  Device memory is host
memory, so a device pointer is a process address and ``cudaMemcpy`` of any
kind is ``memmove``.
"""

from __future__ import annotations

import ctypes
import os
import time
import types

import numpy as np

_PCI_BUS_ID = "0000:00:00.0"


def _workers():
    from woof.backend.host.loader import workers
    return workers()


class Device:
    def __init__(self, device=None):
        ordinal = 0 if device is None else int(device)
        if ordinal != 0:
            raise RuntimeError(f"the host backend has one device; ordinal {ordinal} does not exist")
        self.id = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def use(self):
        return self

    def synchronize(self):
        return None

    @property
    def compute_capability(self):
        return "host"

    @property
    def pci_bus_id(self):
        return _PCI_BUS_ID

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


def _attributes() -> dict:
    """CUDA's device attributes as the host answers them: one worker thread
    per "multiprocessor", CUDA's block limits (a block's threads run on one
    worker, so any CUDA-legal block runs)."""
    return {
        "MultiProcessorCount": _workers(),
        "MaxThreadsPerMultiProcessor": 1024,
        "MaxThreadsPerBlock": 1024,
        "WarpSize": 32,
        "MaxRegistersPerMultiprocessor": 65536,
        "MaxRegistersPerBlock": 65536,
        "MaxSharedMemoryPerBlock": 49152,
        "MaxSharedMemoryPerBlockOptin": 232448,
        "MaxSharedMemoryPerMultiprocessor": 233472,
        "MaxBlockDimX": 1024, "MaxBlockDimY": 1024, "MaxBlockDimZ": 64,
        "MaxGridDimX": 2**31 - 1, "MaxGridDimY": 65535, "MaxGridDimZ": 65535,
        "ComputeCapabilityMajor": 0, "ComputeCapabilityMinor": 0,
        "UnifiedAddressing": 1, "ManagedMemory": 1, "ConcurrentKernels": 1,
    }


class Stream:
    _current: "Stream | None" = None

    def __init__(self, null=False, non_blocking=False, ptr=0, priority=0):
        self._null = null
        self.ptr = 0 if null else id(self)
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
        return None

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
        return False

    def begin_capture(self, mode=None):
        self._capturing = True

    def end_capture(self):
        self._capturing = False
        return _Graph()

    def launch_host_func(self, callback, arg):
        callback(arg)


Stream.null = Stream(null=True)
ExternalStream = Stream


def get_current_stream(device_id=None):
    return Stream._current or Stream.null


class Event:
    def __init__(self, block=False, disable_timing=False, interprocess=False):
        self.t = None
        self.ptr = id(self)

    def record(self, stream=None):
        self.t = time.perf_counter()

    def synchronize(self):
        return None

    @property
    def done(self):
        return True


def get_elapsed_time(start: Event, end: Event) -> float:
    if start.t is None or end.t is None:
        raise RuntimeError("both events must be recorded")
    return (end.t - start.t) * 1e3


def _meminfo():
    total = avail = 0
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key == "MemTotal":
                    total = int(rest.split()[0]) * 1024
                elif key == "MemAvailable":
                    avail = int(rest.split()[0]) * 1024
    except OSError:
        pass
    return avail, total


def _properties(device=0):
    import platform
    _, total = _meminfo()
    return {
        "name": f"woof host backend ({platform.machine()})".encode(),
        "totalGlobalMem": total,
        "major": 0, "minor": 0,
        "multiProcessorCount": _workers(),
        "maxThreadsPerBlock": 1024,
        "maxThreadsPerMultiProcessor": 1024,
        "warpSize": 32,
        "sharedMemPerBlock": 49152,
    }


# ---------------------------------------------------------------- memory

class _Memory:
    """An allocation: its address, size, and the object that owns the bytes."""

    def __init__(self, ptr, size, owner=None, device_id=0):
        self.ptr = int(ptr)
        self.size = int(size)
        self.owner = owner
        self.device_id = device_id

    def __int__(self):
        return self.ptr


def UnownedMemory(ptr, size, owner, device_id=-1):
    if isinstance(ptr, MemoryPointer):
        return ptr.mem
    return _Memory(int(ptr), size, owner, device_id)


def _root(a):
    """The array that owns ``a``'s buffer (the end of its ``base`` chain)."""
    while isinstance(a.base, np.ndarray):
        a = a.base
    return a


def _memory_of(a):
    """(allocation, byte offset) of an array: one allocation object per
    buffer, so every view's ``x.data.mem`` is the same object, as CuPy's."""
    root = _root(a)
    mem = getattr(root, "_woof_mem", None)
    if mem is None:
        owner = getattr(root, "_owner", None)
        if owner is not None:
            mem = owner.mem
        else:
            mem = _Memory(root.__array_interface__["data"][0], root.nbytes, root)
        try:
            root._woof_mem = mem
        except AttributeError:          # a plain NumPy root keeps no attributes
            pass
    return mem, a.__array_interface__["data"][0] - mem.ptr


class MemoryPointer:
    """``cupy.cuda.MemoryPointer(mem, offset)``: an address in an allocation."""

    def __init__(self, mem, offset=0):
        if isinstance(mem, MemoryPointer):
            offset += mem.offset
            mem = mem.mem
        self.mem = mem
        self.offset = int(offset)

    @property
    def ptr(self):
        return self.mem.ptr + self.offset

    @property
    def device_id(self):
        return 0

    def __int__(self):
        return self.ptr

    def __add__(self, n):
        return MemoryPointer(self.mem, self.offset + int(n))

    def copy_from_host(self, src, size):
        ctypes.memmove(self.ptr, int(_address(src)), int(size))

    def copy_to_host(self, dst, size):
        ctypes.memmove(int(_address(dst)), self.ptr, int(size))

    copy_from_host_async = copy_from_host
    copy_to_host_async = copy_to_host

    def copy_from_device(self, src, size):
        ctypes.memmove(self.ptr, int(_address(src)), int(size))

    copy_from_device_async = copy_from_device
    copy_from = copy_from_device
    copy_from_async = copy_from_device

    def memset(self, value, size):
        ctypes.memset(self.ptr, int(value), int(size))

    memset_async = memset


def _address(x):
    if isinstance(x, np.ndarray):
        return x.__array_interface__["data"][0]
    if isinstance(x, MemoryPointer):
        return x.ptr
    if isinstance(x, ctypes.c_void_p):
        return x.value or 0
    return int(x)


#: hostAlloc's blocks by address: the NumPy buffer that owns the bytes.
_HOST_BLOCKS: dict = {}

#: cudaMemoryType
_MEMORY_UNREGISTERED, _MEMORY_HOST, _MEMORY_DEVICE = 0, 1, 2


def _host_alloc(nbytes, flags=0):
    block = np.empty(max(int(nbytes), 1), np.uint8)
    ptr = block.__array_interface__["data"][0]
    _HOST_BLOCKS[ptr] = block
    return ptr


def _free_host(ptr, block=None):
    entry = _HOST_BLOCKS.get(int(ptr))
    if entry is not None and (block is None or entry is block):
        del _HOST_BLOCKS[int(ptr)]


def _pointer_attributes(ptr):
    """A hostAlloc block is page-locked host memory mapped at the same
    address; any other address is "device" memory (a facade array)."""
    ptr = int(ptr)
    for base, block in _HOST_BLOCKS.items():
        if base <= ptr < base + block.nbytes:
            return types.SimpleNamespace(type=_MEMORY_HOST, device=0, hostPointer=ptr, devicePointer=ptr)
    return types.SimpleNamespace(type=_MEMORY_DEVICE, device=0, hostPointer=0, devicePointer=ptr)


def _memcpy(dst, src, size, kind=4):
    if int(size) > 0:
        ctypes.memmove(_address(dst), _address(src), int(size))


def _copy_boxes(dst, dst_pitch, dst_plane, src, src_pitch, src_plane, width, height, depth):
    for z in range(depth):
        for y in range(height):
            ctypes.memmove(dst + z * dst_plane + y * dst_pitch, src + z * src_plane + y * src_pitch, width)


def _memcpy2d(dst, dpitch, src, spitch, width, height, kind=4, stream=0):
    _copy_boxes(_address(dst), int(dpitch), 0, _address(src), int(spitch), 0, int(width), int(height), 1)


def _memcpy3d(params, stream=0):
    """cudaMemcpy3D(Async) from a pointer to cudaMemcpy3DParms (x86-64 layout,
    pitched pointers only)."""
    words = (ctypes.c_size_t * 20).from_address(int(params))
    if words[0] or words[8]:
        raise AttributeError("cudaMemcpy3D with CUDA arrays is not supported by woof's host backend")
    (sx, sy, sz), (sptr, spitch, _sxs, sysize) = words[1:4], words[4:8]
    (dx, dy, dz), (dptr, dpitch, _dxs, dysize) = words[9:12], words[12:16]
    width, height, depth = words[16:19]
    _copy_boxes(dptr + dz * dpitch * dysize + dy * dpitch + dx, dpitch, dpitch * dysize,
                sptr + sz * spitch * sysize + sy * spitch + sx, spitch, spitch * sysize,
                width, height, depth)


def _memset(ptr, value, size):
    ctypes.memset(_address(ptr), int(value) & 0xFF, int(size))


runtime = types.SimpleNamespace(
    memcpy=_memcpy, memcpyAsync=lambda dst, src, size, kind=4, stream=0: _memcpy(dst, src, size, kind),
    memset=_memset,
    memcpy2D=_memcpy2d, memcpy2DAsync=_memcpy2d, memcpy3D=_memcpy3d, memcpy3DAsync=_memcpy3d,
    memsetAsync=lambda ptr, value, size, stream=0: _memset(ptr, value, size),
    memcpyHostToHost=0, memcpyHostToDevice=1, memcpyDeviceToHost=2, memcpyDeviceToDevice=3,
    memcpyDefault=4,
    getDevice=lambda: 0,
    setDevice=lambda d: Device(d) and None,
    getDeviceCount=lambda: 1,
    memGetInfo=_meminfo,
    deviceSynchronize=lambda: None,
    hostAlloc=_host_alloc,
    pointerGetAttributes=_pointer_attributes,
    memoryTypeHost=_MEMORY_HOST,
    memoryTypeDevice=_MEMORY_DEVICE,
    deviceGetLimit=lambda limit: 0,
    deviceSetLimit=lambda limit, value: None,
    freeHost=_free_host,
    getDeviceProperties=_properties,
    runtimeGetVersion=lambda: 0,
    driverGetVersion=lambda: 0,
    deviceGetPCIBusId=lambda device=0: _PCI_BUS_ID,
    memcpyPeer=lambda dst, dst_dev, src, src_dev, size: _memcpy(dst, src, size, 3),
    memcpyPeerAsync=lambda dst, dst_dev, src, src_dev, size, stream=0: _memcpy(dst, src, size, 3),
    is_hip=False,
    CUDARuntimeError=RuntimeError,
)


class _MemoryPool:
    """CuPy's pool: host memory is NumPy's, so the pool lends nothing and
    holds nothing (CuPy's own counters would read 0 on a fresh pool)."""

    def __init__(self, allocator=None):
        self.allocator = allocator

    def used_bytes(self):
        return 0

    def total_bytes(self):
        return 0

    def free_all_blocks(self, stream=None):
        return None

    def set_limit(self, size=None, fraction=None):
        return None

    def get_limit(self):
        return 0

    def n_free_blocks(self):
        return 0

    def free_bytes(self):
        return 0

    def malloc(self, size):
        block = np.empty(max(int(size), 1), np.uint8)
        return MemoryPointer(_Memory(block.__array_interface__["data"][0], int(size), block), 0)


_POOL = _MemoryPool()
_PINNED = _MemoryPool()


def get_default_memory_pool():
    return _POOL


def get_default_pinned_memory_pool():
    return _PINNED


class PinnedMemoryPointer:
    """``cupy.cuda.alloc_pinned_memory``'s result: host bytes, exported
    through the buffer protocol."""

    def __init__(self, nbytes):
        self._host = np.empty(max(int(nbytes), 1), np.uint8)
        self.ptr = self._host.__array_interface__["data"][0]
        self.size = int(nbytes)
        self._host = self._host[:self.size]

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


def get_allocator():
    return None


class using_allocator:
    def __init__(self, allocator=None):
        self.allocator = allocator

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class MemoryHook:
    """CuPy's memory hook; the host facade reports no allocations."""
    name = "MemoryHook"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def malloc_preprocess(self, **kw):
        pass

    def malloc_postprocess(self, **kw):
        pass

    def alloc_preprocess(self, **kw):
        pass

    def alloc_postprocess(self, **kw):
        pass

    def free_preprocess(self, **kw):
        pass

    def free_postprocess(self, **kw):
        pass


class _Graph:
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
                               set_allocator=set_allocator, MemoryPointer=MemoryPointer,
                               UnownedMemory=UnownedMemory)
MemoryPool = _MemoryPool
PinnedMemoryPool = _MemoryPool
OutOfMemoryError = MemoryError


# ---------------------------------------------------------------- the compile route

_MARK = b"// woof-host-image "


def host_image(source: str, options: tuple, filename: str = "kern.cu",
               name_expressions=None) -> bytes:
    """What woof.nvrtc_ptx_cache.compile_using_nvrtc returns on the host
    backend: PTX-looking (callers test ``startswith(b"//")``) and
    self-contained, so an image read back from any cache still loads."""
    import hashlib
    import json
    import zlib
    payload = zlib.compress(json.dumps({"source": source, "options": list(options), "filename": filename,
                                        "name_expressions": list(name_expressions or ())}).encode())
    return _MARK + hashlib.sha256(payload).hexdigest().encode() + b"\n" + payload


def _read_image(image: bytes):
    import json
    import zlib
    head, _, payload = image.partition(b"\n")
    if not head.startswith(_MARK):
        raise RuntimeError("the host backend loads only images from woof's compile route "
                           "(compiled PTX/cubin cannot run on the host)")
    data = json.loads(zlib.decompress(payload))
    return (data["source"], tuple(data["options"]), data["filename"], head[len(_MARK):].decode(),
            tuple(data.get("name_expressions") or ()))


class _FunctionModule:
    """``cupy.cuda.function.Module``: load(image) -> the host-built unit."""

    def __init__(self):
        self._module = None

    def load(self, image):
        if isinstance(image, str):
            image = image.encode()
        source, options, filename, digest, names = _read_image(bytes(image))
        from woof.backend.host.loader import load_unit
        self._module = load_unit(filename.rsplit(".", 1)[0] + ":" + digest[:12], source, options=options,
                                 name_expressions=names)

    def load_file(self, path):
        with open(path, "rb") as f:
            self.load(f.read())

    def get_function(self, name):
        return self._module.get_function(name)

    def get_global_var(self, name):
        return self._module.get_global(name)

    get_global = get_global_var


function = types.SimpleNamespace(Module=_FunctionModule, Function=object)


def _compile_using_nvrtc(source, options=(), arch=None, filename="kern.cu", name_expressions=None,
                         log_stream=None, cache_in_memory=False, jitify=False, method=None):
    """CuPy's direct NVRTC call: the unit is built here, so a unit that does
    not compile fails at this call, as it does under NVRTC."""
    image = host_image(source, tuple(options or ()), filename, name_expressions)
    _FunctionModule().load(image)
    mapping = {n: n for n in (name_expressions or ())} or None
    return image, mapping


def _compile_with_cache_cuda(source, options, arch=None, cache_dir=None, extra_source=None,
                             backend="nvrtc", enable_cooperative_groups=False, name_expressions=None,
                             log_stream=None, jitify=False, *args, **kwargs):
    module = _FunctionModule()
    module.load(host_image(source, tuple(options or ()), "kern.cu", name_expressions))
    return module


class _NVRTCProgram:
    """Present so woof's NVRTC option hooks install; NVRTC never runs here."""

    def __init__(self, src, name="kern.cu", *args, **kwargs):
        self.src = src

    def compile(self, options=(), log_stream=None):
        raise AttributeError("NVRTC is not available on woof's host backend")


def _nvrtc_arch():
    from woof.backend.translate import CUDA_ARCH
    return f"-arch=compute_{CUDA_ARCH // 10}", "ptx"


compiler = types.SimpleNamespace(
    CompileException=RuntimeError, compile_using_nvrtc=_compile_using_nvrtc,
    _compile_using_nvrtc_no_warning=_compile_using_nvrtc,
    _compile_with_cache_cuda=_compile_with_cache_cuda, _NVRTCProgram=_NVRTCProgram,
    _get_arch=lambda: "host",
    _get_arch_for_options_for_nvrtc=lambda arch=None: _nvrtc_arch())


class _NvrtcProgram:
    def __init__(self, source, name, headers=(), include_names=()):
        self.source, self.name = source, name or "kern.cu"
        self.headers = dict(zip(include_names or (), headers or ()))
        self.image, self.log = None, ""


def _compile_program(program, options=()):
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
        raise RuntimeError(f"NVRTC_ERROR_COMPILATION (woof host backend): {exc}") from exc


nvrtc = types.SimpleNamespace(
    getVersion=lambda: (0, 0), NVRTCError=RuntimeError,
    createProgram=_NvrtcProgram, compileProgram=_compile_program,
    getPTX=lambda program: program.image, getCUBIN=lambda program: program.image,
    getProgramLog=lambda program: program.log, destroyProgram=lambda program: None)

driver = types.SimpleNamespace(
    CUDADriverError=RuntimeError, funcGetAttribute=lambda attribute, function_ptr: 0,
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK=0, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES=1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES=2, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES=3,
    CU_FUNC_ATTRIBUTE_NUM_REGS=4, CU_FUNC_ATTRIBUTE_PTX_VERSION=5,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION=6,
    ctxGetCurrent=lambda: 1)


def __getattr__(name):
    raise AttributeError(f"cupy.cuda.{name} is not implemented by woof's host backend")


import sys as _sys  # noqa: E402

for _name in ("compiler", "runtime", "driver", "nvrtc", "function", "memory"):
    _sys.modules.setdefault(f"cupy.cuda.{_name}", globals()[_name])
for _name, _value in (("cupy_backends.cuda.api.runtime", runtime), ("cupy_backends.cuda.api.driver", driver),
                      ("cupy_backends.cuda.libs.nvrtc", nvrtc)):
    _sys.modules.setdefault(_name, _value)
