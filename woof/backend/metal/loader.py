"""Load translated CUDA units as Metal libraries; launch kernels in place.

:func:`load_unit` returns a :class:`MetalModule` with CuPy's ``RawModule``
interface (``get_function``); its functions have CuPy's launch signature,
``kernel(grid, block, args, shared_mem=0, stream=None)``, so every woof
launcher runs unchanged.

Arguments are bound by the kernel's translated signature:

* a pointer parameter takes a facade ``ndarray``; the launch binds its
  storage at the view's byte offset, and a non-``const`` pointer's storage
  is rebound to the launch output, which shares the input's buffer
  (``woof.backend.metal.ext``), so the kernel writes in place;
* a scalar parameter is packed as its C type (``struct`` format from the
  signature), whatever Python or NumPy scalar is passed.

Before encoding, every pending facade value is encoded first
(``facade._array.flush_pending``): a lazy reader of a buffer's old value
must run before the kernel writes over it.
"""

from __future__ import annotations

import os
import struct
from functools import lru_cache

import numpy as np

from woof.backend.translate import TranslatedUnit, translate

MATH_MODE_SAFE = 0


class LaunchError(RuntimeError):
    """A launch whose arguments do not fit the kernel's signature."""


def _dim3(value) -> tuple[int, int, int]:
    if isinstance(value, (int, np.integer)):
        return (int(value), 1, 1)
    dims = tuple(int(v) for v in value) + (1, 1, 1)
    return dims[:3]


class MetalFunction:
    def __init__(self, unit: TranslatedUnit, name: str, library_key: str, module=None):
        self.module = module
        if name not in unit.kernels:
            raise LaunchError(f"no kernel {name!r} in the translated unit "
                              f"(kernels: {', '.join(sorted(unit.kernels))})")
        self.unit = unit
        self.name = name
        self.signature = unit.kernels[name]
        self.library_key = library_key
        self.ptr = 0              # CUfunction handle stand-in (driver.funcGetAttribute)

    # CuPy's Function attributes.  Metal reports no CUDA local frame,
    # registers or PTX version (0 each); the thread limit is the pipeline's.
    local_size_bytes = shared_size_bytes = const_size_bytes = num_regs = 0
    ptx_version = binary_version = 0

    @property
    def max_threads_per_block(self):
        return int(getattr(self.module, "max_threads", {}).get(self.name, 1024)) if self.module else 1024

    @property
    def attributes(self):
        return {"max_threads_per_block": self.max_threads_per_block, "shared_size_bytes": 0,
                "const_size_bytes": 0, "local_size_bytes": 0, "num_regs": 0,
                "ptx_version": 0, "binary_version": 0}

    def __call__(self, grid, block, args, shared_mem=0, stream=None):
        from woof.backend.facade._array import flush_pending, ndarray
        from woof.backend.metal import ext
        sig = self.signature
        params = sig.params
        visible = [p for p in params if p.global_name is None]
        if len(args) != len(visible):
            raise LaunchError(f"{self.name}: {len(args)} arguments for {len(visible)} parameters")
        supplied = iter(args)
        buffers, offsets, writable, slots, targets = [], [], [], [], []
        null_slots, null_table = [], []
        ordinal = -1                      # the pointer's position: slot, or table entry
        blob = bytearray(sig.scalar_size)
        for p in params:
            a = self.module.globals[p.global_name] if p.global_name is not None else next(supplied)
            if p.pointer:
                ordinal += 1
                if a is None or (isinstance(a, (int, np.integer)) and not isinstance(a, (bool, np.bool_))
                                 and int(a) == 0):
                    # nullptr: nothing bound (direct) or the address 0 (table)
                    (null_slots if sig.pointer_mode == "direct" else null_table).append(ordinal)
                    continue
                if isinstance(a, (int, np.integer)) and not isinstance(a, (bool, np.bool_)) and int(a):
                    # an address (``x.data.ptr``): the storage that handed it out
                    from woof.backend.facade._array import storage_at
                    found = storage_at(int(a))
                    if found is None:
                        raise LaunchError(f"{self.name}: parameter {p.name!r} got the address "
                                          f"0x{int(a):x}, which no live facade array handed out")
                    st, off = found
                    a = ndarray((max(st.nbytes - off, 1) // st.dtype.itemsize or 1,), st.dtype,
                                _storage=st, _offset=off)
                if not isinstance(a, ndarray):
                    raise LaunchError(f"{self.name}: parameter {p.name!r} is a pointer; "
                                      f"got {type(a).__name__}")
                slots.append(ordinal if sig.pointer_mode == "direct" else -1)
                buffers.append(a._st.arr)
                offsets.append(a._offset)
                writable.append(not p.const)
                targets.append(a._st if not p.const else None)
            else:
                blob[p.offset:p.offset + p.size] = self._pack(p, a)
        if not any(writable):
            # It writes only through device addresses it was handed (a pointer
            # table).  MLX schedules a primitive through its outputs, so a
            # one-word sentinel rides along at a slot the kernel never reads.
            spare = (sig.scalar_slot + 1) if sig.scalar_slot >= 0 else (
                2 if sig.pointer_mode == "table" else ordinal + 1)
            if spare > 30:
                raise LaunchError(f"{self.name}: no free buffer slot for the launch sentinel")
            sentinel = _sentinel()
            buffers.append(sentinel._st.arr)
            offsets.append(0)
            writable.append(True)
            slots.append(spare)
            targets.append(sentinel._st)
        if sig.pointer_mode == "table" or sig.indirect:
            _guard_residency(self.name)
        blobs, blob_slots = ([bytes(blob)], [sig.scalar_slot]) if sig.scalar_size else ([], [])
        flush_pending()
        if _TRACE:
            with open(_TRACE, "a") as f:
                f.write(self.name + "\n")
        mode = _SERIAL_MODE if (not _SERIAL_ONLY or any(k in self.name for k in _SERIAL_ONLY)) else ""
        if sig.indirect or mode in ("1", "before"):
            # buffers reached through host-built address tables are invisible
            # to MLX's hazard tracking: let everything before finish first
            import mlx.core as mx
            mx.synchronize()
        outputs = ext.load().launch(
            self.library_key, self.unit.msl, self.name, _dim3(grid), _dim3(block),
            buffers, offsets, writable, slots, sig.table_slot, blobs, blob_slots,
            int(shared_mem or 0), MATH_MODE_SAFE, null_slots, null_table)
        written = iter(outputs)
        for st in targets:
            if st is not None:
                st.rebind_written(next(written))
        if sig.indirect or mode in ("1", "after"):
            import mlx.core as mx
            mx.eval(*outputs)       # ... and finish before anything after it
            if mode == "1":
                mx.synchronize()

    def _pack(self, p, value) -> bytes:
        if p.fmt is None:
            if isinstance(value, (bytes, bytearray)) and len(value) == p.size:
                return bytes(value)
            if isinstance(value, np.ndarray) and value.nbytes == p.size:
                return value.tobytes()
            if isinstance(value, np.generic) and value.nbytes == p.size:
                return value.tobytes()
            raise LaunchError(f"{self.name}: by-value parameter {p.name!r} ({p.ctype}, "
                              f"{p.size} bytes) needs its bytes")
        if isinstance(value, np.generic):
            value = value.item()
        try:
            packed = struct.pack("<" + p.fmt, value)
        except struct.error as exc:
            raise LaunchError(f"{self.name}: parameter {p.name!r} ({p.ctype}): {exc}") from None
        if len(packed) != p.size:
            raise LaunchError(f"{self.name}: parameter {p.name!r} packs {len(packed)} bytes "
                              f"for a {p.size}-byte {p.ctype}")
        return packed


_SENTINEL = None


def _sentinel():
    global _SENTINEL
    if _SENTINEL is None:
        from woof.backend.facade._array import ndarray
        _SENTINEL = ndarray((1,), np.uint32)
    return _SENTINEL


def _guard_residency(kernel: str):
    """A table-bound kernel reaches its buffers through GPU addresses, which
    are valid only for resident memory: MLX's residency sets, bounded by
    the wired limit the backend sets (woof.backend.facade.configure)."""
    import mlx.core as mx
    limit = _wired_limit()
    if limit and mx.get_active_memory() > limit:
        raise LaunchError(
            f"{kernel}: {mx.get_active_memory()} bytes are allocated, past the "
            f"{limit}-byte wired limit; the buffers this kernel reaches through its "
            "argument table may not be resident")


_WIRED = None


def _wired_limit():
    return _WIRED


def set_wired_limit():
    """Enable MLX's residency sets for every allocation up to the device's
    recommended working set (called once when the mlx backend installs)."""
    global _WIRED
    import mlx.core as mx
    info = (getattr(mx, "device_info", None) or mx.metal.device_info)()
    _WIRED = int(info["max_recommended_working_set_size"])
    mx.set_wired_limit(_WIRED)
    return _WIRED


_SHARED_STREAM = None


def share_one_stream():
    """Every thread works on one MLX stream, as CUDA's arrays and its null
    stream are shared by every host thread.  MLX's own streams belong to
    the thread that made them: an array computed lazily in one thread could
    not be evaluated in another ("There is no Stream(gpu, N) in current
    thread"), which the ensemble's member threads do.  The stream is made
    the default here and in each thread as it starts."""
    global _SHARED_STREAM
    import threading
    import mlx.core as mx
    if _SHARED_STREAM is not None:
        return _SHARED_STREAM
    _SHARED_STREAM = mx.new_thread_unsafe_stream(mx.gpu)
    mx.set_default_stream(_SHARED_STREAM)
    # The stream is not thread-safe: two threads evaluating on it at once
    # abort the process (LETKF's per-card threads did).  Evaluation is
    # serialised; building lazy graphs stays concurrent.
    lock = threading.RLock()

    def serialised(fn):
        def call(*args, **kwargs):
            with lock:
                return fn(*args, **kwargs)
        call.__name__, call.__doc__ = fn.__name__, fn.__doc__
        return call
    for name in ("eval", "async_eval", "synchronize"):
        setattr(mx, name, serialised(getattr(mx, name)))
    bootstrap = threading.Thread._bootstrap_inner

    def _bootstrap_inner(self):
        mx.set_default_stream(_SHARED_STREAM)
        bootstrap(self)
    threading.Thread._bootstrap_inner = _bootstrap_inner
    return _SHARED_STREAM


#: C type of a module global's element -> the storage dtype it is held in
#: WOOF_METAL_SERIAL=1: every launch waits for all earlier GPU work and
#: finishes before returning -- a diagnostic for ordering hazards.
_SERIAL_MODE = os.environ.get("WOOF_METAL_SERIAL", "")
_SERIAL = _SERIAL_MODE not in ("", "0")
#: WOOF_METAL_SERIAL_ONLY=a,b: serialise only kernels whose name contains one
_SERIAL_ONLY = tuple(x for x in os.environ.get("WOOF_METAL_SERIAL_ONLY", "").split(",") if x)
#: WOOF_METAL_TRACE=path: append each launched kernel's name
_TRACE = os.environ.get("WOOF_METAL_TRACE", "")

_GLOBAL_DTYPES = {"float": np.float32, "double": np.float64, "int": np.int32,
                  "unsigned int": np.uint32, "long": np.int64, "unsigned long": np.uint64,
                  "long long": np.int64, "unsigned long long": np.uint64, "short": np.int16,
                  "unsigned short": np.uint16, "char": np.int8, "signed char": np.int8,
                  "unsigned char": np.uint8, "bool": np.bool_}


class CompileError(RuntimeError):
    """Metal refused a translated unit; the MSL is saved for inspection."""


class MetalModule:
    """CuPy ``RawModule``'s interface over one translated unit.  The Metal
    library is built when the module is made, so a unit whose MSL does not
    compile fails here, naming its MSL file; each kernel's pipeline (most
    of a cold unit's compile time) is built at its first ``get_function``,
    or for every kernel by :meth:`build_all`."""

    def __init__(self, unit: TranslatedUnit, name: str):
        self.unit = unit
        self.name = name
        self.library_key = f"woof:{name}:{unit.key[:24]}"
        self.code = unit.msl
        self._functions: dict[str, MetalFunction] = {}
        self.max_threads: dict[str, int] = {}
        self._build(())
        # module globals: one buffer each, for the module's lifetime, zeroed
        # or holding the declaration's initialiser
        from woof.backend.facade import asarray
        from woof.backend.facade._array import ndarray
        self.globals = {}
        for gname, g in unit.globals.items():
            dtype = _GLOBAL_DTYPES.get(g["ctype"], np.uint8)
            count = g["nbytes"] // np.dtype(dtype).itemsize
            if g.get("init"):
                self.globals[gname] = asarray(np.frombuffer(g["init"], dtype=dtype).copy())
            else:
                self.globals[gname] = ndarray((count,), dtype)

    def get_global(self, name):
        """CuPy's RawModule.get_global: the global's MemoryPointer."""
        from woof.backend.facade._array import MemoryPointer
        if name not in self.globals:
            raise KeyError(f"module {self.name!r} has no global {name!r} "
                           f"(globals: {', '.join(sorted(self.globals)) or 'none'})")
        a = self.globals[name]
        return MemoryPointer(a._st, a._offset)

    get_global_var = get_global

    def _build(self, kernels):
        """Build the library and the named kernels' pipelines."""
        from woof.backend.metal import ext
        kernels = [k for k in kernels if k not in self.max_threads]
        try:
            limits = ext.load().compile(self.library_key, self.unit.msl, kernels, MATH_MODE_SAFE)
        except RuntimeError as exc:
            import os
            import tempfile
            path = os.path.join(tempfile.gettempdir(), f"woof-metal-{self.unit.key[:16]}.metal")
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.unit.msl)
            errors = [line for line in str(exc).splitlines() if "error:" in line][:12] \
                or str(exc).strip().splitlines()[:12]     # a pipeline failure has no `error:` lines
            raise CompileError(f"Metal did not compile the translation of {self.name!r} "
                               f"(MSL saved to {path}):\n" + "\n".join(errors)) from None
        self.max_threads.update(zip(kernels, limits))

    def build_all(self) -> None:
        """Every kernel's pipeline now (tools/metal_translation_census.py)."""
        self._build(sorted(self.unit.kernels))

    def get_function(self, name: str) -> MetalFunction:
        fn = self._functions.get(name)
        if fn is None:
            if name in self.unit.kernels:
                self._build((name,))
            fn = self._functions[name] = MetalFunction(self.unit, name, self.library_key, self)
        return fn

    def compile(self, *a, **k):
        return None


def contracts(options) -> bool:
    """Whether a unit's Metal translation may contract multiply-adds.

    ``fast`` numerics: as NVRTC would after woof.numerics' compile boundary
    (contract unless the effective options still say ``--fmad=false``).
    ``strict``: never.  NVRTC's own contraction choices cannot be reproduced
    on Metal, so strict Metal keeps every operation rounded as written --
    the host tier's ``-ffp-contract=off`` and gfortran -O0's arithmetic.
    """
    from woof import numerics
    if numerics.is_strict():
        return False
    effective = numerics.effective_options(tuple(options or ()))
    return not any(o in ("--fmad=false", "-fmad=false") for o in effective)


def retains_subnormals(name: str, source: str, options) -> bool:
    """Whether CUDA keeps this unit's binary32 subnormals (--ftz=false):
    then its Metal translation must too (woof/backend/metal/f32s.h)."""
    from woof import numerics
    from woof.core import kernels
    from woof.cuda_subnormals import MARKER
    effective = numerics.effective_options(tuple(options or ()))
    if any(o in ("--ftz=false", "-ftz=false") for o in effective):
        return True
    if MARKER.strip() and MARKER.strip() in source:
        return True
    base = name.split("[", 1)[0]
    sets = (kernels._NO_FTZ_MODULES, kernels._SOLAR_NO_FTZ_MODULES,
            kernels._URBAN_NO_FTZ_MODULES, kernels._UNFLUSHED_MODULES)
    return any(base in s for s in sets)


@lru_cache(maxsize=None)
def _translated(source: str, unit: str, contract: bool, retain: bool, defines=(),
                include_dirs=()) -> TranslatedUnit:
    return translate(source, unit=unit, contract=contract, retain_subnormals=retain, defines=defines,
                     include_dirs=include_dirs)


def load_unit(name: str, source: str, *, options=()) -> MetalModule:
    from woof import numerics
    from woof.backend.translate import option_defines, option_include_dirs
    effective = numerics.effective_options(tuple(options or ()))
    unit = _translated(source, name, contracts(options), retains_subnormals(name, source, options),
                       option_defines(effective), option_include_dirs(effective))
    return MetalModule(unit, name)
