"""The facade's ``ndarray``: CuPy's array, over MLX storage.

A CUDA kernel mutates the buffers it is handed, and every view of a buffer
sees the write.  MLX arrays are immutable values.  The facade reconciles the
two with one level of indirection:

* :class:`Storage` owns a flat ``mx.array`` (``arr``) and is shared by every
  view of it.  An in-place write -- a kernel launch, ``a[...] = v``,
  ``a += v`` -- computes a new ``mx.array`` and rebinds ``Storage.arr``.
* :class:`ndarray` is a view: storage, dtype, shape, byte strides and byte
  offset, exactly NumPy's model, so basic indexing never copies.

Ordering.  A rebind leaves earlier lazy readers of the old value pointing at
it, which is right for pure ops (MLX keeps the old buffer alive) but not for
a kernel that writes the buffer in place.  Every storage rebind is recorded
in :data:`PENDING`; the Metal launcher encodes all of them
(``mx.async_eval``) before it encodes an in-place launch, so each lazy
reader runs before the write, in program order, as CUDA's stream would.

binary64.  MLX's GPU has no float64; a float64 facade array is stored as its
binary64 words (``mx.uint64``).  Kernels see ordinary ``double*`` memory
(translated to soft-float ``f64``); array arithmetic on float64 runs through
NumPy on the host (:mod:`woof.backend.facade._ops`).

binary64 complex.  MLX has no 16-byte element, so a complex128 array lives
in a ``uint64`` storage, two words per element (real, imaginary): CUDA's
``double2`` / ``cuDoubleComplex`` memory.  Its values (``_values``) are those
words with a trailing axis of 2, and its arithmetic runs on the host, as
float64's does.
"""

from __future__ import annotations

import operator
import weakref
from functools import reduce

import numpy as np

#: Storages rebound since the last launch (strong refs; cleared by flush()).
PENDING: list = []

#: In-place launch outputs not yet encoded.  Each is an MLX value whose
#: inputs are the buffers' previous values, so MLX orders a chain of launches
#: through the same buffers by itself; encoding them a batch at a time costs
#: one command-buffer commit per batch instead of one per launch.
LAUNCHED: list = []

def _launch_batch() -> int:
    """Launches per encode: ``WOOF_METAL_LAUNCH_BATCH`` (default 16).  One
    under the GPU timer and the serial diagnostics, which read each launch
    as a command buffer of its own."""
    import os
    if os.environ.get("WOOF_METAL_GPU_TIMES") or os.environ.get("WOOF_METAL_SERIAL"):
        return 1
    return max(1, int(os.environ.get("WOOF_METAL_LAUNCH_BATCH", "16")))


LAUNCH_BATCH = _launch_batch()

#: Storages a not-yet-encoded launch reads (id -> storage).  MLX orders a
#: launch after the writers of its inputs, but not before a later in-place
#: writer of them (write-after-read); such a launch encodes the batch first.
LAZY_READS: dict = {}


def order_launch(reads, writes):
    """Before an in-place launch: if it writes a storage that a launch still
    waiting in the batch reads, encode the batch now, in program order."""
    if LAZY_READS and any(id(st) in LAZY_READS for st in writes):
        flush_launched()


def launched(reads):
    """After an in-place launch: what it reads, while it waits in the batch."""
    if LAUNCHED:
        for st in reads:
            LAZY_READS[id(st)] = st

#: Elements one facade allocation may hold: MLX sizes an array with int32.
MAX_ELEMENTS = 2**31 - 1


def _mx():
    import mlx.core as mx
    return mx


#: numpy dtype -> the mx dtype a Storage holds it in.
def storage_mx_dtype(dtype: np.dtype):
    mx = _mx()
    dtype = np.dtype(dtype)
    table = {
        np.dtype(np.float32): mx.float32, np.dtype(np.float16): mx.float16,
        np.dtype(np.float64): mx.uint64,  # binary64 words
        np.dtype(np.int8): mx.int8, np.dtype(np.uint8): mx.uint8,
        np.dtype(np.int16): mx.int16, np.dtype(np.uint16): mx.uint16,
        np.dtype(np.int32): mx.int32, np.dtype(np.uint32): mx.uint32,
        np.dtype(np.int64): mx.int64, np.dtype(np.uint64): mx.uint64,
        np.dtype(np.bool_): mx.bool_, np.dtype(np.complex64): mx.complex64,
        np.dtype(np.complex128): mx.uint64,  # (real, imaginary) word pairs
    }
    try:
        return table[dtype]
    except KeyError:
        raise TypeError(f"the mlx backend has no storage for dtype {dtype}") from None


#: dtypes whose arithmetic runs on the host (MLX's GPU has no binary64)
HOST_DTYPES = (np.dtype(np.float64), np.dtype(np.complex128))


def on_host(dtype) -> bool:
    """True for a dtype the facade computes with on the host."""
    return isinstance(dtype, np.dtype) and dtype in HOST_DTYPES


def storage_dtype(dtype) -> np.dtype:
    """The element dtype of the Storage that holds ``dtype`` arrays."""
    dtype = np.dtype(dtype)
    return np.dtype(np.uint64) if dtype == np.complex128 else dtype


def values_shape(shape, dtype) -> tuple:
    """The shape of an array's values (``_values``): complex128's carry a
    trailing (real, imaginary) axis."""
    shape = tuple(shape)
    return shape + (2,) if np.dtype(dtype) == np.complex128 else shape


def to_storage_values(host: np.ndarray):
    """A host array -> an mx.array in its storage dtype (float64 as words,
    complex128 as word pairs on a trailing axis)."""
    mx = _mx()
    host = np.require(np.asarray(host), requirements="C")   # (keeps 0-d)
    if host.dtype == np.float64:
        return mx.array(host.view(np.uint64))
    if host.dtype == np.complex128:
        return mx.array(host.reshape(-1).view(np.uint64).reshape(host.shape + (2,)))
    return mx.array(host)


#: Bytes of live storages, each rounded up to CuPy's 512-byte allocation
#: unit: the default pool's ``used_bytes`` (MLX's own active memory also
#: holds lazy intermediates and its own rounding).
LIVE_BYTES = [0]


def _release(n):
    LIVE_BYTES[0] -= n


def _count_allocation(st):
    import weakref
    n = (st.nbytes + 511) // 512 * 512
    if n:
        LIVE_BYTES[0] += n
        weakref.finalize(st, _release, n)


#: Active cupy.cuda.MemoryHook objects (innermost last).
MEMORY_HOOKS: list = []


class Storage:
    """A buffer.  ``arr`` is always a flat, row-contiguous mx.array that owns
    its elements: MLX may otherwise keep a broadcast or strided view (a
    filled array can be one scalar with stride 0), and a kernel bound to
    such a buffer would read and write past the one element it holds."""

    __slots__ = ("arr", "dtype", "_aliases", "_memory", "_address", "__weakref__")

    def __init__(self, arr, dtype: np.dtype, *, fresh: bool = False):
        """``fresh``: ``arr`` is a newly computed value no other array holds
        (an upload, an operation's result).  Otherwise it is copied: MLX may
        return an input unchanged from a no-op (reshape, astype to the same
        dtype, contiguous), and two storages sharing one buffer would see
        each other's in-place writes."""
        from woof.backend.facade import _capture
        if _capture.recording() is not None:
            _capture.refuse("a facade array allocation or operation")
        mx = _mx()
        flat = _materialize(arr)
        self.arr = flat if fresh else mx.array(flat)
        self.dtype = storage_dtype(dtype)
        self._aliases = None
        self._memory = None
        self._address = None
        _count_allocation(self)
        if MEMORY_HOOKS:
            # cupy.cuda.MemoryHook: a new storage is a device allocation
            n = self.nbytes
            for hook in MEMORY_HOOKS:
                hook.malloc_preprocess(device_id=0, size=n, mem_size=n)
                hook.malloc_postprocess(device_id=0, size=n, mem_size=n, mem_ptr=0, pmem_id=id(self))
        # A new value is usually still lazy -- a copy or an expression of other
        # buffers.  It must be computed before any in-place launch overwrites
        # what it reads (flush_pending), as CUDA's stream order would have it.
        PENDING.append(self)

    @property
    def nbytes(self) -> int:
        return int(self.arr.size) * self.dtype.itemsize

    def rebind(self, arr):
        """A new value for a storage nobody can address yet (creation)."""
        from woof.backend.facade import _capture
        if _capture.recording() is not None:
            _capture.refuse("a facade array allocation or operation")
        self.arr = _materialize(arr)
        self._address = None          # a new buffer
        PENDING.append(self)

    def rebind_written(self, out):
        """The output of an in-place launch: the same buffer, written.  It
        is enqueued now, so later work is ordered after it as on a CUDA
        stream (writes through pointer tables are invisible to MLX's
        dependency graph)."""
        import mlx.core as mx
        from woof.backend.facade import _capture
        if _capture.recording() is not None:
            _capture.refuse("an in-place write into a facade array (a[...] = v, fill, copy)")
        self.arr = out
        LAUNCHED.append(out)
        if len(LAUNCHED) >= LAUNCH_BATCH:
            flush_launched()

    def alias(self, dtype) -> "Storage":
        """This buffer as ``dtype`` elements: a second storage over the same
        Metal buffer (MLX's ``view`` shares it), for memory handed out
        untyped (``hostAlloc``).  Both see every in-place write."""
        import mlx.core as mx
        from woof.backend.metal import ext
        dtype = storage_dtype(dtype)
        if dtype == self.dtype:
            return self
        if self._aliases is None:
            self._aliases = {}
        st = self._aliases.get(dtype)
        if st is None:
            mx.eval(self.arr)
            arr = self.arr.view(storage_mx_dtype(dtype))
            mx.eval(arr)
            if int(ext.load().gpu_address(arr)) != int(ext.load().gpu_address(self.arr)):
                raise NotImplementedError("MLX copied a reinterpreting view; the memory cannot be shared")
            st = self._aliases[dtype] = Storage(arr, dtype, fresh=True)
        return st

    def address(self) -> int:
        """The buffer's GPU address (evaluating it first).  Recorded, so an
        address handed back as a kernel argument (CUDA code passes
        ``a.data.ptr`` for pointers) resolves to this storage."""
        import mlx.core as mx
        from woof.backend.metal import ext
        base = self._address
        if base is None:
            mx.eval(self.arr)
            base = self._address = int(ext.load().gpu_address(self.arr))
        # Known: an in-place launch's output is the same buffer (rebind_written),
        # so the address stands until rebind() gives the storage a new one.
        # Evaluating for it again waited on every launch still writing the
        # buffer: the microphysics ring copy's table key asked for 390
        # addresses per model step, each a host sync on Metal.
        _ADDRESSES[base] = weakref.ref(self)
        return base


#: GPU base address -> the storage that handed it out (Storage.address)
_ADDRESSES: dict = {}


def storage_at(address: int):
    """(storage, byte offset) of a GPU address a storage handed out, or None."""
    from woof.backend.metal import ext
    for base, ref in list(_ADDRESSES.items()):
        st = ref()
        if st is None:
            del _ADDRESSES[base]
            continue
        if base <= address < base + max(st.nbytes, 1):
            if int(ext.load().gpu_address(st.arr)) != base:   # rebound since
                del _ADDRESSES[base]
                continue
            return st, address - base
    return None


def _materialize(arr):
    mx = _mx()
    return mx.contiguous(arr.reshape(-1))


def flush_pending():
    """Encode every pending storage value (before an in-place launch)."""
    if not PENDING:
        return
    mx = _mx()
    arrays = [s.arr for s in PENDING] + LAUNCHED
    PENDING.clear()
    LAUNCHED.clear()
    LAZY_READS.clear()
    mx.async_eval(*arrays)


def flush_launched():
    """Encode every in-place launch not yet encoded (before anything that
    reaches buffers outside MLX's graph: an address table, a host block, a
    synchronize)."""
    if not LAUNCHED:
        return
    mx = _mx()
    arrays = list(LAUNCHED)
    LAUNCHED.clear()
    LAZY_READS.clear()
    mx.async_eval(*arrays)


def _contiguous_strides(shape, itemsize):
    strides, step = [], itemsize
    for n in reversed(shape):
        strides.append(step)
        step *= max(int(n), 1)
    return tuple(reversed(strides))


def _normalize_shape(shape):
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    return tuple(int(n) for n in shape)


class ndarray:
    """CuPy-compatible array; see the module docstring for the model."""

    __woof_facade__ = True
    __array_priority__ = 100

    def __init__(self, shape, dtype=np.float32, memptr=None, strides=None, order="C", *,
                 _storage=None, _offset=0, _strides=None):
        shape = _normalize_shape(shape)
        dtype = np.dtype(dtype)
        if memptr is not None:
            # CuPy's ndarray(shape, dtype, memptr=...): a view of existing memory
            if isinstance(memptr, tuple):
                memptr = memptr[0]
            if not isinstance(memptr, MemoryPointer):
                raise TypeError("the facade's ndarray(memptr=) takes a facade MemoryPointer")
            _storage, _offset = memptr.storage, memptr.offset
            if _storage.dtype != dtype and memptr.untyped:
                if _offset % dtype.itemsize:
                    raise NotImplementedError("a typed view at an offset that is not a whole element")
                _storage = _storage.alias(dtype)
            if strides is not None:
                _strides = tuple(int(x) for x in strides)
        if _storage is None:
            mx = _mx()
            n = reduce(operator.mul, shape, 1)
            if n > MAX_ELEMENTS:
                raise MemoryError(
                    f"a {dtype} array of {n} elements ({n * dtype.itemsize} bytes) is past the mlx "
                    f"backend's {MAX_ELEMENTS}-element limit per allocation (MLX sizes an array with "
                    "int32): allocate it in pieces")
            words = n * dtype.itemsize // storage_dtype(dtype).itemsize
            _storage = Storage(mx.zeros((words,), dtype=storage_mx_dtype(dtype)), dtype, fresh=True)
        self._st = _storage
        self.dtype = dtype
        self.shape = shape
        self.strides = _strides if _strides is not None else _contiguous_strides(shape, dtype.itemsize)
        self._offset = int(_offset)

    # ---------------------------------------------------------- properties
    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return reduce(operator.mul, self.shape, 1)

    @property
    def itemsize(self):
        return self.dtype.itemsize

    @property
    def nbytes(self):
        return self.size * self.itemsize

    @property
    def flags(self):
        return _Flags(self)

    @property
    def base(self):
        return None

    @property
    def data(self):
        return _Pointer(self)

    @property
    def T(self):
        return self.transpose()

    @property
    def __cuda_array_interface__(self):
        """CUDA's array interface, which woof also uses to tell device
        arrays from host ones (``hasattr``): lazy, so asking costs nothing
        until ``data`` -- the real device address -- is read."""
        return _ArrayInterface(self)

    @property
    def device(self):
        from woof.backend.facade.cuda import Device
        return Device(0)

    def __len__(self):
        if not self.shape:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def __repr__(self):
        return f"facade.ndarray({self.get()!r})"

    # ---------------------------------------------------------- layout
    def is_c_contiguous(self) -> bool:
        # NumPy's flag: the stride of a length-1 axis does not matter
        if self.size <= 1:
            return True
        want = _contiguous_strides(self.shape, self.itemsize)
        return all(n == 1 or s == w for n, s, w in zip(self.shape, self.strides, want))

    def is_f_contiguous(self) -> bool:
        if self.size <= 1:
            return True
        f = tuple(reversed(_contiguous_strides(tuple(reversed(self.shape)), self.itemsize)))
        return all(n == 1 or s == w for n, s, w in zip(self.shape, self.strides, f)) \
            or (self.ndim <= 1 and self.is_c_contiguous())

    def _element_offset(self) -> int:
        st_item = self._st.dtype.itemsize
        if self._offset % st_item:
            raise TypeError("view offset is not aligned to the storage element")
        return self._offset // st_item

    def _same_item(self) -> bool:
        return self._st.dtype.itemsize == self.itemsize

    def _words(self) -> "ndarray":
        """A complex128 view as the uint64 view of its (real, imaginary)
        words: the same bytes, with a trailing axis of 2."""
        return ndarray(self.shape + (2,), np.uint64, _storage=self._st, _offset=self._offset,
                       _strides=self.strides + (8,))

    # ---------------------------------------------------------- read
    def _values(self):
        """This view's elements as an mx.array of shape ``self.shape``
        (storage dtype; float64 as words; complex128 as word pairs, shape
        ``self.shape + (2,)``)."""
        mx = _mx()
        if self.dtype == np.complex128:
            return self._words()._values()
        if not self._same_item():
            return self._reinterpreted_values()
        arr = self._st.arr
        want = storage_mx_dtype(self.dtype)
        if arr.dtype != want:
            # a same-width view of another kind (float32 storage read as
            # uint32): the bits, as CuPy's view gives them
            arr = arr.view(want)
        start = self._element_offset()
        if self.is_c_contiguous():
            flat = arr[start:start + self.size] if (start or self.size != arr.size) else arr
            return flat.reshape(self.shape) if self.shape else flat.reshape(())
        item = self.itemsize
        if any(s % item for s in self.strides):
            raise TypeError("strides are not a multiple of the element size")
        return mx.as_strided(arr, self.shape, tuple(s // item for s in self.strides), start)

    def _reinterpreted_values(self):
        mx = _mx()
        if not self.is_c_contiguous():
            # the storage's bytes read as this dtype, then strided as NumPy
            # would (a float32 column of a uint8 record, a padded slab)
            item = self.itemsize
            if self._offset % item or any(s % item for s in self.strides):
                raise NotImplementedError("view of a reinterpreted dtype at an unaligned stride")
            raw = self._st.arr.view(mx.uint8)
            whole = raw[:raw.size // item * item].view(storage_mx_dtype(self.dtype))
            return mx.as_strided(whole, self.shape, tuple(s // item for s in self.strides),
                                 self._offset // item)
        raw = self._st.arr.view(mx.uint8)
        start = self._offset
        raw = raw[start:start + self.nbytes]
        return raw.view(storage_mx_dtype(self.dtype)).reshape(self.shape)

    def get(self, stream=None, order="C", out=None, blocking=True):
        """Copy to a NumPy array (CuPy's ``ndarray.get``)."""
        from woof.backend.facade import _capture
        if _capture.recording() is not None:
            _capture.refuse("a read back to the host")
        mx = _mx()
        values = self._values()
        mx.eval(values)
        host = np.array(values, copy=True)
        if self.dtype == np.float64:
            host = host.view(np.float64)
        elif self.dtype == np.complex128:
            host = np.ascontiguousarray(host).reshape(-1).view(np.complex128)
        elif host.dtype != self.dtype:
            host = host.astype(self.dtype, copy=False)
        host = host.reshape(self.shape)
        if out is not None:
            out[...] = host
            return out
        return host

    def __array__(self, dtype=None, copy=None):
        # CuPy refuses implicit host conversion; so does the facade, except
        # through the explicit get()/asnumpy() paths.
        raise TypeError("Implicit conversion to a NumPy array is not allowed. "
                        "Please use `.get()` to construct a NumPy array explicitly.")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """NumPy's ufuncs on a facade array run on the device, as CuPy's do
        (``np.maximum(0, x)``): the facade function of the same name."""
        from woof.backend import facade
        fn = getattr(facade, ufunc.__name__, None)
        if method != "__call__":
            fn = getattr(fn, method, None)      # np.add.reduceat(x, ...)
        if fn is None:
            return NotImplemented
        out = kwargs.get("out")
        if isinstance(out, tuple):
            if len(out) != 1:
                return NotImplemented
            kwargs["out"] = out[0]
        return fn(*inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        """NumPy's array functions (``np.where``, ``np.linalg.norm``) on facade
        arrays: the facade's namesake in the matching submodule, as CuPy."""
        from woof.backend import facade
        target = facade
        for part in func.__module__.split(".")[1:]:
            if part.startswith("_"):
                continue
            target = getattr(target, part, None)
        fn = getattr(target, func.__name__, None)
        if fn is None or fn is func:
            return NotImplemented
        return fn(*args, **kwargs)

    def item(self):
        return self.get().item()

    def tolist(self):
        return self.get().tolist()

    def __float__(self):
        return float(self.item())

    def __int__(self):
        return int(self.item())

    def __index__(self):
        return int(self.item())

    def __bool__(self):
        if self.size != 1:
            raise ValueError("The truth value of an array with more than one element is ambiguous.")
        return bool(self.item())

    # ---------------------------------------------------------- write
    def _write(self, values):
        """Store an mx.array (this view's dtype, broadcastable to shape)
        into this view, in place: the storage keeps its buffer and address.
        The copy kernel addresses the buffer in this view's element width,
        so a reinterpreted-dtype view writes like any other."""
        from woof.backend.facade._write import write
        if self.dtype == np.complex128:
            write(self._words(), values)
            return
        write(self, values)

    def _flat_indices(self):
        mx = _mx()
        item = self.itemsize
        idx = mx.array(self._element_offset(), dtype=mx.int64)
        for axis, (n, s) in enumerate(zip(self.shape, self.strides)):
            shape = [1] * self.ndim
            shape[axis] = n
            idx = idx + (mx.arange(n, dtype=mx.int64) * (s // item)).reshape(shape)
        return mx.broadcast_to(idx, self.shape)

    def fill(self, value):
        from woof.backend.facade import _capture
        from woof.backend.facade._ops import scalar_values
        if _capture.defer(ndarray.fill, self, value):
            return
        self._write(scalar_values(value, self.dtype))

    def set(self, host, stream=None):
        host = np.asarray(host, dtype=self.dtype)
        if host.shape != self.shape:
            raise ValueError(f"shape mismatch: {host.shape} vs {self.shape}")
        self._write(to_storage_values(host).reshape(values_shape(self.shape, self.dtype)))

    # ---------------------------------------------------------- views
    def __getitem__(self, key):
        from woof.backend.facade._index import getitem
        return getitem(self, key)

    def __setitem__(self, key, value):
        from woof.backend.facade._index import setitem
        setitem(self, key, value)

    def view(self, dtype=None):
        if dtype is None:
            return ndarray(self.shape, self.dtype, _storage=self._st, _offset=self._offset,
                           _strides=self.strides)
        dtype = np.dtype(dtype)
        if dtype.itemsize != self.itemsize:
            if self.is_c_contiguous():
                last = self.shape[-1] * self.itemsize // dtype.itemsize
                shape = self.shape[:-1] + (last,)
                return ndarray(shape, dtype, _storage=self._st, _offset=self._offset)
            # NumPy: the last axis must be contiguous; it is re-cut, the
            # other axes keep their strides
            if not self.ndim or self.strides[-1] != self.itemsize \
                    or (self.shape[-1] * self.itemsize) % dtype.itemsize:
                raise ValueError("changing the itemsize needs a contiguous last axis")
            last = self.shape[-1] * self.itemsize // dtype.itemsize
            return ndarray(self.shape[:-1] + (last,), dtype, _storage=self._st, _offset=self._offset,
                           _strides=self.strides[:-1] + (dtype.itemsize,))
        return ndarray(self.shape, dtype, _storage=self._st, _offset=self._offset,
                       _strides=self.strides)

    def reshape(self, *shape, order="C"):
        if len(shape) == 1 and not isinstance(shape[0], (int, np.integer)):
            shape = tuple(shape[0])
        shape = list(_normalize_shape(shape))
        if -1 in shape:
            known = reduce(operator.mul, (n for n in shape if n != -1), 1)
            shape[shape.index(-1)] = self.size // known if known else 0
        shape = tuple(shape)
        if reduce(operator.mul, shape, 1) != self.size:
            raise ValueError(f"cannot reshape array of size {self.size} into shape {shape}")
        if self._memory_order(order) == "F":
            # column-major index order: NumPy's a.reshape(s, "F") is
            # a.T.reshape(reversed s).T, a view when a is F-contiguous
            return self.transpose().reshape(tuple(reversed(shape))).transpose()
        if self.is_c_contiguous():
            return ndarray(shape, self.dtype, _storage=self._st, _offset=self._offset)
        strides = _nocopy_strides(self.shape, self.strides, shape, self.dtype.itemsize) \
            if self.size else None
        if strides is not None:
            # NumPy/CuPy's view reshape: strided rows split or merged where
            # the layout allows it (a padded member slab stays its backing).
            return ndarray(shape, self.dtype, _storage=self._st, _offset=self._offset,
                           _strides=tuple(strides))
        return self.copy().reshape(shape)

    def _memory_order(self, order) -> str:
        """"C" or "F": NumPy's "A" and "K" read F only for an array laid out
        column-major (and not also row-major)."""
        order = (order or "C").upper()
        if order in ("A", "K"):
            return "F" if self.is_f_contiguous() and not self.is_c_contiguous() else "C"
        return order

    def ravel(self, order="C"):
        return self.reshape(-1, order=order)

    def flatten(self, order="C"):
        return self.copy().reshape(-1, order=self._memory_order(order))

    def squeeze(self, axis=None):
        keep = [i for i, n in enumerate(self.shape)
                if n != 1 or (axis is not None and i not in _axes(axis, self.ndim))]
        return ndarray(tuple(self.shape[i] for i in keep), self.dtype, _storage=self._st,
                       _offset=self._offset, _strides=tuple(self.strides[i] for i in keep))

    def transpose(self, *axes):
        if not axes or axes == (None,):
            axes = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and not isinstance(axes[0], (int, np.integer)):
            axes = tuple(axes[0])
        return ndarray(tuple(self.shape[a] for a in axes), self.dtype, _storage=self._st,
                       _offset=self._offset, _strides=tuple(self.strides[a] for a in axes))

    def swapaxes(self, a, b):
        axes = list(range(self.ndim))
        axes[a], axes[b] = axes[b], axes[a]
        return self.transpose(axes)

    def copy(self, order="C"):
        if self.ndim > 1 and self._memory_order(order) == "F":
            return self.transpose().copy().transpose()
        mx = _mx()
        if self.dtype == np.complex128:
            return ndarray(self.shape, self.dtype, _storage=Storage(mx.contiguous(self._values()), self.dtype))
        return ndarray(self.shape, self.dtype,
                       _storage=Storage(mx.contiguous(self._values()), self._st.dtype)) \
            if self._same_item() else ndarray(self.shape, self.dtype, _storage=Storage(
                mx.contiguous(self._values()).view(storage_mx_dtype(self.dtype)), self.dtype))

    def astype(self, dtype, copy=True, order="K", casting=None):
        dtype = np.dtype(dtype)
        if dtype == self.dtype and not copy:
            return self
        from woof.backend.facade._ops import astype
        return astype(self, dtype)

    # ---------------------------------------------------------- arithmetic
    def __getattr__(self, name):
        # Reductions, comparisons and the rest of ndarray's methods live in
        # _ops; they are looked up lazily so this file stays the data model.
        from woof.backend.facade import _ops
        fn = _ops.METHODS.get(name)
        if fn is None:
            raise AttributeError(f"facade.ndarray has no attribute {name!r} "
                                 "(not implemented by the mlx backend)")
        return lambda *a, **k: fn(self, *a, **k)


def _axes(axis, ndim):
    if isinstance(axis, (int, np.integer)):
        return {int(axis) % ndim}
    return {int(a) % ndim for a in axis}


class _Flags:
    def __init__(self, a):
        self._a = a

    @property
    def c_contiguous(self):
        return self._a.is_c_contiguous()

    @property
    def f_contiguous(self):
        a = self._a
        return a.is_f_contiguous()

    @property
    def owndata(self):
        return self._a._offset == 0 and self._a.nbytes == self._a._st.nbytes

    def __getitem__(self, key):
        return {"C_CONTIGUOUS": self.c_contiguous, "F_CONTIGUOUS": self.f_contiguous,
                "OWNDATA": self.owndata}[key]


class _ArrayInterface(dict):
    def __init__(self, a):
        super().__init__(shape=a.shape, typestr=a.dtype.str, version=3,
                         strides=None if a.is_c_contiguous() else a.strides, descr=[("", a.dtype.str)])
        self._a = a

    def __missing__(self, key):
        if key == "data":
            return (self._a.data.ptr, False)
        if key == "stream":
            return None
        raise KeyError(key)

    def __contains__(self, key):
        return key in ("data", "stream") or super().__contains__(key)

    def get(self, key, default=None):
        return self[key] if key in self else default


class MemoryPointer:
    """CuPy's MemoryPointer over facade storage: (storage, byte offset)."""

    def __init__(self, storage: Storage, offset: int = 0, size: int | None = None,
                 untyped: bool = False):
        self.storage = storage
        self.offset = int(offset)
        self.size = storage.nbytes - self.offset if size is None else int(size)
        # bytes with no element type of their own (a hostAlloc block): an
        # ndarray over them takes its dtype (Storage.alias)
        self.untyped = untyped

    @property
    def ptr(self):
        return self.storage.address() + self.offset

    @property
    def mem(self):
        return self

    def __add__(self, nbytes):
        return MemoryPointer(self.storage, self.offset + int(nbytes), untyped=self.untyped)

    def _bytes_view(self, nbytes):
        st = self.storage
        item = st.dtype.itemsize
        if self.offset % item or nbytes % item:
            raise NotImplementedError("a byte range not aligned to the storage element")
        return ndarray((nbytes // item,), st.dtype, _storage=st, _offset=self.offset)

    def copy_from_host(self, mem, size):
        """Bytes from a host address (an int, or anything ctypes accepts)."""
        import ctypes
        address = mem if isinstance(mem, int) else ctypes.cast(mem, ctypes.c_void_p).value
        host = np.frombuffer(ctypes.string_at(address, int(size)), dtype=self.storage.dtype)
        self._bytes_view(int(size)).set(host)

    def copy_to_host(self, mem, size):
        import ctypes
        address = mem if isinstance(mem, int) else ctypes.cast(mem, ctypes.c_void_p).value
        data = self._bytes_view(int(size)).get().tobytes()
        ctypes.memmove(address, data, int(size))

    def copy_from_device(self, src, size):
        if isinstance(src, tuple):
            src = src[0]
        src_view = MemoryPointer(src.storage, src.offset)._bytes_view(int(size))
        self._bytes_view(int(size))._write(src_view._values())

    def memset(self, value, size):
        view = self._bytes_view(int(size))
        if value != 0:
            raise NotImplementedError("memset to a non-zero byte")
        view.fill(0)


def _address(mem) -> int:
    """A raw address from an int, a pointer object, or a ctypes pointer."""
    import ctypes
    if isinstance(mem, tuple):
        mem = mem[0]
    if isinstance(mem, (int, np.integer)):
        return int(mem)
    if hasattr(mem, "ptr"):
        return int(mem.ptr)
    return ctypes.cast(mem, ctypes.c_void_p).value


def _nocopy_strides(oldshape, oldstrides, newshape, itemsize):
    """NumPy's ``_attempt_nocopy_reshape`` (C order): the strides that give
    ``newshape`` as a view of the same elements, or None."""
    old = [(n, st) for n, st in zip(oldshape, oldstrides) if n != 1]
    newshape = list(newshape)
    oldnd, newnd = len(old), len(newshape)
    out = [0] * newnd
    oi, oj, ni, nj = 0, 1, 0, 1
    while ni < newnd and oi < oldnd:
        np_, op = newshape[ni], old[oi][0]
        while np_ != op:
            if np_ < op:
                np_ *= newshape[nj]
                nj += 1
            else:
                op *= old[oj][0]
                oj += 1
        for ok in range(oi, oj - 1):
            if old[ok + 1][1] * old[ok + 1][0] != old[ok][1]:
                return None
        out[nj - 1] = old[oj - 1][1]
        for nk in range(nj - 1, ni, -1):
            out[nk - 1] = out[nk] * newshape[nk]
        ni, nj, oi, oj = nj, nj + 1, oj, oj + 1
    last = out[ni - 1] if ni >= 1 else itemsize
    for nk in range(ni, newnd):
        out[nk] = last
    return out


class _Memory:
    """``ndarray.data.mem``: the storage (CuPy's allocation) behind a view."""

    def __init__(self, st):
        self._st = st

    @property
    def ptr(self):
        return self._st.address()

    @property
    def size(self):
        return self._st.nbytes


class _Pointer:
    """``ndarray.data``: the view's device address."""

    def __init__(self, a):
        self._a = a

    @property
    def ptr(self):
        """The real GPU address of the view's first element: woof builds
        device-pointer tables from it, as CUDA code does."""
        return self._a._st.address() + self._a._offset

    @property
    def mem(self):
        """CuPy's ``MemoryPointer.mem``: the whole allocation, whose
        ``ptr`` is its first byte (not the view's) and ``size`` its bytes."""
        # One object per storage while anyone holds it (CuPy compares
        # owners with `is`); a weak link, so storage is freed promptly.
        st = self._a._st
        memory = st._memory() if st._memory is not None else None
        if memory is None:
            import weakref
            memory = _Memory(st)
            st._memory = weakref.ref(memory)
        return memory

    @property
    def size(self):
        return self._a._st.nbytes

    # CuPy's MemoryPointer copies, by raw address (cupy.cuda.runtime.memcpy)
    def copy_from_device(self, src, size, stream=None):
        from woof.backend.facade.cuda import _memcpy
        _memcpy(self.ptr, _address(src), size)

    def copy_from_host(self, mem, size, stream=None):
        from woof.backend.facade.cuda import _memcpy
        _memcpy(self.ptr, _address(mem), size)

    def copy_to_host(self, mem, size, stream=None):
        from woof.backend.facade.cuda import _memcpy
        _memcpy(_address(mem), self.ptr, size)

    copy_from_device_async = copy_from_device
    copy_from_host_async = copy_from_host
    copy_to_host_async = copy_to_host

    def memset_async(self, value, size, stream=None):
        """cudaMemsetAsync from this view's first byte: ``size`` bytes of
        ``value``.  The facade's arrays are typed storage, so a byte range
        must cover whole storage elements, and only a zero fill is spelled
        for every dtype."""
        st, a = self._a._st, self._a
        item = st.dtype.itemsize
        if int(value) != 0 or a._offset % item or int(size) % item:
            raise NotImplementedError("memset of a non-zero byte, or of a partial storage element")
        count = int(size) // item
        if count:
            ndarray((count,), st.dtype, _storage=st, _offset=a._offset).fill(0)

    memset = memset_async


# woof tells device arrays from host ones by `type(x).__module__` too (20
# sites); in a facade process this class IS cupy's ndarray.
ndarray.__module__ = "cupy"
