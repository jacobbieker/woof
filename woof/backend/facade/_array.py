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
"""

from __future__ import annotations

import operator
import weakref
from functools import reduce

import numpy as np

#: Storages rebound since the last launch (strong refs; cleared by flush()).
PENDING: list = []


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
    }
    try:
        return table[dtype]
    except KeyError:
        raise TypeError(f"the mlx backend has no storage for dtype {dtype}") from None


def to_storage_values(host: np.ndarray):
    """A host array -> an mx.array in its storage dtype (float64 as words)."""
    mx = _mx()
    host = np.require(np.asarray(host), requirements="C")   # (keeps 0-d)
    if host.dtype == np.float64:
        return mx.array(host.view(np.uint64))
    return mx.array(host)


class Storage:
    """A buffer.  ``arr`` is always a flat, row-contiguous mx.array that owns
    its elements: MLX may otherwise keep a broadcast or strided view (a
    filled array can be one scalar with stride 0), and a kernel bound to
    such a buffer would read and write past the one element it holds."""

    __slots__ = ("arr", "dtype", "_aliases", "__weakref__")

    def __init__(self, arr, dtype: np.dtype, *, fresh: bool = False):
        """``fresh``: ``arr`` is a newly computed value no other array holds
        (an upload, an operation's result).  Otherwise it is copied: MLX may
        return an input unchanged from a no-op (reshape, astype to the same
        dtype, contiguous), and two storages sharing one buffer would see
        each other's in-place writes."""
        mx = _mx()
        flat = _materialize(arr)
        self.arr = flat if fresh else mx.array(flat)
        self.dtype = np.dtype(dtype)
        self._aliases = None
        # A new value is usually still lazy -- a copy or an expression of other
        # buffers.  It must be computed before any in-place launch overwrites
        # what it reads (flush_pending), as CUDA's stream order would have it.
        PENDING.append(self)

    @property
    def nbytes(self) -> int:
        return int(self.arr.size) * self.dtype.itemsize

    def rebind(self, arr):
        """A new value for a storage nobody can address yet (creation)."""
        self.arr = _materialize(arr)
        PENDING.append(self)

    def rebind_written(self, out):
        """The output of an in-place launch: the same buffer, written.  It
        is enqueued now, so later work is ordered after it as on a CUDA
        stream (writes through pointer tables are invisible to MLX's
        dependency graph)."""
        import mlx.core as mx
        self.arr = out
        mx.async_eval(out)

    def alias(self, dtype) -> "Storage":
        """This buffer as ``dtype`` elements: a second storage over the same
        Metal buffer (MLX's ``view`` shares it), for memory handed out
        untyped (``hostAlloc``).  Both see every in-place write."""
        import mlx.core as mx
        from woof.backend.metal import ext
        dtype = np.dtype(dtype)
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
        mx.eval(self.arr)
        base = int(ext.load().gpu_address(self.arr))
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
    arrays = [s.arr for s in PENDING]
    PENDING.clear()
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
            _storage = Storage(mx.zeros((n,), dtype=storage_mx_dtype(dtype)), dtype, fresh=True)
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
        return self.strides == _contiguous_strides(self.shape, self.itemsize) or self.size <= 1

    def is_f_contiguous(self) -> bool:
        f = tuple(reversed(_contiguous_strides(tuple(reversed(self.shape)), self.itemsize)))
        return self.strides == f or self.size <= 1 or (self.ndim <= 1 and self.is_c_contiguous())

    def _element_offset(self) -> int:
        st_item = self._st.dtype.itemsize
        if self._offset % st_item:
            raise TypeError("view offset is not aligned to the storage element")
        return self._offset // st_item

    def _same_item(self) -> bool:
        return self._st.dtype.itemsize == self.itemsize

    # ---------------------------------------------------------- read
    def _values(self):
        """This view's elements as an mx.array of shape ``self.shape``
        (storage dtype; float64 as words)."""
        mx = _mx()
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
            raise NotImplementedError("non-contiguous view of a reinterpreted dtype")
        raw = self._st.arr.view(mx.uint8)
        start = self._offset
        raw = raw[start:start + self.nbytes]
        return raw.view(storage_mx_dtype(self.dtype)).reshape(self.shape)

    def get(self, stream=None, order="C", out=None, blocking=True):
        """Copy to a NumPy array (CuPy's ``ndarray.get``)."""
        mx = _mx()
        values = self._values()
        mx.eval(values)
        host = np.array(values, copy=True)
        if self.dtype == np.float64:
            host = host.view(np.float64)
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
        from woof.backend.facade._ops import scalar_values
        self._write(scalar_values(value, self.dtype))

    def set(self, host, stream=None):
        host = np.asarray(host, dtype=self.dtype)
        if host.shape != self.shape:
            raise ValueError(f"shape mismatch: {host.shape} vs {self.shape}")
        self._write(to_storage_values(host).reshape(self.shape))

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
            if not self.is_c_contiguous():
                raise ValueError("changing the itemsize needs a contiguous array")
            last = self.shape[-1] * self.itemsize // dtype.itemsize
            shape = self.shape[:-1] + (last,)
            return ndarray(shape, dtype, _storage=self._st, _offset=self._offset)
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
        return self

    @property
    def size(self):
        return self._a._st.nbytes

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
