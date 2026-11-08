"""``cupy.ElementwiseKernel`` for the facade.

CuPy turns ``in_params``/``out_params``/``operation`` into a CUDA kernel with
one thread per element and compiles it with NVRTC.  The facade writes the
same kernel -- index ``i``, each array parameter read as its element, each
output a reference to its element, ``raw`` parameters as pointers, scalars as
values, generic type names bound from the arguments' dtypes -- and compiles
it through ``RawModule``, i.e. the Metal translator, with CuPy's options.
"""

from __future__ import annotations

import numpy as np

from woof.backend._elementwise_source import _TPB, ElementwiseSource, ReductionSource


class ElementwiseKernel(ElementwiseSource):
    def __call__(self, *args, size=None, block_size=None, stream=None, **kwargs):
        from woof.backend.facade import _ops
        from woof.backend.facade._array import ndarray
        if len(args) not in (len(self.inputs), len(self.inputs) + len(self.outputs)):
            raise TypeError(f"{self.name}: {len(args)} arguments for {len(self.inputs)} inputs "
                            f"and {len(self.outputs)} outputs")
        ins, outs = list(args[:len(self.inputs)]), list(args[len(self.inputs):])
        types: dict[str, np.dtype] = {}

        def bind(p, value):
            if p.generic:
                dt = value.dtype if hasattr(value, "dtype") else np.asarray(value).dtype
                types.setdefault(p.type, np.dtype(dt))

        for p, a in zip(self.inputs, ins):
            bind(p, a)
        for p, a in zip(self.outputs, outs):
            bind(p, a)

        def dtype_of(p):
            if p.generic:
                if p.type not in types:
                    raise TypeError(f"{self.name}: cannot infer the type {p.type}")
                return types[p.type]
            return np.dtype(p.type)

        # the element shape: non-raw arrays and given non-raw outputs broadcast together
        shapes = [a.shape for p, a in zip(self.inputs, ins) if not p.raw and isinstance(a, (ndarray, np.ndarray))]
        shapes += [a.shape for p, a in zip(self.outputs, outs) if not p.raw]
        shape = np.broadcast_shapes(*shapes) if shapes else (int(size),) if size is not None else None
        if shape is None:
            raise ValueError(f"{self.name}: no array argument fixes the loop size; pass size=")
        n = int(np.prod(shape)) if size is None else int(size)

        call_args, temps = [], []
        is_array = tuple(p.raw or isinstance(a, (ndarray, np.ndarray)) for p, a in zip(self.inputs, ins))
        for p, a in zip(self.inputs, ins):
            dt = dtype_of(p)
            if p.raw:
                call_args.append(_ops.asarray(a))
            elif isinstance(a, (ndarray, np.ndarray)):
                a = _ops.asarray(a)
                if a.dtype != dt:
                    a = a.astype(dt)
                call_args.append(_ops.ascontiguousarray(_ops.broadcast_to(a, shape)))
            else:
                call_args.append(dt.type(a))
        results = []
        for k, p in enumerate(self.outputs):
            dt = dtype_of(p)
            if k < len(outs):
                o = outs[k]
                if not p.raw and not o.is_c_contiguous():
                    # An output parameter is read-write, as in CuPy: the
                    # operation may read it (``if (x < t) x = 0``), so the
                    # dense stand-in starts with the view's words.
                    tmp = _ops.ascontiguousarray(o)
                    if tmp.dtype != dt:
                        tmp = tmp.astype(dt)
                    temps.append((o, tmp))
                    call_args.append(tmp)
                else:
                    call_args.append(o)
                results.append(o)
            else:
                o = _ops.empty(shape, dt)
                call_args.append(o)
                results.append(o)
        call_args.append(np.int64(n))
        key = (tuple(str(types.get(t)) for t in sorted(types)), is_array)
        fn = self._function(key, types, is_array)
        if n:
            fn(((n + _TPB - 1) // _TPB,), (_TPB,), tuple(call_args))
        for o, tmp in temps:
            _ops.copyto(o, tmp)
        if self.no_return:
            return None
        if len(results) == 1 and not self.return_tuple:
            return results[0]
        return tuple(results)

    def _function(self, key, types, is_array):
        fn = self._modules.get(key)
        if fn is None:
            from woof.backend.facade import RawModule
            code = self._source(types, is_array)
            fn = self._modules[key] = RawModule(code=code, options=self.options or ("-std=c++17",)) \
                .get_function(self.name)
        return fn

class ReductionKernel(ReductionSource):
    __doc__ = ReductionSource.__doc__

    def __call__(self, *args, axis=None, keepdims=False, out=None, stream=None, **_):
        from woof.backend.facade import _ops
        from woof.backend.facade._array import ndarray
        if len(args) not in (len(self.inputs), len(self.inputs) + 1):
            raise TypeError(f"{self.name}: {len(args)} arguments for {len(self.inputs)} inputs")
        ins = list(args[:len(self.inputs)])
        if len(args) > len(self.inputs):
            out = args[-1]
        types: dict[str, np.dtype] = {}
        for p, a in zip(self.inputs, ins):
            if p.generic:
                types.setdefault(p.type, np.dtype(a.dtype if hasattr(a, "dtype") else np.asarray(a).dtype))
        for p in self.outputs:
            if p.generic and p.type not in types:
                if out is None:
                    raise TypeError(f"{self.name}: cannot infer the type {p.type}")
                types[p.type] = np.dtype(out.dtype)

        def dtype_of(p):
            return types[p.type] if p.generic else np.dtype(p.type)

        arrays = [_ops.asarray(a) for a in ins if isinstance(a, (ndarray, np.ndarray))]
        shape = np.broadcast_shapes(*[a.shape for a in arrays]) if arrays else ()
        nd = len(shape)
        axes = tuple(range(nd)) if axis is None else tuple(sorted(a % nd for a in np.atleast_1d(axis)))
        keep = tuple(i for i in range(nd) if i not in axes)
        out_shape = tuple(1 if i in axes else shape[i] for i in range(nd)) if keepdims \
            else tuple(shape[i] for i in keep)
        n_out = int(np.prod([shape[i] for i in keep])) if keep else 1
        n_red = int(np.prod([shape[i] for i in axes])) if axes else 1

        call_args, is_array = [], []
        for p, a in zip(self.inputs, ins):
            dt = dtype_of(p)
            if isinstance(a, (ndarray, np.ndarray)):
                a = _ops.asarray(a)
                if a.dtype != dt:
                    a = a.astype(dt)
                # (kept axes..., reduced axes...), dense: element (o, r) at o*n_red + r
                b = _ops.broadcast_to(a, shape).transpose(keep + axes)
                call_args.append(_ops.ascontiguousarray(b))
                is_array.append(True)
            else:
                call_args.append(dt.type(a))
                is_array.append(False)
        odt = dtype_of(self.outputs[0])
        result = _ops.empty((n_out,), odt)
        call_args += [result, np.int64(n_out), np.int64(n_red)]
        key = (tuple(str(types.get(t)) for t in sorted(types)), tuple(is_array))
        fn = self._modules.get(key)
        if fn is None:
            from woof.backend.facade import RawModule
            fn = self._modules[key] = RawModule(code=self._source(types, is_array),
                                                options=self.options or ("-std=c++17",)) \
                .get_function(self.name)
        if n_out:
            fn((((n_out + _TPB - 1) // _TPB),), (_TPB,), tuple(call_args))
        result = result.reshape(out_shape)
        if out is not None:
            _ops.copyto(out, result)
            return out
        return result
