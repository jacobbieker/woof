"""``cupy.ElementwiseKernel`` and ``cupy.ReductionKernel`` for the host facade.

CuPy turns ``in_params``/``out_params``/``operation`` into a CUDA kernel with
one thread per element.  The host facade writes the same kernel -- index
``i``, each array parameter read as its element, each output a reference to
its element, ``raw`` parameters as pointers, scalars as values, generic
type names bound from the arguments' dtypes -- and builds it with the host
compiler (:mod:`woof.backend.host.loader`), with CuPy's options.
"""

from __future__ import annotations

import numpy as np

from woof.backend._elementwise_source import _TPB, ElementwiseSource, ReductionSource


def _module(code, options, name):
    from woof.backend.host.loader import load_unit
    import hashlib
    unit = f"elementwise-{name}-" + hashlib.sha256(code.encode()).hexdigest()[:12]
    return load_unit(unit, code, options=tuple(options or ()) or ("-std=c++17",)).get_function(name)


def _is_array(a):
    from woof.backend.hostfacade import ndarray
    return isinstance(a, (np.ndarray, ndarray))


class ElementwiseKernel(ElementwiseSource):
    def __call__(self, *args, size=None, block_size=None, stream=None, **kwargs):
        from woof.backend.hostfacade import ndarray
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

        shapes = [a.shape for p, a in zip(self.inputs, ins) if not p.raw and _is_array(a)]
        shapes += [a.shape for p, a in zip(self.outputs, outs) if not p.raw]
        shape = np.broadcast_shapes(*shapes) if shapes else (int(size),) if size is not None else None
        if shape is None:
            raise ValueError(f"{self.name}: no array argument fixes the loop size; pass size=")
        n = int(np.prod(shape)) if size is None else int(size)

        call_args, temps = [], []
        is_array = tuple(p.raw or _is_array(a) for p, a in zip(self.inputs, ins))
        for p, a in zip(self.inputs, ins):
            dt = dtype_of(p)
            if p.raw:
                call_args.append(a)
            elif _is_array(a):
                call_args.append(np.ascontiguousarray(np.broadcast_to(np.asarray(a, dtype=dt), shape)))
            else:
                call_args.append(dt.type(a))
        results = []
        for k, p in enumerate(self.outputs):
            dt = dtype_of(p)
            if k < len(outs):
                o = outs[k]
                if not p.raw and not o.flags.c_contiguous:
                    tmp = np.empty(o.shape, dt)
                    temps.append((o, tmp))
                    call_args.append(tmp)
                else:
                    call_args.append(o)
                results.append(o)
            else:
                o = ndarray._of(np.empty(shape, dt))
                call_args.append(o)
                results.append(o)
        call_args.append(np.int64(n))
        key = (tuple(str(types.get(t)) for t in sorted(types)), is_array)
        fn = self._modules.get(key)
        if fn is None:
            fn = self._modules[key] = _module(self._source(types, is_array), self.options, self.name)
        if n:
            fn(((n + _TPB - 1) // _TPB,), (_TPB,), tuple(call_args))
        for o, tmp in temps:
            o[...] = tmp
        if self.no_return:
            return None
        if len(results) == 1 and not self.return_tuple:
            return results[0]
        return tuple(results)

class ReductionKernel(ReductionSource):
    __doc__ = ReductionSource.__doc__

    def __call__(self, *args, axis=None, keepdims=False, out=None, stream=None, **_):
        from woof.backend.hostfacade import ndarray
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

        arrays = [a for a in ins if _is_array(a)]
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
            if _is_array(a):
                b = np.broadcast_to(np.asarray(a, dtype=dt), shape).transpose(keep + axes)
                call_args.append(np.ascontiguousarray(b))
                is_array.append(True)
            else:
                call_args.append(dt.type(a))
                is_array.append(False)
        odt = dtype_of(self.outputs[0])
        result = ndarray._of(np.empty((n_out,), odt))
        call_args += [result, np.int64(n_out), np.int64(n_red)]
        key = (tuple(str(types.get(t)) for t in sorted(types)), tuple(is_array))
        fn = self._modules.get(key)
        if fn is None:
            fn = self._modules[key] = _module(self._source(types, is_array), self.options, self.name)
        if n_out:
            fn((((n_out + _TPB - 1) // _TPB),), (_TPB,), tuple(call_args))
        result = result.reshape(out_shape)
        if out is not None:
            out[...] = result
            return out
        return result
