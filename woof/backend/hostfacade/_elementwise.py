"""``cupy.ElementwiseKernel`` and ``cupy.ReductionKernel`` for the host facade.

CuPy turns ``in_params``/``out_params``/``operation`` into a CUDA kernel with
one thread per element.  The host facade writes the same kernel -- index
``i``, each array parameter read as its element, each output a reference to
its element, ``raw`` parameters as pointers, scalars as values, generic
type names bound from the arguments' dtypes -- and builds it with the host
compiler (:mod:`woof.backend.host.loader`), with CuPy's options.
"""

from __future__ import annotations

import re

import numpy as np

_CTYPES = {
    "float32": "float", "float64": "double", "float16": "half",
    "int8": "signed char", "int16": "short", "int32": "int", "int64": "long long",
    "uint8": "unsigned char", "uint16": "unsigned short", "uint32": "unsigned int",
    "uint64": "unsigned long long", "bool": "bool",
}
_TPB = 128


class _Param:
    def __init__(self, text: str):
        words = text.split()
        self.raw = words[0] == "raw"
        if self.raw:
            words = words[1:]
        if len(words) != 2:
            raise ValueError(f"cannot read ElementwiseKernel parameter {text!r}")
        self.type, self.name = words
        self.generic = self.type not in _CTYPES
        self.ctype = self.type if self.generic else _CTYPES[self.type]
        self.dtype = None if self.generic else np.dtype(self.type)
        self.is_const = False


def _params(text: str):
    return [_Param(p.strip()) for p in text.split(",") if p.strip()]


def _module(code, options, name):
    from woof.backend.host.loader import load_unit
    import hashlib
    unit = f"elementwise-{name}-" + hashlib.sha256(code.encode()).hexdigest()[:12]
    return load_unit(unit, code, options=tuple(options or ()) or ("-std=c++17",)).get_function(name)


def _is_array(a):
    return isinstance(a, np.ndarray)


class ElementwiseKernel:
    def __init__(self, in_params, out_params, operation, name="kernel", reduce_dims=True,
                 preamble="", no_return=False, return_tuple=False, loop_prep="",
                 after_loop="", options=(), **_):
        self.inputs = _params(in_params)
        self.outputs = _params(out_params)
        for p in self.inputs:
            p.is_const = True
        self.in_params = tuple(self.inputs)
        self.out_params = tuple(self.outputs)
        self.params = self.in_params + self.out_params
        self.nin, self.nout = len(self.inputs), len(self.outputs)
        self.operation = operation
        self.name = re.sub(r"\W", "_", name)
        self.preamble = preamble or ""
        self.loop_prep = loop_prep or ""
        self.no_return = no_return
        self.return_tuple = return_tuple
        self.options = tuple(options or ())
        self._functions = {}

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
                o = np.empty(shape, dt).view(ndarray)
                call_args.append(o)
                results.append(o)
        call_args.append(np.int64(n))
        key = (tuple(str(types.get(t)) for t in sorted(types)), is_array)
        fn = self._functions.get(key)
        if fn is None:
            fn = self._functions[key] = _module(self._source(types, is_array), self.options, self.name)
        if n:
            fn(((n + _TPB - 1) // _TPB,), (_TPB,), tuple(call_args))
        for o, tmp in temps:
            o[...] = tmp
        if self.no_return:
            return None
        if len(results) == 1 and not self.return_tuple:
            return results[0]
        return tuple(results)

    def _source(self, types, is_array):
        def ctype(p):
            if p.generic:
                return _CTYPES[np.dtype(types[p.type]).name]
            return _CTYPES[p.type]

        params, loads = [], []
        for p, array in zip(self.inputs, is_array):
            t = ctype(p)
            if p.raw:
                params.append(f"const {t}* {p.name}")
            elif array:
                params.append(f"const {t}* __woof_in_{p.name}")
                loads.append(f"const {t} {p.name} = __woof_in_{p.name}[i];")
            else:
                params.append(f"{t} {p.name}")
        for p in self.outputs:
            t = ctype(p)
            if p.raw:
                params.append(f"{t}* {p.name}")
            else:
                params.append(f"{t}* __woof_out_{p.name}")
                loads.append(f"{t}& {p.name} = __woof_out_{p.name}[i];")
        typedefs = "".join(f"typedef {_CTYPES[np.dtype(dt).name]} {name};\n" for name, dt in types.items())
        return (f"{typedefs}{self.preamble}\n"
                f'extern "C" __global__ void {self.name}({", ".join(params)}, long long __woof_n) {{\n'
                f"    {self.loop_prep}\n"
                f"    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
                f"    if (i >= __woof_n) return;\n"
                + "".join(f"    {line}\n" for line in loads)
                + f"    {self.operation};\n}}\n")


class ReductionKernel:
    """``cupy.ReductionKernel``: ``map_expr`` per element, folded with
    ``reduce_expr`` (``a``, ``b``) from ``identity`` over the reduced axes,
    then ``post_map_expr`` (``a``) into each output.

    One thread per output element folds its elements serially, in index
    order: deterministic, and exact for integer and logical reductions.  (A
    floating sum differs from CuPy's tree order by rounding.)"""

    def __init__(self, in_params, out_params, map_expr, reduce_expr, post_map_expr,
                 identity, name="reduce_kernel", reduce_type=None, reduce_dims=True,
                 preamble="", options=(), **_):
        self.inputs = _params(in_params)
        self.outputs = _params(out_params)
        if len(self.outputs) != 1 or any(p.raw for p in self.inputs + self.outputs):
            raise ValueError("ReductionKernel: one output and no raw parameters")
        self.map_expr, self.reduce_expr, self.post_map_expr = map_expr, reduce_expr, post_map_expr
        self.identity = identity
        self.name = re.sub(r"\W", "_", name)
        self.reduce_type = reduce_type
        self.preamble = preamble or ""
        self.options = tuple(options or ())
        self._functions = {}

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
        result = np.empty((n_out,), odt).view(ndarray)
        call_args += [result, np.int64(n_out), np.int64(n_red)]
        key = (tuple(str(types.get(t)) for t in sorted(types)), tuple(is_array))
        fn = self._functions.get(key)
        if fn is None:
            fn = self._functions[key] = _module(self._source(types, is_array), self.options, self.name)
        if n_out:
            fn((((n_out + _TPB - 1) // _TPB),), (_TPB,), tuple(call_args))
        result = result.reshape(out_shape)
        if out is not None:
            out[...] = result
            return out
        return result

    def _source(self, types, is_array):
        def ctype(p):
            return _CTYPES[np.dtype(types[p.type]).name] if p.generic else _CTYPES[p.type]

        out = self.outputs[0]
        rtype = self.reduce_type or ctype(out)
        params, loads = [], []
        for p, array in zip(self.inputs, is_array):
            t = ctype(p)
            if array:
                params.append(f"const {t}* __woof_in_{p.name}")
                loads.append(f"const {t} {p.name} = __woof_in_{p.name}[__woof_o * __woof_r + __woof_k];")
            else:
                params.append(f"{t} {p.name}")
        params.append(f"{ctype(out)}* __woof_out")
        typedefs = "".join(f"typedef {_CTYPES[np.dtype(dt).name]} {nm};\n" for nm, dt in types.items())
        return (f"{typedefs}{self.preamble}\n"
                f'extern "C" __global__ void {self.name}({", ".join(params)}, '
                f"long long __woof_n, long long __woof_r) {{\n"
                f"    long long __woof_o = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
                f"    if (__woof_o >= __woof_n) return;\n"
                f"    {rtype} a = ({rtype})({self.identity});\n"
                f"    for (long long __woof_k = 0; __woof_k < __woof_r; ++__woof_k) {{\n"
                + "".join(f"        {line}\n" for line in loads)
                + f"        {rtype} b = ({rtype})({self.map_expr});\n"
                f"        a = ({rtype})({self.reduce_expr});\n"
                f"    }}\n"
                f"    {ctype(out)} {out.name};\n"
                f"    {self.post_map_expr};\n"
                f"    __woof_out[__woof_o] = {out.name};\n}}\n")
