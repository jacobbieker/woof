"""``cupy.ElementwiseKernel`` for the facade.

CuPy turns ``in_params``/``out_params``/``operation`` into a CUDA kernel with
one thread per element and compiles it with NVRTC.  The facade writes the
same kernel -- index ``i``, each array parameter read as its element, each
output a reference to its element, ``raw`` parameters as pointers, scalars as
values, generic type names bound from the arguments' dtypes -- and compiles
it through ``RawModule``, i.e. the Metal translator, with CuPy's options.
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


def _params(text: str):
    return [_Param(p.strip()) for p in text.split(",") if p.strip()]


class ElementwiseKernel:
    def __init__(self, in_params, out_params, operation, name="kernel", reduce_dims=True,
                 preamble="", no_return=False, return_tuple=False, loop_prep="",
                 after_loop="", options=(), **_):
        self.inputs = _params(in_params)
        self.outputs = _params(out_params)
        self.operation = operation
        self.name = re.sub(r"\W", "_", name)
        self.preamble = preamble or ""
        self.loop_prep = loop_prep or ""
        self.no_return = no_return
        self.return_tuple = return_tuple
        self.options = tuple(options or ())
        self._modules = {}

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
                    tmp = _ops.empty(o.shape, dt)
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
