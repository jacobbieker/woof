"""The kernel text shared by the two ``ElementwiseKernel`` facades.

``woof.backend.facade`` (Metal, through the translator) and
``woof.backend.hostfacade`` (the host compiler) read CuPy's
``in_params``/``out_params`` the same way and write the same kernel source
from them; only how each launches and moves arrays differs.  The parsing and
the source writers live here once, so the two backends cannot drift apart in
the text they compile.
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
        # CuPy's ParameterInfo: the C type ("T" for a generic), the dtype
        # (None for a generic), constness (inputs), and raw-ness
        self.ctype = self.type if self.generic else _CTYPES[self.type]
        self.dtype = None if self.generic else np.dtype(self.type)
        self.is_const = False


def _params(text: str):
    return [_Param(p.strip()) for p in text.split(",") if p.strip()]


class ElementwiseSource:
    """CuPy's ``ElementwiseKernel`` parameters and kernel text."""

    def __init__(self, in_params, out_params, operation, name="kernel", reduce_dims=True,
                 preamble="", no_return=False, return_tuple=False, loop_prep="",
                 after_loop="", options=(), **_):
        self.inputs = _params(in_params)
        self.outputs = _params(out_params)
        for p in self.inputs:
            p.is_const = True
        # CuPy's names for the parameter lists
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
        self._modules = {}

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


class ReductionSource:
    """``cupy.ReductionKernel``: ``map_expr`` per element, folded with
    ``reduce_expr`` (``a``, ``b``) from ``identity`` over the reduced axes,
    then ``post_map_expr`` (``a``) into each output.

    One thread per output element folds its elements serially, in index
    order: deterministic, and exact for the bitwise/logical reductions woof
    uses.  (A floating sum would differ from CuPy's tree order by rounding.)
    """

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
        self._modules = {}

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
