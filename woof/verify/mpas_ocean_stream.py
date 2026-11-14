"""The self-describing stream of the generic MPAS-Ocean oracle drivers.

``tools/mpas_ocean_oracle/oracle_stream.F90`` reads it: dt, every config by
name, every field by pool and name with its kind and Fortran shape. A case is
``{"precision", "dims", "config", "dt", "pools": {pool: {name: value}}}`` and
a field table is ``(pool, name, kind, Fortran shape)`` rows whose shapes name
the case's dimensions (``C1``/``E1``/``V1`` are nCells+1 ..., ``LP1`` is
nVertLevels+1).
"""
from __future__ import annotations

import numpy as np

from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

STRKIND = 512
#: oracle_stream.F90 reads pool names this wide
POOL_WIDTH = 24
#: Field kinds: real array, integer array, dimension, dimension array,
#: absent (a package that is off), scratch field
REAL, INT, DIM, DIMS, ABSENT, SCRATCH = 1, 2, 3, 4, 5, 6


def shape(case: dict, fortran_shape: tuple) -> tuple[int, ...]:
    """A field's C-order shape: the Fortran shape's dimension names resolved, reversed."""
    d = case["dims"]
    sizes = {"C1": d["nCells"] + 1, "E1": d["nEdges"] + 1, "V1": d.get("nVertices", 0) + 1, "L": d["nVertLevels"],
             "LP1": d["nVertLevels"] + 1, "T": d.get("nTracers", 0), "ME": d.get("maxEdges", 0),
             "ME2": d.get("maxEdges2", 0), "VD": d.get("vertexDegree", 0), "NA": d.get("nAdvectionCells", 0),
             "NCAT": d.get("nCategories", 0)}
    return tuple(sizes[s] if isinstance(s, str) else s for s in reversed(fortran_shape))


def _text(value: str, width: int) -> bytes:
    raw = value.encode("ascii")
    if len(raw) > width:
        raise ValueError(f"{value!r} is longer than {width} characters")
    return raw.ljust(width, b" ")


def real_dtype(case: dict) -> np.dtype:
    return np.dtype(PRECISIONS[case["precision"]]).newbyteorder("<")


def input_bytes(fields, case: dict) -> bytes:
    """The driver's input stream for ``case`` over the field table ``fields``."""
    real = real_dtype(case)
    config = case["config"]
    parts = [np.array(case["dt"], real).tobytes(), np.array([len(config), len(fields)], "<i4").tobytes()]
    for name in sorted(config):
        value = config[name]
        if isinstance(value, bool):
            parts += [_text(name, 64), np.array([3, int(value)], "<i4").tobytes()]
        elif isinstance(value, int):
            parts += [_text(name, 64), np.array([2, value], "<i4").tobytes()]
        elif isinstance(value, float):
            parts += [_text(name, 64), np.array([1], "<i4").tobytes(), np.array(value, real).tobytes()]
        else:
            parts += [_text(name, 64), np.array([4], "<i4").tobytes(), _text(value, STRKIND)]
    for pool, name, kind, fshape in fields:
        value = case["pools"][pool][name]
        stream_kind = ABSENT if kind in (REAL, INT) and value is None else kind
        rank = len(fshape)
        dims = [shape(case, fshape)[::-1][i] if i < rank else 1 for i in range(3)] if rank else [1, 1, 1]
        code_rank = 4 if (stream_kind == ABSENT and kind == INT) else rank
        parts += [_text(pool, POOL_WIDTH), _text(name, 64),
                  np.array([stream_kind, code_rank, *dims], "<i4").tobytes()]
        if stream_kind in (REAL, INT):
            if tuple(np.shape(value)) != shape(case, fshape):
                raise ValueError(f"{pool}/{name} has shape {np.shape(value)}, not {shape(case, fshape)}")
            parts.append(np.ascontiguousarray(value, real if kind == REAL else "<i4").tobytes())
        elif stream_kind == DIM:
            parts.append(np.array(value, "<i4").tobytes())
        elif stream_kind == DIMS:
            parts.append(np.ascontiguousarray(value, "<i4").tobytes())
    return b"".join(parts)


def read_outputs(fields, raw: bytes, case: dict) -> dict[str, np.ndarray]:
    """The driver's output stream -> ``pool/name`` arrays (present real and integer fields)."""
    out, off = {}, 0
    for pool, name, kind, fshape in fields:
        if kind not in (REAL, INT) or case["pools"][pool][name] is None:
            continue
        dtype = real_dtype(case) if kind == REAL else np.dtype("<i4")
        shp = shape(case, fshape)
        n = int(np.prod(shp))
        out[f"{pool}/{name}"] = np.frombuffer(raw, dtype, n, off).reshape(shp).astype(dtype.newbyteorder("="))
        off += dtype.itemsize * n
    if off != len(raw):
        raise ValueError(f"oracle output: {len(raw) - off} trailing bytes")
    return out


def word_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    """Words that differ as bit patterns (so -0.0 != 0.0), real or integer."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape or a.dtype.itemsize != b.dtype.itemsize:
        return int(max(a.size, b.size))
    word = np.dtype(f"u{a.dtype.itemsize}")
    return int(np.count_nonzero(a.view(word) != b.view(word)))


def on_device(fields, case: dict) -> dict[str, dict]:
    """The case's pools as device arrays (dimensions stay host values, absent fields None)."""
    import cupy as cp
    dev: dict[str, dict] = {}
    for pool, name, kind, _ in fields:
        value = case["pools"][pool][name]
        if kind in (REAL, INT) and value is not None:
            dev.setdefault(pool, {})[name] = cp.asarray(np.ascontiguousarray(value))
        elif kind in (DIM, DIMS):
            dev.setdefault(pool, {})[name] = value
        else:
            dev.setdefault(pool, {})[name] = None
    return dev
