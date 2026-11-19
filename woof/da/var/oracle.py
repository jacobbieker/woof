"""The WRFDA oracle: reading the debug build's array dumps.

``tools/wrfda_oracle/woof_dump.inc`` is hooked into a copy of the pinned
WRFDA 4.7.1 tree (never the pinned tree itself) by
``tools/wrfda_oracle/apply_dump_patch.sh``.  The patched build writes stream
files (``woof_be.bin``, ``woof_xb.bin``, ``woof_iv.bin``, ``woof_u.bin``,
``woof_final.bin``) of records::

    name (32 chars) | kind ('r' real*8, 'i' integer*4) | rank (int4)
    | shape (rank x int4) | values (column-major)

after an 8-byte ``WOOFDUMP`` magic, in the build's byte order (WRFDA's
gfortran configuration sets ``-fconvert=big-endian``).  :func:`read_dump`
returns them as binary64 / int32 NumPy arrays in Fortran index order.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

__all__ = ["read_dump", "array_digest"]

_MAGIC = b"WOOFDUMP"


def read_dump(path) -> dict:
    """Read one dump file; repeated names get ``#2``, ``#3``... suffixes.

    The byte order is the one in which the whole file parses: a scalar
    record's rank (0) reads the same either way round, so no single record
    can decide it.
    """
    data = Path(path).read_bytes()
    if data[:8] != _MAGIC:
        raise ValueError(f"{path}: not a woof WRFDA dump (magic {data[:8]!r})")
    errors = []
    for endian in (">", "<"):
        try:
            return _parse(data, endian, path)
        except ValueError as exc:
            errors.append(f"{'big' if endian == '>' else 'little'}-endian: {exc}")
    raise ValueError(f"{path}: parses in neither byte order ({'; '.join(errors)})")


def _parse(data: bytes, endian: str, path) -> dict:
    off = 8
    out: dict = {}
    while off < len(data):
        if off + 37 > len(data):
            raise ValueError("truncated record header")
        name = data[off : off + 32].decode("ascii").strip()
        kind = data[off + 32 : off + 33]
        off += 33
        rank = int(np.frombuffer(data[off : off + 4], dtype=endian + "i4")[0])
        off += 4
        if not 0 <= rank <= 7 or off + 4 * rank > len(data):
            raise ValueError(f"record {name!r} has rank {rank}")
        shape = tuple(int(v) for v in np.frombuffer(data[off : off + 4 * rank], dtype=endian + "i4"))
        off += 4 * rank
        if any(n < 0 for n in shape):
            raise ValueError(f"record {name!r} has shape {shape}")
        count = int(np.prod(shape)) if rank else 1
        if kind == b"r":
            dt, size = endian + "f8", 8
        elif kind == b"i":
            dt, size = endian + "i4", 4
        else:
            raise ValueError(f"record {name!r} has kind {kind!r}")
        if off + count * size > len(data):
            raise ValueError(f"record {name!r} runs past the end")
        a = np.frombuffer(data[off : off + count * size], dtype=dt)
        off += count * size
        a = a.astype(np.float64 if kind == b"r" else np.int32)
        a = a.reshape(shape, order="F") if rank else a[0]
        key, n = name, 1
        while key in out:
            n += 1
            key = f"{name}#{n}"
        out[key] = a
    return out


def array_digest(a) -> str:
    """SHA-256 of an array's binary64 (or int32) words, little-endian, C order.

    The committed O1/O2 fixtures record these rather than the arrays.
    """
    a = np.asarray(a)
    dt = "<f8" if a.dtype.kind == "f" else "<i4"
    return hashlib.sha256(np.ascontiguousarray(a, dtype=dt).tobytes()).hexdigest()
