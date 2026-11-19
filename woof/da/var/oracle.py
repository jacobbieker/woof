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
    """Read one dump file; repeated names get ``#2``, ``#3``... suffixes."""
    data = Path(path).read_bytes()
    if data[:8] != _MAGIC:
        raise ValueError(f"{path}: not a woof WRFDA dump (magic {data[:8]!r})")
    off = 8
    endian = None
    out: dict = {}
    while off < len(data):
        name = data[off : off + 32].decode("ascii").strip()
        kind = data[off + 32 : off + 33]
        off += 33
        if endian is None:
            rank_be = int.from_bytes(data[off : off + 4], "big")
            endian = ">" if 0 <= rank_be <= 7 else "<"
        rank = int(np.frombuffer(data[off : off + 4], dtype=endian + "i4")[0])
        off += 4
        shape = tuple(int(v) for v in np.frombuffer(data[off : off + 4 * rank], dtype=endian + "i4"))
        off += 4 * rank
        count = int(np.prod(shape)) if rank else 1
        if kind == b"r":
            dt, size = endian + "f8", 8
        elif kind == b"i":
            dt, size = endian + "i4", 4
        else:
            raise ValueError(f"{path}: record {name!r} has kind {kind!r}")
        a = np.frombuffer(data[off : off + count * size], dtype=dt)
        if a.size != count:
            raise ValueError(f"{path}: record {name!r} runs past the end")
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
