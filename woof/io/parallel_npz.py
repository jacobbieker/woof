"""An ``np.savez`` archive written by many threads, byte for byte the archive ``np.savez`` writes.

Breakage it removes (sentinel, 3 km DA test on camp-us-20230710, 2026-10-27): the member restart
(``gpuwmrst_*.npz``) was written by ``np.savez`` on ONE core -- per 16 MiB chunk a ``tobytes`` copy,
a CRC-32 and a ``write`` into the page cache, one after another -- 16 to 68 s per member when six
legs ended together, while the card sat at 0-7 %.  The archive is uncompressed (restart.py: the
``.npy`` payload is the array's memory bytes), so its CRCs need no single thread: the CRC-32 of a
member is combined from the CRCs of its pieces, and ``zlib.crc32`` releases the GIL.

So every array is cut into pieces of at most ``PIECE_BYTES``; worker threads CRC the pieces while
the calling thread writes the same bytes, in file order, straight from the array's memory (no
``tobytes`` copy).  Writes into one file stay on one thread: Linux takes the inode lock for every
buffered write, so more writers would only queue (measured), and one in-order stream
keeps the disk pattern np.savez had.  The
local headers, central directory and end records are then written as ``zipfile`` writes them for
``np.savez``: STORED members, a ZIP64 extra on every local header (``force_zip64=True``), the
same versions, flags, permissions, timestamp and end records.  The file is byte-identical to
``np.savez``'s (tests/test_parallel_npz.py compares them), so every reader of a restart
(``np.load``, ``zipfile``, any zip tool) reads it unchanged.
"""
from __future__ import annotations

import io
import os
import struct
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# One piece of work: big enough that the per-call overhead vanishes, small enough that one big
# array spreads over every worker.
PIECE_BYTES = 32 << 20

_ZIP64_LIMIT = (1 << 31) - 1
_ZIP_FILECOUNT_LIMIT = (1 << 16) - 1
_ZIP64_VERSION = 45
_DEFAULT_VERSION = 20
_CREATE_SYSTEM = 0 if sys.platform == "win32" else 3
_EXTERNAL_ATTR = 0o600 << 16           # zipfile's ?rw------- for members opened for writing
_DATE_TIME = (1980, 1, 1, 0, 0, 0)     # zipfile.ZipInfo's default, what ZipFile.open(name, "w") stamps

_LOCAL = struct.Struct("<4s2B4HL2L2H")
_CENTRAL = struct.Struct("<4s4B4HL2L5H2L")
_END = struct.Struct("<4s4H2LH")
_END64 = struct.Struct("<4sQ2H2L4Q")
_END64_LOCATOR = struct.Struct("<4sLQL")
_ZIP64_LOCAL_EXTRA = struct.Struct("<HHQQ")


def default_workers() -> int:
    """Worker threads: the cores this process may use, at most 16 (the page cache saturates first)."""
    override = os.environ.get("WOOF_NPZ_WRITE_THREADS")
    if override:
        return max(1, int(override))
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    return max(1, min(16, cores))


# --- CRC-32 combination (zlib's crc32_combine, which Python's zlib does not expose) -------------

def _gf2_times(matrix, vector):
    total = 0
    row = 0
    while vector:
        if vector & 1:
            total ^= matrix[row]
        vector >>= 1
        row += 1
    return total


def _gf2_square(matrix):
    return [_gf2_times(matrix, matrix[n]) for n in range(32)]


def _zero_byte_operators():
    """``ops[k]`` advances a CRC-32 over ``2**k`` zero bytes (zlib's crc32_combine matrices)."""
    one_bit = [0xEDB88320] + [1 << n for n in range(31)]
    operator = _gf2_square(_gf2_square(_gf2_square(one_bit)))      # 8 zero bits = 1 byte
    ops = [operator]
    for _ in range(47):                                              # up to 2**47 bytes
        operator = _gf2_square(operator)
        ops.append(operator)
    return ops


_ZERO_BYTES = []
_ZERO_BYTES_LOCK = threading.Lock()


def crc32_combine(crc1: int, crc2: int, length2: int) -> int:
    """CRC-32 of ``A + B`` from ``crc32(A)``, ``crc32(B)`` and ``len(B)``."""
    if not _ZERO_BYTES:
        with _ZERO_BYTES_LOCK:
            if not _ZERO_BYTES:
                _ZERO_BYTES.extend(_zero_byte_operators())
    k = 0
    while length2 > 0:
        if length2 & 1:
            crc1 = _gf2_times(_ZERO_BYTES[k], crc1)
        length2 >>= 1
        k += 1
    return (crc1 ^ crc2) & 0xFFFFFFFF


# --- members ------------------------------------------------------------------------------------

def _staged_layout(value):
    """(shape, dtype) of a member staged through a shared host buffer, else None."""
    staged = getattr(value, "_value", None)
    if staged is not None and not isinstance(value, np.ndarray):
        return tuple(int(n) for n in staged.shape), np.dtype(staged.dtype)
    return None


def _npy_header(value) -> bytes:
    """The .npy header ``_npy_parts`` will write for ``value``.  A staged member's comes from
    its C-ordered shape and dtype, so it is not materialised before its turn."""
    layout = _staged_layout(value)
    if layout is None:
        return _npy_parts(value)[0]
    shape, dtype = layout
    info = {"descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": False, "shape": shape}
    fmt = np.lib.format
    header = io.BytesIO()
    write_header = getattr(fmt, "_write_array_header", None)
    if write_header is not None:
        write_header(header, info, None)
    else:                                                    # pragma: no cover - older numpy
        fmt.write_array_header_1_0(header, info)
    return header.getvalue()


def _member_nbytes(value) -> int:
    layout = _staged_layout(value)
    if layout is None:
        array = np.asanyarray(value)
        return 0 if array.itemsize == 0 else int(array.nbytes)
    shape, dtype = layout
    return int(np.prod(shape, dtype=np.int64)) * dtype.itemsize


def _npy_parts(value):
    """(.npy header bytes, data buffer) exactly as ``np.lib.format.write_array`` writes them."""
    array = np.asanyarray(value)
    if array.dtype.hasobject:
        raise ValueError("an object array cannot go into a pickle-free archive")
    fmt = np.lib.format
    header = io.BytesIO()
    write_header = getattr(fmt, "_write_array_header", None)
    if write_header is not None:
        write_header(header, fmt.header_data_from_array_1_0(array), None)
    else:                                                    # pragma: no cover - older numpy
        fmt.write_array_header_1_0(header, fmt.header_data_from_array_1_0(array))
    if array.itemsize == 0 or array.size == 0:
        data = memoryview(b"")
    elif array.flags.f_contiguous and not array.flags.c_contiguous:
        data = memoryview(np.ascontiguousarray(array.T).reshape(-1)).cast("B")
    else:
        data = memoryview(np.ascontiguousarray(array).reshape(-1)).cast("B")
    return header.getvalue(), data


def _dos_stamp(date_time):
    dosdate = (date_time[0] - 1980) << 9 | date_time[1] << 5 | date_time[2]
    dostime = date_time[3] << 11 | date_time[4] << 5 | (date_time[5] // 2)
    return dostime, dosdate


def _local_header(name: bytes, crc: int, size: int, stamp) -> bytes:
    extra = _ZIP64_LOCAL_EXTRA.pack(1, 16, size, size)
    return _LOCAL.pack(b"PK\003\004", _ZIP64_VERSION, 0, 0, 0, stamp[0], stamp[1], crc,
                       0xFFFFFFFF, 0xFFFFFFFF, len(name), len(extra)) + name + extra


def _central_header(name: bytes, crc: int, size: int, offset: int, stamp) -> bytes:
    fields = []
    if size > _ZIP64_LIMIT:
        fields += [size, size]
        stored_size = 0xFFFFFFFF
    else:
        stored_size = size
    if offset > _ZIP64_LIMIT:
        fields.append(offset)
        stored_offset = 0xFFFFFFFF
    else:
        stored_offset = offset
    extra = (struct.pack("<HH" + "Q" * len(fields), 1, 8 * len(fields), *fields) if fields else b"")
    # The local header's ZIP64 extra already raised both versions to 4.5 (ZipInfo.FileHeader).
    return _CENTRAL.pack(b"PK\001\002", _ZIP64_VERSION, _CREATE_SYSTEM, _ZIP64_VERSION, 0, 0, 0,
                         stamp[0], stamp[1], crc, stored_size, stored_size, len(name), len(extra),
                         0, 0, 0, _EXTERNAL_ATTR, stored_offset) + name + extra


def _member_name(key: str) -> bytes:
    name = key + ".npy"                  # np.savez appends it to every key
    try:
        return name.encode("ascii")
    except UnicodeEncodeError:
        # zipfile would set the UTF-8 flag bit; restart keys are ASCII, so refuse rather than
        # write a header this writer has not been compared against.
        raise ValueError(f"archive member name {key!r} is not ASCII") from None


def _write_all(fd: int, data) -> None:
    view = memoryview(data).cast("B")
    while view.nbytes:
        view = view[os.write(fd, view):]


def _write_at(fd: int, data: bytes, offset: int) -> None:
    os.lseek(fd, offset, os.SEEK_SET)
    _write_all(fd, data)


def write_npz(path, arrays, *, workers: int | None = None) -> int:
    """Write ``arrays`` (name -> array, in archive order) as ``np.savez(path, **arrays)`` would.

    Returns the archive's size in bytes.  Every member carries ``zipfile``'s default timestamp,
    1980-01-01 00:00:00, as ``np.savez``'s do (``ZipFile.open(name, "w")`` stamps no clock).

    Members are materialised one at a time, in archive order, each immediately before it is
    written, as ``np.savez`` does: a checkpoint's device members are staged through ONE reused
    host buffer (woof.io.restart._StagedMember), so materialising them all first left every
    staged member holding the bytes of the last one staged.  The layout (headers, sizes,
    offsets) is computed from shapes and dtypes alone (_npy_header, _member_nbytes).
    """
    stamp = _dos_stamp(_DATE_TIME)
    members = []
    offset = 0
    for key, value in arrays.items():
        name = _member_name(key)
        header = _npy_header(value)
        local_size = _LOCAL.size + len(name) + _ZIP64_LOCAL_EXTRA.size
        size = len(header) + _member_nbytes(value)
        members.append({"name": name, "header": header, "value": value, "size": size,
                        "offset": offset, "local_size": local_size})
        offset += local_size + size
    if len(members) > _ZIP_FILECOUNT_LIMIT:
        raise ValueError("more archive members than this writer has been compared for")
    start_dir = offset

    count = workers if workers is not None else default_workers()
    pool = (ThreadPoolExecutor(max_workers=count, thread_name_prefix="gpuwm-npz-crc")
            if count > 1 else None)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    fd = os.open(os.fspath(path), flags, 0o666)
    try:
        member_crcs = []
        for member in members:
            # Materialise now; CRC its pieces on every core while it is written; drop it.
            header, data = _npy_parts(member.pop("value"))
            if header != member["header"] or len(header) + data.nbytes != member["size"]:
                raise ValueError(f"archive member {member['name']!r} changed between its "
                                 "layout and its write")
            pieces = [data[begin:begin + PIECE_BYTES] for begin in range(0, data.nbytes, PIECE_BYTES)]
            pending = ([pool.submit(zlib.crc32, piece) for piece in pieces] if pool is not None
                       else [zlib.crc32(piece) for piece in pieces])
            _write_all(fd, bytes(member["local_size"]) + header)
            for piece in pieces:
                _write_all(fd, piece)
            crc = zlib.crc32(header)
            remaining = data.nbytes
            for piece_crc in pending:
                length = min(PIECE_BYTES, remaining)
                crc = crc32_combine(crc, piece_crc if pool is None else piece_crc.result(), length)
                remaining -= length
            member_crcs.append(crc)
            del header, data, pieces, pending
        tail = _tail(members, member_crcs, start_dir, stamp)
        _write_at(fd, tail, start_dir)
        for member, crc in zip(members, member_crcs):
            _write_at(fd, _local_header(member["name"], crc, member["size"], stamp),
                      member["offset"])
    finally:
        os.close(fd)
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    return start_dir + len(tail)


def _tail(members, member_crcs, start_dir: int, stamp):
    """Central directory and end records, as ``ZipFile.close`` writes them."""
    directory = b"".join(_central_header(member["name"], crc, member["size"], member["offset"],
                                         stamp) for member, crc in zip(members, member_crcs))
    tail = [directory]
    count_entries, dir_size, dir_offset = len(members), len(directory), start_dir
    if (count_entries > _ZIP_FILECOUNT_LIMIT or dir_offset > _ZIP64_LIMIT
            or dir_size > _ZIP64_LIMIT):
        tail.append(_END64.pack(b"PK\006\006", 44, _ZIP64_VERSION, _ZIP64_VERSION, 0, 0,
                                count_entries, count_entries, dir_size, dir_offset))
        tail.append(_END64_LOCATOR.pack(b"PK\006\007", 0, start_dir + dir_size, 1))
        count_entries = min(count_entries, 0xFFFF)
        dir_size = min(dir_size, 0xFFFFFFFF)
        dir_offset = min(dir_offset, 0xFFFFFFFF)
    tail.append(_END.pack(b"PK\005\006", 0, 0, count_entries, count_entries, dir_size,
                          dir_offset, 0))
    return b"".join(tail)
