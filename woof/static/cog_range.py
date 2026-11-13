"""Cloud-Optimized GeoTIFF block access over byte ranges, and a small
tiled GeoTIFF writer for the windows cut from them.

A COG keeps its whole first image file directory (IFD) -- dimensions,
block size, compression and the block offset/byte-count arrays -- at the
front of the file, so a reader can fetch that head once and then fetch
only the blocks under a window.  This module is the parsing and decoding
half of that; the network half (retries, locks, hashing, the pinned object
identity) is :mod:`woof.static.gem15`, built on the shared fetch machinery
in :mod:`woof.static.highres_fetch`.

The inventory is deliberately closed: little-endian classic TIFF or
BigTIFF, tiled, chunky (pixel-interleaved), 32-bit IEEE float samples,
no compression or Deflate, no predictor.  That is what the GEM-15 objects
are (BigTIFF, 512 x 512 Deflate blocks, predictor 1; the DSM interleaves
five bands); anything else is refused by name rather than misread.

The writer emits what the Rust ``static-fields`` reader and rasterio both
read: a little-endian classic TIFF, 512 x 512 Deflate tiles, one float32
band, GeoKeys naming the projected EPSG code, nodata ``nan``.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Bytes of a COG's head to read first; GEM-15's largest heads (16384 x
#: 8192 DSM cells with overview IFDs) end well inside this.
HEAD_BYTES = 65536

_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8,
              11: 4, 12: 8, 16: 8, 17: 8, 18: 8}
_TYPE_FMT = {1: "B", 3: "H", 4: "I", 8: "h", 9: "i", 11: "f", 12: "d",
             16: "Q", 17: "q"}


class CogFormatError(ValueError):
    """The bytes are not a COG this reader handles; names what is not."""


class CogHeadTooShort(CogFormatError):
    """The first IFD or one of its arrays lies past the bytes given."""

    def __init__(self, needed: int):
        super().__init__(f"the COG head needs the first {needed} bytes")
        self.needed = int(needed)


@dataclass(frozen=True)
class CogLayout:
    """Everything needed to locate and decode one block of image 0."""

    width: int
    height: int
    block_w: int
    block_h: int
    samples_per_pixel: int
    compression: int
    offsets: tuple[int, ...]
    byte_counts: tuple[int, ...]
    #: Affine ``(a, b, c, d, e, f)``: x = c + a*col, y = f + e*row at the
    #: pixel's upper-left corner (PixelIsArea).
    transform: tuple[float, float, float, float, float, float]
    epsg: int | None
    nodata: str | None

    @property
    def blocks_across(self) -> int:
        return -(-self.width // self.block_w)

    @property
    def blocks_down(self) -> int:
        return -(-self.height // self.block_h)

    def block(self, row: int, col: int) -> tuple[int, int]:
        """``(offset, byte_count)`` of block ``(row, col)``; count 0 is a
        sparse (all-nodata) block."""
        index = int(row) * self.blocks_across + int(col)
        return self.offsets[index], self.byte_counts[index]


def parse_head(head: bytes) -> CogLayout:
    """Parse image 0 of a COG from its first bytes.

    Raises :class:`CogHeadTooShort` naming how many bytes are needed when
    the IFD or an array it points to lies past ``head``.
    """
    if len(head) < 16:
        raise CogHeadTooShort(16)
    if head[:2] != b"II":
        raise CogFormatError("only little-endian TIFF is read (byte order "
                             f"{head[:2]!r})")
    magic = struct.unpack_from("<H", head, 2)[0]
    if magic == 43:
        big = True
        if struct.unpack_from("<HH", head, 4) != (8, 0):
            raise CogFormatError("BigTIFF offset size is not 8")
        ifd = struct.unpack_from("<Q", head, 8)[0]
        count_fmt, count_size, entry_size = "<Q", 8, 20
    elif magic == 42:
        big = False
        ifd = struct.unpack_from("<I", head, 4)[0]
        count_fmt, count_size, entry_size = "<H", 2, 12
    else:
        raise CogFormatError(f"TIFF magic {magic} is neither 42 nor 43")
    if ifd + count_size > len(head):
        raise CogHeadTooShort(ifd + count_size)
    entries = struct.unpack_from(count_fmt, head, ifd)[0]
    end = ifd + count_size + entries * entry_size
    if end > len(head):
        raise CogHeadTooShort(end)
    tags: dict[int, tuple] = {}
    for i in range(entries):
        at = ifd + count_size + i * entry_size
        if big:
            tag, kind, count = struct.unpack_from("<HHQ", head, at)
            inline, value_at = 8, at + 12
        else:
            tag, kind, count = struct.unpack_from("<HHI", head, at)
            inline, value_at = 4, at + 8
        size = _TYPE_SIZE.get(kind)
        if size is None:
            continue
        nbytes = size * count
        if nbytes > inline:
            value_at = struct.unpack_from("<Q" if big else "<I", head,
                                          value_at)[0]
        if value_at + nbytes > len(head):
            raise CogHeadTooShort(value_at + nbytes)
        raw = head[value_at:value_at + nbytes]
        if kind == 2:
            tags[tag] = (raw.split(b"\0", 1)[0].decode("ascii", "replace"),)
        elif kind in _TYPE_FMT:
            tags[tag] = struct.unpack(f"<{count}{_TYPE_FMT[kind]}", raw)

    def one(tag, default=None):
        value = tags.get(tag)
        if value is None:
            if default is None:
                raise CogFormatError(f"TIFF tag {tag} is missing")
            return default
        return value[0]

    if 322 not in tags or 324 not in tags:
        raise CogFormatError("the image is stripped, not tiled")
    spp = int(one(277, 1))
    bits = tags.get(258, (1,))
    if any(int(b) != 32 for b in bits) or any(
            int(f) != 3 for f in tags.get(339, (1,))):
        raise CogFormatError(f"samples are not 32-bit float (bits {bits}, "
                             f"format {tags.get(339)})")
    if spp > 1 and int(one(284, 1)) != 1:
        raise CogFormatError("planar (band-separate) layout is not read")
    compression = int(one(259, 1))
    if compression not in (1, 8, 32946):
        raise CogFormatError(f"compression {compression} is not read "
                             "(none or Deflate only)")
    if int(one(317, 1)) != 1:
        raise CogFormatError(f"predictor {one(317)} is not read (1 only)")
    scale = tags.get(33550)
    tie = tags.get(33922)
    if scale is None or tie is None or len(scale) < 2 or len(tie) < 6:
        raise CogFormatError("no ModelPixelScale/ModelTiepoint georeference")
    transform = (float(scale[0]), 0.0, float(tie[3] - tie[0] * scale[0]),
                 0.0, -float(scale[1]), float(tie[4] + tie[1] * scale[1]))
    epsg = None
    keys = tags.get(34735)
    if keys is not None and len(keys) >= 4:
        for k in range(int(keys[3])):
            key, location, _, value = keys[4 + 4 * k: 8 + 4 * k]
            if key == 1025 and location == 0 and value == 2:
                # PixelIsPoint: the tiepoint names a pixel centre.
                transform = (transform[0], 0.0,
                             transform[2] - 0.5 * transform[0], 0.0,
                             transform[4], transform[5] - 0.5 * transform[4])
            if key == 3072 and location == 0:
                epsg = int(value)
    offsets, counts = tags[324], tags.get(325)
    if counts is None or len(counts) != len(offsets):
        raise CogFormatError("block offsets and byte counts disagree")
    layout = CogLayout(
        width=int(one(256)), height=int(one(257)), block_w=int(one(322)),
        block_h=int(one(323)), samples_per_pixel=spp,
        compression=compression, offsets=tuple(int(v) for v in offsets),
        byte_counts=tuple(int(v) for v in counts), transform=transform,
        epsg=epsg, nodata=tags.get(42113, (None,))[0])
    if len(offsets) != layout.blocks_across * layout.blocks_down:
        raise CogFormatError(
            f"{len(offsets)} blocks for a {layout.blocks_down} x "
            f"{layout.blocks_across} block grid")
    return layout


def decode_block(layout: CogLayout, payload: bytes | None, band: int
                 ) -> np.ndarray:
    """One block's ``band`` as float32 ``(block_h, block_w)``; a sparse
    block (``payload`` None or empty) is all NaN."""
    shape = (layout.block_h, layout.block_w)
    if not payload:
        return np.full(shape, np.nan, dtype=np.float32)
    raw = payload if layout.compression == 1 else zlib.decompress(payload)
    spp = layout.samples_per_pixel
    expected = shape[0] * shape[1] * spp * 4
    if len(raw) != expected:
        raise CogFormatError(f"a block decoded to {len(raw)} bytes, not "
                             f"{expected}")
    values = np.frombuffer(raw, dtype="<f4").reshape(shape[0], shape[1], spp)
    return np.ascontiguousarray(values[:, :, int(band)], dtype=np.float32)


def write_tiled_geotiff(path: Path, blocks, *, rows: int, cols: int,
                        block: int, transform, epsg: int) -> None:
    """Write a ``rows`` x ``cols`` grid of ``block`` x ``block`` float32
    tiles (``blocks`` yields them row by row, each ``(block, block)``) as
    a Deflate-tiled GeoTIFF in projected ``epsg``.

    ``transform`` is the upper-left-corner affine of the whole grid.
    Tiles are compressed and written one at a time, so a window is never
    held whole.
    """
    count = rows * cols
    a, _, c, _, e, f = transform
    ascii_nodata = b"nan\0"
    geokeys = (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1,
               int(epsg))
    # (tag, type, values) -- values packed below; sorted by tag.
    entries = [
        (256, 4, (cols * block,)), (257, 4, (rows * block,)),
        (258, 3, (32,)), (259, 3, (8,)), (262, 3, (1,)), (277, 3, (1,)),
        (284, 3, (1,)), (317, 3, (1,)), (322, 3, (block,)),
        (323, 3, (block,)), (324, 4, None), (325, 4, None), (339, 3, (3,)),
        (33550, 12, (float(a), float(-e), 0.0)),
        (33922, 12, (0.0, 0.0, 0.0, float(c), float(f), 0.0)),
        (34735, 3, geokeys), (42113, 2, ascii_nodata),
    ]
    ifd_at = 8
    ifd_size = 2 + 12 * len(entries) + 4
    data_at = ifd_at + ifd_size

    def nbytes(kind, values, n):
        return _TYPE_SIZE[kind] * (len(values) if values is not None else n)

    # Lay out out-of-line tag values after the IFD.
    layout, cursor = {}, data_at
    for tag, kind, values in entries:
        size = nbytes(kind, values, count)
        if size > 4:
            layout[tag] = cursor
            cursor += size + (size & 1)
    tiles_at = cursor
    offsets, counts = [], []
    path = Path(path)
    with path.open("wb") as stream:
        stream.write(b"II*\0" + struct.pack("<I", ifd_at))
        stream.write(b"\0" * (tiles_at - 8))
        position = tiles_at
        written = 0
        for tile in blocks:
            tile = np.ascontiguousarray(tile, dtype="<f4")
            if tile.shape != (block, block):
                raise ValueError(f"tile shape {tile.shape} is not "
                                 f"{(block, block)}")
            payload = zlib.compress(tile.tobytes(), 6)
            stream.write(payload)
            offsets.append(position)
            counts.append(len(payload))
            position += len(payload)
            written += 1
        if written != count:
            raise ValueError(f"{written} tiles written, {count} declared")
        if position >= 2**32:
            raise ValueError("window exceeds a classic TIFF (4 GiB)")
        packed = {324: tuple(offsets), 325: tuple(counts)}
        stream.seek(ifd_at)
        stream.write(struct.pack("<H", len(entries)))
        for tag, kind, values in entries:
            values = packed.get(tag, values)
            if kind == 2:
                raw = values
                n = len(values)
            else:
                raw = struct.pack(f"<{len(values)}{_TYPE_FMT[kind]}", *values)
                n = len(values)
            if len(raw) > 4:
                stream.write(struct.pack("<HHII", tag, kind, n, layout[tag]))
            else:
                stream.write(struct.pack("<HHI", tag, kind, n)
                             + raw.ljust(4, b"\0"))
        stream.write(struct.pack("<I", 0))
        for tag, kind, values in entries:
            if tag not in layout:
                continue
            values = packed.get(tag, values)
            raw = values if kind == 2 else struct.pack(
                f"<{len(values)}{_TYPE_FMT[kind]}", *values)
            stream.seek(layout[tag])
            stream.write(raw)


__all__ = ["CogFormatError", "CogHeadTooShort", "CogLayout", "HEAD_BYTES",
           "decode_block", "parse_head", "write_tiled_geotiff"]
