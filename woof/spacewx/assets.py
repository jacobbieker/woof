"""Pinned coefficient files for the space-weather layers, fetched on demand.

Each asset has a fixed URL, byte count and SHA-256.  It is downloaded once into
the space-weather cache (``~/.woof/spacewx`` or ``$WOOF_SPACEWX_CACHE``), checked
before it is installed, and checked again whenever a cached copy is used.  A
mismatch is refused, never repaired silently.  Nothing here ships in a wheel.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from woof.spacewx.solar_irradiance import cache_root


class AssetError(ValueError):
    pass


@dataclass(frozen=True)
class PinnedAsset:
    name: str
    url: str
    bytes: int
    sha256: str
    subdir: str


def _check(path: Path, asset: PinnedAsset) -> None:
    data = path.read_bytes()
    if len(data) != asset.bytes or hashlib.sha256(data).hexdigest() != asset.sha256:
        raise AssetError(
            f"{path}: {len(data)} bytes, SHA-256 {hashlib.sha256(data).hexdigest()}; the pin is "
            f"{asset.bytes} bytes, {asset.sha256} ({asset.url}). Delete the file to fetch it again.")


def fetch_pinned(asset: PinnedAsset, *, cache=None, opener=None) -> Path:
    """The cached, verified copy of ``asset`` (downloaded the first time)."""
    root = Path(cache) if cache is not None else cache_root()
    target = root / asset.subdir / asset.name
    if target.is_file():
        _check(target, asset)
        return target
    import urllib.request
    opener = opener or (lambda url: urllib.request.urlopen(url, timeout=120))
    target.parent.mkdir(parents=True, exist_ok=True)
    part = unique_part(target)
    try:
        with opener(asset.url) as reply:
            part.write_bytes(reply.read())
        _check(part, asset)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(target)     # atomic: a concurrent fetch of the same pin installs equal bytes
    return target


def unique_part(target: Path) -> Path:
    """A download path beside ``target`` that no other process shares."""
    import os
    import uuid
    return target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part")
