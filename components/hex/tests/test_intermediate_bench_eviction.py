"""Cold receipts must observe the page cache, not only request eviction."""
import os
from pathlib import Path
import sys

import pytest

from bench_projected_intermediate import evict

#: Filesystems whose pages ARE the file: there is no backing store to evict to,
#: so POSIX_FADV_DONTNEED leaves every page resident and evict() refuses.
_MEMORY_FILESYSTEMS = {"tmpfs", "ramfs"}


def _filesystem(path: Path) -> str | None:
    """The type of the mount holding ``path`` (Linux /proc/self/mounts), or None."""

    try:
        lines = Path("/proc/self/mounts").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    path = path.resolve()
    best, kind = -1, None
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        mount = Path(fields[1].replace("\\040", " "))
        if (path == mount or mount in path.parents) and len(mount.parts) > best:
            best, kind = len(mount.parts), fields[2]
    return kind


@pytest.mark.skipif(sys.platform != "linux", reason="C3 component measurements run on Linux boxes")
def test_real_input_file_is_nonresident_after_eviction(tmp_path):
    kind = _filesystem(tmp_path)
    if kind in _MEMORY_FILESYSTEMS:
        # Not a pass: the bench itself refuses a COLD receipt here, correctly.
        # Run with --basetemp on a disk-backed filesystem to exercise it.
        pytest.skip(f"pytest's tmp_path is on {kind}, whose pages are the file and cannot be "
                    "evicted; a COLD measurement needs a disk-backed filesystem (--basetemp)")
    path = tmp_path / "input.grib2"
    with path.open("wb") as handle:
        handle.write(b"input-cache-residency" * 4096)
        handle.flush()
        os.fsync(handle.fileno())
    assert path.read_bytes().startswith(b"input-cache-residency")
    rows = evict([path])
    assert rows[0]["resident_pages"] == 0
    assert rows[0]["pages"] > 0
    assert rows[0]["bytes"] == path.stat().st_size
