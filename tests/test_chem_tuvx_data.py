"""TUV-x data on demand (woof.chem_tuvx_data): pinned, verified before install, cached, never committed.

The pins must name exactly the files the oracle's configuration reads.  The staging contract is exercised on
synthetic files (no network): a local copy and a file:// mirror install verified bytes under TUV-x's relative
paths in the cache, a wrong byte is refused and nothing is installed, an explicit root is verified and never
written, and an unpinned name is refused.
"""
import hashlib
import json
from pathlib import Path

import pytest

from woof import chem_tuvx_data as data

ORACLE = Path(__file__).resolve().parents[1] / "tools" / "tuvx_oracle" / "photolysis.json"


def _paths(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _paths(value)
    elif isinstance(node, list):
        for value in node:
            yield from _paths(value)
    elif isinstance(node, str) and node.startswith("data/"):
        yield node


def test_the_pins_are_the_oracle_configurations_files():
    assert set(_paths(json.loads(ORACLE.read_text()))) == set(data.FILES)
    assert all(len(sha) == 64 and size > 0 for size, sha in data.FILES.values())
    assert data.TUVX_COMMIT in (Path(__file__).resolve().parents[1] / "tools/tuvx_oracle/build.sh").read_text()


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """Two pinned synthetic files laid out as TUV-x's tree, the cache under a temporary home."""
    payloads = {"data/grids/wavelength/w.grid": b"wavelength " * 300,
                "data/cross_sections/X_1.nc": bytes(range(256)) * 40}
    monkeypatch.setattr(data, "FILES", {k: (len(v), hashlib.sha256(v).hexdigest()) for k, v in payloads.items()})
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.delenv(data.ENV_ROOT, raising=False)
    source = tmp_path / "tuv-x"
    for relative, payload in payloads.items():
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        (source / relative).write_bytes(payload)
    return payloads, source


def test_a_local_copy_installs_verified_under_tuvxs_paths(synthetic):
    payloads, source = synthetic
    root = data.ensure(from_dir=source)
    assert root == data.cache_root() and str(root).startswith(str(Path.home()))
    for relative, payload in payloads.items():
        assert (root / relative).read_bytes() == payload
    assert data.ensure(from_dir=source) == root          # idempotent: verified, not refetched
    import shutil
    shutil.rmtree(source)
    assert data.ensure(from_dir=source) == root          # the cache serves without its source


def test_a_mirror_url_installs_verified_bytes(synthetic, monkeypatch):
    payloads, source = synthetic
    monkeypatch.setenv(data.ENV_URL_BASE, source.as_uri())
    root = data.ensure()
    for relative, payload in payloads.items():
        assert (root / relative).read_bytes() == payload


def test_a_wrong_byte_is_refused_and_not_installed(synthetic):
    payloads, source = synthetic
    relative = "data/cross_sections/X_1.nc"
    bad = bytearray(payloads[relative])
    bad[7] ^= 1
    (source / relative).write_bytes(bytes(bad))
    with pytest.raises(data.TuvxDataError, match="X_1.nc"):
        data.ensure([relative], from_dir=source)
    assert not (data.cache_root() / relative).exists()


def test_an_explicit_root_is_verified_and_never_written(synthetic, monkeypatch):
    payloads, source = synthetic
    monkeypatch.setenv(data.ENV_ROOT, str(source))
    assert data.ensure() == source.resolve()
    assert not data.cache_root().exists()
    (source / "data/grids/wavelength/w.grid").write_bytes(b"other")
    with pytest.raises(data.TuvxDataError, match="w.grid"):
        data.ensure()


def test_an_unpinned_file_is_refused(synthetic):
    with pytest.raises(data.TuvxDataError, match="not pinned"):
        data.ensure(["data/cross_sections/HNO3_1.nc"])
