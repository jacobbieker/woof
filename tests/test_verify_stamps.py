"""Verify once, reuse while unchanged: the digest-stamp contract.

What is pinned (:mod:`woof.verify_stamps`):

* a file proven by a full read is not read again while its (device, inode,
  size, mtime_ns, ctime_ns) is identical, in this process (``memo``) or a
  later one (``stamp``);
* a change in ANY of those fields, a missing, corrupt, foreign or
  unsettled stamp, is a full read;
* ``WOOF_VERIFY=full`` reads every byte at every verification and keeps no
  stamp; an unknown policy is refused by name;
* receipts say which happened, per file and in total;
* tampering after the preflight is still caught after the run whenever it
  moves the stat, and what a stat-preserving tamper evades -- and that
  ``full`` catches it -- is demonstrated rather than asserted in prose.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from woof import verify_stamps


@pytest.fixture
def stamps(tmp_path, monkeypatch):
    """A private stamp store, the default policy, and a clock 10 s ahead.

    Every file a test writes is brand new, so on the real clock it is
    never settled and never stamped; the clock ahead lets the test files
    count as settled without sleeping.
    """
    root = tmp_path / "stamp-store"
    monkeypatch.setenv(verify_stamps.STAMPS_ENV, str(root))
    monkeypatch.setenv(verify_stamps.POLICY_ENV, "stamp")
    monkeypatch.setattr(verify_stamps, "_clock_ns",
                        lambda: time.time_ns() + 10_000_000_000)
    verify_stamps.clear_memo()
    yield root
    verify_stamps.clear_memo()


def _counting_hasher(reads):
    def hasher(path):
        reads.append(Path(path).name)
        return verify_stamps.sha256_stream(path)
    return hasher


def _big(path: Path, fill: bytes = b"a", size: int = 2 << 20) -> Path:
    path.write_bytes(fill * size)
    return path


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def test_an_unchanged_file_is_read_once_then_vouched_for(tmp_path, stamps):
    path = _big(tmp_path / "f000.grib2")
    reads: list[str] = []
    first = verify_stamps.file_sha256(path, hasher=_counting_hasher(reads))
    assert first.how == verify_stamps.HASHED
    assert first.digest == hashlib.sha256(path.read_bytes()).hexdigest()

    again = verify_stamps.file_sha256(path, hasher=_counting_hasher(reads))
    assert (again.how, again.digest) == (verify_stamps.MEMO, first.digest)

    # A new process: no memo, the stamp on disk answers.
    verify_stamps.clear_memo()
    later = verify_stamps.file_sha256(path, hasher=_counting_hasher(reads))
    assert (later.how, later.digest) == (verify_stamps.STAMPED, first.digest)
    assert reads == [path.name]

    stamp = json.loads(next(stamps.rglob("*.json")).read_text())
    assert stamp["schema"] == verify_stamps.STAMP_SCHEMA
    assert stamp["path"] == os.path.abspath(path)
    assert stamp["algorithm"] == verify_stamps.FILE_SHA256
    assert stamp["digest"] == first.digest
    status = path.stat()
    assert (stamp["device"], stamp["inode"], stamp["size"],
            stamp["mtime_ns"], stamp["ctime_ns"]) == (
        status.st_dev, status.st_ino, status.st_size,
        status.st_mtime_ns, status.st_ctime_ns)
    assert stamp["verified_at"]


def _stamped(path: Path) -> str:
    digest = verify_stamps.file_sha256(path).digest
    verify_stamps.clear_memo()
    return digest


def _how(path: Path) -> verify_stamps.Verified:
    return verify_stamps.file_sha256(path)


def test_a_size_change_is_read_again(tmp_path, stamps):
    path = _big(tmp_path / "f")
    old = _stamped(path)
    with path.open("ab") as stream:
        stream.write(b"x")
    now = _how(path)
    assert now.how == verify_stamps.HASHED and now.digest != old


def test_a_same_size_rewrite_is_read_again(tmp_path, stamps):
    path = _big(tmp_path / "f")
    old = _stamped(path)
    _big(path, b"b")
    now = _how(path)
    assert now.how == verify_stamps.HASHED and now.digest != old


def test_a_rewound_mtime_does_not_hide_a_rewrite(tmp_path, stamps):
    """``touch -d`` can rewind mtime; it cannot rewind ctime."""
    path = _big(tmp_path / "f")
    before = path.stat()
    old = _stamped(path)
    _big(path, b"b")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    now = _how(path)
    assert now.how == verify_stamps.HASHED and now.digest != old


def test_a_ctime_only_change_is_read_again(tmp_path, stamps):
    path = _big(tmp_path / "f")
    _stamped(path)
    before = path.stat()
    os.chmod(path, 0o600)
    time.sleep(0.01)
    os.chmod(path, 0o644)
    after = path.stat()
    if after.st_ctime_ns == before.st_ctime_ns:
        pytest.skip("this filesystem's ctime did not move on chmod")
    assert (after.st_size, after.st_mtime_ns) == (
        before.st_size, before.st_mtime_ns)
    assert _how(path).how == verify_stamps.HASHED


def test_a_replaced_file_is_read_again(tmp_path, stamps):
    path = _big(tmp_path / "f")
    old = _stamped(path)
    before = path.stat()
    other = _big(tmp_path / "incoming")
    os.utime(other, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(other, path)
    assert path.stat().st_ino != before.st_ino
    now = _how(path)
    assert (now.how, now.digest) == (verify_stamps.HASHED, old)


@pytest.mark.parametrize("damage", [
    "garbage", "empty", "other-digest-format", "other-path", "other-algorithm",
    "unsettled", "boolean-size"])
def test_a_corrupt_or_foreign_stamp_is_a_full_read(tmp_path, stamps, damage):
    path = _big(tmp_path / "f")
    digest = _stamped(path)
    stamp_file = next(stamps.rglob("*.json"))
    document = json.loads(stamp_file.read_text())
    if damage == "garbage":
        stamp_file.write_bytes(b"\x00{not json")
    elif damage == "empty":
        stamp_file.write_bytes(b"")
    else:
        if damage == "other-digest-format":
            document["digest"] = "ZZ" * 32
        elif damage == "other-path":
            document["path"] = str(tmp_path / "elsewhere")
        elif damage == "other-algorithm":
            document["algorithm"] = "md5"
        elif damage == "unsettled":
            document["verified_at_ns"] = document["ctime_ns"] + 1
        elif damage == "boolean-size":
            document["size"] = True
        stamp_file.write_text(json.dumps(document))
    now = _how(path)
    assert (now.how, now.digest) == (verify_stamps.HASHED, digest)
    # ...and the full read replaced the damaged stamp with a good one.
    verify_stamps.clear_memo()
    assert _how(path).how == verify_stamps.STAMPED


def test_a_forged_digest_in_a_valid_stamp_is_what_full_exists_for(
        tmp_path, stamps, monkeypatch):
    """The store is trusted like the inputs are: whoever can write one
    can write the other.  ``full`` reads the bytes regardless."""
    path = _big(tmp_path / "f")
    real = _stamped(path)
    stamp_file = next(stamps.rglob("*.json"))
    document = json.loads(stamp_file.read_text())
    document["digest"] = "0" * 64
    stamp_file.write_text(json.dumps(document))
    assert _how(path).digest == "0" * 64
    monkeypatch.setenv(verify_stamps.POLICY_ENV, "full")
    assert _how(path).digest == real


def test_a_freshly_written_file_is_never_stamped(tmp_path, monkeypatch):
    root = tmp_path / "store"
    monkeypatch.setenv(verify_stamps.STAMPS_ENV, str(root))
    monkeypatch.setenv(verify_stamps.POLICY_ENV, "stamp")
    verify_stamps.clear_memo()
    path = _big(tmp_path / "f")
    reads: list[str] = []
    for _ in range(2):
        assert verify_stamps.file_sha256(
            path, hasher=_counting_hasher(reads)).how == verify_stamps.HASHED
    assert reads == [path.name, path.name]
    assert not root.exists() or not list(root.rglob("*.json"))


def test_small_files_are_always_read(tmp_path, stamps):
    path = tmp_path / "receipt.json"
    path.write_text("{}")
    _stamped(path)
    assert _how(path).how == verify_stamps.HASHED
    assert not list(stamps.rglob("*.json"))


def test_full_policy_reads_every_time_and_keeps_nothing(
        tmp_path, stamps, monkeypatch):
    path = _big(tmp_path / "f")
    _stamped(path)
    monkeypatch.setenv(verify_stamps.POLICY_ENV, "full")
    reads: list[str] = []
    for _ in range(3):
        assert verify_stamps.file_sha256(
            path, hasher=_counting_hasher(reads)).how == verify_stamps.HASHED
    assert reads == [path.name] * 3
    other = _big(tmp_path / "g")
    verify_stamps.file_sha256(other)
    assert len(list(stamps.rglob("*.json"))) == 1


def test_stamps_off_keeps_the_memo_only(tmp_path, stamps, monkeypatch):
    monkeypatch.setenv(verify_stamps.STAMPS_ENV, "off")
    path = _big(tmp_path / "f")
    assert _how(path).how == verify_stamps.HASHED
    assert _how(path).how == verify_stamps.MEMO
    verify_stamps.clear_memo()
    assert _how(path).how == verify_stamps.HASHED


def test_an_unknown_policy_is_refused_by_name(monkeypatch):
    monkeypatch.setenv(verify_stamps.POLICY_ENV, "fulll")
    with pytest.raises(ValueError, match="WOOF_VERIFY='fulll' is not a"):
        verify_stamps.policy()
    with pytest.raises(ValueError, match="--verify-inputs 'maybe'"):
        verify_stamps.set_policy("maybe")


def test_the_flag_sets_the_policy_for_children(monkeypatch):
    import argparse
    monkeypatch.setenv(verify_stamps.POLICY_ENV, "stamp")
    monkeypatch.delenv(verify_stamps.POLICY_ENV)
    parser = argparse.ArgumentParser()
    verify_stamps.add_argument(parser)
    assert (verify_stamps.apply(parser.parse_args([]))
            == verify_stamps.default_policy())
    assert verify_stamps.apply(
        parser.parse_args(["--verify-inputs", "full"])) == "full"
    assert os.environ[verify_stamps.POLICY_ENV] == "full"


def test_windows_defaults_to_full_because_its_ctime_is_creation_time(
        monkeypatch):
    monkeypatch.setenv(verify_stamps.POLICY_ENV, "stamp")
    monkeypatch.delenv(verify_stamps.POLICY_ENV)
    monkeypatch.setattr(verify_stamps.os, "name", "nt")
    assert verify_stamps.policy() == "full"
    monkeypatch.setenv(verify_stamps.POLICY_ENV, "stamp")
    assert verify_stamps.policy() == "stamp"


def test_a_bad_policy_is_refused_before_a_route_moves_a_byte(
        tmp_path, monkeypatch):
    from woof import fetch_routes

    monkeypatch.setenv(verify_stamps.POLICY_ENV, "ful")
    moved: list[str] = []

    def download(url, dest, *, magic, opener=None):
        moved.append(url)
        raise AssertionError("nothing may move")

    with pytest.raises(ValueError, match="WOOF_VERIFY='ful'"):
        fetch_routes.run_plan(_rap_plan(), out=tmp_path / "out",
                              downloader=download, progress=lambda *_: None)
    assert moved == [] and not (tmp_path / "out").exists()


def test_an_unwritable_store_costs_the_reuse_not_the_verification(
        tmp_path, stamps, monkeypatch, capsys):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    monkeypatch.setenv(verify_stamps.STAMPS_ENV, str(blocker / "store"))
    monkeypatch.setattr(verify_stamps, "_WARNED",
                        type(verify_stamps._WARNED)())
    path = _big(tmp_path / "f")
    assert _how(path).digest == hashlib.sha256(path.read_bytes()).hexdigest()
    _big(tmp_path / "g")
    _how(tmp_path / "g")
    assert capsys.readouterr().err.count("could not keep a digest stamp") == 1


def test_many_files_are_read_in_parallel_and_counted_once(tmp_path, stamps):
    paths = [_big(tmp_path / f"f{index}", bytes([65 + index]))
             for index in range(6)]
    with verify_stamps.collecting() as tally:
        first = verify_stamps.files_sha256(paths, workers=3)
    assert [item.digest for item in first] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    assert tally.counts[verify_stamps.HASHED] == 6
    verify_stamps.clear_memo()
    with verify_stamps.collecting() as tally:
        second = verify_stamps.files_sha256(paths, workers=3)
    assert {item.how for item in second} == {verify_stamps.STAMPED}
    assert tally.as_dict()["stamp_reused"] == 6
    assert tally.as_dict()["hashed"] == 0


def test_the_receipt_states_the_policy_and_the_counts(stamps, monkeypatch):
    tally = verify_stamps.Tally()
    tally.add(verify_stamps.HASHED, 10)
    tally.add(verify_stamps.STAMPED, 20)
    receipt = verify_stamps.receipt(tally)
    assert receipt["policy"] == "stamp"
    assert receipt["stamp_store"] == str(stamps)
    assert (receipt["hashed"], receipt["hashed_bytes"]) == (1, 10)
    assert (receipt["stamp_reused"], receipt["stamp_reused_bytes"]) == (1, 20)
    assert "ctime" in receipt["rule"] and "WOOF_VERIFY=full" in receipt["rule"]
    monkeypatch.setenv(verify_stamps.POLICY_ENV, "full")
    receipt = verify_stamps.receipt()
    assert receipt["policy"] == "full" and receipt["stamp_store"] is None


# ---------------------------------------------------------------------------
# woof fetch: a finished table-route folder (the ICON shape)
# ---------------------------------------------------------------------------

def _rap_plan():
    from woof import fetch_routes
    return fetch_routes.resolve_request(
        "rap", cycle=datetime(2026, 8, 16, 0), hours=1)


def _big_downloader():
    def download(url, dest, *, magic, opener=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        body = b"GRIB" + url.encode() * (1 + (2 << 20) // len(url)) + b"7777"
        dest.write_bytes(body)
        return {"name": dest.name, "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(), "url": url}
    return download


def test_a_finished_route_folder_is_read_once_and_then_vouched_for(
        tmp_path, stamps, monkeypatch):
    from woof import fetch_routes

    plan = _rap_plan()
    fetch_routes.run_plan(plan, out=tmp_path, downloader=_big_downloader(),
                          progress=lambda *_: None)
    reads: list[str] = []
    real = verify_stamps.sha256_stream
    monkeypatch.setattr(verify_stamps, "sha256_stream",
                        lambda path: reads.append(Path(path).name) or real(path))

    # Cold: every object is read, once, however often it is asked about.
    assert fetch_routes.request_cached(plan, tmp_path)
    cold = fetch_routes.run_plan(plan, out=tmp_path,
                                 downloader=_big_downloader(),
                                 progress=lambda *_: None)
    assert len(reads) == len(plan.objects)
    assert {entry["verified"] for entry in cold["files"]} <= {"hashed", "memo"}
    assert cold["verification"]["policy"] == "stamp"
    assert cold["verification"]["memo_reused"] == len(plan.objects)

    # Warm, as a new process: nothing is read at all.
    verify_stamps.clear_memo()
    reads.clear()
    assert fetch_routes.request_cached(plan, tmp_path)
    verify_stamps.clear_memo()
    warm = fetch_routes.run_plan(plan, out=tmp_path,
                                 downloader=_big_downloader(),
                                 progress=lambda *_: None)
    assert reads == []
    assert {entry["verified"] for entry in warm["files"]} == {"stamp"}
    assert warm["verification"]["stamp_reused"] == len(plan.objects)
    assert warm["verification"]["hashed"] == 0
    assert all(entry["reused"] for entry in warm["files"])
    on_disk = json.loads((tmp_path / fetch_routes.MANIFEST_NAME).read_text())
    assert on_disk["verification"]["stamp_reused"] == len(plan.objects)

    # A touched file is read again; a damaged one is moved, not reused.
    victim = tmp_path / plan.objects[0].relpath
    body = bytearray(victim.read_bytes())
    body[10] ^= 0xFF
    victim.write_bytes(bytes(body))
    verify_stamps.clear_memo()
    reads.clear()
    assert not fetch_routes.request_cached(plan, tmp_path)
    assert reads == [victim.name]


def test_a_stamp_that_disagrees_with_the_receipt_is_not_read_again(
        tmp_path, stamps, monkeypatch):
    from woof import fetch_routes

    plan = _rap_plan()
    fetch_routes.run_plan(plan, out=tmp_path, downloader=_big_downloader(),
                          progress=lambda *_: None)
    victim = tmp_path / plan.objects[0].relpath
    body = bytearray(victim.read_bytes())
    body[10] ^= 0xFF
    victim.write_bytes(bytes(body))
    verify_stamps.file_sha256(victim)        # stamps the damaged bytes
    verify_stamps.clear_memo()
    reads: list[str] = []
    real = verify_stamps.sha256_stream
    monkeypatch.setattr(verify_stamps, "sha256_stream",
                        lambda path: reads.append(Path(path).name) or real(path))
    assert not fetch_routes.request_cached(plan, tmp_path)
    assert victim.name not in reads


def test_full_policy_reads_a_finished_route_folder_every_time(
        tmp_path, stamps, monkeypatch):
    from woof import fetch_routes

    plan = _rap_plan()
    fetch_routes.run_plan(plan, out=tmp_path, downloader=_big_downloader(),
                          progress=lambda *_: None)
    fetch_routes.request_cached(plan, tmp_path)
    monkeypatch.setenv(verify_stamps.POLICY_ENV, "full")
    payload = fetch_routes.run_plan(plan, out=tmp_path,
                                    downloader=_big_downloader(),
                                    progress=lambda *_: None)
    assert {entry["verified"] for entry in payload["files"]} == {"hashed"}
    assert payload["verification"]["policy"] == "full"
    assert payload["verification"]["hashed"] == len(plan.objects)


# ---------------------------------------------------------------------------
# Prepared caches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("array", [
    np.arange(24, dtype=np.float32).reshape(2, 3, 4),
    np.arange(6, dtype=">i4").reshape(3, 2),
    np.zeros((0, 5), dtype=np.float64),
    np.asfortranarray(np.arange(12, dtype=np.int16).reshape(3, 4)),
    np.array(3.5, dtype=np.float64),
    np.arange(5, dtype=np.uint8),
])
def test_the_streamed_array_digest_is_the_manifest_digest(tmp_path, array):
    from woof.ingest import prepared_cache

    path = tmp_path / "a00000.npy"
    np.save(path, array, allow_pickle=False)
    loaded = np.load(path, allow_pickle=False)
    spec = {"file": path.name, "shape": list(loaded.shape),
            "dtype": str(loaded.dtype), "nbytes": int(loaded.nbytes),
            "sha256": prepared_cache._array_sha256(loaded)}
    streamed = prepared_cache._streamed_array_sha256(path, "k", spec)
    if streamed is None:
        assert array.ndim > 1 and np.isfortran(array)
    else:
        assert streamed == spec["sha256"]
    assert prepared_cache.verify_manifest_array(tmp_path, "k", spec) == "hashed"


def _cache(tmp_path: Path):
    from woof.ingest import prepared_cache

    directory = tmp_path / "cache"
    directory.mkdir()
    arrays = {}
    for index in range(4):
        value = np.full((64, 64, 32), index, dtype=np.float32)
        name = f"a{index:05d}.npy"
        np.save(directory / name, value, allow_pickle=False)
        arrays[f"k{index}"] = {
            "file": name, "shape": list(value.shape),
            "dtype": str(value.dtype), "nbytes": int(value.nbytes),
            "sha256": prepared_cache._array_sha256(value)}
    return directory, arrays


def test_prepared_arrays_are_read_once_then_vouched_for(tmp_path, stamps):
    from woof.ingest import prepared_cache

    directory, arrays = _cache(tmp_path)
    cold = prepared_cache.verify_manifest_arrays(directory, arrays, workers=2)
    assert (cold["hashed"], cold["stamp_reused"]) == (4, 0)
    verify_stamps.clear_memo()
    warm = prepared_cache.verify_manifest_arrays(directory, arrays, workers=2)
    assert (warm["hashed"], warm["stamp_reused"]) == (0, 4)

    # Tampered at its own size: ctime moves, the array is read and refused.
    target = directory / arrays["k2"]["file"]
    body = bytearray(target.read_bytes())
    body[-1] ^= 0x01
    target.write_bytes(bytes(body))
    with pytest.raises(prepared_cache.PreparedCacheCorruptError,
                       match="'k2' fails its manifest"):
        prepared_cache.verify_manifest_arrays(directory, arrays)


def test_a_stamp_for_another_row_vouches_for_nothing(tmp_path, stamps):
    from woof.ingest import prepared_cache

    directory, arrays = _cache(tmp_path)
    prepared_cache.verify_manifest_arrays(directory, arrays)
    verify_stamps.clear_memo()
    row = dict(arrays["k1"], sha256="1" * 64)
    with pytest.raises(prepared_cache.PreparedCacheCorruptError):
        prepared_cache.verify_manifest_array(directory, "k1", row)
    row = dict(arrays["k1"], shape=[32, 64, 64])
    with pytest.raises(prepared_cache.PreparedCacheCorruptError):
        prepared_cache.verify_manifest_array(directory, "k1", row)


def test_a_header_that_disagrees_with_its_row_says_so(tmp_path, stamps):
    """The streamed check's own finding is not relabelled 'unreadable'."""
    from woof.ingest import prepared_cache

    directory, arrays = _cache(tmp_path)
    row = dict(arrays["k0"], shape=[32, 64, 64])
    with pytest.raises(prepared_cache.PreparedCacheCorruptError,
                       match="'k0' fails its manifest"):
        prepared_cache.verify_manifest_array(directory, "k0", row)


def test_verify_all_returns_the_sealable_dict_and_reports_to_collectors(
        tmp_path, stamps, monkeypatch):
    """Callers seal verify_all()'s dict into relocatable receipts: it must
    not carry this machine's stamp store or policy."""
    from woof.ingest import prepared_cache

    directory, arrays = _cache(tmp_path)
    reader = object.__new__(prepared_cache.PreparedCacheReader)
    reader.path = directory
    reader.arrays = arrays
    reader.header = {"content_sha256": "c" * 64, "payload_bytes": sum(
        spec["nbytes"] for spec in arrays.values())}
    with verify_stamps.collecting() as tally:
        result = reader.verify_all()
    assert set(result) == {"schema", "status", "path", "content_sha256",
                           "array_count", "payload_bytes"}
    assert tally.counts[verify_stamps.HASHED] == len(arrays)
    assert reader.last_verification["hashed"] == len(arrays)


def test_a_truncated_array_is_unreadable(tmp_path, stamps):
    from woof.ingest import prepared_cache

    directory, arrays = _cache(tmp_path)
    target = directory / arrays["k0"]["file"]
    target.write_bytes(target.read_bytes()[:-100])
    with pytest.raises(prepared_cache.PreparedCacheCorruptError,
                       match="unreadable"):
        prepared_cache.verify_manifest_array(directory, "k0", arrays["k0"])


# ---------------------------------------------------------------------------
# The forecast: preflight, then a STAT check after the run
# ---------------------------------------------------------------------------

def _preflighted(tmp_path, monkeypatch):
    import test_prepared_single_domain_forecast as fixtures

    fixture = fixtures._prepared_fixture(tmp_path, "20crv3")
    fixtures._bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    return fixtures._preflight_fixture(fixture)


def test_the_post_run_check_reads_no_unchanged_array(
        tmp_path, stamps, monkeypatch):
    from woof import prepared_single_domain_forecast as runner

    inputs = _preflighted(tmp_path, monkeypatch)
    arrays = len(inputs.cache_reader.arrays)
    preflight = inputs.input_verification
    assert preflight["policy"] == "stamp"
    assert preflight["hashed"] >= arrays
    post = runner._verify_inputs_unchanged(inputs)
    assert post["policy"] == "stamp"
    assert post["memo_reused"] >= arrays
    # Only the receipt-sized authority files are read again.
    assert post["hashed"] == len(inputs.authority_paths)


def test_the_post_run_check_catches_a_rewritten_array(
        tmp_path, stamps, monkeypatch):
    from woof import prepared_single_domain_forecast as runner
    from woof.ingest.prepared_cache import PreparedCacheCorruptError

    inputs = _preflighted(tmp_path, monkeypatch)
    spec = inputs.cache_reader.arrays[sorted(inputs.cache_reader.arrays)[0]]
    target = inputs.prepared_cache_path / spec["file"]
    body = bytearray(target.read_bytes())
    body[-1] ^= 0x01
    target.write_bytes(bytes(body))
    with pytest.raises(PreparedCacheCorruptError, match="fails its manifest"):
        runner._verify_inputs_unchanged(inputs)


def test_a_stat_preserving_tamper_evades_stamp_and_full_catches_it(
        tmp_path, stamps, monkeypatch):
    """The documented limit, shown.  A rewrite that left every stat field
    as it was -- here simulated, since a test cannot rewind ctime -- is
    not seen by the stamp policy; ``WOOF_VERIFY=full`` reads the bytes."""
    from woof import prepared_single_domain_forecast as runner
    from woof.ingest.prepared_cache import PreparedCacheCorruptError

    inputs = _preflighted(tmp_path, monkeypatch)
    spec = inputs.cache_reader.arrays[sorted(inputs.cache_reader.arrays)[0]]
    target = inputs.prepared_cache_path / spec["file"]
    frozen = verify_stamps.Identity.of(target)
    body = bytearray(target.read_bytes())
    body[-1] ^= 0x01
    target.write_bytes(bytes(body))
    real_of = verify_stamps.Identity.of.__func__

    def preserved(cls, path):
        if os.path.realpath(path) == os.path.realpath(target):
            return frozen
        return real_of(cls, path)

    monkeypatch.setattr(verify_stamps.Identity, "of", classmethod(preserved))
    runner._verify_inputs_unchanged(inputs)          # evaded under stamp
    monkeypatch.setenv(verify_stamps.POLICY_ENV, "full")
    with pytest.raises(PreparedCacheCorruptError, match="fails its manifest"):
        runner._verify_inputs_unchanged(inputs)


def test_full_policy_rereads_every_array_after_the_run(
        tmp_path, stamps, monkeypatch):
    from woof import prepared_single_domain_forecast as runner

    inputs = _preflighted(tmp_path, monkeypatch)
    monkeypatch.setenv(verify_stamps.POLICY_ENV, "full")
    post = runner._verify_inputs_unchanged(inputs)
    assert post["policy"] == "full"
    assert post["memo_reused"] == post["stamp_reused"] == 0
    assert post["hashed"] == (len(inputs.cache_reader.arrays)
                              + len(inputs.authority_paths))
