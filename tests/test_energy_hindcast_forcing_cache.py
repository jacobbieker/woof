"""The hindcast forcing cache (``woof.energy.forcing_cache``).

One prefetch per (source, cycle) into a shared cache, forecast jobs that
depend on it, a mirror into the exact download folder ``woof go`` picks,
and the queue that claims a forecast job only after its prefetch.  The
"fetch" here is the real :func:`woof.fetch_routes.run_plan` with a
downloader that writes synthetic payloads (right magic, made-up bytes), so
every receipt is the one the real fetch writes and no test touches the
network: the reuse checks run with a downloader that fails if called.
CPU only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tomllib
from datetime import datetime
from pathlib import Path

import pytest

from woof import fetch_routes
from woof.energy import forcing_cache as fc
from woof.energy import hindcast as hc
from woof.energy import workqueue as wq
from woof.energy.contracts import load_plan, load_sites, sha256_file

REPO = Path(__file__).resolve().parents[1]
SITES = REPO / "tests" / "fixtures" / "energy" / "sites_wales.json"
NOW = datetime(2026, 10, 14, 12)
CYCLE = datetime(2025, 11, 1)


# --------------------------------------------------------------------------
# a network-free table-route fetch


def synthetic_payload(magic: str, url: str) -> bytes:
    body = hashlib.sha256(url.encode()).digest() * 32
    if magic == "HDF5":
        size = 64 + len(body)
        head = bytearray(64)
        head[0:8] = b"\x89HDF\r\n\x1a\n"
        head[13] = 8                                  # offsets are 8 bytes
        head[40:48] = size.to_bytes(8, "little")      # end-of-file address
        return bytes(head) + body
    if magic == "GRIB":
        return b"GRIB" + body + b"7777"
    return magic.encode("ascii") + body


def _plan_for(request, out):
    from woof.fetch import parse_cycle

    return fetch_routes.resolve_request(
        request["source"], cycle=parse_cycle(request["cycle"],
                                             request["source"]),
        hours=request["hours"], cadence=request.get("cadence"),
        start_hour=request.get("forecast_start_hour") or 0,
        host=request.get("transport"), member=request.get("member"),
        out=Path(out))


class FakeFetcher:
    """``fetcher(request, out)``: the real route fetch, synthetic bytes."""

    def __init__(self):
        self.calls = []

    def __call__(self, request, out):
        self.calls.append(dict(request))
        plan = _plan_for(request, out)

        def download(url, dest, *, magic, opener=None):
            payload = synthetic_payload(magic, url)
            dest = Path(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
            return {"name": dest.name, "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "url": url}

        keys = frozenset(o.key for o in plan.objects if o.key)
        fetch_routes.run_plan(plan, out=Path(out), downloader=download,
                              probe=lambda *a, **k: True,
                              lister=lambda *a, **k: keys,
                              progress=lambda *a, **k: None)
        fetch_routes.write_handoff(plan, Path(out))
        return {"objects": len(plan.objects)}


def _refuse_network(*args, **kwargs):
    raise AssertionError("the network was asked for an object the mirror "
                         "should have supplied")


# --------------------------------------------------------------------------
# what is shared


@pytest.mark.parametrize("source,mode", [
    ("met-office-global", "shared"), ("ecmwf-open-data", "shared"),
    ("hrrr-native", "shared"), ("rap", "shared"), ("gfs", "direct"),
    ("hrrr", "direct"), ("era5", "direct")])
def test_table_routes_are_shared_and_area_cropping_transports_are_not(
        source, mode):
    got, why = fc.shareability(source)
    assert got == mode and why


def test_cache_urls_are_normalized_and_refusals_are_named(tmp_path):
    assert fc.normalize_url(tmp_path / "c") == f"file://{tmp_path / 'c'}"
    assert fc.normalize_url("gcs://bkt/x/") == "gs://bkt/x"
    assert fc.normalize_url("s3://bkt/wf/cache") == "s3://bkt/wf/cache"
    for bad, match in (("ftp://h/x", "scheme"), ("s3://u:p@bkt/x",
                                                   "credentials"),
                       ("file://rel/x", "absolute"), ("s3:///x", "bucket")):
        with pytest.raises(fc.ForcingCacheRefused, match=match):
            fc.normalize_url(bad)


def test_the_predicted_request_is_the_planners_fetch_window():
    request = fc.predicted_request("ecmwf-open-data", CYCLE, 6.0)
    assert request == {"source": "ecmwf-open-data", "cycle": "2025-11-01T00",
                       "hours": 6, "cadence": 3, "forecast_start_hour": 0,
                       "member": None, "transport": None}
    assert fc.entry_key(request).startswith(
        "entries/ecmwf-open-data/20251101T00Z/")
    argv = fc.fetch_argv(request, "/x")
    assert argv[-2:] == ["--out", "/x"] and "--area" not in argv


# --------------------------------------------------------------------------
# plan


def _request(tmp_path, **overrides):
    record = dict(outdir=tmp_path / "hc", start=CYCLE,
                  end=datetime(2025, 11, 2), sites_path=SITES,
                  source="ecmwf-open-data", vram_gib=80.0,
                  forcing_cache=str(tmp_path / "cache"))
    record.update(overrides)
    return hc.HindcastRequest(**record)


def test_plan_emits_one_prefetch_per_source_and_cycle(tmp_path):
    ids = [s.site_id for s in load_sites(SITES).sites]
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({
        "schema": hc.PARTITION_SCHEMA,
        "sites": {"path": str(SITES), "sha256": sha256_file(SITES)},
        "clusters": [
            {"cluster_id": "west", "site_ids": ids[:10],
             "region": {"name": "west"}},
            {"cluster_id": "mid", "site_ids": ids[10:20],
             "region": {"name": "mid"}},
            {"cluster_id": "east", "site_ids": ids[20:],
             "region": {"name": "east"}}]}))

    def selector(region, cycle, hours, **_):
        return {"west": "ecmwf-open-data", "mid": "ecmwf-open-data",
                "east": "gfs"}[region["name"]]

    request = _request(tmp_path, sites_path=None, partition_path=partition,
                       source=None)
    result = hc.plan_hindcast(request, dry_run=True, now=NOW,
                              source_selector=selector)
    jobs, prefetch = result["jobs"], result["prefetch_jobs"]
    # 4 cycles x 3 clusters; west and mid share each cycle's prefetch.
    assert len(jobs) == 12 and len(prefetch) == 4
    by_id = {p["job_id"]: p for p in prefetch}
    for job in jobs:
        if job["cluster"]["cluster_id"] == "east":
            assert job["forcing"]["mode"] == "direct"
            assert "area" in job["forcing"]["reason"]
            assert "depends_on" not in job
            continue
        (dependency,) = job["depends_on"]
        assert job["forcing"]["prefetch_job_id"] == dependency
        prefetch_job = by_id[dependency]
        assert prefetch_job["segment"]["cycle"] == job["segment"]["cycle"]
        assert prefetch_job["forcing"]["entry"] == job["forcing"]["entry"]
    table = result["manifest"]["prefetch"]
    assert sorted(len(r["dependents"]) for r in table.values()) == [2] * 4
    assert any("8 forecast job(s)" in n and "gfs" in n
               for n in result["manifest"]["notes"])


def test_the_cache_does_not_change_job_identity_and_replanning_is_stable(
        tmp_path):
    plain = hc.plan_hindcast(_request(tmp_path, forcing_cache=None),
                             dry_run=True, now=NOW)
    cached = hc.plan_hindcast(_request(tmp_path), dry_run=True, now=NOW)
    assert [j["job_id"] for j in plain["jobs"]] == \
        [j["job_id"] for j in cached["jobs"]]
    assert "forcing" not in plain["jobs"][0]
    assert "prefetch" not in plain["manifest"]
    first = hc.plan_hindcast(_request(tmp_path), now=NOW)
    again = hc.plan_hindcast(_request(tmp_path), now=NOW)
    assert first["planned_prefetch_job_ids"] == \
        again["planned_prefetch_job_ids"]
    manifest = hc.read_manifest(tmp_path / "hc")
    assert manifest["prefetch_counts"] == {"pending": 4}
    assert manifest["counts"] == {"pending": 4}
    for pid in manifest["prefetch"]:
        job = hc.load_job(hc.locate_job(pid, hindcast_dir=tmp_path / "hc"))
        assert job["kind"] == "prefetch" and job["resources"]["gpus"] == 0
    status = hc.hindcast_status(tmp_path / "hc")
    assert status["prefetch"]["counts"] == {"pending": 4}
    with pytest.raises(hc.HindcastRefused, match="prefetch job"):
        hc.materialize_job(next(iter(manifest["prefetch"])),
                           hindcast_dir=tmp_path / "hc")


def test_adding_a_cache_to_a_planned_directory_binds_its_jobs(tmp_path):
    hc.plan_hindcast(_request(tmp_path, forcing_cache=None), now=NOW)
    manifest = hc.plan_hindcast(_request(tmp_path), resume=True, now=NOW)
    home = tmp_path / "hc"
    for job_id in manifest["planned_job_ids"]:
        job = hc.load_job(hc.locate_job(job_id, hindcast_dir=home))
        assert job["forcing"]["mode"] == "shared"
        assert job["depends_on"] == [job["forcing"]["prefetch_job_id"]]
    assert any("4 job(s) already on disk rebound" in n
               for n in manifest["notes"])
    # A running job keeps its document and is not counted as served.
    running = manifest["planned_job_ids"][0]
    hc.update_job(home, running, status="running")
    again = hc.plan_hindcast(_request(tmp_path, forcing_cache=str(
        tmp_path / "other")), resume=True, now=NOW)
    job = hc.load_job(hc.locate_job(running, hindcast_dir=home))
    assert job["forcing"]["cache_url"].endswith("/cache")
    assert running not in {d for pid, r in again["prefetch"].items()
                           if pid in again["planned_prefetch_job_ids"]
                           for d in r["dependents"]}
    assert any("keep their old forcing binding" in n for n in again["notes"])


# --------------------------------------------------------------------------
# prefetch


def test_prefetch_publishes_once_and_reruns_are_idempotent(tmp_path):
    request = fc.predicted_request("ecmwf-open-data", CYCLE, 6.0)
    cache = f"file://{tmp_path / 'cache'}"
    fetcher = FakeFetcher()
    receipt = fc.prefetch(request, cache, fetcher=fetcher,
                          workdir=tmp_path / "w", job_id="p1")
    assert receipt["status"] == "fetched" and receipt["objects"] == 3
    assert not list((tmp_path / "w").iterdir())       # scratch removed
    store = fc.open_cache(cache)
    entry = fc.read_entry(store, fc.entry_key(request))
    assert entry["schema"] == fc.CACHE_SCHEMA and entry["complete"]
    objects = [f for f in entry["files"] if f["kind"] == "object"]
    assert len(objects) == 3
    for item in entry["files"]:
        blob = tmp_path / "cache" / item["blob"]
        assert hashlib.sha256(blob.read_bytes()).hexdigest() == item["sha256"]
        assert ".fetch-verified" not in item["path"]
    assert objects[0]["fetch_entry"]["url"].startswith("https://")
    assert entry["fetch"]["argv"][:2] == ["woof", "fetch"]
    assert entry["fetch"]["route_table_sha256"] == \
        fetch_routes.packaged_route_table_sha256()
    again = fc.prefetch(request, cache, fetcher=fetcher)
    assert again["status"] == "cached" and len(fetcher.calls) == 1
    # A lost blob is fetched again; the others are not uploaded twice.
    (tmp_path / "cache" / objects[0]["blob"]).unlink()
    repaired = fc.prefetch(request, cache, fetcher=fetcher)
    assert repaired["status"] == "fetched"
    assert (tmp_path / "cache" / objects[0]["blob"]).is_file()
    assert repaired["reused_blobs"] >= 2      # the other objects stayed
    assert {f["path"] for f in entry["files"]} == {
        *(o["path"] for o in objects), "fetch-manifest.json", "SHA256SUMS"}
    assert len(fetcher.calls) == 2


def test_prefetch_refuses_an_unshared_source_and_a_wrong_folder(tmp_path):
    cache = f"file://{tmp_path / 'cache'}"
    gfs = fc.forcing_request("gfs", CYCLE, 6, cadence=3)
    with pytest.raises(fc.ForcingCacheRefused, match="not cached"):
        fc.prefetch(gfs, cache, fetcher=FakeFetcher())
    request = fc.predicted_request("ecmwf-open-data", CYCLE, 6.0)
    other = dict(request, cycle="2025-11-01T06")

    def wrong_cycle(_request, out):
        FakeFetcher()(other, out)

    with pytest.raises(fc.ForcingCacheRefused, match="records"):
        fc.prefetch(request, cache, fetcher=wrong_cycle)
    assert fc.read_entry(fc.open_cache(cache), fc.entry_key(request)) is None


class FakeS3Error(Exception):
    def __init__(self, code: str, status: int):
        super().__init__(code)
        self.response = {"Error": {"Code": code},
                         "ResponseMetadata": {"HTTPStatusCode": status}}


class FakeS3:
    """The S3 calls the cache makes: objects, head, whole-file transfer."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise FakeS3Error("NoSuchKey", 404)
        return {"Body": io.BytesIO(self.objects[Key]), "ETag": '"e"'}

    def put_object(self, Bucket, Key, Body, ContentType=None,
                   IfNoneMatch=None, IfMatch=None):
        if IfNoneMatch == "*" and Key in self.objects:
            raise FakeS3Error("PreconditionFailed", 412)
        self.objects[Key] = bytes(Body)
        return {"ETag": '"e"'}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise FakeS3Error("404", 404)
        return {"ContentLength": len(self.objects[Key])}

    def upload_file(self, Filename, Bucket, Key):
        self.objects[Key] = Path(Filename).read_bytes()

    def download_file(self, Bucket, Key, Filename):
        Path(Filename).write_bytes(self.objects[Key])


def test_an_s3_cache_round_trips_without_holding_files_in_memory(tmp_path):
    client = FakeS3()
    store = fc.CacheStore.open("s3://bkt/wf/cache", s3_client=client)
    request = fc.predicted_request("ecmwf-open-data", CYCLE, 6.0)
    assert fc.prefetch(request, store, fetcher=FakeFetcher())["status"] == \
        "fetched"
    key = fc.entry_key(request)
    assert f"wf/cache/{key}" in client.objects
    entry = fc.read_entry(store, key)
    item = entry["files"][0]
    assert store.size(item["blob"]) == item["bytes"]
    dest = tmp_path / "copy"
    assert store.get_file(item["blob"], dest, sha256=item["sha256"]) is None
    assert sha256_file(dest) == item["sha256"]
    client.objects[f"wf/cache/{item['blob']}"] = b"tampered"
    why = store.get_file(item["blob"], tmp_path / "bad",
                         sha256=item["sha256"])
    assert "hashes to" in why and not (tmp_path / "bad").exists()


def test_a_cloud_cache_without_its_sdk_is_a_named_refusal(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_boto(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("no boto3")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_boto)
    with pytest.raises(wq.QueueBackendUnavailable, match="boto3"):
        fc.CacheStore.open("s3://bkt/cache")


# --------------------------------------------------------------------------
# mirror: the folder woof go picks, reused without the network


@pytest.fixture(scope="module")
def planned(tmp_path_factory):
    """A one-day hindcast of the Wales sites with a forcing cache, its first
    cycle prefetched (fake transport) and its first job materialized by
    the real wrf-nests planner."""

    root = tmp_path_factory.mktemp("forcing")
    request = _request(root)
    manifest = hc.plan_hindcast(request, now=NOW)
    home = root / "hc"
    pid = manifest["planned_prefetch_job_ids"][0]
    fetcher = FakeFetcher()
    record = hc.run_prefetch(pid, hindcast_dir=home, fetcher=fetcher)
    assert record["ok"] and record["status"] == "fetched"
    job_id = hc.read_manifest(home)["prefetch"][pid]["dependents"][0]
    receipt = hc.materialize_job(job_id, hindcast_dir=home, now=NOW)
    return {"root": root, "home": home, "pid": pid, "job_id": job_id,
            "receipt": receipt, "fetcher": fetcher}


def _go_config(planned):
    job_dir = planned["home"] / "jobs" / planned["job_id"]
    plan = load_plan(job_dir / planned["receipt"]["plan"])
    ((config, case_root, _),) = fc._go_steps(plan)
    return plan, config, case_root


def test_the_mirror_fills_the_folder_woof_go_picks_and_fetch_reuses_it(
        planned):
    from woof.fetch import parse_cycle
    from woof.go_cli import (config_fetch_request, managed_download_dir,
                             pin_request, pinned_transport)

    home, job_id = planned["home"], planned["job_id"]
    receipt = hc.mirror_forcing(job_id, hindcast_dir=home)
    assert receipt["schema"] == fc.MIRROR_SCHEMA
    assert receipt["status"] == "mirrored" and receipt["fallback"] is False
    (config_record,) = receipt["configs"]
    assert config_record["mirrored"] == 3 and config_record["missing"] == []
    plan, config, case_root = _go_config(planned)
    payload = tomllib.loads(config.read_text())
    pinned, _ = pinned_transport(payload["fetch"], None)
    data_dir = managed_download_dir(
        case_root, pin_request(config_fetch_request(payload), pinned))
    # woof go chooses the mirrored folder itself: no "-1" sibling.
    assert str(data_dir) == config_record["data_dir"]
    request = fc.config_request(payload["fetch"])
    route_plan = _plan_for(request, data_dir)
    assert fetch_routes.request_cached(route_plan, data_dir)
    # The job's own fetch reuses every object and asks no host.
    manifest = fetch_routes.run_plan(route_plan, out=data_dir,
                                     downloader=_refuse_network,
                                     probe=_refuse_network,
                                     lister=_refuse_network,
                                     progress=lambda *a, **k: None)
    assert all(f["reused"] for f in manifest["files"])
    assert (data_dir / fetch_routes.SHA256SUMS_NAME).is_file()
    on_disk = json.loads((home / "jobs" / job_id / fc.MIRROR_NAME)
                         .read_text())
    assert on_disk["totals"]["mirrored"] == 3
    # Again: the folder is complete, nothing is copied.
    again = hc.mirror_forcing(job_id, hindcast_dir=home)
    assert again["configs"][0]["status"] == "already-local"
    assert len(planned["fetcher"].calls) == 1
    del parse_cycle


def test_the_mirror_falls_back_per_file_and_never_places_bad_bytes(
        planned, tmp_path):
    home, job_id = planned["home"], planned["job_id"]
    job_path = hc.locate_job(job_id, hindcast_dir=home)
    job = hc.load_job(job_path)
    store = fc.open_cache(job["forcing"]["cache_url"])
    entry = fc.read_entry(store, job["forcing"]["entry"])
    plan, config, case_root = _go_config(planned)
    # A fresh case root: copy the plan's config and mirror into it.
    objects = [f for f in entry["files"] if f["kind"] == "object"]
    lost, bad = objects[0], objects[1]
    cache_root = Path(store.url[len("file://"):])
    saved = {i["blob"]: (cache_root / i["blob"]).read_bytes()
             for i in (lost, bad)}
    try:
        (cache_root / lost["blob"]).unlink()
        (cache_root / bad["blob"]).write_bytes(b"GRIB" + b"x" * 10 + b"7777")
        record = fc._mirror_config(config, tmp_path / "case",
                                   forcing=job["forcing"], store=store,
                                   entry_cache={})
    finally:
        for blob, data in saved.items():
            (cache_root / blob).write_bytes(data)
    assert record["status"] == "partial" and record["fallback"] == \
        "direct-fetch"
    missing = {m["path"]: m["why"] for m in record["missing"]}
    assert "absent" in missing[lost["path"]]
    assert "bytes" in missing[bad["path"]] or "hashes" in missing[bad["path"]]
    data_dir = Path(record["data_dir"])
    assert not (data_dir / lost["path"]).exists()
    assert not (data_dir / bad["path"]).exists()
    assert record["mirrored"] == 1
    # Only the two missing objects would move.
    route_plan = _plan_for(fc.config_request(
        tomllib.loads(config.read_text())["fetch"]), data_dir)
    assert not fetch_routes.request_cached(route_plan, data_dir)


def test_a_pinned_host_still_takes_the_cached_objects(planned, tmp_path):
    home, job_id = planned["home"], planned["job_id"]
    job = hc.load_job(hc.locate_job(job_id, hindcast_dir=home))
    _, config, _ = _go_config(planned)
    pinned = tmp_path / "pinned.toml"
    text = config.read_text().replace("[fetch]\n",
                                      '[fetch]\ntransport = "aws"\n', 1)
    pinned.write_text(text)
    assert tomllib.loads(text)["fetch"]["transport"] == "aws"
    record = fc._mirror_config(pinned, tmp_path / "case",
                               forcing=job["forcing"],
                               store=fc.open_cache(job["forcing"]["cache_url"]),
                               entry_cache={})
    assert record["request"]["transport"] == "aws"
    assert record["status"] == "mirrored" and record["mirrored"] == 3
    data_dir = Path(record["data_dir"])
    route_plan = _plan_for(record["request"], data_dir)
    assert route_plan.pinned_host == "aws"
    assert fetch_routes.request_cached(route_plan, data_dir)


def test_an_expired_entry_is_fetched_again_inline(tmp_path):
    hc.plan_hindcast(_request(tmp_path), now=NOW)
    home = tmp_path / "hc"
    manifest = hc.read_manifest(home)
    pid = sorted(manifest["prefetch"])[0]
    fetcher = FakeFetcher()
    hc.run_prefetch(pid, hindcast_dir=home, fetcher=fetcher)
    job = hc.load_job(hc.locate_job(pid, hindcast_dir=home))
    # A lifecycle rule removes the entry; the manifest still says complete.
    (tmp_path / "cache" / job["forcing"]["entry"]).unlink()
    job_id = manifest["prefetch"][pid]["dependents"][0]
    receipt = hc.mirror_forcing(job_id, hindcast_dir=home, fetcher=fetcher,
                                now=NOW)
    assert len(fetcher.calls) == 2
    assert receipt["status"] == "mirrored", receipt
    assert any("ran inline (fetched)" in n for n in receipt["notes"])


def test_one_failing_prefetch_does_not_stop_the_rest(tmp_path):
    hc.plan_hindcast(_request(tmp_path), now=NOW)
    calls = []

    def flaky(request, out):
        calls.append(request["cycle"])
        if request["cycle"].endswith("T06"):
            raise KeyError("an SDK error that is not a woof refusal")
        return FakeFetcher()(request, out)

    result = hc.prefetch_pending(tmp_path / "hc", fetcher=flaky)
    assert len(calls) == 4 and result["failed"] == 1
    assert result["counts"] == {"fetched": 3, "failed": 1}
    status = hc.hindcast_status(tmp_path / "hc")
    assert status["prefetch"]["counts"] == {"complete": 3, "failed": 1}


def test_a_missing_entry_or_another_request_is_not_mirrored(planned,
                                                            tmp_path):
    home, job_id = planned["home"], planned["job_id"]
    job = hc.load_job(hc.locate_job(job_id, hindcast_dir=home))
    plan, config, _ = _go_config(planned)
    store = fc.open_cache(job["forcing"]["cache_url"])
    gone = dict(job["forcing"], entry="entries/x/20250101T00Z/0.json")
    record = fc._mirror_config(config, tmp_path / "a", forcing=gone,
                               store=store, entry_cache={})
    assert record["status"] == "entry-missing" and record["fallback"]
    other = dict(job["forcing"], request=dict(job["forcing"]["request"],
                                              cadence=1))
    record = fc._mirror_config(config, tmp_path / "b", forcing=other,
                               store=store, entry_cache={})
    assert record["status"] == "request-differs" and "cadence" in \
        record["reason"]


def test_run_local_mirrors_before_the_run_and_records_it(planned):
    home = planned["home"]
    manifest = hc.read_manifest(home)
    job_id = sorted(set(manifest["jobs"]) - {planned["job_id"]})[0]
    seen = {}

    def runner(plan_path, dry_run=False, resume=False):
        mirror = json.loads((Path(plan_path).parent.parent / fc.MIRROR_NAME)
                            .read_text())
        seen["mirror"] = mirror
        return {"domains": {d.domain_id: {"status": "complete"}
                            for d in load_plan(plan_path).domains}}

    fetcher = FakeFetcher()
    record = hc.run_local(job_id, hindcast_dir=home, runner=runner,
                          extract=False, fetcher=fetcher, now=NOW)
    assert record["ok"] and record["forcing"]["status"] == "mirrored"
    # Its cycle's entry was not in the cache: the prefetch ran inline once.
    assert len(fetcher.calls) == 1
    assert any("ran inline" in n for n in seen["mirror"]["notes"])
    after = hc.read_manifest(home)
    assert after["jobs"][job_id]["forcing"]["status"] == "mirrored"
    pid = hc.load_job(hc.locate_job(job_id, hindcast_dir=home))[
        "forcing"]["prefetch_job_id"]
    assert after["prefetch"][pid]["status"] == "complete"


def test_a_prefetch_job_carried_away_from_its_directory_runs(planned,
                                                             tmp_path):
    source = hc.locate_job(planned["pid"], hindcast_dir=planned["home"])
    carried = tmp_path / "work" / "job.json"
    carried.parent.mkdir()
    carried.write_bytes(source.read_bytes())
    record = hc.run_local(carried, fetcher=FakeFetcher())
    assert record["kind"] == "prefetch" and record["status"] == "cached"
    assert (carried.parent / fc.PREFETCH_RECEIPT_NAME).is_file()
    dry = hc.run_local(carried, dry_run=True)
    assert dry["dry_run"] and dry["argv"][:2] == ["woof", "fetch"]
    assert dry["download_estimate"]["objects"] == 3


def test_a_declared_donor_folder_is_cached_and_mirrored_whole(tmp_path):
    request = fc.predicted_request("met-office-global", datetime(2026, 6, 1),
                                   6.0)
    plan = _plan_for(request, tmp_path / "x")
    assert [d.source for d in plan.donors] == ["gdas"]

    def with_donor(req, out):
        FakeFetcher()(req, out)
        donor = Path(out) / "donor-gdas"
        donor.mkdir()
        (donor / "gdas.t00z.pgrb2.0p25.f000").write_bytes(
            synthetic_payload("GRIB", "gdas"))
        (donor / "fetch-manifest.json").write_text('{"area": null}\n')

    cache = fc.open_cache(f"file://{tmp_path / 'cache'}")
    fc.prefetch(request, cache, fetcher=with_donor)
    entry = fc.read_entry(cache, fc.entry_key(request))
    assert {f["path"] for f in entry["files"] if f["kind"] == "donor"} == {
        "donor-gdas/gdas.t00z.pgrb2.0p25.f000",
        "donor-gdas/fetch-manifest.json"}
    config = tmp_path / "seg.toml"
    config.write_text(
        '[fetch]\nsource = "met-office-global"\ncycle = "2026-06-01T00"\n'
        'hours = 6\ncadence = 3\n')
    forcing = {"mode": "shared", "entry": fc.entry_key(request),
               "request": request}
    record = fc._mirror_config(config, tmp_path / "case", forcing=forcing,
                               store=cache, entry_cache={})
    assert record["status"] == "mirrored", record
    assert record["donors"] == [{"source": "gdas", "files": 2,
                                 "mirrored": 2}]
    data_dir = Path(record["data_dir"])
    assert (data_dir / "donor-gdas" / "gdas.t00z.pgrb2.0p25.f000").is_file()
    assert fetch_routes.request_cached(_plan_for(request, data_dir),
                                       data_dir)


def test_a_partition_cluster_planner_is_not_handed_a_second_sites_file(
        tmp_path, monkeypatch):
    import sys
    import types

    seen = {}

    def plan_cluster(partition, cluster_id, *, outdir, **kwargs):
        seen.update(kwargs)
        raise hc.HindcastRefused("stop here")

    module = types.ModuleType("woof.energy.partition")
    module.plan_cluster = plan_cluster
    monkeypatch.setitem(sys.modules, "woof.energy.partition", module)
    ids = [s.site_id for s in load_sites(SITES).sites]
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({
        "schema": hc.PARTITION_SCHEMA, "sites": {"path": str(SITES)},
        "clusters": [{"cluster_id": "all", "site_ids": ids}]}))
    manifest = hc.plan_hindcast(_request(tmp_path, sites_path=None,
                                         partition_path=partition), now=NOW)
    with pytest.raises(hc.HindcastRefused, match="stop here"):
        hc.materialize_job(manifest["planned_job_ids"][0],
                           hindcast_dir=tmp_path / "hc", now=NOW)
    assert "sites_path" not in seen and seen["start"] == "2025-11-01T00"


# --------------------------------------------------------------------------
# queue: a forecast job is claimed only after its prefetch


OWNER = {"worker_id": "a", "host": "node-a", "pid": 1, "gpu_uuid": "GPU-a"}


def _queue(tmp_path):
    queue = wq.WorkQueue(wq.open_store(f"file://{tmp_path / 'queue'}"))
    queue.init(wq.QueueSettings(max_attempts=1, lease_s=60.0,
                                heartbeat_s=10.0, backoff_base_s=0.0,
                                backoff_max_s=0.0))
    return queue


def _doc(job_id, **extra):
    return {"schema": wq.JOB_SCHEMA, "job_id": job_id, **extra}


def test_forecast_jobs_wait_for_their_prefetch(tmp_path):
    queue = _queue(tmp_path)
    queue.push("prefetch-src-20251101T00Z-a", _doc(
        "prefetch-src-20251101T00Z-a", kind="prefetch"))
    for name in ("west", "east"):
        queue.push(f"{name}-x", _doc(
            f"{name}-x", depends_on=["prefetch-src-20251101T00Z-a"]))
    queue.push("free", _doc("free"))
    first = queue.claim(OWNER)
    assert first.job_id == "prefetch-src-20251101T00Z-a"
    second = queue.claim(OWNER)
    assert second.job_id == "free"        # the forecasts are blocked
    assert queue.claim(OWNER) is None
    queue.complete(first, {"ok": True})
    claimed = {queue.claim(OWNER).job_id, queue.claim(OWNER).job_id}
    assert claimed == {"west-x", "east-x"}


def test_a_dead_or_unqueued_dependency_does_not_hold_jobs_hostage(tmp_path):
    queue = _queue(tmp_path)
    queue.push("prefetch-a", _doc("prefetch-a", kind="prefetch"))
    queue.push("job-a", _doc("job-a", depends_on=["prefetch-a"]))
    queue.push("job-b", _doc("job-b", depends_on=["prefetch-never-pushed"]))
    first = queue.claim(OWNER)
    assert first.job_id == "prefetch-a"
    assert queue.claim(OWNER).job_id == "job-b"
    assert queue.claim(OWNER) is None
    queue.fail(first, exit_code=2, reason="refused", error_tail="",
               log_key=None, permanent=True)
    assert queue.claim(OWNER).job_id == "job-a"


def test_push_validates_dependencies_and_pushes_prefetch_first(tmp_path,
                                                               capsys):
    with pytest.raises(wq.JobSpecError, match="depends_on"):
        wq.job_from_document(_doc("a", depends_on="prefetch-x"))
    with pytest.raises(wq.JobSpecError, match="itself"):
        wq.job_from_document(_doc("a", depends_on=["a"]))
    queue = _queue(tmp_path)
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "a.json").write_text(json.dumps(_doc("a", depends_on=["zz"])))
    (jobs / "zz.json").write_text(json.dumps(_doc("zz", kind="prefetch")))
    order = []
    original = wq.WorkQueue.push

    def spy(self, job_id, document):
        order.append(job_id)
        return original(self, job_id, document)

    args = argparse.Namespace(queue=queue.url, jobs=[str(jobs)],
                              schema=wq.JOB_SCHEMA, queue_command="push")
    wq.WorkQueue.push = spy
    try:
        assert wq.main(args) == 0
    finally:
        wq.WorkQueue.push = original
    assert order == ["zz", "a"]
    capsys.readouterr()


# --------------------------------------------------------------------------
# cost: per job vs per cycle


def test_the_estimate_shows_the_per_cycle_saving(capsys):
    from woof.energy.cli import register_cli

    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers(dest="command"))

    def run(*extra):
        args = parser.parse_args(
            ["energy", "estimate", "--clusters", "440", "--footprint-km2",
             "20000", "--segments", "1460", "--segment-hours", "6",
             "--spinup-hours", "0", "--instance", "aws:p5.48xlarge",
             *extra])
        assert args.func(args) == 0
        return json.loads(capsys.readouterr().out)

    doc = run()
    forcing = doc["forcing"]
    assert forcing["shared"] and forcing["jobs"] == 440 * 1460
    per_job = forcing["per_job_fetch"]["bytes_from_publishers"]
    per_cycle = forcing["per_cycle_cache"]["bytes_from_publishers"]
    assert per_job == 440 * per_cycle
    assert forcing["saving"]["slot_hours"] > 0
    assert forcing["assumptions"]["publisher_mbps"] == 100.0
    priced = doc["offers"][0]["forcing_usd"]["on_demand"]
    assert priced["saving"] > 0
    assert priced["per_job_fetch"] > priced["per_cycle_cache"]
    gfs = run("--forcing-source", "gfs")["forcing"]
    assert not gfs["shared"] and gfs["saving"]["bytes_from_publishers"] == 0
    given = run("--forcing-gb-per-cycle", "2", "--forcing-objects-per-cycle",
                "10")["forcing"]
    assert given["per_cycle_download"]["bytes"] == 2_000_000_000
    assert given["per_cycle_cache"]["cache_put_requests"] == 1460 * 10
    assert "forcing" not in run("--no-forcing")
    # A default section that cannot be sized is a note, not a refusal.
    args = parser.parse_args(
        ["energy", "estimate", "--clusters", "2", "--footprint-km2", "20",
         "--segments", "3", "--segment-hours", "200", "--instance",
         "aws:p5.48xlarge"])
    assert args.func(args) == 0
    doc = json.loads(capsys.readouterr().out)
    assert "forcing" not in doc
    assert any("forcing section left out" in n for n in doc["notes"])
