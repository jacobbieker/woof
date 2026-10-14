"""The hindcast work queue and its GPU worker (``woof energy queue|worker``).

Every backend is exercised through the same queue operations: the
``file://`` store on a temp directory, and ``s3://`` / ``gs://`` stores
over in-memory fake clients that implement the conditional-write
semantics the real services document (``If-None-Match: *`` /
``If-Match`` ETags; ``ifGenerationMatch``).  No network, no GPU: worker
tests use ``--no-gpu`` or stand-in GPU identities under a temporary
``WOOF_GPU_LOCK_ROOT``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from woof.energy import workqueue as wq
from woof.energy import worker as wk

REPO = Path(__file__).resolve().parents[1]
JOB_SCHEMA = wq.JOB_SCHEMA


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeS3Error(Exception):
    def __init__(self, code: str, status: int):
        super().__init__(code)
        self.response = {"Error": {"Code": code},
                         "ResponseMetadata": {"HTTPStatusCode": status}}


class FakeS3:
    """S3 conditional writes as documented: 412 on a lost condition."""

    def __init__(self, page_size: int = 2):
        self.objects: dict[tuple[str, str], tuple[bytes, str]] = {}
        self.lock = threading.Lock()
        self.page_size = page_size
        self.calls: list[tuple[str, dict]] = []

    def get_object(self, Bucket, Key):
        with self.lock:
            found = self.objects.get((Bucket, Key))
        if found is None:
            raise FakeS3Error("NoSuchKey", 404)
        return {"Body": io.BytesIO(found[0]), "ETag": found[1]}

    def put_object(self, Bucket, Key, Body, ContentType=None,
                   IfNoneMatch=None, IfMatch=None):
        self.calls.append(("put_object", {"Key": Key,
                                          "IfNoneMatch": IfNoneMatch,
                                          "IfMatch": IfMatch}))
        with self.lock:
            current = self.objects.get((Bucket, Key))
            if IfNoneMatch == "*" and current is not None:
                raise FakeS3Error("PreconditionFailed", 412)
            if IfMatch is not None:
                if current is None:
                    raise FakeS3Error("NoSuchKey", 404)
                if current[1] != IfMatch:
                    raise FakeS3Error("PreconditionFailed", 412)
            etag = '"' + hashlib.md5(Body).hexdigest() + '"'
            self.objects[(Bucket, Key)] = (bytes(Body), etag)
        return {"ETag": etag}

    def delete_object(self, Bucket, Key):
        with self.lock:
            self.objects.pop((Bucket, Key), None)
        return {}

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        with self.lock:
            keys = sorted(k for b, k in self.objects
                          if b == Bucket and k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + self.page_size]
        more = start + self.page_size < len(keys)
        response = {"Contents": [{"Key": k} for k in page],
                    "IsTruncated": more}
        if more:
            response["NextContinuationToken"] = str(start + self.page_size)
        return response


class FakeGCSError(Exception):
    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


class FakeBlob:
    def __init__(self, store: "FakeGCS", name: str,
                 generation: int | None = None):
        self.store = store
        self.name = name
        self.generation = generation

    def download_as_bytes(self, if_generation_match=None):
        with self.store.lock:
            found = self.store.objects.get(self.name)
        if found is None:
            raise FakeGCSError(404)
        if if_generation_match is not None and found[1] != if_generation_match:
            raise FakeGCSError(412)
        return found[0]

    def upload_from_string(self, data, content_type=None,
                           if_generation_match=None):
        with self.store.lock:
            current = self.store.objects.get(self.name)
            if if_generation_match is not None:
                generation = 0 if current is None else current[1]
                if generation != if_generation_match:
                    raise FakeGCSError(412)
            self.store.counter += 1
            self.store.objects[self.name] = (bytes(data), self.store.counter)
            self.generation = self.store.counter

    def delete(self):
        with self.store.lock:
            if self.store.objects.pop(self.name, None) is None:
                raise FakeGCSError(404)


class FakeGCS:
    def __init__(self):
        self.objects: dict[str, tuple[bytes, int]] = {}
        self.lock = threading.Lock()
        self.counter = 0

    def bucket(self, name):
        store = self

        class Bucket:
            def get_blob(self, blob_name):
                with store.lock:
                    found = store.objects.get(blob_name)
                return None if found is None else FakeBlob(
                    store, blob_name, found[1])

            def blob(self, blob_name):
                return FakeBlob(store, blob_name)
        return Bucket()

    def list_blobs(self, bucket_name, prefix):
        with self.lock:
            return [FakeBlob(self, name, gen) for name, (_, gen)
                    in sorted(self.objects.items()) if name.startswith(prefix)]


class Clock:
    def __init__(self, now: float = 1_800_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(params=["file", "s3", "gs"])
def store(request, tmp_path):
    if request.param == "file":
        return wq.open_store(f"file://{tmp_path / 'queue'}")
    if request.param == "s3":
        return wq.open_store("s3://bucket/hindcast/2024", s3_client=FakeS3())
    return wq.open_store("gs://bucket/hindcast", gcs_client=FakeGCS())


def _job(job_id: str, **extra) -> dict:
    return {"schema": JOB_SCHEMA, "job_id": job_id, **extra}


def _queue(store, clock=None, **settings) -> wq.WorkQueue:
    queue = wq.WorkQueue(store, clock=clock or Clock())
    defaults = dict(max_attempts=3, lease_s=60.0, heartbeat_s=10.0,
                    backoff_base_s=5.0, backoff_max_s=40.0)
    defaults.update(settings)
    queue.init(wq.QueueSettings(**defaults))
    return queue


OWNER_A = {"worker_id": "a", "host": "node-a", "pid": 1, "gpu_uuid": "GPU-a"}
OWNER_B = {"worker_id": "b", "host": "node-b", "pid": 2, "gpu_uuid": "GPU-b"}


# --------------------------------------------------------------------------
# Stores
# --------------------------------------------------------------------------


def test_every_store_refuses_a_second_create_and_a_stale_swap(store):
    version = store.put_if_absent("leases/x.json", b'{"a": 1}')
    with pytest.raises(wq.PreconditionFailed):
        store.put_if_absent("leases/x.json", b'{"a": 2}')
    newer = store.put_if_match("leases/x.json", b'{"a": 3}', version)
    with pytest.raises(wq.PreconditionFailed):
        store.put_if_match("leases/x.json", b'{"a": 4}', version)
    assert store.get("leases/x.json") == (b'{"a": 3}', newer)
    store.delete("leases/x.json")
    assert store.get("leases/x.json") is None
    with pytest.raises(wq.PreconditionFailed):
        store.put_if_match("leases/x.json", b"{}", newer)


def test_every_store_lists_a_prefix_only(store):
    for name in ("jobs/a.json", "jobs/b.json", "jobs/c.json", "done/a.json",
                 "logs/a/attempt-001.log"):
        store.put(name, b"{}")
    assert store.list("jobs/") == ["jobs/a.json", "jobs/b.json", "jobs/c.json"]
    assert store.list("logs/") == ["logs/a/attempt-001.log"]
    assert store.list("dead/") == []


def test_store_keys_are_held_to_a_safe_alphabet(tmp_path):
    store = wq.open_store(f"file://{tmp_path}")
    with pytest.raises(wq.QueueError):
        store.put("../escape.json", b"{}")


def test_s3_store_sends_the_documented_conditions():
    client = FakeS3()
    store = wq.open_store("s3://bucket/p", s3_client=client)
    version = store.put_if_absent("leases/j.json", b"1")
    store.put_if_match("leases/j.json", b"2", version)
    conditions = [(c["Key"], c["IfNoneMatch"], c["IfMatch"])
                  for _, c in client.calls]
    assert conditions == [("p/leases/j.json", "*", None),
                          ("p/leases/j.json", None, version)]


def test_queue_urls_name_a_backend_or_are_refused(tmp_path):
    for url in ("/abs/path", "file://relative", "http://x/y", "s3:///nokey"):
        with pytest.raises(wq.QueueURLError):
            wq.open_store(url)


@pytest.mark.parametrize("url,module", [("s3://b/p", "boto3"),
                                        ("gs://b/p", "google.cloud")])
def test_a_missing_cloud_sdk_is_a_named_refusal(monkeypatch, url, module):
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(wq.QueueBackendUnavailable, match="install"):
        wq.open_store(url)


# --------------------------------------------------------------------------
# Queue semantics
# --------------------------------------------------------------------------


def test_init_is_idempotent_and_refuses_changed_settings(store):
    queue = _queue(store)
    assert queue.init(queue.settings())["created"] is False
    with pytest.raises(wq.QueueConfigConflict):
        queue.init(wq.QueueSettings(max_attempts=9, lease_s=60.0,
                                    heartbeat_s=10.0))


def test_an_uninitialised_queue_is_refused(store):
    with pytest.raises(wq.QueueNotInitialized):
        wq.WorkQueue(store).push("a", _job("a"))


def test_settings_keep_the_heartbeat_inside_the_lease():
    with pytest.raises(wq.QueueSettingsError):
        wq.QueueSettings(lease_s=60.0, heartbeat_s=40.0).validate()
    settings = wq.QueueSettings(backoff_base_s=10.0, backoff_max_s=35.0)
    assert [settings.backoff_s(n) for n in (1, 2, 3, 4)] == [10, 20, 35, 35]


def test_push_is_idempotent_and_refuses_a_changed_job(store):
    queue = _queue(store)
    assert queue.push("a", _job("a", month=1)) == "queued"
    assert queue.push("a", _job("a", month=1)) == "already-queued"
    with pytest.raises(wq.JobConflict):
        queue.push("a", _job("a", month=2))


def test_job_documents_are_validated(tmp_path):
    with pytest.raises(wq.JobSpecError, match="schema"):
        wq.job_from_document({"schema": "other", "job_id": "a"})
    with pytest.raises(wq.JobSpecError, match="job_id"):
        wq.job_from_document(_job("../a"))
    (tmp_path / "one.json").write_text(json.dumps(_job("same")))
    (tmp_path / "two.json").write_text(json.dumps(_job("same")))
    with pytest.raises(wq.JobSpecError, match="both"):
        wq.read_job_files([tmp_path])


def test_one_claim_per_job_and_the_lease_is_renewed(store):
    clock = Clock()
    queue = _queue(store, clock)
    queue.push("a", _job("a"))
    claim = queue.claim(OWNER_A)
    assert claim.job_id == "a" and claim.attempt == 1 and claim.job["job_id"] == "a"
    assert queue.claim(OWNER_B) is None
    clock.advance(50)
    queue.renew(claim)
    clock.advance(50)  # 100 s after the claim, 50 s after renewal
    assert queue.claim(OWNER_B) is None
    status = queue.status()
    assert status["counts"]["running"] == 1
    assert status["active_workers"][0]["worker_id"] == "a"


def test_an_expired_lease_is_reclaimed_charged_and_completed_once(store):
    clock = Clock()
    queue = _queue(store, clock)
    queue.push("a", _job("a"))
    first = queue.claim(OWNER_A)
    clock.advance(61)
    assert queue.status()["counts"]["expired"] == 1
    second = queue.claim(OWNER_B)
    assert second.attempt == 2 and second.failures == 1
    with pytest.raises(wq.LeaseLost):
        queue.renew(first)
    # Both copies finish: exactly one completion marker.
    assert queue.complete(second, {"exit_code": 0}) is True
    assert queue.complete(first, {"exit_code": 0}) is False
    done = json.loads(store.get("done/a.json")[0])
    assert done["schema"] == wq.DONE_SCHEMA and done["owner"]["worker_id"] == "b"
    assert [k for k in store.list("logs/") if "duplicate" in k]
    failed = json.loads(store.get("failed/a.json")[0])
    assert failed["attempts"][0]["reason"].startswith("lease-expired")
    status = queue.status(include_jobs=True)
    assert status["counts"]["done"] == 1 and status["drained"] is True
    assert store.get("leases/a.json") is None


def test_the_completion_race_has_one_winner_under_threads(store):
    clock = Clock()
    queue = _queue(store, clock)
    queue.push("a", _job("a"))
    claims = [queue.claim(OWNER_A)]
    clock.advance(61)
    claims.append(queue.claim(OWNER_B))
    results = []
    barrier = threading.Barrier(2)

    def finish(claim):
        barrier.wait()
        results.append(queue.complete(claim, {"exit_code": 0}))

    threads = [threading.Thread(target=finish, args=(c,)) for c in claims]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == [False, True]


def test_failures_back_off_then_die_and_requeue_revives(store):
    clock = Clock()
    queue = _queue(store, clock, max_attempts=2)
    queue.push("a", _job("a"))
    claim = queue.claim(OWNER_A)
    assert queue.fail(claim, exit_code=1, reason="runner exit 1",
                      error_tail="boom", log_key=None) == "backoff"
    assert queue.status()["counts"]["backoff"] == 1
    assert queue.claim(OWNER_A) is None
    clock.advance(5.1)
    claim = queue.claim(OWNER_B)
    assert claim.attempt == 2
    assert queue.fail(claim, exit_code=1, reason="runner exit 1",
                      error_tail="boom again", log_key=None) == "dead"
    dead = json.loads(store.get("dead/a.json")[0])
    assert dead["failures"] == 2 and dead["error_tail"] == "boom again"
    clock.advance(1000)
    assert queue.claim(OWNER_A) is None
    assert queue.status()["counts"]["dead"] == 1 and queue.drained()
    assert queue.requeue(all_dead=True)["jobs"] == {"a": "requeued-from-dead"}
    revived = queue.claim(OWNER_A)
    assert revived.failures == 0
    assert json.loads(store.get("failed/a.json")[0])["failures"] == 2


def test_a_permanent_exit_is_dead_at_once(store):
    queue = _queue(store)
    queue.push("a", _job("a"))
    claim = queue.claim(OWNER_A)
    assert queue.fail(claim, exit_code=2, reason="refused", error_tail="",
                      log_key=None, permanent=True) == "dead"


def test_a_lease_expiring_on_the_last_attempt_is_dead(store):
    clock = Clock()
    queue = _queue(store, clock, max_attempts=1)
    queue.push("a", _job("a"))
    queue.claim(OWNER_A)
    clock.advance(61)
    assert queue.claim(OWNER_B) is None
    assert queue.status()["counts"]["dead"] == 1


def test_release_hands_the_job_back_uncharged(store):
    queue = _queue(store)
    queue.push("a", _job("a"))
    claim = queue.claim(OWNER_A)
    assert queue.release(claim, "SIGTERM") is True
    again = queue.claim(OWNER_B)
    assert (again.attempt, again.failures, again.lease["preemptions"]) == (2, 0, 1)


def test_requeue_refuses_done_and_running_jobs(store):
    clock = Clock()
    queue = _queue(store, clock)
    queue.push("a", _job("a"))
    queue.push("b", _job("b"))
    claims = {c.job_id: c for c in (queue.claim(OWNER_A), queue.claim(OWNER_B))}
    queue.complete(claims["a"], {"exit_code": 0})
    with pytest.raises(wq.RequeueRefused, match="done"):
        queue.requeue(["a"])
    with pytest.raises(wq.RequeueRefused, match="running"):
        queue.requeue(["b"])
    queue.fail(claims["b"], exit_code=1, reason="x", error_tail="",
               log_key=None)
    assert queue.requeue(["b"])["jobs"] == {"b": "backoff-cleared"}
    assert queue.claim(OWNER_A).job_id == "b"
    with pytest.raises(wq.RequeueRefused):
        queue.requeue([])


# --------------------------------------------------------------------------
# Worker pieces
# --------------------------------------------------------------------------


def test_gpu_selection():
    inventory = wk.fake_gpus(3)
    assert len(wk.select_gpus("all", inventory)) == 3
    assert [g.uuid for g in wk.select_gpus("2", inventory)] == [
        "GPU-fake-0000", "GPU-fake-0001"]
    assert [g.index for g in wk.select_gpus("GPU-fake-0002", inventory)] == [2]
    for bad in ("4", "0", "GPU-nope", "GPU-fake-0000,GPU-fake-0000"):
        with pytest.raises(wk.GPUSelectionError):
            wk.select_gpus(bad, inventory)


def test_the_worker_does_not_read_gpus_under_no_local_gpu(monkeypatch):
    import argparse

    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    args = argparse.Namespace(no_gpu=False, fake_gpus=None, gpus=None,
                              slots=None)
    with pytest.raises(wk.GPUSelectionError, match="GPUWM_NO_LOCAL_GPU"):
        wk._devices(args)


def test_runner_templates_refuse_unknown_placeholders():
    with pytest.raises(wk.RunnerTemplateError, match="nope"):
        wk.SubprocessRunner("run {job} {nope}")
    with pytest.raises(wk.RunnerTemplateError):
        wk.SubprocessRunner("")


def _request(tmp_path, *, timeout=None, cancel=None) -> wk.RunnerRequest:
    return wk.RunnerRequest(
        "a", tmp_path / "a.json", 1, tmp_path, tmp_path / "runner.log",
        tmp_path / "result.json", dict(os.environ), timeout,
        cancel or threading.Event(), None, "file:///q")


def test_subprocess_runner_logs_times_out_and_cancels(tmp_path):
    runner = wk.SubprocessRunner(
        "{python} -c \"import sys; print('hello', sys.argv[1])\" {job_id}")
    assert runner(_request(tmp_path)).exit_code == 0
    assert "hello a" in (tmp_path / "runner.log").read_text()
    sleeper = wk.SubprocessRunner(
        "{python} -c \"import time; time.sleep(30)\"", term_grace_s=1.0)
    started = time.monotonic()
    outcome = sleeper(_request(tmp_path, timeout=0.5))
    assert outcome.timed_out and time.monotonic() - started < 10
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    outcome = sleeper(_request(tmp_path, cancel=cancel))
    assert outcome.cancelled and outcome.exit_code == -signal.SIGTERM


def _slot(tmp_path, queue, runner, **config) -> wk.Slot:
    values = dict(queue=queue.url, slot=0, worker_id="w0",
                  workdir=str(tmp_path / "slot"), poll_s=0.05)
    values.update(config)
    return wk.Slot(wk.SlotConfig(**values), queue=queue, runner=runner)


def test_a_slot_completes_with_the_runner_result_and_uploads_the_log(tmp_path):
    queue = _queue(wq.open_store(f"file://{tmp_path / 'q'}"))
    queue.push("a", _job("a"))

    def runner(request):
        assert json.loads(request.job_path.read_text())["job_id"] == "a"
        assert request.env["WOOF_ENERGY_JOB_ID"] == "a"
        request.log_path.write_text("ran\n")
        request.result_path.write_text(json.dumps({"store": "s3://x"}))
        return wk.RunnerOutcome(0)

    result = _slot(tmp_path, queue, runner, exit_when_empty=True).run()
    assert (result.exit_code, result.reason, result.counts) == (
        0, "drained", {"completed": 1})
    done = json.loads(queue.store.get("done/a.json")[0])
    assert done["result"] == {"store": "s3://x"}
    assert queue.store.get(done["log_key"])[0] == b"ran\n"


def test_a_slot_charges_failures_and_a_refusal_is_dead(tmp_path):
    queue = _queue(wq.open_store(f"file://{tmp_path / 'q'}"),
                   backoff_base_s=0.0, backoff_max_s=0.0)
    queue.push("a", _job("a"))
    queue.push("b", _job("b"))

    def runner(request):
        request.log_path.write_text(f"error in {request.job_id}\n")
        return wk.RunnerOutcome(2 if request.job_id == "b" else 1)

    result = _slot(tmp_path, queue, runner, exit_when_empty=True).run()
    assert result.counts == {"failed-backoff": 2, "failed-dead": 2}
    dead = json.loads(queue.store.get("dead/b.json")[0])
    assert dead["reason"].startswith("permanent") and "error in b" in dead["error_tail"]
    assert json.loads(queue.store.get("dead/a.json")[0])["failures"] == 3


def test_a_stopped_slot_releases_its_job_uncharged(tmp_path):
    queue = _queue(wq.open_store(f"file://{tmp_path / 'q'}"))
    queue.push("a", _job("a"))
    slot = None

    def runner(request):
        slot.request_stop(signal.SIGTERM)
        assert request.cancel.is_set()
        return wk.RunnerOutcome(-15, cancelled=True)

    slot = _slot(tmp_path, queue, runner)
    result = slot.run()
    assert (result.exit_code, result.counts) == (wk.EXIT_STOPPED,
                                                 {"released": 1})
    lease = json.loads(queue.store.get("leases/a.json")[0])
    assert (lease["state"], lease["failures"], lease["preemptions"]) == (
        "released", 0, 1)


def test_a_heartbeat_cancels_a_copy_completed_elsewhere(tmp_path):
    clock = Clock()
    queue = _queue(wq.open_store(f"file://{tmp_path / 'q'}"), clock,
                   lease_s=1.0, heartbeat_s=0.1)
    queue.push("a", _job("a"))
    stale = queue.claim(OWNER_A)
    clock.advance(2)

    def runner(request):
        assert request.cancel.wait(5)
        return wk.RunnerOutcome(-15, cancelled=True)

    slot = _slot(tmp_path, queue, runner)
    other = queue.claim(OWNER_B)
    queue.complete(stale, {"exit_code": 0})
    assert slot.execute(other, queue.settings()) == "superseded"
    assert queue.store.get("leases/a.json") is None


def test_one_slot_per_gpu_and_a_busy_gpu_is_waited_for(tmp_path, monkeypatch):
    from woof.supervisor import GPUFileLock

    monkeypatch.setenv("WOOF_GPU_LOCK_ROOT", str(tmp_path / "locks"))
    queue = _queue(wq.open_store(f"file://{tmp_path / 'q'}"))
    uuid_ = wk.fake_gpus(1)[0].uuid
    assert wk.worker_lock_path(uuid_).parent == tmp_path / "locks"
    with GPUFileLock(uuid_, path=wk.worker_lock_path(uuid_)):
        result = _slot(tmp_path, queue, None, gpu_uuid=uuid_).run()
        assert result.exit_code == wk.EXIT_REFUSED
    with GPUFileLock(uuid_):
        assert wk.gpu_is_free(uuid_) is False
    assert wk.gpu_is_free(uuid_) is True


# --------------------------------------------------------------------------
# CLI and the multi-process end to end
# --------------------------------------------------------------------------


def _woof(*argv, env=None, check=True, timeout=120):
    environment = dict(os.environ, PYTHONPATH=str(REPO),
                       GPUWM_NO_LOCAL_GPU="1", **(env or {}))
    result = subprocess.run([sys.executable, "-m", "woof", *argv],
                            capture_output=True, text=True, env=environment,
                            timeout=timeout)
    if check:
        assert result.returncode == 0, result.stderr
    return result


def test_the_cli_parses_every_queue_and_worker_door():
    from woof.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["energy", "queue", "init", "file:///q",
                              "--lease-s", "30"])
    assert (args.queue_command, args.lease_s, args.heartbeat_s) == (
        "init", 30.0, None)
    args = parser.parse_args(["energy", "worker", "s3://b/p", "--gpus", "2",
                              "--exit-when-empty"])
    assert args.gpus == "2" and args.runner_cmd == wk.DEFAULT_RUNNER_TEMPLATE
    with pytest.raises(SystemExit):
        parser.parse_args(["energy", "worker", "file:///q", "--no-gpu",
                           "--fake-gpus", "2"])


def test_cli_init_push_status_requeue(tmp_path):
    url = f"file://{tmp_path / 'q'}"
    (tmp_path / "jobs").mkdir()
    for name in ("a", "b"):
        (tmp_path / "jobs" / f"{name}.json").write_text(json.dumps(_job(name)))
    refused = _woof("energy", "queue", "push", url, str(tmp_path / "jobs"),
                    check=False)
    assert refused.returncode == 2 and "QueueNotInitialized" in refused.stderr
    _woof("energy", "queue", "init", url, "--lease-s", "20")
    pushed = json.loads(_woof("energy", "queue", "push", url,
                              str(tmp_path / "jobs")).stdout)
    assert pushed["counts"] == {"queued": 2}
    status = json.loads(_woof("energy", "queue", "status", url, "--jobs").stdout)
    assert status["counts"]["pending"] == 2 and len(status["jobs"]) == 2
    refused = _woof("energy", "queue", "requeue", url, "--job", "a",
                    check=False)
    assert refused.returncode == 0  # pending: nothing to do, nothing refused
    assert json.loads(refused.stdout)["jobs"] == {"a": "already-pending"}


FAKE_RUNNER = """
import json, os, sys, time
job, result, marker, sleep = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
job_id = json.load(open(job))["job_id"]
time.sleep(sleep)
with open(os.path.join(marker, f"{job_id}.{os.getpid()}.ran"), "w") as f:
    f.write(os.environ["WOOF_ENERGY_JOB_ATTEMPT"])
json.dump({"job_id": job_id, "pid": os.getpid()}, open(result, "w"))
print("fake runner finished", job_id)
"""


def _pids() -> list[int]:
    return [int(name) for name in os.listdir("/proc") if name.isdigit()]


def _ppid(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return int(stat.rsplit(")", 1)[1].split()[1])


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="reads /proc")
def test_end_to_end_a_killed_worker_lease_expires_and_runs_once(tmp_path):
    url = f"file://{tmp_path / 'q'}"
    runner = tmp_path / "fake_runner.py"
    runner.write_text(FAKE_RUNNER)
    marker = tmp_path / "ran"
    marker.mkdir()
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    for index in range(5):
        (jobs / f"j{index}.json").write_text(json.dumps(_job(f"j{index}")))
    _woof("energy", "queue", "init", url, "--lease-s", "3",
          "--heartbeat-s", "0.5", "--backoff-base-s", "0.5")
    _woof("energy", "queue", "push", url, str(jobs))
    environment = dict(os.environ, PYTHONPATH=str(REPO),
                       GPUWM_NO_LOCAL_GPU="1",
                       WOOF_GPU_LOCK_ROOT=str(tmp_path / "locks"))
    command = [sys.executable, "-m", "woof", "energy", "worker", url,
               "--runner-cmd",
               f"{{python}} {runner} {{job}} {{result}} {marker} 1.5",
               "--exit-when-empty", "--poll-s", "0.2"]
    workers = [subprocess.Popen(
        command + ["--fake-gpus", "2", "--workdir", str(tmp_path / "w0")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=environment)]
    workers += [subprocess.Popen(
        command + ["--no-gpu", "--workdir", str(tmp_path / f"w{n}")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=environment) for n in (1, 2)]
    store = wq.open_store(url)
    victim = None
    deadline = time.monotonic() + 60
    while victim is None and time.monotonic() < deadline:
        for key in store.list("leases/"):
            found = store.get(key)
            if found is None:
                continue
            lease = json.loads(found[0])
            if lease["state"] == "running":
                victim = lease
                break
        time.sleep(0.05)
    assert victim is not None, "no worker claimed a job"
    slot_pid = victim["owner"]["pid"]
    parent_pid = _ppid(slot_pid)
    assert parent_pid in {w.pid for w in workers}
    # The host dies: worker, slot, and the runner's own session.
    runners = [pid for pid in _pids() if _ppid(pid) == slot_pid]
    os.kill(parent_pid, signal.SIGKILL)
    os.kill(slot_pid, signal.SIGKILL)
    for pid in runners:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    outputs = []
    for worker in workers:
        out, err = worker.communicate(timeout=120)
        outputs.append((worker.returncode, out, err))
    survivors = [(code, out) for code, out, err in outputs if code != -9]
    assert len(survivors) == 2
    for code, out in survivors:
        assert code == 0, outputs
        assert json.loads(out)["schema"] == wk.WORKER_SUMMARY_SCHEMA
    status = json.loads(_woof("energy", "queue", "status", url).stdout)
    assert status["counts"]["done"] == 5 and status["drained"]
    victim_job = victim["job_id"]
    done = json.loads(store.get(f"done/{victim_job}.json")[0])
    assert done["attempt"] == 2 and done["owner"]["pid"] != slot_pid
    failed = json.loads(store.get(f"failed/{victim_job}.json")[0])
    assert failed["attempts"][0]["reason"].startswith("lease-expired")
    ran = sorted(p.name.split(".")[0] for p in marker.iterdir())
    assert ran == [f"j{n}" for n in range(5)], "each job ran to the end once"
    assert len(store.list("done/")) == 5


# --------------------------------------------------------------------------
# Review repairs
# --------------------------------------------------------------------------


def test_a_retried_completion_finds_its_own_marker(store):
    queue = _queue(store)
    queue.push("a", _job("a"))
    claim = queue.claim(OWNER_A)
    assert queue.complete(claim, {"exit_code": 0}) is True
    assert queue.complete(claim, {"exit_code": 0}) is True
    assert not [k for k in store.list("logs/") if "duplicate" in k]


def test_a_dead_lease_without_its_marker_is_repaired_by_the_next_claim(store):
    queue = _queue(store)
    queue.push("a", _job("a"))
    claim = queue.claim(OWNER_A)
    lease = dict(claim.lease, state="dead", owner=None, note="crashed here")
    store.put_if_match("leases/a.json", wq.encode(lease), claim.version)
    assert not queue.drained()
    assert queue.claim(OWNER_B) is None
    assert queue.drained()
    assert json.loads(store.get("dead/a.json")[0])["reason"] == "crashed here"


def test_a_failure_record_outage_does_not_lose_the_dead_marker(store,
                                                              monkeypatch):
    queue = _queue(store, max_attempts=1)
    queue.push("a", _job("a"))
    claim = queue.claim(OWNER_A)

    def outage(*args, **kwargs):
        raise OSError("store unavailable")

    monkeypatch.setattr(queue, "_append_failure", outage)
    assert queue.fail(claim, exit_code=1, reason="x", error_tail="",
                      log_key=None) == "dead"
    assert store.get("dead/a.json") is not None


def test_a_runner_that_dies_with_the_stop_signal_is_released_uncharged(
        tmp_path):
    queue = _queue(wq.open_store(f"file://{tmp_path / 'q'}"))
    queue.push("a", _job("a"))
    slot = None

    def runner(request):
        # The same SIGTERM reached the runner's cgroup first: it exited on
        # its own before the slot asked it to.
        slot.request_stop(signal.SIGTERM)
        return wk.RunnerOutcome(-15)

    slot = _slot(tmp_path, queue, runner)
    assert slot.run().counts == {"released": 1}
    assert json.loads(queue.store.get("leases/a.json")[0])["failures"] == 0


def test_a_slot_stops_after_failing_jobs_in_a_row(tmp_path):
    queue = _queue(wq.open_store(f"file://{tmp_path / 'q'}"), max_attempts=1)
    for index in range(8):
        queue.push(f"j{index}", _job(f"j{index}"))
    result = _slot(tmp_path, queue,
                   lambda request: wk.RunnerOutcome(1)).run()
    assert (result.exit_code, result.reason) == (wk.EXIT_FAILED, "error")
    assert result.counts == {"failed-dead": wk.MAX_CONSECUTIVE_JOB_FAILURES}
    assert queue.status()["counts"]["pending"] == (
        8 - wk.MAX_CONSECUTIVE_JOB_FAILURES)


def test_a_gpu_taken_between_claim_and_start_releases_the_job(
        tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_GPU_LOCK_ROOT", str(tmp_path / "locks"))
    queue = _queue(wq.open_store(f"file://{tmp_path / 'q'}"))
    queue.push("a", _job("a"))
    answers = iter([True, False])
    slot = None

    def probe(uuid_):
        answer = next(answers, None)
        if answer is None:
            slot.request_stop()
            return False
        return answer

    monkeypatch.setattr(wk, "gpu_is_free", probe)

    def runner(request):
        raise AssertionError("the runner must not start on a taken GPU")

    slot = _slot(tmp_path, queue, runner, gpu_uuid=wk.fake_gpus(1)[0].uuid)
    slot.run()
    lease = json.loads(queue.store.get("leases/a.json")[0])
    assert (lease["state"], lease["failures"]) == ("released", 0)


def test_the_worker_refuses_a_default_runner_this_install_lacks(
        tmp_path, monkeypatch, capsys):
    from woof.cli import build_parser

    url = f"file://{tmp_path / 'q'}"
    _queue(wq.open_store(url))
    monkeypatch.setattr(wk, "default_runner_available", lambda: False)
    args = build_parser().parse_args(["energy", "worker", url, "--no-gpu"])
    assert wk.main(args) == wk.EXIT_REFUSED
    assert "hindcast run-local" in capsys.readouterr().err
