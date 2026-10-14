"""Several hindcast jobs per GPU: admission, MPS, plan pricing, cost, deploy.

``woof energy worker --slots-per-gpu N|auto`` packs jobs whose declared
``resources.gpu_mem_gib`` fits each card; ``--mps`` runs the CUDA MPS
control daemon for them.  No GPU is touched: stand-in GPU identities
(``--fake-gpus``) under a temporary ``WOOF_GPU_LOCK_ROOT``, fake runners
that record which jobs share a card, and a fake ``nvidia-cuda-mps-control``
on ``PATH``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from woof.energy import cost
from woof.energy import deploy
from woof.energy import workqueue as wq
from woof.energy import worker as wk

REPO = Path(__file__).resolve().parents[1]
SITES = REPO / "tests" / "fixtures" / "energy" / "sites_wales.json"
OWNER = {"worker_id": "a", "host": "node-a", "pid": 1, "gpu_uuid": "GPU-a"}


def _job(job_id: str, gib=None, **extra) -> dict:
    document = {"schema": wq.JOB_SCHEMA, "job_id": job_id, **extra}
    if gib is not None:
        document["resources"] = {"gpus": 1, "gpu_mem_gib": gib}
    return document


def _queue(path: Path, **settings) -> wq.WorkQueue:
    queue = wq.WorkQueue.open(f"file://{path}")
    defaults = dict(max_attempts=3, lease_s=60.0, heartbeat_s=10.0,
                    backoff_base_s=5.0, backoff_max_s=40.0)
    defaults.update(settings)
    queue.init(wq.QueueSettings(**defaults))
    return queue


def _woof(*argv, env=None, check=True, timeout=180):
    environment = dict(os.environ, PYTHONPATH=str(REPO),
                       GPUWM_NO_LOCAL_GPU="1", **(env or {}))
    result = subprocess.run([sys.executable, "-m", "woof", *argv],
                            capture_output=True, text=True, env=environment,
                            timeout=timeout)
    if check:
        assert result.returncode == 0, result.stderr
    return result


# --------------------------------------------------------------------------
# Queue: declared memory and admission-filtered claims
# --------------------------------------------------------------------------


def test_declared_gpu_memory_is_validated_at_push(tmp_path):
    queue = _queue(tmp_path / "q")
    assert queue.push("a", _job("a", 4.5)) == "queued"
    assert queue.push("b", _job("b")) == "queued"
    assert queue.push("z", _job("z", 0)) == "queued"  # 0: no GPU
    for bad in (-0.5, -1, "8", float("nan"), True):
        with pytest.raises(wq.JobSpecError, match="gpu_mem_gib"):
            queue.push("c", {"schema": wq.JOB_SCHEMA, "job_id": "c",
                             "resources": {"gpu_mem_gib": bad}})
    with pytest.raises(wq.JobSpecError, match="resources"):
        wq.job_from_document({"schema": wq.JOB_SCHEMA, "job_id": "d",
                              "resources": [1]})
    assert wq.declared_gpu_mem_gib(_job("e", 3)) == 3.0
    assert wq.declared_gpu_mem_gib(_job("f")) is None


def test_an_admission_filtered_claim_skips_what_does_not_fit(tmp_path):
    queue = _queue(tmp_path / "q")
    for name, gib in (("big", 20.0), ("small", 2.0), ("none", None)):
        queue.push(name, _job(name, gib))
    seen = []

    def admit(job_id, gib):
        seen.append((job_id, gib))
        return gib is not None and gib <= 4.0

    claim = queue.claim(OWNER, admit=admit)
    assert claim.job_id == "small"
    assert sorted(seen)[0][0] in {"big", "none", "small"}
    assert queue.claim(OWNER, admit=admit) is None
    assert queue.last_unadmitted == 2
    # Rejected jobs were never leased: anyone else may take them.
    assert sorted(k.split("/")[-1] for k in queue.store.list("leases/")) == [
        "small.json"]
    assert queue.claim(OWNER).job_id in {"big", "none"}


def test_admission_reads_are_bounded_and_cached(tmp_path, monkeypatch):
    # One document read serves both gates, so the tighter budget binds.
    monkeypatch.setattr(wq, "MAX_ADMISSION_READS_PER_CLAIM", 3)
    monkeypatch.setattr(wq, "MAX_DEPENDENCY_READS_PER_CLAIM", 3)
    queue = _queue(tmp_path / "q")
    for index in range(8):
        queue.push(f"j{index}", _job(f"j{index}", 50.0))
    reads = []
    original = queue.store.get
    monkeypatch.setattr(queue.store, "get", lambda key: (
        reads.append(key), original(key))[1])
    assert queue.claim(OWNER, admit=lambda *_: False) is None
    assert len(queue._declared) == 3 and len(queue._dependencies) == 3
    assert len([k for k in reads if k.startswith("jobs/")]) == 3
    assert queue.last_unadmitted == 3
    for _ in range(3):
        queue.claim(OWNER, admit=lambda *_: False)
    assert len(queue._declared) == 8


def test_a_claim_needs_its_dependencies_done_and_its_memory_to_fit(tmp_path):
    queue = _queue(tmp_path / "q")
    queue.push("prefetch-c0", _job("prefetch-c0", kind="prefetch"))
    queue.push("f1", _job("f1", 6.0, depends_on=["prefetch-c0"]))
    queue.push("f2", _job("f2", 2.0, depends_on=["prefetch-c0"]))
    assert wq.declared_gpu_mem_gib(_job("x", kind="prefetch")) == 0.0
    seen = []

    def admit(job_id, gib):
        seen.append((job_id, gib))
        return gib <= 4.0

    first = queue.claim(OWNER, admit=admit)
    assert first.job_id == "prefetch-c0" and seen == [("prefetch-c0", 0.0)]
    # Its dependents wait for it, whatever their memory.
    assert queue.claim(OWNER, admit=admit) is None
    assert queue.last_unadmitted == 0
    assert queue.complete(first, {"exit_code": 0})
    claim = queue.claim(OWNER, admit=admit)
    assert claim.job_id == "f2"  # f1 is ready but does not fit
    assert queue.claim(OWNER, admit=admit) is None
    assert queue.last_unadmitted == 1


def test_a_prefetch_is_never_held_back_and_reserves_nothing(tmp_path,
                                                            monkeypatch):
    from woof.supervisor import GPUFileLock

    monkeypatch.setenv("WOOF_GPU_LOCK_ROOT", str(tmp_path / "locks"))
    big, other = _ledger(0), _ledger(1)
    held = GPUFileLock("x", path=big.slot_lock_path).acquire()
    try:
        big.reserve(_claim("huge"), 22.0, [])  # over the card, running alone
        live = other.live_reservations()
        assert not other.fits(2.0, live)
        assert other.fits(0.0, live)
        lock = GPUFileLock("y", path=other.slot_lock_path).acquire()
        try:
            other.reserve(_claim("prefetch-c0"), 0.0, live)
            assert not other.reservation_path.exists()
            assert [r["job_id"] for r in big.live_reservations()] == []
        finally:
            lock.release()
    finally:
        held.release()


# --------------------------------------------------------------------------
# The card ledger
# --------------------------------------------------------------------------


def _ledger(index, **kwargs):
    values = dict(card_gib=16.0, margin_gib=1.0, default_job_gib=None)
    values.update(kwargs)
    return wk.CardLedger(wk.fake_gpus(1)[0].uuid, index, **values)


def _claim(job_id):
    return wq.Claim(job_id, {}, b"", {"attempt": 1, "failures": 0}, "v")


def test_the_ledger_admits_what_fits_and_an_idle_card_takes_anything(
        tmp_path, monkeypatch):
    from woof.supervisor import GPUFileLock

    monkeypatch.setenv("WOOF_GPU_LOCK_ROOT", str(tmp_path / "locks"))
    first, second = _ledger(0), _ledger(1)
    assert first.fits(40.0, []) and first.fits(None, [])
    held = GPUFileLock("x", path=first.slot_lock_path).acquire()
    try:
        first.reserve(_claim("a"), 9.0, [])
        live = second.live_reservations()
        assert [row["job_id"] for row in live] == ["a"]
        assert second.fits(6.0, live)          # 9 + 6 + 1 <= 16
        assert not second.fits(6.5, live)      # 9 + 6.5 + 1 > 16
        assert not second.fits(None, live)     # undeclared: the whole card
        assert _ledger(1, default_job_gib=4.0).fits(None, live)
    finally:
        held.release()
    # Nobody holds slot 0 any more: its reservation file is stale.
    assert first.reservation_path.exists()
    assert second.live_reservations() == []


@pytest.mark.skipif(os.name != "posix", reason="flock inheritance")
def test_a_reservation_lives_as_long_as_the_runner_holding_its_lock(
        tmp_path, monkeypatch):
    from woof.supervisor import GPUFileLock

    monkeypatch.setenv("WOOF_GPU_LOCK_ROOT", str(tmp_path / "locks"))
    ledger = _ledger(0)
    lock = GPUFileLock("x", path=ledger.slot_lock_path).acquire()
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        pass_fds=(lock._stream.fileno(),))
    try:
        # The slot dies (its descriptor closes without an unlock); the
        # runner that inherited the descriptor did not.
        lock._stream.close()
        lock._stream = None
        assert wk._lock_held(ledger.slot_lock_path)
    finally:
        child.kill()
        child.wait()
    assert not wk._lock_held(ledger.slot_lock_path)


def test_sibling_probes_share_and_an_outside_holder_excludes(
        tmp_path, monkeypatch):
    from woof.supervisor import GPUFileLock

    monkeypatch.setenv("WOOF_GPU_LOCK_ROOT", str(tmp_path / "locks"))
    uuid_ = wk.fake_gpus(1)[0].uuid
    assert wk.gpu_is_free_shared(uuid_)  # no lock file yet
    with GPUFileLock(uuid_):  # a forecast outside the queue
        assert not wk.gpu_is_free_shared(uuid_)
    import fcntl

    from woof.supervisor import default_lock_path

    # A sibling slot mid-probe (shared) does not look like a holder.
    with open(default_lock_path(uuid_), "rb") as sibling:
        fcntl.flock(sibling, fcntl.LOCK_SH | fcntl.LOCK_NB)
        assert wk.gpu_is_free_shared(uuid_)


def test_the_slots_flag_has_one_grammar_everywhere():
    from woof.energy.packing import parse_slots_per_gpu

    assert parse_slots_per_gpu("AUTO") == "auto"
    assert parse_slots_per_gpu(3) == 3 and parse_slots_per_gpu(" 2 ") == 2
    for bad in (0, "0", True, "-1", "two", "", "1.5"):
        with pytest.raises(ValueError):
            parse_slots_per_gpu(bad)
    assert deploy.worker_command(_deploy_config(slots_per_gpu="AUTO"))[-2:] \
        == ["--slots-per-gpu", "auto"]
    assert cost._slots_arg("Auto") == "auto"


def test_auto_slots_are_bounded_by_memory_cpus_and_the_cap():
    assert wk.auto_slots_per_gpu(24.0, 1.0, 1, cpus=64) == 6
    assert wk.auto_slots_per_gpu(80.0, 1.0, 1, cpus=64) == 8
    assert wk.auto_slots_per_gpu(80.0, 1.0, 4, cpus=12) == 3
    assert wk.auto_slots_per_gpu(4.0, 1.0, 1, cpus=64) == 1


def _args(**values):
    defaults = dict(slots_per_gpu=1, mem_margin_gib=None, gpu_mem_gib=None,
                    default_job_mem_gib=None, mps=False, mps_dir=None,
                    mps_active_thread_percentage=None)
    defaults.update(values)
    return argparse.Namespace(**defaults)


def test_packing_flags_are_refused_where_they_would_be_ignored():
    gpus = list(wk.fake_gpus(2))
    assert not wk._packing(_args(), gpus).packed
    for values, match in (
            ({"gpu_mem_gib": 16.0}, "--gpu-mem-gib"),
            ({"mem_margin_gib": 2.0}, "--mem-margin-gib"),
            ({"mps": True}, "nothing to share"),
            ({"slots_per_gpu": 2, "mps_dir": "/x"}, "--mps-dir"),
            ({"slots_per_gpu": 2, "mps": True,
              "mps_active_thread_percentage": 0}, "1-100"),
            ({"slots_per_gpu": "two"}, "positive count")):
        with pytest.raises(wk.PackingError, match=match):
            wk._packing(_args(**values), gpus)
    with pytest.raises(wk.PackingError, match="--no-gpu"):
        wk._packing(_args(slots_per_gpu="auto"), [None])
    with pytest.raises(wk.PackingError, match="--gpu-mem-gib"):
        wk.card_memory_gib(gpus[0], None)
    assert wk.card_memory_gib(gpus[0], 16.0) == (16.0, "--gpu-mem-gib")


# --------------------------------------------------------------------------
# MPS
# --------------------------------------------------------------------------

FAKE_MPS = r"""#!{python}
import json, os, sys
pipe = os.environ["CUDA_MPS_PIPE_DIRECTORY"]
log = os.path.join(os.environ["FAKE_MPS_LOG"])
control = os.path.join(pipe, "control")
def record(what):
    with open(log, "a") as f:
        f.write(json.dumps({{"call": what,
            "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pipe": pipe, "log": os.environ.get("CUDA_MPS_LOG_DIRECTORY")}}) + "\n")
if sys.argv[1:] == ["-d"]:
    if os.path.exists(control):
        print("An instance of this daemon is already running", file=sys.stderr)
        sys.exit(1)
    open(control, "w").close()
    record("start")
    sys.exit(0)
command = sys.stdin.read().strip()
if command == "get_server_list":
    sys.exit(0 if os.path.exists(control) else 1)
if command == "quit":
    record("quit")
    os.remove(control)
    sys.exit(0)
sys.exit(3)
"""


def _fake_mps(tmp_path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / wk.MPS_CONTROL
    script.write_text(FAKE_MPS.format(python=sys.executable))
    script.chmod(0o755)
    return bin_dir, tmp_path / "mps-calls.jsonl"


def test_mps_is_refused_without_the_control_binary(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    with pytest.raises(wk.MPSUnavailable, match="nvidia-cuda-mps-control"):
        wk.MpsDaemon(tmp_path / "mps", ["GPU-x"]).start()


def test_mps_refuses_a_socket_path_too_long(tmp_path, monkeypatch):
    bin_dir, _ = _fake_mps(tmp_path)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    with pytest.raises(wk.MPSUnavailable, match="--mps-dir"):
        wk.MpsDaemon(tmp_path / ("d" * 120), ["GPU-x"]).start()


def test_mps_starts_reuses_and_stops(tmp_path, monkeypatch):
    bin_dir, calls = _fake_mps(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_MPS_LOG", str(calls))
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    daemon = wk.MpsDaemon(tmp_path / "mps", ["GPU-a", "GPU-b"],
                          thread_percentage=50).start()
    assert daemon.started and not daemon.reused
    assert daemon.environment() == {
        "CUDA_MPS_PIPE_DIRECTORY": str(tmp_path / "mps" / "pipe"),
        "CUDA_MPS_LOG_DIRECTORY": str(tmp_path / "mps" / "log"),
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "50"}
    # A second worker finds it answering and leaves it alone.
    other = wk.MpsDaemon(tmp_path / "mps", ["GPU-a"]).start()
    assert other.reused and not other.started
    other.stop()
    daemon.stop()
    rows = [json.loads(line) for line in calls.read_text().splitlines()]
    assert [row["call"] for row in rows] == ["start", "quit"]
    assert rows[0]["visible"] == "GPU-a,GPU-b"
    # An externally owned daemon (CUDA_MPS_PIPE_DIRECTORY set) is used,
    # never started or stopped here.
    external = wk.MpsDaemon(tmp_path / "mps", ["GPU-a"]).start()
    external.stop()
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path / "plugin"))
    with pytest.raises(wk.MPSUnavailable, match="no MPS control daemon"):
        wk.MpsDaemon(None, ["GPU-a"]).start()


MPS_RUNNER = """
import json, os, sys
json.dump({k: os.environ.get(k) for k in (
    "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY",
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "CUDA_VISIBLE_DEVICES",
    "WOOF_GPU_LOCK_ROOT", "WOOF_ENERGY_GPU_SLOT")}, open(sys.argv[1], "w"))
"""


def test_the_worker_runs_packed_slots_under_mps(tmp_path):
    bin_dir, calls = _fake_mps(tmp_path)
    url = f"file://{tmp_path / 'q'}"
    _woof("energy", "queue", "init", url, "--lease-s", "20")
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    for index in range(2):
        (jobs / f"m{index}.json").write_text(json.dumps(_job(f"m{index}", 3)))
    _woof("energy", "queue", "push", url, str(jobs))
    runner = tmp_path / "runner.py"
    runner.write_text(MPS_RUNNER)
    locks = tmp_path / "locks"
    result = _woof(
        "energy", "worker", "--queue", url, "--fake-gpus", "1",
        "--slots-per-gpu", "2", "--gpu-mem-gib", "16", "--mps",
        "--mps-dir", str(tmp_path / "mps"),
        "--mps-active-thread-percentage", "50",
        "--runner-cmd", f"{{python}} {runner} {{result}}",
        "--exit-when-empty", "--poll-s", "0.2",
        "--workdir", str(tmp_path / "w"),
        env={"PATH": f"{bin_dir}:{os.environ['PATH']}",
             "FAKE_MPS_LOG": str(calls), "WOOF_GPU_LOCK_ROOT": str(locks)})
    summary = json.loads(result.stdout)
    assert summary["exit_code"] == 0 and summary["totals"] == {"completed": 2}
    assert summary["mps"]["started_here"] is True
    assert summary["packing"]["cards"]["GPU-fake-0000"]["slots"] == 2
    store = wq.open_store(url)
    seen = [json.loads(store.get(f"done/m{i}.json")[0])["result"]
            for i in range(2)]
    for env in seen:
        assert env["CUDA_MPS_PIPE_DIRECTORY"] == str(tmp_path / "mps" / "pipe")
        assert env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "50"
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-fake-0000"
        assert Path(env["WOOF_GPU_LOCK_ROOT"]).parent == locks
        assert env["WOOF_GPU_LOCK_ROOT"].endswith(".runner-locks")
    assert [json.loads(line)["call"] for line in
            calls.read_text().splitlines()] == ["start", "quit"]


def test_a_card_auto_sizes_to_one_slot_still_runs_packed(tmp_path):
    url = f"file://{tmp_path / 'q'}"
    _woof("energy", "queue", "init", url)
    (tmp_path / "j.json").write_text(json.dumps(_job("solo", 2.0)))
    _woof("energy", "queue", "push", url, str(tmp_path / "j.json"))
    result = _woof("energy", "worker", url, "--fake-gpus", "1",
                   "--slots-per-gpu", "auto", "--gpu-mem-gib", "4",
                   "--runner-cmd", "{python} -c pass", "--exit-when-empty",
                   "--poll-s", "0.2", "--workdir", str(tmp_path / "w"),
                   env={"WOOF_GPU_LOCK_ROOT": str(tmp_path / "locks")})
    summary = json.loads(result.stdout)
    assert summary["packing"]["cards"]["GPU-fake-0000"]["slots"] == 1
    assert summary["totals"] == {"completed": 1}
    assert summary["slots"][0]["gpu_slot"] == 0


def test_one_worker_owns_a_card_whatever_its_slot_count(tmp_path,
                                                         monkeypatch):
    from woof.supervisor import GPUFileLock

    url = f"file://{tmp_path / 'q'}"
    _woof("energy", "queue", "init", url)
    locks = tmp_path / "locks"
    env = {"WOOF_GPU_LOCK_ROOT": str(locks)}
    uuid_ = wk.fake_gpus(1)[0].uuid
    monkeypatch.setenv("WOOF_GPU_LOCK_ROOT", str(locks))
    path = wk.worker_lock_path(uuid_)
    with GPUFileLock(uuid_, path=path):  # a single-slot worker's slot
        refused = _woof("energy", "worker", url, "--fake-gpus", "1",
                        "--slots-per-gpu", "auto", "--gpu-mem-gib", "16",
                        "--runner-cmd", "{python} -c pass", "--max-idle-s",
                        "1", "--poll-s", "0.2", env=env, check=False)
    assert refused.returncode == wk.EXIT_REFUSED, refused.stderr
    assert "already has an energy worker" in refused.stderr
    rows = json.loads(refused.stdout)["slots"]
    assert rows and all(row["exit_code"] == wk.EXIT_REFUSED for row in rows)
    assert all(row["pids"] == [] for row in rows)


def test_the_worker_cli_takes_the_queue_as_a_flag_or_a_positional():
    from woof.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["energy", "worker", "--queue", "s3://b/q",
                              "--slots-per-gpu", "auto", "--mps"])
    assert wk._queue_url(args) == "s3://b/q"
    assert args.slots_per_gpu == "auto" and args.mps
    args = parser.parse_args(["energy", "worker", "s3://b/q"])
    assert wk._queue_url(args) == "s3://b/q" and args.slots_per_gpu == 1
    args = parser.parse_args(["energy", "worker", "s3://b/q", "--queue",
                              "s3://b/other"])
    with pytest.raises(wk.WorkerError, match="twice"):
        wk._queue_url(args)
    with pytest.raises(SystemExit):
        parser.parse_args(["energy", "worker", "s3://b/q", "--slots-per-gpu",
                           "0"])


# --------------------------------------------------------------------------
# End to end: 3 fake GPUs x --slots-per-gpu auto, a killed worker reclaimed
# --------------------------------------------------------------------------

OCCUPANCY_RUNNER = r"""
import fcntl, json, os, sys, time
job_path, result, root, sleep = sys.argv[1:5]
job = json.load(open(job_path))
gib = (job.get("resources") or {}).get("gpu_mem_gib")
if job.get("kind") == "prefetch":
    gib = 0.0
host = os.environ["FAKE_HOST"]
started = time.time()
gpu = os.environ["CUDA_VISIBLE_DEVICES"]
card = os.path.join(root, "cards", host, gpu)
os.makedirs(card, exist_ok=True)
mine = os.path.join(card, f"{job['job_id']}.{os.getpid()}")
with open(os.path.join(root, "occupancy.lock"), "a") as guard:
    fcntl.flock(guard, fcntl.LOCK_EX)
    with open(mine, "w") as f:
        json.dump({"gib": gib}, f)
    present = []
    for name in os.listdir(card):
        try:
            present.append(json.load(open(os.path.join(card, name)))["gib"])
        except (OSError, ValueError):
            pass
    with open(os.path.join(root, "occupancy.jsonl"), "a") as log:
        log.write(json.dumps({"host": host, "gpu": gpu, "job": job["job_id"],
                              "present": present, "at": started}) + "\n")
time.sleep(float(sleep))
with open(os.path.join(root, "ran", f"{job['job_id']}.{os.getpid()}"), "w") as f:
    f.write(os.environ["WOOF_ENERGY_JOB_ATTEMPT"])
os.remove(mine)
with open(os.path.join(root, "ends.jsonl"), "a") as log:
    log.write(json.dumps({"job": job["job_id"], "at": time.time()}) + "\n")
json.dump({"job_id": job["job_id"], "gpu": gpu}, open(result, "w"))
"""

#: Declared memory of the end-to-end jobs on 16 GiB cards with a 1 GiB
#: margin: small ones pack, 15 runs alone, 22 (over the card) runs alone on
#: an idle card, None (undeclared) is priced at the whole card.
E2E_GIB = (2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 2.5, 3.5,
           2.0, 3.0, 4.5, 15.0, 22.0, None)


def _descendants(root_pid: int) -> list[int]:
    parents: dict[int, int] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            stat = Path(f"/proc/{name}/stat").read_text()
        except OSError:
            continue
        parents[int(name)] = int(stat.rsplit(")", 1)[1].split()[1])
    found, frontier = [], [root_pid]
    while frontier:
        pid = frontier.pop()
        children = [c for c, p in parents.items() if p == pid]
        found.extend(children)
        frontier.extend(children)
    return found


#: Prefetch jobs (CPU and network only) and the jobs that need them.
E2E_PREFETCH = {"prefetch-c0": [f"p{i:02d}" for i in range(0, 6)],
                "prefetch-c1": [f"p{i:02d}" for i in range(6, 10)]}


def _check_occupancy(rows, card_gib=16.0, margin_gib=1.0):
    """GPU jobs sharing a card fit; 0 GiB (prefetch) rides along free."""

    packed = 0
    for row in rows:
        present = [gib for gib in row["present"] if gib != 0]
        if len(present) > 1:
            packed += 1
            assert None not in present, row
            assert sum(present) + margin_gib <= card_gib + 1e-9, row
    return packed


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="reads /proc")
def test_end_to_end_three_packed_gpus_and_a_killed_worker(tmp_path):
    url = f"file://{tmp_path / 'q'}"
    root = tmp_path / "rec"
    (root / "ran").mkdir(parents=True)
    runner = tmp_path / "occupancy_runner.py"
    runner.write_text(OCCUPANCY_RUNNER)
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    needs = {job: pre for pre, deps in E2E_PREFETCH.items() for job in deps}
    for index, gib in enumerate(E2E_GIB):
        job_id = f"p{index:02d}"
        extra = {"depends_on": [needs[job_id]]} if job_id in needs else {}
        (jobs / f"{job_id}.json").write_text(
            json.dumps(_job(job_id, gib, **extra)))
    for job_id in E2E_PREFETCH:  # undeclared: kind prefetch counts as 0
        (jobs / f"{job_id}.json").write_text(
            json.dumps(_job(job_id, kind="prefetch")))
    _woof("energy", "queue", "init", url, "--lease-s", "5",
          "--heartbeat-s", "1", "--backoff-base-s", "0.5")
    _woof("energy", "queue", "push", url, str(jobs))
    command = [sys.executable, "-m", "woof", "energy", "worker", url,
               "--slots-per-gpu", "auto", "--gpu-mem-gib", "16",
               "--runner-cmd",
               f"{{python}} {runner} {{job}} {{result}} {root} 1.5",
               "--exit-when-empty", "--poll-s", "0.2"]
    base = dict(os.environ, PYTHONPATH=str(REPO), GPUWM_NO_LOCAL_GPU="1")
    # Host A: three cards, killed (worker, slots and runners) mid-flight.
    host_a = subprocess.Popen(
        command + ["--fake-gpus", "3", "--workdir", str(tmp_path / "wa")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=dict(base, FAKE_HOST="A",
                 WOOF_GPU_LOCK_ROOT=str(tmp_path / "locks-a")))
    store = wq.open_store(url)
    deadline = time.monotonic() + 120
    running: list[dict] = []
    while time.monotonic() < deadline and len(running) < 3:
        running = []
        for key in store.list("leases/"):
            found = store.get(key)
            if found is None:
                continue
            lease = json.loads(found[0])
            if (lease["state"] == "running"
                    and store.get(f"done/{lease['job_id']}.json") is None):
                running.append(lease)
        time.sleep(0.05)
    assert len(running) >= 3, "host A did not claim three jobs"
    # The host dies: freeze the whole tree first (a slot could otherwise
    # start one more runner between the listing and the kill), then kill.
    victims: set[int] = set()
    while True:
        found = set(_descendants(host_a.pid)) | {host_a.pid}
        for pid in found - victims:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
        if found <= victims:
            break
        victims |= found
    for pid in victims:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    host_a.communicate(timeout=60)
    assert host_a.returncode == -signal.SIGKILL
    killed = {lease["job_id"] for lease in running
              if store.get(f"done/{lease['job_id']}.json") is None}
    # Host B: one card, auto slots, drains everything (A's leases expire).
    host_b = subprocess.run(
        command + ["--fake-gpus", "1", "--workdir", str(tmp_path / "wb")],
        capture_output=True, text=True, timeout=300,
        env=dict(base, FAKE_HOST="B",
                 WOOF_GPU_LOCK_ROOT=str(tmp_path / "locks-b")))
    assert host_b.returncode == 0, host_b.stderr
    summary = json.loads(host_b.stdout)
    assert summary["schema"] == wk.WORKER_SUMMARY_SCHEMA
    card = summary["packing"]["cards"]["GPU-fake-0000"]
    assert card["slots"] == 4 and card["card_mem_gib"] == 16.0
    status = json.loads(_woof("energy", "queue", "status", url).stdout)
    assert status["counts"]["done"] == len(E2E_GIB) + len(E2E_PREFETCH)
    assert status["drained"]
    # Exactly once: every job ran to the end once, A's killed copies never.
    ran = sorted(path.name.split(".")[0] for path in (root / "ran").iterdir())
    assert ran == sorted([f"p{i:02d}" for i in range(len(E2E_GIB))]
                         + list(E2E_PREFETCH))
    for job_id in killed:
        done = json.loads(store.get(f"done/{job_id}.json")[0])
        assert done["attempt"] >= 2 and done["owner"]["host"]
        failed = json.loads(store.get(f"failed/{job_id}.json")[0])
        assert failed["attempts"][0]["reason"].startswith("lease-expired")
    # Per card, concurrent jobs never exceeded the budget.
    rows = [json.loads(line) for line in
            (root / "occupancy.jsonl").read_text().splitlines()]
    assert {row["host"] for row in rows} == {"A", "B"}
    assert _check_occupancy(rows) > 0, "no two jobs ever shared a card"
    alone = [row for row in rows if row["job"] in ("p15", "p16", "p17")
             and row["host"] == "B"]
    for row in alone:
        assert len([g for g in row["present"] if g != 0]) == 1, row
    # Dependencies before memory: no dependent started before its
    # prefetch finished (a killed prefetch never finished).
    ends = {}
    for line in (root / "ends.jsonl").read_text().splitlines():
        end = json.loads(line)
        ends[end["job"]] = end["at"]
    for prefetch, dependents in E2E_PREFETCH.items():
        starts = [row["at"] for row in rows if row["job"] in dependents]
        assert starts and min(starts) >= ends[prefetch], prefetch


# --------------------------------------------------------------------------
# Plan-time pricing (resources.gpu_mem_gib)
# --------------------------------------------------------------------------


def _hindcast_request(tmp_path, **overrides):
    from woof.energy.hindcast import HindcastRequest

    values = dict(outdir=tmp_path / "hc", start=datetime(2025, 11, 1),
                  end=datetime(2025, 11, 2), sites_path=SITES, source="gfs",
                  vram_gib=24.0)
    values.update(overrides)
    return HindcastRequest(**values)


def test_plan_records_each_job_priced_gpu_memory(tmp_path):
    from woof.energy.hindcast import plan_hindcast

    calls = []

    def pricer(job, sites, *, now=None):
        calls.append(job["job_id"])
        return {"gpu_mem_gib": 5.25, "basis": "test"}

    result = plan_hindcast(_hindcast_request(tmp_path), dry_run=True,
                           gpu_mem_pricer=pricer)
    assert len(result["jobs"]) == 4 and len(calls) == 1
    for job in result["jobs"]:
        assert job["resources"]["gpu_mem_gib"] == 5.25
        assert job["resources"]["gpu_mem_basis"].startswith("test; priced on")
        assert wq.declared_gpu_mem_gib(job) == 5.25
    other = plan_hindcast(
        _hindcast_request(tmp_path), dry_run=True,
        gpu_mem_pricer=lambda job, sites, now=None: {"gpu_mem_gib": None,
                                                     "basis": "x"})
    assert [j["job_id"] for j in other["jobs"]] == [
        j["job_id"] for j in result["jobs"]], "pricing is not identity"


def test_the_default_pricer_uses_the_planners_own_pricing(tmp_path):
    from woof.energy.hindcast import plan_hindcast

    nests = plan_hindcast(_hindcast_request(tmp_path), dry_run=True)
    resources = nests["jobs"][0]["resources"]
    assert 1.0 < resources["gpu_mem_gib"] < 24.0
    assert resources["gpu_mem_basis"].startswith("wrf-nests planner priced")
    tiles = plan_hindcast(_hindcast_request(tmp_path, topology="wrf-tiles"),
                          dry_run=True)
    resources = tiles["jobs"][0]["resources"]
    assert 1.0 < resources["gpu_mem_gib"] < 24.0
    assert "upper bound" in resources["gpu_mem_basis"]
    unpriced = plan_hindcast(_hindcast_request(tmp_path, vram_gib=None),
                             dry_run=True)
    resources = unpriced["jobs"][0]["resources"]
    assert resources["gpu_mem_gib"] is None
    assert resources["gpu_mem_basis"].startswith("unpriced: no --card")
    assert not (Path(os.environ.get("TMPDIR", "/tmp"))
                / "woof-price-only").exists()


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------


def _estimate(clusters, **kwargs):
    table, _ = cost.load_prices(None)
    offers = cost.select_offers(table, ["aws:g5.xlarge"])
    return cost.estimate(mode="summary", campaign=cost.Campaign(4, 6.0, 0.0),
                         offers=offers, price_table=table, clusters=clusters,
                         **kwargs)["offers"][0]


def _clusters(area_km2):
    return [cost.ClusterSpec("c", area_km2, 200.0, None, None, None,
                             area_km2 / 2.0, 10)]


def test_packing_helps_small_domains_and_not_card_filling_ones():
    small = _clusters(20.0)
    base = _estimate(small)
    assert base.get("packing") is None
    assert _estimate(small, slots_per_gpu=1)["gpu_hours"] == base["gpu_hours"]
    timesliced = _estimate(small, slots_per_gpu="auto")
    mps = _estimate(small, slots_per_gpu="auto", mps=True)
    assert mps["gpu_hours"]["central"] < timesliced["gpu_hours"]["central"] \
        < base["gpu_hours"]["central"]
    assert mps["gpu_hours_one_job_per_gpu"] == base["gpu_hours"]
    roles = mps["packing"]["by_role"]
    assert all(row["copies_max"] > 1 and row["compute_gain_max"] > 1.0
               for row in roles.values())
    for label in ("low", "central", "high"):
        assert mps["gpu_hours"][label] <= base["gpu_hours"][label]
    # A grid priced at most of the card packs once: no gain at all.
    card = cost.card_row("a10g")
    big = cost.DomainWork("d", "leaf", 900, 900, 60, 200.0, 1.0, 3600.0,
                          peak_gib=20.0)
    assert cost.packed_copies(big, card, "auto") == 1
    alone, _ = cost._sum_seconds([big], card, 1000.0, 1.0)
    packed, _ = cost._sum_seconds([big], card, 1000.0, 1.0,
                                  slots_per_gpu="auto", mps=True,
                                  realisation=1.0)
    assert packed == alone


def test_packing_curve_saturates_with_cells():
    card = cost.card_row("a10g")
    gains = []
    for side in (60, 200, 600):
        domain = cost.DomainWork("d", "leaf", side, side, 60, 200.0, 1.0,
                                 3600.0, peak_gib=2.0)
        factors = cost.packing_factors(domain, card, 4, mps=True,
                                       realisation=1.0)
        gains.append(factors["ideal_compute_gain"])
        assert 1.0 <= factors["ideal_compute_gain"] <= 4.0
    assert gains[0] > gains[1] > gains[2]
    no_mps = cost.packing_factors(domain, card, 4, mps=False, realisation=1.0)
    assert no_mps["bandwidth"] == no_mps["launch"] == 1.0
    assert no_mps["overhead"] == 0.25
    full = cost.packing_factors(domain, card, 4, mps=True, realisation=1.0)
    assert full["ideal_launch_gain"] == 4.0 and full["launch"] == 0.25


def test_mps_without_packing_is_refused():
    with pytest.raises(cost.CostRefusal, match="nothing is packed"):
        _estimate(_clusters(20.0), mps=True)
    with pytest.raises(cost.CostRefusal, match="slots-per-gpu"):
        _estimate(_clusters(20.0), slots_per_gpu=0)


def test_the_estimate_cli_takes_slots_per_gpu_and_mps():
    result = _woof("energy", "hindcast", "estimate", "--clusters", "10",
                   "--footprint-km2", "100", "--instance", "aws:g5.xlarge",
                   "--slots-per-gpu", "auto", "--mps", "--segments", "4")
    document = json.loads(result.stdout)
    assert document["model"]["assumptions"]["slots_per_gpu"] == "auto"
    assert "packing" in document["offers"][0]
    assert "packed: --slots-per-gpu auto with CUDA MPS" in result.stderr


# --------------------------------------------------------------------------
# Deployment
# --------------------------------------------------------------------------


def _deploy_config(**overrides):
    mapping = {"name": "wf", "image": "reg.example.com/woof:1.0.3-cu12",
               "queue_url": "s3://bkt/queue", "output_url": "s3://bkt/out",
               "gpus_per_worker": 1}
    mapping.update(overrides)
    return deploy.config_from_mapping(mapping)


def test_every_backend_passes_slots_and_mps_to_the_worker():
    config = _deploy_config(slots_per_gpu="auto", mps=True)
    command = deploy.worker_command(config)
    assert command[-3:] == ["--slots-per-gpu", "auto", "--mps"]
    for backend in deploy.BACKENDS:
        if backend == "gcp-batch":
            rendered = deploy.render(_deploy_config(
                slots_per_gpu="auto", mps=True, queue_url="gs://b/q",
                output_url="gs://b/o", gpus_per_worker=None), backend)
        else:
            rendered = deploy.render(config, backend)
        text = "\n".join(rendered.files.values())
        assert "--slots-per-gpu" in text and "--mps" in text, backend
    k8s = deploy.render(config, "k8s")
    example = k8s.files["mps-sharing.example.yaml"]
    assert all(line.startswith("#") for line in example.splitlines()
               if line.strip())
    assert "sharing:" in example and "replicas: 8" in example
    assert any("device plugin" in note for note in k8s.notes)
    plain = deploy.render(_deploy_config(), "k8s")
    assert "mps-sharing.example.yaml" not in plain.files
    assert "--slots-per-gpu" not in deploy.worker_command(_deploy_config())


def test_deploy_refuses_mps_without_packing_and_flags_in_worker_args():
    for overrides, match in (
            ({"mps": True}, "slots_per_gpu"),
            ({"slots_per_gpu": "0"}, "slots_per_gpu"),
            ({"slots_per_gpu": "lots"}, "slots_per_gpu"),
            ({"slots_per_gpu": "01", "mps": True}, "slots_per_gpu"),
            ({"worker_args": ["--mps"]}, "worker_args"),
            ({"worker_args": ["--slots-per-gpu=2"]}, "worker_args")):
        with pytest.raises(deploy.DeployRefused, match=match):
            deploy.render(_deploy_config(**overrides), "slurm")


def test_the_deploy_cli_overrides_slots_and_mps(tmp_path):
    result = _woof("energy", "deploy", "render", "--backend", "k8s",
                   "--name", "wf", "--image", "reg.example.com/woof:1.0.3",
                   "--queue", "s3://b/q", "--output-url", "s3://b/o",
                   "--gpus-per-worker", "1", "--slots-per-gpu", "3", "--mps",
                   "-o", str(tmp_path / "out"))
    summary = json.loads(result.stdout)
    assert summary["worker_command"][-3:] == ["--slots-per-gpu", "3", "--mps"]
    assert "mps-sharing.example.yaml" in summary["files"]


def test_the_entrypoint_adds_packing_flags_to_a_worker_command(tmp_path):
    import shutil

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "woof"
    fake.write_text("#!/bin/sh\necho \"$@\"\n")
    fake.chmod(0o755)
    scratch = tmp_path / "scratch"
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path),
           "WOOF_SCRATCH": str(scratch), "WOOF_SYNC_ON_START": "0",
           "WOOF_ENERGY_QUEUE_URL": "s3://b/q", "WOOF_SLOTS_PER_GPU": "auto",
           "WOOF_ENABLE_MPS": "1", "JOB_COMPLETION_INDEX": "3"}
    entrypoint = REPO / "deploy" / "entrypoint.sh"
    result = subprocess.run([bash, str(entrypoint)], capture_output=True,
                            text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        "energy", "worker", "--queue", "s3://b/q", "--gpus", "all",
        "--slots-per-gpu", "auto", "--mps", "--mps-dir",
        str(scratch / "mps" / "3")]
    # A full command that already carries the flags is left alone.
    result = subprocess.run(
        [bash, str(entrypoint), "woof", "energy", "worker", "--queue",
         "s3://b/q", "--slots-per-gpu", "2", "--mps"],
        capture_output=True, text=True, env=env)
    assert result.stdout.split().count("--slots-per-gpu") == 1
    assert result.stdout.split().count("--mps") == 1
    refused = subprocess.run([bash, str(entrypoint)], capture_output=True,
                             text=True, env={**env, "WOOF_ENABLE_MPS": "maybe"})
    assert refused.returncode == 2 and "WOOF_ENABLE_MPS" in refused.stderr
