"""Kernel compile pools are sized by the CPU the container grants, not by nproc.

Breakage this prevents (U-E5, 2026-10-26): a rented container with nproc 256
and a 61.44-core CFS quota sized the kernel-plan pool from affinity alone, so
a cold kernel cache opened 256 compiler threads and 256 spawn compiler
processes on 61 cores, and the first native parent step of a WOOF Regional
case took 426.9 s for 15 model seconds (later steps ~0.2 s).
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from woof import kernel_compile_notice, kernel_plan, warm_kernels
from woof.ingest import preparation_workers


def _container(monkeypatch, tmp_path, *, nproc=256, quota="6144000 100000"):
    """A cgroup v2 container: ``nproc`` CPUs in affinity, ``quota`` in cpu.max."""
    (tmp_path / "cpu.max").write_text(quota + "\n")
    real = preparation_workers.cgroup_cpu_count
    monkeypatch.setattr(preparation_workers, "cgroup_cpu_count",
                        lambda: real(root=tmp_path, membership="0::/\n"))
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(nproc)), raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv(kernel_plan.WORKERS_ENVIRONMENT, raising=False)
    monkeypatch.delenv(warm_kernels.WORKERS_ENVIRONMENT, raising=False)
    monkeypatch.setattr(preparation_workers, "host_available_bytes", lambda *a, **k: None)


def test_plan_pool_follows_the_cgroup_quota_not_nproc(monkeypatch, tmp_path):
    _container(monkeypatch, tmp_path)
    assert kernel_plan._workers() == 61


def test_plan_pool_keeps_affinity_where_no_quota_binds(monkeypatch, tmp_path):
    _container(monkeypatch, tmp_path, nproc=24, quota="max 100000")
    assert kernel_plan._workers() == 24


def test_plan_pool_explicit_limit_still_applies_under_the_quota(monkeypatch, tmp_path):
    _container(monkeypatch, tmp_path)
    monkeypatch.setenv(kernel_plan.WORKERS_ENVIRONMENT, "8")
    assert kernel_plan._workers() == 8
    monkeypatch.setenv(kernel_plan.WORKERS_ENVIRONMENT, "200")
    assert kernel_plan._workers() == 61


def test_plan_pool_default_is_capped_by_available_memory(monkeypatch, tmp_path):
    """32 spawn compile workers peaked at 13-15 GB host RSS for one cold
    compile: the default holds COMPILE_WORKER_BYTES per worker within
    COMPILE_MEMORY_SHARE of the available memory."""
    from woof.hex import kernel_plan as hex_plan
    _container(monkeypatch, tmp_path)
    monkeypatch.setattr(preparation_workers, "host_available_bytes", lambda *a, **k: 16 << 30)
    assert kernel_plan._workers() == 8              # 16 GiB * 0.5 / 1 GiB
    assert hex_plan._workers() == 8
    monkeypatch.setattr(preparation_workers, "host_available_bytes", lambda *a, **k: 1 << 29)
    assert kernel_plan._workers() == 1              # never below one
    monkeypatch.setattr(preparation_workers, "host_available_bytes", lambda *a, **k: 1 << 40)
    assert kernel_plan._workers() == 61             # the CPU quota binds first
    # the variable overrides the memory cap; the CPU budget still bounds it
    monkeypatch.setattr(preparation_workers, "host_available_bytes", lambda *a, **k: 4 << 30)
    monkeypatch.setenv(kernel_plan.WORKERS_ENVIRONMENT, "12")
    assert kernel_plan._workers() == 12
    assert hex_plan._workers() == 12
    monkeypatch.setenv(kernel_plan.WORKERS_ENVIRONMENT, "200")
    assert kernel_plan._workers() == 61


def test_profile_workers_follow_the_cgroup_quota(monkeypatch, tmp_path):
    _container(monkeypatch, tmp_path, quota="400000 100000")
    card = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(
        memGetInfo=lambda: (96 << 30, 96 << 30))))
    monkeypatch.setitem(sys.modules, "cupy", card)
    assert warm_kernels.profile_workers(28) == 4


def test_cache_census_threads_follow_the_cgroup_quota(monkeypatch, tmp_path):
    _container(monkeypatch, tmp_path, quota="300000 100000")
    seen = []

    class Pool:
        def __init__(self, max_workers, thread_name_prefix):
            seen.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, function, chunks):
            return [function(chunk) for chunk in chunks]

    import concurrent.futures
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", Pool)
    monkeypatch.setattr(kernel_compile_notice, "_entry_architecture", lambda path: None)
    assert kernel_compile_notice._probe_entries([str(n) for n in range(600)]) == [None] * 600
    assert seen == [3]
