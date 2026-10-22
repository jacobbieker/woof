"""History writing waits for a delete-after-commit export to free the disk."""

import inspect
import json

from woof.icechunk_export import LIVE_EXPORT_DELETE_ENV
from woof.io import export_backpressure as bp

GIB = 1024 ** 3


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def _env(domains=None, **extra):
    return {LIVE_EXPORT_DELETE_ENV: json.dumps({"domains": domains}), **extra}


def test_no_export_or_unexported_grid_never_waits():
    free = lambda _path: 0
    assert bp.wait_for_export_space("/x", grid_id=1, largest_frame_bytes=GIB,
                                    env={}, free=free) == 0.0
    assert bp.wait_for_export_space("/x", grid_id=1, largest_frame_bytes=GIB,
                                    env=_env([2]), free=free) == 0.0


def test_a_full_disk_holds_until_the_export_frees_the_reserve():
    clock, lines = Clock(), []
    readings = iter([2 * GIB, 3 * GIB, 7 * GIB, 30 * GIB])
    waited = bp.wait_for_export_space(
        "/x", grid_id=2, largest_frame_bytes=GIB, env=_env([1, 2]),
        free=lambda _path: next(readings), sleep=clock.sleep, clock=clock,
        say=lines.append)
    assert waited == 3 * bp.POLL_SECONDS
    assert "holding the forecast" in lines[0] and "resumes" in lines[-1]


def test_the_reserve_scales_with_the_largest_frame():
    assert bp.reserve_bytes(0) == bp.RESERVE_FLOOR_BYTES
    assert bp.reserve_bytes(10 * GIB) == bp.FRAMES_HELD * 10 * GIB
    clock = Clock()
    readings = iter([20 * GIB, 40 * GIB])
    waited = bp.wait_for_export_space(
        "/x", grid_id=None, largest_frame_bytes=10 * GIB, env=_env(None),
        free=lambda _path: next(readings), sleep=clock.sleep, clock=clock,
        say=lambda _line: None)
    assert waited == bp.POLL_SECONDS


def test_a_dead_exporter_cannot_hold_the_run_past_the_cap():
    clock, lines = Clock(), []
    waited = bp.wait_for_export_space(
        "/x", grid_id=1, largest_frame_bytes=0,
        env=_env(None, **{bp.MAX_WAIT_ENV: "60"}), free=lambda _path: 0,
        sleep=clock.sleep, clock=clock, say=lines.append)
    assert waited >= 60 and "writing anyway" in lines[-1]


def test_the_async_writer_waits_before_queueing_a_frame():
    from woof.io.wrfout import AsyncDomainWrfoutWriter

    source = inspect.getsource(AsyncDomainWrfoutWriter.submit)
    assert "wait_for_export_space(" in source
    assert source.index("wait_for_export_space(") < source.index(
        "get_current_stream()")
