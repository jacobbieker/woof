"""Hold history writing while a delete-after-commit export catches up.

With ``[output.icechunk] delete_after_commit`` the exporter deletes each
wrfout once it is committed, so the output disk holds only the export's
backlog.  The model, though, does not wait for the exporter: on a host
whose CPUs the forecast also uses, the exporter fell 660 frames (11 h of
model time) behind a 1 km tree, the disk filled, and the forecast died
with "No space left on device" -- after which the same exporter, with the
CPUs to itself, cleared the whole backlog in 35 minutes.

So before a frame of an exported grid is queued, the writer looks at the
free space on its disk and, below a reserve, waits for the exporter to
free some.  The reserve is the larger of :data:`RESERVE_FLOOR_BYTES` and
:data:`FRAMES_HELD` of the largest frame this writer has written.  A wait
is capped (:data:`MAX_WAIT_ENV`, default :data:`DEFAULT_MAX_WAIT_S`) so an
exporter that has died cannot hold a run for ever; past the cap the frame
is written and the run meets the disk as it did before.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Mapping

RESERVE_FLOOR_BYTES = 8 * 1024 ** 3
FRAMES_HELD = 3
POLL_SECONDS = 15.0
REPORT_SECONDS = 300.0
MAX_WAIT_ENV = "WOOF_EXPORT_SPACE_MAX_WAIT_S"
DEFAULT_MAX_WAIT_S = 6 * 3600.0


def exported_grid(grid_id: int | None, env: Mapping[str, str]) -> bool:
    """Whether a delete-after-commit export takes this grid's frames."""
    from woof.icechunk_export import LIVE_EXPORT_DELETE_ENV

    raw = env.get(LIVE_EXPORT_DELETE_ENV)
    if not raw:
        return False
    try:
        domains = json.loads(raw).get("domains")
    except (ValueError, AttributeError):
        return False
    return domains is None or (grid_id is not None
                               and int(grid_id) in {int(g) for g in domains})


def reserve_bytes(largest_frame_bytes: int) -> int:
    return max(RESERVE_FLOOR_BYTES, FRAMES_HELD * int(largest_frame_bytes or 0))


def wait_for_export_space(directory, *, grid_id: int | None,
                          largest_frame_bytes: int,
                          env: Mapping[str, str] | None = None,
                          free: Callable[[Path], int | None] | None = None,
                          sleep: Callable[[float], None] = time.sleep,
                          clock: Callable[[], float] = time.monotonic,
                          say: Callable[[str], None] | None = None) -> float:
    """Block until ``directory``'s disk has the reserve free; seconds waited.

    Returns at once (0.0) when no delete-after-commit export takes this
    grid, when free space cannot be read, or when the reserve is free.
    """
    env = os.environ if env is None else env
    if not exported_grid(grid_id, env):
        return 0.0
    if free is None:
        from woof.disk_budget import free_bytes as free
    say = say or (lambda line: print(line, flush=True))
    need = reserve_bytes(largest_frame_bytes)
    try:
        cap = float(env.get(MAX_WAIT_ENV, DEFAULT_MAX_WAIT_S))
    except ValueError:
        cap = DEFAULT_MAX_WAIT_S
    label = "history" if grid_id is None else f"history d{int(grid_id):02d}"
    started = clock()
    reported = None
    while True:
        available = free(Path(directory))
        if available is None or available >= need:
            if reported is not None:
                say(f"{label}: export freed the disk "
                    f"({available / 1024 ** 3:.1f} GiB free) after "
                    f"{clock() - started:.0f} s; writing resumes")
            return clock() - started
        waited = clock() - started
        if waited >= cap:
            say(f"{label}: waited {waited:.0f} s for the "
                f"export to free the disk ({available / 1024 ** 3:.1f} GiB "
                f"free, {need / 1024 ** 3:.1f} GiB wanted); writing anyway")
            return waited
        if reported is None or clock() - reported >= REPORT_SECONDS:
            say(f"{label}: {available / 1024 ** 3:.1f} GiB "
                f"free, below the {need / 1024 ** 3:.1f} GiB reserve; "
                "holding the forecast while the export commits and "
                "deletes frames")
            reported = clock()
        sleep(POLL_SECONDS)
