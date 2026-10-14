"""What every door that packs several GPU processes onto one card shares.

``woof energy worker --slots-per-gpu`` (hindcast jobs from a queue) and
``woof multi-run`` with ``runs_per_gpu`` (independent forecasts from a
plan) both put more than one CUDA process on a physical GPU.  They share
the count grammar, the pinned numbers and the CUDA MPS control daemon
from here, so the two doors cannot drift apart:

* :func:`parse_count_or_auto` -- ``auto`` (any case) or a positive count;
  an int is accepted as itself; anything else is a ValueError naming the
  flag it was given for;
* :data:`AUTO_MAX_SLOTS_PER_GPU` -- the most processes ``auto`` puts on
  one card;
* :data:`DEFAULT_MEM_MARGIN_GIB` -- card memory kept free beside the
  priced reservations.  An assumption, not a measurement: the CUDA
  driver, an MPS server and allocator fragmentation that a run's priced
  peak does not hold;
* :class:`MpsDaemon` -- starts (or reuses) ``nvidia-cuda-mps-control -d``
  under a short pipe/log directory, hands its variables to children and
  stops only a daemon it started.  Without MPS, kernels from separate
  processes time-slice; with it they share the SMs.  MPS has not been
  timed on hardware by this tree.

:mod:`woof.energy.packing` re-exports the numbers and keeps its own
``--slots-per-gpu`` parser; :mod:`woof.energy.worker` subclasses
:class:`MpsDaemon` so its refusal stays a ``WorkerError``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SLOTS_AUTO = "auto"
AUTO_MAX_SLOTS_PER_GPU = 8
DEFAULT_MEM_MARGIN_GIB = 1.0

#: The MPS control binary ``--mps`` starts.
MPS_CONTROL = "nvidia-cuda-mps-control"
#: Longest ``<pipe dir>/control`` socket path accepted (AF_UNIX paths
#: stop at 107 bytes on Linux; the rest is margin).
MPS_PIPE_PATH_MAX = 100
#: Seconds an MPS control command may take before it is a failure.
MPS_COMMAND_TIMEOUT_S = 30.0


def parse_count_or_auto(value: Any, *, flag: str) -> int | str:
    """``"auto"`` or a positive int; anything else is a ValueError."""

    refusal = f"{flag} takes a positive count or 'auto', not {value!r}"
    if isinstance(value, bool):
        raise ValueError(refusal)
    if isinstance(value, int):
        if value >= 1:
            return value
        raise ValueError(refusal)
    text = str(value).strip()
    if text.lower() == SLOTS_AUTO:
        return SLOTS_AUTO
    if text.isascii() and text.isdigit() and int(text) >= 1:
        return int(text)
    raise ValueError(refusal)


class MPSUnavailable(RuntimeError):
    """``--mps`` was asked for and the MPS control daemon cannot run."""


class MpsDaemon:
    """The CUDA MPS control daemon a packing door uses.

    ``root`` holds ``pipe/`` and ``log/`` (default a short directory under
    the system temp: the control socket path must stay under
    :data:`MPS_PIPE_PATH_MAX` characters).  When ``CUDA_MPS_PIPE_DIRECTORY``
    is already set (a daemon this process does not own, such as the NVIDIA
    device plugin's MPS sharing) or a daemon already answers at that pipe
    directory, it is used and left running; otherwise ``-d`` starts one and
    :meth:`stop` sends it ``quit`` (and removes the default root, never a
    directory the caller named).  The control binary missing from
    ``PATH`` is a refusal (:attr:`unavailable_error`).
    """

    #: What a refusal raises; a door narrows it to its own error family.
    unavailable_error: type[Exception] = MPSUnavailable

    def __init__(self, root: Path | None, devices: list[str], *,
                 thread_percentage: int | None = None):
        external = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
        self.external = bool(external)
        #: A default root this object chose, removed after it stops.
        self.default_root: Path | None = None
        if external:
            self.pipe_dir = Path(external)
            self.log_dir = Path(os.environ.get("CUDA_MPS_LOG_DIRECTORY")
                                or self.pipe_dir)
        else:
            if root is None:
                root = self.default_root = (
                    Path(tempfile.gettempdir()) / f"woof-mps-{os.getpid()}")
            root = Path(root)
            self.pipe_dir = root / "pipe"
            self.log_dir = root / "log"
        self.devices = list(devices)
        self.thread_percentage = thread_percentage
        self.control: str | None = None
        self.owned = False
        self.started = False
        self.reused = False

    def note(self, message: str) -> None:
        """Say something the operator should see; doors override it."""

        print(f"woof: {message}", file=sys.stderr, flush=True)

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.environment())
        env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
        if self.devices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(self.devices)
        return env

    def environment(self) -> dict[str, str]:
        """What every child process sharing the card gets."""

        env = {"CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_dir),
               "CUDA_MPS_LOG_DIRECTORY": str(self.log_dir)}
        if self.thread_percentage is not None:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
                self.thread_percentage)
        return env

    def _command(self, *argv: str, stdin: str | None = None
                 ) -> subprocess.CompletedProcess:
        assert self.control is not None
        return subprocess.run([self.control, *argv], input=stdin,
                              capture_output=True, text=True,
                              env=self._env(), timeout=MPS_COMMAND_TIMEOUT_S)

    def answers(self) -> bool:
        try:
            return self._command(stdin="get_server_list\n").returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def start(self) -> "MpsDaemon":
        unavailable = self.unavailable_error
        self.control = shutil.which(MPS_CONTROL)
        if self.control is None:
            raise unavailable(
                f"--mps needs {MPS_CONTROL} on PATH (the NVIDIA driver's MPS "
                "control daemon; in a container, the NVIDIA container "
                "toolkit mounts it with the compute and utility "
                "capabilities); none was found")
        socket_path = str(self.pipe_dir / "control")
        if len(socket_path) > MPS_PIPE_PATH_MAX:
            raise unavailable(
                f"the MPS control socket {socket_path} is "
                f"{len(socket_path)} characters, over the "
                f"{MPS_PIPE_PATH_MAX} a UNIX socket path allows here; give "
                "a shorter --mps-dir")
        if self.external:
            if not self.answers():
                raise unavailable(
                    f"CUDA_MPS_PIPE_DIRECTORY={self.pipe_dir} is set but no "
                    "MPS control daemon answers there")
            self.reused = True
            return self
        self.pipe_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if (self.pipe_dir / "control").exists() and self.answers():
            self.reused = True
            return self
        try:
            started = self._command("-d")
        except (OSError, subprocess.TimeoutExpired) as error:
            raise unavailable(f"{MPS_CONTROL} -d failed: {error}") \
                from error
        if started.returncode != 0:
            raise unavailable(
                f"{MPS_CONTROL} -d exited {started.returncode}: "
                f"{(started.stderr or started.stdout).strip()[-500:]}")
        self.owned = self.started = True
        deadline = time.monotonic() + MPS_COMMAND_TIMEOUT_S
        while not self.answers():
            if time.monotonic() >= deadline:
                self.stop()
                raise unavailable(
                    f"the MPS control daemon started at {self.pipe_dir} "
                    "does not answer get_server_list")
            time.sleep(0.2)
        return self

    def stop(self) -> None:
        if not self.owned:
            return
        self.owned = False
        try:
            self._command(stdin="quit\n")
        except (OSError, subprocess.TimeoutExpired) as error:
            self.note(f"MPS control daemon did not stop cleanly ({error})")
            return
        if self.default_root is not None:
            shutil.rmtree(self.default_root, ignore_errors=True)

    def summary(self) -> dict[str, Any]:
        return {"pipe_dir": str(self.pipe_dir), "log_dir": str(self.log_dir),
                "control": self.control, "started_here": self.started,
                "reused_running_daemon": self.reused,
                "external": self.external,
                "active_thread_percentage": self.thread_percentage}


__all__ = ["AUTO_MAX_SLOTS_PER_GPU", "DEFAULT_MEM_MARGIN_GIB", "MPS_CONTROL",
           "MPS_COMMAND_TIMEOUT_S", "MPS_PIPE_PATH_MAX", "MPSUnavailable",
           "MpsDaemon", "SLOTS_AUTO", "parse_count_or_auto"]
