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
  timed on hardware by this tree;
* :class:`KernelCache` / :func:`parse_kernel_cache` -- whether children
  share one persistent GPU kernel cache (the default) or each start from
  an empty one of their own (``per-run``).

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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


# ---------------------------------------------------------------------------
# GPU kernel caches: shared (default) or isolated per run / slot
# ---------------------------------------------------------------------------

#: The variable CuPy resolves its on-disk kernel cache from.
CUPY_CACHE_ENV = "CUPY_CACHE_DIR"
#: The variable the CUDA driver resolves its JIT (ComputeCache) from.
CUDA_CACHE_ENV = "CUDA_CACHE_PATH"
#: Every child reads and fills one persistent kernel cache.
KERNEL_CACHE_SHARED = "shared"
#: Every child starts from its own new, empty kernel cache.
KERNEL_CACHE_PER_RUN = "per-run"
#: Where CuPy puts its cache when ``CUPY_CACHE_DIR`` is unset (its own
#: documented default, :func:`woof.kernel_compile_notice.cupy_kernel_cache_dir`).
CUPY_DEFAULT_CACHE = "~/.cupy/kernel_cache"
#: Where the driver puts its JIT cache when ``CUDA_CACHE_PATH`` is unset.
CUDA_DRIVER_DEFAULT_CACHE = (
    "the CUDA driver's per-user default (~/.nv/ComputeCache on Linux, "
    "%APPDATA%\\NVIDIA\\ComputeCache on Windows)")
#: Why several processes may share one kernel cache, recorded in receipts.
KERNEL_CACHE_CONCURRENCY = (
    "CuPy (14.x, cupy/cuda/_compiler_cache.py DiskKernelCacheBackend) "
    "writes each compiled kernel to a NamedTemporaryFile in the cache "
    "directory and publishes it with os.replace, and every read checks the "
    "SHA-1 the file starts with, so a concurrent writer is never seen half "
    "written and an unreadable entry is recompiled, not used; its key "
    "includes the source, every option, the NVRTC version and the target "
    "architecture, so one cache serves several cards without a mix-up. "
    "The CUDA driver's JIT cache is the per-user cache every CUDA process "
    "of the user shares by default, and the driver manages concurrent use "
    "of it")


@dataclass(frozen=True)
class KernelCache:
    """Where a packing door's children keep compiled GPU kernels.

    ``shared`` (the default): every child reads and fills ONE persistent
    cache, so only the first run on a machine pays the NVRTC compile
    (typically 1-3 minutes) and every later one -- each packed sibling,
    each of a hindcast worker's thousands of jobs -- starts warm.  With no
    ``directory`` that is the cache an ordinary ``woof run`` from the same
    shell uses: the inherited ``CUPY_CACHE_DIR`` / ``CUDA_CACHE_PATH``, or
    CuPy's ``~/.cupy/kernel_cache`` and the driver's per-user default when
    unset (the child environment is left exactly as inherited).  With a
    ``directory``, ``CUPY_CACHE_DIR=<dir>/cupy`` and
    ``CUDA_CACHE_PATH=<dir>/cuda``.

    ``per-run``: the isolation multi-run had before -- ``<cache>/cupy``
    and ``<cache>/cuda`` under each run's own new cache root, empty at
    launch.  For a cross-device numerical comparison or a forensic run
    that must prove each process compiled its own kernels.

    What is pinned: temporary directories, outputs and GPU locks are
    isolated per child in both modes; only the kernel caches are shared.
    Concurrent writers are safe (:data:`KERNEL_CACHE_CONCURRENCY`).
    ``source`` names where the setting came from, for the receipt;
    ``isolated_word`` is what the door calls ``per-run`` (the energy
    worker says ``per-slot``).
    """

    mode: str = KERNEL_CACHE_SHARED
    directory: Path | None = None
    source: str = "default"
    isolated_word: str = KERNEL_CACHE_PER_RUN

    @property
    def shared(self) -> bool:
        return self.mode == KERNEL_CACHE_SHARED

    @property
    def label(self) -> str:
        """The mode as this door's flag spells it (``per-slot``, ...)."""

        return KERNEL_CACHE_SHARED if self.shared else self.isolated_word

    def paths(self, run_cache: Path,
              parent: Mapping[str, str]) -> tuple[str, str | None]:
        """``(CuPy cache, CUDA JIT cache)`` a child with ``run_cache`` uses.

        The CUDA cache is ``None`` when it is the driver's own default
        (shared, ``CUDA_CACHE_PATH`` unset): the driver resolves it
        (:data:`CUDA_DRIVER_DEFAULT_CACHE`), this process does not."""

        if not self.shared:
            return str(Path(run_cache) / "cupy"), str(Path(run_cache) / "cuda")
        if self.directory is not None:
            return (str(self.directory / "cupy"),
                    str(self.directory / "cuda"))
        cupy = (parent.get(CUPY_CACHE_ENV) or "").strip()
        cuda = (parent.get(CUDA_CACHE_ENV) or "").strip()
        return (cupy or str(Path(CUPY_DEFAULT_CACHE).expanduser()),
                cuda or None)

    def apply(self, environment: dict[str, str], run_cache: Path) -> None:
        """Set (or, shared without a directory, leave) the two variables."""

        if self.shared and self.directory is None:
            return
        cupy, cuda = self.paths(run_cache, environment)
        assert cuda is not None  # set explicitly in both of these modes
        environment[CUPY_CACHE_ENV] = cupy
        environment[CUDA_CACHE_ENV] = cuda

    def prepare(self) -> None:
        """Create a shared ``directory``'s two caches (kept across runs)."""

        if self.shared and self.directory is not None:
            for child in ("cupy", "cuda"):
                (self.directory / child).mkdir(parents=True, exist_ok=True)

    def receipt(self, parent: Mapping[str, str] | None = None
                ) -> dict[str, Any]:
        environment = os.environ if parent is None else parent
        record: dict[str, Any] = {
            "mode": self.label, "source": self.source,
            "directory": None if self.directory is None else str(
                self.directory),
        }
        if self.shared:
            cupy, cuda = self.paths(Path("."), environment)
            if cuda is None:
                record["cuda_cache_basis"] = (
                    "CUDA_CACHE_PATH is unset, so the driver uses "
                    + CUDA_DRIVER_DEFAULT_CACHE)
            record.update(
                cupy_cache=cupy, cuda_cache=cuda,
                basis=("one persistent cache every child reads and fills"
                       + ("" if self.directory is not None else
                          ", inherited exactly as an ordinary woof run "
                          "from this shell finds it")),
                concurrency=KERNEL_CACHE_CONCURRENCY)
            if environment.get("CUDA_CACHE_DISABLE"):
                record["cuda_cache_disable"] = environment["CUDA_CACHE_DISABLE"]
        else:
            record["basis"] = (
                "each child's own new <cache>/cupy and <cache>/cuda, empty "
                "at launch, so every child compiles every kernel it uses")
        return record


def parse_kernel_cache(value: Any, *, flag: str, base: Path,
                       source: str,
                       isolated_word: str = KERNEL_CACHE_PER_RUN
                       ) -> KernelCache:
    """``shared``, ``isolated_word`` or a directory path; never guessed.

    A directory must be spelled as a path (it contains a separator or
    starts with ``~`` or ``.``): a bare word that is not a keyword --
    ``per_run``, ``Shared`` -- is refused rather than silently becoming
    a directory of that name.  Relative paths resolve against ``base``.
    """

    keywords = (KERNEL_CACHE_SHARED, isolated_word)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{flag} takes {keywords[0]!r}, {keywords[1]!r} or a directory "
            f"path, not {value!r}")
    text = value.strip()
    if text == KERNEL_CACHE_SHARED:
        return KernelCache(KERNEL_CACHE_SHARED, None, source, isolated_word)
    if text == isolated_word:
        return KernelCache(KERNEL_CACHE_PER_RUN, None, source, isolated_word)
    looks_like_path = (any(sep in text for sep in {"/", os.sep})
                       or text.startswith(("~", ".")))
    if not looks_like_path:
        raise ValueError(
            f"{flag} takes {keywords[0]!r}, {keywords[1]!r} or a directory "
            f"path, not {text!r}; write ./{text} for a directory of that "
            "name")
    directory = Path(text).expanduser()
    if not directory.is_absolute():
        directory = Path(base) / directory
    directory = directory.resolve()
    if directory.exists() and not directory.is_dir():
        raise ValueError(
            f"{flag} {directory} exists and is not a directory")
    return KernelCache(KERNEL_CACHE_SHARED, directory, source, isolated_word)


__all__ = ["AUTO_MAX_SLOTS_PER_GPU", "CUDA_CACHE_ENV", "CUPY_CACHE_ENV",
           "DEFAULT_MEM_MARGIN_GIB", "KERNEL_CACHE_CONCURRENCY",
           "KERNEL_CACHE_PER_RUN", "KERNEL_CACHE_SHARED", "KernelCache",
           "MPS_CONTROL", "MPS_COMMAND_TIMEOUT_S", "MPS_PIPE_PATH_MAX",
           "MPSUnavailable", "MpsDaemon", "SLOTS_AUTO",
           "parse_count_or_auto", "parse_kernel_cache"]
