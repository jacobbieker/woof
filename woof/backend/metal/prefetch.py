"""Build the Metal units the last run used, in parallel, at startup.

A unit's library compiles from MSL when its module is made, and each
kernel's pipeline at its first launch (woof.backend.metal.loader).  Both ran
one at a time behind a single lock, in the order the forecast reached them,
so a run whose compiled code was not cached paid every compile end to end:
a cold launch (every release changes the translations), and a warm one
whose libraries macOS's own shader cache had evicted.

So each run records the units it loaded -- name, CUDA source, options and
the kernels whose pipelines it built -- under a plan named by the run's
configuration (woof.kernel_plan's tag, with the numerics mode).  The next
run with that plan, from its first module on, translates them on one thread
(a translation-cache hit is a file read) and builds each library and its
pipelines on a pool; the extension compiles each unit under its own lock
without the GIL.  The forecast's own load of a unit translates the same
source the same way, so its compile is a cache hit, or a wait for that one
unit.  The compiled code is the same: the same MSL, options and pipeline
descriptors, built earlier.  A unit whose translation now differs (a new
woof) is still built ahead: its key is the new one.

``WOOF_METAL_PLAN_DIR`` moves the plans (``off``: no plans, no prefetch);
``WOOF_METAL_PREFETCH_WORKERS`` sets the pool (default: 8, at most the CPU
count; ``0`` records plans without prefetching).  With the dynamic-library
route opted in (``WOOF_METAL_DYLIB_MAX_KERNELS``) nothing is prefetched: that
route builds a unit differently, and a prefetched whole library would
pre-empt it.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
from pathlib import Path

PLAN_DIR_ENV = "WOOF_METAL_PLAN_DIR"
WORKERS_ENV = "WOOF_METAL_PREFETCH_WORKERS"
DEFAULT_WORKERS = 8
#: a unit not loaded in this many consecutive runs of its plan leaves it
KEEP_MISSES = 2
MAX_ENTRIES = 1000

_lock = threading.RLock()
_state: dict = {"started": False, "plan": None, "entries": None, "used": set(), "atexit": False,
                "by_library": {},
                "queued": 0, "built": 0, "failed": 0}


def plan_root() -> Path | None:
    env = os.environ.get(PLAN_DIR_ENV)
    if env in ("off", "0"):
        return None
    if env:
        return Path(env).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "woof" / "metal-plan").expanduser()


def _tag() -> str:
    from woof import numerics
    from woof.kernel_plan import default_tag_text
    text = default_tag_text() + "\0" + numerics.mode()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _workers() -> int:
    raw = os.environ.get(WORKERS_ENV, "").strip()
    if raw.isdigit():
        return int(raw)
    return max(1, min(DEFAULT_WORKERS, os.cpu_count() or 1))


def _dylib_route() -> bool:
    from woof.backend.metal.dylib import max_kernels
    return max_kernels() > 0


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(path: Path) -> list:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return [r for r in rows if isinstance(r, dict) and r.get("unit") and r.get("source")]
    except (OSError, ValueError, TypeError):
        return []


def _plan():
    """(plan path or False, rows by library key) for this run, read once."""
    with _lock:
        if _state["plan"] is None:
            root = plan_root()
            if root is None:
                _state["plan"], _state["entries"] = False, {}
            else:
                path = root / f"{_tag()}.json"
                _state["plan"] = path
                _state["entries"] = {r["unit"]: dict(r, kernels=list(r.get("kernels") or []),
                                                     options=list(r.get("options") or []))
                                     for r in _read(path)}
        return _state["plan"], _state["entries"]


def start() -> None:
    """Prefetch the plan's units (once per process)."""
    with _lock:
        if _state["started"]:
            return
        _state["started"] = True
    path, entries = _plan()
    if not path or not entries or _workers() <= 0 or _dylib_route():
        return
    rows = list(entries.values())
    _state["queued"] = len(rows)
    builds: queue.Queue = queue.Queue()
    threads = [threading.Thread(target=_translate, args=(rows, path.parent, builds, _workers()),
                                name="metal-prefetch-translate", daemon=True)]
    threads += [threading.Thread(target=_build, args=(builds,), name=f"metal-prefetch-{n}", daemon=True)
                for n in range(min(_workers(), len(rows)))]
    for thread in threads:
        thread.start()
    # A compile still running when the interpreter ends would outlive the
    # extension's and MLX's static objects (a test's short process died with
    # SIGSEGV): at exit, nothing new starts and the running compiles finish.
    import atexit
    atexit.register(_stop, threads)


_STOP = threading.Event()


def _stop(threads) -> None:
    _STOP.set()
    for thread in threads:
        thread.join()


def _translate(rows, root: Path, builds: queue.Queue, workers: int) -> None:
    from woof.backend.metal.loader import _with_nonce, translate_unit
    try:
        for row in rows:
            if _STOP.is_set():
                break
            try:
                source = (root / "src" / f"{row['source']}.cu").read_text(encoding="utf-8")
                if _digest(source) != row["source"]:
                    raise ValueError("stored source does not match its key")
                unit = _with_nonce(translate_unit(row["name"], source, options=tuple(row["options"])))
                builds.put((f"woof:{row['name']}:{unit.key[:24]}", unit.msl, list(row["kernels"])))
            except Exception:  # noqa: BLE001 -- the forecast's own load reports a failure
                _state["failed"] += 1
    finally:
        for _ in range(workers):
            builds.put(None)


def _build(builds: queue.Queue) -> None:
    from woof.backend.metal import ext
    from woof.backend.metal.loader import MATH_MODE_SAFE
    module = ext.load()
    while True:
        job = builds.get()
        if job is None:
            return
        if _STOP.is_set():
            continue
        library_key, msl, kernels = job
        try:
            module.compile(library_key, msl, kernels, MATH_MODE_SAFE)
            _state["built"] += 1
        except Exception:  # noqa: BLE001 -- the forecast's own compile reports a failure
            _state["failed"] += 1


def _unit_id(name: str, source_digest: str, options) -> str:
    return _digest("\0".join([name, source_digest, *[str(o) for o in options]]))


def note_unit(library_key: str, name: str, source: str, options) -> None:
    """A unit this run loaded (woof.backend.metal.loader.load_unit)."""
    path, entries = _plan()
    if not path:
        return
    with _lock:
        digest = _digest(source)
        unit = _unit_id(name, digest, options)
        _state["used"].add(unit)
        _state["by_library"][library_key] = unit
        row = entries.get(unit)
        if row is None:
            _store_source(path.parent, digest, source)
            entries[unit] = {"unit": unit, "name": name, "source": digest,
                             "options": [str(o) for o in options], "kernels": [], "misses": 0}
            _write(path, entries, final=False)
        else:
            row["misses"] = 0
        if not _state["atexit"]:
            import atexit
            atexit.register(_finish)
            _state["atexit"] = True


def note_kernels(library_key: str, kernels) -> None:
    """Pipelines this run built for a unit it loaded."""
    path, entries = _plan()
    if not path or not kernels:
        return
    with _lock:
        row = entries.get(_state["by_library"].get(library_key))
        if row is None:
            return
        added = [k for k in kernels if k not in row["kernels"]]
        if added:
            row["kernels"].extend(added)
            _write(path, entries, final=False)


def _store_source(root: Path, digest: str, source: str) -> None:
    target = root / "src" / f"{digest}.cu"
    if target.exists():
        return
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.{os.getpid()}.partial")
        partial.write_text(source, encoding="utf-8")
        os.replace(partial, target)
    except OSError:
        pass


def _write(path: Path, entries: dict, *, final: bool) -> None:
    rows = []
    for key, row in entries.items():
        misses = int(row.get("misses", 0))
        if final and key not in _state["used"]:
            misses += 1
        if misses > KEEP_MISSES:
            continue
        rows.append(dict(row, misses=misses))
    rows.sort(key=lambda r: r["unit"] not in _state["used"])   # this run's first, in order
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
        partial.write_text(json.dumps(rows[:MAX_ENTRIES]), encoding="utf-8")
        os.replace(partial, path)
    except OSError:
        pass


def _finish() -> None:
    path, entries = _plan()
    if path:
        with _lock:
            _write(path, entries, final=True)


def status() -> dict:
    """Prefetch counts so far: units queued, built, failed."""
    return {k: _state[k] for k in ("queued", "built", "failed")}
