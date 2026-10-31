"""Census: which CUDA kernel units translate to Metal and compile.

For every unit of ``woof/core/kernels`` (the exact text NVRTC would compile,
``woof.core.kernels.module_source``), translate it with
``woof.backend.translate`` and build it with Metal (``MetalModule``, every
kernel's pipeline).  Each unit lands in one class:

``compiled``        translated and every kernel's pipeline built;
``refused``         the translator refused a construct, by name;
``metal_error``     Metal rejected the translation (first error kept);
``source_error``    the unit does not preprocess/parse on its own (it needs
                    integer defines or companion headers its loader adds);
``translated``      (``--translate-only``) the MSL was produced; nothing was
                    compiled, so no Metal device is needed.

``--translate-only`` stops after translation: it is the half of the census a
machine without a Metal device (a GitHub macOS VM) can run.  ``--gate`` exits
non-zero when any unit is ``refused``, ``metal_error``, ``crash`` or
``timeout``; ``source_error`` is the documented class of units that are only
ever compiled with their loader's defines, and does not fail the gate.

Failures are grouped by a normalised message so the most common construct
is fixed first.  ``--retain`` translates with subnormal retention (``f32s``)
as GPUWM_WRF_EXACT=1 would.  Writes JSON (``--out``) and prints a summary.

Results are cached in ``--out`` (JSON lines) under a *toolchain
fingerprint*: the bytes of the translator, the Metal loader and headers,
and every kernel source.  A cached result whose fingerprint differs is
stale and the unit runs again, so the census reports the tree it runs in.
(It once kept results from before goto and writable globals were
translated, and listed goddard and lin as refused after both compiled.)
``source_error`` units are composed with companion headers by their own
loaders (``woof.core.p3_device`` ...); they are counted apart from the
standalone units.  ``timeout`` is no result in ``--timeout`` seconds, not a
failure: rerun those units alone with ``--only``.

Usage::

    pixi run -e mlx python tools/metal_translation_census.py --jobs 8 \
        --out /tmp/metal-census.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from concurrent.futures import as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


#: The classes ``--gate`` fails on.  ``source_error`` is not one: those units
#: are only ever compiled with the defines their loaders add.
GATE_FAILURES = frozenset({"refused", "metal_error", "crash", "timeout"})


def units() -> list[str]:
    kernels = ROOT / "woof" / "core" / "kernels"
    return sorted(p.stem for p in kernels.glob("*.cu")) + composed_units()


def composed_units() -> list[str]:
    """``composed:<module>``: a woof.core module that assembles its own
    unit (``module_source()``, compiled with its ``MODULE_OPTIONS`` through
    ``cupy.RawModule``) -- the text production compiles for the .cu files
    that do not stand alone."""
    out = []
    for path in sorted((ROOT / "woof" / "core").glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"^def module_source\(", text, re.M) and re.search(r"^MODULE_OPTIONS\b", text, re.M):
            out.append(f"composed:{path.stem}")
    return out


def fingerprint() -> str:
    """sha256 over what decides a unit's class: the translator, the Metal
    loader/launcher and headers, and every kernel source."""
    import hashlib
    h = hashlib.sha256()
    roots = [ROOT / "woof" / "backend" / "translate", ROOT / "woof" / "backend" / "metal",
             ROOT / "woof" / "core" / "kernels", ROOT / "woof" / "numerics.py",
             ROOT / "woof" / "wrf_exact.py", Path(__file__).resolve()]
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files += [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                      and p.suffix in (".py", ".h", ".cuh", ".cu", ".cpp", ".mm", ".m", ".json", ".txt")]
    for path in sorted(files):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _normalise(message: str) -> str:
    m = re.sub(r"program_source:\d+:\d+: ", "", message)
    m = re.sub(r"line \d+", "line N", m)
    m = re.sub(r"'[^']*'", "'X'", m)
    m = re.sub(r"\b\d+\b", "N", m)
    return m.strip()[:160]


def census_one(name: str, retain: bool, translate_only: bool = False) -> dict:
    os.environ.setdefault("WOOF_BACKEND", "mlx")
    if retain:
        os.environ["GPUWM_WRF_EXACT"] = "1"
    import woof  # noqa: F401  -- installs the facade
    from woof.backend.metal.loader import CompileError, load_unit, translate_unit
    from woof.backend.translate import TranslationError
    from woof.core import kernels
    t0 = time.perf_counter()
    try:
        if name.startswith("composed:"):
            import importlib
            mod = importlib.import_module(f"woof.core.{name.split(':', 1)[1]}")
            source, options = mod.module_source(), tuple(mod.MODULE_OPTIONS)
        else:
            source, options = kernels.module_source(name), kernels.module_options(name)
    except Exception as exc:  # noqa: BLE001
        return {"unit": name, "class": "source_error", "message": f"{type(exc).__name__}: {exc}"[:400]}
    try:
        if translate_only:
            unit = translate_unit(name.replace(":", "_"), source, options=options)
            return {"unit": name, "class": "translated", "kernels": sorted(unit.kernels),
                    "seconds": round(time.perf_counter() - t0, 2)}
        module = load_unit(name.replace(":", "_"), source, options=options)
        module.build_all()
    except TranslationError as exc:
        text = str(exc)
        cls = "source_error" if "does not parse" in text or "preprocessing failed" in text else "refused"
        return {"unit": name, "class": cls, "message": text[:600], "key": _normalise(text.splitlines()[-1] if cls == "refused" else text.splitlines()[1] if len(text.splitlines()) > 1 else text)}
    except CompileError as exc:
        lines = [l for l in str(exc).splitlines() if "error:" in l]
        first = lines[0] if lines else str(exc)
        return {"unit": name, "class": "metal_error", "message": "\n".join(lines[:6])[:800],
                "key": _normalise(first)}
    except Exception as exc:  # noqa: BLE001
        return {"unit": name, "class": "crash", "message": f"{type(exc).__name__}: {exc}"[:600],
                "key": f"{type(exc).__name__}: {_normalise(str(exc))}"}
    return {"unit": name, "class": "compiled", "kernels": sorted(module.unit.kernels),
            "seconds": round(time.perf_counter() - t0, 2)}


def _run_isolated(name: str, retain: bool, timeout: float, translate_only: bool = False) -> dict:
    """One unit in its own interpreter: a unit that hangs the translator or
    the Metal compiler is recorded as ``timeout`` instead of stalling."""
    import subprocess
    cmd = ([sys.executable, str(Path(__file__).resolve()), "--one", name] + (["--retain"] if retain else [])
           + (["--translate-only"] if translate_only else []))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return {"unit": name, "class": "timeout", "key": f"no result in {timeout:.0f} s"}
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    if not lines:
        tail = (proc.stderr or proc.stdout)[-600:]
        return {"unit": name, "class": "crash", "message": tail, "key": _normalise(tail.strip().splitlines()[-1] if tail.strip() else "no output")}
    return json.loads(lines[-1])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--retain", action="store_true", help="translate as GPUWM_WRF_EXACT=1")
    parser.add_argument("--only", nargs="*", help="unit names to check")
    parser.add_argument("--one", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per unit")
    parser.add_argument("--out", type=Path, default=Path("/tmp/metal-census.jsonl"),
                        help="JSON lines, appended as units finish; finished units are skipped on rerun")
    parser.add_argument("--fresh", action="store_true", help="ignore results already in --out")
    parser.add_argument("--translate-only", action="store_true",
                        help="translate to MSL and stop: no Metal device is touched")
    parser.add_argument("--gate", action="store_true",
                        help="exit 1 if any unit is refused, metal_error, crash or timeout")
    args = parser.parse_args(argv)
    if args.one:
        print(json.dumps(census_one(args.one, args.retain, args.translate_only)))
        return 0
    done = {}
    stamp = fingerprint()
    current = set(units())
    stale = set()
    if args.out.exists() and not args.fresh:
        for line in args.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if (r.get("retain", False) != args.retain
                        or r.get("translate_only", False) != args.translate_only
                        or r["unit"] not in current):
                    continue
                if r.get("fingerprint") != stamp:
                    stale.add(r["unit"])
                    done.pop(r["unit"], None)
                    continue
                done[r["unit"]] = r
    names = [n for n in (args.only or units()) if args.only or n not in done]
    print(f"toolchain {stamp}: {len(done)} cached, {len(names)} to run"
          + (f" ({len(stale - set(done))} stale results ignored)" if stale else ""), flush=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.jobs) as pool, args.out.open("a") as sink:
        futures = {pool.submit(_run_isolated, n, args.retain, args.timeout, args.translate_only): n
                   for n in names}
        for f in as_completed(futures):
            r = f.result()
            r["retain"] = args.retain
            r["translate_only"] = args.translate_only
            r["fingerprint"] = stamp
            done[r["unit"]] = r
            sink.write(json.dumps(r) + "\n")
            sink.flush()
            print(f"{r['unit']:40s} {r['class']}", flush=True)
    results = sorted((r for r in done.values() if not args.only or r["unit"] in args.only),
                     key=lambda r: r["unit"])
    counts = collections.Counter(r["class"] for r in results)
    print(f"\n{len(results)} units: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    standalone = [r for r in results if r["class"] != "source_error"]
    print(f"standalone units: {counts['compiled']} of {len(standalone)} compile"
          + (f" ({counts['timeout']} timed out: rerun with --only)" if counts["timeout"] else "")
          + f"; {counts['source_error']} are composed by their own loaders")
    groups = collections.defaultdict(list)
    for r in results:
        if r["class"] not in ("compiled", "translated"):
            groups[(r["class"], r.get("key", r.get("message", "")[:120]))].append(r["unit"])
    for (cls, key), members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"{len(members):4d}  {cls:12s} {key}\n        {', '.join(members[:8])}{' ...' if len(members) > 8 else ''}")
    failing = [r["unit"] for r in results if r["class"] in GATE_FAILURES]
    if args.gate and failing:
        print(f"\ngate: {len(failing)} unit(s) in {', '.join(sorted(GATE_FAILURES))}: {', '.join(failing)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
