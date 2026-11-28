"""Which oracle lanes carry over to WRF v4.8.1 unchanged, routine by routine.

A lane pinned to WRF 4.6.1, 4.7.1 or 4.8.0 stays word-exact against 4.8.1
when no WRF routine it exercises changed between its pin and 4.8.1.
Byte-identical files settle that at once.  For a file that differs, this
tool splits both versions into their SUBROUTINE / FUNCTION blocks and lists
the routines whose text changed (plus a flag for any change outside the
routines, a module's declarations).  A lane exercises a routine when its
own files (driver, capture, replay, README) name it, and every routine that
routine can call, transitively, over the WRF tree.  A lane whose
exercised set meets no changed routine is re-pinned to 4.8.1 by this
ledger, with its fixtures as they are.  Any other lane is held for the phase
that ports the change.

    python -m tools.wrf481_source.lanes build --tree 4.8.1=WRF481 \\
        --tree 4.8.0=WRF480 --tree 4.7.1=WRF471 --tree 4.6.1=WRF461

``build`` needs the four source trees (on the oracle host) and writes
``lanes-v4.8.1.json`` beside this file; the tests check the committed
ledger against the repository without the trees.  Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from tools.wrf481_source.manifest import CHANGES, EXCLUDED, ROOT, cited_paths

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "lanes-v4.8.1.json"

#: The F1 lanes (dycore, advection, diffusion, damping, nesting, DFI,
#: bookkeeping) of WRF-4.8.0-PLAN.md, with the release each is pinned to.
F1_LANES = {
    "tools/advect_wrf471_oracle": "4.7.1",
    "tools/bigstep_wrf471_oracle": "4.7.1",
    "tools/deng_wrf471": "4.7.1",
    "tools/dfi_wrf461_oracle": "4.6.1",
    "tools/dfi_wrf471_oracle": "4.7.1",
    "tools/diffopt1_wrf461_oracle": "4.6.1",
    "tools/farms_wrf471_oracle": "4.7.1",
    "tools/madwrf_wrf471_oracle": "4.7.1",
    "tools/ndown_wrf471_oracle": "4.7.1",
    "tools/partial_nest_wrf471_oracle": "4.7.1",
    "tools/relaxation_wrf461_oracle": "4.6.1",
    "tools/relaxation_wrf471_oracle": "4.7.1",
    "tools/sbu_ylin_wrf461_oracle": "4.6.1",
    "tools/sfc_irr_seaice_wrf471": "4.7.1",
    "tools/shcu_deng_wrf471_oracle": "4.7.1",
    "tools/smag2d_wrf461_oracle": "4.6.1",
    "tools/smallstep_wrf471_oracle": "4.7.1",
    "tools/sms3dtke_wrf471_oracle": "4.7.1",
    "tools/tke_km2_wrf461_oracle": "4.6.1",
    "tools/tracer_wrf461_oracle": "4.6.1",
    "tools/trajectory_wrf461_oracle": "4.6.1",
    "tools/uh_wrf461_oracle": "4.6.1",
    "tools/vert_nest_wrf471_oracle": "4.7.1",
    "tools/wrf461_km3_oracle": "4.6.1",
    "tools/wrf461_nest_initial": "4.6.1",
    "tools/wrf480_capture": "4.8.0",
    "tools/wrf480_dycore_confirm": "4.8.0",
    "tools/wrf480_gate": "4.8.0",
    "tools/advect_wrf_exact": "4.7.1",
    "tools/advect_wrf_legacy_oracle": "4.7.1",
    "tools/fork_forms_oracle": "4.6.1",
    "tools/generate_grims_tables.py": "4.6.1",
    "tools/hrrr_single_domain_benchmark.py": "4.6.1",
    "tools/ieva_wrf_oracle": "4.7.1",
    "tools/port_dyn_options": "4.7.1",
    "tools/port_gecros": "4.7.1",
    "tools/wrf_diffopt1_oracle": "4.7.1",
    "tools/wrf_diffusion_oracle": "4.7.1",
    "tools/wrf_em_les_oracle": "4.6.1",
    "tools/wrf_eta_oracle.py": "4.6.1",
    "tools/wrf_exact_localize": "4.6.1",
    "tools/wrf_instrumented": "4.6.1",
}

#: Where routines are looked up for the call closure (Noah-MP v5 is never
#: read: tools/noahmp_options_wrf461_oracle/source_pin.json).
SCANNED = ("dyn_em", "share", "frame", "phys", "main")
_START = re.compile(
    r"^\s*(?:(?:RECURSIVE|PURE|ELEMENTAL|IMPURE)\s+)*"
    r"(?:(?:REAL|INTEGER|LOGICAL|COMPLEX|DOUBLE\s+PRECISION|CHARACTER)"
    r"(?:\s*\([^)]*\)|\*\d+)?\s+)?(SUBROUTINE|FUNCTION)\s+(\w+)", re.I)
_END = re.compile(r"^\s*END\s*(SUBROUTINE|FUNCTION)\b", re.I)
_WORD = re.compile(r"[A-Za-z_]\w*")
_SOURCE_SUFFIXES = (".F", ".F90", ".f90", ".inc", ".h")


def _code(line: str) -> str:
    return line.split("!")[0]


def blocks(text: str, kinds: dict | None = None) -> tuple[dict[str, str], str]:
    """``({routine: its text}, the text outside every routine)``; ``kinds``,
    when given, collects routine -> "subroutine" / "function"."""
    found: dict[str, str] = {}
    stack: list[tuple[str, list[str]]] = []
    outside: list[str] = []
    for line in text.splitlines():
        code = _code(line)
        match = _START.match(code)
        if match and not re.match(r"^\s*END", code, re.I):
            stack.append((match.group(2).lower(), []))
            if kinds is not None:
                kinds[match.group(2).lower()] = match.group(1).lower()
        if stack:
            for _name, buf in stack:
                buf.append(line)
        else:
            outside.append(line)
        if _END.match(code) and stack:
            name, buf = stack.pop()
            found[name] = found.get(name, "") + "\n".join(buf)
    return found, "\n".join(outside)


def _words(text: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(text)}


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def lane_words(lane: str) -> set[str]:
    """Every identifier the lane's tracked text files contain."""
    listed = subprocess.run(["git", "ls-files", "-z", "--", lane], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    words: set[str] = set()
    for rel in filter(None, listed.stdout.split("\0")):
        path = ROOT / rel
        try:
            text = path.read_bytes().decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        words |= _words(text)
    return words


def _code_only(text: str) -> str:
    """Fortran code without comments or blank lines (what reaches words)."""
    lines = (_code(line).rstrip() for line in text.splitlines())
    return "\n".join(line for line in lines if line.strip())


_CALL = re.compile(r"\bCALL\s+(\w+)", re.I)
_USE = re.compile(r"^\s*USE\s+(\w+)", re.I | re.M)
_MODULE = re.compile(r"^\s*MODULE\s+(?!PROCEDURE\b)(\w+)", re.I | re.M)


class TreeIndex:
    """The routines of one WRF tree over :data:`SCANNED`.

    A routine is keyed ``(file, name)``: WRF defines ``gamma``, ``rslf``,
    ``tridi2n`` and many more in several schemes, and only the definition a
    caller resolves to is exercised.  ``defs``: key -> its code (comments
    aside); ``outside``: file -> its module-level code; ``graph``: key ->
    the keys it can call.  ``CALL x`` resolves to ``x`` in the same file,
    else to ``x`` in a module the file USEs, else to every ``x``; a word
    resolves only to a FUNCTION of the same file or of a USEd module (a
    bare identifier equal to some routine's name is not a call: ``beta``,
    ``gamma``, ``driver`` are variables across WRF).
    """

    def __init__(self, tree: Path):
        self.defs: dict[tuple[str, str], str] = {}
        self.outside: dict[str, str] = {}
        kind: dict[tuple[str, str], str] = {}
        modules: dict[str, set[str]] = {}
        uses: dict[str, set[str]] = {}
        for top in SCANNED:
            for path in sorted((tree / top).rglob("*")):
                rel = path.relative_to(tree).as_posix()
                if (not path.is_file() or path.suffix not in _SOURCE_SUFFIXES
                        or rel.startswith(EXCLUDED)):
                    continue
                text = _read(path)
                kinds: dict[str, str] = {}
                found, outside = blocks(text, kinds)
                for name, body in found.items():
                    self.defs[rel, name] = _code_only(body)
                    kind[rel, name] = kinds[name]
                self.outside[rel] = _code_only(outside)
                stripped = _code_only(text)
                for module in _MODULE.findall(stripped):
                    modules.setdefault(module.lower(), set()).add(rel)
                uses[rel] = {m.lower() for m in _USE.findall(stripped)}
        by_name: dict[str, list[tuple[str, str]]] = {}
        for key in self.defs:
            by_name.setdefault(key[1], []).append(key)
        self.graph: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for rel in self.outside:
            seen = {rel} | {f for m in uses[rel] for f in modules.get(m, ())}
            functions = {key[1]: key for key in self.defs
                         if key[0] in seen and kind[key] == "function"}
            for key, body in self.defs.items():
                if key[0] != rel:
                    continue
                edges = set()
                for callee in {c.lower() for c in _CALL.findall(body)}:
                    keys = by_name.get(callee, [])
                    edges |= ({k for k in keys if k[0] == rel}
                              or {k for k in keys if k[0] in seen} or set(keys))
                edges |= {functions[w] for w in _words(body) & functions.keys()}
                self.graph[key] = edges - {key}


def changed_routines(old: TreeIndex, new: TreeIndex) -> dict[tuple[str, str], str]:
    """key -> why the routine does not carry its words from ``old`` to ``new``.

    A routine's own code changed, it went, or it came (a 4.8.1 routine a
    caller now reaches); or its file's module-level code changed in a word
    the routine uses (a declaration, a USE list).  A module-level change in
    a word no routine of the file uses (a renamed USE-only import the file
    never references, a preprocessor guard) does not reach the routines.
    """
    why = {}
    for key, body in old.defs.items():
        if key not in new.defs:
            why[key] = "removed"
        elif new.defs[key] != body:
            why[key] = "changed"
    for key in new.defs.keys() - old.defs.keys():
        why[key] = "added"
    for rel, outside in old.outside.items():
        delta = _words(outside) ^ _words(new.outside.get(rel, ""))
        if not delta:
            continue
        for key, body in old.defs.items():
            used = _words(body) & delta
            if key[0] == rel and used:
                why.setdefault(key, f"module-level code of {rel} changed "
                                    f"({', '.join(sorted(used))})")
    return why


def closure(start: set, graph: dict) -> set:
    seen, todo = set(), [name for name in start if name in graph]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        todo.extend(graph.get(name, ()))
    return seen


def file_delta(old: Path, new: Path) -> dict:
    """Routine-level difference of one file between two releases."""
    a, a_out = blocks(_read(old)) if old.is_file() else ({}, "")
    b, b_out = blocks(_read(new)) if new.is_file() else ({}, "")
    code = lambda table: {n: _code_only(t) for n, t in table.items()}  # noqa: E731
    a, b = code(a), code(b)
    return {"changed": sorted(n for n in a.keys() & b.keys() if a[n] != b[n]),
            "added": sorted(b.keys() - a.keys()),
            "removed": sorted(a.keys() - b.keys()),
            "module_level_changed": _code_only(a_out) != _code_only(b_out)}


#: Lanes whose referee is a whole wrf.exe run (not a routine driven on its
#: own), so they exercise the model's top: the RK step and everything in it.
WHOLE_MODEL = {
    "tools/wrf480_gate": "compares a whole native wrf.exe forecast",
    "tools/wrf480_dycore_confirm": "replays the dycore of a native wrf.exe run",
    "tools/wrf461_nest_initial": "compares a native wrf.exe nest history frame",
}
#: Cited files the path scan misses (a leading slash in a json audit).
EXTRA_CITED = {"tools/port_gecros": ("phys/module_sf_gecros.F",)}

#: Lanes held whatever the routine comparison says.
FORCED_HOLD = {
    "tools/wrf480_capture": "its dumps come from a solve_em hook instrumented "
                            "into a 4.8.0 tree (instrument.py, targets-480.json) "
                            "of a native 4.8.0 run: relabelling would claim a "
                            "4.8.1 capture; rebuild it with the dycore ports",
}
NO_FILES_HOLD = ("cites no WRF source path the ledger can compare, so nothing "
                 "shows its words carry to 4.8.1")

WHOLE_MODEL_SEEDS = {"wrf_init", "wrf_run", "integrate", "solve_interface",
                     "start_domain", "med_initialdata_input"}


def build(trees: dict[str, Path]) -> dict:
    index = {release: TreeIndex(tree) for release, tree in trees.items()}
    graphs = {release: index[release].graph for release in trees}
    changed = {release: changed_routines(index[release], index["4.8.1"])
               for release in trees if release != "4.8.1"}
    lanes = {}
    for lane, pin in F1_LANES.items():
        if not (ROOT / lane).exists():
            continue
        words = lane_words(lane)
        old_tree, new_tree = trees[pin], trees["4.8.1"]
        files = {}
        for rel in [*cited_paths((lane,)), *EXTRA_CITED.get(lane, ())]:
            old, new = old_tree / rel, new_tree / rel
            sha_old, sha_new = _sha(old), _sha(new)
            if sha_old is None and sha_new is None:
                continue
            row = {"sha256_pin": sha_old, "sha256_v4.8.1": sha_new}
            if sha_old != sha_new:
                row["delta"] = (file_delta(old, new) if rel.endswith(_SOURCE_SUFFIXES)
                                else "not Fortran: compared as bytes only")
                row["changes_481"] = CHANGES.get(rel, [])
            files[rel] = row
        # The routines the lane names, among those of the WRF files it
        # pins: a word is a routine of the lane only inside its own scope.
        scope: set[tuple[str, str]] = set()
        for rel in files:
            for release in (pin, "4.8.1"):
                scope |= {key for key in index[release].defs if key[0] == rel}
        start = {key for key in scope if key[1] in words}
        if lane in WHOLE_MODEL:
            for release in (pin, "4.8.1"):
                start |= {key for key in index[release].defs
                          if key[1] in WHOLE_MODEL_SEEDS}
        named = {name for _rel, name in start}
        exercised = (closure(start, graphs[pin]) | closure(start, graphs["4.8.1"]))
        hits = {f"{rel}:{name}": changed[pin][rel, name]
                for rel, name in sorted(exercised & set(changed[pin]))}
        hold = FORCED_HOLD.get(lane) or (None if files else NO_FILES_HOLD)
        lanes[lane] = {"pin": pin,
                       "verdict": "held" if hits or hold else "re-pinned 4.8.1",
                       "held_because": hold,
                       "whole_model": WHOLE_MODEL.get(lane),
                       "named_routines": sorted(named),
                       "exercised_changed_routines": hits,
                       "files": files}
    return {"schema": "woof.wrf481-lanes.v1",
            "meaning": ("A lane is re-pinned to WRF 4.8.1 when no routine it "
                        "names, nor any routine those can call, has different "
                        "code (comments aside) at 4.8.1 than at its pin; its "
                        "fixtures are kept.  Any other lane is held for the "
                        "phase that ports the change."),
            "lanes": lanes}


def load() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def repinned() -> list[str]:
    """The lanes the ledger re-pins to WRF 4.8.1, fixtures as they are."""
    return sorted(lane for lane, row in load()["lanes"].items()
                  if row["verdict"] == "re-pinned 4.8.1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tree", action="append", required=True,
                   help="RELEASE=PATH, for 4.8.1, 4.8.0, 4.7.1 and 4.6.1")
    b.add_argument("--out", type=Path, default=LEDGER)
    sub.add_parser("show")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        trees = dict(item.split("=", 1) for item in args.tree)
        trees = {release: Path(path) for release, path in trees.items()}
        table = build(trees)
        args.out.write_text(json.dumps(table, indent=1) + "\n",
                            encoding="utf-8", newline="\n")
        repinned = sum(row["verdict"] == "re-pinned 4.8.1"
                       for row in table["lanes"].values())
        print(f"{len(table['lanes'])} lanes, {repinned} re-pinned 4.8.1 -> {args.out}")
        return 0
    for lane, row in load()["lanes"].items():
        why = ", ".join(row["exercised_changed_routines"]) or (row.get("held_because") or "")
        print(f"{lane}: {row['verdict']}" + (f" ({why})" if why else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
