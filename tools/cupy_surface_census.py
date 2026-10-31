"""Census: the CuPy API woof uses, against what the facade implements.

Every ``.py`` under ``woof/`` is parsed (never imported).  Each name bound
to CuPy -- ``import cupy``, ``import cupy as cp``, ``import cupyx``,
``from cupy import cuda`` -- is followed through attribute chains
(``cp.cuda.Device``, ``cp.float32``, ``cupyx.scipy.ndimage``).  Each chain
is resolved on :mod:`woof.backend.facade`, the module ``import cupy``
returns under ``WOOF_BACKEND=mlx|cpu``; a chain the facade cannot resolve is
a gap the backend would hit by name (``BackendUnsupported``) at its first
use.

Chains are resolved only as deep as the facade's objects are modules or
classes: ``cp.cuda.Device(0).use`` stops at ``cp.cuda.Device``, because what
follows is a call's result.

``--gate`` exits non-zero when a gap is not in ``--allow`` (default
tools/cupy_surface_allow.txt, one chain per line with a ``#`` reason):
new code that reaches for an unimplemented CuPy attribute fails here, not
on a Mac at run time.

Usage::

    pixi run -e mlx python tools/cupy_surface_census.py [--files] [--gate]
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW = ROOT / "tools" / "cupy_surface_allow.txt"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bindings(tree: ast.AST) -> dict[str, str]:
    """Local name -> the CuPy path it is bound to, for one module."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "cupy" or a.name.startswith(("cupy.", "cupyx")):
                    if a.asname:
                        out[a.asname] = a.name
                    else:
                        out[a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0 \
                and (node.module == "cupy" or node.module.startswith(("cupy.", "cupyx"))):
            for a in node.names:
                if a.name != "*":
                    out[a.asname or a.name] = f"{node.module}.{a.name}"
    return out


def _chain(node: ast.AST) -> list[str] | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return parts[::-1]
    return None


def uses(root: Path = ROOT / "woof") -> dict[str, set[str]]:
    """CuPy path -> the files that use it."""
    found: dict[str, set[str]] = collections.defaultdict(set)
    for path in sorted(root.rglob("*.py")):
        if "backend" in path.relative_to(root).parts[:1]:
            continue                        # the facade itself
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        names = _bindings(tree)
        if not names:
            continue
        _mark_parents(tree)
        rel = str(path.relative_to(ROOT))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and not isinstance(getattr(node, "_parent", None), ast.Attribute):
                chain = _chain(node)
                if chain and chain[0] in names:
                    found[".".join([names[chain[0]]] + chain[1:])].add(rel)
        for local, target in names.items():
            if target != target.split(".")[0]:
                found[target].add(rel)
    return found


def _mark_parents(tree):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node


def resolve(path: str):
    """(ok, depth reached, reason) for one CuPy path on the facade."""
    import importlib
    import inspect
    from woof.backend import facade
    parts = path.split(".")
    obj = facade if parts[0] == "cupy" else facade.cupyx if parts[0] == "cupyx" else None
    if obj is None:
        return False, 0, f"unknown root {parts[0]!r}"
    for depth, name in enumerate(parts[1:], start=1):
        if not (inspect.ismodule(obj) or inspect.isclass(obj) or isinstance(obj, type(facade))):
            return True, depth, "stops at a value"
        try:
            obj = getattr(obj, name)
        except AttributeError as exc:
            return False, depth, str(exc)
        except Exception as exc:  # noqa: BLE001 -- a property that needs a device, etc.
            return True, depth, f"resolves lazily ({type(exc).__name__})"
    return True, len(parts) - 1, "ok"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--files", action="store_true", help="list the files using each gap")
    ap.add_argument("--json", type=Path, help="write the census as JSON")
    ap.add_argument("--gate", action="store_true", help="exit 1 on a gap not in --allow")
    ap.add_argument("--allow", type=Path, default=ALLOW)
    args = ap.parse_args(argv)

    found = uses()
    rows = []
    for path, files in sorted(found.items()):
        ok, depth, reason = resolve(path)
        rows.append({"path": path, "files": sorted(files), "ok": ok, "reason": reason})
    gaps = [r for r in rows if not r["ok"]]
    allowed = set()
    if args.allow.exists():
        for line in args.allow.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                allowed.add(line)
    new = [r for r in gaps if r["path"] not in allowed]
    print(f"{len(rows)} CuPy paths used in {len({f for r in rows for f in r['files']})} files; "
          f"{len(rows) - len(gaps)} resolve on the facade, {len(gaps)} gaps "
          f"({len(gaps) - len(new)} allowed)")
    for r in sorted(gaps, key=lambda r: -len(r["files"])):
        mark = " " if r["path"] in allowed else "!"
        print(f" {mark} {r['path']:50s} {len(r['files']):4d} file(s)")
        if args.files:
            for f in r["files"][:12]:
                print(f"        {f}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    return 1 if (args.gate and new) else 0


if __name__ == "__main__":
    sys.exit(main())
