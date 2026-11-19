"""No forecast path imports woof.da.var (docs/dev/WRFDA-3DVAR-PLAN.md section 4.11).

3D-Var is a separate door: only ``woof/da/var`` itself and the ``woof var``
command module may import it.  A static scan of every module under ``woof/``
holds that, so a default forecast cannot load the package, launch its
kernels or change a word because of it.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "woof"
ALLOWED_PREFIXES = (ROOT / "da" / "var",)
ALLOWED_FILES = {ROOT / "var_cli.py"}


def _is_var(name: str) -> bool:
    return name == "woof.da.var" or name.startswith("woof.da.var.")


def _package(path: Path) -> list[str]:
    """The dotted package a module's relative imports resolve against."""
    parts = list(path.relative_to(ROOT.parent).with_suffix("").parts)
    return parts[:-1]  # a package's __init__ and a plain module alike drop the last part


def _imports_var(tree: ast.AST, path: Path | None = None) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_is_var(a.name) for a in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                if path is None:
                    continue
                base = _package(path)
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                mod = ".".join(base + ([mod] if mod else []))
            if _is_var(mod) or (mod == "woof.da" and any(a.name == "var" for a in node.names)):
                lines.append(node.lineno)
    return lines


def test_relative_imports_are_resolved():
    da = ROOT / "da" / "letkf.py"
    core = ROOT / "core" / "x.py"
    init = ROOT / "da" / "__init__.py"
    for src, path in [("from .var import be_dat", da), ("from . import var", da),
                      ("from ..da.var import reference", core), ("from .var.be_dat import read_be_dat", init),
                      ("from . import var", init)]:
        assert _imports_var(ast.parse(src), path) == [1], src
    for src, path in [("from . import letkf", da), ("from ..da import letkf", core), ("from .var import x", core)]:
        assert _imports_var(ast.parse(src), path) == [], src


def test_no_forecast_module_imports_woof_da_var():
    offenders = []
    for path in sorted(ROOT.rglob("*.py")):
        if any(p in path.parents for p in ALLOWED_PREFIXES) or path in ALLOWED_FILES:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for line in _imports_var(tree, path):
            offenders.append(f"{path.relative_to(ROOT.parent)}:{line}")
    assert offenders == [], f"modules outside the woof var door import woof.da.var: {offenders}"


def test_woof_da_does_not_load_var():
    """``import woof.da`` (the LETKF door's package) leaves woof.da.var unloaded."""
    import subprocess
    import sys

    code = "import sys, woof.da; print('woof.da.var' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
