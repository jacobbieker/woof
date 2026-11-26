"""A function-level import of a module-level name makes that name local to
the whole function, so a branch that did not run the import fails with
UnboundLocalError.  physics_setup_identity did this with ``hashlib`` (bound
only for mp_physics=40), which broke every checkpointed run carrying an
eclipse."""

import ast
from pathlib import Path

import woof.io.restart as restart


def _bound_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    # ``import a.b`` binds ``a``; ``from a import b`` and any alias bind as written.
    return [alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
            for alias in node.names]


def _shadowing_imports(source: str) -> list[tuple[str, int, str]]:
    tree = ast.parse(source)
    module_names = {name for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                    for name in _bound_names(node)}
    found = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        stack = list(function.body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                found += [(function.name, node.lineno, name)
                          for name in _bound_names(node) if name in module_names]
            stack.extend(ast.iter_child_nodes(node))
    return found


def test_restart_functions_do_not_rebind_module_imports():
    assert _shadowing_imports(Path(restart.__file__).read_text()) == []


def test_the_check_catches_a_branch_local_rebind():
    source = "import hashlib\ndef f(a):\n    if a:\n        import hashlib\n    return hashlib\n"
    assert _shadowing_imports(source) == [("f", 4, "hashlib")]


def test_the_check_catches_a_dotted_rebind():
    source = "import os.path\ndef f(a):\n    if a:\n        import os.path\n    return os\n"
    assert _shadowing_imports(source) == [("f", 4, "os")]
