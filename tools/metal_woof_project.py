"""Rewrite pyproject.toml for the metal-woof distribution.

metal-woof is this source built for Apple silicon: the same ``woof``
package, a macOS arm64 platform wheel carrying the macOS bridges, and the
MLX backend's requirements in its base dependencies so that a plain
``pip install metal-woof`` runs on Metal.  Run by
.github/workflows/publish-metal.yml on a throwaway checkout before
``python -m build``; recast-woof's own pyproject is never committed changed.

Usage::

    python tools/metal_woof_project.py --version 1.1.3.1 [pyproject.toml]

The rewrite is textual, so comments and layout survive, and it is checked
with tomllib afterwards: a row that did not move fails the build here
rather than publishing a metal-woof that depends on recast-woof.
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

SOURCE_NAME = "recast-woof"
METAL_NAME = "metal-woof"
DESCRIPTION = ("Recast WOOF for Apple silicon: GPU-native regional weather "
               "forecasting with WRF-class physics on Metal, its CUDA kernels "
               "translated to Metal (an unofficial build of recast-woof)")
METAL_EXTRA = "gpu-mlx"


def rewrite(text: str, version: str) -> str:
    project = tomllib.loads(text)["project"]
    if project["name"] != SOURCE_NAME:
        raise SystemExit(f"expected [project].name = {SOURCE_NAME!r}, "
                         f"got {project['name']!r}")
    extra = project["optional-dependencies"][METAL_EXTRA]

    def once(pattern: str, replacement: str, source: str) -> str:
        result, count = re.subn(pattern, replacement, source, count=1,
                                flags=re.MULTILINE)
        if count != 1:
            raise SystemExit(f"pattern not found exactly once: {pattern}")
        return result

    text = once(r'^name = "recast-woof"$', f'name = "{METAL_NAME}"', text)
    text = once(r'^version = "[^"]+"$', f'version = "{version}"', text)
    text = once(r'^description = "[^"]*"$', f'description = "{DESCRIPTION}"', text)
    # The extras that name the distribution itself (all-mlx and friends).
    text = text.replace(f'"{SOURCE_NAME}[', f'"{METAL_NAME}[')
    # The MLX backend's requirements become base dependencies.
    added = ", ".join(f'"{requirement}"' for requirement in extra)
    text = once(r'^(dependencies = \[(?:[^\]]|\n)*?)\]',
                lambda m: f"{m.group(1)}, {added}]", text)  # type: ignore[arg-type]

    rewritten = tomllib.loads(text)["project"]
    assert rewritten["name"] == METAL_NAME
    assert rewritten["version"] == version
    for requirement in extra:
        assert requirement in rewritten["dependencies"], requirement
    for name, requirements in rewritten["optional-dependencies"].items():
        for requirement in requirements:
            assert not requirement.startswith(f"{SOURCE_NAME}["), (name, requirement)
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True,
                        help="the metal-woof release version, e.g. 1.1.3.1")
    parser.add_argument("pyproject", nargs="?", type=Path,
                        default=Path("pyproject.toml"))
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d+(\.\d+)+", args.version):
        raise SystemExit(f"version {args.version!r} is not a release number")
    args.pyproject.write_text(rewrite(args.pyproject.read_text(encoding="utf-8"),
                                      args.version), encoding="utf-8")
    print(f"{args.pyproject}: {METAL_NAME} {args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
