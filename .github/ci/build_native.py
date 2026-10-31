"""Build the Rust bridges the tests drive, and pack them for the test legs.

    python .github/ci/build_native.py list            # the workspaces, in build order
    python .github/ci/build_native.py build [WS ...]  # cargo build --release --locked --offline
    python .github/ci/build_native.py pack OUT.tar    # every built executable and shared library
    python .github/ci/build_native.py unpack IN.tar   # into the checkout's tools/*/target/release

The workspace list is read from the ``build-native`` pixi task in
pyproject.toml, so CI builds what ``pixi run build-native`` builds and the two
cannot drift.  Every crate is vendored (``.cargo/config.toml`` in each
workspace), so the build needs no network.

The tests find these binaries where a source checkout keeps them,
``tools/<workspace>/target/release`` (woof.rustwx, woof.bridges), so the pack
keeps those paths.
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
import tarfile
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def workspaces() -> list[str]:
    task = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "tool"]["pixi"]["feature"]["base"]["tasks"]["build-native"]
    names = [Path(p).name for p in re.findall(r"cd ((?:tools/|\.\./)[\w.-]+)", task)]
    if not names or len(names) != len(set(names)):
        raise SystemExit(f"cannot read the workspace list from the build-native task: {names}")
    return names


def build(names: list[str]) -> int:
    env = dict(os.environ)
    if sys.platform.startswith("linux"):
        # The build-native task's own compiler choice (pyproject.toml).
        env.update(CC="gcc", CXX="g++", CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="gcc")
    for name in names:
        start = time.monotonic()
        print(f"::group::cargo build {name}", flush=True)
        rc = subprocess.call(["cargo", "build", "--release", "--locked", "--offline"],
                             cwd=ROOT / "tools" / name, env=env)
        print("::endgroup::", flush=True)
        print(f"{name}: exit {rc} in {time.monotonic() - start:.0f} s", flush=True)
        if rc:
            return rc
    return 0


def built_files() -> list[Path]:
    out = []
    for name in workspaces():
        release = ROOT / "tools" / name / "target" / "release"
        if not release.is_dir():
            continue
        for path in sorted(release.iterdir()):
            if not path.is_file() or path.suffix in (".d", ".rlib", ".rmeta"):
                continue
            if path.suffix in (".so", ".dylib") or path.stat().st_mode & stat.S_IXUSR:
                out.append(path)
    return out


def main(argv: list[str]) -> int:
    if argv == ["list"]:
        print("\n".join(workspaces()))
        return 0
    if argv[:1] == ["build"]:
        return build(argv[1:] or workspaces())
    if argv[:1] == ["pack"] and len(argv) == 2:
        files = built_files()
        if not files:
            raise SystemExit("nothing built to pack")
        with tarfile.open(argv[1], "w") as tar:
            for path in files:
                tar.add(path, arcname=path.relative_to(ROOT).as_posix())
        print(f"packed {len(files)} files into {argv[1]}")
        return 0
    if argv[:1] == ["unpack"] and len(argv) == 2:
        with tarfile.open(argv[1]) as tar:
            tar.extractall(ROOT, filter="tar")
            print(f"unpacked {len(tar.getnames())} files")
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
