"""Build and load ``_woof_metal_launch``, the in-place Metal launcher.

The extension links against the installed MLX and must share its nanobind
internals ABI, so it is built on the machine that runs it, once per
(source, MLX, nanobind, Python) combination, into a persistent cache:
``$WOOF_METAL_EXT_DIR``, else ``$XDG_CACHE_HOME/woof/metal-ext`` (default
``~/.cache/woof/metal-ext``).  The build needs CMake and the Xcode Command
Line Tools' clang; it does not need the offline ``metal`` compiler, because
every kernel compiles at run time through Metal's own source compiler.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = (HERE / "launch.cpp", HERE / "CMakeLists.txt")
MODULE = "_woof_metal_launch"

#: Environment override for the build cache directory.
BUILD_DIR_ENV = "WOOF_METAL_EXT_DIR"


def cache_root() -> Path:
    env = os.environ.get(BUILD_DIR_ENV)
    if env:
        return Path(env).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "woof" / "metal-ext").expanduser()


def build_key() -> str:
    """Everything the binary depends on: our sources and the ABI peers."""
    import mlx.core as mx
    import nanobind
    digest = hashlib.sha256()
    for path in SOURCES:
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    for part in (mx.__version__, nanobind.__version__,
                 sysconfig.get_config_var("EXT_SUFFIX") or "", sys.version):
        digest.update(part.encode() + b"\0")
    return digest.hexdigest()[:20]


def _built_module(directory: Path) -> Path | None:
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    candidate = directory / f"{MODULE}{suffix}"
    return candidate if candidate.is_file() else None


def build(directory: Path) -> Path:
    """Configure and build into ``directory``; returns the module path."""
    import mlx
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("building woof's Metal launcher needs cmake on PATH")
    mlx_root = Path(mlx.__path__[0])
    work = directory / "build"
    work.mkdir(parents=True, exist_ok=True)
    configure = [
        cmake, "-S", str(HERE), "-B", str(work),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DMLX_ROOT={mlx_root}",
        f"-DCMAKE_PREFIX_PATH={mlx_root}",
        f"-DWOOF_EXT_OUTPUT_DIR={directory}",
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0",
    ]
    if shutil.which("ninja"):
        configure += ["-G", "Ninja"]
    for command in (configure, [cmake, "--build", str(work), "--config", "Release"]):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"building {MODULE} failed ({' '.join(command[:2])} ...):\n"
                + result.stdout[-4000:] + result.stderr[-4000:])
    built = _built_module(directory)
    if built is None:
        raise RuntimeError(f"{MODULE} build reported success but left no module in {directory}")
    return built


@lru_cache(maxsize=1)
def load():
    """The ``_woof_metal_launch`` module, building it on first use."""
    if sys.platform != "darwin":
        raise RuntimeError("woof's Metal launcher runs on macOS only")
    import mlx.core  # noqa: F401  -- registers mx.array before our module loads
    directory = cache_root() / build_key()
    path = _built_module(directory)
    if path is None:
        directory.mkdir(parents=True, exist_ok=True)
        path = build(directory)
    spec = importlib.util.spec_from_file_location(MODULE, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
