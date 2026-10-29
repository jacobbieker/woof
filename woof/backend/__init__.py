"""Which array and kernel backend a woof process runs on.

``cuda``  CuPy and NVRTC: the production backend, untouched by this package.
``mlx``   Apple-silicon GPUs.  ``import cupy`` resolves to
          :mod:`woof.backend.facade`, a CuPy-compatible module over
          ``mlx.core``; kernels are the same CUDA sources translated to Metal
          (:mod:`woof.backend.translate`) and launched in place
          (:mod:`woof.backend.metal`).
``cpu``   the same facade over NumPy, kernels compiled as host C++.

Selection: ``$WOOF_BACKEND`` when set, else ``cuda`` when CuPy is importable,
else ``mlx`` on darwin-arm64 with MLX installed, else ``cpu``.  The choice is
made once per process by :func:`install`, before any ``woof.core`` module
imports cupy; it refuses to swap backends under a process that already
imported the real CuPy, because mixing device and facade arrays cannot be
made safe.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys

BACKENDS = ("cuda", "mlx", "cpu")
ENV = "WOOF_BACKEND"

_active: str | None = None


class BackendUnavailable(RuntimeError):
    """The requested backend cannot run in this process."""


def _default() -> str:
    if importlib.util.find_spec("cupy") is not None:
        return "cuda"
    if (sys.platform == "darwin" and platform.machine() == "arm64"
            and importlib.util.find_spec("mlx") is not None):
        return "mlx"
    return "cpu"


def requested() -> str:
    """The backend this process would select, without installing it."""
    name = os.environ.get(ENV, "").strip().lower() or _default()
    if name not in BACKENDS:
        raise BackendUnavailable(f"{ENV}={name!r}; expected one of {', '.join(BACKENDS)}")
    return name


def active() -> str:
    """The installed backend (installing the requested one on first use)."""
    return _active if _active is not None else install()


def install(name: str | None = None) -> str:
    """Select the backend for this process; idempotent for the same name."""
    global _active
    name = (name or requested()).lower()
    if name not in BACKENDS:
        raise BackendUnavailable(f"unknown backend {name!r}")
    if _active is not None:
        if _active != name:
            raise BackendUnavailable(f"backend {_active!r} is already installed; "
                                     f"cannot switch to {name!r} in one process")
        return _active
    if name == "cuda":
        _active = name
        return name
    present = sys.modules.get("cupy")
    if present is not None and not getattr(present, "__woof_facade__", False):
        raise BackendUnavailable(
            f"backend {name!r} needs `import cupy` to resolve to woof's facade, "
            "but the real CuPy is already imported in this process")
    if name == "mlx":
        if importlib.util.find_spec("mlx") is None:
            raise BackendUnavailable("backend 'mlx' needs the mlx package "
                                     "(pixi environment `mlx`, or the gpu-mlx extra)")
    from woof.backend import facade
    facade.configure(name)
    sys.modules["cupy"] = facade
    sys.modules["cupyx"] = facade.cupyx
    _active = name
    return name
