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
    # Lazily: `import cupy` finds the facade when something first imports it,
    # so a module that imports no cupy on CUDA imports none here either.
    if not any(isinstance(f, _FacadeFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _FacadeFinder(name))
    _active = name
    return name


class _FacadeLoader:
    """Hands out the facade (``cupy``) or its ``cupyx`` as the module."""

    def __init__(self, backend: str):
        self.backend = backend

    def create_module(self, spec):
        from woof.backend import facade
        if not getattr(facade, "_configured", False):
            facade.configure(self.backend)
            facade._configured = True
        if spec.name in facade.SUBMODULES:
            return facade.SUBMODULES[spec.name]
        return facade if spec.name == "cupy" else facade.cupyx

    def exec_module(self, module):
        return None


class _FacadeFinder:
    """``import cupy`` / ``import cupyx`` -> woof's facade (install())."""

    def __init__(self, backend: str):
        self.backend = backend

    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("cupy_backends", "cupy_backends.cuda", "cupy_backends.cuda.api",
                        "cupy_backends.cuda.libs"):
            # packages only; their modules are the facade's cupy.cuda
            # namespaces, registered when it loads (facade/cuda.py)
            import importlib.machinery
            import cupy.cuda  # noqa: F401
            return importlib.machinery.ModuleSpec(fullname, None, is_package=True)
        if fullname not in ("cupy", "cupyx", "cupyx.scipy", "cupyx.scipy.special", "cupyx.scipy.ndimage"):
            return None
        import importlib.machinery
        spec = importlib.machinery.ModuleSpec(fullname, _FacadeLoader(self.backend),
                                              is_package=fullname in ("cupy", "cupyx", "cupyx.scipy"))
        if fullname == "cupy":
            # `import cupy.cuda`: the facade package's own submodules
            spec.submodule_search_locations = list(
                importlib.util.find_spec("woof.backend.facade").submodule_search_locations)
        return spec
