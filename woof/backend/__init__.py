"""Which array and kernel backend a woof process runs on.

``cuda``  CuPy and NVRTC: the production backend, untouched by this package.
``mlx``   Apple-silicon GPUs.  ``import cupy`` resolves to
          :mod:`woof.backend.facade`, a CuPy-compatible module over
          ``mlx.core``; kernels are the same CUDA sources translated to Metal
          (:mod:`woof.backend.translate`) and launched in place
          (:mod:`woof.backend.metal`).
``host``  any CPU: ``import cupy`` resolves to
          :mod:`woof.backend.hostfacade`, a CuPy-compatible module over
          NumPy; kernels are the same CUDA sources compiled by the host C++
          compiler (:mod:`woof.backend.host.loader`).  Only by name.
``cpu``   no GPU code: ``cpu`` installs no facade, so ``import cupy``
          stays an ImportError on a machine without CuPy.
``hip``   upstream's experimental AMD ROCm adapter (``amd``/``rocm`` alias
          it; ``hip-nvidia`` is its translation qualification mode): CuPy
          built against ROCm, no facade (:mod:`woof.hip_backend`).

Selection: ``$WOOF_BACKEND`` when set, else ``cuda`` when CuPy is importable,
else ``mlx`` on darwin-arm64 with MLX installed, else ``cpu``.  ``host`` is
never chosen by default: it runs the GPU code at CPU speed.  The choice is
made once per process by :func:`install`, before any ``woof.core`` module
imports cupy; it refuses to swap backends under a process that already
imported the real CuPy, because mixing device and facade arrays cannot be
made safe.

The second half of this module is upstream's execution-backend contract
(``select``, ``get_backend``, ``array_module``, ``xp`` and the device
hooks, woof/backend/README.md).  :func:`get_backend` answers for every
backend above: under ``mlx`` and ``host`` its ``xp`` is the facade that
``import cupy`` resolves to, so upstream's compilation-target and
occupancy hooks reach the same CuPy-compatible API on every backend.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys

BACKENDS = ("cuda", "mlx", "cpu", "host", "hip", "hip-nvidia")
#: Spellings of upstream's AMD adapter.
ALIASES = {"amd": "hip", "rocm": "hip", "auto": "cuda"}
#: The backends a facade serves ``import cupy`` for (mlx:
#: :mod:`woof.backend.facade`; host: :mod:`woof.backend.hostfacade`).
FACADE_BACKENDS = ("mlx", "host")
#: The package that is ``cupy`` under each facade backend.
FACADE_PACKAGES = {"mlx": "woof.backend.facade", "host": "woof.backend.hostfacade"}
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
    name = ALIASES.get(name, name)
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
    name = ALIASES.get(name, name)
    if name not in BACKENDS:
        raise BackendUnavailable(f"unknown backend {name!r}")
    if _active is not None:
        if _active != name:
            raise BackendUnavailable(f"backend {_active!r} is already installed; "
                                     f"cannot switch to {name!r} in one process")
        return _active
    if name in ("cuda", "hip", "hip-nvidia"):
        _active = name
        return name
    present = sys.modules.get("cupy")
    if present is not None and not getattr(present, "__woof_facade__", False):
        raise BackendUnavailable(
            f"backend {name!r} needs `import cupy` to resolve to woof's facade, "
            "but the real CuPy is already imported in this process")
    if name == "host":
        from woof.backend.host.loader import CompileError, compiler
        try:
            compiler()
        except CompileError as exc:
            raise BackendUnavailable(str(exc)) from None
    if name == "mlx":
        if importlib.util.find_spec("mlx") is None:
            raise BackendUnavailable("backend 'mlx' needs the mlx package "
                                     "(pixi environment `mlx`, or the gpu-mlx extra)")
    if name not in FACADE_BACKENDS:
        # No facade serves this backend yet (cpu): leave `import cupy` as the
        # ImportError it is on a machine without CuPy, so importorskip and
        # `except ImportError` guards skip GPU work as they always have.  A
        # finder here made every such import raise BackendUnsupported (an
        # AttributeError) instead -- 487 CPU-tier failures on a CuPy-less
        # Linux runner.
        _active = name
        return name
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
        import importlib
        facade = importlib.import_module(FACADE_PACKAGES[self.backend])
        if not getattr(facade, "_configured", False):
            facade.configure(self.backend)
            facade._configured = True
        if spec.name in facade.SUBMODULES:
            return facade.SUBMODULES[spec.name]
        return facade if spec.name == "cupy" else facade.cupyx

    def exec_module(self, module):
        return None


class _AliasLoader:
    """``cupy.<name>`` is the already-imported ``woof.backend.facade.<name>``."""

    def __init__(self, target: str):
        self.target = target

    def create_module(self, spec):
        import importlib
        return importlib.import_module(self.target)

    def exec_module(self, module):
        return None


class _FacadeFinder:
    """``import cupy`` / ``import cupyx`` -> woof's facade (install())."""

    def __init__(self, backend: str):
        self.backend = backend

    def find_spec(self, fullname, path=None, target=None):
        package = FACADE_PACKAGES[self.backend]
        if fullname.startswith("cupy.") and fullname.count(".") == 1:
            # `import cupy.cuda`: the facade's own module object, not a
            # second execution of its file (which kept a second registry of
            # hostAlloc blocks: alloc_pinned_memory registered in one copy and
            # pointerGetAttributes looked in the other)
            import importlib
            import importlib.machinery
            import importlib.util
            # cupy.linalg, cupy.fft, ...: namespaces a facade serves itself
            served = getattr(importlib.import_module(package), "SUBMODULES", {})
            if fullname in served:
                return importlib.machinery.ModuleSpec(fullname, _FacadeLoader(self.backend))
            own = package + "." + fullname.split(".", 1)[1]
            try:
                found = importlib.util.find_spec(own)
            except (ImportError, ValueError):
                found = None
            if found is not None:
                return importlib.machinery.ModuleSpec(fullname, _AliasLoader(own),
                                                      is_package=found.submodule_search_locations is not None)
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
                importlib.util.find_spec(package).submodule_search_locations)
        return spec


# ---------------------------------------------------------------------------
# Upstream's execution-backend contract (shared by both model engines).

from dataclasses import dataclass  # noqa: E402
import importlib  # noqa: E402
import threading  # noqa: E402
from types import ModuleType  # noqa: E402


@dataclass(frozen=True)
class Backend:
    name: str
    xp: object
    is_host: bool

    def synchronize(self):
        if not self.is_host:
            self.xp.cuda.get_current_stream().synchronize()

    def compile(self, code, options=(), name_expressions=None):
        from woof.certify.kernel_manifest import record_module, source_sha256
        module = self.xp.RawModule(code=code, options=options,
                                   name_expressions=name_expressions)
        record_module("backend-" + source_sha256(code)[:16], source=code,
                      options=options, module=module)
        return module

    def to_host(self, array):
        return self.xp.asnumpy(array)


_selected = None
_lock = threading.RLock()
_factories = {}


def register(name, factory):
    """Register an additional backend before selecting it (e.g. ROCm)."""
    with _lock:
        if name in {"auto", "cuda", "cpu", "amd", "mlx", "host"} or name in _factories:
            raise ValueError(f"backend already registered: {name}")
        _factories[name] = factory


def select(name):
    name = ALIASES.get(name, name)
    with _lock:
        if name not in {"cuda", *FACADE_BACKENDS} | _factories.keys():
            raise ValueError(f"unknown forecast backend: {name}")
        if _selected is not None and name not in ("auto", _selected.name):
            raise RuntimeError(f"forecast backend already fixed to {_selected.name}")
        os.environ["WOOF_BACKEND"] = name
    return name


def _cuda():
    from woof.local_gpu import no_local_gpu
    if no_local_gpu():
        raise RuntimeError("local GPU access disabled")
    cp = importlib.import_module("cupy")
    if getattr(cp.cuda.runtime, "is_hip", False):
        raise RuntimeError("CUDA is the default; select --backend amd for experimental ROCm execution")
    return Backend("cuda", cp, False)


def get_backend():
    global _selected
    with _lock:
        if _selected is not None:
            return _selected
        name = os.environ.get("WOOF_BACKEND", "").strip().lower() or _default()
        name = ALIASES.get(name, name)
        if name in _factories:
            _selected = _factories[name]()
        elif name == "cuda":
            _selected = _cuda()
        elif name in FACADE_BACKENDS:
            # mlx / host: `import cupy` is woof's facade once installed.
            install(name)
            _selected = Backend(name, importlib.import_module("cupy"), False)
        elif name == "cpu":
            raise RuntimeError("WOOF_BACKEND=cpu runs no GPU code: there is no "
                               "forecast execution backend in this process")
        else:
            raise ValueError(f"unknown forecast backend: {name}; choose cuda, mlx, host or amd (experimental)")
        os.environ["WOOF_BACKEND"] = _selected.name
        return _selected


def array_module():
    return get_backend().xp


def is_numpy_setup_array(value):
    """Distinguish preprocessing storage from arrays owned by a backend."""
    import numpy as np
    return isinstance(value, np.ndarray) and not getattr(value, "_gpuwm_backend_array", False)


def current_device_id(array_module=None):
    """Current execution domain, including explicitly supplied array APIs."""
    if array_module is None:
        backend = get_backend()
        custom = getattr(backend, "current_device_id", None)
        if custom is not None:
            return int(custom())
        array_module = backend.xp
    return (0 if getattr(array_module, "_gpuwm_host_backend", False)
            else int(array_module.cuda.runtime.getDevice()))


def memory_info(*, array_module=None):
    """Available and total bytes in the selected execution domain."""
    if array_module is None:
        backend = get_backend()
        custom = getattr(backend, "memory_info", None)
        if custom is not None:
            return tuple(map(int, custom()))
        array_module, host = backend.xp, backend.is_host
    else:
        host = getattr(array_module, "_gpuwm_host_backend", False)
    if not host:
        return tuple(map(int,array_module.cuda.runtime.memGetInfo()))
    from pathlib import Path
    values={line.split(':')[0]:int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith(('MemAvailable:', 'MemTotal:'))}
    return values['MemAvailable'],values['MemTotal']


def compilation_target(device=None, *, xp=None):
    """CUDA compilation architecture, or None for native CPU compilation."""
    if xp is not None:
        if getattr(getattr(xp.cuda,'runtime',None),'is_hip',False):
            from woof.hip_backend import architecture
            return architecture(xp,device)
        return (xp.cuda.Device() if device is None else xp.cuda.Device(device)).compute_capability
    backend=get_backend()
    custom = getattr(backend, "compilation_target", None)
    if custom is not None:
        return custom(device)
    if backend.is_host:return None
    if backend.name == 'hip':
        from woof.hip_backend import architecture
        return architecture(backend.xp,device)
    return backend.xp.cuda.Device(device).compute_capability


def image_target(device=None, *, xp=None):
    """Native image ISA, including its gfx or sm prefix."""
    target=compilation_target(device,xp=xp)
    if target is None:return None
    return str(target) if str(target).startswith('gfx') else 'sm_'+str(target)


def active_blocks_per_multiprocessor(function, block_size, dynamic_shared=0, *, xp=None):
    """Occupancy from the selected execution adapter's loaded kernel."""
    if xp is None:
        backend = get_backend()
        custom = getattr(backend, 'active_blocks_per_multiprocessor', None)
        if custom is not None:
            return int(custom(function, block_size, dynamic_shared))
        xp = backend.xp
    if getattr(xp.cuda.runtime, 'is_hip', False):
        from woof.hip_runtime import active_blocks_per_multiprocessor as hip_occupancy
        return hip_occupancy(function, block_size, dynamic_shared)
    return int(xp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
        function, block_size, dynamic_shared))


def consume_backend_option(argv, *, local_backend_commands=()):
    """Allow the common backend option before or after a command word."""
    tokens=list(argv);remaining=[];i=0;command=None
    while i<len(tokens):
        token=tokens[i]
        if command in local_backend_commands:
            remaining.extend(tokens[i:])
            break
        if token=='--backend':
            if i+1==len(tokens):
                raise ValueError('--backend requires a backend name')
            select(tokens[i+1]);i+=2
        elif token.startswith('--backend='):
            select(token.split('=',1)[1]);i+=1
        else:
            remaining.append(token);i+=1
            if command is None and not token.startswith('-'):
                command=token
    if "WOOF_BACKEND" in os.environ:
        select(os.environ["WOOF_BACKEND"].strip().lower())
    return remaining


class _ArrayModule(ModuleType):
    def __getattr__(self, name):
        value = getattr(array_module(), name)
        # Backend selection is immutable. Bind each array API attribute once,
        # so the CUDA hot path calls the original CuPy function directly.
        setattr(self, name, value)
        return value


xp = _ArrayModule("woof.backend.xp")

# GPU adapters share this interface and the canonical kernel sources.
from woof.hip_backend import HipBackend
register('hip', lambda: HipBackend('hip'))
register('hip-nvidia', lambda: HipBackend('hip-nvidia'))


def __getattr__(name):
    if name in ("cuda", "compiler"):
        runtime = array_module().cuda
        return runtime if name == "cuda" else runtime.compiler
    raise AttributeError(name)
