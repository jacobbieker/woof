"""hex's CUDA admission on the MLX (Metal) backend.

The hex dycore resolves every kernel through :class:`woof.hex.cuda_backend.
KernelCache`, whose admission (``require_cuda``) and compile contract
(``compile_platform_binding``) ask CUDA questions -- the runtime version, the
card's compute capability, NVRTC's supported architectures and NVRTC's own
build identity -- that woof's CuPy facade over MLX answers with zeros, so the
whole hex lane refused on a Mac before its first kernel.  The kernels
themselves need nothing CUDA-specific: the facade translates each unit to
Metal Shading Language exactly as it does for the WRF core
(docs/dev/mlx-backend.md).

This module answers those questions for the Metal translation, only on the
``mlx`` backend and only when ``WOOF_HEX_METAL=1`` asks for it (otherwise
:func:`install` returns without touching anything, and the hex lane refuses
on a Mac exactly as before).  It is opt-in because only the partial-depth
band (Phase D5, its oracle and envelope tests) has been measured on Metal;
the rest of the hex lane has not.  It changes no byte of the regional kernel set
(``cuda_backend.regional_admission.REGIONAL_KERNEL_SOURCES``): it rebinds
three module globals of the hex runtime instead of editing them.

* ``runtime._cupy``: the facade, seen through a view whose
  ``cuda.runtime`` reports the hex floor's runtime version and compute
  capability (12.0, the sm the cache keys and the NVRTC options name); the
  device's own name and memory are the Metal device's.
* ``compile_platform_binding``: the compile fingerprint names the Metal
  translator -- ``nvrtc_build`` 0.0.0 (no NVRTC), the translator's own
  source digest in ``nvrtc_library_sha256``, the MLX version and the Apple
  GPU architecture -- so a cache entry can never be mistaken for an NVRTC
  image.
* ``cuda_regional_v841._direct_regional_module``: the regional unit's direct
  NVRTC compile (a cubin load) becomes the facade's ``RawModule`` of the same
  source and options.
* the facade's ``nvrtc.getSupportedArchs`` (absent from the facade) reports
  the one architecture the translator stands in for.
* the MPAS_to_physics preparation unit (``cuda_physics_prep_v841``) is
  translated without contraction in fast numerics too: its validator
  recomputes the producers' words, and Metal's ``fp contract(fast)`` may
  fuse the producer and the checker differently.
* ``cuda_subcycle_graphs.graphs_enabled`` answers no: the facade captures
  no CUDA graph, so the dynamics subcycle runs eager.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

#: The compute capability the translator stands in for: the hex lane's floor.
METAL_STANDIN_COMPUTE = (12, 0)
_INSTALLED = False


def _backend() -> str:
    try:
        from woof.backend import requested
        return requested()
    except Exception:
        return os.environ.get("WOOF_BACKEND", "cuda").strip().lower() or "cuda"


def translator_sha256() -> str:
    """One digest over the CUDA->MSL translator and the Metal prelude."""
    import woof.backend as backend
    root = Path(backend.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(list((root / "translate").glob("*.py"))
                       + list((root / "metal").glob("*.h"))):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


class _RuntimeView:
    """The facade's ``cupy.cuda.runtime`` with the admission answers."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    def runtimeGetVersion(self) -> int:
        return 12090

    def driverGetVersion(self) -> int:
        return 12090

    def getDeviceProperties(self, device: int) -> dict:
        props = dict(self._runtime.getDeviceProperties(device))
        props["major"], props["minor"] = METAL_STANDIN_COMPUTE
        return props


class _CudaView:
    def __init__(self, cuda: Any) -> None:
        self._cuda = cuda
        self.runtime = _RuntimeView(cuda.runtime)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cuda, name)


class _CupyView:
    def __init__(self, cupy: Any) -> None:
        self._cupy = cupy
        self.cuda = _CudaView(cupy.cuda)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cupy, name)


def metal_fingerprint() -> dict[str, str]:
    import numpy
    try:
        import mlx.core as mx
        mlx_version = str(getattr(mx, "__version__", "unknown"))
    except Exception:          # pragma: no cover - mlx env always has it
        mlx_version = "unknown"
    import cupy
    architecture = str(dict(cupy.cuda.runtime.getDeviceProperties(0)).get(
        "architecture", "apple-gpu"))
    return {
        "nvrtc_build": "0.0.0",
        "nvrtc_build_id": "UNKNOWN",
        "nvrtc_library_sha256": translator_sha256(),
        "cuda_driver_version": f"metal-mlx-{mlx_version}",
        "device_compute_capability": f"metal:{architecture}",
        "cupy_version": "woof-facade",
        "numpy_version": numpy.__version__,
    }


def install() -> bool:
    """Admit the hex CUDA lane on the Metal translation (``mlx`` with
    ``WOOF_HEX_METAL=1`` only).  Idempotent; returns whether the Metal
    answers are in place."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if _backend() != "mlx" or os.environ.get("WOOF_HEX_METAL", "").strip() != "1":
        return False
    import cupy
    from cupy.cuda import nvrtc
    if not hasattr(nvrtc, "getSupportedArchs"):
        architecture = METAL_STANDIN_COMPUTE[0] * 10 + METAL_STANDIN_COMPUTE[1]
        nvrtc.getSupportedArchs = lambda: (architecture,)
    from .cuda_backend import compile_contract, runtime

    view = _CupyView(cupy)
    runtime._cupy = lambda: view
    fingerprint = metal_fingerprint()

    def binding() -> dict[str, Any]:
        return {"fingerprint": dict(fingerprint),
                "sha256": compile_contract.canonical_sha256(fingerprint)}

    runtime.compile_platform_binding = binding
    compile_contract.compile_platform_binding = binding

    from . import cuda_regional_v841 as regional

    def direct_regional_module(device_id: int, platform_sha256: str):
        from . import numerics
        module = cupy.RawModule(
            code=regional.CUDA_REGIONAL_SOURCE,
            options=numerics.options(regional.REGIONAL_CUDA_OPTIONS))
        evidence = {"route": "metal-translation",
                    "source_sha256": hashlib.sha256(
                        regional.CUDA_REGIONAL_SOURCE.encode()).hexdigest(),
                    "translator_sha256": fingerprint["nvrtc_library_sha256"]}
        return module, evidence

    regional._direct_regional_module = direct_regional_module

    # MPAS_to_physics preparation: validate_prep_v841_f32 recomputes the
    # producers' fields and compares words, which holds on Metal only when
    # neither side contracts (loader.uncontracted_source).
    from woof.backend.metal import loader
    from . import cuda_physics_prep_v841 as prep
    loader.uncontracted_source(prep._CUDA_SOURCE)

    # The facade's stream capture records no graph (MLX keeps its own lazy
    # graph) and a replay launches nothing, so the dynamics subcycle runs
    # eager: a replay would hand back the captured subcycle's words.
    from . import cuda_subcycle_graphs
    cuda_subcycle_graphs.graphs_enabled = lambda cells=None: False
    _INSTALLED = True
    return True


__all__ = ["METAL_STANDIN_COMPUTE", "install", "metal_fingerprint", "translator_sha256"]
