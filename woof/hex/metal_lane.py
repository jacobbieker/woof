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

This module answers those questions for the Metal translation on the
``mlx`` backend, by default; ``WOOF_HEX_METAL=0`` opts out (then
:func:`install` returns without touching anything, and the hex lane refuses
on a Mac).  The hex GPU gates and an end-to-end regional forecast are
measured against CUDA on Metal, and every word they differ in is attributed
(docs/dev/mlx-backend.md, "hex (MPAS) on Metal").  It changes no byte of the
regional kernel set (``cuda_backend.regional_admission.REGIONAL_KERNEL_SOURCES``):
it rebinds module globals of the hex runtime instead of editing them.

* ``runtime._cupy``: the facade, seen through a view whose
  ``cuda.runtime`` reports the hex floor's runtime version and compute
  capability (12.0, the sm the cache keys and the NVRTC options name); the
  device's own name and memory are the Metal device's.
* ``compile_platform_binding``: the compile fingerprint names the Metal
  translator -- ``nvrtc_build`` 0.0.0 (no NVRTC), the translator's own
  source digest in ``nvrtc_library_sha256``, the MLX version and the Apple
  GPU architecture -- so a cache entry can never be mistaken for an NVRTC
  image.
* (the regional unit's direct NVRTC compile needs nothing: the facade's
  ``compile_using_nvrtc`` translates it and ``function.Module().load``
  loads the translation).
* the facade's ``nvrtc.getSupportedArchs`` (absent from the facade) reports
  the one architecture the translator stands in for.
* the MPAS_to_physics preparation unit (``cuda_physics_prep_v841``) is
  translated without contraction in fast numerics too: its validator
  recomputes the producers' words, and Metal's ``fp contract(fast)`` may
  fuse the producer and the checker differently.

With the lane installed the forecast driver attaches no subcycle graph
(``installed()``): the facade's stream capture records no CUDA graph, so a
replay would launch nothing, and the dynamics subcycle runs eager.

Each answer is the Metal one only while ``cupy`` is woof's facade; a
process (or a test) that puts another ``cupy`` in ``sys.modules`` gets the
hex runtime's own answers for it.
"""

from __future__ import annotations

import hashlib
import os
import sys
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
    """Admit the hex CUDA lane on the Metal translation (``mlx``, unless
    ``WOOF_HEX_METAL=0``).  Idempotent; returns whether the Metal answers
    are in place."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if _backend() != "mlx" or os.environ.get("WOOF_HEX_METAL", "").strip() == "0":
        return False
    import cupy
    from cupy.cuda import nvrtc
    if not hasattr(nvrtc, "getSupportedArchs"):
        architecture = METAL_STANDIN_COMPUTE[0] * 10 + METAL_STANDIN_COMPUTE[1]
        nvrtc.getSupportedArchs = lambda: (architecture,)
    from .cuda_backend import compile_contract, runtime

    facade = cupy

    def on_facade() -> bool:
        return sys.modules.get("cupy") is facade

    view = _CupyView(cupy)
    own_cupy = runtime._cupy
    runtime._cupy = lambda: view if on_facade() else own_cupy()
    fingerprint = metal_fingerprint()
    own_binding = runtime.compile_platform_binding
    own_contract_binding = compile_contract.compile_platform_binding

    def binding() -> dict[str, Any]:
        if not on_facade():
            return own_binding()
        return {"fingerprint": dict(fingerprint),
                "sha256": compile_contract.canonical_sha256(fingerprint)}

    def contract_binding() -> dict[str, Any]:
        return binding() if on_facade() else own_contract_binding()

    runtime.compile_platform_binding = binding
    compile_contract.compile_platform_binding = contract_binding


    # MPAS_to_physics preparation: validate_prep_v841_f32 recomputes the
    # producers' fields and compares words, which holds on Metal only when
    # neither side contracts (loader.uncontracted_source).
    # (by name: this runs while woof.hex.cuda_backend is importing, which
    # the preparation module imports first)
    from woof.backend.metal import loader
    loader.uncontracted_source(module="woof.hex.cuda_physics_prep_v841", attribute="_CUDA_SOURCE")
    _INSTALLED = True
    return True


def installed() -> bool:
    """Whether the hex lane runs on the Metal translation (installed, and
    ``cupy`` is woof's facade)."""
    if not _INSTALLED:
        return False
    import woof.backend.facade as facade
    return sys.modules.get("cupy") is facade


__all__ = ["METAL_STANDIN_COMPUTE", "install", "installed", "metal_fingerprint", "translator_sha256"]
