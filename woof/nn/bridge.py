"""Zero-copy array exchange between woof's device arrays and the NN runtimes.

CUDA.  woof's arrays are CuPy arrays; :func:`to_torch` hands one to
PyTorch through DLPack (``torch.from_dlpack``), which shares the device
buffer, and :func:`from_torch` hands a result back the same way
(``cupy.from_dlpack``).  The model runs on woof's current CuPy stream
(:func:`torch_stream`), so no synchronisation is added between the
dycore's launches and the model's: they are ordered on one stream as CUDA
orders any two launches.

Metal.  woof's ``cupy`` is the facade over MLX (:mod:`woof.backend.facade`):
each array is a view of an ``mx.array``.  :func:`to_mlx` returns that
``mx.array`` (lazy, no copy) and :func:`from_mlx` wraps an MLX result as a
new facade array in its own storage, so the model's graph joins woof's
lazy graph without a host round trip.  PyTorch on Metal runs on ``mps``:
:func:`to_torch` shares the facade's buffer through DLPack and
:func:`from_torch` copies the result back once on the device
(:mod:`woof.grad.bridge` documents the ordering).

Host.  NumPy arrays pass through (the reference runner and tests).

``array_kind`` names which of the three an array is; nothing here imports
torch or MLX until a function that needs them is called.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any

import numpy as np


def array_kind(array: Any) -> str:
    """``"numpy"``, ``"facade"`` (the MLX-backed CuPy facade), ``"host"``
    (the host facade, :mod:`woof.backend.hostfacade`: NumPy memory behind
    CuPy's interface) or ``"cupy"``."""
    if isinstance(array, np.ndarray):
        return "numpy"
    if getattr(type(array), "__woof_facade__", False):
        return "facade"
    if type(array).__module__.startswith("woof.backend.hostfacade"):
        return "host"
    if hasattr(array, "__cuda_array_interface__"):
        return "cupy"
    raise TypeError(f"not a woof array: {type(array).__name__}")


def array_module(array: Any):
    """The array's own module: NumPy, or woof's ``cupy`` (CuPy or a facade),
    as ``cupy.get_array_module`` answers it."""
    if array_kind(array) == "numpy":
        return np
    import cupy
    return cupy.get_array_module(array)


def host_view(array: Any) -> np.ndarray:
    """A NumPy view of a host-facade array's memory (no copy)."""
    return array._a


# ---------------------------------------------------------------- PyTorch
def to_torch(array: Any):
    """A torch tensor sharing ``array``'s memory (CuPy via DLPack, NumPy via
    ``from_numpy``, a facade array as a torch-MPS tensor through DLPack,
    after the MLX queue drains: :func:`woof.grad.bridge.to_torch`)."""
    import torch
    kind = array_kind(array)
    if kind == "numpy":
        return torch.from_numpy(np.ascontiguousarray(array))
    if kind == "host":
        return torch.from_numpy(np.ascontiguousarray(host_view(array)))
    if kind == "facade":
        from woof.grad import bridge as grad_bridge
        if np.dtype(array.dtype) == np.float64:
            array = array.astype(np.float32)          # in MLX: torch-MPS has no float64
        if not torch.backends.mps.is_available():
            return grad_bridge.to_torch(array, device="cpu", copy=True)
        return grad_bridge.to_torch(array)
    return torch.from_dlpack(array)


def from_torch(tensor: Any, like: Any):
    """``tensor`` as an array of ``like``'s kind, sharing memory where the
    kinds allow (CUDA tensor -> CuPy via DLPack)."""
    kind = array_kind(like)
    if kind == "numpy":
        return tensor.detach().cpu().numpy()
    if kind == "facade":
        from woof.grad import bridge as grad_bridge
        return grad_bridge.from_torch(tensor.detach(), like, copy=True)
    import cupy
    if kind == "host":
        return cupy.asarray(tensor.detach().cpu().numpy())
    if not tensor.is_cuda:
        raise TypeError("a CPU tensor cannot be handed back to a CUDA array without a copy")
    return cupy.from_dlpack(tensor.detach())


@contextmanager
def torch_stream(like: Any):
    """Run PyTorch work on woof's current CuPy stream (CUDA arrays only)."""
    if array_kind(like) != "cupy":
        with nullcontext():
            yield
        return
    import cupy
    import torch
    stream = torch.cuda.ExternalStream(cupy.cuda.get_current_stream().ptr)
    with torch.cuda.stream(stream):
        yield


# ---------------------------------------------------------------- MLX
def to_mlx(array: Any):
    """An ``mx.array`` holding ``array``'s values: the facade's own storage
    view (no copy) or a NumPy upload."""
    import mlx.core as mx
    kind = array_kind(array)
    if kind == "facade":
        if array.dtype != np.float32:
            raise TypeError(f"the MLX runner takes float32 arrays, got {array.dtype}")
        return array._values()
    if kind in ("numpy", "host"):
        host = host_view(array) if kind == "host" else array
        return mx.array(np.ascontiguousarray(host, dtype=np.float32))
    raise TypeError("a CUDA array cannot be handed to MLX; use backend = \"torch\" on CUDA")


def from_mlx(values: Any, like: Any):
    """An MLX result as ``like``'s kind: a new facade array over it (no host
    copy) or a NumPy array."""
    kind = array_kind(like)
    if kind == "facade":
        from woof.backend.facade._ops import _wrap
        return _wrap(values, np.float32, fresh=True)
    if kind in ("numpy", "host"):
        import mlx.core as mx
        mx.eval(values)
        host = np.array(values, copy=True)
        if kind == "host":
            import cupy
            return cupy.asarray(host)
        return host
    raise TypeError("an MLX result cannot be handed to a CUDA array")
