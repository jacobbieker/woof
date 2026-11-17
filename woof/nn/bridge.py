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
lazy graph without a host round trip.

Host.  NumPy arrays pass through (the reference runner and tests).

``array_kind`` names which of the three an array is; nothing here imports
torch or MLX until a function that needs them is called.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any

import numpy as np


def array_kind(array: Any) -> str:
    """``"numpy"``, ``"facade"`` (the MLX-backed CuPy facade) or ``"cupy"``."""
    if isinstance(array, np.ndarray):
        return "numpy"
    if getattr(type(array), "__woof_facade__", False):
        return "facade"
    if hasattr(array, "__cuda_array_interface__"):
        return "cupy"
    raise TypeError(f"not a woof array: {type(array).__name__}")


def array_module(array: Any):
    """NumPy for a host array, else woof's ``cupy`` (real CuPy or the facade)."""
    if array_kind(array) == "numpy":
        return np
    import cupy
    return cupy


# ---------------------------------------------------------------- PyTorch
def to_torch(array: Any):
    """A torch tensor sharing ``array``'s memory (CuPy via DLPack, NumPy via
    ``from_numpy``).  The facade has no device pointer PyTorch can use:
    refused, the MLX runner is the Metal route."""
    import torch
    kind = array_kind(array)
    if kind == "numpy":
        return torch.from_numpy(np.ascontiguousarray(array))
    if kind == "facade":
        raise TypeError("a Metal (facade) array cannot be handed to PyTorch without a host copy; "
                        "use backend = \"mlx\" on Metal")
    return torch.from_dlpack(array)


def from_torch(tensor: Any, like: Any):
    """``tensor`` as an array of ``like``'s kind, sharing memory where the
    kinds allow (CUDA tensor -> CuPy via DLPack)."""
    kind = array_kind(like)
    if kind == "numpy":
        return tensor.detach().cpu().numpy()
    import cupy
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
    if kind == "numpy":
        return mx.array(np.ascontiguousarray(array, dtype=np.float32))
    raise TypeError("a CUDA array cannot be handed to MLX; use backend = \"torch\" on CUDA")


def from_mlx(values: Any, like: Any):
    """An MLX result as ``like``'s kind: a new facade array over it (no host
    copy) or a NumPy array."""
    kind = array_kind(like)
    if kind == "facade":
        from woof.backend.facade._ops import _wrap
        return _wrap(values, np.float32, fresh=True)
    if kind == "numpy":
        import mlx.core as mx
        mx.eval(values)
        return np.array(values, copy=True)
    raise TypeError("an MLX result cannot be handed to a CUDA array")
