"""The tensor bridge: woof device arrays <-> PyTorch tensors (plan P1.1).

Every exchange shares memory where the two runtimes can address the same
buffer, and says so when they cannot:

========================  =====================  ==========================
woof array                to torch               back to woof
========================  =====================  ==========================
CuPy (CUDA)               DLPack, zero-copy      DLPack, zero-copy
facade (Metal, MLX)       DLPack to ``mps``,     one device copy
                          zero-copy              (``mx.from_dlpack`` copies)
NumPy / host facade       ``from_numpy``,        ``.numpy()``, zero-copy
                          zero-copy (CPU)        for a CPU tensor
========================  =====================  ==========================

Anything else (a CuPy array to a CPU or MPS tensor, a CPU tensor to CuPy,
a facade array to a CPU tensor, float64 to MPS, a CUDA array on another
device than the tensor's) is refused unless ``copy=True`` names the copy.

Ordering.  CUDA: woof launches on CuPy's current stream and torch on its
own current stream.  :func:`to_torch` makes torch's current stream wait
for CuPy's current stream (a device-side event wait, no host sync), and
:func:`from_torch` makes CuPy's current stream wait for torch's, so
neither side reads a buffer the other is still writing.  The wait is for
the streams current at the call: a tensor used later on another torch
stream needs torch's own ``wait_stream``, as any torch tensor does.  :func:`woof_stream` instead runs the torch work on
CuPy's stream itself (no handoff needed).  Metal: MLX and torch-MPS have
separate command queues and expose no cross-queue event, so the handoff is
a full drain of the producer: ``mx.eval`` + ``mx.synchronize`` before a
facade array is exported, ``torch.mps.synchronize`` before a tensor is
handed back (measured without it: MLX read a recycled buffer's stale
words).

Autograd.  woof's kernels write in place, and torch cannot see a CuPy or
MLX write to a buffer it shares, so a shared view that autograd saved for
backward could be overwritten under it and give a silently wrong
gradient.  The bridge therefore never returns a zero-copy view with
``requires_grad``: :func:`grad_leaf` gives an out-of-place copy that is a
leaf of the graph, :func:`from_torch` refuses a tensor that requires grad
(``.detach()`` it first, deliberately), and :func:`mark_written` bumps a
shared view's version counter after woof writes its buffer, so a backward
that saved it raises instead of using stale words.

Nothing here imports torch, CuPy or MLX until a function needs them; no
forecast path imports this module.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np

from woof.nn.bridge import array_kind, host_view

#: dtypes the bridge exchanges (woof's fields are float32; float64 for the
#: binary64 gradient tests; integer index arrays).
DTYPES = (np.float32, np.float64, np.int32, np.int64)


class BridgeRefused(TypeError):
    """An exchange the bridge will not make silently; the message says why
    and what to pass instead."""


def _torch():
    try:
        import torch
    except ImportError:
        from woof.nn.models import torch_remedy
        raise BridgeRefused(f"PyTorch is not installed in this interpreter: {torch_remedy()}") from None
    return torch


def _check_dtype(dtype) -> None:
    if np.dtype(dtype) not in [np.dtype(d) for d in DTYPES]:
        raise BridgeRefused(f"the bridge exchanges {[np.dtype(d).name for d in DTYPES]}, not {np.dtype(dtype)}")


# ---------------------------------------------------------------- CUDA ordering
def _cupy_stream_for_torch(torch):
    import cupy
    return torch.cuda.ExternalStream(cupy.cuda.get_current_stream().ptr)


def _torch_waits_for_cupy(torch) -> None:
    torch.cuda.current_stream().wait_stream(_cupy_stream_for_torch(torch))


def _cupy_waits_for_torch(torch) -> None:
    import cupy
    event = cupy.cuda.Event(disable_timing=True)
    stream = torch.cuda.current_stream()
    if hasattr(cupy.cuda.Stream, "from_external"):       # CuPy >= 14
        external = cupy.cuda.Stream.from_external(stream)
    else:
        external = cupy.cuda.ExternalStream(stream.cuda_stream)
    event.record(external)
    cupy.cuda.get_current_stream().wait_event(event)


@contextmanager
def woof_stream(like: Any = None):
    """Run the enclosed torch work on woof's current CuPy stream (CUDA);
    a no-op for the other kinds."""
    if like is not None and array_kind(like) != "cupy":
        yield
        return
    torch = _torch()
    if not torch.cuda.is_available():
        yield
        return
    with torch.cuda.stream(_cupy_stream_for_torch(torch)):
        yield


# ---------------------------------------------------------------- Metal ordering
def _mlx_drain(values) -> None:
    """Finish every facade write and the value itself (the facade's own
    synchronize: pending storages, then the MLX queue)."""
    import mlx.core as mx
    from woof.backend.facade.cuda import _drain
    mx.eval(values)
    _drain()


def _mps_drain(torch) -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def release(like: Any) -> None:
    """Order woof's next launches after torch's work so far, for the arrays
    of ``like``'s kind: call it after torch wrote a shared view in place
    (:func:`from_torch` does it for the tensors it hands back).  CUDA: a
    device-side wait of CuPy's stream on torch's; Metal: drain torch's
    queue; host: nothing."""
    kind = array_kind(like)
    if kind == "cupy":
        _cupy_waits_for_torch(_torch())
    elif kind == "facade":
        _mps_drain(_torch())


# ---------------------------------------------------------------- exchange
def to_torch(array: Any, *, device: str | None = None, copy: bool = False):
    """A torch tensor holding ``array``'s values.

    Zero-copy for CuPy -> ``cuda``, facade -> ``mps`` and NumPy/host ->
    ``cpu`` (the default device for each kind); the tensor aliases woof's
    buffer, so a later woof write is visible through it (and
    :func:`mark_written` should follow it if autograd saw the tensor).
    Any other ``device`` needs ``copy=True``."""
    torch = _torch()
    kind = array_kind(array)
    _check_dtype(array.dtype)
    native = {"cupy": "cuda", "facade": "mps", "numpy": "cpu", "host": "cpu"}[kind]
    target = torch.device(device) if device is not None else None
    if kind == "cupy":
        import cupy
        here = torch.device("cuda", array.device.id)
        if target is None:
            target = here
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        if target != here and not copy:
            raise BridgeRefused(f"a CuPy array on cuda:{array.device.id} is not addressable as {target}; "
                                "pass copy=True to copy it")
        with cupy.cuda.Device(array.device.id):
            with torch.cuda.device(here.index):
                _torch_waits_for_cupy(torch)
                tensor = torch.from_dlpack(array)
        return tensor.to(target, copy=True) if target != here else tensor
    if kind == "facade":
        if target is None:
            target = torch.device("mps")
        if np.dtype(array.dtype) == np.float64 and target.type == "mps":
            raise BridgeRefused("torch-MPS has no float64; pass device='cpu', copy=True for a binary64 "
                                "CPU tensor")
        values = array._values()
        _mlx_drain(values)
        if target.type == "mps":
            return torch.from_dlpack(values)
        if not copy:
            raise BridgeRefused(f"a Metal (facade) array is shared with torch on mps only, not {target}; "
                                "pass copy=True to copy it")
        return torch.from_numpy(np.array(values, copy=True)).to(target)
    host = host_view(array) if kind == "host" else array
    if target is not None and target.type != "cpu":
        if not copy:
            raise BridgeRefused(f"a host array is shared with torch on cpu only, not {target}; "
                                "pass copy=True to copy it")
        return torch.from_numpy(np.array(host, copy=True)).to(target)
    if not host.flags.c_contiguous and not copy:
        raise BridgeRefused("a non-contiguous host array cannot be shared; pass copy=True")
    return torch.from_numpy(np.array(host, copy=True) if copy else host)


def from_torch(tensor: Any, like: Any, *, copy: bool = False):
    """``tensor`` as a woof array of ``like``'s kind.

    Zero-copy for a CUDA tensor -> CuPy (same device) and a CPU tensor ->
    NumPy; an MPS tensor -> facade is one device copy (MLX cannot wrap a
    torch buffer), which this function makes after draining torch's
    queue.  A tensor that requires grad is refused: detach it on purpose,
    so a buffer autograd still holds is never handed to an in-place
    kernel by accident."""
    torch = _torch()
    if tensor.requires_grad:
        raise BridgeRefused("this tensor requires grad; hand woof tensor.detach() (or a clone) "
                            "deliberately, so autograd's saved buffers are never written in place")
    kind = array_kind(like)
    _check_dtype(str(tensor.dtype).removeprefix("torch."))
    if kind == "cupy":
        import cupy
        if not tensor.is_cuda:
            if not copy:
                raise BridgeRefused(f"a {tensor.device} tensor cannot become a CuPy array without a copy; "
                                    "pass copy=True")
            return cupy.asarray(tensor.cpu().numpy())
        if tensor.device.index != like.device.id:
            if not copy:
                raise BridgeRefused(f"the tensor is on {tensor.device}, the woof array on "
                                    f"cuda:{like.device.id}; pass copy=True")
            tensor = tensor.to(torch.device("cuda", like.device.id))
        with cupy.cuda.Device(like.device.id):
            with torch.cuda.device(tensor.device.index):
                _cupy_waits_for_torch(torch)
                return cupy.from_dlpack(tensor)
    if kind == "facade":
        import mlx.core as mx
        from woof.backend.facade._ops import _wrap
        if tensor.device.type == "mps":
            _mps_drain(torch)
            values = mx.from_dlpack(tensor)
        elif copy:
            values = mx.array(tensor.cpu().numpy())
        else:
            raise BridgeRefused(f"a {tensor.device} tensor cannot become a Metal array without a copy; "
                                "pass copy=True")
        return _wrap(values, np.dtype(str(tensor.dtype).removeprefix("torch.")), fresh=True)
    if tensor.device.type != "cpu":
        if not copy:
            raise BridgeRefused(f"a {tensor.device} tensor cannot become a host array without a copy; "
                                "pass copy=True")
        tensor = tensor.cpu()
    host = tensor.numpy()
    if kind == "host":
        import cupy
        return cupy.asarray(host)
    return host.copy() if copy else host


def grad_leaf(array: Any, *, device: str | None = None, dtype=None):
    """An out-of-place copy of ``array`` as a torch leaf with
    ``requires_grad``: what an autograd computation should start from, so
    woof's in-place kernels never write a buffer the graph saved."""
    torch = _torch()
    tensor = to_torch(array, device=device, copy=True) if device is not None else to_torch(array)
    tensor = tensor.detach().clone()
    if dtype is not None:
        tensor = tensor.to(getattr(torch, np.dtype(dtype).name))
    return tensor.requires_grad_(True)


def mark_written(*tensors) -> None:
    """Tell autograd that woof wrote these shared views' buffers, so a
    backward that saved one before the write raises instead of reading the
    new words."""
    torch = _torch()
    for tensor in tensors:
        torch.autograd.graph.increment_version(tensor)


def shares_memory(array: Any, tensor: Any) -> bool:
    """Whether ``tensor`` and ``array`` address the same first word (the
    zero-copy tests' check)."""
    kind = array_kind(array)
    if kind == "cupy":
        return tensor.is_cuda and tensor.data_ptr() == array.data.ptr
    if kind == "facade":
        if tensor.device.type != "mps":
            return False
        # MLX and torch-MPS name a shared buffer by the same MTLBuffer plus
        # offset; neither exposes it, so the check is behavioural: a write
        # through torch is read back through MLX.
        torch = _torch()
        probe = tensor.reshape(-1)[:1]
        saved = probe.clone()
        probe.add_(1)
        _mps_drain(torch)
        seen = float(np.asarray(array._values()).reshape(-1)[0])
        probe.copy_(saved)
        _mps_drain(torch)
        return seen == float(saved.cpu()[0]) + 1
    host = host_view(array) if kind == "host" else array
    return tensor.device.type == "cpu" and tensor.data_ptr() == host.ctypes.data
