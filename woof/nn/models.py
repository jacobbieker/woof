"""Column-model runners: PyTorch on CUDA, MLX on Metal, NumPy on the host.

Every runner implements :class:`ColumnModel`: called with a float32
``[n_inputs, column]`` array of the caller's kind (CuPy, facade or NumPy),
it returns ``[n_outputs, column]`` of the same kind, in physical units.
Normalisation (``(x - mean) / std`` in, ``y * std + mean`` out) happens
inside, in the runtime.

:func:`load_column_model` picks the runner: ``backend = "auto"`` follows
woof's own backend (``torch`` under CUDA, ``mlx`` under the MLX facade,
``numpy`` on a host-only run); a backend the artefact's format cannot run
on is refused by name rather than converted.

PyTorch is an optional extra (``recast-woof[nn-torch]``), never a base
dependency.  Its version floor is the Earth2 door's pinned torch
(``woof/authorities/earth2-models.v1.json`` runtime.install_recipes): the
Earth2 doors run torch in a separate interpreter
(:mod:`woof.earth2_env`), which cannot share device memory with the
forecast process, so a column model needs torch in woof's own interpreter;
:func:`torch_remedy` names the same index and version the Earth2 recipe
installs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from woof.nn import bridge
from woof.nn.manifest import ColumnModelManifest, ManifestError, load_manifest, mlp_layers

BACKENDS = ("auto", "torch", "mlx", "numpy")


class ModelUnavailable(RuntimeError):
    """The requested runtime is not installed or cannot run this artefact."""


@runtime_checkable
class ColumnModel(Protocol):
    manifest: ColumnModelManifest
    backend: str

    def __call__(self, x: Any) -> Any: ...


def torch_remedy() -> str:
    """The install line for torch in woof's interpreter, from the Earth2 pin."""
    try:
        from woof import earth2_env
        recipe = earth2_env.runtime()["install_recipes"][earth2_env.recipe_name()]
        line = next(c for c in recipe["commands"] if " torch==" in c)
        version = line.split("torch==", 1)[1].split()[0]
        return (f"pip install torch=={version} --index-url {recipe['torch_index']} "
                "(the torch the Earth2 door pins, installed into woof's own interpreter; "
                "or pip install 'recast-woof[nn-torch]' for PyPI's build)")
    except Exception:
        return "pip install 'recast-woof[nn-torch]'"


def _activation_numpy(name: str):
    if name == "relu":
        return lambda h: np.maximum(h, np.float32(0))
    if name == "tanh":
        return np.tanh
    if name == "silu":
        return lambda h: h / (np.float32(1) + np.exp(-h))
    if name == "gelu":
        from math import sqrt
        c = np.float32(sqrt(2.0 / np.pi))
        return lambda h: np.float32(0.5) * h * (np.float32(1) + np.tanh(c * (h + np.float32(0.044715) * h ** 3)))
    raise ManifestError(f"unknown activation {name!r}")


@dataclass
class NumpyColumnModel:
    """The reference runner (``mlp-npz`` only): float32 NumPy on the host."""

    manifest: ColumnModelManifest
    backend: str = "numpy"

    def __post_init__(self):
        if self.manifest.format != "mlp-npz":
            raise ModelUnavailable(f"the numpy runner reads mlp-npz artefacts, not {self.manifest.format}")
        self._layers = mlp_layers(self.manifest)
        self._act = _activation_numpy(self.manifest.architecture["activation"])
        m = self.manifest
        self._in = (m.input_mean[:, None], m.input_std[:, None])
        self._out = (m.output_mean[:, None], m.output_std[:, None])

    def __call__(self, x):
        device = bridge.array_kind(x) != "numpy"
        host = x.get() if device else x
        h = (np.asarray(host, dtype=np.float32) - self._in[0]) / self._in[1]
        for index, (w, b) in enumerate(self._layers):
            h = w @ h + b[:, None]
            if index < len(self._layers) - 1:
                h = self._act(h)
        y = (h * self._out[1] + self._out[0]).astype(np.float32, copy=False)
        return bridge.array_module(x).asarray(y) if device else y


class TorchColumnModel:
    """PyTorch runner: ``mlp-npz`` built as ``nn.Sequential``, or a
    TorchScript / ``torch.export`` artefact; CUDA tensors exchanged with
    CuPy through DLPack, on woof's stream."""

    def __init__(self, manifest: ColumnModelManifest, *, device: str | None = None):
        try:
            import torch
        except ImportError:
            raise ModelUnavailable(f"PyTorch is not installed in this interpreter: {torch_remedy()}") from None
        if manifest.format == "mlx-safetensors":
            raise ModelUnavailable("an mlx-safetensors artefact runs on Metal only; export mlp-npz for CUDA")
        self.manifest = manifest
        self.backend = "torch"
        self._torch = torch
        if device is None:
            # Metal: torch-MPS shares the facade's buffers (woof.grad.bridge)
            metal = default_backend() == "mlx" and torch.backends.mps.is_available()
            device = "cuda" if torch.cuda.is_available() else ("mps" if metal else "cpu")
        self._device = torch.device(device)
        dev = self._device
        f32 = torch.float32
        self._in_mean = torch.as_tensor(manifest.input_mean, dtype=f32, device=dev)
        self._in_std = torch.as_tensor(manifest.input_std, dtype=f32, device=dev)
        self._out_mean = torch.as_tensor(manifest.output_mean, dtype=f32, device=dev)
        self._out_std = torch.as_tensor(manifest.output_std, dtype=f32, device=dev)
        if manifest.format == "mlp-npz":
            act = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh, "gelu": lambda: torch.nn.GELU("tanh"),
                   "silu": torch.nn.SiLU}[manifest.architecture["activation"]]
            modules = []
            layers = mlp_layers(manifest)
            for index, (w, b) in enumerate(layers):
                linear = torch.nn.Linear(w.shape[1], w.shape[0])
                with torch.no_grad():
                    linear.weight.copy_(torch.from_numpy(w))
                    linear.bias.copy_(torch.from_numpy(b))
                modules.append(linear)
                if index < len(layers) - 1:
                    modules.append(act())
            self._module = torch.nn.Sequential(*modules).to(dev).eval()
        elif manifest.format == "torchscript":
            self._module = torch.jit.load(str(manifest.weights), map_location=dev).eval()
        else:
            self._module = torch.export.load(str(manifest.weights)).module().to(dev)

    def __call__(self, x):
        torch = self._torch
        if bridge.array_kind(x) == "cupy" and self._device.type != "cuda":
            raise ModelUnavailable(
                f"the column model runs on {self._device} but the forecast's arrays are on the GPU; "
                f"a CUDA build of torch is needed: {torch_remedy()}")
        with bridge.torch_stream(x), torch.inference_mode():
            t = bridge.to_torch(x)
            if t.device != self._device or t.dtype != torch.float32:
                t = t.to(device=self._device, dtype=torch.float32)
            # [feature, column] -> the module's batch-first [column, feature]
            h = (t.T - self._in_mean) / self._in_std
            y = self._module(h) * self._out_std + self._out_mean
            y = y.T.contiguous().to(torch.float32)
            return bridge.from_torch(y, x)


class MlxColumnModel:
    """MLX runner: the manifest's MLP as ``mlx.nn`` layers (``mlp-npz`` or
    ``mlx-safetensors`` weights), fed the facade's ``mx.array`` directly."""

    def __init__(self, manifest: ColumnModelManifest):
        try:
            import mlx.core as mx
            import mlx.nn as nn
        except ImportError:
            raise ModelUnavailable("MLX is not installed: pip install 'recast-woof[gpu-mlx]' "
                                   "(Apple silicon only)") from None
        if manifest.format not in ("mlp-npz", "mlx-safetensors"):
            raise ModelUnavailable(f"a {manifest.format} artefact runs on PyTorch only (CUDA, or torch-MPS "
                                   "on Metal with backend = \"torch\"); export mlp-npz for the MLX runner")
        self.manifest = manifest
        self.backend = "mlx"
        self._mx = mx
        widths = [manifest.n_inputs, *manifest.architecture["hidden"], manifest.n_outputs]
        act = {"relu": nn.relu, "tanh": mx.tanh, "gelu": nn.gelu_approx, "silu": nn.silu}[
            manifest.architecture["activation"]]
        self._layers = [nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)]
        if manifest.format == "mlp-npz":
            for linear, (w, b) in zip(self._layers, mlp_layers(manifest)):
                linear.weight = mx.array(w)
                linear.bias = mx.array(b)
        else:
            for linear, (w, b) in zip(self._layers, _safetensors_layers(manifest, widths, mx)):
                linear.weight, linear.bias = w, b
        self._act = act
        self._in_mean = mx.array(manifest.input_mean)[:, None]
        self._in_std = mx.array(manifest.input_std)[:, None]
        self._out_mean = mx.array(manifest.output_mean)[:, None]
        self._out_std = mx.array(manifest.output_std)[:, None]

    def __call__(self, x):
        mx = self._mx
        h = (bridge.to_mlx(x) - self._in_mean) / self._in_std
        for index, linear in enumerate(self._layers):
            # [feature, column] layout: W @ h + b, no transposes of the columns
            h = mx.addmm(linear.bias[:, None], linear.weight, h)
            if index < len(self._layers) - 1:
                h = self._act(h)
        y = h * self._out_std + self._out_mean
        return bridge.from_mlx(y, x)


def _safetensors_layers(manifest: ColumnModelManifest, widths, mx):
    """``mlx-safetensors`` weights as ``[(w, b), ...]`` in layer order.

    ``mlx.nn.Module.save_weights`` names a ``Sequential``'s layers by
    position, activations included (``layers.0``, ``layers.2``, ...), so
    the linear layers are taken in ascending index order, whatever the
    gaps, and every shape is checked against the manifest's widths."""
    import re
    loaded = mx.load(str(manifest.weights))
    pattern = re.compile(r"^(?:.*\.)?(\d+)\.(weight|bias)$")
    found: dict[int, dict[str, object]] = {}
    for key, value in loaded.items():
        match = pattern.match(key)
        if match is None:
            raise ManifestError(f"{manifest.weights}: {key!r} is not a numbered layer's weight or bias")
        found.setdefault(int(match.group(1)), {})[match.group(2)] = value
    order = sorted(found)
    if len(order) != len(widths) - 1:
        raise ManifestError(f"{manifest.weights}: {len(order)} layers; the manifest's widths {widths} "
                            f"need {len(widths) - 1}")
    layers = []
    for position, index in enumerate(order):
        pair = found[index]
        if set(pair) != {"weight", "bias"}:
            raise ManifestError(f"{manifest.weights}: layer {index} needs both weight and bias")
        w, b = pair["weight"].astype(mx.float32), pair["bias"].astype(mx.float32)
        want = (widths[position + 1], widths[position])
        if tuple(w.shape) != want or tuple(b.shape) != (want[0],):
            raise ManifestError(f"{manifest.weights}: layer {index} has weight {tuple(w.shape)} bias "
                                f"{tuple(b.shape)}; the manifest needs {want}")
        layers.append((w, b))
    return layers


def default_backend() -> str:
    """The runner woof's own array backend implies."""
    try:
        from woof.backend import requested
        name = requested()
    except Exception:
        return "numpy"
    if name == "mlx":
        return "mlx"
    if name == "cuda":
        return "torch"
    return "numpy"


def load_column_model(source: str | Path | ColumnModelManifest, *, backend: str = "auto",
                      device: str | None = None) -> ColumnModel:
    """Load an artefact and return the runner for ``backend``."""
    manifest = source if isinstance(source, ColumnModelManifest) else load_manifest(source)
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}; got {backend!r}")
    if backend == "auto":
        backend = default_backend()
    if backend == "torch":
        return TorchColumnModel(manifest, device=device)
    if backend == "mlx":
        return MlxColumnModel(manifest)
    return NumpyColumnModel(manifest)
