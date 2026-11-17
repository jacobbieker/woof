"""The column-model artefact: a manifest beside its weights.

A model directory holds ``manifest.json`` and the weights file it names::

    {
      "schema": "woof.nn.column-model/v1",
      "name": "ysu-heat-moisture-emulator",
      "format": "mlp-npz",                  # or torchscript, torch-export, mlx-safetensors
      "weights": "weights.npz",
      "weights_sha256": "<hex>",
      "inputs":  [{"name": "theta", "units": "K", "levels": 20}, ...],
      "outputs": [{"name": "dtheta", "units": "K s-1", "levels": 20}, ...],
      "normalisation": {"input_mean": [...], "input_std": [...],
                        "output_mean": [...], "output_std": [...]},
      "bounds": {"input_min": [...], "input_max": [...]},
      "output_limit": [...],                # optional, |output| per feature
      "architecture": {"kind": "mlp", "hidden": [64, 64], "activation": "tanh"},
      "training": {...}                     # free-form provenance
    }

Features are the inputs' levels concatenated in manifest order (a surface
field has ``levels = 1``); every per-feature list has that length.  Level
0 is the lowest model level.  The model sees ``(x - input_mean) /
input_std`` and its output is multiplied by ``output_std`` and offset by
``output_mean`` -- the runners apply both, so an artefact trained on
normalised data needs no wrapper.

``weights_sha256`` is checked on every load, and :meth:`identity` (the
weights and manifest hashes) is what a restart identity carries, so a
resume onto different weights is refused.  ``bounds`` is the range the
model was trained on; the safety layer refuses, clips or falls back
outside it (:mod:`woof.nn.safety`).

Formats: ``mlp-npz`` (``w0, b0, w1, b1, ...``, each ``w`` shaped ``(out,
in)``; runs on every backend, so one artefact gives the same model on CUDA
and Metal), ``mlx-safetensors`` (the same MLP's weights saved by
``mlx.nn.Module.save_weights``; Metal only), ``torchscript`` and
``torch-export`` (any module taking ``[column, n_inputs]`` normalised
float32 and returning ``[column, n_outputs]``; CUDA only).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "woof.nn.column-model/v1"
FORMATS = ("mlp-npz", "mlx-safetensors", "torchscript", "torch-export")
#: Formats whose architecture the manifest itself declares.
MLP_FORMATS = ("mlp-npz", "mlx-safetensors")
ACTIVATIONS = ("relu", "tanh", "gelu", "silu")


class ManifestError(ValueError):
    """The artefact cannot be used as written; the message names why."""


@dataclass(frozen=True)
class FieldSpec:
    """One named input or output: its units and how many levels it spans."""

    name: str
    units: str
    levels: int

    def as_json(self) -> dict:
        return {"name": self.name, "units": self.units, "levels": self.levels}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fields(raw: Any, key: str, source: Path) -> tuple[FieldSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise ManifestError(f"{source}: {key!r} must be a non-empty list of fields")
    out = []
    for item in raw:
        if not isinstance(item, dict) or set(item) - {"name", "units", "levels"}:
            raise ManifestError(f"{source}: each {key} entry takes name, units and levels; got {item!r}")
        name, units, levels = item.get("name"), item.get("units"), item.get("levels", 1)
        if not isinstance(name, str) or not name or not isinstance(units, str):
            raise ManifestError(f"{source}: {key} entry {item!r} needs a name and units")
        if not isinstance(levels, int) or isinstance(levels, bool) or levels < 1:
            raise ManifestError(f"{source}: {key} {name!r} levels must be a positive integer")
        out.append(FieldSpec(name, units, levels))
    names = [f.name for f in out]
    if len(set(names)) != len(names):
        raise ManifestError(f"{source}: {key} repeats a name: {names}")
    return tuple(out)


def _vector(raw: Any, length: int, label: str, source: Path, *, positive: bool = False) -> np.ndarray:
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ManifestError(f"{source}: {label} is not a list of numbers ({error})") from None
    if array.shape != (length,):
        raise ManifestError(f"{source}: {label} has {array.size} entries; the fields span {length}")
    if not np.all(np.isfinite(array)):
        raise ManifestError(f"{source}: {label} carries a non-finite value")
    if positive and not np.all(array > 0):
        raise ManifestError(f"{source}: {label} must be positive everywhere (a zero std divides by zero)")
    return array.astype(np.float32)


@dataclass(frozen=True)
class ColumnModelManifest:
    """A validated artefact.  Construct with :func:`load_manifest`."""

    path: Path
    name: str
    format: str
    weights: Path
    weights_sha256: str
    manifest_sha256: str
    inputs: tuple[FieldSpec, ...]
    outputs: tuple[FieldSpec, ...]
    input_mean: np.ndarray
    input_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray
    input_min: np.ndarray
    input_max: np.ndarray
    output_limit: np.ndarray | None
    architecture: Mapping[str, Any] | None
    training: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_inputs(self) -> int:
        return int(sum(f.levels for f in self.inputs))

    @property
    def n_outputs(self) -> int:
        return int(sum(f.levels for f in self.outputs))

    def input_slices(self) -> dict[str, slice]:
        return _slices(self.inputs)

    def output_slices(self) -> dict[str, slice]:
        return _slices(self.outputs)

    def identity(self) -> dict[str, str]:
        """What a restart identity binds: the weights and manifest bytes."""
        return {"schema": SCHEMA, "name": self.name, "format": self.format,
                "weights_sha256": self.weights_sha256,
                "manifest_sha256": self.manifest_sha256}


def _slices(fields: Sequence[FieldSpec]) -> dict[str, slice]:
    out, start = {}, 0
    for spec in fields:
        out[spec.name] = slice(start, start + spec.levels)
        start += spec.levels
    return out


def load_manifest(path: str | Path) -> ColumnModelManifest:
    """Read and validate ``manifest.json`` (or the directory holding it).

    Refuses, naming the defect: an unknown schema or format, fields whose
    level counts disagree with the statistics, a zero standard deviation,
    an inverted training range, an MLP without its architecture, and
    weights whose SHA-256 differs from the one recorded.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    try:
        payload = path.read_bytes()
        raw = json.loads(payload)
    except OSError as error:
        raise ManifestError(f"cannot read the column-model manifest {path}: {error}") from None
    except json.JSONDecodeError as error:
        raise ManifestError(f"{path} is not JSON: {error}") from None
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ManifestError(f"{path}: schema must be {SCHEMA!r}")
    known = {"schema", "name", "format", "weights", "weights_sha256", "inputs", "outputs",
             "normalisation", "bounds", "output_limit", "architecture", "training"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ManifestError(f"{path}: unknown key(s) {unknown}; known: {sorted(known)}")
    name, fmt = raw.get("name"), raw.get("format")
    if not isinstance(name, str) or not name:
        raise ManifestError(f"{path}: name must be a non-empty string")
    if fmt not in FORMATS:
        raise ManifestError(f"{path}: format must be one of {FORMATS}; got {fmt!r}")
    weights_name = raw.get("weights")
    if not isinstance(weights_name, str) or not weights_name:
        raise ManifestError(f"{path}: weights must name the weights file")
    weights = (path.parent / weights_name).resolve()
    if not weights.is_file():
        raise ManifestError(f"{path}: the weights file {weights} does not exist")
    recorded = raw.get("weights_sha256")
    actual = _sha256(weights)
    if recorded != actual:
        raise ManifestError(
            f"{path}: {weights.name} has SHA-256 {actual}, the manifest records {recorded}; "
            "a model whose weights changed under its manifest would run under the old name")
    inputs = _fields(raw.get("inputs"), "inputs", path)
    outputs = _fields(raw.get("outputs"), "outputs", path)
    n_in = sum(f.levels for f in inputs)
    n_out = sum(f.levels for f in outputs)
    norm = raw.get("normalisation")
    if not isinstance(norm, dict) or set(norm) != {"input_mean", "input_std", "output_mean", "output_std"}:
        raise ManifestError(f"{path}: normalisation takes input_mean, input_std, output_mean and output_std")
    bounds = raw.get("bounds")
    if not isinstance(bounds, dict) or set(bounds) != {"input_min", "input_max"}:
        raise ManifestError(f"{path}: bounds takes input_min and input_max (the trained input range)")
    input_min = _vector(bounds["input_min"], n_in, "bounds.input_min", path)
    input_max = _vector(bounds["input_max"], n_in, "bounds.input_max", path)
    if np.any(input_min > input_max):
        raise ManifestError(f"{path}: bounds.input_min exceeds bounds.input_max for some feature")
    limit = raw.get("output_limit")
    output_limit = None if limit is None else _vector(limit, n_out, "output_limit", path, positive=True)
    architecture = raw.get("architecture")
    if fmt in MLP_FORMATS:
        _check_architecture(architecture, path)
    elif architecture is not None and not isinstance(architecture, dict):
        raise ManifestError(f"{path}: architecture must be a table")
    training = raw.get("training") or {}
    if not isinstance(training, dict):
        raise ManifestError(f"{path}: training must be a table")
    return ColumnModelManifest(
        path=path.resolve(), name=name, format=fmt, weights=weights,
        weights_sha256=actual, manifest_sha256=hashlib.sha256(payload).hexdigest(),
        inputs=inputs, outputs=outputs,
        input_mean=_vector(norm["input_mean"], n_in, "normalisation.input_mean", path),
        input_std=_vector(norm["input_std"], n_in, "normalisation.input_std", path, positive=True),
        output_mean=_vector(norm["output_mean"], n_out, "normalisation.output_mean", path),
        output_std=_vector(norm["output_std"], n_out, "normalisation.output_std", path, positive=True),
        input_min=input_min, input_max=input_max, output_limit=output_limit,
        architecture=architecture, training=training)


def _check_architecture(architecture: Any, path: Path) -> None:
    if not isinstance(architecture, dict) or architecture.get("kind") != "mlp":
        raise ManifestError(f"{path}: an MLP format needs architecture = {{kind = \"mlp\", hidden, activation}}")
    hidden = architecture.get("hidden")
    if (not isinstance(hidden, list)
            or not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in hidden)):
        raise ManifestError(f"{path}: architecture.hidden must be a list of positive layer widths")
    if architecture.get("activation") not in ACTIVATIONS:
        raise ManifestError(f"{path}: architecture.activation must be one of {ACTIVATIONS}")


def mlp_layers(manifest: ColumnModelManifest) -> list[tuple[np.ndarray, np.ndarray]]:
    """The ``mlp-npz`` weights as ``[(w, b), ...]``, shapes checked."""
    if manifest.format != "mlp-npz":
        raise ManifestError(f"{manifest.path}: mlp_layers reads mlp-npz, not {manifest.format}")
    widths = [manifest.n_inputs, *manifest.architecture["hidden"], manifest.n_outputs]
    with np.load(manifest.weights, allow_pickle=False) as data:
        layers = []
        for index in range(len(widths) - 1):
            try:
                w = np.ascontiguousarray(data[f"w{index}"], dtype=np.float32)
                b = np.ascontiguousarray(data[f"b{index}"], dtype=np.float32)
            except KeyError:
                raise ManifestError(f"{manifest.weights}: layer {index} (w{index}, b{index}) is missing") from None
            if w.shape != (widths[index + 1], widths[index]) or b.shape != (widths[index + 1],):
                raise ManifestError(
                    f"{manifest.weights}: layer {index} has w{w.shape} b{b.shape}; the manifest's "
                    f"widths {widths} need w{(widths[index + 1], widths[index])}")
            if not (np.all(np.isfinite(w)) and np.all(np.isfinite(b))):
                raise ManifestError(f"{manifest.weights}: layer {index} carries a non-finite weight")
            layers.append((w, b))
        extra = sorted(set(data.files) - {f"{k}{i}" for i in range(len(widths) - 1) for k in "wb"})
    if extra:
        raise ManifestError(f"{manifest.weights}: arrays {extra} belong to no layer of {widths}")
    return layers


def write_mlp_artefact(directory: str | Path, *, name: str, inputs: Sequence[FieldSpec],
                       outputs: Sequence[FieldSpec], layers: Sequence[tuple[np.ndarray, np.ndarray]],
                       activation: str, input_mean, input_std, output_mean, output_std,
                       input_min, input_max, output_limit=None,
                       training: Mapping[str, Any] | None = None) -> ColumnModelManifest:
    """Write an ``mlp-npz`` artefact and return its validated manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "weights.npz"
    arrays = {}
    for index, (w, b) in enumerate(layers):
        arrays[f"w{index}"] = np.asarray(w, dtype=np.float32)
        arrays[f"b{index}"] = np.asarray(b, dtype=np.float32)
    np.savez(weights, **arrays)

    def listed(value):
        return None if value is None else [float(v) for v in np.asarray(value, dtype=np.float64).ravel()]

    manifest = {
        "schema": SCHEMA, "name": name, "format": "mlp-npz", "weights": weights.name,
        "weights_sha256": _sha256(weights),
        "inputs": [f.as_json() for f in inputs], "outputs": [f.as_json() for f in outputs],
        "normalisation": {"input_mean": listed(input_mean), "input_std": listed(input_std),
                          "output_mean": listed(output_mean), "output_std": listed(output_std)},
        "bounds": {"input_min": listed(input_min), "input_max": listed(input_max)},
        "architecture": {"kind": "mlp", "hidden": [int(np.asarray(w).shape[0]) for w, _ in layers[:-1]],
                         "activation": activation},
        "training": dict(training or {}),
    }
    if output_limit is not None:
        manifest["output_limit"] = listed(output_limit)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return load_manifest(directory)
