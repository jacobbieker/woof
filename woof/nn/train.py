"""Train a small column emulator on a tendency tap (offline, NumPy only).

``python -m woof.nn.train TAP OUT`` reads a store written by ``--nn`` in
``mode = "tap"`` (:mod:`woof.nn.tap`) and writes an ``mlp-npz`` artefact
(:mod:`woof.nn.manifest`) that runs on every backend.  The tap's targets
are the scheme's own tendency, so the artefact emulates the scheme: it is
for ``mode = "replace"``.  Under ``mode = "residual"`` its rate would be
added on top of the scheme's (a residual model needs a reference target
the tap does not record).  The trainer is deliberately small: a float32 MLP, Adam,
minibatches, a held-out tail of the record (the last ``holdout`` fraction
of recorded model times, so the score is out of sample in time; samples
are ordered by model time first, and a time recorded twice, as a resumed
run may leave it, keeps its last record), and nothing but
NumPy, so it runs wherever woof does.  It is the demonstration of the
loop, not a research trainer: anything that writes the same artefact may
take its place.

The artefact's statistics come from the training samples: the input and
output means and standard deviations (an input that never varies gets a
unit scale, so it passes through as an offset; an output level's scale is
floored at a thousandth of its field's largest level scale), the trained
input range (``bounds``, what the safety layer screens against), and
``output_limit``: ``limit_factor`` times the largest target magnitude seen
per output level, floored at a hundredth of the field's largest level
limit, so a level where training never saw a non-zero target (cloud
water above the cloud, say) is not clamped to zero online.
The manifest's ``training`` table records the tap, its schema, the split,
the hyper-parameters and the held-out scores.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from woof.nn import NOT_WRF_IDENTICAL
from woof.nn.manifest import write_mlp_artefact
from woof.nn.tap import read_tap


def _columns(array: np.ndarray) -> np.ndarray:
    """``[sample, feature, column]`` to ``[sample * column, feature]``."""
    s, f, c = array.shape
    return np.ascontiguousarray(array.transpose(0, 2, 1).reshape(s * c, f), dtype=np.float32)


def _scale(values: np.ndarray, fields=None) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and standard deviation.  An input that never
    varies gets a unit scale (it is a constant after centring).  An output
    is scaled within its field: a level whose target barely varies (cloud
    water above the cloud, say) takes a floor of a thousandth of the
    field's largest level scale, so the model's normalised error there is
    not multiplied into a physical rate on the scale of one."""
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    if fields is None:
        tiny = np.maximum(np.abs(mean) * 1e-6, 1e-30)
        std = np.where(std > tiny, std, 1.0)
        return mean.astype(np.float32), std.astype(np.float32)
    start = 0
    for spec in fields:
        rows = slice(start, start + spec.levels)
        start += spec.levels
        top = float(std[rows].max())
        std[rows] = np.maximum(std[rows], top * 1e-3) if top > 0.0 else 1e-30
    return mean.astype(np.float32), std.astype(np.float32)


def _by_time(data: dict) -> dict:
    """The record ordered by model time, a time recorded more than once
    (a run resumed from before its last recorded sample appends again)
    keeping its last record."""
    times = np.asarray(data["time_seconds"], dtype=np.float64)
    last = len(times) - 1 - np.unique(times[::-1], return_index=True)[1]
    if np.array_equal(last, np.arange(len(times))):
        return data
    out = dict(data)
    for name in ("inputs", "targets", "time_seconds", "step", "column_index"):
        out[name] = np.asarray(data[name])[last]
    return out


def _limits(limit: np.ndarray, fields) -> np.ndarray:
    """Per-level output limits floored at a hundredth of the field's
    largest level limit (1e-30 for a field never seen non-zero)."""
    limit = np.asarray(limit, dtype=np.float64).copy()
    start = 0
    for spec in fields:
        rows = slice(start, start + spec.levels)
        start += spec.levels
        top = float(limit[rows].max())
        limit[rows] = np.maximum(limit[rows], top * 1e-2)
    return np.maximum(limit, 1e-30)


def _act(name):
    if name == "tanh":
        return np.tanh, lambda h, a: 1.0 - a * a
    if name == "relu":
        return (lambda h: np.maximum(h, 0.0)), (lambda h, a: (h > 0).astype(np.float32))
    if name == "silu":
        def f(h):
            return h / (1.0 + np.exp(-h))

        def d(h, a):
            s = 1.0 / (1.0 + np.exp(-h))
            return s * (1.0 + h * (1.0 - s))
        return f, d
    raise ValueError(f"activation must be tanh, relu or silu; got {name!r}")


def r2_by_output(y: np.ndarray, p: np.ndarray, slices: dict) -> dict:
    """Coefficient of determination per named output over all its levels."""
    out = {}
    for name, rows in slices.items():
        a, b = y[:, rows], p[:, rows]
        ss = float(((a - a.mean(axis=0)) ** 2).sum())
        out[name] = None if ss == 0.0 else 1.0 - float(((a - b) ** 2).sum()) / ss
    return out


def train(tap, out, *, hidden=(128, 128), activation="tanh", epochs=30, batch=1024, lr=1e-3,
          holdout=0.2, seed=0, limit_factor=2.0, name=None, log=print) -> dict:
    """Train on the tap at ``tap``, write the artefact to ``out`` and
    return the held-out scores."""
    data = read_tap(tap)
    attrs = data["attrs"]
    data = _by_time(data)
    n = data["inputs"].shape[0]
    if n < 2:
        raise ValueError(f"{tap}: {n} recorded step(s); training needs at least two (one held out)")
    cut = min(n - 1, max(1, int(round(n * (1.0 - holdout)))))
    x_train, y_train = _columns(data["inputs"][:cut]), _columns(data["targets"][:cut])
    x_test, y_test = _columns(data["inputs"][cut:]), _columns(data["targets"][cut:])
    in_mean, in_std = _scale(x_train)
    out_mean, out_std = _scale(y_train, data["outputs_spec"])
    xn = (x_train - in_mean) / in_std
    yn = (y_train - out_mean) / out_std

    rng = np.random.default_rng(seed)
    widths = [xn.shape[1], *hidden, yn.shape[1]]
    layers = []
    for a, b in zip(widths[:-1], widths[1:]):
        w = (rng.standard_normal((b, a)) * np.sqrt(1.0 / a)).astype(np.float32)
        layers.append([w, np.zeros(b, dtype=np.float32)])
    moments = [[np.zeros_like(p) for p in layer] + [np.zeros_like(p) for p in layer] for layer in layers]
    f, df = _act(activation)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def forward(x):
        acts, pre = [x], []
        h = x
        for index, (w, b) in enumerate(layers):
            z = h @ w.T + b
            if index < len(layers) - 1:
                pre.append(z)
                h = f(z).astype(np.float32, copy=False)
            else:
                h = z
            acts.append(h)
        return acts, pre

    def predict(x):
        return forward((x - in_mean) / in_std)[0][-1] * out_std + out_mean

    start = time.perf_counter()
    for epoch in range(epochs):
        order = rng.permutation(xn.shape[0])
        total = 0.0
        for begin in range(0, order.size, batch):
            idx = order[begin:begin + batch]
            acts, pre = forward(xn[idx])
            grad = (acts[-1] - yn[idx]) * (2.0 / idx.size / yn.shape[1])
            total += float(((acts[-1] - yn[idx]) ** 2).mean()) * idx.size
            t += 1
            for index in range(len(layers) - 1, -1, -1):
                w, b = layers[index]
                gw = grad.T @ acts[index]
                gb = grad.sum(axis=0)
                if index > 0:
                    grad = (grad @ w) * df(pre[index - 1], acts[index])
                for k, (p, g) in enumerate(((w, gw), (b, gb))):
                    m, v = moments[index][k], moments[index][k + 2]
                    m *= beta1
                    m += (1 - beta1) * g
                    v *= beta2
                    v += (1 - beta2) * g * g
                    step = lr * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                    p -= (step * m / (np.sqrt(v) + eps)).astype(np.float32)
        test_mse = float(((((predict(x_test) - out_mean) / out_std) - (y_test - out_mean) / out_std) ** 2).mean())
        log(f"epoch {epoch + 1}/{epochs}: train mse {total / xn.shape[0]:.4f} (normalised), "
            f"held-out mse {test_mse:.4f}")
    seconds = time.perf_counter() - start

    from woof.nn.manifest import _slices
    outputs = data["outputs_spec"]
    slices = _slices(outputs)
    scores = {"held_out_r2": r2_by_output(y_test, predict(x_test), slices),
              "train_r2": r2_by_output(y_train, predict(x_train), slices),
              "held_out_columns": int(x_test.shape[0]), "train_columns": int(x_train.shape[0])}
    limit = _limits(np.abs(y_train).max(axis=0) * limit_factor, outputs)
    training = {"tap": str(Path(tap).resolve()), "tap_schema": attrs.get("schema"),
                "route": attrs.get("route"), "scheme": attrs.get("scheme"), "grid": attrs.get("grid"),
                "target": "the scheme's tendency (an emulator: mode = \"replace\")",
                "steps_train": cut, "steps_held_out": n - cut,
                "held_out_from_seconds": float(data["time_seconds"][cut]), "hidden": list(hidden),
                "activation": activation, "epochs": epochs, "batch": batch, "lr": lr, "seed": seed,
                "seconds": round(seconds, 2), "note": NOT_WRF_IDENTICAL, **scores}
    write_mlp_artefact(
        out, name=name or f"{attrs.get('scheme', 'column')}-mlp", inputs=data["inputs_spec"],
        outputs=outputs, layers=[(w, b) for w, b in layers], activation=activation,
        input_mean=in_mean, input_std=in_std, output_mean=out_mean, output_std=out_std,
        input_min=x_train.min(axis=0), input_max=x_train.max(axis=0), output_limit=limit,
        training=training)
    return training


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tap", help="the tap store (--nn mode = \"tap\")")
    parser.add_argument("out", help="the artefact directory to write")
    parser.add_argument("--hidden", default="128,128", help="hidden layer widths (default 128,128)")
    parser.add_argument("--activation", default="tanh", choices=("tanh", "relu", "silu"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--holdout", type=float, default=0.2,
                        help="the last fraction of recorded steps kept out of training (default 0.2)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)
    hidden = tuple(int(v) for v in args.hidden.split(",") if v)
    result = train(args.tap, args.out, hidden=hidden, activation=args.activation, epochs=args.epochs,
                   batch=args.batch, lr=args.lr, holdout=args.holdout, seed=args.seed, name=args.name)
    print(json.dumps({k: result[k] for k in ("held_out_r2", "train_r2", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
