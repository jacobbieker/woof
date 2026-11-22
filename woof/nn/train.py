"""Train a small column emulator on a tendency tap (offline, NumPy only).

``python -m woof.nn.train TAP OUT`` reads a store written by ``--nn`` in
``mode = "tap"`` (:mod:`woof.nn.tap`) and writes an ``mlp-npz`` artefact
(:mod:`woof.nn.manifest`) that runs on every backend.  A tap's targets
are by default the scheme's own tendency, so the artefact emulates the
scheme: it is for ``mode = "replace"``.  A hex tap recorded with
``tap_target = "nudging"`` under analysis nudging holds the nudging's
increment to the same sources instead -- what the model lacked to stay on
the analyses -- and the artefact trained on it is a correction for
``mode = "residual"``.  The manifest's ``training.intended_mode`` says
which, and the hex seat refuses the other mode.  The trainer is deliberately small: a float32 MLP, Adam,
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

Pooling.  Several taps (other cycles, seasons or meshes) train one model:
``python -m woof.nn.train TAP [TAP ...] OUT``.  Their fields must agree.
Each store is ordered and split on its own (the last ``holdout`` of its
recorded times is held out), and the training parts are pooled.  A store
named with ``--validate`` is held out whole: a different cycle, so the
score is out of sample in weather as well as in time; ``holdout`` then
defaults to 0.  ``--inputs`` keeps a subset of the tap's input fields and
``--levels`` the lowest levels of every 3-D input and output, so one broad
tap serves several models.  The held-out report adds, per held-out store
and for each ``--margins`` value, the fraction of its columns outside the
trained input range widened by that margin (what ``out_of_range =
"fallback"`` would leave to the scheme) and the input fields most often
outside it at the smallest margin (the fraction of columns with any level
of the field outside).  A store may be named only once: a ``--validate``
store is never also trained on.  With ``--keep-best`` the epoch is chosen on
the held-out stores, so their scores are those of the selection
(``selected_on_held_out``); score a further cycle for an unbiased one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from woof.nn import NOT_WRF_IDENTICAL
from woof.nn.manifest import FieldSpec, write_mlp_artefact
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


def _select(data: dict, inputs=None, levels=None) -> dict:
    """``data`` with only the input fields ``inputs`` (tap order kept)
    and the lowest ``levels`` levels of every multi-level field."""
    if inputs is None and levels is None:
        return data
    if levels is not None and int(levels) < 1:
        raise ValueError(f"--levels {levels}: keep at least one level")
    if inputs is not None and not inputs:
        raise ValueError("--inputs names no field")
    specs = data["inputs_spec"]
    known = [f.name for f in specs]
    if inputs is not None:
        unknown = [n for n in inputs if n not in known]
        if unknown:
            raise ValueError(f"--inputs {unknown} are not in the tap's inputs {known}")

    def rows(fields, keep):
        out, kept, start = [], [], 0
        for spec in fields:
            depth = spec.levels if levels is None or spec.levels == 1 else min(spec.levels, int(levels))
            if keep is None or spec.name in keep:
                out.extend(range(start, start + depth))
                kept.append(FieldSpec(spec.name, spec.units, depth))
            start += spec.levels
        return np.asarray(out, dtype=np.int64), tuple(kept)

    in_rows, in_spec = rows(specs, None if inputs is None else set(inputs))
    out_rows, out_spec = rows(data["outputs_spec"], None)
    out = dict(data)
    out["inputs"] = np.asarray(data["inputs"])[:, in_rows]
    out["targets"] = np.asarray(data["targets"])[:, out_rows]
    out["inputs_spec"], out["outputs_spec"] = in_spec, out_spec
    return out


def _load(path, inputs, levels) -> dict:
    data = read_tap(path)
    attrs = data["attrs"]
    data = _select(_by_time(data), inputs, levels)
    data["attrs"] = attrs
    data["path"] = str(Path(path).resolve())
    return data


def _agree(stores) -> None:
    """Pooled stores must carry the same fields, route, scheme and target."""
    first = stores[0]
    for other in stores[1:]:
        for key in ("inputs_spec", "outputs_spec"):
            if other[key] != first[key]:
                raise ValueError(f"{other['path']}: its {key.split('_')[0]} {other[key]} differ from "
                                 f"{first['path']}'s {first[key]}; pooled taps must record the same fields")
        for key in ("route", "scheme", "target"):
            a, b = first["attrs"].get(key), other["attrs"].get(key)
            if key == "target":
                a, b = a or "physics", b or "physics"
            if a != b:
                raise ValueError(f"{other['path']}: {key} {b!r} differs from {first['path']}'s {a!r}")


def fallback_fractions(x: np.ndarray, lo, hi, margins, names=None, top: int = 5) -> dict:
    """The fraction of columns (rows of ``x``, ``[column, feature]``)
    outside ``[lo, hi]`` widened by each margin (the safety screen's own
    bounds), and the ``top`` input fields most often outside at the
    smallest margin: the fraction of columns with any level of the field
    outside."""
    from woof.nn.safety import _bounds

    if not len(margins):
        raise ValueError("name at least one margin")
    if not x.shape[0]:
        raise ValueError("no held-out column to score")
    xt = np.ascontiguousarray(x.T)

    def outside(margin):
        low, high = _bounds(np, lo, hi, float(margin))
        return (xt < low) | (xt > high) | ~np.isfinite(xt)      # [feature, column]

    out = {"fraction": {f"{float(m):g}": float(outside(m).any(axis=0).mean()) for m in margins}}
    if names is not None:
        bad = outside(min(float(m) for m in margins))
        by_field: dict = {}
        for name, row in zip(names, bad):
            by_field[name] = row if name not in by_field else (by_field[name] | row)
        counts = {name: int(row.sum()) for name, row in by_field.items()}
        worst = sorted(((n, name) for name, n in counts.items() if n), reverse=True)[:top]
        out["most_often_outside"] = {name: n / x.shape[0] for n, name in worst}
    return out


def train(tap, out, *, hidden=(128, 128), activation="tanh", epochs=30, batch=1024, lr=1e-3,
          holdout=None, seed=0, limit_factor=2.0, name=None, validate=(), inputs=None, levels=None,
          margins=(0.0, 0.1, 0.3), keep_best=False, log=print) -> dict:
    """Train on the tap(s) at ``tap``, write the artefact to ``out`` and
    return the held-out scores.  ``validate`` names stores held out whole;
    ``holdout`` (default 0.2, or 0 with ``validate``) is the tail of each
    training store held out.  ``keep_best`` keeps the epoch with the
    lowest held-out error instead of the last (the held-out scores are
    then those of the selection, recorded as ``selected_on_held_out``)."""
    taps = [tap] if isinstance(tap, (str, Path)) else list(tap)
    if not taps:
        raise ValueError("name at least one tap store to train on")
    validate = [validate] if isinstance(validate, (str, Path)) else list(validate)
    if holdout is None:
        holdout = 0.0 if validate else 0.2
    if not len(margins):
        raise ValueError("name at least one margin")
    resolved = [str(Path(p).resolve()) for p in (*taps, *validate)]
    repeated = sorted({p for p in resolved if resolved.count(p) > 1})
    if repeated:
        raise ValueError(f"{repeated}: a store is named twice (a --validate store must not also be trained on)")
    stores = [_load(path, inputs, levels) for path in taps]
    checks = [_load(path, inputs, levels) for path in validate]
    _agree(stores + checks)
    data = stores[0]
    attrs = data["attrs"]
    x_parts, y_parts, held = [], [], []
    splits = []
    for store in stores:
        n = store["inputs"].shape[0]
        if holdout > 0.0:
            if n < 2:
                raise ValueError(f"{store['path']}: {n} recorded step(s); training needs at least two "
                                 "(one held out)")
            cut = min(n - 1, max(1, int(round(n * (1.0 - holdout)))))
        else:
            if n < 1:
                raise ValueError(f"{store['path']}: no recorded step to train on")
            cut = n
        store["recorded_steps"] = n
        x_parts.append(_columns(store["inputs"][:cut]))
        y_parts.append(_columns(store["targets"][:cut]))
        if cut < n:
            held.append((f"{store['path']} (last {n - cut} steps)", _columns(store["inputs"][cut:]),
                         _columns(store["targets"][cut:])))
        splits.append({"tap": store["path"], "grid": store["attrs"].get("grid"), "steps_train": cut,
                       "steps_held_out": n - cut,
                       "held_out_from_seconds": (float(store["time_seconds"][cut]) if cut < n else None)})
        store["inputs"] = store["targets"] = None            # the columns above are all training needs
    for store in checks:
        if store["inputs"].shape[0] < 1:
            raise ValueError(f"{store['path']}: no recorded step to validate on")
        held.append((f"{store['path']} (whole)", _columns(store["inputs"]), _columns(store["targets"])))
        store["inputs"] = store["targets"] = None
    if not held:
        raise ValueError("nothing is held out: give holdout > 0 or a --validate store")
    x_train, y_train = np.concatenate(x_parts), np.concatenate(y_parts)
    del x_parts, y_parts
    bounds = np.cumsum([0] + [h[1].shape[0] for h in held])
    x_test = np.concatenate([h[1] for h in held])
    y_test = np.concatenate([h[2] for h in held])
    held = [(label, slice(int(a), int(b))) for (label, _, _), a, b in zip(held, bounds[:-1], bounds[1:])]
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

    def predict(x, chunk=65536):
        parts = [forward((x[i:i + chunk] - in_mean) / in_std)[0][-1] * out_std + out_mean
                 for i in range(0, x.shape[0], chunk)]
        return np.concatenate(parts) if parts else np.zeros((0, len(out_mean)), dtype=np.float32)

    best = (np.inf, None, 0)
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
        if keep_best and test_mse < best[0]:
            best = (test_mse, [[w.copy(), b.copy()] for w, b in layers], epoch + 1)
        log(f"epoch {epoch + 1}/{epochs}: train mse {total / xn.shape[0]:.4f} (normalised), "
            f"held-out mse {test_mse:.4f}")
    seconds = time.perf_counter() - start
    if keep_best and best[1] is not None:
        layers[:] = best[1]

    from woof.nn.manifest import _slices
    outputs = data["outputs_spec"]
    slices = _slices(outputs)
    lo, hi = x_train.min(axis=0), x_train.max(axis=0)
    names = [spec.name for spec in data["inputs_spec"] for _ in range(spec.levels)]
    y_hat = predict(x_test)
    per_store = {label: {"r2": r2_by_output(y_test[part], y_hat[part], slices),
                         "columns": int(part.stop - part.start),
                         "fallback": fallback_fractions(x_test[part], lo, hi, margins, names)}
                 for label, part in held}
    scores = {"held_out_r2": r2_by_output(y_test, y_hat, slices),
              "train_r2": r2_by_output(y_train, predict(x_train), slices),
              "held_out_columns": int(x_test.shape[0]), "train_columns": int(x_train.shape[0]),
              "held_out": per_store}
    limit = _limits(np.abs(y_train).max(axis=0) * limit_factor, outputs)
    target = attrs.get("target") or "physics"
    mode = "residual" if target == "nudging" else "replace"
    what = ("the analysis nudging's increment (a correction: mode = \"residual\")" if target == "nudging"
            else "the scheme's tendency (an emulator: mode = \"replace\")")
    training = {"tap": data["path"] if len(stores) == 1 else [s["path"] for s in stores],
                "validate": [s["path"] for s in checks], "tap_schema": attrs.get("schema"),
                "route": attrs.get("route"), "scheme": attrs.get("scheme"), "grid": attrs.get("grid"),
                "grids": sorted({str(s["attrs"].get("grid")) for s in stores}),
                "target": what, "intended_mode": mode, "splits": splits,
                "steps_train": int(sum(s["steps_train"] for s in splits)),
                "steps_held_out": int(sum(s["steps_held_out"] for s in splits)),
                "held_out_from_seconds": splits[0]["held_out_from_seconds"], "hidden": list(hidden),
                "activation": activation, "epochs": epochs, "batch": batch, "lr": lr, "seed": seed,
                "holdout": holdout,
                "levels": None if levels is None else max(s.levels for s in (*data["inputs_spec"], *data["outputs_spec"])), "margins": [float(m) for m in margins],
                "kept_epoch": best[2] if keep_best and best[1] is not None else epochs,
                "selected_on_held_out": bool(keep_best),
                "seconds": round(seconds, 2), "note": NOT_WRF_IDENTICAL, **scores}
    write_mlp_artefact(
        out, name=name or f"{attrs.get('scheme', 'column')}-mlp", inputs=data["inputs_spec"],
        outputs=outputs, layers=[(w, b) for w, b in layers], activation=activation,
        input_mean=in_mean, input_std=in_std, output_mean=out_mean, output_std=out_std,
        input_min=lo, input_max=hi, output_limit=limit, training=training)
    return training


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", metavar="TAP [TAP ...] OUT",
                        help="the tap store(s) (--nn mode = \"tap\"), then the artefact directory to write")
    parser.add_argument("--validate", action="append", default=[], metavar="TAP",
                        help="a tap store held out whole (another cycle); repeat for several")
    parser.add_argument("--inputs", default=None, help="comma-separated subset of the tap's input fields")
    parser.add_argument("--levels", type=int, default=None,
                        help="keep the lowest LEVELS levels of every 3-D input and output")
    parser.add_argument("--hidden", default="128,128", help="hidden layer widths (default 128,128)")
    parser.add_argument("--activation", default="tanh", choices=("tanh", "relu", "silu"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--holdout", type=float, default=None,
                        help="the last fraction of each store's recorded steps kept out of training "
                             "(default 0.2, or 0 with --validate)")
    parser.add_argument("--margins", default="0,0.1,0.3",
                        help="range margins at which to report held-out fallback (default 0,0.1,0.3)")
    parser.add_argument("--keep-best", action="store_true",
                        help="keep the epoch with the lowest held-out error, not the last")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    args = parser.parse_intermixed_args(argv)
    if len(args.paths) < 2:
        parser.error("name at least one tap store and the artefact directory")
    *taps, out = args.paths
    hidden = tuple(int(v) for v in args.hidden.split(",") if v)
    inputs = None if args.inputs is None else tuple(v for v in args.inputs.split(",") if v)
    margins = tuple(float(v) for v in args.margins.split(",") if v)
    result = train(taps, out, hidden=hidden, activation=args.activation, epochs=args.epochs,
                   batch=args.batch, lr=args.lr, holdout=args.holdout, seed=args.seed, name=args.name,
                   validate=args.validate, inputs=inputs, levels=args.levels, margins=margins,
                   keep_best=args.keep_best)
    print(json.dumps({k: result[k] for k in ("held_out_r2", "train_r2", "held_out", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
