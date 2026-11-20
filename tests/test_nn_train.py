"""woof.nn P2.5: the demonstration trainer reads a tap and writes an artefact
the seat runs; it learns a function it can represent."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from woof.nn.manifest import FieldSpec, load_manifest
from woof.nn.models import load_column_model

pytestmark = pytest.mark.skipif(importlib.util.find_spec("zarr") is None, reason="zarr not installed")


def test_train_learns_a_smooth_column_map_and_writes_a_runnable_artefact(tmp_path):
    from woof.nn.tap import TendencyTap
    from woof.nn.train import train

    rng = np.random.default_rng(0)
    ins = (FieldSpec("theta", "K", 4), FieldSpec("hfx", "W m-2", 1))
    outs = (FieldSpec("dtheta", "K s-1", 4),)
    tap = TendencyTap(tmp_path / "rec.zarr", inputs=ins, outputs=outs, columns=64,
                      attrs={"route": "wrf", "scheme": "pbl", "grid": [4, 8, 8]}, chunk=8)
    mix = rng.standard_normal((4, 5)).astype(np.float32) * 0.3

    def target(x):
        z = (x - np.array([300, 301, 302, 303, 100], dtype=np.float32)[:, None]) / np.float32(5.0)
        return np.float32(1e-4) * np.tanh(mix @ z)

    for step in range(20):
        x = np.vstack([300 + np.arange(4)[:, None] + 5 * rng.standard_normal((4, 64)),
                       100 + 50 * rng.standard_normal((1, 64))]).astype(np.float32)
        tap.append(x, target(x), time_seconds=60.0 * step, step=step, column_index=np.arange(64))
    tap.close()

    result = train(tmp_path / "rec.zarr", tmp_path / "model", hidden=(32,), epochs=60, batch=128,
                   lr=3e-3, log=lambda *_: None)
    assert result["held_out_r2"]["dtheta"] > 0.9, result
    manifest = load_manifest(tmp_path / "model")
    assert manifest.inputs == ins and manifest.outputs == outs
    assert manifest.training["steps_held_out"] == 4 and manifest.output_limit is not None
    model = load_column_model(manifest, backend="numpy")
    x = np.vstack([300 + np.arange(4)[:, None] + 5 * rng.standard_normal((4, 16)),
                   100 + 50 * rng.standard_normal((1, 16))]).astype(np.float32)
    error = np.sqrt(np.mean((model(x) - target(x)) ** 2) / np.mean(target(x) ** 2))
    assert error < 0.3, error  # the artefact runs the trained map, not a fresh one


def test_train_refuses_a_single_step_record(tmp_path):
    from woof.nn.tap import TendencyTap
    from woof.nn.train import train

    tap = TendencyTap(tmp_path / "one.zarr", inputs=(FieldSpec("hfx", "W m-2", 1),),
                      outputs=(FieldSpec("dtheta", "K s-1", 1),), columns=2)
    tap.append(np.ones((1, 2)), np.ones((1, 2)), time_seconds=0.0, step=0, column_index=np.arange(2))
    tap.close()
    with pytest.raises(ValueError, match="at least two"):
        train(tmp_path / "one.zarr", tmp_path / "m", log=lambda *_: None)


def test_train_orders_by_time_and_keeps_the_last_record_of_a_repeated_time():
    from woof.nn.train import _by_time

    data = {"inputs": np.arange(5.0)[:, None, None], "targets": np.arange(5.0)[:, None, None] * 10,
            "time_seconds": np.array([0.0, 60.0, 120.0, 60.0, 120.0]), "step": np.arange(5),
            "column_index": np.zeros((5, 1), dtype=np.int64)}
    out = _by_time(data)
    assert out["time_seconds"].tolist() == [0.0, 60.0, 120.0]
    assert out["inputs"][:, 0, 0].tolist() == [0.0, 3.0, 4.0]  # a resumed run's records win
    ordered = {**data, "time_seconds": np.arange(5.0)}
    assert _by_time(ordered) is ordered  # an ordered record is used as it is


def test_output_limits_are_floored_within_each_field():
    from woof.nn.train import _limits

    fields = (FieldSpec("dqc", "kg kg-1 s-1", 3), FieldSpec("dqi", "kg kg-1 s-1", 2))
    limit = _limits(np.array([4e-6, 0.0, 1e-9, 0.0, 0.0]), fields)
    assert limit[0] == 4e-6 and limit[1] == limit[2] == 4e-8  # a level never seen non-zero is not clamped to 0
    assert np.all(limit[3:] == 1e-30)  # a field never seen non-zero stays at the floor


def _record(path, rng, *, steps, shift=0.0, columns=32, inputs=None, outputs=None, attrs=None):
    from woof.nn.tap import TendencyTap

    ins = inputs or (FieldSpec("theta", "K", 3), FieldSpec("qc", "kg kg-1", 3), FieldSpec("hfx", "W m-2", 1))
    outs = outputs or (FieldSpec("dtheta", "K s-1", 3),)
    tap = TendencyTap(path, inputs=ins, outputs=outs, columns=columns,
                      attrs={"route": "hex", "scheme": "total", "grid": [3, 99], **(attrs or {})}, chunk=4)
    for step in range(steps):
        x = np.vstack([300 + shift + 3 * rng.standard_normal((3, columns)),
                       np.abs(rng.standard_normal((3, columns))) * 1e-4,
                       100 + 30 * rng.standard_normal((1, columns))]).astype(np.float32)
        if inputs is not None:
            x = np.vstack([x[:3], x[6:]])
        y = (1e-5 * (x[:3] - 300 - shift)).astype(np.float32)
        tap.append(x, y, time_seconds=60.0 * step, step=step, column_index=np.arange(columns))
    tap.close()
    return path


def test_pooled_taps_train_one_model_scored_on_a_whole_held_out_store(tmp_path):
    from woof.nn.train import train

    rng = np.random.default_rng(1)
    a = _record(tmp_path / "a.zarr", rng, steps=6, shift=-5.0)
    b = _record(tmp_path / "b.zarr", rng, steps=6, shift=5.0)
    v = _record(tmp_path / "v.zarr", rng, steps=4, shift=0.0)
    result = train([a, b], tmp_path / "m", validate=[v], hidden=(16,), epochs=5, batch=64,
                   log=lambda *_: None)
    # --validate holds the other store out whole and every training step is used
    assert result["holdout"] == 0.0 and result["steps_train"] == 12 and result["steps_held_out"] == 0
    assert result["train_columns"] == 12 * 32 and result["held_out_columns"] == 4 * 32
    assert [s["steps_train"] for s in result["splits"]] == [6, 6]
    (label, report), = result["held_out"].items()
    assert label.endswith("(whole)") and set(report["fallback"]["fraction"]) == {"0", "0.1", "0.3"}
    manifest = load_manifest(tmp_path / "m")
    # the trained range spans both pooled stores
    assert manifest.input_min[0] < 290 and manifest.input_max[0] > 310


def test_fallback_fractions_count_columns_outside_the_widened_range():
    from woof.nn.train import fallback_fractions

    x = np.array([[0.0, 0.5], [1.0, 0.5], [1.05, 0.5], [0.5, 2.0]])   # [column, feature]
    out = fallback_fractions(x, np.array([0.0, 0.0]), np.array([1.0, 1.0]), (0.0, 0.1), names=["a", "b"])
    assert out["fraction"] == {"0": 0.5, "0.1": 0.25}
    assert out["most_often_outside"] == {"a": 0.25, "b": 0.25}
    # a multi-level field counts a column once, however many of its levels are outside
    x3 = np.array([[2.0, 2.0, 2.0], [0.5, 0.5, 0.5]])
    out3 = fallback_fractions(x3, np.zeros(3), np.ones(3), (0.1, 0.0), names=["t", "t", "t"])
    assert out3["most_often_outside"] == {"t": 0.5}
    with pytest.raises(ValueError, match="margin"):
        fallback_fractions(x3, np.zeros(3), np.ones(3), (), names=["t", "t", "t"])


def test_inputs_and_levels_select_from_a_broad_tap(tmp_path):
    from woof.nn.train import train

    rng = np.random.default_rng(2)
    a = _record(tmp_path / "a.zarr", rng, steps=6)
    train(a, tmp_path / "m", inputs=("theta", "hfx"), levels=2, hidden=(8,), epochs=2, batch=64,
          log=lambda *_: None)
    manifest = load_manifest(tmp_path / "m")
    assert manifest.inputs == (FieldSpec("theta", "K", 2), FieldSpec("hfx", "W m-2", 1))
    assert manifest.outputs == (FieldSpec("dtheta", "K s-1", 2),)
    with pytest.raises(ValueError, match="not in the tap's inputs"):
        train(a, tmp_path / "n", inputs=("rho",), epochs=1, log=lambda *_: None)
    with pytest.raises(ValueError, match="at least one level"):
        train(a, tmp_path / "n", levels=0, epochs=1, log=lambda *_: None)
    with pytest.raises(ValueError, match="names no field"):
        train(a, tmp_path / "n", inputs=(), epochs=1, log=lambda *_: None)


def test_pooled_taps_must_record_the_same_fields(tmp_path):
    from woof.nn.train import train

    rng = np.random.default_rng(3)
    a = _record(tmp_path / "a.zarr", rng, steps=3)
    b = _record(tmp_path / "b.zarr", rng, steps=3, inputs=(FieldSpec("theta", "K", 3), FieldSpec("hfx", "W m-2", 1)))
    with pytest.raises(ValueError, match="pooled taps must record the same fields"):
        train([a, b], tmp_path / "m", epochs=1, log=lambda *_: None)
    c = _record(tmp_path / "c.zarr", rng, steps=3, attrs={"scheme": "pbl"})
    with pytest.raises(ValueError, match="scheme"):
        train([a, c], tmp_path / "m", epochs=1, log=lambda *_: None)


def test_main_takes_several_taps_then_the_artefact(tmp_path, capsys):
    from woof.nn.train import main

    rng = np.random.default_rng(4)
    a = _record(tmp_path / "a.zarr", rng, steps=4)
    b = _record(tmp_path / "b.zarr", rng, steps=4)
    assert main([str(a), str(b), str(tmp_path / "m"), "--hidden", "8", "--epochs", "2",
                 "--keep-best", "--margins", "0,0.5"]) == 0
    manifest = load_manifest(tmp_path / "m")
    assert len(manifest.training["tap"]) == 2 and manifest.training["holdout"] == 0.2
    assert 1 <= manifest.training["kept_epoch"] <= 2


def test_a_store_is_named_once_and_selection_is_recorded(tmp_path):
    from woof.nn.train import main, train

    rng = np.random.default_rng(5)
    a = _record(tmp_path / "a.zarr", rng, steps=4)
    b = _record(tmp_path / "b.zarr", rng, steps=4)
    with pytest.raises(ValueError, match="named twice"):
        train([a, b], tmp_path / "m", validate=[a], epochs=1, log=lambda *_: None)
    # options may sit between the taps and the artefact
    assert main([str(a), "--validate", str(b), "--hidden", "8", str(tmp_path / "m"), "--epochs", "2",
                 "--keep-best"]) == 0
    training = load_manifest(tmp_path / "m").training
    assert training["selected_on_held_out"] is True and training["validate"] == [str(b.resolve())]
