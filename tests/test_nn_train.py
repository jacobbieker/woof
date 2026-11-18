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
