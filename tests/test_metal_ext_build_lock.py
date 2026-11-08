"""Concurrent first uses build woof's Metal launcher once (woof.backend.metal.ext.build).

Without the lock, processes starting together configured one shared CMake
tree and most of them failed CMake's compiler check (5 of 6 on an M3 Max).
"""

import importlib.util
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None or sys.platform != "darwin",
                                reason="needs mlx (Apple silicon)")

_LOAD = "import woof; from woof.backend.metal import ext; print(ext.load().__file__)"


def test_processes_starting_together_share_one_build(tmp_path):
    env = dict(os.environ, WOOF_METAL_EXT_DIR=str(tmp_path / "ext"))
    procs = [subprocess.Popen([sys.executable, "-c", _LOAD], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for _ in range(4)]
    results = [p.communicate(timeout=1800) + (p.returncode,) for p in procs]
    for out, err, code in results:
        assert code == 0, err[-2000:]
    paths = {out.strip().splitlines()[-1] for out, _, _ in results}
    assert len(paths) == 1
    (built,) = paths
    leftovers = [p.name for p in (tmp_path / "ext").glob("*/stage-*")]
    assert os.path.isfile(built) and not leftovers
