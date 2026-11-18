"""The default path imports no NN runtime (plan section 8).

A configuration that names no NN option must not import ``woof.nn``,
``torch`` or ``mlx.nn``: importing torch alone loads CUDA libraries of its
own, and an import is the first step to a launch.  Each door's module
graph is imported in a fresh interpreter and the three names are looked
up in ``sys.modules``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The modules a forecast's front doors import before stepping.  A module
#: that cannot import on this host (no device, no CuPy) is reported and
#: skipped inside the child; the CPU-safe ones always run.
DOORS = ("woof.cli", "woof.config", "woof.runtime", "woof.experiment", "woof.core.dycore",
         "woof.core.physics", "woof.hex.cli", "woof.hex.forecast_door", "woof.hex.mpasphys.runtime_v842",
         "woof.hex.mpasphys.moving_v842", "woof.hex.mpasphys.checkpoint_v842", "woof.hex.physics_seam",
         "woof.hex.physics_backend_admission", "woof.io.restart")

_CHILD = r"""
import importlib, json, sys
done, failed = [], {}
for name in json.loads(sys.argv[1]):
    try:
        importlib.import_module(name)
        done.append(name)
    except Exception as error:   # a door this host cannot import
        failed[name] = f"{type(error).__name__}: {error}"[:200]
loaded = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch.")
                or m == "mlx.nn" or m.startswith("mlx.nn.") or m == "woof.nn" or m.startswith("woof.nn."))
print(json.dumps({"done": done, "failed": failed, "loaded": loaded}))
"""


def test_default_doors_import_no_nn_runtime():
    result = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(DOORS)], cwd=ROOT,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert {"woof.cli", "woof.config", "woof.hex.physics_seam", "woof.hex.cli",
            "woof.hex.forecast_door"} <= set(report["done"]), report["failed"]
    assert report["loaded"] == [], (
        f"importing the default doors loaded {report['loaded']}; an NN runtime must load only when "
        "a configuration names an NN option")
