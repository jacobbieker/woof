"""Every CuPy attribute woof uses resolves on the mlx facade, or is listed
with its reason in tools/cupy_surface_allow.txt.

Breakage it prevents: new code reaching for a CuPy function the facade does
not implement passes every CUDA gate and fails on a Mac at its first call
(BackendUnsupported).  The census parses woof's sources; it imports only the
facade module, never CuPy or a GPU.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="the facade's array module needs mlx")
def test_every_cupy_path_woof_uses_resolves_or_is_allowed(capsys):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import cupy_surface_census
    finally:
        sys.path.remove(str(ROOT / "tools"))
    status = cupy_surface_census.main(["--gate"])
    assert status == 0, capsys.readouterr().out
