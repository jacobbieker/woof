"""The cpu backend leaves ``import cupy`` alone until its facade exists.

Breakage this prevents: on a CuPy-less Linux machine the default backend
is ``cpu``, and installing the facade finder for it made every ``import
cupy`` raise BackendUnsupported (an AttributeError) instead of ImportError,
so ``pytest.importorskip("cupy")`` and ``except ImportError`` guards stopped
skipping -- 487 failures in the CPU tier of CI.
"""
import subprocess
import sys

PROBE = """
import sys
from woof import backend
assert backend.install("cpu") == "cpu"
assert backend.active() == "cpu"
assert not any(type(f).__name__ == "_FacadeFinder" for f in sys.meta_path)
print("ok")
"""


def test_cpu_backend_installs_no_cupy_facade():
    env = {"PATH": "", "WOOF_BACKEND": "cpu"}
    result = subprocess.run([sys.executable, "-c", PROBE], capture_output=True,
                            text=True, env=env, timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
