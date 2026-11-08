"""With no [external_ocean] table the coupling costs nothing.

#172 put WRF's cpl_rcv/cpl_snd calls into PhysicsDriver.compute and the
dycore epilogue, each behind ``getattr(driver, "external_ocean", None)``.
Absent the table the module holding them (woof.ocean.external) is never
imported and the driver carries no external_ocean, so every hook falls
through: no array is allocated and no kernel launched.  This runs four
dycore steps of the small mixed land/water case (shortwave on) in a clean
subprocess and checks exactly that.  Bitwise identity against the commit
before #172 was measured once (CUDA and Metal, default and strict
numerics; recorded on the PR), not pinned here, so unrelated physics
changes do not move this test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import requires_gpu


@pytest.mark.gpu
@requires_gpu
def test_no_external_ocean_table_never_loads_or_attaches_the_coupling():
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, str(root / "tests" / "external_ocean_absent_probe.py")],
                         cwd=root, capture_output=True, text=True, env=dict(os.environ),
                         timeout=1800)
    assert out.returncode == 0, out.stderr[-2000:]
    _digest, imported, attached = out.stdout.split()[-3:]
    assert imported == "-", "woof.ocean.external was imported without an [external_ocean]"
    assert attached == "False", "the driver carries an external_ocean without the table"
