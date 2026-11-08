"""With no [external_ocean] table the driver is unchanged, word for word.

#172 put WRF's cpl_rcv/cpl_snd calls into PhysicsDriver.compute and the
dycore epilogue.  Absent the table they must cost nothing: the module
holding them (woof.ocean.external) is never imported, the driver carries no
external_ocean, so no array is allocated and no kernel launched, and four
dycore steps of the small mixed land/water case (shortwave on) produce the
same words as before the change.  The digests below were taken on the
commit before #172 (nemo-column) and on #172 itself, which agreed, on CUDA
(A6000) and Metal (M3 Max), default and strict (GPUWM_WRF_EXACT=1)
numerics.  A deliberate physics change elsewhere moves them; re-take them
from a run without any external ocean.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import requires_gpu

DIGESTS = {
    ("cuda", False): "dad6511645a22240f1b99efe83cc9ec75f7124f7ec6b2e27bbe4a8a14c5a8d07",
    ("mlx", False): "e332afc278bdd9d6823c074b9794f1385f13405a4475c9a55574105e65fdb344",
    ("cuda", True): "5513949cd1fa3208160c71e8bc8a3804c23a11140476888f9350ec74b60a44eb",
    ("mlx", True): "5513949cd1fa3208160c71e8bc8a3804c23a11140476888f9350ec74b60a44eb",
}


@pytest.mark.gpu
@requires_gpu
def test_no_external_ocean_table_leaves_the_run_untouched():
    from woof import backend

    root = Path(__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, str(root / "tests" / "external_ocean_absent_probe.py")],
                         cwd=root, capture_output=True, text=True, env=dict(os.environ),
                         timeout=1800)
    assert out.returncode == 0, out.stderr[-2000:]
    digest, imported, attached = out.stdout.split()[-3:]
    assert imported == "-", "woof.ocean.external was imported without an [external_ocean]"
    assert attached == "False", "the driver carries an external_ocean without the table"
    key = (backend.active(), os.environ.get("GPUWM_WRF_EXACT", "0") == "1")
    if key not in DIGESTS:
        pytest.skip(f"no digest taken for backend {key}")
    assert digest == DIGESTS[key]
