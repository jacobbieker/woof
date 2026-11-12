"""End to end: ``woof go`` on the hex engine, a 1 h regional GFS run (woof.go_hex).

Opt-in (``WOOF_HEX_GO_SMOKE=1``): it fetches GFS, builds a mesh and static
from WPS_GEOG (``$GPUWM_WPS_GEOG``), mints the regional class on first use and
runs the forecast on the host's GPU (CUDA or Metal), which no CI leg carries.
Declared in tools/battery/must_run_gates.txt.  ``WOOF_HEX_GO_SMOKE_CYCLE``
picks the GFS cycle (default 2025-05-30T12).  The contract and MPAS-suite stages
need the hex helpers (``hex_contract_hostprep``, the table decoders,
``rw_mpas_physics_pool``), which the release's Linux helper archive carries; no
macOS build exists yet, so this gate runs on a Linux GPU host.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("WOOF_HEX_GO_SMOKE") != "1",
    reason="opt-in: needs network GFS, WPS_GEOG and a GPU (WOOF_HEX_GO_SMOKE=1)")

TOML = """\
[model]
engine = "hex"

[experiment]
name = "hexgo_smoke"
start_time = {start}
run_seconds = 3600.0

[projection]
map_proj = "lambert"
ref_lat = 39.5
ref_lon = -98.5
truelat1 = 30.0
truelat2 = 60.0
stand_lon = -98.5

[[domain]]
nx = 40
ny = 40
dx = 12000.0
history_interval_s = 3600.0

[fetch]
source = "gfs"
cycle = "{cycle}"
"""


def test_woof_go_runs_a_hex_regional_forecast(tmp_path):
    cycle = os.environ.get("WOOF_HEX_GO_SMOKE_CYCLE", "2025-05-30T12")
    config = tmp_path / "hexgo_smoke.toml"
    config.write_text(TOML.format(start=cycle + ":00:00", cycle=cycle))
    out = tmp_path / "run"
    done = subprocess.run([sys.executable, "-m", "woof.cli", "go", str(config), "--outdir", str(out),
                           "--products", "none"], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    receipt = json.loads((out / "regional-receipt.json").read_text())
    assert receipt["outcome"] == "done"
    assert [stage for stage, row in receipt["stages"].items() if row["outcome"] == "done"] == [
        "mesh", "intermediate", "init", "lbc", "contract", "forecast"]
    plan = json.loads((out / "go-hex-plan.json").read_text())
    assert plan["engine"] == "hex" and plan["hours"] == 1
    frames = sorted((out / "forecast").glob("*history*.nc"))
    assert frames, sorted(path.name for path in (out / "forecast").iterdir())
    import numpy as np
    from netCDF4 import Dataset
    with Dataset(frames[-1]) as frame:
        for name in ("theta", "qv"):
            assert np.isfinite(np.asarray(frame.variables[name][:])).all(), name
