"""A forecast from the woof 3-D-Var analysis against one from WRFDA's ``wrfvar_output`` (plan V1, gate GV1).

On the WRFDA 4.7.1 tutorial (``WOOF_WRFDA471_DIR`` with ``be.dat``, ``ob.ascii``, ``fg``,
``wrfbdy_d01`` and WRFDA's ``wrfvar_output``): the reference-tier analysis and WRFDA's analysis
each become a ``wrf-analysis-bundle/1`` (WRFDA's as the first guess with its eight updated
fields replaced, because ``wrfvar_output``'s ``TITLE`` is not the audited pre-hybrid writer's)
and start a 6 h woof forecast through ``woof run --wrfinput --analysis`` (direct insertion,
``tests/data/wrfda471/forecast.namelist.input``: the tutorial's physics, 360 s step), and
at every history time (12, 15, 18 UTC) the two ``wrfout`` files are compared; a run without ``--analysis``
must differ from both (so the comparison is not vacuous).

Measured (Metal): the two forecasts' ``wrfout`` files are byte-identical at every history
time, because the reference analysis's float32 words are ``wrfvar_output``'s. One-off measurement (Metal, a 12 h run, not a test): the production-tier
(binary32 minimiser) analysis differs from WRFDA's at float32 noise at the start (U 1.9e-6 rms) and
that grows with the tutorial's convection: 3 % of the no-assimilation difference by 6 h and 13-30 %
by 12 h (RAINC 51 %). The forecast is deterministic (two runs are byte-identical), so this is the
model's sensitivity, not noise in the run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

REPO = Path(__file__).resolve().parents[1]
NAMELIST = Path(__file__).parent / "data" / "wrfda471" / "forecast.namelist.input"
RUN_SECONDS = 21600
# history_interval = 180 min: every history time of the 6 h run
TIMES = ("2008-02-05_12_00_00", "2008-02-05_15_00_00", "2008-02-05_18_00_00")


def _dir() -> Path:
    d = os.environ.get("WOOF_WRFDA471_DIR")
    need = ("be.dat", "ob.ascii", "fg", "wrfvar_output", "wrfbdy_d01")
    if not d or not all((Path(d) / n).is_file() for n in need):
        pytest.skip("WOOF_WRFDA471_DIR with be.dat, ob.ascii, fg, wrfvar_output and wrfbdy_d01 is not set")
    return Path(d)


def _forecast(run: Path, out: Path, bundle: Path | None):
    cmd = [sys.executable, "-m", "woof.cli", "run", "--wrfinput", str(run), "--run-seconds", str(RUN_SECONDS),
           "--outdir", str(out), "--products", "none", "--no-supervise"] + (["--analysis", str(bundle)] if bundle else [])
    r = subprocess.run(cmd, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return {t: out / "wrfout" / f"wrfout_d01_{t}" for t in TIMES}


@requires_gpu
def test_forecast_from_woof_analysis_is_wrfdas(tmp_path):
    d = _dir()
    from conftest import cpu_preprocess_gap, netcdf_bridge_gap, netcdf_writer_gap

    from woof.static.rust_bridge import unavailable_reason

    for gap in (netcdf_bridge_gap(), netcdf_writer_gap(), cpu_preprocess_gap("gpuwm_analysis_binary_f64")):
        if gap:
            pytest.skip(gap)
    # woof run builds the history grid's cell geometry with the Rust static-fields bridge
    # (woof.io.history_layout.wrf_cell_geometry); without it every run here stops at step 0.
    static_gap = unavailable_reason()
    if static_gap:
        pytest.skip(f"the Rust static-fields bridge woof run needs is not loadable here ({static_gap})")
    import netCDF4

    from woof.da.var.analysis import WRF_FIELDS, analyse, read_first_guess, write_bundle

    fg = read_first_guess(d / "fg")
    ours = analyse(d / "be.dat", d / "ob.ascii", d / "fg", tier="reference")
    with netCDF4.Dataset(d / "wrfvar_output") as nc:
        nc.set_auto_mask(False)
        theirs = {n: np.asarray(nc.variables[n][0], dtype=np.float64).T for n in WRF_FIELDS}
    write_bundle(tmp_path / "bundle-woof", fg, ours.wrf, recipe={"test": "forecast"}, grid_id="wrfda-tutorial")
    write_bundle(tmp_path / "bundle-wrfda", fg, theirs, recipe={"src": "wrfvar_output"}, grid_id="wrfda-tutorial")
    runs = {}
    for k in ("noda", "woof", "wrfda"):  # one run directory each
        runs[k] = tmp_path / f"run-{k}"
        runs[k].mkdir()
        (runs[k] / "wrfinput_d01").symlink_to(d / "fg")
        (runs[k] / "wrfbdy_d01").symlink_to(d / "wrfbdy_d01")
        shutil.copy(NAMELIST, runs[k] / "namelist.input")
    # the analyses are not the first guess, or the comparison below would pass vacuously
    first = netCDF4.Dataset(d / "fg")
    first.set_auto_mask(False)
    assert any(np.max(np.abs(np.asarray(first.variables[n][0], dtype=np.float64).T - theirs[n])) > 0
               for n in WRF_FIELDS)
    first.close()
    noda = _forecast(runs["noda"], tmp_path / "out-noda", None)
    woof = _forecast(runs["woof"], tmp_path / "out-woof", tmp_path / "bundle-woof")
    wrfda = _forecast(runs["wrfda"], tmp_path / "out-wrfda", tmp_path / "bundle-wrfda")
    for t in TIMES:
        assert woof[t].read_bytes() == wrfda[t].read_bytes(), t
    for t in TIMES:  # the assimilation changed the forecast
        assert woof[t].read_bytes() != noda[t].read_bytes(), t
