"""A short hex forecast on a variable-resolution mesh (opt-in, slow).

The regional hex cases the GPU tests and benchmarks use grade one doubling inside the cut.  This runs one
hour of a case from ``components/hex/bench/varres-cases.v1.json`` -- by default ``edi-vr``, a 3 km core
graded to about 20 km (about 4,600 cells) -- through ``woof hex regional`` on the active GPU backend, and
holds it to what a variable-resolution mesh must do:

* every floating field of every history frame is finite;
* the step is the one the smallest cell admits (the admitted ``dt`` is at most the Courant limit the
  mesh rows record for the cull's shortest ``dcEdge``);
* convection, when on, runs with each column's own spacing (``len_disp / meshDensity**0.25``);
* the transition-band screen of ``tools/hex_varres.py analyze`` flags no bin.

Opt in with ``WOOF_HEX_VARRES_CASE=<work dir>`` naming a case prepared by ``python tools/hex_varres.py
prep NAME --work-dir DIR`` (the directory holding ``NAME/``; the mesh and boundaries are generated there,
never committed) and ``WOOF_HEX_VARRES_NAME`` (default ``edi-vr``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = os.environ.get("WOOF_HEX_VARRES_CASE")
if not ROOT:
    pytest.skip("opt-in: set WOOF_HEX_VARRES_CASE to a work dir prepared by tools/hex_varres.py prep",
                allow_module_level=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import hex_varres as vr  # noqa: E402

pytestmark = pytest.mark.slow
NAME = os.environ.get("WOOF_HEX_VARRES_NAME", "edi-vr")


def _receipt_values(node, wanted, found):
    if isinstance(node, dict):
        for key, value in node.items():
            # Digest scalars are flat keys (``physics/seam/identity/gf_dx_column``): match the last part.
            name = key.rsplit("/", 1)[-1]
            if name in wanted and not isinstance(value, (dict, list)):
                found.setdefault(name, value)
            _receipt_values(value, wanted, found)
    elif isinstance(node, list):
        for value in node:
            _receipt_values(value, wanted, found)
    return found


def test_an_hour_on_a_variable_resolution_mesh(tmp_path):
    data = vr.load_cases()
    case = vr.case_row(NAME, data)
    assert case["kind"] == "regional", "the GPU leg runs the regional cases; glob-vr is a bench case"
    source = Path(ROOT) / NAME
    work = tmp_path / NAME
    shutil.copytree(source, work, ignore=shutil.ignore_patterns("forecast", "products", "grib", "met"))
    for path in work.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        if str(source) in text:
            path.write_text(text.replace(str(source), str(work)), encoding="utf-8")
    argv = [sys.executable, "-m", "woof", "hex", "regional", *vr.case_arguments(case, Path(ROOT)),
            "--cycle", data["cycle_default"], "--source", data["source_default"], "--hours", "1",
            "--history-every-minutes", "60", "--work-dir", str(work),
            "--skip", "mesh", "--skip", "intermediate", "--skip", "init", "--skip", "lbc", "--skip", "contract",
            "--forecast-arg=--physics=mp_physics=6", "--no-products"]
    done = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]

    grid = next((work / "mesh").glob("*-cull.grid.nc"))
    result = vr.analyze(grid, work / "forecast", center=tuple(case["center"]))
    assert result["nonfinite"] == []
    assert len(result["series"]) == 2
    assert result["h_km"][1] / result["h_km"][0] > 4.0, "not a variable-resolution case"
    assert result["series"][-1]["screen"]["flagged"] == [], result["series"][-1]["screen"]

    # The step: the cull's row admits it on the cull's own shortest dcEdge (the finest cell), not a mean.
    import netCDF4

    with netCDF4.Dataset(str(grid)) as g:
        dc_min = float(g["dcEdge"][:].min())
        if dc_min < 1.0:
            dc_min *= vr.EARTH_RADIUS_M
    rows = json.loads((work / "mesh" / "mesh-rows.json").read_text(encoding="utf-8"))["rows"]
    cull = next(row for row in rows if row.get("regional", True) is not False and row["name"].startswith("q"))
    admission = cull["admission"]["timestep_admission"]
    speed = float(admission["courant_policy"]["max_characteristic_speed_m_s"])
    assert float(admission["maximum_admitted_dt_seconds"]) == pytest.approx(
        float(admission["admitted_courant"]) * dc_min / speed, rel=1e-3)
    assert float(cull["dt_seconds"]) <= float(admission["maximum_admitted_dt_seconds"])

    receipt = json.loads((work / "forecast" / "forecast-receipt.json").read_text(encoding="utf-8"))
    found = _receipt_values(receipt, {"scheme", "gf_dx_column"}, {})
    if found.get("scheme") == "cu_grell_freitas":
        assert found.get("gf_dx_column") == "per-column"
