"""Hex forecast smoke of the generic-seam options (opt-in, slow; plan 1.3).

Runs tools/hex_option_matrix.py on a staged hex case through real
``woof hex forecast`` subprocesses and holds every row to the committed
evidence (components/hex/evidence/option-matrix.json): an option recorded as
supported still runs, finite and inside the budget tolerances against the
baseline row; an option recorded as refused is still refused, by the same
named input.

Opt in with ``WOOF_HEX_OPTION_MATRIX_CASE=<case.json>`` (the case format is in
the tool's docstring) on a host with a GPU backend.  By default it runs a
small spread of rows (the baseline, one ran row per component, one refusal);
``WOOF_HEX_OPTION_MATRIX_ROWS=all`` or a comma list picks others.
``WOOF_HEX_OPTION_MATRIX_BACKEND`` labels the results (default: the active
backend's name).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

CASE = os.environ.get("WOOF_HEX_OPTION_MATRIX_CASE")
if not CASE:
    pytest.skip("opt-in: set WOOF_HEX_OPTION_MATRIX_CASE to a staged hex case "
                "(tools/hex_option_matrix.py)", allow_module_level=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import hex_option_matrix as matrix  # noqa: E402

pytestmark = pytest.mark.slow

#: ``WOOF_HEX_OPTION_MATRIX_EVIDENCE`` picks the evidence the case is held to: the quasi-uniform case's
#: (default) or ``option-matrix-varres.json``, the smoke rows on a variable-resolution case
#: (tools/hex_varres.py edi-vr).
EVIDENCE = json.loads(
    (REPO / "components/hex/evidence" / os.environ.get(
        "WOOF_HEX_OPTION_MATRIX_EVIDENCE", "option-matrix.json")).read_text())

DEFAULT_ROWS = (
    "microphysics.lin-mp2",
    "cumulus.kain-fritsch",
    "pbl.mynn",
    "surface_layer.mynn",
    "land_surface.noah",
    "radiation.new-goddard",
    "land_surface.slab",
)


def _rows():
    picked = os.environ.get("WOOF_HEX_OPTION_MATRIX_ROWS")
    if picked == "all":
        return None
    if picked:
        return [name.strip() for name in picked.split(",") if name.strip()]
    return list(DEFAULT_ROWS)


def _backend():
    label = os.environ.get("WOOF_HEX_OPTION_MATRIX_BACKEND")
    if label:
        return label
    from woof import backend  # noqa: PLC0415
    name = backend.active()
    return "metal" if name == "mlx" else name


def test_options_hold_their_recorded_outcome(tmp_path):
    backend = _backend()
    argv = ["run", "--case", CASE, "--out", str(tmp_path),
            "--backend", backend]
    rows = _rows()
    if rows is not None:
        argv += ["--only", *rows]
    assert matrix.main(argv) == 0
    results = json.loads(
        (tmp_path / f"results-{backend}.json").read_text())["rows"]
    problems = [problem for name, row in results.items()
                if (problem := matrix.regression(name, EVIDENCE, backend, row)) is not None]
    assert not problems, "\n".join(problems)
