"""Host checks of tools/hex_option_matrix.py (hex parity plan item 1.3).

The forecasts themselves run in tests/test_hex_option_matrix_smoke.py (opt-in,
slow).  These checks pin what the tool decides without a device: which rows it
runs and with which ``--physics`` assignments, how it reads a door log, the
budget arithmetic on a small synthetic history pair, the grading against the
baseline row, and that the committed evidence covers every row it plans.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import hex_option_matrix as matrix  # noqa: E402
import mpas_column_physics_cases as cases  # noqa: E402

EVIDENCE = REPO / "components/hex/evidence/option-matrix.json"


def test_plan_is_the_column_sweep_menu_without_turbulence():
    names = [row["option"] for row in matrix.plan()]
    expected = [name for name in cases.selections()
                if name.split(".")[0] != "turbulence"]
    assert names == expected
    assert matrix.BASELINE in names


def test_overrides_rebuild_each_selection_from_the_default():
    base = dict(cases.suite.MPAS_DEFAULT_PHYSICS)
    for row in matrix.plan():
        rebuilt = dict(base)
        if "ra_lw_physics" in row["overrides"]:
            rebuilt.pop("ra_physics")
            rebuilt.pop("ra_rrtmg_variant")
        rebuilt.update(row["overrides"])
        assert {k: str(v) for k, v in rebuilt.items()} == {
            k: matrix._value(v) for k, v in row["physics"].items()}, row
    assert matrix.overrides(base, base) == {}


def test_physics_args_carry_the_case_base_and_let_drag_rows_replace_it():
    plain = {"option": "x", "overrides": {"mp_physics": "2"}}
    assert matrix.physics_args(plain, {"config_gwdo_scheme": "off"}) == [
        "--physics", "config_gwdo_scheme=off", "--physics", "mp_physics=2"]
    drag = {"option": "drag.gwdo", "overrides": {"gwd_opt": "1"}}
    assert matrix.physics_args(drag, {"config_gwdo_scheme": "off"}) == [
        "--physics", "gwd_opt=1"]


def test_classify_reads_the_door_verdicts():
    passed = "woof hex: banner\nDOOR mesh=m steps=120 frames=2 status=passed out=/x\n"
    assert matrix.classify_log(passed, 0) == {"outcome": "ran"}
    assert matrix.classify_log(passed, 1)["outcome"] == "error"
    missing = ("woof hex: refused: missing surface input: sf_surface_physics=1 "
               "(slab) reads thermal_inertia (THC) per cell, which the hex "
               "static builder does not stage yet\n")
    verdict = matrix.classify_log(missing, 2)
    assert verdict["outcome"] == "refused"
    assert verdict["needs"] == ["thermal_inertia (THC)"]
    refused = "woof hex: mp_physics=29 is refused by WOOF: RCON-SPEED-01\n"
    assert matrix.classify_log(refused, 2) == {
        "outcome": "refused", "reason": refused.strip()}
    nan = "FloatingPointError: theta is non-finite at step 7\n"
    assert matrix.classify_log(nan, 1)["outcome"] == "non-finite"
    assert matrix.classify_log("Traceback\nKeyError: 'x'\n", 1)[
        "outcome"] == "error"
    stopped = ("REGIONAL class=graded-3997m-dt30-z7-l149 -> REFUSED\n"
               "woof hex: the forecast driver stopped with KeyError: 'x'\n")
    verdict = matrix.classify_log(stopped, 2)
    assert verdict["outcome"] == "error" and "KeyError" in verdict["reason"]


def _write_frame(path, rho, qv, rainnc, qfx, zgrid):
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("nCells", rho.shape[0])
        ds.createDimension("nVertLevels", rho.shape[1])
        ds.createDimension("nVertLevelsP1", rho.shape[1] + 1)
        ds.createVariable("zgrid", "f4", ("nCells", "nVertLevelsP1"))[:] = zgrid
        for name, value in (("rho", rho), ("qv", qv)):
            ds.createVariable(name, "f4", ("Time", "nCells",
                                           "nVertLevels"))[0] = value
        ds.createVariable("rainnc", "f4", ("Time", "nCells"))[0] = rainnc
        ds.createVariable("qfx", "f4", ("Time", "nCells"))[0] = qfx


def test_budgets_on_a_synthetic_pair(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    static = tmp_path / "static.nc"
    with netCDF4.Dataset(static, "w") as ds:
        ds.sphere_radius = 6371229.0
        ds.createDimension("nCells", 3)
        ds.createVariable("areaCell", "f8", ("nCells",))[:] = [1.0, 2.0, 1.0]
        ds.createVariable("bdyMaskCell", "i4", ("nCells",))[:] = [0, 0, 7]
    zgrid = np.tile(np.array([0.0, 100.0, 300.0], np.float32), (3, 1))
    rho = np.ones((3, 2), np.float32)
    _write_frame(tmp_path / "h.00.nc", rho, np.full((3, 2), 0.01, np.float32),
                 np.zeros(3, np.float32), np.zeros(3, np.float32), zgrid)
    # One hour later: 1 kg m-2 rained out of every column's vapour, and the
    # dry mass of cell 2 grew 1 % (as a boundary inflow would).
    rho2 = rho.copy()
    rho2[2] *= 1.01
    qv2 = np.full((3, 2), 0.01, np.float32) - np.float32(1.0 / 300.0) / rho2
    _write_frame(tmp_path / "h.01.nc", rho2, qv2, np.ones(3, np.float32),
                 np.zeros(3, np.float32), zgrid)
    out = matrix.history_budgets(
        [tmp_path / "h.00.nc", tmp_path / "h.01.nc"], static, 1.0)
    assert out["nonfinite"] == []
    assert out["species"] == ["qv"]
    assert out["dry_mass_kg"][0] == pytest.approx(4 * 300.0)
    assert out["dry_rel_change"] == pytest.approx(0.01 * 300 / 1200, rel=1e-5)
    assert out["dry_rel_change_interior"] == pytest.approx(0.0, abs=1e-7)
    assert out["precip_kg"] == pytest.approx(4.0)
    # Vapour fell by exactly what rained, plus cell 2's inflowing moisture.
    assert out["water_residual_rel_interior"] == pytest.approx(0.0, abs=1e-5)
    assert out["evap_estimate_kg"] == 0.0


def test_a_nonfinite_history_field_is_named(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    static = tmp_path / "static.nc"
    with netCDF4.Dataset(static, "w") as ds:
        ds.createDimension("nCells", 2)
        ds.createVariable("areaCell", "f8", ("nCells",))[:] = [1e-6, 1e-6]
    zgrid = np.tile(np.array([0.0, 100.0], np.float32), (2, 1))
    rho = np.ones((2, 1), np.float32)
    _write_frame(tmp_path / "a.nc", rho, rho * 0.01, np.zeros(2, np.float32),
                 np.zeros(2, np.float32), zgrid)
    _write_frame(tmp_path / "b.nc", rho, rho * 0.01,
                 np.array([np.nan, 0.0], np.float32), np.zeros(2, np.float32),
                 zgrid)
    out = matrix.history_budgets([tmp_path / "a.nc", tmp_path / "b.nc"],
                                 static, 1.0)
    assert out["nonfinite"] == ["b.nc:rainnc"]


def _ran(dry, water):
    return {"outcome": "ran", "overrides": {},
            "budgets": {"dry_rel_change": dry, "water_residual_rel": water}}


def test_grading_is_against_the_baseline_row():
    results = {"rows": {
        matrix.BASELINE: _ran(-4e-4, 0.02),
        "a": _ran(-4.5e-4, 0.03),
        "b": _ran(5e-3, 0.02),
        "c": _ran(-4e-4, 0.2),
        "d": {"outcome": "refused", "overrides": {}},
    }}
    rows = matrix.grade(results)["rows"]
    assert rows["a"]["outcome"] == "ran"
    assert rows[matrix.BASELINE]["graded"]["dry_vs_baseline"] == 0.0
    assert rows["b"]["outcome"] == "budget" and "dry mass" in rows["b"]["reason"]
    assert rows["c"]["outcome"] == "budget" and "water" in rows["c"]["reason"]
    assert rows["d"]["outcome"] == "refused"
    # A row fixed on rerun is graded clean again.
    rows["b"]["budgets"]["dry_rel_change"] = -4e-4
    assert matrix.grade(results)["rows"]["b"]["outcome"] == "ran"


def test_publish_scrubs_paths_and_states_one_verdict_per_option():
    loaded = [
        {"backend": "cuda", "case": "c", "hours": 1, "rows": {
            "x": {"overrides": {"mp_physics": "2"}, "outcome": "ran",
                  "wall_seconds": 10.0, "receipt": {"seam": "engine"}},
            "y": {"overrides": {"sf_surface_physics": "1"},
                  "outcome": "refused", "needs": ["thermal_inertia (THC)"],
                  "reason": "missing surface input: ... /home/u/run/out"},
            "z": {"overrides": {"cu_physics": "7"}, "outcome": "error",
                  "reason": "KeyError in /home/u/tree/woof/x.py"}}},
        {"backend": "metal", "case": "c", "hours": 1, "rows": {
            "x": {"overrides": {"mp_physics": "2"}, "outcome": "ran",
                  "wall_seconds": 30.0}}},
    ]
    evidence = matrix.publish(loaded)
    assert "/home/u" not in json.dumps(evidence)
    assert matrix.verdict(evidence["rows"]["x"]) == "supported"
    assert matrix.verdict(evidence["rows"]["y"]) == \
        "refused (needs thermal_inertia (THC))"
    assert matrix.verdict(evidence["rows"]["z"]) == "error"
    text = matrix.table(evidence)
    assert "| x | mp_physics=2 | supported | ran (10 s) | ran (30 s) |" in text
    assert "Counts: error 1, refused 1, supported 1" in text


def test_the_committed_evidence_covers_every_planned_option():
    evidence = json.loads(EVIDENCE.read_text())
    assert evidence["schema"] == matrix.SCHEMA
    planned = {row["option"]: row["overrides"] for row in matrix.plan()}
    assert set(evidence["rows"]) == set(planned), (
        "rerun tools/hex_option_matrix.py for the new or removed options")
    for name, row in evidence["rows"].items():
        assert row["overrides"] == planned[name], name
        assert row["backends"], name
