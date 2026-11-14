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
    gf = {"option": "cumulus.grell-freitas",
          "overrides": {"cu_physics": "3", "ishallow": "1"},
          "hex_partners": {"ishallow": None, "x": "y"}}
    assert matrix.physics_args(gf, {}) == [
        "--physics", "cu_physics=3", "--physics", "x=y"]


def test_classify_reads_the_door_verdicts():
    passed = ("woof hex: the hex model (preview); its formats may change\n"
              "DOOR mesh=m steps=120 frames=2 status=passed out=/x\n")
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
    nan = ("woof hex: the forecast driver stopped with FloatingPointError: "
           "negative accumulated precipitation in rainc  Full traceback: /x\n")
    assert matrix.classify_log(nan, 1)["outcome"] == "non-finite"
    named = ("woof hex: the forecast driver stopped with ConfigurationRefused: "
             "GSFC shortwave requires 2 to 128 layers\n")
    assert matrix.classify_log(named, 1)["outcome"] == "refused"
    crash = "Traceback (most recent call last):\n  File x\nKeyError: 'lin-mp2'\n"
    verdict = matrix.classify_log(crash, 1)
    assert verdict["outcome"] == "error" and "KeyError" in verdict["reason"]
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


def test_a_component_filter_keeps_the_baseline():
    """THE BREAKAGE: ``run --components radiation`` dropped the baseline, so grade() screened nothing and the
    rows were published as supported."""
    options = [row["option"] for row in matrix.plan(["radiation"])]
    assert matrix.BASELINE in options
    assert all(o.split(".")[0] == "radiation" for o in options if o != matrix.BASELINE)


def test_rows_without_a_baseline_are_ungraded_until_it_runs():
    results = {"rows": {"a": _ran(-4e-4, 0.02), "d": {"outcome": "refused", "overrides": {}}}}
    rows = matrix.grade(results)["rows"]
    assert rows["a"]["outcome"] == "ungraded" and matrix.BASELINE in rows["a"]["reason"]
    assert rows["d"]["outcome"] == "refused"
    results["rows"][matrix.BASELINE] = {"outcome": "error", "reason": "boom", "overrides": {}}
    assert matrix.grade(results)["rows"]["a"]["outcome"] == "ungraded"
    results["rows"][matrix.BASELINE] = _ran(-4e-4, 0.02)
    rows = matrix.grade(results)["rows"]
    assert rows["a"]["outcome"] == "ran" and "reason" not in rows["a"] and rows["a"]["graded"]


def _fake_run(monkeypatch, ran):
    def run_row(row, case, work, backend):
        ran.append(row["option"])
        return {"option": row["option"], "backend": backend, "overrides": row["overrides"],
                "physics_args": matrix.physics_args(row, case.get("base_physics", {})),
                "row_args": list(case.get("row_args", {}).get(row["option"], [])),
                "outcome": "refused", "needs": ["x"]}

    monkeypatch.setattr(matrix, "run_row", run_row)


def _case(tmp_path, **changes):
    init = tmp_path / "init.nc"
    if not init.exists():
        init.write_bytes(b"init")
    case = {"forecast_args": ["--init", str(init)], "hours": 1, "label": "c", **changes}
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    return str(path)


def test_a_resume_of_another_case_is_refused(tmp_path, monkeypatch):
    """THE BREAKAGE: a reused --out loaded the previous case's rows, skipped them and graded new rows against
    its baseline."""
    ran = []
    _fake_run(monkeypatch, ran)
    out = str(tmp_path / "out")
    argv = ["run", "--out", out, "--backend", "cuda", "--only", "pbl.myj"]
    assert matrix.main([*argv, "--case", _case(tmp_path)]) == 0
    assert matrix.main([*argv, "--case", _case(tmp_path)]) == 0       # same case: resumed, nothing reruns
    assert ran == [matrix.BASELINE, "pbl.myj"]
    with pytest.raises(SystemExit, match="another case"):
        matrix.main([*argv, "--case", _case(tmp_path, hours=2)])
    (tmp_path / "init.nc").write_bytes(b"a re-staged init")               # same path, other file
    with pytest.raises(SystemExit, match="another case"):
        matrix.main([*argv, "--case", _case(tmp_path)])


def test_a_twin_is_reused_only_for_the_same_effective_configuration(tmp_path, monkeypatch):
    """pbl.myj and surface_layer.eta-similarity have equal overrides; THE BREAKAGE: a row flag for one of them
    was ignored and the other's result copied."""
    by_name = {row["option"]: row for row in matrix.plan()}
    assert by_name["pbl.myj"]["overrides"] == by_name["surface_layer.eta-similarity"]["overrides"]
    rows = ["pbl.myj", "surface_layer.eta-similarity"]
    ran = []
    _fake_run(monkeypatch, ran)
    case = _case(tmp_path, row_args={"surface_layer.eta-similarity": ["--flag"]})
    matrix.main(["run", "--case", case, "--out", str(tmp_path / "a"), "--backend", "cuda", "--only", *rows])
    assert sorted(ran) == sorted([matrix.BASELINE, *rows])
    ran.clear()
    matrix.main(["run", "--case", _case(tmp_path, row_args={}), "--out", str(tmp_path / "b"),
                 "--backend", "cuda", "--only", *rows])
    assert len(ran) == 2          # with no row flags the second IS the first run
    results = json.loads((tmp_path / "b" / "results-cuda.json").read_text())["rows"]
    assert any(row.get("same_run_as") for row in results.values())


def test_publish_refuses_inputs_that_are_not_one_comparison():
    def results(backend, case="c", hours=1, args=("--physics", "a=1"), row_args=()):
        return {"backend": backend, "case": case, "hours": hours, "rows": {
            "x": {"overrides": {"a": "1"}, "physics_args": list(args), "row_args": list(row_args),
                  "outcome": "ran"}}}

    matrix.publish([results("cuda"), results("metal")])
    with pytest.raises(ValueError, match="more than one results file for backend cuda"):
        matrix.publish([results("cuda"), results("cuda")])
    with pytest.raises(ValueError, match="different cases"):
        matrix.publish([results("cuda"), results("metal", case="other")])
    with pytest.raises(ValueError, match="different hours"):
        matrix.publish([results("cuda"), results("metal", hours=6)])
    with pytest.raises(ValueError, match="row x ran with different physics_args"):
        matrix.publish([results("cuda"), results("metal", args=("--physics", "a=2"))])
    with pytest.raises(ValueError, match="row x ran with different row_args"):
        matrix.publish([results("cuda"), results("metal", row_args=("--geog", "/other"))])


def test_the_smoke_holds_each_backend_to_its_own_record():
    """THE BREAKAGE: the cross-backend verdict of a row that ran on CUDA and errored on Metal is "error", so a
    CUDA smoke run accepted any CUDA outcome."""
    evidence = {"rows": {"k": {"backends": {"cuda": {"outcome": "ran"}, "metal": {"outcome": "error"}}},
                         "r": {"backends": {"cuda": {"outcome": "refused", "needs": ["THC"]}}}}}
    assert matrix.regression("k", evidence, "cuda", {"outcome": "ran"}) is None
    assert "recorded ran on cuda, now refused" in matrix.regression("k", evidence, "cuda", {"outcome": "refused"})
    assert matrix.regression("r", evidence, "cuda", {"outcome": "refused", "needs": ["THC"]}) is None
    assert "now needs" in matrix.regression("r", evidence, "cuda", {"outcome": "refused", "needs": ["LAI"]})
    assert "no recorded metal outcome" in matrix.regression("r", evidence, "metal", {"outcome": "refused"})


def test_case_identity_names_inline_option_files(tmp_path):
    """THE BREAKAGE: an --init=PATH argument was skipped as a flag, so a re-staged init at the same path kept
    the old rows on resume."""
    init = tmp_path / "init.nc"
    init.write_bytes(b"one")
    case = {"forecast_args": [f"--init={init}"], "hours": 1}
    first = matrix.case_identity(case)
    init.write_bytes(b"a longer re-staged init")
    assert matrix.case_identity(case) != first
