"""CPU checks for the Breeze regional-benchmark harness (tools/compare/era5_nest)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "tools" / "compare" / "era5_nest"
sys.path.insert(0, str(HARNESS))

import breeze_cases  # noqa: E402
import make_nest_configs  # noqa: E402
import nest_throughput  # noqa: E402
import station_table  # noqa: E402


def test_published_grids_match_their_sources():
    mc3e, ar = breeze_cases.MC3E, breeze_cases.AR
    assert (mc3e.nx, mc3e.ny, mc3e.nz) == (150, 136, 50)
    assert mc3e.cells == 1_020_000
    assert mc3e.center == pytest.approx((36.605, -97.485))
    assert mc3e.lon_bounds == pytest.approx((-105.835, -89.135))
    assert mc3e.lat_bounds == pytest.approx((29.055, 44.155))
    assert (ar.nx, ar.ny, ar.nz) == (540, 315, 50)
    # Nx = Lλ / (1/9) for both boxes.
    for case in (mc3e, ar):
        width = case.lon_bounds[1] - case.lon_bounds[0]
        height = case.lat_bounds[1] - case.lat_bounds[0]
        assert round(width * 9) == case.nx
        assert round(height * 9) == case.ny
    assert ar.cycle == "2025-12-07T12" and mc3e.cycle == "2011-05-20T00"


def test_breeze_column_reproduces_oceananigans_interfaces():
    faces = breeze_cases.breeze_z_interfaces()
    assert len(faces) == 51
    assert faces[:6] == [0.0, 60.0, 129.0, 208.35, 299.6, 404.54]
    assert faces[16] == pytest.approx(3342.99)
    assert faces[-1] == pytest.approx(20002.99)
    spacings = [b - a for a, b in zip(faces, faces[1:])]
    assert max(spacings) == pytest.approx(490.0)


def test_matched_eta_is_a_valid_ladder():
    eta, p_top = breeze_cases.breeze_matched_eta()
    assert len(eta) == 51
    assert eta[0] == 1.0 and eta[-1] == 0.0
    assert all(a > b for a, b in zip(eta, eta[1:]))
    assert p_top == pytest.approx(5472.3, abs=1.0)
    # 60 m surface layer in a standard atmosphere: about 0.72 % of the column.
    assert 1.0 - eta[1] == pytest.approx(0.0075, abs=0.0005)


def test_standard_atmosphere_reference_points():
    assert breeze_cases.standard_pressure_pa(0.0) == pytest.approx(101325.0)
    assert breeze_cases.standard_pressure_pa(11000.0) == pytest.approx(22632, rel=1e-3)
    assert breeze_cases.standard_pressure_pa(20000.0) == pytest.approx(5474.9, rel=1e-3)
    with pytest.raises(ValueError):
        breeze_cases.standard_pressure_pa(40000.0)


def test_breeze_spacing_shrinks_poleward():
    spacing = breeze_cases.breeze_spacing_km(breeze_cases.AR)
    assert spacing["dy"] == pytest.approx(12.35, abs=0.01)
    assert spacing["dx_poleward_edge"] == pytest.approx(6.18, abs=0.01)
    assert spacing["dx_equatorward_edge"] > spacing["dx_center"] > spacing["dx_poleward_edge"]
    coverage = breeze_cases.ground_coverage_km(breeze_cases.MC3E)
    assert coverage["woof_extent_x_km"] == 1800.0
    assert coverage["area_ratio_woof_over_breeze"] == pytest.approx(1.18, abs=0.01)


def _step(sequence, step, model_seconds, ms, domain=1):
    return {"schema": "gpuwm.step-log/v4", "sequence": sequence,
            "emitted_unix_ms": ms, "event": "step", "domain": domain,
            "step": step, "model_seconds": model_seconds}


def test_woof_throughput_uses_the_steady_window():
    records = [{"event": "run_start", "emitted_unix_ms": 0}]
    # A slow first step (compile), then 60 s steps at 0.5 s wall each.
    records.append(_step(2, 1, 60.0, 20_000))
    for step in range(2, 101):
        records.append(_step(step + 1, step, 60.0 * step, 20_000 + 500 * (step - 1)))
    records.append({"event": "run_end", "status": "SUCCESS",
                    "wall_seconds": 80.0, "first_step_excess_seconds": 19.5})
    result = nest_throughput.woof_throughput(records, cells=1_000_000,
                                             warmup_steps=10)
    assert result["kind"] == "measured"
    assert result["window_first_step"] == 10
    assert result["window_steps"] == 90
    assert result["seconds_per_step"] == pytest.approx(0.5)
    assert result["mean_dt_seconds"] == pytest.approx(60.0)
    assert result["sim_hours_per_wall_hour"] == pytest.approx(120.0)
    assert result["cell_steps_per_second"] == pytest.approx(2.0e6)
    assert result["run_sim_hours_per_wall_hour"] == pytest.approx(6000.0 / 80.0)


def test_woof_throughput_ignores_other_domains_and_refuses_empty():
    records = [_step(1, 1, 60.0, 0, domain=2), _step(2, 2, 120.0, 1000, domain=2)]
    with pytest.raises(ValueError):
        nest_throughput.woof_throughput(records, cells=10)


def test_progress_log_is_found_below_a_run_dir(tmp_path):
    run = tmp_path / "go" / "run-x"
    run.mkdir(parents=True)
    (run / "progress.jsonl").write_text("")
    assert nest_throughput.find_progress_log(tmp_path) == run / "progress.jsonl"
    (tmp_path / "progress.jsonl").write_text("")
    assert nest_throughput.find_progress_log(tmp_path) == tmp_path / "progress.jsonl"


def test_published_ar_figures_and_bandwidth_estimate():
    ar = breeze_cases.AR
    whole, one_gpu, two_gpu = ar.published
    documented = nest_throughput.published_throughput(ar, whole)
    assert documented["kind"] == "documented"
    assert documented["sim_hours_per_wall_hour"] == pytest.approx(72 / 3.77)
    assert documented["seconds_per_step"] == pytest.approx(3.77 * 3600 / 47600)
    assert documented["mean_dt_seconds"] == pytest.approx(72 * 3600 / 47600)
    per_iteration = nest_throughput.published_throughput(ar, one_gpu)
    assert per_iteration["cell_steps_per_second"] == pytest.approx(ar.cells / 0.267)
    assert "borrowed" in per_iteration["note"]
    estimate = nest_throughput.published_throughput(ar, whole, scale_to="rtx-a6000")
    assert estimate["kind"] == "estimate"
    ratio = 768.0 / 1555.0
    assert estimate["sim_hours_per_wall_hour"] == pytest.approx(72 / 3.77 * ratio)
    assert estimate["seconds_per_step"] == pytest.approx(3.77 * 3600 / 47600 / ratio)
    assert two_gpu.gpus == 2


def test_an_unnamed_a100_is_scaled_only_under_a_stated_assumption(capsys):
    mc3e = breeze_cases.MC3E
    timing = mc3e.published[0]
    assert timing.card is None
    with pytest.raises(ValueError):
        nest_throughput.published_throughput(mc3e, timing, scale_to="rtx-a6000")
    as_80 = nest_throughput.published_throughput(
        mc3e, timing, scale_to="rtx-a6000", assume_card="a100-80gb")
    assert as_80["assumed_source_card"] == "a100-80gb"
    assert as_80["sim_hours_per_wall_hour"] == pytest.approx(60 / 7.2 * 768 / 2039)
    assert nest_throughput.main(["published", "--case", "mc3e", "--scale-to", "rtx-a6000"]) == 0
    out = capsys.readouterr().out
    assert "(if a100-40gb)" in out and "(if a100-80gb)" in out


def test_a_run_shorter_than_its_warmup_is_flagged():
    records = [_step(1, 1, 60.0, 0), _step(2, 2, 120.0, 10_000), _step(3, 3, 180.0, 10_500)]
    result = nest_throughput.woof_throughput(records, cells=10, warmup_steps=10)
    assert result["steady_window"] is False
    assert "compile" in result["note"]
    assert result["window_first_step"] == 1
    breeze = nest_throughput.breeze_throughput(
        {"window_iterations": 5, "window_model_seconds": 50.0,
         "window_wall_seconds": 100.0, "warmup_iterations": 0}, cells=10)
    assert breeze["steady_window"] is False
    with pytest.raises(ValueError):
        nest_throughput.breeze_throughput(
            {"window_iterations": 0, "window_model_seconds": 0.0,
             "window_wall_seconds": 1.0}, cells=10)


def test_shrink_keeps_centre_and_spacing():
    small = make_nest_configs.shrink(breeze_cases.MC3E, 3)
    assert (small.nx, small.ny) == (50, 45)
    assert small.center == pytest.approx(breeze_cases.MC3E.center)
    full = breeze_cases.breeze_spacing_km(breeze_cases.MC3E)
    part = breeze_cases.breeze_spacing_km(small)
    assert part["dy"] == pytest.approx(full["dy"])
    assert make_nest_configs.shrink(breeze_cases.MC3E, 1) is breeze_cases.MC3E
    with pytest.raises(ValueError):
        make_nest_configs.shrink(breeze_cases.MC3E, 0)


def test_breeze_driver_summary_converts():
    summary = {"window_iterations": 1000, "window_model_seconds": 6000.0,
               "window_wall_seconds": 300.0, "cells": 1_020_000}
    result = nest_throughput.breeze_throughput(summary, cells=1_020_000)
    assert result["sim_hours_per_wall_hour"] == pytest.approx(20.0)
    assert result["cell_steps_per_second"] == pytest.approx(1_020_000 * 1000 / 300)


def test_equal_step_wall_hours():
    assert nest_throughput.equal_step_wall_hours(0.36, 47600) == pytest.approx(4.76)


def test_cli_prints_a_table(capsys):
    assert nest_throughput.main(["published", "--case", "ar", "--scale-to", "rtx-a6000"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("| run | kind |")
    assert "estimate" in out and "documented" in out
    # The two-card figure is never scaled by one card's bandwidth.
    assert out.count("-> rtx-a6000") == 2


_TOML = """[experiment]
restart_interval_s = 3600.0
[[domain]]
nx = 150
ny = 120
radt = 12.0
[fetch]
area = "27.64,-110.63,45.17,-84.34"
"""

_NAMELIST = """&geogrid
 e_we              = 151,
 e_sn              = 121,
/
"""


def test_patches_pin_counts_and_refuse_a_changed_shape():
    case = breeze_cases.MC3E
    text = make_nest_configs.patch_toml(_TOML, case, area="1.00,2.00,3.00,4.00",
                                        radt_minutes=60.0, restarts=False)
    assert "nx = 150\n" in text and "ny = 136\n" in text
    assert 'area = "1.00,2.00,3.00,4.00"' in text
    assert "radt = 60.0" in text
    assert "restart_interval_s = 0.0" in text
    kept = make_nest_configs.patch_toml(_TOML, case, area="a", radt_minutes=None,
                                        restarts=True)
    assert "radt = 12.0" in kept and "restart_interval_s = 3600.0" in kept
    namelist = make_nest_configs.patch_namelist(_NAMELIST, case)
    assert "e_we              = 151," in namelist
    assert "e_sn              = 137," in namelist
    with pytest.raises(RuntimeError):
        make_nest_configs.patch_toml(_TOML.replace("ny = 120\n", ""), case,
                                     area="a", radt_minutes=None, restarts=True)


def test_fetch_area_covers_the_patched_grid():
    projection = {"map_proj": "lambert", "ref_lat": 36.605, "ref_lon": -97.485,
                  "truelat1": 26.605, "truelat2": 46.605, "stand_lon": -97.485}
    small = make_nest_configs.fetch_area(projection, 150, 120, 12000.0)
    large = make_nest_configs.fetch_area(projection, 150, 136, 12000.0)
    s_small, _, n_small, _ = map(float, small.split(","))
    s_large, _, n_large, _ = map(float, large.split(","))
    assert s_large < s_small and n_large > n_small


def test_wizard_argv_carries_the_matched_column():
    argv = make_nest_configs.wizard_argv(
        breeze_cases.MC3E, name="x", out=Path("x.toml"), hours=3, cadence=1,
        provider="arco", physics=make_nest_configs.DEFAULT_PHYSICS, vram_gib=6)
    joined = " ".join(argv)
    assert "--p-top-pa 5472" in joined
    eta = argv[argv.index("--eta-levels") + 1].split(",")
    assert len(eta) == 51
    assert "--cadence 1" in joined and "--cycle 2011-05-20T00" in joined
    assert json.dumps(argv)  # serialisable into the match receipt


def _progress(sequence, step, model_seconds, wall):
    return {"schema_version": "gpuwm.run-plan.event.v1", "sequence": sequence,
            "event": "model_progress", "domain": 1, "outer_step": step,
            "model_seconds": model_seconds, "wall_seconds": wall}


def test_woof_go_event_stream_is_read(tmp_path):
    events = [{"schema_version": "gpuwm.run-plan.event.v1", "sequence": 1,
               "event": "stage_finished", "stage": "prepare", "wall_seconds": 92.0},
              _progress(2, 0, 0.0, 1e-6),
              _progress(3, 1, 60.0, 12.0)]
    for step in range(2, 51):
        events.append(_progress(step + 2, step, 60.0 * step, 12.0 + 0.25 * (step - 1)))
    events.append({"schema_version": "gpuwm.run-plan.event.v1", "sequence": 99,
                   "event": "stage_finished", "stage": "forecast", "wall_seconds": 25.0})
    run = tmp_path / "run-x"
    run.mkdir()
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    log = nest_throughput.find_progress_log(tmp_path)
    assert log == run / "events.jsonl"
    result = nest_throughput.woof_throughput(nest_throughput._records(log),
                                             cells=100, warmup_steps=10)
    assert result["window_steps"] == 40
    assert result["seconds_per_step"] == pytest.approx(0.25)
    assert result["sim_hours_per_wall_hour"] == pytest.approx(240.0)
    assert result["stage_wall_seconds"] == {"prepare": 92.0, "forecast": 25.0}
    assert result["forecast_stage_sim_hours_per_wall_hour"] == pytest.approx(3000.0 / 25.0)


def test_station_table_reads_receipts(tmp_path):
    day = tmp_path / "d01" / "verification" / "2011-05-20"
    day.mkdir(parents=True)
    receipt = {"schema": "gpuwm.verify-visuals.receipt.v1",
               "station_source": "dynamical-asos-parquet",
               "stations": [{"quantity": "temperature_2m", "units": "K",
                             "arms": [{"label": "WOOF", "count": 590,
                                       "bias": -0.82, "rmse": 2.15}]},
                            {"quantity": "dewpoint_2m", "units": "K",
                             "arms": [{"label": "OTHER", "count": 1,
                                       "bias": 0.0, "rmse": 0.0}]}]}
    (day / "verification_030000.json").write_text(json.dumps(receipt))
    (day / "verification_030000.request.json").write_text("{}")
    table = station_table.rows(station_table.load_receipts(tmp_path))
    assert table == [{"valid": "2011-05-20 03:00Z", "quantity": "temperature_2m",
                      "units": "K", "count": 590, "bias": -0.82, "rmse": 2.15,
                      "source": "dynamical-asos-parquet"}]
    text = station_table.markdown(table)
    assert "| 2011-05-20 03:00Z | temperature_2m | 590 | -0.82 K | 2.15 K |" in text
    assert station_table.main([str(tmp_path / "nothing")]) == 1
