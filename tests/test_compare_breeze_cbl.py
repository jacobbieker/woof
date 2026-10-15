"""CPU tests for the woof/Breeze dry-CBL comparison harness
(tools/compare/breeze_cbl): case definition, diagnostics on synthetic
profiles, throughput table and the matched-config screen."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tools.compare.breeze_cbl import analyze, case


def _synthetic_run(tmp_path, *, zi=1200.0, nz=60, label="synthetic",
                   model="woof", fmt="npz", nt=4, dt=0.5, drag_nz=None):
    """A CBL-shaped profile set: well-mixed below zi, capped by a sharp
    inversion, linear heat flux reaching -0.2 Q at zi, a w-variance bump
    peaking at 0.4 zi."""
    z_f = np.linspace(0.0, case.LZ, nz + 1)
    z_c = 0.5 * (z_f[1:] + z_f[:-1])
    theta = np.where(z_c < zi, 310.0, 310.0 + 2.0 + 0.004 * (z_c - zi))
    wth = np.where(z_c < zi, case.HEAT_FLUX * (1 - 1.2 * z_c / zi),
                   -0.2 * case.HEAT_FLUX * np.exp(-(z_c - zi) / 50.0))
    ww = 2.0 * np.exp(-((z_f - 0.4 * zi) / (0.3 * zi)) ** 2)
    t = np.arange(nt) * 600.0
    arrays = {
        "t_seconds": t,
        "z_f": np.tile(z_f, (nt, 1)),
        "theta": np.tile(theta, (nt, 1)),
        "u": np.full((nt, nz), 8.0), "v": np.full((nt, nz), 0.5),
        "ww": np.tile(ww, (nt, 1)),
        "wth_res": np.tile(wth, (nt, 1)),
        "wth_sgs": np.zeros((nt, nz + 1)),
        "tke_res": np.zeros((nt, nz)), "e_sgs": np.zeros((nt, nz)),
    }
    meta = {"model": model, "label": label, "mode": "simulate", "nx": 8,
            "ny": 8, "nz": nz, "dt": dt, "heat_flux": case.HEAT_FLUX,
            "f": case.coriolis_f(), "ug": case.UG, "vg": case.VG,
            "p_surf": case.P_SURFACE, "float_type": "Float32",
            "drag": "bulk",
            "drag_coefficient": case.drag_coefficient(drag_nz or nz),
            "finite": True}
    stem = tmp_path / label
    (tmp_path / f"{label}.json").write_text(json.dumps(meta))
    if fmt == "npz":
        np.savez(str(stem) + ".npz", **arrays)
    else:   # the Breeze driver's JSON form; z_f stays per sample
        (tmp_path / f"{label}_profiles.json").write_text(json.dumps(
            {k: v.tolist() for k, v in arrays.items()}))
    return stem


def test_case_constants_match_breeze_benchmark():
    assert math.isclose(case.coriolis_f(), 8.05e-5, rel_tol=2e-3)
    assert case.theta_profile(0.0) == 309.0
    assert case.theta_profile(600.0) == 309.0
    assert math.isclose(float(case.theta_profile(1600.0)), 313.0)
    # neutral log law, z1 = dz/2: ~0.0054 at 256^2 x 128 (dz = 23.4 m)
    assert math.isclose(case.drag_coefficient(128),
                        (0.4 / math.log(11.71875 / 0.05)) ** 2)
    assert case.drag_coefficient(192) > case.drag_coefficient(64)


def test_bulk_model_limits():
    assert math.isclose(float(case.bulk_model_zi(0.0)), case.Z_INVERSION)
    # encroachment: h^2 = h0^2 + 2 Q t / gamma
    t = 7200.0
    h = float(case.bulk_model_zi(t, 0.0))
    assert math.isclose(h * h, 600.0 ** 2 + 2 * 0.35 * t / 0.004)
    assert case.bulk_model_zi(t, 0.2) > h


def test_diagnostics_recover_synthetic_cbl(tmp_path):
    run = analyze.load_run(_synthetic_run(tmp_path, zi=1200.0))
    s = analyze.summarise_profile_run(run, window_s=1200.0)
    dz = case.LZ / 60
    assert abs(s["zi_grad_m"] - 1200.0) <= dz
    assert abs(s["zi_flux_m"] - 1200.0) <= dz
    assert math.isclose(s["theta_ml_K"], 310.0)
    assert math.isclose(s["ww_max_m2s2"], 2.0, rel_tol=0.02)
    assert abs(s["ww_max_z_m"] - 480.0) <= dz
    assert -0.21 <= s["wth_min_over_q"] <= -0.15
    assert math.isclose(s["speed_ml_ms"], math.hypot(8.0, 0.5))


def test_json_profiles_load_like_npz(tmp_path):
    a = analyze.load_run(_synthetic_run(tmp_path, label="a", fmt="npz"))
    b = analyze.load_run(_synthetic_run(tmp_path, label="b", fmt="json",
                                        model="breeze"))
    for key in ("theta", "ww", "wth_res", "z_f"):
        np.testing.assert_allclose(a[key], b[key])


def test_window_excludes_initial_sample(tmp_path):
    run = analyze.load_run(_synthetic_run(tmp_path, nt=3))
    run["theta"][0] += 100.0          # t = 0 must never enter the mean
    wm = analyze.window_mean(run, window_s=1e9)
    assert wm["n_samples"] == 2
    assert wm["theta"].max() < 400.0


def test_zi_gradient_ignores_surface_superadiabatic_layer():
    z_c = np.arange(25.0, 3000.0, 50.0)
    theta = np.where(z_c < 1000.0, 310.0, 312.0 + 0.004 * (z_c - 1000.0))
    theta[0] += 3.0                   # strong surface-layer gradient
    assert analyze.zi_gradient(theta, z_c) == pytest.approx(1000.0, abs=50)


def test_check_matched_flags_mismatches():
    good = {"label": "a", "nx": 8, "ny": 8, "nz": 60, "dt": 0.5,
            "heat_flux": 0.35, "f": case.coriolis_f(), "ug": 9.0, "vg": 0.0,
            "p_surf": 101325.0, "float_type": "Float32", "drag": "bulk",
            "drag_coefficient": case.drag_coefficient(60)}
    assert analyze.check_matched([good, dict(good, label="b")]) == []
    bad = dict(good, label="c", dt=0.25, drag_coefficient=0.01)
    problems = analyze.check_matched([good, bad])
    assert any("dt=" in p for p in problems)
    assert any("Cd=" in p for p in problems)
    ustar = dict(good, label="d", drag="ustar", drag_coefficient=None)
    assert analyze.check_matched([good, ustar]) == []


def test_throughput_table_medians_and_ratio():
    def rec(label, model, per_step, n=64):
        return {"mode": "timing", "label": label, "model": model,
                "nx": n, "ny": n, "nz": n // 2, "dt": 0.5,
                "seconds_per_step": per_step, "finite": True}
    rows = analyze.throughput_table([
        rec("woof_km3", "woof", [0.10, 0.12, 0.11]),
        rec("woof_km3", "woof", [0.30, 0.10, 0.10]),   # interleaved repeat
        rec("breeze_anelastic", "breeze", [0.05, 0.05, 0.06]),
    ])
    woof = next(r for r in rows if r["label"] == "woof_km3")
    breeze = next(r for r in rows if r["label"] == "breeze_anelastic")
    assert woof["timed_reps"] == 6 and woof["repeats"] == 2
    assert woof["median_seconds_per_step"] == pytest.approx(0.105)
    assert woof["mpts_steps_per_second"] == pytest.approx(
        64 * 64 * 32 / 0.105 / 1e6)
    assert woof["simulated_seconds_per_wall_second"] == pytest.approx(
        0.5 / 0.105)
    assert breeze["speed_vs_woof_km3"] == pytest.approx(0.105 / 0.05)


def test_main_writes_reports(tmp_path):
    stem = _synthetic_run(tmp_path, label="woof_km3")
    stem2 = _synthetic_run(tmp_path, label="breeze", model="breeze",
                           fmt="json", zi=1300.0)
    tdir = tmp_path / "timing"
    tdir.mkdir()
    (tdir / "w.json").write_text(json.dumps({
        "mode": "timing", "label": "woof_km3", "model": "woof", "nx": 8,
        "ny": 8, "nz": 60, "dt": 0.5, "seconds_per_step": [0.1],
        "finite": True, "heat_flux": 0.35, "f": case.coriolis_f(),
        "ug": 9.0, "vg": 0.0, "p_surf": 101325.0, "float_type": "Float32"}))
    out = tmp_path / "report"
    rc = analyze.main(["--timing", str(tdir), "--profiles", str(stem),
                       str(stem2), "--window-minutes", "20", "--out",
                       str(out)])
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["profiles"]) == 2
    assert summary["throughput"][0]["label"] == "woof_km3"
    for png in ("profiles.png", "evolution.png", "throughput.png"):
        assert (out / png).stat().st_size > 1000
    assert "| woof_km3 |" in (out / "summary.md").read_text()


def test_check_matched_is_relative_for_small_quantities():
    base = {"label": "a", "nx": 8, "ny": 8, "nz": 60, "dt": 0.5,
            "heat_flux": 0.35, "f": case.coriolis_f(), "ug": 9.0,
            "vg": 0.0, "p_surf": 101325.0, "float_type": "Float32"}
    other = dict(base, label="b", f=case.coriolis_f(33.8))   # ~1% in f
    assert any("f=" in p for p in analyze.check_matched([base, other]))


def test_series_style_follows_the_label_not_its_position():
    a = analyze.series_style("breeze_se_matched", ["breeze_se_matched"])
    b = analyze.series_style("breeze_se_matched",
                             ["woof_km3", "woof_km2", "breeze_se_matched"])
    assert a == b
    assert a != analyze.series_style("woof_km3", ["woof_km3"])
