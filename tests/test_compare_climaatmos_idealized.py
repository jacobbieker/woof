"""CPU tests for the woof vs ClimaAtmos idealized-benchmark analysis."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools.compare.climaatmos_idealized import analysis as an


def _schar_dhdx(x):
    a, lam, hm = an.SCHAR_A, an.SCHAR_LAMBDA, an.SCHAR_H_MAX
    g = np.exp(-(x / a) ** 2)
    c = np.cos(np.pi * x / lam)
    s = np.sin(np.pi * x / lam)
    return hm * (-2.0 * x / a ** 2 * g * c ** 2
                 - g * 2.0 * c * s * np.pi / lam)


def test_schar_surface_w_is_kinematic_bc():
    # At eta = 0 the linear solution reduces to w = U dh/dx.
    x = np.linspace(-20e3, 20e3, 161)
    _, w = an.schar_steady_state(x, an.schar_height(x), z_top=21e3)
    ref = an.SCHAR_U * _schar_dhdx(x)
    assert np.abs(w - ref).max() < 2e-3 * np.abs(ref).max()


def test_schar_wave_tilts_upstream_and_decays_small_scales():
    # Hydrostatic large-scale part propagates (phase tilts upstream); the
    # lambda = 4 km ridges (k U > N) are evanescent, so the response aloft
    # is much smoother and weaker than at the surface.
    x = np.linspace(-30e3, 30e3, 241)
    _, w0 = an.schar_steady_state(x, np.full_like(x, 50.0), z_top=21e3)
    _, w5 = an.schar_steady_state(x, np.full_like(x, 5000.0), z_top=21e3)
    assert np.abs(w5).max() < 0.5 * np.abs(w0).max()
    assert np.abs(w5).max() > 1e-3


def test_schar_quadrature_converged():
    x = np.linspace(-10e3, 10e3, 41)
    z = np.full_like(x, 3000.0)
    _, w1 = an.schar_steady_state(x, z, z_top=21e3, n_k=2000)
    _, w2 = an.schar_steady_state(x, z, z_top=21e3, n_k=8000)
    assert np.abs(w1 - w2).max() < 1e-4 * np.abs(w2).max()


def test_schar_error_metrics_window():
    z = np.array([100.0, 5000.0, 12000.0, 15000.0])
    ref = np.array([1.0, -1.0, 0.5, 9.0])
    model = np.array([1.1, -0.9, 0.5, -9.0])   # last point is in the sponge
    m = an.schar_error_metrics(model, ref, z)
    assert m["n_points"] == 3
    assert m["rel_l2"] == pytest.approx(np.sqrt(0.02) / 1.5)
    assert m["max_abs_err"] == pytest.approx(0.1)


def test_constant_n_theta_has_target_stability():
    z = np.array([1000.0, 1010.0])
    th = an.constant_n_theta(z)
    n2 = an.GRAV * np.diff(np.log(th))[0] / 10.0
    assert n2 == pytest.approx(an.SCHAR_N ** 2, rel=2e-3)
    p, T = an.background_p_and_T(0.0)
    assert float(p) == pytest.approx(an.MSLP)
    assert float(T) == pytest.approx(an.SCHAR_T_SFC)


def test_eta_levels_monotone_and_pinned():
    zf = np.linspace(0.0, 6400.0, 11)
    eta = an.eta_levels_for_heights(zf, an.isentropic_theta, 1e5, 6400.0)
    assert eta[0] == 1.0 and eta[-1] == 0.0
    assert np.all(np.diff(eta) < 0)
    with pytest.raises(ValueError):
        an.eta_levels_for_heights(zf[::-1], an.isentropic_theta, 1e5, 6400.0)


def test_density_current_ics():
    exner = np.array([1.0])
    s = an.density_current_theta_prime(np.array([0.0]), np.array([3000.0]),
                                       exner, ic="straka")
    c = an.density_current_theta_prime(np.array([0.0]), np.array([2000.0]),
                                       1.0, ic="clima")
    assert s[0] == pytest.approx(-15.0) and c[0] == pytest.approx(-15.0)
    out = an.density_current_theta_prime(np.array([5000.0]),
                                         np.array([3000.0]), exner)
    assert out[0] == 0.0
    with pytest.raises(ValueError):
        an.density_current_theta_prime(0.0, 0.0, 1.0, ic="nope")


def test_front_position_handles_duplicate_unsorted_nodes():
    x = np.array([0.0, 1000.0, 2000.0, 3000.0, 2000.0, 4000.0])
    t = np.array([-5.0, -3.0, -2.0, 0.0, -2.0, 0.0])
    # crossing of -1 K between 2 km (-2) and 3 km (0): 2.5 km
    assert an.front_position_km(x, t) == pytest.approx(2.5)
    assert np.isnan(an.front_position_km(x, np.zeros_like(t)))


def test_density_current_metrics_symmetric_cloud():
    x = np.linspace(-20e3, 20e3, 81)
    z = np.array([50.0, 150.0])
    X, Z = np.meshgrid(x, z)
    thp = np.where(np.abs(X) < 15e3, -3.0, 0.0) - 0.0 * Z
    m = an.density_current_metrics(X, Z, 300.0 + thp)
    assert m["theta_min"] == pytest.approx(-3.0)
    assert m["front_km"] == pytest.approx(14.5 + 0.5 * 2.0 / 3.0)
    assert m["front_asym_km"] < 1e-9
    e = an.straka_errors(m)
    assert e["theta_min_err"] == pytest.approx(-3.0 - an.STRAKA_REF["theta_min"])


def test_summarize_timing_runs(tmp_path):
    woof = {"model": "woof", "case": "dc", "ic": "straka", "mode": "timing",
            "unique_points": 21600, "dt": 1.0, "sec_per_step": 0.01,
            "point_steps_per_sec": 2.16e6, "sim_seconds_per_wall_second": 100.0,
            "nan": False}
    ca = {"model": "ClimaAtmos.jl", "config_file": "/x/plane_schar_mountain_float32_test.yml",
          "mode": "timing", "straka_ic": False, "unique_points": 30000,
          "dt": 0.7, "sec_per_step": 0.02, "point_steps_per_sec": 1.5e6,
          "sim_seconds_per_wall_second": 35.0, "nan": False}
    for name, r in (("woof_dc", woof), ("ca_schar", ca)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "result.json").write_text(json.dumps(r))
    results = an.summarize(tmp_path)
    kinds = {r["run"]: r["case_kind"] for r in results}
    assert kinds == {"woof_dc": "dc_straka", "ca_schar": "schar"}
    md = (tmp_path / "summary.md").read_text()
    assert "| woof_dc | woof | dc_straka | timing | 21600 |" in md
    assert json.loads((tmp_path / "summary.json").read_text())


def test_density_current_metrics_level_mask_with_moving_levels():
    # woof's lowest-level heights move with the cold pool: selecting the
    # surface by min(z) would keep only the coldest columns.
    x = np.linspace(-20e3, 20e3, 81)
    z0 = 35.0 - np.where(np.abs(x) < 15e3, 1.0, 0.0)
    X = np.vstack([x, x])
    Z = np.vstack([z0, z0 + 70.0])
    thp = np.vstack([np.where(np.abs(x) < 15e3, -3.0, 0.0), np.zeros_like(x)])
    level0 = np.zeros(X.shape, dtype=bool)
    level0[0] = True
    m = an.density_current_metrics(X, Z, 300.0 + thp, sfc=level0)
    assert m["front_km"] == pytest.approx(14.5 + 0.5 * 2.0 / 3.0)


def test_ca_interior_faces_layout():
    ncol, nf = 3, 5
    zf = np.tile(np.arange(nf, dtype=float) * 10.0, ncol)
    f = {"sfc_z": np.zeros(ncol), "f_z": zf}
    mask = an.ca_interior_faces(f).reshape(ncol, nf)
    assert not mask[:, 0].any() and not mask[:, -1].any()
    assert mask[:, 1:-1].all()
    with pytest.raises(ValueError):
        an.ca_interior_faces({"sfc_z": np.zeros(ncol), "f_z": zf[::-1]})


def test_summarize_skips_unscorable_runs(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "result.json").write_text(json.dumps(
        {"model": "woof", "case": "schar", "mode": "accuracy",
         "unique_points": 1}))
    assert an.summarize(tmp_path) == []
