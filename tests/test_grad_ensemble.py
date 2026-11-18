"""woof.grad.ensemble: ensemble sensitivity, regression directions and EFSO."""
from __future__ import annotations

import json

import numpy as np
import pytest

from woof.grad import ensemble as ens


def test_ensemble_sensitivity_recovers_a_linear_cost():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 2, 3, 4))
    J = 3.0 * X[:, 1, 2, 3] - 2.0 * X[:, 0, 0, 0] + 0.01 * rng.standard_normal(400)
    es = ens.ensemble_sensitivity(J, X)
    assert es["S"][1, 2, 3] == pytest.approx(3.0, abs=0.1)
    assert es["S"][0, 0, 0] == pytest.approx(-2.0, abs=0.1)
    assert abs(es["S"][1, 1, 1]) < 0.4 and abs(es["r"][1, 1, 1]) < 0.15
    X[:, 0, 1, 1] = 5.0
    assert np.isnan(ens.ensemble_sensitivity(J, X)["S"][0, 1, 1])
    with pytest.raises(ens.EnsembleError):
        ens.ensemble_sensitivity(J[:2], X[:2])


def test_a_regression_direction_is_one_at_its_key_and_zero_beyond_the_cutoff():
    rng = np.random.default_rng(1)
    K, nz, ny, nx = 30, 4, 9, 10
    thp = rng.standard_normal((K, nz, ny, nx))
    u = rng.standard_normal((K, nz, ny, nx + 1))
    u[:, 2, 4, 6] = 2.0 * thp[:, 2, 4, 5] + 0.1 * rng.standard_normal(K)
    d, info = ens.regression_direction({"thp": thp, "u": u}, field="thp", index=(2, 4, 5), mass_shape=(nz, ny, nx),
                                       dx_m=1000.0, horizontal_m=3500.0, vertical_levels=1.5)
    assert d["thp"][2, 4, 5] == pytest.approx(1.0)
    assert d["u"].shape == (nz, ny, nx + 1)
    # u[.., 6] sits half a cell east of the key: GC(500 m / 3500 m) times a regression of about 2.
    from woof.da.letkf import gaspari_cohn
    assert d["u"][2, 4, 6] == pytest.approx(2.0 * float(gaspari_cohn(500.0, 3500.0)), rel=0.1)
    assert np.all(d["thp"][:, :, 9] == 0.0) and np.all(d["thp"][0] == 0.0)
    assert info["members"] == K
    with pytest.raises(ens.EnsembleError):
        ens.regression_direction({"thp": thp, "w": thp[:, :, :5]}, field="thp", index=(0, 0, 0),
                                 mass_shape=(nz, ny, nx), dx_m=1.0, horizontal_m=2.0, vertical_levels=1.0)


def test_grf_members_are_tapered_centred_and_shaped():
    shapes = {"thp": (3, 20, 24), "u": (3, 20, 25), "v": (3, 21, 24)}
    draws = [ens.grf_increments(shapes, seed=7 + m, dx_km=3.0, length_scale_km=15.0, vertical_scale_levels=1.0,
                                thp_k=0.5, wind_ms=1.0, rim=3) for m in range(4)]
    for name, shape in shapes.items():
        assert draws[0][name].shape == shape
    assert np.all(draws[0]["thp"][:, 0, :] == 0.0) and np.all(draws[0]["u"][:, :, -1] == 0.0)
    again = ens.grf_increments(shapes, seed=7, dx_km=3.0, length_scale_km=15.0, vertical_scale_levels=1.0,
                               thp_k=0.5, wind_ms=1.0, rim=3)
    np.testing.assert_array_equal(again["v"], draws[0]["v"])
    members = ens.centred(draws)
    np.testing.assert_allclose(np.mean([m["thp"] for m in members], axis=0), 0.0, atol=1e-12)


def test_observations_sharing_a_gridpoint_go_to_separate_batches():
    flat = np.array([5, 9, 5, 5, 2])
    batches, rows = ens.layered_batches(flat, np.arange(5.0), np.ones(5), np.zeros((3, 5)), shape=(1, 3, 4),
                                        name="t")
    assert len(batches) == 3
    assert [r.tolist() for r in rows] == [[4, 0, 1], [2], [3]]
    assert batches[0].points.flat_index.tolist() == [2, 5, 9]


def _toy_analysis(localization_m, solve_namespace=None):
    from woof.da.letkf import GridGeometry, LetkfConfig, Localization

    rng = np.random.default_rng(3)
    K, shape = 16, (2, 5, 6)
    prior = {"thp": 300.0 + rng.standard_normal((K,) + shape)}
    flat = np.array([3, 17, 40, 41, 17])
    hb = prior["thp"].reshape(K, -1)[:, flat]
    y = hb.mean(axis=0) + np.array([0.8, -0.5, 0.3, -0.9, -0.2])
    sigma = np.full(5, 0.7)
    batches, rows = ens.layered_batches(flat, y, sigma, hb, shape=shape, name="t")
    grid = GridGeometry(dx_m=100.0, dy_m=100.0, heights_m=np.array([0.0, 1000.0]))
    config = LetkfConfig(localization=Localization(localization_m, 1e6), analysis_fields=("thp",), rtps_alpha=0.0)
    increments, ya = ens.letkf_with_obs_space(prior, batches, rows, grid, config, solve_namespace=solve_namespace)
    return prior, increments, ya, flat, y, sigma, hb


def test_letkf_device_solve_matches_the_host():
    cupy = pytest.importorskip("cupy")
    try:
        cupy.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    _, host_inc, host_ya, *_ = _toy_analysis(3000.0)
    _, dev_inc, dev_ya, *_ = _toy_analysis(3000.0, solve_namespace=cupy)
    np.testing.assert_allclose(dev_ya, host_ya, rtol=0, atol=1e-9)
    np.testing.assert_allclose(dev_inc["thp"], host_inc["thp"], rtol=0, atol=1e-9)


def test_letkf_returns_ya_equal_to_the_analysed_state_for_an_identity_operator():
    prior, increments, ya, flat, *_ = _toy_analysis(3000.0)
    analysed = (prior["thp"] + increments["thp"]).reshape(prior["thp"].shape[0], -1)[:, flat]
    np.testing.assert_allclose(ya, analysed, atol=1e-10)
    assert set(increments) == {"thp"}


def test_efso_matches_the_actual_change_of_a_linear_cost():
    # Unlocalised (cutoff far beyond the grid), linear H and linear J: EFSO is exact.
    prior, increments, ya, flat, y, sigma, hb = _toy_analysis(1e5)
    K = prior["thp"].shape[0]
    c = np.random.default_rng(4).standard_normal(prior["thp"].shape[1:])
    analysis = prior["thp"] + increments["thp"]
    jf = np.tensordot(analysis, c, axes=3)
    actual = float(np.sum(c * (analysis.mean(axis=0) - prior["thp"].mean(axis=0))))
    impact = ens.efso(y - hb.mean(axis=0), sigma, ya, jf)
    assert impact.sum() == pytest.approx(actual, rel=1e-3)
    per_obs = ens.efso(y - hb.mean(axis=0), sigma, ya, np.stack([jf, 0 * jf], axis=1), rho=np.ones((5, 2)))
    np.testing.assert_allclose(per_obs, impact)
    table = ens.impact_table(impact, np.array(["a"] * 5), np.array(["upper_temperature"] * 5))
    assert table["all"]["total"] == pytest.approx(actual, rel=1e-3) and table["a/upper_temperature"]["n"] == 5
    with pytest.raises(ens.EnsembleError):
        ens.efso(y, sigma, ya, jf, rho=np.ones((5, 1)))
    assert K == 16


def test_the_members_and_ens_doors(tmp_path, capsys):
    from woof.grad.cli import main

    shape = (3, 20, 24)
    ckpt = tmp_path / "c.npz"
    np.savez(ckpt, **{"state/thp": np.zeros(shape, np.float32), "state/u": np.zeros((3, 20, 25), np.float32),
                      "state/v": np.zeros((3, 21, 24), np.float32)})
    assert main(["members", "--checkpoint", str(ckpt), "--members", "6", "--seed", "1", "--dx-km", "3",
                 "--length-km", "15", "--vertical-levels", "1", "--rim", "3", "--out-dir", str(tmp_path / "m")]) == 0
    states = sorted(str(p) for p in (tmp_path / "m").glob("member_*.npz"))
    from woof.grad.sensitivity import load_direction
    costs = []
    for n, path in enumerate(states):
        J = 100.0 + 10.0 * float(load_direction(path)[0]["thp"][1, 10, 12])
        costs.append(tmp_path / f"J{n}.json")
        costs[-1].write_text(json.dumps({"J": J}))
    capsys.readouterr()
    assert main(["ens", "--states", *states, "--costs", *map(str, costs), "--out", str(tmp_path / "es.npz"),
                 "--regress", "thp", "1", "10", "12", "--direction-out", str(tmp_path / "p.npz"),
                 "--dx-km", "3"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["regression"]["predicted_dJ_per_unit_scale"] == pytest.approx(10.0)
    assert report["fields"]["thp"]["top_points_by_S_times_std"][0]["index"] == [1, 10, 12]
    with np.load(tmp_path / "es.npz") as data:
        assert data["thp/S"][1, 10, 12] == pytest.approx(10.0)


def test_hex_members_and_cell_regression():
    rng = np.random.default_rng(5)
    lat = 55.0 + rng.uniform(-1, 1, 200)
    lon = -3.0 + rng.uniform(-1.5, 1.5, 200)
    interior = np.ones(200, bool)
    interior[:10] = False
    members = [ens.grf_cells(lat, lon, 3, seed=11 + m, dx_km=5.0, length_scale_km=40.0, vertical_scale_levels=1.0,
                             sigma=0.5, interior=interior) for m in range(12)]
    assert members[0].shape == (200, 3) and np.all(members[0][:10] == 0.0)
    assert 0.2 < float(np.std(np.stack(members))) < 0.8
    X = {"theta": np.stack([m["theta"] for m in ens.centred([{"theta": m} for m in members])])}
    d, info = ens.regression_direction_cells(X, field="theta", cell=50, level=1, lat=lat, lon=lon,
                                             horizontal_km=150.0, vertical_levels=2.0)
    assert d["theta"][50, 1] == pytest.approx(1.0) and info["members"] == 12
    from woof.grad.sensitivity import great_circle_km
    far = great_circle_km(lat, lon, lat[50], lon[50]) >= 150.0
    assert np.all(d["theta"][far] == 0.0)


def test_the_letkf_mean_and_efso_doors_on_hand_built_members(tmp_path, capsys):
    from test_grad_obs_cost import NX, NY, P_LEVELS, _at, _obs, write_wrf

    from woof.grad.cli import main
    from woof.grad.obs_cost import ObsSet

    nz, K = P_LEVELS.size, 5
    shifts = [-1.0, -0.5, 0.0, 0.5, 1.0]
    obs = _obs([{"platform": "sonde", "kind": "upper_temperature", "station": "a", "time": _at(1),
                 "lat": 55.6, "lon": -3.3, "pressure_pa": 85000.0, "value": 286.5, "error": 1.0},
                {"platform": "sonde", "kind": "upper_temperature", "station": "b", "time": _at(1),
                 "lat": 55.7, "lon": -3.2, "pressure_pa": 70000.0, "value": 273.0, "error": 1.0}])
    obs.save(tmp_path / "obs.npz")
    histories, checkpoints = [], []
    for m, shift in enumerate(shifts):
        histories.append(str(write_wrf(tmp_path / f"h{m}.nc", hours=(1,), theta_shift=shift)))
        ckpt = tmp_path / f"m{m}" / "gpuwmrst_d01_2026-11-07_01_00_00.npz"
        ckpt.parent.mkdir()
        np.savez(ckpt, **{"state/thp": np.full((nz, NY, NX), shift, np.float32),
                          "state/u": np.full((nz, NY, NX + 1), 4.0 + shift, np.float32),
                          "state/v": np.full((nz, NY + 1, NX), -2.0, np.float32)})
        checkpoints.append(str(ckpt))
    assert main(["cost", "--history", histories[2], "--obs", str(tmp_path / "obs.npz"), "--max-offset-minutes", "30",
                 "--pairs-out", str(tmp_path / "ref.npz"), "--quiet"]) == 0
    capsys.readouterr()
    assert main(["letkf", "--obs", str(tmp_path / "obs.npz"), "--accept", str(tmp_path / "ref.npz"),
                 "--history", *histories, "--checkpoints", *checkpoints, "--fields", "thp", "u",
                 "--dx-km", "6", "--localization-km", "300", "--vertical-km", "8", "--out-dir", str(tmp_path / "an"),
                 "--progress"]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert "letkf solve:" in captured.err and report["solve"] == "host"
    assert report["observations"] == 2 and report["members"] == K
    assert report["spread_ya_over_sigma"] < report["spread_hb_over_sigma"]
    with np.load(tmp_path / "an" / "efso-inputs.npz") as data:
        assert data["ya"].shape == (K, 2) and np.all(np.isfinite(data["ya"]))
    analysed = sorted((tmp_path / "an").glob("member_*/gpuwmrst_d01_2026-11-07_01_00_00.npz"))
    assert len(analysed) == K
    assert main(["mean", "--checkpoints", *map(str, analysed), "--out", str(tmp_path / "mean.npz")]) == 0
    capsys.readouterr()
    pairs = []
    digest = ObsSet.load(tmp_path / "obs.npz").digest()
    for m in range(K):
        path = tmp_path / f"fc{m}.npz"
        np.savez(path, obs_index=np.array([0, 1]), h=np.array([286.0 + shifts[m], 272.0 + shifts[m]]),
                 y=np.array([286.5, 273.0]), sigma=np.array([1.0, 1.0]), obs_sha256=np.asarray(digest))
        pairs.append(str(path))
    stranger = tmp_path / "stranger.npz"
    with np.load(pairs[0]) as data:
        np.savez(stranger, **dict(data, obs_sha256=np.asarray("other")))
    with pytest.raises(SystemExit, match="another observation file"):
        main(["efso", "--analysis", str(tmp_path / "an" / "efso-inputs.npz"), "--obs", str(tmp_path / "obs.npz"),
              "--forecast-pairs", str(stranger), *pairs[1:]])
    assert main(["efso", "--analysis", str(tmp_path / "an" / "efso-inputs.npz"), "--obs", str(tmp_path / "obs.npz"),
                 "--forecast-pairs", *pairs, "--localization-km", "500"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["assimilated"] == 2 and np.isfinite(out["predicted_dJ"])
    assert out["by_type"]["sonde/upper_temperature"]["n"] == 2
    assert ObsSet.load(tmp_path / "obs.npz").digest()


def test_efso_chunks_and_refusals():
    from woof.da.letkf import LetkfConfig, Localization

    rng = np.random.default_rng(3)
    K, n, m = 6, 7, 5
    ya, jf, rho = rng.normal(size=(K, n)), rng.normal(size=(K, m)), rng.uniform(size=(n, m))
    d, s = rng.normal(size=n), rng.uniform(0.5, 2.0, size=n)
    whole = ens.efso(d, s, ya, jf, rho=rho)
    np.testing.assert_allclose(ens.efso(d, s, ya, jf, rho=lambda rows: rho[rows], chunk_rows=2), whole)
    np.testing.assert_allclose(ens.efso(d, s, ya, jf), ens.efso(d, s, ya, jf.sum(axis=1)))
    config = LetkfConfig(localization=Localization(1e5, 1e6), analysis_fields=("thp",), rtps_alpha=0.5)
    prior = {"thp": np.zeros((K, 2, 3, 3))}
    with pytest.raises(ens.EnsembleError, match="unrelaxed"):
        ens.letkf_with_obs_space(prior, [], [], None, config)
