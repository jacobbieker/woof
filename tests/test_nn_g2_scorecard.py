"""tools/nn_g2_scorecard.py: the station-block bootstrap of the G2 T2 scorecard."""

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "tools" / "nn_g2_scorecard.py"
_SPEC = importlib.util.spec_from_file_location("nn_g2_scorecard", _PATH)
scorecard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scorecard)


def _pairs(rng, stations=40, per=25):
    station = np.repeat(np.arange(stations), per)
    base = rng.normal(0.0, 1.5, station.size) + rng.normal(0.0, 1.0, stations)[station]
    return station, base


def test_identical_arms_have_a_zero_difference_and_interval():
    station, base = _pairs(np.random.default_rng(0))
    out = scorecard.station_bootstrap(station, base, base.copy(), draws=200)
    assert out["difference"] == 0.0 and out["interval"] == [0.0, 0.0] and out["stations"] == 40


def test_a_uniformly_better_arm_has_an_interval_below_zero():
    station, base = _pairs(np.random.default_rng(1))
    out = scorecard.station_bootstrap(station, base, 0.8 * base, draws=500)
    assert out["difference"] < 0.0 and out["interval"][1] < 0.0
    assert out["interval"][0] <= out["difference"] <= out["interval"][1]


def test_the_bootstrap_draws_stations_not_pairs():
    # one station carries the whole improvement: resampling stations must widen the interval to cover 0
    rng = np.random.default_rng(2)
    station, base = _pairs(rng, stations=30, per=40)
    test = base.copy()
    test[station == 0] *= 0.2
    out = scorecard.station_bootstrap(station, base, test, draws=2000, level=0.95)
    assert out["difference"] < 0.0 and out["interval"][1] >= 0.0


def test_the_seed_fixes_the_interval():
    station, base = _pairs(np.random.default_rng(3))
    test = base + np.random.default_rng(4).normal(0.0, 0.3, base.size)
    a = scorecard.station_bootstrap(station, base, test, draws=300, seed=7)
    b = scorecard.station_bootstrap(station, base, test, draws=300, seed=7)
    assert a == b


def test_model_t2_is_moved_to_the_station_with_the_lapse_rate():
    from types import SimpleNamespace

    frame = SimpleNamespace(t2=np.array([280.0, 290.0]), terrain=np.array([500.0, 0.0]))
    obs = SimpleNamespace(elevation_m=np.array([0.0, 1000.0]))
    pairing = SimpleNamespace(obs_index=np.array([0, 1]), frame_index=np.array([0, 0]), column=np.array([0, 1]))
    t2 = scorecard._model_t2(obs, [frame], pairing)
    # a cell 500 m above its station warms by 3.25 K; a station 1000 m above its cell cools by 6.5 K
    np.testing.assert_allclose(t2, [283.25, 283.5])
