"""Host-only tests of the subcycle graph key cache under the hex adaptive clock.

woof.hex.cuda_subcycle_graphs keys a captured dynamics subcycle by everything the eager body bakes into its
launches, the step length included.  A fixed-step run keeps one graph (unchanged); an adaptive run keeps a
graph per repeated step length and runs a step length it has seen once (the ramp, a landing) eager.  These
tests drive the controller's key bookkeeping with a stand-in driver: no device, no capture.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from woof.hex import cuda_subcycle_graphs as graphs_module
from woof.hex.cuda_subcycle_graphs import SubcycleGraphs, attach_subcycle_graphs


class _Array:
    pass


class _Stream:
    def is_capturing(self):
        return False

    def wait_event(self, event):
        pass


class _Event:
    def __init__(self):
        self.synchronized = 0

    def synchronize(self):
        self.synchronized += 1


def _cp():
    cuda = SimpleNamespace(get_current_stream=lambda: _Stream())
    return SimpleNamespace(ndarray=_Array, cuda=cuda)


def _driver():
    atmosphere = SimpleNamespace(mesh=None, vertical=None, reference=None, terrain=None)
    return SimpleNamespace(
        cp=_cp(), atmosphere=atmosphere, regional_v841=None, halo_exchanger_v841=None,
        config=SimpleNamespace(config_dt=18.0), v841_context=None, cache=SimpleNamespace(post_launch=None),
        mixing_calls_v841=0, _advance_dynamics_subcycle_v841=lambda state, **kwargs: ("eager", kwargs["outer_dt"]),
    )


def _call(graphs, dt):
    eager_calls = []

    def eager(state, **kwargs):
        eager_calls.append(kwargs["outer_dt"])
        return ("eager", kwargs["outer_dt"])

    state = SimpleNamespace(time_seconds=0.0)
    result = graphs.advance(eager, state, schedule="s", outer_dt=dt)
    return result, eager_calls


def test_fixed_step_attach_keeps_one_graph_and_captures_at_once(monkeypatch):
    for name in ("WOOF_HEX_CUDA_GRAPHS_KEYS", "WOOF_HEX_CUDA_GRAPHS_CAPTURE_AFTER"):
        monkeypatch.delenv(name, raising=False)
    graphs = attach_subcycle_graphs(_driver())
    assert (graphs.keys, graphs.capture_after) == (1, 1)
    assert "graph_keys" not in graphs.stats  # the fixed-step receipt is unchanged


def test_adaptive_attach_policy_and_overrides(monkeypatch):
    for name in ("WOOF_HEX_CUDA_GRAPHS_KEYS", "WOOF_HEX_CUDA_GRAPHS_CAPTURE_AFTER"):
        monkeypatch.delenv(name, raising=False)
    graphs = attach_subcycle_graphs(_driver(), adaptive=True)
    assert (graphs.keys, graphs.capture_after) == (graphs_module.ADAPTIVE_GRAPH_KEYS,
                                                    graphs_module.ADAPTIVE_CAPTURE_AFTER)
    monkeypatch.setenv("WOOF_HEX_CUDA_GRAPHS_KEYS", "3")
    monkeypatch.setenv("WOOF_HEX_CUDA_GRAPHS_CAPTURE_AFTER", "1")
    graphs = attach_subcycle_graphs(_driver(), adaptive=True)
    assert (graphs.keys, graphs.capture_after) == (3, 1)
    monkeypatch.setenv("WOOF_HEX_CUDA_GRAPHS_KEYS", "0")
    with pytest.raises(ValueError, match="positive"):
        attach_subcycle_graphs(_driver(), adaptive=True)


def test_adaptive_graphs_switch(monkeypatch):
    monkeypatch.delenv("WOOF_HEX_CUDA_GRAPHS_ADAPTIVE", raising=False)
    assert graphs_module.adaptive_graphs_enabled()
    monkeypatch.setenv("WOOF_HEX_CUDA_GRAPHS_ADAPTIVE", "0")
    assert not graphs_module.adaptive_graphs_enabled()


def test_a_step_length_seen_once_runs_eager_without_staging():
    graphs = SubcycleGraphs(_driver(), keys=2, capture_after=2)
    for dt in (18.0, 18.84375, 19.734375):
        graphs.note_step()
        for _ in range(3):  # three dynamics subcycles per step, one sighting
            result, eager = _call(graphs, dt)
            assert result == ("eager", dt) and eager == [dt]
            assert graphs.inputs is None and graphs.graph is None
    assert graphs.stats["eager_subcycles"] == 9
    assert graphs.stats["fallback_reasons"] == {"awaiting_repeat": 9}
    assert graphs.fallbacks == []  # an expected eager step is not a fallback row
    assert set(graphs.sightings.values()) == {1}


def test_switch_parks_restores_and_evicts_graphs():
    graphs = SubcycleGraphs(_driver(), keys=2, capture_after=2)
    graphs.last_launch = _Event()

    def give_graph(name):
        graphs.graph = name
        graphs.inputs = {"staged": name}
        graphs.boundaries[("ru", 1)] = name
        graphs.counter_delta = (1, 2, 3)

    graphs._switch("k36")
    give_graph("g36")
    graphs._switch("k20")
    assert graphs.graph is None and graphs.boundaries == {} and graphs.counter_delta == (0, 0, 0)
    assert list(graphs.parked) == ["k36"] and graphs.parked["k36"]["boundaries"] == {("ru", 1): "g36"}
    give_graph("g20")
    graphs._switch("k36")  # back to the settled step: its graph is reused, not captured again
    assert graphs.graph == "g36" and graphs.inputs == {"staged": "g36"}
    assert graphs.boundaries == {("ru", 1): "g36"} and graphs.stats["parked_reuses"] == 1
    assert list(graphs.parked) == ["k20"]
    graphs._switch("k30")  # keys=2: parking g36 evicts the older parked g20
    assert list(graphs.parked) == ["k36"] and graphs.stats["evicted_graphs"] == 1
    assert graphs.last_launch.synchronized == 3  # every switch away from a live graph drains it


def test_a_disabled_key_stays_disabled_when_it_returns():
    graphs = SubcycleGraphs(_driver(), keys=2, capture_after=2)
    graphs._switch("bad")
    graphs.disabled_key, graphs.disabled_error = "bad", {"type": "X"}
    graphs._switch("good")
    assert graphs.disabled_key is None
    graphs._switch("bad")
    assert graphs.disabled_key == "bad" and graphs.disabled_error == {"type": "X"}


def test_single_key_switch_is_the_old_reset():
    graphs = SubcycleGraphs(_driver())
    graphs.last_launch = _Event()
    graphs._switch("a")
    graphs.graph, graphs.disabled_key = "ga", None
    graphs._switch("b")
    assert graphs.graph is None and graphs.parked == {} and graphs.key == "b"
    graphs.disabled_key = "b"
    graphs._switch("a")
    assert graphs.disabled_key is None and graphs.disabled == {}
