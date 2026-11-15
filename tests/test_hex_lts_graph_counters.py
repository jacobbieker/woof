"""Host-only: a replayed subcycle keeps local time stepping's launch counts.

With ``--local-timestep`` the subcycle's rate-class launches are captured with it
(woof.hex.cuda_subcycle_graphs), so a replay runs no Python and the attachment's
``launch_counts`` would stop at the capture.  The graph controller treats them like
its other counters: the captured subcycle's delta is added on every replay, and the
warm-up passes leave them as they were.
"""

from __future__ import annotations

from types import SimpleNamespace

from woof.hex.cuda_subcycle_graphs import SubcycleGraphs


def _driver(lts):
    return SimpleNamespace(cp=None, regional_v841=None, mixing_calls_v841=5, local_timestep=lts)


def test_counters_carry_the_lts_launch_counts():
    lts = SimpleNamespace(launch_counts={"reflux_settles": 3, "acoustic_substep_class_launches": 14})
    graphs = SubcycleGraphs(_driver(lts))
    before = graphs._counters()
    assert before == (5, 0, 0, 14, 3)  # sorted by name: class launches, then settles
    lts.launch_counts["acoustic_substep_class_launches"] += 28
    lts.launch_counts["reflux_settles"] += 6
    graphs.driver.mixing_calls_v841 += 1
    delta = tuple(a - b for a, b in zip(graphs._counters(), before))
    graphs._set_counters(before)  # a warm pass is rolled back
    assert lts.launch_counts == {"reflux_settles": 3, "acoustic_substep_class_launches": 14}
    graphs._set_counters(tuple(a + b for a, b in zip(before, delta)))  # one replay
    assert lts.launch_counts == {"reflux_settles": 9, "acoustic_substep_class_launches": 42}
    assert graphs.driver.mixing_calls_v841 == 6


def test_counters_without_lts_are_the_three_dycore_counters():
    graphs = SubcycleGraphs(_driver(None))
    assert graphs._counters() == (5, 0, 0)
    graphs._set_counters((7, 0, 0))
    assert graphs.driver.mixing_calls_v841 == 7
