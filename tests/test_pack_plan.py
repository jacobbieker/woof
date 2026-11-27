"""``woof pack-plan``: packing a set of runs onto a runner's GPUs.

CPU-only.  The runner is a table row; the contention model is the
measured rows in :mod:`woof.runtime_estimate` or, where a test needs a
round number, a stand-in patched over ``sharing_bracket``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import pytest

from woof import pack_plan as pp
from woof import runtime_estimate as rte
from woof.runner_cards import GIB, RUNNER_CARDS, runner_from_card

from test_runtime_estimate import _config  # noqa: F401 - shared fixture


def _runner(card="rtx-a6000", **changes):
    runner = runner_from_card(RUNNER_CARDS[card])
    return replace(runner, **changes) if changes else runner


def _job(name, gpu=100.0, fixed=10.0, peak=8 * GIB, columns=162_000,
         road="resident"):
    return pp.Job(name=name, config=f"{name}.toml", fixed_s=(fixed, fixed),
                  gpu_s=(gpu, gpu), peak_bytes=peak, columns=columns,
                  road=road, provenance="measured")


@pytest.fixture
def linear(monkeypatch):
    """Each run slows by 0.5 per extra sibling: two together take 1.5x
    each, so packing two gains 2/1.5."""
    def bracket(copies, *, columns, mps):
        s = 1.0 + 0.5 * (copies - 1)
        return {"copies": copies, "per_run_slowdown": (s, s),
                "measured": True, "basis": "test"}
    monkeypatch.setattr(rte, "sharing_bracket", bracket)


def test_alone_one_card_is_serial(linear):
    jobs = [_job("a"), _job("b"), _job("c")]
    s = pp.simulate(jobs, runner=_runner(), slots=1, mps=False)
    assert s.makespan_s == pytest.approx(3 * 110.0)
    assert s.max_on_card == 1


def test_two_per_card_overlaps_with_the_slowdown(linear):
    jobs = [_job("a", fixed=0.0), _job("b", fixed=0.0)]
    s = pp.simulate(jobs, runner=_runner(), slots=2, mps=False)
    assert s.makespan_s == pytest.approx(150.0)
    assert s.max_on_card == 2


def test_preparation_is_not_slowed_by_sharing(linear):
    jobs = [_job("a", fixed=50.0, gpu=0.0), _job("b", fixed=50.0, gpu=0.0)]
    s = pp.simulate(jobs, runner=_runner(), slots=2, mps=False)
    assert s.makespan_s == pytest.approx(50.0)


def test_more_gpus_spread_the_runs(linear):
    jobs = [_job("a"), _job("b")]
    s = pp.simulate(jobs, runner=_runner(gpus=2), slots=1, mps=False)
    assert s.makespan_s == pytest.approx(110.0)
    assert set(s.device.values()) == {0, 1}


def test_memory_admission_caps_concurrency(linear):
    # 20 GiB peaks on a 48 GiB card with a 1 GiB margin: two fit, not three.
    jobs = [_job(n, peak=20 * GIB) for n in "abcd"]
    s = pp.simulate(jobs, runner=_runner(), slots=4, mps=False)
    assert s.max_on_card == 2


def test_a_run_too_big_for_the_card_refuses_the_option(linear):
    jobs = [_job("big", peak=60 * GIB)]
    s = pp.simulate(jobs, runner=_runner(), slots=1, mps=False)
    assert not s.admitted and s.refused[0]["rule"] in ("measured_free",
                                                       "card_total")
    opts = pp.options(jobs, _runner(), max_slots=2)
    best, note = pp.choose(opts)
    assert best is None and "refuses" in note


def test_memory_other_processes_hold_is_respected(linear):
    busy = _runner(free_vram_bytes=12 * GIB)
    jobs = [_job(n, peak=8 * GIB) for n in "ab"]
    s = pp.simulate(jobs, runner=busy, slots=2, mps=False)
    assert s.max_on_card == 1


def test_unpriced_runs_alone(linear):
    jobs = [_job("a", peak=None), _job("b"), _job("c")]
    s = pp.simulate(jobs, runner=_runner(), slots=3, mps=False,
                    order="as-given")
    assert s.start_s["b"] >= s.finish_s["a"]


def test_every_recommended_option_passes_multi_run_admission():
    """The recommender's admissions are multi-run's own decisions."""
    jobs = [_job(n, peak=14 * GIB) for n in "abcdef"]
    runner = _runner()
    for opt in pp.options(jobs, runner, max_slots=4):
        if not opt.admitted:
            continue
        # Replay: at no moment are more runs on the card than admission
        # allows by memory or by the slot cap.
        sched = opt.high
        events = sorted([(sched.start_s[j], 1, j) for j in sched.start_s]
                        + [(sched.finish_s[j], 0, j) for j in sched.finish_s])
        on = []
        for _, kind, name in events:
            if kind:
                job = next(j for j in jobs if j.name == name)
                d = pp.admit(job, on, runner=runner, cap=opt.slots,
                             margin_bytes=GIB)
                assert d["verdict"] == "admitted", d
                on.append(job)
            else:
                on = [j for j in on if j.name != name]


def test_slot_cap_honours_auto_max_and_cores():
    from woof.gpu_packing import AUTO_MAX_SLOTS_PER_GPU

    assert pp.slot_cap(_runner()) == AUTO_MAX_SLOTS_PER_GPU
    assert pp.slot_cap(_runner(cpu_cores=3)) == 3
    assert pp.slot_cap(_runner(cpu_cores=8, gpus=4)) == 2
    assert pp.slot_cap(_runner(), 2) == 2


def test_streamed_runs_gain_nothing(linear):
    jobs = [_job("a", fixed=0.0, road="streamed"),
            _job("b", fixed=0.0, road="streamed")]
    s = pp.simulate(jobs, runner=_runner(), slots=2, mps=False)
    assert s.makespan_s == pytest.approx(200.0)


def test_choose_prefers_the_fastest_then_the_simplest(linear):
    jobs = [_job(n, fixed=0.0) for n in "abcd"]
    opts = pp.options(jobs, _runner(), max_slots=4, allow_mps=False)
    best, _ = pp.choose(opts)
    # linear: 4 together take 2.5x each -> 250 s against 400 s serial.
    assert best.slots == 4
    assert best.makespan[1] == pytest.approx(250.0)


def test_deadline_filters_options(linear):
    jobs = [_job(n, fixed=0.0) for n in "ab"]
    opts = pp.options(jobs, _runner(), max_slots=2, allow_mps=False)
    best, note = pp.choose(opts, deadline_s=120.0)
    assert note and best.slots == 2  # none meets 120 s; fastest shown
    best, note = pp.choose(opts, deadline_s=160.0)
    assert not note and best.latest_finish_s() <= 160.0


def test_metal_has_no_mps_option():
    jobs = [_job("a")]
    m3 = _runner("m3-max", vram_bytes=64 * GIB)
    assert not any(o.mps for o in pp.options(jobs, m3, max_slots=2))


def test_longest_first_beats_a_bad_order_on_two_cards(linear):
    jobs = [_job("s1", gpu=10.0, fixed=0.0), _job("s2", gpu=10.0, fixed=0.0),
            _job("long", gpu=100.0, fixed=0.0)]
    runner = _runner(gpus=2)
    lpt = pp.simulate(jobs, runner=runner, slots=1, mps=False,
                      order="longest-first")
    assert lpt.makespan_s == pytest.approx(100.0)


def test_emitted_plan_loads_in_multi_run(tmp_path, linear):
    from woof.multi_run import load_plan

    cfg = _config(tmp_path, name="member")
    jobs = [replace(_job(f"m{i}"), config=str(cfg)) for i in range(3)]
    opts = pp.options(jobs, _runner(), max_slots=3, allow_mps=False)
    best, _ = pp.choose(opts)
    plan = tmp_path / "plan.toml"
    plan.write_text(pp.multi_run_plan_text(jobs, best), encoding="utf-8")
    loaded = load_plan(plan)
    assert [r.name for r in loaded.runs] == [
        j.name for j in pp.order_jobs(jobs, best.order)]
    assert pp.invocations(best)[0].startswith(
        f"woof multi-run pack-plan.toml --runs-per-gpu {best.slots}")


def _parse(*argv):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    pp.register_cli(sub)
    return parser.parse_args(["pack-plan", *argv])


@pytest.fixture(autouse=True)
def _no_local_calibration(tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_RUNTIME_CALIBRATION",
                       str(tmp_path / "calibration.json"))


def test_cli_json(tmp_path, capsys, monkeypatch):
    import woof.supervisor as supervisor

    monkeypatch.setattr(supervisor, "priced_reservation_bytes",
                        lambda e: 9 * GIB)
    cfg = _config(tmp_path, name="ens")
    args = _parse(str(cfg), "--copies", "4", "--card", "a6000",
                  "--max-slots", "4", "--json")
    assert args.func(args) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == pp.SCHEMA
    assert len(doc["runs"]) == 4
    rec = doc["recommended"]
    assert rec["admitted"] and rec["runs_per_gpu"] in range(1, 5)
    assert rec["makespan_s"][1] <= doc["one_at_a_time_s"][1] + 1e-6
    assert doc["invocations"][0].startswith("woof multi-run")


def test_cli_text_and_emit_plan(tmp_path, capsys, monkeypatch):
    import woof.supervisor as supervisor

    monkeypatch.setattr(supervisor, "priced_reservation_bytes",
                        lambda e: 9 * GIB)
    cfg = _config(tmp_path, name="ens")
    plan = tmp_path / "pack.toml"
    args = _parse(str(cfg), "--copies", "2", "--card", "a6000",
                  "--max-slots", "2", "--emit-plan", str(plan))
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "recommended:" in out and "one at a time" in out
    assert plan.read_text().startswith('schema = "gpuwm.multi-run-plan/v1"')
