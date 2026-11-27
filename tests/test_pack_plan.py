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
    def bracket(copies, *, columns, mps, card=None):
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


def test_longest_first_beats_a_bad_order_on_one_card_with_two_slots(linear):
    # Device assignment is longest-first whatever the order, so order only
    # matters in a card's queue: with two slots, the two short runs
    # sharing first leave the long one to finish alone after them
    # (15 + 100 s); the long one first shares with one short run, then
    # the other (110 s).
    jobs = [_job("s1", gpu=10.0, fixed=0.0), _job("s2", gpu=10.0, fixed=0.0),
            _job("long", gpu=100.0, fixed=0.0)]
    bad = pp.simulate(jobs, runner=_runner(), slots=2, mps=False,
                      order="as-given")
    lpt = pp.simulate(jobs, runner=_runner(), slots=2, mps=False,
                      order="longest-first")
    assert bad.makespan_s == pytest.approx(115.0)
    assert lpt.makespan_s == pytest.approx(110.0)


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
    assert pp.invocations(best, str(plan))[0].startswith(
        f"woof multi-run {plan} --runs-per-gpu {best.slots}")
    # No plan written: an obvious placeholder, not a file that is not there.
    assert pp.NO_PLAN in pp.invocations(best)[0]


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


def test_a_measured_option_wins_when_the_prediction_cannot_separate(
        monkeypatch):
    def bracket(copies, *, columns, mps, card=None):
        s = copies * (0.99 if copies == 3 else 1.0)  # time-slicing
        return {"copies": copies, "per_run_slowdown": (s, s),
                "measured": copies < 3, "basis": "test"}
    monkeypatch.setattr(rte, "sharing_bracket", bracket)
    jobs = [_job(n, fixed=0.0) for n in "abc"]
    opts = pp.options(jobs, _runner(), max_slots=3, allow_mps=False)
    # Three together is 1 % faster but unmeasured; one and two are
    # measured and as fast as each other.
    assert len({o.slots for o in opts}) == 3
    assert len(opts) == 3  # identical runs: one order per setting
    best, _ = pp.choose(opts)
    assert not (best.low.unmeasured or best.high.unmeasured)
    assert best.slots == 1


@pytest.mark.parametrize("gain,mps_wins", [(0.5, True), (0.97, False)])
def test_mps_is_recommended_when_its_measured_row_wins_beyond_the_band(
        monkeypatch, gain, mps_wins):
    def bracket(copies, *, columns, mps, card=None):
        s = (1.0 + 0.5 * (copies - 1)) * (gain if mps else 1.0)
        return {"copies": copies, "per_run_slowdown": (s, s),
                "measured": True, "basis": "test"}
    monkeypatch.setattr(rte, "sharing_bracket", bracket)
    jobs = [_job(n, fixed=0.0) for n in "ab"]
    best, _ = pp.choose(pp.options(jobs, _runner(), max_slots=2))
    assert best.mps is mps_wins


def test_one_per_card_with_more_runs_than_cards_is_not_a_multi_run_plan(
        tmp_path, capsys, monkeypatch, linear):
    jobs = [_job(n) for n in "abc"]
    serial = next(o for o in pp.options(jobs, _runner(), max_slots=1))
    lines = pp.invocations(serial, jobs=jobs, gpus=1)
    assert lines[0] == ("CUDA_VISIBLE_DEVICES=0 sh -c "
                        "'woof go a.toml && woof go b.toml && "
                        "woof go c.toml' &")
    assert not pp.multi_run_queues(serial, 3, 1)
    assert pp.multi_run_queues(serial, 2, 2)
    assert pp.invocations(serial, jobs=jobs[:2], gpus=2)[0].startswith(
        "woof multi-run")
    import woof.supervisor as supervisor

    monkeypatch.setattr(supervisor, "priced_reservation_bytes",
                        lambda e: 9 * GIB)
    cfg = _config(tmp_path, name="ens")
    plan = tmp_path / "pack.toml"
    args = _parse(str(cfg), "--copies", "2", "--card", "a6000",
                  "--max-slots", "1", "--emit-plan", str(plan))
    assert args.func(args) == 0
    assert "not written" in capsys.readouterr().out and not plan.exists()


def test_printed_commands_run_what_simulate_predicts_on_two_cards(linear):
    # Three runs on two cards at one per card: simulate() puts a pair on
    # one card and one on the other, in parallel; the commands are one
    # queue per card, the cards in parallel.
    jobs = [_job(n, gpu=100.0, fixed=0.0) for n in "abc"]
    runner = _runner(gpus=2)
    opt = next(o for o in pp.options(jobs, runner, max_slots=1,
                                     allow_mps=False)
               if o.order == "as-given")
    assert opt.high.makespan_s == pytest.approx(200.0)
    lines = pp.invocations(opt, jobs=jobs, gpus=2)
    queues = [ln for ln in lines if ln.startswith("CUDA_VISIBLE_DEVICES=")]
    assert len(queues) == 2 and all(ln.endswith("&") for ln in queues)
    by_card = {int(q.split("=")[1].split()[0]): q.count("woof go")
               for q in queues}
    assert sorted(by_card.values()) == [1, 2]
    assert by_card == {c: sum(1 for d in opt.high.device.values() if d == c)
                       for c in by_card}


def test_cli_names_the_plan_it_wrote_or_a_placeholder(
        tmp_path, capsys, monkeypatch):
    import woof.supervisor as supervisor

    monkeypatch.setattr(supervisor, "priced_reservation_bytes",
                        lambda e: 9 * GIB)
    cfg = _config(tmp_path, name="ens")
    plan = tmp_path / "other.toml"
    args = _parse(str(cfg), "--copies", "2", "--card", "a6000",
                  "--max-slots", "2", "--emit-plan", str(plan))
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert f"woof multi-run {plan}" in out and "pack-plan.toml" not in out
    args = _parse(str(cfg), "--copies", "2", "--card", "a6000",
                  "--max-slots", "2")
    assert args.func(args) == 0
    assert pp.NO_PLAN in capsys.readouterr().out


def test_same_stem_configs_get_unique_deterministic_names(tmp_path):
    for d in "ab":
        (tmp_path / d).mkdir()
        (tmp_path / d / "member.toml").write_text("")
    (tmp_path / "solo.toml").write_text("")
    paths = [tmp_path / "a" / "member.toml", tmp_path / "b" / "member.toml",
             tmp_path / "solo.toml"]
    assert pp.run_names(paths) == ["a-member", "b-member", "solo"]
    assert pp.run_names([paths[0], paths[0]]) == ["a-member-1", "a-member-2"]
    assert pp.run_names(paths) == pp.run_names(paths)


def test_json_is_strict_when_an_option_is_refused(tmp_path, capsys,
                                                  monkeypatch):
    import woof.supervisor as supervisor

    monkeypatch.setattr(supervisor, "priced_reservation_bytes",
                        lambda e: 60 * GIB)
    cfg = _config(tmp_path, name="big")
    args = _parse(str(cfg), "--card", "a6000", "--max-slots", "2", "--json")
    args.func(args)

    def refuse(name):
        raise ValueError(f"non-standard JSON token {name}")
    doc = json.loads(capsys.readouterr().out, parse_constant=refuse)
    refused = [o for o in doc["options"] if not o["admitted"]]
    assert refused and refused[0]["makespan_s"] == [None, None]
    assert refused[0]["central_makespan_s"] is None


def test_a_streamed_run_is_priced_as_pure_time_slicing(monkeypatch):
    def bracket(*a, **k):
        raise AssertionError("streamed runs do not use the measured rows")
    monkeypatch.setattr(rte, "sharing_bracket", bracket)
    jobs = [replace(_job(n, fixed=0.0), road="streamed") for n in "ab"]
    packed = pp.simulate(jobs, runner=_runner(), slots=2, mps=False)
    alone = pp.simulate(jobs, runner=_runner(), slots=1, mps=False)
    assert packed.admitted and packed.makespan_s == pytest.approx(
        alone.makespan_s)


def test_every_printed_command_line_is_copy_pasteable(linear):
    import shlex

    jobs = [_job(n) for n in "abc"]
    for runner, js, slots in ((_runner(gpus=2), jobs, 1),
                              (_runner(), jobs, 1), (_runner(), jobs, 2)):
        opt = next(o for o in pp.options(js, runner, max_slots=slots)
                   if o.slots == slots)
        for plan in (None, "p lan.toml"):
            for line in pp.invocations(opt, plan, jobs=js,
                                       gpus=runner.gpus):
                if line.startswith("#"):
                    continue
                shlex.split(line)  # balanced quoting
                assert "(" not in line and "..." not in line


def test_plan_round_trips_paths_with_quotes_and_backslashes(linear):
    import tomllib

    odd = ['dir/"quoted".toml', "back\\slash.toml", 'a "b" \\c.toml']
    jobs = [replace(_job(f"m{i}"), config=c) for i, c in enumerate(odd)]
    best, _ = pp.choose(pp.options(jobs, _runner(), max_slots=3,
                                   allow_mps=False))
    doc = tomllib.loads(pp.multi_run_plan_text(jobs, best))
    assert sorted(r["config"] for r in doc["run"]) == sorted(odd)


def test_unknown_card_memory_is_refused_with_a_message(
        tmp_path, capsys, monkeypatch):
    import woof.runtime_estimate as est

    monkeypatch.setattr(est, "resolve_runner",
                        lambda args: _runner(vram_bytes=None))
    cfg = _config(tmp_path, name="ens")
    args = _parse(str(cfg), "--card", "a6000")
    assert args.func(args) == 2
    err = capsys.readouterr().err
    assert "memory is unknown" in err and "--card" in err


def test_printed_packed_commands_carry_a_non_default_margin(linear):
    import shlex

    jobs = [_job(n) for n in "ab"]
    opt = next(o for o in pp.options(jobs, _runner(), max_slots=2,
                                     allow_mps=False) if o.slots == 2)
    lines = pp.invocations(opt, "p.toml", jobs=jobs, margin_gib=4.0)
    assert lines[0].endswith("--runs-per-gpu 2 --mem-margin-gib 4")
    assert lines[1].endswith("--slots-per-gpu 2 --mem-margin-gib 4")
    assert shlex.split(lines[0])[-2:] == ["--mem-margin-gib", "4"]
    assert "--mem-margin-gib" not in pp.invocations(
        opt, "p.toml", jobs=jobs, margin_gib=1.0)[0]
    assert "--mem-margin-gib" not in pp.invocations(
        opt, "p.toml", jobs=jobs)[0]


@pytest.mark.parametrize("as_json", [False, True])
def test_unmet_deadline_exits_3(tmp_path, capsys, monkeypatch, as_json):
    import woof.supervisor as supervisor

    monkeypatch.setattr(supervisor, "priced_reservation_bytes",
                        lambda e: 9 * GIB)
    cfg = _config(tmp_path, name="ens")
    argv = [str(cfg), "--card", "a6000", "--max-slots", "2",
            "--deadline-hours", "0.0001"] + (["--json"] if as_json else [])
    args = _parse(*argv)
    assert args.func(args) == pp.EXIT_DEADLINE_MISSED == 3
    out = capsys.readouterr().out
    if as_json:
        assert json.loads(out)["deadline_met"] is False
    args = _parse(str(cfg), "--card", "a6000", "--max-slots", "2",
                  "--deadline-hours", "1000")
    assert args.func(args) == 0


def test_emit_plan_to_a_missing_directory_is_a_clean_error(
        tmp_path, capsys, monkeypatch):
    import woof.supervisor as supervisor

    monkeypatch.setattr(supervisor, "priced_reservation_bytes",
                        lambda e: 9 * GIB)
    cfg = _config(tmp_path, name="ens")
    args = _parse(str(cfg), "--copies", "2", "--card", "a6000",
                  "--max-slots", "2", "--emit-plan",
                  str(tmp_path / "nope" / "p.toml"))
    assert args.func(args) == 2
    assert "cannot write" in capsys.readouterr().err
