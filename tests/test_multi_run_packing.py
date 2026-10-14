"""CPU-only tests for packing several multi-run forecasts onto one GPU.

Fake GPUs, fake NVML and fake runners: a fake card's device-wide used
memory is its simulated external usage plus whatever the fake runner
processes hold, so admission, waiting and external pressure are exercised
end to end without a device.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

import woof.cli as cli
import woof.multi_run as multi_run
from woof.gpu_packing import AUTO_MAX_SLOTS_PER_GPU, MpsDaemon
from woof.multi_run import (CardMeter, PackingRequest, RunPricing,
                             child_environment, load_plan, packed_admission,
                             packing_request, plan_packing, resolve_devices)
from woof.supervisor import GPUIdentity, GPU_LOCK_ROOT_ENV

GIB = 1024 ** 3


def _inventory() -> tuple[GPUIdentity, ...]:
    return (
        GPUIdentity("GPU-alpha", "600.00", "card alpha", 0),
        GPUIdentity("GPU-beta", "600.00", "card beta", 1),
    )


def _write_plan(tmp_path: Path, *, devices: tuple[object, ...],
                table: str | None = None, preflight: str = "off") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    lines = [
        'schema = "gpuwm.multi-run-plan/v1"',
        'summary = "production-summary.json"',
        f'preflight = "{preflight}"',
    ]
    if table is not None:
        lines.extend(["", "[devices]", table])
    for index, device in enumerate(devices):
        suffix = chr(ord("a") + index)
        (tmp_path / f"config_{suffix}.toml").write_text(
            f"[experiment]\nname = 'fixture_{suffix}'\n", encoding="utf-8")
        lines.extend([
            "",
            "[[run]]",
            f'name = "run_{suffix}"',
            f"device = {device}",
            f'config = "config_{suffix}.toml"',
            f'outdir = "output_{suffix}"',
            f'scratch = "scratch_{suffix}"',
            f'cache = "cache_{suffix}"',
        ])
    plan = tmp_path / "plan.toml"
    plan.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan


class FakeCard:
    """One fake GPU: NVML total, simulated external use, held memory."""

    def __init__(self, total_gib: float, external_gib: float = 0.0):
        self.total = int(total_gib * GIB)
        self.external = int(external_gib * GIB)
        self.held: dict[int, int] = {}
        self.peak_concurrent = 0

    @property
    def used(self) -> int:
        return self.external + sum(self.held.values())


class FakeMeter(CardMeter):
    """Fake NVML: totals, device-wide used and per-PID framebuffer."""

    def __init__(self, cards: dict[str, FakeCard], *,
                 attributes_processes: bool = True):
        self.cards = cards
        self.attributes_processes = attributes_processes
        self.measurements = 0

    def card_total_bytes(self, gpu_uuid):
        card = self.cards.get(gpu_uuid)
        return None if card is None else card.total

    def measure(self, gpu_uuid):
        self.measurements += 1
        card = self.cards[gpu_uuid]
        return {
            "basis": "fake NVML",
            "device_free_bytes": max(0, card.total - card.used),
            "device_total_bytes": card.total,
            "device_wide_used_bytes": card.used,
        }

    def process_bytes(self, gpu_uuid):
        if not self.attributes_processes:
            return {}
        return dict(self.cards[gpu_uuid].held)

    def process_trees(self, roots):
        return {pid: {pid} for pid in roots}


class FakeRunners:
    """A Popen stand-in whose children hold their run's priced memory."""

    def __init__(self, meter: FakeMeter, need: dict[str, float], *,
                 lifetime: int = 3, exit_codes: dict[str, int] | None = None):
        self.meter = meter
        self.need = need
        self.lifetime = lifetime
        self.exit_codes = exit_codes or {}
        self.launched: list[tuple[str, dict]] = []
        self.order: list[str] = []
        self.next_pid = 90_000

    def __call__(self, command, **kwargs):
        outdir = command[command.index("--outdir") + 1]
        name = "run_" + Path(outdir).name.rsplit("_", 1)[1]
        card = self.meter.cards[kwargs["env"]["CUDA_VISIBLE_DEVICES"]]
        self.next_pid += 1
        process = _FakeProcess(self, name, card, self.next_pid)
        card.held[process.pid] = int(self.need[name] * GIB)
        card.peak_concurrent = max(card.peak_concurrent, len(card.held))
        self.launched.append((name, kwargs))
        self.order.append(f"start {name}")
        return process


class _FakeProcess:
    def __init__(self, runners: FakeRunners, name: str, card: FakeCard,
                 pid: int):
        self.runners = runners
        self.name = name
        self.card = card
        self.pid = pid
        self.polls = 0

    def poll(self):
        self.polls += 1
        if self.polls < self.runners.lifetime:
            return None
        if self.pid in self.card.held:
            del self.card.held[self.pid]
            self.runners.order.append(f"end {self.name}")
        return self.runners.exit_codes.get(self.name, 0)

    def terminate(self):  # pragma: no cover - a safety tripwire
        raise AssertionError("multi-run must not terminate children")

    def kill(self):  # pragma: no cover - a safety tripwire
        raise AssertionError("multi-run must not kill children")


def _priced(need: dict[str, float | None]):
    def price(spec):
        value = need[spec.name]
        if value is None:
            return RunPricing(None, "unpriced: fixture")
        return RunPricing(int(value * GIB), "fixture pricing")
    return price


def _packed(tmp_path, monkeypatch, *, cards, need, devices, table,
            attributes_processes=True, lifetime=3, preflight="off"):
    """Plan, pricing, fake NVML and fake runners wired into multi_run."""

    meter = FakeMeter(cards, attributes_processes=attributes_processes)
    runners = FakeRunners(
        meter, {name: value or 0.0 for name, value in need.items()},
        lifetime=lifetime)
    monkeypatch.setenv(GPU_LOCK_ROOT_ENV, str(tmp_path / "locks"))
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)
    monkeypatch.setattr(multi_run, "_card_meter", lambda: meter)
    monkeypatch.setattr(multi_run, "price_run", _priced(need))
    monkeypatch.setattr(multi_run.subprocess, "Popen", runners)
    plan = _write_plan(tmp_path / "plan", devices=devices, table=table,
                       preflight=preflight)
    return plan, meter, runners


def _summary(plan: Path) -> dict:
    return json.loads(
        (plan.parent / "production-summary.json").read_text(encoding="utf-8"))


def _main(plan: Path, **flags) -> int:
    return multi_run.multi_run_main(Namespace(
        plan=plan, summary=None, preflight=None, **flags))


# -- default is today's behaviour ------------------------------------------


def test_default_plan_is_exclusive_and_still_refuses_a_shared_card(tmp_path):
    plan = load_plan(_write_plan(tmp_path, devices=(0, '"GPU-alpha"')))
    assert plan.packing == PackingRequest()
    assert not plan.packing.packed
    with pytest.raises(ValueError, match="unique physical device"):
        resolve_devices(plan, _inventory())


def test_default_run_uses_the_unchanged_executor_and_shared_lock_root(
        tmp_path, monkeypatch):
    monkeypatch.setenv(GPU_LOCK_ROOT_ENV, str(tmp_path / "locks"))
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)

    def no_meter():  # pragma: no cover - the default never measures
        raise AssertionError("an exclusive plan must not measure cards")

    def no_packed(*_args, **_kwargs):  # pragma: no cover - tripwire
        raise AssertionError("an exclusive plan must not pack")

    monkeypatch.setattr(multi_run, "_card_meter", no_meter)
    monkeypatch.setattr(multi_run, "_execute_packed_group", no_packed)
    seen = []

    def execute(runs, *, phase, **_kwargs):
        seen.append([(run.spec.name, run.lock_root) for run in runs])
        records = tuple(multi_run.ProcessRecord(
            run.spec.name, phase, ("x",), run.spec.log, status="succeeded",
            exit_code=0, duration_seconds=1.0) for run in runs)
        return multi_run.GroupOutcome(records, False)

    monkeypatch.setattr(multi_run, "_execute_group", execute)
    plan = _write_plan(tmp_path / "plan", devices=(0, 1))
    assert _main(plan) == 0
    shared = (tmp_path / "locks").resolve()
    assert seen == [[("run_a", shared), ("run_b", shared)]]
    summary = _summary(plan)
    assert summary["packing"]["mode"] == "exclusive"
    assert summary["packing"]["runs_per_gpu"] == 1
    assert "packing" not in summary["runs"][0]


@pytest.mark.parametrize("flags, match", [
    ({"mem_margin_gib": 2.0}, "applies only with runs_per_gpu"),
    ({"mps": True}, "nothing to share"),
    ({"runs_per_gpu": "2", "mps_dir": Path("/tmp/x")},
     "--mps-dir applies only with --mps"),
    ({"runs_per_gpu": "2", "mps": True,
      "mps_active_thread_percentage": 0}, "must be 1-100"),
    ({"runs_per_gpu": "0"}, "positive count or 'auto'"),
    ({"runs_per_gpu": "2", "mem_margin_gib": -1.0}, "nonnegative"),
])
def test_packing_settings_that_would_do_nothing_are_refused(flags, match):
    with pytest.raises(ValueError, match=match):
        packing_request(PackingRequest(), Namespace(**flags))


@pytest.mark.parametrize("table, match", [
    ("runs_per_gpu = 0", "positive count or 'auto'"),
    ("runs_per_gpu = true", "positive count or 'auto'"),
    ('runs_per_gpu = "many"', "positive count or 'auto'"),
    ("slots = 2", "unknown keys"),
    ('mps = "yes"', "true or false"),
    ("runs_per_gpu = 2\nmem_margin_gib = -0.5", "nonnegative"),
])
def test_plan_devices_table_is_validated_never_coerced(tmp_path, table,
                                                        match):
    with pytest.raises(ValueError, match=match):
        load_plan(_write_plan(tmp_path, devices=(0,), table=table))


def test_plan_devices_table_and_flags_compose_with_named_sources(tmp_path):
    plan = load_plan(_write_plan(
        tmp_path, devices=(0, 0), table='runs_per_gpu = "AUTO"\n'
        "mem_margin_gib = 0.5"))
    assert plan.packing.runs_per_gpu == "auto"
    assert plan.packing.cap == AUTO_MAX_SLOTS_PER_GPU
    assert plan.packing.margin_bytes == GIB // 2
    request = packing_request(plan.packing, Namespace(runs_per_gpu="3"))
    assert request.runs_per_gpu == 3
    assert dict(request.sources) == {
        "mem_margin_gib": "plan [devices]", "runs_per_gpu": "--runs-per-gpu"}
    assert len(resolve_devices(
        load_plan(_write_plan(tmp_path / "two", devices=(0, 0),
                              table="runs_per_gpu = 2")),
        _inventory())) == 2


def test_cli_parses_packing_flags():
    args = cli.build_parser().parse_args([
        "multi-run", "plan.toml", "--runs-per-gpu", "auto",
        "--mem-margin-gib", "0.75", "--mps", "--mps-dir", "/tmp/m",
        "--mps-active-thread-percentage", "50"])
    request = packing_request(PackingRequest(), args)
    assert request.runs_per_gpu == "auto"
    assert request.margin_gib == 0.75
    assert request.mps and request.mps_dir == Path("/tmp/m")
    assert request.mps_active_thread_percentage == 50
    plain = cli.build_parser().parse_args(["multi-run", "plan.toml"])
    assert packing_request(PackingRequest(), plain) == PackingRequest()


# -- the admission rule -------------------------------------------------------


def _measured(total_gib, used_gib):
    return lambda: {
        "device_total_bytes": int(total_gib * GIB),
        "device_wide_used_bytes": int(used_gib * GIB),
        "device_free_bytes": int((total_gib - used_gib) * GIB),
        "basis": "fixture"}


def test_admission_counts_only_a_siblings_outstanding_growth():
    sibling = {"name": "a", "priced_bytes": 6 * GIB}
    calls = []

    def observe():
        calls.append("observe")
        return {"a": 2 * GIB}

    def measure():
        calls.append("measure")
        return _measured(24, 8)()

    # 24 GiB card, 8 used (2 by the sibling, 6 by others): free 16, the
    # sibling may still grow 4, so 12 is left for this run's 10 + 1.
    decision = packed_admission(
        need_bytes=10 * GIB, margin_bytes=GIB, card_total_bytes=24 * GIB,
        cap=4, live=[sibling], observe=observe, measure=measure)
    assert decision["verdict"] == "admitted"
    assert decision["siblings_outstanding_bytes"] == 4 * GIB
    assert decision["headroom_bytes"] == 12 * GIB
    # Attribution is sampled before the measurement, never after.
    assert calls == ["observe", "measure"]
    # Unattributed: the sibling's whole priced peak is outstanding.
    unattributed = packed_admission(
        need_bytes=10 * GIB, margin_bytes=GIB, card_total_bytes=24 * GIB,
        cap=4, live=[sibling], observe=dict, measure=_measured(24, 8))
    assert unattributed["verdict"] == "waiting"
    assert unattributed["rule"] == "measured_free"


def test_admission_cap_unpriced_and_sum_rules_never_sample_the_card():
    def measure():  # pragma: no cover - tripwire
        raise AssertionError("not reached")

    row = {"name": "a", "priced_bytes": 4 * GIB}
    kwargs = dict(margin_bytes=GIB, card_total_bytes=16 * GIB,
                  observe=measure, measure=measure)
    assert packed_admission(need_bytes=GIB, cap=1, live=[row],
                            **kwargs)["rule"] == "runs_per_gpu_cap"
    assert packed_admission(need_bytes=None, cap=4, live=[row],
                            **kwargs)["verdict"] == "waiting"
    alone = packed_admission(need_bytes=None, cap=4, live=[], **kwargs)
    assert alone["verdict"] == "admitted-unpriced-alone"
    assert packed_admission(
        need_bytes=GIB, cap=4, live=[{**row, "priced_bytes": None}],
        **kwargs)["rule"] == "unpriced_sibling_alone"
    assert packed_admission(need_bytes=12 * GIB, cap=4, live=[row],
                            **kwargs)["rule"] == "card_total"


def test_measurement_failure_fails_closed_when_alone():
    def broken():
        raise RuntimeError("nvidia-smi timed out")

    kwargs = dict(need_bytes=GIB, margin_bytes=GIB,
                  card_total_bytes=16 * GIB, cap=2, live=[], observe=dict,
                  measure=broken)
    alone = packed_admission(**kwargs)
    assert alone["verdict"] == "refused"
    assert "nvidia-smi timed out" in alone["reason"]
    settling = packed_admission(**kwargs, settling=True)
    assert (settling["verdict"], settling["rule"]) == (
        "waiting", "sibling_memory_settling")


def test_a_just_finished_siblings_memory_is_waited_for_not_refused(
        tmp_path, monkeypatch):
    # run_a exits but its framebuffer is released only a moment later
    # (driver teardown): run_b must wait one recheck, not be refused.
    monkeypatch.setattr(multi_run, "ADMISSION_RECHECK_SECONDS", 0.05)
    cards = {"GPU-alpha": FakeCard(16)}
    plan, meter, runners = _packed(
        tmp_path, monkeypatch, cards=cards,
        need={"run_a": 8, "run_b": 9}, devices=(0, 0),
        table="runs_per_gpu = 2")
    original_measure = meter.measure
    lingering = {}

    def measure(gpu_uuid):
        if "end run_a" in runners.order and "a" not in lingering:
            lingering["a"] = True
            return {**original_measure(gpu_uuid),
                    "device_wide_used_bytes": 8 * GIB,
                    "device_free_bytes": 8 * GIB}
        return original_measure(gpu_uuid)

    meter.measure = measure
    assert _main(plan) == 0
    rules = [(event["run"], event["verdict"], event["rule"])
             for event in _summary(plan)["packing"]["events"]]
    assert rules == [("run_a", "admitted", "fits"),
                     ("run_b", "waiting", "card_total"),
                     ("run_b", "waiting", "sibling_memory_settling"),
                     ("run_b", "admitted", "fits")]
    assert runners.order[-2:] == ["start run_b", "end run_b"]


# -- end to end through multi_run_main ----------------------------------------


def test_two_runs_pack_onto_one_card_concurrently(tmp_path, monkeypatch):
    cards = {"GPU-alpha": FakeCard(24)}
    plan, meter, runners = _packed(
        tmp_path, monkeypatch, cards=cards,
        need={"run_a": 6, "run_b": 6}, devices=(0, 0),
        table="runs_per_gpu = 2")
    assert _main(plan) == 0
    assert cards["GPU-alpha"].peak_concurrent == 2
    assert runners.order[:2] == ["start run_a", "start run_b"]
    environments = [kwargs["env"] for _name, kwargs in runners.launched]
    assert {env["CUDA_VISIBLE_DEVICES"] for env in environments} == {
        "GPU-alpha"}
    roots = [Path(env[GPU_LOCK_ROOT_ENV]) for env in environments]
    assert len(set(roots)) == 2
    shared = (tmp_path / "locks").resolve()
    assert all(root.parent.parent == shared for root in roots)
    # Per-run lock roots are removed once every child was observed.
    assert not roots[0].parent.exists()

    summary = _summary(plan)
    assert summary["status"] == "complete"
    packing = summary["packing"]
    assert packing["mode"] == "packed"
    assert packing["runs_per_gpu"] == 2
    assert packing["settings_source"] == {"runs_per_gpu": "plan [devices]"}
    assert packing["cards"][0]["uuid"] == "GPU-alpha"
    assert packing["cards"][0]["total_bytes"] == 24 * GIB
    assert [row["name"] for row in packing["cards"][0]["runs"]] == [
        "run_a", "run_b"]
    assert [event["verdict"] for event in packing["events"]] == [
        "admitted", "admitted"]
    second = summary["runs"][1]["packing"]
    assert second["pricing"]["priced_peak_bytes"] == 6 * GIB
    run_admission = second["admission"]["run"]
    assert run_admission["verdict"] == "admitted"
    assert run_admission["measured_free_bytes"] == 18 * GIB
    assert run_admission["siblings"][0]["name"] == "run_a"
    assert run_admission["siblings"][0]["observed_bytes"] == 6 * GIB
    assert summary["runs"][1]["gpu_lock_root"] == str(roots[1])


def test_three_runs_on_one_card_queue_behind_the_cap(tmp_path, monkeypatch):
    cards = {"GPU-alpha": FakeCard(48)}
    plan, _meter, runners = _packed(
        tmp_path, monkeypatch, cards=cards,
        need={"run_a": 4, "run_b": 4, "run_c": 4}, devices=(0, 0, 0),
        table="runs_per_gpu = 2")
    assert _main(plan) == 0
    assert cards["GPU-alpha"].peak_concurrent == 2
    assert runners.order.index("start run_c") > runners.order.index(
        "end run_a")
    events = _summary(plan)["packing"]["events"]
    assert [(event["run"], event["verdict"], event["rule"])
            for event in events] == [
        ("run_a", "admitted", "fits"),
        ("run_b", "admitted", "fits"),
        ("run_c", "waiting", "runs_per_gpu_cap"),
        ("run_c", "admitted", "fits"),
    ]


def test_three_runs_pack_together_when_auto_and_memory_allows(
        tmp_path, monkeypatch):
    cards = {"GPU-alpha": FakeCard(24)}
    plan, _meter, _runners = _packed(
        tmp_path, monkeypatch, cards=cards,
        need={"run_a": 6, "run_b": 6, "run_c": 6}, devices=(0, 0, 0),
        table='runs_per_gpu = "auto"')
    assert _main(plan) == 0
    assert cards["GPU-alpha"].peak_concurrent == 3


def test_a_run_that_does_not_fit_waits_for_a_sibling(tmp_path, monkeypatch):
    cards = {"GPU-alpha": FakeCard(16)}
    plan, _meter, runners = _packed(
        tmp_path, monkeypatch, cards=cards,
        need={"run_a": 8, "run_b": 8}, devices=(0, 0),
        table="runs_per_gpu = 4")
    assert _main(plan) == 0
    assert cards["GPU-alpha"].peak_concurrent == 1
    assert runners.order == [
        "start run_a", "end run_a", "start run_b", "end run_b"]
    summary = _summary(plan)
    assert summary["status"] == "complete"
    rules = [(event["run"], event["verdict"], event["rule"])
             for event in summary["packing"]["events"]]
    assert rules == [("run_a", "admitted", "fits"),
                     ("run_b", "waiting", "card_total"),
                     ("run_b", "admitted", "fits")]


def test_external_usage_reduces_what_is_admitted(tmp_path, monkeypatch):
    # Without the external forecast both 6 GiB runs share the 24 GiB card.
    # With 14 GiB held by a process this multi-run does not own, only one
    # fits at a time: the second waits on MEASURED free memory.
    cards = {"GPU-alpha": FakeCard(24, external_gib=14)}
    plan, _meter, runners = _packed(
        tmp_path, monkeypatch, cards=cards,
        need={"run_a": 6, "run_b": 6}, devices=(0, 0),
        table="runs_per_gpu = 2")
    assert _main(plan) == 0
    assert cards["GPU-alpha"].peak_concurrent == 1
    assert runners.order.index("start run_b") > runners.order.index(
        "end run_a")
    events = _summary(plan)["packing"]["events"]
    waiting = [event for event in events if event["verdict"] == "waiting"]
    assert len(waiting) == 1
    assert waiting[0]["rule"] == "measured_free"
    assert waiting[0]["measured_used_bytes"] == 20 * GIB
    assert waiting[0]["measured_free_bytes"] == 4 * GIB


def test_external_usage_that_leaves_no_room_refuses_only_that_run(
        tmp_path, monkeypatch):
    cards = {"GPU-alpha": FakeCard(24, external_gib=20),
             "GPU-beta": FakeCard(24)}
    plan, _meter, runners = _packed(
        tmp_path, monkeypatch, cards=cards,
        need={"run_a": 6, "run_b": 2}, devices=(0, 1),
        table="runs_per_gpu = 2")
    assert _main(plan) == 1
    assert [name for name, _kwargs in runners.launched] == ["run_b"]
    summary = _summary(plan)
    assert summary["status"] == "failed"
    refused = summary["runs"][0]
    assert refused["process"]["status"] == "not_admitted"
    assert "waiting cannot free it" in refused["process"]["error"]
    assert refused["packing"]["admission"]["run"]["verdict"] == "refused"
    assert summary["runs"][1]["process"]["status"] == "succeeded"


def test_an_unpriced_run_runs_alone(tmp_path, monkeypatch):
    cards = {"GPU-alpha": FakeCard(48)}
    plan, _meter, runners = _packed(
        tmp_path, monkeypatch, cards=cards,
        need={"run_a": 2, "run_b": None, "run_c": 2}, devices=(0, 0, 0),
        table="runs_per_gpu = 3")
    assert _main(plan) == 0
    order = runners.order
    assert order.index("start run_b") > order.index("end run_a")
    assert order.index("start run_c") > order.index("end run_b")
    events = _summary(plan)["packing"]["events"]
    assert ("run_b", "admitted-unpriced-alone") in {
        (event["run"], event["verdict"]) for event in events}


def test_a_pack_that_can_never_fit_is_refused_before_anything_is_claimed(
        tmp_path, monkeypatch):
    cards = {"GPU-alpha": FakeCard(24)}
    plan, _meter, runners = _packed(
        tmp_path, monkeypatch, cards=cards,
        need={"run_a": 4, "run_b": 23.5}, devices=(0, 0),
        table="runs_per_gpu = 2")
    with pytest.raises(ValueError, match=r"cannot pack this plan: run "
                       r"'run_b' is priced at 23\.50 GiB.*never be admitted"):
        _main(plan)
    assert runners.launched == []
    assert not (plan.parent / "output_a").exists()
    assert not (plan.parent / "production-summary.json").exists()


def test_a_card_nvml_cannot_size_is_refused(tmp_path, monkeypatch):
    plan, meter, _runners = _packed(
        tmp_path, monkeypatch, cards={"GPU-alpha": FakeCard(24)},
        need={"run_a": 4}, devices=(1,), table="runs_per_gpu = 2")
    with pytest.raises(ValueError, match="GPU-beta.*memory.total"):
        _main(plan)


def _checked(tmp_path, monkeypatch, preflight):
    cards = {"GPU-alpha": FakeCard(16)}
    plan, _meter, runners = _packed(
        tmp_path, monkeypatch, cards=cards,
        need={"run_a": 8, "run_b": 8}, devices=(0, 0),
        table="runs_per_gpu = 2", preflight=preflight)
    original = runners.__call__

    def launcher(command, **kwargs):
        if "check-worker" in command:
            command = [*command, "--outdir",
                       command[command.index("--config") + 1].replace(
                           "config_", "output_").removesuffix(".toml")]
        return original(command, **kwargs)

    monkeypatch.setattr(multi_run.subprocess, "Popen", launcher)
    assert _main(plan) == 0
    summary = _summary(plan)
    assert summary["status"] == "complete"
    assert set(summary["runs"][0]["packing"]["admission"]) == {
        "check", "run"}
    return runners.order, [
        (event["phase"], event["run"], event["verdict"], event["rule"])
        for event in summary["packing"]["events"]]


def test_packed_alloc_checks_are_admitted_by_the_same_rule(
        tmp_path, monkeypatch):
    order, events = _checked(tmp_path, monkeypatch, "alloc")
    assert order == [
        "start run_a", "end run_a", "start run_b", "end run_b",
        "start run_a", "end run_a", "start run_b", "end run_b"]
    assert events[:3] == [("check", "run_a", "admitted", "fits"),
                          ("check", "run_b", "waiting", "card_total"),
                          ("check", "run_b", "admitted", "fits")]


def test_packed_estimate_checks_are_held_only_to_the_cap(
        tmp_path, monkeypatch):
    # An estimate check holds no forecast memory: both checks run at once
    # on the 16 GiB card that runs the two 8 GiB forecasts one at a time.
    order, events = _checked(tmp_path, monkeypatch, "estimate")
    assert order[:2] == ["start run_a", "start run_b"]
    assert events[:2] == [
        ("check", "run_a", "admitted", "estimate_check_cap_only"),
        ("check", "run_b", "admitted", "estimate_check_cap_only")]
    assert order[4:] == [
        "start run_a", "end run_a", "start run_b", "end run_b"]


def test_packed_cards_hold_the_shared_lock_against_other_woof_runs(
        tmp_path, monkeypatch):
    from woof.supervisor import GPUAlreadyLockedError, GPUFileLock

    cards = {"GPU-alpha": FakeCard(24)}
    plan, _meter, runners = _packed(
        tmp_path, monkeypatch, cards=cards, need={"run_a": 4},
        devices=(0,), table="runs_per_gpu = 2")
    original = runners.__call__
    refused = []

    def launcher(command, **kwargs):
        # A plain `woof run` on the shared root while a packed run lives.
        try:
            GPUFileLock("GPU-alpha").acquire().release()
        except GPUAlreadyLockedError:
            refused.append(True)
        return original(command, **kwargs)

    monkeypatch.setattr(multi_run.subprocess, "Popen", launcher)
    assert _main(plan) == 0
    assert refused == [True]
    card = _summary(plan)["packing"]["cards"][0]
    assert card["shared_gpu_lock"]["held_by_this_multi_run"] is True
    GPUFileLock("GPU-alpha").acquire().release()  # released at the end


def test_a_card_whose_shared_lock_is_held_elsewhere_is_shared_by_measurement(
        tmp_path, monkeypatch):
    from woof.supervisor import GPUFileLock

    cards = {"GPU-alpha": FakeCard(24, external_gib=10)}
    plan, _meter, runners = _packed(
        tmp_path, monkeypatch, cards=cards, need={"run_a": 4},
        devices=(0,), table="runs_per_gpu = 2")
    external = GPUFileLock("GPU-alpha", run_id="external-forecast").acquire()
    try:
        assert _main(plan) == 0
    finally:
        external.release()
    assert [name for name, _kwargs in runners.launched] == ["run_a"]
    card = _summary(plan)["packing"]["cards"][0]
    assert card["shared_gpu_lock"]["held_by_this_multi_run"] is False
    assert "measured co-tenant" in card["shared_gpu_lock"]["detail"]


def test_mps_dir_inside_a_plan_path_is_refused(tmp_path, monkeypatch):
    plan, _meter, runners = _packed(
        tmp_path, monkeypatch, cards={"GPU-alpha": FakeCard(24)},
        need={"run_a": 4}, devices=(0,), table="runs_per_gpu = 2")
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    with pytest.raises(ValueError, match="MPS directory .* overlaps run "
                       "'run_a' outdir"):
        _main(plan, mps=True, mps_dir=plan.parent / "output_a" / "mps")
    assert not (plan.parent / "output_a").exists()
    assert runners.launched == []


def test_load_experiment_refuses_an_authority_whose_bytes_do_not_hash(
        tmp_path):
    from woof.config_authority import ConfigAuthority
    from woof.experiment import load_experiment

    config = tmp_path / "c.toml"
    with pytest.raises(ValueError, match="does not hash"):
        load_experiment(config, authority=ConfigAuthority(
            b"[experiment]\n", config, "0" * 64, config))


def test_mps_environment_reaches_every_child_and_daemon_stops(
        tmp_path, monkeypatch):
    started = []

    class _FakeDaemon(MpsDaemon):
        def start(self):
            started.append(list(self.devices))
            self.owned = self.started = True
            return self

        def stop(self):
            started.append("stopped")
            self.owned = False

    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    monkeypatch.setattr(multi_run, "MpsDaemon", _FakeDaemon)
    cards = {"GPU-alpha": FakeCard(24)}
    plan, _meter, runners = _packed(
        tmp_path, monkeypatch, cards=cards,
        need={"run_a": 4, "run_b": 4}, devices=(0, 0),
        table="runs_per_gpu = 2\nmps = true")
    assert _main(plan, mps_dir=tmp_path / "mps",
                 mps_active_thread_percentage=40) == 0
    assert started == [["GPU-alpha"], "stopped"]
    for _name, kwargs in runners.launched:
        assert kwargs["env"]["CUDA_MPS_PIPE_DIRECTORY"] == str(
            tmp_path / "mps" / "pipe")
        assert kwargs["env"]["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "40"
    mps = _summary(plan)["packing"]["mps"]
    assert mps["started_here"] and mps["active_thread_percentage"] == 40


def test_mps_refusal_is_named(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    plan, _meter, runners = _packed(
        tmp_path, monkeypatch, cards={"GPU-alpha": FakeCard(24)},
        need={"run_a": 4}, devices=(0,), table="runs_per_gpu = 2")
    with pytest.raises(ValueError, match="multi-run --mps: .*mps-control"):
        _main(plan, mps=True)
    assert runners.launched == []


def test_interrupt_leaves_queued_runs_not_started(tmp_path, monkeypatch):
    cards = {"GPU-alpha": FakeCard(16)}
    meter = FakeMeter(cards)
    plan = load_plan(_write_plan(tmp_path, devices=(0, 0),
                                 table="runs_per_gpu = 2"))
    runs = resolve_devices(plan, _inventory())
    monkeypatch.setattr(multi_run, "price_run",
                        _priced({"run_a": 8, "run_b": 8}))
    packing, runs = plan_packing(plan, runs, lock_root=tmp_path / "locks",
                                 meter=meter)
    multi_run._prepare_directories(plan)

    class _Interrupting:
        pid = 91_000

        def poll(self):
            raise KeyboardInterrupt

    outcome = multi_run._execute_packed_group(
        runs, phase="run", command_for=multi_run._run_command,
        log_for=lambda run: run.spec.log, packing=packing,
        popen=lambda *_a, **_k: _Interrupting(), poll_seconds=0.001)
    assert outcome.interrupted
    assert [record.status for record in outcome.records] == [
        "running_unobserved", "not_started"]
    assert outcome.records[0].pid == 91_000


def test_packed_child_environment_keeps_isolation(tmp_path, monkeypatch):
    plan = load_plan(_write_plan(tmp_path, devices=(0, 0),
                                 table="runs_per_gpu = 2"))
    runs = resolve_devices(plan, _inventory())
    monkeypatch.setattr(multi_run, "price_run",
                        _priced({"run_a": 1, "run_b": 1}))
    _packing, runs = plan_packing(
        plan, runs, lock_root=tmp_path / "locks",
        meter=FakeMeter({"GPU-alpha": FakeCard(24)}))
    first, second = (child_environment(run, {"TMPDIR": "/elsewhere"})
                     for run in runs)
    assert first["TMPDIR"] == str(runs[0].spec.scratch_dir)
    assert first["CUPY_CACHE_DIR"] != second["CUPY_CACHE_DIR"]
    assert first[GPU_LOCK_ROOT_ENV] != second[GPU_LOCK_ROOT_ENV]
    assert first["CUDA_VISIBLE_DEVICES"] == second["CUDA_VISIBLE_DEVICES"]


def test_price_run_prices_the_captured_bytes_through_the_run_door(
        tmp_path, monkeypatch):
    plan = load_plan(_write_plan(tmp_path, devices=(0,)))
    spec = plan.runs[0]
    seen = {}

    def fake_load(path, *, authority=None):
        seen["path"] = path
        seen["payload"] = authority.payload
        return "experiment"

    import woof.experiment as experiment

    monkeypatch.setattr(experiment, "load_experiment", fake_load)
    monkeypatch.setattr(multi_run, "priced_reservation_bytes",
                        lambda value: 5 * GIB if value == "experiment"
                        else None)
    # The source is changed after capture; pricing must not see it.
    spec.config.write_text("[experiment]\nname = 'changed'\n")
    priced = multi_run.price_run(spec)
    assert priced.need_bytes == 5 * GIB
    assert priced.basis == multi_run.CONFIG_PRICING_BASIS
    assert seen["payload"] == b"[experiment]\nname = 'fixture_a'\n"
    monkeypatch.setattr(multi_run, "priced_reservation_bytes",
                        lambda value: None)
    assert multi_run.price_run(spec).need_bytes is None
    assert multi_run.price_run(spec).basis.startswith("unpriced")


def test_energy_packing_still_exports_its_names():
    import woof.energy.worker as wk
    from woof.energy.packing import parse_slots_per_gpu
    from woof.gpu_packing import MPSUnavailable

    assert parse_slots_per_gpu("AUTO") == "auto"
    with pytest.raises(ValueError, match="--slots-per-gpu takes"):
        parse_slots_per_gpu(0)
    assert issubclass(wk.MpsDaemon, MpsDaemon)
    assert issubclass(wk.MPSUnavailable, wk.WorkerError)
    assert issubclass(wk.MPSUnavailable, MPSUnavailable)
    assert wk.MPS_CONTROL == "nvidia-cuda-mps-control"
