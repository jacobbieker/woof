"""``woof pack-plan``: how to put a set of runs on a runner's GPUs.

Given one or more configurations (or N copies of one) and a runner (this
box, a ``--card`` from the table or a ``[runner]`` TOML, as
``woof estimate`` resolves it), this recommends how many runs share each
card, whether CUDA MPS is on, the order and the card each run goes to,
and says what every option is predicted to take against running the
runs one at a time.

Every option is admitted by the code that admits it at run time.
:func:`simulate` starts a run on a card only when
:func:`woof.multi_run.packed_admission` admits it -- the
:data:`woof.multi_run.PACKED_ADMISSION_RULE`, priced by
:func:`woof.supervisor.priced_reservation_bytes` (the number multi-run
prices a config-form run with), the
:data:`woof.gpu_packing.DEFAULT_MEM_MARGIN_GIB` margin, at most
:data:`woof.gpu_packing.AUTO_MAX_SLOTS_PER_GPU` per card and at most one
per CPU core of the card's share of the host.  A run admission refuses
makes its option unavailable; the plan never recommends one.

How much a run slows when others share its card is
:func:`woof.runtime_estimate.sharing_bracket`: MEASURED rows where a
row exists (``SHARING_ROWS``, each naming its card, grid, copies, MPS
and receipt), interpolated between measured counts, and past them an
UNMEASURED bracket from the measured trend to the worst measured
slowdown per copy.  On the RTX A6000 a 162,000-column run packed two to
four to a card ran 2.24-4.50x slower each, slower than time-slicing: a
card a large domain already fills gains nothing from packing but the
overlap of the runs' start-up.  Each run's alone time is
``woof estimate``'s.  A run streamed through host memory is bus-bound:
it is priced at pure time-slicing (``n`` runs on a card each take ``n``
times as long), so sharing gains it nothing.  multi-run admission has no
streamed-run rule, so packing one is not refused; it is just never
predicted to win by more than start-up overlap.

The simulation is a fluid one: a run's preparation proceeds at its own
pace (CPU), and its forecast at ``1 / slowdown(n)`` where ``n`` is the
number of runs then on its card, re-evaluated whenever a run starts or
finishes.  Runs start in plan order per card (multi-run's head-of-line
rule: a run that does not fit waits, and runs behind it on that card
wait with it).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from woof.runner_cards import CUDA, GIB, Runner, RunnerSpecError

SCHEMA = "woof.pack-plan.v1"
OBJECTIVES = ("makespan", "throughput")
ORDERS = ("as-given", "longest-first")
#: Two options whose predicted makespans (or throughputs) differ by less
#: than this are not separated by the prediction: the session-to-session
#: variance measured on one box is up to 1.30x (SESSION_VARIANCE), so 5 %
#: is well inside the noise.
SEPARATION = 0.05


# --------------------------------------------------------------------------
# the runs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Job:
    """One run: what it costs alone and what admission prices it at."""

    name: str
    config: str
    #: (low, high) seconds of preparation and compile: host work that
    #: packing does not slow.
    fixed_s: tuple[float, float]
    #: (low, high) seconds of forecast on the card, alone.
    gpu_s: tuple[float, float]
    peak_bytes: int | None
    columns: int
    road: str
    provenance: str

    @property
    def alone_s(self) -> tuple[float, float]:
        return (self.fixed_s[0] + self.gpu_s[0],
                self.fixed_s[1] + self.gpu_s[1])

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "config": self.config,
                "alone_s": [round(v, 1) for v in self.alone_s],
                "fixed_s": [round(v, 1) for v in self.fixed_s],
                "gpu_s": [round(v, 1) for v in self.gpu_s],
                "priced_peak_bytes": self.peak_bytes,
                "priced_peak_gib": (None if self.peak_bytes is None else
                                    round(self.peak_bytes / GIB, 2)),
                "columns": self.columns, "road": self.road,
                "rate_provenance": self.provenance}


#: Phases of ``woof estimate`` that run on the card and slow when shared.
GPU_PHASES = ("model steps", "radiation steps", "output", "restart writes",
              "data assimilation")


def job_from_estimate(name: str, config: str, est, peak_bytes: int | None
                      ) -> Job:
    fixed = [0.0, 0.0]
    gpu = [0.0, 0.0]
    for phase in est.phases:
        target = gpu if phase.name in GPU_PHASES else fixed
        target[0] += phase.low_s
        target[1] += phase.high_s
    return Job(name=name, config=config, fixed_s=(fixed[0], fixed[1]),
               gpu_s=(gpu[0], gpu[1]), peak_bytes=peak_bytes,
               columns=est.domains[0].columns, road=est.road,
               provenance=est.rate.provenance)


def jobs_for(configs: Sequence[str | Path], runner: Runner, *,
             copies: int = 1, prepared: bool = False,
             detected: bool = False) -> list[Job]:
    """Each config (``copies`` of each) as a :class:`Job` on ``runner``."""
    from woof.experiment import load_experiment
    from woof.runtime_estimate import estimate_config

    jobs: list[Job] = []
    bases = run_names(configs)
    jobs_seen: list[str] = []
    for config in configs:
        est = estimate_config(config, runner, detected=detected,
                              prepared=prepared)
        try:
            from woof.supervisor import priced_reservation_bytes

            peak = priced_reservation_bytes(load_experiment(config))
        except Exception:  # noqa: BLE001 - unpriced is an answer
            peak = None
        stem = bases[len(jobs_seen)]
        jobs_seen.append(stem)
        for i in range(max(1, int(copies))):
            name = stem if copies <= 1 else f"{stem}-{i + 1}"
            jobs.append(job_from_estimate(name, str(config), est, peak))
    return jobs


def run_names(configs: Sequence[str | Path]) -> list[str]:
    """One run name per config, unique and deterministic: the file stem;
    where stems collide, the parent directory's name before it
    (``a-member``); where those still collide (the same file twice), an
    occurrence index (``a-member-1``, ``a-member-2``)."""
    paths = [Path(c) for c in configs]
    stems = [p.stem for p in paths]
    names = [n if stems.count(n) == 1 else f"{p.resolve().parent.name}-{n}"
             for p, n in zip(paths, stems)]
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if names.count(n) > 1:
            seen[n] = seen.get(n, 0) + 1
            n = f"{n}-{seen[n]}"
        out.append(n)
    return out


# --------------------------------------------------------------------------
# admission (multi-run's own function)
# --------------------------------------------------------------------------
def foreign_used_bytes(runner: Runner) -> int:
    """Card memory held by processes the plan does not own: the measured
    used memory on a detected box, else none."""
    if runner.free_vram_bytes is None or runner.vram_bytes is None:
        return 0
    return max(0, int(runner.vram_bytes) - int(runner.free_vram_bytes))


def admit(job: Job, live: Sequence[Job], *, runner: Runner, cap: int,
          margin_bytes: int) -> dict[str, Any]:
    """:func:`woof.multi_run.packed_admission` for ``job`` beside ``live``.

    The siblings are taken at their full priced peaks (NVML attributes
    nothing to them yet): the worst moment admission sees."""
    from woof.multi_run import packed_admission

    total = int(runner.vram_bytes or 0)
    foreign = foreign_used_bytes(runner)
    return packed_admission(
        need_bytes=job.peak_bytes, margin_bytes=margin_bytes,
        card_total_bytes=total, cap=cap,
        live=[{"name": j.name, "priced_bytes": j.peak_bytes} for j in live],
        observe=lambda: {},
        measure=lambda: {"device_wide_used_bytes": foreign,
                         "device_free_bytes": max(0, total - foreign),
                         "basis": "pack-plan: the runner's measured free "
                                  "memory at plan time" if foreign else
                                  "pack-plan: the card's total (no "
                                  "process was measured on it)"})


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
@dataclass
class Schedule:
    slots: int
    mps: bool
    order: str
    end: str
    makespan_s: float
    finish_s: dict[str, float]
    start_s: dict[str, float]
    device: dict[str, int]
    max_on_card: int
    refused: list[dict[str, Any]] = field(default_factory=list)
    unmeasured: bool = False

    @property
    def admitted(self) -> bool:
        return not self.refused


def assign_devices(jobs: Sequence[Job], gpus: int) -> dict[str, int]:
    """Longest alone time first onto the least-loaded card."""
    load = [0.0] * max(1, gpus)
    out: dict[str, int] = {}
    for job in sorted(jobs, key=lambda j: -sum(j.alone_s)):
        g = min(range(len(load)), key=lambda i: (load[i], i))
        out[job.name] = g
        load[g] += sum(job.alone_s) / 2.0
    return out


def order_jobs(jobs: Sequence[Job], order: str) -> list[Job]:
    if order == "longest-first":
        return sorted(jobs, key=lambda j: -sum(j.alone_s))
    if order == "as-given":
        return list(jobs)
    raise ValueError(f"order is one of {ORDERS}, not {order!r}")


def _slowdown(n: int, job: Job, mps: bool, end: str,
              card: str | None = None) -> tuple[float, bool]:
    """(per-run slowdown with ``n`` on the card, measured?)."""
    from woof.runtime_estimate import sharing_bracket

    if n <= 1:
        return 1.0, True
    if job.road == "streamed":
        # Bus-bound: a second streamed forecast on a 4090 took each to
        # 22.6/11.66 = 1.94x (woof/core/pace.py) -- no throughput gained.
        return float(n), False
    s = sharing_bracket(n, columns=job.columns, mps=mps, card=card)
    lo, hi = s["per_run_slowdown"]
    return (lo if end == "low" else hi), bool(s["measured"])


def simulate(jobs: Sequence[Job], *, runner: Runner, slots: int, mps: bool,
             order: str = "longest-first", end: str = "high",
             margin_gib: float | None = None) -> Schedule:
    """Run the plan in time; ``end`` picks each bracket's ``low`` or
    ``high`` edge (alone times and slowdowns alike)."""
    from woof.gpu_packing import DEFAULT_MEM_MARGIN_GIB

    margin = int((DEFAULT_MEM_MARGIN_GIB if margin_gib is None
                  else margin_gib) * GIB)
    edge = 0 if end == "low" else 1
    ordered = order_jobs(jobs, order)
    device = assign_devices(ordered, runner.gpus)
    queues: dict[int, list[Job]] = {}
    for job in ordered:
        queues.setdefault(device[job.name], []).append(job)
    # The lone fit test first: a run that cannot be admitted onto an
    # empty card is refused by multi-run, and so is the option.
    refused = []
    for job in ordered:
        d = admit(job, [], runner=runner, cap=slots, margin_bytes=margin)
        # On an empty card, waiting has nothing to wait for: multi-run
        # refuses it (a sibling-free "card_total" wait never ends).
        if d["verdict"] not in ("admitted", "admitted-unpriced-alone"):
            refused.append({"run": job.name, "rule": d["rule"],
                            "reason": d["reason"]})
    if refused:
        return Schedule(slots, mps, order, end, math.inf, {}, {}, device, 0,
                        refused)
    t = 0.0
    fixed_left: dict[str, float] = {}
    gpu_left: dict[str, float] = {}
    live: dict[int, list[Job]] = {g: [] for g in queues}
    start: dict[str, float] = {}
    finish: dict[str, float] = {}
    max_on = 0
    unmeasured = False
    guard = 0
    while any(queues.values()) or any(live.values()):
        guard += 1
        if guard > 100_000:
            raise RuntimeError("pack-plan simulation did not converge")
        for g, queue in queues.items():
            while queue:
                job = queue[0]
                d = admit(job, live[g], runner=runner, cap=slots,
                          margin_bytes=margin)
                if d["verdict"] not in ("admitted", "admitted-unpriced-alone"):
                    break
                queue.pop(0)
                live[g].append(job)
                start[job.name] = t
                fixed_left[job.name] = job.fixed_s[edge]
                gpu_left[job.name] = job.gpu_s[edge]
            max_on = max(max_on, len(live[g]))
        # Rates now; the next event is the first phase to end.
        rates: dict[str, tuple[str, float]] = {}
        for g, running in live.items():
            on_card = sum(1 for j in running if fixed_left[j.name] <= 0)
            for job in running:
                if fixed_left[job.name] > 0:
                    rates[job.name] = ("fixed", 1.0)
                else:
                    s, measured = _slowdown(max(1, on_card), job, mps, end,
                                            runner.card_key)
                    unmeasured |= not measured
                    rates[job.name] = ("gpu", 1.0 / s)
        if not rates:
            raise RuntimeError("pack-plan: nothing can start; a run waits "
                               "on a card no sibling holds")
        dt = min((fixed_left[n] if kind == "fixed" else gpu_left[n]) / r
                 for n, (kind, r) in rates.items())
        dt = max(dt, 0.0)
        t += dt
        for name, (kind, r) in rates.items():
            if kind == "fixed":
                fixed_left[name] = max(0.0, fixed_left[name] - dt * r)
                if fixed_left[name] <= 1e-9:
                    fixed_left[name] = 0.0
            else:
                gpu_left[name] = max(0.0, gpu_left[name] - dt * r)
        for g in live:
            done = [j for j in live[g] if fixed_left[j.name] <= 0
                    and gpu_left[j.name] <= 1e-9]
            for job in done:
                live[g].remove(job)
                finish[job.name] = t
    return Schedule(slots, mps, order, end, t, finish, start, device, max_on,
                    [], unmeasured)


# --------------------------------------------------------------------------
# options
# --------------------------------------------------------------------------
def slot_cap(runner: Runner, max_slots: int | None = None) -> int:
    """At most :data:`AUTO_MAX_SLOTS_PER_GPU` per card and one per CPU
    core of the card's share of the host (the energy worker's ``auto``)."""
    from woof.gpu_packing import AUTO_MAX_SLOTS_PER_GPU

    cap = AUTO_MAX_SLOTS_PER_GPU if max_slots is None else int(max_slots)
    if runner.cpu_cores:
        cap = min(cap, max(1, int(runner.cpu_cores) // max(1, runner.gpus)))
    return max(1, cap)


def _finite(v: float) -> float | None:
    """Rounded seconds, or ``None`` (JSON null) for an infinite makespan
    (a refused option): strict JSON has no Infinity."""
    return round(v, 1) if math.isfinite(v) else None


@dataclass
class Option:
    slots: int
    mps: bool
    order: str
    low: Schedule
    high: Schedule

    @property
    def admitted(self) -> bool:
        return self.low.admitted and self.high.admitted

    @property
    def makespan(self) -> tuple[float, float]:
        return self.low.makespan_s, self.high.makespan_s

    @property
    def central_s(self) -> float:
        lo, hi = self.makespan
        return math.sqrt(lo * hi) if lo > 0 and hi > 0 else hi

    def latest_finish_s(self) -> float:
        return max(self.high.finish_s.values(), default=math.inf)

    def throughput_per_hour(self, jobs: int) -> tuple[float, float]:
        lo, hi = self.makespan
        return (jobs * 3600.0 / hi if hi else 0.0,
                jobs * 3600.0 / lo if lo else 0.0)

    def as_dict(self, jobs: int) -> dict[str, Any]:
        tp = self.throughput_per_hour(jobs)
        return {"runs_per_gpu": self.slots, "mps": self.mps,
                "order": self.order, "admitted": self.admitted,
                "refused": self.low.refused or self.high.refused,
                "makespan_s": [_finite(v) for v in self.makespan],
                "central_makespan_s": _finite(self.central_s),
                "runs_per_hour": [round(v, 2) for v in tp],
                "most_on_one_card": max(self.low.max_on_card,
                                        self.high.max_on_card),
                "contention_measured": not (self.low.unmeasured
                                            or self.high.unmeasured),
                "finish_s_high": {k: round(v, 1)
                                  for k, v in self.high.finish_s.items()},
                "device": dict(self.high.device)}


def options(jobs: Sequence[Job], runner: Runner, *,
            max_slots: int | None = None, allow_mps: bool = True,
            margin_gib: float | None = None) -> list[Option]:
    out = []
    mps_choices = [False] + ([True] if allow_mps and runner.backend == CUDA
                             else [])
    for slots in range(1, slot_cap(runner, max_slots) + 1):
        for mps in mps_choices:
            if slots == 1 and mps:
                continue
            for order in ORDERS:
                lo = simulate(jobs, runner=runner, slots=slots, mps=mps,
                              order=order, end="low", margin_gib=margin_gib)
                hi = simulate(jobs, runner=runner, slots=slots, mps=mps,
                              order=order, end="high", margin_gib=margin_gib)
                if any(o.slots == slots and o.mps == mps
                       and o.low.start_s == lo.start_s
                       and o.high.start_s == hi.start_s for o in out):
                    continue  # this order schedules as an earlier one did
                out.append(Option(slots, mps, order, lo, hi))
    return out


def serial_s(jobs: Sequence[Job]) -> tuple[float, float]:
    """One run at a time on one card."""
    return (sum(j.alone_s[0] for j in jobs), sum(j.alone_s[1] for j in jobs))


def choose(opts: Sequence[Option], *, objective: str = "makespan",
           deadline_s: float | None = None) -> tuple[Option | None, str]:
    admitted = [o for o in opts if o.admitted]
    if not admitted:
        return None, "admission refuses every option"
    pool = admitted
    note = ""
    if deadline_s is not None:
        feasible = [o for o in admitted if o.latest_finish_s() <= deadline_s]
        if feasible:
            pool = feasible
        else:
            note = ("no option finishes every run by the deadline at the "
                    "high end; the fastest is shown")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective is one of {OBJECTIVES}")

    def score(o: Option) -> float:  # lower is better
        if objective == "throughput":
            return 1.0 / max(o.throughput_per_hour(len(o.high.finish_s))[0],
                             1e-12)
        return o.central_s

    # Options within SEPARATION of the best are ones the prediction does
    # not separate: among them a measured contention model, then fewer
    # runs per card, then no MPS, then the given order win.
    top = min(score(o) for o in pool)
    close = [o for o in pool if score(o) <= top * (1.0 + SEPARATION)]
    best = min(close, key=lambda o: (
        o.low.unmeasured or o.high.unmeasured, o.slots, o.mps,
        ORDERS.index(o.order), score(o)))
    return best, note


# --------------------------------------------------------------------------
# what to type
# --------------------------------------------------------------------------
def multi_run_plan_text(jobs: Sequence[Job], option: Option) -> str:
    """A ``gpuwm.multi-run-plan/v1`` plan in the recommended order."""
    from woof.multi_run import PLAN_SCHEMA

    lines = [f'schema = "{PLAN_SCHEMA}"', 'summary = "pack-summary.json"',
             'preflight = "estimate"', "", "[devices]",
             f"runs_per_gpu = {option.slots}",
             f"mps = {'true' if option.mps else 'false'}", ""]
    for job in order_jobs(jobs, option.order):
        lines += ["[[run]]", f'name = "{job.name}"',
                  f"device = {option.high.device[job.name]}",
                  f'config = "{job.config}"',
                  f'outdir = "runs/{job.name}"',
                  f'scratch = "scratch/{job.name}"',
                  f'cache = "cache/{job.name}"', ""]
    return "\n".join(lines)


#: Why one run per card with more runs than cards is not a multi-run plan.
SERIAL_NOTE = (
    "woof multi-run gives each run its own card at --runs-per-gpu 1 and "
    "refuses two runs on one card (device_discovery), so it does not queue "
    "more runs than cards; run them one after another")


def multi_run_queues(option: Option, runs: int, gpus: int) -> bool:
    """Whether ``woof multi-run`` runs this option as one plan: a packed
    plan (``runs_per_gpu`` above 1) queues on admission, an unpacked one
    needs a card per run."""
    return option.slots > 1 or runs <= max(1, gpus)


#: What the command names when no plan file was written.
NO_PLAN = "<PLAN.toml: not written; re-run with --emit-plan PLAN.toml>"


def invocations(option: Option, plan_path: str | None = None, *,
                jobs: Sequence[Job] = (), gpus: int = 1) -> list[str]:
    """What to type.  ``plan_path`` is the plan file written, or ``None``
    when none was (the command then names an obvious placeholder)."""
    import shlex

    mps = " --mps" if option.mps else ""
    worker = (f"woof energy worker ... --slots-per-gpu {option.slots}{mps}  "
              "(a hindcast queue of runs like these)")
    if jobs and not multi_run_queues(option, len(jobs), gpus):
        # One queue per card, as simulate() runs them: the cards in
        # parallel, each card's runs one after another.
        per_card: dict[int, list[Job]] = {}
        for job in order_jobs(jobs, option.order):
            per_card.setdefault(option.high.device.get(job.name, 0),
                                []).append(job)
        lines = []
        for card in sorted(per_card):
            chain = " && ".join(f"woof go {shlex.quote(j.config)}"
                                for j in per_card[card])
            lines.append(f"CUDA_VISIBLE_DEVICES={card} sh -c "
                         f"{shlex.quote(chain)} &")
        return lines + [f"wait  ({SERIAL_NOTE}; one queue per card, the "
                        "cards in parallel)", worker]
    plan = shlex.quote(plan_path) if plan_path else NO_PLAN
    return [f"woof multi-run {plan} --runs-per-gpu {option.slots}{mps}",
            worker]


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------
def render_text(jobs: Sequence[Job], runner: Runner, opts: Sequence[Option],
                best: Option | None, note: str, *, objective: str,
                deadline_s: float | None,
                plan_path: str | None = None) -> str:
    from woof.runtime_estimate import fmt_duration

    def span(pair):
        return f"{fmt_duration(pair[0])}-{fmt_duration(pair[1])}"

    lines = [f"woof pack-plan: {len(jobs)} run(s) on {runner.gpus} x "
             f"{runner.name}"
             + ("" if runner.vram_bytes is None else
                f" ({runner.vram_bytes / GIB:.0f} GiB"
                + ("" if not foreign_used_bytes(runner) else
                   f", {foreign_used_bytes(runner) / GIB:.1f} GiB held by "
                   "other processes") + ")"),
             f"  objective: {objective}"
             + ("" if deadline_s is None else
                f", every run done within {fmt_duration(deadline_s)}")]
    lines.append("  runs (alone):")
    for job in jobs:
        peak = ("unpriced (runs alone)" if job.peak_bytes is None
                else f"{job.peak_bytes / GIB:.1f} GiB priced peak")
        lines.append(f"    {job.name}: {span(job.alone_s)}, {peak}, "
                     f"{job.road} road, {job.columns:,} columns "
                     f"[{job.provenance}]")
    lines.append(f"  one at a time on one card: {span(serial_s(jobs))}")
    lines.append(f"  {'per card':>8}  {'MPS':>3}  {'order':<13} "
                 f"{'makespan':>17}  contention")
    seen = set()
    for o in sorted(opts, key=lambda o: (o.slots, o.mps, o.order)):
        if (o.slots, o.mps) in seen and o.order == "as-given" \
                and not o.admitted:
            continue
        seen.add((o.slots, o.mps))
        if not o.admitted:
            refused = (o.low.refused or o.high.refused)[0]
            lines.append(f"  {o.slots:>8}  {'on' if o.mps else 'off':>3}  "
                         f"{o.order:<13} {'REFUSED':>17}  {refused['run']}: "
                         f"{refused['rule']}")
            continue
        lines.append(f"  {o.slots:>8}  {'on' if o.mps else 'off':>3}  "
                     f"{o.order:<13} {span(o.makespan):>17}  "
                     + ("measured" if not (o.low.unmeasured
                                           or o.high.unmeasured)
                        else "UNMEASURED beyond the rows")
                     + ("  <- recommended" if o is best else ""))
    if best is None:
        lines.append(f"  no recommendation: {note}")
        return "\n".join(lines)
    gain = serial_s(jobs)[1] / best.makespan[1] if best.makespan[1] else 0.0
    lines.append(
        f"  recommended: {best.slots} run(s) per card, MPS "
        f"{'on' if best.mps else 'off'}, {best.order}; makespan "
        f"{span(best.makespan)} against {span(serial_s(jobs))} one at a "
        f"time ({gain:.2f}x at the high ends)")
    if note:
        lines.append(f"  note: {note}")
    for job in jobs:
        if job.road == "streamed":
            lines.append(f"  note: {job.name} streams through host memory "
                         "on this card; a streamed run is bus-bound and is "
                         "planned with no gain from sharing")
        if job.peak_bytes is None:
            lines.append(f"  note: {job.name} could not be priced; "
                         "admission runs it alone on its card")
    lines.append("  run it with:")
    for line in invocations(best, plan_path, jobs=jobs,
                           gpus=runner.gpus):
        lines.append(f"    {line}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the door
# --------------------------------------------------------------------------
def main(args) -> int:
    from woof.runtime_estimate import resolve_runner

    detected = not (args.card or args.runner)
    try:
        runner = resolve_runner(args)
    except RunnerSpecError as error:
        print(f"woof pack-plan: {error}", file=sys.stderr)
        return 2
    if args.gpus:
        from dataclasses import replace

        runner = replace(runner, gpus=int(args.gpus))
    try:
        jobs = jobs_for(args.configs, runner, copies=args.copies,
                        prepared=args.prepared, detected=detected)
    except (ValueError, OSError) as error:
        print(f"woof pack-plan: {error}", file=sys.stderr)
        return 2
    deadline = None if args.deadline_hours is None else \
        float(args.deadline_hours) * 3600.0
    opts = options(jobs, runner, max_slots=args.max_slots,
                   allow_mps=not args.no_mps,
                   margin_gib=args.mem_margin_gib)
    best, note = choose(opts, objective=args.objective, deadline_s=deadline)
    plan_written = (best is not None and bool(args.emit_plan)
                    and multi_run_queues(best, len(jobs), runner.gpus))
    if plan_written:
        Path(args.emit_plan).write_text(multi_run_plan_text(jobs, best) + "\n",
                                        encoding="utf-8")
    emitted = str(args.emit_plan) if plan_written else None
    if args.json:
        from woof.multi_run import PACKED_ADMISSION_RULE

        doc = {
            "schema": SCHEMA, "runner": runner.as_dict(),
            "objective": args.objective, "deadline_s": deadline,
            "runs": [j.as_dict() for j in jobs],
            "one_at_a_time_s": [round(v, 1) for v in serial_s(jobs)],
            "options": [o.as_dict(len(jobs)) for o in opts],
            "recommended": None if best is None else best.as_dict(len(jobs)),
            "note": note or None,
            "invocations": [] if best is None else invocations(
                best, emitted, jobs=jobs, gpus=runner.gpus),
            "plan_written": args.emit_plan if plan_written else None,
            "admission_rule": PACKED_ADMISSION_RULE,
        }
        print(json.dumps(doc, indent=2, default=str))
    else:
        print(render_text(jobs, runner, opts, best, note,
                          objective=args.objective, deadline_s=deadline,
                          plan_path=emitted))
        if plan_written:
            print(f"  wrote {args.emit_plan}")
        elif best is not None and args.emit_plan:
            print(f"  not written: {args.emit_plan}: {SERIAL_NOTE}")
    return 0 if best is not None else 1


def register_cli(subparsers):
    p = subparsers.add_parser(
        "pack-plan",
        help="how to pack a set of runs onto the runner's GPUs: runs per "
             "card, MPS, order and card per run, with the predicted "
             "makespan of every option against one at a time; never an "
             "option multi-run admission would refuse",
        description=__doc__.split("\n\n")[0])
    p.add_argument("configs", nargs="+", type=Path, metavar="CONFIG",
                   help="experiment TOML(s) to run")
    p.add_argument("--copies", type=int, default=1, metavar="N",
                   help="N runs of each CONFIG (ensemble members, a "
                        "hindcast batch)")
    p.add_argument("--card", "--gpu", dest="card", metavar="NAME",
                   help="plan for this card from woof estimate's table "
                        "instead of this box's")
    p.add_argument("--runner", type=Path, metavar="TOML",
                   help="a [runner] table (woof estimate --runner)")
    p.add_argument("--gpus", type=int, default=None, metavar="N",
                   help="cards on the runner (default: detected, else 1)")
    p.add_argument("--max-slots", type=int, default=None, metavar="N",
                   help="consider at most N runs per card (default: "
                        "multi-run's auto cap and the host's cores)")
    p.add_argument("--no-mps", action="store_true",
                   help="do not consider CUDA MPS")
    p.add_argument("--mem-margin-gib", type=float, default=None,
                   metavar="GiB",
                   help="card memory kept free beside the priced peaks "
                        "(multi-run --mem-margin-gib; default 1)")
    p.add_argument("--objective", choices=OBJECTIVES, default="makespan",
                   help="makespan: finish the set soonest (default); "
                        "throughput: most runs per hour")
    p.add_argument("--deadline-hours", type=float, default=None,
                   metavar="H", help="every run must finish within H hours "
                                     "of the start (judged at the high end)")
    p.add_argument("--prepared", action="store_true",
                   help="the runs' inputs are already prepared")
    p.add_argument("--emit-plan", type=Path, default=None, metavar="PLAN",
                   help="write the recommended multi-run plan here")
    p.add_argument("--json", action="store_true",
                   help=f"print the {SCHEMA} document")
    p.set_defaults(func=main, mps=False, slots=None, list_cards=False)
    return p


__all__ = [
    "GPU_PHASES", "Job", "OBJECTIVES", "ORDERS",
    "Option", "SCHEMA", "SEPARATION",
    "Schedule", "admit", "assign_devices", "choose", "foreign_used_bytes",
    "invocations", "job_from_estimate", "jobs_for", "multi_run_plan_text",
    "SERIAL_NOTE", "multi_run_queues", "options", "order_jobs",
    "register_cli", "render_text", "serial_s",
    "simulate", "slot_cap",
]
