"""``woof estimate``: how long a run will take on a given runner.

THE QUESTION.  :mod:`woof.core.pace` answers "how fast, on the card in
front of you, at full physics and nz=49", inside ``woof check``.  A user
choosing a box -- an A6000 they own, an H100 they could rent, a Mac --
asks the wider one: given THIS runner's GPU, memory, host and
concurrency, when will this configuration finish, and what is the time
spent on?  This module answers it, phase by phase, as a low-high
bracket, and says for every figure whether it was MEASURED on that card,
SCALED from another card by the model below, CALIBRATED on this box by
``--calibrate``, or SUBSTITUTED from the slowest row on record.

NOTHING HERE IS INVENTED.  Every timing is a row of :data:`ANCHORS` (or
of :mod:`woof.core.pace`, which this module reuses) that names its
card, grid, timestep, contention and source; the only non-measured
figures are vendor nameplates (:mod:`woof.runner_cards`), and the
scaling model that carries a measured rate to a card nobody timed.

THE SCALING MODEL (cards nobody timed)
--------------------------------------
Time per column-step scales as ``(B_anchor / B_card) ** e`` in memory
bandwidth ``B``, with ``e`` bracketed by :data:`BANDWIDTH_EXPONENT`
(0.78-1.0):

* the dynamics and most physics kernels are stencil sweeps that stream
  the state through DRAM once or a few times per step -- bandwidth
  bound, not FLOP bound.  The fleet's measurement: the 1.5.0 reference
  envelope (``docs/manual/08-operational-envelope.md``) ran the same
  3 km HRRR case class at 7.17e-7 s per column-step on an RTX 5090
  (1,792 GB/s) and 1.231e-6 on an RTX 5070 Ti (896 GB/s) -- a 1.72x
  time ratio for a 2.00x bandwidth ratio, ``e = ln 1.72 / ln 2 = 0.78``.
  The FP32 ratio of that pair (104.8 vs 43.9 TFLOPS, 2.39x) does NOT
  explain it, which is the case for bandwidth over FLOPs;
* ``e = 1.0`` (pure bandwidth) is the other end, and is what
  :mod:`woof.energy.cost` and :mod:`woof.track_estimate` already price
  datacentre parts with;
* NOT modelled: Windows/WDDM (the 3080-vs-5070 Ti speedrun gap, 1.75x on
  a 1.18x bandwidth ratio, is the platform -- ``CARD_SPREAD`` in
  :mod:`woof.core.pace`), co-tenants, and PCIe for the resident road.

Apple Silicon: the M3 Max is the only chip timed, always beside an A6000
on the same work.  Its time ratio to the A6000 spans
:data:`METAL_TO_A6000` (1.09x on an 8,000-column dycore step to 4.13x
on a 50 m building domain; the A6000 was shared in most pairs, which
understates the ratio).  A CUDA rate reaches a Metal runner as: scaled to
the A6000 by bandwidth, times that ratio, then scaled from the M3 Max
(400 GB/s) to the target chip by bandwidth.

THE CONFIG MODEL
----------------
Per domain: ``columns x nz/49 x rate`` per step (linear in nz, MEASURED:
:data:`woof.core.pace.REFERENCE_NZ`), ``ceil(window / dt)`` steps over the
domain's own window (a nest starting late runs fewer), and a radiation
step costing ``R`` ordinary steps every ``radt / dt`` steps, with ``R``
bracketed by :data:`RADIATION_STEP_RATIO`.  Every anchor is normalised
to an ORDINARY step at its own cadence under each ``R`` and re-priced at
the configuration's cadence, so an anchor reproduces itself exactly at
its own ``dt`` and ``radt``.  Cumulus cadence (``cudt``) is inside the
anchors and not separately priced: no receipt isolates its cost.  The
physics rung (:func:`tilestream.autoplan.rung_of`) picks the anchors;
MYNN + Noah-MP carries pace's measured 1.86x premium; a rung with no
anchor on its route takes the slowest rung that has one, SUBSTITUTED.
An adaptive clock (``use_adaptive_time_step``) runs fewer, longer steps:
the measured wall ratio is 0.47-0.60 of the fixed clock on the A6000
(``docs/ADAPTIVE-TIMESTEP.md``), so the bracket's low end is scaled by
0.47 and the high end stays the fixed-clock price.

The streamed road (a root the card cannot hold) is priced by
:func:`woof.core.pace.estimate_pace` -- its measured streamed rows, tile
overhead and byte-model bus floor at the runner's pinned bandwidth -- and
is not card-scaled, because a streamed step is bound by the bus and the
tile count, not by DRAM.

PHASES.  ``prepare`` (fetch-to-first-step, the measured fit
``PREP_INTERCEPT_S + PREP_S_PER_MCELL`` of :mod:`woof.energy.cost` plus
31 s the first time on a box), ``kernel compile`` (cold cache only:
22.6-53.8 s measured), ``model steps``, ``radiation``, ``output``
(history frames: 81-86 bytes per cell measured, at 0 s exposed for the
asynchronous writer up to the measured synchronous 65 MB/s), ``restart``
writes, and ``data assimilation`` when ``--da`` names one.

Use ``--calibrate N`` on the runner itself for the honest answer on a
card nobody has timed: it runs N steps of THIS configuration and quotes
measured against predicted (:mod:`woof.runtime_calibration`).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from woof.runner_cards import (CUDA, METAL, RUNNER_CARDS, Runner,
                               RunnerSpecError, card_for, detect_local_runner,
                               load_runner_toml, runner_from_card)

SCHEMA = "woof.runtime-estimate.v1"

WRF = "wrf"
HEX = "hex"
ROUTES = (WRF, HEX)

MEASURED = "measured"
SCALED = "scaled"
CALIBRATED = "calibrated"
SUBSTITUTED = "substituted"

#: See the module docstring: 0.78 from the 1.5.0 envelope's 5090 / 5070 Ti
#: pair, 1.0 the pure-bandwidth bound.
BANDWIDTH_EXPONENT = (0.78, 1.0)

#: Radiation step cost in ordinary steps, (low, high), MEASURED on an RTX
#: A6000: 3.40x for a forced RTE-RRTMGP step on the 325x325x256 LES leaf
#: (3.18 s against 0.935 s, card idle; docs/energy-hindcast/cost.md), and
#: 9.99x for the HRRR suite at 200x200x49 (2405.4 ms against 240.7 ms,
#: card shared; docs/dev/perf-launch-overhead-20261015.md, which also
#: gives 3.66x for the windfarm suite at the same size).  A 44.7x ratio
#: at 128x128x49 (tilestream/RRTMGP-VRAM.md) is left out: its card is
#: not named.
RADIATION_STEP_RATIO = (3.40, 9.99)

#: An M3 Max's time against an RTX A6000 on the same work, (low, high):
#: 1.09-1.13x on an 8,000-column WRF dycore step (docs/nn-physics.md),
#: 1.94-2.03x on a 2,107-column full step (same file), 1.53x on the
#: edi-vr hex mesh (README), 2.39x integration time on the 5-domain
#: buildings run and 4.13x on its 50 m domain (docs/public/BUILDINGS.md),
#: and 4.04x on the Maui 3 km hex mesh (docs/ADAPTIVE-TIMESTEP.md).  The
#: A6000 was shared in most pairs, which makes these ratios LOW.
METAL_TO_A6000 = (1.09, 4.13)

#: The one Apple chip every Metal measurement was taken on.
METAL_REFERENCE_CARD = "m3-max"

#: Session-to-session variance on one box, "up to ~30%"
#: (docs/public/HARDWARE.md): a rate measured on the runner's own card
#: model is quoted from the measurement up to 1.3x it.
SESSION_VARIANCE = 1.30

#: Fixed-clock to adaptive-clock wall ratio, MEASURED on an idle A6000
#: (docs/ADAPTIVE-TIMESTEP.md): Maui 3 km 169 s against 358 s (0.47),
#: Manila 12 km 86 s against 144 s (0.60).  The low end of an adaptive
#: run's bracket is scaled by the smaller.
ADAPTIVE_WALL_RATIO = 0.47

#: History frame bytes per domain CELL for the full tape, (low, high),
#: MEASURED: 198 MB frames at 250x200x49 (80.8 B; docs/public/
#: FIRST-LIGHT.md) and 651 MB at 438x352x49 (86.2 B; docs/public/
#: HARDWARE.md).
FRAME_BYTES_PER_CELL = (80.8, 86.2)

#: The synchronous history write rate the asynchronous writer replaced:
#: a 5.24 GB frame in 81 s (docs/public/OUTPUT-VARIABLES.md) = 65 MB/s.
#: The high end of the output phase assumes no overlap at this rate; the
#: low end assumes the writer thread hides it completely.
SYNC_WRITE_BYTES_PER_SECOND = 65e6

#: NVRTC cold-compile wall, (low, high), MEASURED: about 22.6 s on an RTX
#: 5070 Ti and 53.8 s on an RTX 5090 (docs/manual/08-operational-
#: envelope.md).  Charged only when the kernel cache is cold.
KERNEL_COMPILE_S = (22.6, 53.8)

#: Extra fetch-to-first-step wall the first time woof runs on a box
#: (docs/manual/06-pipeline.md).
FIRST_RUN_ON_BOX_S = 31.0

#: 3-D Var minimisation wall (production tier), MEASURED 2026-11-20
#: (docs/var.md) on an A6000 shared with 2-3 jobs: 2.3-2.4 s for the
#: 90x60x41 tutorial with 41,857 obs and 3.2-4.5 s for a synthetic
#: 200x200x50.  Per analysis.
VAR3D_SOLVE_S = (2.3, 4.5)
VAR3D_MEASURED_CELLS = 200 * 200 * 50

#: LETKF solve seconds per cycle by ensemble size, MEASURED on an RTX 5090
#: on 132x132x49 (docs/da-nowcast-quickstart.md): N=10 6.3 s, N=20 10.1 s,
#: N=36 71.9 s.  Between rows the larger is taken; outside, the nearest
#: row, labelled.
LETKF_SOLVE_S = ((10, 6.3), (20, 10.1), (36, 71.9))


@dataclass(frozen=True)
class Anchor:
    """One measured forecast: the card, the work, the clock, the wall."""

    id: str
    card: str
    route: str
    rung: str
    columns: int
    nz: int
    dt_s: float
    radt_min: float | None
    s_per_step: float
    contention: str
    source: str
    note: str = ""
    #: False only for a bound carried from :mod:`woof.core.pace` that is
    #: not itself a measurement (the moist rung).
    measured: bool = True

    @property
    def rate(self) -> float:
        """Seconds per column-step at the reference nz."""
        from woof.core.pace import REFERENCE_NZ

        return self.s_per_step / (self.columns * self.nz / REFERENCE_NZ)

    @property
    def radiation_fraction(self) -> float | None:
        if self.radt_min is None:
            return None
        return _radiation_fraction(self.dt_s, self.radt_min)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "card": self.card, "route": self.route,
                "rung": self.rung, "columns": self.columns, "nz": self.nz,
                "dt_s": self.dt_s, "radt_min": self.radt_min,
                "s_per_step": self.s_per_step,
                "s_per_column_step_nz49": self.rate,
                "contention": self.contention, "source": self.source,
                "note": self.note, "measured": self.measured}


def _radiation_fraction(dt_s: float, radt_min: float) -> float:
    """Radiation steps per model step: one per ``radt``, at least every
    step (a ``radt`` shorter than ``dt`` fires every step)."""
    if radt_min <= 0:
        return 1.0
    return min(1.0, float(dt_s) / (float(radt_min) * 60.0))


def _pace_anchors() -> tuple[Anchor, ...]:
    """The dry and moist rows of :mod:`woof.core.pace`, as anchors."""
    from woof.core.pace import STEP_RATES

    rows = []
    for rung in ("dry", "moist"):
        rate = STEP_RATES[(rung, "resident")]
        for end, value in (("low", rate.low), ("high", rate.high)):
            rows.append(Anchor(
                id=f"pace-{rung}-resident-{end}", card="rtx-5090", route=WRF,
                rung=rung, columns=1_000_000, nz=49, dt_s=15.0,
                radt_min=None, s_per_step=value * 1_000_000,
                contention="as woof.core.pace states",
                source="woof/core/pace.py STEP_RATES "
                       f"({rung}, resident): {rate.basis[:160]}...",
                measured=rate.measured))
    return tuple(rows)


#: Every measured forecast the estimator prices from.  A new row needs a
#: run with the card, grid, dt and contention named, and a receipt.
ANCHORS: tuple[Anchor, ...] = (
    # -- WRF ARW grid, full physics --------------------------------------
    Anchor("grounding-3km-conus-5090", "rtx-5090", WRF, "full",
           796 * 636, 49, 15.0, 12.0, 1366.03 / 1440,
           "sole occupant confirmed, Linux",
           "docs/public/receipts/obsbattery/grounding-3km-conus-run-report"
           ".json (1440 steps in 1366.03 s)",
           "gpuwm 1.4.1; radt 12 min as woof/core/pace.py states"),
    Anchor("hardware-12km-4090", "rtx-4090", WRF, "full",
           438 * 352, 49, 60.0, 12.0, 400.0 / 360,
           "not stated, Linux",
           "docs/public/HARDWARE.md Linux notes (6 h, 360 steps in 400 s)",
           "Morrison + RTE-RRTMGP + YSU + Noah + KF"),
    Anchor("a6000-manila-12km", "rtx-a6000", WRF, "full",
           150 * 120, 49, 30.0, 12.0, 144.0 / 2880,
           "idle, one forecast at a time",
           "docs/ADAPTIVE-TIMESTEP.md (Manila 12 km fixed 30 s, 2880 steps "
           "in 144 s)", "KF on; 24 h of hourly history inside the wall"),
    Anchor("a6000-maui-3km", "rtx-a6000", WRF, "full",
           150 * 120, 49, 7.5, 12.0, 358.0 / 11520,
           "idle, one forecast at a time",
           "docs/ADAPTIVE-TIMESTEP.md (Maui 3 km fixed 7.5 s, 11520 steps "
           "in 358 s)", "no cumulus; 24 h of hourly history inside the "
           "wall"),
    # -- hex (MPAS v8.4.2 dycore on a mesh), full physics ----------------
    Anchor("hex-point-5090", "rtx-5090", HEX, "full",
           43_884, 55, 5.0, None, 0.2367,
           "card otherwise idle",
           "components/hex/docs/hex-point-hrrr.md (reference arm, s per "
           "step after the first)", "radiation steps inside the mean"),
    Anchor("hex-maui-a6000", "rtx-a6000", HEX, "full",
           7_042, 55, 18.0, None, 359.0 / 2400,
           "shared card",
           "docs/ADAPTIVE-TIMESTEP.md (Maui 3 km hex, fixed, 2400 steps in "
           "359 s)", "levels not stated: taken at the hex default 55 "
           "(woof/hex/device_admission.py MODEL_VERTICAL_LEVELS)"),
    Anchor("hex-maui-m3max", "m3-max", HEX, "full",
           7_042, 55, 18.0, None, 1451.0 / 2400,
           "shared Mac",
           "docs/ADAPTIVE-TIMESTEP.md (Maui 3 km hex on Metal, fixed, 2400 "
           "steps in 1451 s)", "levels taken at the hex default 55"),
) + _pace_anchors()


# --------------------------------------------------------------------------
# rates
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """One way of pricing a step: an anchor, a radiation ratio, and a
    card factor.  ``ordinary`` is s per column-step at nz=49 on the
    target card; a radiation step costs ``radiation_ratio`` of them."""

    anchor: Anchor | None
    ordinary: float
    radiation_ratio: float | None
    card_factor: float
    label: str

    def per_step(self, columns: int, nz: int, f_rad: float) -> tuple[float, float]:
        """(ordinary seconds, radiation extra seconds) per model step."""
        from woof.core.pace import REFERENCE_NZ

        base = self.ordinary * columns * nz / REFERENCE_NZ
        extra = (0.0 if self.radiation_ratio is None
                 else base * (self.radiation_ratio - 1.0) * f_rad)
        return base, extra


@dataclass(frozen=True)
class RateBracket:
    provenance: str
    candidates: tuple[Candidate, ...]
    basis: tuple[str, ...]
    reference_cards: tuple[str, ...]


def _card_bandwidth(runner: Runner) -> tuple[float, str | None]:
    """The runner's bandwidth, or the slowest same-backend row's with a
    note (a SUBSTITUTE) when the runner does not say."""
    if runner.bandwidth_gbs:
        return float(runner.bandwidth_gbs), None
    slowest = min((c for c in RUNNER_CARDS.values()
                   if c.backend == runner.backend),
                  key=lambda c: c.bandwidth_gbs)
    return slowest.bandwidth_gbs, (
        f"the runner states no memory bandwidth and no table row names "
        f"{runner.name!r}, so it is priced as the slowest {runner.backend} "
        f"card in the table ({slowest.name}, {slowest.bandwidth_gbs:g} "
        "GB/s): a SUBSTITUTE; give bandwidth_gbs in a [runner] TOML or run "
        "--calibrate")


def bandwidth_factor(anchor_gbs: float, target_gbs: float
                     ) -> tuple[float, float]:
    """(fast, slow) multipliers on an anchor's time for the target card."""
    ratio = float(anchor_gbs) / float(target_gbs)
    values = [ratio ** e for e in BANDWIDTH_EXPONENT]
    return min(values), max(values)


def _card_factors(anchor: Anchor, runner: Runner, target_gbs: float
                  ) -> tuple[tuple[float, float], str]:
    """Multipliers carrying ``anchor`` to the runner, and how."""
    source = RUNNER_CARDS[anchor.card]
    if anchor.card == runner.card_key:
        return (1.0, 1.0), MEASURED
    if source.backend == runner.backend:
        return bandwidth_factor(source.bandwidth_gbs, target_gbs), SCALED
    if runner.backend == METAL:
        a6000 = RUNNER_CARDS["rtx-a6000"]
        to_a6000 = bandwidth_factor(source.bandwidth_gbs, a6000.bandwidth_gbs)
        ref = RUNNER_CARDS[METAL_REFERENCE_CARD]
        to_chip = bandwidth_factor(ref.bandwidth_gbs, target_gbs)
        return ((to_a6000[0] * METAL_TO_A6000[0] * to_chip[0],
                 to_a6000[1] * METAL_TO_A6000[1] * to_chip[1]), SCALED)
    # A Metal anchor priced for a CUDA card: invert the ratio.
    a6000 = RUNNER_CARDS["rtx-a6000"]
    to_m3 = bandwidth_factor(source.bandwidth_gbs,
                             RUNNER_CARDS[METAL_REFERENCE_CARD].bandwidth_gbs)
    to_card = bandwidth_factor(a6000.bandwidth_gbs, target_gbs)
    return ((to_m3[0] / METAL_TO_A6000[1] * to_card[0],
             to_m3[1] / METAL_TO_A6000[0] * to_card[1]), SCALED)


def _rung_anchors(route: str, rung: str, anchors: Sequence[Anchor]
                  ) -> tuple[list[Anchor], float, str | None]:
    """Anchors for ``rung`` on ``route``: (rows, multiplier, note)."""
    from woof.core.pace import MYNN_PREMIUM

    rows = [a for a in anchors if a.route == route and a.rung == rung]
    if rows:
        return rows, 1.0, None
    if rung == "full+mynn+noahmp":
        rows = [a for a in anchors if a.route == route and a.rung == "full"]
        if rows:
            return rows, MYNN_PREMIUM, (
                f"MYNN + Noah-MP: the full-physics anchors times the MEASURED "
                f"{MYNN_PREMIUM}x premium woof/core/pace.py carries")
    on_route = [a for a in anchors if a.route == route]
    if not on_route:
        on_route = [a for a in anchors if a.route == WRF]
    slowest = max(on_route, key=lambda a: a.rate)
    rows = [a for a in on_route if a.rung == slowest.rung]
    return rows, 1.0, (
        f"no anchor names the {rung!r} rung on the {route} route, so it is "
        f"priced from the slowest rung that has one ({slowest.rung}): a "
        "SUBSTITUTE basis, not a measurement of this configuration; "
        "--calibrate measures it")


def rate_bracket(runner: Runner, route: str, rung: str, *, columns: int,
                 anchors: Sequence[Anchor] | None = None,
                 calibration: Mapping[str, Any] | None = None) -> RateBracket:
    """Every candidate pricing of one step of ``rung`` on ``route``."""
    from woof.core.pace import LAUNCH_BOUND_COLUMNS

    anchors = tuple(ANCHORS if anchors is None else anchors)
    basis: list[str] = []
    if calibration is not None:
        ordinary = float(calibration["ordinary_s_per_column_step_nz49"])
        ratio = calibration.get("radiation_step_ratio")
        ratios = ((float(ratio),) if ratio else RADIATION_STEP_RATIO)
        candidates = tuple(
            Candidate(None, ordinary * f, r, 1.0, f"calibration x{f:g}")
            for r in ratios for f in (1.0, SESSION_VARIANCE))
        basis.append(
            f"CALIBRATED on this runner ({calibration.get('card')}, "
            f"{calibration.get('recorded_utc')}): "
            f"{calibration.get('steps')} measured steps of "
            f"{calibration.get('config')}, {ordinary:.3g} s per ordinary "
            f"column-step at nz=49; quoted up to {SESSION_VARIANCE}x for "
            "session variance")
        return RateBracket(CALIBRATED, candidates, tuple(basis),
                           (str(calibration.get("card")),))
    rows, premium, note = _rung_anchors(route, rung, anchors)
    substituted = note is not None and "SUBSTITUTE" in note
    if note:
        basis.append(note)
    same_card = [a for a in rows if a.card == runner.card_key]
    if same_card:
        rows = same_card
    small = columns < LAUNCH_BOUND_COLUMNS
    sized = [a for a in rows if (a.columns < LAUNCH_BOUND_COLUMNS) == small]
    if sized:
        rows = sized
    else:
        basis.append(
            f"no anchor of this rung is on the same side of "
            f"{LAUNCH_BOUND_COLUMNS:,} columns as this domain ({columns:,}); "
            "per-column rates differ across that line, so treat the bracket "
            "as indicative")
    target_gbs, bw_note = _card_bandwidth(runner)
    if bw_note:
        basis.append(bw_note)
        substituted = True
    provenance = MEASURED
    candidates = []
    for anchor in rows:
        (fast, slow), how = _card_factors(anchor, runner, target_gbs)
        if how != MEASURED or not anchor.measured:
            provenance = SCALED
        f_anchor = anchor.radiation_fraction
        ratios = RADIATION_STEP_RATIO if f_anchor is not None else (None,)
        for ratio in ratios:
            ordinary = anchor.rate * premium
            if ratio is not None:
                ordinary /= 1.0 + (ratio - 1.0) * f_anchor
            factors = (fast, slow * (SESSION_VARIANCE if how == MEASURED
                                     else 1.0))
            for factor in sorted(set(factors)):
                candidates.append(Candidate(
                    anchor, ordinary * factor, ratio, factor,
                    f"{anchor.id} x{factor:.3g}"
                    + ("" if ratio is None else f" R={ratio:g}")))
        basis.append(
            f"{anchor.id}: {anchor.s_per_step:.4g} s/step on "
            f"{RUNNER_CARDS[anchor.card].name} at {anchor.columns:,} "
            f"columns x {anchor.nz}, dt {anchor.dt_s:g} s"
            + ("" if anchor.radt_min is None else
               f", radt {anchor.radt_min:g} min")
            + f" ({anchor.contention}; {anchor.source})"
            + ("" if how == MEASURED else
               f" -> x{fast:.3g}-{slow:.3g} for this card ({how})")
            + ("" if anchor.measured else " [a BOUND, not a measurement]"))
    if substituted:
        provenance = SUBSTITUTED
    cards = tuple(sorted({RUNNER_CARDS[a.card].name for a in rows}))
    return RateBracket(provenance, tuple(candidates), tuple(basis), cards)


# --------------------------------------------------------------------------
# the configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DomainWork:
    """What one domain asks of the card."""

    grid_id: int
    columns: int
    nz: int
    dt_s: float
    window_s: float
    rung: str
    radt_min: float | None
    cu_physics: int
    cudt_min: float | None
    history_s: float | None
    adaptive: bool

    @property
    def steps(self) -> int:
        return 0 if self.dt_s <= 0 else max(1, math.ceil(self.window_s
                                                         / self.dt_s))

    @property
    def radiation_fraction(self) -> float:
        return (0.0 if self.radt_min is None
                else _radiation_fraction(self.dt_s, self.radt_min))

    @property
    def frames(self) -> int:
        if not self.history_s or self.history_s <= 0:
            return 0
        return int(self.window_s // self.history_s) + 1

    def as_dict(self) -> dict[str, Any]:
        return {"grid_id": self.grid_id, "columns": self.columns,
                "nz": self.nz, "dt_s": self.dt_s, "window_s": self.window_s,
                "steps": self.steps, "rung": self.rung,
                "radt_min": self.radt_min,
                "radiation_steps": math.ceil(self.steps
                                             * self.radiation_fraction),
                "cu_physics": self.cu_physics, "cudt_min": self.cudt_min,
                "history_s": self.history_s, "frames": self.frames,
                "adaptive_clock": self.adaptive}


def configured_route(path: str | Path) -> str:
    """``hex`` when the TOML's ``[model]`` names hex, else ``wrf``."""
    try:
        from woof.go_hex import configured_engine

        return HEX if configured_engine(Path(path)) == "hex" else WRF
    except Exception:  # noqa: BLE001 - an unreadable [model] is WRF's
        return WRF


def domains_of(exp) -> list[DomainWork]:
    """Every domain's work, read off a loaded experiment."""
    from tilestream import autoplan

    from woof.config import effective_radt_minutes, radiation_enabled

    run_seconds = float(getattr(exp, "run_seconds", 0.0) or 0.0)
    start = getattr(exp, "start_time", None)
    rows = []
    for domain in exp.domains:
        cfg = domain.run
        offset = 0.0
        own_start = getattr(domain, "start_time", None)
        if own_start is not None and start is not None:
            offset = max(0.0, (own_start - start).total_seconds())
        radt = (effective_radt_minutes(cfg) if radiation_enabled(cfg)
                else None)
        cu = int(getattr(cfg, "cu_physics", 0) or 0)
        rows.append(DomainWork(
            grid_id=int(domain.grid_id),
            columns=int(cfg.nx) * int(cfg.ny), nz=int(cfg.nz),
            dt_s=float(cfg.dt), window_s=max(0.0, run_seconds - offset),
            rung=autoplan.rung_of(cfg), radt_min=radt, cu_physics=cu,
            cudt_min=(float(getattr(cfg, "cudt_minutes", 0.0) or 0.0)
                      if cu else None),
            history_s=float(domain.history_interval_s or 0.0) or None,
            adaptive=bool(getattr(cfg, "use_adaptive_time_step", False))))
    return rows


# --------------------------------------------------------------------------
# the estimate
# --------------------------------------------------------------------------
@dataclass
class Phase:
    name: str
    low_s: float
    high_s: float
    basis: str
    provenance: str = MEASURED

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "low_s": round(self.low_s, 3),
                "high_s": round(self.high_s, 3), "provenance": self.provenance,
                "basis": self.basis}


@dataclass
class Estimate:
    config: str
    route: str
    road: str
    runner: Runner
    domains: list[DomainWork]
    phases: list[Phase]
    rate: RateBracket
    run_seconds: float
    seconds_per_step: tuple[float, float]
    notes: list[str] = field(default_factory=list)
    sharing: Mapping[str, Any] | None = None

    @property
    def low_s(self) -> float:
        return sum(p.low_s for p in self.phases)

    @property
    def high_s(self) -> float:
        return sum(p.high_s for p in self.phases)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "config": self.config,
            "route": self.route,
            "road": self.road,
            "runner": self.runner.as_dict(),
            "run_seconds": self.run_seconds,
            "provenance": self.rate.provenance,
            "reference_cards": list(self.rate.reference_cards),
            "wall_seconds_low": round(self.low_s, 3),
            "wall_seconds_high": round(self.high_s, 3),
            "root_seconds_per_step_low": round(self.seconds_per_step[0], 6),
            "root_seconds_per_step_high": round(self.seconds_per_step[1], 6),
            "phases": [p.as_dict() for p in self.phases],
            "domains": [d.as_dict() for d in self.domains],
            "rate_basis": list(self.rate.basis),
            "notes": list(self.notes),
            "sharing": None if self.sharing is None else dict(self.sharing),
            "model": {
                "bandwidth_exponent": list(BANDWIDTH_EXPONENT),
                "radiation_step_ratio": list(RADIATION_STEP_RATIO),
                "metal_to_a6000": list(METAL_TO_A6000),
                "session_variance": SESSION_VARIANCE,
            },
        }


def _steps_wall(domains: Sequence[DomainWork], candidate_for
                ) -> tuple[float, float, float, float]:
    """(ordinary s, radiation s, root ordinary s/step, root rad s/step)."""
    ordinary = radiation = 0.0
    root = (0.0, 0.0)
    for index, d in enumerate(domains):
        cand = candidate_for(d)
        base, extra = cand.per_step(d.columns, d.nz, d.radiation_fraction)
        ordinary += base * d.steps
        radiation += extra * d.steps
        if index == 0:
            root = (base, extra)
    return ordinary, radiation, root[0], root[1]


def estimate(exp, runner: Runner, *, config: str = "<config>",
             route: str = WRF, prepared: bool = False,
             kernel_cache_cold: bool | None = None,
             first_run_on_box: bool = False,
             da: str | None = None, da_cycles: int = 1, members: int = 0,
             calibration: Mapping[str, Any] | None = None,
             anchors: Sequence[Anchor] | None = None,
             streamed=None, copies: int = 1) -> Estimate:
    """Price ``exp`` on ``runner``.  ``streamed`` is the root's streamed
    envelope (``None``: resident), as :mod:`woof.core.preflight` resolves
    it; :func:`estimate_config` resolves it for a caller."""
    domains = domains_of(exp)
    if not domains:
        raise ValueError(f"{config} carries no domain to price")
    run_seconds = float(getattr(exp, "run_seconds", 0.0) or 0.0)
    notes: list[str] = []
    root = domains[0]
    brackets = {
        (d.rung, d.columns < _launch_bound()): rate_bracket(
            runner, route, d.rung, columns=d.columns, anchors=anchors,
            calibration=calibration)
        for d in domains}
    root_bracket = brackets[(root.rung, root.columns < _launch_bound())]

    # Every combination of candidates is too many for a tree; each domain
    # takes its own cheapest and dearest candidate (by its own wall).
    def pick(d: DomainWork, dearest: bool) -> Candidate:
        cands = brackets[(d.rung, d.columns < _launch_bound())].candidates

        def total(c: Candidate) -> float:
            base, extra = c.per_step(d.columns, d.nz, d.radiation_fraction)
            return base + extra
        return (max if dearest else min)(cands, key=total)

    lo_ord, lo_rad, lo_root, lo_root_rad = _steps_wall(
        domains, lambda d: pick(d, False))
    hi_ord, hi_rad, hi_root, hi_root_rad = _steps_wall(
        domains, lambda d: pick(d, True))
    if any(d.adaptive for d in domains):
        lo_ord *= ADAPTIVE_WALL_RATIO
        lo_rad *= ADAPTIVE_WALL_RATIO
        notes.append(
            f"an adaptive clock runs fewer, longer steps: the low end is the "
            f"fixed-clock price x{ADAPTIVE_WALL_RATIO} (the MEASURED Maui "
            "ratio on an idle A6000, docs/ADAPTIVE-TIMESTEP.md) and the high "
            "end the fixed clock's; the step counts shown are the fixed "
            "clock's upper bound")
    road = "resident"
    step_bracket = (lo_root + lo_root_rad, hi_root + hi_root_rad)
    if streamed is not None and route == WRF:
        from woof.core.pace import estimate_pace

        pace = estimate_pace(
            exp, streamed=streamed, machine=None, conservative=True,
            pinned_bytes_per_second=(int(runner.pinned_gbps * 1e9)
                                     if runner.pinned_gbps else None))
        if pace is not None:
            road = "streamed"
            # Replace the root's resident step cost by pace's streamed one.
            root_steps = root.steps
            lo_ord += (pace.seconds_per_step_low - lo_root - lo_root_rad) \
                * root_steps
            hi_ord += (pace.seconds_per_step_high - hi_root - hi_root_rad) \
                * root_steps
            step_bracket = (pace.seconds_per_step_low,
                            pace.seconds_per_step_high)
            notes.append(
                "the root STREAMS through host memory on this card; its step "
                "is priced by woof.core.pace's measured streamed rows (tile "
                "overhead and bus floor included, radiation inside the "
                "rows), not card-scaled: " + pace.basis[:400] + "...")
    phases: list[Phase] = []
    # ---------------------------------------------------------- prepare
    from woof.energy.cost import PREP_INTERCEPT_S, PREP_S_PER_MCELL

    cells = sum(d.columns * d.nz for d in domains)
    if prepared:
        phases.append(Phase("prepare", 0.0, 0.0,
                            "a prepared bundle is given: no fetch or "
                            "preparation"))
    elif route == HEX:
        phases.append(Phase(
            "prepare", 0.0, 0.0,
            "NOT PRICED: no receipt times the hex door's mesh, init and lbc "
            "stages; --calibrate measures launch to first step",
            SUBSTITUTED))
        notes.append("the hex route's preparation is not priced (no "
                     "receipt); the total excludes it")
    else:
        fit = PREP_INTERCEPT_S + PREP_S_PER_MCELL * cells / 1e6
        high = fit + FIRST_RUN_ON_BOX_S
        phases.append(Phase(
            "prepare", fit if not first_run_on_box else high, high,
            f"fetch-to-first-step fit {PREP_INTERCEPT_S:g} s + "
            f"{PREP_S_PER_MCELL:g} s per Mcell at {cells / 1e6:.2f} Mcells "
            "(GFS, warm caches, MEASURED on test host 1, docs/manual/"
            f"06-pipeline.md), plus {FIRST_RUN_ON_BOX_S:g} s the first time "
            "on a box; host CPU and network are not modelled"))
    # ---------------------------------------------------- kernel compile
    compile_lo, compile_hi = KERNEL_COMPILE_S
    if kernel_cache_cold is False:
        phases.append(Phase("kernel compile", 0.0, 0.0,
                            "the kernel cache on this box is warm"))
    elif kernel_cache_cold is True:
        phases.append(Phase(
            "kernel compile", compile_lo, compile_hi,
            "the kernel cache is COLD: NVRTC cold compile MEASURED 22.6 s "
            "(RTX 5070 Ti) to 53.8 s (RTX 5090), docs/manual/08-operational-"
            "envelope.md; `woof warm-kernels` pays it ahead of time"))
    else:
        phases.append(Phase(
            "kernel compile", 0.0, compile_hi,
            "the runner's kernel cache was not read (not this box): 0 when "
            "warm, up to the MEASURED 53.8 s cold compile"))
    # ------------------------------------------------------- the steps
    prov = root_bracket.provenance
    phases.append(Phase(
        "model steps", lo_ord, hi_ord,
        f"{sum(d.steps for d in domains):,} steps over {len(domains)} "
        f"domain(s) at the ordinary step cost ({prov})", prov))
    n_rad = sum(math.ceil(d.steps * d.radiation_fraction) for d in domains)
    phases.append(Phase(
        "radiation steps", lo_rad, hi_rad,
        f"{n_rad:,} radiation steps costing {RADIATION_STEP_RATIO[0]:g}-"
        f"{RADIATION_STEP_RATIO[1]:g}x an ordinary step (MEASURED on an "
        "A6000), charged as their EXTRA over an ordinary step"
        + ("; inside the streamed rows" if road == "streamed" else ""),
        prov))
    # ------------------------------------------------------- output
    frame_bytes = [(d.frames, d.columns * d.nz) for d in domains]
    out_bytes_hi = sum(n * c * FRAME_BYTES_PER_CELL[1] for n, c in frame_bytes)
    out_bytes_lo = sum(n * c * FRAME_BYTES_PER_CELL[0] for n, c in frame_bytes)
    frames = sum(n for n, _ in frame_bytes)
    phases.append(Phase(
        "output", 0.0, out_bytes_hi / SYNC_WRITE_BYTES_PER_SECOND,
        f"{frames} history frame(s), {out_bytes_lo / 1e9:.2f}-"
        f"{out_bytes_hi / 1e9:.2f} GB at the MEASURED 80.8-86.2 bytes per "
        "cell of the full tape; 0 s when the asynchronous writer hides it, "
        "up to the MEASURED 65 MB/s synchronous rate when it does not "
        "(docs/public/OUTPUT-VARIABLES.md).  An [output] selection writes "
        "less"))
    # ------------------------------------------------------- restart
    restart_s = float(getattr(exp, "restart_interval_s", 0.0) or 0.0)
    restarts = (int(run_seconds // restart_s) if restart_s > 0 else 0)
    if restarts:
        from woof.energy.cost import D2H_GBPS, RESTART_3D_FIELDS

        rbytes = restarts * sum(RESTART_3D_FIELDS * d.columns * d.nz * 4
                                for d in domains)
        phases.append(Phase(
            "restart writes", rbytes / (D2H_GBPS * 1e9),
            rbytes / SYNC_WRITE_BYTES_PER_SECOND,
            f"{restarts} restart set(s) every {restart_s / 3600:g} h, "
            f"~{rbytes / 1e9:.1f} GB at woof.energy.cost's assumed "
            f"{RESTART_3D_FIELDS} 3-D fields (an ASSUMPTION, not counted per "
            f"suite), between the measured {D2H_GBPS:g} GB/s device copy and "
            "the 65 MB/s synchronous write", SCALED))
    # ------------------------------------------------------------- DA
    if da:
        phases.append(_da_phase(da, da_cycles, members, domains, runner,
                                lo_root + lo_root_rad, hi_root + hi_root_rad))
    sharing = None
    if copies > 1:
        sharing = sharing_bracket(copies, columns=root.columns,
                                  mps=runner.mps)
        for phase in phases:
            if phase.name in ("model steps", "radiation steps"):
                phase.low_s *= sharing["per_run_slowdown"][0]
                phase.high_s *= sharing["per_run_slowdown"][1]
    if root.columns < _launch_bound() and road == "resident":
        notes.append(
            f"under {_launch_bound():,} columns a card is not loaded; the "
            "anchors on the same side of that line were preferred")
    if root.cu_physics and root.cudt_min:
        notes.append(
            f"cumulus every {root.cudt_min:g} min is inside the anchors and "
            "not priced separately: no receipt isolates its cost")
    return Estimate(config=config, route=route, road=road, runner=runner,
                    domains=domains, phases=phases, rate=root_bracket,
                    run_seconds=run_seconds, seconds_per_step=step_bracket,
                    notes=notes, sharing=sharing)


def _launch_bound() -> int:
    from woof.core.pace import LAUNCH_BOUND_COLUMNS

    return LAUNCH_BOUND_COLUMNS


def _da_phase(da: str, cycles: int, members: int,
              domains: Sequence[DomainWork], runner: Runner,
              step_lo: float, step_hi: float) -> Phase:
    cells = domains[0].columns * domains[0].nz
    cycles = max(1, int(cycles))
    if da == "3dvar":
        scale = max(1.0, cells / VAR3D_MEASURED_CELLS)
        lo, hi = VAR3D_SOLVE_S[0] * cycles, VAR3D_SOLVE_S[1] * scale * cycles
        return Phase(
            "data assimilation", lo, hi,
            f"{cycles} 3D-Var analysis(es): minimisation MEASURED 2.3-4.5 s "
            "per analysis on an A6000 shared with 2-3 jobs (docs/var.md, "
            "production tier, up to 200x200x50)"
            + (f"; this root is {scale:.1f}x the largest measured grid and "
               "the high end is scaled linearly in cells (UNMEASURED)"
               if scale > 1 else "")
            + "; obs preprocessing is not priced",
            MEASURED if scale <= 1 else SCALED)
    if da == "letkf":
        n = max(1, int(members))
        rows = sorted(LETKF_SOLVE_S)
        above = [s for m, s in rows if m >= n]
        solve = above[0] if above else rows[-1][1]
        label = MEASURED if any(m == n for m, _ in rows) else SCALED
        return Phase(
            "data assimilation", solve * cycles, solve * cycles,
            f"{cycles} LETKF cycle(s) at N={n}: solve {solve:g} s per cycle "
            "MEASURED on an RTX 5090 at 132x132x49 (docs/da-nowcast-"
            "quickstart.md: N=10 6.3 s, N=20 10.1 s, N=36 71.9 s; between "
            "rows the larger is taken, beyond them the largest), not scaled "
            "to this grid or card.  The member forecasts are N runs of this "
            "configuration: price them with --copies", label)
    raise ValueError(f"--da takes 3dvar or letkf, not {da!r}")


# --------------------------------------------------------------------------
# several runs on one card
# --------------------------------------------------------------------------
#: MEASURED slowdowns of one run when ``copies`` runs share a card.
#: ``per_run`` is one run's wall against running alone.
SHARING_ROWS: tuple[Mapping[str, Any], ...] = (
    {"copies": 2, "mps": False, "large": True, "per_run": 22.6 / 11.66,
     "source": "woof/core/pace.py: a streamed HRRR 1200x900x49 forecast on "
               "an RTX 4090 ran 11.66 s/step alone and 22.6 s/step with a "
               "second forecast on the card"},
    {"copies": 4, "mps": False, "large": False, "per_run": 1.0 / 0.93,
     "source": "docs/public/MASS-ENSEMBLE.md: four co-resident 12-3 km "
               "members (88x78 and 272x232, 49 levels) in ordinary contexts "
               "ran 0.93x the one-member step rate, RTX PRO 6000"},
)


def sharing_bracket(copies: int, *, columns: int, mps: bool
                    ) -> dict[str, Any]:
    """One run's slowdown when ``copies`` runs share the card, (low, high).

    The low end is the measured row for the domain's size class scaled
    linearly in copies; the high end is ``copies`` (pure time-slicing: the
    card does the same total work, so each run waits for the others).  A
    copy count no row measured is labelled UNMEASURED.
    """
    copies = max(1, int(copies))
    if copies == 1:
        return {"copies": 1, "per_run_slowdown": (1.0, 1.0),
                "measured": True, "basis": "alone on the card"}
    large = columns >= _launch_bound()
    rows = [r for r in SHARING_ROWS if r["large"] == large
            and r["mps"] == bool(mps)] or [
        r for r in SHARING_ROWS if r["large"] == large] or list(SHARING_ROWS)
    row = min(rows, key=lambda r: abs(r["copies"] - copies))
    per_copy_excess = (row["per_run"] - 1.0) / (row["copies"] - 1)
    low = 1.0 + per_copy_excess * (copies - 1)
    high = max(float(copies), low)
    measured = row["copies"] == copies and row["mps"] == bool(mps)
    return {"copies": copies, "per_run_slowdown": (low, high),
            "measured": measured,
            "basis": (f"{'MEASURED' if measured else 'UNMEASURED: scaled from'}"
                      f" {row['copies']} runs ({row['per_run']:.2f}x each; "
                      f"{row['source']}) at the low end; pure time-slicing "
                      f"({copies}x) at the high end")}


def admitted_copies(exp, runner: Runner, slots: int | str
                    ) -> tuple[int, dict[str, Any]]:
    """How many copies of ``exp`` the runner's card admits by memory:
    :data:`woof.gpu_packing.DEFAULT_MEM_MARGIN_GIB` kept free beside the
    priced peaks (the multi-run rule, :data:`woof.multi_run.
    PACKED_ADMISSION_RULE` part 2), capped at ``slots`` (``auto``:
    :data:`woof.gpu_packing.AUTO_MAX_SLOTS_PER_GPU`, and at most one per
    CPU core, as the energy worker caps it)."""
    from woof.gpu_packing import (AUTO_MAX_SLOTS_PER_GPU,
                                  DEFAULT_MEM_MARGIN_GIB, SLOTS_AUTO)

    cap = AUTO_MAX_SLOTS_PER_GPU if slots == SLOTS_AUTO else int(slots)
    if slots == SLOTS_AUTO and runner.cpu_cores:
        cap = min(cap, int(runner.cpu_cores))
    detail: dict[str, Any] = {"cap": cap, "margin_gib": DEFAULT_MEM_MARGIN_GIB}
    try:
        from woof.supervisor import priced_reservation_bytes

        peak = priced_reservation_bytes(exp)
    except Exception as error:  # noqa: BLE001 - unpriced is an answer
        peak = None
        detail["pricing_error"] = str(error)[:200]
    detail["priced_peak_bytes"] = peak
    if peak is None or not runner.vram_bytes:
        detail["reason"] = ("unpriced, or the card's memory is unknown: an "
                            "unpriced run runs alone on its card "
                            "(PACKED_ADMISSION_RULE)")
        return 1, detail
    room = runner.vram_bytes - DEFAULT_MEM_MARGIN_GIB * (1 << 30)
    by_memory = int(room // peak) if peak > 0 else cap
    detail["by_memory"] = by_memory
    copies = max(0, min(cap, by_memory))
    detail["reason"] = (f"{by_memory} priced peak(s) of "
                        f"{peak / (1 << 30):.2f} GiB fit "
                        f"{runner.vram_bytes / (1 << 30):.1f} GiB less the "
                        f"{DEFAULT_MEM_MARGIN_GIB:g} GiB margin; cap {cap}")
    return copies, detail


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------
def _num(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}"
    if value <= 0:
        return "0"
    return f"{value:#.2g}"


def fmt_duration(seconds: float) -> str:
    """A wall clock in the unit a reader thinks in."""
    if seconds >= 3600.0:
        return f"{_num(seconds / 3600.0)} h"
    if seconds >= 60.0:
        return f"{_num(seconds / 60.0)} min"
    return f"{_num(seconds)} s"


def render_text(est: Estimate, *, calibration_report: Mapping | None = None
                ) -> str:
    r = est.runner
    vram = "?" if r.vram_bytes is None else f"{r.vram_bytes / (1 << 30):.0f} GiB"
    bw = "? GB/s" if not r.bandwidth_gbs else f"{r.bandwidth_gbs:g} GB/s"
    lines = [
        f"woof estimate: {est.config}",
        f"  runner: {r.name} ({r.backend}, {vram}, {bw}"
        + ("" if not r.sm_count else
           f", {r.sm_count} {'GPU cores' if r.backend == METAL else 'SMs'}")
        + ")" + (f" [{r.provenance.get('name', 'table')}]"),
        f"  route: {est.route}, {len(est.domains)} domain(s), {est.road} "
        f"road, {fmt_duration(est.run_seconds)} forecast",
    ]
    for d in est.domains:
        rad = ("no radiation" if d.radt_min is None else
               f"radiation every {d.radt_min:g} min "
               f"(1 in {max(1, round(1 / d.radiation_fraction))} steps)")
        lines.append(
            f"    d{d.grid_id:02d} {d.columns:,} columns x {d.nz}, dt "
            f"{d.dt_s:g} s, {d.steps:,} steps"
            + (" (adaptive: at most)" if d.adaptive else "")
            + f", rung {d.rung}, {rad}")
    lines.append(
        f"  rate: {est.rate.provenance.upper()} "
        f"({', '.join(est.rate.reference_cards) or 'calibration'}); root "
        f"step {_num(est.seconds_per_step[0])}-"
        f"{_num(est.seconds_per_step[1])} s")
    width = max(len(p.name) for p in est.phases)
    lines.append(f"  {'phase':<{width}}  {'low':>9}  {'high':>9}")
    for p in est.phases:
        lines.append(f"  {p.name:<{width}}  {fmt_duration(p.low_s):>9}  "
                     f"{fmt_duration(p.high_s):>9}  [{p.provenance}]")
    lines.append(f"  {'total':<{width}}  {fmt_duration(est.low_s):>9}  "
                 f"{fmt_duration(est.high_s):>9}")
    if est.sharing:
        s = est.sharing
        lines.append(f"  sharing: {s['copies']} runs on this card, each "
                     f"{s['per_run_slowdown'][0]:.2f}-"
                     f"{s['per_run_slowdown'][1]:.2f}x slower "
                     f"({'measured' if s['measured'] else 'UNMEASURED'})")
    if calibration_report:
        c = calibration_report
        lines.append(
            f"  calibration: {c['measured_steps']} steps measured, "
            f"{c['measured_seconds_per_step']:.4g} s/step settled; forecast "
            f"extrapolates to {fmt_duration(c['extrapolated_forecast_s'])} "
            f"against {fmt_duration(c['predicted_forecast_low_s'])}-"
            f"{fmt_duration(c['predicted_forecast_high_s'])} predicted"
            + (" (inside)" if c.get("inside_bracket") else " (OUTSIDE)"))
    for note in est.notes:
        lines.append(f"  note: {note}")
    lines.append("  basis:")
    for b in est.rate.basis:
        lines.append(f"    - {b}")
    for p in est.phases:
        lines.append(f"    - {p.name}: {p.basis}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the door
# --------------------------------------------------------------------------
def resolve_runner(args) -> Runner:
    """``--card`` / ``--runner`` / detection, layered (module docstring of
    :mod:`woof.runner_cards`)."""
    base = None
    if getattr(args, "card", None):
        card = card_for(args.card)
        if card is None:
            raise RunnerSpecError(
                f"--card {args.card!r} names no row of the card table: "
                + ", ".join(RUNNER_CARDS))
        base = runner_from_card(card, source="flag (--card)")
    else:
        # A [runner] TOML layers over the detected box (its CPU count and
        # host RAM), unless it names a card of its own.
        base = detect_local_runner()
    if getattr(args, "runner", None):
        base = load_runner_toml(args.runner, base=base)
    if base is None:
        raise RunnerSpecError(
            "no GPU was detected on this box and no runner was named: give "
            "--card NAME (see --list-cards) or --runner RUNNER.toml")
    changes: dict[str, Any] = {}
    if getattr(args, "slots", None) is not None:
        changes["slots_per_gpu"] = args.slots
    if getattr(args, "mps", False):
        changes["mps"] = True
    return replace(base, **changes) if changes else base


def _cache_cold(runner: Runner, detected: bool) -> bool | None:
    if not detected:
        return None
    try:
        from woof.kernel_compile_notice import kernel_cache_is_cold

        return kernel_cache_is_cold()
    except Exception:  # noqa: BLE001
        return None


def estimate_config(path: str | Path, runner: Runner, *,
                    detected: bool = False, **kwargs) -> Estimate:
    """Load ``path`` and price it on ``runner``, resolving the road the
    planner would take on that card's memory."""
    from woof.experiment import load_experiment

    route = configured_route(path)
    exp = load_experiment(path)
    streamed = None
    if route == WRF and runner.vram_bytes and runner.backend == CUDA:
        try:
            from tilestream import autoplan

            from woof.core.preflight import streamed_forecast_envelope

            machine = autoplan.Machine(
                vram_bytes=int(runner.free_vram_bytes or runner.vram_bytes),
                host_bytes=int(runner.host_ram_bytes or runner.vram_bytes),
                name=runner.name)
            streamed = streamed_forecast_envelope(exp, machine=machine)
        except Exception:  # noqa: BLE001 - unresolvable is resident
            streamed = None
    kwargs.setdefault("kernel_cache_cold", _cache_cold(runner, detected))
    return estimate(exp, runner, config=str(path), route=route,
                    streamed=streamed, **kwargs)


def list_cards_text() -> str:
    rows = [f"{'card':<14} {'backend':<6} {'memory':>9} {'GB/s':>7} "
            f"{'SMs':>5} {'FP32 TF':>8}  source"]
    for c in RUNNER_CARDS.values():
        mem = "unified" if c.memory_mib is None else f"{c.memory_mib / 1024:.0f} GiB"
        rows.append(f"{c.key:<14} {c.backend:<6} {mem:>9} "
                    f"{c.bandwidth_gbs:>7g} {c.sm_count or '-':>5} "
                    f"{c.fp32_tflops or '-':>8}  {c.source}")
    return "\n".join(rows)


def main(args) -> int:
    if getattr(args, "list_cards", False):
        if args.json:
            print(json.dumps({"schema": SCHEMA + "/cards", "cards": [
                c.as_dict() for c in RUNNER_CARDS.values()]}, indent=2))
        else:
            print(list_cards_text())
        return 0
    if args.config is None:
        print("woof estimate: give a CONFIG (or --list-cards)",
              file=sys.stderr)
        return 2
    detected = not (args.card or args.runner)
    try:
        runner = resolve_runner(args)
    except RunnerSpecError as error:
        print(f"woof estimate: {error}", file=sys.stderr)
        return 2
    from woof import runtime_calibration as calib

    calibration = None
    if not args.no_calibration:
        calibration = calib.matching_calibration(
            runner, route=configured_route(args.config), config=args.config)
    copies = 1
    admission = None
    if runner.slots_per_gpu != 1 or args.copies:
        from woof.experiment import load_experiment

        exp = load_experiment(args.config)
        admitted, admission = admitted_copies(exp, runner,
                                              runner.slots_per_gpu)
        copies = (min(admitted, args.copies) if args.copies else admitted)
        copies = max(1, copies)
    try:
        est = estimate_config(
            args.config, runner, detected=detected, prepared=args.prepared,
            first_run_on_box=args.first_run, da=args.da,
            da_cycles=args.da_cycles, members=args.members,
            calibration=calibration, copies=copies)
    except (ValueError, OSError) as error:
        print(f"woof estimate: {error}", file=sys.stderr)
        return 2
    report = None
    if args.calibrate:
        if not detected:
            print("woof estimate: --calibrate runs on THIS box; drop --card "
                  "and --runner", file=sys.stderr)
            return 2
        report = calib.calibrate(
            args.config, steps=args.calibrate, warmup=args.warmup,
            estimate=est, prepared_root=args.prepared_root,
            record=not args.no_record, runner=runner,
            timeout_s=args.calibrate_timeout)
    document = est.to_json()
    if admission is not None:
        document["admission"] = admission
    if report is not None:
        document["calibration"] = report
    if args.json:
        print(json.dumps(document, indent=2, default=str))
    else:
        print(render_text(est, calibration_report=report))
        if admission is not None:
            print(f"  admission: {copies} cop(ies) on one card -- "
                  f"{admission.get('reason')}")
    return 0


def register_cli(subparsers):
    p = subparsers.add_parser(
        "estimate",
        help="how long a run will take on a runner (this box, --card NAME "
             "or --runner RUNNER.toml): a low-high wall clock with a "
             "per-phase breakdown, every figure labelled measured, scaled, "
             "calibrated or substituted",
        description=__doc__.split("\n\n")[0])
    p.add_argument("config", nargs="?", type=Path, metavar="CONFIG",
                   help="experiment TOML (WRF grid, or [model] engine = "
                        "\"hex\")")
    p.add_argument("--card", "--gpu", dest="card", metavar="NAME",
                   help="price on this card from the built-in table instead "
                        "of detecting this box's (see --list-cards)")
    p.add_argument("--runner", type=Path, metavar="TOML",
                   help="a [runner] table: card, backend, vram_gib, "
                        "bandwidth_gbs, sm_count, fp32_tflops, pinned_gbps, "
                        "host_ram_gib, cpu_cores, gpus, slots_per_gpu, mps; "
                        "its keys override the card row and detection")
    p.add_argument("--list-cards", action="store_true",
                   help="print the built-in card table and exit")
    p.add_argument("--slots", type=_slots, default=None, metavar="N|auto",
                   help="runs sharing one card (the multi-run runs_per_gpu "
                        "/ energy worker --slots-per-gpu grammar); admission "
                        "by priced peak memory")
    p.add_argument("--copies", type=int, default=0, metavar="N",
                   help="price N identical runs sharing the card (at most "
                        "what admission allows)")
    p.add_argument("--mps", action="store_true",
                   help="the shared card runs CUDA MPS")
    p.add_argument("--prepared", action="store_true",
                   help="the inputs are already prepared: no prepare phase")
    p.add_argument("--first-run", action="store_true",
                   help="first woof run on this box (adds the measured 31 s)")
    p.add_argument("--da", choices=("3dvar", "letkf"), default=None,
                   help="add a data-assimilation phase")
    p.add_argument("--da-cycles", type=int, default=1, metavar="N")
    p.add_argument("--members", type=int, default=0, metavar="N",
                   help="LETKF ensemble size")
    p.add_argument("--calibrate", type=int, default=0, metavar="N",
                   help="run N model steps of CONFIG on this box (after "
                        "--warmup steps) and extrapolate; reports measured "
                        "against predicted and records a local calibration")
    p.add_argument("--warmup", type=int, default=5, metavar="N",
                   help="steps discarded before --calibrate measures "
                        "(default 5: compile, first touch)")
    p.add_argument("--prepared-root", type=Path, default=None, metavar="DIR",
                   help="with --calibrate: run this prepared bundle instead "
                        "of fetching and preparing")
    p.add_argument("--calibrate-timeout", type=float, default=3600.0,
                   metavar="S", help="give up calibrating after S seconds")
    p.add_argument("--no-record", action="store_true",
                   help="with --calibrate: do not store the calibration")
    p.add_argument("--no-calibration", action="store_true",
                   help="ignore any stored local calibration")
    p.add_argument("--json", action="store_true",
                   help=f"print the {SCHEMA} document")
    p.set_defaults(func=main)
    return p


def _slots(value: str) -> int | str:
    from woof.gpu_packing import parse_count_or_auto

    try:
        return parse_count_or_auto(value, flag="--slots")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


__all__ = [
    "ADAPTIVE_WALL_RATIO", "ANCHORS", "Anchor", "BANDWIDTH_EXPONENT",
    "Candidate", "DomainWork", "Estimate", "FRAME_BYTES_PER_CELL",
    "KERNEL_COMPILE_S", "METAL_TO_A6000", "Phase", "RADIATION_STEP_RATIO",
    "RateBracket", "SCHEMA", "SESSION_VARIANCE", "SHARING_ROWS",
    "admitted_copies", "bandwidth_factor", "configured_route", "domains_of",
    "estimate", "estimate_config", "fmt_duration", "rate_bracket",
    "register_cli", "render_text", "resolve_runner", "sharing_bracket",
]
