"""Adaptive time step on hex (HEX-WRF parity item 4.1): WRF's ``calc_dt`` on the MPAS mesh.

WRF's controller is ported once, in :mod:`woof.core.adaptive_timestep` (graded bit for bit against
``dyn_em/adapt_timestep_em.F``), and the WRF route drives it from :mod:`woof.core.adaptive_clock`.  This
module drives the SAME controller from the hex forecast loop; nothing in the controller is copied.  What
differs is what the controller is handed and what a new step has to reach:

* the CFL numbers.  WRF reduces ``|u| dt msf / dx`` and its eta-coordinate ``|ww| dt rdnw / mu`` over the
  mass grid.  On the Voronoi mesh the horizontal number is the edge Courant number ``|u_n| dt / dcEdge``
  over every owned edge and level, and the vertical one is ``w dt / dz`` in the model's own coordinate:
  the scalar transport's vertical mass flux over the face density, times ``rdzu``, on the interior w
  levels (WRF's k = 2 .. kde-1).  See :class:`MeshCflProbe` for why not the geometric ``|w| / dz``.
* the acoustic substeps.  WRF sets ``time_step_sound = 0`` under an adaptive clock and derives the count
  from the live step (``dyn_em/solve_em.F:451-463``), because a fixed count under a growing dt walks the
  acoustic Courant number up until the split scheme fails.  hex keeps MPAS's own split (three dynamics
  subcycles, which the v8.4.1 CUDA subcycle requires) and raises ``config_number_of_sub_steps`` instead,
  so no acoustic step is longer than the one the mesh row's step was run at
  (:func:`longest_acoustic_step`, :func:`acoustic_substeps`).  Third-order Runge-Kutta takes its first stage as ONE acoustic step of
  ``dt_dyn/3`` whatever the substep count, so there the count cannot hold the bound and the largest step
  is capped instead (:func:`acoustic_ceiling`).
* the deadlines.  The root domain lands on every history frame, the run end, every checkpoint and every
  positive physics period, as :meth:`woof.core.adaptive_clock.AdaptiveClockDriver._to_next_alarm_anywhere`
  does on WRF, through the controller's own two-step look-ahead (:meth:`AdaptiveTimestepController.
  step_to_time`).  ``step_to_output_time`` is therefore always on: hex has no frame interpolation.
* the cadences.  Radiation, surface/PBL and cumulus fire on model TIME (their exact deadlines), not on a
  step count, as the WRF route does under an adaptive clock (``radiation_due_override`` and its two
  siblings on :class:`woof.core.physics.PhysicsDriver`).  A cadence welded to the step (``bldt``/``cudt``
  equal to the mesh row's dt, GF's pinned ``cudt = 0``) runs every step at the live dt, and the cumulus
  rain/NCA clock advances on every adaptive step (#278).

WRF quantises every step to ``1/precision`` seconds with ``precision = 100`` (the denominator of its
rational ESMF clock).  hex's clock is binary64 seconds, and hex refuses a step whose multiples are not exact
in binary64 (:func:`woof.hex.dt_admission.require_cadences`; every step check compares ``start + dt``), so
hex runs the same ``calc_dt`` with ``precision = 64``: every step is a whole number of 1/64 s, every sum of
steps is exact in binary64, and every landing is exact.

Off by default (``--adaptive-timestep off``).  With it off nothing here is imported by the forecast loop,
so the executed path, and every history byte, is the fixed step's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import Any, Iterable, Mapping, Sequence

from woof.core.adaptive_timestep import AdaptiveTimestepController

#: ``calc_dt``'s ``precision`` on hex: a power of two, so every step and every sum of steps is exact in
#: binary64 (WRF's 100 is exact only in its rational clock).
PRECISION = 64

SCHEMA = "woof-hex.adaptive-timestep/v1"

MODES = ("off", "on", "tropical")

#: WRF's namelist defaults (``Registry/registry.dimspec``/``README.namelist``): the vertical Courant target,
#: the horizontal one, and the root domain's growth limit per step in percent.
DEFAULT_TARGET_CFL = 1.2
DEFAULT_TARGET_HCFL = 0.84
DEFAULT_MAX_STEP_INCREASE_PCT = 5

#: ``woof domain --clock tropical-adaptive`` (#276): the vertical Courant target capped at 1.0 (the limit
#: the fixed tropical clock was measured to keep), 5 % growth, the fixed step as the floor.  Admitted only
#: inside the tropics, as on WRF.  WRF's opt-in also writes ``w_damping = 1``; hex has no w-damping term
#: yet (parity item 4.2), which the receipt records.
TROPICAL_TARGET_CFL = 1.0
TROPICAL_MAX_LAT = 25.0

#: WRF fills ``starting/max/min_time_step = -1`` with 4, 8 and 3 x dx(km) (``dyn_em/start_em.F:939-953``,
#: :func:`woof.core.adaptive_clock.wrf_default_clamps`).  hex starts at the mesh row's admitted step, which
#: already carries its own dx rule, and keeps WRF's RATIOS to the starting step: max 2x, min 3/4.
MAX_DT_RATIO = Fraction(2)
MIN_DT_RATIO = Fraction(3, 4)

#: The options this mode cannot run with, each refused by name.  ``attribute`` is read off the parsed
#: arguments; ``off`` is the value that means the option is not in use.
INCOMPATIBLE = (
    ("local_timestep", "--local-timestep", False,
     "local time stepping is the mesh's other way to take a longer step; WRF's adaptive clock and LTS each own "
     "the step, so they are mutually exclusive"),
    ("local_timestep_classing", "--local-timestep-classing", None,
     "an LTS classing instrument; LTS is mutually exclusive with the adaptive step"),
    ("fullstep_local_timestep", "--fullstep-local-timestep", False,
     "full-step local time stepping owns each class's step; it is mutually exclusive with the adaptive step"),
    ("iau_window_seconds", "--iau-window-seconds", None,
     "the IAU forcing is applied as a fraction per fixed step; the window bookkeeping is not wired to a "
     "varying step"),
    ("tc_climate", "--tc-climate", None,
     "the TC tracker samples on a fixed step count"),
    ("les_model", "--les-model", "off",
     "the LES closure attachment rebinds the dynamics subcycle with the fixed step's coefficients"),
    ("nest_triggers", "--nest-triggers", None,
     "the nest triggers look on a fixed step count"),
    ("da_analysis", "--da-analysis", None,
     "DA cycling re-couples an analysis at a fixed-step checkpoint"),
    ("jedi_background", "--jedi-background", False,
     "the JEDI background writer is not wired to a varying step"),
    ("balance_series", "--balance-series", False,
     "the balance series samples on a fixed step count"),
    ("chemistry_config", "--chemistry-config", None,
     "the chemistry transport holds its emission and photolysis clocks on the fixed step"),
    ("wind_farms", "--wind-farms", None,
     "the wind-farm power window is folded per fixed step"),
    ("wrf_hydro", "--wrf-hydro", None,
     "the WRF-Hydro routing clock is a whole number of fixed steps"),
    ("members", "--members", None,
     "an ensemble roster is not wired to the adaptive step yet"),
    ("samehost_devices", "--samehost-devices", None,
     "the same-host ranks share one fixed step"),
)


def add_arguments(parser: Any) -> None:
    """The ``--adaptive-*`` options, shared by the forecast door and the driver it launches."""

    parser.add_argument(
        "--adaptive-timestep", choices=MODES, default="off",
        help="OPT-IN, default off: WRF's adaptive time step (use_adaptive_time_step) on the WOOF column route. "
             "'on' takes WRF's targets (vertical Courant 1.2, horizontal 0.84, 5 %% growth per step) from the mesh "
             "row's step, between 3/4 and 2x of it; 'tropical' is woof domain --clock tropical-adaptive's "
             "(vertical Courant 1.0, the row's step as the floor), refused outside |lat| < 25.  Steps land on "
             "every history frame, checkpoint and physics period; mutually exclusive with local time stepping")
    parser.add_argument("--adaptive-target-cfl", type=float, default=None, metavar="CFL",
                        help="vertical Courant target (WRF target_cfl; default 1.2, tropical 1.0)")
    parser.add_argument("--adaptive-target-hcfl", type=float, default=None, metavar="CFL",
                        help="horizontal (edge) Courant target (WRF target_hcfl; default 0.84)")
    parser.add_argument("--adaptive-max-increase-pct", type=int, default=None, metavar="PCT",
                        help="largest growth of the step per step in percent (WRF max_step_increase_pct; default 5)")
    parser.add_argument("--adaptive-starting-dt", default=None, metavar="SECONDS",
                        help="the first step (WRF starting_time_step; default the mesh row's step)")
    parser.add_argument("--adaptive-min-dt", default=None, metavar="SECONDS",
                        help="the smallest step (WRF min_time_step; default 3/4 of the start, tropical the start)")
    parser.add_argument("--adaptive-max-dt", default=None, metavar="SECONDS",
                        help="the largest step (WRF max_time_step; default 2x the start)")


_FORWARDED = (("adaptive_target_cfl", "--adaptive-target-cfl"), ("adaptive_target_hcfl", "--adaptive-target-hcfl"),
              ("adaptive_max_increase_pct", "--adaptive-max-increase-pct"),
              ("adaptive_starting_dt", "--adaptive-starting-dt"), ("adaptive_min_dt", "--adaptive-min-dt"),
              ("adaptive_max_dt", "--adaptive-max-dt"))


def forwarded_arguments(arguments: Any) -> list[str]:
    """The driver argv for the door's ``--adaptive-*`` selection; empty with the mode off."""

    mode = getattr(arguments, "adaptive_timestep", "off") or "off"
    if mode == "off":
        return []
    argv = ["--adaptive-timestep", mode]
    for attribute, flag in _FORWARDED:
        value = getattr(arguments, attribute, None)
        if value is not None:
            argv += [flag, str(value)]
    return argv


def quantise(seconds: Any) -> Fraction:
    """A step in seconds on the hex lattice: an exact rational with denominator ``PRECISION`` (64)."""

    value = Fraction(str(seconds)) if not isinstance(seconds, Fraction) else seconds
    scaled = value * PRECISION
    if scaled.denominator != 1:
        raise ValueError(f"{seconds} s is not a whole number of 1/{PRECISION} s (the hex adaptive step lattice, "
                         "exact in binary64)")
    return value


def floor_to_precision(seconds: Fraction) -> Fraction:
    """The lattice floor a halved landing step is snapped to (never below one lattice unit)."""

    return max(Fraction(math.floor(seconds * PRECISION), PRECISION), Fraction(1, PRECISION))


def refusal(arguments: Any) -> str | None:
    """The first option the adaptive step cannot run with, as a sentence; ``None`` when there is none."""

    if getattr(arguments, "adaptive_timestep", "off") in (None, "off"):
        return None
    for attribute, flag, off, reason in INCOMPATIBLE:
        value = getattr(arguments, attribute, off)
        if value is None or value is False or value == off or value in ("", "off", "none") or (
                isinstance(value, (tuple, list)) and not value):
            continue
        return f"--adaptive-timestep with {flag}: {reason}"
    if getattr(arguments, "dt", None) not in (None, "row", "auto"):
        return ("--adaptive-timestep with an explicit --dt: the adaptive step starts at the mesh row's admitted "
                "step; name its first step with --adaptive-starting-dt")
    return None


def acoustic_steps(dt: Fraction, substeps: int, order: int, splits: int = 3) -> tuple[Fraction, ...]:
    """Every acoustic step one outer step of ``dt`` takes (``woof.hex.integration.RKSchedule.from_mpas``).

    Exact rationals, so the comparison below is not a float comparison.
    """

    dt_dyn = Fraction(dt) / splits
    if order == 3:
        return (dt_dyn / 3, dt_dyn / substeps, dt_dyn / substeps)
    if order == 2:
        return (dt_dyn / substeps,) * 3
    raise ValueError(f"config_time_integration_order={order}: MPAS-A has only orders 2 and 3")


#: The substep count hex ran every bare forecast at before the 2026-10-11 re-pin to MPAS-A v8.4.2's four
#: (``woof.hex.mpas_defaults.PREVIOUS_ACOUSTIC_SUBSTEPS``, MPAS-A v8.4.1's Registry default).
REFERENCE_ACOUSTIC_SUBSTEPS = 2


def longest_acoustic_step(row_dt: Fraction, row_substeps: int, order: int) -> Fraction:
    """The longest acoustic step the adaptive step may take: what the mesh row's step was run at.

    The larger of the row's own longest acoustic step and the one the same row step takes with MPAS-A
    v8.4.1's two substeps, which every hex forecast ran before the 2026-10-11 re-pin (and MPAS-A ran as its
    default through v8.4.1).  At a 6 s/km row that is ``dx[km]`` seconds, an acoustic Courant number near
    0.34; WRF's own adaptive rule (``num_sound_steps = 2*(INT(300 dt/dx) + 1)``) admits up to about 0.57.
    """

    return max(max(acoustic_steps(row_dt, row_substeps, order)),
               max(acoustic_steps(row_dt, min(int(row_substeps), REFERENCE_ACOUSTIC_SUBSTEPS), order)))


def acoustic_ceiling(row_dt: Fraction, row_substeps: int, order: int) -> Fraction | None:
    """The largest step whose acoustic steps can all stay within :func:`longest_acoustic_step`; ``None`` = unbounded.

    Second order divides every stage into ``dt_dyn/n``, so a large enough ``n`` always holds the bound.  Third
    order takes stage one as one acoustic step of ``dt_dyn/3`` (MPAS ``number_sub_steps(1) = 1``), so the step
    may grow only until that one reaches the bound.
    """

    if order == 2:
        return None
    return longest_acoustic_step(row_dt, row_substeps, order) * 9


def acoustic_substeps(dt: Fraction, row_dt: Fraction, row_substeps: int, order: int) -> int:
    """``config_number_of_sub_steps`` for a step of ``dt``: WRF's adaptive ``time_step_sound`` on the mesh.

    The fewest substeps, never fewer than the row's and even when the row's is even (MPAS stage two takes
    ``n/2`` substeps for half the step, which is exact only for even ``n``), whose every acoustic step is no
    longer than :func:`longest_acoustic_step`.
    """

    longest = longest_acoustic_step(row_dt, row_substeps, order)
    n = int(row_substeps)
    while max(acoustic_steps(dt, n, order)) > longest:
        n += 2 if row_substeps % 2 == 0 else 1
        if n > 64 * max(1, row_substeps):
            raise ValueError(
                f"no acoustic substep count holds a {float(dt):g} s step within the longest acoustic "
                f"step ({float(longest):g} s) at time_integration_order={order}")
    return n


@dataclass(frozen=True)
class AdaptiveTimestepOptions:
    """The resolved ``--adaptive-*`` selection for one run."""

    mode: str
    target_cfl: float
    target_hcfl: float
    max_step_increase_pct: int
    starting_dt: Fraction
    min_dt: Fraction
    max_dt: Fraction
    row_dt: Fraction
    row_substeps: int
    order: int
    notes: tuple[str, ...] = ()

    def substeps(self, dt: Fraction) -> int:
        return acoustic_substeps(dt, self.row_dt, self.row_substeps, self.order)

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "mode": self.mode, "target_cfl": self.target_cfl,
            "target_hcfl": self.target_hcfl, "max_step_increase_pct": self.max_step_increase_pct,
            "starting_dt": str(self.starting_dt), "min_dt": str(self.min_dt), "max_dt": str(self.max_dt),
            "row_dt": str(self.row_dt), "row_acoustic_substeps": self.row_substeps,
            "time_integration_order": self.order, "precision": PRECISION,
            "step_to_output_time": True, "notes": list(self.notes),
        }


def resolve_options(arguments: Any, *, row_dt: float, row_substeps: int, order: int,
                    center_latitude: float | None) -> AdaptiveTimestepOptions | None:
    """The run's adaptive options from its arguments, or ``None`` with the mode off.

    Refuses by name: an incompatible option (:func:`refusal`), the tropical preset outside the tropics,
    clamps out of order, and a ``--adaptive-max-dt`` past the third-order acoustic ceiling.
    """

    mode = getattr(arguments, "adaptive_timestep", "off") or "off"
    if mode == "off":
        return None
    if mode not in MODES:
        raise ValueError(f"--adaptive-timestep {mode!r}: choose one of {', '.join(MODES)}")
    problem = refusal(arguments)
    if problem is not None:
        raise ValueError(problem)
    row = quantise(Fraction(str(float(row_dt))))
    notes: list[str] = []
    target_cfl = DEFAULT_TARGET_CFL
    floor_ratio = MIN_DT_RATIO
    if mode == "tropical":
        if center_latitude is None or not abs(float(center_latitude)) < TROPICAL_MAX_LAT:
            raise ValueError(
                f"--adaptive-timestep tropical is the tropics' opt-in (|lat| < {TROPICAL_MAX_LAT:g}); this mesh "
                f"is centred at {center_latitude!r}: use --adaptive-timestep on")
        target_cfl = TROPICAL_TARGET_CFL
        floor_ratio = Fraction(1)
        notes.append("tropical preset (woof domain --clock tropical-adaptive, #276): target_cfl 1.0, 5 % "
                     "growth, the row's fixed step as the floor; WRF's w_damping = 1 has no hex term yet "
                     "(parity item 4.2)")

    def seconds(name: str, default: Fraction) -> Fraction:
        value = getattr(arguments, name, None)
        if value is None:
            return default
        try:
            result = quantise(value)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(f"--{name.replace('_', '-')} {value}: {error}") from None
        if result <= 0:
            raise ValueError(f"--{name.replace('_', '-')} {value}: a step must be positive")
        return result

    starting = seconds("adaptive_starting_dt", row)
    max_dt = seconds("adaptive_max_dt", floor_to_precision(starting * MAX_DT_RATIO))
    min_dt = seconds("adaptive_min_dt", floor_to_precision(starting * floor_ratio))
    ceiling = acoustic_ceiling(row, int(row_substeps), int(order))
    if ceiling is not None and max_dt > ceiling:
        if getattr(arguments, "adaptive_max_dt", None) is not None:
            raise ValueError(
                f"--adaptive-max-dt {float(max_dt):g}: at time_integration_order=3 the first Runge-Kutta stage "
                f"takes one acoustic step of dt/9, which passes the row's longest acoustic step above "
                f"{float(ceiling):g} s; name a smaller maximum or run order 2")
        max_dt = floor_to_precision(ceiling)
        notes.append(f"max_dt capped at {float(max_dt):g} s: the third-order first stage's single acoustic "
                     "step (dt/9) may not pass the row's longest acoustic step")
    if not min_dt <= starting <= max_dt:
        raise ValueError(
            f"adaptive clamps out of order: min {float(min_dt):g} s, starting {float(starting):g} s, max "
            f"{float(max_dt):g} s (WRF applies max_time_step then min_time_step; hex refuses the inversion)")
    target_cfl = float(getattr(arguments, "adaptive_target_cfl", None) or target_cfl)
    target_hcfl = float(getattr(arguments, "adaptive_target_hcfl", None) or DEFAULT_TARGET_HCFL)
    pct = getattr(arguments, "adaptive_max_increase_pct", None)
    pct = DEFAULT_MAX_STEP_INCREASE_PCT if pct is None else int(pct)
    for name, value in (("--adaptive-target-cfl", target_cfl), ("--adaptive-target-hcfl", target_hcfl)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} {value}: a Courant target is finite and positive")
    if pct < 0:
        raise ValueError(f"--adaptive-max-increase-pct {pct}: the growth limit is not negative")
    return AdaptiveTimestepOptions(
        mode=mode, target_cfl=target_cfl, target_hcfl=target_hcfl, max_step_increase_pct=pct,
        starting_dt=starting, min_dt=min_dt, max_dt=max_dt, row_dt=row, row_substeps=int(row_substeps),
        order=int(order), notes=tuple(notes))


#: One subclass per configuration class, and the step that class's run was admitted at.
_ADAPTIVE_TYPES: dict[type, type] = {}
_ADMITTED: dict[type, tuple[float, int]] = {}


def _validate_admitted(self, *args: Any, **kwargs: Any) -> Any:
    import dataclasses

    base = type(self)._adaptive_base
    dt, substeps = _ADMITTED[base]
    values = {item.name: getattr(self, item.name) for item in dataclasses.fields(self) if item.init}
    return base(**{**values, "config_dt": dt, "config_number_of_sub_steps": substeps}).validate(*args, **kwargs)


def adaptive_step_config(config: Any, *, dt: float, substeps: int) -> Any:
    """``config`` at one adaptive step: the live ``config_dt`` and acoustic substeps, validated as admitted.

    The hex configuration's ``validate`` (re-run on every step's physics receipt) refuses a physics cadence
    that is not a whole number of steps, because on a fixed clock such a cadence has no step to run on.
    Under the adaptive clock every positive cadence is a deadline the clock lands on, so that rule belongs to
    the step the run was ADMITTED at: this returns an instance of a subclass whose ``validate`` validates the
    admitted configuration (the mesh row's step and substeps, every other field this step's), with the same
    fields and payload as the original class.
    """

    import dataclasses

    base = getattr(type(config), "_adaptive_base", type(config))
    if base not in _ADMITTED:
        if type(config) is not base:
            raise RuntimeError("an adaptive configuration must first be derived from the admitted one")
        _ADMITTED[base] = (float(config.config_dt), int(config.config_number_of_sub_steps))
    cls = _ADAPTIVE_TYPES.get(base)
    if cls is None:
        cls = type(f"AdaptiveStep{base.__name__}", (base,),
                   {"__slots__": (), "_adaptive_base": base, "validate": _validate_admitted,
                    "__module__": __name__})
        _ADAPTIVE_TYPES[base] = cls
    values = {item.name: getattr(config, item.name) for item in dataclasses.fields(config) if item.init}
    return cls(**{**values, "config_dt": float(dt), "config_number_of_sub_steps": int(substeps)})


@dataclass
class StepPlan:
    """One step as the clock decided it, before the model takes it."""

    step: int
    start: Fraction
    dt: Fraction
    proposed: Fraction
    stepping_to_time: bool
    vert_cfl: float
    horiz_cfl: float
    substeps: int

    @property
    def end(self) -> Fraction:
        return self.start + self.dt


class HexAdaptiveClock:
    """The root domain's adaptive clock: the WRF controller, exact time, deadlines.  Host-only and pure.

    The sequence per step is :meth:`woof.core.adaptive_clock.AdaptiveClockDriver.__call__`'s for a root
    with no nests: the CFL the last step measured, rescaled to the controller's own proposal when the step
    taken differed from it (ENG-017), then ``first_step``/``next_dt``, ``step_to_time`` to the next deadline
    with a lattice floor on a halved landing, and ``accept`` of the PROPOSAL.
    """

    def __init__(self, options: AdaptiveTimestepOptions, *, end: Fraction,
                 frames: Iterable[Fraction] = (), periods: Iterable[Fraction] = (), start: Fraction = Fraction(0)):
        self.options = options
        self.controller = AdaptiveTimestepController(
            target_cfl=options.target_cfl, target_hcfl=options.target_hcfl,
            max_step_increase_pct=options.max_step_increase_pct, starting_dt=options.starting_dt,
            min_dt=options.min_dt, max_dt=options.max_dt, precision=PRECISION)
        self.end = Fraction(end)
        self.frames = sorted({Fraction(value) for value in frames if 0 <= Fraction(value) <= self.end})
        self.periods = tuple(sorted({Fraction(value) for value in periods if Fraction(value) > 0}))
        self.elapsed = Fraction(start)
        self.step = 0
        #: The CFL pair the last committed step measured, ``None`` before the first.
        self.measured: tuple[float, float] | None = None
        #: The step the model last TOOK, for the CFL rescale.
        self.last_applied: Fraction | None = None
        self.pending: StepPlan | None = None
        self.series: list[dict[str, Any]] = []

    @property
    def finished(self) -> bool:
        return self.elapsed >= self.end

    def deadline(self) -> Fraction:
        """The next time after now that a step must land on exactly."""

        best = self.end
        for frame in self.frames:
            if frame > self.elapsed:
                best = min(best, frame)
                break
        for period in self.periods:
            due = (self.elapsed // period + 1) * period
            best = min(best, due)
        return best

    def propose(self) -> StepPlan:
        if self.pending is not None:
            raise RuntimeError("the adaptive clock already has a step in flight")
        if self.finished:
            raise RuntimeError("the adaptive clock is past the run end")
        ctl = self.controller
        vert, horiz = self.measured if self.measured is not None else (0.0, 0.0)
        applied = self.last_applied
        if (ctl.started and not ctl.stepping_to_time and applied is not None and applied > 0
                and applied != ctl.last_dt):
            scale = float(ctl.last_dt / applied)
            vert, horiz = vert * scale, horiz * scale
        self._started_before = ctl.started
        if not ctl.started:
            dt = ctl.first_step()
        else:
            dt = ctl.next_dt(max_vert_cfl=vert, max_horiz_cfl=horiz)
        dt, stepping = ctl.step_to_time(dt, self.deadline() - self.elapsed, quantise=floor_to_precision)
        proposed = dt
        dt = min(dt, floor_to_precision(dt)) if dt * PRECISION != int(dt * PRECISION) else dt
        self.step += 1
        self.pending = StepPlan(step=self.step, start=self.elapsed, dt=dt, proposed=proposed,
                                stepping_to_time=stepping, vert_cfl=float(vert), horiz_cfl=float(horiz),
                                substeps=self.options.substeps(dt))
        return self.pending

    def commit(self, *, vert_rate: float, horiz_rate: float) -> StepPlan:
        """Close the step in flight with the maximum ``|w|/dz`` and ``|u_n|/dcEdge`` it ended with (1/s)."""

        plan = self.pending
        if plan is None:
            raise RuntimeError("no adaptive step in flight")
        if not (math.isfinite(vert_rate) and math.isfinite(horiz_rate)):
            raise FloatingPointError(f"adaptive CFL probe read a non-finite wind at step {plan.step}")
        self.controller.accept(plan.proposed, max_vert_cfl=plan.vert_cfl, max_horiz_cfl=plan.horiz_cfl,
                               stepping_to_time=plan.stepping_to_time)
        self.elapsed = plan.end
        self.last_applied = plan.dt
        step_seconds = float(plan.dt)
        self.measured = (float(vert_rate) * step_seconds, float(horiz_rate) * step_seconds)
        self.series.append({"step": plan.step, "start": str(plan.start), "dt": str(plan.dt),
                            "substeps": plan.substeps, "stepping_to_time": plan.stepping_to_time,
                            "vert_cfl": self.measured[0], "horiz_cfl": self.measured[1]})
        self.pending = None
        return plan

    def abandon(self) -> None:
        """A refused step: nothing was committed, so neither the clock nor the controller moves."""

        if self.pending is not None:
            self.step -= 1
            self.pending = None
            # first_step is the controller's one side effect before accept.
            self.controller.started = self._started_before

    def state(self) -> dict[str, Any]:
        """Everything the next step depends on, for a checkpoint (both memories, as the WRF route stores)."""

        if self.pending is not None:
            raise RuntimeError("an adaptive checkpoint is taken between steps")
        ctl = self.controller
        return {
            "schema": SCHEMA, "options": self.options.receipt(), "elapsed": str(self.elapsed), "step": self.step,
            "measured": None if self.measured is None else list(self.measured),
            "last_applied": None if self.last_applied is None else str(self.last_applied),
            "controller": {"last_dt": str(ctl.last_dt), "last_max_vert_cfl": ctl.last_max_vert_cfl,
                           "last_max_horiz_cfl": ctl.last_max_horiz_cfl,
                           "stepping_to_time": ctl.stepping_to_time, "started": ctl.started},
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema") != SCHEMA:
            raise ValueError("this checkpoint carries no adaptive clock state")
        if payload.get("options") != self.options.receipt():
            raise ValueError("the checkpoint's adaptive options differ from this run's: "
                             f"{payload.get('options')} != {self.options.receipt()}")
        ctl = self.controller
        memory = payload["controller"]
        ctl.last_dt = Fraction(memory["last_dt"])
        ctl.last_max_vert_cfl = float(memory["last_max_vert_cfl"])
        ctl.last_max_horiz_cfl = float(memory["last_max_horiz_cfl"])
        ctl.stepping_to_time = bool(memory["stepping_to_time"])
        ctl.started = bool(memory["started"])
        self.elapsed = Fraction(payload["elapsed"])
        self.step = int(payload["step"])
        measured = payload.get("measured")
        self.measured = None if measured is None else (float(measured[0]), float(measured[1]))
        last = payload.get("last_applied")
        self.last_applied = None if last is None else Fraction(last)
        self.pending = None

    def summary(self) -> dict[str, Any]:
        dts = [Fraction(row["dt"]) for row in self.series]
        if not dts:
            return {"steps": 0}
        return {"steps": len(dts), "elapsed": str(self.elapsed), "min_dt": float(min(dts)),
                "max_dt": float(max(dts)), "mean_dt": float(sum(dts) / len(dts)),
                "landings": sum(1 for row in self.series if row["stepping_to_time"]),
                "max_vert_cfl": max(row["vert_cfl"] for row in self.series),
                "max_horiz_cfl": max(row["horiz_cfl"] for row in self.series),
                "substeps": sorted({row["substeps"] for row in self.series})}


class MeshCflProbe:
    """The two Courant reductions on the device, one host read per step.

    Horizontal: the edge Courant number ``|u_n| dt / dcEdge`` over owned edges and levels, from the step's
    end-of-step normal velocity.  Vertical: WRF's form, the COORDINATE vertical velocity over the coordinate
    layer depth (WRF reduces ``|ww/mu| rdnw dt``, its eta-dot), on the interior w levels.  On the MPAS
    terrain-following height coordinate that is ``|ww| / rho_zz(face) * rdzu * dt``, ``ww`` the vertical mass
    flux the scalar transport actually moved (``split_flux_w``), observed on every RK stage of the step.  The
    geometric ``|w| dt / dz`` is NOT this number over steep terrain: there the physical w carries the flow
    along the coordinate surfaces (measured on the hexcase at its 30 s row step: 1.27 geometric at the first
    interface above the ground, where nothing crosses a layer).
    """

    def __init__(self, cp: Any, *, dc_edge: Any, rdzu: Any, fzm: Any, fzp: Any, n_cells: int, n_edges: int):
        import numpy as np

        dc = np.asarray(dc_edge, dtype=np.float64).reshape(-1)[:n_edges]
        if dc.size != n_edges or not np.all(np.isfinite(dc)) or float(dc.min()) <= 1.0:
            raise ValueError("the CFL probe needs dcEdge in metres for every owned edge")
        rdzu = np.asarray(rdzu, dtype=np.float64).reshape(-1)
        if rdzu.size < 2 or not np.all(np.isfinite(rdzu[1:])) or not np.all(rdzu[1:] > 0.0):
            raise ValueError("the CFL probe needs a positive rdzu on every interior w level")
        self.cp = cp
        self.n_cells, self.n_edges = int(n_cells), int(n_edges)
        self.n_levels = int(rdzu.size)
        self.inv_dc = cp.asarray((1.0 / dc).astype(np.float32))
        self.rdzu = cp.asarray(rdzu[1:].astype(np.float32))[:, None]
        self.fzm = cp.asarray(np.asarray(fzm, dtype=np.float32).reshape(-1)[1:])[:, None]
        self.fzp = cp.asarray(np.asarray(fzp, dtype=np.float32).reshape(-1)[1:])[:, None]
        self.vertical_max = cp.zeros((), dtype=cp.float32)
        self.observed = 0

    def observe(self, rho: Any, ww: Any) -> None:
        """One scalar-transport call's vertical rate, folded into the step's maximum (no host read)."""

        cp = self.cp
        n, nk = self.n_cells, self.n_levels
        face_rho = self.fzm * rho[1:nk, :n] + self.fzp * rho[:nk - 1, :n]
        rate = cp.max(cp.abs(ww[1:nk, :n]) / face_rho * self.rdzu)
        cp.maximum(self.vertical_max, rate.astype(cp.float32), out=self.vertical_max)
        self.observed += 1

    def attach(self, driver: Any) -> None:
        """Watch the scalar transport's fluxes on this driver only (as the full-step LTS shadow does)."""

        import types

        original = driver._step_device_v841.__func__
        native = original.__globals__["advance_scalar_transport_cuda_v841"]
        probe = self

        def observed(*args: Any, **kwargs: Any) -> Any:
            # cuda_driver: (mesh, context, scalars, stage scalars, source-old rho, rho, flux_u, flux_w, dt, ...)
            probe.observe(args[4], args[7])
            return native(*args, **kwargs)

        private_globals = dict(original.__globals__)
        private_globals["advance_scalar_transport_cuda_v841"] = observed
        instrumented = types.FunctionType(original.__code__, private_globals, original.__name__,
                                          original.__defaults__, original.__closure__)
        instrumented.__kwdefaults__ = original.__kwdefaults__
        driver._step_device_v841 = types.MethodType(instrumented, driver)

    def rates(self, *, normal_velocity: Any) -> tuple[float, float]:
        cp = self.cp
        if not self.observed:
            raise RuntimeError("the adaptive CFL probe saw no scalar transport this step")
        u = normal_velocity[:, :self.n_edges]
        horiz = cp.max(cp.abs(u) * self.inv_dc[None, :]).astype(cp.float32)
        values = cp.asnumpy(cp.stack((self.vertical_max, horiz)))
        self.vertical_max[...] = 0
        self.observed = 0
        return float(values[0]), float(values[1])

    def receipt(self) -> dict[str, Any]:
        return {"horizontal": "max |u_n| dt / dcEdge over owned edges and levels, end of step",
                "vertical": ("max |ww| / rho_zz(face) * rdzu * dt over owned cells and interior w levels "
                             "(WRF's k = 2 .. kde-1), ww the scalar transport's vertical mass flux on every RK "
                             "stage: the coordinate form WRF's eta-dot CFL measures, not geometric |w|/dz")}


@dataclass
class PhysicsCadence:
    """When the column physics fires under a varying step: exact deadlines in model seconds.

    ``None`` for a cadence welded to the step (it runs every step at the live dt); otherwise its period.
    """

    radiation: Fraction | None
    surface_pbl: Fraction | None
    cumulus: Fraction | None

    def periods(self) -> tuple[Fraction, ...]:
        return tuple(value for value in (self.radiation, self.surface_pbl, self.cumulus) if value)

    def due(self, elapsed: Fraction) -> dict[str, bool | None]:
        """Each cadence at a step starting ``elapsed`` seconds into the run: due, not due, or ``None`` (welded).

        Radiation is never welded: a radiation period equal to the step fires every step (``True``).
        """

        def at(period: Fraction | None) -> bool | None:
            return None if period is None else Fraction(elapsed) % period == 0
        radiation = at(self.radiation)
        return {"radiation": True if radiation is None else radiation, "surface_pbl": at(self.surface_pbl),
                "cumulus": at(self.cumulus)}


def physics_cadence(*, row_dt: Fraction, radiation_seconds: float | None, surface_pbl_seconds: float | None,
                    cumulus_seconds: float | None) -> PhysicsCadence:
    """The frozen column batch's three cadences as adaptive deadlines.

    A cadence equal to the row's step is the weld (``bldt``/``cudt`` = dt, GF's ``cudt = 0``): every step.
    """

    def period(seconds: float | None) -> Fraction | None:
        if seconds is None or float(seconds) <= 0.0:
            return None
        value = Fraction(str(float(seconds)))
        return None if value == row_dt else quantise(value)

    return PhysicsCadence(radiation=period(radiation_seconds), surface_pbl=period(surface_pbl_seconds),
                          cumulus=period(cumulus_seconds))


def history_frames(*, start_step: int, capture_steps: Sequence[int], row_dt: float) -> list[Fraction]:
    """A fixed-step schedule's frames as exact seconds (the frames do not move when the step does)."""

    row = Fraction(str(float(row_dt)))
    return [step * row for step in capture_steps if step >= start_step]


class AdaptiveRun:
    """The forecast loop's handle on the adaptive clock: rebinding, landing, measuring, recording.

    The loop calls :meth:`steps` in place of ``range``; each step is decided (and the dycore, its regional
    boundary runtime and the physics seam rebound to it) before the step number is handed out, so a frame or
    checkpoint the step lands on is registered before the loop body asks.  :meth:`after_step` closes the step
    with the CFL it ended with; :meth:`abandon` drops a refused one.
    """

    def __init__(self, options: AdaptiveTimestepOptions, *, stack: Mapping[str, Any], end: Fraction,
                 frames: Iterable[Fraction], checkpoints: Iterable[Fraction], cadence: PhysicsCadence,
                 start_text: str, mesh: Any, vertical: Any, n_cells: int, n_edges: int,
                 lbc_interval: Fraction | None = None, start: Fraction = Fraction(0)):
        backend = stack["backend"]
        if not callable(getattr(backend, "set_adaptive_step", None)):
            raise ValueError(
                f"--adaptive-timestep: the physics seam {type(backend).__name__} takes no adaptive step; the WOOF "
                "column route's frozen batch (arwen_mpas_column, arwen_thompson_column) carries one")
        driver = stack["driver"]
        self.stack = stack
        self.options = options
        self.cadence = cadence
        self.frames = {Fraction(value) for value in frames}
        self.checkpoints = {Fraction(value) for value in checkpoints}
        periods = list(cadence.periods())
        if lbc_interval:
            periods.append(Fraction(lbc_interval))
        self.clock = HexAdaptiveClock(options, end=end, frames=self.frames | self.checkpoints, periods=periods,
                                      start=start)
        from datetime import datetime
        self.start_time = datetime.strptime(start_text, "%Y-%m-%d_%H:%M:%S")
        self.probe = MeshCflProbe(driver.cp, dc_edge=mesh.dcEdge, rdzu=vertical.rdzu, fzm=vertical.fzm,
                                  fzp=vertical.fzp, n_cells=int(n_cells), n_edges=int(n_edges))
        self.probe.attach(driver)
        #: The dycore's own model time (binary64, as it sums the steps) at the end of each step, for the
        #: snapshot and checkpoint clock checks; the exact time is the clock's.
        self.model_time: dict[int, float] = {}
        self.exact_time: dict[int, Fraction] = {}
        self.final_step: int | None = None
        self.plan: StepPlan | None = None

    # -- the loop's side ---------------------------------------------------------------------------------------
    def steps(self, start_step: int, *, register: Any):
        """Yield step numbers until the run end; ``register(step, seconds, checkpoint)`` names a landing."""

        self.clock.step = int(start_step)
        while not self.clock.finished:
            plan = self.clock.propose()
            self.plan = plan
            self._rebind(plan)
            end = plan.end
            if end in self.frames or end in self.checkpoints:
                register(plan.step, end, end in self.checkpoints)
            if end >= self.clock.end:
                self.final_step = plan.step
            yield plan.step
            if self.clock.pending is not None:
                # The body ended without closing the step (a refusal broke the loop elsewhere).
                self.clock.abandon()
                return

    def _rebind(self, plan: StepPlan) -> None:
        import numpy as np

        driver, backend = self.stack["driver"], self.stack["backend"]
        driver.readmit_adaptive_step(adaptive_step_config(driver.config, dt=float(plan.dt),
                                                          substeps=int(plan.substeps)))
        regional = getattr(driver, "regional_v841", None)
        if regional is not None and hasattr(regional, "outer_dt"):
            regional.outer_dt = float(plan.dt)
        # The YSU-GWDO guard demands the run's step bit for bit; the kernel takes dt as an argument and is
        # dt-general (woof.hex.drivers.mpas_mesh_binding rebinds the same constant to the row's step, and the
        # full-step LTS executor to each class's).
        from woof.hex import cuda_gwdo_v841
        cuda_gwdo_v841.X4_DT_SECONDS_F32 = np.float32(float(plan.dt))
        start = float(driver.atmosphere.state.time_seconds)
        backend.set_adaptive_step(dt=float(plan.dt), start_seconds=start, end_seconds=start + float(plan.dt),
                                  due=self.cadence.due(plan.start))

    def after_step(self, step: int) -> None:
        driver = self.stack["driver"]
        plan = self.clock.pending
        if plan is None or plan.step != step:
            raise RuntimeError(f"adaptive step {step} closed out of order")
        saved = driver.atmosphere.saved
        vert, horiz = self.probe.rates(normal_velocity=saved.normal_velocity)
        self.clock.commit(vert_rate=vert, horiz_rate=horiz)
        held = float(driver.atmosphere.state.time_seconds)
        if abs(held - float(plan.end)) > 1.0e-6 * max(1.0, float(plan.end)):
            raise RuntimeError(f"the dycore clock {held!r} s left the adaptive clock {plan.end} s at step {step}")
        self.model_time[step] = held
        self.exact_time[step] = plan.end
        self.plan = None

    def abandon(self) -> None:
        self.clock.abandon()
        self.plan = None

    def dt(self) -> float:
        return float(self.plan.dt) if self.plan is not None else float(self.options.starting_dt)

    def seconds_at(self, step: int) -> float:
        """The model seconds at the end of ``step`` (0 at the start; the planned end for the step in flight)."""

        if step in self.model_time:
            return self.model_time[step]
        if self.plan is not None and self.plan.step == step:
            return float(self.plan.end)
        return float(self.exact_time.get(step, self.clock.elapsed))

    def label(self, seconds: Fraction) -> tuple[str, str]:
        from datetime import timedelta

        valid = self.start_time + timedelta(seconds=float(seconds))
        return valid.strftime("%Y-%m-%d_%H.%M.%S"), valid.strftime("%Y-%m-%d_%H:%M:%S")

    def write_series(self, path: Any) -> None:
        import json
        from pathlib import Path

        with Path(path).open("w", encoding="utf-8") as stream:
            for row in self.clock.series:
                stream.write(json.dumps(row, sort_keys=True) + "\n")

    def receipt(self) -> dict[str, Any]:
        return {**self.options.receipt(), "summary": self.clock.summary(), "cfl": self.probe.receipt(),
                "cadence": {name: (None if value is None else str(value)) for name, value in
                            (("radiation", self.cadence.radiation), ("surface_pbl", self.cadence.surface_pbl),
                             ("cumulus", self.cadence.cumulus))}}
