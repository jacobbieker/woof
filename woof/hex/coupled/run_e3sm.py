"""The E3SM coupled step on the coupling spine, opt-in.

:func:`run` drives a :class:`.interval_e3sm.CouplerInterval` over a
:class:`.clock.CoupledClock`: one coupler step per clock interval (the
clock's cadence is the coupler's time step), with the ocean's and sea ice's
run alarms cut from their coupling periods as ``cime_comp_mod``'s
RASM_OPTION1 loop sees them. In each step the components run where E3SM
runs them:

- the atmosphere's export and this step's inputs arrive first
  (:meth:`Components.atmosphere`, :meth:`Components.inputs`);
- ``cime_run_atmocn_setup``;
- on the ocean's run alarm, ``cime_run_ocn_setup_send``: the window's
  averaged x2o goes to the ocean, which starts its run
  (:meth:`Components.ocean_start`);
- on the sea ice's run alarm, ``cime_run_ice_setup_send``: x2i goes to the
  sea ice, which runs and returns its export (:meth:`Components.seaice`),
  then ``cime_run_ice_recv_post``;
- ``cime_run_update_fractions``;
- on the ocean's next alarm (its window's last coupler step), the ocean's
  run ends and its export comes back (:meth:`Components.ocean_finish`).

Every hand-over is a matched exchange (:func:`.exchange.check_matched`):
a component that returns what it received has it compared, word for word,
with what the coupler sent, and the four coupler vectors are channels of
their own (:data:`CHANNELS`) so a mismatch is refused by name.

Nothing on the default path calls this. The coupler's flux needs the glibc
opt-in (``WOOF_GLIBC239_LIBM64=1``), as :mod:`.interval_e3sm` does. It is
held to the chained Fortran oracle (``tests/test_e3sm_coupled_spine.py``):
the oracle's cases, driven through the spine with components scripted
from the cases, give the oracle's words.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .clock import CoupledClock
from .exchange import FluxChannel, check_matched

#: the coupler's own exchange with the ocean and sea ice (whole MCT vectors, the coupler's units)
CHANNELS = {
    "x2o": FluxChannel("cpl2ocn_x2o", "coupler", "ocean", "the ocean's import, averaged over its window", "x2o"),
    "o2x": FluxChannel("ocn2cpl_o2x", "ocean", "coupler", "the ocean's export", "o2x"),
    "x2i": FluxChannel("cpl2si_x2i", "coupler", "seaice", "the sea ice's import", "x2i"),
    "i2x": FluxChannel("si2cpl_i2x", "seaice", "coupler", "the sea ice's export", "i2x"),
}


class Components(Protocol):
    """What the spine asks the components for. A method that returns a pair
    returns ``(export, received)``; ``received`` (or ``None``) is the
    import as the component holds it, checked against what was sent."""

    def atmosphere(self, step: int):
        """The atmosphere's export on the ocean's grid for coupler step ``step``."""

    def inputs(self, step: int) -> dict | None:
        """This step's :meth:`.CouplerInterval.set_inputs`, or ``None``."""

    def ocean_start(self, step: int, x2o):
        """The ocean's run starts with ``x2o``; returns what it received (or ``None``)."""

    def ocean_finish(self, step: int):
        """The ocean's run ends: its new export."""

    def seaice(self, step: int, x2i):
        """The sea ice runs with ``x2i``: returns ``(i2x, received)``."""


@dataclass(frozen=True)
class Alarms:
    """The driver's ocean and sea-ice alarms, as E3SM v3.2.0 sets them up.

    ``ocean`` and ``seaice`` are the coupling periods (``ocn_cpl_dt``,
    ``ice_cpl_dt``) in coupler steps; ``ocean_offset`` and ``seaice_offset``
    the namelist's ``ocn_cpl_offset`` and ``ice_cpl_offset`` in coupler
    steps; ``startup`` is a run that does not read a restart.

    ``driver-mct/shr/seq_timemgr_mod.F90``: each offset is the namelist's
    (1056-1057) plus one driver step (1074), at most the period (1067-1070)
    and a multiple of the driver step (1076-1079). The sea ice's run alarm rings every
    ``ice_cpl_dt`` from ``CurrTime + offset(ice)`` (1126-1133), the ocean's
    every ``ocn_cpl_dt`` from ``CurrTime + offset(ocn)`` (1194-1201), and
    the ocean's next alarm is that one a driver step earlier (1203-1213).
    ``seq_timemgr_alarmInit`` moves the first ring to the first time after
    ``CurrTime`` on that cadence (1737-1741). ``cime_comp_mod.F90`` advances
    the clock before it reads the alarms (2848, 2855-2860), so coupler step
    ``s`` (from 0) sees the time ``start + (s + 1) * dt``: the ocean runs at
    ``(s - ocean_offset) % ocean == 0``, ends at
    ``(s + 1 - ocean_offset) % ocean == 0`` and the sea ice runs at
    ``(s - seaice_offset) % seaice == 0``.

    The ocean's lag on a startup: ``skip_ocean_run`` is set unless the run
    reads a restart (cime_comp_mod 1869-1873) and holds both ocean alarms
    off until the run alarm has rung a second time (2948-2957). The
    ocean's first window then runs on, unaveraged, to that second ring.
    """

    ocean: int
    seaice: int
    ocean_offset: int = 0
    seaice_offset: int = 0
    startup: bool = False

    def __post_init__(self) -> None:
        for name in ("ocean", "seaice"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"the {name} coupling period must be a positive whole number of coupler steps")
        for name, period in (("ocean_offset", self.ocean), ("seaice_offset", self.seaice)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be a whole number of coupler steps")
            if abs(value) > period:
                raise ValueError(f"{name} ({value}) is larger than its coupling period ({period}): "
                                 "seq_timemgr_clockInit stops ('offset too large')")
        if not isinstance(self.startup, (bool, np.bool_)):
            raise TypeError("startup must be a bool")

    @classmethod
    def from_clock(cls, clock: CoupledClock, *, ocean_period: float, seaice_period: float,
                   ocean_offset: float = 0.0, seaice_offset: float = 0.0, startup: bool = False) -> Alarms:
        """From coupling periods and offsets in seconds, each a whole number of the clock's cadence."""
        steps = {}
        for name, seconds, positive in (("ocean", ocean_period, True), ("seaice", seaice_period, True),
                                        ("ocean_offset", ocean_offset, False),
                                        ("seaice_offset", seaice_offset, False)):
            ratio = seconds / clock.cadence
            if (positive and seconds <= 0.0) or abs(ratio - round(ratio)) > 1e-9:
                label = f"the {name} coupling period" if positive else name
                raise ValueError(f"{label} {seconds}s is not a whole number of coupler steps of {clock.cadence}s")
            steps[name] = int(round(ratio))
        return cls(**steps, startup=startup)

    def at(self, step: int) -> tuple[bool, bool, bool]:
        """The ocean's run alarm, the sea ice's, and the ocean's next alarm at coupler step ``step``."""
        k, m = self.ocean, self.seaice
        ocn_run = (step - self.ocean_offset) % k == 0
        ocn_next = (step + 1 - self.ocean_offset) % k == 0
        ice_run = (step - self.seaice_offset) % m == 0
        if self.startup:
            # ocnrun_count after this step's increment: the run alarm's rings at steps 0..step
            first = self.ocean_offset % k
            rings = 0 if step < first else (step - first) // k + 1
            if rings <= 1:
                ocn_run = ocn_next = False
        return ocn_run, ice_run, ocn_next


def _host(array) -> np.ndarray:
    import cupy as cp
    return cp.asnumpy(array) if not isinstance(array, np.ndarray) else array


def _matched(name: str, sent, received) -> None:
    if received is not None:
        check_matched(CHANNELS[name], _host(sent), _host(received))


def run(clock: CoupledClock, interval, components: Components, *, alarms: Alarms, on_step=None) -> int:
    """Every coupler step of ``clock`` on ``interval``; returns the steps run.
    ``on_step(step, interval)``, if given, is called after each."""
    import cupy as cp
    from .interval_e3sm import CouplerInterval
    if not isinstance(interval, CouplerInterval):
        raise TypeError("interval must be an interval_e3sm.CouplerInterval")
    if not isinstance(alarms, Alarms):
        raise TypeError("alarms must be a run_e3sm.Alarms")
    steps = 0
    for window in clock.intervals():
        s = window.index
        ocn_run, ice_run, ocn_next = alarms.at(s)
        inputs = components.inputs(s)
        if inputs:
            interval.set_inputs(**inputs)
        interval.atmocn_setup(components.atmosphere(s))
        if ocn_run:
            interval.ocn_setup_send()
            x2o = interval.vector("x2o")
            _matched("x2o", x2o, components.ocean_start(s, cp.array(x2o, copy=True)))
        if ice_run:
            interval.ice_setup_send()
            x2i = interval.vector("x2i")
            i2x, received = components.seaice(s, cp.array(x2i, copy=True))
            _matched("x2i", x2i, received)
            interval.ice_recv_post(i2x)
            _matched("i2x", i2x, interval.vector("i2x"))
        interval.update_fractions()
        if ocn_next:
            o2x = components.ocean_finish(s)
            interval.ocn_recv_post(o2x)
            _matched("o2x", o2x, interval.vector("o2x"))
        steps += 1
        if on_step is not None:
            on_step(s, interval)
    return steps
