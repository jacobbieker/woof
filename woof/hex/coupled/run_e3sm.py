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
    """The ocean's and sea ice's coupling periods in coupler steps."""

    ocean: int
    seaice: int

    @classmethod
    def from_clock(cls, clock: CoupledClock, *, ocean_period: float, seaice_period: float) -> Alarms:
        """From coupling periods in seconds, each a whole number of the clock's cadence."""
        periods = {}
        for name, seconds in (("ocean", ocean_period), ("seaice", seaice_period)):
            ratio = seconds / clock.cadence
            if seconds <= 0.0 or abs(ratio - round(ratio)) > 1e-9:
                raise ValueError(f"the {name} coupling period {seconds}s is not a whole number of coupler steps "
                                 f"of {clock.cadence}s")
            periods[name] = int(round(ratio))
        return cls(**periods)

    def at(self, step: int) -> tuple[bool, bool, bool]:
        """The ocean's run alarm, the sea ice's, and the ocean's next alarm at coupler step ``step``."""
        k, m = self.ocean, self.seaice
        return step % k == 0, step % m == 0, step % k == k - 1


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
