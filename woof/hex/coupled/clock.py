"""The shared clock for a coupled run.

One wall clock, several timesteps. Each component advances on its own timestep
by subcycling; the flux exchange fires on the coupling cadence. The cadence
must be a multiple of every component's timestep, or an exchange would land
mid-substep and the subcycle order is ambiguous; the run length must be a
multiple of the cadence, or the last exchange is cut short.

This module is the schedule only. It knows the start, end, cadence and each
component's timestep, and yields the coupling intervals and how many substeps
each component takes in one. It does not advance a component; the run loop that
the component packages supply does the stepping and the exchange, and this clock
tells it when.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class CoupledInterval:
    """One coupling interval: the wall-clock window over which a component subcycles before an exchange."""

    index: int
    start: float
    end: float


@dataclass(frozen=True)
class CoupledClock:
    start: float
    end: float
    cadence: float
    dts: dict[str, float]

    def __post_init__(self) -> None:
        if not self.dts:
            raise ValueError("a coupled clock needs at least one component timestep")
        if self.end <= self.start:
            raise ValueError(f"the end ({self.end}s) must be after the start ({self.start}s)")
        if self.cadence <= 0.0:
            raise ValueError(f"the cadence ({self.cadence}s) must be positive")
        span = self.end - self.start
        if abs(span / self.cadence - round(span / self.cadence)) > 1e-9:
            raise ValueError(
                f"the run length {span}s is not a multiple of the cadence {self.cadence}s; "
                "the last exchange would be cut short")
        for name, dt in self.dts.items():
            if dt <= 0.0:
                raise ValueError(f"the {name} timestep ({dt}s) must be positive")
            if abs(self.cadence / dt - round(self.cadence / dt)) > 1e-9:
                raise ValueError(
                    f"the cadence {self.cadence}s is not a multiple of the {name} timestep "
                    f"{dt}s; an exchange would land mid-substep")

    def intervals(self) -> Iterator[CoupledInterval]:
        count = int(round((self.end - self.start) / self.cadence))
        for i in range(count):
            t0 = self.start + i * self.cadence
            yield CoupledInterval(index=i, start=t0, end=t0 + self.cadence)

    def substeps(self, name: str) -> int:
        try:
            dt = self.dts[name]
        except KeyError:
            raise ValueError(f"{name} is not a component of this clock") from None
        return int(round(self.cadence / dt))

    def receipt(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "cadence": self.cadence,
            "intervals": int(round((self.end - self.start) / self.cadence)),
            "components": {
                name: {
                    "timestep": dt,
                    "substeps_per_cadence": self.substeps(name),
                }
                for name, dt in self.dts.items()
            },
        }
