"""Typed spectral prognostic states and arithmetic helpers."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


@dataclass
class ShallowWaterState:
    vorticity: object
    divergence: object
    geopotential: object
    time_s: float = 0.0
    step: int = 0

    def fields(self) -> tuple[object, ...]:
        return self.vorticity, self.divergence, self.geopotential

    def with_fields(self, fields: Iterable[object], *, time_s: float | None = None, step: int | None = None):
        z, d, p = tuple(fields)
        return replace(
            self,
            vorticity=z,
            divergence=d,
            geopotential=p,
            time_s=self.time_s if time_s is None else float(time_s),
            step=self.step if step is None else int(step),
        )


@dataclass
class PrimitiveDryState:
    vorticity: object
    divergence: object
    temperature: object
    log_surface_pressure: object
    time_s: float = 0.0
    step: int = 0

    def fields(self) -> tuple[object, ...]:
        return (
            self.vorticity,
            self.divergence,
            self.temperature,
            self.log_surface_pressure,
        )

    def with_fields(self, fields: Iterable[object], *, time_s: float | None = None, step: int | None = None):
        z, d, t, p = tuple(fields)
        return replace(
            self,
            vorticity=z,
            divergence=d,
            temperature=t,
            log_surface_pressure=p,
            time_s=self.time_s if time_s is None else float(time_s),
            step=self.step if step is None else int(step),
        )


def state_add(state, tendency, factor: float, *, time_s: float | None = None):
    fields = [a + factor * b for a, b in zip(state.fields(), tendency.fields())]
    return state.with_fields(fields, time_s=time_s)


def state_linear_combination(template, terms: list[tuple[float, object]], *, time_s: float | None = None):
    fields = []
    for index in range(len(template.fields())):
        value = None
        for weight, state in terms:
            term = weight * state.fields()[index]
            value = term if value is None else value + term
        fields.append(value)
    return template.with_fields(fields, time_s=time_s)
