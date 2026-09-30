"""Exact exponential hyperdiffusion in total spherical wavenumber."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .transform import SphericalHarmonicTransform


@dataclass(frozen=True)
class ExponentialHyperdiffusion:
    order: int = 4
    e_folding_time_s_at_truncation: float = 21_600.0
    preserve_degree: int = 1

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("hyperdiffusion order must be >= 1")
        if not math.isfinite(self.e_folding_time_s_at_truncation) or self.e_folding_time_s_at_truncation <= 0:
            raise ValueError("e-folding time must be finite and positive")
        if self.preserve_degree < 0:
            raise ValueError("preserve_degree must be >= 0")

    def factors(self, transform: SphericalHarmonicTransform, dt_s: float):
        xp = transform.backend.xp
        n = np.arange(transform.truncation + 1, dtype=np.float64)
        lam = n * (n + 1.0)
        top = max(1.0, lam[-1])
        exponent = -(float(dt_s) / self.e_folding_time_s_at_truncation) * (lam / top) ** self.order
        factor = np.exp(exponent)
        factor[: self.preserve_degree + 1] = 1.0
        return xp.asarray(factor, dtype=transform.backend.float_dtype)

    def apply(self, coeff, transform: SphericalHarmonicTransform, dt_s: float, *, strength: float = 1.0):
        if strength < 0 or not math.isfinite(strength):
            raise ValueError("diffusion strength must be finite and nonnegative")
        factor = self.factors(transform, dt_s) ** float(strength)
        return transform.project(coeff * factor[:, None])
