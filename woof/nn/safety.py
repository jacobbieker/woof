"""Constraints between a column model and the model state it writes into.

Plan section 4.4.  Each function works on NumPy, CuPy or facade arrays
alike (``[feature or level, column]``) and launches only array operations,
no new kernels.  A seat composes them through :class:`SafetyPolicy`:

* **Trained range.**  ``out_of_range_columns`` flags a column any of whose
  features lies outside ``[input_min, input_max]`` widened by ``margin`` of
  the range.  The policy then ``refuse``s (raises
  :class:`InputOutOfRange`, naming the count and the worst feature),
  ``fallback``s (those columns keep the physics scheme's tendency in a
  replacement, or get no NN increment in a residual) or ``clip``s the
  inputs into range.  Counts are accumulated on the device and read only
  when asked (:meth:`RangeLedger.summary`), so a fallback does not
  synchronise the stream every step.
* **Output limits.**  ``limit_outputs`` clamps each output feature to
  ``|y| <= output_limit`` (the manifest's, typically a multiple of the
  largest training target).
* **Positivity.**  ``positive_tendency`` keeps ``q + dt * dq >= 0``.
* **Column budget.**  ``column_fix`` adds a mass-weighted uniform
  correction so that ``sum_k w_k dq_k`` equals a target per column -- the
  replaced scheme's own column integral, or zero for a residual -- the
  "column fixer" of the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from woof.nn.bridge import array_module

OUT_OF_RANGE = ("fallback", "refuse", "clip")
FIXERS = ("none", "moisture", "moisture+heat")


class InputOutOfRange(RuntimeError):
    """Columns fell outside the trained input range under ``refuse``."""


def _bounds(xp, lo, hi, margin: float):
    lo = xp.asarray(np.asarray(lo, dtype=np.float32))[:, None]
    hi = xp.asarray(np.asarray(hi, dtype=np.float32))[:, None]
    if margin:
        pad = (hi - lo) * np.float32(margin)
        lo, hi = lo - pad, hi + pad
    return lo, hi


def out_of_range_columns(x: Any, lo, hi, margin: float = 0.0):
    """Boolean ``[column]``: a feature of the column lies outside the range."""
    xp = array_module(x)
    lo, hi = _bounds(xp, lo, hi, margin)
    bad = (x < lo) | (x > hi) | ~xp.isfinite(x)
    return bad.any(axis=0)


def worst_feature(x: Any, lo, hi, margin: float = 0.0) -> tuple[int, int]:
    """``(feature index, columns outside)`` for the feature most often out
    of range; a host read, used only to word a refusal."""
    xp = array_module(x)
    lo, hi = _bounds(xp, lo, hi, margin)
    counts = ((x < lo) | (x > hi) | ~xp.isfinite(x)).sum(axis=1)
    counts = counts.get() if hasattr(counts, "get") else np.asarray(counts)
    index = int(np.argmax(counts))
    return index, int(counts[index])


def clip_inputs(x: Any, lo, hi, margin: float = 0.0):
    xp = array_module(x)
    lo, hi = _bounds(xp, lo, hi, margin)
    return xp.minimum(xp.maximum(x, lo), hi)


def limit_outputs(y: Any, limit):
    if limit is None:
        return y
    xp = array_module(y)
    bound = xp.asarray(np.asarray(limit, dtype=np.float32))[:, None]
    return xp.minimum(xp.maximum(y, -bound), bound)


def positive_tendency(dq: Any, q: Any, dt: float):
    """``dq`` raised where needed so that ``q + dt * dq >= min(q, 0)``.

    The model may not take moisture below zero, and where ``q`` is already
    negative (an advection undershoot) it may not deepen it; it is not
    asked to repair it either, which is the moisture fixer's job, not a
    column model's."""
    xp = array_module(dq)
    floor = -xp.maximum(q, np.float32(0)) / np.float32(dt)
    return xp.maximum(dq, floor)


def column_fix(d: Any, weight: Any, target: Any = None):
    """Shift ``d`` (``[level, column]``) uniformly in each column so that
    ``sum_k weight_k d_k`` equals ``target`` (``[column]``, zero if None).
    ``weight`` is the layer mass per unit area (dp/g); a column with no
    mass is left as it is."""
    xp = array_module(d)
    total = (weight * d).sum(axis=0)
    mass = weight.sum(axis=0)
    goal = xp.zeros_like(total) if target is None else target
    safe = xp.where(mass > 0, mass, np.float32(1))
    shift = xp.where(mass > 0, (goal - total) / safe, np.float32(0))
    return d + shift[None, :]


@dataclass
class RangeLedger:
    """Out-of-range column counts, kept on the device until read.

    On Metal the running sum is a facade array like any other: its storage
    is encoded with the next launch (``woof.backend.facade._array.PENDING``),
    so the chain of additions is evaluated step by step and holds no step's
    inputs alive."""

    calls: int = 0
    columns: int = 0
    _flagged: Any = None

    def add(self, mask: Any) -> None:
        self.calls += 1
        self.columns += int(mask.shape[0])
        count = mask.sum(dtype=np.int64) if hasattr(mask, "sum") else np.int64(np.sum(mask))
        self._flagged = count if self._flagged is None else self._flagged + count

    def flagged(self) -> int:
        if self._flagged is None:
            return 0
        value = self._flagged
        return int(value.get() if hasattr(value, "get") else value)

    def summary(self) -> dict:
        flagged = self.flagged()
        return {"calls": self.calls, "columns_seen": self.columns, "columns_out_of_range": flagged,
                "fraction_out_of_range": (flagged / self.columns) if self.columns else 0.0}


@dataclass
class SafetyPolicy:
    """The constraints a seat applies around one model call."""

    out_of_range: str = "fallback"
    margin: float = 0.0
    limit_outputs: bool = True
    positivity: bool = True
    fixer: str = "none"
    ledger: RangeLedger = field(default_factory=RangeLedger)

    def __post_init__(self):
        if self.out_of_range not in OUT_OF_RANGE:
            raise ValueError(f"out_of_range must be one of {OUT_OF_RANGE}; got {self.out_of_range!r}")
        if self.fixer not in FIXERS:
            raise ValueError(f"fixer must be one of {FIXERS}; got {self.fixer!r}")
        if not (isinstance(self.margin, (int, float)) and 0.0 <= float(self.margin) <= 1.0):
            raise ValueError(f"bounds margin must be a fraction of the range in [0, 1]; got {self.margin!r}")

    def screen(self, x: Any, manifest, *, feature_names=None):
        """Apply the range rule to ``x``; returns ``(x, fallback_mask or None)``."""
        lo, hi = manifest.input_min, manifest.input_max
        mask = out_of_range_columns(x, lo, hi, self.margin)
        if self.out_of_range == "clip":
            # Clipping moves a finite value into range; a NaN or inf column
            # has no value to move, so it falls back like any other.
            self.ledger.add(mask)
            xp = array_module(x)
            return clip_inputs(x, lo, hi, self.margin), ~xp.isfinite(x).all(axis=0)
        self.ledger.add(mask)
        if self.out_of_range == "refuse" and bool(mask.any()):
            index, count = worst_feature(x, lo, hi, self.margin)
            name = feature_names[index] if feature_names is not None else f"feature {index}"
            raise InputOutOfRange(
                f"{int(mask.sum())} of {mask.shape[0]} columns are outside {manifest.name}'s trained input "
                f"range (most often {name}, in {count} columns); out_of_range = \"refuse\" stops here "
                "rather than extrapolate. Use \"fallback\" to keep the physics scheme there, or "
                "retrain on data that covers this state")
        return x, mask

    def constrain(self, y: Any, manifest, *, dt: float, state: Mapping[str, Any] | None = None,
                  weight: Any = None, targets: Mapping[str, Any] | None = None):
        """Apply the output rules to ``y`` (``[n_outputs, column]``).

        ``state`` maps a field name to its current ``[level, column]``
        values: ``qv`` (and any other moisture species) for positivity, and
        ``exner`` for the heat fixer.  An output ``d<name>`` is a moisture
        tendency when ``<name>`` starts with ``q``.  ``weight`` is the layer
        mass per unit area (``[level, column]``); ``targets`` maps an output
        name to the column integral the fixer must reproduce (absent: zero).
        The fixer runs before positivity so that the floor has the last word.
        """
        if self.limit_outputs and manifest.output_limit is not None:
            y = limit_outputs(y, manifest.output_limit)
        state = state or {}
        targets = targets or {}
        pieces = []
        touched = False
        for spec, rows in zip(manifest.outputs, manifest.output_slices().values()):
            value = y[rows]
            field = spec.name[1:] if spec.name.startswith("d") else ""
            moisture = field.startswith("q")
            heat = field in ("theta", "t") and self.fixer == "moisture+heat"
            if self.fixer != "none" and weight is not None and (moisture or heat):
                w = weight[:spec.levels]
                if heat:
                    if "exner" not in state:
                        raise ValueError("the moisture+heat fixer needs state['exner'] for the heat budget")
                    w = w * state["exner"][:spec.levels]
                value = column_fix(value, w, targets.get(spec.name))
                touched = True
            if self.positivity and moisture and field in state:
                value = positive_tendency(value, state[field][:spec.levels], dt)
                touched = True
            pieces.append(value)
        if not touched:
            return y
        return array_module(y).concatenate(pieces, axis=0)
