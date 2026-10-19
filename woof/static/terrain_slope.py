"""Smooth the ground steeper than the dynamics hold, and only that ground.

THE DEFECT THIS EXISTS FOR.  A 100 m moving nest under a 300 m and a
1.5 km parent, prepared from 30 m Copernicus terrain over canyon country,
read a steepest slope of 1.56 (57 degrees) on a canyon wall and went
non-finite at its fourth step, two model seconds in.  Its start state was
calm (``w`` zero, ``u`` under 19 m/s); on the lowest model level over the
wall ``w`` went 36, 211 and 874 m/s on the first three steps.  MEASURED
on that prepared tree, changing one thing at a time:

* half the long step (0.25 s) went non-finite as fast, step for step;
* six acoustic substeps, and six with ``epssm`` 0.9, went non-finite by
  step 7; ``diff_opt`` 1 changed nothing (the nest's TKE starts at zero,
  so no LES mixing acts yet);
* three 1-2-1 smoothing passes over the whole nest (steepest slope 0.97)
  went non-finite at step 12; six (steepest 0.75) ran with surface ``w``
  under 8 m/s on the configured clock (0.5 s, four substeps);
* walls cut down to a 0.83 slope and nothing else -- each face moved only
  as far as its excess -- went non-finite at step 107, in a narrow canyon
  whose two walls then met in a sharp V;
* local smoothing of the steep ground only, the method below, ran with
  surface ``w`` under 13 m/s: ten model minutes with the bound at 0.70
  and six (when the run was stopped) at 0.85, the bound used here.

So the ground is the limit, and both its slope and its sharpness matter.
The cure is smoothing, as WRF's own guidance has it, but smoothing every
hill of a 100 m nest to tame a few walls throws away the resolution the
nest exists for.

WHAT THIS DOES.  Each round reads every cell's steepest face the way
:func:`woof.acoustic_adaptation.steepest_slope` reads faces, marks the
cells steeper than :func:`held_slope`, widens the mark by ``_DILATE``
cells and tapers its edge over ``_TAPER`` more (the mark, 1-2-1
smoothed), and moves the marked ground toward its own
``_SMOOTH_PASSES``-pass 1-2-1 smoothing by the taper's weight.  Rounds
repeat until no cell is steeper than the bound or
:func:`steep_ground_rounds` have run.  A cell with no steep face within
reach is never touched, bit for bit, so gentle ground -- every flat
domain -- keeps its exact bytes.  Slopes are read on the map plane, without
the map factor the launch-time reading applies; on the grids nests use
the factor is within a few percent of one.  Where the rounds a spacing
runs leave ground still steeper than the bound, the preparation says so
and the launch-time acoustic rule reads it as before.

THE BOUND is the steepest slope the engine's measured acoustic map holds
with any substep count at the most off-centred row it measured
(:data:`woof.acoustic_adaptation.STABLE_SLOPE_BY_OFFCENTERING`: six
substeps at ``epssm`` 0.5, slope 0.85).  Ground the map holds is left
alone; ground past it, which no measured clock integrated, is smoothed
down to it.

REACH.  A round makes a cell depend on the ground :data:`ROUND_REACH`
cells around it, so the whole pass reaches :func:`steep_ground_reach`
cells and nothing past them.  The caller runs it over the source sampled
that much further out than the halo it already samples
(:mod:`woof.static.highres`), so a cell's heights are the same whichever
footprint it is built in -- what a moving nest's statics corridor needs,
whose crop must equal the footprint built directly.  The rounds are as
many as fit in :data:`STEEP_GROUND_REACH_M` of ground, at most
``_MAX_ROUNDS``, so the source window a domain reads grows by at most that
distance; at spacings where not one round fits (coarser than 500 m) no
pass runs.
"""
from __future__ import annotations

import math

import numpy as np

#: The farthest, on the ground, the pass may move terrain from a steep
#: face; it also bounds how much wider a source window is read.
STEEP_GROUND_REACH_M = 2500.0

_MAX_ROUNDS = 4
_SMOOTH_PASSES = 2
_DILATE = 1
_TAPER = 2

#: Cells a round reads around a cell it changes: the face reading's
#: centred differences (two), then the mark's dilation and taper -- or the
#: smoothing's own passes, were they wider.
ROUND_REACH = max(2 + _DILATE + _TAPER, _SMOOTH_PASSES)


def held_slope() -> float:
    """The steepest ground the pass leaves alone (module docstring)."""
    from woof.acoustic_adaptation import STABLE_SLOPE_BY_OFFCENTERING

    return float(max(row[2] for row in STABLE_SLOPE_BY_OFFCENTERING))


def steep_ground_rounds(dx: float) -> int:
    """Rounds the pass runs at this grid spacing (0: no pass)."""
    dx = float(dx)
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError(f"grid spacing must be positive, got {dx!r}")
    return min(_MAX_ROUNDS,
               int(STEEP_GROUND_REACH_M // (ROUND_REACH * dx)))


def steep_ground_reach(dx: float) -> int:
    """Cells of ground around a cell that its smoothed height depends on."""
    return steep_ground_rounds(dx) * ROUND_REACH


def _one_two_one(h: np.ndarray, passes: int) -> np.ndarray:
    """WPS's 1-2-1 sweeps (x, then y), outer ring held, in float64."""
    h = np.array(h, dtype=np.float64, copy=True)
    for _ in range(passes):
        if h.shape[1] > 2:
            g = h.copy()
            g[:, 1:-1] = 0.5 * h[:, 1:-1] + 0.25 * (h[:, :-2] + h[:, 2:])
            h = g
        if h.shape[0] > 2:
            g = h.copy()
            g[1:-1, :] = 0.5 * h[1:-1, :] + 0.25 * (h[:-2, :] + h[2:, :])
            h = g
    return h


def cell_slopes(h, dx: float, dy: float) -> np.ndarray:
    """Each cell's steepest face, read as ``steepest_slope`` reads faces."""
    h = np.asarray(h, dtype=np.float64)
    ny, nx = h.shape
    out = np.zeros_like(h)
    across_x = (np.gradient(h, float(dx), axis=1) if nx > 1
                else np.zeros_like(h))
    across_y = (np.gradient(h, float(dy), axis=0) if ny > 1
                else np.zeros_like(h))
    if nx > 1:
        face = np.hypot(np.diff(h, axis=1) / float(dx),
                        0.5 * (across_y[:, 1:] + across_y[:, :-1]))
        out[:, 1:] = np.maximum(out[:, 1:], face)
        out[:, :-1] = np.maximum(out[:, :-1], face)
    if ny > 1:
        face = np.hypot(np.diff(h, axis=0) / float(dy),
                        0.5 * (across_x[1:, :] + across_x[:-1, :]))
        out[1:, :] = np.maximum(out[1:, :], face)
        out[:-1, :] = np.maximum(out[:-1, :], face)
    return out


def _dilate(mask: np.ndarray, cells: int) -> np.ndarray:
    mask = mask.copy()
    for _ in range(cells):
        grown = mask.copy()
        grown[1:, :] |= mask[:-1, :]
        grown[:-1, :] |= mask[1:, :]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        mask = grown
    return mask


def smooth_steep_ground(terrain, dx: float, dy: float, *,
                        bound: float | None = None,
                        rounds: int | None = None
                        ) -> tuple[np.ndarray, dict[str, object] | None]:
    """``terrain`` smoothed where it is steeper than ``bound``.

    Returns the heights and a record of what moved, or the input array
    itself and ``None`` where no cell is steeper than ``bound`` (gentle
    ground keeps its exact bytes).  ``rounds`` defaults to
    :func:`steep_ground_rounds` at the finer spacing.
    """
    h = np.asarray(terrain, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError(f"terrain must be (ny, nx), got shape {h.shape}")
    if not np.isfinite(h).all():
        raise ValueError("terrain carries non-finite heights; fill the "
                         "cells no source covers before smoothing it")
    bound = held_slope() if bound is None else float(bound)
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError(f"slope bound must be positive, got {bound!r}")
    rounds = (steep_ground_rounds(min(dx, dy)) if rounds is None
              else int(rounds))
    work = h
    used = 0
    before = None
    for _ in range(rounds):
        slopes = cell_slopes(work, dx, dy)
        steep = slopes > bound
        if not steep.any():
            break
        if before is None:
            before = float(slopes.max())
        weight = _one_two_one(_dilate(steep, _DILATE).astype(np.float64),
                              _TAPER)
        weight = np.minimum(1.0, 2.0 * weight)
        moved = weight > 0.0
        target = _one_two_one(work, _SMOOTH_PASSES)
        work = work.copy()
        work[moved] += weight[moved] * (target[moved] - work[moved])
        used += 1
    if used == 0:
        return h, None
    change = np.abs(work - h)
    changed = change > 0.0
    return work, {
        "bound": bound,
        "rounds": rounds,
        "rounds_used": used,
        "steepest_cell_slope_before": before,
        "cells_changed": int(np.count_nonzero(changed)),
        "max_change_m": (float(change[changed].max()) if changed.any()
                         else 0.0),
    }


__all__ = ["ROUND_REACH", "STEEP_GROUND_REACH_M", "cell_slopes",
           "held_slope", "smooth_steep_ground", "steep_ground_reach",
           "steep_ground_rounds"]
