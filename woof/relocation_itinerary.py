"""A ``[[relocation.move]]`` itinerary that keeps a nest on a known track.

The relocation runner (:mod:`woof.core.relocation_runner`) executes a
scripted itinerary exactly: each ``[[relocation.move]]`` row fires at its
``at_seconds`` cadence opportunity, shifting the tracked mover by whole
cells of its parent, and a ``[relocation.containment]`` parent slides in
whole cells of ITS parent whenever the mover has drifted past a dead-band,
carrying the mover earth-fixed.  What the runner does not do is decide
where the rows should point.  For a target whose 3-D path is already
known -- anything that drifts or flies on a recorded or forecast course --
this module does: it projects the path onto the tree's own grids and writes the rows that
keep every cadence window's stretch of the track a stated number of mover
cells inside the mover, while the sliding parent stays a stated number of
cells inside the static frame it slides over.

THE GEOMETRY.  Three grids matter and only three: the tracked MOVER
(``[relocation] grid_id``), its parent the CONTAINER
(``[relocation.containment] grid_id``), and the container's parent, the
static FRAME (d01 on a three-domain tree).  The track is projected once,
with the frame's own projected grid
(:func:`woof.static.projection.grids_from_projection_config`, the same
parent_start/ratio arithmetic ``woof domain`` writes into namelist.wps),
and measured in mover cells from the frame's west/south mass-cell edge.
The mover's GROUND placement -- its west edge in container cells from the
same edge -- is what an itinerary row changes, because the containment
slide compensates the mover by ``-shift x ratio`` and so never moves it
over the ground (:class:`woof.experiment.ContainmentConfig`).

THE CHOICE.  For every window between two cadence opportunities the
admissible ground placements form an interval per axis (the window's
track extent, held ``margin_cells`` inside both edges); the plan takes the
placement that centres the window's stretch of track, clipped to that
interval and to ``max_move_parent_cells``.  Centring rather than holding
until the margin is about to break keeps the target away from the strip a
move exposes, which is cold-started from the parent and has to spin up.

THE CHECK.  The plan is then replayed against the runner's own rules,
opportunity by opportunity, in the runner's own order: the containment
slide first (its dead-band, its ``round(deviation / ratio)`` request, its
cap, the overlap floor and the SINT-stencil admissibility walk), then the
mover's row (cap, overlap floor, admissibility walk).  Any row the runner
would clamp, any window the clamped placement would not contain, and any
slide that would bring the container closer than ``container_margin_cells``
to the frame's edge is refused here, naming the time and the remedy -- a
track the itinerary cannot follow must not integrate as a nest that
quietly lost it.

What is NOT promised: a move invalidates restart identity exactly as
documented for every relocation (``docs/manual/02-dynamics-grids-nesting.md``
section 2.5.3); an itinerary does not change that.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np

from woof.explain import layered

#: Default distance, in mover cells, every window's stretch of track keeps
#: from the mover's edges.  40 cells is 4 km on a 100 m nest: well clear
#: of the relaxation zone, and of the strip a move cold-starts.
DEFAULT_MARGIN_CELLS = 40

#: Default distance, in frame cells, the sliding container keeps from the
#: frame's edges.  Ten cells clears the frame's own lateral boundary zone
#: and the SINT stencil with room to spare.
DEFAULT_CONTAINER_MARGIN_CELLS = 10

#: Default cadence of the itinerary, model seconds.
DEFAULT_CADENCE_SECONDS = 300.0

#: Mover's-parent cells the automatic dead-band leaves between the mover's
#: worst post-move deviation and the container's admissible edge (the SINT
#: stencil needs two; one more absorbs rounding of the slide request).
_DEADBAND_SAFETY_CELLS = 4

#: Default column (CSV) or key (JSON) names of a path file.  Every name is
#: an argument (``--time-col``/``--lat-col``/``--lon-col``/``--height-col``
#: on the command line); nothing here guesses at a producer's own spelling.
DEFAULT_TIME_COLUMN = "time"
DEFAULT_LAT_COLUMN = "lat"
DEFAULT_LON_COLUMN = "lon"


# ---------------------------------------------------------------------------
# Path input
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Track:
    """A cleaned, time-ordered 3-D path: naive-UTC times, degrees, metres.

    ``heights`` is ``None`` for a path that carries none.  Placement is
    horizontal -- a nest spans the whole column -- so the height rides
    along for the record (its range is reported) and moves nothing.
    """

    times: tuple[datetime, ...]
    lats: np.ndarray
    lons: np.ndarray
    heights: np.ndarray | None = None
    dropped: int = 0

    def __len__(self) -> int:
        return len(self.times)


def _parse_time(value) -> datetime:
    """ISO 8601 (``Z`` or an offset accepted) -> naive UTC."""
    if isinstance(value, datetime):
        stamp = value
    else:
        text = str(value).strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        stamp = datetime.fromisoformat(text.replace(" ", "T", 1)
                                       if "T" not in text else text)
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)
    return stamp


def clean_track(times: Sequence, lats: Sequence, lons: Sequence,
                heights: Sequence | None = None) -> Track:
    """Parse, drop unusable fixes and order a path.

    Dropped: unparseable times, non-finite or out-of-range coordinates,
    the exact ``(0, 0)`` fix a receiver writes when it has no position,
    and (when heights are given) a non-finite height.  Duplicate times
    keep their first fix.  A path left with no fix is refused.
    """
    if not (len(times) == len(lats) == len(lons)):
        raise ValueError(
            f"path columns differ in length: {len(times)} times, "
            f"{len(lats)} latitudes, {len(lons)} longitudes")
    if heights is not None and len(heights) != len(times):
        raise ValueError(
            f"path columns differ in length: {len(times)} times, "
            f"{len(heights)} heights")
    column = heights if heights is not None else [0.0] * len(times)
    kept: dict[datetime, tuple[float, float, float]] = {}
    dropped = 0
    for raw_t, raw_lat, raw_lon, raw_z in zip(times, lats, lons, column):
        try:
            lat = float(raw_lat)
            lon = float(raw_lon)
            z = float(raw_z)
            stamp = _parse_time(raw_t)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if (not math.isfinite(lat) or not math.isfinite(lon)
                or not math.isfinite(z)
                or abs(lat) > 90.0 or abs(lon) > 360.0
                or (lat == 0.0 and lon == 0.0)):
            dropped += 1
            continue
        if stamp in kept:
            dropped += 1
            continue
        kept[stamp] = (lat, ((lon + 180.0) % 360.0) - 180.0, z)
    if not kept:
        raise ValueError(
            "the path has no usable fix: every row was unparseable, out "
            "of range, or the (0, 0) no-position fix")
    order = sorted(kept)
    return Track(
        times=tuple(order),
        lats=np.array([kept[t][0] for t in order], dtype=np.float64),
        lons=np.array([kept[t][1] for t in order], dtype=np.float64),
        heights=(None if heights is None else np.array(
            [kept[t][2] for t in order], dtype=np.float64)),
        dropped=dropped)


def _column(columns: Sequence[str], name: str, path, flag: str) -> str:
    """The column/key named ``name``, exactly, else a refusal naming them."""
    if name in columns:
        return name
    raise ValueError(
        f"{path}: no column {name!r} (pass the right name with {flag}; the "
        f"file has {', '.join(map(repr, columns)) or 'none'})")


def load_track(path: str | Path, *, time_col: str = DEFAULT_TIME_COLUMN,
               lat_col: str = DEFAULT_LAT_COLUMN,
               lon_col: str = DEFAULT_LON_COLUMN,
               height_col: str | None = None) -> Track:
    """A path from a CSV or a JSON list, columns named by the caller.

    CSV: a header row naming the columns.  JSON (``.json``): a list of
    objects keyed by the same names.  ``height_col`` ``None`` reads no
    height.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list) or not all(
                isinstance(row, dict) for row in payload):
            raise ValueError(
                f"{path}: a JSON path is a list of objects, one per fix")
        columns = sorted({key for row in payload for key in row})
        rows = payload
    else:
        reader = csv.DictReader(text.splitlines())
        columns = [name.strip() for name in (reader.fieldnames or ())]
        rows = [{(k.strip() if isinstance(k, str) else k): v
                 for k, v in row.items()} for row in reader]
    t_key = _column(columns, time_col, path, "--time-col")
    lat_key = _column(columns, lat_col, path, "--lat-col")
    lon_key = _column(columns, lon_col, path, "--lon-col")
    z_key = (None if height_col is None
             else _column(columns, height_col, path, "--height-col"))
    return clean_track([row.get(t_key) for row in rows],
                       [row.get(lat_key) for row in rows],
                       [row.get(lon_key) for row in rows],
                       None if z_key is None
                       else [row.get(z_key) for row in rows])


# ---------------------------------------------------------------------------
# Tree geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NestBox:
    """One grid's extent and placement, as ``[[domain]]`` declares it.

    ``nx``/``ny`` are MASS cells (``e_we - 1``); ``i_parent_start`` /
    ``j_parent_start`` are 1-based parent mass cells, WPS convention.
    """

    grid_id: int
    nx: int
    ny: int
    dx: float
    parent_grid_ratio: int = 1
    i_parent_start: int = 1
    j_parent_start: int = 1


@dataclass(frozen=True)
class TreeGeometry:
    """The three grids an itinerary is planned on, plus the clock.

    ``frame_grid`` is the frame's projected grid (anything with
    ``latlon_to_ij``/``ij_to_latlon`` in the frame's own 1-based mass
    coordinates).  ``mover_start_seconds`` is when the mover starts
    integrating (a late-starting nest cannot move before it exists).
    """

    frame_grid: object
    frame: NestBox
    container: NestBox
    mover: NestBox
    root_dt: Fraction
    run_seconds: float
    start_time: datetime | None = None
    mover_start_seconds: float = 0.0
    min_overlap_fraction: float | None = None

    @classmethod
    def from_experiment(cls, exp, *, mover_grid_id: int | None = None
                        ) -> "TreeGeometry":
        """Read the geometry off a loaded experiment.

        ``mover_grid_id`` defaults to the experiment's ``[relocation]
        grid_id`` when it names one, else to the deepest domain.
        """
        from woof.static.projection import grids_from_projection_config

        by_id = {int(dc.grid_id): dc for dc in exp.domains}
        if mover_grid_id is None:
            relocation = getattr(exp, "relocation", None)
            mover_grid_id = getattr(relocation, "grid_id", None)
        if mover_grid_id is None:
            def depth(dc) -> int:
                n = 0
                while int(dc.parent_id) not in (0, int(dc.grid_id)):
                    dc = by_id[int(dc.parent_id)]
                    n += 1
                return n
            mover_grid_id = max(exp.domains, key=depth).grid_id
        mover_grid_id = int(mover_grid_id)
        if mover_grid_id not in by_id:
            raise ValueError(
                f"grid {mover_grid_id} is not a domain of {exp.name!r} "
                f"(have {sorted(by_id)})")
        mover = by_id[mover_grid_id]
        if int(mover.parent_id) in (0, mover_grid_id):
            raise ValueError(layered(
                f"d{mover_grid_id:02d} is the root; an itinerary moves a "
                "nest whose parent slides inside a static frame, so the "
                "mover must be at least two levels down (d03 on a "
                "three-domain tree).",
                "The containment parent is what keeps a long track inside "
                "the tree: the mover follows the target in its parent's "
                "cells and the parent slides over the frame.  A root has "
                "no parent to move inside."))
        container = by_id[int(mover.parent_id)]
        if int(container.parent_id) in (0, int(container.grid_id)):
            raise ValueError(layered(
                f"d{mover_grid_id:02d}'s parent d{int(container.grid_id):02d}"
                " is the root, which cannot slide; a track itinerary needs "
                "a three-level chain (frame, sliding container, mover).",
                "[relocation.containment] slides a NEST in whole cells of "
                "its parent; the root has no parent, so a two-domain tree "
                "can only follow a track as far as its static parent "
                "reaches."))
        frame = by_id[int(container.parent_id)]
        for dc in (mover, container):
            if getattr(dc, "spawn", None) is not None:
                raise ValueError(
                    f"d{int(dc.grid_id):02d} is a dormant (spawned) nest; "
                    "its placement is chosen when it fires, so an "
                    "itinerary has no declared start to plan from")
        if bool(getattr(exp.root.run, "use_adaptive_time_step", False)):
            raise ValueError(layered(
                "the root steps adaptively; a [[relocation.move]] row must "
                "land on a whole number of root steps, which an adaptive "
                "clock cannot promise.  Rebuild the tree with "
                "`woof domain --clock fixed`.",
                "Relocations execute at complete root-step boundaries.  "
                "Admission checks every row against the declared root dt, "
                "and an adaptive root moves its steps off that lattice."))
        grids = grids_from_projection_config(exp)
        frame_grid = grids[[int(dc.grid_id) for dc in exp.domains].index(
            int(frame.grid_id))]
        offset = 0.0
        if getattr(mover, "start_time", None) is not None:
            offset = max(0.0, (mover.start_time
                               - exp.start_time).total_seconds())

        def box(dc) -> NestBox:
            return NestBox(grid_id=int(dc.grid_id), nx=int(dc.run.nx),
                           ny=int(dc.run.ny), dx=float(dc.run.dx),
                           parent_grid_ratio=int(dc.parent_grid_ratio),
                           i_parent_start=int(dc.i_parent_start),
                           j_parent_start=int(dc.j_parent_start))

        relocation = getattr(exp, "relocation", None)
        overlap = (getattr(relocation, "min_overlap_fraction", None)
                   if getattr(relocation, "enabled", False) else None)
        return cls(frame_grid=frame_grid, frame=box(frame),
                   container=box(container), mover=box(mover),
                   root_dt=Fraction(exp.dt_exact(int(exp.root.grid_id))),
                   run_seconds=float(exp.run_seconds),
                   start_time=exp.start_time,
                   mover_start_seconds=offset,
                   min_overlap_fraction=overlap)

    @classmethod
    def from_config(cls, path: str | Path, *,
                    mover_grid_id: int | None = None) -> "TreeGeometry":
        from woof.experiment import load_experiment
        return cls.from_experiment(load_experiment(path),
                                   mover_grid_id=mover_grid_id)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannedMove:
    """One ``[[relocation.move]]`` row, plus where it leaves the mover."""

    at_seconds: float
    di_parent_cells: int
    dj_parent_cells: int
    centre_lat: float
    centre_lon: float


@dataclass(frozen=True)
class RelocationPlan:
    """The itinerary for one track on one tree.

    Not :class:`woof.core.nest_relocation.RelocationPlan` (one move of
    one child); this is the whole run's schedule plus the initial
    placement it assumes.  ``placements`` maps the mover and container
    grid ids to the ``(i_parent_start, j_parent_start)`` the
    ``[[domain]]`` tables must declare.
    """

    mover_grid_id: int
    container_grid_id: int
    cadence_seconds: float
    placements: dict
    moves: tuple[PlannedMove, ...]
    max_move_parent_cells: int | None
    deadband_cells: int
    containment_max_move_parent_cells: int
    min_overlap_fraction: float | None
    margin_cells: int
    container_margin_cells: int
    #: The smallest distance (mover cells) any window's track kept from
    #: the mover's edges, and from the frame's edges (frame cells) the
    #: container kept, over the replayed run.
    worst_margin_cells: float
    worst_container_margin_cells: int
    containment_slides: int
    track_seconds: tuple[float, float]
    summary: dict = field(default_factory=dict)

    def relocation_table(self) -> dict:
        """The ``[relocation]`` table as plain data (what TOML encodes).

        Empty when the path never leaves the initial placement's margins:
        a ``[relocation.containment]`` or a cadence with no row to drive
        is refused by admission (a half-opted-in surface), and a nest
        that never moves needs no relocation at all.
        """
        if not self.moves:
            return {}
        table: dict = {
            "enabled": True,
            "grid_id": self.mover_grid_id,
            "mode": "discrete-cycle-boundary",
            "cadence_seconds": float(self.cadence_seconds),
        }
        if self.max_move_parent_cells is not None:
            table["max_move_parent_cells"] = int(self.max_move_parent_cells)
        if self.min_overlap_fraction is not None:
            table["min_overlap_fraction"] = float(self.min_overlap_fraction)
        table["containment"] = {
            "grid_id": self.container_grid_id,
            "deadband_cells": int(self.deadband_cells),
            "max_move_parent_cells": int(
                self.containment_max_move_parent_cells),
        }
        table["move"] = [
            {"at_seconds": float(m.at_seconds),
             "di_parent_cells": int(m.di_parent_cells),
             "dj_parent_cells": int(m.dj_parent_cells)}
            for m in self.moves]
        return table

    def to_toml(self, *, placement_comment: bool = True) -> str:
        """The ``[relocation]`` block, ready to append to the tree config.

        The initial placement is stated as a comment because it belongs in
        the ``[[domain]]`` tables, not here; ``woof relocation from-track
        --write-config`` writes it there.
        """
        lines = [
            f"# Track itinerary: d{self.mover_grid_id:02d} follows the "
            f"track every {_num(self.cadence_seconds)} s, "
            f"d{self.container_grid_id:02d} slides to contain it "
            "(woof relocation from-track).",
            f"# {len(self.moves)} move(s), {self.containment_slides} "
            f"containment slide(s); the track stays >= "
            f"{self.worst_margin_cells:.0f} d{self.mover_grid_id:02d} "
            f"cells inside d{self.mover_grid_id:02d}, "
            f"d{self.container_grid_id:02d} >= "
            f"{self.worst_container_margin_cells} cells inside its parent.",
        ]
        if placement_comment:
            lines.append("# Initial placement the [[domain]] tables must "
                         "declare (centred where the path is when the "
                         "mover starts):")
            for gid in (self.container_grid_id, self.mover_grid_id):
                i, j = self.placements[gid]
                lines.append(f"#   d{gid:02d}: i_parent_start = {i}, "
                             f"j_parent_start = {j}")
        table = self.relocation_table()
        if not table:
            lines.append("# The path never leaves the initial placement's "
                         "margins: no [relocation] is needed, and none is "
                         "written.")
            return "\n".join(lines) + "\n"
        lines += ["[relocation]"]
        for key in ("enabled", "grid_id", "mode", "cadence_seconds",
                    "max_move_parent_cells", "min_overlap_fraction"):
            if key in table:
                lines.append(f"{key} = {_toml(table[key])}")
        lines += ["", "[relocation.containment]"]
        for key, value in table["containment"].items():
            lines.append(f"{key} = {_toml(value)}")
        for row in table.get("move", ()):
            lines += ["", "[[relocation.move]]"]
            for key, value in row.items():
                lines.append(f"{key} = {_toml(value)}")
        return "\n".join(lines) + "\n"

    def to_json(self) -> dict:
        return {
            "mover_grid_id": self.mover_grid_id,
            "container_grid_id": self.container_grid_id,
            "cadence_seconds": float(self.cadence_seconds),
            "placements": {f"d{gid:02d}": {"i_parent_start": ij[0],
                                           "j_parent_start": ij[1]}
                           for gid, ij in self.placements.items()},
            "relocation": self.relocation_table(),
            "worst_margin_cells": float(self.worst_margin_cells),
            "worst_container_margin_cells": int(
                self.worst_container_margin_cells),
            "containment_slides": int(self.containment_slides),
            "track_seconds": list(self.track_seconds),
            "moves": [{"at_seconds": m.at_seconds,
                       "di_parent_cells": m.di_parent_cells,
                       "dj_parent_cells": m.dj_parent_cells,
                       "centre_lat": round(m.centre_lat, 5),
                       "centre_lon": round(m.centre_lon, 5)}
                      for m in self.moves],
            **self.summary,
        }


def _num(value: float) -> str:
    return f"{value:g}"


def _toml(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(float(value))
        return text if ("." in text or "e" in text) else text + ".0"
    return json.dumps(str(value))


# ---------------------------------------------------------------------------
# Runner replay: the admissibility rules the runner applies
# ---------------------------------------------------------------------------

def _admissible(ips: int, jps: int, child: NestBox, parent: NestBox) -> bool:
    """register_nest's +-2 SINT stencil, the rule the runner walks by."""
    from woof.core.nest_interp import register_nest

    if ips < 1 or jps < 1:
        return False
    try:
        for stagger in ("", "x", "y"):
            register_nest(nri=child.parent_grid_ratio,
                          nrj=child.parent_grid_ratio,
                          i_parent_start=ips, j_parent_start=jps,
                          child_nx=child.nx, child_ny=child.ny,
                          parent_nx=parent.nx, parent_ny=parent.ny,
                          stagger=stagger, wrapper="bdy")
    except ValueError:
        return False
    return True


def _overlap_clamp(di: int, dj: int, child: NestBox,
                   floor: float | None) -> tuple[int, int]:
    from woof.core.nest_relocation import clamp_shift_to_overlap

    out_i, out_j, _ = clamp_shift_to_overlap(
        di, dj, min_overlap_fraction=floor,
        parent_grid_ratio=child.parent_grid_ratio,
        child_nx=child.nx, child_ny=child.ny)
    return out_i, out_j


def _walk(di: int, dj: int, ok) -> tuple[int, int]:
    """Walk a shift toward zero until ``ok`` admits it (runner order)."""
    while (di, dj) != (0, 0) and not ok(di, dj):
        if abs(di) >= abs(dj) and di != 0:
            di -= 1 if di > 0 else -1
        elif dj != 0:
            dj -= 1 if dj > 0 else -1
    return di, dj


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def _model_seconds(times: Sequence[datetime], start: datetime) -> np.ndarray:
    return np.array([(t - start).total_seconds() for t in times],
                    dtype=np.float64)


class _Path:
    """The track in mover cells from the frame's west/south mass-cell edge.

    Before the first fix the target sits at it (waiting at its start);
    after the last it is gone, and nothing past that is checked.
    """

    def __init__(self, seconds: np.ndarray, x: np.ndarray, y: np.ndarray):
        self.t, self.x, self.y = seconds, x, y

    def at(self, t: float) -> tuple[float, float]:
        return (float(np.interp(t, self.t, self.x)),
                float(np.interp(t, self.t, self.y)))

    def extent(self, a: float, b: float) -> tuple[float, float, float, float]:
        """``(xmin, xmax, ymin, ymax)`` over ``[a, b]``, ends interpolated."""
        lo = int(np.searchsorted(self.t, a, side="right"))
        hi = int(np.searchsorted(self.t, b, side="left"))
        xa, ya = self.at(a)
        xb, yb = self.at(b)
        xs = np.concatenate(([xa, xb], self.x[lo:hi]))
        ys = np.concatenate(([ya, yb], self.y[lo:hi]))
        return float(xs.min()), float(xs.max()), float(ys.min()), float(
            ys.max())


def _whole_root_steps(seconds: float, root_dt: Fraction) -> bool:
    steps = Fraction(seconds).limit_denominator(10**6) / root_dt
    return steps.denominator == 1


def itinerary_from_track(track_times: Sequence, lats: Sequence,
                         lons: Sequence, tree_geometry,
                         start_time: datetime | None = None,
                         cadence_s: float = DEFAULT_CADENCE_SECONDS,
                         margin_cells: int = DEFAULT_MARGIN_CELLS, *,
                         container_margin_cells: int =
                         DEFAULT_CONTAINER_MARGIN_CELLS,
                         max_move_parent_cells: int | None = None,
                         deadband_cells: int | None = None,
                         containment_max_move_parent_cells: int | None = None,
                         min_move_parent_cells: int = 1,
                         heights: Sequence | None = None,
                         _suggest: bool = True) -> RelocationPlan:
    """Plan the itinerary that keeps ``tree_geometry``'s mover on a track.

    ``tree_geometry`` is a :class:`TreeGeometry` or a loaded
    :class:`woof.experiment.ExperimentConfig`.  ``start_time`` (naive UTC)
    defaults to the geometry's experiment start.  ``margin_cells`` is in
    mover cells, ``container_margin_cells`` in frame cells.
    ``max_move_parent_cells`` caps one row (mover's-parent cells; ``None``
    writes the largest planned row as the cap, so nothing is clamped).
    ``deadband_cells`` / ``containment_max_move_parent_cells`` configure
    the containment slide; ``None`` derives the widest dead-band the
    container's slack affords and a cap the slide never meets.  A move
    smaller than ``min_move_parent_cells`` on both axes is skipped while
    the current placement still contains the window.  ``heights``
    (metres, optional) completes a 3-D path; placement is horizontal, so
    it is cleaned with the fixes and its range reported, nothing more.

    Refuses (ValueError) when no itinerary on this cadence keeps the
    track inside the margins, naming the first failing time and, where a
    smaller cadence would work, the largest one that does.
    """
    geom = (tree_geometry if isinstance(tree_geometry, TreeGeometry)
            else TreeGeometry.from_experiment(tree_geometry))
    track = clean_track(track_times, lats, lons, heights)
    for name, value in (("max_move_parent_cells", max_move_parent_cells),
                        ("deadband_cells", deadband_cells),
                        ("containment_max_move_parent_cells",
                         containment_max_move_parent_cells),
                        ("min_move_parent_cells", min_move_parent_cells)):
        if value is not None and int(value) < 1:
            raise ValueError(
                f"{name} = {value!r} must be at least 1 cell; the runner's "
                "admission refuses anything smaller")
    start = start_time if start_time is not None else geom.start_time
    if start is None:
        raise ValueError("no start_time: pass one, or a geometry read off "
                         "an experiment")
    start = _parse_time(start)
    if geom.start_time is not None and start != geom.start_time:
        # Rows, the mover's start and the run end are all model seconds
        # from the EXPERIMENT's start; measuring the path from any other
        # zero would fire every row early or late by the difference.
        raise ValueError(
            f"start_time {start.isoformat()} differs from the experiment's "
            f"{geom.start_time.isoformat()}; rows are model seconds from "
            "the experiment start, so the path is measured from it too")
    cadence = float(cadence_s)
    if not math.isfinite(cadence) or cadence <= 0.0:
        raise ValueError(f"cadence {cadence_s!r} must be a positive number "
                         "of model seconds")
    if not _whole_root_steps(cadence, geom.root_dt):
        raise ValueError(
            f"cadence {_num(cadence)} s is not a whole number of root steps "
            f"(root dt = {float(geom.root_dt):g} s); relocations execute at "
            "root-step boundaries, so this cadence never lands on one")
    margin = int(margin_cells)
    if margin < 0 or int(container_margin_cells) < 0:
        raise ValueError("margins must be non-negative cell counts")

    frame, cont, mover = geom.frame, geom.container, geom.mover
    r_c = int(cont.parent_grid_ratio)
    r_m = int(mover.parent_grid_ratio)
    if mover.nx % r_m or mover.ny % r_m:
        raise ValueError(
            f"d{mover.grid_id:02d} is {mover.nx}x{mover.ny} mass cells, not "
            f"a whole number of its ratio-{r_m} parent's cells")
    span_i, span_j = mover.nx // r_m, mover.ny // r_m
    if 2 * margin >= min(mover.nx, mover.ny):
        raise ValueError(
            f"margin {margin} cells on each side leaves no interior in a "
            f"{mover.nx}x{mover.ny} d{mover.grid_id:02d}")

    seconds = _model_seconds(track.times, start)
    run_end = float(geom.run_seconds)
    t0 = float(geom.mover_start_seconds)
    if seconds[-1] < t0:
        raise ValueError(
            f"the track ends {t0 - seconds[-1]:.0f} s before "
            f"{'the experiment starts' if t0 == 0 else 'd%02d starts' % mover.grid_id}"
            f" ({start.isoformat()}Z + {t0:g} s); there is nothing to "
            "follow")
    if seconds[0] >= run_end:
        raise ValueError(
            f"the track starts {seconds[0]:.0f} s after the start, at or "
            f"past the end of the run (run_seconds = {run_end:g}); there "
            "is nothing to follow")
    fx, fy = geom.frame_grid.latlon_to_ij(track.lats, track.lons)
    fx = np.asarray(fx, dtype=np.float64)
    fy = np.asarray(fy, dtype=np.float64)
    scale = r_c * r_m
    path = _Path(seconds, (fx - 0.5) * scale, (fy - 0.5) * scale)
    end = min(run_end, float(seconds[-1]))

    # Opportunities: k * cadence strictly inside the run, after the mover
    # exists (the runner holds a not-yet-started nest and its rows would
    # be retired unexecuted).
    count = max(0, int(math.ceil(run_end / cadence - 1e-9)) - 1)
    opportunities = [k * cadence for k in range(1, count + 1)
                     if k * cadence > t0]
    edges = [t0] + [t for t in opportunities if t < end] + [end]
    windows = list(zip(edges[:-1], edges[1:]))
    if end <= t0:
        windows = [(t0, t0)]

    def feasible(extent) -> tuple[tuple[int, int], tuple[int, int]]:
        """Admissible ground placements (container cells) per axis."""
        xmin, xmax, ymin, ymax = extent
        lo_i = math.ceil((xmax - mover.nx + margin) / r_m - 1e-9)
        hi_i = math.floor((xmin - margin) / r_m + 1e-9)
        lo_j = math.ceil((ymax - mover.ny + margin) / r_m - 1e-9)
        hi_j = math.floor((ymin - margin) / r_m + 1e-9)
        return (lo_i, hi_i), (lo_j, hi_j)

    def centred(extent) -> tuple[int, int]:
        xmin, xmax, ymin, ymax = extent
        return (int(round(((xmin + xmax) / 2.0 - mover.nx / 2.0) / r_m)),
                int(round(((ymin + ymax) / 2.0 - mover.ny / 2.0) / r_m)))

    def refuse_window(t: float, why: str, remedy: str, *,
                      cadence_helps: bool = True):
        suggestion = ""
        if _suggest and cadence_helps:
            better = _largest_working_cadence(
                track, geom, start, cadence, margin,
                container_margin_cells=container_margin_cells,
                max_move_parent_cells=max_move_parent_cells,
                deadband_cells=deadband_cells,
                containment_max_move_parent_cells=(
                    containment_max_move_parent_cells),
                min_move_parent_cells=min_move_parent_cells)
            if better is not None:
                suggestion = (f" A cadence of {_num(better)} s keeps it: "
                              f"rerun with --cadence {_num(better)}.")
            else:
                suggestion = (" No smaller cadence on this tree fixes it.")
        when = start_time_text(start, t)
        raise ValueError(layered(
            f"the track cannot be followed at a {_num(cadence)} s cadence: "
            f"at t = {t:.0f} s ({when}) {why}.{suggestion} {remedy}",
            "Each [[relocation.move]] row holds the nest still until the "
            "next cadence opportunity, so the stretch of track inside one "
            "window must fit the nest's interior with the margin on both "
            "sides; the containment parent must stay clear of its own "
            "parent's edge while it carries the nest; and the runner "
            "clamps any row or slide past its bounds rather than executing "
            "it, which would leave the target behind silently."))

    # Ground placement that centres the first window's stretch of track.
    (lo_i, hi_i), (lo_j, hi_j) = feasible(path.extent(*windows[0]))
    if lo_i > hi_i or lo_j > hi_j:
        refuse_window(windows[0][0],
                      "the first window's track is wider than the "
                      "nest's interior", "Use a smaller cadence or a "
                      "smaller margin.")
    want = centred(path.extent(t0, t0))
    ground = (min(max(want[0], lo_i), hi_i), min(max(want[1], lo_j), hi_j))
    # Container centred on the mover, the containment leg's own notion of
    # centred, in whole frame cells.
    centred_i = (cont.nx - span_i) // 2 + 1
    centred_j = (cont.ny - span_j) // 2 + 1
    c_i = int(round((ground[0] - (centred_i - 1)) / r_c)) + 1
    c_j = int(round((ground[1] - (centred_j - 1)) / r_c)) + 1
    m_i = ground[0] - (c_i - 1) * r_c + 1
    m_j = ground[1] - (c_j - 1) * r_c + 1

    def container_margin(ci: int, cj: int) -> int:
        # The far edges round the container's extent UP: a container
        # whose width is not a whole number of frame cells still covers
        # the partial one.
        return min(ci - 1, cj - 1,
                   frame.nx - (ci - 1) - math.ceil(cont.nx / r_c),
                   frame.ny - (cj - 1) - math.ceil(cont.ny / r_c))

    if container_margin(c_i, c_j) < int(container_margin_cells):
        raise ValueError(layered(
            f"d{cont.grid_id:02d} centred on the start of the track sits "
            f"{container_margin(c_i, c_j)} d{frame.grid_id:02d} cells from "
            f"d{frame.grid_id:02d}'s edge; it must keep "
            f"{int(container_margin_cells)}.  Enlarge d{frame.grid_id:02d}"
            " around the start of the path (or lower --container-margin-cells).",
            "The containment parent starts centred on the nest, which "
            "starts centred on the first fix, so the frame has to reach "
            "at least half the container plus the margin beyond it."))
    if not _admissible(c_i, c_j, cont, frame):
        raise ValueError(
            f"d{cont.grid_id:02d} at ({c_i}, {c_j}) is not an admissible "
            f"placement inside d{frame.grid_id:02d} (the SINT stencil needs "
            f"two d{frame.grid_id:02d} cells clear of the edge); enlarge "
            f"d{frame.grid_id:02d} around the start of the track")
    if not _admissible(m_i, m_j, mover, cont):
        raise ValueError(
            f"d{mover.grid_id:02d} at ({m_i}, {m_j}) is not an admissible "
            f"placement inside d{cont.grid_id:02d}; d{cont.grid_id:02d} is "
            "too small for it")

    # Desired ground placements, window by window.
    desired: list[tuple[float, tuple[int, int]]] = []
    current = ground
    for a, b in windows[1:]:
        ext = path.extent(a, b)
        (lo_i, hi_i), (lo_j, hi_j) = feasible(ext)
        if lo_i > hi_i or lo_j > hi_j:
            span = max(ext[1] - ext[0], ext[3] - ext[2])
            refuse_window(
                a, f"the track covers {span:.0f} d{mover.grid_id:02d} cells "
                f"in one {_num(b - a)} s window, and the interior with "
                f"{margin}-cell margins holds "
                f"{min(mover.nx, mover.ny) - 2 * margin}",
                "Use a smaller cadence, a smaller --margin-cells, or a "
                "larger nest.")
        target = centred(ext)
        target = (min(max(target[0], lo_i), hi_i),
                  min(max(target[1], lo_j), hi_j))
        inside = (lo_i <= current[0] <= hi_i and lo_j <= current[1] <= hi_j)
        if inside and max(abs(target[0] - current[0]),
                          abs(target[1] - current[1])) < int(
                              min_move_parent_cells):
            target = current
        if max_move_parent_cells is not None:
            cap = int(max_move_parent_cells)
            target = (current[0] + max(-cap, min(cap, target[0] - current[0])),
                      current[1] + max(-cap, min(cap, target[1] - current[1])))
        if target != current:
            desired.append((a, (target[0] - current[0],
                                target[1] - current[1])))
        current = target

    largest = max((max(abs(d[0]), abs(d[1])) for _, d in desired), default=0)
    row_cap = (int(max_move_parent_cells) if max_move_parent_cells is not None
               else max(1, largest))
    slack = min((cont.nx - span_i) // 2, (cont.ny - span_j) // 2) - 2
    if deadband_cells is None:
        deadband = slack - largest - _DEADBAND_SAFETY_CELLS + 1
        if deadband < 1:
            refuse_window(
                next((a for a, d in desired
                      if max(abs(d[0]), abs(d[1])) == largest), t0),
                f"one row moves d{mover.grid_id:02d} {largest} "
                f"d{cont.grid_id:02d} cells, more than "
                f"d{cont.grid_id:02d}'s {slack}-cell slack around a "
                "centred nest leaves room for",
                f"Use a smaller cadence or a larger d{cont.grid_id:02d}.")
    else:
        deadband = int(deadband_cells)
    if containment_max_move_parent_cells is None:
        slide_cap = max(1, int(math.ceil((deadband + largest) / r_c)) + 1)
    else:
        slide_cap = int(containment_max_move_parent_cells)

    # Replay against the runner's own rules.
    floor = geom.min_overlap_fraction
    rows = dict(desired)
    ci, cj, mi, mj = c_i, c_j, m_i, m_j
    slides = 0
    worst_container = container_margin(ci, cj)
    worst_margin = math.inf
    executed: list[PlannedMove] = []
    window_of = {a: (a, b) for a, b in windows}

    def check_window(a: float, b: float) -> None:
        nonlocal worst_margin
        xmin, xmax, ymin, ymax = path.extent(a, b)
        g_i = (ci - 1) * r_c + (mi - 1)
        g_j = (cj - 1) * r_c + (mj - 1)
        kept = min(xmin - g_i * r_m, g_i * r_m + mover.nx - xmax,
                   ymin - g_j * r_m, g_j * r_m + mover.ny - ymax)
        worst_margin = min(worst_margin, kept)
        if kept < margin - 1e-6:
            refuse_window(a, f"the replayed placement keeps the track only "
                          f"{kept:.0f} cells inside d{mover.grid_id:02d}",
                          "Use a smaller cadence or a larger "
                          f"d{cont.grid_id:02d}.")

    check_window(*windows[0])
    for t in opportunities:
        # Containment first, exactly as on_period_begin orders it.
        dev_i, dev_j = mi - centred_i, mj - centred_j
        if max(abs(dev_i), abs(dev_j)) >= deadband:
            want_i = int(round(dev_i / r_c))
            want_j = int(round(dev_j / r_c))
            si = max(-slide_cap, min(slide_cap, want_i))
            sj = max(-slide_cap, min(slide_cap, want_j))
            si, sj = _overlap_clamp(si, sj, cont, floor)

            def slide_ok(di, dj, ci=ci, cj=cj, mi=mi, mj=mj) -> bool:
                return (_admissible(ci + di, cj + dj, cont, frame)
                        and _admissible(mi - di * r_c, mj - dj * r_c,
                                        mover, cont))

            si, sj = _walk(si, sj, slide_ok)
            if (si, sj) != (0, 0):
                ci, cj = ci + si, cj + sj
                mi, mj = mi - si * r_c, mj - sj * r_c
                slides += 1
                kept = container_margin(ci, cj)
                worst_container = min(worst_container, kept)
                if kept < int(container_margin_cells):
                    refuse_window(
                        t, f"d{cont.grid_id:02d} slides to {kept} "
                        f"d{frame.grid_id:02d} cells from "
                        f"d{frame.grid_id:02d}'s edge (the margin is "
                        f"{int(container_margin_cells)})",
                        f"Enlarge d{frame.grid_id:02d} along the track "
                        "(woof domain --polygon over the track with a "
                        "wider --buffer-km).", cadence_helps=False)
        if t in rows:
            di, dj = rows[t]
            ri = max(-row_cap, min(row_cap, di))
            rj = max(-row_cap, min(row_cap, dj))
            ri, rj = _overlap_clamp(ri, rj, mover, floor)
            ri, rj = _walk(ri, rj, lambda a, b, mi=mi, mj=mj: _admissible(
                mi + a, mj + b, mover, cont))
            if (ri, rj) != (di, dj):
                refuse_window(
                    t, f"the runner would clamp the row ({di}, {dj}) to "
                    f"({ri}, {rj}) (d{mover.grid_id:02d} at the edge of "
                    f"d{cont.grid_id:02d}, or the overlap floor)",
                    "Use a smaller cadence, a larger "
                    f"d{cont.grid_id:02d}, or a larger "
                    "--containment-max-move.")
            mi, mj = mi + di, mj + dj
            g_i = (ci - 1) * r_c + (mi - 1)
            g_j = (cj - 1) * r_c + (mj - 1)
            lat, lon = geom.frame_grid.ij_to_latlon(
                0.5 + (g_i + span_i / 2.0) / r_c,
                0.5 + (g_j + span_j / 2.0) / r_c)
            executed.append(PlannedMove(at_seconds=float(t),
                                        di_parent_cells=int(di),
                                        dj_parent_cells=int(dj),
                                        centre_lat=float(lat),
                                        centre_lon=float(lon)))
        if t in window_of:
            check_window(*window_of[t])

    max_speed = 0.0
    if len(seconds) > 1:
        dt = np.diff(seconds)
        good = dt > 0
        if good.any():
            dist = np.hypot(np.diff(path.x), np.diff(path.y))[good] * (
                mover.dx)
            max_speed = float(np.max(dist / dt[good]))
    return RelocationPlan(
        mover_grid_id=mover.grid_id, container_grid_id=cont.grid_id,
        cadence_seconds=cadence,
        placements={cont.grid_id: (c_i, c_j), mover.grid_id: (m_i, m_j)},
        moves=tuple(executed),
        max_move_parent_cells=row_cap,
        deadband_cells=deadband,
        containment_max_move_parent_cells=slide_cap,
        min_overlap_fraction=floor,
        margin_cells=margin,
        container_margin_cells=int(container_margin_cells),
        worst_margin_cells=(float(worst_margin) if math.isfinite(worst_margin)
                            else float(margin)),
        worst_container_margin_cells=int(worst_container),
        containment_slides=slides,
        track_seconds=(float(seconds[0]), float(seconds[-1])),
        summary={"fixes": len(track), "fixes_dropped": track.dropped,
                 "max_track_speed_m_s": round(max_speed, 2),
                 **({} if track.heights is None else {
                     "height_range_m": [float(track.heights.min()),
                                        float(track.heights.max())]})})


def start_time_text(start: datetime, t: float) -> str:
    from datetime import timedelta
    return (start + timedelta(seconds=float(t))).strftime("%Y-%m-%dT%H:%MZ")


def _largest_working_cadence(track: Track, geom: TreeGeometry,
                             start: datetime, cadence: float, margin: int,
                             **kwargs) -> float | None:
    """The largest cadence below ``cadence`` the plan succeeds at.

    Searched on whole minutes where those are whole root steps, then on
    whole root steps, so the suggestion is a number someone would type.
    """
    step = float(geom.root_dt)
    lattice = 60.0 if _whole_root_steps(60.0, geom.root_dt) else step
    candidate = math.floor((cadence - 1e-9) / lattice) * lattice
    while candidate >= lattice:
        try:
            itinerary_from_track(track.times, track.lats, track.lons, geom,
                                 start, candidate, margin, _suggest=False,
                                 **kwargs)
            return candidate
        except ValueError:
            candidate -= lattice
    return None


# ---------------------------------------------------------------------------
# Writing it into a config
# ---------------------------------------------------------------------------

#: A TOML table header: ``[name]`` or ``[[name]]`` with an optional
#: trailing comment.  The name must be a bare or dotted key, so the
#: continuation line of a multi-line array (``  [1, 2],``) is not taken
#: for one.
_HEADER = re.compile(
    r"^\s*(\[\[?)\s*([A-Za-z0-9_\-]+(?:\s*\.\s*[A-Za-z0-9_\-]+)*)"
    r"\s*\]\]?\s*(#.*)?$")


def _header(line: str) -> tuple[str, bool] | None:
    """``(dotted name, is_array)`` for a header line, else ``None``."""
    match = _HEADER.match(line)
    if match is None:
        return None
    name = re.sub(r"\s*\.\s*", ".", match.group(2))
    return name, match.group(1) == "[["


def _strip_relocation(text: str) -> str:
    """Remove every ``[relocation]`` / ``[relocation.*]`` table.

    A ``relocation`` written any other way (a dotted top-level key, an
    inline table) is refused rather than half-removed.
    """
    import tomllib

    out: list[str] = []
    skipping = False
    for line in text.splitlines():
        header = _header(line)
        if header is not None:
            name = header[0]
            skipping = name == "relocation" or name.startswith("relocation.")
        if not skipping:
            out.append(line)
    stripped = "\n".join(out).rstrip() + "\n"
    if "relocation" in tomllib.loads(stripped):
        raise ValueError(
            "the config declares relocation outside a [relocation] table "
            "(a dotted key or an inline table); rewrite it as a table or "
            "delete it, so it can be replaced")
    return stripped


def _patch_placements(text: str, placements: dict) -> str:
    """Rewrite i/j_parent_start in the ``[[domain]]`` tables named."""
    import tomllib

    lines = text.splitlines()
    blocks: list[tuple[int, int]] = []
    start = None
    for index, line in enumerate(lines):
        header = _header(line)
        if header is None:
            continue
        if start is not None:
            blocks.append((start, index))
            start = None
        if header == ("domain", True):
            start = index
    if start is not None:
        blocks.append((start, len(lines)))
    remaining = dict(placements)
    for a, b in blocks:
        body = "\n".join(lines[a + 1:b])
        gid = tomllib.loads(body).get("grid_id")
        if gid is None or int(gid) not in remaining:
            continue
        gid = int(gid)
        i, j = remaining.pop(gid)
        for key, value in (("i_parent_start", i), ("j_parent_start", j)):
            pattern = re.compile(rf"^(\s*{key}\s*=\s*)[^#\n]*?(\s*(#.*)?)$")
            for index in range(a + 1, b):
                match = pattern.match(lines[index])
                if match:
                    lines[index] = f"{match.group(1)}{value}{match.group(2)}"
                    break
            else:
                raise ValueError(f"d{gid:02d}'s [[domain]] table declares no "
                                 f"{key} to rewrite")
    if remaining:
        raise ValueError(f"no [[domain]] table for grid(s) "
                         f"{sorted(remaining)}")
    return "\n".join(lines) + "\n"


def _patch_wps_placements(text: str, placements: dict) -> str:
    """Rewrite the &geogrid i/j_parent_start entries for the grids named.

    namelist.wps lists one value per domain in grid-id order, so grid
    ``g`` is entry ``g - 1``.
    """
    lines = text.splitlines()
    for key, axis in (("i_parent_start", 0), ("j_parent_start", 1)):
        pattern = re.compile(rf"^(\s*{key}\s*=\s*)([^/!]*?)(\s*(!.*)?)$",
                             re.IGNORECASE)
        for index, line in enumerate(lines):
            match = pattern.match(line)
            if not match:
                continue
            trailing = match.group(2).rstrip().endswith(",")
            values = [v.strip() for v in match.group(2).split(",")
                      if v.strip()]
            for gid, ij in placements.items():
                if gid - 1 >= len(values):
                    raise ValueError(f"namelist.wps {key} has no entry for "
                                     f"d{gid:02d}")
                values[gid - 1] = str(ij[axis])
            lines[index] = (match.group(1) + ", ".join(values)
                            + ("," if trailing else "") + match.group(3))
            break
        else:
            raise ValueError(f"namelist.wps declares no {key} to rewrite")
    return "\n".join(lines) + "\n"


def write_config_with_plan(config_path: str | Path, plan: RelocationPlan,
                           out_path: str | Path) -> Path:
    """The tree config with the plan's placement and ``[relocation]``.

    Any ``[relocation]`` the input carried is replaced.  The result is
    reloaded through :func:`woof.experiment.load_experiment` before it is
    kept, so admission (rows on root steps, inside the run, containment
    the mover's parent, the moving-nest physics refusals) has already
    passed when this returns.  A companion ``<stem>.namelist.wps`` beside
    the input (the file ``woof go`` binds) is written beside the output
    with the same placement, so the two cannot disagree.

    The output must sit in the input's directory: a config's relative
    paths resolve against its own directory, so a copy written elsewhere
    would silently point them at different files.
    """
    from woof.experiment import load_experiment

    config_path = Path(config_path)
    out_path = Path(out_path)
    if out_path.resolve().parent != config_path.resolve().parent:
        raise ValueError(
            f"write the config beside {config_path.name} (in "
            f"{config_path.resolve().parent}), not in "
            f"{out_path.resolve().parent}: its relative paths resolve "
            "against the directory it sits in")
    if out_path.resolve() == config_path.resolve():
        raise ValueError(f"{out_path} is the input config; name a new file")
    text = config_path.read_text(encoding="utf-8")
    text = _patch_placements(_strip_relocation(text), plan.placements)
    block = plan.to_toml(placement_comment=False)
    text = text.rstrip() + "\n\n" + block
    companion = config_path.with_name(f"{config_path.stem}.namelist.wps")
    out_wps = out_path.with_name(f"{out_path.stem}.namelist.wps")
    wps_text = None
    if companion.is_file():
        wps_text = _patch_wps_placements(
            companion.read_text(encoding="utf-8"), plan.placements)
    elif out_wps.exists():
        raise ValueError(
            f"{out_wps} exists but {config_path.name} has no companion "
            "namelist.wps to derive it from; it would disagree with the "
            "new placement.  Remove it, or name a different output")
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp_wps = out_wps.with_name(out_wps.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        exp = load_experiment(tmp)
        for gid, (i, j) in plan.placements.items():
            dc = exp.domain(gid)
            if (int(dc.i_parent_start), int(dc.j_parent_start)) != (i, j):
                raise ValueError(f"d{gid:02d}'s placement did not round-trip")
        if wps_text is not None:
            tmp_wps.write_text(wps_text, encoding="utf-8")
    except BaseException:
        tmp.unlink(missing_ok=True)
        tmp_wps.unlink(missing_ok=True)
        raise
    if wps_text is not None:
        tmp_wps.replace(out_wps)
    try:
        tmp.replace(out_path)
    except BaseException:
        # No companion is better than one that disagrees with the config
        # beside it.
        if wps_text is not None:
            out_wps.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
        raise
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "relocation",
        help="plan moving-nest relocation itineraries",
        description="Plan [relocation] itineraries for a moving nest.")
    sub = parser.add_subparsers(dest="relocation_command",
                                metavar="SUBCOMMAND", required=True)
    track = sub.add_parser(
        "from-track",
        help="write the [[relocation.move]] itinerary that keeps a nest on "
             "a known path",
        description=(
            "Project a path (time UTC, lat, lon, optionally height) onto a "
            "domain tree and "
            "write the [relocation] block -- [[relocation.move]] rows for "
            "the mover plus a [relocation.containment] slide for its "
            "parent -- that keeps every cadence window's stretch of track "
            "--margin-cells inside the mover, replayed against the "
            "runner's own clamps.  Also states the initial placement, "
            "centred where the path is when the mover starts (its first "
            "fix, for a path that starts after that)."))
    track.add_argument("track", type=Path,
                       help="path file: a CSV with a header row, or a .json "
                            "list of objects, one per fix; column names "
                            "come from --time-col/--lat-col/--lon-col/"
                            "--height-col; (0, 0) fixes are dropped")
    track.add_argument("config", type=Path,
                       help="the domain-tree experiment TOML (projection, "
                            "[[domain]] geometry and start_time)")
    track.add_argument("--time-col", default=DEFAULT_TIME_COLUMN,
                       metavar="NAME",
                       help="column holding the ISO 8601 time, UTC unless "
                            f"it carries an offset (default "
                            f"{DEFAULT_TIME_COLUMN!r})")
    track.add_argument("--lat-col", default=DEFAULT_LAT_COLUMN,
                       metavar="NAME",
                       help="column holding latitude, degrees (default "
                            f"{DEFAULT_LAT_COLUMN!r})")
    track.add_argument("--lon-col", default=DEFAULT_LON_COLUMN,
                       metavar="NAME",
                       help="column holding longitude, degrees (default "
                            f"{DEFAULT_LON_COLUMN!r})")
    track.add_argument("--height-col", default=None, metavar="NAME",
                       help="column holding height, metres (optional; "
                            "placement is horizontal, so it is reported, "
                            "not used)")
    track.add_argument("--grid-id", type=int, default=None,
                       help="the mover (default: [relocation] grid_id, else "
                            "the deepest domain)")
    track.add_argument("--cadence", type=float,
                       default=DEFAULT_CADENCE_SECONDS, metavar="SECONDS",
                       help="seconds between move opportunities (default "
                            f"{_num(DEFAULT_CADENCE_SECONDS)}); a whole "
                            "number of root steps")
    track.add_argument("--margin-cells", type=int,
                       default=DEFAULT_MARGIN_CELLS,
                       help="mover cells the track keeps from the mover's "
                            f"edges (default {DEFAULT_MARGIN_CELLS})")
    track.add_argument("--container-margin-cells", type=int,
                       default=DEFAULT_CONTAINER_MARGIN_CELLS,
                       help="frame cells the sliding parent keeps from the "
                            "frame's edges (default "
                            f"{DEFAULT_CONTAINER_MARGIN_CELLS})")
    track.add_argument("--max-move-parent-cells", type=int, default=None,
                       help="cap on one row (default: the largest planned "
                            "row, so nothing is clamped)")
    track.add_argument("--deadband-cells", type=int, default=None,
                       help="containment dead-band in the parent's cells "
                            "(default: the widest the parent's slack "
                            "affords)")
    track.add_argument("--containment-max-move", type=int, default=None,
                       help="cap on one containment slide, frame cells "
                            "(default: one the slide never meets)")
    track.add_argument("--min-move-parent-cells", type=int, default=1,
                       help="skip a re-centring move smaller than this "
                            "while the window still fits (default 1)")
    track.add_argument("--out", type=Path, default=None,
                       help="write the [relocation] snippet here instead of "
                            "printing it")
    track.add_argument("--write-config", type=Path, default=None,
                       metavar="TOML",
                       help="write the whole config with the placement "
                            "patched into [[domain]] and [relocation] "
                            "replaced, validated by reloading it")
    track.add_argument("--json", action="store_true",
                       help="print the plan as JSON")
    track.set_defaults(func=from_track_main)


def from_track_main(args) -> int:
    from woof.experiment import load_experiment

    exp = load_experiment(args.config)
    geom = TreeGeometry.from_experiment(exp, mover_grid_id=args.grid_id)
    track = load_track(args.track, time_col=args.time_col,
                       lat_col=args.lat_col, lon_col=args.lon_col,
                       height_col=args.height_col)
    plan = itinerary_from_track(
        track.times, track.lats, track.lons, geom, heights=track.heights,
        cadence_s=args.cadence,
        margin_cells=args.margin_cells,
        container_margin_cells=args.container_margin_cells,
        max_move_parent_cells=args.max_move_parent_cells,
        deadband_cells=args.deadband_cells,
        containment_max_move_parent_cells=args.containment_max_move,
        min_move_parent_cells=args.min_move_parent_cells)
    plan = replace(plan, summary={**plan.summary,
                                  "fixes_dropped": track.dropped})
    snippet = plan.to_toml()
    written = None
    if args.write_config is not None:
        written = write_config_with_plan(args.config, plan,
                                         args.write_config)
    if args.out is not None:
        args.out.write_text(snippet, encoding="utf-8")
    if args.json:
        payload = plan.to_json()
        if written is not None:
            payload["config"] = str(written)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.out is None:
        print(snippet, end="")
    lines = [
        f"relocation from-track: {len(track)} fixes "
        f"({track.dropped} dropped), {len(plan.moves)} move(s) at "
        f"{_num(plan.cadence_seconds)} s cadence, "
        f"{plan.containment_slides} containment slide(s); track >= "
        f"{plan.worst_margin_cells:.0f} cells inside "
        f"d{plan.mover_grid_id:02d}, d{plan.container_grid_id:02d} >= "
        f"{plan.worst_container_margin_cells} cells inside its parent"]
    if args.out is not None:
        lines.append(f"  snippet: {args.out}")
    if written is not None:
        lines.append(f"  config: {written} (placement patched, reloaded)")
    import sys
    print("\n".join(lines), file=sys.stderr)
    return 0


__all__ = [
    "DEFAULT_CADENCE_SECONDS", "DEFAULT_CONTAINER_MARGIN_CELLS",
    "DEFAULT_LAT_COLUMN", "DEFAULT_LON_COLUMN", "DEFAULT_MARGIN_CELLS",
    "DEFAULT_TIME_COLUMN", "NestBox", "PlannedMove", "RelocationPlan",
    "Track", "TreeGeometry", "clean_track", "itinerary_from_track",
    "load_track", "write_config_with_plan",
]
