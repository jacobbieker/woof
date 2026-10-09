"""``woof energy plan --topology wrf-nests``: sibling LES nests in one run.

The plan is ONE WOOF experiment configuration.  Its leaf nests sit at
``--dx-m`` (50 or 100 m typical) over corridor clusters of forecast sites,
and a short parent chain forces them inside the same run::

    root      parent_dx_m (default dx_m x 25)    forced by the source
    middle    dx_m x 5 (one or more siblings)    covers clusters of leaves
    leaves    dx_m (one per site cluster)        own the sites

Every ratio is 3 or 5; ``--parent-dx-m`` picks the chain depth
(``parent_dx_m / dx_m`` must be a product of 3s and 5s).

How the layout is found
-----------------------
1.  Sites are projected with WOOF's own projection classes
    (:mod:`woof.static.projection`) on the projection ``woof domain``
    would choose for the centre of the sites
    (:func:`woof.domain_wizard.auto_projection`).
2.  :func:`woof.energy.geometry.cover_with_rectangles` covers the sites
    with leaf rectangles: every site sits ``--corridor-km`` inside its
    leaf, and every leaf corner lies on a parent cell edge.
3.  Each middle level groups its children into as few sibling boxes as
    the per-domain size cap allows, each child keeping at least
    ``spec_bdy_width + blend_width`` parent rows of clearance (the
    loader's own rule, :mod:`woof.experiment`).  The root is centred on
    the projection reference and holds every middle box with the same
    clearance plus a buffer (:data:`ROOT_BUFFER_KM`).
4.  The configuration is rendered by :func:`woof.domain_wizard.
    render_config` (the header, physics profile, guard comments, clock,
    ``[fetch]`` and ``[case_data]`` that ``woof domain`` writes), its
    ``[[domain]]`` tables are replaced by the sibling tree, and the text
    is loaded by the real experiment loader.  The whole tree is priced
    with the wizard's own phase estimate against the ``--card`` /
    ``--vram-gib`` budget (or the measured local card).  When it does
    not fit, the per-leaf size cap shrinks and the layout is rebuilt.

Refusals
--------
* more domains than WRF's ``MAX_DOMAINS`` (21) or ``--max-domains``:
  refused with the leaf count, recommending ``--topology wrf-tiles``
  (offline child tiles, no count limit);
* a tree that cannot fit the card at any leaf size within the domain
  limit: refused with the priced envelope and the budget;
* a source this door cannot emit for (HRRR's own route documents,
  sources prepared from local bytes), a ratio that is not a product of
  3s and 5s, and any emitted text the loader refuses.

Physics
-------
The suite is the one ``woof domain`` emits for the source, verbatim.  On
every domain finer than 1 km the LES gray-zone recipe of
``docs/public/LES.md`` is written per domain: ``bl_pbl_physics = 0``,
``km_opt = 3`` (3-D Smagorinsky: diagnostic, so each sibling carries no
cold-started TKE), ``mix_isotropic = 1`` and ``cu_physics = 0``.  LES is
implemented-unverified in WOOF and the plan's notes say so.
``[static.highres]`` is enabled for domains at or finer than 1 km.

Output
------
Leaves write every :data:`LEAF_HISTORY_INTERVAL_S` seconds with the
``energy`` history preset when this build has one, otherwise an explicit
``history_vars`` list of the fields ``woof energy extract`` reads.
Parents write the ``minimal`` preset hourly.

Python boundary: this module orchestrates.  The only array arithmetic is
per-site projection and bounding boxes on small vectors in numpy, which
``docs/dev/static-rust-port.md`` allows.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Sequence

import numpy as np

from woof.energy.contracts import (ContractError, Plan, PlanDomain, SiteSet,
                                   dump_plan, dump_sites, file_ref,
                                   load_sites)

TOPOLOGY = "wrf-nests"

#: Nest ratios this planner uses (WRF's own guidance: odd ratios 3 or 5).
ALLOWED_RATIOS = (5, 3)
#: Default root spacing as a multiple of the leaf spacing (two ratio-5 steps).
DEFAULT_PARENT_FACTOR = 25
#: Domains finer than this get the LES gray-zone recipe (docs/public/LES.md).
GRAY_ZONE_DX_M = 1000.0
#: Per-domain LES recipe written on every domain finer than 1 km.
LES_RECIPE = {"bl_pbl_physics": 0, "km_opt": 3, "mix_isotropic": 1,
              "cu_physics": 0}
#: Leaf history cadence (s): the energy products are quarter-hourly.
LEAF_HISTORY_INTERVAL_S = 900.0
#: Parent history cadence (s): parents own no sites, so they stay sparse.
PARENT_HISTORY_INTERVAL_S = 3600.0
#: Where ``woof go <config> --outdir <run_dir>`` writes a domain's history
#: (its default run stamp adds ``run-<stamp>/run/wrfout/``), relative to the
#: plan's ``run_dir``; every domain of the config shares that run_dir.
OUTPUT_GLOB = "run-*/run/wrfout/wrfout_d{grid_id:02d}_*"
#: Parent history selection.
PARENT_OUTPUT = {"preset": "minimal"}
#: History preset the leaves use when this build has it.
ENERGY_PRESET = "energy"
#: The wrfout fields ``woof energy extract`` reads, used when the
#: ``energy`` preset is absent.  Names the schema does not know are dropped
#: and the plan's notes list them.
EXTRACT_HISTORY_VARS = (
    "U", "V", "W", "T", "PH", "PHB", "HGT", "P", "PB",
    "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW",
    "T2", "Q2", "U10", "V10", "PSFC", "SWDOWN", "RAINNC", "RAINC",
    "SINALPHA", "COSALPHA", "XLAT", "XLONG",
)
#: Irradiance fields added only when the output schema knows them.
OPTIONAL_HISTORY_VARS = ("SWDDNI", "SWDDIF", "COSZEN")
#: Largest leaf side (mass cells) tried first; the VRAM loop shrinks it.
DEFAULT_LEAF_MAX_CELLS = 600
#: Factor the leaf cap shrinks by at most per VRAM retry.
LEAF_CAP_SHRINK = 0.8
#: Buffer (km) between a middle domain's children and its edge, on top of
#: the loader's spec_bdy_width + blend_width rows.
MIDDLE_BUFFER_KM = 5.0
#: Buffer (km) between the root's children and its edge.  The root is
#: forced by a coarse global source, so it keeps at least this much of its
#: own grid between the lateral boundary and the first nest.
ROOT_BUFFER_KM = 30.0
#: Default source when ``--source`` is not given.  ``woof domain``
#: defaults to ERA5, a reanalysis that cannot reach the operational
#: cycle this planner starts from by default; GFS publishes every
#: 00/06/12/18 UTC cycle, has a fetch front door and needs no account.
DEFAULT_SOURCE = "gfs"
#: Sources this door refuses: their routes need documents only
#: ``woof domain`` writes (HRRR's namelist pair and target document).
_REFUSED_SOURCES = ("hrrr",)


class PlanRefused(ValueError):
    """``woof energy plan --topology wrf-nests`` cannot emit this plan."""


class DomainLimitRefused(PlanRefused):
    """The sites need more domains than one WRF run can carry."""


class VramRefused(PlanRefused):
    """No layout within the domain limit fits the card."""


# --------------------------------------------------------------------------
# small helpers


def default_start_cycle(now: datetime | None = None) -> datetime:
    """The most recent 00/06/12/18 UTC cycle at or before ``now`` (naive UTC)."""

    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    return now.replace(hour=now.hour - now.hour % 6, minute=0, second=0,
                       microsecond=0)


def parse_start(raw: str | None, *, now: datetime | None = None) -> datetime:
    """``--start`` as a naive UTC datetime on a whole hour."""

    if raw is None:
        return default_start_cycle(now)
    text = str(raw).strip()
    if text.endswith("Z"):
        text = text[:-1]
    parsed = None
    for form in ("%Y-%m-%dT%H", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S",
                 "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, form)
            break
        except ValueError:
            continue
    if parsed is None:
        raise PlanRefused(f"--start {raw!r} must be YYYY-MM-DDTHH (UTC)")
    if parsed.minute or parsed.second:
        raise PlanRefused(f"--start {raw!r} must fall on a whole hour: it is "
                          "the source cycle the forecast starts from")
    return parsed


def chain_ratios(dx_m: float, parent_dx_m: float) -> tuple[int, ...]:
    """Nest ratios from the root at ``parent_dx_m`` down to ``dx_m``.

    The total ratio must be an integer product of 3s and 5s; 5s come first
    (outermost), so the default x25 chain is ``(5, 5)``.
    """

    total = parent_dx_m / dx_m
    whole = round(total)
    if whole < 3 or abs(total - whole) > 1e-6 * total:
        raise PlanRefused(
            f"--parent-dx-m {parent_dx_m:g} over --dx-m {dx_m:g} is a ratio of "
            f"{total:g}; the wrf-nests chain needs a whole ratio of at least 3 "
            "built from steps of 3 and 5 (e.g. 5, 9, 15, 25, 45, 75, 125)")
    ratios: list[int] = []
    rest = whole
    for ratio in ALLOWED_RATIOS:
        while rest % ratio == 0:
            ratios.append(ratio)
            rest //= ratio
    if rest != 1:
        raise PlanRefused(
            f"--parent-dx-m {parent_dx_m:g} over --dx-m {dx_m:g} is a ratio of "
            f"{whole}, which is not a product of 3s and 5s; WOOF nests step by "
            "3 or 5 (choose e.g. 5, 9, 15, 25, 45, 75 or 125 times --dx-m)")
    from woof.domain_wizard import MAX_CHAIN_DEPTH

    if len(ratios) > MAX_CHAIN_DEPTH:
        raise PlanRefused(f"a {whole}x chain needs {len(ratios)} nest levels; "
                          f"at most {MAX_CHAIN_DEPTH} are allowed")
    return tuple(ratios)


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")
    return slug or "sites"


def _floor_to(value: int, multiple: int) -> int:
    return (value // multiple) * multiple


def _ceil_to(value: int, multiple: int) -> int:
    return -((-value) // multiple) * multiple


def _wrap180(lon: float) -> float:
    return (lon + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------
# geometry: projection and layout


@dataclass
class _Box:
    """One domain, in whole cells of its own spacing on the global lattice.

    The lattice origin is the projection reference (the root's centre), so
    every edge of every level is an integer multiple of that level's cell.
    """

    level: int
    lo_x: int
    hi_x: int
    lo_y: int
    hi_y: int
    children: list["_Box"] = field(default_factory=list)
    members: np.ndarray | None = None
    parent: "_Box | None" = None
    grid_id: int = 0

    @property
    def nx(self) -> int:
        return self.hi_x - self.lo_x

    @property
    def ny(self) -> int:
        return self.hi_y - self.lo_y


@dataclass
class _Layout:
    levels: list[list[_Box]]
    ratios: tuple[int, ...]
    spacings: tuple[float, ...]
    cap: int
    margin_m: float
    #: Set when the leaves alone broke the domain limit and no parents
    #: were built: ``count`` is then the floor (leaves + one per level).
    truncated: bool = False

    @property
    def domains(self) -> list[_Box]:
        return [box for level in self.levels for box in level]

    @property
    def count(self) -> int:
        if self.truncated:
            return len(self.leaves) + len(self.ratios)
        return sum(len(level) for level in self.levels)

    @property
    def leaves(self) -> list[_Box]:
        return self.levels[-1]


def _projection_for(lat: float, lon: float) -> dict:
    from woof.domain_wizard import _projection_entries

    return _projection_entries(round(lat, 4), round(_wrap180(lon), 4))


def _unit_grid(projection: dict, spacing_m: float):
    """A grid whose mass index maps linearly onto projected metres."""

    from woof.static.projection import projection_class

    cls = projection_class(projection["map_proj"])
    return cls(ref_lat=projection["ref_lat"], ref_lon=projection["ref_lon"],
               truelat1=projection["truelat1"],
               truelat2=projection["truelat2"],
               stand_lon=projection["stand_lon"], dx=spacing_m,
               dy=spacing_m, e_we=2, e_sn=2)


def project_sites(lat: np.ndarray, lon: np.ndarray, projection: dict,
                  spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Sites -> metres east/north of the projection reference (WRF grid)."""

    grid = _unit_grid(projection, spacing_m)
    i, j = grid.latlon_to_ij(np.asarray(lat, dtype=np.float64),
                             np.asarray(lon, dtype=np.float64))
    # known point (the reference) is mass index 1 on a 2-point grid.
    return ((np.asarray(i) - grid.known_x) * spacing_m,
            (np.asarray(j) - grid.known_y) * spacing_m)


def choose_projection(lat: np.ndarray, lon: np.ndarray,
                      spacing_m: float) -> tuple[dict, np.ndarray, np.ndarray]:
    """Projection centred on the sites' extent, and the projected sites.

    The family and true latitudes come from ``woof domain``'s rule for the
    site centroid; the reference then moves to the centre of the projected
    extent so the root (centred on it) wastes no rows.
    """

    lon0 = float(lon[0])
    unwrapped = lon0 + np.array([_wrap180(v - lon0) for v in lon])
    projection = _projection_for(float(np.mean(lat)), float(np.mean(unwrapped)))
    x, y = project_sites(lat, lon, projection, spacing_m)
    grid = _unit_grid(projection, spacing_m)
    xc = 0.5 * (float(x.min()) + float(x.max()))
    yc = 0.5 * (float(y.min()) + float(y.max()))
    lat_c, lon_c = grid.ij_to_latlon(xc / spacing_m + grid.known_x,
                                     yc / spacing_m + grid.known_y)
    projection = _projection_for(float(lat_c), float(lon_c))
    x, y = project_sites(lat, lon, projection, spacing_m)
    return projection, x, y


def _member_indices(rects: Sequence[Any], count: int) -> list[np.ndarray]:
    """Validated member index arrays: a partition of ``range(count)``."""

    seen = np.zeros(count, dtype=np.int64)
    out = []
    for rect in rects:
        members = np.asarray(rect.members)
        if members.dtype == bool:
            members = np.flatnonzero(members)
        members = members.astype(np.int64).ravel()
        if members.size == 0:
            continue
        if members.min() < 0 or members.max() >= count:
            raise PlanRefused("the rectangle cover returned member indices "
                              "outside the site list")
        seen[members] += 1
        out.append(members)
    if not np.all(seen == 1):
        missing = int(np.sum(seen == 0))
        doubled = int(np.sum(seen > 1))
        raise PlanRefused(
            "the rectangle cover did not assign every site to exactly one "
            f"leaf ({missing} unassigned, {doubled} assigned twice); "
            "refusing rather than guess an owner")
    return out


def _overlaps(a: _Box, b: _Box) -> bool:
    return (a.lo_x < b.hi_x and b.lo_x < a.hi_x
            and a.lo_y < b.hi_y and b.lo_y < a.hi_y)


def _merge_overlapping(boxes: list[_Box]) -> list[_Box]:
    """Merge sibling boxes that overlap until none do.

    WRF does not allow nests of one level to overlap, so two siblings
    whose snapped extents intersect become one box (their union), whatever
    the size cap says; the VRAM pricing still sees the result.
    """

    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                ga, gb = boxes[a], boxes[b]
                if not _overlaps(ga, gb):
                    continue
                members = None
                if ga.members is not None and gb.members is not None:
                    members = np.concatenate([ga.members, gb.members])
                union = _Box(level=ga.level, lo_x=min(ga.lo_x, gb.lo_x),
                             hi_x=max(ga.hi_x, gb.hi_x),
                             lo_y=min(ga.lo_y, gb.lo_y),
                             hi_y=max(ga.hi_y, gb.hi_y),
                             children=ga.children + gb.children,
                             members=members)
                boxes = [g for k, g in enumerate(boxes) if k not in (a, b)]
                boxes.append(union)
                merged = True
                break
            if merged:
                break
    return boxes


def _group_boxes(children: list[_Box], *, level: int, child_ratio: int,
                 own_ratio: int | None, pad: int, cap: int) -> list[_Box]:
    """Group child boxes into as few parent boxes as ``cap`` allows.

    Each child is padded by ``pad`` cells of this level and snapped outward
    to this level's own parent cells (``own_ratio``); groups merge
    greedily by the smallest added area while the merged box fits ``cap``.
    """

    snap = own_ratio or 1
    groups: list[_Box] = []
    for child in sorted(children, key=lambda c: (c.lo_y, c.lo_x)):
        box = _Box(level=level,
                   lo_x=_floor_to(child.lo_x // child_ratio - pad, snap),
                   hi_x=_ceil_to(child.hi_x // child_ratio + pad, snap),
                   lo_y=_floor_to(child.lo_y // child_ratio - pad, snap),
                   hi_y=_ceil_to(child.hi_y // child_ratio + pad, snap),
                   children=[child])
        groups.append(box)
    cap = max([cap] + [max(g.nx, g.ny) for g in groups])
    while len(groups) > 1:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                ga, gb = groups[a], groups[b]
                lo_x, hi_x = min(ga.lo_x, gb.lo_x), max(ga.hi_x, gb.hi_x)
                lo_y, hi_y = min(ga.lo_y, gb.lo_y), max(ga.hi_y, gb.hi_y)
                if hi_x - lo_x > cap or hi_y - lo_y > cap:
                    continue
                added = ((hi_x - lo_x) * (hi_y - lo_y)
                         - ga.nx * ga.ny - gb.nx * gb.ny)
                if best is None or added < best[0]:
                    best = (added, a, b, lo_x, hi_x, lo_y, hi_y)
        if best is None:
            break
        _, a, b, lo_x, hi_x, lo_y, hi_y = best
        merged = _Box(level=level, lo_x=lo_x, hi_x=hi_x, lo_y=lo_y, hi_y=hi_y,
                      children=groups[a].children + groups[b].children)
        groups = [g for k, g in enumerate(groups) if k not in (a, b)]
        groups.append(merged)
    groups = _merge_overlapping(groups)
    for group in groups:
        for child in group.children:
            child.parent = group
    return sorted(groups, key=lambda g: (g.lo_y, g.lo_x))


def build_layout(x: np.ndarray, y: np.ndarray, *, dx_m: float,
                 ratios: tuple[int, ...], margin_m: float, cap: int,
                 min_axis: int, clearance_rows: int,
                 domain_limit: int | None = None) -> _Layout:
    """Leaves over the sites, middle siblings over the leaves, one root.

    ``cap`` is the largest leaf side in mass cells.  When ``domain_limit``
    is given the layout stops after the leaves if they alone exceed it.
    """

    from woof.energy import geometry

    depth = len(ratios)
    spacings = [dx_m]
    for ratio in reversed(ratios):
        spacings.insert(0, spacings[0] * ratio)
    leaf_ratio = ratios[-1]
    parent_cell = spacings[-2]
    slack = 2 * leaf_ratio
    rects = geometry.cover_with_rectangles(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64),
        margin_m=margin_m, dx_m=dx_m, max_nx=max(cap - slack, 1),
        max_ny=max(cap - slack, 1), align_m=parent_cell)
    leaves = []
    for members in _member_indices(rects, len(x)):
        xs, ys = x[members], y[members]
        box = _Box(
            level=depth,
            lo_x=math.floor((float(xs.min()) - margin_m) / parent_cell) * leaf_ratio,
            hi_x=math.ceil((float(xs.max()) + margin_m) / parent_cell) * leaf_ratio,
            lo_y=math.floor((float(ys.min()) - margin_m) / parent_cell) * leaf_ratio,
            hi_y=math.ceil((float(ys.max()) + margin_m) / parent_cell) * leaf_ratio,
            members=members)
        while box.nx < min_axis:
            box.lo_x -= leaf_ratio
            box.hi_x += leaf_ratio
        while box.ny < min_axis:
            box.lo_y -= leaf_ratio
            box.hi_y += leaf_ratio
        leaves.append(box)
    leaves = _merge_overlapping(leaves)
    leaves.sort(key=lambda b: (b.lo_y, b.lo_x))
    levels: list[list[_Box]] = [leaves]
    if domain_limit is not None and len(leaves) + depth > domain_limit:
        return _Layout(levels=[[] for _ in range(depth)] + [leaves],
                       ratios=ratios, spacings=tuple(spacings), cap=cap,
                       margin_m=margin_m, truncated=True)
    for level in range(depth - 1, 0, -1):
        pad = max(clearance_rows,
                  math.ceil(MIDDLE_BUFFER_KM * 1000.0 / spacings[level]))
        levels.insert(0, _group_boxes(
            levels[0], level=level, child_ratio=ratios[level],
            own_ratio=ratios[level - 1], pad=pad, cap=cap))
    # root, centred on the lattice origin
    pad = max(clearance_rows, math.ceil(ROOT_BUFFER_KM * 1000.0 / spacings[0]))
    top, r1 = levels[0], ratios[0]
    half_x = max(max(-(b.lo_x // r1) + pad for b in top),
                 max(b.hi_x // r1 + pad for b in top),
                 math.ceil(min_axis / 2))
    half_y = max(max(-(b.lo_y // r1) + pad for b in top),
                 max(b.hi_y // r1 + pad for b in top),
                 math.ceil(min_axis / 2))
    root = _Box(level=0, lo_x=-half_x, hi_x=half_x, lo_y=-half_y,
                hi_y=half_y, children=list(top))
    for box in top:
        box.parent = root
    levels.insert(0, [root])
    grid_id = 0
    for level in levels:
        for box in level:
            grid_id += 1
            box.grid_id = grid_id
    return _Layout(levels=levels, ratios=ratios, spacings=tuple(spacings),
                   cap=cap, margin_m=margin_m)


# --------------------------------------------------------------------------
# configuration text


def _leaf_output(notes: list[str]) -> tuple[dict, str]:
    from woof.io.history_selection import HISTORY_PRESETS, HISTORY_VOCABULARY

    if ENERGY_PRESET in HISTORY_PRESETS:
        return ({"preset": ENERGY_PRESET},
                f"leaves write the '{ENERGY_PRESET}' history preset")
    wanted = list(EXTRACT_HISTORY_VARS) + [
        name for name in OPTIONAL_HISTORY_VARS if name in HISTORY_VOCABULARY]
    unknown = [name for name in wanted if name not in HISTORY_VOCABULARY]
    kept = [name for name in wanted if name in HISTORY_VOCABULARY]
    absent = [name for name in OPTIONAL_HISTORY_VARS
              if name not in HISTORY_VOCABULARY]
    if unknown:
        notes.append("history_vars dropped names the wrfout schema does not "
                     f"know: {', '.join(unknown)}")
    if absent:
        notes.append("irradiance fields "
                     f"{', '.join(absent)} are not in this build's wrfout "
                     "schema, so leaves carry SWDOWN only")
    return ({"history_vars": kept},
            f"no '{ENERGY_PRESET}' history preset in this build, so leaves "
            f"write an explicit history_vars list ({len(kept)} fields)")


def _inline_table(entries: dict) -> str:
    from woof.domain_wizard import _toml_value

    parts = []
    for key, value in entries.items():
        if isinstance(value, (list, tuple)):
            rendered = "[" + ", ".join(_toml_value(v) for v in value) + "]"
        else:
            rendered = _toml_value(value)
        parts.append(f"{key} = {rendered}")
    return "{ " + ", ".join(parts) + " }"


def _render_domain(table: dict, output: dict) -> str:
    from woof.domain_wizard import _render_table

    return (_render_table("domain", table, array_of_tables=True)
            + f"output = {_inline_table(output)}\n")


def _split_domain_tables(text: str) -> tuple[str, str]:
    """(text before the first [[domain]], text after the last one)."""

    lines = text.split("\n")
    try:
        start = lines.index("[[domain]]")
    except ValueError as error:  # pragma: no cover - render_config always writes one
        raise PlanRefused("the wizard emitted no [[domain]] table") from error
    end = start
    while end < len(lines):
        if lines[end] != "[[domain]]":
            break
        end += 1
        while end < len(lines) and lines[end].strip():
            end += 1
        while end < len(lines) and not lines[end].strip():
            end += 1
    return "\n".join(lines[:start]), "\n".join(lines[end:])


def _fetch_and_case(*, source: str, start: datetime, hours: float, name: str,
                    projection: dict, root_n: tuple[int, int],
                    root_dx_m: float, outdir: Path, config_path: Path,
                    notes: list[str]) -> tuple[dict | None, dict | None]:
    """The ``[fetch]`` and ``[case_data]`` tables ``woof domain`` writes."""

    from woof import domain_wizard as dw
    from woof.fetch import parse_cycle, validate_fetch_hints

    try:
        dw._pole_clearance_refusal(projection, *root_n, root_dx_m,
                                   target_option="the sites")
    except ValueError as error:
        raise PlanRefused(str(error)) from error
    refusal = dw.source_coverage_refusal(projection, *root_n, source=source,
                                         root_dx_m=root_dx_m,
                                         target_option="the sites")
    if refusal is not None:
        raise PlanRefused(refusal)
    cadence, fetch_hours = dw.fetch_window(source, hours, 0)
    cycle = parse_cycle(start.strftime("%Y-%m-%dT%H"), source)
    data_dir = (outdir / "data" / name).resolve()
    hints: dict[str, Any] = {
        "source": source, "cycle": cycle.strftime("%Y-%m-%dT%H"),
        "hours": fetch_hours,
        "area": dw.fetch_area_hint(projection, *root_n, source=source,
                                   root_dx_m=root_dx_m, notes=notes),
        "out": str(data_dir),
    }
    from woof import fetch_routes

    if source in fetch_routes.route_ids():
        route = fetch_routes.route_for(source)
        if route.members is not None:
            hints["member"] = fetch_routes.resolve_member(route, None)[0]
    elif source == "era5":
        from woof.era5_member import validate_selection

        selected = validate_selection(product_type="reanalysis",
                                      provider="cds", member=None,
                                      cadence=cadence, cycle=cycle)
        if selected is not None:
            hints["member"] = selected
    if cadence is not None:
        hints["cadence"] = cadence
    emitted = hints if dw.source_has_fetch_front_door(source) else None
    if emitted is not None and not dw.source_fetch_takes_a_crop_box(source):
        emitted = {k: v for k, v in emitted.items()
                   if k not in {"area", "point", "radius_km"}}
    if emitted is not None:
        validate_fetch_hints(emitted, source=str(config_path))
    case_data = None
    if source == "era5":
        from woof.fetch import era5_combined_name

        combined = era5_combined_name(hints.get("era5_provider"))
        case_data = {
            "forcing": [dw._relative_or_absolute(data_dir / combined,
                                                 outdir.resolve())],
            "vtable": dw._PACKAGED_VTABLE.name,
            "forcing_interval_s": dw.source_forcing_interval_seconds("era5"),
            "wps_namelist": f"{config_path.stem}.namelist.wps",
            "geog_root": "${GPUWM_CASE_DATA_ROOT}/WPS_GEOG",
            "sfcp_to_sfcp": True,
            "output_domain": 1,
            "output_title": f"woof {name}",
        }
    return emitted, case_data


def _highres_block() -> str:
    from woof.static.highres_production import default_highres_cache_root

    root = str(default_highres_cache_root()).replace("\\", "/")
    return (
        "\n# High-resolution statics on every domain at or finer than 1 km\n"
        "# (docs/public/HIGHRES-TERRAIN.md): Copernicus/3DEP terrain plus\n"
        "# land cover and soil where the sources reach.  Refuses rather than\n"
        "# fall back to the 30-arc-second baseline.\n"
        "[static.highres]\n"
        "enabled = true\n"
        f'cache_root = "{root}"\n'
        'fields = "auto"\n'
        'on_refuse = "error"\n'
        f"max_dx_m = {GRAY_ZONE_DX_M!r}\n")


def render_plan_config(layout: _Layout, *, name: str, start: datetime,
                       hours: float, projection: dict, source: str,
                       profile: str | None, nz: int | None,
                       fetch_hints: dict | None, case_data: dict | None,
                       leaf_output: dict) -> str:
    """The emitted TOML: ``woof domain``'s author with a sibling tree."""

    from woof import domain_wizard as dw
    import tomllib

    ratios = layout.ratios
    root = layout.levels[0][0]
    template_dims = [(root.nx, root.ny)] + [(r * 24, r * 24) for r in ratios]
    text = dw.render_config(
        name=name, start_time=start, hours=hours, projection=projection,
        dims=template_dims, ratios=ratios, fetch_hints=fetch_hints,
        case_data=case_data, root_dx_m=layout.spacings[0], profile=profile,
        nz=nz, history_interval_s=PARENT_HISTORY_INTERVAL_S,
        nest_history_interval_s=LEAF_HISTORY_INTERVAL_S, clock="fixed")
    raw = tomllib.loads(text)
    templates = raw["domain"]
    head, tail = _split_domain_tables(text)
    chain_km = [s / 1000.0 for s in layout.spacings]
    stale = {"# " + line for line in dw.gray_zone_advisory(chain_km,
                                                          raw["shared"])}
    head_lines = []
    for line in head.split("\n"):
        if line in stale or line.startswith("# GRAY ZONE:"):
            continue
        head_lines.append(line)
    head = "\n".join(head_lines)
    head = head.replace(
        "# Emitted by `woof domain`",
        "# Emitted by `woof energy plan --topology wrf-nests` through "
        "`woof domain`'s author", 1)
    les_levels = [level for level, spacing in enumerate(layout.spacings)
                  if spacing < GRAY_ZONE_DX_M]
    preamble = (
        f"# WRF-NESTS ENERGY PLAN: {len(layout.leaves)} leaf nest(s) at "
        f"{layout.spacings[-1]:g} m over corridor clusters of forecast sites,\n"
        f"# {layout.count} domains in all, chain "
        f"{' -> '.join(f'{s:g}' for s in layout.spacings)} m; leaves are "
        "sibling nests of one run (feedback = 0).\n")
    if les_levels:
        preamble += (
            "# LES GRAY-ZONE RECIPE (docs/public/LES.md) on every domain finer "
            "than 1 km: bl_pbl_physics = 0,\n"
            "# km_opt = 3, mix_isotropic = 1, cu_physics = 0, written per "
            "domain below.  LES is implemented-unverified.\n")
    pieces = [preamble + head.rstrip("\n") + "\n"]
    for box in layout.domains:
        template = dict(templates[min(box.level, len(templates) - 1)])
        table = dict(template)
        table["grid_id"] = box.grid_id
        if box.level == 0:
            table.update(nx=box.nx, ny=box.ny)
        else:
            parent = box.parent
            ratio = ratios[box.level - 1]
            table.update(
                parent_id=parent.grid_id,
                i_parent_start=box.lo_x // ratio - parent.lo_x + 1,
                j_parent_start=box.lo_y // ratio - parent.lo_y + 1,
                parent_grid_ratio=ratio, parent_time_step_ratio=ratio,
                nx=box.nx, ny=box.ny)
        leaf = box.level == len(ratios)
        table["history_interval_s"] = (LEAF_HISTORY_INTERVAL_S if leaf
                                       else PARENT_HISTORY_INTERVAL_S)
        if layout.spacings[box.level] < GRAY_ZONE_DX_M:
            table.update(LES_RECIPE)
            if box.level == 0:
                table["cudt_minutes"] = 0.0
        pieces.append(_render_domain(table, leaf_output if leaf
                                     else PARENT_OUTPUT))
    body = "\n".join(pieces)
    text = body + ("\n" + tail if tail.strip() else "")
    from woof.static.source_defaults import with_source_static_defaults_text

    text = with_source_static_defaults_text(text, source)
    text = text.rstrip("\n") + "\n" + _highres_block()
    emitted = tomllib.loads(text).get("domain", [])
    if [t.get("grid_id") for t in emitted] != [b.grid_id for b in layout.domains]:
        raise PlanRefused("the wizard's emitted text could not be rebuilt "
                          "around the sibling tree (its [[domain]] tables "
                          "were not where this planner expects them)")
    return text


def render_plan_wps(layout: _Layout, projection: dict, *, source: str) -> str:
    """namelist.wps for the sibling tree (parent ids and starts per domain)."""

    from woof import domain_wizard as dw

    interval = dw.source_forcing_interval_seconds(source)
    if (not math.isfinite(interval) or interval <= 0
            or int(interval) != interval):
        raise PlanRefused("the source's forcing interval is not a whole "
                          "number of seconds")
    rows = []
    for box in layout.domains:
        if box.level == 0:
            rows.append((1, 1, 1, 1, box.nx + 1, box.ny + 1))
        else:
            ratio = layout.ratios[box.level - 1]
            parent = box.parent
            rows.append((parent.grid_id, ratio,
                         box.lo_x // ratio - parent.lo_x + 1,
                         box.lo_y // ratio - parent.lo_y + 1,
                         box.nx + 1, box.ny + 1))

    def csv(values):
        return ", ".join(str(v) for v in values) + ","

    num = dw._namelist_number
    return (
        "&share\n"
        " wrf_core = 'ARW',\n"
        f" max_dom = {len(rows)},\n"
        f" interval_seconds = {int(interval)},\n"
        " io_form_geogrid = 2,\n"
        "/\n"
        "&geogrid\n"
        f" parent_id         = {csv(r[0] for r in rows)}\n"
        f" parent_grid_ratio = {csv(r[1] for r in rows)}\n"
        f" i_parent_start    = {csv(r[2] for r in rows)}\n"
        f" j_parent_start    = {csv(r[3] for r in rows)}\n"
        f" e_we              = {csv(r[4] for r in rows)}\n"
        f" e_sn              = {csv(r[5] for r in rows)}\n"
        f" geog_data_res     = {csv(repr('default') for _ in rows)}\n"
        f" dx = {layout.spacings[0]:g},\n"
        f" dy = {layout.spacings[0]:g},\n"
        f" map_proj = '{projection['map_proj']}',\n"
        f" ref_lat   = {num(projection['ref_lat'])},\n"
        f" ref_lon   = {num(projection['ref_lon'])},\n"
        f" truelat1  = {num(projection['truelat1'])},\n"
        f" truelat2  = {num(projection['truelat2'])},\n"
        f" stand_lon = {num(projection['stand_lon'])},\n"
        "/\n")


# --------------------------------------------------------------------------
# sizing


@dataclass(frozen=True)
class _Budget:
    vram_gib: float
    free_bytes: int
    device_profile: Any
    note: str | None
    measured: bool


def _resolve_budget(card: str | None, vram_gib: float | None) -> _Budget:
    from woof.domain_wizard import resolve_sizing_budget

    try:
        sizing = resolve_sizing_budget(card, vram_gib)
    except ValueError as error:
        raise VramRefused(str(error)) from error
    return _Budget(sizing.vram_gib, sizing.free_bytes, sizing.device_profile,
                   sizing.note, sizing.measured)


def _price(exp, *, budget: _Budget, source: str) -> tuple[int, str, int]:
    """(peak envelope bytes, binding phase, alloc estimate bytes)."""

    from woof import domain_wizard as dw

    phases = dw._sizing_phases(
        exp, free_bytes=budget.free_bytes, machine=None, source=source,
        forcing_interval_seconds=dw.source_forcing_interval_seconds(source),
        vram_gib=budget.vram_gib, profile=budget.device_profile,
        forcing_intervals=None)
    return (int(phases.peak_envelope_bytes), str(phases.binding_phase),
            int(phases.forecast.alloc_estimate_bytes))


def _fit_target_bytes(budget: _Budget) -> int:
    from woof.core.preflight import EXTERNAL_MARGIN_BYTES
    from woof.domain_wizard import fit_headroom_bytes

    usable = int(budget.free_bytes) - EXTERNAL_MARGIN_BYTES
    return usable - fit_headroom_bytes(usable)


# --------------------------------------------------------------------------
# the planner


def _gib(value: float) -> float:
    return round(float(value) / (1024 ** 3), 3)


def _domain_limit(max_domains: int | None) -> int:
    from woof.namelist_compat import MAX_DOMAINS

    limit = MAX_DOMAINS
    if max_domains is not None:
        if isinstance(max_domains, bool) or int(max_domains) < 1:
            raise PlanRefused("--max-domains must be a positive integer")
        limit = min(limit, int(max_domains))
    return limit


def _limit_refusal(layout: _Layout, limit: int, max_domains: int | None
                   ) -> DomainLimitRefused:
    from woof.namelist_compat import MAX_DOMAINS

    parents = len(layout.ratios)
    leaves = len(layout.leaves)
    which = (f"--max-domains {max_domains}"
             if max_domains is not None and max_domains < MAX_DOMAINS
             else f"WRF's MAX_DOMAINS ({MAX_DOMAINS})")
    return DomainLimitRefused(
        f"the sites need {leaves} leaf nest(s) at {layout.spacings[-1]:g} m "
        f"(largest leaf side {layout.cap} cells) plus at least {parents} "
        f"parent domain(s), more than the {limit} domains {which} allows in "
        "one run.  Use --topology wrf-tiles, which runs one parent and any "
        "number of offline child tiles, or coarsen --dx-m, narrow "
        "--corridor-km, or plan a smaller set of sites.")


def _ensure_sources(source: str | None) -> str:
    from woof import domain_wizard as dw

    try:
        resolved = dw.resolve_source(source or DEFAULT_SOURCE)
    except ValueError as error:
        raise PlanRefused(str(error)) from error
    if resolved in _REFUSED_SOURCES:
        raise PlanRefused(
            f"--source {resolved}: its route reads namelists and a target "
            "document that only `woof domain` writes; plan wrf-nests from "
            "a global source (gfs, ecmwf-open-data, era5, ...)")
    from woof.runplan import drivability_for

    if drivability_for(resolved).get("requires_source_root"):
        raise PlanRefused(
            f"--source {resolved} is prepared from bytes already on disk and "
            "needs a source_root this planner does not take; emit it with "
            "`woof domain --data-dir` instead")
    return resolved


def build_plan(sites: SiteSet, *, outdir: Path, dx_m: float = 100.0,
               corridor_km: float = 2.0, parent_dx_m: float | None = None,
               start: str | None = None, hours: float = 24.0,
               source: str | None = None, card: str | None = None,
               vram_gib: float | None = None, max_domains: int | None = None,
               nz: int | None = None, sites_path: Path | None = None,
               name: str | None = None,
               now: datetime | None = None) -> Plan:
    """Emit the domains for ``sites`` under ``outdir`` and return the plan
    (already written to ``outdir/plan.json``).

    ``sites_path`` is the sites document the plan binds (``sites_ref``);
    without it a copy of ``sites`` is written to ``outdir/sites.json``.
    ``name`` names the emitted configuration (default from the sites file
    and ``dx_m``).  ``now`` fixes the clock the default start reads.
    """

    plan, _ = _build(sites, outdir=outdir, dx_m=dx_m,
                     corridor_km=corridor_km, parent_dx_m=parent_dx_m,
                     start=start, hours=hours, source=source, card=card,
                     vram_gib=vram_gib, max_domains=max_domains, nz=nz,
                     sites_path=sites_path, name=name, now=now)
    return plan


def _build(sites: SiteSet, *, outdir: Path, dx_m: float, corridor_km: float,
           parent_dx_m: float | None, start: str | None, hours: float,
           source: str | None, card: str | None, vram_gib: float | None,
           max_domains: int | None, nz: int | None, sites_path: Path | None,
           name: str | None, now: datetime | None) -> tuple[Plan, dict]:
    """:func:`build_plan`, also returning the sizing record ``main`` prints."""

    from woof import domain_wizard as dw

    outdir = Path(outdir)
    if len(sites) == 0:
        raise PlanRefused("the sites document has no sites")
    dx_m = float(dx_m)
    if not math.isfinite(dx_m) or dx_m <= 0:
        raise PlanRefused(f"--dx-m {dx_m!r} must be a positive spacing")
    if not math.isfinite(float(hours)) or float(hours) <= 0:
        raise PlanRefused(f"--hours {hours!r} must be positive")
    if nz is not None and nz < 4:
        raise PlanRefused("--nz must be at least 4 (the vertical stencil width)")
    corridor_km = float(corridor_km)
    if not math.isfinite(corridor_km) or corridor_km <= 0:
        raise PlanRefused(f"--corridor-km {corridor_km!r} must be a positive "
                          "half-width in kilometres")
    if parent_dx_m is None:
        parent_dx_m = dx_m * DEFAULT_PARENT_FACTOR
    ratios = chain_ratios(dx_m, float(parent_dx_m))
    limit = _domain_limit(max_domains)
    source_defaulted = source is None
    source = _ensure_sources(source)
    start_defaulted = start is None
    start_time = parse_start(start, now=now)
    budget = _resolve_budget(card, vram_gib)
    target = _fit_target_bytes(budget)
    if target <= 0:
        raise VramRefused(
            f"a {budget.vram_gib:g} GiB card leaves no budget for a forecast "
            f"({budget.free_bytes / 1024 ** 3:.2f} GiB assumed free, less the "
            "external margin and fit headroom); declare a larger --card or "
            "--vram-gib")
    notes: list[str] = []
    profile = dw.resolved_physics_profile(source, None)
    corridor_m = float(corridor_km) * 1000.0
    clearance = dw._CLEARANCE_ROWS
    margin_m = max(corridor_m, clearance * dx_m)
    if margin_m > corridor_m:
        notes.append(f"corridor margin raised from {corridor_m:g} m to "
                     f"{margin_m:g} m so every site clears the leaf's "
                     f"{clearance}-row boundary and blend zone")
    arrays = sites.as_arrays()
    projection, x, y = choose_projection(arrays["lat"], arrays["lon"], dx_m)
    min_axis = dw.polygon_minimum_axis(profile)
    leaf_ratio = ratios[-1]
    min_cap = math.ceil(2.0 * margin_m / dx_m) + 2 * leaf_ratio + 1
    cap = max(DEFAULT_LEAF_MAX_CELLS, min_cap)

    stem = _slug(Path(sites_path).stem if sites_path is not None else "sites")
    name = name or f"{stem}-nests-{dx_m:g}m"
    config_path = outdir / f"{name}.toml"
    wps_path = outdir / f"{name}.namelist.wps"
    leaf_notes: list[str] = []
    leaf_output, output_note = _leaf_output(leaf_notes)

    attempts: list[dict] = []
    while True:
        layout = build_layout(x, y, dx_m=dx_m, ratios=ratios,
                              margin_m=margin_m, cap=cap, min_axis=min_axis,
                              clearance_rows=clearance, domain_limit=limit)
        if layout.count > limit:
            if not attempts:
                raise _limit_refusal(layout, limit, max_domains)
            last = attempts[-1]
            raise VramRefused(
                f"no layout within {limit} domains fits the card: with leaves "
                f"up to {last['cap']} cells the {last['domains']}-domain tree "
                f"prices at {last['peak_gib']:.2f} GiB against a "
                f"{_gib(target):.2f} GiB fit target on a {budget.vram_gib:g} "
                f"GiB card, and smaller leaves need {layout.count} domains.  "
                "Use --topology wrf-tiles (children run one at a time), "
                "declare a larger --card/--vram-gib, or coarsen --dx-m.")
        root = layout.levels[0][0]
        fetch_notes: list[str] = []
        fetch_hints, case_data = _fetch_and_case(
            source=source, start=start_time, hours=float(hours), name=name,
            projection=projection, root_n=(root.nx, root.ny),
            root_dx_m=layout.spacings[0], outdir=outdir,
            config_path=config_path, notes=fetch_notes)
        text = render_plan_config(
            layout, name=name, start=start_time, hours=float(hours),
            projection=projection, source=source, profile=profile, nz=nz,
            fetch_hints=fetch_hints, case_data=case_data,
            leaf_output=leaf_output)
        try:
            exp = dw.experiment_from_text(text, source=str(config_path))
        except (ValueError, TypeError, KeyError) as error:
            raise PlanRefused(f"the emitted configuration does not load: "
                              f"{error}") from error
        peak, phase, alloc = _price(exp, budget=budget, source=source)
        attempts.append({"cap": cap, "domains": layout.count,
                         "peak_gib": _gib(peak)})
        if peak <= target:
            break
        shrink = min(LEAF_CAP_SHRINK, math.sqrt(target / peak))
        new_cap = int(cap * shrink)
        if new_cap < min_cap or new_cap >= cap:
            raise VramRefused(
                f"the {layout.count}-domain tree prices at {_gib(peak):.2f} GiB "
                f"({phase} phase) against a {_gib(target):.2f} GiB fit target "
                f"on a {budget.vram_gib:g} GiB card, and leaves cannot shrink "
                f"below {min_cap} cells (twice the {margin_m:g} m corridor "
                "margin plus alignment).  Use --topology wrf-tiles, declare a "
                "larger --card/--vram-gib, narrow --corridor-km or coarsen "
                "--dx-m.")
        cap = new_cap

    # Every check that can refuse runs before a byte is written: the
    # ownership and footprints come from the accepted in-memory experiment.
    domains = _plan_domains(exp, layout, sites, x, y, config_path=config_path,
                            wps_path=wps_path, outdir=outdir, name=name)
    wps_text = render_plan_wps(layout, projection, source=source)

    outdir.mkdir(parents=True, exist_ok=True)
    stale_plan = outdir / "plan.json"
    if stale_plan.exists():
        stale_plan.unlink()
    dw._write_atomic(config_path, text)
    dw._write_atomic(wps_path, wps_text)
    written = [config_path, wps_path]
    if source == "era5":
        vtable = outdir / dw._PACKAGED_VTABLE.name
        if not vtable.exists():
            shutil.copyfile(dw._PACKAGED_VTABLE, vtable)
            written.append(vtable)

    # The file on disk, through the file loader every front door uses.
    from woof.experiment import load_experiment

    try:
        loaded = load_experiment(config_path)
    except (ValueError, TypeError, KeyError) as error:
        for path in written:
            path.unlink(missing_ok=True)
        raise PlanRefused(f"{config_path} does not load: {error}") from error
    if len(loaded.domains) != layout.count:
        for path in written:
            path.unlink(missing_ok=True)
        raise PlanRefused(f"{config_path} loads {len(loaded.domains)} domains "
                          f"but the plan laid out {layout.count}")

    if sites_path is None:
        sites_path = dump_sites(sites, outdir / "sites.json")
    les = [f"d{box.grid_id:02d}" for box in layout.domains
           if layout.spacings[box.level] < GRAY_ZONE_DX_M]
    notes.extend([
        f"physics profile {profile} as `woof domain` emits it for {source}",
        ("LES gray-zone recipe (docs/public/LES.md) on "
         f"{', '.join(les)}: bl_pbl_physics=0, km_opt=3, mix_isotropic=1, "
         "cu_physics=0; LES is implemented-unverified" if les else
         "no domain is finer than 1 km, so no LES recipe was written"),
        "[static.highres] enabled for domains at or finer than "
        f"{GRAY_ZONE_DX_M:g} m (terrain, land cover and soil where the "
        "sources reach; on_refuse = error)",
        output_note,
        f"estimated peak VRAM {_gib(peak):.2f} GiB ({phase} phase) against a "
        f"{_gib(target):.2f} GiB fit target on a {budget.vram_gib:g} GiB card"
        + ("" if budget.note is None else f"; {budget.note}"),
    ])
    if source_defaulted:
        notes.append(f"--source not given: {DEFAULT_SOURCE}, the operational "
                     "global forecast published on the 00/06/12/18 UTC cycles "
                     "the default --start reads (woof domain's own default, "
                     "ERA5, cannot reach a recent cycle)")
    if start_defaulted:
        notes.append(f"--start not given: {start_time:%Y-%m-%dT%H}Z, the most "
                     "recent 00/06/12/18 UTC cycle by the clock; the source "
                     "may not have published it yet (GFS posts about 3.5-5 h "
                     "after the cycle), in which case pass an earlier --start")
    notes.extend(leaf_notes)
    notes.extend(fetch_notes)
    if len(attempts) > 1:
        notes.append("leaf size cap shrank for VRAM: " + ", ".join(
            f"{a['cap']} cells -> {a['domains']} domains, {a['peak_gib']:.2f} GiB"
            for a in attempts))
    plan = Plan(topology=TOPOLOGY, dx_m=dx_m,
                start=start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                hours=float(hours), domains=domains, source=source,
                sites_ref=file_ref(Path(sites_path).resolve(),
                                   relative_to=outdir),
                notes=notes)
    dump_plan(plan, outdir / "plan.json")
    summary = {
        "config": config_path, "wps_namelist": wps_path,
        "written": written, "peak_bytes": peak, "target_bytes": target,
        "alloc_bytes": alloc, "binding_phase": phase,
        "vram_gib": budget.vram_gib, "layout": layout, "attempts": attempts,
    }
    return plan, summary


def _plan_domains(exp, layout: _Layout, sites: SiteSet, x: np.ndarray,
                  y: np.ndarray, *, config_path: Path, wps_path: Path,
                  outdir: Path, name: str) -> list[PlanDomain]:
    """One PlanDomain per WRF domain, footprints from WOOF's own grids."""

    from woof.static.projection import grids_from_projection_config

    grids = {int(dc.grid_id): grid for dc, grid in
             zip(exp.domains, grids_from_projection_config(exp))}
    by_id = {box.grid_id: box for box in layout.domains}
    if set(grids) != set(by_id):
        raise PlanRefused("the loaded configuration's domains differ from "
                          "the planned layout")
    arrays = sites.as_arrays()
    site_ids = arrays["site_id"]
    spec_bdy = int(exp.spec_bdy_width)
    run_dir = f"runs/{config_path.stem}"
    domains = []
    for grid_id in sorted(by_id):
        box, grid = by_id[grid_id], grids[grid_id]
        nx, ny = box.nx, box.ny
        corners = [(1.0, 1.0), (float(nx), 1.0), (float(nx), float(ny)),
                   (1.0, float(ny))]
        ring = []
        for ci, cj in corners:
            lat, lon = grid.ij_to_latlon(ci, cj)
            ring.append((round(_wrap180(float(lon)), 6), round(float(lat), 6)))
        ring.append(ring[0])
        leaf = box.level == len(layout.ratios)
        owned: tuple[str, ...] = ()
        extra = {"level": box.level, "nx": nx, "ny": ny,
                 "wrf_parent_grid_id": (box.parent.grid_id
                                        if box.parent is not None else 0)}
        if leaf:
            members = np.asarray(box.members)
            lat = arrays["lat"][members]
            lon = arrays["lon"][members]
            i, j = grid.latlon_to_ij(lat, lon)
            i, j = np.asarray(i), np.asarray(j)
            inside = ((i >= 1 + spec_bdy) & (i <= nx - spec_bdy)
                      & (j >= 1 + spec_bdy) & (j <= ny - spec_bdy))
            if not np.all(inside):
                bad = [str(s) for s in site_ids[members][~inside][:5]]
                raise PlanRefused(
                    f"sites {bad} fall outside the interior of leaf "
                    f"d{grid_id:02d} on WOOF's own grid; refusing rather than "
                    "assign them to a boundary zone")
            owned = tuple(str(s) for s in site_ids[members])
            extra["site_count"] = len(owned)
        domains.append(PlanDomain(
            domain_id=f"d{grid_id:02d}", topology=TOPOLOGY,
            role="child" if leaf else "parent",
            dx_m=layout.spacings[box.level], run_dir=run_dir,
            output_glob=OUTPUT_GLOB.format(grid_id=grid_id), footprint=tuple(ring),
            config=config_path.name, wps_namelist=wps_path.name,
            grid_id=grid_id, parent=None, site_ids=owned, extra=extra))
    return domains


def main(args) -> int:
    """``woof energy plan SITES --topology wrf-nests``."""

    sites_path = Path(args.sites)
    try:
        sites = load_sites(sites_path)
    except (OSError, ContractError) as error:
        print(f"woof energy plan: cannot read {sites_path}: {error}",
              file=sys.stderr)
        return 2
    outdir = Path(args.outdir)
    try:
        # Library code below speaks on stdout (wizard notes, warnings);
        # stdout carries only this command's JSON record.
        with contextlib.redirect_stdout(sys.stderr):
            plan, summary = _build(
                sites, outdir=outdir, dx_m=args.dx_m,
                corridor_km=args.corridor_km, parent_dx_m=args.parent_dx_m,
                start=args.start, hours=args.hours, source=args.source,
                card=args.card, vram_gib=args.vram_gib,
                max_domains=args.max_domains, nz=args.nz,
                sites_path=sites_path, name=None, now=None)
    except (PlanRefused, ContractError, ValueError) as error:
        # ValueError: the wizard's own refusals (fetch hints, coverage,
        # projection) reach here unchanged and are reported as refusals.
        print(f"woof energy plan: refused ({type(error).__name__}): {error}",
              file=sys.stderr)
        print(json.dumps({"command": "energy plan", "topology": TOPOLOGY,
                          "refused": type(error).__name__,
                          "reason": str(error)}, indent=2))
        return 2
    layout: _Layout = summary["layout"]
    record = {
        "command": "energy plan", "topology": TOPOLOGY,
        "plan": str(outdir / "plan.json"),
        "config": str(summary["config"]),
        "wps_namelist": str(summary["wps_namelist"]),
        "written": [str(p) for p in summary["written"]],
        "source": plan.source, "start": plan.start, "hours": plan.hours,
        "dx_chain_m": list(layout.spacings),
        "ratios": list(layout.ratios),
        "domains": layout.count,
        "domains_by_level": [len(level) for level in layout.levels],
        "leaves": len(layout.leaves),
        "sites": len(sites),
        "leaf_max_cells": layout.cap,
        "vram": {"card_gib": summary["vram_gib"],
                 "peak_envelope_gib": _gib(summary["peak_bytes"]),
                 "fit_target_gib": _gib(summary["target_bytes"]),
                 "alloc_estimate_gib": _gib(summary["alloc_bytes"]),
                 "binding_phase": summary["binding_phase"]},
        "run_dir": plan.domains[0].run_dir,
        "notes": plan.notes,
    }
    print(json.dumps(record, indent=2, default=str))
    return 0


__all__ = ["TOPOLOGY", "build_plan", "main", "PlanRefused",
           "DomainLimitRefused", "VramRefused", "default_start_cycle",
           "parse_start", "chain_ratios", "build_layout", "project_sites",
           "choose_projection"]
