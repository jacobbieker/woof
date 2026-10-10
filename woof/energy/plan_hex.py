"""``woof energy plan --topology hex-swath``: an MPAS corridor mesh.

The one irregular topology in the energy planners.  Instead of rectangles,
this emits ONE variable-resolution MPAS mesh whose fine cells follow the
power-line corridors themselves, coarsening away from them, and the one
limited-area cut of it that a regional hex forecast integrates.

What is written
---------------
``outdir/mesh_spec.json``
    The resolution spec ``rw_mpas_mesh --spec`` (and ``woof mesh --spec``)
    reads unchanged: a global background plus one row per refinement shape.
    It is the registered ``woof hex mesh-plan --point`` ladder recipe
    (:mod:`woof.hex.mesh_point`) generalised from nested caps to corridors:
    each rung halves the spacing, each rung's ramp is ``18 x`` its own
    spacing, and each rung reaches ``3`` ramps of the next finer rung past
    it.  The finest rung reaches ``corridor_km + ramp`` from every site, so
    the requested spacing is flat out to ``corridor_km`` on both sides of
    every line.  Shapes are DATA rows of the three kinds the generator
    already reads: a line chain becomes ``polygon`` rows (stadium rings from
    :func:`woof.hex.swath.geometry.swath_ring`, split where a ring would
    fold), a point site or a cluster narrower than the rung's reach becomes
    one ``cap``.  Overlapping rows combine by the generator's own rule (the
    finest request wins at every point), so the union of corridors needs no
    polygon clipper.
``outdir/cull_region.json``
    The single ``polygon`` row ``woof hex cull --region`` cuts with: the
    convex hull of every site, buffered by the shipped ``1.35 x`` cull pad
    (:data:`woof.hex.mesh_point.DEFAULT_CULL_PAD_SCALE`) of the finest
    rung's reach.  Convex, so it can never self-intersect; the culler adds
    MPAS's seven boundary-relaxation rings outside it.
``outdir/plan.json``
    One ``PlanDomain``: ``topology="hex-swath"``, ``role="mesh"``, every
    site, the cull ring as its footprint, and ``extra["commands"]`` -- the
    ``woof`` argv lists ``woof energy run`` executes with cwd = the plan
    directory.

The gates, and which one binds
------------------------------
* **Timestep anchor / Courant** (:mod:`woof.hex.dt_admission`,
  :class:`woof.hex.timestep_admission.CourantPolicy`).  The forecast door
  refuses a mesh whose finest ``dcEdge`` admits no ANCHORED timestep.  The
  smallest anchored timestep is 5 s, so ``min dcEdge >= 5 s x 125 m/s / 0.9
  = 694 m``; delivered cells run as fine as 0.848 x their request
  (:data:`woof.hex.mesh_spec_gates.DELIVERED_SPACING_P05`), so a request
  below about 819 m is refused HERE, naming the floor and pointing at
  ``--topology wrf-tiles``.  This is the gate that binds for 50 m and 100 m.
  The numbers are read from those modules at run time, never restated.
* **Transition band / smoothness** (:mod:`woof.hex.mesh_spec_gates`,
  ``woof/data/mpas/mesh-sizing.json``).  The steepest requested spacing
  gradient of the emitted spec is estimated from its own density formula
  (``rw-mpas`` ``density.rs``: a ``tanh`` blend per region, finest wins) and
  refused above the build's ``12.25 %/cell`` ceiling or ``woof mesh``'s
  provisional ``3.06 %/cell`` bound.  When ``rw_mpas_mesh`` is staged the
  plan also runs its ``--dry-run`` through :func:`woof.hex.swath.sizing.
  dry_run`, which applies the transition-band gate on the generator's own
  measurement; the first command repeats it at run time.
* **Dual-edge floor** cannot be decided from a spec
  (:func:`woof.hex.mesh_spec_gates.short_dual_edge_exposure`); for a
  generated mesh it is 3.7e-7 m and cannot bind.  ``woof hex mesh-check``
  measures it after the build and after the cull.
* **Halving ladder** (:func:`woof.hex.mesh_point.ladder_rungs`): the
  generator snaps a request onto ``background / 2^k`` always finer, so the
  background is chosen as ``dx * 2^k`` nearest 120 km, and an explicit
  ``--parent-dx-m`` off that ladder is refused naming the rungs either side.

The capacity estimate
---------------------
Only the CULL reaches a card.  Its cells are an area integral of the spec's
own spacing field over the cut plus the seven boundary rings (``basis:
area_integral``, the convention of :func:`woof.hex.swath.geometry.
predicted_cells_in`), evaluated on a regular grid in an azimuthal-
equidistant projection about the sites.  Bytes come from the limited-area
row of :mod:`woof.hex.device_admission` for the card (a measured
96,582 B/cell over a fixed core on the 32 GiB part, plus that row's
margin).  ``--vram-gib`` without a card is priced on the most demanding
known card profile and the plan says so; with neither, the reference
32 GiB card is used and the plan says so.

What runs, and what does not yet
--------------------------------
``extra["commands"]`` holds every stage that a ``woof`` front door can run
from these documents today: price the spec through the generator with its
gates (``woof hex mesh-plan``), build the global parent pair
(``woof mesh --spec``), check it, cut the limited area (``woof hex cull``)
and check that.  The stages after the cut -- the culled native-free
vertical artifact, a runtime mesh row for ``woof hex forecast --mesh``,
regional initial and boundary conditions, and the forecast that writes
``cuda-history.*.nc`` -- are reachable from the CLI only through ``woof hex
mesh-plan --point --generate`` (a cap, not a corridor) and, for the met
source, only for HRRR (CONUS).  They are listed in
``extra["blocked_stages"]`` with what blocks each, and ``extra["runnable"]``
is ``"mesh-and-cull"``: a run of this plan produces the corridor mesh and
its cut, and extraction refuses on the missing history rather than reading
anything else.

Python boundary (``docs/dev/static-rust-port.md``): this module plans.  Its
numpy work is per-site vectors and one bounded estimation grid (at most
:data:`MAX_ESTIMATE_POINTS` points), never model data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from woof.energy.contracts import (
    ContractError,
    Plan,
    PlanDomain,
    SiteSet,
    dump_plan,
    file_ref,
    load_sites,
)
from woof.hex.errors import MpasPortError
from woof.hex.mesh_point import (
    DEFAULT_BACKGROUND_KM,
    DEFAULT_CULL_PAD_SCALE,
    DEFAULT_LEVELS,
    DEFAULT_RING_FACTOR,
    DEFAULT_TRANSITION_FACTOR,
)
from woof.hex.swath.geometry import (
    BOUNDARY_RINGS as SWATH_BOUNDARY_RINGS,
    EARTH_RADIUS_KM as SWATH_EARTH_RADIUS_KM,
    HEXAGON_AREA_FACTOR as SWATH_HEXAGON_AREA_FACTOR,
)

TOPOLOGY = "hex-swath"

#: Where ``woof energy run`` runs the commands' outputs, relative to the plan.
RUN_DIR = "runs/hex"
#: The history files ``woof hex forecast --out runs/hex/forecast`` writes
#: (``run_cuda_v841_forecast.py``: ``cuda-history.<label>.nc``), relative to
#: :data:`RUN_DIR`.
OUTPUT_GLOB = "forecast/cuda-history.*.nc"
#: File stem of the limited-area cut inside :data:`RUN_DIR`.
CULL_NAME = "corridor"
#: Background spacing the default ladder is anchored nearest to (the
#: registered recipe's 120 km, :data:`woof.hex.mesh_point.DEFAULT_BACKGROUND_KM`).
BACKGROUND_TARGET_KM = DEFAULT_BACKGROUND_KM
#: The registered recipe's ramp (``18 x`` a rung's own spacing,
#: :mod:`woof.hex.mesh_point`) is the SHORTEST ramp tried.  It reads about
#: 12 %/cell -- inside the build's 12.25 %/cell transition-band ceiling but
#: four times ``woof mesh``'s provisional 3.06 %/cell smoothness bound,
#: which the point door never meets because it drives ``rw_mpas_mesh``
#: directly.  This plan builds through ``woof mesh``, so the ramp is widened
#: until the estimate sits :data:`GRADIENT_MARGIN` inside the strictest bound
#: rather than passing ``--allow-rough-mesh``.
MIN_TRANSITION_FACTOR = DEFAULT_TRANSITION_FACTOR
MAX_TRANSITION_FACTOR = 400.0
GRADIENT_MARGIN = 0.9
#: How many ramps of the next finer rung each rung reaches past it (the
#: registered recipe's 3).
RING_FACTOR = DEFAULT_RING_FACTOR
#: The finest rung's blend centre sits this many ramps outside
#: ``corridor_km``, so the requested spacing is within about 5 % of
#: ``dx_m`` everywhere within ``corridor_km`` of a site (``tanh(1.5)``).
FLAT_TRANSITIONS = 1.5
#: The shipped cull pad: the cut sits at this multiple of the finest rung's
#: reach (``evidence/nest-ratio-20260827``: the domain-size knee).
CULL_PAD_SCALE = DEFAULT_CULL_PAD_SCALE
#: MPAS's boundary-relaxation zone, cells (``bdyMaskCell`` 1..7).
BOUNDARY_RINGS = SWATH_BOUNDARY_RINGS
#: Vertical levels every hex anchor, admission row and vertical spec runs at.
HEX_LEVELS = DEFAULT_LEVELS
#: A line chain is split where consecutive sites are farther apart than this
#: (separate parts of a multi-part asset), so no ring bridges open ground.
CHAIN_GAP_KM = 25.0
#: Cap on the estimation grid; past it the grid coarsens and the plan says so.
MAX_ESTIMATE_POINTS = 4_000_000
#: Polyline simplification tolerance as a fraction of ``corridor_km``; every
#: corridor ring is widened by the tolerance so the simplification can only
#: add cells, never uncover a site.
SIMPLIFY_FRACTION = 0.1
#: A cluster becomes one cap at a rung when the cap's fully refined area is
#: at most this multiple of its corridor rings' (see ``corridor_regions``).
CAP_AREA_RATIO = 1.5
#: Points per quarter circle when a convex hull is buffered.
HULL_ARC_POINTS = 16
#: A corridor ring is split at any vertex turning more than this, and every
#: ring is widened by ``1 / cos(turn / 2)`` of its largest remaining turn:
#: ``swath_ring`` offsets along the bisector without a miter, so a bend
#: would otherwise pull the ring inside the requested reach.
MAX_RING_TURN_DEG = 15.0

#: The sphere the swath geometry this module builds on measures with.
EARTH_RADIUS_KM = SWATH_EARTH_RADIUS_KM
HEXAGON_AREA_FACTOR = SWATH_HEXAGON_AREA_FACTOR
MIB = 1024 ** 2


class HexPlanRefusal(RuntimeError):
    """``--topology hex-swath`` cannot plan this request, and says why."""


# --------------------------------------------------------------------------
# gates read from the hex code


def spacing_floor_m(dx_m: float) -> dict[str, Any]:
    """The finest requested spacing the hex forecast route admits today.

    Derived from the anchored timesteps, the Courant policy and the
    delivered-spacing tail, all read from :mod:`woof.hex` at call time.
    """

    from woof.hex import convection_admission, dt_admission
    from woof.hex.mesh_spec_gates import DELIVERED_SPACING_P05
    from woof.hex.timestep_admission import CourantPolicy

    policy = CourantPolicy()
    # The convection ruling, spelled the way the anchor table keys it
    # (``None`` / ``"gf"``, as ``woof.hex.mesh_point.choose_timestep`` reads it).
    ruling = convection_admission.convection_for_spacing(dx_m)
    scheme = None if ruling == convection_admission.SCHEME_OFF else "gf"
    anchored = sorted(
        anchor.dt_seconds for anchor in dt_admission.ADMITTED_TIMESTEPS.values()
        if anchor.cumulus_scheme == scheme
        and float(anchor.surface_pbl_seconds) == float(anchor.dt_seconds))
    if not anchored:
        raise HexPlanRefusal(
            "the hex timestep anchor table holds no "
            f"{'convection-off' if scheme is None else 'Grell-Freitas'} "
            "anchor, so no hex mesh can be admitted at this spacing")
    smallest_dt = float(anchored[0])
    min_dc_edge_m = (smallest_dt * policy.max_characteristic_speed_m_s
                     / policy.safety_factor)
    floor_m = min_dc_edge_m / DELIVERED_SPACING_P05
    return {
        "floor_dx_m": floor_m,
        "min_dc_edge_m": min_dc_edge_m,
        "smallest_anchored_dt_s": smallest_dt,
        "cumulus": "off" if scheme is None else scheme,
        "max_characteristic_speed_m_s": policy.max_characteristic_speed_m_s,
        "courant_safety_factor": policy.safety_factor,
        "delivered_spacing_p05": DELIVERED_SPACING_P05,
        "gate": ("woof.hex.dt_admission.ADMITTED_TIMESTEPS + "
                 "woof.hex.timestep_admission.CourantPolicy, applied by "
                 "woof hex forecast at bind"),
    }


def _check_floor(dx_m: float) -> dict[str, Any]:
    floor = spacing_floor_m(dx_m)
    if dx_m < floor["floor_dx_m"]:
        raise HexPlanRefusal(
            f"--dx-m {dx_m:g} is below the finest spacing the hex route "
            f"admits today, about {floor['floor_dx_m']:.0f} m.  The hex "
            "forecast refuses a mesh whose finest edge admits no anchored "
            f"timestep: the smallest anchored "
            f"(cumulus {floor['cumulus']}) timestep is "
            f"{floor['smallest_anchored_dt_s']:g} s, and at "
            f"{floor['max_characteristic_speed_m_s']:g} m/s with a "
            f"{floor['courant_safety_factor']:g} Courant safety factor that "
            f"needs min dcEdge >= {floor['min_dc_edge_m']:.0f} m; delivered "
            f"cells run as fine as {floor['delivered_spacing_p05']:g} x their "
            f"request, hence {floor['floor_dx_m']:.0f} m "
            f"({floor['gate']}).  For {dx_m:g} m along the corridors use "
            "--topology wrf-tiles (one parent run plus offline child tiles "
            "at any spacing), or ask hex-swath for a coarser --dx-m.")
    return floor


def _smoothness_bounds() -> dict[str, Any]:
    from woof.hex.mesh_spec_gates import MAX_GRADIENT_PER_CELL

    bounds: dict[str, Any] = {
        "transition_band_ceiling_percent_per_cell":
            MAX_GRADIENT_PER_CELL * 100.0,
    }
    try:
        from woof.mpas_mesh import load_sizing

        smooth = load_sizing().smoothness
        bounds["woof_mesh_refuse_above_percent_per_cell"] = \
            smooth.refuse_above_percent_per_cell
        bounds["woof_mesh_warn_above_percent_per_cell"] = \
            smooth.warn_above_percent_per_cell
        bounds["woof_mesh_bound_status"] = smooth.status
    except Exception as error:  # the sizing table is package data
        raise HexPlanRefusal(
            f"the MPAS mesh sizing table cannot be read ({error}), so the "
            "smoothness bound woof mesh applies cannot be checked") from error
    return bounds


# --------------------------------------------------------------------------
# the ladder


def background_km_for(dx_m: float, parent_dx_m: float | None
                      ) -> tuple[float, list[float]]:
    """Background spacing and the halving rungs down to ``dx_m``, km."""

    from woof.hex.mesh_point import ladder_rungs

    fine_km = dx_m / 1000.0
    if parent_dx_m is not None:
        background = parent_dx_m / 1000.0
    else:
        k = max(1, round(math.log2(BACKGROUND_TARGET_KM / fine_km)))
        background = fine_km * 2.0 ** k
    try:
        rungs = ladder_rungs(background, dx_m)
    except MpasPortError as error:
        raise HexPlanRefusal(
            f"--parent-dx-m {parent_dx_m:g} cannot be the hex background for "
            f"--dx-m {dx_m:g}: {error}") from error
    return background, rungs


@dataclass(frozen=True)
class Rung:
    spacing_km: float
    transition_km: float
    reach_km: float      # signed-distance zero of the rung's blend


def ladder(rungs_km: Sequence[float], corridor_km: float,
           widen_km: float = 0.0,
           transition_factor: float = MIN_TRANSITION_FACTOR) -> list[Rung]:
    """Finest rung first; the finest is flat out to ``corridor_km``."""

    finest_first = sorted(rungs_km)
    out: list[Rung] = []
    reach = 0.0
    for index, spacing in enumerate(finest_first):
        transition = transition_factor * spacing
        if index == 0:
            reach = corridor_km + widen_km + FLAT_TRANSITIONS * transition
        else:
            reach = reach + RING_FACTOR * out[-1].transition_km
        out.append(Rung(spacing, transition, reach))
    return out


def choose_ladder(rungs_km: Sequence[float], corridor_km: float,
                  widen_km: float, background_km: float,
                  bound_percent: float) -> tuple[list[Rung], float, float]:
    """The shortest ramp whose steepest gradient clears ``bound_percent``
    with :data:`GRADIENT_MARGIN`: ``(rungs, transition_factor, gradient)``.

    The gradient falls as one over the ramp factor, so the first factor is
    solved from the registered ramp's reading and then stepped until the
    estimate clears.
    """

    target = GRADIENT_MARGIN * bound_percent
    first = ladder(rungs_km, corridor_km, widen_km, MIN_TRANSITION_FACTOR)
    reading = steepest_gradient_percent(first, background_km)
    factor = max(MIN_TRANSITION_FACTOR,
                 math.floor(MIN_TRANSITION_FACTOR * reading / target))
    while True:
        rungs = ladder(rungs_km, corridor_km, widen_km, factor)
        gradient = steepest_gradient_percent(rungs, background_km)
        if gradient <= target:
            return rungs, factor, gradient
        if factor >= MAX_TRANSITION_FACTOR:
            raise HexPlanRefusal(
                f"no ramp up to {MAX_TRANSITION_FACTOR:g} x the rung spacing "
                f"brings the corridor ladder under {target:.2f} %/cell "
                f"(it reads {gradient:.2f}); the gradient bound in the mesh "
                "sizing table may have been edited")
        factor = min(MAX_TRANSITION_FACTOR, factor + 1.0)


def spacing_at_distance(d_km: np.ndarray, rungs: Sequence[Rung],
                        background_km: float) -> np.ndarray:
    """``rw-mpas`` ``density.rs`` spacing for regions sharing one distance.

    ``s = d - reach``; each region blends inverse spacing with
    ``0.5 * (1 - tanh(s / width))`` and the finest request wins.
    """

    d = np.asarray(d_km, dtype=np.float64)
    inv = np.full(d.shape, 1.0 / background_km)
    back = 1.0 / background_km
    for rung in rungs:
        blend = 0.5 * (1.0 - np.tanh((d - rung.reach_km) / rung.transition_km))
        np.maximum(inv, back + (1.0 / rung.spacing_km - back) * blend, out=inv)
    return 1.0 / inv


def steepest_gradient_percent(rungs: Sequence[Rung], background_km: float
                              ) -> float:
    """Steepest ``|dh/dd|`` across a straight corridor, percent per cell."""

    step = rungs[0].spacing_km / 8.0
    far = rungs[-1].reach_km + 6.0 * rungs[-1].transition_km
    d = np.arange(0.0, far + step, step)
    h = spacing_at_distance(d, rungs, background_km)
    return float(np.max(np.abs(np.diff(h)) / step) * 100.0)


# --------------------------------------------------------------------------
# local projection (azimuthal equidistant on the swath sphere)


@dataclass(frozen=True)
class _Aeqd:
    lat0: float
    lon0: float

    def forward(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        phi = np.radians(np.asarray(lat, dtype=np.float64))
        lam = np.radians(np.asarray(lon, dtype=np.float64))
        phi0 = math.radians(self.lat0)
        dlam = lam - math.radians(self.lon0)
        cos_c = (math.sin(phi0) * np.sin(phi)
                 + math.cos(phi0) * np.cos(phi) * np.cos(dlam))
        c = np.arccos(np.clip(cos_c, -1.0, 1.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where(c < 1e-12, 1.0, c / np.sin(c))
        x = EARTH_RADIUS_KM * k * np.cos(phi) * np.sin(dlam)
        y = EARTH_RADIUS_KM * k * (math.cos(phi0) * np.sin(phi)
                                   - math.sin(phi0) * np.cos(phi)
                                   * np.cos(dlam))
        return x, y

    def inverse(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        phi0 = math.radians(self.lat0)
        rho = np.hypot(x, y)
        c = rho / EARTH_RADIUS_KM
        with np.errstate(invalid="ignore", divide="ignore"):
            sin_phi = (np.cos(c) * math.sin(phi0)
                       + np.where(rho > 0.0, y * np.sin(c) * math.cos(phi0)
                                  / rho, 0.0))
            lat = np.degrees(np.arcsin(np.clip(sin_phi, -1.0, 1.0)))
            lam = np.arctan2(x * np.sin(c),
                             rho * math.cos(phi0) * np.cos(c)
                             - y * math.sin(phi0) * np.sin(c))
        lon = (np.degrees(lam) + self.lon0 + 180.0) % 360.0 - 180.0
        return lat, lon


def _centre(lat: np.ndarray, lon: np.ndarray) -> tuple[float, float]:
    phi = np.radians(lat)
    lam = np.radians(lon)
    x = np.mean(np.cos(phi) * np.cos(lam))
    y = np.mean(np.cos(phi) * np.sin(lam))
    z = np.mean(np.sin(phi))
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-9:
        raise HexPlanRefusal("the sites are spread over the whole sphere; a "
                             "limited-area corridor mesh needs a region")
    return (math.degrees(math.atan2(z, math.hypot(x, y))),
            math.degrees(math.atan2(y, x)))


# --------------------------------------------------------------------------
# corridors


@dataclass
class Chain:
    """Sites of one asset in order, as projected km and lat/lon."""

    asset_id: str
    site_ids: list[str]
    lat: np.ndarray
    lon: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @property
    def is_point(self) -> bool:
        return len(self.site_ids) == 1 or float(
            np.max(np.hypot(self.x - self.x[0], self.y - self.y[0]))) < 1e-6


def corridor_chains(sites: SiteSet, projection: _Aeqd) -> list[Chain]:
    """Group sites into ordered per-asset chains, split at gaps."""

    groups: dict[str, list[int]] = {}
    for index, site in enumerate(sites.sites):
        groups.setdefault(site.asset_id, []).append(index)
    arrays = sites.as_arrays()
    x_all, y_all = projection.forward(arrays["lat"], arrays["lon"])
    chains: list[Chain] = []
    for asset_id, members in groups.items():
        chainage = [sites.sites[i].chainage_m for i in members]
        if all(c is not None for c in chainage):
            members = [m for _, m in sorted(zip(chainage, members))]
        start = 0
        for k in range(1, len(members) + 1):
            split = k == len(members)
            if not split:
                a, b = members[k - 1], members[k]
                split = math.hypot(x_all[b] - x_all[a],
                                   y_all[b] - y_all[a]) > CHAIN_GAP_KM
            if split:
                part = members[start:k]
                chains.append(Chain(
                    asset_id=asset_id,
                    site_ids=[sites.sites[i].site_id for i in part],
                    lat=arrays["lat"][part], lon=arrays["lon"][part],
                    x=x_all[part], y=y_all[part]))
                start = k
    return chains


def _simplify(x: np.ndarray, y: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker indices kept (both ends always)."""

    n = len(x)
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        dx, dy = x[j] - x[i], y[j] - y[i]
        length = math.hypot(dx, dy)
        px, py = x[i + 1:j] - x[i], y[i + 1:j] - y[i]
        if length < 1e-12:
            dist = np.hypot(px, py)
        else:
            dist = np.abs(px * dy - py * dx) / length
        k = int(np.argmax(dist))
        if dist[k] > tolerance:
            keep[i + 1 + k] = True
            stack.append((i, i + 1 + k))
            stack.append((i + 1 + k, j))
    return np.flatnonzero(keep)


def _clusters(chains: Sequence[Chain], link_km: float) -> list[list[int]]:
    """Single-linkage groups of chains whose sites come within ``link_km``.

    Linking is done on a ``link_km / 4`` grid: chains sharing a cell are
    joined, then cells within ``link_km`` plus a cell diagonal, so the work
    scales with the area covered rather than the site count squared.  It
    may link chains up to about 1.7 ``link_km`` apart; the groups only
    decide where a rung may use one cap, so that moves no site's
    refinement.
    """

    from scipy.spatial import cKDTree

    cell = link_km / 4.0
    owners, points = [], []
    for i, chain in enumerate(chains):
        xy_chain = np.column_stack([chain.x, chain.y])
        _, first = np.unique(np.floor(xy_chain / cell), axis=0,
                             return_index=True)
        points.append(xy_chain[np.sort(first)])
        owners.append(np.full(len(first), i))
    owner = np.concatenate(owners)
    xy = np.concatenate(points)
    parent = list(range(len(chains)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Chains sharing a grid cell are linked outright (many point chains --
    # towers, turbines -- collapse here), then occupied cells whose centres
    # are within the link distance plus a cell diagonal are linked.
    occupied, inverse = np.unique(np.floor(xy / cell), axis=0,
                                  return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    first_owner = np.full(len(occupied), -1)
    for k in np.argsort(inverse, kind="stable"):
        c = inverse[k]
        if first_owner[c] < 0:
            first_owner[c] = int(owner[k])
        else:
            union(int(owner[k]), int(first_owner[c]))
    centres = (occupied + 0.5) * cell
    for a, b in cKDTree(centres).query_pairs(link_km + math.sqrt(2.0) * cell):
        union(int(first_owner[a]), int(first_owner[b]))
    groups: dict[int, list[int]] = {}
    for i in range(len(chains)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _ring_regions(chain: Chain, reach_km: float, tolerance_km: float
                  ) -> list[list[tuple[float, float]]]:
    """Stadium rings ``[lat, lon]`` covering ``reach_km`` of the chain."""

    from woof.hex.swath.errors import SwathRefusal
    from woof.hex.swath.geometry import swath_ring

    kept = _simplify(chain.x, chain.y, tolerance_km)
    # Split at sharp turns: the bisector offset has no miter, so a piece is
    # only kept whole while every interior turn is under MAX_RING_TURN_DEG.
    x, y = chain.x[kept], chain.y[kept]
    heading = np.arctan2(np.diff(y), np.diff(x))
    turn = np.degrees(np.abs((np.diff(heading) + np.pi) % (2.0 * np.pi)
                             - np.pi))                     # at kept[1:-1]
    cuts = [0] + [k + 1 for k in np.flatnonzero(turn > MAX_RING_TURN_DEG)] \
        + [len(kept) - 1]
    pending: list[tuple[list[tuple[float, float]], float]] = []
    for start, stop in zip(cuts, cuts[1:]):
        if stop <= start:
            continue
        inner = turn[start:stop - 1]
        worst = float(np.max(inner)) if len(inner) else 0.0
        widen = 1.0 / math.cos(math.radians(0.5 * worst))
        pending.append(([(float(chain.lat[kept[k]]), float(chain.lon[kept[k]]))
                         for k in range(start, stop + 1)], reach_km * widen))
    rings: list[list[tuple[float, float]]] = []
    while pending:
        part, reach = pending.pop()
        try:
            rings.append([tuple(p) for p in swath_ring(
                part, [reach] * len(part), cap_points=8)])
        except SwathRefusal as error:
            if len(part) <= 2:
                raise HexPlanRefusal(
                    f"asset {chain.asset_id}: a two-site corridor piece has "
                    f"no valid ring at {reach:.1f} km ({error})") from error
            middle = len(part) // 2
            pending.append((part[:middle + 1], reach))
            pending.append((part[middle:], reach))
    return rings


@dataclass
class RungShapes:
    """One rung's rows: corridor rings (distance field from chain points)
    and caps (``centre_xy``, ``radius``)."""

    rung: Rung
    rings: list[list[tuple[float, float]]] = field(default_factory=list)
    tube_chains: list[int] = field(default_factory=list)
    caps: list[tuple[tuple[float, float], tuple[float, float], float]] = \
        field(default_factory=list)   # (lat/lon, xy, radius_km)


def corridor_regions(chains: Sequence[Chain], clusters: Sequence[Sequence[int]],
                     rungs: Sequence[Rung], projection: _Aeqd,
                     tolerance_km: float) -> list[RungShapes]:
    """The refinement rows of every rung.

    A point chain is a cap.  A cluster (or the whole site set) becomes one
    cap about its centre when that cap's flat zone -- where the rung's
    spacing is fully requested -- is at most :data:`CAP_AREA_RATIO` times the
    corridor rings' flat zone: the cap then costs few extra cells and saves
    many rows.  The finest rung's flat zone is ``corridor_km`` wide, so it
    stays corridor-shaped except for compact clusters.  Every other chain
    is stadium rings.
    """

    def worth_a_cap(members: Sequence[int], radius: float, rung: Rung) -> bool:
        flat = rung.reach_km - FLAT_TRANSITIONS * rung.transition_km
        if flat <= 0.0:
            return True
        tube = sum(2.0 * _chain_length(chains[i]) * flat + math.pi * flat ** 2
                   for i in members)
        return math.pi * (radius + flat) ** 2 <= CAP_AREA_RATIO * tube

    all_x = np.concatenate([c.x for c in chains])
    all_y = np.concatenate([c.y for c in chains])
    whole = _bounding_circle(all_x, all_y)
    out: list[RungShapes] = []
    for rung in rungs:
        shapes = RungShapes(rung)
        if worth_a_cap(range(len(chains)), whole[2], rung):
            shapes.caps.append(_cap(whole, rung.reach_km, projection))
            out.append(shapes)
            continue
        for cluster in clusters:
            cx = np.concatenate([chains[i].x for i in cluster])
            cy = np.concatenate([chains[i].y for i in cluster])
            circle = _bounding_circle(cx, cy)
            if len(cluster) > 1 and worth_a_cap(cluster, circle[2], rung):
                shapes.caps.append(_cap(circle, rung.reach_km, projection))
                continue
            for i in cluster:
                chain = chains[i]
                if chain.is_point:
                    shapes.caps.append(_cap((float(chain.x[0]),
                                             float(chain.y[0]), 0.0),
                                            rung.reach_km, projection))
                else:
                    shapes.rings.extend(_ring_regions(chain, rung.reach_km,
                                                      tolerance_km))
                    shapes.tube_chains.append(i)
        out.append(shapes)
    return out


def _chain_length(chain: Chain) -> float:
    return float(np.sum(np.hypot(np.diff(chain.x), np.diff(chain.y))))


def _bounding_circle(x: np.ndarray, y: np.ndarray
                     ) -> tuple[float, float, float]:
    """Centre of the bounding box and the radius reaching every point."""

    cx = 0.5 * (float(np.min(x)) + float(np.max(x)))
    cy = 0.5 * (float(np.min(y)) + float(np.max(y)))
    return cx, cy, float(np.max(np.hypot(x - cx, y - cy)))


def _cap(circle: tuple[float, float, float], reach_km: float,
         projection: _Aeqd):
    lat, lon = projection.inverse(circle[0], circle[1])
    # A cap at or past the antipode names the complement; the outer rungs
    # of a fine ladder reach most of a hemisphere, so clamp short of it.
    radius = min(circle[2] + reach_km, 0.95 * math.pi * EARTH_RADIUS_KM)
    return ((float(lat), float(lon)), (circle[0], circle[1]), radius)


def mesh_spec_document(shapes: Sequence[RungShapes], background_km: float,
                       name: str) -> dict[str, Any]:
    regions: list[dict[str, Any]] = []
    for rung_shapes in shapes:
        rung = rung_shapes.rung
        for ring in rung_shapes.rings:
            regions.append({
                "shape": {"kind": "polygon",
                          "vertices_deg": [[round(lat, 6), round(lon, 6)]
                                           for lat, lon in ring]},
                "spacing_km": rung.spacing_km,
                "transition_km": rung.transition_km,
            })
        for (lat, lon), _, radius in rung_shapes.caps:
            regions.append({
                "shape": {"kind": "cap",
                          "center_deg": [round(lat, 6), round(lon, 6)],
                          "radius_km": round(radius, 4)},
                "spacing_km": rung.spacing_km,
                "transition_km": rung.transition_km,
            })
    return {"name": name, "background_km": background_km, "regions": regions}


# --------------------------------------------------------------------------
# the cut and the estimate


def _buffered_hull(x: np.ndarray, y: np.ndarray, radius_km: float
                   ) -> np.ndarray:
    """Convex hull of the points dilated by ``radius_km``: (N, 2), CCW."""

    from scipy.spatial import ConvexHull

    pts = np.unique(np.column_stack([x, y]).round(9), axis=0)
    if len(pts) >= 3:
        try:
            pts = pts[ConvexHull(pts, qhull_options="QJ").vertices]
        except Exception:
            pass
    angles = np.linspace(0.0, 2.0 * math.pi, 4 * HULL_ARC_POINTS,
                         endpoint=False)
    circle = radius_km * np.column_stack([np.cos(angles), np.sin(angles)])
    cloud = (pts[:, None, :] + circle[None, :, :]).reshape(-1, 2)
    hull = ConvexHull(cloud)
    return cloud[hull.vertices]


def _inside_convex(polygon: np.ndarray, x: np.ndarray, y: np.ndarray
                   ) -> np.ndarray:
    inside = np.ones(x.shape, dtype=bool)
    n = len(polygon)
    for k in range(n):
        ax, ay = polygon[k]
        bx, by = polygon[(k + 1) % n]
        inside &= (bx - ax) * (y - ay) - (by - ay) * (x - ax) >= 0.0
    return inside


def _densified(chains: Sequence[Chain], indices: Sequence[int], step_km: float
               ) -> np.ndarray:
    parts = []
    for i in indices:
        c = chains[i]
        parts.append(np.column_stack([c.x[:1], c.y[:1]]))
        for k in range(1, len(c.x)):
            length = math.hypot(c.x[k] - c.x[k - 1], c.y[k] - c.y[k - 1])
            n = max(1, int(math.ceil(length / step_km)))
            t = np.arange(1, n + 1) / n
            parts.append(np.column_stack([
                c.x[k - 1] + t * (c.x[k] - c.x[k - 1]),
                c.y[k - 1] + t * (c.y[k] - c.y[k - 1])]))
    return np.concatenate(parts) if parts else np.empty((0, 2))


def _field_spacing(gx: np.ndarray, gy: np.ndarray, shapes: Sequence[RungShapes],
                   chains: Sequence[Chain], background_km: float,
                   step_km: float) -> np.ndarray:
    """The emitted spec's spacing at projected points, km."""

    from scipy.spatial import cKDTree

    back = 1.0 / background_km
    inv = np.full(gx.shape, back)
    tree_cache: dict[tuple[int, ...], np.ndarray] = {}
    for rung_shapes in shapes:
        rung = rung_shapes.rung
        s = np.full(gx.shape, np.inf)
        if rung_shapes.tube_chains:
            key = tuple(rung_shapes.tube_chains)
            if key not in tree_cache:
                pts = _densified(chains, key, step_km)
                tree_cache[key] = cKDTree(pts).query(
                    np.column_stack([gx, gy]))[0]
            s = np.minimum(s, tree_cache[key] - rung.reach_km)
        for _, (cx, cy), radius in rung_shapes.caps:
            s = np.minimum(s, np.hypot(gx - cx, gy - cy) - radius)
        blend = 0.5 * (1.0 - np.tanh(s / rung.transition_km))
        np.maximum(inv, back + (1.0 / rung.spacing_km - back) * blend, out=inv)
    return 1.0 / inv


@dataclass
class CullEstimate:
    polygon_xy: np.ndarray
    ring_latlon: list[tuple[float, float]]
    cut_reach_km: float
    halo_km: float
    spacing_at_cut_km: float
    cells: float
    fine_cells: float
    grid_step_km: float
    grid_points: int
    notes: list[str]


def estimate_cull(chains: Sequence[Chain], shapes: Sequence[RungShapes],
                  rungs: Sequence[Rung], background_km: float,
                  projection: _Aeqd, corridor_km: float) -> CullEstimate:
    all_x = np.concatenate([c.x for c in chains])
    all_y = np.concatenate([c.y for c in chains])
    cut_reach = CULL_PAD_SCALE * rungs[0].reach_km
    cut = _buffered_hull(all_x, all_y, cut_reach)
    notes: list[str] = []
    step = min(corridor_km / 2.0, rungs[0].transition_km / 4.0,
               rungs[0].spacing_km * 2.0)
    # spacing along the cut decides the halo width
    edges = np.vstack([cut, cut[:1]])
    seg = np.hypot(np.diff(edges[:, 0]), np.diff(edges[:, 1]))
    bx, by = [], []
    for k in range(len(cut)):
        n = max(1, int(math.ceil(seg[k] / step)))
        t = np.arange(n) / n
        bx.append(edges[k, 0] + t * (edges[k + 1, 0] - edges[k, 0]))
        by.append(edges[k, 1] + t * (edges[k + 1, 1] - edges[k, 1]))
    bx_all, by_all = np.concatenate(bx), np.concatenate(by)
    h_cut = float(np.max(_field_spacing(bx_all, by_all, shapes, chains,
                                        background_km, step)))
    halo = BOUNDARY_RINGS * h_cut
    outer = _buffered_hull(all_x, all_y, cut_reach + halo)
    xmin, ymin = outer.min(axis=0)
    xmax, ymax = outer.max(axis=0)
    area_box = (xmax - xmin) * (ymax - ymin)
    if area_box / (step * step) > MAX_ESTIMATE_POINTS:
        coarse = math.sqrt(area_box / MAX_ESTIMATE_POINTS)
        notes.append(
            f"cell estimate grid coarsened from {step:.3f} km to "
            f"{coarse:.3f} km to stay within {MAX_ESTIMATE_POINTS:,} points; "
            "the area integral is less exact at that step")
        step = coarse
    xs = np.arange(xmin + 0.5 * step, xmax, step)
    ys = np.arange(ymin + 0.5 * step, ymax, step)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    keep = _inside_convex(outer, gx, gy)
    gx, gy = gx[keep], gy[keep]
    h = _field_spacing(gx, gy, shapes, chains, background_km, step)
    per_point = step * step / (HEXAGON_AREA_FACTOR * h * h)
    cells = float(np.sum(per_point))
    fine = float(np.sum(per_point[h <= rungs[0].spacing_km * 1.05]))
    lat, lon = projection.inverse(cut[:, 0], cut[:, 1])
    ring = [(float(a), float(b)) for a, b in zip(lat, lon)]
    return CullEstimate(cut, ring, cut_reach, halo, h_cut, cells, fine, step,
                        int(len(gx)), notes)


def estimate_parent_cells(shapes: Sequence[RungShapes], chains: Sequence[Chain],
                          background_km: float) -> float:
    """Upper-bound cell count of the GLOBAL graded parent (area by rung).

    Each rung's covered area is the sum of its shapes' areas (corridor
    rings ``2 L r + pi r^2``, caps as spherical caps), overlaps ignored, so
    the figure only errs high.  Informational: the graded generator builds
    from the spec, not from a count, and the first command measures it.
    """

    sphere = 4.0 * math.pi * EARTH_RADIUS_KM ** 2

    def cap_area(r: float) -> float:
        r = min(r, math.pi * EARTH_RADIUS_KM)
        return 2.0 * math.pi * EARTH_RADIUS_KM ** 2 * (
            1.0 - math.cos(r / EARTH_RADIUS_KM))

    def length(i: int) -> float:
        c = chains[i]
        return float(np.sum(np.hypot(np.diff(c.x), np.diff(c.y))))

    areas = []
    for rung_shapes in shapes:
        r = rung_shapes.rung.reach_km
        area = sum(2.0 * length(i) * r + math.pi * r * r
                   for i in rung_shapes.tube_chains)
        area += sum(cap_area(radius) for _, _, radius in rung_shapes.caps)
        areas.append(min(area, sphere))
    cells = 0.0
    previous = 0.0
    for rung_shapes, area in zip(shapes, areas):
        grown = max(area, previous)
        cells += (grown - previous) / (HEXAGON_AREA_FACTOR
                                       * rung_shapes.rung.spacing_km ** 2)
        previous = grown
    cells += (sphere - previous) / (HEXAGON_AREA_FACTOR * background_km ** 2)
    return cells


# --------------------------------------------------------------------------
# capacity


def capacity_verdict(cells: float, card: str | None, vram_gib: float | None
                     ) -> dict[str, Any]:
    """The limited-area admission row applied to the predicted cut."""

    from woof.hex import device_admission
    from woof.hex.mesh_point import CARD_ALIASES, resolve_card

    priced = int(math.ceil(cells))
    notes: list[str] = []
    if card is not None:
        try:
            resolved = resolve_card(card)
        except MpasPortError as error:
            raise HexPlanRefusal(str(error)) from error
        profiles = [resolved["admission_card"]]
        budget_mib = float(resolved["nameplate_mib"])
        basis = f"--card {card} nameplate"
    elif vram_gib is not None:
        profiles = list(device_admission.KNOWN_CARDS)
        budget_mib = float(vram_gib) * 1024.0
        basis = "--vram-gib"
        notes.append(
            "--vram-gib names a budget but not a card, and the hex footprint's "
            "fixed term is a property of the card; priced on the most "
            "demanding known card profile "
            f"({', '.join(profiles)}) so the verdict can only err safe")
    else:
        profiles = ["32gib-170sm"]
        budget_mib = float(CARD_ALIASES["32gib"][2])
        basis = "reference 32 GiB card (no --card or --vram-gib)"
        notes.append(
            "no --card or --vram-gib was given; the cut was priced on the "
            "reference 32 GiB card's limited-area row")
    worst: dict[str, Any] | None = None
    for key in profiles:
        profile = device_admission.KNOWN_CARDS[key]
        model = device_admission.model_for_card(profile,
                                                configuration="limited-area")
        required = model.required_bytes(priced)
        budget_bytes = int(budget_mib * MIB)
        row = {
            "row": device_admission.row_key("limited-area", profile),
            "row_measured": bool(model.measured),
            "cells_priced": priced,
            "predicted_mib": round(model.predict_bytes(priced) / MIB, 1),
            "margin_mib": round(model.margin_bytes() / MIB, 1),
            "required_mib": round(required / MIB, 1),
            "budget_mib": round(budget_mib, 1),
            "budget_basis": basis,
            "fits": required <= budget_bytes,
            "cells_that_fit": int(model.max_cells(budget_bytes)),
            "bytes_per_cell_basis": ("woof.hex.device_admission limited-area "
                                     "row (shaped footprint model)"),
        }
        if worst is None or row["required_mib"] > worst["required_mib"]:
            worst = row
    assert worst is not None
    worst["notes"] = notes
    return worst


# --------------------------------------------------------------------------
# commands


def hex_commands(*, parent_cells: int) -> list[list[str]]:
    """``woof`` argv lists, run in order with cwd = the plan directory.

    ``woof mesh`` needs ``--cells`` or ``--card`` to run at all.  For a
    GRADED spec the count does not size the mesh: ``rw-mpas``
    ``mesh/mod.rs`` passes an explicit target through unscaled
    (``fit_spacing`` off) and ``hierarchy::generate_graded`` builds from the
    spec's own spacings, recording the target only.  ``--card`` would
    instead size the GLOBAL parent against the card, which never runs on it
    (only the cut does), so the planner passes the generator's own count
    when it measured one and its upper-bound estimate otherwise.
    """

    run = RUN_DIR
    parent_grid = f"{run}/parent.grid.nc"
    parent_static = f"{run}/parent.static.nc"
    return [
        ["hex", "mesh-plan", "--spec", "mesh_spec.json", "--json",
         "--out", f"{run}/mesh-plan.json"],
        ["mesh", "--spec", "mesh_spec.json", "--cells", str(int(parent_cells)),
         "--out", parent_grid, "--static-out", parent_static],
        ["hex", "mesh-check", "--grid", parent_grid, "--static",
         parent_static],
        ["hex", "cull", "--parent-grid", parent_grid, "--parent-static",
         parent_static, "--region", "cull_region.json", "--out-dir", run,
         "--name", CULL_NAME],
        ["hex", "mesh-check", "--grid", f"{run}/{CULL_NAME}.grid.nc",
         "--static", f"{run}/{CULL_NAME}.static.nc"],
    ]


BLOCKED_STAGES: tuple[dict[str, str], ...] = (
    {"stage": "vertical",
     "door": "woof.hex.vertical_spec.materialize_vertical_artifact",
     "blocked_by": "the culled native-free vertical artifact `woof hex init "
                   "--capsule/--reference` needs on a limited-area grid is "
                   "minted only inside `woof hex mesh-plan --point --generate`; "
                   "no front door mints it for an authored --spec"},
    {"stage": "register",
     "door": "woof.hex.mesh_rows (runtime rows, $WOOF_HEX_MESH_ROWS)",
     "blocked_by": "`woof hex forecast --mesh` binds only a registered row; "
                   "runtime rows are written only by `woof hex mesh-plan "
                   "--point --generate`, so a corridor mesh has no row"},
    {"stage": "met",
     "door": "woof hex intermediate / woof hex init / woof hex lbc",
     "blocked_by": "regional initial and boundary conditions come from `woof "
                   "hex intermediate`, whose only source row is HRRR (CONUS); "
                   "outside CONUS there is no hex regional met route"},
    {"stage": "forecast",
     "door": "woof hex forecast --out runs/hex/forecast",
     "blocked_by": "needs the three stages above; it would write "
                   "runs/hex/forecast/cuda-history.*.nc (the output_glob)"},
)


# --------------------------------------------------------------------------
# the generator, when staged


def _generator_sizing(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    """``rw_mpas_mesh --dry-run`` with the build's gates, or None if absent.

    A refusal from the gates (or from the engine on this spec) is a plan
    refusal; only a missing engine is skipped, and the plan notes it.
    """

    from woof.hex.mesh_spec_gates import MeshSpecRefusal
    from woof.hex.swath.errors import SwathRefusal
    from woof.hex.swath.sizing import dry_run, resolve_engine

    try:
        engine = resolve_engine()
    except SwathRefusal:
        return None
    try:
        receipt = dry_run(spec, engine=engine)
    except (MeshSpecRefusal, SwathRefusal, KeyError, ValueError) as error:
        raise HexPlanRefusal(
            f"rw_mpas_mesh refuses the corridor spec: {error}") from error
    return dict(receipt)


# --------------------------------------------------------------------------
# the plan


def default_start(now: datetime | None = None) -> str:
    """The most recent 00/06/12/18 UTC cycle, ``YYYY-MM-DDTHH``."""

    now = now or datetime.now(timezone.utc)
    cycle = now.replace(minute=0, second=0, microsecond=0)
    cycle -= timedelta(hours=cycle.hour % 6)
    return cycle.strftime("%Y-%m-%dT%H")


def _check_start(start: str) -> str:
    try:
        datetime.strptime(start, "%Y-%m-%dT%H")
    except ValueError as error:
        raise HexPlanRefusal(
            f"--start {start!r} is not YYYY-MM-DDTHH (UTC)") from error
    return start


def _write_json(document: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".partial")
    staging.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    staging.replace(path)


def build_plan(sites: SiteSet, *, outdir: Path, dx_m: float = 100.0,
               corridor_km: float = 2.0, parent_dx_m: float | None = None,
               start: str | None = None, hours: float = 24.0,
               source: str | None = None, card: str | None = None,
               vram_gib: float | None = None, max_domains: int | None = None,
               nz: int | None = None) -> Plan:
    """Emit the domains for ``sites`` under ``outdir`` and return the plan
    (already written to ``outdir/plan.json``)."""

    outdir = Path(outdir)
    if len(sites) == 0:
        raise HexPlanRefusal("the sites document holds no sites")
    if not (dx_m > 0.0 and math.isfinite(dx_m)):
        raise HexPlanRefusal(f"--dx-m {dx_m!r} must be a positive number")
    if not (corridor_km > 0.0 and math.isfinite(corridor_km)):
        raise HexPlanRefusal(
            f"--corridor-km {corridor_km!r} must be a positive number")
    if nz is not None and nz != HEX_LEVELS:
        raise HexPlanRefusal(
            f"--nz {nz} cannot be honoured by hex-swath: every hex timestep "
            f"anchor, device-admission row and vertical spec runs "
            f"{HEX_LEVELS} levels.  Omit --nz, or use a wrf topology")
    if max_domains is not None and max_domains < 1:
        raise HexPlanRefusal("--max-domains must allow the one hex mesh")
    start = _check_start(start) if start is not None else default_start()

    floor = _check_floor(dx_m)
    bounds = _smoothness_bounds()
    background_km, rungs_km = background_km_for(dx_m, parent_dx_m)
    tolerance_km = SIMPLIFY_FRACTION * corridor_km
    ceiling = bounds["transition_band_ceiling_percent_per_cell"]
    refuse_above = bounds["woof_mesh_refuse_above_percent_per_cell"]
    rungs, transition_factor, gradient = choose_ladder(
        rungs_km, corridor_km, tolerance_km, background_km,
        min(ceiling, refuse_above))

    arrays = sites.as_arrays()
    projection = _Aeqd(*_centre(arrays["lat"], arrays["lon"]))
    chains = corridor_chains(sites, projection)
    clusters = _clusters(chains, 2.0 * rungs[0].reach_km)
    shapes = corridor_regions(chains, clusters, rungs, projection,
                              tolerance_km)
    name = (f"energy corridor {len(sites)} sites {dx_m:g}m "
            f"{corridor_km:g}km on {background_km:g}km")
    spec = mesh_spec_document(shapes, background_km, name)
    cull = estimate_cull(chains, shapes, rungs, background_km, projection,
                         corridor_km)
    parent_estimate = estimate_parent_cells(shapes, chains, background_km)
    verdict = capacity_verdict(cull.cells, card, vram_gib)

    # Every site must sit inside the cut, never in its boundary rings.
    sx, sy = projection.forward(arrays["lat"], arrays["lon"])
    if not np.all(_inside_convex(cull.polygon_xy, sx, sy)):
        raise HexPlanRefusal("internal: a site fell outside the cull polygon")

    if not verdict["fits"]:
        raise HexPlanRefusal(
            f"the corridor cut is about {cull.cells:,.0f} cells "
            f"({cull.fine_cells:,.0f} at the fine spacing), which needs "
            f"{verdict['required_mib']:,.0f} MiB on row {verdict['row']} "
            f"against {verdict['budget_mib']:,.0f} MiB "
            f"({verdict['budget_basis']}); that budget holds "
            f"{verdict['cells_that_fit']:,} cells.  Narrow --corridor-km, "
            "coarsen --dx-m, split the sites into smaller regions and plan "
            "each, name a bigger card, or use --topology wrf-tiles (no count "
            "limit)")

    sizing = _generator_sizing(spec)
    notes: list[str] = []
    measured_parent = None
    if sizing is None:
        notes.append(
            "rw_mpas_mesh is not staged here, so the spec was not priced "
            "through the generator's --dry-run at plan time; the first "
            "command (woof hex mesh-plan) applies the build's gates before "
            "anything is built")
    else:
        measured_parent = sizing.get("predicted_cells")
        band = ((sizing.get("gates_applied_by_hexcore") or {})
                .get("transition_band") or {})
        measured_gradient = band.get("steepest_gradient_percent_per_cell")
        if measured_gradient is not None and measured_gradient > refuse_above:
            raise HexPlanRefusal(
                f"rw_mpas_mesh measures the corridor spec at "
                f"{measured_gradient:.2f} %/cell, past the "
                f"{refuse_above:.2f} %/cell bound woof mesh refuses above")
        notes.append(
            f"rw_mpas_mesh --dry-run priced the global parent at "
            f"{measured_parent:,.0f} cells and measured the steepest "
            f"gradient at {measured_gradient} %/cell; transition-band gate "
            "passed")
    parent_cells = int(math.ceil(measured_parent if measured_parent
                                 else parent_estimate))

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / RUN_DIR).mkdir(parents=True, exist_ok=True)
    _write_json(spec, outdir / "mesh_spec.json")
    cull_shape = {"kind": "polygon",
                  "vertices_deg": [[round(lat, 6), round(lon, 6)]
                                   for lat, lon in cull.ring_latlon]}
    _write_json(cull_shape, outdir / "cull_region.json")

    footprint = tuple((round(lon, 6), round(lat, 6))
                      for lat, lon in cull.ring_latlon)
    footprint = footprint + (footprint[0],)
    estimate = {
        "basis": "area_integral",
        "cull_cells": round(cull.cells, 1),
        "fine_cells": round(cull.fine_cells, 1),
        "cut_reach_km": round(cull.cut_reach_km, 3),
        "spacing_at_cut_km": round(cull.spacing_at_cut_km, 4),
        "boundary_rings": BOUNDARY_RINGS,
        "halo_km": round(cull.halo_km, 3),
        "grid_step_km": round(cull.grid_step_km, 4),
        "grid_points": cull.grid_points,
        "parent_cells_upper_bound": round(parent_estimate, 1),
        "parent_cells_generator": measured_parent,
    }
    mesh = {
        "mesh_spec": "mesh_spec.json",
        "cull_region": "cull_region.json",
        "parent_grid": f"{RUN_DIR}/parent.grid.nc",
        "parent_static": f"{RUN_DIR}/parent.static.nc",
        "grid": f"{RUN_DIR}/{CULL_NAME}.grid.nc",
        "static": f"{RUN_DIR}/{CULL_NAME}.static.nc",
        # What woof energy extract hands sample_mpas(mesh_path=...): the
        # culled init carries latCell/lonCell, bdyMaskCell, cellsOnVertex
        # and zgrid (the CUDA history files carry no zgrid).  Written by the
        # blocked init stage.
        "mesh_path": f"{RUN_DIR}/{CULL_NAME}.init.nc",
        "background_km": background_km,
        "ladder_km": sorted(rungs_km),
        "regions": len(spec["regions"]),
    }
    extra = {
        "commands": hex_commands(parent_cells=parent_cells),
        "runnable": "mesh-and-cull",
        "blocked_stages": [dict(row) for row in BLOCKED_STAGES],
        "transition_factor": transition_factor,
        "ring_factor": RING_FACTOR,
        "rungs": [{"spacing_km": r.spacing_km, "transition_km": r.transition_km,
                   "reach_km": round(r.reach_km, 4)} for r in rungs],
        "gates": {
            "spacing_floor": floor,
            "steepest_gradient_percent_per_cell_estimate": round(gradient, 4),
            **bounds,
            "dual_edge_floor": ("not decidable from a spec "
                                "(woof.hex.mesh_spec_gates."
                                "short_dual_edge_exposure); measured by the "
                                "mesh-check commands"),
        },
        "estimate": estimate,
        "capacity": verdict,
        "corridors": {"chains": len(chains), "clusters": len(clusters),
                      "corridor_km": corridor_km,
                      "simplify_tolerance_km": tolerance_km},
    }
    domain = PlanDomain(
        domain_id="hex",
        topology=TOPOLOGY,
        role="mesh",
        dx_m=dx_m,
        run_dir=RUN_DIR,
        output_glob=OUTPUT_GLOB,
        footprint=footprint,
        config=None,
        grid_id=None,
        site_ids=tuple(site.site_id for site in sites.sites),
        mesh=mesh,
        extra=extra,
    )
    notes = [
        f"hex-swath corridor mesh: {len(spec['regions'])} refinement rows over "
        f"{len(chains)} chains, ladder "
        f"{' > '.join(f'{r:g}' for r in sorted(rungs_km, reverse=True))} km on "
        f"a {background_km:g} km background (ramp {transition_factor:g}x, "
        f"ring {RING_FACTOR:g}x); fine spacing flat to {corridor_km:g} km "
        "from every site",
        f"spacing floor: hex admits >= {floor['floor_dx_m']:.0f} m today "
        f"(smallest anchored dt {floor['smallest_anchored_dt_s']:g} s, "
        f"min dcEdge {floor['min_dc_edge_m']:.0f} m)",
        f"steepest requested gradient estimated at {gradient:.2f} %/cell from "
        "the spec's own density formula (bounds "
        f"{refuse_above:.2f} woof mesh, {ceiling:.2f} transition band)",
        f"cull: convex hull of the sites buffered {cull.cut_reach_km:.1f} km "
        f"({CULL_PAD_SCALE:g} x the fine rung's reach) plus "
        f"{BOUNDARY_RINGS} boundary rings ({cull.halo_km:.1f} km); about "
        f"{cull.cells:,.0f} cells ({estimate['basis']})",
        f"capacity: {verdict['required_mib']:,.0f} MiB required on "
        f"{verdict['row']} against {verdict['budget_mib']:,.0f} MiB "
        f"({verdict['budget_basis']}); cells are priced at the REQUESTED "
        "spacing (the measured point cull, docs hex-point-hrrr, delivered "
        "0.93751 km for 0.9375 km and came out 1.1 % over its prediction), "
        "and the forecast door re-admits the cut's real count at launch",
        *verdict["notes"],
        *cull.notes,
        *notes,
        "runnable today: mesh-and-cull.  " + "; ".join(
            f"{row['stage']}: {row['blocked_by']}" for row in BLOCKED_STAGES),
    ]
    if source is not None:
        notes.append(f"--source {source} is recorded; the hex regional met "
                     "stage that would consume it is blocked (see above)")
    plan = Plan(topology=TOPOLOGY, dx_m=dx_m, start=start, hours=hours,
                domains=[domain], source=source, notes=notes)
    dump_plan(plan, outdir / "plan.json")
    return plan


def main(args) -> int:
    import sys

    outdir = Path(args.outdir)
    try:
        sites = load_sites(args.sites)
        plan = build_plan(
            sites, outdir=outdir, dx_m=args.dx_m,
            corridor_km=args.corridor_km, parent_dx_m=args.parent_dx_m,
            start=args.start, hours=args.hours, source=args.source,
            card=args.card, vram_gib=args.vram_gib,
            max_domains=args.max_domains, nz=args.nz)
        plan.sites_ref = file_ref(args.sites, relative_to=outdir)
        dump_plan(plan, outdir / "plan.json")
    except (HexPlanRefusal, MpasPortError, ContractError, OSError) as error:
        print(f"woof energy plan: REFUSED: {error}", file=sys.stderr)
        return 2
    domain = plan.domains[0]
    record = {
        "schema": "woof-energy.plan.v1",
        "plan": str(outdir / "plan.json"),
        "topology": TOPOLOGY,
        "dx_m": plan.dx_m,
        "domains": len(plan.domains),
        "sites": len(domain.site_ids),
        "cull_cells": domain.extra["estimate"]["cull_cells"],
        "capacity": {k: domain.extra["capacity"][k]
                     for k in ("row", "required_mib", "budget_mib", "fits")},
        "runnable": domain.extra["runnable"],
        "commands": len(domain.extra["commands"]),
        "notes": plan.notes,
    }
    print(json.dumps(record, indent=2, default=str))
    return 0


__all__ = [
    "TOPOLOGY", "RUN_DIR", "OUTPUT_GLOB", "HexPlanRefusal", "background_km_for",
    "build_plan", "capacity_verdict", "hex_commands", "ladder", "main",
    "spacing_at_distance", "spacing_floor_m", "steepest_gradient_percent",
]
