"""``woof energy plan --topology wrf-tiles``: one parent run, many offline tiles.

The irregular WRF topology with no domain-count ceiling.  A single regional
parent configuration is emitted at ``parent_dx_m`` and run once; every
corridor is then covered by one-way offline child tiles at ``dx_m`` that
``woof downscale`` builds from the parent's archived history.  A tile runs
alone on the card, so its size is bounded by the per-tile VRAM budget and
the tile count is bounded by nothing but ``--max-domains``.

How the plan is built
---------------------

1. **Refinement chain.**  ``woof downscale`` takes one integer ratio per
   invocation.  It accepts any ratio of 1 or more, but this planner holds a
   step to ``woof domain``'s blessed nest maximum
   (:data:`woof.domain_wizard.MAX_CHAIN_RATIO`, 8; WRF's guidance is 3 or 5,
   and docs/public/DOWNSCALE.md records the boundary breakage measured at
   ratios 12 and 20).  The default parent is ``dx_m`` x 5; when that parent
   does not fit the card, the chain grows to 5 x 5 (and 5 x 5 x 5), with
   intermediate tiles (``role="parent"``) between the parent run and the
   leaf tiles.  An explicit ``--parent-dx-m`` must be a whole multiple of
   ``dx_m`` that factors into steps of at most 8, or the plan is refused.

2. **Parent.**  ``woof domain --polygon`` emits the parent: the sites'
   bounding box, buffered by the corridor plus every boundary clearance the
   tiles below need, at ``--root-dx parent_dx_m``, sized against
   ``--card``/``--vram-gib`` (default ``24gb``, the ``woof downscale``
   default, so a plan never probes the GPU of the machine it is written on).
   History is written every :data:`woof.downscale.CADENCE_GUIDANCE_SECONDS`
   (900 s, the downscale guidance) and the wizard's hourly checkpoint
   provides the restart evidence ``--parent-restart latest`` binds.  The
   emitted TOML is validated with :func:`woof.experiment.load_experiment`.

3. **Tiles.**  Sites are put on the parent's own projected grid (the
   namelist.wps the wizard wrote, through :mod:`woof.static.projection`),
   in metres whose multiples of the parent spacing are parent cell edges,
   and :func:`woof.energy.geometry.cover_with_rectangles` covers them with
   rectangles snapped to parent cells and no larger than the biggest child
   the per-tile budget affords.  That budget is priced the way
   ``woof downscale --point`` prices it: the child RunConfig derived from
   the parent's (:func:`woof.downscale._derive_child_run_config`), with
   the LES overrides of step 5 applied to an LES leaf, through
   :func:`woof.core.preflight.estimate_experiment`, against the same
   ceiling (:func:`woof.downscale._budget_bytes`).

4. **Placement.**  Each rectangle becomes one ``woof downscale
   --child-config tiles/<id>.toml --ratio R --i-parent-start I
   --j-parent-start J`` invocation.  The start indices are the ones
   ``woof downscale --point`` would reach: the planner centres the
   rectangle on a parent mass point and proves the start with downscale's
   own centred placement (``_centered_placement``, including the SINT
   stencil-coverage gate), so the child that runs covers exactly the cells
   planned.  Every tile must stay clear of its parent's
   specified/relaxation zone, terrain blend and stencil.

5. **The child configuration.**  Each tile's TOML is built by the code
   path ``woof downscale --point`` writes its derived child config with
   (:func:`woof.downscale._derive_child_run_config`,
   ``_with_parent_epssm_label``, ``_render_child_toml`` and the
   ``[static]`` table), from the parent's emitted RunConfig rather than
   from its restart evidence, which does not exist at plan time.  Two
   things are then set that ``--point`` has no flag for:

   * leaf tiles at or finer than :data:`LES_CHILD_DX_M` (250 m) carry the
     LES gray-zone closure of ``configs/les_nest_250m_km3.toml`` d03
     (:data:`LES_PHYSICS_RECIPE`: ``km_opt = 3``, ``bl_pbl_physics = 0``,
     ``mix_isotropic = 1`` ...), and their own vertical ladder
     (``eta_levels``, ``--nz`` or :data:`DEFAULT_LES_CHILD_NZ`);
   * every leaf tile writes the ``energy`` history preset (``[output]
     preset = "energy"``).  Intermediate tiles keep the full inventory,
     because the tiles below them are downscaled from it and the preset
     sheds the land identity and soil state a child is built from.

   The TOML is priced with the overrides in place, read back through the
   door's own loader (:func:`woof.offline_child.resolve_child_run_config`)
   and refused unless every override survives the round trip, then
   written to ``tiles/<id>.toml`` and bound by its sha256
   (``--child-config-sha256``), so an edited file is refused by
   ``woof downscale`` rather than run.  :func:`woof.experiment.
   load_experiment` reads only the experiment schema and refuses the
   legacy ``[grid]``/``[run]`` schema a ``--child-config`` file must use;
   the patched RunConfig is checked through
   :func:`woof.experiment.experiment_from_run_config` and the estimator
   instead.

Python boundary (docs/dev/static-rust-port.md): everything here is
orchestration over per-site and per-tile vectors (projection of the site
coordinates onto the parent grid, a few hundred rectangles); no field data
is touched, so it stays in numpy.
"""

from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from woof.energy.contracts import (
    PLAN_SCHEMA,
    ContractError,
    EnergyNotImplemented,
    Plan,
    PlanDomain,
    SiteSet,
    _write_json,
    dump_plan,
    load_sites,
)

TOPOLOGY = "wrf-tiles"

#: Where ``woof go <config> --outdir runs/parent`` leaves the parent's
#: frames, relative to its run_dir (the run orchestrator's layout).
PARENT_OUTPUT_GLOB = "run-*/run/wrfout/wrfout_d01_*"

#: ``woof downscale --health-interval-seconds`` default: one of the clocks a
#: derived child's fixed step must divide (``clock_seconds``).
DOWNSCALE_HEALTH_INTERVAL_S = 60.0

#: Schema of the per-tile spec document written to ``tiles/<id>.json``.
TILE_SCHEMA = "woof-energy.wrf-tile.v1"

#: The step the planner prefers when it chooses the chain itself.
DEFAULT_STEP_RATIO = 5

#: Most chain steps the planner will try before refusing.
MAX_CHAIN_STEPS = 3

#: Card the tiles and the parent are sized for when neither ``--card`` nor
#: ``--vram-gib`` is given: ``woof downscale --card``'s own default.
DEFAULT_CARD = "24gb"

#: Leaf tiles at or finer than this spacing are in the LES regime
#: ``woof downscale`` states at its door (docs/public/DOWNSCALE.md, "The
#: LES case is stated at the door").
LES_CHILD_DX_M = 250.0

#: The leaf's own vertical ladder (``eta_levels`` in its child TOML, built
#: as ``--child-levels N,STRETCH`` builds it) when ``--nz`` is not given.
#: 60 levels stays under every radiation bound a parent suite can carry at
#: ``p_top = 5000 Pa`` (legacy RRTMG shortwave is the tightest at 63; docs/public/LES.md section 4), and the stretch is the
#: one docs/public/DOWNSCALE.md uses for its LES child.
DEFAULT_LES_CHILD_NZ = 60
CHILD_LEVEL_STRETCH = 2.5

#: Rows beyond a parent's own specified/relaxation zone that a tile must
#: also keep clear of: the five-row terrain blend a downscaled child applies
#: (WRF ``blend_terrain``) and SINT's two-cell donor stencil.
TERRAIN_BLEND_ROWS = 5
SINT_STENCIL_ROWS = 2

#: The LES gray-zone closure of the shipped 250 m child
#: (``configs/les_nest_250m_km3.toml`` d03; docs/public/LES.md and
#: GRAYZONE-NEST.md), with that domain's ``cu_physics = 0``.  Written into
#: the ``[run]`` table of every leaf tile's ``--child-config`` TOML at or
#: finer than :data:`LES_CHILD_DX_M`; every key is a per-domain RunConfig
#: field, and each is read back through ``woof downscale``'s own loader
#: before the plan is written.
LES_PHYSICS_RECIPE = {
    "bl_pbl_physics": 0,
    "km_opt": 3,
    "diff_opt": 2,
    "c_s": 0.25,
    "mix_isotropic": 1,
    "mix_upper_bound": 0.1,
    "isfflx": 1,
    "cu_physics": 0,
}

#: Where the LES recipe comes from, recorded in each tile spec.
LES_RECIPE_SOURCE = "configs/les_nest_250m_km3.toml d03 (docs/public/LES.md)"

#: The history preset name shared across the energy units.
ENERGY_HISTORY_PRESET = "energy"

#: Why an intermediate tile keeps the full history inventory.
_INTERMEDIATE_HISTORY = (
    "an intermediate tile is the parent of the tiles below it, which "
    "woof downscale builds from its full history (land identity, soil "
    "state); the energy preset sheds those fields, so the tile keeps the "
    "full inventory")

#: How each tile's child TOML is derived, recorded in its spec.
_DERIVATION = (
    "woof.downscale._derive_child_run_config, _with_parent_epssm_label, "
    "_render_child_toml and the [static] table: the code path woof "
    "downscale --point writes its derived child config with, run at plan "
    "time on the parent's emitted RunConfig (the parent's restart "
    "evidence, which --point derives from, does not exist until the "
    "parent has run)")


class TilePlanRefusal(ValueError):
    """The wrf-tiles plan cannot be built as asked."""


class SiteOutsideParent(TilePlanRefusal):
    """Sites lie outside the parent, or too close to its boundary zone."""


class TooManyTiles(TilePlanRefusal):
    """The cover needs more tiles than ``--max-domains`` allows."""


class GeometryContractError(RuntimeError):
    """``cover_with_rectangles`` returned a cover this planner cannot use."""


# --------------------------------------------------------------------------
# small helpers


def factor_chain(total: int, *, max_ratio: int | None = None) -> tuple[int, ...]:
    """Split an integer refinement into the fewest steps of at most
    ``max_ratio``; among equally short chains the most balanced wins and
    the coarse step comes first.  Refuses what cannot be split."""

    from woof.domain_wizard import MAX_CHAIN_RATIO, MIN_CHAIN_RATIO

    max_ratio = MAX_CHAIN_RATIO if max_ratio is None else int(max_ratio)
    total = int(total)
    if total < MIN_CHAIN_RATIO:
        raise TilePlanRefusal(
            f"a parent/tile ratio of {total} is not a refinement; the "
            f"parent spacing must be at least {MIN_CHAIN_RATIO}x --dx-m")
    best: list[tuple[int, ...]] = []

    def split(rest: int, cap: int, prefix: tuple[int, ...]):
        if rest == 1:
            best.append(prefix)
            return
        for ratio in range(min(cap, rest), MIN_CHAIN_RATIO - 1, -1):
            if rest % ratio == 0:
                split(rest // ratio, ratio, prefix + (ratio,))

    split(total, max_ratio, ())
    if not best:
        raise TilePlanRefusal(
            f"a total refinement of {total} cannot be split into woof "
            f"downscale steps of {MIN_CHAIN_RATIO} to {max_ratio} (it has a "
            f"prime factor above {max_ratio}); choose --parent-dx-m so that "
            "--parent-dx-m / --dx-m is a product of such steps, e.g. 5, 25 "
            "or 30")
    shortest = min(len(chain) for chain in best)
    candidates = [chain for chain in best if len(chain) == shortest]
    return min(candidates, key=lambda chain: (max(chain), [-r for r in chain]))


def default_start(now: datetime | None = None) -> str:
    """The most recent 00/06/12/18 UTC cycle at or before ``now``."""

    now = datetime.now(timezone.utc) if now is None else now
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    cycle = now.replace(hour=now.hour - now.hour % 6, minute=0, second=0,
                        microsecond=0)
    return cycle.strftime("%Y-%m-%dT%H")


def _parse_start(start: str | None) -> str:
    if start is None:
        return default_start()
    text = str(start).strip()
    for pattern in ("%Y-%m-%dT%H", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%HZ", "%Y-%m-%dT%H:%MZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            parsed = datetime.strptime(text, pattern)
        except ValueError:
            continue
        if parsed.minute or parsed.second:
            raise TilePlanRefusal(
                f"--start {start!r} is not on a whole hour; the parent's "
                "forcing cycle is an hour (YYYY-MM-DDTHH, UTC)")
        return parsed.strftime("%Y-%m-%dT%H")
    raise TilePlanRefusal(
        f"--start {start!r} is not YYYY-MM-DDTHH (UTC)")


def _resolve_capacity(card: str | None, vram_gib: float | None
                      ) -> tuple[float, list[str], str | None]:
    """(capacity GiB, the downscale capacity flags, the default note)."""

    from woof.domain_wizard import declared_card_gib

    if vram_gib is not None:
        capacity = float(vram_gib)
        if not math.isfinite(capacity) or capacity <= 0.0:
            raise TilePlanRefusal(f"--vram-gib {vram_gib!r} is not a size")
        return capacity, ["--vram-gib", f"{capacity:g}"], None
    note = None
    if card is None:
        card = DEFAULT_CARD
        note = (f"no --card/--vram-gib given: the parent and every tile are "
                f"sized for a {DEFAULT_CARD} card (woof downscale's own "
                "default); the plan never probes the GPU of the machine it "
                "is written on")
    try:
        capacity = float(declared_card_gib(card))
    except ValueError as error:
        raise TilePlanRefusal(str(error)) from error
    return capacity, ["--card", str(card)], note


def _ring(grid, nx: int, ny: int, per_edge: int = 8
          ) -> tuple[tuple[float, float], ...]:
    """Closed ``[lon, lat]`` ring of a grid's mass-point extent (1-based
    mass indices 1..nx, 1..ny), densified along each edge."""

    t = np.linspace(0.0, 1.0, per_edge + 1)[:-1]
    i = np.concatenate([1 + t * (nx - 1), np.full(t.size, float(nx)),
                        nx - t * (nx - 1), np.full(t.size, 1.0)])
    j = np.concatenate([np.full(t.size, 1.0), 1 + t * (ny - 1),
                        np.full(t.size, float(ny)), ny - t * (ny - 1)])
    lat, lon = grid.ij_to_latlon(i, j)
    lon = (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0 - 180.0
    points = [(round(float(a), 6), round(float(b), 6))
              for a, b in zip(lon, np.asarray(lat, dtype=np.float64))]
    points.append(points[0])
    return tuple(points)


def _content_sha256(sites: SiteSet) -> str:
    text = json.dumps(sites.to_json(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _relpath(path: Path, start: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(),
                                Path(start).resolve())).as_posix()


# --------------------------------------------------------------------------
# the parent


@dataclass
class _Parent:
    toml: Path
    wps: Path
    grid: Any
    nx: int
    ny: int
    dx: float
    run_config: dict
    history_interval_s: float
    run_seconds: float
    clearance_cells: int
    peak_envelope_gib: float | None
    transcript: str
    #: The root's ``epssm`` was the model's choice (``ExperimentConfig
    #: .auto_epssm``), which its checkpoint labels and ``--point`` hands
    #: down as ``epssm = { auto = VALUE }``.
    auto_epssm: bool = False


def _sites_region(lon: np.ndarray, lat: np.ndarray) -> dict:
    """GeoJSON Polygon of the sites' bounding box (what the parent must
    contain before the buffer)."""

    west, east = float(np.min(lon)), float(np.max(lon))
    south, north = float(np.min(lat)), float(np.max(lat))
    if east - west > 180.0:
        raise TilePlanRefusal(
            "the sites span more than 180 degrees of longitude (they cross "
            "the antimeridian); plan each side of it separately")
    # A degenerate box (one site, or sites on one parallel) still needs a
    # ring with area; the buffer does the real sizing.
    pad = 1e-4
    if east - west < pad:
        west, east = west - pad, east + pad
    if north - south < pad:
        south, north = south - pad, north + pad
    ring = [[west, south], [east, south], [east, north], [west, north],
            [west, south]]
    return {"type": "Polygon", "coordinates": [ring]}


class _ParentDoesNotFit(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _run_domain_wizard(argv: list[str], cwd: Path,
                       log_path: Path) -> tuple[int, str]:
    """Run ``woof domain`` in-process with its output captured; the
    transcript is written to ``log_path`` whatever the outcome."""

    from woof.cli import build_parser
    from woof.domain_wizard import DomainFitError

    parser = build_parser()
    buffer = io.StringIO()
    with contextlib.ExitStack() as stack:
        stack.callback(lambda: log_path.write_text(buffer.getvalue()))
        stack.enter_context(contextlib.chdir(cwd))
        stack.enter_context(contextlib.redirect_stdout(buffer))
        stack.enter_context(contextlib.redirect_stderr(buffer))
        try:
            args = parser.parse_args(["domain", *argv])
            code = args.func(args)
        except DomainFitError as error:
            raise _ParentDoesNotFit(str(error)) from error
        except ValueError as error:
            # The wizard's documented refusals (source coverage, cycle,
            # projection windows) are ValueErrors: not a size question,
            # so no coarser chain is tried.
            raise TilePlanRefusal(
                f"woof domain refused the parent: {error}") from error
        except SystemExit as error:
            code = error.code if isinstance(error.code, int) else 2
    return int(code or 0), buffer.getvalue()


def _emit_parent(*, outdir: Path, region: dict, parent_dx_m: float,
                 buffer_m: float, start: str, hours: int,
                 source: str | None, capacity_flags: list[str],
                 vram_gib: float) -> _Parent:
    """Emit, validate and load the parent configuration."""

    from woof.downscale import CADENCE_GUIDANCE_SECONDS
    from woof.experiment import load_experiment
    from woof.static.projection import grids_from_wps_namelist

    parent_dir = outdir / "parent"
    parent_dir.mkdir(parents=True, exist_ok=True)
    region_path = parent_dir / "region.geojson"
    region_path.write_text(json.dumps(region) + "\n")
    toml = parent_dir / "parent.toml"
    argv = ["--polygon", str(region_path.resolve()),
            "--buffer-km", f"{buffer_m / 1000.0:.3f}",
            "--root-dx", f"{parent_dx_m / 1000.0:g}",
            "--hours", str(int(hours)),
            "--cycle", start,
            "--history-interval", f"{CADENCE_GUIDANCE_SECONDS:g}",
            "--name", "energy_tiles_parent",
            "--out", str(toml.resolve()), *capacity_flags]
    if source is not None:
        argv += ["--source", str(source)]
    # The wizard writes [fetch].out relative to its working directory, and
    # `woof go` reads it relative to the launch directory, while
    # [case_data] paths are relative to the TOML.  Emitting from the plan
    # directory binds both to the same file when every plan command is
    # launched from there (the convention downscale_args also follows).
    code, transcript = _run_domain_wizard(
        argv, outdir.resolve(), (parent_dir / "woof-domain.log").resolve())
    if code != 0:
        # The wizard raises DomainFitError in-process (handled above as
        # a size question); a non-zero return is any other refusal.
        last = [line for line in transcript.strip().splitlines()
                if line.startswith("woof domain:")]
        message = last[-1] if last else transcript.strip()[-2000:]
        raise TilePlanRefusal(
            f"woof domain could not emit the parent at {parent_dx_m:g} m: "
            f"{message}")
    exp = load_experiment(toml)
    if len(exp.domains) != 1:
        raise TilePlanRefusal(
            f"the emitted parent {toml} has {len(exp.domains)} domains; a "
            "wrf-tiles parent is one root")
    root = exp.root
    run = dataclasses.asdict(root.run)
    if abs(float(run["dx"]) - parent_dx_m) > 1e-6 * parent_dx_m:
        raise TilePlanRefusal(
            f"the emitted parent spacing {run['dx']:g} m is not the "
            f"requested {parent_dx_m:g} m")
    if float(run.get("restart_interval_s") or 0.0) <= 0.0:
        raise TilePlanRefusal(
            f"the emitted parent {toml} writes no checkpoint "
            "(restart_interval_s = 0); woof downscale --parent-restart "
            "latest needs one inside the window")
    wps = parent_dir / "parent.namelist.wps"
    grid = grids_from_wps_namelist(wps)[0]
    nx, ny = int(grid.e_we) - 1, int(grid.e_sn) - 1
    if (nx, ny) != (int(run["nx"]), int(run["ny"])):
        raise TilePlanRefusal(
            f"{wps} describes a {nx}x{ny} grid but {toml} a "
            f"{run['nx']}x{run['ny']} one")
    peak = None
    try:
        from woof.core.preflight import GIB, estimate_experiment

        peak = float(estimate_experiment(exp, vram_gib=vram_gib)
                     .peak_envelope_bytes) / GIB
    except (ValueError, NotImplementedError, KeyError, RuntimeError):
        # The wizard has already priced and admitted this parent; the
        # figure here is a summary line, so its absence is reported as
        # null rather than refused.
        peak = None
    spec_bdy = int(run.get("spec_bdy_width") or 5)
    return _Parent(
        toml=toml, wps=wps, grid=grid, nx=nx, ny=ny, dx=float(run["dx"]),
        run_config=run, history_interval_s=float(root.history_interval_s),
        run_seconds=float(exp.run_seconds),
        clearance_cells=spec_bdy + TERRAIN_BLEND_ROWS + SINT_STENCIL_ROWS,
        peak_envelope_gib=peak, transcript=transcript,
        auto_epssm=int(root.grid_id) in {
            int(gid) for gid in (getattr(exp, "auto_epssm", None) or ())})


# --------------------------------------------------------------------------
# tile sizing


class _Pricer:
    """Peak envelope of a derived child, cached, and the largest square."""

    def __init__(self, vram_gib: float):
        from woof.downscale import _budget_bytes

        self.vram_gib = float(vram_gib)
        self.limit = _budget_bytes(self.vram_gib)[1]
        self._cache: dict[tuple, tuple[float, dict]] = {}

    def derive(self, parent_cfg: dict, *, parent_dx: float, ratio: int,
               nx: int, ny: int, run_seconds: float, output_interval_s: float,
               levels: tuple | None, centre_lat: float,
               overrides: dict | None = None) -> dict:
        """The child RunConfig ``woof downscale --point`` would derive,
        with ``overrides`` (the LES recipe) applied on top and the result
        re-validated by the config authority."""

        from woof.config import RunConfig, validate_run_config
        from woof.downscale import _derive_child_run_config

        merged = _derive_child_run_config(
            parent_cfg, parent={"dx": parent_dx, "dy": parent_dx},
            ratio=int(ratio), child_nx=int(nx), child_ny=int(ny),
            run_seconds=run_seconds, output_interval_s=output_interval_s,
            child_eta_levels=levels, centre_lat=centre_lat,
            clock_seconds=(DOWNSCALE_HEALTH_INTERVAL_S,))
        if overrides:
            unknown = sorted(set(overrides) - set(merged))
            if unknown:
                raise ValueError(
                    f"override key(s) {unknown} are not RunConfig fields; "
                    "a --child-config [run] table cannot carry them")
            merged = {**merged, **overrides}
            validate_run_config(RunConfig(**merged))
        return merged

    def price(self, key: tuple, parent_cfg: dict, **kwargs) -> tuple[float, dict]:
        cache_key = key + (kwargs["nx"], kwargs["ny"],
                           round(float(kwargs["centre_lat"]), 4),
                           tuple(sorted((kwargs.get("overrides") or {})
                                        .items())))
        if cache_key not in self._cache:
            from woof.config import RunConfig
            from woof.core.preflight import estimate_experiment
            from woof.experiment import experiment_from_run_config

            merged = self.derive(parent_cfg, **kwargs)
            exp = experiment_from_run_config(
                RunConfig(**merged), datetime(2000, 1, 1, tzinfo=timezone.utc))
            estimate = estimate_experiment(exp, vram_gib=self.vram_gib)
            self._cache[cache_key] = (float(estimate.peak_envelope_bytes),
                                      merged)
        return self._cache[cache_key]

    def largest_square(self, key: tuple, parent_cfg: dict, *, ratio: int,
                       **kwargs) -> int:
        """Largest square child (a multiple of ``2*ratio``) under budget.

        The search starts at the smallest tile this planner can place
        (:func:`_min_span` parent cells plus one cell of snapping slack on
        each side), so a budget below it is refused as a budget question
        rather than surfacing later as an oversized rectangle."""

        unit = 2 * int(ratio)
        smallest = max(2, math.ceil((_min_span(int(ratio)) + 2) / 2))
        errors: list[Exception] = []

        def fits(size: int) -> bool:
            try:
                peak, _ = self.price(key, parent_cfg, ratio=ratio, nx=size,
                                     ny=size, **kwargs)
            except (ValueError, NotImplementedError) as error:
                errors.append(error)
                return False
            return peak <= self.limit

        if not fits(unit * smallest):
            if errors:
                # A size-independent refusal of the derived child config
                # is that refusal, not a VRAM verdict.
                raise TilePlanRefusal(
                    f"the child config derived at ratio {ratio} does not "
                    f"validate (this is not a VRAM question): {errors[-1]}")
            raise TilePlanRefusal(
                f"no child tile at ratio {ratio} fits the {self.vram_gib:g} "
                "GiB per-tile budget (even the smallest legal tile is over "
                "it); use a larger --card/--vram-gib or fewer --nz levels")
        low, high = smallest, smallest * 2
        while fits(unit * high) and unit * high < 4096:
            low, high = high, high * 2
        while low + 1 < high:
            mid = (low + high) // 2
            if fits(unit * mid):
                low = mid
            else:
                high = mid
        return unit * low


def _tile_capacity(pricer: _Pricer, key: tuple, parent_cfg: dict, *,
                   ratio: int, **kwargs) -> int:
    """Largest child extent (cells per axis) a tile may have."""

    return pricer.largest_square(key, parent_cfg, ratio=ratio, **kwargs)


# --------------------------------------------------------------------------
# tile placement


@dataclass
class _Tile:
    domain_id: str
    role: str
    level: int
    ratio: int
    dx: float
    nx: int
    ny: int
    i_parent_start: int
    j_parent_start: int
    i0: int
    j0: int
    centre_lat: float
    centre_lon: float
    grid: Any
    parent_id: str
    parent_wrf_grid_id: int
    wrf_grid_id: int
    sites: np.ndarray
    run_config: dict
    peak_envelope_gib: float
    levels: tuple | None
    min_edge_distance_m: float | None = None
    children: list = field(default_factory=list)
    #: The LES recipe written into this tile's TOML (empty when none).
    overrides: dict = field(default_factory=dict)
    #: The parent's epssm was the model's choice (see :class:`_Parent`).
    parent_auto_epssm: bool = False


def _placement_point(start: int, span: int) -> int:
    """0-based parent index ``woof downscale`` must centre on so that its
    centred placement starts at 1-based ``start`` with ``span`` cells."""

    half = span // 2
    return start - 1 + half if span % 2 else start - 2 + half


def _relax_cells(ratio: int) -> int:
    """Relaxation-zone width (child cells) of a derived child at ``ratio``
    before the share cap: :func:`woof.downscale.child_lateral_zone`."""

    from woof.downscale import CHILD_RELAX_PARENT_CELLS

    return max(4, CHILD_RELAX_PARENT_CELLS * int(ratio))


def _min_span(ratio: int) -> int:
    """Fewest parent cells a child tile may span: room for its own
    specified/relaxation zone and terrain blend on both sides plus an
    interior, and downscale's own floor of two parent cells."""

    zone_cells = 1 + _relax_cells(ratio) + TERRAIN_BLEND_ROWS
    return max(2, math.ceil((2 * zone_cells + 1) / ratio))


def _level_margins(chain: Sequence[int], parent_dx: float,
                   corridor_m: float) -> list[float]:
    """Ground margin each tile level needs around its sites, coarsest
    first.  The leaf needs the corridor; a level above it must contain the
    level below plus its own boundary zone (specified row, relaxation zone
    of ``2 x ratio`` cells as :func:`woof.downscale.child_lateral_zone`
    sizes it, terrain blend, stencil) and one cell of snapping slack."""

    dxs = [float(parent_dx)]
    for ratio in chain:
        dxs.append(dxs[-1] / ratio)
    margins = [float(corridor_m)]
    for level in range(len(chain) - 1, 0, -1):
        clearance = (1 + _relax_cells(int(chain[level - 1]))
                     + TERRAIN_BLEND_ROWS + SINT_STENCIL_ROWS)
        margins.insert(0, margins[0] + (clearance + 1) * dxs[level])
    return margins


def _cells_from_rect(lo: float, hi: float, parent_dx: float,
                     min_span: int) -> tuple[int, int]:
    """(1-based first parent cell, span) of a snapped rectangle axis."""

    a = math.floor(lo / parent_dx + 1e-6)
    b = math.ceil(hi / parent_dx - 1e-6)
    span = max(b - a, 1)
    if span < min_span:
        grow = min_span - span
        a -= grow // 2
        span = min_span
    return a + 1, span


# --------------------------------------------------------------------------
# the plan


def build_plan(sites: SiteSet, *, outdir: Path, dx_m: float = 100.0,
               corridor_km: float = 2.0, parent_dx_m: float | None = None,
               start: str | None = None, hours: float = 24.0,
               source: str | None = None, card: str | None = None,
               vram_gib: float | None = None, max_domains: int | None = None,
               nz: int | None = None) -> Plan:
    """Emit the domains for ``sites`` under ``outdir`` and return the plan
    (already written to ``outdir/plan.json``)."""

    from woof.domain_wizard import _SPEC_BDY_WIDTH
    from woof.energy.geometry import cover_with_rectangles

    outdir = Path(outdir)
    if len(sites) == 0:
        raise TilePlanRefusal("the sites document has no sites to plan for")
    dx_m = float(dx_m)
    corridor_m = float(corridor_km) * 1000.0
    if not (math.isfinite(dx_m) and dx_m > 0.0):
        raise TilePlanRefusal(f"--dx-m {dx_m!r} must be a positive spacing")
    if not (math.isfinite(corridor_m) and corridor_m > 0.0):
        raise TilePlanRefusal("--corridor-km must be positive")
    hours_f = float(hours)
    if not math.isfinite(hours_f) or hours_f <= 0.0 or hours_f != int(hours_f):
        raise TilePlanRefusal(
            f"--hours {hours!r} must be a whole number of hours (woof domain "
            "emits run_seconds = hours * 3600 from an integer)")
    hours_i = int(hours_f)
    if nz is not None and int(nz) < 4:
        raise TilePlanRefusal("--nz must be at least 4 (the vertical "
                              "stencil width)")
    start_text = _parse_start(start)
    capacity, capacity_flags, card_note = _resolve_capacity(card, vram_gib)

    if parent_dx_m is not None:
        total = float(parent_dx_m) / dx_m
        if not math.isfinite(total) or abs(total - round(total)) > 1e-6 * max(total, 1.0):
            raise TilePlanRefusal(
                f"--parent-dx-m {parent_dx_m:g} is not a whole multiple of "
                f"--dx-m {dx_m:g}; woof downscale refines by integer ratios")
        chains = [factor_chain(int(round(total)))]
    else:
        chains = [(DEFAULT_STEP_RATIO,) * steps
                  for steps in range(1, MAX_CHAIN_STEPS + 1)]

    arrays = sites.as_arrays()
    lat = arrays["lat"]
    lon = arrays["lon"]
    site_ids = arrays["site_id"]
    region = _sites_region(lon, lat)
    leaf_levels_n = (int(nz) if nz is not None
                     else DEFAULT_LES_CHILD_NZ if dx_m <= LES_CHILD_DX_M
                     else None)

    notes: list[str] = []
    if card_note:
        notes.append(card_note)
    tried: list[str] = []
    parent: _Parent | None = None
    chain: tuple[int, ...] = ()
    for candidate in chains:
        pdx = dx_m * math.prod(candidate)
        margins = _level_margins(candidate, pdx, corridor_m)
        # The wizard's root keeps spec_bdy_width = 5; two more cells cover
        # snapping and the minimum-span growth of a small tile.
        parent_clearance = (_SPEC_BDY_WIDTH + TERRAIN_BLEND_ROWS
                            + SINT_STENCIL_ROWS)
        buffer_m = margins[0] + (parent_clearance + 2) * pdx
        try:
            parent = _emit_parent(
                outdir=outdir, region=region, parent_dx_m=pdx,
                buffer_m=buffer_m, start=start_text, hours=hours_i,
                source=source, capacity_flags=capacity_flags,
                vram_gib=capacity)
        except _ParentDoesNotFit as error:
            tried.append(f"{pdx:g} m: {error.message}")
            continue
        chain = candidate
        break
    if parent is None:
        raise TilePlanRefusal(
            "no parent fits the card: " + "; ".join(tried)
            + ("" if parent_dx_m is None else
               "; choose a coarser --parent-dx-m or a larger card"))
    if tried:
        notes.append(
            "the one-step parent did not fit the card ("
            + "; ".join(tried) + f"); the plan refines in {len(chain)} "
            "steps with intermediate tiles")

    pricer = _Pricer(capacity)
    tiles_dir = outdir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.json", "*.toml"):
        for stale in tiles_dir.glob(pattern):
            stale.unlink()
    # A refused re-plan must not leave an older plan.json pointing at the
    # tile specs just removed and the parent just re-emitted.
    (outdir / "plan.json").unlink(missing_ok=True)

    counter = {"leaf": 0, "inter": 0}
    all_tiles: list[_Tile] = []

    def plan_level(*, grid, nx: int, ny: int, pdx: float, cfg: dict,
                   clearance: int, members: np.ndarray, ratios: tuple,
                   parent_id: str, parent_wrf_grid_id: int, level: int,
                   margins: list[float], parent_auto_epssm: bool) -> None:
        ratio = int(ratios[0])
        margin_m = margins[0]
        leaf = len(ratios) == 1
        cdx = pdx / ratio
        # The LES closure goes on leaf tiles in the LES regime only; an
        # intermediate tile forces the level below with its parent's
        # physics.
        overrides = (dict(LES_PHYSICS_RECIPE)
                     if leaf and cdx <= LES_CHILD_DX_M + 1e-9 else {})
        m_lat, m_lon = lat[members], lon[members]
        fi, fj = (np.asarray(v, dtype=np.float64)
                  for v in grid.latlon_to_ij(m_lat, m_lon))
        outside = ((fi < 0.5) | (fi > nx + 0.5) | (fj < 0.5) | (fj > ny + 0.5)
                   | ~np.isfinite(fi) | ~np.isfinite(fj))
        if np.any(outside):
            bad = [str(s) for s in site_ids[members][outside][:10]]
            raise SiteOutsideParent(
                f"{int(outside.sum())} site(s) fall outside "
                f"{parent_id} ({nx}x{ny} at {pdx:g} m): {', '.join(bad)}"
                + (" ..." if outside.sum() > 10 else ""))
        mapfac = float(np.max(np.asarray(grid.map_factor(m_lat),
                                         dtype=np.float64)))
        levels = None
        if leaf and leaf_levels_n is not None:
            from woof.downscale import build_child_eta_levels

            levels = build_child_eta_levels(leaf_levels_n,
                                            stretch=CHILD_LEVEL_STRETCH)
        centre_lat = float(np.mean(m_lat))
        price_kwargs = dict(parent_dx=pdx, run_seconds=parent.run_seconds,
                            output_interval_s=parent.history_interval_s,
                            levels=levels, centre_lat=centre_lat,
                            overrides=overrides)
        key = (parent_id, ratio, leaf_levels_n if leaf else None,
               bool(overrides))
        size = _tile_capacity(pricer, key, cfg, ratio=ratio, **price_kwargs)
        # Room for one parent cell of snapping slack on each side.
        max_cells = max(2 * ratio, size - 2 * ratio)
        x = (fi - 0.5) * pdx
        y = (fj - 0.5) * pdx
        rects = cover_with_rectangles(
            x, y, margin_m=margin_m * mapfac, dx_m=cdx, max_nx=max_cells,
            max_ny=max_cells, align_m=pdx)
        seen = np.zeros(members.size, dtype=np.int64)
        min_span = _min_span(ratio)
        for rect in rects:
            local = np.asarray(rect.members, dtype=np.int64)
            if local.size == 0:
                continue
            seen[local] += 1
            i_start, span_i = _cells_from_rect(rect.x_min, rect.x_max, pdx,
                                               min_span)
            j_start, span_j = _cells_from_rect(rect.y_min, rect.y_max, pdx,
                                               min_span)
            child_nx, child_ny = span_i * ratio, span_j * ratio
            if max(child_nx, child_ny) > size:
                raise GeometryContractError(
                    f"cover_with_rectangles returned a {child_nx}x{child_ny} "
                    f"rectangle at {cdx:g} m, larger than the "
                    f"{size}x{size} the per-tile budget allows")
            tile_sites = members[local]
            last_i = i_start - 1 + span_i
            last_j = j_start - 1 + span_j
            if (i_start - 1 < clearance or j_start - 1 < clearance
                    or last_i > nx - clearance or last_j > ny - clearance):
                bad = [str(s) for s in site_ids[tile_sites][:10]]
                raise SiteOutsideParent(
                    f"a tile around site(s) {', '.join(bad)} would reach "
                    f"within {clearance} cells of the edge of {parent_id} "
                    f"({nx}x{ny} at {pdx:g} m), into its boundary zone; "
                    "the sites lie outside the region the parent can force")
            i0 = _placement_point(i_start, span_i)
            j0 = _placement_point(j_start, span_j)
            from woof.downscale import _centered_placement
            from woof.offline_child import OfflineChildContractError

            try:
                placement = _centered_placement(
                    {"nx": nx, "ny": ny}, j0=j0, i0=i0, ratio=ratio,
                    child_nx=child_nx, child_ny=child_ny)
            except (OfflineChildContractError, ValueError) as error:
                raise SiteOutsideParent(
                    f"woof downscale cannot place a {child_nx}x{child_ny} "
                    f"tile in {parent_id}: {error}") from error
            if (placement.i_parent_start, placement.j_parent_start) != (
                    i_start, j_start):
                raise GeometryContractError(
                    f"placement arithmetic disagrees with woof downscale: "
                    f"planned start ({i_start}, {j_start}), downscale "
                    f"({placement.i_parent_start}, "
                    f"{placement.j_parent_start})")
            c_lat, c_lon = grid.ij_to_latlon(float(i0 + 1), float(j0 + 1))
            c_lat = float(np.asarray(c_lat))
            c_lon = (float(np.asarray(c_lon)) + 180.0) % 360.0 - 180.0
            child_grid = grid.nest(i_start, j_start, ratio, child_nx + 1,
                                   child_ny + 1)
            ci, cj = (np.asarray(v, dtype=np.float64) for v in
                      child_grid.latlon_to_ij(lat[tile_sites],
                                              lon[tile_sites]))
            if np.any((ci < 0.5) | (ci > child_nx + 0.5)
                      | (cj < 0.5) | (cj > child_ny + 0.5)):
                raise GeometryContractError(
                    "cover_with_rectangles assigned sites to a rectangle "
                    "that does not contain them once snapped to parent cells")
            edge_cells = np.minimum.reduce([ci - 0.5, child_nx + 0.5 - ci,
                                            cj - 0.5, child_ny + 0.5 - cj])
            # Priced as downscale derives it, centred on the mass point
            # the placement is centred on, with the overrides applied.
            peak, merged = pricer.price(
                key, cfg, ratio=ratio, nx=child_nx, ny=child_ny,
                **{**price_kwargs, "centre_lat": c_lat})
            if leaf:
                counter["leaf"] += 1
                domain_id = f"tile-{counter['leaf']:04d}"
                role = "child"
            else:
                counter["inter"] += 1
                domain_id = f"inter-{counter['inter']:03d}"
                role = "parent"
            from woof.core.preflight import GIB

            tile = _Tile(
                domain_id=domain_id, role=role, level=level, ratio=ratio,
                dx=cdx, nx=child_nx, ny=child_ny, i_parent_start=i_start,
                j_parent_start=j_start, i0=i0, j0=j0, centre_lat=c_lat,
                centre_lon=c_lon, grid=child_grid, parent_id=parent_id,
                parent_wrf_grid_id=parent_wrf_grid_id,
                wrf_grid_id=parent_wrf_grid_id + 1, sites=tile_sites,
                run_config=merged, peak_envelope_gib=peak / GIB,
                levels=levels,
                min_edge_distance_m=float(np.min(edge_cells)) * cdx / mapfac,
                overrides=dict(overrides),
                parent_auto_epssm=bool(parent_auto_epssm))
            all_tiles.append(tile)
            if not leaf:
                inner_clearance = (int(merged["spec_bdy_width"])
                                   + TERRAIN_BLEND_ROWS + SINT_STENCIL_ROWS)
                plan_level(grid=child_grid, nx=child_nx, ny=child_ny,
                           pdx=cdx, cfg=merged, clearance=inner_clearance,
                           members=tile_sites, ratios=ratios[1:],
                           parent_id=domain_id,
                           parent_wrf_grid_id=tile.wrf_grid_id,
                           level=level + 1, margins=margins[1:],
                           # _with_parent_epssm_label labels this tile's
                           # own epssm when its parent's was labelled.
                           parent_auto_epssm=tile.parent_auto_epssm)
            if max_domains is not None and len(all_tiles) > max_domains:
                raise TooManyTiles(
                    f"the cover needs more than {max_domains} tiles "
                    f"(--max-domains {max_domains}); raise --max-domains, "
                    "use a larger --card/--vram-gib for bigger tiles, or a "
                    "smaller --corridor-km")
        if np.any(seen != 1):
            raise GeometryContractError(
                "cover_with_rectangles did not make every site a member of "
                "exactly one rectangle")

    plan_level(grid=parent.grid, nx=parent.nx, ny=parent.ny, pdx=parent.dx,
               cfg=parent.run_config, clearance=parent.clearance_cells,
               members=np.arange(len(sites)), ratios=chain,
               parent_id="parent", parent_wrf_grid_id=1, level=1,
               margins=_level_margins(chain, parent.dx, corridor_m),
               parent_auto_epssm=parent.auto_epssm)

    domains = [_parent_domain(parent, outdir)]
    for tile in all_tiles:
        domains.append(_tile_domain(
            tile, outdir=outdir, site_ids=site_ids, capacity=capacity,
            capacity_flags=capacity_flags, parent=parent,
            start=start_text))

    notes += _plan_notes(parent=parent, chain=chain, dx_m=dx_m,
                         corridor_m=corridor_m, leaf_levels_n=leaf_levels_n,
                         source=source, start=start_text,
                         explicit_start=start is not None)
    plan = Plan(topology=TOPOLOGY, dx_m=dx_m, start=start_text,
                hours=float(hours_i), domains=domains, source=source,
                sites_ref={"count": len(sites),
                           "content_sha256": _content_sha256(sites)},
                notes=notes)
    dump_plan(plan, outdir / "plan.json")
    return plan


def _parent_domain(parent: _Parent, outdir: Path) -> PlanDomain:
    return PlanDomain(
        domain_id="parent", topology=TOPOLOGY, role="parent", dx_m=parent.dx,
        run_dir="runs/parent", output_glob=PARENT_OUTPUT_GLOB,
        footprint=_ring(parent.grid, parent.nx, parent.ny),
        config=_relpath(parent.toml, outdir),
        wps_namelist=_relpath(parent.wps, outdir), grid_id=1, parent=None,
        extra={
            "nx": parent.nx, "ny": parent.ny,
            "history_interval_s": parent.history_interval_s,
            "restart_interval_s": float(
                parent.run_config.get("restart_interval_s") or 0.0),
            "run_seconds": parent.run_seconds,
            "boundary_clearance_cells": parent.clearance_cells,
            "peak_envelope_gib": (None if parent.peak_envelope_gib is None
                                  else round(parent.peak_envelope_gib, 3)),
            "region": _relpath(parent.toml.parent / "region.geojson", outdir),
            "wizard_log": _relpath(parent.toml.parent / "woof-domain.log",
                                   outdir),
        })


class _ChildConfig:
    """One tile's written ``--child-config`` TOML and what was proven."""

    def __init__(self, *, path: Path, relpath: str, sha256: str,
                 history_preset: str, verified_keys: tuple[str, ...]):
        self.path = path
        self.relpath = relpath
        self.sha256 = sha256
        self.history_preset = history_preset
        self.verified_keys = verified_keys


def _child_config_text(tile: _Tile) -> str:
    """The tile's ``--child-config`` TOML, rendered as ``--point`` renders
    its derived config, with the LES recipe already in ``[run]`` (it is in
    ``tile.run_config``) and the leaf's ``[output]`` preset appended."""

    from types import SimpleNamespace

    from woof.downscale import (_derived_static_table, _render_child_toml,
                                _with_parent_epssm_label)
    from woof.io.restart import AUTO_EPSSM_HEADER_KEY

    config = _with_parent_epssm_label(
        dict(tile.run_config),
        {AUTO_EPSSM_HEADER_KEY: True} if tile.parent_auto_epssm else {})
    head = [
        f"# woof energy plan --topology wrf-tiles: {tile.domain_id} "
        f"({tile.role}, {tile.dx:g} m, ratio {tile.ratio} from "
        f"{tile.parent_id}).",
        "# Run as `woof downscale <parent frames> --child-config <this file> "
        "--child-config-sha256 ...`;",
        "# an edited file no longer matches the sha256 the plan recorded and "
        "is refused.",
    ]
    if tile.overrides:
        head.append(
            "# LES gray-zone closure in [run], from " + LES_RECIPE_SOURCE
            + ": " + ", ".join(f"{key} = {value}" for key, value
                               in tile.overrides.items()) + ".")
    head.append("# The body below is rendered by woof downscale's own --point "
                "renderer; its header follows.")
    # The [static] table --point writes with no terrain or GEOG flag.
    static = _derived_static_table(
        SimpleNamespace(parent_terrain=False, geog_root=None))
    text = "\n".join(head) + "\n" + _render_child_toml(config) + static
    if tile.role == "child":
        text += f'\n[output]\npreset = "{ENERGY_HISTORY_PRESET}"\n'
    return text


def _write_child_config(tile: _Tile, *, outdir: Path,
                        start: str) -> _ChildConfig:
    """Write ``tiles/<id>.toml`` and prove ``woof downscale`` reads back
    exactly what was planned: the grid, every LES override and the history
    preset.  Any disagreement refuses the plan; no key is dropped."""

    from woof.config import load_history_selection
    from woof.experiment import experiment_from_run_config
    from woof.offline_child import (require_offline_child_root_forcing,
                                    resolve_child_run_config)

    path = outdir / "tiles" / f"{tile.domain_id}.toml"
    path.write_text(_child_config_text(tile), encoding="utf-8", newline="\n")
    try:
        # The --child-config door's own loader (load_config +
        # validate_run_config), and its specified-boundary gate.
        cfg = resolve_child_run_config(path)
        require_offline_child_root_forcing(cfg)
        selection = load_history_selection(path)
        experiment_from_run_config(
            cfg, datetime.strptime(start, "%Y-%m-%dT%H")
            .replace(tzinfo=timezone.utc))
    except (ValueError, TypeError, KeyError) as error:
        raise TilePlanRefusal(
            f"woof downscale would refuse the child config planned for "
            f"{tile.domain_id} ({path}): {error}") from error
    expected = {"nx": tile.nx, "ny": tile.ny,
                "nz": int(tile.run_config["nz"]),
                "dx": float(tile.run_config["dx"]),
                "dt": float(tile.run_config["dt"]),
                "grid_id": tile.wrf_grid_id, **tile.overrides}
    mismatched = []
    for key, want in expected.items():
        got = getattr(cfg, key, None)
        same = (math.isclose(float(got), float(want), rel_tol=0.0,
                             abs_tol=1e-9)
                if isinstance(want, float) and got is not None
                else got == want)
        if not same:
            mismatched.append(f"{key} = {want!r} planned, {got!r} read back")
    if tile.levels is not None and (
            cfg.eta_levels is None
            or tuple(float(v) for v in cfg.eta_levels)
            != tuple(float(v) for v in tile.levels)):
        mismatched.append("eta_levels: the planned ladder did not round-trip")
    want_preset = ENERGY_HISTORY_PRESET if tile.role == "child" else "full"
    if selection.preset != want_preset:
        mismatched.append(f"[output] preset = {want_preset!r} planned, "
                          f"{selection.preset!r} read back")
    if mismatched:
        raise TilePlanRefusal(
            f"the child config planned for {tile.domain_id} ({path}) does not "
            "read back as planned through woof downscale's loader: "
            + "; ".join(mismatched))
    return _ChildConfig(
        path=path, relpath=_relpath(path, outdir),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        history_preset=selection.preset,
        verified_keys=tuple(sorted(tile.overrides)))


def _downscale_args(tile: _Tile, *, run_dir: str, child: _ChildConfig,
                    capacity_flags: list[str], parent: _Parent) -> list[str]:
    """``woof downscale <parent frames>`` arguments for one tile: the
    ``--child-config`` route with the explicit placement it requires."""

    from woof.downscale import PARENT_RESTART_LATEST

    return ["--parent-domain", str(tile.parent_wrf_grid_id),
            "--parent-restart", PARENT_RESTART_LATEST,
            "--child-config", child.relpath,
            "--child-config-sha256", child.sha256,
            "--ratio", str(tile.ratio),
            "--i-parent-start", str(tile.i_parent_start),
            "--j-parent-start", str(tile.j_parent_start),
            "--max-boundary-interval-seconds",
            f"{parent.history_interval_s:g}",
            *capacity_flags,
            "--render-products", "none",
            "--out", run_dir]


def _tile_domain(tile: _Tile, *, outdir: Path, site_ids: np.ndarray,
                 capacity: float, capacity_flags: list[str],
                 parent: _Parent, start: str) -> PlanDomain:
    parent_run_dir = ("runs/parent" if tile.parent_id == "parent"
                      else f"runs/{tile.parent_id}")
    # The positional downscale argument is the parent's FRAMES directory:
    # the parent run's wrfout/ folder, or an intermediate tile's --out.
    parent_frames = (f"{parent_run_dir}/{Path(PARENT_OUTPUT_GLOB).parent}"
                     if tile.parent_id == "parent" else parent_run_dir)
    run_dir = f"runs/{tile.domain_id}"
    child = _write_child_config(tile, outdir=outdir, start=start)
    downscale_args = _downscale_args(
        tile, run_dir=run_dir, child=child, capacity_flags=capacity_flags,
        parent=parent)
    footprint = _ring(tile.grid, tile.nx, tile.ny)
    owned = (tuple(str(s) for s in site_ids[tile.sites])
             if tile.role == "child" else ())
    leaf = tile.role == "child"
    via = f"--child-config {child.relpath}"
    if tile.overrides:
        physics = {
            "recipe": dict(tile.overrides),
            "source": LES_RECIPE_SOURCE,
            "applied": True,
            "via": f"[run] of {child.relpath} ({via})",
            "verified_keys": list(child.verified_keys),
            "verified_with": "woof.offline_child.resolve_child_run_config "
                             "(the --child-config door's loader)",
        }
    else:
        physics = {
            "recipe": {},
            "source": None,
            "applied": None,
            "reason": (
                "intermediate tile: it forces the tiles below it with its "
                "parent's physics, inherited as woof downscale --point "
                "inherits them" if not leaf else
                f"not in the LES regime (coarser than {LES_CHILD_DX_M:g} "
                "m); parent physics inherited as woof downscale --point "
                "inherits them"),
        }
    if leaf:
        history = {"preset": child.history_preset, "applied": True,
                   "via": f"[output] of {child.relpath} ({via})",
                   "interval_s": parent.history_interval_s}
    else:
        history = {"preset": child.history_preset, "applied": True,
                   "reason": _INTERMEDIATE_HISTORY,
                   "interval_s": parent.history_interval_s}
    spec = {
        "schema": TILE_SCHEMA,
        "domain_id": tile.domain_id,
        "role": tile.role,
        "parent": tile.parent_id,
        "parent_run_dir": parent_run_dir,
        "parent_frames_glob": parent_frames,
        "run_dir": run_dir,
        "level": tile.level,
        "ratio": tile.ratio,
        "dx_m": tile.dx,
        "nx": tile.nx,
        "ny": tile.ny,
        "centre": {"lat": tile.centre_lat, "lon": tile.centre_lon},
        "parent_mass_point_0based": {"i": tile.i0, "j": tile.j0},
        "i_parent_start": tile.i_parent_start,
        "j_parent_start": tile.j_parent_start,
        "parent_domain": tile.parent_wrf_grid_id,
        "wrf_grid_id": tile.wrf_grid_id,
        "footprint": [list(p) for p in footprint],
        "site_count": int(tile.sites.size),
        "min_site_edge_distance_m": (None if tile.min_edge_distance_m is None
                                     else round(tile.min_edge_distance_m, 1)),
        "dt_s": float(tile.run_config["dt"]),
        "nz": int(tile.run_config["nz"]),
        "child_config": {
            "path": child.relpath,
            "sha256": child.sha256,
            "derived_with": _DERIVATION,
            "epssm_auto_label": bool(tile.parent_auto_epssm),
        },
        "child_levels": (None if tile.levels is None else
                         {"nz": len(tile.levels) - 1,
                          "stretch": CHILD_LEVEL_STRETCH, "applied": True,
                          "via": f"eta_levels in {child.relpath}"}),
        "lateral_zone": {
            key: tile.run_config[key]
            for key in ("spec_zone", "relax_zone", "spec_bdy_width",
                        "relax_timescale_s")},
        "physics_overrides": physics,
        "history": history,
        "vram": {"peak_envelope_gib": round(tile.peak_envelope_gib, 3),
                 "capacity_gib": capacity,
                 "basis": "woof.core.preflight.estimate_experiment on the "
                          "child RunConfig in child_config (derived as "
                          "woof downscale --point derives it, with the "
                          "physics overrides applied)"},
        "downscale_args": downscale_args,
        # The orchestrator expands parent_frames_glob to the one frames
        # directory of the finished parent run and substitutes it here.
        "command_template": ["woof", "downscale", "{parent_frames}",
                             *downscale_args],
        "paths_relative_to": "plan directory",
    }
    spec_path = outdir / "tiles" / f"{tile.domain_id}.json"
    _write_json(spec, spec_path)
    return PlanDomain(
        domain_id=tile.domain_id, topology=TOPOLOGY, role=tile.role,
        dx_m=tile.dx, run_dir=run_dir,
        output_glob=f"wrfout_d{tile.wrf_grid_id:02d}_*",
        footprint=footprint, config=_relpath(spec_path, outdir),
        wps_namelist=None, grid_id=1, parent=tile.parent_id, site_ids=owned,
        extra={
            "downscale_args": downscale_args,
            "child_config": child.relpath,
            "child_config_sha256": child.sha256,
            "ratio": tile.ratio, "nx": tile.nx, "ny": tile.ny,
            "parent_domain": tile.parent_wrf_grid_id,
            "wrf_grid_id": tile.wrf_grid_id,
            "peak_envelope_gib": round(tile.peak_envelope_gib, 3),
            "level": tile.level,
        })


def _plan_notes(*, parent: _Parent, chain: tuple[int, ...], dx_m: float,
                corridor_m: float, leaf_levels_n: int | None,
                source: str | None, start: str,
                explicit_start: bool) -> list[str]:
    notes = []
    if not explicit_start:
        notes.append(f"start defaulted to the most recent 00/06/12/18 UTC "
                     f"cycle, {start}; the forcing source must have "
                     "published it before the parent can be fetched")
    notes.append(
        f"refinement chain {parent.dx:g} m -> "
        + " -> ".join(f"{parent.dx / math.prod(chain[:k + 1]):g} m"
                      for k in range(len(chain)))
        + f" (ratios {', '.join(str(r) for r in chain)}); each step is one "
        "woof downscale invocation")
    notes.append(
        f"the parent root at {parent.dx:g} m is forced directly by "
        f"{source or 'the woof domain default source'} at that source's "
        "native spacing; see parent/woof-domain.log for the wizard's "
        "physics and gray-zone advisories")
    notes.append(
        "tiles are one-way offline children (woof downscale --child-config "
        "tiles/<id>.toml with explicit --ratio/--i-parent-start/"
        "--j-parent-start placement, each TOML bound by the sha256 in "
        "its downscale_args): "
        f"boundaries every {parent.history_interval_s:g} s from the parent "
        "history, which is also each tile's history cadence")
    notes.append(
        "launch every plan command from the plan directory: the parent "
        "TOML's [fetch].out and the tiles' downscale_args (--child-config, "
        "--out) are relative to it.  The parent runs as `woof go <config> --outdir "
        f"runs/parent` and its frames match {PARENT_OUTPUT_GLOB!r}; the "
        "orchestrator runs `woof downscale <parent frames directory> "
        "<downscale_args>`, and --parent-restart latest finds the parent's "
        "checkpoints in the run directory above that wrfout/ folder.  A "
        "tile writes its frames directly in its --out (= run_dir), which "
        "is create-only")
    notes.append(
        "each tile's child TOML is derived by " + _DERIVATION + ".  This "
        "is an approximation of --point in one respect: a value the "
        "parent's run settles only at run time and records in its "
        "checkpoint is not seen.  The adaptive step does not matter (the "
        "child takes a fixed step of its own either way) and a "
        "model-chosen epssm is handed down labelled, as --point does; "
        "woof downscale still binds the parent's microphysics from "
        "--parent-restart latest and checks the child's scheme against it")
    if leaf_levels_n is not None:
        notes.append(
            f"leaf tiles carry their own {leaf_levels_n}-level ladder "
            f"(eta_levels in tiles/<id>.toml, built as --child-levels "
            f"{leaf_levels_n},{CHILD_LEVEL_STRETCH:g} builds it), the "
            "vertical half of the LES recipe")
    if dx_m <= LES_CHILD_DX_M:
        notes.append(
            "LES gray-zone closure applied to every leaf tile at "
            f"{dx_m:g} m: "
            + ", ".join(f"{key} = {value}"
                        for key, value in LES_PHYSICS_RECIPE.items())
            + f" in the [run] table of tiles/<id>.toml ({LES_RECIPE_SOURCE}),"
            " each key read back through woof downscale's --child-config "
            "loader before the plan was written")
    else:
        notes.append(
            f"leaf tiles at {dx_m:g} m are coarser than the "
            f"{LES_CHILD_DX_M:g} m LES regime: they inherit the parent's "
            "physics, as woof downscale --point would")
    notes.append(
        f"history preset {ENERGY_HISTORY_PRESET!r} applied to every leaf "
        "tile ([output] in tiles/<id>.toml)"
        + ("; intermediate tiles write the full inventory: "
           + _INTERMEDIATE_HISTORY if len(chain) > 1 else ""))
    zone_m = (1 + _relax_cells(chain[-1]) + TERRAIN_BLEND_ROWS) * dx_m
    if corridor_m <= zone_m:
        notes.append(
            f"--corridor-km {corridor_m / 1000:g} is no wider than a leaf "
            f"tile's boundary zone plus terrain blend ({zone_m:g} m): sites "
            "near a tile edge sit inside the zone where the parent's state "
            "is imposed")
    return notes


# --------------------------------------------------------------------------
# CLI


def main(args) -> int:
    """``woof energy plan --topology wrf-tiles``."""

    sites_path = Path(args.sites)
    outdir = Path(args.outdir)
    sites = load_sites(sites_path)
    try:
        plan = build_plan(
            sites, outdir=outdir, dx_m=args.dx_m,
            corridor_km=args.corridor_km, parent_dx_m=args.parent_dx_m,
            start=args.start, hours=args.hours, source=args.source,
            card=args.card, vram_gib=args.vram_gib,
            max_domains=args.max_domains, nz=args.nz)
    except EnergyNotImplemented as error:
        import sys

        print(f"woof energy plan: {error}; the wrf-tiles planner needs it",
              file=sys.stderr)
        return 2
    from woof.energy.contracts import sha256_file

    plan.sites_ref = {
        "path": _relpath(sites_path, outdir),
        "sha256": sha256_file(sites_path),
        **(plan.sites_ref or {}),
    }
    plan_path = dump_plan(plan, outdir / "plan.json")
    parent = plan.domain("parent")
    tiles = [d for d in plan.domains if d.domain_id != "parent"]
    leaves = [d for d in tiles if d.role == "child"]
    record = {
        "schema": PLAN_SCHEMA,
        "topology": TOPOLOGY,
        "plan": str(plan_path),
        "start": plan.start,
        "hours": plan.hours,
        "dx_m": plan.dx_m,
        "parent": {
            "config": str(outdir / parent.config),
            "wps_namelist": str(outdir / parent.wps_namelist),
            "dx_m": parent.dx_m,
            "nx": parent.extra["nx"],
            "ny": parent.extra["ny"],
            "peak_envelope_gib": parent.extra["peak_envelope_gib"],
        },
        "tile_count": len(tiles),
        "leaf_tiles": len(leaves),
        "intermediate_tiles": len(tiles) - len(leaves),
        "sites": len(sites),
        "tiles": [{"domain_id": d.domain_id, "role": d.role,
                   "parent": d.parent, "dx_m": d.dx_m,
                   "nx": d.extra["nx"], "ny": d.extra["ny"],
                   "peak_envelope_gib": d.extra["peak_envelope_gib"],
                   "sites": len(d.site_ids), "spec": str(outdir / d.config)}
                  for d in tiles],
        "notes": plan.notes,
    }
    print(json.dumps(record, indent=2, default=str))
    return 0


__all__ = [
    "TOPOLOGY", "TILE_SCHEMA", "PARENT_OUTPUT_GLOB", "LES_PHYSICS_RECIPE", "TilePlanRefusal",
    "SiteOutsideParent", "TooManyTiles", "GeometryContractError",
    "build_plan", "default_start", "factor_chain", "main",
]
