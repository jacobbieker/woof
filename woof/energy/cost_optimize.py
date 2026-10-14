"""``woof energy hindcast optimize``: the cheapest hindcast that meets a deadline.

An ESTIMATE built from estimates: every configuration is priced by
:func:`woof.energy.cost.estimate` (its throughput model, utilisation band
and dated price table), and the ranking is only as good as they are.  The
output says so on every row.

What is searched
----------------

The product of

* **leaf spacing** ``--leaf-dx-m`` (default :data:`DEFAULT_LEAF_DX_CANDIDATES`,
  each refused outside the reanalysis range 100-300 m);
* **levels** ``--nz`` (default :data:`DEFAULT_NZ_CANDIDATES`); each count
  must pass :func:`woof.energy.vertical.lean_column_refusal` (>= 8 layers
  below 250 m, 12 m lowest layer, top <= 20 km, no layer deeper than
  1000 m), so nothing below :data:`woof.energy.vertical.HINDCAST_NZ` (49)
  is admitted with the preset controls;
* **segment layout**: ``--spinup-hours`` (default 3, the hindcast driver's)
  and ``--keep-hours`` (default 6: one run per 00/06/12/18 UTC init), so a
  run is ``keep + spin-up`` hours and a period of ``--period-days``
  (default 365) needs ``ceil(period / keep)`` runs;
* **offers**: every price-table row (``--instance`` narrows it).

Each spacing is re-partitioned per card memory, because the clusters (their
bisection to a parent chain that fits) depend on both: the sites are cut
in-process with :func:`woof.energy.partition.build_partition`, with the
source partition's linkage/extent/site/corridor parameters, once per
``(rounded GiB, leaf spacing)`` and cached (in memory, and on disk under
``--partition-dir`` where the chosen configuration's partition can be
picked up).  ``--no-repartition`` prices the given partition as cut: only
its spacing, and only cards with at least the memory it was sized for.

Every domain carries the configuration's ``nz`` (the wind-energy column is
the whole model's), so parents are priced at ``nz`` levels too.  One cut
serves every ``nz``: a cluster is bisected until its parent chain fits the
card, and that chain is sized at the wizard's parent levels, never the leaf
count; ``nz`` moves only the partition's ``estimated_tiles``, which the
estimator does not read (it re-tiles each cluster to each card at the
priced ``nz``, :func:`woof.energy.cost.synth_cluster`).

What is pinned
--------------

* A configuration meets the deadline when one segment's critical path fits
  in it and, with ``--max-instances``, the instance count the central
  estimate needs (``ceil(billed hours / deadline hours)``, spot hours
  carrying the rework) is within the cap.  The count the pessimistic band
  needs is reported beside it.
* Feasible rows are ranked by the ``--rank-market`` (default spot) central
  cost, then its pessimistic cost, then wall clock; the first is
  ``chosen``.  Storage is reported, never ranked.
* The output is ``woof-energy.cost-optimize.v1`` on stdout, a table on
  stderr; ``-o`` also writes it atomically.  Refusals exit 2 with
  ``{"schema": ..., "refused": ...}``.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from woof.card_catalog import card_row
from woof.energy import cost
from woof.energy.contracts import _write_json

OPTIMIZE_SCHEMA = "woof-energy.cost-optimize.v1"

#: Leaf spacings tried by default: the cheap end of the 100-300 m range.
DEFAULT_LEAF_DX_CANDIDATES = (300.0, 250.0, 200.0)
#: Level counts tried by default: the lean column, the LES leaf ladder and
#: the 70-level preset.
DEFAULT_NZ_CANDIDATES = (49, 60, 70)
#: The hindcast driver's spin-up and kept window (woof.energy.hindcast).
DEFAULT_SPINUP_CANDIDATES = (3,)
DEFAULT_KEEP_CANDIDATES = (6,)
DEFAULT_PERIOD_DAYS = 365.0
DEFAULT_TOP = 10
#: Spin-ups the chosen configuration is re-priced at, so the cost of the
#: 3 h default is visible (``spinup_sensitivity``).
SENSITIVITY_SPINUPS = (0, 1, 2, 3)

#: Card memory (GiB) is rounded to this many digits to share partitions
#: between parts that differ by a few MiB (A10G and L4 both present
#: ~22.5 GiB; H100 and A100 80 GB ~80).
_GIB_DIGITS = 0


class OptimizeRefused(cost.CostRefusal):
    """The search cannot run as asked, and the message names why."""


@dataclass(frozen=True)
class Layout:
    keep_hours: int
    spinup_hours: int

    @property
    def segment_hours(self) -> int:
        return self.keep_hours + self.spinup_hours

    def segments(self, period_hours: float) -> int:
        return cost.default_segments(self.segment_hours, self.spinup_hours,
                                     period_hours=period_hours)


_sha256 = cost._sha256


# --------------------------------------------------------------------------
# partitions per (card memory, leaf spacing)
# --------------------------------------------------------------------------
_PARTITION_PARAMS = ("linkage_km", "max_extent_km", "max_sites", "corridor_km",
                     "topology")


class PartitionCache:
    """``build_partition`` once per ``(GiB, leaf spacing)``.

    ``directory`` (optional) keeps each partition as
    ``partition-<dx>m-<gib>gib.json``; a file already there is reused when
    its sites sha256, spacing, capacity and parameters match, so a second
    search does not re-cut the sites.
    """

    def __init__(self, sites_path: Path, params: Mapping[str, Any], *,
                 directory: Path | None = None, log=None):
        from woof.energy.contracts import load_sites

        self.sites_path = Path(sites_path)
        self.sites_sha = _sha256(self.sites_path)
        self.sites = load_sites(self.sites_path)
        self.params = {k: params[k] for k in _PARTITION_PARAMS if k in params}
        self.directory = Path(directory) if directory is not None else None
        self._memo: dict[tuple[float, float], dict[str, Any]] = {}
        self.log = log or (lambda message: None)
        self.built: list[dict[str, Any]] = []

    def get(self, *, gib_key: float, dx: float, card_key: str) -> dict[str, Any]:
        key = (gib_key, float(dx))
        if key in self._memo:
            return self._memo[key]
        from woof.energy import partition as part

        path = None
        if self.directory is not None:
            path = self.directory / f"partition-{dx:g}m-{gib_key:g}gib.json"
        partition = None
        if path is not None and path.is_file():
            try:
                cached = part.load_partition(path)
            except (OSError, ValueError):
                cached = None
            if cached is not None and self._matches(cached, dx, card_key):
                partition = cached
                origin = "reused"
        started = time.monotonic()
        if partition is None:
            # A partition names its sites relative to where it is written
            # (the working directory when it stays in memory).
            sites_ref = {"path": _relpath(self.sites_path,
                                          path.parent if path is not None
                                          else Path.cwd()),
                         "sha256": self.sites_sha}
            try:
                partition = part.build_partition(
                    self.sites, sites_ref=sites_ref, leaf_dx_m=float(dx),
                    card=card_key, **self.params)
            except ValueError as error:
                raise OptimizeRefused(
                    f"partitioning at {dx:g} m for {card_key}: {error}") from error
            origin = "built"
            if path is not None:
                part.dump_partition(partition, path)
        seconds = time.monotonic() - started
        record = {
            "leaf_dx_m": float(dx), "card": card_key,
            "vram_gib": partition.capacity.get("vram_gib"),
            "clusters": len(partition.clusters),
            "path": None if path is None else str(path),
            "origin": origin, "seconds": round(seconds, 2),
            "specs": _specs(partition, dx),
        }
        self.log(f"partition {dx:g} m for {card_key} "
                 f"({record['vram_gib']:.1f} GiB): {record['clusters']} "
                 f"clusters, {origin} in {seconds:.1f} s")
        self._memo[key] = record
        self.built.append({k: v for k, v in record.items() if k != "specs"})
        return record

    def _matches(self, cached, dx: float, card_key: str) -> bool:
        params = cached.parameters
        return (cached.sites_ref.get("sha256") == self.sites_sha
                and math.isclose(float(params.get("leaf_dx_m", -1)), float(dx))
                and cached.capacity.get("card") == card_key
                and all(params.get(k) == v for k, v in self.params.items()))


def _relpath(path: Path, start: Path) -> str:
    from woof.energy.partition import _relpath as relpath

    return relpath(Path(path), Path(start))


def _specs(partition, dx: float) -> list[cost.ClusterSpec]:
    document = {"clusters": [{"cluster_id": c.cluster_id, "region": c.region,
                              "leaf_dx_m": c.leaf_dx_m}
                             for c in partition.clusters]}
    return cost.clusters_from_document(document, "partition", leaf_dx_m=dx,
                                       cover_factor=cost.DEFAULT_COVER_FACTOR)


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------
def check_leaf_dx(values: Sequence[float]) -> tuple[float, ...]:
    from woof.energy.partition import LEAF_DX_MAX_M, LEAF_DX_MIN_M

    out = []
    for value in values:
        dx = float(value)
        if not (math.isfinite(dx) and LEAF_DX_MIN_M <= dx <= LEAF_DX_MAX_M):
            raise OptimizeRefused(
                f"--leaf-dx-m {value!r} is outside the reanalysis range "
                f"[{LEAF_DX_MIN_M:g}, {LEAF_DX_MAX_M:g}] m")
        if dx not in out:
            out.append(dx)
    return tuple(out)


def check_nz(values: Sequence[int], *, p_top_pa: float | None = None
             ) -> tuple[int, ...]:
    """Every count must give a lean wind-energy column (refused otherwise,
    naming the smallest that does when the count is too small)."""

    from woof.energy.vertical import HINDCAST_NZ, lean_column_refusal

    out = []
    for value in values:
        nz = int(value)
        try:
            reason = lean_column_refusal(nz, p_top_pa=p_top_pa)
        except (OSError, RuntimeError, ImportError) as error:
            raise OptimizeRefused(
                f"--nz {nz}: the wind-energy column could not be generated "
                f"to check it ({error}); the WRF compute_eta bridge is "
                "needed") from error
        if reason is not None:
            hint = (f"; the smallest lean column has {HINDCAST_NZ} levels"
                    if nz < HINDCAST_NZ else "")
            raise OptimizeRefused(f"--nz {nz} is not a lean wind-energy "
                                  f"column: {reason}{hint}")
        if nz not in out:
            out.append(nz)
    return tuple(out)


def check_layouts(keeps: Sequence[int], spinups: Sequence[int]
                  ) -> tuple[Layout, ...]:
    layouts = []
    for keep in keeps:
        if int(keep) < 6 or int(keep) % 6:
            raise OptimizeRefused(
                f"--keep-hours {keep} must be a positive multiple of the "
                "6-hourly global cycle (6, 12, 18, 24 ...) so every run "
                "starts at an init")
        for spin in spinups:
            if int(spin) < 0:
                raise OptimizeRefused(f"--spinup-hours {spin} is negative")
            layouts.append(Layout(int(keep), int(spin)))
    return tuple(dict.fromkeys(layouts))


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------
def _gib_key(gib: float) -> float:
    return round(float(gib), _GIB_DIGITS)


def _offer_name(offer: Mapping[str, Any]) -> str:
    return f"{offer['provider']}:{offer['instance']}"


def _deadline_fit(row: Mapping[str, Any], market: str, deadline_days: float,
                  max_instances: int | None) -> dict[str, Any]:
    hours = row["spot_instance_hours"] if market == "spot" else row["instance_hours"]
    budget = deadline_days * 24.0
    critical = row["critical_path_hours_per_segment"]
    need = {band: max(1, math.ceil(hours[band] / budget))
            for band in ("central", "high")}
    instances = need["central"]
    if max_instances is not None:
        instances = min(instances, max_instances)
    wall = max(hours["central"] / instances, critical) / 24.0
    reasons = []
    if critical > budget:
        reasons.append(f"one segment's chain takes {critical:.1f} h, longer "
                       f"than the {deadline_days:g}-day deadline")
    if max_instances is not None and need["central"] > max_instances:
        reasons.append(f"needs {need['central']} instances, above "
                       f"--max-instances {max_instances}")
    return {"market": market, "instances": instances,
            "instances_needed": need,
            "wall_clock_days": round(wall, 3),
            "meets_deadline": not reasons, "reasons": reasons}


def _row(config: Mapping[str, Any], offer_row: Mapping[str, Any], *,
         market: str, deadline_days: float, max_instances: int | None,
         partition: Mapping[str, Any]) -> dict[str, Any]:
    fit = _deadline_fit(offer_row, market, deadline_days, max_instances)
    costs = {m: {k: v for k, v in c.items()}
             for m, c in offer_row["cost_usd"].items()}
    return {
        **config,
        "offer": f"{offer_row['provider']}:{offer_row['instance']}",
        "provider": offer_row["provider"], "instance": offer_row["instance"],
        "card": offer_row["card"]["key"],
        "gpus_per_instance": offer_row["gpus_per_instance"],
        "clusters": partition["clusters"],
        "gpu_hours": offer_row["gpu_hours"],
        "instance_hours": offer_row["instance_hours"],
        "spot_instance_hours": offer_row.get("spot_instance_hours"),
        "cost_usd": costs,
        "deadline": fit,
        "critical_path_hours_per_segment":
            offer_row["critical_path_hours_per_segment"],
        "utilisation": offer_row.get("utilisation"),
        "storage_tb": offer_row["storage"]["stored_tb"],
        "storage_usd_per_month": (offer_row.get("storage_usd") or {}).get(
            "per_month"),
        "partition": {k: partition[k] for k in ("leaf_dx_m", "card", "vram_gib",
                                                "clusters", "path")},
        "kind": "estimate",
    }


def _rank_key(row: Mapping[str, Any], market: str) -> tuple:
    c = row["cost_usd"][market]
    return (c["central"], c["high"], row["deadline"]["wall_clock_days"],
            row["offer"])


def optimize(*, partitions_for, offers: Sequence[Mapping[str, Any]],
             price_table: Mapping[str, Any], leaf_dx: Sequence[float],
             nz: Sequence[int], layouts: Sequence[Layout],
             period_hours: float, deadline_days: float,
             rank_market: str = "spot", max_instances: int | None = None,
             top: int = DEFAULT_TOP, store_preset: str = "full",
             packing_efficiency: float = cost.DEFAULT_PACKING_EFFICIENCY,
             log=None) -> dict[str, Any]:
    """Rank every configuration x offer (see the module docstring).

    ``partitions_for(gib_key, dx, card_key)`` returns a partition record
    (``specs``, ``clusters``, ``path`` ...) or raises
    :class:`OptimizeRefused`; it is how the caller decides between
    re-partitioning and pricing one partition as cut.
    """

    log = log or (lambda message: None)
    if deadline_days <= 0:
        raise OptimizeRefused("--deadline-days must be positive")
    if rank_market not in ("spot", "on_demand"):
        raise OptimizeRefused(f"rank market {rank_market!r} is not spot or "
                              "on_demand")
    from woof.energy.partition import resolve_capacity

    groups: dict[float, list[Mapping[str, Any]]] = {}
    for offer in offers:
        card = card_row(str(offer["card"]))
        assert card is not None
        gib = resolve_capacity(card=card.key)[0]
        groups.setdefault(_gib_key(gib), []).append(offer)
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    model = None
    for dx in leaf_dx:
        for gib_key, members in sorted(groups.items()):
            # The group's smallest card sizes the partition (conservative).
            smallest = min(members, key=lambda o: card_row(o["card"]).memory_mib)
            card_key = card_row(smallest["card"]).key
            try:
                partition = partitions_for(gib_key, dx, card_key)
            except OptimizeRefused as error:
                for offer in members:
                    excluded.append({"offer": _offer_name(offer),
                                     "leaf_dx_m": dx, "reason": str(error)})
                continue
            for levels, layout in itertools.product(nz, layouts):
                segments = layout.segments(period_hours)
                campaign = cost.Campaign(segments, float(layout.segment_hours),
                                         float(layout.spinup_hours))
                document = cost.estimate(
                    mode="partition", campaign=campaign, offers=members,
                    price_table=price_table, clusters=partition["specs"],
                    leaf_nz=int(levels), parent_nz=int(levels),
                    store_preset=store_preset,
                    packing_efficiency=packing_efficiency)
                model = model or document["model"]
                config = {"leaf_dx_m": dx, "nz": int(levels),
                          "segment_hours": layout.segment_hours,
                          "spinup_hours": layout.spinup_hours,
                          "keep_hours": layout.keep_hours,
                          "segments": segments}
                for offer_row in document["offers"]:
                    if not offer_row.get("fits", True) \
                            or rank_market not in offer_row["cost_usd"]:
                        excluded.append({
                            "offer": f"{offer_row['provider']}:"
                                     f"{offer_row['instance']}",
                            **config, "reason": offer_row.get(
                                "refused", f"no {rank_market} price")})
                        continue
                    rows.append(_row(config, offer_row, market=rank_market,
                                     deadline_days=deadline_days,
                                     max_instances=max_instances,
                                     partition=partition))
            log(f"priced {dx:g} m x {len(nz)} level count(s) x "
                f"{len(layouts)} layout(s) on {len(members)} offer(s) "
                f"({gib_key:g} GiB)")
    feasible = sorted((r for r in rows if r["deadline"]["meets_deadline"]),
                      key=lambda r: _rank_key(r, rank_market))
    infeasible = [r for r in rows if not r["deadline"]["meets_deadline"]]
    if not rows:
        raise OptimizeRefused("no configuration could be priced: "
                              + "; ".join(sorted({e['reason'] for e in excluded}))[:600])
    if not feasible:
        fastest = min(infeasible, key=lambda r: (r["critical_path_hours_per_segment"],
                                                 r["deadline"]["instances_needed"]["central"]))
        raise OptimizeRefused(
            f"no configuration meets the {deadline_days:g}-day deadline; the "
            f"closest ({fastest['offer']} at {fastest['leaf_dx_m']:g} m, "
            f"{fastest['nz']} levels) "
            + "; ".join(fastest["deadline"]["reasons"])
            + ".  Raise --deadline-days or --max-instances")
    for rank, row in enumerate(feasible, start=1):
        row["rank"] = rank
    other = "on_demand" if rank_market == "spot" else "spot"
    by_other = sorted((r for r in feasible if other in r["cost_usd"]),
                      key=lambda r: _rank_key(r, other))
    return {"rows": feasible, "top_other": by_other[:top],
            "infeasible": len(infeasible), "excluded": excluded,
            "model": model, "configurations": len(rows),
            "other_market": other}


def spinup_sensitivity(chosen: Mapping[str, Any], *, partitions_for,
                       offer: Mapping[str, Any], price_table, period_hours,
                       market: str, store_preset: str,
                       packing_efficiency: float) -> list[dict[str, Any]]:
    """The chosen configuration and offer re-priced at each of
    :data:`SENSITIVITY_SPINUPS` and its own spin-up (same kept window)."""

    partition = partitions_for(None, chosen["leaf_dx_m"], chosen["card"])
    out = []
    for spin in sorted(set(SENSITIVITY_SPINUPS) | {int(chosen["spinup_hours"])}):
        layout = Layout(int(chosen["keep_hours"]), int(spin))
        campaign = cost.Campaign(layout.segments(period_hours),
                                 float(layout.segment_hours), float(spin))
        document = cost.estimate(
            mode="partition", campaign=campaign, offers=[offer],
            price_table=price_table, clusters=partition["specs"],
            leaf_nz=chosen["nz"], parent_nz=chosen["nz"],
            store_preset=store_preset, packing_efficiency=packing_efficiency)
        row = document["offers"][0]
        out.append({"spinup_hours": spin,
                    "segment_hours": layout.segment_hours,
                    "gpu_hours": row["gpu_hours"]["central"],
                    "cost_usd": {m: c["central"]
                                 for m, c in row["cost_usd"].items()},
                    "relative_to_chosen": None})
    base = next((r for r in out if r["spinup_hours"] == chosen["spinup_hours"]),
                None)
    for r in out:
        if base and base["cost_usd"].get(market):
            r["relative_to_chosen"] = round(
                r["cost_usd"][market] / base["cost_usd"][market], 3)
    return out


# --------------------------------------------------------------------------
# recipe, rendering, CLI
# --------------------------------------------------------------------------
def recipe(chosen: Mapping[str, Any], *, sites: str, params: Mapping[str, Any],
           deadline_days: float, market: str) -> list[str]:
    """The commands that run the chosen configuration (start/end are the
    reader's)."""

    from woof.energy import partition as part

    flags = []
    defaults = {"linkage_km": part.DEFAULT_LINKAGE_KM,
                "max_extent_km": part.DEFAULT_MAX_EXTENT_KM,
                "max_sites": part.DEFAULT_MAX_SITES,
                "corridor_km": part.DEFAULT_CORRIDOR_KM, "topology": "auto"}
    for key, default in defaults.items():
        value = params.get(key)
        if value is not None and value != default:
            flags.append(f"--{key.replace('_', '-')} {value:g}"
                         if isinstance(value, float) else
                         f"--{key.replace('_', '-')} {value}")
    partition_path = (chosen["partition"]["path"]
                      or f"partition-{chosen['leaf_dx_m']:g}m.json")
    market_flag = "spot" if market == "spot" else "on-demand"
    return [
        # The cut is the one priced: sized for its group's smallest card.
        " ".join(["woof energy partition", sites, "--leaf-dx-m",
                  f"{chosen['leaf_dx_m']:g}", "--card",
                  str(chosen["partition"]["card"] or chosen["card"]), *flags,
                  "-o", partition_path]),
        " ".join(["woof energy hindcast estimate --partition", partition_path,
                  "--segments", str(chosen["segments"]),
                  "--segment-hours", str(chosen["segment_hours"]),
                  "--spinup-hours", str(chosen["spinup_hours"]),
                  "--nz", str(chosen["nz"]), "--parent-nz", str(chosen["nz"]),
                  "--instance", chosen["offer"], "--market", market_flag,
                  "--deadline-days", f"{deadline_days:g}"]),
        " ".join(["woof energy hindcast plan --partition", partition_path,
                  "--start YYYY-MM-DD --end YYYY-MM-DD",
                  "--segment-hours", str(chosen["segment_hours"]),
                  "--spinup-hours", str(chosen["spinup_hours"]),
                  "--nz", str(chosen["nz"]), "--card", chosen["card"],
                  "-o", "hindcast"]),
    ]


def _money(value: float | None) -> str:
    return cost._money(value)


def render_table(document: Mapping[str, Any], *, limit: int) -> str:
    market = document["ranking"]["market"]
    deadline = document["constraints"]["deadline_days"]
    lines = [
        f"woof energy hindcast optimize: {document['search']['configurations']} "
        f"configuration x offer rows priced, {len(document['top'])} shown, "
        f"ranked by {market} cost meeting a {deadline:g}-day deadline "
        f"(prices as of {document['prices']['as_of']}, "
        f"{document['prices']['age_days']} days old; ESTIMATES, band = "
        "optimistic..pessimistic throughput and utilisation)",
        f"{'#':>2} {'dx':>4} {'nz':>3} {'run/spin':>8} {'offer':<22} "
        f"{'GPU-h':>9} {'GPU-h band':>19} {market + ' $':>24} "
        f"{'other $':>9} {'inst':>5} {'wall d':>6} {'util':>5}",
    ]
    other = document["ranking"]["other_market"]
    for row in document["top"][:limit]:
        g = row["gpu_hours"]
        c = row["cost_usd"][market]
        o = row["cost_usd"].get(other, {}).get("central")
        lines.append(
            f"{row['rank']:>2} {row['leaf_dx_m']:>4g} {row['nz']:>3} "
            f"{row['segment_hours']:>4}/{row['spinup_hours']:<3} "
            f"{row['offer']:<22} {g['central']:>9,.0f} "
            f"{g['low']:>9,.0f}-{g['high']:<9,.0f} "
            f"{_money(c['central']):>8} ({_money(c['low'])}-{_money(c['high'])})"
            f" {_money(o):>9} {row['deadline']['instances']:>5} "
            f"{row['deadline']['wall_clock_days']:>6.1f} "
            f"{(row.get('utilisation') or {}).get('central', 0):>5.2f}")
    chosen = document["chosen"]
    lines.append(f"chosen: {chosen['offer']} at {chosen['leaf_dx_m']:g} m, "
                 f"{chosen['nz']} levels, {chosen['segment_hours']} h runs "
                 f"with {chosen['spinup_hours']} h spin-up: "
                 f"{_money(chosen['cost_usd'][market]['central'])} {market} "
                 f"({_money(chosen['cost_usd'][market]['low'])}-"
                 f"{_money(chosen['cost_usd'][market]['high'])}), "
                 f"{chosen['deadline']['instances']} instances for "
                 f"{chosen['deadline']['wall_clock_days']:.1f} days")
    for item in document.get("spinup_sensitivity") or []:
        lines.append(f"  spin-up {item['spinup_hours']} h "
                     f"({item['segment_hours']} h runs): "
                     f"{_money(item['cost_usd'].get(market))} {market} "
                     f"(x{item['relative_to_chosen']})")
    for note in document.get("notes") or []:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def _int_list(text: str, what: str) -> tuple[int, ...]:
    try:
        return tuple(int(v) for v in str(text).split(",") if v.strip())
    except ValueError as error:
        raise OptimizeRefused(f"{what} {text!r} is not a comma list of whole "
                              "numbers") from error


def _float_list(text: str, what: str) -> tuple[float, ...]:
    try:
        return tuple(float(v) for v in str(text).split(",") if v.strip())
    except ValueError as error:
        raise OptimizeRefused(f"{what} {text!r} is not a comma list of "
                              "numbers") from error


def run(args, *, log=None, today: date | None = None) -> dict[str, Any]:
    """Build the ``woof-energy.cost-optimize.v1`` document from parsed
    arguments."""

    from woof.energy import partition as part

    log = log or (lambda message: print(f"woof energy hindcast optimize: "
                                        f"{message}", file=sys.stderr))
    if (args.sites is None) == (args.partition is None):
        raise OptimizeRefused("give exactly one of --sites SITES.json (cut "
                              "per card and spacing) or --partition "
                              "PARTITION.json")
    if args.no_repartition and args.partition is None:
        raise OptimizeRefused("--no-repartition prices a --partition as cut; "
                              "it needs --partition")
    leaf_dx = check_leaf_dx(_float_list(args.leaf_dx_m, "--leaf-dx-m")
                            if args.leaf_dx_m else DEFAULT_LEAF_DX_CANDIDATES)
    nz = check_nz(_int_list(args.nz, "--nz") if args.nz
                  else DEFAULT_NZ_CANDIDATES)
    layouts = check_layouts(
        _int_list(args.keep_hours, "--keep-hours") if args.keep_hours
        else DEFAULT_KEEP_CANDIDATES,
        _int_list(args.spinup_hours, "--spinup-hours") if args.spinup_hours
        else DEFAULT_SPINUP_CANDIDATES)
    if not (leaf_dx and nz and layouts):
        raise OptimizeRefused("--leaf-dx-m, --nz and the layouts must each "
                              "name at least one value")
    if args.max_instances is not None and args.max_instances < 1:
        raise OptimizeRefused("--max-instances must be at least 1")
    period_hours = float(args.period_days) * 24.0
    table, notes = cost.load_prices(args.prices, today=today)
    offers = cost.select_offers(table, args.instance)
    market = "spot" if args.rank_market == "spot" else "on_demand"
    inputs: dict[str, Any] = {}
    params: dict[str, Any] = {}
    directory = Path(args.partition_dir) if args.partition_dir else None
    if args.partition is not None:
        given = part.load_partition(Path(args.partition)) \
            if Path(args.partition).is_file() else None
        if given is None:
            raise OptimizeRefused(f"--partition {args.partition} does not exist")
        params = dict(given.parameters)
        inputs["partition"] = {"path": str(args.partition),
                               "sha256": _sha256(Path(args.partition)),
                               "leaf_dx_m": params.get("leaf_dx_m"),
                               "capacity": given.capacity,
                               "clusters": len(given.clusters)}
    if args.partition is not None and args.no_repartition:
        own = float(params["leaf_dx_m"])
        if args.leaf_dx_m and any(abs(dx - own) > 1e-6 for dx in leaf_dx):
            raise OptimizeRefused(
                f"--leaf-dx-m {args.leaf_dx_m} with --no-repartition: the "
                f"partition was cut at {own:g} m and is priced as cut; drop "
                "--no-repartition to re-partition per spacing")
        leaf_dx = (own,)
        sized = float(given.capacity.get("vram_gib") or 0.0)
        record = {"leaf_dx_m": own, "card": given.capacity.get("card"),
                  "vram_gib": sized, "clusters": len(given.clusters),
                  "path": str(args.partition), "origin": "given",
                  "seconds": 0.0, "specs": _specs(given, own)}

        def partitions_for(gib_key, dx, card_key):
            gib = part.resolve_capacity(card=card_key)[0]
            if gib < sized - 1e-9:
                raise OptimizeRefused(
                    f"the partition was sized for {sized:.2f} GiB; {card_key} "
                    f"({gib:.2f} GiB) may not hold its parent chains (drop "
                    "--no-repartition to re-cut it for this card)")
            return record

        built = [{k: v for k, v in record.items() if k != "specs"}]
        sites_label = str(given.sites_path())
    else:
        if args.partition is not None:
            sites_path = given.sites_path()
            if not sites_path.is_file():
                raise OptimizeRefused(
                    f"the partition's sites {sites_path} are missing; pass "
                    "--sites, or --no-repartition to price it as cut")
            if _sha256(sites_path) != given.sites_ref.get("sha256"):
                raise OptimizeRefused(
                    f"the partition's sites {sites_path} no longer have the "
                    "sha256 it was cut from; pass --sites to re-cut them")
        else:
            sites_path = Path(args.sites)
            if not sites_path.is_file():
                raise OptimizeRefused(f"--sites {sites_path} does not exist")
        for key, flag in (("linkage_km", args.linkage_km),
                          ("max_extent_km", args.max_extent_km),
                          ("max_sites", args.max_sites),
                          ("corridor_km", args.corridor_km)):
            if flag is not None:
                params[key] = flag
        cache = PartitionCache(sites_path, params, directory=directory, log=log)
        inputs["sites"] = {"path": str(sites_path), "sha256": cache.sites_sha,
                           "sites": len(cache.sites)}

        def partitions_for(gib_key, dx, card_key):
            if gib_key is None:
                from woof.energy.partition import resolve_capacity

                gib_key = _gib_key(resolve_capacity(card=card_key)[0])
            return cache.get(gib_key=gib_key, dx=dx, card_key=card_key)

        built = cache.built
        sites_label = str(sites_path)
    result = optimize(
        partitions_for=partitions_for, offers=offers, price_table=table,
        leaf_dx=leaf_dx, nz=nz, layouts=layouts, period_hours=period_hours,
        deadline_days=float(args.deadline_days), rank_market=market,
        max_instances=args.max_instances, top=args.top,
        store_preset=args.store_preset,
        packing_efficiency=args.packing_efficiency, log=log)
    rows = result["rows"]
    chosen = dict(rows[0])
    chosen_offer = next(o for o in offers if _offer_name(o) == chosen["offer"])
    sensitivity = spinup_sensitivity(
        chosen, partitions_for=partitions_for, offer=chosen_offer, price_table=table, period_hours=period_hours,
        market=market, store_preset=args.store_preset,
        packing_efficiency=args.packing_efficiency)
    chosen["recipe"] = recipe(chosen, sites=sites_label, params=params,
                              deadline_days=float(args.deadline_days),
                              market=market)
    as_of = date.fromisoformat(str(table["as_of"]))
    today = today or datetime.now(timezone.utc).date()
    age = (today - as_of).days
    stale_after = int(table.get("stale_after_days") or 180)
    model = result["model"] or {}
    notes = list(notes)
    if result["infeasible"]:
        notes.append(f"{result['infeasible']} priced row(s) miss the deadline "
                     "and are not ranked")
    notes.append("every figure is an ESTIMATE from woof energy estimate's "
                 "throughput model (two consumer-card receipts scaled by "
                 "bandwidth, an assumed utilisation curve); spot prices are "
                 "illustrative; time one tile on the rented card before "
                 "committing")
    document = {
        "schema": OPTIMIZE_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "estimate",
        "inputs": inputs,
        "search": {
            "leaf_dx_m": list(leaf_dx), "nz": list(nz),
            "layouts": [{"segment_hours": l.segment_hours,
                         "spinup_hours": l.spinup_hours,
                         "keep_hours": l.keep_hours} for l in layouts],
            "period_days": float(args.period_days),
            "offers": [_offer_name(o) for o in offers],
            "partitions": built,
            "repartitioned": not (args.partition is not None
                                  and args.no_repartition),
            "configurations": result["configurations"],
        },
        "constraints": {"deadline_days": float(args.deadline_days),
                        "max_instances": args.max_instances,
                        "store_preset": args.store_preset,
                        "packing_efficiency": args.packing_efficiency},
        "ranking": {"market": market, "other_market": result["other_market"],
                    "by": [f"{market} central cost", f"{market} pessimistic "
                           "cost", "wall clock"],
                    "storage": "reported per row, not ranked"},
        "model": {
            "throughput_band": model.get("throughput_band"),
            "utilisation": model.get("utilisation"),
            "k_cell_steps_per_s_per_gbs": (model.get("calibration") or {}).get(
                "k_cell_steps_per_s_per_gbs"),
            "status": "estimate: no datacentre card has been timed by this tree",
        },
        "prices": {"as_of": table.get("as_of"), "age_days": age,
                   "stale_after_days": stale_after, "stale": age > stale_after,
                   "currency": table.get("currency", "USD"),
                   "spot": "ILLUSTRATIVE: spot prices move hourly and 8-GPU "
                           "spot capacity is often unavailable",
                   "path": str(Path(args.prices) if args.prices
                               else cost.DEFAULT_PRICES)},
        "top": rows[:args.top],
        "top_by_other_market": result["top_other"],
        "chosen": chosen,
        "spinup_sensitivity": sensitivity,
        "excluded": result["excluded"],
        "notes": notes,
    }
    return document


def main(args) -> int:
    try:
        document = run(args)
    except (ValueError, OSError) as error:
        # CostRefusal/OptimizeRefused, and the partition/sites loaders'
        # ContractError (a ValueError) or an unreadable file.
        print(f"woof energy hindcast optimize: REFUSED: {error}",
              file=sys.stderr)
        print(json.dumps({"schema": OPTIMIZE_SCHEMA, "refused": str(error)},
                         indent=1))
        return 2
    print(render_table(document, limit=args.top), file=sys.stderr)
    if args.output:
        _write_json(document, args.output)
    json.dump(document, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


__all__ = ["OPTIMIZE_SCHEMA", "OptimizeRefused", "Layout", "PartitionCache",
           "optimize", "run", "main", "recipe", "check_nz", "check_leaf_dx",
           "check_layouts", "spinup_sensitivity"]
