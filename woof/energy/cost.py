"""``woof energy estimate``: GPU-hours, wall clock and dollars for a hindcast.

An ESTIMATE, not a measurement.  Every figure is computed from a
throughput model calibrated on two timing receipts taken on consumer
cards in this repository and scaled to other cards by memory bandwidth;
no datacentre card has been timed by this tree.  The JSON carries a
low/high band, the receipts, every assumption and the price table's
"as of" date, so a reader can see what a number rests on.

What is priced
--------------

Three inputs, one per mode, all read duck-typed so sibling documents
can be priced before their schemas settle:

``--plan PLAN.json``
    A ``woof-energy.plan.v1`` (``wrf-tiles`` or ``wrf-nests``): one
    segment's domains, with ``nx``/``ny`` from the plan, ``nz``/``dt``/
    acoustic substeps/history and restart cadence from the emitted
    configs.  Repeated for ``--segments`` segments of ``--segment-hours``.
``--partition PARTITION.json``
    A ``woof-energy.partition.v1``-like document: ``clusters`` rows with
    an area (``leaf_area_km2``/``footprint_km2``/``area_km2``, or a
    ``region`` bbox/GeoJSON whose area is taken as the leaf cover), an
    optional ``leaf_dx_m``, ``chain_m`` and ``est_tile_count``.  Each
    cluster is synthesised into parent, intermediate and leaf tiles the
    way the ``wrf-tiles`` planner shapes them (:func:`synth_cluster`).
``--clusters N --footprint-km2 A``
    Summary statistics when no partition exists yet: ``N`` equal
    clusters sharing ``A`` km2 of farm footprint.

``--job JOB.json`` (a ``woof-energy.hindcast-job.v1``-like document)
supplies segment count, segment hours and spin-up when present; flags
override it.

The model (what is pinned)
--------------------------

Per domain and segment, on a card with bandwidth ``B`` GB/s and ``S`` SMs:

* ``steps = ceil(run_seconds / dt)``; ``cells = nx * ny * nz``.
* ``throughput = K * B * cells / (cells + S * h)`` cell-steps per second,
  where ``K`` is :func:`calibration`'s fit (cell-steps/s per GB/s at
  saturation) and the second factor is an occupancy knee: a grid with few
  cells per SM cannot fill the card (``h`` cells per SM reach half of it).
* ``compute = steps * acoustic * (cells / throughput + L)`` with
  ``acoustic = (n_sound + LARGE_STEP_WEIGHT) / (4 + LARGE_STEP_WEIGHT)``:
  the receipts ran WRF's 4 acoustic substeps and a large step is assumed
  to cost as much as 4 of them (:data:`LARGE_STEP_WEIGHT`).  ``L`` is a
  fixed per-step cost (kernel launches and their tails) that no amount of
  bandwidth removes.

Utilisation (the honesty term).  Throughput is NOT linear in cells: the
knee ``h`` and the per-step cost ``L`` together make a small domain on a
big, fast card pay for idle SMs and launch latency, which is what keeps
the card ranking from favouring an H100 for a 0.3 M-cell tile; and a
large domain on a small card pays for the extra tiles and rims the memory
fit forces (:func:`synth_cluster`).  ``h`` and ``L`` are ASSUMPTIONS, not
measurements: :data:`UTILISATION` holds an optimistic, a central and a
pessimistic pair, ``K`` is refitted on the receipts for each pair (so
each reproduces the receipts), and the cost band takes the envelope over
the pairs (below).  Every offer reports its work-weighted utilisation
(saturated time / modelled time) under each pair.  Replace them with a
measured curve (one domain swept from 0.1 M to 20 M cells on the rented
card) before trusting a card ranking at either end.
* ``output`` and ``checkpoint``: frame and restart bytes over a device-
  to-host rate (:data:`D2H_GBPS`, the staged-copy rate measured in
  ``tilestream/multigpu.py``), exposed on the step loop.
* ``startup``: the measured launch-to-first-step fit
  (:data:`PREP_INTERCEPT_S`, :data:`PREP_S_PER_MCELL`), charged to the GPU
  because the card is rented while it runs.

Per offer (provider, instance): GPU-hours on that offer's card; billed
instance-hours = GPU-hours / (GPUs per instance * packing efficiency),
where the packing efficiency is the fraction of paid card time a job holds
the card at all (parents waited on, staging, the tail of a batch);
wall clock for ``--instances`` instances = the larger of billed hours /
instances and one segment's critical path (parent, intermediates, one
leaf tile: segments and tiles are independent, the chain is not); spot
cost carries ``spot_rework_fraction`` extra time.  The band: ``low`` is
the cheapest of the :data:`UTILISATION` pairs at the optimistic end of
:data:`THROUGHPUT_BAND` (1.5x), ``high`` the dearest at the pessimistic
end (0.5x), so it covers both the bandwidth scaling and the knee.

**Several jobs per card** (``--slots-per-gpu N|auto``, the worker's flag
of the same name; ``--mps`` for CUDA MPS).  ``k`` copies of a domain share
one card, ``k = min(N, floor((card memory - PACKING_MARGIN_GIB) / the
domain's priced peak))`` (``auto``: N = :data:`AUTO_MAX_SLOTS_PER_GPU`,
capped at the offer's vCPUs per GPU when the deploy machine tables know
the instance, as the worker caps it at its CPUs),
so a large domain packs once and gains nothing.  Compute splits into the
occupancy-bound part, ``cells / (K B u(c))`` per step with ``u(c) = c / (c
+ S h)`` (the utilisation pair's knee ``h``), and the per-step cost ``L``:

* with ``--mps`` kernels of separate processes run concurrently: ``k``
  copies fill the card like one grid of ``k * c`` cells, so the ideal gain
  on the occupancy-bound part is ``u(k c) / u(c)`` -- close to ``k`` for a
  grid far below the knee, close to 1 for one that already fills the card
  -- and one copy's launches overlap the others' kernels, so the ideal gain
  on ``L`` is ``k``.  Without MPS kernels time-slice and both gains are 1;
* either way, startup, output and checkpoint time of one job overlap the
  others' work: the ideal is that ``1 - 1/k`` of it is hidden;
* a fraction ``r`` of each ideal is realised, ``r`` =
  :data:`PACKING_REALISATION` (pessimistic, central, optimistic) -- an
  assumption, because no packed run has been timed by this tree; the band
  pairs the pessimistic ``r`` with the pessimistic throughput and
  utilisation pair, the optimistic with the optimistic.

Card-seconds per copy are then ``bandwidth / (1 + r (u(kc)/u(c) - 1)) +
launch / (1 + r (k - 1)) + overheads (1 - r (1 - 1/k))``.  The critical
path stretches: a packed copy shares the card, so its compute wall is
``k`` times its card share.

Storage: leaf tiles' stored hours (segment hours minus spin-up) at the
history cadence, ``--store-preset`` fields on every model level, times
``--compression-ratio`` for an Icechunk/Zarr store (zstd + shuffle on
float32 fields; an assumption), priced per GB-month, per PUT (one per
``--chunk-mib`` chunk) and per GB of one full download.

Forcing downloads (``forcing``): what fetching the boundary data costs,
per job versus once per cycle through the forcing cache
(docs/energy-hindcast/forcing-cache.md).  One cycle's download is the
bytes and objects :func:`woof.download_budget.download_estimate` measures
for ``--forcing-source`` over the segment's fetch window (or
``--forcing-gb-per-cycle``).  **Per job**, every cluster downloads it
from the publisher at ``--publisher-mbps``; **per cycle**, one prefetch
downloads it at that rate and every job reads it from a same-region cache
at ``--cache-read-mbps``, plus the cache's PUTs, GETs and storage for
``--forcing-cache-days``.  Download time is priced as GPU-slot time (the
instance price over its GPUs: the card idles while its job waits for
bytes); publisher egress is ``--publisher-egress-usd-per-gb`` (0: the
public open-data buckets charge the requester nothing) and same-region
cache reads are free.  Both rates and the retention are assumptions,
labelled as such in the document.  A source the cache does not share
(``woof.energy.forcing_cache.shareability``) shows no saving and says why.
The forcing figures are reported beside, not added to, the GPU-hours.

Prices come from ``woof/data/energy/cloud_prices.json`` (``--prices`` for
another copy) and are never fetched; a table older than its
``stale_after_days`` is noted, not refused.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tomllib
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from woof.card_catalog import CARD_CATALOG, CardRow, card_row
from woof.energy.contracts import _write_json

COST_ESTIMATE_SCHEMA = "woof-energy.cost-estimate.v1"
PRICES_SCHEMA = "woof-energy.cloud-prices.v1"
DEFAULT_PRICES = (Path(__file__).resolve().parent.parent / "data" / "energy"
                  / "cloud_prices.json")

GIB = 1024 ** 3
KIB = 1024


class CostRefusal(ValueError):
    """The estimate cannot be made, and the message names what is missing."""


# --------------------------------------------------------------------------
# throughput calibration
# --------------------------------------------------------------------------
#: Cells per SM at which a grid reaches half the card's saturated
#: throughput.  An assumption (one 4,096-cell slab per SM), chosen so the
#: receipts below sit at 83-97% occupancy; it only matters for small grids.
HALF_SATURATION_CELLS_PER_SM = 4096

#: Fixed GPU seconds per large step (at 4 acoustic substeps; scaled like
#: the step's work by the substep count): kernel launches and their tails,
#: which bandwidth does not shorten.  An assumption: about 200 launches a
#: step at about 5 us each.  CUDA graphs would cut it.
STEP_OVERHEAD_S = 0.001


@dataclass(frozen=True)
class Utilisation:
    """One assumed occupancy knee and per-step cost (see the module
    docstring); none of the three is a measurement."""

    label: str
    half_saturation_cells_per_sm: float
    step_overhead_s: float

    def as_dict(self) -> dict[str, Any]:
        return {"half_saturation_cells_per_sm": self.half_saturation_cells_per_sm,
                "step_overhead_s": self.step_overhead_s}


#: The utilisation pairs the band spans: a knee at 2,048 / 4,096 / 16,384
#: cells per SM and 0.5 / 1 / 3 ms per step.  The pessimistic knee puts an
#: H100 (132 SMs) at half throughput below 2.2 M cells, the optimistic one
#: at 0.27 M.  Assumptions until a card is swept (module docstring).
UTILISATION: Mapping[str, Utilisation] = {
    "optimistic": Utilisation("optimistic", 2048.0, 0.0005),
    "central": Utilisation("central", float(HALF_SATURATION_CELLS_PER_SM),
                           STEP_OVERHEAD_S),
    "pessimistic": Utilisation("pessimistic", 16384.0, 0.003),
}
CENTRAL = UTILISATION["central"]

#: WRF's default acoustic substeps; every receipt ran with it.
REFERENCE_SOUND_STEPS = 4

#: Cost of one large (physics + advection) step in acoustic-substep units.
#: An assumption: half of a 4-substep step is the acoustic loop.
LARGE_STEP_WEIGHT = 4.0

#: Multiplicative band on throughput: (pessimistic, optimistic).  The two
#: receipts disagree by 26% per GB/s and the bandwidth scaling to a
#: datacentre card is itself unmeasured; 0.5x-1.5x is a judgement.
THROUGHPUT_BAND = (0.5, 1.5)

#: Timing receipts the throughput constant is fitted on.  ``domains`` are
#: ``(nx, ny, nz, steps)`` for the wall interval ``wall_s``.
CALIBRATION_RECEIPTS: tuple[Mapping[str, Any], ...] = (
    {
        "id": "first-light-rtx4070-single-domain",
        "card": "rtx-4070",
        "domains": ((342, 272, 49, 2160),),
        "wall_s": 523.0,
        "source": "docs/public/FIRST-LIGHT.md section 3 table: 'GPU forecast, "
                  "6 simulated hours, 2160 steps at dt 10 s | 523 s', single "
                  "domain 342x272x49, Linux RTX 4070 12 GB",
        "assumptions": "wall includes hourly output frames; the card's SM "
                       "count and bandwidth are nameplate",
    },
    {
        "id": "live-products-rtx5070ti-three-grid-stack",
        "card": "rtx-5070-ti",
        "domains": ((170, 170, 49, 60), (336, 336, 49, 240),
                    (408, 402, 49, 720)),
        "wall_s": 170.0,
        "source": "woof/live_products.py module docstring: 'that three-grid "
                  "stack takes about 170 s of wall per model hour' on a 5070 "
                  "Ti (12 km 170x170, 3 km 336x336, 1 km 408x402)",
        "assumptions": "nz = 49 (the wizard default) and the 5 s/km clock "
                       "(60/15/5 s, so 60/240/720 steps per model hour) are "
                       "assumed, not read from the receipt",
    },
)

#: Receipts read and deliberately not fitted, with the reason.
EXCLUDED_RECEIPTS: tuple[Mapping[str, str], ...] = (
    {
        "id": "first-light-rtx5090-acceptance-run",
        "source": "docs/public/FIRST-LIGHT.md section 5: 250x200x49, 361 "
                  "steps in ~200 s (~0.55 s/step including output), RTX "
                  "5090 on Windows 11",
        "reason": "a first-time acceptance transcript on WDDM whose wall "
                  "includes launch and first-frame compilation; it implies "
                  "~16x less throughput per GB/s than the two fitted "
                  "receipts and describes a cold first run, not a "
                  "hindcast's steady state",
    },
)


def _occupancy(cells: float, sm_count: int,
               half_saturation: float = HALF_SATURATION_CELLS_PER_SM) -> float:
    return cells / (cells + sm_count * half_saturation)


def calibration(utilisation: Utilisation = CENTRAL) -> dict[str, Any]:
    """Fit ``K`` (cell-steps/s per GB/s at saturation) on the receipts.

    Each receipt gives ``K_i = sum(steps * cells / occupancy) / ((wall -
    steps * L) * B)`` under ``utilisation``'s knee and per-step cost ``L``;
    ``K`` is their geometric mean.  Pure: recomputed from the table above
    every call, so an edited receipt moves every estimate.
    """

    fits = []
    for receipt in CALIBRATION_RECEIPTS:
        card = CARD_CATALOG[receipt["card"]]
        work = sum(nx * ny * nz * steps / _occupancy(
            nx * ny * nz, card.sm_count, utilisation.half_saturation_cells_per_sm)
                   for nx, ny, nz, steps in receipt["domains"])
        steps_total = sum(steps for *_, steps in receipt["domains"])
        busy = float(receipt["wall_s"]) - steps_total * utilisation.step_overhead_s
        if busy <= 0:
            raise CostRefusal(f"receipt {receipt['id']}: the per-step cost "
                              f"{utilisation.step_overhead_s:g} s exceeds its "
                              "wall time")
        k = work / (busy * card.bandwidth_gbs)
        raw = sum(nx * ny * nz * steps for nx, ny, nz, steps in receipt["domains"])
        fits.append({"id": receipt["id"], "card": card.key,
                     "cell_steps_per_s": round(raw / float(receipt["wall_s"])),
                     "k_cell_steps_per_s_per_gbs": round(k, 1),
                     "source": receipt["source"],
                     "assumptions": receipt["assumptions"]})
    k = math.exp(sum(math.log(f["k_cell_steps_per_s_per_gbs"]) for f in fits)
                 / len(fits))
    return {"k_cell_steps_per_s_per_gbs": round(k, 1), "receipts": fits,
            "excluded": [dict(row) for row in EXCLUDED_RECEIPTS],
            "utilisation": {"label": utilisation.label,
                            **utilisation.as_dict()},
            "fit": "geometric mean of per-receipt K"}


# --------------------------------------------------------------------------
# overheads
# --------------------------------------------------------------------------
#: Device-to-host rate a frame or restart is exposed at (GB/s): the staged
#: copy rate measured in tilestream/multigpu.py (~10 GB/s per direction).
#: A datacentre card on PCIe Gen5 is faster; this is the conservative end.
D2H_GBPS = 10.0

#: 3-D float32 fields a restart carries (prognostic state, tendencies held
#: across steps, physics state).  An assumption, not counted per suite.
RESTART_3D_FIELDS = 60

#: Launch-to-first-step: a straight-line fit through the receipt in
#: docs/manual/08-operational-envelope.md section 8.3 (15 s at 0.14 M
#: cells, 37 s at 4.39 M, 79 s at 12.6 M, 100 s at 16.8 M).
PREP_INTERCEPT_S = 14.0
PREP_S_PER_MCELL = 5.1

#: Fields a stored frame carries, as (3-D fields, 2-D fields).  ``energy``
#: counts woof.io.history_selection's energy preset (14 volumes, 18
#: surface/georeference fields); ``full`` is the 82-variable full tape of
#: the same module's measured frame, split 45/37 by assumption.
STORE_PRESETS: Mapping[str, tuple[int, int]] = {
    "energy": (14, 18),
    "full": (45, 37),
}

#: Default lossless compression ratio (stored/raw) of an Icechunk/Zarr
#: store of float32 model fields with zstd + byte shuffle.  An assumption:
#: 2.5-4x is typical for such fields; quantisation would do better.
DEFAULT_COMPRESSION_RATIO = 0.35

#: Fraction of paid GPU time that runs a domain (the rest waits on a
#: parent, stages data or idles at the tail of a batch).  An assumption.
DEFAULT_PACKING_EFFICIENCY = 0.85

#: Most copies ``--slots-per-gpu auto`` packs on one card, and the card
#: memory kept free beside packed jobs (GiB): the worker's own numbers
#: (:mod:`woof.energy.packing`).
from woof.energy.packing import AUTO_MAX_SLOTS_PER_GPU  # noqa: E402
from woof.energy.packing import DEFAULT_MEM_MARGIN_GIB as PACKING_MARGIN_GIB  # noqa: E402

#: Fraction of the ideal packing gain realised: (pessimistic, central,
#: optimistic).  An assumption, not a measurement: MPS contention, shared
#: L2 and memory bandwidth, and CPU launch threads all take some of it.
PACKING_REALISATION = (0.5, 0.75, 1.0)

# --------------------------------------------------------------------------
# synthesis constants (summary and partition modes)
# --------------------------------------------------------------------------
#: Device memory per cell and fixed intercept (upper ends of the fit in
#: docs/manual/08-operational-envelope.md section 8.1: 0.74-0.84 KiB/cell
#: plus 2.9-3.4 GiB), used to size synthesised tiles to a card.
VRAM_KIB_PER_CELL = 0.84
VRAM_INTERCEPT_GIB = 3.4

#: Leaf tiles' lateral rim in leaf cells (spec 1 + relax 10, the plan's
#: lateral_zone), added on every side of every synthesised tile.
TILE_RIM_CELLS = 11

#: Buffer around the leaf cover a parent domain carries on each side
#: (woof domain --polygon's buffer in the Wales fixture plan's log).
PARENT_BUFFER_KM = 16.0

#: Leaf-cover area per km2 of farm footprint: rectangle cover of irregular
#: footprints plus the planner's 2 km corridor.  An assumption.
DEFAULT_COVER_FACTOR = 2.0

#: Wizard clock: 5 s per km of grid spacing (60 s at 12 km).
SECONDS_PER_KM = 5.0

DEFAULT_PARENT_NZ = 49
DEFAULT_HISTORY_S = 900.0
DEFAULT_RESTART_S = 3600.0
#: Stored hours a campaign covers when no segment count is given: one
#: year.  The default segment count is ``ceil(8760 / stored hours)``.
DEFAULT_PERIOD_HOURS = 8760.0
#: The hindcast driver's defaults (woof.energy.hindcast, held equal by
#: test): one 9 h forecast per 00/06/12/18 UTC init, 3 h spin-up, 6 h
#: kept, so a year is 1,460 segments; 300 m leaves on the 49-level
#: wind-energy column (woof.energy.vertical.HINDCAST_NZ).
DEFAULT_SEGMENT_HOURS = 9.0
DEFAULT_SPINUP_HOURS = 3.0
DEFAULT_LEAF_DX_M = 300.0
DEFAULT_LEAF_NZ = 49

#: The source a forcing estimate prices when none is named: the global
#: model the selector takes outside the HRRR/RAP domain.
DEFAULT_FORCING_SOURCE = "met-office-global"
#: The cycle a forcing estimate sizes one download on (sizes are per
#: object and do not move with the date inside one publication era).
FORCING_REFERENCE_CYCLE = "2026-06-01T00"
#: Sustained MB/s one job's fetch draws from a publisher (a public cloud
#: open-data bucket in the same cloud).  An assumption: a NOMADS-class
#: server paces far lower, and 440 jobs of one cycle share its limits.
DEFAULT_PUBLISHER_MBPS = 100.0
#: Sustained MB/s one job reads the forcing cache in its own region.  An
#: assumption (parallel GETs from one instance).
DEFAULT_CACHE_READ_MBPS = 400.0
#: Days a cycle's cache entry is kept before a lifecycle rule deletes it.
DEFAULT_FORCING_CACHE_DAYS = 7.0
#: Days per storage-month.
_DAYS_PER_MONTH = 30.4


@dataclass(frozen=True)
class DomainWork:
    """One domain of one segment, as the cost model reads it."""

    domain_id: str
    role: str                   # parent | intermediate | leaf
    nx: int
    ny: int
    nz: int
    dx_m: float
    dt_s: float
    run_seconds: float
    sound_steps: int = REFERENCE_SOUND_STEPS
    history_s: float = DEFAULT_HISTORY_S
    restart_s: float | None = DEFAULT_RESTART_S
    level: int = 0              # chain depth, 0 for the forced root
    count: int = 1              # identical copies (synthesised tiles)
    peak_gib: float | None = None

    @property
    def cells(self) -> int:
        return self.nx * self.ny * self.nz

    def as_dict(self) -> dict[str, Any]:
        return {"domain_id": self.domain_id, "role": self.role,
                "nx": self.nx, "ny": self.ny, "nz": self.nz,
                "dx_m": self.dx_m, "dt_s": self.dt_s,
                "run_seconds": self.run_seconds,
                "sound_steps": self.sound_steps, "history_s": self.history_s,
                "restart_s": self.restart_s, "level": self.level,
                "count": self.count, "peak_gib": self.peak_gib}


def _stored_fields(preset: str, nz: int) -> int:
    three_d, two_d = STORE_PRESETS[preset]
    return three_d * nz + two_d


def domain_peak_gib(domain: DomainWork) -> float:
    """The domain's priced peak, or the memory fit's figure for its cells."""

    if domain.peak_gib is not None:
        return float(domain.peak_gib)
    return VRAM_INTERCEPT_GIB + domain.cells * VRAM_KIB_PER_CELL * KIB / GIB


def offer_vcpus_per_gpu(offer: Mapping[str, Any]) -> int | None:
    """vCPUs per GPU of an offer, from the deploy renderers' machine tables
    (None when the instance is not in them)."""

    from woof.energy.deploy import AWS_GPU_INSTANCES, GCP_GPU_MACHINES

    table = {"aws": AWS_GPU_INSTANCES, "gcp": GCP_GPU_MACHINES}.get(
        str(offer.get("provider")), {})
    shape = table.get(str(offer.get("instance")))
    if shape is None or shape.gpus < 1:
        return None
    return max(1, shape.vcpus // shape.gpus)


def packed_copies(domain: DomainWork, card: CardRow,
                  slots_per_gpu: int | str,
                  vcpus_per_gpu: int | None = None) -> int:
    """Copies of ``domain`` one card holds at once (module docstring).

    ``auto`` is also bounded by ``vcpus_per_gpu`` (one CPU core drives each
    slot's kernels), as the worker's ``auto`` is."""

    if slots_per_gpu == "auto":
        cap = AUTO_MAX_SLOTS_PER_GPU
        if vcpus_per_gpu is not None:
            cap = min(cap, vcpus_per_gpu)
    else:
        cap = int(slots_per_gpu)
    if cap <= 1:
        return 1
    by_memory = int((card.memory_gib - PACKING_MARGIN_GIB)
                    // domain_peak_gib(domain))
    return max(1, min(cap, by_memory))


def packing_factors(domain: DomainWork, card: CardRow, copies: int, *,
                    mps: bool, realisation: float,
                    utilisation: Utilisation = CENTRAL) -> dict[str, float]:
    """Multipliers on one copy's card-seconds when ``copies`` share a card.

    ``bandwidth`` scales the occupancy-bound part of compute, ``launch``
    the per-step cost, ``overhead`` startup, output and checkpoint (module
    docstring)."""

    if copies <= 1:
        return {"copies": 1, "ideal_compute_gain": 1.0, "compute_gain": 1.0,
                "ideal_launch_gain": 1.0, "launch_gain": 1.0,
                "bandwidth": 1.0, "launch": 1.0, "overhead": 1.0}
    cells = domain.cells
    knee = utilisation.half_saturation_cells_per_sm
    ideal = (_occupancy(copies * cells, card.sm_count, knee)
             / _occupancy(cells, card.sm_count, knee)) if mps else 1.0
    ideal_launch = float(copies) if mps else 1.0
    gain = 1.0 + realisation * (ideal - 1.0)
    launch_gain = 1.0 + realisation * (ideal_launch - 1.0)
    return {"copies": copies, "ideal_compute_gain": ideal,
            "compute_gain": gain, "ideal_launch_gain": ideal_launch,
            "launch_gain": launch_gain, "bandwidth": 1.0 / gain,
            "launch": 1.0 / launch_gain,
            "overhead": 1.0 - realisation * (1.0 - 1.0 / copies)}


def domain_seconds(domain: DomainWork, card: CardRow, k: float, *,
                   factor: float = 1.0,
                   leaf_preset: str = "full",
                   utilisation: Utilisation = CENTRAL,
                   packing: Mapping[str, float] | None = None
                   ) -> dict[str, float]:
    """GPU seconds for one copy of ``domain`` on ``card``, by component.

    ``factor`` scales throughput (the band); overheads are not scaled.
    ``k`` must be :func:`calibration`'s fit under the same ``utilisation``.
    A leaf writes ``leaf_preset`` frames (the fields the store keeps); a
    parent or intermediate writes the full tape its children are built from.
    ``saturated_s`` is the compute a fully occupied card with no per-step
    cost would take; ``saturated_s / compute_s`` is the utilisation.
    ``packing`` (:func:`packing_factors`, priced under the same
    ``utilisation``) turns them into the card-seconds one of several
    co-resident copies costs; ``wall_s`` is then that copy's own wall time
    on the shared card.
    """

    steps = math.ceil(domain.run_seconds / domain.dt_s)
    cells = domain.cells
    saturated = k * factor * card.bandwidth_gbs
    throughput = saturated * _occupancy(
        cells, card.sm_count, utilisation.half_saturation_cells_per_sm)
    acoustic = ((domain.sound_steps + LARGE_STEP_WEIGHT)
                / (REFERENCE_SOUND_STEPS + LARGE_STEP_WEIGHT))
    bandwidth = steps * acoustic * cells / throughput
    launch = steps * acoustic * utilisation.step_overhead_s
    compute = bandwidth + launch
    preset = leaf_preset if domain.role == "leaf" else "full"
    frames = int(domain.run_seconds // domain.history_s) + 1
    frame_bytes = _stored_fields(preset, domain.nz) * domain.nx * domain.ny * 4
    output = frames * frame_bytes / (D2H_GBPS * 1e9)
    restarts = (int(domain.run_seconds // domain.restart_s)
                if domain.restart_s else 0)
    checkpoint = restarts * RESTART_3D_FIELDS * cells * 4 / (D2H_GBPS * 1e9)
    startup = PREP_INTERCEPT_S + PREP_S_PER_MCELL * cells / 1e6
    overheads = output + checkpoint + startup
    wall = compute + overheads
    if packing is not None and packing["copies"] > 1:
        compute = (bandwidth * packing["bandwidth"]
                   + launch * packing["launch"])
        output *= packing["overhead"]
        checkpoint *= packing["overhead"]
        startup *= packing["overhead"]
        wall = compute * packing["copies"] + overheads
    return {"steps": steps, "compute_s": compute, "output_s": output,
            "checkpoint_s": checkpoint, "startup_s": startup,
            "saturated_s": steps * acoustic * cells / saturated,
            "total_s": compute + output + checkpoint + startup,
            "wall_s": wall}


# --------------------------------------------------------------------------
# plan mode
# --------------------------------------------------------------------------
def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text())
    except FileNotFoundError as error:
        raise CostRefusal(f"{what} {path} does not exist") from error
    except json.JSONDecodeError as error:
        raise CostRefusal(f"{what} {path} is not JSON: {error}") from error
    if not isinstance(document, dict):
        raise CostRefusal(f"{what} {path} is not a JSON object")
    return document


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _default_dt(dx_m: float) -> float:
    return SECONDS_PER_KM * dx_m / 1000.0


def _plan_role(domain: Mapping[str, Any], leaf_dx: float, topology: str) -> str:
    """leaf, intermediate or parent for one plan domain.

    ``wrf-tiles``: ``child`` rows are leaf tiles, ``parent`` rows with a
    parent are intermediate tiles, the forced root is the parent.
    ``wrf-nests``: ``child`` rows are leaves, nested non-leaves (level > 0)
    are intermediate, d01 is the parent.
    """

    if domain.get("role") == "child" or float(domain["dx_m"]) <= leaf_dx:
        return "leaf"
    if topology == "wrf-tiles" and domain.get("parent") is not None:
        return "intermediate"
    if topology == "wrf-nests" and int((domain.get("extra") or {}).get("level") or 0) > 0:
        return "intermediate"
    return "parent"


def domains_from_plan(path: str | Path, *,
                      run_seconds: float | None = None) -> list[DomainWork]:
    """One segment's :class:`DomainWork` rows from a ``woof-energy.plan.v1``.

    ``run_seconds`` overrides every domain's run length (a plan written for
    one window priced for segments of another).  ``hex-swath`` plans are
    refused: their cost lives in the plan's own capacity estimate.
    """

    path = Path(path)
    plan = _read_json(path, "plan")
    if plan.get("schema") != "woof-energy.plan.v1":
        raise CostRefusal(f"{path} is {plan.get('schema')!r}, not a "
                          "woof-energy.plan.v1")
    topology = plan.get("topology")
    if topology not in ("wrf-tiles", "wrf-nests"):
        raise CostRefusal(f"{path}: topology {topology!r} is not priced; "
                          "estimate prices wrf-tiles and wrf-nests plans")
    base = path.parent
    hours = float(plan.get("hours") or DEFAULT_SEGMENT_HOURS)
    default_run = run_seconds if run_seconds is not None else hours * 3600.0
    leaf_dx = float(plan.get("dx_m") or 0.0)
    rows: list[DomainWork] = []
    for raw in plan.get("domains") or []:
        if not isinstance(raw, Mapping):
            raise CostRefusal(f"{path}: a domains entry is not an object")
        dx_raw = raw.get("dx_m")
        if isinstance(dx_raw, bool) or not isinstance(dx_raw, (int, float)) \
                or dx_raw <= 0:
            raise CostRefusal(f"{path}: domain {raw.get('domain_id')!r} "
                              "carries no positive dx_m")
        extra = raw.get("extra") or {}
        nx, ny = extra.get("nx"), extra.get("ny")
        nz = dt = None
        sound = REFERENCE_SOUND_STEPS
        history = float(extra.get("history_interval_s") or DEFAULT_HISTORY_S)
        restart = extra.get("restart_interval_s", DEFAULT_RESTART_S)
        level = int(extra.get("level") or 0)
        peak = extra.get("peak_envelope_gib")
        config = raw.get("config")
        if config and str(config).endswith(".json"):          # wrf-tiles tile
            spec = _read_json(base / config, "tile spec")
            nx, ny = spec.get("nx", nx), spec.get("ny", ny)
            nz, dt = spec.get("nz"), spec.get("dt_s")
            level = int(spec.get("level") or level)
            peak = (spec.get("vram") or {}).get("peak_envelope_gib", peak)
            toml_path = (spec.get("child_config") or {}).get("path")
            table = _read_toml(base / toml_path) if toml_path else {}
            run = table.get("run") or {}
            sound = int(run.get("time_step_sound") or sound)
            history = float(run.get("output_interval_s") or history)
            restart = run.get("restart_interval_s", restart)
        elif config:                                          # woof domain TOML
            table = _read_toml(base / config)
            shared = table.get("shared") or {}
            nz = shared.get("nz")
            sound = int(shared.get("time_step_sound") or sound)
            grid_id = int(raw.get("grid_id") or 1)
            for entry in table.get("domain") or []:
                if int(entry.get("grid_id", 0)) == grid_id:
                    nx = entry.get("nx", nx)
                    ny = entry.get("ny", ny)
                    dt = entry.get("time_step")
                    history = float(entry.get("history_interval_s") or history)
            restart = (table.get("experiment") or {}).get(
                "restart_interval_s", restart)
        if nx is None or ny is None:
            raise CostRefusal(f"{path}: domain {raw.get('domain_id')!r} has no "
                              "nx/ny in the plan, its tile spec or its config")
        dx = float(raw.get("dx_m"))
        nz = int(nz or (60 if dx <= 250.0 else DEFAULT_PARENT_NZ))
        dt = float(dt or _default_dt(dx))
        if topology == "wrf-nests":
            # Every nest steps inside one process on one card: each is its
            # own link of the critical path, never run beside another.
            level = len(rows)
        rows.append(DomainWork(
            domain_id=str(raw.get("domain_id")), role=_plan_role(raw, leaf_dx, str(topology)),
            nx=int(nx), ny=int(ny), nz=nz, dx_m=dx, dt_s=dt,
            run_seconds=float(default_run if run_seconds is not None
                              else extra.get("run_seconds") or default_run),
            sound_steps=sound, history_s=history,
            restart_s=float(restart) if restart else None, level=level,
            peak_gib=float(peak) if peak is not None else None))
    if not rows:
        raise CostRefusal(f"{path} lists no domains")
    return rows


# --------------------------------------------------------------------------
# summary / partition mode: synthesise a cluster's domains
# --------------------------------------------------------------------------
def max_tile_cells(card: CardRow) -> int:
    """Largest tile (cells) the memory fit puts on ``card``."""

    from woof.domain_wizard import card_assumed_free_gib, fit_headroom_bytes

    free = card_assumed_free_gib(card.memory_gib) * GIB
    usable = free - fit_headroom_bytes(int(free)) - VRAM_INTERCEPT_GIB * GIB
    if usable <= 0:
        raise CostRefusal(f"{card.key}: {card.memory_gib:.1f} GiB leaves no "
                          "room for a tile after the fixed intercept")
    return int(usable / (VRAM_KIB_PER_CELL * KIB))


def _tiles(area_km2: float, dx_m: float, nz: int, cap_cells: int,
           *, min_tiles: int = 1) -> tuple[int, int]:
    """(tile count, square tile side in cells) covering ``area_km2``."""

    interior = math.sqrt(max(area_km2, 0.0)) * 1000.0 / dx_m
    count = max(1, int(min_tiles))
    while True:
        side = math.ceil(interior / math.sqrt(count)) + 2 * TILE_RIM_CELLS
        if side * side * nz <= cap_cells:
            return count, side
        count = max(count + 1, math.ceil(count * (side * side * nz) / cap_cells))
        if count > 10_000_000:
            raise CostRefusal("a cluster needs more than ten million tiles; "
                              "check the area and leaf spacing")


def default_chain(leaf_dx_m: float) -> tuple[float, ...]:
    """The planner's default chain: one ratio-5 parent over the leaf."""

    return (leaf_dx_m * 5.0, leaf_dx_m)


def synth_cluster(cluster_id: str, leaf_area_km2: float, *, leaf_dx_m: float,
                  chain_m: Sequence[float] | None, card: CardRow,
                  leaf_nz: int, parent_nz: int, run_seconds: float,
                  region_km2: float | None = None,
                  min_leaf_tiles: int = 1) -> list[DomainWork]:
    """A cluster's per-segment domains, shaped as the wrf-tiles planner does.

    The root covers the leaf cover (or ``region_km2`` when larger) plus
    :data:`PARENT_BUFFER_KM` on every side and is itself tiled if it does
    not fit ``card``; every finer level covers the leaf area, tiled to the
    card with a :data:`TILE_RIM_CELLS` rim per tile.
    """

    chain = tuple(float(dx) for dx in (chain_m or default_chain(leaf_dx_m)))
    if list(chain) != sorted(chain, reverse=True) or chain[-1] < leaf_dx_m - 1e-6:
        raise CostRefusal(f"cluster {cluster_id!r}: chain "
                          f"{'->'.join(f'{dx:g}' for dx in chain)} m is not "
                          f"coarsest first down to its {leaf_dx_m:g} m leaf")
    if abs(chain[-1] - leaf_dx_m) > 1e-6:
        chain = (*chain, leaf_dx_m)
    cap = max_tile_cells(card)
    rows: list[DomainWork] = []
    for level, dx in enumerate(chain):
        les = dx <= 250.0
        nz = leaf_nz if les or level == len(chain) - 1 else parent_nz
        if level == 0 and len(chain) > 1:
            cover = max(leaf_area_km2, region_km2 or 0.0)
            side_km = math.sqrt(cover) + 2.0 * PARENT_BUFFER_KM
            count, side = _tiles(side_km ** 2, dx, nz, cap)
            role = "parent"
        else:
            count, side = _tiles(leaf_area_km2, dx, nz, cap,
                                 min_tiles=min_leaf_tiles
                                 if level == len(chain) - 1 else 1)
            role = "leaf" if level == len(chain) - 1 else "intermediate"
        rows.append(DomainWork(
            domain_id=f"{cluster_id}:{role}:{dx:g}m", role=role, nx=side,
            ny=side, nz=nz, dx_m=dx, dt_s=_default_dt(dx),
            run_seconds=run_seconds, level=level, count=count,
            peak_gib=round(VRAM_INTERCEPT_GIB + side * side * nz
                           * VRAM_KIB_PER_CELL * KIB / GIB, 3)))
    return rows


def _ring_area_km2(ring: Sequence[Sequence[float]]) -> float:
    """Spherical-excess-free area of a lon/lat ring (equal-area approx)."""

    ring = list(ring)
    if len(ring) > 1 and tuple(ring[0]) == tuple(ring[-1]):
        ring = ring[:-1]
    if len(ring) < 3:
        return 0.0
    radius = 6371.0
    lat0 = math.radians(sum(p[1] for p in ring) / len(ring))
    pts = [(math.radians(p[0]) * radius * math.cos(lat0),
            math.radians(p[1]) * radius) for p in ring]
    twice = sum(x0 * y1 - x1 * y0
                for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))
    return abs(twice) / 2.0


def _region_area_km2(region: Any) -> float | None:
    if region is None:
        return None
    if isinstance(region, (list, tuple)) and len(region) == 4 and \
            all(isinstance(v, (int, float)) for v in region):
        west, south, east, north = (float(v) for v in region)
        return _ring_area_km2([(west, south), (east, south), (east, north),
                               (west, north)])
    if isinstance(region, Mapping):
        if "bbox" in region and "type" not in region:
            return _region_area_km2(region["bbox"])
        kind = region.get("type")
        if kind == "Feature":
            return _region_area_km2(region.get("geometry"))
        if kind == "FeatureCollection":
            return sum(_region_area_km2(f) or 0.0
                       for f in region.get("features") or [])
        if kind == "Polygon":
            return _ring_area_km2(region["coordinates"][0])
        if kind == "MultiPolygon":
            return sum(_ring_area_km2(poly[0]) for poly in region["coordinates"])
    return None


_AREA_KEYS = ("leaf_area_km2", "footprint_km2", "area_km2")
_TILE_KEYS = ("est_tile_count", "est_tiles", "tile_count")
_DX_KEYS = ("leaf_dx_m", "dx_m")
_CHAIN_KEYS = ("chain_m", "chain", "chain_dx_m")


def _first(row: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


@dataclass(frozen=True)
class ClusterSpec:
    cluster_id: str
    leaf_area_km2: float
    leaf_dx_m: float
    chain_m: tuple[float, ...] | None
    region_km2: float | None = None
    est_tiles: int | None = None
    footprint_km2: float | None = None
    #: Identical clusters this row stands for (summary mode: all of them).
    count: int = 1


def clusters_from_partition(path: str | Path, *, leaf_dx_m: float,
                            cover_factor: float,
                            base_dir: Path | None = None) -> list[ClusterSpec]:
    """Cluster rows from a partition document file (see
    :func:`clusters_from_document`)."""

    path = Path(path)
    return clusters_from_document(_read_json(path, "partition"), path,
                                  leaf_dx_m=leaf_dx_m,
                                  cover_factor=cover_factor, base_dir=base_dir)


def clusters_from_document(document: Mapping[str, Any], path: str | Path, *,
                           leaf_dx_m: float, cover_factor: float,
                           base_dir: Path | None = None) -> list[ClusterSpec]:
    """Cluster rows from a partition document, read duck-typed.

    ``path`` names the document in refusals and anchors relative region
    paths.  A row's own ``leaf_dx_m``/``dx_m`` wins over ``leaf_dx_m``
    (the caller refuses a disagreeing flag, :func:`partition_leaf_dx`).
    An explicit area key is farm footprint and is multiplied by
    ``cover_factor``; a ``region`` (bbox or GeoJSON, inline or a path) is
    taken as the leaf cover as drawn and is not.
    """

    path = Path(path)
    rows = document.get("clusters")
    if not isinstance(rows, list) or not rows:
        raise CostRefusal(f"{path} has no 'clusters' list to price")
    out = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise CostRefusal(f"{path}: clusters[{index}] is not an object")
        cid = str(row.get("cluster_id") or row.get("id") or f"c{index:05d}")
        dx = float(_first(row, _DX_KEYS) or leaf_dx_m)
        chain = _first(row, _CHAIN_KEYS)
        try:
            if isinstance(chain, str):
                chain = [float(v) for v in chain.replace("->", ",").split(",")
                         if v.strip()]
            chain = tuple(float(v) for v in chain) if chain else None
        except (TypeError, ValueError) as error:
            raise CostRefusal(f"{path}: cluster {cid!r} chain {chain!r} is not "
                              "a list of metres") from error
        region = row.get("region")
        if isinstance(region, str):
            candidate = (base_dir or path.parent) / region
            region = _read_json(candidate, "partition region") \
                if candidate.suffix in (".json", ".geojson") else None
        region_km2 = _region_area_km2(region)
        footprint = _first(row, _AREA_KEYS)
        tiles = _first(row, _TILE_KEYS)
        if footprint is not None:
            leaf = float(footprint) * cover_factor
        elif region_km2:
            leaf = region_km2
        else:
            raise CostRefusal(
                f"{path}: cluster {cid!r} carries no area (one of "
                f"{', '.join(_AREA_KEYS)}) and no region whose area can be "
                "read; the estimate needs the area its leaf tiles cover")
        out.append(ClusterSpec(cid, leaf, dx, chain, region_km2,
                               int(tiles) if tiles else None,
                               float(footprint) if footprint is not None else None))
    return out


def partition_leaf_dx(clusters: Sequence[ClusterSpec], wanted: float | None,
                      *, origin: str, path: str | Path) -> None:
    """Refuse a leaf spacing ``wanted`` (from ``origin``) that disagrees
    with a partition cluster's own ``leaf_dx_m``.

    A partition's clusters were cut and sized at their own leaf spacing,
    so pricing them at another would quietly price a partition that was
    never made; re-partition at the spacing instead (``woof energy
    partition --leaf-dx-m``, or ``woof energy hindcast optimize``, which
    re-partitions per spacing and card).
    """

    if wanted is None:
        return
    own = sorted({c.leaf_dx_m for c in clusters
                  if abs(c.leaf_dx_m - float(wanted)) > 1e-6})
    if own:
        raise CostRefusal(
            f"{origin} {float(wanted):g} m disagrees with the partition "
            f"{path}, whose clusters were cut at {', '.join(f'{v:g}' for v in own)} m "
            "leaves; the estimate never re-cuts a partition silently.  "
            "Re-partition (woof energy partition --leaf-dx-m "
            f"{float(wanted):g}) or let woof energy hindcast optimize "
            "re-partition per spacing")


# --------------------------------------------------------------------------
# job and prices
# --------------------------------------------------------------------------
def read_job(path: str | Path) -> dict[str, Any]:
    """Segment facts from a hindcast-job-like document (duck-typed)."""

    document = _read_json(Path(path), "job")
    segments = document.get("segments")
    count = (len(segments) if isinstance(segments, list)
             else document.get("segment_count", segments))
    planner = document.get("planner_args") or document.get("planner") or {}
    if isinstance(planner, list):                       # argv form
        flat = {}
        for flag, value in zip(planner, planner[1:]):
            if str(flag).startswith("--"):
                flat[str(flag)[2:].replace("-", "_")] = value
        planner = flat
    try:
        count = int(count) if count is not None else None
    except (TypeError, ValueError) as error:
        raise CostRefusal(f"job {path}: segment count {count!r} is not a "
                          "number") from error
    if count is not None and count < 1:
        raise CostRefusal(f"job {path} lists no segments to price")
    return {
        "cluster": document.get("cluster") or document.get("cluster_id"),
        "segments": count,
        "segment_hours": document.get("segment_hours"),
        "spinup_hours": document.get("spinup_hours", document.get("spin_up_hours")),
        "leaf_dx_m": planner.get("dx_m") if isinstance(planner, Mapping) else None,
        "nz": planner.get("nz") if isinstance(planner, Mapping) else None,
    }


def load_prices(path: str | Path | None = None, *,
                today: date | None = None) -> tuple[dict[str, Any], list[str]]:
    """The price table and any staleness notes; refuses a malformed table."""

    path = Path(path) if path else DEFAULT_PRICES
    table = _read_json(path, "price table")
    if table.get("schema") != PRICES_SCHEMA:
        raise CostRefusal(f"{path} is {table.get('schema')!r}, not "
                          f"{PRICES_SCHEMA}")
    try:
        as_of = date.fromisoformat(str(table["as_of"]))
    except (KeyError, ValueError) as error:
        raise CostRefusal(f"{path} has no ISO 'as_of' date") from error
    notes = []
    today = today or datetime.now(timezone.utc).date()
    stale = int(table.get("stale_after_days") or 180)
    age = (today - as_of).days
    if age > stale:
        notes.append(f"price table {path.name} is dated {as_of} ({age} days "
                     f"old, past its {stale}-day refresh mark); edit it or "
                     "pass --prices with current rates before quoting")
    for row in table.get("instances") or []:
        if card_row(str(row.get("card"))) is None:
            raise CostRefusal(f"{path}: {row.get('provider')}:"
                              f"{row.get('instance')} names card "
                              f"{row.get('card')!r}, which the card catalog "
                              "(woof.card_catalog) has no row for")
    return table, notes


def select_offers(table: Mapping[str, Any],
                  wanted: Sequence[str] | None) -> list[dict[str, Any]]:
    rows = list(table.get("instances") or [])
    if not wanted:
        return rows
    chosen = []
    for spec in wanted:
        provider, _, instance = str(spec).partition(":")
        hits = [r for r in rows if r["provider"] == provider
                and (not instance or r["instance"] == instance)]
        if not hits:
            known = ", ".join(f"{r['provider']}:{r['instance']}" for r in rows)
            raise CostRefusal(f"--instance {spec!r} is not in the price "
                              f"table; it has {known}")
        chosen.extend(h for h in hits if h not in chosen)
    return chosen


# --------------------------------------------------------------------------
# the estimate
# --------------------------------------------------------------------------
def default_segments(segment_hours: float, spinup_hours: float, *,
                     period_hours: float = DEFAULT_PERIOD_HOURS) -> int:
    """Segments whose stored hours cover ``period_hours``."""

    keep = float(segment_hours) - float(spinup_hours)
    if keep <= 0:
        raise CostRefusal(f"spin-up {spinup_hours:g} h leaves nothing stored "
                          f"of a {segment_hours:g} h segment")
    return max(1, math.ceil(float(period_hours) / keep - 1e-9))


@dataclass(frozen=True)
class Campaign:
    segments: int
    segment_hours: float
    spinup_hours: float

    @property
    def stored_hours(self) -> float:
        return max(0.0, self.segment_hours - self.spinup_hours)


def _sum_seconds(domains: Sequence[DomainWork], card: CardRow, k: float,
                 factor: float, leaf_preset: str = "full",
                 utilisation: Utilisation = CENTRAL, *,
                 slots_per_gpu: int | str = 1, mps: bool = False,
                 realisation: float = 1.0,
                 packing_rows: list | None = None,
                 vcpus_per_gpu: int | None = None
                 ) -> tuple[dict[str, float], float]:
    """Totals by component over ``domains`` (counts applied) and the
    critical path: the costliest copy at each chain level, summed.
    ``packing_rows`` collects each domain's packing record."""

    totals = {"compute_s": 0.0, "output_s": 0.0, "checkpoint_s": 0.0,
              "startup_s": 0.0, "saturated_s": 0.0, "total_s": 0.0}
    by_level: dict[int, float] = {}
    for domain in domains:
        packing = packing_factors(
            domain, card, packed_copies(domain, card, slots_per_gpu,
                                        vcpus_per_gpu),
            mps=mps, realisation=realisation, utilisation=utilisation)
        if packing_rows is not None:
            packing_rows.append((domain, packing))
        one = domain_seconds(domain, card, k, factor=factor,
                             leaf_preset=leaf_preset, utilisation=utilisation,
                             packing=packing)
        for key in totals:
            totals[key] += one[key] * domain.count
        by_level[domain.level] = max(by_level.get(domain.level, 0.0),
                                     one["wall_s"])
    return totals, sum(by_level.values())


def _storage(leaf_rows: Sequence[DomainWork], campaign: Campaign, *,
             preset: str, ratio: float, chunk_mib: float) -> dict[str, Any]:
    frames_per_segment = (campaign.stored_hours * 3600.0
                          / (leaf_rows[0].history_s if leaf_rows else DEFAULT_HISTORY_S))
    raw = 0.0
    columns = 0
    for row in leaf_rows:
        interior = max(1, row.nx - 2 * TILE_RIM_CELLS) * \
            max(1, row.ny - 2 * TILE_RIM_CELLS)
        columns += interior * row.count
        raw += _stored_fields(preset, row.nz) * interior * 4 * row.count
    raw_total = raw * frames_per_segment * campaign.segments
    stored = raw_total * ratio
    return {
        "preset": preset, "fields_per_frame": dict(zip(
            ("three_d", "two_d"), STORE_PRESETS[preset])),
        "leaf_columns_stored": columns,
        "frames_per_segment": round(frames_per_segment, 3),
        "raw_bytes": int(raw_total), "compression_ratio": ratio,
        "stored_bytes": int(stored), "stored_tb": round(stored / 1e12, 3),
        "chunk_mib": chunk_mib,
        "put_requests": int(math.ceil(stored / (chunk_mib * 1024 ** 2))),
        "basis": "leaf tile interiors (rim excluded), stored hours only "
                 "(segment minus spin-up), every model level, float32 before "
                 "compression; the ratio is an assumption",
    }


def forcing_transfer(*, source: str, segment_hours: float, cycles: int,
                     jobs_per_cycle: int,
                     gb_per_cycle: float | None = None,
                     objects_per_cycle: int | None = None,
                     publisher_mbps: float = DEFAULT_PUBLISHER_MBPS,
                     cache_read_mbps: float = DEFAULT_CACHE_READ_MBPS,
                     cache_days: float = DEFAULT_FORCING_CACHE_DAYS,
                     publisher_egress_usd_per_gb: float = 0.0) -> dict[str, Any]:
    """Download volume and time per job versus per cycle (the forcing
    section of the estimate, before prices)."""

    from woof.download_budget import download_estimate
    from woof.energy.forcing_cache import MODE_SHARED, shareability

    if publisher_mbps <= 0 or cache_read_mbps <= 0:
        raise CostRefusal("--publisher-mbps and --cache-read-mbps must be "
                          "positive")
    if cache_days < 0:
        raise CostRefusal("--forcing-cache-days must not be negative")
    try:
        mode, why = shareability(source)
    except Exception as error:
        raise CostRefusal(f"--forcing-source {source!r}: {error}") from error
    basis = None
    if gb_per_cycle is None:
        from woof.domain_wizard import fetch_window

        cadence, hours = fetch_window(source, float(segment_hours), 0)
        request: dict[str, Any] = {"source": source,
                                   "cycle": FORCING_REFERENCE_CYCLE,
                                   "hours": hours}
        if cadence is not None:
            request["cadence"] = cadence
        measured = download_estimate(request)
        if measured.get("bytes") is None:
            raise CostRefusal(
                f"--forcing-source {source}: {measured.get('basis')}; give "
                "--forcing-gb-per-cycle")
        cycle_bytes = float(measured["bytes"])
        objects = int(measured.get("objects") or 1)
        basis = (f"woof.download_budget for {source} {FORCING_REFERENCE_CYCLE} "
                 f"+ {hours} h: {measured.get('basis')}")
    else:
        if gb_per_cycle <= 0:
            raise CostRefusal("--forcing-gb-per-cycle must be positive")
        cycle_bytes = gb_per_cycle * 1e9
        objects = int(objects_per_cycle or 1)
        basis = "--forcing-gb-per-cycle (as given)"
    if objects_per_cycle is not None:
        objects = int(objects_per_cycle)
    jobs = cycles * jobs_per_cycle
    shared = mode == MODE_SHARED
    fetch_h = cycle_bytes / (publisher_mbps * 1e6) / 3600.0
    read_h = cycle_bytes / (cache_read_mbps * 1e6) / 3600.0
    per_job = {"downloads": jobs, "bytes_from_publishers": int(jobs * cycle_bytes),
               "slot_hours": round(jobs * fetch_h, 3),
               "publisher_get_requests": jobs * objects}
    if shared:
        per_cycle = {
            "downloads": cycles,
            "bytes_from_publishers": int(cycles * cycle_bytes),
            "prefetch_slot_hours": round(cycles * fetch_h, 3),
            "cache_read_bytes": int(jobs * cycle_bytes),
            "cache_read_slot_hours": round(jobs * read_h, 3),
            "slot_hours": round(cycles * fetch_h + jobs * read_h, 3),
            "cache_put_requests": cycles * objects,
            "cache_get_requests": jobs * objects,
            "cache_gb_months": round(cycles * cycle_bytes / 1e9 * cache_days
                                     / _DAYS_PER_MONTH, 3)}
    else:
        per_cycle = {**per_job, "cache_put_requests": 0,
                     "cache_get_requests": 0, "cache_gb_months": 0.0,
                     "note": f"not cached: {why}"}
    return {
        "source": source, "shared": shared, "why": why,
        "cycles": cycles, "jobs_per_cycle": jobs_per_cycle, "jobs": jobs,
        "per_cycle_download": {"bytes": int(cycle_bytes), "gb": round(
            cycle_bytes / 1e9, 3), "objects": objects, "basis": basis},
        "per_job_fetch": per_job,
        "per_cycle_cache": per_cycle,
        "saving": {
            "bytes_from_publishers": int(per_job["bytes_from_publishers"]
                                         - per_cycle["bytes_from_publishers"]),
            "slot_hours": round(per_job["slot_hours"]
                                - per_cycle["slot_hours"], 3),
            "download_fraction_avoided": round(
                1.0 - per_cycle["bytes_from_publishers"]
                / max(1.0, per_job["bytes_from_publishers"]), 6)},
        "assumptions": {
            "publisher_mbps": publisher_mbps,
            "cache_read_mbps": cache_read_mbps,
            "forcing_cache_days": cache_days,
            "publisher_egress_usd_per_gb": publisher_egress_usd_per_gb,
            "basis": "download time is GPU-slot time (the card idles while "
                     "its job waits); rates and retention are assumptions, "
                     "not measurements; the prefetch runs on the same "
                     "fleet; decode and preparation time are unchanged by "
                     "the cache and not counted here"},
    }


def _forcing_usd(forcing: Mapping[str, Any], offer: Mapping[str, Any],
                 storage_rates: Mapping[str, Any] | None,
                 markets: Sequence[str]) -> dict[str, Any]:
    """The forcing section priced on one offer."""

    gpus = int(offer["gpus"])
    rates = storage_rates or {}
    put = float(rates.get("usd_per_1000_put", 0.0)) / 1000.0
    get = float(rates.get("usd_per_1000_get", 0.0)) / 1000.0
    month = float(rates.get("usd_per_gb_month", 0.0))
    egress = float(forcing["assumptions"]["publisher_egress_usd_per_gb"])
    per_job = forcing["per_job_fetch"]
    cache = forcing["per_cycle_cache"]
    out: dict[str, Any] = {}
    for market in markets:
        price = offer.get("on_demand_usd_per_hour" if market == "on_demand"
                          else "spot_usd_per_hour")
        if price is None:
            continue
        slot = float(price) / gpus
        direct = (per_job["slot_hours"] * slot
                  + per_job["bytes_from_publishers"] / 1e9 * egress)
        cached = (cache["slot_hours"] * slot
                  + cache["bytes_from_publishers"] / 1e9 * egress
                  + cache["cache_put_requests"] * put
                  + cache["cache_get_requests"] * get
                  + cache["cache_gb_months"] * month)
        out[market] = {"slot_usd_per_hour": round(slot, 4),
                       "per_job_fetch": round(direct, 2),
                       "per_cycle_cache": round(cached, 2),
                       "saving": round(direct - cached, 2)}
    return out


def estimate(*, mode: str, campaign: Campaign, offers: Sequence[Mapping[str, Any]],
             price_table: Mapping[str, Any], instances: int = 1,
             deadline_days: float | None = None, markets: Sequence[str] = ("on_demand", "spot"),
             plan_domains: Sequence[DomainWork] | None = None,
             clusters: Sequence[ClusterSpec] | None = None,
             leaf_nz: int = DEFAULT_LEAF_NZ, parent_nz: int = DEFAULT_PARENT_NZ,
             store_preset: str = "full",
             compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
             chunk_mib: float = 8.0,
             packing_efficiency: float = DEFAULT_PACKING_EFFICIENCY,
             slots_per_gpu: int | str = 1, mps: bool = False,
             inputs: Mapping[str, Any] | None = None,
             notes: Sequence[str] = (),
             forcing: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ``woof-energy.cost-estimate.v1`` document.

    ``forcing`` is :func:`forcing_transfer`'s section; each offer then
    carries ``forcing_usd`` (per job versus per cycle, per market)."""

    if instances < 1:
        raise CostRefusal("--instances must be at least 1")
    if slots_per_gpu != "auto" and (isinstance(slots_per_gpu, bool)
                                    or not isinstance(slots_per_gpu, int)
                                    or slots_per_gpu < 1):
        raise CostRefusal(f"--slots-per-gpu {slots_per_gpu!r} must be a "
                          "positive count or 'auto'")
    packed = slots_per_gpu != 1
    if mps and not packed:
        raise CostRefusal("--mps models concurrent kernels of packed jobs; "
                          "with --slots-per-gpu 1 nothing is packed")
    if not 0 < packing_efficiency <= 1:
        raise CostRefusal("--packing-efficiency must be in (0, 1]")
    if not 0 < compression_ratio <= 1.5:
        raise CostRefusal("--compression-ratio must be in (0, 1.5]")
    cal = calibration()
    # K refitted under each utilisation pair: each reproduces the receipts.
    k_by = {label: calibration(u)["k_cell_steps_per_s_per_gbs"]
            for label, u in UTILISATION.items()}
    run_s = campaign.segment_hours * 3600.0
    spot_rework = float((price_table.get("defaults") or {})
                        .get("spot_rework_fraction", 0.10))
    notes = list(notes)
    out_offers = []
    storage_doc = None
    work_doc: Any = None
    for offer in offers:
        card = card_row(str(offer["card"]))
        assert card is not None
        gpus = int(offer["gpus"])
        if mode == "plan":
            from woof.domain_wizard import card_assumed_free_gib

            domains = list(plan_domains or [])
            free = card_assumed_free_gib(card.memory_gib)
            too_big = [d.domain_id for d in domains
                       if d.peak_gib is not None and d.peak_gib > free]
            per_cluster = [("plan", domains)]
        else:
            too_big = []
            per_cluster = [(c.cluster_id, [
                replace(d, count=d.count * c.count) for d in synth_cluster(
                    c.cluster_id, c.leaf_area_km2, leaf_dx_m=c.leaf_dx_m,
                    chain_m=c.chain_m, card=card, leaf_nz=leaf_nz,
                    parent_nz=parent_nz, run_seconds=run_s,
                    region_km2=c.region_km2,
                    min_leaf_tiles=c.est_tiles or 1)])
                for c in clusters or []]
        row: dict[str, Any] = {
            "provider": offer["provider"], "instance": offer["instance"],
            "region": offer.get("region"), "gpus_per_instance": gpus,
            "card": card.as_dict(), "fits": not too_big,
            "instances": instances,
        }
        if too_big:
            row["refused"] = (f"plan domains {', '.join(too_big[:5])} were sized "
                              f"for a larger card than {card.key} "
                              f"({card.memory_gib:.1f} GiB); re-plan with "
                              f"--card {card.key} to price this offer")
            out_offers.append(row)
            continue
        bands: dict[str, float] = {}
        unpacked: dict[str, float] = {}
        critical = 0.0
        components = None
        cluster_rows = []
        utilised: dict[str, float] = {}
        packing_rows: list = []
        vcpus_per_gpu = offer_vcpus_per_gpu(offer) if packed else None
        realisations = {"central": PACKING_REALISATION[1],
                        "low": PACKING_REALISATION[2],
                        "high": PACKING_REALISATION[0]}
        # (band, throughput factor, utilisation pair): central once; low
        # and high as the envelope over the pairs at either band end.
        evaluations = [("central", 1.0, "central")] + [
            (band, factor, label) for band, factor in (
                ("low", THROUGHPUT_BAND[1]), ("high", THROUGHPUT_BAND[0]))
            for label in UTILISATION]
        for band, factor, ulabel in evaluations:
            seg_total = 0.0
            seg_unpacked = 0.0
            seg_components = {"compute_s": 0.0, "output_s": 0.0,
                              "checkpoint_s": 0.0, "startup_s": 0.0}
            saturated = 0.0
            path = 0.0
            for cid, domains in per_cluster:
                totals, crit = _sum_seconds(
                    domains, card, k_by[ulabel], factor,
                    leaf_preset=store_preset, utilisation=UTILISATION[ulabel],
                    slots_per_gpu=slots_per_gpu, mps=mps,
                    realisation=realisations[band],
                    packing_rows=packing_rows if band == "central" else None,
                    vcpus_per_gpu=vcpus_per_gpu)
                if packed:
                    seg_unpacked += _sum_seconds(
                        domains, card, k_by[ulabel], factor,
                        leaf_preset=store_preset,
                        utilisation=UTILISATION[ulabel])[0]["total_s"]
                seg_total += totals["total_s"]
                saturated += totals["saturated_s"]
                for key in seg_components:
                    seg_components[key] += totals[key]
                path = max(path, crit)
                if band == "central" and mode != "plan":
                    cluster_rows.append({
                        "cluster_id": cid,
                        "leaf_tiles": sum(d.count for d in domains if d.role == "leaf"),
                        "gpu_hours": round(totals["total_s"] * campaign.segments / 3600.0, 2)})
            value = seg_total * campaign.segments
            plain = seg_unpacked * campaign.segments
            if band == "central":
                bands["central"] = value
                unpacked["central"] = plain
                critical = path
                components = {key[:-2] + "_h": round(
                    value * campaign.segments / 3600.0, 2)
                    for key, value in seg_components.items()}
            elif band == "low":
                bands["low"] = min(bands.get("low", value), value)
                unpacked["low"] = min(unpacked.get("low", plain), plain)
            else:
                bands["high"] = max(bands.get("high", value), value)
                unpacked["high"] = max(unpacked.get("high", plain), plain)
            if band == "central" or (band, ulabel) in (
                    ("low", "optimistic"), ("high", "pessimistic")):
                name = {"central": "central", "low": "optimistic",
                        "high": "pessimistic"}[band]
                utilised[name] = (round(saturated / seg_components["compute_s"], 3)
                                  if seg_components["compute_s"] else 1.0)
        gpu_hours = {key: round(bands[key] / 3600.0, 2)
                     for key in ("central", "low", "high")}
        billed = {key: value / (gpus * packing_efficiency)
                  for key, value in gpu_hours.items()}
        # Spot pays for, and waits on, the rework its interruptions cost.
        spot_billed = {key: value * (1.0 + spot_rework)
                       for key, value in billed.items()}

        def wall(hours: Mapping[str, float]) -> dict[str, float]:
            return {key: round(max(value / instances, critical / 3600.0) / 24.0, 3)
                    for key, value in hours.items()}

        cost: dict[str, Any] = {}
        prices = {"on_demand": offer.get("on_demand_usd_per_hour"),
                  "spot": offer.get("spot_usd_per_hour")}
        for market in markets:
            price = prices[market]
            if price is None:
                continue
            hours = spot_billed if market == "spot" else billed
            cost[market] = {"usd_per_instance_hour": price, **{
                key: round(value * float(price), 2)
                for key, value in hours.items()}}
        if packed:
            row["packing"] = _packing_row(packing_rows, slots_per_gpu, mps)
            row["packing"]["vcpus_per_gpu"] = vcpus_per_gpu
            row["gpu_hours_one_job_per_gpu"] = {
                key: round(value / 3600.0, 2)
                for key, value in unpacked.items()}
        row.update({
            "gpu_hours": gpu_hours,
            "gpu_hours_by_component": components,
            "instance_hours": {key: round(v, 2) for key, v in billed.items()},
            "wall_clock_days": wall(billed),
            "critical_path_hours_per_segment": round(critical / 3600.0, 3),
            "utilisation": utilised,
            "cost_usd": cost,
        })
        if "spot" in cost:
            row["spot_instance_hours"] = {key: round(v, 2)
                                          for key, v in spot_billed.items()}
            row["wall_clock_days_spot"] = wall(spot_billed)
        if deadline_days:
            # Sized on the slowest market priced: spot's rework included.
            sizing = spot_billed if "spot" in cost else billed
            row["instances_for_deadline"] = {
                "deadline_days": deadline_days,
                "market": "spot" if "spot" in cost else "on_demand",
                **{key: max(1, math.ceil(v / (deadline_days * 24.0)))
                   for key, v in sizing.items()}}
            if critical / 3600.0 > deadline_days * 24.0:
                row["instances_for_deadline"]["note"] = (
                    "one segment's chain is longer than the deadline; no "
                    "instance count meets it")
        if cluster_rows:
            row["clusters"] = cluster_rows
        storage_rates = (price_table.get("storage") or {}).get(offer["provider"])
        leaf_rows = [d for _, ds in per_cluster for d in ds if d.role == "leaf"]
        storage = _storage(leaf_rows, campaign, preset=store_preset,
                           ratio=compression_ratio, chunk_mib=chunk_mib)
        # Synthesised tiles (and so their rims) depend on the card: every
        # offer carries its own store; the top level shows the first.
        row["storage"] = {key: storage[key] for key in
                          ("leaf_columns_stored", "raw_bytes", "stored_bytes",
                           "stored_tb", "put_requests")}
        if storage_doc is None:
            storage_doc = {**storage, "offer": f"{offer['provider']}:"
                                                f"{offer['instance']}"}
        if storage_rates:
            gb = storage["stored_bytes"] / 1e9
            row["storage_usd"] = {
                "service": storage_rates.get("service"),
                "per_month": round(gb * float(storage_rates["usd_per_gb_month"]), 2),
                "puts_once": round(storage["put_requests"] / 1000.0
                                   * float(storage_rates.get("usd_per_1000_put", 0.0)), 2),
                "egress_full_download_once": round(
                    gb * float(storage_rates.get("egress_usd_per_gb", 0.0)), 2),
            }
        if forcing is not None:
            row["forcing_usd"] = _forcing_usd(forcing, offer, storage_rates,
                                              markets)
            if packed:
                # The two sections are priced independently: a download
                # here holds a whole GPU, so with jobs sharing cards the
                # slot cost of waiting on forcing is an upper bound.
                row["forcing_usd"]["note"] = (
                    "slot-hours are priced at a whole GPU; with "
                    "--slots-per-gpu the card is shared while a job waits "
                    "on forcing, so these are upper bounds")
        if mode != "plan":
            row["work_per_segment"] = _work_summary(mode, per_cluster)
        if work_doc is None:
            work_doc = _work_summary(mode, per_cluster)
        out_offers.append(row)
    return {
        "schema": COST_ESTIMATE_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "estimate",
        "mode": mode,
        "inputs": dict(inputs or {}),
        "campaign": {"segments": campaign.segments,
                     "segment_hours": campaign.segment_hours,
                     "spinup_hours": campaign.spinup_hours,
                     "stored_hours_per_segment": campaign.stored_hours,
                     "simulated_hours": campaign.segments * campaign.segment_hours,
                     "clusters": (sum(c.count for c in clusters)
                                  if clusters is not None else 1)},
        "work_per_segment": work_doc,
        "model": {
            "calibration": cal,
            "throughput_band": {"pessimistic": THROUGHPUT_BAND[0],
                                "optimistic": THROUGHPUT_BAND[1],
                                "meaning": "'low' cost/time is the cheapest "
                                           "utilisation pair at the optimistic "
                                           "throughput, 'high' the dearest at "
                                           "the pessimistic"},
            "utilisation": {
                "pairs": {label: u.as_dict() for label, u in UTILISATION.items()},
                "k_by_pair": k_by,
                "meaning": "throughput is not linear in cells: an occupancy "
                           "knee (half_saturation_cells_per_sm) and a fixed "
                           "per-step cost (step_overhead_s); each offer's "
                           "'utilisation' is saturated time / modelled time, "
                           "work-weighted",
                "status": "ASSUMPTIONS, not measurements: replace with a "
                          "cells sweep timed on the rented card",
            },
            "assumptions": {
                "half_saturation_cells_per_sm": HALF_SATURATION_CELLS_PER_SM,
                "step_overhead_s": STEP_OVERHEAD_S,
                "large_step_weight_in_sound_steps": LARGE_STEP_WEIGHT,
                "d2h_gbps": D2H_GBPS, "restart_3d_fields": RESTART_3D_FIELDS,
                "prep_intercept_s": PREP_INTERCEPT_S,
                "prep_s_per_mcell": PREP_S_PER_MCELL,
                "packing_efficiency": packing_efficiency,
                "slots_per_gpu": slots_per_gpu,
                "mps": mps,
                "packing_margin_gib": PACKING_MARGIN_GIB,
                "packing_realisation": {
                    "pessimistic": PACKING_REALISATION[0],
                    "central": PACKING_REALISATION[1],
                    "optimistic": PACKING_REALISATION[2],
                    "meaning": "fraction of the ideal packing gain realised "
                               "(an assumption: no packed run has been "
                               "timed); 'low' cost uses the optimistic "
                               "value, 'high' the pessimistic"},
                "packing_curve": (
                    "copies k = min(slots, (card GiB - margin) // domain "
                    "peak GiB); under MPS the occupancy-bound compute gains "
                    "1 + r*(u(k*c)/u(c) - 1) with u the pair's knee and the "
                    "per-step cost gains 1 + r*(k - 1); without MPS "
                    "(time-slicing) both gain 1; overheads x (1 - r*(1 - "
                    "1/k))"),
                "spot_rework_fraction": spot_rework,
                "vram_kib_per_cell": VRAM_KIB_PER_CELL,
                "vram_intercept_gib": VRAM_INTERCEPT_GIB,
                "tile_rim_cells": TILE_RIM_CELLS,
                "parent_buffer_km": PARENT_BUFFER_KM,
                "seconds_per_km_clock": SECONDS_PER_KM,
            },
        },
        "prices": {"as_of": price_table.get("as_of"),
                   "currency": price_table.get("currency", "USD"),
                   "sources": price_table.get("sources")},
        "storage": storage_doc,
        **({"forcing": dict(forcing)} if forcing is not None else {}),
        "offers": out_offers,
        "notes": notes,
    }


def _packing_row(rows, slots_per_gpu: int | str, mps: bool) -> dict:
    """Per-role packing summary for one offer (central realisation)."""

    by_role: dict[str, dict[str, Any]] = {}
    for domain, packing in rows:
        agg = by_role.setdefault(domain.role, {
            "domains": 0, "copies_min": None, "copies_max": None,
            "compute_gain_min": None, "compute_gain_max": None,
            "launch_gain_min": None, "launch_gain_max": None,
            "peak_gib_max": 0.0})
        agg["domains"] += domain.count
        for key, value in (("copies", packing["copies"]),
                           ("compute_gain", packing["compute_gain"]),
                           ("launch_gain", packing["launch_gain"])):
            low, high = agg[f"{key}_min"], agg[f"{key}_max"]
            agg[f"{key}_min"] = value if low is None else min(low, value)
            agg[f"{key}_max"] = value if high is None else max(high, value)
        agg["peak_gib_max"] = max(agg["peak_gib_max"],
                                  domain_peak_gib(domain))
    for agg in by_role.values():
        for key in ("compute_gain_min", "compute_gain_max", "launch_gain_min",
                    "launch_gain_max", "peak_gib_max"):
            agg[key] = round(agg[key], 3)
    return {"slots_per_gpu": slots_per_gpu, "mps": mps,
            "by_role": by_role,
            "basis": "central realisation; copies bounded by the card's "
                     "memory less the margin, divided by each domain's "
                     "priced peak"}


def _work_summary(mode: str, per_cluster) -> Any:
    if mode == "plan":
        return [d.as_dict() for _, ds in per_cluster for d in ds]
    by_role: dict[str, dict[str, Any]] = {}
    for _, domains in per_cluster:
        for d in domains:
            agg = by_role.setdefault(d.role, {"domains": 0, "cells": 0,
                                              "dx_m": set(), "nz": set()})
            agg["domains"] += d.count
            agg["cells"] += d.cells * d.count
            agg["dx_m"].add(d.dx_m)
            agg["nz"].add(d.nz)
    return {role: {"domains": v["domains"], "cells": v["cells"],
                   "dx_m": sorted(v["dx_m"]), "nz": sorted(v["nz"])}
            for role, v in by_role.items()}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _money(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 1e6:
        return f"${value / 1e6:,.2f}M"
    if value >= 1e3:
        return f"${value / 1e3:,.1f}k"
    return f"${value:,.0f}"


def render_table(document: Mapping[str, Any]) -> str:
    """The human table ``main`` prints to stderr."""

    camp = document["campaign"]
    lines = [
        f"woof energy estimate ({document['mode']}): {camp['clusters']} "
        f"cluster(s) x {camp['segments']} segments x {camp['segment_hours']:g} h "
        f"(prices as of {document['prices']['as_of']}; ESTIMATE, band = "
        "optimistic..pessimistic throughput and utilisation)",
        f"{'offer':<26} {'card':<15} {'GPU-h':>10} {'GPU-h band':>19} "
        f"{'inst-h':>9} {'wall d':>8} {'on-demand':>22} {'spot':>22} "
        f"{'store TB':>9}",
    ]
    for row in document["offers"]:
        name = f"{row['provider']}:{row['instance']}"
        if not row.get("fits", True):
            lines.append(f"{name:<26} {row['card']['key']:<15} REFUSED: "
                         f"{row['refused']}")
            continue
        g = row["gpu_hours"]
        cost = row["cost_usd"]

        def cell(market: str) -> str:
            c = cost.get(market)
            if not c:
                return "-"
            return (f"{_money(c['central'])} "
                    f"({_money(c['low'])}-{_money(c['high'])})")

        lines.append(
            f"{name:<26} {row['card']['key']:<15} {g['central']:>10,.0f} "
            f"{g['low']:>9,.0f}-{g['high']:<9,.0f} "
            f"{row['instance_hours']['central']:>9,.0f} "
            f"{row['wall_clock_days']['central']:>8,.2f} "
            f"{cell('on_demand'):>22} {cell('spot'):>22} "
            f"{row['storage']['stored_tb']:>9,.1f}")
    forcing = document.get("forcing")
    if forcing:
        save = forcing["saving"]

        def volume(n: float) -> str:
            return f"{n / 1e12:,.2f} TB" if n >= 1e12 else f"{n / 1e9:,.2f} GB"

        def hours(h: float) -> str:
            return f"{h:,.0f}" if h >= 100 else f"{h:,.2f}"

        lines.append(
            f"forcing ({forcing['source']}, "
            f"{forcing['per_cycle_download']['gb']:g} GB per cycle, "
            f"{forcing['jobs_per_cycle']} job(s) per cycle): per job "
            f"{volume(forcing['per_job_fetch']['bytes_from_publishers'])} "
            f"and {hours(forcing['per_job_fetch']['slot_hours'])} slot-h; "
            f"cached per cycle "
            f"{volume(forcing['per_cycle_cache']['bytes_from_publishers'])} "
            f"and {hours(forcing['per_cycle_cache']['slot_hours'])} slot-h"
            + ("" if forcing["shared"] else f" (not shared: {forcing['why']})"))
        for row in document["offers"]:
            priced = (row.get("forcing_usd") or {}).get("on_demand")
            if priced:
                lines.append(
                    f"  {row['provider']}:{row['instance']}: forcing "
                    f"{_money(priced['per_job_fetch'])} per job vs "
                    f"{_money(priced['per_cycle_cache'])} cached (saves "
                    f"{_money(priced['saving'])}, on-demand slot time; "
                    "rates are assumptions)")
        lines.append(f"forcing saving: {volume(save['bytes_from_publishers'])} "
                     f"not downloaded, {hours(save['slot_hours'])} slot-h")
    storage = document.get("storage")
    if storage:
        lines.append(f"store: compressed Icechunk TB ({storage['preset']} "
                     f"preset, every level, ratio "
                     f"{storage['compression_ratio']:g}); wall clock and "
                     "on-demand use instance-hours, spot adds its rework")
    instances = document["offers"][0]["instances"] if document["offers"] else 1
    lines.append(f"wall clock is for {instances} instance(s) of each offer; "
                 "GPU-hours exclude the packing loss, instance-hours include it")
    assumptions = (document.get("model") or {}).get("assumptions") or {}
    if assumptions.get("slots_per_gpu", 1) != 1:
        lines.append(
            f"packed: --slots-per-gpu {assumptions['slots_per_gpu']}"
            + (" with CUDA MPS" if assumptions.get("mps") else
               " without MPS (time-sliced kernels: overheads overlap, "
               "compute does not)")
            + "; GPU-hours are card-hours with jobs sharing cards "
            "(gpu_hours_one_job_per_gpu in the JSON is the unpacked figure)")
    for note in document.get("notes") or []:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse_chain(text: str | None) -> tuple[float, ...] | None:
    if not text:
        return None
    try:
        chain = tuple(float(v) for v in text.replace("->", ",").split(",") if v.strip())
    except ValueError as error:
        raise CostRefusal(f"--chain-m {text!r} is not a comma list of metres") from error
    if not chain or any(v <= 0 for v in chain) or list(chain) != sorted(chain, reverse=True):
        raise CostRefusal(f"--chain-m {text!r} must list positive spacings, "
                          "coarsest first, ending at the leaf")
    return chain


def _slots_arg(value: Any) -> int | str:
    from woof.energy.packing import parse_slots_per_gpu

    if value is None:
        return 1
    try:
        return parse_slots_per_gpu(value)
    except ValueError as error:
        raise CostRefusal(str(error)) from None


def run(args) -> dict[str, Any]:
    """Build the estimate document from parsed arguments."""

    modes = [name for name, given in (("plan", args.plan),
                                      ("partition", args.partition),
                                      ("summary", args.clusters is not None))
             if given]
    if len(modes) != 1:
        raise CostRefusal("give exactly one of --plan PLAN.json, --partition "
                          "PARTITION.json, or --clusters N with --footprint-km2 A")
    mode = modes[0]
    # Flags a mode would silently ignore are refused, so a reader never
    # believes they priced something they did not.
    synth_only = {"--footprint-km2": args.footprint_km2,
                  "--chain-m": args.chain_m, "--leaf-dx-m": args.leaf_dx_m,
                  "--nz": args.nz, "--parent-nz": args.parent_nz,
                  "--cover-factor": args.cover_factor}
    if mode == "plan":
        stray = [flag for flag, value in synth_only.items() if value is not None]
        if stray:
            raise CostRefusal(f"{', '.join(stray)} shape synthesised clusters; "
                              "a --plan carries its own domains")
    if mode == "partition" and args.footprint_km2 is not None:
        raise CostRefusal("--footprint-km2 is summary-mode input; a --partition "
                          "carries each cluster's own area")
    job = read_job(args.job) if args.job else {}

    def pick(flag, key, default, kind):
        value = flag if flag is not None else job.get(key)
        if value is None:
            return default
        try:
            return kind(value)
        except (TypeError, ValueError) as error:
            raise CostRefusal(f"{key} {value!r} (from the job) is not a "
                              "number") from error

    segments = pick(args.segments, "segments", None, int)
    spinup = pick(args.spinup_hours, "spinup_hours", DEFAULT_SPINUP_HOURS, float)
    leaf_dx_given = pick(args.leaf_dx_m, "leaf_dx_m", None, float)
    leaf_dx = DEFAULT_LEAF_DX_M if leaf_dx_given is None else leaf_dx_given
    leaf_nz = pick(args.nz, "nz", DEFAULT_LEAF_NZ, int)
    parent_nz = args.parent_nz if args.parent_nz is not None else DEFAULT_PARENT_NZ
    cover_factor = (args.cover_factor if args.cover_factor is not None
                    else DEFAULT_COVER_FACTOR)
    if (segments is not None and segments < 1) or leaf_dx <= 0 or leaf_nz < 1:
        raise CostRefusal("segments, leaf spacing and levels must be positive")
    inputs: dict[str, Any] = {}
    notes: list[str] = []
    plan_domains = clusters = None
    segment_hours = pick(args.segment_hours, "segment_hours", None, float)
    if mode == "plan":
        plan_path = Path(args.plan)
        plan_doc = _read_json(plan_path, "plan")
        # Only an explicit segment length overrides each domain's own run.
        plan_domains = domains_from_plan(
            plan_path, run_seconds=(segment_hours * 3600.0
                                    if segment_hours is not None else None))
        if segment_hours is None:
            segment_hours = float(plan_doc.get("hours") or DEFAULT_SEGMENT_HOURS)
        inputs["plan"] = {"path": str(plan_path), "sha256": _sha256(plan_path),
                          "topology": plan_doc.get("topology"),
                          "plan_hours": plan_doc.get("hours")}
    else:
        segment_hours = float(segment_hours or DEFAULT_SEGMENT_HOURS)
        chain = _parse_chain(args.chain_m)
        # A partition's clusters carry their own leaves: without an explicit
        # spacing the chain lands on the clusters it ends at (below).
        if chain and (mode == "summary" or leaf_dx_given is not None) \
                and abs(chain[-1] - leaf_dx) > 1e-6:
            raise CostRefusal(f"--chain-m ends at {chain[-1]:g} m but the leaf "
                              f"spacing is {leaf_dx:g} m")
        if mode == "partition":
            part = Path(args.partition)
            clusters = clusters_from_partition(part, leaf_dx_m=leaf_dx,
                                               cover_factor=cover_factor)
            partition_leaf_dx(
                clusters, leaf_dx_given, path=part,
                origin=("--leaf-dx-m" if args.leaf_dx_m is not None
                        else "the job's leaf spacing"))
            if chain and not any(abs(c.leaf_dx_m - chain[-1]) <= 1e-6
                                 for c in clusters):
                raise CostRefusal(
                    f"--chain-m ends at {chain[-1]:g} m but no cluster of "
                    f"{part} has that leaf (they are "
                    f"{', '.join(sorted({f'{c.leaf_dx_m:g}' for c in clusters}))} m); "
                    "it would be ignored")
            if chain:
                # The flag's chain applies only where it ends at the
                # cluster's own leaf; a cluster with another leaf keeps its
                # row's chain or the planner's default.
                clusters = [replace(c, chain_m=chain) if c.chain_m is None
                            and abs(c.leaf_dx_m - chain[-1]) <= 1e-6 else c
                            for c in clusters]
            inputs["partition"] = {"path": str(part), "sha256": _sha256(part),
                                   "clusters": len(clusters)}
        else:
            if args.footprint_km2 is None:
                raise CostRefusal("--clusters needs --footprint-km2 (total farm "
                                  "footprint across all clusters, km2)")
            if args.clusters < 1 or args.footprint_km2 <= 0:
                raise CostRefusal("--clusters and --footprint-km2 must be positive")
            per = args.footprint_km2 / args.clusters
            leaf = per * cover_factor
            # One synthesised cluster stands for N identical ones.
            clusters = [ClusterSpec(f"summary-x{args.clusters}", leaf, leaf_dx,
                                    chain, None, None, per, args.clusters)]
            inputs["summary"] = {"clusters": args.clusters,
                                 "footprint_km2": args.footprint_km2,
                                 "footprint_km2_per_cluster": round(per, 3),
                                 "cover_factor": cover_factor,
                                 "leaf_dx_m": leaf_dx, "chain_m": list(chain or default_chain(leaf_dx)),
                                 "leaf_nz": leaf_nz, "parent_nz": parent_nz}
    if args.job:
        inputs["job"] = {"path": str(args.job), **{k: v for k, v in job.items()
                                                    if v is not None}}
    spinup_given = (args.spinup_hours is not None
                    or job.get("spinup_hours") is not None)
    if (not spinup_given and args.segment_hours is not None
            and float(segment_hours) != DEFAULT_SEGMENT_HOURS):
        keep = float(segment_hours) - spinup
        if keep <= 0 or keep % 6:
            # The hindcast driver refuses the same omission.
            raise CostRefusal(
                f"--segment-hours {float(segment_hours):g} without "
                f"--spinup-hours: the default {spinup:g} h spin-up would keep "
                f"{keep:g} h per run, off the 6-hourly cycle; give "
                "--spinup-hours explicitly (0 keeps each run whole)")
    if spinup < 0:
        raise CostRefusal(f"--spinup-hours {spinup:g} is negative")
    if spinup >= float(segment_hours):
        raise CostRefusal(f"spin-up {spinup:g} h leaves nothing stored of a "
                          f"{segment_hours:g} h segment")
    if segments is None:
        # One year of stored hours (the hindcast's 6 h kept windows: 1,460).
        segments = default_segments(float(segment_hours), spinup)
        notes.append(f"no --segments: {segments} segments of "
                     f"{float(segment_hours):g} h ({spinup:g} h spin-up) store "
                     f"one year ({DEFAULT_PERIOD_HOURS:g} h)")
    campaign = Campaign(int(segments), float(segment_hours), spinup)
    table, price_notes = load_prices(args.prices)
    notes.extend(price_notes)
    offers = select_offers(table, args.instance)
    markets = {"both": ("on_demand", "spot"), "on-demand": ("on_demand",),
               "spot": ("spot",)}[args.market]
    forcing = None
    if not getattr(args, "no_forcing", False):
        jobs_per_cycle = (sum(c.count for c in clusters)
                          if clusters is not None else 1)
        # Asked for by name, a forcing figure that cannot be made is a
        # refusal; the default section is a note when it cannot be sized
        # (a window past the default source's horizon), never the reason
        # the GPU-hours are refused.
        asked = any(getattr(args, name, None) is not None for name in (
            "forcing_source", "forcing_gb_per_cycle",
            "forcing_objects_per_cycle", "publisher_mbps",
            "cache_read_mbps", "forcing_cache_days",
            "publisher_egress_usd_per_gb"))
        try:
            forcing = forcing_transfer(
                source=getattr(args, "forcing_source", None)
                or DEFAULT_FORCING_SOURCE,
                segment_hours=campaign.segment_hours, cycles=campaign.segments,
                jobs_per_cycle=jobs_per_cycle,
                gb_per_cycle=getattr(args, "forcing_gb_per_cycle", None),
                objects_per_cycle=getattr(args, "forcing_objects_per_cycle", None),
                publisher_mbps=getattr(args, "publisher_mbps", None)
                or DEFAULT_PUBLISHER_MBPS,
                cache_read_mbps=getattr(args, "cache_read_mbps", None)
                or DEFAULT_CACHE_READ_MBPS,
                cache_days=(DEFAULT_FORCING_CACHE_DAYS
                            if getattr(args, "forcing_cache_days", None) is None
                            else args.forcing_cache_days),
                publisher_egress_usd_per_gb=getattr(
                    args, "publisher_egress_usd_per_gb", None) or 0.0)
        except CostRefusal as error:
            if asked:
                raise
            notes.append(f"forcing section left out: {error}")
        if forcing is not None and mode == "plan":
            notes.append("forcing: a --plan prices one cluster, so one job "
                         "per cycle and no cache saving; price a partition "
                         "or --clusters N to see it")
    document = estimate(
        mode=mode, campaign=campaign, offers=offers, price_table=table,
        instances=args.instances, deadline_days=args.deadline_days,
        markets=markets, plan_domains=plan_domains, clusters=clusters,
        leaf_nz=leaf_nz, parent_nz=parent_nz,
        store_preset=args.store_preset, compression_ratio=args.compression_ratio,
        chunk_mib=args.chunk_mib, packing_efficiency=args.packing_efficiency,
        slots_per_gpu=_slots_arg(getattr(args, "slots_per_gpu", 1)),
        mps=bool(getattr(args, "mps", False)),
        inputs=inputs, notes=notes, forcing=forcing)
    document["prices"]["path"] = str(Path(args.prices) if args.prices else DEFAULT_PRICES)
    return document


def main(args) -> int:
    try:
        document = run(args)
    except CostRefusal as error:
        print(f"woof energy estimate: REFUSED: {error}", file=sys.stderr)
        print(json.dumps({"schema": COST_ESTIMATE_SCHEMA, "refused": str(error)},
                         indent=1))
        return 2
    print(render_table(document), file=sys.stderr)
    if args.output:
        _write_json(document, args.output)
    json.dump(document, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0
