# Launch overhead on small domains (2026-10-15)

The energy hindcast runs hundreds of small domains (0.1-2 M cells, 49 levels,
300 m), several to a card. This study measures what one model step costs the
host and the GPU on such domains, and why. It then fixes the largest cost
that can be removed without changing any number.

**What was found:**

- An ordinary full-physics step issues about 410 kernels and 130 copies and
  memsets, whatever the domain size.
- That step took **7 blocking device reads**, each one a health check. A
  blocking read stops the CPU until the GPU has finished everything queued.
  On a shared card it also waits out the other processes' time slices.
- Below about 0.5 M cells the step is bound by the host, not the GPU:
  80-88% of a 48,000-cell step's wall time is not GPU work.

**What changed:** the health checks now record their verdicts on the GPU.
The step reads them once, at its end.

- The wind-farm suite went from 7 blocking reads per step to 1, plus the run
  loop's own stability check.
- A 48,000-cell step is 1.34x faster, a 0.49 M-cell step 1.19x and a
  1.96 M-cell step 1.06x.
- The final state is bit-identical in every case run.

**What was not changed:** CUDA graph replay of the step. It would take
another 2.3x off a 48,000-cell step, but only 1.08x at 0.49 M cells. A
hindcast leaf also cannot replay a graph across steps until its lateral
boundary time weight moves off the kernel argument list (section 5).

**The cost model:** its per-step fixed cost `L` (1 ms, "about 200 launches
at 5 us") is 10-20x lower than what was measured. Section 6 proposes new
values. `cost.py` is not changed here.

## 1. Method

### The cases

All cases are built with `tilestream.harness.make_config` and
`physics_inventory.default_builder`. The domain is periodic and flat, with a
seeded WK82 sounding. Steps go through `dycore.step`, and every step ends
with `dycore.stability_report`, the resident run loop's own per-step
readback.

| suite | selectors |
|---|---|
| `windfarm` (the 300 m leaf) | Thompson mp 8, MYNN surface layer and PBL (5/5), Noah (2), RTE-RRTMGP 4/4 (radt 5 min), no cumulus, km_opt 4, dx 300 m, dt 1.5 s, 4 acoustic substeps |
| `hrrr` | `configs/hrrr_native_3km_demo.toml`'s selectors: the same schemes with `ra_rrtmg_variant = "rrtmg_legacy"`, dx 3 km, dt 15 s |
| `dry` | the dry dynamical core |

The real leaves add specified-boundary kernels, which a periodic domain does
not have. The launch counts below are therefore a lower bound for a real
leaf.

### Sizes

| grid | cells |
|---|---:|
| 40x40x30 | 0.05 M |
| 100x100x49 | 0.49 M |
| 200x200x49 | 1.96 M |

The plan's 300x300x49 does not fit in the 6 GiB cap for test jobs, and
neither does 256x256x49. The RRTMGP workspace and the 200 MB physics
temporaries take the rest. So 200x200x49 stands in for the top of the
hindcast range.

### Steps

Each run warms up with one radiation step and three ordinary steps, which
compiles every kernel. It then times 8 ordinary steps and 2 radiation steps.
Radiation steps are forced with the driver's `radiation_due_override`, the
adaptive clock's own hook, so the cadence never decides what is measured.

### Profiler

`tools/profile_launch_overhead.py` runs one lane per process. Its CUPTI
library is `tools/launch_profile/woof_cupti.c`, built on demand against the
toolkit's CUPTI.

CuPy 14 loads `libcuda` and `libcudart` with `dlopen`, so an `LD_PRELOAD`
counter would see nothing. There is also no central launch helper to count
in Python: kernels are `RawModule` functions and CuPy array operations. The
CUPTI callback API sees every driver and runtime call. The activity API
gives the executed kernel, copy and memset intervals.

| lane | what it measures |
|---|---|
| `timing` | Wall time per step: `perf_counter` around the step and its stability check, closed by a device synchronise. Also CUDA-event time and host time per physics section. No CUPTI. |
| `cupti` | Per step: kernel launches, copies, memsets, host synchronisations (explicit syncs plus blocking copies), allocations and total API calls. **GPU busy** is the union of executed kernel, copy and memset intervals. **Host submission time** is wall minus the time spent inside calls that can block. Each launch is tagged with the section that issued it. |
| `alloc` | CuPy memory-pool mallocs and bytes per step. |
| `cprofile` | Top functions by own time, and who calls each blocking read. |

The sections are: dynamics; radiation; surface layer; land; PBL;
microphysics; the driver's other physics glue; and the loop's health check.

**Launch-bound fraction** = (wall - GPU busy) / wall. It is the share of
the step during which the card was not working on this process.

### Conditions

- **Card:** RTX A6000 (84 SMs, 768 GB/s).
- **Host:** Ryzen 9 5950X; driver 610.57, CUDA 13.3 toolkit; CuPy 14.2.0 on
  CUDA 12.9.
- **Co-tenants:** the card was shared throughout. The baseline matrix ran
  beside two forecasts at 100% utilisation, using 8.6 GB and 24.8 GB. The
  A/B ran beside one of them after the other was paused.
- **Effect of sharing:** absolute times are inflated and noisy, and kernel
  durations include time-slice waits. Read **ratios between interleaved
  runs**, not absolutes.
- **GPU work:** every run held `/tmp/woof-gpu-test.lock` with
  `CUPY_GPU_MEMORY_LIMIT=6442450944`.
- **A/B interleaving:** each A/B repetition held the lock once and ran
  every case in order, base, deferred, immediate.

### Free-threaded Python

There is no `python3.14t` on this machine and no CuPy build for it, so
nothing here was measured without the interpreter lock. MULTIGPU.md's
figures (13-25 s versus 5-8 s per radiation step on a two-socket four-card
box) remain the only free-threading data. This study is single-threaded
throughout.

## 2. Baseline (jacob/main)

These are medians over 3 timing runs (range in brackets). GPU busy and the
counts come from one CUPTI run. The baseline matrix ran beside two
co-tenants.

| suite | grid | cells | step | wall ms | GPU busy ms | launch-bound | launches | memcpy | memset | host syncs | API calls | pool mallocs | pool MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| windfarm | 40x40x30 | 0.05 M | ordinary | 49.6 (33.1-53.1) | 9.8 | 0.80 | 390 | 70 | 58 | 9 | 2703 | 95 | 11 |
| windfarm | 40x40x30 | 0.05 M | radiation | 116.4 (72.2-131.0) | 24.3 | 0.79 | 697 | 93 | 67 | 21 | 5011 | 442 | 638 |
| windfarm | 100x100x49 | 0.49 M | ordinary | 100.3 (96.8-104.0) | 53.8 | 0.46 | 409 | 70 | 58 | 9 | 2779 | 95 | 115 |
| windfarm | 100x100x49 | 0.49 M | radiation | 267.6 (259.9-374.5) | 211.8 | 0.21 | 902 | 96 | 76 | 21 | 6899 | 727 | 6018 |
| windfarm | 200x200x49 | 1.96 M | ordinary | 238.6 (231.1-247.0) | 178.7 | 0.25 | 409 | 70 | 58 | 9 | 2779 | 95 | 459 |
| windfarm | 200x200x49 | 1.96 M | radiation | 872.7 (820.1-909.6) | 774.0 | 0.11 | 1460 | 105 | 103 | 21 | 12407 | 1582 | 24071 |
| hrrr | 40x40x30 | 0.05 M | ordinary | 53.0 (48.0-76.7) | 6.3 | 0.88 | 390 | 70 | 58 | 9 | 2703 | 95 | 11 |
| hrrr | 40x40x30 | 0.05 M | radiation | 379.7 (246.9-383.7) | 33.5 | 0.91 | 868 | 186 | 117 | 48 | 7188 | 814 | 1411 |
| hrrr | 100x100x49 | 0.49 M | ordinary | 95.5 (87.2-96.6) | 56.4 | 0.41 | 409 | 70 | 58 | 9 | 2779 | 95 | 115 |
| hrrr | 100x100x49 | 0.49 M | radiation | 587.8 (564.6-1766.4) | 289.4 | 0.51 | 2277 | 470 | 305 | 124 | 19424 | 2800 | 9612 |
| hrrr | 200x200x49 | 1.96 M | ordinary | 240.7 (167.1-249.1) | 205.6 | 0.15 | 409 | 70 | 58 | 9 | 2779 | 95 | 459 |
| hrrr | 200x200x49 | 1.96 M | radiation | 2405.4 (2385.4-24085.1) | 1191.0 | 0.50 | 7646 | 1580 | 1019 | 424 | 66385 | 10461 | 36278 |
| dry | 40x40x30 | 0.05 M | ordinary | 6.1 (3.6-6.2) | 1.3 | 0.78 | 93 | 1 | 0 | 2 | 504 | 14 | 1 |
| dry | 100x100x49 | 0.49 M | ordinary | 18.2 (13.7-18.6) | 14.0 | 0.23 | 93 | 1 | 0 | 2 | 504 | 14 | 6 |
| dry | 200x200x49 | 1.96 M | ordinary | 34.3 (26.8-39.7) | 35.1 | -0.02 | 93 | 1 | 0 | 2 | 504 | 14 | 25 |

The host-sync counts include one synchronise per step by the profiler
itself. CUPTI dropped no activity records.

### Section split (baseline, CUPTI run)

Each cell is launches / GPU-busy ms / host ms. Host time includes the waits
inside blocking reads.

| suite | grid | step | dynamics | radiation | surface layer | land | PBL | microphysics | physics glue | loop health |
|---|---|---|---|---|---|---|---|---|---|---|
| windfarm | 40x40x30 | ordinary | 250/5.0/14.2 | 0/0/0 | 1/0.1/0.3 | 3/0.1/0.6 | 33/1.6/15.1 | 19/0.4/2.0 | 82/0.1/13.8 | 2/0.0/4.3 |
| windfarm | 40x40x30 | radiation | 250/6.2/12.6 | 307/15.3/67.4 | 1/0.1/0.5 | 3/0.0/0.5 | 33/1.9/15.3 | 19/0.4/1.8 | 82/0.1/14.6 | 2/0.0/3.1 |
| windfarm | 100x100x49 | ordinary | 250/41.6/57.0 | 0/0/0 | 1/0.1/0.5 | 3/0.1/0.5 | 33/6.6/22.0 | 19/5.6/1.7 | 101/0.3/14.0 | 2/0.0/3.8 |
| windfarm | 100x100x49 | radiation | 250/46.4/50.7 | 493/153.7/178.8 | 1/0.1/0.4 | 3/0.1/0.5 | 33/3.8/17.8 | 19/6.6/1.7 | 101/0.3/16.8 | 2/0.0/3.9 |
| windfarm | 200x200x49 | ordinary | 250/129.6/177.1 | 0/0/0 | 1/0.1/0.4 | 3/0.1/0.5 | 33/24.7/42.3 | 19/18.3/1.6 | 101/1.2/15.0 | 2/0.1/4.1 |
| windfarm | 200x200x49 | radiation | 250/150.2/161.6 | 1051/576.2/648.9 | 1/0.1/0.4 | 3/0.1/0.4 | 33/23.9/40.3 | 19/19.8/1.5 | 101/1.2/15.5 | 2/0.1/3.0 |
| hrrr | 40x40x30 | ordinary | 250/3.7/17.1 | 0/0/0 | 1/0.1/0.3 | 3/0.1/0.4 | 33/1.8/15.7 | 19/0.4/1.9 | 82/0.1/13.9 | 2/0.0/4.4 |
| hrrr | 40x40x30 | radiation | 250/3.5/17.4 | 478/27.0/322.0 | 1/0.1/0.4 | 3/0.1/0.4 | 33/1.8/15.7 | 19/0.4/1.8 | 82/0.1/15.5 | 2/0.0/3.9 |
| hrrr | 100x100x49 | ordinary | 250/44.7/55.9 | 0/0/0 | 1/0.1/0.4 | 3/0.1/0.5 | 33/3.8/17.4 | 19/6.8/2.1 | 101/0.3/12.9 | 2/0.0/3.9 |
| hrrr | 100x100x49 | radiation | 250/45.3/38.4 | 1868/231.0/516.0 | 1/0.1/0.6 | 3/0.1/0.5 | 33/3.6/15.7 | 19/6.6/2.1 | 101/0.3/9.6 | 2/0.0/2.1 |
| hrrr | 200x200x49 | ordinary | 250/152.8/178.5 | 0/0/0 | 1/0.1/0.3 | 3/0.1/0.4 | 33/21.9/37.5 | 19/26.4/1.6 | 101/1.2/14.0 | 2/0.1/3.0 |
| hrrr | 200x200x49 | radiation | 250/109.0/137.6 | 7237/1026.5/2218.4 | 1/0.1/0.5 | 3/0.1/0.5 | 33/23.6/32.7 | 19/18.0/1.8 | 101/1.2/12.0 | 2/0.1/3.4 |

### What the baseline says

1. **The launch count does not grow with the domain.** An ordinary step is
   390-409 kernels, 70 copies and 58 memsets at every size, and 2,700-2,800
   driver and runtime calls in all.
   - `cudaGetDevice` alone is 2,141 of those calls (5.5 per launch). That is
     CuPy's per-launch device check. It is cheap per call but not free.
   - The dynamics is 250 of the launches; the dry core alone issues 93.
2. **The host waits on the GPU, it does not out-run it.** In the timing lane,
   "issue" (host time until the step returns) equals wall time to within
   0.1 ms in every case. The CPU never got ahead of the card. Seven of the
   nine synchronisations per ordinary step are inside the step. Each one
   drains the queue, so CPU submission and GPU execution run in series
   rather than overlapping.
3. **The blocking reads are all health checks.** cProfile's callers of
   `.get()`, `.item()` and the carrier-contract reductions (windfarm,
   40x40x30, per ordinary step):

   | site | reads per step | what it checks |
   |---|---:|---|
   | `radiation_carriers._all_finite` (GLW, SWDOWN) | 2 | the land surface's radiative inputs are finite |
   | `radiation_carriers._consumer_cells_exist` | 1 | whether any cell is land or ice |
   | `health_ledger.read_status` (microphysics) | 1 | Thompson's surface outputs are finite and in range |
   | `mynn_pbl_gpu._first_refusal` | 2 | MYNN's inputs are finite and positive |
   | `mynn_pbl_runtime.validate_mynn_tendencies` (`_flag_mask`) | 1 | MYNN's tendencies are finite |
   | `dycore.stability_report` | 1 | the run loop's per-step stability gate (kept) |

   Under cProfile those 8 reads took 63% of a 51 ms step. That is CPU time
   spent waiting for the shared card, not doing work.
4. **Small domains are host-bound.**
   - At 0.05 M cells the launch-bound fraction is 0.80-0.88. At 0.49 M it is
     0.41-0.46, and at 1.96 M it is 0.15-0.25.
   - The dry core reaches -0.02 at 1.96 M: GPU-bound, with 93 launches and
     2 syncs.
5. **Legacy RRTMG's radiation step is host-bound at every size.** The SW
   driver prepares per-column scalars on the host (`rrtmg_sw` `hostf`, 12
   reads per call at 40x40). The radiation step therefore took 48 syncs at
   0.05 M cells, 124 at 0.49 M and 424 at 1.96 M, and 2.4 s against 1.2 s
   of GPU work at 1.96 M. HRRR's radiation runs one step in 48. RTE-RRTMGP
   takes 21 syncs, independent of size.
6. **Memory-pool churn is not the bottleneck.**
   - An ordinary step makes 95 pool mallocs (459 MB at 1.96 M cells). At
     about 2-5 us per malloc that is about 0.5 ms per step.
   - A RTE-RRTMGP radiation step at 1.96 M cells makes 1,582 mallocs and
     moves 24 GB through the pool, with 33 raw `cudaMalloc` calls (the 6 GiB
     cap forces the pool to return and regrow). That is worth its own look,
     as an allocation question rather than a launch one.

## 3. The decision

The plan set three candidates. These are their measured payoffs at hindcast
sizes:

| candidate | evidence | verdict |
|---|---|---|
| Remove per-step host syncs | 7 blocking health reads per step, all error-only or input checks. Removing them restores CPU/GPU overlap. The sites are already structured for `woof.core.health_ledger`. | **implemented** |
| CUDA graph replay of the ordinary step | After the sync removal the wind-farm step captures (532 nodes), and replay is 2.3x at 0.05 M cells but 1.08x at 0.49 M (section 5). Specified-boundary leaves bake a per-step time weight into the boundary kernels' arguments. | not implemented; numbers and blockers below |
| Fuse tiny elementwise kernels / cut allocations | No single hotspot. Pool churn is about 0.5 ms per ordinary step, and the 410 launches are spread over 250 dynamics kernels and the schemes. | not implemented |

The threshold for graphs was set before the probe was run: at least 1.15x
at 0.49 M cells (the middle of the hindcast range), on a configuration a
hindcast leaf can use. Neither condition held.

## 4. The change: deferred health reads in the resident step

`dycore.step` now runs its body inside `health_ledger.resident_step()`.

- The schemes' error-only checks record their verdicts in a per-thread
  device ledger (`HealthLedger.record` is one device-to-device OR). The step
  drains the ledger once, at its end: one device read, and none at all if
  nothing recorded.
- A failure is still raised **by the step that produced it**. This happens
  before the run loop's stability gate, output, checkpoint or nest coupling
  sees the state. The exception type and message are the site's own, with a
  note that the check failed earlier in the step.

**Sites deferred** (`health_ledger.RESIDENT_SITES`):

- the microphysics, YSU and Kain-Fritsch output words;
- `physics._validated_array` scheme outputs;
- the surface-radiation carrier finiteness check;
- MYNN's guards. Their input refusals and output check gained a deferred
  form: a device fold of the flag words into one ledger word, with one bit
  per check (`health_ledger.record_span_bits`).

**Sites kept immediate**: RRTMGP's input guard (its eager path also replays
the detailed validators), Grell-Freitas, WDM6 and the urban models. The
tiled/graph ledger (`deferring`) still defers them all, as before.

**Exactness:**

- **Order.** The resident ledger is *ordered*: each slot keeps the sequence
  number of its first failing record. The drain therefore raises the failure
  an immediate read would have raised first, not the site registered first.
- **The all-water footprint.** The carrier check under the resident ledger
  reads the land/ice footprint only on a path that is about to refuse or
  relabel. The finiteness verdict is masked by the footprint on the device.
  So an all-water domain is never refused where the immediate check would
  pass.
  - The tiled ledger keeps its documented over-enforcement, because a read
    inside graph capture is refused outright.
  - A host-resident footprint is evaluated on the host, which needs no
    device read.
- **Exception precedence.** If the step body raises after a deferred check
  failed, the deferred refusal is raised, chained from the later exception.
  A drain that cannot read at all (a sticky CUDA fault) never replaces the
  step's own exception.
- **The failing path.** A deferred input guard lets MYNN run once on inputs
  it rejected before the step raises. MYNN's loops are bounded by the column
  depth (checked in `mynn_pbl.cu`), so this costs work, not a hang.
  `WOOF_HEALTH_READBACK=immediate` stops at the guard itself.

**Opt-out and receipts.**

- `WOOF_HEALTH_READBACK=immediate` restores the per-site reads. An unknown
  value is refused.
- The resident step defers to any ledger that is already active: a tiled
  sweep, a graph capture, or an executor that lags its health check. It also
  steps aside when the stream is capturing. Each case is counted under a
  named reason.
- The four run receipts carry `health_readback`: the mode, its source and
  the deferred, drained and immediate step counts.

**Cost per ordinary step:**

| | blocking reads in the step | launches |
|---|---:|---:|
| before | 7 | 390-409 |
| after | 1 | 407-426 |

The 17 extra launches are the folds and ledger ORs.

### A/B: switch on vs off, same tree

`imm` is this tree with `WOOF_HEALTH_READBACK=immediate`; `new` is the
default; `base` is jacob/main. Each row is the median of 3 interleaved
repetitions (range in brackets). The **off/on** ratio is imm / new. "Digests
equal" covers the carrier inventory after the run, across all three sides
and every repetition.

| suite | grid | step | base wall ms | new wall ms | imm wall ms | off/on | digests equal |
|---|---|---|---|---|---|---|---|
| windfarm | 40x40x30 | ordinary | 29.2 (27.8-36.1) | 20.4 (18.1-26.9) | 27.4 (18.1-27.6) | 1.34x | yes |
| windfarm | 40x40x30 | radiation | 78.5 (69.9-84.5) | 38.2 (38.0-47.3) | 69.1 (50.4-71.1) | 1.81x | yes |
| windfarm | 100x100x49 | ordinary | 62.7 (52.3-63.1) | 53.0 (45.8-54.2) | 63.1 (61.6-67.6) | 1.19x | yes |
| windfarm | 100x100x49 | radiation | 198.2 (148.1-245.8) | 164.7 (109.8-196.4) | 189.8 (189.7-198.6) | 1.15x | yes |
| windfarm | 200x200x49 | ordinary | 164.7 (162.4-168.2) | 153.2 (145.8-155.5) | 162.5 (161.3-164.0) | 1.06x | yes |
| windfarm | 200x200x49 | radiation | 576.7 (461.3-610.6) | 589.4 (582.8-600.5) | 536.4 (516.5-611.5) | 0.91x | yes |
| hrrr | 40x40x30 | ordinary | 30.2 (27.2-37.2) | 20.8 (17.3-24.6) | 25.4 (24.9-45.0) | 1.22x | yes |
| hrrr | 40x40x30 | radiation | 163.9 (133.8-195.8) | 100.6 (46.6-110.5) | 124.1 (91.8-124.3) | 1.23x | yes |
| hrrr | 100x100x49 | ordinary | 62.0 (45.2-67.3) | 53.5 (45.6-54.8) | 62.2 (62.1-63.2) | 1.16x | yes |
| hrrr | 100x100x49 | radiation | 1142.2 (722.2-2062.2) | 407.7 (346.4-438.0) | 371.0 (349.7-394.3) | 0.91x | yes |
| hrrr | 200x200x49 | ordinary | 162.4 (133.8-165.8) | 157.1 (68.9-157.8) | 163.3 (115.6-164.1) | 1.04x | yes |
| hrrr | 200x200x49 | radiation | 7785.1 (3822.1-7853.8) | 1437.1 (606.3-1481.2) | 1418.4 (925.6-1570.7) | 0.99x | yes |

**How to read this table:**

- **Ordinary steps (the 199 in 200 of a hindcast):** 1.34x / 1.19x / 1.06x
  for the wind-farm suite at 0.05 / 0.49 / 1.96 M cells. The HRRR suite
  gives 1.22x / 1.16x / 1.04x.
- **The legacy-RRTMG radiation step:** unchanged within noise. Its host
  reads are the SW driver's per-column preparation, which is not a health
  check (section 2, item 5).
- **The 1.96 M-cell RTE-RRTMGP radiation step (0.91x):** inside the spread
  of its 3 repetitions on a shared card.
- **Base vs imm:** these run the same code. They agree within noise except
  on the legacy-RRTMG radiation step, where `base` was consistently slower.
  `base` ran from a code-only copy of jacob/main at another path, and the
  cause was not established. **The switch comparison (imm vs new) is the
  measurement of this change.**

### CUPTI A/B (one repetition, launch-bound fraction from that run)

| suite | grid | step | side | wall ms | GPU busy ms | launch-bound | launches | host syncs | API calls |
|---|---|---|---|---|---|---|---|---|---|
| windfarm | 40x40x30 | ordinary | imm | 39.0 | 6.0 | 0.85 | 390 | 9 | 2705 |
| windfarm | 40x40x30 | ordinary | new | 25.9 | 9.8 | 0.62 | 407 | 3 | 2759 |
| windfarm | 40x40x30 | radiation | imm | 87.5 | 14.6 | 0.83 | 697 | 21 | 5013 |
| windfarm | 40x40x30 | radiation | new | 49.9 | 19.5 | 0.61 | 741 | 6 | 5148 |
| windfarm | 100x100x49 | ordinary | imm | 63.2 | 40.5 | 0.36 | 409 | 9 | 2781 |
| windfarm | 100x100x49 | ordinary | new | 59.4 | 44.7 | 0.25 | 426 | 3 | 2835 |
| windfarm | 100x100x49 | radiation | imm | 190.1 | 140.4 | 0.26 | 902 | 21 | 6901 |
| windfarm | 100x100x49 | radiation | new | 177.0 | 153.1 | 0.13 | 946 | 6 | 7036 |
| windfarm | 200x200x49 | ordinary | imm | 122.3 | 99.1 | 0.19 | 409 | 9 | 2781 |
| windfarm | 200x200x49 | ordinary | new | 71.7 | 63.8 | 0.11 | 426 | 3 | 2835 |
| hrrr | 40x40x30 | ordinary | imm | 48.8 | 6.4 | 0.87 | 390 | 9 | 2705 |
| hrrr | 40x40x30 | ordinary | new | 30.7 | 5.6 | 0.82 | 407 | 3 | 2759 |
| hrrr | 40x40x30 | radiation | imm | 166.5 | 19.3 | 0.88 | 868 | 48 | 7184 |
| hrrr | 40x40x30 | radiation | new | 69.5 | 17.7 | 0.75 | 912 | 33 | 7280 |
| hrrr | 100x100x49 | ordinary | imm | 59.1 | 30.3 | 0.49 | 409 | 9 | 2781 |
| hrrr | 100x100x49 | ordinary | new | 62.5 | 43.6 | 0.30 | 426 | 3 | 2835 |
| hrrr | 100x100x49 | radiation | imm | 512.4 | 226.7 | 0.56 | 2277 | 124 | 19420 |
| hrrr | 100x100x49 | radiation | new | 505.8 | 233.8 | 0.54 | 2321 | 109 | 19555 |
| hrrr | 200x200x49 | ordinary | imm | 134.5 | 118.8 | 0.12 | 409 | 9 | 2781 |
| hrrr | 200x200x49 | ordinary | new | 155.8 | 143.9 | 0.08 | 426 | 3 | 2835 |

**Host submission time** is wall minus the time inside calls that can block.
For an ordinary step under CUPTI it was:

| suite | 0.05 M | 0.49 M | 1.96 M |
|---|---:|---:|---:|
| windfarm, deferred | 21.6 ms | 20.4 ms | 20.0 ms |
| HRRR, deferred | 28.0 ms | 40.3 ms | 24.0 ms |

It is roughly constant in cells: what the host costs per step is a property
of the physics suite, not the grid. CUPTI's own per-call cost (about 2,800
calls) is in these figures. Without it the timing lane suggests about 15-20
ms.

### Tests

- **`tests/test_resident_health_ledger.py` (CPU):** the switch and its
  refusal, the named fallbacks, the receipt shape and the exact host
  footprint.
- **`tests/test_resident_health_ledger_gpu.py` (card):**
  - raising in the producing step, and clearing for the next;
  - per-site admission and program-order precedence;
  - deferred-over-later-exception chaining, and a broken drain that never
    hides the step's exception;
  - the per-thread ledger;
  - MYNN's deferred refusals naming the check the immediate read names, and
    the zero-forcing mask;
  - the device and host carrier footprint;
  - a 24x20x16 wind-farm run of 4 steps, including a forced radiation step,
    **bit-identical** deferred vs immediate, with the deferred sites shown
    in the ledger;
  - a NaN injected into `qv` stopping in that step with the deferred note.
- **`tests/test_surface_radiation_carrier_contract.py`:**
  `test_the_check_is_capturable_inside_a_cuda_graph` gained a device
  synchronise. It wrote the NaN on the legacy stream and replayed on a
  non-blocking one, so it passed or failed on timing.

## 5. CUDA graph headroom (measured, not implemented)

With the in-step syncs gone, an ordinary wind-farm step captures through
`tilestream.graphcap.capture_step` under a ledger. The probe ran in this
tree, beside one co-tenant:

| grid | nodes | eager ms | replay ms | replay speedup |
|---|---:|---:|---:|---:|
| 40x40x30 | 532 | 15.5 | 6.6 | 2.3x |
| 100x100x49 | 551 | 47.7 | 44.0 | 1.08x |
| 200x200x49 (beside two co-tenants) | 549 | 217.5 | 228.8 | 0.95x |

**What blocks reusing a graph across steps on a hindcast leaf:**

1. **Specified lateral boundaries.** `apply_state_lateral_boundaries` passes
   `dtbc`, the time since the boundary update, as a scalar kernel argument
   that changes every step. A replayed graph would reuse the captured value.
   Moving it to a one-word device scalar written before each replay would
   keep the arithmetic identical (same float). This is the prerequisite.
2. **Radiation steps.** These bake the solar geometry into their arguments
   (graphcap's documented "full fast cadence" mismatch). They would run
   eagerly, keyed by `graphcap.cadence_key`.
3. **The legacy-RRTMG host preparation.** It is not capturable. HRRR's
   radiation steps would stay eager.
4. **graphcap's host fingerprint.** It raises `TypeError` on an
   `rrtmg_legacy` object whose `shape` attribute is a method. That is a
   small fix in `_fingerprint_value`.

For a 0.1-2 M-cell leaf the payoff is 1.0-1.1x on this card. It would be
more on a faster card or a quieter host, where the GPU work shrinks and the
fixed ~20 ms of host work does not. That is a real lever for the smallest
leaves, worth doing after item 1. It also cuts CPU per slot, which is what
packing several jobs per card is limited by.

## 6. What this means for the hindcast cost model

`woof/energy/cost.py` prices a step as
`(n_sound+4)/8 * (cells / (K B u(c)) + L)`, with `u(c) = c / (c + S h)`.

**`L`, the fixed per-step cost.** The model assumes 1 ms ("about 200
launches at about 5 us"). Measured for the wind-farm suite:

- about 410 launches plus 130 copies and memsets per step;
- about 20 ms of host submission per ordinary step at every size;
- after this change, a non-overlapped residual (wall - GPU busy) of 15 ms
  at 0.05 M, 14 ms at 0.49 M and 8-13 ms at 1.96 M cells, beside one
  co-tenant.

Host time and GPU time overlap now that the step has one sync. So the
additive `+ L` form should carry that residual, not the full host time. The
residual is still 10-20x the assumed 1 ms.

**`h`, the occupancy knee.** GPU busy is close to affine in cells:
`busy = (c + S h) / (K B)`. The baseline (two co-tenants) gives an intercept
of 12 ms and a slope of 85 ms per M cells, so `S h` is about 0.14 M cells,
or `h` about 1,700 cells per SM on 84 SMs. The one-co-tenant CUPTI run gives
`h` about 4,300. The current central knee of 4,096 is consistent with the
data. The sharing makes the slope too noisy to narrow it further.

**Proposed pairs.** `K` is refitted per pair on the same receipts, as
`calibration()` does now.

| pair | `h` (cells per SM) | `L` (per step) | basis |
|---|---:|---:|---|
| optimistic | 2,048 | 5 ms | a faster host CPU, a quiet card, or graph replay of the ordinary step (section 5) |
| central | 4,096 | 12 ms | the measured residual after this change at 0.05-0.49 M cells |
| pessimistic | 16,384 | 25 ms | the pre-change residual, or a packed card without MPS where syncs wait on siblings |

**Effect on a 300 m leaf**, 1 hour at dt 1.5 s and 49 levels, from
`domain_seconds` with each pair's refitted `K`. Each cell is utilisation /
ms per step.

| pair | 0.49 M H100 | 0.49 M A100-40 | 0.49 M A10G | 0.49 M L4 | 1.96 M H100 | 1.96 M A100-40 | 1.96 M A10G | 1.96 M L4 |
|---|---|---|---|---|---|---|---|---|
| current central (4,096, 1 ms) | 0.42 / 8.0 | 0.49 / 14.6 | 0.58 / 32.0 | 0.66 / 56.2 | 0.74 / 18.0 | 0.79 / 36.1 | 0.85 / 87.7 | 0.89 / 167.6 |
| proposed optimistic (2,048, 5 ms) | 0.33 / 10.2 | 0.47 / 15.4 | 0.62 / 29.8 | 0.73 / 51.1 | 0.66 / 20.1 | 0.78 / 36.9 | 0.87 / 85.5 | 0.91 / 162.5 |
| proposed central (4,096, 12 ms) | 0.17 / 18.6 | 0.27 / 24.9 | 0.42 / 41.3 | 0.55 / 64.1 | 0.45 / 28.0 | 0.60 / 45.2 | 0.75 / 93.9 | 0.83 / 169.3 |
| proposed pessimistic (16,384, 25 ms) | 0.07 / 39.1 | 0.11 / 50.9 | 0.19 / 78.6 | 0.26 / 110.7 | 0.22 / 47.0 | 0.33 / 67.8 | 0.48 / 122.3 | 0.59 / 198.2 |

As a check, this card (A6000, 768 GB/s, between the A10G and the A100)
measured 53 ms per ordinary 0.49 M-cell step beside a co-tenant. The
current central pair predicts about 25 ms for such a card; the proposed
central pair predicts about 35 ms.

**What the proposal implies:**

1. **Card ranking.** At hindcast leaf sizes the big cards' advantage
   shrinks: an H100 runs a 0.49 M-cell leaf at about 17% utilisation, not
   42%. Packing several leaves per card is therefore worth more than the
   model now says.
2. **Packing without MPS.** `packing_factors` gives no launch gain without
   MPS. But `L` is now mostly host work, and another process's kernels fill
   the card while this process's CPU submits. So part of `L` is hidden even
   without MPS, which HARDWARE.md already says qualitatively. A realisation
   factor on `L` without MPS is the natural next parameter.
3. **Vertical scope of `L`.** `L` is a property of the physics suite and the
   host CPU, not the card. It belongs beside the suite rather than the
   card's row.

`cost.py` is not changed in this PR. The proposal moves the published
utilisation figures in docs/energy-hindcast/cost.md: the H100 at a 0.49
M-cell leaf goes from 0.42 to 0.17 central. That card-ranking decision
belongs to the cost-model owner, ideally after one measurement on a
dedicated datacentre card.

## 7. Interaction with the lagged health check (#80, merged first)

#80 enters `health_ledger.deferring` around each resident STEP op in
`execute_experiment` and reads that ledger with the step's stability record,
one op later. This branch is rebased on it, and the two designs hold
together:

- **#80's ledger takes over.** Inside #80's STEP op, its ledger is active.
  `resident_step` then steps aside (counted as `outer-ledger`) and does not
  drain. #80's lagged read covers the step.
- **Sites #80 could not defer.** MYNN's guards were not ledger-aware. They
  now report into whichever ledger is active, #80's included.
- **Strict mode and other callers.** Under `--health-check strict`, no
  executor ledger exists, so each step drains its own resident ledger at
  its end. The same holds for callers outside `execute_experiment`:
  `runtime.integrate_prepared_case`, offline children, ensembles and tiled
  buffers.
- **The all-water carrier refusal is gone.** #80's lagged ledger
  (`lagged_health._DeferredLedger`) now sets `footprint_reads = True`. So
  the carrier check under it reads the land/ice footprint only on a path
  that is about to refuse. It no longer over-enforces on an all-water
  domain.
  - #80 recorded that over-enforcement as a known behaviour change; it no
    longer applies.
  - The tiled/graph ledger keeps it, because a read inside capture is
    refused outright.
- **`HealthLedger.drain`, merged by hand.**
  - #80's `refuse()` is the one place a drained word becomes an exception.
  - This branch adds the ordered resident ledger and `if_recorded`.
  - It also tags a refusal so it can take precedence over a later error.
  - #80's `_DeferredLedger` does not allocate in its constructor. It sets
    the new attributes itself: unordered, 64 words, every site admitted.

**Retest after the rebase** (467d8042, on the shared A6000):

- **Bit-identity.** Final-state digests are identical across jacob/main
  467d8042, this branch (default) and this branch with
  `WOOF_HEALTH_READBACK=immediate`. Checked on wind-farm and HRRR at
  40x40x30 and 100x100x49, with 4 ordinary steps and 1 radiation step.
- **NaN injection.** The probe runs a wind-farm-physics state through
  `execute_experiment` (dt 1 s, history every 5 s, restart every 10 s) and
  injects a NaN into `u` and `qv` after step 7:
  - Lagged: the run fails with `non-finite state at model step 7 (lagged
    health check: d01 step 7 failed and was detected after d01 step 8 was
    queued; ...)`. The clock stands at 7 s.
  - Strict: the run fails with `non-finite state at model step 7`. The
    clock stands at 6 s.
  - In both modes, and with `WOOF_HEALTH_READBACK=immediate` too, history
    holds 0 s and 5 s and there is no checkpoint. Nothing was written after
    the failure.

## 8. Reproducing

```bash
# one lane, one case (the CUPTI library builds into $WOOF_CUPTI_BUILD)
flock /tmp/woof-gpu-test.lock env CUPY_GPU_MEMORY_LIMIT=6442450944 \
  WOOF_CUPTI_BUILD=/path/to/scratch python tools/profile_launch_overhead.py \
  --suite windfarm --nx 100 --ny 100 --nz 49 --lane cupti --steps 8 \
  --radiation-steps 2 --digest --output out.json
# the switch
WOOF_HEALTH_READBACK=immediate python tools/profile_launch_overhead.py ...
```

The raw JSON (one file per lane, case and side) was kept in scratch while
this note was written. The tables above are its medians, and the per-step
rows sit under `steps` in each file.
